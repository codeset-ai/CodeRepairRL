import os
import json
import re
import subprocess
import shlex
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor


from openai import OpenAI
from trl.extras.vllm_client import AsyncVLLMClient

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)

# Constants for command safety
BAD_CMD = re.compile(r"\b(rm|mv|chmod|chown|truncate|mkfs|>|>>)\b")
TRUNCATE = 8_000  # cap shell output

# Testing mode - when enabled, uses OpenAI API instead of local models
TESTING = os.getenv("TESTING", "0") == "1"

# Endpoint configuration
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
LOCAL_ENDPOINT = os.getenv("LOCAL_ENDPOINT", "http://0.0.0.0:8000/v1")

# Model configuration
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen3-1.7B")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Search/Replace specification from code_mono_repair.py
SYSTEM_PROMPT = """You are a code repair expert tasked with fixing issues in code. You will be provided with:
1. Information about the specific issue (if available)
2. The code segment that needs to be fixed

You have access to two tools:
1. The 'shell' tool: Use this to explore the repository and understand the code
2. The 'submit_patch' tool: Use this ONLY when you're ready to submit your final fix

First, use the 'shell' tool to:
1. Explore the repository structure (`ls`, `find`, etc.)
2. Examine relevant files (`cat`, `head`, `grep`, etc.)
3. Understand the context of the issue
4. Locate the specific files that need modification

Only after thoroughly understanding the codebase and identifying the exact problems, generate a unified git diff and submit it using the 'submit_patch' tool.
"""


def safe_shell(cmd: str, cwd: Path, timeout=4) -> str:
    """Run a shell command safely with timeout and output limits."""
    if BAD_CMD.search(cmd):
        return f"read-only mode: cannot execute `{cmd}`"
    try:
        out = subprocess.check_output(
            shlex.split(cmd), cwd=cwd, timeout=timeout,
            stderr=subprocess.STDOUT, text=True, errors="ignore"
        )
    except subprocess.CalledProcessError as e:
        out = e.output
    except subprocess.TimeoutExpired:
        out = "[command timed out]"
    except Exception as e:
        out = f"[command failed: {e}]"
    return out[:TRUNCATE]


def openai_chat(messages, tools, temperature=0.0, max_tokens=4096):
    """Send a request to the OpenAI API and get a response."""
    # Determine endpoint and model based on TESTING flag
    if TESTING:
        # Use actual OpenAI API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        client = OpenAI(api_key=api_key)
        model = OPENAI_MODEL
        logger.info(f"Using OpenAI API with model: {model}")
    else:
        # Use local vLLM API with OpenAI-compatible interface
        client = OpenAI(api_key="dummy-key", base_url=VLLM_ENDPOINT)
        model = LOCAL_MODEL
        logger.info(f"Using local endpoint at {VLLM_ENDPOINT} with model: {model}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Convert to dict for compatibility with the rest of the code
        return response.choices[0].message.model_dump()
    except Exception as e:
        logger.error(f"Chat API request failed: {str(e)}")
        raise e


class SubmitPatchAgent:
    """Agent that interacts with LLM to analyze code and submit patches."""
    
    SHELL_TOOL = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a read-only shell command inside the repo.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]
            }
        }
    }
    
    SUBMIT_PATCH_TOOL = {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": (
                "Finish the task by emitting SEARCH/REPLACE diff blocks "
                "that implement the required code. This ends the session."
            ),
            "parameters": {
                "type": "object",
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"]
            }
        }
    }

    def __init__(
        self,
        repo: Path,
        task: str,
        max_tool_calls: int = 10,
        temperature: float = 0.0,
    ):
        self.repo = repo
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        self.tools = [self.SHELL_TOOL, self.SUBMIT_PATCH_TOOL]
        
        # Setup trajectory logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("trajectories/minimal_agent/").mkdir(exist_ok=True, parents=True)
        self.trajectory_path = Path(f"trajectories/minimal_agent/trajectory_{timestamp}.jsonl")
        
        # Log initial state
        with open(self.trajectory_path, "w") as f:
            json.dump({
                "repo": str(repo),
                "task": task,
                "timestamp": timestamp,
                "message": {"role": "system", "content": SYSTEM_PROMPT}
            }, f)
            f.write("\n")
            json.dump({
                "message": {"role": "user", "content": task}
            }, f)
            f.write("\n")

    def _chat(self) -> dict:
        """Send a request to the LLM API and get a response."""
        return openai_chat(
            messages=self.history,
            tools=self.tools,
            temperature=self.temperature,
            max_tokens=4096
        )

    def run(self) -> dict:
        """
        Run the agent to analyze code and generate patches.
        
        Returns:
            A dict with keys:
              * 'status': 'ok' | 'error'
              * 'diff': str (only when status='ok')
              * 'reason': str (only when status='error')
        """
        tool_calls = 0
        try:
            while True:
                if tool_calls >= self.max_tool_calls:
                    return {"status": "error", "reason": "max_tool_calls reached"}

                msg = self._chat()
                
                if "tool_calls" not in msg:
                    error_msg = f"Assistant returned plain text without tool calls: {msg}"
                    logger.error(error_msg)
                    return {"status": "error", "reason": error_msg}

                for call in msg["tool_calls"]:
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])
                    cid = call["id"]

                    if name == "shell":
                        logger.info(f"Executing shell command: {args['cmd']}")
                        output = safe_shell(args["cmd"], cwd=self.repo)
                        self._feedback_tool(call, cid, output)
                        tool_calls += 1
                        break  # let the model think again

                    elif name == "submit_patch":
                        diff_text = args.get("diff", "")
                        self._feedback_tool(call, cid, diff_text)
                        logger.info(f"Received patch with {len(diff_text)} characters")
                        
                        # Log the final result to JSONL
                        with open(self.trajectory_path, "a") as f:
                            json.dump({"result": {"status": "ok", "diff": diff_text}}, f)
                            f.write("\n")
                            
                        return {"status": "ok", "diff": diff_text}

                    else:
                        error_msg = f"Unknown tool {name}"
                        logger.error(error_msg)
                        return {"status": "error", "reason": error_msg}

        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            logger.exception(error_msg)
            return {"status": "error", "reason": error_msg}

    def _feedback_tool(self, call, cid, output):
        """Add tool call and result to conversation history and log to JSONL."""
        # Echo assistant tool call
        assistant_msg = {"role": "assistant", "content": None, "tool_calls": [call]}
        self.history.append(assistant_msg)
        
        # Add actual tool result
        tool_msg = {
            "role": "tool",
            "tool_call_id": cid,
            "name": call["function"]["name"],
            "content": output,
        }
        self.history.append(tool_msg)
        
        # Log to JSONL file
        with open(self.trajectory_path, "a") as f:
            json.dump({"message": assistant_msg}, f)
            f.write("\n")
            json.dump({"message": tool_msg}, f)
            f.write("\n")


class MinimalAgent(AsyncVLLMClient):
    """Minimal wrapper for the SubmitPatchAgent using AsyncVLLMClient."""

    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        super().__init__(vllm_url)
        os.environ["OPENAI_API_BASE"] = f"http://{self.host}:{self.server_port}/v1/completions"
        os.environ["OPENAI_API_KEY"] = "dummy"

    def _process_one(self, data: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Process a single prompt and return the diff."""
        assert ("repo" in data and "base_commit" in data and "problem_statement" in data), \
            "Data should contain repo, base_commit and problem_statement"
        
        logger.info(f"[START] Processing {data['repo']} at {data['base_commit']}")
        
        original_dir = os.getcwd()
        
        try:
            logger.info(f"Cloning repo {data['repo']} at commit {data['base_commit']}")
            # Clone the repo into a temporary folder
            temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])
            logger.info(f"Cloned repo to {temp_folder}")
            
            # Change to the repo's root directory
            os.chdir(temp_folder)
            
            logger.info(f"Running agent for {data['repo']} at {data['base_commit']}")
            result = SubmitPatchAgent(
                repo=Path(temp_folder),
                task=data["problem_statement"],
                max_tool_calls=data.get("max_tool_calls", 10),
                temperature=data.get("temperature", 0.0),
            ).run()
            
            if result["status"] == "ok":
                diff = result["diff"]
                messages = []  # We don't capture messages like Aider does
            else:
                diff = ""
                messages = [f"Error: {result['reason']}"]
                
            logger.info(f"Agent finished for {data['repo']} at {data['base_commit']}")
            
        finally:
            logger.info(f"Cleaning up repo dir {temp_folder}")
            clean_repo_dir(temp_folder)
            os.chdir(original_dir)
            logger.info(f"Cleanup done for {data['repo']} at {data['base_commit']}")
            
        logger.info(f"[END] Processing {data['repo']} at {data['base_commit']}")
        return diff, messages

    def generate(self, data: List[Dict[str, Any]], timeout: int = 300, **kwargs) -> List[Dict[str, Any]]:
        """
        Deploys parallel agents to process the given data, returning the updated data with the generated diffs.

        Args:
            data: List of data to process
            timeout: Maximum time in seconds to wait

        Returns:
            The data with an extra "generated_diff" field
        """
        # Process all prompts in parallel
        with ProcessPoolExecutor(max_workers=min(len(data), os.cpu_count() or 4)) as executor:
            # Submit all tasks for parallel execution
            futures = [executor.submit(self._process_one, item) for item in data]
            
            # Collect results
            results = []
            for future in futures:
                try:
                    diff, messages = future.result(timeout=timeout)
                    results.append((diff, messages))
                except Exception as e:
                    logger.exception(f"Error in worker process: {str(e)}")
                    results.append(("", [f"Error: {str(e)}"]))
        
        # Attach the diffs to the corresponding data items
        for item, (diff, messages) in zip(data, results):
            item["generated_diff"] = diff
            item["messages"] = messages
        
        return data


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    from difflib import SequenceMatcher
    # Load example from dataset
    from src.data import get_swe_gym_repo_repair_dataset

    
    logger.info("Loading dataset")
    ds = get_swe_gym_repo_repair_dataset().shuffle(seed=42).select(range(1))
    example = dict(ds[0])
    
    logger.info(f"Testing with repo {example['repo']} at commit {example['base_commit']}")
    
    if TESTING:
        # Testing with OpenAI API
        logger.info("Running in TESTING mode with OpenAI API")
        
        # Check for API key before proceeding
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable is not set. Please set it when using TESTING mode.")
            exit(1)
    else:
        logger.info("Running with local vLLM endpoint")
    
    # Create a temporary directory for testing
    temp_folder = clone_repo_at_commit(handle_to_url(example["repo"]), example["base_commit"])
    
    try:
        # Run SubmitPatchAgent (using either OpenAI or local vLLM based on TESTING flag)
        agent = SubmitPatchAgent(
            repo=Path(temp_folder),
            task=example["problem_statement"],
            max_tool_calls=50,
            temperature=0.0,
        )
        result = agent.run()
        
        print(f"\nRepo: {example['repo']} at {example['base_commit']}")
        print(f"Problem: {example['problem_statement']}")
        print(f"Oracle diff: {example['patch']}")
        print(f"Generated diff: {result.get('diff', '')}")

        print(f"Diff similarity: {SequenceMatcher(None, example['patch'], result.get('diff', '')).ratio()}")
        
        if result["status"] != "ok":
            logger.error(f"Agent failed: {result.get('reason', 'Unknown error')}")
        
    finally:
        # Clean up
        clean_repo_dir(temp_folder)
    
    logger.info("Test completed")


# vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768
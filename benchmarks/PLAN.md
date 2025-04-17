Below is a **fresh, trimmed‑down template** that folds every point you listed into the design – nothing more, nothing less.

---

## 1 Repo layout (clean slate)

```
llm‑bench/
├── benchmarks/                # one folder per task, each exports run_benchmark()
│   ├── aider_polyglot/driver.py
│   ├── gitbug_java/driver.py
│   ├── swe_bench_lite/driver.py
│   ├── swe_bench_verified/driver.py
│   └── lighteval/driver.py
├── job_runner.py              # executes *inside* each Slurm job
├── launcher.py                # submitit orchestrator (run this)
├── singularity.def            # container recipe
└── README.md
```

---

## 2 Container: `singularity.def`

```def
Bootstrap: docker
From: nvidia/cuda:12.4.0-runtime-ubuntu22.04

%post
    apt-get update && apt-get install -y python3 python3-pip git
    pip3 install --upgrade pip
    pip3 install \
        torch==2.2.2+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 \
        vllm==0.8.3 \         # OpenAI‑v1 compatible, faster in‑tree KV cache citeturn1search2
        aider lighteval swebench submitit litellm datasets accelerate

%environment
    export PYTHONUNBUFFERED=1
    # Caches live on the large project partition you gave
    export PROJECT_DIR=/proj/berzelius-2024-336/users/x_bjabj
    export HF_HOME=$PROJECT_DIR/.hf
    export TRANSFORMERS_CACHE=$PROJECT_DIR/.cache/huggingface/transformers
    export HF_DATASETS_CACHE=$PROJECT_DIR/.cache/huggingface/datasets
    export UV_CACHE_DIR=$PROJECT_DIR/.cache/.uv
```

Build once:

```bash
apptainer build bench.sif singularity.def
```

---

## 3 `job_runner.py` – executes per‑benchmark inside the job

```python
# job_runner.py
import argparse, importlib, json, pathlib, subprocess, time, requests, os, uuid

BENCHES = {                    # key → module:function
    "aider_diff":          "benchmarks.aider_diff.driver:run_benchmark",
    "aider_polyglot":      "benchmarks.aider_polyglot.driver:run_benchmark",
    "swe_bench_verified":  "benchmarks.swe_bench_verified.driver:run_benchmark",
    "lighteval":           "benchmarks.lighteval.driver:run_benchmark",
}

def start_vllm(model_path: str, port: int) -> subprocess.Popen:
    """Launch a vLLM 0.8.3 OpenAI‑compatible server (BF16, single‑GPU)."""
    api_key = uuid.uuid4().hex                           # unique but throwaway
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",   # 0.8.3 CLI citeturn1search2
        "--model", model_path,
        "--dtype", "bfloat16",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--api-key", api_key,
        "--gpu-memory-utilization", "0.90",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # poll /health (fast path: usually 45‑60 s even for 34 B on A100)
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(90):            # 90×2 s = 3 min safety net
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return proc, api_key
        except requests.RequestException:
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError("vLLM did not become healthy in 3 min")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_key")
    ap.add_argument("model_path")
    ap.add_argument("tmp_root")
    ap.add_argument("port", type=int)
    args = ap.parse_args()

    vllm_proc, api_key = start_vllm(args.model_path, args.port)

    try:
        module, fn = BENCHES[args.bench_key].split(":")
        run_fn = getattr(importlib.import_module(module), fn)
        metrics = run_fn(f"http://127.0.0.1:{args.port}/v1", api_key,
                         f"{args.tmp_root}/{args.bench_key}")
        pathlib.Path(args.tmp_root, f"{args.bench_key}.json").write_text(
            json.dumps(metrics)
        )
        print(json.dumps({args.bench_key: metrics}, indent=2))
    finally:
        vllm_proc.terminate()
        vllm_proc.wait()

if __name__ == "__main__":
    main()
```

**Key changes from the old draft**

* No heredoc hacks – `job_runner.py` is imported normally.
* One vLLM server per benchmark; port = `8000 + index`.
* Poll `/health` every 2 s, bail after 3 min (your observed 60 s is inside that).

---

## 4 `launcher.py` – one command to fan everything out

```python
#!/usr/bin/env python
import argparse, pathlib, submitit

BENCH_KEYS = [
    "aider_diff",
    "aider_polyglot",
    "swe_bench_verified",
    "lighteval",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="HF model name or local path")
    ap.add_argument("--image", default="bench.sif")
    ap.add_argument("--partition", default="gpuA100")
    ap.add_argument("--gpus", type=int, default=1,
                    help="1 or 2 A100 per job; adjust batch_size downstream")
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--tmproot", default="results")
    args = ap.parse_args()

    pathlib.Path(args.tmproot).mkdir(exist_ok=True)

    ex = submitit.AutoExecutor(folder="logs/%j")
    ex.update_parameters(
        gpus_per_node=args.gpus,
        cpus_per_task=10,            # vLLM + benchmark harness threads
        mem_gb=80,                   # generous, same for all
        timeout_min=sum(int(x) * 60 ** i
                        for i, x in enumerate(reversed(args.time.split(":")))) // 60,
        slurm_partition=args.partition,
        # everything is GPU now – simple
    )

    jobs = {}
    for i, key in enumerate(BENCH_KEYS):
        port = 8000 + i
        jobs[key] = ex.submit(
            "apptainer", "exec", "--nv", "-B", f"{pathlib.Path.cwd()}:/workspace",
            args.image,
            "python", "-m", "job_runner",
            key, args.ckpt, args.tmproot, str(port)
        )
        print(f"{key:<22} → {jobs[key].job_id}  (port {port})")

if __name__ == "__main__":
    main()
```

### Why this is simpler

* **Exactly one** SBATCH template (all GPU): no per‑benchmark overrides.
* Ports are deterministic (`8000+i`), useful if you ever want to poke a running job.
* Everything shares the same HF caches on the large project FS.

---

## 5 Benchmark driver contract (unchanged)

Each `driver.py` keeps the 20‑line shape:

```python
def run_benchmark(openai_url: str, openai_key: str, tmpdir: str) -> dict:
    ...
```

Inside you call the upstream CLI (`aider`, `lighteval`, `swebench`) with
`OPENAI_API_KEY` + `OPENAI_BASE_URL` set to hit the local vLLM server.

---

## 6 Remaining edge‑cases (now mostly gone)

| Potential snag | Why it’s covered |
|----------------|------------------|
| **GPU over‑committed** | All jobs reserve 1 or 2 A100s via Slurm‑GPU binding. |
| **Port clash** | `8000 + i` is unique *within the node*; one job per node ⇒ no conflicts. |
| **vLLM load time** | `/health` polling with a 3‑min cap keeps jobs from hanging indefinitely. |
| **Cache permissions** | Cached path lives in your project directory (user‑writable). |
| **Future vLLM CLI churn** | We call the *documented* 0.8.3 entrypoint (`python -m vllm.entrypoints.openai.api_server`). |

---

### TL;DR

*Run `python launcher.py --ckpt <model> --image bench.sif` and walk away.*  
Each benchmark launches its own vLLM 0.8.3 server on BF16 A100s, logs land in `results/`, and you didn’t have to touch sbatch files at all.
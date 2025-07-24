from datasets import Dataset

def get_codeset_dataset() -> Dataset:
    """
    Get the codeset dataset.
    """
    dataset = [
        {
            "dataset": "gitbug-java",
            "sample-id": "assertj-assertj-vavr-f4d7f276e87c",
            "task": "Some of the tests in this project are failing. Your task is to fix the source code such that all tests pass.",
            "prompt": "Dummy" # Add a dummy "prompt" key for compatibility with trl
        }
    ]
    return Dataset.from_list(dataset)


def codeset_reward_func(success, **kwargs) -> list[float]:
    """Reward function that checks if the extracter answer matches the ground truth answer."""
    return [1.0 if s else 0.0 for s in success]
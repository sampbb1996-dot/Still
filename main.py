def anchor_update(score: float, active: bool) -> float:
    """
    Zero is unstable.
    Doing nothing has cost.
    """

    # decay
    score *= (1 - DECAY)

    # cost of inactivity
    if not active:
        score -= INACTION_PENALTY

    return max(-1.0, min(1.0, score))

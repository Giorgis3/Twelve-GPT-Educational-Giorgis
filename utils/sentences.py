"""
Text helpers for converting raw metric values into readable sentences.

This module is intentionally lightweight and is used by description builders to:
1) map numeric z-scores to qualitative labels (for example, "excellent"), and
2) normalize metric names into human-readable phrases.
"""


def pronouns(gender):
    """
    Return subject, object, and possessive pronouns from a gender label.

    Parameters:
        gender: Any value that can be lowercased (typically "male" or "female").

    Returns:
        tuple[str, str, str]: Pronouns in the order
            (subject, object, possessive), e.g. ("he", "him", "his").

    Notes:
        Historical behavior is preserved: any value other than "male" maps to
        ("she", "her", "her").
    """
    if gender.lower() == "male":
        subject_p, object_p, possessive_p = "he", "him", "his"
    else:
        subject_p, object_p, possessive_p = "she", "her", "her"

    return subject_p, object_p, possessive_p


def describe_level(value, thresholds=[1.5, 1, 0.5, -0.5, -1], words=["outstanding", "excellent", "good", "average", "below average", "poor"]):
    """
    Describe a z-score-like value using default football-scouting buckets.

    Parameters:
        value: Numeric score to classify (typically a z-score).
        thresholds: Descending lower bounds for each bucket transition.
        words: Ordered labels with exactly one more item than thresholds.

    Returns:
        str: Qualitative label corresponding to ``value``.

    Example:
        With defaults, 1.2 -> "excellent", 0.2 -> "average", -1.3 -> "poor".
    """
    return describe(thresholds, words, value)


def describe(thresholds, words, value):
    """
    Map a numeric value to a label using descending thresholds.

    Parameters:
        thresholds: Sequence of lower bounds in descending order.
            Example: [1.5, 1.0, 0.5, -0.5, -1.0]
        words: Labels for each interval. Must satisfy:
            ``len(words) == len(thresholds) + 1``.
        value: Numeric value to map.

    Returns:
        str: Label selected from ``words``.

    How the intervals are interpreted:
        For thresholds ``[t0, t1, t2]`` and words ``[w0, w1, w2, w3]``:
        - value >= t0 -> w0
        - t1 <= value < t0 -> w1
        - t2 <= value < t1 -> w2
        - value < t2 -> w3
    """
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    i = 0

    # Move down buckets until we find the first threshold the value meets.
    while i < len(thresholds) and value < thresholds[i]:
        i += 1

    return words[i]


def format_metric(metric):
    """
    Format a metric key into a short display label.

    Intended for compact UI or sentence fragments where we want readable text
    without extra qualifiers such as "adjusted per90".

    Parameters:
        metric: Raw metric key (for example, ``"npxG_adjusted_per90"``).

    Returns:
        str: Cleaned, capitalized metric label.
    """
    return (
        metric.replace("_", " ")
        .replace(" adjusted per90", "")
        .replace("npxG", "non-penalty expected goals")
        .capitalize()
    )


def write_out_metric(metric):
    """
    Expand a metric key into a more explicit narrative phrase.

    This version is designed for full sentences, so it keeps contextual wording
    (for example "adjusted for possession" and "per 90") and appends "minutes".

    Parameters:
        metric: Raw metric key (for example, ``"passes_adjusted_per90"``).

    Returns:
        str: Verbose metric phrase ready to embed in generated prose.
    """
    return (
        metric.replace("_", " ")
        .replace("adjusted", "adjusted for possession")
        .replace("per90", "per 90")
        .replace("npxG", "non-penalty expected goals")
        + " minutes"
    )

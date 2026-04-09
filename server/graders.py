"""
GST Sahayak — Task Graders
Each grader takes environment state and returns a float score 0.0–1.0.
All graders are deterministic — no randomness in scoring.
"""

from typing import Dict, List

# Validator requires scores strictly in the open interval (0, 1) — not 0.0, not 1.0
_SCORE_MIN = 1e-4
_SCORE_MAX = 1.0 - 1e-4


def _clamp(score: float) -> float:
    """Clamp a score to the open interval (0, 1) as required by the hackathon validator."""
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


# ---------------------------------------------------------------------------
# Task 1 — Invoice Classifier Grader
# ---------------------------------------------------------------------------

def grade_invoice_classifier(
    ground_truth_invoices: List[dict],
    classified_so_far: Dict[str, dict],
) -> float:
    """
    Score = (correct_types / total) + (correct_hsn / total) * 0.5
    Capped at 1.0.
    """
    total = len(ground_truth_invoices)
    if total == 0:
        return _SCORE_MIN

    correct_types = 0
    correct_hsn = 0

    for inv in ground_truth_invoices:
        inv_id = inv["invoice_id"]
        agent_decision = classified_so_far.get(inv_id, {})

        gt_type = inv["_ground_truth_type"]
        gt_hsn = inv["_ground_truth_hsn"]

        agent_type = agent_decision.get("invoice_type")
        agent_hsn = agent_decision.get("hsn_code")

        if agent_type == gt_type:
            correct_types += 1
        if agent_hsn == gt_hsn:
            correct_hsn += 1

    score = (correct_types / total) + (correct_hsn / total) * 0.5
    return _clamp(round(score, 4))


# ---------------------------------------------------------------------------
# Task 2 — ITC Reconciliation Grader (F1 score)
# ---------------------------------------------------------------------------

def grade_itc_reconciliation(
    ground_truth_itc: List[dict],
    itc_decisions: Dict[str, str],
) -> float:
    """
    Treat "accept" as the positive class for invoices that should be accepted.
    F1 = 2 * precision * recall / (precision + recall)
    correct_decision is "accept", "reject", or "flag" (value_mismatch invoices).
    """
    tp = fp = fn = 0

    for item in ground_truth_itc:
        inv_id = item["invoice_id"]
        correct = item["correct_decision"]   # "accept", "reject", or "flag"
        agent_decision = itc_decisions.get(inv_id)  # "accepted", "rejected", "flagged", None

        if agent_decision is None:
            if correct == "accept":
                fn += 1
            continue

        # BUG-9 follow-through: "flag" invoices — accepting them is a false positive;
        # rejecting or flagging them is treated as a true negative (cautious, safe)
        if correct == "flag":
            if agent_decision == "accepted":
                fp += 1
            # rejected or flagged = not fp/fn for F1 purposes
            continue

        agent_accept = agent_decision == "accepted"
        correct_accept = correct == "accept"

        if correct_accept and agent_accept:
            tp += 1
        elif not correct_accept and agent_accept:
            fp += 1
        elif correct_accept and not agent_accept:
            fn += 1
        # True negative (correctly rejected/flagged) doesn't factor into F1 numerator

    # BUG-12 fix: when there are no positive cases (tp+fn==0), the correct answer
    # is to accept nothing — return max score if agent accepted nothing (fp==0)
    if tp + fn == 0:
        return _SCORE_MAX if fp == 0 else _SCORE_MIN

    if tp + fp == 0:
        return _SCORE_MIN

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return _SCORE_MIN

    f1 = 2 * precision * recall / (precision + recall)
    return _clamp(round(f1, 4))


# ---------------------------------------------------------------------------
# Task 3 — Full GSTR-3B Filing Grader
# ---------------------------------------------------------------------------

def _field_accuracy(agent_val: float, truth_val: float) -> float:
    """Single field accuracy: 1 - |agent - truth| / truth, clamped to [0, 1]."""
    if truth_val == 0.0:
        return 1.0 if agent_val == 0.0 else 0.0
    return max(0.0, 1.0 - abs(agent_val - truth_val) / truth_val)


def grade_gstr3b_filing(
    agent_payload: dict,
    ground_truth_payload: dict,
    audit_flags: int,
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    final_score = (field_score * 0.6) + (penalty_score * 0.3) + (time_score * 0.1)
    """
    if agent_payload is None:
        return _SCORE_MIN

    # --- Field accuracy ---
    field_scores = []

    def compare_nested(agent_dict: dict, truth_dict: dict):
        for key in truth_dict:
            truth_val = truth_dict.get(key, 0.0)
            agent_val = agent_dict.get(key, 0.0) if agent_dict else 0.0
            if isinstance(truth_val, dict):
                compare_nested(agent_val, truth_val)
            else:
                field_scores.append(_field_accuracy(float(agent_val), float(truth_val)))

    compare_nested(agent_payload, ground_truth_payload)
    field_score = sum(field_scores) / len(field_scores) if field_scores else 0.0

    # --- Penalty score ---
    penalty_score = max(0.0, 1.0 - audit_flags * 0.1)

    # --- Time score ---
    # BUG-13 fix: avoid hardcoded 15.0; derive from max_steps so it stays correct
    # if max_steps ever changes (threshold is always 75% of max_steps)
    threshold = int(max_steps * 0.75)
    window = max(max_steps - threshold, 1)
    if steps_taken < threshold:
        time_score = 1.0
    elif steps_taken <= max_steps:
        time_score = (max_steps - steps_taken) / window
    else:
        time_score = 0.0

    final = (field_score * 0.6) + (penalty_score * 0.3) + (time_score * 0.1)
    return _clamp(round(final, 4))


# ---------------------------------------------------------------------------
# Reward helpers used by the environment per step
# ---------------------------------------------------------------------------

def reward_classify_invoice(correct_type: bool, correct_hsn: bool) -> float:
    if correct_type and correct_hsn:
        return 0.15
    elif correct_type:
        return 0.05
    else:
        return -0.05


def reward_itc_decision(action_type: str, correct_decision: str) -> float:
    """
    action_type: "accept_itc" | "reject_itc" | "flag_for_review"
    correct_decision: "accept" | "reject" | "flag" (value_mismatch invoices)
    """
    # BUG-9 follow-through: align reward with the prompt rule and grader
    # "flag" means value_mismatch — prompt says flag_for_review is correct
    if correct_decision == "flag":
        if action_type == "flag_for_review":
            return 0.20   # correct decision
        elif action_type == "reject_itc":
            return 0.05   # cautious but not the intended action
        else:             # accept_itc on a disputed invoice — audit risk
            return -0.30

    if action_type == "flag_for_review":
        return 0.05  # safe partial credit for non-flag cases

    agent_accept = action_type == "accept_itc"
    should_accept = correct_decision == "accept"

    if agent_accept and should_accept:
        return 0.20
    elif not agent_accept and not should_accept:
        return 0.20
    elif agent_accept and not should_accept:
        return -0.30  # audit flag — high penalty
    else:
        return -0.15  # wrongly rejected valid ITC


def reward_compute_liability(within_tolerance: bool) -> float:
    return 0.30 if within_tolerance else -0.10


def reward_file_return(field_accuracy: float) -> float:
    if field_accuracy >= 0.95:
        return 1.00
    else:
        return round(field_accuracy * 0.5, 2)
"""
GST Sahayak — Task Graders
Each grader takes environment state and returns a float score strictly in (0, 1).
All graders are deterministic — no randomness in scoring.
"""

from typing import Dict, List

# Validator requires scores strictly in the open interval (0, 1) — never 0.0 or 1.0
_SCORE_MIN = 1e-4          # smallest allowed score
_SCORE_MAX = 1.0 - 1e-4   # largest allowed score


def _clamp(score: float) -> float:
    """Clamp any score/reward to the open interval (_SCORE_MIN, _SCORE_MAX)."""
    return max(_SCORE_MIN, min(_SCORE_MAX, float(score)))


# ---------------------------------------------------------------------------
# Task 1 — Invoice Classifier Grader
# ---------------------------------------------------------------------------

def grade_invoice_classifier(
    ground_truth_invoices: List[dict],
    classified_so_far: Dict[str, dict],
) -> float:
    """
    Score = (correct_types / total) + (correct_hsn / total) * 0.5
    Strictly within (0, 1).
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
        gt_hsn  = inv["_ground_truth_hsn"]

        if agent_decision.get("invoice_type") == gt_type:
            correct_types += 1
        if agent_decision.get("hsn_code") == gt_hsn:
            correct_hsn += 1

    score = (correct_types / total) + (correct_hsn / total) * 0.5
    return _clamp(round(score, 6))


# ---------------------------------------------------------------------------
# Task 2 — ITC Reconciliation Grader (F1 score)
# ---------------------------------------------------------------------------

def grade_itc_reconciliation(
    ground_truth_itc: List[dict],
    itc_decisions: Dict[str, str],
) -> float:
    """
    Treat "accept" as the positive class.
    F1 = 2 * precision * recall / (precision + recall)
    correct_decision: "accept" | "reject" | "flag" (value_mismatch)
    Strictly within (0, 1).
    """
    tp = fp = fn = 0

    for item in ground_truth_itc:
        inv_id = item["invoice_id"]
        correct = item["correct_decision"]          # "accept" | "reject" | "flag"
        agent   = itc_decisions.get(inv_id)         # "accepted"|"rejected"|"flagged"|None

        if agent is None:
            if correct == "accept":
                fn += 1
            continue

        # value_mismatch: accepting = fp; rejecting/flagging = true negative
        if correct == "flag":
            if agent == "accepted":
                fp += 1
            continue

        agent_accept   = agent   == "accepted"
        correct_accept = correct == "accept"

        if correct_accept and agent_accept:
            tp += 1
        elif not correct_accept and agent_accept:
            fp += 1
        elif correct_accept and not agent_accept:
            fn += 1

    # No positive cases — agent correctly accepted nothing
    if tp + fn == 0:
        return _SCORE_MAX if fp == 0 else _SCORE_MIN

    if tp + fp == 0:
        return _SCORE_MIN

    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)

    if precision + recall == 0:
        return _SCORE_MIN

    f1 = 2 * precision * recall / (precision + recall)
    return _clamp(round(f1, 6))


# ---------------------------------------------------------------------------
# Task 3 — Full GSTR-3B Filing Grader
# ---------------------------------------------------------------------------

def _field_accuracy(agent_val: float, truth_val: float) -> float:
    """Per-field accuracy clamped to (_SCORE_MIN, _SCORE_MAX)."""
    if truth_val == 0.0:
        raw = _SCORE_MAX if agent_val == 0.0 else _SCORE_MIN
    else:
        raw = max(0.0, 1.0 - abs(agent_val - truth_val) / truth_val)
    return _clamp(raw)


def grade_gstr3b_filing(
    agent_payload: dict,
    ground_truth_payload: dict,
    audit_flags: int,
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    final_score = (field_score * 0.6) + (penalty_score * 0.3) + (time_score * 0.1)
    Strictly within (0, 1).
    """
    if agent_payload is None:
        return _SCORE_MIN

    # --- Field accuracy ---
    field_scores = []

    def compare_nested(agent_dict: dict, truth_dict: dict):
        for key in truth_dict:
            truth_val = truth_dict.get(key, 0.0)
            agent_val = (agent_dict.get(key, 0.0) if agent_dict else 0.0)
            if isinstance(truth_val, dict):
                compare_nested(agent_val if isinstance(agent_val, dict) else {}, truth_val)
            else:
                field_scores.append(_field_accuracy(float(agent_val), float(truth_val)))

    compare_nested(agent_payload, ground_truth_payload)
    field_score = _clamp(sum(field_scores) / len(field_scores)) if field_scores else _SCORE_MIN

    # --- Penalty score (clamped) ---
    penalty_score = _clamp(max(0.0, 1.0 - audit_flags * 0.1))

    # --- Time score (clamped) ---
    threshold = int(max_steps * 0.75)
    window    = max(max_steps - threshold, 1)
    if steps_taken < threshold:
        time_score = _SCORE_MAX
    elif steps_taken <= max_steps:
        time_score = _clamp((max_steps - steps_taken) / window)
    else:
        time_score = _SCORE_MIN

    final = (field_score * 0.6) + (penalty_score * 0.3) + (time_score * 0.1)
    return _clamp(round(final, 6))


# ---------------------------------------------------------------------------
# Reward helpers used by the environment per step
# ---------------------------------------------------------------------------

def reward_classify_invoice(correct_type: bool, correct_hsn: bool) -> float:
    if correct_type and correct_hsn:
        return 0.15
    elif correct_type:
        return 0.05
    return -0.05


def reward_itc_decision(action_type: str, correct_decision: str) -> float:
    """
    action_type:      "accept_itc" | "reject_itc" | "flag_for_review"
    correct_decision: "accept"     | "reject"      | "flag"
    """
    if correct_decision == "flag":
        if action_type == "flag_for_review":
            return 0.20
        elif action_type == "reject_itc":
            return 0.05
        return -0.30

    if action_type == "flag_for_review":
        return 0.05

    agent_accept  = action_type == "accept_itc"
    should_accept = correct_decision == "accept"

    if agent_accept and should_accept:
        return 0.20
    elif not agent_accept and not should_accept:
        return 0.20
    elif agent_accept and not should_accept:
        return -0.30
    return -0.15


def reward_compute_liability(within_tolerance: bool) -> float:
    return 0.30 if within_tolerance else -0.10


def reward_file_return(field_accuracy: float) -> float:
    """Per-step filing reward — clamped so obs.reward is never exactly 0 or 1."""
    if field_accuracy >= 0.95:
        return _SCORE_MAX
    return _clamp(round(field_accuracy * 0.5, 4))

from typing import Dict, List

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clamp(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, float(score)))


# ---------------- TASK 1 ----------------
def grade_invoice_classifier(
    ground_truth_invoices: List[dict],
    classified_so_far: Dict[str, dict],
) -> float:
    total = len(ground_truth_invoices)
    if total == 0:
        return _SCORE_MIN

    correct_types = 0
    correct_hsn = 0

    for inv in ground_truth_invoices:
        inv_id = inv["invoice_id"]
        agent = classified_so_far.get(inv_id, {})

        if agent.get("invoice_type") == inv["_ground_truth_type"]:
            correct_types += 1
        if agent.get("hsn_code") == inv["_ground_truth_hsn"]:
            correct_hsn += 1

    type_acc = correct_types / total
    hsn_acc = correct_hsn / total

    score = (type_acc * 2.0 + hsn_acc) / 3.0
    return _clamp(round(score, 6))


# ---------------- TASK 2 ----------------
def grade_itc_reconciliation(
    ground_truth_itc: List[dict],
    itc_decisions: Dict[str, str],
) -> float:

    tp = fp = fn = 0

    for item in ground_truth_itc:
        inv_id = item["invoice_id"]
        correct = item["correct_decision"]
        agent = itc_decisions.get(inv_id)

        if agent is None:
            if correct == "accept":
                fn += 1
            continue

        if correct == "flag":
            if agent == "accepted":
                fp += 1
            continue

        agent_accept = agent == "accepted"
        correct_accept = correct == "accept"

        if correct_accept and agent_accept:
            tp += 1
        elif not correct_accept and agent_accept:
            fp += 1
        elif correct_accept and not agent_accept:
            fn += 1

    # Edge cases
    if tp + fn == 0:
        score = _SCORE_MAX if fp == 0 else _SCORE_MIN
        return _clamp(round(score, 6))   # 🔥 FIX

    if tp + fp == 0:
        return _clamp(round(_SCORE_MIN, 6))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return _clamp(round(_SCORE_MIN, 6))

    f1 = 2 * precision * recall / (precision + recall)

    return _clamp(round(f1, 6))   # 🔥 FINAL FIX


# ---------------- TASK 3 ----------------
def _field_accuracy(agent_val: float, truth_val: float) -> float:
    if truth_val == 0:
        raw = _SCORE_MAX if agent_val == 0 else _SCORE_MIN
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

    if agent_payload is None:
        return _SCORE_MIN

    field_scores = []

    def compare(agent, truth):
        for k in truth:
            tv = truth[k]
            av = agent.get(k, 0) if agent else 0
            if isinstance(tv, dict):
                compare(av if isinstance(av, dict) else {}, tv)
            else:
                field_scores.append(_field_accuracy(float(av), float(tv)))

    compare(agent_payload, ground_truth_payload)

    field_score = _clamp(sum(field_scores) / len(field_scores)) if field_scores else _SCORE_MIN
    penalty_score = _clamp(1.0 - audit_flags * 0.1)

    threshold = int(max_steps * 0.75)

    if steps_taken < threshold:
        time_score = _SCORE_MAX
    else:
        time_score = _clamp((max_steps - steps_taken) / max(1, max_steps - threshold))

    final = (field_score * 0.6) + (penalty_score * 0.3) + (time_score * 0.1)
    return _clamp(round(final, 6))


def reward_file_return(field_accuracy: float) -> float:
    return round(_clamp(field_accuracy * 0.5), 4)
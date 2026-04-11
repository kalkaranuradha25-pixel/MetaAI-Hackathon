"""
GST Sahayak — GSTEnvironment
Implements the OpenEnv Environment interface with reset(), step(), and state property.
"""

import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment

from models import GSTAction, GSTObservation, GSTState
from .data_generator import (
    generate_episode,
    invoice_for_agent,
    compute_ground_truth_gstr3b,
)
from .graders import (
    reward_file_return,
    grade_invoice_classifier,
    grade_itc_reconciliation,
    grade_gstr3b_filing,
    _field_accuracy,  # BUG-14 fix: single canonical implementation
    _SCORE_MIN,
    _SCORE_MAX,
    _clamp,
)

# Step-level reward helpers (kept here, not in graders.py, so graders.py
# only exports functions that return values strictly within (0, 1)).

def reward_classify_invoice(correct_type: bool, correct_hsn: bool) -> float:
    if correct_type and correct_hsn:
        return 0.15
    elif correct_type:
        return 0.05
    return -0.05


def reward_itc_decision(action_type: str, correct_decision: str) -> float:
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

VALID_ACTION_TYPES = {
    "classify_invoice",
    "accept_itc",
    "reject_itc",
    "flag_for_review",
    "compute_liability",
    "file_return",
}

# Fixed seeds per task for reproducible baseline scores
TASK_SEEDS = {
    "invoice_classifier": 42,
    "itc_reconciliation": 137,
    "full_gstr3b_filing": 999,
}


class GSTEnvironment(Environment[GSTAction, GSTObservation, GSTState]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()  # BUG-5 fix: initialise base Environment (sets transform, rubric)
        self._episode_data: Optional[dict] = None
        self._classified_so_far: dict = {}
        self._itc_decisions: dict = {}
        self._cumulative_reward: float = 0.0
        self._current_step: int = 0
        self._task: str = "invoice_classifier"
        self._audit_flags: int = 0
        self._liability_computed: bool = False
        self._episode_id: str = str(uuid.uuid4())
        self._last_error: Optional[str] = None
        self._done: bool = False
        self._final_payload: Optional[dict] = None
        self._accepted_itc_ids: list = []
        self._natural_end: bool = False  # True when task ends by processing all invoices

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed: int = None, episode_id: str = None,
              task: str = "invoice_classifier", **kwargs) -> GSTObservation:
        self._task = task
        seed = TASK_SEEDS.get(task, 42)
        self._episode_data = generate_episode(seed, task)

        self._classified_so_far = {}
        self._itc_decisions = {}
        self._cumulative_reward = 0.0
        self._current_step = 0
        self._audit_flags = 0
        self._liability_computed = False
        self._episode_id = str(uuid.uuid4())
        self._last_error = None
        self._done = False
        self._final_payload = None
        self._accepted_itc_ids = []
        self._natural_end = False

        return self._build_observation(reward=_SCORE_MIN, done=False)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action) -> GSTObservation:
        # BUG-7 fix: coerce raw dict to GSTAction (handles HTTPEnvServer JSON path)
        if isinstance(action, dict):
            action = GSTAction(**action)

        if self._done:
            return self._build_observation(reward=_SCORE_MIN, done=True, error="Episode already finished")

        self._current_step += 1
        self._last_error = None

        max_steps = self._episode_data["max_steps"]
        invoices = self._episode_data["invoices"]
        gstr_2b = self._episode_data["gstr_2b"]
        mismatch_ids = self._episode_data["mismatch_ids"]
        ground_truth_invoices = self._episode_data["ground_truth_invoices"]
        ground_truth_itc = self._episode_data["ground_truth_itc"]

        # --- Validate action type ---
        if action.action_type not in VALID_ACTION_TYPES:
            self._cumulative_reward += _clamp(-0.10)
            done = self._current_step >= max_steps
            self._done = done
            return self._build_observation(
                reward=-0.10, done=done,
                error=f"Invalid action_type: {action.action_type}"
            )

        step_reward = 0.0

        # --- classify_invoice ---
        if action.action_type == "classify_invoice":
            inv_id = action.invoice_id
            if inv_id is None:
                step_reward = -0.10
                self._last_error = "classify_invoice requires invoice_id"
            elif inv_id in self._classified_so_far:
                # Repeated action penalty
                step_reward = -0.10
                self._last_error = f"Already classified {inv_id}"
            else:
                gt = next((i for i in ground_truth_invoices if i["invoice_id"] == inv_id), None)
                if gt is None:
                    step_reward = -0.10
                    self._last_error = f"Unknown invoice_id: {inv_id}"
                else:
                    correct_type = action.invoice_type == gt["_ground_truth_type"]
                    correct_hsn = action.hsn_code == gt["_ground_truth_hsn"]
                    step_reward = reward_classify_invoice(correct_type, correct_hsn)
                    self._classified_so_far[inv_id] = {
                        "invoice_type": action.invoice_type,
                        "hsn_code": action.hsn_code,
                    }

        # --- accept_itc / reject_itc / flag_for_review ---
        elif action.action_type in ("accept_itc", "reject_itc", "flag_for_review"):
            inv_id = action.invoice_id
            if inv_id is None:
                step_reward = -0.10
                self._last_error = f"{action.action_type} requires invoice_id"
            elif inv_id in self._itc_decisions:
                step_reward = -0.10
                self._last_error = f"Already decided ITC for {inv_id}"
            else:
                gt_itc = next((i for i in ground_truth_itc if i["invoice_id"] == inv_id), None)
                if gt_itc is None:
                    step_reward = -0.10
                    self._last_error = f"Unknown invoice_id: {inv_id}"
                else:
                    step_reward = reward_itc_decision(action.action_type, gt_itc["correct_decision"])
                    decision_map = {
                        "accept_itc": "accepted",
                        "reject_itc": "rejected",
                        "flag_for_review": "flagged",
                    }
                    self._itc_decisions[inv_id] = decision_map[action.action_type]

                    # Track audit flags (wrongly accepted mismatched ITC)
                    if action.action_type == "accept_itc" and inv_id in mismatch_ids:
                        self._audit_flags += 1
                    # Track accepted ITC for GSTR-3B computation
                    if action.action_type == "accept_itc":
                        self._accepted_itc_ids.append(inv_id)

        # --- compute_liability ---
        elif action.action_type == "compute_liability":
            if self._liability_computed:
                step_reward = -0.10
                self._last_error = "compute_liability already called"
            else:
                # Check if classification + reconciliation is reasonably complete
                all_ids = [i["invoice_id"] for i in invoices]
                classified = len(self._classified_so_far)
                reconciled = len(self._itc_decisions)
                total = len(all_ids)

                # BUG-8 fix: guard against empty invoice list
                if total == 0:
                    step_reward = -0.10
                    self._last_error = "No invoices in episode"
                # Tolerance: at least 70% done before computing
                elif classified / total < 0.7 or (self._task != "invoice_classifier" and reconciled / total < 0.7):
                    step_reward = -0.10
                    self._last_error = "Too many invoices unprocessed for liability computation"
                else:
                    gt_payload = compute_ground_truth_gstr3b(invoices, self._accepted_itc_ids)
                    gt_net = sum(gt_payload["net_tax_liability"].values())

                    # BUG-4 fix: use ground truth outward (not agent's possibly-wrong
                    # classifications) so misclassifications don't poison the liability check.
                    # The agent's only controllable variable here is which ITCs they accepted.
                    gt_outward = sum(gt_payload["outward_taxable_supplies"].values())
                    agent_itc = sum(
                        inv.get("igst", 0) + inv.get("cgst", 0) + inv.get("sgst", 0)
                        for inv in invoices
                        if inv["invoice_id"] in self._accepted_itc_ids
                    )
                    agent_net = max(0.0, gt_outward - agent_itc)

                    if gt_net > 0:
                        within_tolerance = abs(agent_net - gt_net) / gt_net <= 0.05
                    else:
                        within_tolerance = agent_net == 0.0

                    step_reward = reward_compute_liability(within_tolerance)
                    self._liability_computed = True

        # --- file_return ---
        elif action.action_type == "file_return":
            if self._task == "invoice_classifier":
                step_reward = -0.20
                self._last_error = "file_return not valid for invoice_classifier task"
                self._done = True
            elif action.gstr_payload is None:
                step_reward = -0.20
                self._last_error = "file_return requires gstr_payload"
                self._done = True
            else:
                # Check for premature filing (incomplete actions)
                all_ids = [i["invoice_id"] for i in invoices]
                total = len(all_ids)
                # BUG-8 fix: guard against empty invoice list
                classified_pct = len(self._classified_so_far) / total if total > 0 else 1.0
                reconciled_pct = len(self._itc_decisions) / total if total > 0 else 1.0

                if classified_pct < 0.5 or reconciled_pct < 0.5:
                    # Premature filing penalty — no late penalty on this path
                    step_reward = -0.20
                    self._last_error = "Premature filing: complete classification and reconciliation first"
                else:
                    gt_payload = compute_ground_truth_gstr3b(invoices, self._accepted_itc_ids)
                    agent_vals = _flatten_payload(action.gstr_payload)
                    truth_vals = _flatten_payload(gt_payload)

                    all_keys = set(truth_vals.keys())
                    if all_keys:
                        field_accs = [
                            _field_accuracy(agent_vals.get(k, 0.0), truth_vals[k])  # BUG-14
                            for k in all_keys
                        ]
                        avg_field_acc = sum(field_accs) / len(field_accs)
                    else:
                        avg_field_acc = 0.0

                    step_reward = reward_file_return(avg_field_acc)
                    self._final_payload = action.gstr_payload

                    # BUG-5 fix: late-filing penalty only applies to actual filings,
                    # not premature-filing rejections
                    if self._current_step > 55:
                        step_reward += -0.50

                self._done = True

        # --- Natural episode end: all invoices actioned for Task 1 and 2 ---
        if not self._done:
            all_ids = [i["invoice_id"] for i in invoices]
            if self._task == "invoice_classifier" and len(self._classified_so_far) == len(all_ids):
                self._done = True
                self._natural_end = True
            elif self._task == "itc_reconciliation" and len(self._itc_decisions) == len(all_ids):
                self._done = True
                self._natural_end = True

        # --- Check timeout ---
        if self._current_step >= max_steps and not self._done:
            step_reward += -0.50  # timeout penalty
            self._done = True

        self._cumulative_reward += _clamp(step_reward)
        return self._build_observation(reward=step_reward, done=self._done)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> GSTState:
        ep = self._episode_data or {}
        max_steps = ep.get("max_steps", 60)
        # Normalize cumulative_reward to (0, 1) so the OpenEnv validator never sees
        # values outside the legal range.  Raw cumulative is kept in _cumulative_reward
        # for RL training; the state property exposes a per-step average clamped to (0,1).
        if self._current_step > 0:
            norm_reward = _clamp(self._cumulative_reward / max_steps)
        else:
            norm_reward = _SCORE_MIN
        return GSTState(
            episode_id=self._episode_id,
            step_count=self._current_step,
            ground_truth_invoices=ep.get("ground_truth_invoices", []),
            ground_truth_itc=ep.get("ground_truth_itc", []),
            classified_so_far=self._classified_so_far,
            itc_decisions=self._itc_decisions,
            cumulative_reward=norm_reward,
            task=self._task,
            audit_flags=self._audit_flags,
            liability_computed=self._liability_computed,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float, done: bool,
                            error: Optional[str] = None) -> GSTObservation:
        ep = self._episode_data or {}
        invoices = ep.get("invoices", [])
        gstr_2b = ep.get("gstr_2b", [])
        max_steps = ep.get("max_steps", 20)
        context = ep.get("context", "mixed")

        # IDs not yet actioned
        task = self._task
        all_ids = [i["invoice_id"] for i in invoices]
        if task == "invoice_classifier":
            pending = [iid for iid in all_ids if iid not in self._classified_so_far]
        elif task == "itc_reconciliation":
            pending = [iid for iid in all_ids if iid not in self._itc_decisions]
        else:
            # full_gstr3b: pending = unclassified + unreconciled
            unclassified = [iid for iid in all_ids if iid not in self._classified_so_far]
            unreconciled = [iid for iid in all_ids if iid not in self._itc_decisions]
            pending = list(dict.fromkeys(unclassified + unreconciled))  # preserve order, dedup
            # BUG-8 fix: when all invoices are processed guide agent to next mandatory steps
            # so it doesn't spin on an empty list until timeout
            if not pending and not done:
                if not self._liability_computed:
                    pending = ["compute_liability"]
                else:
                    pending = ["file_return"]

        # Compute accumulated ITC
        accumulated_itc = sum(
            inv.get("igst", 0) + inv.get("cgst", 0) + inv.get("sgst", 0)
            for inv in invoices
            if inv["invoice_id"] in self._accepted_itc_ids
        )

        # Compute running tax liability (outward supplies)
        tax_liability = sum(
            inv.get("igst", 0) + inv.get("cgst", 0) + inv.get("sgst", 0)
            for inv in invoices
            if self._classified_so_far.get(inv["invoice_id"], {}).get("invoice_type") in ("B2B", "B2C")
        )

        # Task 1 never files. Task 2 only shows "n/a" when it ends naturally
        # (all ITCs reconciled) — not when file_return is explicitly rejected.
        # Task 3 (full_gstr3b_filing) genuinely requires filing: use filed/failed/in_progress.
        if self._task == "invoice_classifier":
            filing_status = "n/a"
        elif self._task == "itc_reconciliation" and self._natural_end:
            filing_status = "n/a"
        elif self._done and self._final_payload is not None:
            filing_status = "filed"
        elif self._done:
            filing_status = "failed"
        elif self._current_step > 0:
            filing_status = "in_progress"
        else:
            filing_status = "not_started"

        err = error or self._last_error

        # Compute final task score when the episode ends so validators that read
        # obs.metadata["score"] get a value strictly within (0, 1).
        metadata: dict = {}
        if done and self._episode_data is not None:
            gt_invoices = self._episode_data.get("ground_truth_invoices", [])
            gt_itc      = self._episode_data.get("ground_truth_itc", [])
            if self._task == "invoice_classifier":
                metadata["score"] = grade_invoice_classifier(
                    gt_invoices, self._classified_so_far
                )
            elif self._task == "itc_reconciliation":
                metadata["score"] = grade_itc_reconciliation(
                    gt_itc, self._itc_decisions
                )
            elif self._task == "full_gstr3b_filing":
                if self._final_payload is not None:
                    gt_payload = compute_ground_truth_gstr3b(
                        invoices, self._accepted_itc_ids
                    )
                    metadata["score"] = grade_gstr3b_filing(
                        self._final_payload, gt_payload,
                        self._audit_flags, self._current_step, max_steps,
                    )
                else:
                    metadata["score"] = _SCORE_MIN

        return GSTObservation(
            reward=round(_clamp(reward), 2),
            done=done,
            invoices=[invoice_for_agent(inv) for inv in invoices],
            gstr_2b=gstr_2b,
            accumulated_itc=round(accumulated_itc, 2),
            tax_liability=round(tax_liability, 2),
            step=self._current_step,
            max_steps=max_steps,
            episode_context=context,
            pending_actions=pending,
            filing_status=filing_status,
            last_error=err,
            metadata=metadata,
        )


# --- Utility functions ---

def _flatten_payload(payload: dict, prefix: str = "") -> dict:
    """Flatten nested GSTR-3B payload dict into dot-notation keys."""
    result = {}
    for k, v in payload.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_payload(v, key))
        else:
            result[key] = float(v) if v is not None else 0.0
    return result


# _field_accuracy imported from graders.py — BUG-14: single canonical implementation
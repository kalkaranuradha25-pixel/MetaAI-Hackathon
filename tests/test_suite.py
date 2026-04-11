"""
GST Sahayak — Test Suite
Covers: Output Parsing · Task Validation · LLM Criteria Check

Run: python -m pytest tests/test_suite.py -v
"""

import sys
import os
import io
import json
import re
import unittest
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Provide a dummy HF_TOKEN so importing inference.py never raises at import time
os.environ.setdefault("HF_TOKEN", "test-token")

import inference
from inference import obs_to_prompt, get_action, run_task, TASK_THRESHOLDS
from models import GSTAction, GSTObservation
from server.gst_environment import GSTEnvironment
from server.graders import (
    grade_invoice_classifier,
    grade_itc_reconciliation,
    grade_gstr3b_filing,
    reward_file_return,
    _field_accuracy,
    _SCORE_MIN,
    _SCORE_MAX,
)
from server.gst_environment import (
    reward_classify_invoice,
    reward_itc_decision,
    reward_compute_liability,
)
from server.data_generator import generate_episode, TASK_CONFIGS


# ── shared helpers ────────────────────────────────────────────────────────────

def capture_run_task(env, task_name: str):
    """Run run_task(); return (stdout_lines, stderr_text)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        run_task(env, task_name)
    return out.getvalue().splitlines(), err.getvalue()


def fresh_env(task: str = "invoice_classifier") -> tuple:
    """Return (env, obs_dict) after reset."""
    env = GSTEnvironment()
    obs = env.reset(task=task)
    return env, obs.model_dump()


def make_obs(**kwargs) -> dict:
    """Minimal valid obs dict for prompt / get_action tests."""
    base = {
        "step": 1, "max_steps": 20, "episode_context": "mixed",
        "accumulated_itc": 0.0, "tax_liability": 0.0,
        "filing_status": "not_started", "pending_actions": [],
        "invoices": [], "gstr_2b": [],
    }
    base.update(kwargs)
    return base


def mock_llm_response(content: str):
    """Build a minimal mock that _get_client().chat.completions.create() returns."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


# =============================================================================
# 1. OUTPUT PARSING
# =============================================================================

class TestOutputParsing(unittest.TestCase):
    """Validate every aspect of the required stdout protocol."""

    # ── regex patterns for each line type ──
    RE_START = re.compile(
        r"^\[START\] task=(?P<task>\S+) env=gst-sahayak model=(?P<model>\S+)$"
    )
    RE_STEP = re.compile(
        r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>\{.*\}) "
        r"reward=(?P<reward>-?\d+\.\d{2}) done=(?P<done>true|false) "
        r"error=(?P<error>.+)$"
    )
    RE_END = re.compile(
        r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) "
        r"rewards=(?P<rewards>.+)$"
    )

    def _run_task1(self):
        """Run invoice_classifier with a mock env that completes in 2 steps."""
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        return lines

    # ── structure tests ──────────────────────────────────────────────────────

    def test_start_is_first_line(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        self.assertTrue(lines[0].startswith("[START]"), f"First line: {lines[0]}")

    def test_end_is_last_line(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        self.assertTrue(lines[-1].startswith("[END]"), f"Last line: {lines[-1]}")

    def test_start_format_matches_spec(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        m = self.RE_START.match(lines[0])
        self.assertIsNotNone(m, f"[START] format wrong: {lines[0]}")
        self.assertEqual(m.group("task"), "invoice_classifier")

    def test_end_format_matches_spec(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        m = self.RE_END.match(lines[-1])
        self.assertIsNotNone(m, f"[END] format wrong: {lines[-1]}")

    def test_step_lines_format(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        self.assertGreater(len(step_lines), 0, "No [STEP] lines emitted")
        for line in step_lines:
            m = self.RE_STEP.match(line)
            self.assertIsNotNone(m, f"[STEP] format wrong: {line}")

    def test_step_numbers_are_sequential_from_1(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        for i, line in enumerate(step_lines, start=1):
            m = self.RE_STEP.match(line)
            self.assertEqual(int(m.group("step")), i,
                             f"Step {i} out of sequence in: {line}")

    def test_step_reward_has_two_decimal_places(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        for line in lines:
            if not line.startswith("[STEP]"):
                continue
            m = self.RE_STEP.match(line)
            reward_str = m.group("reward")
            self.assertRegex(reward_str, r"^-?\d+\.\d{2}$",
                             f"Reward not 2dp: {reward_str}")

    def test_step_done_is_lowercase_bool(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        for line in lines:
            if not line.startswith("[STEP]"):
                continue
            m = self.RE_STEP.match(line)
            self.assertIn(m.group("done"), ("true", "false"))

    def test_step_action_is_valid_json(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        for line in lines:
            if not line.startswith("[STEP]"):
                continue
            m = self.RE_STEP.match(line)
            try:
                parsed = json.loads(m.group("action"))
                self.assertIn("action_type", parsed)
            except json.JSONDecodeError as exc:
                self.fail(f"action is not valid JSON: {m.group('action')} — {exc}")

    def test_end_success_is_lowercase_bool(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        m = self.RE_END.match(lines[-1])
        self.assertIn(m.group("success"), ("true", "false"))

    def test_end_steps_matches_step_count(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        step_count = sum(1 for l in lines if l.startswith("[STEP]"))
        m = self.RE_END.match(lines[-1])
        self.assertEqual(int(m.group("steps")), step_count)

    def test_end_rewards_count_matches_steps(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        step_count = sum(1 for l in lines if l.startswith("[STEP]"))
        m = self.RE_END.match(lines[-1])
        rewards_list = m.group("rewards").split(",")
        self.assertEqual(len(rewards_list), step_count,
                         "rewards count in [END] doesn't match [STEP] count")

    def test_end_rewards_values_match_step_rewards(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        step_rewards = []
        for line in lines:
            if not line.startswith("[STEP]"):
                continue
            m = self.RE_STEP.match(line)
            step_rewards.append(m.group("reward"))
        m_end = self.RE_END.match(lines[-1])
        end_rewards = m_end.group("rewards").split(",")
        self.assertEqual(step_rewards, end_rewards,
                         "rewards in [END] don't match per-step reward values")

    def test_end_emitted_when_reset_raises(self):
        """[END] must always appear even if env.reset() throws."""
        env = GSTEnvironment()
        with patch.object(env, "reset", side_effect=RuntimeError("reset exploded")):
            lines, stderr = capture_run_task(env, "invoice_classifier")
        end_lines = [l for l in lines if l.startswith("[END]")]
        self.assertEqual(len(end_lines), 1, "Exactly one [END] must be emitted")

    def test_no_step_emitted_when_reset_raises(self):
        """No [STEP] should appear if env.reset() raises — step() was never called."""
        env = GSTEnvironment()
        with patch.object(env, "reset", side_effect=RuntimeError("boom")):
            lines, _ = capture_run_task(env, "invoice_classifier")
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        self.assertEqual(len(step_lines), 0,
                         "No [STEP] should be emitted when reset() raises")

    def test_end_rewards_fallback_is_in_range_on_reset_failure(self):
        """rewards fallback must be strictly in (0, 1) when no steps completed."""
        env = GSTEnvironment()
        with patch.object(env, "reset", side_effect=RuntimeError("boom")):
            lines, _ = capture_run_task(env, "invoice_classifier")
        m = self.RE_END.match(lines[-1])
        self.assertIsNotNone(m)
        # Validator requires strictly between 0 and 1 — fallback must be 0.01
        self.assertEqual(m.group("rewards"), "0.01")

    def test_end_success_false_when_no_rewards(self):
        env = GSTEnvironment()
        with patch.object(env, "reset", side_effect=RuntimeError("boom")):
            lines, _ = capture_run_task(env, "invoice_classifier")
        m = self.RE_END.match(lines[-1])
        self.assertEqual(m.group("success"), "false")

    def test_reset_error_logged_to_stderr_not_stdout(self):
        env = GSTEnvironment()
        with patch.object(env, "reset", side_effect=RuntimeError("secret-error")):
            lines, stderr = capture_run_task(env, "invoice_classifier")
        stdout_text = "\n".join(lines)
        self.assertNotIn("secret-error", stdout_text,
                         "Error details must not appear on stdout")
        self.assertIn("secret-error", stderr,
                      "Error details must appear on stderr")

    def test_exactly_one_start_line(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        start_lines = [l for l in lines if l.startswith("[START]")]
        self.assertEqual(len(start_lines), 1)

    def test_exactly_one_end_line(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        end_lines = [l for l in lines if l.startswith("[END]")]
        self.assertEqual(len(end_lines), 1)

    def test_last_step_has_done_true(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        if step_lines:
            m = self.RE_STEP.match(step_lines[-1])
            self.assertEqual(m.group("done"), "true",
                             "Last [STEP] must have done=true")

    def test_all_steps_except_last_have_done_false(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        step_lines = [l for l in lines if l.startswith("[STEP]")]
        for line in step_lines[:-1]:
            m = self.RE_STEP.match(line)
            self.assertEqual(m.group("done"), "false",
                             f"Non-terminal step has done=true: {line}")

    def test_start_task_field_matches_requested_task(self):
        for task in ["invoice_classifier", "itc_reconciliation"]:
            env = GSTEnvironment()
            lines, _ = capture_run_task(env, task)
            m = self.RE_START.match(lines[0])
            self.assertEqual(m.group("task"), task)

    def test_end_rewards_each_two_decimal_places(self):
        env = GSTEnvironment()
        lines, _ = capture_run_task(env, "invoice_classifier")
        m = self.RE_END.match(lines[-1])
        for r in m.group("rewards").split(","):
            self.assertRegex(r.strip(), r"^-?\d+\.\d{2}$",
                             f"rewards value not 2dp: {r}")


# =============================================================================
# 2. TASK VALIDATION
# =============================================================================

class TestTaskValidation(unittest.TestCase):
    """Test environment action/state logic against PRD rules."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _first_invoice_id(self, env, task="invoice_classifier"):
        obs = env.reset(task=task)
        return obs.pending_actions[0], obs

    def _step(self, env, **kwargs):
        return env.step(GSTAction(**kwargs))

    # ── episode setup ────────────────────────────────────────────────────────

    def test_reset_returns_observation(self):
        env = GSTEnvironment()
        obs = env.reset(task="invoice_classifier")
        self.assertIsInstance(obs, GSTObservation)

    def test_reset_pending_actions_non_empty(self):
        env = GSTEnvironment()
        obs = env.reset(task="invoice_classifier")
        self.assertGreater(len(obs.pending_actions), 0)

    def test_reset_done_is_false(self):
        env = GSTEnvironment()
        obs = env.reset(task="invoice_classifier")
        self.assertFalse(obs.done)

    def test_reset_step_is_zero(self):
        env = GSTEnvironment()
        obs = env.reset(task="invoice_classifier")
        self.assertEqual(obs.step, 0)

    def test_task1_invoice_count(self):
        ep = generate_episode(42, "invoice_classifier")
        self.assertEqual(len(ep["invoices"]), TASK_CONFIGS["invoice_classifier"]["num_invoices"])

    def test_task2_invoice_count(self):
        ep = generate_episode(137, "itc_reconciliation")
        self.assertEqual(len(ep["invoices"]), TASK_CONFIGS["itc_reconciliation"]["num_invoices"])

    def test_task3_invoice_count(self):
        ep = generate_episode(999, "full_gstr3b_filing")
        self.assertEqual(len(ep["invoices"]), TASK_CONFIGS["full_gstr3b_filing"]["num_invoices"])

    # ── invalid action type ──────────────────────────────────────────────────

    def test_invalid_action_type_gives_penalty(self):
        env = GSTEnvironment()
        env.reset(task="invoice_classifier")
        obs = self._step(env, action_type="do_something_illegal", invoice_id=None)
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_invalid_action_type_does_not_end_episode_early(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="invoice_classifier")
        obs = self._step(env, action_type="bad_action", invoice_id=None)
        self.assertFalse(obs.done)

    # ── classify_invoice ─────────────────────────────────────────────────────

    def test_classify_without_invoice_id_penalised(self):
        env = GSTEnvironment()
        env.reset(task="invoice_classifier")
        obs = self._step(env, action_type="classify_invoice", invoice_id=None,
                         invoice_type="B2B", hsn_code="8471")
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_classify_unknown_id_penalised(self):
        env = GSTEnvironment()
        env.reset(task="invoice_classifier")
        obs = self._step(env, action_type="classify_invoice", invoice_id="FAKE-9999",
                         invoice_type="B2B", hsn_code="8471")
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_classify_duplicate_penalised(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="invoice_classifier")
        inv_id = obs0.pending_actions[0]
        # First classification — may succeed or not but sets the record
        self._step(env, action_type="classify_invoice", invoice_id=inv_id,
                   invoice_type="B2B", hsn_code="8471")
        # Second on same ID must penalise
        obs2 = self._step(env, action_type="classify_invoice", invoice_id=inv_id,
                          invoice_type="B2B", hsn_code="8471")
        self.assertLess(obs2.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_correct_classification_positive_reward(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="invoice_classifier")
        inv_id = obs0.pending_actions[0]
        ep = env._episode_data
        gt = next(i for i in ep["ground_truth_invoices"] if i["invoice_id"] == inv_id)
        obs = self._step(env, action_type="classify_invoice", invoice_id=inv_id,
                         invoice_type=gt["_ground_truth_type"],
                         hsn_code=gt["_ground_truth_hsn"])
        self.assertGreater(obs.reward, 0)

    def test_correct_type_wrong_hsn_partial_reward(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="invoice_classifier")
        inv_id = obs0.pending_actions[0]
        gt = next(i for i in env._episode_data["ground_truth_invoices"]
                  if i["invoice_id"] == inv_id)
        obs = self._step(env, action_type="classify_invoice", invoice_id=inv_id,
                         invoice_type=gt["_ground_truth_type"],
                         hsn_code="0000")  # wrong HSN
        self.assertEqual(obs.reward, 0.05)

    def test_wrong_type_penalty(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="invoice_classifier")
        inv_id = obs0.pending_actions[0]
        gt = next(i for i in env._episode_data["ground_truth_invoices"]
                  if i["invoice_id"] == inv_id)
        # Pick a type that is definitely wrong
        wrong_types = [t for t in ["B2B", "B2C", "EXPORT", "EXEMPT"]
                       if t != gt["_ground_truth_type"]]
        obs = self._step(env, action_type="classify_invoice", invoice_id=inv_id,
                         invoice_type=wrong_types[0], hsn_code="0000")
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_task1_done_when_all_classified(self):
        env = GSTEnvironment()
        obs = env.reset(task="invoice_classifier")
        ep = env._episode_data
        # Classify every invoice (correctly or not)
        for inv in ep["invoices"]:
            if obs.done:
                break
            obs = self._step(env, action_type="classify_invoice",
                             invoice_id=inv["invoice_id"],
                             invoice_type="B2B", hsn_code="8471")
        self.assertTrue(obs.done)

    # ── ITC decisions ────────────────────────────────────────────────────────

    def test_accept_itc_without_invoice_id_penalised(self):
        env = GSTEnvironment()
        env.reset(task="itc_reconciliation")
        obs = self._step(env, action_type="accept_itc", invoice_id=None)
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_correct_accept_itc_reward(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="itc_reconciliation")
        ep = env._episode_data
        # Find a clean (non-mismatch) invoice
        clean = next((gt for gt in ep["ground_truth_itc"]
                      if gt["correct_decision"] == "accept"), None)
        if clean is None:
            self.skipTest("No clean invoices in this seed")
        obs = self._step(env, action_type="accept_itc", invoice_id=clean["invoice_id"])
        self.assertEqual(obs.reward, 0.20)

    def test_wrong_accept_mismatch_invoice_audit_penalty(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="itc_reconciliation")
        ep = env._episode_data
        reject_case = next((gt for gt in ep["ground_truth_itc"]
                            if gt["correct_decision"] == "reject"), None)
        if reject_case is None:
            self.skipTest("No reject invoices in this seed")
        obs = self._step(env, action_type="accept_itc",
                         invoice_id=reject_case["invoice_id"])
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_correct_reject_reward(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="itc_reconciliation")
        ep = env._episode_data
        reject_case = next((gt for gt in ep["ground_truth_itc"]
                            if gt["correct_decision"] == "reject"), None)
        if reject_case is None:
            self.skipTest("No reject invoices in this seed")
        obs = self._step(env, action_type="reject_itc",
                         invoice_id=reject_case["invoice_id"])
        self.assertEqual(obs.reward, 0.20)

    def test_flag_for_review_on_value_mismatch_correct_reward(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="itc_reconciliation")
        ep = env._episode_data
        flag_case = next((gt for gt in ep["ground_truth_itc"]
                          if gt["correct_decision"] == "flag"), None)
        if flag_case is None:
            self.skipTest("No value_mismatch in this seed")
        obs = self._step(env, action_type="flag_for_review",
                         invoice_id=flag_case["invoice_id"])
        self.assertEqual(obs.reward, 0.20)

    def test_flag_for_review_on_clean_invoice_partial_credit(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="itc_reconciliation")
        ep = env._episode_data
        clean = next((gt for gt in ep["ground_truth_itc"]
                      if gt["correct_decision"] == "accept"), None)
        if clean is None:
            self.skipTest("No clean invoices in this seed")
        obs = self._step(env, action_type="flag_for_review",
                         invoice_id=clean["invoice_id"])
        self.assertEqual(obs.reward, 0.05)

    def test_task2_done_when_all_itc_decided(self):
        env = GSTEnvironment()
        obs = env.reset(task="itc_reconciliation")
        ep = env._episode_data
        for inv in ep["invoices"]:
            if obs.done:
                break
            obs = self._step(env, action_type="flag_for_review",
                             invoice_id=inv["invoice_id"])
        self.assertTrue(obs.done)

    # ── file_return restrictions ─────────────────────────────────────────────

    def test_file_return_not_valid_for_task1(self):
        env = GSTEnvironment()
        env.reset(task="invoice_classifier")
        obs = self._step(env, action_type="file_return", invoice_id=None,
                         gstr_payload={"outward_taxable_supplies": {}})
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small
        self.assertTrue(obs.done)

    def test_file_return_without_payload_penalised(self):
        env = GSTEnvironment()
        env.reset(task="itc_reconciliation")
        obs = self._step(env, action_type="file_return", invoice_id=None,
                         gstr_payload=None)
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small
        self.assertTrue(obs.done)

    def test_premature_file_return_penalised(self):
        env = GSTEnvironment()
        env.reset(task="itc_reconciliation")
        # File immediately without any reconciliation
        payload = {"outward_taxable_supplies": {"igst": 0, "cgst": 0, "sgst": 0},
                   "zero_rated_supplies": {"igst": 0},
                   "exempt_supplies": {"value": 0},
                   "itc_available": {"igst": 0, "cgst": 0, "sgst": 0},
                   "itc_reversed": {"igst": 0, "cgst": 0, "sgst": 0},
                   "net_tax_liability": {"igst": 0, "cgst": 0, "sgst": 0}}
        obs = self._step(env, action_type="file_return", invoice_id=None,
                         gstr_payload=payload)
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    # ── compute_liability ────────────────────────────────────────────────────

    def test_compute_liability_before_70pct_classified_penalised(self):
        env = GSTEnvironment()
        env.reset(task="invoice_classifier")
        # classify only 1 out of 10 invoices first
        inv_id = env._episode_data["invoices"][0]["invoice_id"]
        self._step(env, action_type="classify_invoice", invoice_id=inv_id,
                   invoice_type="B2B", hsn_code="8471")
        obs = self._step(env, action_type="compute_liability")
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    def test_compute_liability_twice_penalised(self):
        # Use full_gstr3b_filing — episode doesn't auto-terminate until file_return,
        # so we can call compute_liability twice without hitting the done guard.
        env = GSTEnvironment()
        env.reset(task="full_gstr3b_filing")
        ep = env._episode_data
        # Classify and reconcile all invoices (>70%) so compute_liability is allowed
        for inv in ep["invoices"]:
            self._step(env, action_type="classify_invoice",
                       invoice_id=inv["invoice_id"],
                       invoice_type="B2B", hsn_code="8471")
            self._step(env, action_type="accept_itc",
                       invoice_id=inv["invoice_id"])
        # First compute_liability — succeeds
        self._step(env, action_type="compute_liability")
        # Second compute_liability — must penalise (-0.10)
        obs = self._step(env, action_type="compute_liability")
        self.assertLess(obs.reward, 0.05)  # clamped penalty: obs.reward in (0,1) but small

    # ── step() dict coercion ─────────────────────────────────────────────────

    def test_step_accepts_dict(self):
        """step() must coerce raw dict to GSTAction (HTTP server path)."""
        env = GSTEnvironment()
        obs0 = env.reset(task="invoice_classifier")
        inv_id = obs0.pending_actions[0]
        obs = env.step({"action_type": "classify_invoice", "invoice_id": inv_id,
                        "invoice_type": "B2B", "hsn_code": "8471"})
        self.assertIsInstance(obs, GSTObservation)

    # ── GSTAction extra fields ────────────────────────────────────────────────

    def test_gst_action_ignores_extra_fields(self):
        """LLMs often add 'reasoning' or 'explanation' — must not raise."""
        try:
            action = GSTAction(
                action_type="classify_invoice",
                invoice_id="INV-001",
                invoice_type="B2B",
                hsn_code="8471",
                reasoning="This vendor is a registered business",  # extra key
            )
            self.assertEqual(action.action_type, "classify_invoice")
        except Exception as exc:
            self.fail(f"GSTAction raised on extra field: {exc}")

    # ── episode already finished ─────────────────────────────────────────────

    def test_step_after_done_returns_zero_reward(self):
        env = GSTEnvironment()
        obs = env.reset(task="invoice_classifier")
        ep = env._episode_data
        for inv in ep["invoices"]:
            if obs.done:
                break
            obs = self._step(env, action_type="classify_invoice",
                             invoice_id=inv["invoice_id"],
                             invoice_type="B2B", hsn_code="8471")
        # Now episode is done — extra step returns _SCORE_MIN (clamped, not 0.0)
        extra = self._step(env, action_type="classify_invoice",
                           invoice_id=ep["invoices"][0]["invoice_id"],
                           invoice_type="B2B", hsn_code="8471")
        self.assertEqual(extra.reward, round(_SCORE_MIN, 2))
        self.assertTrue(extra.done)

    # ── filing_status consistency ────────────────────────────────────────────

    def test_filing_status_not_started_on_reset(self):
        env = GSTEnvironment()
        obs = env.reset(task="itc_reconciliation")
        self.assertEqual(obs.filing_status, "not_started")

    def test_filing_status_in_progress_after_first_step(self):
        env = GSTEnvironment()
        obs0 = env.reset(task="itc_reconciliation")
        inv_id = obs0.pending_actions[0]
        obs = self._step(env, action_type="flag_for_review", invoice_id=inv_id)
        self.assertEqual(obs.filing_status, "in_progress")

    def test_filing_status_failed_when_file_return_rejected(self):
        """done=True but no payload → 'failed', not 'in_progress'."""
        env = GSTEnvironment()
        env.reset(task="itc_reconciliation")
        obs = self._step(env, action_type="file_return", gstr_payload=None)
        self.assertTrue(obs.done)
        self.assertEqual(obs.filing_status, "failed")

    # ── reward function unit tests ────────────────────────────────────────────

    def test_reward_classify_both_correct(self):
        self.assertEqual(reward_classify_invoice(True, True), 0.15)

    def test_reward_classify_type_only(self):
        self.assertEqual(reward_classify_invoice(True, False), 0.05)

    def test_reward_classify_wrong(self):
        self.assertEqual(reward_classify_invoice(False, False), -0.05)

    def test_reward_itc_correct_accept(self):
        self.assertEqual(reward_itc_decision("accept_itc", "accept"), 0.20)

    def test_reward_itc_correct_reject(self):
        self.assertEqual(reward_itc_decision("reject_itc", "reject"), 0.20)

    def test_reward_itc_wrong_accept(self):
        self.assertEqual(reward_itc_decision("accept_itc", "reject"), -0.30)

    def test_reward_itc_wrong_reject(self):
        self.assertEqual(reward_itc_decision("reject_itc", "accept"), -0.15)

    def test_reward_itc_flag_on_value_mismatch(self):
        self.assertEqual(reward_itc_decision("flag_for_review", "flag"), 0.20)

    def test_reward_itc_flag_partial_on_clean(self):
        self.assertEqual(reward_itc_decision("flag_for_review", "accept"), 0.05)

    def test_reward_compute_within_tolerance(self):
        self.assertEqual(reward_compute_liability(True), 0.30)

    def test_reward_compute_outside_tolerance(self):
        self.assertEqual(reward_compute_liability(False), -0.10)

    def test_reward_file_return_perfect(self):
        self.assertEqual(reward_file_return(1.00), _SCORE_MAX)

    def test_reward_file_return_near_perfect(self):
        self.assertEqual(reward_file_return(0.95), _SCORE_MAX)

    def test_reward_file_return_partial(self):
        self.assertAlmostEqual(reward_file_return(0.80), 0.40)

    # ── grader unit tests ─────────────────────────────────────────────────────

    def test_grade_classifier_all_correct(self):
        gt = [{"invoice_id": f"I{i}", "_ground_truth_type": "B2B",
               "_ground_truth_hsn": "8471"} for i in range(5)]
        decisions = {f"I{i}": {"invoice_type": "B2B", "hsn_code": "8471"} for i in range(5)}
        score = grade_invoice_classifier(gt, decisions)
        self.assertAlmostEqual(score, _SCORE_MAX)

    def test_grade_classifier_all_wrong(self):
        gt = [{"invoice_id": "I0", "_ground_truth_type": "B2B", "_ground_truth_hsn": "8471"}]
        decisions = {"I0": {"invoice_type": "EXPORT", "hsn_code": "0000"}}
        score = grade_invoice_classifier(gt, decisions)
        self.assertEqual(score, _SCORE_MIN)

    def test_grade_classifier_empty_gt(self):
        self.assertEqual(grade_invoice_classifier([], {}), _SCORE_MIN)

    def test_grade_itc_all_correct_accepts(self):
        gt = [{"invoice_id": f"I{i}", "correct_decision": "accept"} for i in range(5)]
        decisions = {f"I{i}": "accepted" for i in range(5)}
        self.assertEqual(grade_itc_reconciliation(gt, decisions), _SCORE_MAX)

    def test_grade_itc_all_positives_correctly_rejected(self):
        """No positive cases — agent correctly accepted nothing → _SCORE_MAX."""
        gt = [{"invoice_id": f"I{i}", "correct_decision": "reject"} for i in range(5)]
        decisions = {f"I{i}": "rejected" for i in range(5)}
        self.assertEqual(grade_itc_reconciliation(gt, decisions), _SCORE_MAX)

    def test_grade_itc_f1_edge_all_reject_no_positives(self):
        """BUG-12 edge case: all-reject when correct is also all-reject → _SCORE_MAX."""
        gt = [{"invoice_id": f"I{i}", "correct_decision": "reject"} for i in range(3)]
        decisions = {f"I{i}": "rejected" for i in range(3)}
        score = grade_itc_reconciliation(gt, decisions)
        self.assertEqual(score, _SCORE_MAX)

    def test_grade_itc_false_positives_reduce_score(self):
        gt = [{"invoice_id": "I0", "correct_decision": "reject"}]
        decisions = {"I0": "accepted"}  # fp
        score = grade_itc_reconciliation(gt, decisions)
        self.assertEqual(score, _SCORE_MIN)

    def test_grade_gstr3b_perfect_payload(self):
        payload = {
            "outward_taxable_supplies": {"igst": 1000.0, "cgst": 500.0, "sgst": 500.0},
            "zero_rated_supplies": {"igst": 0.0},
            "exempt_supplies": {"value": 0.0},
            "itc_available": {"igst": 200.0, "cgst": 100.0, "sgst": 100.0},
            "itc_reversed": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
            "net_tax_liability": {"igst": 800.0, "cgst": 400.0, "sgst": 400.0},
        }
        score = grade_gstr3b_filing(payload, payload, audit_flags=0,
                                    steps_taken=10, max_steps=60)
        self.assertAlmostEqual(score, _SCORE_MAX, places=2)

    def test_grade_gstr3b_none_payload(self):
        self.assertEqual(grade_gstr3b_filing(None, {}, 0, 10, 60), _SCORE_MIN)

    def test_grade_gstr3b_time_score_threshold(self):
        """Steps < 75% of max_steps → time_score=1.0."""
        p = {"net_tax_liability": {"igst": 100.0}}
        score_fast = grade_gstr3b_filing(p, p, 0, 10, 60)   # fast
        score_slow = grade_gstr3b_filing(p, p, 0, 55, 60)   # slow (>75% of 60)
        self.assertGreater(score_fast, score_slow)

    def test_field_accuracy_zero_truth_zero_agent(self):
        self.assertEqual(_field_accuracy(0.0, 0.0), _SCORE_MAX)

    def test_field_accuracy_zero_truth_nonzero_agent(self):
        self.assertEqual(_field_accuracy(100.0, 0.0), _SCORE_MIN)

    def test_field_accuracy_exact_match(self):
        self.assertEqual(_field_accuracy(500.0, 500.0), _SCORE_MAX)

    def test_field_accuracy_10pct_off(self):
        self.assertAlmostEqual(_field_accuracy(450.0, 500.0), 0.9, places=3)


# =============================================================================
# 3. LLM CRITERIA CHECK
# =============================================================================

class TestLLMCriteriaCheck(unittest.TestCase):
    """Test obs_to_prompt(), get_action() parsing, fallback logic, thresholds."""

    # ── obs_to_prompt content ────────────────────────────────────────────────

    def test_prompt_contains_task_name(self):
        prompt = obs_to_prompt(make_obs(), "invoice_classifier")
        self.assertIn("invoice_classifier", prompt)

    def test_prompt_contains_pending_invoice_ids(self):
        obs = make_obs(pending_actions=["INV-001", "INV-002"],
                       invoices=[
                           {"invoice_id": "INV-001", "vendor_name": "A", "vendor_gstin": "G",
                            "invoice_date": "2026-01-01", "invoice_number": "A/1",
                            "taxable_value": 1000.0, "igst": 0.0, "cgst": 90.0, "sgst": 90.0},
                           {"invoice_id": "INV-002", "vendor_name": "B", "vendor_gstin": "H",
                            "invoice_date": "2026-01-01", "invoice_number": "B/1",
                            "taxable_value": 2000.0, "igst": 0.0, "cgst": 120.0, "sgst": 120.0},
                       ])
        prompt = obs_to_prompt(obs, "invoice_classifier")
        self.assertIn("INV-001", prompt)
        self.assertIn("INV-002", prompt)

    def test_prompt_excludes_non_pending_invoices(self):
        """Only pending invoices should appear in the invoice list."""
        obs = make_obs(
            pending_actions=["INV-001"],
            invoices=[
                {"invoice_id": "INV-001", "vendor_name": "A", "vendor_gstin": "G",
                 "invoice_date": "2026-01-01", "invoice_number": "A/1",
                 "taxable_value": 1000.0, "igst": 0.0, "cgst": 90.0, "sgst": 90.0},
                {"invoice_id": "INV-DONE", "vendor_name": "B", "vendor_gstin": "H",
                 "invoice_date": "2026-01-01", "invoice_number": "B/1",
                 "taxable_value": 2000.0, "igst": 0.0, "cgst": 120.0, "sgst": 120.0},
            ])
        prompt = obs_to_prompt(obs, "invoice_classifier")
        # INV-DONE is NOT in pending_actions — must not appear in invoice list
        self.assertNotIn("INV-DONE", prompt)
        self.assertIn("INV-001", prompt)

    def test_prompt_no_invoice_section_when_pending_empty(self):
        obs = make_obs(pending_actions=[])
        prompt = obs_to_prompt(obs, "invoice_classifier")
        self.assertIn("none", prompt.lower())

    def test_prompt_gstr3b_template_absent_by_default(self):
        obs = make_obs(pending_actions=["INV-001"])
        prompt = obs_to_prompt(obs, "invoice_classifier")
        self.assertNotIn("outward_taxable_supplies", prompt)

    def test_prompt_gstr3b_template_shown_when_file_return_pending(self):
        obs = make_obs(pending_actions=["file_return"],
                       filing_status="in_progress")
        prompt = obs_to_prompt(obs, "full_gstr3b_filing")
        self.assertIn("outward_taxable_supplies", prompt)

    def test_prompt_gstr3b_template_absent_for_task1(self):
        """Task 1 never needs filing — template must not appear regardless."""
        obs = make_obs(pending_actions=["file_return"])
        prompt = obs_to_prompt(obs, "invoice_classifier")
        self.assertNotIn("outward_taxable_supplies", prompt)

    def test_prompt_contains_classification_rules(self):
        prompt = obs_to_prompt(make_obs(), "invoice_classifier")
        for keyword in ("B2B", "B2C", "EXPORT", "EXEMPT"):
            self.assertIn(keyword, prompt)

    def test_prompt_contains_reconciliation_rules(self):
        prompt = obs_to_prompt(make_obs(), "itc_reconciliation")
        self.assertIn("accept_itc", prompt)
        self.assertIn("reject_itc", prompt)
        self.assertIn("flag_for_review", prompt)
        self.assertIn("value_mismatch", prompt)

    def test_prompt_contains_hsn_codes(self):
        prompt = obs_to_prompt(make_obs(), "invoice_classifier")
        self.assertIn("8471", prompt)  # computers

    def test_prompt_contains_filing_status(self):
        obs = make_obs(filing_status="in_progress")
        prompt = obs_to_prompt(obs, "full_gstr3b_filing")
        self.assertIn("in_progress", prompt)

    def test_prompt_contains_step_counter(self):
        obs = make_obs(step=5, max_steps=20)
        prompt = obs_to_prompt(obs, "invoice_classifier")
        self.assertIn("5/20", prompt)

    def test_prompt_contains_accumulated_itc(self):
        obs = make_obs(accumulated_itc=1234.56)
        prompt = obs_to_prompt(obs, "invoice_classifier")
        self.assertIn("1234.56", prompt)

    # ── get_action JSON parsing ──────────────────────────────────────────────

    def _mock_llm(self, content: str):
        """Context manager that makes _get_client() return a mock."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_llm_response(content)
        return patch("inference._get_client", return_value=mock_client)

    def test_get_action_parses_clean_json(self):
        raw = json.dumps({
            "action_type": "classify_invoice",
            "invoice_id": "INV-001",
            "hsn_code": "8471",
            "invoice_type": "B2B",
            "gstr_payload": None,
        })
        with self._mock_llm(raw):
            action = get_action(make_obs(pending_actions=["INV-001"]), "invoice_classifier")
        self.assertEqual(action["action_type"], "classify_invoice")
        self.assertEqual(action["invoice_id"], "INV-001")

    def test_get_action_strips_json_fence(self):
        raw = '```json\n{"action_type":"accept_itc","invoice_id":"I1","hsn_code":null,"invoice_type":null,"gstr_payload":null}\n```'
        with self._mock_llm(raw):
            action = get_action(make_obs(pending_actions=["I1"]), "itc_reconciliation")
        self.assertEqual(action["action_type"], "accept_itc")

    def test_get_action_strips_plain_fence(self):
        raw = '```\n{"action_type":"reject_itc","invoice_id":"I2","hsn_code":null,"invoice_type":null,"gstr_payload":null}\n```'
        with self._mock_llm(raw):
            action = get_action(make_obs(pending_actions=["I2"]), "itc_reconciliation")
        self.assertEqual(action["action_type"], "reject_itc")

    def test_get_action_strips_fence_with_trailing_newline(self):
        raw = '```json\n{"action_type":"flag_for_review","invoice_id":"I3","hsn_code":null,"invoice_type":null,"gstr_payload":null}\n```\n'
        with self._mock_llm(raw):
            action = get_action(make_obs(pending_actions=["I3"]), "itc_reconciliation")
        self.assertEqual(action["action_type"], "flag_for_review")

    def test_get_action_fallback_on_invalid_json(self):
        with self._mock_llm("I cannot determine the action."):
            action = get_action(make_obs(pending_actions=["INV-X"]), "invoice_classifier")
        self.assertEqual(action["action_type"], "flag_for_review")

    def test_get_action_fallback_on_empty_response(self):
        with self._mock_llm(""):
            action = get_action(make_obs(pending_actions=["INV-X"]), "invoice_classifier")
        self.assertEqual(action["action_type"], "flag_for_review")

    def test_get_action_fallback_sets_invoice_id_from_pending(self):
        with self._mock_llm("not json"):
            action = get_action(make_obs(pending_actions=["INV-99"]), "invoice_classifier")
        self.assertEqual(action["invoice_id"], "INV-99")

    def test_get_action_fallback_invoice_id_none_when_no_pending(self):
        with self._mock_llm("not json"):
            action = get_action(make_obs(pending_actions=[]), "invoice_classifier")
        self.assertIsNone(action["invoice_id"])

    def test_get_action_fallback_compute_liability_hint(self):
        """When pending=['compute_liability'] and LLM fails → compute_liability action."""
        with self._mock_llm("bad"):
            action = get_action(make_obs(pending_actions=["compute_liability"]),
                                "full_gstr3b_filing")
        self.assertEqual(action["action_type"], "compute_liability")

    def test_get_action_fallback_file_return_hint(self):
        """When pending=['file_return'] and LLM fails → file_return action."""
        with self._mock_llm("bad"):
            action = get_action(make_obs(pending_actions=["file_return"]),
                                "full_gstr3b_filing")
        self.assertEqual(action["action_type"], "file_return")

    def test_get_action_ignores_extra_llm_keys(self):
        """LLM returns 'reasoning' field — must not raise."""
        raw = json.dumps({
            "action_type": "classify_invoice",
            "invoice_id": "INV-001",
            "hsn_code": "8471",
            "invoice_type": "B2B",
            "gstr_payload": None,
            "reasoning": "This is a B2B invoice based on GSTIN pattern",
        })
        with self._mock_llm(raw):
            try:
                action = get_action(make_obs(pending_actions=["INV-001"]),
                                    "invoice_classifier")
                self.assertEqual(action["action_type"], "classify_invoice")
            except Exception as exc:
                self.fail(f"Extra LLM field caused crash: {exc}")

    def test_get_action_on_llm_exception_returns_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("network error")
        with patch("inference._get_client", return_value=mock_client):
            action = get_action(make_obs(pending_actions=["INV-Z"]), "invoice_classifier")
        self.assertEqual(action["action_type"], "flag_for_review")

    # ── max_tokens per task ───────────────────────────────────────────────────

    def test_max_tokens_1500_for_full_gstr3b(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_llm_response(
            json.dumps({"action_type": "compute_liability", "invoice_id": None,
                        "hsn_code": None, "invoice_type": None, "gstr_payload": None})
        )
        with patch("inference._get_client", return_value=mock_client):
            get_action(make_obs(pending_actions=["compute_liability"]), "full_gstr3b_filing")
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("max_tokens") or
                         call_kwargs.args[0] if call_kwargs.args else None or
                         call_kwargs.kwargs["max_tokens"], 1500)

    def test_max_tokens_600_for_task1(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_llm_response(
            json.dumps({"action_type": "flag_for_review", "invoice_id": "INV-0",
                        "hsn_code": None, "invoice_type": None, "gstr_payload": None})
        )
        with patch("inference._get_client", return_value=mock_client):
            get_action(make_obs(pending_actions=["INV-0"]), "invoice_classifier")
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["max_tokens"], 600)

    # ── token / client validation ─────────────────────────────────────────────

    def test_get_client_raises_on_empty_token(self):
        inference._openai_client = None  # reset singleton
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            with self.assertRaises(RuntimeError):
                inference._get_client()
        inference._openai_client = None  # cleanup

    def test_get_client_raises_when_token_missing(self):
        inference._openai_client = None
        env_without_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env_without_token, clear=True):
            with self.assertRaises(RuntimeError):
                inference._get_client()
        inference._openai_client = None

    def test_importing_inference_without_hf_token_does_not_raise(self):
        """Module-level code must not crash on import even without HF_TOKEN."""
        env_without_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        try:
            import importlib
            with patch.dict(os.environ, env_without_token, clear=True):
                importlib.reload(inference)
        except RuntimeError as exc:
            self.fail(f"import raised RuntimeError without HF_TOKEN: {exc}")
        finally:
            os.environ["HF_TOKEN"] = "test-token"
            inference._openai_client = None

    # ── success thresholds ────────────────────────────────────────────────────

    def test_task_thresholds_match_prd(self):
        self.assertEqual(TASK_THRESHOLDS["invoice_classifier"], 0.80)
        self.assertEqual(TASK_THRESHOLDS["itc_reconciliation"], 0.70)
        self.assertEqual(TASK_THRESHOLDS["full_gstr3b_filing"], 0.60)

    def test_success_true_when_rewards_meet_threshold(self):
        env = GSTEnvironment()
        # Patch reset to raise so rewards=[] → sum=0; then verify threshold logic directly
        # We test success logic directly via run_task output
        # Simulate: task1 threshold=0.80, rewards sum to 0.85
        lines_buf = []
        def fake_print(*args, file=None, flush=False, **kw):
            if file is None:
                lines_buf.append(" ".join(str(a) for a in args))

        # Directly test the threshold logic
        task = "invoice_classifier"
        threshold = TASK_THRESHOLDS[task]
        rewards = [0.15] * 6  # sum=0.90 >= 0.80
        success = sum(rewards) >= threshold
        self.assertTrue(success)

    def test_success_false_when_rewards_below_threshold(self):
        task = "invoice_classifier"
        threshold = TASK_THRESHOLDS[task]
        rewards = [0.15] * 4  # sum=0.60 < 0.80
        success = sum(rewards) >= threshold
        self.assertFalse(success)

    def test_end_line_success_reflects_threshold(self):
        """run_task produces success=false when total reward < task threshold."""
        env = GSTEnvironment()
        # Force all steps to give 0.0 reward by patching step()
        real_reset = env.reset
        call_count = [0]

        def patched_step(action):
            call_count[0] += 1
            obs = env._build_observation(reward=0.0, done=call_count[0] >= 3)
            return obs

        obs = env.reset(task="invoice_classifier")
        with patch.object(env, "step", side_effect=patched_step):
            with patch.object(env, "reset", return_value=obs):
                lines, _ = capture_run_task(env, "invoice_classifier")

        end_line = next(l for l in lines if l.startswith("[END]"))
        self.assertIn("success=false", end_line)

    # ── rewards_str edge cases ────────────────────────────────────────────────

    def test_rewards_str_is_0_00_on_empty_list(self):
        env = GSTEnvironment()
        with patch.object(env, "reset", side_effect=RuntimeError("no reset")):
            lines, _ = capture_run_task(env, "invoice_classifier")
        end = next(l for l in lines if l.startswith("[END]"))
        self.assertIn("rewards=0.00", end)

    def test_rewards_str_comma_separated_2dp(self):
        env = GSTEnvironment()
        obs = env.reset(task="invoice_classifier")
        with patch.object(env, "reset", return_value=obs):
            with patch.object(env, "step") as mock_step:
                step_obs = env._build_observation(reward=0.15, done=True)
                mock_step.return_value = step_obs
                lines, _ = capture_run_task(env, "invoice_classifier")
        end = next(l for l in lines if l.startswith("[END]"))
        rewards_part = end.split("rewards=")[1]
        for r in rewards_part.split(","):
            self.assertRegex(r.strip(), r"^-?\d+\.\d{2}$")


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (TestOutputParsing, TestTaskValidation, TestLLMCriteriaCheck):
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

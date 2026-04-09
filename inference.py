"""
GST Sahayak — Inference Script
Runs an LLM agent through all three GST tasks and emits required stdout lines.
Required stdout format:
  [START] task=<task> env=gst-sahayak model=<model>
  [STEP] step=<n> action=<json> reward=<X.XX> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import re
import json
import sys

from openai import OpenAI
from models import GSTAction

# --- Environment variables (hackathon requirements) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")

# Issue-1 fix: defer token validation to first use — importing this module does NOT crash
# Issue-10 fix: reject empty strings too, not just None
_openai_client = None

def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN environment variable is required and must not be empty")
        _openai_client = OpenAI(base_url=API_BASE_URL, api_key=token)
    return _openai_client


TASKS = ["invoice_classifier", "itc_reconciliation", "full_gstr3b_filing"]

# Issue-9 fix: per-task success thresholds from PRD Section 3
TASK_THRESHOLDS = {
    "invoice_classifier": 0.80,
    "itc_reconciliation": 0.70,
    "full_gstr3b_filing": 0.60,
}

# GSTR-3B payload template — used in prompt only when filing is next
_GSTR3B_TEMPLATE = (
    '{"outward_taxable_supplies":{"igst":0.0,"cgst":0.0,"sgst":0.0},'
    '"zero_rated_supplies":{"igst":0.0},'
    '"exempt_supplies":{"value":0.0},'
    '"itc_available":{"igst":0.0,"cgst":0.0,"sgst":0.0},'
    '"itc_reversed":{"igst":0.0,"cgst":0.0,"sgst":0.0},'
    '"net_tax_liability":{"igst":0.0,"cgst":0.0,"sgst":0.0}}'
)


# ---------------------------------------------------------------------------
# Observation → Prompt
# ---------------------------------------------------------------------------

def obs_to_prompt(obs: dict, task: str) -> str:
    """Build a focused prompt — only pending invoices to cut token waste."""
    pending = obs.get("pending_actions", [])

    # Issue-6 fix: send only invoices that are still pending, not the full batch
    all_invoices = obs.get("invoices", [])
    # pending may contain action-hint strings like "compute_liability" — filter those
    pending_ids = {p for p in pending if not p.startswith("ACTION:") and p not in
                   ("compute_liability", "file_return")}
    if pending_ids:
        invoices_to_show = [inv for inv in all_invoices
                            if inv.get("invoice_id") in pending_ids]
    else:
        invoices_to_show = []

    # Issue-4 fix: show full GSTR-3B schema only when the agent actually needs to file
    filing_hint = obs.get("filing_status", "")
    needs_filing = (
        task in ("itc_reconciliation", "full_gstr3b_filing")
        and "file_return" in pending
    )
    payload_section = (
        f"\nGSTR-3B payload template for file_return:\n{_GSTR3B_TEMPLATE}\n"
        if needs_filing else ""
    )

    gstr_2b = obs.get("gstr_2b", [])

    return f"""You are a GST compliance agent for Indian taxation.

Task: {task}
Step: {obs.get('step', 0)}/{obs.get('max_steps', 60)}
Episode context: {obs.get('episode_context', '')}
Accumulated ITC: ₹{obs.get('accumulated_itc', 0):.2f}
Tax liability so far: ₹{obs.get('tax_liability', 0):.2f}
Filing status: {filing_hint}
Next required actions: {pending}

Pending invoices to process:
{json.dumps(invoices_to_show, indent=2) if invoices_to_show else "(none — proceed to compute_liability or file_return)"}

GSTR-2B portal records:
{json.dumps(gstr_2b, indent=2)}

CLASSIFICATION RULES (classify_invoice):
- B2B: both supplier and buyer are GST-registered
- B2C: buyer is an individual / unregistered
- EXPORT: goods/services leaving India (zero-rated)
- EXEMPT: fresh food, healthcare, education (no GST)

RECONCILIATION RULES (ITC decisions):
- mismatch=false → accept_itc
- mismatch_reason in [missing_on_portal, gstin_mismatch, duplicate_invoice] → reject_itc
- mismatch_reason = value_mismatch → flag_for_review

HSN CODES: 8471=computers 8517=phones 6204=garments 3004=pharma
2106=food 9401=furniture 4901=books 8516=appliances 7208=steel 0101=livestock
{payload_section}
Respond ONLY with valid JSON (no markdown, no extra keys):
{{
  "action_type": "classify_invoice|accept_itc|reject_itc|flag_for_review|compute_liability|file_return",
  "invoice_id": "<id or null>",
  "hsn_code": "<4-digit HSN or null>",
  "invoice_type": "<B2B|B2C|EXPORT|EXEMPT|null>",
  "gstr_payload": <full GSTR-3B dict for file_return, else null>
}}"""


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------

def get_action(obs: dict, task: str) -> dict:
    """Call LLM and return parsed action dict. Returns safe fallback on any error."""
    pending = obs.get("pending_actions", [])
    first_pending = pending[0] if pending else None

    try:
        # Issue-7 fix: higher token budget for Task 3 where file_return JSON is large
        max_tok = 1500 if task == "full_gstr3b_filing" else 600

        response = _get_client().chat.completions.create(
            model=MODEL_NAME,
            max_tokens=max_tok,
            messages=[{"role": "user", "content": obs_to_prompt(obs, task)}],
        )
        content = response.choices[0].message.content.strip()

        # Issue-2 fix: strip markdown fences robustly using regex
        content = re.sub(r"^```[a-z]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()

        return json.loads(content)

    except Exception:
        # Issue-3 fix: handle action-hint tokens ("compute_liability", "file_return")
        # that appear in pending_actions for Task 3 when all invoices are processed
        if first_pending in ("compute_liability", "file_return"):
            return {
                "action_type": first_pending,
                "invoice_id": None,
                "hsn_code": None,
                "invoice_type": None,
                "gstr_payload": None,
            }
        # Safe fallback — flag_for_review never causes an audit penalty
        return {
            "action_type": "flag_for_review",
            "invoice_id": first_pending,  # None if no pending invoices
            "hsn_code": None,
            "invoice_type": None,
            "gstr_payload": None,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(env, task_name: str):
    print(f"[START] task={task_name} env=gst-sahayak model={MODEL_NAME}", flush=True)

    rewards = []
    step = 0

    try:
        obs = env.reset(task=task_name)
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
        done = False

        while not done:
            step += 1
            action_dict = get_action(obs_dict, task_name)
            action_str = json.dumps(action_dict, separators=(",", ":"))
            error = "null"
            reward = 0.0

            try:
                result = env.step(GSTAction(**action_dict))

                if isinstance(result, tuple):
                    obs, reward, done, info = result
                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
                    reward = round(float(reward), 2)
                    error = (info.get("error") or "null") if isinstance(info, dict) else "null"
                else:
                    obs = result
                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
                    reward = round(float(obs_dict.get("reward", 0.0)), 2)
                    done = bool(obs_dict.get("done", False))
                    last_err = obs_dict.get("last_error")
                    error = last_err if last_err else "null"

            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc).replace("\n", " ")[:200]

            rewards.append(reward)
            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} "
                f"done={str(done).lower()} error={error}",
                flush=True,
            )

    except Exception as exc:
        # Issue-8 fix: log reset() failures to stderr so they're visible in logs
        print(f"[ERROR] reset failed for task={task_name}: {exc}", file=sys.stderr, flush=True)
        # No [STEP] emitted — env.step() was never called

    finally:
        # Issue-9 fix: per-task success threshold from PRD
        threshold = TASK_THRESHOLDS.get(task_name, 0.5)
        success = sum(rewards) >= threshold
        # Issue-11 fix: emit "0.00" rather than empty string on reset failure
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from server.gst_environment import GSTEnvironment

    env = GSTEnvironment()
    try:
        for task in TASKS:
            run_task(env, task)
    finally:
        if hasattr(env, "close"):
            env.close()

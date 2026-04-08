"""
GST Sahayak — Inference Script
Runs an LLM agent through all three GST tasks and emits required stdout lines.
Required stdout format:
  [START] task=<task> env=gst-sahayak model=<model>
  [STEP] step=<n> action=<json> reward=<X.XX> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import json
import sys

from openai import OpenAI
from models import GSTAction

# --- Environment variables (hackathon requirements) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["invoice_classifier", "itc_reconciliation", "full_gstr3b_filing"]


# ---------------------------------------------------------------------------
# Observation → Prompt
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = {
    "invoice_classifier": """
YOUR JOB THIS EPISODE:
- Classify each invoice by type: B2B, B2C, EXPORT, or EXEMPT
- Assign the correct 4-digit HSN code
- Use action_type "classify_invoice" for every pending invoice

GST TYPE RULES:
- B2B: Buyer has GSTIN (business-to-business)
- B2C: Individual buyer, no GSTIN
- EXPORT: Goods/services leaving India (zero-rated)
- EXEMPT: Fresh food, healthcare, education — no GST

COMMON HSN CODES: 8471=computers, 8517=phones, 6204=garments, 3004=pharma,
2106=food, 9401=furniture, 4901=books, 8516=appliances, 7208=steel""",

    "itc_reconciliation": """
YOUR JOB THIS EPISODE:
- Compare your purchase register against GSTR-2B portal data
- accept_itc: invoice matches portal exactly (mismatch=false)
- reject_itc: clear mismatch — gstin_mismatch, duplicate_invoice, or missing_on_portal
- flag_for_review: minor value difference (safer than accepting risk)

DECISION RULES:
- mismatch=false → accept_itc
- mismatch_reason="missing_on_portal" → reject_itc
- mismatch_reason="gstin_mismatch" → reject_itc
- mismatch_reason="duplicate_invoice" → reject_itc
- mismatch_reason="value_mismatch" → flag_for_review (safe partial credit)""",

    "full_gstr3b_filing": """
YOUR JOB: Complete full GSTR-3B filing in sequence:
1. CLASSIFY all invoices (classify_invoice for each)
2. RECONCILE all ITC (accept_itc/reject_itc/flag_for_review for each)
3. COMPUTE liability (compute_liability — call once when done)
4. FILE the return (file_return with complete GSTR-3B payload)

GSTR-3B payload structure for file_return:
{
  "outward_taxable_supplies": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
  "zero_rated_supplies": {"igst": 0.0},
  "exempt_supplies": {"value": 0.0},
  "itc_available": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
  "itc_reversed": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
  "net_tax_liability": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0}
}
Fill values from the invoices you processed. Do NOT file before classifying and reconciling."""
}

ACTION_SCHEMA = """
Respond with ONLY valid JSON (no markdown fences, no explanation):
{
  "action_type": "<classify_invoice|accept_itc|reject_itc|flag_for_review|compute_liability|file_return>",
  "invoice_id": "<invoice ID or null>",
  "hsn_code": "<4-digit HSN or null>",
  "invoice_type": "<B2B|B2C|EXPORT|EXEMPT or null>",
  "gstr_payload": <GSTR-3B dict for file_return, or null>
}"""


def obs_to_prompt(obs, task: str) -> str:
    invoices = obs.invoices if hasattr(obs, "invoices") else obs.get("invoices", [])
    gstr_2b = obs.gstr_2b if hasattr(obs, "gstr_2b") else obs.get("gstr_2b", [])
    step = obs.step if hasattr(obs, "step") else obs.get("step", 0)
    max_steps = obs.max_steps if hasattr(obs, "max_steps") else obs.get("max_steps", 60)
    context = obs.episode_context if hasattr(obs, "episode_context") else obs.get("episode_context", "")
    itc = obs.accumulated_itc if hasattr(obs, "accumulated_itc") else obs.get("accumulated_itc", 0)
    liability = obs.tax_liability if hasattr(obs, "tax_liability") else obs.get("tax_liability", 0)
    status = obs.filing_status if hasattr(obs, "filing_status") else obs.get("filing_status", "")
    pending = obs.pending_actions if hasattr(obs, "pending_actions") else obs.get("pending_actions", [])

    return f"""You are an expert Indian GST compliance agent.

TASK: {task}
STEP: {step} of {max_steps}
EPISODE CONTEXT: {context}
ACCUMULATED ITC: ₹{itc:.2f}
TAX LIABILITY SO FAR: ₹{liability:.2f}
FILING STATUS: {status}
PENDING INVOICE IDs: {pending}
{TASK_INSTRUCTIONS.get(task, "")}

CURRENT INVOICES:
{json.dumps(invoices, indent=2)}

GSTR-2B PORTAL DATA:
{json.dumps(gstr_2b, indent=2)}
{ACTION_SCHEMA}"""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(llm_response: str) -> dict:
    """Parse LLM response to action dict. Returns safe fallback on failure."""
    try:
        clean = llm_response.strip()
        # Strip accidental markdown fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "action_type": "flag_for_review",
            "invoice_id": None,
            "hsn_code": None,
            "invoice_type": None,
            "gstr_payload": None,
        }


def get_llm_action(obs, task: str) -> dict:
    prompt = obs_to_prompt(obs, task)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_action(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, task: str):
    obs = env.reset(task=task)
    print(f"[START] task={task} env=gst-sahayak model={MODEL_NAME}", flush=True)

    rewards = []
    step = 0

    while True:
        step += 1
        action_dict = get_llm_action(obs, task)
        action_str = json.dumps(action_dict, separators=(",", ":"))

        error_val = "null"
        try:
            action = GSTAction(**action_dict)
            obs = env.step(action)
            reward = round(obs.reward, 2)
            done = obs.done
            if obs.last_error:
                error_val = obs.last_error
        except Exception as exc:
            reward = 0.0
            done = True
            error_val = str(exc).replace("\n", " ")

        rewards.append(reward)
        print(
            f"[STEP] step={step} action={action_str} reward={reward:.2f} "
            f"done={str(done).lower()} error={error_val}",
            flush=True,
        )

        if done:
            break

    success = sum(rewards) >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from server.gst_environment import GSTEnvironment

    env = GSTEnvironment()
    try:
        for task in TASKS:
            run_episode(env, task)
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=", flush=True)
        raise
    finally:
        if hasattr(env, "close"):
            env.close()
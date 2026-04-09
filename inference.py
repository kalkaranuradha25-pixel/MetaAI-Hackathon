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

from openai import OpenAI

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

def obs_to_prompt(obs: dict, task: str) -> str:
    """Convert GSTObservation dict to natural language prompt for the LLM."""
    return f"""You are a GST compliance agent for Indian taxation.

Task: {task}
Step: {obs.get('step', 0)}/{obs.get('max_steps', 60)}
Episode context: {obs.get('episode_context', '')}
Accumulated ITC: {obs.get('accumulated_itc', 0):.2f}
Tax liability so far: {obs.get('tax_liability', 0):.2f}
Filing status: {obs.get('filing_status', '')}
Pending invoice IDs: {obs.get('pending_actions', [])}

Current invoices:
{json.dumps(obs.get('invoices', []), indent=2)}

GSTR-2B mismatches:
{json.dumps(obs.get('gstr_2b', []), indent=2)}

CLASSIFICATION RULES (for classify_invoice):
- B2B: supplier and buyer are both GST-registered businesses
- B2C: buyer is an individual / unregistered
- EXPORT: goods or services leaving India (zero-rated, igst=0)
- EXEMPT: fresh food, healthcare, education (no GST at all)

RECONCILIATION RULES (for ITC decisions):
- mismatch=false -> accept_itc
- mismatch_reason missing_on_portal / gstin_mismatch / duplicate_invoice -> reject_itc
- mismatch_reason value_mismatch -> flag_for_review

COMMON HSN CODES: 8471=computers 8517=phones 6204=garments 3004=pharma
2106=food 9401=furniture 4901=books 8516=appliances 7208=steel 0101=livestock

GSTR-3B payload (only for file_return action):
{{"outward_taxable_supplies":{{"igst":0.0,"cgst":0.0,"sgst":0.0}},"zero_rated_supplies":{{"igst":0.0}},"exempt_supplies":{{"value":0.0}},"itc_available":{{"igst":0.0,"cgst":0.0,"sgst":0.0}},"itc_reversed":{{"igst":0.0,"cgst":0.0,"sgst":0.0}},"net_tax_liability":{{"igst":0.0,"cgst":0.0,"sgst":0.0}}}}

Choose ONE action. Respond ONLY with valid JSON matching this schema:
{{
  "action_type": "classify_invoice|accept_itc|reject_itc|flag_for_review|compute_liability|file_return",
  "invoice_id": "<id or null>",
  "hsn_code": "<hsn or null>",
  "invoice_type": "<B2B|B2C|EXPORT|EXEMPT|null>",
  "gstr_payload": null
}}"""


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------

def get_action(obs: dict, task: str) -> dict:
    """Call LLM and return parsed action dict. Returns safe fallback on any error."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=500,
            messages=[{"role": "user", "content": obs_to_prompt(obs, task)}]
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return json.loads(content)

    except Exception:
        # Safe fallback — flag_for_review never causes audit penalty
        pending = obs.get("pending_actions", [])
        return {
            "action_type": "flag_for_review",
            "invoice_id": pending[0] if pending else None,
            "hsn_code": None,
            "invoice_type": None,
            "gstr_payload": None,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(env, task_name: str):
    obs = env.reset(task=task_name)

    # Support both OpenEnv Observation object and plain dict
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

    print(f"[START] task={task_name} env=gst-sahayak model={MODEL_NAME}", flush=True)

    rewards = []
    step = 0
    done = False

    while not done:
        step += 1

        action_dict = get_action(obs_dict, task_name)
        action_str = json.dumps(action_dict, separators=(",", ":"))

        error = "null"
        reward = 0.0

        try:
            # Accept both (obs, reward, done, info) tuple and OpenEnv Observation object
            result = env.step(action_dict)

            if isinstance(result, tuple):
                obs, reward, done, info = result
                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
                reward = round(float(reward), 2)
                error = info.get("error") or "null" if isinstance(info, dict) else "null"
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

    success = sum(rewards) > 0.5
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
            run_task(env, task)
    finally:
        if hasattr(env, "close"):
            env.close()

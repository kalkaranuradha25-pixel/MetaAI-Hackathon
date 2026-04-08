# agent.md — GST Sahayak Agent Design
> This document covers how the LLM agent works inside `inference.py`: the prompt design, action parsing, the episode loop, and how to extend to a trained RL agent later.

---

## Agent Architecture Overview

The agent is the part that **decides what action to take** given an observation from the environment. For this hackathon prototype, the agent is an LLM (accessed via OpenAI client) that:

1. Receives a `GSTObservation` object from the environment
2. Converts it to a natural language prompt
3. Calls the LLM API
4. Parses the response as a `GSTAction` JSON
5. Sends it to `env.step(action)`
6. Reads `obs.reward` and `obs.done` from the returned observation
7. Repeats until `obs.done == True` or max steps reached

```
GSTObservation → [obs_to_prompt()] → LLM → [parse_action()] → GSTAction → env.step()
                                                                               ↓
                                                              GSTObservation (with reward, done)
```

---

## The Observation-to-Prompt Conversion

This is the most important design decision in the agent. The LLM sees the environment through this prompt. It needs to be:
- Complete (all information needed to make a correct decision)
- Structured (so the LLM can reliably parse it)
- Concise (fits within token limits given 2 vCPU / 8 GB constraint)

### obs_to_prompt() — Full Implementation

```python
import json

def obs_to_prompt(obs, task: str) -> str:
    invoices_text = json.dumps(obs.invoices, indent=2)
    gstr2b_text = json.dumps(obs.gstr_2b, indent=2)

    base_context = f"""You are an expert Indian GST compliance agent.
Your job is to file GST returns correctly and avoid audit penalties.

TASK: {task}
STEP: {obs.step} of {obs.max_steps}
EPISODE CONTEXT: {obs.episode_context}
ACCUMULATED ITC SO FAR: ₹{obs.accumulated_itc:.2f}
TAX LIABILITY SO FAR: ₹{obs.tax_liability:.2f}
FILING STATUS: {obs.filing_status}
PENDING INVOICE IDs: {obs.pending_actions}
"""

    task_instructions = {
        "invoice_classifier": """
YOUR JOB THIS EPISODE:
- Classify each invoice by type: B2B (business-to-business), B2C (business-to-consumer),
  EXPORT (zero-rated export), or EXEMPT (exempt from GST)
- Assign the correct HSN code based on the product/service description
- Use action_type: "classify_invoice" with invoice_id, invoice_type, and hsn_code
- Complete all pending invoices before max_steps

GST TYPE RULES:
- B2B: Buyer has GSTIN, both buyer/seller are GST registered
- B2C: Buyer is an individual/unregistered entity
- EXPORT: Goods/services leaving India, zero-rated
- EXEMPT: Items like fresh food, healthcare, education — no GST applies

COMMON HSN CODES:
- 8471: Computers and laptops
- 8517: Mobile phones
- 6204: Women's garments
- 3004: Pharmaceutical preparations
- 2106: Food preparations
- 9401: Seating furniture
- 4901: Books and printed matter
""",
        "itc_reconciliation": """
YOUR JOB THIS EPISODE:
- Review each invoice in your purchase register against the GSTR-2B portal data
- Decide: accept_itc (claim the credit), reject_itc (don't claim), or flag_for_review
- Mismatches between your records and GSTR-2B are risky — wrong claims trigger audits

DECISION RULES:
- accept_itc: Your invoice matches GSTR-2B data exactly (or within 5% tolerance)
- reject_itc: Clear mismatch — GSTIN wrong, duplicate invoice, or missing on portal
- flag_for_review: Uncertain — value differs slightly, need more info (gives partial credit)

MISMATCH TYPES TO WATCH FOR:
- value_mismatch: Portal shows different taxable value than your invoice
- duplicate_invoice: Same invoice number appears twice
- gstin_mismatch: Vendor GSTIN in your register differs from portal's records
- missing_on_portal: Your invoice not found in GSTR-2B at all (vendor hasn't filed)
""",
        "full_gstr3b_filing": """
YOUR JOB THIS EPISODE:
Complete the full GSTR-3B filing in order:
1. CLASSIFY all invoices first (use classify_invoice for each pending invoice)
2. RECONCILE all ITC claims (accept_itc / reject_itc / flag_for_review for each)
3. COMPUTE liability (use compute_liability when all invoices are processed)
4. FILE the return (use file_return with the complete GSTR-3B payload)

DO NOT skip steps or file before completing classification and reconciliation.
Filing too early or with wrong data triggers penalty (done=true but low reward).

GSTR-3B PAYLOAD STRUCTURE (for file_return action):
{
  "outward_taxable_supplies": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
  "zero_rated_supplies": {"igst": 0.0},
  "exempt_supplies": {"value": 0.0},
  "itc_available": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
  "itc_reversed": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
  "net_tax_liability": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0}
}
"""
    }

    action_schema = """
RESPOND WITH EXACTLY THIS JSON (no markdown, no explanation, just the JSON):
{
  "action_type": "<one of: classify_invoice | accept_itc | reject_itc | flag_for_review | compute_liability | file_return>",
  "invoice_id": "<invoice ID string, or null if not applicable>",
  "hsn_code": "<4-digit HSN code string, or null>",
  "invoice_type": "<B2B | B2C | EXPORT | EXEMPT, or null>",
  "gstr_payload": <GSTR-3B dict for file_return, or null>
}
"""

    current_data = f"""
CURRENT INVOICES:
{invoices_text}

GSTR-2B PORTAL DATA:
{gstr2b_text}
"""

    return base_context + task_instructions.get(task, "") + current_data + action_schema
```

---

## Action Parsing

The LLM response must be parsed into a `GSTAction`. Always wrap in try/except — LLMs occasionally output malformed JSON.

```python
import json
from models import GSTAction

def parse_action(llm_response: str) -> GSTAction:
    """Parse LLM response string into GSTAction. Returns fallback on failure."""
    try:
        # Strip any accidental markdown fences
        clean = llm_response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        
        data = json.loads(clean.strip())
        return GSTAction(**data)
    
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Fallback: flag the first pending invoice for review
        # This is safe — flag_for_review never causes audit penalty
        return GSTAction(
            action_type="flag_for_review",
            invoice_id=None,
            hsn_code=None,
            invoice_type=None,
            gstr_payload=None
        )
```

---

## The Full Episode Loop (inference.py)

```python
import os
import json
from openai import OpenAI
from models import GSTAction

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["invoice_classifier", "itc_reconciliation", "full_gstr3b_filing"]

def get_llm_action(obs, task: str) -> GSTAction:
    prompt = obs_to_prompt(obs, task)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_action(response.choices[0].message.content)


def run_episode(env, task: str):
    """Run one full episode for a given task. Emit required stdout lines."""
    obs = env.reset(task=task)
    print(f"[START] task={task} env=gst-sahayak model={MODEL_NAME}")

    rewards = []
    step = 0

    while True:
        step += 1
        action = get_llm_action(obs, task)
        action_str = action.model_dump_json()   # Pydantic v2

        try:
            obs = env.step(action)
            reward = round(obs.reward, 2)
            done = obs.done
            error = getattr(obs, "last_error", None) or "null"
        except Exception as e:
            reward = 0.0
            done = True
            error = str(e)

        rewards.append(reward)
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error}")

        if done:
            break

    success = sum(rewards) >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")


if __name__ == "__main__":
    from server.gst_environment import GSTEnvironment
    env = GSTEnvironment()
    try:
        for task in TASKS:
            run_episode(env, task)
    finally:
        env.close()
```

---

## How the Agent Selects Actions — Strategy Per Task

### Task 1: invoice_classifier
The LLM needs to pick the correct `invoice_type` and `hsn_code` for each invoice. Key signals in the observation:
- Buyer has GSTIN → likely B2B
- Buyer is an individual name (no GSTIN) → B2C
- Destination is outside India → EXPORT
- Product category (food, medicine, education) → EXEMPT

### Task 2: itc_reconciliation
The LLM compares `obs.invoices` (your purchase register) with `obs.gstr_2b` (portal data). Key signals:
- `mismatch: false` in GSTR-2B → safe to `accept_itc`
- `mismatch: true` with `mismatch_reason: "value_mismatch"` → consider `flag_for_review` (safe)
- `mismatch: true` with `mismatch_reason: "gstin_mismatch"` → `reject_itc` (risky to accept)
- `mismatch: true` with `mismatch_reason: "missing_on_portal"` → `reject_itc` (vendor hasn't filed)
- `mismatch: true` with `mismatch_reason: "duplicate_invoice"` → `reject_itc`

### Task 3: full_gstr3b_filing
The LLM must sequence actions correctly. Suggested order:
1. Process all invoices with `classify_invoice` until `pending_actions` is empty
2. Process all ITC decisions with `accept_itc` / `reject_itc` / `flag_for_review`
3. Call `compute_liability` to lock the numbers
4. Call `file_return` with the GSTR-3B payload

If the LLM calls `file_return` before completing steps 1–3, the environment returns `done=True` but with a heavily penalized reward.

---

## Extending to a Trained RL Agent (Post-Hackathon)

For the hackathon prototype, the agent is an LLM. Post-hackathon, replace the LLM with a trained RL policy:

```python
# Prototype (what you're building now)
action = get_llm_action(obs, task)

# Future (post-hackathon)
action = trained_policy.predict(obs.to_tensor())
```

The environment doesn't change at all — only the agent changes. This is the whole point of building the environment as an OpenEnv server: any agent can connect to it via the HTTP/WebSocket API.

### What RL Training Would Look Like

```python
# Using the OpenEnv client from client.py
from client import GSTEnvClient

async with GSTEnvClient(base_url="http://localhost:7860") as env:
    obs = await env.reset(task="invoice_classifier")
    
    while not obs.done:
        action = policy.select_action(obs)      # your trained model
        obs = await env.step(action)
        replay_buffer.add(obs.reward, obs.done)
    
    policy.update(replay_buffer)
```

The RL framework (GRPO, PPO, etc.) wraps around this client loop. The environment serves it over HTTP just like it serves the LLM agent.

---

## Token Budget Awareness

The observation prompt can get large for Task 3 (20 invoices + 6 mismatches). Approximate prompt sizes:

| Task | Approx prompt tokens | LLM response tokens |
|---|---|---|
| invoice_classifier | ~800 | ~100 |
| itc_reconciliation | ~1200 | ~80 |
| full_gstr3b_filing | ~2000 | ~150 |

These fit comfortably within `gpt-4.1-mini`'s context window. If using a smaller model, truncate older invoices from the prompt (already-actioned ones don't need to stay in context).

---

## Testing the Agent Without API Calls

Use a mock agent for local development to avoid burning API credits:

```python
def mock_agent(obs, task: str) -> GSTAction:
    """Returns a deterministic action for testing the env loop."""
    if obs.pending_actions:
        invoice_id = obs.pending_actions[0]
        if task == "invoice_classifier":
            return GSTAction(
                action_type="classify_invoice",
                invoice_id=invoice_id,
                invoice_type="B2B",
                hsn_code="8471"
            )
        elif task == "itc_reconciliation":
            return GSTAction(action_type="accept_itc", invoice_id=invoice_id)
    
    if task == "full_gstr3b_filing":
        return GSTAction(action_type="compute_liability")
    
    return GSTAction(action_type="flag_for_review")
```

Run this to verify the environment loop works before connecting a real LLM.

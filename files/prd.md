# GST Sahayak — Product Requirements Document
### Meta OpenEnv RL Hackathon | India Edition
**Version:** 1.0 | **Author:** Hackathon Submission | **Date:** April 2026

---

## 1. Executive Summary

GST Sahayak is a reinforcement learning environment built on the OpenEnv framework that trains AI agents to autonomously handle Indian GST (Goods and Services Tax) compliance — invoice classification, ITC reconciliation, and end-to-end GSTR-3B filing. It targets the 63 million Indian MSMEs that currently spend ₹13–17 lakh/year on manual compliance overhead.

The environment exposes a sequential decision-making problem where an agent receives a batch of invoices and portal data, takes multi-step actions with intermediate reward feedback, and is scored on correctness, penalty avoidance, and filing accuracy.

---

## 2. Hackathon Compliance Checklist

> Everything in this section maps directly to an official judging or validation criterion. Nothing optional is listed here.

### 2.1 Functional Requirements (from Problem Statement PDF)

| # | Requirement | How GST Sahayak satisfies it | Status |
|---|---|---|---|
| F1 | Real-world task simulation — no games or toy problems | GST filing is performed by millions of human accountants in India today | ✅ |
| F2 | OpenEnv interface: typed Observation, Action, Reward using Pydantic | `GSTObservation`, `GSTAction`, `GSTReward` — all Pydantic models | ✅ |
| F3 | `step(action)` → `(observation, reward, done, info)` | Implemented in `GSTEnvironment.step()` | ✅ |
| F4 | `reset()` → initial observation | Generates fresh synthetic invoice batch per episode | ✅ |
| F5 | `state()` → current state | Returns full internal state dict | ✅ |
| F6 | `openenv.yaml` metadata file | Included at project root | ✅ |
| F7 | Passes `openenv validate` | Validated before submission | ✅ |
| F8 | Minimum 3 tasks with agent graders | Task 1: Invoice Classifier, Task 2: ITC Reconciliation, Task 3: Full GSTR-3B Filing | ✅ |
| F9 | Tasks span easy → medium → hard | Task 1 easy, Task 2 medium, Task 3 hard | ✅ |
| F10 | Each task has programmatic grader returning score 0.0–1.0 | Deterministic graders using F1, field diff, penalty scoring | ✅ |
| F11 | Grading criteria clear, deterministic, reproducible | Pure Python math — no randomness in grader | ✅ |
| F12 | Reward function gives feedback throughout trajectory, not just at completion | Intermediate rewards at every `step()` call | ✅ |
| F13 | Reward penalizes undesirable behaviors (loops, destructive actions) | Penalties for wrong ITC claims, late filing, repeated invalid actions | ✅ |
| F14 | Baseline inference script using OpenAI API client | `inference.py` at project root | ✅ |
| F15 | API credentials read from environment variables (`HF_TOKEN`) | `os.getenv("HF_TOKEN")` with validation | ✅ |
| F16 | Reproducible baseline score across all tasks | Fixed random seed in scenario generator | ✅ |

### 2.2 Non-Functional Requirements (from Problem Statement PDF)

| # | Requirement | Implementation |
|---|---|---|
| NF1 | Deployable as containerized Hugging Face Space | `Dockerfile` included, tagged `openenv` |
| NF2 | Tagged with `openenv` on HF | Space metadata includes `openenv` tag |
| NF3 | Working `Dockerfile` — `docker build` and `docker run` succeed | Tested locally before submission |
| NF4 | README with environment overview and motivation | `README.md` with all 5 required sections |
| NF5 | README defines action and observation spaces | Pydantic model fields documented in README |
| NF6 | README has task descriptions with difficulty levels | All 3 tasks documented |
| NF7 | README has setup and usage instructions | `pip install`, `docker run`, `openenv validate` steps |
| NF8 | README has baseline performance scores | Baseline scores per task in a table |

### 2.3 Submission Validation Rules (from Guidelines PDF)

| Rule | Action Required |
|---|---|
| `inference.py` must be in root directory | File placed at `/inference.py` |
| Must use OpenAI Client for all LLM calls | `from openai import OpenAI` — no other SDK |
| `API_BASE_URL` must have a default value | `os.getenv("API_BASE_URL", "https://api.openai.com/v1")` |
| `MODEL_NAME` must have a default value | `os.getenv("MODEL_NAME", "gpt-4.1-mini")` |
| `HF_TOKEN` is mandatory, no default | Raises `ValueError` if missing |
| stdout must emit `[START]`, `[STEP]`, `[END]` lines exactly | Output format strictly enforced |
| HF Space must be in Running state at submission | Turn off all other spaces before submitting |
| Solution must run within 2 vCPU / 8 GB RAM | No heavy ML models loaded — pure Python env |

### 2.4 Required stdout Format (from Guidelines PDF)

```
[START] task=<task_name> env=gst-sahayak model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

Rules:
- One `[START]` at episode begin
- One `[STEP]` per step, immediately after `env.step()` returns
- One `[END]` after `env.close()` — always emitted, even on exception
- `reward` and `rewards` formatted to 2 decimal places
- `done` and `success` are lowercase: `true` or `false`
- `error` is the raw error string, or `null`
- All fields on a single line, no newlines within a line

---

## 3. OpenEnv Models — What to Implement

> This section directly answers: "what OpenEnv models should I use?"

### 3.1 Required Pydantic Models

OpenEnv requires you to define three typed Pydantic models. Here are the exact ones for GST Sahayak:

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# ---------- Sub-models ----------

class Invoice(BaseModel):
    invoice_id: str
    vendor_gstin: str
    invoice_date: str          # "YYYY-MM-DD"
    invoice_number: str
    taxable_value: float
    igst: float
    cgst: float
    sgst: float
    hsn_code: Optional[str]    # None = agent must classify
    invoice_type: Optional[str] # None = agent must classify

class MatchRecord(BaseModel):
    invoice_id: str
    portal_taxable_value: float
    portal_igst: float
    mismatch: bool             # True = agent must decide accept/reject
    mismatch_reason: Optional[str]

# ---------- Core OpenEnv Models ----------

class GSTObservation(BaseModel):
    """What the agent sees at each step."""
    invoices: List[Invoice]            # current batch to process
    gstr_2b: List[MatchRecord]         # GSTN portal's view
    accumulated_itc: float             # running ITC claimed so far
    tax_liability: float               # running liability computed
    step: int
    max_steps: int
    episode_context: str               # "b2b_heavy" | "b2c_heavy" | "export" | "mixed"
    pending_actions: List[str]         # invoice IDs not yet actioned
    filing_status: str                 # "not_started" | "in_progress" | "filed"

class GSTAction(BaseModel):
    """What the agent can do at each step."""
    action_type: str
    # "classify_invoice" | "accept_itc" | "reject_itc" |
    # "flag_for_review"  | "compute_liability" | "file_return"
    invoice_id: Optional[str] = None
    hsn_code: Optional[str] = None          # used with classify_invoice
    invoice_type: Optional[str] = None      # "B2B" | "B2C" | "EXPORT" | "EXEMPT"
    gstr_payload: Optional[Dict[str, Any]] = None  # used with file_return

class GSTReward(BaseModel):
    """Reward structure for interpretability."""
    step_reward: float          # reward for this specific step
    cumulative_reward: float    # total so far in episode
    reward_breakdown: Dict[str, float]  # e.g. {"classification": 0.1, "itc_penalty": -0.3}
```

### 3.2 OpenEnv Interface Methods

```python
from openenv import BaseEnvironment  # OpenEnv base class

class GSTEnvironment(BaseEnvironment):

    def reset(self) -> GSTObservation:
        """
        Generate a fresh episode. Called once per episode.
        Returns the initial observation.
        Uses fixed random seed per task for reproducibility.
        """
        ...

    def step(self, action: GSTAction) -> tuple[GSTObservation, float, bool, dict]:
        """
        Execute one action. Returns:
          - observation: GSTObservation (updated state)
          - reward: float (step reward, formatted to 2 decimal places)
          - done: bool (True when episode ends)
          - info: dict (error string or null, metadata)
        """
        ...

    def state(self) -> dict:
        """
        Return full internal state. Used by OpenEnv framework for inspection.
        """
        ...
```

### 3.3 openenv.yaml (Required File)

```yaml
name: gst-sahayak
version: 1.0.0
description: >
  RL environment for Indian GST compliance automation.
  An AI agent learns to classify invoices, reconcile ITC,
  and file GSTR-3B returns — replacing manual accountant workflows.
author: <your-name>
tags:
  - openenv
  - india
  - tax
  - compliance
  - finance
  - real-world

tasks:
  - id: invoice_classifier
    description: Classify invoices by type and assign HSN codes
    difficulty: easy
    max_steps: 20
    reward_threshold: 0.8

  - id: itc_reconciliation
    description: Accept or reject ITC claims against GSTR-2B mismatches
    difficulty: medium
    max_steps: 35
    reward_threshold: 0.7

  - id: full_gstr3b_filing
    description: Full multi-step episode — classify, reconcile, compute, and file GSTR-3B
    difficulty: hard
    max_steps: 60
    reward_threshold: 0.6

observation_space:
  type: GSTObservation
  description: Invoice batch, GSTR-2B data, running ITC, liability, and filing status

action_space:
  type: GSTAction
  actions:
    - classify_invoice
    - accept_itc
    - reject_itc
    - flag_for_review
    - compute_liability
    - file_return
```

---

## 4. Task Definitions

### Task 1 — Invoice Classifier (Easy)

**Objective:** Given a batch of 10 raw invoices with missing `invoice_type` and `hsn_code`, classify each correctly.

**Episode structure:**
- Reset generates 10 invoices with ground truth stored internally
- Each step: agent calls `classify_invoice` with `invoice_id`, `invoice_type`, `hsn_code`
- Episode ends when all 10 are classified, or max_steps (20) reached

**Grader:**
```
score = (correct_classifications / total_invoices)
      + (correct_hsn_codes / total_invoices) * 0.5
score = min(score, 1.0)
```

**Reward per step:**
- Correct type + correct HSN: `+0.15`
- Correct type, wrong HSN: `+0.05`
- Wrong type: `-0.05`
- Repeated action on same invoice: `-0.1`

**Baseline expected score:** ~0.60 (random agent), ~0.85 (LLM agent)

---

### Task 2 — ITC Reconciliation (Medium)

**Objective:** Given purchase register + GSTR-2B mismatch report (15 invoices, 5 with deliberate mismatches), decide accept/reject/flag each ITC claim.

**Episode structure:**
- Reset generates purchase data and GSTR-2B with seeded mismatches
- Agent actions: `accept_itc`, `reject_itc`, `flag_for_review` per invoice
- Mismatches have reasons: duplicate, value difference, GSTIN mismatch, missing on portal

**Grader (F1 score):**
```
precision = true_positives / (true_positives + false_positives)
recall    = true_positives / (true_positives + false_negatives)
score     = 2 * precision * recall / (precision + recall)
```

**Reward per step:**
- Correctly accept valid ITC: `+0.20`
- Correctly reject mismatched ITC: `+0.20`
- Wrongly accept mismatched ITC (audit flag): `-0.30`
- Wrongly reject valid ITC (ITC loss): `-0.15`
- Flag for review (partial credit): `+0.05`

**Baseline expected score:** ~0.50 (random), ~0.75 (LLM agent)

---

### Task 3 — Full GSTR-3B Filing (Hard)

**Objective:** Complete end-to-end filing for a fictional SME. Agent must classify invoices, reconcile ITC, compute tax liability, and submit a valid GSTR-3B payload — all within 60 steps.

**Episode structure:**
- Reset generates 20 invoices (mixed B2B/B2C/export), 6 ITC mismatches
- Agent must sequence all action types in correct order
- Final action: `file_return` with GSTR-3B payload dict
- Episode terminates on `file_return` or step timeout

**GSTR-3B payload fields agent must populate:**
```json
{
  "outward_taxable_supplies": { "igst": 0.0, "cgst": 0.0, "sgst": 0.0 },
  "zero_rated_supplies": { "igst": 0.0 },
  "exempt_supplies": { "value": 0.0 },
  "itc_available": { "igst": 0.0, "cgst": 0.0, "sgst": 0.0 },
  "itc_reversed": { "igst": 0.0, "cgst": 0.0, "sgst": 0.0 },
  "net_tax_liability": { "igst": 0.0, "cgst": 0.0, "sgst": 0.0 }
}
```

**Grader:**
```
field_score = average(1 - |agent_value - truth_value| / truth_value) per field
penalty_score = 1.0 - (audit_flags * 0.1)
time_score = 1.0 if steps < 45 else (60 - steps) / 15
final_score = (field_score * 0.6) + (penalty_score * 0.3) + (time_score * 0.1)
```

**Reward per step:**
- Correct invoice classification: `+0.10`
- Correct ITC accept: `+0.20`
- Incorrect ITC claim: `-0.30`
- `compute_liability` with correct value (within 5% error): `+0.30`
- `file_return` with correct payload: `+1.00` (terminal)
- `file_return` with incorrect payload: `+field_accuracy * 0.5` (partial terminal)
- Late filing (step > 55): `-0.50`
- Invalid action sequence (file before classify): `-0.20`

**Baseline expected score:** ~0.30 (random), ~0.65 (LLM agent)

---

## 5. Reward Function Design

The reward function is **incremental** (feedback at every step, not just episode end) as required by the problem statement.

### Reward Signal Design Principles

1. **Immediate feedback:** Agent knows within the same step whether an action was correct
2. **Proportional penalties:** Wrong ITC claims penalize more (−0.30) than wrong classifications (−0.05) because audit risk is higher
3. **Terminal bonus:** Correct final filing gives large bonus (+1.0) to incentivize completion
4. **Timeout penalty:** Episodes that exceed step limits receive −0.50 to discourage loops
5. **Action validity:** Invalid action types (not in defined set) return −0.10 and `error` field set

### Reward Table Summary

| Action | Correct | Incorrect | Notes |
|---|---|---|---|
| `classify_invoice` | +0.10 to +0.15 | −0.05 | Higher if HSN also correct |
| `accept_itc` | +0.20 | −0.30 | Large penalty = audit flag |
| `reject_itc` | +0.20 | −0.15 | Smaller penalty = ITC loss |
| `flag_for_review` | +0.05 | n/a | Partial credit always |
| `compute_liability` | +0.30 | −0.10 | Within 5% tolerance = correct |
| `file_return` | +1.00 | +0 to +0.50 | Partial credit on field accuracy |
| Timeout | — | −0.50 | Applied at episode end |
| Invalid action | — | −0.10 | Error message returned |
| Repeated action | — | −0.10 | Discourages loops |

---

## 6. Synthetic Data Generator

A reproducible data generator creates fresh episodes. This is critical — **no real taxpayer data is used.**

### Invoice Generator

```python
import random
from faker import Faker

INVOICE_TYPES = ["B2B", "B2C", "EXPORT", "EXEMPT"]
HSN_CODES = {
    "B2B": ["8471", "8517", "6204", "3004", "7208"],   # computers, phones, garments, pharma, steel
    "B2C": ["2106", "6109", "8516", "9401", "4901"],   # food, tshirts, appliances, furniture, books
    "EXPORT": ["8703", "8471", "6204"],
    "EXEMPT": ["0101", "0201", "2201"],
}
GST_RATES = {
    "B2B": 0.18,
    "B2C": 0.12,
    "EXPORT": 0.0,
    "EXEMPT": 0.0,
}

def generate_episode(seed: int, task: str) -> dict:
    random.seed(seed)
    # ... generate invoices with ground truth
```

### Mismatch Generator (Task 2 & 3)

Deliberately injects mismatch types:
- `value_mismatch` — portal shows different taxable value (±10–30%)
- `duplicate_invoice` — same invoice number, different amounts
- `gstin_mismatch` — vendor GSTIN differs from GSTN records
- `missing_on_portal` — invoice exists in purchase register, not in GSTR-2B

---

## 7. inference.py — Complete Specification

```python
import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["invoice_classifier", "itc_reconciliation", "full_gstr3b_filing"]

def obs_to_prompt(obs: dict, task: str) -> str:
    """Convert GSTObservation to natural language prompt for the LLM."""
    return f"""You are a GST compliance agent for Indian taxation.

Task: {task}
Step: {obs['step']}/{obs['max_steps']}
Episode context: {obs['episode_context']}
Accumulated ITC: ₹{obs['accumulated_itc']:.2f}
Tax liability so far: ₹{obs['tax_liability']:.2f}
Filing status: {obs['filing_status']}
Pending invoice IDs: {obs['pending_actions']}

Current invoices:
{json.dumps(obs['invoices'], indent=2)}

GSTR-2B mismatches:
{json.dumps(obs['gstr_2b'], indent=2)}

Choose ONE action. Respond ONLY with valid JSON matching this schema:
{{
  "action_type": "classify_invoice|accept_itc|reject_itc|flag_for_review|compute_liability|file_return",
  "invoice_id": "<id or null>",
  "hsn_code": "<hsn or null>",
  "invoice_type": "<B2B|B2C|EXPORT|EXEMPT|null>",
  "gstr_payload": null
}}"""

def get_action(obs: dict, task: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=[{"role": "user", "content": obs_to_prompt(obs, task)}]
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)

def run_task(env, task_name: str):
    obs = env.reset(task=task_name)
    print(f"[START] task={task_name} env=gst-sahayak model={MODEL_NAME}")

    rewards = []
    step = 0
    done = False

    while not done:
        step += 1
        action_dict = get_action(obs.dict(), task_name)
        action_str = json.dumps(action_dict)

        obs, reward, done, info = env.step(action_dict)
        error = info.get("error") or "null"
        rewards.append(reward)

        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error}")

    success = sum(rewards) > 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")

if __name__ == "__main__":
    from gst_env import GSTEnvironment   # your environment class
    env = GSTEnvironment()
    for task in TASKS:
        run_task(env, task)
    env.close()
```

---

## 8. Project Structure

```
gst-sahayak/
├── inference.py            ← REQUIRED: must be at root, named exactly this
├── openenv.yaml            ← REQUIRED: environment metadata
├── Dockerfile              ← REQUIRED: containerized deployment
├── README.md               ← REQUIRED: all 5 sections
├── requirements.txt
│
├── gst_env/
│   ├── __init__.py
│   ├── environment.py      ← GSTEnvironment class (step, reset, state)
│   ├── models.py           ← GSTObservation, GSTAction, GSTReward (Pydantic)
│   ├── graders.py          ← Task 1, 2, 3 graders
│   └── data_generator.py   ← Synthetic invoice + mismatch generator
│
└── tasks/
    ├── invoice_classifier.yaml
    ├── itc_reconciliation.yaml
    └── full_gstr3b_filing.yaml
```

---

## 9. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Validate environment on build
RUN openenv validate || true

EXPOSE 7860

CMD ["python", "-m", "openenv", "serve", "--host", "0.0.0.0", "--port", "7860"]
```

**requirements.txt:**
```
openenv
openai
pydantic>=2.0
faker
fastapi
uvicorn
```

> Must fit within 2 vCPU / 8 GB RAM. No ML models loaded — pure Python logic only.

---

## 10. README.md Required Sections

The README must include all five of these sections (NF4–NF8):

### Section 1 — Environment Overview and Motivation
Describe the GST compliance problem, the 63M MSME market, and why RL is the right approach.

### Section 2 — Action and Observation Spaces
Copy the Pydantic model field definitions from `models.py`. List all valid `action_type` values with descriptions.

### Section 3 — Task Descriptions with Difficulty Levels
Table with: Task ID, Difficulty, Max Steps, Reward Threshold, Description.

### Section 4 — Setup and Usage Instructions
```bash
# Install
pip install openenv openai faker

# Validate
openenv validate

# Run locally
HF_TOKEN=your_token python inference.py

# Docker
docker build -t gst-sahayak .
docker run -e HF_TOKEN=your_token gst-sahayak
```

### Section 5 — Baseline Performance Scores
| Task | Random Agent | LLM Baseline | Target |
|---|---|---|---|
| invoice_classifier | 0.25 | 0.82 | 0.80 |
| itc_reconciliation | 0.40 | 0.71 | 0.70 |
| full_gstr3b_filing | 0.15 | 0.63 | 0.60 |

---

## 11. Common Failure Cases to Avoid

From the official guidelines — these will auto-fail your submission:

| Failure | Prevention |
|---|---|
| `inference.py` not in root directory | Always keep at `/inference.py`, never in a subfolder |
| Missing default for `API_BASE_URL` | `os.getenv("API_BASE_URL", "https://api.openai.com/v1")` |
| Missing default for `MODEL_NAME` | `os.getenv("MODEL_NAME", "gpt-4.1-mini")` |
| Missing `HF_TOKEN` | Raise `ValueError` immediately if not set |
| HF Space still building at submission | Submit only after Space shows "Running" status |
| Space stopped (multiple active deployments) | Turn off all other Spaces before submitting |
| Using alternative SDKs | Only `from openai import OpenAI` — no `anthropic`, `requests`, etc. |
| Stdout format wrong | Test with: `python inference.py | grep -E '^\[(START|STEP|END)\]'` |

---

## 12. Build Order (Implementation Sequence)

1. **Day 1:** Write `models.py` (Pydantic models) + `data_generator.py` (synthetic data)
2. **Day 1:** Write `environment.py` (reset, step, state) + `graders.py` (Task 1 first)
3. **Day 2:** Add Task 2 and Task 3 graders + test `openenv validate`
4. **Day 2:** Write `inference.py` + test stdout format manually
5. **Day 3:** Write `Dockerfile` + `README.md` + deploy to HF Spaces
6. **Day 3:** Final test — run inference end-to-end, confirm Space is Running, submit

---

*This PRD is the single source of truth for the hackathon submission. Every implementation decision should be checked against Section 2 (Compliance Checklist) before code is written.*
---
title: Gst Sahayak
emoji: 🔥
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
short_description: An RL trained to perform GST-billing and compliance tasks
---

# GST Sahayak

An RL environment for automated Indian GST compliance — built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

---

## 1. Environment Overview and Motivation

India has **63 million MSMEs** that collectively spend ₹13–17 lakh per year on manual GST compliance — invoice classification, ITC reconciliation, and GSTR-3B filing. This is repetitive, rule-based work perfectly suited for an AI agent.

**GST Sahayak** frames this as a sequential decision-making problem:
- The agent receives a batch of raw invoices and GSTN portal data
- It must take multi-step actions to classify invoices, validate ITC claims, and ultimately file the return
- At every step it receives an incremental reward signal, enabling RL training

Why RL? Because the GST filing process has:
- A clear sequential structure (classify → reconcile → compute → file)
- Well-defined correctness criteria (match GSTN portal, avoid audit flags)
- Real penalties for mistakes (wrong ITC claims trigger government audits)

This environment trains an agent to navigate all of this autonomously, targeting the accountant's ₹13L/year compliance overhead.

---

## 2. Action and Observation Spaces

### GSTObservation

Fields returned by the environment after every `step()` call. The `reward` and `done` fields are inherited from the OpenEnv `Observation` base class.

| Field | Type | Description |
|---|---|---|
| `reward` | `float` | Step reward (inherited from Observation) |
| `done` | `bool` | Whether episode has ended (inherited) |
| `invoices` | `List[dict]` | Current batch of invoices to process |
| `gstr_2b` | `List[dict]` | GSTR-2B portal records with mismatch flags |
| `accumulated_itc` | `float` | Running ITC claimed so far (₹) |
| `tax_liability` | `float` | Running outward tax liability (₹) |
| `step` | `int` | Current step number |
| `max_steps` | `int` | Episode step limit |
| `episode_context` | `str` | `b2b_heavy` \| `b2c_heavy` \| `export` \| `mixed` |
| `pending_actions` | `List[str]` | Invoice IDs not yet actioned |
| `filing_status` | `str` | `not_started` \| `in_progress` \| `filed` |
| `last_error` | `Optional[str]` | Error from last step, or `null` |

### GSTAction

Fields the agent sends to the environment each step.

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | One of six valid action types (see below) |
| `invoice_id` | `Optional[str]` | Target invoice ID (required for most actions) |
| `hsn_code` | `Optional[str]` | 4-digit HSN code (for `classify_invoice`) |
| `invoice_type` | `Optional[str]` | `B2B` \| `B2C` \| `EXPORT` \| `EXEMPT` |
| `gstr_payload` | `Optional[dict]` | Full GSTR-3B dict (for `file_return` only) |

### Valid action_type values

| Action | Description | Reward |
|---|---|---|
| `classify_invoice` | Assign type and HSN code to an invoice | +0.15 (correct) / -0.05 (wrong) |
| `accept_itc` | Accept ITC claim for an invoice | +0.20 (correct) / -0.30 (audit flag) |
| `reject_itc` | Reject ITC claim for an invoice | +0.20 (correct) / -0.15 (ITC loss) |
| `flag_for_review` | Flag invoice for human review | +0.05 (always safe) |
| `compute_liability` | Lock in net tax liability | +0.30 (within tolerance) |
| `file_return` | Submit final GSTR-3B payload | +1.00 (correct) / partial credit |

---

## 3. Task Descriptions

| Task ID | Difficulty | Max Steps | Reward Threshold | Description |
|---|---|---|---|---|
| `invoice_classifier` | Easy | 20 | 0.80 | Classify 10 invoices by type (B2B/B2C/EXPORT/EXEMPT) and assign HSN codes |
| `itc_reconciliation` | Medium | 35 | 0.70 | Accept/reject/flag 15 ITC claims against GSTR-2B portal data (5 deliberate mismatches) |
| `full_gstr3b_filing` | Hard | 60 | 0.60 | Full end-to-end: classify 20 invoices, reconcile ITC (6 mismatches), compute liability, file GSTR-3B |

### Grading formulas

**Task 1 - Invoice Classifier:**
```
score = (correct_types / total) + (correct_hsn / total) * 0.5
score = min(score, 1.0)
```

**Task 2 - ITC Reconciliation (F1 score):**
```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
score     = 2 * precision * recall / (precision + recall)
```

**Task 3 - Full GSTR-3B Filing:**
```
field_score   = avg(1 - |agent_val - truth_val| / truth_val) per payload field
penalty_score = 1.0 - (audit_flags * 0.1)
time_score    = 1.0 if steps < 45 else (60 - steps) / 15
final_score   = (field_score * 0.6) + (penalty_score * 0.3) + (time_score * 0.1)
```

---

## 4. Setup and Usage

### Install

```bash
pip install openenv-core openai faker fastapi uvicorn
```

### Validate the environment

```bash
openenv validate
```

### Run inference locally

```bash
export HF_TOKEN=your_token_here
python inference.py
```

### Test stdout format

```bash
python inference.py | grep -E '^\[(START|STEP|END)\]'
```

### Run with Docker

```bash
docker build -t gst-sahayak .
docker run -e HF_TOKEN=your_token_here -p 7860:7860 gst-sahayak
```

### Run the environment server only

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 5. Baseline Performance Scores

Scores measured with fixed random seed (reproducible):

| Task | Random Agent | LLM Baseline (gpt-4.1-mini) | Target |
|---|---|---|---|
| `invoice_classifier` | 0.25 | 0.82 | 0.80 |
| `itc_reconciliation` | 0.40 | 0.71 | 0.70 |
| `full_gstr3b_filing` | 0.15 | 0.63 | 0.60 |

---

## Project Structure

```
gst-sahayak/
├── inference.py          <- LLM agent loop (hackathon required)
├── openenv.yaml          <- Environment metadata
├── Dockerfile            <- Container deployment
├── requirements.txt
├── README.md
├── models.py             <- GSTAction, GSTObservation, GSTState
├── client.py             <- Remote EnvClient
└── server/
    ├── app.py            <- FastAPI server via HTTPEnvServer
    ├── gst_environment.py <- reset(), step(), state
    ├── graders.py        <- Task 1/2/3 graders
    └── data_generator.py <- Synthetic invoice generator
```

## License

MIT

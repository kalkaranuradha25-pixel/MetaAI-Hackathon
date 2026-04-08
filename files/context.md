# context.md — GST Sahayak Project Context
> Load this file at the start of every coding session. It tells you what the project is, what phase it is in, and what decisions have already been locked in.

---

## What This Project Is

**GST Sahayak** is a reinforcement learning environment built on Meta's OpenEnv framework. It trains an LLM-based agent to autonomously handle Indian GST (Goods and Services Tax) compliance tasks — invoice classification, ITC reconciliation, and GSTR-3B filing.

**Hackathon:** Meta PyTorch OpenEnv India Hackathon
**Round 1 deadline:** April 8, 2026 (online submission)
**Finale:** April 25–26, Bangalore (48-hour in-person)
**Prize pool:** $30,000

---

## Current Phase: Hackathon Prototype

- Goal is a **working, deployable RL environment** on Hugging Face Spaces
- NOT a production product — no real taxpayer data, no live GSTN API calls
- All data is **synthetic and seeded** (deterministic, reproducible)
- Primary focus: correct OpenEnv API usage + meaningful incremental RL reward signal
- Turn into a full product only AFTER the hackathon

---

## Critical: The Correct OpenEnv Package

There are two packages named `openenv` on PyPI. This project uses the **meta-pytorch one**.

```bash
# CORRECT — meta-pytorch framework
pip install openenv-core

# WRONG — this is a different RL library by Jianfeng Hou
pip install openenv
```

The meta-pytorch package lives at: `https://github.com/meta-pytorch/OpenEnv`
DeepWiki docs: `https://deepwiki.com/meta-pytorch/OpenEnv`

---

## The Real OpenEnv API

### Base Classes to Import

```python
from openenv.core.env_server.interfaces import Environment   # server base class
from openenv.core.env_server.types import Action, Observation, State  # model bases
from openenv.core import HTTPEnvServer                       # serves the env over HTTP/WS
```

### Step() Returns an Observation Object — NOT a 4-tuple

This is the biggest difference from Gymnasium. OpenEnv's `step()` returns a single `Observation` object. `reward` and `done` are **fields on the Observation**, not separate return values.

```python
# Gymnasium style — WRONG for OpenEnv
obs, reward, done, info = env.step(action)

# OpenEnv style — CORRECT
obs = env.step(action)
print(obs.reward)   # reward is a field
print(obs.done)     # done is a field
```

### Three Required Pydantic Models

Every environment defines exactly three models in `models.py`, each inheriting from an OpenEnv base:

```python
from openenv.core.env_server.types import Action, Observation, State

class GSTAction(Action):
    # Agent sends this TO the environment each step
    # Action base: forbids extra fields (model_config)
    action_type: str
    invoice_id: Optional[str] = None
    hsn_code: Optional[str] = None
    invoice_type: Optional[str] = None
    gstr_payload: Optional[dict] = None

class GSTObservation(Observation):
    # Environment returns this AFTER each step
    # Observation base already includes: reward: float, done: bool
    # Add your env-specific fields below:
    invoices: List[dict]
    gstr_2b: List[dict]
    accumulated_itc: float
    tax_liability: float
    step: int
    max_steps: int
    episode_context: str
    pending_actions: List[str]
    filing_status: str

class GSTState(State):
    # Internal state for debugging, checkpointing, and the OpenEnv web UI
    # State base already includes: episode_id: str, step_count: int
    # Add your env-specific tracking below:
    ground_truth_invoices: List[dict]
    ground_truth_itc: List[dict]
    classified_so_far: dict
    itc_decisions: dict
    cumulative_reward: float
    task: str
```

### The Environment Class

```python
from openenv.core.env_server.interfaces import Environment

class GSTEnvironment(Environment[GSTAction, GSTObservation, GSTState]):

    supports_concurrent_sessions = True  # set False if state is not thread-safe

    def reset(self) -> GSTObservation:
        """Called once per episode. Generate fresh synthetic data, return initial obs."""
        ...

    def step(self, action: GSTAction) -> GSTObservation:
        """
        Process one action. Update internal state. Return observation.
        The observation MUST have reward and done set correctly.
        """
        ...

    @property
    def state(self) -> GSTState:
        """Return current internal state. Used by OpenEnv for inspection/debugging."""
        ...
```

### The FastAPI Server

```python
# server/app.py
from openenv.core import HTTPEnvServer
from .gst_environment import GSTEnvironment

env = GSTEnvironment()
server = HTTPEnvServer(env)
app = server.app      # standard FastAPI app
# Run with: uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Reward Transforms (optional advanced feature)

OpenEnv has a `CompositeTransform` system for chaining reward functions. For the prototype, compute reward directly in `step()`. Use transforms only if you need to layer multiple rubrics.

---

## Project Directory Structure

```
gst-sahayak/
├── inference.py              ← MUST be at root (hackathon rule)
├── openenv.yaml              ← MUST be at root (OpenEnv rule)
├── README.md                 ← MUST have 5 sections (hackathon rule)
├── pyproject.toml
├── requirements.txt
│
├── models.py                 ← GSTAction, GSTObservation, GSTState
├── client.py                 ← EnvClient subclass (for agents connecting remotely)
│
└── server/
    ├── __init__.py
    ├── app.py                ← FastAPI app via HTTPEnvServer
    ├── gst_environment.py    ← GSTEnvironment class
    ├── graders.py            ← Task graders, each returns float 0.0–1.0
    ├── data_generator.py     ← Synthetic invoice + mismatch data
    └── Dockerfile
```

---

## The Three Tasks

| Task ID | Difficulty | Max Steps | Agent's Job |
|---|---|---|---|
| `invoice_classifier` | Easy | 20 | Classify 10 invoices: type (B2B/B2C/EXPORT/EXEMPT) + HSN code |
| `itc_reconciliation` | Medium | 35 | Accept / reject / flag 15 ITC claims against GSTR-2B portal data |
| `full_gstr3b_filing` | Hard | 60 | Full episode: classify → reconcile → compute liability → file GSTR-3B |

Tasks are selected by passing `task` parameter to `reset()`.

---

## Reward Design Principles

1. **Incremental** — reward at every `step()`, not only at episode end (required by problem statement)
2. **Asymmetric penalties** — wrong ITC claims (−0.30) hurt more than wrong classifications (−0.05) because audit risk is higher
3. **Terminal bonus** — correct full filing gives +1.0 to incentivize completion
4. **Loop deterrent** — repeated action on same invoice: −0.10
5. **Timeout** — episode exceeding max_steps: −0.50

---

## Synthetic Data Rules

- No real taxpayer data anywhere in codebase
- GSTINs follow format: `{2-digit state code}{5 alpha}{4 digits}{1 alpha}{1 digit}{1 alpha}{1 digit}` e.g. `27AABCU9603R1ZX`
- All episodes seeded with `random.seed(episode_seed)` for reproducibility
- Mismatch types injected deliberately: `value_mismatch`, `duplicate_invoice`, `gstin_mismatch`, `missing_on_portal`

---

## Hard Constraints — Never Violate

| Rule | Detail |
|---|---|
| `inference.py` location | Must be at project root, named exactly `inference.py` |
| LLM SDK | Only `from openai import OpenAI` — no other SDK in `inference.py` |
| `API_BASE_URL` | `os.getenv("API_BASE_URL", "https://api.openai.com/v1")` — default required |
| `MODEL_NAME` | `os.getenv("MODEL_NAME", "gpt-4.1-mini")` — default required |
| `HF_TOKEN` | `os.getenv("HF_TOKEN")` — raise `ValueError` if None |
| stdout format | `[START]`, `[STEP]`, `[END]` lines exactly as specified |
| Resource limit | Must run in 2 vCPU / 8 GB RAM — no heavy ML models loaded |
| HF Space state | Must be in "Running" state at submission time |

---

## stdout Format for inference.py

```
[START] task=<task_name> env=gst-sahayak model=<MODEL_NAME>
[STEP] step=<n> action=<action_json_str> reward=<X.XX> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

- One `[START]` per episode
- One `[STEP]` per step, immediately after `env.step()` returns
- One `[END]` after `env.close()` — emit even on exception
- `reward` / `rewards` to 2 decimal places
- `done` / `success` lowercase: `true` or `false`

---

## Decisions Already Made — Do Not Revisit

1. Problem domain: Indian GST compliance
2. Data: fully synthetic, no real APIs
3. Reward: incremental at every step
4. Three tasks: invoice_classifier / itc_reconciliation / full_gstr3b_filing
5. Agent: LLM via OpenAI client converting observation to action JSON
6. Environment: pure Python, no ML models, fits 2 vCPU / 8 GB
7. Server: FastAPI via HTTPEnvServer, port 7860
8. No UI work for prototype phase

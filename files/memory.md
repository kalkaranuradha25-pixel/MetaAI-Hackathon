# memory.md — GST Sahayak Running Log
> Update this file every session. It is the persistent memory of what has been done, what is blocked, and what comes next. Never delete old entries — append only.

---

## How to Use This File

- **Before starting work:** Read the last `## Session` block to know where you left off
- **While working:** Update the `Current blockers` and `Progress` fields in the active session
- **Before stopping:** Write a `## Session` block summarizing what you did and what's next
- **When a decision is made:** Add it to `Decisions Log` at the bottom

---

## Project Status

**Overall phase:** Hackathon Prototype
**Target:** Working HF Space deployment for submission
**Deadline:** April 8, 2026 (Round 1)

### File completion status

| File | Status | Notes |
|---|---|---|
| `PRD.md` | ✅ Done | Full requirement mapping |
| `context.md` | ✅ Done | OpenEnv API, constraints, structure |
| `memory.md` | ✅ Done | This file |
| `agent.md` | ✅ Done | Agent design, prompt, action loop |
| `models.py` | ⬜ Not started | GSTAction, GSTObservation, GSTState |
| `server/data_generator.py` | ⬜ Not started | Synthetic invoice builder |
| `server/graders.py` | ⬜ Not started | 3 task graders |
| `server/gst_environment.py` | ⬜ Not started | reset/step/state |
| `server/app.py` | ⬜ Not started | FastAPI via HTTPEnvServer |
| `client.py` | ⬜ Not started | EnvClient subclass |
| `inference.py` | ⬜ Not started | LLM agent loop, stdout format |
| `openenv.yaml` | ⬜ Not started | Manifest |
| `Dockerfile` | ⬜ Not started | Container config |
| `README.md` | ⬜ Not started | All 5 required sections |

---

## Session Log

---

### Session 1 — April 8, 2026

**What was done:**
- Defined problem statement: GST Sahayak — Indian GST compliance RL env
- Chose domain based on market research (63M MSMEs, ₹13–17L compliance overhead)
- Designed 3 tasks: invoice_classifier (easy), itc_reconciliation (medium), full_gstr3b_filing (hard)
- Wrote full PRD.md with all hackathon compliance checkboxes
- Verified actual meta-pytorch OpenEnv API from DeepWiki docs
- Key discovery: `openenv-core` is the correct pip package, NOT `openenv`
- Key discovery: OpenEnv's `step()` returns an Observation object, not a 4-tuple
- Key discovery: Observation base class already has `reward` and `done` fields
- Key discovery: State base class already has `episode_id` and `step_count`
- Wrote context.md, memory.md, agent.md

**Current blockers:**
- None at planning stage
- Next blocker will likely be: getting `openenv-core` import paths correct (package is early-stage, docs may lag code)

**What to do next session:**
1. Install `openenv-core` and verify import paths work: `from openenv.core.env_server.interfaces import Environment`
2. Write `models.py` first — GSTAction, GSTObservation, GSTState
3. Write `data_generator.py` — synthetic invoices with ground truth
4. Write Task 1 grader in `graders.py`
5. Write `gst_environment.py` — reset() and step() for Task 1 only
6. Test the loop: reset → step → check reward → done

---

## Decisions Log
> Append-only. Never delete. Each entry has a date and rationale.

---

**2026-04-08** — Use `openenv-core` not `openenv`
The PyPI package `openenv` is a completely different library (by Jianfeng Hou, for classic RL envs). Meta's framework is `openenv-core`. Using the wrong one will cause silent failures because the import names are different.

**2026-04-08** — step() returns Observation, not a 4-tuple
Confirmed from DeepWiki: OpenEnv's `step()` returns a single Observation object. The `reward` and `done` fields are baked into the Observation base class. All inference.py code must read `obs.reward` and `obs.done`, not unpack a tuple.

**2026-04-08** — All data is synthetic
Decided against using real GST data or calling real GSTN APIs. Reasons: (1) legal/privacy risk, (2) rate limits would break the RL training loop, (3) synthetic data with seeded randomness is more reproducible for grading.

**2026-04-08** — Prototype scope locked
No real GSTN API, no frontend UI, no database, no auth. Just the RL environment, graders, and inference script. Expand to full product post-hackathon.

**2026-04-08** — Three tasks, fixed difficulty order
easy → medium → hard as required by problem statement. Task 1 is self-contained (classification only). Task 2 adds reconciliation decisions. Task 3 is a superset of both plus final filing.

**2026-04-08** — Reward is incremental, not terminal-only
The problem statement explicitly requires rewards throughout the trajectory. Every `step()` call must return a non-zero reward for meaningful actions. Terminal bonus only on correct filing.

**2026-04-08** — Agent uses LLM for inference, not a trained RL model
For the hackathon submission, the agent in `inference.py` is an LLM (via OpenAI client) that reads the observation as a natural language prompt and outputs a JSON action. The RL environment is the contribution — the trained agent comes later.

---

## Known Gotchas

| Gotcha | Detail |
|---|---|
| Wrong pip package | `pip install openenv` installs the wrong library. Use `pip install openenv-core` |
| step() return type | Returns `GSTObservation`, not `(obs, reward, done, info)` |
| Observation base fields | Do NOT redefine `reward` or `done` in `GSTObservation` — they are inherited |
| State base fields | Do NOT redefine `episode_id` or `step_count` in `GSTState` — they are inherited |
| Action model config | OpenEnv Action base forbids extra fields by default. Every field must be declared |
| inference.py location | Must be at project root. If inside a subfolder, submission auto-fails |
| HF Space state | Space must show "Running" at submission time. Submit only after full build |
| stdout format | All three line types must be exact. Test with `python inference.py \| grep -E '^\[(START\|STEP\|END)\]'` |

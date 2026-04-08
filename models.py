"""
GST Sahayak — Pydantic Models
GSTAction, GSTObservation, GSTState inheriting from OpenEnv base types.
"""

from typing import List, Optional, Dict, Any
from openenv.core.env_server.types import Action, Observation, State


class GSTAction(Action):
    """
    What the agent sends to the environment each step.
    action_type options:
      classify_invoice  — assign invoice_type and hsn_code to an invoice
      accept_itc        — accept ITC claim for an invoice
      reject_itc        — reject ITC claim for an invoice
      flag_for_review   — flag invoice for human review (safe partial credit)
      compute_liability — calculate net tax liability (Task 3 step)
      file_return       — submit final GSTR-3B payload (terminal action)
    """
    action_type: str
    invoice_id: Optional[str] = None
    hsn_code: Optional[str] = None
    invoice_type: Optional[str] = None      # "B2B" | "B2C" | "EXPORT" | "EXEMPT"
    gstr_payload: Optional[Dict[str, Any]] = None  # only for file_return


class GSTObservation(Observation):
    """
    What the environment returns after each step.
    Inherits `reward: float` and `done: bool` from Observation base.
    """
    invoices: List[dict]          # current batch of invoices to process
    gstr_2b: List[dict]           # GSTN portal's GSTR-2B match records
    accumulated_itc: float        # running ITC claimed so far
    tax_liability: float          # running liability computed so far
    step: int
    max_steps: int
    episode_context: str          # "b2b_heavy" | "b2c_heavy" | "export" | "mixed"
    pending_actions: List[str]    # invoice IDs not yet actioned
    filing_status: str            # "not_started" | "in_progress" | "filed"
    last_error: Optional[str] = None  # error message from last step, or None


class GSTState(State):
    """
    Full internal state for debugging and OpenEnv inspection.
    Inherits `episode_id: str` and `step_count: int` from State base.
    """
    ground_truth_invoices: List[dict]   # invoices with correct classifications
    ground_truth_itc: List[dict]        # ITC decisions with correct accept/reject
    classified_so_far: Dict[str, dict]  # invoice_id -> {type, hsn} as classified
    itc_decisions: Dict[str, str]       # invoice_id -> "accepted"|"rejected"|"flagged"
    cumulative_reward: float
    task: str
    audit_flags: int                    # count of wrongly accepted mismatched ITCs
    liability_computed: bool            # True after compute_liability action
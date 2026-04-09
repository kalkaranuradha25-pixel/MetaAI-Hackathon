"""
GST Sahayak — Synthetic Data Generator
Produces reproducible, seeded invoice batches and GSTR-2B mismatch data.
No real taxpayer data is used anywhere in this file.
"""

import random
import uuid
from typing import List, Dict, Tuple

# --- Constants ---

INVOICE_TYPES = ["B2B", "B2C", "EXPORT", "EXEMPT"]

# HSN codes by invoice type (realistic categories)
HSN_CODES: Dict[str, List[str]] = {
    "B2B": ["8471", "8517", "6204", "3004", "7208", "9403", "8528"],
    "B2C": ["2106", "6109", "8516", "9401", "4901", "6302", "3305"],
    "EXPORT": ["8703", "8471", "6204", "7113", "8542"],
    "EXEMPT": ["0101", "0201", "2201", "0602", "4906"],
}

# GST rates by invoice type
GST_RATES: Dict[str, float] = {
    "B2B": 0.18,
    "B2C": 0.12,
    "EXPORT": 0.00,
    "EXEMPT": 0.00,
}

# Intra-state vs inter-state split (determines CGST/SGST vs IGST)
INTRA_STATE_PROB = 0.6  # 60% intra-state transactions

# State codes for GSTIN generation
STATE_CODES = ["27", "07", "29", "19", "33", "36", "06", "24", "09", "20"]

# Vendor names (fictional, for display only)
VENDOR_NAMES = [
    "Rajesh Traders", "Mumbai Tech Solutions", "Delhi Pharma Pvt Ltd",
    "Bengaluru Exports", "Chennai Garments", "Hyderabad Steel Works",
    "Pune Auto Parts", "Kolkata Wholesale", "Ahmedabad Textiles",
    "Jaipur Handicrafts", "Surat Diamond", "Coimbatore Machinery",
]

MISMATCH_TYPES = ["value_mismatch", "duplicate_invoice", "gstin_mismatch", "missing_on_portal"]


# --- GSTIN Generator ---

def generate_gstin(rng: random.Random) -> str:
    """Generate a valid-format (but fictional) GSTIN."""
    state = rng.choice(STATE_CODES)
    # 5 alpha chars
    alpha5 = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    # 4 digits
    digits4 = "".join(rng.choices("0123456789", k=4))
    # 1 alpha
    alpha1 = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # check digit pattern: 1Z<digit>
    check = f"1Z{rng.randint(1, 9)}"
    return f"{state}{alpha5}{digits4}{alpha1}{check}"


# --- Invoice Generator ---

def generate_invoice(rng: random.Random, invoice_index: int, episode_seed: int,
                     force_type: str = None) -> dict:
    """Generate one synthetic invoice with ground truth classification."""
    inv_type = force_type if force_type else rng.choice(INVOICE_TYPES)
    hsn = rng.choice(HSN_CODES[inv_type])
    taxable_value = round(rng.uniform(5000, 200000), 2)

    rate = GST_RATES[inv_type]
    intra = rng.random() < INTRA_STATE_PROB

    if rate == 0.0:
        igst = cgst = sgst = 0.0
    elif intra:
        igst = 0.0
        half = round(taxable_value * rate / 2, 2)
        cgst = half
        sgst = half
    else:
        igst = round(taxable_value * rate, 2)
        cgst = 0.0
        sgst = 0.0

    inv_id = f"INV-{episode_seed:04d}-{invoice_index:03d}"
    vendor = rng.choice(VENDOR_NAMES)

    return {
        "invoice_id": inv_id,
        "vendor_name": vendor,
        "vendor_gstin": generate_gstin(rng),
        "invoice_date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "invoice_number": f"{vendor[:3].upper()}/{rng.randint(1000, 9999)}",
        "taxable_value": taxable_value,
        "igst": igst,
        "cgst": cgst,
        "sgst": sgst,
        # Ground truth (hidden from agent in observation)
        "_ground_truth_type": inv_type,
        "_ground_truth_hsn": hsn,
    }


def invoice_for_agent(inv: dict) -> dict:
    """Return invoice dict with ground truth fields stripped (what agent sees)."""
    return {k: v for k, v in inv.items() if not k.startswith("_")}


# --- GSTR-2B Generator ---

def generate_gstr2b(invoices: List[dict], rng: random.Random,
                    num_mismatches: int) -> Tuple[List[dict], List[str]]:
    """
    Generate GSTR-2B portal data. Injects deliberate mismatches into some invoices.
    Returns (gstr_2b_records, list_of_mismatch_invoice_ids).
    """
    mismatch_indices = rng.sample(range(len(invoices)), min(num_mismatches, len(invoices)))
    mismatch_ids = set()
    records = []

    for i, inv in enumerate(invoices):
        if i in mismatch_indices:
            mtype = rng.choice(MISMATCH_TYPES)
            mismatch_ids.add(inv["invoice_id"])

            if mtype == "value_mismatch":
                factor = rng.uniform(0.70, 0.90)  # portal shows 10-30% less
                portal_taxable = round(inv["taxable_value"] * factor, 2)
                portal_igst = round(inv["igst"] * factor, 2)
            elif mtype == "missing_on_portal":
                # Invoice not found in GSTR-2B at all
                records.append({
                    "invoice_id": inv["invoice_id"],
                    "portal_taxable_value": 0.0,
                    "portal_igst": 0.0,
                    "mismatch": True,
                    "mismatch_reason": "missing_on_portal",
                })
                continue
            elif mtype == "gstin_mismatch":
                portal_taxable = inv["taxable_value"]
                portal_igst = inv["igst"]
            else:  # duplicate_invoice
                portal_taxable = inv["taxable_value"]
                portal_igst = inv["igst"]

            records.append({
                "invoice_id": inv["invoice_id"],
                "portal_taxable_value": portal_taxable if mtype != "gstin_mismatch" else inv["taxable_value"],
                "portal_igst": portal_igst if mtype != "gstin_mismatch" else inv["igst"],
                "mismatch": True,
                "mismatch_reason": mtype,
            })
        else:
            # Clean match
            records.append({
                "invoice_id": inv["invoice_id"],
                "portal_taxable_value": inv["taxable_value"],
                "portal_igst": inv["igst"],
                "mismatch": False,
                "mismatch_reason": None,
            })

    return records, list(mismatch_ids)


# --- Episode Generator ---

TASK_CONFIGS = {
    "invoice_classifier": {
        "num_invoices": 10,
        "num_mismatches": 0,
        "max_steps": 20,
        "context_weights": {"b2b_heavy": 0.4, "b2c_heavy": 0.3, "export": 0.15, "mixed": 0.15},
    },
    "itc_reconciliation": {
        "num_invoices": 15,
        "num_mismatches": 5,
        "max_steps": 35,
        "context_weights": {"b2b_heavy": 0.5, "mixed": 0.3, "export": 0.1, "b2c_heavy": 0.1},
    },
    "full_gstr3b_filing": {
        "num_invoices": 20,
        "num_mismatches": 6,
        "max_steps": 60,
        "context_weights": {"mixed": 0.5, "b2b_heavy": 0.3, "export": 0.1, "b2c_heavy": 0.1},
    },
}

CONTEXT_TYPE_DISTRIBUTIONS = {
    "b2b_heavy": {"B2B": 0.7, "B2C": 0.1, "EXPORT": 0.1, "EXEMPT": 0.1},
    "b2c_heavy": {"B2B": 0.1, "B2C": 0.7, "EXPORT": 0.1, "EXEMPT": 0.1},
    "export":    {"B2B": 0.2, "B2C": 0.1, "EXPORT": 0.6, "EXEMPT": 0.1},
    "mixed":     {"B2B": 0.3, "B2C": 0.3, "EXPORT": 0.2, "EXEMPT": 0.2},
}


def pick_episode_context(rng: random.Random, weights: dict) -> str:
    contexts = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(contexts, weights=probs, k=1)[0]


def sample_invoice_type(rng: random.Random, context: str) -> str:
    dist = CONTEXT_TYPE_DISTRIBUTIONS[context]
    types = list(dist.keys())
    probs = list(dist.values())
    return rng.choices(types, weights=probs, k=1)[0]


def generate_episode(seed: int, task: str) -> dict:
    """
    Generate a complete episode for the given task.
    Returns a dict with all data needed to initialize the environment state.
    """
    rng = random.Random(seed)
    cfg = TASK_CONFIGS[task]

    # Pick episode context
    context = pick_episode_context(rng, cfg["context_weights"])

    # Generate invoices (with ground truth hidden)
    invoices = [
        generate_invoice(rng, i, seed, force_type=sample_invoice_type(rng, context))
        for i in range(cfg["num_invoices"])
    ]

    # Generate GSTR-2B portal data
    gstr_2b, mismatch_ids = generate_gstr2b(invoices, rng, cfg["num_mismatches"])

    # Build ground truth maps
    ground_truth_invoices = invoices  # contains _ground_truth_* fields
    ground_truth_itc = [
        {
            "invoice_id": inv["invoice_id"],
            "correct_decision": "reject" if inv["invoice_id"] in mismatch_ids else "accept",
            "mismatch_reason": next(
                (r["mismatch_reason"] for r in gstr_2b if r["invoice_id"] == inv["invoice_id"]),
                None
            ),
        }
        for inv in invoices
    ]

    return {
        "task": task,
        "seed": seed,
        "context": context,
        "max_steps": cfg["max_steps"],
        "invoices": invoices,                      # full with ground truth
        "gstr_2b": gstr_2b,                        # portal data
        "ground_truth_invoices": ground_truth_invoices,
        "ground_truth_itc": ground_truth_itc,
        "mismatch_ids": set(mismatch_ids),
    }


def compute_ground_truth_gstr3b(invoices: List[dict], accepted_itc_ids: List[str]) -> dict:
    """
    Compute the correct GSTR-3B payload from invoice ground truths.
    Used by the grader to score Task 3 file_return actions.
    """
    outward = {"igst": 0.0, "cgst": 0.0, "sgst": 0.0}
    zero_rated = {"igst": 0.0}
    exempt = {"value": 0.0}
    itc_avail = {"igst": 0.0, "cgst": 0.0, "sgst": 0.0}

    for inv in invoices:
        inv_type = inv.get("_ground_truth_type", "B2B")
        tv = inv["taxable_value"]

        if inv_type == "EXPORT":
            zero_rated["igst"] += inv.get("igst", 0.0)
        elif inv_type == "EXEMPT":
            exempt["value"] += tv
        else:
            outward["igst"] += inv.get("igst", 0.0)
            outward["cgst"] += inv.get("cgst", 0.0)
            outward["sgst"] += inv.get("sgst", 0.0)

        # ITC is only for B2B invoices that were accepted
        if inv_type == "B2B" and inv["invoice_id"] in accepted_itc_ids:
            itc_avail["igst"] += inv.get("igst", 0.0)
            itc_avail["cgst"] += inv.get("cgst", 0.0)
            itc_avail["sgst"] += inv.get("sgst", 0.0)

    # Round all values
    for d in [outward, zero_rated, exempt, itc_avail]:
        for k in d:
            d[k] = round(d[k], 2)

    net_liability = {
        "igst": round(max(0, outward["igst"] - itc_avail["igst"]), 2),
        "cgst": round(max(0, outward["cgst"] - itc_avail["cgst"]), 2),
        "sgst": round(max(0, outward["sgst"] - itc_avail["sgst"]), 2),
    }

    return {
        "outward_taxable_supplies": outward,
        "zero_rated_supplies": zero_rated,
        "exempt_supplies": exempt,
        "itc_available": itc_avail,
        "itc_reversed": {"igst": 0.0, "cgst": 0.0, "sgst": 0.0},
        "net_tax_liability": net_liability,
    }

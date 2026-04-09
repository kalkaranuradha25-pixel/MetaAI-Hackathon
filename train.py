"""
GST Sahayak — RL Training Script (REINFORCE / Policy Gradient)
Trains a neural policy directly on the GSTEnvironment without any LLM.

Usage:
    cd gst-sahayak
    python train.py                          # train Task 1 (default)
    python train.py --task invoice_classifier
    python train.py --task itc_reconciliation
    python train.py --task full_gstr3b_filing
    python train.py --task all              # train all 3 in sequence
    python train.py --episodes 300 --eval-every 25
"""

import sys
import os
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Make sure we can import from this project
sys.path.insert(0, os.path.dirname(__file__))

from models import GSTAction
from server.data_generator import (
    generate_episode, invoice_for_agent,
    HSN_CODES, INVOICE_TYPES, TASK_CONFIGS
)
from server.graders import grade_invoice_classifier, grade_itc_reconciliation, _SCORE_MIN

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

INVOICE_TYPES_LIST = ["B2B", "B2C", "EXPORT", "EXEMPT"]   # 4 types
ALL_HSN = sorted({h for codes in HSN_CODES.values() for h in codes})  # ~15 unique HSN codes
HSN_TO_IDX = {h: i for i, h in enumerate(ALL_HSN)}
IDX_TO_HSN = {i: h for h, i in HSN_TO_IDX.items()}
NUM_HSN = len(ALL_HSN)

ITC_ACTIONS = ["accept_itc", "reject_itc", "flag_for_review"]  # 3 choices

DEVICE = torch.device("cpu")  # keep lightweight for 2 vCPU


# -------------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------------

def invoice_to_features(inv: dict) -> torch.Tensor:
    """
    Convert one invoice dict to a fixed-size feature vector (12 dims).

    Key signals the policy learns:
      - Effective tax rate ~18% -> likely B2B
      - Effective tax rate ~12% -> likely B2C
      - Effective tax rate  ~0% -> EXPORT or EXEMPT (zero-rated / truly exempt)
      - Intra-state (CGST+SGST present) vs inter-state (IGST only)
    """
    tv   = float(inv.get("taxable_value", 0))
    igst = float(inv.get("igst", 0))
    cgst = float(inv.get("cgst", 0))
    sgst = float(inv.get("sgst", 0))
    total_tax = igst + cgst + sgst
    eff_rate  = total_tax / max(tv, 1.0)

    feats = [
        min(tv / 200_000.0, 1.0),          # normalized invoice value
        min(total_tax / 40_000.0, 1.0),    # normalized total tax
        float(igst > 0),                   # inter-state transaction
        float(cgst > 0),                   # intra-state transaction
        float(total_tax == 0.0),           # zero tax (EXPORT or EXEMPT)
        float(eff_rate > 0.15),            # effective rate > 15% -> B2B (18%)
        float(0.08 < eff_rate <= 0.15),    # effective rate 8–15% -> B2C (12%)
        float(eff_rate < 0.01),            # near-zero rate -> EXPORT or EXEMPT
        float(igst > 0 and cgst == 0),     # pure IGST (inter-state or export)
        float(cgst > 0 and igst == 0),     # CGST+SGST split (domestic intra)
        float(cgst > 0 and eff_rate > 0.15),  # high-rate intra-state -> B2B
        float(cgst > 0 and 0.08 < eff_rate <= 0.15),  # mid-rate intra -> B2C
    ]
    return torch.tensor(feats, dtype=torch.float32)


def gstr2b_to_feature(match_record: dict) -> float:
    """Single feature: 1.0 if mismatch, 0.0 if clean."""
    return float(match_record.get("mismatch", False))


# -------------------------------------------------------------------------
# Policy Networks
# -------------------------------------------------------------------------

class InvoiceClassifierPolicy(nn.Module):
    """
    Policy for Task 1 — invoice_classifier.
    Input: invoice features (12 dims)
    Output: invoice_type (4) + hsn_code (NUM_HSN)
    """
    def __init__(self, input_dim=12, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.type_head = nn.Linear(hidden, len(INVOICE_TYPES_LIST))
        self.hsn_head  = nn.Linear(hidden, NUM_HSN)

    def forward(self, x):
        h = self.shared(x)
        return self.type_head(h), self.hsn_head(h)

    def select_action(self, inv: dict):
        """Sample action from policy. Returns (invoice_type, hsn_code, log_prob)."""
        feats = invoice_to_features(inv).unsqueeze(0)
        type_logits, hsn_logits = self.forward(feats)

        type_dist = Categorical(logits=type_logits)
        hsn_dist  = Categorical(logits=hsn_logits)

        type_idx = type_dist.sample()
        hsn_idx  = hsn_dist.sample()

        log_prob = type_dist.log_prob(type_idx) + hsn_dist.log_prob(hsn_idx)

        return (
            INVOICE_TYPES_LIST[type_idx.item()],
            IDX_TO_HSN[hsn_idx.item()],
            log_prob,
        )


class ITCReconciliationPolicy(nn.Module):
    """
    Policy for Task 2 — itc_reconciliation.
    Input: invoice features (12) + mismatch indicator (1) = 13 dims
    Output: accept/reject/flag (3)
    """
    def __init__(self, input_dim=13, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(ITC_ACTIONS)),
        )

    def forward(self, x):
        return self.net(x)

    def select_action(self, inv: dict, match_record: dict):
        """Sample ITC decision. Returns (action_type, log_prob)."""
        feats = invoice_to_features(inv)
        mismatch_feat = torch.tensor([gstr2b_to_feature(match_record)], dtype=torch.float32)
        x = torch.cat([feats, mismatch_feat]).unsqueeze(0)

        logits = self.forward(x)
        dist = Categorical(logits=logits)
        idx = dist.sample()
        return ITC_ACTIONS[idx.item()], dist.log_prob(idx)


# -------------------------------------------------------------------------
# Environment wrapper with variable seeds for training
# -------------------------------------------------------------------------

class TrainingEnv:
    """
    Lightweight training wrapper that bypasses the OpenEnv server entirely.
    Runs episodes directly using generate_episode + graders.
    Uses random seeds for variety during training.
    """

    def __init__(self, task: str, seed_range=(0, 500)):
        self.task = task
        self.seed_range = seed_range
        self._data = None
        self._classified = {}
        self._itc_decisions = {}
        self._accepted_itc = []
        self._mismatch_ids = set()
        self._pending = []
        self._step = 0

    def reset(self, seed: int = None):
        if seed is None:
            seed = random.randint(*self.seed_range)
        self._data = generate_episode(seed, self.task)
        self._classified = {}
        self._itc_decisions = {}
        self._accepted_itc = []
        self._mismatch_ids = self._data["mismatch_ids"]
        self._step = 0
        self._pending = [i["invoice_id"] for i in self._data["invoices"]]
        return self

    @property
    def pending(self):
        if self.task == "invoice_classifier":
            return [i for i in self._pending if i not in self._classified]
        else:
            return [i for i in self._pending if i not in self._itc_decisions]

    @property
    def done(self):
        return len(self.pending) == 0

    def get_invoice(self, inv_id: str) -> dict:
        return next(i for i in self._data["invoices"] if i["invoice_id"] == inv_id)

    def get_match_record(self, inv_id: str) -> dict:
        return next((r for r in self._data["gstr_2b"] if r["invoice_id"] == inv_id), {})

    def step_classify(self, inv_id: str, inv_type: str, hsn_code: str) -> float:
        gt = next(i for i in self._data["ground_truth_invoices"] if i["invoice_id"] == inv_id)
        correct_type = inv_type == gt["_ground_truth_type"]
        correct_hsn  = hsn_code == gt["_ground_truth_hsn"]
        self._classified[inv_id] = {"invoice_type": inv_type, "hsn_code": hsn_code}
        if correct_type and correct_hsn:
            return 0.15
        elif correct_type:
            return 0.05
        return -0.05

    def step_itc(self, inv_id: str, action_type: str) -> float:
        gt_itc = next(i for i in self._data["ground_truth_itc"] if i["invoice_id"] == inv_id)
        decision_map = {"accept_itc": "accepted", "reject_itc": "rejected", "flag_for_review": "flagged"}
        self._itc_decisions[inv_id] = decision_map[action_type]
        if action_type == "accept_itc":
            self._accepted_itc.append(inv_id)

        if action_type == "flag_for_review":
            return 0.05
        agent_accept  = action_type == "accept_itc"
        should_accept = gt_itc["correct_decision"] == "accept"
        if agent_accept and should_accept:   return  0.20
        if not agent_accept and not should_accept: return  0.20
        if agent_accept and not should_accept:     return -0.30
        return -0.15

    def final_score(self) -> float:
        if self.task == "invoice_classifier":
            return grade_invoice_classifier(
                self._data["ground_truth_invoices"], self._classified
            )
        elif self.task == "itc_reconciliation":
            return grade_itc_reconciliation(
                self._data["ground_truth_itc"], self._itc_decisions
            )
        return _SCORE_MIN


# -------------------------------------------------------------------------
# REINFORCE training loop
# -------------------------------------------------------------------------

def run_classifier_episode(env: TrainingEnv, policy: InvoiceClassifierPolicy,
                            train: bool = True):
    """Run one full Task 1 episode. Returns (log_probs, rewards, final_score)."""
    env.reset()
    log_probs = []
    rewards = []

    while not env.done:
        inv_id = env.pending[0]
        inv = env.get_invoice(inv_id)

        inv_type, hsn_code, lp = policy.select_action(inv)
        reward = env.step_classify(inv_id, inv_type, hsn_code)

        log_probs.append(lp)
        rewards.append(reward)

    return log_probs, rewards, env.final_score()


def run_itc_episode(env: TrainingEnv, policy: ITCReconciliationPolicy,
                    train: bool = True):
    """Run one full Task 2 episode. Returns (log_probs, rewards, final_score)."""
    env.reset()
    log_probs = []
    rewards = []

    while not env.done:
        inv_id = env.pending[0]
        inv = env.get_invoice(inv_id)
        match = env.get_match_record(inv_id)

        action_type, lp = policy.select_action(inv, match)
        reward = env.step_itc(inv_id, action_type)

        log_probs.append(lp)
        rewards.append(reward)

    return log_probs, rewards, env.final_score()


def reinforce_update(optimizer, log_probs, rewards, gamma=0.99, baseline=0.0):
    """
    REINFORCE policy gradient update.
    G_t = sum of discounted future rewards from step t.
    Loss = -sum(log_prob_t * (G_t - baseline))
    """
    # Compute discounted returns
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize returns (reduces variance)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute loss
    loss = -sum(lp * R for lp, R in zip(log_probs, returns))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for g in optimizer.param_groups for p in g["params"]], max_norm=1.0
    )
    optimizer.step()

    return loss.item()


def train_task(task: str, num_episodes: int = 200, lr: float = 1e-3,
               eval_every: int = 20, eval_seeds: list = None):
    """
    Train a policy on a given task using REINFORCE.
    Prints reward curve and final grader scores.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {task}  |  {num_episodes} episodes  |  lr={lr}")
    print(f"{'='*60}")

    env = TrainingEnv(task)

    if task == "invoice_classifier":
        policy = InvoiceClassifierPolicy().to(DEVICE)
        run_episode = run_classifier_episode
    elif task == "itc_reconciliation":
        policy = ITCReconciliationPolicy().to(DEVICE)
        run_episode = run_itc_episode
    else:
        print(f"  [!] full_gstr3b_filing requires Task 1 + Task 2 policies combined.")
        print(f"      Train invoice_classifier and itc_reconciliation first.")
        return None

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    if eval_seeds is None:
        eval_seeds = [42, 137, 999, 7, 23]   # fixed eval seeds

    # --- Training loop ---
    recent_rewards = []
    best_score = 0.0
    best_state = None

    for ep in range(1, num_episodes + 1):
        policy.train()
        log_probs, rewards, score = run_episode(env, policy, train=True)
        loss = reinforce_update(optimizer, log_probs, rewards)
        ep_total = sum(rewards)
        recent_rewards.append(ep_total)

        # --- Periodic eval ---
        if ep % eval_every == 0:
            policy.eval()
            eval_scores = []
            with torch.no_grad():
                for seed in eval_seeds:
                    env.reset(seed=seed)
                    # Greedy eval: use argmax instead of sampling
                    greedy_score = greedy_eval(env, policy, task)
                    eval_scores.append(greedy_score)

            mean_eval = np.mean(eval_scores)
            mean_train = np.mean(recent_rewards[-eval_every:])
            recent_rewards_str = f"{mean_train:.3f}"
            print(
                f"  Ep {ep:>4}/{num_episodes}"
                f"  train_reward={recent_rewards_str}"
                f"  eval_score={mean_eval:.4f}"
                f"  loss={loss:.4f}"
            )

            if mean_eval > best_score:
                best_score = mean_eval
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}

    # --- Restore best ---
    if best_state:
        policy.load_state_dict(best_state)

    # --- Final evaluation ---
    print(f"\n  Final evaluation (10 seeds):")
    final_seeds = list(range(200, 210))
    policy.eval()
    final_scores = []
    with torch.no_grad():
        for seed in final_seeds:
            env.reset(seed=seed)
            s = greedy_eval(env, policy, task)
            final_scores.append(s)

    mean_final = np.mean(final_scores)
    print(f"  Best eval score seen  : {best_score:.4f}")
    print(f"  Mean final score      : {mean_final:.4f}")

    cfg = TASK_CONFIGS[task]
    threshold = cfg.get("reward_threshold", 0.7)
    status = "[PASS]" if mean_final >= threshold else f"[below {threshold}]"
    print(f"  Target threshold      : {threshold}  {status}")

    return policy


def greedy_eval(env: TrainingEnv, policy, task: str) -> float:
    """Run one greedy episode (argmax actions, no sampling) and return final score."""
    while not env.done:
        inv_id = env.pending[0]
        inv = env.get_invoice(inv_id)

        if task == "invoice_classifier":
            feats = invoice_to_features(inv).unsqueeze(0)
            type_logits, hsn_logits = policy(feats)
            inv_type = INVOICE_TYPES_LIST[type_logits.argmax().item()]
            hsn_code  = IDX_TO_HSN[hsn_logits.argmax().item()]
            env.step_classify(inv_id, inv_type, hsn_code)

        elif task == "itc_reconciliation":
            match = env.get_match_record(inv_id)
            feats = invoice_to_features(inv)
            mf = torch.tensor([gstr2b_to_feature(match)], dtype=torch.float32)
            x = torch.cat([feats, mf]).unsqueeze(0)
            logits = policy(x)
            action_type = ITC_ACTIONS[logits.argmax().item()]
            env.step_itc(inv_id, action_type)

    return env.final_score()


# -------------------------------------------------------------------------
# Random agent baseline
# -------------------------------------------------------------------------

def random_baseline(task: str, num_episodes: int = 50) -> float:
    """Measure random agent score for comparison."""
    env = TrainingEnv(task, seed_range=(1000, 2000))
    scores = []

    for _ in range(num_episodes):
        env.reset()
        while not env.done:
            inv_id = env.pending[0]
            if task == "invoice_classifier":
                inv_type = random.choice(INVOICE_TYPES_LIST)
                hsn_code = random.choice(ALL_HSN)
                env.step_classify(inv_id, inv_type, hsn_code)
            elif task == "itc_reconciliation":
                action = random.choice(ITC_ACTIONS)
                env.step_itc(inv_id, action)
        scores.append(env.final_score())

    return float(np.mean(scores))


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GST Sahayak RL agent")
    parser.add_argument("--task", default="invoice_classifier",
                        choices=["invoice_classifier", "itc_reconciliation",
                                 "full_gstr3b_filing", "all"],
                        help="Which task to train on")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes per task")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Evaluate every N episodes")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tasks = (
        ["invoice_classifier", "itc_reconciliation"]
        if args.task == "all"
        else [args.task] if args.task != "full_gstr3b_filing"
        else ["full_gstr3b_filing"]
    )

    print("\nGST Sahayak — RL Training (REINFORCE)")
    print("Algorithm : REINFORCE with baseline normalization")
    print("Device    :", DEVICE)
    print("Tasks     :", tasks)

    for task in tasks:
        if task == "full_gstr3b_filing":
            print("\n[full_gstr3b_filing]")
            print("  This task requires combining Task 1 + Task 2 policies.")
            print("  Run: python train.py --task invoice_classifier")
            print("       python train.py --task itc_reconciliation")
            print("  Then see train_task3() in train.py for the combined loop.")
            continue

        # Random baseline
        print(f"\nComputing random baseline for {task}...")
        rand_score = random_baseline(task, num_episodes=50)
        print(f"  Random agent score: {rand_score:.4f}")

        # Train
        policy = train_task(
            task,
            num_episodes=args.episodes,
            lr=args.lr,
            eval_every=args.eval_every,
        )

        # Save policy
        if policy is not None:
            save_path = f"trained_{task}.pt"
            torch.save(policy.state_dict(), save_path)
            print(f"\n  Policy saved to: {save_path}")
            print(f"  Load it later:  policy.load_state_dict(torch.load('{save_path}'))")


if __name__ == "__main__":
    main()

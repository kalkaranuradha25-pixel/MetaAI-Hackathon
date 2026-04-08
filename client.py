"""
GST Sahayak — EnvClient
Remote client for connecting to the GSTEnvironment HTTP server.
Used by RL training loops and external agents.
"""

from openenv import GenericEnvClient
from models import GSTAction, GSTObservation


class GSTEnvClient(GenericEnvClient):
    """
    Async client for the GST Sahayak environment server.

    Usage:
        async with GSTEnvClient(base_url="http://localhost:7860") as client:
            obs = await client.reset(task="invoice_classifier")
            while not obs.done:
                action = GSTAction(
                    action_type="classify_invoice",
                    invoice_id=obs.pending_actions[0],
                    invoice_type="B2B",
                    hsn_code="8471",
                )
                obs = await client.step(action)
            print("Final reward:", obs.reward)
    """
    pass

"""
GST Sahayak — FastAPI Server
Serves the GSTEnvironment over HTTP via OpenEnv's HTTPEnvServer.

Run with: uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI
from openenv.core.env_server.http_server import HTTPEnvServer

from .gst_environment import GSTEnvironment
from models import GSTAction, GSTObservation

app = FastAPI(
    title="GST Sahayak",
    description="RL environment for Indian GST compliance automation",
    version="1.0.0",
)

server = HTTPEnvServer(
    env=GSTEnvironment,           # factory — called once per session
    action_cls=GSTAction,
    observation_cls=GSTObservation,
    max_concurrent_envs=4,
)

server.register_routes(app)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

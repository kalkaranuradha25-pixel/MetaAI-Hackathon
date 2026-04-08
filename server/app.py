"""
GST Sahayak — FastAPI Server
Serves the GSTEnvironment over HTTP via OpenEnv's HTTPEnvServer.

Run with: uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openenv.core.env_server.http_server import HTTPEnvServer

from .gst_environment import GSTEnvironment
from models import GSTAction, GSTObservation

app = FastAPI(
    title="GST Sahayak",
    description="RL environment for Indian GST compliance automation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register OpenEnv routes (/reset, /step, /state, /health, /ws, etc.)
server = HTTPEnvServer(
    env=GSTEnvironment,
    action_cls=GSTAction,
    observation_cls=GSTObservation,
    max_concurrent_envs=4,
)
server.register_routes(app)

# ---------------------------------------------------------------------------
# Stateful UI session — single shared environment instance for the dashboard
# ---------------------------------------------------------------------------
_ui_env = GSTEnvironment()


class UIResetRequest(BaseModel):
    task: str = "invoice_classifier"


class UIStepRequest(BaseModel):
    action_type: str
    invoice_id: Optional[str] = None
    invoice_type: Optional[str] = None
    hsn_code: Optional[str] = None
    gstr_payload: Optional[dict] = None


@app.post("/ui/reset")
async def ui_reset(req: UIResetRequest):
    obs = _ui_env.reset(task=req.task)
    return obs.model_dump()


@app.post("/ui/step")
async def ui_step(req: UIStepRequest):
    action = GSTAction(
        action_type=req.action_type,
        invoice_id=req.invoice_id,
        invoice_type=req.invoice_type,
        hsn_code=req.hsn_code,
        gstr_payload=req.gstr_payload,
    )
    obs = _ui_env.step(action)
    return obs.model_dump()


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(str(STATIC_DIR / "index.html"))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

from fastapi import FastAPI
from pydantic import BaseModel
from src.serving.engine import SLMEngine

app = FastAPI(title="Enterprise SLM API")
engine = None  # initialized on startup

class GenerateRequest(BaseModel):
    prompts: list[str]
    temperature: float = 0.7
    max_tokens: int = 512

@app.on_event("startup")
async def startup():
    global engine
    # load configs...
    engine = SLMEngine(model_cfg, serving_cfg)

@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    return {"results": engine.generate(req.prompts)}
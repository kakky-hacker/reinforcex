import uuid

import gymnasium as gym
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
environments = {}


class StepRequest(BaseModel):
    session_id: str
    action: int


class ResetRequest(BaseModel):
    env: str


class ResetResponse(BaseModel):
    session_id: str
    observation: list


class StepResponse(BaseModel):
    observation: list
    reward: float
    done: bool
    info: dict


@app.post("/reset", response_model=ResetResponse)
def reset(data: ResetRequest):
    env = gym.make(data.env)
    obs, _ = env.reset()
    session_id = str(uuid.uuid4())
    environments[session_id] = env
    return {"session_id": session_id, "observation": obs.tolist()}


@app.post("/step", response_model=StepResponse)
def step(data: StepRequest):
    session_id = data.session_id
    action = data.action

    if session_id not in environments:
        raise HTTPException(status_code=404, detail="Session ID not found")

    env = environments[session_id]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        env.close()
        del environments[session_id]

    return {"observation": obs.tolist(), "reward": reward, "done": done, "info": info}

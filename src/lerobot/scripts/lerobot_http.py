# serve_policy_http.py
import os
import logging
import socket
from typing import Any, Dict, Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from PIL import Image
import base64
from io import BytesIO

# --- LeRobot imports ---
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.datasets.utils import build_dataset_frame

# --- Config ---
STATES_LEN = int(os.getenv('STATES_LEN', 8))
MODEL_REPO_ID = os.getenv('MODEL_REPO_ID', 'your_hf_username/your_policy')  # e.g., "lerobot/aloha_mobile_cnn_lstm"


class InferenceRequest(BaseModel):
    image: str  # base64 encoded main camera image
    wrist_image: Optional[str] = None  # optional
    state: List[float]  # robot state, e.g., [x, y, z, qx, qy, qz, qw, gripper]
    prompt: Optional[str] = None  # task description


class InferenceResponse(BaseModel):
    status: int = 0
    message: str = "success"
    result: Dict[str, Any] = {}


# === Global policy and processors (loaded once at startup) ===
policy = None
preprocessor = None
postprocessor = None
dataset_meta = None
device = None


def load_lerobot_policy(model_repo_id: str):
    global policy, preprocessor, postprocessor, dataset_meta, device

    logging.info(f"Loading LeRobot policy from: {model_repo_id}")

    # Load config from HF hub or local path
    cfg = PreTrainedConfig.from_pretrained(model_repo_id)

    # Create policy
    policy = make_policy(cfg, pretrained=True)
    device = get_safe_torch_device(cfg.device)
    policy.to(device).eval()

    # Create a dummy dataset to extract features (needed for preprocessing)
    # We mimic the observation/action space
    from lerobot.common.datasets.features import ObservationFeature, ActionFeature
    from lerobot.common.datasets.factory import make_dataset_stats

    # Dummy features (you may need to adjust based on your model)
    obs_features = {
        "observation.image": ObservationFeature(),
        "observation.state": ObservationFeature(shape=(STATES_LEN,)),
    }
    if cfg.input_shapes.get("observation.images.wrist"):
        obs_features["observation.images.wrist"] = ObservationFeature()

    act_features = {"action": ActionFeature(shape=(cfg.output_shapes["action"],))}

    # Build minimal meta
    dataset_meta = {
        "features": {**obs_features, **act_features},
        "stats": make_dataset_stats(obs_features),  # dummy stats
        "fps": 30,
    }

    # Create preprocessors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=model_repo_id,
        dataset_stats=dataset_meta["stats"],
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
        },
    )

    logging.info("Policy loaded successfully.")


def base64_to_pil(b64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")


def infer_action(image_b64: str, wrist_image_b64: Optional[str], state: List[float], prompt: Optional[str]) -> np.ndarray:
    global policy, preprocessor, postprocessor, dataset_meta, device

    if len(state) != STATES_LEN:
        raise ValueError(f"State length must be {STATES_LEN}, got {len(state)}")

    # Build observation dict matching policy input
    obs_dict = {
        "observation.image": base64_to_pil(image_b64),
        "observation.state": np.array(state, dtype=np.float32),
    }
    if wrist_image_b64:
        obs_dict["observation.images.wrist"] = base64_to_pil(wrist_image_b64)

    # Build dataset-style frame (for preprocessing compatibility)
    observation_frame = {}
    for key, value in obs_dict.items():
        prefix_key = key.replace(".", "_")  # e.g., "observation_image"
        observation_frame[prefix_key] = value

    # Predict action using LeRobot's predict_action
    action_values = predict_action(
        observation=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,  # disable AMP for simplicity
        task=prompt,
        robot_type=None,  # not needed for inference
    )

    # Convert to robot action format (just returns the array)
    robot_action = make_robot_action(action_values, dataset_meta["features"])

    # Extract action array
    if isinstance(robot_action, dict):
        action_array = robot_action["action"]
    else:
        action_array = robot_action

    return np.array(action_array)


# === FastAPI App ===

app = FastAPI(title="LeRobot Policy Inference Server", version="1.0.0")


@app.get("/")
def root():
    return {"message": "LeRobot Policy Server is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/act", response_model=InferenceResponse)
def act(request: InferenceRequest):
    try:
        logging.info(f"[act] Prompt: {request.prompt}, State len: {len(request.state)}")

        action = infer_action(
            image_b64=request.image,
            wrist_image_b64=request.wrist_image,
            state=request.state,
            prompt=request.prompt,
        )

        result = {
            "action": action.tolist(),
            "timestamp": 0.0,
        }
        return InferenceResponse(status=0, result=result, message="success")

    except Exception as e:
        logging.exception(e)
        return InferenceResponse(status=1, message=str(e))


# === Main ===

def main():
    model_repo_id = MODEL_REPO_ID
    load_lerobot_policy(model_repo_id)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = int(os.getenv("PORT", 8080))

    logging.info("Starting LeRobot HTTP server (host: %s, ip: %s, port: %d)", hostname, local_ip, port)
    logging.info("Visit http://%s:%d/docs for API documentation", local_ip, port)

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
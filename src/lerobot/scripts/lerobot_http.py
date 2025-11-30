# inference_server.py
import argparse
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional

import base64
from io import BytesIO
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.control_utils import predict_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 全局变量（在实际部署中建议用 lifespan 或依赖注入）
policy = None
preprocessor = None
postprocessor = None
dataset_meta = None
features = None
device = None
use_amp = False
task = ""
robot_type = ""

app = FastAPI(title="LeRobot Policy Inference Server")

class InferenceRequest(BaseModel):
    image: Optional[str] = None
    wrist_image: Optional[str] = None
    state: Optional[list] = None
    prompt: Optional[str] = None  # 如果请求没给，则使用 default_prompt

class InferenceResponse(BaseModel):
    status: int = 0
    message: str = "success"
    result: Dict[str, Any] = {}
# class ObservationInput(BaseModel):
#     observation: Dict[str, Any]  # e.g., {"image": "...", "state": [...]}

# class ActionOutput(BaseModel):
#     action: Dict[str, Any]


def base64_to_pil(b64_str: str):
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        img_array = np.array(image)
        return img_array
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")
    
@app.post("/act", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    global policy, preprocessor, postprocessor, features, device, use_amp, task, robot_type
    if policy is None:
        raise HTTPException(status_code=500, detail="Policy not loaded")

    try:
        logging.info(f"[act] Received request: {request.prompt}; {request.state}")

        # if len(request.state) != states_len:
        #     return InferenceResponse(status=1, message=f"invalid state length, need size: (1 x {states_len})")
        # 构造输入数据（根据你的 policy 接口调整）
        data = {
            "observation.images.fixed": base64_to_pil(request.image),
            "observation.images.handeye": base64_to_pil(request.wrist_image),
            "observation.state": np.array(request.state),
            "prompt": request.prompt,
        }
        
        action_values = predict_action(
            observation=data,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )
        robot_action = make_robot_action(action_values, features)
        logging.info(f"[act] Predicted action: {robot_action}")
        result = {
                "action": robot_action.tolist(),
            }
        return InferenceResponse(status=0, result=result, message="success")
    except Exception as e:
        logging.exception(e)
        raise InferenceResponse(status=1, message=f"{str(e)}")

def load_policy(policy_path: str, dataset_repo_id: str = None):
    global policy, preprocessor, postprocessor, dataset_meta, features, device, use_amp, task, robot_type

    # Load a dummy dataset to get meta & features (required for preprocessing)
    if dataset_repo_id:
        dataset = LeRobotDataset(dataset_repo_id)
    else:
        # 如果没有提供 dataset_repo_id，尝试从 policy config 中推断或使用默认
        dataset = LeRobotDataset.create(
            repo_id="dummy/dummy",
            fps=30,
            root="/tmp",
            robot_type="koch",
            features={},  # will be overridden
        )

    from lerobot.configs.policies import PreTrainedConfig
    cli_overrides = {}  # 可扩展：从命令行传入覆盖参数
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    policy_cfg.pretrained_path = policy_path

    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    device = get_safe_torch_device(policy.config.device)
    use_amp = policy.config.use_amp
    task = getattr(policy_cfg, "task", "")
    robot_type = getattr(policy_cfg, "robot_type", "")

    # 获取 features（简化：直接从 policy config 或 dataset）
    # 更稳健的方式是从 dataset 加载，但这里假设 policy 输出维度已知
    # 实际中建议传入 dataset_repo_id 来获取真实 features
    if hasattr(dataset, 'features') and dataset.features:
        features = dataset.features
    else:
        # fallback: assume action space from policy
        from lerobot.common.datasets.features import create_action_features
        features = {"action": create_action_features(policy.config.action_dim)}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=getattr(dataset.meta, "stats", {}),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )

    logging.info(f"Policy loaded from {policy_path} on device {device}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, required=True, help="Path or HF repo ID of the pretrained policy")
    parser.add_argument("--dataset-repo-id", type=str, default=None, help="Optional: HuggingFace dataset repo to load metadata/features")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_policy(args.policy_path, args.dataset_repo_id)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
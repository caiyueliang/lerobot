# inference_server.py
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

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

class ObservationInput(BaseModel):
    observation: Dict[str, Any]  # e.g., {"image": "...", "state": [...]}

class ActionOutput(BaseModel):
    action: Dict[str, Any]

@app.post("/predict", response_model=ActionOutput)
async def predict(obs_input: ObservationInput):
    global policy, preprocessor, postprocessor, features, device, use_amp, task, robot_type
    if policy is None:
        raise HTTPException(status_code=500, detail="Policy not loaded")

    try:
        action_values = predict_action(
            observation=obs_input.observation,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )
        robot_action = make_robot_action(action_values, features)
        return ActionOutput(action=robot_action)
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

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
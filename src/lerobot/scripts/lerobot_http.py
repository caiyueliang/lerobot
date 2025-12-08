# inference_server.py
import os
import argparse
import logging
import numpy as np
from PIL import Image
from pathlib import Path
import time
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

logging.basicConfig(level=logging.INFO)

# 全局变量（在实际部署中建议用 lifespan 或依赖注入）
states_len = os.getenv('STATES_LEN', 6)
policy = None
preprocessor = None
postprocessor = None
dataset_meta = None
features = None
device = None
use_amp = False
robot_type = ""
args = None

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


def base64_to_pil(b64_str: str, target_size=(480, 640)):
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        resized_image = image.resize(target_size)
        img_array = np.array(resized_image)
        # logging.warning(f"[act] Received image: {img_array.shape}")
        return img_array
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")
    
@app.post("/act", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    start = time.time()
    global policy, preprocessor, postprocessor, features, device, use_amp, robot_type, args
    if policy is None:
        raise HTTPException(status_code=500, detail="Policy not loaded")

    try:
        logging.info(f"[act] Received request: {request.prompt}; {request.state}")

        # if len(request.state) != states_len:
        #     return InferenceResponse(status=1, message=f"invalid state length, need size: (1 x {states_len})")

        if args.policy == "Pi0.5":
            data = {
                "observation.images.front": base64_to_pil(request.image),
                "observation.images.wrist": base64_to_pil(request.wrist_image),
                "observation.state": np.array(request.state),
            }
        else:
            data = {
                "observation.images.fixed": base64_to_pil(request.image),
                "observation.images.handeye": base64_to_pil(request.wrist_image),
                "observation.state": np.array(request.state),
            }

        task = request.prompt

        time_1 = time.time()
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
        time_2 = time.time()
        logging.warning(f"[act] action_values: {action_values}")
        # robot_action = make_robot_action(action_values, features)
        # logging.warning(f"[act] Predicted action: {robot_action}")
        # list_of_actions = [list(robot_action.values())]
        # logging.warning(f"[act] list_of_actions: {list_of_actions}")
        result = {
                "action": action_values.tolist(),
            }
        logging.warning(f"[time] 预处理: {time_1 - start:.4f}s, 推理: {time_2 - time_1:.4f}s, 总计: {time.time() - start:.4f}s")
        return InferenceResponse(status=0, result=result, message="success")
    except Exception as e:
        logging.exception(e)
        raise InferenceResponse(status=1, message=f"{str(e)}")

def load_policy(policy_path: str, dataset_repo_id: str = None):
    global policy, preprocessor, postprocessor, dataset_meta, features, device, use_amp, robot_type

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


# def find_first_params_dir(root_dir, keyword='params'):
#     """
#     在给定的根目录下，查找第一个包含名为 keyword('params') 的子文件夹的目录，
#     并返回该目录的路径。如果未找到，返回 None。
    
#     :param root_dir: 要搜索的根目录路径（字符串）
#     :return: 第一个包含 keyword 子目录的目录路径（字符串）或 None
#     """
#     for dirpath, dirnames, _ in os.walk(root_dir):
#         if keyword in dirnames:
#             return dirpath
#     return None
def find_first_matching_dir(root_dir, required_file, required_dir=None):
    """
    在给定的根目录下，查找第一个满足以下条件的目录：
    1. 目录下包含名为 required_file 的文件；
    2. 如果 required_dir 不为 None 且非空，则该目录的路径中必须包含名为 required_dir 的某一级目录名。

    参数:
        root_dir (str 或 Path): 要搜索的根目录。
        required_file (str): 必须存在的文件名（仅文件名，不带路径）。
        required_dir (str, optional): 路径中必须包含的目录名。若为 None 或空字符串，则跳过此条件。

    返回:
        Path 或 None: 找到的第一个匹配目录，否则返回 None。
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"根目录 {root_dir} 不存在或不是目录")

    for current_dir in [root] + list(root.rglob('*')):
        if not current_dir.is_dir():
            continue

        # 检查路径是否包含 required_dir（如果指定了）
        if required_dir:
            if required_dir not in current_dir.parts:
                continue

        # 检查该目录下是否有 required_file
        if (current_dir / required_file).is_file():
            return current_dir

    return None

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None, help="Path or HF repo ID of the pretrained policy")
    parser.add_argument("--dataset-repo-id", type=str, default=None, help="Optional: HuggingFace dataset repo to load metadata/features")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()


    # 初始化
    if args.policy is None:
        args.policy = os.getenv("POLICY", None)
    if args.policy_path is None:
        args.policy_path = os.getenv("MODEL_PATH")
        if args.policy_path:
            logging.warning(f"[main] 使用环境变量 MODEL_PATH: {args.policy_path}")
        else:
            logging.warning("[main] 未提供 --policy-path，且环境变量 MODEL_PATH 未设置。")
    if args.dataset_repo_id is None:
        args.dataset_repo_id = os.getenv("DATASET_PATH")
        if args.dataset_repo_id:
            logging.warning(f"[main] 使用环境变量 DATASET_PATH: {args.dataset_repo_id}")
        else:
            logging.warning("[main] 未提供 --dataset-repo-id，且环境变量 DATASET_PATH 未设置。")
    args.host = os.getenv("HOST", args.host)
    args.port = int(os.getenv("PORT", args.port))

    # 动态获取模型加载路径
    required_file = 'config.json'
    base_dir = find_first_matching_dir(
        root_dir=args.policy_path, 
        required_file=required_file)
    if base_dir:
        logging.warning(f"[main] 目录: {base_dir} 中找到 '{required_file}' 子文件夹，使用该目录作为模型路径。")
        args.policy_path = base_dir
        logging.warning(f"[main] new args: {args}")
    else:
        logging.warning(f"[main] 目录: {args.policy_path} 中未找到 '{required_file}' 子文件夹，请检查模型路径。")
        exit(1)

    logging.warning(f"[main] args: {args}")
    load_policy(args.policy_path, args.dataset_repo_id)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
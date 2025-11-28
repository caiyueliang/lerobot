# serve_policy_http.py
import os
import sys
import dataclasses
import enum
import logging
import socket
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import tyro
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# from openpi.policies import policy as _policy
# from openpi.policies import policy_config as _policy_config
# from openpi.training import config as _config
import logging
import base64
from io import BytesIO
from PIL import Image

# 新增导入 - 从record.py中提取的模块
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.configs import parser

# 或者提供默认值
states_len = os.getenv('STATES_LEN', 8)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('lerobot_http.log')
    ]
)
logger = logging.getLogger("lerobot_http")

class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    env: EnvMode = EnvMode.LIBERO
    default_prompt: Optional[str] = None
    port: int = 8080
    record: bool = False
    model_path: str = ""  # 新增参数 - 模型路径
    # policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    # 新增参数 - 从record.py中提取
    device: str = "cuda"  # 推理设备
    use_amp: bool = False  # 是否使用自动混合精度


# # Default checkpoints that should be used for each environment.
# DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
#     EnvMode.ALOHA: Checkpoint(
#         config="pi05_aloha",
#         dir="gs://openpi-assets/checkpoints/pi05_base",
#     ),
#     EnvMode.ALOHA_SIM: Checkpoint(
#         config="pi0_aloha_sim",
#         dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
#     ),
#     EnvMode.DROID: Checkpoint(
#         config="pi05_droid",
#         dir="gs://openpi-assets/checkpoints/pi05_droid",
#     ),
#     EnvMode.LIBERO: Checkpoint(
#         config="pi05_libero",
#         dir="gs://openpi-assets/checkpoints/pi05_libero",
#     ),
# }


# def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
#     if checkpoint := DEFAULT_CHECKPOINT.get(env):
#         return _policy_config.create_trained_policy(
#             _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
#         )
#     raise ValueError(f"Unsupported environment mode: {env}")


# def create_policy(args: Args) -> _policy.Policy:
#     logging.warning(f"[create_policy] args.policy: {args.policy}")
#     match args.policy:
#         case Checkpoint():
#             return _policy_config.create_trained_policy(
#                 _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
#             )
#         case Default():
#             return create_default_policy(args.env, default_prompt=args.default_prompt)


# === 新增：从record.py中提取的模型启动代码 ===

def create_lerobot_policy(args: Args) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """
    从record.py中提取的模型启动代码
    返回: (policy, preprocessor, postprocessor)
    """
    # 构建配置
    policy_cfg = PreTrainedConfig(
        pretrained_path=args.policy.dir,
        device=args.device,
        use_amp=args.use_amp
    )
    
    # 创建空数据集用于获取统计信息（如果需要）
    # 注意：这里可能需要根据实际情况调整
    dataset_stats = None
    
    # 创建策略模型
    policy = make_policy(policy_cfg, ds_meta=None)  # ds_meta可以为None，如果不需要统计信息
    
    # 创建预处理和后处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": {}},  # 可根据需要调整
        },
    )
    
    # 重置策略和处理器
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    
    return policy, preprocessor, postprocessor


def predict_lerobot_action(
    observation: dict,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    task: str = None,
    robot_type: str = None
) -> dict:
    """
    从record.py中提取的预测action代码
    """
    device = get_safe_torch_device(policy.config.device)
    
    # 调用predict_action函数
    action_values = predict_action(
        observation=observation,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=policy.config.use_amp,
        task=task,
        robot_type=robot_type,
    )
    
    # 将action转换为合适的格式
    # 注意：这里可能需要根据实际的数据集特征进行调整
    features = {}  # 需要根据实际情况提供数据集特征
    act_processed = make_robot_action(action_values, features)
    
    return act_processed


# === FastAPI Model Definitions ===

class InferenceRequest(BaseModel):
    image: Optional[str] = None
    wrist_image: Optional[str] = None
    state: Optional[list] = None
    prompt: Optional[str] = None  # 如果请求没给，则使用 default_prompt

class InferenceResponse(BaseModel):
    status: int = 0
    message: str = "success"
    result: Dict[str, Any] = {}


def find_first_params_dir(root_dir):
    """
    在给定的根目录下，查找第一个包含名为 'params' 的子文件夹的目录，
    并返回该目录的路径。如果未找到，返回 None。
    
    :param root_dir: 要搜索的根目录路径（字符串）
    :return: 第一个包含 'params' 子目录的目录路径（字符串）或 None
    """
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'params' in dirnames:
            return dirpath
    return None

def base64_to_pil(b64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")
    
def base64_to_numpy(b64_str: str) -> np.ndarray:
    """将base64图像转换为numpy数组"""
    pil_image = base64_to_pil(b64_str)
    return np.array(pil_image)


# === Main Server Logic ===

class LeRobotPolicyServer:
    def __init__(self, args: Args):
        self.args = args
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.robot_type = "default"  # 根据实际情况设置
        
    def initialize_policy(self):
        """初始化LeRobot策略模型"""
        try:
            logging.info("Initializing LeRobot policy...")
            # self.policy, self.preprocessor, self.postprocessor = create_lerobot_policy(self.args)
            self.policy = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=self.args.model_path, 
                # cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = self.args.model_path
            logging.info("LeRobot policy initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize LeRobot policy: {e}")
            raise
    
    def infer(self, data: dict) -> dict:
        """推理接口"""
        if self.policy is None:
            raise RuntimeError("Policy not initialized")
        
        # 构建观察数据（根据LeRobot的格式要求）
        observation = self._build_observation(data)
        
        # 预测action
        action = predict_lerobot_action(
            observation=observation,
            policy=self.policy,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            task=data.get("prompt"),
            robot_type=self.robot_type
        )
        
        return {
            "actions": action,
            "timestamp": time.time()
        }
    
    def _build_observation(self, data: dict) -> dict:
        """构建LeRobot格式的观察数据"""
        observation = {}
        
        # 添加图像观察
        if data.get("observation/image"):
            observation["observation.image"] = base64_to_numpy(data["observation/image"])
        if data.get("observation/wrist_image"):
            observation["observation.wrist_image"] = base64_to_numpy(data["observation/wrist_image"])
        
        # 添加状态观察
        if data.get("observation/state") is not None:
            observation["observation.state"] = np.array(data["observation/state"], dtype=np.float32)
        
        # 添加提示
        if data.get("prompt"):
            observation["task"] = data["prompt"]
        
        return observation


def main(args: Args) -> None:
    logging.warning(f"[main] old args: {args}")

    # base_dir = find_first_params_dir(root_dir=args.policy.dir)
    # if base_dir:
    #     logging.warning(f"[main] 目录: {base_dir} 中找到 'params' 子文件夹，使用该目录作为模型路径。")
    #     args.policy.dir = base_dir
    #     logging.warning(f"[main] new args: {args}")
    # else:
    #     logging.warning(f"[main] 目录: {args.policy.dir} 中未找到 'params' 子文件夹，请检查模型路径。")
    #     exit(1)

    # 创建策略服务器实例
    policy_server = LeRobotPolicyServer(args)
    
    # 初始化策略模型
    try:
        policy_server.initialize_policy()
    except Exception as e:
        logging.error(f"Failed to initialize policy: {e}")
        return

    # 创建 FastAPI 应用
    app = FastAPI(
        title="OpenPI Policy Inference Server",
        description="Serving robot policy models via HTTP.",
        version="1.0.0"
    )

    @app.get("/")
    def root():
        return {"message": "OpenPI Policy Server is running", "env": args.env.value}

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/act", response_model=InferenceResponse)
    def act(request: InferenceRequest):
        try:
            logging.info(f"[act] Received request: {request.prompt}; {request.state}")

            if request.state and len(request.state) != states_len:
                return InferenceResponse(status=1, message=f"invalid state length, need size: (1 x {states_len})")
            
            # 构造输入数据
            data = {
                "observation/image": request.image,
                "observation/wrist_image": request.wrist_image,
                "observation/state": request.state,
                "prompt": request.prompt or args.default_prompt,
            }

            # 调用策略模型推理
            result = policy_server.infer(data)
            
            # 处理返回结果
            actions = result.get("actions", {})
            timestamp = result.get("timestamp", 0.0)

            # 将action转换为列表格式（根据实际需要调整）
            if isinstance(actions, dict):
                # 如果是字典格式的action，提取数值部分
                action_values = []
                for key in sorted(actions.keys()):
                    if isinstance(actions[key], (int, float)):
                        action_values.append(actions[key])
                    elif isinstance(actions[key], np.ndarray):
                        action_values.extend(actions[key].flatten().tolist())
                action_list = action_values
            elif isinstance(actions, np.ndarray):
                action_list = actions.tolist()
            else:
                action_list = list(actions) if hasattr(actions, '__iter__') else [actions]

            result = {
                "action": action_list[:5] if len(action_list) >= 5 else action_list,
                "timestamp": timestamp,
            }
            return InferenceResponse(status=0, result=result, message="success")

        except Exception as e:
            logging.exception(e)
            return InferenceResponse(status=1, message=f"{str(e)}")

    # 启动前打印信息
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting HTTP server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)
    logging.info("Visit http://%s:%d/docs for API documentation", local_ip, args.port)

    # 使用 Uvicorn 运行应用
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
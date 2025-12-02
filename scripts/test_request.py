#!/usr/bin/env python3
"""
多目+关节状态推理客户端（支持命令行传参）
"""
import argparse
import base64
import requests
import time
import logging
from pathlib import Path

# -------------------- 命令行解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="VLA 推理客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/act",
        help="推理服务完整 URL，默认 http://localhost:8080/act",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="可选 JWT/Bearer token，传入后会自动加到请求头",
    )
    return parser.parse_args()

# -------------------- 图像工具 --------------------
def image_to_base64(img_path: str) -> str:
    """读取图像并转为 base64 字符串"""
    with open(img_path, "rb") as f:
        image_base64= base64.b64encode(f.read()).decode("utf-8")
        # logging.warning(f"[INFO] {img_path} -> {image_base64} \n\n")
    return image_base64

# -------------------- 主流程 --------------------
def main():
    args = parse_args()

    # 1. 本地图像路径（按需修改）
    base_path = "./images/iros_clear_table_in_the_restaurant_20251028_111828/"
    head_img_path = base_path + "head_00050.png"
    wrist_left_img_path = base_path + "wrist_r_00050.png"

    start = time.time()
    # 2. 构造 JSON 请求体
    request_data = {
        "image": image_to_base64(head_img_path),
        "wrist_image": image_to_base64(wrist_left_img_path),
        # "state": [16.13212, 59.35374, -55.23179, 95.854485, 6.9108667, 34.36893],
        "state": [0.13212, 0.35374, -0.23179, 0.854485, 0.9108667, 0.36893],
        "prompt": "Pick up the bowl on the table near the right arm with the right arm.", 
    }
    logging.warning(f"state: {request_data['state']}")
    
    # 3. 构造请求头
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"{args.token}"

    time_1 = time.time()
    # 4. 发送请求
    logging.warning(f"[INFO] POST -> {args.url}")
    response = requests.post(args.url, json=request_data, headers=headers)
    time_2 = time.time()
    logging.warning(f"[time] 预处理: {time_1 - start:.4f}s, 请求: {time_2 - time_1:.4f}s, 总计: {time.time() - start:.4f}s")

    # 5. 处理返回
    if response.status_code == 200:
        result = response.json()
        logging.warning(f"response: {result}")

        if result["status"] == 0:
            logging.warning(f"Action: {result['result']['action']}")
        else:
            logging.warning(f"Error message: {result}")
    else:
        logging.warning(f"Error Code: {response.status_code}")
        logging.warning(f"Error: {response.json()}")

if __name__ == "__main__":
    main()
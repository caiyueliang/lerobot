#!/usr/bin/env python3
"""
多目+关节状态推理客户端（支持命令行传参）
"""
import argparse
import base64
import requests
import logging
from pathlib import Path

# -------------------- 命令行解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="VLA 推理客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/act",
        help="推理服务完整 URL，默认 http://localhost:8000/act",
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

    # 2. 构造 JSON 请求体
    request_data = {
        "image": image_to_base64(head_img_path),
        "wrist_image": image_to_base64(wrist_left_img_path),
        "state": [16.13212, 59.35374, -55.23179, 95.854485, 6.9108667, 34.36893],
        "prompt": "Pick up the bowl on the table near the right arm with the right arm.", 
    }
    print("state:", request_data["state"])
    
    # 3. 构造请求头
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"{args.token}"

    # 4. 发送请求
    print(f"[INFO] POST -> {args.url}")
    response = requests.post(args.url, json=request_data, headers=headers)

    # 5. 处理返回
    if response.status_code == 200:
        result = response.json()
        print("response:", result)

        if result["status"] == 0:
            print("Action:", result["result"]["action"])
        else:
            print("Error message:", result)
    else:
        print("Error Code: ", response.status_code)
        print("Error:", response.json())

if __name__ == "__main__":
    main()
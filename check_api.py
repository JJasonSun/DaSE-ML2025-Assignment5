#!/usr/bin/env python3
"""
check_api.py - 验证项目根目录下的 `.env` 中 API_KEY 与 BASE_URL 是否可用
用法: python check_api.py

脚本会:
 - 加载 `.env`
 - 检查 `API_KEY` 和 `BASE_URL` 是否存在
 - 用 OpenAI 客户端尝试调用 `models.list()` 以验证凭据与可连通性
"""

import os
import sys
import time
from dotenv import load_dotenv
from urllib.parse import urlparse

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def main():
    load_dotenv()

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key or not base_url:
        print("❌ .env 中缺少 API_KEY 或 BASE_URL，请检查。")
        sys.exit(2)

    if OpenAI is None:
        print("⚠️ 未安装 'openai' 库。请先运行: pip install -r requirements.txt")
        sys.exit(3)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 1. 验证 models.list()
        resp = client.models.list()
        if hasattr(resp, "data"):
            models = resp.data
        elif isinstance(resp, list):
            models = resp
        else:
            models = resp.get("data", [])

        print(f"✅ API 可访问。返回了 {len(models)} 个模型（0 也可能表示无可见模型）。")

        print("\n-- 模型列表 --")
        model_ids = []
        for i, m in enumerate(models, 1):
            mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else str(m))
            model_ids.append(mid)
            print(f"  {i}. {mid}")

        test_model = os.getenv("TEST_MODEL")
        if not test_model:
            print("❌ 未找到 TEST_MODEL 环境变量，跳过推理测试。")
            sys.exit(0)

        print(f"\n尝试对模型 '{test_model}' 发送一次普通 ChatCompletion 请求（不启用 thinking）。")

        # 2. 普通 Chat Completion（非流式、不启用 thinking）
        messages = [
            {"role": "system", "content": "You are a helpful test assistant."},
            {"role": "user", "content": "介绍一下你自己"}
        ]

        print("DEBUG: 发送 chat completion 请求...")
        completion = client.chat.completions.create(
            model=test_model,
            messages=messages,
            temperature=0,
            max_tokens=256,
            # 关键：通过 extra_body 下发给“兼容 OpenAI 的第三方服务端”
            extra_body={
                "enable_thinking": False,  # 关键：强制不要推理输出（适配你这个服务端）
            },
        )  # type: ignore


        print("DEBUG: 收到响应，解析中...")
        time.sleep(1)

        try:
            content = completion.choices[0].message.content
        except Exception:
            content = str(completion)

        print("DEBUG finish_reason:", completion.choices[0].finish_reason)
        print("DEBUG message:", completion.choices[0].message)


        print("\n-- 模型响应 --")
        print(content)

        sys.exit(0)

    except Exception as e:
        import traceback
        print("❌ 调用 API 失败:", str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

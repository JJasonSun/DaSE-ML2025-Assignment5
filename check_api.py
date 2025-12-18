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
from dotenv import load_dotenv

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
        resp = client.models.list()
        models = []
        # 兼容不同 SDK 返回格式
        if hasattr(resp, "data"):
            models = resp.data
        elif isinstance(resp, list):
            models = resp
        else:
            # 尝试从 dict 中提取
            try:
                models = resp.get("data", [])
            except Exception:
                models = []

        print(f"✅ API 可访问。返回了 {len(models)} 个模型（若返回为0也可能表明权限/模型不可见）。")

        # 列举所有模型
        print('\n-- 模型列表 --')
        model_ids = []
        for i, m in enumerate(models, 1):
            mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else str(m))
            model_ids.append(mid)
            print(f"  {i}. {mid}")

        # 选择用于测试的模型（优先使用 TEST_MODEL 环境变量）
        test_model = os.getenv('TEST_MODEL') or (model_ids[0] if model_ids else None)
        if not test_model:
            print("❌ 未找到用于测试的模型。")
            sys.exit(0)

        print(f"\n尝试对模型 '{test_model}' 发送一次简单请求以验证推理能力。")

        try:
            # 简单的 Chat 完成调用（同步）
            messages = [
                {"role": "system", "content": "You are a helpful test assistant."},
                {"role": "user", "content": "你是谁？"}
            ]
            print('DEBUG: 发送 chat completion 请求...')
            completion = client.chat.completions.create(
                model=test_model,
                messages=messages,
                temperature=0,
                max_tokens=32
            )

            # 尝试解析响应
            try:
                content = completion.choices[0].message.content
            except Exception:
                # 兼容不同响应格式
                content = str(completion)

            print("\n-- 推理响应 --")
            print(content)
            sys.exit(0)

        except Exception as e:
            import traceback
            print("❌ 推理请求失败:", str(e))
            traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        print("❌ 调用 API 失败:", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

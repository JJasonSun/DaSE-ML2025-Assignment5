#!/usr/bin/env python3
"""
check_api.py - 验证项目根目录下的 `.env` 中 API_KEY 与 BASE_URL 是否可用
用法: python check_api_modified.py

脚本会:
 - 加载 `.env`
 - 检查 `API_KEY`、`BASE_URL` 是否存在（以及可选的 `TEST_MODEL`）
 - 用 OpenAI 客户端调用 `models.list()` 验证凭据与连通性
 - 用 `chat.completions.create()` 发送一次简单对话，验证模型能否正常输出文本

说明（兼容性）：
 - 一些“OpenAI-compatible”服务（例如你测试的 GLM/Qwen 网关）可能返回 `message.reasoning_content`，
   而 `message.content` 为空。本脚本会自动兜底读取 reasoning_content，避免“看起来像空响应”。
 - 若你的服务端有“thinking/推理模式”开关，可通过 extra_body 下发：
   - 默认会下发 {"enable_thinking": False}（可通过环境变量 ENABLE_THINKING=1 来开启）
"""

import os
import sys
import time
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def extract_text_from_chat_completion(completion) -> str:
    """
    兼容提取：优先取 message.content；若为空，再取 message.reasoning_content；再检查 tool_calls。
    """
    try:
        choice0 = completion.choices[0]
    except Exception:
        return str(completion)

    finish_reason = getattr(choice0, "finish_reason", None)
    msg = getattr(choice0, "message", None)
    if msg is None:
        return f"[no message | finish_reason={finish_reason}] raw={str(completion)}"

    # 1) 常规 content
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    # 2) 推理/思考字段（某些 OpenAI-compatible 服务会返回）
    reasoning = getattr(msg, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # 3) 工具调用（content 可能为空）
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        return f"[tool_calls detected | finish_reason={finish_reason}] {tool_calls}"

    return f"[empty content | finish_reason={finish_reason}] raw_message={str(msg)}"


def main():
    load_dotenv()

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    test_model = os.getenv("TEST_MODEL")

    if not api_key or not base_url:
        print("❌ .env 中缺少 API_KEY 或 BASE_URL，请检查。")
        sys.exit(2)

    if OpenAI is None:
        print("⚠️ 未安装 'openai' 库。请先运行: pip install -r requirements.txt")
        sys.exit(3)

    # 是否开启“thinking/推理模式”（默认关闭）
    enable_thinking = _bool_env("ENABLE_THINKING", default=False)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 1) 验证 models.list()
        resp = client.models.list()
        if hasattr(resp, "data"):
            models = resp.data
        elif isinstance(resp, list):
            models = resp
        else:
            models = resp.get("data", [])

        print(f"✅ API 可访问。返回了 {len(models)} 个模型（0 也可能表示无可见模型）。")

        print("\n-- 模型列表 --")
        for i, m in enumerate(models, 1):
            mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else str(m))
            print(f"  {i}. {mid}")

        if not test_model:
            print("\nℹ️ 未找到 TEST_MODEL 环境变量，跳过对话测试。")
            print("   你可以在 .env 里加上：TEST_MODEL=glm-4.5（或其它模型ID）")
            sys.exit(0)

        print(f"\n尝试对模型 '{test_model}' 发送一次普通 ChatCompletion 请求。")
        print(f"Thinking/推理模式：{'开启' if enable_thinking else '关闭'}（通过 ENABLE_THINKING 控制）")

        messages = [
            {"role": "system", "content": "You are a helpful test assistant."},
            {"role": "user", "content": "介绍一下你自己"}
        ]

        # OpenAI-compatible 服务端的扩展参数：通过 extra_body 下发
        extra_body = {"enable_thinking": enable_thinking}

        print("DEBUG: 发送 chat completion 请求...")
        completion = client.chat.completions.create(
            model=test_model,
            messages=messages,
            temperature=0,
            max_tokens=256,
            stream=False,
            extra_body=extra_body,
        )  # type: ignore

        print("DEBUG: 收到响应，解析中...")
        time.sleep(0.3)

        # 打印一些对新手很有用的调试信息
        try:
            finish_reason = completion.choices[0].finish_reason
        except Exception:
            finish_reason = None

        print("DEBUG finish_reason:", finish_reason)

        # 提取文本（兼容 reasoning_content / tool_calls）
        content = extract_text_from_chat_completion(completion)

        print("\n-- 模型响应 --")
        print(content)

        # 额外提示：如果被截断，告诉用户如何处理
        if finish_reason == "length":
            print("\nℹ️ 提示：finish_reason=length 表示输出因 max_tokens 限制被截断。")
            print("   你可以把 max_tokens 调大一些（例如 512 或 1024）。")

        sys.exit(0)

    except Exception as e:
        import traceback
        print("❌ 调用 API 失败:", str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

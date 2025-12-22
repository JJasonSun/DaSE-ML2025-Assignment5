'''Example response with thinking enabled:
{
    "id": "72ff76ef79544fe599ae9e3c62742463",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Machine learning is a field of artificial intelligence that enables computers to learn patterns and make decisions from data without being explicitly programmed for each task. By using algorithms that improve automatically through experience—such as supervised, unsupervised, and reinforcement learning—it can solve problems ranging from image recognition to predictive analytics. This capability is transforming industries by automating processes, personalizing experiences, and uncovering insights that were previously hidden in massive datasets.",
                "role": "assistant",
                "tool_calls": null,
                "reasoning_content": "We need to produce three sentences introducing machine learning. Should be concise, clear, maybe cover definition, purpose, and impact. Three sentences only. No extra."
            }
        }
    ],
    "created": 1765962208,
    "model": "mimo-v2-flash",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 120,
        "prompt_tokens": 60,
        "total_tokens": 180,
        "completion_tokens_details": {
            "reasoning_tokens": 32
        },
        "prompt_tokens_details": null
    }
}
'''

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="https://api.xiaomimimo.com/v1"
)

completion = client.chat.completions.create(
    model="mimo-v2-flash",
    messages=[
        {
            "role": "system",
            "content": "You are MiMo, an AI assistant developed by Xiaomi. Today is date: Tuesday, December 16, 2025. Your knowledge cutoff date is December 2024."
        },
        {
            "role": "user",
            "content": "Introduce machine learning in three sentences."
        }
    ],
    max_completion_tokens=1024,
    extra_body={
        "thinking": {"type": "enabled"}
    }
)

print(completion.model_dump_json())
'''
函数调用的示例，使用小米的 MiMo-V2-Flash 模型。
响应示例：
{
    "id": "aec4f0e8d9cf4446a73b741a26f6ef6f",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "content": "I'll check the current weather in Boston for you.",
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_f3bb2876c4a54da69014b3dd",
                        "function": {
                            "arguments": "{\"location\": \"Boston, MA\"}",
                            "name": "get_current_weather"
                        },
                        "type": "function"
                    }
                ],
                "reasoning_content": null
            }
        }
    ],
    "created": 1765960589,
    "model": "mimo-v2-flash",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 35,
        "prompt_tokens": 351,
        "total_tokens": 386,
        "completion_tokens_details": {
            "reasoning_tokens": 0
        },
        "prompt_tokens_details": null
    }
}
'''

import os
from openai import OpenAI

model_name = os.environ.get("MODEL_NAME", "mimo-v2-flash")

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("BASE_URL")
)

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": "You are MiMo, an AI assistant developed by Xiaomi. Today is date: Tuesday, December 16, 2025. Your knowledge cutoff date is December 2024."
        },
        {
            "role": "user",
            "content": "What is the weather like in Boston today?"
        }
    ],
    max_completion_tokens=1024,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    tool_choice="auto"
)

print(completion.model_dump_json())
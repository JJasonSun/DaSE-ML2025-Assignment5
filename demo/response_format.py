'''
结构化输出示例：从非结构化文本中提取特定字段并以JSON格式返回
响应示例：
{
    "id": "9989ed6b3970459b950a769fe814f217",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "{\n  \"name\": \"Zhang San\",\n  \"age\": 28,\n  \"email\": \"zhangsan@test.com\",\n  \"birthday\": \"1996-05-12\"\n}",
                "role": "assistant",
                "tool_calls": null,
                "reasoning_content": null
            }
        }
    ],
    "created": 1765960917,
    "model": "mimo-v2-flash",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 47,
        "prompt_tokens": 147,
        "total_tokens": 194,
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
            "content": "My name is Zhang San, I am 28 years old, my email address is zhangsan@test.com, and my birthday is May 12, 1996. Please return the results in the following JSON format. Note: The birthday must be in YYYY-MM-DD format. {\"name\": \"string or null\", \"age\": \"int or null\", \"email\": \"string or null\", \"birthday\": \"string or null\"}"
        }
    ],
    max_completion_tokens=1024,
    response_format={
        "type": "json_object"
    }
)

print(completion.model_dump_json())
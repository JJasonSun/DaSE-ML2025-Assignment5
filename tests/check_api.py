'''
一个简单的测试脚本，用于测试API是否可用，以及打印可用模型列表。
请确保在运行此脚本前，已经正确设置了`.env`文件中的`ECNU_API_KEY`、`ECNU_BASE_URL`和`MODEL_NAME`变量。
'''

from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ecnu_constants import DEFAULT_ECNU_BASE_URL, ECNU_MAIN_MODEL_NAME

load_dotenv()

api_key = os.getenv('ECNU_API_KEY')
base_url = os.getenv('ECNU_BASE_URL', DEFAULT_ECNU_BASE_URL)

if not api_key:
    raise ValueError("ECNU_API_KEY is not set in the environment. Please add it to .env before running.")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


def print_available_models():
    try:
        response = client.models.list()
        print("Available models:")
        for model in response.data:
            print(f"  - {model.id}")
    except Exception as exc:
        print("Failed to list models:", exc)


print_available_models()

model_name = os.getenv('MODEL_NAME')
if not model_name:
    model_name = ECNU_MAIN_MODEL_NAME

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "介绍一下自己。"}
    ],
    top_p=0.95,
    temperature=1,
    max_tokens=16000
)

# 解析响应并打印结果
message = completion.choices[0].message

# 如果存在推理内容，则打印推理过程
reasoning_content = getattr(message, 'reasoning_content', None)
if reasoning_content:
    print("--- Reasoning Process ---")
    print(reasoning_content)
    print("-------------------------")

print(completion.model_dump_json()) # 打印完整的响应JSON以供调试

print(f"------- Final Answer of {model_name} --------")

print(message.content) # 最终回答
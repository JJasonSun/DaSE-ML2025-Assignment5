'''
一个简单的测试脚本，用于测试API是否可用，以及打印可用模型列表。
请确保在运行此脚本前，已经正确设置了`.env`文件中的`API_KEY`、`BASE_URL`和`MODEL_NAME`变量。
'''

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key= os.getenv('API_KEY'),
    base_url= os.getenv('BASE_URL')
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
    raise ValueError("MODEL_NAME is not set in the environment. Please add it to .env or set it before running.")


extra_body = {}

# 除了 ecnu 的模型都要支持启用/禁用思考模式
if "ecnu" not in model_name.lower():
    extra_body = {
        "thinking": {
            "type": "enabled" # 可选值: "enabled", "disabled"
        }
    }

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个实用的ai助手"},
        {"role": "user", "content": "请简要介绍一下你自己并随机回复一个数字"}
    ],
    top_p=0.7,
    temperature=0.3,
    extra_body=extra_body,
    stream=False,  # 禁用流式响应
    max_tokens=512
)

# 解析响应并打印结果
message = completion.choices[0].message

# 如果存在推理内容，则打印推理过程
if hasattr(message, 'reasoning_content') and message.reasoning_content:
    print("--- Reasoning Process ---")
    print(message.reasoning_content)
    print("-------------------------")

print(completion.model_dump_json()) # 打印完整的响应JSON以供调试

print(f"------- Final Answer of {model_name} --------")

print(message.content) # 最终回答
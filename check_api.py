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

# 所有的 ecnu 模型都不支持 thinking 参数
if not model_name.lower().startswith("ecnu"):
    extra_body = {
        "thinking": {
            "type": "enabled" # 可选值: "auto", "enabled", "disabled"
        }
    }

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个高精度的检索和计算助手。请根据提供的上下文回答问题，并使用中文回答,直接给出最终答案。"},
        {"role": "user", "content": """Context:
1. Asset Vault Alpha contains a critical resource count of 4931305872608785 units. Security clearance level: OMEGA.
2. Operational Database Beta tracks emergency reserves at 4106935655129089 units. Classification: EYES ONLY.
3. Strategic multiplier designation GAMMA-394 must be applied to all differential calculations.
4. The contingency divisor ZETA-41 is authorized for final resource allocation.

Question: As the lead resource coordinator for Project Chimera, you need to calculate the final emergency deployment allocation. Access the classified asset vault numbers (Alpha and Beta), determine the resource differential, apply the strategic multiplier Gamma, then divide by the contingency divisor Zeta using integer division. What is the precise deployment figure? 请使用中文回答。"""}
    ],
    top_p=0.95,
    temperature=1,
    extra_body=extra_body,
    max_tokens=16000
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
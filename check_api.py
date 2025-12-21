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

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一个ai助手"},
        {"role": "user", "content": "请直接回复一个1-10的数字"}
    ],
    top_p=0.7,
    temperature=0.9
)

print(completion.choices[0].message.content)
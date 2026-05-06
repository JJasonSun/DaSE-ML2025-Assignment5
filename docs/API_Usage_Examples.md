# 大模型 API 调用示例代码文档

本文档整理了各种大语言模型 API 核心功能的调用示例（基于标准 `openai` SDK）。您可以直接参考这些示例将对应能力接入到您自己的业务中。

## 1. 深度思考 (Deep Thinking / Reasoning) 发送

部分最新模型（如特定的 reasoning 模型）支持在返回最终答案前使用额外的 Token 展现内部思维链（Thinking Process）。您可以通过 `extra_body` 参数去启用此功能。

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="https://api.xiaomimimo.com/v1" # 以小米API为例
)

completion = client.chat.completions.create(
    model="mimo-v2-flash",
    messages=[
        {"role": "system", "content": "You are an AI assistant. Today is date: Tuesday, December 16, 2025."},
        {"role": "user", "content": "Introduce machine learning in three sentences."}
    ],
    max_completion_tokens=1024,
    # 通过 extra_body 开启思考模式
    extra_body={
        "thinking": {"type": "enabled"}
    }
)

# 获取完整的大模型响应包含 reasoning_content
print(completion.model_dump_json())
```
*注：在某些模型的响应结构中，思考内容将位于 `message.reasoning_content` 中，而最终回答在 `message.content` 里。*

---

## 2. 文本嵌入 / 向量化 (Text Embeddings)

获取文本的稠密向量（Dense Vector）表示，用于语义检索、知识库搭建或 RAG 场景。

```python
from openai import OpenAI
  
client = OpenAI(
    base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key="your-api-key"
)

# 单个或多个文本嵌入请求（最多传入字符串数组）
response = client.embeddings.create(
    model="ecnu-embedding-small",
    input=["Hello world", "good work"]  # 输入文本。单条通常不超 8192 个字符
)

# 获取第一个文本“Hello world”的向量
print(response.data[0].embedding)

# 获取第二个文本“good work”的向量
print(response.data[1].embedding)
```

---

## 3. 工具调用 (Function Calling / Tool Use)

大语言模型能够在执行时决定是否需要调用外部函数，并要求生成该函数的 JSON 格式入参。这对于搭建 Agent 工作流至关重要。

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("BASE_URL")
)

completion = client.chat.completions.create(
    model=os.environ.get("MODEL_NAME", "mimo-v2-flash"),
    messages=[
        {"role": "user", "content": "What is the weather like in Boston today?"}
    ],
    max_completion_tokens=1024,
    # 定义模型可用的函数及对应的参数 Schema
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
    tool_choice="auto" # 允许大模型自动决定是否调用工具
)

# 如果模型决定使用工具，其响应的 finish_reason 会是 'tool_calls'
print(completion.choices[0].message.tool_calls)
```

---

## 4. 强制 JSON 格式化输出 (JSON Mode)

强制模型在回应的 `content` 中返回符合标准 JSON 格式的一段字符串。常用于信息抽取任务。

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("BASE_URL")
)

completion = client.chat.completions.create(
    model=os.environ.get("MODEL_NAME", "mimo-v2-flash"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", 
            "content": "My name is Zhang San, I am 28 years old, my email address is zhangsan@test.com, and my birthday is May 12, 1996. Please return the results in the following JSON format. {\"name\": \"\", \"age\": 0, \"email\": \"\", \"birthday\": \"\"}"
        }
    ],
    max_completion_tokens=1024,
    # 要求必须输出 JSON 对象
    response_format={
        "type": "json_object"
    }
)

# 打印最终提取出来的 JSON 字符串
print(completion.choices[0].message.content)
```
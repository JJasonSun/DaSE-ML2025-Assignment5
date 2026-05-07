# 通用文本向量ecnu-embedding-small

* **模型简介** ：通用文本向量能力，1024 维
* **上下文** ：8K
* **部署方案** ：本地部署
* **默认限流** ：
* 每分钟请求数（rpm）：600
* 每天请求数（rpd）：无上限

提供 `OpenAI` 兼容的通用文本向量接口。

## 请求方法

POST

## 请求地址

[https://chat.ecnu.edu.cn/open/api/v1/embeddings](https://chat.ecnu.edu.cn/open/api/v1/embeddings)

## 请求参数

| 参数名 | 类型         | 是否必须 | 描述                                                                                                              |
| ------ | ------------ | -------- | ----------------------------------------------------------------------------------------------------------------- |
| model  | string       | 是       | 模型名称，目前可用 `ecnu-embedding-small`，详见 [模型列表](https://developer.ecnu.edu.cn/vitepress/llm/model.html) |
| input  | string/array | 是       | 输入文本,可以是字符串，也可以是字符串数组。不超过 8192 个字符                                                     |

## 返回参数

| 参数名  | 类型 | 描述       |
| ------- | ---- | ---------- |
| created | int  | 请求时间戳 |

## 请求示例

```
POST https://chat.ecnu.edu.cn/open/api/v1/embeddings
Content-Type: application/json
Authorization: Bearer sk-UDK3*******************1Y

{
    "model": "ecnu-embedding-small",
    "input": ["干的不错","一块石头"]
}
```

## 返回示例

```
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        0.0023064255,
        -0.009327292,
        .... (1024 floats total)
        -0.0028842222,
      ],
      "index": 0
    }
  ],
  "model": "ecnu-embedding-small",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

## 代码示例

### 直接调用接口

```
import requests
import json

# 设置API参数
url = "https://chat.ecnu.edu.cn/open/api/v1/embeddings"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-api-key"  # 替换为您的API密钥
}

# 单个文本嵌入请求
data_single = {
    "model": "ecnu-embedding-small",
    "input": "Hello world"
}

response = requests.post(url, headers=headers, data=json.dumps(data_single))
print(response.json()["data"][0]["embedding"])

# 多个文本嵌入请求
data_multi = {
    "model": "ecnu-embedding-small",
    "input": ["Hello world", "good work"]
}

response = requests.post(url, headers=headers, data=json.dumps(data_multi))
embeddings = response.json()["data"]
print(embeddings[0]["embedding"])  # Hello world
print(embeddings[1]["embedding"])  # good work
```

### 使用 openai sdk

```
from openai import OpenAI

client = OpenAI(
    base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key="your-api-key"  # 替换为您的API密钥
)

# 单个文本嵌入请求
response = client.embeddings.create(
    model="ecnu-embedding-small",  # 指定模型
    input="Hello world",           # 
)

# Hello world
print(response.data[0].embedding)

# 多个文本嵌入请求
response = client.embeddings.create(
    model="ecnu-embedding-small",  # 指定模型
    input=["Hello world","good work"]           #
)

# Hello world
print(response.data[0].embedding)
# good work
print(response.data[1].embedding)
```

### 使用 LangChain

```
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
           model="ecnu-embedding-small",
           api_key="your-api-key",  # 替换为您的API密钥
           dimensions=1024,
           check_embedding_ctx_length=False
        )

print(embeddings.embed_query("Hello world"))
```

#### 重要

> LangChain 在 OpenAIEmedding 模式下，默认会将 `input` 的字符串按 `openai` 的模型进行 `Tokenize`。此时提交接口的 `input` 将不再是字符串而是 `token` 数组。这对于非 `openai` 模型下实现的的 OpenAI 兼容 api 是无法支持的。因此，需要设置 `check_embedding_ctx_length=False` 来关闭 `Tokenize` 行为，确保提交接口的 `input` 仍然为字符串。详见 [7815](https://github.com/langchain-ai/langchain/pull/7815), [19743](https://github.com/langchain-ai/langchain/pull/19743), [23984](https://github.com/langchain-ai/langchain/discussions/23984)
>

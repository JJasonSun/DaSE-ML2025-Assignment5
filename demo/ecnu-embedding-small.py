''' 示例代码：使用华东师范大学ECNU-Embedding模型进行文本嵌入
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
'''

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
    input=["Hello world","good work"]   # 输入文本,可以是字符串，也可以是字符串数组。不超过 8192 个字符
)

# Hello world
print(response.data[0].embedding)
# good work
print(response.data[1].embedding)
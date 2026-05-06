# ecnu-rerank说明文档

提供 `Cohere` 兼容的通用文本向量接口。

## 请求方法

POST

## 请求地址

https://chat.ecnu.edu.cn/open/api/v1/rerank

## 请求参数

| 参数名           | 类型   | 是否必须 | 描述                                                                                                     |
| :--------------- | :----- | :------- | :------------------------------------------------------------------------------------------------------- |
| model            | string | 是       | 模型名称，目前可用 `ecnu-rerank`，详见 [模型列表](https://developer.ecnu.edu.cn/vitepress/llm/model.html) |
| documents        | array  | 是       | 文档列表，每个文档不超过 8192 个字符                                                                     |
| query            | string | 是       | 查询文本                                                                                                 |
| return_documents | bool   | 否       | 是否返回文档                                                                                             |
| top_n            | int    | 否       | 返回文档数量，默认为 5                                                                                   |

## 返回参数

| 参数名          | 类型   | 描述       |
| :-------------- | :----- | :--------- |
| results         | array  | 结果列表   |
| index           | int    | 文档索引   |
| document        | string | 文档内容   |
| relevance_score | float  | 相关性分数 |
| id              | string | 请求 ID    |

## 请求示例

```http
POST https://chat.ecnu.edu.cn/open/api/v1/rerank
Authorization: Bearer sk-******5c935b119e
Content-Type: application/json

{
	"documents":[
            "华东师范大学文脉绵长、声誉卓著，是教育部直属并与上海市重点共建的综合性研究型大学",
            "量子计算是计算科学的一个前沿领域",
            "师大始终秉承“智慧的创获，品性的陶熔，民族和社会的发展”的大学理想，恪守“求实创造，为人师表”的校训精神"
			],
	"model":"ecnu-rerank",
	"query":"介绍华东师范大学",
	"return_documents":true,
	"top_n":3
}
```

## 返回示例

```json
{
  "results": [
    {
      "index": 0,
      "document": "华东师范大学文脉绵长、声誉卓著，是教育部直属并与上海市重点共建的综合性研究型大学",
      "relevance_score": 0.9886243997611625
    },
    {
      "index": 2,
      "document": "师大始终秉承“智慧的创获，品性的陶熔，民族和社会的发展”的大学理想，恪守“求实创造，为人师表”的校训精神",
      "relevance_score": 0.3110081860706927
    },
    {
      "index": 1,
      "document": "量子计算是计算科学的一个前沿领域",
      "relevance_score": 0.0012843081201339695
    }
  ],
  "id": "016ccdf3-c344-49f8-bafd-517e4d8b43fc"
}
```

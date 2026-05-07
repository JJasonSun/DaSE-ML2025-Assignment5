## BaseURL

`https://chat.ecnu.edu.cn/open/api/v1`

以下是我们目前支持的模型列表。当前对话模型统一收敛为 `ecnu-max` 和 `ecnu-plus` 两个主模型，其余历史模型名称作为兼容别名保留。

**建议避免并行调用 API 服务，等上一个响应结束后再发起下一个请求，以获得更好的稳定服务**

## 服务状态

如需查看模型服务的实时状态和可用性，请访问[服务状态页面](https://chat.ecnu.edu.cn/status)。

## 对话模型

所有对话模型均已完成本地部署，数据处理和响应均在校内服务器完成。

| 模型名        | 底层模型                                                                     | 上下文 | 思考模式       | 工具调用 | 图片理解 | [限流倍率](https://developer.ecnu.edu.cn/vitepress/llm/limit.html) |
| ------------- | ---------------------------------------------------------------------------- | ------ | -------------- | -------- | -------- | --------------------------------------------------------------- |
| `ecnu-max`  | [DeepSeek-V4-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash) | 1M     | 支持，默认关闭 | 支持     | 不支持   | 3x                                                              |
| `ecnu-plus` | [Qwen3.6-27B](https://modelscope.cn/models/Qwen/Qwen3.6-27B)                    | 256K   | 支持，默认关闭 | 支持     | 支持     | 1x                                                              |

* ecnu-max 定位为面向开发者的旗舰模型，适合高质量复杂任务；
* ecnu-plus 定位为面向开发者的通用模型，兼顾效果、成本与响应速度。

### 备注

针对校内专属AI应用场景（例如 ChatECNU，Agent 平台等），`ecnu-max` 和 `ecnu-reasoner` 具有独立的服务集群以保障服务的稳定性。

如果您的业务对模型响应稳定性有较高要求，请与我们单独联系

### 默认参数

建议参考模型官方文档，合理设置 `temperature`、`top_p` 等参数，以获得更符合预期的响应效果。

## 思考模式

所有对话模型均支持通过 `thinking` 参数开启思考模式。

#### 开启思考模式

```
{
  "model": "ecnu-max",
  "messages": [
    {
      "role": "user",
      "content": "请帮我分析这个问题"
    }
  ],
  "thinking": {
    "type": "enabled"
  }
}
```

#### 关闭思考模式

```
{
  "model": "ecnu-reasoner",
  "messages": [
    {
      "role": "user",
      "content": "请帮我分析这个问题"
    }
  ],
  "thinking": {
    "type": "disabled"
  }
}
```

## 兼容别名

为保持历史兼容，平台仍保留原有模型名称。新接入应用建议优先使用 `ecnu-max` 或 `ecnu-plus`。

| 历史模型名             | 当前等价模型                                      | 说明                         |
| ---------------------- | ------------------------------------------------- | ---------------------------- |
| `ecnu-reasoner`      | `ecnu-max` + `thinking: {"type": "enabled"}`  | 默认开启思考模式的 ecnu-max  |
| `ecnu-reasoner-lite` | `ecnu-plus` + `thinking: {"type": "enabled"}` | 默认开启思考模式的 ecnu-plus |
| `ecnu-turbo`         | `ecnu-plus`                                     | 历史兼容别名                 |
| `ecnu-vl`            | `ecnu-plus`                                     | 历史兼容别名                 |
| `InnoSpark`          | `ecnu-plus`                                     | 历史兼容别名                 |
| `educhat-r1`         | `ecnu-plus`                                     | 历史兼容别名                 |
| `educhat-general`    | `ecnu-plus`                                     | 历史兼容别名                 |
| `educhat-psychology` | `ecnu-plus`                                     | 历史兼容别名                 |
| `ChatECNU`           | `ecnu-plus`                                     | 历史兼容别名                 |
| `gpt-4`              | `ecnu-plus`                                     | 兼容 `gpt-4` 的历史模型名  |

# NutriMind-Agent API 文档

> 文档状态：与当前代码同步  
> Base URL：`http://localhost:9999`  
> Swagger：`http://localhost:9999/docs`  
> ReDoc：`http://localhost:9999/redoc`

## 1. 智能体工作流

图片营养分析不是固定流水线，而是由 LangGraph ReAct 智能体选择并串联工具：

```text
前端上传图片 + 用户问题
  → 后端保存图片并生成 image_id
  → Agent 读取 image_id 和问题
  → Agent 调用 detect_food
  → YOLO/mock 返回 detections
  → Agent 调用 query_food_calories
  → Agent 调用 calculate_total_nutrition
  → Agent 综合工具观察结果生成分析和建议
```

Agent 当前可用工具：

| 工具 | 功能 |
|---|---|
| `detect_food` | 根据安全的 `image_id` 调用 YOLO 或读取 mock YOLO 结果 |
| `query_food_calories` | 查询单项食物每 100g 营养数据 |
| `query_food_by_category` | 按分类查询食物营养数据 |
| `calculate_total_nutrition` | 按食物及估算重量汇总热量和宏量营养素 |
| `save_detection_record` | 将检测记录写入数据库（数据库可用时） |

会话使用 `session_id` 关联。当前检查点使用进程内存，服务重启后历史会丢失。

## 2. 通用约定

### 2.1 认证

当前统一认证契约：浏览器使用 HttpOnly Cookie，脚本、移动端和 Swagger 可使用 Bearer Header。后端两种方式均支持，Header 优先。

登录成功后服务端同时：

1. 在响应 JSON 返回 `access_token`，供脚本或 Swagger 使用。
2. 设置 HttpOnly Cookie `access_token`，供浏览器自动携带。

浏览器前端调用接口时必须允许 Cookie：

```javascript
fetch("http://localhost:9999/api/auth/me", {
  credentials: "include"
})
```

Axios：

```javascript
axios.defaults.withCredentials = true
```

脚本方式仍可使用 Header：

```http
Authorization: Bearer <access_token>
```

标记为“DEBUG 联调”的接口无需登录，但只在 `DEBUG=true` 时开放。

### 2.2 错误格式

```json
{
  "detail": "错误说明"
}
```

### 2.3 YOLO 检测框

```json
{
  "class_name": "rice",
  "class_name_cn": "米饭",
  "confidence": 0.96,
  "bbox": [20, 80, 300, 400]
}
```

`bbox` 顺序为 `[x1, y1, x2, y2]`，`confidence` 范围为 `0-1`。

## 3. 健康检查

### `GET /api/health`

无需认证。

```json
{
  "status": "ok",
  "message": "VisAgent API is running"
}
```

## 4. 认证

### 4.1 注册

`POST /api/auth/register`

```json
{
  "username": "zhangsan",
  "email": "zhangsan@example.com",
  "password": "mypassword123"
}
```

成功状态码：`201`。用户名重复、邮箱重复或字段不合法返回 `400/422`。

### 4.2 登录

`POST /api/auth/login`

```json
{
  "username": "zhangsan",
  "password": "mypassword123"
}
```

成功响应：

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "user": {
    "id": 18,
    "username": "zhangsan",
    "email": "zhangsan@example.com",
    "phone": null,
    "avatar": null,
    "is_active": true,
    "is_superuser": false,
    "roles": [],
    "last_login_at": "2026-07-16T10:00:00",
    "created_at": "2026-07-15T11:14:17"
  }
}
```

同时写入 HttpOnly Cookie。登录响应中的 `user` 与 `/api/auth/me` 使用同一套完整字段。

Cookie 开发环境属性：`HttpOnly=true`、`Secure=false`、`SameSite=lax`、`Path=/`、有效期 30 分钟。生产 HTTPS 部署时应将 `AUTH_COOKIE_SECURE=true`。

### 4.3 当前用户

`GET /api/auth/me`，需要认证。

成功响应：

```json
{
  "id": 18,
  "username": "zhangsan",
  "email": "zhangsan@example.com",
  "phone": null,
  "avatar": null,
  "is_active": true,
  "is_superuser": false,
  "roles": [],
  "last_login_at": "2026-07-16T10:00:00",
  "created_at": "2026-07-15T11:14:17"
}
```

无有效 Bearer Header 或 HttpOnly Cookie 时返回 `401`；账户被禁用时返回 `403`。

### 4.4 登出

`POST /api/auth/logout`，清除 Cookie。

登出是幂等的，不要求有效 Token；即使 Token 已过期也返回 `200` 并清理 Cookie。

## 5. AI 对话

### 5.1 文本/已有检测结果对话

`POST /api/chat/message`，需要认证。

既可只传文本，也可兼容其他服务已经生成的 YOLO `detections`：

```json
{
  "session_id": "meal-001",
  "message": "分析一下这顿饭",
  "detections": [
    {
      "class_name": "rice",
      "class_name_cn": "米饭",
      "confidence": 0.96,
      "bbox": [20, 80, 300, 400]
    }
  ]
}
```

响应：

```json
{
  "session_id": "meal-001",
  "response": "智能体回复",
  "tool_calls": [
    {
      "name": "calculate_total_nutrition",
      "args": {
        "food_items_json": "[...]"
      }
    }
  ],
  "analysis_result": null
}
```

### 5.2 正式图片智能体对话

`POST /api/chat/image`，需要认证，`multipart/form-data`。

| 字段 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `file` | file | 是 | JPG/PNG/WEBP，默认最大 10MB |
| `message` | string | 是 | 用户针对图片的问题 |
| `session_id` | string | 否 | 不传由服务端生成 |

示例：

```bash
curl -X POST http://localhost:9999/api/chat/image \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@meal.jpg" \
  -F "message=识别这顿饭并分析热量" \
  -F "session_id=meal-001"
```

使用真实模型前配置：

```env
DETECTION_MODE=yolo
DEFAULT_DETECTION_MODEL=best.pt
```

模型放置于：`backend/data/models/best.pt`。视觉依赖安装命令：

```bash
uv sync --extra vision
```

### 5.3 模拟 YOLO 图片智能体联调

`POST /api/chat/image/mock`，DEBUG 联调接口，无需认证，`multipart/form-data`。

该接口仍然上传图片并让 Agent 主动调用 `detect_food`；区别仅在于工具读取请求附带的模拟 YOLO 结果，因此当前没有 `best.pt` 也能验证完整工具调用链。

| 字段 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| `file` | file | 是 | 测试图片 |
| `message` | string | 是 | 例如“估算热量并给出建议” |
| `mock_detections` | string | 是 | JSON 数组字符串 |
| `session_id` | string | 否 | 多轮会话 ID |

```bash
curl -X POST http://localhost:9999/api/chat/image/mock \
  -F "file=@meal.jpg" \
  -F "message=估算总热量并给出三条建议" \
  -F 'session_id=mock-meal-001' \
  -F 'mock_detections=[{"class_name":"rice","class_name_cn":"米饭","confidence":0.96,"bbox":[20,80,300,400]},{"class_name":"chicken","class_name_cn":"鸡肉","confidence":0.93,"bbox":[320,100,580,390]}]'
```

响应：

```json
{
  "session_id": "mock-meal-001",
  "image_id": "2a41...",
  "detection_mode": "mock",
  "detections": [
    {
      "class_name": "rice",
      "class_name_cn": "米饭",
      "confidence": 0.96,
      "bbox": [20, 80, 300, 400]
    }
  ],
  "response": "本餐热量为视觉估算值……",
  "tool_calls": [
    {"name": "detect_food", "args": {"image_id": "2a41..."}},
    {"name": "query_food_calories", "args": {"food_name": "rice"}},
    {"name": "calculate_total_nutrition", "args": {"food_items_json": "[...]"}}
  ],
  "analysis_result": null
}
```

验收时应确认 `tool_calls` 至少出现 `detect_food`；营养分析问题还应出现营养查询/计算工具。

### 5.4 直接模拟 detections（兼容联调）

`POST /api/chat/mock-yolo`，DEBUG 联调接口，无需认证，请求体与 `/api/chat/message` 相同。该接口跳过 Agent 调用视觉工具，仅用于测试后半段营养链路，不作为最终主链路。

## 6. 知识库

以下接口均需要认证。

### 6.1 上传文档

`POST /api/knowledge/upload`，`multipart/form-data`，字段 `file`。

当前代码支持：`.pdf`、`.md`、`.txt`、`.text`。上传后进行分块、Embedding 并写入 PGVector。

### 6.2 语义检索

`GET /api/knowledge/search?query=营养学&k=5`

### 6.3 删除文档

`DELETE /api/knowledge/?source=<metadata.source>`

### 6.4 统计

`GET /api/knowledge/stats`

## 7. 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEBUG` | `true` | 是否开放 DEBUG 联调接口 |
| `OPENAI_API_KEY` | - | OpenAI 兼容模型密钥 |
| `OPENAI_BASE_URL` | SiliconFlow | 模型 API 地址 |
| `OPENAI_MODEL` | `Qwen/Qwen3.6-35B-A3B` | 支持工具调用的对话模型 |
| `DETECTION_MODE` | `mock` | `mock` 或 `yolo` |
| `DEFAULT_DETECTION_MODEL` | `best.pt` | `data/models` 下的权重文件名 |
| `MAX_IMAGE_SIZE_MB` | `10` | 图片上传大小限制 |
| `DB_*` | 见 `.env.example` | PostgreSQL 配置 |

## 8. 当前未开放的规划接口

以下模块在参考文档中出现，但当前路由文件为空或没有注册，前端暂时不要调用：

- `/api/camera/*`
- `/api/detection/*`
- `/api/training/*`
- `/api/dashboard/*`
- `/api/auth/change-password`

后续新增时应同步更新本文档与 OpenAPI。

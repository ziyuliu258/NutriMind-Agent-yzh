# Nomad 智能体与后端改动说明

## 0. 当前完成度与使用结论

本次提交不是只定义接口或返回模拟数据的空壳框架。会话数据库读写、PGVector 检索、LLM 回答、Exa HTTP 搜索和联网语料落库均使用真实实现。

但是，当前版本的准确定位是：

> **已经达到测试环境和前后端联调阶段，尚未完成生产环境上架验收。**

原因是当前开发环境没有可供本次测试连接的完整 PostgreSQL + PGVector、有效 Exa API Key 和稳定的线上 LLM 配置，所以自动化测试验证了业务编排、数据库服务逻辑、fallback 判断和 API 返回结构，但没有完成真实外部服务参与的全链路验收。

### 0.1 已真实实现的能力

| 功能 | 实现方式 | 当前状态 |
| --- | --- | --- |
| 会话持久化 | SQLAlchemy 真实数据表和数据库读写 | 已实现 |
| 创建、查询、删除会话 | FastAPI 真实接口 | 已实现 |
| 历史对话恢复 | 从数据库加载消息并传入 LangGraph | 已实现 |
| 私有知识库检索 | PGVector 相似度检索 | 已实现 |
| RAG 回答 | 调用配置的 OpenAI-compatible LLM | 已实现 |
| 联网 fallback | 调用 Exa Search REST API | 已实现 |
| 来源交叉验证 | 合并本地知识和网页搜索来源 | 已实现 |
| 联网语料落库 | 将网页正文及元数据写入 PGVector | 已实现 |
| 用户数据隔离 | 当前用户 ID 由服务端上下文注入 | 已实现 |

### 0.2 尚未完成或需要测试环境验收的事项

| 事项 | 当前情况 | PR 后建议处理方式 |
| --- | --- | --- |
| PostgreSQL + PGVector 真实全链路 | 当前环境无法连接项目 PostgreSQL | 测试环境启动 PostgreSQL/PGVector 后验收上传、检索、问答和落库 |
| Exa 真实联网调用 | 没有使用有效 `EXA_API_KEY` 进行现场调用 | 测试环境配置 Key 后验证结果、超时和失败降级 |
| 真实 LLM Agent 行为 | 自动化测试中外部 LLM 调用使用 mock | 使用项目实际模型验证工具选择、引用和多轮对话 |
| Exa MCP 协议 | 当前接入 Exa REST API，不是 MCP transport | 若验收明确要求 MCP，后续接入 Exa MCP Server |
| 正式数据库迁移 | 项目当前仍使用 `Base.metadata.create_all()` | 上架前补 Alembic migration |
| 图片对话持久化 | 文字消息接口已持久化，图片对话尚未完整接入历史表 | 根据前端会话设计继续接入 `/api/chat/image` |
| 网页语料去重 | 当前允许直接写入 PGVector，尚未做 URL/内容哈希去重 | 上架前增加去重、过期和更新策略 |
| 搜索重试与限流 | 已有超时和错误降级，没有指数重试和用户级限流 | 上架前增加重试、限流和配额控制 |
| 可观测性 | 有基础日志，没有搜索命中率、耗时和费用指标 | 上架前增加监控和告警 |
| 高并发验证 | 尚未进行压力测试 | 测试环境执行并发会话和知识问答压测 |

### 0.3 Exa REST 与 MCP 说明

当前实现使用 Exa 官方搜索 HTTP API：

```text
https://api.exa.ai/search
```

该调用会返回真实网页搜索结果，并且已经封装为 LangGraph Agent 工具，因此 Agent 可以自主进行联网检索和来源交叉验证。

不过，这不等于已经接入 Exa MCP Server。两者区别如下：

- 当前方案：NutriMind 后端直接调用 Exa REST API，再把统一结果提供给 Agent。
- MCP 方案：Agent 或 MCP Client 通过 MCP 协议发现并调用 Exa Server 暴露的工具。

从本次“知识库不足时能够可靠联网搜索”的功能目标看，当前 REST 实现可以进入测试；如果课程或验收标准明确要求展示 MCP 协议、工具发现或 MCP Server 调用，则需要另开任务补充真正的 MCP transport。

### 0.4 PR 结论

本次 PR 可以提交给测试同学进行功能验收，但不应在尚未完成真实外部服务联调时直接标记为“生产可上架”。建议 PR 标签或描述使用：

```text
功能已实现，待测试环境进行 PostgreSQL/PGVector、Exa 和真实 LLM 端到端验收。
```

## 1. 改动范围

本次仅实现测试反馈中由 **Nomad** 负责的智能体和后端能力，主要包括：

- 智能体历史对话断点续聊与新建对话。
- 营养知识库 RAG 问答。
- 知识库不足时的联网检索 fallback。
- 使用联网来源对关键营养知识进行交叉验证。
- 将有价值的联网搜索结果写回用户知识库。
- 为 Agent 注册知识库检索和 Exa 联网搜索工具。

本次没有修改：

- 用户头像右上角同步。
- 前端页面和交互。
- 知识库前端可视化或知识图谱。
- Dashboard、Camera、模型训练等其他成员负责的模块。

## 2. 对话历史与断点续聊

### 2.1 新增数据模型

在 `backend/app/entity/db_models.py` 中新增：

- `ChatSession`：保存用户会话、标题和更新时间。
- `ChatMessage`：保存用户消息、智能体回复和工具调用记录。

会话与用户关联，查询时同时使用 `session_id` 和当前登录用户 ID，避免读取其他用户的聊天历史。

数据库表由项目现有的 `Base.metadata.create_all()` 启动流程自动创建：

- `chat_sessions`
- `chat_messages`

### 2.2 新增会话服务

在 `backend/app/services/chat_service.py` 中实现：

- 创建新会话。
- 根据用户和会话 ID 查询会话。
- 自动获取或创建会话。
- 获取用户会话列表。
- 保存用户消息与智能体回复。
- 将数据库消息转换为 LangChain 消息历史。
- 删除会话及关联消息。

### 2.3 消息接口接入持久化

`POST /api/chat/message` 现在会执行以下流程：

1. 根据当前用户和 `session_id` 查询历史会话。
2. 如果会话不存在，则自动创建。
3. 加载数据库中的历史消息。
4. 保存当前用户消息。
5. 将历史消息传给 LangGraph Agent。
6. 保存最终智能体回复和工具调用记录。

如果数据库暂时不可用，接口会降级使用原有的 LangGraph 进程内历史，仍然可以返回智能体回复。

### 2.4 新增会话 API

#### 创建新对话

```http
POST /api/chat/sessions
Content-Type: application/json

{
  "title": "我的营养计划"
}
```

返回示例：

```json
{
  "session_id": "c75c67fc-a3ac-4990-a9bf-f3e037b21368",
  "title": "我的营养计划",
  "created_at": "2026-07-20T10:00:00",
  "updated_at": "2026-07-20T10:00:00",
  "messages": []
}
```

#### 获取历史对话列表

```http
GET /api/chat/sessions
```

#### 获取完整对话历史

```http
GET /api/chat/sessions/{session_id}
```

前端加载历史后，继续使用相同的 `session_id` 调用 `POST /api/chat/message`，即可从断点继续对话。

#### 删除对话

```http
DELETE /api/chat/sessions/{session_id}
```

## 3. 营养知识库 RAG 问答

### 3.1 新增问答接口

新增：

```http
GET /api/knowledge/ask
```

参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | string | 必填 | 用户的营养知识问题 |
| `k` | integer | `5` | 最多检索的本地/网页结果数，范围 1～10 |
| `verify_web` | boolean | `false` | 是否强制联网交叉验证本地知识 |
| `store_web` | boolean | `true` | 是否允许将网页结果写回知识库 |

请求示例：

```http
GET /api/knowledge/ask?query=成年人每天应该摄入多少蛋白质&k=5&verify_web=true
```

接口不再只返回向量片段，而是返回：

- 一段由 LLM 根据检索资料生成的完整回答。
- 本地知识库关联结果。
- 网页搜索结果。
- 适合前端展示的统一来源列表。
- 是否使用了联网 fallback。
- 是否完成本地资料与网页资料的交叉验证。

简化响应示例：

```json
{
  "code": 200,
  "data": {
    "query": "成年人每天应该摄入多少蛋白质",
    "answer": "健康成年人每日蛋白质需求通常需要结合体重和活动水平判断。[local-1][web-1]",
    "sources": [
      {
        "id": "local-1",
        "type": "knowledge",
        "title": "中国居民膳食指南",
        "url": null,
        "score": 0.31,
        "excerpt": "……"
      },
      {
        "id": "web-1",
        "type": "web",
        "title": "网页资料标题",
        "url": "https://example.org/nutrition",
        "score": null,
        "excerpt": "……"
      }
    ],
    "used_web_fallback": false,
    "cross_verified": true
  }
}
```

### 3.2 回答生成约束

知识库回答使用以下约束：

- 只能根据实际检索到的资料回答。
- 重要结论使用 `[local-1]` 或 `[web-1]` 标记来源。
- 如果本地资料与网页资料冲突，需要明确说明。
- 不允许编造来源。
- 不提供医学诊断。
- 本地知识库和联网检索均无结果时，明确提示资料不足。

## 4. 联网检索 fallback

### 4.1 Exa 搜索服务

新增 `backend/app/services/web_search_service.py`，封装 Exa Search API。

统一返回字段包括：

- 网页标题。
- 网页 URL。
- 相关正文或高亮片段。
- 发布时间。
- 搜索供应商。
- 来源类型。

搜索请求设置了超时，并处理：

- HTTP 错误。
- 网络错误。
- 请求超时。
- 非法 JSON 响应。
- 未配置 API Key。

联网搜索不可用时返回空结果，不会导致整个 Agent 或知识库接口崩溃。

### 4.2 fallback 判断

知识库首先执行用户隔离的 PGVector 相似度检索。在以下情况触发联网搜索：

- 本地知识库没有结果。
- 最佳本地结果的距离分数高于 `KNOWLEDGE_LOCAL_SCORE_THRESHOLD`。
- 调用方主动设置 `verify_web=true`，要求交叉验证。

PGVector 当前使用距离分数，数值越小表示越相关。

### 4.3 联网语料落库

当 `WEB_SEARCH_STORE_RESULTS=true` 且请求允许 `store_web` 时，联网结果会转换为 LangChain `Document` 并写入当前用户的 PGVector 知识库。

写入的元数据包括：

- `source`
- `file_name`
- `title`
- `url`
- `source_type=web`
- `provider=exa`
- 原始查询问题
- 当前用户 ID

这样同类问题后续可以直接从本地 RAG 语料中检索，减少重复联网调用。

联网结果落库失败不会影响当次回答。

## 5. Agent 工具改动

在 `backend/app/services/agent_tools.py` 和 `backend/app/services/agent_graph.py` 中新增两个工具。

### 5.1 `search_nutrition_knowledge`

用途：

- 查询当前用户的私有营养知识库。
- 本地结果不足时自动联网 fallback。
- 可选对本地知识进行网页交叉验证。
- 返回本地片段、网页结果和验证状态。

### 5.2 `search_web_evidence`

用途：

- 直接使用 Exa 联网查找营养资料。
- 验证摄入标准、指南、研究结论等重要信息。
- 返回标题、URL 和正文，供 Agent 在最终回答中引用。

### 5.3 用户隔离安全处理

LLM 工具参数中不允许传入 `user_id`。

当前登录用户 ID 通过 Python `ContextVar` 在服务端请求上下文中注入。即使 LLM 生成了错误或恶意工具参数，也不能通过修改用户 ID 检索其他用户的知识库。

## 6. Agent 提示词更新

在 `backend/app/services/agent_prompts.py` 中增加以下工作规范：

1. 营养知识、指南、研究结论优先查询用户知识库。
2. 本地结果为空或相关性不足时必须使用联网 fallback。
3. 健康风险、推荐摄入量等关键结论需要联网交叉验证。
4. 最终回答应是一段完整、易读的文字。
5. 回答末尾列出实际使用的来源标题和 URL。
6. 不得伪造来源。

## 7. 新增配置

`backend/.env.example` 新增：

```env
EXA_API_KEY=your-exa-api-key
EXA_SEARCH_URL=https://api.exa.ai/search
WEB_SEARCH_ENABLED=true
WEB_SEARCH_STORE_RESULTS=true
KNOWLEDGE_LOCAL_SCORE_THRESHOLD=0.85
```

配置说明：

| 配置 | 说明 |
| --- | --- |
| `EXA_API_KEY` | Exa 搜索 API Key；为空时联网功能自动关闭 |
| `EXA_SEARCH_URL` | Exa Search API 地址 |
| `WEB_SEARCH_ENABLED` | 是否启用联网搜索 |
| `WEB_SEARCH_STORE_RESULTS` | 是否允许将网页语料写入 PGVector |
| `KNOWLEDGE_LOCAL_SCORE_THRESHOLD` | 本地结果触发 fallback 的距离阈值 |

同时将 `.env.example` 中原先硬编码的 OpenAI API Key 改为占位符，避免示例配置泄露凭据。

## 8. 主要改动文件

| 文件 | 改动 |
| --- | --- |
| `backend/app/api/chat.py` | 会话持久化接入及会话管理 API |
| `backend/app/api/knowledge.py` | 新增知识库问答接口 |
| `backend/app/config/settings.py` | 新增 Exa 和 fallback 配置 |
| `backend/app/entity/db_models.py` | 新增会话和消息数据表 |
| `backend/app/entity/schemas.py` | 新增会话请求/响应 Schema |
| `backend/app/services/chat_service.py` | 会话持久化服务 |
| `backend/app/services/knowledge_service.py` | RAG、fallback、交叉验证和联网语料落库 |
| `backend/app/services/web_search_service.py` | Exa 联网检索封装 |
| `backend/app/services/agent_tools.py` | 新增知识库和联网搜索工具 |
| `backend/app/services/agent_graph.py` | 注册工具、恢复持久化历史、安全注入用户上下文 |
| `backend/app/services/agent_prompts.py` | 更新智能体知识检索规范 |
| `backend/.env.example` | 新增联网配置并移除示例明文密钥 |
| `backend/tests/test_api_chat.py` | 会话恢复和 Agent 工具测试 |
| `backend/tests/test_knowledge.py` | 问答响应和联网 fallback 测试 |

## 9. 测试结果

针对本次智能体与知识库改动运行：

```bash
DEBUG=true uv run pytest -q tests/test_api_chat.py tests/test_knowledge.py
```

结果：

```text
24 passed
```

同时完成：

- Python 静态编译检查通过。
- `git diff --check` 通过。

运行完整后端测试时结果为：

```text
102 passed, 11 failed, 1 error
```

剩余失败来自当前主分支已有的非 Nomad 范围问题：

- Dashboard 已限制管理员访问，但旧测试仍使用普通用户并期待 HTTP 200。
- `backend/test_model.py` 被 pytest 收集后缺少 `model` fixture。
- 原有 `DetectionScene` 与 `DetectionTask` 级联删除关系未正确配置。

本次没有修改这些模块。

## 10. 测试同学建议重点验证

1. 创建两个不同的新会话，确认历史互不串联。
2. 服务重启后使用原 `session_id` 继续提问，确认 Agent 能恢复上下文。
3. 用户 A 无法读取用户 B 的会话和知识库内容。
4. 有本地知识时，`/api/knowledge/ask` 能生成一段回答并附带本地来源。
5. 本地没有相关知识时，配置 `EXA_API_KEY` 后能够触发网页 fallback。
6. 设置 `verify_web=true` 后，同时返回本地和网页来源，并标记 `cross_verified=true`。
7. 关闭网络或移除 `EXA_API_KEY` 后，接口能够优雅降级。
8. 开启联网语料落库后，再次查询相似问题能够检索到已保存的网页语料。

## 11. 测试环境准备

测试本次完整功能前，需要准备以下服务和配置。

### 11.1 PostgreSQL 与 PGVector

确认 PostgreSQL 已安装并启用 `vector` 扩展，数据库配置与 `.env` 一致：

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nutrimind-agent
DB_USER=nutrimind-agent
DB_PASSWORD=nutrimind-agent
```

应用启动时会通过现有 `Base.metadata.create_all()` 创建新的会话表。知识库向量表由当前 PGVector 集成初始化。

### 11.2 LLM 与 Embedding

```env
OPENAI_API_KEY=有效的模型服务密钥
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3.6-35B-A3B
```

还需要确认当前模型服务支持项目配置的 Embedding 模型：

```text
Qwen/Qwen3-Embedding-8B
```

### 11.3 Exa 联网搜索

```env
EXA_API_KEY=有效的 Exa API Key
EXA_SEARCH_URL=https://api.exa.ai/search
WEB_SEARCH_ENABLED=true
WEB_SEARCH_STORE_RESULTS=true
KNOWLEDGE_LOCAL_SCORE_THRESHOLD=0.85
```

如果不配置 `EXA_API_KEY`，系统仍可使用本地知识库，但联网 fallback 会返回空结果并执行优雅降级。

### 11.4 建议验收顺序

1. 启动 PostgreSQL/PGVector 和后端服务。
2. 登录测试用户并创建新会话。
3. 连续发送两条相关消息，确认第二条能够理解第一条上下文。
4. 重启后端，再使用原 `session_id` 发送消息，确认历史能够恢复。
5. 上传一份营养文档，调用 `/api/knowledge/search` 确认可以检索。
6. 调用 `/api/knowledge/ask`，确认返回完整回答和 `local-*` 来源。
7. 提问知识库中没有的问题，确认触发 Exa 并返回 `web-*` 来源。
8. 使用 `verify_web=true`，确认同时存在本地和网页来源。
9. 再次查询相似问题，检查联网语料是否已进入当前用户的知识库。
10. 切换到另一个用户，确认无法读取原用户的会话和私有知识。

## 12. 上架前必须补齐的事项

完成测试环境验收后，如准备部署到生产环境，至少还需要：

1. 为 `chat_sessions` 和 `chat_messages` 添加正式 Alembic migration。
2. 为 Exa 调用增加指数退避重试、用户级限流和每日配额。
3. 对网页语料增加 URL 或内容哈希去重机制。
4. 增加网页语料更新时间和过期清理策略。
5. 对搜索来源进行域名白名单、黑名单或可信度分级。
6. 验证真实 LLM 是否始终保留并正确输出来源引用。
7. 补齐图片对话的数据库历史持久化。
8. 执行并发测试、压力测试和数据库连接池验证。
9. 增加搜索耗时、失败率、fallback 命中率和模型费用监控。
10. 根据最终验收要求决定是否接入真正的 Exa MCP Server。

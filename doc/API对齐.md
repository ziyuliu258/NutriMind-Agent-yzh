# NutriMind-Agent API 对齐说明

> 更新日期：2026-07-16  
> 适用对象：前端、后端、智能体、测试与联调人员  
> 当前权威接口文档：[API.md](./API.md)

## 1. 对齐结论

当前认证契约统一为：

- 浏览器前端使用 **HttpOnly Cookie**。
- Swagger、脚本、移动端可以使用 **Bearer Token**。
- 后端同时支持两种凭据来源。
- 请求同时携带 Header 和 Cookie 时，Bearer Header 优先。
- 登录成功后，后端既返回 `access_token`，也设置 HttpOnly Cookie。
- 浏览器前端不需要、也不建议把 Token 写入 LocalStorage。

```text
浏览器前端：HttpOnly Cookie + credentials: include
Swagger/脚本：Authorization: Bearer <access_token>
```

## 2. 为什么之前前后端认证不一致

历史 API 文档宣称支持以下两种认证方式：

```http
Authorization: Bearer <access_token>
```

```http
Cookie: access_token=<access_token>
```

登录接口也会设置 `access_token` Cookie，但原认证依赖中的 Cookie 回退逻辑没有实现：

```python
if not token:
    # 尝试从 Cookie 获取
    pass
```

因此旧代码的实际行为是：

| 前端行为 | 旧后端结果 |
|---|---|
| 发送 Bearer Header | 认证成功 |
| 只依靠登录 Cookie | 返回 `401` |
| 不把 Token 写入 LocalStorage | 无法按 Bearer 方案请求 |

前端采用 HttpOnly Cookie 是合理的安全方案，但后端此前只完成了“设置 Cookie”，没有完成“读取 Cookie”，造成文档、前端与后端行为冲突。

## 3. 已完成的后端修复

### 3.1 Cookie 与 Bearer 双通道

受保护接口现在按以下顺序查找凭据：

1. `Authorization: Bearer <token>`。
2. HttpOnly Cookie `access_token`。
3. 两者都没有时返回 `401`。

### 3.2 登录响应字段统一

登录响应中的 `user` 与 `/api/auth/me` 使用同一套用户字段：

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

普通用户注册请求不能提交 `is_superuser`，新注册用户默认不是管理员。

### 3.3 登出契约统一

`POST /api/auth/logout` 定义为幂等接口：

- 已登录：清除 Cookie并返回 `200`。
- Token 已过期：仍清除 Cookie并返回 `200`。
- 未登录：仍返回 `200`。

响应：

```json
{
  "message": "已登出"
}
```

### 3.4 验证结果

认证与 Agent 回归测试结果：

```text
11 passed
```

测试覆盖：

- Cookie 认证成功。
- Bearer Token 认证成功。
- 无凭据返回 `401`。
- Agent 对话接口认证修改后无回归。

## 4. 前端浏览器调用规范

### 4.1 统一地址

本地开发建议统一使用：

```text
前端：http://localhost:5173
后端：http://localhost:9999
```

不要混用：

```text
localhost
127.0.0.1
```

例如前端使用 `127.0.0.1`、后端使用 `localhost` 时，浏览器可能将其视为不同站点，导致 Cookie 不按预期发送。

### 4.2 Fetch 登录

```javascript
const response = await fetch("http://localhost:9999/api/auth/login", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  credentials: "include",
  body: JSON.stringify({
    username: "zhangsan",
    password: "mypassword123"
  })
});

const data = await response.json();
```

关键配置：

```javascript
credentials: "include"
```

登录后浏览器保存 HttpOnly Cookie。JavaScript 无法读取 HttpOnly Cookie 是正常安全行为。

### 4.3 Fetch 恢复用户身份

页面刷新后调用：

```javascript
const response = await fetch("http://localhost:9999/api/auth/me", {
  credentials: "include"
});

if (response.ok) {
  const user = await response.json();
  // 恢复前端登录状态
}
```

前端不应该通过“LocalStorage 中有没有 Token”判断登录状态，应以 `/api/auth/me` 的结果为准。

### 4.4 Fetch 调用图片智能体

```javascript
const formData = new FormData();
formData.append("file", imageFile);
formData.append("message", "分析这顿饭的热量");
formData.append("session_id", "meal-001");

const response = await fetch("http://localhost:9999/api/chat/image", {
  method: "POST",
  credentials: "include",
  body: formData
});

const result = await response.json();
```

使用 `FormData` 时不要手动设置 `Content-Type`。浏览器会自动添加正确的 multipart boundary。

### 4.5 Fetch 登出

```javascript
await fetch("http://localhost:9999/api/auth/logout", {
  method: "POST",
  credentials: "include"
});
```

## 5. Axios 调用规范

建议创建统一 Axios 实例：

```javascript
import axios from "axios";

export const api = axios.create({
  baseURL: "http://localhost:9999",
  withCredentials: true
});
```

登录：

```javascript
await api.post("/api/auth/login", {
  username: "zhangsan",
  password: "mypassword123"
});
```

恢复身份：

```javascript
const { data: user } = await api.get("/api/auth/me");
```

调用智能体：

```javascript
const formData = new FormData();
formData.append("file", imageFile);
formData.append("message", "分析这顿饭");

const { data } = await api.post("/api/chat/image", formData);
```

## 6. Swagger 与脚本认证

Swagger 和脚本可以继续使用 Bearer Token：

1. 调用 `POST /api/auth/login`。
2. 复制响应中的 `access_token`。
3. 在 Swagger 右上角点击 `Authorize`。
4. 填入 Token。
5. 调用受保护接口。

cURL 示例：

```bash
curl http://localhost:9999/api/auth/me \
  -H "Authorization: Bearer <access_token>"
```

## 7. 后端 Cookie 与 CORS 配置

开发环境 `.env` 示例：

```env
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

AUTH_COOKIE_NAME=access_token
AUTH_COOKIE_SECURE=false
AUTH_COOKIE_SAMESITE=lax
# AUTH_COOKIE_DOMAIN=
```

当前开发 Cookie 属性：

| 属性 | 开发环境值 | 说明 |
|---|---|---|
| `HttpOnly` | `true` | JavaScript 无法读取 Cookie |
| `Secure` | `false` | 本地 HTTP 开发需要；生产必须改为 `true` |
| `SameSite` | `lax` | 降低跨站请求风险 |
| `Path` | `/` | 所有后端路由可使用 |
| `Max-Age` | 30 分钟 | 与 Access Token 有效期一致 |

生产 HTTPS 环境至少应配置：

```env
DEBUG=false
AUTH_COOKIE_SECURE=true
ALLOWED_ORIGINS=https://实际前端域名
```

如果前后端部署在真正的跨站域名，需进一步评估：

- `SameSite=None`
- `Secure=true`
- CSRF Token 或双重提交 Cookie
- 精确的 CORS Origin

## 8. 状态码约定

| 状态码 | 含义 |
|---:|---|
| `200` | 请求成功 |
| `201` | 注册成功 |
| `400` | 业务参数错误，如用户名已存在 |
| `401` | 未提供凭据、Token 无效或过期 |
| `403` | 用户存在但已禁用，或无权限 |
| `404` | 接口或资源不存在 |
| `422` | FastAPI/Pydantic 请求字段校验失败 |
| `500` | 服务器内部错误 |

错误默认使用 FastAPI 格式：

```json
{
  "detail": "错误说明"
}
```

当前认证和 Chat 返回顶层对象；知识库仍使用 `code/message/data` 包装。前端应按具体接口文档解析，不要假设所有接口都有统一包装。

## 9. 管理员身份约定

当前用户响应同时提供：

```json
{
  "is_superuser": false,
  "roles": []
}
```

当前建议：

- `is_superuser=true` 表示系统超级管理员。
- `roles` 用于可扩展的角色权限。
- 普通前端功能不能只通过隐藏按钮实现权限控制，管理员接口必须由后端返回 `403`。
- 当前 Dashboard 路由尚未注册，管理员权限矩阵需要在 Dashboard 实现时补充测试。

## 10. 用户与资源隔离

### 10.1 Chat 会话

正式 Chat 会话内部使用：

```text
(user_id, session_id)
```

作为命名空间。两个用户即使传入相同 `session_id`，也不会共享对话历史。

### 10.2 图片

前端上传图片后，后端生成不可预测的 `image_id`，Agent 工具不接受任意本地文件路径。

正式上线前仍需要进一步持久化图片所有权，并在读取 `image_id` 时校验当前用户。

### 10.3 知识库

当前知识库接口要求认证，但向量数据尚未按用户隔离。正式多用户部署前必须增加 `user_id` 元数据过滤和删除权限校验。

## 11. 当前未实现或未确认的认证能力

### 11.1 Refresh Token

当前配置中存在：

```env
REFRESH_TOKEN_EXPIRE_DAYS=7
```

但目前没有实现：

- `/api/auth/refresh`
- Refresh Token 返回或 Cookie
- Token 轮换
- Refresh Token 撤销

因此前端当前不能使用 Refresh Token。Access Token 过期后需要重新登录。

### 11.2 修改密码

当前没有注册：

```text
POST /api/auth/change-password
```

前端暂时不要调用。

### 11.3 Dashboard

当前没有注册 `/api/dashboard/*` 路由。`Dashboard_API(1).md` 不能作为当前部署环境的有效接口依据。

## 12. 联调验收清单

前后端联调至少验证：

- [ ] 登录响应状态为 `200`。
- [ ] 登录响应包含完整 `user` 字段。
- [ ] 浏览器开发者工具中存在 HttpOnly `access_token` Cookie。
- [ ] 不发送 Bearer Header，只使用 `credentials: include` 可以访问 `/api/auth/me`。
- [ ] 无 Cookie、无 Header 时 `/api/auth/me` 返回 `401`。
- [ ] 禁用用户访问受保护接口返回 `403`。
- [ ] 页面刷新后通过 `/api/auth/me` 恢复登录状态。
- [ ] `/api/chat/image` 可以通过 Cookie 认证。
- [ ] 登出后 Cookie 被清除。
- [ ] 登出后再次访问 `/api/auth/me` 返回 `401`。
- [ ] 前后端统一使用 `localhost`，不与 `127.0.0.1` 混用。

## 13. 唯一文档来源

当前开发和联调以以下内容为准：

1. 实际运行分支的 Swagger：`http://localhost:9999/docs`
2. [API.md](./API.md)
3. 本文档 `API对齐.md`

`api1.0.md`、`api2.0.md`、`api3.0.md`、`Dashboard_API(1).md` 仅作为历史参考，不能覆盖当前运行代码和 Swagger。

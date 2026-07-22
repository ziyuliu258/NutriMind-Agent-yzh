import request from '@/utils/request'

// 经典（非流式）模式的超时放宽：智能体走多工具+联网时可能需要数分钟。
const TEXT_CHAT_TIMEOUT = 300000   // 5 分钟
const IMAGE_CHAT_TIMEOUT = 360000  // 6 分钟

const RETRY_ATTEMPTS = 2           // 首次之外最多再重试 2 次
const RETRY_BASE_DELAY = 800       // 退避基数（ms），按次数线性增大

// 是否可安全重试：仅在“连接层失败、未收到任何响应”时重试。
// 超时（ECONNABORTED）不重试 —— 后端会先落库用户消息再跑智能体，
// 重试会造成对话重复、智能体重复执行。取消/中断/4xx/5xx 也不重试。
function isRetriableError(error) {
  if (error?.name === 'AbortError' || error?.name === 'CanceledError') return false
  if (error?.code === 'ERR_CANCELED' || error?.code === 'ECONNABORTED') return false
  return !error?.response
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(timer)
      const error = new Error('aborted')
      error.name = 'AbortError'
      reject(error)
    }, { once: true })
  })
}

// 用退避重试包裹一次请求工厂。factory(attempt) 每次返回一个新的请求 Promise。
export async function withRetry(factory, { retries = RETRY_ATTEMPTS, signal } = {}) {
  let lastError
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await factory(attempt)
    } catch (error) {
      lastError = error
      if (attempt >= retries || signal?.aborted || !isRetriableError(error)) throw error
      await sleep(RETRY_BASE_DELAY * (attempt + 1), signal)
    }
  }
  throw lastError
}

const sessionPath = (sessionId = '') => `/chat/sessions${sessionId ? `/${encodeURIComponent(sessionId)}` : ''}`

export function createChatSessionApi(title, config = {}) {
  return request.post(sessionPath(), { title: title || null }, { silent: true, ...config })
}

export const getChatSessionsApi = (config = {}) => request.get(sessionPath(), { silent: true, ...config })
export const getChatSessionApi = (sessionId, config = {}) => request.get(
  sessionPath(sessionId),
  { silent: true, ...config },
)
export const deleteChatSessionApi = (sessionId, config = {}) => request.delete(
  sessionPath(sessionId),
  { silent: true, ...config },
)

export function sendChatMessageApi({ sessionId, message, detections } = {}, config = {}) {
  const data = {
    session_id: sessionId,
    message,
  }
  if (Array.isArray(detections) && detections.length) data.detections = detections
  const { signal, ...rest } = config
  return withRetry(
    () => request.post('/chat/message', data, { timeout: TEXT_CHAT_TIMEOUT, silent: true, signal, ...rest }),
    { signal },
  )
}

export function sendChatImageApi(file, { sessionId, message } = {}, config = {}) {
  const data = new FormData()
  data.append('file', file)
  data.append('message', message)
  if (sessionId) data.append('session_id', sessionId)
  const { signal, ...rest } = config
  return withRetry(
    () => request.post('/chat/image', data, { timeout: IMAGE_CHAT_TIMEOUT, silent: true, signal, ...rest }),
    { signal },
  )
}

// ── 流式（SSE）对话 ───────────────────────────────────────────────
// 没有新字节超过该时长才判定连接中断（替代硬性总超时，长请求不再误杀）。
const STREAM_IDLE_TIMEOUT = 90000

// 从累积缓冲中切出完整 SSE 事件（纯函数，便于单测）。
// 返回 { events: 已解析的 JSON 事件数组, rest: 尚未成帧的剩余字符串 }。
export function drainSSEBuffer(buffer) {
  const events = []
  let rest = buffer
  let index
  while ((index = rest.indexOf('\n\n')) !== -1) {
    const frame = rest.slice(0, index)
    rest = rest.slice(index + 2)
    const data = frame
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''))
      .join('\n')
    if (!data) continue
    try {
      events.push(JSON.parse(data))
    } catch { /* 忽略不完整/异常帧 */ }
  }
  return { events, rest }
}
// 读取一个 SSE 响应流，按事件类型分发回调。
async function consumeStream(response, { signal, onEvent }) {
  if (!response.ok) {
    const error = new Error(`stream failed (${response.status})`)
    error.response = { status: response.status }
    throw error
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let idleTimer

  const resetIdle = (reject) => {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = setTimeout(() => reject(new Error('stream idle timeout')), STREAM_IDLE_TIMEOUT)
  }

  await new Promise((resolve, reject) => {
    resetIdle(reject)
    const onAbort = () => {
      reader.cancel().catch(() => {})
      const err = new Error('aborted')
      err.name = 'AbortError'
      reject(err)
    }
    if (signal) {
      if (signal.aborted) return onAbort()
      signal.addEventListener('abort', onAbort, { once: true })
    }
    const pump = async () => {
      const { value, done } = await reader.read()
      if (done) return resolve()
      resetIdle(reject)
      buffer += decoder.decode(value, { stream: true })
      const { events, rest } = drainSSEBuffer(buffer)
      buffer = rest
      for (const event of events) onEvent(event)
      return pump()
    }
    pump().catch(reject).finally(() => {
      if (idleTimer) clearTimeout(idleTimer)
      if (signal) signal.removeEventListener('abort', onAbort)
    })
  })
}

// 流式文本对话。handlers: { onEvent(event) }；event.type ∈ session|token|tool|reset|done|error。
export async function streamChatMessageApi({ sessionId, message, detections } = {}, { signal, onEvent } = {}) {
  const body = { session_id: sessionId, message }
  if (Array.isArray(detections) && detections.length) body.detections = detections
  const response = await fetch('/api/chat/message/stream', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  await consumeStream(response, { signal, onEvent })
}

// 流式图片对话。
export async function streamChatImageApi(file, { sessionId, message } = {}, { signal, onEvent } = {}) {
  const form = new FormData()
  form.append('file', file)
  form.append('message', message)
  if (sessionId) form.append('session_id', sessionId)
  const response = await fetch('/api/chat/image/stream', {
    method: 'POST',
    credentials: 'include',
    headers: { Accept: 'text/event-stream' },
    body: form,
    signal,
  })
  await consumeStream(response, { signal, onEvent })
}

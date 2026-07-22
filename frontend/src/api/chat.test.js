import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/utils/request', () => ({
  default: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
}))

import request from '@/utils/request'
import {
  createChatSessionApi, deleteChatSessionApi, drainSSEBuffer, getChatSessionApi, getChatSessionsApi,
  sendChatImageApi, sendChatMessageApi, withRetry,
} from './chat'

describe('chat api', () => {
  beforeEach(() => {
    request.get.mockReset()
    request.post.mockReset()
    request.delete.mockReset()
  })

  it('sends the documented text chat payload with a generous timeout', () => {
    const controller = new AbortController()
    sendChatMessageApi({ sessionId: 'meal-1', message: '分析这一餐' }, { signal: controller.signal })
    expect(request.post).toHaveBeenCalledWith('/chat/message', {
      session_id: 'meal-1',
      message: '分析这一餐',
    }, expect.objectContaining({ silent: true, timeout: 300000, signal: controller.signal }))
  })

  it('uses multipart fields for image chat without setting content-type manually', () => {
    const file = new Blob(['image'], { type: 'image/jpeg' })
    const controller = new AbortController()
    sendChatImageApi(file, { sessionId: 'meal-image', message: '估算热量' }, { signal: controller.signal })
    const [, formData, config] = request.post.mock.calls[0]
    expect(request.post.mock.calls[0][0]).toBe('/chat/image')
    expect(formData).toBeInstanceOf(FormData)
    expect(formData.get('file')).toBeInstanceOf(Blob)
    expect(formData.get('file').size).toBe(file.size)
    expect(formData.get('message')).toBe('估算热量')
    expect(formData.get('session_id')).toBe('meal-image')
    expect(config).toEqual(expect.objectContaining({ silent: true, timeout: 360000, signal: controller.signal }))
    expect(config.headers).toBeUndefined()
  })

  it('uses the documented persistent session routes', () => {
    createChatSessionApi('训练计划')
    getChatSessionsApi()
    getChatSessionApi('meal/a b')
    deleteChatSessionApi('meal/a b')

    expect(request.post).toHaveBeenCalledWith('/chat/sessions', { title: '训练计划' }, expect.objectContaining({ silent: true }))
    expect(request.get).toHaveBeenNthCalledWith(1, '/chat/sessions', expect.objectContaining({ silent: true }))
    expect(request.get).toHaveBeenNthCalledWith(2, '/chat/sessions/meal%2Fa%20b', expect.objectContaining({ silent: true }))
    expect(request.delete).toHaveBeenCalledWith('/chat/sessions/meal%2Fa%20b', expect.objectContaining({ silent: true }))
  })
})

describe('drainSSEBuffer', () => {
  it('parses complete events and keeps the trailing partial frame', () => {
    const buffer = 'data: {"type":"token","text":"你"}\n\ndata: {"type":"token","text":"好"}\n\ndata: {"type":"do'
    const { events, rest } = drainSSEBuffer(buffer)
    expect(events).toEqual([
      { type: 'token', text: '你' },
      { type: 'token', text: '好' },
    ])
    expect(rest).toBe('data: {"type":"do')
  })

  it('handles the leading-space convention and multi-line data', () => {
    const { events, rest } = drainSSEBuffer('data:{"type":"done","response":"ok"}\n\n')
    expect(events).toEqual([{ type: 'done', response: 'ok' }])
    expect(rest).toBe('')
  })

  it('skips malformed frames without throwing', () => {
    const { events } = drainSSEBuffer('data: not-json\n\ndata: {"type":"reset"}\n\n')
    expect(events).toEqual([{ type: 'reset' }])
  })
})

describe('withRetry', () => {
  it('returns immediately on success without retrying', async () => {
    const factory = vi.fn().mockResolvedValue('ok')
    const result = await withRetry(factory, { retries: 2 })
    expect(result).toBe('ok')
    expect(factory).toHaveBeenCalledTimes(1)
  })

  it('retries connection-level failures then succeeds', async () => {
    const factory = vi.fn()
      .mockRejectedValueOnce({ message: 'Network Error' }) // 无 response → 可重试
      .mockResolvedValueOnce('recovered')
    const result = await withRetry(factory, { retries: 2 })
    expect(result).toBe('recovered')
    expect(factory).toHaveBeenCalledTimes(2)
  })

  it('does NOT retry a timeout (avoids duplicating the conversation)', async () => {
    const factory = vi.fn().mockRejectedValue({ code: 'ECONNABORTED' })
    await expect(withRetry(factory, { retries: 2 })).rejects.toEqual({ code: 'ECONNABORTED' })
    expect(factory).toHaveBeenCalledTimes(1)
  })

  it('does NOT retry server responses (4xx/5xx)', async () => {
    const factory = vi.fn().mockRejectedValue({ response: { status: 500 } })
    await expect(withRetry(factory, { retries: 2 })).rejects.toEqual({ response: { status: 500 } })
    expect(factory).toHaveBeenCalledTimes(1)
  })
})

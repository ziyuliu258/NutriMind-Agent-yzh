// Vitest 全局初始化。
// Node 25+ 默认启用实验性原生 localStorage，但其实现缺少 clear() 且会遮蔽
// jsdom 的 window.localStorage，导致测试中 `localStorage.clear is not a function`。
// 这里用一份最小内存实现覆盖 localStorage / sessionStorage，保证 API 完整可用。
function createMemoryStorage() {
  const store = new Map()
  return {
    get length() { return store.size },
    key(index) { return Array.from(store.keys())[index] ?? null },
    getItem(key) { return store.has(key) ? store.get(key) : null },
    setItem(key, value) { store.set(String(key), String(value)) },
    removeItem(key) { store.delete(String(key)) },
    clear() { store.clear() },
  }
}

for (const name of ['localStorage', 'sessionStorage']) {
  Object.defineProperty(globalThis, name, {
    value: createMemoryStorage(),
    writable: true,
    configurable: true,
  })
}

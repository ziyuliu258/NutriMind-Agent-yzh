"""可信联网检索服务。

当前使用 Exa Search API。它可以直接替换为 Exa MCP 网关：Agent 只依赖本模块
返回的统一结构，不与具体搜索供应商耦合。
"""

import asyncio
import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config.settings import settings

logger = logging.getLogger(__name__)


class WebSearchService:
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        if not settings.WEB_SEARCH_ENABLED or not settings.EXA_API_KEY:
            logger.info("联网检索未启用或未配置 EXA_API_KEY")
            return []
        return await asyncio.to_thread(self._search_sync, query, max(1, min(limit, 10)))

    def _search_sync(self, query: str, limit: int) -> list[dict]:
        payload = json.dumps({
            "query": query,
            "numResults": limit,
            "type": "auto",
            "contents": {"text": {"maxCharacters": 3000}, "highlights": True},
        }).encode("utf-8")
        request = Request(
            settings.EXA_SEARCH_URL,
            data=payload,
            method="POST",
            headers={
                "x-api-key": settings.EXA_API_KEY,
                "content-type": "application/json",
                "user-agent": "NutriMind-Agent/0.1",
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("Exa 联网检索失败: %s", exc)
            return []

        results = []
        for item in body.get("results", []):
            content = item.get("text") or "\n".join(item.get("highlights") or [])
            results.append({
                "title": item.get("title") or item.get("url") or "网页来源",
                "url": item.get("url"),
                "content": (content or "").strip(),
                "published_date": item.get("publishedDate"),
                "source_type": "web",
                "provider": "exa",
            })
        return results


web_search_service = WebSearchService()

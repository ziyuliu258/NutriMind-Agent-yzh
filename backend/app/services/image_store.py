"""Agent 图像临时存储，使用安全 image_id 隔离本地文件路径。"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from app.config.settings import settings


class ImageStore:
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
    SUFFIX_BY_CONTENT_TYPE = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }

    def _image_path(self, image_id: str) -> Path:
        if not image_id or any(ch not in "0123456789abcdef" for ch in image_id):
            raise ValueError("无效的 image_id")
        matches = list(settings.UPLOADS_DIR.glob(f"{image_id}.*"))
        matches = [path for path in matches if path.suffix != ".json"]
        if not matches:
            raise FileNotFoundError(f"图片不存在或已过期: {image_id}")
        return matches[0]

    async def save(
        self,
        content: bytes,
        content_type: str,
        mock_detections: Optional[list[dict]] = None,
    ) -> str:
        if content_type not in self.ALLOWED_CONTENT_TYPES:
            raise ValueError("仅支持 JPG、PNG、WEBP 图片")
        if not content:
            raise ValueError("图片内容为空")
        if len(content) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"图片不能超过 {settings.MAX_IMAGE_SIZE_MB}MB")

        image_id = uuid.uuid4().hex
        suffix = self.SUFFIX_BY_CONTENT_TYPE[content_type]
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        path = settings.UPLOADS_DIR / f"{image_id}{suffix}"
        await asyncio.to_thread(path.write_bytes, content)

        if mock_detections is not None:
            metadata_path = settings.UPLOADS_DIR / f"{image_id}.json"
            payload = json.dumps({"mock_detections": mock_detections}, ensure_ascii=False)
            await asyncio.to_thread(metadata_path.write_text, payload, "utf-8")
        return image_id

    def get_path(self, image_id: str) -> Path:
        return self._image_path(image_id)

    def get_mock_detections(self, image_id: str) -> Optional[list[dict]]:
        self._image_path(image_id)
        metadata_path = settings.UPLOADS_DIR / f"{image_id}.json"
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("mock_detections")

    def delete(self, image_id: str) -> None:
        image_path = self._image_path(image_id)
        image_path.unlink(missing_ok=True)
        (settings.UPLOADS_DIR / f"{image_id}.json").unlink(missing_ok=True)


image_store = ImageStore()

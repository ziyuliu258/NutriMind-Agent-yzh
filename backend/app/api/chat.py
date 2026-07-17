"""NutriMind 营养智能体对话 API。"""

import uuid
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.core.security import get_current_user
from app.config.settings import settings
from app.entity.db_models import User
from app.entity.schemas import BoundingBox, ChatRequest, ChatResponse, ImageChatResponse
from app.services.agent_graph import run_agent
from app.services.image_store import image_store

router = APIRouter(prefix="/api/chat", tags=["AI 对话"])


def _validate_request(request: ChatRequest) -> str:
    message = request.message.strip()
    if not message and not request.detections:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="消息和检测结果不能同时为空",
        )
    return message


async def _invoke_chat(request: ChatRequest, thread_prefix: str, user_id: int | None) -> ChatResponse:
    message = _validate_request(request)
    session_id = request.session_id or str(uuid.uuid4())
    result = await run_agent(
        session_id=f"{thread_prefix}:{session_id}",
        user_message=message,
        detections=[item.model_dump() for item in request.detections] or None,
        user_id=user_id,
    )
    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        tool_calls=result.get("tool_calls", []),
        analysis_result=result.get("analysis_result"),
    )


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """发送消息，可同时携带 YOLO 食物检测结果并延续多轮会话。"""
    return await _invoke_chat(request, f"user:{current_user.id}", current_user.id)


@router.post("/mock-yolo", response_model=ChatResponse)
async def mock_yolo_chat(request: ChatRequest) -> ChatResponse:
    """用模拟 YOLO detections 联调完整智能体链路（仅 DEBUG 模式开放）。"""
    if not settings.DEBUG:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    return await _invoke_chat(request, "mock-yolo", None)


async def _invoke_image_chat(
    *,
    file: UploadFile,
    message: str,
    session_id: str | None,
    thread_prefix: str,
    user_id: int | None,
    mock_detections: list[dict] | None = None,
) -> ImageChatResponse:
    if not message.strip():
        raise HTTPException(status_code=422, detail="message 不能为空")
    content = await file.read(settings.MAX_IMAGE_SIZE_MB * 1024 * 1024 + 1)
    try:
        image_id = await image_store.save(
            content=content,
            content_type=file.content_type or "",
            mock_detections=mock_detections,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    external_session_id = session_id or str(uuid.uuid4())
    result = await run_agent(
        session_id=f"{thread_prefix}:{external_session_id}",
        user_message=message.strip(),
        image_id=image_id,
        user_id=user_id,
    )
    return ImageChatResponse(
        session_id=external_session_id,
        image_id=image_id,
        detection_mode=result.get("detection_mode") or settings.DETECTION_MODE,
        detections=result.get("detections", []),
        response=result["response"],
        tool_calls=result.get("tool_calls", []),
        analysis_result=result.get("analysis_result"),
    )


@router.post("/image", response_model=ImageChatResponse)
async def analyze_image(
    file: UploadFile = File(..., description="餐食图片（JPG/PNG/WEBP，最大 10MB）"),
    message: str = Form(..., max_length=4000),
    session_id: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
) -> ImageChatResponse:
    """正式图片对话入口：Agent 主动调用真实 YOLO 工具（需要模型文件）。"""
    return await _invoke_image_chat(
        file=file,
        message=message,
        session_id=session_id,
        thread_prefix=f"user:{current_user.id}:image",
        user_id=current_user.id,
    )


@router.post("/image/mock", response_model=ImageChatResponse)
async def analyze_image_with_mock_yolo(
    file: UploadFile = File(..., description="任意测试餐食图片（工具调用仍会发生）"),
    message: str = Form(..., max_length=4000),
    mock_detections: str = Form(..., description="YOLO detections JSON 数组"),
    session_id: str | None = Form(default=None),
) -> ImageChatResponse:
    """开发联调入口：上传图片，Agent 调用 detect_food 获取随请求提供的模拟结果。"""
    if not settings.DEBUG:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    try:
        raw_detections = json.loads(mock_detections)
        if not isinstance(raw_detections, list):
            raise ValueError("mock_detections 必须是 JSON 数组")
        detections = [BoundingBox.model_validate(item).model_dump() for item in raw_detections]
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"无效的 mock_detections: {exc}") from exc

    return await _invoke_image_chat(
        file=file,
        message=message,
        session_id=session_id,
        thread_prefix="mock-image",
        user_id=None,
        mock_detections=detections,
    )

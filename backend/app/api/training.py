"""
模型训练 API 路由 — 创建、启动、暂停训练，查看进度与折线图数据。
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.session import get_db
from app.entity.db_models import TrainingTask, User
from app.entity.schemas import (
    ApiResponse,
    ModelInfoResponse,
    TrainingMetricsLineChart,
    TrainingTaskCreate,
    TrainingTaskListResponse,
    TrainingTaskResponse,
)
from app.services.training_service import training_service

router = APIRouter(prefix="/api/training", tags=["模型训练"])


# ------------------------------------------------------------------
# 训练任务生命周期
# ------------------------------------------------------------------


@router.post("/tasks", response_model=ApiResponse, status_code=201)
async def create_training_task(
    config: TrainingTaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建训练任务（状态 = pending），不立即启动。"""
    task = await training_service.create_task(db, config, user_id=current_user.id)
    return ApiResponse(
        code=201,
        message="训练任务已创建",
        data={"task_uuid": task.task_uuid, "id": task.id},
    )


@router.post("/tasks/{task_uuid}/start", response_model=ApiResponse)
async def start_training(
    task_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """启动训练任务。仅 pending 状态可启动。"""
    try:
        task = await training_service.start_task(db, task_uuid)
        return ApiResponse(
            message=f"训练已启动: {task.model_name}",
            data={"task_uuid": task.task_uuid, "status": task.status},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tasks/{task_uuid}/pause", response_model=ApiResponse)
async def pause_training(
    task_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """暂停训练任务。"""
    try:
        task = await training_service.pause_task(db, task_uuid)
        return ApiResponse(
            message="训练已暂停",
            data={"task_uuid": task.task_uuid, "status": task.status},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------
# 查询
# ------------------------------------------------------------------


@router.get("/tasks", response_model=TrainingTaskListResponse)
async def list_training_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """分页查询训练任务列表（按创建时间倒序）。"""
    tasks, total = await training_service.get_tasks(
        db, user_id=current_user.id, page=page, page_size=page_size
    )
    return TrainingTaskListResponse(
        items=[TrainingTaskResponse.model_validate(t) for t in tasks],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/tasks/{task_uuid}", response_model=TrainingTaskResponse)
async def get_training_task(
    task_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取单个训练任务详情（含最终指标）。"""
    task = await training_service.get_task_by_uuid(db, task_uuid)
    if not task:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return TrainingTaskResponse.model_validate(task)


@router.get("/tasks/{task_uuid}/metrics", response_model=ApiResponse)
async def get_training_metrics(
    task_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取 per-epoch 训练指标，前端直接用于折线图。

    返回字段（每个 epoch）：
    - epoch, train_box_loss, train_cls_loss, train_dfl_loss
    - val_box_loss, val_cls_loss, val_dfl_loss
    - precision, recall, mAP50, mAP50_95
    """
    data = await training_service.get_training_metrics(db, task_uuid)
    if data is None:
        raise HTTPException(status_code=404, detail="训练任务不存在或无指标数据")

    return ApiResponse(
        data=TrainingMetricsLineChart(
            task_uuid=data["task_uuid"],
            model_name=data["model_name"],
            epochs=data["epochs"],
            final_metrics=data.get("final_metrics"),
        ).model_dump()
    )


# ------------------------------------------------------------------
# 模型管理
# ------------------------------------------------------------------


@router.get("/models", response_model=ApiResponse)
async def list_models(current_user: User = Depends(get_current_user)):
    """获取已有模型文件列表。"""
    models = await training_service.get_models()
    return ApiResponse(data=[ModelInfoResponse.model_validate(m).model_dump() for m in models])


@router.delete("/models/{model_name}", response_model=ApiResponse)
async def delete_model(model_name: str, current_user: User = Depends(get_current_user)):
    """删除指定模型文件。"""
    ok = await training_service.delete_model(model_name)
    if not ok:
        raise HTTPException(status_code=404, detail="模型文件不存在")
    return ApiResponse(message=f"模型 {model_name} 已删除")

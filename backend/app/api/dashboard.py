"""
Dashboard 看板路由
提供系统总览、检测统计、训练统计、用户统计及用户管理等接口
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.core.security import get_current_user, require_admin
from app.database.session import get_db
from app.entity.db_models import (
    DetectionScene,
    DetectionTask,
    FoodNutrition,
    TrainingTask,
    User,
)
from app.entity.schemas import (
    ApiResponse,
    DashboardOverview,
    DashboardStats,
    DetectionStats,
    TaskStatusCount,
    TrainingStats,
    ToggleUserStatusRequest,
    UpdateUserSuperuserRequest,
    UpdateUserRoleRequest,
    UserDetailResponse,
    UserListResponse,
    UserStats,
)
from app.services.user_service import user_service

logger = get_logger("dashboard")

router = APIRouter(prefix="/api/dashboard", tags=["看板"])


def _serialize_detection_task(
    task: DetectionTask,
    username: Optional[str] = None,
    include_detections: bool = False,
) -> dict:
    """将检测任务转换为管理员监控接口使用的稳定字段。"""
    scene = task.scene
    payload = {
        "id": task.id,
        "task_uuid": task.task_uuid,
        "user_id": task.user_id,
        "username": username,
        "scene_id": task.scene_id,
        "scene_name": (scene.display_name or scene.name) if scene else None,
        "status": task.status,
        "total_objects": task.total_objects,
        "inference_time": task.inference_time,
        "image_url": task.image_url,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
    if include_detections:
        payload.update({
            "detections": task.detections or [],
            "conf_threshold": task.conf_threshold,
            "iou_threshold": task.iou_threshold,
        })
    return payload


# ---------------------------------------------------------------------------
# GET /api/dashboard/overview  — 总览数据
# ---------------------------------------------------------------------------
@router.get("/overview", response_model=ApiResponse)
async def get_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取系统总览数据（需登录）"""
    logger.info(f"用户 {current_user.username} 请求看板总览")

    overview = DashboardOverview(
        total_users=db.query(func.count(User.id)).scalar() or 0,
        active_users=db.query(func.count(User.id)).filter(
            User.is_active.is_(True)).scalar() or 0,
        total_detection_scenes=db.query(
            func.count(DetectionScene.id)).scalar() or 0,
        total_detection_tasks=db.query(
            func.count(DetectionTask.id)).scalar() or 0,
        total_training_tasks=db.query(
            func.count(TrainingTask.id)).scalar() or 0,
        total_food_items=db.query(func.count(FoodNutrition.id)).scalar() or 0,
    )

    return ApiResponse(code=200, message="success", data=overview.model_dump())


# ---------------------------------------------------------------------------
# GET /api/dashboard/detection  — 检测任务统计
# ---------------------------------------------------------------------------
@router.get("/detection", response_model=ApiResponse)
async def get_detection_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取检测任务统计（需登录）"""
    logger.info(f"用户 {current_user.username} 请求检测统计")

    total = db.query(func.count(DetectionTask.id)).scalar() or 0
    completed = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "completed").scalar() or 0
    failed = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "failed").scalar() or 0
    pending = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "pending").scalar() or 0
    processing = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "processing").scalar() or 0

    total_objects = db.query(func.coalesce(
        func.sum(DetectionTask.total_objects), 0)).scalar() or 0
    avg_time = db.query(func.avg(DetectionTask.inference_time)).filter(
        DetectionTask.inference_time.isnot(None)
    ).scalar()

    stats = DetectionStats(
        total=total,
        completed=completed,
        failed=failed,
        pending=pending,
        processing=processing,
        total_objects_detected=int(total_objects),
        avg_inference_time=round(
            float(avg_time), 4) if avg_time is not None else None,
    )

    return ApiResponse(code=200, message="success", data=stats.model_dump())


# ---------------------------------------------------------------------------
# GET /api/dashboard/detection/tasks  — 管理员检测任务列表
# ---------------------------------------------------------------------------
@router.get("/detection/tasks", response_model=ApiResponse)
async def list_admin_detection_tasks(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页大小"),
    task_status: Optional[str] = Query(default=None, alias="status", description="任务状态"),
    user_id: Optional[int] = Query(default=None, description="提交用户 ID"),
    scene_id: Optional[int] = Query(default=None, description="检测场景 ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """管理员分页查看全站检测任务，支持状态、用户和场景过滤。"""
    logger.info(
        f"管理员 {current_user.username} 查询全站检测任务 page={page}"
    )
    query = (
        db.query(DetectionTask, User.username)
        .outerjoin(User, DetectionTask.user_id == User.id)
    )
    if task_status:
        query = query.filter(DetectionTask.status == task_status.strip().lower())
    if user_id is not None:
        query = query.filter(DetectionTask.user_id == user_id)
    if scene_id is not None:
        query = query.filter(DetectionTask.scene_id == scene_id)

    total = query.count()
    rows = (
        query.order_by(DetectionTask.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ApiResponse(
        code=200,
        message="success",
        data={
            "items": [
                _serialize_detection_task(task, username)
                for task, username in rows
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/detection/tasks/{task_uuid}  — 管理员检测任务详情
# ---------------------------------------------------------------------------
@router.get("/detection/tasks/{task_uuid}", response_model=ApiResponse)
async def get_admin_detection_task(
    task_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """管理员查看任意检测任务的完整结果。"""
    logger.info(
        f"管理员 {current_user.username} 查询检测任务详情 uuid={task_uuid}"
    )
    row = (
        db.query(DetectionTask, User.username)
        .outerjoin(User, DetectionTask.user_id == User.id)
        .filter(DetectionTask.task_uuid == task_uuid)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="检测任务不存在")

    task, username = row
    return ApiResponse(
        code=200,
        message="success",
        data=_serialize_detection_task(
            task, username, include_detections=True
        ),
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/training  — 训练任务统计
# ---------------------------------------------------------------------------
@router.get("/training", response_model=ApiResponse)
async def get_training_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取训练任务统计（需登录）"""
    logger.info(f"用户 {current_user.username} 请求训练统计")

    total = db.query(func.count(TrainingTask.id)).scalar() or 0
    completed = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "completed").scalar() or 0
    failed = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "failed").scalar() or 0
    running = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "running").scalar() or 0
    pending = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "pending").scalar() or 0
    paused = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "paused").scalar() or 0

    stats = TrainingStats(
        total=total,
        completed=completed,
        failed=failed,
        running=running,
        pending=pending,
        paused=paused,
    )

    return ApiResponse(code=200, message="success", data=stats.model_dump())


# ---------------------------------------------------------------------------
# GET /api/dashboard/users  — 用户统计
# ---------------------------------------------------------------------------
@router.get("/users", response_model=ApiResponse)
async def get_user_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取用户统计（需登录）"""
    logger.info(f"用户 {current_user.username} 请求用户统计")

    total = db.query(func.count(User.id)).scalar() or 0
    active = db.query(func.count(User.id)).filter(
        User.is_active.is_(True)).scalar() or 0
    superusers = db.query(func.count(User.id)).filter(
        User.is_superuser.is_(True)).scalar() or 0

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    new_today = db.query(func.count(User.id)).filter(
        User.created_at >= today_start).scalar() or 0

    stats = UserStats(
        total=total,
        active=active,
        superusers=superusers,
        new_today=new_today,
    )

    return ApiResponse(code=200, message="success", data=stats.model_dump())


# ===========================================================================
# 用户管理接口
# ===========================================================================


# ---------------------------------------------------------------------------
# GET /api/dashboard/users/list  — 用户列表（分页 + 搜索）
# ---------------------------------------------------------------------------
@router.get("/users/list", response_model=ApiResponse)
async def list_users(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页大小"),
    search: Optional[str] = Query(
        default=None, description="搜索关键词（用户名/邮箱/手机）"),
    is_active: Optional[bool] = Query(default=None, description="启用状态过滤"),
    is_superuser: Optional[bool] = Query(default=None, description="管理员状态过滤"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取用户分页列表，支持关键词搜索和状态过滤"""
    logger.info(f"管理员 {current_user.username} 查询用户列表 page={page}")
    result = user_service.list_users(
        db,
        page=page,
        page_size=page_size,
        search=search,
        is_active=is_active,
        is_superuser=is_superuser,
    )
    return ApiResponse(
        code=200,
        message="success",
        data=UserListResponse(**result).model_dump(),
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/users/{user_id}  — 用户详情
# ---------------------------------------------------------------------------
@router.get("/users/{user_id}", response_model=ApiResponse)
async def get_user_detail(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取指定用户的详细信息（含检测/训练任务统计）"""
    logger.info(f"管理员 {current_user.username} 查询用户详情 id={user_id}")
    detail = user_service.get_user_detail(db, user_id)
    return ApiResponse(
        code=200,
        message="success",
        data=UserDetailResponse(**detail).model_dump(),
    )


# ---------------------------------------------------------------------------
# PUT /api/dashboard/users/{user_id}/status  — 启用/禁用用户
# ---------------------------------------------------------------------------
@router.put("/users/{user_id}/status", response_model=ApiResponse)
async def toggle_user_status(
    user_id: int,
    body: ToggleUserStatusRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """启用或禁用指定用户"""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能修改自己的状态")

    logger.info(
        f"管理员 {current_user.username} 修改用户 {user_id} 状态 -> {'启用' if body.is_active else '禁用'}"
    )
    user = user_service.toggle_user_status(db, user_id, body.is_active)
    return ApiResponse(
        code=200,
        message="success",
        data={"id": user.id, "username": user.username,
              "is_active": user.is_active},
    )


# ---------------------------------------------------------------------------
# PUT /api/dashboard/users/{user_id}/roles  — 修改用户角色
# ---------------------------------------------------------------------------
@router.put("/users/{user_id}/roles", response_model=ApiResponse)
async def update_user_roles(
    user_id: int,
    body: UpdateUserRoleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """修改指定用户的角色（全量替换）"""
    logger.info(
        f"管理员 {current_user.username} 修改用户 {user_id} 角色 -> {body.role_names}"
    )
    new_roles = user_service.update_user_roles(db, user_id, body.role_names)
    return ApiResponse(
        code=200,
        message="success",
        data={"id": user_id, "roles": new_roles},
    )


# ---------------------------------------------------------------------------
# PUT /api/dashboard/users/{user_id}/superuser  — 修改管理员状态
# ---------------------------------------------------------------------------
@router.put("/users/{user_id}/superuser", response_model=ApiResponse)
async def update_superuser_status(
    user_id: int,
    body: UpdateUserSuperuserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """修改指定用户的管理员状态"""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能修改自己的管理员状态")

    logger.info(
        f"管理员 {current_user.username} 修改用户 {user_id} 管理员状态 -> {body.is_superuser}"
    )
    user = user_service.update_superuser_status(db, user_id, body.is_superuser)
    return ApiResponse(
        code=200,
        message="success",
        data={"id": user.id, "username": user.username,
              "is_superuser": user.is_superuser},
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/stats  — 完整看板数据（聚合）
# ---------------------------------------------------------------------------
@router.get("/stats", response_model=ApiResponse)
async def get_full_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取完整看板统计数据（需登录）"""
    logger.info(f"用户 {current_user.username} 请求完整看板统计")

    # --- overview ---
    overview = DashboardOverview(
        total_users=db.query(func.count(User.id)).scalar() or 0,
        active_users=db.query(func.count(User.id)).filter(
            User.is_active.is_(True)).scalar() or 0,
        total_detection_scenes=db.query(
            func.count(DetectionScene.id)).scalar() or 0,
        total_detection_tasks=db.query(
            func.count(DetectionTask.id)).scalar() or 0,
        total_training_tasks=db.query(
            func.count(TrainingTask.id)).scalar() or 0,
        total_food_items=db.query(func.count(FoodNutrition.id)).scalar() or 0,
    )

    # --- detection ---
    det_total = db.query(func.count(DetectionTask.id)).scalar() or 0
    det_completed = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "completed").scalar() or 0
    det_failed = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "failed").scalar() or 0
    det_pending = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "pending").scalar() or 0
    det_processing = db.query(func.count(DetectionTask.id)).filter(
        DetectionTask.status == "processing").scalar() or 0
    det_objects = db.query(func.coalesce(
        func.sum(DetectionTask.total_objects), 0)).scalar() or 0
    det_avg_time = db.query(func.avg(DetectionTask.inference_time)).filter(
        DetectionTask.inference_time.isnot(None)
    ).scalar()

    detection = DetectionStats(
        total=det_total,
        completed=det_completed,
        failed=det_failed,
        pending=det_pending,
        processing=det_processing,
        total_objects_detected=int(det_objects),
        avg_inference_time=round(
            float(det_avg_time), 4) if det_avg_time is not None else None,
    )

    # --- training ---
    tr_total = db.query(func.count(TrainingTask.id)).scalar() or 0
    tr_completed = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "completed").scalar() or 0
    tr_failed = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "failed").scalar() or 0
    tr_running = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "running").scalar() or 0
    tr_pending = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "pending").scalar() or 0
    tr_paused = db.query(func.count(TrainingTask.id)).filter(
        TrainingTask.status == "paused").scalar() or 0

    training = TrainingStats(
        total=tr_total,
        completed=tr_completed,
        failed=tr_failed,
        running=tr_running,
        pending=tr_pending,
        paused=tr_paused,
    )

    # --- users ---
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    users = UserStats(
        total=overview.total_users,
        active=overview.active_users,
        superusers=db.query(func.count(User.id)).filter(
            User.is_superuser.is_(True)).scalar() or 0,
        new_today=db.query(func.count(User.id)).filter(
            User.created_at >= today_start).scalar() or 0,
    )

    stats = DashboardStats(
        overview=overview,
        detection=detection,
        training=training,
        users=users,
    )

    return ApiResponse(code=200, message="success", data=stats.model_dump())


# ---------------------------------------------------------------------------
# GET /api/dashboard/detection/status-distribution  — 检测任务状态分布
# ---------------------------------------------------------------------------
@router.get("/detection/status-distribution", response_model=ApiResponse)
async def get_detection_status_distribution(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取检测任务各状态数量分布（需登录）"""
    logger.info(f"用户 {current_user.username} 请求检测状态分布")

    rows = (
        db.query(DetectionTask.status, func.count(DetectionTask.id))
        .group_by(DetectionTask.status)
        .all()
    )

    distribution: List[TaskStatusCount] = [
        TaskStatusCount(status=str(row[0] or "unknown"), count=row[1])
        for row in rows
    ]

    return ApiResponse(
        code=200,
        message="success",
        data=[item.model_dump() for item in distribution],
    )


# ---------------------------------------------------------------------------
# GET /api/dashboard/training/status-distribution  — 训练任务状态分布
# ---------------------------------------------------------------------------
@router.get("/training/status-distribution", response_model=ApiResponse)
async def get_training_status_distribution(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """获取训练任务各状态数量分布（需登录）"""
    logger.info(f"用户 {current_user.username} 请求训练状态分布")

    rows = (
        db.query(TrainingTask.status, func.count(TrainingTask.id))
        .group_by(TrainingTask.status)
        .all()
    )

    distribution: List[TaskStatusCount] = [
        TaskStatusCount(status=str(row[0] or "unknown"), count=row[1])
        for row in rows
    ]

    return ApiResponse(
        code=200,
        message="success",
        data=[item.model_dump() for item in distribution],
    )

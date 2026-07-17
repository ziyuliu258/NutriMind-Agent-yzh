from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.entity.db_models import User, UserRole, Role, DetectionTask, TrainingTask
from app.core.security import hash_password, verify_password, create_token
from app.core.logger import get_logger

logger = get_logger("user_service")


class UserService:
    def register(self, db: Session, username: str, email: str, password: str) -> User:
        """⽤户注册"""
        # 检查⽤户名是否已存在
        existing_user = db.query(User).filter(
            User.username == username).first()
        if existing_user:
            logger.warning(f"注册失败: 用户名已存在 - {username}")
            raise HTTPException(status_code=400, detail="用户名已存在")

        # 检查邮箱是否已存在
        existing_email = db.query(User).filter(User.email == email).first()
        if existing_email:
            logger.warning(f"注册失败: 邮箱已被注册 - {email}")
            raise HTTPException(status_code=400, detail="邮箱已被注册")

        # 创建⽤户
        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"用户注册成功: {username} (id={user.id})")
        return user

    def login(self, db: Session, username: str, password: str) -> User:
        """⽤户登录"""
        user = db.query(User).filter(User.username == username).first()
        if not user:
            logger.warning(f"登录失败: 用户不存在 - {username}")
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        if not verify_password(password, user.hashed_password):
            logger.warning(f"登录失败: 密码错误 - {username}")
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        if not user.is_active:
            logger.warning(f"登录失败: 账户已禁用 - {username}")
            raise HTTPException(status_code=403, detail="账户已被禁用")

        # 更新最后登录时间
        user.last_login_at = datetime.now()
        db.commit()

        logger.info(f"用户登录成功: {username} (id={user.id})")
        return user

    def create_access_token_for_user(self, user: User) -> str:
        """为⽤户创建 JWT Token"""
        return create_token({"sub": str(user.id)})

    def get_user_by_id(self, db: Session, user_id: int) -> User:
        """根据 ID 获取用户"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        return user

    def get_user_roles(self, db: Session, user: User) -> list:
        """获取用户角色列表"""
        user_roles = db.query(UserRole).filter(
            UserRole.user_id == user.id).all()
        roles = []
        for ur in user_roles:
            role = db.query(Role).get(ur.role_id)
            if role:
                roles.append(role.name)
        return roles

    def change_password(self, db: Session, user_id: int, old_password: str, new_password: str):
        """修改密码"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("用户不存在")

        # 验证旧密码
        if not verify_password(old_password, user.hashed_password):
            raise ValueError("旧密码错误")

        # 新旧密码不能相同
        if old_password == new_password:
            raise ValueError("新密码不能与旧密码相同")

        # 更新密码
        user.hashed_password = hash_password(new_password)
        db.commit()

    # ---------------------------------------------------------
    # 用户管理（Dashboard 管理端）
    # ---------------------------------------------------------

    def list_users(
        self,
        db: Session,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_superuser: Optional[bool] = None,
    ) -> dict:
        """分页获取用户列表，支持搜索和过滤"""
        query = db.query(User)

        if search:
            keyword = f"%{search}%"
            query = query.filter(
                or_(
                    User.username.ilike(keyword),
                    User.email.ilike(keyword),
                    User.phone.ilike(keyword),
                )
            )
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        if is_superuser is not None:
            query = query.filter(User.is_superuser == is_superuser)

        total = query.count()
        users = (
            query.order_by(User.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        items = []
        for user in users:
            roles = self.get_user_roles(db, user)
            items.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "phone": user.phone,
                "avatar": user.avatar,
                "is_active": user.is_active,
                "is_superuser": user.is_superuser,
                "roles": roles,
                "last_login_at": user.last_login_at,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_user_detail(self, db: Session, user_id: int) -> dict:
        """获取用户详情（含关系统计）"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        roles = self.get_user_roles(db, user)
        detection_count = (
            db.query(DetectionTask)
            .filter(DetectionTask.user_id == user_id)
            .count()
        )
        training_count = (
            db.query(TrainingTask)
            .filter(TrainingTask.user_id == user_id)
            .count()
        )

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "phone": user.phone,
            "avatar": user.avatar,
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
            "roles": roles,
            "last_login_at": user.last_login_at,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "total_detection_tasks": detection_count,
            "total_training_tasks": training_count,
        }

    def toggle_user_status(self, db: Session, user_id: int, is_active: bool) -> User:
        """启用/禁用用户"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        user.is_active = is_active
        db.commit()
        db.refresh(user)
        logger.info(f"用户 {user.username} 状态更新为 {'启用' if is_active else '禁用'}")
        return user

    def update_user_roles(self, db: Session, user_id: int, role_names: list[str]) -> list[str]:
        """更新用户角色（全量替换）"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        # 删除旧角色
        db.query(UserRole).filter(UserRole.user_id == user_id).delete()

        # 赋予新角色
        new_roles = []
        for name in role_names:
            role = db.query(Role).filter(Role.name == name).first()
            if not role:
                raise HTTPException(status_code=400, detail=f"角色不存在: {name}")
            ur = UserRole(user_id=user_id, role_id=role.id)
            db.add(ur)
            new_roles.append(role.name)

        db.commit()
        logger.info(f"用户 {user.username} 角色更新为 {new_roles}")
        return new_roles

    def update_superuser_status(
        self, db: Session, user_id: int, is_superuser: bool
    ) -> User:
        """修改用户管理员状态"""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")

        user.is_superuser = is_superuser
        db.commit()
        db.refresh(user)
        logger.info(
            f"用户 {user.username} 管理员状态更新为 {'是' if is_superuser else '否'}"
        )
        return user


# 全局单例
user_service = UserService()

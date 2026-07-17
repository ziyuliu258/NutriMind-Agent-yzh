"""用户个人资料与身体目标服务。"""

from typing import Optional
from datetime import datetime

from sqlalchemy.orm import Session

from app.entity.db_models import User, BodyProfile, GoalProfile, UserRole, Role


class ProfileService:
    """个人资料管理服务。"""

    def get_profile(self, db: Session, user: User) -> dict:
        body = db.query(BodyProfile).filter(BodyProfile.user_id == user.id).first()
        goal = db.query(GoalProfile).filter(GoalProfile.user_id == user.id).first()
        roles = self._get_roles(db, user)

        return {
            "account": self._account_dict(user, roles),
            "body_profile": self._body_dict(body) if body else None,
            "goal": self._goal_dict(goal) if goal else None,
        }

    def update_profile(self, db: Session, user: User, data: dict) -> dict:
        """部分更新（upsert），显式 null 清空，未传保持不变。"""
        if "account" in data and data["account"]:
            acc = data["account"]
            if "phone" in acc:
                user.phone = acc["phone"]
            db.flush()

        if "body_profile" in data and data["body_profile"] is not None:
            bp = db.query(BodyProfile).filter(BodyProfile.user_id == user.id).first()
            if not bp:
                bp = BodyProfile(user_id=user.id)
                db.add(bp)
            self._apply_body(bp, data["body_profile"])

        if "goal" in data and data["goal"] is not None:
            gp = db.query(GoalProfile).filter(GoalProfile.user_id == user.id).first()
            if not gp:
                gp = GoalProfile(user_id=user.id)
                db.add(gp)
            self._apply_goal(gp, data["goal"])

        db.flush()
        return self.get_profile(db, user)

    # ── helpers ──

    @staticmethod
    def _get_roles(db: Session, user: User) -> list[str]:
        rows = db.query(UserRole).filter(UserRole.user_id == user.id).all()
        ids = [r.role_id for r in rows]
        if ids:
            roles = db.query(Role).filter(Role.id.in_(ids)).all()
            return [r.name for r in roles]
        return []

    @staticmethod
    def _account_dict(u: User, roles: list[str]) -> dict:
        return {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "phone": u.phone,
            "avatar": u.avatar,
            "is_active": u.is_active,
            "is_superuser": u.is_superuser,
            "roles": roles,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        }

    @staticmethod
    def _body_dict(bp: BodyProfile) -> dict:
        return {
            "current_weight_kg": bp.current_weight_kg,
            "height_cm": bp.height_cm,
            "birth_date": bp.birth_date.isoformat() if bp.birth_date else None,
            "sex_for_calculation": bp.sex_for_calculation,
            "activity_level": bp.activity_level,
        }

    @staticmethod
    def _goal_dict(gp: GoalProfile) -> dict:
        return {
            "mode": gp.mode,
            "target_weight_kg": gp.target_weight_kg,
            "daily_calories_kcal": gp.daily_calories_kcal,
            "protein_target_g": gp.protein_target_g,
            "training_days_per_week": gp.training_days_per_week,
            "started_at": gp.started_at.isoformat() if gp.started_at else None,
            "updated_at": gp.updated_at.isoformat() if gp.updated_at else None,
        }

    @staticmethod
    def _apply_body(bp: BodyProfile, data: dict):
        for f in ("current_weight_kg", "height_cm", "sex_for_calculation", "activity_level"):
            if f in data:
                setattr(bp, f, data[f])
        if "birth_date" in data:
            v = data["birth_date"]
            bp.birth_date = datetime.fromisoformat(v) if isinstance(v, str) else v

    @staticmethod
    def _apply_goal(gp: GoalProfile, data: dict):
        for f in ("mode", "target_weight_kg", "daily_calories_kcal",
                  "protein_target_g", "training_days_per_week"):
            if f in data:
                setattr(gp, f, data[f])
        if "started_at" in data:
            v = data["started_at"]
            gp.started_at = datetime.fromisoformat(v) if isinstance(v, str) else v


profile_service = ProfileService()
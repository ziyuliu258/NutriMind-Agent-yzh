import asyncio
from unittest.mock import AsyncMock, patch

from sqlalchemy.exc import SQLAlchemyError
from langchain_core.messages import AIMessage, ToolMessage

from fastapi.testclient import TestClient

from app.core.security import get_current_user
from app.entity.db_models import BodyProfile, GoalProfile, User
from main import app
from app.config.settings import settings
from app.services.agent_tools import calculate_total_nutrition, detect_food, query_food_calories
from app.services.image_store import image_store
from app.services import agent_graph
from app.services import chat_service
from app.services.agent_prompts import format_user_profile_for_prompt


def _mock_user():
    return User(id=7, username="agent-user", email="agent@example.com", is_active=True)


def test_chat_message_passes_detection_context():
    app.dependency_overrides[get_current_user] = _mock_user
    agent_result = {
        "response": "这餐约含 250 kcal。",
        "tool_calls": [{"name": "query_food_calories", "args": {"food_name": "apple"}}],
        "analysis_result": None,
    }
    try:
        with patch("app.api.chat.run_agent", new=AsyncMock(return_value=agent_result)) as mocked:
            response = TestClient(app).post(
                "/api/chat/message",
                json={
                    "session_id": "meal-001",
                    "message": "分析一下",
                    "detections": [{
                        "class_name": "apple",
                        "class_name_cn": "苹果",
                        "confidence": 0.95,
                        "bbox": [10, 20, 200, 300],
                    }],
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["session_id"] == "meal-001"
    assert response.json()["tool_calls"][0]["name"] == "query_food_calories"
    assert mocked.await_args.kwargs["session_id"] == "user:7:meal-001"
    assert mocked.await_args.kwargs["detections"][0]["class_name"] == "apple"


def test_chat_injects_saved_nutrition_profile(db):
    user = _mock_user()
    user.hashed_password = "test-hash"
    db.add(user)
    db.add(BodyProfile(
        user_id=user.id, current_weight_kg=72, height_cm=175,
        sex_for_calculation="male", activity_level="moderate",
    ))
    db.add(GoalProfile(
        user_id=user.id, mode="cut", target_weight_kg=68,
        daily_calories_kcal=2000, protein_target_g=130,
        training_days_per_week=3,
    ))
    db.commit()
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        with patch(
            "app.api.chat.run_agent",
            new=AsyncMock(return_value={"response": "已结合目标", "tool_calls": [], "analysis_result": None}),
        ) as mocked:
            response = TestClient(app).post("/api/chat/message", json={"message": "晚饭怎么吃？"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    profile = mocked.await_args.kwargs["user_profile"]
    assert profile["body_profile"]["current_weight_kg"] == 72
    assert profile["goal"]["daily_calories_kcal"] == 2000
    assert "phone" not in profile["body_profile"]


def test_profile_prompt_formats_goals_without_account_data():
    prompt = format_user_profile_for_prompt({
        "body_profile": {"current_weight_kg": 72, "activity_level": "moderate"},
        "goal": {"mode": "cut", "daily_calories_kcal": 2000, "protein_target_g": 130},
    })

    assert "主要目标: 减脂" in prompt
    assert "当前体重: 72 kg" in prompt
    assert "每日热量目标: 2000 kcal" in prompt
    assert "日常活动水平: 中等活动" in prompt


def test_real_exa_sources_are_appended_to_agent_response():
    message = ToolMessage(
        content='{"success":true,"provider":"exa","results":['
        '{"title":"WHO healthy diet","url":"https://www.who.int/example","provider":"exa"},'
        '{"title":"No URL","content":"ignored"}]}',
        tool_call_id="web-1",
        name="search_web_evidence",
    )

    sources = agent_graph._web_sources_from_tool_message(message)
    response = agent_graph._append_web_sources("这是联网后的回答。", sources)

    assert sources == [{
        "title": "WHO healthy diet",
        "url": "https://www.who.int/example",
        "provider": "exa",
    }]
    assert "### 联网来源" in response
    assert "[WHO healthy diet](https://www.who.int/example)" in response


def test_knowledge_fallback_web_results_are_recognized_as_sources():
    message = ToolMessage(
        content='{"local_results":[],"web_results":['
        '{"title":"Exa result","url":"https://example.com/source","provider":"exa"}],'
        '"used_web_fallback":true}',
        tool_call_id="knowledge-1",
        name="search_nutrition_knowledge",
    )

    assert agent_graph._web_sources_from_tool_message(message)[0]["url"] == "https://example.com/source"


def test_chat_rejects_empty_request():
    app.dependency_overrides[get_current_user] = _mock_user
    try:
        response = TestClient(app).post("/api/chat/message", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_chat_generates_session_id():
    app.dependency_overrides[get_current_user] = _mock_user
    try:
        with patch(
            "app.api.chat.run_agent",
            new=AsyncMock(return_value={"response": "你好", "tool_calls": [], "analysis_result": None}),
        ):
            response = TestClient(app).post(
                "/api/chat/message",
                json={"message": "你好"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["session_id"]


def test_mock_yolo_endpoint_passes_detections_without_auth():
    agent_result = {
        "response": "识别到苹果，按 200g 估算约 104 kcal。",
        "tool_calls": [{"name": "calculate_total_nutrition", "args": {}}],
        "analysis_result": None,
    }
    with patch("app.api.chat.run_agent", new=AsyncMock(return_value=agent_result)) as mocked:
        response = TestClient(app).post(
            "/api/chat/mock-yolo",
            json={
                "session_id": "mock-meal-001",
                "message": "分析热量",
                "detections": [{
                    "class_name": "apple",
                    "class_name_cn": "苹果",
                    "confidence": 0.95,
                    "bbox": [10, 20, 200, 300],
                }],
            },
        )

    assert response.status_code == 200
    assert mocked.await_args.kwargs["session_id"] == "mock-yolo:mock-meal-001"
    assert mocked.await_args.kwargs["detections"][0]["class_name"] == "apple"


class _OfflineDatabaseSession:
    def query(self, *_args, **_kwargs):
        raise SQLAlchemyError("database offline")

    def rollback(self):
        pass

    def close(self):
        pass


def test_builtin_nutrition_fallback_without_database():
    with patch("app.services.agent_tools.SessionLocal", return_value=_OfflineDatabaseSession()):
        apple = asyncio.run(query_food_calories("apple"))
        total = asyncio.run(
            calculate_total_nutrition(
                '[{"food_name":"apple","estimated_weight_g":200},'
                '{"food_name":"rice","estimated_weight_g":150}]'
            )
        )

    assert "52.0 kcal" in apple
    assert "总热量: 278 kcal" in total


def test_mock_image_endpoint_passes_image_id_to_agent(tmp_path):
    original_uploads_dir = settings.UPLOADS_DIR
    settings.UPLOADS_DIR = tmp_path
    agent_result = {
        "response": "识别到苹果，估算热量约 104 kcal。",
        "tool_calls": [
            {"name": "detect_food", "args": {"image_id": "generated"}},
            {"name": "calculate_total_nutrition", "args": {}},
        ],
        "detections": [{
            "class_name": "apple",
            "class_name_cn": "苹果",
            "confidence": 0.95,
            "bbox": [10, 20, 200, 300],
        }],
        "detection_mode": "mock",
        "analysis_result": None,
    }
    try:
        with patch("app.api.chat.run_agent", new=AsyncMock(return_value=agent_result)) as mocked:
            response = TestClient(app).post(
                "/api/chat/image/mock",
                data={
                    "message": "分析热量",
                    "session_id": "image-meal-001",
                    "mock_detections": '[{"class_name":"apple","class_name_cn":"苹果",'
                    '"confidence":0.95,"bbox":[10,20,200,300]}]',
                },
                files={"file": ("meal.jpg", b"fake-jpeg-content", "image/jpeg")},
            )
    finally:
        settings.UPLOADS_DIR = original_uploads_dir

    assert response.status_code == 200
    assert response.json()["detection_mode"] == "mock"
    assert response.json()["detections"][0]["class_name"] == "apple"
    assert mocked.await_args.kwargs["image_id"]
    assert "detections" not in mocked.await_args.kwargs


def test_detect_food_tool_reads_mock_yolo_result(tmp_path):
    original_uploads_dir = settings.UPLOADS_DIR
    settings.UPLOADS_DIR = tmp_path
    try:
        image_id = asyncio.run(
            image_store.save(
                b"fake-image",
                "image/jpeg",
                mock_detections=[{
                    "class_name": "rice",
                    "class_name_cn": "米饭",
                    "confidence": 0.92,
                    "bbox": [1, 2, 3, 4],
                }],
            )
        )
        result = asyncio.run(detect_food(image_id))
    finally:
        settings.UPLOADS_DIR = original_uploads_dir

    assert '"mode": "mock"' in result
    assert '"class_name": "rice"' in result


def test_langgraph_runs_detection_then_nutrition_tools(tmp_path):
    class _FakeToolCallingModel:
        async def ainvoke(self, messages):
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            if not tool_messages:
                image_context = next(
                    message.content for message in messages if "image_id=" in str(message.content)
                )
                image_id = image_context.split("image_id=", 1)[1].split("。", 1)[0]
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "detect_food",
                        "args": {"image_id": image_id},
                        "id": "detect-1",
                        "type": "tool_call",
                    }],
                )
            if len(tool_messages) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "calculate_total_nutrition",
                        "args": {
                            "food_items_json": '[{"food_name":"rice","estimated_weight_g":150}]'
                        },
                        "id": "nutrition-1",
                        "type": "tool_call",
                    }],
                )
            return AIMessage(content="识别到米饭，估算约 174 kcal。")

    original_uploads_dir = settings.UPLOADS_DIR
    settings.UPLOADS_DIR = tmp_path
    agent_graph._agent_executor = None
    try:
        image_id = asyncio.run(
            image_store.save(
                b"fake-image",
                "image/jpeg",
                mock_detections=[{
                    "class_name": "rice",
                    "class_name_cn": "米饭",
                    "confidence": 0.92,
                    "bbox": [1, 2, 3, 4],
                }],
            )
        )
        with patch("app.services.agent_graph._get_llm", return_value=_FakeToolCallingModel()), patch(
            "app.services.agent_tools.SessionLocal", return_value=_OfflineDatabaseSession()
        ):
            result = asyncio.run(
                agent_graph.run_agent(
                    session_id="graph-image-test",
                    user_message="分析热量",
                    image_id=image_id,
                )
            )
    finally:
        settings.UPLOADS_DIR = original_uploads_dir
        agent_graph._agent_executor = None

    assert [call["name"] for call in result["tool_calls"]] == [
        "detect_food",
        "calculate_total_nutrition",
    ]
    assert result["detections"][0]["class_name"] == "rice"
    assert "174 kcal" in result["response"]


def test_persistent_chat_session_can_resume(db):
    user = User(
        id=701, username="resume-user", email="resume@example.com",
        hashed_password="hash", is_active=True,
    )
    db.add(user)
    db.commit()

    session = chat_service.create_session(db, user.id, "resume-001", "营养问答")
    chat_service.append_message(db, session, "user", "燕麦有什么营养？")
    chat_service.append_message(db, session, "assistant", "燕麦富含膳食纤维。")

    loaded = chat_service.get_session(db, user.id, "resume-001")
    history = chat_service.history_as_langchain(loaded)
    assert loaded.title == "营养问答"
    assert [message.content for message in history] == ["燕麦有什么营养？", "燕麦富含膳食纤维。"]


def test_agent_registers_knowledge_and_web_tools():
    tool_names = {tool.name for tool in agent_graph.AGENT_TOOLS}
    assert "search_nutrition_knowledge" in tool_names
    assert "search_web_evidence" in tool_names

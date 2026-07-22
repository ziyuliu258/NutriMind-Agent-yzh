"""
Agent 图编排模块 — 使用 LangGraph 将 LLM 与工具节点连接。

职责：
- 构建 StateGraph 定义 Agent 工作流（agent_node → 工具调用 → agent_node → END）
- 编译可执行图（带内存检查点，支持多轮对话）
- 提供 run_agent 入口函数，将 YOLOv11 检测结果注入 Agent 上下文

流程：
    用户消息 + YOLOv11 检测结果
        → agent_node (LLM 推理，决定是否调用工具)
        → router_node (条件路由: 工具调用 → tool_node, 否则 → END)
        → tool_node (执行工具后返回 agent_node)
        → agent_node (基于工具结果生成最终回复)
        → END
"""

import json
import logging
import uuid
from typing import Annotated, Any, AsyncIterator, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config.settings import settings
from app.services.agent_prompts import (
    SYSTEM_PROMPT,
    format_detection_for_prompt,
    format_user_profile_for_prompt,
)
from app.services.agent_tools import (
    agent_user_id,
    calculate_total_nutrition,
    detect_food,
    query_food_by_category,
    query_food_calories,
    save_detection_record,
    search_nutrition_knowledge,
    search_web_evidence,
    vision_verify_food,
)

logger = logging.getLogger(__name__)

# ==================================================================
# Agent 状态定义
# ==================================================================


class AgentState(TypedDict, total=False):
    """LangGraph Agent 的状态结构。

    messages: 对话历史（自动追加，使用 LangGraph 的 add_messages reducer）
    detections: YOLOv11 检测结果列表（可选，有检测结果时传入）
    user_id: 当前用户 ID（可选，用于个性化查询）
    analysis_result: 最终分析结果（Agent 完成后填充）
    """
    messages: Annotated[list[BaseMessage], add_messages]
    detections: Optional[list[dict]]
    user_id: Optional[int]
    analysis_result: Optional[str]
    image_id: Optional[str]
    user_profile: Optional[dict]


# ==================================================================
# 工具列表（注册给 LLM 调用）
# ==================================================================

# 注意：LangChain @tool 装饰器创建的是同步工具，但我们的实现是 async 的。
# 这里使用 Tool 的 StructuredTool 包装来确保兼容性。
from langchain_core.tools import StructuredTool

AGENT_TOOLS = [
    StructuredTool.from_function(
        coroutine=detect_food,
        name="detect_food",
        description="识别已上传餐食图片中的食物。必须传入系统提供的 image_id，可选置信度阈值。返回标准 YOLO detections JSON。",
    ),
    StructuredTool.from_function(
        coroutine=query_food_calories,
        name="query_food_calories",
        description="查询指定食物的营养数据（每100g）。输入为食物名称（中文或英文）。返回热量、蛋白质、脂肪、碳水化合物和膳食纤维。",
    ),
    StructuredTool.from_function(
        coroutine=query_food_by_category,
        name="query_food_by_category",
        description="查询指定分类下的所有食物及其营养数据。输入为食物分类名称（如 fruit, meat, vegetable, staple）。",
    ),
    StructuredTool.from_function(
        coroutine=calculate_total_nutrition,
        name="calculate_total_nutrition",
        description="根据食物清单和估算重量计算总营养摄入。输入为 JSON 字符串: [{'food_name': 'apple', 'estimated_weight_g': 200}]。",
    ),
    StructuredTool.from_function(
        coroutine=save_detection_record,
        name="save_detection_record",
        description="将检测结果持久化保存到数据库。输入为 user_id(int), scene_id(int), detections_json(str)。",
    ),
    StructuredTool.from_function(
        coroutine=vision_verify_food,
        name="vision_verify_food",
        description="视觉识别兜底。当 YOLO 检测失败、模型不存在、或检测结果低置信度时，调用多模态 LLM 直接观察图片识别食物。输入为 image_id。返回识别的食物列表 JSON。",
    ),
    StructuredTool.from_function(
        coroutine=search_nutrition_knowledge,
        name="search_nutrition_knowledge",
        description="检索当前用户的营养知识库；本地资料不足时自动联网。输入 query 和可选 verify_web，用户身份由系统安全注入。返回本地片段和网页来源 JSON。",
    ),
    StructuredTool.from_function(
        coroutine=search_web_evidence,
        name="search_web_evidence",
        description="使用 Exa 真正联网搜索公开网页，用于最新信息、用户明确要求联网或需要网址来源的问答。输入 query 和可选 limit，返回本次搜索得到的标题、URL、正文和发布时间 JSON。",
    ),
]

# ``query_food_by_category`` 是“浏览某个分类的全部食物”的工具，不应用来
# 猜测单个食物的分类。例如“薯条的热量”只需查询薯条本身，不应扩展为土豆、主食、
# 零食等多次查询。
CATEGORY_BROWSE_TOOL_NAME = "query_food_by_category"
DIRECT_AGENT_TOOLS = [
    tool for tool in AGENT_TOOLS if tool.name != CATEGORY_BROWSE_TOOL_NAME
]

# 业务级保护：最多允许模型发起三批工具调用。每一批仍可并行查询多项食物；
# 达到上限后强制基于已有工具结果生成答案，避免 ReAct 循环。
MAX_TOOL_ROUNDS = 3

# 框架级兜底。正常路径最多约 10 个图步骤，保留余量给未来扩展，不能再把它当作
# 控制工具循环的唯一手段。
RECURSION_LIMIT = 30

# ==================================================================
# LLM 实例
# ==================================================================

_llm: Optional[ChatOpenAI] = None


def _get_base_llm() -> ChatOpenAI:
    """获取未绑定工具的 LLM 实例。"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL,
            temperature=0.3,
            streaming=True,
        )
    return _llm


def _get_llm(tools: Optional[list[StructuredTool]] = None):
    """获取绑定指定工具集的 LLM 实例。"""
    return _get_base_llm().bind_tools(AGENT_TOOLS if tools is None else tools)


def _get_final_llm() -> ChatOpenAI:
    """获取不绑定工具的收尾 LLM，确保收尾节点不会继续触发工具调用。"""
    return _get_base_llm()


# ==================================================================
# 图节点定义
# ==================================================================


def _model_messages(state: AgentState, *, force_final_answer: bool = False) -> list[BaseMessage]:
    """构造发送给模型的消息，并避免历史系统消息重复叠加。"""
    system_content = SYSTEM_PROMPT
    profile_context = format_user_profile_for_prompt(state.get("user_profile"))
    if profile_context:
        system_content = f"{system_content}\n\n{profile_context}"
    if force_final_answer:
        system_content += (
            "\n\n## 本轮必须收尾\n"
            "已完成本轮允许的工具查询。请只依据现有工具结果直接给出最终回答；"
            "不要继续查询、不要猜测替代食物或分类，也不要提及工具轮次限制。"
            "若信息不足，请明确说明缺少什么，而不是继续调用工具。"
        )

    return [SystemMessage(content=system_content)] + [
        message for message in state["messages"] if not isinstance(message, SystemMessage)
    ]


def _latest_user_request(state: AgentState) -> str:
    """取得当前用户的原始问题，跳过系统注入的图片/检测上下文。"""
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, HumanMessage):
            continue
        content = str(message.content).strip()
        if content.startswith("[IMAGE_CONTEXT]") or content.startswith("YOLOv11 视觉模型"):
            continue
        return content.lower()
    return ""


def _category_browse_requested(state: AgentState) -> bool:
    """仅在用户明确要浏览一个分类的食物列表时开放分类工具。"""
    request = _latest_user_request(state)
    if not request:
        return False

    category_terms = (
        "分类", "类别", "水果", "蔬菜", "肉类", "主食", "零食", "饮料", "奶制品",
        "fruit", "meat", "vegetable", "staple", "snack", "beverage", "dairy",
    )
    strong_browse_terms = (
        "分类下", "类别下", "分类里", "类别里", "列出", "列表", "清单", "所有", "全部",
        "all foods", "food list", "list foods", "show foods",
    )
    enumeration_terms = (
        "哪些食物", "有哪些食物", "有什么食物", "哪些食品", "有哪些食品",
        "哪些", "有哪些", "有什么", "which foods", "what foods",
    )
    nutrition_terms = ("营养", "热量", "卡路里", "蛋白质", "脂肪", "碳水", "纤维", "多少")

    has_category = any(term in request for term in category_terms)
    if not has_category:
        return False
    if any(term in request for term in strong_browse_terms):
        return True
    # “这个零食有什么营养”不应被误判为“浏览零食分类”。
    if any(term in request for term in nutrition_terms):
        return False
    return any(term in request for term in enumeration_terms)


def _tools_for_state(state: AgentState) -> list[StructuredTool]:
    """普通单食物问题不暴露分类浏览工具，避免模型擅自扩展查询范围。"""
    return AGENT_TOOLS if _category_browse_requested(state) else DIRECT_AGENT_TOOLS


def _current_turn_tool_rounds(state: AgentState) -> int:
    """统计本轮用户消息之后已执行过的工具批次，忽略历史会话。"""
    messages = state.get("messages", [])
    last_human_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    return sum(
        1
        for message in messages[last_human_index + 1:]
        if isinstance(message, AIMessage) and message.tool_calls
    )


async def agent_node(state: AgentState) -> dict:
    """Agent 节点：调用 LLM 进行推理。

    将当前对话历史（messages）发给 LLM，LLM 可以选择：
    - 直接回复用户
    - 调用一个或多个工具

    Args:
        state: 当前 Agent 状态

    Returns:
        包含 LLM 响应（AIMessage）的字典
    """
    # 不让历史会话中的工具调用影响本轮；本轮达到上限后改用无工具模型收尾。
    if _current_turn_tool_rounds(state) >= MAX_TOOL_ROUNDS:
        return await final_answer_node(state)

    llm_with_tools = _get_llm(_tools_for_state(state))
    messages = _model_messages(state)
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


async def final_answer_node(state: AgentState) -> dict:
    """在工具轮次达到上限后，不绑定工具地生成最终回答。"""
    response = await _get_final_llm().ainvoke(
        _model_messages(state, force_final_answer=True)
    )
    return {"messages": [response]}


async def router_node(state: AgentState) -> str:
    """路由节点：判断 LLM 是否需要调用工具。

    Args:
        state: 当前 Agent 状态

    Returns:
        "direct_tools" / "category_tools" — 需要调用对应的工具节点
        END — 不需要调用工具，结束流程
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "category_tools" if _category_browse_requested(state) else "direct_tools"
    return END


# ==================================================================
# 图编译
# ==================================================================

# 内存检查点（保存对话历史，支持多轮对话）
_memory = MemorySaver()


def build_agent_graph() -> StateGraph:
    """构建 LangGraph Agent 工作流图。

    图结构：
        START → agent_node → router_node:
                   ├── 对应工具节点 → agent_node（最多三轮后在 agent 内强制收尾）
                   └── END

    Returns:
        未编译的 StateGraph
    """
    workflow = StateGraph(AgentState)

    # 两个工具节点与模型的工具集保持一致：没有明确分类浏览意图时，分类工具
    # 即使被模型意外请求也不会执行。ToolNode 必须作为图中的原生节点运行，
    # 才能获得 LangGraph 注入的工具运行时和流式回调。
    direct_tools_node = ToolNode(DIRECT_AGENT_TOOLS)
    category_tools_node = ToolNode(AGENT_TOOLS)

    # 注册节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("direct_tools", direct_tools_node)
    workflow.add_node("category_tools", category_tools_node)

    # 边
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        router_node,
        {
            "direct_tools": "direct_tools",
            "category_tools": "category_tools",
            END: END,
        },
    )
    workflow.add_edge("direct_tools", "agent")
    workflow.add_edge("category_tools", "agent")

    return workflow


def get_agent_executor():
    """获取编译好的 Agent 执行器（带内存检查点）。

    Returns:
        已编译的 CompiledGraph，可直接调用 ainvoke
    """
    graph = build_agent_graph()
    return graph.compile(checkpointer=_memory)


# 模块级单例
_agent_executor = None


def _get_executor():
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = get_agent_executor()
    return _agent_executor


def _web_sources_from_tool_message(message: ToolMessage) -> list[dict]:
    """从联网相关工具的真实返回值中提取可展示来源。"""
    if getattr(message, "name", None) not in {
        "search_web_evidence", "search_nutrition_knowledge",
    }:
        return []
    try:
        payload = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return []
    results = payload.get("results") or payload.get("web_results") or []
    sources = []
    for item in results:
        url = item.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        sources.append({
            "title": item.get("title") or url,
            "url": url,
            "provider": item.get("provider") or payload.get("provider") or "exa",
        })
    return sources


def _collect_tool_artifacts(name: str | None, content: Any) -> dict:
    """从单个工具输出中提取图谱可用信息：联网来源、detect_food 检测结果。

    流式与非流式共用。content 可能是 ToolMessage.content 字符串。
    """
    artifacts: dict = {"web_sources": [], "detections": [], "detection_mode": None}
    text = content if isinstance(content, str) else getattr(content, "content", "")
    if name == "detect_food":
        try:
            tool_result = json.loads(text)
            if tool_result.get("success"):
                artifacts["detections"] = tool_result.get("detections", [])
                artifacts["detection_mode"] = tool_result.get("mode")
        except (json.JSONDecodeError, TypeError):
            logger.warning("无法解析 detect_food 工具结果")
    elif name in {"search_web_evidence", "search_nutrition_knowledge"}:
        message = ToolMessage(content=text or "", tool_call_id="artifact", name=name)
        artifacts["web_sources"] = _web_sources_from_tool_message(message)
    return artifacts


def _append_web_sources(response: str, sources: list[dict]) -> str:
    """确定性地把实际联网结果附在回答末尾，避免模型漏写或编造网址。"""
    unique = []
    seen_urls = set()
    for source in sources:
        if source["url"] in seen_urls:
            continue
        seen_urls.add(source["url"])
        unique.append(source)
    if not unique:
        return response

    lines = ["### 联网来源"]
    for source in unique[:8]:
        title = str(source["title"]).replace("[", "").replace("]", "").strip()
        lines.append(f"- [{title}]({source['url']})")
    source_block = "\n".join(lines)
    if all(source["url"] in response for source in unique[:8]):
        return response
    return f"{response.rstrip()}\n\n{source_block}"


# ==================================================================
# 公开入口
# ==================================================================


def _prepare_run(
    *,
    session_id: str,
    user_message: str,
    detections: Optional[list[dict]] = None,
    image_id: Optional[str] = None,
    user_id: Optional[int] = None,
    history: Optional[list[BaseMessage]] = None,
    user_profile: Optional[dict] = None,
) -> Optional[tuple[AgentState, dict]]:
    """构造 LangGraph 初始状态与运行配置（run_agent 与 stream_agent 共用）。

    返回 (initial_state, config)；没有任何可用输入时返回 None。
    """
    messages = list(history or [])

    # 图片只以安全 image_id 进入状态；由 Agent 自主决定并调用 detect_food。
    if image_id:
        messages.append(
            HumanMessage(
                content=(
                    "[IMAGE_CONTEXT]\n"
                    f"用户已上传一张餐食图片，image_id={image_id}。\n"
                    "你必须先调用 detect_food 工具观察图片检测结果，不能猜测图片内容；"
                    "得到食物列表后，再按用户问题决定是否查询和计算营养。"
                )
            )
        )

    # 兼容已有系统直接提供检测结果的场景
    if detections:
        detection_text = format_detection_for_prompt(detections)
        context_msg = (
            "YOLOv11 视觉模型检测到以下食物：\n\n"
            f"{detection_text}\n\n"
            "请查询这些食物的营养数据，计算总营养，并给出饮食建议。"
        )
        messages.append(HumanMessage(content=context_msg))

    # 用户消息
    if user_message:
        messages.append(HumanMessage(content=user_message))

    if not messages:
        return None

    # 数据库历史存在时，每次调用使用独立图线程，避免 MemorySaver 与持久化历史重复叠加。
    thread_id = f"{session_id}:{uuid.uuid4()}" if history is not None else session_id
    config = {
        "configurable": {"thread_id": thread_id},
        # 框架级兜底；业务级工具轮数限制由图中的 MAX_TOOL_ROUNDS 负责。
        "recursion_limit": RECURSION_LIMIT,
    }

    initial_state: AgentState = {
        "messages": messages,
        "detections": detections,
        "image_id": image_id,
        "user_id": user_id,
        "user_profile": user_profile,
        "analysis_result": None,
    }
    return initial_state, config


async def run_agent(
    session_id: str,
    user_message: str,
    detections: Optional[list[dict]] = None,
    image_id: Optional[str] = None,
    user_id: Optional[int] = None,
    history: Optional[list[BaseMessage]] = None,
    user_profile: Optional[dict] = None,
) -> dict:
    """运行 Agent 管道的主入口函数。

    将用户的自然语言消息和 YOLOv11 检测结果一起传给 Agent，
    由 LLM 驱动工具调用完成营养分析，最后返回结构化结果。

    典型用法：
        result = await run_agent(
            session_id="session-001",
            user_message="我刚刚吃了这些食物，请帮我分析一下",
            detections=[{"class_name": "apple", "class_name_cn": "苹果", "confidence": 0.95, "bbox": [10,20,200,300]}],
            user_id=1,
        )

    Args:
        session_id: 会话标识（用于 LangGraph 内存检查点，支持多轮对话）
        user_message: 用户的自然语言输入
        detections: YOLOv11 检测结果列表（可选）
        user_id: 用户 ID（可选）

    Returns:
        {
            "response": "Agent 的最终文本回复",
            "tool_calls": ["..."],        # 工具调用日志（可选）
            "analysis_result": "...",     # 结构化分析结果（可选）
        }
    """
    executor = _get_executor()

    prepared = _prepare_run(
        session_id=session_id, user_message=user_message, detections=detections,
        image_id=image_id, user_id=user_id, history=history, user_profile=user_profile,
    )
    if prepared is None:
        return {"response": "未提供任何消息或检测结果。", "tool_calls": [], "analysis_result": None}
    initial_state, config = prepared

    try:
        # 执行图
        user_token = agent_user_id.set(user_id or 0)
        try:
            final_state = await executor.ainvoke(initial_state, config)
        finally:
            agent_user_id.reset(user_token)

        # 提取最后一条 AI 消息作为回复
        all_messages = final_state.get("messages", [])
        response_text = ""
        tool_calls_log = []
        web_sources = []
        detected_foods = []
        detection_mode = None

        for msg in all_messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    response_text = msg.content
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_log.append({
                            "name": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                        })
            elif isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "detect_food":
                try:
                    tool_result = json.loads(msg.content)
                    if tool_result.get("success"):
                        detected_foods = tool_result.get("detections", [])
                        detection_mode = tool_result.get("mode")
                except (json.JSONDecodeError, TypeError):
                    logger.warning("无法解析 detect_food 工具结果")
            elif isinstance(msg, ToolMessage):
                web_sources.extend(_web_sources_from_tool_message(msg))

        response_text = _append_web_sources(response_text, web_sources)
        if web_sources and not any(
            call["name"] in {"search_web_evidence", "exa_web_search"}
            for call in tool_calls_log
        ):
            tool_calls_log.append({
                "name": "exa_web_search",
                "args": {"source_count": len({source["url"] for source in web_sources})},
            })
        analysis = final_state.get("analysis_result")

        return {
            "response": response_text,
            "tool_calls": tool_calls_log,
            "analysis_result": analysis,
            "detections": detected_foods,
            "detection_mode": detection_mode,
        }

    except Exception as exc:
        logger.exception("Agent 执行失败")
        return {
            "response": "抱歉，营养分析暂时不可用，请稍后重试。",
            "tool_calls": [],
            "analysis_result": None,
            "detections": [],
            "detection_mode": None,
        }


async def stream_agent(
    session_id: str,
    user_message: str,
    detections: Optional[list[dict]] = None,
    image_id: Optional[str] = None,
    user_id: Optional[int] = None,
    history: Optional[list[BaseMessage]] = None,
    user_profile: Optional[dict] = None,
) -> AsyncIterator[dict]:
    """流式运行 Agent，逐步产出事件供 SSE 转发。

    产出的事件类型：
        {"type": "token", "text": "..."}         模型生成的文本片段（逐字）
        {"type": "tool", "name": "..."}           开始调用某个工具
        {"type": "reset"}                          清空前端已累积文本（工具调用前的内容非最终答案）
        {"type": "done", "response", "tool_calls", "detections", "detection_mode"}
        {"type": "error", "message": "..."}
    """
    executor = _get_executor()
    prepared = _prepare_run(
        session_id=session_id, user_message=user_message, detections=detections,
        image_id=image_id, user_id=user_id, history=history, user_profile=user_profile,
    )
    if prepared is None:
        yield {"type": "done", "response": "未提供任何消息或检测结果。",
               "tool_calls": [], "detections": [], "detection_mode": None}
        return
    initial_state, config = prepared

    final_text = ""            # 最后一段（最近一次工具调用之后）的模型文本 = 最终答案
    tool_calls_log: list[dict] = []
    web_sources: list[dict] = []
    detected_foods: list[dict] = []
    detection_mode: Optional[str] = None
    streamed_any = False

    user_token = agent_user_id.set(user_id or 0)
    try:
        async for event in executor.astream_events(initial_state, config, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                text = getattr(chunk, "content", "") if chunk is not None else ""
                if text:
                    final_text += text
                    streamed_any = True
                    yield {"type": "token", "text": text}
            elif kind == "on_tool_start":
                name = event.get("name", "unknown")
                tool_calls_log.append({"name": name, "args": event.get("data", {}).get("input", {})})
                # 工具调用前若已输出文本，那不是最终答案 —— 让前端清空并重新累积
                if final_text:
                    final_text = ""
                    yield {"type": "reset"}
                yield {"type": "tool", "name": name}
            elif kind == "on_tool_end":
                name = event.get("name")
                output = event.get("data", {}).get("output")
                artifacts = _collect_tool_artifacts(name, output)
                web_sources.extend(artifacts["web_sources"])
                if artifacts["detections"]:
                    detected_foods = artifacts["detections"]
                    detection_mode = artifacts["detection_mode"]

        response_text = _append_web_sources(final_text, web_sources)
        if web_sources and not any(
            call["name"] in {"search_web_evidence", "exa_web_search"}
            for call in tool_calls_log
        ):
            tool_calls_log.append({
                "name": "exa_web_search",
                "args": {"source_count": len({source["url"] for source in web_sources})},
            })
        if not response_text:
            response_text = "智能体已完成处理，但没有返回可显示的文字。"
        yield {
            "type": "done",
            "response": response_text,
            "tool_calls": tool_calls_log,
            "detections": detected_foods,
            "detection_mode": detection_mode,
        }
    except Exception:
        logger.exception("Agent 流式执行失败")
        yield {"type": "error", "message": "抱歉，营养分析暂时不可用，请稍后重试。"}
    finally:
        agent_user_id.reset(user_token)

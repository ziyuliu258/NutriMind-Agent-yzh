"""
Agent 系统提示词模块 — 定义 LLM 的角色、行为规范和输出格式。

职责：
- 定义营养师角色的系统提示词（System Prompt）
- 提供卡路里分析、饮食建议、食物查询等场景的提示词模板
- 将 YOLOv11 检测结果格式化为 LLM 可读的文本
"""

# ==================================================================
# 系统提示词
# ==================================================================

SYSTEM_PROMPT = """你是一位专业的 AI 营养师，名叫 NutriMind。你是一个会自主使用工具的 ReAct 智能体，而不是只靠常识回答的聊天机器人。你的任务是：

1. **接收食物检测结果**：系统会通过 YOLOv11 视觉模型识别用户餐盘中的食材清单，
   以 JSON 格式提供给你，包含每种食物的名称、置信度和数量信息。

2. **计算营养成分**：根据食材清单，调用工具函数查询每种食物的营养数据
   （热量、蛋白质、脂肪、碳水化合物、膳食纤维等），并汇总计算总卡路里。

3. **给出饮食建议**：结合用户的健康目标（减重/增肌/维持体重/健康饮食），
   提供个性化的饮食分析和改进建议。

## 工作流程
当用户上传图片并要求营养分析时，严格按以下步骤处理：
1. 调用 detect_food(image_id) 获取 YOLO 检测结果
2. 检查 detect_food 是否成功：
   a. 成功 → 提取检测到的食物列表
   b. 失败（success=false）→ 调用 vision_verify_food(image_id) 进行视觉兜底识别
3. 检测结果中有 low_confidence=True 的食物 → 调用 vision_verify_food(image_id) 二次确认
4. 提取所有食物名称，对低置信度结果提醒用户确认
5. 调用 query_food_calories 查询每项食物的营养数据
6. 根据检测类别的典型份量调用 calculate_total_nutrition 计算整体摄入
7. 明确说明重量和热量是视觉估算值，再提供分析与建议

如果用户只要求识别图片，可在食物识别后直接解释结果；如果用户询问营养或热量，必须继续调用营养工具。普通文字问答不需要强制调用视觉工具。

## 工具使用边界
- 用户询问单个食物（例如“薯条的热量”）时，只查询该食物的营养数据；不要因为原料、翻译或猜测而改查土豆、主食、零食等替代项。
- `query_food_by_category` 仅用于用户明确要求“列出/浏览某个分类下有哪些食物”的问题，不能用于给单个食物判定分类，也不要用它补充单食物营养数据。
- 同一名称已经查到结果后，直接据此回答；只有明确返回未找到时，才可尝试一次同义名称。
- 已取得满足问题所需的数据后，必须输出最终回答，不要为了补充背景继续调用工具。

当用户询问营养知识、指南、研究结论或数据库中没有的食物时：
1. 优先调用 search_nutrition_knowledge 检索用户知识库
2. 本地结果为空或相关性不足时，必须使用其联网 fallback
3. 对健康风险、摄入标准等关键结论，调用 search_web_evidence 交叉验证
4. 最终回答写成清晰的一段话，并在末尾列出实际使用的来源标题和 URL；不得伪造来源

当用户明确要求“联网搜索”“查最新信息”“给出网址/来源”，或问题依赖实时、近期信息时：
1. 必须调用 search_web_evidence，不得只凭模型记忆回答
2. 只引用工具本次实际返回的网页，不得编造或猜测 URL
3. 在回答末尾使用 Markdown 链接列出来源标题和完整网址

## 输出格式
- 先用一个简短的表格列出各项食物的营养数据
- 然后给出总热量和三大宏量营养素（蛋白质/脂肪/碳水）的汇总
- 最后提供 2-3 条实用的饮食建议

## 安全准则（非常重要！）
- ❌ 不要给出任何医学诊断或治疗建议
- ❌ 不要推荐极端饮食方案（如极低卡路里饮食）
- ✅ 对于有特殊健康状况的用户，始终建议咨询专业医生或临床营养师
- ✅ 使用 "建议每日摄入"、"推荐参考" 等非绝对化表述
- ✅ 检测到的食物置信度 < 0.25（low_confidence=True）时，先调用 vision_verify_food(image_id) 进行视觉二次验证，再告知用户

## 语气风格
- 友好、鼓励、专业
- 用通俗易懂的语言解释营养概念
- 尊重不同饮食文化和个人偏好
"""


def format_user_profile_for_prompt(profile: dict | None) -> str:
    """把已保存的身体资料和目标转换为不含账户隐私的 Agent 上下文。"""
    if not profile:
        return ""

    body = profile.get("body_profile") or {}
    goal = profile.get("goal") or {}
    mode_labels = {"cut": "减脂", "muscle": "增肌", "maintain": "维持体重"}
    sex_labels = {"male": "男", "female": "女", "unspecified": "未指定"}
    activity_labels = {
        "sedentary": "久坐", "light": "轻度活动", "moderate": "中等活动",
        "high": "高活动量", "very_high": "极高活动量",
    }
    fields = [
        ("主要目标", mode_labels.get(goal.get("mode"), goal.get("mode"))),
        ("当前体重", _with_unit(body.get("current_weight_kg"), "kg")),
        ("目标体重", _with_unit(goal.get("target_weight_kg"), "kg")),
        ("身高", _with_unit(body.get("height_cm"), "cm")),
        ("出生日期", _date_value(body.get("birth_date"))),
        ("计算性别", sex_labels.get(body.get("sex_for_calculation"), body.get("sex_for_calculation"))),
        ("日常活动水平", activity_labels.get(body.get("activity_level"), body.get("activity_level"))),
        ("每日热量目标", _with_unit(goal.get("daily_calories_kcal"), "kcal")),
        ("每日蛋白质目标", _with_unit(goal.get("protein_target_g"), "g")),
        ("每周训练频率", _with_unit(goal.get("training_days_per_week"), "天/周")),
    ]
    lines = [f"- {label}: {value}" for label, value in fields if value not in (None, "")]
    if not lines:
        return ""
    return (
        "[USER_PROFILE]\n"
        "以下是当前登录用户主动保存的身体与目标资料。回答时应结合这些资料进行个性化建议；"
        "不要把目标值误当作今日实际摄入或训练记录，也不要擅自补全缺失字段。\n"
        + "\n".join(lines)
    )


def _with_unit(value, unit: str) -> str | None:
    return None if value is None else f"{value} {unit}"


def _date_value(value) -> str | None:
    if value is None:
        return None
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]

# ==================================================================
# 场景提示词模板
# ==================================================================

CALORIE_ANALYSIS_PROMPT = """请分析以下通过 YOLOv11 视觉检测识别的食物清单，计算各食材的营养成分：

检测结果（JSON）：
{detections_json}

用户背景信息：
{user_context}

请：
1. 列出每项食物的估算重量和热量
2. 计算这顿饭的总热量
3. 判断这顿饭的营养均衡程度
"""

DIET_ADVICE_PROMPT = """基于以下营养分析结果，为用户提供个性化饮食建议：

营养摘要：
{calorie_summary}

用户信息：
{user_profile}

用餐场景：
{meal_context}

请根据用户的健康目标（如果未提供，默认为均衡饮食），给出 3 条具体的改进建议。
每条建议应包含：当前问题 → 具体建议 → 可执行的行动方案。
"""

FOOD_QUERY_PROMPT = """请查询以下食物的详细营养信息：
食物名称：{food_name}

返回食物每 100g 的热量、蛋白质、脂肪、碳水化合物和膳食纤维含量。
如果数据库中没有精确匹配，请查找最相似的食物。
"""

# ==================================================================
# 工具函数
# ==================================================================


def format_detection_for_prompt(detections: list[dict]) -> str:
    """将检测结果列表格式化为 LLM 友好的文本描述。

    Args:
        detections: DetectionResponse.detections 的列表，每项包含：
            class_name, class_name_cn, confidence, bbox

    Returns:
        格式化后的文本，例如：
        "1. 苹果 (apple) — 置信度: 95.3%，估算重量: 200g"
    """
    if not detections:
        return "（未检测到任何食物）"

    lines = []
    for i, det in enumerate(detections, 1):
        name = det.get("class_name_cn") or det.get("class_name", "未知食物")
        name_en = det.get("class_name", "unknown")
        conf = det.get("confidence", 0) * 100
        low = det.get("low_confidence", conf < 25)

        # 根据类别估算典型重量（非常粗略，实际应由用户输入）
        typical_weights = {
            "apple": 200, "banana": 120, "orange": 180, "rice": 150,
            "bread": 50, "egg": 55, "chicken": 150, "beef": 150,
            "broccoli": 100, "carrot": 80, "tomato": 100, "potato": 150,
            "milk": 250, "water": 250, "coffee": 250, "tea": 250,
            "noodle": 200, "pasta": 200, "pizza": 200, "cake": 100,
        }
        estimated_weight = typical_weights.get(name_en.lower(), 100)

        warning = " ⚠️ 低置信度" if low else ""
        line = (
            f"{i}. {name} ({name_en}) — "
            f"置信度: {conf:.1f}%, "
            f"估算重量: {estimated_weight}g"
            f"{warning}"
        )
        lines.append(line)

    return "\n".join(lines)

"""DeepSeek 对话模型的统一入口。"""

from langchain_deepseek import ChatDeepSeek

from research_agent.config import get_settings


class MissingApiKeyError(RuntimeError):
    """没有配置 DEEPSEEK_API_KEY。"""


def get_chat_model(temperature: float = 0.3) -> ChatDeepSeek:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise MissingApiKeyError(
            "未配置 DEEPSEEK_API_KEY：请把 .env.example 复制为 .env 并填入 Key。"
        )
    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        temperature=temperature,
        timeout=180,
        max_retries=2,
        # DeepSeek 默认开启思考模式，此时带工具的请求必须回传 reasoning_content，
        # 否则接口返回 400，且 temperature 不生效。这里显式关闭。
        extra_body={"thinking": {"type": "disabled"}},
    )

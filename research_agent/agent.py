"""主智能体：ReAct 工具调用智能体，对话记忆存入 SQLite。"""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_tool_call
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.llm import get_chat_model
from research_agent.projects import thread_belongs_to_project
from research_agent.prompts import SYSTEM_PROMPT_TEMPLATE
from research_agent.tools.evidence import build_evidence_matrix
from research_agent.tools.idea import generate_ideas
from research_agent.tools.literature import research_literature
from research_agent.tools.notes import save_note
from research_agent.tools.paper import ask_paper, compare_papers, list_papers, read_paper
from research_agent.tools.search import search_arxiv, search_web

TOOLS = [
    research_literature,
    search_arxiv,
    search_web,
    read_paper,
    ask_paper,
    compare_papers,
    build_evidence_matrix,
    list_papers,
    generate_ideas,
    save_note,
]
# 每轮对话（用户的一次提问）的调用上限。真实运行中，模型曾为一份“简短综述”连续精读 36 篇论文
# 仍不停手；每次精读都要把全文发给模型，不设上限就会无限消耗用户的时间和 API 费用。
MAX_PAPER_READS_PER_TURN = 5
MAX_LITERATURE_REVIEWS_PER_TURN = 1
MAX_IDEA_REPORTS_PER_TURN = 1
MAX_EVIDENCE_MATRICES_PER_TURN = 1
MAX_TOOL_CALLS_PER_TURN = 20
CONVERSATION_TITLE_CHARS = 30  # 会话列表里标题最多显示的字符数


@wrap_tool_call
def report_tool_errors(request, handler):
    """工具抛出异常时，把错误作为工具结果交还给模型。

    否则对话记录里会留下一条没有结果的工具调用，这个会话之后的每次请求都会被接口拒绝。
    """
    try:
        return handler(request)
    except Exception as exc:
        call = request.tool_call
        return ToolMessage(
            content=f"工具 {call['name']} 执行失败（{type(exc).__name__}: {exc}）。",
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )


def open_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Streamlit 的每次交互可能在不同线程里执行，连接必须允许跨线程使用；SqliteSaver 内部有锁。
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_agent(checkpointer: BaseCheckpointSaver, model: BaseChatModel | None = None):
    return create_agent(
        model=model or get_chat_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(
            max_paper_reads=MAX_PAPER_READS_PER_TURN,
            max_literature_reviews=MAX_LITERATURE_REVIEWS_PER_TURN,
            max_idea_reports=MAX_IDEA_REPORTS_PER_TURN,
            max_evidence_matrices=MAX_EVIDENCE_MATRICES_PER_TURN,
        ),
        middleware=[
            report_tool_errors,
            # 超出上限的调用会被拒绝并告知模型，模型仍可用已有信息把回答写完。
            ToolCallLimitMiddleware(tool_name=read_paper.name, run_limit=MAX_PAPER_READS_PER_TURN),
            ToolCallLimitMiddleware(
                tool_name=research_literature.name, run_limit=MAX_LITERATURE_REVIEWS_PER_TURN
            ),
            ToolCallLimitMiddleware(
                tool_name=generate_ideas.name, run_limit=MAX_IDEA_REPORTS_PER_TURN
            ),
            ToolCallLimitMiddleware(
                tool_name=build_evidence_matrix.name,
                run_limit=MAX_EVIDENCE_MATRICES_PER_TURN,
            ),
            ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS_PER_TURN),
        ],
        checkpointer=checkpointer,
    )


@dataclass(frozen=True)
class Conversation:
    thread_id: str
    title: str
    updated_at: datetime  # 最后一次活动的时间（本地时区）


def _conversation_title(first_question: str) -> str:
    title = " ".join(first_question.split())
    if len(title) > CONVERSATION_TITLE_CHARS:
        title = title[:CONVERSATION_TITLE_CHARS] + "…"
    return title or "未命名会话"


def list_conversations(agent, project_id: str | None = None) -> list[Conversation]:
    """列出保存在 SQLite 里的会话，最近活动的排在前面，标题取会话的第一个问题。

    会话清单直接从检查点表里查，不另外维护一份，避免两处记录不一致。
    """
    with agent.checkpointer.cursor(transaction=False) as cursor:
        # 检查点编号按时间递增，每个会话最大的编号就是它最后一次活动。
        cursor.execute(
            "SELECT thread_id FROM checkpoints WHERE checkpoint_ns = '' "
            "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
        )
        thread_ids = [row[0] for row in cursor.fetchall()]
    if project_id is not None:
        thread_ids = [
            thread_id
            for thread_id in thread_ids
            if thread_belongs_to_project(thread_id, project_id)
        ]
    conversations = []
    for thread_id in thread_ids:
        state = agent.get_state({"configurable": {"thread_id": thread_id}})
        first_question = next(
            (m.text for m in state.values.get("messages", []) if isinstance(m, HumanMessage)), ""
        )
        conversations.append(
            Conversation(
                thread_id=thread_id,
                title=_conversation_title(first_question),
                updated_at=datetime.fromisoformat(state.created_at).astimezone(),
            )
        )
    return conversations


def delete_conversation(agent, thread_id: str) -> None:
    """永久删除一个会话的全部对话记录。"""
    agent.checkpointer.delete_thread(thread_id)


@dataclass(frozen=True)
class AnswerText:
    """主智能体回答的一段增量文字。"""

    text: str


@dataclass(frozen=True)
class ToolStarted:
    call_id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolFinished:
    call_id: str
    result: str
    failed: bool


def stream_turn(agent, config: dict, prompt: str) -> Iterator[AnswerText | ToolStarted | ToolFinished]:
    """执行一轮对话，把执行过程转成界面可以直接渲染的事件。"""
    stream = agent.stream(
        {"messages": [HumanMessage(prompt)]}, config, stream_mode=["updates", "messages"]
    )
    for mode, payload in stream:
        if mode == "messages":
            chunk, metadata = payload
            # 精读、问答、idea 生成等工具内部也会调用模型，那些输出来自 tools 节点；
            # 只有 model 节点的文字才是主智能体对用户说的话。
            if metadata.get("langgraph_node") == "model" and isinstance(chunk, AIMessage) and chunk.text:
                yield AnswerText(chunk.text)
            continue
        for update in payload.values():
            for message in (update or {}).get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        yield ToolStarted(call["id"], call["name"], call["args"])
                elif isinstance(message, ToolMessage):
                    yield ToolFinished(message.tool_call_id, message.text, message.status == "error")


@dataclass
class ToolStep:
    name: str
    args: dict
    result: str | None = None
    failed: bool = False


@dataclass
class Turn:
    role: str  # "user" 或 "assistant"
    content: str
    steps: list[ToolStep] = field(default_factory=list)


def history_to_turns(messages: list[AnyMessage]) -> list[Turn]:
    """把对话记录整理成界面要展示的轮次：每轮助手回复带上它期间调用的工具及结果。"""
    turns: list[Turn] = []
    steps: list[ToolStep] = []
    steps_by_call_id: dict[str, ToolStep] = {}
    for message in messages:
        if isinstance(message, HumanMessage):
            if steps:  # 上一轮在给出最终回答前中断了
                turns.append(Turn("assistant", "", steps))
                steps = []
            turns.append(Turn("user", message.text))
        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                step = ToolStep(name=call["name"], args=call["args"])
                steps_by_call_id[call["id"]] = step
                steps.append(step)
            if not message.tool_calls:
                turns.append(Turn("assistant", message.text, steps))
                steps = []
        elif isinstance(message, ToolMessage):
            step = steps_by_call_id.get(message.tool_call_id)
            if step is not None:
                step.result = message.text
                step.failed = message.status == "error"
    if steps:
        turns.append(Turn("assistant", "", steps))
    return turns

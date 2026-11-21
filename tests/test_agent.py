from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from research_agent import agent as agent_module
from research_agent.agent import (
    AnswerText,
    ToolFinished,
    ToolStarted,
    build_agent,
    delete_conversation,
    history_to_turns,
    list_conversations,
    open_checkpointer,
    stream_turn,
)
from research_agent.tools import literature as literature_tool
from research_agent.tools import notes, paper as paper_tools


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture
def checkpointer(tmp_path):
    saver = open_checkpointer(tmp_path / "data" / "checkpoints.sqlite")
    yield saver
    saver.conn.close()


def test_agent_exposes_all_planned_tools():
    assert [tool.name for tool in agent_module.TOOLS] == [
        "research_literature",
        "search_arxiv",
        "search_web",
        "read_paper",
        "ask_paper",
        "compare_papers",
        "build_evidence_matrix",
        "list_papers",
        "generate_ideas",
        "save_note",
    ]


def test_agent_calls_a_tool_then_answers_with_its_result(
    scripted_model, checkpointer, tmp_path, monkeypatch
):
    monkeypatch.setattr(notes, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path / "out"))
    model = scripted_model(
        responses=[
            _tool_call("save_note", {"title": "综述", "content": "正文"}, "call-1"),
            "已保存笔记。",
        ]
    )
    agent = build_agent(checkpointer, model=model)

    result = agent.invoke({"messages": [HumanMessage("帮我存一份综述")]}, _config("t1"))

    messages = result["messages"]
    assert [type(message) for message in messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    saved = list((tmp_path / "out").glob("*.md"))
    assert len(saved) == 1
    assert messages[2].content == f"已保存到 outputs/{saved[0].name}"
    assert messages[3].content == "已保存笔记。"
    # 模型第一次调用时能看到系统提示词，第二次调用时能看到工具结果
    assert isinstance(model.calls[0][0], SystemMessage)
    assert "科研助理智能体" in model.calls[0][0].content
    assert model.calls[1][-1].content == messages[2].content


def test_same_thread_remembers_earlier_turns_and_other_threads_do_not(scripted_model, checkpointer):
    model = scripted_model(responses=["你好，小王。", "你叫小王。", "我不知道你的名字。"])
    agent = build_agent(checkpointer, model=model)

    agent.invoke({"messages": [HumanMessage("我叫小王")]}, _config("t1"))
    agent.invoke({"messages": [HumanMessage("我叫什么？")]}, _config("t1"))
    agent.invoke({"messages": [HumanMessage("我叫什么？")]}, _config("t2"))

    second_call = [message.content for message in model.calls[1] if not isinstance(message, SystemMessage)]
    third_call = [message.content for message in model.calls[2] if not isinstance(message, SystemMessage)]
    assert second_call == ["我叫小王", "你好，小王。", "我叫什么？"]
    assert third_call == ["我叫什么？"]


def test_conversation_survives_reopening_the_checkpoint_file(scripted_model, tmp_path):
    db_path = tmp_path / "checkpoints.sqlite"
    first = open_checkpointer(db_path)
    build_agent(first, model=scripted_model(responses=["记住了。"])).invoke(
        {"messages": [HumanMessage("课题是 RAG")]}, _config("t1")
    )
    first.conn.close()

    second = open_checkpointer(db_path)
    state = build_agent(second, model=scripted_model(responses=[])).get_state(_config("t1"))
    second.conn.close()

    assert [message.content for message in state.values["messages"]] == ["课题是 RAG", "记住了。"]


def test_conversations_are_listed_most_recently_active_first_titled_by_their_first_question(
    scripted_model, checkpointer
):
    agent = build_agent(checkpointer, model=scripted_model(responses=["一", "二", "三"]))
    assert list_conversations(agent) == []

    agent.invoke({"messages": [HumanMessage("调研 RAG 的最新进展")]}, _config("t1"))
    agent.invoke({"messages": [HumanMessage("精读 1706.03762")]}, _config("t2"))
    agent.invoke({"messages": [HumanMessage("再说说评测")]}, _config("t1"))  # t1 变成最近活动的会话

    conversations = list_conversations(agent)

    assert [(item.thread_id, item.title) for item in conversations] == [
        ("t1", "调研 RAG 的最新进展"),
        ("t2", "精读 1706.03762"),
    ]
    newest, older = (item.updated_at for item in conversations)
    assert older <= newest
    assert abs(datetime.now().astimezone() - newest) < timedelta(minutes=1)


def test_conversation_title_is_a_single_short_line(scripted_model, checkpointer):
    agent = build_agent(checkpointer, model=scripted_model(responses=["好的。"]))
    question = "请帮我调研：\n  大语言模型智能体的长期记忆机制，包括存储结构、检索方式、遗忘策略和评测基准"
    agent.invoke({"messages": [HumanMessage(question)]}, _config("t1"))

    title = list_conversations(agent)[0].title

    assert title == "请帮我调研： 大语言模型智能体的长期记忆机制，包括存储结构、…"
    assert len(title) == agent_module.CONVERSATION_TITLE_CHARS + 1


def test_deleting_a_conversation_removes_its_history_and_leaves_the_others(scripted_model, checkpointer):
    agent = build_agent(checkpointer, model=scripted_model(responses=["一", "二"]))
    agent.invoke({"messages": [HumanMessage("要删掉的")]}, _config("gone"))
    agent.invoke({"messages": [HumanMessage("要留下的")]}, _config("kept"))

    delete_conversation(agent, "gone")

    assert [item.thread_id for item in list_conversations(agent)] == ["kept"]
    assert agent.get_state(_config("gone")).values == {}
    assert [m.content for m in agent.get_state(_config("kept")).values["messages"]] == ["要留下的", "二"]


def test_tool_exception_becomes_an_error_result_and_the_thread_stays_usable(
    scripted_model, checkpointer, monkeypatch
):
    def broken_store():
        raise RuntimeError("vector store offline")

    monkeypatch.setattr(paper_tools, "get_store", broken_store)
    model = scripted_model(
        responses=[_tool_call("list_papers", {}, "call-1"), "论文库暂时不可用。", "好的。"]
    )
    agent = build_agent(checkpointer, model=model)

    result = agent.invoke({"messages": [HumanMessage("库里有哪些论文？")]}, _config("t1"))
    tool_message = result["messages"][2]

    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call-1"
    assert "工具 list_papers 执行失败（RuntimeError: vector store offline）" in tool_message.content
    assert result["messages"][3].content == "论文库暂时不可用。"

    followup = agent.invoke({"messages": [HumanMessage("那先这样")]}, _config("t1"))
    assert followup["messages"][-1].content == "好的。"


def test_paper_reads_are_capped_per_turn_and_the_budget_resets_next_turn(
    scripted_model, checkpointer, monkeypatch
):
    """精读要把全文发给模型，模型一口气读几十篇会无上限地消耗用户的 API 额度。"""
    read_sources = []

    def fake_report(source, **kwargs):
        read_sources.append(source)
        return f"笔记：{source}"

    monkeypatch.setattr(paper_tools, "read_paper_report", fake_report)
    monkeypatch.setattr(paper_tools, "get_chat_model", lambda: None)
    monkeypatch.setattr(paper_tools, "get_store", lambda: None)
    limit = agent_module.MAX_PAPER_READS_PER_TURN
    attempts = limit + 2
    model = scripted_model(
        responses=[
            *[_tool_call("read_paper", {"source": f"paper-{i}"}, f"call-{i}") for i in range(attempts)],
            "综述写好了。",
            _tool_call("read_paper", {"source": "next-turn"}, "call-next"),
            "读完了。",
        ]
    )
    agent = build_agent(checkpointer, model=model)

    result = agent.invoke({"messages": [HumanMessage("写一份综述")]}, _config("t1"))

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert read_sources == [f"paper-{i}" for i in range(limit)]
    assert [message.status for message in tool_messages] == ["success"] * limit + ["error"] * 2
    assert "read_paper" in tool_messages[-1].content
    assert result["messages"][-1].content == "综述写好了。"

    agent.invoke({"messages": [HumanMessage("再读一篇")]}, _config("t1"))

    assert read_sources[-1] == "next-turn"


def test_total_tool_calls_are_capped_per_turn(scripted_model, checkpointer, monkeypatch):
    listed = []
    monkeypatch.setattr(paper_tools, "get_store", lambda: None)
    monkeypatch.setattr(paper_tools, "format_paper_list", lambda store: listed.append(1) or "论文库是空的。")
    limit = agent_module.MAX_TOOL_CALLS_PER_TURN
    model = scripted_model(
        responses=[*[_tool_call("list_papers", {}, f"call-{i}") for i in range(limit + 1)], "到此为止。"]
    )
    agent = build_agent(checkpointer, model=model)

    result = agent.invoke({"messages": [HumanMessage("一直查")]}, _config("t1"))

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(listed) == limit
    assert [message.status for message in tool_messages] == ["success"] * limit + ["error"]
    assert result["messages"][-1].content == "到此为止。"


def test_system_prompt_tells_the_model_the_reading_budget(scripted_model, checkpointer):
    model = scripted_model(responses=["你好。"])

    build_agent(checkpointer, model=model).invoke({"messages": [HumanMessage("你好")]}, _config("t1"))

    system_prompt = model.calls[0][0].content
    assert f"每轮对话最多执行 {agent_module.MAX_PAPER_READS_PER_TURN} 次 read_paper" in system_prompt
    assert (
        f"最多启动 {agent_module.MAX_LITERATURE_REVIEWS_PER_TURN} 次 research_literature"
        in system_prompt
    )
    assert f"{agent_module.MAX_IDEA_REPORTS_PER_TURN} 次 generate_ideas" in system_prompt
    assert "{" not in system_prompt  # 占位符都已填上


def test_literature_reviews_are_capped_per_turn(
    scripted_model, checkpointer, monkeypatch, tmp_path
):
    calls = []

    class FakeGraph:
        def invoke(self, state):
            calls.append(state)
            return {"report": "# 开发综述\n\n正文", "papers": [object()]}

    monkeypatch.setattr(literature_tool, "get_chat_model", lambda temperature: None)
    monkeypatch.setattr(literature_tool, "build_literature_graph", lambda llm, search, **kwargs: FakeGraph())
    monkeypatch.setattr(
        literature_tool, "write_note", lambda title, content, output_dir: tmp_path / "note.md"
    )
    monkeypatch.setattr(
        literature_tool, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path)
    )
    model = scripted_model(
        responses=[
            _tool_call("research_literature", {"topic": "RAG"}, "call-1"),
            _tool_call("research_literature", {"topic": "RAG 评测"}, "call-2"),
            "已完成。",
        ]
    )
    agent = build_agent(checkpointer, model=model)

    result = agent.invoke({"messages": [HumanMessage("分别调研两个方向")]}, _config("t1"))

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert calls == [{"topic": "RAG"}]
    assert [message.status for message in tool_messages] == ["success", "error"]
    assert "research_literature" in tool_messages[-1].content
    assert result["messages"][-1].content == "已完成。"


def test_stream_turn_reports_tool_steps_and_streams_only_the_agents_own_answer(
    scripted_model, checkpointer, store, monkeypatch
):
    store.add_paper(
        [
            Document(
                page_content="The learning rate is 0.0003.",
                metadata={"paper_id": "p1", "title": "Paper One", "page": 4, "chunk": 0, "total_pages": 9},
            )
        ]
    )
    inner_model = scripted_model(responses=["工具内部模型的回答（第4页）"])
    monkeypatch.setattr(paper_tools, "get_chat_model", lambda: inner_model)
    monkeypatch.setattr(paper_tools, "get_store", lambda: store)
    final_answer = "论文里写的学习率是 0.0003，出自第 4 页。"
    model = scripted_model(
        responses=[_tool_call("ask_paper", {"question": "learning rate"}, "call-1"), final_answer]
    )
    agent = build_agent(checkpointer, model=model)

    events = list(stream_turn(agent, _config("t1"), "学习率是多少？"))

    assert events[0] == ToolStarted("call-1", "ask_paper", {"question": "learning rate"})
    assert isinstance(events[1], ToolFinished)
    assert (events[1].call_id, events[1].failed) == ("call-1", False)
    assert events[1].result.startswith("工具内部模型的回答（第4页）")
    assert "《Paper One》（p1）第 4 页" in events[1].result
    answer_events = events[2:]
    assert all(isinstance(event, AnswerText) for event in answer_events)
    assert len(answer_events) > 1  # 回答是分块流式到达的
    assert "".join(event.text for event in answer_events) == final_answer
    assert len(inner_model.calls) == 1


def test_stream_turn_marks_a_failed_tool_step(scripted_model, checkpointer, monkeypatch):
    def broken_store():
        raise RuntimeError("vector store offline")

    monkeypatch.setattr(paper_tools, "get_store", broken_store)
    model = scripted_model(responses=[_tool_call("list_papers", {}, "call-1"), "暂时不可用。"])
    agent = build_agent(checkpointer, model=model)

    events = list(stream_turn(agent, _config("t1"), "库里有什么？"))

    assert events[0] == ToolStarted("call-1", "list_papers", {})
    assert isinstance(events[1], ToolFinished) and events[1].failed
    assert "vector store offline" in events[1].result
    assert "".join(event.text for event in events[2:]) == "暂时不可用。"


def test_history_to_turns_groups_tool_steps_under_the_assistant_reply():
    messages = [
        HumanMessage("调研 RAG"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_arxiv", "args": {"query": "rag"}, "id": "c1"},
                {"name": "search_web", "args": {"query": "rag blog"}, "id": "c2"},
            ],
        ),
        ToolMessage("[1] Paper", tool_call_id="c1"),
        ToolMessage("网页搜索失败", tool_call_id="c2", status="error"),
        _tool_call("save_note", {"title": "t", "content": "c"}, "c3"),
        ToolMessage("已保存到 outputs/x.md", tool_call_id="c3"),
        AIMessage("综述如下……"),
        HumanMessage("谢谢"),
        AIMessage("不客气。"),
    ]

    turns = history_to_turns(messages)

    assert [(turn.role, turn.content) for turn in turns] == [
        ("user", "调研 RAG"),
        ("assistant", "综述如下……"),
        ("user", "谢谢"),
        ("assistant", "不客气。"),
    ]
    assert [(s.name, s.args, s.result, s.failed) for s in turns[1].steps] == [
        ("search_arxiv", {"query": "rag"}, "[1] Paper", False),
        ("search_web", {"query": "rag blog"}, "网页搜索失败", True),
        ("save_note", {"title": "t", "content": "c"}, "已保存到 outputs/x.md", False),
    ]
    assert turns[3].steps == []


def test_history_to_turns_keeps_steps_of_an_interrupted_turn():
    messages = [
        HumanMessage("读论文"),
        _tool_call("read_paper", {"source": "1706.03762"}, "c1"),
        HumanMessage("还在吗"),
        _tool_call("list_papers", {}, "c2"),
    ]

    turns = history_to_turns(messages)

    assert [(turn.role, turn.content, [s.name for s in turn.steps]) for turn in turns] == [
        ("user", "读论文", []),
        ("assistant", "", ["read_paper"]),
        ("user", "还在吗", []),
        ("assistant", "", ["list_papers"]),
    ]
    assert turns[1].steps[0].result is None

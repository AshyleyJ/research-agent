"""用 Streamlit 的无头测试框架驱动 app.py，模型换成脚本化的假模型。"""

import json
from types import SimpleNamespace

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from research_agent import agent as agent_module
from research_agent import config as config_module
from research_agent.config import PROJECT_ROOT, Settings
from research_agent.paper import store as store_module
from research_agent.paper.store import PaperStore
from research_agent.tools import notes

APP_PATH = str(PROJECT_ROOT / "app.py")
CONVERSATIONS_HEADING = "历史会话（{} 个）"
PAPERS_HEADING = "论文库（{} 篇）"
NOTES_HEADING = "已保存的笔记（{} 份）"


@pytest.fixture
def app_env(tmp_path, monkeypatch, embeddings, scripted_model):
    """把 app.py 用到的配置、论文库和模型都指向临时目录与假模型。"""

    def configure(api_key: str = "test-key", responses: list | None = None):
        settings = Settings(
            _env_file=None,
            deepseek_api_key=api_key,
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "outputs",
        )
        settings.ensure_dirs()
        env.settings = settings
        env.model = scripted_model(responses=responses or [])
        monkeypatch.setattr(config_module, "get_settings", lambda: settings)
        monkeypatch.setattr(notes, "get_settings", lambda: settings)
        monkeypatch.setattr(agent_module, "get_chat_model", lambda: env.model)
        return env

    def new_page(thread: str = "thread-1") -> AppTest:
        page = AppTest.from_file(APP_PATH, default_timeout=60)
        page.query_params["thread"] = thread
        return page.run()

    opened = []
    real_open = agent_module.open_checkpointer

    def tracking_open(path):
        saver = real_open(path)
        opened.append(saver)
        return saver

    paper_store = PaperStore(tmp_path / "chroma", embeddings)
    monkeypatch.setattr(store_module, "get_store", lambda: paper_store)
    monkeypatch.setattr(agent_module, "open_checkpointer", tracking_open)
    st.cache_resource.clear()
    env = SimpleNamespace(configure=configure, new_page=new_page, store=paper_store)
    yield env
    st.cache_resource.clear()
    for saver in opened:
        saver.conn.close()


def _save_note_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "save_note", "args": {"title": "RAG 综述", "content": "正文"}, "id": "call-1"}],
    )


def _subheaders(page: AppTest) -> list[str]:
    return [item.value for item in page.sidebar.subheader]


def test_without_api_key_the_page_explains_setup_and_offers_no_chat(app_env):
    app_env.configure(api_key="")

    page = app_env.new_page()

    assert not page.exception
    assert "DEEPSEEK_API_KEY" in page.warning[0].value
    assert len(page.chat_input) == 0
    assert PAPERS_HEADING.format(0) in _subheaders(page)


def test_new_conversation_offers_examples_and_clicking_one_runs_it(app_env):
    app_env.configure(responses=["这是综述。"])
    page = app_env.new_page()
    suggestions = page.main.button
    assert len(suggestions) == 3

    suggestions[0].click().run()

    assert not page.exception
    assert [message.name for message in page.chat_message] == ["user", "assistant"]
    assert page.chat_message[0].markdown[0].value.startswith("调研 LLM 智能体记忆机制")
    assert page.chat_message[1].markdown[0].value == "这是综述。"
    assert len(page.main.button) == 0  # 有对话后不再显示任务建议


def test_chat_turn_shows_tool_step_answer_and_the_saved_note(app_env):
    app_env.configure(responses=[_save_note_call(), "综述已保存。"])
    page = app_env.new_page()
    assert NOTES_HEADING.format(0) in _subheaders(page)

    page.chat_input[0].set_value("写一份 RAG 综述并保存").run()

    assert not page.exception
    user, assistant = page.chat_message
    assert user.markdown[0].value == "写一份 RAG 综述并保存"
    step = assistant.status[0]
    assert step.label == "保存笔记 · save_note"
    assert step.state == "complete"
    assert json.loads(step.json[0].value) == {"title": "RAG 综述", "content": "正文"}
    assert step.text[0].value.startswith("已保存到 outputs/")
    assert assistant.markdown[0].value == "综述已保存。"

    saved = list(app_env.settings.output_dir.glob("*.md"))
    assert len(saved) == 1
    assert NOTES_HEADING.format(1) in _subheaders(page)
    assert [button.label for button in page.sidebar.download_button] == [
        saved[0].name,
        "下载全部笔记",
    ]


def test_conversation_is_restored_for_the_same_thread_after_restart(app_env):
    app_env.configure(responses=[_save_note_call(), "综述已保存。"])
    app_env.new_page(thread="kept").chat_input[0].set_value("写一份综述").run()

    st.cache_resource.clear()  # 相当于重启程序：智能体重新从 SQLite 文件加载
    app_env.configure(responses=[])
    restored = app_env.new_page(thread="kept")
    other = app_env.new_page(thread="fresh")

    assert [message.name for message in restored.chat_message] == ["user", "assistant"]
    assert restored.chat_message[0].markdown[0].value == "写一份综述"
    assert restored.chat_message[1].status[0].label == "保存笔记 · save_note"
    assert restored.chat_message[1].markdown[0].value == "综述已保存。"
    assert len(other.chat_message) == 0


def _conversation_titles(page: AppTest) -> list[str]:
    return [button.label for button in page.sidebar.button if button.key and button.key.startswith("open-")]


def _chat_texts(page: AppTest) -> list[str]:
    return [message.markdown[0].value for message in page.chat_message]


def test_starting_a_new_conversation_keeps_the_old_one_reachable_from_the_sidebar(app_env):
    app_env.configure(responses=["第一个回答。", "第二个回答。"])
    page = app_env.new_page(thread="t1")
    assert CONVERSATIONS_HEADING.format(0) in _subheaders(page)
    page.chat_input[0].set_value("问题一").run()
    assert _conversation_titles(page) == ["问题一"]

    page.sidebar.button(key="new-conversation").click().run()

    assert not page.exception
    assert _chat_texts(page) == []  # 新会话是空的
    assert _conversation_titles(page) == ["问题一"]  # 旧会话还在列表里

    page.chat_input[0].set_value("问题二").run()
    assert _conversation_titles(page) == ["问题二", "问题一"]  # 最近活动的在前

    page.sidebar.button(key="open-t1").click().run()

    assert not page.exception
    assert _chat_texts(page) == ["问题一", "第一个回答。"]
    assert CONVERSATIONS_HEADING.format(2) in _subheaders(page)


def test_deleting_another_conversation_removes_it_and_keeps_the_current_one_open(app_env):
    app_env.configure(responses=["第一个回答。", "第二个回答。"])
    app_env.new_page(thread="t1").chat_input[0].set_value("问题一").run()
    page = app_env.new_page(thread="t2")
    page.chat_input[0].set_value("问题二").run()

    page.sidebar.button(key="delete-t1").click().run()
    assert _conversation_titles(page) == ["问题二", "问题一"]  # 只点垃圾桶还不会删，要再确认一次
    page.sidebar.button(key="confirm-delete-t1").click().run()

    assert not page.exception
    assert _conversation_titles(page) == ["问题二"]
    assert _chat_texts(page) == ["问题二", "第二个回答。"]
    assert len(app_env.new_page(thread="t1").chat_message) == 0  # 记录确实从数据库里删掉了


def test_cancelling_a_delete_keeps_the_conversation(app_env):
    app_env.configure(responses=["第一个回答。"])
    page = app_env.new_page(thread="t1")
    page.chat_input[0].set_value("问题一").run()

    page.sidebar.button(key="delete-t1").click().run()
    page.sidebar.button(key="cancel-delete-t1").click().run()

    assert not page.exception
    assert _conversation_titles(page) == ["问题一"]
    assert _chat_texts(page) == ["问题一", "第一个回答。"]
    assert [button.key for button in page.sidebar.button if "confirm" in (button.key or "")] == []


def test_deleting_the_open_conversation_switches_to_an_empty_new_one(app_env):
    app_env.configure(responses=["第一个回答。"])
    page = app_env.new_page(thread="t1")
    page.chat_input[0].set_value("问题一").run()

    page.sidebar.button(key="delete-t1").click().run()
    page.sidebar.button(key="confirm-delete-t1").click().run()

    assert not page.exception
    assert _conversation_titles(page) == []
    assert _chat_texts(page) == []
    assert len(page.main.button) == 3  # 回到带任务建议的空白会话
    assert len(app_env.new_page(thread="t1").chat_message) == 0


def test_failed_tool_step_is_marked_as_error(app_env, monkeypatch):
    def broken_store():
        raise RuntimeError("vector store offline")

    list_call = AIMessage(content="", tool_calls=[{"name": "list_papers", "args": {}, "id": "call-1"}])
    app_env.configure(responses=[list_call, "论文库暂时不可用。"])
    page = app_env.new_page()
    # 页面首次渲染之后再让论文库出故障：侧边栏已经画完，接下来只有工具会碰到它。
    monkeypatch.setattr("research_agent.tools.paper.get_store", broken_store)

    page.chat_input[0].set_value("库里有什么").run()

    assert not page.exception
    step = page.chat_message[1].status[0]
    assert step.label == "查看论文库 · list_papers"
    assert step.state == "error"
    assert "vector store offline" in step.text[0].value
    assert page.chat_message[1].markdown[0].value == "论文库暂时不可用。"


def test_model_failure_is_shown_as_an_error_and_the_page_keeps_working(app_env):
    app_env.configure(responses=[])  # 假模型没有预设回复，一调用就抛异常
    page = app_env.new_page()

    page.chat_input[0].set_value("你好").run()

    assert not page.exception
    assert "这一轮没有完成" in page.error[0].value

    app_env.model.responses = ["现在好了。"]
    app_env.model.calls.clear()
    page.chat_input[0].set_value("再试一次").run()

    assert not page.exception
    assert page.chat_message[-1].markdown[0].value == "现在好了。"


def test_uploading_a_pdf_adds_it_to_the_library(app_env, sample_pdf):
    app_env.configure()
    page = app_env.new_page()

    page.sidebar.file_uploader[0].upload("My Paper.pdf", sample_pdf.read_bytes(), "application/pdf")
    page.sidebar.form_submit_button[0].click().run()

    assert not page.exception
    assert "paper_id: My-Paper" in page.sidebar.success[0].value
    assert PAPERS_HEADING.format(1) in _subheaders(page)
    assert any("Citation Graph Retrieval for Science QA" in item.value for item in page.sidebar.markdown)
    assert app_env.store.has_paper("My-Paper")
    assert (app_env.settings.papers_dir / "My-Paper.pdf").is_file()


def test_uploading_a_broken_pdf_reports_the_problem_and_adds_nothing(app_env):
    app_env.configure()
    page = app_env.new_page()

    page.sidebar.file_uploader[0].upload("broken.pdf", b"this is not a pdf", "application/pdf")
    page.sidebar.form_submit_button[0].click().run()

    assert not page.exception
    assert "不是有效的 PDF 文件" in page.sidebar.error[0].value
    assert PAPERS_HEADING.format(0) in _subheaders(page)
    assert list(app_env.settings.papers_dir.glob("*.pdf")) == []


def test_creating_and_switching_project_starts_an_isolated_workspace(app_env):
    app_env.configure()
    page = app_env.new_page()

    page.sidebar.text_input(key="new-project-name").set_value("长文档 RAG 评测")
    page.sidebar.button(key="create-project").click().run()

    assert not page.exception
    project_id = page.query_params["project"]
    assert project_id != "default"
    assert page.query_params["thread"].startswith(f"{project_id}:")
    assert _subheaders(page).count(PAPERS_HEADING.format(0)) == 1
    assert _subheaders(page).count(NOTES_HEADING.format(0)) == 1

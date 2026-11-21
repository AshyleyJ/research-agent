from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from research_agent.llm import MissingApiKeyError
from research_agent.tools import notes
from research_agent.workflows.idea_graph import SearchQueries
from scripts import run_demos


class CheckModel:
    """自检用的假模型：三种调用方式分别可控。"""

    def __init__(self, tool_calls: list, structured_error: Exception | None = None):
        self._tool_calls = tool_calls
        self._structured_error = structured_error

    def invoke(self, messages):
        return AIMessage("我是一个测试模型。")

    def bind_tools(self, tools):
        return SimpleNamespace(invoke=lambda messages: AIMessage("直接回答", tool_calls=self._tool_calls))

    def with_structured_output(self, schema):
        def invoke(messages):
            if self._structured_error:
                raise self._structured_error
            return SearchQueries(queries=["graph neural network survey"])

        return SimpleNamespace(invoke=invoke)


ARXIV_CALL = {"name": "search_arxiv", "args": {"query": "rag survey"}, "id": "c1"}


def test_connectivity_check_passes_when_all_three_call_styles_work(monkeypatch, capsys):
    monkeypatch.setattr(run_demos, "get_chat_model", lambda: CheckModel([ARXIV_CALL]))

    assert run_demos.check_connectivity() is True

    output = capsys.readouterr().out
    assert "[通过] 普通对话：我是一个测试模型。" in output
    assert '[通过] 工具调用：{"query": "rag survey"}' in output
    assert '[通过] 结构化输出：["graph neural network survey"]' in output


def test_connectivity_check_reports_each_failure_and_keeps_checking(monkeypatch, capsys):
    model = CheckModel(tool_calls=[], structured_error=ValueError("400 tool_choice unsupported"))
    monkeypatch.setattr(run_demos, "get_chat_model", lambda: model)

    assert run_demos.check_connectivity() is False

    output = capsys.readouterr().out
    assert "[通过] 普通对话" in output
    assert "[失败] 工具调用：RuntimeError: 模型没有发起工具调用" in output
    assert "[失败] 结构化输出：ValueError: 400 tool_choice unsupported" in output


def test_main_stops_with_a_hint_when_the_api_key_is_missing(monkeypatch, capsys):
    def no_key():
        raise MissingApiKeyError("未配置 DEEPSEEK_API_KEY")

    monkeypatch.setattr(run_demos, "get_chat_model", no_key)
    monkeypatch.setattr("sys.argv", ["run_demos"])

    assert run_demos.main() == 1
    assert "未配置 DEEPSEEK_API_KEY" in capsys.readouterr().out


def test_main_does_not_run_scenarios_when_the_check_fails_or_with_check_flag(monkeypatch):
    ran = []
    monkeypatch.setattr(run_demos, "run_scenario", lambda *args: ran.append(args))

    monkeypatch.setattr(run_demos, "check_connectivity", lambda: False)
    monkeypatch.setattr("sys.argv", ["run_demos"])
    assert run_demos.main() == 1

    monkeypatch.setattr(run_demos, "check_connectivity", lambda: True)
    monkeypatch.setattr("sys.argv", ["run_demos", "--check"])
    assert run_demos.main() == 0

    assert ran == []


def test_main_runs_all_scenarios_by_default_and_only_the_selected_one_on_request(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(run_demos, "check_connectivity", lambda: True)
    monkeypatch.setattr(run_demos, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path))
    monkeypatch.setattr(run_demos, "run_scenario", lambda index, *rest: ran.append(index))

    monkeypatch.setattr("sys.argv", ["run_demos"])
    assert run_demos.main() == 0
    monkeypatch.setattr("sys.argv", ["run_demos", "--scenario", "3"])
    assert run_demos.main() == 0

    assert ran == [1, 2, 3, 3]


def test_scenario_runs_every_prompt_in_one_conversation_and_saves_the_transcript(
    monkeypatch, scripted_model, tmp_path, capsys
):
    settings = SimpleNamespace(output_dir=tmp_path / "outputs", deepseek_model="test-model")
    monkeypatch.setattr(notes, "get_settings", lambda: settings)
    monkeypatch.setattr(run_demos, "get_settings", lambda: settings)
    model = scripted_model(
        responses=[
            AIMessage("", tool_calls=[{"name": "save_note", "args": {"title": "综述", "content": "正文"}, "id": "c1"}]),
            "综述已保存。",
            "第二个问题的回答。",
        ]
    )
    real_build_agent = run_demos.build_agent
    monkeypatch.setattr(run_demos, "build_agent", lambda saver: real_build_agent(saver, model=model))

    path = run_demos.run_scenario(2, "测试场景", ["写综述并保存", "再问一句"], tmp_path / "demo-out")

    assert path == tmp_path / "demo-out" / "scenario-2.md"
    transcript = path.read_text(encoding="utf-8")
    assert transcript.startswith("# 演示场景：测试场景\n\n- 模型：test-model\n")
    assert transcript.index("写综述并保存") < transcript.index("`save_note`") < transcript.index("综述已保存。")
    assert transcript.index("综述已保存。") < transcript.index("再问一句") < transcript.index("第二个问题的回答。")
    assert '参数：`{"title": "综述", "content": "正文"}`' in transcript
    assert "已保存到 outputs/" in transcript
    # 第二个问题在同一会话里：模型第三次被调用时能看到第一轮的内容
    assert "写综述并保存" in model.prompt_text(2)
    assert "综述已保存。" in capsys.readouterr().out


def test_transcript_marks_failed_steps_and_truncates_long_results():
    messages = [
        HumanMessage("读论文"),
        AIMessage("", tool_calls=[
            {"name": "read_paper", "args": {"source": "x"}, "id": "c1"},
            {"name": "search_web", "args": {"query": "y"}, "id": "c2"},
        ]),
        ToolMessage("A" * 5000, tool_call_id="c1"),
        ToolMessage("网络不通", tool_call_id="c2", status="error"),
        AIMessage("读完了。"),
    ]

    transcript = run_demos.render_transcript("标题", messages)

    assert "**工具调用（完成）：`read_paper`**" in transcript
    assert "**工具调用（失败）：`search_web`**" in transcript
    assert "A" * run_demos.RESULT_PREVIEW_CHARS + "\n…（共 5000 字符）" in transcript
    assert "A" * (run_demos.RESULT_PREVIEW_CHARS + 1) not in transcript

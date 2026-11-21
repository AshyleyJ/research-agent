from research_agent.tools import idea as idea_tool
from research_agent.tools.search import PaperHit
from research_agent.workflows.idea_graph import (
    MAX_REVISIONS,
    Critique,
    Idea,
    IdeaList,
    IdeaReview,
    SearchQueries,
    build_idea_graph,
)


class FakeStructuredLLM:
    """按 schema 分别排队返回结构化结果，并记录每次调用的 schema 和提示词。"""

    def __init__(self, outputs: dict):
        self.outputs = {schema: list(items) for schema, items in outputs.items()}
        self.calls: list[tuple[type, str]] = []

    def with_structured_output(self, schema, **kwargs):
        llm = self

        class Runner:
            def invoke(self, messages):
                llm.calls.append((schema, "\n".join(message.content for message in messages)))
                return llm.outputs[schema].pop(0)

        return Runner()

    def prompts_for(self, schema) -> list[str]:
        return [prompt for called_schema, prompt in self.calls if called_schema is schema]


def _hit(arxiv_id: str, title: str) -> PaperHit:
    return PaperHit(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Ada Lovelace", "Alan Turing", "Grace Hopper", "Edsger Dijkstra"],
        published="2024-05-01",
        summary=f"Abstract of {title}.",
        url=f"http://arxiv.org/abs/{arxiv_id}",
    )


def _idea(title: str, references: list[int] | None = None) -> Idea:
    return Idea(
        title=title,
        motivation=f"{title} 的动机",
        method=f"{title} 的方法",
        difference=f"{title} 的区别",
        feasibility=f"{title} 的可行性",
        references=references if references is not None else [1],
    )


def _ideas(*titles: str) -> IdeaList:
    return IdeaList(ideas=[_idea(title) for title in titles])


def _critique(*scores: tuple[int, int], comment: str = "评语") -> Critique:
    return Critique(
        reviews=[
            IdeaReview(idea_index=index, novelty=novelty, feasibility=feasibility, comment=comment)
            for index, (novelty, feasibility) in enumerate(scores, start=1)
        ]
    )


class FakeSearch:
    def __init__(self, results: dict[str, list[PaperHit]] | None = None, error: Exception | None = None):
        self.results = results or {}
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, max_results: int) -> list[PaperHit]:
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.results.get(query, [])


QUERIES = SearchQueries(queries=["llm agents planning", "tool use benchmark"])
SEARCH_RESULTS = {
    "llm agents planning": [_hit("2401.00001v1", "Planning Agents"), _hit("2401.00002v1", "ReAct Revisited")],
    "tool use benchmark": [_hit("2401.00002v1", "ReAct Revisited"), _hit("2401.00003v1", "ToolBench")],
}


def _run(llm: FakeStructuredLLM, search: FakeSearch, topic: str = "LLM 智能体的规划能力") -> str:
    return build_idea_graph(llm, search).invoke({"topic": topic})["report"]


def test_ideas_that_pass_review_are_reported_without_revision():
    llm = FakeStructuredLLM(
        {
            SearchQueries: [QUERIES],
            IdeaList: [_ideas("A", "B", "C")],
            Critique: [_critique((4, 3), (5, 5), (4, 4), comment="切入点清晰")],
        }
    )
    search = FakeSearch(SEARCH_RESULTS)

    report = _run(llm, search)

    assert [query for query, _ in search.calls] == ["llm agents planning", "tool use benchmark"]
    assert len(llm.prompts_for(IdeaList)) == 1
    assert len(llm.prompts_for(Critique)) == 1
    generate_prompt = llm.prompts_for(IdeaList)[0]
    assert "LLM 智能体的规划能力" in generate_prompt
    assert "[1] Planning Agents（2024）" in generate_prompt
    assert "[3] ToolBench（2024）" in generate_prompt
    assert generate_prompt.count("] ReAct Revisited（") == 1  # 多条检索式命中的同一篇论文只保留一次
    assert "[4]" not in generate_prompt

    assert report.startswith("# 研究 idea 报告：LLM 智能体的规划能力")
    assert "- 检索式：llm agents planning；tool use benchmark" in report
    assert "- 参考文献：3 篇" in report
    assert "- 修订轮数：0" in report
    assert "- 评审结论：全部 idea 达到评审阈值" in report
    assert "## Idea 1：A" in report and "## Idea 3：C" in report
    assert "**动机**：A 的动机" in report
    assert "**评审：新颖性 4/5，可行性 3/5（达标）。切入点清晰**" in report
    assert (
        "[1] Ada Lovelace, Alan Turing, Grace Hopper 等. Planning Agents. "
        "arXiv:2401.00001v1, 2024. http://arxiv.org/abs/2401.00001v1"
    ) in report


def test_failing_ideas_are_revised_with_the_reviewers_feedback():
    llm = FakeStructuredLLM(
        {
            SearchQueries: [QUERIES],
            IdeaList: [_ideas("旧A", "旧B", "旧C"), _ideas("新A", "新B", "新C")],
            Critique: [
                _critique((2, 4), (4, 4), (4, 4), comment="与已有工作重复"),
                _critique((4, 4), (4, 4), (4, 4)),
            ],
        }
    )

    report = _run(llm, FakeSearch(SEARCH_RESULTS))

    assert len(llm.prompts_for(IdeaList)) == 2
    assert len(llm.prompts_for(Critique)) == 2
    revise_prompt = llm.prompts_for(IdeaList)[1]
    assert "### Idea 1：旧A" in revise_prompt
    assert "新颖性 2/5，可行性 4/5（未达标）。与已有工作重复" in revise_prompt
    assert "新颖性 4/5，可行性 4/5（达标）" in revise_prompt
    assert "## Idea 1：新A" in report
    assert "旧A" not in report
    assert "- 修订轮数：1" in report
    assert "- 评审结论：全部 idea 达到评审阈值" in report


def test_revision_stops_at_the_limit_and_report_says_threshold_not_met():
    rounds = MAX_REVISIONS + 1
    llm = FakeStructuredLLM(
        {
            SearchQueries: [QUERIES],
            IdeaList: [_ideas(f"第{n}版A", f"第{n}版B", f"第{n}版C") for n in range(1, rounds + 1)],
            Critique: [_critique((4, 4), (4, 2), (4, 4)) for _ in range(rounds)],
        }
    )

    report = _run(llm, FakeSearch(SEARCH_RESULTS))

    assert len(llm.prompts_for(IdeaList)) == rounds
    assert len(llm.prompts_for(Critique)) == rounds
    assert f"## Idea 1：第{rounds}版A" in report
    assert f"- 修订轮数：{MAX_REVISIONS}" in report
    assert "已达修订上限" in report and "仍有 idea 未达评审阈值" in report
    assert "新颖性 4/5，可行性 2/5（未达标）" in report


def test_no_ideas_are_generated_when_literature_search_fails():
    llm = FakeStructuredLLM({SearchQueries: [QUERIES], IdeaList: [], Critique: []})
    search = FakeSearch(error=ConnectionError("arxiv unreachable"))

    report = _run(llm, search)

    assert len(search.calls) == 2
    assert llm.prompts_for(IdeaList) == []
    assert "没有生成 idea：未能检索到相关文献" in report
    assert "arxiv unreachable" in report
    assert "llm agents planning；tool use benchmark" in report
    assert "## Idea" not in report


def test_no_ideas_are_generated_when_search_returns_nothing():
    llm = FakeStructuredLLM({SearchQueries: [QUERIES], IdeaList: [], Critique: []})

    report = _run(llm, FakeSearch(results={}))

    assert "这些检索式没有返回任何论文" in report
    assert llm.prompts_for(IdeaList) == []


def test_report_drops_references_outside_the_retrieved_list_and_fails_unreviewed_ideas():
    always = MAX_REVISIONS + 1
    idea_lists = [
        IdeaList(ideas=[_idea("A", references=[3, 1, 7, 0, 3]), _idea("B", references=[99])])
        for _ in range(always)
    ]
    only_first_reviewed = Critique(
        reviews=[IdeaReview(idea_index=1, novelty=9, feasibility=5, comment="好")]
    )
    llm = FakeStructuredLLM(
        {
            SearchQueries: [QUERIES],
            IdeaList: idea_lists,
            Critique: [only_first_reviewed for _ in range(always)],
        }
    )

    report = _run(llm, FakeSearch(SEARCH_RESULTS))

    idea_a, idea_b = report.split("## Idea 2：B")
    idea_b = idea_b.split("## 参考文献")[0]
    assert "**依据文献**：[1] [3]" in idea_a
    assert "新颖性 5/5，可行性 5/5（达标）" in idea_a  # 越界分数被限制到 5
    assert "**依据文献**：无" in idea_b
    assert "评审者没有给出这个 idea 的评分，按未达标处理" in idea_b
    assert len(llm.prompts_for(IdeaList)) == always  # 漏评的 idea 不算达标，因此一直修订到上限


def test_generate_ideas_tool_runs_the_workflow_and_returns_the_report(monkeypatch):
    llm = FakeStructuredLLM(
        {
            SearchQueries: [QUERIES],
            IdeaList: [_ideas("A", "B", "C")],
            Critique: [_critique((4, 4), (4, 4), (4, 4))],
        }
    )
    temperatures = []

    def fake_get_chat_model(temperature):
        temperatures.append(temperature)
        return llm

    monkeypatch.setattr(idea_tool, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(idea_tool, "arxiv_search", FakeSearch(SEARCH_RESULTS))

    report = idea_tool.generate_ideas.invoke({"topic": "LLM 智能体"})

    assert temperatures == [idea_tool.IDEA_TEMPERATURE]
    assert report.startswith("# 研究 idea 报告：LLM 智能体")
    assert "## Idea 3：C" in report

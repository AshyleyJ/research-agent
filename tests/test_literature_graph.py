from types import SimpleNamespace
import pytest

from research_agent.tools import literature as literature_tool
from research_agent.tools.search import PaperHit
from research_agent.workflows.literature_graph import (
    LiteratureCritique,
    LiteratureQueries,
    LiteratureReview,
    ReviewSection,
    build_literature_graph,
)


class FakeStructuredLLM:
    def __init__(self, outputs):
        self.outputs = {schema: list(items) for schema, items in outputs.items()}
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        llm = self

        class Runner:
            def invoke(self, messages):
                llm.calls.append((schema, "\n".join(message.content for message in messages)))
                return llm.outputs[schema].pop(0)

        return Runner()

    def prompts_for(self, schema):
        return [prompt for called_schema, prompt in self.calls if called_schema is schema]


def _hit(arxiv_id, title):
    return PaperHit(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Ada Lovelace", "Alan Turing", "Grace Hopper", "Edsger Dijkstra"],
        published="2024-05-01",
        summary=f"Abstract of {title}.",
        url=f"http://arxiv.org/abs/{arxiv_id}",
    )


def _review(title, heading="方法演进", refs=None):
    return LiteratureReview(
        title=title,
        abstract="这是摘要。",
        sections=[
            ReviewSection(
                heading=heading,
                summary="这是带依据的论述 [1] [2] [9]。",
                references=refs or [1, 2, 9],
            )
        ],
        research_gaps=["缺少统一评测。"],
        takeaways=["需要继续比较。"],
        references=[2, 8],
    )


def _critique(
    coverage=4, evidence=4, clarity=4, needs_revision=False, issues=None
):
    return LiteratureCritique(
        coverage=coverage,
        evidence=evidence,
        clarity=clarity,
        needs_revision=needs_revision,
        issues=issues or [],
    )


class FakeSearch:
    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.calls = []

    def __call__(self, query, max_results):
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.results.get(query, [])


QUERIES = LiteratureQueries(queries=["retrieval augmented generation survey", "long document rag"])
SEARCH_RESULTS = {
    "retrieval augmented generation survey": [
        _hit("2401.00001v1", "RAG Survey"),
        _hit("2401.00002v1", "Long RAG"),
    ],
    "long document rag": [
        _hit("2401.00002v1", "Long RAG"),
        _hit("2401.00003v1", "Document Reranker"),
    ],
}


def test_literature_workflow_plans_deduplicates_writes_and_passes_review():
    llm = FakeStructuredLLM(
        {
            LiteratureQueries: [QUERIES],
            LiteratureReview: [_review("检索增强生成综述")],
            LiteratureCritique: [_critique()],
        }
    )
    search = FakeSearch(SEARCH_RESULTS)

    result = build_literature_graph(llm, search).invoke({"topic": "长文档 RAG"})
    report = result["report"]

    assert [query for query, _ in search.calls] == [
        "retrieval augmented generation survey",
        "long document rag",
    ]
    assert len(result["papers"]) == 3
    prompt = llm.prompts_for(LiteratureReview)[0]
    assert prompt.count("Long RAG（2024）") == 1
    assert "[3] Document Reranker（2024）" in prompt
    assert report.startswith("# 检索增强生成综述")
    assert "> 文献数：3 篇 | 修订：0 次 | 质量检查：评审通过" in report
    assert "依据：[1] [2]" in report
    assert "[9]" not in report
    assert "arXiv:2401.00001v1" in report and "arXiv:2401.00002v1" in report
    assert "arXiv:2401.00003v1" not in report  # 没有被正文引用，不塞进参考文献凑数
    assert llm.prompts_for(LiteratureCritique)[0].count("综述") >= 1


def test_failed_review_triggers_one_revision_and_then_rechecks():
    llm = FakeStructuredLLM(
        {
            LiteratureQueries: [QUERIES],
            LiteratureReview: [_review("初稿"), _review("修订稿", "核心挑战")],
            LiteratureCritique: [
                _critique(2, 3, 4, True, ["补齐评测方法", "补充近两年进展"]),
                _critique(4, 4, 4),
            ],
        }
    )

    result = build_literature_graph(llm, FakeSearch(SEARCH_RESULTS)).invoke(
        {"topic": "长文档 RAG"}
    )
    report = result["report"]

    assert len(llm.prompts_for(LiteratureReview)) == 2
    assert len(llm.prompts_for(LiteratureCritique)) == 2
    revise_prompt = llm.prompts_for(LiteratureReview)[1]
    assert "初稿" in revise_prompt
    assert "补齐评测方法" in revise_prompt
    assert report.startswith("# 修订稿")
    assert "修订：1 次 | 质量检查：评审通过" in report
    assert "## 核心挑战" in report
    assert "初稿" not in report


def test_literature_workflow_reports_search_failure_without_calling_writer():
    llm = FakeStructuredLLM(
        {
            LiteratureQueries: [QUERIES],
            LiteratureReview: [],
            LiteratureCritique: [],
        }
    )

    result = build_literature_graph(llm, FakeSearch(error=ConnectionError("arxiv down"))).invoke(
        {"topic": "长文档 RAG"}
    )

    assert result["papers"] == []
    assert "没有生成综述：未能检索到相关文献" in result["report"]
    assert "arxiv down" in result["report"]
    assert llm.prompts_for(LiteratureReview) == []
    assert llm.prompts_for(LiteratureCritique) == []


def test_research_literature_tool_automatically_saves_only_successful_report(
    tmp_path, monkeypatch
):
    llm = FakeStructuredLLM(
        {
            LiteratureQueries: [QUERIES],
            LiteratureReview: [_review("自动保存综述")],
            LiteratureCritique: [_critique()],
        }
    )
    search = FakeSearch(SEARCH_RESULTS)
    saved = []
    monkeypatch.setattr(literature_tool, "get_chat_model", lambda temperature: llm)
    monkeypatch.setattr(literature_tool, "arxiv_search", search)
    monkeypatch.setattr(
        literature_tool, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path)
    )

    def fake_write_note(title, content, output_dir):
        path = output_dir / "saved.md"
        path.write_text(content, encoding="utf-8")
        saved.append((title, content))
        return path

    monkeypatch.setattr(literature_tool, "write_note", fake_write_note)

    output = literature_tool.research_literature.invoke({"topic": "长文档 RAG"})

    assert output.startswith("# 自动保存综述")
    assert output.endswith("已自动保存到 outputs/saved.md")
    assert saved[0][0] == "文献综述-长文档 RAG"
    assert "# 自动保存综述" in saved[0][1]


def test_empty_structured_draft_is_retried_before_critique():
    llm = FakeStructuredLLM({
        LiteratureQueries: [QUERIES],
        LiteratureReview: [None, _review("恢复后的综述")],
        LiteratureCritique: [_critique()],
    })
    result = build_literature_graph(llm, FakeSearch(SEARCH_RESULTS)).invoke({"topic": "RAG"})
    assert result["report"].startswith("# 恢复后的综述")
    assert "必须调用 LiteratureReview" in llm.prompts_for(LiteratureReview)[1]
    assert len(llm.prompts_for(LiteratureCritique)) == 1


def test_empty_structured_output_has_bounded_retry_and_clear_error():
    llm = FakeStructuredLLM({LiteratureQueries: [None, None]})
    with pytest.raises(RuntimeError, match="已重试一次"):
        build_literature_graph(llm, FakeSearch()).invoke({"topic": "RAG"})
    assert len(llm.calls) == 2

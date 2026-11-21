import threading

from langchain_core.messages import AIMessage
import pytest

from research_agent.workflows import research_workers as workers
from research_agent.workflows.research_workers import (
    ResearchFinding, ResearchResult, SearchGate, WorkerResult,
    merge_results, run_worker,
)
from test_literature_graph import (
    FakeStructuredLLM, LiteratureQueries, LiteratureReview, LiteratureCritique,
    _hit, _review, _critique,
)
from research_agent.workflows import literature_graph
from research_agent.workflows.research_evidence import EvidenceCards


def call(query, ident="1"):
    return AIMessage(content="", tool_calls=[
        {"name": "search_arxiv", "args": {"query": query}, "id": ident}
    ])


class WorkerModel:
    def __init__(self, replies, summary=None):
        self.replies = iter(replies)
        self.summary = summary or ResearchResult()
        self.histories = []

    def bind_tools(self, tools):
        assert [t.name for t in tools] == ["search_arxiv"]
        return self

    def invoke(self, messages):
        self.histories.append(list(messages))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def with_structured_output(self, schema):
        assert schema is ResearchResult
        parent = self
        class Summary:
            def invoke(self, messages):
                return parent.summary
        return Summary()


def test_agent_observes_results_and_changes_query_then_filters_unknown_sources():
    seen = []
    def search(query, limit):
        seen.append(query)
        return [] if query == "broad" else [_hit("2401.00001v2", "Evidence")]
    model = WorkerModel([call("broad"), call("focused", "2"), AIMessage(content="done")],
        ResearchResult(findings=[
            ResearchFinding(statement="supported", source_ids=["2401.00001v2"]),
            ResearchFinding(statement="invented", source_ids=["9999.99999"]),
            ResearchFinding(statement="mixed", source_ids=["2401.00001", "fake"]),
        ]))
    result = run_worker(model, SearchGate(search, 0), "progress", "topic", [])
    assert seen == ["broad", "focused"]
    assert model.histories[1][-1].content == "[]"
    assert [f.statement for f in result.findings] == ["supported"]
    assert result.papers[0].title == "Evidence"
    assert result.status == "partial"


def test_parallel_tool_calls_cannot_bypass_three_search_budget():
    reply = AIMessage(content="", tool_calls=[
        {"name": "search_arxiv", "args": {"query": str(i)}, "id": str(i)} for i in range(8)
    ])
    seen = []
    result = run_worker(WorkerModel([reply]), SearchGate(lambda q, n: seen.append(q) or [], 0), "progress", "topic", [])
    assert len(seen) == 3
    assert result.exit_reason == "search_budget"


def test_failure_after_search_retains_tool_evidence():
    result = run_worker(WorkerModel([call("q"), RuntimeError("model down")]),
        SearchGate(lambda q, n: [_hit("1v1", "Preserved")], 0), "limitations", "topic", [])
    assert result.status == "partial"
    assert result.papers[0].title == "Preserved"
    assert result.exit_reason == "error"


def test_search_failure_is_observed_and_agent_can_recover():
    def search(q, n):
        if q == "bad":
            raise TimeoutError("offline")
        return [_hit("1", "Recovered")]
    model = WorkerModel([call("bad"), call("good", "2"), AIMessage(content="done")])
    result = run_worker(model, SearchGate(search, 0), "progress", "topic", [])
    assert "检索失败" in model.histories[1][-1].content
    assert result.papers and result.errors and result.status == "partial"


def test_step_limit_stops_invalid_tool_loop_and_expired_budget_stops_before_model():
    invalid = AIMessage(content="", tool_calls=[{"name": "save_note", "args": {}, "id": "x"}])
    model = WorkerModel([invalid] * 4)
    result = run_worker(model, SearchGate(lambda q, n: pytest.fail("unexpected search"), 0), "progress", "topic", [])
    assert result.exit_reason == "step_budget"
    assert len(model.histories) == 4
    result = run_worker(WorkerModel([]), SearchGate(None, 0), "progress", "topic", [], time_budget=0)
    assert result.exit_reason == "time_budget"


def test_merge_deduplicates_versions_balances_roles_and_remaps_sources():
    results = [
        WorkerResult("progress", papers=[_hit("1v1", "A"), _hit("2", "B")]),
        WorkerResult("limitations", papers=[_hit("1v2", "A2"), _hit("3", "C")], findings=[
            ResearchFinding(statement="retained [99]", source_ids=["1"]),
            ResearchFinding(statement="excluded", source_ids=["3"]),
        ]),
    ]
    merged = merge_results(results, max_papers=2)
    assert [p.title for p in merged["papers"]] == ["A", "B"]
    assert "retained  [1]" in merged["research_notes"]
    assert "excluded" not in merged["research_notes"]
    assert "[99]" not in merged["research_notes"]


def test_workers_execute_concurrently_with_independent_roles(monkeypatch):
    barrier = threading.Barrier(2)
    def worker(llm, gate, role, topic, seeds):
        barrier.wait(timeout=3)
        return WorkerResult(role)
    monkeypatch.setattr(workers, "run_worker", worker)
    assert [r.role for r in workers.investigate(None, None, "topic", [])] == ["progress", "limitations"]


@pytest.mark.parametrize("has_papers", [True, False])
def test_multi_agent_graph_handles_partial_and_total_failure(monkeypatch, has_papers):
    results = [WorkerResult("progress", papers=[_hit("1", "A")] if has_papers else [], status="partial"),
               WorkerResult("limitations", errors=["offline"], gaps=["验证证据缺失"])]
    monkeypatch.setattr(literature_graph, "investigate", lambda *a: results)
    model = FakeStructuredLLM({
        LiteratureQueries: [LiteratureQueries(queries=["seed"])],
        EvidenceCards: [EvidenceCards(cards=[])] if has_papers else [],
        LiteratureReview: [_review("Report")] if has_papers else [],
        LiteratureCritique: [_critique()] if has_papers else [],
    })
    result = literature_graph.build_literature_graph(model, None, mode="multi_agent").invoke({"topic": "topic"})
    assert "offline" in result["report"]
    assert "仅依据摘要" in result["report"]
    if has_papers:
        assert "验证证据缺失" in model.prompts_for(LiteratureReview)[0]
    else:
        assert not model.prompts_for(LiteratureReview)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        literature_graph.build_literature_graph(None, None, mode="typo")


def test_tool_passes_configured_mode_without_changing_public_input(monkeypatch):
    from types import SimpleNamespace
    from research_agent.tools import literature
    seen = []
    def build(llm, search, mode):
        seen.append(mode)
        return SimpleNamespace(invoke=lambda state: {"report": "no sources", "papers": []})
    monkeypatch.setattr(literature, "build_literature_graph", build)
    monkeypatch.setattr(literature, "get_settings", lambda: SimpleNamespace(literature_mode="multi_agent"))
    monkeypatch.setattr(literature, "get_chat_model", lambda **kwargs: None)
    assert literature.research_literature.invoke({"topic": "topic"}) == "no sources"
    assert seen == ["multi_agent"]


def test_shared_search_cache_normalizes_queries_and_does_not_cache_errors():
    calls = []
    def search(query, limit):
        calls.append(query)
        if query == "failure":
            raise TimeoutError("retryable")
        return [_hit("1", "Evidence")]
    gate = SearchGate(search, 0)
    import time
    deadline = time.monotonic() + 10
    assert gate.search_with_cache("RAG  evaluation", deadline)[1] is False
    assert gate.search_with_cache("rag evaluation", deadline)[1] is True
    assert len(calls) == 1
    for _ in range(2):
        with pytest.raises(TimeoutError):
            gate.search("failure", deadline)
    assert calls.count("failure") == 2


def test_role_specific_queries_are_not_shared_between_workers(monkeypatch):
    seen = {}
    def worker(llm, gate, role, topic, seeds):
        seen[role] = seeds
        return WorkerResult(role)
    monkeypatch.setattr(workers, "run_worker", worker)
    workers.investigate(None, None, "topic", {"progress": ["methods"], "limitations": ["failures"]})
    assert seen == {"progress": ["methods"], "limitations": ["failures"]}

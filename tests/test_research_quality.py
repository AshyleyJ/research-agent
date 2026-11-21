import pytest
from research_agent.workflows import literature_graph as graph
from research_agent.workflows.research_evidence import EvidenceCard, EvidenceCards, validate_cards
from research_agent.workflows.research_workers import WorkerResult
from test_literature_graph import FakeStructuredLLM, LiteratureQueries, LiteratureReview, LiteratureCritique, _hit, _review, _critique


def issue(ident="I1", **changes):
    values = dict(issue_id=ident, category="evidence", location="结论", quote="所有方法都失败", reason="缺少证据",
                  action="retrieve", severity="blocking", status="open", search_query="targeted failure analysis")
    return graph.ReviewIssue(**(values | changes))


def test_evidence_cards_reject_fabricated_quotes_sources_and_keep_unknowns():
    from dataclasses import replace
    paper = replace(_hit("1v2", "A"), summary="This exact abstract sentence supports only a limited result.")
    card = EvidenceCard(source_id="1", task_setting="QA", relevance="direct", claim="有限结果",
                        quote=paper.summary, evidence_type="author_result")
    cards, rejected = validate_cards([card], [paper])
    assert cards[0].quote == paper.summary and rejected == 0
    cards, rejected = validate_cards([
        card.model_copy(update={"quote": "This entirely fabricated quotation does not exist."}),
        card.model_copy(update={"source_id": "unknown"}),
    ], [paper])
    assert rejected == 2 and len(cards) == 1
    assert cards[0].relevance == "unknown" and cards[0].quote == ""


def test_advisory_does_not_block_but_omitted_blocker_survives_re_review():
    advisory = _critique(needs_revision=True).model_copy(update={"details": [issue(severity="advisory")]})
    assert graph.review_passes(graph.reconcile_critique(advisory))
    previous = _critique().model_copy(update={"details": [issue()]})
    current = graph.reconcile_critique(_critique(), previous)
    assert not graph.review_passes(current)
    assert current.details[0].status == "open"
    resolved = _critique().model_copy(update={"details": [issue(status="resolved")]})
    assert graph.review_passes(graph.reconcile_critique(resolved, previous))


def test_bad_scores_still_block_with_only_advisory():
    review = _critique(coverage=2).model_copy(update={"details": [issue(severity="advisory")]})
    assert not graph.review_passes(graph.reconcile_critique(review))


def test_model_schema_has_only_one_collection_for_review_issues():
    schema = LiteratureCritique.model_json_schema()
    assert "details" in schema["properties"]
    assert "issues" not in schema["properties"]
    # Legacy serialized critiques remain readable.
    assert _critique(issues=["旧格式意见"]).issues == ["旧格式意见"]


def test_structured_retry_receives_parse_error_and_unwraps_raw_result():
    from langchain_core.messages import AIMessage
    model = FakeStructuredLLM({LiteratureQueries: [
        {"raw": AIMessage(content=""), "parsed": None, "parsing_error": ValueError("bad field type")},
        {"raw": AIMessage(content=""), "parsed": LiteratureQueries(queries=["valid"]), "parsing_error": None},
    ]})
    result = graph._invoke_structured(model, LiteratureQueries, [])
    assert result.queries == ["valid"]
    assert "bad field type" in model.prompts_for(LiteratureQueries)[1]


@pytest.mark.parametrize("search_fails", [False, True])
def test_supplement_is_bounded_preserves_numbering_and_rechecks_issues(monkeypatch, search_fails):
    monkeypatch.setattr(graph, "investigate", lambda *a: [WorkerResult("progress", papers=[_hit("1", "Original")])])
    from research_agent.workflows.research_workers import SearchGate
    monkeypatch.setattr(graph, "SearchGate", lambda fn: SearchGate(fn, 0))
    issues = [issue("I0", search_query="initial"), issue(), issue("I2", search_query="second"), issue("I3", search_query="third")]
    reviews = [_critique(needs_revision=True).model_copy(update={"details": issues}),
               _critique(needs_revision=True).model_copy(update={"details": issues})]
    llm = FakeStructuredLLM({
        LiteratureQueries: [LiteratureQueries(queries=["initial"], progress_queries=["methods"], limitations_queries=["failures"])],
        EvidenceCards: [EvidenceCards(cards=[]), EvidenceCards(cards=[])],
        LiteratureReview: [_review("Initial"), _review("Revised")],
        LiteratureCritique: reviews,
    })
    # Initial queries in the merged state must reflect actual worker calls.
    monkeypatch.setattr(graph, "investigate", lambda *a: [WorkerResult("progress", papers=[_hit("1", "Original")], queries=["initial"])])
    calls = []
    def search(q, n):
        calls.append(q)
        if search_fails:
            raise TimeoutError("offline")
        return [_hit("1", "Original"), _hit(str(len(calls) + 1), q)]
    result = graph.build_literature_graph(llm, search, "multi_agent").invoke({"topic": "QA"})
    assert calls == ["targeted failure analysis", "second"]
    assert result["papers"][0].title == "Original"
    assert len(result["papers"]) == (1 if search_fails else 3)
    assert result["revisions"] == 1 and result["supplement_done"]
    assert len(llm.prompts_for(LiteratureCritique)) == 2
    assert "上一轮意见" in llm.prompts_for(LiteratureCritique)[1]
    assert "已达修订上限" in result["report"]
    if search_fails:
        assert "offline" in " ".join(result["search_errors"])


def test_rewrite_only_issue_does_not_trigger_retrieval(monkeypatch):
    monkeypatch.setattr(graph, "investigate", lambda *a: [WorkerResult("progress", papers=[_hit("1", "A")])])
    llm = FakeStructuredLLM({
        LiteratureQueries: [LiteratureQueries(queries=["seed"])],
        EvidenceCards: [EvidenceCards(cards=[])],
        LiteratureReview: [_review("Initial"), _review("Revised")],
        LiteratureCritique: [_critique().model_copy(update={"details": [issue(action="rewrite")]}),
                             _critique().model_copy(update={"details": [issue(action="rewrite", status="resolved")]})],
    })
    result = graph.build_literature_graph(llm, lambda *a: pytest.fail("unneeded search"), "multi_agent").invoke({"topic": "QA"})
    assert not result["supplement_done"]
    assert graph.review_passes(result["critique"])

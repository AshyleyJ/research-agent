from datetime import datetime
from types import SimpleNamespace

import pytest

from research_agent.tools import search


def _arxiv_result(short_id: str, title: str, n_authors: int = 2):
    return SimpleNamespace(
        get_short_id=lambda: short_id,
        title=title,
        authors=[SimpleNamespace(name=f"Author {i}") for i in range(1, n_authors + 1)],
        published=datetime(2024, 3, 5),
        summary="A summary\nspanning two lines.",
        entry_id=f"http://arxiv.org/abs/{short_id}",
    )


class FakeArxivClient:
    """results_for(query) 决定每条检索式返回什么；searches 记录收到的全部检索。"""

    results_for = staticmethod(lambda query: [])
    error: Exception | None = None
    searches: list = []

    def __init__(self, **kwargs):
        pass

    def results(self, search_obj):
        FakeArxivClient.searches.append(search_obj)
        if FakeArxivClient.error:
            raise FakeArxivClient.error
        return iter(FakeArxivClient.results_for(search_obj.query))


@pytest.fixture
def fake_arxiv(monkeypatch):
    FakeArxivClient.results_for = staticmethod(lambda query: [])
    FakeArxivClient.error = None
    FakeArxivClient.searches = []
    monkeypatch.setattr(search.arxiv, "Client", FakeArxivClient)
    return FakeArxivClient


def _settings(tavily_api_key: str = ""):
    return SimpleNamespace(tavily_api_key=tavily_api_key)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("graph rag", "(ti:graph OR abs:graph) AND (ti:rag OR abs:rag)"),
        (
            "A Survey of Retrieval-Augmented Generation",
            "(ti:Survey OR abs:Survey) AND (ti:Retrieval OR abs:Retrieval) "
            "AND (ti:Augmented OR abs:Augmented) AND (ti:Generation OR abs:Generation)",
        ),
        ('ti:"graph rag" AND cat:cs.CL', 'ti:"graph rag" AND cat:cs.CL'),
        ("au:Vaswani", "au:Vaswani"),
        ('"attention is all you need"', '"attention is all you need"'),
        ("transformers OR mamba", "transformers OR mamba"),
        ("检索增强生成", "检索增强生成"),
    ],
)
def test_build_arxiv_query(text, expected):
    assert search.build_arxiv_query(text) == expected


def test_search_arxiv_formats_each_paper(fake_arxiv):
    fake_arxiv.results_for = staticmethod(
        lambda query: [
            _arxiv_result("2403.00001v1", "Graph  RAG\nfor Science"),
            _arxiv_result("2403.00002v2", "Many Authors", n_authors=6),
        ]
    )

    output = search.search_arxiv.invoke({"query": "graph rag", "max_results": 2})

    assert "[1] Graph RAG for Science" in output
    assert "arXiv: 2403.00001v1" in output
    assert "发表: 2024-03-05" in output
    assert "链接: http://arxiv.org/abs/2403.00001v1" in output
    assert "摘要: A summary spanning two lines." in output
    assert "[2] Many Authors" in output
    assert "Author 1, Author 2, Author 3, Author 4 等" in output
    assert [s.query for s in fake_arxiv.searches] == ["(ti:graph OR abs:graph) AND (ti:rag OR abs:rag)"]


def test_search_arxiv_falls_back_to_raw_query_when_strict_query_finds_nothing(fake_arxiv):
    fake_arxiv.results_for = staticmethod(
        lambda query: [_arxiv_result("2403.00009v1", "Loose Match")] if query == "graph rag" else []
    )

    output = search.search_arxiv.invoke({"query": "graph rag"})

    assert [s.query for s in fake_arxiv.searches] == [
        "(ti:graph OR abs:graph) AND (ti:rag OR abs:rag)",
        "graph rag",
    ]
    assert "[1] Loose Match" in output


def test_search_arxiv_clamps_max_results(fake_arxiv):
    search.search_arxiv.invoke({"query": "x", "max_results": 500})
    assert {s.max_results for s in fake_arxiv.searches} == {search.MAX_RESULTS_LIMIT}


def test_search_arxiv_reports_empty_result(fake_arxiv):
    output = search.search_arxiv.invoke({"query": "nothing here"})
    assert "没有检索到" in output
    assert "nothing here" in output
    assert len(fake_arxiv.searches) == 2


def test_search_arxiv_returns_readable_error_instead_of_raising(fake_arxiv):
    fake_arxiv.error = ConnectionError("network down")
    output = search.search_arxiv.invoke({"query": "x"})
    assert "arXiv 检索失败" in output
    assert "network down" in output


def test_search_web_uses_tavily_when_key_is_configured(monkeypatch):
    calls = {}

    class FakeTavily:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def search(self, query, max_results):
            calls["query"] = query
            return {"results": [{"title": "T", "url": "https://t.example", "content": "tavily body"}]}

    def ddgs_must_not_be_used():
        raise AssertionError("配置了 Tavily Key 时不应调用 DDGS")

    monkeypatch.setattr(search, "get_settings", lambda: _settings("tvly-key"))
    monkeypatch.setattr(search, "TavilyClient", FakeTavily)
    monkeypatch.setattr(search, "DDGS", ddgs_must_not_be_used)

    output = search.search_web.invoke({"query": "agent survey"})

    assert calls == {"api_key": "tvly-key", "query": "agent survey"}
    assert "[1] T" in output
    assert "链接: https://t.example" in output
    assert "摘要: tavily body" in output


def test_search_web_falls_back_to_ddgs_without_tavily_key(monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results):
            return [{"title": "D", "href": "https://d.example", "body": "ddgs body"}]

    def tavily_must_not_be_used(**kwargs):
        raise AssertionError("没有 Tavily Key 时不应调用 Tavily")

    monkeypatch.setattr(search, "get_settings", lambda: _settings(""))
    monkeypatch.setattr(search, "TavilyClient", tavily_must_not_be_used)
    monkeypatch.setattr(search, "DDGS", FakeDDGS)

    output = search.search_web.invoke({"query": "agent survey"})

    assert "[1] D" in output
    assert "链接: https://d.example" in output
    assert "摘要: ddgs body" in output


def test_search_web_returns_readable_error_instead_of_raising(monkeypatch):
    class BrokenDDGS:
        def text(self, query, max_results):
            raise TimeoutError("timed out")

    monkeypatch.setattr(search, "get_settings", lambda: _settings(""))
    monkeypatch.setattr(search, "DDGS", BrokenDDGS)

    output = search.search_web.invoke({"query": "x"})

    assert "网页搜索失败" in output
    assert "timed out" in output

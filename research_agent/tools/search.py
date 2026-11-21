"""搜集资料：arXiv 检索与网页搜索。"""

import re
from dataclasses import dataclass

import arxiv
from ddgs import DDGS
from langchain_core.tools import tool
from tavily import TavilyClient

from research_agent.config import get_settings

MAX_RESULTS_LIMIT = 10
ABSTRACT_CHAR_LIMIT = 700
SNIPPET_CHAR_LIMIT = 400
ARXIV_QUERY_SYNTAX_RE = re.compile(r'\b(?:ti|abs|all|au|cat|co|jr|id):|\b(?:AND|OR|ANDNOT)\b|"')
ARXIV_STOP_WORDS = {"a", "an", "the", "of", "for", "in", "on", "and", "or", "with", "to", "via", "using"}


@dataclass(frozen=True)
class PaperHit:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    summary: str
    url: str


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    snippet: str


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _clip(text: str, limit: int) -> str:
    text = _one_line(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _clamp_max_results(max_results: int) -> int:
    return max(1, min(int(max_results), MAX_RESULTS_LIMIT))


def build_arxiv_query(text: str) -> str:
    """把一串关键词变成“每个词都出现在标题或摘要中”的 arXiv 检索式。

    arXiv 接口对不带字段的多词检索相关度很差（实测检索 RAG 综述会排出图像生成论文），
    限定到标题和摘要并用 AND 连接后结果才切题。已经带字段前缀、布尔运算符或引号的
    检索式视为用户自己写好的，原样返回。
    """
    if ARXIV_QUERY_SYNTAX_RE.search(text):
        return text
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9]+", text)
        if word.lower() not in ARXIV_STOP_WORDS
    ]
    if not words:
        return text
    return " AND ".join(f"(ti:{word} OR abs:{word})" for word in words)


def _run_arxiv_query(client: arxiv.Client, query: str, max_results: int) -> list[PaperHit]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [
        PaperHit(
            arxiv_id=result.get_short_id(),
            title=_one_line(result.title),
            authors=[_one_line(author.name) for author in result.authors],
            published=result.published.strftime("%Y-%m-%d"),
            summary=_one_line(result.summary),
            url=result.entry_id,
        )
        for result in client.results(search)
    ]


def arxiv_search(query: str, max_results: int = 5) -> list[PaperHit]:
    """检索 arXiv，按相关度返回论文。网络或接口出错时抛出异常。"""
    max_results = _clamp_max_results(max_results)
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=2)
    strict_query = build_arxiv_query(query)
    hits = _run_arxiv_query(client, strict_query, max_results)
    if not hits and strict_query != query:
        # 关键词太多时“全部命中”可能一篇都没有，退回原始检索式保证召回。
        hits = _run_arxiv_query(client, query, max_results)
    return hits


def format_paper_hits(hits: list[PaperHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        authors = ", ".join(hit.authors[:4]) + (" 等" if len(hit.authors) > 4 else "")
        blocks.append(
            f"[{index}] {hit.title}\n"
            f"    作者: {authors}\n"
            f"    发表: {hit.published} | arXiv: {hit.arxiv_id}\n"
            f"    链接: {hit.url}\n"
            f"    摘要: {_clip(hit.summary, ABSTRACT_CHAR_LIMIT)}"
        )
    return "\n\n".join(blocks)


def web_search(query: str, max_results: int = 5) -> list[WebHit]:
    """网页搜索：配置了 TAVILY_API_KEY 用 Tavily，否则用 DuckDuckGo。出错时抛出异常。"""
    max_results = _clamp_max_results(max_results)
    tavily_key = get_settings().tavily_api_key
    if tavily_key:
        response = TavilyClient(api_key=tavily_key).search(query, max_results=max_results)
        return [
            WebHit(title=item["title"], url=item["url"], snippet=item.get("content", ""))
            for item in response.get("results", [])
        ]
    return [
        WebHit(title=item["title"], url=item["href"], snippet=item.get("body", ""))
        for item in DDGS().text(query, max_results=max_results)
    ]


def format_web_hits(hits: list[WebHit]) -> str:
    return "\n\n".join(
        f"[{index}] {hit.title}\n"
        f"    链接: {hit.url}\n"
        f"    摘要: {_clip(hit.snippet, SNIPPET_CHAR_LIMIT)}"
        for index, hit in enumerate(hits, start=1)
    )


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """在 arXiv 上检索学术论文，返回标题、作者、发表日期、arXiv 编号、链接和摘要。

    Args:
        query: 3 到 5 个英文关键词，用空格分隔（如 "LLM agent memory survey"），要求每个词都出现在
            标题或摘要中。中文主题请先译成英文关键词；也可以直接写 arXiv 检索语法（如 ti:"graph rag"）。
        max_results: 返回条数，1 到 10。
    """
    try:
        hits = arxiv_search(query, max_results)
    except Exception as exc:  # 工具边界：把失败原因交还给模型，而不是中断对话
        return f"arXiv 检索失败（{type(exc).__name__}: {exc}）。可以稍后重试，或改用 search_web。"
    if not hits:
        return f"arXiv 上没有检索到与“{query}”相关的论文，可以换一组英文关键词再试。"
    return format_paper_hits(hits)


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """搜索互联网，返回网页标题、链接和摘要。适合查博客、新闻、代码仓库、数据集等 arXiv 之外的资料。

    Args:
        query: 搜索关键词。
        max_results: 返回条数，1 到 10。
    """
    try:
        hits = web_search(query, max_results)
    except Exception as exc:  # 工具边界：把失败原因交还给模型，而不是中断对话
        return f"网页搜索失败（{type(exc).__name__}: {exc}）。可以稍后重试，或改用 search_arxiv。"
    if not hits:
        return f"没有搜索到与“{query}”相关的网页。"
    return format_web_hits(hits)

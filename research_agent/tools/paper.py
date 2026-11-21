"""读论文：结构化精读、基于检索增强的问答、论文库列表。"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from research_agent.config import get_settings
from research_agent.llm import get_chat_model
from research_agent.paper.loader import (
    LoadedPaper,
    PaperLoadError,
    chunk_paper,
    load_paper,
    slugify,
    store_paper,
)
from research_agent.paper.store import PaperStore, get_store
from research_agent.projects import (
    get_current_project_id,
    project_papers_dir,
)
from research_agent.prompts import (
    ASK_PAPER_SYSTEM,
    ASK_PAPER_TEMPLATE,
    COMPARE_PAPERS_SYSTEM,
    COMPARE_PAPERS_TEMPLATE,
    MERGE_NOTES_TEMPLATE,
    READ_PAPER_SYSTEM,
    READ_PAPER_TEMPLATE,
    SEGMENT_NOTES_TEMPLATE,
)

# 单次总结可容纳的论文字符数。英文约 4 字符一个 token，12 万字符约 3 万 token，
# 绝大多数会议论文都在此范围内；超过则分段总结再合并。
SINGLE_PASS_CHAR_BUDGET = 120_000
RETRIEVAL_TOP_K = 5
COMPARE_MIN_PAPERS = 2
COMPARE_MAX_PAPERS = 6
COMPARE_TOP_K_PER_PAPER = 4


def _ask(llm: BaseChatModel, system: str, user: str) -> str:
    return llm.invoke([SystemMessage(system), HumanMessage(user)]).text.strip()


def split_into_segments(pages: list[str], char_budget: int) -> list[str]:
    """把带页码标记的各页依次装进若干段，每段不超过 char_budget（单页超长时独占一段）。"""
    segments: list[str] = []
    current = ""
    for page_number, text in enumerate(pages, start=1):
        if not text.strip():
            continue
        marked = f"[第{page_number}页]\n{text.strip()}\n\n"
        if current and len(current) + len(marked) > char_budget:
            segments.append(current)
            current = ""
        current += marked
    if current:
        segments.append(current)
    return segments


def summarize_paper(
    paper: LoadedPaper, llm: BaseChatModel, char_budget: int = SINGLE_PASS_CHAR_BUDGET
) -> str:
    segments = split_into_segments(paper.pages, char_budget)
    if len(segments) == 1:
        return _ask(
            llm, READ_PAPER_SYSTEM, READ_PAPER_TEMPLATE.format(title=paper.title, text=segments[0])
        )
    notes = [
        _ask(
            llm,
            READ_PAPER_SYSTEM,
            SEGMENT_NOTES_TEMPLATE.format(
                title=paper.title, index=index, total=len(segments), text=segment
            ),
        )
        for index, segment in enumerate(segments, start=1)
    ]
    return _ask(
        llm,
        READ_PAPER_SYSTEM,
        MERGE_NOTES_TEMPLATE.format(title=paper.title, notes="\n\n".join(notes)),
    )


def _index_paper(
    paper: LoadedPaper, store: PaperStore, project_id: str | None = None
) -> LoadedPaper:
    """确保向量库里是这篇论文的当前内容；内容没变且已入库时不重复计算向量。"""
    project_id = project_id or get_current_project_id()
    if paper.content_changed or not store.has_paper(paper.paper_id, project_id):
        store.add_paper(chunk_paper(paper, project_id), project_id)
    return paper


def ingest_paper(
    source: str,
    store: PaperStore,
    papers_dir: Path,
    project_id: str | None = None,
) -> LoadedPaper:
    """加载论文并确保它已写入向量库。"""
    return _index_paper(load_paper(source, papers_dir), store, project_id)


def ingest_uploaded_pdf(
    filename: str,
    data: bytes,
    store: PaperStore,
    papers_dir: Path,
    project_id: str | None = None,
) -> LoadedPaper:
    """保存用户上传的 PDF 并入库，paper_id 取自文件名。

    同名文件再次上传时以新内容为准；无法解析时抛出 PaperLoadError，已有的同名论文保持不变。
    """
    return _index_paper(
        store_paper(slugify(Path(filename).stem), data, papers_dir), store, project_id
    )


def read_paper_report(
    source: str,
    *,
    llm: BaseChatModel,
    store: PaperStore,
    papers_dir: Path,
    char_budget: int = SINGLE_PASS_CHAR_BUDGET,
    project_id: str | None = None,
) -> str:
    paper = ingest_paper(source, store, papers_dir, project_id)
    summary = summarize_paper(paper, llm, char_budget)
    return (
        f"paper_id: {paper.paper_id}\n"
        f"标题: {paper.title}\n"
        f"页数: {len(paper.pages)}\n\n"
        f"{summary}"
    )


def answer_question(
    question: str,
    *,
    llm: BaseChatModel,
    store: PaperStore,
    paper_id: str | None = None,
    k: int = RETRIEVAL_TOP_K,
    project_id: str | None = None,
) -> str:
    project_id = project_id or get_current_project_id()
    if paper_id and not store.has_paper(paper_id, project_id):
        return f"论文库中没有 paper_id 为“{paper_id}”的论文。可以先用 list_papers 查看已入库论文，或用 read_paper 读入它。"
    docs = store.search(question, k=k, paper_id=paper_id, project_id=project_id)
    if not docs:
        return "论文库中还没有可检索的内容。请先用 read_paper 读入论文。"

    context = "\n\n".join(
        f"《{doc.metadata['title']}》（{doc.metadata['paper_id']}）第{doc.metadata['page']}页：\n"
        f"{doc.page_content}"
        for doc in docs
    )
    answer = _ask(llm, ASK_PAPER_SYSTEM, ASK_PAPER_TEMPLATE.format(question=question, context=context))

    sources = sorted(
        {(doc.metadata["paper_id"], doc.metadata["title"], doc.metadata["page"]) for doc in docs}
    )
    source_lines = "\n".join(
        f"- 《{title}》（{pid}）第 {page} 页" for pid, title, page in sources
    )
    return f"{answer}\n\n检索到的出处：\n{source_lines}"


def compare_papers_report(
    question: str,
    paper_ids: list[str],
    *,
    llm: BaseChatModel,
    store: PaperStore,
    k: int = COMPARE_TOP_K_PER_PAPER,
    project_id: str | None = None,
) -> str:
    """按论文分别检索证据，再生成一份带页码来源的横向对比报告。"""
    project_id = project_id or get_current_project_id()
    normalized = list(dict.fromkeys(paper_id.strip() for paper_id in paper_ids if paper_id.strip()))
    if not COMPARE_MIN_PAPERS <= len(normalized) <= COMPARE_MAX_PAPERS:
        return (
            f"论文对比需要 {COMPARE_MIN_PAPERS} 到 {COMPARE_MAX_PAPERS} 篇不同的论文，"
            f"当前收到 {len(normalized)} 篇。可以先用 list_papers 查看 paper_id。"
        )

    missing = [
        paper_id for paper_id in normalized if not store.has_paper(paper_id, project_id)
    ]
    if missing:
        return (
            "论文库中找不到这些 paper_id："
            + "、".join(missing)
            + "。可以先用 list_papers 查看已入库论文，或用 read_paper 读入。"
        )

    retrieval_query = (
        f"{question} research problem method dataset benchmark result limitation comparison"
    )
    grouped_docs = []
    for paper_id in normalized:
        docs = store.search(
            retrieval_query, k=k, paper_id=paper_id, project_id=project_id
        )
        if not docs:
            return f"论文 {paper_id} 在向量库中没有可检索内容，无法进行可靠对比。"
        grouped_docs.append((paper_id, docs))

    context = "\n\n".join(
        f"### {paper_id}《{docs[0].metadata['title']}》\n"
        + "\n\n".join(
            f"（第{doc.metadata['page']}页）{doc.page_content}" for doc in docs
        )
        for paper_id, docs in grouped_docs
    )
    answer = _ask(
        llm,
        COMPARE_PAPERS_SYSTEM,
        COMPARE_PAPERS_TEMPLATE.format(question=question, context=context),
    )
    sources = sorted(
        {
            (doc.metadata["paper_id"], doc.metadata["title"], doc.metadata["page"])
            for _, docs in grouped_docs
            for doc in docs
        }
    )
    source_lines = "\n".join(
        f"- 《{title}》（{paper_id}）第 {page} 页" for paper_id, title, page in sources
    )
    return f"{answer}\n\n## 检索到的页码依据\n{source_lines}"


def format_paper_list(store: PaperStore, project_id: str | None = None) -> str:
    papers = store.list_papers(project_id)
    if not papers:
        return "论文库是空的。可以用 read_paper 读入论文，或在界面侧边栏上传 PDF。"
    return "\n".join(
        f"- paper_id: {info.paper_id} | 标题: {info.title} | {info.total_pages} 页"
        for info in papers
    )


@tool
def read_paper(source: str) -> str:
    """精读一篇论文并把它存入论文库，返回 paper_id 和结构化精读笔记（研究问题、方法、实验、贡献、局限）。

    Args:
        source: 论文来源，可以是 arXiv 编号（如 1706.03762）、arXiv 或 PDF 链接、本地 PDF 路径，
            或论文库中已有论文的 paper_id。
    """
    try:
        project_id = get_current_project_id()
        return read_paper_report(
            source,
            llm=get_chat_model(),
            store=get_store(),
            papers_dir=project_papers_dir(get_settings(), project_id),
            project_id=project_id,
        )
    except PaperLoadError as exc:
        return f"读取论文失败：{exc}"


@tool
def ask_paper(question: str, paper_id: str | None = None) -> str:
    """针对论文库中已入库的论文提问，基于原文片段回答并给出页码出处。

    Args:
        question: 要问的问题。
        paper_id: 只在这篇论文里找答案；不填则在整个论文库里检索。
    """
    return answer_question(question, llm=get_chat_model(), store=get_store(), paper_id=paper_id)


@tool
def compare_papers(question: str, paper_ids: list[str]) -> str:
    """对比已入库的多篇论文。分别检索每篇论文的相关原文片段，生成按研究问题、方法、
    数据与评测、主要发现、局限展开的 Markdown 横向对比表，并附页码来源。

    Args:
        question: 对比目的，例如“比较它们的检索方法和评测设计”。
        paper_ids: 要对比的 paper_id 列表，2 到 6 篇且必须已经入库。
    """
    return compare_papers_report(
        question, paper_ids, llm=get_chat_model(), store=get_store()
    )


@tool
def list_papers() -> str:
    """列出论文库中已入库的全部论文及其 paper_id。"""
    return format_paper_list(get_store())

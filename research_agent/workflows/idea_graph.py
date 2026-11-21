"""idea 生成工作流：检索文献 -> 生成 idea -> 评审打分 -> 不达标则修订 -> 输出报告。

体现“反思”模式：生成者和评审者是两个不同角色的模型调用，
是否达标由代码按评分阈值判定，修订轮数有上限。
"""

from collections.abc import Callable
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from research_agent.prompts import (
    IDEA_CRITIQUE_SYSTEM,
    IDEA_CRITIQUE_TEMPLATE,
    IDEA_GENERATE_SYSTEM,
    IDEA_GENERATE_TEMPLATE,
    IDEA_QUERY_TEMPLATE,
    IDEA_REVISE_TEMPLATE,
)
from research_agent.tools.search import PaperHit

MAX_QUERIES = 3
PAPERS_PER_QUERY = 5
MAX_PAPERS = 10
NUM_IDEAS = 3
MAX_REVISIONS = 2
NOVELTY_PASS_SCORE = 4
FEASIBILITY_PASS_SCORE = 3
ABSTRACT_CHAR_LIMIT = 600

SearchFn = Callable[[str, int], list[PaperHit]]


class SearchQueries(BaseModel):
    """用于 arXiv 检索的英文检索式。"""

    queries: list[str] = Field(description="2 到 3 条英文检索式")


class Idea(BaseModel):
    title: str = Field(description="idea 的标题，一句话概括")
    motivation: str = Field(description="动机：针对已有工作的什么不足或空白")
    method: str = Field(description="方法思路：数据、模型或算法、评估方式")
    difference: str = Field(description="与已有工作的区别")
    feasibility: str = Field(description="可行性分析：所需资源、主要风险")
    references: list[int] = Field(description="所依据的文献编号，只能取自给出的文献列表")


class IdeaList(BaseModel):
    """一组研究 idea。"""

    ideas: list[Idea]


class IdeaReview(BaseModel):
    idea_index: int = Field(description="被评审 idea 的编号，从 1 开始")
    novelty: int = Field(description="新颖性，1 到 5 的整数")
    feasibility: int = Field(description="可行性，1 到 5 的整数")
    comment: str = Field(description="评语：主要问题和改进方向")


class Critique(BaseModel):
    """对一组研究 idea 的逐个评审。"""

    reviews: list[IdeaReview]


class IdeaState(TypedDict, total=False):
    topic: str
    queries: list[str]
    search_errors: list[str]
    papers: list[PaperHit]
    ideas: list[Idea]
    reviews: list[IdeaReview | None]  # 与 ideas 一一对应；评审漏评的位置为 None
    revisions: int
    report: str


def review_passes(review: IdeaReview | None) -> bool:
    return (
        review is not None
        and review.novelty >= NOVELTY_PASS_SCORE
        and review.feasibility >= FEASIBILITY_PASS_SCORE
    )


def all_reviews_pass(reviews: list[IdeaReview | None]) -> bool:
    """没有任何评审（即没有 idea）不算达标。"""
    return bool(reviews) and all(review_passes(review) for review in reviews)


def _format_papers(papers: list[PaperHit]) -> str:
    return "\n\n".join(
        f"[{index}] {paper.title}（{paper.published[:4]}）\n摘要：{paper.summary[:ABSTRACT_CHAR_LIMIT]}"
        for index, paper in enumerate(papers, start=1)
    )


def _format_ideas(ideas: list[Idea], reviews: list[IdeaReview | None] | None = None) -> str:
    blocks = []
    for index, idea in enumerate(ideas, start=1):
        block = (
            f"### Idea {index}：{idea.title}\n"
            f"动机：{idea.motivation}\n"
            f"方法思路：{idea.method}\n"
            f"与已有工作的区别：{idea.difference}\n"
            f"可行性：{idea.feasibility}\n"
            f"依据文献：{idea.references}"
        )
        if reviews is not None:
            block += "\n" + _format_review(reviews[index - 1])
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_review(review: IdeaReview | None) -> str:
    if review is None:
        return "评审：评审者没有给出这个 idea 的评分，按未达标处理。"
    verdict = "达标" if review_passes(review) else "未达标"
    return (
        f"评审：新颖性 {review.novelty}/5，可行性 {review.feasibility}/5（{verdict}）。{review.comment}"
    )


def _align_reviews(ideas: list[Idea], reviews: list[IdeaReview]) -> list[IdeaReview | None]:
    """按 idea_index 把评审对齐到 idea 上，分数限制在 1 到 5。"""
    by_index = {review.idea_index: review for review in reviews}
    aligned: list[IdeaReview | None] = []
    for index in range(1, len(ideas) + 1):
        review = by_index.get(index)
        if review is not None:
            review = review.model_copy(
                update={
                    "novelty": max(1, min(5, review.novelty)),
                    "feasibility": max(1, min(5, review.feasibility)),
                }
            )
        aligned.append(review)
    return aligned


def render_report(state: IdeaState) -> str:
    topic = state["topic"]
    papers = state.get("papers", [])
    queries = "；".join(state.get("queries", [])) or "无"

    if not papers:
        errors = state.get("search_errors", [])
        reason = "检索出错：" + "；".join(errors) if errors else "这些检索式没有返回任何论文"
        return (
            f"# 研究 idea 报告：{topic}\n\n"
            f"没有生成 idea：未能检索到相关文献（{reason}）。\n\n"
            f"使用的检索式：{queries}\n\n"
            "没有文献依据就无法判断新颖性，因此不输出 idea。可以稍后重试，或换一种主题表述。"
        )

    ideas, reviews = state["ideas"], state["reviews"]
    if not ideas:
        verdict = f"修订 {MAX_REVISIONS} 轮后模型仍没有给出任何 idea，可以换一种主题表述重试"
    elif all_reviews_pass(reviews):
        verdict = "全部 idea 达到评审阈值"
    else:
        verdict = f"已达修订上限（{MAX_REVISIONS} 轮），仍有 idea 未达评审阈值，见各 idea 的评审意见"
    lines = [
        f"# 研究 idea 报告：{topic}",
        "",
        f"- 检索式：{queries}",
        f"- 参考文献：{len(papers)} 篇（arXiv）",
        f"- 修订轮数：{state.get('revisions', 0)}",
        f"- 评审阈值：新颖性不低于 {NOVELTY_PASS_SCORE}/5，可行性不低于 {FEASIBILITY_PASS_SCORE}/5",
        f"- 评审结论：{verdict}",
    ]
    for index, (idea, review) in enumerate(zip(ideas, reviews), start=1):
        valid_refs = sorted({ref for ref in idea.references if 1 <= ref <= len(papers)})
        refs = " ".join(f"[{ref}]" for ref in valid_refs) or "无"
        lines += [
            "",
            f"## Idea {index}：{idea.title}",
            "",
            f"**动机**：{idea.motivation}",
            "",
            f"**方法思路**：{idea.method}",
            "",
            f"**与已有工作的区别**：{idea.difference}",
            "",
            f"**可行性**：{idea.feasibility}",
            "",
            f"**{_format_review(review)}**",
            "",
            f"**依据文献**：{refs}",
        ]
    lines += ["", "## 参考文献", ""]
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors[:3]) + (" 等" if len(paper.authors) > 3 else "")
        lines.append(
            f"[{index}] {authors}. {paper.title}. arXiv:{paper.arxiv_id}, {paper.published[:4]}. {paper.url}"
        )
    return "\n".join(lines)


def build_idea_graph(llm: BaseChatModel, search_fn: SearchFn):
    """llm 负责规划检索式、生成和评审；search_fn(query, max_results) 负责检索文献。"""

    def retrieve(state: IdeaState) -> IdeaState:
        planned = llm.with_structured_output(SearchQueries).invoke(
            [HumanMessage(IDEA_QUERY_TEMPLATE.format(topic=state["topic"]))]
        )
        queries = [query.strip() for query in planned.queries if query.strip()][:MAX_QUERIES]
        papers: dict[str, PaperHit] = {}
        errors: list[str] = []
        for query in queries:
            try:
                hits = search_fn(query, PAPERS_PER_QUERY)
            except Exception as exc:  # 单条检索失败不应中断整个流程，原因记入报告
                errors.append(f"“{query}”：{type(exc).__name__}: {exc}")
                continue
            for hit in hits:
                papers.setdefault(hit.arxiv_id, hit)
        return {
            "queries": queries,
            "search_errors": errors,
            "papers": list(papers.values())[:MAX_PAPERS],
            "revisions": 0,
        }

    def generate(state: IdeaState) -> IdeaState:
        papers_text = _format_papers(state["papers"])
        previous = state.get("ideas")
        if previous:
            prompt = IDEA_REVISE_TEMPLATE.format(
                topic=state["topic"],
                papers=papers_text,
                reviewed_ideas=_format_ideas(previous, state["reviews"]),
                num_ideas=NUM_IDEAS,
            )
        else:
            prompt = IDEA_GENERATE_TEMPLATE.format(
                topic=state["topic"], papers=papers_text, num_ideas=NUM_IDEAS
            )
        result = llm.with_structured_output(IdeaList).invoke(
            [SystemMessage(IDEA_GENERATE_SYSTEM), HumanMessage(prompt)]
        )
        update: IdeaState = {"ideas": result.ideas[:NUM_IDEAS]}
        if previous:
            update["revisions"] = state["revisions"] + 1
        return update

    def critique(state: IdeaState) -> IdeaState:
        result = llm.with_structured_output(Critique).invoke(
            [
                SystemMessage(IDEA_CRITIQUE_SYSTEM),
                HumanMessage(
                    IDEA_CRITIQUE_TEMPLATE.format(
                        topic=state["topic"],
                        papers=_format_papers(state["papers"]),
                        ideas=_format_ideas(state["ideas"]),
                    )
                ),
            ]
        )
        return {"reviews": _align_reviews(state["ideas"], result.reviews)}

    def report(state: IdeaState) -> IdeaState:
        return {"report": render_report(state)}

    def after_retrieve(state: IdeaState) -> str:
        return "generate" if state["papers"] else "report"

    def after_critique(state: IdeaState) -> str:
        approved = all_reviews_pass(state["reviews"])
        return "report" if approved or state["revisions"] >= MAX_REVISIONS else "generate"

    graph = StateGraph(IdeaState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("report", report)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, ["generate", "report"])
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", after_critique, ["generate", "report"])
    graph.add_edge("report", END)
    return graph.compile()

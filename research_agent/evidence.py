"""论文证据矩阵：抽取结构化字段、保存 CSV、供界面和智能体共同使用。"""

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.paper.loader import slugify
from research_agent.paper.store import PaperStore
from research_agent.prompts import EVIDENCE_MATRIX_SYSTEM, EVIDENCE_MATRIX_TEMPLATE
from research_agent.projects import get_current_project_id

MIN_PAPERS = 2
MAX_PAPERS = 6
MAX_CHARS_PER_PAPER = 10_000


class EvidenceRow(BaseModel):
    paper_id: str = Field(description="论文编号，必须逐字使用输入中的 paper_id")
    research_problem: str
    method: str
    datasets: str
    metrics: str
    findings: str
    limitations: str


class EvidenceMatrix(BaseModel):
    rows: list[EvidenceRow]
    synthesis: str
    selection_advice: str


@dataclass(frozen=True)
class EvidenceMatrixResult:
    matrix: EvidenceMatrix
    titles: dict[str, str]
    sources: dict[str, list[int]]

    def csv_bytes(self) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "paper_id",
                "title",
                "research_problem",
                "method",
                "datasets",
                "metrics",
                "findings",
                "limitations",
                "source_pages",
            ],
        )
        writer.writeheader()
        for row in self.matrix.rows:
            data = row.model_dump()
            data["title"] = self.titles.get(row.paper_id, "")
            data["source_pages"] = ", ".join(
                str(page) for page in self.sources.get(row.paper_id, [])
            )
            writer.writerow(data)
        return buffer.getvalue().encode("utf-8-sig")


def _normalize_ids(paper_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(paper_id.strip() for paper_id in paper_ids if paper_id.strip()))


def _paper_context(docs, max_chars: int) -> str:
    blocks = []
    used = 0
    for doc in docs:
        block = f"（第{doc.metadata['page']}页）{doc.page_content.strip()}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                blocks.append(block[:remaining])
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _empty_row(paper_id: str) -> EvidenceRow:
    return EvidenceRow(
        paper_id=paper_id,
        research_problem="原文未说明",
        method="原文未说明",
        datasets="原文未说明",
        metrics="原文未说明",
        findings="原文未说明",
        limitations="原文未说明",
    )


def build_evidence_matrix(
    paper_ids: list[str],
    *,
    llm: BaseChatModel,
    store: PaperStore,
    focus: str = "",
    project_id: str | None = None,
) -> EvidenceMatrixResult:
    project_id = project_id or get_current_project_id()
    normalized = _normalize_ids(paper_ids)
    if not MIN_PAPERS <= len(normalized) <= MAX_PAPERS:
        raise ValueError(f"证据矩阵需要 {MIN_PAPERS} 到 {MAX_PAPERS} 篇不同论文。")

    titles = {info.paper_id: info.title for info in store.list_papers(project_id)}
    missing = [paper_id for paper_id in normalized if paper_id not in titles]
    if missing:
        raise ValueError("论文库中找不到这些 paper_id：" + "、".join(missing))

    contexts = []
    sources = {}
    for paper_id in normalized:
        docs = store.paper_documents(paper_id, project_id)
        if not docs:
            raise ValueError(f"论文 {paper_id} 没有可用的原文片段。")
        sources[paper_id] = sorted({int(doc.metadata["page"]) for doc in docs})
        contexts.append(
            f"### {paper_id}《{titles[paper_id]}》\n{_paper_context(docs, MAX_CHARS_PER_PAPER)}"
        )

    result = llm.with_structured_output(EvidenceMatrix).invoke(
        [
            SystemMessage(EVIDENCE_MATRIX_SYSTEM),
            HumanMessage(
                EVIDENCE_MATRIX_TEMPLATE.format(
                    focus=focus.strip() or "提炼论文核心贡献与局限",
                    papers="\n\n".join(contexts),
                )
            ),
        ]
    )
    by_id = {row.paper_id: row for row in result.rows}
    rows = [by_id.get(paper_id, _empty_row(paper_id)) for paper_id in normalized]
    matrix = result.model_copy(update={"rows": rows})
    return EvidenceMatrixResult(matrix=matrix, titles=titles, sources=sources)


def save_evidence_csv(result: EvidenceMatrixResult, output_dir: Path, focus: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = slugify(focus.strip() or "论文证据矩阵", 40)
    path = output_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.csv"
    path.write_bytes(result.csv_bytes())
    return path


def render_evidence_markdown(result: EvidenceMatrixResult, csv_path: Path) -> str:
    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "# 论文证据矩阵",
        "",
        "| paper_id | 标题 | 研究问题 | 方法 | 数据 | 指标 | 主要发现 | 局限 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.matrix.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    clean(row.paper_id),
                    clean(result.titles.get(row.paper_id, "")),
                    clean(row.research_problem),
                    clean(row.method),
                    clean(row.datasets),
                    clean(row.metrics),
                    clean(row.findings),
                    clean(row.limitations),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## 综合比较",
        "",
        result.matrix.synthesis,
        "",
        "## 选择建议",
        "",
        result.matrix.selection_advice,
        "",
        f"CSV 已保存到 `outputs/{csv_path.name}`。",
    ]
    return "\n".join(lines)

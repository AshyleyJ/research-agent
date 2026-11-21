"""论文证据矩阵工具。"""

from langchain_core.tools import tool

from research_agent.config import get_settings
from research_agent.evidence import (
    build_evidence_matrix as create_matrix,
    render_evidence_markdown,
    save_evidence_csv,
)
from research_agent.llm import get_chat_model
from research_agent.paper.store import get_store
from research_agent.projects import get_current_project_id, project_output_dir


@tool
def build_evidence_matrix(paper_ids: list[str], focus: str = "") -> str:
    """从 2 到 6 篇已入库论文中抽取研究问题、方法、数据、指标、主要发现和局限，
    生成 Markdown 证据矩阵与比较总结，并把同一张表保存为可下载 CSV。

    Args:
        paper_ids: 要分析的 paper_id 列表，必须已经入库。
        focus: 可选的分析关注点，例如“比较检索增强方法和评测设计”。
    """
    project_id = get_current_project_id()
    result = create_matrix(
        paper_ids,
        llm=get_chat_model(),
        store=get_store(),
        focus=focus,
        project_id=project_id,
    )
    path = save_evidence_csv(
        result, project_output_dir(get_settings(), project_id), focus=focus
    )
    return render_evidence_markdown(result, path)

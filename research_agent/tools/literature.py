"""文献调研：调用多角色工作流，并自动把综述保存成可下载笔记。"""

from langchain_core.tools import tool

from research_agent.config import get_settings
from research_agent.llm import get_chat_model
from research_agent.tools.notes import write_note
from research_agent.tools.search import arxiv_search
from research_agent.projects import get_current_project_id, project_output_dir
from research_agent.workflows.literature_graph import build_literature_graph

LITERATURE_TEMPERATURE = 0.2


@tool
def research_literature(topic: str) -> str:
    """围绕一个主题完成完整文献调研：规划多条检索式、检索并去重 arXiv 文献、
    撰写带编号引用的中文综述、执行质量评审和最多一次修订，最后自动保存为 Markdown 笔记。

    适合“调研某个方向”“系统梳理文献”“写一份综述”等需要多篇论文的任务。
    如果只是查找少量论文，请改用 search_arxiv。

    Args:
        topic: 需要调研的研究主题，中英文均可，越具体越好。
    """
    graph = build_literature_graph(
        get_chat_model(temperature=LITERATURE_TEMPERATURE), arxiv_search,
        mode=getattr(get_settings(), "literature_mode", "workflow"),
    )
    result = graph.invoke({"topic": topic})
    report = result["report"]
    if not result.get("papers"):
        return report

    path = write_note(
        f"文献综述-{topic}",
        report,
        project_output_dir(get_settings(), get_current_project_id()),
    )
    return f"{report}\n\n---\n已自动保存到 outputs/{path.name}"

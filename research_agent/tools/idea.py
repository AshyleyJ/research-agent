"""给出 idea：把 idea 生成工作流包装成智能体可调用的工具。"""

from langchain_core.tools import tool

from research_agent.llm import get_chat_model
from research_agent.tools.search import arxiv_search
from research_agent.workflows.idea_graph import build_idea_graph

IDEA_TEMPERATURE = 0.7


@tool
def generate_ideas(topic: str) -> str:
    """围绕一个研究主题生成研究 idea 报告：自动检索 arXiv 文献，提出若干 idea，经评审打分和修订后输出 Markdown 报告（含动机、方法思路、与已有工作的区别、可行性、参考文献）。耗时较长，约一到两分钟。

    Args:
        topic: 研究主题或方向，中英文均可，越具体越好。
    """
    graph = build_idea_graph(get_chat_model(temperature=IDEA_TEMPERATURE), arxiv_search)
    return graph.invoke({"topic": topic})["report"]

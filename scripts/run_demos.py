"""用真实 API 做连通性自检，并依次跑三个演示场景，过程保存到 outputs/demo/。

在项目根目录运行：
    python -m scripts.run_demos                自检通过后跑全部场景
    python -m scripts.run_demos --scenario 3   自检通过后只跑第 3 个场景
    python -m scripts.run_demos --check        只做自检
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.agent import (
    AnswerText,
    ToolFinished,
    ToolStarted,
    build_agent,
    history_to_turns,
    stream_turn,
)
from research_agent.config import get_settings
from research_agent.llm import MissingApiKeyError, get_chat_model
from research_agent.tools.search import search_arxiv
from research_agent.workflows.idea_graph import SearchQueries

RESULT_PREVIEW_CHARS = 1500

# 每个场景是同一会话里依次提出的若干个问题。
SCENARIOS = [
    (
        "文献调研并保存综述",
        ["调研 LLM 智能体记忆机制方向近两年的代表性工作，写一份简短综述并保存成笔记"],
    ),
    (
        "精读论文并追问细节",
        [
            "精读 arXiv 论文 1706.03762",
            "这篇论文的 base 模型用了几个注意力头？训练用的什么优化器，学习率怎么调整？请给出页码",
        ],
    ),
    (
        "生成研究 idea",
        ["围绕“面向长文档问答的检索增强生成”给我几个适合硕士生做的研究 idea，并把报告保存下来"],
    ),
]


def check_connectivity() -> bool:
    """依次检验智能体依赖的三种调用方式：普通对话、工具调用、结构化输出。"""
    llm = get_chat_model()

    def plain() -> str:
        return llm.invoke([HumanMessage("用一句话介绍你自己。")]).text.strip()

    def tool_call() -> str:
        reply = llm.bind_tools([search_arxiv]).invoke(
            [HumanMessage("请用工具在 arXiv 上检索 retrieval augmented generation survey。")]
        )
        if not reply.tool_calls:
            raise RuntimeError(f"模型没有发起工具调用，而是直接回答：{reply.text[:100]}")
        return json.dumps(reply.tool_calls[0]["args"], ensure_ascii=False)

    def structured() -> str:
        result = llm.with_structured_output(SearchQueries).invoke(
            [HumanMessage("研究主题：图神经网络。请给出 2 条用于 arXiv 检索的英文检索式。")]
        )
        return json.dumps(result.queries, ensure_ascii=False)

    all_ok = True
    for name, check in [("普通对话", plain), ("工具调用", tool_call), ("结构化输出", structured)]:
        try:
            print(f"[通过] {name}：{check()}")
        except Exception as exc:  # 自检的目的就是把任何失败原因打印出来
            all_ok = False
            print(f"[失败] {name}：{type(exc).__name__}: {exc}")
    return all_ok


def render_transcript(title: str, messages: list) -> str:
    lines = [f"# 演示场景：{title}", "", f"- 模型：{get_settings().deepseek_model}", f"- 时间：{datetime.now():%Y-%m-%d %H:%M}"]
    for turn in history_to_turns(messages):
        if turn.role == "user":
            lines += ["", "## 用户", "", turn.content]
            continue
        lines += ["", "## 智能体"]
        for step in turn.steps:
            result = step.result if step.result is not None else "（没有执行完）"
            if len(result) > RESULT_PREVIEW_CHARS:
                result = result[:RESULT_PREVIEW_CHARS] + f"\n…（共 {len(result)} 字符）"
            status = "失败" if step.failed else "完成"
            lines += [
                "",
                f"**工具调用（{status}）：`{step.name}`**",
                "",
                f"参数：`{json.dumps(step.args, ensure_ascii=False)}`",
                "",
                "```text",
                result,
                "```",
            ]
        lines += ["", turn.content]
    return "\n".join(lines) + "\n"


def run_scenario(index: int, title: str, prompts: list[str], demo_dir: Path) -> Path:
    print(f"\n===== 场景 {index}：{title} =====")
    agent = build_agent(InMemorySaver())
    config = {"configurable": {"thread_id": f"demo-{index}"}}
    for prompt in prompts:
        print(f"\n[用户] {prompt}")
        for event in stream_turn(agent, config, prompt):
            if isinstance(event, ToolStarted):
                print(f"\n[调用工具] {event.name} {json.dumps(event.args, ensure_ascii=False)[:200]}")
            elif isinstance(event, ToolFinished):
                print(f"[工具{'失败' if event.failed else '完成'}] {' '.join(event.result.split())[:200]}")
            elif isinstance(event, AnswerText):
                print(event.text, end="", flush=True)
        print()
    demo_dir.mkdir(parents=True, exist_ok=True)
    path = demo_dir / f"scenario-{index}.md"
    path.write_text(
        render_transcript(title, agent.get_state(config).values["messages"]), encoding="utf-8"
    )
    print(f"\n已保存过程记录：{path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="只做连通性自检")
    parser.add_argument(
        "--scenario", type=int, choices=range(1, len(SCENARIOS) + 1), help="只跑指定编号的场景"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # 控制台编码不支持的字符用 ? 代替，而不是报错中断

    try:
        connected = check_connectivity()
    except MissingApiKeyError as exc:
        print(exc)
        return 1
    if not connected:
        print("\n自检未通过，没有运行演示场景。")
        return 1
    if args.check:
        return 0

    demo_dir = get_settings().output_dir / "demo"
    for index, (title, prompts) in enumerate(SCENARIOS, start=1):
        if args.scenario in (None, index):
            run_scenario(index, title, prompts, demo_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

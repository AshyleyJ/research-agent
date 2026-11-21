"""Evidence cards retain exact abstract excerpts, never inferred source metadata."""

import re
from typing import Literal
from pydantic import BaseModel, Field

from research_agent.workflows.research_workers import source_id


class EvidenceCard(BaseModel):
    source_id: str
    task_setting: str = Field(description="摘要中的任务设定；不明确则写未知")
    relevance: Literal["direct", "adjacent", "unknown"]
    claim: str = Field(description="这段证据能够支持的一个有限结论")
    quote: str = Field(description="逐字复制摘要中支撑结论的连续原文，不翻译或拼接")
    evidence_type: Literal["author_problem", "author_result", "unknown"]
    unknowns: list[str] = Field(default_factory=list)


class EvidenceCards(BaseModel):
    cards: list[EvidenceCard]


def normalize(text):
    return " ".join(text.split())


def validate_cards(cards, papers):
    sources = {source_id(p.arxiv_id): p for p in papers}
    valid = {}
    rejected = 0
    for card in cards:
        key = source_id(card.source_id)
        quote = normalize(card.quote)
        if key not in sources or len(quote) < 20 or quote not in normalize(sources[key].summary):
            rejected += 1
            continue
        if key not in valid:
            valid[key] = card.model_copy(update={"source_id": key})
    # Missing/invalid cards remain explicit unknowns, not fabricated evidence.
    for key in sources:
        valid.setdefault(key, EvidenceCard(source_id=key, task_setting="未知", relevance="unknown",
            claim="待核验，不据此作强结论", quote="", evidence_type="unknown",
            unknowns=["未取得通过原文片段校验的证据卡"]))
    return list(valid.values()), rejected


def format_cards(cards, papers):
    numbers = {source_id(p.arxiv_id): i for i, p in enumerate(papers, 1)}
    lines = ["证据卡（相关性/证据类型为模型标注；仅原文片段存在性经过代码校验，不代表语义已验证）："]
    for card in cards:
        if card.source_id not in numbers:
            continue
        lines.append(f"[{numbers[card.source_id]}] 任务={card.task_setting}；相关性={card.relevance}；类型={card.evidence_type}")
        lines.append("有限结论：" + re.sub(r"\[\d+\]", "", card.claim))
        lines.append("摘要原句：" + card.quote)
        lines.append("未知项：" + "；".join(card.unknowns))
    return "\n".join(lines)

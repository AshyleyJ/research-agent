import csv
import io

import pytest
from langchain_core.documents import Document

from research_agent.evidence import (
    EvidenceMatrix,
    EvidenceRow,
    build_evidence_matrix,
    save_evidence_csv,
)


class StructuredOnlyModel:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def with_structured_output(self, schema):
        model = self

        class Runner:
            def invoke(self, messages):
                model.prompts.append("\n".join(message.content for message in messages))
                return model.result

        return Runner()


def _chunk(paper_id, page, text, title):
    return Document(
        page_content=text,
        metadata={
            "paper_id": paper_id,
            "title": title,
            "page": page,
            "chunk": 0,
            "total_pages": 5,
        },
    )


def _matrix():
    return EvidenceMatrix(
        rows=[
            EvidenceRow(
                paper_id="p1",
                research_problem="long document QA",
                method="sparse retrieval",
                datasets="NQ",
                metrics="accuracy",
                findings="efficient",
                limitations="no reranker",
            )
        ],
        synthesis="两篇论文采用不同的检索路径。",
        selection_advice="资源有限时优先 p1。",
    )


def test_evidence_matrix_uses_each_selected_paper_and_fills_missing_rows(store):
    store.add_paper(
        [
            _chunk("p1", 1, "Problem: long document QA. Method: sparse retrieval.", "Paper One"),
            _chunk("p2", 3, "Problem: QA. Method: dense retrieval and reranking.", "Paper Two"),
        ],
        project_id="a",
    )
    llm = StructuredOnlyModel(_matrix())

    result = build_evidence_matrix(
        ["p1", "p2", "p1"], llm=llm, store=store, focus="检索方法", project_id="a"
    )

    assert [row.paper_id for row in result.matrix.rows] == ["p1", "p2"]
    assert result.matrix.rows[1].method == "原文未说明"
    assert result.titles == {"p1": "Paper One", "p2": "Paper Two"}
    assert result.sources == {"p1": [1], "p2": [3]}
    prompt = llm.prompts[0]
    assert "散" not in prompt  # 中文关注点不会误当作论文内容
    assert "检索方法" in prompt
    assert "sparse retrieval" in prompt and "dense retrieval" in prompt


def test_evidence_matrix_rejects_unknown_papers_before_model_call(store):
    llm = StructuredOnlyModel(_matrix())

    with pytest.raises(ValueError, match="2 到 6"):
        build_evidence_matrix(["p1"], llm=llm, store=store, project_id="a")
    with pytest.raises(ValueError, match="找不到"):
        build_evidence_matrix(["p1", "missing"], llm=llm, store=store, project_id="a")

    assert llm.prompts == []


def test_evidence_matrix_csv_is_excel_friendly_and_keeps_sources(store, tmp_path):
    store.add_paper(
        [
            _chunk("p1", 1, "sparse method", "Paper One"),
            _chunk("p2", 3, "dense method", "Paper Two"),
        ],
        project_id="a",
    )
    result = build_evidence_matrix(
        ["p1", "p2"], llm=StructuredOnlyModel(_matrix()), store=store, project_id="a"
    )

    path = save_evidence_csv(result, tmp_path, focus="检索方法")

    assert path.suffix == ".csv"
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = list(
        csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
    )
    assert rows[0]["title"] == "Paper One"
    assert rows[0]["source_pages"] == "1"
    assert rows[1]["paper_id"] == "p2"
    assert rows[1]["method"] == "原文未说明"

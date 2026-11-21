from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from research_agent.paper.loader import PaperLoadError
from research_agent.tools import paper as paper_tools
from research_agent.tools.paper import (
    answer_question,
    compare_papers_report,
    format_paper_list,
    ingest_uploaded_pdf,
    read_paper_report,
    split_into_segments,
)


def _chunk(paper_id: str, page: int, text: str, title: str) -> Document:
    return Document(
        page_content=text,
        metadata={"paper_id": paper_id, "title": title, "page": page, "chunk": 0, "total_pages": 5},
    )


def test_split_into_segments_packs_pages_within_budget_and_skips_blank_pages():
    pages = ["a" * 40, "   ", "b" * 40, "c" * 40]

    segments = split_into_segments(pages, char_budget=120)

    assert len(segments) == 2
    assert "[第1页]" in segments[0] and "[第3页]" in segments[0]
    assert "[第2页]" not in "".join(segments)
    assert "[第4页]" in segments[1]


def test_split_into_segments_gives_an_oversized_page_its_own_segment():
    segments = split_into_segments(["short", "x" * 500, "short"], char_budget=100)

    assert len(segments) == 3
    assert "x" * 500 in segments[1]


def test_read_paper_summarises_short_paper_in_one_call(scripted_model, store, papers_dir, sample_pdf):
    llm = scripted_model(responses=["## 研究问题\n科学问答中的检索增强（第1页）"])

    report = read_paper_report(str(sample_pdf), llm=llm, store=store, papers_dir=papers_dir)

    assert len(llm.calls) == 1
    prompt = llm.prompt_text(0)
    assert "Citation Graph Retrieval for Science QA" in prompt
    assert "[第1页]" in prompt and "[第3页]" in prompt
    assert "71.3 accuracy" in prompt
    assert report.startswith("paper_id: sample-paper\n标题: Citation Graph Retrieval for Science QA\n页数: 3\n")
    assert report.endswith("## 研究问题\n科学问答中的检索增强（第1页）")
    assert store.has_paper("sample-paper")


def test_read_paper_summarises_long_paper_by_segments_then_merges(
    scripted_model, store, papers_dir, sample_pdf
):
    llm = scripted_model(responses=["笔记一", "笔记二", "笔记三", "## 研究问题\n合并后的笔记"])

    report = read_paper_report(
        str(sample_pdf), llm=llm, store=store, papers_dir=papers_dir, char_budget=150
    )

    assert len(llm.calls) == 4
    assert "第 1/3 部分" in llm.prompt_text(0) and "[第1页]" in llm.prompt_text(0)
    assert "第 3/3 部分" in llm.prompt_text(2) and "71.3 accuracy" in llm.prompt_text(2)
    merge_prompt = llm.prompt_text(3)
    assert "笔记一" in merge_prompt and "笔记二" in merge_prompt and "笔记三" in merge_prompt
    assert "71.3 accuracy" not in merge_prompt
    assert report.endswith("## 研究问题\n合并后的笔记")


def test_reading_the_same_paper_twice_does_not_duplicate_chunks(
    scripted_model, store, papers_dir, sample_pdf
):
    llm = scripted_model(responses=["第一次", "第二次"])

    read_paper_report(str(sample_pdf), llm=llm, store=store, papers_dir=papers_dir)
    report = read_paper_report("sample-paper", llm=llm, store=store, papers_dir=papers_dir)

    assert [p.chunks for p in store.list_papers()] == [3]
    assert report.endswith("第二次")


def test_uploaded_pdf_is_saved_under_its_file_name_and_ingested(store, papers_dir, sample_pdf):
    paper = ingest_uploaded_pdf("我的 论文(终稿).pdf", sample_pdf.read_bytes(), store, papers_dir)

    assert paper.paper_id == "我的-论文-终稿"
    assert paper.path == papers_dir / "我的-论文-终稿.pdf"
    assert paper.path.read_bytes() == sample_pdf.read_bytes()
    assert [(p.paper_id, p.total_pages) for p in store.list_papers()] == [("我的-论文-终稿", 3)]


def test_uploading_the_same_file_name_again_replaces_the_old_content(
    store, papers_dir, sample_pdf, make_pdf, tmp_path
):
    ingest_uploaded_pdf("draft.pdf", sample_pdf.read_bytes(), store, papers_dir)
    revised = make_pdf(tmp_path / "revised.pdf", pages=["Completely new content about diffusion models."])

    paper = ingest_uploaded_pdf("draft.pdf", revised.read_bytes(), store, papers_dir)

    assert len(paper.pages) == 1
    assert [(p.paper_id, p.total_pages, p.chunks) for p in store.list_papers()] == [("draft", 1, 1)]
    assert "diffusion models" in store.search("contrastive retriever citation graphs", k=1)[0].page_content


def test_uploading_an_invalid_pdf_raises_and_leaves_nothing_behind(store, papers_dir):
    with pytest.raises(PaperLoadError, match="不是有效的 PDF 文件"):
        ingest_uploaded_pdf("broken.pdf", b"this is not a pdf", store, papers_dir)

    assert list(papers_dir.iterdir()) == []
    assert store.list_papers() == []


def test_a_failed_re_upload_keeps_the_existing_paper_intact(store, papers_dir, sample_pdf):
    ingest_uploaded_pdf("draft.pdf", sample_pdf.read_bytes(), store, papers_dir)

    with pytest.raises(PaperLoadError):
        ingest_uploaded_pdf("draft.pdf", b"this is not a pdf", store, papers_dir)

    assert (papers_dir / "draft.pdf").read_bytes() == sample_pdf.read_bytes()
    assert [(p.paper_id, p.total_pages, p.chunks) for p in store.list_papers()] == [("draft", 3, 3)]


def test_re_reading_an_edited_local_file_refreshes_the_library(
    scripted_model, store, papers_dir, sample_pdf, make_pdf, tmp_path
):
    source = tmp_path / "draft.pdf"
    source.write_bytes(sample_pdf.read_bytes())
    llm = scripted_model(responses=["旧版笔记", "新版笔记"])
    read_paper_report(str(source), llm=llm, store=store, papers_dir=papers_dir)
    make_pdf(source, pages=["A rewritten single page about diffusion models."])

    report = read_paper_report(str(source), llm=llm, store=store, papers_dir=papers_dir)

    assert "页数: 1" in report
    assert "diffusion models" in llm.prompt_text(1)
    assert [(p.total_pages, p.chunks) for p in store.list_papers()] == [(1, 1)]
    assert "diffusion models" in store.search("contrastive retriever citation graphs", k=1)[0].page_content


def test_ask_paper_answers_from_retrieved_chunks_and_lists_page_sources(scripted_model, store):
    store.add_paper(
        [
            _chunk("p1", 2, "The retriever is trained with a contrastive loss.", "Paper One"),
            _chunk("p1", 4, "The learning rate is 0.0003 and batch size is 64.", "Paper One"),
        ]
    )
    llm = scripted_model(responses=["学习率为 0.0003（第4页）。"])

    output = answer_question("What is the learning rate?", llm=llm, store=store, k=1)

    assert len(llm.calls) == 1
    prompt = llm.prompt_text(0)
    assert "What is the learning rate?" in prompt
    assert "《Paper One》（p1）第4页" in prompt
    assert "learning rate is 0.0003" in prompt
    assert "contrastive loss" not in prompt
    assert output == "学习率为 0.0003（第4页）。\n\n检索到的出处：\n- 《Paper One》（p1）第 4 页"


def test_ask_paper_restricts_retrieval_to_the_given_paper(scripted_model, store):
    store.add_paper([_chunk("p1", 1, "learning rate 0.1", "Paper One")])
    store.add_paper([_chunk("p2", 7, "learning rate 0.2", "Paper Two")])
    llm = scripted_model(responses=["0.2（第7页）"])

    output = answer_question("learning rate", llm=llm, store=store, paper_id="p2")

    assert "learning rate 0.2" in llm.prompt_text(0)
    assert "learning rate 0.1" not in llm.prompt_text(0)
    assert "- 《Paper Two》（p2）第 7 页" in output
    assert "Paper One" not in output


def test_ask_paper_does_not_call_model_for_unknown_paper_or_empty_library(scripted_model, store):
    llm = scripted_model(responses=[])

    assert "论文库中还没有可检索的内容" in answer_question("q", llm=llm, store=store)

    store.add_paper([_chunk("p1", 1, "text", "Paper One")])
    assert "没有 paper_id 为“nope”的论文" in answer_question(
        "q", llm=llm, store=store, paper_id="nope"
    )
    assert llm.calls == []


def test_compare_papers_retrieves_each_selected_paper_and_returns_page_sources(
    scripted_model, store
):
    store.add_paper(
        [
            _chunk("p1", 2, "Method uses sparse retrieval and BM25.", "Paper One"),
            _chunk("p1", 5, "Evaluation uses Natural Questions.", "Paper One"),
        ]
    )
    store.add_paper(
        [
            _chunk("p2", 3, "Method uses dense retrieval and a reranker.", "Paper Two"),
            _chunk("p2", 7, "Evaluation uses HotpotQA.", "Paper Two"),
        ]
    )
    store.add_paper([_chunk("p3", 1, "Unrelated method and benchmark.", "Paper Three")])
    llm = scripted_model(responses=["| 方法 | 稀疏检索 | 稠密检索 |\n\n稀疏方案更轻量。"])

    output = compare_papers_report(
        "比较两篇论文的检索方法和评测设计",
        ["p1", "p2", "p1"],
        llm=llm,
        store=store,
    )

    assert len(llm.calls) == 1
    prompt = llm.prompt_text(0)
    assert "### p1《Paper One》" in prompt
    assert "### p2《Paper Two》" in prompt
    assert "sparse retrieval" in prompt and "dense retrieval" in prompt
    assert "Natural Questions" in prompt and "HotpotQA" in prompt
    assert "Paper Three" not in prompt
    assert output.startswith("| 方法 | 稀疏检索 | 稠密检索 |")
    assert "## 检索到的页码依据" in output
    assert "- 《Paper One》（p1）第 2 页" in output
    assert "- 《Paper Two》（p2）第 3 页" in output


def test_compare_papers_validates_count_and_known_ids_before_calling_model(
    scripted_model, store
):
    store.add_paper([_chunk("p1", 1, "known paper", "Paper One")])
    llm = scripted_model(responses=[])

    too_few = compare_papers_report("compare", ["p1", "p1"], llm=llm, store=store)
    unknown = compare_papers_report("compare", ["p1", "missing"], llm=llm, store=store)
    too_many = compare_papers_report(
        "compare", [f"p{i}" for i in range(1, 8)], llm=llm, store=store
    )

    assert "需要 2 到 6 篇不同" in too_few
    assert "找不到这些 paper_id：missing" in unknown
    assert "当前收到 7 篇" in too_many
    assert llm.calls == []


def test_format_paper_list(store):
    assert "论文库是空的" in format_paper_list(store)

    store.add_paper([_chunk("p1", 1, "text", "Paper One")])

    assert format_paper_list(store) == "- paper_id: p1 | 标题: Paper One | 5 页"


def test_read_paper_tool_returns_readable_message_for_unknown_source(
    monkeypatch, scripted_model, store, papers_dir
):
    llm = scripted_model(responses=[])
    monkeypatch.setattr(paper_tools, "get_chat_model", lambda: llm)
    monkeypatch.setattr(paper_tools, "get_store", lambda: store)
    monkeypatch.setattr(paper_tools, "get_settings", lambda: SimpleNamespace(papers_dir=papers_dir))

    output = paper_tools.read_paper.invoke({"source": "no-such-paper"})

    assert output.startswith("读取论文失败：无法识别论文来源“no-such-paper”")
    assert llm.calls == []

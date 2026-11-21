import dataclasses

import pymupdf
import pytest
import requests

from research_agent.paper import loader
from research_agent.paper.loader import (
    PaperLoadError,
    chunk_paper,
    download_pdf,
    load_paper,
    resolve_source,
    slugify,
)


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.mark.parametrize(
    "source, expected_id, expected_url",
    [
        ("1706.03762", "1706.03762", "https://arxiv.org/pdf/1706.03762"),
        ("arXiv:1706.03762v5", "1706.03762", "https://arxiv.org/pdf/1706.03762v5"),
        ("https://arxiv.org/abs/2401.12345v2", "2401.12345", "https://arxiv.org/pdf/2401.12345v2"),
        ("https://arxiv.org/pdf/2401.12345", "2401.12345", "https://arxiv.org/pdf/2401.12345"),
    ],
)
def test_resolve_source_recognises_arxiv_ids_and_links(papers_dir, source, expected_id, expected_url):
    resolved = resolve_source(source, papers_dir)
    assert resolved.paper_id == expected_id
    assert resolved.url == expected_url


def test_resolve_source_gives_other_urls_a_stable_distinct_id(papers_dir):
    first = resolve_source("https://example.org/a/paper.pdf", papers_dir)
    again = resolve_source("https://example.org/a/paper.pdf", papers_dir)
    other = resolve_source("https://example.org/b/paper.pdf", papers_dir)
    assert first.url == "https://example.org/a/paper.pdf"
    assert first.paper_id.startswith("paper-")
    assert first.paper_id == again.paper_id
    assert first.paper_id != other.paper_id


def test_resolve_source_accepts_local_file_and_stored_paper_id(papers_dir, sample_pdf):
    local = resolve_source(str(sample_pdf), papers_dir)
    assert local.paper_id == "sample-paper"
    assert local.path == sample_pdf

    (papers_dir / "2401.99999.pdf").write_bytes(sample_pdf.read_bytes())
    (papers_dir / "我的论文.pdf").write_bytes(sample_pdf.read_bytes())
    assert resolve_source("2401.99999", papers_dir).url is not None  # 形如 arXiv 编号的优先按 arXiv 处理
    assert resolve_source("我的论文", papers_dir).path == papers_dir / "我的论文.pdf"


def test_resolve_source_rejects_unknown_source(papers_dir):
    with pytest.raises(PaperLoadError, match="无法识别论文来源"):
        resolve_source("no-such-paper", papers_dir)


def test_slugify_keeps_chinese_and_strips_unsafe_characters():
    assert slugify("基于 RAG 的: 综述/2024?") == "基于-RAG-的-综述-2024"
    assert slugify("***") == "paper"


def test_load_paper_copies_local_pdf_and_extracts_pages_and_title(papers_dir, sample_pdf):
    paper = load_paper(str(sample_pdf), papers_dir)

    assert paper.paper_id == "sample-paper"
    assert paper.path == papers_dir / "sample-paper.pdf"
    assert paper.path.is_file()
    assert paper.title == "Citation Graph Retrieval for Science QA"
    assert len(paper.pages) == 3
    assert "contrastive retriever" in paper.pages[1]

    by_id = load_paper("sample-paper", papers_dir)
    assert by_id.pages == paper.pages


def test_title_ignores_the_larger_vertical_arxiv_stamp(papers_dir, tmp_path, make_pdf):
    pdf = make_pdf(
        tmp_path / "stamped.pdf",
        pages=["Body text of the paper."],
        title_text="Attention Is All You Need",
        side_stamp="arXiv:1706.03762v7 [cs.CL] 2 Aug 2023",
    )

    assert load_paper(str(pdf), papers_dir).title == "Attention Is All You Need"


def _pdf_with_runs(path, lines: list[list[tuple]]):
    """每行由若干 (文字, 字号) 或 (文字, 字号, 字体) 片段首尾相接排在同一条基线上。

    字体用 PDF 内置字体的简称：helv（默认）、hebo 粗体、tibo/tiit 衬线粗体/斜体、cobo 等宽粗体。
    """
    doc = pymupdf.open()
    page = doc.new_page()
    y = 100.0
    for runs in lines:
        x = 72.0
        for text, size, *font in runs:
            fontname = font[0] if font else "helv"
            page.insert_text((x, y), text, fontsize=size, fontname=fontname)
            x += pymupdf.get_text_length(text, fontname=fontname, fontsize=size)
        y += 60
    doc.save(path)
    doc.close()
    return path


def test_title_in_small_caps_is_read_as_whole_lines(papers_dir, tmp_path):
    big, small = 22, 16
    pdf = _pdf_with_runs(
        tmp_path / "smallcaps.pdf",
        [
            [("R", big), ("EACT: ", small), ("S", big), ("YNERGIZING ", small), ("R", big), ("EASONING", small)],
            [("A", big), ("ND ", small), ("A", big), ("CTING", small)],
            [("Abstract text in the normal body font size of the page.", 10)],
        ],
    )

    assert load_paper(str(pdf), papers_dir).title == "REACT: SYNERGIZING REASONING AND ACTING"


def test_title_keeps_its_second_line_when_one_word_is_set_in_a_larger_font(papers_dir, tmp_path):
    # 真实论文（arXiv 2504.19413）：产品名用了另一种字体的更大字号，标题其余部分和第二行是正常的标题字体。
    pdf = _pdf_with_runs(
        tmp_path / "mixed-title.pdf",
        [
            [("Mem0", 21.9, "cobo"), (": Building Production-Ready AI Agents with", 19.9, "hebo")],
            [("Scalable Long-Term Memory", 19.9, "hebo")],
            [("Prateek Chhikara, Dev Khant", 10, "hebo")],
            [("Abstract text in the normal body font size of the page.", 10)],
        ],
    )

    title = load_paper(str(pdf), papers_dir).title

    assert title == "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"


def test_title_excludes_author_lines_set_in_the_same_size_but_another_font(papers_dir, tmp_path):
    # 真实论文（arXiv 2601.18642）：标题只有 12 号粗体，作者行同样是 12 号，只是换成了斜体。
    pdf = _pdf_with_runs(
        tmp_path / "same-size-authors.pdf",
        [
            [("FADEMEM: BIOLOGICALLY-INSPIRED FORGETTING FOR AGENT", 12, "tibo")],
            [("MEMORY", 12, "tibo")],
            [("Lei Wei, Xiao Peng, Xu Dong, Niantao Xie", 12, "tiit")],
            [("Large language models deployed as autonomous agents.", 9, "tiro")],
        ],
    )

    title = load_paper(str(pdf), papers_dir).title

    assert title == "FADEMEM: BIOLOGICALLY-INSPIRED FORGETTING FOR AGENT MEMORY"


def test_title_ignores_a_larger_drop_cap_letter(papers_dir, tmp_path):
    # 真实的 IEEE 模板 PDF 中，首字下沉的字母在提取结果里是单独的一行。
    pdf = _pdf_with_runs(
        tmp_path / "dropcap.pdf",
        [
            [("Retrieval-Augmented Generation: A Survey", 22)],
            [("L", 30)],
            [("arge language models show impressive capabilities.", 10)],
        ],
    )

    assert load_paper(str(pdf), papers_dir).title == "Retrieval-Augmented Generation: A Survey"


def test_title_falls_back_to_file_name_when_first_page_has_no_usable_text(papers_dir, tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "notes-2024.pdf", pages=["x", "Real content is on page two."])

    assert load_paper(str(pdf), papers_dir).title == "notes-2024"


def test_load_paper_downloads_arxiv_pdf_once(papers_dir, sample_pdf, monkeypatch):
    requested = []

    def fake_get(url, **kwargs):
        requested.append(url)
        return FakeResponse(sample_pdf.read_bytes())

    monkeypatch.setattr(loader.requests, "get", fake_get)

    paper = load_paper("2401.12345", papers_dir)
    load_paper("2401.12345", papers_dir)

    assert requested == ["https://arxiv.org/pdf/2401.12345"]
    assert paper.paper_id == "2401.12345"
    assert (papers_dir / "2401.12345.pdf").is_file()


def test_download_pdf_retries_once_after_network_error(sample_pdf, monkeypatch):
    attempts = []

    def flaky_get(url, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            raise requests.ConnectionError("reset")
        return FakeResponse(sample_pdf.read_bytes())

    monkeypatch.setattr(loader.requests, "get", flaky_get)

    data = download_pdf("https://example.org/p.pdf")

    assert len(attempts) == 2
    assert data == sample_pdf.read_bytes()


def test_failed_download_reports_the_cause_and_stores_nothing(papers_dir, monkeypatch):
    attempts = []

    def failing_get(url, **kwargs):
        attempts.append(url)
        raise requests.Timeout("too slow")

    monkeypatch.setattr(loader.requests, "get", failing_get)

    with pytest.raises(PaperLoadError, match="too slow"):
        load_paper("2401.12345", papers_dir)
    assert len(attempts) == 2  # 首次加一次重试
    assert list(papers_dir.iterdir()) == []


def test_download_that_is_not_a_pdf_is_rejected_and_stores_nothing(papers_dir, monkeypatch):
    monkeypatch.setattr(loader.requests, "get", lambda url, **kwargs: FakeResponse(b"<html>login</html>"))

    with pytest.raises(PaperLoadError, match="不是 PDF"):
        load_paper("https://example.org/p.pdf", papers_dir)
    assert list(papers_dir.iterdir()) == []


def test_http_error_status_is_reported_as_download_failure(papers_dir, monkeypatch):
    monkeypatch.setattr(loader.requests, "get", lambda url, **kwargs: FakeResponse(b"", status_code=404))

    with pytest.raises(PaperLoadError, match="404"):
        load_paper("2401.12345", papers_dir)


def test_load_paper_reports_pdf_without_text(papers_dir, tmp_path):
    blank = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(blank)
    doc.close()

    with pytest.raises(PaperLoadError, match="提取不到文字"):
        load_paper(str(blank), papers_dir)
    assert list(papers_dir.iterdir()) == []


def test_invalid_local_file_is_not_copied_and_can_be_loaded_once_fixed(papers_dir, tmp_path, sample_pdf):
    source = tmp_path / "thesis.pdf"
    source.write_bytes(b"this is not a pdf")

    with pytest.raises(PaperLoadError, match="thesis.pdf 不是有效的 PDF 文件"):
        load_paper(str(source), papers_dir)
    assert list(papers_dir.iterdir()) == []

    source.write_bytes(sample_pdf.read_bytes())  # 上面失败后文件没有被占用，可以直接覆盖
    paper = load_paper(str(source), papers_dir)

    assert paper.paper_id == "thesis"
    assert len(paper.pages) == 3


def test_content_changed_tracks_whether_the_library_copy_was_updated(
    papers_dir, tmp_path, sample_pdf, make_pdf
):
    source = tmp_path / "draft.pdf"
    source.write_bytes(sample_pdf.read_bytes())

    first = load_paper(str(source), papers_dir)
    same_file_again = load_paper(str(source), papers_dir)
    by_paper_id = load_paper("draft", papers_dir)
    make_pdf(source, pages=["A rewritten single page about diffusion models."])
    after_edit = load_paper(str(source), papers_dir)

    assert first.content_changed is True
    assert same_file_again.content_changed is False
    assert by_paper_id.content_changed is False
    assert after_edit.content_changed is True
    assert len(after_edit.pages) == 1
    assert (papers_dir / "draft.pdf").read_bytes() == source.read_bytes()


def test_chunk_paper_keeps_page_numbers_and_splits_long_pages(papers_dir, sample_pdf):
    paper = load_paper(str(sample_pdf), papers_dir)
    long_page = "Sentence about transformers. " * 200
    paper = dataclasses.replace(paper, pages=[*paper.pages, long_page])

    chunks = chunk_paper(paper)

    assert {chunk.metadata["paper_id"] for chunk in chunks} == {"sample-paper"}
    assert {chunk.metadata["title"] for chunk in chunks} == {paper.title}
    assert {chunk.metadata["total_pages"] for chunk in chunks} == {4}
    assert sorted({chunk.metadata["page"] for chunk in chunks}) == [1, 2, 3, 4]
    page_four = [chunk for chunk in chunks if chunk.metadata["page"] == 4]
    assert len(page_four) > 1
    assert [chunk.metadata["chunk"] for chunk in page_four] == list(range(len(page_four)))
    assert all(len(chunk.page_content) <= loader.CHUNK_SIZE for chunk in chunks)

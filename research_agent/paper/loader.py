"""论文获取与解析：把 arXiv 编号、PDF 链接或本地 PDF 变成带页码的文本块。

约定：每篇已加载的论文都在 papers_dir 下存为 `<paper_id>.pdf`，
因此之后可以直接用 paper_id 再次引用它。
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_agent.projects import DEFAULT_PROJECT_ID

ARXIV_ID_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

DOWNLOAD_TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MIN_TITLE_LINE_CHARS = 4
SAME_SIZE_TOLERANCE = 0.5  # 字号相差不超过这个值（磅）视为同一字号；PDF 里同一字号常有 0.1 左右的误差
USER_AGENT = "research-agent/0.1 (course project)"


class PaperLoadError(Exception):
    """论文无法获取或解析，消息可直接展示给用户。"""


@dataclass(frozen=True)
class PaperSource:
    paper_id: str
    url: str | None = None
    path: Path | None = None


@dataclass(frozen=True)
class LoadedPaper:
    paper_id: str
    title: str
    path: Path
    pages: list[str]
    content_changed: bool  # 这次加载是否写入了新内容（首次加载，或文件内容与库中的不同）


def slugify(text: str, max_length: int = 60) -> str:
    """生成可作文件名的标识：保留字母、数字、中文，其余变成连字符。"""
    slug = re.sub(r"[^\w.-]+", "-", text, flags=re.UNICODE).strip("-.")
    return slug[:max_length].rstrip("-.") or "paper"


def resolve_source(source: str, papers_dir: Path) -> PaperSource:
    source = source.strip().strip("\"'")

    arxiv_match = ARXIV_ID_RE.match(source) or ARXIV_URL_RE.search(source)
    if arxiv_match:
        arxiv_id, version = arxiv_match.group(1), arxiv_match.group(2) or ""
        return PaperSource(paper_id=arxiv_id, url=f"https://arxiv.org/pdf/{arxiv_id}{version}")

    if source.lower().startswith(("http://", "https://")):
        stem = Path(urlparse(source).path).stem or "paper"
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        return PaperSource(paper_id=f"{slugify(stem, 40)}-{digest}", url=source)

    local_path = Path(source).expanduser()
    if local_path.is_file():
        return PaperSource(paper_id=slugify(local_path.stem), path=local_path)

    stored = papers_dir / f"{source}.pdf"
    if stored.is_file():
        return PaperSource(paper_id=source, path=stored)

    raise PaperLoadError(
        f"无法识别论文来源“{source}”：它不是 arXiv 编号、PDF 链接、存在的本地文件，也不是已入库的 paper_id。"
    )


def download_pdf(url: str, retries: int = 1) -> bytes:
    """下载 PDF 并返回其内容；网络错误时重试 retries 次。"""
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            response = requests.get(
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            continue
        if not response.content.startswith(b"%PDF"):
            raise PaperLoadError(f"{url} 返回的内容不是 PDF 文件。")
        return response.content
    raise PaperLoadError(f"下载 {url} 失败（{type(last_error).__name__}: {last_error}）。")


def _is_horizontal(line: dict) -> bool:
    cosine, sine = line["dir"]
    return abs(cosine - 1.0) < 0.01 and abs(sine) < 0.01


def _line_text(line: dict) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _max_size(line: dict) -> float:
    return max(span["size"] for span in line["spans"])


def _line_style(line: dict) -> tuple[str, float]:
    """一行的排版样式：行内字符最多的字体，以及该字体在这一行里的最大字号。"""
    chars_per_font: Counter[str] = Counter()
    for span in line["spans"]:
        chars_per_font[span["font"]] += len(span["text"].strip())
    font = chars_per_font.most_common(1)[0][0]
    return font, max(span["size"] for span in line["spans"] if span["font"] == font)


def _largest_font_text(page: pymupdf.Page) -> str:
    """取页面上字号最大的那几行横排文字，通常就是论文标题。

    做法：先找出字号最大的一行（并列时取最靠上的）作为标题首行，再取与它排版样式相同的行。
    几个来自真实论文的细节：
    - 只看横排文字：arXiv 会在页面左侧加一条竖排的大字号水印
      （如 “arXiv:1706.03762v7 [cs.CL] 2 Aug 2023”），不排除的话它会被当成标题。
    - 行的字号取行内最大值：ICLR 等模板的标题是小型大写，只有每个词的首字母是大字号。
    - 找标题首行时忽略过短的行：IEEE 模板正文首字下沉的单个字母比标题还大。
    - 比较样式时既看字号也看字体（见 _line_style）：有的模板标题和作者行字号相同，只是字体不同；
      有的标题里个别词（如产品名）换了字体和字号，不能因此丢掉标题的其余几行。
    """
    lines = [
        line
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        if _line_text(line) and _is_horizontal(line)
    ]
    candidates = [line for line in lines if len(_line_text(line)) >= MIN_TITLE_LINE_CHARS]
    if not candidates:
        return ""
    largest = max(_max_size(line) for line in candidates)
    first_title_line = min(
        (line for line in candidates if largest - _max_size(line) <= SAME_SIZE_TOLERANCE),
        key=lambda line: line["bbox"][1],
    )
    title_font, title_size = _line_style(first_title_line)
    # 用包围盒重新取字，而不是直接拼接各段文字：小型大写的各段文字之间不带空格。
    return " ".join(
        page.get_textbox(line["bbox"])
        for line in lines
        if _line_style(line)[0] == title_font
        and abs(_line_style(line)[1] - title_size) <= SAME_SIZE_TOLERANCE
    )


def _guess_title(doc: pymupdf.Document, fallback: str) -> str:
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if len(meta_title) > 5 and not meta_title.lower().endswith((".pdf", ".dvi", ".tex", ".doc", ".docx")):
        return meta_title
    title = " ".join(_largest_font_text(doc[0]).split()) if doc.page_count else ""
    return title[:200] if len(title) > 5 else fallback


def parse_pdf(data: bytes, name: str) -> tuple[str, list[str]]:
    """解析 PDF 内容，返回 (标题, 每页文本)。name 只用于错误提示和标题兜底。

    从内存而不是文件路径打开：PyMuPDF 打开无效文件失败后，在 Windows 上仍会占用
    文件句柄，导致之后无法删除或覆盖该文件。
    """
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
            title = _guess_title(doc, fallback=Path(name).stem)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise PaperLoadError(f"{name} 不是有效的 PDF 文件。") from exc
    if not any(page.strip() for page in pages):
        raise PaperLoadError(f"{name} 中提取不到文字，可能是扫描件，本工具不做 OCR。")
    return title, pages


def store_paper(paper_id: str, data: bytes, papers_dir: Path) -> LoadedPaper:
    """解析 PDF 内容并存为 papers_dir/<paper_id>.pdf。

    先解析、成功后才写入：坏文件不会留在论文库目录里，也不会覆盖已有的同名论文。
    """
    stored = papers_dir / f"{paper_id}.pdf"
    title, pages = parse_pdf(data, stored.name)
    content_changed = not stored.is_file() or stored.read_bytes() != data
    if content_changed:
        papers_dir.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(data)
    return LoadedPaper(
        paper_id=paper_id, title=title, path=stored, pages=pages, content_changed=content_changed
    )


def load_paper(source: str, papers_dir: Path) -> LoadedPaper:
    """获取并解析一篇论文，确保其 PDF 存放在 papers_dir/<paper_id>.pdf。"""
    resolved = resolve_source(source, papers_dir)
    stored = papers_dir / f"{resolved.paper_id}.pdf"
    if resolved.path is not None:
        data = resolved.path.read_bytes()
    elif stored.is_file():
        data = stored.read_bytes()  # 已下载过的论文不重复下载
    else:
        data = download_pdf(resolved.url)
    return store_paper(resolved.paper_id, data, papers_dir)


def chunk_paper(paper: LoadedPaper, project_id: str = DEFAULT_PROJECT_ID) -> list[Document]:
    """按页切块，每块带 paper_id、标题和页码，用于检索后标注出处。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [
        Document(
            page_content=piece,
            metadata={
                "paper_id": paper.paper_id,
                "title": paper.title,
                "page": page_number,
                "chunk": chunk_index,
                "total_pages": len(paper.pages),
                "project_id": project_id,
            },
        )
        for page_number, text in enumerate(paper.pages, start=1)
        for chunk_index, piece in enumerate(splitter.split_text(text))
    ]

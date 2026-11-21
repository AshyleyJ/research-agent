"""记录：把成果保存为 Markdown 文件。"""

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from research_agent.config import get_settings
from research_agent.paper.loader import slugify
from research_agent.projects import get_current_project_id, project_output_dir


@dataclass(frozen=True)
class NoteInfo:
    path: Path
    title: str
    size_bytes: int
    modified_at: datetime


def list_notes(output_dir: Path) -> list[NoteInfo]:
    """列出已保存的 Markdown 笔记，最近修改的排在前面。"""
    notes = []
    for path in output_dir.glob("*.md"):
        if not path.is_file():
            continue
        stat = path.stat()
        notes.append(
            NoteInfo(
                path=path,
                title=path.stem,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return sorted(notes, key=lambda note: note.modified_at, reverse=True)


def build_notes_archive(output_dir: Path) -> bytes:
    """把全部 Markdown 笔记打包成 ZIP，供浏览器一次下载。"""
    notes = list_notes(output_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for note in notes:
            archive.writestr(note.path.name, note.path.read_bytes())
    return buffer.getvalue()


def _available_path(directory: Path, stem: str) -> Path:
    """同一秒保存同名笔记时追加序号，避免静默覆盖。"""
    candidate = directory / f"{stem}.md"
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{suffix}.md"
        suffix += 1
    return candidate


def write_note(title: str, content: str, output_dir: Path) -> Path:
    """写入 `<时间戳>-<标题>.md`；正文没有一级标题时用 title 补上。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(title)}"
    path = _available_path(output_dir, stem)
    body = content.strip()
    if not body.startswith("# "):
        body = f"# {title.strip()}\n\n{body}"
    path.write_text(body + "\n", encoding="utf-8", newline="\n")
    return path


@tool
def save_note(title: str, content: str) -> str:
    """把文献综述、论文精读笔记、idea 报告等成果保存为 Markdown 文件，用户可在界面侧边栏下载。

    Args:
        title: 笔记标题，会用作文件名的一部分。
        content: 完整的 Markdown 正文。
    """
    path = write_note(
        title, content, project_output_dir(get_settings(), get_current_project_id())
    )
    return f"已保存到 outputs/{path.name}"

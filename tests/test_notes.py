import re
import os
import io
import zipfile
from types import SimpleNamespace

from research_agent.tools import notes
from research_agent.projects import reset_current_project, set_current_project


def test_write_note_adds_title_heading_and_uses_timestamped_safe_filename(tmp_path):
    path = notes.write_note("RAG 综述: 2024/上半年", "正文第一段。", tmp_path / "outputs")

    assert path.parent == tmp_path / "outputs"
    assert re.fullmatch(r"\d{8}-\d{6}-RAG-综述-2024-上半年\.md", path.name)
    assert path.read_text(encoding="utf-8") == "# RAG 综述: 2024/上半年\n\n正文第一段。\n"


def test_write_note_keeps_existing_top_level_heading(tmp_path):
    path = notes.write_note("标题", "\n# 自带标题\n\n正文\n", tmp_path)

    assert path.read_text(encoding="utf-8") == "# 自带标题\n\n正文\n"


def test_save_note_tool_writes_into_output_dir_and_reports_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "get_settings", lambda: SimpleNamespace(output_dir=tmp_path))

    message = notes.save_note.invoke({"title": "idea 报告", "content": "内容"})

    saved = list(tmp_path.glob("*.md"))
    assert len(saved) == 1
    assert message == f"已保存到 outputs/{saved[0].name}"
    assert saved[0].read_text(encoding="utf-8") == "# idea 报告\n\n内容\n"


def test_write_note_does_not_overwrite_an_existing_note_with_the_same_timestamp(
    tmp_path, monkeypatch
):
    class FixedDatetime(notes.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 21, 15, 0, 0)

    monkeypatch.setattr(notes, "datetime", FixedDatetime)

    first = notes.write_note("同名笔记", "第一版", tmp_path)
    second = notes.write_note("同名笔记", "第二版", tmp_path)

    assert first.name == "20261121-150000-同名笔记.md"
    assert second.name == "20261121-150000-同名笔记-2.md"
    assert first.read_text(encoding="utf-8").endswith("第一版\n")
    assert second.read_text(encoding="utf-8").endswith("第二版\n")


def test_list_notes_returns_markdown_metadata_newest_first(tmp_path):
    older = tmp_path / "older.md"
    newer = tmp_path / "newer.md"
    ignored = tmp_path / "ignore.txt"
    older.write_text("# older\n", encoding="utf-8", newline="\n")
    newer.write_text("# newer\n\nbody\n", encoding="utf-8", newline="\n")
    ignored.write_text("not a note", encoding="utf-8")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    result = notes.list_notes(tmp_path)

    assert [(note.path.name, note.title, note.size_bytes) for note in result] == [
        ("newer.md", "newer", len("# newer\n\nbody\n".encode("utf-8"))),
        ("older.md", "older", len("# older\n".encode("utf-8"))),
    ]
    assert result[0].modified_at > result[1].modified_at


def test_build_notes_archive_contains_every_markdown_note(tmp_path):
    notes.write_note("笔记一", "正文一", tmp_path)
    notes.write_note("笔记二", "正文二", tmp_path)

    data = notes.build_notes_archive(tmp_path)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        contents = [archive.read(name).decode("utf-8") for name in names]
    assert len(names) == 2
    assert all(name.endswith(".md") for name in names)
    assert any("正文一" in content for content in contents)
    assert any("正文二" in content for content in contents)


def test_save_note_tool_writes_into_the_current_project(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "data",
    )
    monkeypatch.setattr(notes, "get_settings", lambda: settings)
    token = set_current_project("project-a")
    try:
        message = notes.save_note.invoke({"title": "项目笔记", "content": "正文"})
    finally:
        reset_current_project(token)

    path = tmp_path / "outputs" / "projects" / "project-a" / "项目笔记.md"
    saved = list(path.parent.glob("*.md"))
    assert len(saved) == 1
    assert message == f"已保存到 outputs/{saved[0].name}"

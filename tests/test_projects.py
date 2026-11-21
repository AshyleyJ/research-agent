from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent import projects
from research_agent.config import Settings
from research_agent.projects import (
    DEFAULT_PROJECT_ID,
    ProjectCatalog,
    get_current_project_id,
    make_thread_id,
    project_output_dir,
    project_papers_dir,
    reset_current_project,
    set_current_project,
    thread_belongs_to_project,
)


def test_catalog_creates_default_project_and_persists_new_projects(tmp_path):
    catalog = ProjectCatalog(tmp_path)

    initial = catalog.list_projects()
    created = catalog.create("长文档 RAG 评测")
    reopened = ProjectCatalog(tmp_path).list_projects()

    assert [project.project_id for project in initial] == [DEFAULT_PROJECT_ID]
    assert initial[0].name == "默认项目"
    assert created.name == "长文档 RAG 评测"
    assert [(project.project_id, project.name) for project in reopened] == [
        (DEFAULT_PROJECT_ID, "默认项目"),
        (created.project_id, "长文档 RAG 评测"),
    ]


def test_catalog_rejects_empty_and_duplicate_names(tmp_path):
    catalog = ProjectCatalog(tmp_path)
    catalog.create("记忆机制")

    with pytest.raises(ValueError, match="不能为空"):
        catalog.create("  ")
    with pytest.raises(ValueError, match="已经存在"):
        catalog.create("记忆机制")


def test_project_directories_and_threads_are_isolated(tmp_path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
    )

    default_id = make_thread_id(DEFAULT_PROJECT_ID)
    other_id = make_thread_id("project-a")

    assert project_papers_dir(settings, DEFAULT_PROJECT_ID) == settings.papers_dir
    assert project_output_dir(settings, DEFAULT_PROJECT_ID) == settings.output_dir
    assert project_papers_dir(settings, "project-a") == settings.data_dir / "projects" / "project-a" / "papers"
    assert project_output_dir(settings, "project-a") == settings.output_dir / "projects" / "project-a"
    assert thread_belongs_to_project("legacy-thread", DEFAULT_PROJECT_ID)
    assert thread_belongs_to_project(default_id, DEFAULT_PROJECT_ID)
    assert thread_belongs_to_project(other_id, "project-a")
    assert not thread_belongs_to_project(other_id, DEFAULT_PROJECT_ID)


def test_current_project_context_is_scoped_to_caller(tmp_path):
    token = set_current_project("project-a")
    try:
        assert get_current_project_id() == "project-a"
    finally:
        reset_current_project(token)

    assert get_current_project_id() == DEFAULT_PROJECT_ID

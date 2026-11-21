"""研究项目：项目清单、当前项目上下文和项目级目录。"""

import json
import os
import threading
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_agent.config import Settings

DEFAULT_PROJECT_ID = "default"
DEFAULT_PROJECT_NAME = "默认项目"
PROJECT_NAME_MAX_CHARS = 40

_CURRENT_PROJECT: ContextVar[str] = ContextVar("current_project_id", default=DEFAULT_PROJECT_ID)
_CATALOG_LOCK = threading.RLock()


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(
            project_id=str(data["project_id"]),
            name=str(data["name"]),
            created_at=str(data.get("created_at", "")),
        )


class ProjectCatalog:
    """保存在 data/projects.json 的轻量项目清单。"""

    def __init__(self, data_dir: Path):
        self.path = data_dir / "projects.json"

    def list_projects(self) -> list[Project]:
        with _CATALOG_LOCK:
            projects = self._read()
            if not any(project.project_id == DEFAULT_PROJECT_ID for project in projects):
                projects.insert(0, self._default_project())
                self._write(projects)
            return projects

    def get(self, project_id: str) -> Project | None:
        return next((project for project in self.list_projects() if project.project_id == project_id), None)

    def create(self, name: str) -> Project:
        clean_name = " ".join(name.split())[:PROJECT_NAME_MAX_CHARS]
        if not clean_name:
            raise ValueError("项目名称不能为空。")
        with _CATALOG_LOCK:
            projects = self.list_projects()
            if any(project.name.casefold() == clean_name.casefold() for project in projects):
                raise ValueError(f"已经存在名为“{clean_name}”的项目。")
            project = Project(
                project_id=uuid.uuid4().hex[:12],
                name=clean_name,
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            self._write([*projects, project])
            return project

    def _read(self) -> list[Project]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            projects = [Project.from_dict(item) for item in data.get("projects", [])]
        except (OSError, ValueError, TypeError, KeyError):
            projects = []
        return projects

    def _write(self, projects: list[Project]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        payload = {"projects": [project.to_dict() for project in projects]}
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.path)

    @staticmethod
    def _default_project() -> Project:
        return Project(
            project_id=DEFAULT_PROJECT_ID,
            name=DEFAULT_PROJECT_NAME,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )


def set_current_project(project_id: str) -> Token:
    return _CURRENT_PROJECT.set(project_id or DEFAULT_PROJECT_ID)


def reset_current_project(token: Token) -> None:
    _CURRENT_PROJECT.reset(token)


def get_current_project_id() -> str:
    return _CURRENT_PROJECT.get() or DEFAULT_PROJECT_ID


def project_papers_dir(settings: "Settings", project_id: str | None = None) -> Path:
    project_id = project_id or get_current_project_id()
    if project_id == DEFAULT_PROJECT_ID:
        return settings.papers_dir
    return settings.data_dir / "projects" / project_id / "papers"


def project_output_dir(settings: "Settings", project_id: str | None = None) -> Path:
    project_id = project_id or get_current_project_id()
    if project_id == DEFAULT_PROJECT_ID:
        return settings.output_dir
    return settings.output_dir / "projects" / project_id


def thread_prefix(project_id: str) -> str:
    return f"{project_id}:"


def make_thread_id(project_id: str) -> str:
    return f"{thread_prefix(project_id)}{uuid.uuid4().hex[:12]}"


def thread_belongs_to_project(thread_id: str, project_id: str) -> bool:
    if thread_id.startswith(thread_prefix(project_id)):
        return True
    # 升级前创建的会话没有项目前缀，统一归入默认项目。
    return project_id == DEFAULT_PROJECT_ID and ":" not in thread_id

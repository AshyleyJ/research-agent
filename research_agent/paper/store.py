"""论文向量库：Chroma 持久化存储 + 本地 bge-m3 向量。"""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from research_agent.config import get_settings
from research_agent.projects import DEFAULT_PROJECT_ID, get_current_project_id

COLLECTION_NAME = "papers"
# Chroma 在向量数达到 sync_threshold（默认 1000）时把 HNSW 索引写成磁盘文件，而底层 hnswlib
# 在 Windows 上写不进含非 ASCII 字符（例如中文）的路径，且不报错，之后整个库都无法读取。
# 把阈值设到达不到的大小后，向量只保存在 chroma.sqlite3 里，索引在启动时从中重建
# （实测 6000 条 1024 维向量重建不到 1 秒），不再依赖那些磁盘文件。
HNSW_SYNC_THRESHOLD = 1_000_000_000
_STORE_INIT_LOCK = threading.Lock()
_STORE: "PaperStore | None" = None


@dataclass(frozen=True)
class PaperInfo:
    paper_id: str
    title: str
    total_pages: int
    chunks: int


class PaperStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings):
        self._db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_configuration={"hnsw": {"sync_threshold": HNSW_SYNC_THRESHOLD}},
        )
        self._migrate_legacy_metadata()

    def _migrate_legacy_metadata(self) -> None:
        """给升级前没有 project_id 的向量块补上默认项目，避免旧库不可见。"""
        data = self._db.get(include=["metadatas"])
        ids = data["ids"]
        metadatas = data["metadatas"] or []
        migrate_ids = []
        migrate_metadatas = []
        for item_id, metadata in zip(ids, metadatas):
            metadata = dict(metadata or {})
            if "project_id" not in metadata:
                metadata["project_id"] = DEFAULT_PROJECT_ID
                migrate_ids.append(item_id)
                migrate_metadatas.append(metadata)
        if migrate_ids:
            self._db._collection.update(ids=migrate_ids, metadatas=migrate_metadatas)

    @staticmethod
    def _project_filter(project_id: str, paper_id: str | None = None) -> dict:
        conditions = [{"project_id": project_id}]
        if paper_id is not None:
            conditions.append({"paper_id": paper_id})
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def has_paper(self, paper_id: str, project_id: str | None = None) -> bool:
        project_id = project_id or get_current_project_id()
        return bool(
            self._db.get(where=self._project_filter(project_id, paper_id), limit=1)["ids"]
        )

    def add_paper(self, chunks: list[Document], project_id: str | None = None) -> int:
        """写入一篇论文的全部文本块；同一 paper_id 的旧内容会先被替换掉。"""
        if not chunks:
            return 0
        project_id = project_id or get_current_project_id()
        paper_id = chunks[0].metadata["paper_id"]
        self.delete_paper(paper_id, project_id)
        normalized = [
            doc.model_copy(update={"metadata": {**doc.metadata, "project_id": project_id}})
            for doc in chunks
        ]
        ids = [
            f"{project_id}:{paper_id}-p{doc.metadata['page']}-c{doc.metadata['chunk']}"
            for doc in normalized
        ]
        self._db.add_documents(normalized, ids=ids)
        return len(normalized)

    def delete_paper(self, paper_id: str, project_id: str | None = None) -> None:
        project_id = project_id or get_current_project_id()
        existing = self._db.get(where=self._project_filter(project_id, paper_id))["ids"]
        if existing:
            self._db.delete(ids=existing)

    def list_papers(self, project_id: str | None = None) -> list[PaperInfo]:
        project_id = project_id or get_current_project_id()
        papers: dict[str, PaperInfo] = {}
        for meta in self._db.get(
            where=self._project_filter(project_id), include=["metadatas"]
        )["metadatas"]:
            current = papers.get(meta["paper_id"])
            papers[meta["paper_id"]] = PaperInfo(
                paper_id=meta["paper_id"],
                title=meta["title"],
                total_pages=meta["total_pages"],
                chunks=(current.chunks if current else 0) + 1,
            )
        return sorted(papers.values(), key=lambda info: info.paper_id)

    def search(
        self,
        query: str,
        k: int = 5,
        paper_id: str | None = None,
        project_id: str | None = None,
    ) -> list[Document]:
        project_id = project_id or get_current_project_id()
        return self._db.similarity_search(
            query, k=k, filter=self._project_filter(project_id, paper_id)
        )

    def paper_documents(self, paper_id: str, project_id: str | None = None) -> list[Document]:
        project_id = project_id or get_current_project_id()
        data = self._db.get(
            where=self._project_filter(project_id, paper_id),
            include=["documents", "metadatas"],
        )
        documents = data["documents"] or []
        metadatas = data["metadatas"] or []
        docs = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(documents, metadatas)
        ]
        return sorted(
            docs,
            key=lambda doc: (doc.metadata.get("page", 0), doc.metadata.get("chunk", 0)),
        )


class LazyEmbeddings(Embeddings):
    """第一次真正计算向量时才加载模型。

    浏览论文列表等操作用不到向量，这样界面启动时不必等待模型加载（首次运行还要下载）。
    """

    def __init__(self, factory: Callable[[], Embeddings]):
        self._factory = factory
        self._loaded: Embeddings | None = None
        self._load_lock = threading.Lock()

    def _model(self) -> Embeddings:
        # 智能体会并行执行多个工具调用；加锁保证它们同时第一次用到模型时只加载一份。
        with self._load_lock:
            if self._loaded is None:
                self._loaded = self._factory()
            return self._loaded

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model().embed_query(text)


def load_local_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.hf_endpoint:
        # 必须在导入 huggingface 相关库之前设置才生效，所以下面的导入放在函数内。
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_store() -> PaperStore:
    global _STORE
    if _STORE is None:
        with _STORE_INIT_LOCK:
            if _STORE is None:
                _STORE = PaperStore(
                    get_settings().chroma_dir, LazyEmbeddings(load_local_embeddings)
                )
    return _STORE

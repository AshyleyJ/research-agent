import time
from concurrent.futures import ThreadPoolExecutor

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.documents import Document

from research_agent.paper.loader import chunk_paper, load_paper
from research_agent.paper.store import LazyEmbeddings, PaperStore


def _chunk(paper_id: str, page: int, text: str, title: str = "T", total_pages: int = 2) -> Document:
    return Document(
        page_content=text,
        metadata={"paper_id": paper_id, "title": title, "page": page, "chunk": 0, "total_pages": total_pages},
    )


def test_ingested_pdf_can_be_retrieved_with_page_metadata(store, papers_dir, sample_pdf):
    paper = load_paper(str(sample_pdf), papers_dir)

    added = store.add_paper(chunk_paper(paper))
    hits = store.search("what learning rate and accuracy on the SciQA benchmark", k=1)

    assert added == 3
    assert store.has_paper("sample-paper")
    assert hits[0].metadata["page"] == 3
    assert hits[0].metadata["paper_id"] == "sample-paper"
    assert "71.3" in hits[0].page_content


def test_search_can_be_restricted_to_one_paper(store):
    store.add_paper([_chunk("a", 1, "graph neural networks for molecules")])
    store.add_paper([_chunk("b", 1, "graph neural networks for traffic")])

    hits = store.search("graph neural networks", k=5, paper_id="b")

    assert [hit.metadata["paper_id"] for hit in hits] == ["b"]


def test_papers_with_the_same_id_are_isolated_between_projects(store):
    store.add_paper([_chunk("shared", 1, "alpha project content", "A")], project_id="a")
    store.add_paper([_chunk("shared", 1, "beta project content", "B")], project_id="b")

    assert [(paper.paper_id, paper.title) for paper in store.list_papers("a")] == [
        ("shared", "A")
    ]
    assert [(paper.paper_id, paper.title) for paper in store.list_papers("b")] == [
        ("shared", "B")
    ]
    assert "alpha" in store.search("project content", k=1, project_id="a")[0].page_content
    assert "beta" in store.search("project content", k=1, project_id="b")[0].page_content
    assert len(store.paper_documents("shared", "a")) == 1


def test_paper_documents_are_ordered_by_page_and_chunk(store):
    store.add_paper(
        [
            _chunk("p", 2, "page two", "Paper"),
            _chunk("p", 1, "page one", "Paper"),
        ],
        project_id="a",
    )

    docs = store.paper_documents("p", "a")

    assert [doc.metadata["page"] for doc in docs] == [1, 2]


def test_list_papers_aggregates_chunks_per_paper(store):
    assert store.list_papers() == []

    store.add_paper([_chunk("b", 1, "one", title="Paper B"), _chunk("b", 2, "two", title="Paper B")])
    store.add_paper([_chunk("a", 1, "three", title="Paper A", total_pages=9)])

    papers = store.list_papers()

    assert [(p.paper_id, p.title, p.total_pages, p.chunks) for p in papers] == [
        ("a", "Paper A", 9, 1),
        ("b", "Paper B", 2, 2),
    ]


def test_re_adding_a_paper_replaces_its_old_chunks(store):
    store.add_paper([_chunk("a", 1, "old text one"), _chunk("a", 2, "old text two")])
    store.add_paper([_chunk("a", 1, "new text")])

    assert [p.chunks for p in store.list_papers()] == [1]
    assert store.search("text", k=5)[0].page_content == "new text"


def test_has_paper_is_false_for_unknown_id_and_after_delete(store):
    assert not store.has_paper("a")
    store.add_paper([_chunk("a", 1, "text")])
    store.delete_paper("a")
    assert not store.has_paper("a")


def test_embedding_model_is_loaded_only_when_vectors_are_needed(tmp_path, embeddings):
    loads = []

    def factory():
        loads.append(1)
        return embeddings

    lazy_store = PaperStore(tmp_path / "lazy", LazyEmbeddings(factory))

    assert lazy_store.list_papers() == []
    assert not lazy_store.has_paper("a")
    assert loads == []

    lazy_store.add_paper([_chunk("a", 1, "contrastive retriever")])
    hits = lazy_store.search("contrastive retriever", k=1)

    assert loads == [1]  # 之后的调用复用同一个模型
    assert hits[0].metadata["paper_id"] == "a"


def test_embedding_model_is_loaded_once_when_first_used_from_several_threads(embeddings):
    """智能体会并行执行多个工具调用，它们可能同时第一次用到向量模型。"""
    loads = []

    def slow_factory():
        loads.append(1)
        time.sleep(0.2)  # 模型加载很慢，其他线程会在这期间进来
        return embeddings

    lazy = LazyEmbeddings(slow_factory)

    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(lazy.embed_query, ["a", "b", "c", "d"]))

    assert loads == [1]
    assert len(vectors) == 4


def test_papers_persist_across_store_instances(tmp_path, embeddings):
    first = PaperStore(tmp_path / "persist", embeddings)
    first.add_paper([_chunk("a", 1, "persistent text")])

    second = PaperStore(tmp_path / "persist", embeddings)

    assert second.has_paper("a")


def test_large_library_under_a_non_ascii_path_survives_a_restart(tmp_path, embeddings):
    """回归：Chroma 把 HNSW 索引落盘时写不进含中文的路径（Windows），重启后整个库不可用。

    落盘在向量数超过 1000 时触发，所以要写入足够多的文本块才能复现。
    """
    persist_dir = tmp_path / "科研智能体" / "chroma"
    first = PaperStore(persist_dir, embeddings)
    for paper in range(11):
        first.add_paper(
            [_chunk(f"paper-{paper}", page, f"topic{paper} detail{page}", total_pages=100) for page in range(1, 101)]
        )
    del first
    SharedSystemClient.clear_system_cache()  # 丢弃内存中的索引，相当于重启程序

    reopened = PaperStore(persist_dir, embeddings)

    assert len(reopened.list_papers()) == 11
    hit = reopened.search("topic7 detail42", k=1)[0]
    assert (hit.metadata["paper_id"], hit.metadata["page"]) == ("paper-7", 42)

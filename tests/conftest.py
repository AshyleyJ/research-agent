import hashlib
import json
import math
import re
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ScriptedChatModel(BaseChatModel):
    """按顺序返回预设回复，并记录每次收到的消息，用来替代真实模型。"""

    responses: list
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next_reply(self, messages) -> AIMessage:
        self.calls.append(messages)
        if len(self.calls) > len(self.responses):
            raise AssertionError(f"模型被调用了 {len(self.calls)} 次，超出预设的 {len(self.responses)} 条回复")
        reply = self.responses[len(self.calls) - 1]
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next_reply(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """像真实模型一样分块输出：文字每 4 个字符一块，工具调用作为一个分块。"""
        reply = self._next_reply(messages)
        if reply.tool_calls:
            pieces = [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": call["name"], "args": json.dumps(call["args"]), "id": call["id"], "index": index}
                        for index, call in enumerate(reply.tool_calls)
                    ],
                )
            ]
        else:
            pieces = [
                AIMessageChunk(content=reply.content[start : start + 4])
                for start in range(0, len(reply.content), 4)
            ]
        for piece in pieces:
            chunk = ChatGenerationChunk(message=piece)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def bind_tools(self, tools, **kwargs):
        return self

    def prompt_text(self, call_index: int) -> str:
        return "\n".join(str(message.content) for message in self.calls[call_index])


class BagOfWordsEmbeddings(Embeddings):
    """把词哈希到固定维度的词袋向量：无需下载模型，且共享词越多的文本越相近。"""

    DIMENSIONS = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.DIMENSIONS
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def write_pdf(
    path: Path, pages: list[str], title_text: str | None = None, side_stamp: str | None = None
) -> Path:
    """生成一份多页 PDF；title_text 以大字号写在第一页顶部，
    side_stamp 以更大的字号竖排写在第一页左侧（模拟 arXiv 水印）。"""
    doc = pymupdf.open()
    for index, body in enumerate(pages):
        page = doc.new_page()
        top = 72
        if index == 0 and side_stamp:
            page.insert_text((40, 600), side_stamp, fontsize=28, rotate=90)
        if index == 0 and title_text:
            page.insert_text((72, top), title_text, fontsize=22)
            top += 40
        page.insert_textbox(pymupdf.Rect(72, top, 540, 780), body, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def make_pdf():
    return write_pdf


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    return write_pdf(
        tmp_path / "source" / "sample paper.pdf",
        pages=[
            "Abstract. We study retrieval augmented generation for scientific question answering.",
            "Method. Our approach uses a contrastive retriever trained on citation graphs.",
            "Experiments. On the SciQA benchmark the model reaches 71.3 accuracy with a learning rate of 0.0003.",
        ],
        title_text="Citation Graph Retrieval for Science QA",
    )


@pytest.fixture
def papers_dir(tmp_path) -> Path:
    directory = tmp_path / "papers"
    directory.mkdir()
    return directory


@pytest.fixture
def embeddings() -> Embeddings:
    return BagOfWordsEmbeddings()


@pytest.fixture
def scripted_model():
    return ScriptedChatModel


@pytest.fixture
def store(tmp_path, embeddings):
    from research_agent.paper.store import PaperStore

    return PaperStore(tmp_path / "chroma", embeddings)

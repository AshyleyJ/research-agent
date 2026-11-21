"""Streamlit 界面。启动：streamlit run app.py"""

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from research_agent.agent import (
    AnswerText,
    ToolFinished,
    ToolStarted,
    ToolStep,
    build_agent,
    delete_conversation,
    history_to_turns,
    list_conversations,
    open_checkpointer,
    stream_turn,
)
from research_agent.config import Settings, get_settings
from research_agent.evidence import (
    EvidenceMatrixResult,
    build_evidence_matrix,
    save_evidence_csv,
)
from research_agent.llm import get_chat_model
from research_agent.paper.loader import PaperLoadError
from research_agent.paper.store import get_store
from research_agent.projects import (
    DEFAULT_PROJECT_ID,
    Project,
    ProjectCatalog,
    make_thread_id,
    project_output_dir,
    project_papers_dir,
    reset_current_project,
    set_current_project,
    thread_belongs_to_project,
)
from research_agent.tools.notes import build_notes_archive, list_notes
from research_agent.tools.paper import ingest_uploaded_pdf

RESULT_PREVIEW_CHARS = 4000
MAX_NOTES_SHOWN = 15
MAX_EVIDENCE_FILES_SHOWN = 5
PENDING_DELETE_KEY = "pending_delete"  # session_state 里记录“已点垃圾桶、等待确认”的会话编号
EVIDENCE_STATE_KEY = "evidence-matrix-state"
PROJECT_STATE_KEY = "current-project-id"
PROJECT_SWITCH_KEY = "project-switch-pending"
PROJECT_ERROR_KEY = "project-create-error"
TOOL_LABELS = {
    "research_literature": "完整文献调研",
    "search_arxiv": "检索 arXiv",
    "search_web": "搜索网页",
    "read_paper": "精读论文",
    "ask_paper": "论文问答",
    "compare_papers": "多论文对比",
    "build_evidence_matrix": "论文证据矩阵",
    "list_papers": "查看论文库",
    "generate_ideas": "生成研究 idea",
    "save_note": "保存笔记",
}
SUGGESTIONS = [
    (
        ":material/search:",
        "调研一个新方向",
        "多路检索文献，写出带编号引用的综述。",
        "调研 LLM 智能体记忆机制方向近两年的代表性工作，写一份简短综述并保存成笔记",
    ),
    (
        ":material/article:",
        "精读一篇论文",
        "阅读全文并提取方法、实验、结论和局限。",
        "精读 arXiv 论文 1706.03762，然后告诉我 base 模型用了几个注意力头、训练用的什么优化器",
    ),
    (
        ":material/lightbulb:",
        "生成研究 idea",
        "结合最新文献，提出可执行且经过评审的选题。",
        "围绕“面向长文档问答的检索增强生成”给我几个适合硕士生做的研究 idea",
    ),
]


@dataclass(frozen=True)
class WorkspaceSnapshot:
    papers: list
    conversations: int
    notes: int
    evidence_files: int


@st.cache_resource
def load_agent():
    return build_agent(open_checkpointer(get_settings().checkpoint_path))


def new_thread_id(project_id: str) -> str:
    return make_thread_id(project_id)


def current_thread_id(project_id: str) -> str:
    """会话编号放在网址参数里，刷新页面后仍能从 SQLite 里取回这次对话。"""
    thread_id = st.query_params.get("thread", "")
    if not thread_id or not thread_belongs_to_project(thread_id, project_id):
        thread_id = new_thread_id(project_id)
        st.query_params["thread"] = thread_id
    return thread_id


def switch_project(project_id: str) -> None:
    st.session_state[PROJECT_STATE_KEY] = project_id
    st.session_state[PROJECT_SWITCH_KEY] = True
    st.query_params["project"] = project_id
    st.query_params["thread"] = new_thread_id(project_id)
    st.session_state.pop(PENDING_DELETE_KEY, None)
    st.session_state.pop("queued_prompt", None)


def handle_project_selection() -> None:
    switch_project(st.session_state["project-selector"])


def handle_project_creation(catalog: ProjectCatalog) -> None:
    try:
        created = catalog.create(st.session_state.get("new-project-name", ""))
    except ValueError as exc:
        st.session_state[PROJECT_ERROR_KEY] = str(exc)
    else:
        st.session_state.pop(PROJECT_ERROR_KEY, None)
        switch_project(created.project_id)


def current_project(catalog: ProjectCatalog) -> Project:
    projects = catalog.list_projects()
    requested = st.query_params.get("project", DEFAULT_PROJECT_ID)
    if st.session_state.pop(PROJECT_SWITCH_KEY, False):
        requested = st.session_state.get(PROJECT_STATE_KEY, requested)
    selected = next(
        (project for project in projects if project.project_id == requested),
        next(project for project in projects if project.project_id == DEFAULT_PROJECT_ID),
    )
    st.session_state[PROJECT_STATE_KEY] = selected.project_id
    if st.query_params.get("project") != selected.project_id:
        st.query_params["project"] = selected.project_id
    return selected


def _step_title(tool_name: str) -> str:
    return f"{TOOL_LABELS.get(tool_name, tool_name)} · {tool_name}"


def render_step(step: ToolStep) -> None:
    finished = step.result is not None and not step.failed
    with st.status(
        _step_title(step.name),
        type="step",
        state="complete" if finished else "error",
        expanded=False,
    ):
        st.caption("参数")
        st.json(step.args)
        st.caption("结果")
        st.text(_preview(step.result) if step.result is not None else "这一步没有执行完。")


def _preview(text: str) -> str:
    if len(text) <= RESULT_PREVIEW_CHARS:
        return text
    return text[:RESULT_PREVIEW_CHARS] + f"\n…（共 {len(text)} 字符，此处只显示开头）"


def add_uploaded_paper(uploaded, settings: Settings, project_id: str) -> None:
    try:
        with st.spinner("正在解析并写入论文库（首次使用要加载向量模型，请稍候）…"):
            paper = ingest_uploaded_pdf(
                uploaded.name,
                uploaded.getvalue(),
                get_store(),
                project_papers_dir(settings, project_id),
                project_id=project_id,
            )
    except PaperLoadError as exc:
        st.error(str(exc))
    else:
        st.success(f"已入库：{paper.title}（paper_id: {paper.paper_id}）")


def open_thread(thread_id: str) -> None:
    st.query_params["thread"] = thread_id
    st.rerun()


def render_conversations(agent, current_thread: str, project_id: str) -> int:
    """历史会话列表：点标题切换过去；点垃圾桶后再确认一次才删除。还没提过问的新会话不在列表里。"""
    conversations = list_conversations(agent, project_id)
    st.subheader(f"历史会话（{len(conversations)} 个）")
    if not conversations:
        st.caption("提问之后，会话会保存在这里，随时可以回来接着聊。")
    for conversation in conversations:
        thread_id = conversation.thread_id
        title_column, delete_column = st.columns([5, 1], vertical_alignment="center")
        if title_column.button(
            conversation.title,
            key=f"open-{thread_id}",
            type="primary" if thread_id == current_thread else "secondary",
            help=f"最后活动：{conversation.updated_at:%m-%d %H:%M}",
            width="stretch",
        ):
            open_thread(thread_id)
        if delete_column.button(
            "", icon=":material/delete:", key=f"delete-{thread_id}", type="tertiary", help="删除这个会话"
        ):
            st.session_state[PENDING_DELETE_KEY] = thread_id
        if st.session_state.get(PENDING_DELETE_KEY) == thread_id:
            st.caption("删除后这个会话的对话记录无法恢复。")
            confirm_column, cancel_column = st.columns(2)
            if confirm_column.button(
                "确认删除", key=f"confirm-delete-{thread_id}", type="primary", width="stretch"
            ):
                del st.session_state[PENDING_DELETE_KEY]
                delete_conversation(agent, thread_id)
                # 删掉的是正在看的会话时，换到一个新的空白会话。
                open_thread(
                    new_thread_id(project_id)
                    if thread_id == current_thread
                    else current_thread
                )
            if cancel_column.button("取消", key=f"cancel-delete-{thread_id}", width="stretch"):
                del st.session_state[PENDING_DELETE_KEY]
                st.rerun()
    return len(conversations)


def render_sidebar(
    settings: Settings,
    agent,
    current_thread: str,
    project: Project,
    catalog: ProjectCatalog,
) -> WorkspaceSnapshot:
    """agent 为 None 表示还没配置 API Key，此时不显示会话相关的内容。"""
    with st.sidebar:
        st.header("科研智能体", icon=":material/science:")
        st.caption("项目化科研工作台")
        projects = catalog.list_projects()
        project_names = {item.project_id: item.name for item in projects}
        st.selectbox(
            "研究项目",
            options=[item.project_id for item in projects],
            index=[item.project_id for item in projects].index(project.project_id),
            format_func=lambda project_id: project_names[project_id],
            key="project-selector",
            on_change=handle_project_selection,
        )

        with st.expander("新建研究项目", icon=":material/create_new_folder:"):
            project_name = st.text_input(
                "项目名称", placeholder="例如：长文档 RAG 评测", key="new-project-name"
            )
            st.button(
                "创建并切换",
                key="create-project",
                icon=":material/add:",
                width="stretch",
                on_click=handle_project_creation,
                args=(catalog,),
            )
        if error := st.session_state.pop(PROJECT_ERROR_KEY, None):
            st.error(error)

        conversation_count = 0
        if agent is not None:
            if st.button(
                "新建会话",
                key="new-conversation",
                icon=":material/add_comment:",
                type="primary",
                width="stretch",
            ):
                open_thread(new_thread_id(project.project_id))
            conversation_count = render_conversations(
                agent, current_thread, project.project_id
            )

        with st.expander("添加论文", icon=":material/upload_file:"):
            st.subheader("上传论文")
            with st.form("upload-form", clear_on_submit=True, border=False):
                uploaded = st.file_uploader(
                    "PDF 文件", type="pdf", label_visibility="collapsed"
                )
                submitted = st.form_submit_button(
                    "加入论文库", icon=":material/add:", width="stretch"
                )
            if submitted and uploaded is not None:
                add_uploaded_paper(uploaded, settings, project.project_id)

        papers = get_store().list_papers(project.project_id)
        with st.expander(
            f"论文库（{len(papers)} 篇）",
            icon=":material/library_books:",
        ):
            st.subheader(f"论文库（{len(papers)} 篇）")
            if not papers:
                st.caption("还没有论文。可以上传 PDF，或让智能体按 arXiv 编号读入。")
            for info in papers:
                with st.container(border=True, gap="small"):
                    st.markdown(f"**{info.title}**")
                    st.caption(f"`{info.paper_id}` · {info.total_pages} 页")

        output_dir = project_output_dir(settings, project.project_id)
        notes = list_notes(output_dir)
        evidence_files = sorted(
            output_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        with st.expander(
            f"研究产出（{len(notes) + len(evidence_files)}）",
            icon=":material/folder_open:",
        ):
            st.subheader(f"已保存的笔记（{len(notes)} 份）")
            if not notes:
                st.caption("智能体保存的综述、精读笔记和 idea 报告会出现在这里。")
            for note in notes[:MAX_NOTES_SHOWN]:
                st.download_button(
                    note.path.name,
                    data=note.path.read_bytes(),
                    file_name=note.path.name,
                    mime="text/markdown",
                    key=f"download-{note.path.name}",
                    width="stretch",
                )
            if notes:
                st.download_button(
                    "下载全部笔记",
                    data=build_notes_archive(output_dir),
                    file_name="科研智能体笔记.zip",
                    mime="application/zip",
                    key="download-all-notes",
                    width="stretch",
                    icon=":material/folder_zip:",
                )

            st.subheader(f"证据矩阵（{len(evidence_files)} 份）")
            for path in evidence_files[:MAX_EVIDENCE_FILES_SHOWN]:
                st.download_button(
                    path.name,
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/csv",
                    key=f"download-evidence-{path.name}",
                    width="stretch",
                    icon=":material/table_view:",
                )
    return WorkspaceSnapshot(
        papers=papers,
        conversations=conversation_count,
        notes=len(notes),
        evidence_files=len(evidence_files),
    )


def render_workspace_summary(project: Project, snapshot: WorkspaceSnapshot) -> None:
    st.markdown(f"### {project.name}")
    st.caption("当前项目的资料、对话与研究产出")
    with st.grid("auto", min_column_width=150, gap="small"):
        st.metric("论文", str(len(snapshot.papers)), border=True)
        st.metric("历史会话", str(snapshot.conversations), border=True)
        st.metric("笔记", str(snapshot.notes), border=True)
        st.metric("证据矩阵", str(snapshot.evidence_files), border=True)


def render_evidence_matrix(papers, settings: Settings, project_id: str) -> None:
    """显式触发的证据矩阵生成区；不在页面刷新时自动调用模型。"""
    with st.expander(
        "论文证据矩阵",
        icon=":material/table_view:",
        expanded=EVIDENCE_STATE_KEY in st.session_state,
    ):
        st.caption("选择 2–6 篇论文，抽取研究问题、方法、数据、指标、发现和局限。")
        if len(papers) < 2:
            st.info("当前项目至少需要 2 篇论文才能生成证据矩阵。", icon=":material/info:")
            return
        paper_names = {paper.paper_id: paper.title for paper in papers}
        with st.form("evidence-matrix-form", border=False):
            selected = st.multiselect(
                "选择论文",
                options=[paper.paper_id for paper in papers],
                format_func=lambda paper_id: paper_names[paper_id],
                default=[paper.paper_id for paper in papers[:2]],
                max_selections=6,
            )
            focus = st.text_input(
                "分析关注点",
                placeholder="例如：比较检索方法和评测设计",
            )
            submitted = st.form_submit_button(
                "生成证据矩阵",
                icon=":material/table_view:",
                disabled=len(selected) < 2,
                width="stretch",
            )
        if submitted:
            with st.spinner("正在逐篇抽取证据并生成矩阵…"):
                try:
                    result = build_evidence_matrix(
                        selected,
                        llm=get_chat_model(),
                        store=get_store(),
                        focus=focus,
                        project_id=project_id,
                    )
                    csv_path = save_evidence_csv(
                        result,
                        project_output_dir(settings, project_id),
                        focus=focus,
                    )
                except Exception as exc:
                    st.error(f"证据矩阵生成失败（{type(exc).__name__}: {exc}）")
                else:
                    st.session_state[EVIDENCE_STATE_KEY] = {
                        "project_id": project_id,
                        "result": result,
                        "csv_path": str(csv_path),
                    }
                    st.rerun()

        state = st.session_state.get(EVIDENCE_STATE_KEY)
        if not state or state.get("project_id") != project_id:
            return
        result: EvidenceMatrixResult = state["result"]
        rows = [
            {
                "paper_id": row.paper_id,
                "标题": result.titles.get(row.paper_id, ""),
                "研究问题": row.research_problem,
                "方法": row.method,
                "数据": row.datasets,
                "指标": row.metrics,
                "主要发现": row.findings,
                "局限": row.limitations,
            }
            for row in result.matrix.rows
        ]
        st.dataframe(rows, hide_index=True, key="evidence-matrix-table")
        st.markdown("**综合比较**")
        st.markdown(result.matrix.synthesis)
        st.markdown("**选择建议**")
        st.markdown(result.matrix.selection_advice)
        csv_path = state["csv_path"]
        st.download_button(
            "下载证据矩阵 CSV",
            data=result.csv_bytes(),
            file_name=Path(csv_path).name,
            mime="text/csv",
            key="download-current-evidence",
            icon=":material/download:",
        )


def run_turn(agent, config: dict, prompt: str) -> bool:
    """执行一轮对话并实时展示工具调用和回答。成功返回 True。"""
    st.chat_message("user", avatar=":material/person:").markdown(prompt)
    with st.chat_message("assistant", avatar=":material/science:"):
        steps_area = st.container()
        answer_area = st.empty()
        status_boxes = {}
        answer = ""
        try:
            for event in stream_turn(agent, config, prompt):
                if isinstance(event, AnswerText):
                    answer += event.text
                    answer_area.markdown(answer + "▌")
                elif isinstance(event, ToolStarted):
                    answer = ""  # 调用工具前的过渡性文字不保留，与历史记录的显示一致
                    answer_area.empty()
                    box = steps_area.status(
                        _step_title(event.name),
                        type="step",
                        state="running",
                        expanded=False,
                    )
                    box.caption("参数")
                    box.json(event.args)
                    status_boxes[event.call_id] = box
                elif isinstance(event, ToolFinished):
                    box = status_boxes[event.call_id]
                    box.caption("结果")
                    box.text(_preview(event.result))
                    box.update(state="error" if event.failed else "complete")
        except Exception as exc:  # 界面边界：把模型或网络错误显示出来，而不是让页面崩溃
            answer_area.empty()
            st.error(f"这一轮没有完成（{type(exc).__name__}: {exc}）。可以直接重新提问。")
            return False
        answer_area.markdown(answer)
    return True


def main() -> None:
    st.set_page_config(
        page_title="科研智能体",
        page_icon=":material/science:",
        layout="wide",
    )
    settings = get_settings()
    catalog = ProjectCatalog(settings.data_dir)
    project = current_project(catalog)
    context_token = set_current_project(project.project_id)
    try:
        agent = load_agent() if settings.deepseek_api_key else None
        thread_id = current_thread_id(project.project_id)
        snapshot = render_sidebar(settings, agent, thread_id, project, catalog)

        st.title("科研智能体", icon=":material/science:")
        st.caption(
            "主控 ReAct · 文献调研 / 论文精读 / idea 研究三个专职工作流"
            f"　|　模型：{settings.deepseek_model}"
        )

        if agent is None:
            st.warning(
                "还没有配置 DEEPSEEK_API_KEY。请把项目根目录的 `.env.example` 复制为 `.env`，"
                "填入 Key 后重启本程序。论文上传、项目切换和论文库不受影响。"
            )
            st.stop()

        render_workspace_summary(project, snapshot)
        render_evidence_matrix(snapshot.papers, settings, project.project_id)
        config = {"configurable": {"thread_id": thread_id}}
        turns = history_to_turns(agent.get_state(config).values.get("messages", []))

        for turn in turns:
            avatar = ":material/person:" if turn.role == "user" else ":material/science:"
            with st.chat_message(turn.role, avatar=avatar):
                for step in turn.steps:
                    render_step(step)
                if turn.content:
                    st.markdown(turn.content)

        if not turns:
            st.subheader("开始一次研究", icon=":material/play_arrow:")
            st.caption("选择一种常见任务，或者直接在下方输入完整问题。")
            for column, (icon, title, description, prompt_text) in zip(
                st.columns(3, gap="medium"), SUGGESTIONS
            ):
                with column.container(border=True, height="stretch"):
                    st.markdown(f"{icon} **{title}**")
                    st.caption(description)
                    st.space("stretch")
                    if st.button(
                        "开始",
                        key=f"suggestion-{title}",
                        type="primary",
                        width="stretch",
                    ):
                        st.session_state["queued_prompt"] = prompt_text
                        st.rerun()

        prompt = st.chat_input(
            "例如：帮我调研多模态 RAG 的最新进展",
            submit_mode="disable",
        ) or st.session_state.pop("queued_prompt", None)
        if prompt and run_turn(agent, config, prompt):
            st.rerun()  # 从对话记录重新渲染，保证显示内容与保存的记录一致
    finally:
        reset_current_project(context_token)


main()

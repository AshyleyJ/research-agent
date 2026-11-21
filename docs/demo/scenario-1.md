# 演示场景：文献调研并保存综述

- 模型：deepseek-flash
- 时间：2026-11-21 10:48

## 用户

调研 LLM 智能体记忆机制方向近两年的代表性工作，写一份简短综述并保存成笔记

## 智能体

**工具调用（完成）：`search_arxiv`**

参数：`{"query": "LLM agent memory survey", "max_results": 10}`

```text
[1] Understanding the planning of LLM agents: A survey
    作者: Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang 等
    发表: 2024-02-05 | arXiv: 2402.02716v1
    链接: http://arxiv.org/abs/2402.02716v1
    摘要: As Large Language Models (LLMs) have shown significant intelligence, the progress to leverage LLMs as planning modules of autonomous agents has attracted more attention. This survey provides the first systematic view of LLM-based agents planning, covering recent works aiming to improve planning ability. We provide a taxonomy of existing works on LLM-Agent planning, which can be categorized into Task Decomposition, Plan Selection, External Module, Reflection and Memory. Comprehensive analyses are conducted for each direction, and further challenges for the field of research are discussed.

[2] A Survey on the Memory Mechanism of Large Language Model based Agents
    作者: Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li 等
    发表: 2024-04-21 | arXiv: 2404.13501v1
    链接: http://arxiv.org/abs/2404.13501v1
    摘要: Large language model (LLM) based agents have recently attracted much attention from the research and industry communities. Compared with original LLMs, LLM-based agents are featured in their self-evolving capability, which is the basis for solving real-world problems that need long-term and complex agent-environment interactions. The key component to support agent-environment interactions is the memory of the agents. While previous studies have proposed many promising memory mech
…（共 9201 字符）
```

**工具调用（完成）：`search_arxiv`**

参数：`{"query": "LLM agent long-term memory management", "max_results": 10}`

```text
[1] Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents
    作者: Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan 等
    发表: 2026-01-05 | arXiv: 2601.01885v3
    链接: http://arxiv.org/abs/2601.01885v3
    摘要: Large language model (LLM) agents face fundamental limitations in long-horizon reasoning due to finite context windows, making effective memory management critical. Existing methods typically handle long-term memory (LTM) and short-term memory (STM) as separate components, relying on heuristics or auxiliary controllers, which limits adaptability and end-to-end optimization. In this paper, we propose Agentic Memory (AgeMem), a unified framework that integrates LTM and STM management directly into the agent's policy. AgeMem exposes memory operations as tool-based actions, enabling the LLM agent to autonomously decide what and when to store, retrieve, update, summarize, or discard information.…

[2] MemRefine: LLM-Guided Compression for Long-Term Agent Memory
    作者: Minjae Kim, Jinheon Baek, Soyeong Jeong, Sung Ju Hwang
    发表: 2026-06-11 | arXiv: 2606.13177v1
    链接: http://arxiv.org/abs/2606.13177v1
    摘要: Large language model (LLM) agents are increasingly expected to operate over long-term interactions, where information from past dialogues must be preserved and recalled to support future tasks. However, as interactions accumulate, the memory store grows without bound and fills with redundant entries that inflate storage
…（共 9363 字符）
```

**工具调用（完成）：`search_arxiv`**

参数：`{"query": "MemGPT memory operating system LLM agent", "max_results": 8}`

```text
[1] MemGPT: Towards LLMs as Operating Systems
    作者: Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang 等
    发表: 2023-10-12 | arXiv: 2310.08560v2
    链接: http://arxiv.org/abs/2310.08560v2
    摘要: Large language models (LLMs) have revolutionized AI, but are constrained by limited context windows, hindering their utility in tasks like extended conversations and document analysis. To enable using context beyond limited context windows, we propose virtual context management, a technique drawing inspiration from hierarchical memory systems in traditional operating systems that provide the appearance of large memory resources through data movement between fast and slow memory. Using this technique, we introduce MemGPT (Memory-GPT), a system that intelligently manages different memory tiers in order to effectively provide extended context within the LLM's limited context window, and utilize…

[2] eMEM: A Hybrid Spatio-Temporal Memory System For Embodied Agents
    作者: A. Haroon Rasheed, Maria Kabtoul
    发表: 2026-06-02 | arXiv: 2606.03374v2
    链接: http://arxiv.org/abs/2606.03374v2
    摘要: We present eMEM (Embodied Memory), a hybrid graph-based memory system for embodied agents operating in physical environments. Current agent memory architectures, such as Generative Agents, MemGPT, and A-MEM, treat memory as text streams or knowledge graphs, but embodied agents require memory that is simultaneously searchable by meaning, space, and time. eMEM fills this gap with a multi-index
…（共 3615 字符）
```

**工具调用（完成）：`search_arxiv`**

参数：`{"query": "agentic memory Zettelkasten note linking dynamic memory", "max_results": 8}`

```text
[1] A-MEM: Agentic Memory for LLM Agents
    作者: Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao 等
    发表: 2025-02-17 | arXiv: 2502.12110v11
    链接: http://arxiv.org/abs/2502.12110v11
    摘要: While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected know…
```

**工具调用（完成）：`read_paper`**

参数：`{"source": "2404.13501"}`

```text
paper_id: 2404.13501
标题: A Survey on the Memory Mechanism of Large Language Model based Agents
页数: 39

## 研究问题
论文关注LLM智能体的记忆机制，指出以往研究虽提出许多记忆机制但分散于不同论文，缺乏整体视角的系统总结与比较，未能抽象出通用有效的设计模式（第1页）。论文围绕三个关键问题展开：LLM智能体的记忆“是什么”、“为什么需要”记忆、以及“如何实现与评估”记忆（第4页）。

## 核心方法
论文先给出任务、环境、试次与步骤等基本定义，区分狭义记忆（仅同一试次内历史信息）与广义记忆（还包含跨试次信息与外部知识）（第7-9页）。论文提出记忆辅助的智能体-环境交互包含三个阶段，对应记忆写入、记忆管理、记忆读取三种操作，并给出统一的形式化演化函数（第9-10页）。记忆实现从记忆来源（试次内信息、跨试次信息、外部知识）、记忆形式（文本形式与参数化形式，后者含微调与知识编辑）、记忆操作（写入、管理、读取）三个维度综述，并以表格汇总代表性模型（第11-19页）。记忆评估分为直接评估（主观评估关注连贯性与合理性；客观评估包括结果正确性、参考准确率、时间与硬件成本）和间接评估（对话、多源问答、长上下文应用及其他任务如成功率、探索程度）（第20-23页）。论文还综述了记忆增强的智能体应用场景，包括角色扮演与社会模拟、个人助理、开放世界游戏、代码生成、推荐、特定领域专家系统及其他应用（第23-27页）。

## 实验设置与主要结果
文中未说明。

## 主要贡献
论文形式化定义了记忆模块，并从认知心理学、自我进化、智能体应用三个视角全面分析其对LLM智能体的必要性（第4、10-11页）。论文系统总结现有关于记忆模块设计与评估的研究，提供清晰的分类体系与直观见解（第4页）。论文展示典型智能体应用，说明记忆模块在不同场景中的重要性（第4页）。论文分析现有记忆模块的关键局限并提出潜在解决方向；据作者所知，这是首篇关于LLM智能体记忆机制的综述（第4页）。

## 局限与不足
现有工作局限方面：参数化记忆仍研究不足；多智能体应用中的记忆（同步、通信、信息不对称）有待探索；基于记忆的终身学习面临时间性、记忆重叠、海量存储与遗忘机制等挑战；人形智能体的记忆需符合人类认知规律并具备知识边界（第27-28页）。评估方面的局限：间接评估易受记忆以外因素影响而产生偏差；直接评估虽更可靠，但据作者所知尚无专为LLM智能体记忆模块设计的开源基准（第23页）。

## 可以借鉴或继续研究的方向
文中未说明。
```

**工具调用（完成）：`read_paper`**

参数：`{"source": "2505.00675"}`

```text
paper_id: 2505.00675
标题: Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics
页数: 64

## 研究问题
现有记忆综述多聚焦应用层面（如个性化对话），忽视支配记忆动态的原子操作，且多采用基于类型和认知启发的视角，缺乏统一的操作形式化，部分综述只覆盖高层操作（如写、管理、读）而遗漏索引等操作（第1页）。因此需要一个统一、系统的框架来组织和演化智能体记忆，并明确记忆研究范围、分析技术实现、提供基准与工具等实践基础（第1页）。第32–61页附录部分未提出新的研究问题，主要是对论文筛选与分析方法、以及已有方法/数据集的补充说明（第44页）。第62–64页则聚焦于按工具层级（组件级、框架级、应用层、产品级）分类梳理记忆管理与利用的工具（第62页、第63页、第64页）。

## 核心方法
论文将记忆分为参数化记忆（隐式存于模型权重）与上下文记忆（显式外部数据，分结构化与非结构化），并按时间跨度区分长期与短期上下文（第1页）。定义六种核心记忆操作并归为三类：编码（巩固、索引）、演化（更新、遗忘）、适应（检索、压缩），并从认知心理学引入情景、语义、程序、工作记忆作为补充视角（第1—2页）。研究方法上，先分析37篇种子论文确定记忆类型与核心操作分类，再收集2022—2025年NeurIPS、ICLR、ICML、ACL、EMNLP、NAACL等顶会超过30,000篇论文，用GPT-4o-mini对摘要做相关性打分，保留≥8/10的论文，得到3,923篇高相关论文（第2—3页）。附录进一步说明该GPT评分流水线（1–10打分、阈值≥8）及相对引用指数（RCI）的构造：以2025年4月20日为引用收集日期，比较线性、指数、log-log三种模型，log-log拟合最佳，估计参数为β̂=1.878、α̂=1.297，RCI_i=C_i/Ĉ_i，RCI≥1视为高于同龄中位引用（第44—45页）。工具部分提出按“工具层级”划分的分类体系（组件级、框架级、应用层、产品级），每个工具按记忆分类、操作、功能、输入/输出、示例用途、来源类型等维度描述（第62页、第63页、第64页）。

## 实验设置与主要结果
RCI的对数-对数回归模型拟合最佳（R²=0.97），语义相关性过滤与RCI影响评估结合构成可复现的分析基础（第3页）。长期记忆评估方面，现有基准主要分知识型问答与多轮对话两类，问答类常假设静态记忆，多轮对话基准（如LoCoMo、LongMemEval）跨20—30轮但多数仍将对话历史视为静态上下文（第11页）；检索与生成存在错配，先进模型在2Wiki和MemoryBank上Recall@5超过90，但生成指标（如F1）落后30分以上（第11—12页）。参数化记忆修改的SOTA分析显示提示类方法整体表现强、元学习方法普遍较弱，多数方法仅测试1,000至5,000次编辑，除MemoryLLM支持多达65万次更新外扩展性有限（第17页）。附录C的趋势分析表明2023年是LLM出现后的关键年份，长上下文与参数记忆相关发表的数量和质量均激增，长上下文从2022年前代表性最低的主题之一变为2024年最突出主题（第45页）；弦图分析显示研究主要集中在参数记忆与上下文非结构化记忆，压缩与检索研究最多，索引关注较少（第46—47页）。会议分布上，压缩、遗忘、更新更多出现在ML会议，检索与巩固更多出现在NLP会议，索引在两类会议中均代表性不足（
…（共 2609 字符）
```

**工具调用（完成）：`save_note`**

参数：`{"title": "LLM智能体记忆机制近两年代表性工作综述", "content": "# LLM 智能体记忆机制：近两年代表性工作综述\n\n> 调研范围：2024–2026 年 arXiv 上的综述、代表性记忆系统与评估类工作。以下内容均基于检索到的论文标题与摘要，以及两篇核心综述的精读笔记（标注页码）。\n\n## 一、总体判断\n\n近两年该方向已从\"给智能体加一个向量库\"的工程做法，收敛为三条相对清晰的研究线：\n\n1. **框架化 / 分类学**：多篇综述试图统一\"记忆是什么、有哪些操作、怎么评估\"，代表是 Zhang et al. 的 2404.13501 与 Du et al. 的 2505.00675。\n2. **系统与架构**：从分层虚拟上下文（MemGPT）、Zettelkasten 式动态组织（A-MEM），走向图结构、层次化、价值感知与预算受限的记忆管理。\n3. **记忆的\"动力学\"与可靠性**：写入策略、遗忘/压缩、经验跟随（experience-following）、记忆投毒与过时记忆等新问题开始被单独研究。\n\n一个明显的趋势是：**研究重心正从\"存什么、怎么检索\"转向\"记忆如何演化、如何被策略性地管理\"**。Luo et al. (2605.06716) 把这一演化概括为 Storage → Reflection → Experience 三阶段，与上述观察一致。\n\n## 二、分类框架：两篇核心综述\n\n### 1. Zhang et al., *A Survey on the Memory Mechanism of LLM-based Agents* (arXiv:2404.13501)\n\n- **形式化**：区分狭义记忆（同一试次内历史）与广义记忆（含跨试次信息与外部知识）（第 7–9 页）；把记忆辅助的交互拆成**写入—管理—读取**三阶段，并给出统一的演化函数（第 9–10 页）。\n- **三维度梳理**：记忆来源（试次内 / 跨试次 / 外部知识）、记忆形式（文本 / 参数化，后者含微调与知识编辑）、记忆操作（写、管、读）（第 11–19 页）。\n- **评估**：直接评估（主观连贯性；客观正确率、参考准确率、时间与硬件成本）与间接评估（对话、多源问答、长上下文任务等）（第 20–23 页）。\n- **指出的空白**：参数化记忆研究不足；多智能体记忆（同步、通信、信息不对称）待探索；终身学习面临时间性、记忆重叠、海量存储与遗忘机制挑战；且**尚无专为记忆模块设计的开源基准**（第 23、27–28 页）。\n\n### 2. Du et al., *Rethinking Memory in LLM-based Agents* (arXiv:2505.00675)\n\n- **表示**：参数化记忆（隐于权重）vs 上下文记忆（显式外部数据，分结构化/非结构化），并按时间跨度分长期/短期（第 1 页）。\n- **六种原子操作**，归为三类：编码（巩固 consolidation、索引 indexing）、演化（更新 updating、遗忘 forgetting）、适应（检索 retrieval、压缩 condensation）（第 1–2 页）。这一\"操作视角\"比\"写/管/读\"更细，尤其补上了**索引**这一此前被忽略的环节。\n- **方法论亮点**：用 GPT-4o-mini 对 2022–2025 年顶会 3 万余篇摘要打分（阈值 ≥8/10），筛出 3,923 篇高相关论文；并提出相对引用指数 RCI 做趋势分析（第 2–3、44–45 页）。\n- **关键实证观察**：\n  - 检索与生成存在**错配**——先进模型在 2Wiki、MemoryBank 上 Recall@5 超 90，但 F1 落后 30 分以上（第 11–12 页）。\n  - 现有长期记忆基准多把对话历史当作**静态上下文**，忽视动态操作与时间连续性（第 11 页）。\n  - 参数化编辑方法多数只测 1,000–5,000 次编辑，扩展性有限（第 17 页）。\n  - 多源记忆的**冲突**未解决，检索内容与参数内容常未经一致性检查即合并，导致幻觉与事实漂移（第 19、21 页）。\n\n> 两篇综述互补：2404.13501 提供\"来源—形式—操作\"的静态分类与评估框架，2505.00675 提供\"原子操作 + 生命周期\"的动态视角，并给出可复现的文献计量证据。\n\n## 三、代表性系统工作\n\n### 分层与虚拟上下文\n- **MemGPT** (arXiv:2310.08560)：借鉴操作系统分层内存，提出 *virtual context management*，在快/慢记忆层间搬运数据，用有限上下文窗口模拟大内存。是后续大量工作的架构原型。\n- **CALMem** (arXiv:2605.20724)：应用层双记忆架构，不改模型、不锁定供应商，专门针对\"上下文压缩导致历史不可逆丢失\"和\"会话结束记忆归零\"两个问题。\n\n### 结构化 / 图式记忆\n- **A-MEM** (arXiv:2502.12110)：按 Zettelkasten 原则构建互链知识网络，让智能体**动态组织**记忆，而非固定结构与固定操作。\n- **SGMem** (arXiv:2509.21212)：句子级图记忆，在 chunk 内建图，跨 turn / round / session 捕捉关联，并把原始对话与摘要、事实、洞见等生成记忆混合供给 LLM。\n- **HiGMem** (arXiv:2604.18349)：两级 event–turn 层次记忆，用 LLM 引导检索以替代纯向量相似度，缓解\"证据集膨胀\"（召回提升有限但精度下降、上下文成本上升）。\n- **Graph-based Personalized Memory** (arXiv:2609.08599)：面向长期个性化助手的图式记忆综述，覆盖表示、演化、检索、评估四环节。\n\n### 反思与个性化\n- **Reflective Memory Management (RMM)** (arXiv:2503.08026)：针对\"记忆粒度僵化\"与\"检索机制固定\"两个问题，引入前瞻/回顾式反思来做自适应粒度与检索。\n- **Hello Again!** (arXiv:2406.05925)：以事件摘要 + persona 管理支撑长期对话个性化。\n\n### 记忆管理的策略化与自动化\n- **AgeMem** (arXiv:2601.01885)：把长期与短期记忆管理**统一进策略**，将记忆操作暴露为工具动作，由 LLM 自主决定何时存、取、更新、摘要、丢弃，实现端到端优化，替代启发式或外挂控制器。\n- **MemRouter** (arXiv:2605.00356)：把\"是否写入\"从每轮 LLM 解码改为**基于 embedding 的路由**，仅训练 12M 参数，在 LoCoMo 上做匹配 harness 对比。\n- **MemRefine** (arXiv:2606.13177)：提出**存储预算受限的记忆管理**任务——在固定预算下压缩已建好的记忆库，保留对未来交互有用的信息。\n- **MemLens** (arXiv:2607.25992)：价值感知的记忆管理系统，把记忆记录当作一等数据对象，用 Shapley 式评估 + 交互式分析面板暴露完整记忆生命周期。\n\n### 可靠性\n- **Xiong et al.** (arXiv:2505.16067)：实证研究记忆增删对长期表现的影响，发现 **experience-following** 现象——任务输入与检索到的记忆输入高度相似时，智能体会复制记忆中的行为（包括错误）。\n- **Zhao et al.** (arXiv:2610.41056)：把记忆鲁棒性形式化为同时处理**过时记忆**与**记忆投毒**，指出二者要求相反的写入决策，因此不存在固定写入规则；并提出 evidence-conditioned accuracy (ECA) 度量。\n\n## 四、评估现状与缺口\n\n| 维度 | 现状 | 缺口 |\n|---|---|---|\n| 基准 | LoCoMo、LongMemEval、MemoryBank 等，跨 20–30 轮 | 多把历史当静态上下文，缺动态操作与时间连续性评估（2505.00675, 第 11 页） |\n| 指标 | Recall@k、F1、成功率 | 检索指标与生成指标严重错配（第 11–12 页）；ECA 等证据条件化指标刚出现 |\n| 专用基准 | — | 2404.13501 指出尚无专为记忆模块设计的开源基准（第 23 页） |\n| 成本 | 时间与硬件成本被纳入直接评估 | 存储预算、上下文成本尚未成为标准报告项 |\n\n## 五、值得关注的研究机会\n\n1. **动态评估协议**：设计能检验\"写入—更新—遗忘\"操作正确性的基准，而非只测最终答案（2505.00675, 第 11–12 页）。\n2. **索引与结构化—非结构化融合**：索引在现有工作中代表性最低，是明确的空白（第 47 页）。\n3. **多源记忆冲突**：异构来源间的一致性与来源归因（第 19、21 页）。\n4. **记忆鲁棒性**：过时记忆与投毒的统一处理（2610.41056），以及 experience-following 带来的错误放大（2505.16067）。\n5. **预算受限下的记忆管理**：把存储/上下文成本作为一等约束（2606.13177、2607.25992）。\n6. **参数化记忆的可扩展编辑与终身学习**：现有方法难以扩展到数千次以上编辑（2505.00675, 第 17 页；2404.13501, 第 27–28 页）。\n\n## 六、参考文献\n\n**综述**\n- Zhang et al. *A Survey on the Memory Mechanism of LLM-based Agents*. arXiv:2404.13501 — http://arxiv.org/abs/2404.13501\n- Du et al. *Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics*. arXiv:2505.00675 — http://arxiv.org/abs/2505.00675\n- Luo et al. *From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms*. arXiv:2605.06716 — http://arxiv.org/abs/2605.06716\n- Nguyen et al. *Graph-Based Personalized Memory for LLM Agents*. arXiv:2609.08599 — http://arxiv.org/abs/2609.08599\n- Huang et al. *Understanding the Planning of LLM Agents: A Survey*. arXiv:2402.02716 — http://arxiv.org/abs/2402.02716\n\n**系统**\n- Packer et al. *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560 — http://arxiv.org/abs/2310.08560\n- Xu et al. *A-MEM: Agentic Memory for LLM Agents*. arXiv:2502.12110 — http://arxiv.org/abs/2502.12110\n- Wu et al. *SGMem: Sentence Graph Memory for Long-Term Conversational Agents*. arXiv:2509.21212 — http://arxiv.org/abs/2509.21212\n- Cao et al. *HiGMem: A Hierarchical and LLM-Guided Memory System*. arXiv:2604.18349 — http://arxiv.org/abs/2604.18349\n- Tan et al. *In Prospect and Retrospect: Reflective Memory Management*. arXiv:2503.08026 — http://arxiv.org/abs/2503.08026\n- Li et al. *Hello Again! LLM-powered Personalized Agent for Long-term Dialogue*. arXiv:2406.05925 — http://arxiv.org/abs/2406.05925\n- Yu et al. *Agentic Memory (AgeMem)*. arXiv:2601.01885 — http://arxiv.org/abs/2601.01885\n- Hu et al. *MemRouter: Memory-as-Embedding Routing*. arXiv:2605.00356 — http://arxiv.org/abs/2605.00356\n- Kim et al. *MemRefine: LLM-Guided Compression for Long-Term Agent Memory*. arXiv:2606.13177 — http://arxiv.org/abs/2606.13177\n- Wei et al. *MemLens: A Value-Aware Memory Management System*. arXiv:2607.25992 — http://arxiv.org/abs/2607.25992\n- Jena et al. *CALMem: Application-Layer Dual Memory*. arXiv:2605.20724 — http://arxiv.org/abs/2605.20724\n\n**评估与可靠性**\n- Xiong et al. *How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior*. arXiv:2505.16067 — http://arxiv.org/abs/2505.16067\n- Zhao et al. *Accept Changes, Reject Attacks: Robust Memory Management for LLM Agents*. arXiv:2610.41056 — http://arxiv.org/abs/2610.41056\n\n---\n*说明：除 2404.13501 与 2505.00675 为精读（结论带页码）外，其余工作仅依据标题与摘要，未涉及其摘要之外的实验细节。*\n"}`

```text
已保存到 outputs/20261121-104759-LLM智能体记忆机制近两年代表性工作综述.md
```

综述已完成并保存为 **`outputs/20261121-104759-LLM智能体记忆机制近两年代表性工作综述.md`**。

## 核心结论

近两年该方向已从“给智能体加一个向量库”的工程做法，收敛为三条研究线：

1. **框架化**：多篇综述试图统一“记忆是什么、有哪些操作、怎么评估”
2. **系统架构**：分层虚拟上下文 → Zettelkasten 式动态组织 → 图结构/层次化/价值感知/预算受限
3. **记忆动力学与可靠性**：写入策略、遗忘压缩、experience-following、记忆投毒等新问题被单独研究

总体趋势是**研究重心从“存什么、怎么检索”转向“记忆如何演化、如何被策略性地管理”**。

## 两篇核心综述（已精读，结论带页码）

- **Zhang et al. (2404.13501)**：提出“写入—管理—读取”三阶段形式化，按来源/形式/操作三维度梳理，并指出**尚无专为记忆模块设计的开源基准**（第 23 页）
- **Du et al. (2505.00675)**：提出六种原子操作（巩固、索引、更新、遗忘、检索、压缩），补上了此前被忽略的**索引**环节；并给出关键实证观察——检索与生成严重错配（Recall@5 超 90 但 F1 落后 30 分以上，第 11–12 页）

## 代表性系统（按架构思路分组）

- **分层/虚拟上下文**：MemGPT、CALMem
- **图式/结构化**：A-MEM、SGMem、HiGMem
- **反思与个性化**：RMM、Hello Again!
- **策略化管理**：AgeMem（记忆操作作为工具动作）、MemRouter（embedding 路由替代逐轮解码）、MemRefine（预算受限压缩）、MemLens（价值感知）
- **可靠性**：experience-following 实证研究、过时记忆与投毒的统一处理

## 值得关注的研究机会

动态评估协议、索引与结构化—非结构化融合、多源记忆冲突、记忆鲁棒性、预算受限管理、参数化记忆的可扩展编辑。

**说明**：除上述两篇综述外，其余工作仅依据标题与摘要撰写，未涉及其摘要之外的实验细节。如果你希望深入某一支线（比如把 AgeMem、MemRouter 这类“记忆策略化”工作精读对比），可以告诉我，下一轮继续。

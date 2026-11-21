# 长文档问答中的检索增强生成：给定摘要范围内的代表性方法、评测局限与失败案例初步调研

> 主题：Retrieval augmented generation for long-document question answering: representative methods, evaluation limitations and failure cases
> 检索式：iterative retrieval reranking long document question answering；hierarchical index retrieval long context question answering；long-document RAG benchmark evaluation limitations failure cases；long document QA benchmark evaluation limitations；retrieval augmented generation failure cases hallucination；QUEST-LOFT retrieval evaluation protocol multi-document reasoning
> 文献数：10 篇 | 修订：1 次 | 质量检查：评审通过

## 摘要

本报告仅依据给定的 10 篇论文标题与摘要，对长文档问答（long-document QA）中检索增强生成（RAG）的代表性方法、评测局限与失败案例做摘要级初步调研，不构成系统综述，未核对全文、实验表格与附录。按摘要所述任务设定，直接相关文献为长文档视觉证据问答基准 [2]、多模态图 RAG 与文档级 VQA [6]、QUEST-LOFT 上的检索式问答评测 [4]；其中 [2][6] 面向视觉丰富文档，[4] 面向信息分散于多文档的检索式问答。给定文献中未见纯文本单篇超长文档 RAG 方法的摘要，故本报告不覆盖该方向，也不对其作任何方法学判断。迭代检索与重排类工作 [1][5][7] 摘要未涉及长文档设定，仅作背景参考，不与直接相关文献比较优劣。所有数值、因果与强弱表述均限于摘要原句可支持的范围。

## 范围界定与文献关系表

本报告只处理“长文档问答中的检索增强生成”，材料为给定的 10 篇论文摘要，属摘要级初步调研，不对方法做排名。按摘要所述任务设定归类如下（相关性标注为待核验的模型标注，仅摘要原句存在性经代码校验）：[2] VELD-QA，长文档视觉证据定位与问答，5,859 份真实 PDF、372,445 页，直接相关，摘要未给出 token 阈值；[6] 多模态图 RAG，长文档视觉丰富文档理解与文档级 VQA，直接相关，仅称受上下文窗口限制，阈值未知；[4] QUEST-LOFT，信息分散于多文档、需结合复杂推理的检索式问答评测，直接相关，长度阈值未知；[1] ToR，多跳问答，相邻；[7] any-hop 迭代重排，开放域单跳/多跳问答，相邻；[5] RIDER，开放域问答段落重排序，相邻；[8] Ada-LEval，长上下文 LLM 评测基准，相邻（评测侧），支持至 128k token 样本；[10] KV-cache 基准，长上下文服务效率评测，相邻（系统侧），按上下文长度分桶；[3] MRQA 数据增强与 [9] MusT-RAG，相邻（弱），摘要均未涉及长文档设定。集中声明如下，后文不再重复：表中“直接/相邻”是本次按任务设定做的归纳，不是既有分类学；相邻文献只作背景参考，不与直接相关文献比较优劣；多数摘要未给出“长文档”的长度阈值，本报告一律记为未知，不自行定义；未检索到某方向不等于该方向在领域内空白，只表示本次给定材料无相应摘要。

依据：[1] [2] [3] [4] [5] [6] [7] [8] [9] [10]

## 直接相关文献的方法要点

本节只归纳三篇直接相关文献摘要所述的方法构成，不比较其相对增益。VELD-QA 的贡献在基准与评测设计：15,000 个人工撰写问答对、5,859 份真实 PDF、共 372,445 页，平衡图表、表格与复杂版式任务；人工标注识别支撑页与证据元素并描述所引视觉内容，从而使检索评估与解读评估可以分开进行 [2]。多模态图 RAG 工作针对长文档受上下文窗口限制的问题，引入知识图谱汇总文档全局知识，构建多模态图 RAG；作者指出多模态 RAG 虽通过检索相关页面缓解窗口限制，但在需要整体理解文档的视觉问答上仍存在困难，并因缺乏文档级整体理解 VQA 的标注数据集而提出 DLVQA 基准，提供参考摘要与对应支撑事实 [6]。QUEST-LOFT 工作对检索式问答在信息分散于多文档、需结合复杂推理的问题上的表现做深入分析，基于细致人工评测发布更新后的数字，并报告 RAG 在配合包含推理与证据的结构化输出格式、可选地追加答案再验证时，可显著优于长上下文方法 [4]。三篇摘要均未给出检索器与生成器的解耦消融、k 值或上下文预算设定，故本报告不描述其内部组件贡献。

依据：[2] [4] [6]

## 评测局限：基准设计、长度区间与指标口径

作者指出的评测局限可归为三类，均为摘要层面的自述，未经本报告独立核验。其一，标注粒度不足导致失败类型不可分离：VELD-QA 作者指出，已有长文档 QA 基准很少同时具备大规模人工撰写问题与细粒度视觉证据标注，从而限制了对证据定位与解释失败的诊断 [2]；多模态图 RAG 工作指出，现有基于 LLM 的知识图谱方法多依赖综合性、多样性、赋能等间接证据评估 QA 性能，缺乏文档级整体理解 VQA 的标注数据集，构成有效评测的障碍 [6]。其二，长度区间混杂与超长设定缺位：Ada-LEval 作者指出，L-Eval、LongBench 等既有长文本基准把 2k 至 32k+ 的不同长度样本混在一起，难以按长度区间评估能力，且未覆盖 100k+ token 的超长设定；其 TSort 与 BestAnswer 子集支持对测试长度做可控操纵，可生成至 128k token 的样本 [8]。其三，人工评测与自动指标的关系不明：QUEST-LOFT 工作称基于细致人工评测发布了更新后的数字，但摘要未说明更新的方向与幅度，故本报告不对自动指标与人工判定的一致性作任何判断 [4]。此外，KV-cache 优化基准提示长文档 QA 的系统级评测宜同时报告质量与效率，而非单一指标；该文报告压缩率本身是端到端性能的差预测指标，不同机制在不同负载上表现差异明显 [10]。以上均为各作者针对自身关注点提出的局限，摘要未提供跨基准的频率统计，本报告不据此判断哪类局限更普遍。

依据：[2] [4] [6] [8] [10]

## 失败案例：作者问题陈述与作者报告结果的区分

摘要中出现的失败描述多为作者提出方法时的动机陈述，而非独立的错误分析，二者需分开陈述。作者问题陈述类：链式检索增强推理存在两点问题——检索到的不相关段落可能误导推理，链结构中单个错误可能引发级联错误 [1]；多步检索引入的相关但不支撑答案的文档会削弱噪声敏感的 reader [7]；多模态 RAG 在需要整体理解文档的视觉问答上仍存在困难 [6]；当前 RAG 在信息分散于多文档或需结合复杂推理的问题上表现不佳，且该局限同样适用于长上下文模型方法 [4]。作者报告的实验结果类：在 VELD-QA 上，检索类方法显著低于提供真实支撑页的 oracle 设置，给予全文访问并未弥合该差距；即使定位正确，oracle 准确率上限为 81.7%，作者据此认为视觉解读与推理是独立瓶颈；用其视觉元素标注丰富输入可使差距缩小 17–24%，摘要未说明该比例的基数与口径（是相对差距缩小还是绝对点提升），也未说明对应四种证据设置中的哪一种，故本报告不将其解读为绝对准确率提升 [2]。这些数值限于该基准与该实验设置，摘要未说明能否外推到纯文本长文档 QA。可核验边界：上述失败描述均无失败类型的频次分布或定量错误分析，[1][4][6][7] 的摘要未报告失败案例的具体类型分布或比例，因此本报告不给出失败模式的排序，也不使用“高频”“普遍”一类表述。

依据：[1] [2] [4] [6] [7]

## 相邻任务的背景参考

以下文献摘要未涉及长文档设定，仅作背景参考，不与直接相关文献比较。迭代检索与重排方面：Nie 等提出统一的 any-hop 开放域问答框架，迭代地检索、重排与过滤文档并自适应决定何时停止检索，核心是图结构重排模型做多文档交互 [7]；ToR 以问题为根节点、检索段落为其他节点，沿推理路径动态决定发起新搜索、拒绝或接受，作者称树结构分别处理每个检索段落可缓解无关段落对推理路径的误导，推理路径多样性可降低单个推理错误对整体的影响，并另提出剪枝与有效扩展两种搜索优化策略 [1]；RIDER 仅依据重排前 reader 的 top 预测对检索段落重排，不需训练，作者报告 top-1 检索准确率与 EM 的提升，并在 reader 输入仅 1,024 tokens 时给出 NQ 与 TriviaQA 上的 EM 数值 [5]。领域无关机器阅读理解方面，作者报告一种简单的负采样技术特别有效，尽管它通常用于包含不可回答问题的数据集，并结合按域采样在 MRQA 评测中取得较好名次 [3]。领域适配方面，MusT-RAG 通过音乐专用向量库 MusWikiDB 与在推理、微调中使用上下文信息，将通用 LLM 适配到纯文本音乐问答，作者报告其优于传统微调，且 MusWikiDB 比通用 Wikipedia 语料更有效 [9]。系统效率方面，KV-cache 优化基准在 LongBench 风格的多文档 QA、单文档 QA 等负载上比较量化、剪枝与合并类机制，作者报告 KIVI4 跨模型质量最稳定、SnapKV 长上下文吞吐最强、CaM 在部分 QA 负载上增益大但对负载高度敏感 [10]；该文未评估检索增强流程本身。这些工作提示检索侧资源构建、重排与输入预算会影响下游表现，但其任务设定与长文档问答不同，摘要未提供可迁移性证据。

依据：[1] [3] [5] [7] [9] [10]

## 本次调研的证据缺口与后续问题

- 本次给定文献中未见纯文本单篇超长文档 RAG 方法的摘要，故本报告未覆盖该方向；后续需专门检索该方向的代表方法与数据集，并核对其“长文档”的长度定义与文档长度分布。
- 仅依据摘要，未核对任何全文、实验表格与附录，无法确认失败案例的统计口径（如幻觉率如何判定）、显著性检验细节，以及 [2] 中 17–24% 差距缩小的基数与口径。
- 缺少对长文档 QA 评测协议本身的系统比较：摘要层面未见对 LongBench、L-Eval、QUEST 等基准在标注质量、答案等价性判定、自动指标与人工判定一致性上的审计；[4] 仅称发布人工评测后的更新数字，未给方向与幅度。
- “检索失败”与“生成失败”的归因框架缺乏独立验证：[2] 提供了分离定位与解读的标注设计，但其结论能否迁移到纯文本长文档 QA，摘要未提供证据。
- 长文档（单篇超长）与多文档/开放域语料的界限在给定文献中未被清晰区分，各方法的长度阈值多为未知，需在全文层面核对。
- 摘要层面未见统一协议下“检索增强 vs 长上下文”的对照复现；[4] 的结果限于 QUEST-LOFT 单一基准与特定输出格式、再验证设置，[10] 的效率结论依赖负载，缺少同条件比较。
- 多跳/跨文档长文档 QA 中“部分正确”答案如何计分，摘要层面未见评测方法研究。
- 失败案例报告规范（是否报告拒绝率、检索深度、k 值、上下文预算）未见跨论文实证审计；本报告只能指出给定摘要普遍未报告这些项，不能推断领域内不存在相关规范。

## 结论

- 给定文献中与主题直接相关的是长文档视觉证据问答基准 [2]、多模态图 RAG 与文档级 VQA [6]、QUEST-LOFT 上的检索式问答评测 [4]；其中 [2][6] 面向视觉丰富文档，[4] 面向多文档分散信息，纯文本单篇超长文档 RAG 方法在给定材料中无摘要级证据。
- 作者指出的失败问题集中在三处：无关段落误导与错误级联 [1][7]、证据定位与解读的分离瓶颈 [2]、信息分散于多文档且需复杂推理时的性能不足 [4]；这些多为方法动机陈述，非独立错误分析。
- 评测局限主要来自标注粒度不足 [2][6]、长度区间混杂与超长设定缺位 [8]，以及人工评测与自动指标关系不明 [4]。
- “RAG 是否优于长上下文”在摘要层面不是单一结论：[4] 的结果限于 QUEST-LOFT 单一基准与特定结构化输出、再验证设置；[10] 表明效率结论取决于负载。
- 本报告为摘要级初步调研，所有数值与比较均限于摘要原句；未检索到的方向不等于领域空白，需以全文核验后再下结论。

## 参考文献

[1] Li Jiapeng, Liu Runze, Li Yabo 等. Tree of Reviews: A Tree-based Dynamic Iterative Retrieval Framework for Multi-hop Question Answering. arXiv:2404.14464v1, 2024. http://arxiv.org/abs/2404.14464v1
[2] Zhivar Sourati, Marianne Menglin Liu, Rongguang Wang 等. VELD-QA: A Large-Scale Dataset for Visual Evidence Grounding and Question Answering in Long Documents. arXiv:2610.20964v2, 2026. http://arxiv.org/abs/2610.20964v2
[3] Shayne Longpre, Yi Lu, Zhucheng Tu 等. An Exploration of Data Augmentation and Sampling Techniques for Domain-Agnostic Question Answering. arXiv:1912.02145v1, 2019. http://arxiv.org/abs/1912.02145v1
[4] Nathan Scales, Nathanael Schärli, Olivier Bousquet. Evaluation of retrieval-based QA on QUEST-LOFT. arXiv:2511.06125v1, 2025. http://arxiv.org/abs/2511.06125v1
[5] Yuning Mao, Pengcheng He, Xiaodong Liu 等. Rider: Reader-Guided Passage Reranking for Open-Domain Question Answering. arXiv:2101.00294v3, 2021. http://arxiv.org/abs/2101.00294v3
[6] Yi-Cheng Wang, Chu-Song Chen. Multimodal Graph RAG for Long-range Visually Rich Document Understanding. arXiv:2606.28780v2, 2026. http://arxiv.org/abs/2606.28780v2
[7] Ping Nie, Yuyu Zhang, Arun Ramamurthy 等. Answering Any-hop Open-domain Questions with Iterative Document Reranking. arXiv:2009.07465v5, 2020. http://arxiv.org/abs/2009.07465v5
[8] Chonghua Wang, Haodong Duan, Songyang Zhang 等. Ada-LEval: Evaluating long-context LLMs with length-adaptable benchmarks. arXiv:2404.06480v2, 2024. http://arxiv.org/abs/2404.06480v2
[9] Daeyong Kwon, SeungHeon Doh, Juhan Nam. MUST-RAG: MUSical Text Question Answering with Retrieval Augmented Generation. arXiv:2507.23334v2, 2025. http://arxiv.org/abs/2507.23334v2
[10] Nikita Agrawal, Ruben Mayer. Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving. arXiv:2607.05399v1, 2026. http://arxiv.org/abs/2607.05399v1

## 质量评审

覆盖度：3/5
证据一致性：4/5
结构清晰度：5/5
需要修订：否
修改意见：
- I1 | blocking | resolved | scope | rewrite
  位置：摘要；范围界定与文献关系表；直接相关文献的方法要点；小结第1条；原句：给定文献中未见纯文本单篇超长文档 RAG 方法的摘要，故本报告不覆盖该方向，也不对其作任何方法学判断。
  原因/处理：已解决：摘要明确收缩交付范围并声明不覆盖纯文本单篇超长文档 RAG；范围界定节把 [1][5][7] 标为相邻、仅作背景参考，不再作为“代表性方法路线”主体；方法要点节只归纳 [2][4][6] 且声明不比较相对增益；小结第1条重复该覆盖边界。标题也已改为“给定摘要范围内”。覆盖度仍为 3 分，原因是给定材料本身缺少纯文本长文档 RAG 摘要，属材料限制而非稿件缺陷，不构成阻断。
- I2 | advisory | resolved | overclaim | rewrite
  位置：小结第4条；直接相关文献的方法要点（[4] 段）；原句：“RAG 是否优于长上下文”在摘要层面不是单一结论：[4] 的结果限于 QUEST-LOFT 单一基准与特定结构化输出、再验证设置；[10] 表明效率结论取决于负载。
  原因/处理：已解决：原“孰优取决于输出格式与验证步骤”的一般性判断已删除，改为限定于单一基准与特定设置的表述，并在方法要点节声明不比较相对增益。
- I3 | advisory | resolved | evidence | disclose
  位置：失败案例：作者报告的实验结果类；原句：摘要未说明该比例的基数与口径（是相对差距缩小还是绝对点提升），也未说明对应四种证据设置中的哪一种，故本报告不将其解读为绝对准确率提升
  原因/处理：已解决：已显式披露 17–24% 的基数与口径未知，并限定不解读为绝对准确率提升。
- I4 | advisory | resolved | citation | rewrite
  位置：评测局限 其三；证据缺口第3条；原句：QUEST-LOFT 工作称基于细致人工评测发布了更新后的数字，但摘要未说明更新的方向与幅度，故本报告不对自动指标与人工判定的一致性作任何判断
  原因/处理：已解决：原“自动指标下的原始分数需要人工核验”的推论已删除，改为不作一致性判断，与摘要可支持范围一致。
- I5 | advisory | resolved | structure | rewrite
  位置：范围界定与文献关系表 末段集中声明；原句：集中声明如下，后文不再重复：表中“直接/相邻”是本次按任务设定做的归纳，不是既有分类学；相邻文献只作背景参考，不与直接相关文献比较优劣
  原因/处理：已解决：限定语已合并为范围界定节末的集中声明，其余章节仅保留必要单句限定，重复披露明显减少。
- I6 | advisory | resolved | scope | rewrite
  位置：相邻任务的背景参考 末段；评测局限 末段；原句：系统效率方面，KV-cache 优化基准在 LongBench 风格的多文档 QA、单文档 QA 等负载上比较量化、剪枝与合并类机制……该文未评估检索增强流程本身。
  原因/处理：已解决：[10] 已从方法路线编号中移除，降级到“相邻任务的背景参考”，并保留“未评估检索增强流程本身”的限定；评测局限处仅以一句提示质量与效率同报。
- I7 | advisory | open | evidence | disclose
  位置：失败案例 节标题与首段；原句：摘要中出现的失败描述多为作者提出方法时的动机陈述，而非独立的错误分析，二者需分开陈述。
  原因/处理：新增（advisory）：该节已按“作者问题陈述/作者报告结果”二分，并声明无频次分布、不作失败模式排序，处理得当；仅建议在节末补一句说明 [2] 的 oracle 81.7% 上限与 17–24% 缩小均来自同一基准的特定证据设置，避免读者把两者当作独立复现证据。

## 证据卡

证据卡（相关性/证据类型为模型标注；仅原文片段存在性经过代码校验，不代表语义已验证）：
[1] 任务=多跳问答（multi-hop question answering），在思维链推理中引入检索增强；相关性=adjacent；类型=author_problem
有限结论：作者指出链式检索增强推理存在两个问题：检索到的不相关段落可能误导推理，链结构中的单个错误可能引发级联错误。
摘要原句：However, these chain methods have the following problems: 1) Retrieved irrelevant paragraphs may mislead the reasoning; 2) An error in the chain structure may lead to a cascade of errors.
未知项：未涉及长文档设定；未报告失败案例的具体类型分布
[2] 任务=长文档视觉证据定位与问答（long-document visual evidence grounding and QA）；相关性=direct；类型=author_problem
有限结论：作者指出已有长文档QA基准很少同时具备大规模人工撰写问题与细粒度视觉证据标注，从而限制了对证据定位与解释失败的诊断。
摘要原句：Existing long-document QA benchmarks rarely combine large-scale human-authored questions with detailed visual evidence annotations, limiting diagnosis of evidence localization and interpretation failures.
未知项：未说明非视觉纯文本长文档QA的覆盖情况
[3] 任务=领域无关的机器阅读理解问答（MRQA 2019 Shared Task）；相关性=adjacent；类型=author_result
有限结论：作者发现一种简单的负采样技术特别有效，尽管它通常用于包含不可回答问题的数据集。
摘要原句：We find a simple negative sampling technique to be particularly effective, even though it is typically used for datasets that include unanswerable questions, such as SQuAD 2.0.
未知项：未涉及检索增强或长文档设定
[4] 任务=基于检索的问答在QUEST-LOFT上的评测（信息分散于多文档、需结合复杂推理）；相关性=direct；类型=author_problem
有限结论：作者指出当前RAG方法在信息分散于多文档或需结合复杂推理的问题上表现不佳，且该局限同样适用于长上下文语言模型方法。
摘要原句：Despite the popularity of retrieval-augmented generation (RAG) as a solution for grounded QA in both academia and industry, current RAG methods struggle with questions where the necessary information is distributed across many documents or where retrieval needs to be combined with complex reasoning.
未知项：未明确QUEST-LOFT文档长度阈值
[5] 任务=开放域问答的段落重排序（Retriever-Reader架构）；相关性=adjacent；类型=author_result
有限结论：作者提出无需训练的RIDER重排序方法，仅依据重排序前reader的top预测对检索段落重排序，并报告了top-1检索准确率与EM的提升。
摘要原句：We show that RIDER, despite its simplicity, achieves 10 to 20 absolute gains in top-1 retrieval accuracy and 1 to 4 Exact Match (EM) gains without refining the retriever or reader.
未知项：未涉及长文档QA设定
[6] 任务=长文档视觉丰富文档理解的多模态图RAG与文档级VQA；相关性=direct；类型=author_problem
有限结论：作者指出多模态RAG虽通过检索相关页面缓解上下文窗口限制，但在需要整体理解文档的视觉问答上仍存在困难。
摘要原句：Though recent multimodal retrieval-augmented generation (MMRAG) can address this challenge by retrieving relevant pages. It still struggles with the visual question answering (VQA) requiring holistic comprehension of a document.
未知项：未量化失败案例比例
[7] 任务=开放域任意跳数问答（单跳与多跳），迭代检索、重排序与过滤；相关性=adjacent；类型=author_problem
有限结论：作者指出多步文档检索常引入更多相关但不支持答案的文档，从而削弱对噪声敏感的下游reader模块。
摘要原句：Also, multi-step document retrieval often incurs higher number of relevant but non-supporting documents, which dampens the downstream noise-sensitive reader module for answer extraction.
未知项：未涉及长文档设定
[8] 任务=长上下文LLM评测基准（TSort与BestAnswer子集）；相关性=adjacent；类型=author_problem
有限结论：作者指出已有长文本基准将不同长度的测试样本混在一起，难以评估模型在不同长度区间上的能力，且未覆盖100k+ token的超长设定。
摘要原句：These datasets include test samples of varying lengths (from 2k to 32k+) entangled together, making it challenging to assess model capabilities across different length ranges. Moreover, they do not cover the ultralong settings (100k+ tokens) that the latest LLMs claim to achieve.
未知项：未专门针对检索增强QA
[9] 任务=纯文本音乐问答（MQA），基于RAG的领域适配；相关性=adjacent；类型=author_result
有限结论：作者提出MusT-RAG框架，通过音乐专用向量库MusWikiDB及在推理与微调中使用上下文信息，将通用LLM适配到音乐问答。
摘要原句：To optimize RAG for the music domain, we (1) propose MusWikiDB, a music-specialized vector database for the retrieval stage, and (2) utilizes context information during both inference and fine-tuning processes to effectively transform general-purpose LLMs into music-specific models.
未知项：未涉及长文档QA设定
[10] 任务=长上下文服务中KV-cache优化的基准评测（含LongBench风格多文档QA、单文档QA、few-shot与摘要）；相关性=adjacent；类型=author_result
有限结论：作者指出压缩率本身难以预测端到端性能，不同KV-cache机制在不同工作负载上表现差异明显。
摘要原句：The results show that the compression ratio alone is a poor predictor of end-to-end performance.
未知项：未评估检索增强流程；未涉及失败案例分析

## 修订记录

- I1：rewrite → resolved；已解决：摘要明确收缩交付范围并声明不覆盖纯文本单篇超长文档 RAG；范围界定节把 [1][5][7] 标为相邻、仅作背景参考，不再作为“代表性方法路线”主体；方法要点节只归纳 [2][4][6] 且声明不比较相对增益；小结第1条重复该覆盖边界。标题也已改为“给定摘要范围内”。覆盖度仍为 3 分，原因是给定材料本身缺少纯文本长文档 RAG 摘要，属材料限制而非稿件缺陷，不构成阻断。
- I2：rewrite → resolved；已解决：原“孰优取决于输出格式与验证步骤”的一般性判断已删除，改为限定于单一基准与特定设置的表述，并在方法要点节声明不比较相对增益。
- I3：disclose → resolved；已解决：已显式披露 17–24% 的基数与口径未知，并限定不解读为绝对准确率提升。
- I4：rewrite → resolved；已解决：原“自动指标下的原始分数需要人工核验”的推论已删除，改为不作一致性判断，与摘要可支持范围一致。
- I5：rewrite → resolved；已解决：限定语已合并为范围界定节末的集中声明，其余章节仅保留必要单句限定，重复披露明显减少。
- I6：rewrite → resolved；已解决：[10] 已从方法路线编号中移除，降级到“相邻任务的背景参考”，并保留“未评估检索增强流程本身”的限定；评测局限处仅以一句提示质量与效率同报。

## 调查记录

仅依据摘要，未核对全文。
- progress：partial，退出原因 search_budget，检索请求 3 次，缓存命中 0 次，耗时 28.973s
  - 查询：iterative retrieval reranking long document question answering
  - 查询：hierarchical index retrieval long context question answering
  - 查询：long-document RAG benchmark evaluation limitations failure cases
  - 证据缺口：检索预算在第二轮耗尽，未能获得专门针对“lost in the middle”/长上下文位置偏置与 RAG 检索失败的证据，该方向在本轮结果中无摘要级支撑。
  - 证据缺口：多数方法（ToR、SF-RAG、GroundedKG-RAG、PAGE-RAG、MHier-RAG、混合图+向量工作）仅见摘要，缺少具体数据集、基线对比数值与消融，无法判断各路线相对增益幅度与适用范围边界。
  - 证据缺口：评测协议层面证据薄弱：仅知 MultiHop-RAG 与 MultiTableQA 两个基准的存在与构成，未见关于指标选择（EM/F1 vs 忠实度/证据召回）、token 预算设定、检索器与生成器解耦评测等协议差异的系统性证据。
  - 证据缺口：失败案例证据零散且多为作者自述动机（无关段落误导、错误级联、图投影不完备、跨页碎片化、幻觉），缺少独立的失败模式分类或定量错误分析。
  - 证据缺口：长文档（单篇超长文档）与多文档/开放域语料的界限在检索结果中未被清晰区分，各方法的“长文档”定义与文档长度分布不明。
  - 证据缺口：未检索到针对长文档 RAG 的成本/延迟与准确率权衡的量化证据（GroundedKG-RAG 仅定性提及资源消耗与延迟高）。
- limitations：partial，退出原因 search_budget，检索请求 3 次，缓存命中 0 次，耗时 36.231s
  - 查询：long document QA benchmark evaluation limitations
  - 查询：retrieval augmented generation failure cases hallucination
  - 查询：QUEST-LOFT retrieval evaluation protocol multi-document reasoning
  - 证据缺口：仅有摘要级证据：未能核对任何论文的全文、实验表格、附录或具体指标定义，因此无法确认失败案例的统计口径（如“幻觉率”如何判定）与显著性检验细节。
  - 证据缺口：缺少对长文档 QA 评测协议本身的系统比较：未找到专门审计 LongBench、L-Eval、∞Bench、QUEST 等基准的标注质量、答案等价性判定、以及自动指标与人工判定一致性的研究（仅见 QUEST-LOFT 提到人工评测后数字更新，但摘要未给出差异幅度）。
  - 证据缺口：未找到针对“检索失败 vs 生成失败”归因框架的独立验证研究；VELD-QA 提供了分离定位与解读的标注设计，但其结论是否可迁移到纯文本长文档 QA 尚无证据。
  - 证据缺口：未找到长文档 QA 中检索诱导幻觉的定量复现研究（现有定量证据来自胸部 X 光报告生成这一医学多模态场景，属跨任务外推）。
  - 证据缺口：未找到关于失败案例报告规范（如是否报告拒绝率、检索深度、k 值、上下文预算）的实证审计，SLO 工作仅提出报告规范主张而未做跨论文统计。
  - 证据缺口：未找到反证：即“检索增强在长文档 QA 上稳定优于长上下文”的强证据；现有摘要反而显示差距依赖输出格式、验证步骤与负载，但缺乏统一协议下的对照复现。
  - 证据缺口：未检索到针对多跳/跨文档长文档 QA 中“部分正确”答案如何计分的评测方法研究（摘要层面缺失）。

---
已自动保存到 outputs/20261121-232127-文献综述-Retrieval-augmented-generation-for-long-document-questi.md
# 关于固定扰动上界、Tube MPC 保守性以及“大扰动估计 + 小扰动 Tube”思路的笔记

整理时间：2026-04-03

## 1. 背景问题

在 Tagliabue & How 2024 的线性 RTMPC 部分，作者给定了一个固定的外力扰动上界，并基于该上界离线设计鲁棒管。对他们使用的模型、内环和输入约束，这样做可能是合理的；但当模型、约束或执行器能力发生变化时，同样大小的扰动上界可能会导致：

- 误差不变集 `Z` 变大；
- 约束收紧 `X ⊖ Z`、`U ⊖ KZ` 后的可行域明显变小；
- QP 变得极为保守，甚至无解；
- 即便有解，输入余量也会非常小，跟踪性能显著下降。

这不是你的实现独有的问题，而是 tube MPC 的典型局限之一：扰动集合越大，鲁棒性越强，但保守性和可行性压力也越大。

## 2. 这篇论文自己有没有承认这一点

有，而且比较明确。

Tagliabue & How 在讨论部分提到：

- 他们当前采用的是“easy-to-compute fixed-size approximations of the tube”；
- 未来希望使用“varying cross-sections” 的 tube；
- 对于旋转动力学中的较大不确定性，他们计划结合自适应控制器和更新后的模型/环境估计。

对应你本地论文文本中的位置：

- [papers/tagliabue_how_2024_tube_guided.txt](/home/zxy/work/papers/tagliabue_how_2024_tube_guided.txt#L1808)
- [papers/tagliabue_how_2024_tube_guided.txt](/home/zxy/work/papers/tagliabue_how_2024_tube_guided.txt#L1824)

另外，论文在额外对比里还把 `MPC+DO`（带扰动观测器的 MPC）作为 baseline，说明作者本身就承认“先估计扰动，再把估计结果送入预测/控制”是一条合理路线：

- [papers/tagliabue_how_2024_tube_guided.txt](/home/zxy/work/papers/tagliabue_how_2024_tube_guided.txt#L1266)

结论：你指出的局限并不是在“反驳论文”，而是在抓住论文本身也承认的一个设计边界。

## 3. 相关文献是否讨论了“大扰动集合导致 Tube MPC 过于保守”

结论：是，很多文献都把“保守性”视为 tube MPC 的核心问题之一。

### 3.1 经典 tube MPC 文献

1. D. Q. Mayne, M. M. Seron, S. V. Rakovic, “Robust model predictive control of constrained linear systems with bounded disturbances,” Automatica, 2005.  
   链接：https://doi.org/10.1016/j.automatica.2004.08.019

这篇文章是 tube MPC 的经典来源之一。它的基本结构就是：给定有界扰动集，构造不变误差集，再对约束进行收紧。它没有否认大扰动集的问题，相反，它的整个方法都默认：扰动集越大，误差不变集越大，收紧越强。

2. D. Limon, I. Alvarado, T. Alamo, E. F. Camacho, “Robust tube-based MPC for tracking of constrained linear systems with additive disturbances,” Journal of Process Control, 2010.  
   链接：https://www.sciencedirect.com/science/article/abs/pii/S0959152409002169

这是 tracking 场景下 tube MPC 的代表文献之一，也沿用了“扰动集 -> tube -> 收紧”的基本思想。对你的问题同样适用：如果扰动集放大，tracking tube 和 tightened set 会一起膨胀，性能和可行性都会受影响。

### 3.2 直接讨论 tube MPC 保守性的文献

3. S. V. Rakovic et al., “Fully Parameterized Tube MPC,” IFAC, 2011.  
   链接：https://www.sciencedirect.com/science/article/abs/pii/S1474667016436104

摘要直接指出：tube MPC 高效，但会保守；文章的目标就是降低这种保守性。

这和你现在遇到的现象高度一致：不是 RTMPC 原理错了，而是固定 tube + 固定大扰动上界在当前模型下过于保守。

## 4. 是否有文献支持“先估计/学习大扰动，再让 tube 处理剩余扰动”

结论：有，而且这是比“完全抛弃 tube”更成熟的方向。

### 4.1 disturbance observer + tube MPC

4. X. Zhang et al., “Inverse-Dynamics- and disturbance-Observer-Based tube model predictive tracking control of uncertain robotic manipulator,” Journal of the Franklin Institute, 2023.  
   链接：https://doi.org/10.1016/j.jfranklin.2023.04.005

核心思想：先用逆动力学和扰动观测器减少大部分不确定性，再用 tube MPC 保证剩余误差下的鲁棒跟踪。

5. “Disturbance rejection tube model predictive levitation control of maglev trains,” High-speed Railway, 2024.  
   链接：https://doi.org/10.1016/j.hspr.2024.01.001

摘要直接把方法称为 `DO-TMPLC`，本质就是 disturbance-observer-based tube MPC。

### 4.2 更接近你想法的文献：把不同尺度/类型的不确定性分开处理

6. S. Lucia et al., “A Combined Multi-stage and Tube-based MPC Scheme for Constrained Linear Systems,” IFAC, 2018.  
   链接：https://doi.org/10.1016/j.ifacol.2018.11.043

该文的摘要指出：对小扰动使用较简单的反馈/管方法，对大不确定性使用 multi-stage 方法，从而在保守性和计算量之间取得更好的折中。

这篇文献虽然不是“observer + tube”，但它支持一个很重要的思想：  
**不是所有不确定性都必须由同一种鲁棒机制处理。**

7. “Composite anti-disturbance control with disturbance prediction and utilization: A tube MPC approach,” Automatica, 2026.  
   检索页链接：https://www.sciencedirect.com/science/article/abs/pii/S0005109825005977

从摘要看，这篇文章非常接近你的想法：  
它先通过 observer/prediction 获得扰动估计，并把估计结果显式放入 nominal prediction；然后对剩余误差和未建模部分再构造 tube。

可以把它视为你当前思路的最接近参考之一。

## 5. 是否有“自适应更新扰动集/不确定集，从而减小 Tube 保守性”的文献

结论：有，而且这条路线很成熟。

8. Y. Abdelsalam et al., “Adaptive Tube-Enhanced Multi-Stage Nonlinear Model Predictive Control,” IFAC-PapersOnLine, 2021.  
   链接：https://doi.org/10.1016/j.ifacol.2021.08.244

摘要明确说：在线更新 uncertainty set，得到更紧的 non-falsified set，从而显著改善性能。

9. S. Subramanian et al., “On the Practical Design of Tube-enhanced Multi-stage Nonlinear Model Predictive Control,” IFAC-PapersOnLine, 2022.  
   链接：https://doi.org/10.1016/j.ifacol.2022.07.489

摘要强调这种 tube-enhanced multi-stage 方案能在“较宽范围不确定性”下以更低保守性工作。

10. A. Sasfi, M. N. Zeilinger, J. Kohler, “Robust adaptive MPC using control contraction metrics,” Automatica, 2023.  
    arXiv 链接：https://arxiv.org/abs/2209.11713  
    DOI：https://doi.org/10.1016/j.automatica.2023.111169

摘要明确指出：在线模型自适应能够 reduce conservatism during online operation。

11. M. Taleb Ziabari et al., “Tube-MPC for a class of uncertain continuous nonlinear systems with application to surge problem,” Kybernetika, 2017.  
    链接：https://eudml.org/doc/294824

这篇文章甚至直接处理了“bounded disturbances with unknown upper bound”的情形，使用自适应方法估计不确定性。

## 6. 神经网络/学习方法是否也有人用于“减小固定 tube 的问题”

结论：有，但它们通常不是“完全替代鲁棒界”，而是学习更好的不确定性表示或 tube 动力学。

12. D. D. Fan, A.-A. Agha-mohammadi, E. A. Theodorou, “Deep Learning Tubes for Tube MPC,” NeurIPS Workshop, 2020.  
    链接：https://cseweb.ucsd.edu/~jmcauley/workshops/scmls20/papers/scmls20_paper_10.pdf

这篇文章直接提出学习 time-varying tube / tube dynamics，而不是使用固定、解析上难以刻画但又保守的 tube。

13. D. Lapandic et al., “Meta-Learning Augmented MPC for Disturbance-Aware Motion Planning and Control of Quadrotors,” 2024.  
    链接：https://arxiv.org/abs/2410.06325

这篇文章对四旋翼更接近：用 learned disturbance model 做 disturbance-aware planning，再由带安全界的 tracking controller 保证局部安全。

## 7. 对你当前想法的判断

你的原始想法可以表述为：

> 对较大的风扰，不完全交给固定鲁棒管处理，而是先通过估计器或学习模型显式计算/补偿；其余较小的残差扰动再由鲁棒管处理，从而减小 `Z`，扩大可行控制区间。

这个想法总体上是合理的，而且和上面的 observer-based tube MPC、adaptive tube MPC、combined multi-stage/tube MPC 的思想高度一致。

但需要注意，真正严谨的说法不是：

> “大扰动不用鲁棒管，小扰动才用鲁棒管”

而应该是：

> “可预测或可估计的主导扰动分量先被估计/补偿；tube 只针对剩余残差扰动、估计误差和未建模项来设计。”

原因是：

- 如果你直接把“大扰动”完全交给神经网络，而不给网络误差一个上界，那么严格的鲁棒约束保证就会丢失；
- 如果你能给“估计后的残差”一个较小且可信的上界，那么 residual tube 就能既保留鲁棒性，又显著减小保守性。

所以从控制理论角度，你的方案是合理的；从严格保证角度，它成立的关键在于：

1. 是否能把扰动写成 `d = d_hat + d_res`；
2. 是否能对 `d_res` 给出可信上界；
3. 是否能把估计器/学习器的误差并入该上界。

## 8. 对你当前项目最可落地的建议

如果你的目标是“在当前 Tagliabue 线性 RTMPC 复现基础上，提出一个合理改进”，我建议优先走下面这条路线，而不是立刻把大扰动全交给神经网络：

### 方案 A：MPC / RTMPC + 扰动估计 + residual tube

形式上写成：

- 实际外力扰动：`f_ext`
- 在线估计：`f_hat`
- 残差：`r = f_ext - f_hat`

名义模型里显式补偿 `f_hat`，tube 只针对 `r` 设计。

优点：

- 逻辑清晰；
- 和现有线性 RTMPC 框架兼容；
- 可以直接解释你现在 `gamma_u` 过小的问题；
- 文献支撑最强。

### 方案 B：固定小 tube + 对大扰动用 multi-stage / scenario-tree

如果你希望处理更极端的大风扰，另一条更“鲁棒 MPC 正统”的路线是：

- 小扰动：tube / ancillary feedback
- 大扰动：multi-stage / scenario tree / explicit disturbance branches

这和上面提到的 combined multi-stage and tube-based MPC 非常接近。

### 方案 C：学习扰动模型或学习 tube，但保留 residual bound

如果你确实想引入神经网络，建议不是“NN 完全替代 tube”，而是：

- NN / GP / meta-learning 学习主导风扰或 tube 宽度演化；
- 再给学习误差一个保守上界；
- 最终仍保留一个较小的 residual tube。

这比“纯网络补偿”更适合作为控制论文里的可辩护方案。

## 9. 对你当前代码工作的直接启发

你目前的问题已经很具体：

- 固定 `force_bound_mg = 0.35` 对你当前模型过大；
- `z_half` 随之变大；
- `u_half = |K| z_half` 也变大；
- `U ⊖ KZ` 过小，`gamma_u` 很小甚至接近无解。

这不是单纯“调参没调好”，而是你已经在代码里复现出了文献中 tube MPC 的典型保守性现象。

因此，一个自然的研究推进方向就是：

1. 保留你现有 linear RTMPC 作为 baseline；
2. 加一个外力估计器或简化 disturbance observer；
3. 统计残差扰动边界；
4. 用残差边界重新计算 tube；
5. 对比 `gamma_u`、QP 可行率、tracking error、success rate。

## 10. 当前总结

总结成一句话：

固定大扰动上界在 tube MPC 中确实会带来显著保守性；你提出的“主导大扰动先估计/学习，小扰动和估计误差再交给 residual tube”的思路是合理的，并且有 observer-based tube MPC、adaptive tube MPC、combined multi-stage/tube MPC 等文献支持。真正需要注意的不是“能不能这么做”，而是“估计误差是否还能给出明确上界”，因为这决定了你能否继续保留鲁棒性和可行性保证。

## 11. 期刊/会议影响因子与相关指标说明

说明：

- 下表中的数值是我在 2026-04-03 查到的“当前可公开查到的指标”，不是论文发表当年的历史指标。
- 对期刊，优先记录官方页面给出的 Journal Impact Factor（若官方可见）。
- 对会议、workshop、arXiv 预印本，通常没有官方 Journal Impact Factor，此时标注为“不适用”。
- 对没有公开官方 JIF 的 venue，我补充了可查到的 CiteScore 或 SJR，便于粗略比较，但它们和 JIF 不是同一个指标。

| 文献编号 | Venue | 类型 | 当前可查指标 | 备注 |
|---|---|---|---|---|
| 2 | Automatica | 期刊 | JIF 5.9 | 官方 ScienceDirect 页面可见 |
| 3 | Journal of Process Control | 期刊 | JIF 3.9 | 官方 ScienceDirect 页面可见 |
| 4 | IFAC conference proceedings / IFAC proceedings venue | 会议/会议论文集 | 无官方 JIF | 该类 IFAC 会议论文一般不按期刊 JIF 统计 |
| 5 | IFAC-PapersOnLine | 会议论文集平台 | 无官方 JIF；CiteScore 1.8 | 官方 ScienceDirect 页面未列 JIF，仅列 CiteScore |
| 6 | IFAC-PapersOnLine | 会议论文集平台 | 无官方 JIF；CiteScore 1.8 | 同上 |
| 7 | Automatica | 期刊 | JIF 5.9 | 该文最终发表在 Automatica |
| 8 | Kybernetika | 期刊 | JIF 2.2 | 官方主页当前给出的 2024 JCR IF |
| 9 | Journal of the Franklin Institute | 期刊 | JIF 4.2 | 官方 ScienceDirect 页面可见 |
| 10 | High-Speed Railway | 期刊 | 官方页面未见 JIF；CiteScore 1.7 | ScienceDirect 期刊页仅公开 CiteScore |
| 11 | IFAC-PapersOnLine | 会议论文集平台 | 无官方 JIF；CiteScore 1.8 | 同上 |
| 12 | Automatica | 期刊 | JIF 5.9 | 官方 ScienceDirect 页面可见 |
| 13 | NeurIPS Workshop | workshop 论文 | 无官方 JIF | workshop 通常不按期刊 JIF 统计 |
| 14 | arXiv preprint | 预印本 | 不适用 | arXiv 不是期刊或会议期刊化平台 |

如果你后面要写汇报或论文，建议不要把这些数值直接写成“该论文影响因子是多少”，更严谨的表述应该是：

- “该论文发表于 Automatica，当前官方页面显示 JIF 为 5.9（检索日期：2026-04-03）”
- “该论文发表于 IFAC-PapersOnLine，对应会议论文平台无官方 JIF，官方页面显示 CiteScore 为 1.8”
- “该论文为 arXiv 预印本，不适用期刊影响因子”

## 参考链接

1. Tagliabue & How 2024 本地文本  
   [papers/tagliabue_how_2024_tube_guided.txt](/home/zxy/work/papers/tagliabue_how_2024_tube_guided.txt)

2. Mayne et al. 2005, Robust MPC of constrained linear systems with bounded disturbances  
   https://doi.org/10.1016/j.automatica.2004.08.019

3. Robust tube-based MPC for tracking of constrained linear systems with additive disturbances  
   https://www.sciencedirect.com/science/article/abs/pii/S0959152409002169

4. Fully Parameterized Tube MPC  
   https://www.sciencedirect.com/science/article/abs/pii/S1474667016436104

5. Adaptive Tube-Enhanced Multi-Stage Nonlinear Model Predictive Control  
   https://doi.org/10.1016/j.ifacol.2021.08.244

6. On the Practical Design of Tube-enhanced Multi-stage Nonlinear Model Predictive Control  
   https://doi.org/10.1016/j.ifacol.2022.07.489

7. Robust adaptive MPC using control contraction metrics  
   https://arxiv.org/abs/2209.11713

8. Tube-MPC for a class of uncertain continuous nonlinear systems with application to surge problem  
   https://eudml.org/doc/294824

9. Inverse-Dynamics- and disturbance-Observer-Based tube model predictive tracking control of uncertain robotic manipulator  
   https://doi.org/10.1016/j.jfranklin.2023.04.005

10. Disturbance rejection tube model predictive levitation control of maglev trains  
    https://doi.org/10.1016/j.hspr.2024.01.001

11. A Combined Multi-stage and Tube-based MPC Scheme for Constrained Linear Systems  
    https://doi.org/10.1016/j.ifacol.2018.11.043

12. Composite anti-disturbance control with disturbance prediction and utilization: A tube MPC approach  
    https://www.sciencedirect.com/science/article/abs/pii/S0005109825005977

13. Deep Learning Tubes for Tube MPC  
    https://cseweb.ucsd.edu/~jmcauley/workshops/scmls20/papers/scmls20_paper_10.pdf

14. Meta-Learning Augmented MPC for Disturbance-Aware Motion Planning and Control of Quadrotors  
    https://arxiv.org/abs/2410.06325

15. Automatica 官方指标页  
    https://www.sciencedirect.com/journal/automatica/about/insights

16. Journal of Process Control 官方指标页  
    https://www.sciencedirect.com/journal/journal-of-process-control/about/insights

17. Journal of the Franklin Institute 官方指标页  
    https://www.sciencedirect.com/journal/journal-of-the-franklin-institute/about/insights

18. IFAC-PapersOnLine 官方页面  
    https://www.sciencedirect.com/journal/ifac-papersonline

19. Kybernetika 官方主页  
    https://www.kybernetika.cz/home.html

20. High-Speed Railway 官方页面  
    https://www.sciencedirect.com/journal/high-speed-railway/issues

21. High-Speed Railway SJR 页面  
    https://www.scimagojr.com/journalsearch.php?clean=0&q=21101240975&tip=sid

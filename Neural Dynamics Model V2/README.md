
#### 计算神经动力学模型设计

下面给你一个我认为**最科学、最简洁、最容易让审稿人信服、同时最有机会自然复现实验排序**的方案。
我只给**一个模型**，不列备选。

---

# 最优方案：**PC/RAII 主导的“表面剪切波相干积分 + 频率选择 + 快适应放电”模型**

## 一句话概括

把仿真1输出的皮肤内**时空剪切应力场**，先映射成 **Pacinian/RAII 通道主导的有效机械驱动**，再经过 **表面波传播延迟积分**、**RAII 频率选择滤波**、**快适应脉冲生成**，最后从**群体放电强度**和**群体放电空间收缩度**两个读出量，分别预测实验1里的**强度偏好**和**空间清晰度偏好**。

---

# 为什么我选这个，而不是更复杂的全皮肤四通道模型

## 1) 你的任务频段决定了主导通道

你所有刺激都围绕 **200 Hz**。
对掌侧无毛皮肤，在这个频段，**Pacinian corpuscle / Aβ RAII afferent** 是最有可能主导可觉察强度的通道，这一点有很强文献基础：

- Johnson 2001
- Handler & Ginty 2021
- Talbot / Mountcastle / Bolanowski 系列
- Quindlen 2016
- Ziolkowski 2025

SA1 更偏静态形状/边缘；RAI/Meissner 更偏较低频 flutter。
你这个问题不是“完整触觉全模态重建”，而是“200 Hz 超声调制模式比较”。
所以用**PC/RAII 主导模型**是第一性原理上最合理的降维。

## 2) 你的仿真输入不是位移压入，而是剪切应力时空场

你现在手里最可靠的输入，不是经典触针压入皮肤的 indentation，而是 k-Wave 得到的：

- `tau_xy`
- `tau_xz`
- `tau_yz`
- `tau_roi_steady`

这天然更适合一个以**剪切波传播**为中心的模型，而不是照搬 TouchSim 的整手压入模型。

## 3) 它天然不含“五种方法先验”

这个模型只吃：

- 时空剪切场
- 固定的皮肤表面波速
- 固定的 RAII 频率响应
- 固定的快适应放电机制

它**不知道**输入来自 DLM_2、DLM_3、ULM_L、LM_L 还是 LM_C。
排序如果出来，是机制自己出来的，不是凑出来的。

## 4) 它能自然解释你最关键的现象

这个模型最重要的优点是，它会自然偏好同时满足两件事的刺激：

- **200 Hz 频率保真高**
- **时空波前相干积分强**

这正好对应你论文主线里的：

- Frequency Fidelity
- Spatiotemporal Coherence / Coherent Integration

而这两个条件刚好是：

- **ULM_L** 最强
- **LM_L** 因倍频和方向翻转最差
- **DLM_2** 次强
- **DLM_3 / LM_C** 中等

这不是硬编码，而是模型结构的必然偏好。

---

# 模型总结构

我建议把模型写成 4 个阶段：

## Stage 0. 输入场定义

来自仿真1的稳态 ROI 三维时空数据：

- `tau_roi_steady_xy(x,y,t)`
- `tau_roi_steady_xz(x,y,t)`
- `tau_roi_steady_yz(x,y,t)`
- `roi_x_vec`
- `roi_y_vec`
- `t_vec_steady`

你**不要**把 `FFI`、`DWCI`、`tau_rms`、`tau_peak` 直接喂进模型。
这些应该是**模型外诊断指标**，不是模型输入。
否则审稿人会认为你在用人工设计指标替代机制。

---

## Stage 1. 机械场 → 有效受体驱动场

先把三分量剪切应力张量压缩成一个**等效剪切驱动**：

\[
\tau_{\text{eq}}(x,y,t)=\sqrt{\tau_{xy}^2+\tau_{xz}^2+\tau_{yz}^2}
\]

然后去掉静态偏置：

\[
\tilde{\tau}(x,y,t)=\tau_{\text{eq}}(x,y,t)-\langle \tau_{\text{eq}}(x,y,\cdot)\rangle_t
\]

这里故意只保留**动态部分**，因为：

- RAII/PC 对静态压入几乎不持续响应
- Ziolkowski 2025 也支持“瞬态检测由 inner core 主导”

---

## Stage 2. 表面波相干积分（核心）

这是整个模型最关键的一步。

对每一个虚拟 RAII 受体 \(i\)（位于位置 \(r_i\)），它接收的不是单点值，而是周围区域通过皮肤表面剪切波传播后，在时间上叠加到该点的输入：

\[
m_i(t)=\sum_{j \in \Omega_i} K(d_{ij}) \, \tilde{\tau}(r_j,\, t-d_{ij}/c_s)
\]

其中：

- \(d_{ij} = \|r_j-r_i\|\)
- \(c_s = 5~\text{m/s}\)（直接用你的仿真设定）
- \(K(d)=\exp(-d/\lambda)\)
- \(\lambda\) 取 **4 mm**

解释：

- 这是一个**因果传播积分核**
- 如果某种调制产生的波前在时空上能“对齐”，它们在受体处就会**同相叠加**
- 如果调制导致倍频、相位破坏、方向翻转或旋转分散，则积分效果下降

这一步就是你论文叙事里最想讲的
**Coherent Integration** 的最简实现。

### 为什么这一步科学

它对应三类文献事实：

1. **超声触觉主要通过皮肤表面波/剪切波传播**
   - Reardon 2023
2. **PC/RAII 有很大有效汇聚范围，能整合远处振动**
   - Johnson 2001
   - Saal et al. 2017
3. **PC inner core 和终末结构会放大/选择瞬态与振动输入**
   - Loewenstein & Skalak 1966
   - Quindlen 2016
   - Ziolkowski 2025

---

## Stage 3. RAII 频率选择滤波

对 \(m_i(t)\) 施加一个固定的 RAII 频率响应滤波器 \(h_{\text{PC}}(t)\)：

\[
u_i(t)= (h_{\text{PC}} * m_i)(t)
\]

建议直接实现成一个**二阶带通滤波器**，目标特性：

- 峰值频率：**200 Hz**
- 有效通带：**约 60–450 Hz**
- 对 400/600 Hz 有明显衰减，但不是完全砍掉

这点很重要：
如果你把滤波做得过窄，审稿人会怀疑你是在“专门打压 LM_L”。
真实的 Pacinian 不是只对 200 Hz 响应，而是**在 200 附近最敏感**。

### 这一层为什么必须有

因为你的实验最关键的异常现象之一就是：

- **LM_L 的绝对倍频崩塌**
- **ULM_L 的基频保真优势**
- **DLM_3 的高次谐波污染**

如果没有这层固定的 RAII 频率选择，模型会错误地把“总能量大”直接解释成“更强”，这会和你的实验冲突。

---

## Stage 4. 快适应脉冲生成

最后把 \(u_i(t)\) 送进最简的 leaky integrate-and-fire 神经元：

\[
\tau_m \frac{dV_i}{dt} = -V_i + g\,[u_i(t)]_+
\]

当 \(V_i \ge V_{\text{th}}\) 时发放一个 spike，然后：

- \(V_i \leftarrow 0\)
- 进入绝对不应期 \(t_{\text{ref}}\)

固定参数建议：

- \(\tau_m = 2~\text{ms}\)
- \(t_{\text{ref}} = 2~\text{ms}\)
- \(V_{\text{th}} = 1\)
- \(g\) 用统一归一化增益，不按方法变化

这里我**不建议**再加复杂 Hodgkin–Huxley 或多离子通道。
原因很简单：你没有足够数据约束这些参数，复杂只会让审稿人觉得可疑。
LIF 足够表达“快适应 + 频率锁定 + 群体放电”。

---

# 这个模型里，Lamellar Schwann cell (LSC) 怎么处理？

这是个必须回答的问题，因为 2025 Science Advances 的结果很新，也很关键。

## 处理方式

**不单独建 LSC 细胞模型。**

而是把 LSC 对终末敏感性的增强作用，**吸收到 Stage 2 的有效 inner-core 相干积分核里**。

## 为什么这么做是对的

因为：

1. 你没有人类手掌下 Pacinian inner core 的参数
2. Ziolkowski 2025 证明 LSC 是 mechanosensitive 并能增强 terminal sensitivity，但并没有给出可直接迁移到人类皮肤的完整动力学参数
3. 如果你强行建“LSC + terminal 双细胞耦合 + gap junction + mechanosensitive current”，会立刻过复杂

所以最合理的审稿人友好说法是：

> 本模型将 inner core（包括 afferent terminal 与 lamellar Schwann cell 的联合机械整合作用）视为一个有效时空积分算子，而不显式区分每个细胞组分，以避免在人类参数未知条件下引入不可验证自由度。

这是**科学且克制**的处理。

---

# 输入与输出定义

## 输入

对每个调制条件 \(m\)：

### 必需输入

- `tau_roi_steady_xy` : \((N_x,N_y,N_t)\)
- `tau_roi_steady_xz` : \((N_x,N_y,N_t)\)
- `tau_roi_steady_yz` : \((N_x,N_y,N_t)\)
- `roi_x_vec` : \((N_x,)\)
- `roi_y_vec` : \((N_y,)\)
- `t_vec_steady` : \((N_t,)\)

### 固定生理参数

- `c_s = 5.0 m/s`
- `lambda_space = 4 mm`
- `pc_bandpass_peak = 200 Hz`
- `tau_m = 2 ms`
- `t_ref = 2 ms`

---

## 输出

### 一级输出：神经动力学输出

对每个虚拟受体 \(i\)：

- `spike_times_i`
- `rate_i`
- `vector_strength_i_200Hz`

### 二级输出：群体读出

定义每个受体的**有效振动编码权重**：

\[
w_i = r_i \cdot VS_i(200\text{ Hz})
\]

其中：

- \(r_i\) 是稳态窗口内发放率
- \(VS_i\) 是对 200 Hz 的相位锁定强度

然后生成两个最终分数：

#### 1. 强度分数

\[
S_{\text{intensity}}=\log\left(1+\sum_i w_i\right)
\]

含义：
只有**既发得多、又锁定在目标频率上**的神经活动才贡献主观强度。

#### 2. 清晰度分数

先把 \(w_i\) 映射为空间图 \(W(x,y)\)，再求包含总质量 90% 的最小面积 \(A_{90}\)：

\[
S_{\text{clarity}}=-\log(A_{90})
\]

含义：
在同等有效神经驱动下，空间分布越收缩，感觉越“像一个清晰的点”。

---

# 从模型输出映射到实验1主观结果

你实验1是 2-AFC 配对比较，所以最干净的映射方式不是回归绝对评分，而是直接生成**成对选择概率**。

对于任意方法 \(A\) 与 \(B\)：

\[
P(A \succ B)=\sigma(\hat S_A-\hat S_B)
\]

其中：

- \(\sigma\) 是 logistic 函数
- \(\hat S\) 是对 5 个条件中心化后的 standardized score

你分别对两个任务做：

- 强度任务：用 \(S_{\text{intensity}}\)
- 清晰度任务：用 \(S_{\text{clarity}}\)

然后输出：

- 预测排序
- 预测 pairwise 胜率矩阵
- 与实验 Bradley–Terry 分数的秩相关
- 与实验 pairwise 胜率的 MAE / rank agreement

## 重要说明

这里**不要**为了贴合实验结果，再拟合方法特异参数。
最多允许：

- 一个全局 z-score 标准化
- 一个统一 logistic 映射

这不会引入方法先验，也不会让人觉得你在“用心理物理结果反推模型”。

---

# 这个模型为什么最可能自然复现实验排序

## 对强度排序

### ULM_L 最高

因为它最容易同时满足：

- 延迟积分后波前同相叠加强
- 200 Hz 保真最好
- 快适应 PC 通道持续被有效驱动

### LM_L 最低

因为它天然存在：

- 双向往复导致 400 Hz 成分增强
- 换向造成相位破坏
- delayed coherent integration 被打断

### DLM_2 次强

因为：

- 两点切换可形成较强定时波前叠加
- 基频污染比 DLM_3 少
- 有效 200 Hz 锁定较强

### DLM_3 和 LM_C 中间

因为：

- 它们能形成旋转型或环形波场
- 但能量更分散到多个方向/更复杂的谐波
- 在 RAII 通道里不如单向相干压缩高效

这与你现有结果是对齐的，但不是人为对齐。

---

# 审稿人最容易接受的写法

你在论文里不要说“我们模拟了所有触觉受体”。
这会给自己找麻烦。

你应该写：

> We developed a minimal mechanistically grounded peripheral neural model tailored to the dominant vibrotactile channel engaged by 200-Hz ultrasonic haptics: the Pacinian/RAII pathway. The model converts simulated spatiotemporal shear stress fields into afferent population responses via causal wave-propagation integration, fixed RAII frequency tuning, and fast-adapting spike generation.

核心关键词：

- minimal
- mechanistically grounded
- Pacinian-dominant
- causal wave-propagation integration
- fixed frequency tuning
- fast-adapting spike generation

---

# Python 项目实现架构

下面是我建议的**最简但完整**架构。

```text
swim_neural_model/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── experiment1.yaml
│   └── pacinian_model.yaml
├── data/
│   ├── raw/
│   │   └── experiment1_data.mat
│   ├── processed/
│   └── results/
├── src/
│   └── swim_model/
│       ├── __init__.py
│       ├── io/
│       │   └── load_kwave_mat.py
│       ├── preprocessing/
│       │   ├── shear_equivalent.py
│       │   ├── detrend_and_window.py
│       │   └── receptor_lattice.py
│       ├── mechanics/
│       │   └── coherent_integration.py
│       ├── neural/
│       │   ├── pacinian_filter.py
│       │   ├── lif.py
│       │   └── population_simulator.py
│       ├── readout/
│       │   ├── intensity_score.py
│       │   ├── clarity_score.py
│       │   └── pairwise_prediction.py
│       ├── evaluation/
│       │   ├── compare_bt.py
│       │   └── compare_reaction_time.py
│       └── viz/
│           ├── plot_spike_rasters.py
│           ├── plot_population_maps.py
│           └── plot_pairwise_matrix.py
├── scripts/
│   ├── 01_preprocess_kwave.py
│   ├── 02_run_population_model.py
│   ├── 03_predict_experiment1.py
│   └── 04_make_figures.py
└── tests/
    ├── test_filter.py
    ├── test_delay_kernel.py
    └── test_readout.py
```

---

# 各模块职责

## `load_kwave_mat.py`

读取你 MATLAB 导出的 `.mat`：

- `results[m].tau_roi_steady_xy`
- `results[m].tau_roi_steady_xz`
- `results[m].tau_roi_steady_yz`
- `results[m].roi_x_vec`
- `results[m].roi_y_vec`
- `results[m].t_vec_steady`

输出统一 Python dict。

---

## `shear_equivalent.py`

实现：

```python
tau_eq = np.sqrt(tau_xy**2 + tau_xz**2 + tau_yz**2)
tau_dyn = tau_eq - tau_eq.mean(axis=-1, keepdims=True)
```

---

## `receptor_lattice.py`

在 ROI 上生成一组虚拟 RAII 受体位置。
建议用规则网格，间距 2 mm，边缘外扩 0，不做随机采样。

原因：

- 可重复
- 不引入采样噪声
- 只用于相对比较，足够

---

## `coherent_integration.py`

对每个 receptor 计算传播延迟积分：

```python
m_i(t) = sum_j exp(-d_ij/lambda_) * tau_dyn[j, t - d_ij/cs]
```

实现时用：

- 预计算距离矩阵
- 预计算 delay index
- 用 numpy / numba 向量化

---

## `pacinian_filter.py`

实现固定 RAII 滤波器。
推荐用 `scipy.signal.butter` 或 `iirpeak` 做一个固定带通。

---

## `lif.py`

标准 LIF：

```python
dv = (-v + gain * np.maximum(u, 0)) * dt / tau_m
if v >= 1:
    spike
    v = 0
    refractory = ref_steps
```

---

## `population_simulator.py`

对 5 个调制方法分别跑：

- Stage 1 preprocessing
- Stage 2 coherent integration
- Stage 3 RAII filter
- Stage 4 LIF spiking

输出每个方法的：

- spike times
- firing rate map
- vector strength map

---

## `intensity_score.py`

```python
w_i = rate_i * vector_strength_i
S_intensity = np.log1p(w.sum())
```

---

## `clarity_score.py`

```python
W = spatial_map_of(w_i)
W = gaussian_filter(W, sigma=1)
A90 = minimal_area_containing_90_percent_mass(W)
S_clarity = -np.log(A90)
```

---

## `pairwise_prediction.py`

```python
score = (score - score.mean()) / score.std()
p_ab = 1 / (1 + np.exp(-(score_a - score_b)))
```

---

# README 里应该怎么写

你 README 最核心只写 4 件事：

## 1. 模型目标

从 k-Wave 时空剪切场预测：

- 外周 RAII/PC 样群体放电
- 实验1中的主观强度与空间清晰度排序

## 2. 模型假设

- 200 Hz 超声触觉主要由 Pacinian/RAII 通道主导
- inner-core 作用等效为 causal spatiotemporal integration kernel
- fixed RAII frequency tuning
- no method-specific parameters

## 3. 输入

MAT 文件中的 `tau_roi_steady_xy/xz/yz` 和坐标/时间向量

## 4. 输出

- receptor spike trains
- intensity score
- clarity score
- pairwise preference matrix

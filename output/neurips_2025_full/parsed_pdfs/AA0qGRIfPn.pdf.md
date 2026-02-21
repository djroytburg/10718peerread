## Why 1 + 1 &lt; 1 in Visual Token Pruning: Beyond Naïve Integration via Multi-Objective Balanced Covering

Yangfu Li ♣ , Hongjian Zhan ♣ , ♡ , ∗ , Tianyi Chen ♠ , Qi Liu ♣ , Yu-jie Xiong ♢ , Yue Lu ♣

♣ East China Normal University, ♠ Shanghai Jiao Tong University ♡ Chongqing Institute of East China Normal University ♢ Shanghai University of Engineering Science {yfli\_cee, qiliu}@stu.ecnu.edu.cn, hjzhan@cee.ecnu.edu.cn guleimurray@sjtu.edu.cn, xiong@sues.edu.cn, ylu@cs.ecnu.edu.cn

## Abstract

Existing visual token pruning methods target prompt alignment and visual preservation with static strategies, overlooking the varying relative importance of these objectives across tasks, which leads to inconsistent performance. To address this, we derive the first closed-form error bound for visual token pruning based on the Hausdorff distance, uniformly characterizing the contributions of both objectives. Moreover, leveraging ϵ -covering theory, we reveal an intrinsic trade-off between these objectives and quantify their optimal attainment levels under a fixed budget. To practically handle this trade-off, we propose Multi-Objective Balanced Covering (MoB), which reformulates visual token pruning as a bi-objective covering problem. In this framework, the attainment trade-off reduces to budget allocation via greedy radius trading. MoB offers a provable performance bound and linear scalability with respect to the number of input visual tokens, enabling adaptation to challenging pruning scenarios. Extensive experiments show that MoB preserves 96.4% of performance for LLaVA-1.5-7B using only 11.1% of the original visual tokens and accelerates LLaVA-Next-7B by 1.3-1.5 × with negligible performance loss. Additionally, evaluations on Qwen2-VL and Video-LLaVA confirm that MoB integrates seamlessly into advanced MLLMs and diverse vision-language tasks. The code is available at https://github.com/YChenL/MoB .

## 1 Introduction

Multimodal large language models (MLLMs) have shown impressive performance across a variety of vision-language tasks, including visual understanding [30, 27, 20], visual question answering [40, 16, 37], and visual-language reasoning [9, 47, 45]. Since visual data exhibits much higher spatial redundancy than language, MLLMs are typically required to encode visual inputs as numerous tokens, resulting in substantial computational overhead.

To address this issue, visual token pruning methods are proposed to accelerate MLLMs by selecting representative subsets of visual tokens. Most pruning methods focus on two distinct objectives: Visual Preservation (VP) [6, 8, 59, 46], which retains tokens by minimizing redundancy or maximizing visual salience, and Prompt Alignment (PA) [58, 51, 48], which selects tokens most relevant to the prompt. Recently, several multi-objective approaches [31, 51, 42] have been proposed to integrate VP and PA through various complex strategies. Counterintuitively, these methods do not exhibit dominant superiority compared to single-objective approaches, as shown in Figure 1(a). This observation naturally raises a question: Does integrating different objectives offer fundamental advantages?

∗ Corresponding Author.

Figure 1: (a) Comparison of single- vs. bi-objective pruning methods on LLaVA-1.5-7B at a 66 . 7% pruning rate; (b) distribution of the prompt-visual coupling, revealing two distinct patterns across various tasks: weak coupling (large distance) and strong coupling (small distance); (c) radar charts of LLaVA-1.5-7B with visual tokens reduced from 576 to 192 , 128 , and 64 ( left-to-right ), demonstrating the consistent improvements of MoB across 10 well-recognized benchmarks.

<!-- image -->

Inspired by this question, we formulate preservation using the Hausdorff distance between the original and pruned token sets and derive the first closed-form error bound for visual token pruning (Lemma 1). This bound depends on VP and PA, while it is also affected by a prompt-visual coupling, measured by the Hausdorff distance between prompt and visual tokens. Notably, we identify two patterns of this coupling across popular benchmarks, as presented in Figure 1(b): weak coupling with large distance ( e.g. , TextVQA, POPE) and strong coupling with small distance ( e.g. , MMB, VizWiz). Our further analysis reveals that the effectiveness of the pruning objectives varies under distinct coupling patterns (Lemma 2). However, existing multi-objective methods overlook this variation and integrate VP and PA via constant strategies, yielding inconsistent improvements over single-objective baselines.

To quantify the effect of prompt-visual coupling, we reexamine visual token pruning from a geometric covering perspective. In this view, the retained tokens can be thought of as the union of two disjoint covers for prompt and visual tokens, where each objective corresponds to a Hausdorff covering radius, and the prompt-visual coupling is represented by the inter-cover diameter. By analyzing the geometric relationship between the radii and the diameter, we reveal an intrinsic trade-off between the two objectives (Theorem 1), which identifies the optimal attainment level of each objective to achieve the performance ceiling under a fixed pruning budget and prompt-visual coupling.

For a practical solution to this trade-off, we propose Multi-objective Balanced Covering (MoB), a training-free visual token pruning method with provable performance guarantees and multilinear complexity (Theorem 2). MoB partitions the retained tokens into two disjoint subsets for PA and VP, employing greedy radius-trading strategies to reduce the trade-off in objective attainment to a budget allocation problem. This allows MoB to achieve the optimal balance under each coupling pattern by selecting appropriate subset sizes. As shown in Figure 1(c), MoB consistently outperforms both single-objective and multi-objective baselines by a clear margin at identical pruning rates. Besides, MoB accelerates LLaVA-Next-7B by 1 . 3 -1 . 5 × with negligible performance loss. Ablation studies further validate our theoretical analysis. Our key contributions are summarized as follows:

- ❶ To our knowledge, we present the first closed-form error bound for visual token pruning and its practical relaxation, characterizing the contributions of the two objectives to preservation quality.
- ❷ We quantify the trade-off between the objectives and identify their optimal attainment level under a fixed budget and prompt-visual coupling, offering valuable insights into visual token pruning.
- ❸ Wepropose Multi-objective Balanced Covering (MoB) for training-free visual token pruning, which reduces the trade-off of objective attainment to a budget allocation problem via two greedy radiustrading strategies, yielding both a provable performance guarantee and multilinear scalability.

- ❹ Extensive experiments across 14 public benchmarks demonstrate the superiority of MoB. For instance, it retains 96 . 4% and 97 . 9% performance for LLaVA-1.5-7B and Video-LLaVA-7B with an 88 . 9% reduction ratio, outperforming the second-best method by 2 . 7% and 1 . 6% , respectively. MoBcan also be readily incorporated into advanced MLLMs, such as LLaVA-Next and Qwen2-VL.

## 2 Background

## 2.1 Related Work

Multimodal Large Language Model (MLLM). MLLMs [30, 21, 60, 28] have achieved remarkable progress in vision-language reasoning, owing to their robust cross-modality modeling via attention mechanisms [43, 34]. However, the spatial redundancy inherent in visual signals typically leads to a large number of input tokens [25, 22, 29, 44], particularly in high-resolution images and multi-frame videos ( e.g. , 2048 tokens in Video-LLaVA [27]). This issue exacerbates the quadratic scaling problem of attention mechanisms, posing significant computational challenges. Moreover, to further enhance the visual capability by incorporating high-quality details, advanced MLLMs are now designed to support higher resolution images [24, 11, 10, 4], thereby necessitating the processing of even more visual tokens ( e.g. , 2880 tokens in LLaVA-NEXT [29]). In these scenarios, effectively selecting representative visual tokens becomes a critical requirement for the real-world application of MLLMs.

Visual Token Pruning. Due to the spatial redundancy, inputs to MLLMs contain numerous less informative visual tokens. Visual token pruning accelerates MLLMs by selectively retaining only the most critical tokens during inference. Existing methods typically focus on either visual preservation (VP) [6, 38, 8, 52, 57, 32, 46] or prompt alignment (PA) [58, 51, 48]. VP-driven methods, such as ToMe [6] and LLaVA-PruMerge [38], reduce redundancy by merging similar tokens, while FastV [8] and FasterVLM [57] select tokens based on visual salience. PA-driven approaches like SparseVLM [58] rely on cross-modal attention to identify prompt-relevant tokens. More recently, MustDrop [31] integrates VP and PA through a multi-stage pruning pipeline, reporting notable improvements. Despite these advances, existing methods largely overlook the varying relative importance of VP and PA across different scenarios. In this paper, we formally characterize the contribution of each objective under a fixed pruning budget, and propose an algorithm that balances these objectives per scenario, yielding consistent improvements across diverse pruning conditions.

## 2.2 Preliminaries

Pipeline of MLLM. MLLMs perform vision-language reasoning by jointly processing multimodal inputs in a shared representation space. Formally, given visual tokens V (1) extracted from the visual inputs and prompt tokens P (1) encoded from user prompts, the multimodal input is defined as

<!-- formula-not-decoded -->

where N and L denote the numbers of visual and prompt tokens, respectively. We regard both V (1) and P (1) as compact sets on d -dimensional Euclidean space ( R d , ∥ · ∥ ) . The input X (1) is then fed into a language model F [1 ,I ] with I transformer block, and the final output is given by

<!-- formula-not-decoded -->

In particular, each f ℓ follows the standard Transformer ( e.g. , multi-head self-attention [43], layer normalization [3, 50]). The intermediate feature for any layer ℓ ∈ { 2 , . . . , I } is defined as

<!-- formula-not-decoded -->

with V ( ℓ ) and P ( ℓ ) representing the visual and prompt tokens after ℓ -1 layers, respectively.

Visual Token Pruning. To accelerate MLLMs with minimal performance loss, visual token pruning selectively removes less-informative visual tokens at chosen intermediate layers of the language model F [1 ,I ] . Specifically, for any chosen layer f ℓ , ℓ ∈ { 2 , . . . , I } , pruning algorithms first select a subset S ( ℓ ) ⊆ V ( ℓ ) of size K ( i.e. , pruning budget) and form the pruned input X ( ℓ ) s = S ( ℓ ) ⊔ P ( ℓ ) . The corresponding output before and after pruning are then defined as

<!-- formula-not-decoded -->

Table 1: Summary of notation used in the theoretical framework.

| Notation                                                                                                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Notation                                                                                                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ℓ, I d P ( ℓ ) X ( ℓ ) N, L F [1 , I ] y, y s C ℓ S p , S v ϵ p N ( X , ϵ ) a, b ϵ 0 V δ , P δ z D 2 k α ( η,k,L Θ( · ) | Pruning layer index; Final layer index. Embedding dimension. Prompt tokens at layer ℓ , i.e. , P ⊆ R d . All tokens at layer ℓ , i.e. , X = V ⊔P . #visual tokens, &#124;V&#124; = N ; #prompt tokens, &#124;P&#124; = L . Full model (layers 1 . . . I ). Outputs with full tokens X / pruned tokens X s . Lipschitz constant of F [ ℓ, I ] w.r.t. d H . Prompt center / Visual center set, S = S p ⊔ S v . Covering radius for P , d H ( S p , P ) . Covering number of X at radius ϵ . Covering-number lower/upper constants for P . Validity radius for covering bounds. δ -dilation of V and P . Radius scaling factor ( > 1 ). Trade-off constant 1 /z 2 . Fold for the proposed nearest-neighbor covering. Alignment constant η ( bkL/a ) 1 /d eff . Asymptotically equal (same order); i.e. , ∃ c 1 , c 2 > 0 , n 0 : c 1 g ( n ) ≤ f ( n ) ≤ c 2 g ( n ) for n ≥ n 0 . | f ℓ V ( ℓ ) S ( ℓ ) X ( ℓ ) s K F [ ℓ, I ] d H ( A , B η K p , K v ϵ v d eff a ′ , b ′ δ B ( c, ϵ ) D 1 ϵ ∗ S ′ p β Ω( · ) | Transformer block at layer ℓ . Visual tokens at layer ℓ , i.e. , V ⊆ R d . Retained visual tokens at layer ℓ , i.e. , S ⊆ V . Retained tokens at layer ℓ , i.e. , X s = S ⊔P . Pruning budget, &#124;S&#124; = K . Submodel from layer ℓ to I . Hausdorff distance between sets A and B . Visual-prompt coupling bound: d H ( V , P ) ≤ η . Budgets for S p and S v , K = K p + K v . Covering radius for V , d H ( S v , V ) . Effective dimension of V , P . Covering-number lower/upper constants for V . Small dilation radius ( δ ≪ η ). Ball { x : ∥ x - c ∥ ≤ ϵ } . Trade-off constant (4 aa ′ ) 1 /d eff . Optimal radius max { η/z, √ D 1 K - 1 /d eff } . Candidate set before final truncation by K p . Preservation constant 2 ( b ′ ) 1 /d eff . Asymptotic lower bound (at least on the order of g i.e. , ∃ c > 0 , n 0 : f ( n ) ≥ c g ( n ) for n ≥ n 0 . |

Finally, the objective of visual token pruning is formulated as

<!-- formula-not-decoded -->

Notation. For brevity we omit the layer index ( ℓ ) and simply write X = V ⊔ P and X s = S ⊔ P to denote the input and its pruned counterpart at an arbitrary layer f ℓ . We use F to denote any composition mapping of the full model F [1 ,I ] . Finally, we let ∥ · ∥ denote the Euclidean norm.

## 3 Methodology

## 3.1 Revisiting Visual Token Pruning: Insights into Prompt-Visual Coupling

As shown in Fig. 1(a), multi-objective pruning methods fail to achieve the expected improvements, and objective-specific methods exhibit inconsistent performance across benchmarks. These observations motivate us to reexamine the problem of visual token pruning. We begin by introducing Assumption 1, which quantifies pruning performance in terms of the preservation of the original token set.

Assumption 1 (Lipschitz Continuity w.r.t. the Hausdorff Distance) . Assume every partial composition F (from layer ℓ to I ) of the language model is Lipschitz continuous w.r.t. the Hausdorff distance with constant C ℓ ≥ 1 . Formally, for any intermediate token sets X , X s ⊂ R d ,

<!-- formula-not-decoded -->

where d H is the Hausdorff distance induced by the Euclidean norm:

<!-- formula-not-decoded -->

Subsequently, we measure the preservation of the original token set X using three pairwise distances among visual tokens V , retained tokens S , and prompt tokens P , thereby establishing a unified performance bound for various visual token pruning algorithms, as presented in Lemma 1.

Lemma 1 (An Error Bound for Visual Token Pruning) . Under Assump 1, given a token set with its pruned counterpart X = V ⊔ P , X s = S ⊔ P ⊆ R d , the pruning error bound is given by:

<!-- formula-not-decoded -->

Remark. Here d H ( S , P ) and d H ( S , V ) describe the prompt alignment and visual preservation, while d H ( V , P ) is an inherent term that describes the prompt-visual coupling of input data.

/84

/85

/84

/86

/87

/86

/87

/88

/89

/i255

/88

/91

/92

/86

/89

/41

/39

/0/1 /2 /3 /i255 /5 /6 /7 /8 /i255 /8 /6 /9 /10 /i255 /6 /11 /9 /i255 /6 /12 /i255 /3 /1 /9 /i255 /1 /6 /3 /13 /2 /14 /14 /6 /6 /11 /10 /i255 /12 /9 /2 /3 /15 /7 /9 /16 /42 /18 /43 /36 /44 /45 /46 /49 /10 /i255 /3 /1 /9 /7 /9 /i255 /2 /i255 /12 /7 /50 /10 /13 /9 /9 /i255 /50 /11 /i255 /3 /1 /9 /i255 /50 /38 /2 /31 /9 /16 /0/1 /9 /7 /9 /i255 /50 /10 /i255 /50 /3 /16 /51/52 /2 /7 /50 /10 /53 /1 /2 /11 /31 /1 /2 /50 /54 /i255 /55 /56 /57 /58 /59 /60 /61 /62 /58 /57 /63 /57 /64 /57 /65 /62 /66 /59 /i255 /68 /62 /59 /61 /69 /70 /57 /62 /63 /i255 /59 /69 /57 /i255 /71 /57 /58 /59 /60 /61 /62 /58 /57 /i255 /68 /62 /59 /61 /69 /72 /73 /58 /59 /62 /66 /59 /i255 /74 /63 /75 /76 /i255 /59 /69 /57 /i255 /71 /57 /58 /59 /60 /61 /62 /58 /57 /i255 /68 /62 /59 /61 /69 /77 /75 /63 /58 /59 /60 /61 /62 /58 /57 /63 /57 /64 /57 /65 /62 /66 /59 /i255 /68 /62 /59 /61 /69 /0/1 /50 /78 /1 /i255 /78 /6 /15 /11 /3 /7 /79 /i255 /50 /10 /i255 /1 /50 /31 /1 /14 /50 /31 /1 /3 /9 /8 /16 /51/80 /2 /38 /2 /50 /78 /2 /81 /2 /50 /82 /54 /i255 /55 /83 /57 /64 /57 /65 /62 /66 /59 /i255 /68 /62 /59 /61 /69 Figure 2: Illustration of prompt-visual coupling with two distinct patterns: In fine-grained tasks ( e.g. POPE), only a few patches are critical, so the worst-case patch lies far from best-case ones, resulting in a large Hausdorff distance and making prompt alignment valuable. In coarse-grained tasks ( e.g. MMB), many relevant patches contain the answer cues; thus, the worst-case patch remains close to best-case ones, yielding a small Hausdorff distance and making visual preservation more efficient.

/17 /18

/19

/20

/i255

/22

/23

/24

/25

/26

/27

/28

/29 /30

/14

/2 /7

/31

/9 /32

/33

/34

/35

/36

/37

/23

/28

/29

/i255

/22

/23

/24

/25

/26

/27

/28

/29 /30

/10

/38 /2

/14

/14

/32

/33

/34

/40 /39

<!-- image -->

Proof in Appendix E.1. By Lemma 1, in practical settings where |S| ≪ |V| , pruning performance is governed by a non-trivial interaction among visual preservation, prompt alignment, and prompt-visual coupling. However, existing multi-objective methods typically overlook the coupling term d H ( V , P ) and statically combine the two objectives across tasks, limiting their effectiveness. Our empirical evidence across popular benchmarks validates two distinct patterns of d H ( V , P ) , each favoring different pruning objectives, as shown in Figure 2. To further explicate the effect of prompt-visual coupling, we introduce Assumption 2 and propose a practical relaxed error bound in Lemma 3.

Assumption 2 (Prompt-Visual Coupling Bound) . We assume the input visual data and prompts are not entirely unrelated; hence, there exists a constant η &gt; 0 for any intermediate token set X = V ⊔ P ⊆ R d such that d H ( V , P ) ≤ η , ensuring the reasonability of vision-language reasoning.

Lemma 2 (A Relaxed Error Bound under Practical Budgets) . Under Assumptions 1 and 2, let X = V ⊔ P , X s = S ⊔ P ⊆ R d with |S| = K ≪ N . Partition the retained token set S into two disjoint subsets: S = S p ⊔ S v , devoted to prompt alignment d H ( S p , P ) and visual preservation d H ( S v , V ) , respectively. Then, the pruning error bound reduces to

<!-- formula-not-decoded -->

Proof in Appendix E.2. As Lemma 2 indicates, under weak coupling (large η ), most visual regions are distant from prompt tokens in the semantic space. Consequently, if S p misses the critical patches, d H ( S p , P ) dominates the pruning error, making the selection of S p i.e. , prompt alignment, more significant. Conversely, under strong coupling (small η ), d H ( S p , P ) tends to decrease in tandem with d H ( S v , V ) , reducing the marginal benefit of prompt alignment. To further guide pruning methods design, we next quantify this trade-off governed by η through an ϵ -covering argument.

## 3.2 Quantifying Prompt-Visual Trade-Off: A Geometric Covering Perspective

We first introduce some geometric metrics in Definition 1, recasting each objective term d H ( S p , P ) and d H ( S v , V ) as covering radii and the coupling term d H ( V , P ) as an inter-cover diameter. Next, we relate each recasted objective to its token budget |S p | , |S v | via covering regularity in Lemma 3. Finally, by loading the budget constraint and applying the triangle inequality between radii and diameter, we derive a quantitative trade-off jointly governed by K and η in Theorem 1.

Definition 1 ( ϵ -cover, Covering Number, and Covering Regularity) . Let ( R d , ∥ · ∥ ) be the d -dimensional Euclidean space and let X ⊆ R d be a compact set.

- (a) ϵ -cover. if there exists a finite set C = { c 1 , . . . , c M } ⊂ R d , an ϵ -cover of X is given by

<!-- formula-not-decoded -->

where C is the collection of covering centers, and ϵ is the covering radius.

- (b) Covering number. The minimum cardinality of C is the covering number of X at radius ϵ :

<!-- formula-not-decoded -->

/93

/94

/95

/84

/96

/97

/98

/95

- (c) Covering regularity. We say that X satisfies d -dimensional covering regularity if there exist constants 0 &lt; A ≤ B and ϵ 0 &gt; 0 such that

<!-- formula-not-decoded -->

Based on Definition 1(a) (b), S p , S v ⊆ V can be thought of as two collections of centers such that

<!-- formula-not-decoded -->

where the radii are given by ϵ p := d H ( S p , P ) , ϵ v := d H ( S v , V ) , and the covering numbers satisfy N ( P , ϵ p ) ≤ |S p | , N ( V , ϵ v ) ≤ |S v | . Thereby, we derive a lower bound of the required budget, i.e. , |S p | , |S v | , to improve each objective, i.e. , ϵ p , ϵ v , based on d eff -dimensional covering regularity.

Lemma 3 (Covering Number Bounds) . Given P , V ⊂ R d with an effective dimension d eff . Suppose their δ -dilations V δ := ⋃ v ∈V B ( v, δ ) , P δ := ⋃ p ∈P B ( p, δ ) ( δ ≪ η ) satisfy d eff -dimensional covering regularity; thus, there exist constants b&gt;a&gt; 0 , b ′ &gt;a ′ &gt; 0 and ϵ 0 &gt;δ such that

<!-- formula-not-decoded -->

Remark. Previous work suggests that both visual and language embeddings concentrate on a low-dimensional manifold, so the effective covering dimension satisfies the typical relation d eff ≪ d .

Proof in Appendix E.3. Lemma 3 demonstrates that once the radius ( i.e. , the objective) falls below ϵ 0 , any further improvement of it demands a Θ( ϵ -d eff ) increase in the number of selected token.

By loading Lemma 3 into the budget constraint: |S p | + |S v | = K , and applying a two-step triangle inequality between the covering radii ϵ p , ϵ v and the inter-cover diameter η , we establish a K -η -bound in Theorem 1(b), which quantifies the trade-off governed by the budget and prompt-visual coupling.

Theorem 1 (Trade-off between Prompt Alignment and Visual Preservation) . Under Assumption 2 and the covering-regularity hypothesis of Lemma 3 with constants a, a ′ , d eff &gt; 0 , there exist a radius-scaling factor z &gt; 1 such that η/z &gt; δ and K &lt; N ( P , η/z ) + N ( V , η/z ) , for every pruning results S = ( S p ⊔ S v ) ⊆ V with budget K satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark (Optimal Attainment Level) . D 1 K -2 /d eff is completely determined by the pruning budget, while D 2 η 2 quantifies the effect of prompt-visual coupling. The optimal attainment level per objective is given by ϵ ∗ = max { η/z, √ D 1 K -1 /d eff } . Any attempt to reduce one objective below ϵ ∗ forces the other above ϵ ∗ , thereby increasing the overall pruning error.

Remark (Effect of Budget and Coupling Strength) . As K decreases, z correspondingly shrinks, ultimately making D 2 η 2 dominate the bound; while as K increases, both of the terms reduce, thereby diminishing the trade-off and tightening the overall error bound.

Proof in Appendix E.4. Theorem 1 characterizes the optimal attainment level for each objective under a fixed pruning budget and prompt-visual coupling. However, it is actually very challenging to dynamically determine the attainment level per objective during the pruning process. To address this, we propose Multi-objective Balanced Covering, which leverages the monotonic relationship between covering radii and numbers to reduce the trade-off of attainment to a budget-allocation problem.

## 3.3 Multi-Objective Balanced Covering: From Trade-Off to Budget Allocation

Motivated by the insights in §3.2, Multi-objective Balanced Covering (MoB) recasts visual token pruning as bi-objective covering. Specifically, given a token set X = V ⊔ P ⊆ R d with a budget K , the retained token set S is defined as the union of a prompt center set S p and a visual center set S v :

<!-- formula-not-decoded -->

MoB then selects the cover centers ( i.e. , retained tokens) by minimizing the overall maximum radius:

<!-- formula-not-decoded -->

In practice, MoB solves this problem approximately by two sequential greedy covering procedures: selection of prompt center set S p with budget K p , and selection of visual center set S v with the remaining budget K -K p . By the covering number bounds given in Lemma 3, we have

<!-- formula-not-decoded -->

where d eff is the effective dimension of V , P . Accordingly, by selecting the unique budget K p ( i.e. , fixing the remaining budget K -K p ) under each coupling pattern, MoB ensures ϵ p , ϵ v = Ω ( max { η/z, √ D 1 K -1 /d eff } ) , thus yielding provable performance guarantees across scenarios.

Normalization. For efficiency, MoB applies L2 normalization to each x ∈ X so that ∥ x ∥ = 1 . Hence, for any token pair x 1 , x 2 ∈ X , the Euclidean distance can be induced by their cosine similarity:

<!-- formula-not-decoded -->

Selection of Prompt Center Set S p . Since all s p ∈ V lie outside P , a typical solution for minimizing the radius ϵ p is Nearest-Neighbor covering (NN covering) [15], which uniformly allocates the nearest s p ∈ V for each prompt token. However, the contribution of each prompt token is inequivalent, especially under weak prompt-visual coupling; thus, equal allocation risks missing the 'best-case tokens.' To remedy this, we introduce a k -fold NN covering procedure. Formally, let L = |P| and k &gt; 1 be a hyperparameter; we first utilize a temporary budget of kL to form a candidate set.

<!-- formula-not-decoded -->

thereby over-sampling the k nearest visual tokens for each prompt token. Subsequently, we refine the candidate set by selecting the final K p centers that maximize their worst-case alignment with P :

<!-- formula-not-decoded -->

By concentrating the limited budget on those visual tokens most strongly aligned with the key prompt tokens, this strategy ensures a better preservation of the critical regions in the visual input. We determine the appropriate k by ablation to avoid the oversampling of a few salient prompt tokens.

Selection of Visual Center Set S v . Unlike the prompt center selection, each visual center s v lies in V . Thereby, we employ Farthest Point Sampling (FPS) [36] on the remaining tokens, i.e. , V \ S , to select the visual centers, which makes the visual centers S v well-spread over V , minimizing the covering radius ϵ v . Concretely, FPS operates by iteratively selecting the token farthest ( i.e. , the most different) from the current centers S , where the distance is given by

<!-- formula-not-decoded -->

Subsequently, we initialize the visual centers with the empty set, i.e. , S (1) v := ∅ . We then successively add the farthest visual token to the current centers S ( i ) v ⊔ S p until it contains a total of K elements. Hence, the visual centers at the subsequent iteration, S ( i +1) v , is given by:

<!-- formula-not-decoded -->

More details of the proposed MoB algorithm are provided in Appendix B.

Theorem 2 (Performance Guarantee) . Under Assump 1 and the covering-regularity of Lem 3 with consts a, a ′ , d eff &gt; 0 and b&gt;a, b ′ &gt;a ′ , for any budget split ( K p , K -K p ) , covering fold k , and token set X = V ⊔ P ⊆ R d with |V| = N , |P| = L , d H ( V , P ) ≤ η , the following hold: (a) Performance bound: The Performance degradation caused by MoB is upper bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (b) Multilinear complexity: The complexity of MoB is given by T MoB = O ( N ( L + K ) d ) .

Remark (Coupling Trade-off) . Under weak coupling (large α ), minimizing the bound requires a larger K p . Conversely, under strong coupling (small α ), the alignment term decays rapidly, favoring visual preservation (increasing K -K p ). Specially, under perfect coupling ( η = 0 ), the bound simplifies to ∥ ∆ y ∥ ≤ C ℓ β ( K -K p ) -1 /d eff , i.e. , MoB reduces to pure visual preservation.

Remark (Budget Scaling) . As the budget K increases, the preservation term β ( K -K p ) -1 /d eff decays, requiring a corresponding increase in K p (and thus a reduction in the alignment term) to re-balance the trade-off and further lower the overall error bound.

Remark (Scalability) . MoB exhibits a multilinear scalability w.r.t #visual tokens N , #prompt tokens L , and #retained tokens K ( K,L ≪ N ), making it easily adaptable to more challenging scenarios involving large token counts, e.g. , higher-resolution inputs or multi-frame video.

Proof in Appendix E.5.

| Method                     | Objectives   | Strong Coupling   | Strong Coupling   | Strong Coupling   | Strong Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Avg.   |
|----------------------------|--------------|-------------------|-------------------|-------------------|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|--------|
|                            |              | MMB               | MMB CN            | SQA               | VizWiz            | GQA             | MME             | POPE            | VQA T           | VQA V2          | OCR             | Avg.   |
| LLaVA-1.5-7B               |              |                   | w/o               | Pruning,          | N = 576           | ; Token         | Reduction       | Rate            | = 0.0%          |                 |                 |        |
| Vanilla [28]               | -            | 64.7              | 58.1              | 69.5              | 50.0              | 61.9            | 1862            | 85.9            | 58.2            | 78.5            | 297             | 100%   |
| LLaVA-1.5-7B               |              |                   | Pruning           | budget            | K =               | 192 ; Token     | Reduction       | Rate            | = 66.7%         |                 |                 |        |
| FastV (ECCV'24) [8]        | VP           | 61.2              | 57.0              | 67.3              | 50.8              | 52.7            | 1612            | 64.8            | 52.5            | 67.1            | 291             | 91.2%  |
| SparseVLM (ICML'25) [58]   | PA           | 62.5              | 53.7              | 69.1              | 50.5              | 57.6            | 1721            | 83.6            | 56.1            | 75.6            | 292             | 96.3%  |
| MustDrop (24.11) [31]      | PA VP        | 62.3              | 55.8              | 69.2              | 51.4              | 58.2            | 1787            | 82.6            | 56.5            | 76.0            | 289             | 97.2%  |
| DART (EMNLP'25) [46]       | VP           | 63.6              | 57.0              | 69.8              | 51.2              | 60.0            | 1856            | 82.8            | 57.4            | 76.7            | 296             | 98.8%  |
| MoB (w/o η -prior)         | PA VP        | 63.8              | 57.5              | 70.0              | 52.4              | 61.2            | 1858            | 84.5            | 58.2            | 77.9            | 304             | 100.2% |
| + η -prior                 | -            | 64.1              | 57.8              | 70.1              | 52.5              | 61.4            | 1860            | 84.8            | 58.5            | 78.3            | 307             | 100.6% |
| LLaVA-1.5-7B               |              |                   | Pruning           | budget            | K =               | 128 ; Token     | Reduction       | Rate            | = 77.8%         |                 |                 |        |
| FastV (ECCV'24)            | VP           | 56.1              | 56.4              | 60.2              | 51.3              | 49.6            | 1490            | 59.6            | 50.6            | 61.8            | 285             | 86.4%  |
| SparseVLM (ICML'25)        | PA           | 60.0              | 51.1              | 67.1              | 51.4              | 56.0            | 1696            | 80.5            | 54.9            | 73.8            | 280             | 93.8%  |
| MustDrop (24.11)           | PA VP        | 61.1              | 55.2              | 68.5              | 52.1              | 56.9            | 1745            | 78.7            | 56.3            | 74.6            | 281             | 95.6%  |
| DART (EMNLP'25)            | VP           | 63.2              | 57.5              | 69.1              | 51.7              | 58.7            | 1840            | 80.1            | 56.4            | 75.9            | 296             | 98.0%  |
| MoB (w/o η -prior)         | PA VP        | 63.2              | 57.3              | 69.3              | 52.8              | 60.7            | 1842            | 81.7            | 57.5            | 77.2            | 299             | 99.2%  |
| + η -prior                 | -            | 63.5              | 57.5              | 69.6              | 52.7              | 60.9            | 1845            | 82.1            | 57.8            | 77.5            | 299             | 99.4%  |
| LLaVA-1.5-7B               |              |                   | Pruning           | budget            | K =               | 64 ; Token      | Reduction       | Rate            | = 88.9%         |                 |                 |        |
| FastV (ECCV'24)            | VP           | 48.0              | 52.7              | 51.1              | 50.8              | 46.1            | 1256            | 48.0            | 47.8            | 55.0            | 245             | 77.3%  |
| SparseVLM (ICML'25)        | PA           | 56.2              | 46.1              | 62.2              | 50.1              | 52.7            | 1505            | 75.1            | 51.8            | 68.2            | 180             | 84.6%  |
| MustDrop (24.11)           | PA VP        | 60.0              | 53.1              | 63.4              | 51.2              | 53.1            | 1612            | 68.0            | 54.2            | 69.3            | 267             | 90.1%  |
| DART (EMNLP'25)            | VP           | 60.6              | 53.2              | 69.8              | 51.6              | 55.9            | 1765            | 73.9            | 54.4            | 72.4            | 270             | 93.7%  |
| MoB (w/o η -prior)         | PA VP        | 61.7              | 54.2              | 69.7              | 52.0              | 59.0            | 1806            | 77.2            | 57.0            | 75.5            | 277             | 96.3%  |
| + η -prior                 | -            | 62.1              | 54.5              | 69.8              | 52.1              | 59.0            | 1806            | 77.2            | 57.0            | 75.5            | 277             | 96.4%  |
| LLaVA-Next-7B Vanilla [29] |              |                   | w/o               | Pruning,          | N = 2880          | ; Token         | Reduction       | Rate            | = 0.0%          |                 |                 |        |
|                            | -            | 67.4              | 60.6              | 70.1              | 57.6              | 64.2            | 1851            | 86.5            | 64.9            | 81.8            | 517             | 100%   |
| LLaVA-Next-7B              |              |                   | Pruning           | budget            | K =               | 320 ; Token     |                 | Reduction Rate  | = 88.9%         |                 |                 |        |
| FastV (ECCV'24)            | VP           | 61.6              | 51.9              | 62.8              | 53.1              | 55.9            | 1661            | 71.7            | 55.7            | 71.9            | 374             | 86.4%  |
| SparseVLM (ICML'25)        | PA           | 60.6              | 54.5              | 66.1              | 52.0              | 56.1            | 1533            | 82.4            | 58.4            | 71.5            | 270             | 85.9%  |
| MustDrop (24.11)           | PA VP        | 62.8              | 55.1              | 68.0              | 54.0              | 57.3            | 1641            | 82.1            | 59.9            | 73.7            | 382             | 90.4%  |
| FasterVLM (24.12) [57]     | VP           | 61.6              | 53.5              | 66.5              | 52.6              | 56.9            | 1701            | 83.6            | 56.5            | 74.0            | 401             | 89.8%  |
| DART (EMNLP'25)            | VP           | 65.3              | 58.2              | 68.4              | 56.1              | 61.7            | 1710            | 84.1            | 58.7            | 79.1            | 406             | 93.9%  |
| MoB (with η -prior)        | PA VP        | 65.8              | 58.9              | 68.7              | 57.0              | 62.6            | 1760            | 84.4            | 60.2            | 80.1            | 418             | 95.4%  |

Table 2: Partial comparison of image understanding on the LLaVA-7B series. For MoB, we set K p ∈ { 64 , 48 , 32 } and k ∈ { 4 , 6 , 8 } , corresponding to token-reduction rates of { 88 . 9% , 77 . 8% , 66 . 7% } . For MoB with the η prior, we use K p ∈ { 3 K 8 , K 4 , K 4 } with k = 3 K p 40 for strong-coupling benchmarks and K p ∈ { K 2 , 7 K 16 , 5 K 12 } with k = K p 8 for weak-coupling benchmarks, corresponding to the same token-reduction rates; the pruning layer is fixed at ℓ = 2 . Blue and Orange denote the best and the second. See Appendix C.4 for the detailed setting, and see Appendix D.1 for the full results.

## 4 Experimental Results

Experiment Setting. We perform a comprehensive evaluation of the proposed MoB and several representative methods on two visual tasks: image understanding and visual understanding, together with an efficiency analysis. Our experiments employ four popular MLLMs and include a total of 14 widely recognized benchmarks. For further details regarding the benchmarks, models, baselines, and implement details please refer to Appendix C.

Image Understanding. Table 2 and Table 3 report the evaluation results across a variety of imageunderstanding tasks on LLaVA series and Qwen2VL, respectively. We highlight five key observations: (a) MoB consistently outperforms all base-

Table 3: Comparative experiments on image understanding with Qwen2-VL-7B.

| Method        | GQA                                            | MME                                            | POPE                                           | VQA T                                          | MMB                                            | SQA                                            | Avg.                                           |
|---------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|
| Qwen2-VL-7B   | w/o Pruning ; Token Reduction Rate = 0.0% 82.1 | w/o Pruning ; Token Reduction Rate = 0.0% 82.1 | w/o Pruning ; Token Reduction Rate = 0.0% 82.1 | w/o Pruning ; Token Reduction Rate = 0.0% 82.1 | w/o Pruning ; Token Reduction Rate = 0.0% 82.1 | w/o Pruning ; Token Reduction Rate = 0.0% 82.1 | w/o Pruning ; Token Reduction Rate = 0.0% 82.1 |
| Vanilla [44]  | 62.2                                           | 2317                                           | 86.1                                           |                                                | 80.5                                           | 84.7                                           | 100%                                           |
| Qwen2-VL-7B   | Token Reduction Rate = 66.7%                   | Token Reduction Rate = 66.7%                   | Token Reduction Rate = 66.7%                   | Token Reduction Rate = 66.7%                   | Token Reduction Rate = 66.7%                   | Token Reduction Rate = 66.7%                   | Token Reduction Rate = 66.7%                   |
| FastV         | 58.0                                           | 2130                                           | 82.1                                           | 77.3                                           | 76.1                                           | 80.0                                           | 94.0%                                          |
| DART          | 60.2                                           | 2245                                           | 83.9                                           | 80.5                                           | 78.9                                           | 81.4                                           | 97.0%                                          |
| MoB (with η ) | 61.8                                           | 2268                                           | 84.7                                           | 81.1                                           | 79.5                                           | 82.3                                           | 98.4%                                          |
| Qwen2-VL-7B   | Token Reduction Rate = 77.8%                   | Token Reduction Rate = 77.8%                   | Token Reduction Rate = 77.8%                   | Token Reduction Rate = 77.8%                   | Token Reduction Rate = 77.8%                   | Token Reduction Rate = 77.8%                   | Token Reduction Rate = 77.8%                   |
| FastV         | 56.7                                           | 2031                                           | 79.2                                           | 72.0                                           | 74.1                                           | 78.3                                           | 91.0%                                          |
| DART          | 58.5                                           | 2175                                           | 82.1                                           | 75.3                                           | 77.3                                           | 79.6                                           | 94.3%                                          |
| MoB (with η ) | 59.4                                           | 2203                                           | 82.8                                           | 75.8                                           | 78.1                                           | 80.4                                           | 95.2%                                          |
| Qwen2-VL-7B   | Token Reduction Rate = 88.9%                   | Token Reduction Rate = 88.9%                   | Token Reduction Rate = 88.9%                   | Token Reduction Rate = 88.9%                   | Token Reduction Rate = 88.9%                   | Token Reduction Rate = 88.9%                   | Token Reduction Rate = 88.9%                   |
| FastV         | 51.9                                           | 1962                                           | 76.1                                           | 60.3                                           | 70.1                                           | 75.8                                           | 84.4%                                          |
| DART          | 55.5                                           | 2052                                           | 77.9                                           | 61.8                                           | 72.0                                           | 77.6                                           | 87.4%                                          |
| MoB (with η ) | 56.5                                           | 2094                                           | 78.5                                           | 62.7                                           | 72.8                                           | 78.4                                           | 88.6%                                          |

/0 /1

/0 /2 /3

/4 /3

/5 /i255

/7 /0 /8

/7 /0

Figure 3: Performance-Latency trade-off comparisons across four benchmarks on LLaVA-Next-7B.

<!-- image -->

lines on LLaVA-1.5-7B in most cases. This will be more pronounced when incorporating the η -prior, which highlights the inherent advantage of our approach; (b) single-objective baselines exhibit complementary strengths under different coupling patterns, whereas MoB consistently outperforms all baselines, demonstrating the benefit of balanced objectives; (c) the superiority of MoB becomes even more significant under aggressive token reduction. Specifically, the improvement of MoB over the best baseline in average scores increases from 1.8% at a 66.7% token reduction to 2.7% at an 88.8% reduction on LLaVA-1.5-7B; (d) MoB matches the performance of the vanilla LLaVA-1.5-7B with only 33.3% of visual tokens, which may be attributed to the mitigation of hallucinations caused by redundant tokens; and (e) MoB scales seamlessly to advanced models, preserving 95.2% performance on Qwen2-VL-7B using only 22.2% of visual tokens. These observations demonstrate the superiority of MoB in leveraging limited visual tokens while minimizing performance degradation.

Video Understanding. As presented in Table 4, MoB is general and can be readily extended to more challenging video scenarios without incurring additional cost. Specifically, MoB preserves 97.9% of average performance for Video-LLaVA7B using only 6.6% of the visual tokens, which sets new records in most VideoQA benchmarks, achieving 1.6% and 4.7% improvements over TwigVLM and VisionZip, respectively. These results validate the generalization ability of MoB.

Table 4: Comparative experiments on video understanding with Video-LLaVA-7B.

| Method                 | TGIF                         | MSVD                         | MSRV                         | ActNet                       | Avg.                         |
|------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| Video-LLaVA-7B         | Token Reduction Rate = 0.0%  | Token Reduction Rate = 0.0%  | Token Reduction Rate = 0.0%  | Token Reduction Rate = 0.0%  | Token Reduction Rate = 0.0%  |
| Vanilla [27]           | 47.1                         | 69.8                         | 56.7                         | 43.1                         | 100%                         |
| Video-LLaVA-7B         | Token Reduction Rate = 93.4% | Token Reduction Rate = 93.4% | Token Reduction Rate = 93.4% | Token Reduction Rate = 93.4% | Token Reduction Rate = 93.4% |
| FastV (ECCV'24)        | 23.1                         | 38.0                         | 19.3                         | 30.6                         | 52.1%                        |
| SparseVLM (ICML'25)    | 44.7                         | 68.2                         | 31.0                         | 42.6                         | 86.5%                        |
| VisionZip (24.12) [51] | 42.4                         | 63.5                         | 52.1                         | 43.0                         | 93.2%                        |
| TwigVLM (ICCV'25) [39] | 44.7                         | 68.3                         | 54.6                         | 41.5                         | 96.3%                        |
| MoB (with η -prior)    | 45.3                         | 68.8                         | 55.2                         | 42.8                         | 97.9%                        |

Efficiency Analysis. We present a performance-latency trade-off measured on an NVIDIA A80080GB GPU in Figure 3. The results show that (a) MoB achieves a strong performance-latency trade-off, delivering a 1 . 3 -1 . 5 × speed-up for LLaVA-NEXT-7B with negligible performance loss; (b) due to ignoring the K -η trade-off, the multi-stage method MustDrop is outperformed by singleobjective methods FastV and SparseVLM on MME and POPE, and suffers significant performance drops as token budgets shrink ( i.e. , latency decreases). In contrast, MoB consistently maintains a robust trade-off across all benchmarks, surpassing all the baselines by a clear margin; (c) MoB does not rely on attention scores to identify important tokens, making it compatible with flash attention and more efficient than attention-based methods such as SparseVLM and FastV.

## 5 Ablation and Discussion

Impact of ⟨ K,η,K p , ⟩ . We study the impact of K , η , and K p on pruning performance across four benchmarks: GQA and TextVQA (weak coupling); VizWiz and MMB (strong coupling). As shown in Figure 4, the results can be interpreted by Theorem 1 and Theorem 2(a), respectively.

A. Theorem 1 Perspective: When K is large, e.g. , K = 192 , the trade-off is governed by D 1 K -2 /d eff , hence the trade-off intensity remains nearly identical across benchmarks. Conversely, When K is small, especially K = 64 , in weak-coupling benchmarks, the trade-off turns to be governed by D 2 η 2 ; thus, the trade-off intensity is obviously more pronounced in GQA and TextVQA than that in VizWiz and MMB. These observations exactly confirm the validity of Theorem 1.

B. Theorem 2(a) Perspective. (a) Under weak coupling, the alignment term α ( η, k, L )( K p ) -1 /d eff is amplified, which requires a larger K p to suppress the overall error. However, across benchmarks sharing the same coupling pattern, the optimal K p values exhibit only minor variation. (b) Increasing the total budget K pushes the optimal K p upward to rebalance the two bound terms. Since the prompt length L is fixed, adding more tokens yields diminishing returns for prompt alignment, which is reflected in the declining ratio K p /K . These validate the performance bound in Theorem 2(a).

/9

/10

/11

/12

/13

/14

/15

/11

/47

/35

/20

/i255

/37

/33

/38

/34

/39

/40

/33

/i255

/41

/38

/42

/43

/33

/i255

/44

/45 /46

/32 /33

/34

/35

/i255

/37

/33

/38

/34

/39

/40

/33

/i255

/41

/38

/42

/43

/33

/i255

/44

/45 /46

/32 /33

/34

/35

/i255

/37

/33

/38

/34

/39

/40

/33

/i255

/41

/38

/42

/43

/33

/i255

/44

/45 /46

/0 /1 /2 /3 /4 /5 /6 /2 /3 /7 /5 /6 /2 /8 /9 /10 /11 /12 /13 /14 /13 /15 /16 /0 /1 /2 /4 /6 /5 /0 /2 /26 /27 /5 /26 /2 /4 /7 /5 /4 /2 /4 /6 /5 /0 /2 /26 /0 /2 /26 /0 /2 /7 /0 /1 /5 /27 /7 /0 /1 /5 /29 /7 /3 /27 /5 /6 /30 /7 /0 /7 /5 /1 /7 /0 /1 /5 /31 /7 /0 /1 /5 /4 /30 /7 /3 /27 /5 /29 /7 /3 /27 /5 /1 /7 /3 /31 /5 /29 /30 /7 /3 /27 /5 /4 /7 /3 /31 /5 /31 /7 /3 /31 /5 /4 /30 Figure 4: Comprehensive ablation on the budget configuration ⟨ K p , K ⟩ across four benchmarks with distinct prompt-visual coupling η on LLaVA-1.5-7B, where K = { 64 , 128 , 192 } ; the mean relative slope (%) is given by 100 x n -x 1 ∑ n -1 i =1 y i +1 -y i y i , quantifying the trade-off intensity; the ratio K p K reflects the cost-effectiveness of prompt alignment, and the box plot presents the distribution of η . D 58 57 62 61 60 59 56

<!-- image -->

/48

/47

/49

/50

/49

/50

/51

/52

/17

/32 /33

/18

/17

/34

/19

/35

/20

/i255

/37

/33

/38

/34

/39

/40

/33

/i255

/41

/38

/42

/43

/33

/i255

/44

/45 /46

/17

/32 /33

/18

/17

/34

/23

/i255

/51

/54

/55

/52

/56

<!-- image -->

Dataset

Figure 5: Ablation on the ratio of k/K p .

Figure 6: Ablation on the pruning layer.

Remarkably, the experimental results suggest that simply determining the optimal K p for each of the two coupling patterns suffices to guarantee effective generalization across all scenarios.

Impact of Covering Fold k . We chose the covering fold k by examining the normalized ratio k/K p across eight benchmarks and nine budget configurations. As shown in Figure 5, (a) weak-coupling benchmarks generally require a larger k to ensure critical region coverage, whereas strong-coupling settings suffice with a smaller k ; (b) benchmarks with longer prompts impose a lower cap on k to preserve sampling diversity and avoid redundant selection of salient tokens. Notably, weak-coupling benchmarks with long prompts ( e.g. , GQA, TextVQA) exhibit a narrowly clustered optimal k/K p range, reflecting their strict requirement to cover key tokens without excessive redundancy.

Impact of Pruning Layer. As shown in Figure 6, (a) models with visual token pruning consistently achieve a more favorable performance-efficiency trade-off than the vanilla model on both benchmarks. (b) Pruning in deeper layers provides more significant benefits for the weak-coupling TextVQA than strong-coupling MME. We attribute this to stronger cross-modal interactions in deeper MLLM layers, which facilitate identification of answer-relevant tokens under weak coupling, whereas pruning in shallow layers disrupts these interactions and incurs greater performance degradation.

## 6 Conclusion

In this paper, we present a comprehensive analysis of visual token pruning, deriving the first closedform error bound with a practical relaxation. Leveraging ϵ -covering theory, we quantify the intrinsic trade-off between the fundamental pruning objectives, i.e. , visual preservation and prompt alignment, and identify their optimal attainment levels under a fixed pruning budget. Building on these insights, we introduce MoB, a training-free algorithm for visual token pruning. Based on greedy radius trading, MoB ensures the near-optimal attainment per objective via budget allocation, offering a provable performance bound and multilinear scalability. Experimental results indicate that MoB matches the performance (100.6%) of LLaVA-1.5-7B with only 33.3% of visual tokens and can be seamlessly integrated into advanced MLLMs, such as LLaVA-Next-7B and Qwen2-VL-7B. Our work advances the understanding of visual token pruning and offers valuable insights for future MLLM compression.

Limitations. Our theoretical guarantees rely on assumption 1, which is generally satisfied in practice but may not hold for all MLLMs. Besides, MoB applies a preliminary search to select the proper K p , which potentially introduces extra tuning overhead in practical applications. Future work will focus on developing an adaptive K p selection mechanism driven by online estimation of the coupling η .

/57

/58

/59

/54

/60

/61

/62

/59

## Acknowledgments

The work was performed at the Shanghai Key Laboratory of Multidimensional Information Processing, East China Normal University; the Institute of Natural Sciences and School of Mathematical Sciences, Shanghai Jiao Tong University; and the Chongqing Key Laboratory of Precision Optics, Chongqing Institute of East China Normal University, with joint support from the National Natural Science Foundation of China (62176091), the Natural Science Foundation of Chongqing (CSTB2024NSCQMSX0877), the Science and Technology Commission of Shanghai Municipality (21DZ2203100) and the Fundamental Research Funds for the Central Universities.

## References

- [1] Yash Akhauri, Ahmed F AbouElhamayed, Yifei Gao, Chi-Chih Chang, Nilesh Jain, and Mohamed S Abdelfattah. Tokenbutler: Token importance is predictable. arXiv preprint arXiv:2503.07518 , 2025.
- [2] Kazi Hasan Ibn Arif, JinYi Yoon, Dimitrios S Nikolopoulos, Hans Vandierendonck, Deepu John, and Bo Ji. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models in resource-constrained environments. arXiv preprint arXiv:2408.10945 , 2024.
- [3] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450 , 2016.
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 , 2025.
- [5] Kaichen Zhang* Fanyi Pu* Xinrun Du Yuhao Dong Haotian Liu Yuanhan Zhang Ge Zhang Chunyuan Li Bo Li*, Peiyuan Zhang* and Ziwei Liu. Lmms-eval: Accelerating the development of large multimoal models, March 2024.
- [6] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In The Eleventh International Conference on Learning Representations , 2022.
- [7] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies , pages 190-200, 2011.
- [8] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision , pages 19-35. Springer, 2024.
- [9] Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, and Ziwei Liu. Large language models are visual reasoning coordinators. Advances in Neural Information Processing Systems , 36:70115-70140, 2023.
- [10] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 , 2024.
- [11] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 24185-24198, 2024.
- [12] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 6904-6913, 2017.

- [13] Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. Attention score is not all you need for token importance indicator in kv cache reduction: Value also matters. arXiv preprint arXiv:2406.12335 , 2024.
- [14] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 3608-3617, 2018.
- [15] Dorit S Hochbaum and David B Shmoys. A best possible heuristic for the k-center problem. Mathematics of operations research , 10(2):180-184, 1985.
- [16] Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, and Zhuowen Tu. Bliva: A simple multimodal llm for better handling of text-rich visual questions. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 2256-2264, 2024.
- [17] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 6700-6709, 2019.
- [18] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2758-2766, 2017.
- [19] Kasper Green Larsen and Jelani Nelson. The johnson-lindenstrauss lemma is optimal for linear dimensionality reduction. arXiv preprint arXiv:1411.2404 , 2014.
- [20] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 , 2024.
- [21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning , pages 19730-19742. PMLR, 2023.
- [22] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814 , 2024.
- [23] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355 , 2023.
- [24] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 26763-26773, 2024.
- [25] Youwei Liang, GE Chongjian, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Evit: Expediting vision transformers via token reorganizations. In International Conference on Learning Representations , 2022.
- [26] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. A survey of multimodel large language models. In Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering , pages 405-409, 2024.
- [27] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection, 2024.
- [28] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 26296-26306, 2024.
- [29] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.

- [30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023.
- [31] Ting Liu, Liangtao Shi, Richang Hong, Yue Hu, Quanjun Yin, and Linfeng Zhang. Multi-stage vision token dropping: Towards efficient multimodal large language model. arXiv preprint arXiv:2411.10803 , 2024.
- [32] Xuyang Liu, Ziming Wang, Yuhang Han, Yingyao Wang, Jiale Yuan, Jun Song, Bo Zheng, Linfeng Zhang, Siteng Huang, and Honggang Chen. Compression with global guidance: Towards training-free high-resolution mllms acceleration. arXiv preprint arXiv:2501.05179 , 2025.
- [33] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences , 67(12):220102, 2024.
- [34] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems , 32, 2019.
- [35] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS) , 2022.
- [36] Carsten Moenning and Neil A Dodgson. Fast marching farthest point sampling. Technical report, University of Cambridge, Computer Laboratory, 2003.
- [37] Yingzhe Peng, Xinting Hu, Jiawei Peng, Xin Geng, Xu Yang, et al. Live: Learnable in-context vector for visual question answering. Advances in Neural Information Processing Systems , 37:9773-9800, 2024.
- [38] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388 , 2024.
- [39] Zhenwei Shao, Mingyang Wang, Zhou Yu, Wenwen Pan, Yan Yang, Tao Wei, Hongyuan Zhang, Ning Mao, Wei Chen, and Jun Yu. Growing a twig to accelerate large vision-language models. arXiv preprint arXiv:2503.14075 , 2025.
- [40] Zhenwei Shao, Zhou Yu, Meng Wang, and Jun Yu. Prompting large language models with answer heuristics for knowledge-based visual question answering. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition , pages 14974-14983, 2023.
- [41] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 8317-8326, 2019.
- [42] Xudong Tan, Peng Ye, Chongjun Tu, Jianjian Cao, Yaoxin Yang, Lin Zhang, Dongzhan Zhou, and Tao Chen. Tokencarve: Information-preserving visual token compression in multimodal large language models. arXiv preprint arXiv:2503.10501 , 2025.
- [43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [44] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191 , 2024.
- [45] Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin, Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan, Quanzeng You, and Hongxia Yang. Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning. arXiv preprint arXiv:2401.06805 , 2024.

- [46] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for important tokens in multimodal language models: Duplication matters more. arXiv preprint arXiv:2502.11494 , 2025.
- [47] Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Zhe Chen, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, et al. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. Advances in Neural Information Processing Systems , 37:69925-69975, 2024.
- [48] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247 , 2024.
- [49] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the 25th ACM international conference on Multimedia , pages 1645-1653, 2017.
- [50] Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and Junyang Lin. Understanding and improving layer normalization. Advances in neural information processing systems , 32, 2019.
- [51] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. arXiv preprint arXiv:2412.04467 , 2024.
- [52] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. arXiv preprint arXiv:2409.10197 , 2024.
- [53] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 9127-9134, 2019.
- [54] Yuanhan Zhang Bo Li Songyang Zhang Wangbo Zhao Yike Yuan Jiaqi Wang Conghui He Ziwei Liu Kai Chen Dahua Lin Yuan Liu, Haodong Duan. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281 , 2023.
- [55] Zheng Zhan, Yushu Wu, Zhenglun Kong, Changdi Yang, Yifan Gong, Xuan Shen, Xue Lin, Pu Zhao, and Yanzhi Wang. Rethinking token reduction for state space models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing , pages 16861697, 2024.
- [56] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024.
- [57] Qizhe Zhang, Aosong Cheng, Ming Lu, Zhiyong Zhuo, Minqi Wang, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster. arXiv preprint arXiv:2412.01818 , 2024.
- [58] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417 , 2024.
- [59] Yiwu Zhong, Zhuoming Liu, Yin Li, and Liwei Wang. Aim: Adaptive inference of multi-modal llms via token merging and pruning. arXiv preprint arXiv:2412.03248 , 2024.
- [60] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592 , 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: See Section 1 for the main claims; see Sections 3 to 5 and Appendices D and E for the detailed contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Section 6 for the discussion on the limitations.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The assumptions are clearly stated in the statement of the theorems: any partial composition F of the language model satisfies a Lipschitz Continuity w.r.t. the Hausdorff Distance with constant C ℓ ≥ 1 , the δ -dilations V δ := ⋃ v ∈V B ( v, δ ) , P δ := ⋃ p ∈P B ( p, δ ) ( δ ≪ η ) satisfy d eff -dimensional covering regularity with constants a, a ′ , b, b ′ . See Theorems 1 and 2. The proof of Lemma 1 can be found in Appendix E.1, the proof of Lemma 2 can be found in Appendix E.2, the proof of Lemma 3 can be found in Appendix E.3, the proof of Theorems 1 and 2 are provided in Appendices E.4 and E.5, respectively.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: See Appendices B and C for the information needed to reproduce the main experimental results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code is provided in the supplemental material.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The experiment does not involve any training process. See Section 5 and appendix C for all the test details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experimental results are accompanied by significance tests, and crossvalidation conducted using a publicly available third-party framework.

Guidelines:

- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: See Appendix C for the computer resources needed to reproduce the experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: See Section 6 and appendix A for the discussion on the broader impacts.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The assets used in this paper are properly credited, and the license and terms are respected.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: MoB is an inference-time algorithm that prunes visual tokens in pre-trained multimodal LLMs-LLaVA-1.5-7B, LLaVA-Next-7B, Qwen2-VL-7B and Video-LLaVA7B. At Transformer layer ℓ = 2 , it removes redundant vision tokens using a bi-objective covering rule (see Section 3 and appendices B and C). The LLM weights remain frozen ; no additional data, gradient updates, or prompt engineering are used. Thus the LLMs serve as essential yet unmodified back-bones whose intermediate embeddings are the input to MoB.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

In the appendix, we provide additional information as listed below:

- §A provides the broader impacts of MoB
- §B provides the algorithm details and pseudocode of MoB
- §C provides the overview of the data, models, baselines and implementation details.
- §D provides the additional experimental results.
- §E provides the omitted technical details.

## A Broader Impacts and Limitations

Theory Impacts. Beyond the visual setting, MoB's theoretical lens-balancing Visual Preservation (retaining sufficient context) and Prompt Alignment (isolating 'golden evidence')-naturally transfers to language domain. It makes the key challenging (context vs. evidence) in long-context LLM explicit and offers actionable guidance for token-level compression and scheduling under fixed context budgets. In practice, this perspective informs RAG (calibrating recall vs. precision) and summarization/LLM memory (trading coherence vs. conciseness).

Application Impacts. The proposed MoB yields substantial acceleration of MLLMs with negligible performance loss, thereby enabling high-resolution vision-language models to operate on resource-constrained platforms such as edge devices and mobile systems while supporting low-latency applications-including assistive technologies for the visually impaired, autonomous navigation, and AR/VR. Besides, MoB potentially benefits other redundancy-heavy domains ( e.g. , point clouds and multi-sensor fusion), guiding efficient token-level compression beyond vision.

Theory Limitations. The theoretical analysis (lemma 1 and theorem 1) and the performance guarantees (theorem 2) rely on assumption 1 (Lipschitz Continuity) and lemma 3 (Covering Regularity). In embedding spaces that violate metric properties or exhibit highly irregular token distributions, these conditions may fail to hold, and the provable performance bounds may no longer apply.

Application Limitations. Our deployment presently requires an a priori estimate of η to set the pruning hyperparameters K p and k . When η is misestimated for a new model, domain, or input distribution, the selected K p and k can deviate from their optimum, leading to suboptimal speed-accuracy trade-offs.

## B Algorithm

## Algorithm 1 Multi-Objective Balanced Covering (MoB)

Require: Visual token V ∈ R N × d , Prompt token P ∈ R L × d , Budget K p , K v , Covering fold k Ensure: Index list for select tokens S ∈ N Kp + Kv

- 1: Normalize all token embeddings to unit ℓ 2 norm: V ← V / ∥ V ∥ 2 , row , P ← P / ∥ P ∥ 2 , row

## Step 1. Select Prompt Centers via Nearest-Neighbor Covering

- 2: Compute cosine-similarity matrix via PV ⊤ : M ← PV ⊤ ▷ M ∈ R L × N .
- 3: Retrieve k nearest token indices per prompt:

<!-- formula-not-decoded -->

▷ C idx , C sim ∈ R L × k collects index and similarity of k closest centers per prompt token. # Deduplicate candidate indices

- 4: Flatten index and similarity arrays: C idx ← Flatten ( C idx ) , C sim ← Flatten ( C sim ) ▷ C idx ∈ N Lk , C sim ∈ R Lk
- 5: Remove duplicate indices, preserving associated similarities:

<!-- formula-not-decoded -->

- 6: Identify topK p prompt centers by similarity: i p ← ArgTopK ( C ∗ sim , K p )
- 7: Form the prompt-center index list: S p ← C [ i p

<!-- formula-not-decoded -->

- ∗ idx ] ▷ S p ∈ N K p

## Step 2. Select Visual Centers via Farthest-Point Sampling

- 8: Initialize selected centers: S ← S
- p # Initialize token-to-prompt minimum distances
- 9: Compute pairwise minimum distances between all tokens and selected prompt centers:

<!-- formula-not-decoded -->

▷ Selected centers have zero distance in d ∈ R N .

## # Farthest-Point Sampling

- 10: for t = 1 to K v do
- 11: Select the token farthest from current centers: i ∗ ← ArgMax ( d ) , S ← Concat ( S , i ∗ ) ▷ Selected tokens are excluded (distance = 0) from further sampling.
- 12: Compute cosine distances to the newly selected token: d ∆ ← 1 N -VV [ i ∗ ] ⊤
- 13: Update each token's minimum distance: d ← ElementwiseMin ( d , d ∆ ) ▷ Distance of newly selected token i ∗ set to zero in d .
- 14: end for
- 15: return S

## Algorithm 2 Compute Prompt-Visual Coupling

Require: Visual embeddings V ∈ R nv × d , Prompt embeddings P ∈ R np × d Ensure: Hausdorff distance h ( V , P )

## Step 1. Compute Pairwise Euclidean Distances

- 1: Compute distance matrix via cdist : D ← cdist( V , P , p = 2

## Step 2. Directed Hausdorff Distances

- 2: Visual-to-prompt directed distance:

<!-- formula-not-decoded -->

- 3: Prompt-to-visual directed distance:

<!-- formula-not-decoded -->

## Step 3. Final Hausdorff Distance

- 4: return max ( h v → p , h p → v )
- ) ▷ D ∈ R nv × np

## C Experiment Details

## C.1 Benchmarks

Our experiments evaluate the vision-language reasoning abilities of multimodal large language models using a comprehensive suite of widely recognized benchmarks. For image understanding tasks, we assess performance on ten public benchmarks: GQA, MMBench (MMB) and MMBench-CN (MMB CN ), MME, POPE, VizWiz, ScienceQA (SQA), VQA V2 , TextVQA (VQA T ), and OCRBench (OCR). For video understanding tasks, we conduct experiments on four popular benchmarks: TGIFQA (TGIF), MSVD-QA (MSVD), MSRVTT-QA (MSRV), and ActivityNet-QA (ActNet). The following section provides a concise overview of these benchmarks:

GQA [17] leverages scene graphs, questions, and images to evaluate visual scene understanding and reasoning. By incorporating detailed spatial relationships and object-level attributes, it poses significant challenges for models to perform accurate visual reasoning in complex environments.

MMBench [54] introduces a hierarchical evaluation framework where model capabilities are dissected into three levels. Level-1 focuses on basic perception and reasoning; Level-2 subdivides these abilities into six distinct sub-skills; and Level-3 further refines the evaluation into 20 specific dimensions. Its Chinese counterpart, MMBench-CN , adopts a similar structure.

MME [26] rigorously tests perceptual and cognitive abilities across 14 sub-tasks. By employing carefully crafted instruction-answer pairs and succinct instructions, MME minimizes data leakage and provides a robust, fair assessment of a model's multifaceted performance.

POPE [23] targets the evaluation of object hallucination by posing binary questions about object presence in images. It quantifies hallucination levels using metrics, e.g. , accuracy, recall, precision, and F1 score, offering a precise and focused measure of model reliability.

VizWiz [14] is a visual question answering benchmark derived from interactions with blind users. Comprising over 31 , 000 image-question pairs with 10 human-annotated answers per query, it encapsulates the challenges of low-quality image capture and conversational spoken queries, thereby emphasizing real-world visual understanding.

ScienceQA [35] spans multiple scientific domains by organizing questions into 26 topics, 127 categories, and 379 skills. This hierarchical categorization provides a diverse and rigorous testbed for evaluating multimodal understanding, multi-step reasoning, and interpretability across natural, language, and social sciences.

VQA V2 [12] challenges models with open-ended questions based on 265 , 016 images that depict a variety of real-world scenes. Each question is paired with 10 human-annotated answers, facilitating a thorough evaluation of a model's capacity to interpret and respond to diverse visual queries.

TextVQA [41] focuses on the integration of text within visual content. It evaluates a model's proficiency in reading and reasoning about textual information embedded in images, thereby requiring a balanced understanding of both visual and linguistic cues.

OCRBench [33] is a comprehensive benchmark for evaluating the OCR capabilities of multi-modal language models across five key tasks: text recognition, scene text-centric and document-oriented VQA, key information extraction, and handwritten mathematical expression recognition.

TGIF-QA [18] adapts the visual question answering task to the video domain by focusing on GIFs. With 165 K question-answer pairs, it incorporates tasks, e.g. , counting repetitions, identifying repeating actions, detecting state transitions, and frame-specific question answering, thereby demanding detailed spatio-temporal analysis.

MSVD-QA [49] builds upon the MSVD dataset by pairing 1 , 970 video clips with approximately 50 . 5 K QA pairs. Questions are categorized into five distinct types, e.g. , what, who, how, when, and where, making it a versatile tool for evaluating video understanding.

MSRVTT-QA [7] features 10 K video clips and 243 K QA pairs designed to test the integration of visual and temporal information. Its structure, which parallels that of MSVD-QA through the inclusion of five question types, further enriches the evaluation landscape for video-based tasks.

ActivityNet-QA [53] provides 58 K human-annotated question-answer pairs drawn from 5 . 8 K videos. Its focus on questions related to motion, spatial relationships, and temporal dynamics necessitates long-term spatio-temporal reasoning, thus serving as a benchmark for advanced video understanding.

## C.2 Multi-modal Large Language Models

We evaluate MoB using various open-source multimodal large language models (MLLMs). For image understanding tasks, experiments are conducted on the LLaVA series, including LLaVA-1.5-7B and LLaVA-Next-7B, as well as the Qwen-VL series, such as Qwen2-VL-7B. Specifically, LLaVA-Next and Qwen2-VL are utilized to validate performance on high-resolution images, i.e. , those with a larger number of visual tokens. For video understanding tasks, we employ Video-LLaVA-7B as the baseline model, following the settings reported in its original paper to ensure a fair comparison.

LLaVA-1.5-7B [28] is a robust vision-language model built on the LLaV A framework. It processes images resized to 224 × 224 and tokenizes them into roughly 572 visual tokens using a patch-based vision encoder. This design balances fine-grained visual representation with computational efficiency, making it effective for diverse multimodal tasks.

LLaVA-Next-7B [29] extends the LLaVA-1.5 by incorporating refined training strategies and data curation. It supports higher-resolution inputs (up to 448 × 448 ), yielding up to 2880 visual tokens. These enhancements improve its visual reasoning capabilities and enable more precise alignment between visual content and language but also incur significantly increased computational cost.

Qwen2-VL-7B [44] augments the Qwen2 language model with visual input capabilities. This model leverages cross-modal pretraining to seamlessly merge vision and language, demonstrating strong performance in complex visual question answering and comprehensive scene understanding.

Video-LLaVA-7B [27] extends the LLaVA framework into the temporal domain by processing video inputs. It is designed to capture both spatial and temporal dynamics, enabling effective video comprehension and video-based question answering with coherent and context-aware responses.

## C.3 Baselines

To validate the superiority of the proposed MoB, we construct a robust baseline that integrates a comprehensive set of representative existing methods, which encompass single-stage methods with both two distinct objectives and several multi-stage methods.

ToMe [6] employs a lightweight token-matching scheme to merge visually similar tokens across transformer layers, thereby reducing computation without additional training. Its simple yet effective design makes it well suited for real-time applications.

FastV [8] leverages attention maps in the early layers to identify and prune non-critical tokens, significantly reducing initial computational overhead. This focus on early-stage reduction allows the model to operate more efficiently while maintaining performance.

SparseVLM [58] ranks tokens based on cross-modal attention to assess image-prompt relevance and adopts adaptive sparsity ratios to retain key information. It further incorporates a token recycling mechanism to balance the trade-off between efficiency and accuracy.

HiRED [2] allocates token budgets across image partitions by using CLS token attention and then selects the most informative tokens within each partition. This spatially aware approach ensures balanced reduction while preserving contextual details.

LLaVA-PruMerge [38] combines pruning and merging strategies by dynamically removing less important tokens using sparse CLS-visual attention. It then clusters the retained tokens based on key similarity, ensuring that crucial visual features remain intact.

PyramidDrop [48] adopts a progressive token-dropping strategy across different model stages, resulting in a pyramid-like token structure. This method carefully balances the reduction of tokens with the preservation of performance as the processing advances.

MustDrop [31] integrates several token-reduction strategies including spatial merging, text-guided pruning, and output-aware cache policies. Its multi-faceted approach efficiently reduces token counts across various stages of the model.

VisionZip [51] first selects dominant tokens that capture the majority of an image's information and then merges the remaining tokens based on semantic similarity. This approach dramatically reduce token redundancy while accelerating inference and maintaining robust performance.

FasterVLM [57] evaluates token importance using CLS attention in the encoder and prunes tokens before they interact with the language model. This preemptive reduction streamlines the overall process and enhances model efficiency.

GlobalCom 2 [32] employs a hierarchical strategy by coordinating thumbnail tokens to allocate adaptive retention ratios for high-resolution crops. This approach successfully preserves local details while providing effective global context reduction.

DART [46] leverages token duplication to guide its pruning process instead of relying solely on attention scores. By selecting a small set of pivot tokens and retaining only those with minimal redundancy, DART achieves significant acceleration in a training-free manner.

TokenCarve [42] implements a two-stage, training-free compression framework that preserves critical visual information during aggressive token reduction. It first prunes low-information tokens using an information-preservation guided selection and then merges the remaining tokens based on similarity to minimize accuracy loss.

TwigVLM [39] accelerates large vision-language models by appending a lightweight twig block to an early layer of a frozen base VLM. It utilizes twig-guided token pruning coupled with selfspeculative decoding to boost generation speed while retaining high accuracy even under aggressive token reduction.

## C.4 Implement Details

From Theorems 1 and 2, the balance between the visual preservation and prompt alignment, i.e. , the optimal budget K p applied for covering prompt tokens P , is jointly determined by the total budget K and the visual-prompt coupling η . To ensure fair comparison, we evaluate two settings.

(i) Without η prior. This setting deliberately avoids any benchmark-specific prior (w/o η prior). MoB adjusts K p solely as a function of K to balance the two objectives. Based on an ablation over ⟨ K,K p ⟩ , we set K p ∈{ 64 , 48 , 32 } and k ∈{ 4 , 6 , 8 } , corresponding to token-reduction rates of { 88 . 9% , 77 . 8% , 66 . 7% } .

(ii) With η prior. To verify the K -η -K p relationship formulated in Theorems 1 and 2, we introduce a coarse benchmark prior on η . Specifically, we do not meticulously search the optimal hyperparameters for MoB, i.e. , K p and the covering fold k , per benchmark. Instead, we partition benchmarks by their empirical η distribution into two groups (strong v.s. weak coupling) and employ the same configuration per group . From a joint ablation over ⟨ K,η,K p ⟩ , for image understanding we set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which again yield token-reduction rates of { 88 . 9% , 77 . 8% , 66 . 7% } .

As for video understanding, we set K p = 3 K 8 , k = 3 K p 40 for MSVD, MSRV, and ActNet, and K p = K 2 , k = K p 8 for TGIF. Unless otherwise stated, the pruning layer index is fixed to ℓ = 2 for both image and video tasks. The same configurations are applied across all MLLMs, and all baselines are run with their default settings.

To ensure reproducibility, we cross-validated our experimental results using the publicly available MLLMsevaluation tool lmms-eval (v0.3.0) [56, 5], with the random seed set to 1234 . All experiments were conducted on 4 × Nvidia A800-80GB GPUs paired with 2 × Intel Xeon ® Gold 6348 CPUs. The implementation was carried out in Python 3.10 using PyTorch 2.1.2 and CUDA 11.8.

## D Additional Experimental Results

## D.1 Quantitative Comparison

Table 5: Full results on image understanding with the LLaVA-7B Series. For MoB, we set K p ∈ { 64 , 48 , 32 } and k ∈ { 4 , 6 , 8 } , corresponding to token-reduction rates of { 88 . 9% , 77 . 8% , 66 . 7% } . For MoB with the η prior, we use K p ∈ { 3 K 8 , K 4 , K 4 } with k = 3 K p 40 for strong-coupling benchmarks and K p ∈ { K 2 , 7 K 16 , 5 K 12 } with k = K p 8 for weak-coupling benchmarks, corresponding to the same token-reduction rates; the pruning layer is fixed at ℓ = 2 . B and O denote the best and the second.

| Method                                               | Objectives   | Strong Coupling   | Strong Coupling   | Strong Coupling   | Strong Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Weak Coupling   | Avg.        |
|------------------------------------------------------|--------------|-------------------|-------------------|-------------------|-------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------|
| Method                                               | Objectives   | MMB               | MMB CN            | SQA               | VizWiz            | GQA             | MME             | POPE            | VQA T           | VQA V2          | OCR             | Avg.        |
| LLaVA-1.5-7B                                         |              |                   | w/o               | Pruning,          | N = 576           | ; Token         | Reduction       | Rate            | = 0.0%          |                 |                 |             |
| Vanilla [28]                                         | -            | 64.7              | 58.1              | 69.5              | 50.0              | 61.9            | 1862            | 85.9            | 58.2            | 78.5            | 297             | 100%        |
| LLaVA-1.5-7B                                         |              |                   | Pruning           | budget            | K = 192           | ; Token         | Reduction       | Rate            | = 66.7%         |                 |                 |             |
| ToMe (ICLR'23)                                       | VP           | 60.5              | -                 | 65.2              | -                 | 54.3            | 1563            | 72.4            | 52.1            | 68.0            | -               | 88.5%       |
| [6] FastV (ECCV'24) [8]                              | VP           | 61.2              | 57.0              | 67.3              | 50.8              | 52.7            | 1612            | 64.8            | 52.5            | 67.1            | 291             | 91.2%       |
|                                                      |              |                   |                   | 68.4              | 50.1              | 58.7            |                 |                 | 47.4            | 74.9            | 190             |             |
| HiRED (AAAI'25) [2]                                  | VP           | 62.8              | 54.7 52.9         | 67.9              | 50.1              | 54.3            | 1737            | 82.8 71.3       | 54.3            | 70.6            | 253             | 91.5% 90.8% |
| LLaVA-PruMerge (24.05) [38] SparseVLM (ICML'25) [58] | VP           | 59.6 62.5         | 53.7              | 69.1              | 50.5              | 57.6            | 1632 1721       | 83.6            | 56.1            | 75.6            | 292             | 96.3%       |
|                                                      | PA           | 63.3              |                   |                   |                   |                 |                 | 82.3            | 56.1            | 75.1            | 290             |             |
| PyramidDrop (CVPR'25) [48] [55]                      | PA           |                   | 56.8              | 68.8              | 51.1              | 57.1            | 1797            | 82.5            | 55.7            | 74.4            |                 | 96.7%       |
| FiCoCo-V (EMNLP'24) MustDrop (24.11) [31]            | VP           | 62.3              | 55.3              | 67.8              | 51.0              | 58.5 58.2       | 1732            | 82.6            | 56.5            |                 | -               | 96.1% 97.2% |
| VisionZip (24.12) [51]                               | PA VP        | 62.3              | 55.8              | 69.2              | 51.4              | 59.3            | 1787 1783       | 85.3            | 57.3            | 76.0 76.8       | 289 -           | 97.7%       |
|                                                      | VP           | 63.0              | -                 | 68.9              | -                 |                 |                 |                 |                 |                 |                 |             |
| DART (EMNLP'25) [46]                                 | VP           | 63.6              | 57.0              | 69.8              | 51.2              | 60.0            | 1856            | 82.8            | 57.4            | 76.7            | 296             | 98.8%       |
| TokenCarve (25.03) [42]                              | PA VP        | 63.0              | -                 | 69.1              | 50.9              | -               | 1830            | 84.9            | 58.4            | 78.0            | -               | 99.3%       |
| TwigVLM (ICCV'25) [39]                               | PA           | 64.0              | -                 | 68.8              | -                 | 61.2            | 1848            | 87.2            | 58.0            | 78.1            | -               | 99.5%       |
| MoB (w/o η -prior)                                   | PA VP        | 63.8              | 57.5              | 70.0              | 52.4              | 61.2            | 1858            | 84.5            | 58.2            | 77.9            | 304             | 100.2%      |
| + η -prior                                           | -            | 64.1              | 57.8              | 70.1              | 52.5              | 61.4            | 1860            | 84.8            | 58.5            | 78.3            | 307             | 100.6%      |
| LLaVA-1.5-7B                                         |              |                   | Pruning           | budget            | K = 128           | ; Token         | Reduction       | Rate            | = 77.8%         |                 |                 |             |
| ToMe (ICLR'23)                                       | VP           | 53.3              | -                 | 59.6              | -                 | 52.4            | 1343            | 62.8            | 49.1            | 63.0            | -               | 80.4%       |
| FastV (ECCV'24)                                      | VP           |                   |                   | 60.2              | 51.3              | 49.6            | 1490            | 59.6            | 50.6            | 61.8            | 285             | 86.4%       |
| HiRED (AAAI'25)                                      | VP           | 56.1 61.5         | 56.4 53.6         | 68.1              | 51.3              | 57.2            | 1710            | 79.8            | 46.1            | 73.4            | 191             | 90.2%       |
| LLaVA-PruMerge (24.05)                               | VP           | 58.1              | 51.7              | 67.1              | 50.3              | 53.3            | 1554            | 67.2            | 54.3            | 68.8            | 248             | 88.8%       |
| SparseVLM (ICML'25)                                  |              | 60.0              | 51.1              |                   |                   |                 |                 | 80.5            | 54.9            | 73.8            |                 | 93.8%       |
|                                                      | PA           |                   |                   | 67.1              | 51.4              | 56.0            | 1696            |                 |                 |                 | 280             | 95.1%       |
| PyramidDrop (CVPR'25) FiCoCo-V (EMNLP'24)            | PA VP        | 61.6 61.1         | 56.6 54.3         | 68.3 68.3         | 51.0 49.4         | 56.0 57.6       | 1761 1711       | 82.3 82.2       | 55.1 55.6       | 72.9 73.1       | 287 -           | 94.9%       |
| MustDrop (24.11)                                     | PA VP        | 61.1              | 55.2              | 68.5              | 52.1              | 56.9            | 1745            | 78.7            | 56.3            | 74.6            | 281             | 95.6%       |
| VisionZip (24.12)                                    |              |                   | -                 | 68.9              | -                 |                 |                 |                 |                 |                 |                 | 96.2%       |
| DART (EMNLP'25)                                      | VP           | 62.0              | 57.5              | 69.1              | 51.7              | 57.6 58.7       | 1762 1840       | 83.2 80.1       | 56.8            | 75.6 75.9       | - 296           | 98.0%       |
| TokenCarve (25.03)                                   | VP PA VP     | 63.2 62.7         | -                 | 68.9              | 51.0              | -               | 1829            | 84.5            | 56.4 58.1       | 77.3            | -               | 99.0%       |
| TwigVLM (ICCV'25)                                    | PA           | 63.5              | -                 | 69.5              | -                 | 60.6            | 1818            | 86.6            | 57.8            | 77.9            | -               | 99.0%       |
| -prior)                                              |              |                   |                   |                   |                   | 60.7            | 1842            |                 | 57.5            | 77.2            | 299             |             |
| MoB (w/o η + η -prior                                | PA VP -      | 63.2 63.5         | 57.3 57.5         | 69.3 69.6         | 52.8 52.7         | 60.9            | 1845            | 81.7 82.1       | 57.8            | 77.5            | 299             | 99.2% 99.4% |
| LLaVA-1.5-7B                                         |              |                   | Pruning           | budget            | K = 64            | ; Token         | Reduction       | Rate            | = 88.9%         |                 |                 |             |
| ToMe (ICLR'23)                                       | VP           | 43.7              | -                 | 50.0              | -                 | 48.6            | 1138            | 52.5            | 45.3            | 57.1            | -               | 70.1%       |
| FastV (ECCV'24)                                      | VP           | 48.0              |                   |                   |                   |                 | 1256            |                 |                 |                 |                 |             |
| HiRED (AAAI'25)                                      | VP           | 60.2              | 52.7              | 51.1              | 50.8              | 46.1 54.6       | 1599            | 48.0 73.6       | 47.8            | 55.0            | 245 191         | 77.3% 87.0% |
| LLaVA-PruMerge (24.05)                               | VP           | 55.3              | 51.4 49.1         | 68.2 68.1         | 50.2 50.1         | 51.9            | 1549            | 65.3            | 44.2 54.0       | 69.7 67.4       | 250             | 87.4%       |
| SparseVLM (ICML'25)                                  | PA           | 56.2              | 46.1              | 62.2              | 50.1              | 52.7            | 1505            | 75.1            | 51.8            | 68.2            | 180             | 84.6%       |
| PyramidDrop (CVPR'25)                                | PA           | 58.8              | 50.5              | 68.6              | 50.7              | 41.9            | 1561            | 55.9            | 45.9            | 69.2            | 250             | 78.1%       |
| FiCoCo-V (EMNLP'24)                                  |              |                   |                   |                   |                   |                 |                 | 76.0            | 53.6            | 71.3            |                 |             |
|                                                      | VP           | 60.3              | 53.0              | 68.1              | 49.8              | 52.4            | 1591            |                 |                 |                 | -               | 91.5%       |
| MustDrop (24.11) VisionZip (24.12)                   | PA VP VP     | 60.0 60.1         | 53.1 -            | 63.4 69.0         | 51.2 -            | 53.1 55.1       | 1612 1690       | 68.0 77.0       | 54.2 55.5       | 69.3 72.4       | 267 -           | 90.1% 92.8% |
| DART (EMNLP'25)                                      | VP           | 60.6              | 53.2              | 69.8              | 51.6              | 55.9            | 1765            | 73.9            | 54.4            | 72.4            | 270             | 93.7%       |
|                                                      |              | 62.0              |                   |                   |                   | -               |                 | 79.9            | 57.0            | 74.8            | -               | 97.0%       |
| TokenCarve (25.03)                                   | PA VP        |                   | -                 | 69.7              | 51.4              |                 | 1754            |                 | 55.8            | 75.6            |                 |             |
| TwigVLM (ICCV'25) -prior)                            | PA           | 60.4              | -                 | 70.0              | -                 | 58.8            | 1760            | 82.7            |                 |                 | -               | 96.1%       |
| MoB (w/o η + η -prior                                | PA VP -      | 61.7 62.1         | 54.2 54.5         | 69.7 69.8         | 52.0 52.1         | 59.0 59.0       | 1806 1806       | 77.2 77.2       | 57.0 57.0       | 75.5 75.5       | 277 277         | 96.3% 96.4% |
| LLaVA-Next-7B                                        |              |                   | w/o               | Pruning, N        | = 2880            | ; Token         | Reduction       | Rate            | = 0.0%          |                 |                 |             |
| Vanilla [29]                                         | -            | 67.4              | 60.6              | 70.1              | 57.6              | 64.2            | 1851            | 86.5            | 64.9            | 81.8            | 517             | 100%        |
| LLaVA-Next-7B                                        |              |                   | Pruning           | budget            | K = 320           | ; Token         |                 | Reduction Rate  | = 88.9%         |                 |                 |             |
| FastV (ECCV'24)                                      | VP           | 61.6              | 51.9              | 62.8              | 53.1              | 55.9            | 1661            | 71.7            | 55.7            | 71.9            | 374             | 86.4%       |
| HiRED (AAAI'25)                                      | VP           | 64.2              | 55.9              | 66.7              | 54.2              | 59.3            | 1690            | 83.3            | 58.8            | 75.7            | 404             | 91.8%       |
| LLaVA-PruMerge (24.05)                               | VP           | 61.3              | 55.3              | 66.4              | 54.0              | 53.6            | 1534            | 60.8            | 50.6            | 69.7            | 146             | 79.9%       |
| SparseVLM (ICML'25)                                  | PA           | 60.6              | 54.5              | 66.1              | 52.0              | 56.1            | 1533            | 82.4            | 58.4            | 71.5            | 270             | 85.9%       |
| PyramidDrop (CVPR'25)                                | PA           | 63.4              | 56.2              | 67.5              | 54.1              | 56.4            | 1663            | 77.6            | 54.4            | 73.5            | 259             | 86.8%       |
| MustDrop (24.11)                                     | PA VP        | 62.8              | 55.1              | 68.0              | 54.0              | 57.3            | 1641            | 82.1            | 59.9            | 73.7            | 382             | 90.4%       |
| VisionZip (24.12)                                    | VP           | 63.1              | -                 | 67.3              | -                 | 59.3            | 1702            | -               | 58.9            | 76.2            | -               | 93.0%       |
| FasterVLM (24.12) [57]                               | VP           | 61.6              | 53.5              | 66.5              | 52.6              | 56.9            | 1701            | 83.6            | 56.5            | 74.0            | 401             | 89.8%       |
| GlobalCom 2 (25.01) [32]                             | VP           | 61.8              | 53.4              | 67.4              | 54.6              | 57.1            | 1698            | 83.8            | 57.2            | 76.7            | 375             | 90.3%       |
| DART (EMNLP'25)                                      | VP           | 65.3              | 58.2              | 68.4              | 56.1              | 61.7            | 1710            | 84.1            | 58.7            | 79.1            | 406             | 93.9%       |
| TwigVLM (ICCV'25)                                    | PA           | 65.0              | -                 | 68.7              | -                 | 62.2            | 1758            | -               | 57.4            | 79.7            | -               | 95.4%       |
| MoB (with η -prior)                                  | PA VP        | 65.8              | 58.9              | 68.7              | 57.0              | 62.6            | 1760            | 84.4            | 60.2            | 80.1            | 418             | 95.4%       |

## D.2 Visualization

/37

/0 /1

/2

/3 /4

/5

/i255

/7 /4

/8

/4

<!-- image -->

/38

/37

/39

/40

/39

/40

/41

/42

/i255

/41

/44

/45

/38

/46

Figure 7: Visualization of the selected prompt and visual centers under weak coupling.

/9 /10

/11

/12 /13

/14

/15

/16

/17

/35/20

/22

/14

/i255

/25

/10

/22

/23

/26

/i255

/11

/36

/i255

/24 /22

/14

/32

/20

/i255

/27

/15

/i255

/14

/20

/27

/15

/29 /30

/47

/48

/49

/50

/51

/52

/53

/49

/51

/52

/51

/53

/54

/53

/54

/55

/53

/i255

/55

/52

/57

/55

/58

<!-- image -->

<!-- image -->

/9 /10

/11

/12 /13

/16

/15

/14

/17

/18 /19

/21

/20

/19

/21

/i255

/i255

/24

/23

/25

/26

/14

/i255

/19

/26

/23

/i255

/15

/15

/11

/21

/20

/23

/14

/26

/27

/i255

/14

/26

/28

/14

/i255

/29 /20

/14

/19

/i255

/14

/19

/20

/15

/i255

/20

/12 /23

/30

/26

/i255

/13

/11

/15

/14

/26

/27

/i255

/11

/24

/i255

/14

/29 /20

/31

/26

/10 /32

<!-- image -->

/9 /10

/11

/12 /13

/14

/15

/16

/17

/18/19

/26

/10

/26

/i255

/20

/15

/i255

/20

/14

/33

/i255

/34

/35

/i255

/36

/19

/23

/24

/30

/19

/23

/20

/35

/i255

/37

/35

/i255

/38

/26

/29 /i255

/39

/11

/10

/40

/35

/i255

/41

/35

/i255

/18 /23

/15

/19

/20

/24

/30

/14

/11

/24

/35

/i255

/42

/35

/i255

/9

/23

/10

/20

/32

<!-- image -->

/i255

/11

/43

/i255

/14

/19

/26

/12 /23

/20

/i255

/30

/26

/33

/34

/i255

/29 /11

/14

/i255

/35

/i255

/27

/11

/24

/44

/14

/15

/35

/i255

/45/32

/9 /10

/11

/12 /13

/14

/16

/15

/17

/18/19

/20

/21

/19

/i255

/20

/15

/i255

/26

/19

/14

/i255

/12 /23

/20

/24

/i255

/14

/11

/13

/20

/21

<!-- image -->

/35

/i255

/46

/19

/26

/i255

/12 /23

/24

/i255

/20

/15

/i255

/19

/11

/47

/27

/20

/24

/30

/i255

/14

/19

/26

/i255

/15

/20

/30

/24

/35

/i255

/45 /32

/9 /10

/11

/12 /13

/14

/15

/16

/17

/18/19

/20

/21

/19

/i255

/20

/15

/i255

/10

/20

/30

/19

/14

/33

/i255

/45 /i255

/41

Figure 8: Visualization of the selected prompt and visual centers under strong coupling.

/13

/20

/21

/14

/26 29

/27

/i255

/20

/26

/i255

/27

/20

/15

/14

/i255

/12 /26

/24

/24

/10

/11

/20

/49

/26

/24

/43

/i255

/26

/i255

/11

/13

/i255

/14

/48

/14

/18/19

/23

/16

/17

/10

/26

/24

/50

/15

/i255

/13

/47

/23

/48

/27

/20

/47

/19

/i255

/41

/35

/45 /i255

/37

/10

/26

/33

/i255

/14

/44

/20

/21

/i255

/13

/24

/i255

/14

/19

/26

/9 /10

/11

/12 /13

/14

/15

/30

/10

/11

/44

/24

/27

/35

/i255

/45 /32

/59

/60

/61

/62

/63

/64

/65

/61

MoB formulates visual token pruning as a bi-objective covering problem over ( V , P ) , which is expected to gather query-relevant, fine-grained evidence with S p while preserving global scene context with S v . The visualizations (Figures 7 and 8) qualitatively validate this design: tokens in S p concentrate in regions aligned with the text query and key visual evidence, whereas elements of S v spread more uniformly across the image to maintain the overall context. Together, this complementary allocation enables MoB to retain the most informative visual content for each image-query pair, accounting for its strong empirical performance.

## D.3 Additional Ablation &amp; Discussion

Table 6: Detailed ablation on the covering fold k for GQA and TextVQA.

| GQA          | GQA     | GQA     | GQA     | GQA     | GQA     | GQA     | GQA     | GQA     | GQA     | GQA     | GQA     | GQA     |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| ⟨ K,K p ⟩    | 0       | 2       | 4       | 6       | 8       | 12      | 16      | 24      | 32      | 48      | 64      | 96      |
| ⟨ 64 , 32 ⟩  | 58.3    | 58.8    | 59.0    | -       | 58.7    | -       | 58.2    | -       | 57.4    | -       | -       | -       |
| ⟨ 128 , 64 ⟩ | 60.2    | -       | 60.5    | -       | 60.7    | -       | 60.6    | -       | 60.0    | -       | 59.5    |         |
| ⟨ 192 , 96 ⟩ | 60.6    | -       | -       | 61.1    | -       | 61.2    | -       | 60.9    | -       | 60.7    | -       | 60.5    |
| TextVQA      | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA | TextVQA |
| ⟨ K,K p ⟩    | 0       | 2       | 4       | 6       | 8       | 12      | 16      | 24      | 32      | 48      | 64      | 96      |
| ⟨ 64 , 32 ⟩  | 56.5    | 56.9    | 57.0    | -       | 56.8    | -       | 56.5    | -       | 56.2    | -       | -       | -       |
| ⟨ 128 , 64 ⟩ | 57.1    | -       | 57.5    | -       | 57.7    | -       | 57.7    | -       | 57.2    | -       | 56.8    | -       |
| ⟨ 192 , 96 ⟩ | 57.8    | -       | -       | 58.2    | -       | 58.2    | -       | 58.1    | -       | 57.7    | -       | 57.5    |

To assess MoB's sensitivity to the covering-fold parameter k -particularly under weak coupling with long prompts-we conduct a detailed ablation on k using GQA and TextVQA.

As Table 6 demonstrates, MoB is not overly sensitive to the choice of k , particularly within a clear optimal range. For instance, in both two benchmarks, performance only varies by approximately 0 . 3% for k values between [2 , 8] under ⟨ K = 64 , K p = 32 ⟩ setting.

There is also a principled, theoretical reason for this robustness, which stems from the relationship between the covering fold k , the budget K p , and the length L of prompt tokens P . From covering theory, every prompt token p ∈ P is covered by at least one visual token v ∈ V under the condition K p ≥ kL , thereby ensuring the performance guarantee of MoB. Therefore, as selected k satisfies k ≤ K p /L , the performance will remain stable.

Heuristic for estimating k . In practice, a robust range for k can be inferred from the prompt length L . Given the analysis above, we expect an adaptive, per-sample search for a fine-grained k to yield only limited gains, so we rely on this length-based heuristic instead.

## D.4 Real-life Application

Figure 9: Observations of prompt-visual coupling η across 9 popular benchmarks.

<!-- image -->

Open-domain recipe. MoB is task-agnostic, which does not require pre-defined task labels and can operate online by classifying each sample's coupling pattern. For a given target model, we adopt a two-stage strategy:

- Offline calibration. Analyze the empirical η distributions on a set of representative benchmarks (as shown in Figure 9) and set a robust threshold τ that separates weak vs. strong coupling.
- Online classification and inference. For each incoming query, compute its Hausdorff distance using Algorithm 2 with tractable bilinear complexity O ( NLd ) . Classify the sample by comparing this value to τ , then apply the corresponding budget configuration ( e.g. , K p , k ) and run MoB + forward inference. In practice, this online cost is negligible relative to the pruned forward pass.

Computational Overhead. We provide a detailed cost breakdown for online computation of the Hausdorff distance using Algorithm 2 with complexity O ( NLd ) on LLaVA-1.5-7B and LLaVANext-7B, where N , L , and d denote the numbers of visual tokens, prompt tokens, and the feature dimension, respectively. As shown in Table 7, the measured cost (TFLOPs) of exact Hausdorff computation is orders of magnitude smaller than that of MoB itself and the model's forward pass, yielding a negligible overhead.

Table 7: Computation cost in LLaVA-7B series (TFLOPs)

| LLaVA-1.5 ( N = 576 , L = 10 , d = 4096 )   | LLaVA-1.5 ( N = 576 , L = 10 , d = 4096 )   | LLaVA-1.5 ( N = 576 , L = 10 , d = 4096 )   | LLaVA-1.5 ( N = 576 , L = 10 , d = 4096 )   | LLaVA-1.5 ( N = 576 , L = 10 , d = 4096 )   |
|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
| Model                                       | Vanilla                                     | K = 64                                      | K = 128                                     | K = 192                                     |
| Forward                                     | 8.2                                         | 1.0                                         | 1.9                                         | 2.8                                         |
| Compute d H                                 | 2 . 3e - 5                                  | 2 . 3e - 5                                  | 2 . 3e - 5                                  | 2 . 3e - 5                                  |
| MoB                                         | -                                           | 1 . 7e - 4                                  | 3 . 3e - 4                                  | 4 . 8e - 4                                  |
| LLaVA-Next ( N = 2880 , L = 10 , d = 4096 ) | LLaVA-Next ( N = 2880 , L = 10 , d = 4096 ) | LLaVA-Next ( N = 2880 , L = 10 , d = 4096 ) | LLaVA-Next ( N = 2880 , L = 10 , d = 4096 ) | LLaVA-Next ( N = 2880 , L = 10 , d = 4096 ) |
| Model                                       | Vanilla                                     | K = 320                                     | K = 640                                     | K = 960                                     |
| Forward                                     | 40.5                                        | 4.6                                         | 9.1                                         | 13.6                                        |
| Compute d H                                 | 1 . 2e - 4                                  | 1 . 2e - 4                                  | 1 . 2e - 4                                  | 1 . 2e - 4                                  |
| MoB                                         | -                                           | 3 . 9e - 3                                  | 7 . 6e - 3                                  | 1 . 1e - 2                                  |

Concretely, computing d H ( e.g. , ∼ 1 . 2 × 10 -4 TFLOPs on LLaVA-Next) is insignificant relative to the pruned forward pass ( e.g. , ∼ 4 . 6 TFLOPs at K = 320 ) and, more importantly, to the savings from pruning ( ∼ 35 . 9 TFLOPs). Thus, exact online estimation is not a practical bottleneck; its cost is dwarfed by the efficiency gains of our method. Further acceleration is possible with standard techniques ( e.g. , heuristic support sampling or low-dimensional random projections), although it is unnecessary in our settings.

- Heuristic Sampling: It computes the distance on smaller support sets of the tokens ( V ′ ⊂ V , P ′ ⊂ P ) , which can be constructed via random sampling [46] or more advanced heuristics such as Key-Norm selection [1, 13]. This reduces complexity to O ( N ′ L ′ d ) , where |V ′ | = N ′ , |P ′ | = L ′ .
- Random Projections: For a more theoretically grounded approach, the Johnson-Lindenstrauss (JL) lemma [19] allows us to project embeddings to a much lower dimension ( d ′ ≪ d ) while preserving geometric structure, reducing complexity to O ( NLd ′ ) .

Potential Extensions. A natural extension is to maintain an online estimate of the coupling statistic η during inferencee.g. , a running summary of an approximate ˆ η computed from shallow-layer tokens. As more samples are processed, we expect the empirical distribution of ˆ η to become bimodal (consistent with the benchmark patterns in Figure 9), enabling a data-driven threshold to be derived on the fly that separates weak vs. strong coupling regimes. Using this live threshold, MoB could adapt K p (and k ) per sample or per mini-batch by selecting from a small budget pool or by scheduling K p as a function of ˆ η , with conservative warm-up and safeguards for distribution shift.

## E Omitted Technical Details

## E.1 Proof of Lemma 1

Restatement of Lemma 1 (An Error Bound for Visual Token Pruning) . Under Assumption 1, given any token set with its pruned counterpart X = V ⊔ P , X s = S ⊔ P ⊆ R d , the pruning error bound is given by:

<!-- formula-not-decoded -->

Remark. Here d H ( S , P ) and d H ( S , V ) describe the prompt alignment and visual preservation, while d H ( V , P ) is an inherent term that describes the prompt-visual coupling of input data.

Proof. The intermediate input for any layer and its pruned counterpart are given by

<!-- formula-not-decoded -->

By Equation (1), the Hausdorff distance is symmetric, i.e. ,

<!-- formula-not-decoded -->

and induced by Euclidean distance.

## Step 1. Bound the one-sided distances.

We analyze the distances by considering the membership of the points in the subsets.

Direction 1 ( X → X s ) For any x ∈ X :

Case (i): If x ∈ P , then since P ⊂ X s ,

<!-- formula-not-decoded -->

Case (ii): If x ∈ V , then the candidate points in X s = S ⊔ P can be chosen either from S or P . Thus,

<!-- formula-not-decoded -->

Taking the supremum over x ∈ V yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Equation (1), we derive the distance in direction 1:

<!-- formula-not-decoded -->

Direction 2 ( X s →X ) For any y ∈ X s :

Case (i): If y ∈ P , then as P ⊂ X ,

<!-- formula-not-decoded -->

Case (ii): If y ∈ S , the candidate points in X = V ⊔ P can be chosen from either V or P ; hence

<!-- formula-not-decoded -->

Taking the supremum over y ∈ S yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Equation (1), we derive the distance in direction 2:

<!-- formula-not-decoded -->

## Step 2. Combine the bounds.

By Equation (1), combining the bounds in (E1-2) and (E1-3), we obtain

<!-- formula-not-decoded -->

Based on (E1-1), we have

<!-- formula-not-decoded -->

Loading the Assumption 1, we have the output discrepancy is bounded by

<!-- formula-not-decoded -->

This completes the proof.

## E.2 Proof of Lemma 2

Restatement of Lemma 2 (A Relaxed Error Bound under Practical Budgets) . Under Assumptions 1 and 2, let X = V ⊔ P , X s = S ⊔ P ⊆ R d with |S| = K ≪ N . Partition the retained token set S into two disjoint subsets: S = S p ⊔ S v , devoted to prompt alignment d H ( S p , P ) and visual preservation d H ( S v , V ) , respectively. Then, the pruning error bound reduces to

<!-- formula-not-decoded -->

Proof. By Lemma 1, we obtain

<!-- formula-not-decoded -->

Since min { a, b } ≤ max { a, b } , we have

<!-- formula-not-decoded -->

For any p ∈ P , we have

<!-- formula-not-decoded -->

Taking the supremum over p ∈ P yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Assumption 2 ( d H ( V , P ) ≤ η ) and the triangle inequality for Hausdorff distance, we have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Similarly, one can show that

<!-- formula-not-decoded -->

Loading the maximum of (E2-2), (E2-3) and d H ( V , P ) into (E2-1), we obtain

<!-- formula-not-decoded -->

Since d H ( S p , P ) ≥ 0 , d H ( S v , V ) ≥ 0 , η ≥ 0 , we have

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

This completes the proof.

Similarly, since S v ⊂ S ,

Thus, by Equation (1),

<!-- formula-not-decoded -->

## E.3 Proof of Lemma 3

Restatement of Lemma 3 ( d eff -regular lower bound on covering numbers) . Given P , V ⊂ R d with an effective dimension d eff . Suppose their δ -dilations V δ := ⋃ v ∈V B ( v, δ ) , P δ := ⋃ p ∈P B ( p, δ ) ( δ ≪ η ) satisfy d eff -dimensional covering regularity; thus, there exist constants b&gt;a&gt; 0 , b ′ &gt;a ′ &gt; 0 and ϵ 0 &gt;δ such that

<!-- formula-not-decoded -->

Remark Previous work suggests that both visual and language embeddings concentrate on a lowdimensional manifold, so the effective covering dimension satisfies the typical relation d eff ≪ d .

Proof. We prove the two-sided bound for P ; the argument for V is identical.

## Notation.

- N ( X,r ) : minimal number of closed balls of radius r covering X .

<!-- formula-not-decoded -->

## Step 1. Transfer trick for small ϵ .

Fix ϵ ∈ ( δ, ϵ 0 ] and define ϵ ′ = min { ϵ + δ, ϵ 0 } .

If ϵ ≤ ϵ 0 -δ (so ϵ ′ = ϵ + δ ), then any ϵ -cover { z i } i m =1 of P satisfies for each y ∈ P δ :

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Note: For ϵ &gt; ϵ 0 -δ , the above transfer argument is not applied.

## Step 2. Lower bound on N ( P , ϵ ) .

Split into two cases:

Case I: ϵ ≤ ϵ 0 -δ . Since P δ satisfies d eff -dimensional covering regularity; loading the lower-bound for P δ at radius ϵ ′ = ϵ + δ , there exists a constant a δ ≥ 0 such that

<!-- formula-not-decoded -->

Based on (E3-1), we obtain

Since ϵ &gt; ϵ 0 -δ, we have

Hence

<!-- formula-not-decoded -->

Since δ ≤ ϵ , it follows that ϵ + δ ≤ 2 ϵ ; thus, we have

<!-- formula-not-decoded -->

Case II: ϵ &gt; ϵ 0 -δ . Define ˜ a := ( ϵ 0 -δ ) d eff , such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since any nonempty set P has covering number at least one, the following holds

<!-- formula-not-decoded -->

Therefore, set a := min { a δ 2 -d eff , ˜ a } &gt; 0 , combining (E3-2) and (E3-3) yields

<!-- formula-not-decoded -->

Similarly, V holds N ( V , ϵ ) ≥ a ′ ϵ -d eff , ∀ ϵ ∈ ( δ, ϵ 0 ] .

## Step 3. Upper bound on N ( P , ϵ ) .

Since P δ satisfies d eff -dimensional covering regularity, there exists a constant b δ ≥ a δ ≥ 0 such that

<!-- formula-not-decoded -->

Since P ⊆ P δ , we have N ( P , ϵ ) ≤ N ( P δ , ϵ ) ; thus, the following holds

<!-- formula-not-decoded -->

Based on the monotonicity of covering numbers , for every radius ϵ ≥ δ , we have

<!-- formula-not-decoded -->

Therefore, set b := max { b δ , N ( P , δ ) } , for all ϵ ∈ ( δ, ϵ 0 ] we have

<!-- formula-not-decoded -->

Likewise for V , the following holds N ( V , ϵ ) ≤ b ′ ϵ -d eff , ∀ ϵ ∈ ( δ, ϵ 0 ] .

## Step 4. Combine the bounds.

Based on (E3-4) and (E3-5), for all ϵ ∈ ( δ, ϵ 0 ] the following holds

<!-- formula-not-decoded -->

This completes the proof.

## E.4 Proof of Theorem 1

Restatement of Theorem 1 (Trade-off between Prompt Alignment and Visual Preservation) . Under Assumption 2 and the covering-regularity hypothesis of Lemma 3 with constants a, a ′ , d eff &gt; 0 , there exist a radius-scaling factor z &gt; 1 such that η/z &gt; δ and K &lt; N ( P , η/z ) + N ( V , η/z ) , for every pruning results S = ( S p ⊔ S v ) ⊆ V with budget K satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark (Optimal Attainment Level) . The term D 1 K -2 /d eff is completely determined by the pruning budget, while D 2 η 2 quantifies the effect of prompt-visual coupling. Hence, the optimal attainment level per objective is given by ϵ ∗ = max { η/z, √ D 1 K -1 /d eff } . Any attempt to reduce one objective below ϵ ∗ forces the other above ϵ ∗ , thereby increasing the overall pruning error.

Remark (Effect of Budget and Coupling Strength) . As K decreases, z correspondingly shrinks ( D 2 growing as a power function), ultimately making D 2 η 2 dominate the bound; while as K increases, both of the terms reduce, thereby diminishing the trade-off and tightening the overall error bound.

Proof. We begin the proof by noting

<!-- formula-not-decoded -->

## Step 1. Quantify the impact of budget K .

By Lemma 3, for all ϵ p , ϵ v ∈ ( δ, ϵ 0 ] , we have

<!-- formula-not-decoded -->

By AM-GM inequality, we have K p K v ≤ ( K 2 ) 2 ; thus, loading (E4-1) we have

<!-- formula-not-decoded -->

Define D 1 := (4 aa ′ ) 1 /d eff &gt; 0 , the K -bound is established by

<!-- formula-not-decoded -->

## Step 2. Quantify the impact of prompt-visual coupling η .

Based on the budget condition, the radius-scaling factor z holds

<!-- formula-not-decoded -->

For contradiction, we suppose two covering radii is simultaneously small, such that ϵ p &lt; η/z and ϵ v &lt; η/z . Then, the monotonicity of covering numbers gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence contradicting (E4-3). Therefore at least one of ϵ p , ϵ v is ≥ η/z . Consequently

<!-- formula-not-decoded -->

Define D 2 := 1 z 2 &gt; 0 , the η -bound is given by

<!-- formula-not-decoded -->

## Step 3. Combine the impacts.

By (E4-2) and (E4-4), we have

<!-- formula-not-decoded -->

This completes the proof.

## E.5 Proof of Theorem 2

Restatement of Theorem 2 (Performance Guarantee) . Under Assumption 1 and the coveringregularity of Lemma 3 with constants a, a ′ , d eff &gt; 0 and b&gt;a, b ′ &gt;a ′ , for any budget split ( K p , K -K p ) , covering fold k , and token set X = V ⊔ P ⊆ R d with |V| = N , |P| = L , and d H ( V , P ) ≤ η , the following hold:

- (a) Performance bound: The Performance degradation caused by MoB is upper bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (b) Multilinear complexity: The complexity of MoB is given by T MoB = O ( N ( L + K ) d ) .

Remark (Coupling Trade-off) . Under weak coupling (large α ( η, k, L ) ), minimizing the bound requires a larger K p . Conversely, under strong coupling (small α ( η, k, L ) ), the alignment term decays rapidly, favoring visual preservation (increasing K -K p ). Specially, under perfect coupling ( η = 0 ), the bound simplifies to ∥ ∆ y ∥ ≤ C ℓ β ( K -K p ) -1 /d eff , i.e. , MoB reduces to pure visual preservation.

Remark (Budget Scaling) . As the total budget K increases, the preservation term β ( K -K p ) -1 /d eff decays, requiring a corresponding increase in K p (and thus a reduction in the alignment term) to rebalance the trade-off and further lower the overall error bound.

Remark (Scalability) . MoB exhibits a multilinear scalability with respect to visual tokens N , prompt tokens L , and retained tokens K (especially K,L ≪ N ), making it readily adaptable to more challenging scenarios, such as advanced MLLMs with higher-resolution inputs or multi-frame video.

## Notation.

- The intermediate input X is formulated as

<!-- formula-not-decoded -->

Particularly, V , P are compact sets with d eff effective dimensions.

- We define the pruned intermediate input as

<!-- formula-not-decoded -->

- The budget configuration is given by ⟨ K p , K v ⟩ , where K p + K v = K .

Proof. We separately proof the Performance Guarantee &amp; Complexity in Part A &amp; Part B

## Part A: Performance Guarantee

## Part A-1: Performance Guarantee of prompt alignment

Step A-1.1: Bound of the radius derived by k -fold NN-covering

Given any union set before K p -truncation

<!-- formula-not-decoded -->

we define

<!-- formula-not-decoded -->

By previous work [15], NN-covering achieves a 1 -approximation for the k -center problem with sufficient budget; i.e. , specifically for any p ∈ P we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

Based on Assumption 2, since s ∈ S ′ p ⊆ V , the upper bound of the radius ϵ ′ p is given by

<!-- formula-not-decoded -->

Step A-1.2: Impact of K p -truncation on the radius

Based on Lemma 3, we have

In particular:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and also

Combining the upper and lower bound for ϵ p and ϵ ′ p , respectively in terms of b, K p , K ′ p , we obtain

<!-- formula-not-decoded -->

That is, truncating from K ′ p to K p centers increases the radius by at most the factor

<!-- formula-not-decoded -->

Since kL ≥ K ′ p , loading into above, we have

<!-- formula-not-decoded -->

By loading (E5-1) into the above, the performance guarantee of prompt alignment is given by

<!-- formula-not-decoded -->

## Part A-2: Performance Guarantee of Visual Preservation

By previous work [36], FPS achieves a 2 -approximation for the k -center problem:

<!-- formula-not-decoded -->

where ϵ ⋆ ( K v ) is the optimal radius with K v centers. Based on Lemma 3, we have

<!-- formula-not-decoded -->

thereby, the upper bound of optimal radius is given by

<!-- formula-not-decoded -->

By loading the above into (E5-3), the performance guarantee of visual preservation is given by

<!-- formula-not-decoded -->

## Part A-3: Performance Guarantee of MoB

By substituting (E5-2) and (E5-4) into Lemma 2, the performance guarantee of the MoB is given by:

<!-- formula-not-decoded -->

where α ( η, k, L ) = η ( b k L/a ) 1 /d eff , β = 2 b ′ 1 /d eff .

This completes the proof of Part A.

<!-- formula-not-decoded -->

## Part B: Complexity

Since k ≪ K p ≤ K ∼ L ≪ N , we restrict our complexity analysis to the leading-order terms.

## Part B-1: Normalization

MoB do a L 2 normalization for each token x ∈ X ⊆ R d ; thus, the complexity is given by

<!-- formula-not-decoded -->

## Part B-2: Selection of Prompt Center

Firstly, MoB calculates the cosine similarity with each p ∈ P and v ∈ V via a matrix multiplication:

<!-- formula-not-decoded -->

which leads a complexity of T step 1 -1 = O ( N Ld ) . Subsequent, MoB do a topk retrieval in the first dimension of M sim the select k most closed centers for each prompt token p ∈ P , which can be reduced to a partial sorting, thereby leading to a complexity of T step 1 -2 = O ( N L log k ) . Finally, MoB merge the selected result of each p ∈ P , and truncated the topK p ones with largest similarity, leading to a T step 1 -3 = O ( Lk log K p ) . Consequently, the total complexity T p -select of prompt center selection is given by:

<!-- formula-not-decoded -->

## Part B-3: Selection of Visual Center

Initially, MoB calculates the minimum distance (used in FPS) with each visual token v ∈ V\S p := V ′ and the selected prompt centers via a matrix multiplication together with an argmin operator:

<!-- formula-not-decoded -->

thus, the complexity is given by

<!-- formula-not-decoded -->

Subsequently, in K -K p iterations, MoB add the tokens with largest minimum distance with an argmax operator in d FPS , and update the d FPS with an inner production together with an N -K p -dimensional element-wise comparison; thus the complexity is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, the total complexity T v -select of visual center selection is given by:

<!-- formula-not-decoded -->

## Part B-4: Totally complexity

By (E5-5), (E5-6) and (E5-7), the totally complexity of MoB is given by

<!-- formula-not-decoded -->

This completes the proof of Part B.

Combining the Part A &amp; B, we complete the proof.
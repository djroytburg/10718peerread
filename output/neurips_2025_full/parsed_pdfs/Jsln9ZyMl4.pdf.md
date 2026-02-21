## The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets

Yujun Kim

∗ Chaewon Moon ∗ Chulhee Yun KAIST

{kyujun02, chaewon.moon, chulhee.yun}@kaist.ac.kr

## Abstract

We study the parameter complexity of robust memorization for ReLU networks: the number of parameters required to interpolate any given dataset with ϵ -separation between differently labeled points, while ensuring predictions remain consistent within a µ -ball around each training sample. We establish upper and lower bounds on the parameter count as a function of the robustness ratio ρ = µ/ϵ . Unlike prior work, we provide a fine-grained analysis across the entire range ρ ∈ (0 , 1) and obtain tighter upper and lower bounds that improve upon existing results. Our findings reveal that the parameter complexity of robust memorization matches that of non-robust memorization when ρ is small, but grows with increasing ρ .

## 1 Introduction

The topic of memorization investigates the expressive power of neural networks required to fit any given dataset exactly. This line of inquiry seeks to determine the minimal network size-measured in the number of parameters, or equivalently, parameter complexity-needed to interpolate any finite collection of N labeled examples. A number of works study both upper and lower bounds on the parameter complexity [Baum, 1988, Yun et al., 2019, Bubeck et al., 2020, Park et al., 2021]. The VC-dimension implies a lower bound of Ω( √ N ) [Chervonenkis, 2015, Goldberg and Jerrum, 1995, Bartlett et al., 2019], while Vardi et al. [2021] show that ˜ Θ( √ N ) parameters suffice for ReLU networks. Together, these results establish that memorizing any N distinct samples with ReLU networks can be done with ˜ Θ( √ N ) parameters, tight up to logarithmic factors.

Wenow turn to a more challenging task beyond mere interpolation of data: robust memorization . We aim to quantify the additional parameter complexity required for a network to remain robust against adversarial attacks, going beyond standard non-robust memorization. To address the sensitivity of neural networks to small adversarial perturbations [Szegedy et al., 2014, Goodfellow et al., 2015, Ding et al., 2019, Gowal et al., 2021, Zhang et al., 2021, Bastounis et al., 2025], we consider the setting in which not only the data points but all points within a distance µ -referred to as the robustness radius -from each data point must be mapped to the corresponding label. More concretely, for any dataset with ϵ -separation between differently labeled data points, the network must memorize the dataset and the prediction must remain consistent within a µ -ball centered at each training sample. As will be seen shortly, the parameter complexity for robust memorization is governed by the robustness ratio ρ = µ/ϵ ∈ (0 , 1) rather than the individual values of µ and ϵ . However, a precise understanding of how this complexity scales with ρ remains limited.

∗ Authors contributed equally to this paper.

## 1.1 What is Known So Far?

Existing Lower Bounds. Since classical memorization requires Ω( √ N ) parameters, it follows that robust memorization must also satisfy a lower bound of at least Ω( √ N ) parameters for any ρ ∈ (0 , 1) . A lower bound specific to robust memorization is established by the work of Li et al. [2022], which shows that for input dimension d , Ω( √ Nd ) parameters are necessary for robust memorization under ℓ 2 -norm for sufficiently large ρ . However, the authors do not characterize the range of ρ over which this lower bound remains valid. Our Proposition 3.3 presented later shows that the Ω( √ Nd ) lower bound can be extended to the range ρ ∈ ( √ 1 -1 /d, 1 ) . Combining these observations, we obtain the following unified lower bound: suppose that for any dataset D with input dimension d and size N , there exists a neural network with at most P parameters that robustly memorizes D with robustness ratio ρ under ℓ 2 -norm. Then, the number of parameters P must satisfy

<!-- formula-not-decoded -->

where the d term accounts for the parameters connected to the input neurons. In the setting d = O ( √ N ) , the lower bounds increase discontinuously from √ N to √ Nd .

While our main analysis focuses on the ℓ 2 -norm, there also exist results under the ℓ ∞ -norm. In particular, Yu et al. [2024] show that under the ℓ ∞ -norm and certain assumptions, ρ -robust memorization requires the first hidden layer to have width at least d . Our analysis not only strengthens but also generalizes this ℓ ∞ -norm result by removing the assumption on the dataset-made in prior work-that the number of data points must be greater than d .

Existing Upper Bounds. From the work of Yu et al. [2024], it is proven that O ( Nd 2 ) parameters suffice for any ρ ∈ (0 , 1) . See Appendix D.2 for an analysis of the parameter complexity of their construction. Furthermore, Egosi et al. [2025] show that for ρ ∈ ( 0 , 1 √ d ) , a network of width log N suffices for ρ -robust memorization. Although they do not explicitly quantify the total number of parameters, their construction with a width log N network requires ˜ O ( N ) parameters, as we verify in Appendix D.3. Additionally, we state that their construction implicitly yields a smooth interpolation between ˜ O ( N ) and ˜ O ( Nd 2 ) as ρ varies within the intermediate range (1 / √ d, 1 / 6 √ d ) .

To sum up, the existing upper bound states that for any dataset D with input dimension d and size N , there exists a neural network that achieves robust memorization on D with the robustness ratio ρ under ℓ 2 -norm, with the number of parameters P bounded as follows:

<!-- formula-not-decoded -->

When d = O ( N ) , the upper bound transitions continuously from ˜ O ( N ) to ˜ O ( Nd 2 ) .

## 1.2 Summary of Contribution

We investigate how the number of parameters required for robust memorization in ReLU networks varies with the robustness ratio ρ . We improve both upper and lower bounds on the minimal number of parameters over all possible ρ ∈ (0 , 1) , which are tight in some regimes and substantially reduce the existing gap elsewhere. The improvement across different regimes of ρ is visualized in Figure 1.

- Necessary Conditions for Robust Memorization. We show that the first hidden layer must have a width of at least ρ 2 min { N,d } , by constructing a dataset that cannot be robustly memorized using a smaller width. Consequently, the network must have at least Ω( ρ 2 min { N,d } d ) parameters. Moreover, we prove that at least Ω( √ N/ (1 -ρ 2 )) parameters are necessary for ρ ≤ √ 1 -1 /d by analyzing the VC-dimension. Combining these two results, we obtain a tighter lower bound on the parameter complexity of robust memorization of the form

<!-- formula-not-decoded -->

<!-- image -->

√

Figure 1: Summary of parameter bounds on a log-log scale when d = Θ( N ). We omit constant factors in both axes. Solid blue and red curves show the sufficient (Theorem 4.2) and necessary (Theorem 3.1) numbers of parameters, respectively; the solid black curves are the best prior bounds. Light-blue shading highlights our improvement in the upper bound, and light-red shading highlights our improvement in the lower bound. The cross-hatched area marks the remaining gap. Notably, this gap disappears in the smallest ρ regime. The yellow and green dashed line denotes the first term (Proposition 3.2) and the second term (Proposition 3.3) in Theorem 3.1, respectively.

- Sufficient Conditions for Robust Memorization. We establish improved upper bounds on the parameter count by analyzing three distinct regimes of ρ , tightening the bound in each case. For ρ ∈ ( 0 , 1 5 N √ d ] , we achieve robust memorization using ˜ O ( √ N ) parameters, matching the existing lower bound. For ρ ∈ ( 1 5 N √ d , 1 5 √ d ] , we obtain robust memorization with ˜ O ( Nd 1 / 4 ρ 1 / 2 ) parameters up to an arbitrarily small error, which interpolates between the existing lower bound Ω( √ N ) and the existing upper bound ˜ O ( N ) . Finally, for larger values of ρ , where ρ ∈ ( 1 5 √ d , 1 ) , robust memorization is achieved with ˜ O ( Nd 2 ρ 4 ) parameters, which interpolates between the existing upper bound ˜ O ( N ) and ˜ O ( Nd 2 ) .

All together, we provide, to the best of our knowledge, the first theoretical analysis of parameter complexity for robust memorization that characterizes its dependence on the robustness ratio ρ over the entire range ρ ∈ (0 , 1) . Notably, when ρ &lt; 1 5 N √ d , the same number of parameters as in classical (non-robust) memorization suffices for robust memorization. These results suggest that, in terms of parameter count, achieving robustness against adversarial attacks is relatively inexpensive when the robustness radius is small. As the radius grows, however, the number of required parameters increases, reflecting the rising cost of achieving stronger robustness.

## 2 Preliminaries

## 2.1 Notation

Throughout the paper, we use d to denote the input dimension of the data, N to denote the number of data points in a dataset, and C to denote the number of classes for a classification task. For a natural number n ∈ N , [ n ] denotes the set { 1 , 2 , . . . , n } .

For two sets A,B ⊆ R d , we denote the ℓ 2 -norm distance between A and B as dist 2 ( A,B ) := inf {∥ a -b ∥ 2 | a ∈ A, b ∈ B } , where ∥·∥ 2 denotes the Euclidean norm. When either A or B is a singleton set, such as { a } or { b } , we identify the set with the element and write a or b in place of A or B , respectively; for example, dist 2 ( a , B ) . In the case d = 1 , we omit the subscript 2 and write dist( · , · ) to denote the standard absolute distance on R . We use B 2 ( x , µ ) = { x ′ | ∥ x ′ -x ∥ 2 &lt; µ } to denote an open Euclidean ball centered at x with a radius µ .

We use ˜ O ( · ) to hide the poly-logarithmic dependencies in problem parameters such as N , d , and ρ .

## 2.2 Dataset and Robust Memorization

̸

For d ≥ 1 and N ≥ C ≥ 2 , let D d,N,C be the collection of all datasets of the form D = { ( x i , y i ) } N i =1 ⊂ R d × [ C ] , such that x i = x j for all i = j and has at least one data point per each class label. Hence, any D ∈ D d,N,C is a pairwise distinct d -dimensional dataset of size N with labels in [ C ] .

̸

Definition 2.1. For D ∈ D d,N,C , the separation constant ϵ D is defined as

̸

<!-- formula-not-decoded -->

̸

Since the datasets we consider have at least one data point for each class label, the set we minimize over is nonempty. Moreover, since we consider D with x i = x j for all i = j , we have ϵ D &gt; 0 . Next, we define robust memorization of the given dataset.

̸

Definition 2.2. For D ∈ D d,N,C and a given robustness ratio ρ ∈ (0 , 1) , define the robustness radius as µ := ρϵ D . We say that a function f : R d → R ρ -robustly memorizes D if

<!-- formula-not-decoded -->

and B 2 ( x i , µ ) is referred to as the robustness ball of x i .

̸

̸

When ρ = 0 , robust memorization reduces to classical memorization, which requires f ( x i ) = y i for all ( x i , y i ) ∈ D . We emphasize that the range ρ ∈ (0 , 1) covers the entire regime in which robust memorization is possible. Specifically, for ρ &gt; 1 , requiring memorization of ρϵ D -radius neighbor of each data point leads to a contradiction as B 2 ( x i , ρϵ D ) ∩ B 2 ( x j , ρϵ D ) = ∅ for some y i = y j . Moreover, if ρ = 1 , any continuous function f cannot ρ -robustly memorize D . If f is continuous and 1 -robustly memorizes D , we have f ( B 2 ( x i , ϵ D )) = { y i } for all i ∈ [ N ] , where B 2 ( x i , ϵ D ) is the closed ball with center x i and radius ϵ D . Since B 2 ( x i , ϵ D ) ∩ B 2 ( x j , ϵ D ) = ∅ for some y i = y j , this leads to a contradiction.

## 2.3 ReLU Neural Network

We define the neural network f recursively over L layers:

<!-- formula-not-decoded -->

where the activation σ ( u ) := max { 0 , u } is the element-wise ReLU . We use d 1 , . . . , d L -1 to denote the widths of the L -1 hidden layers. We define the width of the network to be the maximum hidden layer width, max ℓ ∈ [ L -1] d ℓ . For ℓ ∈ [ L ] , the symbols W ℓ ∈ R d ℓ × d ℓ -1 and b ℓ ∈ R d ℓ denote the weight matrix and the bias vector for the ℓ -th layer, respectively; here, we use the convention d 0 = d and d L = 1 .

We count the number of parameters P of f as the count of all entries in the weight matrices and biases { W ℓ , b ℓ } L ℓ =1 (including entries set to zero), as

<!-- formula-not-decoded -->

This reflects the common convention of parameter counting in practice. The set of neural networks with input dimension d and at most P parameters is denoted as

<!-- formula-not-decoded -->

Although less relevant in practice, some prior work counts only nonzero entries when reporting the number of parameters. Appendix E adopts this alternative counting scheme and explains how our results translate under it, enabling comparisons with prior studies from a different perspective. Even then, the key findings of this paper remain true: for small ρ , robustness incurs no additional parameter cost, whereas as ρ grows, the number of required parameters increases.

̸

̸

## 2.4 Why Only ρ = µ/ϵ D Matters

We describe both necessary and sufficient conditions for robust memorization in terms of the ratio ρ = µ/ϵ D , rather than describing it in terms of individual values µ and ϵ D . This is because the results remain invariant under scaling of the dataset.

Specifically regarding the sufficient condition, suppose f ρ -robustly memorizes D with robustness radius µ = ρϵ D . Then for any c &gt; 0 , the scaled dataset c D := { ( c x i , y i ) } N i =1 , whose separation ϵ c D = cϵ D , can be ρ -robustly memorized with robustness radius cµ by the scaled function x ↦→ f ( 1 c x ) . Moreover, the scaled function can be implemented through a network with the same number of parameters as the neural network f via scaling the first hidden layer weight matrix by 1 /c .

On the other hand, this implies that the necessary condition can also be characterized in terms of ρ . Suppose we have a dataset D with a fixed ϵ D for which ρ -robustly memorizing it requires a certain number of parameters P . Then, the scaled dataset c D with a separation ϵ c D = cϵ D also requires the same number of parameters for ρ -robust memorization. If c D can be ρ -robustly memorized with less than P parameters, then by parameter rescaling from the previous paragraph, D can also be ρ -robustly memorized with less than P parameters, leading to a contradiction.

Hence, the robustness ratio ρ = µ/ϵ D captures the essential difficulty of robust memorization, independent of scaling. We henceforth state our upper and lower bounds in terms of ρ .

## 3 Necessary Number of Parameters for Robust Memorization

In this section, we establish necessity conditions on the number of parameters and the width of neural networks for robust memorization, expressed in terms of the robustness ratio ρ ∈ (0 , 1) . The following theorem presents our main lower bound result on the parameter complexity of robust memorization.

Theorem 3.1. Let ρ ∈ (0 , 1) . Suppose for any D ∈ D d,N, 2 , there exists a neural network f ∈ F d,P that can ρ -robustly memorize D . Then, the number of parameters P must satisfy

<!-- formula-not-decoded -->

The proof of Theorem 3.1 is provided in Appendix A.1. The theorem states a necessary condition on the number of parameters for binary classification ( C = 2 ). The same bound applies to C &gt; 2 : any classifier that robustly memorizes a multiclass dataset can be converted into a one-vs-rest binary classifier by appending a final two-parameter layer (one weight and one bias) that separates a designated label from the others. Therefore, a multiclass task requires at least the parameter scale needed for the binary case. Hence, Theorem 3.1 extends to C &gt; 2 . Moreover, while Theorem 3.1 focuses on ℓ 2 -norm, we extend the necessity results to general ℓ p -norm in Theorem C.5. The lower bound on the number of parameters consists of two parts: one derived from the requirement on the first hidden layer width and the other from the VC-dimension.

First Term: Necessary Condition by the First Hidden Layer Width. The first term Ω(( ρ 2 min { N,d } +1) d ) comes from the following proposition on the first hidden layer width.

Proposition 3.2. There exists D ∈ D d,N, 2 such that, for any ρ ∈ (0 , 1) , any neural network f : R d → R that ρ -robustly memorizes D must have the first hidden layer width at least ρ 2 min { N -1 , d } .

For any fixed N,d , we can choose a single dataset D that enforces the bound simultaneously for all ρ ∈ (0 , 1) : every ρ -robust memorizer of D must have the first hidden layer width at least ρ 2 min { N -1 , d } . Section 5.1 treats the simple case N -1 = d to illustrate the construction and provide a sketch of proof, while Appendix A.2 provides the full proof for the general case.

Proposition 3.2 for the ℓ 2 -norm extends to the general ℓ p -norm in Proposition C.6. For every p ≥ 2 , the same lower bound on the first hidden layer width, ρ 2 min { N -1 , d } , holds. For 1 ≤ p &lt; 2 , a nontrivial lower bound still holds. Furthermore, for the ℓ ∞ -norm, we strengthen the result of Yu et al. [2024]-while they show that width at least d is necessary when N &gt; d and ρ ≥ 0 . 8 , we obtain the stronger width requirement min { N -1 , d } for any ρ ∈ (1 / 2 , 1) , without the assumption N &gt; d , as formalized in Proposition C.7.

We now discuss the implications of Proposition 3.2 on the parameter complexity in Theorem 3.1. Since the input dimension is d , any neural network f : R d → R with the first hidden layer width m must have at least md parameters. Moreover, we have a trivial lower bound m ≥ 1 . Hence, the lower bound of width m becomes max { ρ 2 min { N -1 , d } , 1 } ≥ 1 2 ( ρ 2 min { N -1 , d } +1) , yielding a necessity of Ω(( ρ 2 min { N,d } +1) d ) parameters in Theorem 3.1. The width from Proposition 3.2 dominates over the trivial lower bound of 1 whenever ρ ≥ 1 / √ min { N -1 , d } .

Let us compare the result with Egosi et al. [2025], where they show logarithmic width in N is sufficient under the restricted condition of ρ ≤ 1 / √ d for robust memorization. Our necessary condition on width does not conflict with their logarithmic sufficiency, as their sufficiency holds only under ρ ≤ 1 / √ d , in which our lower bound becomes trivial.

On the other hand, the necessary condition on width by Egosi et al. [2025] given as 2 log N/ log(4832 ρ -1 ) exceeds the trivial lower bound 1 only when ρ ≥ 4832 /N . Even in the case where their lower bound becomes nontrivial, their bound is still at the ˜ O (1) scale, so that our lower bound either becomes tighter or matches their bound up to a polylogarithmic factor over all ρ ∈ (0 , 1) . As a side note, although we generally ignore polylogarithmic factors, we may also consider logarithmic terms for completeness. Under this consideration, the lower bound of Egosi et al. [2025] remains logarithmically nontrivial while ours remains trivial for 4832 /N &lt; ρ &lt; 1 / √ min { N -1 , d } , provided that such ρ exists.

Second Term: Necessary Condition by the VC-Dimension. Now, let us look at the necessary number of parameters given by the VC-dimension of the function class.

Proposition 3.3. Let ρ ∈ ( 0 , √ 1 -1 d ] . Suppose for any D ∈ D d,N, 2 , there exists f ∈ F d,P that ρ -robustly memorizes D . Then, the number of parameters P must satisfy

<!-- formula-not-decoded -->

The detailed proof of Proposition 3.3 is in Appendix A.3 and its extension to the ℓ p -norm appears in Proposition C.8. Before presenting our approach, we briefly review how the existing bound is obtained using VC-dimension arguments. Gao et al. [2019], Li et al. [2022] prove that for sufficiently large ρ , whenever F d,P contains ρ -robust memorizer of any D ∈ D d,N, 2 , then VC -dim( F d,P ) = Ω( Nd ) . Combining this with a known upper bound VC -dim( F d,P ) = O ( P 2 ) [Goldberg and Jerrum, 1995], they obtain P = Ω( √ Nd ) .

However, the prior lower bound Ω( √ Nd ) is only known to apply for sufficiently large ρ , without specifying the precise range. Before our result, the only lower bound applicable to all ρ -including small ρ regime-was the one that trivially comes from non-robust memorization: Ω( √ N ) . A wide range of ρ lacks a VC-dimension-based lower bound tailored to robust memorization.

In Proposition 3.3, we carefully characterize how the VC-dimension scales over the range ρ ∈ (0 , √ 1 -1 / d ] . In this range of ρ , we show whenever F d,P contains ρ -robust memorizer of any D ∈ D d,N, 2 , then VC -dim( F d,P ) = Ω( N / 1 -ρ 2 ) ; this thus gives the tighter bound P = Ω( √ N / 1 -ρ 2 ) . At the endpoint ρ = √ 1 -1 / d , Proposition 3.3 implies that Ω( √ Nd ) parameters are required. Therefore, the same lower bound applies for all ρ ≥ √ 1 -1 / d , characterizing the regime in which the existing bound of √ Nd holds. By combining Proposition 3.3 over ρ ∈ (0 , √ 1 -1 / d ] and the Ω( √ Nd ) bound over ρ ∈ ( √ 1 -1 / d , 1) , we obtain the second term Ω(min { 1 / √ 1 -ρ 2 , √ d } √ N ) in Theorem 3.1.

Finally, we clarify why Proposition 3.3 is stated for ρ ≤ √ 1 -1 / d and why, for ρ &gt; √ 1 -1 / d , this approach cannot improve upon the √ Nd scale. Any such improvement via VC-dimension would require showing that VC -dim( F d,P ) strictly exceeds Nd , i.e., that a ρ -robust memorizer in R d shatters more than Nd points. Our shattering argument shows that robustly memorizing two arbitrary points forces shattering of (a subset of) the standard basis directions in R d ; iterating over N / 2 disjoint pairs can yield Nd / 2 shattered points. Consequently, our current construction neither establishes that a robust memorizer of N points can shatter beyond the Nd scale, nor that a robust memorizer of two points can shatter beyond the d scale. Thus, within this framework, the VC-dimension cannot

be pushed beyond Nd scale, and the induced parameter lower bound does not improve beyond the √ Nd scale for ρ &gt; √ 1 -1 / d .

## 4 Sufficient Number of Parameters for Robust Memorization

In this section, we establish sufficient conditions on the number of parameters for robust memorization, thereby complementing the lower bounds presented in the previous section. In fact, one of our upper bound results is derived under a relaxed definition of robust memorization. For this, we define ρ -robust memorization error of a neural network.

Definition 4.1. For any D ∈ D d,N,C , we define the ρ -robust memorization error of a network f : R d → R on D as

̸

<!-- formula-not-decoded -->

where µ = ρϵ D . When L ρ ( f, D ) &lt; η , we say f can ρ -robustly memorize D with error at most η .

Note that if a network f ρ -robustly memorizes D (as in Definition 2.2), then the error is zero; that is, by definition L ρ ( f, D ) = 0 .

We now state our main upper bounds, showing that any given dataset in D d,N,C can be ρ -robustly memorized by a network with ρ -dependent number of parameters.

Theorem 4.2. For any dataset D ∈ D d,N,C and η ∈ (0 , 1) , the following statements hold:

- (i) If ρ ∈ ( 0 , 1 5 N √ d ] , there exists f ∈ F d,P with P = ˜ O ( √ N ) that ρ -robustly memorizes D .
- (ii) If ρ ∈ ( 1 5 N √ d , 1 5 √ d ] , there exists f ∈ F d,P with P = ˜ O ( Nd 1 4 ρ 1 2 ) that ρ -robustly memorizes D with error at most η .
- (iii) If ρ ∈ ( 1 5 √ d , 1 ) , there exists f ∈ F d,P with P = ˜ O ( Nd 2 ρ 4 ) that ρ -robustly memorizes D .

We note that we omitted the trivial additive factor d that accounts for parameters connected to input neurons. The three regimes in Theorem 4.2-each referred to as small, moderate, and large ρ regime respectively-collectively cover all values of ρ ∈ (0 , 1) and provide explicit upper bound complexity for robust memorization. Moreover, the constructions behind Theorem 4.2 use a single network architecture that depends only on the problem parameters N,d,C,ρ and not on the dataset: for every D ∈ D d,N,C and given ρ , choosing appropriate weights and biases on this same architecture achieves the stated guarantee.

We present a proof sketch in Section 5.2 and the detailed proof in Appendix B. The extended version of Theorem 4.2, which additionally states the explicit bounds on depth, width, and bit complexity is presented as Theorem B.1. Importantly, the upper bound on the number of parameters in Theorem 4.2 does not come at the cost of implausible bit complexity. In fact, Remark B.2 shows that the constructions in Theorem 4.2(i) and 4.2(ii) can be implemented with bit complexities that match the necessary bit complexity required for networks with the stated parameter counts. The extension of Theorem 4.2 to the ℓ p -norm setting is given in Theorem C.11.

In contrast to prior results, Theorems 4.2(i) and 4.2(ii) provide the first upper bounds for robust memorization that are sublinear in N . Notably, our construction reveals a continuous interpolationdriven by the robustness ratio ρ -from the classical memorization complexity of Θ( √ N ) to the existing upper bound of ˜ O ( N ) in Theorem 4.2(ii), and further from ˜ O ( N ) to ˜ O ( Nd 2 ) as shown in Theorem 4.2(iii). This demonstrates how the sufficient parameter complexity increases gradually with ρ , capturing the full spectrum of the robustness ratio.

Tight Bounds for Robust Memorization with Small ρ . Theorem 4.2(i) establishes a tight upper bound ˜ O ( √ N ) on the number of parameters required for robust memorization when the robustness ratio satisfies ρ &lt; 1 5 N √ d . Since VC-dimension theory [Goldberg and Jerrum, 1995] implies that any network exactly memorizing given N arbitrary samples must use at least Ω( √ N ) parameters, our construction is optimal up to logarithmic factors. This shows that, for sufficiently small ρ , robust memorization requires the same parameter complexity ˜ Θ( √ N ) as classical (non-robust) memorization.

Perfect Robust Memorization with Threshold Activation Function. Theorem 4.2(ii) builds upon the techniques in Theorem 4.2(i), extending the applicability from small values of ρ to moderate ones. However, the extension requires the allowance of an arbitrarily small robust memorization error. As discussed in Section 5.2 and shown Figure 4, the error arises because ReLU -only networks can represent only continuous functions. Near discontinuous transition regions, they incur small errors-though these can be made arbitrarily small. In contrast, if we are allowed to use discontinuous threshold activation in combination with ReLU network, we can achieve ρ -robust memorizationand therefore zero robust memorization error-even in the moderate regime using ˜ O ( Nd 1 / 4 ρ 1 / 2 ) parameters, the same rate as Theorem 4.2(ii).

Tight Bounds of Width. For small and moderate ρ , our construction shows width ˜ O (1) is sufficient, recovering the logarithmic width sufficiency of Egosi et al. [2025]. For large ρ , our construction shows width of ˜ O ( ρ 2 d ) is sufficient for ρ -robust memorization. A complementary lower bound (Proposition 3.2) requires width at least ρ 2 min { N -1 , d } is also necessary, which matches with our upper bound when N &gt; d . As a result, when the number of data points exceeds the data dimension, our results tightly characterize the required width up to polylogarithmic factors across the entire range ρ ∈ (0 , 1) .

## 5 Key Proof Ideas

In this section, we outline the sketch of proof for some of the results from Sections 3 and 4.

## 5.1 Proof Sketch for Proposition 3.2

We briefly overview the sketch of the proof for Proposition 3.2. For simplicity, we sketch the case N = d +1 , where Proposition 3.2 reduces to showing that the first hidden layer must have width at least ρ 2 d . To this end, we construct the dataset D = { ( e j , 1) } j ∈ [ d ] ∪ { ( 0 , 2) } , assigning label 1 to the standard basis points and label 2 to the origin, as shown in Figure 2a.

̸

Let f be an ρ -robust memorizer of D with the first hidden layer width m , and let W ∈ R m × d denote the weight matrix of the first hidden layer. Since ϵ D = 1 / 2 , the robustness radius is µ = ρϵ D = ρ/ 2 . For any j ∈ [ d ] , take any x ∈ B 2 ( e j , µ ) and x ′ ∈ B 2 ( 0 , µ ) . Then, f ( x ) = 1 and f ( x ′ ) = 2 must hold, implying Wx = Wx ′ . Therefore, x -x ′ should not lie in the null space of W . All such possible differences x -x ′ form a ball of radius 2 µ around each standard basis point, illustrated as the gray ball in Figure 2b. Thus, the distance between each standard basis point and the null space of W must be at least 2 µ ; otherwise, some gray balls intersect with the null space.

The null space of W is a d -m dimensional space, assuming that W has full row rank. (The full proof generalizes even without this assumption.) By Lemma A.1, the distance between the set of standard basis points and any subspace of dimension d -m is at most √ m/d . Therefore, we have ρ = 2 µ ≤ dist 2 ( { e j } j ∈ [ d ] , Null( W ) ) ≤ √ m/d and thus the first hidden layer width satisfies m ≥ ρ 2 d .

Figure 2: In (a), blue balls have label 1; the red ball has label 2. (b) illustrates the distance between Null( W ) ⊂ R 3 and the standard basis for W = [1 1 -1] with the first hidden layer width 1.

<!-- image -->

## 5.2 Proof Sketch for Theorem 4.2

We now highlight the key construction techniques used to prove Theorem 4.2.

## Separation-Preserving Dimensionality Reduction.

All three results in Theorem 4.2 leverage a strengthened version of the Johnson-Lindenstrauss (JL) lemma (Lemma B.18) to project data from a high-dimensional space R d (left in Figure 3) to a lower-dimensional space R m (right), while preserving pairwise distances up to a multiplicative factor. Specifically, any pair of points that are 2 ϵ D -separated in R d can remain at least 4 5 √ m d ϵ D -separated after the projection. Meanwhile, each robustness ball of radius µ is preserved under the

Figure 3: Separation-Preserving Projection

<!-- image -->

projection because our strengthened JL lemma uses randomized orthonormal projections [Matousek, 2013]. Since the geometry is preserved-specifically, the separation remains at least 4 5 √ m d times its original value and the robustness radius is unchanged under projection-we can ρ -robustly memorize data points in R d by projecting them to R m and memorizing the projected points, provided that projected robustness balls do not overlap, i.e., as long as ρ ≤ 2 5 √ m d .

In Theorems 4.2(i) and 4.2(ii), we project to R m with m = O (log N ) in the first hidden layer. The remaining layers have width O ( m ) , so the network width is O ( m ) = O (log N ) , i.e., constant up to polylogarithmic factors. This logarithmic projection is valid only for ρ = O (1 / √ d ) : projected ρ -balls remain disjoint as long as ρ ≤ 2 5 √ m/d = ˜ O (1 / √ d ) . If ρ exceeds this scale, the projected balls overlap. For the largerρ regime, Theorem 4.2(iii) increases the projection dimension. As long as ρ ≤ 2 5 √ m/d , the projected robustness balls remain disjoint; accordingly, taking m ∝ ρ 2 d maintains disjointness. Consequently, the width is proportional to ρ 2 d , and the parameter count is proportional to ρ 4 d 2 .

The idea of separation-preserving dimension reduction and deriving conditions under which robustness balls remain disjoint after projection is concurrently proposed by Egosi et al. [2025]. However, their approach to ensuring the separability of robustness balls is substantially different from ours. Since the classical JL lemma does not inherently guarantee the preservation of ball separability, the authors do not rely on the JL lemma directly. Instead, they establish a probabilistic analogue through a technically involved analysis that bounds the probability that a random projection satisfies the required separation property. In contrast, we employ a strengthened version of the JL lemma and give a straightforward proof that there exists a projection preserving separability; see Appendix B.5.

Mapping to Lattices from Grid. For Theorem 4.2(i) and 4.2(ii), we utilize the ˜ O ( √ N ) -parameter memorization devised by Vardi et al. [2021]. In order to adopt the technique, it is necessary to assign a scalar value in R to each data point. This is because the construction memorizes the data after projecting them onto R . Furthermore, this scalar assignment must meaningfully reflect the spatial structure of the data-preserving relative distances and neighborhood relationships of robustness ball.

We achieve this using grid-based lattice mapping. Specifically, we first reduce the dimension to m = O (log N ) . Then we partition R m into a regular grid, and assign an integer index to each grid cell. Through this grid indexing, we map each unit cube ∏ j ∈ [ m ] [ z j , z j + 1) to an index z 1 R m -1 + z 2 R m -2 + · · · + z m for each z = ( z 1 , · · · , z m ) ∈ Z m and some sufficiently large integer R . Finally, we associate each index with the label of the projected robustness ball contained in that cell. The network then memorizes the mapping from each grid index to its corresponding label.

Under the condition on ρ in Theorem 4.2(i), after an appropriate translation of the projected data, every projected robustness ball can be contained in a single grid cell in a way that no cell contains balls of two different labels; see Figure 4a. Hence, the label is constant on each cell that contains a ball, and all points in the ball can be associated with the cell's grid index.

What remains is implementability with ReLU networks. The grid-indexing map is discontinuous, while ReLU networks are continuous and can only approximate it. Consequently, approximation errors can occur only in thin neighborhoods of cell boundaries (the purple bands in Figure 4a).

Theorem 4.2(i) guarantees a translation that places every (projected) robustness ball strictly inside a cell and sufficiently far from all cell boundaries so that the ReLU -based indexing is accurate on the entire ball. Hence, each ball is disjoint from the purple error-tolerant regions, every point in the ball is mapped to the same grid index, and this yields ρ -robust memorization using only ˜ O ( √ N ) parameters.

However, in Theorem 4.2(ii), we consider larger ρ , where projected robustness balls can overlap more than one grid cell and may intersect the error-tolerant regions where the ReLU -based indexing is inaccurate. As ρ grows, the number of such balls increases. To cope with this regime, we use a sequential memorization strategy. We robustly memorize only the subset whose robustness balls are disjoint from the error-tolerant regions. The remaining balls may intersect those regions, but any resulting error is confined to those error-tolerant regions and can be made arbitrarily small by narrowing the error-tolerant regions.

In particular, we partition the N points into multiple groups of approximately equal size and, at each stage, we robustly memorize one group, which we call the active group of this stage and we call the remaining groups of data points as inactive groups. We apply a translation so that the robustness balls of the active group lie strictly inside grid cells and away from the error-tolerant regions, while inactive balls may cross cell boundaries, provided they do not interfere with the cells occupied by the active group of this stage; see Figure 4b. The grid indexing is then implemented by a ReLU approximator whose error-tolerant regions are chosen sufficiently thin-by increasing the slope as in Lemma B.16-so that indexing is exact on the active balls. Any error for the inactive balls is confined to those thin error-tolerant regions. By Lemma B.11, the portion of a robustness ball covered by the error region scales with the region's width, and this width decreases as the ReLU slope grows; hence, the error can be driven arbitrarily small. The active group is robustly memorized using the construction of Theorem 4.2(i), and inactive balls do not interfere with the labels assigned in this stage. Iterating the stages and composing the resulting subnetworks yields memorization of all N points with arbitrarily small error.

<!-- image -->

<!-- image -->

- (a) The setting for Theorem 4.2(i), where each robust ball is entirely contained within a single grid cell, and no two balls with different labels occupy the same cell. This guarantees well-defined indexing without ambiguity.

1

- (b) The relaxed setting in Theorem 4.2(ii) allows some balls to extend across adjacent grid cell boundaries, as long as they do not interfere with the specific cells being memorized at that step.

Figure 4: Grid-based Lattice Mapping.

## 6 Conclusion

We present a tighter characterization of the parameter complexity necessary and sufficient for robust memorization across the full range of robustness ratio ρ ∈ (0 , 1) . Our results establish matching upper and lower bounds for small ρ , and show that robustness demands significantly more parameters than classical memorization as ρ grows. These findings highlight how robustness fundamentally increases memorization difficulty under adversarial attacks.

We establish tight complexity bounds in the regime where ρ &lt; 1 5 N √ d . However, in the remaining cases, a gap between the upper and lower bounds persists. A precise characterization of the parameter complexity for some ρ remains open and is essential for a complete understanding of the trade-off between robustness and network complexity.

## Acknowledgement

This work was supported by three Institute of Information &amp; communications Technology Planning &amp;Evaluation (IITP) grants (No. RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST); No. RS-2022-II220184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics; No. RS-2024-00457882, National AI Research Lab Project) funded by the Korean government (MSIT) and the InnoCORE program of the Ministry of Science and ICT (No. N10250156).

## References

- Peter L Bartlett, Nick Harvey, Christopher Liaw, and Abbas Mehrabian. Nearly-tight vc-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research , 20(63):1-17, 2019.
- Alexander Bastounis, Anders C Hansen, and Verner Vlaˇ ci´ c. The mathematics of adversarial attacks in ai - why deep learning is unstable despite the existence of stable neural networks, 2025. URL https://arxiv.org/abs/2109.06098 .
- Eric B Baum. On the capabilities of multilayer perceptrons. Journal of complexity , 4(3):193-215, 1988.
- Sébastien Bubeck, Ronen Eldan, Yin Tat Lee, and Dan Mikulincer. Network size and weights size for memorization with two-layers neural networks, 2020. URL https://arxiv.org/abs/2006. 02855 .
- AYaChervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. In Measures of complexity: festschrift for alexey chervonenkis , pages 11-30. Springer, 2015.
- Gavin Weiguang Ding, Kry Yik Chau Lui, Xiaomeng Jin, Luyu Wang, and Ruitong Huang. On the sensitivity of adversarial robustness to input data distributions, 2019. URL https://arxiv.org/ abs/1902.08336 .
- Amitsour Egosi, Gilad Yehudai, and Ohad Shamir. Logarithmic width suffices for robust memorization, 2025. URL https://arxiv.org/abs/2502.11162 .
- Ruiqi Gao, Tianle Cai, Haochuan Li, Cho-Jui Hsieh, Liwei Wang, and Jason D Lee. Convergence of adversarial training in overparametrized neural networks. Advances in Neural Information Processing Systems , 32, 2019.
- Paul W Goldberg and Mark R Jerrum. Bounding the vapnik-chervonenkis dimension of concept classes parameterized by real numbers. Machine Learning , 18(2-3):131-148, 1995.
- Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples, 2015. URL https://arxiv.org/abs/1412.6572 .
- Sven Gowal, Chongli Qin, Jonathan Uesato, Timothy Mann, and Pushmeet Kohli. Uncovering the limits of adversarial training against norm-bounded adversarial examples, 2021. URL https: //arxiv.org/abs/2010.03593 .
- Binghui Li, Jikai Jin, Han Zhong, John Hopcroft, and Liwei Wang. Why robust generalization in deep learning is difficult: Perspective of expressive power. Advances in Neural Information Processing Systems , 35:4370-4384, 2022.
- Jiri Matousek. Lectures on discrete geometry , volume 212. Springer Science &amp; Business Media, 2013.
- Sejun Park, Jaeho Lee, Chulhee Yun, and Jinwoo Shin. Provable memorization via deep neural networks using sub-linear parameters, 2021. URL https://arxiv.org/abs/2010.13363 .
- Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks, 2014. URL https://arxiv.org/ abs/1312.6199 .

- Matus Telgarsky. Benefits of depth in neural networks. CoRR , abs/1602.04485, 2016. URL http://arxiv.org/abs/1602.04485 .
- Gal Vardi, Gilad Yehudai, and Ohad Shamir. On the optimal memorization power of ReLU neural networks, 2021. URL https://arxiv.org/abs/2110.03187 .
- Lijia Yu, Xiao-Shan Gao, and Lijun Zhang. OPTIMAL ROBUST MEMORIZATION WITH RELU NEURAL NETWORKS. In The Twelfth International Conference on Learning Representations , 2024. URL https://openreview.net/forum?id=47hDbAMLbc .
- Chulhee Yun, Suvrit Sra, and Ali Jadbabaie. Small relu networks are powerful memorizers: a tight analysis of memorization capacity, 2019. URL https://arxiv.org/abs/1810.07770 .
- Chongzhi Zhang, Aishan Liu, Xianglong Liu, Yitao Xu, Hang Yu, Yuqing Ma, and Tianlin Li. Interpreting and improving adversarial robustness of deep neural networks with neuron sensitivity. IEEE Transactions on Image Processing , 30:1291-1304, 2021. ISSN 1941-0042. doi: 10.1109/tip. 2020.3042083. URL http://dx.doi.org/10.1109/TIP.2020.3042083 .

## Contents

| 1 Introduction   | 1 Introduction                                            | 1 Introduction                                                                                                                                                      | 1   |
|------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
|                  | 1.1                                                       | What is Known So Far? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   | 2   |
|                  | 1.2                                                       | Summary of Contribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   | 2   |
|                  | 2 Preliminaries                                           | 2 Preliminaries                                                                                                                                                     | 3   |
|                  | 2.1                                                       | Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                | 3   |
|                  | 2.2                                                       | Dataset and Robust Memorization . . . . . . . . . . . . . . . . . . . . . . . .                                                                                     | 4   |
|                  | 2.3                                                       | ReLU Neural Network . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                     | 4   |
|                  | 2.4                                                       | Why Only ρ = µ/ϵ D Matters . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    | 5   |
|                  | 3                                                         | Number of Parameters for Robust Memorization                                                                                                                        |     |
|                  | Necessary                                                 | Necessary                                                                                                                                                           | 5   |
|                  | 4 Sufficient Number of Parameters for Robust Memorization | 4 Sufficient Number of Parameters for Robust Memorization                                                                                                           | 7   |
|                  | 5 Key Proof Ideas                                         | 5 Key Proof Ideas                                                                                                                                                   | 8   |
|                  | 5.1                                                       | Proof Sketch for Proposition 3.2 . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                  | 8   |
|                  | 5.2                                                       | . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                                                 |     |
|                  |                                                           | Proof Sketch for Theorem 4.2                                                                                                                                        | 9   |
|                  | Conclusion                                                | Conclusion                                                                                                                                                          | 10  |
|                  | A Proofs for Section 3                                    | A Proofs for Section 3                                                                                                                                              | 15  |
|                  | A.1                                                       | Explicit Proof of Theorem 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   | 15  |
|                  | A.2                                                       | Necessary Condition on Width for Robust Memorization . . . . . . . . . . . .                                                                                        | 16  |
|                  | A.3                                                       | Necessary Condition on Parameters for Robust Memorization . . . . . . . . . .                                                                                       | 17  |
|                  | A.4                                                       | Lemmas for Appendix A . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                     | 20  |
|                  | Proofs for Section 4                                      | Proofs for Section 4                                                                                                                                                | 22  |
|                  | B.1                                                       | Sufficient Condition for Robust Memorization with Small Robustness Radius .                                                                                         | 22  |
|                  | B.2                                                       | Sufficient Condition for Near-Perfect Robust Memorization with Moderate Robust- ness Radius . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 26  |
|                  |                                                           | B.2.1 Memorization of Integers with Sublinear Parameters in N . . . . . . .                                                                                         | 30  |
|                  |                                                           | B.2.2 Precise Control of Robust Memorization Error . . . . . . . . . . . . .                                                                                        | 33  |
|                  | B.3                                                       | Sufficient Condition for Robust Memorization with Large Robustness Radius .                                                                                         | 39  |
|                  | B.4                                                       | Lemmas for Lattice Mapping . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    | 44  |
|                  | B.5                                                       | Dimension Reduction via Careful Analysis of the Johnson-Lindenstrauss Lemma                                                                                         | 48  |
|                  | B.6                                                       | Lemmas for Bit Complexity . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                     | 52  |
|                  | Extensions to ℓ p -norm                                   | Extensions to ℓ p -norm                                                                                                                                             | 55  |
|                  | C.1                                                       | Extension of Necessity Condition to ℓ p -norm . . . . . . . . . . . . . . . . . .                                                                                   | 55  |
|                  |                                                           | C.1.1 Lemmas for Appendix C.1 . . . . . . . . . . . . . . . . . . . . . . . .                                                                                       | 59  |
|                  | C.2 Extension of Sufficiency Condition to ℓ p -norm       | . . . . . . . . . . . . . . . . .                                                                                                                                   | 60  |

| D   | Comparison to Existing Bounds   | Comparison to Existing Bounds                                    |   64 |
|-----|---------------------------------|------------------------------------------------------------------|------|
|     | D.1                             | Summary of Parameter Complexity across ℓ p -norms . . . . . . .  |   64 |
|     | D.2                             | Parameter Complexity of the Construction by Yu et al. [2024] . . |   64 |
|     | D.3                             | Parameter Complexity of the Construction by Egosi et al. [2025]  |   65 |
| E   | Nonzero Parameter Counts        | Nonzero Parameter Counts                                         |   66 |
|     | E.1                             | Nonzero Parameter Counts: An illustration. . . . . . . . . . . . |   66 |
|     | E.2                             | Nonzero Parameter Counts: Lower Bounds . . . . . . . . . . .     |   66 |
|     | E.3                             | Nonzero Parameter Counts: Upper Bounds . . . . . . . . . . .     |   67 |
|     | E.4                             | Lemmas for Nonzero Parameter Count . . . . . . . . . . . . . .   |   72 |

## A Proofs for Section 3

## A.1 Explicit Proof of Theorem 3.1

Theorem 3.1. Let ρ ∈ (0 , 1) . Suppose for any D ∈ D d,N, 2 , there exists a neural network f ∈ F d,P that can ρ -robustly memorize D . Then, the number of parameters P must satisfy

<!-- formula-not-decoded -->

Proof. From Proposition 3.2, we obtain D ∈ D d,N, 2 such that any f : R d → R that ρ -robustly memorizes D must have the first hidden layer width at least ρ 2 min { N -1 , d } . By the assumption of Theorem 3.1, there exists f ∈ F d,P that ρ -robustly memorizes D with the first hidden layer width m ≥ ρ 2 min { N -1 , d } . With the trivial lower bound that m ≥ 1 , we have

<!-- formula-not-decoded -->

Since we count all parameters according to Equation (3), the number of parameters in the first layer is ( d +1) m . Therefore,

<!-- formula-not-decoded -->

In addition, for ρ ∈ ( 0 , √ 1 -1 d ] , using Proposition 3.3 gives the lower bound of parameters

<!-- formula-not-decoded -->

For ρ ∈ ( 0 , √ 1 -1 d ] , we have 1 √ 1 -ρ 2 ≤ √ d so that the following relation holds:

<!-- formula-not-decoded -->

For ρ ∈ (√ 1 -1 d , 1 ) , the lower bound P = Ω( √ Nd ) obtained by the case ρ = √ 1 -1 d also can be applied. In this case, 1 √ 1 -ρ 2 &gt; √ d so that the following relation holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

serves as the lower bound on the number of parameters.

By combining the bounds from Proposition 3.2 and Proposition 3.3, we conclude:

<!-- formula-not-decoded -->

Hence, in both ρ regimes,

## A.2 Necessary Condition on Width for Robust Memorization

Proposition 3.2. There exists D ∈ D d,N, 2 such that, for any ρ ∈ (0 , 1) , any neural network f : R d → R that ρ -robustly memorizes D must have the first hidden layer width at least ρ 2 min { N -1 , d } .

Proof. To prove Proposition 3.2, we consider two cases based on the relationship between N -1 and d . In the first case, where N -1 ≤ d , establishing the proposition requires that the first hidden layer has width at least ρ 2 ( N -1) . In the second case, where N -1 &gt; d , the required width is at least ρ 2 d . For each case, we construct a dataset D ∈ D d,N, 2 such that any network that ρ -robustly memorizes D must have a first hidden layer of width no smaller than the corresponding bound.

Case I : N -1 ≤ d . Let D = { ( e j , 2) } j ∈ [ N -1] ∪ { ( 0 , 1) } . Then, D has separation constant ϵ D = 1 / 2 . Let f be a neural network that ρ -robust memorizes D , and denote the width of its first hidden layer as m . Denote by W ∈ R m × d the weight matrix of the first hidden layer of f . Assume for contradiction that m&lt;ρ 2 ( N -1) .

Let µ = ρϵ D denote the robustness radius. Then, the network f must distinguish every point in B 2 ( e j , µ ) from every point in B 2 ( 0 , µ ) , for all j ∈ [ N -1] . Therefore, for any x ∈ B 2 ( e j , µ ) and x ′ ∈ B 2 ( 0 , µ ) , we must have

̸

<!-- formula-not-decoded -->

or equivalently, x -x ′ / ∈ Null( W ) , where Null( · ) denotes the null space of a given matrix. Note that

<!-- formula-not-decoded -->

Hence, it is necessary that B 2 ( e j , 2 µ ) ∩ Null( W ) = ∅ for all j ∈ [ N -1] , or equivalently,

<!-- formula-not-decoded -->

Since dim(Col( W ⊤ )) ≤ m , where Col( · ) denotes the column space of the given matrix, it follows that dim(Null( W )) ≥ d -m . Using Lemma A.2, we can upper bound the distance between the set { e j } j ∈ [ N -1] ⊆ R d and any subspace of dimension d -m .

Let Z ⊆ Null( W ) be a subspace such that dim( Z ) = d -m , and apply Lemma A.2 with substitutions d = d , t = N -1 , k = d -m and Z = Z . The conditions of lemma, namely t ≤ d and k ≥ d -t , are satisfied since N -1 ≤ d and m&lt;ρ 2 ( N -1) ≤ N -1 . Therefore, we obtain the bound

<!-- formula-not-decoded -->

By combining the above inequality with Equation (5), we obtain

<!-- formula-not-decoded -->

where (a) follows from that Z ⊆ Null( W ) . Since ϵ D = 1 / 2 , we have 2 µ = 2 ρϵ D = ρ , so Equation (6) becomes

<!-- formula-not-decoded -->

This implies that m ≥ ρ 2 ( N -1) , contradicting the assumption m &lt; ρ 2 ( N -1) . Therefore, the width requirement m ≥ ρ 2 ( N -1) is necessary. This concludes the statement for the case N -1 ≤ d .

Case II : N -1 &gt; d . We construct the first d +1 data points in the same manner as in Case I, using the construction for N = d +1 . For the remaining N -d -1 data points, we set them sufficiently distant from the first d +1 data points to ensure that the separation constant remains ϵ D = 1 / 2 .

In particular, we set x d +2 = 2 e 1 , x d +3 = 3 e 1 , · · · , x N = ( N -d ) e 1 and assign y d +2 = y d +3 = · · · = y N = 2 . Compared to the case N = d +1 , this construction preserves ϵ D while adding more data points to memorize. Since the first d +1 data points are constructed as in the case N = d +1 , the same lower bound applies. Specifically, by the result of Case I, any network that ρ -robustly

memorizes this dataset must have a first hidden layer of width at least ρ 2 (( d +1) -1) = ρ 2 d . This concludes the argument for the case N -1 &gt; d .

Combining the results from the two cases N -1 ≤ d and N -1 &gt; d completes the proof of the proposition.

## A.3 Necessary Condition on Parameters for Robust Memorization

For sufficiently large ρ , Gao et al. [2019] and Li et al. [2022] prove that, for any D ∈ D d,N,C , if there exists f ∈ F d,P that ρ -robustly memorizes D , the number of parameters P should satisfy P = Ω( √ Nd ) . However, the authors do not characterize the range of ρ over which this lower bound remains valid.

Motivated from Gao et al. [2019] and Li et al. [2022], we establish a lower bound that depends on ρ in the regime ρ ≤ √ 1 -1 /d , which becomes √ Nd when ρ = √ 1 -1 /d . This implies that the existing lower bound √ Nd remains valid for ρ ∈ [ √ 1 -1 /d, 1) . As a result, we obtain a lower bound that holds continuously from ρ ≈ 0 up to ρ ≈ 1 , and thus interpolates between the lower bound √ N for memorization to the lower bound √ Nd for robust memorization.

Proposition 3.3. Let ρ ∈ ( 0 , √ 1 -1 d ] . Suppose for any D ∈ D d,N, 2 , there exists f ∈ F d,P that ρ -robustly memorizes D . Then, the number of parameters P must satisfy

<!-- formula-not-decoded -->

Proof. To prove the statement, we show that for any D ∈ D d,N, 2 , if there exists a network f ∈ F d,P that ρ -robustly memorizes D , then

<!-- formula-not-decoded -->

If the above bound holds, then as VC -dim( F d,P ) = O ( P 2 ) , it follows that P = Ω( √ N/ (1 -ρ 2 )) .

Let k := ⌊ 1 1 -ρ 2 ⌋ . To establish the desired VC-dimension lower bound, it suffices to show that

<!-- formula-not-decoded -->

This implies Equation (7), as desired. To this end, it suffices to construct k · ⌊ N 2 ⌋ points in R d that can be shattered by F d,P . These points are organized as an union of ⌊ N 2 ⌋ groups, each group consisting of k points.

Step 1 (Constructing Ω( N/ (1 -ρ 2 )) points X to be shattered by F d,P ).

We begin by constructing the first group. Since ρ ∈ ( 0 , √ d -1 d ] , we have k = ⌊ 1 1 -ρ 2 ⌋ ∈ (1 , d ] .

Define the first group X 1 := { e j } k j =1 ⊆ R d , consisting of the first k standard basis vectors in R d . The remaining ⌊ N 2 ⌋ -1 groups are constructed by translating X 1 . For each l = 1 , · · · ⌊ N 2 ⌋ , define

<!-- formula-not-decoded -->

where c l := 2 d 2 ( l -1) · e 1 ensures that the groups are sufficiently distant from one another. Note that c 1 = 0 , so that X 1 is consistent with the definition above. Now, define X := ∪ l ∈ [ ⌊ N/ 2 ⌋ ] X l as the union of all groups, comprising k ×⌊ N 2 ⌋ points in total.

Step 2 (Showing F d,P shatter X ).

2 We follow the definition of VC-dimension by Bartlett et al. [2019]. Note that the VC-dimension of a real-valued function class is defined as the VC-dimension of sign ( F ) := { sign ◦ f | f ∈ F} . Since we consider the label set [2] = { 1 , 2 } for robust memorization while the VC-dimension requires the label set { +1 , -1 } , we take an additional step of an affine transformation in the last step of the proof.

Figure 5: Reduction of Shattering to Robust Memorization. The cross marks refer to the points to be shattered, and the circular dots refer to the points for robust memorization. The centers of robustness balls change with respect to the labels of the points to be shattered.

<!-- image -->

We claim that for any D ∈ D d,N, 2 , if there exists a network f ∈ F d,P that ρ -robustly memorizes D , then the point set X is shattered by F d,P . To prove the claim, consider an arbitrary labeling Y = { y l,j } l ∈ [ ⌊ N/ 2 ⌋ ] ,j ∈ [ k ] of the points in X , where each label y l,j ∈ {± 1 } corresponds to the point x l,j := c l + e j ∈ X .

Given the labeling Y , we construct D ∈ D d,N, 2 with labels in { 1 , 2 } such that any function f ∈ F d,P that ρ -robustly memorizes D can be affinely transformed to f ′ = 2 f -3 ∈ F d,P , which satisfies f ′ ( x l,j ) = y l,j ∈ {± 1 } for all x l,j ∈ X . In other words, f ′ exactly memorizes the given labeling Y over X , thereby showing that X is shatterd by F d,P . The affine transformation is necessary to match the { 1 , 2 } -valued outputs of f with the {± 1 } labeling required for the shattering argument.

For each l ∈ [ ⌊ N/ 2 ⌋ ] , define the index sets

<!-- formula-not-decoded -->

which partition the group-wise labeling { y l,j } j ∈ [ k ] ⊂ Y into positive and negative indices. We then define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let y 2 l -1 = 2 , y 2 l = 1 , and define the dataset D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N, 2 . Figure 5 illustrates the first group l = 1 with k = 3 where the labels gives the index sets J + 1 = { 1 , 3 } and J -1 = { 2 } . The blue and red dots denote the points x 1 and x 2 , respectively.

To analyze the separation constant ϵ D , we consider the distance between pairs of points with different labels. Specifically, for each l , the two points x 2 l -1 and x 2 l have opposite labels by construction. Consider their distance:

<!-- formula-not-decoded -->

̸

where (a) holds since J + l ∩ J -l = ∅ and J + l ∪ J -l = [ k ] . Now, for l = l ′ , consider the distance between x 2 l -1 and x 2 l ′ , which again correspond to different labels. We have:

<!-- formula-not-decoded -->

In particular, we can take

<!-- formula-not-decoded -->

where (a) follows from the triangle inequality, (b) uses dist 2 ( c l , x 2 l -1 ) = dist 2 ( c l ′ , x 2 l ′ ) = √ k , (c) and (e) use k ≤ d , and (d) holds for all d ≥ 2 . Thus, we conclude that ϵ D ≥ √ k .

Let f ∈ F d,P be a function that ρ -robustly memorizes D . We begin by deriving a lower bound on the robustness radius µ in order to verify that f ′ = 2 f -3 correctly memorizes the given labeling Y over X . Define ϕ ( t ) := √ t -1 t . The function ϕ is strictly increasing for t ≥ 1 , and maps [1 , ∞ ) onto [0 , 1) . Hence, it admits an inverse ϕ -1 : [0 , 1) → [1 , ∞ ) , defined as ϕ -1 ( ρ ) = 1 1 -ρ 2 . Therefore, we have

<!-- formula-not-decoded -->

Given ϵ D ≥ √ k and ρ ≥ √ k -1 k , it follows that µ = ρϵ D ≥ √ k -1 . Thus, any function f that ρ -robustly memorizes D must also memorize all points within an ℓ 2 -ball of radius √ k -1 centered at each point in D .

Next, for x l,j ∈ X with positive label y l,j = +1 , we have

̸

<!-- formula-not-decoded -->

Now consider a sequence { z n } n ∈ N such that z n → x l,j as n →∞ and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which satisfies such properties. Then, z n ∈ B ( x 2 l -1 , µ ) for all n , and by robustness of f , f ( z n ) = f ( x 2 l -1 ) = 2 . By continuity of f , we have

<!-- formula-not-decoded -->

Similarly, for x l,j ∈ X with negative label y l,j = -1 , we have ∥ x l,j -x 2 l ∥ 2 = √ k -1 , so that f ( x l,j ) = 1 .

Since we can adjust the weight and the bias of the last hidden layer, F d,P is closed under affine transformation; that is, af + b ∈ F d,P whenever f ∈ F d,P . In particular, f ′ := 2 f -3 ∈ F d,P . This f ′ satisfies f ′ ( x l,j ) = 2 f ( x l,j ) -3 = 2 · 2 -3 = +1 whenever y l,j = +1 and f ′ ( x l,j ) = 2 f ( x l,j ) -3 = 2 · 1 -3 = -1 whenever y l,j = -1 . Thus, sign ◦ f ′ perfectly classifies X according to the given labeling Y . Since such f ′ ∈ F d,P exists for an arbitrary labeling Y , it follows that F d,P shatters X , completing the proof of the theorem.

## A.4 Lemmas for Appendix A

The following lemma upper bounds the ℓ 2 -distance between the standard basis and any subspace of a given dimension.

Lemma A.1. Let { e j } j ∈ [ d ] ⊆ R d denote the standard basis of R d . Then, for any k -dimensional subspace Z ⊆ R d ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let { u 1 , u 2 , · · · , u k } ⊆ R d be an orthonormal basis of Z , and denote each u l = ( u l 1 , u l 2 , · · · , u ld ) ⊤ . Let U ∈ R d × k be the matrix whose columns are u 1 , · · · , u k , so that

<!-- formula-not-decoded -->

Then the projection matrix P onto Z is given by

<!-- formula-not-decoded -->

Now, for each standard basis vector e j , the squared norm of its projection onto Z is:

<!-- formula-not-decoded -->

where the last equality holds as u l are orthonormal. Moreover,

<!-- formula-not-decoded -->

This proves the first statement of the lemma. To prove the second statement, observe that for any v ∈ R d , we can write

<!-- formula-not-decoded -->

so that ∥ v ∥ 2 2 = ∥ Proj Z ( v ) ∥ 2 2 + ∥ Proj Z ⊥ ( v ) ∥ 2 2 . Noticing dist 2 ( v , Z ) = ∥ Proj Z ⊥ ( v ) ∥ 2 together with the first statement, we have

<!-- formula-not-decoded -->

which concludes the second statement.

In particular,

The next lemma generalizes Lemma A.1 to the case where we consider only the distance to a subset of the standard basis, instead of the whole standard basis.

Lemma A.2. Let 1 ≤ t ≤ d , and let { e j } j ∈ [ t ] ⊆ R d denote the first t standard basis vectors. Then, for any k -dimensional subspace Z ⊆ R d with k ≥ d -t , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By taking dimensions,

<!-- formula-not-decoded -->

Now, consider the restriction of R d to R t by the linear map

<!-- formula-not-decoded -->

Since Col( Q ⊤ ) = span { e 1 , . . . , e t } , the projection satisfies:

<!-- formula-not-decoded -->

By applying Lemma A.1 with the restricted space R t , we obtain

<!-- formula-not-decoded -->

Since Z ⊇ Z ∩ Col( Q ⊤ ) , it follows that

<!-- formula-not-decoded -->

This proves the first statement. To prove the second statement, for any v ∈ R d , decompose v as v = Proj Z ( v ) + Proj Z ⊥ ( v ) , and note that ∥ v ∥ 2 2 = ∥ Proj Z ( v ) ∥ 2 2 + ∥ Proj Z ⊥ ( v ) ∥ 2 2 . Using dist 2 ( v , Z ) = ∥ Proj Z ⊥ ( v ) ∥ 2 together with the first statement, we have

<!-- formula-not-decoded -->

which concludes the second statement.

In particular,

## B Proofs for Section 4

In this section, we prove an extended version of Theorem 4.2, which additionally states the explicit bounds on depth, width, and bit complexity, in addition to the sufficient number of parameters. We present the ℓ p -norm version of Theorem 4.2 in Theorem C.11.

Theorem B.1. For any dataset D ∈ D d,N,C and η ∈ (0 , 1) , the following statements hold:

- (i) If ρ ∈ ( 0 , 1 5 N √ d ] , there exists f with ˜ O ( √ N ) parameters, depth ˜ O ( √ N ) , width ˜ O (1) and bit complexity ˜ O ( √ N ) that ρ -robustly memorizes D .
- (ii) If ρ ∈ ( 1 5 N √ d , 1 5 √ d ] , there exists f with ˜ O ( Nd 1 4 ρ 1 2 ) parameters, depth ˜ O ( Nd 1 4 ρ 1 2 ) , width ˜ O (1) and bit complexity ˜ O ( 1 / d 1 4 ρ 1 2 ) that ρ -robustly memorizes D with error at most η .
- (iii) If ρ ∈ ( 1 5 √ d , 1 ) , there exists f with ˜ O ( Nd 2 ρ 4 ) parameters, depth ˜ O ( N ) , width ˜ O ( ρ 2 d ) and bit complexity ˜ O ( N ) that ρ -robustly memorizes D .

Here, the bit complexity is defined as a bit needed per parameter under a fixed point precision. To prove Theorem B.1, we decompose it into three theorems (Theorems B.3, B.5 and B.14), each corresponding to one of the cases in the statement. Their proofs are provided in Appendices B.1 to B.3, respectively.

Remark B.2 (Tight Bit Complexity) . The bit complexities in Theorems B.1(i) and B.1(ii) are essentially tight within our construction framework. Vardi et al. [2021] provide a lower bound on bit complexity using upper and lower bounds on VC-dimension. In particular, for a network with P nonzero parameters (refer Appendix E for detailed analysis on nonzero parameters) and bit complexity B , the VC-dimension is upper bounded as

<!-- formula-not-decoded -->

Since VC-dimension is lower bounded by N by the robust memorization, combining these two bounds suggests the necessary bit complexity required under our constructions in Theorem 4.2. For simplicity, assume the case where the omitted d in the upper bound is not dominant. In Theorem E.2, we show that under our constructions, the number of nonzero parameters satisfies P = ˜ O ( √ N ) for small ρ and P = ˜ O ( Nd 1 / 4 ρ 1 / 2 ) for moderate ρ . Consequently, the bit complexity becomes

<!-- formula-not-decoded -->

respectively, which matches the upper bounds.

## B.1 Sufficient Condition for Robust Memorization with Small Robustness Radius

Theorem B.3. Let ρ ∈ ( 0 , 1 5 N √ d ] . For any dataset D ∈ D d,N,C , there exists f with ˜ O ( √ N ) parameters, depth ˜ O ( √ N ) , width ˜ O (1) and bit complexity ˜ O ( √ N ) that ρ -robustly memorizes D .

Proof. For given ρ and D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N,C , we construct a network f ∈ F d,P that ρ -robustly memorizes D with ˜ O ( √ N ) parameters. The construction proceeds in four stages. In each stage, we define a function implementable by a neural network, such that their composition yields a ρ -robust memorizer for D .

Stage I (Projection onto log -scale Dimension and Scaling via the First Hidden Layer Weight Matrix). By Lemma B.20, we obtain an integer m = ˜ O (log N ) and a 1-Lipschitz linear map ϕ : R d → R m such that the projected dataset D ′ := { ( ϕ ( x i ) , y i ) } i ∈ [ N ] ∈ D m,N,C satisfies the separation bound

<!-- formula-not-decoded -->

We define f proj : R d → R m as f proj ( x ) = 11 9 · √ d ϵ D ϕ ( x ) , which is 11 9 · √ d ϵ D -Lipschitz.

We apply Lemma B.23 with f proj whose depth is 1 , ν = min { 109 11880 √ m, 1 88 µ, 1 360 N , 1 } and ¯ R := max {∥ x ∥ 2 | x ∈ B 2 ( x i , µ ) for some i ∈ [ N ] } to obtain ¯ f proj with the same number of parameters, depth and width and ˜ O (1) bit complexity such that

<!-- formula-not-decoded -->

We set the first hidden layer bias b ∈ R m so that

<!-- formula-not-decoded -->

where the comparison between two vectors is element-wise.

̸

We claim that for D ′′ := { ( σ ( ¯ f proj ( x i ) + b ) , y i ) } i ∈ [ N ] , we have (i) ϵ D ′′ ≥ √ m/ 2 and (ii) for ρ ′′ := 1 4 Nϵ D′′ , if g ( x ) ∈ F m,P can ρ ′′ -robustly memorize D ′′ , then g ◦ σ ◦ ( ¯ f proj ( x ) + b ) can ρ -robustly memorize D . For any i = j with y i = y j , we have

̸

<!-- formula-not-decoded -->

where (a) holds by the construction of b (Equation (10)). For simplicity, we denote

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

where (a) holds from ∥ a + b ∥ 2 2 ≥ ( ∥ a ∥ 2 -∥ b ∥ 2 ) 2 . By the construction of ¯ f (Equation (9)),

<!-- formula-not-decoded -->

so we have

Now we derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) is by the definition of f proj , (b) is by the definition of D ′ and its separation constant, and (c) follows from Equation (8).

Plugging this inequality to Equation (11) gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) holds from ν ≤ 109 11880 √ m , and (b) holds from Equation (12). This proves the first claim ϵ D ′′ ≥ √ m/ 2 . To prove the second claim, let µ := ρϵ D and µ ′′ := ρ ′′ ϵ D ′′ . Then,

<!-- formula-not-decoded -->

where (a) and (j) are by Equation (10), (b) and (g) are by the construction of ¯ f (Equation (9)), (c) is because ν ≤ 1 88 µ , (d) is because f proj is 11 9 · √ d ϵ D -Lipschitz, (e) uses µ = ρϵ D , (f) uses ρ ≤ 1 5 N √ d , (h) is because ν ≤ 1 360 N and (i) is because ρ ′′ ϵ D ′′ = 1 4 Nϵ D′′ ϵ D ′′ = 1 4 N .

Hence, g ( x ) memorizing the robustness ball B 2 ( σ ( ¯ f proj ( x i ) + b ) , ρ ′′ ϵ D ′′ ) on projected space leads to g ◦ σ ◦ ( ¯ f proj ( x ) + b ) memorizing the robustness ball for D . In other words, if g ( x ) ∈ F m,P can ρ ′′ -robustly memorize D ′′ , then g ◦ σ ◦ ( ¯ f proj ( x )+ b ) can ρ -robustly memorize D . With ρ ′′ = 1 4 Nϵ D′′ , Stage II to IV aims to find a ρ ′′ -robust memorizer g of D ′′ .

Stage II (Translation for Distancing from Lattice via the Bias) For simplicity of the notation, let z i := σ ( ¯ f proj ( x i ) + b ) for each i ∈ [ N ] , so that D ′′ = { ( z i , y i ) } i ∈ [ N ] . Recall that ρ ′′ = 1 4 Nϵ D′′ gives the robustness radius is µ ′′ = ρ ′′ ϵ D ′′ = 1 4 N .

By applying Lemma B.15 to z 1 , · · · , z N , we obtain a translation vector b 2 = ( b 21 , · · · , b 2 m ) ∈ R m with bit complexity ⌈ log(6 N ) ⌉ such that

<!-- formula-not-decoded -->

i.e., the translated points { z i -b 2 } i ∈ [ N ] are coordinate-wise far from the integer lattice. Moreover, by additional translation to { z i -b 2 } i ∈ [ N ] (by adding some natural number, coordinate-wise), we can ensure all coordinates are positive while keeping the property Equation (13). Hence, we may assume without loss of generality b 2 also has the property

<!-- formula-not-decoded -->

Let us denote D ′′′ = { ( z ′ i , y i ) } i ∈ [ N ] , where z ′ i := z i -b 2 . Then ϵ D ′′′ = ϵ D ′′ ( ≥ √ m/ 2) . For ρ ′′′ := ρ ′′ = 1 4 Nϵ D′′ , we have the robustness radius µ ′′′ := ρ ′′′ ϵ D ′′′ = ρ ′′ ϵ D ′′ = µ ′′ = 1 4 N . Define f trans as f trans ( z ) := z -b 2 . Then, f trans can be implemented via one hidden layer in a neural network, with O ( m 2 ) parameters.

Upon the two layers constructed from stage I and II, it suffices to construct a network that ρ ′′′ -robustly memorizes D ′′′ since the translation preserves separation and ball containment properties. Note that the robustness balls after stage II are not affected when passing the σ , by Equations (13) and (14).

Stage III (Grid Indexing) From Equation (13), each z ′ i ∈ R m is at least 4 3 µ ′′′ distant away from any lattice hyperplane H z,j := { z ∈ R m | z j = z } with any j ∈ [ m ] and z ∈ Z . Thus, each robustness ball of D ′′′ lies completely within a single integer lattice (or unit grid) of the form ∏ m j =1 [ n j , n j +1) , for some ( n 1 , · · · , n m ) ∈ Z m .

̸

Moreover, as ϵ D ′′′ ≥ √ m/ 2 , for any i = i ′ with y i = y i ′ , we have ∥ z ′ i -z ′ i ′ ∥ 2 ≥ √ m . Since sup {∥ z -z ′ ∥ 2 | z , z ′ ∈ ∏ m j =1 [ n j , n j +1) } = √ m , two such points z ′ i and z ′ i ′ that corresponds to distinct labels cannot lie in the same grid. Since each µ ′′′ -ball lies within a single grid, we conclude that no two µ ′′′ -ball with different labels lie within the same grid.

̸

We define R := ⌈ max i ∈ [ N ] ∥ z ′ i ∥ ∞ (= max i ∈ [ N ] ,j ∈ [ m ] ( z ′ i,j )) ⌉ ∈ N . Our goal in this stage is to construct Flatten mapping defined as

<!-- formula-not-decoded -->

This maps each grid ∏ m j =1 [ n j , n j +1 ) onto the point ∑ m j =1 R j -1 n j .

However, since Flatten is discontinuous due to the use of floor functions, we construct Flatten , which is a continuous approximation that exactly matches Flatten in the region of our interest. By applying Lemma B.16 to γ = 1 4 N and n = ⌈ log 2 R ⌉ , we obtain the network Floor := Floor ⌈ log 2 R ⌉ with O (log 2 R ) parameters such that

<!-- formula-not-decoded -->

Moreover, since we apply γ = 1 / 4 N to Lemma B.16, the lemma guarantees that Floor n can be exactly implemented with O ( n +log N ) = O (log R +log N ) bit complexity. In particular, we can define our network Flatten with O (log R +log N +log R m -1 ) = O (log R +log N +log N log R ) = ˜ O (1) bit complexity as

<!-- formula-not-decoded -->

This implementation is valid-i.e. Flatten( z ) = Flatten( z ) -in the region of interest ( { z ∈ [0 , R ] m | dist 2 ( z j , Z ) ≥ 1 2 N for all j ∈ [ m ] } ) characterized by the margin guaranteed by Equation (13).

As Floor : R → R can be implemented with width 5 and depth O (log 2 R ) network (Lemma B.16), Flatten can be implemented with width 5 m and depth O (log 2 R ) network. Thus, we can construct Flatten with O ( m 2 log 2 R ) = ˜ O ( m 2 ) parameters.

By Equations (13) and (15), we guarantee that each robustness ball lies in the region where the Flatten is properly approximated by Flatten . i.e.

<!-- formula-not-decoded -->

Since Flatten maps each unit grid into a point and each robustness ball of D ′′′ lies on a single unit grid, we conclude

<!-- formula-not-decoded -->

Let m i := Flatten( z ′ i ) . Then each robustness ball around x i is mapped to m i . We have m i ∈ Z ∩ [0 , R m +1 ] for all i ∈ [ N ] , since

<!-- formula-not-decoded -->

where (a) is by ∥ z ′ i ∥ ∞ ≤ R .

Stage IV (Memorization) Finally, it remains to memorize N points { ( m i , y i ) } N i =1 ⊂ Z ≥ 0 × [ C ] . Since multiple robustness balls for D ′′′ with the same label may correspond to the same grid index in Stage III, it is possible that for some i = j with y i = y j , we have m i = m j . Let N ′ ≤ N denote the number of distinct pairs ( m i , y i ) . It remains to memorize these N ′ distinct data points in R .

<!-- formula-not-decoded -->

̸

Since m i = Flatten( x i ) ≤ R m +1 , we apply Theorem B.4 from Vardi et al. [2021] with r = R m +1 to construct f mem : R → R with width 12 and depth

<!-- formula-not-decoded -->

such that f mem ( m i ) = y i .

The final network f : R d → R is defined as

<!-- formula-not-decoded -->

The depth 1 network ¯ f proj ( x ) + b has width m , and also the depth 1 network f trans has width m . Flatten has width 5 m and depth O (log 2 R ) and f mem has width 12 and depth ˜ O ( √ N ) . The total construction requires ˜ O ( md + m 2 + m 2 + √ N ) = ˜ O ( d + √ N ) parameters, where each term md,m 2 , m 2 , and √ N comes from f proj , f trans , Flatten , and f mem respectively. The width of the final network is ˜ O (1) and the depth is ˜ O ( √ N ) .

The bit complexity of ¯ f proj is ˜ O (1) and that of b is

<!-- formula-not-decoded -->

The network f trans has the bit complexity log(max {∥ z i ∥ ∞ | i ∈ [ N ] } ) = ˜ O (1) . Flatten has the bit complexity ˜ O (1) , and f mem needs at most ˜ O ( √ N ) . Hence, the bit complexity of the final network is ˜ O ( √ N ) .

The following is the classical memorization upper bound of parameters used in the proof of Theorem B.3

̸

Theorem B.4 (Classical Memorization, Theorem 3.1 from Vardi et al. [2021]) . Let N,d,C ∈ N , and r, ϵ &gt; 0 , and let ( x 1 , y 1 ) , . . . , ( x N , y N ) ∈ R d × [ C ] be a set of N labeled samples with ∥ x i ∥ ≤ r for every i and ∥ x i -x j ∥ ≥ 2 ϵ for every i = j . Denote R := 5 rN 2 ϵ -1 √ πd . Then, there exists a neural network F : R d → R with width 12 and depth

<!-- formula-not-decoded -->

and bit complexity bounded by O (log d + √ N log N · max { log R, log C } ) such that F ( x i ) = y i for every i ∈ [ N ] .

## B.2 Sufficient Condition for Near-Perfect Robust Memorization with Moderate Robustness Radius

Theorem B.5. Let ρ ∈ ( 0 , 1 5 √ d ] , and η ∈ (0 , 1) . For any dataset D ∈ D d,N,C , there exists f with ˜ O ( Nd 1 4 ρ 1 2 ) parameters, depth ˜ O ( Nd 1 4 ρ 1 2 ) , width ˜ O (1) and bit complexity ˜ O ( 1 / d 1 4 ρ 1 2 ) that ρ -robustly memorizes D with error at most η .

Proof. For given ρ , any desired error η , and D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N,C , we construct a network f that ρ -robustly memorizes D with ˜ O ( Nd 1 4 ρ 1 2 ) parameters.

Stage I (Projection onto log -scale Dimension and Scaling via the First Hidden Layer Weight Matrix).

The first stage closely follows that of Theorem B.3. By Lemma B.20, we obtain an integer m = ˜ O (log N ) and a 1-Lipschitz linear map ϕ : R d → R m such that the projected dataset D ′ := { ( ϕ ( x i ) , y i ) } i ∈ [ N ] ∈ D m,N,C satisfies the separation bound

<!-- formula-not-decoded -->

We define f proj : R d → R m as f proj ( x ) = 11 9 · √ d ϵ D ϕ ( x ) , which is 11 9 · √ d ϵ D -Lipschitz.

We apply Lemma B.23 with f proj whose depth is 1 , ν = min { 109 11880 √ m, 1 88 µ, 1 360 N , 1 } and ¯ R := max {∥ x ∥ 2 | x ∈ B 2 ( x i , µ ) for some i ∈ [ N ] } to obtain ¯ f proj with the same number of parameters, depth and width and ˜ O (1) bit complexity such that

<!-- formula-not-decoded -->

We set the first hidden layer bias b ∈ R m so that

<!-- formula-not-decoded -->

where the comparison between two vectors is element-wise.

We obtain the grouping scale α ∈ [0 , 1] here for Stage II-we call α the grouping scale, as we group the points by approximately N α points per group in Stage II. From the ρ condition, we have 1 5 ρ √ d ≥ 1 . Thus, there exists α ∈ [0 , 1] such that satisfies ⌈ N α ⌉ = ⌊ 1 5 ρ √ d ⌋ . Let us bound the ρ in terms of α . Since ⌈ N α ⌉ = ⌊ 1 5 ρ √ d ⌋ ≤ 1 5 ρ √ d , we have

<!-- formula-not-decoded -->

̸

We claim that for D ′′ := { ( σ ( ¯ f proj ( x i ) + b ) , y i ) } i ∈ [ N ] ∈ D m,N,C , we have (i) ϵ D ′′ ≥ √ m/ 2 and (ii) for ρ ′′ := 1 4 ⌊ N α ⌋ ϵ D′′ , if g ( x ) ∈ F m,P can ρ ′′ -robustly memorize D ′′ , then g ◦ σ ◦ ( ¯ f proj ( x i ) + b ) can ρ -robustly memorize D . For any i = j with y i = y j , we have

̸

<!-- formula-not-decoded -->

where (a) holds by the construction of b (Equation (19)). For simplicity, we denote

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

where (a) holds from ∥ a + b ∥ 2 2 ≥ ( ∥ a ∥ 2 -∥ b ∥ 2 ) 2 . By the construction of ¯ f (Equation (18)),

<!-- formula-not-decoded -->

so we have

<!-- formula-not-decoded -->

Now we derive

<!-- formula-not-decoded -->

where (a) is by the definition of f proj , (b) is by the definition of D ′ and its separation constant, and (c) follows from Equation (17).

Plugging this inequality to Equation (21) gives

<!-- formula-not-decoded -->

where (a) holds from ν ≤ 109 11880 √ m , and (b) holds from Equation (22). This proves the first claim ϵ D ′′ ≥ √ m/ 2 . To prove the second claim, let µ := ρϵ D and µ ′′ := ρ ′′ ϵ D ′′ . Then,

<!-- formula-not-decoded -->

where (a) and (j) are by Equation (19), (b) and (g) are by the construction of ¯ f (Equation (18)), (c) is because ν ≤ 1 88 µ , (d) is because f proj is 11 9 · √ d ϵ D -Lipschitz, (e) uses µ = ρϵ D , (f) uses ρ ≤ 1 5 N √ d , (h) is because ν ≤ 1 360 N and (i) is because ρ ′′ ϵ D ′′ = 1 4 Nϵ D′′ ϵ D ′′ = 1 4 N .

<!-- formula-not-decoded -->

Hence, g ( x ) memorizing the robustness ball B 2 ( σ ( ¯ f proj ( x i ) + b ) , ρ ′′ ϵ D ′′ ) on projected space leads to g ◦ σ ◦ ( ¯ f proj ( x ) + b ) memorizing the robustness ball for D . In other words, if g ( x ) ∈ F m,P can ρ ′′ -robustly memorize D ′′ , then g ◦ σ ◦ ( ¯ f proj ( x ) + b ) can ρ -robustly memorize D . With ρ ′′ = 1 4 ⌊ N α ⌋ ϵ D′′ , Stage II aims to find a ρ ′′ -robust memorizer g of D ′′ . For simplicity of the notation, let z i := σ ( ¯ f proj ( x i ) + b ) for each i ∈ [ N ] , so that D ′′ = { ( z i , y i ) } i ∈ [ N ] .

Stage II (Memorizing N α Points at Each Layer): Using the grouping scale α obtain in Stage I, we group N data points to ⌈ N 1 -α ⌉ groups with index { I j } N 1 -α j =1 , each with | I j | ≤ ⌊ N α ⌋ +1 . Then, we construct ˜ f j that memorizes data points and their robustness balls with index I j , and the error rate remains small for other data points and their robustness balls.

For each j ∈ [ ⌈ N 1 -α ⌉ ] , we apply Lemma B.13 with error rate η ← η N 1 -α , α ← α , D ← D ′′ ∈ D m,N,C , ρ ← ρ ′′ and I ← I j . Then it satisfies that ϵ D ′′ ≥ √ m/ 2 , ρ ′′ = 1 4 ⌊ N α ⌋ ϵ D′′ , and | I | ≤ ⌊ N α ⌋ +1 . Thus, we obtain a neural network ˜ f j with width O ( m ) = ˜ O (1) , depth ˜ O ( N α 2 ) and ˜ O ( N α 2 + m 2 ) = ˜ O ( N α 2 ) parameters and bit complexity ˜ O ( N α 2 + m ) = ˜ O ( N α 2 ) such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We define for each j :

<!-- formula-not-decoded -->

so that the last coordinate is given as y + σ ( ˜ f j ( z ) -y ) = max { ˜ f j ( z ) , y } . Finally, we define the full robust memorizing network as

<!-- formula-not-decoded -->

̸

We now verify the correctness of the construction. For any x ∈ B ( x i , ρϵ D ) and z = f proj ( x ) ∈ B ( z i , ρ ′′ ϵ D ′′ ) , since we partition [ N ] into disjoint groups { I j } j ∈ [ N 1 -α ] , there exists a unique index j i such that i ∈ I j i and thus ˜ f j i ( z ) = y i holds. For all j = j i , the networks satisfy ˜ f j ( z ) ∈ { 0 , y i } with high probability, so none of them can exceed y i . Since the final network outputs the maximum among y and all ˜ f j ( z ) , we have f ( x ) = y i as long as each ˜ f j ( z ) ∈ { 0 , y i } . Therefore, it suffices to show that ˜ f j ( z ) ∈ { 0 , y i } holds for ∀ j , namely,

<!-- formula-not-decoded -->

Considering the contrapositive, we have

̸

<!-- formula-not-decoded -->

Since each ˜ f j satisfies P z ∼ Unif( B ( z i ,ρ ′′ ϵ D′′ )) [ ˜ f j ( z ) ∈ { 0 , y i } ] ≥ 1 -η N 1 -α for all j ∈ [ N 1 -α ] , we upper bound the error probability using the union bound:

̸

<!-- formula-not-decoded -->

≤ η, where (a) holds by Equation (23). Hence,

<!-- formula-not-decoded -->

̸

We verify width, depth and the number of parameters of f . Recall that the final network is

<!-- formula-not-decoded -->

The depth 1 network ¯ f proj ( x ) + b has width m = ˜ O (1) . For j ∈ [ ⌈ N 1 -α ⌉ ] , each ˜ f j has width ˜ O (1) , depth ˜ O ( N α 2 ) and ˜ O ( N α 2 ) parameters.

The network ¯ f proj needs dm = ˜ O ( d ) parameters. Hence, the number of parameters of f is

<!-- formula-not-decoded -->

where (a) holds by ⌈ N α ⌉ = ⌊ 1 5 ρ √ d ⌋ . The width of f is ˜ O (1) and the depth of f is ˜ O ( N 1 -α × N α 2 ) = ˜ O ( Nd 1 4 ρ 1 2 ) .

The bit complexity of ¯ f proj is ˜ O (1) and that of b is log(max {∥ ¯ f proj ( x ) ∥ ∞ | x ∈ B 2 ( x i , µ ) for some i ∈ [ N ] } ) = ˜ O (1) . The network f j has the same bit complexity as ˜ f j , which is ˜ O ( N α 2 ) = ˜ O ( 1 / d 1 4 ρ 1 2 ) . Hence, the bit complexity of the final network is ˜ O ( 1 / d 1 4 ρ 1 2 ) .

The above construction is motivated by the need to handle overlapped robustness balls with the same label. We transform the construction of classical memorization in Vardi et al. [2021] in two key directions: first, from memorizing isolated data points x i to memorizing entire robustness neighborhoods B p ( x i , µ ) ; and second, to ensuring correct classification even within regions where multiple robustness balls with the same label overlap. To accomplish this, we introduce disjoint, integer-aligned interval encodings and carefully control the error propagation caused by dimension reduction, as addressed in Lemma B.11.

## B.2.1 Memorization of Integers with Sublinear Parameters in N

Lemmas in this subsection are a slight extension of those in Vardi et al. [2021], adapted to our integer-based encoding scheme.

From here, BIN i : j ( n ) denotes the bit string from position i to j (inclusive) in the binary representation of n . For example, BIN 1:3 (37) = 4 , since (37) 10 = (100101) 2 so that BIN 1:3 (37) = (100) 2 = (4) 10 .

Lemma B.6. Let η &gt; 0 and m,n ∈ N with m&lt;n . Then, there exists a neural network F : R → R with width 2, depth 2 and bit complexity ˜ O (1) such that

<!-- formula-not-decoded -->

Proof. We construct a network F :

<!-- formula-not-decoded -->

It satisfies the requirements with depth 2 and width 2. The bit complexity is O (log m + log n + log(1 /η )) = ˜ O (1) .

Lemma B.7. Let η ∈ (0 , 1) , and let m 1 &lt; · · · &lt; m N be natural numbers. Let N 1 , N 2 ∈ N satisfy N 1 · N 2 ≥ N , and let w 1 , . . . , w N 1 ∈ N . Then, there exists a neural network F : R → R with width 4 , depth 3 N 1 +2 and bit complexity ˜ O (1) such that,

<!-- formula-not-decoded -->

where we define m N +1 = · · · = m N 1 N 2 = m N .

Proof. Let j ∈ [ N 1 ] . We define network blocks ˜ F j : R → R and F j : R 2 → R 2 as follows. By applying Lemma B.6, we construct ˜ F j such that:

<!-- formula-not-decoded -->

As a result, for any i ∈ [ N ] , any x ∈ [ m i , m i +1 -η ] satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we define the network F ( x ) = ( 0 1 ) ⊤ F N 1 ◦ · · · ◦ F 1 (( x 0 )) .

̸

We now verify the correctness of the construction. For i ∈ [ N ] , let x ∈ [ m i , m i + 1 -η ] . For j = ⌈ i N 2 ⌉ , we have ˜ F j ( x ) = 1 , and for all j ′ = j , ˜ F j ′ ( x ) = 0 . Therefore, the output of F satisfies F ( x ) = w j = w ⌈ i N 2 ⌉ .

The width of each F j is at most the width required to implement ˜ F j , plus two additional units to carry the values of x and y . Since the width of ˜ F j is 2, the width of F is at most 4 . Each block F j has depth 3, and F is a composition of N 1 blocks. Additionally, one layer is used for the input to get x ↦→ ( x 0 ) , and another to extract the last coordinate of the final input. Thus, the total depth of F is

<!-- formula-not-decoded -->

3 N 1 +2 . The bit complexity is O (1) .

Lemma B.8 (Lemma A.7, Vardi et al. [2021]) . Let n ∈ N and let i, j ∈ N with i &lt; j ≤ n . Denote Telgarsky's triangle function by φ ( z ) := σ ( σ (2 z ) -σ (4 z -2)) . Then, there exists a neural network F : R 2 → R 3 with width 5 and depth 3( j -i +1) , and bit complexity n +2 , such that for any x ∈ N

<!-- formula-not-decoded -->

In the following lemma, note that ρ does not refer to the robustness ratio.

̸

Lemma B.9 (Extension of Lemma A.5, Vardi et al. [2021]) . Let η &gt; 0 , and let n, ρ, c ∈ N and u, w ∈ N . Assume that for all ℓ, k ∈ { 0 , 1 , . . . , n -1 } with ℓ = k , the bit segments of u satisfy

̸

<!-- formula-not-decoded -->

Then, there exists a neural network F : R 3 → R with width 12 , depth 3 n · max { ρ, c } +2 n +2 and bit complexity n max { ρ, c } +2 , such that the following holds:

For every x &gt; 0 , if there exist j ∈ { 0 , 1 , . . . , n -1 } such that

<!-- formula-not-decoded -->

then the network satisfies

Next, we define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We define the triangle function φ ( z ) := σ ( σ (2 z ) -σ (4 z -2)) as introduced by Telgarsky [2016]. For i ∈ { 0 , 1 , . . . , n -1 } , we construct a network block F i :

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

To compute y i , we first extract the relevant bit segments from u and w using Lemma B.8. We define two subnetworks F w i , F u i :

<!-- formula-not-decoded -->

A subnetwork F u i maps the pair of triangle encodings of u to the updated encodings for i + 1 , along with the extracted bits BIN i · ρ +1:( i +1) · ρ ( u ) . A subnetwork F w i does the same for w , yielding BIN i · c +1:( i +1) · c ( w ) .

We then construct a network with width 2 and depth 2 to obtain y i from inputs BIN i · ρ +1:( i +1) · ρ ( u ) and x . Firstly, we use Lemma B.6 to construct a network that output ˜ y i :

<!-- formula-not-decoded -->

Secondly, we construct the following 1 -layer network that use ˜ y i as input:

<!-- formula-not-decoded -->

This ensures that the output is BIN i · c +1:( i +1) · c ( w ) if ˜ y i = 1 , and the output is 0 if ˜ y i = 0 since BIN i · c +1:( i +1) · c ( w ) ≤ 2 c .

Finally, the full network F is constructed as a composition:

<!-- formula-not-decoded -->

where for x, w, u &gt; 0 :

(1) H : R 3 → R 6 is a 1-layer network that maps ( x, w, u ) to the required initial encoding inputs, namely:

<!-- formula-not-decoded -->

(2) G : R 6 → R is a 1-layer network that outputs the last coordinate.

We verify the correctness of the construction. The output of the full network is given by:

<!-- formula-not-decoded -->

If there exists j ∈ { 0 , 1 . . . , n -1 } such that x ∈ [BIN ρ · j +1: ρ · ( j +1) ( u ) , BIN ρ · j +1: ρ · ( j +1) ( u )+1 -η ] , then by the construction we obtain y j = BIN c · j +1: c · ( j +1) ( w ) , while y ℓ = 0 for all ℓ = j . This is because the bit-encoded intervals are disjoint as BIN ρ · ℓ +1: ρ · ( ℓ +1) ( u ) = BIN ρ · k +1: ρ · ( k +1) ( u ) . Hence, the final output of F is:

<!-- formula-not-decoded -->

We now analyze the width and depth of the constructed network F . Each block F i comprises F w i and F u i , each of width 5. In addition, two neurons are used to process x and y , resulting in a total width of 12. The outputs ˜ y i and y i are produced by additional layers with width 2 and 1, respectively, both of which are smaller than 12 . We also compose the networks H and G , with width 6 and 1, respectively, again remaining within 12 .

Each of the networks F u i and F w i has depth at most 3 max { ρ, c } . The layers obtaining ˜ y i and y i contribute an additional 2 layers, resulting in a total depth of 3 max { ρ, c } + 2 for each block F i . Composing all n such blocks, and including one additional layer each for H and G , the total depth of the network F is 3 n · max { ρ, c } +2 n +2 .

The bit complexity of F u i , F w i and H is bounded by n max { ρ, c } +2 , and all other parts of the network require less bit complexity. Hence, the bit complexity of F is bounded by n max { ρ, c } +2 .

## B.2.2 Precise Control of Robust Memorization Error

Lemma B.13 constructs the network for Stage II in Theorem B.5, while the robust memorization error is controlled in Lemma B.11.

̸

Lemma B.10. Let N,C ∈ N , and let ( m 1 , y 1 ) , . . . , ( m N , y N ) ∈ D 1 ,N,C ⊂ N × [ C ] be a set of N labeled samples with m i = m j for every i = j . Then, there exists a neural network F : R → R with width 12 , depth ˜ O ( √ N ) , ˜ O ( √ N ) parameters and bit complexity ˜ O ( √ N ) such that

̸

<!-- formula-not-decoded -->

Proof. Let M = { m i } i ∈ [ N ] . We group the elements in M to ⌈ √ N ⌉ groups, each containing at most ⌊ √ N ⌋ +1 natural numbers inside. For each interval indexed by j ∈ { 1 , . . . , ⌈ √ N ⌉} , we define two integers w j , u j ∈ N to encode the integer m i ∈ M and the corresponding labels y i as follows.

For each i ∈ [ N ] , letting j := ⌈ i ⌊ √ N ⌋ +1 ⌉ , k := i mod ( ⌊ √ N ⌋ +1) and R := max i ∈ [ N ] m i , we define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, in each group j , the integer u j contains log 2 R bits per integer, which represent the k -th integer in this group. In the same manner, w j contains log 2 C bits per integer, which represent the label of the k -th integer in this group.

By applying Lemma B.7 to η = 1 2 , we construct a neural network F 1 that maps m ∈ M to their corresponding groups, and maps m ∈ N \ ⋃ j ∈ [ ⌈ √ N ⌉ ] [ m ( j -1)( ⌊ √ N ⌋ +1)+1 , m j ( ⌊ √ N ⌋ +1) + 1) to 0 . Thus, all natural numbers are assigned to their corresponding group or 0 .

For each i ∈ [ N ] , we define the group index

<!-- formula-not-decoded -->

̸

̸

Then, the network F 1 maps any input m ∈ M to the representation

<!-- formula-not-decoded -->

and F 1 ( m ) = ( m 0 0 ) for m ∈ N \ ⋃ j ∈ [ ⌈ √ N ⌉ ] [ m ( j -1)( ⌊ √ N ⌋ +1)+1 , m j ( ⌊ √ N ⌋ +1) +1) . The network F 1 has width 9, depth ˜ O ( √ N ) and bit complexity ˜ O (1) .

Now, we apply Lemma B.9 to construct a network F 2 : R 3 → R with the following property. For each i ∈ [ N ] , j ∈ [ ⌈ √ N ⌉ ] , and k ∈ { 0 , . . . , ⌊ √ N ⌋ } , suppose that m i is the k -th integer in the j -th group. Then, the network satisfies :

<!-- formula-not-decoded -->

Moreover, for m ∈ N \ M , F 2 (( m w j u j )) = 0 or F 2 (( m 0 0 )) = 0 . Thus, the network F 2 extracts the label corresponding to each data point from the encoded label set of the group to which the interval belongs or outputs 0 . The network F 2 has width 12, depth ˜ O ( √ N ) and bit complexity ˜ O ( √ N ) .

Finally, we define the classifier network F : R d → R as

<!-- formula-not-decoded -->

The overall network F has width 12 and depth ˜ O ( √ N ) , which corresponds to the maximum width and total depth of its component networks. The bit complexity of F is ˜ O ( √ N ) .

Lemma B.11. Let B 2 ( x 0 , µ ) be a Euclidean ball with center x 0 ∈ R d and radius µ &gt; 0 . Let u ∈ R d be a unit vector, and define the affine function f ( x ) := 1 2 µ ( u ⊤ x + b ) for some b ∈ R . Then for any interval I ⊂ R of length η , the volume fraction of the ball mapped into I satisfies:

<!-- formula-not-decoded -->

where V d = π d/ 2 Γ( d 2 +1) denotes the volume of the d -dimensional unit ball.

Proof. Let x = x 0 + µ y , so that y ∈ B d ( 0 , 1) . Under this change of variables,

<!-- formula-not-decoded -->

Thus, f ( x ) ∈ I if and only if u ⊤ y ∈ J , where

<!-- formula-not-decoded -->

is an interval of length 2 η . We define the preimage of I under f with the intersection of B 2 ( x 0 , µ ) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The distribution of u ⊤ y , where y ∼ Unif( B 2 ( 0 , 1)) , has density

<!-- formula-not-decoded -->

Then,

Thus,

Hence,

<!-- formula-not-decoded -->

Lemma B.12. For all integers d ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where V d = π d/ 2 Γ ( d 2 +1 ) is the volume of the d -dimensional unit ball.

Proof. Set

Let x = d 2 and define

<!-- formula-not-decoded -->

Differentiating and using the digamma function ψ = Γ ′ / Γ , we get

<!-- formula-not-decoded -->

since ψ is strictly increasing. Hence R d is strictly decreasing in d . Therefore max d ≥ 1 R d = R 1 = V 1 /V 0 = 2 , which proves R d ≤ 2 with equality only at d = 1 .

Lemma B.13. Let η ∈ (0 , 1) , α ∈ [0 , 1] and D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N,C be a dataset with separation ϵ D ≥ √ d/ 2 , and let the robustness ratio be ρ = 1 4 ⌊ N α ⌋ ϵ D . Then, for any index set I ⊆ [ N ] with | I | ≤ ⌊ N α ⌋ + 1 , there exists a neural network f with width O ( d ) , depth ˜ O ( N α 2 ) , ˜ O ( N α 2 + d 2 ) parameters and ˜ O ( N α 2 + d ) bit complexity such that:

<!-- formula-not-decoded -->

A network f , obtained from this lemma, memorizes each data point and its robustness ball for all indices i ∈ I . f maps every other data point and its robustness ball to either its correct label or 0 with high probability 1 -η .

Proof. We construct a network proceeding in three stages. In each stage, we define subnetworks such that their composition satisfies the requirements.

Stage I (Translation for Distancing from Lattice via the Bias) We first translate the data points so that for i ∈ I , the robustness ball centered at x i lies far from integer lattice boundaries. This ensures that each ball lies entirely within a single unit grid cell. By applying Lemma B.15 to the points { x i } i ∈ I , we obtain a translation vector b = ( b 1 , · · · , b d ) ∈ R d with bit complexity ⌈ log(6 | I | ) ⌉ such that

<!-- formula-not-decoded -->

i.e., the translated points { x i -b } i ∈ I are coordinate-wise far from the integer lattice. Additionally, we apply an integer-valued translation (coordinate-wise) so that all coordinates of the points { x i -b } i ∈ [ N ] become positive, while preserving the distance property in Equation (24). Hence, without loss of generality, we can assume b also has the property

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let D ′ := { ( x ′ i , y i ) } i ∈ [ N ] , where x ′ i := x i -b . Then ϵ D ′ = ϵ D . For ρ ′ := ρ = 1 4 ⌊ N α ⌋ ϵ D , we have the robustness radius µ ′ := ρ ′ ϵ D ′ = ρϵ D = 1 4 ⌊ N α ⌋ . Define f trans as f trans ( x ) := x -b . Then, f trans can be implemented via one hidden layer with O ( d 2 ) parameters in a neural network.

Since the translation preserves separation ( ϵ D = ϵ D ′ ) and ball containment properties (robustness ball of D is mapped to the robustness ball of D ′ through the translation), it suffices to construct a network that satisfies the requirements with ρ ← ρ ′ and D ← D ′ . Observe that the robustness balls after Stage I are not affected when passing the σ , by Equations (24) and (25).

Stage II (Grid Indexing) From Equation (24), each x ′ i ∈ R d (for i ∈ I ) is at least 4 3 µ ′ distant from any lattice hyperplane H z,j := { x ∈ R d | x j = z } for each j ∈ [ d ] and z ∈ Z . Hence, each robustness ball centered at x ′ i (for i ∈ I ) lies completely within a single integer lattice (or unit grid) ∏ d j =1 [ n j , n j +1) , for some ( n 1 , · · · , n d ) ∈ Z d . Moreover, for any x ∈ B 2 ( x ′ i , µ ′ ) , the distance from the integer lattice remains at least µ ′ .

̸

Furthermore, by the separation condition ϵ D ′ = ϵ D ≥ √ d/ 2 , for any i = i ′ with y i = y i ′ , we have ∥ x ′ i -x ′ i ′ ∥ 2 ≥ √ d . Since sup {∥ x -x ′ ∥ 2 | x , x ′ ∈ ∏ d j =1 [ n j , n j +1) } = √ d , two such points cannot lie in the same grid. Recall the separation condition holds for all data points x ′ i for i ∈ [ N ] and each ball B 2 ( x ′ i , µ ′ ) (for i ∈ I ) lies within a single grid. We conclude that for each i ∈ I , the robustness ball B 2 ( x ′ i , µ ′ ) is not intersected by any other robustness ball B 2 ( x ′ j , µ ′ ) with a different label, for any j ∈ [ N ] , i.e., no ball with a different label overlaps the grid cell containing B 2 ( x ′ i , µ ′ ) .

̸

We define R := ⌈ max i ∈ I ∥ x ′ i ∥ ∞ (= max i ∈ I,j ∈ [ d ] ( x ′ i,j )) ⌉ ∈ N . Our goal in this stage is to construct Flatten mapping defined as

<!-- formula-not-decoded -->

This maps each grid ∏ d j =1 [ n j , n j +1 ) onto the point ∑ d j =1 R j -1 n j .

However, since Flatten is discontinuous due to the use of floor functions, we construct Flatten which is a continuous approximation that exactly matches Flatten on the region ⋃ i ∈ I B 2 ( x ′ i , µ ′ ) , and incurs only a small error on the remaining region ⋃ i ∈ [ N ] \ I B 2 ( x ′ i , µ ′ ) . We choose large enough t ∈ N so that for η ′ := 1 /t , we have η ′ ≤ V d 2 dV d -1 µ ′ η where V d = π d/ 2 Γ( d 2 +1) denotes the volume of the d -dimensional unit ball. Moreover, we can take such t ∈ N which at the same time satisfies

<!-- formula-not-decoded -->

By Lemma B.16, for γ := 1 /t = η ′ and n := ⌈ log 2 R ⌉ , we obtain the network Floor := Floor ⌈ log 2 R ⌉ with O (log 2 R ) parameters such that

<!-- formula-not-decoded -->

Since we apply γ = 1 /t to Lemma B.16, Floor can be implemented with O ( n +log t ) = O (log R + log( d 2 ⌊ N α ⌋ /η )) = O (log( dRN/η )) bit complexity. In particular, we can define our network Flatten with O (log( dRN/η ) + log R d -1 ) = O (log( dRN/η ) + d log R ) = ˜ O ( d ) bit complexity as

<!-- formula-not-decoded -->

As Floor : R → R can be implemented with width 5 and depth O (log 2 R ) network (Lemma B.16), Flatten can be implemented with width 5 d and depth O (log 2 R ) network. Thus, we can construct Flatten with O ( d 2 log 2 R ) = ˜ O ( d 2 ) parameters.

We first observe that this implementation is valid on the region ⋃ i ∈ I B 2 ( x ′ i , µ ′ ) . For i ∈ I and x ∈ B 2 ( x ′ i , µ ′ ) , we have

<!-- formula-not-decoded -->

where (a) holds by Equation (24), (b) holds since η &lt; 1 , (c) holds since V d V d -1 ≤ 2 by Lemma B.12, and (d) holds from the choice of η ′ . Thus, for any i ∈ I and any x ∈ B 2 ( x ′ i , µ ′ ) , x j satisfies the

requirement in Equation (26). Therefore, we guarantee that each robustness ball centered at x ′ i for i ∈ I lies in the region where the Flatten is properly approximated by Flatten . i.e.

<!-- formula-not-decoded -->

Since Flatten maps each unit grid into a point and each robustness ball centered at x ′ i for i ∈ I lies on a single unit grid, we conclude

<!-- formula-not-decoded -->

Let m i := Flatten( x ′ i ) for i ∈ I . Then for i ∈ I , each robustness ball centered at x ′ i is mapped to m i . We have m i ∈ Z ∩ [0 , R d +1 ] for all i ∈ I , since

<!-- formula-not-decoded -->

where (a) is by ∥ x ′ i ∥ ∞ ≤ R .

Next, we consider the case i ∈ [ N ] \ I . Note that the lattice distance condition in Equation (24) applies only to the subset { ( x i , y i ) } i ∈ I , rather than the entire dataset. As a result, for indices i ∈ [ N ] \ I , the distance from the lattice is not guaranteed. Thus, it can lie across the lattice.

For i ∈ [ N ] \ I , we analyze the error of Flatten on the remaining region ⋃ i ∈ [ N ] \ I B 2 ( x ′ i , µ ′ ) . For i ∈ [ N ] \ I , we have

̸

<!-- formula-not-decoded -->

where (a) follows from Equation (26) and the fact that Flatten( x ) = Flatten( x ) whenever x j -⌊ x j ⌋ &gt; η ′ for all j ∈ [ d ] , (b) follows from Lemma B.11 applied to a unit vector u = e j , b = 0 , and an interval I j µ ′ , and (c) holds by the choice of η ′ . Hence, we have

̸

<!-- formula-not-decoded -->

̸

We observe at what happens if Flatten( x ) = Flatten( x ) for i ∈ [ N ] \ I and x ∈ B 2 ( x ′ i , µ ′ ) . To ensure that no robustness ball centered at x i for i ∈ [ N ] \ I is mapped to grid index m j with a different label, namely, satisfying j ∈ I with y i = y j , we define label-specific grid index sets. For each class c ∈ [ C ] , define the set

<!-- formula-not-decoded -->

where G c is the collection of all grid indices m i assigned to data points in I that have label y i = c . In other words, G c contains all grid cells that are claimed by class c . The set G represents all valid grid indices.

̸

Recall that for each i ∈ I , the robustness ball B 2 ( x ′ i , µ ′ ) is not intersected by any other robustness ball B 2 ( x ′ j , µ ′ ) with a different label. Specifically, consider i ∈ [ N ] \ I . For j ∈ I with y i = y j , the robustness ball B 2 ( x ′ i , µ ′ ) can have a portion that intersects the grid containing B 2 ( x ′ j , µ ′ ) , then the portion is mapped to the corresponding grid index m j . However, for j ∈ I with y i = y j , the robustness ball never intersects the grid, and is never mapped to m j . Formally, if Flatten( x ) ∈ G , it must be Flatten( x ) ∈ G y i . Otherwise, Flatten( x ) / ∈ G , i.e., the robustness ball does not intersect any selected grid. Thus, combining the probabilities,

<!-- formula-not-decoded -->

where (a) holds since P x ∈ Unif( B 2 ( x ′ i ,µ ′ )) [Flatten( x ) ∈ G y i or Flatten( x ) / ∈ G ] = 1 , and (b) follows by Equation (28). Hence, if we memorize { ( m i , y i ) } i ∈ I and map other integer N \ { m i } i ∈ I to zero, B 2 ( x ′ i , µ ′ ) for i ∈ I is exactly mapped to y i , and with high probability 1 -η , B 2 ( x ′ i , µ ′ ) for i ∈ [ N ] \ I is mapped to either y i or 0 .

̸

Stage III (Memorization) Finally, we construct the network to memorize ⌊ N α ⌋ points { ( m i , y i ) } I i =1 ⊂ Z ≥ 0 × [ C ] . Since multiple robustness balls for D ′ with the same label may correspond to the same grid index in Stage II, it is possible that for some i = j with y i = y j , we have m i = m j . Let N ′ ≤ | I | denote the number of distinct pairs ( m i , y i ) . It remains to memorize these N ′ distinct data points in R .

Applying Lemma B.10, we obtain a neural network f mem with width 12, depth ˜ O ( N ) , ˜ O ( N α 2 ) parameters and bit complexity ˜ O ( N α 2 ) satisfying:

<!-- formula-not-decoded -->

For m ∈ N , f mem ( m ) = c for some c ∈ [ C ] if and only if m ∈ G c .

The final network f : R d → R is defined as

<!-- formula-not-decoded -->

Let us verify the correctness of the construction.

For i ∈ I and any x ∈ B ( x i , ρϵ D ) , we have

<!-- formula-not-decoded -->

where (a) holds since Flatten ◦ σ ◦ f trans ( x )= m i , (b) holds since m i ∈ N , and (c) follows that f is constructed to memorize { ( m i , y i ) } I i =1 .

Next, consider i ∈ [ N ] \ I . We observe

<!-- formula-not-decoded -->

where (a) holds from the construction of f mem , (b) holds using x ′ := σ ◦ f trans ( x ) , (c) holds by Equation (25) and (d) holds by Equation (29). This concludes the proof.

The depth 1 network f trans has width d . Flatten has width 5 d and depth O (log 2 R ) and f mem has width 12 and depth ˜ O ( N α 2 ) . The total construction requires ˜ O ( d 2 + d 2 + N α 2 ) = ˜ O ( d 2 + N α 2 )

parameters, where each term d 2 , d 2 , and N α 2 comes from f trans , Flatten , and f mem respectively. The width of the final network is O ( d ) and the depth is ˜ O ( N α 2 ) .

The bit complexity of f trans is O (log N α , log(max {∥ x i ∥ ∞ | i ∈ [ N ] } )) = ˜ O (1) . Flatten has the bit complexity ˜ O ( d ) , and f mem needs at most ˜ O ( N α 2 ) . Hence, the bit complexity of the final network is ˜ O ( N α 2 + d ) .

## B.3 Sufficient Condition for Robust Memorization with Large Robustness Radius

Theorem B.14. Let ρ ∈ ( 1 5 √ d , 1 ) . For any dataset D ∈ D d,N,C , there exists f with ˜ O ( Nd 2 ρ 4 ) parameters, depth ˜ O ( N ) , width ˜ O ( ρ 2 d ) and bit complexity ˜ O ( N ) that ρ -robustly memorizes D .

Proof. Let D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N,C be given. We divide the proof into five cases, the first case under ρ ∈ [1 / 3 , 1) , the second case under ρ ∈ (1 / 5 √ d, 1 / 3) and d &lt; 600 log N , the third case under ρ ∈ (1 / 5 √ d, 1 / 3) and N &lt; 600 log N ≤ d , the fourth case under ρ ∈ (1 / 5 √ d, 1 / 3) , N ≥ d ≥ 600 log N , and finally the fifth case under ρ ∈ (1 / 5 √ d, 1 / 3) and d &gt; N ≥ 600 log N . To check that these cases cover all the cases, refer to Figure 6.

Figure 6: Different cases for Theorem B.14. The left child is for the answer 'Yes', and the right child is for the answer 'No'

<!-- image -->

The first two cases follow easily from prior works, while the remaining cases require careful analysis using dimension reduction techniques. The most interesting cases are cases IV and V. While we track the width, depth, and parameter complexity for each case, we initially implement them using infinite precision. We address the bit complexity by approximating the infinite precision network using a finite precision network at the very last part of the proof. As a spoiler, the bit complexity of all cases is handled within a unified framework using Lemma B.23. Let us deal with each case one by one.

Case I: ρ ∈ [1 / 3 , 1) . In the first case, where ρ ∈ [1 / 3 , 1) , the result directly follows from the prior result by Yu et al. [2024]. In particular, we apply Lemma D.2. Let us denote R := max i ∈ [ N ] ∥ x i ∥ 2 and γ := (1 -ρ ) ϵ D . Note that R ≥ ∥ x i ∥ ∞ for all i ∈ [ N ] as ∥ x ∥ 2 ≥ ∥ x ∥ ∞ for all x ∈ R d . By applying Lemma D.2, there exists f ∈ F d,P with P = O ( Nd 2 (log( d γ 2 ) + log R )) parameters that ρ -robustly memorize D . The number of parameters can be further bounded as follows:

<!-- formula-not-decoded -->

where (a) is due to ρ = Ω(1) , (b) hides the logarithmic factors. Moreover, by Lemma D.2, the network has width O ( d ) = O ( ρ 2 d ) and depth ˜ O ( N ) .

Case II: ρ ∈ (1 / 5 √ d, 1 / 3) and d &lt; 600 log N . In the second case, where d &lt; 600 log N and (1 / 5 √ d, 1 / 3) , the result also directly follows from the prior result by Yu et al. [2024]. In particular, we apply Lemma D.2. Let us denote R := max i ∈ [ N ] ∥ x i ∥ 2 and γ := (1 -ρ ) ϵ D . Note that R ≥ ∥ x i ∥ ∞ for all i ∈ [ N ] as ∥ x ∥ 2 ≥ ∥ x ∥ ∞ for all x ∈ R d . By Lemma D.2, there exists f ∈ F d,P with

P = O ( Nd 2 (log( d γ 2 ) + log R )) parameters that ρ -robustly memorize D . The number of parameters can be further bounded as follows:

<!-- formula-not-decoded -->

where (a) is due to d ≤ 600 log N , (b) hides the logarithmic factors, and (c) is because N ≤ 625 Nd 2 ρ 4 for all ρ ∈ ( 1 5 √ d , 1 3 ) . Moreover, by Lemma D.2, the network has width O ( d ) = O (log N ) = ˜ O (1) = ˜ O ( ρ 2 d ) and depth ˜ O ( N ) .

Case III: ρ ∈ (1 / 5 √ d, 1 / 3) and N &lt; 600 log N ≤ d . In the third case, where N &lt; 600 log N ≤ d and (1 / 5 √ d, 1 / 3) , we first apply Proposition B.21 to D to obtain 1-Lipschitz linear φ : R d → R N such that D ′ := { ( φ ( x i ) , y i ) } i ∈ [ N ] has ϵ D ′ = ϵ D . This is possible as d ≥ N .

We apply Lemma D.2 by Yu et al. [2024] to D ′ . Let us denote R := max i ∈ [ N ] ∥ φ ( z i ) ∥ 2 and γ := (1 -ρ ) ϵ D ′ . Note that R ≥ ∥ φ ( z i ) ∥ ∞ for all i ∈ [ N ] as ∥ z ∥ 2 ≥ ∥ z ∥ ∞ for all z ∈ R N . By Lemma D.2, there exists f 1 ∈ F N,P with P = O ( N · N 2 (log( N γ 2 ) + log R )) parameters that ρ -robustly memorize D ′ . f 1 has width O ( N ) and depth ˜ O ( N ) .

Let f = f 1 ◦ φ . This can be implemented by changing the first hidden layer matrix of f 1 by composing φ . This is possible because φ is linear. f has at most dN additional parameters compared to f 1 , and has same width and depth as f 1 . Since f 1 is 1-Lipschitz and ϵ D ′ = ϵ D , every robustness ball of D is mapped to the robustness ball of D ′ via f 1 . As f 1 ρ -robustly memorizes D ′ , the composed f satisfies the desired property.

The number of parameters can be further bounded as follows:

<!-- formula-not-decoded -->

where (a) is due to N ≤ 600 log N , and (b) hides the logarithmic factors. The width of f is O ( N ) = O (log( N )) = ˜ O (1) = ˜ O ( ρ 2 d ) . The depth of f is ˜ O ( N ) .

Case IV: ρ ∈ (1 / 5 √ d, 1 / 3) , and N ≥ d ≥ 600 log N . In the fourth case, where d ≥ 600 log N , we utilize the dimension reduction technique by Proposition B.19. We apply Proposition B.19 to D with m = max {⌈ 9 dρ 2 ⌉ , ⌈ 600 log N ⌉ , ⌈ 10 log d ⌉} and α = 1 / 5 . Let us first check that the specified m satisfies the condition 24 α -2 log N ≤ m ≤ d for the proposition to be applied. α = 1 / 5 and m ≥ 600 log N ensure the first inequality 24 α -2 log N ≤ m . The second inequality m ≤ d is decomposed into three parts. Since ρ ≤ 1 3 , we have 9 dρ 2 ≤ d so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Additionally, as N ≥ 2 , we have d ≥ 600 log N ≥ 600 log 2 ≥ 400 . By Lemma B.22, this implies 10 log d ≤ d and therefore

<!-- formula-not-decoded -->

Gathering Equations (30) to (32) proves m ≤ d .

By the Proposition B.19, there exists 1-Lipschitz linear mapping ϕ : R d → R m and β &gt; 0 such that D ′ := { ( ϕ ( x i ) , y i ) } i ∈ [ N ] ∈ D m,N,C satisfies

<!-- formula-not-decoded -->

As m ≥ 10 log d , the inequality β ≥ 1 2 √ m d is also satisfied by Proposition B.19. Therefore, we have

<!-- formula-not-decoded -->

Moreover, 600 log N ≤ d implies

where (a) is by the definition of m . Moreover, since ϕ is 1-Lipschitz linear,

<!-- formula-not-decoded -->

for all i ∈ [ N ] . Hence, by letting R := max i ∈ [ N ] {∥ x i ∥ 2 } , we have ∥ ϕ ( x i ) ∥ 2 ≤ R for all i ∈ [ N ] .

Now, we set the first layer hidden matrix as the matrix W ∈ R m × d corresponding to ϕ under the standard basis of R d and R m . Moreover, set the first hidden layer bias as b := 2 R 1 = 2 R (1 , 1 , · · · , 1) ∈ R m . Then, we have

<!-- formula-not-decoded -->

for all x ∈ B 2 ( x i , ϵ D ) for all i ∈ [ N ] , where the comparison between two vectors are element-wise. This is because for all i ∈ [ N ] , j ∈ [ m ] and x ∈ B 2 ( x , ϵ D ) , we have

<!-- formula-not-decoded -->

where (a) is by Equation (35), (b) is by the triangle inequality, and (c) is due to R &gt; ϵ D .

We construct the first layer of the neural network as f 1 ( x ) := σ ( Wx + b ) which includes the activation σ . Then, by above properties, D ′′ := { ( f 1 ( x i ) , y i ) } i ∈ [ N ] satisfies

<!-- formula-not-decoded -->

This is because for i = j with y i = y j we have

̸

̸

<!-- formula-not-decoded -->

where (a) is by Equation (36), (b) is by the definition of the ϵ D ′ , (c) is by Equation (33), and (d) is by Equation (34). By Lemma D.2 applied to D ′′ ∈ D m,N,C , there exists f 2 ∈ F m,P with P = O ( Nm 2 (log( m ( γ ′′ ) 2 ) + log R ′′ )) number of parameters that 5 6 -robustly memorize D ′′ , where

<!-- formula-not-decoded -->

Here (a) is by Equation (37). Moreover f 2 has width O ( m ) and depth ˜ O ( N ) by Lemma D.2.

Now, we claim that f := f 2 ◦ f 1 ρ -robustly memorize D . For any i ∈ [ N ] , take x ∈ B 2 ( x i , ρϵ D ) . Then, by Equation (36), we have f 1 ( x ) = Wx + b and f 1 ( x i ) = Wx i + b so that

<!-- formula-not-decoded -->

Moreover, combining Equations (37) and (38) results ∥ f 1 ( x ) -f 1 ( x i ) ∥ 2 ≤ 5 6 ϵ D ′′ . Since f 2 5 6 -robustly memorize D ′′ , we have

<!-- formula-not-decoded -->

In particular, f ( x ) = y i for any x ∈ B 2 ( x i , ρϵ D ) , concluding that f is a ρ -robust memorizer D . Regarding the number of parameters to construct f , notice that f 1 consists of ( d +1) m = ˜ O ( d 2 ρ 2 )

parameters as m = ˜ O ( dρ 2 ) . f 2 consists of ˜ O ( Nm 2 ) = ˜ O ( Nd 2 ρ 4 ) parameters. Since the case IV assumes N ≥ d and large ρ regime deals with ρ ≥ 1 5 √ d , we have

<!-- formula-not-decoded -->

Therefore, f in total consists of ˜ O ( d 2 ρ 2 + Nd 2 ρ 4 ) = ˜ O ( Nd 2 ρ 4 ) number of parameters. Moreover, since f has the same width as f 2 and depth one larger than the depth of f 2 , it follows that f has width O ( m ) = ˜ O ( ρ 2 d ) and depth ˜ O ( N ) . This proves the theorem for the fourth case.

Case V: ρ ∈ (1 / 5 √ d, 1 / 3) , and d &gt; N ≥ 600 log N . The last case combines the two techniques used in Cases III and IV. We first apply Proposition B.21 to D to obtain 1-Lipschitz linear φ : R d → R N such that D ′ := { ( φ ( x i ) , y i ) } i ∈ [ N ] ∈ D N,N,C has ϵ D ′ = ϵ D . Note that we can apply the proposition since d ≥ N .

Next, we apply Proposition B.19 to D ′ ∈ D N,N,C with m = max {⌈ 9 Nρ 2 ⌉ , ⌈ 600 log N ⌉} and α = 1 / 5 . Let us first check that the specified m satisfies the condition 24 α -2 log N ≤ m ≤ N for the proposition to be applied. α = 1 / 5 and m ≥ 600 log N ensure the first inequality 24 α -2 log N ≤ m . The second inequality m ≤ N is decomposed into two parts. Since ρ ≤ 1 3 , we have 9 Nρ 2 ≤ N so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Gathering Equations (30) and (31) proves m ≤ N . Additionally, as N ≥ 2 , we have N ≥ 600 log N ≥ 600 log 2 ≥ 400 . By Lemma B.22, this implies 10 log N ≤ N .

By the Proposition B.19, there exists 1-Lipschitz linear mapping ϕ : R N → R m and β &gt; 0 such that D ′′ := { ( ϕ ( φ ( x i )) , y i ) } i ∈ [ N ] ∈ D m,N,C satisfies

<!-- formula-not-decoded -->

As m ≥ 600 log N ≥ 10 log N , the inequality β ≥ 1 2 √ m N is also satisfied by Proposition B.19. Therefore, we have

<!-- formula-not-decoded -->

where (a) is by the definition of m . Moreover, since φ and ϕ are both 1-Lipschitz linear, ϕ ◦ φ : R d → R m is also 1-Lipschitz linear. Therefore,

<!-- formula-not-decoded -->

for all i ∈ [ N ] . Hence, by letting R := max i ∈ [ N ] {∥ x i ∥ 2 } , we have ∥ ϕ ( φ ( x i )) ∥ 2 ≤ R for all i ∈ [ N ] .

Now, we set the first layer hidden matrix as the matrix W ∈ R m × d corresponding to ϕ ◦ φ under the standard basis of R d and R m . Moreover, set the first hidden layer bias as b := 2 R 1 = 2 R (1 , 1 , · · · , 1) ∈ R m . Then, we have

<!-- formula-not-decoded -->

for all x ∈ B 2 ( x i , ϵ D ) for all i ∈ [ N ] , where the comparison between two vectors are element-wise. This is because for all i ∈ [ N ] , j ∈ [ m ] and x ∈ B 2 ( x , ϵ D ) , we have

<!-- formula-not-decoded -->

where (a) is by Equation (43), (b) is by the triangle inequality, and (c) is due to R ≥ ϵ D .

We construct the first layer of the neural network as f 1 ( x ) := σ ( Wx + b ) which includes the activation σ . Next, we show that, D ′′ := { ( f 1 ( x i ) , y i ) } i ∈ [ N ] satisfies

<!-- formula-not-decoded -->

Moreover, 600 log N ≤ N implies

̸

by the above properties. This is because for i = j with y i = y j we have

̸

<!-- formula-not-decoded -->

where (a) is by Equation (44), (b) is by the definition of the ϵ D ′′ , (c) is by Equation (41), (d) is by Equation (42), and (e) is because ϵ D ′ = ϵ D .

By Lemma D.2 applied to D ′′ ∈ D m,N,C , there exists f 2 ∈ F m,P with P = O ( Nm 2 (log( m ( γ ′′ ) 2 ) + log R ′′ )) number of parameters that 5 6 -robustly memorize D ′′ , where

<!-- formula-not-decoded -->

Here, (a) is by Equation (45). Moreover, f 2 has width O ( m ) and depth ˜ O ( N ) by Lemma D.2.

Now, we claim that f := f 2 ◦ f 1 ρ -robustly memorize D . For any i ∈ [ N ] , take x ∈ B 2 ( x i , ρϵ D ) . Then, by Equation (44), we have f 1 ( x ) = Wx + b and f 1 ( x i ) = Wx i + b so that

<!-- formula-not-decoded -->

Moreover, putting Equation (45) to Equation (46) results ∥ f 1 ( x ) -f 1 ( x i ) ∥ 2 ≤ 5 6 ϵ D ′′ . Since f 2 5 6 -robustly memorize D ′′ , we have

<!-- formula-not-decoded -->

In particular, f ( x ) = y i for any x ∈ B 2 ( x i , ρϵ D ) , concluding that f is a ρ -robust memorizer D .

Regarding the number of parameters to construct f , notice that f 1 consists of ( d +1) m = ˜ O ( Ndρ 2 ) parameters as m = ˜ O ( Nρ 2 ) . f 2 consists of ˜ O ( Nm 2 ) = ˜ O ( N 3 ρ 4 ) parameters. Since the case V assumes N &lt; d and large ρ regime deals with ρ ≥ 1 5 √ d , we have

<!-- formula-not-decoded -->

Therefore, f in total consists of ˜ O ( N 3 ρ 4 + Ndρ 2 ) = ˜ O ( Nd 2 ρ 4 ) number of parameters. Moreover, since f has the same width as f 2 and depth one larger than the depth of f 2 , it follows that width of f is O ( m ) = ˜ O ( ρ 2 N ) = ˜ O ( ρ 2 d ) and the depth of f is ˜ O ( N ) . This proves the theorem for the last case.

Bounding the Bit Complexity. Now, let us analyze how we can implement the above network under a finite precision. We have demonstrated that for every five cases, the depth ˜ O ( N ) and width ρ 2 d suffice for constructing f that robustly memorizes D .

Let R := max i ∈ [ N ] ∥ x i ∥ 2 + µ . Let D = ˜ O ( ρ 2 d ) and L = ˜ O ( N ) denote the width and the depth of the constructed network. Let M be the maximum absolute value of the parameter used for constructing f . Finally let ν = 0 . 1 . By Lemma B.23, there exists ¯ f with ˜ O ( N ) bit complexity, that approximates f uniformly over B 2 ( 0 , R ) with error at most ν , where ˜ O ( · ) here hides polylogarithmic terms in D,M,L and R . i.e.

<!-- formula-not-decoded -->

Finally, to handle the error ν , we use the floor function approximation from Lemma B.16. By Lemma B.16 with γ = 1 / 10 , there exists Floor : R → R with depth n := ⌈ log 2 ( C +1) ⌉ and width 5 such that Floor( x ) = ⌊ x ⌋ for all x ∈ [0 , C +1) with x -⌊ x ⌋ &gt; γ = 0 . 1 . Moreover, the lemma guarantees that Floor can be exactly implemented with O ( n + log 10) = O (log C ) = ˜ O (1) bit complexity.

Thus, if y ′ ∈ R satisfies | y ′ -y | ≤ ν = 0 . 1 for some y ∈ [ C ] , then

<!-- formula-not-decoded -->

In particular, ⌊ y ′ +0 . 5 ⌋ = y and ⌊ y ′ +0 . 5 ⌋ -( y ′ +0 . 5) ∈ (0 . 1 , 1) so that Floor( y ′ ) = ⌊ y ′ ⌋ = y . For x ∈ B 2 ( x i , µ ) , we have f ( x ) = y i so that the approximation ¯ f outputs y ′ = ¯ f ( x ) such that

<!-- formula-not-decoded -->

This shows Floor( ¯ f ( x )) = Floor( y ′ ) = y i . Moreover, Floor ◦ ¯ f can be implemented with parameters, width, and depth of the same scale as ¯ f , and bit complexity ˜ O ( N ) . This finishes the proof for the bit complexity.

## B.4 Lemmas for Lattice Mapping

Lemma B.15 (Avoiding Being Near Grid) . Let N,d ∈ N and x 1 , · · · , x N ∈ R d . Then, there exists a translation vector b ∈ R d such that:

<!-- formula-not-decoded -->

i.e., the translated points { x i -b } i ∈ [ N ] are coordinate-wise 1 2 N -far from the integer lattice.

Moreover, there exists ¯ b ∈ R d which has bit complexity ⌈ log(6 N ) ⌉ and satisfies

<!-- formula-not-decoded -->

Proof. For each coordinate j ∈ [ d ] , consider the set { x i,j } i ∈ [ N ] of all j -th coordinate values. For x ∈ R , let { x } := x -⌊ x ⌋ denote the fractional part of x . We consider the collection of fractional parts {{ x i,j }} i ∈ [ N ] . Without loss of generality, we may assume 0 ≤ { x 1 ,j } &lt; { x 2 ,j } &lt; · · · &lt; { x N,j } &lt; 1 .

For each j ∈ [ d ] , define the maximum fractional gap g j ∈ [0 , 1) as

<!-- formula-not-decoded -->

We claim:

<!-- formula-not-decoded -->

Suppose for a contradiction that g j &lt; 1 N for some j ∈ [ d ] . Then, we have by the definition of g j ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equations (47) and (48) lead to a contradiction, proving the claim.

Now, we define the translation of the j -th coordinate, b j ∈ R , based on the location where the maximum g j is attained. If the maximum in the definition of g j occurs by the difference of some consecutive pair ( { x i ′ ,j } , { x i ′ +1 ,j } ) satisfying { x i ′ +1 ,j } - { x i ′ ,j } = g j , we set

<!-- formula-not-decoded -->

In this case, dist( x i ′ ,j -b j , Z ) = dist( x i ′ +1 ,j -b j , Z ) = 1 2 g j ≥ 1 2 N . For other i , dist( x i,j -b j , Z ) is even larger by the order relation within {{ x i,j }} i ∈ [ N ] .

Otherwise, if the maximum in the definition of g j is attained as 1 -{ x N,j } + { x 1 ,j } = g j , we define

<!-- formula-not-decoded -->

In this case, dist( x 1 ,j -b j , Z ) = dist( x N,j -b j , Z ) = 1 2 g j ≥ 1 2 N . For other i , dist( x i,j -b j , Z ) is even larger by the order relation within {{ x i,j }} i ∈ [ N ] .

We define the full translation vector b = ( b 1 , . . . , b d ) ∈ R d . Then the translated points { x i -b } i ∈ [ N ] satisfy:

<!-- formula-not-decoded -->

Intuitively, b j is chosen as the midpoint of the widest gap between fractional values, ensuring that all fractional parts after the translation are at least g j 2 away from the nearest integer. Therefore, the translated points are coordinate-wise 1 2 N -far from lattice points.

We define ¯ b such that each of its coordinates is equal to the first ⌈ log(6 N ) ⌉ bits of the corresponding coordinate of b . Then, for all j ∈ [ d ] , we have | b j -¯ b j | ≤ 1 2 log(6 N ) ≤ 1 6 N . Using ¯ b with bit complexity ⌈ log(6 N ) ⌉ , we can still ensure the distance 1 3 N from the lattice points.

<!-- formula-not-decoded -->

The following lemma shows that we can approximate the floor function using a logarithmic number of ReLU units with respect to the length of the interval of interest.

Lemma B.16 (Floor Function Approximation) . For any n ∈ N and any γ ∈ (0 , 1) , there exists an n -layer network Floor n with width 5 and 5 n ReLU units such that

<!-- formula-not-decoded -->

Moreover, if γ = 1 t for some t ∈ N , then Floor n can be exactly implemented with 2 n +log 2 t bit complexity under a fixed point precision.

Proof. To reconcile the discontinuity of the floor function with the continuity of ReLU networks, we first define a discontinuous ideal building block that exactly replicates the floor function on the target interval [0 , 2 n ) . We then approximate this building block using a continuous neural network with ReLU activations.

The ideal building block ∆ is defined as:

<!-- formula-not-decoded -->

For n ∈ N , define the function Floor n by:

<!-- formula-not-decoded -->

We will show by induction that Floor n = ⌊ x ⌋ for all x ∈ [0 , 2 n ) .

For the base case n = 1 ,

<!-- formula-not-decoded -->

This proves the base case: for all x ∈ [0 , 2) , we have Floor 1 ( x ) = ⌊ x ⌋ .

For the inductive step, assume that Floor n ( x ) = ⌊ x ⌋ holds for all x ∈ [0 , 2 n ) . We aim to prove that Floor n +1 ( x ) = ⌊ x ⌋ for all x ∈ [0 , 2 n +1 ) .

Recall that:

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Therefore, by induction,

<!-- formula-not-decoded -->

Next, we define the σ approximation ∆ γ,n of the discontinuous block ∆ as:

<!-- formula-not-decoded -->

where γ n = γ 2 n . Check Figure 7 for an illustration of how ∆ γ,n looks like on [0 , 1] . It is straightforward to check that

<!-- formula-not-decoded -->

Figure 7: Plot of the ReLU-based approximation ∆ γ,n ( x ) of the ideal discontinuous building block ∆( x ) on [0 , 1] .

<!-- image -->

We now explain why this approximation remains valid under recursive composition up to depth n .

Let us define the variable x ′ := -x 2 n +1 , so that x = 2 n (1 -x ′ ) and x ′ ∈ (0 , 1] . Our target function is:

<!-- formula-not-decoded -->

We are given the assumption x - ⌊ x ⌋ &gt; γ , and we aim to express this in terms of x ′ to ensure ∆ γ,n n ( x ′ ) = ∆ n ( x ′ ) . We proceed step-by-step:

<!-- formula-not-decoded -->

Since x ′ ∈ (0 , 1] , we only need to consider k ∈ [2 n ] , i.e.,

<!-- formula-not-decoded -->

We will now prove by induction on n the following statement:

<!-- formula-not-decoded -->

For the base case n = 1 , by construction of ∆ γ, 1 ( x ) , we know ∆ γ, 1 ( x ) = ∆( x ) for all x ∈ [0 , 1 2 -γ 2 ] ∪ [ 1 2 , 1 -γ 2 ] , which contains the union ⋃ k ∈ [2] ( k -1 2 , k -γ 2 ) . Hence the base case holds.

For the inductive step, assume the claim holds for n . We show it holds for n +1 . By using γ/ 2 in place of γ for the inductive hypothesis, we have

<!-- formula-not-decoded -->

Let x ∈ ⋃ k ∈ [2 n +1 ] ( k -1 2 n +1 , k -γ 2 n +1 ) . We analyze two cases based on x ∈ [0 , 1 2 ) or x ∈ [ 1 2 , 1) .

First, consider the case x ∈ ⋃ k ∈ [2 n ] ( k -1 2 n +1 , k -γ 2 n +1 ) ⊂ [0 , 1 2 ) . Then x &lt; k -γ 2 n +1 ≤ 1 2 -γ n +1 , so ∆ γ,n +1 ( x ) = 2 x . Let y := 2 x . Then:

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the equation (a) follows by Equation (50).

Second, consider the case x ∈ ⋃ k ∈ [2 n +1 ] \ [2 n ] ( k -1 2 n +1 , k -γ 2 n +1 ) ⊂ [ 1 2 , 1) . Then 1 2 ≤ x &lt; k -γ 2 n +1 ≤ 1 -γ n +1 , so ∆ γ,n +1 ( x ) = 2 x -1 . Let y := 2 x -1 . Then:

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

where the equation (a) follows by Equation (50).

Therefore, by induction, we have shown that for any γ ∈ (0 , 1) and any n ∈ N ,

<!-- formula-not-decoded -->

We now define the ReLU-based floor approximation by

<!-- formula-not-decoded -->

Recall that the ideal target function is given by

<!-- formula-not-decoded -->

Let us denote x ′ := -x 2 n +1 . When x -⌊ x ⌋ &gt; γ , the value x ′ satisfies

<!-- formula-not-decoded -->

so that ∆ γ,n n ( x ′ ) = ∆ n ( x ′ ) by the result above. Therefore, we conclude:

<!-- formula-not-decoded -->

Finally to prove the additional statement regarding the bit complexity, consider the case γ = 1 t for some t ∈ N . By Equation (51), the bit complexity to implement Floor n is upper bounded by n plus the bit complexity to implement ∆ γ,n . Now, it suffices to consider the bit complexity required to implement ∆ γ,n . From Equation (49), observe that 1 /γ n = 2 n /γ = 2 n × t for γ = 1 /t . Since t ∈ N , this can be exactly implemented with log(2 n × t ) = n +log 2 t bit complexity. Thus, Floor n can be implemented exactly with 2 n +log 2 t bit complexity.

## B.5 Dimension Reduction via Careful Analysis of the Johnson-Lindenstrauss Lemma

We begin with a lemma that states a concentration of the length of the projection.

Lemma B.17 (Lemma 15.2.2, Matousek [2013]) . For a unit vector x ∈ S d -1 , let

<!-- formula-not-decoded -->

be the mapping of x onto the subspace spanned by the first m coordinates. Consider x ∈ S d -1 chosen uniformly at random. Then, there exists β such that ∥ ϕ ( x ) ∥ 2 is sharply concentrated around β ,

<!-- formula-not-decoded -->

where for m ≥ 10 log d , we have β ≥ 1 2 √ m d .

Based on the above concentration inequality, we state the Johnson-Lindenstrauss lemma, in a version which reflects the benefit on the ratio of the norm preserved when the projecting dimension increases. The proof follows that of Theorem 15.2.1 in Matousek [2013] with a slight modification.

Lemma B.18 (Strengthened Version of the Johnson-Lindenstrauss Lemma) . For N ≥ 2 , let X ⊆ R d be an N point set. Then, for any α ∈ (0 , 1) and 24 α -2 log N ≤ m ≤ d , there exists a 1-Lipschitz linear mapping ϕ : R d → R m and β &gt; 0 such that

<!-- formula-not-decoded -->

for all x , x ′ ∈ X . Moreover, β ≥ 1 2 √ m d whenever m ≥ 10 log d .

̸

Proof. If x = x ′ , the inequality trivially holds for any ϕ . Hence, it suffices to find ϕ that satisfies Equation (52) for all x , x ′ ∈ X with x = x ′ . Consider a random m -dimensional subspace L , and ϕ be a projection onto L . For any fixed x = x ′ ∈ X , Lemma B.17 implies that ∥ ∥ ∥ ϕ ( x -x ′ ∥ x -x ′ ∥ 2 ) ∥ ∥ ∥ 2 is concentrated around some constant β . i.e.

̸

<!-- formula-not-decoded -->

where we use β ≥ 1 2 √ m d at (a), m ≥ 24 α -2 log N at (b), and N ≥ 2 at (c). Similarly,

<!-- formula-not-decoded -->

By linearity of ϕ , we have ϕ ( x -x ′ ) = ϕ ( x ) -ϕ ( x ′ ) . Taking the union bound over the two probability bounds above, the following event happens with probability at most 2 /N 2 :

<!-- formula-not-decoded -->

̸

Next, we take a union bound over all N ( N -1) 2 pairs x , x ′ ∈ X with x = x ′ . Then, the probability that Equation (53) happens for any x , x ′ ∈ X with x = x ′ is at most 2 N 2 × N ( N -1) 2 = 1 -1 N &lt; 1 . Hence, there exists a m -dimensional subspace L such that Equation (53) does not hold for any pair of x , x ′ ∈ X . In other words, there exists a m -dimensional subspace L such that

̸

<!-- formula-not-decoded -->

̸

for all x = x ′ . By Lemma B.17, β ≥ 1 2 √ m d whenever m ≥ 10 log d . This concludes the lemma.

Proposition B.19 (Lipschitz Projection with Separation) . For N ≥ 2 , let D = { ( x i , y i ) } N i =1 ∈ D d,N,C . For any α ∈ (0 , 1) and 24 α -2 log N ≤ m ≤ d , there exists 1-Lipschitz linear mapping ϕ : R d → R m and β &gt; 0 such that D ′ := { ( ϕ ( x i ) , y i ) } N i =1 ∈ D m,N,C satisfies

<!-- formula-not-decoded -->

In particular, D ′ ∈ D m,N,C whenever D ∈ D d,N,C . Moreover, β ≥ 1 2 √ m d whenever m ≥ 10 log d .

Proof. Let X = { x i } N i =1 . By Lemma B.18, there exists 1-Lipschitz linear mapping ϕ : R d → R m and β &gt; 0 such that

<!-- formula-not-decoded -->

for all i, j ∈ [ N ] .

The inequality ϵ D ′ ≥ (1 -α ) βϵ D follows from the inequality from Lemma B.18. In particular,

̸

<!-- formula-not-decoded -->

where we use Equation (54) at (a).

̸

We next show D ′ ∈ D m,N,C whenever D ∈ D d,N,C . To show this, we need to prove ϕ ( x i ) = ϕ ( x j ) for all i = j . Since 1 -α &gt; 0 and β &gt; 0 , we have ∥ ϕ ( x i ) -ϕ ( x j ) ∥ 2 ≥ (1 -α ) β ∥ x i -x j ∥ 2 &gt; 0 whenever x i = x j . Moreover, D ∈ D d,N,C indicates that x i = x j whenever i = j . All together, we have ϕ ( x i ) = ϕ ( x j ) for all i = j so that D ′ ∈ D m,N,C .

̸

̸

Lemma B.20 (Projection onto log -scale Dimension) . Let D ∈ D d,N,C . Then, there exist an integer m = ˜ O (log N ) and a 1-Lipschitz linear map ϕ : R d → R m such that the projected dataset D ′ = { ( ϕ ( x i ) , y i ) } i ∈ [ N ] ∈ D m,N,C satisfies the separation bound

<!-- formula-not-decoded -->

Proof. Let α = 1 / 6 and m := min { d, max {⌈ 24 α -2 log N ⌉ , ⌈ 10 log d ⌉}} , then m = ˜ O (log N ) . We construct the linear mapping into m dimension by dividing the cases into d &lt; 24 α -2 log N or d ≥ 24 α -2 log N .

For the case d &lt; 24 α -2 log N , we have d &lt; max {⌈ 24 α -2 log N ⌉ , ⌈ 10 log d ⌉} , and therefore m = d . We consider the identity map ϕ : R d → R d (= R m ) , which is 1-Lipschitz. We have D ′ := { ( ϕ ( x i ) , y i ) } i ∈ [ N ] = { ( x i , y i ) } i ∈ [ N ] = D , so that ϵ D ′ = ϵ D &gt; 5 12 ϵ D = 5 12 √ m d ϵ D .

Otherwise, for the case d ≥ 24 α -2 log N , we first observe that m ≤ d . Since 24 α -2 log N ≤ d , we have

<!-- formula-not-decoded -->

Additionally, as N ≥ 2 , we have d ≥ 24 α -2 log N ≥ 864 log 2 ≥ e 4 . By Lemma B.22, this implies 10 log d ≤ d and therefore

<!-- formula-not-decoded -->

By Equations (55) and (56), we have max {⌈ 24 α -2 log N ⌉ , ⌈ 10 log d ⌉} ≤ d . Thus, it follows m = max {⌈ 24 α -2 log N ⌉ , ⌈ 10 log d ⌉} ≤ d . By Proposition B.19 with α = 1 6 , there exists 1Lipschitz linear mapping ϕ : R d → R m and β &gt; 0 such that D ′ = { ( ϕ ( x i ) , y i ) } i ∈ [ N ] satisfies ϵ D ′ ≥ 5 6 βϵ D . Since m = max {⌈ 24 α -2 log N ⌉ , ⌈ 10 log d ⌉} ≥ 10 log d , the inequality β ≥ 1 2 √ m d is also satisfied by Proposition B.19. Therefore, ϵ D ′ ≥ 5 6 βϵ D ≥ 5 12 √ m d ϵ D .

In both cases, we have 1-Lipschitz linear map ϕ such that D ′ = { ( ϕ ( x i ) , y i ) } i ∈ [ N ] has separation

<!-- formula-not-decoded -->

Proposition B.21 (Natural Projection of High Dimensional Data) . For d ≥ N , let D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N,C . Then, there exists 1-Lipschitz linear mapping φ : R d → R N such that D ′ = { ( φ ( x i ) , y i ) } i ∈ [ N ] ∈ D N,N,C satisfies

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

Proof. Consider the tall matrix X ∈ R d × N defined as

<!-- formula-not-decoded -->

Then dimCol( X ) ≤ N ≤ d . Take any subspace V such that Col( X ) ⊆ V ⊆ R d and dim V = N , and let B = { v 1 , · · · , v N } be an orthonormal basis of V . Let V ∈ R d × N be the matrix whose columns consist of vectors in B :

<!-- formula-not-decoded -->

Define φ : R d → R N as φ ( x ) = V ⊤ x . We first verify that φ is 1-Lipschitz. For any x ∈ R d , let x = x V + x V ⊥ where x V ∈ V and x V ⊥ ∈ V ⊥ . Then, x V = V z for some z ∈ R N , as x V ∈ Col( V ) . Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) is by Equation (58), (b) is because ∥ x V ∥ 2 2 = ∥ ∥ ∥ ∑ i ∈ [ N ] z i v i ∥ ∥ ∥ 2 2 = ∑ i ∈ [ N ] z 2 i = ∥ z ∥ 2 2 , and (c) is because ∥ x ∥ 2 2 = ∥ x V ∥ 2 2 + ∥ x V ⊥ ∥ 2 2 . Moreover, whenever x ∈ V , then the equality holds for (c) of Equation (59). Therefore, ∥ ∥ V ⊤ x ∥ ∥ 2 = ∥ x ∥ 2 for all x ∈ V .

Since φ is linear

Therefore, we have

<!-- formula-not-decoded -->

where (a) is by Equation (59). This shows that φ is 1-Lipschitz.

Next, for i, j ∈ [ N ] , we have

<!-- formula-not-decoded -->

where the last equality holds because x i -x j ∈ Col( X ) ⊆ V .

This shows that

̸

̸

<!-- formula-not-decoded -->

This shows that D ′ also has the desired property.

Lemma B.22. For t ≥ e 4 , we have t ≥ 10 log t .

Proof. Define u ( t ) := t -10 log t on the domain (0 , ∞ ) . Then, for all t &gt; 10 ,

<!-- formula-not-decoded -->

so that u is an increasing function on (10 , ∞ ) . In particular,

<!-- formula-not-decoded -->

This concludes that u ( t ) ≥ 0 for all t ≥ e 4 , or equivalently, t ≥ 10 log t for all t ≥ e 4 .

## B.6 Lemmas for Bit Complexity

The following lemma bounds how much bit complexity is sufficient for implementing the parameters of the neural network in order to obtain the required precision of the output. Note that we do not require the network to output scala values. i.e. the following lemma also applies to neural networks that output vectors.

Lemma B.23. Let f be a neural network of P parameters, depth L and width D in which the parameters have infinite precision. Let R ≥ 1 be the radius of the domain in which we want to approximate f . If all the parameters of f are bounded by some M ≥ 1 , then for any 0 &lt; ν &lt; 1 , there exists ¯ f , which is implemented with P parameters, depth L , width D , and ˜ O ( L ) bit complexity such that

<!-- formula-not-decoded -->

where ˜ O ( · ) hides a polylogarithmic dependency on D,M,L,R and 1 /ν .

Proof. Let f : R d → R be the neural network defined as

<!-- formula-not-decoded -->

where W ℓ ∈ R d ℓ × d ℓ -1 , b ℓ ∈ R d ℓ with d ℓ ≤ D for all ℓ ∈ [ L ] . Although a ℓ depends on x for all ℓ = 0 , · · · , L -1 , we omit x in the notation. Note that d 0 = d . Given that every elements of W ℓ and b ℓ are bounded by M , for any 0 &lt; ζ ≤ M , there exists ¯ W ℓ and ¯ b ℓ that can be implemented with ⌈ log 2 ( M/ζ ) ⌉ bit complexity in which

<!-- formula-not-decoded -->

Using the approximated parameters ¯ W ℓ and ¯ b ℓ , we recursively define ¯ f : R d → R , the finite-precision approximation of f .

<!-- formula-not-decoded -->

Similarly, although ¯ a ℓ depends on x for all ℓ = 0 , · · · , L -1 , we omit x in the notation.

Let us denote the difference of parameters as ∆ W ℓ := ¯ W ℓ -W ℓ , ∆ b ℓ := ¯ b ℓ -b ℓ for ℓ ∈ [ L ] and the difference of layer outputs ∆ a ℓ := ¯ a ℓ -a ℓ for ℓ ∈ [ L -1] . It is straightforward to check

<!-- formula-not-decoded -->

where the norm ∥·∥ and ∥·∥ F for the matrix denote the spectral norm and the Frobenius norm, respectively.

We first claim that there exists a degree 2 L +1 polynomial S on D,M,L and R such that ∥ a ℓ ∥ 2 ≤ S for all ℓ ∈ [ L -1] .

<!-- formula-not-decoded -->

Thus for all ℓ ∈ [ L -1] ,

<!-- formula-not-decoded -->

This proves the first claim. Moreover, S is composed of two monomials whose coefficients are all 1. We next claim that the error ∥ ∆ a L -1 ∥ ≤ Qζ for some degree 4 L +1 polynomial Q on D,M,L and R . Consider the following recurrence

<!-- formula-not-decoded -->

Thus noting that ∆ a 0 = x -x = 0 we have,

<!-- formula-not-decoded -->

where the last inequality follows from ζ ≤ M . Let Q := DL ( S + 1)(2 DM ) L -1 . Since S is a degree 2 L +1 polynomial on D,M,L and R , it follows that Q is a degree 4 L +1 polynomial on D,M,L and R . This proves the second claim. Moreover, Q is composed of three monomials whose coefficients are at most 2 L .

Thus,

<!-- formula-not-decoded -->

where we use ζ ≤ M in the last inequality. Now, by letting ζ := ν DS +2 DMQ + D , it follows that

<!-- formula-not-decoded -->

for all x with ∥ x ∥ 2 ≤ R . Thus, it suffices to have log 2 ( M/ζ ) = log 2 (( DS +2 DMQ + D ) M/ν ) bit complexity to attain an approximation of accuracy ν uniformly over the bounded domain with radius R . ( DS +2 DMQ + D ) M is a degree 4 L +4 polynomial on D,M,L and R . Moreover, it is composed of 2 + 3 + 1 = 6 monomials, whose coefficients are at most 2 L +1 . Hence, it follows that

<!-- formula-not-decoded -->

bit complexity suffices.

## C Extensions to ℓ p -norm

In this section, we extend the previous results on ℓ 2 -norm to arbitrary p -norm, where p ∈ [1 , ∞ ] .

In the following, we use dist p ( · , · ) to denote the ℓ p -norm distance between two points, a point and a set, or two sets. For the case d = 1 , we omit the notation p since every ℓ p -norm in 1-dimension denotes the absolute value.

We denote B p ( x , µ ) = { x ′ ∈ R d ∣ ∣ ∥ x ′ -x ∥ p &lt; µ } an open ℓ p -ball centered at x with a radius µ . Definition C.1. For D ∈ D d,N,C , the separation constant ϵ D ,p under ℓ p -norm is defined as

̸

<!-- formula-not-decoded -->

̸

As we consider D with x i = x j for all i = j , we have ϵ D ,p &gt; 0 . Next, we define robust memorization under ℓ p -norm.

̸

Definition C.2. For D ∈ D d,N,C , p ∈ [1 , ∞ ] , and a given robustness ratio ρ ∈ (0 , 1) , define the robustness radius as µ = ρϵ D ,p . We say that a function f : R d → R ρ -robustly memorizes D under the ℓ p -norm if

<!-- formula-not-decoded -->

and B p ( x i , µ ) is referred as the robustness ball of x i .

Similarly, we extend the notion of ρ -robust memorization error to ℓ p -norm.

Definition C.3. Let D ∈ D d,N,C be a class(or point)-separated dataset. The ρ -robust error of a network f : R d → R on D under the ℓ p -norm is defined as

̸

<!-- formula-not-decoded -->

The following inclusion between p -norm balls with different p -values is well known.

Lemma C.4 (Inclusion Between Balls) . Let 0 &lt; p &lt; q ≤ ∞ . Then, for any x ∈ R d and µ &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any p ∈ [1 , ∞ ] , let us denote or equivalently,

<!-- formula-not-decoded -->

throughout this section. For 0 &lt; p &lt; q ≤ ∞ , we have

<!-- formula-not-decoded -->

since ∥ x ∥ q ≤ ∥ x ∥ p ≤ d 1 p -1 q ∥ x ∥ q . In particular, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.1 Extension of Necessity Condition to ℓ p -norm

Theorem C.5. Let ρ ∈ (0 , 1) . Suppose for any D ∈ D d,N, 2 , there exists a neural network f ∈ F d,P that can ρ -robustly memorize D under ℓ p -norm. Then, the number of parameters P must satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. This follows by combining Proposition C.6 and Proposition C.8.

Proposition C.6. There exists D ∈ D d,N, 2 such that any neural network f : R d → R that ρ -robustly memorizes D under ℓ p -norm must have the first hidden layer width at least

- ρ 2 min { N -1 , d } if p ≥ 2 .

<!-- formula-not-decoded -->

Proof. We take D the same dataset as in Proposition 3.2. Recall that in the proof of Proposition 3.2, we take the dataset D = { e j , 2 } j ∈ [ N -1] ∪ { 0 , 1 } when N ≤ d + 1 , with additional data points (2 e 1 , 2) , (3 e 1 , 2) , · · · , (( N -d ) e 1 , 2) when N &gt; d + 1 . This has a separation ϵ D ,p = 1 2 under ℓ p -norm for all p ≥ 1 , on the both case N ≤ d +1 and N &gt; d +1 . Let f be a neural network that robustly memorizes D under ℓ p -norm. Since ϵ D ,p = ϵ D , 2 , the robustness radius µ under ℓ 2 -norm satisfies µ = ρϵ D ,p = ρϵ D , 2 . With this in mind, we now prove the proposition. The statement of the proposition consists of two parts, p ≥ 2 and 1 ≤ p &lt; 2 .

Part I: p ≥ 2 . First, we prove the result under p ≥ 2 Robust memorization under ℓ p -norm implies

<!-- formula-not-decoded -->

where µ = ρϵ D ,p = ρϵ D , 2 . For p ≥ 2 , we have B 2 ( x i , µ ) ⊆ B p ( x i , µ ) by Lemma C.4. Thus,

<!-- formula-not-decoded -->

Since µ = ρϵ D , 2 this implies that f ρ -robustly memorize D under ℓ 2 -norm. By Proposition 3.2, f should have the first hidden layer width at least ρ 2 min { N -1 , d } .

Part II: 1 ≤ p &lt; 2 . Next, we prove the result under 1 ≤ p &lt; 2 . Robust memorization under ℓ p -norm implies

<!-- formula-not-decoded -->

where µ = ρϵ D ,p = ρϵ D , 2 . For 1 ≤ p &lt; 2 , we have B 2 ( x , d 1 2 -1 p µ ) ⊆ B p ( x i , µ ) by applying p = p and q = 2 to Lemma C.4. Since γ p ( d ) = d 1 p -1 2 , we have B 2 ( x i , µ/γ p ( d )) ⊆ B p ( x i , µ ) . In particular, f memorize every µ/γ p ( d ) neighbor around the data point under ℓ 2 -norm. Let

<!-- formula-not-decoded -->

Then, f memorize every µ/γ p ( d ) = ρ ′ ϵ D , 2 radius neighbor around each data point under ℓ 2 -norm. In other words, f ρ ′ -robustly memorize D under ℓ 2 -norm. By Proposition 3.2, f should have the first hidden layer width at least ( ρ ′ ) 2 min { N -1 , d } . Putting back ρ ′ = ρ γ p ( d ) concludes the desired statement.

Proposition C.7. There exists a point separated D ∈ D d,N, 2 such that any neural network that ρ -robustly memorizes D under ℓ ∞ -norm must have the first hidden layer width at least

- ρ 2 min { d, N -1 } if ρ ∈ (0 , 1 2 ] .
- min { d, N -1 } if ρ ∈ ( 1 2 , 1) .

Proof. The first bullet is an immediate corollary of Proposition C.6, so we focus on the second bullet for ρ ∈ (1 / 2 , 1) . To prove the second bullet, we consider two cases based on the relationship between N -1 and d . In the first case, where N -1 ≤ d , establishing the proposition requires that the first hidden layer has width at least N -1 . In the second case, where N -1 &gt; d , the required width is at least d . For each case, we construct a dataset D ∈ D d,N, 2 such that any network that ρ -robustly memorizes D must have a first hidden layer of width no smaller than the corresponding bound.

Case I : N -1 ≤ d . Let D = { ( e j , 2) } j ∈ [ N -1] ∪ { ( 0 , 1) } . Then, D has a separation constant ϵ D , ∞ = 1 / 2 under ℓ ∞ -norm. Let f be a ρ -robust memorizer of D under ℓ ∞ -norm whose first hidden layer width is m . Let W ∈ R m × d denote the first hidden weight matrix. Suppose for a contradiction, m&lt;N -1 .

Let µ = ρϵ D , ∞ denote the robustness radius. Then, f has to distinguish every point in each B µ ( e j ) from every point in B µ ( 0 ) for all j ∈ [ N -1] . Therefore, for x ∈ B ∞ ( e j , µ ) and x ′ ∈ B ∞ ( 0 , µ ) , we have

̸

<!-- formula-not-decoded -->

or equivalently, x -x ′ / ∈ Null( W ) . Moreover

<!-- formula-not-decoded -->

Hence, it is necessary to have B ∞ ( e j , 2 µ ) ∩ Null( W ) = ∅ for all j ∈ [ N -1] , or equivalently,

<!-- formula-not-decoded -->

for all j ∈ [ N -1] .

Since dimCol( W ⊤ ) ≤ dim R m = m , we have dimNull( W ) ≥ d -m . Using Lemma C.10, we can upper bounds the maximum possible distance between { e j } j ∈ [ N -1] ⊆ R d and arbitrary subspace of a fixed dimension.

Take Z ⊆ Null( W ) such that dim Z = d -m and substitute d = d , t = N -1 , k = d -m and Z = Z into Lemma C.10. The assumptions t ≤ d for the lemma are satisfied since N -1 ≤ d . The additional assumption k ≥ d -t +1 is equivalent to d -m ≥ d -( N -1) + 1 and is satisfied since m&lt;N -1 . Therefore, we have

<!-- formula-not-decoded -->

By combining the above inequality with Equation (63),

<!-- formula-not-decoded -->

where (a) is due to Z ⊆ Null( W ) . Since ϵ D , ∞ = 1 / 2 , we have 2 µ = 2 ρϵ D , ∞ = ρ so that Equation (64) becomes ρ ≤ 1 / 2 . This contradicts our assumption ρ ∈ (1 / 2 , 1) , and therefore the width requirement m ≥ N -1 is necessary. This concludes the proof for the case N -1 ≤ d .

Case II : N -1 &gt; d . We construct the first d + 1 data points in the same manner as in Case I, using the construction for N = d + 1 . For the remaining N -d -1 data points, we set them sufficiently distant from the first d +1 data points to keep ϵ D , ∞ = 1 / 2 . In particular, we can set x d +2 = 2 e 1 , x d +3 = 3 e 1 , · · · , x N = ( N -d ) e 1 and y d +2 = y d +3 = · · · = y N = 2 . Compared to the case N = d +1 , we have ϵ D , ∞ unchanged while having more data points to memorize. By the necessity for the case N = d +1 , this dataset also requires the first hidden layer width at least ( d +1) -1 = d . This concludes the statement for the case N -1 &gt; d .

Combining the result of the two cases N -1 ≤ d and N -1 &gt; d concludes the proof of the theorem.

Proposition C.8. For p ∈ [1 , ∞ ) , let ρ ∈ ( 0 , ( 1 -1 d ) 1 /p ] . Suppose for any D ∈ D d,N, 2 there exists f ∈ F d,P that ρ -robustly memorizes D under ℓ p -norm. Then, the number of parameters P must satisfy P = Ω( √ N 1 -ρ p ) .

Proof. The main idea of the proof is the same as Proposition 3.3. We construct ⌊ N 2 ⌋ × ⌊ 1 1 -ρ p ⌋ number of data points that can be shattered by F d,P . This proves VC -dim( F d,P ) ≥ ⌊ N 2 ⌋×⌊ 1 1 -ρ p ⌋ = Ω( N/ (1 -ρ p )) . Since VC -dim( F d,P ) = O ( P 2 ) , this proves P = Ω( √ N/ (1 -ρ p )) .

For simplicity of the notation, let us denote k := ⌊ 1 1 -ρ p ⌋ . To prove the lower bound on the VCdimension, we construct k ×⌊ N 2 ⌋ points in R d that can be shattered by F d,P . As in the proof of Proposition 3.3, we define ⌊ N 2 ⌋ × k number of points as ⌊ N 2 ⌋ groups, where each group consists of k points.

We start by constructing the first group. Since ρ ∈ (0 , ( d -1 d ) 1 /p ] , we have k = ⌊ 1 1 -ρ p ⌋ ∈ [1 , d ] . The first group X 1 := { e j } k j =1 ⊆ R d is defined as the set of the first k vectors in the standard basis of R d . The remaining ⌊ N 2 ⌋ -1 groups are simply constructed as a translation of X 1 . In particular, for l ∈ [ ⌊ N 2 ⌋ ] , we define

<!-- formula-not-decoded -->

where c l := 2 d 2 ( l -1) × e 1 ensures that each group is sufficiently far from one another. Note that c 1 = 0 ensures X 1 also satisfies the consistency of the notation. Now, define X = ∪ l ∈ [ ⌊ N/ 2 ⌋ ] X l , the union of all ⌊ N 2 ⌋ groups which consists of k ×⌊ N 2 ⌋ points.

We claim that if for any D ∈ D d,N, 2 , there exists f ∈ F d,P that ρ -robustly memorizes D under ℓ p -norm, then X is shattered by F d,P . To prove the claim, suppose we are given arbitrary label Y = { y l,j } l ∈ [ ⌊ N/ 2 ⌋ ] ,j ∈ [ d ] of X , where y l,j ∈ {± 1 } denotes the label for x l,j := c l + e j ∈ X . Given the label Y , we construct D ∈ D d,N, 2 such that whenever f ∈ F d,P ρ -robustly memorize D under ℓ p -norm, then its affine translation f ′ = 2 f -3 ∈ F d,P satisfies f ′ ( x l,j ) = y l,j for all x l,j ∈ X .

<!-- formula-not-decoded -->

Furthermore, define y 2 l -1 = 2 , y 2 l = 1 and let D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N, 2 . To consider the separation ϵ D , 2 , notice that

<!-- formula-not-decoded -->

̸

where (a) is due to J + l ∩ J -l = ∅ and J + l ∪ J -l = [ k ] . For l = l ′ ,

<!-- formula-not-decoded -->

where (a) is by the triangle inequality under ℓ p -norm (namely, the Minkowski inequality), (b) uses d p ( c l , x 2 l -1 ) = d p ( c l ′ , x 2 l ′ ) = k 1 /p , (c),(e) is by k ≤ d , and (d) holds for all d ≥ 2 and p ≥ 1 . Thus, we have ϵ D ,p ≥ k 1 /p .

Take f ∈ F d,P that ρ -robustly memorize D . We first lower bound the robustness radius µ . Since t ϕ ↦→ p √ t -1 t is an strictly increasing function from t ≥ 1 onto [0 , 1) 3 , it has a well defined inverse mapping ϕ -1 : [0 , 1) → [1 , ∞ ) defined as ϕ -1 ( ρ ) = 1 1 -ρ p . Therefore,

<!-- formula-not-decoded -->

3 ϕ is a composition of two strictly increasing one-to-one corresponding functions t ↦→ t -1 t from [1 , ∞ ) onto [0 , 1) and u ↦→ p √ u from [0 , 1) onto [0 , 1)

Since ϵ D ,p ≥ k 1 /p and ρ ≥ ( k -1 k ) 1 /p , we have µ = ρϵ D ,p ≥ ρk 1 /p ≥ ( k -1) 1 /p . Thus, every f that ρ -robustly memorizes D must also memorize ( k -1) 1 /p radius open ℓ p -ball around each point in D as the same label as the data point.

Moreover, for x l,j ∈ X with positive label y l,j = +1 , we have

̸

<!-- formula-not-decoded -->

Take a sequence of points { z n } n ∈ N such that z n → x l,j as n →∞ 4 and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

satisfies such properties. Then, we have f ( z n ) = f ( x 2 l -1 ) = 2 for all n ∈ N . Moreover, by the continuity of f (under the usual topology),

<!-- formula-not-decoded -->

Similarly, for x l,j with negative label y l,j = -1 , we have ∥ x l,j -x 2 l ∥ p = ( k -1) 1 /p , so that f ( x l,j ) = 1 .

Since we can adjust the weight and the bias of the last hidden layer, F d,P is closed under affine transformation; that is, af + b ∈ F d,P whenever f ∈ F d,P . In particular, f ′ := 2 f -3 ∈ F d,P . This f ′ satisfies f ′ ( x l,j ) = 2 f ( x l,j ) -3 = 2 · 2 -3 = +1 whenever y l,j = +1 and f ′ ( x l,j ) = 2 f ( x l,j ) -3 = 2 · 1 -3 = -1 whenever y l,j = -1 . Thus, sign ◦ f ′ perfectly classify X with the label Y . Since we can take such f ′ ∈ F d,P given an arbitrary label Y of X , it follows that F d,P shatters X , concluding the proof of the theorem.

## C.1.1 Lemmas for Appendix C.1

Lemma C.9. Let { e j } j ∈ [ d ] ⊆ R d denote the standard basis in R d . Then, for any k -dimensional subspace Z of R d with k ≥ 1 we have,

<!-- formula-not-decoded -->

Proof. For any subspace Z ′ of Z , we have

<!-- formula-not-decoded -->

As every k -dimensional subspace of R d with k ≥ 1 has a one-dimensional subspace, it suffices to prove the second statement for k = 1 . i.e., for any one-dimensional subspace Z of R d ,

<!-- formula-not-decoded -->

4 We consider the convergence of the sequence on the usual topology induced by ℓ 2 -norm.

for all n ∈ N . In particular,

̸

Let Z = Span ( z ) , where z = ( z 1 , · · · , z d ) = 0 . Without loss of generality, let ∥ z ∥ ∞ = 1 and take j ∈ [ d ] such that | z j | = 1 . Let z ′ = z j 2 z ∈ Z . Then,

<!-- formula-not-decoded -->

where (a) is by | z j | = 1 , and (b) is by ∥ z ∥ ∞ = 1 . Therefore,

<!-- formula-not-decoded -->

concluding the statement.

The following lemma generalizes Lemma C.9 to the case where we consider only the distance to a subset of the standard basis, instead of the whole standard basis.

Lemma C.10. For 1 ≤ t ≤ d , let { e j } j ∈ [ t ] ⊆ R d denote the first t vectors from the standard basis in R d . Then, for any k -dimensional subspace Z of R d with k ≥ d -t +1 ,

<!-- formula-not-decoded -->

Proof. Similar to Lemma A.2, we start by considering the dimension of the intersection between Z and R t , both as a subspace of R d . Let Q = [ e 1 e 2 · · · e t ] ⊤ ∈ R t × d . Then,

<!-- formula-not-decoded -->

By considering the dimension,

<!-- formula-not-decoded -->

Under the assumption k ≥ d -t +1 , we have

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

where (b) is by Lemma C.9.

## C.2 Extension of Sufficiency Condition to ℓ p -norm

Theorem C.11. Let p ∈ [1 , ∞ ] . For any dataset D ∈ D d,N,C and η ∈ (0 , 1) , the following statements hold:

- (i) If ρ ∈ ( 0 , 1 5 N √ dγ p ( d ) ] , there exists f ∈ F d,P with P = ˜ O ( √ N ) that ρ -robustly memorizes D under ℓ p -norm.

- (ii) If ρ ∈ ( 1 5 N √ dγ p ( d ) , 1 5 √ dγ p ( d ) ] , there exists f ∈ F d,P with P = ˜ O ( Nd 1 4 ρ 1 2 γ p ( d ) 1 2 ) that ρ -robustly memorizes D under ℓ p -norm with error at most η .
- (iii) If ρ ∈ ( 1 5 √ dγ p ( d ) , 1 γ p ( d ) ) , there exists f ∈ F d,P with P = ˜ O ( Nd 2 ρ 4 γ p ( d ) 4 ) that ρ -robustly memorizes D under ℓ p -norm.

To prove Theorem C.11, we decompose it into three theorems (Theorems C.12 to C.14), each corresponding to one of the cases in the statement. They are following.

Theorem C.12. Let ρ ∈ ( 0 , 1 5 N √ dγ p ( d ) ] and p ∈ [1 , ∞ ] . For any dataset D ∈ D d,N,C , there exists f ∈ F d,P with P = ˜ O ( √ N ) that ρ -robustly memorizes D under ℓ p -norm.

Proof. Let ρ ′ = γ p ( d ) ρ . Then, we have ρ ′ ∈ ( 0 , 1 5 N √ d ] from the condition of ρ . By Theorem 4.2(i), there exists f ∈ F d,P with P = ˜ O ( √ N ) that ρ ′ -robustly memorizes D under ℓ 2 -norm. In other words, it holds f ( x ′ ) = y i , for all ( x i , y i ) ∈ D and x ′ ∈ B 2 ( x i , ρ ′ ϵ D , 2 ) .

We consider two cases depending on whether p ≥ 2 or p &lt; 2 , which affect the direction of inclusion between ℓ p and ℓ 2 balls.

Case I : p ≥ 2 . In this case, we have

<!-- formula-not-decoded -->

where (a) holds by Equation (61) and (b) holds by Lemma C.4 applying p = 2 and q = p .

′ ′

Thus, for all ( x i , y i ) ∈ D and x ∈ B p ( x i , ρϵ D ,p ) , it also holds f ( x ) = y i . In other words, f ρ -robustly memorizes D under ℓ p -norm with ˜ O ( √ N ) parameters.

Case II : p &lt; 2 . In this case, we have

<!-- formula-not-decoded -->

where (a) holds by Equation (62) and (b) holds by Lemma C.4 applying p = p and q = 2 .

Thus, for all ( x i , y i ) ∈ D and x ′ ∈ B p ( x i , ρϵ D ,p ) , it also holds f ( x ′ ) = y i . In other words, f ρ -robustly memorizes D under ℓ p -norm with ˜ O ( √ N ) parameters.

Theorem C.13. Let ρ ∈ ( 1 5 N √ dγ p ( d ) , 1 5 √ dγ p ( d ) ] and p ∈ [1 , ∞ ] . For any dataset D ∈ D d,N,C , there exists f ∈ F d,P with P = ˜ O ( Nd 1 4 ρ 1 2 γ p ( d ) 1 2 ) that ρ -robustly memorizes D under ℓ p -norm with error at most η .

Proof. Let ρ ′ = γ p ( d ) ρ . Then, we have ρ ′ ∈ ( 1 5 N √ d , 1 5 √ d ) from the condition of ρ .

We consider two cases depending on whether p ≥ 2 or p &lt; 2 , which affect the direction of inclusion between ℓ p and ℓ 2 balls.

Case I : p ≥ 2 . In this case, we have:

<!-- formula-not-decoded -->

where (a) holds by Equation (61) and (b) holds by Lemma C.4 applying p = 2 and q = p .

Case II : p &lt; 2 . In this case, we have:

<!-- formula-not-decoded -->

where (a) holds by Equation (62) and (b) holds by Lemma C.4 applying p = p and q = 2 . Thus, in both cases, it holds:

<!-- formula-not-decoded -->

We define η ′ = η Vol( B p ( x i ,ρϵ D ,p )) Vol( B 2 ( x i ,ρ ′ ϵ D , 2 ) . We apply Theorem 4.2(ii) with the robustness ratio ρ ′ and the error rate η ′ , then we obtain f ∈ F d,P with P = ˜ O ( Nd 1 4 ρ ′ 1 2 ) = ˜ O ( Nd 1 4 ρ 1 2 γ p ( d ) 1 2 ) that ρ ′ -robustly memorizes D with error at most η ′ under ℓ 2 -norm. In other words, for all ( x i , y i ) ∈ D , it holds that

̸

<!-- formula-not-decoded -->

̸

For simplicity, we denote E = { x ∈ R d | f ( x ′ ) = y i } . Then, we have

̸

<!-- formula-not-decoded -->

̸

where (a) holds by Equation (65), (b) holds by Equation (66), and (c) holds by the definition of η ′ . Thus, for all ( x i , y i ) ∈ D , it holds:

̸

<!-- formula-not-decoded -->

In other words, f ρ -robustly memorizes D under ℓ p -norm with error at most η and ˜ O ( Nd 1 4 ρ 1 2 γ p ( d ) 1 2 ) parameters.

Theorem C.14. Let ρ ∈ ( 1 5 √ dγ p ( d ) , 1 γ p ( d ) ) and p ∈ [1 , ∞ ] . For any dataset D ∈ D d,N,C , there exists f ∈ F d,P with P = ˜ O ( Nd 2 ρ 4 γ p ( d ) 4 ) that ρ -robustly memorizes D under ℓ p -norm.

Proof. Let ρ ′ = γ p ( d ) ρ . Then, we have ρ ′ ∈ ( 1 5 √ d , 1 ) from the condition of ρ . By Theorem 4.2(iii), there exists f ∈ F d,P with P = ˜ O ( Nd 2 ρ ′ 4 ) = ˜ O ( Nd 2 ρ 4 γ p ( d ) 4 ) that ρ ′ -robustly memorizes D under ℓ 2 -norm. In other words, it holds f ( x ′ ) = y i , for all ( x i , y i ) ∈ D and x ′ ∈ B 2 ( x i , ρ ′ ϵ D , 2 ) .

We consider two cases depending on whether p ≥ 2 or p &lt; 2 , which affect the direction of inclusion between ℓ p and ℓ 2 balls.

Case I : p ≥ 2 . In this case, we have:

<!-- formula-not-decoded -->

where (a) holds by Equation (61) and (b) holds by Lemma C.4 applying p = 2 and q = p .

Thus, for all ( x i , y i ) ∈ D and x ′ ∈ B p ( x i , ρϵ D ,p ) , it also holds f ( x ′ ) = y i . In other words, f ρ -robustly memorizes D under ℓ p -norm with ˜ O ( Nd 2 ρ 4 γ p ( d ) 4 ) parameters.

Case II : p &lt; 2 . In this case, we have:

<!-- formula-not-decoded -->

where (a) holds by Equation (62) and (b) holds by Lemma C.4 applying p = p and q = 2 .

Thus, for all ( x i , y i ) ∈ D and x ′ ∈ B p ( x i , ρϵ D ,p ) , it also holds f ( x ′ ) = y i . In other words, f ρ -robustly memorizes D under ℓ p -norm with ˜ O ( Nd 2 ρ 4 γ p ( d ) 4 ) parameters.

## D Comparison to Existing Bounds

## D.1 Summary of Parameter Complexity across ℓ p -norms

Table 1: Summary of our results and a comparison with prior works. We omit the constants for the range of ρ . γ p ( d ) = 1 under p = 2 reduces to the results in Sections 3 and 4.

|    | ℓ p -norm   | Robustness Ratio ρ                      | Bound on Parameters                                    |
|----|-------------|-----------------------------------------|--------------------------------------------------------|
| LB | p > 2       | (0 , 1)                                 | Ω ( min { N,d } dρ 2 ) , Proposition C.6               |
| LB | p ≤ 2       | (0 , 1)                                 | Ω ( min { N,d } d ( ρ/γ p ( d )) 2 ) , Proposition C.6 |
| LB | p = ∞       | (1/2, 1)                                | Ω(min { N,d } d ) , Proposition C.7                    |
| LB | p = ∞       | 0.8                                     | Ω ( d 2 ) , Yu et al. [2024] 1                         |
| LB | p < ∞       | ( 0 , (1 - 1 d ) 1 /p ]                 | Ω (√ N 1 - ρ p ) , Proposition C.8                     |
| LB | p = 2       | ρ → 1                                   | Ω ( √ Nd ) , Li et al. [2022] 2                        |
| UB | p = p       | (0 , 1 γ p ( d ) N √ d )                | ˜ O ( √ N ) , Theorem C.12                             |
| UB | p = p       | ( 1 γ p ( d ) N √ d , 1 γ p ( d ) √ d ) | ˜ O ( Nd 1 / 4 ( ργ p ( d )) 1 / 2 ) , Theorem C.13    |
| UB | p = p       | ( 0 , 1 γ p ( d ) √ d )                 | ˜ O ( N ) , Egosi et al. [2025]                        |
| UB | p = p       | ( 1 γ p ( d ) √ d , 1 γ p ( d ) )       | ˜ O ( Nd 2 ( ργ p ( d )) 2 ) , Theorem C.14            |
| UB | p = p       | ( 1 γ p ( d ) √ d , 1 γ p ( d ) )       | ˜ O ( Nd 3 ( ργ p ( d )) 6 ) , Egosi et al. [2025]     |
| UB | p = p       | (0 , 1)                                 | ˜ O ( Nd 2 p 2 ) , Yu et al. [2024] 3                  |

## D.2 Parameter Complexity of the Construction by Yu et al. [2024]

We now analyze the number of parameters of the network construction proposed by Yu et al. [2024], which provides the upper bound not depending on ρ , but still applies to all ρ ∈ (0 , 1) .

Lemma D.1 (Theorem B.6, Yu et al. [2024]) . Let p ∈ N . For any dataset D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N,C , let R &gt; 1 by any real value with ∥ x i ∥ ∞ ≤ R for all i ∈ [ N ] . For ρ ∈ (0 , 1) , define γ := (1 -ρ ) ϵ D ,p &gt; 0 . Then, there exists a network with width O ( d ) , and depth O ( Np (log( d γ p ) + p log R +log p )) that ρ -robustly memorize D under ℓ p -norm.

We note that in the Yu et al. [2024] uses the notation λ p D / 2 for ϵ D ,p , and the radius λ p D / 2 -γ in the original statement corresponds to the value µ := ρϵ D ,p in our notation. We count parameters in their construction in the following lemma, specifically in the case p = 2 . Although the original statement of Yu et al. [2024] includes a parameter count, they consider a different parameter counting strategy-by counting only the number of nonzero parameters. We therefore count the number of all parameters following Equation (3) in the subsequent lemma. Note that results and comparison under nonzero parameter counts are provided in Appendix E.

Lemma D.2. For any D ∈ D d,N,C and ρ ∈ (0 , 1) , define γ := (1 -ρ ) ϵ D &gt; 0 and R &gt; 1 with ∥ x i ∥ ∞ ≤ R for all i ∈ [ N ] . Then, there exists a neural network f such that ρ -robustly memorizes D using at most O ( Nd 2 (log( d γ 2 ) + log R )) parameters. Moreover, the network has width O ( d ) and depth ˜ O ( N ) .

Proof. By applying Lemma D.1 with p = 2 , we obtain a neural network f that ρ -robustly memorizes D with width O ( d ) , and depth L = O ( N (log( d γ 2 + log R ))) . In their construction, d l = Θ( d ) through all l , as the input x propagates over the layers using a width d . We count all parameters as defined in Equation (3), so we can upper bound the number of parameters used for the construction

of f as follows:

<!-- formula-not-decoded -->

## D.3 Parameter Complexity of the Construction by Egosi et al. [2025]

We observe that although Egosi et al. [2025] do not explicitly quantify the total number of parameters in their construction, it implicitly yields a network with O ( Nd 3 ρ 6 ) parameters. Specifically, we can establish the following:

For any D ∈ D d,N,C and ρ ∈ ( 1 √ d , 1) , there exists a neural network f that ρ -robustly memorizes D using ˜ O ( Nd 3 ρ 6 ) parameters.

This result follows from the network constructed in Theorem 4.4 of Egosi et al. [2025]. The proof of Theorem 4.4 proceeds under the assumption that for 7 ≤ k ≤ d +5 , and ρ ≤ 1 4 √ e √ k -6 d N -2 k -6 . Given this range, Theorem 4.2 of Egosi et al. [2025] is applied to construct a robust memorizer of the projected data from R d to R k . Figures 4 and 5 in their paper illustrate this construction. In this construction, the projected point propagates through the network Θ( Nk ) times. The width of the network scales with k , while the other component, that is not propagating the point remains constant in width. Thus, the number of parameters used for the construction is given by:

<!-- formula-not-decoded -->

To translate this to a bound in terms of ρ , we analyze the relationship between ρ and k . For k ≥ 4 log N +6 , we verify the following inequality:

<!-- formula-not-decoded -->

where (a) holds by N = e log N . Therefore, for ρ = 1 4 e √ k -6 d , the network ρ -robustly memorizes D with Θ( Nk 3 ) parameters. From the relationship between ρ and k , solving for k in terms of ρ yields k = Θ( dρ 2 ) . Since the minimum value of k under the assumption is 7 , the minimum achievable ρ is 1 4 e 1 √ d .

Thus, for ρ &gt; 1 √ d , the construction yields a network that ρ -robustly memorizes D with Θ( Nk 3 ) = Θ( Nd 3 ρ 6 ) parameters, as desired.

## E Nonzero Parameter Counts

While our main parameter counting method follows the approach of counting all parameters, including zeros, as defined in Equation (3), some prior works on memorization and robust memorization adopt a different parameter counting strategy-counting only the nonzero parameters. We emphasize that counting all parameters, including zeros, better aligns with how the matrices are stored in practice. Nevertheless, we also present how our results extend to the case of counting only nonzero parameters, offering an alternative perspective for interpreting our findings and comparing them with prior work.

In contrast to Equation (4), let us define the set of neural networks with input dimension d and at most P nonzero parameters by

<!-- formula-not-decoded -->

## E.1 Nonzero Parameter Counts: An illustration.

We provide the corresponding illustration of Figure 1 under only nonzero parameter counting in Figure 8, combining Theorem E.1 and Theorem E.2.

Figure 8: Summary of parameter bounds, counting only nonzero parameters on a log-log scale when d = Θ( √ N ) . We omit constant factors in both axes. Solid blue and red curves show the sufficient (Theorem E.2) and necessary (Theorem E.1) numbers of parameters, respectively; the solid black curve is the best prior bound. Light-blue shading highlights our improvement in the upper bound, and light-red shading highlights our improvement in the lower bound. The cross-hatched area marks the remaining gap.

<!-- image -->

## E.2 Nonzero Parameter Counts: Lower Bounds

The lower bound in Theorem 3.1 that counts all parameters consists of two terms: one based on the network width and another based on the VC-dimension. Although the lower bound by VC-dimension remains valid even when counting only nonzero parameters, the lower bound on the first hidden layer width can be translated into a lower bound on parameters only if we also include zero-valued parameters in the parameter counting convention. As a result, we obtain the following lower bound consisting of only the lower bound from the VC-dimension.

Theorem E.1. Let ρ ∈ (0 , 1) . Suppose for any D ∈ D d,N, 2 , there exists a neural network f ∈ ¯ F d,P that can ρ -robustly memorize D . Then, the number of parameters P must satisfy

<!-- formula-not-decoded -->

The main reason why the VC-dimension lower bound remains valid even for the nonzero parameter count is because the key relation VC -dim( ¯ F d,P ) = O ( P 2 ) [Goldberg and Jerrum, 1995] holds even for the ¯ F d,P instead of F d,P . Below, we provide an explicit proof of the Theorem E.1.

Proof. Since F d,P ⊆ ¯ F d,P , we have VC -dim( F d,P ) ≤ VC -dim( ¯ F d,P ) . In particular, by Equation (7), we have for ρ ∈ ( 0 , √ 1 -1 d ] that

<!-- formula-not-decoded -->

By [Goldberg and Jerrum, 1995], we have VC -dim( ¯ F d,P ) = O ( P 2 ) . Combining the two relations proves that for ρ ∈ ( 0 , √ 1 -1 d ] ,

<!-- formula-not-decoded -->

Since 1 √ 1 -ρ 2 ≤ √ d for ρ ∈ ( 0 , √ 1 -1 d ] , the following relation holds:

<!-- formula-not-decoded -->

For ρ ∈ (√ 1 -1 d , 1 ) , the lower bound P = Ω( √ Nd ) obtained by the case ρ = √ 1 -1 d also can be applied. Since 1 √ 1 -ρ 2 &gt; √ d for ρ ∈ (√ 1 -1 d , 1 ) , the following relation holds:

<!-- formula-not-decoded -->

As a result, applying P = Ω (√ N 1 -ρ 2 ) for ρ ∈ ( 0 , √ 1 -1 d ] and Ω( √ Nd ) for ρ ∈ (√ 1 -1 d , 1 ) results in

<!-- formula-not-decoded -->

## E.3 Nonzero Parameter Counts: Upper Bounds

While upper bounds on parameter counts of all parameters in Theorem 4.2 are naturally an upper bound for parameter counts of nonzero parameters, we provide a tighter upper bound regarding the nonzero parameters.

Theorem E.2. For any dataset D ∈ D d,N,C and η ∈ (0 , 1) , the following statements hold:

- (i) If ρ ∈ ( 0 , 1 5 N √ d ] , there exists f ∈ ¯ F d,P with P = ˜ O ( √ N + d ) that ρ -robustly memorizes D .
- (ii) If ρ ∈ ( 1 5 N √ d , 1 5 √ d ] , there exists f ∈ ¯ F d,P with P = ˜ O ( Nd 1 4 ρ 1 2 + d ) that ρ -robustly memorizes D with error at most η .
- (iii) If ρ ∈ ( 1 5 √ d , 1 ) , there exists f ∈ ¯ F d,P with P = ˜ O ( Ndρ 2 + d ) that ρ -robustly memorizes D .

In comparison to the total parameter count as in Theorem 4.2, only Theorem E.2(iii) have a modified rate from P = ˜ O ( Nd 2 ρ 4 ) to P = ˜ O ( Ndρ 2 ) . Below, we provide an explicit proof of Theorem E.2. The d term in the parameter bounds of all three cases comes from the upper bound on the parameters of the first hidden layer.

Proof. Upper bounds on all parameter counts are natural upper bounds on the nonzero parameter counts. Since Theorem E.2(i) and Theorem E.2(ii) claims the same rate as Theorem 4.2(i) and

Theorem 4.2(ii) respectively, they trivially follows from Theorem 4.2. Another way of speaking, F d,P ⊆ ¯ F d,P and the first two cases directly follow from Theorem 4.2.

Now let us prove Theorem E.2(iii). Here, we mainly follow the proof of Theorem B.14, where instead of counting every parameter using Lemma D.2, we count only the nonzero parameters using Lemma E.3. We divide the cases into five, following Theorem B.14 as in Figure 6.

Let D = { ( x i , y i ) } i ∈ [ N ] ∈ D d,N,C be given. We divide the proof into five cases, the first case under ρ ∈ [1 / 3 , 1) , the second case under ρ ∈ (1 / 5 √ d, 1 / 3) and d &lt; 600 log N , the third case under ρ ∈ (1 / 5 √ d, 1 / 3) and N &lt; 600 log N ≤ d , the fourth case under ρ ∈ (1 / 5 √ d, 1 / 3) , N ≥ d ≥ 600 log N , and finally the fifth case under ρ ∈ (1 / 5 √ d, 1 / 3) and d &gt; N ≥ 600 log N . To check that these cases cover all the cases, refer to Figure 6.

Case I: ρ ∈ [1 / 3 , 1) . Let us denote R := max i ∈ [ N ] ∥ x i ∥ 2 and γ := (1 -ρ ) ϵ D . Note that R ≥ ∥ x i ∥ ∞ for all i ∈ [ N ] as ∥ x ∥ 2 ≥ ∥ x ∥ ∞ for all x ∈ R d . By applying Lemma E.3, there exists f ∈ ¯ F d,P with P = O ( Nd (log( d γ 2 ) + log R )) nonzero parameters that ρ -robustly memorize D . The number of nonzero parameters can be further bounded as follows:

<!-- formula-not-decoded -->

where (a) is due to ρ = Ω(1) , (b) hides the logarithmic factors.

Case II: ρ ∈ (1 / 5 √ d, 1 / 3) and d &lt; 600 log N . Let us denote R := max i ∈ [ N ] ∥ x i ∥ 2 and γ := (1 -ρ ) ϵ D . Note that R ≥ ∥ x i ∥ ∞ for all i ∈ [ N ] as ∥ x ∥ 2 ≥ ∥ x ∥ ∞ for all x ∈ R d . By Lemma E.3, there exists f ∈ ¯ F d,P with P = O ( Nd (log( d γ 2 ) + log R )) nonzero parameters that ρ -robustly memorize D . The number of nonzero parameters can be further bounded as follows:

<!-- formula-not-decoded -->

where (a) is due to d ≤ 600 log N , (b) hides the logarithmic factors, and (c) is because N ≤ 25 Ndρ 2 for all ρ ∈ ( 1 5 √ d , 1 3 ) .

Case III: ρ ∈ (1 / 5 √ d, 1 / 3) and N &lt; 600 log N ≤ d . We first apply Proposition B.21 to D to obtain 1-Lipschitz linear φ : R d → R N such that D ′ := { ( φ ( x i ) , y i ) } i ∈ [ N ] has ϵ D ′ = ϵ D . This is possible as d ≥ N .

Take b ∈ R N such that φ ( x ) -b ≥ 0 for all x ∈ B 2 ( φ ( x i ) , ρϵ D ′ ) , ensuring that σ does not affect the output of the first hidden layer. Let D ′′ = { ( φ ( x i ) -b , y i ) } i ∈ [ N ] . Then, ϵ D = ϵ D ′ = ϵ D ′′ . For simplicity of the notation, let us denote z i := φ ( x i ) -b . Moreover, the first hidden layer is defined as f 1 ( x ) = φ ( x ) -b .

We apply Lemma E.3 to D ′′ . Let us denote R := max i ∈ [ N ] ∥ φ ( z i ) ∥ 2 and γ := (1 -ρ ) ϵ D ′′ . Note that R ≥ ∥ z i ∥ ∞ for all i ∈ [ N ] as ∥ z ∥ 2 ≥ ∥ z ∥ ∞ for all z ∈ R N . By Lemma E.3, there exists f 2 ∈ ¯ F N,P with P = O ( N · N (log( N γ 2 ) + log R )) nonzero parameters that ρ -robustly memorize D ′′ .

Let f = f 2 ◦ σ ◦ f 1 . Since f 1 is 1-Lipschitz and ϵ D ′′ = ϵ D , every robustness ball of D is mapped to the robustness ball of D ′′ via f 1 . Since the σ does not affect the first hidden layer output of the robustness ball, and f 2 ρ -robustly memorizes D ′′ , the composed f satisfies the desired property The number of nonzero parameters can be further bounded as follows:

<!-- formula-not-decoded -->

where (a) is due to N ≤ 600 log N , and (b) hides the logarithmic factors.

Case IV: ρ ∈ (1 / 5 √ d, 1 / 3) , and N ≥ d ≥ 600 log N . We utilize the dimension reduction technique by Proposition B.19. We apply Proposition B.19 to D with m =

max {⌈ 9 dρ 2 ⌉ , ⌈ 600 log N ⌉ , ⌈ 10 log d ⌉} and α = 1 / 5 . Let us first check that the specified m satisfies the condition 24 α -2 log N ≤ m ≤ d for the proposition to be applied. α = 1 / 5 and m ≥ 600 log N ensure the first inequality 24 α -2 log N ≤ m . The second inequality m ≤ d is decomposed into three parts. Since ρ ≤ 1 3 , we have 9 dρ 2 ≤ d so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Additionally, as N ≥ 2 , we have d ≥ 600 log N ≥ 600 log 2 ≥ 400 . By Lemma B.22, this implies 10 log d ≤ d and therefore

<!-- formula-not-decoded -->

Gathering Equations (68) to (70) proves m ≤ d .

By the Proposition B.19, there exists 1-Lipchitz linear mapping ϕ : R d → R m and β &gt; 0 such that D ′ := { ( ϕ ( x i ) , y i ) } i ∈ [ N ] ∈ D m,N,C satisfies

<!-- formula-not-decoded -->

As m ≥ 10 log d , the inequality β ≥ 1 2 √ m d is also satisfied by Proposition B.19. Therefore, we have

<!-- formula-not-decoded -->

where (a) is by the definition of m . Moreover, since ϕ is 1-Lipchitz linear,

<!-- formula-not-decoded -->

for all i ∈ [ N ] . Hence, by letting R := max i ∈ [ N ] {∥ x i ∥ 2 } , we have ∥ ϕ ( x i ) ∥ 2 ≤ R for all i ∈ [ N ] .

Now, we set the first layer hidden matrix as the matrix W ∈ R m × d corresponding to ϕ under the standard basis of R d and R m . Moreover, set the first hidden layer bias as b := 2 R 1 = 2 R (1 , 1 , · · · , 1) ∈ R m . Then, we have

<!-- formula-not-decoded -->

for all x ∈ B 2 ( x i , ϵ D ) for all i ∈ [ N ] , where the comparison between two vectors are element-wise. This is because for all i ∈ [ N ] , j ∈ [ m ] and x ∈ B 2 ( x , ϵ D ) , we have

<!-- formula-not-decoded -->

where (a) is by Equation (73), (b) is by the triangle inequality, and (c) is due to R &gt; ϵ D .

We construct the first layer of the neural network as f 1 ( x ) := σ ( Wx + b ) which includes the activation σ . Then, by above properties, D ′′ := { ( f 1 ( x i ) , y i ) } i ∈ [ N ] satisfies

<!-- formula-not-decoded -->

This is because for i = j with y i = y j we have

̸

̸

<!-- formula-not-decoded -->

Moreover, 600 log N ≤ d implies

Moreover, 600 log N ≤ N implies

<!-- formula-not-decoded -->

where (a) is by Equation (74), (b) is by the definition of the ϵ D ′ , (c) is by Equation (71), and (d) is by Equation (72). By Lemma E.3 applied to D ′′ ∈ D m,N,C , there exists f 2 ∈ F m,P with P = O ( Nm (log( m ( γ ′′ ) 2 ) + log R ′′ )) nonzero number of parameters that 5 6 -robustly memorize D ′′ , where

<!-- formula-not-decoded -->

Here (a) is by Equation (75).

Now, we claim that f := f 2 ◦ f 1 ρ -robustly memorize D . For any i ∈ [ N ] , take x ∈ B 2 ( x i , ρϵ D ) . Then, by Equation (74), we have f 1 ( x ) = Wx + b and f 1 ( x i ) = Wx i + b so that

<!-- formula-not-decoded -->

Moreover, combining Equations (75) and (76) results ∥ f 1 ( x ) -f 1 ( x i ) ∥ 2 ≤ 5 6 ϵ D ′′ . Since f 2 5 6 -robustly memorize D ′′ , we have

<!-- formula-not-decoded -->

In particular, f ( x ) = y i for any x ∈ B 2 ( x i , ρϵ D ) , concluding that f is a ρ -robust memorizer D . Regarding the number of parameters to construct f , notice that f 1 consists of ( d +1) m = ˜ O ( d 2 ρ 2 ) parameters (and thus ˜ O ( d 2 ρ 2 ) nonzero parameters) as m = ˜ O ( dρ 2 ) . f 2 consists of ˜ O ( Nm ) = ˜ O ( Ndρ 2 ) nonzero parameters. Since the case IV assumes N ≥ d , we have

<!-- formula-not-decoded -->

Therefore, f in total consists of ˜ O ( d 2 ρ 2 + Ndρ 2 ) = ˜ O ( Ndρ 2 ) number of nonzero parameters. This proves the theorem for the fourth case.

Case V: ρ ∈ (1 / 5 √ d, 1 / 3) , and d &gt; N ≥ 600 log N . The last case combines the two techniques used in Cases III and IV. We first apply Proposition B.21 to D to obtain 1-Lipschitz linear φ : R d → R N such that D ′ := { ( φ ( x i ) , y i ) } i ∈ [ N ] ∈ D N,N,C has ϵ D ′ = ϵ D . Note that we can apply the proposition since d ≥ N .

Next, we apply Proposition B.19 to D ′ ∈ D N,N,C with m = max {⌈ 9 Nρ 2 ⌉ , ⌈ 600 log N ⌉} and α = 1 / 5 . Let us first check that the specified m satisfies the condition 24 α -2 log N ≤ m ≤ N for the proposition to be applied. α = 1 / 5 and m ≥ 600 log N ensure the first inequality 24 α -2 log N ≤ m . The second inequality m ≤ N is decomposed into two parts. Since ρ ≤ 1 3 , we have 9 Nρ 2 ≤ N so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Gathering Equations (68) and (69) proves m ≤ N . Additionally, as N ≥ 2 , we have N ≥ 600 log N ≥ 600 log 2 ≥ 400 . By Lemma B.22, this implies 10 log N ≤ N .

By the Proposition B.19, there exists 1-Lipchitz linear mapping ϕ : R N → R m and β &gt; 0 such that D ′′ := { ( ϕ ( φ ( x i )) , y i ) } i ∈ [ N ] ∈ D m,N,C satisfies

<!-- formula-not-decoded -->

As m ≥ 600 log N ≥ 10 log N , the inequality β ≥ 1 2 √ m N is also satisfied by Proposition B.19. Therefore, we have

<!-- formula-not-decoded -->

where (a) is by the definition of m . Moreover, since φ and ϕ are both 1-Lipchitz linear, ϕ ◦ φ : R d → R m is also 1-Lipschitz linear. Therefore,

<!-- formula-not-decoded -->

for all i ∈ [ N ] . Hence, by letting R := max i ∈ [ N ] {∥ x i ∥ 2 } , we have ∥ ϕ ( φ ( x i )) ∥ 2 ≤ R for all i ∈ [ N ] .

Now, we set the first layer hidden matrix as the matrix W ∈ R m × d corresponding to ϕ ◦ φ under the standard basis of R d and R m . Moreover, set the first hidden layer bias as b := 2 R 1 = 2 R (1 , 1 , · · · , 1) ∈ R m . Then, we have

<!-- formula-not-decoded -->

for all x ∈ B 2 ( x i , ϵ D ) for all i ∈ [ N ] , where the comparison between two vectors are element-wise. This is because for all i ∈ [ N ] , j ∈ [ m ] and x ∈ B 2 ( x , ϵ D ) , we have

<!-- formula-not-decoded -->

where (a) is by Equation (81), (b) is by the triangle inequality, and (c) is due to R ≥ ϵ D .

We construct the first layer of the neural network as f 1 ( x ) := σ ( Wx + b ) which includes the activation σ . Next, we show that, D ′′ := { ( f 1 ( x i ) , y i ) } i ∈ [ N ] satisfies

<!-- formula-not-decoded -->

̸

by the above properties. This is because for i = j with y i = y j we have

̸

<!-- formula-not-decoded -->

where (a) is by Equation (82), (b) is by the definition of the ϵ D ′′ , (c) is by Equation (79), (d) is by Equation (80), and (e) is because ϵ D ′ = ϵ D .

By Lemma E.3 applied to D ′′ ∈ D m,N,C , there exists f 2 ∈ F m,P with P = O ( Nm (log( m ( γ ′′ ) 2 ) + log R ′′ )) nonzero number of parameters that 5 6 -robustly memorize D ′′ , where

<!-- formula-not-decoded -->

Here, (a) is by Equation (83).

Now, we claim that f := f 2 ◦ f 1 ρ -robustly memorize D . For any i ∈ [ N ] , take x ∈ B 2 ( x i , ρϵ D ) . Then, by Equation (82), we have f 1 ( x ) = Wx + b and f 1 ( x i ) = Wx i + b so that

<!-- formula-not-decoded -->

Moreover, putting Equation (83) to Equation (84) results ∥ f 1 ( x ) -f 1 ( x i ) ∥ 2 ≤ 5 6 ϵ D ′′ . Since f 2 5 6 -robustly memorize D ′′ , we have

<!-- formula-not-decoded -->

In particular, f ( x ) = y i for any x ∈ B 2 ( x i , ρϵ D ) , concluding that f is a ρ -robust memorizer D .

Regarding the number of nonzero parameters to construct f , notice that f 1 consists of ( d +1) m = ˜ O ( Ndρ 2 ) nonzero parameters as m = ˜ O ( Nρ 2 ) . f 2 consists of ˜ O ( Nm ) = ˜ O ( N 2 ρ 2 ) nonzero parameters. Since the case V assumes N &lt; d , we have

<!-- formula-not-decoded -->

Therefore, f in total consists of ˜ O ( Ndρ 2 + N 2 ρ 2 ) = ˜ O ( Ndρ 2 ) number of nonzero parameters. This proves the theorem for the last case.

Nonzero Parameter Counts: Existing Upper Bounds. In Section 1.1, the existing upper bound is stated by counting all parameters. When counting only the nonzero parameters, the corresponding existing upper bound takes a different form. Specifically, for any dataset D with input dimension d and size N , there exist a neural network that achieves robust memorization on D with the robustness ratio ρ under ℓ 2 -norm, with the number of parameters P bounded as follows:

<!-- formula-not-decoded -->

This is the counterpart to Equation (2) that considers all parameter counts. As in the case of full parameter count, the first and the third case in Equation (85) directly follow from Yu et al. [2024] and Egosi et al. [2025] respectively. The work by Egosi et al. [2025] can be implicitly improved to the second case under the moderate ρ condition, using the same translation technique provided in Appendix D.3.

## E.4 Lemmas for Nonzero Parameter Count

Here, we state Lemmas D.1 and D.2-that corresponds to Theorem B.6 of Yu et al. [2024]-to its original version that contains the nonzero parameter count with ℓ 2 -norm into the consideration.

Lemma E.3 (Theorem B.6, Yu et al. [2024]) . For any D ∈ D d,N,C and ρ ∈ (0 , 1) , define γ := (1 -ρ ) ϵ D &gt; 0 and R &gt; 1 with ∥ x i ∥ ∞ ≤ R for all i ∈ [ N ] . Then, there exists a neural network f with width O ( d ) , depth O ( N (log( d γ 2 ) + log R )) that ρ -robustly memorizes D using at most O ( Nd (log( d γ 2 ) + log R )) nonzero parameters.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We clarify the main claims in the abstract and introduction through the theorems in Sections 3 and 4. The abstract and introduction clearly state the claims made by the paper, with a comparison to the prior works.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of our result in Sections 3 to 5, as well as Section 6. Guidelines:

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

## Answer: [Yes]

Justification: We specify the basic settings and network architectures in Section 2. The complete proof of all the results is given in the supplementary material.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper is fully theoretical. We contain no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper is fully theoretical. We have no experiments, and therefore no data and code to reproduce any experimental results.

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

Answer: [NA]

Justification: The paper is fully theoretical. We contain no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper is fully theoretical. We contain no experiments.

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

Answer: [NA]

Justification: The paper is fully theoretical. We contain no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research fully complies with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is fully theoretical. The main focus is on the mathematical aspect without a direct relation to downstream applications.

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

Justification: The paper is fully theoretical and poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper is fully theoretical. We do not use existing code or dataset.

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

Justification: The paper is fully theoretical. We do not release new dataset/code/model.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper is fully theoretical. We do not involve any crowdsourcing nor human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper is fully theoretical. We do not involve any crowdsourcing nor human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We used LLM only for editing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
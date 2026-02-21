## Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification

## Yuanfan Li

School of Mathematical Sciences Zhejiang University 22335070@zju.edu.cn

## Zheng-Chu Guo

School of Mathematical Sciences Zhejiang University guozc@zju.edu.cn

## Yunwen Lei

Department of Mathematics The University of Hong Kong leiyw@hku.hk

## Yiming Ying ∗

School of mathematics and statistics Univerity of Sydney yiming.ying@sydney.edu.au

## Abstract

Recent advances have significantly improved our understanding of the generalization performance of gradient descent (GD) methods in deep neural networks. A natural and fundamental question is whether GD can achieve generalization rates comparable to the minimax optimal rates established in the kernel setting. Existing results either yield suboptimal rates of O (1 / √ n ) , or focus on networks with smooth activation functions, incurring exponential dependence on network depth L . In this work, we establish optimal generalization rates for GD with deep ReLU networks by carefully trading off optimization and generalization errors, achieving only polynomial dependence on depth. Specifically, under the assumption that the data are NTK separable from the margin γ , we prove an excess risk rate of ˜ O ( L 6 / ( nγ 2 )) , which aligns with the optimal SVM-type rate ˜ O (1 / ( nγ 2 )) up to depth-dependent factors. A key technical contribution is our novel control of activation patterns near a reference model, enabling a sharper Rademacher complexity bound for deep ReLU networks trained with gradient descent.

## 1 Introduction

Deep neural networks trained through first-order optimization methods have achieved remarkable empirical success in diverse domains [Krizhevsky et al., 2012]. Despite their widespread adoption, a rigorous theoretical understanding of their optimization dynamics and generalization behavior remains incomplete, particularly for ReLU networks. The inherent challenges arise from the nonsmoothness and non-convexity of the loss landscape induced by ReLU activations and network architectures, which complicate the classical analysis. Intriguingly, empirical evidence demonstrates that over-parameterized models often achieve zero training error but still generalize well even in the absence of explicit regularization [Zhang et al., 2016]. This phenomenon has spurred significant theoretical research to understand its underlying mechanisms.

A prominent line of research uses the neural tangent kernel (NTK) framework to analyze neural network training [Jacot et al., 2018]. In the infinite-width limit, gradient descent (GD) dynamics can be characterized by functions in the NTK's reproducing kernel Hilbert space (RKHS), with convergence guarantees established for both shallow and deep networks [Du et al., 2019b,a]. These results demonstrate that GD converges to global minima within a local neighborhood of initialization,

∗ Corresponding author

Table 1: Comparison of learning neural networks with GD on NTK separable data with prior works. Here m is the network width, L is the network depth, n is the sample size, γ is the NTK-margin, T is the number of iterations.

|                  | Activation   | Width              | Training error   | Generalization error      | Network   |
|------------------|--------------|--------------------|------------------|---------------------------|-----------|
| Ji and Telgarsky | ReLU         | ˜ Ω ( 1 γ 8 )      | ˜ O ( 1 γ 2 T )  | ˜ O ( 1 γ 2 √ n )         | Shallow   |
| Lei et al.       | ReLU         | ˜ Ω ( 1 γ 8 )      | ˜ O ( 1 γ 2 T )  | ˜ O ( 1 γ 2 n )           | Shallow   |
| Chen et al.      | ReLU         | ˜ Ω ( L 22 γ 8 )   | ˜ O ( 1 γ 2 T )  | ˜ O ( e O ( L ) γ √ m n ) | Deep      |
| Taheri et al.    | Smooth       | ˜ Ω ( 1 γ 6 L +4 ) | ˜ O ( 1 γ 2 T )  | ˜ O ( e O ( L ) γ 2 n )   | Deep      |
| Ours             | ReLU         | ˜ Ω ( L 16 γ 8 )   | ˜ O ( 1 γ 2 T )  | ˜ O ( L 6 γ 2 n )         | Deep      |

provided that the network width is sufficiently large. In particular, the appealing work [Arora et al., 2019a] showed that, if the network width m = ˜ O (poly( n, 1 /λ 0 , L )) , then the generalization error is of the order L √ y ⊤ ( H ∞ ) -1 y n , where H ∞ denotes the NTK gram matrix over the training data and λ 0 = λ min ( H ∞ ) . However, the assumption λ 0 &gt; 0 is a strong assumption because it tends to zero if the size of the training data tends to infinity as shown by Su and Yang [2019]. Ji and Telgarsky [2020] achieved logarithmic width requirements for NTK-separable data with a margin γ . They derived the risk bound of order 1 / ( γ 2 √ n ) for two-layer ReLU networks using Rademacher complexity. The recent work Chen et al. [2021] extended their results from shallow to deep neural networks, the authors considered the NTK feature learning and proved the bound of ˜ O ( e O ( L ) γ √ m n ) . Recently, Lei et al. [2026] derived the bound of 1 / ( γ 2 n ) for two-layer ReLU networks. However, all the above bounds explicit suboptimal 1 √ n dependence on the sample size n or only focus on shallow networks.

Complementing the NTK framework, another line of research employs algorithmic stability to analyze neural networks. In particular, Liu et al. [2020] demonstrated that the Hessian spectral norm scales with width m as ˜ O ( 1 √ m ) , providing the theoretical basis for subsequent studies on generalization in overparameterized networks [Richards and Kuzborskij, 2021, Lei et al., 2022, Taheri and Thrampoulidis, 2024, Taheri et al., 2025]. The work [Taheri and Thrampoulidis, 2024] achieves the bound of ˜ O (1 /nγ 2 ) for shallow neural networks, which is almost optimal, as illustrated by Shamir [2021], Schliserman and Koren [2023]. More recently, Taheri et al. [2025] extended this line of work to deep networks, obtaining a generalization bound of ˜ O ( e O ( L ) / ( nγ 2 )) . However, this approach often assume smooth activation functions and can not apply to the non-smooth ReLU networks. In summary, these works either consider smooth neural networks or develop generalization bounds with exponential dependency on L . These limitations motivate two fundamental questions:

Can we develop optimal risk bounds of 1 / ( γ 2 n ) for deep ReLU networks through refined Rademacher complexity analysis? Furthermore, is it possible to replace the exponential dependence on L with poly( L ) by using neural networks with width as a polylogarithmic function of n ?

In this paper, we provide affirmative answers to both questions. Our main contributions are listed below.

1. We prove that gradient decent with step size η and T iterations achieves the convergence rate of F S ( W ) / ( ηT ) , where F S ( W ) = 3 ηT L S ( W ) + ∥ W -W (0) ∥ 2 F , W is a reference model, W (0) is the initialization point and L S ( · ) is the training error. This indicates that all iterates remain within a neighborhood of W . By refining the analysis of ReLU activation patterns, we reduce the overparameterization requirement by a factor of L 6 as compared to Chen et al. [2021] (see Remark 1 ).
2. We establish a population risk bound of ˜ O ( L 4 F ( W ) /n ) , where F ( W ) extends the empirical counterpart F S ( W ) to the population setting. Our analysis introduces two key technical contributions. First, we derive sharper Rademacher complexity bounds for the hypothesis class encompassing all gradient descent iterates. A central innovation is our representation of the

complexity via products of sparse matrices, whose norms are tightly controlled using optimizationinformed estimates (see Remark 2). Second, by leveraging the covering number techniques, we prove that ReLU networks remain ˜ O ( L 2 ) -Lipschitz continuous in a neighborhood around the initialization-a substantial improvement over previous exponential bounds [Xu and Zhu, 2024, Taheri et al., 2025] (see Remark 3).

3. For NTK separable data with a margin γ , we show that neural networks with m = ˜ O ( L 16 log( n/δ )(log n ) 8 /γ 8 ) can achieve ˜ O ( L 6 / ( γ 2 n )) risk. This improves on Chen et al. [2021]'s ˜ O ( e O ( L ) γ -1 √ m/n ) by simultaneously (a) removing the exponential depth dependence, (b) eliminating the √ m width factor and (c) achieving the optimal dependence on sample complexity (see Table 1 for a comparison with existing work).

## 2 Related Works

## 2.1 Optimization

The foundational work of Jacot et al. [2018] introduced the Neural Tangent Kernel (NTK) framework, demonstrating that in the infinite-width limit, neural networks behave as linear models with a fixed tangent kernel [Liu et al., 2020, Lee et al., 2019]. This lazy training regime [Chizat et al., 2019], where parameters remain close to initialization, enables gradient descent to converge to global optima near initialization [Du et al., 2019a, Arora et al., 2019a]. These analyses showed that the training dynamics can be governed by the NTK Gram matrix, which leads to substantial overparameterization ( m ≳ n 6 /λ 4 0 ). This was later improved by Oymak and Soltanolkotabi [2020]. They showed that if the square-root of the number of the network parameters exceeds the size of the training data, randomly initialized gradient descent converges at a geometric rate to a nearby global optima. The work (Ji and Telgarsky [2020]) achieved polylogarithmic width requirements for logistic loss by leveraging the 1-homogeneity of two-layer ReLU networks. However, it should be mentioned that this special property does not hold for deep networks. The NTK framework was extended to deep architectures by Arora et al. [2019b] for CNNs and by Du et al. [2019b] for ResNets using the last-layer NTK. Xu and Zhu [2024] pointed out that such a characterization is loose, only capturing the contribution from the last layer. They further gave the uniform convergence of all layers as m →∞ and convergence guarantee for stochastic gradient descent (SGD) in streaming data setting. Allen-Zhu et al. [2019] showed that the optimization landscape is almost-convex and semi-smooth, based on which they proved that SGD can find global minima. Cao and Gu [2019] introduced the neural tangent random feature and showed the convergence of SGD under the overparameterized assumption m ≳ n 7 .

## 2.2 Generalization

The NTK framework has yielded generalization bounds scaling as √ y ⊤ ( H ∞ ) -1 y /n [Arora et al., 2019a, Cao and Gu, 2019]. This data-dependent complexity measure helps to distinguish between random labels and true labels. Li and Liang [2018] showed that SGD trained networks can achieve small test error on specific structured data. A very popular approach to studying the generalization of neural networks is via the uniform convergence, which analyzes generalization gaps in a hypothesis space using tools such as Rademacher complexity or covering numbers [Neyshabur et al., 2015, Bartlett et al., 2017, Golowich et al., 2018, Liu et al., 2024]. However, this could lead to vacuous generalization bound in some cases [Nagarajan and Kolter, 2019]. Moreover, these bounds typically exhibit exponential dependence on depth L , thus often leading to loose bounds [Chen et al., 2021]. This capacity-based method usually results in the generalization rate of the order ˜ O (1 / √ n ) . Recent work has also exploited stability arguments for generalization guarantees [Richards and Kuzborskij, 2021, Lei et al., 2022, Taheri and Thrampoulidis, 2024, Deora et al., 2023, Taheri et al., 2025]. The main idea of algorithmic stability is to study how the perturbation of training samples would affect the output of an algorithm [Rogers and Wagner, 1978]. The connection to generalization bound was established in Bousquet and Elisseeff [2002]. Hardt et al. [2016] gave the stability analysis of SGD for Lipschitz, smooth and convex problems. Lei and Ying [2020] further studied SGD under much wilder assumptions. Liu et al. [2020] identified weak convexity of neural networks, enabling stability analyses with polynomial width requirements for quadratic loss [Richards and Kuzborskij, 2021, Lei et al., 2022]. Moreover, Taheri and Thrampoulidis [2024], Taheri et al. [2025] obtained generalization bounds of order ˜ O (1 /n ) by using a generalized local quasi-convexity property for

sufficiently parameterized networks. However, these methods depend on smooth activations, and whether similar or even better bound can be established for deep ReLU networks is still unknown. The recent work derived excess risk bounds of order ˜ O (1 /n ) for shallow ReLU networks [Lei et al., 2026].

## 3 Preliminaries

Notation Throughout the paper, we denote a ≲ b if there exists a constant c &gt; 0 such that a ≤ cb , and denote a ≍ b if both a ≲ b and b ≲ a hold. We use the standard notation O ( · ) , Ω( · ) and use ˜ O ( · ) , ˜ Ω( · ) to hide polylogarithmic factors. Denote by I {·} the indicator function (i.e., taking the value 1 if the argument holds true, and 0 otherwise). We use N ( µ, σ 2 ) to denote the Gaussian distribution of mean µ and variance σ 2 . For a positive integer n , we denote [ n ] := { 1 , . . . , n } . For a vector x ∈ R d , we use ∥ x ∥ 2 to denote its Euclidean norm. For a matrix A ∈ R m × n , we denote ∥ A ∥ 2 and ∥ A ∥ F the corresponding spectral norm and Frobenius norm respectively. The (2 , 1) -norm of A is defined as ∥ A ∥ 2 , 1 = ∑ n j =1 ∥ A : j ∥ 2 . Let ⟨· , ·⟩ be the inner product of a vector or a matrix, i.e., for any matrices A,B ∈ R m × n , we have ∥ A ∥ 2 F = tr ( A ⊤ A ) and ⟨ A,B ⟩ = tr ( A ⊤ B ) . Let L ∈ N , A = ( A 1 , . . . , A L ) and B = ( B 1 , . . . , B L ) be two collections of arbitrary matrices such that A i and B i have the same size for all i ∈ [ L ] . We define ⟨ A , B ⟩ = ∑ L i =1 tr ( A ⊤ i B i ) . Denote ∥ A ∥ 2 , ∞ = max l ∥ A l ∥ 2 . For a matrix W , we define ( w r ) ⊤ the r -th row of W . Denote ∥ · ∥ 0 the l 0 -norm which is the number of nonzero entries of a matrix or a vector. We denote C ≥ 1 as an absolute value, which may differ from line to line.

Let X ⊆ R d be the input space, Y = { 1 , -1 } be the output space, and Z = X × Y . Let ρ be a probability measure defined on Z . Let S = { z i = ( x i , y i ) : i = 1 , . . . , n } be a training dataset drawn from ρ . Let W := R m × d × ( R m × m ) L -1 be the parameter space. W 1 ∈ R m × d and W l ∈ R m × m for l = 2 , . . . , L is the weight of the l -th hidden layer. W = ( W 1 , . . . , W L ) ∈ W denotes the collection of weight matrices for all layers. Let a = ( a 1 , . . . , a m ) ⊤ ∈ R m be the weight vector of the output layer and σ ( · ) = max {· , 0 } denote the ReLU activation function. For x ∈ X , we consider the L -layer deep ReLU neural networks with width m as follows,

<!-- formula-not-decoded -->

Given an input x ∈ X and parameter matrix W = ( W 1 , · · · , W L ) of an L -layer ReLU network f W ( x ) . We denote the output of the l -th layer by h l ( x ) = √ 2 m σ ( W l h l -1 ( x )) with h 0 ( x ) = x . Then f W ( x ) = a ⊤ h L ( x ) . We define B R ( W ) = { ˜ W ∈ W : max l ∥ W -˜ W l ∥ 2 ≤ R } . The performance of the network f W ( x ) is measured by the following empirical risk L S ( W ) and population risk L ( W ) , respectively:

<!-- formula-not-decoded -->

where we use logistic loss ℓ ( z ) := log(1 + exp( -z )) throughout this paper. We further assume the following symmetric initialization [Nitanda and Suzuki, 2020, Kuzborskij and Szepesvári, 2023, Xu and Zhu, 2024]:

Assumption 1 (Symmetric initialization) . Without loss of generality, we assume the network width m is even, and a r + m 2 = -a r ∈ {-1 , +1 } for 1 ≤ r ≤ m/ 2 . W (0) ∈ W satisfies

<!-- formula-not-decoded -->

We remark that this initialization is for theoretical simplicity, using general initialization techniques will not affect the main results. We fix the output weights { a r } and use Gradient Descent (GD) to train the weight matrix W [Ji and Telgarsky, 2020, Arora et al., 2019a, Zou et al., 2018].

Definition 1 (Gradient Descent) . GD updates { W ( k ) } by

<!-- formula-not-decoded -->

where η &gt; 0 is the step size.

Note that in each layer we employ √ 2 /m as the regularization factor instead of the conventional √ 1 /m [Ji and Telgarsky, 2020], which is due to E x ∼N (0 , 1) σ 2 ( x ) = 1 / 2 for our activation function σ ( · ) . This scaling matches both the theoretical framework of Du et al. [2019a] and the initialization scheme of [He et al., 2015] (where weights w l r ∼ N (0 , 2 /m ) ). As will be shown later (Appendix A), this regularization ensures stable gradient propagation and maintains consistent variance across layers.

The following assumption is standard in the literature [Cao and Gu, 2019, Ji and Telgarsky, 2020, Chen et al., 2021].

Assumption 2. We assume X = S d -1 be the sphere.

Throughout the paper, we assume that Assumptions 1 and 2 always hold true.

Error decomposition In this work, we analyze the performance of gradient descent through the lens of population risk. To facilitate this analysis, we decompose the population risk L ( W ( T )) as follows

<!-- formula-not-decoded -->

where the first term captures the generalization gap, quantifying the network's performance on unseen data. The second term represents the optimization error, which reflects GD's ability to find global minima. We will use tools in the optimization theory to study the optimization error [Ji and Telgarsky, 2020, Schliserman and Koren, 2022], and Rademacher complexity to control the generalization gap [Mohri et al., 2018].

## 4 Main Results

In this section, we present the main results. In Section 4.1, we show the optimization analysis of gradient descent. In Section 4.2, we use Rademacher complexity to control the generalization gap. In Section 4.3, we apply our generalization results to NTK-separable data with a margin γ .

## 4.1 Optimization Analysis

We introduce the following notations for a reference model W

<!-- formula-not-decoded -->

Without loss of generality, we assume F S ( W ) ≥ 1 .

Theorem 1. Let Assumptions 1, 2 hold. If m ≳ L 16 (log m ) 4 log( nL/δ ) F 4 S ( W ) , η ≤ min { 4 / (5 L ) , 1 / (20 L ˜ F S ( W )) } , then with probability at least 1 -δ , for all t ≤ T we have

<!-- formula-not-decoded -->

Remark 1 . Our theorem shows that the convergence rate is bounded by the optimization error of a reference model, implying that any low-loss reference point guarantees good convergence. While prior works relied on NTK-induced solutions [Richards and Kuzborskij, 2021, Arora et al., 2019a], we prove that there exists a reference model near initialization under the milder Assumption 3. Furthermore, our analysis implies that all training iterates remain within a neighborhood of the reference point, and thus near initialization, aligned with previous observations but without studying the kernel or the corresponding Gram matrix directly [Du et al., 2019a,b].

Here we provide the proof sketch and compare it with previous works. The starting point is to show deep ReLU networks admit almost convexity ( Lemma 19 ):

<!-- formula-not-decoded -->

for W , W ∈ B R ( W (0)) .

Then we can show all the iterates remain in B 2 √ F S ( W ) ( W (0)) and the following inequality holds (Lemma 21),

<!-- formula-not-decoded -->

Telescoping gives the theorem. Chen et al. [2021] introduce the following neural tangent random feature (NTRF) function class:

<!-- formula-not-decoded -->

They show that gradient descent achieves a training loss of at most 3 ϵ NTRF , where ϵ NTRF denotes the minimal loss over the NTRF function class (see Theorem 3.3 therein). In contrast, our approach directly analyzes the GD iterates and shows that the existence of a nearby reference point with small training error is sufficient to ensure convergence. For a fair comparison, under Assumption 3, both analyses yield an optimization error of ˜ O (1 /T ) . However, our method significantly relaxes the overparameterization requirement, improving the width dependence by a factor of L 6 (see Remark 5).

## 4.2 Generalization Analysis

We use Rademacher complexity to study the generalization gap, which measures the ability of a function class to correlate random noises.

Definition 2 (Rademacher complexity) . Let F be a class of real-valued functions over a space X , S 1 = { x 1 , · · · , x n } ⊂ X . We define the following empirical Rademacher complexity as

<!-- formula-not-decoded -->

where ϵ = ( ϵ i ) i ∈ [ n ] ∼ {± 1 } n are independent Rademacher variables, i.e., taking values in {± 1 } with the same probability.

We further define the following worst-case Rademacher complexity,

<!-- formula-not-decoded -->

We define G = sup z ℓ ( yf W ( x )) , and

<!-- formula-not-decoded -->

We consider the following function space

<!-- formula-not-decoded -->

where the parameter space is defined as

<!-- formula-not-decoded -->

Here we use F ( W ) instead of F S ( W ) to get a data-independent hypothesis space. We will show F ( W ) is an upper bound of F S ( W ) with high probability. According to Theorem 1, all the iterations fall into W 1 with high probability. We use the following lemma to relate the generalization gap of smooth loss function with Rademacher complexity.

Lemma 1 (Srebro et al. [2010]) . Let G ′ = sup z, W ∈W 1 ℓ ( yf W ( x )) . For any 0 &lt; δ &lt; 1 , we have with probability at least 1 -δ/ 2 over S , for any W ∈ W 1 ,

<!-- formula-not-decoded -->

Now we need to control R S 1 ,n ( F ) and G ′ . As will be shown in Lemma 22, with high probability there holds

<!-- formula-not-decoded -->

To estimate G ′ , we employ covering numbers to derive a uniform upper bound of f W ( x ) -f W ( x ) . Then we use the smoothness of ℓ to show that for all G ′ -2 G ≲ L 4 log mF ( W ) . Plugging these bounds into Lemma 1 gives the generalization gap. Combined with Theorem 1, we derive the following excess risk error. The full proofs are provided in Appendix C.

Theorem 2. Let Assumptions 1, 2 hold. If m ≳ L 16 d (log m ) 5 log( nL/δ ) F 4 S ( W ) , η ≤ min { 4 / (5 L ) , 1 / (20 L ˜ F S ( W )) } , ηT ≍ n , then with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Remark 2 (Improved Rademacher complexity) . In previous work [Chen et al., 2021], they derived the bound of ˜ O ( 4 L L √ mF ( W ) /n ) . Specifically, they use the generalization analysis in Bartlett et al. [2017], which requires to estimate the following term

<!-- formula-not-decoded -->

Note that ∏ L l =1 ∥ W l ∥ 2 could lead to exponential dependence on the depth. Moreover, ∥ ( W l ) ⊤ -( W l ) ⊤ ∥ 2 , 1 ≤ √ m ∥ W l -W l ∥ F , inducing an explicit √ m term. However, we reduce the dependence of L to polynomial ( L 2 ). Furthermore, we sharpen the dependence on width from √ m to logarithmic terms. The key idea is to introduce the following expression

<!-- formula-not-decoded -->

where ̂ G l L, 0 ( x i ) is a matrix defined in (36), h l 0 is the output of l -th layer of f W (0) . We further show that ̂ G l L, 0 ( x i ) is of the order ˜ O ( L/ √ m ) with high probability, from which we can derive R S 1 ,n ( F ) = ˜ O ( L 2 / √ n ) . We will show that this improved Rademacher complexity is crucial to get almost optimal risk bounds in NTK separable data.

Remark 3 (Analysis of Lipschitzness) . To bound the term G ′ = sup z, W ∈W 1 ℓ ( yf W ( x )) , we analyze the difference f W ( x ) -f W ( x ) . Since both W , W ∈ B R ( W (0)) for some R , we only need to study the local variation f W ( x ) -f W (0) ( x ) . This approach necessitates characterizing the uniform behavior of deep networks in B R ( W (0)) , specifically establishing control over their Lipschitz constants near initialization. Existing works usually lead to an exponential dependence on L [Xu and Zhu, 2024, Taheri et al., 2025], thus resulting in a e O ( L ) term in the generalization bound. In particular, Lemma F.3 and F.5 in Liu et al. [2020] pointed out that ∥ h l ( x ) ∥ ≤ C L , ∥ ∂f W ( x ) /∂h l ( x ) ∥ 2 ≤ C L -l +1 √ m . Based on these observations, Taheri et al. [2025] showed that

<!-- formula-not-decoded -->

On the other hand, to analyze the output difference near initialization, we observe that

<!-- formula-not-decoded -->

reducing our task to bounding the hidden layer perturbation. Previous approaches, including Xu and Zhu [2024] and Du et al. [2019b], employ a recursive estimation:

<!-- formula-not-decoded -->

where in the second inequality they used ∥ h l 0 ( x ) ∥ 2 ≤ C L and ∥ W l (0) ∥ 2 ≲ √ m . Although this method provides a straightforward bound, it leads to an exponential dependence on depth L due to the recursive nature of the estimation.

In contrast to previous work, we develop the covering-number strategy to avoid the exponential dependence on depth. Specifically, we first show that for any finite set of size N : K = { x 1 , · · · , x N } , if m = ˜ Ω( L 10 log( N ) R 2 ) , then ∥ h l ( x i ) -h l 0 ( x i ) ∥ 2 = ˜ O ( L 2 R √ m ) holds for i ∈ [ N ] , l ∈ [ L ] (Lemma 15). We further take a 1 / ( C L √ m ) -covering D = { x j : j = 1 , . . . , | D |} of the input space. Recall that the input space X = S d -1 , it is well known from Corollary 4.2.13 in Vershynin [2018] that the number of 1 / ( C L √ m ) -covering is given by | D | ≤ (1 + 2 C L √ m ) d . Applying Lemma 15 to D derives that if m = ˜ Ω( L 10 log( | D | ) R 2 ) , then

<!-- formula-not-decoded -->

Note that although the covering number could be exponential in L , we only require logarithm of it, thus leading to polynomial dependence. For any input x , we use the closest cover point x j ∈ D to approximate ∥ h l ( x ) -h l ( x j ) ∥ 2 , ∥ h l 0 ( x ) -h l 0 ( x j ) ∥ 2 . Combining these yields the key technical lemma (Lemma 16):

<!-- formula-not-decoded -->

This implies that the network is ˜ O ( L 2 ) -Lipschitz near initialization. More details can be found in Lemma 16 and its proof.

## 4.3 Optimal rates on NTK-separable data

In this section, we apply our general analysis to NTK-separable data [Ji and Telgarsky, 2020, Nitanda et al., 2020, Chen et al., 2021, Taheri and Thrampoulidis, 2024, Deora et al., 2023], and obtain the optimal rates.

Assumption 3. There exists γ &gt; 0 and a collection of matrices W ∗ = { W 1 ∗ , · · · , W L ∗ } satisfying ∑ L l =1 ∥ W l ∗ ∥ 2 F = 1 , such that

<!-- formula-not-decoded -->

This means that the dataset is separable by the NTK feature at initialization with a margin γ . Nitanda et al. [2020] pointed out that this assumption is weaker than positive eigenvalues of NTK Gram matrix, which has been widely used in the literature [Du et al., 2019b,a, Arora et al., 2019a]. With the above assumption, we have the following optimal risk bound on NTK separable data. The proof is given in Appendix D.

Theorem 3. Let Assumptions 1, 2, 3 hold. If m ≳ L 16 d (log m ) 5 log( nL/δ )(log T ) 8 /γ 8 , η ≤ 4 / (5 L ) , ηT ≍ n , then with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Remark 4 (Proof sketch) . To apply the result in Theorem 2, we need to estimate F ( W ) , for which it suffices to bound L ( W ) and G = sup z ℓ ( yf W ( x )) . For the first part, we control it by L S ( W ) using Bernstein inequality (Eq.(14)). Let W = W (0) + 2 log T W ∗ /γ , plugging into (4) obtains ℓ ( y i f W ( x i )) ≤ 1 /T and further L S ( W ) ≤ 1 /T , implying F S ( W ) = ˜ O (1 /γ 2 ) . In order to control ℓ ( yf W ( x )) , we leverage the ˜ O ( L 2 ) -Lipschitzness of f W ( x ) . Indeed, for any x ∈ X , there holds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 5 (Discussion on optimization error) . Under Assumption 3, Theorem 3.3 and Proposition 4.2 of Chen et al. [2021] show that when the network width satisfies m = ˜ Ω( L 22 /γ 8 ) , the training error is of the order ˜ O (1 /T ) . We achieve the same guarantee under a significantly milder width condition of m = ˜ Ω( L 16 /γ 8 ) . This improvement is enabled by two key technical advances: a sharper bound for (4), and a tighter estimate of the iterate distance ∥ W ( t ) -W ∥ F . Specifically, we improve the bound in (4) by a factor of L 1 / 3 , and show that ∥ W ( t ) -W ∥ F = ˜ O (1 /γ ) , improving upon the previous ˜ O ( √ L/γ ) bound. Together, these refinements reduce the required network width by a factor of L 6 .

Remark 6 (Comparison) . Under similar assumptions, Ji and Telgarsky [2020] derived the bound ˜ O ( 1 γ 2 √ n ) for shallow networks, which was recently improved to ˜ O ( 1 γ 2 n ) based on an improved control of the Rademacher complexity [Lei et al., 2026]. For deep ReLU networks, Chen et al. [2021] developed the bound of the order ˜ O ( e O ( L ) γ √ m n ) via Rademacher complexity [Bartlett et al., 2017], suffering from exponential depth dependence, explicit width requirement and suboptimal √ 1 /n scaling. Taheri et al. [2025] improved the result to ˜ O ( e O ( L ) γ 2 n ) for deep networks. The dependence on n, γ is optimal up to a logarithmic factor [Shamir, 2021, Schliserman and Koren, 2023]. However, their results require smooth activations and exponential width in L . Our rate is almost-optimal and enjoys polynomial dependence over the network depth. Furthermore, our bound holds under the overparameterization ˜ Ω(1 /γ 8 ) , matching the requirement in Ji and Telgarsky [2020], Chen et al. [2021]. This is much better than 1 /γ 6 L +4 in Taheri et al. [2025]. To the best of our knowledge, these are the best generalization bound and width condition for deep neural networks.

## 5 Experiments

In this section, we make some experimental verifications to support our theoretical analysis. Our excess risk analysis in Theorem 3 imposes an NTK separability assumption, which has been validated in the literature. For example, [Ji and Telgarsky, 2020] demonstrates that Assumption 3 holds for a noisy 2-XOR distribution, where the dataset is structured as follows:

<!-- formula-not-decoded -->

Here, the factor 1 √ d -1 ensures that ∥ x ∥ 2 = 1 , × above denotes the Cartesian product, and the label y only depends on the first two coordinates of the input x . As shown in Ji and Telgarsky [2020], this dataset satisfies Assumption 3 with 1 /γ = O ( d ) , which implies that our excess risk bound in Theorem 3 becomes O ( d 2 /n ) for this dataset. We conducted numerical experiments and observed that the test error decays linearly with d 2 /n . The population loss for the test error is computed over all 2 d points in the distribution.

Settings We train two-layer ReLU networks by gradient descent on noisy 2-XOR data. We fix the width m = 128 , T = 500 , η = 0 . 1 . We have conducted two experiments. With a fixed dimension d = 6 , we vary the sample size n . The results are presented in Figure 1a. With a fixed sample size n = 64 , we vary the dimension d and the corresponding table is provided in Figure 1b.

<!-- image -->

<!-- image -->

In both experiments, we observe that the test error is of the order d 2 /n (approximately 0 . 15 d 2 /n ). This shows the consistency between our excess risk bounds in Theorem 3 and experimental results. We conducted the experiments on Google Colab. A simple demonstration reproducing our numerical experiments is available as a Google Colab notebook at: https://github.com/YuanfanLi2233/ nips2025-optimal .

## 6 Conclusion and Future Work

In this paper, we present optimization and generalization analysis of gradient descent-trained deep ReLU networks for classification tasks. We explore the optimization error of F S ( W ) / ( ηT ) under a milder overparameterization requirement than before. We establish sharper bound of Rademacher complexity and Lipschtiz constant for neural networks. This helps to derive generalization bound of order ˜ O ( F ( W ) /n ) . For NTK-separable data with a margin γ , our methods lead to the optimal rate of ˜ O (1 / ( nγ 2 )) . We improve the existing analysis and require less overparameterization than previous works.

There remain several interesting questions for future works. First, it is an interesting question to extend our methods to SGD. Second, while we establish polynomial Lipschitz constants near initialization, investigating whether similar bounds hold far from initialization would deepen our theoretical understanding. Finally, we only consider fully-connected neural networks. It is interesting to study the generalization analysis of networks with other architectures, such as CNNs and Resnets [Du et al., 2019b].

## Acknowledgement

The authors are grateful to the anonymous reviewers for their thoughtful comments and constructive suggestions. The work of Yuanfan Li and Zheng-Chu Guo is partially supported by the National Natural Science Foundation of China (Grants No. 12271473 and U21A20426). The work by Yunwen Lei is partially supported by the Research Grants Council of Hong Kong [Project No. 17302624]. Yiming's work is partially supported by Australian Research Council (ARC) DP250101359.

## References

Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via over-parameterization. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 242-252. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/allen-zhu19a.html .

Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 322-332. PMLR, 09-15 Jun 2019a. URL https: //proceedings.mlr.press/v97/arora19a.html .

Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. Advances in neural information processing systems , 32, 2019b.

- Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. Advances in neural information processing systems , 30, 2017.
- Olivier Bousquet and André Elisseeff. Stability and generalization. Journal of machine learning research , 2 (Mar):499-526, 2002.
- Yuan Cao and Quanquan Gu. Generalization bounds of stochastic gradient descent for wide and deep neural networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/file/ cf9dc5e4e194fc21f397b4cac9cc3ae9-Paper.pdf .
- Zixiang Chen, Yuan Cao, Difan Zou, and Quanquan Gu. How much over-parameterization is sufficient to learn deep relu networks?, 2021. URL https://arxiv.org/abs/1911.12360 .
- Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. Advances in neural information processing systems , 32, 2019.
- Puneesh Deora, Rouzbeh Ghaderi, Hossein Taheri, and Christos Thrampoulidis. On the optimization and generalization of multi-head attention. arXiv preprint arXiv:2310.12680 , 2023.
- Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 1675-1685. PMLR, 09-15 Jun 2019a. URL https://proceedings.mlr.press/v97/du19c.html .
- Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes overparameterized neural networks, 2019b. URL https://arxiv.org/abs/1810.02054 .
- Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. In Conference On Learning Theory , pages 297-299. PMLR, 2018.
- Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In International conference on machine learning , pages 1225-1234. PMLR, 2016.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision , pages 1026-1034, 2015.
- Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems , 31, 2018.
- Ziwei Ji and Matus Telgarsky. Polylogarithmic width suffices for gradient descent to achieve arbitrarily small test error with shallow relu networks, 2020. URL https://arxiv.org/abs/1909.12292 .
- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 25. Curran Associates, Inc., 2012. URL https://proceedings.neurips. cc/paper\_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf .
- Ilja Kuzborskij and Csaba Szepesvári. Learning lipschitz functions by gd-trained shallow overparameterized relu neural networks, 2023. URL https://arxiv.org/abs/2212.13848 .
- Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems , 32, 2019.
- Yunwen Lei and Yiming Ying. Fine-grained analysis of stability and generalization for stochastic gradient descent. In International Conference on Machine Learning , pages 5809-5819. PMLR, 2020.
- Yunwen Lei, Rong Jin, and Yiming Ying. Stability and generalization analysis of gradient methods for shallow neural networks. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 38557-38570. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ fb8fe6b79288f3d83696a5d276f4fc9d-Paper-Conference.pdf .
- Yunwen Lei, Puyu Wang, Yiming Ying, and Ding-Xuan Zhou. Optimization and generalization of gradient descent for shallow relu networks with minimal width. Journal of Machine Learning Research , 27:1-35, 2026.

- Yuanzhi Li and Yingyu Liang. Learning overparameterized neural networks via stochastic gradient descent on structured data. Advances in neural information processing systems , 31, 2018.
- Chaoyue Liu, Libin Zhu, and Misha Belkin. On the linearity of large non-linear models: when and why the tangent kernel is constant. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 15954-15964. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ b7ae8fecf15b8b6c3c69eceae636d203-Paper.pdf .
- Fanghui Liu, Leello Dadi, and Volkan Cevher. Learning with norm constrained, over-parameterized, two-layer neural networks. Journal of Machine Learning Research , 25(138):1-42, 2024. URL http://jmlr.org/ papers/v25/22-1250.html .
- Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning . MIT press, 2018.
- Vaishnavh Nagarajan and J Zico Kolter. Uniform convergence may be unable to explain generalization in deep learning. Advances in Neural Information Processing Systems , 32, 2019.
- Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Conference on learning theory , pages 1376-1401. PMLR, 2015.
- Atsushi Nitanda and Taiji Suzuki. Optimal rates for averaged stochastic gradient descent under neural tangent kernel regime. arXiv preprint arXiv:2006.12297 , 2020.
- Atsushi Nitanda, Geoffrey Chinot, and Taiji Suzuki. Gradient descent can learn less over-parameterized two-layer neural networks on classification problems, 2020. URL https://arxiv.org/abs/1905.09870 .
- Samet Oymak and Mahdi Soltanolkotabi. Toward moderate overparameterization: Global convergence guarantees for training shallow neural networks. IEEE Journal on Selected Areas in Information Theory , 1(1): 84-105, 2020.
- Dominic Richards and Ilja Kuzborskij. Stability &amp; generalisation of gradient descent for shallow neural networks without the neural tangent kernel. Advances in neural information processing systems , 34:8609-8621, 2021.
- William H Rogers and Terry J Wagner. A finite sample distribution-free performance bound for local discrimination rules. The Annals of Statistics , pages 506-514, 1978.
- Matan Schliserman and Tomer Koren. Stability vs implicit bias of gradient methods on separable data and beyond. In Conference on Learning Theory , pages 3380-3394. PMLR, 2022.
- Matan Schliserman and Tomer Koren. Tight risk bounds for gradient descent on separable data. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 68749-68759. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ d8ca28a32c05cd3b9b0940e43720f31b-Paper-Conference.pdf .
- Ohad Shamir. Gradient methods never overfit on separable data. Journal of Machine Learning Research , 22(85): 1-20, 2021. URL http://jmlr.org/papers/v22/20-997.html .
- Nathan Srebro, Karthik Sridharan, and Ambuj Tewari. Smoothness, low noise and fast rates. Advances in neural information processing systems , 23, 2010.
- Lili Su and Pengkun Yang. On learning over-parameterized neural networks: A functional approximation perspective. Advances in Neural Information Processing Systems , 32, 2019.
- Hossein Taheri and Christos Thrampoulidis. Generalization and stability of interpolating neural networks with minimal width. Journal of Machine Learning Research , 25(156):1-41, 2024. URL http://jmlr.org/ papers/v25/23-0422.html .
- Hossein Taheri, Christos Thrampoulidis, and Arya Mazumdar. Sharper guarantees for learning neural network classifiers with gradient methods. In The Thirteenth International Conference on Learning Representations , 2025. URL https://openreview.net/forum?id=h7GAgbLSmC .
- Roman Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- Jiaming Xu and Hanjing Zhu. Overparametrized multi-layer neural networks: Uniform concentration of neural tangent kernel and convergence of stochastic gradient descent. Journal of Machine Learning Research , 25 (94):1-83, 2024. URL http://jmlr.org/papers/v25/23-0740.html .

- Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530 , 2016.
- Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Stochastic gradient descent optimizes overparameterized deep relu networks, 2018. URL https://arxiv.org/abs/1811.08888 .

## A Technical Lemmas

We define the diagonal sign matrix Σ l ( x ) with l ∈ [ L ] by

<!-- formula-not-decoded -->

Then the deep ReLU network has the following matrix product representation:

<!-- formula-not-decoded -->

together with the presentation of h l ( x ) :

<!-- formula-not-decoded -->

We further define G l l ( x ) = √ 2 /m Σ l ( x ) and

<!-- formula-not-decoded -->

from which we can rewrite f W ( x ) as

<!-- formula-not-decoded -->

Hence, for l ∈ [ L ] , we have

Similarly, we define

<!-- formula-not-decoded -->

We denote Σ l 0 ( x ) , h l 0 ( x ) , G a b, 0 ( x ) , H a b, 0 ( x ) as (10), (12), (13) and (14) with W = W (0) .

## A.1 Properties of the Initialization

Given a set of N points on the sphere K = { x 1 , · · · , x N } . We provide general results for any finite set K , then it can be applied to specific choices of K , for example, the training dataset S 1 = { x 1 , · · · , x n } .

Lemma 2 (Theorem 4.4.5 in Vershynin [2018]) . With probability at least 1 -L exp( -Cm ) over the random choice of W (0) , there exists an absolute constant c 0 &gt; 1 such that for any l ∈ [ L ] , there holds

<!-- formula-not-decoded -->

For a sub-exponential random variable X , its sub-exponential norm is defined as follows:

<!-- formula-not-decoded -->

X -E X is sub-exponential too, satisfying

<!-- formula-not-decoded -->

If Y is a sub-gaussian random variable, we define the sub-gaussian norm of Y by

<!-- formula-not-decoded -->

Suppose Y ∼ N (0 , r 2 ) , then σ ( Y ) is also sub-guassian and we have

<!-- formula-not-decoded -->

We have the following lemma:

<!-- formula-not-decoded -->

Lemma 3 (Lemma 2.7.6 in Vershynin [2018]) . A random variable X is sub-gaussian if and only if X 2 is sub-exponential. Moreover,

<!-- formula-not-decoded -->

Now we introduce Bernstein inequality with respect to ∥ · ∥ ϕ 1 ,

Lemma 4 (Theorem 2.8.2 in Vershynin [2018]) . Let X 1 , · · · , X m be independent, mean zero, subexponential random variables, and d = ( d 1 , · · · , d m ) ∈ R m , K ≥ max r ∥ X r ∥ ϕ 1 . Then for every t ≥ 0 , we have

<!-- formula-not-decoded -->

for some absolute constant c .

We introduce the following technical lemma related to the conditional expectation of Gaussian indicator function.

̸

Lemma 5. Suppose w is a m -dim Gaussian random vector with distribution N (0 , I ) . Let c = 0 , b be two given vectors of m -dim. Then we have the following property

<!-- formula-not-decoded -->

Proof. Let u = ⟨ w , c ⟩ , v = ⟨ w , b ⟩ . Then u ∼ N (0 , ∥ c ∥ 2 2 ) , v ∼ N (0 , ∥ b ∥ 2 2 ) . We decompose v into a component dependent on u and an independent residual z :

<!-- formula-not-decoded -->

where z ∼ N ( 0 , ∥ b ∥ 2 2 -⟨ c , b ⟩ 2 ∥ c ∥ 2 2 ) is independent of u . Hence, we have

<!-- formula-not-decoded -->

where the second equality is due to E [ I { u ≥ 0 } u 2 ] = ∥ c ∥ 2 2 2 and the independence of u , z . The third equality follows from the distribution of z . The proof is completed.

The following lemma studies the output of initialization at each layer.

Lemma 6. For any δ &gt; 0 , if m ≳ L 2 log( NL/δ ) , then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. This result directly follows Corollary A.2 in Zou et al. [2018], and we give the proof here for completeness. Note that for 1 ≤ i ≤ N, 1 ≤ l ≤ L ,

<!-- formula-not-decoded -->

Condition on h l -1 0 ( x i ) , we have ⟨ w l j (0) , h l -1 0 ( x i ) ⟩ ∼ N (0 , ∥ h l -1 0 ( x i ) ∥ 2 2 ) , hence

<!-- formula-not-decoded -->

By (16) and Lemma 3, we have

<!-- formula-not-decoded -->

where the last inequality is due to (17). Let X r = 2 σ 2 ( ⟨ w l r (0) , h l -1 0 ( x i ) ⟩ ) - ∥ h l -1 0 ( x i ) ∥ 2 2 , d r = 1 /m,K = C ∥ ∥ h l -1 0 ( x i ) ∥ ∥ 2 2 and apply Lemma 4. We have for any 0 ≤ t ≤ 1 ,

<!-- formula-not-decoded -->

Taking union bounds over i, l , there holds for any 1 ≤ i ≤ N and 1 ≤ l ≤ L ,

<!-- formula-not-decoded -->

Since m ≳ L 2 log( NL/δ ) , let t = √ log( NL/δ ) /m , we have with probability at least 1 -δ , there holds

<!-- formula-not-decoded -->

Now we show the following inequality holds with probability at least 1 -δ ,

<!-- formula-not-decoded -->

When l = 0 , it is true. If (20) holds for l ∈ [ L -1] , then ∥ h l 0 ( x i ) ∥ 2 2 ≤ 4 / 3 . Combined with (19), we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Hence, (20) holds for all i ∈ [ N ] , l ∈ [ L ] , which implies the lemma.

Lemma 7. For any x ∈ X , we have

<!-- formula-not-decoded -->

Proof. Note the r -th row of Σ L 0 ( x ) W L (0) is I {⟨ w L r (0) , h L -1 0 ( x ) ≥ 0 } w L r (0) . Since a r = -a r + m 2 and w L r (0) = w L r + m 2 (0) for all r ∈ [ m 2 ] , we have

<!-- formula-not-decoded -->

It then follows that for all l ∈ [ L -1] ,

<!-- formula-not-decoded -->

Hence, the proof is completed.

Lemma 8. Suppose m ≳ L 2 log( NL/δ ) , then with probability at least 1 -δ , for all i ∈ [ N ] ,

<!-- formula-not-decoded -->

Proof. By Hoeffding inequality, condition on h L -1 0 ( x i ) , with probability at least 1 -δ , there holds

<!-- formula-not-decoded -->

Combined with Lemma 6, we have

<!-- formula-not-decoded -->

The proof is completed.

Let Σ 1 , Σ 2 ∈ R m × m be two diagonal matrices with entries in { 0 , 1 } .

Lemma 9. Suppose m ≳ L 2 log( NL/δ ) , s ≲ m/ ( L 2 log m ) , then with probability at least 1 -δ we have for all i ∈ [ N ] , 2 ≤ a ≤ b ≤ L ,

<!-- formula-not-decoded -->

Proof. We need to prove that for any v ∈ S m -1 , there holds

<!-- formula-not-decoded -->

Note that ∥ Σ 1 v ∥ 0 ≤ s and ∥ Σ 1 v ∥ 2 ≤ 1 . Let P = { v ∈ S m -1 : ∥ v ∥ 0 ≤ s } . We only need to prove that the following inequality holds with probability at least 1 -δ :

<!-- formula-not-decoded -->

Let S be a subspace of S m -1 that has at most s non-zero coordinates. For such a subspace, we choose a 1 / 2 -cover of it and denote this cover by Q . By Lemma 4.2.13 in Vershynin [2018],

<!-- formula-not-decoded -->

The number of such subspaces is M = ( m s ) . We denote all subspaces by S 1 , · · · , S M , and the corresponding covers Q 1 , · · · , Q M . Let ⋃ Q = { v 1 , · · · , v M ′ } with M ′ ≤ ( m s ) 5 s . We first prove that (23) is true for all v j , then it holds simultaneously for all elements in P .

For a unit vector v , we define

<!-- formula-not-decoded -->

and v a -1 ( x i ) = v . Note that condition on h l -1 0 ( x i ) , v l -1 ( x i ) , we take expectation over w l r (0) , applying Lemma 5 implies that

<!-- formula-not-decoded -->

It then follows that

<!-- formula-not-decoded -->

Similar to the proof of Lemma 6, for every v j , 1 ≤ j ≤ M ′ , we apply Lemma 4 to get

<!-- formula-not-decoded -->

Taking the union bounds for all j, l, i yields

<!-- formula-not-decoded -->

where we have used ( m s ) ≤ ( em/s ) s ≤ ( em ) s in the second inequality and the last inequality is due to m ≳ L 2 log( NL/δ ) , s ≲ m/ ( L 2 log m ) . Hence, we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

For any unit vector v with ∥ v ∥ 0 ≤ s , consider the subspace S containing it and the corresponding 1 / 2 - cover Q . There exists a unit vector v j , j ∈ [ M ′ ] , ∥ v -v j ∥ 0 ≤ s and ∥ v -v j ∥ 2 ≤ 1 / 2 . Thus,

<!-- formula-not-decoded -->

Taking sup to the both sides yields

<!-- formula-not-decoded -->

which implies sup v ∈P ∥ H a b, 0 ( x i ) v ∥ 2 ≲ 1 . Hence (23) holds and the proof is completed.

Remark 7 . Our proofs are inspired by Lemma A.9 in Zou et al. [2018], which establishes the estimates under the condition s ≳ log( NL/δ ) . However, we eliminate this assumption via a more refined analysis.

Lemma 10. Suppose m ≳ L 2 log( NL/δ ) , then with probability at least 1 -δ , we have for all i ∈ [ N ] , 2 ≤ a ≤ b ≤ L ,

<!-- formula-not-decoded -->

Although the left-hand side of above inequality could be the production of L terms, it is bounded by ˜ O ( L ) . This lemma shows that the introduction of ReLU activation can avoid exponential explosion.

Proof. For any unit vector v , we decompose it as v = v 1 + · · · + v q , where v j , j ∈ [ q ] are all s -sparse vectors on different coordinates. Therefore,

<!-- formula-not-decoded -->

Here we choose s ≍ m/ ( L 2 log m ) , then q ≲ m/s ≲ L 2 log m . Applying Lemma 9, we have

<!-- formula-not-decoded -->

where we have used Cauchy-Schwartz's inequality in the second inequality. Hence,

<!-- formula-not-decoded -->

The proof is completed.

From the above lemma, we know that if m ≳ L 2 log( NL/δ ) log m , then

<!-- formula-not-decoded -->

Remark 8 . In Lemma 10, we introduce a useful technique that decomposes the unit vector into sparse components. This approach reduces the covering number from 5 m to 5 s ( m s ) , making it easier for high-probability bounds to hold. A related method appears in Lemma 7.3 of Allen-Zhu et al. [2019], but their width exhibits polynomial dependence on n . In contrast, our analysis achieves polylogarithmic width, substantially relaxing the overparameterization requirement.

Using similar techniques we can obtain the following lemma:

Lemma 11. Suppose m ≳ L 2 log( NL/δ ) , then with probability at least 1 -δ , for all i ∈ [ N ] , 2 ≤ a ≤ b ≤ L, ∥ Σ 1 ∥ 0 , ∥ Σ 2 ∥ 0 ≤ s ≲ m/ ( L 2 log m ) ,

<!-- formula-not-decoded -->

Proof. If s = 0 , the above inequality becomes 0 ≲ 1 /L , which holds true. Now we assume s ≥ 1 . Similar to Lemma 9, we only need to prove that for any s -sparse unit vector v there holds

<!-- formula-not-decoded -->

We use the same notation as in Lemma 9, it then follows that for all j ∈ [ M ′ ] , a ≤ l ≤ b, i ∈ [ N ] ,

<!-- formula-not-decoded -->

For a fixed Σ 1 , we assume Σ 1 = diag { d 1 , · · · , d m } with d r ∈ { 0 , 1 } , ∑ r d r ≤ s, r ∈ [ m ] . We have

<!-- formula-not-decoded -->

Condition on v b -1 j ( x i ) , there holds

<!-- formula-not-decoded -->

Let X r = 2 m ( ⟨ w b r (0) , v b -1 j ( x i ) ⟩ ) 2 -2 m ∥ v b -1 j ( x i ) ∥ 2 2 and d = ( d 1 , · · · , d m ) . Then X r are mean-zero sub-exponential random variables, following similar discussions in Lemma 6, we have ∥ X r ∥ ϕ 1 ≤ C m ∥ v b -1 j ( x i ) ∥ 2 2 . Moreover, ∥ d ∥ 2 2 = ∑ m r =1 d 2 r = ∑ m r =1 d r ≤ s, ∥ d ∥ ∞ = 1 . Applying Lemma 4, we have

<!-- formula-not-decoded -->

Choosing t = 1 /L 2 and note that s ≤ m/ ( L 2 log m ) , we have

<!-- formula-not-decoded -->

where the first inequality is due to (26) and ∑ m r =1 d r ≤ s , the last inequality results from (27).

Taking union bounds over all x i , Σ 1 , v j , l , and note that

<!-- formula-not-decoded -->

We have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

Following same techniques of using 1 / 2 -cover in the proof of Lemma 9, we can prove (25), and then complete the proof of the lemma.

Additionally, apply similar methods in Lemma 10 by decomposing the unit vector into s -sparse vectors, we have

<!-- formula-not-decoded -->

Remark 9 . To analyze the influence of the sparse matrix Σ 1 on (25), we propose a key technical improvement: instead of resorting to covering number arguments as in Zou et al. [2018], Allen-Zhu et al. [2019], we leverage a weighted Bernstein inequality. Particularly, existing methods require taking another union bound over both all s -sparse subspaces and their covers (of size ∼ ( m s ) 9 s ), whereas our analysis only needs to union over the sparse subspaces themselves (of cardinality ( m s ) ). Our method directly demonstrates that sparsity inherently lowers computational costs by avoiding the need for dense covers. The simplicity of our technique also underscores the intrinsic benefits of sparse structures in optimization.

## A.2 Properties of Perturbation Terms

Recall that

<!-- formula-not-decoded -->

For any l ∈ [ L ] , let ̂ W l and the diagonal matrix ̂ Σ l ( x ) be the matrices with the same size of W l (0) and Σ l 0 ( x ) , respectively. Define

<!-- formula-not-decoded -->

Lemma 12. Let ̂ G a b ( x ) with 1 ≤ a ≤ b ≤ L be the matrix defined in (29) . Assume max l ∈ [ L ] ∥ ̂ W l ∥ 2 ≤ R ≲ √ m/ ( L 2 √ log m ) , m ≳ L 2 log( NL/δ ) and ̂ Σ l ( x i ) , ̂ Σ l ( x i ) + Σ l 0 ( x i ) ∈ [ -1 , 1] m × m , ∥ ̂ Σ l ( x i ) ∥ 0 ≤ s ≲ m/ ( L 2 log m ) for all i ∈ [ N ] , l ∈ [ L ] . Then, with probability at least 1 -δ for all 1 ≤ a ≤ b ≤ L, i ∈ [ N ] , there holds

<!-- formula-not-decoded -->

Proof. The proof is similar to that of Lemma 8.6 in Allen-Zhu et al. [2019], the differences lie in the dependence of L . We first prove that for any 1 ≤ a ≤ b ≤ L ,

<!-- formula-not-decoded -->

̸

We define a diagonal matrix ( ̂ Σ l 1 ( x i )) k,k = I { ̂ Σ l ( x i ) k,k = 0 } , and ∥ ̂ Σ l 1 ( x i ) ∥ 0 ≤ s . Therefore, ̂ Σ l ( x i ) = ̂ Σ l 1 ( x i ) ̂ Σ l ( x i ) ̂ Σ l 1 ( x i ) . We decompose the left term of (30) into 2 b -a terms and control them respectively. Each matrix can be written as (ignoring the superscripts and x i ).

<!-- formula-not-decoded -->

Then, with probability at least 1 -δ , there holds:

- By Lemma 9, ∥ ∥ ∥ Σ 0 √ 2 m W (0) · · · ̂ Σ 1 ∥ ∥ ∥ 2 ≲ 1 .
- By Lemma 11, ∥ ∥ ∥ ̂ Σ 1 √ 2 m W (0) · · · √ 2 m W (0) ̂ Σ 1 ∥ ∥ ∥ 2 ≲ 1 /L .
- By (28), ∥ ∥ ∥ ̂ Σ 1 √ 2 m W (0) · · · Σ 0 √ 2 m W (0) ∥ ∥ ∥ 2 ≲ √ log m .
- When there is no ̂ Σ , by Lemma 10, ∥ ∥ ∥ Σ 0 √ 2 m W (0) · · · Σ 0 √ 2 m W (0) ∥ ∥ ∥ 2 ≲ L √ log m .

Combined with these results, counting the number of ̂ Σ , we obtain

<!-- formula-not-decoded -->

where in the second inequality we have used ( b -a j ) ≤ ( e ( b -a ) /j ) j ≤ ( eL/j ) j , the last inequality is due to ∑ L j =1 ( e/j ) j converges and it is bounded by a constant. Now we have proved (30).

Denote Σ ′ = Σ 0 + ̂ Σ , through similar expansion, Σ ′ √ 2 m ( W (0) + ̂ W ) · · · (Σ ′ ) √ 2 m ( W (0) + ̂ W )

is the sum of following terms

<!-- formula-not-decoded -->

Since ∥ Σ ′ ∥ 2 ≲ 1 , using Eq. (30), we have

<!-- formula-not-decoded -->

Note that max l ∈ [ L ] ∥ ̂ W l ∥ 2 ≤ R ≲ √ m/ ( L 2 √ log m ) , then by counting the number of ̂ W , we have

<!-- formula-not-decoded -->

The proof is completed.

Denote ˜ Σ( x ) , ˜ h l ( x ) , ˜ G a b ( x ) as (10), (12),(13) when W = ˜ W .

Lemma 13 (Claim 11.2 and Proposition 11.3 in Allen-Zhu et al. [2019]) . For any W , ˜ W ∈ B R ( W (0)) . There exists a series of diagonal matrices { (Σ ′′ ) l ∈ R m × m } l ∈ [ L ] with entries in [ -1 , 1] such that for any l ∈ [ L ] , there holds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above lemma shows that the difference of ReLU networks can be expressed explicitly as the operations of matrices. The main idea is to show that σ ( a ) -σ ( b ) = ( I [ a ≥ 0] -ξ )( a -b ) for ξ ∈ [ -1 , 1] . Now we introduce the following Bernstein inequality under bounded distributions.

Lemma 14 (Theorem 2.8.4 in Vershynin [2018]) . Let X 1 , · · · , X N be independent, mean-zero random variables, such that | X i | ≤ K for all i . Then for every t ≥ 0 , we have

<!-- formula-not-decoded -->

where λ 2 = ∑ N i =1 E X 2 i is the sum of the variance.

The following lemma shows that under overparameterized setting, the outputs and activation patterns for deep relu networks near initialization do not change much.

Lemma 15. Suppose m ≳ L 10 log( NL/δ )(log m ) 4 R 2 . Then with probability at least 1 -δ , for any W ∈ B R ( W (0)) , i ∈ [ N ] and l ∈ [ L ] , there holds

<!-- formula-not-decoded -->

Proof. We prove these two inequalities by induction. Note that (31) holds for l = 0 . Now we suppose (31) holds for l -1 . Let κ &gt; 0 be a constant. For i ∈ [ N ] and l ∈ [ L ] , we define A l ( x i ) = { r ∈ [ m ] : I {⟨ w l r , h l -1 ( x i ) ⟩ ≥ 0 } ̸ = I {⟨ w l r (0) , h l -1 0 ( x i ) ⟩ ≥ 0 }} , then ∥ Σ l ( x i ) -Σ l 0 ( x i ) ∥ 0 = | A l ( x i ) | . Furthermore, we decompose A l ( x i ) into two parts based on the behavior of w l r (0) :

A l 1 ( x i ) = { r ∈ A l ( x i ) : |⟨ w l r (0) , h l -1 0 ( x i ) ⟩| ≤ κ } and A l 2 ( x i ) = { r ∈ A l ( x i ) : |⟨ w l r (0) , h l -1 0 ( x i ) ⟩| &gt; κ } . We will control | A l 1 ( x i ) | and | A l 2 ( x i ) | respectively.

For r ∈ [ m ] , we define F l r,i = I {|⟨ w l r (0) , h l -1 0 ( x i ) ⟩| ≤ κ } . From Lemma 6 we know that 2 / 3 ≤ ∥ h l -1 0 ( x i ) ∥ 2 2 ≤ 4 / 3 . Condition on h l -1 0 ( x i ) , ⟨ w l r (0) , h l -1 0 ( x i ) ⟩ ∼ N (0 , ∥ h l -1 0 ( x i ) ∥ 2 2 ) , then we have

<!-- formula-not-decoded -->

Then by Lemma 14, choose K = 1 , t = mCκ,λ 2 ≤ mCκ , it then follows that

<!-- formula-not-decoded -->

Hence, taking union bounds over l, i , with probability at least 1 -CnL exp( -mκ ) , there holds for all i, l ,

<!-- formula-not-decoded -->

For r ∈ A l 2 ( x i ) , since I {⟨ w l r , h l -1 ( x i ) ⟩ ≥ 0 } ̸ = I {⟨ w l r (0) , h l -1 0 ( x i ) ⟩ ≥ 0 } , we have ( ⟨ w l r , h l -1 ( x i ) ⟩ - ⟨ w l r (0) , h l -1 0 ( x i ) ⟩ ) 2 ≥ |⟨ w l r (0) , h l -1 0 ( x i ) ⟩| 2 &gt; κ 2 .

We deduce that

<!-- formula-not-decoded -->

By assumption ∥ h l -1 ( x i ) -h l -1 0 ( x i ) ∥ 2 ≲ L 2 R √ log m/m and Lemma 6, we get

<!-- formula-not-decoded -->

Combined with (33), we have

From (32) and (34) we know that

<!-- formula-not-decoded -->

where in the last inequality we choose κ = L 4 / 3 (log m ) 1 / 3 R 2 / 3 m -1 / 3 . Hence, due to the overparameterization of m , we have with probability at least 1 -δ , for i ∈ [ N ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying Lemma 13, we have

<!-- formula-not-decoded -->

where ̂ G k l, 0 ( x i ) is defined as

<!-- formula-not-decoded -->

It then follows that

<!-- formula-not-decoded -->

Our overparameterization requirement implies that R ≲ √ m/ ( L 2 √ log m ) . Hence, by Lemma 12, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality results from Lemma 6. As a result, (31) holds for l . We have completed the proof of the lemma.

The above lemma and Lemma 6 imply that with probability at least 1 -δ , for all l ∈ [ L ] , i ∈ [ N ] ,

<!-- formula-not-decoded -->

Remark 10 . Although our approach shares similarities with Lemma B.3 in Zou et al. [2018], our analysis relaxes the required conditions. Specifically, we only require R/ √ m = ˜ O ( L -5 ) , whereas their result demands the stricter scaling R/ √ m = ˜ O ( L -11 ) . Furthermore, compared to Lemma 8.2 in Allen-Zhu et al. [2019], they derive the bound ∥ h l ( x i ) -h l 0 ( x i ) ∥ 2 ≲ RL 5 / 2 √ log m/ √ m , which is worse than our result by a factor of √ L .

The following lemma shows the uniform concentration property of deep ReLU networks, which is crucial in the generalization analysis.

Lemma 16. Let R ≥ 1 be a constant. Assume m ≳ L 11 d (log m ) 5 log( L/δ ) R 2 . Then with probability at least 1 -δ , for W ∈ B R ( W (0)) , l ∈ [ L ] , we have

<!-- formula-not-decoded -->

Proof. We consider the 1 / ( C L √ m ) -cover of S d -1 and denote it by D = { x 1 , · · · , x | D | } . By Lemma 4.2.13 in Vershynin [2018],

<!-- formula-not-decoded -->

Therefore,

Note that Lemma 15 holds for any finite set K = { x 1 , · · · , x N } . Letting K = D , we obtain that if m ≳ L 11 d (log m ) 5 log( L/δ ) R 2 ≳ L 10 log( | D | L/δ )(log m ) 4 R 2 , then

<!-- formula-not-decoded -->

For any x ∈ X , there exists x j ∈ D with ∥ x -x j ∥ 2 ≤ 1 / ( C L √ m ) . It then follows that

<!-- formula-not-decoded -->

where the first inequality is due to σ ( · ) is 1 -Lipschitz. In the second inequality we have used ∥ W l ∥ 2 ≤ ∥ W l (0) ∥ 2 + R ≲ √ m due to Lemma 2. Similarly, we derive that

<!-- formula-not-decoded -->

Therefore, combined with (40), we have

<!-- formula-not-decoded -->

where the last inequality results from R ≥ 1 .

The proof is completed.

Remark 11 . This lemma is a property of deep ReLU networks near initialization that does not depend on the training data. Compared to prior work, while Allen-Zhu et al. [2019], Zou et al. [2018] only establishes bounds for the training data, we prove the uniform convergence over the entire input space. Previous work on uniform concentration demonstrated that sup x ∈X ∥ h l ( x ) -h l 0 ( x ) ∥ 2 ≲ C L R/ √ m [Xu and Zhu, 2024]. We present a significant improvement, reducing the dependence on L from exponential to polynomial.

In the following part, we apply previous technical lemmas to K = S 1 and get properties of deep neural networks over the training dataset.

Lemma 17. Suppose m ≳ L 10 log( nL/δ )(log m ) 4 R 2 . Then with probability at least 1 -δ for all W ∈ B R ( W (0)) , l ∈ [ L ] , i ∈ [ n ]

<!-- formula-not-decoded -->

Proof. For the case l = L ,

<!-- formula-not-decoded -->

where the last inequality is due to Lemma 15.

Now we suppose l &lt; L , then

<!-- formula-not-decoded -->

where the last equality is according to Lemma 7. Applying Lemma 12 and Lemma 15, there holds

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

where we have used (41) in the third inequality. The proof is completed.

Lemma 18. Assume m ≳ L 10 log( nL/δ )(log m ) 4 R 2 . Then with probability at least 1 -δ , for any W ∈ B R ( W (0)) , i ∈ [ n ] and l ∈ [ L ] , there holds

<!-- formula-not-decoded -->

Proof. Since ∥ xy ⊤ ∥ F = ∥ x ∥ 2 ∥ y ∥ 2 for two vectors x, y , we have

<!-- formula-not-decoded -->

Using (38) and Lemma 17, we have

<!-- formula-not-decoded -->

Applying Lemma 15 and (24), we obtain

<!-- formula-not-decoded -->

It then follows that

<!-- formula-not-decoded -->

The proof is completed.

## B Proofs for Optimization

Lemma 19. Suppose m ≳ L 10 log( nL/δ )(log m ) 4 R 2 , then with probability at least 1 -δ , for i ∈ [ n ] , ˜ W , W ∈ B R ( W (0)) , we have

<!-- formula-not-decoded -->

This lemma shows that deep ReLU networks near initialization are almost linear.

Proof. Note that

<!-- formula-not-decoded -->

Since f W ( x i ) = a ⊤ h L ( x i ) , applying Lemma 13, we obtain

<!-- formula-not-decoded -->

the last inequality is due to Lemma 15. We further let

<!-- formula-not-decoded -->

By Lemma 12, we have

<!-- formula-not-decoded -->

Following the proof of Lemma 17, we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where in the last inequality we have used (38). The proof is completed.

The following lemma shows that L S is almost convex near initialization. It becomes more convex as the width grows.

Lemma 20. Suppose m ≳ L 10 log( nL/δ )(log m ) 4 R 2 , then with probability at least 1 -δ , we have for ˜ W , W ∈ B R ( W (0)) ,

<!-- formula-not-decoded -->

Proof. Since ℓ is 1 / 4 -smooth, it enjoys the co-coercivity, i.e., ℓ ( a ) ≥ ℓ ( b ) + ( a -b ) ℓ ′ ( b ) + 2( ℓ ′ ( a ) -ℓ ′ ( b )) 2 , which implies that

<!-- formula-not-decoded -->

We combine the above inequality with Lemma 19 and obtain

<!-- formula-not-decoded -->

The following lemma shows how the distance between gradient descent iterators and the reference model would change after a single gradient descent.

Lemma 21. Suppose m ≳ L 10 log( nL/δ )(log m ) 4 R 2 . Then with probability at least 1 -δ , for η ≤ 4 / (5 L ) and ˜ W , W ∈ B R ( W (0)) ,

<!-- formula-not-decoded -->

Proof. By Lemma 8 and Lemma 18 we know that ∥ ∥ ∥ ∂f W ( x i ) ∂ W l ∥ ∥ ∥ F ≤ 2 , hence ∥ ∥ ∥ ∂f W ( x i ) ∂ W ∥ ∥ ∥ F ≤ 2 √ L , which implies that

<!-- formula-not-decoded -->

where we have used the standard inequality ( a + b ) 2 ≤ 5 a 2 +5 b 2 / 4 . Then by applying Lemma 20 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the proof is completed.

Proof of Theorem 1. We prove it through induction. It holds for t = 0 . Suppose it holds for k = 0 , · · · , t -1 , then we have,

<!-- formula-not-decoded -->

plugging R = 2 √ F S ( W ) and m into Lemma 21, we have

<!-- formula-not-decoded -->

Telescoping and note that ˜ F S ( W ) ≤ L S ( W ) , we obtain

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

≤ ∥

W

(0)

-

W

∥

2

F

+(2 + 20

ηL

˜

F

S

(

W

Hence, when m ≳ F 4 S ( W )(log m ) 4 L 16 , η ≤ 1 / (20 L ˜ F S ( W )) , there holds

<!-- formula-not-decoded -->

))

ηT

L

S

(

W

)

.

This implies that

<!-- formula-not-decoded -->

Therefore, the induction holds for t , the proof is completed.

## C Proofs for Generalization

From (35), we know that there exists ̂ G l L, 0 ( x i ) , 1 ≤ l ≤ L, i ∈ [ n ] , such that

<!-- formula-not-decoded -->

Let E 1 be the event (w.r.t. W (0) ) that ∥ ̂ G l L, 0 ( x i ) ∥ 2 ≤ CL √ log m/m for all W ∈ W 1 . Let E 2 be the event such that ∥ h l -1 0 ( x i ) ∥ 2 2 ≤ 4 / 3 , 1 ≤ l ≤ L, i ∈ [ n ] . Then we have the following bound on Rademacher complexity:

Lemma 22. Let F and W 1 be defined in (7) and (8) , respectively. If the events E 1 , E 2 hold, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to Cauchy-Schwartz inequality. Since the event E 1 holds, we have for all W ∈ W 1 , 1 ≤ k ≤ l ≤ L, i ∈ [ n ] ,

<!-- formula-not-decoded -->

where we have used ∥ AB ∥ F ≤ ∥ A ∥ 2 ∥ B ∥ F for two matrices A,B . Therefore,

<!-- formula-not-decoded -->

For l ∈ [ L ] , let ( g l 1 ) ⊤ , · · · , ( g l m ) ⊤ be the rows of matrix ̂ G l L, 0 (˜ x i )( W l -W l (0)) , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore,

This implies that

<!-- formula-not-decoded -->

The last inequality is due to the event E 2 . Combined with (44), we have

<!-- formula-not-decoded -->

The proof is completed.

Now we provide the proof for Theorem 2

Proof of Theorem 2. We first control G ′ = sup z, W ∈W 1 ℓ ( yf W ( x )) . We denote ¯ h l ( x ) as the output of l -th layer of the network f W ( x ) . Then f W ( x ) = a ⊤ ¯ h L ( x ) . By the definition of F ( W ) , we have max l ∥ W l -W l (0) ∥ 2 ≤ ∥ W -W (0) ∥ F ≤ √ F ( W ) . For W ∈ W 1 , there holds ∥ W l -W l (0) ∥ 2 ≤ ∥ W l -W l (0) ∥ 2 + ∥ W l -W l ∥ 2 ≤ 2 √ F ( W ) for all l ∈ [ L ] . By the overparameterization of m and Lemma 16,

<!-- formula-not-decoded -->

Since logistic loss ℓ is 1 / 4 -smooth, the following property holds,

|

ℓ

′

(

x

)

| ≤

√

It then follows that for any W ∈ W 1 , z ∈ Z ,

<!-- formula-not-decoded -->

where we have used ab ≤ 2 a 2 + b 2 / 8 in the second inequality. Hence, G ′ ≤ 2 G + CL 4 log mF ( W ) . According to Lemma 14, we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

It then follows that

ℓ

(

x

)

/

2

,

x

∈

R

.

<!-- formula-not-decoded -->

which implies that F S ( W ) ≤ F ( W ) . Combined with Theorem 1, we know that with probability at least 1 -δ , W ( t ) ∈ W 1 . It means that all the iterates are in the hypothsesis space. Furthermore, events E 1 , E 2 hold due to Lemma 6 and (37) in Lemma 15. Hence, by Lemma 1 and Lemma 22, there holds

<!-- formula-not-decoded -->

As a result,

<!-- formula-not-decoded -->

from which we derive

<!-- formula-not-decoded -->

where we have used ηT ≍ n . The proof is completed.

## D Proofs on NTK separability

Proof of Theorem 3. We show that there exists W with small F ( W ) for NTK separable data. Let W = W (0) + λ W ∗ . Choose λ = 2log T/γ . Applying Lemma 19 and letting R = λ , we know that if m ≳ L 16 d (log m ) 5 log( nL/δ )(log T ) 2 /γ 8 , then with probability at least 1 -δ , for all i ∈ [ n ] , there holds

<!-- formula-not-decoded -->

where we have used f W (0) ( x ) = 0 for any x ∈ X due to Lemma 7. Therefore, by Assumption 3, we have

<!-- formula-not-decoded -->

As a result,

<!-- formula-not-decoded -->

where we have used log(1 + x ) ≤ x . For any x ∈ X , Lemma 16 implies that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where the last inequality is due to log(1+ t ) ≤ log(2 t ) ≤ 2 log( t ) for t ≥ 2 . Note that if x 2 ≤ αx + β , then x 2 ≤ α 2 +2 β . Combined with (45), it then follows that (let x = √ L ( W ) )

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Applying Theorem 2 and note that ˜ F S ( W ) ≤ L S ( W ) ≤ 1 T , there holds

<!-- formula-not-decoded -->

The proof is completed.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We clarify our contributions and the scope of the paper in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the conclusion, we list several limitations of the paper, e.g., our analysis is only applicable when the gradient descent iterates stay around the initialization point.

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

## Answer: [Yes]

Justification: We present all the necessary assumptions and include the proof of all theoretical results.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: As a theoretical paper, our main contributions are theoretical analyses and proofs. The numerical experiments are designed solely to provide basic validation of our theoretical findings. We have fully disclosed all experimental details in Section 5 of the paper.

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

Answer: [Yes]

Justification: The details and code of experiments can be found in Section 5.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The details of experiments are provided in Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a leGel of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our paper is primarily theoretical in nature. The simple numerical experiments are designed as deterministic verifications of our theoretical results rather than empirical evaluations requiring statistical analysis.

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

Justification: Given the simplicity of our numerical experiments (which are minimal and for verification only), we conducted our experiments on Google Colab.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper confirm with the NeurIPS Code of Ethics. Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and nevatiGe societal impacts of the work performed?

Answer: [NA]

Justification: This is a theoretical paper, and there is no societal impact of the work performed.

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

Justification: This is a theoretical paper, which poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This is a theoretical paper, and we do not use any existing assets.

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

Justification: We do not release any new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This is a theoretical paper, which does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not inGolGe crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper inGolves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLM in preparing the submission.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
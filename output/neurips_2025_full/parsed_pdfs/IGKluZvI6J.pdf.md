## Nonparametric Quantile Regression with ReLU-Activated Recurrent Neural Networks

Hang Yu 1 , 2 , ∗ , Lyumin Wu 3 , ∗ , Wen-Xin Zhou 4 , Zhao Ren 5 , †

- 1 National Key Laboratory for Novel Software Technology, Nanjing University, China 2 School of Artificial Intelligence, Nanjing University, China
- 3 Department of AI and Data Science, The University of Hong Kong, China
- 4 Department of Information and Decision Sciences, University of Illinois Chicago, USA 5 Department of Statistics, University of Pittsburgh, USA

## Abstract

This paper investigates nonparametric quantile regression using recurrent neural networks (RNNs) and sparse recurrent neural networks (SRNNs) to approximate the conditional quantile function, which is assumed to follow a compositional hierarchical interaction model. We show that RNN- and SRNN-based estimators with rectified linear unit (ReLU) activation and appropriately designed architectures achieve the optimal nonparametric convergence rate, up to a logarithmic factor, under stationary, exponentially β -mixing processes. To establish this result, we derive sharp approximation error bounds for functions in the hierarchical interaction model using RNNs and SRNNs, exploiting their close connection to sparse feedforward neural networks (SFNNs). Numerical experiments and an empirical study on the Dow Jones Industrial Average (DJIA) further support our theoretical findings.

## 1 Introduction

Quantile regression (QR) [Koenker and Bassett, 1978] provides a flexible framework for estimating conditional quantiles of a response variable, offering a more comprehensive characterization of the conditional distribution than least squares regression, which focuses solely on the conditional mean. By modeling different quantiles, QR reveals how covariate effects vary across the response distribution, making it particularly valuable when errors are non-normal, heteroscedastic, or when the focus is on tail behavior. Since its introduction, QR has evolved through a wide range of methodological developments, including linear QR [Koenker and Bassett, 1978], quantile autoregression [Koenker and Xiao, 2006], quantile regression forests [Meinshausen, 2006], and quantile boosting [Zheng, 2012], among others. An early step toward integrating neural networks into QR was taken by White [1992], who established theoretical guarantees for single-layer feedforward networks. Building on this foundation, subsequent research has focused on multi-layer feedforward neural networks (FNNs) with ReLU activation functions [Nair and Hinton, 2010]. Assuming that the true conditional quantile function admits a compositional structure consisting of lower-dimensional component functions, Shen et al. [2021] derived a sub-optimal convergence rate for ReLU-based FNN estimators in the presence of heavy-tailed response distributions, which was later refined in a subsequent work [Shen et al., 2025]. Extending this line of work, Padilla et al. [2022] investigated nonparametric QR using ReLU-activated sparse feedforward neural networks (SFNNs) with bounded parameters, achieving optimal convergence rates under Hölder smoothness or Besov-space assumptions on the quantile

∗ Equal contribution.

† Correspondence: Zhao Ren &lt;zren@pitt.edu&gt;

function. More recently, Feng et al. [2024] addressed the problem of covariate shift in nonparametric QR via ReLU FNNs, obtaining minimax-optimal rates under an adaptive self-calibration condition.

However, FNNs often struggle to capture temporal dynamics and sequential dependencies, thereby limiting their effectiveness in modeling dependent data. In contrast, recurrent neural networks (RNNs) [Rumelhart et al., 1986], specifically designed for sequential data, are naturally better suited to such settings. Their capacity to retain and integrate information across time steps has proven effective in applications such as time series forecasting, sequence-to-sequence learning [Sutskever et al., 2014], and modeling long-term dependencies [Hochreiter and Schmidhuber, 1997, Chung et al., 2014]. Motivated by these strengths, we investigate RNNs and their sparse variants (SRNNs) as predictive function classes and establish their theoretical properties within the framework of nonparametric QR.

Recent studies have investigated the theoretical properties of RNNs in the many-to-one setting. Several works have established PAC-style generalization guarantees for RNNs [Chen et al., 2020, Tu et al., 2020, Cheng et al., 2025]. More closely related to our setting are results on their approximation properties under dependent data. In particular, Jiao et al. [2024] showed that RNNs can attain the optimal convergence rate in nonparametric regression when the data are stationary and β -mixing. Building upon these developments, the present study establishes statistical guarantees for the performance of RNNs in QR, which requires distinct technical tools from those employed in least squares estimation. In what follows, we formally define the model, outline the main contributions, and introduce the notation.

Model. Consider sequentially stationary observations { ( x t , y t ) } n t =1 , where any consecutive N observations share the same joint distribution as Z = (( X 1 , Y 1 ) , . . . , ( X N , Y N )) . Here, Y i ∈ R denotes the random outcome of interest, and X i ∈ R d x is a d x -dimensional covariate vector for i ∈ [ N ] . Motivated by the recurrent structure of RNNs, we assume that y t depends on the sequence of covariates ( x t -N +1 , . . . , x t ) . The formal regularity conditions governing this dependence are specified in Section 3.2. Given a quantile level τ ∈ (0 , 1) of interest, we define the conditional τ -th quantile of y t (or Y N ) given x t -N +1 , . . . , x t (or X 1 , . . . , X N ) as

<!-- formula-not-decoded -->

where f 0 : R d x × N → R is the unknown conditional quantile function. Equivalently, the relationship between Y N and ( X 1 , . . . , X N ) can be expressed in additive form as

<!-- formula-not-decoded -->

where ϵ denotes the regression error satisfying P ( ϵ ≤ 0 | X 1 , . . . , X N ) = τ .

A key motivating example is the broad class of nonlinear autoregressive (AR) models with timevarying conditional variance.

Example (Nonlinear AR Model with Time-varying Conditional Variances) . Consider the heteroscedastic nonlinear AR model

<!-- formula-not-decoded -->

where ϕ : R p → R and σ : R q → R are unknown functions, { ϵ t } ∞ t = -∞ is a sequence of independently and identically distributed (i.i.d.) random variables satisfying q τ ( ϵ t ) = 0 . The integers p and q denote the AR order, with p ≤ q . This model conforms to the general formulation in (1.1) by setting N = q , Y N = W t , and X i = W t -( N -i +1) for i ∈ [ q ] . Under this representation, the conditional quantile function f 0 ( X 1 , . . . , X N ) is given by ϕ ( W t -1 , . . . , W t -p ) .

Contributions. We summarize the main contributions of this work as follows.

- To the best of our knowledge, this work provides the first approximation error bounds for functions within a hierarchical interaction model [Kohler and Langer, 2021] using both RNNs and SRNNs. These bounds highlight the ability of such architectures to effectively capture the complexity inherent in hierarchical interaction structures. Our analysis builds upon the close connections between SFNNs and SRNNs, as established in Lemma 4 and Lemma 14 of the supplementary material. In particular, we show that any SRNN can be represented by an SFNN with a slightly larger but less sparse architecture, and vice versa, indicating that their respective function classes are comparable in expressive power. This result complements the equivalence between FNNs and RNNs previously demonstrated by Jiao et al. [2024].

- Built upon the established approximation error bounds, we conduct a comprehensive error analysis for nonparametric QR with weakly dependent data using RNNs and SRNNs. Specifically, we estimate the true conditional quantile function f 0 within a hierarchical interaction model characterized by intrinsic smoothness γ ⋆ . We show that, for a stationary exponentially β -mixing sequence of n observations, the empirical risk minimizers based on both RNNs and SRNNs achieve a convergence rate of n -2 γ ⋆ / (2 γ ⋆ +1) under the squared L 2 norm, up to a logarithmic factor. This rate coincides with the minimax-optimal rate established by SchmidtHieber [2020]. Furthermore, we derive a slower convergence rate for the case of algebraically β -mixing dependence.

̸

Notations. For two sequences { a n } and { b n } , we write a n ≳ b n if there exists a constant C &gt; 0 , independent of n , such that a n ≥ Cb n . We write a n ≲ b n if b n ≳ a n . Additionally, we use the notation a n ≍ b n when both a n ≲ b n and a n ≳ b n hold. For any α ∈ R , let ⌊ α ⌋ denote the largest integer that is strictly smaller than α , and ⌈ α ⌉ denote the smallest integer that is strictly larger than α . We denote N = { 1 , 2 , . . . } , N 0 = N ∪ { 0 } , and Z = { . . . , -2 , -1 , 0 , 1 , 2 , . . . } . For any N ∈ N , we define [ N ] as { 1 , . . . , N } . For any set S , we denote its cardinality by |S| . For x = ( x 1 , . . . , x d ) ⊤ ∈ R d , we define ∥ x ∥ p = ( ∑ d i =1 | x i | p ) 1 /p , ∥ x ∥ ∞ = max i | x i | , ∥ x ∥ 0 = ∑ i ✶ ( x i = 0) . For A = [ a i,j ] ∈ R m × n , we define ∥ A ∥ 0 = ∑ i ∑ j ✶ ( a i,j = 0) and vec( A ) = ( a 1 , 1 , . . . , a 1 ,n , . . . , a m, 1 , . . . , a m,n ) ⊤ ∈ R mn . Moreover, for any real-valued function h defined on a domain X , we define ∥ h ∥ ∞ = sup x ∈X | h ( x ) | . Let P X be a probability measure on X . The L p norm of h ( 1 ≤ p &lt; ∞ ) with respect to P X is defined as ∥ h ∥ p = ( E X ∼ P X | h ( X ) | p ) 1 /p .

## 2 Methodologies

In this section, we formally define the architectures of RNNs and SRNNs, both employing the ReLU activation function, σ ( x ) = max { x, 0 } , applied elementwise to vector inputs. Building on these architectures, we then introduce nonparametric QR estimators based on RNNs and SRNNs, constructed through empirical risk minimization (ERM) over overlapping subsequences of the data.

## 2.1 RNNs and SRNNs

An RNN is characterized by the following parameters: the input dimension d x , the output dimension d y , the width W , and the depth L . Given a time horizon N , an RNN processes an input sequence X := ( x (1) , . . . , x ( N ) ) ∈ R d x × N sequentially through three types of layers: an input layer p , a sequence of recurrent layers { r l } L l =1 , and an output layer q . The architecture generates an output sequence Y := ( y (1) , . . . , y ( N ) ) ∈ R d y × N according to

<!-- formula-not-decoded -->

where P ∈ R W × d x , Q ∈ R d y × W , and V l = ( v (1) l , . . . , v ( N ) l ) ∈ R W × N for l ∈ [ L ] ∪ { 0 } . For each time step t ∈ [ N ] and each layer l ∈ [ L ] , the recurrent operation r ( t ) l is defined by

<!-- formula-not-decoded -->

where A l , B l ∈ R W × W , c l ∈ R W , and r (0) l = 0 ∈ R W .

By composing all L +2 layers, the overall RNN function r θ can be expressed as

<!-- formula-not-decoded -->

where θ = ( vec ( P ) ⊤ , vec ( A 1 ) ⊤ , vec ( B 1 ) ⊤ , c ⊤ 1 , . . . , vec ( A L ) ⊤ , vec ( B L ) ⊤ , c ⊤ L , vec ( Q ) ⊤ ) ⊤ ∈ R d θ , with d θ = (2 W 2 + W ) L + W ( d x + d y ) . In the many-to-one RNN setting considered here, the prediction function corresponds to the final element of the output sequence, denoted by r ( N ) θ .

We define RNN as the class of RNN prediction functions with bounded outputs:

<!-- formula-not-decoded -->

̸

For simplicity, we assume K ≥ 1 throughout and omit it when boundedness is either understood or not required. Likewise, when the input and output dimensions d x and d y are clear from context, we denote the class more compactly as RNN ( W,L ) .

Such RNN architectures, particularly those with large width or depth, are often substantially overparameterized in practice, which can lead to severe overfitting during training. Introducing sparsity provides an effective mechanism to mitigate this overparameterization. From a theoretical perspective, sparsity reduces the effective hypothesis space, thereby improving generalization bounds. We now formally define RNN sparsity. For a recurrent layer r l , define its sparsity as T ( r l ) = ∥ A l ∥ 0 + ∥ B l ∥ 0 + ∥ c l ∥ 0 . For the input and output layers, we set T ( p ) = ∥ P ∥ 0 and T ( q ) = ∥ Q ∥ 0 , respectively. The total sparsity of an RNN r θ is then given by T ( r θ ) = T ( p ) + T ( q ) + ∑ L l =1 T ( r l ) . Using this notation, we define the class of SRNNs as

<!-- formula-not-decoded -->

where s denotes the sparsity budget. SRNNs combine the sequential representation power of RNNs with the computational and statistical advantages of sparse architectures, making them particularly attractive for both theoretical analysis and practical implementation in resource-constrained environments.

## 2.2 Nonparametric quantile regression

As is standard in the QR literature, we begin by imposing regularity conditions on the noise variable ϵ in (1.1), conditioned on the covariates X 1 , . . . , X N , as formalized in the following assumption.

Assumption 1. The conditional density of ϵ given X 1 , . . . , X N , denoted by p ϵ | X 1 ,...,X N , exists and is continuous over its support. Moreover, it satisfies, almost surely over X 1 , . . . , X N ,

<!-- formula-not-decoded -->

for constants ¯ p ≥ p &gt; 0 . In addition, there exists a constant l 0 &gt; 0 such that, almost surely over X 1 , . . . , X N , | p ϵ | X 1 ,...,X N ( u 1 ) -p ϵ | X 1 ,...,X N ( u 2 ) | ≤ l 0 | u 1 -u 2 | for all u 1 , u 2 ∈ R .

This assumption is standard in the literature and has been adopted in prior works such as Belloni and Chernozhukov [2011], Belloni et al. [2019], and Padilla et al. [2022]. It plays a crucial role in linking the excess risk to the squared L 2 error of the estimators, as established in Theorem 3.

Under Assumption 1, the target function f 0 is the unique minimizer of the population check loss

<!-- formula-not-decoded -->

where Π denotes the joint distribution of X 1 , . . . , X N and ρ τ ( u ) = ( τ -✶ ( u &lt; 0)) u is the check loss. Hereafter, we assume that the joint distribution Π has compact support on [0 , 1] d x × N . Given a dataset { ( x t , y t ) } n t =1 , we form overlapping subsequences to construct the training sample

<!-- formula-not-decoded -->

We then estimate the target function f 0 by ERM over a function class F , yielding the estimator

<!-- formula-not-decoded -->

In the sections that follow, we derive error bounds for the cases where F = RNN d x , 1 ( W,L,K ) and F = SRNN d x , 1 ( W,L,K,s ) , respectively.

## 3 Statistical Theory

In this section, we begin by introducing the hierarchical interaction model and derive error bounds for function approximation under this framework using both RNNs and SRNNs. Building on these results, we first establish oracle-type inequalities and subsequently use them to obtain separate upper bounds on the L 2 error of QR estimators based on RNNs and SRNNs, under the assumption that the data-generating process satisfies a stationary β -mixing condition.

## 3.1 Approximation error bounds

The theoretical performance of neural networks critically depends on the properties of the underlying function class. A commonly adopted assumption in nonparametric statistics is that the true regression function belongs to a Hölder class. We recall its definition below, following [Stone, 1982].

Definition 1 (Hölder Class of Functions C β d ( X , K ) ) . Given a domain X ⊆ R d , a positive Hölder smoothness parameter β , and a constant K &gt; 0 , the β -Hölder function class is defined as

̸

<!-- formula-not-decoded -->

where r = ⌊ β ⌋ , s = β -r , ∂ α = ∂ α 1 · · · ∂ α d with α = ( α 1 , . . . , α d ) ∈ N d 0 and ∥ α ∥ 1 = ∑ d i =1 α i . Moreover, we refer to γ = β/d as the dimension-adjusted degree of smoothness of C β d ( X , K ) .

Specifically, any function f ∈ C β d ( X , K ) is bounded in magnitude by K . Without loss of generality, we assume throughout the paper that K ≥ 1 . Stone [1982] established that the minimax convergence rate for estimating a regression function under the L 2 norm over the Hölder class C β d ( X , K ) is n -γ/ (2 γ +1) . However, in neural network applications, the input dimension d is often large, resulting in a small value of the dimension-adjusted smoothness γ , which in turn leads to slow convergence rates. To mitigate the impact of high dimensionality, we consider a class of functions with a compositional structure, known as the hierarchical interaction model, which captures intrinsic low-dimensional structures and allows for more favorable approximation and estimation properties.

Definition 2 (Hierarchical Interaction Model) . Let l , d ∈ N be positive integers, and let K ≥ 1 . Suppose P ⊆ [1 , ∞ ) × N is a parameter set such that sup ( β,t ) ∈P max { β, t } &lt; ∞ . The hierarchical interaction model H l d ( P , K ) is defined recursively as follows:

(i) A function h : R d → R belongs to the first-level model H 1 d ( P , K ) if there exist ( β, t ) ∈ P , a function h 1 ∈ C β t ( R t , K ) , and a set of indices { j 1 , . . . , j t } ⊆ { 1 , . . . , d } such that h ( x ) = h 1 ( x j 1 , . . . , x j t ) for x = ( x 1 , . . . , x d ) ⊤ ∈ R d .

(ii) For l &gt; 1 , a function h belongs to the hierarchical interaction model H l d ( P , K ) if there exist ( β, t ) ∈ P , a function h l ∈ C β t ( R t , K ) , and functions u 1 , . . . , u t ∈ H l -1 d ( P , K ) such that h ( x ) = h l ( u 1 ( x ) , . . . , u t ( x )) for x ∈ R d .

In analogy to the Hölder class, we define the intrinsic smoothness of H l d ( P , K ) by

<!-- formula-not-decoded -->

This quantity can be interpreted as the effective smoothness of the least regular component in the composition. Crucially, it does not depend on the ambient input dimension, thereby mitigating the curse of dimensionality. Extensive research has established the minimax-optimal convergence rates for models related to the hierarchical interaction model in nonparametric regression, demonstrating that these models retain favorable statistical performance even in high-dimensional settings [Bauer and Kohler, 2019, Schmidt-Hieber, 2020, Kohler and Langer, 2021]. Moreover, the hierarchical interaction model encompasses a broad class of structured functions, including classical models such as additive models, single-index models, and other compositionally structured function classes [Kohler and Langer, 2021].

Based on the preceding definitions, we impose the following assumption on the true quantile function f 0 introduced in (1.1).

Assumption 2. Let K ≥ 1 , l ∈ N , and P ⊂ [1 , ∞ ) × N satisfy sup ( β,t ) ∈P max { β, t } &lt; ∞ . The true quantile function f 0 belongs to the hierarchical interaction model H l d x × N ( P , K ) .

The following theorem provides an error bound for approximating functions within a hierarchical interaction model using RNNs.

Theorem 1. Under Assumption 2, for any W 0 , L 0 ≥ 3 , and a probability measure µ on [0 , 1] d x × N that is absolutely continuous with respect to the Lebesgue measure, the following inequality holds

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Here, the positive constants c 1 -c 3 depend on ( l, N, P , K ) .

Theorem 1 establishes the ability of RNNs to approximate hierarchical functions. In contrast, while Jiao et al. [2024] also explored the approximation capabilities of RNNs, their analysis relied on the restrictive assumption that the true regression function belongs to a Hölder class. As a result, their approximation error bounds exhibit a strong dependence on the input dimension. By leveraging the benefits of the hierarchical interaction model in high dimensions, a more thorough exploration of the theoretical properties of SRNNs becomes possible. This is particularly relevant as SRNNs are commonly used in high-dimensional data scenarios. We then present the following theorem, which provides a result for SRNNs that is not achievable by Jiao et al. [2024].

Theorem 2. Under Assumption 2, for any W 0 ≥ sup ( β,t ) ∈P max { ( β +1) t , ( K +1) e t } , L 0 ≥ 1 , and a probability measure µ on [0 , 1] d x × N that is absolutely continuous with respect to the Lebesgue measure, the following inequality holds

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Here, the positive constants c 4 -c 7 depend on ( l, N, P , K ) .

The approximation error bound established in Theorem 2 demonstrates the effectiveness of SRNNs in approximating hierarchical functions. The key distinction between the error bounds in Theorem 1 and Theorem 2 lies in the construction of the neural networks. Specifically, Theorem 1 is based on RNNs constructed using the methodology in Theorem 3.3 of Jiao et al. [2023], while Theorem 2 employs SRNNs constructed through local Taylor approximations, following the approach of Yarotsky [2017]. Importantly, the difference in the resulting error bounds becomes negligible when applied to the derivation of convergence rates for QR estimators. As demonstrated in Theorem 4 and Theorem 5, both approximation bounds can be incorporated into oracle inequalities, leading to optimal rates for estimators based on RNNs and SRNNs, respectively.

## 3.2 Error bounds for QR estimators

The statistical performance of neural network estimators in regression tasks critically depends on the distribution of the observed data. Classical statistical theory typically assumes that observations are i.i.d. In contrast, we consider a more general setting where the observations { ( x t , y t ) } n t =1 form a stationary β -mixing sequence. To proceed, we introduce the following definitions.

Definition 3 (Stationarity) . A sequence of random vectors { z t } ∞ t = -∞ is said to be stationary if, for any given t ∈ Z and m,k ∈ N 0 , the distribution of the random matrix ( z t , z t +1 , . . . , z t + m ) is identical to that of ( z t + k , z t + k +1 , . . . , z t + m + k ) .

Definition 4 ( β -mixing [Bradley, 1983]) . Let { z t } ∞ t = -∞ be a sequence of random vectors. For any i, j ∈ Z ∪{-∞ , + ∞} , define σ j i = σ ( z i , z i +1 , . . . , z j ) as the σ -algebra generated by z k , i ≤ k ≤ j . For any a ∈ N , the β -mixing coefficient of the stochastic process { z t } ∞ t = -∞ is defined as

<!-- formula-not-decoded -->

We say that { z t } ∞ t = -∞ is algebraically β -mixing if there exist positive constants β 0 and r &gt; 1 such that β ( a ) ≤ β 0 /a r for all a . Similarly, it is said to be exponentially β -mixing if there exist positive constants β 0 , β 1 and r such that β ( a ) ≤ β 0 exp( -β 1 a r ) for all a .

The concept of β -mixing has been extensively studied in the literature [Yu, 1994, Mohri and Rostamizadeh, 2010, Phandoidaen and Richter, 2020, Jiao et al., 2024]. In particular, a number of

procedures have been developed to estimate the mixing rate r [McDonald et al., 2011, 2015]. We emphasize that our data assumption includes the nonlinear AR model with time-varying conditional variances, which was not considered in Jiao et al. [2024] due to their differing stationarity assumption on the sequential observations.

The following theorem presents oracle-type inequalities for the QR estimator when the function class is specified as RNNs and SRNNs.

Theorem 3. Assume Assumption 1 holds, that { ( x t , y t ) } n t =1 is a stationary β -mixing sequence, and ∥ f 0 ∥ ∞ ≤ K for some K ≥ 1 . For any positive integer ℓ such that n ≥ 4 ℓN , we define m max = 2( n -N +1) / ( Nℓ ) and m min = ( n -N +1) / (2 Nℓ ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, there exists a constant c 8 &gt; 0 , such that for any u ≥ 1 , the ERM estimator ̂ f in (2.1) satisfies P ( ∥ ̂ f -f 0 ∥ 2 ≥ c 8 ( δ a + δ b + √ u/m max ) ) ≲ Nℓe -u + n ⌊ log 2 (2 K/δ b ) ⌋ β (( ℓ -1) N +1) . (3.3)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, there exists a constant c 9 &gt; 0 , such that for any u ≥ 1 , the ERM estimator ̂ f in (2.1) satisfies P ( ∥ ̂ f -f 0 ∥ 2 ≥ c 9 ( δ a + δ b + √ u/m max ) ) ≲ Nℓe -u + n ⌊ log 2 (2 K/δ b ) ⌋ β (( ℓ -1) N +1) . (3.4)

Remark 1. To the best of our knowledge, the established oracle inequalities for QR under dependence represent a new contribution and are technically nontrivial. To place these contributions in context, we begin with a concise overview of the analytical framework, highlighting the key methodological innovations that arise at each stage of the analysis.

- Novel decomposition. First, we introduce a donut-shaped decomposition that has not appeared in the context of nonparametric regression with neural networks. Specifically, we introduce donutshaped sets, which allow us to decompose the probability bound P ( ∥ ̂ f -f 0 ∥ 2 &gt; δ ⋆ ) into manageable components by bounding each term separately, where δ ⋆ = c 8 ( δ a + δ b + √ u/m max ) for RNNs and δ ⋆ = c 9 ( δ a + δ b + √ u/m max ) for SRNNs. This decomposition differs from the direct argument used in Eq. (16) of Jiao et al. [2024], whose analysis of the excess risk critically relies on the squared loss. To handle the check loss function, we develop the novel decomposition described above.
- Refined blocking technique. Next, we introduce a refined blocking technique to relate the mixing sequence to its i.i.d. counterparts, which differs from the approach in Jiao et al. [2024]. In Step 2 of their proof of Theorem 13, part of the data within each partition is discarded to facilitate analysis under the squared loss. In comparison, our new partitioning procedure, i.e., Figure 2, retains all observations, thereby enabling a more comprehensive analysis under dependent data. Moreover, while Lemma 16 in Jiao et al. [2024] plays a central role in their argument, it cannot be directly applied to our setting. Instead, we employ a probabilistic counterpart, i.e., Lemma 5, to carry out our analysis.
- Sharper inequality . Finally, we develop a novel and sharp empirical process inequality, i.e., Lemma 7, that underpins the tightness of our oracle inequality. One reason the result of Shen et al. [2021] lacks tightness is that their analysis relies on a non-sharp application of the Bernstein inequality. To address this limitation, we derive Lemma 7 by building on Theorem 7.3 of Bousquet [2003] and Corollary 5.1 of Chernozhukov et al. [2014].

For both function classes, the non-asymptotic bound in Theorem 3 comprises three components: the approximation error δ a , the stochastic error δ b , and a dependence-adjustment term that accounts for the discrepancy between dependent and independent sequences. The integer ℓ serves as a key parameter, commonly introduced in time series analysis [Nobel and Dembo, 1993, Yu, 1994], to bridge the behavior of mixing sequences and their i.i.d counterparts. By appropriately tuning the network parameters and selecting ℓ to balance the trade-offs among these components, the ensuing theorems establish the convergence rates of the corresponding estimators when the target function f 0 exhibits a hierarchical interaction structure.

Theorem 4. Let RNN d x , 1 ( W,L,K ) be the hypothesis class F and assume that the probability measure Π on [0 , 1] d x × N is absolutely continuous with respect to the Lebesgue measure.

(i) Suppose Assumption 1 and Assumption 2 hold and { ( x t , y t ) } n t =1 is a stationary exponentially β -mixing sequence. Let W 0 , L 0 ≥ 3 satisfy W 0 L 0 ≍ ( n/ (log n ) (6+1 /r ) ) 1 / (4 γ ⋆ +2) , and define W and L according to (3.1). Then, there exists a constant c 10 &gt; 0 such that the ERM estimator ̂ f in (2.1) satisfies

<!-- formula-not-decoded -->

(ii) Suppose Assumption 1 and Assumption 2 hold and { ( x t , y t ) } n t =1 is a stationary algebraically β -mixing sequence. Let W 0 , L 0 ≥ 3 satisfy W 0 L 0 ≍ ( n (1 -1 /r ) / (log n ) 7 ) 1 / (4 γ ⋆ +2) , and define W and L according to (3.1). Then, there exists a constant c 11 &gt; 0 such that the ERM estimator ̂ f in (2.1) satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 4 shows that the RNN-based QR estimator ̂ f , with a suitably chosen network architecture, achieves the minimax-optimal convergence rate n -2 γ ⋆ / (2 γ ⋆ +1) under the squared L 2 norm in a stationary exponentially β -mixing setting, up to a logarithmic factor. Under stationary algebraically β -mixing conditions, the convergence rate under squared L 2 norm slows to n -(1 -1 /r )2 γ ⋆ / (2 γ ⋆ +1) . Nevertheless, as r →∞ , the exponent -(1 -1 /r )2 γ ⋆ / (2 γ ⋆ +1) approaches -2 γ ⋆ / (2 γ ⋆ +1) for fixed γ ⋆ , indicating that the convergence rate becomes arbitrarily close to the optimal rate. Our results differ from prior QR studies in two key aspects. First, unlike the FNNs used in previous works [Shen et al., 2021, 2024], we employ RNNs as the approximating function class. Second, rather than assuming independent covariates as in Sangnier et al. [2016] and Padilla et al. [2022], we consider a more general dependence structure, where the sequence { ( x t , y t ) } n t =1 is stationary and β -mixing. Furthermore, compared to the convergence rate of RNNs in nonparametric regression [Jiao et al., 2024], our results capture intrinsic low-dimensional structures of the target function, thereby achieving optimal performance while circumventing the curse of dimensionality.

The following theorem establishes the convergence rate of the SRNN-based estimator.

Theorem 5. Let SRNN d x , 1 ( W,L,K,s ) be the hypothesis class F and assume that the probability measure Π on [0 , 1] d x × N is absolutely continuous with respect to the Lebesgue measure.

(i) Suppose Assumption 1 and Assumption 2 hold and { ( x t , y t ) } n t =1 forms a stationary exponentially β -mixing sequence. Let W 0 ≥ sup ( β,t ) ∈P max { ( β +1) t , ( K +1) e t } and L 0 ≥ 3 satisfy W 0 ≍ ( n/ (log n ) (4+1 /r ) ) 1 / (2 γ ⋆ +1) and L 0 ≍ log n . Define W , L , and s according to (3.2). Then, there exists a constant c 12 &gt; 0 such that the ERM estimator ̂ f in (2.1) satisfies

<!-- formula-not-decoded -->

(ii) Suppose Assumption 1 and Assumption 2 hold and { ( x t , y t ) } n t =1 forms a stationary algebraically β -mixing sequence. Let W 0 ≥ sup ( β,t ) ∈P max { ( β +1) t , ( K +1) e t } and L 0 ≥ 3 satisfy W 0 ≍ ( n (1 -1 /r ) / (log n ) 5 ) 1 / (2 γ ⋆ +1) and L 0 ≍ log n . Define W , L , and s according to (3.2). Then, there exists a constant c 13 &gt; 0 such that the ERM estimator ̂ f in (2.1) satisfies

<!-- formula-not-decoded -->

Similar to Theorem 4, under stationary exponentially β -mixing conditions, ̂ f attains the minimaxoptimal convergence rate, up to a logarithmic factor. Under algebraic β -mixing, the convergence rate is slower but approaches the optimal rate as r → ∞ for fixed γ ⋆ . Compared to Theorem 4, Theorem 5 imposes more restrictions on the choice of width W and length L of SRNNs, owing to technical reasons. However, when fixing the length L to be the same for both theorems, Theorem 5 provides guarantees for wider SRNNs. This aligns with the practical setting better because sparsity is commonly considered for deep and wide neural networks.

## 4 Numerical Study

In this section, we conduct numerical experiments to evaluate the finite-sample performance of RNNand SRNN-based QR estimators in comparison to quantile random forest (QRF) and FNN-based estimators. All experiments are implemented in Python. The QRF estimator is trained using the scikit-garden package, and the number of trees is set as 100. For all NN-based estimators, we employ early stopping to mitigate overfitting, as proposed by [Raskutti et al., 2014]. Specifically, the dataset is split into training (80%) and validation (20%) sets. At the end of each epoch, the model's validation performance is evaluated using the mean check loss, and training is terminated if the validation loss does not improve for 20 consecutive epochs. For the FNN-based estimator, we adopt L = 2 hidden layers with widths W 1 = W 2 = 200 , following the architecture in Padilla et al. [2022]. For the RNN- and SRNN-based estimators, we use L = 3 layers with hidden width W = 100 and ReLU activation. To induce sparsity, we prune the smallest 40% of parameters and finetune the remaining weights post-training. The hyperparameters above are fixed throughout the main experiments, and we further investigate the impact of varying these parameters in Appendix D. All neural networks are implemented in PyTorch .

We conduct experiments on the following two nonlinear AR models at quantile levels τ = 0 . 1 and τ = 0 . 5 to validate the theoretical results established in our analysis.

## · Model 1(SIM 1 -1 ):

<!-- formula-not-decoded -->

where Φ( · ) is the standard normal distribution function.

## • Model 2:

where

<!-- formula-not-decoded -->

In the experiments, we examine both light-tailed and heavy-tailed noise distributions: the standard normal distribution N (0 , 1) and a scaled Student's t -distribution with 2 . 25 degrees of freedom t 2 . 25 , respectively. Theorem 1 in Chen and Chen [2000] ensures that the data generated from (4.1) and (4.2) are strictly stationary and exponentially β -mixing. For each model, we generate 110,000 data points, discard the first 100 as a burn-in period, and split the remaining data by assigning the last 100,000 points to the test set and the rest to the training set. For each trained estimator, we compute the mean squared error (MSE) on the test set. We perform 500 Monte Carlo repetitions for each experiment. As shown in Figure 1, the RNN- and SRNN-based estimators exhibit significantly lower variance and superior overall performance compared to the QRF- and FNN-based estimators.

<!-- formula-not-decoded -->

Figure 1: MSE comparison for Model 1 and Model 2 with different noise distributions and quantile levels.

<!-- image -->

## 5 Application

In this section, we conduct experiments on the DJIA dataset 3 to empirically validate our approach under the stationarity assumption. To further assess robustness in nonstationary settings, we provide a complementary case study on GDP forecasting in Appendix C. Since log-returns of stock prices are widely regarded as approximately stationary, the DJIA dataset is well-suited for evaluation in the stationary environment. We partition the data chronologically, allocating the first 19 years for training and the final year for evaluation. We assess the out-of-sample predictive performance of RNN-, SRNN-, FNN-, and QRF-based estimators in a 30-business-day-ahead forecasting task, with implementations following the specifications in Section 4. All models are trained by minimizing the empirical check loss and evaluated on the test set using the same criterion.

Table 1 presents the mean empirical check loss at five quantile levels. The results show that the RNN and SRNN estimators consistently outperform FNN and QRF methods at all quantile levels. Moreover, the SRNNs achieve predictive accuracy comparable to RNNs while inducing sparsity. These empirical findings are consistent with, and provide validation for, our theoretical results.

Table 1: Out-of-sample prediction errors at different quantiles for DJIA growth analysis.

| Model   | τ = 0 . 1   | τ = 0 . 25   | τ = 0 . 5   | τ = 0 . 75   | τ = 0 . 9   |
|---------|-------------|--------------|-------------|--------------|-------------|
| QRF     | 0 . 456     | 0 . 698      | 0 . 817     | 0 . 616      | 0 . 365     |
| FNN     | 0 . 538     | 0 . 735      | 0 . 810     | 0 . 662      | 0 . 404     |
| RNN     | 0 . 410     | 0.640        | 0 . 760     | 0 . 562      | 0 . 306     |
| SRNN    | 0.406       | 0 . 647      | 0.759       | 0.561        | 0.305       |

## 6 Conclusion

This study investigates the convergence properties of nonparametric quantile regression using RNNs and SRNNs. Error bounds are derived for the approximation of functions within a hierarchical interaction model using RNNs and SRNNs, respectively. Based on these error bounds, we demonstrate that, for a stationary, exponentially β -mixing sequence of n observations, the empirical risk minimizers of both RNN- and SRNN-based methods achieve the optimal convergence rate.

Future research could explore the approximation capabilities of RNNs and SRNNs for nonparametric quantile process regression. Instead of focusing on a fixed quantile level τ , a useful extension would involve approximating the entire quantile process indexed by τ ∈ [ τ L , τ U ] ⊆ (0 , 1) , which complements the results of Shen et al. [2024] using FNNs.

3 We use DJIA data from Jan 1, 2000, to Dec 31, 2020, obtained from https://www.investing.com .

## Acknowledgment

Zhao Ren was supported in part by the National Science Foundation (DMS-2113568) and the National Institutes of Health (NIGMS R01GM157600). Wen-Xin Zhou was supported in part by the National Science Foundation (DMS-2401268) and the Australian Research Council Discovery Project Grant (DP230100147).

## References

- T. Adrian, N. Boyarchenko, and D. Giannone. Vulnerable growth. American Economic Review , 109 (4):1263-1289, 2019.
- M. Anthony and P. L. Bartlett. Neural Network Learning: Theoretical Foundations . Cambridge University Press, 2009.
- P. L. Bartlett, N. Harvey, C. Liaw, and A. Mehrabian. Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research , 20(63): 1-17, 2019.
- B. Bauer and M. Kohler. On deep learning as a remedy for the curse of dimensionality in nonparametric regression. The Annals of Statistics , 47(4):2261-2285, 2019.
- A. Belloni and V. Chernozhukov. ℓ 1-penalized quantile regression in high-dimensional sparse models. The Annals of Statistics , 39(1):82-130, 2011.
- A. Belloni, V. Chernozhukov, D. Chetverikov, and I. Fernández-Val. Conditional quantile processes based on series or many regressors. Journal of Econometrics , 213(1):4-29, 2019.
- O. Bousquet. Concentration inequalities for sub-additive functions using the entropy method. In Stochastic Inequalities and Applications , pages 213-247, 2003.
- R. C. Bradley. Absolute regularity and functions of Markov chains. Stochastic Processes and Their Applications , 14(1):67-77, 1983.
- M. Chen and G. Chen. Geometric ergodicity of nonlinear autoregressive models with changing conditional variances. Canadian Journal of Statistics , 28(3):605-614, 2000.
- M. Chen, X. Li, and T. Zhao. On generalization bounds of a family of recurrent neural networks. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS) , pages 1233-1243, 2020.
- X. Cheng, K. Huang, and S. Ma. Generalization and risk bounds for recurrent neural networks. Neurocomputing , 616:128825, 2025.
- V. Chernozhukov, D. Chetverikov, and K. Kato. Gaussian approximation of suprema of empirical processes. The Annals of Statistics , 42(4):1564-1597, 2014.
- J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint , arXiv:1412.3555, 2014.
- J. Fan, Y. Gu, and W.-X. Zhou. How do noise tails impact on deep ReLU networks? The Annals of Statistics , 52(4):1845-1871, 2024.
- X. Feng, X. He, Y. Jiao, L. Kang, and C. Wang. Deep nonparametric quantile regression under covariate shift. Journal of Machine Learning Research , 25(385):1-50, 2024.
- S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation , 9(8):1735-1780, 1997.
- Y. Jiao, G. Shen, Y. Lin, and J. Huang. Deep nonparametric regression on approximate manifolds: Nonasymptotic error bounds with polynomial prefactors. The Annals of Statistics , 51(2):691-716, 2023.

- Y. Jiao, Y . Wang, and B. Yan. Approximation bounds for recurrent neural networks with application to regression. arXiv preprint , arXiv:2409.05577, 2024.
- R. Koenker and G. Bassett. Regression quantiles. Econometrica , 46(1):33-50, 1978.
- R. Koenker and Z. Xiao. Quantile autoregression. Journal of the American Statistical Association , 101(475):980-990, 2006.
- M. Kohler and S. Langer. On the rate of convergence of fully connected deep neural network regression estimates. The Annals of Statistics , 49(4):2231-2249, 2021.
- D. J. McDonald, C. R. Shalizi, and M. Schervish. Estimating beta-mixing coefficients. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS) , pages 516-524, 2011.
- D. J. McDonald, C. R. Shalizi, and M. Schervish. Estimating beta-mixing coefficients via histograms. Electronic Journal of Statistics , 9(2):2855 - 2883, 2015.
- D. J. McDonald, C. R. Shalizi, and M. Schervish. Nonparametric risk bounds for time-series forecasting. Journal of Machine Learning Research , 18(32):1-40, 2017.
- N. Meinshausen. Quantile regression forests. Journal of Machine Learning Research , 7(6):983-999, 2006.
- M. Mohri and A. Rostamizadeh. Stability bounds for stationary φ -mixing and β -mixing processes. Journal of Machine Learning Research , 11(2):789-814, 2010.
- V. Nair and G. E. Hinton. Rectified linear units improve restricted Boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML) , pages 807-814, 2010.
- A. Nobel and A. Dembo. A note on uniform laws of averages for dependent processes. Statistics &amp; Probability Letters , 17(3):169-172, 1993.
- O. H. M. Padilla, W. Tansey, and Y. Chen. Quantile regression with ReLU networks: Estimators and minimax rates. Journal of Machine Learning Research , 23(247):1-42, 2022.
- N. Phandoidaen and S. Richter. Forecasting time series with encoder-decoder neural networks. arXiv preprint , arXiv:2009.08848, 2020.
- G. Raskutti, M. J. Wainwright, and B. Yu. Early stopping and non-parametric regression: An optimal data-dependent stopping rule. Journal of Machine Learning Research , 15(1):335-366, 2014.
- D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature , 323(6088):533-536, 1986.
- M. Sangnier, O. Fercoq, and F. d'Alché Buc. Joint quantile regression in vector-valued RKHSs. In Advances in Neural Information Processing Systems 29 (NIPS) , pages 3693-3701, 2016.
- J. Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU activation function. The Annals of Statistics , 48(4):1875-1897, 2020.
- G. Shen, Y. Jiao, Y. Lin, J. L. Horowitz, and J. Huang. Deep quantile regression: Mitigating the curse of dimensionality through composition. arXiv preprint , arXiv:2107.04907, 2021.
- G. Shen, Y. Jiao, Y. Lin, J. L. Horowitz, and J. Huang. Nonparametric estimation of non-crossing quantile regression process with deep ReQU neural networks. Journal of Machine Learning Research , 25(88):1-75, 2024.
- G. Shen, R. Dai, G. Wu, S. Luo, C. Shi, and H. Zhu. Deep distributional learning with non-crossing quantile network. arXiv preprint , arXiv:2504.08215, 2025.
- C. H. Song, G. Hwang, J. H. Lee, and M. Kang. Minimal width for universal property of deep RNN. Journal of Machine Learning Research , 24(121):1-41, 2023.

- C. J. Stone. Optimal global rates of convergence for nonparametric regression. The Annals of Statistics , 10(4):1040-1053, 1982.
- I. Sutskever, O. Vinyals, and Q. V. Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems 27 (NIPS) , pages 3104-3112, 2014.
- Z. Tu, F. He, and D. Tao. Understanding generalization in recurrent neural networks. In Proceedings of the 8th International Conference on Learning Representations (ICLR) , 2020.
- M. Vidyasagar. Learning and generalisation: with applications to neural networks . Springer Science &amp;Business Media, 2013.
- H. White. Nonparametric estimation of conditional quantiles using neural networks. In Computing Science and Statistics , pages 190-199, 1992.
- D. Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks , 94: 103-114, 2017.
- B. Yu. Rates of convergence for empirical processes of stationary mixing sequences. The Annals of Probability , 22(1):94-116, 1994.
- S. Zheng. Qboost: Predicting quantiles with boosting for regression and binary classification. Expert Systems with Applications , 39(2):1687-1697, 2012.

## Appendix

The appendix is organized as follows: Appendix A provides the proofs of the main theorems presented in Section 3. Appendix B contains the proofs of all auxiliary lemmas referenced in Appendix A. Appendix C presents an additional case study on GDP forecasting that further supports our theoretical findings. Appendix D provides a sensitivity analysis of the hyperparameters for the SRNNs.

## A Proofs of Theorems

We first provide the definition of FNNs for our analysis. An FNN f θ : R d x → R d y can be represented as

<!-- formula-not-decoded -->

where each feedforward layer f l ( x ) is defined as

<!-- formula-not-decoded -->

For each layer l , the weight matrix is A l ∈ R W l × W l -1 and the bias vector is b l ∈ R W l . The initial dimension W 0 = d x and the final dimension W L = d y correspond to the input and output sizes, respectively. The complete parameter vector θ is defined as θ = ( vec ( A 1 ) ⊤ , b ⊤ 1 , . . . , vec ( A L ) ⊤ , b ⊤ L ) ⊤ ∈ R ∑ L l =1 W l ( W l -1 +1) . The width W of the network is defined as max { W 1 , . . . , W L -1 } . We denote FNN d x ,d y ( W,L,K ) as a class of FNNs with width W , depth L and bounded output, defined as

<!-- formula-not-decoded -->

Similar to SRNNs, we define T ( f l ) = ∥ A l ∥ 0 + ∥ b l ∥ 0 , for l ∈ [ L ] and define T ( f θ ) = ∑ L l =1 T ( f l ) as the sparsity of the SFNN f θ . Moreover, we define the class of SFNNs as follows

<!-- formula-not-decoded -->

To maintain consistent input formatting, if the input to an FNN or SFNN is a sequence X ∈ R d x × N , we first stack its columns into a vector and then use this vector as the input to the neural network, that is, f θ ( X ) = f θ ((( x (1) ) ⊤ , . . . , ( x ( N ) ) ⊤ ) ⊤ ) . For notational simplicity, the distinction between representing the input as a sequence X or as its vectorized form is not explicitly made where the context is unambiguous.

For simplicity, we assume K ≥ 1 throughout and omit it when boundedness is understood or not required in the notation of the function classes FNN , SFNN , RNN and SRNN .

## A.1 Proof of Theorem 1

We first introduce two lemmas: the first establishes an error bound for using FNNs to approximate functions within a hierarchical interaction model, and the second demonstrates that an FNN can be represented by an RNN prediction function.

Lemma 1 (Proposition 3.4 in Fan et al. [2024]) . Given a hierarchical interaction model H l d ( P , K ) , for any W 0 , L 0 ≥ 3 , and a probability measure µ on [0 , 1] d x × N that is absolutely continuous with respect to the Lebesgue measure, the following inequality holds

<!-- formula-not-decoded -->

where W = c 15 ⌈ W 0 log W 0 ⌉ and L = c 16 ⌈ L 0 log L 0 ⌉ . Here, the positive constants c 14 -c 16 depend on ( l, P , K ) .

Lemma 2 (Proposition 2 in Jiao et al. [2024]) . For any FNN ¯ f ∈ FNN d x × N,d y ( W,L ) , there exists an RNN prediction function f ∈ RNN d x ,d y (( d x +1) W +1 , 2 L +2 N ) such that

<!-- formula-not-decoded -->

Proof of Theorem 1. Applying Lemma 1 to H l d x × N ( P , K ) , there exist positive constants c 14 -c 16 such that

<!-- formula-not-decoded -->

where W ′ = c 15 ⌈ W 0 log W 0 ⌉ , L ′ = c 16 ⌈ L 0 log L 0 ⌉ .

By Lemma 2, we have that for any ¯ f ∈ FNN d x × N, 1 ( W ′ , L ′ , K ) there exists an RNN prediction function f ∈ RNN d x , 1 (( d x +1) W ′ +1 , 2 L ′ +2 N,K ) such that f ( X ) = ¯ f ( X ) , X ∈ [0 , 1] d x × N .

Combining the two results above, we complete the proof of the theorem.

## A.2 Proof of Theorem 2

Following a similar line of reasoning as in the proof of Theorem 1, we introduce two key lemmas that extend Lemma 1 and Lemma 2 to the sparse setting. The first lemma establishes an error bound for approximating functions under a hierarchical interaction model using SFNNs. The second lemma shows that an SFNN can be equivalently represented by an SRNN prediction function. Proofs of these lemmas are deferred to Appendix B.1 and Appendix B.2, respectively.

Lemma 3. Given a hierarchical interaction model H l d ( P , K ) , for any W 0 ≥ sup ( β,t ) ∈P max { ( β + 1) t , ( K +1) e t } , L 0 ≥ 1 , and a probability measure µ on [0 , 1] d x × N that is absolutely continuous with respect to the Lebesgue measure, there exist W , L , s &gt; 0 such that the following inequality holds

<!-- formula-not-decoded -->

where W = c 18 W 0 , L = c 19 L 0 and s = c 20 L 0 W 0 . Here c 17 -c 20 are positive constants depending on ( l, P , K ) .

Lemma 4. For any SFNN ¯ f ∈ SFNN d x × N, 1 ( W,L,s ) , there exists an SRNN prediction function f ∈ SRNN d x , 1 (( d x +1) W +1 , 2 L +2 N,s +3( d x +1) WN +6( L + N )( d x +1) W ) such that

<!-- formula-not-decoded -->

Proof of Theorem 2. Applying Lemma 3 to H l d x × N ( P , K ) , there exist positive constants c 17 -c 20 such that

<!-- formula-not-decoded -->

where W ′ = c 18 W 0 , L ′ = c 19 L 0 and s ′ = c 20 L 0 W 0 .

Furthermore, Lemma 4 establishes that for any SFNN ¯ f ∈ SFNN d x × N, 1 ( W ′ , L ′ , K, s ′ ) , there exists an SRNN prediction function f ∈ SRNN d x , 1 (( d x +1) W ′ +1 , 2 L ′ +2 N,K,s ′ +3( d x + 1) W ′ N +6( L ′ + N )( d x +1) W ′ ) such that f ( X ) = ¯ f ( X ) , X ∈ [0 , 1] d x × N .

Combining the two results above, we complete the proof of the theorem.

## A.3 Proof of Theorem 3

As a preparatory step for the proof of Theorem 3, we present the following three lemmas:

Lemma 5 (Lemma 2 in Nobel and Dembo [1993]) . Let h be a real-valued Borel measurable function, and let { z t } ∞ t = -∞ be a stationary β -mixing sequence of random vectors. We define a block of length

□

□

L as a sequence of L consecutive vectors from { z t } ∞ t = -∞ . Consider a sequence C consisting of m such blocks, where the gap between any two consecutive blocks is a +1 . That is, for any two consecutive blocks in C , the index of the first vector in the second block exceeds the index of the last vector in the first block by exactly a +1 . Then for any constant t , the following bound holds

<!-- formula-not-decoded -->

Here ˜ C comprises m independent blocks, each drawn from the same distribution as in C .

The lemma above enables us to extend concentration results for i.i.d. data to the setting of β -mixing data. Similar results can be found in Yu [1994], Vidyasagar [2013], and McDonald et al. [2017].

Lemma 6. Under Assumption 1, for any function f : [0 , 1] d × N → [ -K,K ] , the population check loss function satisfies

<!-- formula-not-decoded -->

where c 21 = min { p/ (8 K ) , p 2 / (32 Kl 0 ) } and c 22 = ¯ p/ 2 .

In the analysis of Theorem 3, we consider a generic function class F . The main difference between (i) and (ii) lies in the properties of δ b , which are determined by whether F = RNN d x , 1 ( W,L,K ) or SRNN d x , 1 ( W,L,K,s ) . We present a key concentration result for each of these function classes with their corresponding δ b . For any F , we define F ( δ ) = { f ∈ F | ∥ f -f 0 ∥ 2 ≤ δ } . For convenience, we denote z t = (( x t -N +1 , y t -N +1 ) , . . . , ( x t , y t )) and S ′ = { z t } n t = N . Additionally, for any f , we introduce the following notation:

<!-- formula-not-decoded -->

With the above definition, we have the following lemma.

Lemma 7. Let z 1 , . . . , z n be i.i.d. random vectors drawn from the distribution of Z , and let g be defined as in (A.1). We choose δ b = WL √ log(max { W,L } ) log n/n for RNNs, and δ b = WL √ log( WL 2 ) log n/n for SRNNs. Then there exists a universal constant c 23 &gt; 0 such that for any δ ≥ δ b and 0 ≤ x ≤ nδ 2 , the following inequality holds uniformly over both F = RNN d x , 1 ( W,L,K ) and F = SRNN d x , 1 ( W,L,K,s )

<!-- formula-not-decoded -->

Proof of Theorem 3. First, let m = n -N +1 and, for a given u ≥ 1 , define δ ⋆ = C ( δ a + δ b + √ 2 Nℓu/m ) , where C is given by

<!-- formula-not-decoded -->

Here, the constants c 21 and c 22 are specified in Lemma 6 and c 23 is specified in Lemma 7. We emphasize that in Theorem 3, both c 8 and c 9 can be set to 4 C . Next, for an integer i ∈ N , we define the donut-shaped sets as

<!-- formula-not-decoded -->

With the definition of D i , we can write

<!-- formula-not-decoded -->

Therefore, it reduces to bounding each probability P ( ̂ f ∈ D i ) separately. Following Lemma 6, for any f ∈ D i , we have

<!-- formula-not-decoded -->

We next derive an upper bound of the right-hand side of (A.4). By the definition of δ a , there exists f m ∈ F such that ∥ f m -f 0 ∥ 2 ≤ 2 δ a . Now, if ̂ f ∈ D i , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows from the definition of ̂ f . By Lemma 6, it follows that R τ ( f m ) -R τ ( f 0 ) ≤ 4 c 22 δ 2 a . For any set S , we denote

<!-- formula-not-decoded -->

Then the earlier inequality can be further bounded as

<!-- formula-not-decoded -->

Combining (A.4) and (A.5), we obtain an upper bound of the probability P ( ̂ f ∈ D i ) as

<!-- formula-not-decoded -->

Figure 2: Illustration of the Partitioning Procedure. Here, m j = ∣ ∣ B j ∣ ∣ and m a j = ∣ ∣ B a j ∣ ∣ . We first split S ′ into N sequences. Each sequence B j is then partitioned into ℓ equidistant sub-sequences B a j .

We next relate the dependent observations to independent observations. We partition S ′ into Nℓ sequences step by step. We first partition it into N equidistant sequences {B j } N -1 j =0 . For j ∈ { 0 } ∪ [ N -1] , we define B j := { b j,i } m j i =1 , where b j,i = z iN + j , j ∈ { 0 } ∪ [ N -1] and m j = |B j | . Let r 1 = m mod N denote the remainder. Then, for j &lt; r 1 , m j = ⌈ m/N ⌉ , and for j ≥ r 1 , m j = ⌈ m/N ⌉ -1 . We continue partitioning each B j into ℓ equidistant sub-sequences {B a j } ℓ a =1 .

For a ∈ [ ℓ ] , we define B a j := { b j,a + kℓ } m a j -1 k =0 , where m a j = |B a j | . Let r 2 = m j mod ℓ denote the remainder. Then, for a ≤ r 2 , we have m a j = ⌈ m j /ℓ ⌉ , and for a &gt; r 2 , we have m a j = ⌈ m j /ℓ ⌉ -1 . The detail is shown in Figure 2. With the partition, we have

<!-- formula-not-decoded -->

Plugging it into (A.6) and noting that f m ∈ F (2 i δ ⋆ ) for all i ≥ 1 (since 2 δ a ≤ 2 i δ ⋆ for all i ≥ 1 ), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the fourth inequality follows from the choice of C in (A.2) and the last inequality follows from the union bound.

For each B a j , we leverage the property of β -mixing to relate it to ˜ B a j . The key here is that the blocks comprising ˜ B a j are mutually independent, with each block ˜ b containing exactly N elements. Furthermore, the elements within each block ˜ b ∈ ˜ B a j are identically distributed to those in the corresponding block b ∈ B a j .

For any j ∈ { 0 } ∪ [ N -1] and a ∈ [ ℓ ] , we denote c i = m Nℓ c 21 16 2 2 i δ 2 ⋆ . By Lemma 5, we have

<!-- formula-not-decoded -->

For any j ∈ { 0 } ∪ [ N -1] and a ∈ [ ℓ ] , we next bound the probability

<!-- formula-not-decoded -->

where the first inequality follows that m/ ( Nℓ ) ≥ m a j / 2 and the last inequality follows that c 23 /C ≤ c 21 / 32 .

To this end, we choose δ = 2 i δ ⋆ and x = 2 2 i u . Since C ≥ 1 , we have δ ≥ δ b and 0 ≤ x ≤ m a j δ 2 . Then, Lemma 7 yields

<!-- formula-not-decoded -->

where the second inequality follows that m/ ( Nℓ ) ≤ 2 m a j .

Combining (A.9) with (A.3), (A.7) and (A.8) implies

<!-- formula-not-decoded -->

where the last inequality uses the fact that u ≥ 1 . This proves the claim.

## A.4 Proof of Theorem 4

According to Theorem 1, we obtain

<!-- formula-not-decoded -->

Recall that

<!-- formula-not-decoded -->

We now analyze how these bounds behave under different β -mixing conditions.

· Case 1: Exponentially β -mixing. Recall that for exponentially β -mixing, there exist positive constants β 0 , β 1 , and r such that β ( a ) ≤ β 0 exp( -β 1 a r ) for all a . By plugging (A.10) and (A.11) into (3.3), and ℓ ≍ (log n ) 1 /r such that β (( ℓ -1) N + 1) ≲ 1 /n 2 , together with setting W 0 L 0 ≍ ( n/ (log n ) (6+1 /r ) ) 1 / (4 γ ⋆ +2) , we obtain that there exists a constant c 10 &gt; 0 such that

<!-- formula-not-decoded -->

· Case 2: Algebraically β -mixing. Recall that for algebraically β -mixing, there exist positive constants β 0 and r &gt; 1 such that β ( a ) ≤ β 0 /a r for all k . By plugging (A.10) and (A.11) into (3.3), and choosing ℓ ≍ n 1 /r log n , together with setting W 0 L 0 ≍ ( n (1 -1 /r ) / (log n ) 7 ) 1 / (4 γ ⋆ +2) , we obtain that there exists a constant c 11 &gt; 0 such that

<!-- formula-not-decoded -->

## A.5 Proof of Theorem 5

According to Theorem 2, we obtain

<!-- formula-not-decoded -->

Recall that

<!-- formula-not-decoded -->

□

We now analyze how these bounds behave under different β -mixing conditions.

· Case 1: Exponentially β -mixing. By plugging (A.12) and (A.13) into (3.4), and choosing ℓ ≍ (log n ) 1 /r such that β (( ℓ -1) N + 1) ≲ 1 /n 2 , together with setting W 0 ≍ ( n/ (log n ) (4+1 /r ) ) 1 / (2 γ ⋆ +1) , L 0 ≍ log n , and s ≍ W 0 L 0 , we obtain that there exists a constant c 12 &gt; 0 such that

<!-- formula-not-decoded -->

- Case 2: Algebraically β -mixing. By plugging (A.12) and (A.13) into (3.4), and choosing ℓ ≍ n 1 /r log n , together with setting W 0 ≍ ( n (1 -1 /r ) / (log n ) 5 ) 1 / (2 γ ⋆ +1) , L 0 ≍ log n , and s ≍ W 0 L 0 , we obtain that there exists a constant c 13 &gt; 0 such that

<!-- formula-not-decoded -->

## B Proofs of Lemmas

## B.1 Proof of Lemma 3

This proof is adapted from Schmidt-Hieber [2020]. In preparation for the proof of Lemma 3, we first present two auxiliary lemmas that will be useful in the analysis:

Lemma 8 (Theorem 5 in Schmidt-Hieber [2020]) . For any function f 0 ∈ C β d ( [0 , 1] d , K ) , W 0 ≥ max { ( β +1) d , ( K +1) e d } and L 0 ≥ 1 , the following inequality holds

<!-- formula-not-decoded -->

where W = c 26 W 0 , L = c 27 L 0 and s = c 28 L 0 W 0 . Here, the positive constants c 24 -c 28 depend on ( β, d, K ) .

Lemma 9. Let f 1 ∈ SFNN d 1 ,d 2 ( W 1 , L 1 , s 1 ) and f 2 ∈ SFNN d ′ 1 ,d ′ 2 ( W 2 , L 2 , s 2 ) . Their aggregation and composition satisfy the following properties

Aggregation rule: If L 1 = L 2 and d 1 = d ′ 1 , then f = ( f 1 , f 2 ) ⊤ ∈ SFNN d 1 ,d 2 + d ′ 2 ( W 1 + W 2 , L 1 , s 1 + s 2 ) .

Composition rule: If d 2 = d ′ 1 , then f = f 2 ◦ σ ( f 1 ) ∈ SFNN d 1 ,d ′ 2 (max { W 1 , W 2 } , L 1 + L 2 + 1 , s 1 + s 2 ) .

We omit the proof for Lemma 9 since it is straightforward to obtain the result. We refer readers to Section 7 in Schmidt-Hieber [2020] for more details.

Proof of Lemma 3. In this lemma, we consider a fixed input domain within [0 , 1] d . By the definition of hierarchical interaction model, for any f 0 ∈ H l d ( P , K ) , there exists a sequence of functions { g i } l i =1 such that

<!-- formula-not-decoded -->

where g 1 : [0 , 1] t 1 → [ -K,K ] t 2 and g i : [ -K,K ] t i → [ -K,K ] t i +1 for i = 2 , . . . , l . While this formulation appears slightly different from our original definition of the hierarchical interaction model, they are indeed equivalent. Specifically, for each level i = 2 , . . . , l , we have g i = ( h i, 1 , . . . , h i,t i +1 ) ⊤ , where each component function h i,j ∈ C β i t i ([ -K,K ] t i , K ) for j ∈ [ t i +1 ] . Similarly, g 1 = ( h 1 , 1 , . . . , h 1 ,t 2 ) ⊤ with h 1 ,j ∈ C β 1 t 1 ([0 , 1] t 1 , K ) for j ∈ [ t 2 ] . Here, ( β i , t i ) ∈ P for i ∈ [ l ] , t 1 ≤ d and t l +1 = 1 .

To facilitate the application of Lemma 8, the sequence of functions { g i } l i =1 is transformed into another sequence { ¯ h i } l i =1 as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Building upon this transformation, it follows that

<!-- formula-not-decoded -->

where ¯ h 1 ,j ∈ C β 1 t 1 ( [0 , 1] t 1 , 1 ) for j ∈ [ t 2 ] , ¯ h i,j ∈ C β i t i ( [0 , 1] t i , (2 K ) β i ) for i = 2 , . . . , l -1 and j ∈ [ t i +1 ] , and ¯ h l,j ∈ C β l t l ( [0 , 1] t l , K (2 K ) β l ) for j ∈ [ t l +1 ] .

By Lemma 8, for each i ∈ [ l ] and j ∈ [ t i +1 ] , there exist positive numbers c 24 , c 25 , c 26 , c i 27 and c 28 and a function ¯ f i,j ∈ SFNN ( W i , L i , s i ) , such that

<!-- formula-not-decoded -->

where L i = c 26 L 0 , W i = c i 27 W 0 , and s i = c 28 L 0 W 0 . However, ¯ f i,j may not map into [0 , 1] . For analytical convenience, we further transform it into ¯ f ′ i,j defined by

<!-- formula-not-decoded -->

This transformation can be implemented by adding two additional layers with four parameters to each ¯ f i,j . Under this transformation, for each i ∈ [ l -1] and j ∈ [ t i +1 ] , there exists a function ¯ f ′ i,j ∈ SFNN ( W i +2 , L i , s i +4) , taking values in [0 , 1] , such that

<!-- formula-not-decoded -->

We next construct ¯ f , which provides a consistent approximation to f 0 via the aggregation and composition rules in Lemma 9. For i ∈ [ l -1] , we aggregate ¯ f ′ i, 1 , . . . , ¯ f ′ i,t i +1 into a single function ¯ f ′ i defined by ¯ f ′ i = ( ¯ f ′ i, 1 , . . . , ¯ f ′ i,t i +1 ) ⊤ , then ¯ f ′ i ∈ SFNN ( t i +1 ( W i + 2) , L i + 2 , t i +1 ( s i + 4)) . By composition, we construct ¯ f = ¯ f l ◦ σ ( ¯ f ′ l -1 ) ◦ · · · ◦ σ ( ¯ f ′ 1 ) = ¯ f l ◦ ¯ f ′ l -1 ◦ · · · ◦ ¯ f ′ 1 . Then ¯ f ∈ SFNN ( W,L,s ) , with L = 3( l -1) + ∑ l i =1 L i = c 29 L 0 , W = max i t i +1 ( W i +2) = c 30 W 0 , and s = ∑ l i =1 t i +1 ( s i +4) = c 31 L 0 W 0 . The approximation error between f 0 and ¯ f can then be bounded as follows

<!-- formula-not-decoded -->

where the second inequality follows from the definition of Hölder smoothness.

Since K ∏ l -1 j =1 (2 K ) β j +1 is a constant, it follows that

<!-- formula-not-decoded -->

To ensure that the output lies within [ -K,K ] , we define ¯ f ′ by ¯ f ′ = min {∥ f 0 ∥ ∞ / ∥ ¯ f ∥ ∞ , 1 } · ¯ f. It follows that

<!-- formula-not-decoded -->

Then, we directly establish the existence of a constant c 33 such that

<!-- formula-not-decoded -->

Choosing c 17 = c 33 , c 18 = c 29 , c 19 = c 30 and c 20 = c 31 completes the proof.

□

## B.2 Proof of Lemma 4

In this subsection, we first formally define the architectures of modified recurrent neural networks (MRNNs) and their sparse variants (SMRNNs). First, we define a new activation function that operates on specific components instead of all components. Given an index set I ⊆ N , the modified activation function σ I is defined as

<!-- formula-not-decoded -->

Using this modified activation function, for any l ∈ [ L ] we define the modified recurrent operation as

<!-- formula-not-decoded -->

Here, ˜ A l , ˜ B l ∈ R W × W , ˜ c l ∈ R W , and ˜ r (0) l = 0 ∈ R W .

We denote by MRNN d x ,d y ( W,L,K,s ) the class of neural network functions defined with the structure of RNNs but composed of modified recurrent layers instead of original recurrent layers. It is clear that RNN d x ,d y ( W,L,K,s ) ⊆ MRNN d x ,d y ( W,L,K,s ) . Similarly, SMRNN d x ,d y ( W,L,K,s ) is defined as the class following the structure of SRNNs, also utilizing modified recurrent layers.

The proof builds on Lemma 10 and Lemma 11, which construct two intermediate representations. Specifically, Lemma 10 shows that an SFNN can be equivalently expressed as an SMRNN prediction function, and Lemma 11 further shows that an SMRNN can be represented by an SRNN. Proofs of these lemmas are deferred to Appendix B.5 and Appendix B.6, respectively.

Lemma 10. For any SFNN ¯ f ∈ SFNN d x × N, 1 ( W,L,s ) , there exists an SMRNN prediction function ˜ f ∈ SMRNN d x , 1 (( d x +1) W,N + L, s +3( d x +1) WN ) such that

<!-- formula-not-decoded -->

Lemma 11. For any SMRNN prediction function ˜ f ∈ SMRNN d x ,d y ( W,L,s ) , there exists an SRNN prediction function f ∈ SRNN d x ,d y ( W +1 , 2 L, s +6 LW ) such that

<!-- formula-not-decoded -->

Proof of Lemma 4. From Lemma 10, we know there exists an SMRNN prediction function ˜ f ∈ SMRNN d x , 1 (( d x +1) W,N + L, s +3( d x +1) WN ) such that

<!-- formula-not-decoded -->

Then, applying Lemma 11 to the function ˜ f , we obtain an SRNN prediction function f ∈ SRNN d x , 1 (( d x +1) W +1 , 2 L +2 N,s +3( d x +1) WN +6( d x +1)( L + N ) W ) , such that

<!-- formula-not-decoded -->

This completes the proof.

## B.3 Proof of Lemma 6

To establish the lower bound, we utilize the Lipschitz continuity of p ϵ | X 1 ,...,X N ( · ) , which implies that p ϵ | X 1 ,...,X N ( u ) ≥ p/ 2 when | u | ≤ p/ (2 l 0 ) . By applying Lemma S6 from the supplementary materials of Padilla et al. [2022], we derive the following result:

<!-- formula-not-decoded -->

□

<!-- formula-not-decoded -->

Furthermore, as both f and f 0 are bounded by K , the squared difference satisfies the inequality ( f ( X 1 , . . . , X N ) -f 0 ( X 1 , . . . , X N )) 2 ≤ 2 K | f ( X 1 , . . . , X N ) -f 0 ( X 1 , . . . , X N ) | . By incorporating this result into (B.1) and assuming K ≥ 1 , we arrive at the desired lower bound for the excess quantile risk.

For the upper bound, we have the following decomposition:

<!-- formula-not-decoded -->

✶ ✶ ✶ Recall that R τ ( f ) = E [ ρ τ ( Y N -f ( X 1 , . . . , X N ))] . Choosing u = ϵ and v = f ( X 1 , . . . , X N ) -f 0 ( X 1 , . . . , X N ) , we have

<!-- formula-not-decoded -->

where the last inequality follows from Assumption 1.

## B.4 Proof of Lemma 7

To prove this lemma, we first introduce the definition of the uniform covering number.

Definition 5 (Uniform covering number) . Let d ∈ N and F = { f | X → R } be a class of real-valued functions on X . For ϵ &gt; 0 , the uniform covering number of F under the supremum norm is defined as

<!-- formula-not-decoded -->

where F| x 1 ,..., x d = { ( f ( x 1 ) , . . . , f ( x d )) ⊤ | f ∈ F} ⊆ R d and N ( ϵ, ∥ · ∥ ∞ , W ) is the ϵ -covering number of a subset W ⊆ R d under the supremum norm ∥ · ∥ ∞ .

Given the above definition, we next present two lemmas characterizing the covering numbers of RNNs and SFNNs.

Lemma 12 (RNN covering number bound) . Let F N = RNN d x , 1 ( W,L,K ) . Then we have

<!-- formula-not-decoded -->

Lemma 13 (SFNN covering number bound) . Let F N = SFNN d x , 1 ( W,L,K,s ) . Then we have

<!-- formula-not-decoded -->

The following lemma shows that any SRNN can be represented by an SFNN, which will be used to characterize the covering number of SRNNs.

Lemma 14. For any SRNN prediction function f ∈ SRNN d x ,d y ( W,L,s ) , there exists an SFNN ¯ f ∈ SFNN d x × N,d y ( (2 N -1) W, ( N +1) L +2 , 2 Ns +2 N 2 WL ) such that

<!-- formula-not-decoded -->

Finally, we provide two technical lemmas that serve as essential components in the proof.

Lemma 15 (Theorem 7.3 in Bousquet [2003]) . Let z 1 , . . . , z n be i.i.d. random vectors drawn from the distribution of Z , and F be a measurable class of functions such that E f ( Z ) = 0 for any f ∈ F . Assume sup f ∈F ∥ f ∥ ∞ ≤ A and let σ be a positive constant such that σ 2 ≥ sup f ∈F E f 2 ( Z ) . Then, for any x &gt; 0 ,

<!-- formula-not-decoded -->

□

Lemma 16 (Corollary 5.1 in Chernozhukov et al. [2014]) . Denote S = ([0 , 1] d × R ) N and let z 1 , . . . , z n ∈ S be i.i.d. random vectors drawn from the distribution of Z ∈ S . Let F be a measurable class of functions S → R , to which a measurable envelope F is attached. Assume that ∥ F ∥ 2 &lt; ∞ , and let σ &gt; 0 be any positive constant such that sup f ∈F E f 2 ( Z ) ≤ σ 2 ≤ ∥ F ∥ 2 2 . Furthermore, we assume that there exist constants A ≥ e and ν ≥ 1 such that sup Q N ( ϵ ∥ F ∥ Q 2 , ∥ · ∥ Q 2 , F ) ≤ ( A/ϵ ) ν for any 0 &lt; ϵ ≤ 1 , where the supremum is taken over all n -discrete probability measures Q on S and N ( ϵ, ∥ · ∥ Q 2 , F ) is the ϵ -covering number of F under the L 2 ( Q ) norm. Then,

<!-- formula-not-decoded -->

where ¯ F = max 1 ≤ i ≤ n F ( z i ) .

Proof of Lemma 7. Recall that F ( δ ) = { f ∈ F | ∥ f -f 0 ∥ 2 ≤ δ } .

Since ρ τ ( · ) is a Lipschitz function, we have

<!-- formula-not-decoded -->

Therefore, sup f ∈F ( δ ) | g ( f, z i ) -E Z g ( f, Z ) | ≤ 4 K =: A . Moreover,

<!-- formula-not-decoded -->

which further implies

<!-- formula-not-decoded -->

Denoting E ( δ ) = E sup f ∈F ( δ ) ∣ ∣ n -1 ∑ n i =1 g ( f, z i ) -E Z g ( f, Z ) ∣ ∣ , Lemma 15 gives

<!-- formula-not-decoded -->

for any x ≥ 0 .

Now, we establish an upper bound of the expectation E ( δ ) . We denote M n ( δ ) = { g ( f, z i ) | f ∈ F ( δ ) } .

<!-- formula-not-decoded -->

Combining the Lipschitz continuity of ρ τ ( · ) and Lemma 12 gives that for any ϵ ∈ (0 , K ) ,

<!-- formula-not-decoded -->

Also, the Lipschitz property of ρ τ ( · ) implies that F = 2 K is an envelope function of M n ( δ ) . Thus, for any discrete probability measure Q supported on n points, we have

<!-- formula-not-decoded -->

where the L 2 norm is taken with respect to Q . Applying Lemma 16, we have

<!-- formula-not-decoded -->

for any δ ≥ 1 /n . Thus, when δ ≥ δ b , we have E ( δ ) ≲ δδ b . By combining this and (B.3), there exists a universal positive constant c 34 &gt; 0 such that

<!-- formula-not-decoded -->

holds for any 0 ≤ x ≤ nδ 2 and δ ≥ δ b .

<!-- formula-not-decoded -->

Combining the Lipschitz continuity of ρ τ ( · ) , Lemma 13, and Lemma 14 yields that for any ϵ ∈ (0 , K ) and s ≍ WL ,

<!-- formula-not-decoded -->

Also, the Lipschitz property of ρ τ ( · ) implies that F = 2 K is an envelope function of M n ( δ ) . Thus, for any discrete probability measure Q supported on n points, we have

<!-- formula-not-decoded -->

where the L 2 norm is taken with respect to Q . Applying Lemma 16, we have

<!-- formula-not-decoded -->

for any δ ≥ 1 /n . Thus, when δ ≥ δ b , we have E ( δ ) ≲ δδ b . By combining this and (B.3), there exists a universal positive constant c 35 &gt; 0 such that

<!-- formula-not-decoded -->

holds for any 0 ≤ x ≤ nδ 2 and δ ≥ δ b . Choosing c 23 = max { c 34 , c 35 } completes the proof.

## B.5 Proof of Lemma 10

For any given SFNN ¯ f , we first construct an SMRNN ˜ r θ N +1 such that ˜ r ( N ) θ N +1 is equivalent to the first layer of ¯ f . Subsequently, building upon ˜ r θ N +1 , we sequentially construct ˜ r θ N + L by directly leveraging the remaining L -1 layers of ¯ f , ensuring that ˜ r θ N + L is equivalent to ¯ f .

Proof of Lemma 10. Our construction begins with a key result for building the initial SMRNN ˜ r θ N +1 .

For any given A k, 1 , . . . , A k,N ∈ R 1 × d x for k ∈ [ W ] and a bias vector c ∈ R W , we construct an SMRNN r θ N +1 = r N +1 ◦ r N ◦ · · · ◦ r 1 ◦ p : R d x × N → R ( d x +1) W × N of width ( d x +1) W , depth N +1 . The layers { r l } N +1 l =1 and p are defined explicitly as follows for each t ∈ [ N ] and l ∈ [ N ] .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

□

Combining Lemma 6 in Song et al. [2023] and Step 2 of the proof of Lemma 8 in Jiao et al. [2024] gives that r θ N +1 constructed above satisfies the following property:

<!-- formula-not-decoded -->

In this construction, we have T ( p ) = Wd x , T ( r N +1 ) ≤ W + d x , and for l ∈ [ N ] :

<!-- formula-not-decoded -->

We bound the sparsity of r θ N +1 by

<!-- formula-not-decoded -->

By applying this result to a given SFNN ¯ f , parameterized as ( vec ( ˜ B 1 ) ⊤ , ˜ c ⊤ 1 , . . . , vec ( ˜ B L ) ⊤ , ˜ c ⊤ L ) ⊤ , we proceed to define the corresponding parameters required for initializing the SMRNN ˜ r θ N +1 . Here, ˜ B 1 ∈ R W × ( d x N ) , ˜ B l ∈ R W × W for l = 2 , . . . , L -1 , ˜ B L ∈ R 1 × W , ˜ c l ∈ R W for l ∈ [ L -1] , and ˜ c L ∈ R d y . Specifically, we set c = ˜ c 1 , and for each i ∈ [ W ] and j ∈ [ N ] , we define A i,j = ( ˜ B 1 ) i, ( j -1) d x +1: jd x . The notation ( ˜ B 1 ) i, ( j -1) d x +1: jd x refers to the sub-vector of the i -th row of ˜ B 1 corresponding to entries from column ( j -1) d x +1 to jd x . By applying these definitions to the construction described in (B.4), we obtain ˜ r θ N +1 . It then follows from (B.4) that

<!-- formula-not-decoded -->

We next construct the remaining L -1 layers using the 2 -nd to L -th layers of the SFNN ¯ f . Define

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then SMRNN ˜ r θ N + L = q ◦ ˜ r N + L ◦ · · · ◦ ˜ r N +2 ◦ ˜ r θ N +1 satisfies ˜ r ( N ) θ N + L ( X ) = ¯ f ( X ) .

Finally, we compute the sparsity of ˜ r θ N + L ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality follows from (B.5) and d y = 1 .

## B.6 Proof of Lemma 11

Proof of Lemma 11. This proof is adapted from the proof of Lemma 3 in Song et al. [2023]. We first show that any modified recurrent layer can be represented by the composition of a sequence of recurrent layers and linear maps. Specifically, for any modified recurrent layer ˜ r : R d × N → R d × N satisfying ˜ r ( t ) ( X ) = σ I ( ˜ A ˜ r ( t -1) ( X ) + ˜ B x ( t ) + ˜ c ) , there exist linear maps p : R d × N → R ( d +1) × N and q : R ( d +1) × N → R d × N , along with recurrent layers r 1 , r 2 : R ( d +1) × N → R ( d +1) × N , such that

<!-- formula-not-decoded -->

where X ⊆ R d is a compact set.

We present the construction of r 1 , r 2 , p, q directly as follows. For any t ∈ [ N ] , we choose

<!-- formula-not-decoded -->

where V ∈ R ( d +1) × N , v ( t ) is the t -th column of V , A = ( ˜ A 0 ) ( I k I d -k -1 d -k 0 ) , and z 0 = max t ∈ [ N ] ,i ∈ [ d ] sup X ∈X N ∣ ∣ ( ˜ A ˜ r ( t -1) ( X ) + ˜ B x ( t ) + ˜ c ) i ∣ ∣ . By the choice of z 0 , for any t ∈ [ N ] , we have

<!-- formula-not-decoded -->

We next prove that our construction satisfies the following property by induction

<!-- formula-not-decoded -->

For t = 1 , we have

<!-- formula-not-decoded -->

□

<!-- formula-not-decoded -->

where the second-to-last equality follows from the choice of z 0 .

Assuming the induction hypothesis holds for t -1 , we have

<!-- formula-not-decoded -->

With the above align, we further obtain

<!-- formula-not-decoded -->

where the last equality follows from the choice of z 0 .

Thus, we have demonstrated that our construction satisfies (B.7).

By the definition of q , for any t ∈ [ N ] we have

<!-- formula-not-decoded -->

Thus, our construction satisfies (B.6).

Applying (B.6) to any SMRNN ˜ r ˜ θ = ˜ q ◦ ˜ r L ◦· · ·◦ ˜ r 1 ◦ ˜ p with depth L and width W and setting d = W , we know that there exist 2 L recurrent layers r 1 , . . . , r 2 L and 2 L linear maps p 1 , . . . , p L , q 1 , . . . , q L such that for any X ∈ [0 , 1] d x × N

<!-- formula-not-decoded -->

By setting q = ˜ qq L , p = p 1 ˜ p , ¯ r 2 l +1 = r 2 l +1 p l +1 q l for l ∈ [ L -1] , and r θ = q ◦ r 2 L ◦ ¯ r 2 L -1 ◦ r 2 L -2 ◦ · · · ◦ r 4 ◦ ¯ r 3 ◦ r 2 ◦ r 1 ◦ p , we obtain

<!-- formula-not-decoded -->

It is clear that r θ has depth 2 L and width W +1 .

Finally, we analyze the sparsity, for l ∈ [ L -1] ,

<!-- formula-not-decoded -->

## B.8 Proof of Lemma 13

We first present the following lemma and definitions of Vapnik-Chervonenkis dimension and Pseudodimension of a real-valued function class .

Lemma 17 (Lemma 17 in Bartlett et al. [2019]) . Suppose W ≤ M and let P 1 , . . . , P M be polynomials of degree at most D in W variables. Then we have

<!-- formula-not-decoded -->

Definition 6 (VC-dimension) . Let F : R d x → R be a class of real-valued functions. The VCdimension of F , denoted as VCdim( F ) , is the largest integer m ∈ N for which there exist points ( x 1 , . . . , x m ) ∈ R d x × m such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (B.8), (B.9), and (B.10), we obtain

<!-- formula-not-decoded -->

where the first inequality follows from k +1 ≤ W and ∥ ˜ c l + z 0 1 W ∥ 0 ≤ ∥ ˜ c l ∥ 0 + W .

## B.7 Proof of Lemma 12

Our covering number bound for RNNs is consistent with Lemma 12 in Jiao et al. [2024], which itself is based on Theorem 7 in Bartlett et al. [2019]. For completeness, we provide a full proof of the covering number bound for RNNs.

Proof of Lemma 12. By Lemma 9 in Jiao et al. [2024], we have

<!-- formula-not-decoded -->

Then it follows that

<!-- formula-not-decoded -->

By Theorem 7 in Bartlett et al. [2019], for any W,L,K ≥ 0 we have

<!-- formula-not-decoded -->

Combining (B.11) with (B.12) yields the desired bound and completes the proof.

□

□

Here, sgn( · ) is the sign function defined as

<!-- formula-not-decoded -->

Furthermore, the quantity

<!-- formula-not-decoded -->

is referred to as the growth function of the function class F .

Definition 7 (Pseudo-dimension) . Given a real-valued function class F : R d x → R , the pseudodimension, denoted as Pdim( F ) is the largest m ∈ N for which there exist ( x 1 , . . . , x m ) ∈ R d x × m and ( y 1 , . . . , y m ) ∈ R m such that for any ( b 1 , . . . , b m ) ∈ { 0 , 1 } m , there exists f ∈ F such that for any i ∈ [ m ] :

<!-- formula-not-decoded -->

Proof of Lemma 13. We prove this bound in three steps. First, we reformulate the problem as an inequality depending only on the VC-dimension, using the relationships between the pseudodimension, the VC-dimension, and the covering number. Second, we adapt the proof of Theorem 7 in Bartlett et al. [2019] to establish a bound for the growth function of the prediction function class derived from a sparse neural network with a fixed zero-parameter configuration. Finally, we account for all possible sparse structures to derive the overall VC-dimension bound for the prediction function class.

## • Step 1: From Covering Number to VC-dimension.

By Theorem 12.2 in Anthony and Bartlett [2009], for any ϵ ∈ (0 , K ) , we have

<!-- formula-not-decoded -->

If n &lt; Pdim ( F N ) , then

<!-- formula-not-decoded -->

On the other hand, if n ≥ Pdim ( F N ) , then

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

By Theorem 14.1 in Anthony and Bartlett [2009], there exists a function class F N +1 generated from the SFNN with width W , length L +1 and sparsity s +1 such that

<!-- formula-not-decoded -->

In the remainder of the proof, we restrict our analysis to the case of N for simplicity. The results obtained can be directly extended to the case of N +1 .

## • Step 2: Cardinality Bound for SFNNs with fixed sparsity structure.

First, we formally define the function class generated from the SFNN with a fixed sparsity structure as follows:

<!-- formula-not-decoded -->

Here, M = ∑ L -1 l =1 W l + ∑ L -1 l =0 W l W l +1 is the total number of parameters in the SFNN and ξ ∈ { 0 , 1 } M is a binary vector satisfying ∥ ξ ∥ 0 = s . Each entry of ξ corresponds to an entry of θ in the SFNN: if ξ i = 1 , then θ i is treated as a free parameter; otherwise, if ξ i = 0 , θ i is fixed to zero. Thus, F ξ N represents the subclass of F N constrained by the sparsity pattern specified by ξ .

In this step, we consider a function class F ξ N with a fixed parameter ξ . Since F ξ N is a parameterized function class whose parameter vector has M -s zero entries, any function in F ξ N can be represented as g ( x , θ ′ ) , where θ ′ ∈ R s . Given m ≥ s and x 1 , . . . , x m , our goal is to derive an upper bound for the cardinality of the following set:

<!-- formula-not-decoded -->

Recall that the ReLU activation function σ ( x ) = max { x, 0 } is a piecewise-defined function. Consequently, g ( x , θ ′ ) is also a piecewise-defined function with respect to θ ′ . To apply Lemma 17, we partition R s into disjoint regions S = { I 1 , . . . , I D } , where I i ∩ I j = ∅ for i = j and i, j ∈ [ D ] . Within each region I i , for all j ∈ [ m ] , the function g ( x j , θ ′ ) reduces to a polynomial function. We denote the number of free parameters in the first l layers by s l ξ . We construct this partition recursively with the goal for S l to satisfy the following properties for any l ∈ [ L ] :

̸

<!-- formula-not-decoded -->

2. For any I ∈ S l -1 , w ∈ [ W l ] , and j ∈ [ m ] , the input to the w -th neuron at level l is a polynomial function f I,w,j of degree at most l , which depends on s l ξ variables.

We first select S 0 = R s . The input to any neuron at the first layer is an affine function of degree at most 1, depending on at most s 1 ξ variables. Consequently, S 0 satisfies both required properties.

Given S 0 , . . . , S n -1 that satisfy the two properties, we proceed to construct S n . For any region I ∈ S n -1 . Since f I,w,j is a polynomial function of degree at most n for each w ∈ [ W n ] and j ∈ [ m ] , we derive the following inequality by applying Lemma 17.

<!-- formula-not-decoded -->

This result implies that we can partition I into at most 2(2 eW l mL/ ( s n ξ )) s n ξ disjoint regions. Within each region, all these polynomials maintain the same sign. Consequently, for any j ∈ [ m ] the output of any neuron at the n -th level remains a polynomial of degree at most n .

We define S n to be the set of all disjoint regions generated from all I ∈ S n -1 . Since each layer connecting the n -th and ( n +1) -th layers can increase the degree by at most one, for any j ∈ [ m ] , the input to any neuron at the ( n +1) -th layer is a polynomial of degree at most n +1 , depending on at most s n +1 ξ variables. Thus, S n satisfies both of the required properties. By induction, we have

<!-- formula-not-decoded -->

where W ′ = max { W,d y } . As implied by Lemma 17, the cardinality of any partition in S L -1 is bounded by 2 ( 2 eW ′ ml/s L ξ ) s L ξ . By AM-GM inequality, we obtain

<!-- formula-not-decoded -->

- Step 3: Obtain bound on VCdim ( F N ) .

In this step, we consider all possible fixed sparsity structures ξ , of which there are at most ( M s ) ≤ M s . Summing (B.15) over all such structures yields, for any ( x 1 , . . . , x m ) ∈ R d x × m ,

<!-- formula-not-decoded -->

Let ξ max be the sparsity structure that attains the maximum in (B.16):

<!-- formula-not-decoded -->

The upper bound in (B.16) can thus be expressed as M s 2 L ( 2 eW ′ m ∑ L l =1 l ∑ L l =1 s l ξ max ) ∑ L l =1 s l ξ max .

Choosing m as VCdim( F N ), we have

<!-- formula-not-decoded -->

By Lemma 18 in Bartlett et al. [2019], we obtain

<!-- formula-not-decoded -->

Finally, to ensure the completeness of this proof, we need to show VCdim ( F N ) ≥ s . This result follows directly from Theorem 3 in Bartlett et al. [2019].

Combining (B.13), (B.14), and (B.17), we obtain the desired bound.

## B.9 Proof of Lemma 14

This proof is adapted from the proof of Lemma 9 in Jiao et al. [2024]. As a first step, we demonstrate that for any recurrent layer r : R W × N → R W × N satisfying r ( t ) ( X ) = σ ( Ar ( t -1) ( X )+ B x ( t ) + c ) , there exists an SFNN ¯ f with width (2 N -1) W and length N +1 , such that

<!-- formula-not-decoded -->

Here, X ∈ R W × N . We construct ¯ f as ¯ f = f N +1 ◦ f N ◦ · · · ◦ f 1 with the parameters defined as follows

<!-- formula-not-decoded -->

□

<!-- formula-not-decoded -->

In these block matrices, each block is of size W × W . ¯ A 1 has 2 N -1 rows and N columns of blocks. For j = 2 , . . . , N , ¯ A j has 2 j -4 identity matrices in the upper left corner and 2 N -2 j identity matrices in the lower right corner, and ¯ b j has (2 j -2) W zeros above c . ¯ A N +1 has N rows and 2 N -1 columns of blocks.

We verify that this construction satisfies (B.18) through direct calculation.

<!-- formula-not-decoded -->

This derivation relies on the property σ ( x ) -σ ( -x ) = x , which is applied sequentially across the last three equalities. Thus, our construction for ¯ f satisfies (B.18).

Next, we extend this construction to a complete SRNN r θ = q ◦ r L ◦ · · · ◦ r 1 ◦ p with width W and depth L . For an input sequence X ∈ [0 , 1] d x × N , we can find SFNNs ¯ f 1 , · · · , ¯ f L with width (2 N -1) W and length N +1 , as well as layers ¯ p and ¯ q , such that

<!-- formula-not-decoded -->

Here, the layer ¯ p is constructed to map X to the stacked outputs of p applied to each input x ( t ) , and the final layer ¯ q applies q to the N -th segment of its input vector. Choosing ¯ f = ¯ q ◦ ¯ f L ◦· · ·◦ ¯ f 2 ◦ ¯ f 1 ◦ ¯ p completes the construction.

Finally, we analyze the sparsity

<!-- formula-not-decoded -->

In conclusion, we obtain

<!-- formula-not-decoded -->

## C GDP Growth Analysis

Analyzing GDP growth is a cornerstone of economic research, providing crucial insights into an economy's overall health and trajectory. Given its capacity to effectively capture and forecast upside and downside economic risks, QR has become an indispensable tool in GDP growth analysis [Adrian et al., 2019].

To further evaluate the practical advantages of RNN-based QR estimators under potentially nonstationary settings, we conduct an empirical analysis using real GDP data 4 . We compare the out-of-sample prediction performance of RNN-, SRNN-, FNN-, and QRF-based estimators for one-quarter-ahead GDP forecasting. For all models, the input sequence length is fixed at N = 4 , corresponding to four consecutive quarterly GDP observations x t -3 to x t , representing one year of lagged values. The target variable y t is the GDP level at time t +1 . All models were implemented according to the specifications detailed in Section 4.

The dataset is partitioned chronologically, with the most recent 30% reserved for testing. For all NN-based estimators, the preceding 70% is used for training and validation. During training, 20% of this subset is held out for validation, and early stopping is applied based on the validation check loss.

4 We utilize U.S. GDP growth data from April 1947 to December 2024, accessible at https://fred. stlouisfed.org/series/A191RL1Q225SBEA .

□

All models are trained by minimizing the empirical check loss and evaluated on the test set using the same metric.

Table 2 presents the mean empirical check loss for each model at quantile levels τ = 0 . 1 , 0 . 25 , 0 . 5 , 0 . 75 , and 0 . 9 . The results indicate that the RNN- and SRNN-based estimators consistently outperform both the FNN and QRF methods across most quantile levels. Furthermore, the SRNN achieves predictive accuracy comparable to that of the standard RNN while inducing sparsity. This suggests that the sparse architecture does not compromise performance.

Table 2: Out-of-sample prediction errors at different quantiles for GDP growth analysis.

| Model   | τ = 0 . 1   | τ = 0 . 25   | τ = 0 . 5   | τ = 0 . 75   | τ = 0 . 9   |
|---------|-------------|--------------|-------------|--------------|-------------|
| QRF     | 0 . 849     | 1 . 225      | 1 . 410     | 1 . 246      | 0 . 911     |
| FNN     | 0 . 867     | 1 . 180      | 1 . 773     | 2 . 505      | 2 . 657     |
| RNN     | 0.835       | 1.113        | 1.349       | 1.154        | 0 . 904     |
| SRNN    | 0 . 837     | 1 . 700      | 1.211       | 1.200        | 0.898       |

To assess the robustness of RNN-based estimators in QR in the nonstationary environment, we extend our analysis to include in-sample estimators, enabling a direct comparison of in-sample and out-of-sample performance for both RNN and SRNN methods. Adrian et al. [2019] showed that GDP growth volatility is primarily driven by lower quantiles, while upper quantiles remain relatively stable over time. Motivated by this, we focus our comparison on quantile levels τ = 0 . 05 , 0 . 1 , and 0 . 25 , where tail behavior is most prominent. For the in-sample predictions, the full dataset is used for training and validation. The comparative results, shown in Figure 3, reveal a striking alignment between in-sample and out-of-sample quantile estimates. In these figures, the x -axis represents time, while the y -axis denotes the GDP growth rate. This consistency is particularly noteworthy, given that major financial crises (e.g., 2007-2009), which represent substantial tail events, are absent from the data used for out-of-sample estimation. These findings highlight the reliability and generalizability of RNN-based QR methods, even under adverse and previously unseen market conditions.

<!-- image -->

(a) RNN

(b) SRNN

Figure 3: In-sample and out-of-sample comparison for RNN and SRNN models.

## D Hyperparameter Sensitivity Analysis

In this section, we conduct additional experiments varying network depth L , width W , and pruning ratio on Model 1 in Section 4 under the t 2 . 25 distribution to assess hyperparameter sensitivity.

We first vary L ∈ { 2 , 3 , 4 , 5 , 6 , 7 } and W ∈ { 16 , 32 , 64 , 128 , 256 , 512 , 1024 } , evaluating each ( L, W ) configuration under the same data and training budget. As shown in Table 3, performance generally improves as both the network depth L and width W increase. When either L or W is small, the model tends to underfit, leading to worse MSE. In contrast, when both are sufficiently large, the model achieves its best performance, suggesting that adequate depth and width are essential for capturing the heavy-tailed characteristics of the t 2 . 25 distribution.

Table 3: SRNN MSE across different L and W .

|   L \ W |    16 |    32 |    64 |   128 |   256 |   512 |   1024 |
|---------|-------|-------|-------|-------|-------|-------|--------|
|       2 | 0.056 | 0.054 | 0.054 | 0.053 | 0.06  | 0.065 |  0.097 |
|       3 | 0.055 | 0.051 | 0.051 | 0.05  | 0.054 | 0.058 |  0.073 |
|       4 | 0.055 | 0.05  | 0.052 | 0.05  | 0.051 | 0.052 |  0.07  |
|       5 | 0.05  | 0.05  | 0.05  | 0.052 | 0.05  | 0.05  |  0.057 |
|       6 | 0.06  | 0.05  | 0.053 | 0.05  | 0.049 | 0.049 |  0.056 |
|       7 | 0.05  | 0.05  | 0.05  | 0.05  | 0.049 | 0.049 |  0.056 |

We next study sparsity while fixing other settings as in Section 4. As reported in Table 4, the MSE exhibits a non-monotonic trend with respect to the pruning ratio: moderate pruning improves performance, suggesting an effective regularization effect, whereas aggressive pruning degrades performance-likely due to the removal of critical parameters. These results indicate an interior optimum that balances compactness and representational capacity.

| Pruning ratio   |   0.2 |   0.3 |   0.4 |   0.5 |   0.6 |   0.7 |   0.8 |
|-----------------|-------|-------|-------|-------|-------|-------|-------|
| MSE             | 0.046 | 0.044 |  0.04 | 0.039 | 0.038 | 0.036 | 0.038 |

Table 4: Effect of pruning ratio on SRNN MSE.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately stated the contributions and scope of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Our work focuses on a fixed quantile level. We discuss this limitation of our results and future direction in Section 6.

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

Justification: All the theoretical results in the paper are accompanied by a full set of assumptions, with complete and correct proofs included in the Appendix.

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

Justification: We clearly disclose all information necessary for reproducing the presented experimental results, thereby ensuring their reproducibility.

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

## Answer: [No]

Justification: Our experiments are easily reproducible, and we provide all necessary information for their re-implementation in the paper. As this paper focuses on theoretical analysis, the code is not provided.

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

Justification: This paper provides all details essential for understanding the results in Section 4, Section 5, Appendix C, and Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: This paper reports appropriate information about the statistical significance.

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

Answer: [No]

Justification: Our experiments utilized relatively small datasets with minimal computational requirements that can be easily reproduced on standard personal computers, making detailed specifications of compute resources unnecessary.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed the NeurIPS Code of Ethics and ensured that our paper conforms to it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Due to its theoretical nature, this paper does not discuss societal impacts.

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

Justification: We do not make publicly available any data or models that pose a high risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The datasets used in this study are the DJIA and GDP, both of which are publicly available.

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

Justification: We don't introduce any new assets in this paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
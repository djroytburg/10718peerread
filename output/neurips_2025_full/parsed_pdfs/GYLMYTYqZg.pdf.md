## Low-Precision Streaming PCA

## Sanjoy Dasgupta

University of California San Diego sadasgupta@ucsd.edu

## Shourya Pandey

University of Texas at Austin shouryap@utexas.edu

## Abstract

Low-precision Streaming PCA estimates the top principal component in a streaming setting under limited precision. We establish an information-theoretic lower bound on the quantization resolution required to achieve a target accuracy for the leading eigenvector. We study Oja's algorithm for streaming PCA under linear and nonlinear stochastic quantization. The quantized variants use unbiased stochastic quantization of the weight vector and the updates. Under mild moment and spectral-gap assumptions on the data distribution, we show that a batched version achieves the lower bound up to logarithmic factors under both schemes. This leads to a nearly dimension-free quantization error in the nonlinear quantization setting. Empirical evaluations on synthetic streams validate our theoretical findings and demonstrate that our low-precision methods closely track the performance of standard Oja's algorithm.

## 1 Introduction

Quantization (or discretization) is the mapping of a continuous set of values to a small, finite set of outputs close to the original values; standard methods for quantization include rounding and truncation. The current popularity of training large-scale Machine Learning models has brought a renewed focus on quantization, though its origins go back to the 1800s. Some early examples include least-squares methods applied to large-scale data analysis in the early nineteenth century [Sti86]. In 1867, discretization was introduced for the approximate calculation of integrals [Rie67], and the effects of rounding errors in integration were examined in 1897 [She97]. For an excellent survey and history of quantization, see [GKD ` 22].

In the context of efficient model training, it is natural to ask the following: does training a model require the full precision of 32- or 64-bit representation, or is it possible to achieve comparable performance using significantly fewer bits? Mixed-precision training (using 16 -bit floats with 32 -bit accumulators) is now standard on GPUs and TPUs, yielding 1 . 5 ˆ to 3 ˆ speedups with negligible accuracy loss on large transformers and CNNs [MNA ` 18]. Binary Neural Networks (BNNs), which constrain weights and activations to ˘ 1 , can achieve up to 32 ˆ memory compression and replace multiplications with bitwise operations. This has been shown to approach nearly full-precision ImageNet accuracy with careful training [HCS ` 16].

Theoretical analysis of the effect of low-precision computation on optimization problems has received significant attention [LD19, AGL ` 17, SZOR15, SLZ ` 18, LDX ` 17, ZLK ` 17]. Complementary strategies leverage stochastic rounding to mitigate quantization bias during LLM training. Ozkara et al. [OYP25] present theoretical analyses of implicit regularization and convergence properties of Adam when using BF16 with stochastic rounding, demonstrating up to 1 . 5 ˆ throughput gains and 30% memory reduction over standard mixed precision [OYP25].

## Syamantak Kumar

University of Texas at Austin syamantak@utexas.edu

## Purnamrita Sarkar

University of Texas at Austin purna.sarkar@utexas.edu

Consider the set of values that can be exactly represented in the quantization scheme, which we call the quantization grid . For example, fixed-point arithmetic [Yat09] uses linear quantization (LQ), where the quantization grid consists of points spaced uniformly at a distance δ (also denoted by quanta ). [LDX ` 17] analyze Stochastic Gradient Descent (SGD)-based optimization algorithms for LQ, and [SYK21] perform Learned Image Compression (LIC) under 8 -bit fixed-point arithmetic. Nonlinear quantization (NLQ) grids with logarithmic spacing are also widely used [KWW ` 17, NTSW ` 22, XLY ` 24, YIY21, ZMK22, ZWG ` 23] in low-precision training.

To illustrate the importance of the quantization scheme, consider the example of rounding, where each input is mapped to the value in the quantization grid closest to it. The following toy iterative optimization algorithm demonstrates that rounding can cause the solution to remain stuck at the initial vector. Consider the update scheme w t ' w t ´ 1 ` η g t , followed by rounding each coordinate of w t . Here η is the learning rate and g t is the gradient evaluated at time t . Suppose max i } g t p i q} ď 1 . Assume that w 0 is quantized using the LQ scheme and that η ă δ { 2 . For any coordinate i , we have | w 1 p i q ´ w 0 p i q| ' η ¨ | g t p i q| ď η . Since η ă δ { 2 , after rounding, w 1 p i q is mapped back to the original quantized value w 0 p i q , i.e., w 1 ' w 0 . As a result, the algorithm fails to make progress. We address this issue by using stochastic rounding . In this approach, each value is randomly mapped to one of the closest two quanta with the probabilities chosen such that the quantized value is unbiased.

Principal Component Analysis. PCA [Pea01, Zie03] is a dimension-reduction technique that extracts the directions of largest variance from the data. Suppose we observe n independent samples X i P R d from a zero-mean distribution with covariance Σ . PCAseeks a unit vector v 1 that maximizes variance, which is any eigenvector of Σ associated with its largest eigenvalue λ 1 . Under mild tail conditions on the X i , the top eigenvector ˆ v of the sample covariance 1 n ř n i ' 1 X i X J i is a nearly rate-optimal estimator of the true principal direction v 1 [Wed72, JJK ` 16, Ver10].

Despite its statistical appeal, constructing the covariance matrix itself takes Ω p nd 2 q time and Ω p d 2 q space, which is prohibitive for large d and n . A popular remedy is Oja's algorithm [Oja82], a single-pass streaming algorithm inspired by Hebbian learning [Heb49]. Starting from a (random) unit vector u 0 , for each incoming datum X i the algorithm performs the update

<!-- formula-not-decoded -->

Here, η ą 0 is the learning rate which may vary across iterations. The batched version of Oja's method partitions the data into b batches B 1 , . . . B b of size n { b each and replaces the above update with the averages of the gradients within a batch:

<!-- formula-not-decoded -->

The entire procedure completes in O p nd q time and uses O p d q space. The scalability and simplicity of Oja's algorithm have motivated extensive analysis across statistics, optimization, and theoretical computer science [JJK ` 16, AZL17, CYWZ18, YHW18, HW19, MP22, Mon22, KS24b, KS24a, JKL ` 24, KPS25]. These works establish precise convergence rates, error bounds under various noise models, and extensions to sparse or dependent-data settings. When operating with β bits, the overall complexity for streaming PCA (and that of the batched variant) grows polynomially with β (for fixed n, d ); Table 1 gives evidence towards this fact.

Table 1: Benchmarking runtimes 1 for the experiment described in Appendix F.1

|             | 64 bits              | 16 bits                  |
|-------------|----------------------|--------------------------|
| Runtime (s) | 0 . 0274 ˘ 0 . 00136 | 0 . 000398 ˘ 0 . 0000235 |

## Our Contributions.

1. We present a general theorem for streaming PCA with iterates that are composed of independent data (as in standard Oja's algorithm) and a noise vector that is mean zero, conditioned on the filtration up until now, which may be of independent interest .
2. We obtain new lower bounds for estimating the principal eigenvector under both quantization schemes. The quantization error depends linearly in the dimension d for the linear scheme and dimension-independent (up to logarithmic factors) for the non-linear scheme.

1 The experiments were conducted by representing the data and intermediate variables in double precision (64 bits) and half precision (16 bits) datatypes.

3. Our batched version of Oja's algorithm matches the lower bounds under both quantization schemes. The quantization error of the batched version with logarithmic quantization is nearly dimension-free . We also provide a procedure to make the failure probability of the algorithm arbitrarily small.

Section 2 introduces the problem setup and defines the linear and logarithmic quantization schemes. Section 3 presents the main results, including lower and upper bounds for Oja's algorithm with and without batching for both quantization schemes. Section 4 provides proof sketches, Section 5 reports experimental results, and Section 6 concludes the paper.

## 2 Problem Setup and Preliminaries

We use r n s to denote t i P N | i ď n u . Scalars are denoted by regular letters, while vectors and matrices are represented by boldface letters. I P R d ˆ d represents the d -dimensional identity matrix. ∥ . ∥ denotes the ℓ 2 euclidean norm for vectors and ∥ . ∥ op denotes the operator norm for matrices. For a, b P R , we write a À b if and only if there exists an absolute constant C ą 0 such that a ď Cb. ˜ O, ˜ Ω represent order notations that hide logarithmic factors. S d ´ 1 is the set of unit vectors in R d .

We operate under the following assumption on the data distribution.

Assumption 1. t X i u i Pr n s are mean-zero iid vectors in R d drawn from distribution D supported on the unit ball. Let Σ : ' E X ' D ' XX J ‰ denote the data covariance, with eigenvalues λ 1 ą λ 2 , ¨ ¨ ¨ , λ d and corresponding eigenvectors v 1 , v 2 , ¨ ¨ ¨ v d . We assume D V , M ą 0 such that

<!-- formula-not-decoded -->

Assumption 1 enforces standard moment bounds used to analyze PCA in the stochastic setting. Similar assumptions are also used in [HP14, SRO15, Sha16a, Sha16b, JJK ` 16, AZL17, BDWY16, XHDS ` 18] to derive near-optimal sample complexity bounds for Oja's rule. We assume a bounded range for ease of analysis, and it can be generalized to subgaussian data (see [LSW21, KS24a, Lia21]).

The misalignment between the estimated top eigenvector u and the true eigenvector u 1 is measured using the principal angle between the two vectors. The sin-squared error between any two non-zero vectors u , v is defined as sin 2 p u , v q ' 1 ´ p u J v q 2 } u } 2 } v } 2 .

## 2.1 Quantization Schemes and Rounding

Linear quantization : Let δ ą 0 , and let β ą 0 be the number of bits used by the low-precision model to represent numbers. A linear quantization scheme uniformly spaces on the real line. Define

<!-- formula-not-decoded -->

We call δ the quantization gap for the quantization grid Q L .

Logarithmic (non-linear) quantization : The error resulting from rounding an element x in the range r´ δ 2 β ´ 1 , δ p 2 β ´ 1 ´ 1 qs using the linear quantization scheme is an additive δ . Here, we present a well-known non-linear quantization scheme where the error scales with the quantized value.

The quantization grid Q NL in the logarithmic quantization scheme with parameters ζ and δ 0 is defined as follows: Let q 0 ' 0 and q i ` 1 ' p 1 ` ζ q q i ` δ 0 @ i P N . Then,

<!-- formula-not-decoded -->

where N ' 2 β ´ 1 . Henceforth, non-linear quantization refers to logarithmic quantization.

These two quantization schemes are widely used in practice [YIY21, DSLZ ` 18, LDS19, DMM ` 18]. Our analysis of the logarithmic scheme lifts to floating-point quantization commonly used in lowprecision computing. The Floating Point Quantization (FPQ) is a widely adopted variation on the Logarithmic quantization scheme, where adjacent values in the quantization grid are multiplicatively close. FPQ and other logarithmic schemes are used in most modern programming languages such as C++, Python, and MATLAB, and broadly standardized (IEEE 754 floating-point standard [Kah96]).

Another quantization scheme for low-precision training is the power-of-two quantization [PRSS ` 22], which rounds to the nearest power of two. All these schemes are similar in principle to our scheme; Lemma A.9 in the appendix establishes a relationship between the distance of a vector from its

quantization under NLQ. This Lemma applies to FPQ and to most other logarithmic quantization schemes. Our proofs can be modified to work with any such scheme.

Stochastic Rounding. A natural quantization scheme is to round x to any of the closest values in the quantization grid. We can randomize to ensure that the expectation of the quantized number is equal to x . For this, we use a stochastic rounding scheme. For any x within the range of the quantization grid Q , suppose u and ℓ are adjacent values in Q such that ℓ ď x ă u . Define

<!-- formula-not-decoded -->

where p p x q : ' p x ´ ℓ q{p u ´ ℓ q . This choice of probability ensures

<!-- formula-not-decoded -->

## 3 Main Results

## 3.1 Lower Bounds

In this section, we establish worst-case lower bounds for the quantized PCA for both linear and logarithmic quantization schemes under the mild assumption that the quantized vectors under consideration have bounded norm. This assumption is reasonable because (i) gradient-based algorithms and other typical algorithms for PCA are usually self-normalizing, ensuring that the norms of the iterates are controlled, and (ii) the quantized vectors are close to the true vectors in norm.

Lemma 1. [Lower bound for linear quantization] Let d ą 1 and δ ą 0 such that δ 2 d ď 0 . 5 . Let V L denote the set of non-zero quantized vectors w P R d using the linear quantization scheme (3) such that } w } P r 1 { 2 , 2 s . Then, sup v 1 P S d ´ 1 inf w P V L sin 2 p w , v 1 q ' Ω p δ 2 d q .

Lemma 2. [Lower bound for logarithmic quantization] Let d ą 1 and δ 0 , ζ ą 0 such that ζ ă 0 . 1 and δ 2 0 d ă 0 . 5 . Let V NL be the set of non-zero quantized vectors w P R d using the logarithmic scheme (4) such that ∥ w ∥ P r 1 { 2 , 2 s . Then, sup v 1 P S d ´ 1 inf w P V NL sin 2 p w , v 1 q ' Ω p ζ 2 ` δ 2 0 d q .

At first glance, the results of Lemmas 1 and 2 may appear similar. However, the parameter δ 0 is substantially smaller than δ . In Section 3.4, we select optimal values for δ , δ 0 , and ζ given a fixed bit budget β for the low-precision model and show that δ 2 d ' Θ p d 4 ´ β q while ζ 2 ` δ 2 0 d ' ˜ Θ p 4 ´ β q where the tilde hides a log 2 d factor. Hence, the lower bound for the logarithmic quantization scheme is nearly independent of the dimension. The proofs of the lower bounds are deferred to Appendix B.

## 3.2 Quantized Batched Oja's Algorithm

In this section, we present an algorithm that uses stochastic quantization for the batch version of Oja's algorithm (see Eq 2). We start by computing the quantized version w i of the normalized vector u i ´ 1 from the last step. Then, we quantize each X j p X T j w i ´ 1 q and compute the average of the quantized gradient updates. This average gradient is quantized again and added to w i .

## Algorithm 1 Quantized Oja's Algorithm with Batches

Require: Data t X i u i Pr n s , quantization grid Q , learning rate η , number of batches b

(

<!-- formula-not-decoded -->

The final vector that results from the batched Oja's rule (Eq 2) without quantization is

<!-- formula-not-decoded -->

where D i ' ř j P B i X j X T j {p n { b q is the empirical covariance matrix of the i th batch. Since X i are IID and the batches are disjoint, D i are also IID. The key observation for Algorithm 1 is that even with the quantization, the vector u b can be written as

<!-- formula-not-decoded -->

Each Ξ i is a rank-one matrix resulting from the stochastic quantization. Conditioned on an appropriately chosen filtration σ p X 1 , . . . , X i , u 0 , . . . , u i ´ 1 q , Ξ i is mean zero; Algorithm 1 defines quantization variables ξ 1 ,i , ξ a,i , and ξ 2 ,i for all i P r b s . The rank one noise Ξ i is Ξ i : ' p η ξ a,i ` ξ 2 ,i `p I ` η D i q ξ 1 ,i q u T i ´ 1 . Since the stochastic updates are conditionally unbiased (equation (6)),

<!-- formula-not-decoded -->

Similarly E r ξ a,i | D 1 , . . . , D i , w 0 , . . . , w i ´ 1 s ' 0 , as it can be written as

<!-- formula-not-decoded -->

## 3.3 Guarantees for Low-Precision Oja's Algorithm

Before presenting our main result, we present a general result that can apply to other noisy variants of Oja's rule and is of independent interest. The proof is deferred to Appendix Section D. Consider Oja's algorithm on matrices A i P R d ˆ d , such that A i ' η D i ` Ξ i where D i are IID random matrices with E r D i s ' Σ .

Let S i be the set of all random vectors ξ in the first i iterations of the algorithm and F i ´ denote the σ -algebra generated by the random D 1 , . . . , D i and S i ´ 1 . Define the operator E i r . s : ' E r . | F i ´ s . We assume the noise term Ξ i is measurable with respect to the filtration F i ´ and unbiased conditioned on F i ´ , i.e., E i r Ξ i | F i ´ s ' 0 d ˆ d .Let V 0 , ν, M , κ, and κ 1 be non-negative parameters such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 1. Let d, n, b P N and u 0 ' N p 0 , I d q . Let η : ' α log n b p λ 1 ´ λ 2 q be the learning rate where α is chosen to satisfy Lemma A.2, and suppose max p bη 2 M 2 log p d q , bκ 2 log d q ' O p 1 q . Then, with probability at least 0 . 9 , the vector u b from equation 7 satisfies ∥ u b ∥ P r 1 ´ κ 1 , 1 ` κ 1 s and

<!-- formula-not-decoded -->

Remark 1 (Matching the Upper and Lower Bounds) . In the LQ scheme with gap δ , each coordinate of the noise vector ξ is bounded by δ almost surely. In particular, this implies κ ' O p δ ? d q and κ 1 ' O p δ 2 d q (see Appendix Section D) and the resulting error due to quantization matches the lower bound in Lemma 1. In the NLQ scheme with parameters ζ and δ 0 , the i th coordinate of the noise vectors ξ is bounded by ζ | u i | ` δ 0 , where u is the vector being quantized. Since the vectors in consideration are bounded in norm by 1 , this implies κ ' O p ζ ` δ 0 ? d q and κ 1 ' O p ζ 2 ` δ 2 0 d q (see Appendix Section D). The resulting error matches the lower bound in Lemma 2 as long as the output vector has norm in the range r 1 { 2 , 2 s .

Remark 2. Theorem 1 relies on the observation that accumulating the quantization error only b times in Algorithm 1 leads to a smaller sin 2 error. Moreover, choosing an appropriate batch size reduces the variance parameter V 0 by a factor of n { b because of averaging.

Remark 3 (Hyperparameters and eigengap) . The choice of the learning rate η ' α log n n p λ 1 ´ λ 2 q is also present in other works on streaming PCA [HP14, SOR14, Sha16a, Sha16b, AZL17, HNWTW20, JNN19, BDF13] to derive the statistically optimal sample complexity (up to logarithmic factors). If a smaller learning rate η is used (for example, by using an upper bound U on the eigengap λ 1 ´ λ 2 ), then the first error term of Theorem 1 will be larger, leading to a slightly larger sin-squared error. A similar argument applies to the choice of the batch size.

Remark 4 (Known n in the learning rate) . The length of the stream n is an input in Theorem 1, and the learning rate is constant over time. To handle variable learning rates using only constant-rate updates, a standard doubling trick [ACBFS95] can be used. Specifically, the time horizon is divided into blocks that double in size: the k th block has size 2 k ´ 1 and Oja's algorithm run on that block uses a learning rate corresponding to that block's size. When the algorithm run on this block terminates, the older estimate of the top eigenvector run on the previous block is replaced by this new estimate. This scheme effectively simulates a decaying learning rate while keeping the analysis tractable.

## 3.4 Choosing the Optimal Quantization Parameters

To ensure a fair comparison between the linear and logarithmic quantization schemes, we fix a budget β for the total number of bits used by the low-precision model. Moreover, our algorithms require that numbers in, say, p´ 2 , 2 q are representable by the quantization scheme. Therefore, we must ensure that the upper and lower limits of the scheme cover this range.

The largest number representable in the linear quantization scheme is δ p 2 β ´ 1 q and the smallest negative number representable is ´ δ ¨ 2 β . We choose δ ' 2 2 ´ β , which covers the range p´ 2 , 2 q .

To motivate the choice of ζ and δ 0 , we note that the floating point scheme is a discretization of the logarithmic quantization scheme. The parameter δ 0 in the logarithmic scheme represents the smallest representable positive real, which in the FPQ scheme is equal to 4 ¨ 2 ´ 2 βe ´ 1 , where β e is the number of bits used to represent the exponent. The parameter ζ represents multiplicative growth between adjacent quanta and is analogous to 2 ´ β m in the FPQ scheme, where β m is the number of bits to represent the mantissa, and β ' β m ` β e . Assuming ζ ' 2 ´ β m and δ 0 ' 4 ¨ 2 ´ 2 βe ´ 1 , where β m and β e are positive integers, the largest representable number is

<!-- formula-not-decoded -->

To represent numbers in p´ 2 , 2 q , it suffices to ensure β m ě 3 . This allows some freedom to select β m and β e such that the factor κ 1 ' ζ 2 ` δ 2 0 d is minimized. We choose

<!-- formula-not-decoded -->

which is valid as long as β ě max p 8 , log 2 d q and β m ě 3 . We justify this choice in appendix D.3.

With this choice of β e and β m , the parameters ζ and δ 0 satisfy

<!-- formula-not-decoded -->

With this setting, we present two immediate corollaries of Theorem 1 with a fixed budget β . The proofs are deferred to Appendix Section D.

Theorem 2. [Oja's Algorithm with Batches]

1. Suppose Q ' Q L and δ, b satisfy δ ' 2 2 ´ β ' O ´ λ 1 ´ λ 2 α ? d log p n q ¯ and b ' Θ ´ α 2 log 2 p n q p λ 1 ´ λ 2 q 2 ¯ . Then, with probability at least 0 . 9 , the output w b of Algorithm 1 satisfies

<!-- formula-not-decoded -->

2. Suppose Q ' Q NL with ζ and δ 0 as in equation (10) , such that ζ ` δ 0 ? d ' O ´ λ 1 ´ λ 2 α ? d log p n q ¯ , and batch size b ' Θ ´ α 2 log 2 p n q p λ 1 ´ λ 2 q 2 ¯ . Then, with probability at least 0 . 9 , the output w b of Algorithm 1 satisfies

<!-- formula-not-decoded -->

Theorem 3. [Oja's Algorithm]

1. Suppose Q ' Q L , and δ, b satisfy δ ' 2 2 ´ β ' O ´ min ´ λ 1 ´ λ 2 α ? d log p n q , 1 ? dn ¯¯ and b ' n . Then, with probability at least 0 . 9 , the output w n of Algorithm 1 satisfies

<!-- formula-not-decoded -->

2. Suppose Q ' Q NL with ζ and δ 0 as in equation (10) , such that ζ ` δ 0 ? d ă O ´ min ´ λ 1 ´ λ 2 α ? d log p n q , 1 ? dn ¯¯ , and batch size b ' n . Then, with probability at least 0 . 9 , the output w n of Algorithm 1 satisfies

<!-- formula-not-decoded -->

Under linear quantization (LQ), the quantization error term scales as d { 4 β , whereas under nonlinear/logarithmic quantization (NLQ) it is only p β 2 ` log 2 d q{ 4 β . Thus, NLQ achieves a nearly dimension-independent error resulting from quantization, making it especially advantageous in high-dimensional settings.

The errors of Oja's algorithm with batching due to quantization are ˜ O p d 4 ´ β q and ˜ O p 4 ´ β q in the two cases of linear and logarithmic quantization, which are an n factor larger than the corresponding errors without batching. Theorem 2 and 3 show that batching significantly improves the performance under quantization. They further show that the NLQ scheme, when suitably optimized, gives nearly dimension-independent dependence on the quantization error. In comparison, the error resulting from quantization in LQ suffers the most from higher dimensions. In Figure 1 we see that unquantized algorithms (standard and batched) have similar and best performance. See Section 5 for detailed experimental evidence supporting the theory.

Figure 1: We study the effect of different quantization strategies on mean sin 2 -error over 10 runs as the number of samples grows on the x axis. Standard uses b ' n batches whereas Batched uses b ' 10 batches. Among the quantization algorithms, we see that in sin 2 error, Standard LQ &gt; Batched LQ and Standard NLQ &gt; Batched NLQ.

<!-- image -->

Remark 5. Theorems 2 and 3 are stated with a constant probability of success. In Section 3.5 we provide a quantized probability boosting algorithm (Algorithm 2) which boosts the probability of success from a constant to 1 ´ θ for arbitrary θ P p 0 , 1 s .

## 3.5 Boosting the Probability of Success

Quantized Oja's algorithm produces an estimate whose error is within the target threshold with constant success probability. This section addresses this gap by presenting a standard probability boosting framework to let the failure probability θ be arbitrarily small.

Algorithm 2 begins by partitioning m data t X i u i Pr m s into r ' Θ p log 1 { θ q disjoint batches of size n each and runs the algorithm A on each batch. The output vectors t u i u i Pr r s are then aggregated using the boosting procedure SuccessBoost. This procedure looks for a popular vector u i close to at least half of the other vectors and returns any such vector. A general argument for SuccessBoost for arbitrary distance metrics can be found in [KLL ` 23, KS24a].

## Algorithm 2 Probability Boosted Oja's Algorithm

```
Require: Data t X i u i Pr m s , algorithm A , quantization grid Q L p ϵ q , failure probability θ , error ϵ 1: r Ð r 20 log p 1 { θ q s , n Ð t m { r u 2: for i ' 1 to r do 3: B i Ðtp i ´ 1 q n, p i ´ 1 q n ` 1 , . . . , p i ´ 1 q n ` n u 4: u i Ð A pt X j u j P B i q 5: procedure ˜ ρ ( x , y ) 6: return Q ` sin 2 p x , y q , Q L p ϵ q ˘ 7: procedure SuccessBoost( t u i u i Pr r s , ρ, ϵ ) 8: for i ' 1 to r do 9: c i Ð|t j P r r s : ρ p u i , u j q ď 5 ϵ u| 10: if c i ě 0 . 5 r then 11: return u i return K 12: ¯ u Ð SuccessBoost pt u i u i Pr r s , ˜ ρ, ϵ q 13: return ¯ u
```

We use a quantized version ˜ ρ as a proxy for the sin 2 error in the SuccessBoost procedure. ˜ ρ uses the linear quantization grid

<!-- formula-not-decoded -->

where the gap ϵ is set to the upper bound on the error guaranteed by Theorem 2 or Theorem 3 depending on the algorithm A in use.

Standard arguments for SuccessBoost apply when the error ˜ ρ is either computed exactly. The difference in our setting is that we the error function ˜ ρ is only approximately a metric and does not behave as intended if the computed value is outside the quantization range. To highlight the second point, consider the unbounded quantization grid

<!-- formula-not-decoded -->

With this grid, ˇ ˇ ˜ ρ p x , y q ´ sin 2 p x , y q ˇ ˇ is bounded by O p ϵ q almost surely. We extend the argument to show that Lemma 3 holds even with the bounded grid Q L p ϵ q : ' Q L p ϵ, β q , which truncates values outside the range r´ 2 β ´ 1 ϵ, p 2 β ´ 1 ´ 1 q ϵ s to its endpoints. This requires a modest assumption that the number of bits β ě 4 , which is already assumed when optimizing the parameters in Section 3.4.

Lemma 3. Let d ą 1 , β ě 4 , ϵ P p 0 , 0 . 75 q , θ P p 0 , 1 q , and r ' r 20 log p 1 { θ q s . Let v P R d be a unit vector and u 1 , u 2 , . . . , u r be independent random vectors such that Pr ` sin 2 p u i , v q ď ϵ ˘ ě 0 . 9 . Let ˜ ρ be the function defined in Algorithm 2 with the quantization grid Q L p ϵ, β q . Then, the vector ¯ u : ' SuccessBoost ` t u i u i Pr r s , ˜ ρ, ϵ ˘ satisfies

<!-- formula-not-decoded -->

The proof of Lemma 3 is in Appendix E.

Algorithm 2 has a constant overhead in the error compared to algorithm A . The probability of success is amplified from 0 . 9 to 1 ´ θ . The number of samples needed to achieve the same error (up to constant factors) as A blows up only by a multiplicative factor Θ p log 1 { θ q . If algorithm A runs in O p nd q time and O p d q space, which is the case for Oja's algorithm and its batch variants, then Algorithm 2 takes O p nd log p 1 { θ q ` d log 2 p 1 { θ qq time and O p d log p 1 { θ qq space.

## 4 Proof Techniques

Our proof of Theorem 1 has three main parts. Let Z b ' ś 1 i ' b p I ` A i q where A i : ' η D i ` Ξ i as described in equation (7). First, note that the sin-squared error can be written as 1 ´ ` u J b v 1 ˘ 2 ' } V K V K J Z b u 0 } 2 {} Z b u 0 } 2 . Using the one-step power method result shown in Lemma 6 from [JJK ` 16], for a fixed θ P p 0 , 1 q , with probability atleast 1 ´ θ ,

<!-- formula-not-decoded -->

This makes our strategy clear for the subsequent proof. We bound the numerator by bounding E r Tr p V K J Z b Z J b V K qs and applying Markov's inequality. For the denominator, we lower bound ∥ ∥ Z J b v 1 ∥ ∥ by decomposing it as

<!-- formula-not-decoded -->

and upper-bounding } Z b ´p I ` η Σ q b } . For both the numerator and the denominator, we use the following intermediate bound, which controls the p p, q q -norm for a random matrix X defined as ~ X ~ p,q ' E r ∥ X ∥ q p s 1 { q , where ∥ X ∥ p represents the Schattenp norm.

Proposition 1. Let the noise term Ξ , defined in (9) , be bounded as ∥ Ξ ∥ ď κ almost surely. Under Assumption 1, for η P p 0 , 1 q , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Proposition 1 adapts the arguments for matrix product concentration from [HNWTW20]. which also include results for a general sequence of matrices adapted to a suitable filtration.

From Proposition 1 with q ' 2 , p ' 2 ` 2 log d , we get

<!-- formula-not-decoded -->

This allows us to control the lower bound via Markov's inequality, by substituting in equation (13).

To control the numerator, we show the following result (Lemma 4),

Lemma 4. Let Assumption 1 hold and let γ : ' 2 p η 2 M 2 ` κ 2 q . If bγ p 1 ` 2 log p d qq ď 1

, then

<!-- formula-not-decoded -->

The proof of Lemma 4 follows Lemma 10 of [JJK ` 16] to show, for β t : ' E r Tr p V K J Z t Z J t V K qs ,

<!-- formula-not-decoded -->

At this step, we deviate from their proof and appeal to Proposition 1 for bounding E r} Z t ´ 1 } 2 s . Setting ϕ : ' p 1 ` ηλ 1 q 2 , γ : ' 2 p η 2 M 2 ` κ 2 q and p : ' max p 2 , a 2 log d {p bγ qq , we get

<!-- formula-not-decoded -->

Unrolling the recursion and using this bound proves Lemma 4. The proof of Theorem 1 then follows from the one-step power method guarantee in equation 12. Detailed proofs are in Appendix C.

## 5 Experiments

Figure 2: Variation of sin 2 -error with (a) sample size, (b) dimension, and (c) quantization bits.

<!-- image -->

We generate n samples from a d dimensional distribution selected by choosing a random orthonormal matrix Q , setting Σ : ' Q Λ Q J for Λ ii : ' i ´ 2 and sampling datapoints i.i.d from N p 0 , Σ q . We compare six variants of Oja's algorithm for estimating v 1 , the leading eigenvector of Σ . The baseline is the standard full precision update in Eq 1 ( standard ). standard\_LQ and standard\_NLQ use Algorithm 1 with b ' n and Q p ., Q L q and Q p ., Q NL q respectively. The batched variant follows Eq 2 with b ' 100 (for Figures 2a and 2b) and b ' 25 (for Figure 2c) equal-sized batches. Finally, we combine the batched schedule by running Algorithm 1 with Q p ., Q L q ( batched\_LQ ) and with Q p ., Q NL q ( batched\_NLQ ). All experiments were done on a personal computer with a single CPU.

The low-precision methods rely on Eq 10 to choose quantization parameters for a target number of bits β ' 8 . Given the dimension d , these routines compute a uniform quantization step δ uni , an exponential step δ exp , and a multiplicative-growth factor α exp to cover a fixed dynamic range. Each configuration is run for R ' 100 independent trials. In Experiment 1 we fix d ' 100 and vary n P t 1000 , 2000 , 3000 , 4000 , 5000 u ; in Experiment 2 we fix n ' 5000 and vary d P t 100 , 200 , 300 , 400 , 500 u . Every trial begins from a random Gaussian vector normalized to unit length. We set the learning rate to η ' 2 ln p n q n p λ 1 ´ λ 2 q for the standard method and to η ' 2 ln p n q b p λ 1 ´ λ 2 q for the batched methods. Upon completion we record the final excess error sin 2 p ˆ w , v 1 q ' 1 ´p ˆ w J v 1 q 2 and report the mean. The first two use the log -log scale and the third uses the log scale for the y -axis.

As shown in Figure 2a, all methods improve as the number of samples n grows except standard\_LQ and standard\_NLQ . The errors of these two methods, as expected from Theorem 3, grow linearly with n . In contrast, the batched\_LQ and batched\_NLQ 's quantization errors do not depend linearly on n and improve over the standard counterparts. Figure 2b shows how the error varies with the data dimension d . Since V grows mildly with d , for our data distribution, all methods other than standard\_LQ and batched\_LQ do not grow with d . These two methods grow linearly with d , confirming our theoretical findings in the first results under Theorems 2 and 3. Finally, Figure 2c compares the errors with the bit budget β . As β increases from 4 to 12, linear and logarithmic quantization schemes steadily reduce their error and converge toward the full-precision result by β ' 12 . The batched quantizers require only 6-8 bits to achieve comparable performance to the full-precision batched error, whereas the standard\_LQ and standard\_NLQ need at least 10 bits to reach the same performance. The variability of the full precision methods arises from the randomness of initializations. Appendix F provides experiments on additional real-world and synthetic data.

## 6 Conclusion

We study the effect of linear (LQ) and logarithmic (NLQ) stochastic quantization on Oja's algorithm for streaming PCA. We obtain new lower bounds under both quantization settings and show that the batch variant of our quantized streaming algorithm achieves the lower bound up to logarithmic factors. The lower bound on the quantization error resulting from our logarithmic quantization is dimension-free. In contrast, the quantization error under the LQ scheme depends linearly in d , which is problematic in high dimensions. We also show a surprising phenomenon under quantization: the quantization error of standard Oja's algorithm scales with n under both NLQ and LQ schemes, while batch updates with a small batch size does not incur this dependence. These theoretical observations are validated via experiments. A limitation of our analysis is that we estimate the first principal component only. Deflation-based approaches (see e.g. [JKL ` 24, Mac08, SJS09]) provide an interesting future direction for extending this work for retrieving the top k principal components.

## Acknowledgments and Disclosure of Funding

We gratefully acknowledge NSF grants 2217069, 2019844, and DMS 2109155.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We prove theoretical results on the effect of quantization on streaming PCA. The abstract and introduction summarize the contributions and put them in the broader scope of low-precision computation.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the conclusion, we state that our work is about estimating the first principal component. Extending to k principal components is part of future work.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best

judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide complete proofs in the appendix.

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

Justification: Our experimental section has all the parameters of the experiments for reproducibility.

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

Justification: We will submit the code with the supplementary material. We only provided synthetic experiments.

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

Justification: We specify the learning rate, data-generating distributions, and other parameters clearly in the experiments section.

Guidelines:

- The answer NA means that the paper does not include experiments.

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide error bars for the figures in the experimental section.

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

Justification: All experiments were done on our personal device with a single CPU, which we mention in the experimental section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our work conforms with the NeurIPS code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is primarily theoretical and has no societal impact.

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

Justification: Our work is primarily theoretical, and we do not release data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use any existing assets - our contributions are primarily theoretical.

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

Justification: We do not release new assets - our contributions are primarily theoretical.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work is primarily theoretical - we do not use crowdsourcing or human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our work is primarily theoretical - we do not use crowdsourcing or human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLMs other than for writing or editing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## References

- [ACBFS95] Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. Gambling in a rigged casino: The adversarial multi-armed bandit problem. In Proceedings of IEEE 36th annual foundations of computer science , pages 322-331. IEEE, 1995.
- [AGL ` 17] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-efficient SGD via gradient quantization and encoding. In Advances in Neural Information Processing Systems , pages 1707-1718, 2017.
- [AGO ` 13] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Juan-Luis ReyesOrtiz. A public domain dataset for human activity recognition using smartphones. In Proceedings of the 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN) , pages 437-442, 2013.
- [AZL17] Zeyuan Allen-Zhu and Yuanzhi Li. First efficient convergence for streaming k-pca: a global, gap-free, and near-optimal rate. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS) , pages 487-492. IEEE, 2017.
- [BDF13] Akshay Balsubramani, Sanjoy Dasgupta, and Yoav Freund. The fast convergence of incremental pca. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 26 , pages 3174-3182. Curran Associates, Inc., 2013.
- [BDWY16] Maria-Florina Balcan, Simon Shaolei Du, Yining Wang, and Adams Wei Yu. An improved gap-dependency analysis of the noisy power method. In Vitaly Feldman, Alexander Rakhlin, and Ohad Shamir, editors, 29th Annual Conference on Learning Theory , volume 49 of Proceedings of Machine Learning Research , pages 284-309, Columbia University, New York, New York, USA, 23-26 Jun 2016. PMLR.
- [CYWZ18] Minshuo Chen, Lin Yang, Mengdi Wang, and Tuo Zhao. Dimensionality reduction for stationary time series via stochastic nonconvex optimization. Advances in Neural Information Processing Systems , 31, 2018.
- [DMM ` 18] Dipankar Das, Naveen Mellempudi, Dheevatsa Mudigere, Dhiraj Kalamkar, Sasikanth Avancha, Kunal Banerjee, Srinivas Sridharan, Karthik Vaidyanathan, Bharat Kaul, Evangelos Georganas, et al. Mixed precision training of convolutional neural networks using integer operations. arXiv preprint arXiv:1802.00930 , 2018.
- [DPHZ23] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314 , 2023. https: //arxiv.org/abs/2305.14314 .
- [DSLZ ` 18] Christopher De Sa, Megan Leszczynski, Jian Zhang, Alana Marzoev, Christopher R Aberger, Kunle Olukotun, and Christopher Ré. High-accuracy low-precision training. arXiv preprint arXiv:1803.03383 , 2018.
- [GKD ` 22] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. In Low-power computer vision , pages 291-326. Chapman and Hall/CRC, 2022.
- [HCS ` 16] Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks. In Advances in Neural Information Processing Systems , 2016.
- [Heb49] Donald O. Hebb. The Organization of Behavior: A Neuropsychological Theory . John Wiley &amp; Sons, New York, 1949.
- [HNWTW20] De Huang, Jonathan Niles-Weed, Joel A. Tropp, and Rachel Ward. Matrix concentration for products, 2020.
- [HP14] Moritz Hardt and Eric Price. The noisy power method: A meta algorithm with applications. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 27 , pages 2861-2869. Curran Associates, Inc., 2014.

- [HW19] Amelia Henriksen and Rachel Ward. AdaOja: Adaptive Learning Rates for Streaming PCA. arXiv e-prints , page arXiv:1905.12115, May 2019.
- [JJK ` 16] Prateek Jain, Chi Jin, Sham Kakade, Praneeth Netrapalli, and Aaron Sidford. Streaming pca: Matching matrix bernstein and near-optimal finite sample guarantees for oja's algorithm. In Proceedings of The 29th Conference on Learning Theory (COLT) , June 2016.
- [JKL ` 24] Arun Jambulapati, Syamantak Kumar, Jerry Li, Shourya Pandey, Ankit Pensia, and Kevin Tian. Black-box k-to-1-pca reductions: Theory and applications. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 2564-2607. PMLR, 30 Jun-03 Jul 2024.
- [JL84] William B. Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space. In Contemporary Mathematics , volume 26, page 189-206, 1984.
- [JNN19] Prateek Jain, Dheeraj Nagaraj, and Praneeth Netrapalli. SGD without Replacement: Sharper Rates for General Smooth Convex Functions. arXiv e-prints , page arXiv:1903.01463, March 2019.
- [Kah96] William Kahan. Ieee standard 754 for binary floating-point arithmetic. Lecture Notes on the Status of IEEE , 754(94720-1776):11, 1996.
- [KLL ` 23] Jonathan Kelner, Jerry Li, Allen X Liu, Aaron Sidford, and Kevin Tian. Semi-random sparse recovery in nearly-linear time. In The Thirty Sixth Annual Conference on Learning Theory , pages 2352-2398. PMLR, 2023.
- [KPS25] Syamantak Kumar, Shourya Pandey, and Purnamrita Sarkar. Beyond sin-squared error: linear time entrywise uncertainty quantification for streaming pca. In Proceedings of the Forty-First Conference on Uncertainty in Artificial Intelligence , UAI '25. JMLR.org, 2025.
- [KS24a] Syamantak Kumar and Purnamrita Sarkar. Oja's algorithm for streaming sparse pca. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [KS24b] Syamantak Kumar and Purnamrita Sarkar. Streaming pca for markovian data. Advances in Neural Information Processing Systems , 36, 2024.
- [KWW ` 17] Urs Köster, Tristan J. Webb, Xin Wang, Marcel Nassar, Arjun K. Bansal, William H. Constable, O˘ guz H. Elibol, Scott Gray, Stewart Hall, Luke Hornof, Amir Khosrowshahi, Carey Kloss, Ruby J. Pai, and Naveen Rao. Flexpoint: An adaptive numerical format for efficient training of deep neural networks. In Advances in Neural Information Processing Systems , volume 30, pages 1742-1750, 2017.
- [LBBH98] Yann LeCun, León Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE , 86(11):22782324, 1998.
- [LD19] Zheng Li and Christopher M. De Sa. Dimension-free bounds for low-precision training. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32 (NeurIPS 2019) , pages 11728-11738, 2019.
- [LDS19] Zheng Li and Christopher De Sa. Dimension-free bounds for low-precision training. In Advances in Neural Information Processing Systems , 2019.
- [LDX ` 17] Hao Li, Soham De, Zheng Xu, Christoph Studer, Hanan Samet, and Tom Goldstein. Training quantized nets: A deeper understanding. In Advances in Neural Information Processing Systems , pages 5813-5823, 2017.
- [Lia21] Xin Liang. On the optimality of the oja's algorithm for online pca, 2021.

- [LSW21] Robert Lunde, Purnamrita Sarkar, and Rachel Ward. Bootstrapping the error of oja's algorithm. Advances in Neural Information Processing Systems , 34:6240-6252, 2021.
- [Mac08] Lester Mackey. Deflation methods for sparse pca. Advances in neural information processing systems , 21, 2008.
- [MNA ` 18] Paulius Micikevicius, Sharan Narang, Gabriel Alben, Gregory Diamos, Erich Elsen, David Garcia, Dmitry Ginsburg, Michael Houston, Oleksii Kuchaiev, Sanjo Venkatesh, and Hao Wu. Mixed precision training. In International Conference on Learning Representations , 2018.
- [Mon22] Jean-Marie Monnez. Stochastic approximation of eigenvectors and eigenvalues of the q-symmetric expectation of a random matrix. Communications in Statistics-Theory and Methods , pages 1-15, 2022.
- [MP22] Nikos Mouzakis and Eric Price. Spectral guarantees for adversarial streaming pca, 2022.
- [NTSW ` 22] Miloš Nikoli´ c, Enrique Torres Sanchez, Jiahui Wang, Ali Hadi Zadeh, Mostafa Mahmoud, Ameer Abdelhadi, Kareem Ibrahim, and Andreas Moshovos. Schrödinger's fp: Dynamic adaptation of floating-point containers for deep learning training. arXiv preprint arXiv:2204.13666 , 2022.
- [Oja82] Erkki Oja. Simplified neuron model as a principal component analyzer. Journal of mathematical biology , 15:267-273, 1982.
- [OYP25] Kaan Ozkara, Tao Yu, and Youngsuk Park. Stochastic rounding for llm training: Theory and practice. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS) , 2025. https://arxiv.org/abs/2502.20566 .
- [Pea01] Karl Pearson. Liii. on lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science , 2(11):559-572, 1901.
- [PRSS ` 22] Dominika Przewlocka-Rus, Syed Shakib Sarwar, H Ekin Sumbul, Yuecheng Li, and Barbara De Salvo. Power-of-two quantization for low bitwidth and hardware compliant neural networks. arXiv preprint arXiv:2203.05025 , 2022.
- [Rie67] Bernhard Riemann. Ueber die Darstellbarkeit einer Function durch eine trigonometrische Reihe . Dieterich, 1867. In German.
- [SFD ` 14] Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns. In Interspeech , 2014.
- [Sha16a] Ohad Shamir. Convergence of stochastic gradient descent for pca. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48 , ICML'16, page 257-265. JMLR.org, 2016.
- [Sha16b] Ohad Shamir. Fast stochastic algorithms for svd and pca: Convergence properties and convexity. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning , volume 48 of Proceedings of Machine Learning Research , pages 248-256, New York, New York, USA, 20-22 Jun 2016. PMLR.
- [She97] William Fleetwood Sheppard. On the calculation of the most probable values of frequency-constants for data arranged according to equidistant division of a scale. Proceedings of the London Mathematical Society , 1(1):353-380, 1897.
- [SJS09] Reza Sameni, Christian Jutten, and Mohammad B Shamsollahi. A deflation procedure for subspace decomposition. IEEE Transactions on Signal Processing , 58(4):23632374, 2009.

- [SLZ ` 18] Christopher De Sa, Megan Leszczynski, Jian Zhang, Alana Marzoev, Christopher R. Aberger, Kunle Olukotun, and Christopher Ré. High-accuracy low-precision training. arXiv preprint arXiv:1803.03383 , 2018.
- [SOR14] Christopher De Sa, Kunle Olukotun, and Christopher Ré. Global convergence of stochastic gradient descent for some nonconvex matrix problems. CoRR , abs/1411.1134, 2014.
- [SRO15] Christopher De Sa, Christopher Re, and Kunle Olukotun. Global convergence of stochastic gradient descent for some non-convex matrix problems. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning , volume 37 of Proceedings of Machine Learning Research , pages 23322341, Lille, France, 07-09 Jul 2015. PMLR.
- [Sti86] S. M. Stigler. The History of Statistics: The Measurement of Uncertainty before 1900 . Harvard University Press, Cambridge, 1986.
- [SYK21] Heming Sun, Lu Yu, and Jiro Katto. Learned image compression with fixed-point arithmetic. In 2021 Picture Coding Symposium (PCS) , pages 1-5. IEEE, 2021.
- [SYKM17] Ananda Theertha Suresh, Felix X. Yu, Harsha Kumar, and H. Brendan McMahan. Distributed mean estimation with limited communication. arXiv preprint arXiv:1611.00349 , 2017.
- [SZOR15] Christopher M. De Sa, Ce Zhang, Kunle Olukotun, and Christopher Ré. Taming the wild: A unified analysis of hogwild-style algorithms. In Advances in Neural Information Processing Systems , pages 2674-2682, 2015.
- [Ver10] Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027 , 2010.
- [Wed72] Per-Åke Wedin. Perturbation bounds in connection with singular value decomposition. BIT Numerical Mathematics , 12:99-111, 1972.
- [WXY ` 17] Wei Wen, Chunpeng Xu, Felix Yan, Chunyi Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In Advances in Neural Information Processing Systems , 2017.
- [XHDS ` 18] Peng Xu, Bryan He, Christopher De Sa, Ioannis Mitliagkas, and Chris Re. Accelerated stochastic power iteration. In International Conference on Artificial Intelligence and Statistics , pages 58-67. PMLR, 2018.
- [XLY ` 24] Yongqi Xu, Yujian Lee, Gao Yi, Bosheng Liu, Yucong Chen, Peng Liu, Jigang Wu, Xiaoming Chen, and Yinhe Han. Bitq: Tailoring block floating point precision for improved dnn efficiency on resource-constrained devices. arXiv preprint arXiv:2409.17093 , 2024.
- [XMHK23] Lu Xia, Stefano Massei, Michiel E. Hochstenbach, and Barry Koren. On the influence of stochastic roundoff errors and their bias on the convergence of the gradient descent method with low-precision floating-point computation, 2023.
- [Yat09] Randy Yates. Fixed-point arithmetic: An introduction. Digital Signal Labs , 81(83):198, 2009.
- [YGG ` 24] Tao Yu, Gaurav Gupta, Karthick Gopalswamy, Amith R. Mamidala, Hao Zhou, Jeffrey Huynh, Youngsuk Park, Ron Diamant, Anoop Deoras, and Luke Huan. Collage: Lightweight low-precision strategy for llm training. In Proceedings of the 41st International Conference on Machine Learning , 2024.
- [YHW18] Puyudi Yang, Cho-Jui Hsieh, and Jane-Ling Wang. History pca: A new algorithm for streaming pca. arXiv preprint arXiv:1802.05447 , 2018.
- [YIY21] Hisakatsu Yamaguchi, Makiko Ito, and Katsuhiro Yoda. Training deep neural networks in 8-bit fixed point with dynamic shared exponent management. In Proceedings of the 2021 Design, Automation &amp; Test in Europe Conference (DATE) , 2021.

- [Zie03] Eric R Ziegel. Principal component analysis. Technometrics , 45(3):276-277, 2003.
- [ZLK ` 17] Hantian Zhang, Jerry Li, Kaan Kara, Dan Alistarh, Ji Liu, and Ce Zhang. ZipML: Training linear models with end-to-end low precision, and a little bit of deep learning. In Proceedings of the 34th International Conference on Machine Learning , pages 4035-4043, 2017.
- [ZMK22] Sai Qian Zhang, Bradley McDanel, and T. Kung, H.˙Fast: Dnn training under variable precision block floating point with stochastic rounding. In Proceedings of the 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA) , pages 846-860, 2022.
- [ZWG ` 23] Jiajun Zhou, Jiajun Wu, Yizhao Gao, Yuhao Ding, Chaofan Tao, Boyu Li, Fengbin Tu, Kwang-Ting Cheng, Hayden Kwok-Hay So, and Ngai Wong. Dybit: Dynamic bit-precision numbers for efficient quantized neural network inference. arXiv preprint arXiv:2302.12510 , 2023.

The Appendix is organized as follows:

1. Section A provides utility results useful in subsequent proofs.
2. Section B provides the proof of the lower bound described in Section 3.1
3. Section C proves helper lemmas for the results in Section 4.
4. Section D proves Theorems 1, 2 and 3.
5. Section E proves the boosting result (Lemma 3) and end to end analysis of Algorithm 1 followed by the boosting algorithm 2.
6. Section F provides additional experiments.
7. Section G provides more related work.

## A Utlity Results

Lemma A.1. Let l ď x ď u be reals, and define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Throughout the proof, we condition on the fixed x and treat all randomness as coming from the independent choices made by the quantizer.

- (i) Unbiasedness. We have

<!-- formula-not-decoded -->

- (ii) Boundedness. By definition, after rounding, we always round any x P r u, l s to either u or l . Therefore, | Q p x, Q q ´ x | ď u ´ l .

(iii) Variance bound. Using the variance of a Bernoulli random variable, we have,

<!-- formula-not-decoded -->

since t p 1 ´ t q ď 1 { 4 for all reals t .

Lemma A.2 (Choice of learning rate) . Let η : ' α log p n q b p λ 1 ´ λ 2 q . Then, under Assumption 1, for θ P p 0 , 1 q , η satisfies

<!-- formula-not-decoded -->

for α ą 1 , b ě 250 α 2 log 2 p n q log ` d θ ˘ { p λ 1 ´ λ 2 q 2 , and κ 2 b ď 0 . 004 { log ` d θ ˘ .

Proof. For Lemma A.8, we require,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For Theorem A.4, we require,

where θ P p 0 , 1 q represents the failure probability. It is not hard to see that (A.15) implies (A.14). Therefore it suffices to ensure

<!-- formula-not-decoded -->

Setting each term smaller than 0 . 004 , it suffices to have

<!-- formula-not-decoded -->

which completes the proof for the first condition.

The second condition on η follows by setting η ď 1 and solving for b . This yields

<!-- formula-not-decoded -->

Since α ą 1 , the first term is larger than the second one, which completes the proof.

Lemma A.3. Let w and ξ be vectors in R d such that ∥ w ∥ ' 1 and w ` ξ ‰ 0 . Then,

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Lemma A.4. Let x and y be unit vectors in R d . Then,

<!-- formula-not-decoded -->

Proof. We express sin 2 p x , y q in terms of ∥ x ´ y ∥ and ∥ x ` y ∥ . Since ∥ x ´ y ∥ 2 ' ∥ x ∥ 2 ` ∥ y ∥ 2 ´ 2 x J y ' 2 ´ 2 cos p x , y q and ∥ x ` y ∥ 2 ' 2 ` 2 cos p x , y q ,

<!-- formula-not-decoded -->

The upper bound on sin 2 p x , y q follows immediately from the above equations. For the lower bound, note that at least one of ∥ x ´ y ∥ 2 and ∥ x ` y ∥ 2 is at least 2 because their sum is equal to 4 . If ∥ x ` y ∥ 2 ě 2 , then sin 2 p x , y q ě ∥ x ´ y ∥ 2 { 2 . Otherwise, sin 2 p x , y q ě ∥ x ` y ∥ 2 { 2 .

Lemma A.5. Let x , y , and z be non-zero vectors in R d . Then,

<!-- formula-not-decoded -->

Proof. For unit vectors u and v in R d ,

<!-- formula-not-decoded -->

By parallelogram law,

<!-- formula-not-decoded -->

## B Lower Bounds

## Proof of Lemma 1

<!-- formula-not-decoded -->

Consider any a vector w P V L , and let ˜ w ' w {} w } . Since w P V L , w p i q ' 0 or | w p i q| ě δ { 2 . In particular, | v 1 p i q ´ w p i q| ě δ { 6 and | v 1 p i q ` w p i q| ě δ { 6 for all i P r d ´ 1 s . It follows that

<!-- formula-not-decoded -->

and ∥ v 1 ` w ∥ 2 ě δ 2 p d ´ 1 q 36 similarly. The Lemma follows from A.4.

## Proof of Lemma 2

Proof. It suffices to construct two unit vectors v 1 and v 2 such that inf w P V NL sin 2 p w , v 1 q ' Ω p ζ 2 q and inf w P V NL sin 2 p w , v 2 q ' Ω p δ 2 0 d q .

Let v 1 be the vector in R d with coordinates

<!-- formula-not-decoded -->

For the sake of contradiction, suppose there exists w 1 P V NL such that sin 2 p w 1 , v 1 q ď ζ 2 { 100 . Let ˜ w 1 : ' w 1 {} w 1 } . By Lemma A.4,

<!-- formula-not-decoded -->

Flipping the sign of w 1 if necessary, we may assume ∥ v 1 ´ ˜ w 1 ∥ 2 2 ď ζ 2 { 50 . So,

<!-- formula-not-decoded -->

The bound ζ ď 0 . 1 ensures v 1 p 1 q ě 20 { 29 and v 1 p 2 q ´ v 1 p 1 q ě ζ { 3 , which also implies ˜ w 1 p 2 q ´ ˜ w 1 p 1 q ě ζ { 3 ´ 2 ζ { 7 ' ζ { 21 ą 0 . It follows that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Under the logarithmic quantization scheme, it can be inductively shown that

<!-- formula-not-decoded -->

for all non-negative integers k such that q k P Q NL . In particular, w 1 p 2 q` δ 0 { ζ w 1 p 1 q` δ 0 { ζ must be an integral power of 1 ` ζ , contradicting

<!-- formula-not-decoded -->

Therefore, inf w 1 P V NL sin 2 p w 1 , v 1 q ě ζ 2 { 100 .

The other bound is similar to the linear case. let v 2 be the vector with coordinates

<!-- formula-not-decoded -->

Any w 2 P V NL satisfies w 2 p i q ' 0 or | w 2 p i q| ě δ 0 for all i P r d s . Since } w 2 } P r 1 { 2 , 2 s , the normalized vector ˜ w 2 ' w 2 {} w 2 } satisfies | ˜ w 2 p i q| ' 0 or | ˜ w 2 p i q| ě δ 0 { 2 for all i P r d s .

In particular | v 2 p i q ´ ˜ w 2 p i q| ě δ 0 { 6 and | v 2 p i q ` ˜ w 2 p i q| ě δ 0 { 6 for all i P r d s . By Lemma A.4,

<!-- formula-not-decoded -->

## C Proof of Results in Section 4

For ease of exposition, all results in this section are stated with a generic number of data n . We apply these results with different choices of n (e.g. number of batches b ) for proving the main theorems (Theorem 1, 2, 3). Consider Oja's Algorithm applied to the matrices A i P R d ˆ d , such that A i ' η D i ` Ξ i where D i are independent with E r D i s ' Σ . Let S i be the set of all random vectors ξ resulting from the quantizations in the first i iterations of the algorithm, and let F i ´ denote the σ -field generated by D 1 , . . . , D i and S i ´ 1 , and denote E i r . s : ' E r . | F i ´ s . We assume the noise term Ξ i is conditionally unbiased, i.e., E i r Ξ i s ' 0 d ˆ d .

<!-- formula-not-decoded -->

Recall the update rule

<!-- formula-not-decoded -->

We bound the numerator and denominator in (A.17) separately.

For the numerator, we will show that } ś 1 t ' n p I ` A t q ´ p I ` η Σ q n } is small. Let Y i ' I ` A i for i P r n s , and let t Z i u 0 ď i ď n be defined as

<!-- formula-not-decoded -->

Note that Z i ´ 1 is measurable w.r.t F i ´ .

We are now ready to state our first result. Note that

<!-- formula-not-decoded -->

where A i ' η D i ` Ξ i and D i are independent d ˆ d random matrices with mean Σ .

## C.1 Proof of Proposition 1

Proposition A.1. [Proposition 1 in main paper]Let the noise term Ξ , defined in (9) , be bounded as ∥ Ξ ∥ ď κ almost surely. Under Assumption 1, for η P p 0 , 1 q and b ą 0 , we have

<!-- formula-not-decoded -->

where Z 0 ' I , ϕ : ' p 1 ` ηλ 1 q 2 , γ : ' 2 p η 2 M 2 ` κ 2 q , and C p : ' p ´ 1 .

Proof. Recall the notation Y i : ' I ` A i for all i . Then,

<!-- formula-not-decoded -->

Note that m i ' 1 ` ηλ 1 and

<!-- formula-not-decoded -->

The last line uses Eq 9. Thus σ i ' η M ` κ 1 ` ηλ 1 . Note that ν ď 2 p η 2 M 2 ` κ 2 q . The same argument as in Theorem 7.4 in [HNWTW20] gives the bound.

Lemma A.6. Under Assumption 1, and with η set according to Lemma A.2 with b ' n ,

<!-- formula-not-decoded -->

where γ : ' 2 p η 2 M 2 ` κ 2 q and e ' exp p 1 q is the Napier's constant.

Proof. By Proposition A.1, for any positive real p ,

<!-- formula-not-decoded -->

where ϕ ' p 1 ` ηλ 1 q 2 , γ ' 2 p η 2 M 2 ` κ 2 q , and C p ' p ´ 1 .

If t 2 e 2 nγ ă 2 , then e ¨ exp ´ ´ t 2 2 e 2 nγ ¯ ě 1 and the Lemma holds trivially. Otherwise, let p : ' t 2 e 2 nγ ě 2 . Since t ď e , C p nγ ď pnγ ď t 2 e 2 ď 1 . Therefore, exp p C p nγ q ´ 1 ď eC p nγ ď t 2 e , which implies

<!-- formula-not-decoded -->

Lemma A.7. Under Assumption 1 and with η set according to Lemma A.2 with b ' n ,

<!-- formula-not-decoded -->

where γ ' 2 p η 2 M 2 ` κ 2 q . Moreover, if 2 nγ p 1 ` 2 log p d qq ď 1 , then

<!-- formula-not-decoded -->

Proof. Using Proposition A.1 ϕ : ' p 1 ` ηλ 1 q 2 , and γ : ' 2 p η 2 M 2 ` κ 2 q ,

<!-- formula-not-decoded -->

Set p : ' max ´ 2 , b 2 log d nγ ¯ . Then, ∥ Z 0 ∥ p, 2 ' d 1 p ď exp ` pnγ 2 ˘ . Therefore,

<!-- formula-not-decoded -->

For the second result, set p : ' 2 p 1 ` log p d qq . Then, C p nγ ď 1 by assumption and ∥ Z 0 ∥ p ' d 1 { p ď ? e . By Proposition A.1,

<!-- formula-not-decoded -->

## C.2 Proof of Lemma 4

Lemma A.8 (Lemma 4 in main paper) . Let Assumption 1 hold and η be set according to Lemma A.2 with b ' n . Define γ : ' 2 p η 2 M 2 ` κ 2 q . If 2 nγ p 1 ` 2 log p d qq ď 1 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The last line used E r Ξ i | F i ´ s ' 0 and that Z i ´ 1 is measurable with respect to F i ´ . In other words,

<!-- formula-not-decoded -->

For the first term, following the analysis of Lemma 10 of [JJK ` 16],

<!-- formula-not-decoded -->

The second term can be bounded as

<!-- formula-not-decoded -->

Combining (A.19) and (A.20), we obtain the recurrence

<!-- formula-not-decoded -->

By Lemma A.7, we have for γ : ' 2 p η 2 M 2 ` κ 2 q ,

<!-- formula-not-decoded -->

where s ' p η 2 V 0 ` κ 1 q exp ` 2 ? 2 nγ log d ˘ . Unrolling the recursion,

<!-- formula-not-decoded -->

where the third inequality holds because x ď 2 . 35 p 1 ´ e ´ x q for x ď 2 and the last inequality holds because β 0 ď d and 2 . 35 exp p 2 ? 2 nγ log d q 2 ď 2 . 35 exp p ? 2 q 2 ă 5 .

## D Proofs of Theorems 1, 2, and 3

## D.1 Proof of Theorem 1

We are now ready to present the proof of Theorem 1, which follows from the following Theorem A.4 and setting a constant failure probability for θ .

Theorem A.4. Fix θ P p 0 , 1 q . Then, for w being the output of Algorithm 1, under assumption 1, learning rate η ' α log n b p λ 1 ´ λ 2 q with α is set as in Lemma A.2, κ 1 ď 1 { 2 , and

<!-- formula-not-decoded -->

where γ : ' 2 p η 2 M 2 ` κ 2 q . Then, with probability at least 1 ´ 3 θ ,

<!-- formula-not-decoded -->

Proof. Note that by Algorithm 1 and the definition of Z in (A.18),

<!-- formula-not-decoded -->

Since v 1 v J 1 ` V K V K J ' I d ,

<!-- formula-not-decoded -->

By Lemma 6 from [JJK ` 16], with probability at least 1 ´ θ ,

<!-- formula-not-decoded -->

By Lemma A.7 with q ' 2 and p ' 2 p 1 ` log p d qq ,

<!-- formula-not-decoded -->

For the numerator, we use Lemma A.8 and Markov's inequality to get

<!-- formula-not-decoded -->

with probability at least 1 ´ θ .

The denominator can be bounded as

<!-- formula-not-decoded -->

Using Lemma A.6, with probability atleast 1 ´ θ ,

<!-- formula-not-decoded -->

where the last line follows since p 1 ` x q ě exp ` x ´ x 2 ˘ for all x ě 0 . From equations (A.22), (A.23), and the assumption a 2 e 2 bγ log p d { θ q ď 1 { 2 , it follows that with probability 1 ´ 3 θ ,

<!-- formula-not-decoded -->

Since w Ð Q p u b , Q q , by Lemma A.9 and using ∥ ξ ∥ ď κ ď 0 . 5 ,

<!-- formula-not-decoded -->

The result follows by using equations (A.24), (A.25), and Lemma A.5.

## D.2 Proofs of Theorems 2 and 3

Next, we apply Theorem A.4 to analyze the quantized version of Oja's algorithm as described in Algorithm 1. The idea is to show that the error from the rounding operation can be incorporated into the noise in the iterates of Oja's algorithm, which have mean zero. For this subsection, we will use:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first state and prove some intermediate results needed to prove Theorems 2 and Theorems 3. Theorem A.5. Let d, n, b P N , and let t X i u i Pr n s be a set of n IID vectors in R d satisfying assumption 1. Let η : ' α log n b p λ 1 ´ λ 2 q be the learning rate set as in Lemma A.2. Suppose the quantization grid Q ' Q L , and a 4 e 2 b p 4 η 2 ` 9 δ 2 d q log p d { θ q ď 1 2 . Then, with probability at least 0 . 9 , the output w of Algorithm 1 satisfies

<!-- formula-not-decoded -->

Proof. In order to apply Theorem 1, we come up with valid choices of V 0 , κ , and κ 1 . Since each D i is symmetric and t X i u i Pr n s are independent,

<!-- formula-not-decoded -->

Next,

Also observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By equation A.27,

<!-- formula-not-decoded -->

As for κ , we have

Thus we have:

We have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We are now ready to obtain the sin-squared error. Note that M ď 2 , since } X i } ď 1 almost surely, for all i P r n s . By Theorem A.4, with probability at least 1 ´ 3 θ ,

<!-- formula-not-decoded -->

as long as a 2 e 2 bγ log p d { θ q ď 1 2 . Our parameter choices are V 0 ' b V n , κ ' 3 δ ? d, and κ 1 ' 6 δ 2 d .

<!-- formula-not-decoded -->

Lemma A.9. Let u ' Q p w , Q NL q , where u P R d and Q NL is defined in equation 4. Then,

<!-- formula-not-decoded -->

Proof. Let ξ ' Q p w , Q NL q´ w . Say w i ą 0 . Let k be the unique integer such that w i P r q k , q k ` 1 s . Equivalently for negative w i , say the bin is r´ q k ` 1 , ´ q k s . We have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem A.6. Fix θ P p 0 , 1 q . Let the initial vector u 0 ' N p 0 , I q . Let the number of batches b and quantization scale δ be such that a 4 e 2 b p 4 η 2 ` 32 δ 2 0 d ` 98 ζ 2 q log p d { θ q ď 1 { 2 . Then, under assumption 1with η set as α log n b p λ 1 ´ λ 2 q , where α is set as in Lemma A.2, δ 0 ? d ď 0 . 25 , and ζ ď 0 . 25 , with probability at least 1 ´ 3 θ , the output w b of Algorithm 1 gives:

<!-- formula-not-decoded -->

Proof. In order to apply Theorem 1 we need to bound V , κ and κ 1 . We start with the first. For us, D i is defined in Eq 9. Let R i denote the random variables in the quantization up to and including the i th update.

Our analysis is analogous to the previous theorem. Note that the V 0 parameter is as in Eq A.26.

Now we will work out κ and κ 1 since those are the only quantities that change for the nonlinear quantization. Recall that we have,

<!-- formula-not-decoded -->

Now we obtain the Frobenius norm of ξ a,i , ξ 1 , and ξ 2 under the nonlinear quantization. We start with the norm of w i , a quantized version of a unit vector u i ´ 1 .

By Lemma A.9, } w i } ď 1 ` δ 0 ? d ` ζ . Let s j ' X j p X T j w i q . Then,

<!-- formula-not-decoded -->

Another application of Lemma A.9 gives:

<!-- formula-not-decoded -->

which implies } ξ a,i } ď δ 0 ? d ` 1 . 5 ζ . Next, we bound ξ 1 ,i ' Q p u i ´ 1 , Q NL q´ u i ´ 1 . By Lemma A.9,

<!-- formula-not-decoded -->

Finally we bound ξ 2 ,i . Recall that:

<!-- formula-not-decoded -->

Since each ∥ ∥ X j X J j w i ∥ ∥ ď 1 ` δ 0 ? d ` ζ ,

<!-- formula-not-decoded -->

By Lemma A.9,

<!-- formula-not-decoded -->

In all, it follows that

<!-- formula-not-decoded -->

We are ready to obtain the sin-squared error. Note that M ď 2 , since } X i } ď 1 almost surely, for all i P r n s . By Theorem A.4, with probability at least 1 ´ 3 θ ,

<!-- formula-not-decoded -->

as long as a 2 e 2 bγ log p d { θ q ď 1 2 . Our parameter choices are V 0 ' b V n , κ ' 4 δ 0 ? d ` 7 ζ, and κ 1 ' p 4 δ 0 ? d ` 7 ζ q 2 . Therefore,

<!-- formula-not-decoded -->

## D.2.1 Finishing the Proofs of Theorems 2 and 3

Proof of Theorem 2. For the linear quantization scheme, we apply Theorem A.5 with θ ' 1 { 30 and b ' Θ ´ α 2 log 2 n log d p λ 1 ´ λ 2 q 2 ¯ . Moreover, since δ ' ˜ O ´ λ 1 ´ λ 2 α ? d ¯ , the condition a 4 e 2 b p 4 η 2 ` 9 δ 2 d q log p d { θ q ď 1 2 holds. The Theorem follows by substituting these values into the bound of Theorem A.5.

The proof of the logarithmic scheme follows analogously from Theorem A.6.

Proof of Theorem 3. We set θ ' 1 { 30 . For the linear quantization scheme, we apply Theorem A.5 with b ' n . Moreover, since δ ' 2 2 ´ β ' O ´ min ´ λ 1 ´ λ 2 α ? d log p n q , 1 ? dn ¯¯ , the condition a 4 e 2 b p 4 η 2 ` 9 δ 2 d q log p d { θ q ď 1 2 holds. The Theorem follows by substituting these values into the bound of Theorem A.5.

For the non-linear scheme, the proof follows analogously from Theorem A.6.

## D.3 Optimal Choice of Parameters

We want to minimize the quantity

<!-- formula-not-decoded -->

where ζ ' 2 ´ β m and δ 0 ' 4 ¨ 2 ´ 2 βe ´ 1 . Here, β m and β e are the number of bits used by the mantissa and the exponent, respectively, and satisfy the constraint

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To find β e that minimizes f p β e q we differentiate with respect to β e and set it to 0 .

<!-- formula-not-decoded -->

It is optimal to take β e such that

Then, so

<!-- formula-not-decoded -->

Therefore, we choose

<!-- formula-not-decoded -->

Equivalently, β e ` 2 β e ' 2 β ` log 2 p 8 d ln 2 q . This in particular implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This choice of β ˚ e is valid as long as it does not make β ˚ m non-positive. This is true as long as β ě max p 8 , log 2 p d qq . With these values of β ˚ e and β ˚ m ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## E Proof of Boosting Lemma (Lemma 3)

In this section, we present the proof of the boosting procedure. Our boosting procedure requires a modest assumption that the number of bits β ě 4 , which is already assumed in Section 3.4 while optimizing the parameters.

## Proof of Lemma 3

Proof. For each i P r r s , define the indicator random variable

χ i : ' ✶ ` sin 2 p u i , v q ď ϵ ˘ . Then, by the guarantees of A , Pr p χ i ' 1 q ě 1 ´ p , where p ' 0 . 1 . Let S : ' t i P r r s : χ i ' 1 u , and define the event

<!-- formula-not-decoded -->

The Chernoff bound for the sum of independent Bernoulli random variables gives

<!-- formula-not-decoded -->

By linearity of expectation, E r| S |s ě p 1 ´ p q r . Setting θ ' 1 { 3 ,

<!-- formula-not-decoded -->

It suffices to show that if the event E holds, then ¯ u is well-defined and has small sin-squared error with v . Recall,

<!-- formula-not-decoded -->

Conditioned on E , any i that belongs to the set S satisfies c i ě 0 . 6 r . Indeed, Lemma A.5 gives for any i, j P S

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

because 4 ϵ is within the range of the bounded grid Q L p ϵ q : ' Q L p ϵ, β q defined in (11). Therefore, the algorithm does not return K and ¯ u is well-defined.

Now, | ˜ ρ p ¯ u , u j q| ď 5 ϵ for at least 0 . 5 r indices j P r r s and | S | ě 0 . 6 r . In particular, there exists an index j ˚ P S for which | ˜ ρ p ¯ u , u j ˚ q| ď 5 ϵ . Since 5 ϵ is strictly inside the grid Q L p ϵ q , we get sin 2 p ¯ u , u j ˚ q ď 6 ϵ . We conclude

<!-- formula-not-decoded -->

Theorem A.7 puts everything together and applies Lemma 3 to obtain the final high probability result. Theorem A.7. Suppose A is the Oja's algorithm with the setting of Theorem 2 or 3. Let ϵ be the probability 0 . 9 error bound guaranteed by Theorem 2, r ' r 20 log p 1 { θ q s , and m ' nr . Let t X i u i Pr m s be n IID data drawn from a distribution satisfying assumption 1, and u j Ð A pt X i u p j ´ 1 q n ` 1 ď i ď jn for all j P r r s . Then, the output of algorithm 2 satisfies

<!-- formula-not-decoded -->

which implies with probability at least 1 ´ θ .

Proof. The vectors u 1 , . . . , u r are mutually independent. By Theorem 2, Pr ` sin 2 p u i , v 1 q ą ϵ ˘ ď 0 . 1 @ i P r r s . Therefore, Lemma 3 applies and the theorem follows.

## F Experimental Details

## F.1 Additional Synthetic Experiments

<!-- image -->

(a) Varying sample size n , fixed d ' 100 , bits ' 8 .

<!-- image -->

(b) Varying dimension d , fixed n ' 5000 , bits ' 8 .

<!-- image -->

12

Bits

(c) Varying bits β , fixed n ' 1000 , d ' 100 .

Figure A.1: Variation of sin 2 -error with: (a) sample size, (b) dimension, and (c) quantization bits.

We generate synthetic datasets via the procedure described in [LSW21]. The generation process takes as input the number of samples, n , the dimension d and an eigenvalue decay parameter λ . We defer the details of the generation process to the Appendix Section F. Given the sample size n , dimension d , and decay exponent λ in the eigenvalues, we first draw an n ˆ d matrix Z with independent entries uniformly distributed on r´ ? 3 , ? 3 s so that each coordinate has unit variance. We then build a kernel matrix K P R d ˆ d with entries K ij ' exp ` ´| i ´ j | 0 . 01 ˘ and define a variance profile σ i ' 5 i ´ λ

for i ' 1 , . . . , d . The population covariance is formed as Σ ' p σσ J q ˝ K , where ˝ denotes the Hadamard product. Computing the eigendecomposition of Σ yields its square root Σ 1 { 2 , and the observed data matrix is taken as X ' ` Σ 1 { 2 Z J ˘ J . We then extract the largest two eigenvalues λ 1 ą λ 2 of Σ and the associated top eigenvector v 1 for evaluation. Figure A.1 shows the results for this dataset, which shows similar trends as the experiments described in Figure 2.

## F.2 Real data experiments

This section presents experiments on two real-world datasets. For each dataset, we show sin 2 error with respect to the true offline eigenvector, used as a proxy for the ground truth, varying with the number of bits. The results are plotted in Figure A.2.

The goal of this section is to determine whether real-world experiments reflect the behavior of batched vs. standard methods with linear and logarithmic quantization. Therefore, we use the eigengap computed offline as a proxy of the true eigengap. If we wanted to compute the eigengap in an online manner, we could split the dataset randomly into a holdout set S and a training set r n sz S ; run Oja's algorithm with quantization on a range of eigengaps with outputs u 1 , . . . , u m , and select the one with the largest arg max i u T i p ř j P S D j D T j q u i for a held out set S .

Figure A.2: Variation of sin 2 -error with bits for (a) HAR dataset (b) MNIST dataset.

<!-- image -->

Time series + missing data : The Human Activity Recognition (HAR) Dataset [AGO ` 13] contains smartphone sensor readings from 30 subjects performing daily activities (walking, sitting, standing, etc.). Each data instance is a 2.56-second window of inertial sensor signals represented as a feature vector. Here, n ' 7352 and d ' 561 . For each datum, we also replace 10% of features randomly by zero to simulate missing data.

Image data : We use the MNIST dataset [LBBH98] of images of handwritten digits (0 through 9). Here, n ' 60 , 000 , d ' 784 , with each image normalized to a 28 ˆ 28 pixel resolution.

These results collectively highlight that using the true offline eigengap (i) under stochastic rounding, batching provides a significant boost in performance since the quantization error does not depend linearly on n , and (ii) the logarithmic quantization attains a nearly dimension-free quantization error in comparison to linear quantization across a wide range of number of bits.

## G Related Work

In this section, we provide some more related work on low-precision optimization. [DPHZ23] introduced QLoRA, which back-propagates through a frozen 4-bit quantized LLM into LoRA modules, enabling efficient finetuning of 65B-parameter models on a single 48 GB GPU with full 16-bit performance retention. Earlier works [XMHK23] examined the impact of stochastic round-off errors and their bias on gradient descent convergence under low-precision arithmetic. [YGG ` 24] propose Collage , a lightweight low-precision scheme for LLM training in distributed settings, combining block-wise quantization with feedback error to stabilize large-scale pretraining. Finally, communication-efficient distributed SGD techniques, such as 1-bit SGD with error feedback

[SFD ` 14] and randomized sketching primitives (e.g., Johnson-Lindenstrauss projections [JL84]), further underscore the broad efficacy of low-precision computation.

Low-Precision Optimization : Reducing the bit-width of model parameters and gradient updates has proven effective for alleviating communication and memory bottlenecks in large-scale learning. QSGD [AGL ` 17] uses randomized rounding to compress each coordinate to a few bits while preserving unbiasedness, incurring only an O p ? d { 2 β q increase in gradient noise for β bits. [WXY ` 17] maps gradients to t´ 1 , 0 , ` 1 u plus a shared scale and demonstrates negligible accuracy loss on ImageNet and CIFAR benchmarks. [SYKM17] achieve optimal communication-accuracy trade-offs via randomized rotations and scalar quantization. More recently, 'dimension-free' analyses such as [LDS19] avoid scaling the required error rate with model dimension, instead depending on a suitably defined smoothness parameter.
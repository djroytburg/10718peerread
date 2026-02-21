## A Finite Sample Analysis of Distributional TD Learning with Linear Function Approximation

Yang Peng ∗

Kaicheng Jin †

Liangyu Zhang ‡

Zhihua Zhang §

## Abstract

In this paper, we study the finite-sample statistical rates of distributional temporal difference (TD) learning with linear function approximation. The aim of distributional TD learning is to estimate the return distribution of a discounted Markov decision process for a given policy π . Previous works on statistical analysis of distributional TD learning mainly focus on the tabular case. In contrast, we first consider the linear function approximation setting and derive sharp finite-sample rates. Our theoretical results demonstrate that the sample complexity of linear distributional TD learning matches that of classic linear TD learning. This implies that, with linear function approximation, learning the full distribution of the return from streaming data is no more difficult than learning its expectation (value function). To derive tight sample complexity bounds, we conduct a fine-grained analysis of the linear-categorical Bellman equation and employ the exponential stability arguments for products of random matrices. Our results provide new insights into the statistical efficiency of distributional reinforcement learning algorithms.

## 1 Introduction

Distributional policy evaluation [Morimura et al., 2010, Bellemare et al., 2017, 2023], which aims to estimate the return distribution of a policy in an Markov decision process (MDP), is crucial for many uncertainty-aware or risk-sensitive tasks [Lim and Malik, 2022, Kastner et al., 2023]. Unlike the classic policy evaluation that only focuses on expected returns (value functions), distributional policy evaluation captures uncertainty and risk by considering the full distributional information. To solve a distributional policy evaluation problem, in the seminal work Bellemare et al. [2017] proposed distributional temporal difference (TD) learning, which can be viewed as an extension of classic TD learning [Sutton, 1988].

Although classic TD learning has been extensively studied [Bertsekas and Tsitsiklis, 1995, Tsitsiklis and Van Roy, 1996, Bhandari et al., 2018, Dalal et al., 2018, Patil et al., 2023, Li et al., 2024a,b, Chen et al., 2024, Samsonov et al., 2024a,b, Wu et al., 2024], the theoretical understanding of distributional TD learning, which is important for risk-sensitive tasks [Wang and Zhou, 2020, Wang et al., 2018, Moghimi and Ku, 2025, ´ Avila Pires et al., 2025, Qi et al., 2025], remains relatively underdeveloped. Recent works [Rowland et al., 2018, B¨ ock and Heitzinger, 2022, Zhang et al., 2025, Rowland et al., 2024a,b, Peng et al., 2024] have analyzed distributional TD learning (or its model-based variants) in the tabular setting. Especially, Rowland et al. [2024b] and Peng et al. [2024] demonstrated that in the

∗ School of Mathematical Sciences, Peking University; email: pengyang@pku.edu.cn .

† School of Mathematical Sciences, Peking University; email: kcjin@pku.edu.cn .

‡ School of Statistics and Data Science, Shanghai University of Finance and Economics; email: zhangliangyu@sufe.edu.cn .

§ School of Mathematical Sciences, School of Computer Science, Peking University; Center for Intelligent Computing, Great Bay University; email: zhzhang@math.pku.edu.cn .

tabular setting, learning the return distribution (in terms of the 1 -Wasserstein distance 5 ) is statistically as easy as learning its expectation. However, in practical scenarios, where the state space is extremely large or continuous, the function approximation [Dabney et al., 2018b,a, Rowland et al., 2019, Yang et al., 2019, Freirich et al., 2019, Yue et al., 2020, Nguyen-Tang et al., 2021, Zhou et al., 2021, Luo et al., 2022, Wenliang et al., 2024, Sun et al., 2024, Cho et al., 2024, Shen et al., 2025] becomes indispensable. This raises a new open question: When function approximation is employed, does learning the return distribution remain as statistically efficient as learning its expectation?

To answer this question, we consider the simplest form of function approximation, i.e. , linear function approximation, and investigate the finite-sample performance of linear distributional TD learning. In distributional TD learning, we need to represent the infinite-dimensional return distributions with some finite-dimensional parametrizations to make the algorithm tractable. Previous works [Bellemare et al., 2019, Lyle et al., 2019, Bellemare et al., 2023] have proposed various linear distributional TD learning algorithms under different parameterizations, namely categorical and quantile parametrizations. In this paper, we consider the categorical parametrization and propose an improved version of the linear-categorical TD learning algorithm ( Linear-CTD ). We then analyze the non-asymptotic convergence rate of Linear-CTD . Our analysis reveals that, with the Polyak-Ruppert tail averaging [Ruppert, 1988, Polyak and Juditsky, 1992] and a proper constant step size, the sample complexity of Linear-CTD matches that of classic linear TD learning ( Linear-TD ) [Li et al., 2024b, Samsonov et al., 2024b]. Thus, this confirms that learning the return distribution is statistically no more difficult than learning its expectation when the linear function approximation is employed.

Notation. In the following parts of the paper, ( x ) + := max { x, 0 } for any x ∈ R . ' ≲ ' (resp. ' ≳ ') means no larger (resp. smaller) than up to a multiplicative universal constant, and a ≃ b means a ≲ b and a ≳ b hold simultaneously. The asymptotic notation f ( · ) = ˜ O ( g ( · )) (resp. ˜ Ω( g ( · )) ) means that f ( · ) is order-wise no larger (resp. smaller) than g ( · ) , ignoring logarithmic factors of polynomials of (1 -γ ) -1 , λ -1 min , α -1 , ε -1 , δ -1 , K , ∥ ψ ⋆ ∥ Σ ϕ , ∥ θ ⋆ ∥ I K ⊗ Σ ϕ . We will explain the concrete meaning of the notation once we have encountered them for the first time.

We denote by δ x the Dirac measure at x ∈ R , 1 the indicator function, ⊗ the Kronecker product (see Appendix A), 1 K ∈ R K the all-ones vector, 0 K ∈ R K the all-zeros vector, I K ∈ R K × K the identity matrix, ∥ u ∥ the Euclidean norm of any vector u , ∥ B ∥ the spectral norm of any matrix B , and ∥ u ∥ B := √ u ⊤ Bu when B is positive semi-definite (PSD). B 1 ≼ B 2 stands for B 2 -B 1 is PSD for any symmetric matrices B 1 , B 2 . And ∏ t k =1 B k is defined as B t B t -1 · · · B 1 for any matrices { B k } t k =1 with appropriate sizes. For any matrix B =[ b (1) , . . . , b ( n )] ∈ R m × n , we define its vectorization as vec ( B )=( b (1) ⊤ , . . . , b ( n ) ⊤ ) ⊤ ∈ R mn . Given a set A , we denote by ∆( A ) the set of all probability distributions over A . For simplicity, we abbreviate ∆([0 , (1 -γ ) -1 ]) as P .

Contributions. Our contribution is two-fold: in algorithms and in theory. Algorithmically, we propose an improved version of the linear-categorical TD learning algorithm ( Linear-CTD ). Rather than using stochastic semi-gradient descent to update the parameter as in Bellemare et al. [2019], Lyle et al. [2019], Bellemare et al. [2023], we directly formulate the linear-categorical projected Bellman equation into a linear system and apply a linear stochastic approximation to solve it. The resulting Linear-CTD can be viewed as a preconditioned version [Chen, 2005, Li, 2017] of the vanilla linear categorical TD learning algorithm proposed in Bellemare et al. [2023, Section 9.6]. By introducing a preconditioner, our Linear-CTD achieves a finite-sample rate independent of the number of supports K in the categorical parameterization, which the vanilla version cannot attain. We provide both theoretical and experimental evidence to demonstrate this advantage of our Linear-CTD .

Theoretically, we establish the first non-asymptotic guarantees for distributional TD learning with the linear function approximation. Specifically, we show that in the generative model setting, with the Polyak-Ruppert tail averaging and a constant step size, we need

<!-- formula-not-decoded -->

online interactions with the environment to ensure Linear-CTD yields a ε -accurate estimator with high probability, when the error is measured by the µ π -weighted 1 -Wasserstein distance. We also

5 Solving distributional policy evaluation ε -accurately in the 1 -Wasserstein distance sense is harder than solving classic policy evaluation ε -accurately, as the absolute difference of value functions is always bounded by the 1 -Wasserstein distance between return distributions.

extend the result to the Markovian setting. Our sample complexity bounds match those of the classic Linear-TD with a constant step size [Li et al., 2024b, Samsonov et al., 2024b], confirming the same statistical tractability of distributional and classic value-based policy evaluations. To establish these theoretical results, we analyze the linear-categorical Bellman equation in detail and apply the exponential stability argument proposed in Samsonov et al. [2024b]. Our analysis of the linearcategorical Bellman equation lays the foundation for subsequent algorithmic and theoretical advances in distributional reinforcement learning with function approximation.

Organization. The remainder of this paper is organized as follows. In Section 2, we recap Linear-TD and tabular categorical TD learning. In Section 3, we introduce the linear-categorical parametrization, and use the linear-categorical projected Bellman equation to derive Linear-CTD . In Section 4, we employ the exponential stability arguments to analyze the statistical efficiency of Linear-CTD . The proof is outlined in Section 5. In Section 6, we conclude our work. See Appendix B for more related work. In Appendix F, we compare various concepts and results between Linear-TD and Linear-CTD . In Appendix G, we empirically validate the convergence of Linear-CTD and compare it with prior algorithms through numerical experiments, confirming our theoretical findings. Details of the proof are given in the appendices.

## 2 Backgrounds

In this section, we recap the basics of policy evaluation and distributional policy evaluation tasks.

## 2.1 Policy Evaluation

A discounted MDP is defined by a 4 -tuple M = ⟨S , A , P , γ ⟩ . We assume that the state space S and the action space A are both Polish spaces, namely complete separable metric spaces. P ( · , · | s, a ) is the joint distribution of reward and next state condition on ( s, a ) ∈ S × A . We assume that all rewards are bounded random variables in [0 , 1] . And γ ∈ (0 , 1) is the discount factor.

Given a policy π : S → ∆( A ) and an initial state s 0 = s ∈ S , a random trajectory { ( s t , a t , r t ) } ∞ t =0 can be sampled: a t | s t ∼ π ( · | s t ) , ( r t , s t +1 ) | ( s t , a t ) ∼ P ( · , · | s t , a t ) , for any t ∈ N . We assume the Markov chain { s t } ∞ t =0 has a unique stationary distribution µ π ∈ ∆( S ) . We define the return of the trajectory as G π ( s ) := ∑ ∞ t =0 γ t r t . The value function V π ( s ) is the expectation of G π ( s ) , and V π := ( V π ( s )) s ∈S ∈ R S . It is known that V π satisfies the Bellman equation:

<!-- formula-not-decoded -->

or in a compact form V π = T π V π , where T π : R S → R S is called the Bellman operator. In the task of policy evaluation, we aim to find the unique solution V π of the equation for some given policy π .

Tabular TD Learning. The policy evaluation problem is reduced to solving the Bellman equation. However, in practical applications T π is usually unknown and the agent only has access to the streaming data { ( s t , a t , r t ) } ∞ t =0 . In this circumstance, we can solve the Bellman equation through linear stochastic approximation (LSA). Specifically, in the t -th time-step the updating scheme is

̸

<!-- formula-not-decoded -->

We expect V t to converge to V π as t tends to infinity. This algorithm is known as TD learning, however, it is computationally tractable only in the tabular setting.

Linear Function Approximation and Linear TD Learning. In this part, we introduce linear function approximation and briefly review the more practical Linear-TD . To be concrete, we assume there is a d -dimensional feature vector for each state s ∈ S , which is given by the feature map ϕ : S → R d . We consider the linear function approximation of value functions:

<!-- formula-not-decoded -->

µ π -weighted norm ∥ V ∥ µ π := ( E s ∼ µ π [ V ( s ) 2 ]) 1 / 2 , and linear projection operator Π π ϕ : R S → V ϕ :

<!-- formula-not-decoded -->

One can check that the linear projected Bellman operator Π π ϕ T π is a γ -contraction in the Polish space ( V ϕ , ∥·∥ µ π ) . Hence, Π π ϕ T π admits a unique fixed point V ψ ⋆ , which satisfies ∥ V π -V ψ ⋆ ∥ µ π ≤ (1 -γ 2 ) -1 / 2 ∥ V π -Π π ϕ V π ∥ µ π [Bellemare et al., 2023, Theorem 9.8]. In Appendix C.1, we show that ψ ⋆ ∈ R d is the unique solution to the linear system for ψ ∈ R d :

<!-- formula-not-decoded -->

In the subscript of the expectation, we abbreviate s ∼ µ π ( · ) , a ∼ π ( ·| s ) , ( r, s ′ ) ∼ P ( · , · | s, a ) as s, a, r, s ′ . For brevity, we will use such abbreviations in this paper when there is no ambiguity. We can use LSA to solve the linear projected Bellman equation (Eqn. 4). As a result, at the t -th time-step, the updating scheme of Linear-TD is

<!-- formula-not-decoded -->

## 2.2 Distributional Policy Evaluation

In certain applications, we are not only interested in finding the expectation of random return G π ( s ) but also want to find the whole distribution of G π ( s ) . This task is called distributional policy evaluation. We use η π ( s ) ∈ P to denote the distribution of G π ( s ) and let η π := ( η π ( s )) s ∈S ∈ P S . Then η π satisfies the distributional Bellman equation:

<!-- formula-not-decoded -->

where the RHS is the distribution of r 0 + γG π ( s 1 ) conditioned on s 0 = s . Here b r,γ ( x ) := r + γx for any x ∈ R , and f # ν ∈ P is defined as f # ν ( A ) := ν ( { x : f ( x ) ∈ A } ) for any function f : R → R , probability measure ν ∈ P and Borel set A ⊂ R . The distributional Bellman equation can also be written as η π = T π η π . The operator T π : P S → P S is called the distributional Bellman operator. In this task, our goal is to find η π for some given policy π .

Tabular Distributional TD Learning. In analogy to tabular TD learning (Eqn. (2)), in the tabular setting, we can solve the distributional Bellman equation by LSA and derive the distributional TD learning rule given the streaming data { ( s t , a t , r t ) } ∞ t =0 :

̸

<!-- formula-not-decoded -->

We comment the algorithm above is not computationally feasible as we need to manipulate infinitedimensional objects (return distributions) at each iteration.

Categorical Parametrization and Tabular Categorical TD Learning. In order to deal with return distributions in a computationally tractable manner, we consider the categorical parametrization as in Bellemare et al. [2017], Rowland et al. [2018, 2024b], Peng et al. [2024]. To be compatible with linear function approximation introduced in the next section, which cannot guarantee non-negative outputs, we will work with P sign , the signed measure space with total mass 1 as in Bellemare et al. [2019], Lyle et al. [2019], Bellemare et al. [2023] instead of standard probability space P ⊂ P sign :

<!-- formula-not-decoded -->

For any ν ∈ P sign , we define its cumulative distribution function (CDF) as F ν ( x ) := ν ([0 , x ]) . We can naturally define the L 2 and L 1 distances between CDFs as the Cram´ er distance ℓ 2 and 1 -Wasserstein distance W 1 in P sign , respectively. The distributional Bellman operator (see Eqn. (6)) can also be extended to the product space ( P sign ) S without modifying its definition.

The space of all categorical parametrized signed measures with total mass 1 is defined as

<!-- formula-not-decoded -->

which is an affine subspace of P sign . Here { x k = kι K } K k =0 are K +1 equally-spaced points of the support, ι K =[ K (1 -γ )] -1 is the gap between adjacent points, and p k is the 'probability' (may be negative) that ν assigns to x k . We define the categorical projection operator Π K : P sign → P sign K as

<!-- formula-not-decoded -->

Following Bellemare et al. [2023, Proposition 5.14], one can show that Π K ν ∈ P sign K is uniquely represented with a vector p ν = ( p k ( ν )) K -1 k =0 ∈ R K , where

<!-- formula-not-decoded -->

We lift Π K to the product space by defining [ Π K η ] ( s ) := Π K η ( s ) . One can check that the categorical Bellman operator Π K T π is a √ γ -contraction in the Polish space (( P sign K ) S , ℓ 2 ,µ π ) , where ℓ 2 ,µ π ( η 1 , η 2 ) := ( E s ∼ µ π [ ℓ 2 2 ( η 1 ( s ) , η 2 ( s ))]) 1 / 2 is the µ π -weighted Cram´ er distance between η 1 , η 2 ∈ ( P sign ) S . Similarly, W 1 ,µ π ( η 1 , η 2 ) := ( E s ∼ µ π [ W 2 1 ( η 1 ( s ) , η 2 ( s ))]) 1 / 2 . Hence, the categorical projected Bellman equation η = Π K T π η admits a unique solution η π,K , which satisfies W 1 ,µ π ( η π , η π,K ) ≤ (1 -γ ) -1 ℓ 2 ,µ π ( η π , Π K η π ) [Rowland et al., 2018, Proposition 3]. Applying LSA to solving the equation yields tabular categorical TD learning, and the iteration rule is given by

̸

<!-- formula-not-decoded -->

## 3 Linear-Categorical TD Learning

In this section, we propose our Linear-CTD algorithm (Eqn. (13)) by combining the linear function approximation (Eqn. (3)) with the categorical parametrization (Eqn. (7)). We first introduce the space of linear-categorical parametrized signed measures with total mass 1 :

<!-- formula-not-decoded -->

which is an affine subspace of ( P sign K ) S . Here p k ( s ; θ )= F k ( s ; θ ) -F k -1 ( s ; θ ) , and

<!-- formula-not-decoded -->

is CDF of η θ ( s ) at x k ( F -1 ( s ; · ) ≡ 0 , F K ( s ; · ) ≡ 1 ), where x k ↦→ ( k +1) / ( K +1) is the CDF of the discrete uniform distribution ν on { x 0 , . . . , x K } used for normalization 6 . In many cases, especially when formulating and implementing algorithms, it is much more convenient and efficient to work with the matrix version of the parameter Θ :=( θ (0) , . . ., θ ( K -1)) ∈ R d × K rather than with θ = vec ( Θ ) . We define the linear-categorical projection operator Π π ϕ ,K : ( P sign ) S → P sign ϕ ,K as follows:

<!-- formula-not-decoded -->

Π π ϕ ,K is in fact an orthogonal projection (see Proposition D.2), and thus is non-expansive. The following proposition characterizes Π π ϕ ,K , whose proof can be found in Appendix D.2.

Proposition 3.1. For any η ∈ ( P sign ) S , Π π ϕ ,K η is uniquely given by η ˜ θ , where ˜ θ = vec ( ˜ Θ ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following proposition characterizes the categorical projected Bellman operator Π K T π when it acts on the parametrized η θ , whose proof can be found in Appendix D.3.

Proposition 3.2. For any θ ∈ R dK and s ∈ S , we abbreviate p T π η θ ( s ) as ˜ p θ ( s ) , then

<!-- formula-not-decoded -->

where for any r ∈ [0 , 1] and j, k ∈ { 0 , 1 , . . . , K } ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

6 The normalizer ν is indispensable for achieving tight sample complexity bound, and it also guarantees our estimator has total mass 1 , making the estimator more interpretable.

Since Π π ϕ ,K T π is a √ γ -contraction in ( P sign ϕ ,K , ℓ 2 ,µ π ) ( T π is √ γ -contraction [Bellemare et al., 2023, Lemma 9.14]), in Theorem 4.1, using Proposition 3.1 and 3.2, we generalize the linear projected Bellman equation (Eqn. (4)) to the distributional setting. The proof can be found in Appendix D.4.

Theorem 3.1. The linear-categorical projected Bellman equation η θ = Π π ϕ ,K T π η θ admits a unique solution η θ ⋆ , where the matrix parameter Θ ⋆ is the unique solution to the linear system for Θ ∈ R d × K

<!-- formula-not-decoded -->

In analogy to the approximation bounds of ∥ V π -V ψ ⋆ ∥ µ π and W 1 ,µ π ( η π , η π,K ) , the following lemma answers how close η θ ⋆ is to η π , whose proof can be found in Appendix D.5.

Proposition 3.3 (Approximation Error of η θ ⋆ ) . It holds that

<!-- formula-not-decoded -->

where the first error term K -1 (1 -γ ) -3 is due to the categorical parametrization, and the second error term (1 -γ ) -2 ℓ 2 2 ,µ π ( Π K η π , Π π ϕ ,K η π ) is due to the additional linear function approximation.

As before, we solve Eqn. (12) by LSA and get Linear-CTD given the streaming data { ( s t , a t , r t ) } ∞ t =0 :

<!-- formula-not-decoded -->

for any t ≥ 1 , where α is the constant step size. It is easy to verify that, in the special case of the tabular setting ( ϕ ( s ) = ( 1 { s =˜ s } ) ˜ s ∈S ), Linear-CTD is equivalent to tabular categorical TD learning (Eqn. (9)). In this paper, we consider the Polyak-Ruppert tail averaging ¯ θ T := ( T/ 2+1) -1 ∑ T t = T/ 2 θ t (we use an even number T ) as in the analysis of Linear-TD in Samsonov et al. [2024b]. Standard theory of LSA [Mou et al., 2020] says under some conditions, if we take an appropriate step size α , ¯ θ T will converge to the solution θ ⋆ with rate T -1 / 2 as T →∞ . In Figure 1, we empirically validate the convergence of Linear-CTD through numerical experiments. See Appendix G for details of the numerical experiments.

Figure 1. Convergence results under varying K for our Linear-CTD algorithm with step size α = 0 . 01 . These curves exhibit similar trends, demonstrating our algorithm's robustness across different K values.

<!-- image -->

Remark 1 (Comparison with Existing Linear Distributional TD Learning Algorithms) . Our Linear-CTD can be regarded as a preconditioned version of vanilla stochastic semi-gradient descent (SSGD) with the probability mass function (PMF) representation [Bellemare et al., 2023, Section 9.6]. See Appendix D.6 for the PMF representation, and Appendix D.7 for a self-contained derivation of SSGD with PMF representations. The preconditioning technique is a commonly used methodology to accelerate solving optimization problems by reducing the condition number. We precondition the

vanilla algorithm by removing the matrix C ⊤ C (see Eqn. (26) ), whose condition number scales with K 2 (Lemma H.2). By introducing the preconditioner ( C ⊤ C ) -1 , our Linear-CTD (Eqn. (13) ) can achieve a convergence rate independent of K , which the vanilla form cannot achieve. See Remark 5 and Appendix G for theoretical and experimental evidence respectively.

We comment that Linear-CTD (Eqn. (13) ) is equivalent to SSGD with CDF representation, which was also considered in Lyle et al. [2019]. The difference is that our Linear-CTD normalizes the distribution so that the total mass of return distributions always be 1 , while the algorithm in Lyle et al. [2019] does not. See Appendix D.7 for a self-contained derivation of SSGD with CDF representations.

The previously mentioned works, as well as our Linear-CTD , are all limited by the use of signed measures (Eqn. (7) ), which makes them less interpretable. Bellemare et al. [2023, Section 9.5] proposed a softmax-based linear-categorical algorithm, which is more closer to the practical C51 algorithm [Bellemare et al., 2017] and it uses standard probability measures. However, the nonlinearity due to softmax makes it difficult for analysis. We will leave the analysis for it as future work.

Remark 2 ( Linear-CTD is mean-preserving) . A key property of Linear-CTD is mean preservation. That is, if we use identical initializations ( E G ∼ η θ 0 [ G ] = V ψ 0 ) and an identical data stream to update in both Linear-CTD and Linear-TD , it follows that E G ∼ η θ t [ G ] = V ψ t for all t . However, the mean-preserving property does not hold for the SSGD with the PMF representation. In this sense, our Linear-CTD is indeed the generalization of Linear-TD . See Appendix D.8 for details.

## 4 Non-Asymptotic Statistical Analysis

In our task, the quality of estimator η ¯ θ T is measured by µ π -weighted 1 -Wasserstein error W 1 ,µ π ( η ¯ θ T , η π ) . By triangle inequality, the error can be decomposed into the approximation error and the estimation error: W 1 ,µ π ( η π , η ¯ θ T ) ≤ W 1 ,µ π ( η π , η θ ⋆ )+ W 1 ,µ π ( η θ ⋆ , η ¯ θ T ) . Proposition 3.3 already provided an upper bound for the approximation error W 1 ,µ π ( η π , η θ ⋆ ) , so it suffices to control the estimation error W 1 ,µ π ( η θ ⋆ , η ¯ θ T ) , denoted L ( ¯ θ T ) .

In the following theorem, we give non-asymptotic convergence rates of L ( ¯ θ T ) . We start from the generative model setting, i.e. , in the t -th iteration, we collect samples s t ∼ µ π ( · ) , a t ∼ π ( ·| s t ) , ( r t , s ′ t ) ∼ P ( · , ·| s t , a t ) from the generative model, and we replace s t +1 with s ′ t in Eqn. (13). We give L p and high-probability convergence results in this setting. These results can be extended to the Markovian setting, i.e. , using the streaming data { ( s t , a t , r t ) } ∞ t =0 .

## 4.1 L 2 Convergence

We first provide non-asymptotic convergence rates of E 1 / 2 [( L ( ¯ θ T )) 2 ] , which do not grow with the number of supports K . The L p ( p &gt; 2 ) convergence results can be found in Theorem E.1.

Theorem 4.1 ( L 2 Convergence) . For any K ≥ (1 -γ ) -1 and α ∈ (0 , (1 - √ γ ) / 76) , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the upper bound, the first term of order T -1 / 2 is dominant. The second term of order T -1 has a worse dependence on α -1 , leading to a sample size barrier (Eqn. (15)). The third term, corresponding to the initialization error, decays at a geometric rate and can be thus ignored. To prove Theorem 4.1, we conduct a fine-grained analysis of the linear-categorical Bellman equation and apply the exponential stability argument [Samsonov et al., 2024b]. We outline the proof in Section 5. In Remark 3, we compare our Theorem 4.1 with the L 2 convergence rate of classic Linear-TD and conclude that learning the distribution of the return is as easy as learning its expectation (value function) with linear function approximation. Rowland et al. [2024b], Peng et al. [2024] discovered this phenomenon in the tabular setting, and we extended it to the function approximation setting.

Remark 3 (Comparison with Convergence Rate of Linear-TD ) . The only difference between our Theorem 4.1 and the tight L 2 convergence rate of classic Linear-TD (see Appendix C.2) lies in replacing ∥ ψ ⋆ ∥ Σ ϕ (resp. ∥ ψ 0 -ψ ⋆ ∥ ) in Linear-TD with ∥ θ ⋆ ∥ V 1 (resp. ∥ θ 0 -θ ⋆ ∥ V 2 ). We claim that the two pairs should be of the same order, respectively. Note that ∥ θ ⋆ ∥ V 1 and ∥ θ 0 -θ ⋆ ∥ V 2 are of order O ((1 -γ ) -1 ) if η θ ⋆ ( s ) and η θ 0 ( s ) are valid probability distributions for all s ∈ S . This is because in this case, F k ( s ; θ ) = ϕ ( s ) ⊤ θ ( k )+( k +1) / ( K +1) ∈ [0 , 1] for θ ∈ { θ ⋆ , θ 0 } . While in Linear-TD , ∥ ψ ⋆ ∥ Σ ϕ and ∥ ψ 0 -ψ ⋆ ∥ are also of order O ((1 -γ ) -1 ) if V ψ ( s ) = ϕ ( s ) ⊤ ψ ∈ [0 , (1 -γ ) -1 ] for all s ∈ S and ψ ∈ { ψ ⋆ , ψ 0 } . It is thus reasonable to consider the two pairs with the same order, respectively. Similar arguments also hold in other convergence results presented in this paper. Therefore, in this sense, our results match those of Linear-TD .

One can translate Theorem 4.1 into a sample complexity bound.

Corollary 4.1. Under the same conditions as in Theorem 4.1, for any ε &gt; 0 , suppose

<!-- formula-not-decoded -->

Then it holds that E 1 / 2 [( L ( ¯ θ T )) 2 ] ≤ ε .

Instance-Independent Step Size. If we take the largest possible instance-independent step size, i.e. , α ≃ (1 -γ ) , and consider ε ∈ (0 , 1) , we obtain the sample complexity bound

<!-- formula-not-decoded -->

Optimal Instance-Dependent Step Size. If we take the optimal instance-dependent step size α ≃ (1 -γ ) λ min which involves the unknown λ min , we obtain a better sample complexity bound

<!-- formula-not-decoded -->

There is a sample size barrier in the bound, that is, the dependence on λ min is the optimal λ -1 min only when ε = ˜ O ( √ λ min ) , or equivalently, we require a large enough (independent of ε ) update steps T .

These results match the recent results for classic Linear-TD with a constant step size [Li et al., 2024b, Samsonov et al., 2024b]. It is possible to break the sample size barrier in Eqn. (15) as in Linear-TD by applying variance-reduction techniques [Li et al., 2023a]. We leave it for future work.

## 4.2 Convergence with High Probability and Markovian Samples

Applying the L p convergence result (Theorem E.1) with p = 2log(1 /δ ) and Markov's inequality, we immediately obtain the high-probability convergence result.

Theorem 4.2 (High-Probability Convergence) . For any ε &gt; 0 and δ ∈ (0 , 1) , suppose K ≥ (1 -γ ) -1 , α ∈ (0 , (1 - √ γ ) / [38 log( T/δ 2 )]) , and

<!-- formula-not-decoded -->

Then with probability at least 1 -δ , it holds that L ( ¯ θ T ) ≤ ε . Here, the ˜ O ( · ) does not hide polynomials of log(1 /δ ) (but hides logarithm terms of log(1 /δ ) ).

Again, we will obtain concrete sample complexity bounds as in Eqn. (14) or Eqn. (15) if we use different step sizes. Compared with the theoretical results for classic Linear-TD , our results match Samsonov et al. [2024b, Theorem 4]. Samsonov et al. [2024b] also considered the constant step size, but obtained a worse dependence on log (1 /δ ) than Wu et al. [2024, Theorem 4] which uses the polynomial-decaying step size α t = α 0 t -β with β ∈ (1 / 2 , 1) instead.

Remark 4 (Markovian Setting) . Using the same argument as in the proof of Samsonov et al. [2024b, Theorem 6], one can immediately derive a high-probability sample complexity bound in the Markovian setting. Compared with the bound in the generative model setting (Theorem 4.2), the bound in the Markovian setting will have an additional dependency on t mix log( T/δ ) , where t mix is the mixing time of the Markov chain { s t } ∞ t =0 in S . We omit this result for brevity.

## 5 Proof Outlines

In this section, we outline the proofs of our main results (Theorem 4.1). We first state the theoretical properties of the linear-categorical Bellman equation and the exponential stability of Linear-CTD . Finally, we highlight some key steps in proving these results.

## 5.1 Vectorization of Linear-CTD

In our analysis, it will be more convenient to work with the vectorization version of the updating scheme of Linear-CTD (Eqn. (13)):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We denote by ¯ A and ¯ b the expectations of A t and b t , respectively. Using exponential stability arguments, we can derive an upper bound for ∥ ¯ A ( ¯ θ T -θ ⋆ ) ∥ . The following lemma further translates it to an upper bound for L ( ¯ θ T ) = W 1 ,µ π ( η θ ⋆ , η ¯ θ T ) , whose proof is given in Appendix E.1.

<!-- formula-not-decoded -->

## 5.2 Exponential Stability Analysis

First, we introduce some notations. Letting e t : = A t θ ⋆ -b t , we denote by C A (resp. C e ) the almost sure upper bound for max {∥ A t ∥ , ∥ A t -¯ A ∥} (resp. ∥ e t ∥ ), and Σ e := E [ e t e ⊤ t ] the covariance matrix of e t . The following lemma provides useful upper bounds, whose proof is given in Appendix E.2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let Γ ( α ) t := ∏ t i =1 ( I -α A i ) for any α &gt; 0 and t ∈ N . The exponential stability of Linear-CTD is summarized in the following lemma, whose proof can be found in Appendix E.3.

<!-- formula-not-decoded -->

The following theorem states the L 2 convergence of ∥ ¯ A ( ¯ θ T -θ ⋆ ) ∥ based on exponential stability arguments. For the general L p convergence, please refer to Samsonov et al. [2024b, Theorem 2].

Theorem 5.1. [Samsonov et al., 2024b, Theorem 1] For any α ∈ (0 , α 2 , ∞ ) , it holds that

<!-- formula-not-decoded -->

Combining these lemmas with Theorem 5.1, we can immediately obtain Theorem 4.1.

Remark 5 (Convergence of SSGD with PMF representation) . In Appendix E.5, we give counterparts of these lemmas and Theorem 4.1 for vanilla SSGD with PMF representation. The results imply that the step size in the algorithm should scale with (1 - √ γ ) /K 2 and the sample complexity grows with K . In Appendix G.2, we verify through numerical experiments that as K increases, to ensure convergence, the step size of the vanilla algorithm indeed needs to decay at a rate of K -2 . In contrast, the step size of our Linear-CTD does not need to be adjusted when K increases. Moreover, we find that Linear-CTD empirically consistently outperforms the vanilla algorithm under different K .

## 5.3 Key Steps in the Proofs

Here we highlight some key steps in proving the above theoretical results.

Bounding the Spectral Norm of Expectation of Kronecker Products. In proving that the L ( θ ) can be upper-bounded by ∥ ¯ A ( θ -θ ⋆ ) ∥ (Lemma 5.1), as well as in verifying the exponential stability condition (Lemma 5.3), one of the most critical steps is to show

<!-- formula-not-decoded -->

By Lemma H.3, we have ∥ C ˜ G ( r ) C -1 ∥≤ √ γ for any r ∈ [0 , 1] . In addition, one can check that ∥ E s,s ′ [ Σ -1 / 2 ϕ ϕ ( s ) ϕ ( s ′ ) ⊤ Σ -1 / 2 ϕ ] ∥≤ 1 . One may speculate that the property ∥ B 1 ⊗ B 2 ∥ = ∥ B 1 ∥∥ B 2 ∥ (Lemma A.3) is enough to get the desired conclusion. However, the two matrices in the Kronecker product are not independent, preventing us from using this simple property to derive the conclusion. On the other hand, since we only have the upper bound E s,s ′ [ ∥ Σ -1 / 2 ϕ ϕ ( s ) ϕ ( s ′ ) ⊤ Σ -1 / 2 ϕ ∥ ] ≤ d , simply moving the expectation in Eqn. (17) outside the norm will lead to a loose d √ γ bound. To resolve this problem, we leverage the fact that the second matrix is rank1 and prove the following result. The proof can be found in the derivation following Eqn. (29).

Lemma 5.4. For any random matrix Y and random vectors x , z , suppose ∥ Y ∥ ≤ C Y almost surely, E [ xx ⊤ ] ≼ C x I d 1 and E [ zz ⊤ ] ≼ C z I d 2 for some constants C Y , C x , C z &gt; 0 . Then it holds that

<!-- formula-not-decoded -->

Remark 6 (Matrix Representation of Categorical Projected Bellman operator) . The matrix C ˜ G ( r ) C -1 also appears in Rowland et al. [2024b, Proposition B.2] as the matrix representation of the categorical projected Bellman operator Π K T π of a specific one-state MDP. As a result, ∥ C ˜ G ( r ) C -1 ∥≤ √ γ because Π K T π is a √ γ -contraction in ( P , ℓ 2 ) . Our Lemma H.3 provides a new analysis by directly analyzing the matrix.

Bounding the Norm of b t . In proving Lemma 5.2, the most involved step is to upper-bound ∥ b t ∥ (Eqn. (16)). To this end, we need to upper-bound the following term:

<!-- formula-not-decoded -->

Term (18) is also related to the categorical projected Bellman operator Π K T π . Specifically, let ν = 1 K +1 ∑ K k =0 δ x k be the discrete uniform distribution. One can show that Term (18) equals

<!-- formula-not-decoded -->

where we used the fact that Π K is non-expansive and an upper bound for ℓ 2 (( b r,γ ) # ( ν ) , ν ) when K ≥ (1 -γ ) -1 (Lemma H.4). The full proof can be found in the derivation following Eqn. (30).

## 6 Conclusions

In this paper, we have bridged a critical theoretical gap in distributional reinforcement learning by establishing the non-asymptotic sample complexity of distributional TD learning with linear function approximation. Specifically, we have proposed Linear-CTD , which is derived by solving the linear-categorical projected Bellman equation. By carefully analyzing the Bellman equation and using the exponential stability arguments, we have shown tight sample complexity bounds for the proposed algorithm. Our finite-sample rates match the state-of-the-art sample complexity bounds for conventional TD learning. These theoretical findings demonstrate that learning the full return distribution under linear function approximation can be statistically as easy as conventional TD learning for value function estimation. Our numerical experiments have provided empirical validation of our theoretical results. Finally, we have noted that it would be possible to improve the convergence rates by applying variance-reduction techniques or using polynomial-decaying step sizes, which we leave for future work.

## Acknowledgments and Disclosure of Funding

The authors would like to thank the reviewers for their constructive feedback, which helped us improve the quality of our work.

This work has been supported by the National Key Research and Development Project of China (No. 2022YFA1004002) and the National Natural Science Foundation of China (No. 12271011 and No. 12350001).

## References

- B. ´ Avila Pires, M. Rowland, D. Borsa, Z. D. Guo, K. Khetarpal, A. Barreto, D. Abel, R. Munos, and W. Dabney. Optimizing return distributions with distributional dynamic programming. Journal of Machine Learning Research , 26(185):1-90, 2025. URL http://jmlr.org/papers/v26/ 25-0210.html .
- N. B¨ auerle and J. Ott. Markov decision processes with average-value-at-risk criteria. Mathematical Methods of Operations Research , 74:361-379, 2011.
- M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning. In International conference on machine learning , pages 449-458. PMLR, 2017.
- M. G. Bellemare, N. Le Roux, P. S. Castro, and S. Moitra. Distributional reinforcement learning with linear function approximation. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 2203-2211. PMLR, 2019.
- M. G. Bellemare, W. Dabney, and M. Rowland. Distributional Reinforcement Learning . MIT Press, 2023. http://www.distributional-rl.org .
- D. P. Bertsekas and J. N. Tsitsiklis. Neuro-dynamic programming: an overview. In Proceedings of 1995 34th IEEE conference on decision and control , volume 1, pages 560-564. IEEE, 1995.
- J. Bhandari, D. Russo, and R. Singal. A finite time analysis of temporal difference learning with linear function approximation. In Conference on learning theory , pages 1691-1692. PMLR, 2018.
- M. B¨ ock and C. Heitzinger. Speedy categorical distributional reinforcement learning and complexity analysis. SIAM Journal on Mathematics of Data Science , 4(2):675-693, 2022. doi: 10.1137/ 20M1364436. URL https://doi.org/10.1137/20M1364436 .
- K. Chen. Matrix preconditioning techniques and applications . Cambridge University Press, 2005.
- Z. Chen, S. T. Maguluri, S. Shakkottai, and K. Shanmugam. A lyapunov theory for finite-sample guarantees of markovian stochastic approximation. Operations Research , 72(4):1352-1367, 2024.
- T. Cho, S. Han, K. Lee, S. Ju, D. Kim, and J. Lee. Tractable and provably efficient distributional reinforcement learning with general value function approximation. arXiv e-prints , pages arXiv2407, 2024.
- Y. Chow and M. Ghavamzadeh. Algorithms for cvar optimization in mdps. Advances in neural information processing systems , 27, 2014.
- W. Dabney, G. Ostrovski, D. Silver, and R. Munos. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning , pages 1096-1105. PMLR, 2018a.
- W. Dabney, M. Rowland, M. Bellemare, and R. Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI conference on artificial intelligence , volume 32, 2018b.
- G. Dalal, B. Sz¨ or´ enyi, G. Thoppe, and S. Mannor. Finite sample analyses for td (0) with function approximation. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- Y. Duan and M. J. Wainwright. A finite-sample analysis of multi-step temporal difference estimates. In Learning for Dynamics and Control Conference , pages 612-624. PMLR, 2023.

- A. Durmus, E. Moulines, A. Naumov, and S. Samsonov. Finite-time high-probability bounds for polyak-ruppert averaged iterates of linear stochastic approximation. Mathematics of Operations Research , 2024.
- D. Freirich, T. Shimkin, R. Meir, and A. Tamar. Distributional multivariate policy evaluation and exploration with the bellman gan. In International Conference on Machine Learning , pages 1983-1992. PMLR, 2019.
- C. D. Godsil. Inverses of trees. Combinatorica , 5:33-39, 1985.
- R. A. Horn and C. R. Johnson. Topics in matrix analysis . Cambridge university press, 1994.
- D. Huo, Y. Chen, and Q. Xie. Bias and extrapolation in markovian linear stochastic approximation with constant stepsizes. In Abstract Proceedings of the 2023 ACM SIGMETRICS International Conference on Measurement and Modeling of Computer Systems , pages 81-82, 2023.
- T. Kastner, M. A. Erdogdu, and A.-m. Farahmand. Distributional model equivalence for risk-sensitive reinforcement learning. Advances in Neural Information Processing Systems , 36:56531-56552, 2023.
- T. Kastner, M. Rowland, Y. Tang, M. A. Erdogdu, and A. massoud Farahmand. Categorical distributional reinforcement learning with kullback-leibler divergence: Convergence and asymptotics. In Forty-second International Conference on Machine Learning , 2025. URL https://openreview.net/forum?id=f4qxkR6GQK .
- C. Lakshminarayanan and C. Szepesvari. Linear stochastic approximation: How far does constant step-size and iterate averaging go? In International conference on artificial intelligence and statistics , pages 1347-1355. PMLR, 2018.
- G. Li, C. Cai, Y . Chen, Y . Wei, and Y . Chi. Is q-learning minimax optimal? a tight sample complexity analysis. Operations Research , 72(1):222-236, 2024a.
- G. Li, W. Wu, Y. Chi, C. Ma, A. Rinaldo, and Y. Wei. High-probability sample complexities for policy evaluation with linear function approximation. IEEE Transactions on Information Theory , 2024b.
- T. Li, G. Lan, and A. Pananjady. Accelerated and instance-optimal policy evaluation with linear function approximation. SIAM Journal on Mathematics of Data Science , 5(1):174-200, 2023a.
- X. Li, J. Liang, and Z. Zhang. Online statistical inference for nonlinear stochastic approximation with markovian data. arXiv preprint arXiv:2302.07690 , 2023b.
13. X.-L. Li. Preconditioned stochastic gradient descent. IEEE transactions on neural networks and learning systems , 29(5):1454-1466, 2017.
- S. H. Lim and I. Malik. Distributional reinforcement learning for risk-sensitive policies. Advances in Neural Information Processing Systems , 35:30977-30989, 2022.
- Y. Luo, G. Liu, H. Duan, O. Schulte, and P. Poupart. Distributional reinforcement learning with monotonic splines. In International Conference on Learning Representations , 2022. URL https: //openreview.net/forum?id=C8Ltz08PtBp .
- C. Lyle, M. G. Bellemare, and P. S. Castro. A comparative analysis of expected and distributional reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 4504-4511, 2019.
- A. Marthe, A. Garivier, and C. Vernade. Beyond average return in markov decision processes. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. URL https: //openreview.net/forum?id=mgNu8nDFwa .
- M. Moghimi and H. Ku. Beyond CVar: Leveraging static spectral risk measures for enhanced decisionmaking in distributional reinforcement learning. In Forty-second International Conference on Machine Learning , 2025. URL https://openreview.net/forum?id=WeMpvGxXMn .

- T. Morimura, M. Sugiyama, H. Kashima, H. Hachiya, and T. Tanaka. Nonparametric return distribution approximation for reinforcement learning. In Proceedings of the 27th International Conference on Machine Learning (ICML-10) , pages 799-806, 2010.
- W. Mou, C. J. Li, M. J. Wainwright, P. L. Bartlett, and M. I. Jordan. On linear stochastic approximation: Fine-grained polyak-ruppert and non-asymptotic concentration. In Conference on Learning Theory , pages 2947-2997. PMLR, 2020.
- W. Mou, A. Pananjady, M. Wainwright, and P. Bartlett. Optimal and instance-dependent guarantees for markovian linear stochastic approximation. In Conference on Learning Theory , pages 2060-2061. PMLR, 2022.
- T. Nguyen-Tang, S. Gupta, and S. Venkatesh. Distributional reinforcement learning via moment matching. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 9144-9152, 2021.
- E. Noorani, C. N. Mavridis, and J. S. Baras. Exponential td learning: A risk-sensitive actor-critic reinforcement learning algorithm. In 2023 American Control Conference (ACC) , pages 4104-4109. IEEE, 2023.
- G. Patil, L. Prashanth, D. Nagaraj, and D. Precup. Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation. In International Conference on Artificial Intelligence and Statistics , pages 5438-5448. PMLR, 2023.
- Y. Peng, L. Zhang, and Z. Zhang. Statistical efficiency of distributional temporal difference learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=eWUM5hRYgH .
- B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization , 30(4):838-855, 1992.
- Z. Qi, C. Bai, Z. Wang, and L. Wang. Distributional off-policy evaluation in reinforcement learning. Journal of the American Statistical Association , jun 2025. doi: 10.1080/01621459.2025.2506197. Forthcoming.
- C. Qu, S. Mannor, and H. Xu. Nonlinear distributional gradient temporal-difference learning. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 5251-5260. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/qu19b.html .
- M. Rowland, M. Bellemare, W. Dabney, R. Munos, and Y. W. Teh. An analysis of categorical distributional reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 29-37. PMLR, 2018.
- M. Rowland, R. Dadashi, S. Kumar, R. Munos, M. G. Bellemare, and W. Dabney. Statistics and samples in distributional reinforcement learning. In International Conference on Machine Learning , pages 5528-5536. PMLR, 2019.
- M. Rowland, Y. Tang, C. Lyle, R. Munos, M. G. Bellemare, and W. Dabney. The statistical benefits of quantile temporal-difference learning for value estimation. In International Conference on Machine Learning , pages 29210-29231. PMLR, 2023.
- M. Rowland, R. Munos, M. G. Azar, Y. Tang, G. Ostrovski, A. Harutyunyan, K. Tuyls, M. G. Bellemare, and W. Dabney. An analysis of quantile temporal-difference learning. Journal of Machine Learning Research , 25:1-47, 2024a.
- M. Rowland, L. K. Wenliang, R. Munos, C. Lyle, Y. Tang, and W. Dabney. Near-minimax-optimal distributional reinforcement learning with a generative model. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024b. URL https://openreview. net/forum?id=JXKbf1d4ib .
- D. Ruppert. Efficient estimations from a slowly convergent robbins-monro process. Technical report, Cornell University Operations Research and Industrial Engineering, 1988.

- S. Samsonov, E. Moulines, Q.-M. Shao, Z.-S. Zhang, and A. Naumov. Gaussian approximation and multiplier bootstrap for polyak-ruppert averaged linear stochastic approximation with applications to TD learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024a. URL https://openreview.net/forum?id=S0Ci1AsJL5 .
- S. Samsonov, D. Tiapkin, A. Naumov, and E. Moulines. Improved high-probability bounds for the temporal difference learning algorithm via exponential stability. In The Thirty Seventh Annual Conference on Learning Theory , pages 4511-4547. PMLR, 2024b.
- D. Serre. Matrices: Theory and Applications . Graduate texts in mathematics. Springer, 2002. ISBN 9780387954608. URL https://books.google.com.hk/books?id=RDnUIFYgkrUC .
- G. Shen, R. Dai, G. Wu, S. Luo, C. Shi, and H. Zhu. Deep distributional learning with non-crossing quantile network. arXiv preprint arXiv:2504.08215 , 2025.
- R. Srikant and L. Ying. Finite-time error bounds for linear stochastic approximation andtd learning. In Conference on Learning Theory , pages 2803-2830. PMLR, 2019.
- K. Sun, Y. Zhao, W. Liu, B. Jiang, and L. Kong. Distributional reinforcement learning with regularized wasserstein loss. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=CiEynTpF28 .
- R. S. Sutton. Learning to predict by the methods of temporal differences. Machine learning , 3:9-44, 1988.
- Y. Tang, R. Munos, M. Rowland, B. Avila Pires, W. Dabney, and M. Bellemare. The nature of temporal difference errors in multi-step distributional reinforcement learning. Advances in Neural Information Processing Systems , 35:30265-30276, 2022.
- Y. Tang, M. Rowland, R. Munos, B. ´ A. Pires, and W. Dabney. Off-policy distributional q ( lambda ): Distributional rl without importance sampling. arXiv preprint arXiv:2402.05766 , 2024.
- J. Tsitsiklis and B. Van Roy. Analysis of temporal-diffference learning with function approximation. Advances in neural information processing systems , 9, 1996.
- R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018. doi: 10.1017/9781108231596.
- H. Wang and X. Y. Zhou. Continuous-time mean-variance portfolio selection: A reinforcement learning framework. Mathematical Finance , 30(4):1273-1308, 2020.
- K. Wang, K. Zhou, R. Wu, N. Kallus, and W. Sun. The benefits of being distributional: Small-loss bounds for reinforcement learning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 2275-2312. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 06fc38f5c21ae66ef955e28b7a78ece5-Paper-Conference.pdf .
- K. Wang, O. Oertell, A. Agarwal, N. Kallus, and W. Sun. More benefits of being distributional: Second-order bounds for reinforcement learning. In Forty-first International Conference on Machine Learning , 2024. URL https://openreview.net/forum?id=kZBCFQe1Ej .
- L. Wang, Y. Zhou, R. Song, and B. Sherwood. Quantile-optimal treatment regimes. Journal of the American Statistical Association , 113(523):1243-1254, 2018.
- L. K. Wenliang, G. Deletang, M. Aitchison, M. Hutter, A. Ruoss, A. Gretton, and M. Rowland. Distributional bellman operators over mean embeddings. In Forty-first International Conference on Machine Learning , 2024. URL https://openreview.net/forum?id=j2pLfsBm4J .
- H. Wiltzer, J. Farebrother, A. Gretton, and M. Rowland. Foundations of multivariate distributional reinforcement learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024a. URL https://openreview.net/forum?id=aq3I5B6GLG .

- H. Wiltzer, J. Farebrother, A. Gretton, Y. Tang, A. Barreto, W. Dabney, M. G. Bellemare, and M. Rowland. A distributional analogue to the successor representation. In International Conference on Machine Learning (ICML) , 2024b.
- R. Wu, M. Uehara, and W. Sun. Distributional offline policy evaluation with predictive error guarantees. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 37685-37712. PMLR, 23-29 Jul 2023. URL https://proceedings.mlr.press/v202/wu23s.html .
- W. Wu, G. Li, Y. Wei, and A. Rinaldo. Statistical Inference for Temporal Difference Learning with Linear Function Approximation. arXiv preprint arXiv:2410.16106 , 2024.
- D. Yang, L. Zhao, Z. Lin, T. Qin, J. Bian, and T.-Y. Liu. Fully parameterized quantile function for distributional reinforcement learning. Advances in neural information processing systems , 32, 2019.
- Y. Yue, Z. Wang, and M. Zhou. Implicit distributional reinforcement learning. Advances in Neural Information Processing Systems , 33:7135-7147, 2020.
- H. Zhang and F. Ding. On the kronecker products and their applications. Journal of Applied Mathematics , 2013(1):296185, 2013.
- L. Zhang, Y. Peng, J. Liang, W. Yang, and Z. Zhang. Estimation and Inference in Distributional Reinforcement Learning. The Annals of Statistics , 2025.
- F. Zhou, Z. Zhu, Q. Kuang, and L. Zhang. Non-decreasing quantile function network with efficient exploration for distributional reinforcement learning. In Z.-H. Zhou, editor, Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21 , pages 3455-3461. International Joint Conferences on Artificial Intelligence Organization, 8 2021. doi: 10.24963/ ijcai.2021/476. URL https://doi.org/10.24963/ijcai.2021/476 . Main Track.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We justify our claims in the abstract and introduction using rigorous proof.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations and future work in Section Conclusions.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide full assumptions and proof.

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

Justification: We disclose all details of experiments.

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

## Answer: [Yes]

Justification: We provide the code in supplemental material.

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

Justification: We disclose all details of experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: N/A.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer 'Yes' if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
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

Justification: We provide full information on the computer resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research confirms with the NeurIPS Code of Ethics in every respect.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our paper only focuses on the theory of RL, and there is no societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our paper only focuses on theory of RL, there is no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly use and cite these assets.

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

Justification: We do not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in our paper does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Kronecker Product

In this section, we will introduce some properties of Kronecker product used in our paper. See Zhang and Ding [2013] for a detailed treatment of Kronecker product.

For any matrices A ∈ R m × n and B ∈ R p × q , the Kronecker product A ⊗ B is an matrix in R mp × nq , defined as

<!-- formula-not-decoded -->

Lemma A.1. The Kronecker product is bilinear and associative. Furthermore, for any matrices B 1 , B 2 , B 3 , B 4 such that B 1 B 3 , B 2 B 4 can be defined, it holds that ( B 1 ⊗ B 2 ) ( B 3 ⊗ B 4 ) = ( B 1 B 3 ) ⊗ ( B 2 B 4 ) (mixed-product property).

Proof. See Basic properties and Theorem 3 in Zhang and Ding [2013].

Lemma A.2. For any matrices B 1 , B 2 , B 3 such that B 1 B 2 B 3 can be defined, it holds that vec ( B 1 B 2 B 3 ) = ( B ⊤ 3 ⊗ B 1 ) vec ( B 2 ) .

Proof.

<!-- formula-not-decoded -->

LemmaA.3. For any matrices B 1 and B 2 , it holds that ∥ B 1 ⊗ B 2 ∥ = ∥ B 1 ∥ ∥ B 2 ∥ , ( B 1 ⊗ B 2 ) ⊤ = B ⊤ 1 ⊗ B ⊤ 2 . Furthermore, if B 1 and B 2 are invertible/orthogonal/diagonal/symmetric/normal, B 1 ⊗ B 2 is also invertible/orthogonal/diagonal/symmetric/normal and ( B 1 ⊗ B 2 ) -1 = B -1 1 ⊗ B -1 2 .

Proof. See Basic properties, Theorem 5 and Theorem 7 in Zhang and Ding [2013].

Lemma A.4. For any K,d ∈ N and PSD matrices B 1 , B 3 ∈ R K × K , B 2 , B 4 ∈ R d × d with B 1 ≼ B 3 and B 2 ≼ B 4 , it holds that B 1 ⊗ B 2 , B 3 ⊗ B 4 are also PSD matrices, furthermore, B 1 ⊗ B 2 ≼ B 3 ⊗ B 4 .

Proof. Consider the spectral decomposition B i = Q i D i Q ⊤ i , for any i ∈ [4] , by Lemma A.1 and Lemma A.3, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and are also spectral decomposition of ( B 1 ⊗ B 2 ) and ( B 3 ⊗ B 4 ) respectively. It is easy to see that they are PSD. Furthermore,

<!-- formula-not-decoded -->

LemmaA.5. For any K,d,d 1 , d 2 ∈ N , vectors u , v ∈ R d and matrices B 1 ∈ R K × d 1 , B 2 ∈ R d 2 × K , B 3 ∈ R K × K , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, for any matrix B 4 ∈ R d 1 × d 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

And in the same way,

<!-- formula-not-decoded -->

Furthermore,

<!-- formula-not-decoded -->

## B Related Work

Comparisons of Theoretical Results with the Previous Work In Table 1, we summarize our main theoretical results and comparing them with prior work. In the table, when the task is policy evaluation, the sample complexity is defined in terms of the µ π -weighted L 2 norm as the measure of error; when the task is distributional policy evaluation, the sample complexity is defined in terms of the µ π -weighted W 1 metric as the measure of error. The table gives a clear comparison of theoretical results with prior work.

Distributional Reinforcement Learning. Distributional TD learning was first proposed in Bellemare et al. [2017]. Following the distributional perspective in Bellemare et al. [2017], Qu et al. [2019] proposed a distributional version of the gradient TD learning algorithm, Tang et al. [2022] proposed a distributional version of multi-step TD learning, Tang et al. [2024] proposed a distributional version of off-policy Q( λ ) and TD( λ ) algorithms, and Wu et al. [2023] proposed a distributional version of fitted Q evaluation to solve the distributional offline policy evaluation problem. Qi et al. [2025] also considered the distributional off-policy evaluation problem. Wiltzer et al. [2024b] proposed an approach for evaluating the return distributions for all policies simultaneously when the reward is deterministic or in the finite-horizon setting. Wiltzer et al. [2024a] studied distributional policy evaluation in the multivariate reward setting and proposed corresponding TD learning algorithms. Beyond the tabular setting, Bellemare et al. [2019], Lyle et al. [2019], Bellemare et al. [2023] proposed various distributional TD learning algorithms with linear function approximation under different parametrizations.

A series of recent studies have focused on the theoretical properties of distributional TD learning. Rowland et al. [2018], B¨ ock and Heitzinger [2022], Zhang et al. [2025], Rowland et al. [2024a,b],

Table 1. Sample complexity of algorithms for solving policy evaluation and distributional policy evaluation. Here, Vanilla Linear-CTD refers to the stochastic semi-gradient descent algorithm with the probability mass function representational (see Eqn. (27) in Appendix D.7). The contraction analysis in [Bellemare et al., 2023, Section 9.7] means that they provided a contraction analysis for the dynamic programming version of Vanilla Linear-CTD algorithm.

| Paper                                | Sample Complexity                                                      | Method                            |
|--------------------------------------|------------------------------------------------------------------------|-----------------------------------|
| Samsonov et al. [2024b]              | ˜ O ( ∥ ψ ⋆ ∥ 2 Σ ϕ +1 (1 - γ ) 2 λ min ( 1 ε 2 + 1 λ min ) )          | Linear-TD                         |
| Li et al. [2023a]                    | ˜ O ( ∥ ψ ⋆ ∥ 2 Σ ϕ +1 ε 2 (1 - γ ) 2 λ min )                          | Linear-TD with Variance Reduction |
| Bellemare et al. [2023, Section 9.7] | Contraction Analysis                                                   | Vanilla Linear-CTD                |
| This Work (Theorem E.2)              | ˜ O ( K 4 ( ∥ θ ⋆ ∥ 2 V 1 +1) (1 - γ ) 2 λ min ( 1 ε 2 + K 2 λ min ) ) | Vanilla Linear-CTD                |
| This Work (Theorem 4.1)              | ˜ O ( ∥ θ ⋆ ∥ 2 V 1 +1 (1 - γ ) 2 λ min ( 1 ε 2 + 1 λ min ) )          | Linear-CTD                        |

Peng et al. [2024], Kastner et al. [2025] analyzed the asymptotic and non-asymptotic convergence of distributional TD learning (or its model-based variants) in the tabular setting. Among these works, Rowland et al. [2024b], Peng et al. [2024] established that in the tabular setting, learning the full return distribution is statistically as easy as learning its expectation in the model-based and model-free settings, respectively. And Bellemare et al. [2019] provided an asymptotic convergence result for categorical TD learning with linear function approximation.

Beyond the problem of distributional policy evaluation, Rowland et al. [2023], Wang et al. [2023, 2024] showed that theoretically the classic value-based reinforcement learning could benefit from distributional reinforcement learning. B¨ auerle and Ott [2011], Chow and Ghavamzadeh [2014], Marthe et al. [2023], Noorani et al. [2023], Moghimi and Ku [2025], ´ Avila Pires et al. [2025] considered optimizing statistical functionals of the return, and proposed algorithms to solve this harder problem.

Stochastic Approximation. Our Linear-CTD falls into the category of LSA. The classic TD learning, as one of the most classic LSA problems, has been extensively studied [Bertsekas and Tsitsiklis, 1995, Tsitsiklis and Van Roy, 1996, Bhandari et al., 2018, Dalal et al., 2018, Patil et al., 2023, Duan and Wainwright, 2023, Li et al., 2024a,b, Samsonov et al., 2024a, Wu et al., 2024]. Among these works, Li et al. [2024b], Samsonov et al. [2024b] provided the tightest bounds for Linear-TD with constant step sizes, which is also considered in our paper. While Wu et al. [2024] established the tightest bounds for Linear-TD with polynomial-decaying step sizes.

For general stochastic approximation problems, extensive works [Lakshminarayanan and Szepesvari, 2018, Srikant and Ying, 2019, Mou et al., 2020, 2022, Huo et al., 2023, Li et al., 2023b, Durmus et al., 2024, Samsonov et al., 2024b, Chen et al., 2024] have provided solid theoretical understandings.

## C Omitted Results and Proofs in Section 2

## C.1 Linear Projected Bellman Equation

It is worth noting that, Π π ϕ : ( R S , ∥·∥ µ π ) → ( V ϕ , ∥·∥ µ π ) is an orthogonal projection.

We aim to derive Eqn. (4). It is easy to check that, for any V ∈ R S , Π π ϕ V is uniquely give by V ˜ ψ where

<!-- formula-not-decoded -->

Hence, by the definition of Bellman operator (Eqn. (1)), ψ ⋆ is the unique solution to the following system of linear equations for ψ ∈ R d

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

## C.2 Convergence Results for Linear TD Learning

It is worthy noting that, Linear-TD is equivalent to the stochastic semi-gradient descent (SSGD) update.

∥ ∥

In Linear-TD , our goal is to find a good estimator ˆ ψ such that ∥ ∥ V ˆ ψ -V ψ ⋆ ∥ ∥ µ π = ∥ ∥ ∥ ˆ ψ -ψ ⋆ ∥ ∥ ∥ Σ ϕ ≤ ε . Samsonov et al. [2024b] considered the Polyak-Ruppert tail averaging ¯ ψ T := ( T/ 2 + 1) -1 ∑ T t = T/ 2 ψ t , and showed that in the generative model setting with constant step size α ≃ (1 -γ ) λ min ,

<!-- formula-not-decoded -->

is sufficient to guarantee that ∥ ∥ V ¯ ψ T -V ψ ⋆ ∥ ∥ µ π ≤ ε . They also provided sample complexity bounds when taking the instance-independent ( i.e. , not dependent on unknown quantity) step size, and in the Markovian setting.

## C.3 Categorical Parametrization is an Isometry

Proposition C.1. The affine space ( P sign K , ℓ 2 ) is isometric with ( R K , √ ι K ∥·∥ C ⊤ C ) , in the sense that, for any ν p 1 , ν p 2 ∈ P sign K , it holds that ℓ 2 2 ( ν p 1 , ν p 2 ) = ι K ∥ p 1 -p 2 ∥ 2 C ⊤ C , where C is defined in Eqn. (11) .

Proof.

<!-- formula-not-decoded -->

## C.4 Categorical Projection Operator is Orthogonal Projection

Proposition C.2. [Bellemare et al., 2023, Lemma 9.17] For any ν ∈ P sign and ν p ∈ P sign K , it holds

<!-- formula-not-decoded -->

## C.5 Categorical Projected Bellman Operator

The following lemma characterizing Π K T π is useful for both practice and theoretical analysis. Proposition C.3. For any η ∈ ( P sign ) S and s ∈ S , it holds that

<!-- formula-not-decoded -->

And in the same way, for any r ∈ [0 , 1] and s ′ ∈ S , it holds that

<!-- formula-not-decoded -->

where ˜ G and g is defined in Theorem 3.1.

This proposition is a special case of Proposition 3.2, whose proof can be found in Appendix D.3.

## D Omitted Results and Proofs in Section 3

## D.1 Linear-Categorical Parametrization is an Isometry

Proposition D.1. The affine space ( P sign ϕ ,K , ℓ 2 ,µ π ) is isometric with ( R dK , √ ι K ∥·∥ I K ⊗ Σ ϕ ) , in the sense that, for any η θ 1 , η θ 2 ∈ P sign ϕ ,K , it holds that ℓ 2 2 ,µ π ( η θ 1 , η θ 2 ) = ι K ∥ θ 1 -θ 2 ∥ 2 I K ⊗ Σ ϕ .

Proof. For any η θ ∈ P sign ϕ ,K , we denote F θ ( s ) = ( F k ( s ; θ )) K -1 k =0 ∈ R K , then it holds that

<!-- formula-not-decoded -->

## D.2 Linear-Categorical Projection Operator

Proposition 3.1 is an immediate corollary of the following lemma. For any ν ∈ P sign K , we define F ν = ( F k ( ν )) K -1 k =0 = ( ν ([0 , x k ])) K -1 k =0 ∈ R K , and for any η ∈ ( P sign ) S , we define p η ( s ) = p Π K η ( s ) and F η ( s ) = F Π K η ( s ) .

Lemma D.1. For any η ∈ ( P sign ) S , θ ∈ R dK and s ∈ S , it holds that

<!-- formula-not-decoded -->

Furthermore, it holds that

<!-- formula-not-decoded -->

Proof. According to Proposition C.2, one has

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also have the following matrix representation:

<!-- formula-not-decoded -->

Proposition D.2. For any η ∈ ( P sign ) S and η θ ∈ P sign ϕ ,K , it holds that

<!-- formula-not-decoded -->

The proof is straightforward and almost the same as that of Proposition C.2 if we utilize the affine structure.

## D.3 Proof of Proposition 3.2

Proof. Recall the definition of the distributional Bellman operator Eqn. (6) and categorical projection operator Eqn. (8), we have

<!-- formula-not-decoded -->

Hence, let W = Θ C -⊤ and w = vec ( W ) = ( C -1 ⊗ I d ) θ (see Appendix D.6 for their meaning), then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

## D.4 Linear-Categorical Projected Bellman Equation (Proof of Theorem 3.1)

Combining Proposition 3.1 with Proposition 3.2, we know that Θ ⋆ is the unique solution to the following system of linear equations for Θ ∈ R d × K

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

which is the desired conclusion. The uniqueness and existence of the solution is guaranteed by the fact that the LHS is an invertible linear transformation of Θ , which is justified by Eqn. (28).

## D.5 Proof of Proposition 3.3

Proof. By the basic inequality (Lemma H.1), we only need to show

<!-- formula-not-decoded -->

where we used Bellemare et al. [2023, Proposition 9.18 and Eqn. (5.28)].

## D.6 Linear-Categorical Parametrization with Probability Mass Function Representation

We introduce new notations for linear-categorical parametrization with probability mass function (PMF) representation. Let W := Θ C -⊤ = ( θ (0) , θ (1) -θ (0) , · · · , θ ( K -1) -θ ( K -2)) ∈ R d × K and the vectorization of W , w := vec ( W ) = ( C -1 ⊗ I d ) θ ∈ R dK . We abbreviate p η θ as p w in this section. Then by Lemma A.2, for any s ∈ S , it holds that

<!-- formula-not-decoded -->

PMF and CDF representations are equivalent because C is invertible.

For any η w 1 , η w 2 ∈ P sign ϕ ,K , by Proposition C.1,

<!-- formula-not-decoded -->

hence the affine space ( P sign ϕ ,K , ℓ 2 ,µ π ) is isometric with the Euclidean space ( R Kd , √ ι K ∥·∥ ( C ⊤ C ) ⊗ Σ ϕ ) if we consider the PMF representation.

Following the proof of Lemma D.1 in Appendix D.2, we can also derive the gradient when we use the PMF parametrization:

<!-- formula-not-decoded -->

## D.7 Stochastic Semi-Gradient Descent with Linear Function Approximation

We denote by T π t the corresponding empirical distributional Bellman operator at the t -th iteration, which satisfies

<!-- formula-not-decoded -->

## D.7.1 Probability Mass Function Representation

Consider the stochastic semi-gradient descent (SSGD) with the probability mass function (PMF) representation

<!-- formula-not-decoded -->

where ∇ W stands for taking gradient w.r.t. W t -1 ∈ R d × K in the first term η w t -1 ( s t ) (the second term is regarding as a constant, that's why we call it a semi-gradient). We can check that ∇ W ℓ 2 2 ( η w t -1 ( s t ) , [ T π t η w t -1 ] ( s t ) ) is an unbiased estimate of ∇ W ℓ 2 2 ,µ π ( η w t -1 , T π η w t -1 ) .

Now, let us compute the gradient term. By Eqn. (23), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p T π t η w t -1 ( s t ) = p Π K T π t η w t -1 ( s t ) = ( p k ([ T π t η w t -1 ] ( s t ) )) k =0 ∈ R K . Now, we turn to compute p T π η w ( s t ) . According to Eqn. (8),

<!-- formula-not-decoded -->

which has the same form as Eqn. (20). Following the proof of Proposition 3.2 in Appendix D.3, one can show that

<!-- formula-not-decoded -->

Hence, the update scheme is

<!-- formula-not-decoded -->

Note that our Linear-CTD (Eqn. (13)) is equivalent to

<!-- formula-not-decoded -->

in the PMF representation. Compared to Eqn. (27), the SSGD (Eqn. (26)) has an additional C ⊤ C , and the step size is 2 ι K α .

## D.7.2 Cumulative Distribution Function Representation

Consider the SSGD with the CDF representation

<!-- formula-not-decoded -->

where ∇ Θ stands for taking gradient w.r.t. Θ t -1 = θ t -1 C ⊤ ∈ R d × K in the first term η θ t -1 ( s t ) (the second term is regarding as a constant). One can check that ∇ Θ ℓ 2 2 ( η θ t -1 ( s t ) , [ T π t η θ t -1 ] ( s t ) ) is an unbiased estimate of ∇ Θ ℓ 2 2 ,µ π ( η θ t -1 , T π η θ t -1 ) .

Now, let us compute the gradient term. By Eqn. (19) and Eqn. (25), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, the update scheme is

<!-- formula-not-decoded -->

which has the same form as Linear-CTD (Eqn. (13)) with the step size 2 αι K .

## D.8 Linear-CTD is mean-preserving

We will show that our Linear-CTD is mean-preserving, which was first discovered by Lyle et al. [2019, Proposition 8]. In this section, we assume the first coordinate of the feature is a constant, i.e. , ϕ 1 ( s ) = 1 / √ d for any s ∈ S . As stated before, we will always assume this to ensure P ϕ ,K can be uniquely defined.

Proposition D.3. Let V θ := ( V θ ( s )) s ∈S be the value function corresponding to θ , i.e. , V θ ( s ) = E G ∼ η θ ( s ) [ G ] , then for any initialization of the Linear-TD parameter ψ 0 , there exists a (not unique) corresponding Linear-CTD parameter θ 0 such that V θ 0 = V ψ 0 , furthermore, for any t ≥ 1 and even number T ≥ 2 , it holds that

<!-- formula-not-decoded -->

Proof of Proposition D.3. Recall that the updating scheme of Linear-TD is given by

<!-- formula-not-decoded -->

And the updating scheme of Linear-CTD is given by

<!-- formula-not-decoded -->

Let V θ := ( V θ ( s )) s ∈S be the value function corresponding to θ , we have

<!-- formula-not-decoded -->

Hence, if we take ψ 0 , 1 = √ d 2(1 -γ ) , ψ 0 ,i = 0 for any i ∈ { 2 , . . . , d } , and θ 0 = 0 dK , it holds that

<!-- formula-not-decoded -->

We can also show that for any ψ 0 ∈ R d , there exists θ 0 ∈ R d × K such that V ψ 0 = V θ 0 . That is we need to find θ 0 such that for any s ∈ S ,

<!-- formula-not-decoded -->

We can take θ 0 satisfying the following equations to make the above equation hold

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Furthermore, for any t ≥ 1 , we have for any s ∈ S ,

<!-- formula-not-decoded -->

We need to check that, if V θ t -1 = V ψ t -1 , it holds that

<!-- formula-not-decoded -->

which is the direct corollary of the following fact

<!-- formula-not-decoded -->

by Lemma D.2. And we can obtain V ¯ θ T = V ¯ ψ T by using the facts that P sign ϕ ,K is an affine space, V ϕ is a linear space and taking expectation is a linear operator.

Lemma D.2. For any ν ∈ P sign , it holds that

<!-- formula-not-decoded -->

Proof. By Eqn. (8), Π K ν ∈ P sign K is uniquely identified with a vector p ν = ( p k ( ν )) K -1 k =0 ∈ R K , where

<!-- formula-not-decoded -->

Hence, for any x ∈ [0 , (1 -γ ) -1 ] , we define x lb := max { y ∈ { x 0 , . . . , x K } : x ≤ y } , then

<!-- formula-not-decoded -->

therefore,

<!-- formula-not-decoded -->

## E Omitted Results and Proofs in Section 4

## E.1 Proof of Lemma 5.1

Proof. By Lemma H.1 and Eqn. (22), we have

<!-- formula-not-decoded -->

We only need to show that

<!-- formula-not-decoded -->

or equivalently,

Recall

<!-- formula-not-decoded -->

then for any θ ∈ R dK with ∥ θ ∥ = 1 ,

<!-- formula-not-decoded -->

It suffices to show that

<!-- formula-not-decoded -->

For brevity, we abbreviate C ˜ G ( r ) C -1 as Y ( r ) = ( y ij ( r )) K i,j =1 ∈ R K × K . Thus, it suffices to show that

<!-- formula-not-decoded -->

For any vectors w = ( w (0) ⊤ , · · · , w ( K -1) ⊤ ) and v = ( v (0) ⊤ , · · · , v ( K -1) ⊤ ) in R dK , we define the corresponding matrices W = ( w (0) , · · · , w ( K -1)) and V = ( v (0) , · · · , v ( K -1)) in R d × K , then ∥ w ∥ = ∥ W ∥ F = √ tr ( W ⊤ W ) and ∥ v ∥ = ∥ V ∥ F = √ tr ( V ⊤ V ) . With these notations, we have

<!-- formula-not-decoded -->

it is easy to check that

<!-- formula-not-decoded -->

hence

<!-- formula-not-decoded -->

where we used ∥ Y ( r ) ∥ ≤ √ γ for any r ∈ [0 , 1] by Lemma H.3, and Cauchy-Schwarz inequality. To summarize, we have shown the desired result

(

-

1

I

K

⊗

Σ

ϕ

## E.2 Proof of Lemma 5.2

Proof. For simplicity, we omit t in the random variables, for example, we use A to denote a random matrix with the same distribution as A t . In addition, we omit the subscripts in the expectation, the involving random variables are s ∼ µ π , a ∼ π ( · | s ) , ( r, s ′ ) ∼ P ( · , · | s, a ) .

Bounding C A . By Lemma A.3,

<!-- formula-not-decoded -->

where we used Lemma H.3. Hence, C A ≤ 2(1 + √ γ ) .

2

)

¯

⊤

(

-

ϕ

1

)

¯

(

-

ϕ

1

2

)

- √

2

A

I

K

⊗

Σ

A

I

K

⊗

Σ

≽

(1

γ

)

I

dK

.

Bounding C e . By Eqn. (32),

<!-- formula-not-decoded -->

Hence

As for ∥ b ∥ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Proposition C.3 with η ∈ P sign K satisfying η (˜ s ) = ν for all ˜ s ∈ S , where ν = ( K +1) -1 ∑ K k =0 δ x k is the discrete uniform distribution, we can derive that, for any r ∈ [0 , 1] and s ′ ∈ S , it holds that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where we used the orthogonal decomposition (Proposition C.2) and an upper bound for ℓ 2 (( b r,γ ) # ( ν ) , ν ) (Lemma H.4).

In summary,

<!-- formula-not-decoded -->

Hence, C e ≤ √ 2(1 + γ ) ∥ θ ⋆ ∥ +3 √ K (1 -γ ) .

Bounding tr ( Σ e ) .

<!-- formula-not-decoded -->

By Eqn. (33),

<!-- formula-not-decoded -->

And by Lemma H.4,

To summarize,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.3 Proof of Lemma 5.3

Proof. For simplicity, we use the same abbreviations as in Appendix E.2. As in the proof of [Lemma 2 Samsonov et al., 2024b], we only need to show that, for any p ∈ N , α ∈ ( 0 , (1 - √ γ ) / (38 p ) ) , it holds that

<!-- formula-not-decoded -->

Let B := A + A ⊤ -α A ⊤ A which satisfies ( I dK -α A ) ⊤ ( I dK -α A ) = I dK -α B . To give an upper bound of E [( I dK -α B ) p ] , it suffices to show that

<!-- formula-not-decoded -->

if we take α ∈ ( 0 , (1 - √ γ ) / (2(1 + γ )) ) .

Given these results, we have, when α ∈ ( 0 , (1 - √ γ ) / (38 p ) ) , it holds that

<!-- formula-not-decoded -->

Lower Bound of E [ B ] . To show E [ B ] ≽ (1 - √ γ ) I K ⊗ Σ ϕ , we first show that E [ A + A ⊤ ] ≽ 2(1 - √ γ ) I K ⊗ Σ ϕ , which is equivalent to ( I K ⊗ Σ ϕ ) -1 2 E [ A + A ⊤ ] ( I K ⊗ Σ ϕ ) -1 2 ≽ 2(1 - √ γ ) , where

<!-- formula-not-decoded -->

Then, for any θ ∈ R dK with ∥ θ ∥ = 1 ,

<!-- formula-not-decoded -->

where we used the result Eqn. (29).

<!-- formula-not-decoded -->

Next, we give an upper bound for E [ A ⊤ A ] , we need to compute the following terms: by Lemma A.4,

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

And by Lemma A.4 and Lemma H.3,

<!-- formula-not-decoded -->

To summarize, by the basic inequality ( B 1 -B 2 ) ⊤ ( B 1 -B 2 ) ≼ 2 ( B ⊤ 1 B 1 + B ⊤ 2 B 2 ) , we have A ⊤ A ≼ 2 I K ⊗ ( ϕ ( s ) ϕ ( s ) ⊤ + γ ϕ ( s ′ ) ϕ ( s ′ ) ⊤ ) , (32)

and, after taking expectation,

Hence,

<!-- formula-not-decoded -->

because α ∈ ( 0 , (1 - √ γ ) / (2(1 + γ )) ) .

Now, we aim to give an upper bound for E [ B 2 ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining these together, we obtain

<!-- formula-not-decoded -->

if we take α ∈ ( 0 , (1 - √ γ ) / (2(1 + γ )) ) .

Upper Bound of E [ B p ] . Because B 2 is always PSD, we have the following upper bound

<!-- formula-not-decoded -->

We first give an almost-sure upper bound for ∥ B ∥ . By Lemma 5.2, ∥ A ∥ ≤ 1+ √ γ . And by Eqn. (32),

<!-- formula-not-decoded -->

where we used the fact that ( B 1 + B 2 ) 2 ≼ (1 + β ) B 2 1 +(1 + β -1 ) B 2 2 for any symmetric matrices B 1 , B 2 , since β B 2 1 + β -1 B 2 2 -B 1 B 2 -B 2 B 1 = ( √ β B 1 -√ β -1 B 2 ) 2 ≽ 0 , β ∈ (0 , 1) to be determined; and the fact that A 2 + ( A ⊤ ) 2 ≼ A ⊤ A + AA ⊤ since the square of the skew-symmetric matrix is negative semi-definite ( A -A ⊤ ) 2 ≼ 0 . By Eqn. (34) and Eqn. (33), we have

<!-- formula-not-decoded -->

thus, by α ∈ ( 0 , (1 - √ γ ) / (2(1 + γ )) ) , it holds that

<!-- formula-not-decoded -->

As for E [ AA ⊤ ] , by the basic inequality ( B 1 -B 2 ) ( B 1 -B 2 ) ⊤ ≼ 2 ( B 1 B ⊤ 1 + B 2 B ⊤ 2 ) , we have

<!-- formula-not-decoded -->

By Eqn. (31), we have

And by Lemma H.3,

<!-- formula-not-decoded -->

To summarize, we have and after taking expectation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By putting everything together (Eqn. (36), Eqn. (37), Eqn. (38)), we have

<!-- formula-not-decoded -->

where we take β = 1 - √ γ √ 8(1+ γ ) . Therefore, by Eqn. (35),

<!-- formula-not-decoded -->

## E.4 L p Convergence

Theorem E.1 ( L p Convergence) . For any K ≥ (1 -γ ) -1 , p &gt; 2 and α ∈ (0 , (1 - √ γ ) / [38( p + log T )]) , it holds that

<!-- formula-not-decoded -->

Proof. Combining Lemma 5.1, Lemma 5.2 and Lemma 5.3 with [Theorem 2 Samsonov et al., 2024b], we have

<!-- formula-not-decoded -->

I K ⊗ Σ ϕ

## E.5 Convergence Results for SSGD with the PMF Representation

In this section, we present the counterparts of Lemma 5.1, Lemma 5.2, Lemma 5.3 and Theorem 4.1 for stochastic semi-gradient descent (SSGD) with the probability mass function (PMF) representation. These results will additionally depend on K . The additional K -dependent terms arise because the condition number of C ⊤ C scales with K 2 (Lemma H.2). These terms are inevitable within our theoretical framework. The proofs of these results require only minor modifications to the original proofs, and we omit them for brevity.

In fact, in Appendix G, we validate some theoretical results through numerical experiments. To be concrete, we find that empirically, as K increases, to ensure convergence, the step size of the vanilla algorithm in [Bellemare et al., 2023, Section 9.6] indeed needs to decay at a rate of K -2 . In contrast, the step size of our Linear-CTD does not need to be adjusted when K increases. Moreover, we find that Linear-CTD empirically consistently outperforms the vanilla algorithm under different K .

Recall Eqn. (26), the updating scheme of the algorithm is

<!-- formula-not-decoded -->

which is equivalent to

<!-- formula-not-decoded -->

here we drop the additional 2 ι K in the step size for brevity. Letting Θ PMF ,t := W t C ⊤ be the CDF parameter, the algorithm becomes

<!-- formula-not-decoded -->

Here, we add the subscript PMF to the original notations to indicate the difference. Then, the algorithm corresponds to the following linear system for θ ∈ R dK

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Compared to our Linear-CTD (Eqn. (16)), this algorithm has an additional matrix CC ⊤ with the condition number of order K 2 (see Lemma H.2).

Now, we are ready to state the theoretical results for the algorithm.

Lemma E.1. For any θ ∈ R dK , it holds that

<!-- formula-not-decoded -->

Lemma E.1 achieves the same order of bound as prior results for Linear-CTD (Lemma 5.1), as the minimum eigenvalue of CC ⊤ remains Ω(1) (Lemma H.2). However, from the numerical experiments (Figure 6) in Appendix G.2, we observe that after substituting ¯ θ t into θ , as K grows, the RHS grows with K , while the LHS remains almost unchanged. This might be because when the matrix CC ⊤ acts on the relevant random vectors, the stretching coefficient ( i.e. , ∥ ∥ CC ⊤ x ∥ ∥ / ∥ x ∥ for some vector x ) is usually of order K 2 rather than a constant order. For example, consider the case where the matrix CC ⊤ acts on a random vector X that follows a uniform distribution over the surface of unit sphere ( ∥ X ∥ = 1 ). Since the k -th largest eigenvalue of the matrix CC ⊤ is of order k 2 , by Hanson-Wright inequality [Vershynin, 2018, Theorem 6.2.1], we have ∥ ∥ CC ⊤ X ∥ ∥ is of order K 2 with high probability.

Lemma E.2. It holds that

<!-- formula-not-decoded -->

Lemma E.2 introduces an additional factor of K 2 (or K 4 ) compared to previous results for Linear-CTD (Lemma 5.2) , since the maximum eigenvalue of CC ⊤ is of order K 2 .

Lemma E.3. For any p ≥ 2 , let a PMF ≃ (1 - √ γ ) λ min and α PMF p, ∞ ≃ (1 - √ γ ) / ( pK 2 ) ( α PMF p, ∞ p ≤ 1 / 2 ). Then for any α ∈ ( 0 , α PMF p, ∞ ) , u ∈ R dK and t ∈ N

<!-- formula-not-decoded -->

Table 2: Comparison between Linear-TD and Linear-CTD .

| Concepts                   | Linear-TD                                                                 | Linear-CTD                                                                                          |
|----------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Parametrization            | V ψ ( s ) = ϕ ( s ) ⊤ ψ                                                   | F k ( s ; θ ) = ϕ ( s ) ⊤ θ ( k )+ k +1 K +1                                                        |
| Bellman Operator           | ( T π V )( s ) = E [ r 0 + γV ( s 1 ) &#124; s 0 = s ]                    | ( T π η )( s ) = E [( b r 0 ,γ ) # η ( s 1 ) &#124; s 0 = s ]                                       |
| Projection Operator        | V ˜ ψ = Π π ϕ V , ˜ ψ = Σ - 1 ϕ E s ∼ µ π [ ϕ ( s ) V ( s )]              | η ˜ θ = Π π ϕ ,K η , ˜ Θ = Σ - 1 ϕ E s ∼ µ π [ ϕ ( s )( F η ( s ) - F ν ) ⊤ ]                       |
| Projected Bellman Equation | V ψ = Π π ϕ T π V ψ , [ See Eqn. (4) ]                                    | η θ = Π π ϕ ,K T π η θ , [ See Eqn. (12) ]                                                          |
| Update Rule                | ψ t ← ψ t - 1 - α ϕ ( s t ) [ ( ϕ ( s t ) - γ ϕ ( s t +1 )) ⊤ ψ t - 1 - r | [ See Eqn. (13) ]                                                                                   |
| A t                        | ϕ ( s t ) ϕ ( s t ) ⊤ - γ ϕ ( s t ) ϕ ( s t +1 ) ⊤                        | [ I K ⊗ ( ϕ ( s t ) ϕ ( s t ) ⊤ )] - [( C ˜ G - 1 ( r t ) C - 1 ) ⊗ ( ϕ ( s t ) ϕ ( s t +1 ) ⊤ )] √ |
| Key Quantity in A t        | γ                                                                         | C ˜ G - 1 ( r t ) C - 1 with spectral norm γ                                                        |
| b t                        | r t ϕ ( s t )                                                             | 1 K +1 [ C ( ∑ K j =0 g j ( r t ) - 1 K )] ⊗ ϕ ( s t )                                              |
| Key Quantity in b t        | r t ≤ 1                                                                   | K - 3 / 2 (1 - γ ) - 1 ∥ C ( ∑ K j =0 g j ( r t ) - 1 K ) ∥ ≤ 1                                     |
| Measure of Error           | ∥ V - V π ∥ µ π = ( E s ∼ µ π [( V ( s ) - V π ( s )) 2 ]) 1 / 2          | W 1 ,µ π ( η , η π ) = ( E s ∼ µ π [ W 2 1 ( η ( s ) , η π ( s ))]) 1 / 2                           |
| Approximation Error        | ∥ V ψ ⋆ - V π ∥ µ π ≤ (1 - γ 2 ) - 1 / 2 ∥ Π π ϕ V π - V π ∥ µ π          | [ See Proposition 3.3 ]                                                                             |
| Sample Complexity          | ˜ O ( ∥ ψ ⋆ ∥ 2 Σ ϕ +1 (1 - γ ) 2 λ min ( 1 ε 2 + 1 λ min ) )             | ˜ O ( ∥ θ ⋆ ∥ 2 V 1 +1 (1 - γ ) 2 λ min ( 1 ε 2 + 1 λ min ) )                                       |

As before, in this lemma, a PMF does not depend on K because the minimum eigenvalue of CC ⊤ is Ω(1) , and α PMF p, ∞ scales with K -2 because the maximum eigenvalue of CC ⊤ is of order K 2 .

Theorem E.2. For any K ≥ (1 -γ ) -1 and α ∈ (0 , α PMF p, ∞ ) , it holds that

<!-- formula-not-decoded -->

This theorem for the PMF version algorithm yields an upper bound that is K 3 times looser than Theorem 4.1 for our Linear-CTD . The appearance of the K 3 factor is due to the fact that the condition number of the redundant matrix CC ⊤ is of order K 2 . This factor is unavoidable within our theoretical analysis framework.

However, from the numerical experiments (Table 3 and Table 4) in Appendix G.2, we can only observe that our Linear-CTD consistently outperforms the PMF version algorithm under different values of K , but the performance gap does not increase significantly when K becomes larger as predicted by Theorem 4.1 and Theorem E.2. The reason for this might be, as discussed after Lemma E.1: in the experimental environment we have set, when the matrix CC ⊤ acts on the vectors it encountered, the stretching coefficient is usually of order K 2 rather than a constant order. See the numerical experiments (Figure 6) in Appendix G.2 for some evidence.

## F Comparison between Linear-TD and Linear-CTD

To further improve the readability of the paper, we compare various concepts and results between Linear-TD and Linear-CTD in Table 2.

## G Numerical Experiment

In this appendix, we validate the proposed Linear-CTD algorithm (Eqn. (13)) with numerical experiments, and show its advantage over the baseline algorithm, stochastic semi-gradient descent (SSGD) with the probability mass function (PMF) representation (Eqn. (39)).

To empirically evaluate our Linear-CTD algorithm, we consider a 3 -state MDP with γ = 0 . 75 . When the number of states is finite, we denote by Φ = ( ϕ ( s )) s ∈S ∈ R d ×S the feature matrix. Here, we set the feature matrix Φ to be a full-rank matrix in R 3 × 3 . The following experiments share zero initialization θ 0 = 0 with max iteration =500000 and batch size =25 .

All of the experiments are conducted on a server with 4 NVIDIA RTX 4090 GPUs and Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz.

## G.1 Empirical Convergence of Linear-CTD

We employ the Linear-CTD algorithm in the above environment and have the following convergence results in Figure 2. This figure shows the negative logarithm of 1 K ∥ ∥ ¯ θ t -θ ⋆ ∥ ∥ 2 I K ⊗ Σ ϕ = (1 -

Figure 2. Convergence results under varying K for our Linear-CTD algorithm with step size α = 0 . 01 . These curves exhibit similar trends, demonstrating our algorithm's robustness across different K values.

<!-- image -->

γ ) ℓ 2 2 ,µ π ( η ¯ θ t , η θ ⋆ ) along iterations. We observe that our Linear-CTD algorithm can converge for different values of K when we set the step size as α = 0 . 01 .

## G.2 Comparison with SSGD with the PMF Representation

First, we repeat the same experiment as in the previous section for the baseline algorithm, SSGD with the PMF representation. The experimental results in Figure 3 demonstrate that when the baseline algorithm uses a fixed step size α = 0 . 01 , it does not converge when K is large ( K ≥ 44 ). The results in Figure 2 and Figure 3 verify the advantage of our Linear-CTD over the baseline algorithm as mentioned in Remark 5: when K increases, the step size of the baseline algorithm needs to decay (Lemma E.3). In contrast, the step size of our Linear-CTD does not need to be adjusted when K increases (Lemma 5.3).

Next, we will verify that the maximum step size α PMF , ( K ) ∞ that ensures the convergence of the baseline algorithm scales with K -2 , as predicted in Lemma E.3. Then we will compare the convergence rate of the baseline algorithm with that of our Linear-CTD algorithm.

In Figure 4, we employ the baseline algorithm with different step sizes under fixed K = 150 , and we find that the baseline algorithm converges when the step size does not exceed 8 . 6e -4 , and it does not converge when the step size exceeds 8 . 7e -4 . This indicates that α PMF , (150) ∞ ∈ [8 . 6e -4 , 8 . 7e -4] in this environment, providing a good approximation of α PMF , (150) ∞ .

We repeat the above experiments under varying K , looking for a step size that can ensure convergence (a lower bound of α PMF , ( K ) ∞ ) and a step size that leads to divergence (an upper bound of α PMF , ( K ) ∞ ) such that the two step sizes are as close as possible and thereby we can get a good approximation of α PMF , ( K ) ∞ . The results are summarized in Table 3.

In Figure 5, we use the approximate values of α PMF , ( K ) ∞ provided in Table 3 to perform a quadratic function fitting of 1 /α PMF , ( K ) ∞ with respect to K . We find that α PMF , ( K ) ∞ indeed approximately scales with K -2 , which verifies our theoretical result (Lemma E.3).

To compare the statistical efficiency of our Linear-CTD algorithm and the baseline algorithm, in Table 3, we also report the number of iterations required for the error to reach below 2e -6 when the

Figure 3. Convergence results under varying K for the baseline algorithm, SSGD with the PMF representation with step size α = 0 . 01 . We remark that when K = 45 , the program reports errors of inf and nan. In contrast to results of Linear-CTD in Figure 2, the baseline algorithm no longer converges when K is large ( K ≥ 44 ).

<!-- image -->

Figure 4. Convergence results with different step sizes for the baseline algorithm, SSGD with the PMF representation under fixed K = 150 . We remark that when we take α = 8 . 8e -4 , the program reports errors of inf and nan. The baseline algorithm converges when the step size does not exceed 8 . 6e -4 , and it does not converge when the step size exceeds 8 . 7e -4 . Therefore, α PMF , (150) ∞ ∈ [8 . 6e -4 , 8 . 7e -4] in this environment.

<!-- image -->

step size satisfies α ≈ 0 . 2 α PMF , ( K ) ∞ . In addition, we present the parallel results of our Linear-CTD in Table 4. In Table 4, we find that the value of α ( K ) ∞ for our Linear-CTD algorithm is much larger than α PMF , ( K ) ∞ , and it does not decrease significantly with the growth of K . Moreover, by comparing the Iterations columns in Table 3 and Table 4, we find that the sample complexity of our Linear-CTD does not increase significantly with the growth of K , and Linear-CTD empirically consistently outperforms the baseline algorithm under different K .

However, the performance gap does not increase significantly as expected when K increases as predicted by Theorem 4.1 and Theorem E.2. The reason for this might be that, as discussed after Lemma E.1, in the experimental environment we have set, when the matrix CC ⊤ acts on the vectors

where

<!-- formula-not-decoded -->

In our theoretical analysis, we first give an upper bound of the RHS, and then apply Lemma E.1 bound the loss function L ( θ ) in the LHS with the RHS. However, since the minimum eigenvalue

Table 3. Lower and upper bounds of the maximum step size α PMF , ( K ) ∞ that ensures the convergence under varying K for the baseline algorithm, SSGD with the PMF representation. The bounds are determined using the same method as that in Figure 4. The Iterations column refers to the number of iterations required for the error to reach below 2e -6 when the step size satisfies α ≈ 0 . 2 α PMF , ( K ) ∞ .

|   K |   Lower Bound of α PMF , ( K ) ∞ |   Upper Bound of α PMF , ( K ) ∞ |   Iterations |
|-----|----------------------------------|----------------------------------|--------------|
|  30 |                         0.021    |                          0.022   |        37245 |
|  45 |                         0.009    |                          0.0095  |        39262 |
|  75 |                         0.0034   |                          0.0035  |        38286 |
| 105 |                         0.00175  |                          0.0018  |        38123 |
| 150 |                         0.00086  |                          0.00087 |        38556 |
| 225 |                         0.00038  |                          0.00039 |        38317 |
| 300 |                         0.00021  |                          0.00022 |        38999 |
| 375 |                         0.000135 |                          0.00014 |        38674 |
| 450 |                         9.5e-05  |                          9.8e-05 |        38506 |

Figure 5. The approximate values of of maximum step sizes 1 /α PMF , ( K ) ∞ under varying K . Here we take the average of the upper and lower bounds of α PMF , ( K ) ∞ provided in Table 3 as an approximation of α PMF , ( K ) ∞ and perform quadratic regression of 1 /α PMF , ( K ) ∞ on K . This fit achieves a mean squared error of 425 . 85 and R 2 of 0 . 99996 , which indicates that 1 /α PMF , ( K ) ∞ indeed grows quadratically with respect to K , aligning with our theoretical results (Lemma E.3).

<!-- image -->

it encountered, the stretching coefficient ( i.e. , ∥ ∥ CC ⊤ x ∥ ∥ / ∥ x ∥ for some vector x ) is usually of order K 2 rather than a constant order.

We verify this conjecture through the following experiment. We focus on the LHS and RHS of Eqn. (40) in Lemma E.1:

<!-- formula-not-decoded -->

Table 4. Lower and upper bounds of the maximum step size α ( K ) ∞ that ensures the convergence under varying K for our Linear-CTD . The bounds are determined using the same method as that in Figure 4. The Iterations column refers to the number of iterations required for the error to reach below 2e -6 the step size satisfies α ≈ 0 . 2 α ( K ) ∞ .

|   K |   Lower Bound of α ( K ) ∞ |   Upper Bound of α ( K ) ∞ |   Iterations |
|-----|----------------------------|----------------------------|--------------|
|  30 |                       1.65 |                       1.7  |        17908 |
|  45 |                       1.65 |                       1.7  |        17925 |
|  75 |                       1.65 |                       1.7  |        17942 |
| 105 |                       1.6  |                       1.65 |        18623 |
| 150 |                       1.5  |                       1.55 |        21947 |
| 225 |                       1.55 |                       1.6  |        20890 |
| 300 |                       1.55 |                       1.6  |        20890 |
| 375 |                       1.55 |                       1.65 |        20595 |
| 450 |                       1.5  |                       1.55 |        21947 |

of the matrix CC ⊤ in the RHS is only of a constant order, we are unable to have a term of 1 /K 2 in the RHS. Therefore, our conjecture can be verified by checking whether the bound provided in Lemma E.1 is tight in this environment, which is presented in Figure 6. The left sub-graph of Figure 6 corresponds to the LHS, and the right sub-graph corresponds to the RHS. We omit the constants that are independent of K . From Figure 6, we can find that the LHS remains almost unchanged under different K , but the RHS increases as K becomes larger. This indicates that the stretching coefficient of the matrix CC ⊤ that we frequently encounters during the iterative process grows with K rather than remaining a constant order. A similar analysis also holds for a PMF in Lemma E.3, and we omit it for brevity. These factors result in the performance gap between our Linear-CTD algorithm and the baseline algorithm not increasing significantly when K becomes larger, as predicted by Theorem 4.1 and Theorem E.2.

Figure 6. LHS and RHS of Eqn. (40) in Lemma E.1 under varying K . The left sub-graph corresponds to the LHS, and the right sub-graph corresponds to the RHS. We omit the constants that are independent of K . We can find that the LHS remains almost unchanged under different K , but the RHS increases as K becomes larger, indicating that the stretching coefficient of the matrix CC ⊤ that we frequently encounters during the iterative process grows with K rather than remaining a constant order.

<!-- image -->

## H Other Technical Lemmas

<!-- formula-not-decoded -->

Proof. By Cauchy-Schwarz inequality,

<!-- formula-not-decoded -->

Lemma H.2. Let C ∈ R K × K be the matrix defined in Eqn. (11) , it holds that the eigenvalues of C T C are 1 / (4 cos 2 ( kπ/ (2 K +1)) for k ∈ [ K ] , and thus

<!-- formula-not-decoded -->

Proof. One can check that

<!-- formula-not-decoded -->

Then, one can work with the the inverse of C ⊤ C and calculate its singular values by induction, which has similar forms to the analysis of Toeplitz's matrix. See Godsil [1985] for more details.

<!-- formula-not-decoded -->

Proof. One can check that

It is clear that

<!-- formula-not-decoded -->

By Lemma I.2 and an upper bound on the spectral norm (Riesz-Thorin interpolation theorem) [Serre, 2002, Theorem 7.3], we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

LemmaH.4. Suppose K ≥ (1 -γ ) -1 , ν = ( K +1) -1 ∑ K k =0 δ x k is the discrete uniform distribution, then for any r ∈ [0 , 1] , it holds that

<!-- formula-not-decoded -->

Proof. Let ˜ ν be the continuous uniform distribution on [ 0 , (1 -γ ) -1 + ι K ] , we consider the following decomposition

<!-- formula-not-decoded -->

By definition, we have

<!-- formula-not-decoded -->

By the contraction property, we have

<!-- formula-not-decoded -->

We only need to bound ℓ 2 (˜ ν, ( b r,γ ) # (˜ ν )) . We can find that ( b r,γ ) # (˜ ν ) is the continuous uniform distribution on [ r, r + γι K + γ (1 -γ ) -1 ] , and the upper bound is less than the upper bound of ν , namely, r + γι K + γ (1 -γ ) -1 ≤ (1 -γ ) -1 + γι K &lt; (1 -γ ) -1 + ι K . Hence

<!-- formula-not-decoded -->

To summarize, we have

<!-- formula-not-decoded -->

where we used the assumption K ≥ (1 -γ ) -1 .

## I Analysis of the Categorical Projected Bellman Matrix

Recall that ˜ G ( r ) = G ( r ) -1 ⊤ K ⊗ g K ( r ) . We extend the definition in Theorem 3.1 and let g j,k ( r ) = h (( r + γx j -x k ) /ι K ) + = h ( r/ι K + γj -k ) for j, k ∈ { 0 , 1 , · · · , K } where h ( x ) = (1 -| x | ) + .

Lemma I.1. For any r ∈ [0 , 1] and any k ∈ { 0 , 1 , · · · , K } , in g k ( r ) there is either only one nonzero entry or two adjacent nonzero entries.

Proof. It is clear that h ( x ) &gt; 0 ⇐⇒ -1 &lt; x &lt; 1 . Let k j ( r ) = min { k : g j,k ( r ) &gt; 0 } , then k j ( r ) = min { k : r/ι K + γj -k &lt; 1 } = min { k : 0 ≤ r/ι K + γj -k &lt; 1 } . The existence of k j ( r ) is due to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following results are immediate corollaries.

## Corollary I.1.

## Corollary I.2.

As a result,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma I.2. All entries in C ˜ G ( r ) C -1 are non-negative. ∥ ∥ ∥ C ˜ G ( r ) C -1 ∥ ∥ ∥ ∞ = γ and ∥ ∥ ∥ C ˜ G ( r ) C -1 ∥ ∥ ∥ 1 ≤ 1 .

Proof. By definition the entries of ˜ G ( r ) are

<!-- formula-not-decoded -->

Using the previous corollaries, through direct calculation we have that if k j +1 ( r ) = k j ( r ) ,

<!-- formula-not-decoded -->

And if k j +1 ( r ) = k j ( r ) + 1 ,

<!-- formula-not-decoded -->

As a result, all entries in C ˜ G ( r ) C -1 is non-negative. Moreover, the sum of each column and ∥ ∥ ∥ C ˜ G ( r ) C -1 ∥ ∥ ∥ ∞ is γ since

<!-- formula-not-decoded -->

Moreover, the row sum of C ˜ G ( r ) C -1 is

<!-- formula-not-decoded -->

Thus, it holds that ∥ ∥ ∥ C ˜ G ( r ) C -1 ∥ ∥ ∥ 1 ≤ 1 .
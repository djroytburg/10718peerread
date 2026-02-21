## Statistical Analysis of the Sinkhorn Iterations for Two-Sample Schrödinger Bridge Estimation

## Ibuki Maeda 1

maeda.ibuki554@mail.kyutech.jp

## Atsushi Nitanda

Rentian Yao 2

rentian2@math.ubc.ca

## 3 , 4 , 5

atsushi\_nitanda@a-star.edu.sg

1 Department of Artificial Intelligence, Kyushu Institute of Technology, Japan 2 Department of Mathematics, University of British Columbia, Canada

3 Institute of High Performance Computing, Agency for Science, Technology and Research (A ⋆ STAR), Singapore 4 Centre for Frontier AI Research, Agency for Science, Technology and Research (A ⋆ STAR), Singapore 5 College of Computing and Data Science, Nanyang Technological University, Singapore

## Abstract

The Schrödinger bridge problem seeks the optimal stochastic process that connects two given probability distributions with minimal energy modification. While the Sinkhorn algorithm is widely used to solve the static optimal transport problem, a recent work (Pooladian and Niles-Weed, 2024) proposed the Sinkhorn bridge , which estimates Schrödinger bridges by plugging optimal transport into the time-dependent drifts of SDEs, with statistical guarantees in the one-sample estimation setting where the true source distribution is fully accessible. In this work, to further justify this method, we study the statistical performance of intermediate Sinkhorn iterations in the two-sample estimation setting, where only finite samples from both source and target distributions are available. Specifically, we establish a statistical bound on the squared total variation error of Sinkhorn bridge iterations: O (1 /m +1 /n + r 4 k ) ( r ∈ (0 , 1)) , where m and n are the sample sizes from the source and target distributions, respectively, and k is the number of Sinkhorn iterations. This result provides a theoretical guarantee for the finite-sample performance of the Schrödinger bridge estimator and offers practical guidance for selecting sample sizes and the number of Sinkhorn iterations. Notably, our theoretical results apply to several representative methods such as [SF] 2 M, DSBM-IMF, BM2, and lightSB(-M) under specific settings, through the previously unnoticed connection between these estimators.

## 1 Introduction

In recent years, there has been increasing interest in Schrödinger bridge problems, with applications in mathematical biology (Chizat et al., 2022; Lavenant et al., 2024; Yao et al., 2025), Bayesian posterior sampling (Heng et al., 2024), generative modeling (De Bortoli et al., 2021; Wang et al., 2021), among others. The Schrödinger bridge problem aims to minimize the Kullback-Leibler (KL) divergence with respect to the Wiener measure, the distribution of Brownian motion over the path space, given two fixed marginal distributions µ, ν as source and target distributions.

The Schrödinger Bridge problem is also strongly connected to the entropic optimal transport (EOT) problem, which offers an efficient and regularized alternative to classical optimal transport, and is solvable via the Sinkhorn algorithm (Cuturi, 2013). The Sinkhorn algorithm operates in the dual space, optimizing the Schrödinger potentials between empirical distributions, and enjoys wellestablished convergence guarantees (Franklin and Lorenz, 1989; Peyré and Cuturi, 2018). Recently, Pooladian and Niles-Weed (2024) introduced the Sinkhorn bridge method, which plugs in the optimal Schrödinger potentials into the time-dependent drifts of stochastic differential equations to estimate

the Schrödinger bridge between the source and target distributions µ and ν . Their analysis established statistical bounds, adaptive to the intrinsic dimension of the target distribution ν , in the one-sample estimation setting, where the true source distribution µ is fully known.

## 1.1 Contributions

In this work, to better understand the algorithmic and statistical convergence rates of the Sinkhorn bridge method, we study the statistical performance of intermediate estimators obtained by Sinkhorn iterations in the two-sample estimation setting, where only empirical distributions µ m and ν n of m -and n -finite samples from µ and ν are available.

Statistical guarantee for Sinkhorn bridge iterations. First, we establish the following statistical bound (Theorem 1) for the Sinkhorn bridge estimator by leveraging analytical tools for the Schrödinger bridge in the one-sample setting (Pooladian and Niles-Weed, 2024) and for entropic optimal transport (Stromme, 2023b):

<!-- formula-not-decoded -->

where P ∗ ∞ , ∞ and P ∗ m,n denote the true Schrödinger bridge and its Sinkhorn bridge estimator based on the optimal EOT solution, and P ∗ , [0 ,τ ] ∞ , ∞ and P ∗ , [0 ,τ ] m,n are restrictions of them on the time-interval [0 , τ ] . The result improves upon the bound established in (Pooladian and Niles-Weed, 2024) by extending the analysis to the two-sample setting and sharpening the rate in n from n -1 / 2 to n -1 at the cost of a deterioration in the dependence on ε and R , where ε is the strength of entropic regularization and R is the radius of the data supports.

Next, we analyze the intermediate iterations of the Sinkhorn algorithm and prove a convergence rate (Theorem 2) for the corresponding path measures:

<!-- formula-not-decoded -->

where P k, [0 ,τ ] m,n denotes the Sinkhorn bridge estimator at the k -th Sinkhorn iteration for µ m and ν n . Together, these results imply the convergence rate E [ TV 2 ( P ∗ , [0 ,τ ] ∞ , ∞ , P k, [0 ,τ ] m,n )] = O (1 /m + 1 /n + r 4 k ) ( r ∈ (0 , 1)) regarding the number of samples m and n , and the number of Sinkhorn iterations k .

Relationship with the other Schrödinger Bridge estimators. Remarkably, our theoretical results on the Sinkhorn bridge can be directly applied to the representative Schrödinger Bridge solvers: [SF] 2 M(Tong et al., 2023), DSBM-IMF (Shi et al., 2023; Peluchetti, 2023), BM2 (Peluchetti, 2024), and lightSB(-M) (Korotin et al., 2023; Gushchin et al., 2024). First, the optimal estimator produced by these methods coincide with the Sinkhorn bridge estimator (Proposition 1). Moreover, when DSBM-IMF and BM2 are initialized with the reference process, the estimator after k iterations matches the Sinkhorn bridge obtained after k iterations of the Sinkhorn algorithm (Proposition 2). Furthermore, Algorithm 1, which learns the Sinkhorn bridge drift via a neural network, is a special case of the procedure proposed in [SF] 2 M. Consequently, our generalization-error analysis applies to all of these methods, endowing the algorithms with theoretical guarantees.

## 1.2 Related Work

Schrödinger bridge problem. The connection between entropic optimal transport and the Schrödinger bridge (SB) problem has been extensively studied; for a comprehensive overview, see the survey by Léonard (2013). More recently, there has been growing interest in computational approaches to the SB problem, particularly those leveraging deep learning techniques (De Bortoli et al., 2023; Shi et al., 2022; Bunne et al., 2023; Tong et al., 2023). In parallel, several studies have explored classical statistical methods for estimating the SB (Bernton et al., 2019; Pavon et al., 2021; Vargas et al., 2021). For example, Bernton et al. (2019) proposed a sampling framework based on trajectory refinement via approximate dynamic programming. Pavon et al. (2021) and Vargas et al. (2021) introduced methods that directly estimate intermediate densities using maximum likelihood principles. Specifically, Pavon et al. (2021) presented a scheme that explicitly models the target

density and updates weights accordingly, while Vargas et al. (2021) estimated forward and backward drifts using Gaussian processes, optimized via a likelihood-based objective. Assuming full access to the source distribution, the statistical performance of the Schrödinger bridge estimator was analyzed in the one-sample setting by Pooladian and Niles-Weed (2024). Other estimators, such as those based on neural approximations or efficient algorithmic variants, have been evaluated in recent works (Korotin et al., 2023; Stromme, 2023a).

Entropic optimal transport. Entropic Optimal Transport (EOT) introduces an entropy term to the standard Optimal Transport (OT) problem (Cuturi, 2013), primarily proposed to improve computational efficiency and smooth the transport plan. This regularization makes the OT problem strictly convex, enabling the application of the Sinkhorn algorithm, an efficient iterative method for the dual problem (Cuturi, 2013). The introduction of the Sinkhorn algorithm has allowed the OT problem, previously computationally very expensive, to obtain approximate solutions in realistic time even for large-scale datasets, leading to a dramatic expansion of its applications in the field of machine learning. For example, there are diverse applications such as measuring distances between distributions in generative modeling (Genevay et al., 2018), and shape matching or color transfer in computer graphics and computer vision (Solomon et al., 2015). The theoretical aspects of EOT have also been deeply studied. Analyses regarding the convergence rate of the Sinkhorn algorithm (Franklin and Lorenz, 1989) and the statistical properties (such as sample complexity) when estimating EOT from finite samples (Genevay et al., 2019) have been established. Furthermore, EOT is known to have a close relationship with the Schrödinger Bridge (SB) problem (Léonard, 2013). EOT in a static setting is equivalent to the problem of matching the marginal distributions of the SB problem, which deals with dynamic stochastic processes, and both share common mathematical structures in duality and optimization algorithms. This connection suggests that theoretical and computational advances in one field can contribute to the other.

## 1.3 Notations

For a metric space X , let P ( X ) be the space of probability distributions over X , and P 2 ( X ) be the subset of P ( X ) with finite second-order moment. Let R d be the d -dimensional Euclidean space and B ( a, R ) = { x | ∥ x -a ∥ 2 ≤ R } ⊂ R d be the Euclidean ball of radius R &gt; 0 centered at a ∈ R d . For real numbers a and b , we set a ∨ b = max( a, b ) and a ∧ b = min( a, b ) . We write a ≲ b to denote that there exists a constant C &gt; 0 , independent to a and b , such that a ≤ C b . We denote by δ x the Dirac measure concentrated at the point x ∈ R d . We denote by ✶ d ∈ R d the d -dimensional vector whose components are all equal to one. For a measure µ and a function f , we write µ ( f ) = ∫ f ( x ) µ ( dx ) . Let TV( µ, ν ) be the total variation distance between two probability distributions µ and ν . For ε &gt; 0 and any real number or real-valued function h , write exp ε ( h ) := exp( h/ε ) . We denote the d -dimensional probability simplex by ∆ d := { α ∈ R d ≥ 0 | ∑ d i =1 α i = 1 } .

## 2 Preliminaries

## 2.1 Entropic Optimal Transport

Let µ, ν ∈ P 2 ( R d ) and ε &gt; 0 be fixed. The entropic optimal transport (EOT) cost between µ and ν is

<!-- formula-not-decoded -->

with the cost function c ( x, y ) := 1 2 ∥ x -y ∥ 2 2 . Here, Π( µ, ν ) ⊂ P 2 ( R d × R d ) denotes the set of couplings with marginals µ and ν , and

<!-- formula-not-decoded -->

is the Kullback-Leibler divergence between π and µ ⊗ ν . If π is not absolutely continuous with respect to µ ⊗ ν , we set H ( π ∥ µ ⊗ ν ) = + ∞ .

The optimization problem in Eq. (3) is strictly convex with respect to π , and thus it admits a unique solution π ∗ ∞ , ∞ ∈ Π( µ, ν ) . Furthermore, the problem has a dual formulation given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The dual problem (4) admits the Schrödinger potentials f ∗ ∞ , ∞ and g ∗ ∞ , ∞ as optimal solutions, and the first-order optimality condition implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is known as Schrödinger system. Note that the solution is constant-shift invariant, i.e. for any a ∈ R , ( f ∗ ∞ , ∞ + a, g ∗ ∞ , ∞ -a ) is also a solution to the Schrödinger system (5). It can be shown that with a constraint ν ( g ∗ ∞ , ∞ ) = 0 , the solution to (5) is unique. The optimal solution π ∗ ∞ , ∞ of the primal problem (3) can be expressed using the dual solution via

<!-- formula-not-decoded -->

We refer to (Nutz, 2021) for more details of entropic optimal transport problems.

## 2.2 Schrödinger Bridge

Let Ω = C ([0 , 1]; R d ) be the path space consisting of all continuous maps from the time interval [0 , 1] to R d . Let W ε be the law of a reversible Brownian motion on R d with volatility ε . The Schrödinger Bridge problem is then defined through the following entropy minimization problem:

<!-- formula-not-decoded -->

where P t ∈ P ( R d ) is the marginal distribution at time t ∈ [0 , 1] of the path-space distribution P , i.e. X t ∼ P t if X is a stochastic process with distribution P . Though W ε is an unbounded positive measure over Ω , the above optimization problem is still well defined as illustrated by Léonard (2013).

The Schrödinger Bridge problem is closely connected with the EOT problem. Precisely, the optimal solution P ∗ ∞ , ∞ ∈ P (Ω) of Eq. (7) can be derived by first solving the EOT problem (3) and then connect the optimal coupling with the Brownian bridge W ε | x 0 ,x 1 = W ε ( · | X 0 = x 0 , X 1 = x 1 ) , i.e.,

<!-- formula-not-decoded -->

The optimal solution P ∗ ∞ , ∞ can also be characterized as the path measure of the SDE defined by

<!-- formula-not-decoded -->

where B t is the standard Brownian motion, and the drift coefficient is given by

<!-- formula-not-decoded -->

with γ ∗ ,t ∞ , ∞ ( y, z ) = exp ε ( g ∗ ∞ , ∞ ( y ) -1 2(1 -t ) ∥ y -z ∥ 2 2 ) . We refer to Chen et al. (2021); Léonard (2012, 2013) for more details of the Schrödinger Bridge.

## 2.3 Sinkhorn Algorithm

Given the empirical distributions µ m = 1 m ∑ m i =1 δ X i and ν n = 1 n ∑ n j =1 δ Y j , we consider the empirical EOT problem OT ε ( µ m , ν n ) . Let f ∗ m,n and g ∗ m,n be the optimal Schrödinger potentials. The Sinkhorn algorithm is an iterative method for approximating these optimal potentials by alternately updating the dual potentials f ( k ) m,n and g ( k ) m,n as follows.

<!-- formula-not-decoded -->

We note that f ( k ) m,n , g ( k ) m,n can be interpreted as m - and n -dimensional vectors, respectively. For efficient computation, it is common to rewrite these updates in an alternative form. Let the kernel matrix K ∈ R m × n be defined as K ij = exp ε ( -1 2 ∥ X i -Y j ∥ 2 2 ) , and denote the vectors u ( k ) ∈ R m and v ( k ) ∈ R n by u ( k ) i = exp ε ( f ( k ) ( X i )) and v ( k ) ( Y j ) = exp ε ( g ( k ) ( Y j )) . Then the update rule (10) becomes

<!-- formula-not-decoded -->

where ⊘ denotes element-wise division 1 . This reformulation greatly improves computational efficiency and enables large-scale applications.

Let d Hilb be the Hilbert (pseudo-)metric on R d + defined by

<!-- formula-not-decoded -->

It is obvious that d Hilb ( x, y ) = 0 if and only if there exists λ ∈ R + such that x = λy . Let the initial values be ( u (0) , v (0) ) = ( ✶ m , ✶ n ) . Denote by ( u ( k ) , v ( k ) ) the solution obtained after k iterations of the Sinkhorn algorithm (10), and by ( u ∗ , v ∗ ) = (exp( f ∗ m,n /ε ) , exp( g ∗ m,n /ε )) the optimal solution. Then, it is shown that (Franklin and Lorenz, 1989; Peyré and Cuturi, 2018),

<!-- formula-not-decoded -->

Here, λ ( K ) &lt; 1 is defined as λ ( K ) := √ γ ( K ) -1 √ γ ( K )+1 where γ ( K ) := max i,j,k,l K ik K jl K jk K il . This result implies the exponential convergence of the Sinkhorn iterates u ( k ) and v ( k ) .

## 2.4 Convergence Rate of One-Sample Estimation

Pooladian and Niles-Weed (2024) analyzed the one-sample estimation task for the Schrödinger Bridge. In this setting, full access to the source distribution µ is assumed, while the target distribution ν is observed only through the empirical distribution ν n = 1 n ∑ n j =1 δ Y j based on i.i.d. samples Y 1 , . . . , Y n . Let ( f ∗ ∞ ,n , g ∗ ∞ ,n ) denote the optimal dual potentials solving OT ε ( µ, ν n ) , and let P ∗ ∞ ,n ∈ P (Ω) be the corresponding Schrödinger Bridge. Under Assumptions 1, 2 and 3 stated below, Pooladian and Niles-Weed (2024, Theorem 4.1) proved that there exists a constant C ν , depending only on ν and geometric constants of its support supp( ν ) , such that for any τ ∈ [0 , 1) the following holds:

<!-- formula-not-decoded -->

where d ν is the intrinsic dimensionality of supp( ν ) .

## 3 Main Results

In this section, we provide statistical and algorithmic convergence rates of Sinkhorn bridge (Pooladian and Niles-Weed, 2024) in two-sample estimation setting.

1 The standard Sinkhorn algorithm updates as u ( k +1) = 1 m ⊘ ( mKv ( k ) ) , v ( k +1) = 1 n ⊘ ( nK T u ( k +1) ) . This update is derived from the formulation of the primal problem 3 that incorporates the regularization term H ( π ) , and there is no substantive difference.

## 3.1 Sinkhorn Bridge in Two-sample Case

Given empirical distributions µ m and ν n , the Sinkhorn bridge constructs the estimator P k m,n ∈ P (Ω) for the Schrödinger bridge between true distributions µ and ν as follows. First, the Sinkhorn algorithm runs for k iterations to solve OT ε ( µ m , ν n ) , yielding the dual potentials f ( k ) m,n and g ( k ) m,n . Then, the estimator P k m,n is defined as the path measure of the following SDE starting from µ :

<!-- formula-not-decoded -->

where the drift function b ( k ) m,n is

<!-- formula-not-decoded -->

## 3.2 Convergence Rates of Sinkhorn Bridge

First, we analyze the statistical convergence rate of the path measure P ∗ m,n ∈ P (Ω) using the optimal Schrödinger potentials f ∗ m,n and g ∗ m,n . Specifically, P ∗ m,n is defined analogously to P k m,n by replacing the potential g ( k ) m,n with its optimal counterpart g ∗ m,n . For details of the definition, see Appendix A.1. The analysis will be carried out under the following assumptions.

Assumption 1. The potentials g ∗ ∞ , ∞ , g ∗ ∞ ,n , g ∗ m,n satisfy

<!-- formula-not-decoded -->

Assumption 2. The supports of µ and ν are subsets of a ball with radius R centered at the origin.

Assumption 3. The measure ν is supported on a compact, smooth, connected Riemannian manifold ( N,h ) without boundary, embedded in R d and equipped with the submanifold geometry induced by its inclusion into R d ; furthermore, dim N = d ν ≥ 3 . Moreover, ν admits a Lipschitz continuous and strictly positive density with respect to the Riemannian volume measure on N .

The manifold hypothesis suggests that data distributions often lie on lower-dimensional manifolds, and statistical learning methods can exploit this structure. In the EOT literature, Stromme (2023b) established the Minimum Intrinsic Dimension (MID) scaling, where the minimum of the intrinsic dimensions governs the convergence rate. For the Sinkhorn bridge under one-sample setting, Pooladian and Niles-Weed (2024) demonstrated that the convergence rate is independent of the ambient dimension d and depends only on the intrinsic dimension of the target distribution ν . Following these works, we also impose data-manifold assumptions solely on the target distribution ν .

The following theorem provides the squared total variation error between P ∗ ∞ , ∞ and P ∗ m,n .

Theorem 1 (Statistical convergence rate) . Suppose Assumptions 1, 2 and 3 hold. Let P ∗ ∞ , ∞ be the path measure of the true Schrödinger bridge for marginals µ, ν . Then, it follows that for any τ ∈ [0 , 1) ,

<!-- formula-not-decoded -->

where P ∗ , [0 ,τ ] ∞ , ∞ and P ∗ , [0 ,τ ] m,n are restrictions of P ∗ ∞ , ∞ and P ∗ m,n on the time-interval [0 , τ ] .

Remark 1. In this section and related proofs, we suppress non-dominant terms such as polynomial factors of d ν , geometric characteristics of supp( ν ) , and exponential terms of order O ( d ν ) in uniform constants, in the notation ≲ .

We make the following observations: (1) Regarding the data dimensions, the intrinsic dimension d ν governs the growth of the convergence rate, as in Stromme (2023b); Pooladian and Niles-Weed (2024). Although our bound, unlike theirs, involves the ambient dimension d , the dependence is merely linear. Moreover, we note that the term d arises from the covering number of the data space

and can also be replaced by the intrinsic dimension of the source distribution µ by additionally imposing a manifold assumption on µ . (2) The degeneration of the estimated SDE toward a specific sample point Y j arises as τ → 1 because the estimated drift b ( k ) m,n points toward a particular sample Y j near t = 1 , causing the deviation from the target distribution ν at the terminal time t = 1 . Therefore, it is common practice to restrict the time interval to [0 , τ ] ( τ &lt; 1) to ensure generalization. (3) Our result extends and sharpens the result of Pooladian and Niles-Weed (2024) to the two-sample setting, improving the convergence rate in n from n -1 / 2 to n -1 at the cost of a slight deterioration regarding, ε , R , and d .

In the next theorem, we give the algorithmic convergence rate of P k m,n → P ∗ m,n attained by running the Sinkhorn iterations k →∞ .

Theorem 2 (Algorithmic convergence rate) . Under Assumptions 1 and 2, we get for any τ ∈ [0 , 1) ,

<!-- formula-not-decoded -->

where P k, [0 ,τ ] m,n are restrictions of P k m,n on the time-interval [0 , τ ] .

These results immediately imply the convergence rate E [ TV 2 ( P k, [0 ,τ ] m,n , P ∗ , [0 ,τ ] ∞ , ∞ )] = O (1 /m +1 /n + r 4 k ) for some r ∈ (0 , 1) regarding the sample size m and n , and the number of Sinkhorn iterations k .

## 3.3 Drift Estimation using Neural Networks in the Schrödinger Bridge Problem

The drift estimator (13) of the Sinkhorn bridge between µ m and ν n is defined as an empirical average regarding ν n . As a result, simulating the SDE (12) using a time discretization scheme (e.g., Euler-Maruyama approximation) incurs an O ( n ) -computational cost at every discretization step. To improve computational efficiency, we consider a neural network-based drift approximation.

Recall that W ε | x 0 ,x 1 is the Brownian bridge connecting x 0 and x 1 , so its marginal distribution at time t ∈ [0 , 1] is a Gaussian distribution W ε t | x 0 ,x 1 = N ( (1 -t ) x 0 + tx 1 , εt (1 -t ) I d ) . For dual potentials f ( k ) m,n ∈ L 1 ( µ m ) and g ( k ) m,n ∈ L 1 ( ν n ) obtained by the Sinkhorn algorithm, we define the joint distribution π ( k ) m,n analogously to Eq. (6) by replacing the optimal potentials and µ, ν with f ( k ) m,n , g ( k ) m,n , and µ m , ν n . Using π ( k ) m,n , we define the mixture of Brownian bridge (a.k.a. reciprocal process ): Π ( k ) = ∫ W ε | x 0 ,x 1 d π ( k ) m,n ( x 0 , x 1 ) ∈ P (Ω) . The drift function b ( k ) m,n in Eq. (13) is known to have the following expression ( Markovian projection (Shi et al., 2023)) (see Appendix B.2):

<!-- formula-not-decoded -->

Therefore, we see that the function b ( k ) m,n minimizes the following functional L :

<!-- formula-not-decoded -->

where Π ( k ) t 1 is the marginal distribution of Π ( k ) on time points t and 1 . This suggests training a neural network b θ with parameter θ to minimize the above functional. We note that samples from Π ( k ) t 1 can be obtained by sampling ( x 0 , x 1 ) ∼ π ( k ) m,n , t ∼ U [0 , τ ] , and x t ∼ W ε t | x 0 ,x 1 and discarding x 0 . Therefore, with this sampling scheme, SGD can be efficiently applied to minimize L ( b θ ) . The detail of the procedure is described in Algorithm 1. This procedure yields a neural network approximation of the drift function defined by the Sinkhorn bridge.

## 3.4 Other Estimators of Schrödinger Bridge

Our theoretical results on Sinkhorn bridge (Pooladian and Niles-Weed, 2024) can be directly applied to the representative Schrödinger Bridge solvers such as [SF] 2 M (Tong et al., 2023), lightSB(M) (Korotin et al., 2023; Gushchin et al., 2024), DSBM-IMF (Shi et al., 2023; Peluchetti, 2023), and BM2 (Peluchetti, 2024). First, we show that the optimal estimator produced by each of these methods

## Algorithm 1 Drift Approximation via Neural Network

input Joint distribution π m,n defined by potentials f ∈ L 1 ( µ m ) , g ∈ L 1 ( ν n ) ; neural network b θ output Trained neural network b θ that approximates the drift function repeat

Sample batch of pairs { x i 0 , x i 1 } N i =1 ∼ π m,n

Sample batch t i ∼ U [0 , τ ]

0 1 t t i | x 0 ,x 1

For each triplet ( x i , x i , t i ) , sample x i ∼ W ε i i (Brownian bridge)

Compute the loss:

<!-- formula-not-decoded -->

Update θ using ∇ L ( θ ) until converged;

coincides with P ∗ m,n attained by Sinkhorn bridge with k → ∞ (Proposition 1). Moreover, when DSBM-IMF and BM2 are initialized based on the reference process, the estimator after k iterations matches P k m,n obtained after k iterations of the Sinkhorn bridge (Proposition 2). Furthermore, it follows from the proof of Proposition 1 that Algorithm 1, which learns the Sinkhorn bridge drift via a neural network, is a special case of the algorithm proposed in [SF] 2 M. The proofs of these results are postponed to the Appendix. Consequently, our statistical and algorithmic convergence analysis applies to these methods, endowing the theoretical guarantees.

Proposition 1. The optimal estimators produced by [SF] 2 M, lightSB(-M), DSBM-IMF and BM2 coincide with P ∗ m,n attained by Sinkhorn bridge with k →∞ .

Proposition 2. With initialization based on the reference process, both DSBM-IMF and BM2 produce the same estimator as P k m,n of the Sinkhorn bridge for all iterations k .

## 4 Experiments

In this section, we verify our theoretical findings through numerical experiments. In Section 4.1, we verify the dependence of the statistical error on the sample size presented in Theorem 1, and the algorithmic convergence rate presented in Theorem 2. Next, in Section 4.2, we evaluate the neural network approximation of drift coefficient as illustrated in Section 3.3. All experimental implementations are conducted with PyTorch 2.6.0 (Paszke et al., 2019).

## 4.1 Experimental Verification of Theorems 1 and 2

We use 3 -dimensional normal distributions µ = N (0 , I 3 ) and ν = N (0 , B ) as source and target distributions, where B ∈ R 3 × 3 is a random positive definite matrix. Under this setting, Bunne et al. (2023, Eq. (25)-(29)) provided explicit expressions for the drift coefficient b ∗ ∞ , ∞ and the marginal distribution P ∗ ,t ∞ , ∞ . We use these expressions to evaluate the errors of the drift b ∗ m,n and the corresponding Schrödinger Bridge estimators P ∗ m,n in the finite-sample settings. These estimators are approximated by running the Sinkhorn bridge with a sufficiently large number of iterations ( k →∞ ) .

Statistical convergence. To verify that the estimator b ∗ m,n converges to the true drift b ∗ ∞ , ∞ when the sample sizes m and n increase, we draw m samples from µ and n samples from ν , respectively, and compute the estimator b ∗ m,n ( · , t ) for an arbitrary time t ∈ [0 , 1) . Subsequently, we evaluate the mean squared error (MSE) against the true drift b ∗ ∞ , ∞ ( · , t ) .

<!-- formula-not-decoded -->

To compute this, the norm ∥ · ∥ L 2 ( P ∗ ,t ∞ , ∞ ) is approximated using Monte Carlo with 10,000 samples drawn from P ∗ ,t ∞ , ∞ . The expectation over the samples is computed by averaging over 10 independent sampling trials. With a fixed parameter ε = 0 . 1 , we generated heatmaps for several t ∈ [0 , 1) while varying the sample sizes m,n used in the estimator definition (Figure 1).

Figure 1: Heatmaps illustrating MSEsample ( m,n,t ) as a function of sample sizes m and n for various time points t . The error decreases roughly proportionally to ( m -1 + n -1 ) .

<!-- image -->

As evident from the figure, the mean squared error deteriorates as t → 1 , but the overall convergence rate remains unchanged. For all t shown in the figure, the convergence rate is observed to be approximately proportional to ( m -1 + n -1 ) . This aligns with the result predicted by Theorem 1.

Algorithmic convergence. To verify that the estimator P ( k ) , [0 ,τ ] m,n exponentially approaches P ∗ , [0 ,τ ] m,n as the iteration count k increases, we obtain m and n independent samples from distributions µ and ν , respectively. We then consider the integral of the difference between estimators b ( k ) m,n ( · , t ) and b ∗ m,n ( · , t ) over the interval [0 , τ ] . Specifically, we evaluate the mean squared error integrated over t ∈ [0 , τ ] :

<!-- formula-not-decoded -->

Time integrals over the interval [0 , τ ] are approximated via Monte Carlo integration by uniformly sampling 1 , 000 time-points. The norm ∥ · ∥ L 2 ( P ∗ ,t m,n ) is also estimated via Monte Carlo integration, using 1 , 000 samples drawn from P ∗ ,t m,n . We set m = n = 1 , 000 and compute the expectation by averaging over 10 independent samplings. With the regularization parameter fixed at ε = 0 . 005 , we generate graphs of the integral values over [0 , τ ] for varying k and multiple τ values (Figure 2). For all τ values shown in the figure, the convergence rate exhibits exponential decay with respect to k . This observation corroborates the theoretical prediction in Theorem 2.

<!-- image -->

Moreover, under the experimental setup of Section 4.2, we illustrate in Figure 3 the evolution of the drift when the number of Sinkhorn iterations is set to 1, 5, and 10. In the figure, it is observed that as the number of iterations k

k

Figure 2: Integrated mean squared error as a function of Sinkhorn iterations k for multiple integration intervals [0 , τ ] .

increases, the drift b ( k ) m,n rapidly converges to the optimal drift b ∗ m,n .

## 4.2 Experimental Verification of Neural Network-Based Drift Estimation

Next, we evaluate the effectiveness of the drift approximation by a neural network using Algorithm 1. In this experiment, we set ε = 0 . 1 , defined µ as the eight-Gaussians distribution and ν as the moons distribution, and drew 1 , 000 independent samples from each. Using these samples, the Sinkhorn algorithm is employed to approximate the optimal EOT coupling π ∗ m,n between µ m and ν n , and, via Algorithm 1, a neural network approximation b θ of the drift b ∗ m,n is obtained. We employ an 4 -layer neural network with 512 -512 -512 hidden neurons for b θ , and train it using the AdamW optimizer with a learning rate of 1 × 10 -3 , weight decay of 1 × 10 -5 , and a mini-batch size of 4 , 096 . Finally, starting from each sample of µ m , we simulate trajectories using either b ( k ) m,n or the neural network drift b θ , and present the results in Figure 3. Trajectory simulations are performed using the Euler-Maruyama approximation with 1 , 000 discretization steps.

Figure 3: From left to right, the simulation results of the Schrödinger bridge using the estimated drifts b (1) m,n , b (5) m,n , and b (10) m,n obtained by terminating the Sinkhorn iteration after 1, 5, and 10 iterations, respectively, the optimal drift b ∗ m,n , and the neural network-approximated drift b θ .

<!-- image -->

## 5 Conclusion and Discussion

In this study, we provide a comprehensive analysis of statistical guarantees for the Schrödinger Bridge problem. Specifically, we establish theoretical guarantees in two key settings: (i) the two-sample estimation task and (ii) intermediate estimators during the learning process.

Our main contributions are as follows. First, we establish a statistical convergence analysis in the two-sample estimation setting for the Schrödinger bridge estimator with k →∞ , which demonstrates a statistical convergence rate of O ( 1 n + 1 m ) . Second, we establish new convergence guarantees for the dual potentials obtained during intermediate iterations of the Sinkhorn algorithm, proving exponential convergence in the finite-sample setting. These results allow for a clear estimation of the number of samples m,n and Sinkhorn iterations k required to achieve a desired precision. Experimental results align with our theoretical analysis, confirming that the error decreases at a rate of approximately ( m -1 + n -1 ) with respect to the sample size, and that the error decreases exponentially with respect to the number of Sinkhorn iterations k . These findings strengthen the statistical guarantees for existing methods proposed in works such as Korotin et al. (2023); Peluchetti (2023); Gushchin et al. (2024); Shi et al. (2023); Peluchetti (2024) through the connections among these methods. The theoretical framework of this study deepens the understanding of generalization error in the Schrödinger Bridge problem and offers new insights into the convergence properties and stability of the Sinkhorn algorithm during intermediate iterations.

Future research directions include extending the analysis to more complex distributional settings and more general reference processes, and removing linear dependence on the ambient dimension d . Another interesting research direction is to explore alternative optimization methods for EOT problems and analyze their performance using our theoretical framework beyond the Sinkhorn algorithm.

## Acknowledgment

This research is supported by the National Research Foundation, Singapore, Infocomm Media Development Authority under its Trust Tech Funding Initiative, and the Ministry of Digital Development and Information under the AI Visiting Professorship Programme (award number AIVP-2024-004). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore, Infocomm Media Development Authority, and the Ministry of Digital Development and Information.

## References

Bernton, E., Heng, J., Doucet, A., and Jacob, P. E. (2019). Schrödinger Bridge Samplers.

- Bunne, C., Hsieh, Y.-P., Cuturi, M., and Krause, A. (2023). The Schrödinger Bridge between Gaussian Measures has a Closed Form.
- Chen, Y., Georgiou, T. T., and Pavon, M. (2021). Stochastic control liaisons: Richard sinkhorn meets gaspard monge on a schrodinger bridge. Siam Review , 63(2):249-313.

Chen, Y., Li, P., Liu, D., and Wang, H. (2023). On talagrand's functional and generic chaining.

- Chizat, L., Zhang, S., Heitz, M., and Schiebinger, G. (2022). Trajectory inference via mean-field langevin in path space. Advances in Neural Information Processing Systems , 35:16731-16742.
- Cohen, S. N. and Fausti, E. (2024). Hyperbolic contractivity and the hilbert metric on probability measures.
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems , 26.
- De Bortoli, V., Thornton, J., Heng, J., and Doucet, A. (2021). Diffusion schrödinger bridge with applications to score-based generative modeling. Advances in Neural Information Processing Systems , 34:17695-17709.
- De Bortoli, V., Thornton, J., Heng, J., and Doucet, A. (2023). Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling.
- Fortet, R. (1940). Résolution d'un système d'équations de m. schrödinger. Journal de mathématiques pures et appliquées , 19(1-4):83-105.
- Franklin, J. and Lorenz, J. (1989). On the scaling of multidimensional matrices. Linear Algebra and its Applications , 114-115:717-735.
- Genevay, A., Chizat, L., Bach, F., Cuturi, M., and Peyré, G. (2019). Sample Complexity of Sinkhorn Divergences. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics , pages 1574-1583. PMLR.
- Genevay, A., Peyre, G., and Cuturi, M. (2018). Learning Generative Models with Sinkhorn Divergences. In Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics , pages 1608-1617. PMLR.
- Gushchin, N., Kholkin, S., Burnaev, E., and Korotin, A. (2024). Light and Optimal Schrödinger Bridge Matching.
- Heng, J., De Bortoli, V., and Doucet, A. (2024). Diffusion schrödinger bridges for bayesian computation. Statistical Science , 39(1):90-99.
- Korotin, A., Gushchin, N., and Burnaev, E. (2023). Light Schrödinger Bridge. https://arxiv.org/abs/2310.01174v3.
- Kullback, S. (1968). Probability densities with given marginals. The Annals of Mathematical Statistics , 39(4):1236-1243.
- Lavenant, H., Zhang, S., Kim, Y.-H., Schiebinger, G., et al. (2024). Toward a mathematical theory of trajectory inference. The Annals of Applied Probability , 34(1A):428-500.
- Lemmens, B. and Nussbaum, R. (2013). Birkhoff's version of hilbert's metric and its applications in analysis.
- Léonard, C. (2012). From the schrödinger problem to the monge-kantorovich problem. Journal of Functional Analysis , 262(4):1879-1920.
- Léonard, C. (2013). A survey of the Schrödinger problem and some of its connections with optimal transport.
- Nutz, M. (2021). Introduction to entropic optimal transport. Lecture notes, Columbia University .
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. (2019). Pytorch: An imperative style, highperformance deep learning library. In Advances in Neural Information Processing Systems 32 , pages 8024-8035.

- Pavon, M., Trigila, G., and Tabak, E. G. (2021). The Data-Driven Schrödinger Bridge. Communications on Pure and Applied Mathematics , 74(7):1545-1573.
- Peluchetti, S. (2023). Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling. https://arxiv.org/abs/2304.00917v2.
- Peluchetti, S. (2024). BM2: Coupled Schrödinger Bridge Matching.
- Peyré, G. and Cuturi, M. (2018). Computational Optimal Transport. https://arxiv.org/abs/1803.00567v4.
- Pooladian, A.-A. and Niles-Weed, J. (2024). Plug-in estimation of Schrödinger bridges. https://arxiv.org/abs/2408.11686v1.
- Rüschendorf, L. (1995). Convergence of the iterative proportional fitting procedure. The Annals of Statistics , pages 1160-1174.
- Shi, Y., Bortoli, V. D., Deligiannidis, G., and Doucet, A. (2022). Conditional simulation using diffusion Schrödinger bridges. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence , pages 1792-1802. PMLR.
- Shi, Y., De Bortoli, V ., Campbell, A., and Doucet, A. (2023). Diffusion Schrödinger Bridge Matching.
- Solomon, J., de Goes, F., Peyré, G., Cuturi, M., Butscher, A., Nguyen, A., Du, T., and Guibas, L. (2015). Convolutional wasserstein distances. ACM Transactions on Graphics , 34(4):66:1-66:11.
- Stromme, A. (2023a). Sampling From a Schrödinger Bridge. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics , pages 4058-4067. PMLR.
- Stromme, A. J. (2023b). Minimum intrinsic dimension scaling for entropic optimal transport.
- Tong, A., Malkin, N., Fatras, K., Atanackovic, L., Zhang, Y., Huguet, G., Wolf, G., and Bengio, Y. (2023). Simulation-free Schrödinger bridges via score and flow matching. https://arxiv.org/abs/2307.03672v3.
- Vargas, F., Thodoroff, P., Lawrence, N. D., and Lamacraft, A. (2021). Solving Schrödinger Bridges via Maximum Likelihood. Entropy , 23(9):1134.
- Vershynin, R. (2018). High-Dimensional Probability: An Introduction with Applications in Data Science . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 1 edition.
- Wang, G., Jiao, Y., Xu, Q., Wang, Y., and Yang, C. (2021). Deep generative learning via schrödinger bridge. In International conference on machine learning , pages 10794-10804. PMLR.
- Yao, R., Nitanda, A., Chen, X., and Yang, Y. (2025). Learning density evolution from snapshot data. arXiv preprint arXiv:2502.17738 .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims in the abstract and introduction accurately reflect the theoretical claims in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We discuss the limitations of our work in the section "Conclusion and Discussion".

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

Justification: We list all assumptions in the main text and include complete proofs in the Appendix.

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

Justification: To support reproducibility, we provide information on the experimental setup and hyperparameter configurations in Section 4.

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

Justification: We submit the source code and data through OpenReview.

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

Justification: We provide information on the experimental setup and hyperparameter configurations in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report the average accuracies over 10 independent runs.

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

Justification: We use a single A100 GPU card for our experiments. All examples are reproducible within one hour.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research is a theoretical study on the Schrödinger bridge and does not raise any ethical concerns. Therefore, the paper does not violate the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our research conducts a statistical analysis of existing methods for the Schrödinger bridge problem. As such, the paper does not present any direct positive or negative societal impacts.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

## Answer: [NA]

Justification: Our research focuses on theoretical studies of the Schrödinger bridge problem, and all experiments are designed to validate the theoretical findings. As such, the paper does not pose any risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

## Answer: [Yes]

Justification: We cite the original papers for the assets used in our experiments.

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

Justification: Our research focuses on theoretical studies of the Schrödinger bridge problem, and all experiments are designed to validate the theoretical findings. As such, the paper does not release any new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: Our research does not rely on LLMs for any important, original, or nonstandard components of the core methods.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

## A Omitted Proofs

## A.1 Proof of Theorem 1

We here provide the complete definitions of path measures P k m,n , P ∗ m,n ∈ P (Ω) . Using the Sinkhorn iterations ( f ( k ) m,n , g ( k ) m,n ) with respect to µ m and ν n , we define the drift function b ( k ) m,n as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Then, P ( k ) m,n is defined as the path measure of the following SDE:

<!-- formula-not-decoded -->

P ∗ m,n ∈ P (Ω) is also defined in the similar way by replacing the Sinkhorn iterations to the corresponding optimal Schrödinger potentials. That is, using the optimal Schrödinger potentials ( f ∗ m,n , g ∗ m,n ) with respect to µ m and ν n , we define the drift function b ∗ m,n as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Then, P ∗ m,n is defined as the path measure of the following SDE:

<!-- formula-not-decoded -->

Theorem 1 (Statistical convergence rate) . Suppose Assumptions 1, 2 and 3 hold. Let P ∗ ∞ , ∞ be the path measure of the true Schrödinger bridge for marginals µ, ν . Then, it follows that for any τ ∈ [0 , 1) ,

<!-- formula-not-decoded -->

where P ∗ , [0 ,τ ] ∞ , ∞ and P ∗ , [0 ,τ ] m,n are restrictions of P ∗ ∞ , ∞ and P ∗ m,n on the time-interval [0 , τ ] .

Proof of Theorem 1. The proof of Theorem 1 is based on the main idea presented in Pooladian and Niles-Weed (2024); Stromme (2023b): we introduce the following entropic plan.

<!-- formula-not-decoded -->

Here, the function ¯ f ∞ ,n : R d → R is defined as follows:

<!-- formula-not-decoded -->

Denoting by ¯ P ∞ ,n the measure obtained by substituting the coupling (17) into (8), Proposition 3.1 in Pooladian and Niles-Weed (2024) implies that the drift ¯ b ∞ ,n with law ¯ P ∞ ,n is expressed as follows:

<!-- formula-not-decoded -->

By incorporating the path measure ¯ P ∞ ,n into the bound via the triangle inequality and subsequently applying Pinsker's inequality, we obtain:

<!-- formula-not-decoded -->

We analyze these two terms separately. For the first term, we apply Proposition 4.3 of Pooladian and Niles-Weed (2024)

<!-- formula-not-decoded -->

For the second term, we apply Girsanov's theorem to derive the difference between the drifts.

<!-- formula-not-decoded -->

where for the second inequality, we used Lemma 1.

Lemma 1. Under assumptions of Theorem 1, we have

<!-- formula-not-decoded -->

Proof of Lemma 1. We perform the expansion while keeping Y j and z fixed

<!-- formula-not-decoded -->

From the case ( p, q ) = (1 , ∞ ) of Lemma 12, we have

<!-- formula-not-decoded -->

By applying Lemma 11, we obtain

<!-- formula-not-decoded -->

## A.2 Proof of Theorem 2

Theorem 2 (Algorithmic convergence rate) . Under Assumptions 1 and 2, we get for any τ ∈ [0 , 1) ,

<!-- formula-not-decoded -->

where P k, [0 ,τ ] m,n are restrictions of P k m,n on the time-interval [0 , τ ] .

Proof of Theorem 2. We start by applying Girsanov's theorem to obtain a difference in the drifts

<!-- formula-not-decoded -->

where we applied Lemma 2.

Lemma 2. Under assumptions of Theorem 1, we have

<!-- formula-not-decoded -->

Proof of Lemma 2. We perform the expansion while keeping Y j and z fixed

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We define w ∗ , w k , v ∗ , v k ∈ R n as follows:

<!-- formula-not-decoded -->

where both w ∗ and w k satisfy the conditions of a probability vector, i.e., w ∗ j , w k j &gt; 0 , ∑ n j =1 w ∗ j = ∑ n j =1 w k j = 1 .

<!-- formula-not-decoded -->

Since

<!-- formula-not-decoded -->

the quantity λ ( K ) can be upper bounded as follows:

<!-- formula-not-decoded -->

Since the upper bound of λ ( K ) is independent of Y j and z , the proof is complete.

## A.3 Proof of Proposition 1

Proposition 1. The optimal estimators produced by [SF] 2 M, lightSB(-M), DSBM-IMF and BM2 coincide with P ∗ m,n attained by Sinkhorn bridge with k →∞ .

Proof of Proposition 1. It suffices to show that the drift estimated by each method coincides with b ∗ m,n .

## Proof of optimal estimator for [SF] 2 M

The SDE representation of the Brownian bridge W ε | x 0 ,x 1 with endpoints ( x 0 , x 1 ) is given by

<!-- formula-not-decoded -->

The marginal density of the Brownian bridge at time t is p t ( x t | x 0 , x 1 ) = N ( x t | (1 -t ) x 0 + tx 1 , εt (1 -t ) I d ) . There exists an ordinary differential equation (ODE) that preserves the same marginal distributions p t ( x t | x 0 , x 1 ) as the SDE (18). This ODE, referred to as the probability flow ODE , takes the form

<!-- formula-not-decoded -->

Consequently, the drift term of the Brownian bridge SDE can be decomposed into the sum of the probability flow ODE drift u ◦ t ( x t | x 0 , x 1 ) and the score function ∇ log p t ( x t | x 0 , x 1 ) :

<!-- formula-not-decoded -->

In the [SF] 2 Mframework, let π ∗ m,n denote the optimal coupling for the entropic optimal transport (EOT) between µ m and ν n , and consider the mixture distribution Π ∗ = π ∗ m,n W ε | x 0 ,x 1 . Under Π ∗ , two neural networks v θ and s φ are trained to minimize the following objectives:

<!-- formula-not-decoded -->

The estimator for the Schrödinger bridge drift in [SF] 2 Mis then defined as

<!-- formula-not-decoded -->

When both models are sufficiently expressive, the optimal solutions admit the following conditional expectation representations:

<!-- formula-not-decoded -->

Therefore, the drift estimator in [SF] 2 Mcan be written as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proof of optimal estimator for LightSB(-M)

Let S ( µ ) denote the set of Schrödinger bridges whose source distribution is µ . In lightSB(-M), the drift estimator g v is defined with a function v : R d → R as follows:

<!-- formula-not-decoded -->

and let S v be the law of the SDE

<!-- formula-not-decoded -->

For any coupling π m,n ∈ Π( µ m , ν n ) , consider its Brownian-bridge mixture Π = π m,n W ε | x 0 ,x 1 . Then, by (Gushchin et al., 2024, Theorem 3.1),

<!-- formula-not-decoded -->

Restricting S to S v , one obtains (Gushchin et al., 2024, Theorem 3.2)

<!-- formula-not-decoded -->

where Π t 1 denotes the marginal law of Π at times t and 1 , and C ( π m,n ) is a constant independent of S v . lightSB(-M) trains a parameterised model v θ that minimises

<!-- formula-not-decoded -->

The drift g v θ ∗ obtained from the learned v θ ∗ serves as the drift estimator produced by lightSB(-M). When the model capacity is sufficiently large, g v θ ∗ satisfies

<!-- formula-not-decoded -->

## Proof of optimal estimator for DSBM-IMF and BM2

Let F ( v f ) and B ( v b ) denote the laws induced by the following SDEs:

<!-- formula-not-decoded -->

Here, in the second SDE, dt represents an infinitesimal negative time step, and ¯ B t denotes the time-reversed Brownian motion.

Given drifts v f ′ and v b ′ , define the following loss functions:

<!-- formula-not-decoded -->

In DSBM-IMF, starting from initial drifts v (0) f and v (0) b , we perform the following iterative updates:

<!-- formula-not-decoded -->

In BM2, given initial drifts v (0) f and v (0) b , the iterative procedure is:

<!-- formula-not-decoded -->

For both methods, F ( v ( k ) f ) and B ( v ( k ) b ) converge to Π ∗ as k → ∞ (Shi et al., 2023, Theorem 8), (Peluchetti, 2024, Lemma 1).

Therefore, letting v ∗ f and v ∗ b denote the limits of v ( k ) f and v ( k ) b as k → ∞ , we have that v ∗ f equals b ∗ m,n :

<!-- formula-not-decoded -->

## A.4 Proof of Proposition 2

Proposition 2. With initialization based on the reference process, both DSBM-IMF and BM2 produce the same estimator as P k m,n of the Sinkhorn bridge for all iterations k .

Proof of Proposition 2. We begin by introducing the Iterative Proportional Fitting (IPF) method (Fortet, 1940; Kullback, 1968; Rüschendorf, 1995). IPF provides one means of solving Eq. (7) and generates a sequence of path measures ( ˜ P ( k ) ) k ∈ N according to

<!-- formula-not-decoded -->

We initialize with ˜ P (0) = W ε . It is known (Léonard, 2013; Nutz, 2021) that the sequence ( ˜ P ( k ) ) k ∈ N satisfies

<!-- formula-not-decoded -->

Here the coupling π f,g is defined by

<!-- formula-not-decoded -->

Hence it suffices to show that the drifts estimated by each algorithm coincide with b ( k ) m,n .

When the initial drift for DSBM is set to v (0) b = 0 , the sequence B ( v (0) b ) , F ( v (1) f ) , B ( v (1) b ) , F ( v (2) f ) , B ( v (3) b ) , . . . coincides with the IPF sequence ( ˜ P ( k ) ) k ∈ N (Shi et al., 2023, Proposition 10). Consequently, v ( k +1) f coincides with b ( k ) m,n .

<!-- formula-not-decoded -->

When the initial drift for BM2 is initialized as v (0) f = v (0) b = 0 , the sequence B ( v (0) b ) , F ( v (1) f ) , B ( v (2) b ) , . . . coincides with the IPF sequence ( ˜ P ( k ) ) k ∈ N (Peluchetti, 2024, Theorem 1). Thus, v (2 k +1) f is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, the sequence F ( v (0) f ) , B ( v (1) b ) , F ( v (2) f ) , . . . corresponds to IPF with the update order reversed, and the same conclusion holds for v (2 k ) f .

## B Technical Lemmas

## B.1 Lemmas used for Theorems 1 and 2

Lemma 3 (Stromme, 2023b, Proposition 14) . Under assumptions of Theorem 1, we have

<!-- formula-not-decoded -->

.

In the following, p ∗ ∞ , ∞ denotes the Radon-Nikodym derivative regarding µ ⊗ ν , defined in Eq. (6), and p ∗ m,n is also defined in a similar manner replacing f ∗ ∞ , ∞ , g ∗ ∞ , ∞ with f ∗ m,n , g ∗ m,n .

Lemma 4 (Stromme, 2023b, Proposition 15) . The population dual potentials f ∗ ∞ , ∞ and g ∗ ∞ , ∞ are 2 R -Lipschitz over supp( µ ) and supp( ν ) , respectively. The extended empirical dual potentials f ∗ m,n and g ∗ m,n are also 2 R -Lipschitz over supp( µ ) and supp( ν ) , respectively. In particular, the population density p ∗ ∞ , ∞ and the extended empirical density p ∗ m,n are each 4 R ε -log-Lipschitz in each of their variables over supp( µ ⊗ ν ) .

Lemma 5 (Stromme, 2023b, Lemma 25) . Under assumptions of Theorem 1, with probability at least 1 -1 n e -20 R 2 /ε

<!-- formula-not-decoded -->

Lemma 6 (Stromme, 2023b, Lemma 26) . Under assumptions of Theorem 1, we have

<!-- formula-not-decoded -->

And, with probability at least 1 -1 n e -20 R 2 /ε

<!-- formula-not-decoded -->

Lemma 7 (Stromme, 2023b, Lemma 27 adapted) . Under the assumptions of Theorem 1, if ε/R is sufficiently small, then

<!-- formula-not-decoded -->

Lemma 7 are the extensions of Stromme (2023b, Lemma 27) that only treats the case of m = n . This extension can be readily verified by a slight modification of the proof in Stromme (2023b).

Lemma 8. Let U m ( y ) = ( µ m -µ ) ( p ∗ ∞ , ∞ ( · , y ) ) . Then the following holds:

<!-- formula-not-decoded -->

.

Proof. We first note that µ ( p ∗ ∞ , ∞ ( · , y )) = 1 since the marginal of π ∗ ∞ , ∞ on y is ν , and hence U m ( y ) = µ m ( p ∗ ∞ , ∞ ( · , y )) -1 . Hoeffding's lemma, with 0 ≤ p ∗ ∞ , ∞ ≲ ( R ε ) d ν , implies that there exists a constant C &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, U m ( y ) is subgaussian with variance proxy C 4 m ( R ε ) 2 d ν . Orlicz ψ 2 -norm is defined for a random variable ξ by

<!-- formula-not-decoded -->

Chen et al. (2023, Theorem 3.7 and Theorem 4.1) with φ ( x ) := x 2 / 2 , p = 2 and d ( y, y ′ ) := ∥ U m ( y ) -U m ( y ′ ) ∥ ψ 2 imply the following estimates:

<!-- formula-not-decoded -->

Let Z i := p ∗ ∞ , ∞ ( X i , y ) -p ∗ ∞ , ∞ ( X i , y ′ ) . Since each Z i is an independent, zero-mean random variable, by Vershynin (2018, Proposition 2.6.1) d ( y, y ′ ) is bounded above as follows:

<!-- formula-not-decoded -->

For each Z i we have ∥ · ∥ ψ 2 ≲ ∥ · ∥ L ∞ ( µ ) , hence

∥ Z i ∥ ψ 2 = ∥ ∥ p ∗ ∞ , ∞ ( X i , y ) -p ∗ ∞ , ∞ ( X i , y ′ ) ∥ ∥ ψ 2 ≲ ∥ ∥ p ∗ ∞ , ∞ ( · , y ) -p ∗ ∞ , ∞ ( · , y ′ ) ∥ ∥ L ∞ ( µ ) .

By | e a -e b | ≤ e a ∨ b | a -b | and Lemma 4, 6, for any x ,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

This concludes N (supp( ν ) , d, r ) ≤ N ( supp( ν ) , ∥ · ∥ 2 , r A ) ( A = C √ m ( R ε ) d ν +1 where C is a constant hidden in the above inequality).

We evaluate the first term in (23) by using Proposition 43 in Stromme (2023b) that shows there exists 0 &lt; c ν ≤ R such that, for any 0 &lt; δ ≤ c ν , the covering number satisfies δ -d ν ≲ N (supp( ν ) , ∥ · ∥ 2 , δ ) ≲ δ -d ν .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ≲ also hides the intrinsic dimensionality d ν .

From (24), the second term of (23) is bounded as

<!-- formula-not-decoded -->

Combining the results of (26) and (25) completes the proof.

<!-- formula-not-decoded -->

Lemma 9. Let V n ( x ) = ( ν n -ν ) ( p ∗ ∞ , ∞ ( x, · ) ) . Then the following holds:

<!-- formula-not-decoded -->

Proof. Lemma 9 can be established by a minor modification of the proof of Lemma 8. The only difference is the evaluation of the covering number: the bound over supp( µ ) is replaced by that over B (0 , R ) . This substitution introduces a dependence on the ambient dimension d .

Lemma 10. It follows that

<!-- formula-not-decoded -->

Proof. Let

<!-- formula-not-decoded -->

Using ¯ f ∞ ,n , we evaluate ∥ f ∗ ∞ , ∞ -f ∗ m,n ∥ 2 L ∞ ( µ ) as follows.

<!-- formula-not-decoded -->

For the first term, the Schrödinger system (5) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where V n ( x ) := ( ν n -ν )( p ∗ ∞ , ∞ ( x, · )) and we used ν ( p ∗ ∞ , ∞ ( x, · )) = 1 .

For any η ∈ (0 , 1)

, the following holds:

<!-- formula-not-decoded -->

For the first term in (28), the mean value theorem implies that for | u | &lt; 1 we have | log(1 + u ) | ≤ | u | 1 -| u | . Hence,

<!-- formula-not-decoded -->

For the second term in (28), since exp ε ( -10 R 2 ) ≤ p ∗ ∞ , ∞ ≤ exp ε (10 R 2 ) , an application of Markov's inequality yields

<!-- formula-not-decoded -->

Setting η = 1 / 2 and applying Lemma 9 yields

<!-- formula-not-decoded -->

For the second term in (27), Eq. (6.4) in Stromme (2023b) implies

<!-- formula-not-decoded -->

Let E denote the event on which Lemma 5 holds. From Lemma 3 and P [ E C ] ≤ 1 n , it follows that

Combining this with Lemma 7 and Lemma 5 yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 11. It follows that

<!-- formula-not-decoded -->

Proof. Let

<!-- formula-not-decoded -->

Using ¯ g m, ∞ , we evaluate ∥ g ∗ ∞ , ∞ -g ∗ m,n ∥ 2 L ∞ ( ν ) as follows.

<!-- formula-not-decoded -->

For the first term, the Schrödinger system (5) implies

<!-- formula-not-decoded -->

where U m ( y ) := ( µ m -µ )( p ∗ ∞ , ∞ ( · , y )) and used µ ( p ∗ ∞ , ∞ ( · , y )) = 1 .

For any η ∈ (0 , 1) , the following holds:

<!-- formula-not-decoded -->

For the first term in (31), the mean value theorem implies that for | u | &lt; 1 we have | log(1 + u ) | ≤ | u | 1 -| u | . Hence,

<!-- formula-not-decoded -->

For the second term in (31), since exp ε ( -10 R 2 ) ≤ p ∗ ∞ , ∞ ≤ exp ε (10 R 2 ) , an application of Markov's inequality yields

<!-- formula-not-decoded -->

Setting η = 1 / 2 and applying Lemma 8 yields

<!-- formula-not-decoded -->

For the second term of (30), since the gradient of the log-sum-exp function is the softmax and its ℓ 1 -norm equals 1 , we obtain

<!-- formula-not-decoded -->

Therefore, by Lemma 10, we have

<!-- formula-not-decoded -->

Combining the above results, we obtain

<!-- formula-not-decoded -->

Lemma 12 (Lipschitzness of softmax) . For λ &gt; 0 and d ≥ 2 , define σ λ : R d → ∆ d by

<!-- formula-not-decoded -->

Then for any 1 ≤ p, q ≤ ∞ ,

<!-- formula-not-decoded -->

Proof. Let s = σ λ ( z ) . The Jacobian of σ λ at z is

<!-- formula-not-decoded -->

By the fundamental theorem of calculus along the segment t ↦→ y + t ( x -y ) , hence, for any 1 ≤ p, q ≤ ∞ ,

<!-- formula-not-decoded -->

It suffices to bound sup z ∥ J ( z ) ∥ q → p .

Step 1: Decomposition. With the standard basis { e i } d i =1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and by the triangle inequality and ∥ e i -e j ∥ p = 2 1 /p (with the convention 2 1 / ∞ = 1 ),

<!-- formula-not-decoded -->

Step 2: Maximization over s ∈ ∆ d . Let i min = arg min i u i , i max = arg max i u i . Since | u i -u j | ≤ | u i max -u i min | and ∑ i&lt;j s i s j = 1 2 ( 1 -∑ d i =1 s 2 i ) ,

<!-- formula-not-decoded -->

The last inequality is tight when s i max = s i min = 1 2 and all other s i = 0 . Therefore,

<!-- formula-not-decoded -->

Because σ λ ( R d ) is the interior of ∆ d and the right-hand side of (38) is continuous in s , the supremum over z equals the supremum over s ∈ ∆ d .

Thus for any u ∈ R d ,

Step 3: Maximization over ∥ u ∥ q = 1 . For any a, b ∈ R and q ≥ 1 , | a -b | q ≤ 2 q -1 ( | a | q + | b | q ) . With a = u i max , b = u i min and ∥ u ∥ q = 1 ,

<!-- formula-not-decoded -->

with equality when u has exactly two nonzero entries u i max = 2 -1 /q and u i min = -2 -1 /q .

Step 4: Combine. From (38)-(40),

<!-- formula-not-decoded -->

Plug this into (36) to obtain the claim. The bound is tight by the equality cases noted in (39) and (40).

Lemma 13. For any probability vectors u, v ∈ ∆ d ,

<!-- formula-not-decoded -->

Proof. By Cohen and Fausti (2024)[Theorem 5.1],

<!-- formula-not-decoded -->

Since tanh( x ) ≤ x for x ≥ 0 and d Hilb ( u, v ) ≥ 0 , letting x = d Hilb ( u,v ) 4 gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This proves (41).

Lemma 14. For any u , v ∈ R d + , the following holds:

<!-- formula-not-decoded -->

Here, ⊙ denotes the element-wise product.

Proof. The first two properties are established in (Lemmens and Nussbaum, 2013, Lemma 2.1). For the third property, we have

<!-- formula-not-decoded -->

## B.2 Lemma used for the drift estimation using neural networks

We provide an alternative expression of the Sinkhorn bridge drift estimator as a Markovian projection.

Recall that W ε | x 0 ,x 1 is the Brownian bridge connecting x 0 and x 1 , so its marginal distribution at time t ∈ [0 , 1] is a Gaussian distribution W ε t | x 0 ,x 1 = N ( (1 -t ) x 0 + tx 1 , εt (1 -t ) I d ) . For any distributions µ ′ , ν ′ ∈ P ( R d ) , and dual potentials f ∈ L 1 ( µ ′ ) , g ∈ L 1 ( ν ′ ) , we define

<!-- formula-not-decoded -->

We here only consider the case where π becomes a joint distribution π ∈ P ( R d × R d ) , that is, ∫ dπ ( x 0 , x 1 ) = 1 . Using π ( k ) m,n , we define the mixture of Brownian bridge (reciprocal process): Π = ∫ W ε | x 0 ,x 1 dπ ( x 0 , x 1 ) ∈ P (Ω) .

Lemma 15. Under the above setting, it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Given x t ( t &lt; 1) , we denote by Π( x 1 | X t = x t ) the conditional density of Π with respect to Lebesgue measure dx 1 . Note that the marginal density Π t 1 of Π on time points t and 1 with respect to dx t dx 1 satisfies that for a measurable A ⊂ R d

<!-- formula-not-decoded -->

Therefore, we get

<!-- formula-not-decoded -->

This means

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

## C Omitted Experiments

## C.1 Stopping and guidance

Theorems 1 and 2 suggest that beyond a certain point additional Sinkhorn iterations are not beneficial. The total error decomposes into a 'sampling error' and an 'optimization error'. The latter decreases

Table 1: Comparison of ∫ 0 . 9 0 MSE sample ( m,n,t ) dt across intrinsic dimensions d ν .

|   d ν | ∫ 0 . 9 0 MSE sample ( m,n,t ) dt   |
|-------|-------------------------------------|
|     5 | 0 . 031                             |
|    10 | 0 . 138                             |
|    20 | 0 . 562                             |
|    30 | 1 . 178                             |

only through iterations, whereas the former does not. By the triangle inequality for the total variation distance,

<!-- formula-not-decoded -->

Once the second term on the right-hand side (optimization error) becomes no larger than the first term (estimation error), further iterations no longer yield a meaningful reduction of the total error. Hence it is reasonable to choose the stopping point as the smallest k such that

<!-- formula-not-decoded -->

From Theorems 1 and 2 , a sufficient condition for this is

<!-- formula-not-decoded -->

Since log ( tanh ( R 2 ε )) &lt; 0 , iterations are beneficial only when B m + A + B n &lt; 1 . Equivalently, unless the sample size is large enough to make the numerator negative-roughly ( m ∧ n ) ≈ ( A ∨ B ) -the reduction achievable by iterations is smaller than the estimation error, leading to wasted computational resources.

In Figure 3, the fact that b (1) m,n already captures the shape of ν n to a certain extent is consistent with this reasoning. Increasing k becomes meaningful only when µ m and ν n are sufficiently close to the true distributions µ and ν , namely when m and n are sufficiently large.

## C.2 Intrinsic dimension

Theorem 1 shows that the orders of the orders of R and ε depend on the intrinsic dimension d ν rather than the ambient dimension d . In this section, we verify that the error varies with d ν under the following setup.

We fix the ambient dimension at d = 50 , the regularization strength at ε = 0 . 5 , the evaluation-interval endpoint at τ = 0 . 9 , and the sample sizes at m = n = 10000 . The distribution µ is sampled uniformly from the unit hypercube [0 , 1] d , while ν is sampled uniformly from the d ν -dimensional manifold embedded in the unit sphere of radius 1 ,

<!-- formula-not-decoded -->

By varying d ν , we examine whether the convergence behavior of the estimation error with respect to the sample size depends on the manifold dimension.

Specifically, for each d ν ∈ { 5 , 10 , 20 , 30 } we compare

<!-- formula-not-decoded -->

The results are reported in Table 1. An increasing trend of the error with larger manifold dimension is observed.

## C.3 The role of epsilon

We examine the numerical stability of the regularization parameter ε in the Sinkhorn algorithm and provide selection guidelines in relation to the sample sizes ( m,n ) . Using the same setup as

Table 2: Representative ε achieving the target error δ = 1 . 0 for varying m ∧ n (same setup as Experiments 4.1).

|   m ∧ n | ε                |
|---------|------------------|
|      10 | 2 . 684 × 10 8   |
|      20 | 7 . 979 × 10 6   |
|      30 | 3 . 200 × 10 1   |
|      40 | 6 . 797 × 10 - 1 |
|      50 | 3 . 291 × 10 - 1 |
|      60 | 1 . 299 × 10 - 1 |
|      70 | 2 . 010 × 10 - 1 |
|      80 | 9 . 766 × 10 - 4 |

Experiments 4.1, for each given m,n we search for the smallest ε that achieves the target error δ = 1 . 0 .

As m ∧ n increases, the ε required to meet the target error exhibits a two-phase behavior: a steep initial decrease followed by an asymptotically mild decay. Representative values are reported in Table 2.

By Theorem 1, the ε required to satisfy the target error δ obeys

<!-- formula-not-decoded -->

This indicates a steep improvement in the smallm ∧ n regime followed by logarithmically slow decay, which aligns with the empirical results. Since excessively small ε can cause numerical instability, using the above expression as a lower-bound guideline for selecting ε is effective.
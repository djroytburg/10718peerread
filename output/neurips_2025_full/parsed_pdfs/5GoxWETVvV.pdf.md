## Sampling by averaging: A multiscale approach to score estimation

## Paula Cordero-Encinar

Department of Mathematics Imperial College London paula.cordero-encinar22@imperial.ac.uk

## Sebastian Reich

Institut für Mathematik Universität Potsdam sereich@uni-potsdam.de

## Andrew B. Duncan

Department of Mathematics Imperial College London a.duncan@imperial.ac.uk

## O. Deniz Akyildiz

Department of Mathematics Imperial College London

deniz.akyildiz@imperial.ac.uk

## Abstract

Weintroduce a novel framework for efficient sampling from complex, unnormalised target distributions by exploiting multiscale dynamics. Traditional score-based sampling methods either rely on learned approximations of the score function or involve computationally expensive nested Markov chain Monte Carlo (MCMC) loops. In contrast, the proposed approach leverages stochastic averaging within a slowfast system of stochastic differential equations (SDEs) to estimate intermediate scores along a diffusion path without training or inner-loop MCMC. Two algorithms are developed under this framework: MULTALMC, which uses multiscale annealed Langevin dynamics, and MULTCDIFF, based on multiscale controlled diffusions for the reverse-time Ornstein-Uhlenbeck process. Both overdamped and underdamped variants are considered, with theoretical guarantees of convergence to the desired diffusion path. The framework is extended to handle heavy-tailed target distributions using Student's t-based noise models and tailored fast-process dynamics. Empirical results across synthetic and real-world benchmarks, including multimodal and high-dimensional distributions, demonstrate that the proposed methods are competitive with existing samplers in terms of accuracy and efficiency, without the need for learned models.

## 1 Introduction

Efficiently sampling from complex unnormalised probability distributions is a fundamental challenge across many scientific domains, including statistics, chemistry, computational physics, and biology, see, e.g. Gelman et al. [2013], Lelièvre et al. [2010], Liu [2008]. Classical Markov Chain Monte Carlo (MCMC) methods provide asymptotically unbiased samples under mild assumptions on the target [Durmus and Moulines, 2017, Mousavi-Hosseini et al., 2023]. However, they often become computationally inefficient in practice, especially for high-dimensional, multimodal targets, due to the need for long Markov chains to ensure adequate mixing and convergence. To alleviate these issues, annealing-based strategies construct a sequence of smoother intermediate distributions bridging a simple base distribution and the target. This principle underlies algorithms such as Parallel Tempering [Geyer and Thompson, 1995, Swendsen and Wang, 1986], Annealed Importance Sampling [Neal, 2001], and Sequential Monte Carlo [Del Moral et al., 2006, Doucet et al., 2001].

Recently, score-based diffusion methods [Hyvärinen, 2005, Song and Ermon, 2020] have emerged as powerful models capable of generating high-quality samples. These approaches simulate a reverse

diffusion process guided by time-dependent score functions. While successful in generative modelling, they rely on access to training samples-a setting that differs from sampling problems, where only the unnormalised target density is available. Therefore, traditional score-matching techniques are not applicable. To address this, a growing body of work has explored ways to estimate the score function using only the target density, either through neural networks using variational objectives [Chen et al., 2025, Richter and Berner, 2024, Vargas et al., 2024] or training-free alternatives based on MCMC [Chen et al., 2024, Grenioux et al., 2024, Huang et al., 2024, Phillips et al., 2024, Saremi et al., 2024]. In this work, we present a novel training-free sampling framework that eliminates the need for the inner MCMC loop required in previous works to approximate the score function. Our main contributions are as follows:

- We demonstrate how a multiscale system of SDEs can be leveraged to enable efficient sampling along a diffusion path, obviating the need to estimate the score function.
- In particular, we propose two algorithms: MULTALMC: Multiscale Annealed Langevin Monte Carlo (Section 3.1) and MULTCDIFF: Multiscale Controlled Diffusions (Section 3.2) based on discretisations of two different multiscale SDEs: annealed Langevin dynamics for general noise schedules and the reverse SDE associated with an Ornstein-Uhlenbeck (OU) noising process, respectively. We also explore accelerated versions using underdamped dynamics.
- We provide a theoretical analysis of the proposed methods (Section 4) and demonstrate their effectiveness across different synthetic and real-world benchmarks (Section 5).

## 2 Background and related work

## 2.1 Diffusion paths

The reverse process in diffusion models consists in sampling along a path of probability distributions ( µ t ) t ∈ [0 , 1] , which starts at a simple distribution and ends at the target distribution π ∝ e -V π on R d . Following Chehab and Korba [2024], the intermediate distributions can be expressed as follows

<!-- formula-not-decoded -->

where ∗ denotes the convolution operation, ν describes the base or noising distribution, and λ t is an increasing function called schedule, such that, λ t ∈ [0 , 1] and λ 1 = 1 . We refer to the path ( µ t ) t ∈ [0 , 1] in (1) as the diffusion path . By choosing λ t = e -2 T (1 -t ) for some fixed T , we recover the path corresponding to a forward OU noising process, a widely used approach in diffusion models [Benton et al., 2024, Chen et al., 2023, Song et al., 2021]. In this case, the reverse-time dynamics can be characterised [Anderson, 1982], requiring only to estimate the score functions ∇ log µ t along the path. However, for general diffusion paths, the reverse process cannot be described by a closed-form SDE due to intractable control terms in the drift, which can be estimated using neural networks as in Albergo et al. [2023]. Alternatively, sampling from the diffusion path can be achieved through annealed Langevin dynamics [Cordero-Encinar et al., 2025, Song and Ermon, 2019], which also relies on score estimation but avoids dealing with intractable drift terms.

The diffusion path has demonstrated very good performance in the generative modelling literature where the score of the intermediate distributions can be estimated empirically using score matching techniques [Song and Ermon, 2019]. In the context of sampling, however, estimating the score is more challenging, as samples from the data distribution are not available and instead we only have access to an unnormalised target probability distribution. In particular, under the assumption that ν, π are bounded and have finite second-order moments, and that |∇ log ν | and |∇ V π | are bounded, the expression of the score of the intermediate distributions is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ρ t,x ( y ) ∝ ν x -y √ 1 -λ t e -V π ( y/ λ t ) and we have used that ∇ ( f ∗ g ) = ( ∇ f ) ∗ g = f ∗ ( ∇ g ) when f and g are differentiable functions. Notably, in the limits λ t = 0 and λ t = 1 , the conditional distribution ρ t,x converges to a Dirac delta function. The score in (2) involves an expectation over ρ t,x , which typically requires MCMC sampling and results in a computationally expensive nested-loop structure. We avoid this by exploiting multiscale dynamics, as described in the following section.

## 2.2 Multiscale dynamics and stochastic averaging

Consider the slow-fast system of SDEs

<!-- formula-not-decoded -->

where 0 &lt; ε ≪ 1 controls the scale separation between the slow X t and the fast component Y t . We assume that, for ε = 1 and fixed t and X t , Y t is ergodic. As ε → 0 , the fast component Y t evolves on a much shorter timescale than X t due to the time-rescaling property of Brownian motion. Intuitively, this means that Y t rapidly reaches its stationary distribution while X t remains substantially unchanged. In the limit ε → 0 , the slow-process X t converges to the averaged dynamics ¯ X t which is given by

<!-- formula-not-decoded -->

where ρ t,x is the invariant measure of the frozen process d Y s = f ( t, x, Y s )d s + √ 2d ˜ B s . In computational statistics, stochastic slow-fast systems have been leveraged to simulate the averaged dynamics efficiently [Akyildiz et al., 2024, Harvey et al., 2011, Pavliotis and Stuart, 2008, Weinan et al., 2005]. In our context of sampling from diffusion paths, the averaged drift ¯ b ( t, x ) corresponds to the score ∇ log µ t ( x ) , which is defined as an expectation (see Eq. (2)). By leveraging multiscale dynamics, we can approximate this expectation without relying on a nested-loop sampling structure, a key insight that underpins our method, as summarised in Section 3.

## 2.3 Related work

Diffusion-based samplers Several recent works have proposed sampling algorithms based on the diffusion path. We divide them into two categories: neural samplers and learning-free samplers. Neural samplers such as those in Chen et al. [2025], Noble et al. [2025], Richter and Berner [2024], Vargas et al. [2024], Zhang and Chen [2022] estimate the time-dependent drift function of the diffusion process using a parametrised model, typically a neural network, by solving a variational inference problem defined over the space of path measures. As in diffusion generative models, their performance is limited by the expressiveness of the chosen model class. In contrast, learning-free samplers [Chen et al., 2024, Grenioux et al., 2024, He et al., 2024b, Huang et al., 2024, Phillips et al., 2024, Saremi et al., 2024] do not require any training similar to our work. Instead, they estimate the score function ∇ log µ t using MCMC samples from the conditional distribution ρ t,x . That is, these approaches involve a bi-level structure, with an inner MCMC loop used at each step of the outer diffusion-based sampler. In this sense, they implement a Langevin-within-Langevin strategy. Our approach avoids nested loops by leveraging multiscale dynamics along the diffusion path. This results in a Langevin-by-Langevin sampler that is both simpler and more computationally efficient.

Besides, while acceleration techniques based on underdamped dynamics [Duncan et al., 2017, Monmarché, 2020] have shown promise in the generative modelling literature [Dockhorn et al., 2022a,b, Holderrieth et al., 2024], and recent work by Blessing et al. [2025] introduces neural underdamped diffusion samplers, no training-free accelerated samplers based on diffusion processes have been proposed. Addressing this gap is one of the key contributions of our work.

Proximal samplers Our approach is also closely related to the class of proximal samplers [Chen et al., 2022, Lee et al., 2021b]. These methods define an auxiliary joint distribution of the form ˜ π ( x, z ) ∝ π ( x ) exp( -∥ x -z ∥ 2 / (2 η )) and perform Gibbs sampling by alternating between the convolutional distribution ˜ π ( z | x ) and the denoising posterior ˜ π ( x | z ) . In practice, sampling from the denoising posterior ˜ π ( x | z ) involves finding a mode using gradient-based optimisation techniques, followed by rejection sampling around that mode. This step can be computationally expensive and inefficient, especially in high-dimensional settings. Our method improves upon these approaches by replacing the proximal-point-style posterior sampling step with a fast-timescale dynamics, enabling more efficient exploration of the denoising posterior. Furthermore, unlike proximal samplers that operate at a fixed noise level, our approach targets the entire diffusion path, smoothly interpolating between the base distribution and the target.

## 3 Sampling using multiscale dynamics

A key challenge in sampling from the diffusion path in (1) is accurately estimating the score function when we do not have direct access to samples. Since the score of the intermediate distributions along the diffusion path is computed as an expectation over the conditional distribution ρ t,x (see Eq. (2)), we propose to sample from the time-dependent conditional distribution ρ t,x using an alternative diffusion process with a shorter timescale than the original dynamics following the path in (1). This enables both processes to run in parallel, unlike existing methods that execute them sequentially. In those approaches, each iteration of the original diffusion process (outer loop) requires multiple time steps of the process targeting the conditional distribution ρ t,x (inner loop) to learn the score function, leading to increased computational cost and a higher number of evaluations of the target potential. By employing different timescales for each diffusion process, our method improves sampling efficiency while ensuring convergence to the correct target distribution, as justified by stochastic averaging theory [Liu et al., 2020]. We explore this approach in two settings: (i) when the original diffusion is a Langevin dynamics driven by the scores of the intermediate probability distribution ∇ log µ t (Section 3.1), and (ii) when we use the reverse SDE corresponding to an OU noising process (Section 3.2).

## 3.1 MULTALMC: Multiscale Annealed Langevin Monte Carlo

Following Cordero-Encinar et al. [2025], Song and Ermon [2019], we use a time inhomogeneous Langevin SDE to sample from the diffusion path (1) given by

<!-- formula-not-decoded -->

where ˆ µ t denotes the reparametrised probability distributions from the diffusion path (ˆ µ t = µ κt ) t ∈ [0 , 1 /κ ] , for some 0 &lt; κ &lt; 1 , X 0 ∼ µ 0 = ν and ( B t ) t ≥ 0 is a Brownian motion. We consider both Gaussian diffusion paths where the base distribution ν is a Normal distribution and heavy-tailed diffusions corresponding to a Student's t base distribution. Since the scores ∇ log ˆ µ t are intractable and computed as expectations over ρ t,x (see Eq. (2)), we adopt a multiscale system to approximate the averaged dynamics in (3). The fast process targetting the conditional distribution ρ t,x can follow any fast-mixing dynamics. In particular, we explore the use of overdamped Langevin dynamics or a modified It ˆ o diffusion [He et al., 2024a], but it is important to remark that our framework is more general. When using overdamped Langevin dynamics, the fast process is given by

<!-- formula-not-decoded -->

where ˆ ρ t,X t is a reparametrised version of ρ t,X t and ( ˜ B t ) t ≥ 0 is a Brownian motion on R d . This suggests sampling from the following stochastic slow-fast system

<!-- formula-not-decoded -->

This system converges to the averaged dynamics described in (3) as ε → 0 , since taking the expectation of the drift term in the X t dynamics with respect to Y t ∼ ˆ ρ t,X t recovers the score function ∇ log ˆ µ t (see Equation (2) and Section 4). For numerical implementation, we use an adaptive scheme that alternates between the two expressions for X t depending on the value of λ κt in order to avoid numerical instabilities as λ κt approaches 0 or 1 . The resulting system of SDEs can be discretised using different numerical schemes, however, standard integrators such as Euler-Maruyama (EM) will be unstable when ε is small, due to the large-scale separation between the slow and fast processes. To mitigate this, we leverage discretisations which remain stable, despite the stiffness of the SDE, such as the exponential integrator [Hochbruck and Ostermann, 2010] and the SROCK method [Abdulle et al., 2018]. Furthermore, since ρ t,x converges to a Dirac delta distribution centred at 0 when λ κt = 0 and centred at x when λ κt = 1 , we apply a further slight modification to the slow-fast system to improve numerical stability (see App. B.1.1 for details).

Accelerated MULTALMC When implemented in practice, the overdamped system in (4) requires a very small value of κ (which corresponds to a slowly varying dynamics driven by ∇ log ˆ µ t ) to accurately follow the marginals of the path (ˆ µ t ) t ∈ [0 , 1 /κ ] and ultimately sample from the target distribution. This results in a large number of discretisation steps, thus making the method computationally expensive, see App. E.1 for more details. In contrast, underdamped dynamics exhibit faster mixing times [Bou-Rabee and Eberle, 2023, Eberle and Lörler, 2024], which intuitively helps the system explore the space more efficiently and better track the intermediate distributions along the path (ˆ µ t ) t ∈ [0 , 1 /κ ] . These dynamics have also shown strong empirical performance in the neural sampling literature [Blessing et al., 2025], requiring fewer discretisation steps. Motivated by these advantages, we propose augmenting the state space with auxiliary velocity variables ¯ V t , and adopting the following underdamped Langevin diffusion to sample from the path (ˆ µ t ) t ∈ [0 , 1 /κ ]

<!-- formula-not-decoded -->

The mass parameter M determines the coupling between position X t and velocity V t , while the friction coefficient Γ controls the strength of noise injected into the velocity component. Both M and Γ are symmetric positive definite matrices, in practice we consider M = mId and Γ = γId . The behaviour of the system is governed by the interplay between M and Γ [McCall, 2010]. Based on this, we propose to use the following underdamped slow-fast system to sample from the target distribution

<!-- formula-not-decoded -->

Note that in the underdamped setting, both X t , V t are treated as slow variables, and as ε → 0 , this system converges to the averaged dynamics in (5). To implement the system in practice, we combine a symmetric splitting scheme (OBABO) [Monmarché, 2020] for the slow-dynamics, and discretise the fast-dynamics using SROCK method [Abdulle et al., 2018]. We will evaluate the performance of the proposed sampler based on this underdamped slow-fast system (Algorithm 3), referred to as MULTALMC, under two choices of the reference distribution ν : a standard Gaussian and a Student's t distribution. A simplified version of the implementation is presented in Alg. 1, while a more detailed description is provided in App. B.1.2.

Heavy-tailed diffusions When the target distribution has heavy tails, standard Gaussian diffusions often fail to capture the correct tail behaviour [Pandey et al., 2025, Shariatian et al., 2025]. This motivates the use of heavy-tailed noising processes, which have been shown to offer theoretical guarantees in such settings [Cordero-Encinar et al., 2025].

We propose sampling along a heavy-tailed diffusion path (ˆ µ ) , where the reference distribu- t t ∈ [0 , 1 /κ ] tion ν is chosen as a Student's t-distribution with tail index α , i.e., ν ∝ (1 + ∥ x ∥ 2 /α ) -( α + d ) / 2 . This can be implemented using the underdamped slow-fast system defined in (6). However, when the target distribution is heavy-tailed, the use of overdamped Langevin dynamics for the fast component results in slow convergence [Mousavi-Hosseini et al., 2023, Wang, 2006]. This motivates the consideration of alternative diffusion processes for the fast dynamics that mix more efficiently under such conditions. Specifically, we explore a modified Itô diffusion proposed in He et al. [2024a], defined as

<!-- formula-not-decoded -->

This modified Itô diffusion for heavy-tailed distributions is shown to have faster convergence guarantees when the target has finite variance [He et al., 2024a]. By replacing the fast process in (6) with the diffusion defined in (7), we obtain samplers better suited for heavy-tailed targets, with improved convergence properties.

## 3.2 MULTCDIFF: Multiscale Controlled Diffusions

Annealed Langevin dynamics (3) offers a convenient approach to sampling from the target distribution for general schedules λ t that interpolate between the base and target distributions in finite time.

## Algorithm 1 MULTALMC sampler: accelerated version

Require: Schedule function λ t , value for λ δ and ˜ λ , friction coefficient Γ , mass parameter M , number of sampling steps L , time discretisation 0 = T 0 &lt; · · · &lt; T L = 1 /κ , step size h l = T l +1 -T l . Constants µ , ν , κ for SROCK step from (14), (15).

<!-- formula-not-decoded -->

for l = 0 to L do

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Half-step for velocity component

<!-- formula-not-decoded -->

Full EM-step for position component X l +1 = X l + h l M -1 V ′′ l

<!-- formula-not-decoded -->

Half-step for velocity component

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Half-step for velocity

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Full EM-step for position component X l +1 = X l + h l M -1 V l

Half-step for velocity component

<!-- formula-not-decoded -->

end for

<!-- formula-not-decoded -->

However, a key drawback of annealed Langevin dynamics is that it introduces bias, even if perfectly simulated (see Section 4.2). This bias can be corrected by incorporating control terms. Although such control terms are generally intractable for arbitrary schedules, they can be computed explicitly in the case of an OU noising process. The multiscale approach introduced in the previous section naturally extends to this controlled setting, in both overdamped and underdamped regimes. Given the improved efficiency of underdamped dynamics, we focus our discussion on the underdamped formulation (see App. B.2.1 for details on the overdamped case). When using an OU process as the forward noising - → - →

model, the underdamped diffusion, initialised at X 0 ∼ π and V 0 ∼ N (0 , I ) , is defined as

<!-- formula-not-decoded -->

In this case, to ensure efficient and stable dynamics, we adopt the critical damping regime, where Γ 2 = 4 M . This choice leads to fast convergence without oscillations [McCall, 2010]. The corresponding time-reversed SDE is

<!-- formula-not-decoded -->

where q t is the marginal distribution of the forward process which has the following expression

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with φ = N (0 , I ) and conditional distribution ρ t ( x, v | y, v 0 ) normally distributed with mean m t ( y, v 0 ) and covariance Σ t given by

<!-- formula-not-decoded -->

We observe that v 0 can be analytically integrated out in the expression for ρ t ( x, v | y, v 0 ) . Moreover, the conditional distribution ρ t ( v | x, y ) remains Gaussian, with mean ˜ m t ( x, y ) and covariance σ 2 t I , where ˜ m t ( x, y ) is linear in y . This structure allows us to express the score ∇ v log q t ( x, v ) as

<!-- formula-not-decoded -->

where f t is a linear function of its arguments, see App. B.2 for a detailed derivation. Using the expressions above, we define a multiscale SDE system that enables sampling from the target distribution via the reverse SDE (8), without requiring prior estimation of the denoiser E Y | x,v [ Y ] that appears in the expression for ∇ v log q T -t ( x, v ) (9). In this formulation, the fast process is modelled by an overdamped Langevin SDE that targets the conditional distribution Y | ( ← -X t , ← -V t ) , given by

<!-- formula-not-decoded -->

Combining this fast process with the reverse SDE, we obtain the following stochastic slow-fast system, which we use for sampling from the target distribution

<!-- formula-not-decoded -->

To implement this sampler, referred to as MULTCDIFF, we construct a novel integrator inspired by Dockhorn et al. [2022b]. Specifically, we leverage the symmetric Trotter splitting and the Baker-Campbell-Hausdorff formula [Strang, 1968, Trotter, 1959, Tuckerman, 2010] to design a stable and efficient symmetric splitting scheme, see App. B.2 for details. We also outline in the appendix the difficulties of extending controlled diffusions to the heavy-tailed scenario.

## 4 Theoretical guarantees

In this section, we identify the different sources of error in the proposed sampling algorithms. For MULTALMC, the total error consists of three components: the discretisation error from numerically solving the slow-fast system, the convergence error of the slow-fast system to its averaged dynamics, and the bias arising from the time-inhomogeneous averaged system, whose marginals differ from those of the diffusion path in (1)-this last component is discussed in 4.2. For MULTCDIFF, the error bound includes an initialisation error present in the error bounds of diffusion models [Benton et al., 2024, Chen et al., 2023], the discretisation error of the slow-fast system, and the convergence of the multiscale system to the averaged dynamics. The discretisation error depends on the choice of numerical integrator, underscoring the importance of an accurate and stable integrator scheme. We do not analyse this error in detail as it follows established methods [Leimkuhler et al., 2024, Monmarché, 2020]. We now study the convergence properties of the slow component X t to the solution of the corresponding averaged dynamics that evolves along the time-inhomogeneous diffusion path in (1).

## 4.1 Convergence of the slow-fast system to the averaged dynamics

Building on results from stochastic averaging theory [Liu et al., 2020], our goal is to derive sufficient conditions on the coefficients of the multiscale systems-namely, (4), (6), and (10)-that guarantee strong convergence of the slow component to the averaged process as ε → 0 . We focus on the case where the base distribution is Gaussian ν ∼ N (0 , I ) . Extending the convergence analysis to the case where ν is a heavy-tailed distribution, such as a Student's t distribution, presents significant technical challenges, which fall outside of the scope of this work. We outline these challenges in App. C.2. To establish convergence guarantees, we assume the following regularity conditions.

A1. The target distribution π has density with respect to the Lebesgue measure, which we write π ∝ e -V π . The potential V π has Lipschitz continuous gradients, with Lipschitz constant L π . In addition, V π is strongly convex with convexity parameter M π &gt; 0 .

A2. The schedule λ t : R + → [0 , 1] is a non-decreasing function of t , weakly differentiable and Hölder continuous with exponent γ 1 in (0 , 1] .

Under these assumptions, we obtain the following result.

Theorem 4.1. Let the base distribution ν ∼ N (0 , I ) . Suppose the target distribution π and the schedule function λ t satisfy assumptions A 1 and A 2, respectively. Then, for any ε ∈ (0 , ε 0 ) , where ε 0 is specified in App. C.1, and any given initial conditions, there exists unique solutions { ( X ε t , Y ε t ) , t ≥ 0 } , { ( X ε t , V ε t , Y ε t ) , t ≥ 0 } and { ( ← -X ε t , ← -V ε t , Y ε t ) , t ≥ 0 } to the slow-fast stochastic systems (4) , (6) and (10) , respectively. Furthermore, for any p &gt; 0 , it holds that

<!-- formula-not-decoded -->

where ¯ X t denotes the solution of the averaged system and X ε t is the corresponding slow component of the multiscale systems (4) , (6) and (10) .

The proof is provided in App. C.1. The theorem applies to both the annealed Langevin dynamics and the controlled diffusion. We note that the strong convexity condition in A 1-which guarantees exponential convergence of the fast process to its unique invariant measure when ε = 1 is fixed-is a restrictive assumption. However, as shown in Section 5, our proposed algorithms show strong empirical performance on benchmark distributions that do not satisfy this condition. Extending the theoretical analysis to cover non-log-concave targets remains an important future direction.

## 4.2 Bias of the annealed Langevin dynamics

The bias of the overdamped annealed Langevin dynamics (3) has been studied in prior work [CorderoEncinar et al., 2025, Guo et al., 2025]. Here, we focus on quantifying the bias introduced by the underdamped averaged system (5) relative to the true diffusion path in the augmented state space with the velocity. At fixed time t , the invariant distribution of the underdamped averaged system takes the form

<!-- formula-not-decoded -->

Note that the Hamiltonian is separable, meaning that x and v are independent. It is important to emphasise that, even when simulated exactly, diffusion annealed Langevin dynamics introduces a bias, as the marginal distributions of the solution of (5) do not exactly match the prescribed distributions (ˆ p t ) t . Akey quantity for characterising this bias will be the action of the curve of probability measures µ = ( µ t ) t ∈ [0 , 1] defined in (1) interpolating between the base distribution ν and the target distribution π , denoted by A ( µ ) . As noted by Guo et al. [2025], the action serves as a measure of the cost of transporting ν to π along the specified path. Formally, for an absolutely continuous curve of probability measures [Lisini, 2007] with finite second-order moment, the action is given by

<!-- formula-not-decoded -->

Based on Theorem 1 from Guo et al. [2025], we have the following result.

Theorem 4.2. Let Q U-ALD = ( q t, U-ALD ) t ∈ [0 , 1 /κ ] be the path measure of the diffusion annealed Langevin dynamics (5) , and Q = (ˆ p t ) t ∈ [0 , 1 /κ ] that of a reference SDE such that the marginals at each time have distribution ˆ p t (11) . If q 0 , U-ALD = ˆ p 0 , the KL divergence between the path measures is given by

<!-- formula-not-decoded -->

The action A ( µ ) is bounded when the target π has bounded second order moment and under mild conditions on the schedule function, see Cordero-Encinar et al. [2025, Lemmas 3.3 and 4.2]. Full details and a complete proof are provided in App. C.3.

## 5 Numerical experiments

We now evaluate the performance of both proposed underdamped samplers, based on the annealed Langevin dynamics (Section 3.1) and the controlled diffusions formulations (Section 3.2) across a range of benchmark distributions. Full details of each benchmark are provided in App. D.

- Mixture of Gaussians (MoG) and mixture of Student's t (MoS) distributions .
- Rings: Two-dimensional distribution supported on a circular manifold.
- Funnel: 10-dimensional distribution defined by π ( x ) ∝ N ( x 1 ; 0 , η 2 ) ∏ d i =2 N ( x i ; 0 , e x 1 ) for x = ( x i ) 10 i =1 ∈ R 10 with η = 3 [Neal, 2003].
- Double well potential (DW): d -dimensional distribution given by π ∝ exp( -∑ m i =1 ( x 2 i -δ ) 2 -∑ d i = m +1 x 2 i ) with m ∈ N and δ ∈ (0 , ∞ ) . Ground truth samples are obtained using rejection sampling with a Gaussian mixture proposal distribution [Midgley et al., 2023].
- Examples from Bayesian statistics: Posterior distributions arising from Bayesian logistic regression tasks on the Ionosphere (dimension 35) and Sonar (dimension 61) datasets.
- Statistical physics model: Sampling metastable states of the stochastic Allen-Cahn equation ϕ 4 model (dimension 100) [Albergo et al., 2019, Gabrié et al., 2022]. At the chosen temperature, the distribution has two well distinct modes with relative weight controlled by a parameter h . Following Grenioux et al. [2024], we estimate the relative weight between the two modes.

While we focus here on the underdamped versions of our algorithms, we have also evaluated their overdamped counterparts. These approximately require an order of magnitude more steps to achieve comparable performance. Detailed results and comparisons are presented in App. E.1. Besides, we have set the base distribution, denoted as ν in our algorithms, to be a standard Gaussian. In the case of the mixture of Student's t benchmark distributions, we compare the performance of using either a standard Gaussian or a Student's t base distribution. For the latter, we further examine the impact of using a modified Itô diffusion for the fast dynamics (7), as opposed to a standard Langevin diffusion.

We compare our approach against a representative selection of related sampling methods: Sequential Monte Carlo (SMC) [Del Moral et al., 2006, Doucet et al., 2001], Annealed Importance Sampling (AIS) [Neal, 2001], Underdamped Langevin Monte Carlo (ULMC) [Cheng et al., 2018, Neal et al., 2011], Parallel Tempering (PT) [Geyer and Thompson, 1995, Swendsen and Wang, 1986], Diffusive Gibbs Sampling (DiGS) [Chen et al., 2024], Reverse Diffusion Monte Carlo (RDMC) [Huang et al., 2024] and Stochastic Localisation via Iterative Posterior Sampling (SLIPS) [Grenioux et al., 2024]. We ensure a fair comparison by using the same number of energy evaluations and the same initialisation based on a standard Gaussian distribution for all baselines. For completeness, we also report the performance of SLIPS with optimal initialisation (App. D)-which assumes knowledge of the distribution's scalar variance, unavailable in practice. For the ϕ 4 model, we only compare our results against a Laplace approximation and SLIPS algorithm, as the other methods provide degenerate samples. Since our approach is learning-free, we exclude comparisons with neural-based samplers as it would be difficult to make comparisons at equal computational budget.

Our methods consistently achieve strong performance across all benchmarks, outperforming all baselines. They also exhibit low variance in results (see Tables 1 and 2). Among our samplers, the one based on controlled diffusions, MULTCDIFF, generally achieves slightly better results than the annealed Langevin-based sampler, MULTALMC. Additionally, Figure 1a illustrates the estimation of the relative weight between the two modes for the ϕ 4 model, where our algorithms demonstrate

Table 1: Metrics for different benchmarks averaged across 30 seeds. The metric for the mixture of Gaussian (MoG) and Rings is the entropy regularised Wasserstein-2 distance (with regularisation parameter 0.05) and the metric for the Funnel is the sliced Kolmogorov-Smirnov distance.

| Algorithm   | 8-MoG ( ↓ ) ( d = 2)   | 40-MoG ( ↓ ) ( d = 2)   | 40-MoG ( ↓ ) ( d = 50)   | Rings ( ↓ ) ( d = 2)   | Funnel ( ↓ ) ( d = 10)   |
|-------------|------------------------|-------------------------|--------------------------|------------------------|--------------------------|
| SMC         | 5 . 26 ± 0 . 19        | 4 . 79 ± 0 . 16         | 52 . 17 ± 1 . 32         | 0 . 29 ± 0 . 05        | 0 . 034 ± 0 . 005        |
| AIS         | 5 . 53 ± 0 . 23        | 5 . 01 ± 0 . 20         | 65 . 83 ± 1 . 66         | 0 . 20 ± 0 . 03        | 0 . 040 ± 0 . 006        |
| ULMC        | 6 . 90 ± 0 . 28        | 8 . 17 ± 0 . 72         | 109 . 26 ± 2 . 90        | 0 . 35 ± 0 . 06        | 0 . 123 ± 0 . 011        |
| PT          | 3 . 91 ± 0 . 10        | 0 . 92 ± 0 . 08         | 21 . 04 ± 0 . 55         | 0 . 20 ± 0 . 04        | 0 . 033 ± 0 . 004        |
| DiGS        | 1 . 22 ± 0 . 09        | 0 . 95 ± 0 . 03         | 21 . 87 ± 0 . 97         | 0 . 20 ± 0 . 02        | 0 . 037 ± 0 . 005        |
| RDMC        | 1 . 01 ± 0 . 21        | 0 . 98 ± 0 . 07         | 36 . 10 ± 0 . 62         | 0 . 33 ± 0 . 01        | 0 . 080 ± 0 . 007        |
| SLIPS       | 1 . 15 ± 0 . 12        | 1 . 04 ± 0 . 06         | 23 . 71 ± 0 . 65         | 0 . 26 ± 0 . 01        | 0 . 039 ± 0 . 006        |
| MULTALMC    | 0 . 65 ± 0 . 07        | 0 . 91 ± 0 . 04         | 20 . 58 ± 0 . 71         | 0 . 18 ± 0 . 02        | 0 . 032 ± 0 . 005        |
| MULTCDIFF   | 0 . 62 ± 0 . 10        | 0 . 93 ± 0 . 06         | 20 . 13 ± 0 . 59         | 0 . 19 ± 0 . 03        | 0 . 031 ± 0 . 005        |

Table 2: Performance metrics for different sampling benchmarks averaged across 30 seeds. The metric for the double well potential (DW) is the entropy regularised Wasserstein-2 distance (with regularisation parameter 0.05) and the metric for the Bayesian logistic regression on Ionosphere and Sonar datasets is the average predictive posterior log-likelihood on a test dataset.

| Algorithm   | 5-dim DW( ↓ ) ( m = 5 , δ = 4)   | 10-dim DW( ↓ ) ( m = 5 , δ = 3)   | 50-dim DW( ↓ ) ( m = 5 , δ = 2)   | Ionosphere ( ↑ ) ( d = 35)   | Sonar ( ↑ ) ( d = 61)   |
|-------------|----------------------------------|-----------------------------------|-----------------------------------|------------------------------|-------------------------|
| SMC         | 4 . 06 ± 0 . 13                  | 11 . 86 ± 0 . 70                  | 28 . 92 ± 0 . 99                  | - 87 . 74 ± 0 . 10           | - 111 . 00 ± 0 . 11     |
| AIS         | 4 . 11 ± 0 . 17                  | 12 . 30 ± 0 . 61                  | 39 . 20 ± 1 . 07                  | - 88 . 11 ± 0 . 13           | - 111 . 15 ± 0 . 17     |
| ULMC        | 7 . 87 ± 0 . 51                  | 25 . 21 ± 1 . 00                  | 62 . 74 ± 1 . 52                  | - 116 . 37 ± 1 . 85          | - 173 . 61 ± 2 . 48     |
| PT          | 2 . 39 ± 0 . 20                  | 4 . 97 ± 0 . 43                   | 16 . 13 ± 0 . 89                  | - 87 . 91 ± 0 . 09           | - 112 . 99 ± 0 . 10     |
| DiGS        | 2 . 51 ± 0 . 18                  | 5 . 02 ± 0 . 42                   | 17 . 04 ± 0 . 95                  | - 88 . 02 ± 0 . 12           | - 110 . 53 ± 0 . 25     |
| RDMC        | 3 . 68 ± 0 . 32                  | 9 . 57 ± 0 . 75                   | 25 . 21 ± 0 . 73                  | - 108 . 44 ± 1 . 13          | - 130 . 20 ± 1 . 36     |
| SLIPS       | 2 . 46 ± 0 . 21                  | 5 . 09 ± 0 . 39                   | 18 . 15 ± 0 . 50                  | - 87 . 34 ± 0 . 11           | - 110 . 14 ± 0 . 10     |
| MULTALMC    | 2 . 08 ± 0 . 16                  | 4 . 48 ± 0 . 40                   | 14 . 03 ± 0 . 63                  | - 86 . 85 ± 0 . 09           | - 109 . 05 ± 0 . 13     |
| MULTCDIFF   | 1 . 95 ± 0 . 24                  | 4 . 23 ± 0 . 37                   | 13 . 98 ± 0 . 56                  | - 86 . 33 ± 0 . 10           | - 109 . 60 ± 0 . 21     |

<!-- image -->

h

- (a) Estimated mode weight ratio of ϕ 4 for different h .
- (b) Regularised W 2 for MoS in different dimensions.

<!-- image -->

Figure 1: Results for sampling benchmarks.

competitive performance. For the mixture of Student's t-distributions, Figure 1b shows that using the modified Itô diffusion for the fast dynamics, combined with a Student's t base distribution ν (red boxplots) outperforms both our other proposed algorithms and the baselines. Further experimental details and additional results are provided in Appendices D and E.

## 6 Discussion

In this work, we introduce a framework based on multiscale diffusions for sampling from an unnormalised target density. In particular, we propose two samplers, MULTALMC and MULTCDIFF depending on the dynamics used for the slow process: annealed Langevin dynamics or controlled diffusions, respectively. We establish theoretical guarantees for the convergence of these sampling algorithms and illustrate their performance on a range of high-dimensional benchmark distributions.

Our approach has certain limitations. Notably, the theoretical guarantees rely on stringent assumptions, which we aim to relax in future work. Additionally, the current method requires manual tuning of hyperparameters such as step size δ , scale separation parameters ε and friction coefficient Γ . Automating this tuning process remains an important direction for future research. Further research could explore extending the controlled diffusion framework to heavy-tailed target distributions, which pose additional challenges or developing more efficient numerical schemes for implementing the proposed multiscale samplers.

## Acknowledgements

PCE gratefully acknowledges support from the EPSRC through the Centre for Doctoral Training in Modern Statistics and Statistical Machine Learning (StatML), grant no. EP/S023151/1. SR's work was partially funded by the Deutsche Forschungsgemeinschaft (DFG) under Project-ID 318763901 SFB1294.

## References

- Assyr Abdulle, Ibrahim Almuslimani, and Gilles Vilmart. Optimal Explicit Stabilized Integrator of Weak Order 1 for Stiff and Ergodic Stochastic Differential Equations. SIAM/ASA Journal on Uncertainty Quantification , 6(2):937-964, 2018.
- ODeniz Akyildiz, Michela Ottobre, and Iain Souttar. A multiscale perspective on maximum marginal likelihood estimation. arXiv preprint arXiv:2406.04187 , 2024.
- M. S. Albergo, G. Kanwar, and P. E. Shanahan. Flow-based generative models for Markov chain Monte Carlo in lattice field theory. Phys. Rev. D , 100:034515, Aug 2019.
- Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 , 2023.
- Luigi Ambrosio and Bernd Kirchheim. Rectifiable sets in metric and Banach spaces. Mathematische Annalen , 318(3):527-555, 2000.
- Luigi Ambrosio, Nicola Gigli, and Giuseppe Savaré. Gradient Flows: In Metric Spaces and in the Space of Probability Measures . Lectures in Mathematics. ETH Zürich. Birkhäuser Basel, 2008.
- Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications , 12(3):313-326, 1982.
- Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d -linear convergence bounds for diffusion models via stochastic localization. In The Twelfth International Conference on Learning Representations , 2024.
- Richard Bertram and Jonathan E. Rubin. Multi-timescale systems and fast-slow analysis. Mathematical Biosciences , 287:105-121, 2017.
- Denis Blessing, Xiaogang Jia, Johannes Esslinger, Francisco Vargas, and Gerhard Neumann. Beyond ELBOs: A large-scale evaluation of variational methods for sampling. In Forty-first International Conference on Machine Learning , 2024.
- Denis Blessing, Julius Berner, Lorenz Richter, and Gerhard Neumann. Underdamped Diffusion Bridges with Applications to Sampling. In The Thirteenth International Conference on Learning Representations , 2025.
- Nawaf Bou-Rabee and Andreas Eberle. Mixing time guarantees for unadjusted Hamiltonian Monte Carlo. Bernoulli , 29(1):75 - 104, 2023.
- Omar Chehab and Anna Korba. A practical diffusion path for sampling. In ICML 2024 Workshop on Structured Probabilistic Inference &amp; Generative Modeling , 2024.
- Junhua Chen, Lorenz Richter, Julius Berner, Denis Blessing, Gerhard Neumann, and Anima Anandkumar. Sequential Controlled Langevin Diffusions. In The Thirteenth International Conference on Learning Representations , 2025.
- Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In The Eleventh International Conference on Learning Representations , 2023.
- Wenlin Chen, Mingtian Zhang, Brooks Paige, José Miguel Hernández-Lobato, and David Barber. Diffusive Gibbs sampling. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 7731-7747. PMLR, 21-27 Jul 2024.

- Yongxin Chen, Sinho Chewi, Adil Salim, and Andre Wibisono. Improved analysis for a proximal algorithm for sampling. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pages 2984-3014. PMLR, 02-05 Jul 2022.
- Xiang Cheng, Niladri S. Chatterji, Peter L. Bartlett, and Michael I. Jordan. Underdamped Langevin MCMC: A non-asymptotic analysis. In Sébastien Bubeck, Vianney Perchet, and Philippe Rigollet, editors, Proceedings of the 31st Conference On Learning Theory , volume 75 of Proceedings of Machine Learning Research , pages 300-323. PMLR, 06-09 Jul 2018.
- Sinho Chewi, Chen Lu, Kwangjun Ahn, Xiang Cheng, Thibaut Le Gouic, and Philippe Rigollet. Optimal dimension dependence of the Metropolis-Adjusted Langevin Algorithm. In Mikhail Belkin and Samory Kpotufe, editors, Proceedings of Thirty Fourth Conference on Learning Theory , volume 134 of Proceedings of Machine Learning Research , pages 1260-1300. PMLR, 15-19 Aug 2021.
- Paula Cordero-Encinar, O. Deniz Akyildiz, and Andrew B. Duncan. Non-asymptotic Analysis of Diffusion Annealed Langevin Monte Carlo for Generative Modelling. arXiv preprint arXiv:2502.09306 , 2025.
- Marco Cuturi, Laetitia Meng-Papaxanthos, Yingtao Tian, Charlotte Bunne, Geoff Davis, and Olivier Teboul. Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein. arXiv preprint arXiv:2201.12324 , 2022.
- Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential Monte Carlo samplers. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 68(3):411-436, 2006.
- Tim Dockhorn, Arash Vahdat, and Karsten Kreis. GENIE: Higher-Order Denoising Diffusion Solvers. In Advances in Neural Information Processing Systems , 2022a.
- Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-Based Generative Modeling with CriticallyDamped Langevin Diffusion. In International Conference on Learning Representations (ICLR) , 2022b.
- Arnaud Doucet, Nando de Freitas, and Neil Gordon. Sequential Monte Carlo Methods in Practice . Springer New York, 2001.
- A. B. Duncan, N. Nüsken, and G. A. Pavliotis. Using Perturbed Underdamped Langevin Dynamics to Efficiently Sample from Probability Distributions. Journal of Statistical Physics , 169(6): 1098-1131, 2017.
- Alain Durmus and Éric Moulines. Nonasymptotic convergence analysis for the unadjusted Langevin Algorithm. The Annals of Applied Probability , 27(3):1551-1587, 2017.
- Andreas Eberle and Francis Lörler. Non-reversible lifts of reversible diffusion processes and relaxation times. Probability Theory and Related Fields , 2024.
- Marylou Gabrié, Grant M. Rotskoff, and Eric Vanden-Eijnden. Adaptive Monte Carlo augmented with normalizing flows. Proceedings of the National Academy of Sciences , 119(10):e2109420119, 2022.
- Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. Bayesian Data Analysis . Chapman and Hall/CRC., 3rd edition, 2013.
- Charles J. Geyer and Elizabeth A. Thompson. Annealing Markov Chain Monte Carlo with Applications to Ancestral Inference. Journal of the American Statistical Association , 90(431):909-920, 1995.
- Louis Grenioux, Alain Oliviero Durmus, Eric Moulines, and Marylou Gabrié. On sampling with approximate transport maps. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 11698-11733. PMLR, 23-29 Jul 2023.

- Louis Grenioux, Maxence Noble, Marylou Gabrié, and Alain Oliviero Durmus. Stochastic localization via iterative posterior sampling. In Forty-first International Conference on Machine Learning , 2024.
- Wei Guo, Molei Tao, and Yongxin Chen. Provable Benefit of Annealed Langevin Monte Carlo for Nonlog-concave Sampling. In The Thirteenth International Conference on Learning Representations , 2025.
- Emily Harvey, Vivien Kirk, Martin Wechselberger, and James Sneyd. Multiple Timescales, Mixed Mode Oscillations and Canards in Models of Intracellular Calcium Dynamics. Journal of Nonlinear Science , 21(5):639-683, 2011.
- Ye He, Tyler Farghly, Krishnakumar Balasubramanian, and Murat A. Erdogdu. Mean-Square Analysis of Discretized Itô Diffusions for Heavy-tailed Sampling. Journal of Machine Learning Research , 25(43):1-44, 2024a.
- Ye He, Kevin Rojas, and Molei Tao. Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024b.
- Marlis Hochbruck and Alexander Ostermann. Exponential integrators. Acta Numerica , 19:209-286, 2010.
- Peter Holderrieth, Yilun Xu, and Tommi Jaakkola. Hamiltonian Score Matching and Generative Flows. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- Xunpeng Huang, Hanze Dong, Yifan Hao, Yian Ma, and Tong Zhang. Reverse Diffusion Monte Carlo. In The Twelfth International Conference on Learning Representations , 2024.
- Aapo Hyvärinen. Estimation of Non-Normalized Statistical Models by Score Matching. Journal of Machine Learning Research , 6(24):695-709, 2005.
- Ioannis Karatzas and Steven E. Shreve. Brownian Motion and Stochastic Calculus . Springer New York, NY, 1991.
- Yin Tat Lee, Ruoqi Shen, and Kevin Tian. Lower Bounds on Metropolized Sampling Methods for Well-Conditioned Distributions. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , 2021a.
- Yin Tat Lee, Ruoqi Shen, and Kevin Tian. Structured Logconcave Sampling with a Restricted Gaussian Oracle. In Mikhail Belkin and Samory Kpotufe, editors, Proceedings of Thirty Fourth Conference on Learning Theory , volume 134 of Proceedings of Machine Learning Research , pages 2993-3050. PMLR, 15-19 Aug 2021b.
- Benedict Leimkuhler and Charles Matthews. Rational construction of stochastic numerical methods for molecular sampling. Applied Mathematics Research eXpress , 2013(1):34-56, 2013.
- Benedict Leimkuhler, Daniel Paulin, and Peter A. Whalley. Contraction rate estimates of stochastic gradient kinetic Langevin integrators. ESAIM: M2AN , 58(6):2255-2286, 2024.
- Tony Lelièvre, Mathias Rousset, and Gabriel Stoltz. Free Energy Computations . Imperial College Press, 2010.
- Stefano Lisini. Characterization of absolutely continuous curves in Wasserstein spaces. Calculus of Variations and Partial Differential Equations , 28(1):85-120, 2007.
- Jun S. Liu. Monte Carlo Strategies in Scientific Computing . Springer New York, NY, 2008.
- Wei Liu, Michael Röckner, Xiaobin Sun, and Yingchao Xie. Averaging principle for slow-fast stochastic differential equations with time dependent locally Lipschitz coefficients. Journal of Differential Equations , 268(6):2910-2948, 2020.
- Martin W. McCall. Classical mechanics : from Newton to Einstein : a modern introduction . Wiley, Hoboken, N.J., 2010.

- Laurence Illing Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, and José Miguel Hernández-Lobato. Flow annealed importance sampling bootstrap. In The Eleventh International Conference on Learning Representations , 2023.
- Pierre Monmarché. High-dimensional MCMC with a standard splitting scheme for the underdamped Langevin diffusion. Electronic Journal of Statistics , 2020.
- Alireza Mousavi-Hosseini, Tyler K. Farghly, Ye He, Krishna Balasubramanian, and Murat A. Erdogdu. Towards a Complete Analysis of Langevin Monte Carlo: Beyond Poincaré Inequality. In Gergely Neu and Lorenzo Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory , volume 195 of Proceedings of Machine Learning Research , pages 1-35. PMLR, 12-15 Jul 2023.
- Radford M. Neal. Annealed importance sampling. Statistics and Computing , 11(2):125-139, 2001.
- Radford M. Neal. Slice sampling. The Annals of Statistics , 31(3):705 - 767, 2003.
- Radford M Neal et al. MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo , 2(11):2, 2011.
- Maxence Noble, Louis Grenioux, Marylou Gabrié, and Alain Oliviero Durmus. Learned referencebased diffusion sampler for multi-modal distributions. In The Thirteenth International Conference on Learning Representations , 2025.
- Kushagra Pandey, Jaideep Pathak, Yilun Xu, Stephan Mandt, Michael Pritchard, Arash Vahdat, and Morteza Mardani. Heavy-Tailed Diffusion Models. In The Thirteenth International Conference on Learning Representations , 2025.
- Grigorios A Pavliotis and Andrew Stuart. Multiscale methods: averaging and homogenization , volume 53. Springer Science &amp; Business Media, 2008.
- Gabriel Peyré and Marco Cuturi. Computational Optimal Transport: With Applications to Data Science. Foundations and Trends® in Machine Learning , 11(5-6):355-607, 2019.
- Angus Phillips, Hai-Dang Dau, Michael John Hutchinson, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. Particle Denoising Diffusion Sampler. In Forty-first International Conference on Machine Learning , 2024.
- Lorenz Richter and Julius Berner. Improved sampling via learned diffusions. In The Twelfth International Conference on Learning Representations , 2024.
- Saeed Saremi, Ji Won Park, and Francis Bach. Chain of Log-Concave Markov Chains. In The Twelfth International Conference on Learning Representations , 2024.
- Katharina Schuh and Iain Souttar. Conditions for uniform in time convergence: applications to averaging, numerical discretisations and mean-field systems. arXiv preprint arXiv:2412.05239 , 2024.
- Dario Shariatian, Umut Simsekli, and Alain Oliviero Durmus. Denoising Lévy Probabilistic Models. In The Thirteenth International Conference on Learning Representations , 2025.
- Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems , 32, 2019.
- Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 12438-12448. Curran Associates, Inc., 2020.
- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations , 2021.
- Gilbert Strang. On the Construction and Comparison of Difference Schemes. SIAM Journal on Numerical Analysis , 5(3):506-517, 1968.

- Robert H. Swendsen and Jian-Sheng Wang. Replica Monte Carlo Simulation of Spin-Glasses. Phys. Rev. Lett. , 57:2607-2609, Nov 1986.
- H. F. Trotter. On the Product of Semi-Groups of Operators. Proceedings of the American Mathematical Society , 10(4):545-551, 1959.
- Mark E Tuckerman. Statistical mechanics: Theory and Molecular Simulation . Oxford university press, 2010.
- Francisco Vargas, Shreyas Padhy, Denis Blessing, and Nikolas Nüsken. Transport meets Variational Inference: Controlled Monte Carlo Diffusions. In The Twelfth International Conference on Learning Representations , 2024.
- Fengyu Wang. Functional inequalities Markov semigroups and spectral theory . Elsevier, 2006.
- E Weinan, Di Liu, and Eric Vanden-Eijnden. Analysis of multiscale methods for stochastic differential equations. Communications on Pure and Applied Mathematics , 58(11):1544-1585, 2005.
- Keru Wu, Scott Schmidler, and Yuansi Chen. Minimax Mixing Time of the Metropolis-Adjusted Langevin Algorithm for Log-Concave Sampling. Journal of Machine Learning Research , 23(270): 1-63, 2022.
- Qinsheng Zhang and Yongxin Chen. Path integral sampler: A stochastic control approach for sampling. In International Conference on Learning Representations , 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract summarises the main contributions. In addition, we describe these contributions in more detail at the end of Section 1. Each of our claims is clearly linked to the relevant sections of the paper for further detail.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We outline the limitations of our methodology in Section 6 and provide a more detailed discussion in Appendix F.

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

Justification: We provide full assumptions for all the theoretical results. Furthermore, the complete proofs of our results are included in Appendix C.

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

Justification: Detailed information necessary to reproduce the main experimental results-including the methodology and hyperparameter choices for both our algorithms and the baselines-is provided in Appendices B and D.

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

Justification: The code is available at the following anonymised link https://anonymous. 4open.science/r/sampling\_by\_averaging-41F2 .

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

Justification: The experimental setup for our algorithms is detailed in the main paper, specifically in Sections 3 and 5. Additional implementation details, including those for baseline methods, are provided in Section D of the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All numerical experiments report error estimates in the form of standard deviations computed over multiple runs with different random seeds.

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

Justification: The compute resources and runtimes for the experiments are provided in Appendix D.4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have read the code of conduct and ensure that the paper complies with it. Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We address broader impacts in Appendix G.

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

Justification: This paper proposes a general purpose algorithm for sampling from a probability distribution, which does not have potential for direct misuse or dual-use. We do not use any scraped datasets in the numerical experiments which could lead to potential misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly acknowledge and cite all assets and resources used in the paper.

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

Justification: The paper introduces a sampling method for unnormalised probability distributions and does not release new assets.

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

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Preliminaries

Multiscale methods and stochastic averaging. Before presenting a formal definition, we first provide the intuition behind multiscale methods, following Schuh and Souttar [2024]. In many areas of science and engineering [Bertram and Rubin, 2017, Weinan et al., 2005], one often encounters systems that are difficult to analyse directly. However, there may exist a related system that is more tractable, though it does not produce exactly the same behaviour. In such cases, the simpler system can be viewed as an approximation of the more complex one. In our setting, the simpler system is a slow-fast system, while the complex system corresponds to the averaged dynamics. We develop these concepts in the following.

To conduct a theoretical analysis of the convergence of the proposed sampling algorithms, we examine the general stochastic slow-fast system studied in Liu et al. [2020]

<!-- formula-not-decoded -->

where ε is a small positive parameter describing the ratio of time scales between the slow component X ε t and the fast component Y ε t , and B t and ˜ B t are mutually independent standard Brownian motions on a complete probability space (Ω , F , P ) and {F t , t ≥ 0 } is the natural filtration generated by B t and ˜ B t . Let ¯ X t denote the solution of the following averaged equation

<!-- formula-not-decoded -->

where ¯ b ( t, x ) = ∫ R m b ( t, x, y ) ρ t,x (d y ) and ρ t,x denotes the unique invariant measure for the transition semigroup of the corresponding frozen process, which can be informally defined as the fast process with ε = 1 and fixed slow dynamics X ε t = x ,

<!-- formula-not-decoded -->

where ˜ B ′ s is a Brownian motion on another complete probability space.

We begin by presenting the assumptions underlying the analysis in Liu et al. [2020] and stating their main results, which will form the basis of our study in Appendix C.1.

The assumptions on the coefficients of the stochastic slow-fast system (12) are the following.

A3 (Coefficients of the slow process) . (i) There exists θ 1 ≥ 0 such that for any t, R ≥ 0 , x i ∈ R n , y ∈ R m with ∥ x i ∥ ≤ R ,

<!-- formula-not-decoded -->

where K t ( R ) is an R + -valued F t -adapted process satisfying for all R,T,p ∈ [0 , ∞ ) ,

<!-- formula-not-decoded -->

Furthermore, there exists R 0 &gt; 0 , such that for any R ≥ R 0 , T ≥ 0 ,

<!-- formula-not-decoded -->

(ii) There exist constants θ 2 , θ 3 ≥ 1 and γ 1 ∈ (0 , 1] such that for any x ∈ R n , y, y 1 , y 2 ∈ R m and T &gt; 0 with t, s ∈ [0 , T ] ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where C T &gt; 0 and Z t is some random variable satisfying E Z 2 T &lt; ∞ .

(iii) There exist λ 1 ≥ 0 , C &gt; 0 , θ 4 ≥ 2 and θ 5 , θ 6 ≥ 1 such that for any t &gt; 0 , x ∈ R n , y ∈ R m ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

A4 (Coefficients of the fast process) . (i) There exists β ≥ 0 such that for any t ≥ 0 , x ∈ R n , y 1 , y 2 ∈ R m ,

<!-- formula-not-decoded -->

(ii) For any T &gt; 0 , there exists γ 2 ∈ (0 , 1] , C T &gt; 0 , α i ≥ 1 , i = 1 , 2 , 3 , 4 such that for any t, s ∈ [0 , T ] and x i ∈ R n , y i ∈ R m , i = 1 , 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(iii) For some fixed k ≥ 2 and any T &gt; 0 , there exist C T,k , β k &gt; 0 such that for any t ∈ [0 , T ] , x ∈ R n , y ∈ R m ,

<!-- formula-not-decoded -->

where λ 2 = 0 if λ 1 = 0 , and λ 2 &gt; 0 otherwise.

Note that assumption A 4 on the coefficients of the fast process guarantees that the frozen process has a unique stationary measure and that the fast process converges to it exponentially fast. The following theorems establish the existence and uniqueness of solutions to the system in (12), as well as its convergence to the corresponding averaged dynamics.

Theorem A.1 (Theorem 2.2 [Liu et al., 2020]) . If assumptions A 3 and A 4 hold with λ 1 &gt; 0 , let ε 0 = λ 2 λ 1 . Then for any ε ∈ (0 , ε 0 ) and any given initial values x ∈ R n , y ∈ R m , there exists a unique solution { ( X ε t , Y ε t ) , t ≥ 0 } to the system (12) and for all T &gt; 0 , we have ( X ε t , Y ε t ) ∈ C ([0 , T ]; R n ) × C ([0 , T ]; R m ) , P -almost surely. Furthermore, for all t ∈ [0 , T ] , the solution ( X ε t , Y ε t ) is given by

<!-- formula-not-decoded -->

Theorem A.2 (Theorem 2.3 [Liu et al., 2020]) . If assumptions A 3 and A 4 hold with λ 1 &gt; 0 and k &gt; ˜ θ 2 where ˜ θ 2 = max { 4 θ 1 , 2 θ 2 +2 , 2 θ 6 , 4 α 2 , θ 4 θ 5 , 2 α 1 θ 4 } . Then, for any 0 &lt; p &lt; 2 k θ 4 we have

<!-- formula-not-decoded -->

Optimal transport. Let v = ( v t : R d → R d ) be a vector field and µ = ( µ t ) t ∈ [ a,b ] be a curve of probability measures on R d with finite second-order moments. µ is generated by the vector field v if the continuity equation

<!-- formula-not-decoded -->

holds for all t ∈ [ a, b ] . The metric derivative of µ at t ∈ [ a, b ] is then defined as

<!-- formula-not-decoded -->

If | ˙ µ | t exists and is finite for all t ∈ [ a, b ] , we say that µ is an absolutely continuous curve of probability measures. Ambrosio and Kirchheim [2000] establish weak conditions under which a curve of probability measures with finite second-order moments is absolutely continuous.

By Ambrosio et al. [2008, Theorem 8.3.1] we have that among all velocity fields v t which produce the same flow µ , there is a unique optimal one with smallest L p ( µ t ; X ) -norm. This is summarised in the following lemma.

Lemma A.3 (Lemma 2 from Guo et al. [2025]) . For an absolutely continuous curve of probability measures µ = ( µ t ) t ∈ [ a,b ] , any vector field ( v t ) t ∈ [ a,b ] that generates µ satisfies | ˙ µ | t ≤ ∥ v t ∥ L 2 ( µ t ) for almost every t ∈ [ a, b ] . Moreover, there exists a unique vector field v ⋆ t generating µ such that | ˙ µ | t = ∥ v ⋆ t ∥ L 2 ( µ t ) almost everywhere.

We also introduce the action of the absolutely continuous curve ( µ t ) t ∈ [ a,b ] since it will play a key role in our convergence results. In particular, we define the action A ( µ ) as

<!-- formula-not-decoded -->

## Girsanov's theorem. Consider the SDE

<!-- formula-not-decoded -->

for t ∈ [0 , T ] , where ( B t ) t ∈ [0 ,T ] is a standard Brownian motion in R d . Denote by P X the path measure of the solution X = ( X t ) t ∈ [0 ,T ] of the SDE, which characterises the distribution of X over the sample space Ω .

The KL divergence between two path measures can be characterised as a consequence of Girsanov's theorem [Karatzas and Shreve, 1991]. In particular, the following result will be central in our analysis. Lemma A.4. Consider the following two SDEs defined on a common probability space (Ω , F , P )

<!-- formula-not-decoded -->

with the same initial conditions X 0 , Y 0 ∼ µ 0 . Denote by P X and P Y the path measures of the processes X and Y , respectively. It follows that

<!-- formula-not-decoded -->

## B Sampling using multiscale dynamics

In this section, we provide a detailed description of the implementation of our proposed algorithms including the numerical discretisation schemes used.

## B.1 MULTALMC: Multiscale Annealed Langevin Monte Carlo

Here, we present further details of the overdamped and underdamped versions of the sampler.

## B.1.1 Overdamped system

In the overdamped setting, we propose a new sampling strategy based on the following stochastic slow-fast system

<!-- formula-not-decoded -->

As discussed in the main text, the conditional distribution ρ t,x converges to a Dirac delta centred at 0 when λ κt = 0 , and to a Dirac at x when λ κt = 1 . These degenerate limits can cause numerical instabilities at the endpoints of the diffusion schedule. To mitigate this, we define 0 &lt; λ δ ≪ 1 and consider the modified dynamics

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that if the slow-fast system converges to its corresponding averaged dynamics, then running the modified multiscale system up to time t δ , where λ κt δ = 1 -λ δ , we can approximate the distribution ˆ µ t δ that is close to the target π , provided λ δ is sufficiently small. For t ≥ t δ , the dynamics in (13) reduce to standard overdamped Langevin dynamics targetting π , initialised at ˆ µ t δ . This warm start is known to significantly improve convergence rates to the target distribution [Chewi et al., 2021, Lee et al., 2021a, Wu et al., 2022].

We now analyse different numerical schemes for implementing the sampler in practice. As we have just mentioned, when λ κt ≥ 1 -λ δ , the dynamics in (13) reduce to standard overdamped Langevin dynamics, which can be discretised using any preferred numerical method. Therefore, we focus on the regime 0 ≤ λ κt &lt; 1 -λ δ .

For general base distributions ν , we propose to use the following numerical scheme that combines Euler-Maruyama for the slow dynamics with the SROCK method [Abdulle et al., 2018] for the fast dynamics. Before presenting the complete algorithm in Algorithm 2, we introduce the necessary notation and coefficients for the SROCK update.

Denote by T s the Chebychev polynomials of first kind and define the coefficients in the SROCK update as follows

<!-- formula-not-decoded -->

and for all i = 2 , . . . , s ,

<!-- formula-not-decoded -->

Additionally, when ν ∼ N (0 , I ) , we can further exploit the linear structure of the score function for a Gaussian distribution to design an efficient exponential integrator scheme. Below, we derive the corresponding update rules, distinguishing between the regimes 0 ≤ λ κt &lt; ˜ λ and ˜ λ ≤ λ κt ≤ 1 .

Regime 1: 0 ≤ λ κt &lt; ˜ λ . Define the modified schedule

<!-- formula-not-decoded -->

The slow-fast system (13) simplifies to

<!-- formula-not-decoded -->

The exponential integrator scheme is then expressed as

<!-- formula-not-decoded -->

where given a time discretisation 0 ≤ T 0 &lt; · · · &lt; T M of the corresponding time interval, we define t -= T l -1 when t ∈ [ T l -1 , T l ) . The explicit update rule is then

<!-- formula-not-decoded -->

where ξ l ∼ N (0 , I 2 d ) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 2 MULTALMC sampler: overdamped version

Require: Schedule function λ t , value for λ δ and ˜ λ , number of sampling steps L , time discretisation 0 = T 0 &lt; · · · &lt; T L = 1 /κ , step size h l = T l +1 -T l . Constants for SROCK step from (14), (15).

Initial samples X 0 ∼ N (0 , I ) , Y 0 ∼ N (0 , I ) . Define the schedule

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

EM for slow dynamics

<!-- formula-not-decoded -->

SROCK for fast dynamics

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

end for return X L

Regime 2: ˜ λ ≤ λ κt &lt; 1 -λ δ . Here, the dynamics become

<!-- formula-not-decoded -->

In this case, the exponential integrator scheme can be formulated as follows

<!-- formula-not-decoded -->

The corresponding update rule is

<!-- formula-not-decoded -->

where ξ l ∼ N (0 , I 2 d ) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.1.2 Underdamped system

Similarly in the overdamped case, to mitigate numerical instabilities arising from the degeneracy of ρ t,x when λ κt = 0 and λ κt = 1 , we consider the following modified dynamics.

<!-- formula-not-decoded -->

Note that when λ κt ≥ 1 -λ δ , the dynamics in (16) reduce to standard underdamped Langevin dynamics with a warm start, which can be discretised using any preferred numerical scheme. Therefore, we focus on the regime 0 ≤ λ κt &lt; 1 -λ δ . We propose to use a hybrid discretisation method that combines the OBABO splitting scheme [Monmarché, 2020] with an SROCK update [Abdulle et al., 2018] for the fast-dynamics. The algorithm is explictly defined in Algorithm 3, where √ Γ denotes the matrix square root of the friction coefficient, which is well defined since Γ is a symmetric positive definite matrix. That is, √ Γ is any matrix satisfying √ Γ √ Γ ⊺ = Γ .

It is important to mention that for implementation, we set the mass parameter to M = I and consider a time-dependent friction coefficient Γ t to control the degree of acceleration. Additionally, the same discretisation scheme applies to the heavy-tailed diffusion, with a modified fast process defined by Eq. (7).

## B.2 MULTCDIFF: Multiscale Controlled Diffusions

## B.2.1 Overdamped system

Diffusion models typically consider a stochastic process ( - → X t ) t ∈ [0 ,T ] constructed by initialising - → X 0 at the target distribution π and then evolving according to the OU SDE

<!-- formula-not-decoded -->

Under mild regularity conditions [Anderson, 1982], the dynamics of the reverse process ( ← -X t ) t ∈ [0 ,T ] are described by the following SDE

<!-- formula-not-decoded -->

where q t denotes the marginal distribution of the solution of the forward process (17), given by

<!-- formula-not-decoded -->

with ρ t ( x | y ) being the Gaussian transition density N ( e -t y, σ 2 t I ) , where σ 2 t = 1 -e -2 t . Using this, the score function ∇ log q T -t ( ← -X t ) can be expressed as

<!-- formula-not-decoded -->

Since the score term involves an intractable expectation, we approximate the averaged dynamics in (18) using a multiscale SDE system. The fast process of the system is modelled by an overdamped

## Algorithm 3 MULTALMC sampler: accelerated version

Require: Schedule function λ t , value for λ δ and ˜ λ , friction coefficient Γ (scalar), mass parameter M (scalar), number of sampling steps L , time discretisation 0 = T 0 &lt; · · · &lt; T L = 1 /κ , step size h l = T l +1 -T l . Constants for SROCK step from (14), (15).

Initial samples X 0 ∼ N (0 , I ) , V 0 ∼ N (0 , MI ) , Y 0 ∼ N (0 , I ) . Define the schedule

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Half-step for velocity component

<!-- formula-not-decoded -->

Full EM-step for position component

<!-- formula-not-decoded -->

Full SROCK-step for fast component

<!-- formula-not-decoded -->

Half-step for velocity component

<!-- formula-not-decoded -->

if 1 -λ δ ≤ λ κT l ≤ 1 then

Half-step for velocity component V ′ l = ( 1 -h l 2 Γ M -1 ) V l + √ h l Γ ξ (1) l V ′′ l = V ′ l -h l 2 ∇ V π ( X l )

Full EM-step for position

<!-- formula-not-decoded -->

component -1 ′′

Half-step for velocity component

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

end for

<!-- formula-not-decoded -->

Langevin SDE targetting the conditional distribution Y | ← -X t , which takes the form

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Putting everything together, we obtain the following multiscale system of SDEs

<!-- formula-not-decoded -->

This system of SDEs can be discretised using either an exponential integrator scheme or a numerical scheme that combines Euler-Maruyama for the slow dynamics with the SROCK method [Abdulle et al., 2018] for the fast component, similarly to the discretisation of MULTALMC in the overdamped case (see Appendix B.1.1).

## B.2.2 Underdamped system

When using an OU noising process, the forward underdamped diffusion has the following form

<!-- formula-not-decoded -->

which can be written compactly as

<!-- formula-not-decoded -->

where - → Z t = ( - → X t , - → V t ) and

<!-- formula-not-decoded -->

where, as in Appendix B.1.2, √ Γ denotes the matrix square root of the friction coefficient Γ , which is a symmetric and positive definite. The solution of this SDE is given by [Karatzas and Shreve, 1991]

<!-- formula-not-decoded -->

As discussed in the main text, there is a crucial balance between the mass M and friction coefficient Γ . We focus on the critically damping regime by setting Γ 2 = 4 M . This provides an ideal balance between the Hamiltonian and the Ohnstein-Uhlenbeck components of the dynamics, which leads to faster convergence without oscillations [McCall, 2010]. Under this setting, the matrix A can be written as

<!-- formula-not-decoded -->

Then, the matrix exponential e At has the following form

<!-- formula-not-decoded -->

On the other hand, the reverse SDE is given by

<!-- formula-not-decoded -->

where q t is the marginal distribution of the solution of the forward noising process which has the following expression

<!-- formula-not-decoded -->

with φ = N (0 , I ) and conditional distribution ρ t ( x, v | y, v 0 ) given by a Normal distribution with mean and covariance matrix

<!-- formula-not-decoded -->

which results into and covariance matrix

<!-- formula-not-decoded -->

We can rewrite Eq. (20) in terms of ρ t ( x, v | y ) as

<!-- formula-not-decoded -->

We note that the conditional distribution ρ t ( v | x, y ) remains Gaussian, with mean

<!-- formula-not-decoded -->

which is linear in y and covariance σ 2 t I , where

<!-- formula-not-decoded -->

This provides the following expression for the score

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Using the derivations above, we define a multiscale SDE system that enables sampling from the target distribution via the reverse SDE, without requiring prior estimation of the denoiser E Y | x,v [ Y ] . We consider a fast process governed by an overdamped Langevin SDE targeting the conditional distribution of Y given the current state ( ← -X t , ← -V t ) , which takes the form

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Combining altogether, we arrive at the following multiscale system of SDEs

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We note that v 0 in Eq. (20) can be integrated out analytically. Since both ρ t ( x, v | y, v 0 ) and φ ( v 0 ) are Gaussian, it follows that ρ t ( x, v | y ) is also Gaussian with mean

<!-- formula-not-decoded -->

Inspired by Dockhorn et al. [2022b], we propose a novel discretisation scheme which leverages symmetric splitting techniques [Leimkuhler and Matthews, 2013] justified by the symmetric Trotter splitting and the Baker-Campbell-Hausdorff formula [Strang, 1968, Trotter, 1959, Tuckerman, 2010]. The l -th iteration of the proposed numerical scheme consists of the following steps, with h l denoting the step size.

1. Evolve the following system of SDEs exactly for half-step h l / 2

<!-- formula-not-decoded -->

2. Evolve the velocity component (full step) under the following ODE using Euler's method

<!-- formula-not-decoded -->

3. Evolve the fast dynamics (full step) using SROCK method

<!-- formula-not-decoded -->

4. Evolve the system in Eq. (22) exactly for half-step h l / 2 .

We summarise the steps above in the form of a concise algorithm. To do so, we first derive the exact solution of the SDE system (22) used in Step 1. Define the matrices

<!-- formula-not-decoded -->

The matrix exponential e ˆ At takes the form

<!-- formula-not-decoded -->

The solution to the system (22) at time t , given an initial condition ( X 0 , V 0 ) , can be expressed as

<!-- formula-not-decoded -->

where the mean and covariance matrix are defined by

<!-- formula-not-decoded -->

The explicit expressions for the entries of the covariance matrix ˆ Σ t are

<!-- formula-not-decoded -->

The full algorithm is presented in Algorithm 4.

Challenges for heavy-tailed controlled diffusions. When using a heavy-tailed noising process given by the convolution with a Student's t distribution with tail index α , the forward underdamped diffusion takes the form

<!-- formula-not-decoded -->

The non-linear drift term in the velocity dynamics prevents us from obtaining analytic solutions via standard techniques for linear SDEs. As a result, we cannot directly reproduce the computations used in the case of an OU noising process. A detailed investigation of this regime is left for future work.

## C Theoretical guarantees

In this section, we provide detailed proofs of the theoretical results presented in the paper and discuss the additional challenges posed by heavy-tailed diffusions.

<!-- formula-not-decoded -->

## Algorithm 4 MULTCDIFF sampler

Require: Friction coefficient Γ (scalar), M = Γ 2 / 4 , number of sampling steps L , time discretisation 0 = T 0 &lt; · · · &lt; T L = T , step size h l = T l +1 -T l . Constants for SROCK step from (14), (15). Functions ˆ m t ( · , · ) and ˆ Σ t from (23).

Initial samples X 0 ∼ N (0 , I ) , V 0 ∼ N (0 , MI ) , Y 0 ∼ N (0 , I ) .

for l = 0 to L do

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.1 Convergence of the slow-fast system to the averaged dynamics

Based on the stochastic averaging results from [Liu et al., 2020] presented in Appendix A, we restate our convergence result, Theorem 4.1, and provide the proof.

Theorem C.1. [Theorem 4.1 restated] Let the base distribution ν ∼ N (0 , I ) . Suppose the target distribution π and the schedule function λ t satisfy assumptions A 1 and A 2, respectively. Then, for any ε ∈ (0 , ε 0 ) , where ε 0 is specified in the proof, and any given initial conditions, there exists unique solutions { ( X ε t , Y ε t ) , t ≥ 0 } , { ( X ε t , V ε t , Y ε t ) , t ≥ 0 } and { ( ← -X ε t , ← -V ε t , Y ε t ) , t ≥ 0 } to the slow-fast stochastic systems (4) , (6) and (10) , respectively. Furthermore, for any p &gt; 0 , it holds that

<!-- formula-not-decoded -->

where ¯ X t denotes the solution of the averaged system and X ε t is the corresponding slow component of the multiscale systems (4) , (6) and (10) .

Proof. If the coefficients of the proposed slow-fast systems (4), (6) and (10) satisfy assumptions A 3 and A 4, then the result follows directly from Theorems A.1 and A.2. Thus, we verify below that these assumptions hold in all cases.

It is important to mention that although the theorem specifically addresses the case where ν ∼ N (0 , I ) , assumption A 3 only requires that ∇ log ν is Lipschitz continuous with constant L ν and has a maximiser x ⋆ . Without loss of generality, we assume x ⋆ = 0 , i.e., ∇ log ν (0) = 0 . Therefore, we show that assumption A 3 holds for general base distributions ν satisfying these properties.

Besides, for the systems based on annealed Langevin diffusion (4) and (6), we instead consider their modified versions (13) and (16), respectively, as these reflect the dynamics used in practice during implementation. We observed in Appendix B.1 that these systems reduced to

- A multiscale system with time-independent coefficients for 0 ≤ λ κt &lt; λ δ .
- Amultiscale system with time-dependent coefficients for λ δ ≤ λ κt &lt; 1 -λ δ . Here, we distinguish between two different dynamics depending on whether λ κt &lt; ˜ λ or λ κt ≥ ˜ λ .
- Standard Langevin dynamics for 1 -λ δ ≤ λ κt ≤ 1 .

We analyse the first two cases, as standard Langevin dynamics is known to converge to its invariant distribution when initialised with a warm start [Chewi et al., 2021, Lee et al., 2021a, Wu et al., 2022]. See Appendix B.1 for further discussion.

We denote by m ⋆ a minimiser of the target potential V π .

Overdamped MULTALMC (4) or (13) The coefficients of the slow-fast system for λ κt ∈ [0 , 1 -λ δ ] have the following expressions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We consider two regimes based on the value of λ κt : Regime 1 λ κt ∈ [0 , ˜ λ ] and Regime 2 λ κt ∈ [ ˜ λ, 1 -λ δ ] . We analyse the convergence within each regime separately.

Substituting the expressions of the coefficients, it follows that assumption A 3 holds with θ 1 = 0 , θ 2 = θ 3 = θ 5 = θ 6 = 1 , θ 4 = 2 and Z T = ∥ m ⋆ ∥ √ ˜ λ/ 2 is a constant random variable in both regimes, as shown below.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the following, we can assume without loss of generality that s ≤ t

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which leads to the constant λ 1 in assumption A 3 being given by λ 1 = max { L ν 1 -˜ λ , L π ˜ λ } .

<!-- formula-not-decoded -->

On the other hand, to verify that assumption A 4 holds, we recall that for a standard Normal distribution ∇ log ν ( x ) = -x . Besides, note that we can express the drift term of the fast process, f ( t, x, y ) , in a compact form by rewriting (24) as

<!-- formula-not-decoded -->

where the schedule λ ′ κt is defined by

<!-- formula-not-decoded -->

Therefore, we adopt this compact form in the following analysis and omit explicit distinctions between regimes to simplify notation. Using this, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the potential V π is strongly convex with constant M π , then V π satisfies the dissipativity inequality ⟨∇ V π ( x ) , x ⟩ ≥ a π ∥ x ∥ 2 -b π . Therefore, we have that for any k ≥ 2

<!-- formula-not-decoded -->

where λ 2 is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof for the overdamped MULTALMC, where the constant ε 0 in the theorem statement is explicitly given by

<!-- formula-not-decoded -->

Underdamped MULTALMC (6) or (16) In this case, since we introduce an auxiliary velocity variable, the slow component consists of both position and velocity, ( x, v ) . Accordingly, the coefficients of the slow-fast system are given by the following expressions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the coefficients of the fast process are the same as in the overdamped setting, then assumption A 4 also holds in the underdamped case. To verify assumption A 3, we adopt the same two-regime framework defined in the overdamped case and analyse convergence within each regime separately.

2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the following, we can assume without loss of generality that s ≤ t

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes that assumption A 3 holds with θ 1 = 0 , θ 2 = θ 3 = θ 5 = θ 6 = 1 , θ 4 = 2 and Z T = ∥ m ⋆ ∥ √ ˜ λ/ 2 is a constant random variable in both regimes. Besides, the constant ε 0 in the theorem statement is given by

<!-- formula-not-decoded -->

Underdamped MULTCDIFF (10) or (21) From Appendix B.2, the coefficients of the slow-fast system are given by the following expressions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where m 1 ,T -t , m 3 ,T -t , ˜ Σ T -t are defined in Appendix B.2. Substituting these coefficients, it follows that assumption A 3 holds with θ 1 = 0 , θ 2 = θ 3 = θ 5 = θ 6 = 1 , θ 4 = 2 and Z T is a constant random variable, as shown below.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the expressions in Appendix B.2, which imply the existence of a constant ˜ γ 1 ∈ (0 , 1] such that

<!-- formula-not-decoded -->

2 ⟨ ( x, v ) , b ( t, ( x, v ) , y ) ⟩

<!-- formula-not-decoded -->

∥ b ( t, ( x, v ) , y ) ∥

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We finally check that assumption A 4 holds.

<!-- formula-not-decoded -->

where we have used that, for all t , the precision matrix ˜ Σ -1 T -t is positive definite and λ min ( ˜ Σ -1 T -t ) denotes its smallest eigenvalue.

<!-- formula-not-decoded -->

where the existence of ˜ γ 2 ∈ (0 , 1] follows from the expressions for m 1 ,T -t , m 3 ,T -t , ˜ Σ T -t provided in Appendix B.2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recalling that the strong convexity of V π implies a dissipativity inequality, we have that for any k ≥ 2

<!-- formula-not-decoded -->

which concludes the proof.

## C.2 Challenges of extension to heavy-tailed diffusions

When considering heavy-tailed diffusions, that is, when the base distribution ν is a Student's t distribution, the stochastic slow-fast systems (4) and (6) could still be used to derive sampling algorithms. However, Theorem 4.1 showing convergence to the averaged dynamics does not apply in this setting, as it relies on exponential ergodicity of the frozen process. In the case where the fast process follows overdamped Langevin dynamics targetting a heavy-tailed distribution, convergence is sub-exponential or polynomial rather than exponential [Wang, 2006, Chapter 4], and thus the frozen process fails to meet the required condition.

As discussed in Section 3.1, instead of using overdamped Langevin dynamics for the fast process, an alternative is to consider different diffusion processes that target the conditional distribution ρ t,x more efficiently. Motivated by the work of He et al. [2024a], we propose employing a natural Itô diffusion that arises in the context of the weighted Poincaré inequality, which has the following expression

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α denotes the tail index of the noising distribution ν . This results into the following slow-fast system in the underdamped setting

<!-- formula-not-decoded -->

However, although the modified Itô diffusion (25) offers improved convergence properties [He et al., 2024a], it does not guarantee exponential ergodicity of the frozen process under mild conditions. As a result, assumption A 4 is not satisfied in this setting. We leave the analysis of the convergence of (26) to the averaged dynamics (5) for future work.

## C.3 Bias of the annealed Langevin dynamics

We formally quantify the bias introduced by the underdamped averaged system (5) relative to the true diffusion path which is given by (ˆ p t ) t ∈ [0 , 1 /κ ] with

<!-- formula-not-decoded -->

Theorem C.2 (Theorem 4.2 restated) . Let Q U-ALD = ( q t, U-ALD ) t ∈ [0 , 1 /κ ] be the path measure of the diffusion annealed Langevin dynamics (5) , and Q = (ˆ p t ) t ∈ [0 , 1 /κ ] that of a reference SDE such that the marginals at each time have distribution ˆ p t defined in (27) . If q 0 , U-ALD = ˆ p 0 , the KL divergence between the path measures is given by

<!-- formula-not-decoded -->

Proof. Let Q be the path measure corresponding to the following reference SDE

<!-- formula-not-decoded -->

The vector field ˆ v = ((ˆ v 1 ,t , ˆ v 2 ,t )) t ∈ [0 , 1 /κ ] is designed such that ( ¯ Z t , ¯ U t ) ∼ ˆ p t for all t ∈ [0 , 1 /κ ] . Using the Fokker-Planck equation, we have that

<!-- formula-not-decoded -->

This implies that ˆ v t = (ˆ v 1 ,t , ˆ v 2 ,t ) satisfies the continuity equation and hence generates the curve of probability measures (ˆ p t ) t . Leveraging Lemma A.3, we choose ˆ v to be the one that minimises the L 2 (ˆ p t ) norm, resulting in ∥ ˆ v t ∥ L 2 (ˆ p t ) = ∣ ∣ ∣ ˙ ˆ p ∣ ∣ ∣ t being the metric derivative. Using the form of Girsanov's theorem given in Lemma A.4 we have

<!-- formula-not-decoded -->

where we have used that ∣ ∣ ∣ ˙ ˆ p ∣ ∣ ∣ t = κ | ˙ p | t and the change of variable formula.

To conclude, we note from (27) that the position x and the velocity v variable are independent, and that the marginal distribution of the velocity v remains constant along the diffusion path (ˆ p t ) t ∈ [0 , 1] .As a result, the metric derivative simplifies to

<!-- formula-not-decoded -->

Consequently, the action A ( p ) satisfies or

<!-- formula-not-decoded -->

where µ = ( µ t ) t ∈ [0 , 1] is the diffusion path defined in (1).

Finally note that from the data processing inequality, it follows that the KL divergence between the marginals at final time is bounded by KL ( ˆ p 1 /κ || q 1 /κ, U-DALD ) ≤ KL( Q || Q U-ALD ) .

By Cordero-Encinar et al. [2025, Lemmas 3.3 and 4.2], the action A ( µ ) is bounded when the base distribution ν is either a Gaussian or a Student's t distribution, provided that the target distribution π has finite second order moment and the annealing schedule λ t satisfies the following assumption.

A5. Let λ t : R + → [0 , 1] be non-decreasing in t and weakly differentiable, such that if ν ∼ N (0 , I ) there exists a constant C λ satisfying either of the following conditions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or if ν follows a Student's t distribution, then condition (28) holds for some constant C λ .

## D Experiment details

The code to reproduce our experiments is available at https://github.com/paulaoak/ sampling\_by\_averaging.git . All experiments were implemented using JAX.

## D.1 Target distributions

Mixture of Gaussians (MoG) and mixture of Student's t (MoS) distributions The 8 mixture of Gaussians distribution (8-MoG) consists of 8 equally weighted Gaussian distributions with mean m i = 10 × (1 + cos(2 πi/ 8) , 1 + sin(2 πi/ 8)) for i ∈ { 9 , . . . , 7 } and covariance 0 . 7 I 2 . We have shifted the distribution instead of considering the usual benchmark centred at 0 to make sampling more challenging. We note that some of the baselines get stuck in modes close to the initial standard Gaussian distribution, unlike our methods.

For the mixture of 40 Gaussians in dimensions 2 and 50 , the modes are equally weighted, with means sampled uniformly from a hypercube of side length 40 , and all the covariance matrices set to the identity matrix I .

The mixture of Student's t distributions consists of 10 standard Student's t distributions, each with 2 degrees of freedom ( t 2 ). Following [Blessing et al., 2024, Chen et al., 2025], the mean of each component is sampled uniformly from a hypercube with a side length of 10. We evaluate this benchmark in dimensions 2, 10 and 50.

Rings This distribution is defined by the inverse polar reparameterisation of a distribution p z which has itself a decomposition into two univariate marginals: p r and p θ . The radial component p r is a mixture of 4 Gaussian distributions N ( i +1 , 0 . 15 2 ) for i ∈ { 0 , . . . , 3 } describing the radial positions. The angular component p θ is a uniform distribution over the interval [0 , 2 π ] .

Funnel The density of this distribution is given by π ( x ) ∝ N ( x 1 ; 0 , η 2 ) ∏ d i =2 N ( x i ; 0 , e x 1 ) for x = ( x i ) 10 i =1 ∈ R 10 with η = 3 [Neal, 2003].

Double well potential (DW) The unnormalised density of the d -dimensional DW is given by π ∝ exp( -∑ m i =1 ( x 2 i -δ ) 2 -∑ d i = m +1 x 2 i ) with m ∈ N and a separation parameter δ ∈ (0 , ∞ ) . This distribution has 2 m modes, and larger values of δ make sampling more challenging due to higher energy barriers. Ground truth samples are obtained using rejection sampling with a Gaussian mixture proposal distribution [Midgley et al., 2023].

Bayesian logistic regression examples We consider binary classification problems on two benchmark datasets: Ionosphere (dimension 35) and Sonar (dimension 61). Given a training dataset D = { ( x j , y j ) } N j =1 where x j ∈ R d and y j , the posterior distribution over the model parameters ( w,b ) , w ∈ R d and b ∈ R , is defined by

<!-- formula-not-decoded -->

where σ denotes the sigmoid function. Following [Grenioux et al., 2024], we place independent Gaussian priors on the parameters p ( w,b ) = N ( w ; 0 , I d ) N ( b ; 0 , 2 . 5 2 ) .

Statistical physics model: ϕ 4 distribution The goal is to sample metastable states of the stochastic Allen-Cahn equation ϕ 4 model (dimension 100) [Albergo et al., 2019, Gabrié et al., 2022]. The ϕ 4 model is a continuous relaxation of the Ising model which is used to study phase transitions in statistical mechanics. Following [Albergo et al., 2019, Gabrié et al., 2022, Grenioux et al., 2024], we consider a version of the model discretised on a 1-dimensional grid of size d = 100 . Each configuration of the model is represented by a d -dimensional vector ( ϕ i ) d i =1 . We clip the field to 0 at the boundaries by defining ϕ 0 = ϕ d +1 = 0 . The negative log-density of the distribution is defined as

<!-- formula-not-decoded -->

We fix the parameters a = 0 . 1 and β = 20 to ensure a bimodal regime, and vary the value of h . We denote by w + the statistical occurrence of configurations with ϕ d/ 2 &gt; 0 and w -the statistical occurrence of configurations with ϕ d/ 2 &lt; 0 . At h = 0 , the distribution is invariant under the symmetry ϕ →-ϕ , so we expect w + = w -. For h &gt; 0 , the negative mode becomes dominant.

When the dimension d is large, the relative probabilities of the two modes can be estimated by Laplace approximations at 0-th and 2-nd order. Let ϕ h + and ϕ h -denote the local maxima of the distribution, these approximations yield respectively

<!-- formula-not-decoded -->

where H h is the Hessian of the function ϕ → ln π h ( ϕ ) .

For this last benchmark, we only compare our method with SLIPS [Grenioux et al., 2024], as it is the only baseline capable of accurately recovering samples from the target distribution while preserving the correct relative mode weights.

## D.2 Evaluation metrics

To evaluate the quality of the generated samples, we consider the entropy regularised Wasserstein-2 distance [Peyré and Cuturi, 2019], with regularisation parameter ε = 0 . 05 , for distributions where true samples are available. This metric can be efficiently computed in JAX using the OTT library [Cuturi et al., 2022].

For the Funnel benchmark, we assess the sample quality using the Kolmogorov-Smirnov distance. Specifically, we use the sliced version introduced in Grenioux et al. [2023, Appendix D.1].

In the Bayesian logistic regression tasks, performance is measured via the mean predictive loglikelihood, computed as p ( w,b |D test ) , where D test is a held-out test dataset not used during training.

## D.3 Algorithms and hyperparameters

All algorithms are initialised with samples drawn from a standard Gaussian distribution to ensure a fair comparison. This contrasts with the approach in Grenioux et al. [2024], where the initial samples are drawn using prior knowledge of the target distribution via a quantity denoted as R π , which is upper bounded by the scalar variance of the target π . While we observe that initialising from this distribution can improve performance for our methods, such information is typically unavailable in real-world scenarios. Nevertheless, for the ϕ 4 model, we consider the same implementation for SLIPS as described in Grenioux et al. [2024], given the complexity of this benchmark.

Besides, to ensure consistency, all algorithms use the same number of energy evaluations, as detailed for each benchmark distribution in Table 3.

Table 3: Number of energy evaluations for each benchmark

|                    | 8-MoG ( d = 2)           | 40-MoG ( d = 2)           | 40-MoG ( d = 50)          | Rings ( d = 2)            | Funnel ( d = 10)         |
|--------------------|--------------------------|---------------------------|---------------------------|---------------------------|--------------------------|
| Energy evaluations | 3 × 10 5                 | 3 × 10 5                  | 5 × 10 6                  | 3 × 10 5                  | 5 × 10 5                 |
|                    | 5-dimDW ( m = 5 , δ = 4) | 10-dimDW ( m = 5 , δ = 3) | 50-dimDW ( m = 5 , δ = 2) | Ionosphere ( d = 35)      | Sonar ( d = 61) 3 × 10 6 |
| Energy evaluations | 3 × 10 5                 | 5 × 10 5                  | 5 × 10 6                  | 1 × 10 6                  |                          |
|                    | ϕ 4 model ( d = 100)     | 10-MoS ( d = 2)           | 10-MoS ( d = 10) 5 × 10   | 10-MoS ( d = 50) 1 × 10 6 |                          |
| Energy evaluations | 1 × 10 8                 | 3 × 10 5                  | 5                         |                           |                          |

Hyperparameter selection For each baseline algorithm, we perform a grid search over a predefined set of hyperparameter values. Selection is based on the corresponding performance metric, computed using 4096 samples. The selected hyperparameters for each algorithm are summarised below.

- The SMC and AIS algorithms define a sequence of annealed distributions µ k for k ∈ [0 , K ] . At each intermediate distribution, we perform n = 64 MCMCsteps. The parameter n MCMC is selected from the interval [20 , 100] using a grid step of 4 . The total number of distributions K is chosen such that the product n × K matches the number of target evaluations specified in Table 3.
- For ULMC algorithm, we tune three hyperparameters: the mass M , the friction coefficient Γ and the step size h . In all benchmarks, we fix the mass to the identity matrix M = I . The step size is chosen within the following grid h ∈ { 0 . 001 , . . . , 0 . 009 , 0 . 01 , . . . , 0 . 09 , 0 . 10 , 0 . 11 , . . . , 0 . 20 } . The selected values for the step sizes h are provided in Table 4. Additionally, we use a non-constant friction coefficient. Specifically, when running the algorithm for L steps, the friction coefficient remains constant at Γ min = 1 for the first L/ 2 steps and then increases linearly from Γ min to Γ max = 5 during the remaining steps. The values of Γ min and Γ max are chosen within the grid { 0 . 001 , 0 . 005 , 0 . 01 , , . . . , 0 . 09 , 0 . 1 , . . . , 0 . 9 , 1 , 1 . 5 , . . . , 10 } .
- For the PT algorithm, the number of chains K is selected from the interval [1 , 20] with a grid step of 2. The corresponding temperatures are chosen to be equally spaced on a logarithmic scale, with the minimum temperature fixed at 1 and the maximum temperature chosen from the grid { 10 2 , 10 3 , 10 4 , 10 5 , 10 6 , 10 7 } . In our experiments, we use K = 5 parallel chains with temperatures { 1 . 0 , 5 . 6 , 31 . 6 , 177 . 8 , 1000 . 0 } . Each chain employs a Hamiltonian Monte Carlo sampler with n leapfrog steps per sample and a step size h specified in Table 4. The grid for the step size is the same as that of ULMC. The number of leapfrog steps n is chosen so that the total number of target evaluations matches the computational budget defined in Table 3.
- The DiGS algorithm uses K noise levels ranging from α K to α 1 . The number of noise levels K is selected within the interval [1 , 20] with a grid step of 2. The maximum and minimum noise levels, α 1 and α K , respectively, are chosen from the grid { 0 . 05 , . . . , 0 . 09 , 0 . 1 , 0 . 2 , . . . , 1 } . At each noise level, we perform n Gibbs Gibbs sweeps, each consisting of n MALA denoising steps using MALA with step size h . The number of Gibbs sweeps n Gibbs is selected from the interval [50 , 500] using a grid step of 50, while the grid for the step size is the same as that of ULMC. The final values of K ,

Table 4: Selected step sizes for ULMC and PT algorithm across experiments.

|      | 8-MoG   | 40-MoG 2-dim   | 40-MoG 50-dim   | Rings      | Funnel   |
|------|---------|----------------|-----------------|------------|----------|
| ULMC | 0 . 05  | 0 . 05         | 0 . 01          | 0 . 05     | 0 . 03   |
| PT   | 0 . 10  | 0 . 10         | 0 . 03          | 0 . 10     | 0 . 05   |
|      | 5-dimDW | 10-dimDW       | 50-dimDW        | Ionosphere | Sonar    |
| ULMC | 0 . 05  | 0 . 05         | 0 . 01          | 0 . 04     | 0 . 04   |
| PT   | 0 . 10  | 0 . 05         | 0 . 05          | 0 . 05     | 0 . 03   |

α K , α 1 , n Gibbs, and h are provided in Table 5 for each experiment. The number of MALA steps, n MALA, is determined based on the computational budget.

Table 5: Selected hyperparameters for DiGS algorithm across experiments.

|         | 8-MoG   | 40-MoG 2-dim   | 40-MoG 50-dim   | Rings      | Funnel   |
|---------|---------|----------------|-----------------|------------|----------|
| K       | 3       | 1              | 5               | 1          | 3        |
| α K     | 0 . 1   | 0 . 1          | 0 . 1           | 0 . 1      | 0 . 1    |
| α 1     | 0 . 5   | 0 . 1          | 0 . 9           | 0 . 1      | 0 . 5    |
| n Gibbs | 200     | 300            | 100             | 300        | 200      |
| h       | 0 . 05  | 0 . 10         | 0 . 01          | 0 . 10     | 0 . 03   |
|         | 5-dimDW | 10-dimDW       | 50-dimDW        | Ionosphere | Sonar    |
| K       | 1       | 3              | 5               | 3          | 3        |
| α K     | 0 . 1   | 0 . 1          | 0 . 1           | 0 . 1      | 0 . 1    |
| α 1     | 0 . 1   | 0 . 5          | 0 . 8           | 0 . 5      | 0 . 5    |
| n Gibbs | 300     | 200            | 100             | 200        | 200      |
| h       | 0 . 06  | 0 . 01         | 0 . 005         | 0 . 03     | 0 . 01   |

- The RDMC algorithm has a hyperparameter T , corresponding to the final time of the OU process, its value is provided in Table 6. All other parameters follow the implementation detailed in [Grenioux et al., 2024].
- For the SLIPS algorithm, we adopt the implementation and hyperparameters provided in the original paper [Grenioux et al., 2024]. To ensure a fair comparison with other algorithms initialised from a standard Gaussian distribution, we set the scalar variance parameter R π = √ d , yielding σ = R π / √ d = 1 . While this choice may degrade SLIPS' performance compared to using an optimally tuned R π , estimating such a parameter in practical scenarios is often non trivial. Nevertheless, for completeness, we report below the performance of SLIPS with its optimal parameters alongside our algorithms. In this setting, the results are comparable. However, a key advantage of our method is that it does not require estimation of the scalar variance. Moreover, if we initialise our algorithms using the same informed choice R π by setting the base distribution ν ∼ N (0 , σ 2 I ) , with σ = R π / √ d , we observe a performance improvement, particularly in the overdamped regime.
- For all numerical experiments in the main text, we use the underdamped version of our algorithms, as it yields improved performance. For MULTALMC, the required hyperparameters include the schedule function λ t , the values of λ δ and ˜ λ , the mass matrix M , ε , the friction coefficient Γ , the step size h , and the number of SROCK steps s . The grid for the step size is the same as that of ULMC. The scale separation parameter ε is chosen within the grid { 0 . 001 , 0 . 005 , 0 . 01 , . . . , 0 . 09 , 0 . 10 , 0 . 11 , . . . , 0 . 20 } . The parameter ˜ λ is chosen from the interval [0 . 3 , 0 . 7] using a grid step of 0.05. In all experiments, we select λ δ = 0 . 01 , ˜ λ = 0 . 6 , M = I , and s = 5 . Additionally, we consider a time-dependent friction coefficient. Specifically, when running the algorithm for L steps, the friction coefficient remains constant at Γ min for the first L/ 2 steps and then increases linearly from Γ min to Γ max during the remaining steps. The values of Γ min and Γ max are selected within the grid { 0 . 001 , 0 . 005 , 0 . 01 , , . . . , 0 . 09 , 0 . 1 , . . . , 0 . 9 , 1 , 1 . 5 , . . . , 10 } . The mass matrix is set to the identity. It is worth noting that the number of SROCK steps can be reduced, which may result in a small compromise in performance. This effect is further analysed in Appendix E.2. Lastly, the number of iterations is determined based on the computational budget specified in Table 3.

Table 6: Selected hyperparameters for RDMC algorithm across experiments.

|    | 8-MoG         | 40-MoG 2-dim   | 40-MoG 50-dim   | Rings         | Funnel        |
|----|---------------|----------------|-----------------|---------------|---------------|
| T  | - log(0 . 80) | - log(0 . 75)  | - log(0 . 70)   | - log(0 . 85) | - log(0 . 90) |
|    | 5-dimDW       | 10-dimDW       | 50-dimDW        | Ionosphere    | Sonar         |
| T  | - log(0 . 70) | - log(0 . 70)  | - log(0 . 75)   | - log(0 . 95) | - log(0 . 95) |

Table 7: Metrics for different benchmarks averaged across 30 seeds. The metric for the mixture of Gaussian (MoG), Rings and the double well potential (DW) is the entropy regularised Wasserstein-2 distance (with regularisation parameter 0.05), the metric for the Funnel is the sliced KolmogorovSmirnov distance and the metric for the Bayesian logistic regression on Ionosphere and Sonar datasets is the average predictive posterior log-likelihood on a test dataset. We compare the performance of our algorithms (initialised with a standard Gaussian distribution) with that of SLIPS with an optimal value of the parameter R π , refer to as SLIPS ( R π ) .

| Algorithm     | 8-MoG ( ↓ ) ( d = 2)           | 40-MoG ( ↓ ) ( d = 2)           | 40-MoG ( ↓ ) ( d = 50)          | Rings ( ↓ ) ( d = 2)       | Funnel ( ↓ ) ( d = 10)   |
|---------------|--------------------------------|---------------------------------|---------------------------------|----------------------------|--------------------------|
| SLIPS ( R π ) | 0 . 66 ± 0 . 11                | 0 . 98 ± 0 . 06                 | 20 . 85 ± 0 . 63                | 0 . 19 ± 0 . 02            | 0 . 029 ± 0 . 007        |
| MULTALMC      | 0 . 65 ± 0 . 07                | 0 . 91 ± 0 . 04                 | 20 . 58 ± 0 . 71                | 0 . 18 ± 0 . 02            | 0 . 032 ± 0 . 005        |
| MULTCDIFF     | 0 . 62 ± 0 . 10                | 0 . 93 ± 0 . 06                 | 20 . 13 ± 0 . 59                | 0 . 19 ± 0 . 03            | 0 . 031 ± 0 . 005        |
| Algorithm     | 5-dim DW( ↓ ) ( m = 5 , δ = 4) | 10-dim DW( ↓ ) ( m = 5 , δ = 3) | 50-dim DW( ↓ ) ( m = 5 , δ = 2) | Ionosphere ( ↑ ) ( d = 35) | Sonar ( ↑ ) ( d = 61)    |
| SLIPS ( R π ) | 2 . 05 ± 0 . 30                | 4 . 45 ± 0 . 36                 | 14 . 99 ± 0 . 59                | - 86 . 72 ± 0 . 10         | - 109 . 38 ± 0 . 12      |
| MULTALMC      | 2 . 08 ± 0 . 16                | 4 . 48 ± 0 . 40                 | 14 . 03 ± 0 . 63                | - 86 . 85 ± 0 . 09         | - 109 . 05 ± 0 . 13      |
| MULTCDIFF     | 1 . 95 ± 0 . 24                | 4 . 23 ± 0 . 37                 | 13 . 98 ± 0 . 56                | - 86 . 33 ± 0 . 10         | - 109 . 60 ± 0 . 21      |

Table 8: Selected hyperparameters for MULTALMC algorithm across experiments.

|                       | 8-MoG                                     | 40-MoG 2-dim                              | 40-MoG 50-dim                             | Rings                                    | Funnel                                    |
|-----------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|------------------------------------------|-------------------------------------------|
| λ t ε Γ min , Γ max h | Linear 0 . 10 0 . 07 , 0 . 5 0 . 005      | Linear 0 . 05 0 . 01 , 0 . 5 0 . 005      | Linear 0 . 05 0 . 01 , 0 . 5 0 . 001      | Linear 0 . 10 0 . 1 , 0 . 5 0 . 005      | Cosine-like 0 . 07 0 . 01 , 0 . 5 0 . 003 |
|                       | 5-dimDW                                   | 10-dimDW                                  | 50-dimDW                                  | Ionosphere                               | Sonar                                     |
| λ t ε Γ min , Γ max h | Cosine-like 0 . 10 0 . 01 , 0 . 5 0 . 005 | Cosine-like 0 . 07 0 . 01 , 0 . 5 0 . 001 | Cosine-like 0 . 05 0 . 01 , 0 . 5 0 . 001 | Cosine-like 0 . 10 0 . 1 , 0 . 5 0 . 005 | Cosine-like 0 . 10 0 . 1 , 0 . 5 0 . 005  |
|                       | ϕ 4 model                                 | 10-MoS 2-dim                              | 10-MoS 10-dim                             | 10-MoS 50-dim                            |                                           |
| λ t ε Γ min , Γ max h | Cosine-like 0 . 05 0 . 01 , 0 . 5 0 . 001 | Linear 0 . 10 0 . 1 , 0 . 5 0 . 005       | Linear 0 . 10 0 . 1 , 0 . 5 0 . 005       | Linear 0 . 05 0 . 1 , 0 . 5 0 . 001      |                                           |

On the other hand, for MULTCDIFF, the following hyperparameters need to be specified: the friction coefficient Γ , the scale separation parameter ε , the final time T of the OU process, the time discretisation 0 = T 0 &lt; · · · &lt; T L = T , and the number of SROCK steps s . The values of Γ , ε and s are set to be the same as in the MULTALMC algorithm. The time discretisation is chosen such that the difference λ T l +1 -λ T l is constant, where λ t denotes the OU schedule. The number of iterations L is determined based on the computational budget for each benchmark.

## D.4 Computation time

All experiments were conducted on a GPU server consisting of eight Nvidia GeForce RTX 3090 Ti GPU cards, 896 GB of memory and 14TB of local on-server data storage. Each GPU has 10496 cores as well as 24 GB of memory.

Recall that to ensure a fair comparison, we fixed the number of energy evaluations across all algorithms. As a result, the runtime of ULMC, DiGS, RDMC, SLIPS, and our proposed methods, MULTALMC and MULTCDIFF, are broadly similar. In contrast, algorithms such as SMC, AIS, and PT exhibit longer runtimes due to their accept/reject steps. This is demonstrated in Figure 2,

which shows average computation times (over 30 random seeds) for each method on the 40-MoG benchmark in 50 dimensions.

Figure 2: Boxplots of computation times for different algorithms on the 40-MoG benchmark in 50 dimensions, averaged over 30 random seeds.

<!-- image -->

In addition, we analyse how the runtime of our methods scales with dimensionality. To that end, Table 9 includes performance metrics, number of energy evaluations, and computation times for the 40-MoG benchmark across different dimensions, averaged over 30 random seeds. The performance metric used is the entropy-regularised Wasserstein-2 distance, W δ 2 , with regularisation parameter δ = 0 . 05 .

Table 9: Number of energy evaluations, entropy regularised Wasserstein-2 distance (with regularisation parameter 0.05) and computations times for our sampling methods, MULTALMC (Algorithm 3) and MULTCDIFF (Algorithm 4) evaluated on the 40-MoG benchmark across different dimensions. The results are averaged over 30 random seeds.

|                    |                    | d = 2           | d = 5           | d = 10          | d = 20          | d = 50           |
|--------------------|--------------------|-----------------|-----------------|-----------------|-----------------|------------------|
| Energy evaluations | Energy evaluations | 3 × 10 5        | 6 × 10 5        | 1 × 10 6        | 2 × 10 6        | 5 × 10 6         |
| W δ 2              | MULTALMC           | 0 . 91 ± 0 . 04 | 1 . 64 ± 0 . 12 | 4 . 10 ± 0 . 27 | 8 . 76 ± 0 . 34 | 20 . 58 ± 0 . 71 |
| W δ 2              | MULTCDIFF          | 0 . 93 ± 0 . 06 | 1 . 59 ± 0 . 20 | 3 . 92 ± 0 . 36 | 8 . 68 ± 0 . 41 | 20 . 13 ± 0 . 59 |
| Time (s)           | MULTALMC           | 35 ± 5          | 57 ± 9          | 111 ± 13        | 223 ± 28        | 517 ± 28         |
| Time (s)           | MULTCDIFF          | 33 ± 6          | 58 ± 7          | 104 ± 15        | 231 ± 21        | 502 ± 30         |

## E Additional numerical experiments

## E.1 Comparison of overdamped and underdamped dynamics

As noted in the main text, the overdamped version of MULTALMC requires a small value of κ , which corresponds to a slowly varying dynamics driven by ∇ log ˆ µ t to perform well in practice. However, this leads to a large number of discretisation steps since we use a small step size with respect to this slowly changing dynamics resulting in high computational costs. In contrast, the underdamped version achieves better performance without the need for such small step sizes, thanks to the faster convergence properties of underdamped dynamics [Eberle and Lörler, 2024]. To further analyse this, we compare in Table 10 the performance and number of energy evaluations of the overdamped and underdamped versions of MULTALMC, as specified in Algorithms 2 and 3, respectively, on the 40-MoG benchmark across different dimensions. The results show that the overdamped version

requires approximately an order of magnitude more energy evaluations (and hence time steps) to achieve performance comparable to the underdamped version.

Table 10: Performance metrics on the 40-MoG benchmark across varying dimensions, averaged over 30 runs with different random seeds, along with the number of energy evaluations for the overdamped and underdamped versions of MULTALMC, as defined in Algorithms 2 and 3, respectively.

| MULTALMC    |                     | d = 2                    | d = 5                    | d = 10                   | d = 20                   | d = 50                    |
|-------------|---------------------|--------------------------|--------------------------|--------------------------|--------------------------|---------------------------|
| Overdamped  | W δ 2 # evaluations | 1 . 12 ± 0 . 10 1 × 10 6 | 1 . 88 ± 0 . 22 8 × 10 6 | 4 . 46 ± 0 . 39 2 × 10 7 | 9 . 23 ± 0 . 67 6 × 10 7 | 21 . 33 ± 0 . 98 1 × 10 8 |
| Underdamped | W δ 2 # evaluations | 0 . 91 ± 0 . 04 3 × 10 5 | 1 . 64 ± 0 . 12 6 × 10 5 | 4 . 10 ± 0 . 27 1 × 10 6 | 8 . 76 ± 0 . 34 2 × 10 6 | 20 . 58 ± 0 . 71 5 × 10 6 |

## E.2 Analysis of the impact of the number of SROCK steps in the discretisation

The number of SROCK steps used in our algorithms-MULTALMC and MULTCDIFF-as implemented in Algorithms 2, 3, and 4, plays a critical role in balancing computational efficiency and numerical performance. Increasing the number of inner SROCK steps generally leads to better performance [Abdulle et al., 2018]; however, this comes at the cost of greater computational overhead. Ideally, we aim to identify the minimum number of steps required to maintain strong performance and computational efficiency.

In the numerical experiments presented in Section 5, we use five SROCK steps. We further analyse the sensitivity of our accelerated methods, MULTALMC (Algorithm 3) and MULTCDIFF (Algorithm 4), to the number of SROCK steps. To this end, we report the entropy regularised Wasserstein-2 distance for the 40-MoG benchmark in 50 dimensions (Table 11) when running our methods with varying numbers of SROCK steps while keeping the number of sampling steps L for the slow process fixed.

Table 11: Performance on the 40-MoG benchmark in 50 dimensions using varying numbers of SROCK steps in the discretisation of MULTALMC and MULTCDIFF, as proposed in Algorithms 3 and 4, respectively. The number of sampling steps L for the slow process is held fixed. The standard deviation is computed over 30 runs with different random seeds. The regularisation parameter for the entropy regularised Wasserstein-2 distance W δ 2 is set to δ = 0 . 05 .

| SROCK steps   | SROCK steps   | 3                | 5                | 7                | 9                | 11               |
|---------------|---------------|------------------|------------------|------------------|------------------|------------------|
| W δ 2         | MULTALMC      | 21 . 11 ± 0 . 74 | 20 . 58 ± 0 . 71 | 20 . 29 ± 0 . 68 | 20 . 08 ± 0 . 50 | 19 . 99 ± 0 . 45 |
| W δ 2         | MULTCDIFF     | 20 . 92 ± 0 . 63 | 20 . 13 ± 0 . 59 | 20 . 01 ± 0 . 57 | 19 . 96 ± 0 . 48 | 19 . 89 ± 0 . 40 |

## E.3 Stability across hyperparameter values

Our algorithms demonstrate robustness and stability across a broad range of hyperparameter values, as shown in Figures 3 and 4. These results suggest that while some tuning is beneficial, our methods do not require highly sensitive or exhaustive hyperparameter optimisation.

## F Limitations and future work

We elaborate here on the limitations of our method discussed in Section 6.

First, the theoretical guarantees presented in Section 4 rely on stringent assumptions, such as strong convexity of the potential V π and Lipschitz continuity of the score function of the target ∇ log π . Relaxing these assumptions, e.g., to strong convexity outside a compact region or to satisfying weak functional inequalities, could broaden the applicability of our analysis. The former would accommodate target distributions that are multimodal within a compact region and Gaussian-like in the tails, while the latter could capture heavy-tailed distributions.

<!-- image -->

- (a) Performance against scale separation parameter ε .
- (b) Performance against number of steps.
- (a) Performance against scale separation parameter ε .
- (b) Performance against number of steps.

<!-- image -->

Figure 3: Ablation results on the 40-component Mixture of Gaussians benchmark in 50 dimensions. Results are averaged over 30 random seeds.

<!-- image -->

<!-- image -->

Figure 4: Ablation results on the Sonar benchmark. Results are averaged over 30 random seeds. The performance metric is the average predictive posterior log-likelihood on a test dataset.

Additionally, the current method requires manual tuning of hyperparameters such as step size δ , scale separation parameters ε and friction coefficient Γ . Automating this tuning process, similar to the approach in Blessing et al. [2025], is a valuable direction for improving usability and robustness.

Further research could explore extending the controlled diffusion framework to heavy-tailed target distributions, which pose additional challenges as explained in Appendix B.2.2, or developing more efficient numerical schemes for implementing the proposed multiscale samplers.

## G Broader impact

This work introduces MULTALMC and MULTCDIFF, two training-free, multiscale diffusion samplers that enable efficient, provably accurate sampling from complex distributions. These methods have the potential to significantly reduce the computational and environmental footprint of generative modelling and Bayesian inference. Potential societal benefits include faster scientific discovery, improved uncertainty quantification, and broader participation in high-impact ML research. As with all general-purpose algorithms, misuse is possible, e.g. lowering the cost of harmful synthetic content generation or accelerating harmful molecule design.
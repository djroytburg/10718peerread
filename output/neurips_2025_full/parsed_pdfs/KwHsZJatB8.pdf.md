## Stabilizing LTI Systems under Partial Observability: Sample Complexity and Fundamental Limits

## Ziyi Zhang, Yorie Nakahira, Guannan Qu

Department of Electrical and Computer Engineering Carnegie Mellon University Pittsburgh, PA 15213

ziyizhan,yorie,gqu@andrew.cmu.edu

## Abstract

We study the problem of stabilizing an unknown partially observable linear timeinvariant (LTI) system. For fully observable systems, the state-of-the-art approaches leverage an unstable/stable subspace decomposition to achieve sample complexity that depends only on the number of unstable modes, independent of the dimension of the system state. However, it remains open whether such sample complexity can be achieved for partially observable systems because such systems do not admit a uniquely identifiable unstable subspace. In this paper, we propose LTS-P, a novel technique that leverages compressed singular value decomposition (SVD) on the 'lifted' Hankel matrix to estimate the unstable subsystem up to an unknown transformation. Then, we design a stabilizing controller that integrates a robust stabilizing controller for the unstable mode and a small-gain-type assumption on the stable subspace. We show that LTS-P achieves state-of-the-art, dimension-free sample complexity that scales only with the number of unstable modes. This substantially reduces data requirements for stabilizing high-dimensional systems, particularly those dominated by stable dynamics.

## 1 Introduction

Learning-based control of unknown dynamical systems is of critical importance for many autonomous control systems [3, 30, 6, 24, 12]. Despite recent advances, many existing methods make strong assumptions such as open-loop stability, availability of an initial stabilizing controller, and fully observable systems [15, 41]. However, these assumptions may not hold in practice. Motivated by this gap, this paper studies the problem of stabilizing an unknown, partially-observable, unstable system without access to an initial stabilizing controller. In particular, we consider the following linear time-invariant (LTI) system:

<!-- formula-not-decoded -->

where x t ∈ R n and u t ∈ R d u , y t ∈ R d y are the state, control input, and observed output at time step t ∈ { 0 , . . . , T -1 } , respectively. The system also has additive observation noise v t ∼ N (0 , σ 2 v I ) . While there are works studying system identification for partially observable LTI systems [15, 43, 41, 34, 51], they do not address the problem of stabilization, and many assume the system is open-loop stable [15, 41, 34]. Other adaptive control approaches can address the learn-to-stabilize problem for fully observable systems based on Lyapunov methods [35, 37], but there are few systematic approaches to construct a Lyapunov function in a way that optimizes sample complexities or transient performance during learning.

In the special case of fully-observable LTI system ( C = I ), Chen and Hazan [9] reveals that the transient performance during the learn-to-stabilize process suffers from exponential blow-up, i.e. the

system state can blow up exponentially in the state dimension. This phenomenon arises because stabilization requires accurate identification of the full system dynamics, which in turn necessitates at least n samples along a single trajectory. During this identification phase, the system can remain unstable and grow exponentially. To relieve this problem, Hu et al. [19] proposed a framework that separates the unstable component and focuses on stabilizing these subsystems. This reduces the sample complexity to only grow with the number of unstable eigenvalues, rather than the full state dimension n . This result was later extended to noisy setting in Zhang et al. [50]. To date, this dependence on the number of unstable eigenvalues remains the best sample complexity for the learn-to-stabilize problem in the fully observable case.

In contrast, we address a more challenging setting of partially observable systems, and answer the following research question: Is it possible to stabilize a partially observable LTI system by only identifying an unstable component (to be properly defined) with a sample complexity independent of the overall state dimension ? This question introduces two key technical challenges. First, partially observable systems typically require a dynamic controller, which renders the stable feedback controllers in the existing approach inapplicable [19]. Second, the construction in 'unstable subspace' in Hu et al. [19] is not uniquely defined in the partially observable case. This is because the state and the A,B,C matrices are not uniquely identifiable from the learner's perspective, as any similarity transformation of the matrices can yield the same input-output behavior.

Contribution. In this paper, we answer the above question in the affirmative. Firstly, we propose a novel definition of 'unstable component' to be a low-rank version of the transfer function of the original system that only retains the unstable eigenvalues. Based on this unstable component, we propose LTS-P, which leverages compressed singular value decomposition (SVD) on a 'lifted' Hankel matrix to estimate the unstable component. Then, we design a robust stabilizing controller for the unstable component, and show that it stabilizes the full system under a small-gain-type assumption on the H ∞ norm of the stable component. Importantly, our theoretical analysis shows that the sample complexity of the proposed approach only scales with the dimension of unstable component, i.e. the number of unstable eigenvalues, as opposed to the dimension of the full state space. We also conduct simulations to validate the effectiveness of LTS-P, showcasing its ability to efficiently stabilize partially observable LTI systems with reduced samples (Figures 1 and 2).

Moreover, the technical innovations underlying our approach are of independent interest. We show that by using compressed singular value decomposition (SVD) on a properly defined lifted Hankel matrix, we can estimate the unstable component of the system. While this is conceptually related to classical model reduction techniques [13, Chapter 4.6], our setting departs significantly: unlike prior work for stable dynamics and bounded Hankel matrices [13], we address systems with unstable modes, where the associated Hankel matrices may grow unboundedly over time. This distinction renders standard identification and reduction techniques inapplicable [15, 41, 34]. Interestingly, the H ∞ norm condition on the stable component, derived from the small gain theorem, is a necessary and sufficient condition for stabilization. This characterization reveals the exact subspace of the system that must be estimated for stabilization to be possible. As a result, our analysis not only supports the optimality of LTS-P in terms of sample complexity but also informs the fundamental limit of stabilizability.

Related Work. Our work is closely related to learn-to-stabilize on multiple trajectories and learn-tocontrol with known stabilizing controllers. In addition, we will also briefly cover existing literature in learn-to-stabilize on a single trajectory, adaptive control, and system identification.

Learn-to-stabilize on multiple trajectories. There are also works that do not assume open-loop stability and learn the full system dynamics before designing a stabilizing controller, typically incurring a sample complexity of ˜ Θ( poly ( n )) [12, 45, 51]. Recently, a model-free approach via the policy gradient method offers a novel perspective with the same sample complexity [36]. While these techniques are developed for fully observable systems, the proposed algorithm, LTS-P, tackles a significantly more challenging setting of partially observable, unstable systems. As detailed above, this setting introduces fundamental technical challenges that require novel algorithmic and analytical tools beyond those used in fully observable systems. Moreover, compared with those works, the sample complexity of the proposed algorithm does not depend on n and only scales with the number of unstable modes.

Learn to control with known stabilizing controller. Extensive research has been conducted on stabilizing LTI systems under stochastic noise [5, 22, 25, 30]. One branch of research uses the

model-free approach to learn the optimal controller [16, 21, 30, 46, 49]. Those algorithms typically require a known stabilizing controller as an initialization policy for the learning process. Another line of research utilizes model-based approaches, which require an initial stabilizing controller to learn the system dynamics before designing the controllers [10, 32, 38, 52]. On the other hand, we focus on learn-to-stabilize. Our method can be used as the initial policy in these methods to remove their requirement for initial stabilizing controllers.

Learn-to-stabilize on a single trajectory. Learning to stabilize a linear system in an infinite time horizon is a classic problem in control [7, 26, 27]. Early works achieved the regret bounds of 2 O ( n ) O ( √ T ) , which rely on assumptions of observability and strictly stable transition matrices [1, 20]. Subsequent work improved the regret to 2 ˜ O ( n ) + ˜ O ( poly ( n ) √ T ) [9, 28]. Recently, Hu et al. [19] proposed an algorithm that requires ˜ O ( k ) samples, where k is the number of unstable modes. While these techniques are developed for fully observable systems using a single trajectory, we consider a different problem of learning to stabilize a partially observable system using multiple trajectories.

Adaptive control. Adaptive control is a well-established methodology for controlling systems with uncertain or time-varying parameters [8, 35, 37]. Existing literature has established techniques for stabilizing unknown systems with asymptotic stability [9, 14, 29, 44, 45]. Other works use past trajectory to estimate the system dynamics and then design the controller [4, 11, 31]. While these works often assume stability, fully observable systems and are based on Lyapunov methods, we focus on partially observable, unstable systems and build our algorithm based on compressed SVD on the lifted Hankel matrix.

System identification. Existing literature has developed a variety of techniques to estimate system parameters [33, 40, 43, 42, 47]. Hankel matrix are also used in techniques such as Eigensystem Realization Algorithm (ERA), subspace identification, etc [23, 39]. Our work utilizes a similar approach to partially determine the system parameters before constructing the stabilizing controller. While these works focus on identifying the system dynamics, we close the loop and establish state-of-the-art and optimal sample complexity guarantees for the stabilization problem.

## 2 Problem Statement

Notations. We use ∥·∥ to denote the L 2 -norm for vectors and the spectral norm for matrices. We use M ∗ to represent the conjugate transpose of M . We use σ min ( · ) and σ max ( · ) to denote the smallest and largest singular value of a matrix, and κ ( · ) to denote the condition number of a matrix. We use the standard big O ( · ) , Ω( · ) , Θ( · ) notation to highlight dependence on a certain parameter, hiding all other parameters. We use f ≲ g , f ≳ g , f ≍ g to mean f = O ( g ) , f = Ω( g ) , f = Θ( g ) respectively while only hiding numeric constants . We provide an indexing of notations at Appendix I.

̸

For simplicity, we primarily deal with the system where D = 0 . For the case where D = 0 , we can easily estimate D in the process and subtract Du t to obtain a new observation measure not involving control input. We briefly introduce the method for estimating D and how to apply the proposed algorithm in the case when D = 0 in Appendix F.

̸

Learn-to-stabilize. As the unknown system as defined in (1) can be unstable, the goal of the learn-to-stabilize problem is to return a controller that stabilizes the system using data collected from interacting with the system on M rollouts. More specifically, in each rollout, the learner can determine u t and observe y t for a rollout of length T starting from x 0 , which we assume x 0 = 0 for simplicity of proof.

Goal. The sample complexity of stabilization is the number of samples, MT , needed for the learner to return a stabilizing controller. Standard system identification and certainty equivalence controller design need at least Θ( n ) samples for stabilization, as Θ( n ) is the number of samples needed to learn the full dynamical system. In this paper, our goal is to study whether it is possible to stabilize the system with sample complexity independent from n .

## 2.1 Background on H ∞ control

In this section, we briefly introduce the background of H -infinity control. First, we define the open loop transfer function of system (1) from u t to y t to be

<!-- formula-not-decoded -->

which reflects the cumulative output of the system in the infinite time horizon. Next, we introduce the H ∞ space on transfer functions in the z -domain.

Definition 2.1 ( H ∞ -space) . Let H ∞ denote the Banach space of matrix-valued functions that are analytic and bounded outside of the unit sphere. Let RH ∞ denote the real and rational subspace of H ∞ . The H ∞ -norm is defined as

<!-- formula-not-decoded -->

where the second equality is a simple application of the Maximum modulus principle. We also denote C ≥ 1 = { z ∈ C : | z | ≥ 1 } be the complement of the unit disk in the complex domain. For any transfer function G , we say it is internally stable if G ∈ RH ∞ .

The H ∞ norm of a transfer function is crucial in robust control, as it represents the amount of modeling error the system can tolerate without losing stability, due to the small gain theorem [53]. Abundant research has been done in H ∞ control design to minimize the H ∞ norm of transfer functions [53]. In this work, H ∞ control play an important role as we treat the stable component (to be defined later) of the system as a modeling error and show that the control we design can stabilize despite the modeling error.

## 3 Algorithm Idea

In this paper, we assume the matrix A does not have marginally stable eigenvalues, and the eigenvalues are ranked in decreasing order of magnitude, i.e. | λ 1 | ≥ | λ 2 | ≥ · · · ≥ | λ k | &gt; 1 &gt; | λ k +1 | ≥ · · · ≥ | λ n | . The high-level idea of the paper is to first decompose the system dynamics into the unstable and stable components (Section 3.1), estimate the unstable component via a low-rank approximation of the lifted Hankel matrix (Section 3.2), and design a robust stabilizing for the unstable component that stabilizes the whole system (Section 3.3).

## 3.1 Decomposition of Dynamics

Given the eigenvalues, we have the following decomposition for the system dynamics matrix:

<!-- formula-not-decoded -->

where R = Q -1 , and the columns of Q 1 ∈ R n × k are an orthonormal basis for the invariant subspace of the unstable eigenvalues λ 1 , . . . , λ k , with N 1 inheriting eigenvalues λ 1 , . . . , λ k from A . Similarly, columns of Q 2 ∈ R n × ( n -k ) form an orthonormal basis for the invariant subspace of the unstable eigenvalues λ k +1 , . . . , λ n and N 2 inherit all the stable eigenvalues λ k +1 , . . . , λ n .

Given the decomposition of the matrix A , our key idea is to only estimate the unstable component of the dynamics, which we define below. Consider the transfer function of the original system:

<!-- formula-not-decoded -->

Therefore, the original system is an additive decomposition into the unstable transfer function F ( z ) , which we refer to as the unstable component, and the stable transfer function ∆( z ) , which we refer to as the stable component.

## 3.2 Approximate low-rank factorization of the lifted Hankel matrix

In this section, we define a 'lifted' Hankel matrix, show it admits a rank k approximation, based on which the unstable component can be estimated.

If each rollout has length T , we can decompose T := m + p + q +2 and estimate the following 'lifted' Hankel matrix where the ( i, j ) -th block is [ H ] ij = CA m + i + j -2 B where i = 1 , . . . , p , j = 1 , . . . , q . In other words,

<!-- formula-not-decoded -->

for some m,p,q that we will select later. We call this Hankel matrix 'lifted' as it starts with CA m B . This 'lifting' is essential to our approach, as raising A to the power of m can separate the stable and unstable components and facilitate better estimation of the unstable component, which will become clear later on.

Define

<!-- formula-not-decoded -->

Then we have the factorization H = OC , indicating that H is of rank at most n .

Rank k approximation. We now show that H has a rank k approximation corresponding to the unstable component. Given the decomposition of A in (3), we can write each block of the lifted Hankel matrix as

<!-- formula-not-decoded -->

If ℓ is resonably large, using the fact that N ℓ 1 ≫ N ℓ 2 ≈ 0 , we can have CA ℓ B ≈ CQ 1 N ℓ 1 R 1 B . Therefore, we know that when m is reasonably large, we have H can be approximately factorized as H ≈ ˜ O ˜ C where:

<!-- formula-not-decoded -->

As ˜ O has k columns, ˜ O ˜ C has (at most) rankk . We also use the notation

<!-- formula-not-decoded -->

to denote this rankk approximation of Hankel. As from H to ˜ H , the only thing that are omitted are of order O ( N m 2 ) , it is reasonable to expect that ∥ H -˜ H ∥ ≤ O ( λ m k +1 ) , i.e. this rank k approximation has exponentially decaying error in m , as shown in Lemma C.3.

Estimating unstable component of dyamics. In the actual algorithm (to be introduced in Section 4), ˜ H is to be estimated and therefore not known perfectly.However, to illustrate the methodology of the proposed method, for this subsection, we consider ˜ H to be known perfectly and show that the unstable component F ( z ) can be recovered perfectly.

Suppose we have the following factorization of ˜ H for some ¯ O ∈ R d y × k , ¯ C ∈ R k × d u , (which has infinite possible solutions), ˜ H = ¯ O ¯ C . We show in Lemma D.1 there exists an invertible S , such that ¯ O = ˜ O S, ¯ C = S -1 ˜ C . Therefore, from the construction of ˜ O and ˜ C in (7), we see that ¯ O 1 = CQ 1 N m/ 2 1 S , where ¯ O 1 represent the first block submatrix of ¯ O . 1 Solving for ¯ N 1 for the equation ¯ O 2: p = ¯ O 1: p -1 ¯ N 1 , we can get ¯ N 1 = S -1 N 1 S . After which we can get ¯ C = ¯ O 1 ¯ N -m/ 2 1 = CQ 1 S , and ¯ B = ¯ N -m/ 2 1 ¯ C 1 = S -1 R 1 B . In summary, we get the following realization of the system:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

whose transfer function is exactly F ( z ) , the unstable component of the original system.

1 For row block or column block matrics M , we use M i to denote its i 'th block, and M i : j to denote the submatrix formed by the i, i +1 , . . . , j 'th block.

## Algorithm 1 LTS-P: learning the Hankel Matrix

- 1: for i = 1 : M do
- 2: Generate and apply T Gaussian control inputs [ u ( i ) 0 , . . . , u ( i ) T -1 ] i.i.d ∼ N (0 , σ 2 u ) . Collect y ( i ) = [ y ( i ) 0 , . . . , y ( i ) T -1 ] .
- 4: Compute ˆ Φ with (17).
- 3: end for
- 5: Recover ˆ H with (18).
- 6: Compute ˆ ˜ H from ˆ H with (19).
- 7: Compute ˆ O and ˆ C with (20).
- 8: Recover ˆ N 1 , ˆ N m 2 1 ,O , ˆ N m 2 1 ,C with (21), (22), (23),respectively.
- 9: Compute ˆ C, ˆ B with (24).
- 10: Design H ∞ controller with ( ˆ N 1 , ˆ C, ˆ B ) .

## 3.3 Robust stabilizing controller design

After the unstable component F ( z ) is estimated, we can treat the unobserved part of the system as a disturbance ∆( z ) and then synthesize a robust stabilizing controller. Suppose we design a controller u ( z ) = K ( z ) y ( z ) for F ( z ) , and denote its sensitivity function as:

<!-- formula-not-decoded -->

Now let's look at what if we applied the same K ( z ) to the full system F full ( z ) in (4). In this case, the closed loop system is stable if and only if the transfer function F full K ( z ) = ( I -F full ( z ) K ( z )) -1 is analytic for all | z | ≥ 1 , see e.g. Chapter 5 of [53]. Note that,

<!-- formula-not-decoded -->

The Small Gain Theorem shows a necessary and sufficient condition for the system to have a stabilizing controller:

Lemma 3.1 (Theorem 8.1 of [53]) . Given γ &gt; 0 , the closed loop transfer function defined in (10) is internally stable for any ∥ ∆( z ) ∥ H ∞ ≤ γ if and only if ∥ K ( z ) F K ( z ) ∥ H ∞ &lt; 1 γ .

Since ∆( z ) is stable and therefore has bounded H ∞ norm, it suffices to find a controller such that

<!-- formula-not-decoded -->

in order to stabilize the original full system.

## 4 Algorithm

Based on the ideas developed in Section 3, we now design an algorithm that learns to stabilize from data. The pseudocode is provided in Algorithm 1.

Step 1: Approximate Low-Rank Lifted Hankel Estimation. Section 3 shows that if ˜ H defined in (8) is known, then we can design a controller satisfying (11) to stabilize the system. In this section, we discuss a method to estimate ˜ H with singular value decomposition (SVD) of the lifted Hankel metrix.

Data collection and notation. Consider we sample M trajectories, each with length T . To simplify notation, for each trajectory i ∈ { 1 , . . . , M } , we organize input and output data as

<!-- formula-not-decoded -->

where each u ( i ) j ∼ N (0 , σ 2 u ) are independently selected. We also define the a new matrix for the observation noise as

<!-- formula-not-decoded -->

Substituting the above into (1), we obtain

<!-- formula-not-decoded -->

We further define an upper-triangular Toeplitz matrix and a matrix of observed system dynamics:

<!-- formula-not-decoded -->

Note that the lifted Hankel matrix H in (5) can be recovered from Φ above as they contain the same block submatrices. The measurement data (14) in each rollout can be written as

<!-- formula-not-decoded -->

Therefore, we can estimate Φ by the following ordinary least square problem:

<!-- formula-not-decoded -->

We then estimate the lifted Hankel matrix as follows:

<!-- formula-not-decoded -->

Let ˆ H = ˆ U H ˆ Σ H ˆ V ∗ H denote the singular value decomposition of ˆ H , and we define the k -th order estimation of ˆ H :

<!-- formula-not-decoded -->

where ˆ U ( ˆ V ) is the matrix of the top k left (right) singular vector in ˆ U H ( ˆ V H ), and ˆ Σ is the matrix of the top k singular values in ˆ Σ H .

Step 2: Esitmating unstable transfer function F. With the SVD ˆ ˜ H = ˆ U ˆ Σ ˆ V ∗ , we further do the following factorization:

<!-- formula-not-decoded -->

With the above, we can estimate ¯ N 1 , ¯ C, ¯ B similar to the procedure introduced in Section 3.2:

<!-- formula-not-decoded -->

To reduce the error of estimating CQ 1 and R 1 B and avoid compounding error caused by raising ˆ N 1 to the m/ 2 'th power, we also directly estimate ¯ N m 2 1 from both ˆ O and ˆ C .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In practice, the estimation obtained from (22) and (23) are very similar, and we can use either one to estimate ¯ N m 2 1 . Lastly, we estimate ¯ C and ¯ B as follows:

<!-- formula-not-decoded -->

where for simplicity, we use ̂ N -m 2 1 ,O , ̂ N -m 2 1 ,C to denote the invserse of ̂ N m 2 1 ,O , ̂ N m 2 1 ,C . We will provide accuracy bounds for those estimations in Appendix B. With the above estimations, we are ready to design a controller with the following transfer function,

<!-- formula-not-decoded -->

Step 3: Designing robust controller. After estimating the system dynamics and obtain the estimated transfer function in (25) as discussed in Section 3.3, we can design a stabilizing controller by existing H ∞ control methods to minimize ∥ K ( z ) F K ( z ) ∥ H ∞ . The details on designing the robust controller can be found in robust control documentations, e.g. Chapter 7 of Dullerud and Paganini [13].

What if k is not known a priori? The proposed method requires knowledge of k , the number of unstable eigenvalues. If k is not known, we show in Lemma B.1 and Lemma C.3 that the first k singular values of ˆ H and H increase exponentially with m , and the remaining singular values decrease exponentially with m . Therefore, with a reasonably sized m and if p, q are chosen to be larger than k , there will be a large spectral gap among the singular values of ˆ H . The learner can use the location of the spectral gap to determine the value of k .

## 5 Main Results

In this section, we provide the sample complexity needed for the proposed algorithm to return a stabilizing controller. We first introduce a standard assumption on controllabilty and observability.

Assumption 5.1. The LTI system ( A,B ) is controllable, and ( A,C ) is observable.

We also need an assumption on the existence of a controller that meets the small gain theorem's criterion (11).

Assumption 5.2. There exists a controller K ( z ) that stabilizes plant F ( z ) , such that for some fixed ϵ ∗ &gt; 0 , ∥ K ( z ) F K ( z ) ∥ H ∞ &lt; 1 ∥ ∆( z ) ∥ H ∞ +3 ϵ ∗ .

To state our main result, we introduce some system theoretic quantity.

Controllability/observability Gramian for unstable subsystem. For the unstable subsystem N 1 , R 1 B,CQ 1 , its α -observability Gramian is G ob = [ ( CQ 1 ) ∗ ( CQ 1 N 1 ) ∗ . . . ( CQ 1 N α -1 1 ) ∗ ] ∗ , and its α -controllability Gramian is G con = [ R 1 B,N 1 R 1 B,.. . , N α -1 1 R 1 B ] . Per Lemma H.1 and Assumption 5.1, we know the N 1 , R 1 B,CQ 1 subsystem is both controllable and observable, and hence we can select α to be the smallest integer such that both G ob , G con are rankk . Note that we always have α ≤ k . Our main theorem will use the following system-theoretic parameters: α , κ ( G ob ) , κ ( G con ) (the condition numbers of G ob , G con respectively), and σ min ( G ob ) , σ min ( G con ) (the smallest singular values of G ob , G con , respectively).

Umbrella upper bound L . We use a constant L to upper bound the norms ∥ A ∥ , ∥ B ∥ , ∥ C ∥ , ∥ N 1 ∥ , ∥ R 1 B ∥ , ∥ CQ 1 ∥ . We will use the Jordan form decomposition for N 1 = P 1 Λ P -1 1 , and let L upper bound ∥ P 1 ∥∥ P -1 1 ∥ = κ ( P 1 ) . We also use L in sup | z | =1 ∥ ( zI -N 1 ) -1 ∥ ≤ L | λ k |-1 . Lastly, we use L to upper bound the constant in Gelfand's formula (Lemma H.2) for N 2 and

<!-- formula-not-decoded -->

With the above preparations, we are ready to state the main theorem.

Theorem 5.3. Suppose Assumption 5.1, Assumption 5.2 holds and N 1 is diagonalizable. In the regime where | λ k | -1 , 1 -| λ k +1 | , ϵ ∗ are small, 2 we set (recall throughout the paper ≍ only hides numerical constants)

<!-- formula-not-decoded -->

and we set p = q = m (hence each trajectory's length is T = 3 m +2) and the number of trajectories

<!-- formula-not-decoded -->

for some δ ∈ (0 , 1) . Then, with probability at least 1 -δ , ˆ K ( z ) obtained by Algorithm 1 stabilizes the original dynamical system (2) .

2 This regime is the most interesting and challenging regime. For more general values of | λ k | -1 , 1 -| λ k +1 | , ϵ ∗ , see the bound in (106) which takes a more complicated form.

The total number of samples needed for the algorithm is MT = (3 m + 2) M . In the bound in Theorem 5.3, the only constant that explicitly grows with k is the controllability/observability index α = O ( k ) which appears in the bound for m . Therefore, the sample complexity MT = O ( m 2 ) = O ( α 2 ) = O ( k 2 ) grows quadratically with k and independent from system state dimension n .In the regime k ≪ n , this significantly reduces the sample complexity of stabilization compared to methods that estimate the full system dynamics. To the best of our knowledge, this is the first result that achieves stabilization of an unknown partially observable LTI system with sample complexity independent from the system state dimension n . To validate the claim in Theorem 5.3, we use the LTS-P to stabilize a randomly generated partially observable system and compare the result with two state-of-the-art benchmarks [43, 51] in Appendix A. In Appendix B, we provide a high-level overview of the proof of Theorem 5.3, with detailed proof steps in Appendices C to E.

Dependence on system theoretic parameters. As the system becomes less observable and controllable, κ ( G con ) , κ ( G ob ) increases and σ min ( G con ) , σ min ( G ob ) decreases, increasing m and the sample complexity MT grows in the order of (log κ ( G ob ) κ ( G con ) min( σ min ( G ob ) ,σ min ( G con )) ) 2 . Moreover, when | λ k +1 | , | λ k | are closer to 1 , the sample complexity also increases in the order of (max( 1 1 -| λ k +1 | , 1 | λ k |-1 log 1 | λ k |-1 )) 2 . This makes sense as the unstable and stable components of the system would become closer to marginal stability and harder to distinguish apart. Lastly, if the ϵ ∗ in Assumption 5.2 becomes smaller, we have a smaller margin for stabilization, which also increases the sample complexity in the order of (log 1 ϵ ∗ ) 2 .

Non-diagonalization case. Theorem 5.3 assumes N 1 is diagonalizable, which is only used in a single place in the proof in Lemma D.1. More specifically, we need to upper bound J t ( J ∗ ) -t for some Jordan block J and integer t , and in the diagonalizable case an upper bound is 1 as the Jordan block is size 1 . In the case that N 1 is not diagonalizable, a similar bound can still be proven with dependence on the size of the largest Jordan block of N 1 , denoted as γ . Eventually, this will lead to an additional multiplicative factor γ in the sample complexity on m (cf. (124)), whereas the requirements for p, q, M is the same. The derivations can be found in Appendix G.

Necessity of Assumption 5.2. Assumption 5.2 is needed if one only estimates the unstable component of the system, treating the stable component as unknown. This is because the small gain theorem is an if-and-only-if condition. If Assumption 5.2 is not true, then no matter what controller K ( z ) one designs, there must exist an adversarially chosen stable component ∆( z ) (not known to the learner) causing the system to be unstable, even if F ( z ) is perfectly learned. For a specific construction of such a stability-breaking ∆( z ) , see the proof of the small gain theorem, e.g. Theorem 8.1 of Zhou and Doyle [53]. Although Hu et al. [19], Zhang et al. [50] do not explictly impose this assumption, they impose similar assumptions on the eigenvalues, e.g. the | λ 1 || λ k +1 | &lt; 1 in Theorem 4.2 of Zhang et al. [50], and we believe the small gain theorem and Assumption 5.2 is the intrinsic reason underlying those eigenvalue assumptions in Zhang et al. [50]. We believe this requirement of Assumption 5.2 is the fundamental limit of the unstable + stable decomposition approach. One way to break this fundamental limit is that instead of doing unstable + stable decomposition, we learn a larger component corresponding to eigenvalues λ 1 , . . . , λ ˜ k for some ˜ k &gt; k , which effectively means we learn the unstable component and part of the stable component of the system. Doing so, Assumption 5.2 will be easier to satisfy as when ˜ k approaches n , ∥ ∆( z ) ∥ H ∞ will approach 0 .

Relation to hardness of control results. Recently, there has been related work on the hardness of the learn-to-stabilize problem [9, 48]. Our setting does not conflict with these hardness results for the following reasons. Firstly, our result focuses on the k ≪ n regime. In the regime k = n , our result does not improve over other approaches. Secondly, our Assumption 5.2 is related to the co-stabilizability concept in Zeng et al. [48], as Assumption 5.2 effectively means the existence of a controller that stabilizes the system for all possible ∆( z ) . In some sense, our results complements these results showing when is learn-to-stabilize easy under certain assumptions.

## 6 Conclusion

In this paper, we examined the necessary and sufficient conditions for the stabilization of a partially observable LTI system and proposed a novel SVD-based robust controller design framework. Future work includes studying the transfer function from the process noise w t to y t , and improving the complexity bound in Theorem 5.3.

## 7 Acknowledgement

This work is supported in part by NSF 2154171, NSF CAREER 2339112, CMU CyLab seed funding, in part by by a grant from the Commonwealth of Pennsylvania, Department of Community and Economic Development, in part by National Science Foundation under Grant No. 2442948, and in part by the Department of the Navy, Office of Naval Research, under award number N00014-23-12252. The views expressed are those of the authors and do not reflect the official policy or position of the US Navy, Department of Defense or the US Government.

## References

- [1] Yasin Abbasi-Yadkori and Csaba Szepesvári. Regret bounds for the adaptive control of linear quadratic systems. In Sham M. Kakade and Ulrike von Luxburg, editors, Proceedings of the 24th Annual Conference on Learning Theory , volume 19 of Proceedings of Machine Learning Research , pages 1-26, Budapest, Hungary, 09-11 Jun 2011. PMLR.
- [2] L. V. Ahlfors. Complex Analysis . McGraw-Hill Book Company, 2 edition, 1966.
- [3] Randal W. Beard, George N. Saridis, and John T. Wen. Galerkin approximations of the generalized hamilton-jacobi-bellman equation. Automatica , 33(12):2159-2177, 1997. ISSN 0005-1098. doi: https://doi.org/10.1016/S0005-1098(97)00128-3.
- [4] Julian Berberich, Johannes Köhler, Matthias Muller, and Frank Allgöwer. Data-driven model predictive control with stability and robustness guarantees. IEEE Transactions on Automatic Control , PP:1-1, 06 2020. doi: 10.1109/TAC.2020.3000182.
- [5] Leila Bouazza, Benjamin Mourllion, Abdenacer Makhlouf, and Abderazik Birouche. Controllability and observability of formal perturbed linear time invariant systems. International Journal of Dynamics and Control , 9, 12 2021. doi: 10.1007/s40435-021-00786-4.
- [6] Steven Bradtke, B.Erik Ydstie, and Andrew Barto. Adaptive linear quadratic control using policy iteration. Proceedings of the American Control Conference , 3, 09 1994. doi: 10.1109/ ACC.1994.735224.
- [7] Han-Fu Chen and Ji-Feng Zhang. Convergence rates in stochastic adaptive tracking. International Journal of Control , 49(6):1915-1935, 1989. doi: 10.1080/00207178908559752.
- [8] Kaiwen Chen and Alessandro Astolfi. Adaptive control for systems with time-varying parameters. IEEE Transactions on Automatic Control , 66(5):1986-2001, 2021. doi: 10.1109/TAC.2020.3046141.
- [9] Xinyi Chen and Elad Hazan. Black-box control for linear dynamical systems. CoRR , abs/2007.06650, 2020.
- [10] Alon Cohen, Tomer Koren, and Yishay Mansour. Learning linear-quadratic regulators efficiently with only √ T regret. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 1300-1309. PMLR, 09-15 Jun 2019.
- [11] Claudio De Persis and Pietro Tesi. Formulas for data-driven control: Stabilization, optimality, and robustness. IEEE Transactions on Automatic Control , 65(3):909-924, 2020. doi: 10.1109/ TAC.2019.2959924.
- [12] N. et al. Dean S. Mania, H. Matni. On the sample complexity of the linear quadratic regulator. Foundations of Computational Mathematics , 20:633-679, 2020. doi: https://doi.org/10.1007/ s10208-019-09426-y.
- [13] Geir E. Dullerud and Fernando Paganini. A Course in Robust Control Theory . Springer, New York, NY, 2000.
- [14] Mohamad Kazem Shirani Faradonbeh. Non-Asymptotic Adaptive Control of Linear-Quadratic Systems . PhD thesis, University of Michigan, 2017.

- [15] Salar Fattahi. Learning partially observed linear dynamical systems from logarithmic number of samples. In Ali Jadbabaie, John Lygeros, George J. Pappas, Pablo A. Parrilo, Benjamin Recht, Claire J. Tomlin, and Melanie N. Zeilinger, editors, Proceedings of the 3rd Conference on Learning for Dynamics and Control , volume 144 of Proceedings of Machine Learning Research , pages 60-72. PMLR, 07 - 08 June 2021.
- [16] Maryam Fazel, Rong Ge, Sham M. Kakade, and Mehran Mesbahi. Global convergence of policy gradient methods for the linear quadratic regulator, 2019.
- [17] Roger A. Horn and Charles R. Johnson. Matrix Analysis . Cambridge University Press, 1985.
- [18] Roger A. Horn and Charles R. Johnson. Matrix Analysis . Cambridge University Press, USA, 2nd edition, 2012. ISBN 0521548233.
- [19] Yang Hu, Adam Wierman, and Guannan Qu. On the sample complexity of stabilizing lti systems on a single trajectory. In Neurips , pages 1-2, 2022. doi: 10.1109/Allerton49937.2022.9929403.
- [20] Morteza Ibrahimi, Adel Javanmard, and Benjamin Van Roy. Efficient reinforcement learning for high dimensional linear quadratic systems. In Neural Information Processing Systems , 2012.
- [21] Joao Paulo Jansch-Porto, Bin Hu, and Geir E. Dullerud. Policy learning of mdps with mixed continuous/discrete variables: A case study on model-free control of markovian jump systems. CoRR , abs/2006.03116, 2020. URL https://arxiv.org/abs/2006.03116 .
- [22] Zhong-Ping Jiang and Yuan Wang. A converse lyapunov theorem for discrete-time systems with disturbances. Systems &amp; Control Letters , 45(1):49-58, 2002. ISSN 0167-6911. doi: https://doi.org/10.1016/S0167-6911(01)00164-5.
- [23] Joe Juang and Richard Pappa. An eigensystem realization algorithm for modal parameter identification and model reduction. Journal of Guidance Control and Dynamics , 8, 11 1985. doi: 10.2514/3.20031.
- [24] Karl Krauth, Stephen Tu, and Benjamin Recht. Finite-time analysis of approximate policy iteration for the linear quadratic regulator. Advances in Neural Information Processing Systems , 32, 2019.
- [25] S.M. Kusii. Stabilization and attenuation of bounded perturbations in discrete control systems. Journal of Mathematical Sciences , 2018. doi: https://doi.org/10.1007/s10958-018-3848-3.
- [26] T.L Lai. Asymptotically efficient adaptive control in stochastic regression models. Advances in Applied Mathematics , 7(1):23-45, 1986. ISSN 0196-8858. doi: https://doi.org/10.1016/ 0196-8858(86)90004-7.
- [27] Tze Leung Lai and Zhiliang Ying. Parallel recursive algorithms in asymptotically efficient adaptive control of linear stochastic systems. SIAM Journal on Control and Optimization , 29 (5):1091-1127, 1991.
- [28] Sahin Lale, Kamyar Azizzadenesheli, Babak Hassibi, and Anima Anandkumar. Explore more and improve regret in linear quadratic regulators. CoRR , abs/2007.12291, 2020. URL https://arxiv.org/abs/2007.12291 .
- [29] Bruce D. Lee, Anders Rantzer, and Nikolai Matni. Nonasymptotic regret analysis of adaptive linear quadratic control with model misspecification, 2023.
- [30] Yingying Li, Yujie Tang, Runyu Zhang, and Na Li. Distributed reinforcement learning for decentralized linear quadratic control: A derivative-free policy optimization approach. IEEE Transactions on Automatic Control , 67(12):6429-6444, 2022. doi: 10.1109/TAC.2021.3128592.
- [31] Wenjie Liu, Yifei Li, Jian Sun, Gang Wang, and Jie Chen. Robust control of unknown switched linear systems from noisy data. ArXiv , abs/2311.11300, 2023.
- [32] Horia Mania, Stephen Tu, and Benjamin Recht. Certainty equivalence is efficient for linear quadratic control. Proceedings of the 33rd International Conference on Neural Information Processing Systems , 2019.

- [33] Samet Oymak and Necmiye Ozay. Non-asymptotic identification of lti systems from a single trajectory. 2019 American Control Conference (ACC) , pages 5655-5661, 2018.
- [34] Samet Oymak and Necmiye Ozay. Non-asymptotic identification of lti systems from a single trajectory. 2019 American Control Conference (ACC) , pages 5655-5661, 2018. URL https: //api.semanticscholar.org/CorpusID:49275079 .
- [35] B. Pasik-Duncan. Adaptive control [second edition, by karl j. astrom and bjorn wittenmark, addison wesley (1995)]. IEEE Control Systems Magazine , 16(2):87-, 1996.
- [36] Juan C. Perdomo, Jack Umenberger, and Max Simchowitz. Stabilizing dynamical systems via policy gradient methods. ArXiv , abs/2110.06418, 2021.
- [37] Jing Sun Petros A. Ioannou. Robust adaptive control. Autom. , 37(5):793-795, 2001.
- [38] Orestis Plevrakis and Elad Hazan. Geometric exploration for online control. In Proceedings of the 34th International Conference on Neural Information Processing Systems , NIPS'20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.
- [39] Mehran Pourgholi, Mohsen Mohammadzadeh Gilarlue, Touraj Vahdaini, and Mohammad Azarbonyad. Influence of hankel matrix dimension on system identification of structures using stochastic subspace algorithms. Mechanical Systems and Signal Processing , 186:109893, 2023. ISSN 0888-3270. doi: https://doi.org/10.1016/j.ymssp.2022.109893.
- [40] Tuhin Sarkar and Alexander Rakhlin. Near optimal finite time identification of arbitrary linear dynamical systems. In International Conference on Machine Learning , 2018.
- [41] Tuhin Sarkar, Alexander Rakhlin, and Munther A. Dahleh. Finite-time system identification for partially observed lti systems of unknown order. ArXiv , abs/1902.01848, 2019.
- [42] Max Simchowitz, Horia Mania, Stephen Tu, Michael I. Jordan, and Benjamin Recht. Learning without mixing: Towards a sharp analysis of linear system identification. Annual Conference Computational Learning Theory , 2018.
- [43] Yue Sun, Samet Oymak, and Maryam Fazel. Finite sample system identification: Optimal rates and the role of regularization. In Alexandre M. Bayen, Ali Jadbabaie, George Pappas, Pablo A. Parrilo, Benjamin Recht, Claire Tomlin, and Melanie Zeilinger, editors, Proceedings of the 2nd Conference on Learning for Dynamics and Control , volume 120 of Proceedings of Machine Learning Research , pages 16-25. PMLR, 10-11 Jun 2020.
- [44] Anastasios Tsiamis and George J. Pappas. Linear systems can be hard to learn. In 2021 60th IEEE Conference on Decision and Control (CDC) , pages 2903-2910, 2021. doi: 10.1109/ CDC45484.2021.9682778.
- [45] Stephen Tu and Benjamin Recht. The gap between model-based and model-free methods on the linear quadratic regulator: An asymptotic viewpoint. In Annual Conference Computational Learning Theory , 2018.
- [46] Wei Wang, Xiangpeng Xie, and Changyang Feng. Model-free finite-horizon optimal tracking control of discrete-time linear systems. Appl. Math. Comput. , 433(C), nov 2022. ISSN 00963003. doi: 10.1016/j.amc.2022.127400.
- [47] Yu Xing, Benjamin Gravell, Xingkang He, Karl Henrik Johansson, and Tyler H. Summers. Identification of linear systems with multiplicative noise from multiple trajectory data. Automatica , 144:110486, 2022. ISSN 0005-1098. doi: https://doi.org/10.1016/j.automatica.2022.110486.
- [48] Xiong Zeng, Zexiang Liu, Zhe Du, Necmiye Ozay, and Mario Sznaier. On the hardness of learning to stabilize linear systems. 2023 62nd IEEE Conference on Decision and Control (CDC) , pages 6622-6628, 2023.
- [49] Kaiqing Zhang, Bin Hu, and Tamer Basar. Policy optimization for H 2 linear control with H ∞ robustness guarantee: Implicit regularization and global convergence. In Alexandre M. Bayen, Ali Jadbabaie, George Pappas, Pablo A. Parrilo, Benjamin Recht, Claire Tomlin, and Melanie Zeilinger, editors, Proceedings of the 2nd Conference on Learning for Dynamics and Control , volume 120 of Proceedings of Machine Learning Research , pages 179-190. PMLR, 10-11 Jun 2020.

- [50] Ziyi Zhang, Yorie Nakahira, and Guannan Qu. Learning to stabilize unknown LTI systems on a single trajectory under stochastic noise. In The 41st Conference on Uncertainty in Artificial Intelligence , 2025.
- [51] Yang Zheng and Na Li. Non-asymptotic identification of linear dynamical systems using multiple trajectories. IEEE Control Systems Letters , PP:1-1, 12 2020. doi: 10.1109/LCSYS. 2020.3042924.
- [52] Yang Zheng, Luca Furieri, Maryam Kamgarpour, and N. Li. Sample complexity of linear quadratic gaussian (lqg) control for output feedback systems. Conference on Learning for Dynamics &amp; Control , 2020.
- [53] K. Zhou and J. C. Doyle. Essentials of Robust Control . Prentice-Hall, 1998.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction accurately reflects the main contributions, which is later more formally stated in main theorem and justified by proofs and simulations in the appendices.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: The paper extensively discusses the limitations of this branch of studies and discusses each assumption separately.

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

Justification: The paper offers an explicit statement of assumptions as well as a full proof.

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

Justification: The code is attached to the submission, which exactly reproduces the desired result. In the simulation section, we also extensively discuss how to reproduce the result.

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

Justification: The code is attached in the supplemental material.

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

Justification: This paper does not include experiments using outside dataset.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The graphs contain shaded areas reflecting one standard deviation above and below average.

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

Justification: The experiment is simple and is not device-dependent.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This is a general-purpose paper and does not have any immediate ethical interpretation.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper is for general control theory and does not have any immediate social impact.

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

Justification: This paper contains no risky data.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper does not use existing assets.

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

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This paper does not involve LLM usage.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Simulations

Figure 1: The above figure shows the length of rollouts needed to identify and stabilize an unstable system with the unstable dimension k = 5 . The solid line shows the average length of rollouts the learner takes to stabilize the system. The shaded area shows the standard deviation of the length of rollouts. The proposed method requires the shortest rollouts and has the smallest standard deviation.

<!-- image -->

Figure 2: The above figure shows the number of rollouts needed to identify and stabilize an unstable system with the unstable dimension k = 5 . The solid line shows the average number of rollouts the learner takes to stabilize the system. The shaded area shows the standard deviation of the number of rollouts. The proposed method requires the least number rollouts and has small standard deviation.

<!-- image -->

In this section, we compare our algorithm to the two existing benchmarks: nuclear norm regularization in Sun et al. [43] and the Ho-Kalman algorithm in Zheng and Li [51]. We use a more complicated system than (1):

<!-- formula-not-decoded -->

where there are both process noise w t and observation noise v t . we fix the unstable part of the LTI system to dimension 5 , i.e. k = 5 , and observe the effect of increasing dimension n on the system. For each dimension n , we randomly generate a matrix with k unstable eigenvalues λ i ∼ unif (1 . 1 , 2)

and n -k stable eigenvalues λ j ∼ unif (0 , 0 . 5) . The basis for these eigenvalues is also initialized randomly. In both parts of the simulations, we fix m = 4 and p = q = ⌊ T -4 2 ⌋ for the proposed algorithm, LTS-P.

In the first part of the simulations, we fix the number of rollouts to be 4 times the dimension of the system and let the system run with an increasing length of rollouts until a stabilizing controller is generated. The system is simulated in three different settings, each with noise w t , v t ∼ N (0 , σ ) , with σ = 0 . 4 , 0 . 6 , 0 . 8 . For each of the three algorithms, the simulation is repeated for 30 trials, and the result is shown in Figure 1. The proposed method requires the shortest rollouts. The algorithm in Sun et al. [43] roughly have the same length of rollouts across different dimensions. This is to be expected, as Sun et al. [43] only uses the last data point of each rollout, so a longer trajectory does not generate more data. However, unlike the proposed algorithm, Sun et al. [43] still needs to estimate the entire system, which is why it still needs a longer rollout. The length of the rollouts of Zheng and Li [51] grows linearly with the dimension of the system. We also see fluctuations of the length of rollouts required for stabilization in Figure 1 because the matrix generated for each n has a randomly generated eigenbasis, which influences the amount of data needed for stabilization, as shown in Theorem 5.3. Overall, this simulation shows that the length of rollouts of the proposed algorithm remains O ( poly ( k )) regardless of the dimension of the system n .

In the second part of the simulation, we fix the length of each trajectory to 70 and examine the number of trajectories needed before the system stabilizes. We do not increase the length of trajectory in this part of the simulation with increasing n , since Sun et al. [43] only uses the last data point, and does not show significant performance improvement with longer trajectories, as shown in the first part of the simulation, so it would be an unfair comparison if the trajectory is overly long. Similar to the first part, the system is simulated in three different settings and the result is shown in Figure 2. Overall, the proposed method requires the least number of trajectories. When the n ⪆ k , the proposed algorithm requires a similar number of data with Zheng and Li [51], as the proposed algorithm is the same to that in Zheng and Li [51] when k = n . When n ≫ k , the proposed algorithm outperforms both benchmarks.

Figure 3: The above figure shows the probability of stabilization for a randomly generated matrix with n = 15 and k = 5 .

<!-- image -->

In the last simulation, we fix a randomly generated matrix with dimension n = 15 and an unstable subspace of dimension k = 5 . We further set the variance of the process and observation noise to σ = 0 . 4 . As shown in Figure 3, the stabilization probability is most enhanced by adding more trajectories, but longer trajectories also demonstrate a significant effect for stabilization, as proven in Theorem 5.3.

## B Proof Outline

In this section, we will give a high-level overview of the key proof ideas for the main theorem. The full proof details can be found in Appendix C (for Step 1), Appendix D (for Step 2), and Appendix E (for Step 3).

Proof Structure. The proof is largely divided into three steps following the idea developed in Section 3. In the first step, we examine how accurately the learner can estimate the low-rank lifted Hankel matrix ˜ H . In the second step, we examine how accurately the learner estimates the transfer function of the unstable component from ˜ H . In the last step, we provide a stability guarantee.

Step 1. We show that ˆ ˜ H obtained in (19) is an accurate estimate of ˜ H .

Lemma B.1. With probability at least 1 -δ , the estimation ˆ ˜ H obtained in (19) satisfies

<!-- formula-not-decoded -->

where ˆ ϵ is the estimation error for ∥ H -ˆ H ∥ that decays in the order of O ( 1 √ M ) ; ˜ ϵ is the error ∥ ∥ ∥ H -˜ H ∥ ∥ ∥ that decays in the order of O (( | λ k +1 | +1 2 ) m ) . See Theorem C.2 and Lemma C.3 for the exact constants.

We later (in Step 2, 3) will show that for a sufficiently small ϵ , the robust controller produced by the algorithm will stabilize the system in (1). We will invoke Lemma B.1 at the end of the proof in Step 3 to pick m,p,q, M values that achieve the required ϵ (in Appendix E).

Step 2. In this step, we will show that N 1 , R 1 B,CQ 1 can be estimated up to an error of O ( ϵ ) and up to a similarity transformation, given that ∥ ∥ ∥ ˆ ˜ H -˜ H ∥ ∥ ∥ &lt; ϵ .

Lemma B.2. The estimated system dynamics ˆ N 1 satisfies the following bound for some invertible matrix ˆ S :

<!-- formula-not-decoded -->

where c 2 , c 1 are constants depending on system theoretic parameters, and ˜ o min ( m ) is a quantity that grows with m . See Lemma D.1 for the definition of these quantities.

Lemma B.3. Up to transformation ˆ S (same as the one in Lemma B.2), we can bound the difference between ˆ C and the unstable component of the system CQ 1 as follows:

<!-- formula-not-decoded -->

where ˜ c min ( m ) is a quantity that grows with m defined in Lemma D.1.

Lemma B.4. Up to transformation ˆ S -1 ( ˆ S is the same as the one in Lemma B.2), we can bound the difference between ˆ B and the unstable component of the system R 1 B as follows:

<!-- formula-not-decoded -->

Step 3. We show that, under sufficiently accurate estimation, the controller ˆ K ( z ) returned by the algorithm stabilizes plant F ( z ) and hence ∥ ˆ K ( z ) F ˆ K ( z ) ∥ H ∞ is well defined. This is done via the following helper proposition.

Proposition B.5. Recall the definition of the estimated transfer function in (25) . Suppose the estimation errors ϵ N , ϵ C , ϵ B are small enough such that sup | z | =1 ∥ ∥ ∥ F ( z ) -ˆ F ( z ) ∥ ∥ ∥ &lt; ϵ ∗ . Then, the following two statements hold: (a) If K ( z ) stabilizes F ( z ) and ∥ K ( z ) F K ( z ) ∥ H ∞ &lt; 1 ϵ ∗ , then K ( z ) stabilizes plant ˆ F ( z ) as well; (b) similarly, if ˆ K ( z ) stabilizes ˆ F ( z ) with ∥ ˆ K ( z ) ˆ F ˆ K ( z ) ∥ H ∞ &lt; 1 ϵ ∗ , then ˆ K ( z ) stabilizes plant F ( z ) as well.

Then, we upper bound ∥ ˆ K ( z ) F ˆ K ( z ) ∥ H ∞ and use small gain theorem (Lemma 3.1) to show that the stable mode ∆( z ) does not affect stability either, and therefore, the controller ˆ K ( z ) stabilizes the full system, F full ( z ) in (4). Leveraging the error bounds in Step 2 and Step 1, we will also provide the final sample complexity bound. The detailed proof can be found in Appendix E.

## C Proof of Step 1: Bounding Hankel Matrix Error

To further simplify notation, define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so (17) can be simplified to

<!-- formula-not-decoded -->

respectively. An analytic solution to (30) is ˆ Φ := Y U † , where U † := U ∗ ( UU ∗ ) -1 is the right psuedo-inverse of U . In this paper, we will use ˆ ˜ Φ to recover ˜ H , and we want to bound the error between ˜ H and ˆ ˜ H .

<!-- formula-not-decoded -->

where the third equality used (16). We can bound the estimation error (31) by

<!-- formula-not-decoded -->

We then prove a lemma that links the difference between Φ and Hankel matrix H .

Lemma C.1. Given two Hankel matrices H ♣ , H ♢ constructed from Φ ♣ , Φ ♢ with (18) , respectively, then

<!-- formula-not-decoded -->

Proof. From the construction in (18), we see that each column of H ♣ -H ♢ is a submatrix of Φ ♣ 2+ m : T -Φ ♢ 2+ m : T , then

<!-- formula-not-decoded -->

Similarly, each row of H ♣ -H ♢ is a submatrix of Φ ♣ 2+ m : T -Φ ♢ 2+ m : T . Therefore,

<!-- formula-not-decoded -->

Combining (33) and (34) gives the desired result.

We then bound the gap between H ( k ) and ˆ ˜ H by adapting Theorem 3.1 of Zheng and Li [51]. For completeness, we put the theorem and proof below:

Theorem C.2. For any δ &gt; 0 , the following inequality holds when M &gt; 8 d u T + 4( d u + d y + 4) log(3 T/δ ) with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof. By Lemma C.1, we have

<!-- formula-not-decoded -->

Bounding ∥ ( UU ∗ ) ∥ -1 and ∥ vU ∗ ∥ with Proposition 3.1 and Proposition 3.2 of [51] gives the result in the theorem statement.

## Lemma C.3.

<!-- formula-not-decoded -->

Proof. We first bound H and ˜ H entry-wise,

<!-- formula-not-decoded -->

Applying Gelfand's formula (Lemma H.2), we obtain

<!-- formula-not-decoded -->

Moreover, we have the following well-known inequality (see, for example, [17]), for Λ ∈ R n 1 × n 2 ,

<!-- formula-not-decoded -->

Applying (36) to (35) gives us the desired inequality.

We are now ready to prove Lemma B.1.

Proof of Lemma B.1. We first provide a bound on ˆ ˜ H -ˆ H . As ˆ ˜ H is the rankk approximation of ˆ H , we have ∥ ∥ ∥ ˆ ˜ H -ˆ H ∥ ∥ ∥ ≤ ∥ ∥ ∥ ˜ H -ˆ H ∥ ∥ ∥ , then

<!-- formula-not-decoded -->

where we use Theorem C.2 and Lemma C.3. Therefore, we obtain the following bound

<!-- formula-not-decoded -->

## D Proof of Step 2: Bounding Dynamics Error

In the previous section, we proved Lemma B.1 and provided a bound for ∥ ˆ ˜ H -˜ H ∥ . In this section, we use this bound to derive the error bound for our system estimation. We start with the following lemma that bounds the difference between ˜ O and ˆ O and the difference between ˜ C and ˆ C up to a transformation.

Lemma D.1. Given N 1 is diagnalizable, and recall N 1 = P 1 Λ P -1 1 denotes the eigenvalue decomposition of N 1 , where Λ = diag( λ 1 , . . . , λ k ) is the diagonal matrix of eigenvalues. Consider ˆ ˜ H = ˆ O ˆ C and ˜ H = ˜ O ˜ C , where we recall ˆ O , ˆ C are obtained by doing SVD ˆ ˜ H = ˆ U ˆ Σ ˆ V ∗ and setting ˆ O = ˆ U ˆ Σ 1 / 2 , ˆ C = ˆ Σ 1 / 2 ˆ V ∗ (cf. (20) ). Suppose ∥ ∆ H ∥ = ∥ ∥ ∥ ˜ H -ˆ ˜ H ∥ ∥ ∥ ≤ ϵ , then there exists an invertible k -byk matrix ˆ S such that

<!-- formula-not-decoded -->

(b) there exists c 1 , c 2 &gt; 0 such that c 1 I ⪯ ˆ S ⪯ c 2 I , where given σ s , σ s in (49) ,

<!-- formula-not-decoded -->

(c) we can bound σ min ( ˜ O ) and σ min ( ˜ C )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Proof of part(a) . Recall the definition of ˜ H

<!-- formula-not-decoded -->

We can represent ˜ O and ˜ C as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, note that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

where we have defined the matrix ˆ S above. We can also expand ˆ S as follows:

<!-- formula-not-decoded -->

Further, the last term in the last equation in (42) can be written as,

<!-- formula-not-decoded -->

Putting (42), (44) together, we have and

<!-- formula-not-decoded -->

which proves one side of part (a). To finish part(a), we are left to bound ∥ ∥ ∥ ˜ C ˆ S -1 ˆ C ∥ ∥ ∥ . With the expression of ˆ S in (43), we obtain

<!-- formula-not-decoded -->

Let E := ˆ O ˆ S -˜ O so ˜ O = ˆ O ˆ S -E . By (42) and (44), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we can obtain the following inequalities:

<!-- formula-not-decoded -->

Substituting (45), we get

<!-- formula-not-decoded -->

where the last inequality requires

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which we will verify at the end of the proof of Theorem 5.3 that our selection of algorithm parameters guarantees is true.

Proof of part(b) . We first examine the relationship between G 1 and G 2 . Recall α is the maximum of the controllability and observability index. Therefore, the matrix

<!-- formula-not-decoded -->

which is exactly G ∗ con ( P ∗ 1 ) -1 , is full column rank. For any full column rank matrix Γ , let's Γ † = (Γ ∗ Γ) -1 Γ ∗ denote its Moore-Penrose inverse that satisfies Γ † Γ = I . With this, for nonnegative integer j we define the following matrix, which is essentially [ G 1 ] jα :( j +1) α -1 'divided by' [ G 2 ] jα :( j +1) α -1 in the Moore-Penrose sense:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As in the main Theorem 5.3 we assumed the N 1 is diagonalizable, Λ is a diagonal matrix with possibly complex entries. As such, Λ m 2 + jα (Λ -m 2 -jα ) ∗ is a diagonal matrix with all entries having modulus 1 . 3 Therefore, we have

<!-- formula-not-decoded -->

for all j . Recall the condition in the main theorem 5.3 requires p = q . Without loss of generality, we can assume α divides p and q , 4 then G 1 and G 2 can be related in the following way:

<!-- formula-not-decoded -->

With the above relation, we are now ready to bound the norm of ˆ S . We calculate

<!-- formula-not-decoded -->

If we examine the middle term ˆ V ˆ Σ ˆ V ∗ , we can expand it as

<!-- formula-not-decoded -->

Using Lemma H.3 on (52), we obtain

<!-- formula-not-decoded -->

Therefore, we can bound ˆ V ˆ Σ ˆ V ∗ :

<!-- formula-not-decoded -->

3 This is the only place where the diagonalizability of N 1 is used in the proof of Theorem 5.3. If N 1 is not diagonalizable, Λ will be block diagonal consisting of Jordan blocks, and we can still bound Λ m 2 + jα (Λ -m 2 -jα ) ∗ with dependence on the size of the Jordan block. See Appendix G for more details.

4 If α does not divide p or q , then the last block in (50) will take a slightly different form but can still be bounded similarly.

where we used σ 2 s I ⪯ S ∗ CB S CB ⪯ σ 2 s I and σ min ( S CB ) ≥ σ s , as S CB is the block-diagonal matrix of S ( j ) CB (cf. (49), (50)). Therefore, plugging (53) into (51), we are ready to bound ˆ S ∗ ˆ S has follows:

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

If we further pick ϵ such that

<!-- formula-not-decoded -->

(54) and (55) can be simplified as

<!-- formula-not-decoded -->

Proof of part (c). Observe that ˜ O 1: α = G ob N m/ 2 1 . Therefore,

σ

Similarly, we have

<!-- formula-not-decoded -->

We are now ready to prove Lemma B.2

Proof of Lemma B.2. Let E = ˜ O ˆ O ˆ S , by Lemma D.1, ∥ E ∥ ≤ ϵ σ min ( ˜ C ) . Moreover, ˜ O 1: p -1 = ˆ O 1: p -1 ˆ S + E 1: p -1 and ˜ O 2: p = ˆ O 2: p ˆ S + E 2: p . Given that ˜ O 2: p = ˜ O 1: p -1 N 1 , we have N 1 = ( ˜ O ∗ 1: p -1 ˜ O 1: p -1 ) -1 ˜ O ∗ 1: p -1 ˜ O 2: p , where we have used by the condition in the main theorem,

<!-- formula-not-decoded -->

so that ˜ O 1: p -1 is full column rank. Recall N 1 is estimated by

<!-- formula-not-decoded -->

Given that ˜ O 2: p = ˜ O 1: p -1 N 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ E := E 1: p -1 N 1 ˆ S -1 -E 2: p ˆ S -1 . Substituting (62) into (59), we obtain

<!-- formula-not-decoded -->

min

O

( ˜

)

≥

σ

min

O

( ˜

1:

α

)

≥

σ

min

(

G

ob

)

σ

min

(

N

m/

1

2

) := ˜

o

min

(

m

)

.

Therefore,

<!-- formula-not-decoded -->

where the last inequality requires (47).

Following similar arguments from the above proof for the estimation of ̂ N m 2 1 ,O in (22), we have the following collarary.

## Corollary D.2. We have

<!-- formula-not-decoded -->

Proof. Let E = ˜ O ˆ O ˆ S , then ∥ E ∥ ≤ ϵ σ min ( ˜ C ) . Moreover, ˜ O 1: p -m 2 = ˆ O 1: p -m 2 ˆ S + E 1: p -m 2 and ˜ O m 2 +1: p = ˆ O m 2 +1: p ˆ S + E m 2 +1: p . Given that ˜ O m 2 +1: p = ˜ O 1: p -m 2 N m 2 1 , we have N m 2 1 = ( ˜ O ∗ 1: p -m 2 ˜ O 1: p -m 2 ) -1 ˜ O ∗ 1: p -m 2 ˜ O m 2 +1: p , where we have used by the condition in the main theorem,

<!-- formula-not-decoded -->

so that ˜ O 1: p -m 2 is full column rank. Recall in (22), we estimate N m 2 1 by

<!-- formula-not-decoded -->

Given that ˜ O m 2 +1: p = ˜ O 1: p -m 2 N m 2 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ E := E 1: p -m 2 N m 2 1 ˆ S -1 -E m 2 +1: p ˆ S -1 . Substituting (67) into (64), we obtain

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

where (70) requires

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which we will verify at the end of the proof of Theorem 5.3, and (71) requires (47).

By using an idental argument, we can also prove a similar corollary for ̂ N m 2 1 ,C . The proof is omitted. Corollary D.3. Assuming p, q &gt; m 2 +1 , we have

<!-- formula-not-decoded -->

With the above two corallaries, we are now ready to prove Lemma B.3.

Proof of Lemma B.3. By the definition of ˜ O in (7), we have

<!-- formula-not-decoded -->

Furthermore, we recall by the way ˆ C is estimated in (24), we obtain

<!-- formula-not-decoded -->

Let E O 1 := ˜ O 1 -ˆ O 1 ˆ S . Substituting it in (74) leads to

<!-- formula-not-decoded -->

Substituting in (73), we have

<!-- formula-not-decoded -->

By Corollary D.2, we obtain

<!-- formula-not-decoded -->

where (76) is by Lemma D.1 and requires (72), and (78) requires

<!-- formula-not-decoded -->

The proof of Lemma B.4 undergoes a very similar procedure.

Proof of Lemma B.4. By the definition of ˜ C in (7), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let E C 1 := ˜ C 1 -ˆ S -1 ˆ C 1 . Substitute in (81), we get

<!-- formula-not-decoded -->

Substitute in (80), we obtain

<!-- formula-not-decoded -->

By Corollary D.3, we obtain

<!-- formula-not-decoded -->

where (83) is by Lemma D.1 and requires (72), and (85) requires

<!-- formula-not-decoded -->

Further, by (24), we have

## E Proof of Step 3

Before we analyze the analytic condition of the transfer functions, we need to analyze the values of F ( z ) and ˆ F ( z ) on the unit circle through the following lemma:

Lemma E.1. We have

<!-- formula-not-decoded -->

Proof. We have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

Substituting (88) into (87), we obtain

<!-- formula-not-decoded -->

(90)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step uses inf | z | =1 σ min ( zI -N 1 ) = 1 sup | z | =1 ∥ ( zI -N 1 ) -1 ∥ ≥ | λ k |-1 L . Eq. (90) uses

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will verify (91) and (92) at the end of the proof of Theorem 5.3 to make sure it is met by our choice of algorithm parameters. Lastly, substituting max( ∥ R 1 B ∥ , ∥ CQ 1 ∥ ) ≤ L and using sup | z | =1 ∥ ( zI -N 1 ) -1 ∥ ≤ L | λ k |-1 give the desired result.

We are now ready to prove Proposition B.5,

Proof of Proposition B.5. Weonly prove part (a). The proof of part (b) is similar and is hence omitted. Suppose K ( z ) is a controller that satisfies the condition in part(a). Our goal is to show this K ( z ) stabilizes ˆ F ( z ) , or in other words, ˆ F K ( z ) is analytic on C ≥ 1 . Let ∆ F ( z ) := F ( z ) -ˆ F ( z ) , we have

<!-- formula-not-decoded -->

Note that to show that ( I -ˆ F ( z ) K ( z )) -1 is analytic on C ≥ 1 , we only need to show ( I -ˆ F ( z ) K ( z )) has the same number of zeros on C ≥ 1 as that of I -F ( z ) K ( z ) , because we know ( I -F ( z ) K ( z )) is non-singular (i.e. has no zeros) on C ≥ 1 . Given (91), we also know I -F ( z ) K ( z ) and I -ˆ F ( z ) K ( z ) has the same number of poles on C ≥ 1 . By Rouche's Theorem (Theorem H.4), I -ˆ F ( z ) K ( z ) has the same number of zeros on C ≥ 1 as I -F ( z ) K ( z ) if

<!-- formula-not-decoded -->

where the last condition is satisfied by the condition in part (a) of this proposition. Therefore, I -ˆ F ( z ) K ( z ) has no zeros on C ≥ 1 and the proof is concluded.

We are now ready to prove Theorem 5.3.

Proof of Theorem 5.3. Suppose for now the number of samples is large enough such that the condition in Proposition B.5 holds: sup | z | =1 ∥ ∥ ∥ F ( z ) -ˆ F ( z ) ∥ ∥ ∥ &lt; ϵ ∗ . At the end of the proof we will provide a sample complexity bound for which this holds.

By Assumption 5.2, there exists K ( z ) that stabilizes F ( z ) with

<!-- formula-not-decoded -->

Therefore, by Proposition B.5, K ( z ) stabilizes plant ˆ F ( z ) as well. Therefore, ∥ K ( z ) ˆ F K ( z ) ∥ H ∞ is well defined and can be evaluated by only taking sup over | z | = 1 :

<!-- formula-not-decoded -->

where we substituted in Assumption 5.2 in the last inequality. In Algorithm 1, we find a controller ˆ K ( z ) that stabilizes ˆ F ( z ) and minimizes ∥ ˆ K ( z ) ˆ F ˆ K ( z ) ∥ H ∞ . As K ( z ) stabilizes ˆ F ( z ) and satisfies the upper bound in (93), the ˆ K ( z ) returned by the algorithm must stabilize ˆ F ( z ) and is such that ∥ ˆ K ( z ) ˆ F ˆ K ( z ) ∥ H ∞ is upper bounded by the RHS of (93). By Proposition B.5, we know ˆ K ( z ) stabilizes F ( z ) as well. Therefore, ∥ ˆ K ( z ) F ˆ K ( z ) ∥ H ∞ is well defined and can be evaluated on taking the sup over | z | = 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As ˆ K ( z ) is such that ∥ ˆ K ( z ) ˆ F ˆ K ( z ) ∥ H ∞ satisfies the upper bound by the RHS in (93), we have

<!-- formula-not-decoded -->

Therefore, by small gain theorem (Lemma 3.1), we know the controller ˆ K ( z ) stabilizes the original full system.

For the remaining of the proof, we analyze the number of samples required to satisfy the condition in Proposition B.5, i.e.

<!-- formula-not-decoded -->

where the middle step is by Lemma E.1. By Lemma B.2, Lemma B.4 and Lemma B.3, we obtain the following sufficient condition,

<!-- formula-not-decoded -->

By simple computation, we obtain the following sufficient condition:

<!-- formula-not-decoded -->

which can be simplified into the following sufficient condition:

<!-- formula-not-decoded -->

Moreover, we also need to satisfy (47),(56),(79),(86),(91),(92), which requires

<!-- formula-not-decoded -->

A sufficient condition that merges both (96) and (97) is:

<!-- formula-not-decoded -->

Plugging in the definition of c 1 , c 2 , ˜ o min ( m ) , ˜ c min ( m ) , we have c 1 c 2 = √ σ min ( P 1 ) 2 σ s 8 σ max ( P 1 ) 2 ¯ σ s =

<!-- formula-not-decoded -->

dition number of a matrix. Further, as ∥ ( N -1 1 ) m/ 2 ∥ ≤ L ( 1 | λ k | +1 2 ) m/ 2 , we have σ min ( N m/ 2 1 ) ≥ 1 L ( 2 1 | λ k | +1 ) m/ 2 . Therefore, the condition (98) can be replaced with the following sufficient condition

<!-- formula-not-decoded -->

where the ≲ above only hides numerical constants.

To meet (58), (63), we set p = q = max( α + m 2 , m ) . We also set m large enough so that the right hand side of (99) is lower bounded by √ m 3 . Using the simple fact that for a &gt; 1 , a m/ 2 = √ a m/ 2 √ a m/ 2 ≥ √ a m/ 2 m 1 . 5 e 1 . 5 ( 1 6 log a ) 1 . 5 , and replacing a with 2 | λ k | | λ k | +1 , a sufficient condition for such an m is

<!-- formula-not-decoded -->

Recall that ϵ = 2ˆ ϵ +2˜ ϵ . With the above condition on m , it now suffices to require max(ˆ ϵ, ˜ ϵ ) ≲ √ m 3 . Plugging in the definition of ˆ ϵ, ˜ ϵ in Theorem C.2,Lemma C.3, we need

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To satisfy (102), recall T = m + p + q +2 . Let us also require

<!-- formula-not-decoded -->

so that p = q = max( α + m 2 , m ) = m , so we have T ( T +1)min( p, q ) ≲ m 3 . Therefore, it suffices to require M ≳ ( σ v σ u ) 2 ( d u + d y ) log m δ to satisfy (102).

To satisfy (103), we need

<!-- formula-not-decoded -->

To summarize, collecting the requirements on m (100),(104),(105) and also (72), we have the final complexity on m :

<!-- formula-not-decoded -->

The final requirement on p, q is,

<!-- formula-not-decoded -->

The final complexity on the number of trajectories is

<!-- formula-not-decoded -->

Lastly, in the most interesting regime that | λ k | -1 , 1 -| λ k +1 | , ϵ ∗ is small, we have log 2 | λ k | | λ k | +1 ≍ | λ k | -1 , min(( | λ k | -1) 2 , | λ k | -1) = ( | λ k | -1) 2 , min( ϵ ∗ , 1) = ϵ ∗ . In this case, (106) can be simplified:

<!-- formula-not-decoded -->

̸

## F System identification when D = 0

̸

In the case when D = 0 in (1), the transfer function becomes

<!-- formula-not-decoded -->

and the estimated transfer function becomes ˆ F ( z ) = ˆ D + ˆ C ˆ S ( zI -ˆ S -1 ˆ N 1 ˆ S ) -1 ˆ S -1 ˆ B , so the gap between estimated transfer function and the ground-truth transfer function can be bounded as follows

<!-- formula-not-decoded -->

The second term in (112) can be bounded by Lemma E.1. Therefore, in this section, we focus on providing a bound for ∥ ∥ ∥ D -ˆ D ∥ ∥ ∥ . The recursive relationship of the process and measurement data will change from (14) to the following:

<!-- formula-not-decoded -->

Therefore, we can estimate each block of the Hankel matrix as follows:

<!-- formula-not-decoded -->

from which we can easily obtain the D matrix and use Φ to design the controller, as in the rest of the main text. Fortunately, from (113), we see that Φ D can also be estimated via (17), i.e.

<!-- formula-not-decoded -->

̸

In particular, we see that even in the case where D = 0 , the estimation error of D does not affect the estimation of Φ or ˆ ˜ Φ . Therefore, the error bound of N 1 , CQ 1 , R 1 B in Lemma B.2, Lemma B.3,Lemma B.4 still holds. The error of estimating D can be bounded as follows:

Lemma F.1 (Corollary 3.1 of [51]) . Let ˆ D denote the first block submatrix in (115) , then

<!-- formula-not-decoded -->

where L is a constant depending on A,B,C,D , the dimension constants n, m, d u , d y , and the variance of control and system noise σ u , σ v , σ w .

Therefore, we have proved a bound in the place of Lemma E.1. The rest of the proof will follow the same line as Appendix E.

## G Bounding ˆ S when N 1 is not diagonalizable

In proving Theorem 5.3, we used the diagonalizability assumption in Lemma D.1. More specifically, it was used in (48) to upper and lower bound Λ m 2 + jα (Λ -m 2 -jα ) ∗ for j = 0 , 1 , . . . , p α , which is reflected in the value of ¯ σ s , σ s . In the diagonalizable case, Λ m 2 + jα (Λ -m 2 -jα ) ∗ is a diagonal matrix with all entries having modulus 1 regardless of the value of j . Now in the non-diagonalizable case, we bound Λ m 2 + jα (Λ -m 2 -jα ) ∗ , provide new values for ¯ σ s , σ s , and analyze how it affect the final sample complexity.

Consider

<!-- formula-not-decoded -->

where k J is the number of Jordan blocks, and each J i is a Jordan block that is either a scalar or a square matrix with eigenvalues on the diagonal, 1 's on superdiagona1, and zeros everywhere else.

Without loss of generality, assume J 1 is the largest Jordan block with eigenvalue λ satisfying | λ | &gt; 1 and size γ , then J 1 = λI +Γ , where Γ is the nilpotent super-diagonal matrix of 1 's such that Γ i = 0 for all i ≥ γ . Therefore, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where ¯ λ is the conjugate of λ . Because λ m 2 + jα ¯ λ -m 2 -jα has modulus 1 , we have

<!-- formula-not-decoded -->

We can then calculate,

<!-- formula-not-decoded -->

where the second inequality requires m ≥ 4 γ , and the last inequality uses jα ≤ p = m . To calculate the smallest singular value, note that ∑ γ i =0 ( m 2 + jα i ) Γ i λ i is an upper triangular matrix with diagonal entries being 1 , therefore its smallest singular value is lower bounded by 1 . Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As such, the constants ¯ σ s , σ s in (49) need to be modified to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This will affect the sample complexity calculation step in (99), which will become

<!-- formula-not-decoded -->

where the only difference is the additional factor in the denominator highlighted in blue. As this factor is only polynomial in m , we can merge it with the exponential factor ( 2 | λ k | | λ k | +1 ) m/ 2 such that

<!-- formula-not-decoded -->

Therefore, (100) will be changed into:

<!-- formula-not-decoded -->

and lastly, the final simplified complexity on m will be changed into

<!-- formula-not-decoded -->

which only adds a multiplicative factor in γ and the additional γ in the log . Given such a change in the bound for m , the bound for other algorithm parameters p, q, M is the same (except for the changes caused by their dependance on m ).

## H Additional Helper Lemmas

Lemma H.1. Given Assumption 5.1 is satisfied, then N 1 , CQ 1 is observable, N 1 , R 1 B is controllable.

Proof. Let w denote any unit eigenvector of N 1 with eigenvalue λ , then

<!-- formula-not-decoded -->

Therefore, Q 1 w is an eigenvector of A . As A,C is observable, by PBH test, this leads to

<!-- formula-not-decoded -->

By PBH Test, this directly leads to ( N 1 , CQ 1 ) is observable.The controllability part is similar.

Lemma H.2 (Gelfand's formula) . For any square matrix X , we have

<!-- formula-not-decoded -->

In other words, for any ϵ &gt; 0 , there exists a constant ζ ϵ ( X ) such that

<!-- formula-not-decoded -->

Further, if X is invertible, let λ min ( X ) denote the eigenvalue of X with minimum modulus, then

<!-- formula-not-decoded -->

The proof can be found in existing literatures (e.g. [18].

Lemma H.3. For two matrices H,E , 1 2 H ∗ H -E ∗ E ⪯ ( H + E ) ∗ ( H + E ) ⪯ 2 H ∗ H +2 E ∗ E .

Proof. For the upper bound, notice that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

For the lower bound, we have

<!-- formula-not-decoded -->

where in the last inequality, we used (125). Rearrange the terms, we get the lower bound.

<!-- formula-not-decoded -->

Theorem H.4 (Rouche's theorem) . Let D ⊂ C be a simply connected domain, f and g two meromorphic functions on D with a finite set of zeros and poles F . Let γ be a positively oriented simple closed curve which avoids F and bounds a compact set K . If | f -g | &lt; | g | along γ , then

<!-- formula-not-decoded -->

where N f (resp. P f ) denotes the number of zeros (resp. poles) of f within K , counted with multiplicity (similarly for g ).

For proof, see e.g. [2].

## I Indexing

For the convenience of readers, we provide a table summarizing all constants appearing in the bounds.

Table 1: Lists of parameters and constants appearing in the bound.

| Constant   | Appearance   | Explanation                                                       |
|------------|--------------|-------------------------------------------------------------------|
| T          | Section 1    | Length of each roll out.                                          |
| M          | Section 3    | Number of roll outs.                                              |
| m          | Section 3    | number of open-loop steps for converging to unstable state space. |
| p, q       | Section 3    | dimension of Hankel matrix to be estimated.                       |

Table 2: System theoretic parameters.

| Constant                                                                                              | Appearance                                                                                       | Explanation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H O , C ˜ O , ˜ C ˜ H ˆ H ˆ ˜ H ˆ O , ˆ C G ob G con ˜ o min ( m ) ˜ c min ( m ) α C C ,C B ζ ϵ ( · ) | (5) (6) (7) (8) (18) (19) (20) Section 5 Section 5 Lemma D.1 Lemma D.1 Appendix C (48) Lemma H.2 | Ground truth Hankel matrix factorization of H . rank- k approximation of O , C , respectively. rank- k approximation of H . numerical approximation of H . rank- k approximation of ˆ H . factorization of ˆ ˜ H . Observability Gramian Controllability Gramian a positive real number such that σ min ( ˜ O ) ≥ ˜ o min ( m ) . a positive real number such that σ min ( ˜ C ) ≥ ˜ c min ( m ) . Controllability index. modified controllability and observability matrices. Gelfand constant for the norm of matrix exponents |

Table 3: Shorthand notations (introduced in proofs).

| Constant          | Appearance                                                                 | Explanation                                                                                                                                                                                                                                                                                                                                                            |
|-------------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| σ s σ ϵ ϵ ϵ ϵ ∗ ϵ | Lemma D.1 Lemma D.1 Lemma B.2 Lemma B.3 Lemma B.4 Assumption 5.2 Lemma B.1 | σ s := σ min ( G ob ) σ max ( G con ) σ min ( P 1 ) 2 σ s := σ max ( G ob ) σ min ( G con ) σ max ( P 1 ) 2 ϵ N := 4 Lc 2 ϵ c 1 ˜ o min ( m ) ϵ C := 2 ( 4 c 2 L c 1 ˜ o min ( m ) + 1 ˜ c min ( m ) ) ϵ ϵ B := 8 ( c 2 L c 1 ˜ c min ( m ) + 1 ˜ o min ( m ) ) ϵ ϵ := 2ˆ ϵ +2˜ ϵ 8 σ v √ T ( T +1)( d u + d y )min { p,q } log(27 σ u √ M 3 &#124; λ k +1 &#124; +1 m |
| s                 |                                                                            |                                                                                                                                                                                                                                                                                                                                                                        |
| N                 |                                                                            |                                                                                                                                                                                                                                                                                                                                                                        |
| C                 |                                                                            |                                                                                                                                                                                                                                                                                                                                                                        |
| B                 |                                                                            |                                                                                                                                                                                                                                                                                                                                                                        |
| ˆ ϵ               | Theorem C.2                                                                | := T/δ )                                                                                                                                                                                                                                                                                                                                                               |
| ˜ ϵ               | Lemma C.3                                                                  | ˜ ϵ := L ( 2 )                                                                                                                                                                                                                                                                                                                                                         |
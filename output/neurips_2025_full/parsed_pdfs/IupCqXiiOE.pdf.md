## Multi-agent Markov Entanglement

## Shuze Chen

Graduate School of Business Columbia University

New York, NY 10027

shuze.chen@columbia.edu

## Tianyi Peng

Graduate School of Business Columbia University New York, NY 10027

tianyi.peng@columbia.edu

## Abstract

Value decomposition has long been a fundamental technique in multi-agent reinforcement learning and dynamic programming. Specifically, the value function of a global state ( s 1 , s 2 , . . . , s N ) is often approximated as the sum of local functions: V ( s 1 , s 2 , . . . , s N ) ≈ ∑ N i =1 V i ( s i ) . This approach has found various applications in modern reinforcement learning systems. However, the theoretical justification for why this decomposition works so effectively remains underexplored. In this paper, we uncover the underlying mathematical structure that enables value decomposition. We demonstrate that a Markov decision process (MDP) permits value decomposition if and only if its transition matrix is not 'entangled'-a concept analogous to quantum entanglement in quantum physics. Drawing inspiration from how physicists measure quantum entanglement, we introduce how to measure the 'Markov entanglement' and show that this measure can be used to bound the decomposition error in general multi-agent MDPs. Using the concept of Markov entanglement, we proved that a widely-used class of policies, the index policy, is weakly-entangled and enjoys a sublinear O ( √ N ) scale of decomposition error for N -agent systems. Finally, we show Markov entanglement can be efficiently estimated, guiding practitioners on the feasibility of value decomposition.

## 1 Introduction

Learning the value function given certain policy, or policy evaluation , is one of the most fundamental tasks in RL. Significant attention has been paid to single-agent policy evaluation [39, 8, 40]. However, when it comes to multi-agent reinforcement learning (MARL), single-agent methodologies typically suffer from the curse of dimensionality : the state space of the system scales exponentially with the number of agents. To tackle this problem, one common technique is value decomposition,

<!-- formula-not-decoded -->

where V i is some local function that can be learned independently by each agent. It quickly follows that this decomposition greatly reduces the computation complexity from exponential to linear dependency on the number of agents N .

The remaining question is whether this decomposition is effective. This is non-trivial due to the coupling of agents-individual agent's action and transition depend on other agents. For example, in a ride-hailing platform, if one driver took the order, then other drivers are not allowed fulfill the same order. As a result, value decomposition may lose information and introduce bias without considering the global constraints.

In the past several decades, both positive and negative results have been reported. Back to the last century, [49, 47] apply Lagrange relaxations to decompose the global value and obtain the well-known

Whittle index policy. The Lagrange decomposition idea has also been proved successful in many other important multi-agent tasks such as network revenue management [1, 50], resource allocation [27, 7], and online matching [11, 12, 36, 28]. However, Lagrange decomposition relies on the knowledge of system dynamics and [2] show its decomposition error can be arbitrarily bad for general multi-agent MDPs. In more recent days, practitioners apply online (deep) reinforcement learning to train a local value function for each individual agent. This practice gives birth to state-of-the-art dispatching policies in ride-hailing platforms and has been well recognized by the operations research community, such as DiDi Chuxing [33] (Daniel H. Wagner Prize, 2020) and Lyft [4] (Franz Edelman Laureates, 2024). Intervention policies based on a similar value decomposition idea also demonstrate substantial empirical advantages and have been deployed by a behavioral health platform in Kenya [5] (Pierskalla Award, 2024). In broader MARL literature, value decomposition serves as one key component of centralized training and decentralized execution (CTDE) paradigm, achieving strong empirical performance [38, 29, 35]. However, recent research has started reflecting on the invalidity and potential flaw of value decomposition in practice [25, 16].

Despite all these empirical success and failures, there remains little theoretical understanding of whether and how we can decompose the value function in multi-agent MDPs.

## 1.1 This paper

In this paper, we will uncover the underlying mathematical structure that enables/disables value decomposition. Our new theoretical framework quantifies the inter-dependence of agents in multiagent MDPs and systematically characterizes the effectiveness of value decomposition. For simplicity, we will demonstrate the main results through two-agent MDPs indexed by agent A and B . We later extend our results to general N -agent MDPs in Appendix H.

We start with a trivial example where two agents are independent, i.e. each following independent MDPs. It's clear that the global value function can be decomposed as the sum of value functions of local MDPs. As two agents are independent, it holds P π ( s ′ A , s ′ B | s A , s B ) = P π ( s ′ A | s A ) · P π ( s ′ B | s B ) , or in matrix form,

<!-- formula-not-decoded -->

where ⊗ is the tensor product or Kronecker product of matrices. The important question is whether we can extend beyond this trivial case of independent subsystems.

A Sufficient and Necessary Condition We introduce a new condition called 'Markov Entanglement' to describe the intrinsic structure of transition dynamics in multi-agent MDPs.

Definition 1 (Markov Entanglement) . Consider a two-agent MDP with transition P π AB . If there exists

<!-- formula-not-decoded -->

then P π AB is separable; otherwise entangled.

Compared with the preceding example of independent subsystems, Markov entanglement offers an intuitive interpretation: a two-agent MDP is separable if it can be expressed as a linear combination of independent subsystems . We then demonstrate,

<!-- formula-not-decoded -->

where V π AB is decomposable if there exist local value functions V A , V B such that V π AB ( s A , s B ) = V A ( s A ) + V B ( s B ) for all ( s A , s B ) . This result sharply unravels the secret structure of system dynamics governing value decomposition. As a sufficient condition, our finding strictly generalizes the previous independent subsystem example, extending it to scenarios involving interacting and coupled agents. As a necessary condition, we prove that exact value decomposition under any reward kernel requires the system dynamics to be separable. Taken together, this result provides a complete characterization of when exact value function decomposition is possible in multi-agent MDPs.

More interestingly, our Markov entanglement condition turns out be a mathematical counterpart of quantum entanglement in quantum physics, whose definition is provided below.

Definition 2 (Quantum Entanglement) . Consider a two-party quantum state ρ AB . If there exists

<!-- formula-not-decoded -->

then ρ AB is separable; otherwise entangled.

The quantum state is represented by a density matrix , a positive semidefinite matrix with unit trace, analogous to transition matrix in the Markov world. The concept of quantum entanglement describes the inter-dependence of particles in a quantum system, while Markov entanglement describes that of agents in a Markov system.

Finally, we introduce several novel proof techniques concerning the sufficient and necessary condition, including an 'absorbing' technique for separable transition matrices and a novel characterization of the linear space spanned by tensor products of transition matrices. We believe these techniques hold independent interest for the broader RL community.

Decomposition Error in General Multi-agent MDPs Despite the precise characterization of Markov entanglement and exact value decomposition, general multi-agent MDPs can exhibit arbitrary complexity, with agents intricately entangled. This raises a critical question: can value decomposition serve as a meaningful approximation in such scenarios? To address this, we introduce a mathematical quantification to measure the Markov entanglement in general multi-agent MDPs,

<!-- formula-not-decoded -->

where P SEP is the set of all separable transition matrices and d ( · , · ) is some distance measure. In other words, the degree of Markov entanglement is determined by its distance to the closest separable transition matrix. This concept can also find its counterpart in quantum physics, with the measure of quantum entanglement defined as

<!-- formula-not-decoded -->

where ρ SEP is the set of all separable quantum states. In quantum physics, various distance measures have been designed for density matrices and capture different physical interpretations [31]. In the Markov world, we analogously design distance measures for transition matrices and relate them to the value decomposition error,

<!-- formula-not-decoded -->

where ∥·∥ depends on the distance we use to measure Markov entanglement. We explore diverse distance measures including the well-known total variation distance and its stationary distribution weighted variant. We also design a novel agent-wise distance incorporating the multi-agent structure, which may be of independent interest to the MARL community. We further demonstrate how different distance measures give birth to the decomposition error in different norms.

Applications of Markov Entanglement Finally, we leverage our Markov entanglement theory to analyze several structured multi-agent MDPs. We prove that a widely-used class of index policies is asymptotically separable, exhibiting a decomposition error that scales as O ( √ N ) with the number of agents N . This result theoretically justifies the practical effectiveness of value decomposition for index-based policies. Our proof builds on innovations that integrate Markov entanglement with mean-field analysis. We also show that Markov entanglement admits an efficient empirical estimation, thus helping practitioners determine when value decomposition is feasible.

## 1.2 Other related work

In the first section, we have reviewed typical empirical works on value decomposition. Here, we complement that discussion with related literature on theoretical insights.

Prior theoretical research has extensively investigated the decomposition of optimal value functions in multi-agent settings. A prominent area involves Lagrange relaxation, with the Restless Multi-Armed

Bandit (RMAB, [49]) as a foundational model. Lagrange relaxation decouples the constraint of agents, yielding a decomposable value that upper bounds the original value. The per-agent decomposition error is proven to decay asymptotically to zero [47, 48, 41] and enjoys a quadratic or exponential rate [20, 21, 11, 51, 52]. Other work generalizes to Weakly-Coupled MDPs (WCMDPs) [6, 13, 19]. However, [2] showed Lagrange relaxation can have arbitrarily large errors and proposed an alternative decomposition called Approximate Linear Programs (ALP), which is proven to have tighter error [12]. Despite these advancements, characterizing decomposition error for general multi-agent MDPs remains unknown. In contrast, our Markov entanglement theory analyzes value decomposition for general multi-agent MDPs under arbitrary policies, including optimal ones.

Another line of theoretical work has concentrated on policy optimization via value decomposition. Despite reported empirical successes, rigorous theoretical analysis remains challenging. [5] derived an approximation ratio for a specific index policy on a two-state RMAB. [43, 16] analyzed the convergence of the CTDE paradigm under strong exploration assumptions, while also highlighting scenarios of divergence. In contrast, our work instead focuses on policy evaluation rather than optimization. This enables us to derive clear and interpretable bounds on the decomposition error for general finite-state multi-agent MDPs that only require the existence of a stationary distribution.

Notations We abbreviate subscripts ( s ) := ( s 1: N ) := ( s 1 , s 2 , . . . , s N ) . Particularly, for two-agent case, when the context is clear, we abbreviate ( s ) := ( s AB ) := ( s A , s B ) . Let [ N ] = { 1 , 2 , . . . , N } and Z + be the set of positive integers.

## 2 Model

We consider a standard two-agent MDP M AB ( S , A , P , r A , r B , γ ) with joint state space S = S A ×S B and joint action space A = A A ×A B where A,B represent two agents. For simplicity, let |S A | = |S B | = | S | and |A A | = |A B | = | A | . For agents at global state s = ( s A , s B ) with action a = ( a A , a B ) taken, the system will transit to s ′ = ( s ′ A , s ′ B ) according to transition kernel s ′ ∼ P ( · | s , a ) and each agent i ∈ { A,B } will receive its local reward r i ( s i , a i ) . The global reward r AB is defined as the summation of local rewards r AB ( s , a ) := r A ( s A , a A ) + r B ( s B , a B ) , or in vector form r AB ∈ R | S | 2 | A | 2 := r A ⊗ e + e ⊗ r B , where ⊗ is the tensor product and e = 1 ∈ R | S || A | is the vector of all ones. 1 We further assume the local rewards are bounded, i.e. for agent i ∈ { A,B } , | r i ( s i , a i ) | ≤ r i max for all ( s i , a i ) .

Given any global policy π : S → ∆( A ) , the global Q-value under policy π is defined as the discounted summation of global rewards Q π AB ( s , a ) = E [∑ ∞ t =0 γ t r AB ( s t , a t ) | π, ( s 0 , a 0 ) = ( s , a ) ] where γ ∈ [0 , 1) is the discount factor. The value function is then defined as V π AB ( s ) = E a ∼ π ( ·| s ) [ Q π AB ( s , a )] . We denote P π AB ∈ R | S | 2 | A | 2 ×| S | 2 | A | 2 as the transition matrix induced by π where P π AB ( s ′ , a ′ | s , a ) = P ( s ′ | s , a ) · π ( a ′ | s ′ ) . Then by the Bellman Equation, we have Q π AB = ( I -γ P π AB ) -1 r AB . Our objective is to decompose this global Q-value Q π AB as the summation of some local functions Q A and Q B , i.e. Q π AB ( s , a ) = Q A ( s A , a A ) + Q B ( s B , a B ) , or in vector form,

<!-- formula-not-decoded -->

Notice we formally introduce our research question using Q-value instead of V-value function as in the introduction. Q-value decomposition is a stronger result that implies V-value function decomposition. It also turns out that Q-value further incorporates action information enabling more general theoretical analysis. More discussions can be found in Appendix B.

## 2.1 Local (Q-)value functions

Recent literature offers several algorithms for learning local (Q-)values. In this paper, we use a meta-algorithm framework in 1 to summarize their underlying principles.

This meta-algorithm framework is simple and intuitive: each agent independently fits its local Qvalues based on its local observations. Notably, the framework requires no prior knowledge of the MDP, and learning can be performed in a fully decentralized manner. Furthermore, we use term meta in that we do not pose restrictions on how agents estimate their local Q-values. For tabular case, one

1 In Appendix J.4, we extend our results to multi-agent MDP model where the global cannot be decomposed.

## Meta Algorithm 1: Leaning Local Q-value Functions

Require: Global policy π ; horizon length T .

- 1: Execute π for T epochs and obtain D = { ( s t AB , a t AB , r t AB , s t +1 AB , a t +1 AB ) } T -1 t =1 .
- 2: Each agent i ∈ { A,B } fits Q π i using local observations D i = { ( s t i , a t i , r t i , s t +1 i , a t +1 i ) } T -1 t =1 .

can plug in Temporal Difference (TD) learning [39] or its variants. For large-scale problems, one can apply linear function approximations (e.g. [5, 24, 8]) or more sophisticated neural networks (e.g. [33, 38, 29]).

Despite the flexibility in fitting local value functions, it is helpful to call out a particular approach: TD learning for local Q-values in the tabular case, as it facilitates the analysis and reveals the structure of value decomposition in the next section.

Local TD learning. Although each agent's environment is not Markovian in a local sense (it is, more precisely, partially observed Markovian), one can still define its 'marginalized' local transition matrix under the stationary distribution. Mathematically, for agent A , we denote P π A ∈ R | S || A |×| S || A | as its local transition where

<!-- formula-not-decoded -->

Here, µ π AB ∈ ∆( S ) denotes the global stationary distribution under policy π (for convenience, we assume π induces a unichain, i.e. µ π AB is unique and strictly positive). 2 Given this "marginalized" local transition, the local Q-values obtained by Meta Algorithm 1 using tabular TD learning converge to the solution of the following 'marginalized' Bellman equation:

<!-- formula-not-decoded -->

By symmetry, we can derive analogous results for agent B , obtaining its transition matrix P π B and local Q-values Q π B . Next, we show how Q π A and Q π B contribute to the exact value decomposition.

## 3 Exact value decomposition

To begin, recall the key condition we identify in the introduction: Markov Entanglement in Definition 1. Our first theorem shows that an MDP with no Markov entanglement is indeed sufficient for the exact value decomposition. More importantly, local TD learning (or Meta Algorithm 1 more generally) is guaranteed to recover such decomposition, i.e. Q π AB = Q π A ⊗ e + e ⊗ Q π B .

Theorem 1. Consider a two-agent MDP M AB and policy π : S → ∆( A ) . If two agents are separable, i.e. there exists K ∈ Z + , measure { x j } j ∈ [ K ] , and transition matrices { P ( j ) A , P ( j ) B } j ∈ [ K ]

such that P π AB = ∑ K j =1 x j P ( j ) A ⊗ P ( j ) B . Then it holds P π A = ∑ K i =1 x j P ( j ) A and P π B = ∑ K j =1 x j P ( j ) B . Furthermore, the Eq. (2) holds

<!-- formula-not-decoded -->

This theorem establishes that even when the system is not independent, as long as it can be represented as a linear combination of independent subsystems , the global Q-value admits an exact decomposition.

An illustrative example of coupling and Markov entanglement To elucidate the concept of Markov entanglement, we present an example of two-agent MDP where agents are coupled but not entangled. Consider a two-agent MDP M AB with |A A | = |A B | = 2 , where action 1 means activate and 0 means idle. Each agent i ∈ { A,B } has its own local transition kernel P i . We examine the following policy: at each time-step, we randomly activate one agent and keep another idle, i.e.

2 For µ π AB ( s B , a B | s A , a A ) to be well-defined, we require µ π AB ( s A , a A ) &gt; 0 . If µ π AB ( s A , a A ) = 0 , then action a A is never taken in state s A under policy π , and we exclude such pairs by restricting the feasible action set A ( s A ) . All theoretical results apply to the remaining valid state-action pairs.

π ( a | s ) = 1 / 2 if a = (0 , 1) or a = (1 , 0) . Consequently, this policy couples the agents through the constraint a A + a B = 1 at each timestep. However, we will demonstrate that despite this coupling, there's no entanglement. Specifically, we construct the following decomposition

<!-- formula-not-decoded -->

where P a i refers to the transition matrix of agents i ∈ { A,B } taking action a ∈ { 0 , 1 } . Intuitively, the right-hand side of Eq. (4) describes how at each time step, the global system randomly selects between two possible transitions: P 0 A ⊗ P 1 B or P 1 A ⊗ P 0 B , each with equal probability (akin to rolling a fair dice). This example thus clearly demonstrates a coupled system can still be separable and thus admits an exact value decomposition.

Proof of sufficiency Theorem 1 admits a simple proof based on the several basic properties of tensor product. First of all, given P π AB = ∑ K j =1 x j P ( j ) A ⊗ P ( j ) B , we can plug this into the formulation of P π A in Eq. (3) and quickly verify P π A = ∑ K i =1 x i P ( i ) A . It remains to show Eq. (2). Notice that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we refer to ( i ) as an 'absorbing' technique based on the bilinearity and mixed-product property of tensor product 3 . Specifically, since Pe = e for any transition matrix P , we have for any t ,

<!-- formula-not-decoded -->

Similar results can be derived for P π B such that ( I -γ P π AB ) -1 ( e ⊗ r B ) = e ⊗ Q π B . Finally, combining the above results, we have

<!-- formula-not-decoded -->

## 3.1 Necessary condition for the exact value decomposition

We then investigate whether Markov entanglement is necessary for the exact Q-value decomposition. The answer is in general no, since one can construct trivial counterexamples such as r A = r B = 0 or γ = 0 , where the decomposition trivially holds. On the other hand, we focus on a stronger and more general concept of the exact value decomposition that holds under any reward kernel given γ &gt; 0 . Formally, we present the following theorem.

Theorem 2. Consider a two-agent Markov MDP M AB with discount factor γ &gt; 0 and π : S → ∆( A ) . Suppose there exists local functions Q i : r i → R | S || A | for i ∈ { A,B } such that Q π AB = Q A ( r A ) ⊗ e + e ⊗ Q B ( r B ) holds for any pair of reward r A , r B , then A,B must be separable.

Combined with Theorem 1, we conclude Markov entanglement serves as a sufficient and necessary condition for the exact value decomposition. We also emphasize that Theorem 2 considers general local functions Q i . This generality accommodates all methods for fitting local Q i , such as deep neural networks, provided that the training relies solely on the local observations of agent i .

There exist other possible ways for value decomposition. For example, [38, 16] consider Q π AB ( s , a ) = L A ( s A , a A , r AB ) + L B ( s B , a B , r AB ) where L A , L B are learned jointly via minimizing the global Bellman error 4 ; [35, 29, 37, 42] consider general monotonic operations beyond

3 We introduce several basic properties of tensor product in Appendix A.

4 In Appendix E, we provide an example of entangled MDP that allows for an exact value decomposition where L A depends on both r A and r B .

additive decompositions. These methods introduce possibly richer representations at the cost of more sophisticated implementations and less interpretability, which is beyond the scope of this paper.

Proof sketch of necessity Our proof builds on several novel techniques. Recall P SEP is the set of all separable transition matrices.

Step 1: Understanding the orthogonal complement. If a transition matrix is entangled, it will have non-zero component in the orthogonal complement of P SEP, which we construct as

<!-- formula-not-decoded -->

where ε j = (1 , 0 , . . . , 0 , -1 , 0 , . . . , 0) ⊤ with the first element 1 and ( j +1) -th element -1 . Then, we study an intermediate transition matrix (1 -γ )( I -γ P π AB ) -1 . We show if it's entangled, we are able to construct r A , r B based on its component in P ⊥ SEP such that Q π AB is not decomposable under this pair of rewards. We thus conclude decomposable Q π AB = ⇒ separable (1 -γ )( I -γ P π AB ) -1 .

Step 2: Connecting to 'inverse'. Finally, we complete the proof via a lemma showing separable (1 -γ )( I -γ P π AB ) -1 ⇐⇒ separable P π AB . The ⇐ = side is straightforward since ( I -γ P π AB ) -1 is the Neumann series of γ P π AB . For the converse = ⇒ , we seek to invert this Neumann series. This is achieved by a careful analysis of the operator norm of I -(1 -γ )( I -γ P π AB ) -1 .

## 4 Value decomposition error in general two-agent MDPs

In general, the system transition P π AB can be arbitrarily entangled. In these scenarios, we investigate when value decomposition Q π A ⊗ e + e ⊗ Q π B is an effective approximation of Q π AB . As mentioned in the introduction, we define the measure of Markov entanglement in Eq. (1) as certain distance between P π AB and its closet separable transition matrix. We will examine several distance measures for transition matrices and relate them to the decomposition error.

## 4.1 Entry-wise error bound

Total variation distance One widely used metric for transition matrices is Total Variation (TV) distance. Specifically, for two transition matrices P , P ′ ∈ R | S | 2 | A | 2 ×| S | 2 | A | 2 , define

<!-- formula-not-decoded -->

where D TV is the total variation distance between probability measures. While TV distance is straightforward, it does not take into account the inherent multi-agent structure.

Agent-wise distance We thus introduce a more refined distance specially designed for multi-agent MDPs. Formally, the Agent-wise Total Variation (ATV) distance between two transition matrices P , P ′ ∈ R | S | 2 | A | 2 ×| S | 2 | A | 2 w.r.t agent A is defined as

<!-- formula-not-decoded -->

The ATV distance w.r.t agent B can be defined similarly. Intuitively, compared to TV , ATV focuses on an individual agent and measures the difference between its local transitions. One can also verify ATV is tighter distance, i.e. ∥ P -P ′ ∥ ATV A ≤ ∥ P -P ′ ∥ TV . We can plug ATV into Eq. (1) and obtain the measure of Markov entanglement w.r.t ATV distance E i ( P π AB ) := min P ∈P SEP ∥ P π AB -P ∥ ATV i for i ∈ { A,B } . In fact, one can also verify

<!-- formula-not-decoded -->

The following theorem connects these measures to the value decomposition error.

Theorem 3. Consider a two-agent Markov system M AB and policy π : S → ∆( A ) with the measure of Markov entanglement E A ( P π AB ) , E B ( P π AB ) defined in Eq. (7), then the decomposition error is entry-wise bounded by the measure of Markov entanglement,

<!-- formula-not-decoded -->

## 4.2 Error weighted by stationary distribution

Entry-wise error bound is a very strong result for Q-value decomposition. This comes with the entry-wise TV bounds in both TV and ATV distance. An alterative choice is to consider an error weighted by the stationary distribution. Formally, consider

<!-- formula-not-decoded -->

We note that this norm is clearly weaker than the entry-wise norm. Nevertheless, a stationary distribution weighted error bound is sufficient in many practical scenarios. Similar ideas are also quite common in policy evaluation literature [14, 40, 9].

Distance weighted by stationary distribution To analyze this µ π AB -weight decomposition error, we analogously propose the µ π AB -weighted distance measure of Markov entanglement. Specifically, we have the following µ π AB -weighted version of Eq. (7).

<!-- formula-not-decoded -->

Eq. (8) substitutes the µ π AB -weighted average for the maximum operator in Eq. (7). Finally, we have the following variant of Theorem 3.

Theorem 4. Under the same setup as Theorem 3 with µ π AB -weighted measure of Markov entanglement E A ( P π AB ) , E B ( P π AB ) defined in Eq. (8), the µ π AB -weighted decomposition error is bounded,

<!-- formula-not-decoded -->

Compared to Theorem 3, Theorem 4 measures a weaker µ π AB -weighted decomposition error, while the condition on P π AB is also relaxed, requiring only a weighted average bound in Eq. (8).

## 4.3 Multi-agent Markov entanglement

Finally, we extend the results to multi-agent MDPs with the measure of Markov entanglement E 1: N ( P π 1: N ) for an N -agent MDP. The extension is relatively straightforward. We demonstrate the extension of Theorem 4 below and more details can be found in Appendix H.

Theorem 5. Consider a N -agent MDP M 1: N with the measure of Markov entanglement E i ( P π 1: N ) w.r.t ATV distance, the µ π 1: N -weighted decomposition error is bounded by the measure of Markov entanglement,

<!-- formula-not-decoded -->

## 5 Applications of Markov Entanglement

In this section, we apply Markov entanglement and demonstrate a widely-used class of index policies is asymptotically separable. To begin, we introduce the model of Restless Multi-Armed Bandit (RMAB, [49]). In an N -agent RMAB, each agent follows a homogeneous two-action MDP with action 1 meaning activate and 0 idle. A central decision maker will activate M ≤ N agents at each timestep and leave other agents idle. In other words, agents transit independently but are coupled under constraint ∑ N i =1 a i = M . In RMAB, arguably the most classical and widely-used policy is the index policy, which we formally define as

Definition 3 (Index Policy) . There exists a priority index ν s for each local state s . The decision maker will always activate agents in the descending order of the priority until the budget constraint M is met. Ties are resolved fairly via uniform random sampling of agents at the same state.

The index policy traces back to the well-known Gittins Index [46], Whittle Index [49, 47, 20], and fluid-based index policies [41, 21]. [33, 4, 5, 30, 44, 3] apply data-driven method to optimize index policies and report great empirical success in industrial implementations. Understanding the mystery behind such success calls for a theory for general index policies. We then present our main theorem.

Theorem 6. Consider an N -agent restless multi-armed bandit. For any index policy satisfying mild technical conditions, there exists constant C independent of N , such that for any agent i ∈ [ N ] , its µ π 1: N -weighted measure of Markov entanglement is bounded, E i ( P π 1: N ) ≤ C/ √ N .

Theorem 6 requires two standard technical conditions for index policies: non-degenerate and uniform global attractor property, which are used in almost all related theoretical work [47, 41, 20, 21] and are detailed in Appendix I. Theorem 6 justifies index polices are asymptotically separable. Combined with an N -agent version of Theorem 4, we obtain the sublinear decomposition error for index policies

<!-- formula-not-decoded -->

This sublinear error result explains why the value decomposition in [33, 4, 5] manages to effectively approximate the global value function in large-scale practical applications.

## 5.1 Efficient verification of value decomposition

For practitioners, verifying the feasibility of value decomposition is challenging due to the exponential computational complexity of estimating the global Q-value. As a solution, Markov entanglement offers an efficient way to empirically test whether value decomposition can be safely applied. Consider the µ π AB -weighted measure of Markov entanglement in Eq. (8), we have

<!-- formula-not-decoded -->

In other words, we can apply a Monte-Carlo estimation for E A ( P π AB ) . Notice Eq. (9) is convex for P A , which enables efficient solutions. As a result, Eq. (9) provides an efficient estimation of Markov entanglement via simulation and can be easily extend to N -agent MDPs.

Numerical experiments. Finally, we empirically study the value decomposition for the index policy on a circulant RMAB benchmark [3, 52, 10, 18] that has 4 different states each local agent. As a result, the global state space scales as large as 4 1800 &gt; 10 1000 for N = 1800 agents. The specific transitions and rewards are introduced in Appendix K. For each RMAB instance, we sample a trajectory of length T = 5 N and use the collected data to i) solve Eq. (9) to estimate the measure of Markov entanglement; ii) train local Q-value decomposition. It quickly follows from the results in Figure 1:

√

The estimated Markov entanglement decays as O (1 / N ) in the left panel, consistent with theoretical predictions. This also implies a low decomposition error scaling of O ( √ N ) , as seen in the right panel. Furthermore, the simulated trajectory has a length of T = 5 N while the global state space has size | S | N , making both entanglement estimation and local Q-value decomposition sample-efficient.

## 6 Discussions

Comparison with quantum entanglement One notable difference between the definition of Markov and quantum entanglement is that the former does not require coefficients x ≥ 0 . In Appendix C, we show there exist separable two-agent MDPs that can only be represented by linear combinations but not convex combinations of independent subsystems, highlighting a structural difference between Markov and quantum entanglement. Finally, we emphasize that our analogy to quantum entanglement is mostly in the mathematical formulation; there is no clear physical interpretation analogy between Markov and quantum entanglement.

<!-- image -->

0

ErrorWeightedbyStationaryDistribution

LocalTD

O(VN)

300

600

900

1200

1500

1800

NumberofAgents

Figure 1: Circulant RMAB under an index policy. Left: empirical estimation of Markov entanglement multiplied by the number of agents, NE 1 ( P π 1: N ) . Right: µ -weighted decomposition error.

Relations to Influenced-based MARL There's another line of MARL research that explicitly models the influence of other agents as intrinsic rewards for exploration [45, 26]. It turns out the mutual information can be viewed as the measure of Markov entanglement under KL-divergence. Specifically, we can rewrite mutual information in [45] as

<!-- formula-not-decoded -->

This is highly related to our measure of Markov entanglement under a µ π -weighted agent-wise KL-divergence, which we can define as

<!-- formula-not-decoded -->

Intuitively, the measure of Markov entanglement can be viewed as how closely one agent can be approximated as an independent subsystem. This characterization aligns naturally with mutual information. Furthermore, since KL-divergence provides an upper bound for total variation distance, it consequently bounds our Markov entanglement measure relative to the ATV distance introduced in our paper. This connection demonstrates that influence-based MARL methods naturally fit within our theoretical framework, corresponding to a specialized distance measure.

## 7 Conclusion

This paper established the mathematical foundation of value decomposition in MARL. Drawing inspiration from quantum physics, we propose the idea of Markov entanglement and prove that it serves as a sufficient and necessary condition for the exact value decomposition. We further characterize the decomposition error in general multi-agent MDPs through the measure of Markov entanglement. As application examples, we prove widely-used index policies are asymptotically separable and suggest practitioners using Markov entanglement as a proxy for estimating the effectiveness of value decomposition.

## Acknowledgments and Disclosure of Funding

We thank all anonymous reviewers for their constructive comments. We are also grateful to Prof. Tongyang Li for valuable and insightful discussions.

## References

- [1] Daniel Adelman. Dynamic bid prices in revenue management. Operations Research , 55(4): 647-661, 2007.

6.0

5.0

4.0

3.0

2.0

1.0

0.0

- [2] Daniel Adelman and Adam J. Mersereau. Relaxations of weakly coupled stochastic dynamic programs. Operations Research , 56(3):712-727, 2008. doi: 10.1287/opre.1070.0445.
- [3] Konstantin E Avrachenkov and Vivek S Borkar. Whittle index based q-learning for restless bandits with average reward. Automatica , 139:110186, 2022.
- [4] Xabi Azagirre, Akshay Balwally, Guillaume Candeli, Nicholas Chamandy, Benjamin Han, Alona King, Hyungjun Lee, Martin Loncaric, Sébastien Martin, Vijay Narasiman, Zhiwei (Tony) Qin, Baptiste Richard, Sara Smoot, Sean Taylor, Garrett van Ryzin, Di Wu, Fei Yu, and Alex Zamoshchin. A better match for drivers and riders: Reinforcement learning at lyft. INFORMS Journal on Applied Analytics , 54(1):71-83, 2024.
- [5] Jackie Baek, Justin J Boutilier, Vivek F Farias, Jonas Oddur Jonasson, and Erez Yoeli. Policy optimization for personalized interventions in behavioral health. arXiv preprint arXiv:2303.12206 , 2023.
- [6] Santiago R. Balseiro, David B. Brown, and Chen Chen. Dynamic pricing of relocating resources in large networks. Management Science , 67(7):4075-4094, 2021.
- [7] Santiago R. Balseiro, Haihao Lu, and Vahab Mirrokni. The best of many worlds: Dual mirror descent for online allocation problems. Operations Research , 71(1):101-119, 2023.
- [8] Dimitri Bertsekas and John N Tsitsiklis. Neuro-dynamic programming . Athena Scientific, 1996.
- [9] Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. Operations Research , 69(3):950-973, 2021.
- [10] Arpita Biswas, Gaurav Aggarwal, Pradeep Varakantham, and Milind Tambe. Learn to intervene: An adaptive learning policy for restless bandits in application to preventive healthcare. In Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI 2021) , pages 4036-4049, 2021.
- [11] David B Brown and Jingwei Zhang. Dynamic programs with shared resources and signals: Dynamic fluid policies and asymptotic optimality. Operations Research , 70(5):3015-3033, 2022.
- [12] David B. Brown and Jingwei Zhang. Technical note-on the strength of relaxations of weakly coupled stochastic dynamic programs. Operations Research , 71(6):2374-2389, 2023. doi: 10.1287/opre.2022.2287.
- [13] David B. Brown and Jingwei Zhang. Fluid policies, reoptimization, and performance guarantees in dynamic resource allocation. Operations Research , 73(2):1029-1045, 2025.
- [14] Qi Cai, Zhuoran Yang, Jason D Lee, and Zhaoran Wang. Neural temporal-difference learning converges to global optima. In Advances in Neural Information Processing Systems , volume 32, 2019.
- [15] Shao-Hung Chan, Zhe Chen, Teng Guo, Han Zhang, Yue Zhang, Daniel Harabor, Sven Koenig, Cathy Wu, and Jingjin Yu. The league of robot runners competition: Goals, designs, and implementation. In ICAPS 2024 System's Demonstration track , 2024.
- [16] Zehao Dou, Jakub Grudzien Kuba, and Yaodong Yang. Understanding value decomposition algorithms in deep cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2202.04868 , 2022.
- [17] Vivek Farias, Hao Li, Tianyi Peng, Xinyuyang Ren, Huawei Zhang, and Andrew Zheng. Correcting for interference in experiments: A case study at douyin. In Proceedings of the 17th ACM Conference on Recommender Systems , pages 455-466, 2023.
- [18] Jing Fu, Yoni Nazarathy, Sarat Moka, and Peter G. Taylor. Towards q-learning the whittle index for restless bandits. In 2019 Australian New Zealand Control Conference (ANZCC) , 2019.
- [19] Nicolas Gast, Bruno Gaujal, and Chen Yan. Reoptimization nearly solves weakly coupled markov decision processes. arXiv preprint arXiv:2211.01961 , 2022.

- [20] Nicolas Gast, Bruno Gaujal, and Chen Yan. Exponential asymptotic optimality of whittle index policy. Queueing Syst. Theory Appl. , 104(1-2):107-150, may 2023.
- [21] Nicolas Gast, Bruno Gaujal, and Chen Yan. Linear program-based policies for restless bandits: Necessary and sufficient conditions for (exponentially fast) asymptotic optimality. Mathematics of Operations Research , 49(4):2468-2491, 2024.
- [22] Carlos Guestrin, Daphne Koller, and Ronald Parr. Multiagent planning with factored mdps. In Advances in Neural Information Processing Systems , volume 14. MIT Press, 2001.
- [23] Carlos Guestrin, Daphne Koller, Ronald Parr, and Shobha Venkataraman. Efficient solution algorithms for factored mdps. Journal of Artificial Intelligence Research , 19(1):399-468, 2003.
- [24] Benjamin Han, Hyungjun Lee, and Sébastien Martin. Real-time rideshare driver supply values using online reinforcement learning. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , KDD '22, page 2968-2976, 2022.
- [25] Yitian Hong, Yaochu Jin, and Yang Tang. Rethinking individual global max in cooperative multi-agent reinforcement learning. Advances in neural information processing systems , 35: 32438-32449, 2022.
- [26] Natasha Jaques, Angeliki Lazaridou, Edward Hughes, Caglar Gulcehre, Pedro Ortega, Dj Strouse, Joel Z. Leibo, and Nando De Freitas. Social influence as intrinsic motivation for multi-agent deep reinforcement learning. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 3040-3049. PMLR, 09-15 Jun 2019.
- [27] Igor Kadota, Elif Uysal-Biyikoglu, Rahul Singh, and Eytan Modiano. Minimizing the age of information in broadcast wireless networks. In 2016 54th Annual Allerton Conference on Communication, Control, and Computing (Allerton) , pages 844-851. IEEE, 2016.
- [28] Yash Kanoria and Pengyu Qian. Blind dynamic resource allocation in closed networks via mirror backpressure. Management Science , 70(8):5445-5462, 2024.
- [29] Anuj Mahajan, Tabish Rashid, Mikayel Samvelyan, and Shimon Whiteson. Maven: Multi-agent variational exploration. Advances in neural information processing systems , 32, 2019.
- [30] Khaled Nakhleh, Santosh Ganji, Ping-Chun Hsieh, I-Hong Hou, and Srinivas Shakkottai. Neurwin: Neural whittle index network for restless bandits via deep rl. In Advances in Neural Information Processing Systems , volume 34, pages 828-839, 2021.
- [31] Michael A Nielsen and Isaac L Chuang. Quantum computation and quantum information . Cambridge university press, 2010.
- [32] Ian Osband and Benjamin Van Roy. Near-optimal reinforcement learning in factored mdps. In Proceedings of the 28th International Conference on Neural Information Processing Systems , NIPS'14, page 604-612, 2014.
- [33] Zhiwei (Tony) Qin, Xiaocheng Tang, Yan Jiao, Fan Zhang, Zhe Xu, Hongtu Zhu, and Jieping Ye. Ride-hailing order dispatching at didi via reinforcement learning. INFORMS Journal on Applied Analytics , 50(5):272-286, 2020. doi: 10.1287/inte.2020.1047.
- [34] Naveen Janaki Raman, Zheyuan Ryan Shi, and Fei Fang. Global rewards in restless multi-armed bandits. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [35] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder De Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. Monotonic value function factorisation for deep multi-agent reinforcement learning. Journal of Machine Learning Research , 21(178):1-51, 2020.
- [36] Ibrahim El Shar and Daniel R. Jiang. Weakly coupled deep q-networks. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.

- [37] Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, and Yung Yi. QTRAN: Learning to factorize with transformation for cooperative multi-agent reinforcement learning. In Proceedings of the 36th International Conference on Machine Learning , volume 97, pages 5887-5896. PMLR, 2019.
- [38] Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore Graepel. Value-decomposition networks for cooperative multi-agent learning based on team reward. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems , AAMAS '18, page 2085-2087, 2018.
- [39] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . The MIT Press, 2018. ISBN 0262039249.
- [40] John Tsitsiklis and Benjamin Van Roy. Analysis of temporal-diffference learning with function approximation. In Advances in Neural Information Processing Systems , volume 9. MIT Press, 1996.
- [41] Ina Maria Verloop. Asymptotically optimal priority policies for indexable and nonindexable restless bandits. Annals of Applied Probability , 26:1947-1995, 2016.
- [42] Jianhao Wang, Zhizhou Ren, Terry Liu, Yang Yu, and Chongjie Zhang. Qplex: Duplex dueling multi-agent q-learning. arXiv preprint arXiv:2008.01062 , 2020.
- [43] Jianhao Wang, Zhizhou Ren, Beining Han, Jianing Ye, and Chongjie Zhang. Towards understanding cooperative multi-agent q-learning with value factorization. Advances in Neural Information Processing Systems , 34:29142-29155, 2021.
- [44] Kai Wang, Lily Xu, Aparna Taneja, and Milind Tambe. Optimistic whittle index policy: Online learning for restless bandits. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 10131-10139, 2023.
- [45] Tonghan Wang, Jianhao Wang, Yi Wu, and Chongjie Zhang. Influence-based multi-agent exploration. arXiv preprint arXiv:1910.05512 , 2019.
- [46] Richard Weber. On the gittins index for multiarmed bandits. The Annals of Applied Probability , 2(4):1024 - 1033, 1992.
- [47] Richard R. Weber and Gideon Weiss. On an index policy for restless bandits. Journal of Applied Probability , 27(3):637-648, 1990.
- [48] Richard R. Weber and Gideon Weiss. Addendum to 'on an index policy for restless bandits'. Advances in Applied Probability , 23(2):429-430, 1991.
- [49] P. Whittle. Restless bandits: Activity allocation in a changing world. Journal of Applied Probability , 25:287-298, 1988.
- [50] Dan Zhang and Daniel Adelman. An approximate dynamic programming approach to network revenue management with customer choice. Transportation Science , 43(3):381-394, 2009.
- [51] Xiangyu Zhang and Peter I Frazier. Restless bandits with many arms: Beating the central limit theorem. arXiv preprint arXiv:2107.11911 , 2021.
- [52] Xiangyu Zhang and Peter I Frazier. Near-optimality for infinite-horizon restless bandits with many arms. arXiv preprint arXiv:2203.15853 , 2022.

## Contents

| 1 Introduction   | 1 Introduction                                                                                                  | 1     |
|------------------|-----------------------------------------------------------------------------------------------------------------|-------|
|                  | 1.1 This paper . . . . . . . . . . . . . . . . . . . . . . .                                                    | 2     |
|                  | 1.2 Other related work . . . . . . . . . . . . . . . . . .                                                      | 3     |
| 2                | Model                                                                                                           | 4     |
|                  | 2.1 Local (Q-)value functions . . . . . . . . . . . . . . .                                                     | 4     |
| 3                | Exact value decomposition                                                                                       | 5     |
|                  | 3.1 Necessary condition for the exact value decomposition                                                       | 6     |
| 4                | Value decomposition error in general two-agent MDPs                                                             | 7     |
|                  | 4.1 Entry-wise error bound . . . . . . . . . . . . . . . .                                                      | 7     |
|                  | 4.2 Error weighted by stationary distribution . . . . . . .                                                     | 8     |
|                  | 4.3 Multi-agent Markov entanglement . . . . . . . . . .                                                         | 8     |
| 5                | Applications of Markov Entanglement                                                                             | 8     |
|                  | 5.1 Efficient verification of value decomposition . . . . .                                                     | 9     |
| 6                | Discussions                                                                                                     | 9     |
| 7                | Conclusion                                                                                                      | 10    |
| A                | Linear algebra with tensor product                                                                              | 15    |
| B                | Decompose value functions                                                                                       | 15    |
| C                | Comparison with quantum entanglement                                                                            | 15    |
| D                | Proof of Theorem 2                                                                                              | 16    |
| E                | Decomposition via general functions                                                                             | 18    |
| F                | Proof of Theorem 3                                                                                              | 18    |
| G                | Proof of Theorem 4                                                                                              | 20    |
| H                | Results for multi-agent MDPs                                                                                    | 22    |
| I                | Proof of Theorem 6                                                                                              | 24    |
| J                | Extensions of Markov entanglement                                                                               | 27    |
|                  | J.1 (Weakly-)coupled MDPs . . . . . . . . . . . . . . .                                                         | 27    |
|                  | J.2 Coupled MDPs with exogenous information . . . . . J.3 Factored MDPs . . . . . . . . . . . . . . . . . . . . | 28 30 |

| J.4   | Fully cooperative Markov games . . . . . . . .    |   31 |
|-------|---------------------------------------------------|------|
| K     | Simulation environments                           |   31 |
| K.1   | Monte-Carlo estimation of Markov entanglement     |   32 |
| K.2   | Learning local Q-values . . . . . . . . . . . . . |   32 |
| K.3   | Sample Complexity and Computation . . . . .       |   33 |

## A Linear algebra with tensor product

We briefly introduce the basic properties of tensor product or Kronecker product. Let A ∈ R m 1 × n 1 , B ∈ R m 2 × n 2 , then

<!-- formula-not-decoded -->

Tensor product satisfies the following basic properties,

- 1. Bilinearity For any matrix A , B , C and constant k , it holds k ( A ⊗ B ) = ( k A ) ⊗ B = A ⊗ ( k B ) , ( A + B ) ⊗ C = A ⊗ C + B ⊗ C , and A ⊗ ( B + C ) = A ⊗ B + A ⊗ C .
- 2. Mixed-product Property For any matrix A , B , C , D , if AC and BD form valid matrix product, then ( A ⊗ B )( C ⊗ D ) = ( AC ) ⊗ ( BD ) .

## B Decompose value functions

Compared to the decomposition of Q-value, the value function further requires the reward to be state-dependent . To illustrate, notice by Bellman equation,

<!-- formula-not-decoded -->

where we abuse notation and denote P π AB ( s ′ | s ) = ∑ a π ( a | s ) P ( s ′ | s , a ) and reward r π AB ( s ) = ∑ a π ( a | s ) r AB ( s , a ) . A key subtlety arises because r π AB may not be decomposable-even when r AB is decomposable-unless the reward r AB is state-dependent. Consequently, we cannot directly apply the "absorbing" equation as in the proof of Theorem 1.

On the other hand, Q-value decomposition bypasses the state-dependence assumption and provides a stronger condition that directly implies value function decomposition. As a result, while learning local value functions may seem more intuitive, we recommend learning local Q-values instead and using them to approximate the global value function.

## C Comparison with quantum entanglement

It turns out that our Markov entanglement condition serves as a mathematical counterpart of quantum entanglement in quantum physics. We provide the formal definition of the latter for comparison.

Definition 4 (Two-party Quantum Entanglement) . Consider a two-party quantum system composed of two subsystems A and B . The joint state ρ AB is separable if there exists K ∈ Z + , a probability measure { x j } j ∈ [ K ] , and density matrices { ρ ( j ) A , ρ ( j ) B } j ∈ [ K ] such that

<!-- formula-not-decoded -->

If there exists no such decomposition, ρ AB is entangled .

The density matrices are square matrices satisfying certain properties such as positive semidefiniteness and trace normalization, which can be viewed as the counterparts of transition matrices

in the Markov world. Despite the similarities in mathematical form, quantum entanglement imposes an additional constraint requiring { x j } j ∈ [ K ] to be a probability measure, i.e. x ≥ 0 . In contrast, our Markov entanglement defined in Definition 1 permits general linear coefficients { x j } j ∈ [ K ] as long as ∑ k j =1 x j = 1 . This distinction raises the important question of whether negative coefficients are indeed necessary in characterizing Markov entanglement.

To start with, we introduce the set of all separable transition matrices

<!-- formula-not-decoded -->

where K ∈ Z + and { P ( j ) A , P ( j ) B } j ∈ [ K ] are transition matrices. P ≥ 0 calls for every element of P SEP to be a valid transition matrix. It's clear that a transition matrix P π AB is separable if and only if P π AB ∈ P SEP. On the other hand, a direct analogy of quantum entanglement gives us the following set that further requires non-negative coefficients,

<!-- formula-not-decoded -->

Interestingly, it turns out P + SEP ⊈ P SEP. In other words, there exist separable two-agent MDPs that can only be represented by linear combinations but not convex combinations of independent subsystems. Specifically, consider the following basis

<!-- formula-not-decoded -->

And the corresponding transition matrix we provide is

<!-- formula-not-decoded -->

One can also verify P can not be represented by the convex combination of tensor products of these basis. This result justifies the necessity of negative coefficients in x and highlights a structural difference between Markov entanglement and quantum entanglement

## D Proof of Theorem 2

We provide the full proof of Theorem 2 in this section.

Step 1: Characterize the Orthogonal Complement. To start with, we consider the smallest subspace containing all transition matrices Ω P := span(P) where P are the set of all transition matrices in R m × m . We then study the dimension of Ω P .

Lemma 1. The dimension of Ω P is dim(Ω P ) = m 2 -m+1 .

Proof. Let Z ij ∈ R m × m such that

<!-- formula-not-decoded -->

One basis for all transition matrices is given by { Z ij } i,j ∈ [ m ] whose cardinarlity is m 2 -m +1 .

Let Ω P ⊗ 2 := span(P 1 ⊗ P 2 ) be the minimal subspace containing all separable transition matrices. It quickly follows that

<!-- formula-not-decoded -->

We then construct the orthogonal complement of Ω P ⊗ 2 under Frobenius inner product. Let { ε j } j ∈ [ m -1] be a set of vector in R m such that ε j = (1 , 0 , . . . , 0 , -1 , 0 , . . . , 0) ⊤ with the first element 1 and j +1 -th element -1 . Notice that

<!-- formula-not-decoded -->

for all ε j . Consider the following subspace

<!-- formula-not-decoded -->

We then show Ω ′ is exactly the orthogonal complement of Ω P ⊗ 2 . First, notice that

<!-- formula-not-decoded -->

and thus dim(Ω ′ ) + dim(Ω P ⊗ 2 ) = m 4 . Moreover, one can verify for any X ∈ Ω P ⊗ 2 and Y ∈ Ω ′ , Tr( X ⊤ Y ) = 0 . As a result, it holds

<!-- formula-not-decoded -->

Step 2: Connection to 'Inverse" The decomposition of Q-value ultimately concerns with the properties of ( I -γ P π AB ) -1 . The following lemma bridges this gap.

Lemma2. Given any transition matrix P and γ &gt; 0 , P is separable if and only if (1 -γ )( I -γ P ) -1 is separable.

Proof. ( ⇒ ) One can verify that ( I -γ P ) e = (1 -γ ) e , which implies (1 -γ )( I -γ P ) -1 is a transition matrix. Moreover, (1 -γ )( I -γ P ) -1 = (1 -γ ) ∑ ∞ i =0 ( γ P ) i falls in Ω P ⊗ 2 as P ∈ Ω P ⊗ 2 .

( ⇐ ) This side is more involved. Denote U := (1 -γ )( I -γ P ) -1 . Then if the spectral radius ρ ( I -U ) &lt; 1 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies U -1 = 1 1 -γ ( I -γ P ) ∈ Ω P ⊗ 2 and thus P ∈ Ω P ⊗ 2 , finishing the proof. It then suffices to show ρ ( I -U ) &lt; 1 . Notice that

<!-- formula-not-decoded -->

Let λ i ( P ) = a + bi and taking modulus for both side

<!-- formula-not-decoded -->

We conclude the proof given ρ ( I -U ) = max i | λ i ( I -U ) | &lt; 1 .

̸

̸

Step 3: Put it together By Lemma 2, if P π AB is entangled, then (1 -γ )( I -γ P π AB ) -1 is also entangled. Then there exists Y ∈ Ω ′ = 0 such that Tr( Y ⊤ ( I -γ P π AB ) -1 ) = 0 . We apply singular value decomposition to all W 1 1: j , W 2 1: j and conclude there exists some j and u , v ∈ R m such that either Tr( ( e ε ⊤ j ) ⊗ ( vu ⊤ ) ( I -γ P π AB ) -1 ) = 0 or Tr( ( vu ⊤ ) ⊗ ( e ε ⊤ j ) ( I -γ P π AB ) -1 ) = 0 . We assume the former without loss of generality, it holds

̸

<!-- formula-not-decoded -->

Now set r A = 0 and r B = v . Since Q π AB is decomposable, there exists some local function Q A , Q B such that

<!-- formula-not-decoded -->

Left multiply by ( ε ⊤ j ⊗ u ⊤ ) , we have

<!-- formula-not-decoded -->

Then set r A = 0 and r B = -v , we can similarly derive

<!-- formula-not-decoded -->

̸

This gives use ( ε ⊤ j ⊗ u ⊤ )( Q A ( 0 ) ⊗ e ) = 0 , which is a contradiction.

## E Decomposition via general functions

Entangled P precludes the local decomposition with local value functions, but may admit decompositions with more general functions. Consider P = 1 4 ( ee ⊤ ) ⊗ ( ee ⊤ ) + δ ( ϵe ⊤ ) ⊗ ( eϵ ⊤ ) , where e = [1 , 1] , ϵ = [1 -1] . Clearly such P is entangled. We also have P k = 1 4 ( ee ⊤ ) ⊗ ( ee ⊤ ) for k ≥ 2 . Then ( I -γP ) -1 = I + γ + γ 2 4 ( ee ⊤ ) ⊗ ( ee ⊤ ) + δγ ( ϵe ⊤ ) ⊗ ( eϵ ⊤ ) . Then for any r A , r B , we have ( I -γ P ) -1 ( r A ⊗ e + e ⊗ r B ) = r A ⊗ e + h A ( γ + γ 2 ) / 2 e ⊗ e + r B ⊗ e + h B ( γ + γ 2 ) / 2 e ⊗ e +2 δγ ( ϵ ⊤ r B ) ϵ ⊗ e where h A = e ⊤ r A , h B = e ⊤ r B .

## F Proof of Theorem 3

Additional Notations For (semi-)norm ∥ · ∥ α and norm ∥ · ∥ β , we define the α, β -norm for matrix A as

<!-- formula-not-decoded -->

We further abbreviate ∥ A ∥ α := ∥ A ∥ α,α . Moreover, we define the operator | x | taking the absolute value of each element of vector or matrix x .

To prove the theorem, we introduce the key technique of analyzing perturbation bounds of the transition matrix, which is also used in [17].

Lemma 3 (Lemma 1 in [17]) . Let P , P ′ ∈ R n × n such that ( I -P ) -1 and ( I -P ′ ) -1 exist. Then it holds

<!-- formula-not-decoded -->

We are then ready to prove the main theorem.

̸

̸

̸

Proof of Theorem 3. Let P A , P B be the optimal solution to Eq. (7) w.r.t agent A,B . For any subset of state-action pairs of agent A , F ⊆ S A ×A A , we have

<!-- formula-not-decoded -->

where the last inequality follows from the definition of agent-wise total variation distance. Since the result holds for any F and ( s A , a A ) ∈ S A ×A A , we have

<!-- formula-not-decoded -->

and similar results hold for P π B .

Next we have

<!-- formula-not-decoded -->

where ( i ) also follows the same 'absorbing' technique in the proof of Theorem 1. For ( I ) , apply Lemma 3, it holds

<!-- formula-not-decoded -->

̸

where ( i ) follows by the definition of agent-wise total variation distance when ∥ r A ∥ ∞ = 0 , and also trivially hold when ∥ r A ∥ ∞ = 0 . Similarly, for ( II ) we have

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

We can derive similar results for agent B , i.e.,

<!-- formula-not-decoded -->

Put it all together we have

<!-- formula-not-decoded -->

## G Proof of Theorem 4

We first introduce the µ -weighted ATV distance Formally, we introduce the following norm.

Definition 5 ( µ -norm) . Given a transition matrix P ∈ R |S||A|×|S||A| with occupancy measure 5 µ ∈ R |S||A| , for any vector x ∈ R |S||A| the µ -norm is defined as

<!-- formula-not-decoded -->

One can verify that µ -norm satisfies triangle inequality and is a valid norm when µ ( s, a ) &gt; 0 for all ( s, a ) . Otherwise µ -norm is a semi-norm in general. We then introduce the distance

Definition 6 ( µ -weighted Agent-wise Total Variation Distance) . Given probability distribution µ ∈ R | S | 2 | A | 2 , the µ -weighted total variation distance between two transition matrices P , P ′ ∈ R | S | 2 | A | 2 ×| S | 2 | A | 2 w.r.t agent A is defined as

<!-- formula-not-decoded -->

The µ -weighted ATV distance w.r.t agent B can be defined similarly. We claim that the µ -weighted ATV is also a counterpart of ATV distance in Definition 6. This follows from the constrained optimization formulation of ATV

<!-- formula-not-decoded -->

Thus µ -ATV substitutes µ -norm for the original ℓ ∞ -norm. We plug µ -weighted ATV into Eq. (1) and obtain the corresponding measure of Markov entanglement E ( P π AB ) and E A ( P π AB ) . Similar to ATV in Eq. (7), this µ -weighted version of E A ( P π AB ) admits the following formulation

<!-- formula-not-decoded -->

5 Since µ ∈ R |S||A| is the stationary distribution of P ∈ R |S||A|×|S||A| , we use 'stationary distribution" and 'occupancy measure" exchangeably when the context is clear.

This recovers Eq. (8) that substitutes the µ -weighted average for the maximum operator in Eq. (7). Thus intuitively, E ( P π AB ) w.r.t µ -weighted ATV distance measures how closely agent A can be approximated as an independent subsystem under the stationary distribution .

We provide the proof for two agents here, one can easily generalize the proof to multi-agent scenarios. Compared to the proof of Theorem 3, this proof follows similar framework and differs in several details.

The first one is the following lemma for the 'localized' stationary distribution

Lemma 4. P π A has stationary distribution µ π A with

<!-- formula-not-decoded -->

In other words, the local stationary distribution of each agent is exactly the marginal distribution of global µ π AB .

Proof of Lemma 4. We proof by verify the definition of stationary distribution. For any ( s ′ A , a ′ A ) , it holds

<!-- formula-not-decoded -->

where the last equation follows from the definition of µ π AB . Hence we conclude that ∑ s B ,a B µ π AB ( s A , s B , a A , a B ) is a stationary distribution of P π A .

We are then ready to prove Theorem 4. We first note that similar to ATV distance in Eq. (7), the optimal solution to E A ( P π AB ) w.r.t µ π AB -weighted ATV distance also only depends on P A . Thus, let P A , P B be the optimal solutions to E A ( P π AB ) , E B ( P π AB ) respectively.

Let x ∈ R |S A ||A A | with ∥ x ∥ ∞ = 1 . Following the same technique in the proof of Theorem 4, we have

<!-- formula-not-decoded -->

where the second last inequality follows from Lemma 4. We then conclude

<!-- formula-not-decoded -->

and similar results hold for P π B . We then apply the decomposition

For ( I ) , we have

<!-- formula-not-decoded -->

where ( i ) follows from the fact that for any x

<!-- formula-not-decoded -->

For ( II ) one can use Lemma 4 to verify

<!-- formula-not-decoded -->

And similar results to ( I ) holds. We then conclude the proof of Theorem 4.

## H Results for multi-agent MDPs

In quantum physics, the concept of quantum entanglement of two-party system can be well extended to multi-party system. In this section, we demonstrate a similar extension of two-agent Markov entanglement to multi-agent settings. We begin with the model of multi-agent MDPs.

Consider an N -agent MDP M 1: N ( S , A , P , r 1: N , γ ) with joint state space S = × N i =1 S i and joint action space A = × N i =1 A i . For simplicity, we assume |S i | = |S| and |A i | = |A| for each agent i . For agents at global state s = ( s 1 , s 2 , . . . , s N ) with action a = ( a 1 , a 2 , . . . , a N ) taken, the system will transit to s ′ = ( s ′ 1 , s ′ 2 , . . . , s ′ N ) according to transition kernel s ′ ∼ P ( · | s , a ) and each agent i ∈ [ N ] will receive its local reward r i ( s i , a i ) . The global reward r 1: N is defined as the summation of local rewards r 1: N ( s , a ) := ∑ N i =1 r i ( s i , a i ) , or in vector form,

<!-- formula-not-decoded -->

We further assume the local rewards are bounded, i.e. for agent i ∈ [ N ] , | r i ( s i , a i ) | ≤ r i max for all ( s i , a i ) . Given any global policy π : S → ∆( A ) , we denote P π 1: N ∈ R |S| N |A| N ×|S| N |A| N as the transition matrix induced by π where P π 1: N ( s ′ 1: N , a ′ 1: N | s 1: N , a 1: N ) := P ( s ′ 1: n | s 1: N , a 1: N ) π ( a ′ 1: N | s ′ 1: N ) . Then the global Q-value is defined by Bellman Equation Q π 1: N = ( I -γ P π 1: N ) -1 r 1: N . The local Q-values follow the similar framework to Meta Algorithm 1 where each agent i ∈ [ N ] fits Q π i using its local observations. We then sum up local Q-values to approximate the global Q-value, i.e.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To illustrate the extension, we first provide the definition of multi-party quantum entanglement here for reference.

Definition 7 (Multi-party Quantum Entanglement) . Consider a multi-party quantum system composed of N subsystems, indexed by [ N ] . The joint state ρ 1: N is separable if there exists K ∈ Z + , probability distribution { x i } i ∈ [ K ] , and density matrices { ρ ( j ) 1: N } j ∈ [ K ] such that

<!-- formula-not-decoded -->

If there exists no such decomposition, ρ 1: N is called entangled .

Analogically, we define the Multi-agent Markov Entanglement,

Definition 8 (Multi-agent Markov Entanglement) . Consider a N -agent Markov system M 1: N and policy π : S → ∆( A ) , the agents are separable under policy π if there exists K ∈ Z + , measure

{ x j } j ∈ [ K ] satisfying ∑ K j =1 x j = 1 , and transition matrices { P ( j ) 1: N } j ∈ [ K ] such that

<!-- formula-not-decoded -->

If there exists no such decomposition, the agents are entangled under policy π .

For clarity, we use superscript s i to denote the i -th element in state space and subscript s i to represent the state at i -th arm. Furthermore, we denote S -i := S \ s i and s := s 1: N := { s 1 , s 2 , . . . , s N } is the profile of N -arms.

Given any global policy π , for any agent i ∈ [ N ] ,

<!-- formula-not-decoded -->

Definition 9 (Measure of Multi-agent Markov Entanglement) . Consider a N -agent Markov system M 1: N with joint state space S = × N i =1 S i and action space A = × N i =1 A i . Given any policy π : S → ∆( A ) , the measure of Markov entanglement of N agents is

<!-- formula-not-decoded -->

where d ( · , · ) is some distance measure.

The following theorem generalizes the results of value-decomposition for two-agent Markov systems in Theorem 3 to multi-agent Markov systems.

Theorem 7. Consider a N -agent MDP M 1: N with joint state space S = × N i =1 S i and action space A = × N i =1 A i . Given any policy π : S → ∆( A ) with the measure of Markov entanglement E i ( P π 1: N ) w.r.t ATV distance, it holds for any agent i ,

<!-- formula-not-decoded -->

where P i is the optimal solution of Eq. (13). Furthermore, the decomposition error is entry-wise bounded by the measure of Markov entanglement,

<!-- formula-not-decoded -->

The proof mainly follows the following lemma, which generalizes the key technique used in Theorem 1.

Lemma 5. For any agent i , it holds

<!-- formula-not-decoded -->

The lemma follows from the property of tensor product. We can also extend Theorem 4 to multi-agent MDPs.

Theorem 8. Consider a N -agent MDP M 1: N with joint state space S = × N i =1 S i and action space A = × N i =1 A i . Given any policy π : S → ∆( A ) with the measure of Markov entanglement E i ( P π 1: N ) w.r.t the µ π 1: N -weighted agent-wise total variation distance, it holds for any agent i ,

<!-- formula-not-decoded -->

where P i is the optimal solution of Eq. (13) and µ π i is the stationary distribution of the projected transition P π i . Furthermore, the µ π 1: N -weighted decomposition error is bounded by the measure of Markov entanglement,

<!-- formula-not-decoded -->

## I Proof of Theorem 6

We first provide an overview of the proof and introduce the technical assumptions.

To begin, we consider the system configuration m ∈ ∆ |S| where m s = 1 N ♯ { Agents in state s } is the proportion of agents in state s . When N →∞ , the transition between configurations will become deterministic under index policy and m will approach its mean-field limit m ∗ . Furthermore, in the mean-field, each agent's local transition will only depend its local state. As a result, the system will de-couple and become separable as N →∞ .

To formalize this intuition, we introduce the following lemma that connects Markov entanglement measure with the mean-field analysis

Lemma 6. The measure of Markov entanglement w.r.t µ π 1: N -weighted ATV distance is bounded by the deviation of mean-field configuration,

<!-- formula-not-decoded -->

where the expectation is taking over the stationary distribution m ∼ µ π 1: N .

We thus focus on the deviation from m to m ∗ . We extend the concentration analysis from [20, 21] to derive a new stability bound for the RHS. Specifically, we finishing the proof via demonstrating the deviation decays at the rate O (1 / √ N ) .

One caveat here is that we have to restrict chaotic behaviors in the mean-field limit. We thus introduce two technical assumptions.

We first define the transition of configuration under index policy π as ϕ π : ∆ |S| → ∆ |S| such that

<!-- formula-not-decoded -->

For t &gt; 0 , we denote Φ t := ( ϕ π ) t apply the transition mapping for t rounds.

Assumption A (Uniform Global Attractor Property (UGAP)) . There exists a uniform global attractor m ∗ of ϕ π ( · ) , i.e. for all ε &gt; 0 , there exists T ( ε ) such that for all t ≥ T ( ε ) and all m ∈ ∆ |S| , one has ∥ Φ t ( m ) -m ∗ ∥ ∞ &lt; ε .

The UGAP assumption ensures the uniqueness of m ∗ and guarantees fast convergence from any initial m to m ∗ .

Assumption B (Non-degenerate RMAB) . There exists state s ∈ S such that 0 &lt; π ∗ ( s, 0) &lt; 1 , where π ∗ is the policy under m ∗ .

The non-degenerate assumption further restricts cyclic behavior in the mean-field limit.

Non-degenerate and UGAP are two standard technical assumptions for the index policy, which restrict chaotic behavior in asymptotic regime and will be further introduced in subsequent sections. We note here these two assumptions are also used in almost all theoretical work on index policies [47, 41, 20, 21].

Proof of Theorem 6. In the subsequent proof, we let ν 1 &gt; ν 2 &gt; ν 3 &gt; · · · &gt; ν | S | . This does not lose generality in that we can always exchange state index. The proof consists of several steps

Step 1: Find m ∗ Recall the transition mapping for configurations ϕ π : ∆ |S| → ∆ |S| ,

<!-- formula-not-decoded -->

Notice that the definition of ϕ π does not depend on N . We adapt from Lemma B.1 in [20] defined specially for Whittle Index,

Lemma 7 (Piecewise Affine) . Given any index policy π , ϕ π is a piecewise affine continuous function with |S| affine pieces.

When the context is clear, we abbreviate ϕ π as ϕ . For any m ∈ ∆ |S| , define s ( m ) ∈ [ |S| ] be the state such that ∑ s ( m ) -1 i =1 m i ≤ α &lt; ∑ s ( m ) i =1 m i . Lemma 7 characterizes for any m ∈ Z i := { m ∈ ∆ |S| | s ( m ) = i } , there exists K s ( m ) , b s ( m ) such that

<!-- formula-not-decoded -->

By Brouwer fixed point theorem, there exists a fixed point m ∗ such that ϕ ( m ∗ ) = m ∗ . The UGAP condition guarantees the uniqueness of m ∗ . Our choice of π ∗ is the corresponding policy under m ∗ .

## Step 2: Connecting policy entanglement with the deviation of stationary distribution Combine Proposition 9 with the RMAB model, we have

Lemma 8. The measure of Markov entanglement w.r.t µ π 1: N -weighted ATV distance is bounded by the deviation of mean-field configuration,

<!-- formula-not-decoded -->

where the expectation is taking over the stationary distribution m ∼ µ π 1: N .

Proof. Given the homogeneity of agents, we first demonstrate for any two agent i, j , it holds

<!-- formula-not-decoded -->

To see this, we first notice by the definition of index policy

| π ( a i = a | s i = s, m ) -π ∗ ( a | s ) | = | π ( a j = a | s j = s, m ) -π ∗ ( a | s ) | . It then suffices to prove ∑ s i = s,s 1: N = m µ ( s 1: N ) = ∑ s j = s,s 1: N = m µ ( s 1: N ) . If ∑ s i = s,s 1: N = m µ ( s 1: N ) ≤ ∑ s j = s,s 1: N = m µ ( s 1: N ) , we can exchange the agent index of i and j . This will result in the same stationary distribution and ∑ s i = s,s 1: N = m µ ( s 1: N ) ≥ ∑ s j = s,s 1: N = m µ ( s 1: N ) and thus the equation. We then rewrite the bound in Proposition 9,

<!-- formula-not-decoded -->

For any configuration m and state s , we have

<!-- formula-not-decoded -->

where | k s | ≤ ( |S| 1) ∥ m -m ∗ ∥ ∞ N representing the additional fraction of state s to be activated due to the deviation from m ∗ and | ℓ s | ≤ ∥ m -m ∗ ∥ ∞ N representing the deviation of m s from m ∗ s . The results then hold by taking summation over s and expectation over m .

Step 3: Concentrations and local stability To bound E [ ∥ m -m ∗ ∥ ∞ ] , we start with several technical lemmas from previous RMAB literature. We use the same notation Φ t = ϕ (Φ t -1 ) .

Lemma 9 (One-step Concentration, Lemma 1 in [21]) . Let ϵ [1] = m [1] -ϕ ( m [0]) , it holds

<!-- formula-not-decoded -->

Lemma 10 (Multi-step Concentration, Lemma C.4 in [20]) . There exists a positive constant K such that for all t ∈ N and δ &gt; 0 ,

<!-- formula-not-decoded -->

Lemma 11 (Local Stability, Lemma C.5 in [20]) . Under non-degenerate and UGAP:

- (i) K s ( m ∗ ) is a stable matrix, i.e. its spectral radius is strictly less than 1 .
- (ii) For any ϵ , there exists T ( ϵ ) &gt; 0 such that for all m ∈ ∆ |S| , ∥ ∥ Φ T ( ϵ ) ( m ) -m ∗ ∥ ∥ ∞ &lt; ϵ .

The first result implies there exists some matrix norm ∥·∥ β such that ∥ ∥ K s ( m ∗ ) ∥ ∥ β &lt; 1 . By the equivalence of norms, there exists constant C 1 β , C 2 β &gt; 0 such that for all x ∈ R |S|

<!-- formula-not-decoded -->

Combine the second result of Lemma 11 and non-degenerate condition, we can construct a neighborhood N of m ∗ such that N = B ( m ∗ , ϵ ) ∩ ∆ |S| ∈ Z s ( m ∗ ) where ϵ &gt; 0 and B ( m ∗ , ϵ ) = { m | ∥ m -m ∗ ∥ ∞ &lt; ϵ } is an open ball. We next show that m [0] under stationary distribution will concentrate in N with high probability. Let ˜ T = T ( ϵ/ 2) such that for all m ∈ ∆ |S| , ∥ Φ ˜ T ( m ) -m ∗ ∥ ∞ &lt; ϵ/ 2 . It holds

̸

<!-- formula-not-decoded -->

where ( i ) follows from the stationarity m [ ˜ T ] and m [0] are i.i.d and the constant u = ( ϵ 2(1+ K + K 2 + ··· + K ˜ T ) ) 2 does not depend on N .

Step 4: Put it together Finally, we are ready to bound E [ ∥ m -m ∗ ∥ ∞ ] . Notice for all m [0] ∈ N , we have

<!-- formula-not-decoded -->

Taking ∥·∥ β on both side,

<!-- formula-not-decoded -->

Taking expectation on both side,

<!-- formula-not-decoded -->

By stationarity, one have E [ ∥ m [1] -m ∗ ∥ β ] = E [ ∥ m [0] -m ∗ ∥ β ] . This refines the above inequality,

<!-- formula-not-decoded -->

We combine Lemma 8 and conclude the proof of Theorem 6.

## J Extensions of Markov entanglement

## J.1 (Weakly-)coupled MDPs

Weakly-coupled MDPs (WCMDP) are a rich class of multi-agent model that capture many real-world applications such as supply chain management, queuing network and resource allocations [2, 12, 36]. Compared to general multi-agent MDP, WCMDP further ensures each agent follow its local transition while the agents' actions are coupled with each other. Formally,

Definition 10 (Weakly-coupled MDPs) . An N -agent MDP M 1: N ( S , A , P , r 1: N , γ ) is a weaklycoupled MDP if

- Each agent has local transition kernel P i such that ∀ s , a , s ′ , P ( s ′ | s , a ) = ∏ N i =1 P i ( s ′ i | s i , a i ) .
- At global state s , agents' joint actions a are subject to m coupling constraints ∑ N i =1 d ( s i , a i ) ≤ b ∈ R m .

We then demonstrate that this weakly-coupled structure can further refine the analysis of Markov entanglement measure.

Proposition 9. Consider a N -agent weakly-coupled MDP M 1: N ( S , A , P , r 1: N , γ ) . Given any policy π : S → ∆( A ) with measure of Markov entanglement E i ( P π 1: N ) w.r.t the µ π 1: N -weighted agent-wise total variation distance, it holds for i ∈ [ N ] ,

<!-- formula-not-decoded -->

where π ′ : S i →A i is any local policy for agent i .

Proof of Proposition 9. We demonstrate the proof for two-agent WCMDP and the generalization to multi-agent WCMDP is straightforward. Consider P π ′ A be the transition of agent A under local policy

π ′ . We focus on agent A

<!-- formula-not-decoded -->

where ( i ) follows from the transition structure of weakly coupled MDP P ( s ′ | s , a ) = P ( s ′ A | s A , a A ) · P ( s ′ B | s B , a B ) ; and ( ii ) comes from the fact that P π ( s ′ | s ) = ∑ a π ( a | s ) P ( s ′ | s , a ) and ∑ s µ π ( s ) P π ( s ′ | s ) = µ π ( s ′ ) .

Proposition 9 establishes an upper bound for Markov entanglement in WCMDP. Intuitively, this bound characterizes how agent i can be viewed as making independent decisions . It takes advantage of the weakly-coupled structure and shaves off the transition in Markov entanglement measure.

## J.2 Coupled MDPs with exogenous information

In many practical scenarios, the agents' transitions and actions are coupled by a shared exogenous signal. For example, in ride-hailing platforms, the specific dispatch is related to the exogenous order at the current moment [33, 24, 4]; in warehouse routing, the scheduling of robots is also related to the exogenous task revealed so far [15].

We will then enrich our framework by incorporating these exogenous information. At each timestep t , there will an exogenous information z t revealed to the decision maker. z t is assumed to evolve following a Markov chain independent of the action and transition of agents. We assume z t ∈ Z and Z is finite.

Given the current state s and exogenous information z , the policy is given by π : S × Z → ∆( ˜ A ) , where ˜ A refers to the set of feasible actions. We then have the global transition depending on exogenous information z ,

<!-- formula-not-decoded -->

and global Q-value Q π ABz ∈ R |S| N |A| N |Z| ,

<!-- formula-not-decoded -->

We assume the system is unichain and the stationary distribution is µ π ABz . Then we can derive the local transition under new algorithm by

<!-- formula-not-decoded -->

Given the local transition, we have the local value Q π Az = ( I -γ P Az ) -1 ( r Az ) via Bellman Equation.

Combined with exogenous information, we consider the following value decomposition

<!-- formula-not-decoded -->

We start by introducing agent-wise Markov entanglement defined for each agent

<!-- formula-not-decoded -->

Proposition 10. If the system is agent-wise separable for all agents, then

<!-- formula-not-decoded -->

Proof. The proof is basically the same as Theorem 1. One can first quickly show that P Az = ∑ K j =1 x j P ( j ) Az . And then it holds

<!-- formula-not-decoded -->

We then provide the measure of Markov entanglement with exogenous information w.r.t agent-wise total variation distance.

<!-- formula-not-decoded -->

Similar to Theorem 3, we can connect this measure of Markov entanglement with the value decomposition error.

Theorem 11. Consider a N -agent Markov system M 1: N . Given any policy π : S → ∆( A ) with the measure of Markov entanglement E i ( P π 1: N , Z ) w.r.t the agent-wise total variation distance, it holds for any agent i ,

<!-- formula-not-decoded -->

Furthermore, the decomposition error is entry-wise bounded by the measure of Markov entanglement,

<!-- formula-not-decoded -->

In practice, exogenous information is often discussed in the context of (weakly-)coupled MDPs, where each agent independent evolves by P i ( s i +1 | s i , a i , z ) . Interestingly, we can derive a similar result to Proposition 9 that shaves off the transition in entanglement analysis.

Proposition 12. Consider a N -agent Weakly Coupled Markov system M 1: N . Given any policy π : S → ∆( A ) and its measure of Markov entanglement E i ( P π 1: N , Z ) w.r.t the µ π 1: N -weighted agent-wise total variation distance, it holds

<!-- formula-not-decoded -->

for any policies π ′ .

Proof. We provide the proof for two-agent MDP, which can be easily generalized to N -agent case.

<!-- formula-not-decoded -->

## J.3 Factored MDPs

Another common class of multi-agent MDPs is Factored MDPs (FMDPs, [22, 23, 32]), which explicitly model the structured dependencies in state transitions. For instance, in a server cluster, the state transition of each server depends only on its neighboring servers. Formally, we define

Definition 11 (Factored MDPs) . An N -agent MDP M 1: N ( S , A , P , r 1: N , γ ) is a factored MDP if each agent i has neighbor set Z i ∈ [ N ] such that its transition is affected by all its neighbors, i.e. P ( s ′ i | s , a ) = P ( s ′ i | s Z i , a Z i ) .

The neighbor set | Z i | is often assumed to be much smaller compared to the number of agents N . This helps to encode exponentially large system very compactly. We show this idea can also be captured in Markov entanglement. Consider the measure of Markov entanglement w.r.t ATV distance in Eq. (7),

<!-- formula-not-decoded -->

Thus we conclude the agent-wise Markov entanglement will only depend on its neighbor set.

Meta Algorithm 2: Q-value Decomposition with Shared Reward

Require: Global policy π ; horizon length T .

- 1: Execute π for T epochs and obtain D = { ( s t AB , a t AB , r t AB , s t +1 AB , a t +1 AB ) } T -1 t =1 .
- 2: Each agent i ∈ { A,B } fits Q π i using local observations D i = { ( s t i , a t i , r i , s t +1 i , a t +1 i ) } T -1 t =1 where the local reward ( r A , r B ) is learned via solving

<!-- formula-not-decoded -->

## J.4 Fully cooperative Markov games

In fully cooperative settings, only a global reward will be reviewed to all agents. Unlike the modeling in section 2, this global reward may not necessarily be decomposed as the summation of local rewards. In this case, we propose meta algorithm 2 as an extension of meta algorithm 1.

This algorithm follows similar framework of meta algorithm 1 and differs at we now learn the closet local reward decomposition from data. When the reward is completely decomposable, meta algorithm 2 recovers meta algorithm 1. Thus intuitively, the more accurate we can decompose the global reward, the less decomposition error we have. Formally, we define the measure of reward entanglement

<!-- formula-not-decoded -->

This measure characterizes how accurate we can decompose the global reward under stationary distribution. We then obtain an extension of Theorem 4

Proposition 13. Consider a fully cooperative two-agent Markov system M AB . Given any policy π : S → ∆( A ) with the measure of Markov entanglement E A ( P π AB ) , E B ( P π AB ) w.r.t the µ π AB -weighted agent-wise total variation distance and the measure of reward entanglement e ( r AB ) , it holds

<!-- formula-not-decoded -->

where r A max , r B max is the bound of optimal solution of Eq. (17).

Although Proposition 1 offers a theoretical guarantee for general two-agent fully cooperative Markov games, its utility is greatest in systems with low reward and transition entanglement. Fully cooperative settings remain inherently challenging-for instance, even the asymptotically optimal Whittle Index may achieve only a 1 N -approximation ratio for RMABs with global rewards [34]. In practice, most research [38, 35] relies on sophisticated deep neural networks to learn decompositions in such settings. We thus defer a more refined analysis of fully cooperative scenarios to future work.

## K Simulation environments

In this section, we empirically study the value decomposition for index policies. Our simulations build on a circulant RMAB benchmark, which is widely used in the literature [3, 52, 10, 18].

Circulant RMAB A circulant RMAB has four states indexed by { 0 , 1 , 2 , 3 } . Transition kernels P a = p ( s, 0 , s ′ ) s,s ′ ∈ S for action a = 0 and a = 1 are given by

<!-- formula-not-decoded -->

The reward solely depends on the state and is unaffected by the action:

<!-- formula-not-decoded -->

We set the discount factor to γ = 0 . 5 and require N/ 5 arms to be pulled per period. Initially, there are N/ 6 arms in state 0 , N/ 3 arms in state 1 and N/ 2 arms in state 2 , the same as [52]. We then test an index policy with priority: state 2 &gt; state 1 &gt; state 0 &gt; state 3 .

## K.1 Monte-Carlo estimation of Markov entanglement

For each RMAB instance, we simulate a trajectory of length T = 6 N and collect data for the later 5 N epochs. Notice RMAB is a special instance of WCMDP, we thus apply the result in Proposition 9

<!-- formula-not-decoded -->

Notice Eq. (18) is convex for π ′ and π ′ only takes support of size | S || A | = 8 . we thus apply efficient convex optimization solvers. We replicate this experiment for 10 independent runs to obtain the mean estimation and standard error in the left panel of Figure 1.

## K.2 Learning local Q-values

For each RMAB instance, we simulate a trajectory of length T = 6 N , reserving the later T = 5 N epochs as the training phase for each agent to fit local Q-value functions. During testing, we estimate the µ -weighted decomposition error using 50 simulations sampled from the stationary distribution.

The ground-truth Q π 1: N is approximated via Monte Carlo learning [39], with each estimate derived from 30 -step simulations averaged over 3 N independent runs. Due to the high computational cost of Monte Carlo methods-especially for very large RMABs-we limit the training phase to 10 independent runs and use the mean local Q-value as an approximation. Error bars represent the standard error for both Monte Carlo estimates and µ -weighted decomposition errors.

In addition to µ -weighted error, we also introduce a concept of relative error, defined as ∥ ∥ ∥ Q π 1: N ( s , a ) -∑ N i =1 Q π i ( s i , a i ) ∥ ∥ ∥ µ π 1: N / ∥ Q π 1: N ∥ µ π 1: N . This relative error reflects the approximate ratio of our value decomposition. We present our simulation results below.

Figure 2: Value Decomposition error in circulant RMAB under an index policy. Left: µ -weighted decomposition error. Right: Relative error, ∥ decomposition error ∥ µ / ∥ Q π 1: N ∥ µ

<!-- image -->

<!-- image -->

It immediately follows that the µ -weighted error grows at a sublinear rate O ( √ N ) and the relative error decays at rate O (1 / √ N ) . This justifies our theoretical guarantees in Theorem 6. Furthermore, we notice the relative error is no larger than 3% over all data points. As a result, the meta algorithm 1 is able to provide a very close approximation especially for large-scale MDPs even with small amount of training data T = 5 N while the global state space has size | S | N .

## K.3 Sample Complexity and Computation

While each RMAB instance has an exponentially large state space | S | N , we show that our empirical estimation of Markov entanglement-along with the decomposition error-converges quickly. Specifically, we illustrate these errors for an RMAB instance with with 900 agents in Figure 3. As exhibits in Figure 3, both errors decay and converges within T = 3 N samples. Furthermore,

Figure 3: Different errors in RMAB with 900 agents: empirical estimation of Markov entanglement (blue); µ π 1: N -weighted decomposition error (green); the true measure of Markov estimated with T = 10 N samples (red dashed line).

<!-- image -->

the empirical estimation of Markov entanglement converges in T &lt; N samples, demonstrating its efficiency. Finally, we use standard convex optimization solvers to compute Markov entanglement, which can be run efficiently on a single CPU.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes. Our main contributions are also detailed in section 1.1. Also see Appendix J, H for more theoretical results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, we discuss other possible value decompositions in section 3.1 and Appendix E, J.4.

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

Justification: Theorem 1, 2, 3 and 4 hold for general multi-agent as long as a stationary distribution exists (see section 2). Theorem 6 relies on standard assumptions for index polices, detailed in Appendix I. All proofs are included in main text or Appendix.

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

Justification: Our empirical results build on a publicly-accessible RMAB benchmark in [3, 52, 10, 18], detailed in Appendix K. We upload the codes and instructions to recover the results.

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

Justification: Our empirical results build on a publicly-accessible RMAB benchmark in [3, 52, 10, 18], detailed in Appendix K. We upload the codes and instructions to recover the results.

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

Justification: See Appendix K.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We detail the calculation of error bars in Appendix K.

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

Justification: See Appendix K.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We followed the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: This work focuses on establishing a new mathematical foundation for MARL. This work is not related to any private or personal data, and there's no explicit negative social impacts.

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

Justification: We do not foresee any high risk for misuse of this work.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use existing assets and our empirical simulations are based on synthetic models, whose proposers have been appropriately cited.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
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

Justification: We do not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
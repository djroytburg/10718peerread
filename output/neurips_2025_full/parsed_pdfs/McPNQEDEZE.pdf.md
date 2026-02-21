## Regret Bounds for Adversarial Contextual Bandits with General Function Approximation and Delayed Feedback

## Orin Levy ∗

Blavatnik School of Computer Science Tel Aviv University orinlevy@mail.tau.ac.il

## Alon Cohen

School of Electrical Engineering Tel Aviv University and Google Research alonco@tauex.tau.ac.il

## Liad Erez ∗

Blavatnik School of Computer Science Tel Aviv University liaderez@mail.tau.ac.il

## Yishay Mansour

Blavatnik School of Computer Science Tel Aviv University and Google Research mansour.yishay@gmail.com

## Abstract

We present regret minimization algorithms for the contextual multi-armed bandit (CMAB) problem over K actions in the presence of delayed feedback, a scenario where loss observations arrive with delays chosen by an adversary. As a preliminary result, assuming direct access to a finite policy class Π we establish an optimal expected regret bound of O ( √ KT log | Π | + √ D log | Π | ) where D is the sum of delays. For our main contribution, we study the general function approximation setting over a (possibly infinite) contextual loss function class F with access to an online least-square regression oracle O over F . In this setting, we achieve an expected regret bound of O ( √ KT R T ( O ) + √ d max Dβ ) assuming FIFO order, where d max is the maximal delay, R T ( O ) is an upper bound on the oracle's regret and β is a stability parameter associated with the oracle. We complement this general result by presenting a novel stability analysis of a Hedge-based version of Vovk's aggregating forecaster as an oracle implementation for least-square regression over a finite function class F and show that its stability parameter β is bounded by log |F| , resulting in an expected regret bound of O ( √ KT log |F| + √ d max D log |F| ) which is a √ d max factor away from the lower bound of Ω( √ KT log |F| + √ D log |F| ) that we also present.

## 1 Introduction

Contextual Multi-Armed Bandit (CMAB) is a natural extension of the well-studied Multi-Armed Bandit (MAB) model that has gained considerable attention over the past decade. (See, e.g., [24, 34]). CMAB describes a sequential decision-making problem with exterior factors that affect the decision taken by the learner. We refer to this side information as the context . In this setting, the context x is revealed to the learner at the start of each round. Then, the learner chooses an action a out of a finite set A containing K actions and suffers a loss for that choice, where the context determines the loss. The learner's goal is to minimize the cumulative loss incurred throughout an interaction of T rounds. In this model, action selection strategies are context-dependent and referred to as policies , and the

* Equal contribution.

learner ultimately aims to minimize regret , that is, the learner's cumulative loss in comparison to that of the best contextual action selection rule, i.e., the optimal policy .

CMAB can describe various real-life online scenarios where there are external factors that affect the loss incurred by any choice of action. One such application is online advertising, where the reaction of a user to a presented advertisement (i.e., clicking or ignoring) is heavily dependent on the user's needs (e.g., if they would like to buy a new car), hobbies, and personal preferences. All of the above can be encoded in the user's browsing history and cookies. Thus, the user's cookies can refer to the external factors that affect the user's implied loss. CMAB has been studied under various assumptions and frameworks, which we review in the sequel. In this paper, we consider the adversarial CMAB model (see, e.g., [6, 13]), where the context in each round is chosen by a possibly adaptive adversary from a (possibly infinite) context space X .

Returning to the online advertising example, in such an application, delayed feedback is practically unavoidable. Consider the scenario where a sequence of users enters the application one after another. The algorithm then needs to present them with advertisements, even though the feedback of previous users has not arrived yet. As the application takes time to process each user's feedback, observations will arrive with an inherent delay. In other real-life scenarios such as communication on a physical network, it is also natural to assume that observations arrive at the order in which they are distributed into the network; that is, they arrive in a First-In-First-Out (FIFO) order. In the main theoretical framework of general function approximation that we consider in this work, this additional assumption enables us to obtain highly nontrivial regret guarantees. Such real-life applications motivate the setting of MAB with delayed feedback , which has also vastly studied in recent years, either when the environment is adversarial [8, 10, 35] or stochastic [17, 21, 37]. This leads us to the following fundamental question: What are the achievable regret guarantees in adversarial CMAB under adversarial delayed feedback? In this work, we address this question by considering the two main settings studied in adversarial CMAB literature, and derive delay-robust algorithms for them.

We start with the simpler setting of policy class learning , considered in [5, 12] where the context is stochastic, and in [6] for adversarial contexts. In this setting, the learner has direct access to a finite class Π ⊆ A X of deterministic mappings from contexts to actions (i.e., policy class), and its performance is compared against the best policy in Π . We note that using policy class-based approaches, a running-time complexity of O ( | Π | ) is unavoidable in general. It is thus natural to also consider the more challenging setting of general function approximation [4, 13, 16, 33], with the goal of obtaining an algorithm that is both computationally efficient and enjoys rate-optimal regret.

In the function approximation framework, the learner has indirect access to a class of loss functions F ⊆ [0 , 1] X×A where each function defines a mapping from context and action to a loss value in [0 , 1] . They also assume realizability, meaning the true loss function f ⋆ is within the class, i.e., f ⋆ ∈ F . The learner accesses the function class via an online regression oracle, and measures its performance with respect to that of the best contextual policy π ⋆ : X → A of the true CMAB. In this setting, in addition to the standard least-squares regret assumption required from the oracle, our approach will require a stability assumption that will be discussed later. Furthermore, we present novel stability analysis of a Hedge-based version of Vovk's aggregating forecaster [39] and derive an expected regret bound for this setting, assuming finite and realizable loss function classes.

Summary of our main contributions. We present delay-adapted algorithms for CMAB with general function approximation and analyze the regret of the proposed methods. In more detail, our main results are summarized as follows:

(1) For the policy class learning setup, we establish a regret bound of O ( √ KT log | Π | + √ D log | Π | ) where D is the sum of delays. This bound is optimal, as stated in our lower bound in Corollary C.3.

(2) Given access to a finite contextual loss function class F via an online least-square regression oracle O F sq over F , we present a delay-adapted version of function approximation methods for CMAB, as specified in Algorithm 1. This algorithm can be seen as a delay-adapted version of SquareCB [13], formalized using techniques presented in [15, 26]. For this algorithm we prove in Theorem 4.6 an expected regret bound of O ( √ KT R T ( O F sq ) + √ d max Dβ ) assuming observations arrive in FIFO order, where d max is the maximal delay, R T ( O ) is an upper bound on the oracle's regret and β is a parameter given by an additional assumption that the oracle in use is sufficiently stable. We also prove (in Appendix C.1) a lower bound showing that without an additional assumption on the oracle, no algorithm can guarantee sublinear regret in the presence of delays in the function approximation

setting. To our knowledge, our work is the first to consider delayed feedback in adversarial CMAB in the fully general function approximation framework.

(3) To complement and strengthen this result, we analyze a hedge-based version of Vovk's aggregating forecaster [39] as an online least-squares regression oracle for finite loss function classes. We show that it enjoys a constant expected regret while also exhibiting nontrivial cumulative stability guarantees for realizable finite classes F (Theorem 4.11), which implies a constant bound on its stability parameter β ≤ log |F| and in turn an expected regret bound of O ( √ KT log |F| + √ d max D log |F| ) for Algorithm 1 when used with this oracle. We emphasize that the proof of this oracle's stability properties constitutes a significant part of the technical novelties of our work.

## 1.1 Additional related work

Contextual MAB. CMAB has been vastly studied over the years, under diverse assumptions, regarding the contexts, the function class or the oracles in use, if any. Previous works divide into two main lines. The first is policy class learning, starting from the fundamental EXP4 algorithm for adversarially chosen contexts [6], to Agarwal et al. [5], Dudik et al. [12] that consider stochastic contexts and present computationally efficient algorithms for this problem. They obtain an optimal regret of ˜ O ( √ KT log | Π | ) . Dudik et al. [12] also considered constant delayed feedback d and obtained regret bound of ˜ O ( √ K log | Π | ( d + √ T )) .

The second line is the realizable function approximation setting, which has also been studied for stochastic CMAB, starting from Langford and Zhang [23] to Agarwal et al. [3], Simchi-Levi and Xu [33], Xu and Zeevi [40] in which an optimal regret of ˜ O ( √ TK log |F| ) has been shown, where F ⊆ [0 , 1] X×A is a finite contextual reward or loss function class, accessed via an offline regression oracle. Adversarial CMAB has also gained much attention recently, in the following significant line of works [13, 14, 16, 42], where an online regression is being used to access the function class F , with an optimal regret bound of ˜ O ( √ KT R T ( O )) , where R T ( O ) is the oracle's regret.

Regret guarantees for linear CMAB first studied by Abe and Long [2] and the SOTA algorithms are those of Abbasi-Yadkori et al. [1], Chu et al. [11]. Contextual MDPs (which are an extension of MAB, that has multiple states and dynamics) have been studied under function approximation assumptions for both stochastic context [25, 27, 32] and adversarial contexts with Levy et al. [26] being the most relevant to our setting as it studies adversarial CMDP, and inspired our algorithm and analysis. Generalized Linear CMDPs and smooth CMDPs have also been studied, see, e.g., [31, 30].

Online Learning with Delayed Bandit Feedback. Delayed feedback has been an area of considerable interest in various online MAB problems in the past few years, with the first work on adversarial MABwith a constant delay d by Cesa-Bianchi et al. [10]. Subsequent results for adversarial MAB with arbitrary delays have been established by [8, 35], with Thune et al. [35] being the first work to introduce the skipping technique which adapts to delay sequences that may contain a relatively small number of very large delays. Zimmert and Seldin [43] proposed the first algorithm for adversarial MABwith arbitrary delays that does not require any prior knowledge of the delays.

The study of delayed feedback in MAB has also been extended in several works to more general learning settings. Such settings include linear bandits [19, 37], generalized linear bandits [18], combinatorial semi-bandits [36] and bandit convex optimization [28]. Another prevalent generalization of MAB, in which delayed feedback has been studied, is Reinforcement Learning, specifically tabular MDPs [20, 22, 36], with Jin et al. [20] who first suggested the use of biased delay-adapted loss estimators which inspired our loss estimators used in Algorithm 3.

In CMAB, delayed feedback is far less explored. In the framework of function approximation, Vernade et al. [38] consider the linear case with stochastic delays, and Zhou et al. [41] study generalized linear CMAB with stochastic delays and contexts; both of which are special cases of the general function approximation setting studied in this paper. For stochastic contexts, we believe that obtaining delay-adapted regret bounds in the general function approximation setting can be done by extending the approach of Simchi-Levi and Xu [33], which operates in phases of exponentially increasing lengths. It seems that the presence of delays in this setting will only affect the regret for rounds in which the delay is larger than current batch size, which quickly becomes much larger than the maximal delay. We thus focus on the adversarial setting where it is much less clear how to handle delayed feedback.

## 2 Problem Setup

We consider adversarial contextual MAB (CMAB) with adversarial delayed bandit feedback .

Contextual MAB. Formally, CMAB is defined by a tuple ( X , A , ℓ ) where X is the context space, which is assumed to be large or even infinite, and A = { 1 , 2 , . . . , K } is a finite action space. ℓ : X × A → [0 , 1] forms an expected loss function, that is, for ( x, a ) ∈ X × A , ℓ ( x, a ) = E [ L ( x, a ) | x, a ] where L ( x, a ) ∈ [0 , 1] is sampled independently from an unknown distribution, related to the context x and the action a . In adversarial CMAB , the learner faces a sequential decisionmaking game that is played for T rounds according to the following protocol, for t = 1 , 2 , . . . , T : (1) Adversary reveals a context x t ∈ X to the learner; (2) Learner chooses action a t ∈ A and suffers loss L ( x t , a t ) .

A policy π defines a mapping from context to a distribution over actions, i.e., π : X → ∆( A ) . The learner's cumulative performance is compared to that of the best (deterministic) policy π ⋆ : X → A .

Delayed feedback. The learner observes delayed bandit feedback, where the sequence of delays can be adversarial. Formally, delays are determined by a sequence of numbers d 1 , . . . , d T ∈ { 0 , 1 , . . . , T } . In each round t , after choosing an action a t , the learner observes the pairs ( s, L ( x s , a s )) for all rounds s ≤ t with s + d s = t ; crucially, only the loss values are delayed, whereas the contexts x t are each observed at the start of round t . We consider a setting where the sequence of delays ( d t ) T t =1 as well as the contexts ( x t ) T t =1 are generated by an adversary. 0 We denote the sum of delays by D = ∑ d t =1 d t and the maximal delay by d max = max t ∈ [ T ] d t .

Learning objective. We aim to minimize regret , which is the difference between the cumulative loss of the learner and that of the best-fixed policy π ⋆ , i.e., R T := ∑ T t =1 ℓ ( x t , a t ) -ℓ ( x t , π ⋆ ( x t )) .

We consider two different learning settings for CMAB with delayed feedback. The first is Policy Class Learning , in which the CMAB algorithm has direct access to a finite policy class Π ⊆ A X . In this setting, the benchmark π ⋆ is the best policy among the class, i.e., π ⋆ ∈ arg min π ∈ Π ∑ T t =1 ℓ ( x t , π ( x t )) . Next, we consider the setting of Online Function Approximation , where the CMAB algorithm has access to a realizable contextual loss class F ⊆ X × A → [0 , 1] , where realizability means that there exists a function f ⋆ ∈ F such that for all ( x, a ) ∈ X × A it holds that f ⋆ ( x, a ) = ℓ ( x, a ) . Then, the learner's goal is to compete against π ⋆ ( x ) ∈ arg min a ∈A f ⋆ ( x, a ) , for all x ∈ X .

## 3 Warmup: Policy Class Learning

We begin with a simple formulation of the CMAB problem which considers a finite but structureless policy class Π ⊆ A X , indexed by Π = { π 1 , . . . , π N } . We remark that in this formulation, the loss vectors ( L ( x t , · )) T t =1 may also be generated by an adversary.

## 3.1 Algorithm: EXP4 with Delay-Adapted Loss Estimators

For this setting, we present a variant of the well-studied EXP4 algorithm [6] (fully presented in Appendix A), that incorporates delay-robust loss estimators specialized to the CMAB setting. At a high-level, using direct access to Π , the algorithm performs multiplicative weight updates over the N -dimensional simplex ∆ N , while using all of the feedback that arrives in each round t to construct loss estimators, denoted by ˆ c t ∈ R N + and defined in Equation (2). These estimators are inspired by Jin et al. [20], and are reminiscent of the standard importance-weighted loss estimators, with an additional term in the denominator which induces an under-estimation bias and allows for a simplified analysis. Interestingly, these estimators exhibit a coupling between the context x t , which arrives at a given round t , and the sampling distribution p t + d t from a future round, and can be thought of as a mechanism that incentivizes actions whose sampling probability has increased between rounds t and t + d t , with respect to the context x t . The main result for Algorithm 3 is given bellow.

Theorem 3.1. Algorithm 3 attains expected regret bound of

<!-- formula-not-decoded -->

0 In the function approximation setting we assume the delay sequence satisfies a FIFO property, see Section 4 for details.

where the expectation is over the algorithm's randomness.

<!-- formula-not-decoded -->

Intuitively, in the policy class setting we can directly optimize over Π by using Hedge updates which exhibit stability properties that are crucial in the presence delayed feedback. Such stability properties are much harder to obtain in the function approximation setting where the contextual bandit algorithm does not have direct access to the policy class, and in particular is required to be computationally efficient; see Section 4 for details. We also remark that Algorithm 3 requires an upper bound on the sum of delays D , however, it can be made adaptive by utilizing a 'doubling' mechanism as suggested in, e.g., [22]. The description of Algorithm 3 and the proof of Theorem 3.1 appear in Appendix A.

Matching lower bound. In Appendix C.2, we prove a matching lower bound showing that the upper bound obtained in Theorem 3.1 is optimal up to constant factors. To our knowledge, this is the first tight lower bound that applies to policy class learning with delayed feedback, as previous lower bounds (e.g., [10]) exhibit a delay dependence of √ D rather than √ D log N .

## 4 Online Function Approximation

In this section, we provide regret guarantees for CMAB with delayed feedback under the framework of online function approximation [13, 16]. In this setting, the learner has access to a class of loss functions F ⊆ X × A → [0 , 1] , where each function f ∈ F maps a context x ∈ X and an action a ∈ A to a loss ℓ ∈ [0 , 1] . We use F to approximate the context-dependent expected loss of any action a ∈ A for any context x ∈ X . The CMAB algorithm can access F using an online least-squares regression (OLSR) oracle that will operate under the following standard realizability assumption.

Assumption 4.1. There exists a function f ⋆ ∈ F such that for all ( x, a ) ∈ X ×A , f ⋆ ( x, a ) = ℓ ( x, a ) .

We assume access to a classical, non-delayed, online regression oracle with respect to the square loss function h sq (ˆ y, y ) = (ˆ y -y ) 2 . The oracle, which we denote by O F sq , is given as input at each round t the past observations ( x s , a s , L s ( x s , a s )) t -1 s =1 and outputs a function ˆ f t : X × A → [0 , 1] . A general formulation of the online oracle model is discussed in Foster and Rakhlin [13]. We make use of the following standard online least-squares expected regret assumption of the oracle:

Assumption 4.2 (Least-Squares Oracle Regret) . The oracle O F sq guarantees that for every sequence { ( x t , a t , L t ) } T t =1 where a t ∼ p t and L t ∈ [0 , 1] has E [ L t | x t , a t ] = ℓ ( x t , a t ) , the expected leastsquares regret is bounded as ∑ T t =1 E [( ˆ f t ( x t , a t ) -ℓ ( x t , a t )) 2 ] ≤ R T ( O F sq ) .

Note that a stronger high-probability version of Assumption 4.2 is made in Foster and Rakhlin [13] in order to prove high probability regret bounds for CMAB, but for our use the weaker version suffices. Assumption 4.1 and Assumption 4.2 (or variants of it for other loss functions) are necessary to derive regret bounds for adversarial CMAB and are extensively used literature (see e.g., [13, 14]). However, a general implementation of online least-square regression oracle for a function class might be unstable. To justify the importance of stability, we show in the following result (proven in Appendix C.1) that Assumption 4.1 and Assumption 4.2 alone do not suffice for sublinear regret with function approximation in the presence of delays.

Theorem 4.3. For any CMAB algorithm ALG in the function approximation setting there exists a contextual bandit instance with fixed delay d = 1 over a realizable loss class F with |F| = T +1 and an online oracle O F sq satisfying R T ( O F sq ) = 0 , on which ALG attains regret R T = Ω( T ) .

Hence, we impose the following stability assumption on the oracle.

Assumption 4.4 ( β -stability) . Let ˆ f 1 , ˆ f 2 , . . . , ˆ f T denote the function sequence outputted by the non-delayed oracle O F sq on the observation sequence { ( x t , a t , L t ) } T t =1 . We assume that for some β &gt; 0 , it holds that E [ ∑ T t =1 ∥ ˆ f t -ˆ f t +1 ∥ 2 ∞ ] ≤ β, where in ∥·∥ ∞ we take supremum over x ∈ X and a ∈ A and the expectation is over the loss realizations { L t } T t =1 .

We denote a β -stable OLSR oracle for the function class F by O F ,β sq . As a specific example of such an oracle, we present a Hedge-based version of Vovk's aggregating forecaster (Algorithm 2) and prove that it guarantees least-squares regret of O (log |F| ) while simultaneously satisfying Assumption 4.4 with β = O (log |F| ) , which we use to derive an expected regret bound for this setting.

Our use of a non-delayed OLSR oracle to handle delayed CMAB setup with function approximation requires us to make an assumption on the delay sequence, namely, that observations arrive in FIFO order. 1 This is formalized in the following assumption and further explained in the following.

Assumption 4.5 (FIFO) . We assume the delay sequence ( d 1 , . . . , d T ) satisfies s + d s ≤ t + d t whenever 1 ≤ s ≤ t ≤ T . In particular, if the observation from time t does not arrive (that is, t + d t &gt; T ) then neither do all observations from rounds t ′ &gt; t .

## 4.1 Algorithm: Delay-Adapted Function Approximation for CMAB

Wepresent Algorithm DA-FA (Algorithm 1) for regret minimization in CMAB with delayed feedback for the function approximation framework. Algorithm 1 essentially uses the most up-to-date approximation of the loss until delayed observations arrive. When they arrive, the algorithm feeds them to the oracle one by one, ignores the midway approximations, and uses only the newest loss approximation. In each round t = 1 , 2 , . . . , T the algorithm operates as follows. Let α ( t ) &lt; t denote the number of observations that arrived at round t . Denote these observations by { ( s t i , L ( x s t i , a s t i )) } α ( t ) i =1 , where s t 1 ≤ . . . ≤ s t α ( t ) denote the time steps of the non-delayed related context and action associated with these delayed loss observations. It then holds that s t i + d s t i = t for all i ∈ [ α ( t )] . Note that we assume that the delayed observations arrive in FIFO order, meaning that the delayed observation from round τ always arrives before (or in parallel to) that of round τ +1 for all τ ∈ [ T ] . Then, for i = 1 , . . . , α ( t ) , we feed the oracle with the example ( x s t i , a s t i , L ( x s t i , a s t i )) by order and observe the predicted function ˆ f t -d s t i . Let τ t = t -d s t α ( t ) = s t α ( t ) denote the index of the last observed delayed loss. After processing all the data that arrived, the current context x t is revealed and the algorithm uses the last predicted function ˆ f τ t to solve the regularized convex optimization problem specified in Equation (1), and plays an action sampled from the resulted distribution.

## Algorithm 1 Delay-Adapted Function Approximation for CMAB (DA-FA)

- 1: inputs: Function class F for loss approximation, learning rate γ , β -stable OLSR oracle O F ,β sq .
- 2: for round t = 1 , . . . , T do
- 3: observe α ( t ) &lt; t losses { ( s t i , L ( x s t i , a s t i )) } α ( t ) i =1 where ∀ i ∈ [ α ( t )] , s t i + d s t i = t and s t 1 ≤ . . . ≤ s t α ( t ) .
- 4: for i = 1 , 2 , . . . , α ( t ) do
- 5: update O F ,β sq with the example (( x s t i , a s t i ) , L ( x s t i , a s t i )) .
- 6: observe the oracle's output ˆ f t -d s t i ←O F ,β sq .
- 7: let τ t := t -d s t α ( t ) = s t α ( t ) denote index of the last observed loss.
- 8: use ˆ f τ t as the current loss approximation.
- 9: observe context x t ∈ X .
- 10: solve

<!-- formula-not-decoded -->

- 11: play the action a t sampled from p t

The regret bound for Algorithm 1 is given in the following.

Theorem 4.6. Let γ = √ KT/ R T O F ,β sq . Then Algorithm 1 has an expected regret bound of

<!-- formula-not-decoded -->

1 In particular, this includes the case of fixed delay d .

In particular, this implies the expected regret is bounded as

<!-- formula-not-decoded -->

We remark that Theorem 4.6 actually holds with high probability whenever Assumption 4.2 and Assumption 4.4 hold in high probability rather than in expectation.

Why FIFO order is needed. In the following analysis, we make use of the assumption that the observations arrive in FIFO order to argue that the realized functions that the oracle outputs throughout the process correspond to functions that are outputs of the oracle on the non-delayed observation sequence. Otherwise, the delay can cause a permutation in the order of observations, inducing a sequence of realized outputs that might be different than those of the non-delayed oracle, in which case we cannot relate the realized regret to the regret of the oracle on the non-delayed sequence.

Computational efficiency. The optimization problem in Equation (1) is convex and can be solved efficiently to arbitrary precision. Thus Algorithm 1 is clearly efficient, assuming an efficient oracle.

## 4.2 Analysis

In this subsection, we analyze Algorithm 1, proving Theorem 4.6. Our main technical challenge is reflected in the regret decomposition. As in all previous literature regarding delayed feedback, the main challenge is to derive a bound where the sum of delays D is separated from the number of actions K . Usually, this separation is obtained by an appropriate choice of loss estimators. In our case, however, the loss is estimated by the oracle, and hence not transparent to the algorithm. Our way to create the desired separation is via the regret decomposition described in the following. Let { ˆ f 1 , ˆ f 2 , . . . , ˆ f T } denote the functions predicted by the OLSR oracle on the non-delayed observation sequence { ( x 1 , a 1 , L ( x 1 , a 1 )) , . . . , ( x T , a T , L ( x T , a T )) } . That is, ˆ f i +1 = O F ,η sq ( · ; ( x 1 , a 1 , L ( x 1 , a 1 )) , . . . , ( x i , a i , L ( x i , a i ))) , ∀ i ∈ [ T -1] . For convenience, we denote the optimal (randomized) policy by p ⋆ ( ·| x ) for all x ∈ X . Then, the regret is given by R T = ∑ T t =1 ( p t -p ⋆ ( · | x t )) · ℓ ( x t , · ) and can be decomposed and bounded as follows.

<!-- formula-not-decoded -->

In the above decomposition, term ( a ) is the regret on the first d max steps and hence bounded trivially by d max . Term ( b ) is the regret with respect to the approximated delayed loss. Term ( c ) is the approximation error with respect to the policy induced by p t when considering the non-delayed approximated loss, and will be bounded by the oracle's regret. Term ( d ) is the approximation error with respect to the optimal p ⋆ ( ·|· ) when considering the non-delayed approximated loss. We remark that the Assumption 4.5 is used when bounding terms ( c ) and ( d ) . Lastly, term ( e ) is the regret caused by the delay drift in approximation. This term will be shown to be bounded by √ d max Dβ , with no direct dependence on the number of actions K . We bound each term individually in the following lemmas, and then combine the results to conclude Theorem 4.6. We begin with term ( b ) , whose bound follows from first-order optimality conditions for convex optimization.

Lemma 4.7 (Term (b) bound) . It holds true that

<!-- formula-not-decoded -->

Term ( c ) is bounded using the AM-GM inequality, and applying Assumption 4.2.

Lemma 4.8 (Term (c) bound) . It holds that

<!-- formula-not-decoded -->

Term ( d ) is bounded using the AM-GM inequality to change the measure from p ⋆ ( ·| x t ) to p t , to then apply the non-delayed oracle's regret bound.

Lemma 4.9 (Term (d) bound) . The following holds true

<!-- formula-not-decoded -->

The proofs of Lemmas Lemmas 4.7 to 4.9 are inspired by those of Levy et al. [26], and included for completeness in Appendix B.1.

Lastly, we bound the delay-dependent term ( e ) . This is where we need to make use of the oracle's stability given in Assumption 4.4 in order to obtain the bound given in the following lemma, whose full proof can also be found in Appendix B.1.

Lemma 4.10 (Term (e) bound) . Under Assumption 4.4, the following holds true.

<!-- formula-not-decoded -->

Proof sketch. Using Hölder's inequality, it holds that

<!-- formula-not-decoded -->

where σ t is the number of pending observations (that is, which have not yet arrived) as of round t . Taking expectation while using Jensen's inequality, the Cauchy-Schwarz inequality and Assumption 4.4, the above can be further bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used that σ t ≤ d max and ∑ t σ t = D .

We now have what we need to prove Theorem 4.6.

Proof of Theorem 4.6. Putting Lemmas 4.7 to 4.10 together, the expected regret of Algorithm 1 bounded by E [ R T ] ≤ d max + 2 KT γ + 2 γ R T ( O F ,β sq ) + 2 √ d max Dβ. Choosing γ = √ KT R T ( O F ,β sq ) yields the bound.

To prove the second statement of the theorem, we note that whenever an observation from given round t arrives with the maximal delay d max , Assumption 4.5 implies that all observations from rounds t ′ ∈ { t +1 , . . . , t + d max -1 } arrive after round t + d max . Therefore, we can lower bound the sum of delays as a function of d max as

<!-- formula-not-decoded -->

Therefore, d max = O ( √ D ) and the bound follows.

## 4.3 Stability analysis of Hedge-based Vovk's aggregating forecaster

In this section, we present a concrete online least squares regression oracle implementation that enjoys an expected square-loss regret bound of O (log |F| ) while simultaneously satisfying Assumption 4.4 with β ≲ log |F| (see Corollary 4.13 and Lemma 4.12). We then use this result to derive an expected regret bound of O ( √ KT log |F| + √ d max D log |F| ) for Algorithm 1 using this oracle. The oracle is a Hedge-based version of Vovk's aggregating forecaster [39] applied for the square loss (see Algorithm 2 for details). Interestingly, even though Algorithm 2 uses a constant (that is, large) step size, its square-loss regret is independent of T , and perhaps more surprisingly, the expected sum of KL-deviations between consecutive iterates q t and q t +1 is also independent of T . This crucial property allows us to use this general purpose oracle in order to obtain a non-trivial regret bound in the general function approximation setting.

Hedge-based version of Vovk's aggregating forecaster. Algorithm 2 performes Hedge updates over

Algorithm 2 Hedge-based Vovk's aggregating forecaster

- 1: parameters: (finite) function class F ⊆ X × A → [0 , 1] , step size η &gt; 0
- 2: Initialize q 1 ∈ ∆ F as the uniform distribution over F .
- 3: for round t = 1 , 2 , . . . , T : do
- 4: Return ˆ f t = ∑ f ∈F q t ( f ) f to contextual bandit algorithm.
- 5: Observe feedback z t = ( x t , a t ) and realized loss y t ∈ [0 , 1] .
- 6: Update q t as follows: q t +1 ( f ) ∝ q t ( f ) e -η ( f ( z t ) -y t ) 2 , ∀ f ∈ F .

the finite function class F using the squared loss. That is, it maintains a distribution over functions q t ∈ ∆( F ) and in each round t returns an aggregation of the functions in F by the weights q t .

In the next theorem, we prove an expected regret bound and use it to establish stability guarantees for Algorithm 2. We emphasize that while regret analyses of Vovk's aggregating forecaster exist in the literature (e.g. [9]), the resulting stability property is novel, and in particular only holds under realizability. The full proof appears in Appendix B.2.

Theorem 4.11 (Regret and stability guarantee for finite function classes) . For t ∈ [ T ] denote by q t ∈ ∆( F ) the probability measure over functions in F computed by Algorithm 2 at time step t .

Then, the following holds for any η ≤ 1 / 18 : (1) Expected regret:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the expectation is taken over the loss realizations y 1 , . . . , y T .

Regret bound. We use the latter result to derive an oracle-specific regret bound for finite and realizable function classes. Our regret bound, stated in Corollary 4.13, follows from Theorem 4.11 using Lemma 4.12, which proves that Algorithm 2 with constant step size satisfies Assumption 4.4 for β = O (log |F| ) (see the next lemma). Applying this result to Theorem 4.6 yields the bound.

Lemma 4.12 (Stability of Algorithm 2) . Under Assumption 4.1, for a finite function class F , Algorithm 2 with step size η = 1 / 18 satisfies Assumption 4.4 with β = 2log |F| .

Proof. Using the form of ˆ f 1 , . . . , ˆ f T , the outputs of Algorithm 2, for all t ∈ [ T ] it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(2) Stability:

where we used the fact that ∥ f ∥ ∞ ≤ 1 for all f ∈ F and Pinsker's inequality. Taking a square, summing over t and taking expectations we get,

<!-- formula-not-decoded -->

where we have used the second result of Corollary 4.13.

We can now immediately obtain the following expected regret bound for Algorithm 1 when used with Algorithm 2 as a square-loss oracle.

Corollary 4.13. Let F denote a realizable loss function class, where |F| &lt; ∞ . Suppose we use Algorithm 2 as an oracle implementation for online least-square regression with η = 1 / 18 and choose γ = √ KT 36 log |F| for Algorithm 1. Then,

<!-- formula-not-decoded -->

Lower bound. In Corollary C.5 of Appendix C.2 we state and prove the first lower bound of

<!-- formula-not-decoded -->

for CMAB under delayed feedback assuming realizable function approximation. The lower bound implies that our result in Corollary 4.13 is far by a √ d max factor from the optimal regret bound.

## 5 Conclusions and Discussion

In this paper we presented regret minimization algorithms for adversarial CMAB with delayed feedback, where both the contexts and delays are chosen by a possibly adaptive adversary. We considered the problem under the two mainstream frameworks for adversarial CMAB learning: online function approximation and policy class learning.

For the policy class learning setup, we presented Algorithm 3 and proved that it obtains an expected regret bound that is optimal up to logarithmic factors.

For online function approximation, we presented Algorithm 1 and analyzed its regret under a stability assumption related to the online regression oracle in use, which affects the delay-dependent term of our bound. Additionally, we analyzed the expected regret of a version of Vovk's aggregating forecaster and shown it satisfies the required stability guarantees, allowing us to obtain a nontrivial expected regret bound of O ( √ KT log( |F| ) + √ d max D log |F| ) , which is optimal up to √ d max .

Our work leaves some open questions that we believe are very interesting for future research. One possible direction is to remove the √ d max factor from the delay-dependent term in our bound, which is presumably sub-optimal. Furthermore, as our lower bound in Theorem 4.3 shows, without any assumption on the oracle in use other than Assumption 4.2, linear regret is unavoidable. We therefore find it interesting to investigate different or weaker assumptions on the oracle that enable non-trivial regret guarantees.

## Acknowledgements

We would like to thank the reviewers for their helpful comments.

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 882396 and grant agreement No. 101078075). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. This work received additional support from the Israel Science Foundation (ISF, grant numbers 993/17 and 2549/19), Tel Aviv University Center for AI and Data Science (TAD), the Yandex Initiative for Machine Learning at Tel Aviv University, the Len Blavatnik and the Blavatnik Family Foundation.

AC is supported by the Israeli Science Foundation (ISF) grant no. 2250/22.

## References

- [1] Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- [2] N. Abe and P. M. Long. Associative reinforcement learning using linear probabilistic concepts. In ICML , pages 3-11. Citeseer, 1999.
- [3] A. Agarwal, M. Dudík, S. Kale, J. Langford, and R. Schapire. Contextual bandit learning with predictable rewards. In Artificial Intelligence and Statistics , pages 19-26. PMLR, 2012.
- [4] A. Agarwal, M. Dudik, S. Kale, J. Langford, and R. Schapire. Contextual bandit learning with predictable rewards. In N. D. Lawrence and M. Girolami, editors, Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics , volume 22 of Proceedings of Machine Learning Research , pages 19-26, La Palma, Canary Islands, 21-23 Apr 2012. PMLR.
- [5] A. Agarwal, D. Hsu, S. Kale, J. Langford, L. Li, and R. Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In E. P. Xing and T. Jebara, editors, Proceedings of the 31st International Conference on Machine Learning , volume 32, pages 1638-1646, 2014.
- [6] P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The nonstochastic multiarmed bandit problem. SIAM journal on computing , 32(1):48-77, 2002.
- [7] A. Beygelzimer, J. Langford, L. Li, L. Reyzin, and R. E. Schapire. Contextual bandit algorithms with supervised learning guarantees. In G. J. Gordon, D. B. Dunson, and M. Dudík, editors, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, AISTATS 2011, Fort Lauderdale, USA, April 11-13, 2011 , volume 15 of JMLR Proceedings , pages 19-26. JMLR.org, 2011. URL http://proceedings.mlr.press/v15/ beygelzimer11a/beygelzimer11a.pdf .
- [8] I. Bistritz, Z. Zhou, X. Chen, N. Bambos, and J. Blanchet. Online exp3 learning in adversarial bandits with delayed feedback. Advances in neural information processing systems , 32, 2019.
- [9] N. Cesa-Bianchi and G. Lugosi. Prediction, learning, and games . Cambridge university press, 2006.
- [10] N. Cesa-Bianchi, C. Gentile, Y. Mansour, and A. Minora. Delay and cooperation in nonstochastic bandits. In Conference on Learning Theory , pages 605-622. PMLR, 2016.
- [11] W. Chu, L. Li, L. Reyzin, and R. Schapire. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics , pages 208-214. JMLR Workshop and Conference Proceedings, 2011.
- [12] M. Dudik, D. Hsu, S. Kale, N. Karampatziakis, J. Langford, L. Reyzin, and T. Zhang. Efficient optimal learning for contextual bandits. arXiv preprint arXiv:1106.2369 , 2011.
- [13] D. Foster and A. Rakhlin. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. In International Conference on Machine Learning , pages 3199-3210. PMLR, 2020.
- [14] D. J. Foster and A. Krishnamurthy. Efficient first-order contextual bandits: Prediction, allocation, and triangular discrimination. Advances in Neural Information Processing Systems , 34:1890718919, 2021.
- [15] D. J. Foster, C. Gentile, M. Mohri, and J. Zimmert. Adapting to misspecification in contextual bandits. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 612, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 84c230a5b1bc3495046ef916957c7238-Abstract.html .
- [16] D. J. Foster, S. M. Kakade, J. Qian, and A. Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487 , 2021.
- [17] M. A. Gael, C. Vernade, A. Carpentier, and M. Valko. Stochastic bandits with arm-dependent delays. In International Conference on Machine Learning , pages 3348-3356. PMLR, 2020.

- [18] B. Howson, C. Pike-Burke, and S. Filippi. Delayed feedback in generalised linear bandits revisited. In International Conference on Artificial Intelligence and Statistics , pages 6095-6119. PMLR, 2023.
- [19] S. Ito, D. Hatano, H. Sumita, K. Takemura, T. Fukunaga, N. Kakimura, and K.-I. Kawarabayashi. Delay and cooperation in nonstochastic linear bandits. Advances in Neural Information Processing Systems , 33:4872-4883, 2020.
- [20] T. Jin, T. Lancewicki, H. Luo, Y . Mansour, and A. Rosenberg. Near-optimal regret for adversarial mdp with delayed bandit feedback. Advances in Neural Information Processing Systems , 35: 33469-33481, 2022.
- [21] T. Lancewicki, S. Segal, T. Koren, and Y. Mansour. Stochastic multi-armed bandits with unrestricted delay distributions. In International Conference on Machine Learning , pages 5969-5978. PMLR, 2021.
- [22] T. Lancewicki, A. Rosenberg, and Y. Mansour. Learning adversarial markov decision processes with delayed feedback. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 7281-7289, 2022.
- [23] J. Langford and T. Zhang. The epoch-greedy algorithm for multi-armed bandits with side information. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems , volume 20. Curran Associates, Inc., 2007. URL https://proceedings.neurips.cc/paper\_files/paper/2007/file/ 4b04a686b0ad13dce35fa99fa4161c65-Paper.pdf .
- [24] T. Lattimore and C. Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- [25] O. Levy and Y. Mansour. Optimism in face of a context: Regret guarantees for stochastic contextual mdp. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 8510-8517, 2023.
- [26] O. Levy, A. Cohen, A. B. Cassel, and Y. Mansour. Efficient rate optimal regret for adversarial contextual mdps using online function approximation. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 19287-19314. PMLR, 2023.
- [27] O. Levy, A. B. Cassel, A. Cohen, and Y. Mansour. Eluder-based regret for stochastic contextual mdps. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 . OpenReview.net, 2024.
- [28] B. Li, T. Chen, and G. B. Giannakis. Bandit online learning with unknown delays. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 993-1002. PMLR, 2019.
- [29] H. B. McMahan and M. J. Streeter. Tighter bounds for multi-armed bandits with expert advice. In COLT 2009 - The 22nd Conference on Learning Theory, Montreal, Quebec, Canada, June 18-21, 2009 , 2009. URL http://www.cs.mcgill.ca/%7Ecolt2009/papers/023.pdf#page=1 .
- [30] A. Modi and A. Tewari. No-regret exploration in contextual reinforcement learning. In R. P. Adams and V. Gogate, editors, Proceedings of the Thirty-Sixth Conference on Uncertainty in Artificial Intelligence, UAI 2020, virtual online, August 3-6, 2020 , volume 124 of Proceedings of Machine Learning Research , pages 829-838. AUAI Press, 2020. URL http://proceedings. mlr.press/v124/modi20a.html .
- [31] A. Modi, N. Jiang, S. Singh, and A. Tewari. Markov decision processes with continuous side information. In F. Janoos, M. Mohri, and K. Sridharan, editors, Algorithmic Learning Theory, ALT 2018, 7-9 April 2018, Lanzarote, Canary Islands, Spain , volume 83 of Proceedings of Machine Learning Research , pages 597-618. PMLR, 2018. URL http://proceedings.mlr. press/v83/modi18a.html .
- [32] J. Qian, H. Hu, and D. Simchi-Levi. Offline oracle-efficient learning for contextual mdps via layerwise exploration-exploitation tradeoff. arXiv preprint arXiv:2405.17796 , 2024.

- [33] D. Simchi-Levi and Y. Xu. Bypassing the monster: A faster and simpler optimal algorithm for contextual bandits under realizability. Mathematics of Operations Research , 47(3):1904-1931, 2022.
- [34] A. Slivkins et al. Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning , 12(1-2):1-286, 2019.
- [35] T. S. Thune, N. Cesa-Bianchi, and Y. Seldin. Nonstochastic multiarmed bandits with unrestricted delays. Advances in Neural Information Processing Systems , 32, 2019.
- [36] D. van der Hoeven, L. Zierahn, T. Lancewicki, A. Rosenberg, and N. Cesa-Bianchi. A unified analysis of nonstochastic delayed feedback for combinatorial semi-bandits, linear bandits, and mdps. In The Thirty Sixth Annual Conference on Learning Theory , pages 1285-1321. PMLR, 2023.
- [37] C. Vernade, O. Cappé, and V. Perchet. Stochastic bandit models for delayed conversions. arXiv preprint arXiv:1706.09186 , 2017.
- [38] C. Vernade, A. Carpentier, T. Lattimore, G. Zappella, B. Ermis, and M. Brueckner. Linear bandits with stochastic delayed feedback. In International Conference on Machine Learning , pages 9712-9721. PMLR, 2020.
- [39] V. G. Vovk. Aggregating strategies. In Proceedings of the third annual workshop on Computational learning theory , pages 371-386, 1990.
- [40] Y. Xu and A. Zeevi. Upper counterfactual confidence bounds: a new optimism principle for contextual bandits. arXiv preprint arXiv:2007.07876 , 2020.
- [41] Z. Zhou, R. Xu, and J. Blanchet. Learning in generalized linear contextual bandits with stochastic delays. Advances in Neural Information Processing Systems , 32, 2019.
- [42] Y. Zhu, D. J. Foster, J. Langford, and P. Mineiro. Contextual bandits with large action spaces: Made practical. In International Conference on Machine Learning , pages 27428-27453. PMLR, 2022.
- [43] J. Zimmert and Y. Seldin. An optimal algorithm for adversarial bandits with arbitrary delays. In International Conference on Artificial Intelligence and Statistics , pages 3285-3294. PMLR, 2020.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We prove each claim in detail, on the main paper or the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of our results in Section 5.

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

Justification: All the assumptions are stated directly in the section of the relevant results. We provide full proof to each of our results in the main paper or in the appendix.

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

Justification: Our results are theoretical, and we have no experiments.

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

Justification: Our results are theoretical and we have no experiments.

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

Justification: Our results are theoretical and we have no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our results are theoretical, and we have no experiments.

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

Justification: Our results are theoretical, and we have no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our results are theoretical, and we have no experiments. We thus believe that our results respects the code of ethics of NeurIPS.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our results are theoretical, and we have no experiments. We thus believe that our results have no social impact, nor negative or positive.

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

Justification: Our results are theoretical, and we have no experiments.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Our results are theoretical, and we have no experiments.

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

Justification: Our results are theoretical, and we have no experiments.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our results are theoretical, and we have no experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our results are theoretical, and we have no experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our results are theoretical, and we have no experiments. We did not use LLMs beyond editing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Proofs for Section 3

In this section, we analyze the regret of Algorithm 3 in the policy class setting and prove Theorem 3.1.

## Algorithm 3 EXP4 with Delay-Adapted Loss Estimators (EXP4-DALE)

- 1: inputs:
- Finite policy class Π ⊆ X → A with | Π | = N ,
- Upper bound on the sum of delays, D .
- Step size η &gt; 0 .
- 2: Initialize p 1 ∈ ∆ N as the uniform distribution over Π .
- 3: for round t = 1 , . . . , T do
- 4: Receive context x t ∈ X .
- 5: Sample π ∼ p t and play a t = π ( x t ) .
- 6: Observe feedback ( s, L ( x s , a s )) for all s ≤ t with s + d s = t and construct loss estimators

<!-- formula-not-decoded -->

where we define Q s,a = ∑ N i =1 p s,i I [ π i ( x s ) = a ] and ˜ Q t s,a = ∑ N i =1 p t,i I [ π i ( x s ) = a ] . 7: Update

<!-- formula-not-decoded -->

Throughout this section, we use the notation E t [ · ] to denote an expectation conditioned on the entire history up to round t . We define the standard (unbiased) importance-weighted loss estimators by

<!-- formula-not-decoded -->

Theorem A.1. Algorithm 3 attains the following expected regret bound:

<!-- formula-not-decoded -->

Proof. The regret may be decomposed as follows:

<!-- formula-not-decoded -->

where c t,i = L ( x t , π i ( x t )) for i ∈ [ N ] . The OMD term can be bounded by referring to Lemma 9 of [35] which asserts that

<!-- formula-not-decoded -->

while noting that this lemma does not require a specific form of loss estimators, only that they are nonnegative, as is the case for our delay-adapted estimators defined in Equation (2). We also note that the Bias 2 term is non-positive in expectation, since the delay-adapted estimators satisfy E t [ˆ c t,i ] ≤ c t,i for i ∈ [ N ] . Thus, to conclude the proof we are left with bounding the Drift and Bias 1 terms, whose bounds are given in Lemma A.2 and Lemma A.3 that follow.

Proof of Theorem 3.1. First, we show that

<!-- formula-not-decoded -->

Indeed, using the definition of the delay-adapted loss estimators ˆ c t , it holds that

<!-- formula-not-decoded -->

Thus, using Theorem A.1 together with Lemma A.4 gives the bound claimed in Theorem 3.1.

Lemma A.2 (Bounding the Drift term) . The Drift term given in Equation (5) is bounded in expectation as follows:

<!-- formula-not-decoded -->

Proof. First, we note that the delay-adapted loss estimators ˆ c t are upper-bounded by the standard, conditionally unbiased importance-weighted estimators ˜ c t defined in Equation (4). Therefore, we can bound the Drift term as follows:

<!-- formula-not-decoded -->

where the last step follows from Hölder's inequality and the fact that ∥ c t ∥ ∞ ≤ 1 .

Lemma A.3. The Bias 1 term given in Equation (5) is bounded in expectation as follows:

<!-- formula-not-decoded -->

Proof. We note losses and loss estimators can be indexed by actions rather than policies and use the the notation c t,a = L ( x t , a ) and ˆ c t,a = c t,a I [ a t = a ] M t,a where M t,a = max { Q t,a , ˜ Q t + d t t,a } . Therefore, using the fact that E t [ˆ c t,a ] = c t,a Q t,a M t,a , the Bias 1 term can be bounded as follows:

Proof. Define

<!-- formula-not-decoded -->

Now, by the definition of Q t,a , ˜ Q t + d t t,a and the triangle inequality, we have

<!-- formula-not-decoded -->

concluding the proof.

Lemma A.4 (Distribution drift) . The following holds for the iterates { p t } T t =1 of Algorithm 3:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where R ( p ) = ∑ N i =1 p i log p i , so that p t = arg min p ∈ ∆ Π F t ( p ) . Note that R ( · ) is 1 -strongly convex with respect to ∥·∥ 1 , and therefore F t ( · ) are 1 /η -strongly convex. Thus, using first-order optimality conditions for p t and p t +1 , we have:

<!-- formula-not-decoded -->

Summing the two inequalities, we obtain

<!-- formula-not-decoded -->

where ˜ c s,i are the standard (unbiased) importance-weighted loss estimators. Taking expectations while using E [( · ) 2 ] ≥ ( E [ · ]) 2 and Hölder's inequality, we obtain

<!-- formula-not-decoded -->

where m t = |{ s : s + d s = t }| is the number of observations that arrive on round t . Dividing through by the right-hand side of the inequality above, we obtain

<!-- formula-not-decoded -->

and using the triangle inequality we have

<!-- formula-not-decoded -->

where M t,d t is the number of observations that arrive between rounds t and t + d t -1 . Using Lemma C.7 in [20], we conclude the proof via

<!-- formula-not-decoded -->

## B Proofs for Section 4

## B.1 Proofs for Subsection 4.2

In this subsection, we provide the proofs of the lemmas required to derive regret guarantees for algorithm DA-FA (Algorithm 1), proving Theorem 4.6.

Consider the following regret decomposition,

<!-- formula-not-decoded -->

We bound each term individually in the following lemmas and claims, and then we combine all the bounds to derive Theorem 4.6.

Claim B.1. With probability 1 , it holds that

<!-- formula-not-decoded -->

Proof. This follows immediately by the fact that ℓ ( · ) is bounded in [0 , 1] .

Lemma B.2 (Restatement of Lemma 4.7) . With probability 1 , it holds that

<!-- formula-not-decoded -->

Proof. For t ∈ { d max +1 , d max +2 , . . . , T } , let R t ( p ) denote the objective of the convex minimization problem in Equation (1), i.e,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since p ⋆ ( ·| x t ) is a feasible solution and p t is the optimal solution, by first-order optimality conditions we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Which implies that

We conclude that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Lemma B.3 (Restatement of Lemma 4.8) . It holds true that

<!-- formula-not-decoded -->

Proof. For this term, we apply the oracle expected regret bound for the non-delayed function approximation. By Assumption 4.2 the following holds.

<!-- formula-not-decoded -->

where in the final transition we used Assumption 4.5 which implies that the observations are given to the oracle in the same order that they arrive to the CMAB algorithm, which allows us to invoke the regret guarantee of the non-delayed oracle.

Lemma B.4 (Restatement of Lemma 4.9) . It holds true that

<!-- formula-not-decoded -->

Proof. For this term, we would like to use a change-of-measure technique using AM-GM to be able to apply the oracle's expected regret bound for the non-delayed function approximation. Again, by Assumption 4.2 the following holds.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the final transition we used Assumption 4.5 as in Lemma 4.8.

We now proceed to prove our final lemma.

Lemma B.5 (Restatement of Lemma 4.10) . Under Assumption 4.4 it holds true that

<!-- formula-not-decoded -->

Proof. Using Hölder's inequality and the triangle inequality, we have

<!-- formula-not-decoded -->

where σ t is the number of pending observations (that is, which have not yet arrived) as of round t . Now, using the Cauchy-Schwarz inequality, the fact that E [ √ · ] ≤ √ E [ · ] and Assumption 4.4, we finally obtain

<!-- formula-not-decoded -->

where we used the fact that σ t ≤ d max for all t (since at every round σ t can increase at most by one and an observation can remain pending for at most d max rounds), and the fact that ∑ t σ t = D , which follows since when summing the delays, each delay d t contributes once to exactly d t rounds with pending observations, and all pending observations are covered in this manner.

We can now prove Theorem 4.6.

Theorem B.6 (Restatement of Theorem 4.6) . Let γ = √ KT R T ( O F ,η sq ) . Then the following expected regret bound holds for Algorithm 1.

<!-- formula-not-decoded -->

Proof of Theorem 4.6. Putting the results of Claim B.1 (taking expectation on both sides) and Lemmas 4.7 to 4.10 all together, the expected regret is bounded as follows.

<!-- formula-not-decoded -->

Choosing γ = √ KT R T ( O F ,η sq )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.2 Regret and Stability analysis of Vovk's aggregating forecaster for the square-loss

We consider a hedge-based version of Vovk's aggregating forecaster [39], presented in Algorithm 2 for the square loss under the realizability assumption (Assumption 4.1) and a finite function class F .

We denote by z t = ( x t , a t ) ∈ X × A the input of each function f ∈ F ⊆ X × A → [0 , 1] at time step t ∈ [ T ] , where x 1 , . . . , x T ∈ X is a sequence of contexts generated throughout, and a 1 , . . . , a T ∈ A is the sequence of actions, where a i was chosen for the context x i , for all i ∈ [ T ] . Also, let y 1 , . . . , y T ∈ [0 , 1] are such that E [ y t | z t ] = f ⋆ ( z t ) , and f ( z t ) ∈ [0 , 1] for all f ∈ F . We consider the square loss, and prove the following guarantee for the iterates of Algorithm 2.

Theorem B.7 (Restatement of Theorem 4.11) . For t ∈ [ T ] denote by q t ∈ ∆( F ) the probability measure over functions in F computed by Algorithm 2 at time step t , for the t -1 -length prefix of the sequence { ( z τ , y τ ) } T τ =1 .

Then, the sequence of measures { q t } T t =1 satisfies the followings for any η ≤ 1 / 18 :

1. Expected regret:
2. Stability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. WLOG, since Hedge is invariant under adding a constant loss in each round we can subtract ( f ⋆ ( z t ) -y t ) 2 from the loss of all functions. In particular, after the subtraction, f ⋆ has a cumulative loss of 0 . Therefore q t ( f ) ∝ w t ( f ) , and w t +1 ( f ) = w t ( f ) e -η (( f ( z t ) -y t ) 2 -( f ⋆ ( z t ) -y t ) 2 ) . Denote W t = ∑ f w t ( f ) .

We have, as W 1 = |F| and W T +1 ≥ 1 (since w t ( f ⋆ ) = w 1 ( f ⋆ ) = 1 for all t ), that

<!-- formula-not-decoded -->

On the other hand, for small enough η (smaller than a constant),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rearranging, we obtain that

<!-- formula-not-decoded -->

Taking expectation over y 1 , . . . , y T :

<!-- formula-not-decoded -->

If, suppose η ≤ 1 / 18 then we immediately obtain

<!-- formula-not-decoded -->

and the expected regret of Algorithm 2 can now be bounded by

<!-- formula-not-decoded -->

where we have used Jensen's inequality. This concludes the proof of part 1 . of the theorem. For the second part, we observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, taking expectation over y 1 , . . . , y T and using part 1. we obtain

<!-- formula-not-decoded -->

yields the second part of the theorem.

## C Lower Bounds

## C.1 Proof of Theorem 4.3

In this subsection, we present a lower bound indicating that an additional assumption on the oracle is necessary in order to obtain sub-linear regret in the general function approximation setting. The lower bound shows that with no additional assumption on the least squares oracle, any algorithm incurs linear regret in the presence of delayed feedback, even for a constant delay of d = 1 .

Theorem C.1 (Restatement of Theorem 4.3) . For any CMAB algorithm ALG in the function approximation setting (that is, ALG can only access F via the oracle) there exists a CMAB instance with constant delay d = 1 over a realizable loss function class F with |F| = T +1 with an online oracle O F sq satisfying R T ( O F sq ) = 0 , on which ALG attains regret R T = Ω( T ) .

Proof. Consider a CMAB instance over X = { x 1 , x 2 , . . . , x T } , A = { a 1 , a 2 } and F = { f 1 , f 2 , . . . , f T , f ⋆ } , where f ⋆ is the true loss function. The functions in F are sampled randomly as follows:

<!-- formula-not-decoded -->

The online sequence of contexts is defined by x 1 , x 2 , x 3 , . . . , x T in order. We consider an oracle which at round t outputs the function ˆ f t = f t . It is easy to see that the least-squares regret of this oracle is zero because f t ( x t , · ) = f ⋆ ( x t , · ) . Now, at round t , due to the delay, ALG only has relevant information on x t given by { f 1 ( x t , · ) , . . . , f t -1 ( x t , · ) } , all of which are random i.i.d. Ber ( 1 2 ) random variables, with the true loss f ⋆ ( x t , · ) being either (1 , 0) or (0 , 1) with equal probability and independently of the previous observations of ALG . Therefore, however ALG chooses the next action a ( t ) , with probability 1 2 it will incur a loss of 1 while simultaneously the other action will have a loss of zero. This means that in expectation over the random construction of F , the algorithm will incur Ω ( T 2 ) regret. By the probabilistic method, we know that there exists a fixture of F depending on ALG on which ALG suffers linear regret, as claimed.

## C.2 Lower Bounds for Contextual MAB with Delayed Feedback

In this subsection, we establish lower bounds on the expected regret for CMAB with delayed feedback. Our construction is based on the approach of [10] via a reduction from the full information variant with non-delayed feedback using a blocking argument.

We begin with a lower bound for the policy class setting with a finite policy class Π , which relies on a reduction from the problem of (agnostic) prediction with expert advice, for which known lower bounds exist in the literature (see e.g. [9]).

We then present a lower bound for the realizable function approximation setting with a finite loss function class F , and for that we construct an explicit hard instance for the full-information nondelayed variant, with a regret lower bound of Ω ( √ T log |F| ) .

We remark that while a regret lower bound of Ω ( √ KT + √ D ) can be immediately inferred from the results of [9] who consider the special case of multi-armed bandits, our goal is to show that the

dependence on log |F| where F appears jointly with the delay dependence. While the dependence of √ KT log |F| is known to be tight for CMAB with general function approximation, it is nontrivial that the delay dependent term also contains a dependence on log |F| , which we prove in the construction that follows.

In our construction we consider the full feedback setting, where for each round t and observed context x t ∈ X , the learner observes the entire loss vector ( ℓ ( x t , a )) a ∈A after choosing an action a t ∈ A .

Theorem C.2. There exists a finite policy class Π ⊆ A X mapping contexts to actions, a delay sequence ( d 1 , . . . , d T ) with maximal delay d and sum of delays D = Θ( dT ) such that for any CMAB algorithm there is instance of the CMAB problem for which the algorithm incurs expected regret

<!-- formula-not-decoded -->

Proof. We observe that CMAB with a policy class can be viewed as a special case of the prediction with expert advice framework [9], where each policy corresponds to an expert, provides a prediction for each context. Hence, the classical lower bound of Ω( √ T log | Π | ) for the full-information expert setting (see Cesa-Bianchi and Lugosi [9], chapter 2) applies in the absence of delays.

Returning to the delayed CMAB problem, construct a delay sequence ( d 1 , . . . , d T ) in which d is the maximal delay and D = ∑ T t =1 d t = Θ( dT ) as follows:

Divide the time horizon into T/ ( d +1) blocks, each containing d +1 consecutive rounds. For each block b ∈ { 0 , 1 , . . . , T / ( d +1) -1 } and each round τ ∈ { b ( d +1) , b ( d +1)+1 , . . . , ( b +1)( d +1) -1 } , define the delay as d τ = d -( τ -b ( d +1)) . That is, within each block, the delays decrease from d to 0 , in this corresponding order. This also implies that D = T d +1 ∑ d i =0 i = T d +1 · ( d +1)( d +0) 2 = Td 2 . This construction ensures that feedback from all rounds within a block is revealed simultaneously at the end of the block.

The loss sequence is constructed as follows: Consider the loss sequence ( ℓ 1 , . . . , ℓ T/ ( d +1) ) given by a lower bound construction for prediction with expert advice over T/ ( d +1) rounds. The loss of the first round of each block b is defined to be ℓ b , and remains the same throughout the block. Now, note that given this construction, the algorithm essentially faces a prediction with expert advice problem over T/ ( d +1) rounds (the rounds on which information is obtained), with loss values in the range [0 , d +1] . We remark that we can assume without loss of generality that the algorithm fixes a policy π b at the start of block b and uses it to play actions throughout the entire block, as it does not learn new information within the block.

Thus, we can aggregate each block into a single 'super-round' of a reduced expert problem. Specifically, for block b , define the aggregate loss of each expert π as ℓ b ( π ) = ∑ ( b +1)( d +1) -1 τ = b ( d +1) ℓ ( x τ , π ( x τ )) . Even if we allow the algorithm to observe full feedback, it essentially observes the full aggregated loss vector over actions in each block, so this construction corresponds to a well-defined instance of prediction with expert advice over the T/ ( d +1) rounds which are the initial rounds of the blocks.

The resulting reduced problem has T/ ( d +1) rounds with losses in [0 , d +1] . Applying the lower bound from Cesa-Bianchi and Lugosi [9] to this reduced problem yields:

<!-- formula-not-decoded -->

which completes the proof.

We now combine this result with the classical lower bound of Ω( √ KT log | Π | ) for CMAB with bandit feedback and a finite policy class, which is based on reductions from prediction with expert advice (see, e.g., [29, 6, 7, 9]). This yields the following lower bound for CMAB with delayed feedback in the policy class setting:

Corollary C.3. For CMAB with delayed bandit feedback and a finite policy class Π , the expected regret satisfies

<!-- formula-not-decoded -->

To prove a corresponding regret lower bound for the realizable function approximation setting, we similarly require a regret lower bound of Ω ( √ T log |F| ) for the full-information non-delayed variant of the problem. Such a lower bound, however, does not exist in the literature as far as we are aware, so we exhibit an explicit construction in the following lemma.

Lemma C.4. Let A = { a 1 , a 2 } be action set, and let X = { x 1 , . . . , x n } be a set of n contexts where n ≤ T . Then for any CMAB algorithm there exists a finite loss function class F ⊆ {X ×A → [0 , 1] } of size |F| = 2 n and a CMAB instance which is realizable with respect to F , on which the expected regret of the CMAB algorithm is lower bounded by

<!-- formula-not-decoded -->

̸

Proof. Across all of the instances which we construct, the context is chosen uniformly at random from X . We define the function class F as the set of 2 n functions f which, for each x ∈ X , are defined via f ( x, a i ) = 1 2 -ε and f ( x, a j ) = 1 2 for the other action a j = a i (that is, each function in F has a distinct choice of optimal actions across all n contexts), and we choose ε = √ n/ 100 T .

Prior to the interaction, a function f ⋆ ∈ F is selected uniformly at random and the losses are defined to be Bernoulli random variables according to f ⋆ , ensuring realizability holds. More specifically, ℓ ( x, a ) will be a Bernoulli random variable with parameter f ⋆ ( x, a ) for all x ∈ X , a ∈ A .

Now, by standard arguments of statistical estimation, since the true loss function f ⋆ was sampled at random, as long as a given context x ∈ X has not appeared more than Ω(1 /ε 2 ) times, the CMAB algorithm must incur instantaneous regret of ε conditioned on this context. Since the contexts are sampled uniformly at random and the loss values for one context reveal no information about the loss for different contexts, the algorithm must incur expected regret of at least Ω( εt ) on the first t rounds as long as each context has been sampled o (1 /ε 2 ) times. With high probability, all contexts are sampled sufficiently many times only after t = Ω ( n/ε 2 ) rounds, implying that the expected regret of the algorithm over T rounds is lower bounded by

<!-- formula-not-decoded -->

which concludes the proof.

We remark that the construction in the above proof is similar to the lower bound given by [3] for the bandit case, but here the proof is considerably simpler as the algorithm is not required to perform exploration in order to obtain sufficient feedback.

Thus, by combining the lower bound for contextual bandits with function approximation under bandit feedback [3] with the delayed feedback result above using the same reduction as we described in the proof of Theorem C.2, we obtain:

Corollary C.5. For any CMAB algorithm in the realizable function approximation setting over finite loss function classes, there exists a finite function class F and a distribution over losses which is realizable by F , for which the expected regret satisfies

<!-- formula-not-decoded -->
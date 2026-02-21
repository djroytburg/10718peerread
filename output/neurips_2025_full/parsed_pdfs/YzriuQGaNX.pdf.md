## REINFORCE Converges to Optimal Policies with Any Learning Rate

Samuel Robertson 1 ∗

1 University of Alberta

∗ 1 2

Bo Dai Dale Schuurmans Jincheng Mei 2

Thang D. Chu 1

Csaba Szepesvári

2 Google DeepMind

2 3

3 Georgia Institute of Technology

{smrobert,thang}@ualberta.ca

{bodai,schuurmans,szepi,jcmei}@google.com

## Abstract

We prove that the classic REINFORCE stochastic policy gradient (SPG) method converges to globally optimal policies in finite-horizon Markov Decision Processes (MDPs) with any constant learning rate. To avoid the need for small or decaying learning rates, we introduce two key innovations in the stochastic bandit setting, which we then extend to MDPs. First , we identify a new exploration property: the online SPG method samples every action infinitely often, improving on previous results that only guaranteed at least two actions would be sampled infinitely often. This means SPG inherently achieves asymptotic exploration without modification. Second , we eliminate the assumption of unique mean reward values, a condition that previous convergence analyses in the bandit setting relied on, but that does not translate to MDPs. Our results deepen the theoretical understanding of SPG in both bandit problems and MDPs, with a focus on how it handles the explorationexploitation trade-off when standard analysis techniques for optimization and stochastic approximation methods cannot be applied, as is the case with large constant learning rates.

## 1 Introduction

Policy gradient (PG) methods constitute one of the most popular classes of algorithms for reinforcement learning (RL). In the PG paradigm, a learner acts according to a parameterized policy; the expected return is directly optimized by computing its gradient with respect to the policy parameters and performing stochastic gradient ascent. PG methods have played a key role in the advancements of deep RL (Lillicrap et al., 2019; Schulman et al., 2017a,b): combined with deep neural networks, PG algorithms have shown strong empirical performance across many domains, including robotics Akkaya et al. (2019), games Vinyals et al. (2019), and large language model training (Rafailov et al., 2024; Ouyang et al., 2022).

Despite PG methods' conceptual simplicity and rich set of practical applications, known theoretical guarantees on their performance come with restrictive assumptions. In particular, convergence proofs either require oracle access to the exact gradient (Liu et al., 2024; Agarwal et al., 2020), which is akin to demanding that the reward function and dynamics of the environment are known to the learner, or they impose harsh constraints on the learning rate used for stochastic gradient ascent (Mei et al., 2024b; Klein et al., 2024). Both of these assumptions are violated in typical applications. In the stochastic setting, where the rewards and transition probabilities are unknown and must be estimated from interaction with the environment, convergence of the classic REINFORCE algorithm (Williams,

* Equal contributions

1 2

1992) has only been shown under the assumption that the learning rate is either sufficiently small (Klein et al., 2024) or decaying (Zhang et al., 2020a). In this work we study REINFORCE with the standard softmax parameterization, and narrow the gap between theory and practice by providing the first proof in the stochastic setting that REINFORCE will globally converge to an optimal policy in tabular finite-horizon Markov Decision Processes (MDPs) with any constant learning rate. Along the way we show new results about the stochastic gradient bandit algorithm (Sutton and Barto, 2018; Mei et al., 2024a), which is the special case induced by applying REINFORCE to a bandit problem. Specifically, we show that the stochastic gradient bandit algorithm automatically achieves sufficient exploration for global convergence with an arbitrary constant learning rate; in doing so we remove a key assumption in prior work, and thus resolve an open problem posed by Mei et al. (2024a). Our results in the bandit setting extend to a more general 'nonstationary bandit problem', where the reward function is allowed to drift mildly across timesteps. This extension is then embedded into the RL setting where, with some additional arguments, we derive the convergence of REINFORCE. In summary, the main contributions of this work are threefold:

- i) We show that the stochastic gradient bandit algorithm will select every arm infinitely often (i.o.) in any bandit problem and with any learning rate. We find it surprising that this strong property emerges from such a simple algorithm, without any explicit hacks to encourage exploration. We obtain a counterpart result in the RL setting, but the bandit case is independently interesting, and also critical for our second contribution:
2. ii) In the bandit setting we remove the central assumption of Mei et al. (2024a), that no two arms have the same expected reward, and prove that the stochastic gradient bandit algorithm still converges to an optimal policy. For bandits this assumption is impossible to verify without access to the true reward function (at which point the bandit problem is already solved), but more importantly it renders the extension to RL virtually impossible.
3. iii) In RL we provide the first proof that REINFORCE converges with large learning rates in the stochastic setting. This requires the first two contributions: the exploration result is applied directly to RL, and the bandit result is extended to a nonstationary bandit problem that can be embedded into an MDP.

Positioning our work, to our best knowledge, we note that existing convergence results for stochastic policy gradient (SPG) methods typically suffer from one of the following drawbacks: (i) they rely on decaying learning rate schedules for convergence guarantees (Zhang et al., 2020a; Ding et al., 2022, 2024; Mei et al., 2023), a requirement not aligned with the constant rates commonly used in practice; (ii) they study constant learning rates (Mei et al., 2024b; Klein et al., 2024), but provide guarantees only for rates considered impractically small; or (iii) they are restricted to the simplest bandit settings by assumptions on the reward structure (Mei et al., 2024a), preventing extensions to RL. Filling this gap, our work provides rigorous convergence guarantees for SPG using practical learning rates in MDPs, without requiring uniqueness of the optimal policy.

## 2 Related Work

In the exact gradient setting, where the rewards and transition probabilities of the MDP are known, Agarwal et al. (2020) exploited the notion of gradient dominance (Polyak, 1963) to show that PG algorithms with the tabular softmax parameterization can attain asymptotic convergence towards a globally optimal policy. Mei et al. (2020b) leveraged Łojasiewicz-like inequalities to show a O (1 /t ) convergence rate, where t is the number of update iterations. Moreover, Zhang et al. (2020b) exploited the 'hidden convexity' property of MDPs to establish a global optimality result. In summary, these works exploited a variety of conditions for global convergence results of vanilla PG algorithms in the exact gradient setting.

There have also been several techniques developed to improve the convergence rate of PG algorithms, including entropy regularization (Mei et al., 2020b), normalization (Mei et al., 2022), natural gradients (Cen et al., 2021; Lan, 2022; Khodadadian et al., 2021; Xiao, 2022), and accelerated learning rates (Chen et al., 2024). Some of these techniques have known convergence rates in the exact setting (Liu et al., 2024), but the results are unlikely to carry over to the stochastic setting. Indeed, all of these techniques except accelerated learning rates have been shown not to globally converge in the stochastic setting due to 'over-committal' behavior in the update rule (Mei et al., 2021), and

accelerated learning rates have no convergence guarantee; the noise in the gradient causes these methods to commit overzealously to suboptimal policies.

PG algorithms for the stochastic setting, or SPG algorithms, have also received theoretical attention. The common approach is to pair decaying or sufficiently small learning rates with SPG in order to control the noise in the gradient estimates. In particular, Zhang et al. (2020a) showed that REINFORCE Williams (1992) with a O (1 / √ t ) learning rate converges to an ϵ -optimal policy at a ˜ O ( ϵ -2 ) rate. Ding et al. (2024) demonstrated that SPG algorithms with softmax parameterizations, entropy regularization, and a O (1 /t ) learning rate enjoy a ˜ O ( ϵ -2 ) sample complexity. In addition to controlling the learning rates, variance reduction techniques have been developed to reduce the effect of noisy estimates (Masiha et al., 2023; Fatkhullin et al., 2023). The main drawbacks of these approaches are that they introduce decaying learning rates (Zhang et al., 2020a; Ding et al., 2024; Mei et al., 2023), incur expensive subroutines (Masiha et al., 2023), or assume prior knowledge of the true state-value function (Mei et al., 2023).

Recently, Mei et al. (2024b) analyzed softmax SPG in the bandit setting and proved that it converges to the optimal policy with a sufficiently small constant learning rate. However, their learning rate depends on prior knowledge of the reward gap between the optimal and next-best arm, which is typically unknown. Klein et al. (2024) proved global convergence for SPG algorithms with constant learning rates in finite-horizon MDPs, but their technique strongly requires the learning rate to be sufficiently small in order to induce a smoothness property of the objective function. The most relevant work to ours is that of Mei et al. (2024a), who recently showed that SPG algorithms with arbitrarily large learning rates converge to an optimal policy in the bandit setting. The major caveat is the assumption that there are no ties in the mean reward among arms-even among suboptimal ones. This strong assumption raises concerns about convergence in a bandit problem with any equally good arms (e.g. any bandit problem with its worst arm duplicated), and also hinders the extension to RL, since the analogous assumption would be that no state has ties in Q -values under any policy .

## 3 Challenges of Non-Unique Solutions

## 3.1 Non-Uniqueness of Policies in RL

In standard optimization, it is well known that gradient-based algorithms can exhibit non-convergence of their parameters (or iterates) when multiple optimal solutions exist (Absil et al., 2005). To avoid this challenge, existing results for the SPG algorithm in the K -armed bandit setting (Mei et al., 2024b,a) rely on the following assumption, which implies the uniqueness of the globally optimal policy.

̸

Assumption 3.1 (True mean reward has no ties) . For all a, b ∈ [ K ] , if a = b , then r ( a ) = r ( b ) .

̸

̸

In Assumption 3.1 [ K ] := { 1 , . . . , K } denotes the set of K arms and r ( a ) is the true mean reward for arm a ∈ [ K ] . Assumption 3.1 implies that there is a unique optimal arm, which we denote a ∗ := arg max a ∈ [ K ] r ( a ) . This results in a unique one-hot globally optimal policy π ∗ with π ∗ ( a ∗ ) = 1 and π ∗ ( a ) = 0 for all a = a ∗ .

Figure 1: Classical examples of finite-horizon MDPs. Transitions are labelled with (deterministic) rewards, and double circles indicate terminal states.

<!-- image -->

However, extending to the RL setting presents the challenge of multiple optimal policies, a scenario which is not prevented by the straightforward extension of Assumption 3.1 to each state, since

Assumption 3.1 only constrains immediate rewards. In contrast to bandits, RL involves sequential decisions, and different action sequences (trajectories) can yield the same (maximal) cumulative reward. This situation is common in tasks like navigation with alternative optimal paths; consider the tree-structured MDP shown in Fig. 1a (state space S = { s 1 , . . . , s 13 } , action space A = { a 1 , a 2 , a 3 } ). Here, both s 1 → s 2 → s 5 and s 1 → s 3 → s 8 are optimal paths with total reward 3 . Because previous bandit convergence analyses (Mei et al., 2024b,a) critically rely on the assumption of a unique optimal policy, they cannot be directly applied to RL problems exhibiting such non-uniqueness.

## 3.2 SPG Policy Non-Convergence in the Presence of Ties

On the other hand, Mei et al. (2024b, Remark 5.3) conjectured that the SPG algorithm could still achieve convergence even without Assumption 3.1. Their conjecture was based on the idea that SPG has a 'self-reinforcing' property, causing the probability of only one arm to eventually become dominant and converge to 1 , thus resulting in a stationary one-hot optimal policy as t →∞ . That is, π t ( a ∗ ) → 1 for only one optimal arm as t →∞ , almost surely, even when multiple optimal arms exist . If this behavioral property holds, the latter part of the convergence proof can utilize the contradiction-based arguments presented in (Mei et al., 2024b, Theorem 5.1, Claim 2).

Our first major finding, supported by both empirical evidence and theoretical analysis, is that the aforementioned conjecture is incorrect: SPG-like algorithms do not necessarily converge to a single policy in the presence of multiple solutions. To demonstrate this, we designed a bandit experiment with two optimal arms (mean 0 . 2 ) and one suboptimal arm (mean -0 . 1 ). Fig. 2a shows typical runs of the stochastic gradient bandit algorithm (Algorithm 1) with initial parameters θ 0 := (0 , 0 , 0) and learning rates η ∈ { 1 , 10 } for 10 5 iterations, revealing that, while the total probability of optimal arms converges to 1 ( ∑ a ∈A ∗ π t ( a ) → 1 ), the probabilities of individual optimal arms (1 and 2) display non-stationary behavior. We observed analogous behavior in a similar experiment on a tree-structured MDP using REINFORCE (Williams, 1992) with η ∈ { 0 . 1 , 0 . 5 } , as shown in Fig. 2b, where optimal action probabilities from state s 1 fail to converge to a unique action. Moreover, we prove the following theorem, which supports the phenomena observed in simulation.

Proposition 3.2 (Non-Stationary Convergence) . Consider a K -armed bandit with all arms being equally good, i.e. A ∗ = [ K ] , and at least one arm has a nonzero probability of generating a nonzero reward. Running Algorithm 1 with any η &gt; 0 leads to

<!-- formula-not-decoded -->

for all a ∈ [ K ] . Therefore ( π t ) t ≥ 0 does not converge to any one-hot policy.

Proof sketch 1 . We first argue that the sequence of parameters ( θ t ) t ≥ 0 does not converge to a fixed finite vector in R K . Since all arms are equally good, ( θ t ) t ≥ 0 is a martingale, and we can use the fact that it doesn't converge to a finite value and standard martingale results to show that it will undergo unbounded oscillations, implying the result.

<!-- image -->

(a) Bandit

<!-- image -->

(b) RL

Figure 2: Fig. 2a shows that the total probability of optimal arms converges to 1, but the probabilities of individual optimal arms are non-stationary (i.e. π t ( a 1 ) oscillates). Fig. 2b shows similar nonstationary behavior for optimal actions in an RL setting with multiple optimal trajectories.

## 3.3 Limitations of Standard Analysis

Global convergence for SPG algorithms is typically established through a two-stage proof: (i) establish convergence to a stationary point, and (ii) demonstrate (often by contradiction) that the

attained stationary point is globally optimal. This methodology originates from seminal work on PG in the exact gradient setting (Agarwal et al., 2020, Theorem 5.1).

Our findings demonstrate two key points: (i) ties in trajectory or policy value can exist in RL settings regardless of assumptions on immediate rewards; and (ii) the typical two-stage proof strategy for arguing global convergence of SPG cannot be directly extended to RL. However, as shown in the next section and suggested by the above simulations, convergence results can be obtained even with ties, but this requires new analysis. This is primarily because our approach needs to carefully reason about the per-timestep expected progress in distinguishing optimal from suboptimal actions despite the presence of these ties. Specifically, we prove that the learned policy eventually converges to assign all probability mass to the optimal set (a form of 'generalized one-hot policy'), i.e. ∑ a ∈A ∗ π t ( a ) → 1 as t →∞ .

## 4 An Illustrative Bandit Setting

This section presents new insights into the exploration properties of the SPG algorithm. We first analyze the simplest bandit setting for illustration and then extend the results to RL.

## 4.1 Stochastic Gradient Bandit

We consider a stochastic multi-armed bandit problem with K ≥ 2 arms and rewards bounded in [ -R,R ] (where R &gt; 0 ). At each iteration t ≥ 1 , the learner selects an arm a t ∈ [ K ] := { 1 , . . . , K } and observes a reward r t sampled from a fixed distribution P a t ∈ M 1 ([ -R,R ]) . 1 The true mean reward for arm a ∈ [ K ] is r ( a ) := ∫ R -R xP a ( dx ) . The set of optimal arms is denoted A ∗ := arg max a ∈ [ K ] r ( a ) .

The learner aims to find a policy π ∈ M 1 ([ K ]) that maximizes expected reward. We use the softmax parameterization over R K : for θ ∈ R K and a ∈ [ K ] ,

<!-- formula-not-decoded -->

The optimization problem the learner is solving thus has the objective

<!-- formula-not-decoded -->

We study the stochastic gradient bandit algorithm (Algorithm 1), which performs stochastic gradient ascent on Eq. (3) (Sutton and Barto, 2018; Mei et al., 2024b). Given θ 0 and learning rate η &gt; 0 the algorithm iteratively updates parameters using the information it receives from single interactions. The stream of parameters generated will be referred to as ( θ t ) t ≥ 0 , and we will use π t := π t for the policy used to select a t +1 .

.

| Algorithm 1 Stochastic gradient bandit algorithm   | Algorithm 1 Stochastic gradient bandit algorithm                             |
|----------------------------------------------------|------------------------------------------------------------------------------|
| 1:                                                 | input θ 0 ∈ R K , η > 0                                                      |
| 2:                                                 | for t ≥ 0 do                                                                 |
| 3:                                                 | Select a t +1 ∼ π t , and observe r t +1 ∼ P a t +1 .                        |
| 4:                                                 | θ t +1 ( a t +1 ) ← θ t ( a t +1 )+ η (1 - π t ( a t +1 )) r t +1            |
| 5:                                                 | for a ∈ [ K ] , a = a t +1 do θ t +1 ( a ) ← θ t ( a ) - ηπ t ( a ) r t +1 . |
| 6:                                                 |                                                                              |
| 7:                                                 | end for                                                                      |
| 8:                                                 | end for                                                                      |

## 4.2 A Novel Exploration Lemma

We detail the reason why existing results (Mei et al., 2024a) do not generalize, even to bandit settings with reward ties. Mei et al. (2024a, Lemma 2) establishes an exploration property for SPG, showing

1 Where M 1 ( S ) denotes the collection of probability distributions over the set S .

̸

that at least two distinct arms are sampled i.o. Their subsequent convergence proof (Mei et al., 2024a, Theorem 2) relies on the argument that at least one of these i.o. sampled actions must be optimal. However, in the presence of reward ties, it is possible for two actions to share the same reward value (the sub-optimal action's interval from (Mei et al., 2024a, Eq. (15)) no longer exists). Consequently, the arguments that construct a contradiction to show 'at least one of these i.o. sampled actions must be optimal' are no longer valid.

Given the failure of existing approaches with reward ties, new analytical results are required for convergence proofs, even in the bandit setting. Our second key finding is a generalized exploration property for SPG: we establish that despite reward ties, every arm is sampled i.o. To formalize this, we define N t ( a ) as the number of times action a ∈ [ K ] has been sampled up to iteration t ≥ 1 , i.e.

<!-- formula-not-decoded -->

The asymptotic count is N ∞ ( a ) := lim t →∞ N t ( a ) , which is either finite or infinite. If N ∞ ( a ) &lt; ∞ , action a is only sampled finitely many times; if N ∞ ( a ) = ∞ , action a is sampled i.o.

Lemma 4.1 (Bandit Exploration) . Using Algorithm 1 with any constant learning rate η ∈ Θ(1) , every arm is almost surely played infinitely often. That is, ∀ a ∈ [ K ] : N ∞ ( a ) = ∞ almost surely.

Proof sketch 2 . For any arm a ′ ∈ [ K ] such that N ∞ ( a ′ ) &lt; ∞ , the Extended Borel-Cantelli (Breiman, 1992) Lemma implies ∑ ∞ t =0 π t ( a ′ ) &lt; ∞ . Since such an arm is sampled only finitely many times, its parameter θ t ( a ′ ) remains bounded, sup t | θ t ( a ′ ) | &lt; ∞ , and its probability converges to zero: lim t →∞ π t ( a ′ ) = 0 . Without loss of generality, let a ∈ [ K ] be an arm with N ∞ ( a ) &lt; ∞ . The condition lim t →∞ π t ( a ) = 0 requires that some parameter grows unboundedly, i.e. lim t →∞ max a ′ ∈ [ K ] θ t ( a ′ ) = ∞ . To preserve the total probability mass, this necessitates that some parameter must diverge to negative infinity: lim t →∞ min a ′ ∈ [ K ] θ t ( a ′ ) = -∞ . Thus, there exists at least one arm b ∈ [ K ] such that lim inf t →∞ θ t ( b ) = -∞ . Furthermore, since the sum of probabilities for finitely sampled arms is finite, any arm b with lim inf t →∞ θ t ( b ) = -∞ must be sampled infinitely often ( N ∞ ( b ) = ∞ ).

We use these properties of arms a (finitely sampled, bounded parameter) and b (infinitely sampled, parameter unbounded below) to construct a proof by contradiction. The fact that arm b is sampled infinitely often despite its parameter repeatedly dropping to arbitrarily low values implies that θ t ( b ) must periodically increase to become larger than θ t ( a ) (and other bounded parameters) infinitely often. Consider the event C t := { θ t ( b ) &lt; θ t ( a ) , a t = b } . We first show that if θ t ( b ) ≤ θ t ( a ) and the parameter update causes θ t +1 ( b ) &gt; θ t +1 ( a ) , this implies a t = b . We then prove that the event C t occurs only a finite number of times. However, for arm b to be sampled infinitely often ( N ∞ ( b ) = ∞ ) while lim inf θ t ( b ) = -∞ and θ t ( a ) is bounded, it must be sampled infinitely often during periods when θ t ( b ) &lt; θ t ( a ) . This contradicts the finding that C t occurs only finitely often, proving our initial assumption ( N ∞ ( a ) &lt; ∞ for some arm a ) is false.

## 4.3 Convergence Without the Assumption of Unique Rewards

Our new result about the exploration of SPG in the bandit setting, Lemma 4.1, allows us to remove an assumption necessary for the results of prior work (Mei et al., 2024a,b), namely that there are no ties in the true mean rewards of the arms (Assumption 3.1). However, this requires new analysis beyond the exploration proof. In this section we sketch out the steps used to show our central result in the bandit setting: that Algorithm 1 converges almost surely regardless of the learning rate.

Theorem 4.2 (Convergence in Bandits) . In the bandit setting of Section 4.1 without Assumption 3.1, Algorithm 1 with any η ∈ Θ(1) almost surely converges to playing optimal arms,

<!-- formula-not-decoded -->

The proof of this theorem breaks into two propositions, the first of which being that the sum of parameters of optimal arms tends to infinity (excluding the trivial case where all arms are equally good and Section 4.1 holds vacuously).

̸

Proposition 4.3 (Infinite Optimal Parameters) . If A ∗ = [ K ] then lim t →∞ ∑ a ∈A ∗ θ t ( a ) = ∞ a.s.

The second proposition states that all finite arms individually have their parameters diverge to negative infinity.

Proposition 4.4 (Negative Infinite Suboptimal Parameters) . For every suboptimal arm b ∈ [ K ] \ A ∗ , lim t →∞ θ t ( b ) = -∞ a.s.

Equipped with these two propositions, the proof of Theorem 4.2 becomes straightforward enough that we need not resort to a proof sketch:

̸

Proof of Theorem 4.2. If A ∗ = [ K ] then ∑ a ∈A ∗ π t ( a ) = 1 for all t ≥ 0 and the result holds vacuously. Henceforth suppose A ∗ = [ K ] . We have that lim t →∞ ∑ a ∈A ∗ π t ( a ) = 1 -lim t →∞ ∑ b ∈ [ K ] \A ∗ π t ( b ) , so it suffices to show that, for all b ∈ [ K ] \ A ∗ , lim t →∞ π t ( b ) = 0 . To this end fix b ∈ [ K ] \ A ∗ . We have the following bound from expanding the definition of π t :

<!-- formula-not-decoded -->

Proposition 4.4 implies that the upper limit in Eq. (8) approaches 0 and Proposition 4.3 implies that the lower limit goes to infinity. Thus lim t →∞ π t ( b ) = 0 , concluding the proof.

The proofs of Propositions 4.3 and 4.4 are long and technical, and we refer the reader to the appendix for the details.

## 5 Reinforcement Learning

The results in RL depend on the results of Section 4, but in order to apply them we will need to port them to a slightly generalized bandit problem. We describe the necessary modifications in the following subsection, before proceeding to MDPs.

## 5.1 Nonstationary Bandit Setting

We still consider a K -armed bandit, with K ≥ 2 and rewards in [ -R,R ] . The interaction between the learner and the environment is much the same as in Section 4.1, with the exception that now the reward distributions are allowed to change across timesteps. That is, we change out the distribution P a ∈ M 1 ([ -R,R ]) of rewards given that arm a is played with a sequence of such distributions ( P t a ) t ≥ 1 , and the reward at each iteration t ≥ 1 is sampled from P t a t ∈ M 1 ([ -R,R ]) ; we also allow the expected rewards given that an arm is played to vary over time, so r ( a ) becomes ( r t ( a )) t ≥ 1 , and we have E [ r t | a t = a ] = r t ( a ) .

̸

However, we constrain the setting in two ways. First, we suppose that there exists a filtration ( F t ) t ≥ 0 such that P t , r t are F t -1 -measurable and a t , r t are F t -measurable. Intuitively, F t contains the information available to the learner at iteration t , and this assumption means that the reward distributions (and thus their means) may only depend on the arms played and rewards observed up to the current timestep, as well as additional sources of randomness that are independent of the future. The second constraint on the environment is that we assume the existence of a 'true' mean reward vector r ∈ [ -R,R ] K , and suppose that there exists some random timestep τ such that, for all t ≥ τ and all a ∈ [ K ] , | r ( a ) -r t ( a ) | ≤ ∆ / 3 , where ∆ := min a,b ∈ [ K ] : r ( a ) = r ( b ) | r ( a ) -r ( b ) | is the minimum nonzero gap in the 'true' mean reward between any two arms. This says that eventually the expected reward of playing arm a will settle down to a neighbourhood of r ( a ) , and in particular that the arms in A ∗ := arg max a ∈ [ K ] r ( a ) have the highest expected reward after iteration τ . Given these modifications to the bandit setting, we can extend the results of Section 4 with minimal changes. The algorithm stays exactly the same, with the only modification to Algorithm 1 being that, at line 3, r t +1 ∼ P a t +1 becomes r t +1 ∼ P t +1 a t +1 .

After extending all the bandit results to the nonstationary bandit setting, we can finally apply them for a result in RL.

## 5.2 Reinforcement Learning Setting

We consider a finite-horizon MDP, defined by the tuple M = ( H , S , A , { r h } H -1 h =0 , { P h } H -1 h =0 , ρ ) , where H = { 0 , 1 , . . . , H -1 } is the index set of timesteps in an episode; S = S 0 ∪ . . . ∪ S H -1 and A = A 0 ∪ . . . ∪ A H -1 are finite state and action spaces, respectively, with S h ( A h = ∪ s ∈S h A s ) being the sets of possible states and (actions) at step h ∈ H , and A s is the set of possible actions from state s; r h : S h × A h → [ -R,R ] is a reward function that is bounded by R &gt; 0 ; P h : S h × A h → M 1 ( S h +1 ) is the transition function; and ρ : S 0 → M 1 ( S 0 ) is the initial state distribution. We denote π := ( π h ) H -1 h =0 as a time-dependent policy where π h : S h →M 1 ( A h ) is the policy in the horizon h . An episode proceeds under the following protocol. At the beginning of the episode, the learner selects a non-stationary policy π . The episode then evolves through s 0 ∼ ρ and a h ∼ π h ( · | s h ) , s h +1 ∼ p h ( · | s h , a h ) , r h = r h ( a h , s h ) for all h ∈ H . We define the trajectory τ := ( s 0 , a 0 , r 0 , s 1 , . . . , s H -1 , a H -1 , r H -1 ) . Therefore, the probability of a given trajectory τ is

<!-- formula-not-decoded -->

We also define the value functions and action-value functions for h ∈ H

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The goal is to find a time-dependent policy π ∗ that maximizes the state-value function at time 0, i.e. V π 0 ( ρ ) := E s ∼ ρ [ V π 0 ( s )] :

<!-- formula-not-decoded -->

We also define optimal state and state-action value function, V ∗ h := V π ∗ h and Q ∗ h := Q π ∗ h . In this paper, we focus on softmax parameterized policies. Specifically, we parameterize each π h by θ h for all h ∈ H by

<!-- formula-not-decoded -->

where θ h ∈ R A h with A h := ∑ s ∈ S h |A s | for all h ∈ H . To improve the readability, we will sometimes write π t in place of π θ t and π h t in place of π h θ h t . The true gradient of the Eq. (12) is

<!-- formula-not-decoded -->

where I [ a h = a ] is the indicator function of whether action a is played in the horizon h , for all s ∈ S h , a ∈ A h , h ∈ H . Since we are in the stochastic setting, we will use REINFORCE estimator to estimate the gradient and update the parameters

<!-- formula-not-decoded -->

The REINFORCE algorithm is shown in Algorithm 2.

## Algorithm 2 REINFORCE

- 1: for each episode do
- 2: Sample a trajectory τ using ρ, { π θ h } H -1 h =1 , { P h } H -1 h =1
- 3: for all a ∈ |A| , s ∈ |S| do
- 4: Use Eq. (15) to update θ ( s, a )
- 5: end for
- 6: end for

We first show an exploration result, the counterpart to the exploration result shown above in the bandit setting, before sketching the proof of our main theorem in RL.

## 5.3 RL Exploration Lemma

Lemma 5.1. Running REINFORCE with any η ∈ Θ(1) in a finite-horizon MDP M , for all h ∈ H , for all reachable s ∈ S h and for all a ∈ A s we have, almost surely, that every reachable state action pair will be visited i.o, i.e N ∞ ( s, a ) = ∞ .

Proof sketch 3 . First, we show that for all horizon h ∈ H , if s ∈ S h is reachable and played i.o, then all actions a ∈ S are also played i.o by Lemma 4.1. Next, we use induction to show that for all horizon h ∈ H , if s ∈ S h is reachable visited i.o, then s ′ ∈ S h +1 is also visited i.o. Therefore, for all h ∈ H , all reachable state-action pairs ( s, a ) ∈ S × A will be played i.o.

## 5.4 Convergence in finite-horizon MDP

Theorem 5.2 (Convergence in RL) . For the MDP defined as above, using REINFORCE with constant learning rate η ∈ Θ(1) , we have, almost surely, for all s ∈ S 0 , V π t 0 ( s ) → V ∗ 0 ( s ) as t →∞

Proof sketch 4 . We show the convergence theorem using the backward induction. Suppose for all horizon h ∈ { h ′ , . . . , H -1 } , we have ∑ a ∈A ∗ s π h t ( a | s ) → 1 for all s ∈ S h , we want to show that ∑ a ∈A ∗ s π h -1 t ( a | s ) → 1 for all s ∈ S h -1 . Since ∑ a ∈A ∗ s π h t ( a | s ) → 1 for all s ∈ S h , where h ∈ { h ′ , . . . , H -1 } , we know that there exists time step τ s.t V ∗ h ( s ) -V π t h ( s ) ≤ δ 3 , where δ is the minimum non-zero gap between two Q-values. For all a ∈ A s h -1 , there exists a minimum gap of δ 3 in the Q-value. Therefore, applying the bandit convergence result, we know that ∑ a ∈A ∗ s π h -1 t ( a | s ) → 1 as t → ∞ for all s ∈ S h -1 . Recursively, we know that for all s ∈ S 0 , ∑ a ∈A ∗ s π 0 θ t ( a | s ) → 1 as t →∞ .

We also provide the statement and proof of convergence rate in the appendix (Theorem E.3).

## 5.5 Simulations

We conduct several experiments to illustrate the convergence behavior of REINFORCE algorithm in the finite-horizon setting. Experiments are performed using a chain MDP (Fig. 1b) with state space S = { s 0 , . . . , s 3 , T 1 , T 2 } , where T 1 and T 2 are terminal states, and action space A = { a 0 , a 1 } . Taking action a 0 in any state yields a mean reward of 0 . 5 and transitions to a terminal state T 1 . Taking action a 1 in state s i ( i ∈ { 0 , 1 , 2 } ) yields a mean reward of -0 . 5 and transitions to state s i +1 . In state s 3 , action a 1 yields a mean reward of 7 and transitions to a terminal state T 2 . The policy is parameterized using a softmax function, and parameters are initialized to 0 ∈ R |S|×|A| . For each learning rate η , the REINFORCE algorithm is run for 10 5 episodes across 30 seeds. Performance is evaluated by measuring the average suboptimality gap from the initial state distribution ρ , defined as V ∗ 0 ( ρ ) -V π t 0 ( ρ ) , over the 30 seeds. Our first experiment (Fig. 3a) demonstrates the benefits of using a large learning rate. Previous convergence analysis of REINFORCE (Theorem 4.1 Klein et al., 2024) relies on small constant learning rates, which can significantly impede practical training speed. For instance, the analysis in Klein et al. (2024) guarantees convergence with η = 1 5 H 2 R √ T , where T

is the number of training episodes. In our environment ( H = 4 , R = 7 , T = 10 5 ), this corresponds to an extremely small learning rate η ≈ 10 -7 . Therefore, we evaluated REINFORCE algorithm with larger learning rates η ∈ { 0 . 00001 , 0 . 001 , 0 . 1 } . Fig. 3a shows that the suboptimality gap remains nearly constant for η = 0 . 00001 , whereas it decreases substantially faster as η increases from 0 . 001 to 0 . 1 . This demonstrates the practical benefit of employing larger learning rates for accelerated convergence, supported by our theoretical guarantees. We further explore the effect of even larger learning rates η ∈ { 0 . 5 , 1 , 2 } , presented in Fig. 3b. These rates, while potentially accelerating learning if updates are favorable, generally slow down convergence compared to the moderately large rates. The suboptimality curves exhibit more abrupt changes and show less consistent improvement over episodes. The large shaded regions indicate significantly higher variance with these very large learning rates. This suggests that large steps can easily push parameters away from optimal configurations, leading to prolonged exploration of suboptimal regions until a corrective update is sampled. Finally, Fig. 4 illustrates the evolution of the learned policy for optimal actions at each horizon. For all learning rates, we observe, on average, that the probability of selecting optimal actions converges first for the last horizon, then for the second-to-last, and so on, proceeding backward through the horizon. This backward convergence pattern in policy probabilities is consistent with our proof strategy for the convergence of the REINFORCE algorithm, which relies on a backward

induction approach. We also extend our experiments to demonstrate the relationship between the algorithm's performance and different learning rates. Details on the experimental setups and results can be found in the section Appendix F. Overall, we consistently find a "bowl-shaped" relationship between the learning rate and performance, meaning both exessively small and exessively large learning rates lead to high suboptimality, while middling values achieve the smallest suboptimality. The specific shape and optimal point of this bowl vary significantly with the environment's structure.

Remark 5.3 . It is worth noting that not all environments exhibit this specific backward convergence pattern in learning optimal policy.

<!-- image -->

- (a) Benefits of using larger learning rates.

<!-- image -->

Episodes (t)

- (b) Drawbacks of using exessively large learning rates.

Figure 3: Fig. 3a shows that using a larger learning rate can improve the performance of REINFORCE, while Fig. 3b shows that excessively large learning rates have substantial variance, which can slow down the convergence rate.

## 6 Conclusions and Future work

This work enhances our understanding of the convergence properties of the widely used REINFORCE algorithm. Our novel proof offers deeper insights into the exploration effects of stochastic gradient methods and raises new research questions. Notably, recent findings by Mei et al. (2024a) indicate a convergence rate of O (log( t ) /t ) for stochastic gradient bandit algorithms. As demonstrated in Fig. 3b, REINFORCE with excessively large learning rates exhibits high variance, impeding convergence. Future work could explore optimal learning rate schedules to harness the initial benefits of larger rates while subsequently mitigating variance.

## 7 Acknowledgement and Disclosure of funding

The authors would like to thank Jeffrey Rosenthal for pointing out relevant results and materials regarding non-convergence results. Csaba Szepesvári and Dale Schuurmans gratefully acknowledge funding from the Canada CIFAR AI Chairs Program, Amii and NSERC.

Figure 4: These figures show the convergence rate of the optimal policy in each horizon for different learning rates. In particular, we observe that the optimal policy of the last horizon will converge first, then the second-to-last one until the first horizon. This observation aligns with our analysis.

<!-- image -->

## References

- P.-A. Absil, R. Mahony, and B. Andrews. Convergence of the iterates of descent methods for analytic cost functions. SIAM Journal on Optimization , 16(2):531-547, 2005.
- A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift, 2020. URL https://arxiv.org/abs/1908.00261 .
- I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, et al. Solving rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113 , 2019.
- M. Bramson, J. Quastel, and J. S. Rosenthal. When can martingales avoid ruin? preprint , 2004.
- L. Breiman. Probability . SIAM, 1992.
- S. Cen, C. Cheng, Y. Chen, Y. Wei, and Y. Chi. Fast global convergence of natural policy gradient methods with entropy regularization, 2021. URL https://arxiv.org/abs/2007.06558 .
- Y.-J. Chen, N.-C. Huang, C. pei Lee, and P.-C. Hsieh. Accelerated policy gradient: On the convergence rates of the nesterov momentum for reinforcement learning. In Forty-first International Conference on Machine Learning , 2024. URL https://openreview.net/forum?id=aeXRBnLoPP .
- B. Davis. Divergence properties of some martingale transforms. The Annals of Mathematical Statistics , 40(5): 1852-1854, 1969.
- Y. Ding, J. Zhang, and J. Lavaei. On the global optimum convergence of momentum-based policy gradient, 2022. URL https://arxiv.org/abs/2110.10116 .
- Y. Ding, J. Zhang, H. Lee, and J. Lavaei. Beyond exact gradients: Convergence of stochastic soft-max policy gradient methods with entropy regularization, 2024. URL https://arxiv.org/abs/2110.10117 .
- J. L. Doob. Measure theory , volume 143. Springer Science &amp; Business Media, 2012.
- I. Fatkhullin, A. Barakat, A. Kireeva, and N. He. Stochastic policy gradient methods: Improved sample complexity for fisher-non-degenerate policies, 2023. URL https://arxiv.org/abs/2302.01734 .
- D. A. Freedman. On tail probabilities for martingales. the Annals of Probability , pages 100-118, 1975.
- S. Khodadadian, P. R. Jhunjhunwala, S. M. Varma, and S. T. Maguluri. On the linear convergence of natural policy gradient algorithm, 2021. URL https://arxiv.org/abs/2105.01424 .
- S. Klein, S. Weissmann, and L. Döring. Beyond stationarity: Convergence analysis of stochastic softmax policy gradient methods, 2024. URL https://arxiv.org/abs/2310.02671 .
- G. Lan. Policy mirror descent for reinforcement learning: Linear convergence, new sampling complexity, and generalized problem classes, 2022. URL https://arxiv.org/abs/2102.00135 .
- T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y . Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning, 2019. URL https://arxiv.org/abs/1509.02971 .
- J. Liu, W. Li, and K. Wei. Elementary analysis of policy gradient methods. arXiv preprint arXiv:2404.03372 , 2024. URL https://arxiv.org/abs/2404.03372 .
- S. Masiha, S. Salehkaleybar, N. He, N. Kiyavash, and P. Thiran. Stochastic second-order methods improve best-known sample complexity of sgd for gradient-dominated function, 2023. URL https://arxiv.org/ abs/2205.12856 .
- J. Mei, C. Xiao, B. Dai, L. Li, C. Szepesvári, and D. Schuurmans. Escaping the gravitational pull of softmax. Advances in Neural Information Processing Systems , 33:21130-21140, 2020a.
- J. Mei, C. Xiao, C. Szepesvari, and D. Schuurmans. On the global convergence rates of softmax policy gradient methods, 2020b. URL https://arxiv.org/abs/2005.06392 .
- J. Mei, B. Dai, C. Xiao, C. Szepesvari, and D. Schuurmans. Understanding the effect of stochasticity in policy optimization, 2021. URL https://arxiv.org/abs/2110.15572 .
- J. Mei, Y. Gao, B. Dai, C. Szepesvari, and D. Schuurmans. Leveraging non-uniformity in first-order non-convex optimization, 2022. URL https://arxiv.org/abs/2105.06072 .
- J. Mei, W. Chung, V. Thomas, B. Dai, C. Szepesvari, and D. Schuurmans. The role of baselines in policy gradient optimization, 2023. URL https://arxiv.org/abs/2301.06276 .

- J. Mei, B. Dai, A. Agarwal, S. Vaswani, A. Raj, C. Szepesvari, and D. Schuurmans. Small steps no more: Global convergence of stochastic gradient bandits for arbitrary learning rates. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024a. URL https://openreview.net/forum? id=q9dKv1AK6l .
- J. Mei, Z. Zhong, B. Dai, A. Agarwal, C. Szepesvari, and D. Schuurmans. Stochastic gradient succeeds for bandits, 2024b. URL https://arxiv.org/abs/2402.17235 .
- L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155 .
- B. Polyak. Gradient methods for the minimisation of functionals. USSR Computational Mathematics and Mathematical Physics , 3(4):864-878, 1963. ISSN 0041-5553. doi: https://doi.org/10.1016/0041-5553(63) 90382-3. URL https://www.sciencedirect.com/science/article/pii/0041555363903823 .
- R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290 .
- J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. Trust region policy optimization, 2017a. URL https://arxiv.org/abs/1502.05477 .
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017b. URL https://arxiv.org/abs/1707.06347 .
- R. Sutton and A. Barto. Reinforcement Learning, second edition: An Introduction . Adaptive Computation and Machine Learning series. MIT Press, 2018. ISBN 9780262039246. URL https://books.google.ca/ books?id=sWV0DwAAQBAJ .
- O. Vinyals, I. Babuschkin, W. M. Czarnecki, M. Mathieu, A. Dudzik, J. Chung, D. H. Choi, R. Powell, T. Ewalds, P. Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. nature , 575 (7782):350-354, 2019.
- R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8:229-256, 1992.
- L. Xiao. On the convergence rates of policy gradient methods, 2022. URL https://arxiv.org/abs/2201. 07443 .
- J. Zhang, J. Kim, B. O'Donoghue, and S. Boyd. Sample efficient reinforcement learning with reinforce, 2020a. URL https://arxiv.org/abs/2010.11364 .
- J. Zhang, A. Koppel, A. S. Bedi, C. Szepesvari, and M. Wang. Variational policy gradient method for reinforcement learning with general utilities, 2020b. URL https://arxiv.org/abs/2007.02151 .

☞

In this appendix we will deal repeatedly with almost sure events, i.e. events that occur with probability 1. We typically mention this throughout the proofs, except for in one important case where 'a.s.' is omitted to reduce clutter: whenever statements involving conditional expectations (by extension conditional probabilities, variances) do not have an explicit probabilistic quantification, they are understood to hold almost surely. Of course, this is the only possible interpretation for such statements, since conditional expectations are only defined up to a set of measure 0.

## A Technical Tools

We begin with some fundamental results from probability theory. The first is a generalization of the Borel-Cantelli Lemma.

Lemma A.1 (Extended Borel-Cantelli Lemma, Corollary 5.29 of Breiman (1992)) . Given a filtration ( F t ) t ≥ 0 and a sequence of events ( A t ) t ≥ 0 with A t ∈ F t for all t ≥ 0 ,

<!-- formula-not-decoded -->

That is, ( A t ) t ≥ 0 occurs i.o. if and only if ∑ t ≥ 0 P ( A t |F t -1 ) is infinite, up to a set of measure zero.

Our analysis relies critically and repeatedly on the celebrated inequality of Freedman. The version we will use is similar to the one stated by Mei et al. (2024a,b). Since we require a general filtration, we include the original statement by Freedman below in Lemma A.2, followed by the statement and derivation of the form most convenient to us in Lemma A.3. Whenever we mention 'Freedman's inequality' elsewhere in this work it shall refer to the latter.

Lemma A.2 ((Original) Freedman's Inequality, Theorem 1.6 of Freedman (1975)) . Given a filtered probability space with filtration ( F t ) t ≥ 0 , an adapted sequence of random variables ( X t ) t ≥ 1 , and constants a, b &gt; 0 , if ∀ t ≥ 1 : E [ X t |F t -1 ] = 0 and | X t | ≤ 1 then

<!-- formula-not-decoded -->

Lemma A.3 (Freedman's Inequality) . Let ( X t ) t ≥ 1 be a random sequence adapted to the filtration ( F t ) t ≥ 0 , B ≥ 0 be a constant such that ∀ t ≥ 0 : | X t | ≤ B , and denote V t := ∑ i ∈ [ t ] Var[ X i |F i -1 ] . For any δ ∈ (0 , 1] , it holds with probability 1 -δ that

<!-- formula-not-decoded -->

Remark A.4 . The derivation of Lemma A.3 closely follows the proof of Theorem C.3 of Mei et al. (2024b). We aimed for a simple bound rather than a tight one.

Proof. Fix ϵ ∈ (0 , 1) , and let S t := ∑ i ∈ [ t ] X i and V t := ∑ i ∈ [ t ] Var[ X i |F i -1 ] . First we will suppose that E [ X t |F t -1 ] = 0 and | X t | ≤ 1 for all t ≥ 1 , and show that

<!-- formula-not-decoded -->

For x ≥ 1 let g ( x ) := 3 log(( x +2) 2 /ϵ ) , and we have

<!-- formula-not-decoded -->

Setting x := V t in Eq. (25) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To control the term appearing in the exp above, we will use the following inequality, which holds for u ≥ 2 and i ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since g ( i ) ≥ 3 log(4) ≥ 2 , we can combine the above two displays by setting u := g ( i ) and conclude

<!-- formula-not-decoded -->

We are finished showing Eq. (19). We can apply this result to both ( X t ) t ≥ 0 and ( -X t ) t ≥ 0 , setting ϵ := δ/ 2 in each application, whence a union bound guarantees that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Given a random sequence ( X t ) t ≥ 1 that satisfies | X t | ≤ B for some B ≥ 1 , we can apply Eq. (41) to the sequence ( X t /B ) t ≥ 1 :

<!-- formula-not-decoded -->

with probability 1 -δ . Combining Eq. (41), which holds for | X t | ≤ 1 , and Eq. (44), which holds for | X t | ≤ B where B ≥ 1 , and upper bounding max( B 2 , 1) ≤ B 2 +1 , we can remove the requirement that B ≥ 1 and conclude that, if E [ X t |F t -1 ] = 0 and | X t | ≤ B for all t ≥ 1 , with probability 1 -δ

<!-- formula-not-decoded -->

To remove the assumption that ∀ t ≥ 1 : E [ X t |F t -1 ] = 0 and get the desired result, we apply Eq. (45) to ( X t -E [ X t |F t -1 ]) t ≥ 1 , noting that if | X t | ≤ B then | X t -E [ X t |F t -1 ] | ≤ 2 B .

The following result applies Freedman's Inequality to a sequence of bounded, and eventually (conditionally) self-bounded, random variables. It says that if the conditional expectations are not summable then the variables themselves will not be summable. We expect that the result is folklore, but cannot find a reference.

Lemma A.5 (Freedman Divergence Trick) . Let ( X t ) t ≥ 1 be a random sequence adapted to the filtration ( F t ) t ≥ 0 and B ≥ 0 be a constant such that ∀ t ≥ 0 : | X t | ≤ B . Suppose ∑ t ≥ 1 E [ X t |F t -1 ] = ∞ and, for some random (a.s. finite) index τ ≥ 1 and constant C ≥ 0 , for all t ≥ τ , Var[ X t |F t -1 ] ≤ C E [ X t |F t -1 ] . Then ∑ t ≥ 1 X t = ∞ a.s.

Remark A.6 . Note that the result does not require τ to be a stopping time.

Proof. For t ≥ 0 let S t := ∑ i ∈ [ t ] X i , S t := ∑ i ∈ [ t ] E [ X i |F i -1 ] , and V t := ∑ i ∈ [ t ] Var[ X i |F i -1 ] . For any δ ∈ (0 , 1] , we can apply Freedman's Inequality (Lemma A.3) to ( X t ) t ≥ 1 . This gives that, with probability 1 -δ , for any t ≥ τ ,

<!-- formula-not-decoded -->

By assumption lim t S t = ∞ , so lim t S t -S τ = ∞ as well. Clearly, the subtrahend in the display above is o ( S t -S τ ) . Hence, taking the limit of t →∞ , we have lim t S t = ∞ with probability 1 -δ . Since δ was arbitrary, this also holds with probability one (by taking δ → 0 ).

Finally, we will need a classic result of Doob.

Lemma A.7 (Doob's Martingale Convergence Theorem (Doob, 2012)) . Given a random sequence ( X t ) t ≥ 1 adapted to the filtration ( F t ) t ≥ 0 , if ∀ t ≥ 1 : E [ X t |F t -1 ] ≤ X t -1 and sup t ≥ 0 E [ -min( X t , 0)] &lt; ∞ , then ( X t ) t ≥ 1 converges a.s. In particular, X t → X a.s. as t →∞ , where X := lim sup t X t and E [ | X | ] &lt; ∞ .

## B Bandits

In this section all results are stated in the bandit setting described in Section 4.1. The first subsection establishes the results needed for the proof of convergence in bandits, Theorem 4.2. The second subsection proves that convergence is not always to a one-hot policy, Proposition 3.2

## B.1 Convergence

We begin with a simple but crucial property of Algorithm 1, which follows from a symmetry of the update rule.

Lemma B.1 (Conservation of mass) . For all t ≥ 0 , ∑ a ∈ [ K ] θ t ( a ) = ∑ a ∈ [ K ] θ 0 ( a ) .

Proof. Proceeding by induction, the base is tautological; recalling that a t is the arm played at time t , we have

<!-- formula-not-decoded -->

The rest of the proofs in this section will refer to the filtration ( F t ) t ≥ 0 defined by F t := σ (( a i , r i ) i&lt;t ) , and we adopt the shorthands E t [ · ] := E [ · |F t ] and Var t [ · ] := Var[ · |F t ] . The following result is a stronger version of Lemma 2 of Mei et al. (2024a), and it guarantees that Algorithm 1 explores enough to keep trying all arms forever regardless of the observations.

Lemma 4.1 (Bandit Exploration) . Using Algorithm 1 with any constant learning rate η ∈ Θ(1) , every arm is almost surely played infinitely often. That is, ∀ a ∈ [ K ] : N ∞ ( a ) = ∞ almost surely.

Proof. The first step is to show that, for any arm b ∈ [ K ] , if |{ t ≥ 0 : a t = b }| &lt; ∞ then sup t | θ t ( b ) | &lt; ∞ a.s. Picking b ∈ [ K ] and setting m := sup( { 0 } ∪ { t ≥ 0 : a t = b } ) , without assuming |{ t ≥ 0 : a t = b }| &lt; ∞ we have the bound

<!-- formula-not-decoded -->

Also, the Extended Borel-Cantelli Lemma (Lemma A.1) applied to ( F t ) t ≥ 0 with the event sequence A t := { a t = b } implies

<!-- formula-not-decoded -->

If |{ t ≥ 0 : a t = b }| &lt; ∞ then m&lt; ∞ and ∑ t ≥ 0 I [ a t = b ] &lt; ∞ , and the latter inequality together with Eq. (61) implies ∑ t ≥ 0 π t ( b ) &lt; ∞ a.s, thus Eq. (60) yields sup t | θ t ( b ) | ≤ α ( b ) &lt; ∞ a.s. A union bound over b ∈ [ K ] implies that almost surely

<!-- formula-not-decoded -->

We are ready to fix an arm a ∈ [ K ] and show that the event E := { |{ t ≥ 0 : a t = a }| &lt; ∞ } has probability 0. For the remainder of the proof until the almost the very end we will work under the

assumption that E occurs. On E we have α ( a ) &lt; ∞ a.s, which implies ∑ t ≥ 0 π t ( a ) &lt; ∞ a.s, which in turn implies lim t π t ( a ) = 0 a.s. The definition of π t ( a ) gives us

<!-- formula-not-decoded -->

so from lim t π t ( a ) = 0 a.s. we get lim t max b ∈ [ K ] θ t ( b ) = ∞ a.s. Then conservation of mass (Lemma B.1) implies that lim t min b ∈ [ K ] θ t ( b ) = -∞ a.s. By Eq. (62) all arms that are selected only finitely often have parameters bounded away from -∞ a.s, so there is a.s. an arm b that is played i.o. with lim inf t θ t ( b ) = -∞ . We will refer to such an arm as b for the remainder of the proof. However, because b is played i.o, another application of the Extended Borel-Cantelli Lemma (to ( F t ) t ≥ 0 with events A t := { a t = b } ) yields ∑ t ≥ 0 π t ( b ) = ∞ a.s. Since ∑ t ≥ 0 π t ( a ) &lt; ∞ a.s, we have that π t ( b ) &gt; π t ( a ) , and equivalently θ t ( b ) &gt; θ t ( a ) , for infinitely many t ≥ 0 a.s. In summary, θ t ( b ) oscillates from being arbitrarily low to being larger than θ t ( a ) ≥ -α ( a ) . 2

We will now argue that, for sufficiently large t , if θ t ( b ) ≤ θ t ( a ) but θ t +1 ( b ) &gt; θ t +1 ( a ) then a t = b . Let T be the minimum timestep such that, for all t ≥ T ,

̸

<!-- formula-not-decoded -->

Since we are working on the event E we have a t = a for only finitely many t , log( ηR ) + α ( a ) &lt; ∞ a.s, and lim t max c ∈ [ K ] θ t ( c ) = ∞ a.s; taken together, these observations imply that T &lt; ∞ exists a.s.

̸

For t ≥ T , suppose θ t ( b ) ≤ θ t ( a ) and a t = b , and we must show

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since θ t ( b ) ≤ θ t ( a ) we have π t ( b ) ≤ π t ( a ) , and standard inequalities yield

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

≤

ηR

exp(

θ

t

(

a

))

exp(max

-

c

∈

[

exp(

K

]

θ

θ

t

(

t

(

c

b

))

))

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus Eq. (69) holds and we have established that, for all t ≥ T , if θ t ( b ) ≤ θ t ( a ) and θ t +1 ( b ) &gt; θ t +1 ( a ) then a t = b . Since θ t ( b ) fluctuates from below θ t ( a ) to above it i.o, we have that the events in the sequence ( B t ) t ≥ 0 defined by B t := { θ t ( b ) ≤ θ t ( a ) , a t = b } ∈ F t +1 occur i.o. a.s. Applying the Extended Borel-Cantelli Lemma to ( F t ) t ≥ 0 and ( B t ) t ≥ 0 implies that ∑ t ≥ 0 P ( B t |F t ) = ∞ a.s. However,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At this point we have that, on event E , both ∑ t ≥ 0 P ( B t |F t ) &lt; ∞ a.s. and ∑ t ≥ 0 P ( B t |F t ) = ∞ a.s. Since these events are mutually exclusive they both occur with probability 0, and since they are jointly exhaustive we have P ( E ) = 0 .

2 It is easy to see that θ t ( b ) must also become arbitrarily large i.o, but this is not necessary for the proof.

The proof of our main result in the bandit setting, that lim t →∞ ∑ a ∈A ∗ π t ( a ) = 1 , is broken into two propositions: the first guarantees that lim t →∞ ∑ a ∈A ∗ θ t ( a ) = ∞ , in particular that as the time steps get large at least one a ∈ A ∗ will have an arbitrarily large parameter 3 ; the second proposition says that lim t →∞ θ t ( b ) = -∞ for all b ∈ [ K ] \ A ∗ . Taken together, the propositions imply that eventually some (potentially time step dependent) optimal arm dominates every suboptimal arm, establishing Convergence in Bandits (Theorem 4.2). We now turn to proving the two propositions.

̸

The subsequent proofs will go a little smoother with some extra notation; we define ∆ := min a,b ∈ [ K ]: r ( a ) = r ( b ) | r ( a ) -r ( b ) | to be the minimum nonzero gap between expected rewards of arms and r ( A ∗ ) := max a ∈ [ K ] r ( a ) to be the maximum attainable expected reward. Finally, we overload π t ( · ) to take sets as input, i.e, given S ⊂ [ K ] we let π t ( S ) := ∑ a ∈S π t ( a ) be the probability that an arm in S is selected. With these abbreviations in hand, the first proposition is as follows.

̸

Proposition 4.3 (Infinite Optimal Parameters) . If A ∗ = [ K ] then lim t →∞ ∑ a ∈A ∗ θ t ( a ) = ∞ a.s.

Proof. For t ≥ 0 , let X t := ∑ a ∈A ∗ θ t +1 ( a ) -θ t ( a ) , such that ∑ t i =0 X i = ∑ a ∈A ∗ θ t +1 ( a ) -θ 0 ( a ) . By the update rule of Algorithm 1, note also that

<!-- formula-not-decoded -->

The conditional expectation of X t given F t can be lower bounded by

<!-- formula-not-decoded -->

3 Excluding the trivial case where A ∗ = [ K ] , i.e. all arms are equally good.

and the conditional variance can be upper bounded by

<!-- formula-not-decoded -->

Thus for all t ≥ 0 we have Var t [ X t ] ≤ ηR 2 ∆ -1 E t [ X t ] , | X t | ≤ ηR , and X t is F t +1 -measurable. Setting b := ηR, τ := 0 , and c := ηR 2 ∆ -1 , we need only to prove that ∑ t ≥ 0 E t [ X t ] = ∞ , at which point we can apply the Freedman Divergence Trick (Lemma A.5) to conclude

<!-- formula-not-decoded -->

Thus in the remainder of the proof we turn our attention to showing ∑ t ≥ 0 E t [ X t ] = ∞ . Applying Eq. (84) and η ∆ &gt; 0 , we need only show that

<!-- formula-not-decoded -->

̸

Lemma 4.1 together with ∅ ̸ = A ∗ = [ K ] implies that

<!-- formula-not-decoded -->

Since P ( a t ∈ A ∗ |F t ) = π t ( A ∗ ) and P ( a t / ∈ A ∗ |F t ) = 1 -π t ( A ∗ ) , the Extended Borel-Cantelli Lemma (Lemma A.1) applied to Eq. (95) furnishes ∑ t ≥ 0 π t ( A ∗ ) = ∑ t ≥ 0 (1 -π t ( A ∗ )) = ∞ a.s. We now break into cases to show that Eq. (94) holds regardless of the behavior of π t ( A ∗ ) .

If π t ( A ∗ ) ≥ 1 / 2 only finitely often then we can set u := max { t ≥ 0 : π t ( A ∗ ) ≥ 1 / 2 } for

<!-- formula-not-decoded -->

Similarly, if π t ( A ∗ ) &lt; 1 / 2 only finitely often then u := max { t ≥ 0 : π t ( A ∗ ) &lt; 1 / 2 } gives us

<!-- formula-not-decoded -->

We can narrow our focus to the case where π t ( A ∗ ) is both above and below 1 / 2 i.o. In particular, there must be infinitely many t ≥ 0 such that π t ( A ∗ ) &lt; 1 / 2 but π θ t +1 ( A ∗ ) ≥ 1 / 2 , and for such t we have

<!-- formula-not-decoded -->

The above equation is of the form x/ ( x + y ) , where x := ∑ a ∈A ∗ exp( θ t +1 ( a )) and y := ∑ b ∈ [ K ] \A ∗ exp( θ t +1 ( b )) . Since x/ ( x + y ) is increasing in x and decreasing in y for x, y &gt; 0 , and | θ t +1 ( c ) -θ t ( c ) | ≤ ηR for all c ∈ [ K ] , we can maximize the right hand side for the upper bound

<!-- formula-not-decoded -->

Also, π t ( A ∗ ) &lt; 1 / 2 yields ∑ a ∈A ∗ exp( θ t ( a )) &lt; ∑ b ∈ [ K ] \A ∗ exp( θ t ( b )) , so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Connecting the above displays, there are infinitely many t ≥ 0 with π t ( A ∗ ) &lt; 1 / 2 and π θ t +1 ( A ∗ ) ≥ 1 / 2 , and for such t we have 1 -π θ t +1 ( A ∗ ) &gt; 1 -exp(2 ηR ) / (exp(2 ηR ) + 1) = (exp(2 ηR ) + 1) -1 . Therefore π θ t +1 ( A ∗ )(1 -π θ t +1 ( A ∗ )) ≥ (2 exp(2 ηR ) + 2) -1 i.o, establishing Eq. (94).

The second proposition has a more complicated proof, due to the technical difficulty added by having multiple suboptimal arms with the same expected value. Controlling the suboptimal arms will be much more convenient with the following extra notation. Letting n := |{ r ( a ) : a ∈ [ K ] }| be the size of the range of the expected reward vector r , we partition the arms into (Φ i ) i ∈ [ n ] , where Φ i := arg min a ∈ [ K ] \∪ j&lt;i Φ j r ( a ) . Thus Φ 1 is the set of arms with minimal expected reward, Φ 2 is the set of arms with the second lowest expected reward, and so forth, culminating with Φ n = A ∗ . Given i ∈ [ n ] , we will use the shorthands Φ -i := ∪ j&lt;i Φ j and Φ + i := ∪ j&gt;i Φ j . Note that Φ -i and Φ + i are the sets of arms with lower, respectively higher, expected reward than the arms in Φ i . First we will conjure up a couple bound that hold for the increments of suboptimal parameters.

Lemma B.2 (Bounds on the Expectation and Variance of Increments for Suboptimal Arms) . For any i ∈ [ n -1] , for any b ∈ Φ i , we have the bounds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The next proposition will be applied inductively to control the relationship between the expectation and variance of arbitrary suboptimal arms.

Lemma B.3. For constants C , C ′ ≥ 0 and i ∈ [ n -1] , if all b ∈ Φ -i satisfy lim t →∞ θ t ( b ) = -∞ a.s. then there a.s. exists a finite timestep τ such that, for all c ∈ Φ -i +1 , ∑ a ∈A ∗ θ τ ( a ) ≥ C + C ′ θ τ ( c ) .

Proof. Without loss of generality suppose C ′ ≥ 1 . Throughout the proof we will use the following two constants, which depend on C and C ′ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Fix ϵ ∈ (0 , 1] , and define

<!-- formula-not-decoded -->

noting that D &lt; ∞ . Proposition 4.3 says that lim t ∑ a ∈A ∗ θ t ( a ) = ∞ a.s, and we have by assumption that lim t θ t ( b ) = -∞ a.s. for all b ∈ Φ -i . Together these observations guarantee an a.s. finite

timestep µ such that

<!-- formula-not-decoded -->

Consider the collection of sequences { ( X b t ) t&gt;µ : b ∈ Φ -i } defined by X b t := θ t ( b ) -θ t -1 ( b ) , and the sequence ( Y t ) t&gt;µ defined by Y t := ∑ a ∈A ∗ θ t ( a ) -θ t -1 ( a ) . For each b ∈ Φ -i , ( X b t ) t&gt;µ satisfies the requirements of Freedman's Inequality (Lemma A.3) with B := ηR ; also, ( Y t ) t&gt;µ does with B := KηR . Therefore we can apply Freedman's Inequality with δ := ϵ/K to all of these sequences simultaneously and take a union bound to conclude that, with probability 1 -ϵ ( | Φ -i | +1) /K ≤ 1 -ϵ ,

<!-- formula-not-decoded -->

for all t &gt; µ . Let E denote the event that both Eqs. (122) and (123) hold at all such t . We will argue that, on E ,

<!-- formula-not-decoded -->

for all t ≥ µ by strong induction. Thus let t ≥ µ , and suppose that Eq. (124) holds with k in place of t , for all µ ≤ k &lt; t , noting that it holds for k = µ by the definition of µ . Eqs. (84) and (91) together imply that

<!-- formula-not-decoded -->

for k &gt; µ , so the assumption that event E holds implies

<!-- formula-not-decoded -->

Now pick an arbitrary b ∈ Φ -i . Without loss of generality, say b ∈ Φ j for some j &lt; i . For µ ≤ k &lt; t the inductive hypothesis implies that there exists a ∈ A ∗ such that θ k ( a ) ≥ U 1 /K . Thus

<!-- formula-not-decoded -->

where the final inequality above follows from U 1 ≥ K log( K + 4 K/ ∆) . Defining the constant γ := (∆ / 2 + r ( b ) + R ) / (∆ + r ( b ) + R ) , we have γ ≤ (∆ + 4 R ) / (2∆ + 4 R ) , which implies γ/ (1 -γ ) ≤ 1 + 4 R/ ∆ ≤ π k (Φ + j ) /π k (Φ -j ) . Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining Eqs. (104) and (135) produces

<!-- formula-not-decoded -->

From Eq. (130) we have π k ( a ) ≥ π k (Φ -i ) ≥ π k (Φ j ) , so 1 -π k (Φ j ) ≥ 1 / 2 . Thus Eqs. (105) and (136) together provide

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since b ∈ Φ -i was arbitrary, this concludes the inductive argument. We have shown that, on E , Eq. (124) holds.

Define the stopping time ν by

<!-- formula-not-decoded -->

and define ( Z t ) t ≥ µ by Z t := ∑ b ∈ Φ i max( θ min( t,ν ) ( b ) , 0) . We will show that ( Z t ) t ≥ µ is a supermartingale, i.e. for all t ≥ µ , E t [ Z t +1 -Z t ] ≤ 0 . If t ≥ ν then we have E t [ Z t +1 -Z t ] = 0 , so assume t &lt; ν . Let B := { b ∈ Φ i : θ t ( b ) ≥ ηR } and C := { c ∈ Φ i : θ t ( c ) &lt; ηR } , so

<!-- formula-not-decoded -->

The terms in the sum on the left of Eq. (144) can be bounded by

<!-- formula-not-decoded -->

where the last inequality above comes from Eq. (136) and the fact that ν &gt; t . 4 For the sum on the right of Eq. (144), we can bound the terms by

̸

<!-- formula-not-decoded -->

Combining Eqs. (144), (146) and (149) produces

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

4 Specifically, ν &gt; t implies the inductive hypothesis that was used to prove Eq. (136), and Φ j can be replaced with Φ i .

Eqs. (123) and (138) imply

̸

If π t ( C ) = 0 then the above is negative, so we may assume π t ( C ) &gt; 0 . Since ν &gt; t , there is some b ∈ Φ i with θ t ( b ) ≥ U 2 ≥ ηR , so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also from ν &gt; t , we have that ∑ a ∈A ∗ θ t ( a ) ≥ U 1 , so at least one a ∈ A ∗ satisfies θ t ( a ) ≥ U 1 /K . Fixing such an a gives

<!-- formula-not-decoded -->

We will break into two cases, first assuming that π t (Φ i ) ≤ 1 / 2 . In this case we can upper bound Eq. (151) by

<!-- formula-not-decoded -->

On the other hand, if π t (Φ i ) &gt; 1 / 2 then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Starting once more from the right hand side of Eq. (151), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In concert, Eqs. (164) and (170) together with Eq. (151) imply that E t [ Z t +1 -Z t ] ≤ 0 when µ ≤ t &lt; ν . Therefore ( Z t ) t ≥ µ is a submartingale, and it is clear from its definition that Z t is bounded below by 0 at all times. We can apply Lemma A.7 and conclude that ( Z t ) t ≥ µ converges a.s. to a random variable Z with E [ | Z | ] ≤ ∞ .

We will again break into two cases, first assuming that ν = ∞ , i.e. the stopping time never stops. In this case lim t Z t = lim t ∑ b ∈ Φ i max( θ t ( b ) , 0) , and this quantity will a.s. converge to a finite value; because each summand is nonnegative, this implies that all b ∈ Φ i satisfy lim sup t θ t ( b ) &lt; ∞ . From the assumption that ∀ c ∈ Φ -i : lim t θ t ( c ) = -∞ , we have that, for all b ∈ Φ -i +1 = Φ i ∪ Φ -i , lim sup t θ t ( b ) &lt; ∞ . By Proposition 4.3, there a.s. exists a finite timestep τ such that ∑ a ∈A ∗ θ τ ( a ) ≥ C + C ′ max c ∈ Φ -i +1 lim sup t θ t ( c ) ≥ C + C ′ max c ∈ Φ -i +1 θ τ ( c ) , as desired.

The other case is that ν &lt; ∞ ; this implies either the event E fails to occur (since E implies Eq. (124)), or ∀ b ∈ Φ i : θ ν ( b ) &lt; U 2 . On event E , for all b ∈ Φ -i +1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and setting τ := ν gives the desired result. Therefore, regardless of whether or not ν &lt; ∞ , the only way we don't have the desired result is if E fails to occur, which happens with probability at most ϵ . Since ϵ was arbitrary, it can be taken to 0, and the desired result will hold a.s.

Having shown the above lemma, we are ready to establish that the parameters of suboptimal arms diverge to -∞ .

Proposition 4.4 (Negative Infinite Suboptimal Parameters) . For every suboptimal arm b ∈ [ K ] \ A ∗ , lim t →∞ θ t ( b ) = -∞ a.s.

Proof. Since Φ n = A ∗ and ∪ i ∈ [ n ] Φ i = [ K ] , the set of suboptimal arms is ∪ i ∈ [ n -1] Φ i . Thus we will perform induction over i ∈ [ n -1] , proving that all b ∈ Φ i satisfy lim t θ t ( b ) = -∞ a.s. from the inductive hypothesis that

<!-- formula-not-decoded -->

Note that Eq. (172) is vacuously satisfied for i = 1 . Fix an arbitrary ϵ ∈ (0 , 1] , and define

<!-- formula-not-decoded -->

noting that D &lt; ∞ . Let τ be the first timestep such that

<!-- formula-not-decoded -->

and note that τ is a stopping time. Also, τ &lt; ∞ a.s. by applying Lemma B.3 (which is applicable due to the inductive hypothesis in Eq. (172)) with C := ( K +1) D + K log( K +4 K/ ∆) and C ′ := K .

now we can apply freedman's lemma to both the suboptimal arm and optimal sum, and conclude that the suboptimal arm goes to -∞ wp 1 -δ . since δ was arbitrary the result becomes a.s, and the induction goes through meaning that the whole thing does.

Consider the collection of sequences { ( X b t ) t&gt;τ : b ∈ Φ -i +1 } defined by X b t := θ t ( b ) -θ t -1 ( b ) , and the sequence ( Y t ) t&gt;τ defined by Y t := ∑ a ∈A ∗ θ t ( a ) -θ t -1 ( a ) . For each b ∈ Φ -i +1 , ( X b t ) t&gt;τ satisfies the requirements of Freedman's Inequality (Lemma A.3) with B := ηR ; also, ( Y t ) t&gt;τ does with B := KηR . Therefore we can apply Freedman's Inequality with δ := ϵ/K to all of these sequences simultaneously and take a union bound to conclude that, with probability 1 -ϵ ( | Φ -i +1 | +1) /K ≤ 1 -ϵ ,

<!-- formula-not-decoded -->

for all t &gt; τ . Let E denote the event that both Eqs. (175) and (176) hold at all such t . We will argue that, on E ,

<!-- formula-not-decoded -->

for all t ≥ τ by strong induction. Thus let t ≥ τ , and suppose that Eq. (177) holds with k in place of t , for all τ ≤ k &lt; t , noting that it holds for k = τ by the definition of τ . Eqs. (84) and (91) together imply that

<!-- formula-not-decoded -->

for k &gt; τ , so the assumption that event E holds implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now pick an arbitrary b ∈ Φ -i +1 . Without loss of generality, say b ∈ Φ j , where j ∈ [ i ] . For τ ≤ k &lt; t the inductive hypothesis implies that there exists a ∈ A ∗ such that θ k ( a ) ≥ log( K + 4 K/ ∆) + max b ∈ Φ -i +1 θ k ( b ) . Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Defining the constant γ := (∆ / 2 + r ( b ) + R ) / (∆+ r ( b ) + R ) , we have γ ≤ (∆+4 R ) / (2∆+4 R ) , which implies γ/ (1 -γ ) ≤ 1 + 4 R/ ∆ ≤ π k (Φ + j ) /π k (Φ -j ) . Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining Eqs. (104) and (189) produces

<!-- formula-not-decoded -->

From Eq. (184) we have π k ( a ) ≥ π k (Φ -i +1 ) ≥ π k (Φ j ) , so 1 -π k (Φ j ) ≥ 1 / 2 . Thus Eqs. (105) and (190) together provide

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and multiplying both sides of Eq. (194) by K before adding K log( K +4 K/ ∆) implies the second inequality of Eq. (177) (since b ∈ Φ -i +1 was arbitrary) This concludes the inductive argument over t ≥ τ . We have shown that, on E , Eq. (177) holds. In fact, on event E , we can also use Eqs. (176) and (192) together with the fact that ∑ t ≥ 1 Var t -1 [ X b t ] = ∞ (using Eq. (105)) to conclude that lim t θ t ( b ) = -∞ for an arbitrary b ∈ Φ -i +1 . This finishes off the inductive argument over i ∈ [ n -1] .

The above results are all that is needed for the proof of Theorem 4.2.

Eqs. (176) and (192) imply

## B.2 Non-stationary Convergence

Here we show that, in general, the stochastic gradient bandit algorithm (Algorithm 1) only converges to 'generalized one-hot policies', i.e. ∑ a ∈A ∗ π t ( a ) = 1 , and not 'true one-hot policies', i.e. ∃ a ∈ A ∗ : π t ( a ) = 1 . Among the optimal arms, there will be permanent non-stationary behavior.

Proposition 3.2 (Non-Stationary Convergence) . Consider a K -armed bandit with all arms being equally good, i.e. A ∗ = [ K ] , and at least one arm has a nonzero probability of generating a nonzero reward. Running Algorithm 1 with any η &gt; 0 leads to

<!-- formula-not-decoded -->

for all a ∈ [ K ] . Therefore ( π t ) t ≥ 0 does not converge to any one-hot policy.

̸

Proof. We will first argue that ( θ t ) t ≥ 0 does not converge to a fixed vector. By way of contradiction, suppose lim t θ t ( a ) = φ ∈ R K . Let a ∈ [ K ] be an arm such that P ( r t = 0 | a t = a ) &gt; 0 , which exists by assumption. Without loss of generality, there exist ϵ, δ &gt; 0 such that P ( | r t | &gt; ϵ | a t = a ) &gt; δ . Since θ t → φ , we have that max b ∈ [ K ] | θ t +1 ( b ) -θ t ( b ) | → 0 . Also, convergence implies that lim t π t ( a ) ∈ (0 , 1) , in particular there exists β ∈ (0 , 1) such that eventually π t ( a ) &lt; β . Since a is selected i.o. and every time it is selected | r t +1 | &gt; ϵ with fixed probability δ &gt; 0 , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i.o. This constant lower bound on the step taken contradicts the assumption of convergence.

We will now argue that, for all x ∈ R and a ∈ [ K ] , θ t ( a ) &lt; x occur i.o. or else θ t ( a ) will converge. By conservation of mass this implies the desired result. Since the mean reward is the same across all arms, ( θ t ( a )) t ≥ 0 is a martingale. Given any (possibly random) time τ , the stopped random process induced by running ( θ t ( a ) -θ τ ( a )) t ≥ τ until the time it hits the set [ x -θ τ ( a ) , -∞ ) is thus a martingale bounded below, and Doob's martingale convergence theorem implies it will converge. Either the value converges without hitting the set, in which case θ t ( a ) converges, or else θ t ( a ) dips below x i.o.

## C Nonstationary Bandit Setting

Proposition C.1 (Infinite Optimal Parameters) . If A ∗ = [ K ] then lim t →∞ ∑ a ∈A ∗ θ t ( a ) = ∞ almost surely.

̸

Proof. For t ≥ 0 , let X t := ∑ a ∈A ∗ θ t +1 ( a ) -θ t ( a ) , such that ∑ t i =0 X i = ∑ a ∈A ∗ θ t +1 ( a ) -θ 0 ( a ) . By the update rule of Algorithm 1, note also that

<!-- formula-not-decoded -->

The conditional expectation of X t given F t can be lower bounded by

<!-- formula-not-decoded -->

and the conditional variance can be upper bounded by

<!-- formula-not-decoded -->

Thus for all t ≥ τ we have Var t [ X t ] ≤ η 3 R 2 ∆ -1 E t [ X t ] , | X t | ≤ ηR , and X t is F t +1 -measurable. Setting b := ηR and c := ηR 2 ∆ -1 , we need only to prove that ∑ t ≥ τ E t [ X t ] = ∞ , at which point we can apply the Freedman Divergence Trick (Lemma A.5) to conclude

<!-- formula-not-decoded -->

Thus in the remainder of the proof we turn our attention to showing ∑ t ≥ τ E t [ X t ] = ∞ . Applying Eq. (205) and η ∆ / 3 &gt; 0 , we need only show that

<!-- formula-not-decoded -->

̸

Lemma 4.1 together with ∅ ̸ = A ∗ = [ K ] implies that

<!-- formula-not-decoded -->

Since P ( a t ∈ A ∗ |F t ) = π t ( A ∗ ) and P ( a t / ∈ A ∗ |F t ) = 1 -π t ( A ∗ ) , the Extended Borel-Cantelli Lemma (Lemma A.1) applied to Eq. (95) furnishes ∑ t ≥ τ π t ( A ∗ ) = ∑ t ≥ τ (1 -π t ( A ∗ )) = ∞ a.s. We now break into cases to show that Eq. (215) holds regardless of the behavior of π t ( A ∗ ) .

If π t ( A ∗ ) ≥ 1 / 2 only finitely often then we can set u := max { t ≥ 0 : π t ( A ∗ ) ≥ 1 / 2 } for

<!-- formula-not-decoded -->

Similarly, if π t ( A ∗ ) &lt; 1 / 2 only finitely often then u := max { t ≥ 0 : π t ( A ∗ ) &lt; 1 / 2 } gives us

<!-- formula-not-decoded -->

We can narrow our focus to the case where π t ( A ∗ ) is both above and below 1 / 2 i.o. In particular, there must be infinitely many t ≥ 0 such that π t ( A ∗ ) &lt; 1 / 2 but π θ t +1 ( A ∗ ) ≥ 1 / 2 , and for such t we have

<!-- formula-not-decoded -->

The above equation is of the form x/ ( x + y ) , where x := ∑ a ∈A ∗ exp( θ t +1 ( a )) and y := ∑ b ∈ [ K ] \A ∗ exp( θ t +1 ( b )) . Since x/ ( x + y ) is increasing in x and decreasing in y for x, y &gt; 0 , and | θ t +1 ( c ) -θ t ( c ) | ≤ ηR for all c ∈ [ K ] , we can maximize the right hand side for the upper bound

<!-- formula-not-decoded -->

Also, π t ( A ∗ ) &lt; 1 / 2 yields ∑ a ∈A ∗ exp( θ t ( a )) &lt; ∑ b ∈ [ K ] \A ∗ exp( θ t ( b )) , so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Connecting the above displays, there are infinitely many t ≥ 0 with π t ( A ∗ ) &lt; 1 / 2 and π θ t +1 ( A ∗ ) ≥ 1 / 2 , and for such t we have 1 -π θ t +1 ( A ∗ ) &gt; 1 -exp(2 ηR ) / (exp(2 ηR ) + 1) = (exp(2 ηR ) + 1) -1 . Therefore π θ t +1 ( A ∗ )(1 -π θ t +1 ( A ∗ )) ≥ (2 exp(2 ηR ) + 2) -1 i.o, establishing Eq. (215).

Proposition C.2 (Finite Suboptimal Parameters) . For every suboptimal arm b ∈ [ K ] \ A ∗ , lim t →∞ θ t ( b ) = -∞ a.s.

Remark C.3 . The proof remains virtually unchanged from the proof of Proposition 4.4, and the necessary changes are identical to the ones made for the proof of Proposition C.1.

Theorem C.4. In the non-stationary bandit setting described as above, Algorithm 1 with any η ∈ Θ(1) almost surely converges to playing optimal arms,

<!-- formula-not-decoded -->

## D Reinforcement Learning

Define the MDP M = ( H , S , A , { r h } H -1 h =0 , { P h } H -1 h =0 , ρ ) . Let N t ( s, a ) := ∑ t ≥ 0 I { s t = s, a t = a } be the total number of visitations of state-action pair ( s, a ) until episode t . We denote that P h +1 t ( s h +1 = s ′ | s h = s ) as the probability of visiting state s ′ in the horizon h +1 from the state s in the horizon h during the episode t . First, we extend the bandit exploration lemma (Lemma 4.1) to obtain its counterpart in the RL setting.

Lemma D.1 (RL exploration (Lemma 5.1)) . Using the REINFORCE algorithm with any η ∈ Θ(1) under the finite-horizon MDP M defined as above, for all h ∈ H , for all reachable s ∈ S h and for all a ∈ A s we have, almost surely, that every reachable state action pair will be visited i.o, i.e N ∞ ( s, a ) = ∞ .

Proof. First, for all h ∈ H , for a given reachable s ∈ S h that is played infinite often, every action a ∈ A s will be played i.o. by the bandit exploration result (Lemma 4.1) . In other words, for all h ∈ H , for a reachable state s that is played i.o, we have, almost surely that,

<!-- formula-not-decoded -->

Next, for all h ∈ H , we want to show that every reachable state s ∈ S h will be visited i.o. by induction. Suppose for a given h ∈ H , for some reachable s ∈ S h and there exists an action a ∈ A s such that P h +1 ( s h +1 = s ′ | s h = s, a h = a ) &gt; 0 for some s ′ ∈ S h +1 , if s is visited i.o, s ′ is also visited i.o. For the base case h = 0 , for some reachable states s ∈ S 0 , i.e ρ ( s ) &gt; 0 we have,

<!-- formula-not-decoded -->

since ρ ( s ) is a constant for every episode. Therefore, every reachable state s ∈ S 0 is visited i.o. For the inductive case, if any reachable states s ∈ S h is visited i.o, then any reachable states s ′ ∈ S h +1 is also visited i.o. A state s ′ is reachable if there exists an action a ∈ A s from a reachable state s ∈ S h such that P h +1 ( s ′ | s, a ) &gt; 0 . Denote c := min s ∈S h min a ∈A s P h +1 ( s ′ | s, a ) be the minimum transition probability from the horizon h to h +1 among states and actions. For reachable s ′ from s , we have .

<!-- formula-not-decoded -->

Therefore, if s ∈ S h is reachable and visited i.o, then any reachable states s ′ ∈ S h +1 from s will be visited i.o. Combined Eq. (226) and Eq. (228), for all h ∈ H , we have that any reachable state-action ( s, a ) ∈ S h ×A h pairs will be visited i.o. we know that every state-action pair will be visited i.o.

Next, we obtain the convergence of REINFORCE in the finite-horizon setting.

Theorem D.2 (RL convergence (Theorem 5.2)) . For the MDP defined as above, using the algorithm REINFORCE with constant learning rate η ∈ Θ(1) , we have, almost surely, for all s ∈ S 0 , V π t 0 ( s ) → V ∗ 0 ( s ) as t →∞ .

̸

Proof. We denote δ := min s min a,b ∈A s ,a = b | Q ( s, a ) -Q ( s, b ) | &gt; 0 to be the minimum non-zero gap between Q -values. Denote A ∗ h = { a | a = arg max a ∈A s r ( s, a ) } is the set of optimal action at a given state s . We also denote C := max s max a min b ( Q ( s, a ) -Q ( s, b )) . We want to prove by backward induction that for all reachable state s 0 ∈ S 0 , ∑ a ∈A ∗ 0 π 0 t ( a | s ) → 1 as t →∞ . Suppose for all h ′ ∈ { h, . . . , H -1 } , for all reachable s ∈ S h ′ , we have ∑ a ∈A ∗ h ′ π h ′ t ( a | s ) → 1 as t →∞ , we want to prove that for all reachable s ∈ S h -1 , we have ∑ a ∈A ∗ h -1 π h -1 t ( a | s ) → 1 as t →∞ . . In the base case h = H -1 , the REINFORCE update rule (Algorithm 2) is reduced to,

<!-- formula-not-decoded -->

This is the bandit update rule (Algorithm 1) for a given reachable state s ∈ S H -1 . By Theorem 4.2, for a given reachable state s ∈ S H -1 , using the stochastic gradient bandit algorithm with constant learning rate η ∈ Θ(1) , we will have, almost surely that ∑ a ∗ ∈A ∗ h π H -1 t ( a ∗ | s ) → 1 as t → ∞ . By Lemma D.1 , any reachable states s ∈ S H -1 will be sampled i.o. Hence, using REINFORCE with η ∈ Θ(1) , that for all reachable s ∈ S H -1 that are played i.o, we have, almost surely, ∑ a ∈A ∗ H -1 π H -1 t ( a | s ) → 1 as t →∞ . In other words, V π t H -1 ( s ) → V ∗ H -1 ( s ) as t →∞ .

For inductive case, suppose for all h ′ ∈ { h, . . . , H -1 } , for all reachable s ∈ S h ′ , we have ∑ a ∈A ∗ h ′ π h ′ t ( a | s ) → 1 as t → ∞ , we want to prove that for all reachable s ∈ S h -1 , we have ∑ a ∈A ∗ h -1 π h -1 t ( a | s ) → 1 as t →∞ . By the induction hypothesis, for all h ∈ H , for all reachable s ∈ S h , V π t h ( s ) → V ∗ h ( s ) and Q π t h ( s, a ) → Q ∗ h ( s, a ) for all a ∈ A s as t →∞ . First, we note that

<!-- formula-not-decoded -->

We denote that η h ′ ( t ) := ∑ a ′ ̸∈A ∗ h ′ π h ′ t ( a ′ | s ) . For the first term C 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since the horizon H is fixed and r h ≤ R for all h ∈ H , then

<!-- formula-not-decoded -->

By the induction hypothesis, we have, for all h ′ ∈ { h, . . . , H -1 } , we have that γ h ′ ( t ) → 0 as t →∞ .

For the second term C 2 , we have Q ∗ h ′ ( s, a ′ ) -Q π t h ′ ( s, a ′ ) ≤ α h ′ ( t ) , where α h ′ ( t ) → 0 as t →∞ by induction hypothesis. Therefore,

<!-- formula-not-decoded -->

Denote ϵ h ( t ) := Cγ h ′ ( t ) + α h ′ ( t ) and ϵ h ( t ) → 0 as t →∞ . Hence, for sufficiently large timestep τ , such that for all t ≥ τ , for all h ′ = h, . . . , H -1 , for all reachable s ∈ S h ′ , we have that

<!-- formula-not-decoded -->

where δ is the minimum possible gap Q-value defined as above. The existence of τ is guaranteed since ϵ h ( t ) → 0 as t →∞ . First, for a given reachable s ∈ S h -1 , such that for any actions a ∈ A s , we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also, for any actions a ∈ A s , we have,

<!-- formula-not-decoded -->

By definition of δ , for a given reachable s ∈ S h -1 , we know that Q ∗ h -1 ( s, a ) -Q ∗ h -1 ( s, b ) ≥ δ for any a, b ∈ A s such that a = b , then we have

<!-- formula-not-decoded -->

̸

Note that δ is a non-zero gap by definition. Note that the update rule of Algorithm 2,

<!-- formula-not-decoded -->

is equivalent to the update rule in the nonstationary bandit setting, by considering only the updates to the arms at state s and the full trajectory's rewards as the observed rewards.

Since by definition

Therefore, we have

<!-- formula-not-decoded -->

∑ H -1 h ′ = h -1 r h ′ is an unbiased estimate of Q π t h -1 ( s h -1 , a h -1 ) , up to nonstationarity in π t which eventually diminishes below δ/ 3 . Also, note that

<!-- formula-not-decoded -->

since r ( s, a ) ≤ R, ∀ s ∈ S , a ∈ A . Since the sample return ∑ H -1 h ′ = h -1 r h ′ is a bounded, unbiased estimator of Q π t h -1 ( s h -1 , a h -1 ) , and there is a minimum gap of δ/ 3 between Q values among different actions within the same state, we can apply the convergence result from the nonstationary bandit setting (Theorem C.4) to conclude that ∑ a ∈A ∗ h -1 ( s ) π h -1 t ( a | s ) → 1 as t → ∞ for all reachable s ∈ S h -1 . Therefore the induction hypothesis holds, and we conclude that, using REINFORCE with η ∈ Θ(1) , ∑ a ∈A ∗ h π h t ( a | s ) → 1 as t → ∞ for all s ∈ S h (or V π t 0 ( s ) → V ∗ 0 ( s ) as t → ∞ for all s ∈ S 0 ).

## E Convergence rate

To obtain the convergence rate of the REINFORCE algorithm (Algorithm 2), we first generalize the convergence rate result from the bandit setting with the uniqueness assumption to the one without it. Then, we also obtain the convergence rate in the non-stationary bandit setting before showing the rate of the REINFORCE algorithm.

Theorem E.1. In the bandit setting where multiple arms can have a same reward, for a large enough τ , for all T &gt; τ , the average sub-optimality decreases at a rate O ( log T T ) . Formally, for a constant c , we have

<!-- formula-not-decoded -->

Proof. By Eq. (84), we have

<!-- formula-not-decoded -->

By Theorem 4.2, we have lim t π θ t ( A ∗ ) = 1 a.s. Therefore, for a large enough t , we have

<!-- formula-not-decoded -->

By Lemma 4.1, we know that every action a ∈ [ K ] will be played i.o. In other words, for all a ∈ [ K ] , ∑ t ≥ 0 π t ( a ) = ∞ . Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Eq. (91), we have

<!-- formula-not-decoded -->

Since the conditional expecation and variance of the bound sequence { θ t +1 ( A ∗ ) -θ t ( A ∗ ) } t ≥ 0 are proportional, we can use the Lemma A.5 to show that the expectation will dominate the variance eventually. Therefore, for all large enough t ≥ τ , for some constant C &gt; 0

<!-- formula-not-decoded -->

It is easy to see that sup t θ t ( a ) &lt; ∞ for all a ∈ [ K ] \ A ∗ . Therefore, for a large enough t ≥ τ , we have which implies that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (Mei et al., 2024a, Lemma 15) with x n = ∑ t -1 s = τ (1 -π θ s ( A ∗ )) &gt; 0 , x n +1 = ∑ t s = τ (1 -π θ s ( A ∗ )) &gt; 0 , c = C &gt; 0 , B = ( K -|A ∗ | ) ≥ 1 , gives us for all t ≥ τ ,

<!-- formula-not-decoded -->

where M = max { B, 1 C log( C ( K -|A ∗ | )) , 1 -π θ τ ( A ∗ ) } Finally, for all s ≥ τ and T &gt; τ , we have

<!-- formula-not-decoded -->

Summing from τ to T , we have

<!-- formula-not-decoded -->

Theorem E.2. In the non-stationary bandit setting, for a large enough τ , then for all T &gt; τ , the average sub-optimality decreases at a rate O ( log T T ) . Formally, for a constant c , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Repeating the same analysis with Eq. (205), Eq. (212), Theorem C.4, we have, for all t ≥ τ ′′ ,

<!-- formula-not-decoded -->

where M = max { B, 1 C log( C ( K -|A ∗ | )) , 1 -π θ τ ′′ ( A ∗ ) } . Also, from the non-stationary bandit setting, there exists τ ′ such that for all t ≥ τ ′ ,

<!-- formula-not-decoded -->

for all a ∈ [ K ] . Therefore, for all s ≥ max { τ ′ , τ ′′ } and T &gt; max { τ ′ , τ ′′ } , we have

<!-- formula-not-decoded -->

Summing from τ := max { τ ′ , τ ′′ } to T, we have

<!-- formula-not-decoded -->

Theorem E.3. In the finite-horizon MDP setting, for a large enough τ , for all T &gt; τ , for all s ∈ S 0 , the average sub-optimality decreases at a rate O ( log T T ) . Formally, for a constant c , we have

<!-- formula-not-decoded -->

Proof. Repeating the same analysis, we have, for each h ∈ { 0 , ..., H -1 } , for all s ∈ S h , for all t ≥ τ h ,

<!-- formula-not-decoded -->

where M h = max {|A s | - |A ∗ s | , 1 C log( C ( |A s | - |A ∗ s | )) , 1 -π θ τ h ( A ∗ s | s ) } . Also, there exists τ ′ h such that for all t ≥ τ ′ h ,

<!-- formula-not-decoded -->

for all a ∈ A s . Therefore, for all horizon h ∈ { 0 , ..., H -1 } , for all t ≥ max { τ h , τ ′ h } and T &gt; max { τ h , τ ′ h } , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Since for all h ∈ { 0 , ..., H -1 } , for all s ∈ S h , for all a ∈ A s , lim t Q π t h ( s, a ) = Q ∗ h ( s, a ) , we can take δ → 0 . Therefore,

<!-- formula-not-decoded -->

Figure 5: Average last-iterate suboptimality gap of ChainMDP size 4

<!-- image -->

## F Additional experiments

Specifically, we measured the average suboptimality in the last episodes over 30 of the algorithm in longer ChainMDP, DeepSea environment and CartPole environment. For ChainMDP, we extended the lengths of the environment to H = { 4 , 5 , 6 } and measured the average suboptimality gap across 100 learning rates (from exp( -9) to exp(1) ). For each length, we observed a clear bowl-shaped curve. As complexity (chain length) increased, the specific thresholds of the bowl shape varied slightly, but the optimal learning rate remained consistently around η ≈ . 95 . Next, we gradually increase the complexity of our evaluation by testing the REINFORCE algorithm (Algorithm 2) on the deep sea treasure environment. The agent operates in a square gridworld of a given depth d = { 5 , 6 , 7 } ). It starts at the top left corner and its goal is to reach the bottom right corner and receive a reward of 1 . The agent has two action 1 and 2. While taking action 1 leads the agent downwards and receives no reward, taking the other leads the agent downwards and to the right and receives a reward of -0 . 001 . Similar to the previous environment, for different depths, we measure the average suboptimality of the agent trained from 10 6 episodes over 30 seeds using 100 different learning rates from exp( -9) to exp(7) . We observed a similar "bowl" shape across the learning rates. However, the thresholds are different from the previous analysis. Specifically, the learning rate η = 10 has the lowest suboptimality. Finally, we evaluate the performance of the REINFORCE algorithm (Algorithm 2) in the Cartpole environment. Specifically, we measure the average return received by the agent from 10 5 episodes over 5 seeds using η = { 10 -5 , 10 -4 , 10 -2 , 1 } . Again, we observed a similar "bowl" shape across learning rates. The learning rate η = 0 . 01 achieves the highest average return (approximately 150 ), while the average return of the others stay around 25 . Overall, we consistently find a "bowl-shaped" relationship between the learning rate and performance, and the specific shape and optimal point of this bowl vary significantly with the environment's structure.

<!-- image -->

Learning rates (n)

Figure 6: Average last-iterate suboptimality gap of ChainMDP size 5

<!-- image -->

Learning rates (n)

Figure 7: Average last-iterate suboptimality gap of ChainMDP size 6

Figure 8: Average last-iterate suboptimality gap of DeepSea depth 5

<!-- image -->

<!-- image -->

Learning rates (n)

Figure 9: Average last-iterate suboptimality gap of DeepSea depth 6

<!-- image -->

Figure 10: Average last-iterate suboptimality gap of DeepSea depth 7

<!-- image -->

Episodes

Figure 11: Average suboptimality gap of CartPole

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction are shown in Lemma 4.1, Theorem 4.2, and Theorem 5.2.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the conclusion (Section 6) we point out that REINFORCE exhibits high variance when using excessively large learning rates and highlight choosing an optimal learning rate schedule as potential future work. The specific technical limitations of all results are apparent from the formal descriptions of the algorithms, settings, and results. In particular we state all assumptions explicitly in the settings, including that the state and action sets are finite and the rewards are bounded.

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

Justification: We numbered each theorem, formula, and proof in the paper; all results are proved formally in the supplemental material. We have provided intuitive proof sketches where possible.

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

Justification: We described the experiments in detail in Section 5.5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: Our experiments are simple enough that they can be recreated easily using the information provided in Section 5.5. We are studying classic algorithms and use only simulated data, which should be easily reproducible from the detailed descriptions.

## Guidelines:

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

Justification: We do specify the list of values for each hyperparameter (learning rate, seed, horizon, training episodes).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All figure contain error bars except for figures that are explicitly stated to plot individual runs for demonstrative purposes.

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

Justification: The environments (ChainMDP and TreeMDP) are simple and the experiments are small enough to run on any modern machine (e.g. an Intel Macbook 2019).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We make sure to preserve anonymity, no new datasets are introduced, and we do not conduct research on human subjects.

## Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The focus on the paper is on theoretical guarantees of convergence for classic algorithms, and empirical validation of the proofs. There are no specific societal impact concerns beyond those of the field of RL as a whole.

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

Justification: The work does not involve releasing high-risk artifacts such as pretrained models, large-scale datasets, or generative tools that could be misused.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use any existing assets.

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

Justification: We do not introduce new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our research does not require crowdsourcing or involve human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our research does not require crowdsourcing or involve human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
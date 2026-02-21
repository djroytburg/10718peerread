## Stackelberg Learning with Outcome-based Payment

## Tom Yan

Carnegie Mellon University tyyan@cmu.edu

## Chicheng Zhang

University of Arizona chichengz@cs.arizona.edu

## Abstract

With businesses starting to deploy agents to act on their behalf, an emerging challenge that businesses have to contend with is how to incentivize other agents with differing interests to work alongside its own agent. In present day commerce, payment is a common way that different parties use to economically align their interests. In this paper, we study how one could analogously learn such payment schemes for aligning agents in the decentralized multi-agent setting. We model this problem as a Stackelberg Markov game, in which the leader can commit to a policy and also designate a set of outcome-based payments. We are interested in answering the question: when do efficient learning algorithms exist? To this end, we characterize the computational and statistical complexity of planning and learning in general-sum and cooperative games. In general-sum games, we find that planning is computationally intractable. In cooperative games, we show that learning can be statistically hard without payment and efficient with payment, showing that payment is necessary for learning even with aligned rewards. Altogether, our work aims to consolidate our theoretical understanding of outcome-based payment algorithms that can economically align decentralized agents.

## 1 Introduction

Increasingly, we are seeing businesses deploying agents to carry out tasks on their behalf. In the coming agentic era, we will inevitably have multiple, decentralized agents interacting together. An emerging challenge that businesses may have to face is how to incentivize other agents to work alongside its agent. This challenge requires addressing a central difficulty in decentralized multi-agent systems, which is that of differing interests.

In present day commerce, payment is a standard way that two parties use to resolve this challenge and more closely align their business interests. This inspires us to study the overarching question in this paper: how can we analogously implement such payment schemes in the multi-agent setting and enable economic alignment ? That is, if I am a business looking to use payment to incentivize another business (and/or its agent) to work with my agent, how can I learn a good policy for my agent along with a payment scheme to go with it?

On a technical level, this setting may be viewed as a Stackelberg Markov game. In this paper, we study the two-player Stackelberg game, where one player (leader) commits to a policy taking into account the best response to the policy by the other player (follower). We focus on Stackelberg Markov games in particular as agents will be interacting over multiple turns and potentially long horizons. Finally, to model the payment aspect, the leader is able to also increase the reward of the follower in the Markov game, which may be viewed as a form of reward shaping in line with the existing formulation in the literature [Ben-Porat et al., 2024, Bollini et al., 2024, Ivanov et al., 2024, Scheid et al., 2024, Wu et al., 2024].

In this work, we aim to consolidate the theoretical foundations of Stackelberg learning with payment, as complexity results have yet to be established for two-player Stackelberg Markov games. We focus

Table 1: Planning &amp; learning settings where computationally and statistically efficient algorithms exist.

|                         | Without Payment                               | With Payment                                      | With Payment                                     |
|-------------------------|-----------------------------------------------|---------------------------------------------------|--------------------------------------------------|
| Planning, Learning      | DAG                                           | Tree                                              | DAG                                              |
| Cooperative General Sum | /check , (Theorem 5.15.2) , (Proposition 4.1) | /check , /check /check , /check (Proposition 4.3) | /check , /check (Theorem 6.46.5) , (Theorem 4.2) |

on a fundamental question: is there an efficient algorithm that can provably compute or learn the optimal policy and payment? Indeed, this is an important question to address as businesses in the future would want payment schemes with provable guarantees , so as to ensure that their expenditure is optimal.

Contributions: We analyze the planning and learning setting through both the computational and statistical lens. Please see Table 1 for an overview of our results.

1. We begin by considering planning in general-sum games. Is there an efficient algorithm that can return the optimal policy and payment? We prove that such a computationally efficient algorithm cannot exist unless NP=P, and identify the structural property of the MDP that results in this hardness. To complement the negative results, we develop an efficient algorithm, applicable when this property is removed.
2. Next, we turn to Cooperative games, which is a broad subclass of Markov games useful for modeling e.g. the interaction between AI service-providers and their users. Moreover, planning is computationally efficient in this setting, making it plausible that efficient learning algorithms may be attainable. As the rewards are already aligned, we begin by considering learning in the Stackelberg game without payment. Surprisingly, however, we find that an efficient algorithm cannot exist, this time in the statistical sense. We identify structural properties of the MDP that result in statistical hardness, and develop an efficient algorithm for when such properties are removed to complement our negative results.
3. Finally, we study learning in Cooperative games with payment. Can payment be used to alleviate the statistical hardness of learning? We answer this in the affirmative by showing that we can adapt existing no-regret RL algorithms to enable sample-efficient learning. In closing, we also use this setting to contrast the two different payment settings we study. We derive matching upper and lower regret bounds for when the leader has to make payments upfront versus on-the-fly, allowing us to quantitatively assess the benefits of being able to make payments on-the-fly.

## 2 Formulation

## 2.1 Stackelberg Markov Game

We consider the standard two-player, episodic finite-horizon Markov game M parameterized by 〈 S, A, B, H, P, ρ , r L , r F 〉 with state space S , initial state distribution s 0 ∼ ρ , transitions P and episode length H . The leader has action set A and reward r L ∈ [ -1 , 1] , the follower action set B and reward r F ∈ [ -1 , 1] . In the case that the game is cooperative, r L = r F .

In the problem of online learning for Stackelberg Markov games, the learner plays the role of the leader, where apriori the reward functions r L , r F and the transitions are unknown to the leader. At each episode k ∈ [ T ] , the leader commits first to a policy π k . The follower best responds to π k with µ ( π k ) ∈ argmax µ V π k ,µ ( s 0 ; r F ) . One may view best response as the equilibrium behavior of the follower to the leader policy.

After the episode, the leader and the follower observe the resultant trajectory τ k = { ( s i , a i , b i , r L ( s i , a i , b i ) , r F ( s i , a i , b i ) } H i =1 realized by the chosen policies in M : a i ∼ π k ( s i ) , b i ∼ µ ( π k )( s i ) , s i +1 ∼ P i ( ·| s i , a i , b i ) .

This trajectory is the outcome of the policies' interaction, which in turn determines the outcome-based payment the follower receives.

Leader Payment: Following existing formulations in prior literature, the leader can increase r F by creating outcome-based payment b k i ( s i , a i , b i ) , if state-actions s i , a i , b i are realized during the episode, s i , a i , b i ∈ τ k . This results in a modified Markov game where the leader is able to additionally assign payment, with the payment function having signature b k i : S × A × B → R + .

We note that the outcome-based payment need not correspond to direct monetary transfer. For example, we may be interested in modeling the setting where the leader is an AI-service-provider and the follower is a customer user. The leader spends money to improve its agent, and this improved agent adds additional value (e.g. more saved time) for the user during its use. But during this interaction, there is no direct transfer of money from the company to the user.

Thus, to model indirect payments in addition to direct ones, we introduce a final piece of notation, multiplier κ ∈ R + . κ · b k i ( s i , a i , b i ) corresponds to the proportional cost to the leader in creating payment (reward) b k i ( s i , a i , b i ) for the follower. We believe proportionality is a natural assumption to make, and verily κ = 1 corresponds to direct payment.

## 2.2 Payment Settings

To complete the formulation, we touch on the two types of payment settings considered in this paper.

Trajectory Payment: The first is the existing payment setting commonly studied in prior literature, which we term trajectory payment. Here, a payment is made by the leader for every state-action on the realized trajectory. This form of payment is considered in principal-agent contracting literature, where the trajectory informing how much the leader will be paying ex-post [Dutting et al., 2021].

Moreover, this form of payment corresponds to the trendy outcome-based pricing model, which is experiencing rapid adoption by several notable SaaS companies due to the rising usage of AI agents [Stripe, 2025, Intercom, 2025, Zendesk, 2025]. Indeed, this marks a fundamental paradigm shift in software pricing in industry, moving from seat-based subscriptions (traditional SaaS) and usage-based models (cloud infrastructure) to now outcome-based pricing in the agent era [Boston Consulting Group, 2025, Sequoia Capital, 2025]. This also makes it imperative then to bolster our theoretical understanding of outcome based pricing, which we study in this paper.

Upfront Payment: In this paper, we will also consider a setting that we term upfront payment. As the name suggests, the leader pays for every state-action in the MDP, regardless of the realized trajectory. Note that the follower is still paid based on the realized trajectory. This is more realistic in settings where the leader pays indirectly to the benefit of the follower, and is bound by temporal constraints such that the payment cannot be made on-the-fly.

For a motivating example, consider the AI-service provider setting discussed earlier. The company invests before deployment to improve the agent's functionality, which means that the user (follower) gains added value (reward) on the trajectory realized during the agent's use. However, the key temporal constraint is that the company cannot improve its agent on-the-fly, as the users are using it. Thus, this makes upfront payment a more realistic model of the leader's expenditure. The leader had to invest upfront to improve the agent's capabilities in all states, even though this includes off-trajectory states that are not visited during the interaction with the user. For instance, suppose the agent is a computer-using-agent [Anthropic, 2024]. The user may use it to handle emails, and the agent would act in states of the computer corresponding to the inbox. However, even though the company had also invested to improve the agent's capabilities in coding, the user may not invoke the agent to do so (perhaps due to excessive risk). And so, the agent would not have acted in other states of the computer corresponding to the codebase.

More generally, there is sizable body of economics contracting literature studying settings where only ex-ante (upfront) payment is possible. Some reasons for this include non-enforceable contracts, where the principal can renege upon observing the outcome Hart and Moore [1988]. Another cause for this may be non-verifiable outcomes; that is, when outcomes cannot be verified, ex-post contracts become unenforceable as there is no way to condition legally binding payments Aghion and Holden [2011]. Finally, one other reason may simply be that the agent is risk-averse, thus preferring upfront payment in face of stochastic outcomes Laffont and Martimort [2002].

Leader Optimization: Putting it all together, we can now write down the resulting Stackelberg game under the two payment settings.

Definition 2.1. In Stackelberg Markov games with trajectory payment, the leader optimizes:

<!-- formula-not-decoded -->

In Stackelberg Markov games with upfront payment, the leader optimizes:

<!-- formula-not-decoded -->

Before moving on, we highlight the generality of the class of games we are studying. The class of Stackelberg Markov games with payment generalizes Stackelberg Markov games. Indeed, constraining the leader to zero payment (i.e. ( π , b ) = ( π , 0) ) corresponds to the leader's policy space in Stackelberg Markov Games. Analogously, for Cooperative Stackelberg Markov games with payment studied in the later sections, this class of games generalizes Cooperative Stackelberg Markov games.

## 3 Related Works

As we focus on Stackelberg Markov games with payment, our paper is most related to two lines of work. The first is the line of work studying the complexity of Stackelberg policy computation in Markov games. And the second is algorithms for computing optimal payment schemes in MDPs. We cover both lines of work below, and include a discussion on additional related works in Appendix E.

Stackelberg Optimal Policies in Markov Games without Payment: Due to the wide applicability of the Stackelberg Markov games, there has been a long line of work seeking to understand how to compute optimal leader policies with provable guarantees.

For planning, Conitzer and Sandholm [2006], Letchford et al. [2012], Letchford and Conitzer [2010] study the computational tractability of optimal Stackelberg policy computation in Markov games and subclasses thereof. For stochastic MDPs, they establish that computing the optimal Stackelberg policy is NP-Hard.

For learning, Zhao et al. [2023] studies the statistical complexity in cooperative bandit games. Bai et al. [2021] studies the statistical complexity in bandit-RL games, a particular subclass of Markov games. Our work differs from this line of work in focusing on Markov games, which are more general than bandit-RL games and have longer horizon than bandit settings. Moreover, the leader is allowed to use payments to shape the follower's rewards. As we will see, this turns out to be crucial for improved exploration during learning in certain settings.

Learning the Optimal Payment Scheme in MDPs: Recently, there has been burgeoning interest in computing optimal payment schemes for contracting agents to act in MDP environments, wherein the leader may increase the follower's rewards as a form of reward shaping to incentivize the follower to play policies desirable to the leader.

The single-agent MDP setting, where only the follower acts in the MDP and the leader incentivizes, is formulated by Ben-Porat et al. [2024], Chen et al. [2022]. This is followed by a series of interesting work by Bollini et al. [2024], Ivanov et al. [2024], Wu et al. [2024], studying learning under a variety of different payment functions taking as input the state, the state-action or the state-next-state. Our work adds to this line of work by focusing on two-player Markov games, which generalize the single-player setting. Furthermore, while previous works mostly focus on trajectory payment, we also consider upfront payment, applicable in settings where the leader cannot pay on the fly due to temporal constraints. We derive tight regret guarantees to contrast the two differing payment settings.

The paper closest in formulation to that of ours is that by Scheid et al. [2024], who considers the same state-action based payment function in the bandit setting. Our work differs in focusing on Markov games, with a longer horizon than that in bandit settings. This in turn introduces difficulty in terms of exploration, and requires a more nuanced optimal payment computation beyond the binary search approach used in [Scheid et al., 2024].

Finally, as payment may be viewed as strategic reward shaping, our analysis is also related to existing RL literature that seeks to theoretically quantify the benefits of reward shaping [Ng et al., 1999]. Gupta et al. [2022] quantifies how statistical sample complexity is improved by reward shaping in the single-agent setting. By contrast, in our work, we study improved sample complexity in two-player cooperative Stackelberg Markov games.

## 4 Planning in General-sum Games

In this section, we ask: is there an efficient algorithm that can compute the optimal policy and payment in general-sum games? We investigate the computational complexity of such an algorithm, starting with the planning setting, where the Markov game dynamics and reward functions are known.

Our main finding is that there is no such computationally efficient algorithm unless NP=P. Outcomebased payment does not alleviate the computational intractability of computing the optimal Stackelberg policy, even in planning [Conitzer and Sandholm, 2006]. We identify that when the MDP has DAG structure, this leads to computational intractability. Later in the section, we complement this negative result with a positive result for when the MDP has tree structure. All proofs in this section may be found in Appendix A.

## 4.1 Hardness Results

We first derive a result showing that it is NP-Hard to compute the optimal leader policy even in deterministic MDPs, without payment. Note that in [Conitzer and Sandholm, 2006], computational intractability is demonstrated in stochastic MDPs.

Proposition 4.1. Under Markov games that are deterministic DAGs, it is NP-Hard to compute the optimal policy:

<!-- formula-not-decoded -->

Helpfully, deterministic MDPs allow us to provide guarantees for both two payment settings. As we show in the proof, the optimal payment scheme pays zero in off-policy states, which can be readily characterized in deterministic MDPs. This result is intuitive as paying in off-policy states only incentivizes the follower to deviate off-policy, which is undesirable and increases leader total payment. With this result, we can derive that the optimal payment scheme is the same under trajectory and upfront payment. Thus, we use same construction, which provides a reduction to the PARTITION problem, to prove computational intractability under both payment settings.

Theorem 4.2. Under Markov games that are deterministic DAGs, it is NP-Hard to compute the optimal policy and optimal trajectory payment:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and it is also NP-Hard to compute the optimal policy and optimal upfront payment:

<!-- formula-not-decoded -->

In closing, we note that the optimal objective value of the subset of Markov games used to reduce to the PARTITION problem is an integral multiple of 1 / 2 . Due to this, we have that computational intractability in planning implies computational intractability in learning. In more detail, let M ∗ be the optimal objective value, which is an integer multiple of 1 / 2 . Suppose by contradiction that we had an algorithm with sublinear regret T α ( α &lt; 1 ). We can then set T large enough such that T α /T &lt; 1 / 2 . This allows us to infer M ∗ exactly by rounding to the nearest 1 / 2 , giving us a computationally efficient algorithm for answering the decision version of the PARTITION problem, which is a contradiction.

Algorithm 1 Planning Algorithm for MDP with Deterministic Tree Structure

Require: Pre-computed policy π -∈ argmin π V π ,µ ( π ) ( s 0 ; r F ) (efficiently computed via Nash-VI) for all root to leaf paths τ = s 1 , a 1 , b 1 , s 2 , a 2 , b 2 , ..., s H , a H , b H do Define π ( s i ) = a i for s i , a i ∈ τ . In every other state s ′ i /negationslash∈ τ , let π ( s ′ i ) = π -( s ′ i ) . Compute µ ( π ) and compute follower Q-values, Q π ,µ ( π ) ( · , · , · ) . Solve for the minimal payment scheme using LP:

/negationslash

<!-- formula-not-decoded -->

## end for

Output the leader policy π and payment scheme of the path τ with maximal return ∑ s i ,a i ,b i ∈ τ r L ( s i , a i , b i ) -κ · b τ ( π ) .

## 4.2 Positive Results

To complement our negative results, we show that positive results are attainable in MDPs without DAG structure. That is, in general-sum games where the MDP has tree structure, there is a polynomial-time algorithm for learning the optimal leader policy and payment. We describe our planning algorithm, Algorithm 1 that forms the crux of our approach to learning in this setting, and is applicable under both trajectory and upfront payment.

Proposition 4.3. Under Markov games that are deterministic trees, there exists a polynomial-time planning algorithm that computes the optimal policy and payment.

Remark 4.4. To complete the result, we note in Appendix A that there is a simple exploration strategy using payment for general-sum, deterministic trees, as exploration needs to only recover rewards. This strategy allows us to reduce learning to planning, and then apply Algorithm 1.

Before moving on, we note that in this general-sum game, the leader behaves in a zero-sum like manner in off-policy states in Algorithm 1. This incentivizes the follower to take the desired policy and allows the leader to minimize the total payment needed to incentivize such policy.

Finally, due to the intractability of computing a global Stackelberg optimum, it is natural to consider computing a local Stackelberg optimum instead, so that the policy and payment scheme does attain some guarantees. Building on existing results on first order methods in Stackelberg games [Shen et al., 2024], we derive a first order approach to this end. Note that while our paper is concerned with global Stackelberg optimality guarantees, we use this to illustrate that a more relaxed solution concept can be computed, if desired.

## 5 Learning in Cooperative Games without Payment

The computational intractability in the general-sum case prompts us to investigate whether efficient algorithms are attainable in significant subclasses of Markov games. Cooperative games are a broad subclass of Markov games useful for modeling e.g. the aforementioned AI-service based setting. Indeed, since the goal of the assistant agent is to aid the user, their rewards are aligned. And so, such settings correspond to a two-player cooperative game, making it an important subclass of Markov games to understand.

Moreover, on a technical level, it seems that there is hope for efficient algorithms as planning is efficient in cooperative games (e.g. via Nash-VI as in Bai and Jin [2020]). And so, in this section, we study the question: is there an efficient learning algorithm in cooperative games? We delve into this by first considering cooperative games without payment, which has yet to be addressed in the prior literature. Since the rewards are already aligned, we might expect that there are efficient learning algorithms. To our surprise, however, we find that learning in Cooperative Markov games

can be prohibitively hard, this time in the statistical sense. All proofs in this section may be found in Appendix B.

Structural properties of MDP: We identify the specific MDP properties under which exploration can be statistically intractable, along with complementary positive results. In a nutshell, we find that if the MDP has deterministic tree structure, then efficient algorithms are possible. However, allowing for stochastic or DAG transitions leads to statistical hardness.

Theorem 5.1. There exists a turn-based Stochastic Tree Markov game such that: any (possibly randomized) algorithm that returns the optimal leader policy with probability at least 1 / 2 requires at least Ω (2 | S | ) number of episodes.

Theorem 5.2. There exists a turn-based Deterministic DAG Markov game such that: any (possibly randomized) algorithm that returns the optimal leader policy with probability at least 1 / 2 requires at least Ω (2 | H | ) number of episodes.

Proposition 5.3. Under Markov games that are deterministic trees, then there exists a polynomialtime algorithm that can learn a near-optimal leader policy.

We remark that the statistical intractability results are based on a 'needle-in-the-haystack' construction, where only a specific combination of leader actions is optimal. Structural properties of the MDP like stochastic or DAG transitions allow us to embed this construction in the MDP. Combined with the follower best responding instead of coordinating exploration with the leader, we can show that an exponential number of samples is needed by the leader to find the right combination, even if the rewards are already aligned.

Relaxing Follower Best Response behavior: As the statistical hardness is due to both the structural property of the MDP and the best response nature of the follower, a natural question one may ask is: can relaxing the latter alleviate statistical hardness and allow for efficient learning across all MDPs?

The natural way to relax best response is to consider best response under λ -entropy-regularization, which generalizes follower best response (corresponding to when λ = ∞ ). This behavior model is often used to model human behavior in human-AI interaction and behavioral economics literature [Ziebart et al., 2010, Reddy et al., 2018, McKelvey and Palfrey, 1995]. However, we again find that learning with this follower behavior does not allow for more sample efficient exploration:

Theorem 5.4. There exists a turn-based Deterministic DAG Markov game such that: any (possibly randomized) algorithm that outputs the optimal policy given λ -Entropy-regularized best response with probability at least 1 / 2 requires at least Ω (exp( λ 2 H/ 8)) episodes if λ ≤ 1 and Ω (exp( H/ 8)) episodes if λ &gt; 1 .

In closing, we offer a conceptual interpretation of the technical results in this section, using the example of the assistant agent and the user. Our results suggest that the service provider company can have difficulty exploring, due to the user's best response. Indeed, users are simply looking to use the agent wherever it is at its best, and will not use the agent for the sake of its improvement. In particular, this means that users are not willing to use the agent in states that it currently does not currently excel in. Even though, these are precisely the states that the agent needs to obtain more training samples in. And so, this suggests that if the company wants to efficiently explore to learn an even better agent, incentivized exploration is needed.

## 6 Learning in Cooperative Games with Payment

In sum, we know from the previous section that in Stackelberg games, coordinated exploration is necessary for efficient learning. And so, in this section, we study how payment can be used to align the follower and enable efficient leader exploration. Our overall finding is that payment can lead to efficient exploration, and alleviate the statistical hardness in cooperative games without payment. All proofs in this section may be found in Appendix C.

## 6.1 Regret Guarantees in Cooperative Games

We study regret guarantees under the standard reinforcement learning setup with unknown transitions and unknown rewards, which can be stochastic.

Learning protocol: At each episode k ∈ [ T ] , the leader commits first to a policy π k and a payment function b k . The follower best responds to π k with µ ( π k ) ∈ argmax µ V π k ,µ ( s 0 ; r F + b k ) . After the episode, the leader and the follower observe the resultant trajectory τ k = { ( s i , a i , b i , r L ( s i , a i , b i ) } H i =1 realized by the chosen policies in M (recall that r L = r F ). The goal of the learner is to minimize its Stackelberg regret, defined as follows:

Definition 6.1. In Stackelberg games with trajectory payment, the Stackelberg regret is defined as:

<!-- formula-not-decoded -->

The regret under upfront payment regret may be defined analogously.

Towards analyzing Stackelberg regret, we characterize the optimal policy and trajectory payment when r L = r F ; we can analogously show the same result under upfront payment.

Lemma 6.2. For any π ∗ , b ∗ such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With this, we have that the optimal payment scheme in any cooperative game must be zero, as one would intuitively expect with already aligned rewards. This allows us to decompose Stackelberg regret into regret due to sub-optimality in policy and regret due to payment used during exploration, which will be responsible for the differing rates between trajectory and upfront payment.

Moreover, we note an interesting contrast due to this result. As we just saw, learning can be prohibitively hard in the absence of payment. Hence, we have that payment is not necessary in planning, but is crucial for learning (efficiently).

The crux of our positive results is that we can apply the canonical optimism under uncertainty principle to achieving sublinear Stackelberg regret. This follows from the observation that payment enables optimism in learning, which the leader can operationalize by setting payments according to its bonuses. This incentivizes the follower to also explore optimistically. A key lemma for bounding the policy regret portion of Stackelberg regret goes as follows.

Lemma 6.3. Suppose we can construct an optimistic MDP M k of the true MDP M . Let the optimal leader policy under M k be π k , then:

<!-- formula-not-decoded -->

Note that because the leader knows M k , they know the policy µ M k ( π k ) that they would like to incentivize the follower to play. Using this, we show that one can also bound the regret due to the cumulative payment, to obtain the following regret guarantees.

Theorem 6.4. UCB-VI-FP (Algorithm 2) incurs O ( T 1 / 2 ) regret under trajectory payment. This is tight as there exists a subset of Markov games, where any learning algorithm must incur Ω ( T 1 / 2 ) regret.

Theorem 6.5. There exists an algorithm, leveraging UCB-VI-FP as subroutine, that incurs O ( T 2 / 3 ) regret under upfront payment.

## 6.2 Contrasting Trajectory Payment with Upfront Payment

Finally, as positive results are attainable in Cooperative Markov games, we can analyze the difference in regret rates under the two different payment settings. What is the benefit afforded by settings where the leader can pay on-the-fly? Towards answering this question, we analyze the simple setup of unknown, deterministic rewards. Helpfully, this learning task already a sizable contrast in terms of regret between the two settings. We provide tight bounds on regret guarantees under both payment settings to contrast the two payment settings.

## Algorithm 2 UCB-VI with Follower Payment (UCB-VI-FP)

```
Initialize Q h ( s, a, b ) = H for all h ∈ [ H ] , s, a, b ∈ S h × A × B . for k = 1 , ..., T do for h = H,..., 1; s, a, b ∈ S h × A × B do /triangleright construct M k Compute estimated transitions from data in buffer: ˆ P h ( s ′ | s, a, b ) = N k h ( s,a,b,s ′ ) N k h ( s,a,b ) Compute optimistic rewards of M k from reward samples in buffer: ˆ r k h ( s, a, b ) = ¯ r k h ( s, a, b ) + c √ H 2 N k h ( s,a,b ) /triangleright standard bonus for stochastic rewards Q h ( s, a, b ) = min( H, ˆ r k h ( s, a, b ) + ˆ P k h V h +1 ( s, a, b )) V h ( s ) = max a,b Q h ( s, a, b ) end for Leader commits to Stackelberg policy π k : π k ( s h ) = argmax a max b Q h ( s h , a, b ) . Set outcome-based payment scheme: β k h ( s h , a h , b h ) = 2 · c √ H 2 | S | N k h ( s,a,b ) . for h = 1 , ..., H do Leader plays a k h ∼ π k ( s k h ) , follower plays b k h via µ ( π k ) Transition to s k h +1 ∼ P ( ·| s k h , a k h , b k h ) and save data ( s k h , a k h , b k h , s k h +1 ) in buffer end for end for
```

Proposition 6.6. UCB-VI-FP with indicator bonus incurs constant O ( | S || A || B | ) regret under trajectory payment, where we designate reward under indicator bonus to be ˆ r ( s, a, b ) = 1 { if ( s, a, b ) is unvisited } and r ( s, a, b ) o.w.

As the regret bound is constant in T , we have that the bound must be tight. Next, we derive regret rates under upfront payment, whose regret lower bound requires a significantly nuanced probabilistic argument using Yao's lemma.

Proposition 6.7. There exists an algorithm, leveraging UCB-VI-FP with indicator bonus as subroutine, that incurs O ( T 1 / 2 ) regret under upfront payment.

Proposition 6.8. There exists a subset of Markov Game instances such that any learning algorithm has to incur Ω ( T 1 / 2 ) regret under upfront payment.

The construction of the negative result reveals the key difference in two payment schemes. In a nutshell, upfront payment is affected by difficult-to-reach states ( /epsilon1 -significant states [Jin et al., 2020]). On the other hand, trajectory payment is unaffected as the payment is made only if the follower does reach such a state. That is, the leader's payment for actions in that statement is weighted by the visitation probability.

And so, the key difficulty in exploration under upfront payment is that when payment is needed to incentivize the follower to reach insignificant states, a lot of the payment can be wasted even if the follower is aligned, due to the low visitation probability. This is directly responsible for the sizable change in the regret guarantee, going from O (1) to Ω ( T 1 / 2 ) . Overall, this suggests that if the leader cannot pay on-the-fly, the payment scheme should factor in the reachability of states.

## 7 Discussion

In this work, we study learning in Stackelberg Markov games with payment. To consolidate the theoretical foundations of this setting, we chart the computational and statistical complexity of both planning and learning.

Future Work: Due to the intractability of general-sum settings, we believe that there is much more work to be done in analyzing more specific subclasses of Markov games. Which other subclasses of Markov games are such that efficient algorithms are attainable?

Limitations: One key assumption underlying our paper is that the follower's action can be observed by the leader. We believe that this can be realistic for modeling certain digital settings (such as computers), wherein the agent's actions can be readily tracked (computer-using-agent's actions can be logged and monitored) [Anthropic, 2024, Sumers et al., 2025]. With that said, handling the case

for when the follower's action is not observable is very important, especially in physical environment where monitoring is not possible. And we believe that results from the setting we study can serve as a stepping stone towards results in partial information settings with unobserved actions.

Another key underlying assumption is that the leader can readily observe the follower's reward, either directly or through the follower's report. It is conceivable that in cases the leader cannot observe the reward directly, the follower may not report their reward truthfully. In such settings, we note two observations. Let ( π ∗ ( r ) , b ∗ ( r )) denote an optimal policy under reported follower reward r . Let r F denote the true reward and r ′ F the reported reward.

First, if we are in the cooperative setting, we observe that there is no incentive for the follower to misreport. Because the leader payment is zero, truthful reporting yields the highest return: V π ∗ ( r F ) ,µ ( π ∗ ( r F )) ( s 0 ; r F ) ≥ V π ∗ ( r ′ F ) ,µ ( π ∗ ( r ′ F )) ( s 0 ; r F ) .

Second, in the general-sum bandit setting with direct payment considered by Scheid et al. [2024], the payment can now be nonzero but the follower's gain from misreporting is bounded.

Proposition 7.1. Suppose the follower can misreport r F up to ∆ , ‖ r ′ F -r F ‖ 1 ≤ ∆ . In the bandit setting, the follower's return can change by at most:

<!-- formula-not-decoded -->

and the leader's return can change by at most:

<!-- formula-not-decoded -->

However, an open question is whether such a bound carries over to the Markov game case. How much could the follower gain from misreporting r F up to ∆ ? Are there algorithms that can induce truthfulness, while still attaining some optimality guarantees? We believe there is a fruitful line of work to be done to handle cases where the leader cannot directly observe and/or verify the follower rewards.

## 8 Acknowledgement

TY is grateful for the support of the NSF GRFP. CZ is supported by NSF Award IIS-2440266.

## References

- Omer Ben-Porat, Yishay Mansour, Michal Moshkovitz, and Boaz Taitler. Principal-agent reward shaping in mdps. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 9502-9510, 2024.
- Matteo Bollini, Francesco Bacchiocchi, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Contracting with a reinforcement learning agent by playing trick or treat. arXiv preprint arXiv:2410.13520 , 2024.
- Dima Ivanov, Paul Dütting, Inbal Talgam-Cohen, Tonghan Wang, and David C Parkes. Principal-agent reinforcement learning: Orchestrating ai agents with contracts. arXiv preprint arXiv:2407.18074 , 2024.
- Antoine Scheid, Daniil Tiapkin, Etienne Boursier, Aymeric Capitaine, El Mahdi El Mhamdi, Éric Moulines, Michael I Jordan, and Alain Durmus. Incentivized learning in principal-agent bandit games. arXiv preprint arXiv:2403.03811 , 2024.
- Jibang Wu, Siyu Chen, Mengdi Wang, Huazheng Wang, and Haifeng Xu. Contractual reinforcement learning: Pulling arms with invisible hands. arXiv preprint arXiv:2407.01458 , 2024.
- Paul Dutting, Tim Roughgarden, and Inbal Talgam-Cohen. The complexity of contracts. SIAM Journal on Computing , 50(1):211-254, 2021.
- Stripe. Outcome-based pricing: A guide for businesses. Stripe Resources, 2025. URL https: //stripe.com/en-br/resources/more/outcome-based-pricing .

- Intercom. Pricing AI agents: What does value-based pricing really mean for AI?, May 2025. URL https://www.intercom.com/blog/pricing-ai-agents/ .
- Zendesk. Zendesk first in CX industry to offer outcome-based pricing for AI agents. Zendesk Newsroom, 2025. URL https://www.zendesk.com/newsroom/articles/ zendesk-outcome-based-pricing/ .
- Boston Consulting Group. Rethinking B2B software pricing in the agentic AI era. BCG Publications, 2025. URL https://www.bcg.com/publications/2025/ rethinking-b2b-software-pricing-in-the-era-of-ai .
- Sequoia Capital. Pricing in the AI era: From inputs to outcomes, with Paid CEO Manny Medina. Sequoia Capital Podcast, 2025. URL https://sequoiacap.com/podcast/ pricing-in-the-ai-era-from-inputs-to-outcomes-with-paid-ceo-manny-medina/ .
- Anthropic. Developing a computer use model, 2024. URL https://www.anthropic.com/news/ developing-computer-use .
- Oliver Hart and John Moore. Incomplete contracts and renegotiation. Econometrica: Journal of the Econometric Society , pages 755-785, 1988.
- Philippe Aghion and Richard Holden. Incomplete contracts and the theory of the firm: What have we learned over the past 25 years? Journal of Economic Perspectives , 25(2):181-197, 2011.
- Jean-Jacques Laffont and David Martimort. The theory of incentives: the principal-agent model . Princeton university press, 2002.
- Vincent Conitzer and Tuomas Sandholm. Computing the optimal strategy to commit to. In Proceedings of the 7th ACM conference on Electronic commerce , pages 82-90, 2006.
- Joshua Letchford, Liam MacDermed, Vincent Conitzer, Ronald Parr, and Charles Isbell. Computing optimal strategies to commit to in stochastic games. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 26, pages 1380-1386, 2012.
- Joshua Letchford and Vincent Conitzer. Computing optimal strategies to commit to in extensive-form games. In Proceedings of the 11th ACM conference on Electronic commerce , pages 83-92, 2010.
- Geng Zhao, Banghua Zhu, Jiantao Jiao, and Michael Jordan. Online learning in stackelberg games with an omniscient follower. In International Conference on Machine Learning , pages 4230442316. PMLR, 2023.
- Yu Bai, Chi Jin, Huan Wang, and Caiming Xiong. Sample-efficient learning of stackelberg equilibria in general-sum games. Advances in Neural Information Processing Systems , 34:25799-25811, 2021.
- Siyu Chen, Donglin Yang, Jiayang Li, Senmiao Wang, Zhuoran Yang, and Zhaoran Wang. Adaptive model design for markov decision process. In International Conference on Machine Learning , pages 3679-3700. PMLR, 2022.
- Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml , volume 99, pages 278-287. Citeseer, 1999.
- Abhishek Gupta, Aldo Pacchiano, Yuexiang Zhai, Sham Kakade, and Sergey Levine. Unpacking reward shaping: Understanding the benefits of reward engineering on sample complexity. Advances in Neural Information Processing Systems , 35:15281-15295, 2022.
- Han Shen, Zhuoran Yang, and Tianyi Chen. Principled penalty-based methods for bilevel reinforcement learning and rlhf. arXiv preprint arXiv:2402.06886 , 2024.
- Yu Bai and Chi Jin. Provable self-play algorithms for competitive reinforcement learning. In International conference on machine learning , pages 551-560. PMLR, 2020.
- Brian D Ziebart, J Andrew Bagnell, and Anind K Dey. Modeling interaction via the principle of maximum causal entropy. 2010.

- Sid Reddy, Anca Dragan, and Sergey Levine. Where do you think you're going?: Inferring beliefs about dynamics from behavior. Advances in Neural Information Processing Systems , 31, 2018.
- Richard D McKelvey and Thomas R Palfrey. Quantal response equilibria for normal form games. Games and economic behavior , 10(1):6-38, 1995.
- Chi Jin, Akshay Krishnamurthy, Max Simchowitz, and Tiancheng Yu. Reward-free exploration for reinforcement learning. In International Conference on Machine Learning , pages 4870-4879. PMLR, 2020.
- Theodore Sumers, Raj Agarwal, Nathan Bailey, Tim Belonax, Brian Clarke, Jasmine Deng, Evan Frondorf, Kyla Guru, Keegan Hankes, Jacob Klein, Lynx Lean, Kevin Lin, Linda Petrini, Madeleine Tucker, Ethan Perez, Mrinank Sharma, and Nikhil Saxena. Monitoring computer use via hierarchical summarization, 2025. URL https://alignment.anthropic.com/2025/ summarization-for-monitoring .
- Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International conference on machine learning , pages 263-272. PMLR, 2017.
- Han Zhong, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Can reinforcement learning find stackelberg-nash equilibria in general-sum markov games with myopically rational followers? Journal of Machine Learning Research , 24(35):1-52, 2023.
- Siyu Chen, Mengdi Wang, and Zhuoran Yang. Actions speak what you want: Provably sampleefficient reinforcement learning of the quantal stackelberg equilibrium from strategic feedbacks. arXiv preprint arXiv:2307.14085 , 2023.
- Guru Guruganesh, Yoav Kolumbus, Jon Schneider, Inbal Talgam-Cohen, Emmanouil-Vasileios Vlatakis-Gkaragkounis, Joshua Wang, and S Weinberg. Contracting with a learning agent. Advances in Neural Information Processing Systems , 37:77366-77408, 2024.
- Hsu Kao, Chen-Yu Wei, and Vijay Subramanian. Decentralized cooperative reinforcement learning with hierarchical information structure. In International Conference on Algorithmic Learning Theory , pages 573-605. PMLR, 2022.
- Maria-Florina Balcan, Avrim Blum, Nika Haghtalab, and Ariel D Procaccia. Commitment without regrets: Online learning in stackelberg security games. In Proceedings of the sixteenth ACM conference on economics and computation , pages 61-78, 2015.
- Matthias Gerstgrasser and David C Parkes. Oracles &amp; followers: Stackelberg equilibria in deep multi-agent reinforcement learning. In International Conference on Machine Learning , pages 11213-11236. PMLR, 2023.
- Stephan Dempe and Alain B Zemkoho. On the karush-kuhn-tucker reformulation of the bilevel optimization problem. Nonlinear Analysis: Theory, Methods &amp; Applications , 75(3):1202-1218, 2012.
- Roi Naveiro and David Ríos Insua. Gradient methods for solving stackelberg games. In International conference on algorithmic decision theory , pages 126-140. Springer, 2019.
- Vinzenz Thoma, Barna Pásztor, Andreas Krause, Giorgia Ramponi, and Yifan Hu. Contextual bilevel reinforcement learning for incentive alignment. Advances in Neural Information Processing Systems , 37:127369-127435, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the claims made, including the contributions made in the paper and important assumptions and limitations.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss this in the Discussions section.

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

Justification: We have written down all the proofs for the results in the paper, which are in the appendix due to the page limit of the main paper.

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

Justification: This is a theoretical paper that does not include experiments.

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

Justification: The paper does not include experiments requiring code.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conform in every respect with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a theoretical papers focused on foundational research. We do not anticipate this paper will have any immediate societal impact, especially negative ones.

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

Justification: We believe the paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

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

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
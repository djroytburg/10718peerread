## Adapting to Stochastic and Adversarial Losses in Episodic MDPs with Aggregate Bandit Feedback

## Shinji Ito

The University of Tokyo and RIKEN shinji@mist.i.u-tokyo.ac.jp

## Kevin Jamieson

University of Washington jamieson@cs.washington.edu

## Arnab Maiti

University of Washington arnabm2@uw.edu

## Haipeng Luo

University of Southern California haipengl@usc.edu

## Taira Tsuchiya

The University of Tokyo and RIKEN tsuchiya@mist.i.u-tokyo.ac.jp

## Abstract

We study online learning in finite-horizon episodic Markov decision processes (MDPs) under the challenging aggregate bandit feedback model, where the learner observes only the cumulative loss incurred in each episode, rather than individual losses at each state-action pair. While prior work in this setting has focused exclusively on worst-case analysis, we initiate the study of best-of-both-worlds (BOBW) algorithms that achieve low regret in both stochastic and adversarial environments. We propose the first BOBW algorithms for episodic tabular MDPs with aggregate bandit feedback. In the case of known transitions, our algorithms achieve O (log T ) regret in stochastic settings and O ( √ T ) regret in adversarial ones. Importantly, we also establish matching lower bounds, showing the optimality of our algorithms in this setting. We further extend our approach to unknowntransition settings by incorporating confidence-based techniques. Our results rely on a combination of FTRL over occupancy measures, self-bounding techniques, and new loss estimators inspired by recent advances in online shortest path problems. Along the way, we also provide the first individual-gap-dependent lower bounds and demonstrate near-optimal BOBW algorithms for shortest path problems with bandit feedback.

## 1 Introduction

This paper considers online learning problems for finite-horizon episodic Markov decision processes (MDPs) with aggregate bandit feedback [Efroni et al., 2021, Cohen et al., 2021]. In this feedback model, the learner receives feedback on the aggregate loss in each episode, which is the sum of losses for all state-action pairs in the learner's trajectory of that episode, rather than individual losses for each state-action pair. The aggregate bandit feedback model naturally arises in various real-world applications where only trajectory-level outcomes are observable. For example, in personalized healthcare, a sequence of medical treatments is administered, but only the final patient outcome (e.g., recovery rate) is observed, without attributing effects to individual actions. Similarly, in application to the design of educational programs, students experience a curriculum composed of multiple learning activities, while feedback is typically available only in the form of an overall test score.

Table 1: Regret bounds for episodic MDPs with known transitions. Here π ∗ is an optimal policy and S ∗ is the set of states that can be reached by π ∗ . 'TC' stands for computational time complexity and a checkmark ( ✓ ) indicates that an efficient implementation is possible.

̸

̸

| Algorithm                              | Stochastic                                                              | Adversarial                                               | TC   |
|----------------------------------------|-------------------------------------------------------------------------|-----------------------------------------------------------|------|
| Bubeck et al. [2012]                   | √ &#124; S &#124; 2 &#124; A &#124; T log &#124; A &#124;               | √ &#124; S &#124; 2 &#124; A &#124; T log &#124; A &#124; |      |
| Lancewicki and Mansour [2025]          | √ &#124; S &#124;&#124; A &#124; LT log ι                               | √ &#124; S &#124;&#124; A &#124; LT log ι                 | ✓    |
| Dann et al. [2023a], Ito et al. [2024] | &#124; S &#124; 2 &#124; A &#124; log &#124; A &#124; log T ∆ min       | √ &#124; S &#124; 2 &#124; A &#124; T log &#124; A &#124; |      |
| This study (Tsallis entropy)           | ∑ s = s L ,a = π ∗ ( s ) log T ∆( s,a ) + L &#124; S &#124; log T ∆ min | √ &#124; S &#124;&#124; A &#124; LT                       | ✓    |
| This study (Log-barrier)               | ∑ s = s L ,a = π ∗ ( s ) log T ∆( s,a )                                 | √ &#124; S &#124;&#124; A &#124; LT log T                 | ✓    |
| Lower bound                            | ∑ s ∈ S ∗ ,a :∆( s,a ) > 0 log T ∆( s,a )                               | √ &#124; S &#124;&#124; A &#124; LT                       |      |

̸

̸

In the study of online learning for episodic MDPs, two different models of the loss (or reward) function are commonly considered: the stochastic model and the adversarial model . In the stochastic setting, it is typically assumed that the loss function ℓ t at each episode t is independently drawn from an unknown fixed distribution. In contrast, the adversarial model makes no such probabilistic assumptions and allows the loss function ℓ t to vary arbitrarily over time. In various online learning/bandit problems, including those with individual loss feedback in episodic MDPs, it is well known that one can achieve instance-dependent O (polylog T ) -regret in the stochastic setting, and ˜ O ( √ T ) -regret in the adversarial setting, where T is the number of rounds/episodes and the ˜ O ( · ) notation hides logarithmic factors in parameters. However, to the best of our knowledge, prior work on episodic MDPs with aggregate bandit feedback has focused exclusively on the worst-case analysis (i.e., O ( √ T ) -bounds at best), and no algorithm is known to achieve instance-dependent O (polylog T ) -regret under the stochastic loss model with aggregate bandit feedback and unknown transitions.

This paper focuses on the design of algorithms that can effectively handle both stochastic and adversarial loss models. More specifically, we aim to develop a single algorithm that, without any prior knowledge about the nature of the environment, achieves O (polylog T ) -regret in stochastic settings and ˜ O ( √ T ) -regret in adversarial settings. Such algorithms are referred to as best-of-bothworlds (BOBW) algorithms [Bubeck and Slivkins, 2012]. While many prior works design separate algorithms tailored to either the stochastic or adversarial model, real-world applications often involve uncertainty about the true nature of the environment, making BOBW algorithms especially valuable in practice. Although BOBW algorithms have been developed for various settings-including episodic MDPs with individual loss feedback [Jin and Luo, 2020, Jin et al., 2021, 2023]-no such algorithm was known for episodic MDPs with aggregate bandit feedback with unknown transition, prior to this work.

## 1.1 Contribution

This paper presents the first BOBW algorithms for episodic tabular MDPs with aggregate bandit feedback and unknown transitions. More specifically, we consider layered MDPs with L -layers, and begin by considering the setting where the transition probability matrix P is known, designing an algorithm that achieves ˜ O ( √ T ) -regret in adversarial environments and O (log T ) -regret in stochastic environments. We then extend this approach to the more realistic and challenging setting where the transition matrix P is unknown, with the help of the techniques by Jin et al. [2021] for handling unknown transitions.

Our results are accomplished by combining some algorithmic frameworks including follow-theregularized-leader (FTRL) over occupancy measures and self-bounding techniques [Wei and Luo, 2018, Zimmert and Seldin, 2021] with key ideas from the recent study by Maiti et al. [2025] on the online shortest path problem with bandit feedback. More precisely, our algorithm is inspired by their loss estimation method for the shortest path problem, which plays a central role in our design. By adopting their loss estimation approach, not only can we construct an estimator using only bandit feedback, but we also find that its second moment is well-controlled (see Lemma 4 and the subsequent

Table 2: Regret bounds for episodic MDPs with unknown transition. log ι = O (log( | S || A | T )) .

| Algorithm                     | Stochastic                                             | Adversarial                                                            | TC   |
|-------------------------------|--------------------------------------------------------|------------------------------------------------------------------------|------|
| Efroni et al. [2021]          | √ &#124; S &#124; 4 &#124; A &#124; 3 LT log ι         | -                                                                      |      |
| Cohen et al. [2021]           | √ ( &#124; S &#124;&#124; A &#124; ) O (1) T           | √ ( &#124; S &#124;&#124; A &#124; ) O (1) T                           | ✓    |
| Lancewicki and Mansour [2025] | √ &#124; S &#124; 2 &#124; A &#124; LT log ι           | √ &#124; S &#124; 2 &#124; A &#124; LT log ι                           | ✓    |
| This study                    | ( &#124; S &#124;&#124; A &#124; ) O (1) log 2 ι ∆ min | √ ( &#124; A &#124; + L ) &#124; S &#124; 2 &#124; A &#124; LT log 2 ι | ✓    |

Table 3: Comparison of regret bounds for online shortest path problems with bandit feedback. The quantity c ∗ &gt; 0 represents the instance-dependent constant characterizing the asymptotic lower bound for linear bandits [Lattimore and Szepesvari, 2017], and it is known that c ∗ ≲ | E | ∆ min as shown in Lee et al. [2021]. Here E ′ = ⋃ v ∈ V ∪{ s } ∂ + v \ { π ∗ ( v ) } and ˜ E = ⋃ v ∈ V ∗ ∪{ s } ∂ + v \ { π ∗ ( v ) } , where ∂ + v ⊂ E is the set of outgoing edges from v , π ∗ is an optimal policy and V ∗ is the set of nodes reached by π ∗ . Please refer to Definition 2 for further details.

| Algorithm                              | Stochastic                                             | Adversarial                                          | TC   |
|----------------------------------------|--------------------------------------------------------|------------------------------------------------------|------|
| Bubeck et al. [2012]                   | √ &#124; E &#124; T log &#124;P&#124;                  | √ &#124; E &#124; T log &#124;P&#124;                |      |
| Lattimore and Szepesvari [2017]        | c ∗ log T                                              | -                                                    |      |
| Lee et al. [2021]                      | c ∗ log T log( &#124;P&#124; T )                       | √ &#124; E &#124; T log( &#124;P&#124; T )           |      |
| Dann et al. [2023a], Ito et al. [2024] | &#124; E &#124; log &#124;P&#124; log T ∆ min          | √ &#124; E &#124; T log &#124;P&#124;                |      |
| This study (Tsallis entropy)           | ∑ e ∈ E ′ log T ∆( e ) + &#124; V &#124; 2 log T ∆ min | √ &#124; E &#124; LT                                 | ✓    |
| This study (Log-barrier)               | ∑ e ∈ E ′ log T ∆( e )                                 | √ &#124; E &#124; LT log T                           | ✓    |
| Lower bound                            | c ∗ log T ≳ ∑ e ∈ ˜ E log T ∆( e )                     | √ &#124; E &#124; T log &#124;P&#124; / log &#124; E |      |

discussion). This property is highly beneficial for designing BOBW algorithms. Building on this insight, we propose an efficient and nearly optimal BOBW algorithm for the online shortest path problem with bandit feedback, which naturally extends to the case of MDPs with known transitions.

However, extending this estimation idea to the unknown-transition setting requires substantial care. The new estimator contains negative terms, and thus, naively replacing the occupancy measure with its upper confidence bound, as done in prior work [Jin and Luo, 2020, Jin et al., 2021], does not necessarily yield an optimistic estimator. To address this, we carefully design the estimator so that it is optimistic in expectation and its second moment remains well-controlled. This allows us not only to effectively handle aggregate bandit feedback, but also, perhaps surprisingly, to avoid the technically involved loss-shifting technique used in prior analyses [Jin et al., 2021, 2023], thereby simplifying the overall regret analysis. We refer the reader to Section 3 for a detailed discussion of these estimator constructions.

The regret upper bounds established in this work and in prior studies are summarized in Tables 1, 2, and 3. For detailed definitions of the symbols used in the tables, we refer the reader to Section 2. 'TC' stands for computational time complexity and a checkmark ( ✓ ) indicates that an efficient implementation is possible. The symbol ι &gt; 0 denotes a polynomial factor in other parameters such as T , | S | , and | A | . In Table 3 for the shortest path problem, P denotes the set of all directed paths, and it holds that log |P| ≲ min {| V | , L log | E |} , where L is the maximum number of edges in a path of the given graph. Here, X ≲ Y means X = O ( Y ) in this paper. Similarly, X ≳ Y means X = Ω( Y ) .

As shown in Tables 1 and 3, we propose computationally efficient BOBW algorithms that achieve nearly tight regret bounds for known-transition MDPs and the online shortest path problem. We note here that the corresponding lower bounds for stochastic environments are also new contributions of this paper. The adversarial lower bounds for known-transition MDPs and online

̸

shortest paths are due to [Lancewicki and Mansour, 2025] and [Maiti et al., 2025], respectively. For unknown-transition MDPs (Table 2), we propose the first BOBW algorithm that achieves an O ( ∑ s = s L ,a = π ∗ ( s ) L 4 | S | log ι + | S || A | log 2 ι ∆( s,a ) + ( L 3 | S | 2 )( | S | + | A | ) log ι + L | S | 2 | A | log 2 ι ∆ min ) -regret bound in the stochastic setting and simultaneously achieves an upper bound of ˜ O ( √ ( | A | + L ) | S | 2 | A | LT ) in the adversarial setting. Moreover, all of our BOBW algorithms exhibit robustness to corrupted stochastic environments, achieving regret bounds of the form O ( U + √ UC ) , where U is the stochastic regret and C is the corruption level. These results are established using an argument similar to the standard self-bounding technique [Zimmert and Seldin, 2021, Jin et al., 2021, 2023].

̸

Due to differences in the problem settings, several caveats must be taken into account when making comparisons. First, while prior work on episodic MDPs assumes that the per-step loss or reward within each episode is O (1) , our setting assumes that the aggregate loss or reward over an entire episode is O (1) . To align the scales, we multiply the regret bounds from prior work by a factor of 1 /L . As noted in Remark 1 in the appendix, our setting is strictly more general. In addition, some prior works [Cohen et al., 2021, Lancewicki and Mansour, 2025] consider non-layered settings, and we reinterpret their bounds in terms of the layered setting by replacing | S | L with | S | where appropriate. Furthermore, while we evaluate the expected regret defined in Section 2, some of the prior works [Efroni et al., 2021, Lancewicki and Mansour, 2025, Lee et al., 2021] establish high-probability regret bounds.

Tables 1 and 3 include the bounds achieved by applying algorithms developed for finite-armed linear bandits [Bubeck et al., 2012, Lattimore and Szepesvari, 2017, Lee et al., 2021, Dann et al., 2023a, Ito et al., 2024], where the feature space dimension is | S || A | or | E | , and the number of arms is calculated as | A | | S | or |P| , respectively. We note that due to the exponential number of arms in this approach, it is generally unclear whether an efficient implementation is feasible. In contrast, the other algorithms listed, including our proposed methods, can be implemented efficiently using dynamic programming or convex optimization techniques. We include additional related work in Appendix A.

## 2 Problem setup

## 2.1 Episodic Markov decision process

̸

In this paper, we consider finite-horizon Markov decision processes (MDPs) with finite actions and finite states. The model is defined by a tuple ( S, A, P ) , where S is the finite set of states, A is the set of actions, and P : S × A × S → [0 , 1] is the transition function that defines the probability of moving from one state to another given an action. We assume that the state space S consists of ( L +1) layers : S can be expressed as a disjoint union as S = ⋃ L k =0 S k , where S 0 = { s 0 } (initial state), S L = { s L } (terminal state), S k = ∅ for k ∈ [ L -1] , and S k ∪ S k ′ = ∅ for k = k ′ . Transitions from the k -th layer are allowed only to the ( k +1) -th layer, i.e., for any k ∈ { 0 , 1 , . . . , L -1 } , ∑ s ′ ∈ S k +1 P ( s ′ | s, a ) = 1 holds for all s ∈ S k and a ∈ A and P ( s ′ | s, a ) = 0 for all s ∈ S k , a ∈ A and s ′ ∈ S \ S k +1 . Let k ( s ) denote the index of the layer to which the state s belongs. In known-transition setting, we assume that the transition function P is known to the player. On the other hand, in unknown-transition setting, the player does not know the transition function P and it is learned through interactions with the environment.

̸

In episodic MDPs, the player interacts with the environment in a sequence of episodes. Before each episode t ∈ [ T ] , the player selects a policy π t ∈ Π := { π : ( S \ { s L } ) × A → [0 , 1] | ∑ a ∈ A π ( a | s ) = 1 } and the environment selects a loss function ℓ t : S × A → [0 , 1] . Each episode t ∈ [ T ] consists of a sequence of time steps. The initial state s t 0 is set to s 0 for all episodes t ∈ [ T ] . At each time step k ∈ { 0 , 1 , . . . , L -1 } , the player chooses an action a t k ∈ A according to the policy π t , i.e., a t k follows the distribution π t ( ·| s t k ) , and state s t k +1 is sampled from the transition function P ( ·| s t k , a t k ) given the current state s t k and action a t k . At the end of the episode, the player observes the aggregate loss feedback c t ∈ [0 , 1] such that E [ c t | (( s t k , a t k )) k ∈ [ L -1] ] = ∑ L -1 k =0 ℓ t ( s t k , a t k ) , which corresponds to the sum of the losses incurred at each time step in the episode. We note that the player does not observe the individual losses ℓ t ( s t k , a t k ) for k ∈ [ L -1] . All the information revealed to the player in each episode is the state-action trajectory (( s t k , a t k )) k ∈ [ L -1] and the aggregate loss c t . We

assume that the loss function ℓ t is chosen so that ∑ L -1 k =0 ℓ t ( s t k , a t k ) ∈ [0 , 1] for any possible trajectory (( s t k , a t k )) k ∈ [ L -1] . For notational convenience, we suppose that ℓ t ( s L , a ) is set to 0 for all a ∈ A .

For a transition P , a loss function ℓ : S × A → R , and a policy π ∈ Π , let Q P,π ( s, a ; ℓ ) and V P,π ( s ; ℓ ) express the values of the V - and Q - functions, i.e., we set V P,π ( s L ) = Q P,π ( s L , a ) = 0 and recursively define

<!-- formula-not-decoded -->

̸

for s = s L and a ∈ A . The Q -function and V -function satisfy the equality stated in the following lemma, serves as a fundamental property that underpins both the justification of our loss estimator and the derivation of the regret upper bound.

Lemma 1. Suppose ¯ ℓ is defined by ¯ ℓ ( s, a ) = Q π ( s, a ; ℓ ) -V π ( s ; ℓ ) for some π ∈ Π and for all s ∈ S and a ∈ A . We then have

<!-- formula-not-decoded -->

for any π ′ ∈ Π , s ∈ S and a ∈ A .

In addition, we define the occupancy measure q P,π : S × A → [0 , 1] by

<!-- formula-not-decoded -->

for s ∈ S k and a ∈ A . We also denote q P,π ( s ) = ∑ a ∈ A q P,π ( s, a ) = Pr [ s k = s ] for s ∈ S k , for notational convenience. Let Q P = { q P,π | π ∈ Π } be the set of occupation measures induced by the transition P . We note that Q P is a closed convex set. From the definitions of V , Q and q , we have E [ c t | π t , ℓ t ] = V P,π t ( s 0 , ℓ t ) = 〈 ℓ t , q P,π t 〉 := ∑ L -1 k =0 ∑ s ∈ S k ∑ a ∈ A ℓ t ( s, a ) q P,π t ( s, a ) . Using these concepts, we define the regret :

<!-- formula-not-decoded -->

where E [ · ] denotes the expectation with respect to all the randomness of the environment and the player. We also denote Reg T = max π ∗ ∈ Π Reg T ( π ∗ ) . Hereafter, when it is clear from the context, we may omit P and π for simplicity. Additionally, for notational convenience, we denote Q π t ( · ) = Q P,π ( · ; ℓ t ) , V π t ( · ) = V P,π ( · ; ℓ t ) , q t = q P,π t , and q ∗ = q P,π ∗ .

## 2.2 Regime of environments

We consider several different regimes as models for the environment generating the loss function ℓ t . In an adversarial regime , ℓ t can be chosen arbitrarily by an adversary, depending on the history (( s τ k , a τ k )) k ∈{ 0 ,...,L -1 } ,τ ∈ [ t -1] so far and the policy π t chosen by the player. In a stochastic regime , the loss function ℓ t is independently and identically drawn from an unknown distribution for each episode t ∈ [ T ] . For a stochastic environment, let ℓ ∗ : S × A → [0 , 1] denote the expected loss function, which is defined as ℓ ∗ ( s, a ) = E [ ℓ t ( s, a )] for all s ∈ S and a ∈ A . We then have Reg T = E [ ∑ T t =1 ⟨ ℓ ∗ , q t -q ∗ ⟩ ] = E [ ∑ T t =1 ∑ s ∈ S \{ s L } ∑ a ∈ A ∆( s, a ) q t ( s, a ) ] , where q ∗ ∈ arg min q ∈Q ⟨ ℓ ∗ , q ⟩ and ∆ : S × A → [0 , 1] is defined as ∆( s, a ) = Q π ∗ ( s, a ; ℓ ∗ ) -V π ∗ ( s ; ℓ ∗ ) = Q π ∗ ( s, a ; ℓ ∗ ) -min a ′ ∈ A Q π ∗ ( s, a ′ ; ℓ ∗ ) for an optimal policy π ∗ that minimizes V π ( s 0 ; ℓ ∗ ) . We note that the optimal policy π ∗ is unique if and only if the set arg min a ∈ A { Q π ∗ ( s, a ; ℓ ∗ ) } = { a ∈ A | ∆( s, a ) = 0 } consists of a single action for all s ∈ S . Here, as a generalization of a stochastic regime admitting a unique optimal policy, we can define the an adversarial regime with self-bounding constraints:

Definition 1 (self-bounding regime for MDPs) . Let π ∗ : S → A be a deterministic policy. Suppose that ∆ : S × A → [0 , 1] satisfies ∆( s, a ) &gt; 0 for all s ∈ S \ { s L } and a ∈ A \ { π ∗ ( s ) } . Let C ≥ 0 . The environment is in an adversarial regime with a ( π ∗ , ∆ , C ) self-bounding constraint (or, more concisely, a ( π ∗ , ∆ , C ) -self-bounding regime) if it holds for any algorithm that

<!-- formula-not-decoded -->

̸

We also denote ∆ min = min s = s L ,a = π ∗ ( s ) ∆( s, a ) . As discussed by Zimmert and Seldin [2021], this is a general regime that includes stochastic environments with adversarial corruption, where the parameter C corresponds to the total amount of corruption. For more details, see, e.g., [Zimmert and Seldin, 2021, Jin and Luo, 2020, Jin et al., 2021].

̸

## 2.3 Online shortest path problem

In online shortest path problem, the player is given a directed acyclic graph (DAG) G = ( V ∪ { s, g } , E ) , where s and g are the source and target vertices, respectively, V is the set of other vertices, and E is the set of edges. Denote m = | E | and n = | V | . Let L denote the maximum number of edges in a s -g path in G . In each round t ∈ [ T ] , the environment chooses a loss function ℓ t : E → [0 , 1] and the player chooses a path p t from s to g . At the end of the round, the player can only observe the aggregate loss feedback c t ∈ [0 , 1] such that E [ c t | ℓ t , p t ] = ∑ e ∈ p t ℓ t ( e ) , while the player does not observe the individual losses ℓ t ( e ) for e ∈ E . The definition of regret, the regimes of the environment, and other related concepts are defined in a similar way to in the case of episodic MDPs. More details on the model are given in the appendix. We also note that the online shortest path problem can be interpreted as an 'almost' special case of episodic MDPs with known transitions, but it is not necessarily an exact special case. For details, see Remark 2 in the appendix.

## 3 Core idea: construction of loss estimator with aggregate feedback

A key aspect of the proposed algorithms lies in how to estimate the loss function in a setting where only the limited aggregate loss feedback is available. In this paper, inspired by the approach of Maiti et al. [2025] to the online shortest path problem with bandit feedback, we extend the idea to the MDP setting. We begin by briefly reviewing their approach.

## 3.1 Review of the approach of Maiti et al. [2025] for the online shortest path problem

̸

The online shortest path algorithm by Maiti et al. [2025] maintains an s -g flow q t ∈ Q ⊆ [0 , 1] E of capacity 1 . Note that Q can be interpreted as a convex hull of the set of all s -g paths. In the following, we denote q ( v ) = ∑ e ∈ ∂ + v q ( e ) for any q ∈ [0 , 1] E and v ∈ V ∪ { s } , where ∂ + v ⊂ E is the set of outgoing edges from v . From q t , it samples a path p t ∈ { 0 , 1 } E in a Markovian way, i.e., we choose a path as follows: (i) We first initialize p t ∈ { 0 , 1 } E by p t ( e ) = 0 for all e ∈ E and set v ← s . (ii) While v = g , Pick e ∈ ∂ + v with probability q t ( e ) q t ( v ) , set p t ( e ) ← 1 , and transition to e 's terminal vertex e + , i.e., set v ← e + . We then have E [ p t | q t ] = q t , i.e., each edge e ∈ E is included in the path p t with probability q t ( e ) .

After constructing the path p t as described above and obtaining the aggregate loss feedback c t such that E [ c t | ℓ t , p t ] = ⟨ ℓ t , p t ⟩ , the loss estimator ̂ L t ( p ) for any s -g path p = ( s = v 0 , e 0 , v 1 , e 1 , . . . , v L -1 , e L -1 , v L = g ) is defined as follows: 1

<!-- formula-not-decoded -->

We then have E [ ̂ L t ( p ) | q t , ℓ t ] = ⟨ ℓ t , p ⟩ for any s -g path p . In fact, the conditional expectation given q t , ℓ t satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ¯ L t ( v → v ′ ) represents the conditional expectation of the cost of the subpath of p t from v to v ′ , given that p t goes through v and v ′ . By combining (5), (6) and (7), we obtain E [ ̂ L t ( p ) | q t , ℓ t ] =

1 The construction method by Maiti et al. [2025] does not exactly match the one described below, as they add a uniform shift and incorporate implicit exploration Neu [2015]. These adjustments are designed to obtain high-probability regret bounds, but they are not essential in this study, which focuses on expected regret bounds.

∑ L -1 k =1 ℓ t ( e k ) = ⟨ ℓ t , p ⟩ . In this paper, we define the loss estimator ̂ ℓ t ∈ R E by ̂ ℓ t ( e ) = c t · ( p t ( e ) q t ( e ) -p t ( e -) q t ( e -) ) , where e -∈ V ∪ { s } is the initial vertex of the edge e ∈ E . As we have 〈 ̂ ℓ t , p 〉 = ̂ L t ( p ) -c t for any s -g path p , we can use ̂ ℓ t as an loss estimator in our FTRL framework.

## 3.2 Loss estimator for MDPs with known transition

Let q t ∈ Q be the occupancy measure for the policy π t . Suppose that the trajectory (( s t k , a t k )) L -1 k =0 is generated according to the policy π t and c t is the observed aggregate loss feedback. For any k ∈ { 0 , 1 , . . . , L -1 } and for any s ∈ S k and a ∈ A , denote I t ( s ) = I [ s t k = s ] and I t ( s, a ) = I [( s t k , a t k ) = ( s, a )] . Note that we then have E [ I t ( s ) | π t ] = q t ( s ) and E [ I t ( s, a ) | π t ] = q t ( s, a ) . Inspired by the approach of Maiti et al. [2025], we define the loss estimator as in the following lemma:

Lemma 2. The loss estimator ̂ ℓ t : S × A → R defined as ̂ ℓ t ( s, a ) = c t · ( I t ( s,a ) q t ( s,a ) -I t ( s ) q t ( s ) ) satisfies

<!-- formula-not-decoded -->

From this and Lemma 1, we have Reg T ( π ∗ ) = E [ ∑ T t =1 ( V π t ( s 0 ; ̂ ℓ t ) -V π ∗ ( s 0 ; ̂ ℓ t ) )] = E [ ∑ T t =1 〈 ̂ ℓ t , q t -q ∗ 〉] . Thanks to this, we can use ̂ ℓ t instead of ℓ t in our FTRL-based algorithms.

## 3.3 Loss estimator for MDPs with unknown transition

In the case of unknown transitions, when attempting to construct a loss estimator in the same manner as in Lemma 2, a key difficulty arises from the fact that the true value of q t is not available. To address this issue, one may follow the approach of Jin et al. [2020] and compute an upper confidence bound u t for q t , using it as a surrogate in the estimator. However, naively replacing q t with u t in the definition of ̂ ℓ t in Lemma 2 introduces yet another issue. Specifically, as ̂ ℓ t in Lemma 2 contains a negative term (i.e., -c t I t ( s ) q t ( s ) ), substituting q t with its upper bound u t may lead to an undesirable positive bias in the estimator, which creates an obstacle in the regret analysis. To derive a valid regret upper bound, it is essential that the estimator is optimistic , i.e., its expectation must act as a lower confidence bound on ¯ ℓ t in (8). To this end, we define the following novel loss estimator:

<!-- formula-not-decoded -->

We can evaluate the expectation of ℓ u t in a manner similar to the proof of Lemma 2, as follows:

<!-- formula-not-decoded -->

̸

We here have Q π t ( s, a ; ℓ t ) -V π t ( s ; ℓ t ) = Q π t ( s, a ; ℓ t ) -∑ a ′ ∈ A π t ( a ′ | s ) Q π t ( s, a ′ ; ℓ t ) = (1 -π t ( a | s )) Q π t ( s, a ; ℓ t ) -∑ a ′ = a π t ( a ′ | s ) Q π t ( s, a ′ ; ℓ t ) ≥ -(1 -π t ( a | s )) as Q π t ( s, a ; ℓ t ) ∈ [0 , 1] , and thus: Q π t ( s, a ; ℓ t ) -V π t ( s ; ℓ t ) + 1 -π t ( a | s ) ≥ 0 . Hence, under the condition of u t ( s ) ≥ q t ( s ) , the value of (10) is a lower bound on ¯ ℓ t ( s, a ) := Q π t ( s, a ; ℓ t ) -V π t ( s ; ℓ t ) , i.e., ℓ u t ( s, a ) is an optimistic estimator of ¯ ℓ t ( s, a ) . In addition, the gap between them is at most u t ( s ) -q t ( s ) u t ( s ) (1 -π t ( a | s )) :

Lemma 3. Under the condition of u t ( s ) ≥ q t ( a ) , we have

<!-- formula-not-decoded -->

## 3.4 Second moment of loss estimators

Our proposed algorithm, like those of Jin et al. [2020], Jin and Luo [2020], Jin et al. [2021], is based on the Follow-the-Regularized-Leader (FTRL) framework over occupancy measures. In this framework, the second moment of the loss estimator plays a crucial role. The second moment of the loss estimator introduced in this section can be bounded as follow:

Lemma 4. Loss estimators ̂ ℓ t in Lemma 2 and ℓ u t in (9) satisfy

<!-- formula-not-decoded -->

When applying the self-bounding technique to derive an O (polylog T ) regret bound, the (1 -π ( a | s )) factor in this lemma plays a crucial role. In prior work [Jin et al., 2021, 2023], the original loss estimator did not exhibit this factor in its second moment, and hence the analysis relied on a carefully designed shifting function to apply a loss-shifting trick and extract the desired (1 -π ( a | s )) factor. However, this significantly complicated the analysis. In contrast, our regret analysis does not require the loss-shifting trick, as the self-bounding technique can be applied directly. As a result, we avoid the technically involved analysis necessitated by the loss-shifting trick in previous work.

## 4 Algorithm and regret bounds

## 4.1 Warmup: online shortest path problem

As a warm-up, let us consider the algorithm for the online shortest path problem. Following the ap- proach of Maiti et al. [2025], we update a point q t on the s -g unit flow polytope Q (i.e., the convex hull of all s -g paths) using the following FTRL framework: q t ∈ arg min q ∈Q {〈 ∑ t -1 τ =1 ̂ ℓ τ , q 〉 + ψ t ( q ) } , where ̂ ℓ τ is given as in Section 3.1 and ψ t ( q ) is a regularizer function defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we define ρ t ( e ) = c 2 t p t ( e ) ( 1 -q t ( e ) q t ( e -) ) 2 . We then have the following regret upped bounds:

Theorem 1. Let p ∗ ∈ { 0 , 1 } E be an arbitrary s -g path and π ∗ : V ∪ { s } → E be such that π ∗ ( v ) ∈ ∂ + v for all v ∈ V ∪ { s } and p ∗ ( e ) = 1 = ⇒ π ∗ ( e -) = e . In the case of Tsallis entropy,

<!-- formula-not-decoded -->

If ψ t is given by log-barrier regularizer, we have

<!-- formula-not-decoded -->

Corollary 1. We have Reg T ≲ √ mL ( n + T ) + n log T in the Tsallis entropy case and Reg T ≲ √ mL ( n + T ) log T in the log-barrier case. Simultaneously, under the condition of Reg T ( p ∗ ) ≥ E [ ∑ T t =1 ∑ v ∈ V ∪{ s } ∑ e ∈ ∂ + v \{ π ∗ ( v ) } ∆( e ) p t ( e ) ] -C for some ∆ ∈ [0 , 1] E and C ≥ 0 , we have Reg T ( p ∗ ) ≲ U + √ UC , where U = ∑ v ∈ V ∪{ s } ∑ e ∈ ∂ + v \{ π ∗ ( v ) } log T ∆( e ) + n 2 log T ∆ min in the Tsallis entropy case and U = ∑ v ∈ V ∪{ s } ∑ e ∈ ∂ + v \{ π ∗ ( v ) } log T ∆( e ) for the log-barrier case.

The tightness of the gap-dependent upper bound derived here is discussed in the appendix.

## 4.2 MDPs with known transition

The algorithm design for episodic MDPs with known transitions is almost identical to the case of the shortest path problem. Specifically, we apply the FTRL framework over the set of all occupancy measures as the feasible region, replace each edge e ∈ E in the regularization functions in (Tsallis entropy) and (log-barrier) with a state-action pair ( s, a ) ∈ S ⊆ { s L } × A , and redefine ρ t as ρ t ( s, a ) = c 2 t I t ( s, a ) (1 -π t ( a | s )) 2 . With this setup, we obtain the following regret bound:

̸

Corollary 2. We have Reg T ≲ √ | S || A | LT + | S || A | log T in the Tsallis entropy case and Reg T ≲ √ | S || A | LT log T in the log-barrier case. Simultaneously, under the condition of (4) , we have Reg T ( π ∗ ) ≲ U + √ UC , where U = ∑ s = s L ∑ a = π ∗ ( s ) log T ∆( s,a ) + L | S | log T ∆ min in the Tsallis entropy case and U = ∑ s = s L ∑ a = π ∗ ( s ) log T ∆( s,a ) for the log-barrier case.

̸

The gap-dependent upper bound achieved by the log-barrier regularization in this corollary is tight. In fact, the following lower bound holds:

Theorem 2. Consider stochastic environment in which c t follows a Bernoulli distribution of parameter ∑ L -1 k =1 ℓ ∗ ( s k t , a k t ) where we assume that this value is in [3 / 8 , 5 / 8] for any possible trajectories. Define ∆ : S × A → [0 , 1] by ∆( s, a ) = Q π ∗ ( s, a ; ℓ ∗ ) -V π ∗ ( s ; ℓ ∗ ) for an optimal policy π ∗ . Let S ∗ be the set of all states s ∈ S \ { s L } such that q π ∗ ( s ) &gt; 0 for some optimal policy π ∗ . Then, for any consistent algorithms, we have lim inf T →∞ Reg T log T ≳ ∑ s ∈ S ∗ ∑ a ∈ A :∆( s,a ) &gt; 0 1 ∆( s,a ) .

## 4.3 MDPs with unknown transition

Our proposed algorithm for MDPs with unknown transitions adopts an epoch-based approach, similar to prior work [Jin et al., 2021, 2020]. In each epoch i , it updates both the transition probability estimates and their corresponding confidence intervals. Based on these, we compute an upper confidence bound u t on the occupancy measure q t , and use it to construct the loss estimator ℓ u t as defined in (9). Note that u t ( s, a ) can be efficiently computed using the COMP-UOB procedure proposed by [Jin et al., 2020].

We then define the adjusted loss estimator as ̂ ℓ t := ℓ u t -B i , where B i is a bonus term derived from the confidence width. Unlike prior work, our choice of the loss estimator ℓ u t allows us to avoid scaling B i by an additional factor of L . The policy for each episode is selected by applying FTRL over the estimated occupancy measure space using ̂ ℓ t . The regularization function used here matches the Tsallis entropy regularizer from Section 4.2, except that the learning rate η t is reset at the beginning of each epoch, and a small log-barrier term is added to stabilize updates.

Anotable improvement over prior work [Jin et al., 2021] is that the second-moment bound established in Lemma 4 allows us to bypass the loss-shifting technique. As a result, our regret bounds exhibit improved dependence on the horizon L . We refer the reader to Algorithm 1 and the appendix E for full details. Our algorithm, constructed in this way, achieves the following upper bounds:

Theorem 3. In the bandit feedback setting, Algorithm 1 with δ = 1 T 3 and ι = | S || A | T δ guarantees Reg T ( π ⋆ ) = ˜ O ( L | S | √ | A | T + | S || A | √ LT + L 2 | S | 3 | A | 2 ) and simultaneously Reg T ( π ⋆ ) = O ( U + √ UC + V ) under Condition (4) , where V = L 2 | S | 3 | A | 2 ln 2 ι and U is defined as

̸

<!-- formula-not-decoded -->

̸

We defer the proof of the above theorem to Appendix F.

## 5 Conclusion

This paper initiated the study of best-of-both-worlds (BOBW) algorithms for finite-horizon episodic MDPs with aggregate bandit feedback. We proposed efficient algorithms that achieve low regret in both stochastic and adversarial settings, and established nearly tight upper and lower bounds under both known- and unknown-transition settings. Our approach is built upon FTRL over occupancy measures, combined with carefully designed loss estimators that are optimistic in expectation and admit tight second-moment bounds.

Despite these contributions, many open questions remain. A central limitation of our approach is its reliance on occupancy measure updates via FTRL, which-while grounded in convex optimization and thus computationally feasible to some extent-still requires solving a convex problem in each

̸

̸

<!-- formula-not-decoded -->

round. Moreover, this framework does not easily extend beyond tabular MDPs to more general representations such as linear models or function approximation.

A promising direction to address these limitations is to adopt policy optimization-based methods [Shani et al., 2020, Luo et al., 2021]. In particular, a recent paper by [Lancewicki and Mansour, 2025] has shown that near-optimal and efficient adversarial regret bounds can be achieved through policy optimization. Combining this line of work with the techniques in [Dann et al., 2023a] may yield BOBW algorithms that are both computationally efficient and more broadly applicable.

Another important direction for future research is to extend the present results beyond the online shortest path problem to other combinatorial optimization settings, or to more challenging MDP formulations such as stochastic shortest path problems [Chen et al., 2021]. Addressing these challenges may lead to a more comprehensive understanding of online learning under aggregate feedback.

## Acknowledgments

Ito is supported by JSPS KAKENHI Grant Number JP25K03184. Jamieson is funded in part by NSF Award CAREER 2141511 and Microsoft Grant for Customer Experience Innovation. Luo is funded by NSF award IIS-1943607. Tsuchiya is supported by JSPS KAKENHI Grant Number JP24K23852.

## References

- Sébastien Bubeck and Aleksandrs Slivkins. The best of both worlds: Stochastic and adversarial bandits. In Proceedings of the 25th Annual Conference on Learning Theory , volume 23, pages 42.1-42.23, 2012.

Sébastien Bubeck, Nicolo Cesa-Bianchi, and Sham Kakade. Towards minimax policies for online linear optimization with bandit feedback. In Conference on Learning Theory , volume 23, pages 41.1-41.14, 2012.

Sébastien Bubeck, Michael Cohen, and Yuanzhi Li. Sparsity, variance and curvature in multi-armed bandits. In Proceedings of Algorithmic Learning Theory , volume 83, pages 111-127. PMLR, 2018.

- Clément L Canonne. A short note on an inequality between KL and TV. arXiv preprint arXiv:2202.07198 , 2022.
- Asaf Cassel, Haipeng Luo, Aviv Rosenberg, and Dmitry Sotnikov. Near-optimal regret in linear MDPs with aggregate bandit feedback. In Proceedings of the 41st International Conference on Machine Learning , volume 235, pages 5757-5791. PMLR, 2024.
- Niladri Chatterji, Aldo Pacchiano, Peter Bartlett, and Michael Jordan. On the theory of reinforcement learning with once-per-episode feedback. In Advances in Neural Information Processing Systems , volume 34, pages 3401-3412. Curran Associates, Inc., 2021.
- Liyu Chen, Haipeng Luo, and Chen-Yu Wei. Minimax regret for stochastic shortest path with adversarial costs and known transition. In Conference on Learning Theory , pages 1180-1215. PMLR, 2021.
- Alon Cohen, Haim Kaplan, Tomer Koren, and Yishay Mansour. Online Markov decision processes with aggregate bandit feedback. In Conference on Learning Theory , pages 1301-1329. PMLR, 2021.
- Chris Dann, Chen-Yu Wei, and Julian Zimmert. A blackbox approach to best of both worlds in bandits and beyond. In The Thirty Sixth Annual Conference on Learning Theory , pages 5503-5570. PMLR, 2023a.
- Christoph Dann, Chen-Yu Wei, and Julian Zimmert. Best of both worlds policy optimization. In Proceedings of the 40th International Conference on Machine Learning , volume 202, pages 6968-7008. PMLR, 2023b.
- Yonathan Efroni, Nadav Merlis, and Shie Mannor. Reinforcement learning with trajectory feedback. In Proceedings of the AAAI conference on artificial intelligence , volume 35-8, pages 7288-7295, 2021.
- Liad Erez and Tomer Koren. Towards best-of-all-worlds online learning with feedback graphs. In Advances in Neural Information Processing Systems , volume 34, pages 28511-28521. Curran Associates, Inc., 2021.
- Shinji Ito. Parameter-free multi-armed bandit algorithms with hybrid data-dependent regret bounds. In Conference on Learning Theory , pages 2552-2583. PMLR, 2021.
- Shinji Ito, Taira Tsuchiya, and Junya Honda. Nearly optimal best-of-both-worlds algorithms for online learning with feedback graphs. In Advances in Neural Information Processing Systems , volume 35, pages 28631-28643. Curran Associates, Inc., 2022.
- Shinji Ito, Taira Tsuchiya, and Junya Honda. Adaptive learning rate for follow-the-regularized-leader: Competitive analysis and best-of-both-worlds. In The Thirty Seventh Annual Conference on Learning Theory , pages 2522-2563. PMLR, 2024.
- Chi Jin, Tiancheng Jin, Haipeng Luo, Suvrit Sra, and Tiancheng Yu. Learning adversarial Markov decision processes with bandit feedback and unknown transition. In International Conference on Machine Learning , pages 4860-4869. PMLR, 2020.
- Tiancheng Jin and Haipeng Luo. Simultaneously learning stochastic and adversarial episodic MDPs with known transition. In Advances in Neural Information Processing Systems , volume 33, pages 16557-16566. Curran Associates, Inc., 2020.
- Tiancheng Jin, Longbo Huang, and Haipeng Luo. The best of both worlds: stochastic and adversarial episodic MDPs with unknown transition. In Advances in Neural Information Processing Systems , volume 34, pages 20491-20502. Curran Associates, Inc., 2021.
- Tiancheng Jin, Junyan Liu, Chloé Rouyer, William Chang, Chen-Yu Wei, and Haipeng Luo. Noregret online reinforcement learning with adversarial losses and transitions. In Advances in Neural Information Processing Systems , volume 36, pages 38520-38585. Curran Associates, Inc., 2023.
- Tal Lancewicki and Yishay Mansour. Near-optimal regret using policy optimization in online MDPs with aggregate bandit feedback. arXiv preprint arXiv:2502.04004 , 2025.

- Tor Lattimore and Csaba Szepesvari. The end of optimism? an asymptotic analysis of finite-armed linear bandits. In Artificial Intelligence and Statistics , pages 728-737. PMLR, 2017.
- Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, Mengxiao Zhang, and Xiaojin Zhang. Achieving near instance-optimality and minimax-optimality in stochastic and adversarial linear bandits simultaneously. In International Conference on Machine Learning , pages 6142-6151. PMLR, 2021.
- Haipeng Luo, Chen-Yu Wei, and Chung-Wei Lee. Policy optimization in adversarial MDPs: Improved exploration via dilated bonuses. Advances in Neural Information Processing Systems , 34:2293122942, 2021.
- Arnab Maiti, Zhiyuan Fan, Kevin Jamieson, Lillian J. Ratliff, and Gabriele Farina. Efficient nearoptimal algorithm for online shortest paths in directed acyclic graphs with bandit feedback against adaptive adversaries, 2025.
- Saeed Masoudian, Julian Zimmert, and Yevgeny Seldin. A best-of-both-worlds algorithm for bandits with delayed feedback with robustness to excessive delays. In Advances in Neural Information Processing Systems , volume 37, pages 141071-141102. Curran Associates, Inc., 2024.
- Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. In Advances in Neural Information Processing Systems , volume 28, pages 3168-3176. Curran Associates, Inc., 2015.
- Lior Shani, Yonathan Efroni, Aviv Rosenberg, and Shie Mannor. Optimistic policy optimization with bandit feedback. In International Conference on Machine Learning , pages 8604-8613. PMLR, 2020.
- Taira Tsuchiya, Shinji Ito, and Junya Honda. Best-of-both-worlds algorithms for partial monitoring. In Proceedings of The 34th International Conference on Algorithmic Learning Theory . PMLR, 2023a.
- Taira Tsuchiya, Shinji Ito, and Junya Honda. Stability-penalty-adaptive follow-the-regularizedleader: Sparsity, game-dependency, and best-of-both-worlds. In Advances in Neural Information Processing Systems , volume 36, 2023b.
- Chen-Yu Wei and Haipeng Luo. More adaptive algorithms for adversarial bandits. In Conference On Learning Theory , pages 1263-1291. PMLR, 2018.
- Julian Zimmert and Yevgeny Seldin. An optimal algorithm for adversarial bandits with arbitrary delays. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics , volume 108, pages 3285-3294. PMLR, 2020.
- Julian Zimmert and Yevgeny Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research , 22(28):1-49, 2021.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Theorems match the claims made in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] .

Justification: Limitations and future works are discussed in the conclusion section.

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

Justification: Every assumption on our problem setting is clearly mentioned in the introduction.

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

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

Justification: Our paper is theoretical in nature.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our paper is theoretical in nature.

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

Justification: Our paper is theoretical in nature.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Our paper is theoretical in nature.

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

Justification: Our paper is theoretical in nature.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper is theoretical in nature.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper is theoretical in nature.

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

## Contents

| 1 Introduction   | 1 Introduction                                                                       |   1 |
|------------------|--------------------------------------------------------------------------------------|-----|
|                  | Contribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   2 |
| 2                | Problem setup                                                                        |   4 |
|                  | Episodic Markov decision process . . . . . . . . . . . . . . . . . . . . . . . .     |   4 |
|                  | Regime of environments . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     |   5 |
|                  | Online shortest path problem . . . . . . . . . . . . . . . . . . . . . . . . . . .   |   6 |
| 3                | Core idea: construction of loss estimator with aggregate feedback                    |   6 |
|                  | Review of the approach of Maiti et al. [2025] for the online shortest path problem   |   6 |
|                  | Loss estimator for MDPs with known transition . . . . . . . . . . . . . . . . .      |   7 |
|                  | Loss estimator for MDPs with unknown transition . . . . . . . . . . . . . . . .      |   7 |
|                  | Second moment of loss estimators . . . . . . . . . . . . . . . . . . . . . . . .     |   7 |
| 4                | Algorithm and regret bounds                                                          |   8 |
|                  | Warmup: online shortest path problem . . . . . . . . . . . . . . . . . . . . . .     |   8 |
|                  | MDPs with known transition . . . . . . . . . . . . . . . . . . . . . . . . . . .     |   8 |
|                  | MDPs with unknown transition . . . . . . . . . . . . . . . . . . . . . . . . . .     |   9 |
| 5                | Conclusion                                                                           |   9 |
| Appendix         |                                                                                      |  20 |
| A                | Additional related work                                                              |  21 |
| B                | Auxiliary lemmas                                                                     |  22 |
|                  | Lemmas for FTRL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      |  22 |
|                  | B.1.1 Stability terms for one-dimensional functions . . . . . . . . . . . . . .      |  22 |
| C                | Online shortest path problem with bandit feedback                                    |  24 |
|                  | Notation and problem setup . . . . . . . . . . . . . . . . . . . . . . . . . . . .   |  24 |
|                  | Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  |  25 |
|                  | Regret analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  |  26 |
|                  | C.3.1 Analysis for Tsallis-entropy case . . . . . . . . . . . . . . . . . . . . .    |  28 |
|                  | C.3.2 Analysis for log-barrier case . . . . . . . . . . . . . . . . . . . . . . .    |  31 |
| D                | MDPs with known transition                                                           |  35 |
|                  | Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  |  35 |
|                  | D.2.1 Analysis for Tsallis-entropy case . . . . . . . . . . . . . . . . . . . . .    |  37 |

|            | D.2.2 Analysis for                                                                                             | log-barrier case . . . . .                                                                                     |   38 |
|------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|------|
| D.3 E      | Lower bound for stochastic MDPs . . . . . . . . . . . . . . . . . . Algorithm for MDPs with unknown transition | Lower bound for stochastic MDPs . . . . . . . . . . . . . . . . . . Algorithm for MDPs with unknown transition |   38 |
|            |                                                                                                                |                                                                                                                |   41 |
| E.1        | Confidence set of the true transition . . . . . . . . . . .                                                    | Confidence set of the true transition . . . . . . . . . . .                                                    |   42 |
| E.2        | Loss estimator and Regularizer . . . . .                                                                       | . . .                                                                                                          |   42 |
| E.3        | Main Result . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                          | Main Result . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                          |   43 |
| F Analysis | of BOBWwith unknown transitions                                                                                | of BOBWwith unknown transitions                                                                                |   44 |
| F.1        | Auxiliary lemmas . . . . . . . . . . . .                                                                       | . . .                                                                                                          |   45 |
| F.2        | Technical lemmas to analyze ESTREG . . .                                                                       | .                                                                                                              |   49 |
| F.3        | Adversarial regret guarantee                                                                                   | . . . . . . . . .                                                                                              |   52 |
| F.4        | Stochastic regret guarantee                                                                                    | . . . . . . . . . .                                                                                            |   53 |
|            | F.4.1                                                                                                          | Self-bounding terms and related lemma                                                                          |   53 |
|            | F.4.2                                                                                                          | Proof for the stochastic world . . . .                                                                         |   54 |

## A Additional related work

FTRL and best-of-both-worlds algorithms In episodic tabular MDPs with adversarial losses, Jin and Luo [2020] is the first to propose a BOBW algorithm under known transitions. Jin et al. [2021] extended this to the unknown-transition setting, which is further improved and extend to the setting where the transition can vary over episodes [Jin et al., 2023]. Subsequently, policy optimization algorithms was shown to achieve BOBW guarantees with improved computational efficiency [Dann et al., 2023b]. In spite of these developments, our work is the first to consider BOBW algorithms under aggregate feedback. We build on the analysis of Jin et al. [2021]. While we may improve our bounds by using the optimistic transition technique from the recent work by Jin et al. [2023], it remains unclear whether this technique can be effectively combined with our loss estimator, which can be negative.

Akey challenge in achieving BOBW is the design and analysis of the regularizer in FTRL. In this work, we consider two types of regularizers (see Section 4.1). The first one is a hybrid regularizer [Bubeck et al., 2018] that combines the Tsallis entropy with a small-coefficient log-barrier. The first BOBW algorithm based on the Tsallis entropy was initially explored by Zimmert and Seldin [2021], and the hybrid regularizers to ensure the stability of FTRL have been shown to be powerful in obtaining BOBW guarantees for complex online learning problems or for obtaining adaptive bounds [Zimmert and Seldin, 2020, Erez and Koren, 2021, Ito et al., 2022, 2024, Tsuchiya et al., 2023a,b, Masoudian et al., 2024]. The second regularizer we consider is the log-barrier regularizer with adaptive learning rates, developed in Wei and Luo [2018], Ito [2021]. As we show in this paper, although the strong regularization of the log-barrier can introduce an additional O ( √ log T ) multiplicative factor in adversarial settings, it ensures a strong stability of FTRL.

Episodic MDPs with aggregate feedback. Recently, MDPs with aggregate feedback have received growing attention. For example, in tabular MDPs, aggregate feedback has been studied in both the stochastic and adversarial settings (see [Efroni et al., 2021, Cohen et al., 2021, Chatterji et al., 2021]), as well as in the context of policy optimization [Lancewicki and Mansour, 2025]. Similar interest has emerged for linear MDPs as well (see [Cassel et al., 2024]). However, to the best of our knowledge, our work is the first to study best-of-both-worlds guarantees in the setting of aggregate feedback.

## Remarks in comparing results

Remark 1 (On the scale of loss) . In existing studies on online learning for MDPs with adversarial losses, it's common to assume that ℓ t ( s, a ) ∈ [0 , 1] for all s and a . Such setting is reduced to our

setting by scaling the losses by a factor of 1 /L , which therefore can be regarded as a special case of our setting. If the loss is given by this reduction, values of losses have the same scale of O (1 /L ) for all layers. In contrast, in our setting, the losses may have different scales (possibly &gt; 1 /L ) for each layer, which can be interpreted to be more general.

Remark 2 (On online shortest path problems and episodic MDPs) . The online shortest path problem can be interpreted as an 'almost' special case of episodic MDPs with known transitions, but it is not necessarily an exact special case. Intuitively, the vertices v ∈ V in the shortest path problem correspond to the states s ∈ S in an MDP, and selecting one outgoing edge e ∈ ∂ + v from a vertex v corresponds to choosing an action a ∈ A in the MDP. The shortest path problem however differs in several aspects: the set of vertices V does not necessarily have a hierarchical structure, the number of edges in a path from the source to the sink is not necessarily fixed, and the set ∂ + v of outgoing edges available for selection can vary depending on the vertex v . Therefore, the shortest path problem cannot always be directly interpreted as an MDP. Consequently, regret bounds for MDPs do not immediately translate to results for the shortest path problem. We therefore provide a separate discussion on the online shortest path problem. However, the overall framework for algorithm design and analysis is similar to that for episodic MDPs with known transitions.

## B Auxiliary lemmas

## B.1 Lemmas for FTRL

## B.1.1 Stability terms for one-dimensional functions

Lemma 5. Let ϕ : R ≥ 0 → R be defined as ϕ ( x ) = -2 √ x and D ϕ ( y, x ) be the Bregman divergence associated with ϕ , i.e.,

<!-- formula-not-decoded -->

Then, for any x ∈ (0 , 1) , ℓ ∈ R and η &gt; 0 such that η √ xℓ &gt; -1 , we have

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

Lemma 6. Let η &gt; 0 and β &gt; 0 . Suppose that ϕ : R &gt; 0 → R is defined as ϕ ( x ) = -2 η √ x -β ln( x ) . Let D ϕ ( y, x ) be the Bregman divergence associated with ϕ . Then, for any x ∈ (0 , 1) , ℓ ∈ R and η such that xℓ ≥ -β 2 , we have

<!-- formula-not-decoded -->

Proof. If ℓ ≥ 0 , it immediately follows from Lemma 5 that the left-hand side of (12) is bounded by ηx 3 / 2 ℓ 2 from above. We next consider the case of ℓ &lt; 0 . The derivative of ℓ · ( x -y ) -D ϕ ( y, x ) in y is given as

<!-- formula-not-decoded -->

This is a monotone decreasing function and hence the maximum of ℓ · ( x -y ) -D ϕ ( y, x ) is attained at y ∗ ∈ R &gt; 0 such that g ( y ∗ ) = 0 . As we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we have

<!-- formula-not-decoded -->

where the last inequality follows from the assumption of ℓx ≥ -β 2 . We hence have

<!-- formula-not-decoded -->

If 0 &lt; -ηℓ √ x ≤ 1 / 2 , we then have (1 + ηℓ √ x ) -2 -1 ≤ -6 ηℓ √ x , which implies that the value of (13) is at most -ℓx · ( -6 ηℓ √ x ) = 6 ηx 3 / 2 ℓ 2 . If -ηℓ √ x &gt; 1 / 2 , we then have the value of (13) is at most -ℓx &lt; -ℓx · ( -2 ηℓ √ x ) = 2 ηx 3 / 2 ℓ 2 . This completes the proof.

Lemma 7. Let ϕ : R &gt; 0 → R be defined as ϕ ( x ) = -log( x ) and D ϕ ( y, x ) be the Bregman divengence associated with ϕ , i.e.,

<!-- formula-not-decoded -->

Then, for any x ∈ (0 , 1) , ℓ ∈ R and η &gt; 0 such that ηxℓ &gt; -1 , we have

<!-- formula-not-decoded -->

Consequently, if ηℓx ≥ -1 2 , we have

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

For fixed x , ℓ and η , this value is maximized when 1 y = 1 x + ηℓ . We then have

<!-- formula-not-decoded -->

As we have -log(1 + a ) + a ≤ 1 2 a 2 + I [ a &lt; 0] · | a | 3 for a ≥ -1 / 2 , we have

<!-- formula-not-decoded -->

Proof of Lemma 1. We can show this by backward induction in layers. For s = s L , (1) is clear as both sides are equal to 0 . For s ∈ S k with k &lt; L ,

<!-- formula-not-decoded -->

where the second equality follows from the induction hypothesis and the definition of ¯ ℓ . We hence have

<!-- formula-not-decoded -->

## C Online shortest path problem with bandit feedback

In this section, we analyze our algorithm for online shortest path problem. Specifically, we prove Theorems 4 and 5, which together directly imply Corollary 1 in the main body. In addition, we prove our lower bound result in Theorem 6.

## C.1 Notation and problem setup

- G = ( V ∪ { s, g } , E ) : a directed acyclic graph.
- s : Source node.
- g : Sink node.
- V : Set of vertices that are neither sources nor sinks.
- E ⊆ ( V ∪ { s } ) × ( V ∪ { g } ) : set of directed edges.
- e -, e + ∈ V ∪ { s, g } : initial and terminal vertices of an edge e ∈ E , i.e., e = ( e -, e + ) .
- ∂ -v, ∂ + v ⊆ E : sets of incoming and outgoing edges of a vertex v ∈ V ∪ { s, g } , i.e., ∂ -v = { e ∈ E | e + = v } , ∂ + v = { e ∈ E | e -= v } .
- n = | V | .
- m = | E | .
- P ⊆ { 0 , 1 } E : set of (vector representations of) s -g paths.

- Q = conv( P ) ⊆ [0 , 1] E : set of s -g flows of value 1 , equivalently convex hull of P . Note Q = { q ∈ [0 , 1] E | ∑ e ∈ ∂ + s q ( e ) = ∑ e ∈ ∂ -g q ( e ) = 1 , ∑ e ∈ ∂ + v q ( e ) = ∑ e ∈ ∂ -v q ( e )( ∀ v ∈ V ) } .
- L = max p ∈P {∥ p ∥ 1 } ≤ | V | +1 : maximum length of s -g paths.
- Without loss of generality, we assume that every vertex v ∈ V admits a path from s to g passing through v .

In each round t ∈ [ T ] , an environment chooses ℓ t ∈ R E ≥ 0 and then the player chooses p t ∈ P , after which the player observes a feedback c t ∈ [0 , 1] such that E [ c t | ℓ t , p t ] = ⟨ ℓ t , p t ⟩ . We here assume that ℓ t satisfies ⟨ ℓ t , p ⟩ ≤ 1 for all p ∈ P . The performance of the player is evaluated in terms of regret defined as:

<!-- formula-not-decoded -->

where the expectation is taken over all randomness arising from the environment and the algorithm.

## C.2 Algorithm

The algorithm updates q t ∈ Q by an FTRL approach and picks p t ∈ P so that E [ p t | q t ] = q t , following the technique of [Maiti et al., 2025].

In the following, for q ∈ Q and v ∈ V , we denote

<!-- formula-not-decoded -->

for the notational simplicity.

Let p v ∈ P be an s -g path that passes through v , i.e., p v ( v ) = 1 . Define q 0 ∈ Q by

<!-- formula-not-decoded -->

We then have q 0 ( v ) ≥ 1 / | V | = 1 /n for any v ∈ V .

Define q t by

<!-- formula-not-decoded -->

where ̂ ℓ τ is an estimator for ℓ τ defined later. The regularizer ψ t is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ρ τ ∈ [0 , 1] will be defined later.

Based on q t ∈ Q , we pick p t ∈ P in the same way as in [Maiti et al., 2025]:

- Initialize p ∈ { 0 , 1 } E by p ( e ) = 0 for all e ∈ E and set v ← s .

̸

- While v = g :
- -Pick e ∈ ∂ + v with probability q t ( e ) /q t ( v ) .
- -Set p ( e ) ← 1 and transition to the next node e + , i.e., v ← e + .

We then have E [ p t | q t ] = q t .

After outputting p t , we get feedback c t ∈ [0 , 1] such that E [ c t | p t , ℓ t ] = ⟨ ℓ t , p t ⟩ . Based on this, we define ̂ ℓ t ∈ R E by

<!-- formula-not-decoded -->

Note that the notation of (14) applies to p ∈ P ⊆ Q as well, and that p t ( v ) = 1 if and only if the path passes through the node v . Then, it holds for any q ∈ Q that

<!-- formula-not-decoded -->

We note that, an alternative definition of ̂ ℓ t given as

<!-- formula-not-decoded -->

also satisfies (20) similarly, and hence we have

<!-- formula-not-decoded -->

for any q ∈ Q . Therefore, using ̂ ℓ ′ t in (21) instead of ̂ ℓ t in (19) does not change the behavior of the algorithm.

We now state the following that can be proved in a similar way as in [Maiti et al., 2025]:

Lemma 8. For any q, q ′ ∈ Q , we have

<!-- formula-not-decoded -->

where the expectation is taken w.r.t. p t .

Note that the effect of the path-length does not appear here. That is, we do not need to assume that the path lengths are the same.

## C.3 Regret analysis

Definition 2 (consistent policy) . Define Π = { π : V ∪ { s } → E | π ( v ) ∈ ∂ + v ( ∀ v ∈ V ∪ { s } ) } . Let p ∗ ∈ P be an arbitrary s -g path. Let E ∗ ⊆ E and V ∗ ⊆ V denote the sets of edges and nodes included in p ∗ , i.e., E ∗ = { e ∈ E | p ∗ ( e ) = 1 } and V ∗ = { v ∈ V | p ∗ ( v ) = 1 } . We say π ∗ ∈ Π is consistent with p ∗ ∈ P if and only if π ∗ ( v ) ∈ E ∗ for all v ∈ V ∗ ∪{ s } . We denotes E ′ = E \ Im( π ∗ ) .

Definition 3 (self-bounding regime for online shortest path) . Let p ∗ ∈ P be an arbitrary s -g path and suppose that π ∗ ∈ Π is consistent with p ∗ . Suppose that ∆ ∈ [0 , 1] E satisfies ∆( e ) &gt; 0 for all e ∈ E ′ = E \ Im( π ∗ ) . The environment is in a ( p ∗ , π ∗ , ∆ , C ) -self-bounding regime if it holds that

<!-- formula-not-decoded -->

Remark 3. An example of ∆ ∈ [0 , 1] E can be constructed as follows: Assume that ℓ t follows an identical distribution independently for all t ∈ [ T ] and denote ℓ ∗ = E [ ℓ t ] . For each u, v ∈ V ∪{ s, g } , let dist( u, v ) denote the length of u -v shortest path w.r.t. the weight ℓ ∗ . (Set dist( u, v ) = + ∞ if there is no u -v path.) For each e ∈ E , define ∆ ∈ [0 , 1] E by

<!-- formula-not-decoded -->

Then, if p ∗ ∈ P is a shortest s -g path, then (22) holds. In fact, for any s -g path p ∈ P expressed as a sequence of ( s = v 0 , e 1 , v 1 , e 2 , v 2 , . . . , v h -1 , e h , g = v h ) , we have

<!-- formula-not-decoded -->

which implies Reg T ( p ∗ ) = E [ ∑ T t =1 ⟨ ∆ , p t ⟩ ] . Suppose π ∗ is chosen so that π ∗ ( v ) ∈ arg min e ∈ ∂ + v { ∆( e ) } . Then, ∆( e ) for all e ∈ E ′ = E \ Im( π ∗ ) if and only if the shortest v -g path is unique for all v ∈ V ∪ { s } .

Remark 4. An issue of the definition of ∆ in (23) is the requirement for a strong assumption that the v -g shortest path is unique for all v ∈ V ∪ { s } to ensure that ∆( e ) &gt; 0 for all e ∈ E ′ . We can relax this assumption by some alternative definitions of ∆ . When using the following definition, it suffices to assume that the s -g shortest path is unique: Define L ′ = max p ∈P { ∑ e ∈ E ′ p ( e ) } ≤ L . For e ∈ E ′ and k ∈ [ L ′ ] , define ˜ P ( e, k ) ⊆ P and ˜ ∆( e ) by

<!-- formula-not-decoded -->

Wethen have ∑ e ∈ E ′ ˜ ∆( e ) p ( e ) ≤ ∑ e ∈ E ′ ∆( e ) p ( e ) for all p ∈ P , which implies that the environment is in ( p ∗ , π ∗ , ˜ ∆) as well. Further, as we have p ∗ / ∈ ˜ P ( e, k ) for any e ∈ E ′ and k , we have

<!-- formula-not-decoded -->

for any e ∈ E ′ . Hence, this value is positive as long as the s -g shortest path is unique. We also have ˜ ∆( e ) ≥ min e ′ ∈ E ′ ∆( e ) for any e ∈ E ′ .

<!-- formula-not-decoded -->

Let q 0 ∈ Q be such that q 0 ( e ) ≥ 1 /m . For any p ∈ Q and ε ∈ [0 , 1] , set q

<!-- formula-not-decoded -->

Using Lemma 8 and standard analysis for FTRL (see, e.g., Exercise 28.12 of [Lattimore and Szepesvári, 2020]), we obtain:

<!-- formula-not-decoded -->

where D t ( · , · ) represents the Bregman divergence associated with ψ t defined by (17) or (18), and we set ψ t ( q ) = 0 for t = 0 as an exception.

## C.3.1 Analysis for Tsallis-entropy case

Theorem 4 (First part of Theorem 1) . Let p ∗ ∈ P be an arbitrary s -g path and suppose that π ∗ ∈ Π is consistent with p ∗ . Then the proposed algorithm with the Tsallis-entropy regularizer (17) achieves:

<!-- formula-not-decoded -->

Corollary 3 (First part of Corollary 1) . In the adversarial regime, the proposed algorithm with the Tsallis-entropy regularizer (17) achieves Reg T = O ( √ mLT + m log T ) . Further, if the environment is in a ( p ∗ , π ∗ , ∆ , C ) -self-bounding regime given in Definition 3, we then have Reg T ( p ∗ ) ≲ U + √ UC + m log T , where we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the following, we provide a proof of Theorem 4.

Lemma 9. When we use the Tsallis-entropy regularizer (17) , stability terms are bounded as

<!-- formula-not-decoded -->

Proof. To bound the stability term, we can apply Lemma 6 with ℓ = ̂ ℓ t ( e ) , x = q t ( e ) , η = η t , and β = 2 . In fact, we can verify that -̂ ℓ t ( e ) q t ( e ) ≤ q t ( e ) q t ( e -) ≤ 1 ≤ β 2 . Hence, from Lemma 6, we have

<!-- formula-not-decoded -->

where we used the fact that p t ( v ) = 0 implies p t ( e ) = 0 for e ∈ ∂ + v . Taking the conditional expectation w.r.t. p t given q t , we obtain:

<!-- formula-not-decoded -->

Further, as we have (1 -x ) ≤ 2(1 - √ x ) for x ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

By combining the above inequalities,

<!-- formula-not-decoded -->

The second inequality in (31) follows from the fact that q t ( e ) ≤ q t ( v ) for any e ∈ ∂ + v .

Lemma 10. Let p ∗ be a path consisting of V ∗ ∈ V and E ∗ ⊆ E . Suppose that q ∗ is given by (25) . For t ≥ 2 , if q ∗ is a path consisting of V ∗ ∈ V and E ∗ ⊆ E , penalty terms are bounded as

<!-- formula-not-decoded -->

In the case of t = 1 , the bound includes an O ( m log( m/ε )) term in addition to the above.

Proof. Suppose that t ≥ 2 . We note it follows from η t = 1 / √ t that 1 η t -1 -1 η t = Θ( η t ) . We hence have

<!-- formula-not-decoded -->

The equality in (32) follows from E \ E ∗ = ( ⋃ v ∈ V ∪{ s } ( ∂ + v \ π ∗ ( v ))) ∪ ( V \ V ∗ ) . In the case of t = 1 , the penalty term includes an additional term of

<!-- formula-not-decoded -->

which completes the proof.

Lemmas 9 and 10 combined with (26) immediately lead to (27). Given this, the next step to (28) follows from the following lemma:

Lemma 11. For any v ∈ V \ V ∗ and any q ∈ Q , we have

<!-- formula-not-decoded -->

Proof. As Q is a convex hull of P and both sides of (33) are linear in q , it suffices to show (33) for q ∈ P . Then, for any q ∈ P , if RHS of (33) is positive, then it is at least 1 , and hence (33) holds. Therefore, it suffices to show that RHS = 0 = ⇒ LHS = 0 for q ∈ P . Suppose q corresponds to the sequence of ( s = v 0 , e 1 , v 1 , . . . , e h , v h = g ) . If RHS = 0 , then q consists of V ∗ = { v ∗ 1 , . . . , v ∗ h ∗ = g } and E ∗ = { π ∗ ( v ∗ j ) } j =0 , 1 ,...,h ∗ . In fact, we can show this in induction in j , under the condition that ∑ e ∈ ∂ + v ′ \{ π ∗ ( v ′ ) } q ( e ) = 0 for all v ′ ∈ V ∗ ∪ { s } : q ( v ∗ j ) = 1 = ⇒ ∑ e ∈ ∂ + v ∗ j q ( e ) = 1 = ⇒ q ( π ∗ ( v ∗ j )) = 1 = ⇒ q ( v ∗ j +1 ) = 1 .

Proof of Theorem 4. We choose ε = 1 T ∈ [0 , 1] in the definition of q ∗ in (25). Then, Lemmas 9 and 10 combined with (26) lead to (27). The other inequality (28) follows from

<!-- formula-not-decoded -->

where we used Lemma 11 in the second inequality.

Proof of Corollary 3. As we have ∑ e ∈ E q ( e ) ≤ L and ∑ v ∈ V q ( v ) ≤ L for any q ∈ Q , from the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Combining Theorem 4 with this and ∑ T t =1 1 √ t ≲ √ T , we obtain Reg T ≲ √ mLT + m log T . By using the Cauchy-Schwarz inequality, we obtain the following:

<!-- formula-not-decoded -->

Similarly, we have have

<!-- formula-not-decoded -->

Hence, from Theorem 4, Jensen's inequality, and the assumption of self-bounding regime in Definition 3, we have

<!-- formula-not-decoded -->

which implies Reg ( p ∗ ) ≲ U + √ UC + m log T .

T

## C.3.2 Analysis for log-barrier case

In the case of the log-barrier regularizer, we have the following regret bounds:

Theorem 5 (Second part of Theorem 1) . The proposed algorithm with the log-barrier regularizer (18) achieves:

<!-- formula-not-decoded -->

Corollary 4 (Second part of Corollary 1) . In the adversarial regime, the proposed algorithm with the log-barrier regularizer (18) achieves

<!-- formula-not-decoded -->

Reg T ( p ∗ ) ≲ U + √ UC + m log T , where we define

<!-- formula-not-decoded -->

To show these, we use the following lemmas:

Lemma 12. When we use the log-barrier regularizer (18) , we have

<!-- formula-not-decoded -->

Proof. As we have -η t q t ( e ) ̂ ℓ t ( e ) ≤ η t q t ( e ) q t ( e -) ≤ η t ≤ 1 / 2 , we can use Lemma 7 to bound the stability terms:

<!-- formula-not-decoded -->

where we used the fact that q ( v ) = 0 implies q ( e ) = 0 for e ∈ ∂ + ( v ) and for any q ∈ Q .

Define ρ t ( e ) by

<!-- formula-not-decoded -->

Define q ∗ by (25) with ε = m/T . Then, from (26) and Lemma 12, we have

<!-- formula-not-decoded -->

where the last inequality follows from the setting of η t ( e ) :

<!-- formula-not-decoded -->

This completes the proof of Theorem 5.

Proof of Corollary 4. We can show (34) by using the Cauchy-Schwarz inequality, Jensen's inequality, and the fact that ∑ e p t ( e ) ≤ L . The other one can be shown via the following:

<!-- formula-not-decoded -->

Hence, by using the condition of the self-bounding constraint in Definition 3, we obtain

<!-- formula-not-decoded -->

This implies that Reg T ( p ∗ ) ≲ U + √ UC + m log T .

## C.4 Lower bound for stochastic environments

As our problem is a special case of the linear bandit problem, we can apply the lower bound given by [Lattimore and Szepesvari, 2017, Corollary 2], which is characterized by the following optimization problem:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

However, deriving an explicit expression for the optimal value of this optimization problem is not straightforward. In fact, we have not yet identified such an expression. Instead, in what follows, we present a lower bound by exploiting the specific structure inherent to this problem.

Consider stochastic environments specified by ℓ ∗ such that ⟨ ℓ ∗ , p ⟩ ∈ [3 / 8 , 5 / 8] for all p ∈ P . We suppose that c t ∈ { 0 , 1 } follows a Bernoulli distribution of parameter ⟨ ℓ, p t ⟩ . Suppose that ∆ is defined as (23). We then have ⟨ ℓ ∗ , p ⟩ -min p ∗ ∈P ⟨ ℓ ∗ , p ∗ ⟩ = ⟨ ∆ , p ⟩ . Denote

<!-- formula-not-decoded -->

Theorem 6. Consider an arbitrary consistent algorithm, i.e., assume that there exists ε ∈ (0 , 1) such that Reg T ≤ MT 1 -ε holds for any instances, where M &gt; 0 is a parameter independent of T . We then have

<!-- formula-not-decoded -->

Remark 5. If ∆ is given by (23), we have ¯ ∆( e ) = ∆( e ) for e ∈ ∂ + v where v ∈ V ∗ := { v ∈ V | dist( s, v ) + dist( v, g ) = dist( s, g ) } . In fact, for such an edge e , we have

<!-- formula-not-decoded -->

We hence have

<!-- formula-not-decoded -->

Proof of Theorem 6. Let p ∗ ∈ arg min p ∈P {⟨ ℓ ∗ , p ⟩} . As we have

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

̸

In the following, we evaluate the value of lim inf T →∞ 1 log T E [ ∑ T t =1 p t ( e ) ] for any fixed ˜ e ∈ E such that δ := ¯ ∆(˜ e ) &gt; 0 . Consider a modified environment given by ˜ ℓ such that ˜ ℓ (˜ e ) = ℓ ∗ (˜ e ) -2 δ and ˜ ℓ ( e ) = ℓ ∗ ( e ) for e = ˜ e . Then, as we have p ∗ (˜ e ) = 0 , it holds for any p ∈ P that

<!-- formula-not-decoded -->

where the inequality follows from the definition of δ = ¯ ∆(˜ e ) given in (36). Further, as there exists p ∈ P such that p ( e ) = 1 and ⟨ ∆ , p ⟩ = δ , we have

<!-- formula-not-decoded -->

Hence, from (39) and (40), the regret for the environment given by ˜ ℓ satisfies

<!-- formula-not-decoded -->

where ˜ E [ · ] represents the expected value when feedback is generated by an environment associated with ˜ ℓ . On the other hand, the regret for the environment given by ℓ ∗ satisfies

<!-- formula-not-decoded -->

Let TV denote the total variation distance between trajectories (( p t , c t )) T t =1 for environments with ℓ ∗ and ˜ ℓ . Then, as we have 1 T ∑ T t =1 p t (˜ e ) ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

We hence have

<!-- formula-not-decoded -->

where the last inequality follows from (41) and (42). Here, from the Bretagnolle-Huber inequality (e.g., [Canonne [2022], Corollary 4]) and the chain rule of the KL divergence, we have

<!-- formula-not-decoded -->

Combining the above inequalities, we obtain

<!-- formula-not-decoded -->

Consequently, we have

<!-- formula-not-decoded -->

for any ˜ e ∈ E such that ∆(˜ e ) &gt; 0 . By combining this with (38), we obtain (37).

## D MDPs with known transition

In this section, we analyze our algorithm for episodic MDPs with known transitions. Specifically, we prove Theorems 7 and 8, which together directly imply Corollary 2 in the main body. In addition, we provide our lower bound result in Theorem 9.

## D.1 Algorithm

The algorithm's construction is almost identical to that of the shortest path case. The unbiased loss estimator is defined in Lemma 2.

Proof of Lemma 2. For notational simplicity, we omit the conditioning in expectations throughout this proof. Fix an arbitrary s ∈ S k and a . We then have

<!-- formula-not-decoded -->

Similarly, for any fixed s ∈ S k , we have

<!-- formula-not-decoded -->

By combining the above two equalities, we obtain

<!-- formula-not-decoded -->

In the following, we denote

<!-- formula-not-decoded -->

By combining Lemmas 1 and 2, we obtain the following expression of the regret:

<!-- formula-not-decoded -->

where we denote q ∗ = q π ∗ . To upper bound the value of ∑ T t =1 〈 ̂ ℓ t , q t -q ∗ 〉 , we choose q t ∈ Q by using the following FTRL approach similarly to the case of online shortest path problem:

<!-- formula-not-decoded -->

̸

where

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

where ρ τ ∈ [0 , 1] will be defined later.

## D.2 Regret analysis

In our regret analysis, we use the following lemma:

Lemma 13 (First part of Lemma 4) . If ̂ ℓ t is given by as in Lemma 2, the expectation of ̂ ℓ ( s, a ) 2 taken w.r.t. the randomness of p satisfies

<!-- formula-not-decoded -->

Proof. For notational simplicity, we omit the subscript t throughout this proof. From the definition of ̂ ℓ , we have

<!-- formula-not-decoded -->

We also use q 0 and ˜ q ∗ , which are defined as follows in the analysis: For all s ∈ S and a ∈ A , let q s,a ∈ arg max q ′ ∈Q q ′ ( s, a ) . Define q 0 ∈ Q by q 0 = 1 | S || A | ∑ s,a q s,a . For q ∗ = q π ∗ ∈ Q and ε ∈ [0 , 1] , define ˜ q ∗ by

<!-- formula-not-decoded -->

Then, it holds for any q ∈ Q , s ∈ S and a ∈ A that

<!-- formula-not-decoded -->

From (43), by a similar way to (26), we can show that

<!-- formula-not-decoded -->

## D.2.1 Analysis for Tsallis-entropy case

Theorem 7. For any deterministic policy π ∗ ∈ Π , the proposed algorithm with the Tsallis-entropy regularizer (44) achieves:

Reg T ( π ∗ )

<!-- formula-not-decoded -->

̸

̸

̸

̸

In the proof of this theorem, we can bound the stability term by using the following lemma:

Lemma 14. When we use the Tsallis-entropy regularizer (17) , stability terms are bounded as

̸

<!-- formula-not-decoded -->

̸

̸

̸

This can be shown in a similar way as Lemma 9, by using Lemmas 6 and 13. Furthermore, the penalty term can be bounded in the same manner as done by Jin and Luo [2020], using their Lemma 6 with α = 0 . By combining these results, Theorem 7 can be established in the same way as Theorem 4.

## D.2.2 Analysis for log-barrier case

In the case of the log-barrier regularizer, we have the following regret bounds:

Theorem 8. For any deterministic policy π ∗ , the proposed algorithm with the log-barrier regularizer (45) achieves:

̸

<!-- formula-not-decoded -->

̸

To show these, we use the following lemmas:

Lemma 15. When we use the log-barrier regularizer (45) , we have

̸

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

where D ϕ is the Bregman divergence associated with ϕ ( x ) = -ln( x ) . As we have

<!-- formula-not-decoded -->

we can apply Lemma 7 to obtain the following:

<!-- formula-not-decoded -->

Define ρ t ( s, a ) by

<!-- formula-not-decoded -->

Then, Theorem 8 can be established in the same way as Theorem 5.

Lastly, results in Corollary 2 follows from Theorems 7 and 8 by an argument similar to that used for Corollaries 3 and 4.

## D.3 Lower bound for stochastic MDPs

Consider stochastic environment in which c t follows a Bernoulli distribution of parameter ⟨ ℓ ∗ , p t ⟩ , where we assume that ℓ ∗ : S × A → [0 , 1] satisfies ⟨ ℓ ∗ , p ⟩ ∈ [3 / 8 , 5 / 8] for any possible trajectories p and ℓ ∗ ( s L , a ) = 0 for all a ∈ A . Define ∆ : S × A → [0 , 1] by optimal Q function values:

<!-- formula-not-decoded -->

We then have min a ∈ A ∆( s, a ) = 0 for all s ∈ S and ∥ ∆ ∥ 1 ≤ 1 / 4 . Also, we have

<!-- formula-not-decoded -->

as V ∗ ( s L ) = ℓ ∗ ( s L , a ) = 0 for all a ∈ A . Denote

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also define

<!-- formula-not-decoded -->

Then, for any consistent algorithms, we have

<!-- formula-not-decoded -->

Theorem 9. Consider an arbitrary consistent algorithm, i.e., assume that there exists ε ∈ (0 , 1) such that Reg T ≤ MT 1 -ε holds for all instances, where M &gt; 0 is a parameter independent of T . Then for any MDP with ℓ ∗ : S × A → [0 , 1] satisfying ⟨ ℓ ∗ , p ⟩ ∈ [3 / 8 , 5 / 8] for any possible trajectories p and ℓ ∗ ( s L , a ) = 0 for all a ∈ A , we have

<!-- formula-not-decoded -->

Proof. Let q ∗ ∈ Q ∗ . As we have

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

In the following, we evaluate the value of lim inf T →∞ 1 log T E [ ∑ T t =1 q t ( s, a ) ] for any fixed ˜ s ∈ S ∗ and ˜ a ∈ A such that δ := ∆(˜ s, ˜ a ) &gt; 0 . Let q ∗ ∈ arg max q ∈Q ∗ { q (˜ s ) } and denote ¯ q = max q ∈Q ∗ { q (˜ s ) } = q ∗ (˜ s ) &gt; 0 . Then, from Lemma 16 below, there exists c ∈ (0 , δ ] such that ⟨ ∆ , q ⟩ ≥ c max { 0 , q (˜ s ) -¯ q } + δq (˜ s, ˜ a ) .

̸

Consider a modified environment given by ˜ ℓ such that ˜ ℓ (˜ s, ˜ a ) = ℓ ∗ (˜ s, ˜ a ) -δ -c/ 2 and ˜ ℓ ( s, a ) = ℓ ∗ ( s, a ) for ( s, a ) = (˜ s, ˜ a ) . Then, as we have q ∗ (˜ s, ˜ a ) = 0 , it holds for any q ∈ Q that

<!-- formula-not-decoded -->

Further, as there exists q ∈ Q such that 〈 ˜ ℓ, q -q ∗ 〉 = -c ¯ q 2 , we have

<!-- formula-not-decoded -->

In fact, such an occupancy measure q can be constructed from a corresponding policy ˜ π : S → A , by modifying a policy π ∗ : S → A corresponding to q ∗ ∈ Q ∗ such that q ∗ ( s ) = ¯ q . We set ˜ π ( s ) = π ∗ ( s ) for all s = ˜ s and set ˜ π (˜ s ) = ˜ a . An occupancy measure q corresponding to ˜ π satisfies q (˜ s, ˜ a ) = ¯ q and ⟨ ℓ ∗ , q ⟩ = ¯ qδ , which implies that 〈 ˜ ℓ, q 〉 = c ¯ q 2 . Hence, the regret for the environment given by ˜ ℓ satisfies

̸

<!-- formula-not-decoded -->

where ˜ E [ · ] represents the expected value when feedback is generated by an environment associated with ˜ ℓ . On the other hand, the regret for the environment given by ℓ ∗ satisfies

<!-- formula-not-decoded -->

Let TV denote the total variation distance between trajectories (( q t , p t , c t )) T t =1 for environments with ℓ ∗ and ˜ ℓ . Then, as we have 1 ¯ qT ∑ T t =1 min { ¯ q, q t (˜ s, ˜ a ) } ∈ [0 , 1] , we have

<!-- formula-not-decoded -->

We hence have

<!-- formula-not-decoded -->

where the last inequality follows from (55) and (56). Here, from the Bretagnolle-Huber inequality (e.g., [Canonne [2022], Corollary 4]) and the chain rule of the KL divergence, we have

<!-- formula-not-decoded -->

Combining the above inequalities and c ∈ (0 , δ ] , we obtain

<!-- formula-not-decoded -->

Consequently, we have

<!-- formula-not-decoded -->

for any ˜ s ∈ S ∗ and ˜ a ∈ A such that ∆(˜ s, ˜ a ) &gt; 0 . By combining this with (52), we obtain (51).

Lemma 16. Suppose that s ∈ S ∗ and denote ¯ q = max q ∈Q ∗ { q ( s ) } . Then, there exists c &gt; 0 such that the following holds for all q ∈ Q :

<!-- formula-not-decoded -->

Proof. Suppose that s ∈ S ∗ ∩ S k . Decompose ∆ as ∆ = ∆ &lt;k +∆ ≥ k , where

<!-- formula-not-decoded -->

.

We define q &lt;k and q ≥ k in the same way. Define f ( x ) = inf q ∈Q : q ( s )= x ⟨ ∆ , q ⟩ . We then have

<!-- formula-not-decoded -->

The last equality follows from the fact that, for any q ∈ Q (corresponding to π ∈ Π ) such that q ( s ) = x , there exists q ′ ∈ Q such that q ( s ) = x , ⟨ ∆ &lt;k , q ′ ⟩ = ⟨ ∆ &lt;k , q ⟩ , and ⟨ ∆ ≥ k , q ′ ⟩ = 0 . Such an occupancy measure q ′ can be constructed by a policy π ′ ∈ Π given as π ′ ( s, a ) = π ( s, a ) for s ∈ ⋃ k ′ &lt;k S k ′ and π ′ ( s, a ) = π ∗ ( s, a ) for s ∈ ⋃ k ′ ≥ k S k ′ .

Define x = min q ∈Q { q ( s ) } and ¯ x = max q ∈Q { q ( s ) } . We note that x ≤ ¯ q ≤ ¯ x and f ( x ) &lt; + ∞ if and only if x ≤ x ≤ ¯ x . As Q is a polytope, f ( x ) is a piecewise linear function in x , i.e., there exists a finite sequence of real numbers x 0 = x &lt; x 1 &lt; x 2 &lt; · · · &lt; x n = ¯ x ∈ R such that f ( x ) is an affine function in each interval [ x i , x i +1 ] . From the definition of Q ∗ and ¯ q , we have f ( x ) &gt; 0 for any x &gt; ¯ q . Indeed, if q ( s ) &gt; ¯ q then q / ∈ Q ∗ , which means that ⟨ ∆ , q ⟩ &gt; 0 . Hence, c ∈ R defined as

<!-- formula-not-decoded -->

is positive. (When ¯ q = ¯ x , i.e., when there is no x i &gt; ¯ q , we let c be an arbitrary positive number.) Further, as f ( x ) ≥ 0 for all x and f ( x ) is affine in each interval [ x i , x i +1 ] , we have f ( x ) ≥ c max { 0 , x -¯ q } for all x ≤ ¯ x . From this, we have

<!-- formula-not-decoded -->

## E Algorithm for MDPs with unknown transition

In this section, we present the details for our best-of-both-worlds algorithm for MDPs with unknown transitions. Similar to Jin et al. [2021], our algorithm proceeds in epochs. In each epoch, we execute FTRL using our novel loss estimator and the current empirical estimates of the transitions. At the end of the epoch, we update these empirical estimates. We refer the reader to Algorithm 1 for full details. In this section, we use E t [ · ] to denote the conditional expectation E [ ·|F t -1 ] , where F t -1 is the past filtration.

## E.1 Confidence set of the true transition

In this section, we use the same confidence sets used by prior works Jin et al. [2020, 2021]. For each epoch i , we define the empirical transition ¯ P i as:

<!-- formula-not-decoded -->

For each epoch i , we define the confidence width B i as follows:

<!-- formula-not-decoded -->

where δ is some confidence parameter.

For each epoch i , the confidence set P i of the true transition is defined as follows:

<!-- formula-not-decoded -->

As shown in Lemma 2 of Jin et al. [2020], the true transition P lies in the confidence P i for all epoch i with probability at least 1 -4 δ .

## E.2 Loss estimator and Regularizer

We begin by presenting our novel loss estimator:

<!-- formula-not-decoded -->

where u t ( s, a ) denotes the upper occupancy measure of ( s, a ) under policy π t , and is defined as

<!-- formula-not-decoded -->

where i ( t ) denotes the epoch to which round t belongs. Note that u t ( s, a ) can be efficiently computed using the COMP-UOB procedure proposed in [Jin et al., 2020].

To build intuition for why this estimator enables best-of-both-worlds guarantees, we now consider the corresponding loss estimator in the setting with known transitions.

<!-- formula-not-decoded -->

In the unknown transition case under semi-bandit feedback, Jin et al. [2021] considered the loss estimator ℓ t ( s,a ) · I t ( s,a ) u t ( s,a ) , which ensures that ℓ t ( s, a ) -E t [ ℓ t ( s,a ) · I t ( s,a ) u t ( s,a ) ] ≥ 0 whenever u t ( s, a ) ≥ q t ( s, a ) . This inequality plays a key role in establishing their best-of-both-worlds result.

In our setting, the role of ℓ t ( s, a ) is played by the pseudo-loss ¯ ℓ t ( s, a ) := E t [ ℓ q t ( s, a )] . When u t ( s, a ) ≥ q t ( s, a ) , we show an analogous inequality: ¯ ℓ t ( s, a ) -E t [ ℓ u t ( s, a )] ≥ 0 , which is similarly crucial for our analysis.

We begin by analyzing the pseudo-loss ¯ ℓ t ( s, a ) as follows:

<!-- formula-not-decoded -->

By Lemma 1, we have ⟨ ¯ ℓ t , q t -q ∗ ⟩ = ⟨ ℓ t , q t -q ∗ ⟩ , implying that the pseudo-loss ¯ ℓ t can indeed play the role of ℓ t in our setting.

Next, we compute the expectation of ℓ u t ( s, a ) as follows:

<!-- formula-not-decoded -->

To analyze this expression, observe that:

<!-- formula-not-decoded -->

and thus:

π

Q

t

(

s, a

)

-

V

π

t

(

s

) + 1

-

π

t

(

a

|

s

̸

)

≥

0

.

Now, under the condition that u t ( s ) ≥ q t ( s ) , we have q t ( s ) u t ( s ) ≤ 1 , and therefore:

<!-- formula-not-decoded -->

That is, ℓ u t ( s, a ) is an optimistic estimator of the pseudo-loss ¯ ℓ t ( s, a ) , which plays a crucial role in our regret analysis.

In addition to bounding the expectation, we also analyze the second moment of the loss estimator. This analysis is facilitated by the careful introduction of the term (1 -π t ( a | s )) into our loss estimator, which yields the following bound:

<!-- formula-not-decoded -->

Using this bound, we obtain a control on the stability term in the regret analysis:

<!-- formula-not-decoded -->

This upper bound plays a crucial role in enabling a regret analysis based on self-bounding terms. In particular, it allows us to bypass the loss-shifting technique employed in Jin et al. [2021], while still controlling the stability term.

To leverage this upper bound, we use the following regularizer in epoch i :

̸

<!-- formula-not-decoded -->

̸

where η t = 1 √ t -t i +1 and β = 1024 L . The log-barrier term is included to stabilize the updates, following the approach of Jin and Luo [2020].

## E.3 Main Result

Theorem 10. In the bandit feedback setting, Algorithm 1 with δ = 1 T 3 and ι = | S || A | T δ guarantees Reg T ( π ⋆ ) = ˜ O ( L | S | √ | A | T + | S || A | √ LT + L 2 | S | 3 | A | 2 ) and simultaneously Reg T ( π ⋆ ) = O ( U + √ UC + V ) under Condition (4) , where V = L 2 | S | 3 | A | 2 ln 2 ι and U is defined as

̸

̸

<!-- formula-not-decoded -->

We defer the proof of the above theorem to Appendix F.

## F Analysis of BOBW with unknown transitions

For any time-step t , let i ( t ) denote the epoch the time-step t is part of. Let E t [ · ] := E [ ·|F t -1 ] be the conditional expectation, where F t -1 is the past filtration. Recall the definition of ℓ u t and ℓ q t from Eq. (60) and Eq. (62) respectively. Also recall that ̂ ℓ t = ℓ u t -B i ( t ) . Let ¯ ℓ t be a pseudo-loss such that ¯ ℓ t ( s, a ) := E t [ ℓ q t ( s, a )] = Q π t ( s, a ) -V π t ( s ) .

Now we define the conditional expectation of ̂ ℓ t as follows:

<!-- formula-not-decoded -->

Definition 4. For any policy π , the estimated state-action and state value functions associated with ¯ P i ( t ) and loss function ˜ ℓ t are defined as:

<!-- formula-not-decoded -->

On the other hand, the true state-action and value functions are defined as:

<!-- formula-not-decoded -->

where P denotes the true transition function.

Moreover, we define pseudo state-action and value functions as follows:

<!-- formula-not-decoded -->

Let A be the event that P ∈ P i for all epochs i ≥ 1 . Moreover, we also define A i to be the event P ∈ P i .Note that the value of ✶ {A i } gets determined based on observations prior to epoch i only. Let ι = T |S||A| δ and let δ = 1 T 3 ∈ (0 , 1) .

We decompose the regret against policy π as follows:

<!-- formula-not-decoded -->

Note that, the second term (restated below) is controlled by the FTRL process.

<!-- formula-not-decoded -->

## F.1 Auxiliary lemmas

We often use the following lemma to handle the small-probability event A c while taking the expectation.

Lemma 17 (Jin et al. [2021]) . Suppose that a random variable X satisfies the following conditions:

- Conditioning on event E , X &lt; Y where Y &gt; 0 is another random variable;
- X &lt; C holds where C is another random variable which ensures E [ C |E c ] ≤ D for some fixed D ∈ R + .

Then, we have

<!-- formula-not-decoded -->

We next restate the performance difference lemma.

Lemma18 (Performance difference lemma) . Suppose ¯ ℓ is defined by ¯ ℓ ( s, a ) = Q π ( s, a ; ℓ ) -V π ( s ; ℓ ) for some π ∈ Π and for all s ∈ S and a ∈ A . We then have

<!-- formula-not-decoded -->

for any π ′ ∈ Π , s ∈ S and a ∈ A .

We immediately get following corollary.

Corollary 5. ∀ ( s, a ) ∈ S × A , we have -1 ≤ ¯ V π t ( s ) ≤ 1 and -1 ≤ ¯ Q π t ( s, a ) ≤ 1 .

We next state the following lemma.

Lemma 19. If event A holds, then ∑ s ′ ∈ S k ( s )+1 ( ¯ P i ( t ) ( s ′ | s, a ) -P ( s ′ | s, a ) ) ¯ V π t ( s ′ ) -B i ( t ) ( s, a ) ≤ 0 .

Proof. When B i ( t ) ( s, a ) = 2 , we have

<!-- formula-not-decoded -->

where the inequality follows from the fact -1 ≤ ¯ V π t ( s ′ ) ≤ 1 .

On the other hand, when ∑ s ′ ∈ S k ( s )+1 B i ( t ) ( s, a, s ′ ) = B i ( t ) ( s, a ) , we have

<!-- formula-not-decoded -->

where the second line follows from the definition of event A .

We next state the following proposition

Proposition 1. For all ( s, a ) ∈ S × A , we have 0 ≤ Q π t ( s, a ) -V π t ( s ) + 1 -π t ( a | s ) ≤ 2 .

Proof. As we have

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We next state the following proposition.

<!-- formula-not-decoded -->

Proof. The following holds by Proposition 1, given that the event A occurs:

<!-- formula-not-decoded -->

We next state the following proposition.

<!-- formula-not-decoded -->

Proof. Due to Proposition 2 and the total loss of any trajectory is between 0 and 1 , we have ˜ ℓ t ( s, a ) ≤ ¯ ℓ t ( s, a ) -B i ( t ) ( s, a ) ≤ Q π t ( s, a ) ≤ 1 . On the otherhand, due to Proposition 1:

<!-- formula-not-decoded -->

We next state the following lemma.

Lemma 20. If event A holds, the following holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We prove this result via a backward induction from layer L to layer 0 .

Base case: for s L , ˜ Q π t ( s, a ) = ¯ Q π t ( s, a ) = 0 holds always.

Specifically, we have:

Induction step: Assume that ˜ Q π t ( s, a ) ≤ Q π t ( s, a ) holds for all states s with k ( s ) &gt; h . Then, for any state s with k ( s ) = h , we have

<!-- formula-not-decoded -->

This concludes the induction.

The following lemma follows directly from Lemma C.1.2 in Jin et al. [2021].

Lemma 21 (Jin and Luo [2020]) . Algorithm 1 ensures u t ( s ) ≥ 1 | S | t for all t and s .

Lemma 22. Algorithm 1 ensures the following:

<!-- formula-not-decoded -->

We also have:

<!-- formula-not-decoded -->

Proof. Due to Lemma 21, we have the following:

<!-- formula-not-decoded -->

Next, we have:

<!-- formula-not-decoded -->

Lemma 23. Algorithm 1 ensures the following:

<!-- formula-not-decoded -->

Proof. Due to Eq. (64), we have the following:

<!-- formula-not-decoded -->

We immediately get the following corollary.

Corollary 6. Algorithm 1 ensures the following:

<!-- formula-not-decoded -->

̸

Let ϕ H ( q ) = -∑ s = s L ∑ a ∈ A √ q ( s, a ) and ϕ L ( q ) = -β ∑ s = s L ∑ a ∈ A ln q ( s, a ) . Recall that ϕ t ( q ) = ϕ H ( q ) + ϕ L ( q ) and β = 1024 L . Now we prove the following proposition.

̸

Proposition 4. If event A holds, then || ̂ ℓ t || ( ∇ 2 ϕ t ( ̂ q t )) -1 ≤ 1 8 .

Proof. We have the following:

̸

<!-- formula-not-decoded -->

̸

̸

We now state the following lemma, which follows from arguments identical to those in Lemma C.1.8 of Jin et al. [2021], and provides an upper bound for E [ T ∑ t =1 ∑ s = s L ∑ a ∈ A ̂ q t ( s, a ) · B i ( t ) ( s, a ) 2 ] .

Lemma 24. Algorithm 1 ensures the following:

̸

<!-- formula-not-decoded -->

Finally, we state the following lemma on the learning rates and the number of epochs.

Lemma 25 (Jin et al. [2021]) . According to the design of the learning rate η t = 1 √ t -t i ( t ) +1 , the following inequalities hold:

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, Algorithm 1 ensures that N ≤ 4 | S || A | (log T +1) where N is the number of epochs.

## F.2 Technical lemmas to analyze ESTREG

We defined the estimated regret in each epoch i as follows:

<!-- formula-not-decoded -->

Lemma 26. With β = 1024 L , for any epoch i , Algorithm 1 ensures

<!-- formula-not-decoded -->

for any policy π , and simultaneously

<!-- formula-not-decoded -->

for any deterministic policy π : S → A .

Proof. If the event A i does not hold, we have the following:

̸

<!-- formula-not-decoded -->

̸

Due to the second part of Lemma 22, we have the following:

<!-- formula-not-decoded -->

̸

Recall that ϕ H ( q ) = -∑ s = s L ∑ a ∈ A √ q ( s, a ) . Also recall that ̂ Q t ( s, a ) = ̂ ℓ t ( s, a ) + E s ′ ∼ ¯ P ( ·| s,a ) [ ̂ V t ( s ′ )] and ̂ V t ( s ) = E a ∼ π t ( ·| s ) [ ̂ Q t ( s, a )] (with ̂ V L ( s L ) = 0 ). Due to Proposition 4,

̸

̸

̸

̸

̸

̸

we get the following by using the same argument as [Jin and Luo, 2020, Lemma 5]:

̸

<!-- formula-not-decoded -->

Now, we condition on the event A i . Recall the definition of ℓ u t from Eq. (60). Now we have the following:

<!-- formula-not-decoded -->

where we get the last inequality as as u t ( s ) ≤ q t ( s ) and u t ( s, a ) ≤ q t ( s, a ) .

̂ As E t [( I t ( s, a ) -I t ( s ) π t ( a | s )) 2 ] = q ( s )(1 -π t ( a | s )) π 2 t ( a | s ) + q ( s ) π t ( a | s )(1 -π t ( a | s )) 2 and E t [ I t ( s )] = q t ( s ) , we have the following:

<!-- formula-not-decoded -->

We now proceed to prove Eq. (74) and Eq. (75).

̸

Proving Eq. (74) In this case, we consider the second term inside the minimum in Eq. (76), and derive a straightforward upper bound to ϕ H ( q ) -ϕ H ( ̂ q t ) using the Cauchy-Schwarz inequality, yielding ϕ H ( q ) -ϕ H ( ̂ q t ) ≤ ∑ s = s L ∑ a ∈ A √ ̂ q t ( s, a ) ≤ √ L | S || A | . This gives

̸

<!-- formula-not-decoded -->

̸

̸

̸

Therefore, by Lemma 17, Eq. (77), Eq. (78), and tower rule, we have for any policy π that,

̸

<!-- formula-not-decoded -->

̸

where the second step follows from ∑ s = s L ∑ a ∈ A √ ̂ q t ( s, a ) ≤ √ L | S || A | . This completes the proof of Eq. (74).

Proving Eq. (75) In this case, note that since π is a deterministic policy, it follows that

̸

<!-- formula-not-decoded -->

̸

By applying [Jin and Luo, 2020, Lemma 16] with α = 0 to upper bound the first term, and using [Jin and Luo, 2020, Lemma 19] to upper bound the second term, we arrive at

̸

<!-- formula-not-decoded -->

̸

̸

̸

Therefore, considering the Eq. (76) and using the inequalities 1 η t -1 η t -1 ≤ η t from Lemma 25 and ( a + b ) 2 ≤ 2 a 2 +2 b 2 , we have

̸

̸

<!-- formula-not-decoded -->

̸

Next, observe that for a deterministic policy π , we have:

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

Therefore, by Eq. (79), Lemma 17, Eq. (77), Eq. (78), Eq. (80) and tower rule, we have for any deterministic policy π : S → A that,

̸

̸

<!-- formula-not-decoded -->

## F.3 Adversarial regret guarantee

Recall from Eq. (65) that the regret decomposes as:

<!-- formula-not-decoded -->

We now analyse each term separately.

First, we have the following:

̸

<!-- formula-not-decoded -->

where the second equality follows from Eq. (64) and ¯ ℓ t ( s, a ) = Q π t ( s, a ) -V π t ( s ) .

Due to the above equality, proposition 1, Lemma 17, the fact that ̂ q t ( s, a ) ≤ u t ( s, a ) under event A and analysis of ERR 1 from Appendix C.2 Jin et al. [2021], we get the following:

<!-- formula-not-decoded -->

Next, we have the following due to Lemma 20, Lemma 17, Corollary 5, and Corollary 6:

<!-- formula-not-decoded -->

According to Eq. (74) of Lemma 26, we have

<!-- formula-not-decoded -->

̸

̸

̸

Finally, by combining the bounds for ERR 1 , ERR 2 , and ESTREG, we obtain:

<!-- formula-not-decoded -->

## F.4 Stochastic regret guarantee

## F.4.1 Self-bounding terms and related lemma

In this section, we adopt the definition of self-bounding terms and the related lemmas from Jin et al. [2021].

Definition 5 (Self-bounding Terms) . For some mapping π ⋆ : S → A , define the following:

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

Lemma 27. Suppose Condition (4) holds. Then we have for any α ∈ R + ,

̸

<!-- formula-not-decoded -->

̸

Lemma 28. Suppose Condition (4) holds. Then we have for any β ∈ R + ,

<!-- formula-not-decoded -->

Lemma 29. Suppose Condition (4) holds. Then we have for any α, β ∈ R + ,

̸

<!-- formula-not-decoded -->

̸

Lemma 30. Suppose Condition (4) holds. Then we have for any β ∈ R + ,

<!-- formula-not-decoded -->

Lemma 31. Suppose Condition (4) holds. Then we have for any α ∈ R + ,

̸

<!-- formula-not-decoded -->

̸

Lemma 32. Suppose Condition (4) holds. Then we have for any β ∈ R + ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.4.2 Proof for the stochastic world

Similarly to the proof in Appendix C.3 of Jin et al. [2021], we decompose the sum of ERR 1 and ERR 2 into four terms ERRSUB, ERROPT, OCCDIFF and BIAS:

̸

̸

<!-- formula-not-decoded -->

where ¯ E π t is defined as

<!-- formula-not-decoded -->

We now begin by upper bounding E [ OCCDIFF ] . First observe that we have the following:

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

Under event A , we further have ∑ T t =1 ∑ s = s L ∑ a = π ⋆ ( s ) | q t ( s, a ) -̂ q t ( s, a ) | · ∣ ∣ ∣ ˜ Q π ⋆ t ( s, a ) -˜ V π ⋆ t ( s ) ∣ ∣ ∣ ≤ 5 L ∑ T t =1 ∑ s = s L ∑ a = π ⋆ ( s ) | q t ( s, a ) -̂ q t ( s, a ) | as -3 ≤ ˜ ℓ t ( s, a ) ≤ 1 under event A . If event A doesn't hold, we have ∑ T t =1 ∑ s = s L ∑ a = π ⋆ ( s ) | q t ( s, a ) -̂ q t ( s, a ) | · ∣ ∣ ∣ ˜ Q π ⋆ t ( s, a ) -˜ V π ⋆ t ( s ) ∣ ∣ ∣ ≤ 12 L | S | 2 | A | T 2 due to Corollary 6. Hence due to Lemma 17, we have the following

̸

̸

̸

<!-- formula-not-decoded -->

̸

where the last line follows from the definition of Q 3 and Lemma D.3.10 of Jin et al. [2021].

̸

̸

̸

̸

̸

̸

̸

̸

Next, due to Lemma 17, Lemma 20, and Corollary 6, we have E [ BIAS ] ≤ δ · O ( L | S | 2 AT 2 ) . The first two terms ERRSUB and ERROPT are bounded differently. First, under event A , we have

<!-- formula-not-decoded -->

where the last line uses the definition of the event A along with the fact that q t ( s, a ) ≤ u t ( s, a ) and -3 ≤ ˜ ℓ t ( s, a ) ≤ 1 under this event. Next, observe that the range of ¯ E π t is O ( L | S | t ) , as established by Proposition 1 and Corollary 6, which implies that the range of both ERRSUB and ERROPT is O ( L 2 | S | T 2 ) . Thus, it suffices to add a term of order O ( δ · L 2 | S | T 2 ) to account for the event A c .

By the exact same analysis as in Appendix C.3 and Appendix B.2 of Jin et al. [2021] and the fact that | ˜ V ( s ) | ≤ O ( L ) under event A , we have the following:

<!-- formula-not-decoded -->

We are now left with bounding ESTREG using self-bounding terms.

Term ESTREG Based on Eq. (75) from Lemma 26, summing over all epochs gives the following upper bound for E [ ESTREG ] :

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

By the exact same analysis as in Appendix C.3 of Jin et al. [2021], we bound the first term as follows:

̸

<!-- formula-not-decoded -->

̸

Again by using the exact same analysis as in Appendix C.3 of Jin et al. [2021], we bound the second term as follows:

̸

̸

<!-- formula-not-decoded -->

̸

Thus, we obtain the final bound on E [ ESTREG ] :

<!-- formula-not-decoded -->

Recall that δ = 1 /T 3 and ι = | S || A | T δ . Finally, by combining the bounds of each term, we finally have

<!-- formula-not-decoded -->

Using the self-bounding lemmas (27-32) and the exact same analysis as in Appendix C.3 of Jin et al. [2021], we get Reg T ( π ⋆ ) is bounded by O ( U + √ UC + V ) where V = L 2 | S | 3 | A | 2 ln 2 ι and U is defined as

̸

<!-- formula-not-decoded -->

̸

This completes the entire proof.
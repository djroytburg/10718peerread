## Non-Stationary Structural Causal Bandits

Yeahoon Kwon 1 Yesong Choe 1 Soungmin Park 1 Neil Dhir 2 ∗ Sanghack Lee 1 ∗ 1 Graduate School of Data Science, Seoul National University 2 Focused Energy Inc. {dataofyou, yesong, tjdals0653, sanghack}@snu.ac.kr neil.dhir@focused-energy.co

## Abstract

We study the problem of sequential decision-making in environments governed by evolving causal mechanisms. Prior work on structural causal bandits-formulations that integrate causal graphs into multi-armed bandit problems to guide intervention selection-has shown that leveraging the causal structure can reduce unnecessary interventions by identifying possibly-optimal minimal intervention sets (POMISs). However, such formulations fall short in dynamic settings where reward distributions may vary over time, due to their static-and thus myopic-nature focuses on immediate rewards and overlooks the long-term effects of interventions. In this work, we propose a non-stationary structural causal bandit framework that leverages temporal structural causal models to capture evolving dynamics over time. We characterize how interventions propagate over time by developing graphical tools and assumptions, which form the basis for identifying non-myopic intervention strategies. Within this framework, we devise POMIS + , which captures the existence of variables that contribute to maximizing both immediate and long-term rewards. Our framework provides a principled way to reason about temporally-aware interventions by explicitly modeling information propagation across time. Empirical results validate the effectiveness of our approach, demonstrating improved performance over myopic baselines.

## 1 Introduction

The Multi-Armed Bandit (MAB) problem is a classic decision-making scenario where an agent sequentially selects actions (arms), each with an unknown but fixed reward distribution, to maximize cumulative reward [Sutton and Barto, 2018]. A common assumption in MAB formulations is that arms are independent, meaning that the reward distribution of one arm does not depend on another [Lai and Robbins, 1985, Auer et al., 2002a]. However, in many real-world settings, actions are not independent-hidden factors may simultaneously influence both the choice of action and the observed reward, introducing unobserved confounders (UCs). Bareinboim et al. [2015], Forney et al. [2017], Lattimore et al. [2016], Zhang and Bareinboim [2017] demonstrate that in the presence of UCs, standard bandit algorithms guarantees-such as convergence to the optimal arm and sublinear regret-no longer hold as observed rewards may not accurately reflect the true effect of actions. To address this, Lee and Bareinboim [2018] use structural causal models (SCMs) [Pearl, 2009] and causal diagrams to identify which variables an agent should intervene on to learn an optimal policy.

In many real-world applications, the reward distribution associated with an action is not fixed but changes over time due to shifting contexts, user behavior or environmental factors. Such settings motivate the study of non-stationary multi-armed bandit (NS-MAB) problems, which extend the classical formulation to handle temporal variation in reward dynamics. This dynamic formulation requires agents to continuously adapt to evolving environments by incorporating temporal information

∗ Corresponding authors

into their decision-making process. In particular, it relaxes the assumption of fixed reward distributions that underpins the classical MAB setting. While prior approaches to NS-MAB address temporal shifts in reward through statistical modeling-such as bounded change models [Auer et al., 2002b] or recharging payoffs [Papadigenopoulos et al., 2022]-they typically model non-stationarity in purely statistical terms, overlooking the underlying causal mechanisms responsible for reward shifts. Yet, in many sequential decision-making problems, such shifts are driven by latent and evolving causal mechanisms. In such cases, ignoring the structural causes behind the reward changes can limit an agent's ability to make informed interventions. In contrast, we frame non-stationarity through the lens of SCMs, allowing us to reason about how interventions propagate over time. The details are provided in Appendix E.

To our knowledge, the non-stationary MAB problem has not yet been studied through the causal lens. As such one could consider adapting existing causal methods for stationary settings. In particular, Lee and Bareinboim [2019] propose a latent projection-based method to identify possibly-optimal arms under partial causal knowledge (as detailed in Appendix J.1). However, that approach does not explicitly account for temporal order, treating the non-stationary process as if it were stationary. As a result, it fails to reveal which arms influence which rewards over time. This matters because, in non-stationary environments, identifying truly optimal strategies-those informed by both causal structure and temporal dynamics-is crucial. It enables more effective and timely interventions in real-world settings such as healthcare, education, and resource allocation, where decisions must often be made under uncertainty and limited budgets. Intervening without regard to temporal structure may appear optimal in the short term but lead to sub-optimal cumulative outcomes, ultimately limiting long-term performance. By explicitly modeling temporal structure, our approach overcomes this by capturing how causal effects unfold over time. This enables non-myopic strategies that align with long-term reward maximization, which is essential for planning in dynamic environments.

In this work, we approach the NS-MAB problem through the causal lens by introducing a structural framework that captures temporal dynamics via a rolled-out causal graph 2 (§3). We show how prior methods like POMIS fail to account for shifting causal structures over time and propose POMIS + that identifies non-myopic intervention strategies with theoretical guarantees (§5). We provide a graphical formulation of POMIS + that characterizes how interventions propagate across time, which supports systematic identification of non-myopic optimal interventions (§6). Building on this, we develop an efficient algorithm for computing intervention sequences using these graphical structures (§7). We empirically evaluate POMIS + across three non-stationary tasks, demonstrating its superiority over the myopic baseline in regret and optimal arm selection by capturing long-term causal effects through temporally-aware interventions (§8). Appendix A outlines the paper's structure.

Contributions First, we propose a framework-non-stationary structural causal bandit (NS-SCMMAB)-that combines the nature of NS-MAB with the underlying SCMs. Our framework models temporal dynamics not only in the reward distributions but also in the underlying causal structure, capturing how interventions propagate over time. This formulation enables temporally-aware decisionmaking, where an intervention at time t may have effects on both immediate and future outcomes. Second, to support this, we devise POMIS + , which extends the original POMIS by incorporating variables from future time steps to better evaluate the long-term value of interventions. Third, we theoretically establish that a partial assignment to preceding variables can strictly improve the expected reward in the temporal manner. Finally, we design an algorithm that identifies such nonmyopic intervention sequences and demonstrates its superiority over the myopic strategy through experiments across a range of non-stationary environments.

## 2 Preliminaries

We introduce notation from causal inference to understand multi-outcome causal MAB problems. Capital letters denote a single variable, and the domain of X is denoted D ( X ) . Bold capital X = { X 1 , . . . , X n } represents a set of variables. Additionally, lowercase x ∈ D ( X ) represents a value of variable X , and the set of values is described as bold lowercase x ∈ D ( X ) = × X ∈ X D ( X ) . We denote x [ W ] as the values of x corresponding to the intersection W ∩ X .

2 A rolled-out causal graph is a time-unfolded representation of an SCM that explicitly captures temporal dependencies across time steps. Similar constructions appear in causal modeling over time, see e.g., Koller and Friedman [2009, p. 203].

Figure 1: (a) shows the causal diagram for stationary SCM-MAB. The graph notation for a rolled-out version of the stationary MAB is shown in (b) indexing each round by its corresponding time step.

<!-- image -->

We adopt the structural causal model (SCM) [Pearl, 2009] framework. An SCM M is a tuple ⟨ U , V , F , P ( U ) ⟩ , where V is a set of endogenous variables and U is a set of exogenous variables. Each f V ∈ F is a function that determines each endogenous variable V . That is, V ← f V ( pa V , u V ) where Pa V ⊆ V \{ V } and U V ⊆ U . P ( U ) is a joint distribution over the exogenous variables. The SCM induces a causal diagram G , which includes directed edges encoding functional relationships between endogenous variables, and bi-directed edges encoding UCs. We adopt family relationships Pa ( · ) G , Ch ( · ) G , An ( · ) G , and De ( · ) G to denote parents, children, ancestors and descendants of a given variable where ancestors and descendants include its argument.

We consider a discrete-time setting where each time step t ∈ { 1 , . . . , T } (abbreviated t ∈ [ T ] ) corresponds to a distinct point in time. We denote by V t ⊆ V a set of variables at each time step t , focusing on the specific variables indexed by the subscript t . Let U t ⊆ U be the set of exogenous variables, including exogenous variable U V t for every V t ∈ V t and unobserved confounders for every V i t , V j t ∈ V t if U V i t and U V j t are correlated. V t&lt; ⊂ V is the set of variables whose time index is strictly greater than t . Let X t ⊆ V t \{ Y t } denote the set of manipulative variables, where Y t is the outcome variable. Then, let N ⊆ V \ X t denote the set of non-manipulative variables. We use blackboard bold letter V = { V 1 , . . . , V T } to represent the collection of all time-indexed variables throughout the sequence and use F = {F 1 , . . . , F T } to denote the collection of all time-specific functions.

A vertex-induced subgraph is represented by G [ V ′ ] where V ′ ⊆ V . Given a causal diagram G , a time-specific subgraph can be constructed as G [ ⋃ t + τ -1 i = t V i ] , where t denotes the starting time step and τ represents the length of the time window. In particular, when τ = 1 , we refer to this graph as a 'time slice', denoted by G [ V t ] . The probability of Y = y , when variables X are fixed to x , is denoted P ( y | do( X = x )) using the do-operator-an intervened probability. The graphical representation of the intervention is denoted G X ; a mutilated graph where the incoming edges onto X are removed.

## 3 Non-Stationary Multi-Armed Bandit as a Structural Causal Model

Wedefine the NS-SCM-MAB and identify the problems that arise when an existing stationary SCMMAB solution is applied to the NS-MAB problem by reviewing the SCM-MAB [Lee and Bareinboim, 2018]. Let M = ⟨ U , V , F , P ( U ) ⟩ be an SCM, and Y ∈ V be a reward variable, where D ( Y ) ⊆ R . The arms of the bandits are defined by the possible values x of the manipulative variable set X , where X ⊆ V \ { Y } , and x ∈ D ( X ) . Each arm is associated with a reward distribution P ( Y | do( X = x )) . The expected reward of an

Figure 2: SCM with non-stationary structure.

<!-- image -->

arm is denoted by E [ Y | do( X = x )] , written as µ x . The optimal value of x for the expected reward is denoted by x ∗ , i.e., x ∗ = argmax x ∈ D ( X ) µ x . For clarity, we represent the information given to an agent interacting with an SCM-MAB as J G , Y K . We now discuss time-expanded notation. When we expand it to the time horizon, for every t ∈ [ T ] , the causal structure among time-indexed variables is identical across all time steps (illustrated in Fig. 1(b)). The arms of the bandits are defined by the possible values x t of the set X t ⊆ V t \ { Y t } and x t ∈ D ( X t ) . The reward distribution is

defined P ( Y t | do( X t = x t ) , 1 t&gt; 1 · I 1: t -1 ) where I 1: t -1 = do( { X i = x i } t -1 i =1 ) denotes previous interventions and 1 t&gt; 1 is the indicator function. We denote E [ Y t | do( X t = x t ) , 1 t&gt; 1 · I 1: t -1 ] as µ x t ,I 1: t -1 , abbreviated as µ x t . The optimal value for X t is denoted x ∗ t .

̸

Non-stationarity in structural causal models Before defining non-stationarity , we clarify our understanding of stationarity . In the context of bandits, stationarity means that the reward distribution induced from pulling one arm (strategy) among all possible arms remains constant across all time steps t ∈ [ T ] . Interpreted from the perspective of the SCM-MAB, this implies that the distribution over the outcome variable Y -under interventions on any variables in the set of action nodes X ⊆ V \ Y -remains unchanged, regardless of earlier interventions, as discussed in Appendix D. Formally, we can express this as follows: for every pair of time steps t, t ′ ∈ [ T ] with t = t ′ , a reward distribution P ( y t | do( x t ) , 1 t&gt; 1 · I 1: t -1 ) = P ( y t ′ | do( x t ′ ) , 1 t ′ &gt; 1 · I 1: t ′ -1 ) . From this point of view, we introduce the NS-SCM-MAB where the reward distribution evolves over time.

Definition 3.1 (Non-stationary SCM-MAB) . Given a causal diagram G , let M = ⟨ U , V , F , P ( U ) ⟩ be an SCM and Y ⊂ V be a set of reward variables Y = ⋃ T i =0 { Y i } , where T &gt; 1 . An SCM-MAB ⟨M , Y ⟩ is said to be non-stationary, for some t, t ′ ∈ [ T ] with t &lt; t ′ , if the reward distribution

̸

<!-- formula-not-decoded -->

The disparity between reward distributions is referred to as a reward distribution shift .

The NS-SCM-MAB addresses the problem of arm selection in environments where the interventional distribution over reward variables may shift over time. This is illustrated in Fig. 2. As shown in the diagram, the value from the previous time step ( t ) can influence the subsequent time step ( t ′ ), as indicated by the red edges. This temporal influence causes the reward distribution to change over time, reflecting the non-stationary nature of the underlying SCM. To understand this setting from the agent's perspective, it is important to recognize that the topological ordering of variable generation is subordinated to the temporal ordering-variables associated with earlier time steps are always instantiated before those at later time steps. Based on this temporal structure, we define the notion of a temporal model within an SCM. A temporal model is an SCM that captures the time-specific causal mechanisms as perceived by the agent at a given time step.

̸

Definition 3.2 (Temporal Model) . Let M = ⟨ U , V , F , P ( U ) ⟩ be an SCM. For some t ∈ [ T ] with T &gt; 1 , we define values v ⋆ t and u ⋆ t , which correspond to the values of V ⋆ t = Pa ( V t ) \ V t and the set of correlated variables U ⋆ t between V t ∈ V t and V t ′ ∈ V t ′ for every t = t ′ , respectively. A temporal model M t | v ⋆ t , u ⋆ t is defined as ⟨ U t , V t , F t , P ( U t | u ⋆ t ) ⟩ where each f V t ∈ F t is a function that determines V t given predetermined values v ⋆ t and u ⋆ t . That is, V t ← f V t ( pa V t [ V t ] ∪ v ⋆ t , u V t ∪ u ⋆ t ) where Pa V t ⊆ V \ { V t } and U V t ⊆ U t .

Concisely, we denote M t | v ⋆ t , u ⋆ t by M t . Since the agent can only manipulate variables within the current time step, the temporal model is regarded as an underlying mechanism from the agent's perspective for each time step. The key point is that the mechanism may vary across time steps, which arises from information propagation , resulting in a reward distribution shift. A myopic agent, however, fails to adapt to this change and follows a strategy which chooses actions based on the current information only.

̸

Non-stationarity caused by information propagation We introduce the concept of 'information propagation' which induces the reward distribution shift between the two subsequent temporal models, and examine how it arises from the perspective of temporal models. To illustrate this, we consider two temporal models M t = ⟨ U t , V t , F t , P ( U t | u ⋆ t ) ⟩ and M t +1 = ⟨ U t +1 , V t +1 , F t +1 , P ( U t +1 | u ⋆ t +1 ) ⟩ . To isolate the effect of structural changes, we fix the exogenous distributions by assuming P ( U t | u ⋆ t ) = P ( U t +1 | u ⋆ t +1 ) and P ( u ⋆ t ) = P ( u ⋆ t +1 ) . The agent identifies that the data generation for M t +1 occurs only after the data generation governed by M t . Therefore, the agent at time t can neither observe the values of variables at a future time nor intervene in advance on future variables. Under the environment, f V t ∈ F t = f V t +1 ∈ F t +1 where F t , F t +1 ∈ F implies that the distribution induced from the SCM-MAB ⟨M t , Y t ⟩ may differ from the distribution from ⟨M t +1 , Y t +1 ⟩ . Such function inequality arises due to predetermined values v ⋆ t +1 ∈ D ( V ⋆ t +1 ) where V ⋆ t +1 = Pa V t +1 \ V t +1 , which is determined at time step t . As a result, information propagation refers to the process by which the predetermined values, generated within a temporal model M t , influence the generation of a value in another temporal model M t +1 .

To understand the concept better, we examine a simple case of two subsequent temporal models depicted in Fig. 2. At time step t , the value of X t is generated from the function f X t = U X t and the rest of the values are generated according to the topological (causal) order. At time step t ′ , the function generating X t ′ is f X t ′ = U X t ′ ⊕ z t , where z t is the value of Z t generated at time step t . Thus, at time step t ′ , the function f X t ′ utilizes z t , treating it as a conditional parameter (i.e., constant at t ′ ) to generate the value of X t ′ . In this setting, the reward distribution of the arms may change over time, introducing additional complexity and challenges.

Graph representation of the NS-SCM-MAB The graph representation of the NS-SCM-MAB captures the dynamics of the underlying non-stationary structure and the dependencies between variables of different time slices. Each time slice corresponds to a set of variables V t ∈ V and the relationships between these slices are depicted by edges, which we elaborate on in the sequel, representing the information propagation across time. To establish the conditional independence structure between the causal diagrams of the NS-SCM-MAB and its probability distribution, we assume the time-slice Markov property as a temporal extension of the Markov property.

Assumption 3.1 (Time-slice Markov) . Given a causal diagram G and a probability distribution P relative to G , for every time step t ∈ [ T ] , if ( V t ⊥ ⊥ V &lt;t -1 | V t -1 ) G where V t ∈ V , then P is said to be time-slice Markov relative to G .

Assumption 3.1 illustrates that under non-stationary dynamics, the SCM-MAB exhibits a first-order Markovian structure where each time slice depends only on the immediately preceding one.

̸

In addition, the graphical structure and topology of each time slice are identical across all time steps; that is, G [ V t ] = G [ V t ′ ] for all t = t ′ ∈ [ T ] . This reflects the bandit nature of our setting, where each temporal model corresponds to a repeated instance of the same underlying causal structure. Given any two time slices G [ V t ] and G [ V t ′ ] , this property implies stationarity between the corresponding temporal model MABs ⟨M t , Y t ⟩ and ⟨M t ′ , Y t ′ ⟩ , provided that no information propagation occurs across time steps, as illustrated in Fig. 1(b).

If there exists at least one edge between G [ V t ] and G [ V t ′ ] , where V t ∈ V t and V t ′ ∈ V t ′ , this edge is represented as a transition edge ( V t , V t ′ ) . The edge indicates that information propagation from a temporal model M t to another M t ′ , conforming to the two subsequent time slices G [ V t ] and G [ V t ′ ] , occurs at every time step. Notably, Assumption 3.1 implies that transition edges may exist only between consecutive time slices (i.e., first-order), thereby prohibiting connections between non-adjacent time slices.

## 4 Motivating Example

The primary challenge we address is identifying action sequences that maximize the expected reward under the NS-SCM-MAB setting. We first contemplate the solution of the stationary SCM-MAB problem given a causal diagram G to understand the non-stationary causal bandit. In the stationary MABsetting, choosing a set of arms can be interpreted as identifying a set of nodes Z from V \ { Y } that maximize the expected reward for outcome variable Y , formally written as

<!-- formula-not-decoded -->

Previously, Lee and Bareinboim [2018] proposed a do-calculus -based approach to exclude intervention sets that may induce sub-optimal expected rewards, ultimately identifying possibly optimal minimal intervention sets (POMISs) [Lee and Bareinboim, 2018, Def. 2]. We demonstrate, however, that this approach may itself lead to sub-optimality in the NS-SCM-MAB setting.

In the non-stationary setting, unlike the stationary setting where the agent only computes a POMISs once, the agent explores the search space at every time step to identify the POMISs for each timeindexed reward variable. For instance, in Fig. 3(a), we may consider the process by which an agent selects an intervention set at each time step. We can identify the POMISs by utilizing the concepts of Minimal UC-Territory (MUCT) [Lee and Bareinboim, 2018, Def. 3] and Interventional Border (IB) [Lee and Bareinboim, 2018, Def. 4], which characterize the POMISs graphically.

At time step t = 1 , from the agent's perspective, the set of possible exploration sets is {∅ , { X 1 } , { Z 1 } , { X 1 , Z 1 }} . We use an algorithm enumerating all POMISs [Lee and Bareinboim,

Figure 3: (a) Non-stationary causal diagram G with t ∈ { 1 , 2 , 3 } . (b-d) illustrate examples of MUCT/IB based on the agent's movement at each time step. (e) represents the agent's non-myopic intervention strategy at t = 2 (red box). (f) depicts a tree diagram of all possible scenarios when action selection starts with { Z 1 } (red text denotes POMIS + ).

<!-- image -->

2018, Alg. 1] with respect to J G [ V 1 ] , Y 1 K . The algorithm generates IB ( G [ V 1 ] , Y 1 ) = ∅ and IB ( G [ V 1 ] X 1 , Y 1 ) = { Z 1 } from MUCT ( G [ V 1 ] , Y 1 ) = { X 1 , Z 1 , Y 1 } and MUCT ( G [ V 1 ] X 1 , Y 1 ) = { Y 1 } . These are illustrated in Fig. 3(b) and Fig. 3(c), with the MUCT shaded in red and the IB shaded in blue. Thus, we get POMISs with respect to J G [ V 1 ] , Y 1 K -i.e., POMIS 1 = {∅ , { Z 1 }} . As a result, following the same procedure at each time step t = 1 , 2 , 3 , the agent obtains the following POMISs:

<!-- formula-not-decoded -->

Up to these time steps, the agent has identified the intervention set within its observable range at each time step. We next examine whether the identified intervention sets remain valid over an extended time horizon under the non-stationary setting. As illustrated with shaded blue in Fig. 3(d), MUCT ( G [ V 1 ∪ V 2 ] , Y 2 ) induces IB ( G [ V 1 ∪ V 2 ] , Y 2 ) = { X 1 } . Consequently, under the broader causal diagram G [ V 1 ∪ V 2 ] , we deduce that POMIS 2 = {{ X 1 } , { Z 2 }} . By applying the same procedure to J G [ V 1 ∪ V 2 ∪ V 3 ] , Y 3 K , we obtain POMIS 3 = {{ X 2 } , { Z 3 }} . Eventually, we obtain the following POMISs with a broader horizon:

<!-- formula-not-decoded -->

This differs from the previously computed myopic evaluations. These discrepancies highlight that variables such as { X 1 } or { X 2 } , excluded under myopic evaluations, may be optimal when viewed from a broader temporal perspective-while seemingly valid options like ∅ become sub-optimal.

Therefore, the agent should have intervened on { X 1 } and { X 2 } in advance because those variables are not manipulable for each current time step; { X 1 } for t = 2 and { X 2 } for t = 3 . However, the variables may not have been considered optimal for a reward variable at an advanced time step. This is supported by the inequality below:

<!-- formula-not-decoded -->

which is derived from µ x ∗ 2 = ∑ z 2 µ z 2 P ( z 2 | do( x ∗ 2 )) ≤ ∑ z 2 µ z ∗ 2 P ( z 2 | do( x ∗ 2 )) = µ z ∗ 2 given J G [ V 2 ] , Y 2 K . This inequality shows that the expected reward for intervening on { X 2 } is strictly dominated by that of { Z 2 } , making { X 2 } a suboptimal intervention for Y 2 . Nonetheless, this does not imply that { X 2 } should be excluded from the intervention set. Instead, it suggests that { X 2 } and { Z 2 } must be jointly intervened upon (i.e., { X 2 , Z 2 } in Fig. 3(e)): while { Z 2 } is required to optimize the immediate reward Y 2 , { X 2 } is necessary to influence the future reward Y 3 . Importantly, intervening on { X 2 } does not block the causal influence of { Z 2 } on Y 2 , thereby allowing both short-term and long-term rewards to be considered simultaneously. Similarly, for t = 1 , the

intervention set is assembled as { X 1 , Z 1 } . Finally, we obtain the set of intervention sequences: { ( { Z 1 } , { Z 2 } , { Z 3 } ) , ( { Z 1 } , { X 2 , Z 2 } , ∅ ) , ( ∅ , { Z 2 } , { Z 3 } ) , ( ∅ , { Z 2 } , { X 3 , Z 3 } ) , ( ∅ , { X 2 , Z 2 } , ∅ ) , ( { X 1 , Z 1 } , ∅ , { Z 3 } ) } -the details of which are formalized in Def. 5.1. We can depict the tree structure of these sequences as shown in Fig. 3(f), where each branch illustrates a valid path if Z 1 is intervened at the first time. These combinations will be discussed in detail in §5.

Building on the above example, we formally define the concept of POMIS for the NS-SCM-MAB and propose a graphical method to capture its dynamic evolution while conserving optimality.

## 5 Conservation of Optimality

We examine theoretical guarantees for identifying optimal intervention sets under the NS-SCMMAB problem. From §4, we know that myopic intervention (i.e., pulling arms within the current manipulable variables) does not always guarantee optimality for a broader horizon. Rather, the agent must consider the sequence in which variables should be intervened on for subsequent time steps-a non-myopic strategy. To formalize this notion, we define an intervention sequence as follows.

Definition 5.1 (Intervention Sequence) . Given J G , Y K , let S = ( X 1 , . . . , X T ) be an intervention sequence where X t ⊆ V t \ { Y t } for t ∈ [ T ] .

Now, consider intervention sequence ( { X 1 , Z 1 } , ∅ , { Z 3 } ) in Fig. 3. In the first intervention set of the sequence, { X 1 , Z 1 } , Eq. (3) shows that X 1 ∈ V 1 may serve as a predetermined value for the temporal model M 2 , potentially contributing to the maximization of the total reward. For a model M , the maximized sum of expected rewards over two time steps t = 1 , 2 can be divided into two sums of maximized expected rewards, each corresponding to the temporal model M 1 and M 2

<!-- formula-not-decoded -->

In this equation, we define X = { X 1 , Z 1 , ∅} , Z = { Z 1 } and K = ∅ . The value x 1 is a predetermined value of M 2 that is taken from the do-assignment at time t = 1 . From this example, Thm. 5.1 shows that there exists an optimal temporal model under the specific pre-determined values.

Theorem 5.1. (Existence of Optimal Partial Assignment). Given information J G , Y K , let X t ′ be a POMIS with respect to J G [ V t ′ ] , Y t ′ K for the subsequent two time steps t, t ′ ∈ [ T ] with t &lt; t ′ . Then, there exists an assignment w ∗ for a subset of variables W ⊆ Pa ( V t ′ ) = V ⋆ t ′ such that

<!-- formula-not-decoded -->

achieves the maximum expected reward for any v ⋆ t ′ with w ∗ = v ⋆ t ′ [ W ] .

Remark 5.1 . The subset W ⊆ Pa ( V t ′ ) in Thm. 5.1, when combined with the intervention set X t ′ , can be interpreted as a POMIS with respect to J G [ V t ∪ V t ′ ] , Y t ′ K . In other words, although W lies in time step t and X t ′ in time step t ′ , their union forms a possibly-optimal intervention set for Y t ′ under the expanded temporal context, highlighting the intertemporal nature of optimal control.

Theorem 5.1 implies that once the optimal partial assignment w ∗ is fixed, the remaining variables in Pa ( V t ′ ) \ W have no influence on the expected reward of Y t ′ . Therefore, the agent must select these variables with the original POMIS at t .

Definition 5.2 (POMIS with Future Support (POMIS + )) . Given information J G , Y K , for subsequent time step t, t ′ ∈ [ T ] with t &lt; t ′ , let X t be a POMIS with respect to J G [ V t ] , Y t K . If there exists W t ⊆ V t \ { Y t } satisfying Thm. 5.1 such that µ x t = µ x t ∪ w t for every temporal model M t conforming to G [ V t ] , then ( X t ∪ W t ) is called POMIS with future support for Y t ′ , denoted POMIS + t,t ′ .

As the name suggests, POMIS + extends the original POMIS by incorporating one additional W t . This set consists of variables that contribute to the maximization of the expected reward at a subsequent time step. However, the selection of W t is constrained by the temporal structure of causal diagram G .

Proposition 5.1. (Temporal Dependency). Given causal diagram G and a collection of time-specific variables V , POMIS + t,t +1 ⊆ V t \ { Y t } for every t ∈ [ T ] under Assumption 3.1.

Proposition 5.1 implies that in causal diagram G , we can determine POMIS + by only exploring the causal diagram window G [ V t ∪ V t +1 ] .

```
Algorithm 1 Computing all intervention sequences 1: function POMIS + ( G , V , Y , T ) 2: Input : G a causal diagram; V a collection of time indexed variables; Y a collection of time indexed rewards; T the time horizon of an episode 3: return POMIS + SEQ ( G , V , Y , T ) 4: function POMIS + SEQ( G , V , Y , T , [ I + = ∅ ] , [ Q = ∅ ] ) 5: S , Y t , G ′ ←∅ , Y [ T ] , G [ An ( Y t ) G ] 6: P ←{ ( MUCT ( G ′ W , Y t ) , IB ( G ′ W , Y t )) } W ⊆ An ( Y t ) G \{ Y t } 7: for (Xs, Ts) ∈ P do 8: I + , Q← IB + ( V , Xs , I + ) , QIB ( G , V , Ts , Q , I + ) (by Alg. 3) 9: t e ← the earliest time index t such that t ∈ keys ( I + ) 10: if t e > 1 then S ← S ∪ ( POMIS + SEQ ( G , V , Y , t e -1 , I + , Q ) ) 11: else S ← the product combination of all sets { I + [ t ] ∪ q t | q t ∈ Q [ t ] } t ∈ keys ( I + ) ∩ keys ( Q ) (by Prop. 6.1) 12: return S
```

## 6 Graphical Characterization of POMIS +

To graphically characterize POMIS + , we introduce two novel graphical concepts-interventional borders over future rewards (IB + ) and qualified interventional borders (QIB)-which together enable the construction of POMIS + based on causal topological properties.

Definition 6.1 (IB for the Subsequent Time Steps (IB + )) . Given J G , Y K , for any W ⊆ An ( Y t ′ ) G \ { Y t ′ } , define Z ( W ) = IB ( G [ ⋃ t ′ i = t V i ] W , Y t ′ ) . Then, for each j ∈ { t, . . . , t ′ } , Z ( W ) ∩ V j is called an Interventional Border at time slice j for Y t ′ , denoted by IB + j,t ′ .

Intuitively, IB + t,t ′ is computed for the future reward Y t ′ but resides within V t . Therefore, we must identify a qualified interventional border under J G [ V t ] , Y t K -that is, an IB that satisfies certain conditions to remain effective even when IB + t,t ′ is fixed.

Definition 6.2 (Qualified IB (QIB)) . Let IB + t,t ′ ( G [ ⋃ t ′ i = t V i ] , Y t ′ ) = X + where t, t ′ ∈ [ T ] with t &lt; t ′ . If IB ( G [ V t ] X + ∪ W , Y t ) = X for every W ⊆ V t \ { Y t } , then X is a Qualified Interventional Border for a reward variable Y t , denoted QIB t .

According to the definition, the QIB is selected such that the variables in IB + t,t ′ , though fixed to optimize Y t ′ , do not block the causal influence on the current reward Y t . By taking the union of both IB + t,t ′ and QIB t , we can identify POMIS + t,t ′ .

Proposition 6.1. (Composition of POMIS + ). Given J G , Y K , IB + t,t ′ ( G [ ⋃ t ′ i = t V i ] , Y t ′ ) ∪ QIB t ( G [ V t ] , Y t ) is a POMIS + t,t ′ for t, t ′ ∈ [ T ] and t &lt; t ′ .

As shown in Fig. 4, an IB for Y t ′ spans over two time steps (blue shaded), and { X t } is an IB + t,t ′ from Def. 6.1. The candidates for QIB t w.r.t. J G [ V t ] , Y t K are { Z t } and { W t } . If { W t } is chosen for QIB t , then one candidate for POMIS + t,t ′ is { X t , W t } (light gray), whereas if the choice is { Z t } , then POMIS + t,t ′ is { X t , Z t } (dark gray). However, by Def. 6.2, we choose { Z t } for QIB t (purple shaded) not { W t } , which can be explained using do-calculus: we can show that µ x ∗ t ,w ∗ t = µ x ∗ t ≤ µ z ∗ t = µ x ∗ t ,z ∗ t . This inequality suggests that { X t , Z t } is determined as POMIS + t,t ′ rather than { X t , W t } . Note that, under Assumption 3.1, it is sufficient to consider only a

Figure 4: POMIS for the subsequent time step.

<!-- image -->

subsequent time step to obtain POMIS + without examining other time steps. In other words, given J G , Y K , the union of IB + t,t +1 ( G [ V t ∪ V t +1 ] , Y t +1 ) and QIB t ( G [ V t ] , Y t ) constitutes a POMIS + t,t +1 .

## 7 Algorithmic Characterization

We propose a recursive algorithm (Alg. 1) for constructing optimal intervention strategies under nonstationarity, which systematically explores the graph in a backward manner while avoiding exhaustive

Figure 5: Cumulative regret (top) and optimal arm selection probabilities (bottom) across tasks (columns), with solid for Thompson Sampling and dashed for KL-UCB; shaded bands denote ± 1 SD.

<!-- image -->

enumeration of all variable subsets. Given a causal diagram G and a sequence of reward variables Y , the algorithm first constructs a subgraph G ′ consisting of ancestors of the target reward variable Y t . Then, all valid MUCT and IB pairs are identified in the graph G ′ (Line 6). The algorithm iteratively explores each MUCT/IB pair, updating two key maps: I + and Q , corresponding to IB + (Def. 6.1) and QIB (Def. 6.2), respectively (Line 8). Both I + and Q map each time t to the selected intervention variables via the functions IB + and QIB . These maps are propagated through the recursive structure of the algorithm, gradually expanding the intervention variables at each time step. If the earliest time-step of variables currently stored in IB + has not yet reached start time ( t = 1) , the algorithm recursively calls itself to evaluate earlier time steps, taking the current I + and Q as arguments (Line 10). Conversely, suppose IB + reaches the start time, indicating completion of the backward enumeration. In this case, the algorithm generates all possible intervention combinations for each time step based on the final IB + and QIB maps, as stated in Prop. 6.1. It then enumerates all possible intervention sequences by taking the Cartesian product of these timestep-specific combinations, adding each sequence to the intervention sequence container S (Line 11). The construction of POMIS + sequences leverages graphical characteristics (i.e., IB + and QIB) to reduce redundant computations across time. The time complexity is exponential in both the horizon T and the number of variables | V | , specifically O (2 2 | V |· T ) . The details of the time complexity are available in Appendix H. We now show that the recursive enumeration performed by the algorithm is exhaustive and correct with respect to the definition of POMIS + sequences.

Theorem 7.1. (Soundness and Completeness). Given information J G , Y K , Alg. 1 returns all intervention sequences composed solely of POMIS + sets.

## 8 Experiments

We evaluate how effectively the POMIS + strategy captures long-term effects compared to the myopic POMIS baseline. We conduct experiments 3 on three settings, each designed to highlight different aspects of temporal intervention planning. Detailed specifications for each task are provided in App. I. We report two metrics-cumulative regret (CR) and optimal arm selection probability (OAP)-under two MAB solvers: Thompson Sampling (TS) [Thompson, 1933] and KL-UCB [Cappé et al., 2013].

Task 1 (Fig. 3(a)): As shown in Fig. 5(a), CR of the TS for the POMIS + strategy converges around step 20K (solid blue line). In contrast, the sequences constructed from the myopic POMIS strategy show no sign of convergence (solid red line). Similar patterns are observed for the OAP result. Quantitatively, by step 100K, the POMIS + strategy achieves a CR of 1.9K ± 29 (TS) and 4.8K ± 43 (KL-UCB), with optimal-arm probabilities (OAP) of 98.5 ± 1.7% and 97.0 ± 2.4%, respectively. Meanwhile, the POMIS baseline performs substantially worse, with CR values exceeding 34K and 37K, while OAP scores remain near zero. This result is attributed to the fact that myopic strategies fail to include intervention nodes such as X 1 for MUCT ( G [ V 2 ] , Y 2 ) and X 2 for MUCT ( G [ V 3 ] , Y 3 ) . The inability to access such variables leads to critical long-term failures in non-stationary settings.

3 A Python implementation can be found at: https://github.com/yeahoon-k/NS-SCMMAB .

Task 2 (Fig. 4): As shown in the second column of Fig. 5, the POMIS + strategy consistently outperforms the myopic POMIS baseline in both cumulative regret and optimal arm selection probability. By step 100K, POMIS + achieves substantially lower regrets-1.1K ± 32 (TS) and 2.8K ± 44 (KLUCB)-compared to 11.8K ± 77 and 14.1K ± 48 for the POMIS strategy. Furthermore, the OAP of POMIS + reaches 96.5 ± 2.6% (TS) and 90.0 ± 4.2% (KL-UCB), while the POMIS baseline remains near zero. These results demonstrate that POMIS + successfully captures the long-term effect of X t on Y t ′ , which is overlooked by the myopic strategy.

Task 3 (Fig. 6): The final experiment is based on the SCM, which intentionally violates the time-slice Markov assumption (Assumption 3.1) by introducing long-range dependencies and latent interactions across time (Appendix F). As shown in Fig. 5(c), POMIS + outperforms the baseline, though the gap is smaller than in previous tasks. By step 100K, the CR of POMIS + is 3.7K ± 80 (TS) and 8.6K ± 69 (KL-UCB), improving upon the POMIS baseline regret of 5.1K ± 89 and 11.2K ± 72, respectively. In terms of OAP, POMIS + achieves 87.0 ± 4.7% (TS) and 62.0 ± 6.7% (KL-UCB), while POMIS reaches only 80.0 ± 5.6% and 48.5 ± 6.9%, respectively. These results highlight that although the advantage of POMIS + is par-

Figure 6: G for Task 3.

<!-- image -->

tially reduced due to the absence of time-local independence, it still provides superior performance by leveraging earlier interventions that propagate across time via DUCs.

## 9 Discussion

We assume access to the full causal structure during evaluation. In many real-world settings, however, the agent may have only limited or partial access to future causal diagrams rather than complete knowledge of them. Extending our framework to handle partially observed or uncertain temporal structures-while maintaining optimality-remains an important direction for future work.

While our proposed algorithm is theoretically sound and complete, its computational complexity is constrained by the exponential growth of the search space. A promising direction is to exploit repeated or modular structures that frequently appear in non-stationary environments-for instance, by developing segment-wise variants of our algorithm that reuse computed intervention sequences across recurring causal patterns. Such extensions could preserve theoretical soundness while substantially improving scalability and applicability to larger, real-world domains.

For the experimental setting, each experimental trial corresponds to a complete causal rollout, where outcomes from all time steps are aggregated into a single episode-level reward. This episodic formulation enables fair comparison across intervention sequences based on their overall causal influence over time. Although interventions are not executed sequentially across evolving environments, the causal dependencies among time-indexed variables are fully encoded in the temporal models, ensuring valid evaluation of the structural optimality of POMIS + sequences. Further discussions are provided in Appendix K.

## 10 Conclusion

In this work, we present a novel framework; the NS-SCM-MAB, that integrates non-stationary multiarmed bandit problems with structural causal models. Our formulation captures both evolving reward distributions and shifting causal structures, enabling temporally-aware decision-making that accounts for how interventions propagate over time. To formalize this, we introduce POMIS + , a temporallyextended notion of optimal intervention sets, and provide theoretical justification showing that conditioning on preceding variables can achieve improved expected rewards in dynamic environments. Building on this, we develop an algorithm to identify non-myopic intervention strategies, and demonstrate its empirical advantages over myopic baselines across a variety of non-stationary settings. We believe this work provides a foundation for causal reasoning in sequential decision-making under temporal dynamics and opens up new avenues for research at the intersection of causal inference and non-stationary environments.

## Acknowledgments and Disclosure of Funding

We thank anonymous reviewers for constructive comments to improve the manuscript. This work was partly supported by the IITP (RS-2022-II220953/25%, RS-2025-02263754/25%) and NRF (RS-202300211904/25%, RS-2023-00222663/25%) grants funded by the Korean government. Yesong Choe was supported in part by Basic Science Research Program through the NRF funded by the Ministry of Education (RS-2025-25418030).

## References

- Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning , 47(2):235-256, 2002a.
- Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. The nonstochastic multiarmed bandit problem. SIAM journal on computing , 32(1):48-77, 2002b.
- Elias Bareinboim, Andrew Forney, and Judea Pearl. Bandits with unobserved confounders: A causal approach. Advances in Neural Information Processing Systems , 28:1342-1350, 2015.
- Olivier Cappé, Aurélien Garivier, Odalric-Ambrym Maillard, Rémi Munos, and Gilles Stoltz. Kullback-leibler upper confidence bounds for optimal sequential allocation. The Annals of Statistics , pages 1516-1541, 2013.
- Andrew Forney, Judea Pearl, and Elias Bareinboim. Counterfactual data-fusion for online reinforcement learners. In International Conference on Machine Learning , pages 1156-1164. PMLR, 2017.
- John C Gittins. Bandit processes and dynamic allocation indices. Journal of the Royal Statistical Society: Series B (Methodological) , 41(2):148-164, 1979.
- Wen Huang and Xintao Wu. Robustly improving bandit algorithms with confounded and selection biased offline data: A causal approach. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 20438-20446, 2024.
- Daphne Koller and Nir Friedman. Probabilistic Graphical Models: Principles and Techniques Adaptive Computation and Machine Learning . The MIT Press, 2009. ISBN 0262013193.
- Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in applied mathematics , 6(1):4-22, 1985.
- Finnian Lattimore, Tor Lattimore, and Mark D Reid. Causal bandits: Learning good interventions via causal inference. In Advances in Neural Information Processing Systems , pages 1181-1189, 2016.
- Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- Sanghack Lee and Elias Bareinboim. Structural causal bandits: where to intervene? Advances in Neural Information Processing Systems , 31, 2018.
- Sanghack Lee and Elias Bareinboim. Structural causal bandits with non-manipulable variables. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 4164-4172, 2019.
- Behzad Nourani-Koliji, Steven Bilaj, Amir Rezaei Balef, and Setareh Maghsudi. Piecewise-stationary combinatorial semi-bandit with causally related rewards. In ECAI 2023 , pages 1787-1794. IOS Press, 2023.
- Orestis Papadigenopoulos, Constantine Caramanis, and Sanjay Shakkottai. Non-stationary bandits under recharging payoffs: Improved planning with sublinear regret. Advances in Neural Information Processing Systems , 35:20325-20337, 2022.
- Judea Pearl. Causality: models, reasoning and inference . Cambridge university press, 2009.
- Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018.

- William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika , 25(3-4):285-294, 1933.
- Jin Tian and Judea Pearl. On the identification of causal effects. Technical Report R-290-L, Department of Computer Science, University of California, Los Angeles, CA, 2003.
- Thomas Sadanand Verma. Invariant Properties of Causal Models. Manuscript, 1992. University of California, Los Angeles.
- Thomas Sadanand Verma and Judea Pearl. Equivalence and Synthesis of Causal Models. In Proceedings of the Sixth Conference on Uncertainty in Artificial Intelligence , pages 220-227, Cambridge, MA, 7 1990.
- Peter Whittle. Restless bandits: Activity allocation in a changing world. Journal of applied probability , 25(A):287-298, 1988.
- Junzhe Zhang and Elias Bareinboim. Transfer learning in multi-armed bandit: a causal approach. In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems , pages 1778-1780, 2017.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We accurately state our abstract and introduction based on our paper's main claims named POMIS + , which is a temporally-aware extension of optimal intervention sets that accounts for both immediate and future rewards in non-stationary causal bandit settings.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We list our assumptions early in the paper and refer to them throughout the paper as these assumptions, naturally, have an impact on the work presented. We further discuss the assumptions at length on page four after they have been stated. Further we have derived the time-complexity of our main algorithm (see Alg. 1) to ensure that the reader is fully informed of the computational overhead required to use our work. While we have sought to demonstrate a representative set of graphs to experiment with, we naturally cannot test them all but have provided adequate proofs which show that our method is sound and complete for any SCM that conforms to the assumptions we make in this work.

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

Justification: We introduce a number of theorems with accompanying proofs. We make an assumption in this paper, all of which are listed in the body: Assumption 3.1. All our theorems, propositions and corollaries are found in the body of the paper, we also reproduce each statement in Appendix G followed by its corresponding proof. Supporting assumptions, lemmas and theorems are adequately referenced in all our contributions.

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

Justification: We provide the full set of parameters used to produce our experimental results, all the details can be found in Appendix I. In addition, we will make open-source the source-code post-review.

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

Justification: We make the code openly available on GitHub, with clear instructions to reproduce the main experimental results.

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

Justification: We provide all necessary details to reproduce our results in full. §8 and Appendix I contain the necessary parameter settings used to reproduce the plots in §8. As noted above, we will also release the code post-review. With that the reader is fully equipped to test out our method and reproduce our results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We display uncertainty quantification ('error bars') on the plots which show the cumulative regret as well as the optimal arm-selection probabilities - see Fig. 5. In the body of the text we only report the mean value to reduce clutter but those results are also available.

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

Justification: This paper provides sufficient information about the computational resources required to reproduce the experiments, which is available in Appendix I.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms fully with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [No]

Justification: This is fundamental research, and the theorems and methods we present are only evaluated for their technical accuracy and compared to reference methods, to demonstrate the increased utility of our work. The work we present is fundamental within the realm of causal inference and optimal decision-making (bandits). We cannot at this point in time construe any negative societal impacts that may stem from our work.

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

This is fundamental research and the work presented herein does not pose any such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The only relevant comparison to the work presented in this paper is that by Lee and Bareinboim [2018]. All theorems and methods from that paper have been cited fully throughout the paper and appendix. This is only true for the theory, no additional assets (e.g., code, data, models) were used in this work.

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

Answer: [Yes]

Justification: Yes. We will provide our code via GitHub. The repository will include a README file with detailed instructions on how to use the code and reproduce the main results. All new assets introduced in the paper are documented and provided alongside the code.

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

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Organization

A schematic of our framework and its logical flow is illustrated in Fig. 7. The diagram visually maps out how the paper develops, starting from the formulation of the non-stationary SCM-MAB problem in §3, and introducing the temporal model along with key graphical assumptions about the nature of nonstationarity. §5 establishes that in a temporal model, partial assignments of predetermined variables can affect optimal interventions at future time steps, motivating the definition of POMIS + as a nonmyopic intervention. This leads into the graphical characterization in §6, which defines key structures (IB + , QIB) that construct POMIS + . Finally, §7 presents an algorithm that enumerates optimal intervention sequences based on these graphical elements. The structure clarifies the dependencies between the paper's components and highlights how our contributions build upon one another.

Figure 7: An overall schema of this paper.

<!-- image -->

## B Preliminaries

Here we present definitions of MIS, POMIS, MUCT, and IB.

Definition B.1. (Minimal Intervention Set (MIS) [Lee and Bareinboim, 2018]). A set of variables X ⊆ V \ { Y } is said to be a minimal intervention set relative to [ G , Y ] if there is no X ′ ⊂ X such that µ x [ X ′ ] = µ x for every SCM conforming to the G .

Definition B.2. (Possibly-Optimal Minimal Intervention Set (POMIS) [Lee and Bareinboim, 2018]). Given information [ G , Y ] , let X be a MIS. If there exists an SCM conforming to G such that µ x ∗ &gt; ∀ Z ∈ Z \{ X } µ z ∗ , where Z is the set of MISs with respect to G and Y , then X is a possiblyoptimal minimal intervention set with respect to the information [ G , Y ] .

Definition B.3. ( C -component [Tian and Pearl, 2003]). In a causal diagram G , two variables are said to be in the same confounded component (for short, C -component or CC ( · ) G ) if and only if they are connected by a bi-directed edge (i.e., a path composed solely of V i ↔ V j ).

In this paper, we denote CC ( X ) G = ⋃ X ∈ X CC ( X ) G .

Definition B.4. (Unobserved-Confounders' Territory [Lee and Bareinboim, 2018]). Given information [ G , Y ] , let H be G [ An ( Y ) G ] . A set of variables T ⊆ V ( H ) containing Y , where V ( H ) is the set of variables in H , is called a UC-territory on G with respect to Y if De ( T ) H = T and CC ( T ) H = T .

A UC-territory T is said to be minimal if no T ′ ⊂ T is a UC-territory. A minimal UC-Territory (MUCT) for G and Y can be constructed by extending a set of variables, starting from { Y } , alternatively updating the set with the c-component and descendants of the set.

Definition B.5. (Interventional Border [Lee and Bareinboim, 2018]). Let T be a minimal UC-territory on G with respect to Y . Then, X = pa ( T ) G \ T is called an interventional border for G with respect to Y .

## Algorithm 2 Minimal unobserved confounders' territory

̸

- 1: function MUCT ( G , Y ) 2: H = G [ An ( Y ) G ] 3: Q = { Y } ; T = { Y } 4: while Q = ∅ do 5: remove an element Q 1 from Q 6: W = CC ( Q 1 ) H ; T = T ∪ W ; Q = ( Q ∪ de ( W ) H ) \ T 7: return T

## C Nomenclature

NS

Non-stationary

SCM

Structural causal model

RL

Reinforcement learning

MAB

Multi-armed bandit

MIS

Minimal intervention set

UC

Unobserved confounder

POMIS

Possibly-optimal minimal intervention set

MUCT

Minimal unobserved confounders' territory

IB

Interventional border

QIB

Qualified interventional border

DUC

Dynamically unobserved confounder

CR

Cumulative regret

OAP

Optimal arm-selection probability

KL-UCB

Kullback-Leibler upper confidence bound

## D Stationary SCM-MAB

The cumulative regret of the stationary setting is given by:

<!-- formula-not-decoded -->

where x † denotes the globally optimal arm, defined as

<!-- formula-not-decoded -->

where µ x t is the arm played at round t . The SCM-MAB setting assumes that the agent has full access to the causal graph G of M , although its parametrization remains unknown-i.e., the agent knows the structure G , but not the structural functions F or the distribution over exogenous variables P ( U ) . Furthermore, the causal graph G is assumed to be static, meaning that the underlying causal structure of the domain does not change over time. As a result, the agent interacts with the same causal model in each round.

In this setting, there is no confounding across time slices, and thus, no information propagates between rounds. Consequently, the reward distribution remains fixed across rounds.

In practice, under the stationary setting, the agent effectively observes, in each round, a causal diagram G composed of temporally disconnected slices, as illustrated in Fig. 1(b). This stationary (and also non-stationary) formulation typically assumes that the number of arms is constant throughout the interaction. The graphical structure and topology of each time slice are identical across all rounds.

The SCM-MAB framework inherently introduces dependencies between arms, stemming from the underlying causal relationships among endogenous and exogenous variables. Lee and Bareinboim [2018, 2019] identified two structural properties that can be derived from any SCM-MAB framework:

1. Arm equivalence: a characterization of arms that share identical reward distributions, determined using constraints from do-calculus, and
2. Partial-orders among arms: under what topological conditions one arm can be optimal.

Leveraging these properties, one can identify minimal intervention sets (MIS) that constitute a nonredundant collection of informative interventions. In addition, Lee and Bareinboim [2019] identified both MIS and POMIS for the stationary setting with non-manipulative variables. However, these characterizations rest on the assumption that the causal graph G remains static and stationarymeaning that no information is carried over from previous decisions. §3 extends beyond the stationary assumption to present the SCM-MAB framework in the non-stationary setting.

## E Comparison with Conventional Non-Stationary Bandit Algorithms

We contrast our approach to non-stationarity in the SCM-MAB with traditional non-stationary bandit settings. Our focus is on how causal modeling provides a structured explanation for reward distribution shifts over time, as opposed to treating them as statistical artifacts. These build upon early foundational work on dynamic allocation and index policies [Whittle, 1988, Gittins, 1979].

Conventional NS-bandit formulations Conventional non-stationary bandit algorithms aim to maintain low regret with respect to a comparator (or competitor) class-a predefined set of benchmark policies that may adapt over time to account for changing environments (e.g., policies that allow a limited number of switches between arms, comparators that track the best-performing arm over recent time windows, or strategies that assume bounded changes in the underlying reward distribution). These algorithms are typically categorized into two regimes: adversarial and stochastic bandits, illustrated in Table 1.

Table 1: Representative non-stationary bandit settings and algorithms (adapted from Lattimore and Szepesvári [2020]).

| Regime      | Setting              | Description                                                              | Representative Algorithms                           |
|-------------|----------------------|--------------------------------------------------------------------------|-----------------------------------------------------|
| Adversarial | L -switching         | The identity of the optimal arm may change abruptly up to L times        | Exp3.S, AdaHedge                                    |
| Adversarial | Variation budget     | Total variation in reward sequences is bounded by V                      | Rexp3, Adapt-EvE                                    |
| Stochastic  | Piecewise-stationary | Rewards are stationary within inter- vals, with occasional change-points | Sliding-window UCB, Change- point Thompson Sampling |
| Stochastic  | Drifting             | Expected rewards evolve smoothly over time                               | Discounted UCB, Sliding-window UCB, SW-TS           |

Each setting assumes non-stationarity as a statistical property: either abrupt shifts, or slowly drifting rewards. Some algorithms require knowledge of the number of switches L , while others are designed to be adaptive. In the variation budget setting, the cumulative amount of change in reward distributions is bounded by a budget V , offering a finer-grained control of non-stationarity than simple change-point models.

Causal non-stationarity (our perspective) Our framework models non-stationarity as a consequence of causal information propagation across time. Specifically, transition edges in the causal graph G (e.g., X t Z t ′ ) induce changes in the downstream reward variables (e.g. Y t ′ ), illustrated in Fig. 2. This structure directly captures why the reward distribution changes.

For instance, whereas traditional settings might treat E [ Y t | do( x )] as shifting arbitrarily with t , our approach identifies structural causes: P ( Y t | do( X t )) is influenced by information propagation

from previous variables (e.g. X t -1 , Z t -1 ). This allows us to model the mechanism behind reward distribution shifts.

Moreover, the presence of arcs (or edges) between time slices in G determines where and how changes occur. This is closely aligned with the 'mean payoff drift' interpretation in traditional models [Lattimore and Szepesvári, 2020, Chapter 31], but we interpret it in terms of explicit graphical information in G .

Summary Most traditional algorithms detect or adapt to change, but do not explain it. Our approach, by contrast, offers a mechanism-based explanation of non-stationarity-one grounded in a causal understanding of the system under investigation, which is itself a strong assumption-via the SCM structure. This allows for:

- Identification of reward-relevant intervention targets (POMIS + )
- Intervention sequence planning backed by theoretical guarantees derived from the underlying causal structure

Ultimately, our method treats non-stationarity as a structured phenomenon emergent from a dynamic causal model, rather than as an arbitrary change in observed statistics.

## F Semi Time-Slice Markovian Non-Stationary SCM-MAB

Figure 8: IB + ranges over four time steps.

<!-- image -->

We begin by formalizing dynamic unobserved confounders (DUCs), which represent exogenous variables inducing dependencies across time.

Definition F.1. (Dynamic unobserved confounders (DUCs)). Given J G , Y K , let U ⋆ t denote the set of exogenous variables that induce dependencies between variables in V t and V t ′ for some t &lt; t ′ . If U j ∈ U ⋆ t , then we refer to U j as a dynamic unobserved confounder (DUC), indicating that U j introduces confounding effects that persist across time.

When such DUCs are permitted in the graphical structure, we say the temporal graph satisfies the semi-time-slice Markov property.

̸

Definition F.2 (Semi-time-slice Markov) . A temporal graph G satisfies the semi-time-slice Markov property if it allows the presence of dynamic unobserved confounders (DUCs) across time. That is, the graph permits bidirected edges between variables V t ∈ V t and V t ′ ∈ V t ′ for t = t ′ , representing confounding induced by exogenous variables U j ∈ U ⋆ t that simultaneously affect both time slices.

Under the semi-time-slice Markov assumption induced by dynamic unobserved confounders (Def. F.1), we need to perform a time-expanded search for valid intervention targets across multiple time steps. Consider the example in Fig. 8. When we calculate IB + for MUCT ( G [ ⋃ 4 i =1 V i ] , Y 4 ) , the result consists of four sets: IB + 1 , 4 = { X 1 } , IB + 2 , 4 = { W 2 } , IB + 3 , 4 = { W 3 } and IB + 4 , 4 = { W 4 } (blue shaded). Given those IB + at each time step, we can determine each QIB t according to Def. 6.2. For example, when IB + 1 , 4 = { X 1 } is given, the IB corresponding to the MUCT ( G [ V 1 ] X 1 ∪ Z 1 , Y 1 ) = { Y 1 } is Z 1 , hence QIB 1 = { Z 1 } . Similarly, for MUCT ( G [ V 2 ] X 2 ∪ W 2 , Y 2 ) , we can select QIB 2 = { Z 2 } , and for MUCT ( G [ V 3 ] X 3 ∪ W 3 , Y 3 ) , we can choose QIB 3 = { Z 3 } (purple shaded). For now, by Prop. 6.1, we can obtain POMIS + s by taking the union of each IB + t,t ′ and QIB t -POMIS + 1 , 4 = { X 1 , Z 1 } , POMIS + 2 , 4 = { W 2 , Z 2 } , POMIS + 3 , 4 = { W 3 , Z 3 } and POMIS + 4 , 4 = { W 4 } . These POMIS + sets can then be used as components of an intervention sequence in the NS-SCM-MAB setting.

## G Proofs

Theorem 5.1. (Existence of Optimal Partial Assignment). Given information J G , Y K , let X t ′ be a POMIS with respect to J G [ V t ′ ] , Y t ′ K for the subsequent two time steps t, t ′ ∈ [ T ] with t &lt; t ′ . Then, there exists an assignment w ∗ for a subset of variables W ⊆ Pa ( V t ′ ) = V ⋆ t ′ such that

<!-- formula-not-decoded -->

achieves the maximum expected reward for any v ⋆ t ′ with w ∗ = v ⋆ t ′ [ W ] .

Proof. Let t, t ′ ∈ [ T ] with t &lt; t ′ be two time steps. Given the information J G , Y K , we fix an arbitrary target variable Y t ′ ∈ Y located in the time slice G [ V t ′ ] .

We now consider any temporal model M t ′ that conforms to the time slice G [ V t ′ ] . Let X t ′ be a POMIS for Y t ′ with respect to the time slice G [ V t ′ ] . The goal is to show that under any temporal model conforming to G [ V t ′ ] , there exists a partial assignment w ∗ that preserves the optimality of X t ′ .

By the definition of POMIS, there exists an intervention assignment x ∗ t ′ ∈ D ( X t ′ ) such that:

<!-- formula-not-decoded -->

where Z is the set of all MISs for Y t ′ in G [ V t ′ ] .

Now consider how expectations are evaluated in the temporal model M t ′ : they are conditioned on predetermined variables v ⋆ t ′ ∈ D ( Pa ( V t ′ )) . Hence, there must exist at least one such v ⋆ t ′ under which the above inequality holds.

Fix any such v ⋆ t ′ . Since the domain D ( Pa ( V t ′ )) is finite, and the conditional expectation E M t ′ | v ⋆ t ′ [ Y t ′ | do( · )] is a deterministic function of v ⋆ t ′ and the do-intervention, there exists a minimal subset W ⊆ Pa ( V t ′ ) such that w ∗ = v ⋆ t ′ [ W ] satisfies:

<!-- formula-not-decoded -->

That is, fixing the partial assignment w ∗ suffices to preserve the optimality of the POMIS X t ′ regardless of how the remaining variables in Pa ( V t ′ ) \ W are instantiated.

Since the choice of M t ′ was arbitrary (subject to conforming to G [ V t ′ ] ), this concludes that such a subset W and partial assignment w ∗ must exist under any temporal model consistent with the time-slice causal structure.

Proposition 5.1. (Temporal Dependency). Given causal diagram G and a collection of time-specific variables V , POMIS + t,t +1 ⊆ V t \ { Y t } for every t ∈ [ T ] under Assumption 3.1.

Proof. For the sake of contradiction, suppose that the proposition does not hold. Then, there exists some t ∈ [ T ] such that:

<!-- formula-not-decoded -->

That is, there exists a POMIS + set X + t,t +1 such that:

<!-- formula-not-decoded -->

By this existence, X must either lie in a future time slice ( X ∈ V t ′ with t &lt; t ′ ), or X = Y t .

We now argue that such an X cannot be part of any valid POMIS + set under Assumption 3.1. Recall that under the assumption, the only causal influences on Y t +1 must flow through variables in V t (i.e., no time cycles and no backward edges from V t ′ with t &lt; t ′ ).

Furthermore, POMIS + t,t +1 is defined as the minimal set in V t such that the intervention at time t maximizes the expected reward at time t + 1 . Including any X / ∈ V t \ { Y t } violates both this minimality and the validity of the do-intervention within time t .

This leads to a contradiction. Hence, for all t ∈ [ T ] , we must have:

<!-- formula-not-decoded -->

We now prove the composition property of POMIS + t,t ′ , which guarantees that interventions on IB + t,t ′ do not block the causal effect of QIB t on Y t . Before proceeding, we formalize the fact that the set constructed by IB + t,t ′ satisfies the partial assignment condition required by Thm. 5.1.

Proposition G.1 (IB + satisfies the condition of Thm. 5.1) . Let t &lt; t ′ and let X t ′ be a POMIS for Y t ′ in G [ V t ′ ] . Then, the set IB + t,t ′ ( G [ ⋃ t ′ i = t V i ] , Y t ′ ) identifies a subset of Pa ( V t ′ ) such that there exists a partial assignment w ∗ over this set which satisfies the condition of Thm. 5.1.

Proof. Let t &lt; t ′ and let X t ′ be a POMIS for Y t ′ in G [ V t ′ ] .

From Thm. 5.1, this implies the existence of a subset W ⊆ Pa ( V t ′ ) and a partial assignment w ∗ such that for all v ⋆ t ′ with v ⋆ t ′ [ W ] = w ∗ , the reward under do( x ∗ t ′ ) can be maximized.

′

Now consider the construction of IB + t,t ′ ( G [ ⋃ t i = t V i ] , Y t ′ ) . By definition, this set is formed by computing the interventional border IB in the full unrolled graph G , and selecting from it only those variables that reside in V t .

Since the IB of Y t ′ is known to be a POMIS, and since IB + is a subset of this IB restricted to variables available at time t , the values assigned to IB + in any temporal model appear as part of some v ⋆ t ′ ∈ D ( Pa ( V t ′ )) .

Note that although Thm. 5.1 is stated in terms of a temporal model M t ′ , the structure of M t ′ -including its structural functions and the set of predetermined variables Pa ( V t ′ ) -is dependent on the M . Since IB + t,t ′ is computed from the global graph G [ ⋃ t ′ i = t V i ] , it selects variables that reflect this SCM-induced structure, and thus corresponds to the W required in Thm. 5.1.

Therefore, IB + t,t ′ fulfills the role of W in Thm. 5.1, ensuring that fixing its values preserves the optimality of X t ′ consistent with w ∗ .

Proposition 6.1. (Composition of POMIS + ). Given J G , Y K , IB + t,t ′ ( G [ ⋃ t ′ i = t V i ] , Y t ′ ) ∪ QIB t ( G [ V t ] , Y t ) is a POMIS + t,t ′ for t, t ′ ∈ [ T ] and t &lt; t ′ .

Proof. Let X t := QIB t ( G [ V t ] , Y t ) and W t := IB + t,t ′ ( G [ ⋃ t ′ i = t V i ] , Y t ′ ) . By Def. 6.2, X t is a Qualified Interventional Border (QIB) for Y t with respect to G [ V t ] , which implies that X t is a POMIS for Y t . Thus, the first condition of Def. 5.2 is satisfied.

Next, as established in Prop. G.1, the set W t constructed via IB + t,t ′ satisfies the condition of Thm. 5.1: there exists a partial assignment w t over W t such that the expected reward for Y t ′ under do( x t ′ ) is maximized.

Moreover, due to the construction of QIB t , which requires that IB ( G [ V t ] Z ∪ W t , Y t ) = X t for some Z ⊆ V t \ { Y t } , it follows that interventions on W t do not interfere with the causal effect of X t on Y t within G [ V t ] .

Therefore, the expected reward at time t remains unchanged µ x t = µ x t ∪ w t . Hence, by Def. 5.2, the combined set X t ∪ W t is a POMIS + t,t ′ .

Theorem 7.1. (Soundness and Completeness). Given information J G , Y K , Alg. 1 returns all intervention sequences composed solely of POMIS + sets.

Proof. We prove the theorem by showing (i) soundness: every sequence returned by Alg. 1 is composed solely of POMIS + , and (ii) completeness: every valid sequence of POMIS + is returned.

(Soundness) Fix the final time step T . Let G ′ = G [ An ( Y T ) G ] and suppose the algorithm selects a valid MUCT/IB pair ( X , T ) under the G ′ . IB + ( V , X , I + ) updates the IB + map by assigning each X ∈ X to its respective time step t X . For each t ∈ keys ( I + ) , the function QIB ( G , V , T , Q , I + ) constructs a mutilated graph G [ V t ] I + [ t ] , and from it computes all valid POMISs (i.e., q t ) that do not intersect with T . In the base case (i.e., when the earliest t e = 0 ), the Cartesian product over all t gives sequences {I + [ t ] ∪ q t | q t ∈ Q [ t ] } . By Prop. 6.1, each I + [ t ] ∪ q t is a valid POMIS + set at time t . Thus, every sequence in S consists of only POMIS + sets.

(Completeness) The algorithm recursively traces backward from t = T to earlier time steps, guided by the smallest time index in the current IB + map, denoted t e , and recurses with horizon t e -1 . At each step, the algorithm considers all valid MUCT/IB pairs for Y t . The correctness of this step is guaranteed by the soundness and completeness of the algorithm POMISs from Lee and Bareinboim [2018, Theorem 9]. Each recursive call updates the maps I + and Q by accumulating IB + and QIB sets across all considered time steps. Once the earliest time index in IB + reaches t = 0 , the recursion terminates. At this point, the algorithm forms the Cartesian product ∏ t ∈ keys ( I + ) ∩ keys ( Q ) {I + [ t ] ∪ q t | q t ∈ Q [ t ] } , enumerating every possible combination across time. Since every possible IB + and QIB configuration is explored and retained, the final set S contains all valid sequences composed solely of POMIS + sets.

## H Algorithmic Characterization of POMIS +

```
Algorithm 3 Update I + and Q from I + 1: function IB + ( V , X , I + ) 2: for each X ∈ X do 3: Identify the time step t such that X ∈ V t 4: if t / ∈ I + then 5: I + [ t ] ← [ ] 6: append X to I + [ t ] 7: return I + 8: function QIB ( G , V , T , Q , I + ) 9: for each t ∈ keys ( I + ) do 10: if t / ∈ Q then 11: G t , Q [ t ] ←G [ V t ] I + [ t ] , [ ] 12: for each X ∈ POMISs ( G t , Y t ) do 13: if X ∩ T = ∅ then 14: append X to Q [ t ] 15: return Q
```

In this appendix, we provide a detailed explanation of the algorithmic components used in the enumeration of POMIS + intervention sequences, as introduced in §7. At first, we define and clarify the role of the two internal maps, I + and Q , which correspond to the Interventional border for the subsequent time steps (IB + ) and qualified interventional borders (QIB), respectively.

Map of interventional border for the subsequent time steps ( I + ) Given a POMIS candidate X obtained from the MUCT-IB procedure, we identify the time step t associated with each variable X ∈ X , and store it in the map I + [ t ] . The map I + thus organizes variables in X by their corresponding time step, preserving the temporal alignment of possible interventions. This time-indexed representation allows the algorithm to recursively evaluate which time steps still require backward expansion, and is central to the structure of Alg. 1. This structure ensures that only the earliest relevant time step is explored recursively, avoiding redundant expansion into irrelevant subgraphs.

Mapof qualified interventional Border ( Q ) Once I + is populated, the QIB map Q is computed by evaluating each subgraph G [ V t ] after mutilating it with the intervention set I + [ t ] . From this mutilated subgraph, we re-run the POMISs algorithm on the local reward variable Y t to identify any remaining minimal intervention sets. These are stored in Q [ t ] only if they do not overlap with the current MUCT set T , ensuring independence across temporal intervention levels. The QIB thus captures any residual variables at time t that are necessary to optimize the local reward, complementing the already selected IB + .

The recursive design of Alg. 1 ensures that the enumeration of intervention sequences terminates when all relevant intervention variables are assigned by time t = 0 . At that point, the algorithm takes the Cartesian product of all IB + and QIB sets across time steps to generate complete intervention

sequences. This backward recursive approach avoids exhaustive enumeration by pruning irrelevant branches early and leveraging graph locality in the causal diagram.

Figure 9: Illustration of the intermediate stage of the algorithm after two recursive calls.

<!-- image -->

Example traces of recursive enumeration. Fig. 9 illustrates an intermediate stage in the execution of our recursive POMIS + enumeration algorithm. The color-coded graph shows how MUCT, IB + , and QIB evolve across time steps as the algorithm proceeds backward.

At the final time step t = 3 (corresponding to Y 4 in the figure), the algorithm computes the MUCT and IB from the subgraph G [ V 3 ∪ V 4 ] , resulting in a MUCT set { X 4 , Z 4 , Y 4 } (highlighted in red). The corresponding IB + consists of the set { X 3 , W 4 } , indicated in blue. Next, the QIB is computed by evaluating the mutilated subgraph G [ V 3 ] X 3 ,W 4 , yielding { Z 3 } as an intervention set (highlighted in purple).

Since the current earliest time step in IB + is t = 3 &gt; 1 , the algorithm proceeds recursively with T ← 2 . In this second round, the algorithm again computes the MUCT/IB pair for Y 2 , resulting in MUCT { X 2 , Z 2 , Y 2 } (red), IB + = { X 1 , W 2 } (blue), and QIB = { Z 1 } (purple), after mutilating G [ V 1 ] with { X 1 , W 2 } .

Figure 10: Illustration of the intermediate stage of the algorithm after three recursive calls.

<!-- image -->

Fig. 10 also provides one of the recursive traces of our POMIS + enumeration algorithm over steps, starting from the final reward Y 4 at time t = 4 and proceeding backward to Y 1 at t = 1 . As before, red nodes indicate variables in the MUCT set, blue nodes indicate IB + variables and purple nodes denote QIB elements at each stage.

At time t = 4 , the algorithm identifies the MUCT { Y 4 , Z 4 , X 4 } (highlighted in red), computed from the subgraph G [ V 3 ∪ V 4 ] . The IB + map is then updated to include { X 3 , W 4 } (blue), and since { Z 3 } influences to Y 3 in G [ V 3 ] X 3 ,W 4 , the QIB is { Z 3 } at this step.

Proceeding to time t = 2 , the algorithm observes that a single node Y 3 can be a MUCT. The corresponding IB + is updated to { Z 2 } (purple), with no further QIB identified. At time t = 1 , the MUCT set becomes { Y 1 , Z 1 , X 1 } (red). The IB + is now { Z 1 } (blue), as Z 1 is needed to activate paths to Y 2 under the current mutilated graph. Again, no QIB is generated at this time. Finally, at t = 0 , the algorithm computes MUCT as { X 1 , Z 1 , Y 1 } (red), and identifies IB + = { W 1 } (blue), completing the backward exploration.

This example highlights how our algorithm incrementally expands the maps I + and Q over time by tracing from Y T back to earlier rewards. Each recursive call explores a new MUCT/IB combination while updating time-specific intervention targets, eventually enabling full enumeration when the earliest IB + reaches time t = 1 .

Time Complexity of Alg. 1. The recursive construction of POMIS + sequences explores possible intervention combinations over a time horizon T . In the worst case, each time step requires enumer-

ating subsets of ancestors of Y t to compute MUCT/IB pairs, followed by QIB exploration-both involving searches over subsets of the variable set V (i.e., O (2 2 | V | )) . Each of these steps incurs an exponential cost of up to 2 | V | , resulting in 2 2 | V | complexity per time step. After T recursion, the total worst-case time complexity becomes O (2 2 | V |· T ) .

## I Experiment Details

In this section, we provide detailed specifications of the SCMs used in our experiments to ensure reproducibility. For each experiment, the simulation was repeated 200 times using the corresponding SCM. All experiments were run on a dual-socket Intel Xeon Gold 5317 system with 24 physical cores (48 logical threads) at 3.0GHz.

Task 1. We implement an SCM composed of three time steps t = 1 , 2 , 3 with variable sets { X t , Z t , Y t } and structural equations defined to model the propagation of intervention effects over time. The probability distributions over exogenous variables U are defined as: We use the following probabilities over exogenous variables:

<!-- formula-not-decoded -->

The structural functions f V for each endogenous variable V are as follows (where ⊕ denotes binary XOR and v is the valuation of its parents):

<!-- formula-not-decoded -->

Figure 11: SCM definition for Task 1.

Task 2. We conducted the experiment with a two-step SCM defined over variables { W t , X t , Z t , Y t } for t = 1 , 2 , as illustrated in Fig. 4. The structure explicitly models how intervention effects on X t propagate across time via X t +1 and influence downstream outcomes Y t and Y t +1 through intermediary variables Z t .

The exogenous distribution P ( U ) is parameterized to highlight the long-term influence of early interventions. The assigned probabilities are as follows:

<!-- formula-not-decoded -->

We define the structural equations in Fig. 12 (where ⊕ denotes binary XOR). Notably, the structural

<!-- formula-not-decoded -->

Figure 12: SCM definition for Task 2.

functions for X 2 and Y 2 include X 1 as a parent, creating a direct dependency between early decisions and future rewards. This design allows us to test the effectiveness of DUC-based POMIS + strategies in capturing delayed causal influences that are missed by myopic approaches.

Task 3. To clearly illustrate how information from early IB + and QIB components can propagate across multiple time steps in the absence of the time-slice Markov assumption (Assumption 3.1), we design an SCM inducing Fig. 6. This structure focuses on exposing the long-range effect of early interventions. The SCM contains three time steps t = 1 , 2 , 3 over variables { W t , X t , Y t } , with unobserved confounding between X t and Y t at each step. A bidirected edge between X 2 and X 3 further emphasizes the departure from the time-slice Markov assumption. This configuration enables us to test whether POMIS + can successfully capture reward-relevant information originating from earlier steps.

The structural functions f V for each endogenous variable V are defined as follows:

<!-- formula-not-decoded -->

Figure 13: SCM definition for Task 3.

The probability distributions over exogenous variables U are defined as:

<!-- formula-not-decoded -->

## J Limited Alternative Methods to Approach on NS-SCM-MAB

## J.1 Projection-based approach on NS-SCM-MAB

Figure 14: MUCT and IB in the G [ V \ Y ]

<!-- image -->

Lee and Bareinboim [2019] builds on the framework of Lee and Bareinboim [2018] by relaxing the assumption that every variable in the system must be manipulable, while still operating under a stationary setting. Their method uses latent projections (Verma and Pearl [1990], Verma [1992]) to identify possibly-optimal intervention targets and infer dependency relations among them, even when only partial causal information is available.

In our setting, although the underlying process is non-stationary, the complete causal graph is revealed after all rounds. This allows the latent projection technique to be applied post hoc for identifying possibly-optimal arms. Nevertheless, such an approach overlooks the temporal dynamics inherent to the non-stationary process, thereby failing to capture how specific interventions affect rewards across different time steps. In what follows, we compare this projection-based approach with our method.

First of all, we introduce the projection-based approach. In a stationary setting with non-manipulable variables, Lee and Bareinboim [2019] characterize possibly-optimal arms using the latent projection technique [Verma and Pearl, 1990, Verma, 1992]. They begin with a causal diagram G = ⟨ V , E ⟩ , and define a set of non-manipulative variables N ⊂ V \ { Y } , where Y is the target variable. To construct

the latent projection onto the manipulative variables V \ N , they consider an augmented graph ̂ G that explicitly represents unobserved confounders. They initialize a graph H = ⟨ V \ N , ∅⟩ , then add edges as follows:

1. A directed edge between V i and V j if V i → V j ∈ G or there exists a directed path from V i to V j where all non-end vertices in the path between there are in N .
2. A bi-directed edge between V i and V j if V i ↔ V j ∈ G ; or there exists directed paths from an unobserved confounder to V i and V j in ˆ G where all non-end vertices are in N .

Let G [ V ′ ] denote the causal diagram resulting from projecting G onto V ′ . They prove that P N G ,Y = P H ,Y (i.e., POMISs given ⟨G , Y, N ⟩ = POMISs given ⟨H , Y ⟩ ) via two propositions, which ensures that the optimality of an arm remains under (i) projection from G to H and (ii) the reverse projection.

̸

Proposition J.1 (Causal Identification without Non-manipulative Variables [Lee and Bareinboim, 2019]) . Given an SCM M 1 = M = ⟨ V , U , F , P ( U ) ⟩ , there exists an SCM M 2 = M V \ N = ⟨ V , U , F ′ , P ( U ) ⟩ such that P 1 x ( y ) = P 2 x ( y ) for any X , Y ⊂ V \ N and Y = ∅ .

̸

Proposition J.2 (Causal Identification under the Projected Graph [Lee and Bareinboim, 2019]) . Given a causal diagram G , let H = G [ V \ N ] . For a SCM M 1 = M [ V \ N ] = ⟨ V \ N , U , F ′ , P ( U ) ⟩ conforming to H , there exists a SCM M 2 = M = ⟨ V , U , F , P ( U ) ⟩ that conforms to G such that P 1 x ( y ) = P 2 x ( y ) , for any X , Y ⊆ V \ N and Y = ∅ .

Theorem J.1 (POMIS Invariance under Projection) . Given a causal diagram G = ⟨ V , E ⟩ , a reward variable Y ∈ V , and a set of non-manipulable variables N ⊆ V \ { Y } , let H be the projection of G onto V \ N . Then,

<!-- formula-not-decoded -->

where P N G ,Y denotes a set of POMISs given ⟨G , Y, N ⟩ .

Based on the projection-based approach introduced above, we can derive Fig. 8 via the latent projection. In Fig. 14, we obtain POMIS = { Z 1 , X 1 , Z 2 , W 2 , Z 3 , W 3 , W 4 } . This result coincides with the outcome of our proposed POMIS + -based method (Fig. 8) in terms of maximizing the cumulative reward Y ′ = ∑ 4 t =1 Y t . However, since the graph in Fig. 14 is constructed by projecting all temporal models into a single static, stationary structure, it lacks temporal interpretability. We argue that such a projection is insufficient for modeling and analyzing non-stationary bandit problems, where the notion of time is inseparable from the causal dynamics.

Non-stationarity in bandits inherently involves modeling how reward distributions evolve over time, and our method is specifically designed to capture and exploit this temporal evolution. The significant advantage of our framework lies in its ability to identify time-specific intervention sets for each reward variable Y t . This makes POMIS + not only optimal in terms of cumulative reward but also more interpretable and practically applicable to real-world scenarios where intervention constraints or objectives vary over time. Below, we present one example that highlights this property.

Illustrative case: Sequential treatment with risk constraint Suppose each time-indexed intervention variable Z t , X t , W t represents a distinct medication administered at time t :

- Z t : fast-acting drug that strongly affects immediate health ( Y t ) but may induce side effects.
- X t : slow-acting drug that influences future outcomes (e.g., Y t +1 , Y t +2 ).
- W t : a supportive drug that amplifies or stabilizes drug effects in the future.

Now consider a scenario where we impose a safety constraint on early reward:

Let A = ( { Z 1 , X 1 } , { Z 2 , W 2 } , { Z 3 , W 3 } , { W 4 } ) denote the intervention sequence obtained from the POMIS + method (or projection-based method) across all time steps, and let a 0 ∈ D ( A [0]) be a joint intervention assignment to the first intervention set.

<!-- formula-not-decoded -->

to ensure that early treatments do not induce excessive physiological stress. A projection-based method (Fig. 14) optimizes E [ Y ′ ] and may select ( Z 1 = 1) if it increases Y ′ overall-despite the fact that Z 1 directly affects Y 1 and may violate the safety constraint.

In contrast, our method (Fig. 8) distinguishes that:

- Z 1 is QIB 1 influencing primarily Y 1 , while
- X 1 is IB + 1 , 4 that contributes to Y 4 through the path X 1 → X 2 → X 3 →··· → Y 4 .

By leveraging this temporal decomposition, we can selectively intervene on X 1 while avoiding Z 1 , satisfying the constraint on Y 1 and still improving long-term outcomes. Formally, we solve:

<!-- formula-not-decoded -->

Summary This example highlights the limitations of projection in non-stationary bandits. While projection-based methods may yield high cumulative rewards, they are blind to when and how specific interventions act. Our POMIS + framework enables temporally structured intervention planning, allowing for interpretable, constraint-aware, and sequentially optimal policies. In domains such as medicine or education-where interventions at each stage must consider safety or ethical constraints-such temporal disentanglement is essential.

## J.2 Forced stationary approach on NS-SCM-MAB

Figure 15: (a) Non-stationary causal diagram G and (b, c) mutilated graphs

<!-- image -->

In the main text, we analyzed the non-stationary SCM-MAB setting in which the reward distribution changes over time due to the causal influence of past interventions (see Def. 3.1). While our main algorithmic framework leverages time-specific causal structure, it is also possible to consider an alternative approach: forced stationarity via direct intervention. The idea is to identify intervention sets that block the information propagation (see §3) across time slices-thus preventing reward distribution shifts. This allows the agent to reuse previously learned interventional effects, reducing the need to re-identify POMIS sets or re-calculate expected rewards at every time step. Before formalizing this concept, we introduce the notion of transition edge blocking.

Definition J.1 (Block) . Let G be a causal diagram and V t the set of endogenous variables at time t . An intervention do( X t = x t ) is said to block transition edges if, in the mutilated graph G X t , all incoming edges from variables in previous time slices V &lt;t to X t are removed. This operation prevents the information propagation from V &lt;t to V t via X t .

When such a blocking intervention is applied, the reward variable Y t becomes conditionally isolated from its historical influences. This motivates the following definition of forced stationarity, grounded in the interventional reward distributions formalized in Def. 3.1.

Definition J.2 (Forced Stationarity) . Given a non-stationary SCM-MAB ⟨M , Y ⟩ , we say that an intervention set F ⊆ V t induces forced stationarity over reward variable Y t if for every t &lt; t ′ , and for every f ∈ D ( F ) ,

<!-- formula-not-decoded -->

In other words, the reward distribution remains invariant over time under repeated interventions on F .

This formalization allows us to distinguish between actual stationarity (inherent in the SCM) and stationarity that is induced by intervention. The following lemma clarifies the relationship between blocking transition edges and enforcing reward stationarity.

Lemma J.1 (Blocking transition edges induces forced stationarity) . Let G be a causal diagram, and suppose an intervention on { Z t } blocks the transition edge Z t -1 → Z t . Then, we have:

<!-- formula-not-decoded -->

Therefore, the intervention enforces reward stationarity across time.

Proof. By intervening on Z t , we remove the transition edge from Z t -1 to Z t (Def. J.1), thereby preventing any causal effect from prior interventions from propagating to Z t . As a result, Y t under do( Z t ) becomes conditionally independent of earlier actions, and its interventional distribution same as that of Y t -1 under do( Z t -1 ) .

The idea of forced stationarity provides a mechanism to simplify learning in non-stationary SCMMABs: if one can construct a policy that always intervenes on such blocking variables (e.g., Z t ), the resulting reward distribution no longer shifts over time. This allows the agent to reuse previously learned information without recalculating POMISs at each step.

However, such interventions may reduce the optimality of the resulting policy. Although reward distributions may appear stationary under intervention F , the values themselves may not be maximized. Blocking information from previous time slices may prevent the agent from exploiting long-term causal pathways that yield higher rewards.

To illustrate this trade-off, consider the causal graphs in Fig. 15. In Fig. 15(b), we apply an intervention on X 2 , which removes the incoming edge from X 1 to X 2 , thereby blocking temporal information transfer and enforcing stationarity. While this simplifies the learning problem, it overlooks alternative intervention options. In Fig. 15(c), we do not intervene on X 2 , but instead on W 2 -a variable that is structurally valid and potentially in maximizing reward, even though it does not block the transition path.

This example reveals an important limitation of the forced stationarity paradigm: sets such as { X 2 } and { W 2 } both are interventional border (IB), yet only X 2 induces stationarity when intervened. If one naively prioritizes only those IB variables that block transition edges, the policy may become overly myopic and ignore possibly higher-rewarding options. In contrast, our framework (i.e., POMIS + ) retains the full temporal structure and dynamically adapts to changes in reward distribution without forcibly blocking information. This enables both fine-grained interpretability and optimal intervention selection over time.

While the limitations of forced stationarity are evident in scenarios where blocking discards useful long-term causal dependencies, we emphasize that this strategy is not inherently flawed. In fact, under certain structural conditions, forced stationarity may serve as a practical approach to simplifying learning in temporally complex environments. For example, if the reward variable Y t consistently depends on a repeated set of parent variables over time-such as through recurring transition edges-then blocking those transitions can yield invariant reward distributions without sacrificing performance.

In such cases, domain knowledge or structural priors can inform targeted interventions that induce stationarity while still capturing the most influential causal paths. This highlights a broader point: rather than rejecting forced stationarity entirely, it can be selectively applied when supported by sufficient knowledge about the system's temporal regularities.

## K More Discussions

## K.1 Relaxing Natural Characteristics on Temporal Structure

Our main theoretical development naturally assumes both an identical graphical structure across time slices and the existence of transition edges between them. These assumptions are introduced to reflect natural characteristics of non-stationary environments and to facilitate tractable inference. Importantly, the assumption of identical structure is not essential to the correctness of our algorithm: even when the assumption is relaxed, our method remains sound and complete, provided that the full temporal causal graph is accessible.

In contrast, the existence of transition edges plays a more critical role. Without such edges, causal information cannot propagate across time, and the algorithm may fail to identify relevant variables at earlier steps, limiting its ability to construct non-myopic intervention strategies.

- The property of identical time slices reflects the natural structure of stationary bandit models (see, Fig. 1(b)), where variables and their dependencies are replicated across time for each time slice. While this natural assumption simplifies the search space by enabling repeated reasoning patterns, which is a characteristic of bandit settings, it is not required for the correctness of the graphical analysis or the definitions of IB + and QIB.
- The existence of transition edges between time slices enables information to propagate forward in time, allowing the algorithm to identify earlier variables that support future interventions. Without transition edges, information cannot propagate across time, preventing the algorithm from identifying long-range dependencies or constructing effective nonmyopic strategies. More concretely, certain paths to earlier supporting variables (i.e., IB + ) become unreachable, limiting the algorithm's ability to exploit temporal structure for longterm optimization.

This flexibility ensures that our framework is broadly applicable, even in domains where temporal structure varies over time. We emphasize that these assumptions serve as modeling conveniences that facilitate analysis and interpretation, rather than strict algorithmic requirements.

## K.2 Non-Stationarity in Causal Bandits vs. Causality in Non-Stationary Bandits

It is useful to distinguish our formulation from alternative approaches that combine causality and non-stationarity in opposite directions. Our work builds upon structural causal bandits (SCM-MABs) [Lee and Bareinboim, 2018, 2019], which assume a fixed underlying causal graph and utilize it to guide interventions. We extend this line of work by introducing non-stationarity at the level of the causal structure itself-modeling how causal dependencies and reward mechanisms evolve over time. In this sense, we develop a causal bandit framework that internalizes non-stationarity as a structural property.

By contrast, recent efforts that introduce causal reasoning into non-stationary bandit settings typically begin with traditional statistical formulations-such as sliding-window or change-point models-and incorporate causal estimators to correct for confounding or adapt to changing environments (details are available in Appendix E). For instance, Huang and Wu [2024] propose a method for handling confounded and selection-biased offline data by deriving robust causal bounds for each arm. Similarly, Nourani-Koliji et al. [2023] address piecewise-stationary bandits with causally related rewards by detecting changes in both reward distributions and causal structure using a Generalized Likelihood Ratio(GLR)-based change-point detector. In these settings, causality is layered onto a fundamentally statistical treatment of temporal change, often without structural assumptions about the underlying dynamics.

The distinction is more than philosophical. Our model-based approach enables:

- Reasoning about how non-stationarity arises (via temporal models and transition edges),
- Derivation of non-myopic intervention strategies through temporally-aware graphical structures (POMIS + ), and
- Theoretical guarantees grounded in structural assumptions.

In contrast, approaches based on statistical modeling typically emphasize adaptation-e.g., through sliding windows, discounting, or change-point detection-without modeling the underlying structural mechanisms that generate non-stationarity. These methods act as black-box in the sense that they detect shifts or trends in the data but do not explain them in terms of causal relationships or system dynamics.

## K.3 POMIS + Sequences as Composition of Possibly Optimal Intervention Sets within Temporal Models

An intervention sequence and the causal responses of each time-indexed variable determine the expected cumulative reward in a sequential decision problem. In our framework, this is formal-

ized by an SCM M together with a sequence of possibly-optimal intervention sets-namely, POMIS + -identified across time.

While SCM-level definitions of POMIS (cf. Def. B.2) evaluate optimality based on the existence of at least one SCM where a given intervention set performs optimally, temporal decision-making imposes a stricter constraint: each reward Y t must be evaluated in the context of prior interventions. This context naturally gives rise to a family of temporal models M t | v ⋆ t , where v ⋆ t represents the predetermined values induced by earlier actions. Thus, the optimal intervention at each time step may be determined within this temporal model, reflecting only the information propagation available at time t .

In this subsection, we show that the expected cumulative reward under a full SCM can be equivalently decomposed into a sum of expected rewards under temporal models. Each such term corresponds to an intervention on a POMIS + set at time t , selected with respect to M t | v ⋆ t . The result is formalized in Prop. K.1, which clarifies how the temporal roll-out of POMIS + constructs a compositionally valid and possibly-optimal global intervention plan.

Proposition K.1 (Composition of Temporal Optima) . Let M be an SCM conforming to G , and let ( X 1 , . . . , X T ) be an optimal intervention sequence with realizations x ∗ 1 , . . . , x ∗ T . Let v ⋆ t denote the predetermined variables at time t given past interventions. Suppose each intervention set X t is constructed as POMIS t ∪ W t , where W t ⊆ Pa ( V t ′ ) satisfies the condition in Thm. 5.1. Then:

<!-- formula-not-decoded -->

Proof. We begin by expanding the expected cumulative reward under the full SCM M :

<!-- formula-not-decoded -->

For each t ∈ [ T ] , let v ⋆ t denote the predetermined assignment to Pa ( V t ) induced by prior interventions do( x ∗ 1 , . . . , x ∗ t -1 ) . Then the reward at time t can be equivalently computed in the conditioned temporal model M t | v ⋆ t as:

<!-- formula-not-decoded -->

Applying this to each t and summing, we obtain:

<!-- formula-not-decoded -->

as claimed.

This proposition formalizes how the cumulative reward in a full SCM can be decomposed into a sequence of rewards, each governed by a temporal model conditioned on prior interventions (i.e., predetermined values). It supports the view that the POMIS + sequence acts as a possibly-optimal plan across time, even though the optimality of individual interventions may depend on specific prior values.

While each temporal model M t | v ⋆ t determines the optimal intervention at time t under a fixed history of prior interventions, the globally optimal sequence across the entire SCM must be selected jointly, since each temporal model depends on the conditioning induced by earlier decisions.

Corollary K.1 (Local Optimality Does Not Imply Global Optimality) . There exists an SCM M and intervention sequences ( X 1 , . . . , X T ) such that for each t , X t is a possibly-optimal intervention set in the temporal model M t | v ⋆ t , but the joint sequence ( X 1 , . . . , X T ) is not globally optimal for E M [ ∑ T t =1 Y t ] .

This corollary illustrates that local optimality under temporal models does not guarantee global optimality in the full SCM. Due to the interdependencies between time steps, early interventions may have long-term consequences that are not captured by myopic optimization. Therefore, joint planning across time is necessary to achieve globally optimal decisions.
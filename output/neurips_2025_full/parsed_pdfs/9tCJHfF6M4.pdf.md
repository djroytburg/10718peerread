## Tightening Regret Lower and Upper Bounds in Restless Rising Bandits

## Cristiano Migali

Politecnico di Milano, Milan, Italy cristiano.migali@mail.polimi.it

## Gianmarco Genalti

Politecnico di Milano, Milan, Italy gianmarco.genalti@polimi.it

## Marco Mussi

Politecnico di Milano, Milan, Italy marco.mussi@polimi.it

## Alberto Maria Metelli

Politecnico di Milano, Milan, Italy albertomaria.metelli@polimi.it

## Abstract

Restless Multi-Armed Bandits (MABs) are a general framework designed to handle real-world decision-making problems where the expected rewards evolve over time, such as in recommender systems and dynamic pricing. In this work, we investigate from a theoretical standpoint two well-known structured subclasses of restless MABs: the rising and the rising concave settings, where the expected reward of each arm evolves over time following an unknown non-decreasing and a non-decreasing concave function, respectively. By providing a novel methodology of independent interest for general restless bandits, we establish new lower bounds on the expected cumulative regret for both settings. In the rising case, we prove a lower bound of order Ω p T 2 { 3 q , matching known upper bounds for restless bandits; whereas, in the rising concave case, we derive a lower bound of order Ω p T 3 { 5 q , proving for the first time that this setting is provably more challenging than stationary MABs. Then, we introduce Rising Concave Budgeted Exploration ( RC-BE p α q ), a new regret minimization algorithm designed for the rising concave MABs. By devising a novel proof technique, we show that the expected cumulative regret of RC-BE p α q is in the order of r O p T 7 { 11 q . These results collectively make a step towards closing the gap in rising concave MABs, positioning them between stationary and general restless bandit settings in terms of statistical complexity.

## 1 Introduction

Multi-Armed Bandits (MABs, Lattimore and Szepesvári, 2020) are a well-known framework to model decision-making problems, where, for each round, an agent chooses ( pulls ) an action ( arm ) among a set of available actions and observes a reward , i.e., numerical feedback which represents the goodness of the choice. In this setting, the goal of the learner is to minimize the expected cumulative regret accumulated during the interaction, i.e., the sum over time of the difference between the expected reward of the optimal arm and that of the chosen one. The standard MAB setting considers stationary reward distributions. However, in many real-world decision-making problems, the expected rewards of available actions can vary over time due to changes in the surrounding environment, such as shifting in consumer preferences for online marketplaces (Wu et al., 2018) or evolving health status of patients in treatment selection during clinical trials (Aziz et al., 2021). To address such dynamics, the restless MABs framework (Tekin and Liu, 2012) has been introduced. This model generalizes the classical MAB setting by explicitly incorporating the non-stationarity of the arms. 1

1 With a slight abuse of terminology, we will use the words non-stationary and restless interchangeably.

Table 1: Existing and new bounds for the restless , restless rising and restless rising concave settings. The arrow Ñ points from the previous best result to the improved one presented in this paper.

|              | Holds for   | Holds for       | Holds for               |                                                                    |
|--------------|-------------|-----------------|-------------------------|--------------------------------------------------------------------|
|              | Restless    | Restless Rising | Restless Rising Concave | Result                                                             |
| Lower Bounds | ✓ ✓ ✓       | ✓ ✗ ✓           | ✓ ✗ ✗ ✓                 | Ω p T 1 { 2 q Ω p T 2 { 3 q Ω p T 1 { 2 qÑ Ω p T 2 { 3 1 { 2 3 { 5 |
|              | ✓           | ✓               |                         | q Ω p T qÑ Ω p T q                                                 |
| Upper Bounds | ✓ ✗ ✗       | ✓ ✗ ✗           | ✓ ✓ ✓                   | O p T 2 { 3 q r O p T 2 { 3 q O p T 2 { 3 qÑ r O p T 7 { 11 q      |

Non-stationarity in bandit problems has been addressed through a variety of models and methods, such as restless bandits with abrupt changes in the reward distribution (e.g., Garivier and Moulines, 2011), smoothly evolving expected rewards (e.g., Trovò et al., 2020), and settings where the total variation of expected rewards is bounded over time (e.g., Besbes et al., 2014). These frameworks allow the expected rewards to fluctuate in complex ways, such as increasing and then decreasing, without constraints on their direction of change. In contrast, there are important classes of bandit models that enforce monotonicity on the expected rewards. These include rising bandits (Heidari et al., 2016; Metelli et al., 2022), where expected rewards are non-decreasing, and rotting bandits (Levine et al., 2017; Seznec et al., 2019, 2020), where they are non-increasing. Such models are well-suited for capturing structured real-world dynamics, including online model selection (Metelli et al., 2022), hyperparameter optimization (Mussi et al., 2024), and recommendation systems (Levine et al., 2017).

Motivation. In this work, we focus on the restless rising bandits and restless rising concave bandits and we aim to characterize them from a theoretical standpoint since several fundamental questions remain unresolved. In the general restless bandit setting, where the expected rewards may vary over time with bounded variation over T rounds, the minimax regret is known to be lower bounded by Ω p T 2 { 3 q (Besbes et al., 2014). 2 However, no regret lower bound has been derived for the specific class of non-decreasing (rising) or non-decreasing concave (rising concave) restless bandits yet, making the classical lower bound for stationary bandits, Ω p T 1 { 2 q (Lattimore and Szepesvári, 2020, Thm. 15.2), the best available reference, and leaving the following question open.

Question 1 : Is it possible to conceive regret lower bounds for restless rising and restless rising concave bandits that are strictly larger than the Ω p T 1 { 2 q bound for stationary bandits?

The currently available algorithms for restless rising bandits are those designed for general restless bandits with bounded variation, which achieve a regret upper bound of order O p T 2 { 3 q (Besbes et al., 2014). When incorporating concavity, more specific algorithms have been proposed (Metelli et al., 2022), but unfortunately, they fail to improve the regret order. This generates the following question.

Question 2 : Is it possible to devise algorithms for restless rising and rising concave bandits whose regret upper bounds are strictly smaller than the O p T 2 { 3 q bound for general restless bandits?

Original Contribution. In this paper, we aim to provide an answer to the research questions presented above, making a step towards the complete statistical characterization of restless rising and restless rising concave bandits. The contribution is summarized as follows:

- In Section 3, we provide a general recipe for deriving regret lower bounds for restless bandits, which generalizes the construction of Besbes et al. (2014) and is of potential independent interest (Lemma 3.1). We then specialize this construction to the cases of rising and rising concave bandits. First, we derive a lower bound of order Ω p T 2 { 3 q for rising bandits, showing that this setting shares the same statistical complexity as general restless bandits (Theorem 3.2) and answering negatively

2 We use Ω p¨q and O p¨q to highlight the dependence on T in the lower and upper bounds, respectively, omitting constant factors. For upper bounds, we also use r O p¨q to suppress logarithmic dependencies on T too.

- to Question 2 for rising bandits. Second, for restless rising concave bandits, we show that the regret is at least of order Ω p T 3 { 5 q , showing that this setting is more challenging than stationary MABs (Theorem 3.3). These results provide a positive answer to Question 1 for both settings.
- In Section 4, we present Rising Concave Budgeted Exploration ( RC-BE p α q ), a novel regret minimization algorithm for restless rising concave MABs, which extends Budgeted Exploration (Jia et al., 2023). By devising a novel analysis, we provide an upper bound on its regret of order r O p T 7 { 11 q (Theorem 4.4) with no requested knowledge of the learning horizon or of the total variation. This result improves upon the current best upper bound of order O p T 2 { 3 q and provides a positive answer to Question 2 for rising concave bandits.

Numerical simulations are provided in Section 5. Related works are discussed in Appendix A. Omitted proofs are provided in Appendices B and C for lower and upper bounds, respectively. A summary of known and new results presented in this paper is provided in Table 1.

## 2 Setting

A restless K -armed MAB (Tekin and Liu, 2012; Lattimore and Szepesvári, 2020) is defined as a vector of probability distributions ν ' p ν i q i P J K K , where ν i : N ě 1 Ñ ∆ p R q . 3 Let T P N ě 1 be the learning horizon, at each round t P J T K , the agent selects an arm I t P J K K and observes a reward R t ' X I t ,t where X i,t ' ν i p t q for all i P J K K , t P N ě 1 . We denote the random table with all possible rewards as X ' p X i,t q i P J K K ,t P N ě 1 . For every arm i P J K K , we define its expected reward µ i : N ě 1 Ñ R as the expectation of the reward obtained by pulling such arm, i.e., µ i p t q ' E X ' ν i p t q r X s and denote the vector of expected reward functions as µ ' p µ i q i P J K K . We assume that the expected rewards are bounded in r 0 , 1 s , and that the realizations are σ -subgaussian. 4

Rising Bandits. We revise the rising bandits notion, i.e., MABs with non-decreasing expected rewards (Heidari et al., 2016). Such a property is captured by the following assumption.

Assumption 2.1 (Non-Decreasing expected reward) . Let ν be a restless MAB. For every arm i P J K K and round t P N ě 1 , the function µ i p t q is non-decreasing in t . In particular, defining the increments:

<!-- formula-not-decoded -->

We introduce a further assumption on the concavity of the expected rewards (Heidari et al., 2016).

Assumption 2.2 (Concave expected reward) . Let ν be a restless MAB. For every arm i P J K K and round t P N ě 1 , the function µ i p t q is concave in t , i.e.:

<!-- formula-not-decoded -->

Formally, we call restless rising a restless MAB in which Assumption 2.1 holds, and restless rising concave a restless MAB in which both Assumptions 2.1 and 2.2 hold. From now on, we omit the adjective restless for the sake of conciseness.

Learning Problem. Let t P N ě 1 be a round, we denote with H t ' p I l , R l q t l ' 1 the history of observations up to t . A (non-stationary deterministic) policy is a function π : H t ´ 1 ÞÑ I t mapping a history to an arm, that is abbreviated as π p t q : ' π p H t ´ 1 q . We define the performance of a policy π in a restless MAB ν as the expected cumulative reward collected over the T rounds, formally:

<!-- formula-not-decoded -->

A policy π ˚ ν is optimal if it maximizes the expected cumulative reward: π ˚ ν P arg max π t J ν p π, T qu . In restless MABs, the optimal policy does not explicitly depend on T and consists of pulling in each round the arm with the highest expected reward: π ˚ ν p t q P arg max i P J K K µ i p t q for every t P N ě 1 . Denoting with J ˚ ν p T q : ' J ν p π ˚ ν , T q the expected cumulative reward of an optimal policy, the suboptimal policies π are evaluated via the expected cumulative regret :

<!-- formula-not-decoded -->

3 Let a, b P N ě 1 , b ě a , we denote with J a, b K : ' t a . . . , b u , with J a K : ' J 1 , a K , and with ∆ p X q the set of probability measures over the measurable set X .

4 A random variable X is σ -subgaussian if E r e λ p X ´ E r X sq s ď e σ 2 λ 2 2 , for every λ P R .

Instances Characterization. To characterize an instance ν , we introduce the following quantity, namely the cumulative increment , defined for every t 1 , t 2 P N ě 1 with t 1 ď t 2 as:

<!-- formula-not-decoded -->

The cumulative increment extends to an arbitrary interval with t 1 and t 2 as extremes the analogous notion Υ µ p T, q q employed in (Metelli et al., 2022), restricting to q ' 1 . It is immediate to show that Υ ν p t 1 , t 2 q P r 0 , K s since Υ ν p t 1 , t 2 q ď ř t 2 ´ 1 l ' t 1 ř i P J K K γ i p l q ď ř i P J K K 1 ' K . Analogously to what is done in (Besbes et al., 2014), we consider the class of instances whose cumulative increment over the learning horizon T is bounded by a variation budget V T P p 0 , K s , which we assume known, formally Υ ν p 1 , T q ď V T . Then, we call, respectively, E σ r p T, V T q and E σ c p T, V T q the set of rising MABs and rising concave MABs instances, with σ -subgaussian rewards, whose Υ ν p 1 , T q satisfies the previous inequality.

## 3 Lower Bounds

In this section, we analyze the statistical complexity of the learning problem in both the rising and rising concave settings. To this end, we provide a regret lower bound suffered by any deterministic policy π on a class of instances which are rising and rising concave, respectively. 5 In particular, we show that rising MABs are not easier than restless MABs with bounded variation (Besbes et al., 2014, Thm. 1) and that rising concave MABs are harder than stationary MABs (Lattimore and Szepesvári, 2020, Thm. 15.2). The analysis is carried out as follows. We develop a general recipe for regret lower bound construction on a richer class of restless MABs, described in Section 3.1. Then we specialize it to both the settings of interest (Sections 3.2 and 3.3).

## 3.1 General Recipe for the Lower Bound

We consider a class of restless MABs with the following structure. The set of rounds N ě 1 is split into windows. Let p ∆ w q w P N ě 1 where ∆ w P N ě 1 be a sequence of window widths . A window consists of a set of rounds J s w , e w K Ă N ě 1 where s w : ' ř w ´ 1 l ' 1 ∆ l ` 1 and e w : ' ř w l ' 1 ∆ l , for w P N ě 1 . For each window index w P N ě 1 , we define two functions µ w , r µ w : J ∆ w K Ñr 0 , 1 s , which we call base and modified trend respectively, that describe how the expected rewards of the arms evolve in J s w , e w K . In particular, in each window, at most one arm among the K has expected reward that follows the modified trend, while all the others have expected rewards that follow the base trend. The arm whose expected reward follows the modified trend can change between windows. We further enforce µ w p t q ď r µ w p t q for all w P N ě 1 , t P J ∆ w K , 6 so that the arm whose expected reward follows the modified trend is the optimal one. More formally, let w p t q : ' min t w P N ě 1 s.t. e w ě t u be the index of the window which contains the round t P N ě 1 . For each sequence o ' p o w q w P N ě 1 with o w P J 0 , K K in each window of index w and for each subgaussian parameter σ ě 1 , we define an instance ν σ o ' p ν σ o ,i q i P J K K as follows:

<!-- formula-not-decoded -->

where ψ p µ, σ q is a probability distribution with parameters µ P r 0 , 1 s , σ ě 1 such that if X ' ψ p µ, σ q , then:

<!-- formula-not-decoded -->

First of all observe that µ P r 0 , 1 s , σ ě 1 imply µ {p 2 σ q P r 0 , 1 { 2 s , so that the distribution is welldefined. Furthermore, if X ' ψ p µ, σ q , then, in virtue of Hoeffding's lemma, X is σ -subgaussian, and, by direct calculation, it has expected value equal to µ . Notice that, if o w ' 0 , all the arms follow the

5 Since we are considering stochastic bandits, our lower bounds can be generalized to stochastic policies, yielding analogous results, at the cost of additional notational complexity.

6 We consider µ w p t q and r µ w p t q both in the domain t P J ∆ w K instead of in the domain J s w , e w K , for the sake of simplicity in the notation, as every window is defined independently from the others.

base trend, otherwise, o w corresponds to the only arm following the modified trend. We denote with µ ' p µ w q w P N ě 1 and r µ ' p r µ w q w P N ě 1 the sequences of base and modified trends respectively, and with E σ µ , r µ ' t ν σ o s.t. o P J 0 , K K N ě 1 u the class of instances that they induce by varying the sequence o of optimal arms in each window. The following result, whose proof is deferred to Appendix B, holds.

Lemma 3.1 (General Lower Bound) . Under the assumption that µ w p t q ď r µ w p t q for all w P N ě 1 , t P J ∆ w K , for any deterministic policy π , subgaussian parameter σ ě 1 , and learning horizon T P N ě 1 , it holds that:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

for all w P J w p T q K , with D KL p¨}¨q being the Kullback-Leibler divergence of the two distributions (formally defined in Appendix B).

This result highlights the trade-off in designing a 'challenging' restless instance. On the one hand, we do not want to make the base and modified trends too far apart, otherwise it would be easy for the agent to discern one from the other. This is reflected in Equation (3), as the term D µ , r µ ,T,σ w increases when the two trends diverge and contributes to reducing the regret lower bound since A µ , r µ ,T w is non-negative by construction. On the other hand, we want to maximize the area A µ , r µ ,T w between the two trends. In this way, under the assumption that D µ , r µ ,T,σ w is small enough so that the factor that multiplies A µ , r µ ,T w is non-negative, we increase the regret lower bound.

## 3.2 Specializing the Lower Bound for the Rising Setting

In this part, we apply Lemma 3.1 to provide a regret lower bound for the class E σ r p T, V T q , holding for any deterministic policy π . To this end, we construct sequences of window widths p ∆ r ,w q w P N ě 1 and of base and modified trends µ r , r µ r such that E σ µ r , r µ r Ď E σ r p T, V T q . A representation of the structure of the instances is depicted in Figure 1. We choose windows of the same width. In each window, the base and modified trend are both constant, the latter is greater than the former by a quantity ε r ą 0 and the value of the modified trend in a window corresponds to the value of the base trend in the next window. In this way, we guarantee that the instances are rising no matter which arm follows the modified trend. In Appendix B, we formalize the instances and we prove that the following holds.

Theorem 3.2 (Lower Bound for the Rising Setting) . For any deterministic policy π , subgaussian parameter σ ě 1 , and learning horizon T P N ě 1 , T ě σ 2 K min t 1 , V T u ´ 2 , it holds that:

<!-- formula-not-decoded -->

The orders of growth for T , K , and V T in this result match the upper bound for the general restless case with bounded variation (Besbes et al., 2014, Thm. 2) when V T ď 1 . 7 This implies that rising MABs are not easier than general restless MABs with bounded variation despite the additional assumption. Thus, the characterization of the statistical complexity of this setting is completed.

## 3.3 Specializing the Lower Bound for the Rising Concave Setting

In this part, we provide a regret lower bound for the class E σ c p T, V T q holding for any deterministic policy π . In analogy to Section 3.2 for the rising setting, we construct sequences of window widths

7 Webelieve this is an artifact of the analysis since, in our the lower bound construction, we have Υ p 1 , T q ď 1 .

Figure 1: Base (dashed) and modified (solid) trends of the lower bound instances for the rising setting.

<!-- image -->

Figure 2: Base (dashed) and modified (solid) trends of the lower bound instances for the rising concave setting.

<!-- image -->

p ∆ c ,w q w P N ě 1 and of base and modified trends µ c , r µ c such that E σ µ c , r µ c Ď E σ c p T, V T q . Arepresentation of the instances is depicted in Figure 2. We choose again windows of the same width. In each window, the base and modified trends share the same starting and ending values. Furthermore, the end value of expected rewards in a window matches the start value of expected rewards in the next window. The end value is greater than the start value to guarantee that the instances are rising. The base trend joins the two endpoints of the expected rewards of each window with a single segment, while the modified trend uses two segments. At the beginning, it rises with a slope greater than that of the base trend until half the window. At this point, the distance between the base and the modified trend in the window is maximum. Then, the modified trend keeps rising, but with a slope that is smaller than that of the base trend, until the two trends meet at the end of the window. The pattern repeats and the slopes are chosen in such a way that the slope of the second part of the modified trend in a window (which is the smallest slope in a window) corresponds to the slope of the first part of the modified trend in the next window (which is the greatest slope of an expected reward in a window). In this way, we guarantee that the instances are rising and concave, no matter the choice of which arm follows the modified trend. In Appendix B, we formally present the instances and we prove the following result.

Theorem 3.3 (Lower Bound for the Rising Concave Setting) . For any deterministic policy π , subgaussian parameter σ ě 1 , and learning horizon T P N ě 1 , T ě 2 10 σ 2 K min t 1 , V T u ´ 2 , it holds that:

<!-- formula-not-decoded -->

This result proves that regret minimization in rising concave MABs represents a harder learning problem w.r.t. stationary MABs which are characterized by the usual Ω p T 1 { 2 q lower bound.

## 4 Upper Bound for the Rising Concave Setting

In this section, we present a novel regret minimization algorithm, Rising Concave Budgeted Exploration ( RC-BE p α q ), designed for rising concave MABs (Algorithm 1), and analyze its performance by providing an upper bound of the expected cumulative regret suffered on a generic instance ν P E σ c p T, V T q . We show that this upper bound attains a strictly smaller rate w.r.t. the lower bound on the expected cumulative regret on a generic restless MAB with bounded variation (Besbes et al., 2014), and thus that rising concave MABs are indeed an easier setting w.r.t. them.

Algorithm. RC-BE p α q is an improvement of the Budgeted Exploration ( BE ) algorithm (Jia et al., 2023), originally designed for 2 -armed general restless bandits. 8 The original BE algorithm works as follows. The learning horizon T is split in windows of ∆ P N ě 1 rounds each. In each window, the algorithm restarts. At the beginning of each window, the agent carries out an exploration phase which consists of several round-robin cycles. In particular, the agent keeps track of the arms alive in

8 The extension of BE to K -armed bandits is proposed in the unpublished preprint (Jia et al., 2024) for the case of smooth MABs. However, we have found soundness issues in the analysis proposed there (see Appendix F). For this reason, we will develop an independent analysis which overcomes these issues.

```
Algorithm 1 RC-BE p α q . 1: Input : α ě 1 , K P N ě 2 2: Initialize w Ð 1 , d Ð 1 , A Ð J K K , B Ð A , ˆ S i Ð 0 , @ i P J K K 3: for t P J T K do 4: if d ' ∆ p α q w ` 1 then 5: Increment w Ð w ` 1 6: Reset d Ð 1 , A Ð J K K , B Ð A , ˆ S i Ð 0 , @ i P J K K 7: end if 8: Pull I t P B 9: Remove B Ð B zt I t u 10: Observe R t ' X I t ,t 11: Update ˆ S I t Ð ˆ S I t ` R t 12: if B ' tu then 13: Compute ˆ S ˚ Ð max i P A ˆ S i 14: for i P J K K do 15: if i P A and ˆ S i ` B p α q w ă ˆ S ˚ then 16: Remove A Ð A zt i u 17: end if 18: end for 19: Reset B Ð A 20: end if 21: Increment d Ð d ` 1 22: end for
```

the current window in a set A Ď J K K , initialized to J K K at the beginning of each window, and, in each round-robin cycle, pulls each of these arms once. The agent cumulates the observed rewards for each arm in the variables ˆ S i with i P J K K . At the end of each round-robin cycle, the agent compares the cumulative reward of each alive arm with the maximum cumulative reward among alive arms ˆ S ˚ : ' max i P A ˆ S i . If for i P A we have ˆ S i ` B ă ˆ S ˚ , where B ą 0 is a parameter of the algorithm, we say that arm i has run out of budget and the agent removes it from the set of alive arms. It can happen that, after several round-robin cycles, the set of alive arms becomes a singleton: A ' t ˆ i ˚ u . In this case, no more eliminations can happen and the agent will commit to the remaining arm ˆ i ˚ .

RC-BE p α q extends the original algorithm as follows. It exploits the concavity of the instance through increasing window widths ∆ p α q w : ' r w α s and corresponding budgets B p α q w : ' 2 p 1 ` 2 σ p ∆ p α q w ln p 2 K ∆ p α q w qq 1 { 2 q . The rationale is the following. The algorithm suffers a high regret in windows during which the optimal arm changes. Indeed, in windows where no change happens, the algorithm is likely to commit to the best arm, suffering no regret after the initial exploration phase. Conversely, in windows where the optimal arm changes, the algorithm could commit to an arm that then becomes suboptimal, or it could fail in estimating the optimal arm. In this case, the regret increases with the distance of the expected rewards of ˆ i ˚ and the actual optimal arm in round t : i ˚ t P arg max i P J K K µ i p t q . Thanks to the concavity, the maximum increment max i P J K K γ i p t q decreases as t increases. Thus, as time passes, if the optimal arm changes, it takes longer for the expected rewards of ˆ i ˚ and i ˚ t to diverge significantly. Hence, we can restart the algorithm with a lower frequency, which is equivalent to having windows with increasing width.

Regret Analysis. RC-BE p α q partitions the set of rounds N ě 1 in windows J s p α q w , e p α q w K with s p α q w : ' ř w ´ 1 l ' 1 ∆ p α q l ` 1 and e p α q w : ' ř w l ' 1 ∆ p α q l , for w P N ě 1 . Let w p α q p t q ' min t w P N ě 1 s.t. e p α q w ě t u be the index of the window that contains the round t P N ě 1 . Thus, the learning horizon T is split in w p α q p T q windows. In what follows, we bound the regret suffered by RC-BE p α q on the set of windows W which enjoy certain properties that we introduce later. To this end, we denote the regret suffered by a policy π on a set of windows W Ă N ě 1 , | W | ă 8 as:

<!-- formula-not-decoded -->

Now, we present the properties which induce the classes of windows of interest for the analysis. In particular, we need to formally characterize the fact that, in a window, the optimal arm can change. To this end, we introduce the following definitions, in analogy to what is done in (Jia et al., 2024).

Definition 4.1 (Overtaking) . An arm i P J K K overtakes an arm j P J K K at time t P N ě 2 if µ i p t ´ 1 q ď µ j p t ´ 1 q and µ i p t q ě µ j p t q . Formally, we write i Ò t j (note that i Ò t i ).

Definition 4.2 (Crossing) . Two arms i, j P J K K cross at time t P N ě 2 , if i Ò t j or j Ò t i . Formally, we write i ˆ t j (note that i ˆ t i ).

We introduce a binary relation for arms that cross in the w -th window. For w P N ě 1 , i, j P J K K :

Let w ˆ ` be the transitive closure of w ˆ . w ˆ ` is an equivalence relation since w ˆ is reflexive and symmetric. For an arm i P J K K , we denote with r i s w ˆ ` its equivalence class w.r.t. w ˆ ` . Let:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

be the set of optimal arms in window w . Furthermore, we define I ˆ w : ' r i ˚ w s w ˆ ` for some i ˚ w P I ˚ w . Observe that the definition is well posed since, in virtue of Lemma C.2, it does not depend on the choice of i ˚ w . For w P N ě 1 , i P J K K , we define the diameter of its equivalence class w.r.t. w ˆ ` as

<!-- formula-not-decoded -->

We use the shorthand d ˚ w for d w p i ˚ w q where i ˚ w P I ˚ w . The following lemma decomposes the regret suffered by RC-BE p α q during the w -th window as the sum of the regret due to the exploration phase plus the regret due to the commitment phase.

Lemma 4.1. For all restless rising concave MABs ν , α ě 1 , w P N ě 1 we have that:

<!-- formula-not-decoded -->

Thus, the regret due to exploration is proportional to the budget B p α q w , while the regret suffered during the commitment phase depends on the width of the window ∆ p α q w and on the diameter d ˚ w of I ˆ w . In windows where the optimal arm does not change, I ˆ w is a singleton and, thus, its diameter is 0 . This reflects the fact that, in such windows, the algorithm suffers only the regret due to the exploration.

We now provide an upper bound for d w p i q with w P N ě 1 , i P K which exploits concavity.

<!-- formula-not-decoded -->

J K Lemma 4.2. For all restless rising concave MABs ν , α ě 1 , w P N ě 1 , i P J K K , we have that:

Recall that Υ ν p 1 , e p α q w q is upper bounded by K . Thus, as expected, eventually the upper bound of the diameter decreases as w increases. This reflects what we informally stated before. As time goes, due to the concavity, it takes more time for the expected rewards of arms which have crossed to diverge significantly. Thus, it makes sense to increase the width of the windows over time.

We now discriminate between two kinds of windows: those in which the expected rewards of arms which cross (and thus of the arms which belong to I ˆ w ) do not diverge significantly and those in which, instead, the converse happens. More formally, let d P p 0 , K s :

<!-- formula-not-decoded -->

In the second class of windows, we have no upper bound to the diameter d ˚ w other than that of Lemma 4.2, which considers a worst-case scenario in which the divergence of the expected rewards of the arms which cross is the maximum possible. We now show that this scenario, in the rising concave setting, can happen only a limited number of times. In particular, this is translated into an upper bound to the number of windows in W ą d p T q , which is captured by the following lemma.

Lemma 4.3. For all restless rising concave MABs ν , α ě 1 , T P N ě 1 , d P p 0 , K s , we have that:

<!-- formula-not-decoded -->

Informally, this lemma states that, in the rising concave setting, it cannot happen in too many windows that the expected rewards of arms which cross diverge significantly (i.e., more than d ).

We use this fact to conclude the analysis. In particular, observe that we can always upper bound the regret suffered on a set of windows W as R ν p π, W q ď | W | max w P W R ν p π, t w uq . We use this to upper bound the regret suffered on both W ď d p T q and W ą d p T q . In the first case, we observe that | W ď d p T q| ď w p α q p T q and use the definition of W ď d p T q together with Lemma 4.1 to bound max w P W ď d p T q R ν p RC-BE p α q , t w uq . In the second case, we use Lemma 4.3 to upper bound | W ą d p T q| and Lemma 4.1 together with Lemma 4.2 to deal with max w P W ą d p T q R ν p RC-BE p α q , t w uq . These observations lead to the following result which is formally proven in Appendix C.

Theorem 4.4 (Upper Bound for the Rising Concave Setting) . For all restless rising concave MABs ν , α ě 1 , T P N ě 24 , we have that:

<!-- formula-not-decoded -->

In particular, for α 1 ' 8 { 3 , we get:

<!-- formula-not-decoded -->

Furthermore, for

we get:

<!-- formula-not-decoded -->

By looking at the algorithm and at Theorem 4.4, we observe how by selecting α ' 8 { 3 , we achieve a regret of order r O p T 7 { 11 q without the knowledge of either the total variation V T or the learning horizon T , making it an anytime algorithm, at the price of a worse dependence on K and V T . This result shows that the regret minimization problem in rising concave MABs is indeed easier w.r.t. general restless MABs with bounded variation (Besbes et al., 2014) and rising MABs. Indeed, the regret r O p T 7 { 11 q in our upper bound is smaller than that of the lower bound for restless MABs with bounded variation (Besbes et al., 2014, Theorem 1) and rising MABs (Theorem 3.2), i.e., Ω p T 2 { 3 q .

## 5 Numerical Simulations

In this section, we present the results of numerical simulation of RC-BE p α q compared to state-of-theart algorithms for restless, restless rising concave, and stationary MABs. 9

Baselines. We consider the baseline algorithms: Rexp3 (Besbes et al., 2014), an algorithm for restless MABs based on a variation budget; R-less-UCB (Metelli et al., 2022), an algorithm for restless rising concave MABs; and UCB1 (Auer et al., 2002a; Bubeck, 2010), one of the most effective

9 Additional simulations are reported in Appendix E. The code to reproduce the results is available at https://github.com/m1gwings/rcbealpha-experiments .

<!-- formula-not-decoded -->

under the additional assumptions ν P E σ c p T, V T q ,

<!-- formula-not-decoded -->

,

Figure 3: Instance and results of the experimental validation.

<!-- image -->

algorithms for stationary MABs. The choices of the parameters of the algorithms are reported in Appendix E.

Setting. The algorithms are evaluated for T ' 5 ¨ 10 6 rounds on synthetic instances with K ' 5 arms. The stochasticity is realized by adding Gaussian noise with standard deviation σ ' 0 . 1 . The curves of the expected rewards have the functional form f p t q ' c p 1 ´ exp p´ sat { T qq for t P J T K where a, c P p 0 , 1 s , s ' 50 , and are reported in Figure 3a. We compare the algorithms in terms of empirical cumulative regret p R ν p π, t q which is the empirical counterpart of the expected cumulative regret R ν p π, t q at round t averaged over multiple independent runs. In each simulation, the parameter α of RC-BE p α q is set to α ' 8 { 3 , as suggested by Theorem 4.4.

Results. The empirical cumulative regret suffered by the algorithms is shown in Figure 3b. We observe that RC-BE p α q is the algorithm that achieves the lowest regret at the horizon. UCB1 has the lowest regret in the first rounds, afterwards its regret starts increasing when the optimal arm changes. This is consistent with the fact that we are violating the stationarity assumption on which the algorithm relies. Rexp3 is an algorithm which restarts at a fixed frequency. In particular, the number of restarts has order T 1 { 3 . Thus, in this simulation, there are « 10 2 restarts, and, by looking at the figure, it is not possible to appreciate the behavior of the algorithm between one restart and the next. For this reason, Rexp3 shows a cumulative regret which increases linearly. This is consistent with the fact that the algorithm is not anytime. R-less-UCB , consistently with its theoretical guarantees, shows a sublinear growth of the cumulative regret. Its estimator relies on a rested model of the evolution of the expected rewards of the arms, penalizing the empirical performance.

## 6 Discussion and Conclusions

In this paper, we studied the restless rising and rising concave MABs, where the expected rewards of the arms are non-decreasing and non-decreasing concave in the number of played rounds, respectively. We derived lower bounds to the expected cumulative regret in both settings. The lower bound in the rising setting has order Ω p T 2 { 3 q and implies that the non-decreasing expected reward assumption does not simplify the learning problem w.r.t. the general restless setting with bounded variation, and so that all the algorithms which are optimal for the general setting are optimal also in this special subclass, closing in this way the gap present in the literature. Thus, for the rising setting, we provided a positive answer to our Question 1 and a negative answer to our Question 2 . The lower bound in the rising concave setting has order Ω p T 3 { 5 q and implies that rising concave MABs represent a statistically harder problem w.r.t. stationary MABs. After having presented two statistical barriers for these settings, we developed a learning algorithm with the goal of exploiting the more structured model of rising concave MABs. To this end, we designed RC-BE p α q , and we derived an upper bound to its expected regret of order r O p T 7 { 11 q . This result implies that the non-decreasing expected reward assumption, together with the concave expected reward assumption, simplifies the learning problem w.r.t. the same setting without concavity. Thus, for the rising concave setting, we provided a positive answer for both Question 1 and Question 2 . The natural future research direction includes closing the gap in rising concave MABs which is now only 7 { 11 ´ 3 { 5 ' 2 { 55 in the exponent of T .

## Acknowledgments

Funded by the European Union - Next Generation EU within the project NRPP M4C2, Investment 1.3 DD. 341 - 15 March 2022 - FAIR - Future Artificial Intelligence Research - Spoke 4 - PE00000013 - D53C22002380006.

## References

Tor Lattimore and Csaba Szepesvári. Bandit Algorithms . Cambridge University Press, 2020.

- Qingyun Wu, Naveen Iyer, and Hongning Wang. Learning contextual bandits in a non-stationary environment. In International Conference on Research &amp; Development in Information Retrieval (SIGIR) , pages 495-504. ACM, 2018.
- Maryam Aziz, Emilie Kaufmann, and Marie-Karelle Riviere. On multi-armed bandit designs for dose-finding trials. Journal of Machine Learning Research , 22(14):1-38, 2021.
- Cem Tekin and Mingyan Liu. Online learning of rested and restless bandits. IEEE Transactions on Information Theory , 58(8):5588-5611, 2012.
- Aurélien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit problems. In Algorithmic Learning Theory (ALT) , pages 174-188. Springer Berlin Heidelberg, 2011.
- Francesco Trovò, Stefano Paladino, Marcello Restelli, and Nicola Gatti. Sliding-window thompson sampling for non-stationary settings. Journal of Artificial Intelligence Research , 68:311-364, 2020.
- Omar Besbes, Yonatan Gur, and Assaf Zeevi. Stochastic multi-armed-bandit problem with nonstationary rewards. In Advances in Neural Information Processing Systems (NIPS) , page 199-207. MIT Press, 2014.
- Hoda Heidari, Michael Kearns, and Aaron Roth. Tight policy regret bounds for improving and decaying bandits. In International Joint Conference on Artificial Intelligence (IJCAI) , 2016.
- Alberto M. Metelli, Francesco Trovò, Matteo Pirola, and Marcello Restelli. Stochastic rising bandits. In Proceedings of the International Conference on Machine Learning (ICML) , volume 162 of Proceedings of Machine Learning Research , pages 15421-15457. PMLR, 2022.
- Nir Levine, Koby Crammer, and Shie Mannor. Rotting bandits. In Advances in Neural Information Processing Systems (NIPS) , pages 3074-3083, 2017.
- Julien Seznec, Andrea Locatelli, Alexandra Carpentier, Alessandro Lazaric, and Michal Valko. Rotting bandits are no harder than stochastic ones. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , volume 89 of Proceedings of Machine Learning Research , pages 2564-2572. PMLR, 2019.
- Julien Seznec, Pierre Ménard, Alessandro Lazaric, and Michal Valko. A single algorithm for both restless and rested rotting bandits. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , volume 108 of Proceedings of Machine Learning Research , pages 3784-3794. PMLR, 2020.
- Marco Mussi, Alessandro Montenegro, Francesco Trovò, Marcello Restelli, and Alberto M. Metelli. Best arm identification for stochastic rising bandits. In Proceedings of the International Conference on Machine Learning (ICML) , volume 235 of Proceedings of Machine Learning Research , pages 36953-36989. PMLR, 2024.
- Su Jia, Qian Xie, Nathan Kallus, and Peter I. Frazier. Smooth non-stationary bandits. In Proceedings of the International Conference on Machine Learning (ICML) , volume 202 of Proceedings of Machine Learning Research , pages 14930-14944. PMLR, 2023.
- Su Jia, Qian Xie, Nathan Kallus, and Peter I. Frazier. Smooth non-stationary bandits. CoRR , abs/2301.12366, 2024.

- Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning , 47:235-256, 2002a.
- Sébastien Bubeck. Bandits games and clustering foundations . PhD thesis, Université des Sciences et Technologie de Lille-Lille I, 2010.
- Siwei Wang, Longbo Huang, and John C. S. Lui. Restless-ucb, an efficient and low-complexity algorithm for online restless bandits. In Advances in Neural Information Processing Systems (NeurIPS) , volume 33, pages 11878-11889. Curran Associates, Inc., 2020.
- Peter Auer, Pratik Gajane, and Ronald Ortner. Adaptively tracking the best bandit arm with an unknown number of distribution changes. In Proceedings of the Annual Conference on Learning Theory (COLT) , volume 99 of Proceedings of Machine Learning Research , pages 138-158. PMLR, 2019.
- Fang Liu, Joohyun Lee, and Ness Shroff. A change-detection based framework for piecewisestationary multi-armed bandit problem. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- Lilian Besson, Emilie Kaufmann, Odalric-Ambrym Maillard, and Julien Seznec. Efficient changepoint detection for tackling piecewise-stationary bandits. Journal of Machine Learning Research , 23(77):1-40, 2022.
- Yang Cao, Zheng Wen, Branislav Kveton, and Yao Xie. Nearly optimal adaptive procedure with change detection for piecewise-stationary bandit. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , pages 418-427. PMLR, 2019.
- Lilian Besson and Emilie Kaufmann. What doubling tricks can and can't do for multi-armed bandits. CoRR , abs/1803.06971, 2018.
- Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing , 32(1):48-77, 2002b.
- Yifang Chen, Chung-Wei Lee, Haipeng Luo, and Chen-Yu Wei. A new algorithm for non-stationary contextual bandits: Efficient, optimal and parameter-free. In Proceedings of the Annual Conference on Learning Theory (COLT) , volume 99 of Proceedings of Machine Learning Research , pages 696-726. PMLR, 2019.
- Yang Li, Jiawei Jiang, Jinyang Gao, Yingxia Shao, Ce Zhang, and Bin Cui. Efficient automatic cash via rising bandits. In AAAI Conference on Artificial Intelligence , pages 4763-4771, 2020.
- Leonardo Cella, Massimiliano Pontil, and Claudio Gentile. Best model identification: A rested bandit formulation. In Proceedings of the International Conference on Machine Learning (ICML) , volume 139 of Proceedings of Machine Learning Research , pages 1362-1372. PMLR, 2021.
- Sho Takemori, Yuhei Umeda, and Aditya Gopalan. Model-based best arm identification for decreasing bandits. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , volume 238 of Proceedings of Machine Learning Research , pages 1567-1575. PMLR, 2024.
- Gianmarco Genalti, Marco Mussi, Nicola Gatti, Marcello Restelli, Matteo Castiglioni, and Alberto M. Metelli. Graph-triggered rising bandits. In Proceedings of the International Conference on Machine Learning (ICML) , volume 235 of Proceedings of Machine Learning Research , pages 15351-15380. PMLR, 2024a.
- Gianmarco Genalti, Marco Mussi, Nicola Gatti, Marcello Restelli, Matteo Castiglioni, and Alberto M. Metelli. Bridging rested and restless bandits with graph-triggering: Rising and rotting. CoRR , abs/2409.05980, 2024b.
- Achim Klenke. Probability Theory: A Comprehensive Course . Universitext. Springer, 3 edition, 2020.
- Thomas M. Cover and Joy A. Thomas. Elements of Information Theory (2nd Edition) . Wiley, 2006.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: -

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limits of the paper and the future research directions in order to address them.

## Guidelines:

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

Justification: All the statements are provided with proofs in the appendix.

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

Justification: The compared algorithms and the structure of all the simulated instances are listed in Section 5 and Appendix E. Additional details needed to reproduce the results, like all the seeds of pseudo-random generators used in the simulations, can be found in the code which is made available in the supplemental material.

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

Justification: The code is made available in the supplemental material. All the simulations are on synthetic instances, thus no external data is needed.

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

Justification: The parameters of all the compared algorithms are explicitly listed in Appendix E.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report 95% confidence intervals on the cumulative regret curves used to compare the algorithms.

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

Justification: The information is provided in Section E.2.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper is coherent with NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: -

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

Justification: -

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: -

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

Justification: -

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Related Works

Restless Bandits. In the original restless MAB setting, introduced by Tekin and Liu (2012), the evolution of the expected reward of each arm was described by a Markov chain. Several algorithms have been proposed to deal with this new framework, e.g., Restless-UCB (Wang et al., 2020), which relies on the optimistic estimation of the transition kernel of the underlying chain. Over time, the term restless acquired a broader meaning, encompassing all bandits in which the expected reward changes as time passes. Such arbitrary evolution can be described by a function that maps each round to the expected reward of a given arm. This is the type of restless bandit we target in this work. There are two families of methods to tackle restless MABs: passive (e.g., Garivier and Moulines, 2011; Besbes et al., 2014; Auer et al., 2019; Trovò et al., 2020) and active (e.g., Liu et al., 2018; Besson et al., 2022; Cao et al., 2019). Passive methods base their estimates on the recent feedback, forgetting obsolete observations. Active methods try to detect the changes in arms' expected rewards and use only the observations gathered after the last change. Among the most common passive approaches we find methods based on discounted rewards, e.g., D-UCB (Garivier and Moulines, 2011), or adaptive sliding window, e.g., SW-UCB (Garivier and Moulines, 2011). Both algorithms suffer a r O p T 1 { 2 q regret in the setting in which expected rewards change abruptly a fixed number of times over the time horizon, and such number is known. Auer et al. (2019) obtained a similar result in the same setting, without knowing the number of changes, by resorting on the doubling trick (Besson and Kaufmann, 2018). Another common setting is the one that allows the expected rewards to evolve arbitrarily, with the only constraint that the maximum absolute difference between the expected rewards of an arm in one round and the next, summed over the time horizon, is smaller than or equal to a variation budget V T (Besbes et al., 2014). The Rexp3 algorithm (Besbes et al., 2014), a modification of the Exp3 (Auer et al., 2002b) policy, originally designed for adversarial MABs, shows a regret bound of O p T 2 { 3 q under the knowledge of the variation budget V T . The need for the knowledge of such quantity has been removed by Chen et al. (2019) by means of the doubling trick. In (Trovò et al., 2020), an approach which combines a Thompson-Sampling-like algorithm with a sliding window, shows theoretical guarantees in both the abruptly and smoothly changing settings.

Rising Bandits. Rising concave MABs have been introduced in the deterministic setting by Heidari et al. (2016) and Li et al. (2020), where the rewards observed by the agent in each round are not affected by noise. In their formulation of the problem, the rewards of an arm are non-decreasing in the number of times such an arm has been pulled and satisfy the decreasing marginal return assumption, i.e., the increment in the reward observed between one pull and the next of the same arm is non-increasing in the number of pulls. The online algorithm designed by Heidari et al. (2016) to minimize the regret relies on an optimistic estimate of the cumulative reward that can be obtained by pulling a given arm. Indeed, in this setting, Heidari et al. (2016) show that the optimal policy consists of repeatedly pulling the arm with the highest cumulative reward over the horizon. Li et al. (2020) use the rising concave MAB framework to model the problem of parameter optimization in machine learning and design an algorithm based on iterative elimination of unpromising arms that has good empirical performance. Cella et al. (2021) consider a setting in which the reward is increasing in expectation and the observations are affected by noise. However, in their framework, the expected rewards are constrained to follow a specific parametric form known to the agent. The authors analyze the setting under both the regret minimization and best arm identification frameworks. Anyway, the given parametric form makes this setting not applicable to an arbitrary expected reward evolution that satisfies the non-decreasing assumption. Recently, a surge of approaches has been designed for addressing other learning problems in stochastic rising concave MABs, including regret minimization (e.g., Metelli et al., 2022) and best arm identification (e.g., Takemori et al., 2024; Mussi et al., 2024). Finally, Genalti et al. (2024a,b) proposes a novel framework that interpolates between rested and restless MABs, still assuming the rising concave condition.

## B Lower Bounds

In this appendix, we provide the proofs of the results presented in Section 3 in the main paper.

## B.1 General Recipe for the Lower Bound

The goal of this section is to prove Lemma 3.1. Remember that we work with rewards which follow the distribution ψ p µ, σ q defined in Section 3.1. The result is obtained through techniques from the

adversarial literature in which the instance is also affected by randomness. Thus, we define two probability distributions over J 0 , K K N ě 1 , which induce probability distributions over the instances in E σ µ , r µ . In particular, let ξ, r ξ P ∆ p J 0 , K K q defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for o P J 0 , K K . We can extend ξ and r ξ to probability distributions over J 0 , K K N ě 1 via infinite product (see Example 1.63 of Klenke 2020):

<!-- formula-not-decoded -->

r τ models a random instance in which, in each window, we choose independently and uniformly one arm whose expected reward follows the modified trend, while the expected rewards of all the other arms follow the base trend. τ w instead models a random instance which behaves like r τ up to window w P N ě 1 (excluded); from window w onward all arms follow the base trend. For technical reasons which will be clear in what follows, we need to build a probability space in which the randomness over the instance and the randomness over the rewards are unlinked. Observe that with the current construction this is not the case. Indeed, X is sampled from ν σ o , but o is also a random element. To this end, let s ' p s i,t q i P J K K ,t P N ě 1 ' λ : ' b i P J K K ,t P N ě 1 Unif p 0 , 1 q where Unif p 0 , 1 q is the uniform distribution with support r 0 , 1 s . Then, we can redefine:

<!-- formula-not-decoded -->

where µ o ,i p t q is defined in analogy to Equation (2). In this way, we moved the dependency from o inside the definition of the random variables, preserving their distributions. Indeed, once o is fixed, we have X i,t ' ψ p µ o ,i p t q , σ q . For consistency with the notation, we introduce the random variables O ' p O w q w P N ě 1 where O w p o q ' o w . The probability distributions that we just defined, induce probability density functions over finite reward sequences taking into account the randomness both in the instance and in the rewards. In particular, let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for w P N ě 1 , i P J K K , r 1 , . . . , r T P t´ 1 { 2 σ, 3 { 2 σ u . We use p w and r p w,i to denote also all the conditional and marginal distributions; disambiguation happens through the arguments, e.g., p w p r s w | r 1 , . . . , r s w ´ 1 q .

To obtain the result, we use the following tools from information theory (Cover and Thomas, 2006).

Definition B.1 ( L 1 Distance of Two Discrete Probability Density Functions) . Let p, q be two discrete probability density functions defined over the finite set X , we define their L 1 distance as:

<!-- formula-not-decoded -->

Definition B.2 (Kullback-Leibler Divergence of Two Discrete Probability Density Functions) . Let p, q be two discrete probability density functions defined over the finite set X , we define their Kullback-Leibler divergence as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ν , ξ are probability distributions with discrete support and p ν , p ξ are their corresponding discrete probability density functions.

By extension, we define:

We now state and prove a generalization of Lemma A.1 in (Auer et al., 2002b) which we then use to derive Lemma 3.1.

<!-- formula-not-decoded -->

Proof. To simplify the notation, let t 1 : ' s w , t 2 : ' min t e w , T u . The lhs of Equation (4) can be written as:

<!-- formula-not-decoded -->

where line (5) can be found in (Chapter 11, Cover and Thomas, 2006). Again, from (Lemma 11.6.1 Cover and Thomas, 2006), we have that:

<!-- formula-not-decoded -->

From the chain rule of entropy:

<!-- formula-not-decoded -->

Because of how τ w and r τ are defined, we have that:

<!-- formula-not-decoded -->

and thus term (b) is 0 because of the properties of D KL p¨}¨q . To deal with term (a) we need to work on the expressions of r p w,i p r t | r 1 , . . . , r t ´ 1 q and p w p r t | r 1 , . . . , r t ´ 1 q for t P J t 1 , t 2 K . First of all observe that the arm that the agent pulls at round t is fully determined by the past sequence of observed rewards r 1 , . . . , r t ´ 1 since the policy π is deterministic. As remarked in Section 2, we denote it through π p t q , omitting the dependence on r 1 , . . . , r t ´ 1 . Now: 10

<!-- formula-not-decoded -->

10 With slight abuse of notation, we will use the symbol ψ p x | µ, σ q to denote the p.d.f. associated to the distribution ψ p µ, σ q .

where line (6) follows from the fact that, under the event O w ' i , X π p t q ,t is independent from X π p 1 q , 1 , . . . , X π p t ´ 1 q ,t ´ 1 and follows distribution ψ p r µ w p t ´ s w ` 1 q , σ q if π p t q ' i , ψ p µ w p t ´ s w ` 1 q , σ q otherwise. Thus, we conclude:

<!-- formula-not-decoded -->

From analogous calculations, it is possible to derive:

<!-- formula-not-decoded -->

Thanks to the last results and the definition of D KL p¨}¨q :

<!-- formula-not-decoded -->

The lemma follows by chaining the results.

We are ready to prove Lemma 3.1.

Lemma 3.1 (General Lower Bound) . Under the assumption that µ w p t q ď r µ w p t q for all w P N ě 1 , t P J ∆ w K , for any deterministic policy π , subgaussian parameter σ ě 1 , and learning horizon T P N ě 1 , it holds that:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

for all w P J w p T q K , with D KL p¨}¨q being the Kullback-Leibler divergence of the two distributions (formally defined in Appendix B).

Proof. For o P J 0 , K K N ě 1 , t P J T K , let i ˚ o ,t P arg max i P J K K µ o ,i p t q . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under the assumption r µ w p t q ě µ w p t q for all w P N ě 1 , t P J ∆ w K , we have:

<!-- formula-not-decoded -->

Then, observing that O w ' 0 has probability 0 under r τ :

<!-- formula-not-decoded -->

where line (7) follows from Lemma B.1 with f corresponding to the function from the observed rewards to the arm I t pulled in round t , which is well defined for deterministic policies, and line (8) follows from Cauchy-Schwarz inequality applied to a vector of K ones and the vector of the terms under square root. The result follows from the definitions of D µ , r µ ,T,σ w and A µ , r µ ,T w .

## B.2 Specializing the Lower Bound for the Rising Setting

The goal of this section is to prove Theorem 3.2.

Theorem 3.2 (Lower Bound for the Rising Setting) . For any deterministic policy π , subgaussian parameter σ ě 1 , and learning horizon T P N ě 1 , T ě σ 2 K min t 1 , V T u ´ 2 , it holds that:

<!-- formula-not-decoded -->

Proof. First of all, we need to formally define the sequences of window widths, base, and modified trends. Let ∆ r ,w ' ∆ r : ' X σ 2 { 3 T 2 { 3 K 1 { 3 min t 1 , V T u ´ 2 { 3 \ and:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all w P N ě 1 where ε r : ' 1 4 min t 1 , V T u{ w p T q ą 0 . Observe that r µ r ,w p ∆ r q ď µ r ,w ` 1 p 1 q for all w P N ě 1 , hence, for any choice of o P J 0 , K K N ě 1 , ν σ r , o satisfies Assumption 2.1. Furthermore, for all o P J 0 , K K N ě 1 , the expected rewards of the arms change at most between one window and the next, i.e., w p T q ´ 1 times in the learning horizon, and the magnitude of the increment is at most 2 ε r , thus:

<!-- formula-not-decoded -->

Hence E σ µ r , r µ r Ď E σ r p T, V T q indeed holds. Finally, it is easy to verify that 0 ď µ r ,w p t q ď r µ r ,w p t q ď 1 for all w P N ě 1 , t P J ∆ r K , so that the assumptions of Lemma 3.1 are satisfied. From Lemma D.1, we have that:

<!-- formula-not-decoded -->

The choice of ∆ r ,w ' ∆ r implies ε r ď 1 4 σ 2 { 3 T ´ 1 { 3 K 1 { 3 min t 1 , V T u 1 { 3 once we observe that w p T q ' r T { ∆ r s . Then:

<!-- formula-not-decoded -->

Thus, observing that K ě 2 , we have

<!-- formula-not-decoded -->

Since A µ r , r µ r ,T w ' ε r p min t e w , T u´ s w ` 1 q , by plugging the previous results in Lemma 3.1, assuming that T ě σ 2 K min t 1 , V T u ´ 2 which guarantees T ě ∆ r , we have:

<!-- formula-not-decoded -->

where the last step follows from the definition of ε r and the fact that t x u ě x { 2 and r x s ď 2 x for x ě 1 .

## B.3 Specializing the Lower Bound for the Rising Concave Setting

The goal of this section is to prove Theorem 3.3.

Theorem 3.3 (Lower Bound for the Rising Concave Setting) . For any deterministic policy π , subgaussian parameter σ ě 1 , and learning horizon T P N ě 1 , T ě 2 10 σ 2 K min t 1 , V T u ´ 2 , it holds that:

<!-- formula-not-decoded -->

Proof. First of all, we need to formally define the sequences of window widths, base, and modified trends. Let N c : ' P σ ´ 2 { 5 T 1 { 5 K ´ 1 { 5 min t 1 , V T u 2 { 5 T , ∆ c ,w ' ∆ c : ' r T { N c s for all w P N ě 1 . Observe that ∆ c is defined in such a way that w p T q ' r T { ∆ c s ď r T {p T { N c q s ' N c . Furthermore, being σ, K ě 1 , we have:

<!-- formula-not-decoded -->

so that T { N c ě 1 and ∆ c ď 2 T { N c since r x s ď 2 x for x ě 1 . Let m 0 : ' 1 4 min t 1 , V T u{ T P p 0 , 1 q , m w : ' p 2 N c ´ w q m 0 {p 2 N c q for w P J 2 N c K . p m w q 2 N c w ' 0 are the slopes of the segments which

constitute the trends. Observe that m 0 ą m 1 ą ¨ ¨ ¨ ą m 2 N c ´ 1 ą m 2 N c ' 0 . We are ready to define the trends:

<!-- formula-not-decoded -->

for all w P N ě 1 . In what follows, with a slight abuse of notation, we will regard µ c ,w and r µ c ,w as defined on r 0 , ∆ c s . Observe that, as we informally stated before, µ c ,w p 0 q ' r µ c ,w p 0 q , and µ c ,w p ∆ c q ' r µ c ,w p ∆ c q ' µ c ,w ` 1 p 0 q for all w P N ě 1 . Furthermore, it is easy to check that the slope of the second segment of the modified trend in a window is equal to the slope of the first segment of the modified trend in the next window. Thus, because of what we remarked when we informally introduced the construction, for any choice of o P J 0 , K K N ě 1 , ν σ c , o satisfies Assumptions 2.1 and 2.2. Furthermore, in each window with index w P J w p T q K , the maximum increment of the expected reward of an arm, corresponds to the slope of the first half of the modified trend m 2 w ´ 2 . Thus:

<!-- formula-not-decoded -->

Hence E σ µ c , r µ c Ď E σ c p T, V T q indeed holds. Finally, by calculations analogous to what we did above to bound the cumulative increment, one can verify that:

<!-- formula-not-decoded -->

which, together with the previous remarks, implies 0 ď µ c ,w p t q ď r µ c ,w p t q ď 1 for all w P N ě 1 , t P J ∆ c K , so that the assumptions of Lemma 3.1 are satisfied. The maximum distance between the two trends in a window is attained for t ' ∆ c 2 and has value:

<!-- formula-not-decoded -->

Hence, in virtue of Lemma D.1, we have:

<!-- formula-not-decoded -->

for all w P J w p T q K . Thus, remembering that K ě 2 , we get:

<!-- formula-not-decoded -->

Now, let's lower bound the expression of A µ c , r µ c ,T w for w P J w p T q ´ 1 K :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where line (9) follows from the fact that t x u ě x { 2 for x ě 1 together with ∆ c { 2 ě 1 being T ą N c when T ě 4 (which is guaranteed by the constraint on T ), and that x ď x 2 for x ě 1 . Finally, T ě 2 10 σ 2 K min t 1 , V T u ´ 2 guarantees w p T q ´ 1 ě N c { 4 , and thus, by Lemma 3.1 in conjunction with the results we just proved, we have:

<!-- formula-not-decoded -->

where line (10) follows from our choice of N c and from the fact that r x s ď 2 x for x ě 1 .

## C Upper Bound for the Rising Concave Setting

In this appendix, we provide the proofs of the results presented in Section 4 in the main paper.

## C.1 Additional notation

We begin by introducing the additional notation required for the analysis. Let:

<!-- formula-not-decoded -->

be respectively the cumulative reward and cumulative expected reward by RC-BE p α q for arm i P J K K in the first d P J 0 , ∆ p α q w K rounds of window w P N ě 1 . Let N w be the number of round-robin cycles of window w P N ě 1 , where we also count the degenerate cycles in which we pull the only remaining alive arm ˆ i ˚ . Let t w,l be the round in which the l -th round-robin cycle (with l P J N w K ) is started during window w P N ě 1 . Analogously, let N i,w be the number of times arm i P J K K is pulled in the w -th window (with w P N ě 1 ) and t i,w,l the round in which arm i is pulled for the l -th time (with l P J N i,w K ) during window w . For simplicity in the notation, we define d w,l ' t w,l ´ s p α q w ` 1 and d i,w,l ' t i,w,l ´ s p α q w ` 1 . Finally, we define the good events:

<!-- formula-not-decoded -->

for i P J K K , w P N ě 1 , d P J ∆ p α q w K , δ P p 0 , 1 s , and

<!-- formula-not-decoded -->

for i P J K K , δ P p 0 , 1 s .

## C.2 Concentration

We start the analysis with a concentration result for ˆ S i,w,d .

Lemma C.1 (Concentration) . For every w P N ě 1 , δ P p 0 , 1 s , we have that:

<!-- formula-not-decoded -->

Proof. For i P J K K , d P J 0 , ∆ p α q w K , λ P R , let:

<!-- formula-not-decoded -->

Let t 1 : ' s p α q w ` d ´ 1 to ease the notation. Observe that I t 1 is F w,d ´ 1 -measurable and that X i,t 1 is independent from F w,d ´ 1 . Furthermore, we can rewrite ˆ S i,w,d as

<!-- formula-not-decoded -->

Then:

<!-- formula-not-decoded -->

where in the last line we use the properties of conditional expectation (Klenke, 2020) and the sub-gaussianity of X i,t 1 . Thus, by induction:

<!-- formula-not-decoded -->

Then, thanks to Markov inequality, for every ε P R :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

An analogous bound holds for

Then, thanks to a union bound,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.3 Proof of Lemma 4.1

The goal of this section is to prove Lemma 4.1. To this end, we need several intermediate results. We start by proving that I ˆ w is indeed well-defined.

<!-- formula-not-decoded -->

Finally:

<!-- formula-not-decoded -->

Proof. If µ i ˚ w p t 1 q ' µ j ˚ w p t 1 q for some t 1 P J s p α q w , e p α q w K , then it must be i ˚ w w ˆ j ˚ w . Thus, assume µ i ˚ w p t 1 q ă µ j ˚ w p t 1 q for some t 1 P J s p α q w , e p α q w K . If i ˚ w and j ˚ w do not cross, then

<!-- formula-not-decoded -->

which is a contradiction with the fact that i ˚ w P I ˚ w .

We now prove a very useful property of w ˆ ` .

Lemma C.3. Let i, j, k P J K K . If i w ˆ ` j and there exists t 1 P J s p α q w , e p α q w K such that µ i p t 1 q ď µ k p t 1 q ď µ j p t 1 q , then k P r i s w ˆ ` .

Proof. If µ k p t 1 q ' µ i p t 1 q or µ k p t 1 q ' µ j p t 1 q then the statement is trivial. Consider µ i p t 1 q ă µ k p t 1 q ă µ j p t 1 q . We proceed by contradiction. Assume that it is not true that k w ˆ ` i . Let I 1 ' ␣ l P r i s w ˆ ` s.t. µ l p t 1 q ă µ k p t 1 q ( and I 2 ' ␣ l P r i s w ˆ ` s.t. µ l p t 1 q ą µ k p t 1 q ( . Since I 1 Y I 2 Ď r i s w ˆ ` , I 1 X I 2 ' tu , I 1 , I 2 ‰ tu there must be i 1 P I 1 , i 2 P I 2 such that i 1 w ˆ i 2 . But, since it is not true that k w ˆ ` i , it cannot be k w ˆ i 1 nor k w ˆ i 2 . Thus it must be

<!-- formula-not-decoded -->

But this is absurd since i 1 w ˆ i 2 , concluding the proof.

This leads to the following corollary.

Corollary C.4. Let i P I ˆ w , j R I ˆ w , then:

<!-- formula-not-decoded -->

Proof. By contrapositive, if µ j p t 1 q ě µ i p t 1 q for some t 1 P J s p α q w , e p α q w K , then there exists k P arg max l P J K K µ l p t 1 q such that µ i p t 1 q ď µ j p t 1 q ď µ k p t 1 q and thus j P I ˆ w by Lemma C.3.

We are ready to prove Lemma 4.1.

Lemma 4.1. For all restless rising concave MABs ν , α ě 1 , w P N ě 1 we have that:

<!-- formula-not-decoded -->

Proof. We start by proving that, under event G w, p 2 K ∆ p α q w q ´ 1 , at least one arm in I ˆ w is always alive in each round-robin cycle. We need to consider all the eliminations which happen at the end of a round-robin cycle, except for the last, in which eliminations are irrelevant (remember that the window ends at the end of the last round-robin cycle and the algorithm is restarted). To this end, let n P J N w ´ 1 K . For an arm i P J K K , to eliminate an arm j P J K K at the end of the n -th round-robin cycle, it must be:

<!-- formula-not-decoded -->

which, under event G w, p 2 K ∆ p α q w q ´ 1 , implies

<!-- formula-not-decoded -->

if and only if

<!-- formula-not-decoded -->

which implies, being the instance rising:

<!-- formula-not-decoded -->

and thus, because of the choice of B p α q w , it must be:

<!-- formula-not-decoded -->

Thus, in virtue of Corollary C.4, it cannot be i R I ˆ w , j P I ˆ w . But, to eliminate all alive arms in I ˆ w , we would need at least one cycle in which an elimination of the kind above happens. Hence there will always be at least an arm in I ˆ w alive. Let i ˆ w,n be such arm during the n -th round-robin cycle. Let's bound the regret of a generic arm j P J K K during the w -th window, under event G w, p 2 K ∆ p α q w q ´ 1 .

<!-- formula-not-decoded -->

where the last line follows from the fact that we have not eliminated arm j at the end of the p N j,w ´ 1 q -th round robin cycle. Thus, the regret during the w -th window, under event G w, p 2 K ∆ p α q w q ´ 1 , is upper bounded as:

<!-- formula-not-decoded -->

Finally, in virtue of Lemma C.1:

<!-- formula-not-decoded -->

## C.4 Proof of Lemma 4.2

The goal of this section is to prove Lemma 4.2. To this end, we need several intermediate results. We start with a lower bound to e p α q w .

Lemma C.5. For any α ě 1 , w P N ě 1 it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. If w ' 1 , we trivially have

Now, suppose w ě 2 , then

<!-- formula-not-decoded -->

Now we introduce the results through which we exploit the concavity of the instance.

Lemma C.6. For any restless rising concave MAB ν , t 1 , t 2 P N ě 1 , t 2 ě t 1 ě 2 , we have:

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Before proving Lemma 4.2, we need an intermediate upper bound to d w p i q .

Lemma C.7. For all restless rising concave MABs ν , α ě 1 , w P N ě 1 , i P J K K , we have that:

<!-- formula-not-decoded -->

Proof. If j w ˆ ` k , there must exist distinct i 1 , . . . , i n different from j and k ( n P J 0 , |r i s w ˆ ` | ´ 2 K ) such that j w ˆ i 1 , i 1 w ˆ i 2 , . . . , i n ´ 1 w ˆ i n , i n w ˆ k . Then, for t P J s p α q w , e p α q w K , we have:

<!-- formula-not-decoded -->

We are ready to prove Lemma 4.2.

Lemma 4.2. For all restless rising concave MABs ν , α ě 1 , w P N ě 1 , i P J K K , we have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Analogously, if t ă t 1 , we have

<!-- formula-not-decoded -->

We conclude that, if j w ˆ k , then | µ j p t q ´ µ k p t q| ď Υ ν ´ s p α q w , e p α q w ¯ for all t P J s p α q w , e p α q w K . Thus, in virtue of Lemma C.7, if j w ˆ ` k , then:

<!-- formula-not-decoded -->

For w ě 2 , by applying iteratively Lemma C.6, we have

<!-- formula-not-decoded -->

where in the last line we used Lemma C.5, the fact that r x s ď 2 x for x ě 1 , and the definition of ∆ p α q w . The same upper bound holds trivially for w ' 1 since s p α q 1 ' e p α q 1 ' 1 .

## C.5 Proof of Lemma 4.3

The goal of this section is to prove Lemma 4.3. To get the result, we start by providing an upper bound to the number of times an arm i overtakes arm j and the expected rewards diverge by a quantity greater than G ą 0 . To this end, we need to prove two auxiliary results.

Lemma C.8. Let t Ò , ˆ t, t Ó P N ě 1 , t Ó ą ˆ t ě t Ò , G P p 0 , 1 s , i, j P J K K such that

Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We start by proving Equation (11). Suppose γ j p ˆ t q ě γ i p t Ò ´ 1 q . Then:

<!-- formula-not-decoded -->

which is a contradiction with the definition of ˆ t . Thus it must be γ j p ˆ t q ă γ i p t Ò ´ 1 q . Analogously, suppose γ j p ˆ t q ă γ i p t Ó q . Then:

<!-- formula-not-decoded -->

which is a contradiction with the definition of t Ó . Thus it must be γ j p ˆ t q ě γ i p t Ó q . We now prove Equation (12):

<!-- formula-not-decoded -->

and thus

<!-- formula-not-decoded -->

Finally, we prove Equation (13):

<!-- formula-not-decoded -->

Lemma C.9. Let M P N ě 1 , M ě 2 , m 1 ą m 2 ą ¨ ¨ ¨ ą m M ą m M ` 1 ą 0 , then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We regard m 1 ą m M ` 1 ą 0 as fixed constants and study the functions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

defined for m 1 ą m 2 ą ¨ ¨ ¨ ą m M ą m M ` 1 . Observe that the functions are defined on an open set and their values tend to infinity when the input tends to the border of the domain. We show that they have only one stationary point, which then must be a minimum point. We start by proving Equation (14). Let k P J 2 , M K :

if and only if

<!-- formula-not-decoded -->

The linear system above is equivalent to:

<!-- formula-not-decoded -->

Thus m M ` 1 ' Mm 2 ´p M ´ 1 q m 1 , and then

<!-- formula-not-decoded -->

By plugging this result into Equation (16), we get the coordinates of the minimum point:

<!-- formula-not-decoded -->

Thus:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now prove Equation (15) analogously:

<!-- formula-not-decoded -->

if and only if if and only if

and then

Finally:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that we get the same linear system of the previous case, with the difference that the variables are now ln m i . Thus, the solution is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma C.10. Let G P p 0 , 1 s , T 1 P N ě 1 , M P N ě 1 , i, j P J K K such that there exist rounds which satisfy

Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Observe that, since

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

and thus

<!-- formula-not-decoded -->

Now, assume M ě 3 . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

J K

<!-- formula-not-decoded -->

where line (17) follows from Lemma C.8, line (18) follows from the fact that x a ´ x is non-decreasing for a ě 0 and the concavity, line (19) follows from Lemma C.9, and line (20) follows from the fact that γ i p t Ò 1 ´ 1 q ď 1 and γ i p t Ò M ´ 1 q ě G T 1 . Now, if M ě 1 ` ln p T 1 { G q , by Lemma D.2, we have exp ´ ln p T 1 { G q M ´ 1 ¯ ´ 1 ď 3 ln p T 1 { G q M ´ 1 , and thus, by the chain of inequalities above:

<!-- formula-not-decoded -->

Thus, by considering all possible cases, we have:

<!-- formula-not-decoded -->

We are now ready to prove Lemma 4.3.

Lemma 4.3. For all restless rising concave MABs ν , α ě 1 , T P N ě 1 , d P p 0 , K s , we have that:

<!-- formula-not-decoded -->

Proof. Let w P W ą d p T q . Then there exists i P J K K such that d w p i q ą d . But, in virtue of Lemma C.7, we have:

<!-- formula-not-decoded -->

Thus, there must be j, k P r i s w ˆ ` and t P J s p α q w , e p α q w K such that j w ˆ k and

<!-- formula-not-decoded -->

Observe that it must be either i ˆ t 1 j for t 1 ď t or i ˆ t 1 j for t 1 ą t , with t 1 P J s p α q w ` 1 , e p α q w K . W.l.o.g. we assume that i overtakes j . In the first case, window w must contain one of the rounds in which i overtakes j and then their expected rewards diverge by at least d { K . In the second case, window w must contain either the first round in which i overtakes j and which is right after one of the rounds in which i overtakes j and their expected rewards diverge by at least d { K or the first time in which i overtakes j . In virtue of Lemma C.10 with G ' d K and T 1 ' e p α q w p α q p T q , the rounds described in the first case are in number no more than 4 ln ´ 3 e p α q w p α q p T q K { d ¯ p d { K q ´ 1 { 2 , while the rounds described in the second case are in number no more than 4 ln ´ 3 e p α q w p α q p T q K { d ¯ p d { K q ´ 1 { 2 ` 1 for a fixed choice of i, j P K . Since we have at most K 2 such choices, it must be:

<!-- formula-not-decoded -->

## C.6 Proof of Theorem 4.4

The goal of this section is to prove Theorem 4.4. We start with an upper bound to w p α q p T q , e p α q w p α q p T q , and Υ ν ´ 1 , e p α q w p α q p T q ¯ .

Lemma C.11. For all restless rising concave MABs ν , α ě 1 , T P N ě 2 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Westart by proving Equation (21). If w P N ě 1 , w ě p 2 p 1 ` α q T q 1 {p 1 ` α q , then, by Lemma C.5, we have:

<!-- formula-not-decoded -->

Thus it must be w p α q p T q ď p 2 p 1 ` α q T q 1 {p 1 ` α q . We now use Equation (21) to prove Equation (22).

<!-- formula-not-decoded -->

where in line (24) we use the definition of ∆ p α q w , Equation (21), and the fact that r x s ď 2 x for x ě 1 . Finally, we prove Equation (23).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where line (25) follows by applying iteratively Lemma C.6 and line (26) follows from the fact that T ě 2 and by Equation (22).

We are ready to prove Theorem 4.4.

Theorem 4.4 (Upper Bound for the Rising Concave Setting) . For all restless rising concave MABs ν , α ě 1 , T P N ě 24 , we have that:

<!-- formula-not-decoded -->

In particular, for α 1 ' 8 { 3 , we get:

<!-- formula-not-decoded -->

Furthermore, for we get:

<!-- formula-not-decoded -->

under the additional assumptions ν P E σ c p T, V T q ,

<!-- formula-not-decoded -->

,

<!-- formula-not-decoded -->

Proof. Let d 1 : ' KT ´p α { 2 q{p 1 ` α q P p 0 , K s . Then:

<!-- formula-not-decoded -->

where line (27) follows from Lemma 4.1, line (28) follows from Lemma 4.3 and Lemma 4.2, line (29) follows from Lemma C.11, the definition of ∆ p α q w , the fact that r x s ď 2 x for x ě 1 , and the definition of B p α q w , line (30) follows from the fact that the expression inside max is increasing in w , Lemma C.11, and the fact that r x s ď 2 x for x ě 1 , and line (31) follows from T ě 24 . Furthermore:

<!-- formula-not-decoded -->

where line (32) follows from Lemma 4.1, line (33) follows from the definitions of B p α q w and W ď d 1 p T q , and line (34) follows from Lemma C.11, T ě 24 , r x s ď 2 x for x ě 1 , and the definition of d 1 . By summing the previous results:

<!-- formula-not-decoded -->

Finally, observe that, under the additional assumption ν P E σ c p T, V T q , we have Υ ν p 1 , T q ď V T , and the additional constraint on T guarantees α 2 ě 1 .

## D Technical Lemmas

Lemma D.1. Let µ 1 , µ 2 P r 0 , 1 s with µ 1 ď µ 2 and σ ě 1 . Then:

<!-- formula-not-decoded -->

where D KL p¨}¨q is the Kullback-Leibler divergence defined in Appendix B, and ψ p µ, σ q is the distribution defined in Section 3.1.

Proof. Let p p µ, σ q : ' 1 4 ` µ 2 σ . Consider the function:

<!-- formula-not-decoded -->

for x P r 0 , µ 2 ´ µ 1 s . Then:

<!-- formula-not-decoded -->

By direct evaluation, we have f p 0 q ' f 1 p 0 q ' 0 . Furthermore, since µ P r 0 , 1 s , σ ě 1 , imply p p µ, σ q P r 1 { 4 , 3 { 4 s , then:

<!-- formula-not-decoded -->

Finally:

<!-- formula-not-decoded -->

The result follows from the fact that D KL p ψ p µ 1 , σ q} ψ p µ 2 , σ qq ' f p µ 2 ´ µ 1 q .

Lemma D.2.

<!-- formula-not-decoded -->

Proof. Let f p x q ' e x ´ 1 . Then: f 1 p x q ' e x ' f 2 p x q . Thus, by Taylor's theorem, if x P r 0 , 1 s , there exists ξ P p 0 , 1 q such that

<!-- formula-not-decoded -->

## E Numerical Simulations

In this appendix, we present additional numerical simulations which compare RC-BE p α q with the baseline algorithms reported in Section 5. Furthermore, we provide information regarding the compute resources used to run the simulations.

Baselines. We consider the following baseline algorithms:

- Rexp3 (Besbes et al., 2014), an algorithm for restless MABs based on a variation budget for the expected rewards of the arms over the learning horizon.
- R-less-UCB (Metelli et al., 2022), an algorithm for restless rising concave MABs which relies on the optimism principle and exploits the structure of the setting through a specifically crafted estimator.

Figure 4: Piecewise linear instance.

<!-- image -->

- UCB1 (Auer et al., 2002a; Bubeck, 2010), one of the most effective algorithms for stationary MABs.
- The choices of the parameters of the algorithms that we compared are the following:
- Rexp3 : V T ' K since, as remarked in Section 2, in the rising setting the cumulative increment is always smaller than or equal to K ; ∆ T ' r p K ln p K qq 1 { 3 p T { V T q 2 { 3 s ; γ ' min ! 1 , a K ln p K q{p ∆ T p e ´ 1 qq ) as recommended in (Besbes et al., 2014).
- R-less-UCB : h i,t ' t ϵN i,t ´ 1 u where N i,t ´ 1 is the number of times arm i has been pulled by the agent in the first t ´ 1 rounds, with ϵ P p 0 , 1 { 2 q ; α ą 2 as prescribed in (Metelli et al., 2022). In particular, we choose ϵ ' 0 . 25 ; α ' 2 . 1 .
- UCB1 : the upper confidence bound interval for arm i at round t is σ a 4 ln p t q{ N i,t ´ 1 .

## E.1 Additional Instances

Piecewise Linear Instance. The piecewise linear curves that describe the evolution of the expected rewards in the simulation have the following functional form:

<!-- formula-not-decoded -->

for t P J T K where µ i , µ e P r 0 , 1 s , µ i ď µ e . After the flattening time t flat P J T K , the expected rewards of the arms stop increasing. The expected reward curves of the simulated instance are reported in Figure 4a. The algorithms are evaluated on T ' 5 ¨ 10 6 rounds. The standard deviation of the noise is σ ' 0 . 1 . The empirical cumulative regret suffered by the algorithms is shown in Figure 4b. We can observe that RC-BE p α q is the algorithm that achieves the lowest regret at the horizon. The behavior of all other algorithms is explained by the same observations stated for the exponential instance in Section 5. Conversely to what happens in the exponential instance, in this case, UCB1 shows a better performance than R-less-UCB . This is due to the fact that the change of the optimal arm happens later in time and the distance between the expected rewards of the first and last optimal arms is less w.r.t. the exponential instance presented in Section 5.

Constant Instance. In this simulation, the expected rewards of the arms do not change with time (i.e., stationary MABs). The expected reward curves of the simulated instance are reported in Figure 5a. The algorithms are evaluated on T ' 10 6 rounds. The standard deviation of the noise is σ ' 0 . 01 . The empirical cumulative regret suffered by the algorithms is shown in Figure 5b. UCB1 is the algorithm that achieves the lowest regret. This is consistent with the fact that the instance is stationary. RC-BE p α q has the second-best performance. The reduction of the standard deviation of the noise leads to smaller confidence bounds and, thus, a better performance, for R-less-UCB . Conversely, Rexp3 is not able to exploit this fact, being based on the Exp3 algorithm which is designed for the adversarial setting.

Figure 5: Constant instance.

<!-- image -->

## E.2 Compute Resources

The simulations were run on a single CPU core with a clock frequency of 2 . 60 GHz . The system has a 8 . 0 GiB RAM. For each algorithm, we report the approximate time required to simulate a single run on the exponential instance with 5 ¨ 10 6 rounds:

- Rexp3 : 5 min 50s ;

- R-less-UCB : 8 min ;

- UCB1 : 3 min 30s ;

- RC-BE p α q : 1 min 50s .

## F Flaw in the Original Analysis of K -armed Budgeted Exploration

<!-- image -->

(a) Reward functions.

(b) Derivatives of the reward functions.

Figure 6: Example instance.

In this appendix, we highlight a flaw in the original analysis of the extension of Budgeted Exploration in the K -armed setting, which is presented in the unpublished preprint (Jia et al., 2024). For notation and definitions, refer to the original paper. The analysis relies on the following proposition, stated in Lemma I.7: "First, we observe that on the clean event C , any arm in A ˚ can never be eliminated for "losing" to an arm in p A ˚ q c ". It is possible to construct a counterexample which satisfies the hypotheses of the lemma and violates the previous proposition. We now show

how. We work with 3 arms. We describe the evolution of the expected reward of the arms only in a certain window. This is sufficient for the construction of the counterexample since the lemma regards the behavior of the algorithm in a single window. The window is composed of 17 W rounds, with W P N ě 2 to be chosen later. The expected rewards of the arms are defined as follows:

<!-- formula-not-decoded -->

where f a : r 0 , 1 s Ñ r´ 1 , 1 s is a 2 -Hölder function with Lipschitz constant L ą 0 for a P J 3 K . Such functions and their derivatives are depicted in Figure 6a and Figure 6b, respectively. More specifically, we choose:

- The function in which the expected rewards of the first arm are embedded as:

<!-- formula-not-decoded -->

- The function in which the expected rewards of the second arm are embedded as:

<!-- formula-not-decoded -->

- The function in which the expected rewards of the third arm are embedded as:

<!-- formula-not-decoded -->

The definitions rely on the constants d, ε ą 0 , ε ă d ď 1 { 2 , which we choose later. To guarantee that the functions are well-defined, we impose:

<!-- formula-not-decoded -->

We work with deterministic rewards, which can be regarded as a special realization under the clean event C . Let Z total ,t a be the cumulative reward of arm a P J K K observed up to round t P J 17 W K , included. Assuming there is no elimination before round 3 W (we choose d and ε in such a way that this is true), we have that:

<!-- formula-not-decoded -->

Then:

Let:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where B is the budget of the algorithm. These choices are such that we eliminate arm 3 at the end of round 3 W (and not before), losing to arm 1 . Arm 2 , instead, stays alive. To satisfy d ď 1 { 2 , it is sufficient to require W ě 3 B . After round 3 W , the algorithm pulls only arms 1 and 2 . When r 1 p t q ě r 2 p t q , their difference is at most ε . Thus:

<!-- formula-not-decoded -->

Hence, arm 2 is not eliminated before round 5 W (included). By the choice of the instance, in virtue of Equation (35), after round 5 W , we have r 1 p t q ' 1 { 2 ´ d . Thus, after each round robin cycle, which takes 2 rounds, Z total ,t 2 ´ Z total ,t 1 increases by d ´ ε . Then:

<!-- formula-not-decoded -->

This means that, at some point after round 5 W , arm 1 will be eliminated, losing to arm 2 . But it is evident that 1 P A ˚ and 2 P p A ˚ q c . However, it is important to notice that 2 P I ˆ w , consistent with our analysis. It remains to show that there are choices of B , W , T , and L which satisfy the hypotheses of the lemma and the additional requirements we imposed. In particular, they need to satisfy:

<!-- formula-not-decoded -->

.

It is clear that such an assignment exists. Furthermore, we can find such an assignment even when we restrict the budget to the natural choice which has order W 1 { 2 .
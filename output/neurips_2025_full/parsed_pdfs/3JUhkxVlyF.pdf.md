## Mechanism Design for LLM Fine-tuning with Multiple Reward Models

Haoran Sun 1 , Yurong Chen 2 ∗ , Siwei Wang 3 , Xu Chu 1 , Wei Chen 3 , Xiaotie Deng 1 ∗ 1 CFCS, School of Computer Science, Peking University 2 Inria, École Normale Supérieure, PSL Research University 3 Microsoft Research Asia sunhaoran0301@stu.pku.edu.cn, yurong.chen@inria.fr , {chu\_xu, xiaotie}@pku.edu.cn, {siweiwang, weic}@microsoft.com ,

## Abstract

Fine-tuning large language models (LLMs) to aggregate multiple preferences has attracted considerable research attention. With aggregation algorithms advancing, a potential economic scenario arises where fine-tuning services are provided to agents with different preferences. In this context, agents may benefit from strategically misreporting their preferences, but this could harm the aggregation performance. This paper addresses such incentive issues by framing it as a mechanism design problem: an LLM provider determines the fine-tuning objective (training rule) and the pricing scheme (payment rule) for agents. We primarily focus on training rules that maximize social welfare subject to certain regularizations, referred to as SW-Max rules. First, we show that under most circumstances, truthful reporting is sub-optimal with simply a SW-Max rule, thereby highlighting the necessity of payments. Second, we extend the VCG payment to implement SW-Max rules in dominant-strategy incentive compatibility (DSIC). We characterize sufficient conditions for payment equivalence and derive the necessary conditions for a payment rule to implement a SW-Max rule in DSIC and other principles. Third, we demonstrate that our mechanism is approximately DSIC with perturbed input, showcasing its robustness against the inevitable errors in real-world applications. Experiments on real LLM training results further confirm the practical implications of our results.

## 1 Introduction

As large language models (LLMs) [61, 74] become increasingly widespread, users are seeking models that not only possess general capabilities but also align with their individual values. Reinforcement Learning from Human Feedback (RLHF) [14, 57] has emerged as a mainstream approach to achieve this alignment, where a reward model guides the reinforcement learning process using feedback signals that reflect human preferences.

However, standard RLHF becomes resource-intensive when catering to diverse preferences. Training separate LLMs for every individual or group within a community, each with unique preferences, is often impractical due to prohibitive computational costs and potential data privacy concerns. A more feasible alternative is to train a unified model that reflects collective values while still accommodating distinct needs. Multiple-Objective RLHF (MORLHF) [5, 78], which aims to efficiently integrate multiple preferences into a single model, offers a promising avenue for this. Further studies aim to improve MORLHF algorithms from various perspectives, including efficiency [41, 62, 70], accuracy [18, 26, 63, 82], and fairness [11].

∗ Corresponding Authors.

As these techniques advance, we explore a practical economic scenario: a platform offering a fine-tuning service to aggregate diverse preferences from various groups into a single LLM. These 'groups'-such as different departments within a company or hospitals in the same city with various specializations-share the same core values but have slightly different focuses. Given these shared values and the high cost of fine-tuning, developing separate LLMs for each entity is often inefficient. Nevertheless, each group must provide its specific preferences to account for these differing focuses. Finally, the training cost is shared among the groups and can be non-uniform due to their differentiated preferences.

A critical issue in this process is that groups may strategically misreport their preferences to manipulate the aggregate objective for a more favorable outcome. As illustrated in a simplified RLHF framework (see Figure 1), a group's true preference (rm 1 ) could be misreported as a polarized one ( ˜ rm 1 ) to steer the model toward a more desirable outcome. However, this behavior distorts the training objective, resulting in a suboptimal model for the overall community. Given the potential profitability of such strategies and the growing economic importance of LLMs, ensuring truthful preference reporting is as critical as the training algorithm itself. We therefore formalize this scenario to study its incentives. Our findings indicate that many commonly used training objectives lead to profitable misreporting strategies. However, we also demonstrate that a simple incentive-compatible cost allocation scheme can incentivize truthful reporting, and under certain conditions, this scheme is uniquely determined.

Specifically, we model this as a multi-parameter mechanism design problem involving a fine-tuning service provider and multiple groups of agents. The mechanism consists of a training rule , which aggregates the reported sizes w i (representing a group's scale) and preferences from different groups, and a payment rule to determine their respective charges. The fine-tuning process is implemented through RLHF, with reward models representing the groups' preferences. Our focus is on training objectives aimed at maximizing social welfare with a regularization constraint, referred to as SW-Max training rules. Our technical contributions, which extend beyond standard mechanism design due to the unique complexities of LLM fine-tuning objectives, are summarized as follows:

1. We show that mechanisms using only SW-Max training rules are vulnerable to profitable preference misreporting (Theorem 4.2 and Theorem 4.3) . This finding highlights the need for a payment rule to resolve incentive issues.
2. We extend the VCG payment to ensure truthfulness for SW-Max training rules (Proposition 4.4) and further establish the uniqueness of this payment under certain conditions (Theorem 4.9 and Corollary 4.10). Based on that, we derive necessary conditions for payment rules to implement a SW-Max training rule in more principles (Theorem 4.11) .
3. We demonstrate that our mechanism is approximately DSIC in the presence of input perturbations (Theorem 4.12). This finding highlights the robustness of our mechanism against the inevitable measurement errors in real-world applications.
4. Experiments on practical LLM setups empirically validate the existence of profitable misreporting strategies and demonstrate the efficacy of our mechanism in incentivizing truthful reporting (Section 5).

Related Work. Several recent studies have also examined incentive issues in RLHF and LLMs. Duetting et al. [25] proposed a preference aggregation mechanism that satisfies monotonicity with respect to bids; however, their work does not address strategic misreporting of preferences, which is the central challenge we tackle. Other works that consider strategic preference reporting have different focuses, such as implementing truthful rules with KL-divergence for ad auctions [71], analyzing the implementability of various training rules [59], or modifying the RLHF objective to achieve approximate truthfulness while preserving convergence [10]. In contrast, our work adopts a theoretical perspective to analyze representative training rules, providing a comprehensive understanding of incentive issues in RLHF. Specifically, our analysis of payment equivalence helps characterize all possible payment rules that implement a training rule in DSIC.

Our research also connects to classic literature on auction design [52-54] and facility location problems [21, 58]. Compared to the classic auction model, we have to consider the necessary regularization term, which makes the training rule (or the allocation rule in the auction) more complicated and prevents vanilla VCG from being applied. In facility locations, agents can benefit by misreporting a more polarized preference. The idea of such a strategy is similar to our model.

<!-- image -->

̸

Figure 1: An illustration of the incentive issue in LLM preference aggregation. When using a basic training rule ψ in RLHF for two groups (Alices and Bobs), fixing Bobs' report ˜ rm 2 , Alices can gain a higher utility by strategically reporting ˜ rm ′ 1 = rm 1 than truthfully reporting ˜ rm 1 = rm 1 . On the other hand, we have ASW( θ ; - → rm , ⃗ w,θ init ) &gt; ASW( θ ′ ; - → rm , ⃗ w,θ init ) , which means that such strategic behavior also harms the training objective.

However, due to the complexity of the training rules that aim to catch the LLM fine-tuning scenarios and the normalization constraints of the reward models, the reporting strategies can be more complex. Further, combined with the discretized input spaces of the agents, most of our results cannot be directly derived from existing literature.

Paper Organization. The remainder of the paper is organized as follows. Section 2 introduces the necessary preliminaries, and Section 3 formulates the RLHF Game. We then analyze the properties of mechanisms composed of SW-Max training rules and payment rules in Section 4, followed by a presentation of our experimental results in Section 5. Finally, Section 6 offers concluding remarks and discusses potential future research directions.

## 2 Preliminaries

Large Language Models. In this paper, LLMs are abstracted as stochastic mappings from a prompt set, denoted by X , to a probability distribution over sequences of length up to K in the output space [25]. Let T represent the set of all tokens, and define T ∗ := ∅ ∪ T ∪ T 2 ∪ . . . ∪ T K as the set of sequences with lengths up to K . An LLM parameterized by θ is a function LLM θ : X → ∆( T ∗ ) . The space of LLM parameters is denoted by Θ , and it is assumed that the LLM can express any function within this space. Our theoretical model operates on each prompt independently, so we focus on a fixed prompt scenario and omit its notation for simplicity. We denote LLM θ ( x ) the probability of a sequence x generated by the model LLM θ .

Reward Modeling. In RLHF, a reward model is a function rm : X × T ∗ → R , which maps a prompt-response pair to a real number, indicating humans' satisfaction with the response based on the prompt. Similar to the LLM case, we focus on a fixed prompt scenario, so rm ( x ) represents the scalar feedback for a response x ∈ T ∗ . Following prior empirical work for RLHF [57, 78], we mainly consider two types of normalization constraints for the reward model: (1) The summation of the rewards over T ∗ is normalized to 1 , i.e. ∑ x ∈ T ∗ rm ( x ) = 1 . (2) The maximum of the rewards over T ∗ is normalized to 1 , i.e. max x ∈ T ∗ rm ( x ) = 1 . Furthermore, we also assume that the output rewards are all non-negative, i.e., rm ( x ) ≥ 0 for all x ∈ T ∗ . The set of all reward model functions satisfying these conditions is denoted by R . Unless otherwise specified, the results in this paper hold under both normalization schemes.

## 3 Formulation of the RLHF Game

In this section, we present the formal description of the RLHF Game. The game involves one LLM provider and n groups of agents , denoted by [ n ] = { 1 , 2 , . . . , n } . The provider has an initial model

LLM θ init with positive probability for all sequences, i.e., LLM θ init ( x ) &gt; 0 for all x ∈ T ∗ . Each group i has w i agents who share the same preference represented by a reward model rm i . Let R and W ⊆ N + denote the domains for each group's reward model and group size, respectively. The group size w should be an integer, and we assume an upper bound ¯ w for W , which is public information. The exact reward model rm i and the size w i are group i 's private information. For an agent in group i , the valuation when it receives a model LLM θ is denoted by v i ( θ ; rm i ) , defined as follows.

Definition 3.1. An agent's valuation of model LLM θ is its expected reward on the sequences generated by it: v ( θ ; rm ) = E x ∼ LLM θ rm ( x ) = ∑ x ∈ T ∗ LLM θ ( x ) rm ( x ) .

In practice, this can be obtained by averaging the reward of the sequences sampled from an LLM. We also discuss the influence of possible errors in this process in Section 4.3.

Remark on the group size ⃗ w . We introduce the concept of group size to ensure that our model encompasses a broader range of scenarios. As the scales of different groups may vary, our training objective has to account for this factor to ensure fairness. Groups are also allowed to over-report their sizes to attain a higher status in fine-tuning. The case ⃗ w = 1 represents a special scenario where each group consists of exactly one agent and is included in our general model. In certain results, we note that the general model is technically more difficult than the ⃗ w = 1 case.

The provider first announces the mechanism, including a training rule ψ and a payment rule p ,

<!-- formula-not-decoded -->

Both rules take n reported reward models, n reported sizes, and an initial model as input and output the objective fine-tuned model and each group's payment, respectively. The provider can choose not to charge the users by setting p always equal to 0 . In this case, the model coincides with most previous work on designing empirical algorithms, where agents' incentives are not considered [18, 26, 41, 63, 76, 78, 82]. Specifically, the training rule seeks the model that maximizes a certain objective function OBJ. That is, ψ ( - → rm , ⃗ w,θ init ) ∈ arg max θ ∈ Θ OBJ ( θ ; - → rm , ⃗ w,θ init ) , with ties broken based on further ordering of v i ( θ ; rm i ) s.

After observing the announced mechanism ( ψ , p ), each group i reports a reward model, ˜ rm i , and its group size ˜ w i . Based on the reported information, the provider fine-tunes the model and gets the final model with parameter θ final = ψ ( - → ˜ rm , ⃗ ˜ w,θ init ) . Each member in the group has access to the fine-tuned model, so the valuation for group i is w i v i ( θ final ; rm i ) . The provider then charges each group i a one-time payment according to the payment rule, p i ( - → ˜ rm , ⃗ ˜ w,θ init ) . All groups have quasi-linear utilities, i.e., group i 's utility is the valuation it attains minus the payment:

<!-- formula-not-decoded -->

The groups may strategically report, thus - → ˜ rm and ⃗ ˜ w do not necessarily equal the true - → rm and ⃗ w . The LLM provider's goal is to achieve its training objective based on the group's true preferences, taking into account that misreporting may distort the training outcome. To this end, it is crucial to incentivize all groups to report their information truthfully so that the provider has access to the groups' private information. These desiderata for the mechanism are formally defined as follows.

Definition 3.2. A mechanism ( ψ, p ) satisfies dominant-strategy incentive compatibility (DSIC) if ∀ i , rm i , w i , rm ′ i , w ′ i , - → rm -i , ⃗ w -i , θ init, we have

<!-- formula-not-decoded -->

Definition 3.3. A mechanism ( ψ, p ) satisfies individually rationality (IR) if ∀ i , rm i , w i , - → rm -i , ⃗ w -i , θ init, we have

<!-- formula-not-decoded -->

DSIC means that truthfully reporting the reward model and the group size yields the highest utility for any group, regardless of other groups' reports. IR means that truthfulness always yields non-negative utilities. When a mechanism ( ψ, p ) satisfies DSIC, IR, or both DSIC and IR, we say that the payment rule p implements ψ in DSIC, IR, or both DSIC and IR. When we say the implementability of a training rule, we refer to the property of DSIC.

## 4 Incentives in the RLHF Game

This section explores incentive design within the RLHF Game framework. Our focus is mainly on a set of training rules that aims at maximizing social welfare with regularization, which balances efficiency and fairness and is commonly used in practice to aggregate various preferences [8, 56]. Denote D f ( p || q ) := E q ( x ) f ( p ( x ) /q ( x )) the divergence between probability distributions p and q measured by function f , the formal definition follows.

Definition 4.1 (SW-Max Training Rules) . A Social Welfare-Maximizing training rule fine-tunes the model to maximize the summation of the groups' valuations subject to a regularization measured by f -divergence [3, 19, 70]. Formally, the training objective is

<!-- formula-not-decoded -->

where f is convex on R + and f (1) = 0 . We use ASW( θ ; - → rm , ⃗ w,θ init ) to denote the affine social welfare.

This defines a set of training rules, and the function f includes the most commonly used regularization terms in training a model. For example, f ( x ) = λx log x refers to KL-divergence, f ( x ) = λ ( x -1) 2 refers to χ 2 divergence, f ( x ) = λ | x -1 | refers to total variation. We denote ψ ∈ Ψ SW that ψ belongs to this set.

In the following subsections, we will first establish the necessity of a payment rule for SW-Max training rules. Then, we construct DSIC mechanisms for these training rules using affine maximizer payments and demonstrate payment equivalence properties for certain distance measures f . Next, we address the influence of noise input on the DSIC property. Finally, we discuss the efficient implementations of the mechanisms in practice.

## 4.1 Necessity of Payment Rule

We start by showing that without payment rules, groups have incentives to misreport their preferences under most circumstances. Our discussion focuses on strategies other than simply inflating the group size w i . We assume that for ∀ - → rm , ⃗ w,θ init, the fine-tuned model θ = ψ ( - → rm , ⃗ w,θ init ) satisfies that LLM θ ( x ) &gt; 0 for ∀ x ∈ T ∗ . This mainly excludes extreme cases where the outcomes remain largely unchanged regardless of input, which may make the analysis meaningless. Based on this, we comprehensively analyze the relationship between optimal strategy and truthful reporting. We start with two cases with strong intuition.

Theorem 4.2. In the RLHF Game with mechanism ( ψ, p ) that ψ ∈ Ψ SW and p ≡ 0 , for group i , define s i := |{ r | r = rm i ( x ) , x ∈ T ∗ }| and rm i := min x ∈ T ∗ rm i ( x ) :

1. If s i = 1 , truthfully reporting is the optimal strategy regardless of other groups' reports.
2. If s i ≥ 2 and rm i &gt; 0 , there is a strategy that yields strictly higher utility than truthfully reporting regardless of other groups' reports.

s i = 1 is an unusual case in which group i has the same preference values for all x , resulting in the same valuation for any model θ . In such a case, all strategies bring the same utility and hence are optimal. However, when s i ≥ 2 and rm i &gt; 0 , group i can report rm ′ i that assigns a lower value to x 1 = arg min x ∈ T ∗ rm i ( x ) (and a larger value to x 2 = arg max x ∈ T ∗ rm i ( x ) in summation normalization). By doing so, group i pretends to prefer x 1 less, thereby increasing the likelihood that the resulting fine-tuned model generates the outcomes it prefers more. The condition rm i &gt; 0 ensures that group i is not completely uninterested in any x , which is more realistic in practice.

̸

Further, we consider the case that s i ≥ 2 and rm i = 0 . Since the minimum value is already 0 , the strategy above cannot be applied. We need to analyze in more detail how the training results change when one group adjusts its reported preferences. Under certain smoothness conditions of the function f , we derive a function t ( x ) to estimate the gradient of the valuation for group i over the reported value rm i ( x ) . Based on this function, we show that if t ( x ) = 0 for some x , it is always possible to find a suitable direction and magnitude to report rm ′ i ( x ) = rm i ( x ) , allowing group i to achieve higher utility. The result is summarized in the following theorem. Due to the complicated form of the function t , we provide a detailed version in the Theorem B.2.

̸

̸

Theorem 4.3 (Simplified version of Theorem B.2) . In the RLHF Game with mechanism ( ψ, p ) that ψ ∈ Ψ SW and p ≡ 0 , when f is strongly convex and C 2 -smooth, there exists a function t , when t ( x , - → rm , ⃗ w,θ init ) = 0 for some x ∈ T ∗ , truthfully reporting is not the optimal strategy.

The properties of f stated in Theorem 4.3 are also considered in optimization theory [48] and encompass a wide range of divergence measures. Combining Theorem 4.2 and Theorem 4.3, we provide a comprehensive analysis that covers the entire space of s i and rm i . While the second theorem offers only a sufficient condition for the suboptimality of truthful reporting, we demonstrate in the proof that this condition is highly likely to occur , illustrating the impossibility of a mechanism that aims to maximize social welfare to incentivize truthfulness without payments.

## 4.2 Affine Maximizer Payment

After establishing the necessity of payment rules in this scenario, we mainly address two questions in this part:

1. Given a training rule ψ , can we find a payment rule p such that the mechanism ( ψ, p ) satisfies DSIC? This is the so-called implementability of a training rule ψ .
2. For an implementable training rule ψ , can we identify the relationship between the payment rules p s among all DSIC mechanisms ( ψ, p ) .

For the first question, since there is an additional regularization term, we can not directly apply the vanilla VCG payment [15, 34, 75] to the SW-Max training rules. To address this problem, we define ASW -i ( θ ; - → rm , ⃗ w,θ init ) , the affine social welfare function that excludes the contribution of group i from the social welfare:

<!-- formula-not-decoded -->

Then, the vanilla VCG payment can be generalized to the following form, which is also known as the affine maximizer payment rule [64] p AFF :

<!-- formula-not-decoded -->

Following the proof of the classic VCG mechanism, we show that p AFF implements SW-Max training rules in both DSIC and IR, implying that truthfully reporting both reward models and group sizes constitutes a dominant Nash Equilibrium under this mechanism.

Proposition 4.4. For any ψ ∈ Ψ SW , mechanism ( ψ, p AFF ) satisfies DSIC and IR, and the payment is non-negative.

The availability of the affine maximizer payment derives from the additive property of SW-Max training rules. However, this method does not apply to training rules where the objective function cannot be decomposed into additive components, such as Nash Social Welfare and the fairnessoriented objective defined in MaxMin-RLHF [11]. The implementability of an arbitrary training rule is characterized by the concept of cycle monotonicity, which is discussed in Section E but is not the focus of this paper.

The second question is more general, so we consider the concept of payment equivalence [4] as a bridge, which is defined as:

Definition 4.5 (Payment Equivalence) . An implementable training rule ψ satisfies payment equivalence if for any two mechanisms ( ψ, p ) and ( ψ, p ′ ) satisfying DSIC, there exists a function g i such that for ∀ rm i ∈ R , w i ∈ W

<!-- formula-not-decoded -->

Or equivalently, when fixing - → rm -i , ⃗ w -i and θ init, there is a constant c such that p ′ i ( rm i , w i ) = p i ( rm i , w i ) + c for all rm i ∈ R , w i ∈ W .

Payment equivalence indicates that the only way to modify a mechanism ( ψ, p ) to ( ψ, p ′ ) while maintaining the property of DSIC is to add a term that is independent of i 's report to group i 's payment function p i . Thus, the payment equivalence of ψ is sometimes interpreted as the uniqueness of the payment rule p that implements it in DSIC. This notion is particularly useful in the case that

we can figure out a certain DSIC mechanism ( ψ, p ) for ψ because any other payment rules p ′ that also implement it in DSIC can be divided into p and an independent part.

In the context of the RLHF Game, the domain of the reward models and group sizes affects payment equivalence. When ⃗ w ≡ 1 , groups only report reward models, with the domain R containing all normalized reward models rm. Since this forms a connected set in Euclidean space, we can apply the result from Nisan et al. [55] to show:

Proposition 4.6. When ⃗ w ≡ 1 is public information, and the agents only report the reward models, all implementable training rules satisfy payment equivalence.

However, when the group size ⃗ w is also a part of the private information for all groups, the domain of the whole private information becomes R×W that is no longer a connected set because W ⊆ N + . To get a more meticulous characterization of the property, we define the continuity of a training rule.

Definition 4.7 (Continuous Training Rule) . A training rule ψ is continuous if for any ϵ &gt; 0 , there exists a δ &gt; 0 such that for any θ init , - → rm, - → rm ′ , ⃗ w and ⃗ w ′ , if max x ∈ T ∗ | ∑ n i =1 ( w i rm i ( x ) -w ′ i rm ′ i ( x )) | ≤ δ , then max x ∈ T ∗ | LLM θ ( x ) -LLM θ ′ ( x ) | ≤ ϵ , where θ := ψ ( - → rm , ⃗ w,θ init ) and θ ′ := ψ ( - → rm ′ , ⃗ w ′ , θ init ) .

The continuity requests that the training outcome be similar if the reported values are similar. This definition is natural, and we identify several continuous SW-Max training rules.

Proposition 4.8. SW-Max training rules with regularizations KL-divergence, f KL ( x ) = λx log x , and χ 2 divergence, f 2 ( x ) = λ ( x -1) 2 ( λ &gt; 0 is a constant) are continuous.

Based on the continuity, we show a sufficient condition of payment equivalence for general training rules.

Theorem 4.9. An implementable training rule ψ satisfies payment equivalence if it is continuous and for ∀ i , - → rm -i , ⃗ w -i , θ init there exists rm ∗ i and θ such that ψ (( rm ∗ i , - → rm -i ) , ( w i , ⃗ w -i ) , θ init ) ≡ θ for all w i ∈ W . In the maximum normalization case, rm ∗ i must be 1 .

We provide some intuitions of the theorem. Here, when fixing - → rm -i , ⃗ w -i , and θ init, if we can find a rm ∗ i such that when group i reports rm ∗ i then the reported w i will not affect the training result, rm ∗ i actually serves to connect different w i ∈ W . For SW-Max training rules, we observe that the reward model rm that assigns the same value for all x s, i.e., ∀ x , rm ( x ) = 1 for maximum normalization, and rm ( x ) = 1 / | T ∗ | for summation normalization, serves the role of rm ∗ i . With the continuity of the training rule, this makes the domain of R×W connected in another sense that can also induce payment equivalence. Based on this, we derive the payment equivalence property:

Corollary 4.10. Each continuous training rule ψ ∈ Ψ SW satisfies payment equivalence.

As a continuous SW-Max training rule always satisfies payment equivalence, we can establish the relationship between p AFF and any other payment rule that implements it in DSIC. Combined with the inherent property of p AFF , we derive the necessary conditions for a payment rule to satisfy more conditions, such as non-negativity and IR.

Theorem 4.11. Given a continuous training rule ψ ∈ Ψ SW and a payment rule p implements it in DSIC: If p is always non-negative, it holds that for all i , - → rm, ⃗ w , and θ init ,

<!-- formula-not-decoded -->

If p implements ψ in IR, then for any ϵ &gt; 0 and i , there exists - → rm -i , ⃗ w -i , and θ init, such that for all rm i and w i ,

<!-- formula-not-decoded -->

This result implies that if we want to design a payment p to satisfy all these properties, p AFF is a 'lower bound' for p , and p should be sufficiently close to p AFF in some inputs.

## 4.3 Approximate Valuation

In this part, we study the influence of errors generated in practice on the incentive property in the RLHF Game. We abstract it as an approximate valuation problem [13]. Formally, when group i reports its reward model rm i , the mechanism may not use rm i but rather a noisy reward model ̂ rm i

Figure 2: The simulation result for the mechanism ( ψ, p AFF ) on real LLM setup. We set the group number n = 2 , and the group size ( w 1 , w 2 ) for each column is in the title. We report the valuation, the payment, and the utility for group 1 when adopting different reporting parameters α and β (defined in Section 5). Truthfully reporting ( α = 1 and β = 1 ) brings the highest utility for all cases.

<!-- image -->

as the input. We assume that the noise is independently generated and there is an underlying joint distribution F ( ·| - → rm ) for the - → ̂ rm. This abstraction captures various errors that may occur in practical training. One example is that the calculation of valuation defined in Definition 3.1 requires sampling sequences from LLM, which may result in a deviation from the true valuation.

When the groups are rational, they could be aware of the noise and consider the influence of that on their utility. For group i with reward model rm i and group size w i , it will computes an expected utility U i for reporting ( rm ′ i , w ′ i ) given by

<!-- formula-not-decoded -->

We consider the case that the noisy input reward models ̂ rm i and the reported reward models rm i are close. In that case, we show that when using a training rule ψ ∈ Ψ SW , the distance between the true optimal point and the training outcome with noisy input is bounded. Based on that, we calculate the utility of a group under the mechanism ( ψ, p AFF ) and derive the approximate incentive compatibility of the mechanism.

Theorem 4.12. Assume that for any noisy input - → ̂ rm generated from F ( ·| - → rm ) , and i ∈ [ n ] , there is

<!-- formula-not-decoded -->

Then, with a training rule ψ ∈ Ψ SW , ( ψ, p AFF ) ensures that each group i can benefit at most 2 w i ϵ from misreporting the reward model.

This theoretical result guarantees a considerable utility for truthful reporting. Since the maximum gain of misreporting for group i is less than 2 w i ϵ regardless of the others' reports, groups will tend to truthfully report in cases where finding the optimal strategy and modifying its reward model is costlier than 2 w i ϵ .

## 4.4 Efficient Implementation of the Mechanism

At the end of the whole section, we discuss how p AFF can be implemented in practice, as Proposition 4.4 and Theorem 4.11 show that it is 'unique' to implement SW-Max training rules in DSIC. As is defined in Equation (1), we have to compute ψ ( - → rm -i , ⃗ w -i , θ init ) for each i aside from the final model θ ∗ := ψ ( - → rm , ⃗ w,θ init ) . From the definition ψ ( - → rm -i , ⃗ w -i , θ init ) := max θ ∈ Θ OBJ ( θ ; - → rm -i , ⃗ w -i , θ init ) , finding a maximum over whole space Θ requires a whole training process. This results in n additional

trainings when we have n groups. To address this problem, we propose two heuristic methods that approximate the payment computation; the core of both is to take the maximum on a constraint space Θ ′ instead of the whole space Θ .

## · Heuristic 1: Intermediate Models

During the training to obtain the model ψ ( - → rm , ⃗ w,θ init ) , we usually save intermediate models at different training steps. We can set Θ ′ to be these intermediate models. This requires no additional training and maintains payment non-negativity since θ ∗ is also in Θ ′ , but DSIC is not strictly guaranteed as Θ ′ depends on group i 's report. However, the complex dependency makes strategic manipulation practically difficult.

## · Heuristic 2: Early-Stopped Training

We can perform early-stopped training to compute the ψ ( - → rm -i , ⃗ w -i , θ init ) . This means that we use a less powerful Θ ′ that is only dependent on - → rm -i , ⃗ w -i , θ init. Since the independence of group i , this preserves DSIC theoretically. However, the payment may be negative as ψ ( - → rm , ⃗ w,θ init ) may outperform the early-stopped ψ ( - → rm -i , ⃗ w -i , θ init ) in terms of ASW -i .

These heuristics provide a practical trade-off: Heuristic 1 offers maximum computational efficiency with relaxed theoretical guarantees, while Heuristic 2 preserves DSIC with moderate additional cost. From a theory perspective, we can derive the following result based on Theorem 4.11.

Corollary 4.13. Given a continuous training rule ψ ∈ Ψ SW , if the payment rule p implements it in DSIC, IR and is always non-negative, then for any ϵ &gt; 0 , there exists i , - → rm -i , ⃗ w -i and θ init, such that for all rm i and w i , denote rm = ( rm i , - → rm -i ) and w = ( w i , ⃗ w -i ) , we have

<!-- formula-not-decoded -->

This indicates that any payment rule p that satisfies all these properties must closely approximate p AFF in certain inputs. This somewhat showcases a tradeoff between theoretical guarantees and computational efficiency . A more rigorous analysis of the efficiency loss caused by these heuristics or an 'impossibility theorem' regarding efficient implementation is left for future work.

## 5 Empirical Study

In this section, we present an empirical evaluation of the proposed mechanism. Our objectives are twofold: first, to demonstrate that in practical LLM settings, agents can benefit from misreporting their preferences and distorting the learning outcomes; and second, to intuitively show how our mechanism incentivizes truthful reporting 2 .

Models and Datasets. Our experimental setup follows the literature on Multiple-Objective RLHF [62, 70, 78]. We consider two tasks: the Helpful Assistants task [5] and the Reddit Summary task [72], using Llama-2 7b [74] as the base model for both. For the Helpful Assistants task, the initial model LLM θ init is obtained by supervised fine-tuning a Llama-2 7b model on the Anthropic-HH dataset [5]. We then apply two reward models during the RLHF process to measure harmlessness and humor, respectively. For the Reddit Summary task, the model is fine-tuned on the Summarize-from-Feedback dataset [72], with two reward models assessing the summary's quality and faithfulness.

We formulate these tasks as two mechanism design scenarios: the 'Harmless vs. Humor' game for the Helpful Assistants task, and the 'Faithful vs. Summary' game for the Reddit Summary task. In each game, we assume that there are two groups whose joint preferences are captured by a reward model. For example, in 'Harmless vs. Humor, ' group 1 prioritizes harmlessness, while group 2 values humor. The corresponding reward models for these preferences are denoted as rm 1 (harmlessness) and rm 2 (humor), with synthetic group size vectors ( w 1 , w 2 ) selected from { (3 , 7) , (5 , 5) , (7 , 3) } , varying across different settings.

Implementation Details. We implement the basic training rule from Definition 4.1, using the KL-divergence as the distance measure f . To balance model optimality with training cost, we simplify the problem by replacing the entire parameter space Θ with a representative finite set Θ ′ .

2 The code for the simulation is available at GitHub.

Models are first trained using single reward models and then combined via the Rewarded Soups technique [62] to produce a set of hybrid models, { θ 1 , θ 2 , . . . , θ K } , which constitute Θ ′ . Optimality is then defined over this finite set. As shown in Rame et al. [62], this approach reduces training costs while maintaining performance comparable to full multi-objective fine-tuning.

Given the large space of potential misreporting strategies, we focus on two simple ones:

- Strategy (1) : ˜ rm i = rm i and ˜ w i = αw i Naïve overstatement : Exaggerating group size to gain more influence, requiring no knowledge of other groups.
- Strategy (2) : ˜ rm i = β rm i +(1 -β ) rm -i and ˜ w i = w i Strategic manipulation : Leveraging other groups' preferences to downplay opposing outcomes,

requiring some information about conflicts.

Intuitively, α = 1 and β = 1 represent truthful reporting. Increasing α or β allows a group to gain more influence in the training process. Our experiments confirm that both strategies can be profitable. However, the DSIC of our mechanism ensures that truthful reporting yields higher utility than any misreporting strategy.

Result Analysis. Since the outputs of different reward models have varying scales, we normalize all reward values to [0 , 1] , where the maximum and minimum values are 1 and 0, respectively. We then report the normalized valuations, payments, and utilities of group i for different reporting strategies in Figure 2. Each column represents a specific RLHF Game with a given group size ( w 1 , w 2 ) .

As shown in the figure, increasing α or β leads to a higher valuation for the group, confirming that groups can benefit from simple misreporting in the absence of payments. However, when the payment p AFF is applied, it increases with α or β , offsetting the gains in valuation. This ensures that truthful reporting ( α = 1 , β = 1 ) maximizes utility in all cases. Additional simulation settings are provided in Appendix F.

## 6 Conclusion and Future Work

This paper studies incentive issues in a potential economic scenario where a platform offers LLM fine-tuning services to aggregate preferences and agents strategically report to get a preferred outcome. We focus on aggregation objectives that maximize social welfare subject to regularization constraints, referred to as SW-Max rules. Through a comprehensive analysis of strategic reporting, we demonstrate the critical role of payment schemes in incentivizing truthful reporting under SW-Max rules. We derive sufficient conditions for payment equivalence and identify necessary conditions for implementing SWMax rules within additional constraints. Moreover, we analyze how perturbed input will influence the mechanism to account for practical errors that inevitably arise and show that the mechanism satisfies approximate DSIC. Finally, we conduct experiments within real-world LLM setups, showcasing how the proposed mechanism effectively incentivizes truthful reporting.

Building on our proposed scenario and formulated model, we identify several promising directions for future research from both theoretical and empirical perspectives. First, exploring and modeling more general training rules could enhance our understanding of the framework. As noted in Appendix E, cycle monotonicity is a necessary and sufficient condition for implementability, but its validation is complex. Identifying a simpler condition to ensure implementability and investigating properties like payment equivalence for these rules are critical next steps. Second, studying preference aggregation across multiple models, particularly with diversity considerations, is a valuable direction. Third, as discussed in Section 4.4, developing mechanisms or criteria that balance computational efficiency and incentive compatibility in the RLHF Game could improve its real-world applicability. Finally, applying mechanism design theory to other large language model contexts, such as API pricing, retrieval-augmented generation (RAG), and prompt engineering, offers significant opportunities for further exploration.

## Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant No. 62572010 and No. 62506010). We thank all anonymous reviewers for their helpful feedback.

## References

- [1] Saaket Agashe, Yue Fan, and Xin Eric Wang. Evaluating multi-agent coordination abilities in large language models. arXiv preprint arXiv:2310.03903 , 2023.
- [2] Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. arXiv preprint arXiv:2305.16867 , 2023.
- [3] Syed Mumtaz Ali and Samuel D Silvey. A general class of coefficients of divergence of one distribution from another. Journal of the Royal Statistical Society: Series B (Methodological) , 28(1):131-142, 1966.
- [4] Itai Ashlagi, Mark Braverman, Avinatan Hassidim, and Dov Monderer. Monotonicity and implementability. Econometrica , 78(5):1749-1772, 2010.
- [5] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862 , 2022.
- [6] Dirk Bergemann and Juuso Välimäki. The dynamic pivot mechanism. Econometrica , 78(2): 771-789, 2010.
- [7] Sushil Bikhchandani, Shurojit Chatterji, Ron Lavi, Ahuva Mu'alem, Noam Nisan, and Arunava Sen. Weak monotonicity characterizes deterministic dominant-strategy implementation. Econometrica , 74(4):1109-1132, 2006.
- [8] Stephen P Boyd and Lieven Vandenberghe. Convex optimization . Cambridge university press, 2004.
- [9] Patrick Briest, Shuchi Chawla, Robert Kleinberg, and S Matthew Weinberg. Pricing randomized allocations. In Proceedings of the twenty-first annual ACM-SIAM symposium on Discrete Algorithms , pages 585-597. SIAM, 2010.
- [10] Thomas Kleine Buening, Jiarui Gan, Debmalya Mandal, and Marta Kwiatkowska. Strategyproof reinforcement learning from human feedback. arXiv preprint arXiv:2503.09561 , 2025.
- [11] Souradip Chakraborty, Jiahao Qiu, Hui Yuan, Alec Koppel, Furong Huang, Dinesh Manocha, Amrit Singh Bedi, and Mengdi Wang. Maxmin-rlhf: Towards equitable alignment of large language models with diverse human preferences. arXiv preprint arXiv:2402.08925 , 2024.
- [12] Yiting Chen, Tracy Xiao Liu, You Shan, and Songfa Zhong. The emergence of economic rationality of gpt. Proceedings of the National Academy of Sciences , 120(51):e2316205120, 2023.
- [13] Alessandro Chiesa, Silvio Micali, and Zeyuan Allen Zhu. Mechanism design with approximate valuations. In Proceedings of the 3rd Innovations in Theoretical Computer Science conference , pages 34-38, 2012.
- [14] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems , 30, 2017.
- [15] Edward H Clarke. Multipart pricing of public goods. Public choice , pages 17-33, 1971.
- [16] Vincent Conitzer and Tuomas Sandholm. Self-interested automated mechanism design and implications for optimal combinatorial auctions. In Proceedings of the 5th ACM Conference on Electronic Commerce , pages 132-141, 2004.
- [17] Vincent Conitzer, Rachel Freedman, Jobst Heitzig, Wesley H Holliday, Bob M Jacobs, Nathan Lambert, Milan Mossé, Eric Pacuit, Stuart Russell, Hailey Schoelkopf, et al. Social choice for ai alignment: Dealing with diverse human feedback. arXiv preprint arXiv:2404.10271 , 2024.
- [18] Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward model ensembles help mitigate overoptimization. arXiv preprint arXiv:2310.02743 , 2023.

- [19] Imre Csiszár. On information-type measure of difference of probability distributions and indirect observations. Studia Sci. Math. Hungar. , 2:299-318, 1967.
- [20] Michael Curry, Tuomas Sandholm, and John Dickerson. Differentiable economics for randomized affine maximizer auctions. arXiv preprint arXiv:2202.02872 , 2022.
- [21] Zvi Drezner and Horst W Hamacher. Facility location: applications and theory . Springer Science &amp; Business Media, 2004.
- [22] Zhijian Duan, Haoran Sun, Yurong Chen, and Xiaotie Deng. A scalable neural network for dsic affine maximizer auction design. Advances in Neural Information Processing Systems , 36, 2024.
- [23] Zhijian Duan, Haoran Sun, Yichong Xia, Siqiang Wang, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng, and Xiaotie Deng. Scalable virtual valuations combinatorial auction design by combining zeroth-order and first-order optimization method. arXiv preprint arXiv:2402.11904 , 2024.
- [24] Kumar Avinava Dubey, Zhe Feng, Rahul Kidambi, Aranyak Mehta, and Di Wang. Auctions with llm summaries. arXiv preprint arXiv:2404.08126 , 2024.
- [25] Paul Duetting, Vahab Mirrokni, Renato Paes Leme, Haifeng Xu, and Song Zuo. Mechanism design for large language models. arXiv preprint arXiv:2310.10826 , 2023.
- [26] Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D'Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244 , 2023.
- [27] Meta Fundamental AI Research Diplomacy Team (FAIR)†, Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu, et al. Human-level play in the game of diplomacy by combining language models with strategic reasoning. Science , 378(6624):1067-1074, 2022.
- [28] Caoyun Fan, Jindou Chen, Yaohui Jin, and Hao He. Can large language models serve as rational players in game theory? a systematic analysis. arXiv preprint arXiv:2312.05488 , 2023.
- [29] Soheil Feizi, MohammadTaghi Hajiaghayi, Keivan Rezaei, and Suho Shin. Online advertisements with llms: Opportunities and challenges. arXiv preprint arXiv:2311.07601 , 2023.
- [30] Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, and Jun Wang. Chessgpt: Bridging policy learning and language modeling. Advances in Neural Information Processing Systems , 36, 2024.
- [31] Roberto Gallotta, Graham Todd, Marvin Zammit, Sam Earle, Antonios Liapis, Julian Togelius, and Georgios N Yannakakis. Large language models and games: A survey and roadmap. arXiv preprint arXiv:2402.18659 , 2024.
- [32] Kanishk Gandhi, Dorsa Sadigh, and Noah D Goodman. Strategic reasoning with language models. arXiv preprint arXiv:2305.19165 , 2023.
- [33] Ian Gemp, Yoram Bachrach, Marc Lanctot, Roma Patel, Vibhavari Dasagi, Luke Marris, Georgios Piliouras, and Karl Tuyls. States as strings as strategies: Steering language models with game-theoretic solvers. arXiv preprint arXiv:2402.01704 , 2024.
- [34] Theodore Groves. Incentives in teams. Econometrica: Journal of the Econometric Society , pages 617-631, 1973.
- [35] Shangmin Guo, Haochuan Wang, Haoran Bu, Yi Ren, Dianbo Sui, Yu-Ming Shang, and Siting Lu. Large language models as rational players in competitive economics games. arXiv preprint arXiv:2308.10032 , 2023.
- [36] Shangmin Guo, Haoran Bu, Haochuan Wang, Yi Ren, Dianbo Sui, Yuming Shang, and Siting Lu. Economics arena for large language models. arXiv preprint arXiv:2401.01735 , 2024.

- [37] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680 , 2024.
- [38] Birgit Heydenreich, Rudolf Müller, Marc Uetz, and Rakesh V Vohra. Characterization of revenue equivalence. Econometrica , 77(1):307-316, 2009.
- [39] Radosveta Ivanova-Stenzel and Timothy C Salmon. Revenue equivalence revisited. Games and Economic Behavior , 64(1):171-192, 2008.
- [40] Athul Paul Jacob, Yikang Shen, Gabriele Farina, and Jacob Andreas. The consensus game: Language model generation via equilibrium search. arXiv preprint arXiv:2310.09139 , 2023.
- [41] Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, and Prithviraj Ammanabrolu. Personalized soups: Personalized large language model alignment via post-hoc parameter merging. arXiv preprint arXiv:2310.11564 , 2023.
- [42] Philippe Jehiel, Moritz Meyer-Ter-Vehn, and Benny Moldovanu. Mixed bundling auctions. Journal of Economic Theory , 134(1):494-512, 2007.
- [43] Benjamin Laufer, Jon Kleinberg, and Hoda Heidari. Fine-tuning games: Bargaining and adaptation for general-purpose models. arXiv preprint arXiv:2308.04399 , 2023.
- [44] Anton Likhodedov and Tuomas Sandholm. Methods for boosting revenue in combinatorial auctions. In AAAI , pages 232-237, 2004.
- [45] Nunzio Lorè and Babak Heydari. Strategic behavior of large language models: Game structure vs. contextual framing. arXiv preprint arXiv:2309.05898 , 2023.
- [46] David G Luenberger, Yinyu Ye, et al. Linear and nonlinear programming , volume 2. Springer, 1984.
- [47] Weiyu Ma, Qirui Mi, Xue Yan, Yuqiao Wu, Runji Lin, Haifeng Zhang, and Jun Wang. Large language models play starcraft ii: Benchmarks and a chain of summarization approach. arXiv preprint arXiv:2312.11865 , 2023.
- [48] James Melbourne. Strongly convex divergences. Entropy , 22(11):1327, 2020.
- [49] Mitsunobu Miyake. On the incentive properties of multi-item auctions. International Journal of Game Theory , 27:1-19, 1998.
- [50] Gabriel Mukobi, Hannah Erlebach, Niklas Lauffer, Lewis Hammond, Alan Chan, and Jesse Clifton. Welfare diplomacy: Benchmarking language model cooperation. arXiv preprint arXiv:2310.08901 , 2023.
- [51] Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. arXiv preprint arXiv:2312.00886 , 2023.
- [52] Roger B Myerson. Incentive compatibility and the bargaining problem. Econometrica: journal of the Econometric Society , pages 61-73, 1979.
- [53] Roger B Myerson. Optimal auction design. Mathematics of operations research , 6(1):58-73, 1981.
- [54] Noam Nisan and Amir Ronen. Algorithmic mechanism design. In Proceedings of the thirty-first annual ACM symposium on Theory of computing , pages 129-140, 1999.
- [55] Noam Nisan et al. Introduction to mechanism design (for computer scientists). Algorithmic game theory , 9:209-242, 2007.
- [56] Jorge Nocedal and Stephen J Wright. Numerical optimization . Springer, 1999.

- [57] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems , 35:27730-27744, 2022.
- [58] Susan Hesse Owen and Mark S Daskin. Strategic facility location: A review. European journal of operational research , 111(3):423-447, 1998.
- [59] Chanwoo Park, Mingyang Liu, Kaiqing Zhang, and Asuman Ozdaglar. Principled rlhf from heterogeneous feedback via personalization and preference aggregation. arXiv preprint arXiv:2405.00254 , 2024.
- [60] Alessandro Pavan, Ilya Segal, and Juuso Toikka. Dynamic mechanism design: A myersonian approach. Econometrica , 82(2):601-653, 2014.
- [61] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. OpenAI , 2018.
- [62] Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. Advances in Neural Information Processing Systems , 36, 2024.
- [63] Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. Warm: On the benefits of weight averaged reward models. arXiv preprint arXiv:2401.12187 , 2024.
- [64] Kevin Roberts. The characterization of implementable choice rules. Aggregation and revelation of preferences , 12(2):321-348, 1979.
- [65] Jean-Charles Rochet. A necessary and sufficient condition for rationalizability in a quasi-linear context. Journal of mathematical Economics , 16(2):191-200, 1987.
- [66] Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715 , 2024.
- [67] Michael Saks and Lan Yu. Weak monotonicity suffices for truthfulness on convex domains. In Proceedings of the 6th ACM conference on Electronic commerce , pages 286-293, 2005.
- [68] Tuomas Sandholm and Anton Likhodedov. Automated design of revenue-maximizing combinatorial auctions. Operations Research , 63(5):1000-1025, 2015.
- [69] Xiao Shao, Weifu Jiang, Fei Zuo, and Mengqing Liu. Swarmbrain: Embodied agent for realtime strategy game starcraft ii via large language models. arXiv preprint arXiv:2401.17749 , 2024.
- [70] Ruizhe Shi, Yifang Chen, Yushi Hu, ALisa Liu, Noah Smith, Hannaneh Hajishirzi, and Simon Du. Decoding-time language model alignment with multiple objectives. arXiv preprint arXiv:2406.18853 , 2024.
- [71] Ermis Soumalias, Michael J Curry, and Sven Seuken. Truthful aggregation of llms with an application to online advertising. arXiv preprint arXiv:2405.05905 , 2024.
- [72] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems , 33:3008-3021, 2020.
- [73] Pingzhong Tang and Tuomas Sandholm. Mixed-bundling auctions with reserve prices. In AAMAS , pages 729-736, 2012.
- [74] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.

- [75] William Vickrey. Counterspeculation, auctions, and competitive sealed tenders. The Journal of finance , 16(1):8-37, 1961.
- [76] Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint arXiv:2401.06080 , 2024.
- [77] Shenzhi Wang, Chang Liu, Zilong Zheng, Siyuan Qi, Shuo Chen, Qisen Yang, Andrew Zhao, Chaofei Wang, Shiji Song, and Gao Huang. Avalon's game of thoughts: Battle against deception through recursive contemplation. arXiv preprint arXiv:2310.01320 , 2023.
- [78] Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. Advances in Neural Information Processing Systems , 36, 2024.
- [79] Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and Yang Liu. Exploring large language models for communication games: An empirical study on werewolf. arXiv preprint arXiv:2309.04658 , 2023.
- [80] Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. Language agents with reinforcement learning for strategic play in the werewolf game. arXiv preprint arXiv:2310.18940 , 2023.
- [81] Rui Yang, Xiaoman Pan, Feng Luo, Shuang Qiu, Han Zhong, Dong Yu, and Jianshu Chen. Rewards-in-context: Multi-objective alignment of foundation models with dynamic preference adjustment. arXiv preprint arXiv:2402.10207 , 2024.
- [82] Shun Zhang, Zhenfang Chen, Sunli Chen, Yikang Shen, Zhiqing Sun, and Chuang Gan. Improving reinforcement learning from human feedback with efficient reward model ensemble. arXiv preprint arXiv:2401.16635 , 2024.
- [83] Yadong Zhang, Shaoguang Mao, Tao Ge, Xun Wang, Adrian de Wynter, Yan Xia, Wenshan Wu, Ting Song, Man Lan, and Furu Wei. Llm as a mastermind: A survey of strategic reasoning with large language models. arXiv preprint arXiv:2404.01230 , 2024.

## Limitation

The main limitation of this paper is that we mainly consider the SW-Max training rules and their theoretical properties. Further study could consider more training rules and extend our model to the DPO scenario, in which each group only provides pairs of data rather than a reward model.

## A Further Related Work

In this section, we review relevant research across various domains that are related to our paper, including works on RLHF with multiple reward models, multi-parameter auctions, and the intersection of game theory and LLMs.

## A.1 RLHF with Multiple Reward Models

Research involving multiple reward models primarily focuses on developing algorithms to enhance practical performance. Some studies design methods simultaneously satisfying multiple preferences [11, 41, 62, 63, 70, 78, 81]. They develop more efficient algorithms to extend the Pareto frontier among different objectives [41, 62, 70, 81] and balance issues from various perspectives [11, 59, 63].

Additionally, there is a body of work that trains multiple models for a single preference and then ensembles them to improve the robustness of RLHF [18, 82], mitigate the influence of incorrect and ambiguous preferences in the dataset [76], and reduce reward hacking [26]. Unlike these approaches, our work considers how to collect misaligned preferences truthfully from different agents. As we have mentioned, these works are often assumed to be accessible to humans' actual preferences, neglecting the incentive issue for motivating rational agents to report truthfully.

## A.2 Multi-parameter Auctions

Several studies have explored the properties relevant to our paper in various multi-parameter auction scenarios, such as implementability [4, 7, 16, 49, 65, 67] and payment equivalence [6, 38, 39, 60]. Another central topic in auction theory is to design mechanisms that satisfy DSIC and IR while maximizing the expected revenue for the auctioneer. Although the single-parameter scenario has been resolved by Myerson [53], the optimal auction design for multi-parameter settings remains an open question. Therefore, there is a stream of research focusing on a specific subset, affine maximizer auctions, which inherently satisfy DSIC and IR [9, 42, 44, 64, 68, 73], and proposing optimizations to enhance empirical performance [20, 22, 23]. Compared to these works, we are the first to discuss the property of payment equivalence and the revenue-maximizing solution for SW-Max training rules in the scenario of fine-tuning LLMs.

## A.3 Game Theory and LLMs

In addition to the work we review in the primary related work, others have explored the intersection of game theory and large language models from different perspectives. A line of work studies other LLM-related scenarios from the algorithmic game theory perspective. Laufer et al. [43] abstracted the fine-tuning process as a bargaining game and characterized the perfect sub-game equilibria. Dubey et al. [24] proposed an auction where bidders compete to place their content within a summary generated by an LLM. Conitzer et al. [17] considered incorporating social choice theory in LLM alignment. Feizi et al. [29] explored the potential for leveraging LLMs in online advertising systems.

More broadly, some research has proposed algorithms for training LLMs inspired by concepts in game theory, such as Nash learning from human feedback [51], consensus game [40], direct Nash optimization [66], and Gemp et al. [33]. And various studies assess LLMs from a gametheoretical perspective, examining aspects such as rationality [12, 28], behavior in matrix games [2, 32, 45], and performance in strategic games like auctions [35, 36], Werewolf [79, 80], Avalon [77], Diplomacy [27, 50], card game [30] and electronic game [1, 47, 69]. There are also comprehensive surveys [31, 37, 83].

## B Omitted Proofs in Section 4.1

Theorem 4.2. In the RLHF Game with mechanism ( ψ, p ) that ψ ∈ Ψ SW and p ≡ 0 , for group i , define s i := |{ r | r = rm i ( x ) , x ∈ T ∗ }| and rm i := min x ∈ T ∗ rm i ( x ) :

1. If s i = 1 , truthfully reporting is the optimal strategy regardless of other groups' reports.
2. If s i ≥ 2 and rm i &gt; 0 , there is a strategy that yields strictly higher utility than truthfully reporting regardless of other groups' reports.

Proof. If s i = 1 , the group gets the same utility from all training outcomes. Therefore, any strategy is optimal. We then analyze the case s i ≥ 2 and rm i &gt; 0 in the following. First, the optimization of ψ can be written as an equivalent constraint programming problem on the LLM θ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Because of the assumption that the optimal policy satisfies LLM θ ( x ) &gt; 0 for all x ∈ T ∗ , we can infer that the condition LLM θ ( x ) ≥ 0 , ∀ x ∈ T ∗ is not active for the optimal solution. Since the convexity of the function f , by KKT condition, the necessary condition for the optimal θ ∗ is that there exists µ ∈ R [46], such that

<!-- formula-not-decoded -->

Under Definition 3.1, ∂v i ∂ LLM θ ( x ) = rm i ( x ) , so we have

<!-- formula-not-decoded -->

We mainly discuss the strategies other than simply over-reporting the group size ⃗ w . We omit the notation ⃗ w for simplicity.

̸

Next, our main technique is to construct a report reward model rm ′ i = rm i for group i such that v i ( ψ (( rm ′ i , - → rm -i ) , θ init ); rm i ) &gt; v i ( ψ (( rm i , - → rm ) , θ init ); rm i ) holds for all - → rm -i and θ init .

The Summation Normalization Case. We first discuss the case of the reward model being normalized by summation. We take the x 1 ∈ arg max x ∈ T ∗ rm i ( x ) , x 2 ∈ arg min x ∈ T ∗ rm i ( x ) . Since min x ∈ T ∗ rm i ( x ) &gt; 0 , we have rm i ( x 1 ) &lt; 1 and rm i ( x 2 ) &gt; 0 . Then we take a small ϵ &lt; min { 1 -rm i ( x 1 ) , rm i ( x 2 ) } and define rm ′ i as:

̸

<!-- formula-not-decoded -->

̸

Intuitively, by reporting rm ′ i , group i pretends to value more for the most preferred x and less for the least preferred x . Let θ = ψ (( rm i , - → rm -i ) , θ init ) and θ ′ = ψ (( rm ′ i , - → rm -i ) , θ init ) , we use µ and µ ′ to denote the variable in the necessary condition for LLM θ and LLM θ ′ , and we can derive the following results.

(a) LLM θ ′ ( x 1 ) &gt; LLM θ ( x 1 ) and LLM θ ′ ( x 2 ) &lt; LLM θ ( x 2 ) . We prove the former by contradiction: if LLM θ ′ ( x 1 ) ≤ LLM θ ( x 1 ) , then by the convexity of f , we have

<!-- formula-not-decoded -->

With rm ′ i ( x 1 ) &gt; rm i ( x 1 ) , we can infer that µ ′ &gt; µ . However, since for all x = x 1 , we have rm ′ i ( x ) ≤ rm i ( x ) , to satisfy the optimal condition in (OPT), there must be for all x = x 1 ,

<!-- formula-not-decoded -->

̸

̸

Which is equivalent to LLM θ ′ ( x ) &lt; LLM θ ( x ) , and hence results in ∑ x ∈ T ∗ LLM θ ′ ( x ) &lt; ∑ x ∈ T ∗ LLM θ ( x ) = 1 . The latter, LLM θ ′ ( x 2 ) &lt; LLM θ ( x 2 ) , can be proved by totally same method.

(b) The order of LLM θ ( x ) and LLM θ ′ ( x ) for all x / ∈ { x 1 , x 2 } is consistent. Without loss of generality, we assume there is x 3 / ∈ { x 1 , x 2 } such that LLM θ ′ ( x 3 ) ≥ LLM θ ( x 3 ) . Then we have

<!-- formula-not-decoded -->

Then, we can infer that µ ′ ≤ µ . For all x / ∈ { x 1 , x 2 } , to satisfy Equation (OPT), there must be

<!-- formula-not-decoded -->

which is equivalent to LLM θ ′ ( x ) ≥ LLM θ ( x ) . Similarly, if there is x 3 / ∈ { x 1 , x 2 } such that LLM θ ′ ( x 3 ) ≤ LLM θ ( x 3 ) , then for all x / ∈ { x 1 , x 2 } , there is LLM θ ′ ( x ) ≤ LLM θ ( x ) .

Finally, with the results in (a) and (b), when LLM θ ′ ( x ) ≤ LLM θ ( x ) for all x / ∈ { x 1 , x 2 } , the change in the utility of group i can be calculated by

̸

<!-- formula-not-decoded -->

̸

̸

When LLM θ ′ ( x ) ≥ LLM θ ( x ) for all x = x 1 , x 2 , the change in the utility of group i can be calculated by

̸

<!-- formula-not-decoded -->

̸

Note that both (2) and (3) are because of rm i ( x 1 ) ≥ rm i ( x 2 ) . And unless rm i ( x 1 ) = rm i ( x 2 ) , which is excluded by s i ≥ 2 , the ' &gt; 's are hold.

̸

̸

̸

̸

The Maximum Normalization Case. The case of the reward model being normalized by the maximum is similar. We take the x 1 ∈ arg min x ∈ T ∗ rm i ( x ) . Since min x ∈ T ∗ rm i ( x ) &gt; 0 , we have rm i ( x 1 ) &gt; 0 . Then we take a small ϵ &lt; rm i ( x 1 ) and define rm ′ i as:

̸

<!-- formula-not-decoded -->

With the same technique, we first show that LLM θ ′ ( x 1 ) &lt; LLM θ ( x 1 ) and LLM θ ′ ( x ) &gt; LLM θ ( x ) for all x = x 1 . After that, it is easy to derive that when s i ≥ 2 , the change in the utility of group i satisfies

<!-- formula-not-decoded -->

Lemma B.1. When the training rule ψ ∈ Ψ SW , and the divergence function f is α -strongly convex and C 2 -smooth, then ψ satisfies Definition 4.7.

Proof. As is shown in the proof of Theorem 4.2, we have two Lagrangian variables µ and µ ′ for ( - → rm , ⃗ w ) and ( - → rm , ⃗ w ) , respectively. We have the following equations:

<!-- formula-not-decoded -->

Firstly, we have | µ ′ -µ | ≤ max x ∈ T ∗ | ∑ n i =1 w i rm i ( x ) -∑ n i =1 w ′ i rm ′ i ( x ) | . Otherwise, without loss of generality, assume that µ ′ -µ &gt; max x ∈ T ∗ | ∑ n i =1 w i rm i ( x ) -∑ n i =1 w ′ i rm ′ i ( x ) | , then we can derive that ∀ x ∈ T ∗ ,

<!-- formula-not-decoded -->

This means that LLM θ ( x ) &lt; LLM θ ′ ( x ) for all x , which leads the contradiction. Therefore, we have for all x ∈ T ∗

<!-- formula-not-decoded -->

By C 2 -smoothness of f and the α -strongly convexity, we have for all x ∈ T ∗

<!-- formula-not-decoded -->

Therefore, for any ϵ &gt; 0 , if | ∑ n i =1 w i rm i ( x ) -∑ n i =1 w ′ i rm ′ i ( x ) | &lt; αϵ 2 , then | LLM θ ( x ) -LLM θ ′ ( x ) | ≤ ϵ .

Theorem B.2 (Detailed version of Theorem 4.3) . In the RLHF Game with mechanism ( ψ, p ) that ψ ∈ Ψ SW and p ≡ 0 , when f is α -strongly convex and C 2 -smooth, suppose group i has preference rm i and group size w i , other groups report ( - → rm -i , ⃗ w -i ) and the initial model θ init, we define

<!-- formula-not-decoded -->

in which θ = ψ ( - → rm , ⃗ w,θ init ) . When s i ≥ 2 and rm i = 0 :

̸

̸

1. For the maximum normalization case, if there exist x 1 ∈ T ∗ , t ( x 1 ) = 0 and 0 &lt; rm i ( x 1 ) &lt; 1 , truthful reporting is not the optimal strategy.

̸

2. For the summation normalization case, if there exist x 1 ∈ T ∗ , t ( x 1 ) &lt; 0 and 0 &lt; rm i ( x 1 ) &lt; 1 , truthful reporting is not the optimal strategy.
3. For the summation normalization case, if there exist x 1 ∈ T ∗ , t ( x 1 ) &gt; 0 and we can also find x 2 ∈ T ∗ , such that 1 &gt; rm i ( x 1 ) ≥ rm i ( x 2 ) &gt; 0 and 1 LLM θ init ( x 1 ) f ′′ ( LLM θ ( x 1 ) LLM θ init ( x 1 ) ) &lt; 1 LLM θ init ( x 2 ) f ′′ ( LLM θ ( x 2 ) LLM θ init ( x 2 ) ) , truthful reporting is not the optimal strategy.

Proof. As is shown in the proof of Theorem 4.2, the necessary condition for the solution θ is that there exists a µ ∈ R such that

<!-- formula-not-decoded -->

And by Lemma B.1, we can also use the condition Definition 4.7.

The Maximum Normalization Case (1). Without loss of generality, we assume that there exists x 1 such that t ( x 1 ) &gt; 0 , we take 0 &lt; ϵ &lt; 1 -rm i ( x 1 ) to construct a report rm ′ i

̸

<!-- formula-not-decoded -->

Suppose that µ ′ is the Lagrangian variable for the optimal solution θ ′ when reporting rm ′ i , we can derive that

<!-- formula-not-decoded -->

̸

With a similar analyze in the proof of Theorem 4.2, we can induce that µ ′ &gt; µ and LLM θ ′ ( x ) &lt; LLM θ ( x ) for all x = x 1 . By the C 2 -smoothness of f , for each x = x 1 , there exits a LLM θ ′ ( x ) ≤ z ≤ LLM θ ( x ) such that

̸

<!-- formula-not-decoded -->

For convenience, we let LLM θ ′′ ( x ) refer to the corresponding z for x , note that LLM θ ′′ is not necessarily a distribution. We then compute the change in the group i 's utility:

̸

̸

<!-- formula-not-decoded -->

Then, we show that when the ϵ we choose is sufficiently small, the above term is positive. We define the lower bound:

<!-- formula-not-decoded -->

Since function f is α -strongly convex, δ 1 ≥ α &gt; 0 . By the C 2 -smoothness of the f , there exists an δ 2 &gt; 0 , such that for each θ, θ ′ satisfying max x | LLM θ ( x ) -LLM θ ′ ( x ) | &lt; δ 2 , we have

<!-- formula-not-decoded -->

̸

Besides, because of the Definition 4.7, there exists δ 3 , such that for each ( ⃗ w, - → rm ) and ( ⃗ w ′ , - → rm ′ ) satisfying max x ∈ T ∗ | ∑ n i =1 w i rm i ( x ) -∑ n i =1 w ′ i rm ′ i ( x ) | ≤ δ 3 , we have max x | LLM θ ( x ) -LLM θ ′ ( x ) | &lt; δ 2 .

Combining these, we set ϵ &lt; δ 3 w i , then it is suffice to show that

̸

̸

<!-- formula-not-decoded -->

̸

̸

This means that

̸

<!-- formula-not-decoded -->

Combined with µ ′ &gt; µ , the proof concludes.

The Summation Normalization Case (2). Assume that there exists x 1 such that t ( x 1 ) &lt; 0 , we select x 2 := arg max x ∈ T ∗ rm i ( x ) and take 0 &lt; ϵ &lt; min { rm i ( x 1 ) , 1 -rm i ( x 2 ) } to construct a report rm ′ i

<!-- formula-not-decoded -->

̸

Still, we use µ ′ to denote the Lagrangian variable for the optimal solution θ ′ when reporting rm ′ i . Then, there are two possibilities for the relationship between µ and µ ′ . If µ ≤ µ ′ , by the optimal condition OPT, for all x = x 2 , we have LLM θ ( x ) ≥ LLM θ ′ ( x ) . Since x 2 has the highest reward value, such a change in the training outcome must be more preferred by the group i . Therefore, we only have to consider the case that µ &gt; µ ′ . Similarly, in this case, for all x = x 1 , we have LLM θ ( x ) &lt; LLM θ ′ ( x ) . By the C 2 -smoothness of f , for each x = x 1 , there exits a LLM θ ( x ) ≤ z ≤ LLM θ ′ ( x ) such that

<!-- formula-not-decoded -->

Let LLM θ ′′ ( x ) refer to the corresponding z for x , we then compute the change in the group i 's utility:

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

With the same technique we used in the maximum normalized case (1), we can show that with sufficient small ϵ &gt; 0 , the above term ∑ x = x 1 ( rm i ( x 1 ) -rm i ( x )) LLM θ init ( x ) f ′′ ( LLM θ ′′ ( x ) LLM θ init ( x ) ) &lt; t ( x 1 ) 2 &lt; 0 . Combined with µ ′ &lt; µ , the proof concludes.

̸

The Summation Normalization Case (3). Assume that there exists x 1 such that t ( x 1 ) &gt; 0 ,and x 2 , rm i ( x 1 ) ≥ rm i ( x 2 ) &gt; 0 , we take 0 &lt; ϵ &lt; min { rm i ( x 2 ) , 1 -rm i ( x 1 ) } to construct a report rm ′ i

<!-- formula-not-decoded -->

Still, we use µ ′ to denote the Lagrangian variable for the optimal solution θ ′ when reporting rm ′ i . Since we know for sure that LLM θ ( x 1 ) &lt; LLM θ ′ ( x 1 ) and LLM θ ( x 2 ) &gt; LLM θ ′ ( x 2 ) , by the C 2 -smoothness of f , LLM θ ′ ( x 2 ) ≤ LLM θ ′′ ( x 2 ) ≤ LLM θ ( x 2 ) and LLM θ ( x 1 ) ≤ LLM θ ′′ ( x 1 ) ≤ LLM θ ′ ( x 1 ) such that

<!-- formula-not-decoded -->

Let δ 1 := min x LLM θ init ( x ) , by the C 2 -smoothness of the f , there exists an δ 2 &gt; 0 , such that for each θ, θ ′ satisfying max x | w i rm i ( x ) -w ′ i rm ′ i ( x ) | &lt; δ 2 , we have

<!-- formula-not-decoded -->

We take ϵ &lt; δ 2 w i and first prove that when taking such ϵ , there is µ ≤ µ ′ . By contradiction, if µ ′ &lt; µ , then by condition Equation (OPT), for all x / ∈ { x 1 , x 2 } , there is LLM θ ′ ( x ) &gt; LLM θ ( x ) . Therefore, LLM θ ′ ( x 1 ) -LLM θ ( x 1 ) = ∑ x / ∈{ x 1 , x 2 } ( LLM θ ( x ) -LLM θ ′ ( x )) + LLM θ ( x 2 ) -LLM θ ′ ( x 2 ) &lt; LLM θ ( x 2 ) -LLM θ ′ ( x 2 ) . However, by Equation (2), if µ ′ &lt; µ , we get

<!-- formula-not-decoded -->

By Equation (3), we can derive that

<!-- formula-not-decoded -->

and thus, we get

<!-- formula-not-decoded -->

which brings the contradiction.

After proving that µ ≤ µ ′ , we know that for all x / ∈ { x 1 , x 2 } ,LLM θ ( x ) ≥ LLM θ ′ ( x ) . Then, by the C 2 -smoothness of f , for each x = x 1 , there exits a LLM θ ′ ( x ) ≤ z ≤ LLM θ ( x ) such that

<!-- formula-not-decoded -->

Let LLM θ ′′ ( x ) refer to the corresponding z for x , we then compute the change in the group i 's utility:

̸

<!-- formula-not-decoded -->

̸

̸

With the same technique we used in the maximum normalized case (1), we can show that with sufficient small ϵ &gt; 0 , the above term ∑ x = x 1 ( rm i ( x 1 ) -rm i ( x )) LLM θ init ( x ) f ′′ ( LLM θ ′′ ( x ) LLM θ init ( x ) ) &gt; t ( x 1 ) 2 &gt; 0 . Combined with µ ′ &lt; µ , the proof concludes.

## C Omitted Proofs in Section 4.2

Proposition 4.4. For any ψ ∈ Ψ SW , mechanism ( ψ, p AFF ) satisfies DSIC and IR, and the payment is non-negative.

Proof. We assume that for group i , the true reward model is rm i , and the agent number is w i . The reports of other groups are ( - → rm -i , ⃗ w -i ) and the initial model is θ init .

<!-- formula-not-decoded -->

We compare the utility between reporting ( rm i , w i ) and any other ( rm ′ i , w ′ i ) . For convenience, we first simplify the notations by letting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The valuation of group i is the valuation for each agent multiplied by the real agent number:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to the payment rule p AFF , the payment p i for ( rm i , w i ) and p ′ i for ( rm ′ i , w ′ i ) is

<!-- formula-not-decoded -->

Therefore, we can calculate the change in the utility:

<!-- formula-not-decoded -->

The last inequality holds by the definition of θ

<!-- formula-not-decoded -->

̸

̸

̸

Therefore, we can conclude that, for all - → rm, ⃗ w , rm ′ i , w ′ i , we have

<!-- formula-not-decoded -->

(2) ( ψ, p AFF ) satisfies IR.

We reuse the notations above and denote θ -i to be the optimal parameter for groups except for i , i.e. θ -i = ψ ( - → rm -i , ⃗ w -i , θ init ) . When group i truthfully report its reward model rm i and agent number w i , the utility can be written as:

<!-- formula-not-decoded -->

Therefore, we can conclude that, for all - → rm, ⃗ w , we have

<!-- formula-not-decoded -->

Proposition 4.6. When ⃗ w ≡ 1 is public information, and the agents only report the reward models, all implementable training rules satisfy payment equivalence.

Proof. We follow the result Theorem 1.37 in Nisan et al. [55].

Lemma C.1 (Theorem 1.37 in Nisan et al. [55]) . Let R i be group i 's preference domain. Assume that the R 1 , R 2 , . . . , R n are connected sets in the Euclidean space, then all implementable training rules ψ satisfy payment equivalence.

In our paper, we assume that for all i ∈ [ n ] , R i is the set of all non-negative and normalized | T ∗ | -dim vectors. Either in the summation normalization case or the maximum normalization case, this is a connected set in the Euclidean space. Hence, the theorem holds.

Proposition 4.8. SW-Max training rules with regularizations KL-divergence, f KL ( x ) = λx log x , and χ 2 divergence, f 2 ( x ) = λ ( x -1) 2 ( λ &gt; 0 is a constant) are continuous.

Proof. (1) For f KL ( x ) = λx log x (KL-divergence), since T ∗ is a finite set, we can rewrite the training rule ψ as an optimization problem as follows:

<!-- formula-not-decoded -->

Since for KL divergence, the optimal model LLM θ must satisfy that LLM θ ( x ) &gt; 0 , for all x ∈ T ∗ . The necessary condition for an optimal θ is that there exists µ ∈ R , such that

<!-- formula-not-decoded -->

Similarly, for the input ( - → rm ′ , ⃗ w ′ ) , there exists µ ′ ∈ R , such that the optimal θ ′ satisfies

<!-- formula-not-decoded -->

For convenience, we define ∆( x ) = ∑ n i =1 w ′ i rm ′ i ( x ) -∑ n i =1 w i rm i ( x ) . Then the relationship between LLM θ ( x ) and LLM θ ′ ( x ) is given by

<!-- formula-not-decoded -->

Note that we also have the condition

<!-- formula-not-decoded -->

Since ∑ x ∈ T ∗ LLM θ ( x ) e 1 λ (∆( x )+ µ -µ ′ ) = e 1 λ ( µ -µ ′ ) ∑ x ∈ T ∗ LLM θ ( x ) e 1 λ ∆( x ) , we can infer that

<!-- formula-not-decoded -->

This is equivalent to

<!-- formula-not-decoded -->

Thus, the difference for LLM θ ( x ) and LLM θ ′ ( x ) can be bounded by

<!-- formula-not-decoded -->

For any δ &gt; 0 , when we set max x ∈ T ∗ | ∆( x ) | ≤ min { λ 2 log 1 1 -δ , λ 2 log(1 + δ ) } , we have

<!-- formula-not-decoded -->

(2) For f 2 ( x ) = λ ( x -1) 2 ( χ 2 divergence), since T ∗ is a finite set, we can rewrite the training rule ψ as an optimization problem as follows:

<!-- formula-not-decoded -->

Since we have assumed a relatively large λ so that the optimal model LLM θ satisfies that LLM θ ( x ) &gt; 0 , for all x ∈ T ∗ . The necessary condition for an optimal θ is that there exists µ ∈ R , such that

<!-- formula-not-decoded -->

Similarly, for the input ( - → rm ′ , ⃗ w ′ ) , there exists µ ′ ∈ R , such that the optimal θ ′ satisfies

<!-- formula-not-decoded -->

For convenience, we define ∆( x ) = ∑ n i =1 w ′ i rm ′ i ( x ) -∑ n i =1 w i rm i ( x ) Then the relationship between LLM θ ( x ) and LLM θ ′ ( x ) is given by

<!-- formula-not-decoded -->

Note that we also have the condition

<!-- formula-not-decoded -->

Since ∑ x ∈ T ∗ LLM θ ( x ) = 1 , we can infer that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, the difference for LLM θ ( x ) and LLM θ ′ ( x ) can be bounded by

<!-- formula-not-decoded -->

For any δ &gt; 0 , when we set max x ∈ T ∗ | ∆( x ) | ≤ λδ , we have

<!-- formula-not-decoded -->

Theorem 4.9. An implementable training rule ψ satisfies payment equivalence if it is continuous and for ∀ i , - → rm -i , ⃗ w -i , θ init there exists rm ∗ i and θ such that ψ (( rm ∗ i , - → rm -i ) , ( w i , ⃗ w -i ) , θ init ) ≡ θ for all w i ∈ W . In the maximum normalization case, rm ∗ i must be 1 .

Proof. We prove the equivalent version of payment equivalence: For any group i , when fixing other groups reports ( - → rm -i , ⃗ w -i ) and θ init, any two payment rules p , p ′ that implement ψ in DSIC must satisfy that there exists a constant c , such that p i ( rm i , w i ) -p ′ i ( rm i , w i ) = c for any rm i and w i . Therefore, in the rest of the proof, we suppose fixed ( - → rm -i , ⃗ w -i ) and θ init and will omit these notations.

Firstly, we introduce a new notation t i to represent the combination ( rm i , w i ) , whose domain is R × W . Without specially claim, t i is used to represented for the rm i and w i with the same superscript and subscript, for example, t k i = ( rm k i , w k i ) . Then, we define the functions l ( · , · ) and V ( · , · ) as follows. l ( t ′ i , t i ) is the change in valuation from misreporting type t ′ i to reporting type t i truthfully. In formal,

<!-- formula-not-decoded -->

And V ( t ′ i , t i ) refers to the smallest values of l on a finite and distinct path from t ′ i to t i

<!-- formula-not-decoded -->

We prove the following lemma, which is a special case in Heydenreich et al. [38],

Lemma C.2 (Heydenreich et al. [38]) . In the RLHF Game, an implemented training rule ψ satisfies payment equivalence if for any agent i , and any types t i , t ′ i , we have

<!-- formula-not-decoded -->

Proof. Assume there is a mechanism ( ψ, p ) that satisfies DSIC. For any two types t i , t ′ i and a finite and distinct sequence [ t ′ i , t 1 i , . . . , t k i , t i ] , let t 0 i = t ′ i and t k +1 i = t i , we have that

<!-- formula-not-decoded -->

This can be rewritten as

<!-- formula-not-decoded -->

This is equivalent to

Sum over j , we get the following inequality

<!-- formula-not-decoded -->

Since this holds for arbitrary finite and distinct sequences, we can infer that V ( t ′ i , t i ) ≥ p ( t i ) -p ( t ′ i ) . Similarly, there is V ( t i , t ′ i ) ≥ p ( t ′ i ) -p ( t i ) . Combining these results with V ( t i , t ′ i ) = -V ( t ′ i , t i ) , there is

<!-- formula-not-decoded -->

which means that p ( t ′ i ) -p ( t i ) = V ( t i , t ′ i ) . Note that this holds for arbitrary t i and t ′ i . Therefore, when for some t i , the payment p ( t i ) is determined, then the payment for all other t ′ i s is determined. For example, if there are any two payment rules p and p ′ both implement ψ in DSIC, and we set the payment when i reports preference rm defined in Equation (5) and w i = 1 as p ∗ and p ′∗ respectively, then ∀ t i

<!-- formula-not-decoded -->

Note that p ∗ and p ′∗ are not influenced by i 's report, but they may vary for different - → rm -i , ⃗ w -i and θ init, which means that we can consider the term p ∗ -p ′∗ as a function f on ( - → rm -i , θ init ) .

Then, we show that the training rule satisfying the conditions in Theorem 4.9 is sufficient for the condition stated in Lemma C.2. Firstly, we show that for any t i , t ′ i , we have V ( t i , t ′ i ) + V ( t ′ i , t i ) ≥ 0 . By definition of the function V ( · , · ) , V ( t i , t ′ i ) and V ( t ′ i , t i ) correspond to the shortest path from t i to t ′ i and from t ′ i to t i respectively, which means that V ( t i , t ′ i ) + V ( t ′ i , t i ) is the shortest weight for a cycle that goes through t i and t ′ i . Since the SW-Max training rule is implementable, we know that the weight for any cycle is non-negative by cycle monotonicity [65]. Therefore, V ( t i , t ′ i ) + V ( t ′ i , t i ) ≥ 0 must be satisfied.

Then we show that for any t i , t ′ i and ϵ &gt; 0 , V ( t i , t ′ i ) + V ( t ′ i , t i ) ≤ ϵ . We prove this by constructing a finite and distinct sequence [ t i , t 1 i , . . . , t k i , t ′ i ] such that

<!-- formula-not-decoded -->

This suffices for proving V ( t i , t ′ i ) + V ( t ′ i , t i ) ≤ ϵ since V ( t i , t ′ i ) and V ( t ′ i , t i ) are the lower bound for ∑ k j =0 l ( t j i , t j +1 i ) and ∑ k j =0 l ( t j +1 i , t j i ) respectively.

Initially, we rewrite the LHS of Equation (4) by using the definition of the function l ( · , · ) .

<!-- formula-not-decoded -->

In the above equations, θ j = ψ ( t j i ) for 0 ≤ j ≤ k refers to the fine-tuned model when group i reports t j i .

By the condition, when - → rm -i , ⃗ w -i and θ init are fixed, there exits δ &gt; 0 such that if max x ∈ T ∗ | w i rm i ( x ) -w ′ i rm ′ i ( x ) | ≤ δ , then max x ∈ T ∗ | LLM θ ( x ) -LLM θ ′ ( x ) | ≤ ϵ 4 ¯ w (in maximum normalization case, we take ϵ 4 ¯ w | T ∗ | ), where θ := ψ (( rm i , - → rm -i ) , ( w i , ⃗ w -i ); θ init ) and θ ′ := ψ (( rm ′ i , - → rm -i ) , ( w ′ i , ⃗ w -i ); θ init ) .

We construct the sequence P as follows: we set k = 2 n , n ≥ ¯ w δ +1 and let t 0 i = t i , t k +1 i = t ′ i . For each 0 ≤ j ≤ n ,

<!-- formula-not-decoded -->

And for each n +1 ≤ j ≤ 2 n +1 ,

<!-- formula-not-decoded -->

Note that the rm ∗ i is the one given by the condition in Theorem 4.9. In this construction, any rm j i is either an weighted average of rm and rm ∗ i or rm ′ and rm ∗ i . This ensures that all reward models in the sequence are valid (normalized by summation or maximum and non-negative). We can then divide the above equation into three parts, making the w i the same in the first and the last parts.

<!-- formula-not-decoded -->

We first claim that (b) equals 0 . This is because of the property of rm n i = rm n +1 i = rm ∗ i , which can induces LLM θ n = LLM θ n +1 .

Then we turn to (a). By the construction, for any x ∈ T ∗ and 0 ≤ j ≤ n -1 , | w j i rm j i ( x ) -w j i rm j +1 i ( x ) | ≤ ¯ w n ≤ δ , so that | LLM θ j ( x ) -LLM θ j +1 ( x ) | ≤ ϵ 4 ¯ w holds for all x . Then we can derive that:

<!-- formula-not-decoded -->

The case is similar to (c). By the construction, for any x ∈ T ∗ and n +1 ≤ j ≤ 2 n , | w j i rm j i ( x ) -w j i rm j +1 i ( x ) | ≤ ¯ w n ≤ δ , so that | LLM θ j ( x ) -LLM θ j +1 ( x ) | ≤ ϵ 4 ¯ w holds for all x . Then we can

derive that:

<!-- formula-not-decoded -->

Combining the results from (a), (b), and (c), we have that under this construction,

<!-- formula-not-decoded -->

By the arbitrariness of ϵ &gt; 0 , this is suffice to demonstrate that V ( t i , t ′ i ) + V ( t i , t ′ i ) ≤ 0 .

Therefore, it is proven that

<!-- formula-not-decoded -->

which means that V ( t i , t ′ i ) = -V ( t ′ i , t i ) . By Lemma C.2, this is a sufficient condition for the payment equivalence of ψ .

Corollary 4.10. Each continuous training rule ψ ∈ Ψ SW satisfies payment equivalence.

Proof. We construct the reward model as follows and show that this satisfies the condition in Corollary 4.10 for when the mechanism uses SW-Max training rules.

<!-- formula-not-decoded -->

We prove this by contradiction. Assuming that there exist i , - → rm -i , ⃗ w -i , θ init , w i , w ′ i such that

̸

<!-- formula-not-decoded -->

We denote the further tie-breaking rule as ≻ - → rm . Then, considering the optimality of θ , we have one of the following satisfied.

<!-- formula-not-decoded -->

or

ASW( θ ; ( rm ∗ i , - → rm -i ) , ( w i , ⃗ w -i ) , θ init ) = ASW( θ ′ ; ( rm ∗ i , - → rm -i ) , ( w i , ⃗ w -i ) , θ init ) , and LLM θ ≻ - → rm LLM θ ′ .

Note that v i ( θ ; rm ∗ i ) = v i ( θ ′ ; rm ∗ i ) , and ASW( θ ; ( rm ∗ i , - → rm -i ) , ( w i , ⃗ w -i ) , θ init ) = ( w ′ i -w i ) v i ( θ ; rm ∗ i ) + ASW( θ ; ( rm ∗ i , - → rm -i ) , ( w ′ i , ⃗ w -i ) , θ init ) , we have

<!-- formula-not-decoded -->

or

ASW( θ ; ( rm ∗ i , - → rm -i ) , ( w ′ i , ⃗ w -i ) , θ init ) = ASW( θ ′ ; ( rm ∗ i , - → rm -i ) , ( w ′ i , ⃗ w -i ) , θ init ) , and LLM θ ≻ - → rm LLM θ ′ . Both cases contradicted the optimality of θ ′ .

Theorem 4.11. Given a continuous training rule ψ ∈ Ψ SW and a payment rule p implements it in DSIC: If p is always non-negative, it holds that for all i , - → rm, ⃗ w , and θ init ,

<!-- formula-not-decoded -->

If p implements ψ in IR, then for any ϵ &gt; 0 and i , there exists - → rm -i , ⃗ w -i , and θ init, such that for all rm i and w i ,

<!-- formula-not-decoded -->

Proof. For a continuous SW-Max training rule ψ , we know that it satisfies payment equivalence. By the definition of payment equivalence, for any other payment rule p that also implements ψ in DSIC, there exists a function g i such that

<!-- formula-not-decoded -->

Non-negative Payment. To ensure that p i ( - → rm , ⃗ w,θ init ) ≥ 0 always satisfied, we have the equivalent condition:

<!-- formula-not-decoded -->

However, for any - → rm -i , ⃗ w -i , θ init, when we set rm i to the uniform reward model Equation (5), we have shown in the previous proof that this will not change the training outcome regardless of the value of w i and hence does not impact the ASW -i . This means that the payment defined by the affine maximizer is exactly 0 , and the RHS of the above equation will always be non-negative. Therefore, there must be g i ≥ 0 for all inputs, which means that for all i , - → rm, ⃗ w , and θ init, we have p i ( - → rm , ⃗ w,θ init ) ≥ p AFF i ( - → rm , ⃗ w,θ init ) .

Individually Rationality. To ensure the utility of any group is not negative, we have to constrain the function g i as follows:

<!-- formula-not-decoded -->

where we denote u AFF i the utility of group i under the mechanism. We construct an extreme case such that the RHS can be sufficiently small. Without loss of generality, we assume that T ∗ = { x 1 , x 2 } . The initial model LLM θ init ( x 1 ) = ϵ , LLM θ init ( x 2 ) = 1 -ϵ . Group i has preference rm i ( x 1 ) = 1 and rm i ( x 2 ) = 0 , and other groups have opposite preference: rm j ( x 1 ) = 0 and rm j ( x 2 ) = 1 for j = i . The group size is set to w k = 1 for all k ∈ [ n ] .

̸

In this case, as we have ∑ n k =1 w k rm k ( x 1 ) &lt; ∑ n k =1 w k rm k ( x 2 ) , we can directly derived from the optimal condition Equation (OPT) that the final model satisfies that LLM θ ( x 1 ) ≤ LLM θ init ( x 1 ) . Since p AFF is always non-negative, the utility of group i is at most rm i ( x 1 ) · LLM θ init ( x 1 ) = ϵ . To ensure that p implements ψ in IR, we have to set g i ( - → rm -i , ⃗ w -i , θ init ) ≤ ϵ for this case. This is equivalent to p i ( - → rm , ⃗ w,θ init ) ≤ p AFF i ( - → rm , ⃗ w,θ init ) .

## D Omitted Proofs in Section 4.3

′ ′

Lemma D.1. For any rm , rm , if max x ∈ T ∗ | rm ( x ) -rm ( x ) | = ϵ , then for any model θ , we have

<!-- formula-not-decoded -->

Proof. We can derive that

<!-- formula-not-decoded -->

Lemma D.2. Assume that for any noisy input - → ̂ rm generated from F ( ·| - → rm ) , and i ∈ [ n ] , there is

<!-- formula-not-decoded -->

Then for any ψ ∈ Ψ SW and - → ̂ rm generated from F ( ·| - → rm ) , the distance between the training outcome and the optimal is bounded by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let ˆ θ = ψ ( - → ̂ rm , ⃗ w,θ init ) and θ = ψ ( - → rm , ⃗ w,θ init ) . ˆ θ is the optimal parameter for biased input, and θ is the optimal parameter for the true input.

<!-- formula-not-decoded -->

(1) and (3) can be directly induced by Lemma D.1, and (2) holds by the definition of ˆ θ .

<!-- formula-not-decoded -->

Theorem 4.12. Assume that for any noisy input - → ̂ rm generated from F ( ·| - → rm ) , and i ∈ [ n ] , there is

<!-- formula-not-decoded -->

Then, with a training rule ψ ∈ Ψ SW , ( ψ, p AFF ) ensures that each group i can benefit at most 2 w i ϵ from misreporting the reward model.

Proof. Recall that the calculation of payment in p AFF is

<!-- formula-not-decoded -->

Let ⃗ w = ( w i , ⃗ w -i ) , the utility function can be written as:

<!-- formula-not-decoded -->

where we define θ = ψ (( rm ′ i , - → rm -i ) , ⃗ w,θ init ) , and θ -i = ψ ( - → rm -i , ⃗ w -i , θ init ) . Note that the term ASW -i ( θ -i ; - → rm , ⃗ w,θ init ) is not influenced by the change of rm i or w i .

Therefore, we can derive that for any - → - →

̸

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

All the ˆ θ in the above inequalities refers to the optimal parameter for input ( ̂ rm i , - → rm -i ) , ⃗ w,θ init , i.e. ˆ θ = ψ (( ̂ rm i , - → rm -i ) , ⃗ w,θ init ) . Specifically, (1) and (3) come from the bounded distance between rm i and ̂ rm i (Lemma D.1). (2) and (5) hold by the definitions: ˆ θ = ψ (( ̂ rm i , - → rm -i ) , ⃗ w,θ init ) = arg max θ ′ ∈ Θ ASW( θ ′ ; ( ̂ rm i , - → rm -i ) , ⃗ w,θ init ) and θ = ψ (( rm i , - → rm -i ) , ⃗ w,θ init ) = arg max θ ′ ∈ Θ ASW( θ ′ ; ( rm i , - → rm -i ) , ⃗ w,θ init ) . And (4) holds since the inner term is irrelevant to ̂ rm i .

Therefore, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E Further Discussion on General Training Rules

In practice, some other training principles do not belong to SW-Max training rules, including those that maximize the Nash Social Welfare and focus more on fairness issues, like MaxMin-RLHF [11]. As an initial study on the incentive property of the RLHF Game, we primarily consider the mainstream training rules, SW-Max training rules, that aim to maximize social welfare under certain regularization.

Therefore, analyzing the properties of general forms of training rules is out of the scope of this paper. However, we also make a preliminary step for analyzing the two questions proposed in Section 4.2. The second question is partly included in Theorem 4.9, and for the implementability of a training rule, we utilize the notion of cycle monotonicity proposed by Rochet [65], which is a generalized version of monotonicity defined in a single-parameter scenario [53]. In the RLHF Game, we use the notation t i to represent the combination of ( rm i , w i ) with the same superscript and subscript. We define the function l ( t ′ i , t i ; ⃗ t -i , θ init ) := w i v i ( ψ (( t i , ⃗ t -i ) , θ init ); rm i ) -w i v i ( ψ (( t ′ i , ⃗ t -i , θ init )); rm i ) to measure group i 's valuation gains from misreporting ( t ′ i ) to truthfully reporting ( t i ) under ⃗ t -i and θ init. The cycle monotonicity is defined based on this function:

Definition E.1 (Cycle Monotonicity) . The training rule ψ satisfies cycle monotonicity if for any group i , t i , t ′ i ∈ R × W , any finite, distinct sequence of reward models [ t i , t 1 i , t 2 i , . . . , t k i , t ′ i ] ( k ≥ 0 ), and any ⃗ t -i , θ init, defining t 0 i = t k +2 i := t i and t k +1 i := t ′ i , we have

<!-- formula-not-decoded -->

For general training rules, cycle monotonicity is a sufficient and necessary condition for implementability.

Proposition E.2 (Rochet [65]) . A training rule ψ is implementable if and only if it satisfies cycle monotonicity.

Proof. We fix the other groups' report - → rm -i , ⃗ w -i , θ init, and also omit their notations for simplicity.

We first prove the necessity: if ψ is implementable, it satisfies cycle monotonicity. Since ψ is implementable, there exists p such that ( ψ, p ) satisfies DSIC. We use notation t j i to represent the combination of ( rm j i , w j i ) . For any types t i , t ′ i ∈ R × W , any finite and distinct sequence of types [ t i , t 1 i , t 2 i , . . . , t k i , t ′ i ] , k ≥ 0 , we let t 0 i = t k +2 i := t i and t k +1 i := t ′ i . By the property of DSIC, we have

<!-- formula-not-decoded -->

By definition of the function l , this is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the arbitrariness of the sequence [ t i , t 1 i , t 2 i , . . . , t k i , t ′ i ] , this means that ψ satisfies cycle monotonicity.

Then, we prove the sufficiency: By cycle monotonicity, we have that for any finite and distinct sequence [ t i , t 1 i , t 2 i , . . . , t k i , t ′ i ] ,

<!-- formula-not-decoded -->

By the arbitrariness of the sequence, we can infer that

<!-- formula-not-decoded -->

Since l ( t ′ i , t i ) is bounded, V ( t i , t ′ i ) is also finite and V ( t i , t ′ i ) ≥ -l ( t ′ i , t i ) . Then, we can establish a payment rule p such that for any agent i ,

<!-- formula-not-decoded -->

Sum over all j , we get where t ∗ ∈ R × W is a certain type.

Then, for any t i = ( rm i , w i ) , we have

<!-- formula-not-decoded -->

Note that (2) comes from the definition of V that:

<!-- formula-not-decoded -->

This means that mechanism ( ψ, p ) satisfies DSIC, and suffices to show that ψ is implementable.

Validating whether a training rule satisfies cycle monotonicity is a complex task. Thus, finding a more concise condition that can induce the implementability for a general training rule or a subset of training rules is a valuable further direction.

## F Additional Experimental Results

Synthetic RLHF Game. We construct a synthetic RLHF Game: We set the group number to be 5 and assume the size of the outcome space to be 10 . Each group's preference is first sampled from a uniform distribution U [0 , 1] 10 and then normalized. The group sizes are uniformly sampled from { 1 , 2 , . . . , 10 } 10 .

We consider the misreporting strategy that is used to prove Theorem 4.2. Specifically, given a group's preference rm. We first find the most preferred and the least preferred outcome x 1 = arg max x rm ( x ) , x 2 = arg min x rm ( x ) . Then we set the reported reward model to be ˜ rm ( x 1 ) = rm ( x 1 ) + ϵ , ˜ rm ( x 2 ) = rm ( x 2 ) -ϵ , and ˜ rm ( x ) = rm ( x ) for other x s.

Table 1: Average changes in valuation and utility when adopting the misreporting strategy from Theorem 4.2, holding other groups' reports fixed. The parameter ϵ controls the extent of deviation from truthful reporting. As shown in the table, such a misreporting strategy brings valuation gain but decreases the utility.

| Reporting Parameter ϵ   | Type   | 0.001    | 0.002    |   0.005 |    0.01 |     0.02 |     0.05 |      0.1 |
|-------------------------|--------|----------|----------|---------|---------|----------|----------|----------|
| ∆ Valuation (*1e2)      | Mean   | 0.1073   | 0.2096 < |  0.4881 |  0.8667 |   1.3674 |   1.7978 |   1.8154 |
|                         | Std    | < 0.0001 | 0.0001   |  0.0003 |  0.0004 |   0.0013 |   0.0026 |   0.0032 |
| ∆ Utility (*1e4)        | Mean   | -0.1064  | -0.4135  | -2.3696 | -8.1557 | -23.7334 | -53.1552 | -55.8977 |
| ∆ Utility (*1e4)        | Std    | < 0.0001 | 0.0001   |  0.0011 |  0.0046 |   0.0196 |   0.0573 |   0.1415 |

We let group 1 use this strategy and maintain the other group truthfully reporting. The payment is set according to the mechanism introduced in Section 4.2. We take 100 , 000 samples and the average change in valuation and utility for group 1 is reported in Table 1. The result shows that such a strategy can indeed improve the valuation and is, hence, beneficial when there is no payment. However, with the introduced payment, no strategy will bring higher utility than truthfully reporting.

More Complex Preferences. The experiment setup of this part follows the Section 5. We consider two scenarios with more complex, multiple preferences.

1. We simulated scenarios from data reported in [70] (Table 6), involving three groups, each valuing helpfulness, harmlessness, and humor, respectively. Normalization and other settings follow our paper. The true group sizes and the numerical results are shown in the tables below.
2. We examined a scenario where the group's preference is a linear combination of two reward models. Specifically, group 1 values 0.2 × Helpfulness + 0.8 × Harmlessness, group 2 values 0.8 × Helpfulness + 0.2 × Harmlessness, and group 3 values Humor.

All of the above results show that truthfully reporting is among the optimal strategies under the mechanism.

Table 2: Valuation, utility, and social welfare outcomes when varying reporting parameters for Group 1, with other groups' reports held fixed ( α = 1 means truthful reporting). Group sizes are set as ( w 1 , w 2 , w 3 ) = (3 , 2 , 1) . The three groups value Helpfulness, Harmlessness, and Humor, respectively. The highest value in each row is highlighted in bold .

| Reporting Parameter α         |   0.2 |   0.5 |    1 |   1.5 |   2 |   3 |
|-------------------------------|-------|-------|------|-------|-----|-----|
| Valuation                     |  0    |  0.79 | 2.66 |   3   | 3   | 3   |
| Utility (= Valuation-Payment) |  0    |  0.44 | 0.57 |   0.5 | 0.5 | 0.5 |
| Social Welfare                |  2.51 |  2.94 | 3.08 |   3   | 3   | 3   |

Table 3: Valuation, utility, and social welfare outcomes when varying reporting parameters for Group 1, with other groups' reports held fixed ( α = 1 means truthful reporting). Group sizes are set as ( w 1 , w 2 , w 3 ) = (4 , 5 , 3) . The three groups value Helpfulness, Harmlessness, and Humor, respectively. The highest value in each row is highlighted in bold .

| Reporting Parameter α         |   0.2 |   0.5 |    1 |   1.5 |     2 |     3 |
|-------------------------------|-------|-------|------|-------|-------|-------|
| Valuation                     |  0    |  0    | 1.05 |  1.05 |  3.54 |  4    |
| Utility (= Valuation-Payment) |  0    |  0    | 0.43 |  0.43 | -1.83 | -2.51 |
| Social Welfare                |  6.51 |  6.51 | 6.94 |  6.94 |  4.68 |  4    |

Table 4: Valuation, utility, and social welfare outcomes when varying reporting parameters for Group 1, with other groups' reports held fixed ( β = 1 means truthful reporting). Group sizes are set as ( w 1 , w 2 , w 3 ) = (5 , 5 , 2) . The three groups value Helpfulness, Harmlessness, and Humor, respectively. The highest value in each row is highlighted in bold .

| Reporting Parameter β         |   0.5 |   0.8 |    1 |   1.5 |     2 |     3 |
|-------------------------------|-------|-------|------|-------|-------|-------|
| Valuation                     |  0    |  0.33 | 1.31 |  4.67 |  5    |  5    |
| Utility (= Valuation-Payment) |  0    |  0.09 | 0.2  | -0.72 | -1.01 | -1.01 |
| Social Welfare                |  6.01 |  6.1  | 6.2  |  5.29 |  5    |  5    |

Table 5: Valuation, utility, and social welfare outcomes when varying reporting parameters for Group 1, with other groups' reports held fixed ( β = 1 means truthful reporting). Group sizes are set as ( w 1 , w 2 , w 3 ) = (3 , 1 , 4) . The three groups value Helpfulness, Harmlessness, and Humor, respectively. The highest value in each row is highlighted in bold .

| Reporting Parameter β         |   0.5 |   0.8 |    1 |   1.5 |     2 |     3 |
|-------------------------------|-------|-------|------|-------|-------|-------|
| Valuation                     |  0.79 |  0.79 | 0.79 |  0.79 |  2.66 |  3    |
| Utility (= Valuation-Payment) |  0.79 |  0.79 | 0.79 |  0.79 | -1.04 | -1.58 |
| Social Welfare                |  5.37 |  5.37 | 5.37 |  5.37 |  3.54 |  3    |

Table 6: Valuation, utility, and social welfare outcomes when varying reporting parameters for Group 1, with other groups' reports held fixed ( α = 1 means truthful reporting). Group sizes are set as ( w 1 , w 2 , w 3 ) = (2 , 3 , 1) . The three groups value 0 . 8 × Helpfulness + 0 . 2 × Harmlessness, 0 . 2 × Helpfulness +0 . 8 × Harmlessness, and Humor, respectively. The highest value in each row is highlighted in bold .

| Reporting Parameter α         |   0.2 |   0.5 |    1 |   1.5 |    2 |    3 |
|-------------------------------|-------|-------|------|-------|------|------|
| Valuation                     |  0.53 |  0.53 | 1.03 |  1.51 | 1.6  | 1.6  |
| Utility (= Valuation-Payment) |  0.52 |  0.52 | 0.61 |  0.39 | 0.31 | 0.31 |
| Social Welfare                |  2.92 |  2.92 | 3.01 |  2.79 | 2.71 | 2.71 |

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We are sure that the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The discussion is put in the appendix.

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

Justification: All assumptions and rigorous proofs are provided.

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

Justification: The full information combined with the code is provided.

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

## Answer: [Yes]

Justification: We use open-source datasets, and some data is simulated from certain distributions, which are described in the paper.

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

Justification: All details are provided.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: For the deterministic numerical simulation, there are no error bars. For others, we have provided clarification on the error bars.

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

Justification: All information is provided in the README file of the code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have checked.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

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

Justification: The paper poses no such risks.

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
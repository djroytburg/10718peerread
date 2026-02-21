## Structure Matters: Dynamic Policy Gradient

Sara Klein 1 , 3 Xiangyuan Zhang 2 Tamer Ba¸ sar 2 Simon Weissmann 1 Leif Döring 1 Institut for Mathematics, University of Mannheim, 68138 Mannheim {simon.weissmann, leif.doering}@uni-mannheim.de

1

2 Department of Electrical and Computer Engineering, and Coordinated Science Laboratory, University of Illinois Urbana-Champaign, Urbana, IL 61801 {xz7, basar1}@illinois.edu

3 August-Wilhelm Scheer Institute for Digital Products and Processes gGmbH, 66123 Saarbrücken sara.klein@aws-institut.de

## Abstract

In this work, we study γ -discounted infinite-horizon tabular Markov decision processes (MDPs) and introduce a framework called dynamic policy gradient (DynPG). The framework directly integrates dynamic programming with (any) policy gradient method, explicitly leveraging the Markovian property of the environment. DynPG dynamically adjusts the problem horizon during training, decomposing the original infinite-horizon MDP into a sequence of contextual bandit problems. By iteratively solving these contextual bandits, DynPG converges to the stationary optimal policy of the infinite-horizon MDP. To demonstrate the power of DynPG, we establish its non-asymptotic global convergence rate under the tabular softmax parametrization, focusing on the dependencies on salient but essential parameters of the MDP. By combining classical arguments from dynamic programming with more recent convergence arguments of policy gradient schemes, we prove that softmax DynPG scales polynomially in the effective horizon (1 -γ ) -1 . Our findings contrast recent exponential lower bound examples for vanilla policy gradient.

## 1 Introduction

The overarching goal in reinforcement learning (RL) is to train an agent that interacts with an unknown environment. This is mathematically modeled through a Markov decision process (MDP), where the agent actively learns a policy that, given a state, executes actions and receives rewards in return. The objective is to find a policy that maximizes the expected discounted reward. We categorize RL algorithms into two groups. Rather than the conventional separation between value-based and policy-based algorithms, we distinguish between algorithms that leverage the model's structure and those that do not. The first class utilizes the dynamic programming (DP) principle, such as value iteration, policy iteration, or Q-learning. Those algorithms optimize essentially single-period problems, where a single reward feedback is used in the update procedure and the future payoff is estimated by evaluation. The second class considers the entire multi-period problem at once and solves it using classical optimization, as in policy gradient (PG) methods [29, 27, 13, 10]. Over time, hybrid approaches integrating both methods, such as actor-critic (AC) methods, have consistently demonstrated superior performance in practical applications even though dynamic programming is used rather indirectly. In AC, the critic suggests a baseline, estimating either the value or action-value function. Dynamic programming becomes essential during the critic update process, effectively diminishing variance in gradient estimation. The essence of employing dynamic programming lies in the fact that parameter updates are not solely reliant on new trajectories, information is bootstrapped. Efforts of a broad research community have led to tremendous success of AC type algorithms (such as natural policy gradient (NPG), trust region policy optimization (TRPO), proximal policy optimization

(PPO)) but these have not yet been backed by a sufficient theoretical understanding. This paper contributes to the basic theory of PG methods by studying the following questions:

To what extent can the Markov property of an MDP be exploited directly to improve convergence of PG methods? How much improvement is gained compared to vanilla PG?

This paper addresses both points by merging dynamic programming and policy gradient into dynamic policy gradient (DynPG) and analyzing its convergence behavior. DynPG can be seen as a hybrid RL algorithm that utilizes policy gradient to optimize a sequence of contextual bandits. In each iteration, the algorithm extends the horizon of the MDP by adding an additional epoch in the beginning and shifting trained policies to the future, as in applying the Bellman operator. A policy for the newly added epoch is trained by policy gradient where previous policies are used to determine future actions (cf., Figure 1). DynPG is not an AC method. It reduces the variance as far as possible by applying the previously trained, and therefore fixed, policies to generate a trajectory. This leads to stable estimates of the rewards-to-go and improves convergence behavior. In this work, we first present a general error analysis which is compatible with any optimization scheme to address each contextual bandit problem such as PG, NPG or policy mirror descent [30]. We then employ vanilla PG to present an explicit complexity bound under softmax parametrization.

## 1.1 Related work

The idea of using dynamic programming in searching the optimal policy is not new and dates back to the early 2000s, having been revisited several times in recent decades. Policy search by dynamic programming (PSDP) [3, 11, 21] is most closely related to DynPG and searches for optimal policies in a restricted policy class without explicitly using gradient ascent to find the optimal policy. Next to this CBMPI in [22] is also related to DynPG. Here the greedy policy is based on a previous value-iteration step. However in DynPG, no value iteration is necessary. There is a line of similar algorithms like approximate policy iteration (API), gradient temporal difference, policy dynamic programming and numerous other variations [5, 25, 26, 2, 9]. In these works, however, PG was not used as policy optimization step. In [12], a similar concept, also called dynamic policy gradient, was employed to tackle finite-time MDPs without discounting. Their motivation to utilize dynamic programming stemmed from seeking non-stationary optimal policies in finite-time MDPs. In contrast to their findings, our objective is to identify a stationary optimal policy and incorporate discounting into the framework. The present paper can be seen as a continuation of that line of research towards infinitetime horizon MDPs. Lastly, in their study of linear-quadratic continuous control and estimation problems, [33, 34, 35] proposed a receding-horizon policy gradient algorithm, which also integrates dynamic programming-based receding-horizon control with policy gradient methods.

The discount factor γ ∈ (0 , 1) plays an essential role in the convergence behavior of RL algorithms, as they explicitly depend on the contraction property of the Bellman operator. The closer γ to one, the slower the Bellman operator contracts. Similarly, the convergence of PG methods also heavily depends on γ but establishing a clear dependence is generally challenging due to the non-convex optimization landscape. Early works only concern convergence to stationary points [18, 23, 17]. More recently, specific to tabular settings and the softmax parametrization, [1, 16] have derived upper bounds on the convergence rate to a global optimum by utilizing a gradient domination property [19]. As shown in [16], vanilla softmax PG has a sublinear convergence rate of O ( ϵ -1 ) with respect to the error tolerance ϵ . Dependencies on other salient but essential parameters of the MDP crucially affect the overall convergence behavior of PG methods, e.g., the effective horizon (1 -γ ) -1 . Notably, [15] constructed a counterexample such that vanilla PG could take an exponential time (with respect to (1 -γ ) -1 ) to converge. Although the optimal solution can be reached in just |S| steps of exact value iteration in the counter example, vanilla softmax PG is very inefficient by not leveraging the inherent structure of the MDP. In contrast, for softmax DynPG, we establish an explicit dependency on the discount factor. The convergence rate under exact gradients scales with (1 -γ ) -4 up to logarithmic factors; thus, we delineate the unknown model-dependent constant C ( γ ) in the rate of vanilla softmax PG [16]. As a result, DynPG can efficiently address the counterexample of [15]; see Section 5. Table 1 summarizes the complexity bounds for softmax PG and softmax DynPG.

Table 1: Comparison of convergence rate under exact gradients

| algorithm                 | complexity bounds                                              | reference       |
|---------------------------|----------------------------------------------------------------|-----------------|
| softmax PG upper bound    | O ( C ( γ )(1 - γ ) - 6 ϵ - 1 )                                | [16, Thm. 4]    |
| softmax PG lower bound    | &#124;S&#124; 2 Ω((1 - γ ) - 1 ) gradient steps for ϵ = 0 . 15 | [15, Thm. 1]    |
| softmax DynPG upper bound | O ( (1 - γ ) - 4 ϵ - 1 log((1 - γ ) - 2 ϵ - 1 ) )              | [ our Thm. 4.8] |

## 1.2 Main contributions and outline

The main contribution of this article is to introduce and analyze a way to directly combine policy gradient and dynamic programming ideas. The algorithm we introduce in Section 3 is called DynPG (dynamic policy gradient method). The algorithm (provably) circumvents recent worst case lower bound problems for standard PG that prove impracticability of plain vanilla PG in delicate environments. In Section 4 we provide a detailed error decomposition to show the convergence behavior of DynPG in general frameworks. For the tabular softmax parametrization we prove rigorous upper bounds on the convergence rate in Section 4.2 for the theoretical setting of exact gradients. Most importantly, the γ -dependence is harmless, DynPG does not suffer from exponential dependencies in γ ! A simple example is presented in Section 3 to show in a sample based setting how DynPG can beat PG in environments that trick PG into so-called committal behavior.

## 2 Preliminaries

Let the tuple M = ( S , A , p, r, γ ) denote an MDP, where S represents the state space, A represents the action space, p : S × A → ∆( S ) is the transition kernel mapping each state-action pair to the probability simplex of all possible next states ∆( S ) , r : S × A → R is the reward function mapping each state-action pair to a reward, µ ∈ ∆( S ) denotes the initial state distribution, and γ ∈ (0 , 1) is the discount factor. We assume that the reward is bounded such that r ( s, a ) ∈ [ -R ∗ , R ∗ ] for all ( s, a ) ∈ S × A and the state and action spaces are finite. Hereafter, we use capital and lowercase letters to distinguish random state and actions from deterministic ones. For x ∈ R d , we denote its supremum norm by ∥ x ∥ ∞ = max i =1 ,...,d | x i | ; we refer to Appendix A for complete notation conventions.

A policy π : S → ∆( A ) is a mapping from a state s ∈ S to a distribution over the action space, i.e., π ( ·| s ) ∈ ∆( A ) . The set of all policies is denoted by Π . We refer to h ∈ N as the deterministic timehorizon, where the case h = ∞ corresponds to the standard infinite-horizon MDP. Non-stationary policies with time-horizon h are denoted by π h := { π h -1 , . . . , π 0 } ∈ Π h . Let V 0 ≡ 0 and, for all h &gt; 0 , define the h -step value function of policies π h ∈ Π h as

<!-- formula-not-decoded -->

When µ is a Dirac measure at s we let V π h h ( s ) := V π h h ( δ s ) . Subsequently, we interpret a function V : S → R as a vector in R |S| . Additionally, we use V π h ( µ ) := V { π,...,π } h ( µ ) to denote the value function of the stationary policy π being applied h times in a row. For h = ∞ the resulting infinite-horizon discounted MDP admits a stationary optimal policy [20]. We define V ∗ ∞ ( µ ) := sup π ∈ Π V π ∞ ( µ ) and use π ∗ to denote a stationary policy that achieves V π ∗ ∞ ( µ ) = V ∗ ∞ ( µ ) . In contrast, when h is finite, the finite-horizon MDP optimization problem needs non-stationary optimal policies; thus, we define V ∗ h ( µ ) := sup π h ∈ Π h V π h h ( µ ) and use π ∗ h := { π ∗ h -1 , . . . , π ∗ 0 } ∈ Π h to denote a sequence of policies that achieves V π ∗ h h ( µ ) = V ∗ h ( µ ) .

For any function V ∈ R |S| and stationary policy π , the Bellman expectation operator T π : R |S| → R |S| is defined for every s ∈ S by

<!-- formula-not-decoded -->

The Bellman's optimality operator T ∗ : R |S| → R |S| is then for every s ∈ S given by

<!-- formula-not-decoded -->

The operators T π and T ∗ are γ -contracting with unique fixed points denoted by V π ∞ and V ∗ ∞ , respectively. In addition, the optimal h -step value functions V ∗ h can be obtained by iteratively applying T ∗ and, as h →∞ , V ∗ h converges to V ∗ ∞ (c.f. Lemma C.1).

Vanilla Policy Gradient. Policy gradient is a policy search method for which a family of differential parameterized policies ( π θ ) θ ∈ R d is fixed, say π θ ( ·| s ) := π ( ·| s, θ ) , and the optimal policy is searched using gradient ascent:

<!-- formula-not-decoded -->

Here, η &gt; 0 is the step size. Analyzing the convergence of vanilla PG is generally non-trivial due to the non-convexity of V π θ ∞ and the approximation error of the policy approximation family. Specialized to tabular MDPs, [1] has shown global convergence of vanilla PG under the softmax parametrization, and [16] established a non-asymptotic convergence rate. It should be emphasized that convergence is slow and constants can easily dominate convergence rates.

Tabular Softmax Policy. Policy gradient can be applied in the tabular setting (i.e. considering separately each state-action pair). Although the tabular setting is not used in practical applications, it is the most tractable setting for a complete mathematical analysis and sheds light on general principles. We introduce the logit function θ : S × A → R and the softmax policy parametrized by θ ∈ R |S||A|

<!-- formula-not-decoded -->

The tabular softmax policy can approximate any stationary and therefore also any optimal policy, whereas other parameterizations such as neural networks may induce an approximation error. This error needs to be considered in the convergence analysis.

Policy Search Framework. Inspired by dynamic programming, the authors in [3, 11] proposed Policy Search by Dynamic Programming to search policies { π H -1 , · · · , π 0 } ∈ ˜ Π H , where H is the problem horizon given as an input to the algorithm and ˜ Π ⊆ Π is the set of all deterministic policies. Formally, given H and ˜ Π , PSDP computes

<!-- formula-not-decoded -->

for h = 0 , . . . , H -1 . PSDP solves an ( h +1) -step MDP initialized at an S 0 ∼ µ in the iteration indexed by h . Specifically, it finds the optimal deterministic policy for selecting the first action A 0 , denoted by ˜ π ∗ h , but then all the remaining actions in the episode { A 1 , · · · , A h -1 } are selected according to the sequence of deterministic policies ˜ π ∗ h := { ˜ π ∗ h -1 , · · · , ˜ π ∗ 0 } . Note that for all h , ˜ π ∗ h have been computed in previous iterations and are kept fixed. Compared to vanilla PG, PDSP exploits the Markovian property of the environment, rendering each iteration into solving a contextual bandit problem. While considering deterministic policies may suffice in tabular MDPs, no explicit computational procedures have been provided in [3, 11] to solve the optimization problem in (4). Further, determining the policy to be applied post-training is not immediately evident. The author in [21] proposse to apply the non-stationary policy ˜ π ∗ H in a loop. Still, to solve the discussed infinite horizon MDP, a stationary policy is sufficient. We provide answers to these issues using DynPG.

## 3 Dynamic Policy Gradient

The DynPG Algorithm. DynPG starts by solving a one-step contextual bandit problem and then incrementally extends the problem horizon by one in each iteration, which is done by appending the new decision epoch in front of the current problem horizon (see the illustration in Figure 1). In each iteration, the parametrized (stochastic) policy responsible for sampling action A 0 from the newly-added epoch is optimized using gradient ascent. In subsequent time steps, DynPG applies the previous convergent policies to sample actions. Upon convergence of the current iteration, the

## Algorithm 1: DynPG

<!-- image -->

<!-- image -->

2

3

Figure 1: DynPG solves a sequence of contextual bandit problems, iteratively storing the convergent policies to memory and applying them accordingly as fixed policies in later iterations.

policy is stored in a designated memory location, which will be utilized in future iterations. We define ˆ π ∗ h := { ˆ π ∗ h -1 , · · · , ˆ π ∗ 0 } ∈ Π h with convention ˆ π ∗ 0 = ∅ to denote the learned policies. DynPG returns the first policy in Λ when certain user-defined convergence criteria have been met. For example, this is the case if the relative value improvements between two consecutive DynPG iterations are small i.e., ∥ V ˆ π ∗ h +1 -V ˆ π ∗ h ∥ ∞ ≤ ϵ .

<!-- formula-not-decoded -->

Hereafter, we let V 0 ≡ 0 ; the construction of DynPG implies

<!-- formula-not-decoded -->

which will be employed in the analyses of the convergence rate.

Compared to vanilla PG in (2), DynPG dynamically adjusts the episode horizon during the execution of the algorithm. This results in two notable advantages. Firstly, we observe that DynPG effectively reduces the variance in gradient estimation through the utilization of non-changing future policies. In contrast, the estimation of the gradient in vanilla PG involves assessing rewards-to-go based on the stationary policy π θ , which changes during training. Secondly, DynPG requires significantly fewer samples, a topic we discuss in detail later in this section. Moreover, each DynPG iteration has more benign optimization landscapes, as they are essentially contextual bandit problems. Specifically, DynPG has a simpler PG theorem (cf. Theorem C.3) and under softmax parametrization a smaller smoothness constant, enabling the selection of a more aggressive step size, η h , to enhance convergence.

In contrast to PSDP, we specify the set of policies ˜ Π to be a parameterized class of differentiable policies and a computational procedure, namely PG, for the inner loop. It is straight forward to incorporate function approximation (e.g. using neural networks) to the DynPG algorithm. This preserves the model-free optimization characteristic of gradient methods, while still exploiting the underlying structure of MDPs by Dynamic Programming. DynPG can be further modified with additional enhancements such as regularization, natural policy gradient or policy mirror descent in ever optimization epoch. In addition, we show that applying the policy of the last training epoch as stationary policy is sufficient. This is a non-trivial result and our analysis in Section 4 reveals that in general it requires additional training to obtain a good stationary policy compared to using the non-stationary policy ˆ π ∗ H in a loop as proposed in [21]. In the tabular softmax case we specify these additional computational cost explicitly.

On the total sample complexity. For each gradient step in standard policy gradient, one must run the MDP until termination or up to a stochastic horizon H ∼ Geom (1 -γ ) to obtain an unbiased sample of the gradient [32]. DynPG, on the other hand, only requires h interactions with the environment to sample an unbiased estimator of the gradient in epoch h . Thus, comparing other policy

̂

̂

̂

̂

̂

̂

̂

̂

̂

̂

̂

gradient methods to DynPG solely based on the number of gradient steps is inadequate. Instead, for fairness, the number of samples (interactions with the environment) should be compared in practical implementations. For DynPG the total sample complexity is given by ∑ H -1 h =0 ( h +1) N h . This results in a trade-off between increasing samples required for estimation and more accurate training to obtain convergence (cf. Section 4.1).

A Numerical Example. To demonstrate DynPG's effectiveness we present a numerical study of a canonical example where vanilla PG suffers from committal behavior. A detailed description of the MDP and the experimental setup can be found in Appendix B. We note that the theoretical analysis for softmax DynPG in Section 4.2 demonstrates its practical usefulness in fine-tuning the learning rates.

In the figure to the right we plotted the success probability to achieve an overall error in the value function of less than ϵ = 0 . 01 from the optimal, i.e. V ∗ ∞ -V ˆ π ∗ H ∞ ≤ 0 . 01 . The success probability is calculated by executing each algorithm 2000 times with randomly sampled initial states and the x-axis is the number of interactions with the environment used by both samplebased vanilla PG and sample-based DynPG. We observe that vanilla PG struggles solving this MDP, suffering from the high reward variance in certain states. Vanilla PG tends to concentrate on large rewards samples while DynPG circumvents this committal behavior by more accurate estimation of the future Q-values. It can be observed that the performance of DynPG and vanilla PG are similar for the first 400 interactions with the environment. However, vanilla

Figure 2: Success probability of achieving the suboptimality gap of ϵ = 0 . 01 in the overall error.

<!-- image -->

PG fails to converge to the optimal policy and can only reach the success probability of around 0 . 8 under the given training budget. As usually for variants of vanilla PG, we expect that DynPG will not always be more efficient than vanilla PG. While vanilla PG is stuck in committal behavior in the environment chosen for the simulation, there are other (simpler) environments in which vanilla PG performs well. In that case our more complex algorithm will not beat vanilla PG or other equally well-performing variants, but we expect comparable performance instead.

DynAC. The DynPG algorithm requires to store a sequence of policies which for large problems might result in memory issues. To mitigate the problem there is a natural actor-critic type extension, that we call DynAC. Since an analysis is out of the scope of this article DynAC is described in the appendix, compare Section E.

## 4 Convergence Analysis

Our numerical example suggests improved convergence properties for DynPG. In this section we provide complexity bounds for exact gradients to keep notation simple. The reader with interest in bounds for estimated gradients is referred to Appendix F for the corresponding analysis.

In Section 4.1 four different layers of approximations were introduced that appear in the analysis of DynPG. Based on this we present the asymptotic global convergence of DynPG under the assumption of small enough optimization errors and rich enough parametrization class. In Section 4.2, we establish non-asymptotic global convergence rates of DynPG for the softmax parametrization under the assumption that exact gradients are available. We provide suitable choices of H , N h and η h such that DynPG achieves an error in the value functions of at most ϵ . Proofs of all results are deferred to Appendix D.

## 4.1 Error Decomposition and General Convergence

The analysis of DynPG employs four different layers of approximations, including 1) adopting parametrizations incapable of modeling the optimal policy perfectly (approximation error), 2) truncating the infinite time horizon to a finite horizon (truncation error), 3) utilizing a finite number of gradient updates to approximately solve each optimization problem (accumulated optimization error), and 4) applying the first policy of ˆ π h as a stationary policy in solving finite-horizon MDP (stationary policy error). We formally quantify these errors in the following proposition where the terms on the right-hand side of (6) correspond to the aforementioned errors, respectively.

Proposition 4.1. The overall error of DynPG after H iterations can be decomposed as follows

<!-- formula-not-decoded -->

The error decomposition gives clear insights into algorithm design, for which it is necessary to discuss the practical implications of the four summands.

1. To control the approximation error a rich enough policy parametrization is required.
2. To keep the truncation error small, DynPG should run at least H ≈ log(1 -γ ) log( γ ) rounds.
3. The third summand shows that approximation errors in earlier iterations are discounted more than approximation errors in later iterations. To achieve optimal training efficiency, we will thus require a geometrically decreasing optimization error across the iterations of DynPG. Recall the sample complexity discussion at the end of Section 3 to note that a trade-off between more accurate training and increasing samples required for estimating the gradient must be made to obtain the best performance of DynPG.
4. DynPG approximates the value function by truncation at a fixed time H and then replaces the optimal time-dependent policy by a stationary policy. As mentioned for PSDP, one could also apply the non-stationary policy ˆ π ∗ h to approximate the value function. This would cause the fourth error term to vanish. Thus, in what follows, we distinguish between the overall error in (6) and the value function error :

<!-- formula-not-decoded -->

It is important to note that the factor (1 -γ ) -1 in the stationary policy error causes an additional dependence on the effective horizon in the complexity bounds for softmax DynPG.

For the remainder of this section, we will proceed under the assumption of zero approximation error stemming from the parametrization. This condition generally holds true when the class of policies ( π θ ) can effectively approximate all deterministic policies, such as achieved through tabular softmax. Under this circumstance, it follows that T ∗ = sup θ T π θ and ∥ V ∗ ∞ -sup θ V π θ ∞ ∥ ∞ = 0 . Subsequently, we demonstrate that a certain reduction in the optimization error is adequate for achieving global convergence within the parametrized policy space.

Assumption 4.2. The class ( π θ ) has zero approximation error and there exists a positive sequence ϵ h such that

- ∑ H -1 t =0 γ H -h -1 ϵ h → 0 for H →∞ ,
- the policies obtained by DynPG satisfy ∥ T ∗ ( V ˆ π ∗ h h ) -V ˆ π ∗ h +1 h +1 ∥ ∞ ≤ ϵ h for all h ≥ 0 .

As an example for such a sequence one might think of ϵ h = cγ h for some c &gt; 0 . Under this assumption we prove convergence directly from the error decomposition in Proposition 4.1.

Corollary 4.3. Let Assumption 4.2 hold. Then, Algorithm 1 generates a sequence of non-stationary policies ˆ π ∗ H ∈ Π H that satisfy

<!-- formula-not-decoded -->

Furthermore, the overall error vanishes in the limit:

<!-- formula-not-decoded -->

Remark 4.4. The condition ∑ H -1 h =0 γ H -h -1 ϵ h → 0 for H →∞ implies that the optimization error must decrease with increasing h to guarantee convergence. Hence, training must become more precise over time. It is advisable to increase the number of gradient steps N h and decrease the step size η h .

## 4.2 Convergence Rates under Tabular Softmax Parametrization - Exact Gradients

In this section, we assume that all gradients are known explicitly, a strong assumption for practical use but standard for a rigorous analysis. The analysis is extended in Appendix F to estimated gradients.

One might ask whether Assumption 4.2 is reasonable and whether there are situations in which the condition on the algorithm holds. Indeed, we will show that this is the case for the softmax class of policies. More precisely, for any ϵ h &gt; 0 , we can specify a step size η h and a number of gradient steps N h such that Assumption 4.2 is satisfied. In a second step, we optimize H and the error sequence ( ϵ h ) to obtain the optimal sample complexity for DynPG under softmax parametrization, where we distinguish again between the value function error and the overall error.

Assumption 4.5. Suppose that µ ( s ) &gt; 0 for all s ∈ S such that ∥ ∥ 1 µ ∥ ∥ ∞ &lt; ∞ . Furthermore, the parametrization in DynPG ( π θ ) θ ∈ R |S||A| is chosen to be the tabular softmax parametrization introduced in Equation (3) . We further assume a uniform distribution over the action space as the initialization of θ 0 in Algorithm 1.

Theorem 4.6. Let Assumption 4.5 hold true and the gradient be accessed exactly. Let h ≥ 0 , ϵ h &gt; 0 and Λ = ˆ π ∗ h ∈ Π h be a collection of h arbitrary policies. Then, using step size η h = 1 -γ 2 R ∗ (1 -γ h +1 )

and gradient steps N h = 4 R ∗ (1 -γ h +1 ) |A| 2 (1 -γ ) ϵ h ∥ ∥ 1 µ ∥ ∥ ∞ in DynPG guarantees that the policy ˆ π ∗ h of iteration h achieves

<!-- formula-not-decoded -->

By Corollary 4.3, we obtain from Theorem 4.6 convergence for softmax DynPG by choosing a sufficient decreasing error sequence ( ϵ h ) .

Remark 4.7. The proof is adapted from [12, Lem. 3.4] to the discounted setting and is provided in Appendix D.2. Note that we will not directly apply [16, Thm. 4] with γ = 0 to prove this theorem for contextual bandits because the authors assumed rewards in [0 , 1] . Even when we make the same assumption, the problem horizon increases with h such that the contextual bandits we consider, V { π θ , Λ } h +1 , will have maximal reward greater then 1 after the first iteration. Therefore we provide a more general framework with bounded rewards in [ -R ∗ , R ∗ ] . Furthermore, the smoothness constant used in [16] can be improved, which results in tighter bounds.

In order to obtain complexity bounds from Theorem 4.6 for a given accuracy ϵ we optimize the total number of gradient steps ∑ H h =0 N h ( ϵ h ) with respect to H and ( ϵ h ) under the constraint that the overall error in (6) or the value function error in (7) is bounded by ϵ . We provide a detailed solution to the described constrained optimization problem in Appendix D.3. Parts of the proof are motivated by a similar optimization problem solved in [28, Sec. 3.2]. We summarize the complexity bounds for both error types in the following.

Theorem 4.8. [cf. Theorem D.6 and Theorem D.8 for detailed versions] Let Assumption 4.5 hold true and the gradient be accessed exactly. Choose ϵ &gt; 0 .

1. Overall error: We can specify H , ( N h ) H h =0 and ( η h ) H h =0 such that,

<!-- formula-not-decoded -->

accumulated gradient steps are required to achieve ∥ V ∗ ∞ -V ˆ π ∗ H ∞ ∥ ∞ ≤ ϵ .

2. Value function error: We can specify H , ( N h ) H -1 h =0 and ( η h ) H -1 h =0 such that,

<!-- formula-not-decoded -->

accumulated gradient steps are required to achieve ∥ V ∗ ∞ -V ˆ π ∗ H H ∥ ∞ ≤ ϵ .

Remark 4.9. We address each case individually and offer comprehensive versions of Theorem 4.8 in Theorem D.6 and Theorem D.8 delineating explicit selections for H , ( N h ) H h =0 , and ( η h ) H h =0 .

To obtain the convergence behavior of DynPG in terms of γ close to 1 , in Lemma D.9 we derive that log( γ -1 ) is asymptotically equivalent to the term (1 -γ ) . Combined with Theorem 4.8, we find that the required gradient steps for γ close to 1 behave like

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for the value function error. Compared to softmax PG, we observe an additional log( ϵ -1 ) factor in the convergence rate and future research could explore the possibility of eliminating the log-factor. In terms of γ , however, DynPG offers a resilient upper bound, which, in comparison to vanilla PG, remains polynomial in the effective horizon.

## 5 Breaking the lower bound example with DynPG

In [15] a lower bound example was given for which softmax PG takes exponentially many steps in the expected horizon (1 -γ ) -1 to converge. More precisely, it is shown that at least |S| 2 Ω((1 -γ ) -1 ) gradient steps are required to approximate the optimal value function with ϵ = 0 . 15 accuracy. The constructed MDP is designed such that the unknown constant C ( γ ) in the upper bound on softmax PG (see Table 1) is exponential in the effective time horizon (1 -γ ) -1 . To understand this behavior, we take a closer look at the definition of C ( γ ) := ( min s ∈S inf n ≥ 1 π θ n ( a ∗ ( s ) | s ) ) -1 in [16], where a ∗ ( s ) denotes the optimal action in state s . To ensure small C ( γ ) the probability of choosing the best action should not get close to 0 during training. But by construction in [15], finding the best action in state s decreases as long as the probability of choosing the best action a previous state is not close enough to 1. This results in the phenomenon that the gradient steps required to converge towards a ∗ ( s ) grows at least geometrically as s increases. However, using DP one can solve the MDP within |S| steps of exact value iteration [15, Lemma 1]. This already implies that DynPG easily circumvents this exponential convergence time by employing DP. As future actions are determined by the previously trained policies, DynPG can evaluate the MDP under a non-stationary policy during training and thereby avoid the above described phenomenon. The upper bound on the complexity of softmax DynPG provides a theoretical proof that the needed gradient steps scale at most polynomial in (1 -γ ) -1 !

## 6 Conclusion and Future Work

This theory paper contributes to the understanding of PG methods by directly including dynamic programming to gradient based policy search. To this end, we have introduced DynPG, an algorithm for the overall error and

that directly combines DP and gradient ascent to solve γ -discounted infinite horizon MDPs. We provide mathematically rigorous performance estimates (for exact gradients in the main text, estimated gradients in Appendix F) in simplified situations which are typical for theory papers on PG methods. The main strength of the algorithm, in contrast to vanilla PG, is to provably avoid exponential γ -dependence. We have provided theoretical evidence that dynamic programming, combined with PG methods, can address well-known issues inherent to vanilla PG. This opens up a range of further avenues for future research, such as:

- Investigate whether the complexity bounds in the exact gradient setting are tight (numerically or theoretically) and explore the possibility of eliminating the log-factor.
- Analyze the variant DynAC from Section E. [14] present a sample complexity analysis of AC methods. Combining their results to control the critic error with the SGD analysis could result in convergence rates to achieve a small value function error in DynAC.
- Perform a simulation study of DynPG and DynAC on standard tabular environments and explore how numerical instability issues in standard PG can be overcome.
- Combine DynPG and DynAC with deep RL implementations for PG algorithms such as PPO, TRPO.

## Acknowledgements

The first author SK thankfully acknowledges the funding support by the Hanns-Seidel-Stiftung e.V. and is grateful to the DFG RTG1953 'Statistical Modeling of Complex Systems and Processes' for funding this research. The research of SK was also supported by the research project 'SURF - Smart, user-centered regional flexibility platform for grid and market' funded by the Federal Ministry for Economic Affairs and Energy [03EI4092D]. The research of the second and third authors (XZ and TB) was sponsored in part by the US Army Research Office and was accomplished under Grant Number W911NF-24-1-0085.

## References

- [1] Alekh Agarwal, Sham M. Kakade, Jason D. Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research , 22(98):1-76, 2021.
- [2] Mohammad Gheshlaghi Azar, Vicenç Gómez, and Hilbert J Kappen. Dynamic policy programming. The Journal of Machine Learning Research , 13(1):3207-3245, 2012.
- [3] J. Bagnell, Sham M Kakade, Jeff Schneider, and Andrew Ng. Policy search by dynamic programming. In S. Thrun, L. Saul, and B. Schölkopf, editors, Advances in Neural Information Processing Systems , volume 16. MIT Press, 2003.
- [4] Amir Beck. First-Order Methods in Optimization . Society for Industrial and Applied Mathematics, Philadelphia, PA, 2017.
- [5] Dimitri Bertsekas and John N Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1996.
- [6] Dimitri P Bertsekas. Dynamic Programming and Optimal Control. 2 . Athena Scientific, Belmont, Mass, 2. ed. edition, 2001.
- [7] Yuhao Ding, Junzi Zhang, and Javad Lavaei. Beyond exact gradients: Convergence of stochastic soft-max policy gradient methods with entropy regularization. arXiv Preprint , arXiv:2110.10117, 2022.
- [8] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 1587-1596. PMLR, 7 2018.
- [9] Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the Nineteenth International Conference on Machine Learning , pages 267-274, 2002.
- [10] Sham M Kakade. A natural policy gradient. In Advances in Neural Information Processing Systems , volume 14. MIT Press, 2001.

- [11] Sham M. Kakade. On the sample complexity of reinforcement learning. PhD thesis, University College London, 2003.
- [12] Sara Klein, Simon Weissmann, and Leif Döring. Beyond stationarity: Convergence analysis of stochastic softmax policy gradient methods. In The Twelfth International Conference on Learning Representations , 2024.
- [13] Vijay Konda and John Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems , volume 12. MIT Press, 1999.
- [14] Harshat Kumar, Alec Koppel, and Alejandro Ribeiro. On the sample complexity of actor-critic method for reinforcement learning with function approximation. Machine Learning , 112(7):2433-2467, 2023.
- [15] Gen Li, Yuting Wei, Yuejie Chi, and Yuxin Chen. Softmax policy gradient methods can take exponential time to converge. Mathematical Programming , 201(1):707-802, 2023.
- [16] Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, and Dale Schuurmans. On the global convergence rates of softmax policy gradient methods. In Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 6820-6829. PMLR, 13-18 Jul 2020.
- [17] Matteo Papini, Damiano Binaghi, Giuseppe Canonaco, Matteo Pirotta, and Marcello Restelli. Stochastic variance-reduced policy gradient. In Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 4026-4035. PMLR, 10-15 Jul 2018.
- [18] Matteo Pirotta, Marcello Restelli, and Luca Bascetta. Adaptive step-size for policy gradient methods. In Advances in Neural Information Processing Systems , volume 26. Curran Associates, Inc., 2013.
- [19] B.T. Polyak. Gradient methods for the minimisation of functionals. USSR Computational Mathematics and Mathematical Physics , 3(4):864-878, 1963.
- [20] M.L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons, 2005.
- [21] Bruno Scherrer. Approximate policy iteration schemes: A comparison. In Eric P. Xing and Tony Jebara, editors, Proceedings of the 31st International Conference on Machine Learning , volume 32 of Proceedings of Machine Learning Research , pages 1314-1322, Bejing, China, 22-24 Jun 2014. PMLR.
- [22] Bruno Scherrer, Mohammad Ghavamzadeh, Victor Gabillon, Boris Lesner, and Matthieu Geist. Approximate modified policy iteration and its application to the game of tetris. J. Mach. Learn. Res. , 16(49):1629-1676, 2015.
- [23] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning , volume 37 of Proceedings of Machine Learning Research , pages 1889-1897, Lille, France, 07-09 Jul 2015. PMLR.
- [24] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . MIT press, 2018.
- [25] Richard S Sutton, Hamid Maei, and Csaba Szepesvári. A convergent o(n) temporal-difference algorithm for off-policy learning with linear function approximation. In D. Koller, D. Schuurmans, Y. Bengio, and L. Bottou, editors, Advances in Neural Information Processing Systems , volume 21. Curran Associates, Inc., 2008.
- [26] Richard S Sutton, Hamid Reza Maei, Doina Precup, Shalabh Bhatnagar, David Silver, Csaba Szepesvári, and Eric Wiewiora. Fast gradient-descent methods for temporal-difference learning with linear function approximation. In Proceedings of the 26th annual international conference on machine learning , pages 993-1000, 2009.
- [27] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems , volume 12. MIT Press, 1999.
- [28] Simon Weissmann, Ashia Wilson, and Jakob Zech. Multilevel optimization for inverse problems. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pages 5489-5524. PMLR, 02-05 Jul 2022.
- [29] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8(3):229-256, 1992.

- [30] Lin Xiao. On the convergence rates of policy gradient methods. Journal of Machine Learning Research , 23(282):1-36, 2022.
- [31] Rui Yuan, Robert M. Gower, and Alessandro Lazaric. A general sample complexity analysis of vanilla policy gradient. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics , volume 151 of Proceedings of Machine Learning Research , pages 3332-3380. PMLR, 28-30 Mar 2022.
- [32] Kaiqing Zhang, Alec Koppel, Hao Zhu, and Tamer Ba¸ sar. Global convergence of policy gradient methods to (almost) locally optimal policies. SIAM Journal on Control and Optimization , 58(6):3586-3612, 2020.
- [33] Xiangyuan Zhang and Tamer Ba¸ sar. Revisiting LQR control from the perspective of receding-horizon policy gradient. IEEE Control Systems Letters , 7:1664-1669, 2023.
- [34] Xiangyuan Zhang, Bin Hu, and Tamer Ba¸ sar. Learning the Kalman filter with fine-grained sample complexity. In 2023 American Control Conference (ACC) , pages 4549-4554. IEEE, 2023.
- [35] Xiangyuan Zhang, Saviz Mowlavi, Mouhacine Benosman, and Tamer Ba¸ sar. Global convergence of receding-horizon policy search in learning estimator designs. arXiv Preprint , arXiv:2309.04831, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction summarizes the theoretical findings in this paper; all claims are proven.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in Section 3.

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

Justification: Full proofs of all results are provided in the appendix.

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

Justification: Experimental setup is explained in Section B.

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

Justification: Code to reproduce the experiments is available in an anonymous repository, see Section B.

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

Justification: Experimental setup and all parameter details are explained in Section B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experimental results are numerically stable and you achieve the same results in multiple runs. Behavior for different parameter settings is discussed in Section B.

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

Justification: Toy example which can be run without specific computer resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We agree with the NeurIPS Code of Ethics.

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

## A Notations

| S                                           | state space                                                          |
|---------------------------------------------|----------------------------------------------------------------------|
| A                                           | action space                                                         |
| p ( ·&#124; s,a )                           | transition kernel in ( s,a ) ; in ∆( S )                             |
| r ( s,a )                                   | expected reward in ( s,a )                                           |
| γ ∈ (0 , 1)                                 | discount factor                                                      |
| R ∗ > 0                                     | maximal absolute value of rewards                                    |
| ∥ x ∥ ∞ = max i =1 ,...,d &#124; x i &#124; | supremum norm for x ∈ R d                                            |
| π : S → ∆( A ) ∈ Π                          | stationary policy (and stationary policy set)                        |
| π h = { π h - 1 , . . .,π 0 } ∈ Π h         | h -step non-stationary policy (and the set of all such policies)     |
| µ                                           | initial state distribution                                           |
| V π h h ( µ )                               | h -step discounted value function; policy π h , start distribution µ |
| V ∗ ∞ ( µ )                                 | optimal discounted value function                                    |
| π ∗                                         | optimal policy s.t. V ∗ ∞ ( µ ) = V π ∗ ∞ ( µ )                      |
| V ∗ h ( µ )                                 | optimal discounted h -step reward function                           |
| π ∗ h                                       | optimal h -step policy s.t. V ∗ h ( µ ) = V π ∗ h ∞ ( µ )            |
| V π 0 ≡ 0 and π 0 = {}                      | conventions used in the manuscript                                   |
| Q π h h ( s,a )                             | h -step Q-function                                                   |
| T π γ                                       | one-step Bellman operator under policy π                             |
| T ∗ γ                                       | one-step Bellman optimality operator                                 |
| π V                                         | stationary greedy policy after V ∈ R &#124;S&#124;                   |
| Q V ( s,a )                                 | Q -matrix obtained from V ∈ R &#124;S&#124;                          |
| η h , N h                                   | learning rate, number of training steps in epoch h                   |
| Λ                                           | set of trained policies in the algorithm                             |
| ˆ π ∗ h                                     | trained policy by DynPG in epoch h                                   |
| ˆ π ∗ h                                     | set of trained policy by DynPG from h - 1 to 0                       |
| π θ                                         | stationary parametrized policy                                       |
| ( π h ) ∞                                   | apply π h in a loop for the infinite horizon problem                 |

## B Details of the Numerical Example

The example is an extension of [24, Example 6.7], which has been used to compare different variants of Q-learning algorithms, as it suffers from overestimation of the Q-values. Since the overestimation problem in Q-learning and the committal behavior problem in policy gradient are closely related [8], we decided to use this example. The example is certainly artificial, as the number of actions and the rewards in the MDP are chosen to trap policy gradient from convergence. If we vary the MDP parameters from the current setting, DynPG consistently performs better, but the advantage over vanilla PG will become less significant. The code is available at https: //github.com/Sara-Klein/StructureMatters-DynPG .

The MDP is defined as follows:

- The state space is given by S := { 0 , . . . , 6 } ; States 0 , 3 , 6 are the terminal states and states 1 , 2 , 4 , 5 are the initial states. We sample s 0 uniformly from { 1 , 2 , 4 , 5 } .
- The action space is given by A := { 0 , . . . , 299 } .
- The state transitions and state-dependent actions are visualized in Figure 3. Each node represents a state, with squared ones being terminal states and elliptical ones being initial states. Each arrow represents an action that deterministically transits from one state to another. From state 1 to state 0 , there is a total of 300 possible actions, succinctly visualized using the dots.
- Taking any a ∈ A from state 1 reaches state 0 and receives a reward of r (1 , a ) ∼ N ( -0 . 3 , 10) .
- Taking any of the 5 possible actions from state 4 reaches state 5 and receives the reward of r (4 , a ) ∼ N (1 . 25 , 1 . 25) .
- Taking any of the 5 possible actions from state 5 reaches state 6 and receives the reward of r (5 , a ) ∼ N (1 . 25 , 1 . 25) .
- Taking any of the 3 possible actions from state 2 receives the reward of r (2 , a ) = 0 .

Figure 3: Visualization of the MDP state transitions.

<!-- image -->

Experimental setup: We evaluated the performance of (stochastic) vanilla PG and (stochastic) DynPG under two different discount factors, γ = 0 . 9 and γ = 0 . 99 . We used the tabular softmax parametrization studied in the convergence analysis for both algorithms. In DynPG, we used the 1 -batch Monte-Carlo estimator to sample the gradient according to Theorem A.3. In vanilla PG, we chose the classical REINFORCE 1 -batch estimator with truncation horizon 3 , such that the estimator is also unbiased due to the episodic setting (the maximum episode length in our example is 3 ).

In DynPG, we chose the step size η h and number of training steps N h according to Theorem D.6 and Theorem D.8, and only fine-tuned the constants 2 and 45 :

<!-- formula-not-decoded -->

We want to emphasize that the choices of η h and N h consistently perform well under different choices of γ (see therefore a second convergence picture with γ = 0 . 9 in Figure 4), which underscores that the algorithmic parameters developed in our theory also provide good guidance in practice. For a fair comparison, we fine-tuned η = 2 1 -γ 1 -γ 6 for stochastic vanilla PG, which is much larger than the pessimistic η = c ∗ (1 -γ ) 3 suggested in [16].

Figure 4: Success probability of achieving the sub-optimality gap of ϵ = 0 . 01 in the overall error.

<!-- image -->

## C Auxiliary Results

## C.1 Q-function, Advantage Function and Greedy Policy

First, we define the state-action value of π h -1 ∈ Π h -1 at any ( s, a ) ∈ S × A as

<!-- formula-not-decoded -->

For h ≥ 1 and π h ∈ Π h we define the advantage function

<!-- formula-not-decoded -->

For V ∈ R |S| , a greedy policy π V chooses the action that maximizes the Q -matrix 1 :

<!-- formula-not-decoded -->

By the definitions of the Bellman operators, it holds that a policy π is greedy with respect to V if and only if T π V = T ∗ V [6].

## C.2 Dynamic Programming - Finite Time Approximation

Lemma C.1. Let V 0 ≡ 0 . For any h ≥ 1 , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1 When multiple actions are equally optimal, π V selects an arbitrary one among them.

Proof. C.f. [6, Prop. 1.2.1]

The first claim, follows directly from the definition of the Bellman optimality operator

<!-- formula-not-decoded -->

For the second part, we fix h ≥ 0 . Recall that π ∗ ∈ Π denotes a stationary optimal policy for the infinite-time problem and π ∗ h ∈ Π h an optimal non-stationary policy.

We divide the proof of the second claim into two cases.

Case 1: Assume that V ∗ ∞ ( s ) -V ∗ h ( s ) ≤ 0 for all s ∈ S . Note that V ∗ ∞ ( s ) ≥ V ( π ∗ h ) ∞ ∞ ( s ) , where ( π ∗ h ) ∞ denotes that we apply the finite time policy π ∗ h in a loop for the infinite time problem. We have

<!-- formula-not-decoded -->

where R ∗ bounds the absolute value of rewards.

Case 2: Assume that V ∗ ( s ) -V ∗ h ( s ) &gt; 0 for all s ∈ S . Note that V π ∗ h h ( s ) ≥ V π ∗ h ( s ) due to the optimality of π ∗ h over the finite-time horizon. Further, by the definition of V ∗ ∞ ( s ) = V π ∗ ∞ ( s ) , we have

<!-- formula-not-decoded -->

Hence, we arrive at

<!-- formula-not-decoded -->

## C.3 Error bound for stationary policy

The following Lemma is inspired by error bounds presented in [6, Sec. 1.3]. The purpose of this result is to establish validation of applying the last policy trained in DynPG as stationary policy.

Lemma C.2. For any V ∈ R |S| and policy π ∈ Π it holds that

<!-- formula-not-decoded -->

Proof. Consider any state s ∈ S . Since V π ∞ is the unique fixed point of the operator T π , we have

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

for any s ∈ S . We define the mappings s ↦→ g π ( s ) = T π ( V )( s ) -V ( s ) and s ↦→ J π ( s ) = V π ∞ ( s ) -V ( s ) , s ∈ S . Then, the above equation simplifies to

<!-- formula-not-decoded -->

By definition J π satisfies

<!-- formula-not-decoded -->

for all s ∈ S , and therefore is a solution of the Bellman equation with an auxiliary reward function ( s, a ) ↦→ ˜ r ( s, a ) = g π ( s ) . Note that this reward function is also bounded and by the uniqueness of the solution of the Bellman equation it has to hold that

<!-- formula-not-decoded -->

Define β = min s ∈S g π ( s ) and β = max s ∈S g π ( s ) . Then,

<!-- formula-not-decoded -->

It follows that for any s ∈ S ,

<!-- formula-not-decoded -->

By definition of g π and J π this yields the claim.

## C.4 Policy Gradient Theorem for DynPG

Theorem C.3. Suppose that π h ∈ Π h is fixed for any h ≥ 0 , with the convention π 0 = ∅ . Then, for any differentiable parametrized family, say ( π θ ) θ ∈ R d (i.e. θ ↦→ π θ ( s, a ) , θ ∈ R d is differentiable for all ( s, a ) ∈ S × A ), it holds that

<!-- formula-not-decoded -->

Proof. The proof can be found in [12, Thm A.6]. However, we will need an index shift and additionally to consider the discount factor γ . An h -step trajectory τ h = ( s 0 , a 0 , . . . , s h -1 , a h -1 ) under policy { π θ , π h } and initial state distribution δ s occurs with probability

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the log trick yields that

<!-- formula-not-decoded -->

Let W h be the set of all trajectories from 0 to h -1 . Then, the set W h is finite due to the assumption that state and action space are finite. For s ∈ S we have

<!-- formula-not-decoded -->

## C.5 Performance Difference Lemma

Lemma C.4. Let π h +1 , π ′ h +1 ∈ Π h +1 . Then, it holds that

<!-- formula-not-decoded -->

(ii) If both policies only differ in the first policy, i.e. π h = π ′ h , then the above equation simplifies to

<!-- formula-not-decoded -->

Proof. A similar proof can be found in [12, Thm A.6]. However, we will need an index shift and additionally to consider the discount factor γ . First, let π h +1 , π ′ h +1 ∈ Π h +1 be two arbitrary policies. We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the fifth equation we used the convention V 0 ≡ 0 , in the sixth equation the definition of the Q-function, and in the last equation the definition of the advantage function. Second, suppose that π h +1 and π ′ h +1 agree on all policies besides π h , i.e. π h = π ′ h . Then, for any t &gt; 0 , it holds that

<!-- formula-not-decoded -->

This proves the claim.

## C.6 Convergence under Gradient Domination

Lemma C.5. Let f : R d → R be L -smooth (i.e. L -Lipschitz continuity of the gradient), f ∗ = sup x f ( x ) &lt; ∞ . Denote the gradient ascent procedure by x n +1 = x n + α ∇ f ( x n ) for some x 1 ∈ R d and α = 1 L and assume that f satisfies the following gradient domination property for some b &gt; 0 along the gradient trajectory,

<!-- formula-not-decoded -->

Then the convergence rate

<!-- formula-not-decoded -->

holds true if f ∗ -f ( x 1 ) ≤ 2 αb 2 .

Proof. We apply the descent lemma of smooth functions [4, Lem 5.7] on -f and obtain

<!-- formula-not-decoded -->

Now for gradient ascent updates we have that

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

Exploiting the gradient domination along the gradient trajectory results in

<!-- formula-not-decoded -->

As α = 1 L , we have

<!-- formula-not-decoded -->

When f ∗ -f ( x 1 ) ≤ 2 αb 2 , then f ∗ -f ( x n ) ≤ 2 αb 2 n by [12, Lem. B.7].

## D Proofs of Section 4

## D.1 Proof for Section 4.1

Proof for Proposition 4.1. Recall the claim:

<!-- formula-not-decoded -->

The first inequality follows directly from the triangle inequality of the supremum norm. Note here that we cannot simplify the first error further as it will depend on the chosen parametrization. Before we prove the second inequality, note that

<!-- formula-not-decoded -->

We treat the second, third and fourth terms separately.

For the second term , we prove that ∥ ∥ ∥ sup θ V π θ ∞ -sup θ 0 ,...,θ H -1 V { π θ H -1 ,...π θ 0 } H ∥ ∥ ∥ ∞ ≤ γ H R ∗ 1 -γ similar to Lemma C.1. Let s ∈ S be arbitrary but fixed. If sup θ V π θ ∞ ( s ) -sup θ 0 ,...,θ H -1 V { π θ H -1 ,...π θ 0 } H ( s ) ≥ 0 , then

<!-- formula-not-decoded -->

where we used (11) in the second inequality. On the other hand, if sup θ V π θ ∞ ( s ) -sup θ 0 ,...,θ H -1 V { π θ H -1 ,...π θ 0 } H ( s ) &lt; 0 , then

<!-- formula-not-decoded -->

Collecting these together, we obtain that ∥ ∥ ∥ sup θ V π θ ∞ -sup θ 0 ,...,θ H -1 V { π θ H -1 ,...π θ 0 } H ∥ ∥ ∥ ∞ ≤ γ H R ∗ 1 -γ , since s ∈ S was chosen arbitrary.

For the third term , we show that

<!-- formula-not-decoded -->

holds for all H ≥ 1 by induction. For H = 1 we have by (5) that

<!-- formula-not-decoded -->

with the convention V 0 ≡ 0 and ˆ π ∗ 0 = ∅ . So assume that Equation (12) holds for some H ≥ 1 ; then for H +1 we have

<!-- formula-not-decoded -->

where we used (5), as well as (11), and the induction assumption. This yields the desired claim (12) for all H ≥ 1 .

For the fourth error term , we have to deal with the error of applying the final policy as stationary policy. We use Lemma C.2 to arrive at

<!-- formula-not-decoded -->

where we used again (5) in the last line.

Proof of Corollary 4.3. Adjusting (7) to zero approximation error and exploiting Assumption 4.2 we obtain

<!-- formula-not-decoded -->

For the second part of the theorem note that V π ∗ H H converges in the supremum norm to V ∗ ∞ by the first part. This implies that V π ∗ H H is a Cauchy sequence with respect to the supremum norm, i.e. ∥ V π ∗ H +1 H +1 -V π ∗ H H ∥ ∞ → 0 for H →∞ . The claim follows directly from Proposition 4.1.

## D.2 Proof of Theorem 4.6

Before we prove Theorem 4.6, we will discuss the similarities and differences between DynPG and the finite-time Dynamic Policy Gradient (FT-DynPG) Algorithm proposed in [12].

1. FT-DynPG has a prefixed time horizon H and trains policy backwards in time. In contrast, DynPG adds arbitrarily many policies in the beginning and can therefore be applied without prefixed H .
2. Given a fixed time horizon H , FT-DynPG returns a non-stationary policy ˆ π ∗ H for an H -step MDP, but without discounting γ = 1 . When DynPG is run for the same fixed number of iterations H , then FT-DynPG and DynPG only differ by the discount factor: The value functions V h,H defined in [12] for FT-DynPG are by the strong Markov property equivalent to V 0 ,H -h via index shifting. The function V 0 ,H -h differs from our H -h -step value function V H -h defined in (1) only by the discount factor γ .

In order to prove Theorem 4.6, we have to adapt the proof for [12, Lem. 3.4] to an additional discount factor. Note, that we cannot use [16, Thm. 2] for bandits, as we consider contextual bandits. Further, we cannot use [16, Thm. 4] with γ = 0 as they just consider positive rewards in [0 , 1] which is inconvenient for our contextual bandit setting where the maximal rewards grows when we add a new time-epoch in the beginning.

Therefore, we will adapt the proofs in [16, 12] and start with deriving the smoothness of our objective functions.

Lemma D.1. Suppose π θ = softmax ( θ ) . Then, for arbitrary π h ∈ Π h and µ ∈ ∆( S ) , the function θ ↦→ V { π θ , π h } h +1 ( µ ) is L h -smooth with L h = 2 R ∗ (1 -γ h +1 ) (1 -γ ) .

Proof. The proof is similar to [12, Lem. B.8]. Note that we can interpret V { π θ , π h } h +1 ( µ ) as a contextual bandit problem, i.e. a discounted (infinite-time) MDP with discount factor 0 . The reward of the contextual bandit problem is almost surely bounded in [ -1 -γ h +1 1 -γ R ∗ , 1 -γ h +1 1 -γ R ∗ ] , because

<!-- formula-not-decoded -->

We can apply [31, Lem. 4.4 and Lem. 4.8] with R max = 1 -γ h +1 1 -γ R ∗ , G 2 = 1 -1 |A| ≤ 1 and F = 1 to obtain the smoothness constant L h = 2 R ∗ (1 -γ h +1 ) (1 -γ ) .

Second, we obtain the following gradient domination property.

Lemma D.2. Under Assumption 4.5 it holds for any π h ∈ Π h that

<!-- formula-not-decoded -->

where a ∗ ( s ) denotes the (unique) action taken after the greedy policy π V π h h .

Remark D.3. Without loss of generality, we assume that the action a ∗ ( s ) is unique for any fixed future policy π h , otherwise one can consider min a optimal action in s π θ ( a | s ) .

Proof. First note from Theorem C.3 that under the tabular softmax parametrization we have

<!-- formula-not-decoded -->

Hence, with the derivative of the softmax function

<!-- formula-not-decoded -->

it holds that

<!-- formula-not-decoded -->

We deduce from Lemma C.4, Equation (10), that

<!-- formula-not-decoded -->

Finally, the rest of the proof is derived as in [12, Lem. B.9]

<!-- formula-not-decoded -->

where a ∗ ( s ) denotes the action taken after the greedy policy π V π h h (c.f. Remark D.3).

The next step is to show that the term min s ∈S π θ ( a ∗ ( s ) | s ) can be bounded (uniformly in s ∈ S ) from below by 1 |A| along the gradient ascent trajectory, when softmax is initialized uniformly.

Lemma D.4. Let Assumption 4.2 hold and denote by ( θ n ) n ≥ 0 the gradient ascent sequence in epoch h ≥ 0 of DynPG under the fixed future policy ˆ π ∗ h ∈ Π h . Suppose further that η h = 1 -γ 2 R ∗ (1 -γ h +1 ) is chosen as in Theorem 4.6. Then, for any s ∈ S , it holds that

<!-- formula-not-decoded -->

where a ∗ ( s ) denotes the (unique) action taken after the greedy policy π V ˆ π ∗ h h (see Remark D.3).

Proof. The proof is adapted from [16, Lem. 5] and [12, Lem. B.10, Prop. 3.3]. First, we define the sets

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim 1: It holds that θ n ∈ R 2 = ⇒ θ n +1 ∈ R 2 .

To prove this claim, we first deduce from (13) that

<!-- formula-not-decoded -->

̸

Let a = a ∗ ( s ) be arbitrary. We consider two cases to proof claim 1:

̸

1. Suppose that π θ n ( a ∗ ( s ) | s ) ≥ π θ n ( a | s ) for any a = a ∗ ( s ) . Then, is holds that θ n ( s, a ∗ ( s )) ≥ θ n ( s, a ) by the definition of softmax. Next, as θ n ∈ R ∗ 2 , we derive

<!-- formula-not-decoded -->

̸

Thus, by the definition of the softmax function, π θ n +1 ( a ∗ ( s ) | s ) ≥ π θ n +1 ( a | s ) for any a = a ∗ ( s ) . Further, because a ∗ ( s ) is the greedy action, we arrive at

<!-- formula-not-decoded -->

such that θ n +1 ∈ R 2 ( s ) by (14).

2. Suppose that π θ n ( a ∗ ( s ) | s ) &lt; π θ n ( a | s ) for any a = a ∗ ( s ) . Then, θ n ( s, a ∗ ( s )) -θ n ( s, a ) &lt; 0 by definition of the softmax function. As θ n ∈ R 2 ( s ) , it holds that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Thus, we obtain

<!-- formula-not-decoded -->

By the ascent lemma for smooth functions [16, Lem. 18] it follows monotonicity in the objective function (due to small enough step size η h = 1 L h with L h the smoothness constant

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We rearrange (14) and obtain that θ ∈ R 2 ( s ) is equivalent to

<!-- formula-not-decoded -->

̸

̸

We deduce by θ n ∈ R 2 ( s ) that

<!-- formula-not-decoded -->

and thus θ n +1 ∈ R 2 ( s ) .

This proves claim 1.

Claim 2: If θ n ∈ R 2 ( s ) , then it holds that π θ n +1 ( a ∗ ( s ) | s ) ≥ π θ n ( a ∗ ( s ) | s ) . Line by line as in Claim 2 of [12, Lem. B.10]. Claim 3: It holds that θ n ∈ R 1 ( s ) = ⇒ θ n ∈ R 2 ( s ) . Let θ n ∈ R 1 ( s ) . As a ∗ ( s ) is optimal we have for any a = a ∗ ( s ) that

̸

<!-- formula-not-decoded -->

Further by θ n ∈ R 1 ( s ) it holds

<!-- formula-not-decoded -->

Hence, by (13), we deduce that θ n ∈ R 2 ( s ) .

## Conclusion of the proof by combining claim 1, claim 2 and claim 3:

Since θ 0 is initialized such that softmax is the uniform distribution, we have that θ 0 ∈ R 1 ( s ) for all s ∈ S . By claim 3, we have that θ 0 ∈ R 2 ( s ) and by claim 1 it follows that θ n ∈ R 2 ( s ) for all n ≥ 0 and all s ∈ S . Finally, by claim 2, it follows that min n ≥ 0 π θ n ( s, a ∗ ( s )) = π θ 0 ( s, a ∗ ( s )) = 1 |A| for any s ∈ S .

We finally we collect all preliminary results to prove Theorem 4.6.

Proof of Theorem 4.6. First, note that the tabular softmax parametrization can approximate any deterministic policy arbitrarily well. As optimal policies in finite horizon MDPs are deterministic, we have that sup θ ∈ R |S||A| V { π θ , ˆ π ∗ h } h +1 ( s ) = T ∗ ( V ˆ π ∗ h h )( s ) for all s ∈ S (zero approximation error induced by the parametrization). Moreover, for any s ∈ S

<!-- formula-not-decoded -->

For any θ ∈ R d it holds that

<!-- formula-not-decoded -->

Moreover, by Lemma D.1 the function θ ↦→ V { π θ , ˆ π ∗ h } h +1 ( µ ) is L h -smooth and fulfills the gradient domination property along the gradient ascent steps with b = 1 |A| (Lemma D.2 and Lemma D.4). Hence, we can apply Lemma C.5 with b = 1 |A| , α = η h = 1 -γ 2 R ∗ (1 -γ h +1 ) and n = N h , such that

<!-- formula-not-decoded -->

By the choice of N h we deduce for any s ∈ S

<!-- formula-not-decoded -->

which proves the claim.

## D.3 Proof of Theorem 4.8

We will treat the two cases of value function error and overall error separately and start with the value function error.

Value function error. Note that under the tabular softmax parametrization we have zero approximation error and the can upper bound in (7) simplifies to

<!-- formula-not-decoded -->

where ϵ h are upper bounds on one step optimization errors ∥ ∥ sup θ ∈ R d T π θ ( V ˆ π ∗ h h ) -V ˆ π ∗ h +1 h +1 ∥ ∥ ∞ according to Theorem 4.6.

In order to minimize the accumulated number of gradient steps, we have to solve the optimization problem:

<!-- formula-not-decoded -->

In [28, Sec. 3.2] a similar optimization problem was considered and it was shown that asymptotically (as ϵ → 0 ) it suffices to bound both error terms in the constraint by ϵ 2 . The first condition, i.e. γ H R ∗ 1 -γ ≤ ϵ 2 , leads to the criterion H ≥ log γ ( (1 -γ ) ϵ 2 R ∗ ) . To minimize the number of gradient steps we fix H = ⌈ log γ ( (1 -γ ) ϵ 2 R ∗ )⌉ . It remains to solve the following optimization problem

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We provide a solution of the optimization problem (16) by solving the following more general one:

Lemma D.5. Fix H &gt; 0 . Let ( a h ) H -1 h =0 and ( b h ) H -1 h =0 be strictly positive sequences. For any d &gt; 0 the optimization problem

<!-- formula-not-decoded -->

is optimally solved for c h = C d,H ( b h a h ) -1 2 , with C H,d = d ( ∑ H -1 h =0 ( a h b h ) 1 2 ) -1 .

Hence, the minimum of the optimization problem is given by 1 d ( ∑ H -1 h =0 ( a h b h ) 1 2 ) 2 .

Proof. The result and the proof is inspired by [28, Lem. 3.8].

We solve the constrained optimization problem by employing the Lagrange method, i.e

<!-- formula-not-decoded -->

The first order conditions are given by

<!-- formula-not-decoded -->

We deduce from the first equations, that there exists a constant C H,d such that c h = C H,d ( b h a h ) -1 2 for all h = 0 , . . . , H -1 . Using this in the second equation we can solve for the constant

<!-- formula-not-decoded -->

Using the minima c h = C H,d ( b h a h ) -1 2 in the optimization function ∑ H -1 h =0 a h c -1 h , we obtain the minimum

<!-- formula-not-decoded -->

Finally, we are ready to state the detailed version of Theorem 4.8 for the value function error.

Theorem D.6. [Detailed version of Theorem 4.8 (2.)] Let Assumption 4.5 hold true and the gradient be accessed exactly. Choose ϵ &gt; 0 and set

<!-- formula-not-decoded -->

for h = 0 , . . . H -1 . Then, the non-stationary policy ˆ π ∗ H obtained by DynPG under exact gradients achieves ∥ V ∗ ∞ -V ˆ π ∗ H H ∥ ∞ ≤ ϵ . The total number of gradient steps are given by

<!-- formula-not-decoded -->

Proof. First, note that the choice of ϵ h in the theorem is the solution of the optimization problem

<!-- formula-not-decoded -->

Therefore, choose a h = (1 -γ h +1 ) , b h = γ H -h -1 , c h = ϵ h and d = ϵ 2 in Lemma D.5, where we excluded the constant 4 R ∗ |A| 2 (1 -γ ) ∥ ∥ ∥ 1 µ ∥ ∥ ∥ ∞ . Then,

<!-- formula-not-decoded -->

is the optimal solution to this problem. For H = ⌈ log( (1 -γ ) ϵ 2 R ∗ ) log( γ ) ⌉ we have that γ H R ∗ 1 -γ ≤ ϵ 2 . Using these ( ϵ h ) H -1 h =0 in Theorem 4.6 results in a value function error for DynPG bounded by

<!-- formula-not-decoded -->

For the optimal complexity bound we ignore the Gauss-brackets in N h and derive

<!-- formula-not-decoded -->

where we used Jensen's inequality in the last step. Further it holds that

<!-- formula-not-decoded -->

Finally, note that we can rewrite H to be

<!-- formula-not-decoded -->

Overall error. To deal with the overall error, we first derive the following upper bound.

Lemma D.7. Assume zero approximation error, i.e. T ∗ = sup θ T π θ . Further, assume bounded optimization errors ∥ T ∗ ( V ˆ π ∗ h h ) -T ˆ π ∗ h ( V ˆ π ∗ h h ) ∥ ∞ ≤ ϵ h for all h = 0 , . . . , H . If ϵ H ≤ (1 -γ ) ∑ H -1 h =0 γ H -h -1 ϵ h , then the overall error of DynPG is bounded by

<!-- formula-not-decoded -->

Proof. First, we have by triangle inequality

<!-- formula-not-decoded -->

By Proposition 4.1 we obtain, under zero approximation error, that

<!-- formula-not-decoded -->

By the assumption ϵ H ≤ (1 -γ ) ∑ H -1 h =0 γ H -h -1 ϵ h , it holds further that

<!-- formula-not-decoded -->

We obtain

<!-- formula-not-decoded -->

We deduce for the overall error (by Proposition 4.1 and under zero approximation error) that

<!-- formula-not-decoded -->

It is important to notice that the overall error is upper bounded by the same error terms, γ H R ∗ 1 -γ + ∑ H -1 h =0 γ H -h -1 ϵ h , as the value function error (under zero approximation error) up to the constant 3 1 -γ . Thus, we can obtain the result for the overall error by substituting ϵ with (1 -γ ) ϵ 3 .

Theorem D.8. [Detailed version of Theorem 4.8 (1.)]

Let Assumption 4.5 hold true and the gradient be accessed exactly. Choose ϵ &gt; 0 and set

<!-- formula-not-decoded -->

Then, the stationary policy ˆ π ∗ H obtained by DynPG under exact gradients achieves ∥ V ∗ ∞ -V ˆ π ∗ H ∞ ∥ ∞ ≤ ϵ . The total number of gradient steps are given by

<!-- formula-not-decoded -->

Proof. The optimization procedure is the same as for the value function error by substituting ϵ with (1 -γ ) ϵ 3 . Moreover, note that

<!-- formula-not-decoded -->

Thus, ϵ H ≤ (1 -γ ) ϵ 6 is sufficient for Lemma D.7 to hold, and we obtain that using these ( ϵ h ) 's in Theorem 4.6 results in an overall error for DynPG given by

<!-- formula-not-decoded -->

Note that we can again rewrite H to be

<!-- formula-not-decoded -->

Thus, the complexity bounds for the optimization epochs h = 0 , . . . , H are given by

<!-- formula-not-decoded -->

where we used the calculations in the proof of Theorem D.6 in the first step and substituted ϵ by (1 -γ ) ϵ 3 .

Lemma D.9. The term log ( γ -1 ) = -log( γ ) is asymptotically equivalent to (1 -γ ) for γ ↑ 1 .

Proof. By the definition of asymptotic equivalence for two functions f and g we have to show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E DynPG in Large or Complex Environments

In the following, we aim to discuss a challenge that may arise when applying DynPG in practice. In very complex MDPs, where the policy parametrization needs to be rich, the DynPG approach in Algorithm 1 can suffer from a storage problem. The number of policy parameters that need to be stored for future decisions scale linearly with the number of training steps. In order to circumvent this problem for practical applications we propose an actor-critic based modification of DynPG, called Dynamic Actor-Critic (DynAC). The general idea behind actor-critic in vanilla PG is to introduce a so called critic, a second parametrized class of functions ( Q w ) w ∈ R l , which approximates the Q-function Q π θ in the policy gradient theorem. Using the critic as estimator of the Q-values no more roll-outs are needed to estimate the rewards-to-go. In the actor-critic variant of DynPG, we include an additional training procedure after optimizing policy ˆ π ∗ h to update the critic based on the old critic,

Q w h ≈ Q ˆ π ∗ h -1 ,..., ˆ π ∗ 0 h , and the newly trained policy, ˆ π ∗ h , using the relation:

<!-- formula-not-decoded -->

The critic is used in the policy gradient theorem as an estimator for Q π h h +1 such that the new gradient in DynAC is given by

<!-- formula-not-decoded -->

It holds by L'Hospital rule that

<!-- formula-not-decoded -->

```
Algorithm 2: DynAC Input: Initial state distribution µ , class of policies ( π θ ) θ ∈ R d , class of critic parametrization ( Q w ) w ∈ R l . Result: Approximation of π ∗ , denoted as ˆ π ∗ . 1 Set h = 0 ; 2 Train Q w 1 to approximate the reward function r ; 3 while Convergence criterion not met do 4 Initialize θ 0 (e.g., θ 0 ≡ 0 ); 5 Choose α h and N h (cf., Remark 4.4); 6 for n = 0 , . . . , N h -1 do 7 Sample S ∼ µ and A ∼ π θ ( ·| S ) ; 8 Set G Dyn-AC = ∇ θ log( π θ ( A | S )) Q w h +1 ( S, A ) ; 9 Update θ n +1 = θ n + α h G ; 10 end 11 Set ˆ π ∗ h = π θ Nh ; 12 Train w h +2 s.t. for all s, a : Q w h +2 ( s, a ) ≈ E S ∼ p ( ·| s,a ) ,A ∼ ˆ π ∗ h ( ·| S ) [ r ( s, a ) + γQ w h +1 ( S, A )] ; 13 Set h = h +1 ; 14 end 15 Return ˆ π ∗ = ˆ π ∗ h -1 ;
```

where S ∼ µ and A ∼ π θ ( ·| s ) and π θ is the policy we are currently training for epoch h +1 . This way there is no need to store all trained policies only the currently trained policy and the current critic are needed. For applications where the performance of vanilla PG is poor and the parametrized policy needs to be very rich such that storing many policies might cause storage issues, we suggest to keep this variant of DynPG in mind. We call this approach DynAC for dynamic actor-critic and summarize the steps in Algorithm 2.

A theoretical analysis of this approach would require an assumption on the approximation error to train the Q-functions. As the gradients are no longer unbiased, convergence towards the global optimum cannot be guaranteed and the bias errors will appear in the overall error. We leave further investigations of DynAC to future work, as an in-depth exploration would exceed the scope of this paper.

## F Convergence Rates under Tabular Softmax Parametrization - Sampled Gradients

In this section we drop the exact gradient assumption and analyze DynPG for sample based gradient estimates (generated by MDP roll-outs). We emphasize that the main insight into DynPG (breaking the (possibly) exponential γ -dependence) is part of the exact gradient analysis. Even though the estimates provided here are not relevant for practical use, we add this discussion for completeness.

For a fixed future policy ˆ π ∗ h , consider K h trajectories ( s i k , a i k ) h k =0 , for i = 1 , . . . , K h , generated by s i 0 ∼ µ , a i 0 ∼ π θ and a i k ∼ ˆ π ∗ h -k for 1 ≤ k ≤ h . In analogy to the gradient estimator for vanilla PG (motivated by the policy gradient theorem) the estimator of ∇ θ V π θ , ˆ π ∗ h h +1 ( µ ) is defined by

<!-- formula-not-decoded -->

where ˆ Q i h +1 = ∑ h k =0 γ k r ( s i k , a i k ) is an unbiased estimator of Q ˆ π ∗ h h +1 ( s i 0 , a i 0 ) . Then the stochastic PG update for training the parameter θ in epoch h is given by

<!-- formula-not-decoded -->

Our main result for the optimization of the contextual bandits in DynPG is given as follows.

Theorem F.1. Let Assumption 4.5 hold true and DynPG be used with stochastic updates from (19) . Suppose, we are in epoch h with fixed future policy Λ = ˆ π ∗ h and for any δ h &gt; 0 and ϵ h &gt; 0 , choose

<!-- formula-not-decoded -->

To prove Theorem F.1 we adapt the proof ideas in [12, Sec. D.2] for finite-time dynamic policy gradient to our discounted setting. We restate all results and show how to adapt the proofs.

We first verify that the gradient estimator in Appendix F is unbiased and has bounded variance.

Lemma F.2. Consider the estimator ˆ G K h ( θ ) from Appendix F. For any K h &gt; 0 it holds that

<!-- formula-not-decoded -->

and under softmax parametrization

<!-- formula-not-decoded -->

Proof. By the definition of ̂ G K h ( θ ) we have

<!-- formula-not-decoded -->

as in [12, Lem. D.6] and from the DynPG policy gradient theorem (Theorem C.3), we obtain the unbiasedness directly.

For the second claim, we first obtain that

<!-- formula-not-decoded -->

Note that we have nearly the same term as in the proof of [12, Lem. D.6]. We can follow line by line the arguments there and have to replace the bounds ∑ H -1 k = h r ( S k , A k ) ≤ ( H -h ) R ∗ with our upper bound ∑ h k =0 γ k r ( S k , A k ) ≤ 1 -γ h +1 1 -γ R ∗ .

So replacing ( H -h ) with 1 -γ h +1 1 -γ , we obtain that

<!-- formula-not-decoded -->

We continue to follow the proof in [12, Sec. D.2] and introduce a stopping time to control the non-uniform gradient domination inequality along the exact gradient trajectories.

Recall that we consider the optimization of DynPG at epoch h . Let ( ¯ θ n ) n ≥ 0 be the stochastic process from (19) and let ( θ n ) n ≥ 0 be the deterministic sequence generated by DynPG with exact gradients. Assume in the following that the initial parameter agrees, i.e. θ 0 = ¯ θ 0 , and the step size η h is the same for both processes.

We denote by ( F n ) n ≥ 0 the natural filtration of ( ¯ θ n ) n ≥ 0 . As we assume uniform initialization, the nonuniform gradient domination property holds along the deterministic gradient scheme with constant c h = min n ≥ 0 min s ∈S π θ n ( a ∗ ( s ) | s ) = 1 |A| (Lemma D.4).

Therefore we define the stopping time

<!-- formula-not-decoded -->

Employing this stopping time, we can control the non-uniform gradient domination property along the stochastic gradient trajectory, before the stopping time is hit.

Lemma F.3. Let Assumption 4.5 hold true and DynPG be used with stochastic updates from (19) . Suppose, we are in epoch h with fixed future policy Λ = ˆ π ∗ h . Then, it holds almost surely that min 0 ≤ n ≤ τ min s ∈S π ¯ θ n ( a ∗ ( s ) | s ) ≥ 1 2 |A| is strictly positive.

<!-- formula-not-decoded -->

We consider the two events { n ≤ τ } and { n ≥ τ } separately. On the event { n ≤ τ h } we obtain the following convergence rate.

Lemma F.4. Let Assumption 4.5 hold true and DynPG be used with stochastic updates from (19) . Suppose, we are in epoch h with fixed future policy Λ = ˆ π ∗ h . Suppose, that

- the batch size ( K h ) n ≥ 45 64 |A| 2 N 3 2 h (1 -1 2 √ N h ) n 2 is increasing for some N h ≥ 1
- the step size η h = 1 -γ 2(1 -γ h +1 ) R ∗ √ N h .

Then,

<!-- formula-not-decoded -->

Proof. Line by line as in [12, Lem. D.8]. Note that we have a different smoothness constant and a different variance control. In both constants one has to replace ( H -h ) in [12, Lem. D.8] with 1 -γ h +1 1 -γ and c h with 1 |A ] .

Next, we bound the probability of the event { n ≥ τ } by δ for a large enough batch size K h .

Lemma F.5. Let Assumption 4.5 hold true and DynPG be used with stochastic updates from (19) . Suppose, we are in epoch h with fixed future policy Λ = ˆ π ∗ h . For any δ &gt; 0 , suppose that

- the batch size K h ≥ 5 |A| 2 n 3 δ 2
- the step size η h ≤ 1 √ nL h .

Then, we have P ( n ≥ τ ) &lt; δ .

<!-- formula-not-decoded -->

Using the above Lemmas we can prove the convergence of each contextual bandit problem.

Proof of Theorem F.1. Proof works as in [12, Lem. D.10], we can separate the probability using the stopping time and under the zero approximation error of softmax we have that

<!-- formula-not-decoded -->

where the second inequality is due to Lemma F.4 and Lemma F.5. The last inequality follows by our choice of N h as in [12, Lem. D.10] by replacing c h with 1 |A| and ( H -h ) with 1 -γ h +1 1 -γ .

From the previous contextual bandit result, we can derive a sample complexity result under which we obtain convergence in stochastic DynPG with arbitrary high probability.

Theorem F.6. Let Assumption 4.5 hold true and DynPG be used with stochastic updates from (19) . Choose ϵ &gt; 0 , δ &gt; 0 and set

<!-- formula-not-decoded -->

Then, the stationary policy ˆ π ∗ H obtained by stochastic DynPG achieves P ( ∥ V ∗ ∞ -V ˆ π ∗ H ∞ ∥ ∞ &lt; ϵ ) ≥ 1 -δ . The total number of samples is bounded by

<!-- formula-not-decoded -->

Note first, that the constant c 2 in the sample complexity is indeed a natural number, hence independent of any model parameters. The concrete values of H , ( N h ) H h =0 , ( η h ) H h =0 and ( K h ) H h =0 are derived in the proof below.

Remark F.7. We obtain again a sample complexity that scales at most polynomial in (1 -γ ) -1 and guarantees convergence in high probability. In [7] a sample complexity for softmax PG with entropy regularization is derived, where even larger batch sizes are required, which leads to an exponentially

bad sample complexity with respect to (1 -γ ) -1 . Still, we do not expect the rates in Theorem F .6 to be tight, but they prove that convergence in stochastic settings is achievable. We also want to remind the reader of the sample based example presented in Line 13, where no batch is needed for the implementation.

Proof. By our choice of N h , η h and K h and Theorem F.1, it holds that

<!-- formula-not-decoded -->

where in the first inequality we have used we used that

<!-- formula-not-decoded -->

Next we introduce the event C = { ∀ h = 0 , . . . , H : ∥ sup θ V { π θ , Λ } h +1 -V { π ¯ θ N h , Λ } h +1 ∥ ∞ &lt; ϵ h } . By Lemma D.7 we deduce that under the event C it holds almost surely that

<!-- formula-not-decoded -->

by our choices of ϵ h 's and H . So C ⊆ {∥ V ∗ ∞ -V ˆ π ∗ H ∞ ∥ ∞ &lt; ϵ } and therefore,

<!-- formula-not-decoded -->

To upper bound the total number of samples, we suppress the Gauss-brackets for simplicity and compute

<!-- formula-not-decoded -->

where we used Jensen's inequality twice and defined the natural numbers c 1 := 5 ∗ (144) 4 and c 2 := c 1 ∗ 6 8 which are independent of any model parameters. Next, since γ ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

and obtain

<!-- formula-not-decoded -->

Finally, for H = ⌈ log γ ( (1 -γ ) 2 ϵ 6 R ∗ )⌉ = ⌈ log((1 -γ ) -1 ϵ -1 2 R ∗ ) log( γ -1 ) ⌉ , there holds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->
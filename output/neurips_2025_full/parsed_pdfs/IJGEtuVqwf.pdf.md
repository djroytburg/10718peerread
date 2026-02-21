## Robust Reinforcement Learning in Finance: Modeling Market Impact with Elliptic Uncertainty Sets

## Shaocong Ma

Department of Computer Science University of Maryland College Park, MD 20742, USA

scma0908@umd.edu

## Heng Huang ˚

Department of Computer Science University of Maryland College Park, MD 20742, USA heng@umd.edu

## Abstract

In financial applications, reinforcement learning (RL) agents are commonly trained on historical data, where their actions do not influence prices. However, during deployment, these agents trade in live markets where their own transactions can shift asset prices, a phenomenon known as market impact. This mismatch between training and deployment environments can significantly degrade performance. Traditional robust RL approaches address this model misspecification by optimizing the worst-case performance over a set of uncertainties, but typically rely on symmetric structures that fail to capture the directional nature of market impact. To address this issue, we develop a novel class of elliptic uncertainty sets. We establish both implicit and explicit closed-form solutions for the worst-case uncertainty under these sets, enabling efficient and tractable robust policy evaluation. Experiments on single-asset and multi-asset trading tasks demonstrate that our method achieves superior Sharpe ratio and remains robust under increasing trade volumes, offering a more faithful and scalable approach to RL in financial markets.

## 1 Introduction

Reinforcement learning (RL) has emerged as a promising decision-making framework for quantitative trading strategies [1, 2], including portfolio optimization [3-13], automatic trading [3, 14-19], market making [20-22], and option hedging [23, 24]. RL's appeal in finance lies in its ability to learn adaptive strategies directly from data, without strong market assumptions [1]. This makes it wellsuited for capturing complex market dynamics and aligning with the sequential nature of financial decision-making.

One of the primary challenges in training a robust and consistently profitable RL agent lies in handling the market impact [25-27]; that is, the influence of the agent's own trades on asset prices in the deployed environment. For example, when a trader buys or sells a large volume of an asset, it can temporarily drive the price up or down, respectively (as illustrated in Figure 1). Typically, RL agents are trained on historical market data where market impact is absent. However, during deployment, the environment shifts from a passive historical setting to the real market, where the agent's actions actively affect prices. This discrepancy between training and deployment environments undermines the optimality and robustness of the learned policy, and leads us to the central question in this paper:

- Q : Can we train RL agents on historical data while robustly accounting for market impact during deployment?

˚ This work was partially supported by NSF IIS 2347592, 2348169, DBI 2405416, CCF 2348306, CNS 2347617, RISE 2536663.

<!-- image -->

Relative Time (ms)

RelativeTime (ms)

Figure 1: Market impact illustration using AMZN stock on June 21, 2012, based on 5-level LOBSTER data [47]. The left panel shows the price response to executing a buy order of 100 shares at the time t ' 0 ms, which consumes ask-side liquidity within 1000 ms and induces an immediate upward shift in price. The right panel shows the analogous impact of a sell order. This plot indicates that the transition dynamics induced by trading are not symmetrically distributed around the nominal kernel.

To address this central question, we adopt the framework of robust RL [28-46], which is designed to handle model misspecification ; that is, the mismatch between the training and deployment environments. The robust RL framework explicitly acknowledges that the real market environment may differ from the simulated training environment, and it aims to learn policies that are resilient under a range of perturbations.

However, existing robust RL approaches face a key limitations in financial applications; that is, traditional uncertainty sets are typically symmetric, which fail to capture the directional nature of market impact. Addressing this challenge motivates our threefold contributions :

- (1) We propose a novel class of uncertainty sets, elliptic uncertainty sets (Definition 3.2), which generalize traditional ℓ p -norm balls to the ellipse-like structure. These sets better capture the empirically observed directional nature of market impact as illustrated in Figure 1.
- (2) On the theoretical side, we derive closed-form solutions for solving the worst-case transition kernel under the proposed uncertainty sets (Theorem 3.4). Furthermore, under certain conditions, we present the explicit solutions (Theorem 3.5). This development significantly broadens the scope of tractable robust RL problems beyond symmetric (ball-shaped) uncertainty sets, enabling more faithful representation of market impact.
- (3) We empirically evaluate our approach on real-world financial data using trade-level market impact simulations in Section 4. Experimental results demonstrate that our method consistently outperforms the standard single-asset intra-day trading strategy and existing RL baselines in terms of Sharpe ratio. Moreover, we validate the robustness of our method under increasing strategy volume, confirming its effectiveness in high-volume regimes.

## 1.1 Related Work

Existing Approaches to Handle Market Impact The most common approach for handling market impact is to simulate the electronic market more accurately. By applying high-fidelity market simulators, often built upon limit order book (LOB) dynamics [25, 26, 48-50], trade-level data [51, 52], or large-scale agent-based simulators [53-56], it captures more detailed market microstructure and reduces the gap between the simulated and the real trading environments. Prominent approaches to incorporating market impact into backtesting include agent-based simulation frameworks [53-56], data-driven LOB reconstruction models [57, 58, 50], and hybrid systems that integrate historical data replay with synthetic order flow generation [59, 60]. However, access to high-quality market data is often limited, and simulating market environments with agent-based systems remains prohibitively expensive. Therefore, a practical alternative is to train the agent directly on historical data without market impact while still encouraging it to account for potential worst-case scenarios.

Robust RL in Finance Robust RL, with its intrinsic ability to handle model misspecification, provides a natural framework for incorporating market impact considerations during training. Jaimungal et al. [3] propose a robust reinforcement learning framework based on rank-dependent utility to address uncertainty in financial decision-making, demonstrating the effectiveness of robust RL in portfolio allocation, benchmark strategy optimization, and statistical arbitrage. Shi et al. [5] formulate portfolio optimization as a robust RL problem to enhance resilience. We et al. [24] extend robust risk-aware RL to manage the risks associated with path-dependent financial derivatives, showcasing its effectiveness in complex hedging scenarios. However, existing work primarily focuses on fully symmetric uncertainty sets, which fail to capture the directional characteristics of financial markets. Addressing this limitation is the central focus of our paper.

Modeling the Uncertainty Set in Robust RL The uncertainty set captures the discrepancy between training and deployment environments. However, robust RL becomes computationally intractable when the uncertainty set is highly irregular [28, 44, 39, 40]. To mitigate this issue, it is common to impose structural assumptions that enable tractable solutions. We highlight several representative structures, with further details deferred to Appendix A.1. The R -contamination model [31] defines an uncertainty set as a sphere of radius R centered around the nominal transition kernel, admitting analytical solutions for robust policy evaluation. The ℓ p -norm uncertainty sets are also widely studied due to their closed-form solutions [39, 40]. The integral probability metric (IPM) and double-sampling uncertainty sets have been shown to allow efficient computation [43]. Other works include uncertainty sets based on the Wasserstein metric and f -divergence, which have also received considerable attention [61-63].

## 2 Preliminaries: Robust Reinforcement Learning

We focus on the discounted infinite-horizon Markov Decision Processes (MDPs) [64], formally defined as a five-tuple p S , A , P , r, γ q , where S and A denote the state and action spaces 2 , respectively. The transition kernel P p s 1 | s, a q specifies the probability of transitioning to state s 1 from state s after taking action a . The reward function r : S Ñr 0 , 1 s assigns a bounded reward to each state, and the discount factor γ P p 0 , 1 q models the agent's preference for immediate rewards over future ones.

Instead of assuming a fixed transition kernel P , we account for the effects of model misspecification . Let P 0 denote the nominal transition kernel, and define the p s, a q -uncertainty set U s,a Ă R | S | at the point p s, a q P S ˆ A as

<!-- formula-not-decoded -->

where C s,a Ă R | S | is a convex set, 1 | S | is an all-one vector with the dimension | S | (for convenience, we will usually omit the subscript | S | ). The convex set is commonly chosen as a ball-shaped set (e.g. C s,a ' B p p β s,a q : ' t u s,a P R | S | | } u s,a } p ď β s,a u for the ℓ p -norm uncertainty set), where β s,a ě 0 is a radius parameter that quantifies the allowable deviation. The zero-sum constraint u J s,a 1 | S | ' 0 ensures the perturbed transition P p¨ | s, a q ` u s,a remains a valid probability distribution. The uncertainty set U and the robust transition model are then defined as

ą

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

respectively. Throughout this paper, we assume the parameter of uncertainty set is chosen appropriately such that all elements in the robust transition model is well-defined [39, 40]. Given these notations in place, we define the value function of the policy π with the uncertainty u as the value

2 For theoretical analysis, we restrict our attention to MDPs with finite state and action spaces. This assumption avoids the technical complications arising from continuous or hybrid spaces, which, although explored in some prior work [65-68], remain analytically open without imposing additional assumptions, especially for robust RL problems. Nevertheless, our empirical results extend to continuous settings, demonstrating the practical applicability of our approach beyond the theoretical scope.

Figure 2: Illustration of the transition kernel in a simplified robust RL setting using an ℓ 8 -norm uncertainty set. Only one of the upward (buy) or downward (sell) shift is plausible in a given scenario; however, the ball-shaped uncertainty set must include both shifts due to its symmetric structure, which motivates us to propose an uncertainty set with non-symmetric structures.

<!-- image -->

function with the transition probability P u P P :

<!-- formula-not-decoded -->

The robust value function is the worst-case value function over all uncertainties; that is V π p s q : ' min u P U V π u p s q . Similarly, we can also define the robust Q-function and the robust advantage function as Q π p s, a q : ' min u E ' ř 8 t ' 0 γ t r p s t , a t q | s 0 ' s, a 0 ' a, P u , π ‰ and A π p s, a q : ' Q π p s, a q ´ V π p s q , respectively.

The goal of robust RL is to learn a parameterized policy π θ that maximizes the worst-case value function V π θ p s 0 q , where s 0 denotes the initial state. A standard approach applies the robust policy gradient formula [28]:

<!-- formula-not-decoded -->

where d π θ is the stationary distribution induced by π θ , and Q π θ denotes the robust Q-function. This formulation reduces robust RL to a gradient-based optimization problem, shifting the main challenge of robust RL to accurately evaluate the robust value function, which is our focus in Section 3.3.

## 3 Modeling Market Impact with Elliptic Uncertainty Sets

## 3.1 Limitations of Symmetric Uncertainty Sets

Robust MDPs offer an ideal framework to handle model misspecification by allowing the transition kernel to deviate within a prescribed uncertainty set. However, most existing formulations adopt symmetric structures, typically the ball defined by a specific norm, that treat all directions of perturbation equally. While mathematically convenient, these symmetric structures often fail to reflect the directional nature of real-world uncertainties.

Symmetry in this context typically refers to invariance under the signed permutation group (see Appendix A.2 for details). A canonical example is the ℓ p -norm ball:

<!-- formula-not-decoded -->

which satisfies the property that for any u P B p p β q , all signed permutations of u are also contained in the set. It enforces an implicit assumption of isotropic uncertainty, equally plausible in all directions, which often includes unrealistic perturbations. We demonstrate it in the following example:

Example 3.1 (Symmetric Sets Fail to Capture Directional Uncertainty) . In financial markets, large buy or sell orders induce directional shifts in asset prices due to liquidity consumption (see Figure 1). In Figure 2, we consider the following classical ℓ 8 -norm p s, a q -uncertainty set:

<!-- formula-not-decoded -->

This set includes, for example, the perturbation vectors

<!-- formula-not-decoded -->

Both u 1 and u 2 satisfy the norm and mean constraints, and since they are signed permutations of each other, they must either both belong to U s,a or be excluded together. However, this symmetry fails to reflect market realities: under a buy action, u 1 represents a plausible upward shift due to liquidity-driven market impact, while u 2 , corresponding to a downward shift, is implausible. Thus, the symmetric structure forces inclusion of perturbations that contradict the directional market impact, potentially leading to overly conservative or unrealistic robust policies.

This observation underscores the importance of developing a robust RL framework that can capture the directional nature of environment shifts observed in financial markets. To this end, we introduce a novel class of elliptic uncertainty sets , which generalize traditional norm-bounded sets by allowing non-symmetric perturbations, while retaining closed-form tractability under certain conditions.

On the Conservativeness of Robust Policies Although our proposed elliptic uncertainty sets better capture the directional nature of market impact, this alone does not immediately clarify why they improve robustness. To illustrate the underlying intuition, we present a simple example:

- The robust value function V π : ' min u P U V π u models the worst-case discounted future return. If the uncertainty set U is larger, it is more conservative, as it yields smaller value.
- In the ideal case, the uncertainty set consists of a single element u 0 that exactly characterizes the MDP induced by the true market impact. Training an RL agent on the robust MDP with U ' u 0 is then equivalent to training directly on the real LOB data.
- In the less ideal case where U includes additional elements, e.g. U ' t u 0 , u 1 u . The robust value function may still achieve its minimum at u 0 ( u 0 ' arg min u min u P U V π u ). In this scenario, the robust formulation reduces to the ideal case. Otherwise, if the minimum is attained at some u ‰ u 0 , the robust value becomes strictly smaller than V π u 0 , making the policy more conservative.

Unfortunately, neither our approach nor standard robust RL methods can theoretically rule out this latter 'overly conservative' scenario. Our guiding intuition, however, is that smaller uncertainty sets are less likely to contain such overly conservative elements. By trimming down the traditional uncertainty sets, our method reduces the risk of unnecessary conservatism, though it does not eliminate it entirely.

## 3.2 The Elliptic Uncertainty Sets

The elliptic uncertainty sets generalize the classical ℓ p -norm uncertainty set by incorporating directional non-symmetry. Formally, we have the following definition:

Definition 3.2 (Elliptic p s, a q -Uncertainty Set) . For each state-action pair p s, a q P S ˆ A , the elliptic p s, a q -uncertainty set is defined as

<!-- formula-not-decoded -->

where t u s,a n u N n ' 1 P R | S | are called the foci of the ellipse, β s,a ě 0 is the uncertainty size , and } ¨ } : R d Ñ R is an arbitrary norm. Particularly, when } ¨ } is taken as the ℓ p -norm ( p P r 1 , `8s ), U s,a is called the ℓ p -ellipse uncertainty set .

Here the constraint u J 1 | S | ' 0 ensures the perturbed transition still defines a valid probability distribution. For convenience, we will omit the superscript in u s,a n and the subscript in β s,a when the context is clear. Importantly, we note that it is unavoidable that the element in U s,a may not be a valid probability distribution when some entries of u ` P 0 p¨| s, a q are negative. To avoid this scenario, we include the following regular condition, which is also presented in the ℓ p -norm uncertainty set literature [39, 40]:

Assumption 3.3. For the given MDP p S , A , P 0 , r, γ q , for any set of foci t u s,a n u N n ' 1 , there exists a constant β ą 0 such that for all β s,a ă β , Eq. (3) induces valid probability transitions; that is, all entries of u ` P 0 p¨| s, a q are non-negative.

Connection to the Classical Ellipse Our definition draws directly from the geometric characterization of an ellipse: the set of points for which the sum of distances to multiple foci is bounded by a constant β s,a . We show that it recovers classical structures as special cases with the following two concrete examples:

- (1) When N ' 1 and u 1 ' 0 , the set Eq. (3) reduces to the ℓ p -norm ball

<!-- formula-not-decoded -->

which aligns with the standard uncertainty set used in robust RL (e.g., [39, 40, 28]).

- (2) When N ' 2 and p ' 2 , Eq. (3) becomes } u ´ u 1 } 2 `} u ´ u 2 } 2 ď β. Defining the midpoint ¯ u : ' u 1 ` u 2 2 , there exists a matrix A such that this constraint is equivalent to a classical quadratic form of an ellipse:

<!-- formula-not-decoded -->

where the detailed derivation is given in Lemma B.10.

These examples demonstrate that our formulation significantly generalizes classical ellipses by allowing arbitrary norms and accommodating multiple foci, thereby enabling more flexible modeling of non-symmetric perturbations. However, such generality often comes at the cost of increased complexity. To address this concern, in the next section, we show that, under certain parameter choices, our proposed uncertainty set remains as trackable as traditional ℓ p -norm uncertainty sets.

Limitations &amp; Potential Directions As our approach is mainly adopted from the ℓ p -norm uncertainty set, it does not guarantee that the resulting distributions are absolutely continuous with respect to the nominal transition distribution even with Assumption 3.3. One potential solution is incorporating f -divergence or distributionally robust RL techniques [69-74].

## 3.3 Solving the Worst-Case Uncertainty

Solving the worst-case uncertainty u ˚ : ' arg min V π u p s 0 q plays a crucial role in efficient robust policy evaluation. The ℓ p -norm uncertainty set [39, 40] and the R -contamination model [31] are popular as the solution u ˚ is closed-form. When the uncertainty set is complicated, many existing robust RL methods require an external optimization loop to determine the worst-case transition probability, which can be impractical in real-world scenarios.

To address this issue, we derive an implicit solution (Theorem 3.4) for the worst-case uncertainty that avoids additional interaction with the environment. Under certain conditions, we further provide an explicit closed-form solution (Theorem 3.5), enabling direct computation without the need for external solvers.

We start with recapping some backgrounds in robust TD learning. Let V denote all functions mapping from the state space S to the Euclidean space R . Given that U is an arbitrary uncertainty set, the robust Bellman operator associated with the policy π , T π U : V Ñ V , is defined as

<!-- formula-not-decoded -->

As shown by [75, 28], when γ P p 0 , 1 q , the robust Bellman operator is a contraction operator, which admits the unique fixed point as the robust value function V π P V . When U is given by the ℓ p -ellipse uncertainty set (Eq. (3)), then

<!-- formula-not-decoded -->

It turns out that if we can solve the optimization problem

<!-- formula-not-decoded -->

the robust Bellman operator is just the standard Bellman operator (over the nominal transition probability) with a solved shift. As the result, we can simply apply the standard TD-learning with adding this correction term to solve the desired robust value function. In the following theorem, we present a general recipe of solving this optimization problem.

Theorem 3.4 (Implicit) . Let d : ' | S | be the cardinal of state space. Suppose that t u n u N n ' 1 Ă R d for each p s, a q P S ˆ A , the uncertainty size β ě 0 , and v P R d . Then there exists λ ˚ and µ ˚ such that the optimization problem defined by Eq. (4) is solved by

<!-- formula-not-decoded -->

Remark. We say a solution of the optimization problem is 'implicit' if it can be represented as an equation in the form g p u, v, β, t u n uq ' 0 . In this theorem, as the right-hand side

<!-- formula-not-decoded -->

is proper, convex, and coercive; we can surely re-write it as the sub-gradient form by letting

<!-- formula-not-decoded -->

Then we obtain the implicit representation g p u, v, β, t u n uq ' 0 . Moreover, we derive the formula of λ ˚ and µ ˚ beyond the existence; the full result is presented in Theorem B.12 with more details.

The implicit solution has already shown significant advances compared to some of existing robust RL methods which typically require to solve the worst-case transition probability using additional state-action sample generated from the agent-environment iteration. However, it is still (slightly) impractical to solve additional convex optimization problems in every iteration. Fortunately, under certain conditions, the solution can be 'explicit'; that is, we can write the optimal solution in the form of u ˚ ' f p v, β, t u n uq .

Theorem 3.5 (Explicit) . Let d : ' | S | be the cardinal of state space. Suppose that t u n u N n ' 1 Ă R d for each p s, a q P S ˆ A , the uncertainty size β ě 0 , and v P R d . The optimization problem defined by Eq. (4) is explicitly solved in the following cases:

(a) Let N ' 1 and 1 p ` 1 q ' 1 . Then the minimizer of Eq. (4) is given by

<!-- formula-not-decoded -->

Here sign and | ¨ | are coordinate-wise sign function and absolute value, respectively; d is the coordinate-wise product; and µ ˚ : ' arg min µ P R } v ` µ 1 } q .

- (b) Let p ' 1 and N ' 2 . Suppose that β ą } u 2 ´ u 1 } 1 . Then the minimizer of Eq. (4) is given by

<!-- formula-not-decoded -->

Here sign , | ¨ | , and I are coordinate-wise sign function, absolute value, and indicator function, respectively; d is the coordinate-wise product;

<!-- formula-not-decoded -->

- (c) Let p ' 2 and N ' 2 . Suppose that β ą } u 2 ´ u 1 } 2 . Then the minimizer of Eq. (4) is given by

<!-- formula-not-decoded -->

Here Ω : ' I ´ 1 β 2 p u 2 ´ u 1 qp u 2 ´ u 1 q J ;

<!-- formula-not-decoded -->

Remark. This result presents a clean form of the worst-case uncertainty u ˚ under certain conditions. Unlike the implicit case where it takes an additional root-finding algorithm to solve an approximated u ˚ , in the explicit case, if the current value function v P V is given and the parameters of the uncertainty set ( u i and β ) have been determined, the uncertainty u ˚ can be explicitly solved. The full proof is presented in Theorem B.13.

## 3.4 Robust TD Learning Algorithm

Given Theorem 3.4 and Theorem 3.5 in place, we immediately obtain the robust TD learning algorithm for robust policy evaluation. Given the current value function v , we can calculate u ˚ using the implicit and the explicit formula; assume the current state-action pair is given as p s, a, s 1 q , then the updated value function is given by

<!-- formula-not-decoded -->

We can further simplify this update rule by using an unbiased estimator of that:

<!-- formula-not-decoded -->

where s 1 ' P 0 p¨| s, a q . The formulation leads us to Algorithm 1.

## Algorithm 1: Robust Policy Evaluation

Input: The target policy π , the foci t p i u N i ' 1 Ă R d , and t β s,a u the uncertainty size

Sample the initial state s 0 from the initial distribution;

Initialize the value function v 0 ;

for

t

´

0

,

1

,

2

, . . . , T

'

Sample the action

a

1

t

do

'

Calculate

t

using

u

π

p

˚

Robust TD-learning:

i

u

v

N

i

1

'

p

p¨|

s

t

1

,

s

t

q

; Transition from

β

q '

t

u

, and

;

v

s,a

v

p

q `

s

end

Output: The final value function v T

Remark. The convergence of this algorithm, as well as its corresponding Actor-Critic-style policy gradient algorithm, follows directly by applying the standard proof routine from the robust RL literature (e.g. [43]). For completeness, we include the convergence result and full proof in the supplementary material.

## 4 Experiments

To validate our theoretical findings and demonstrate the practical effectiveness of robust RL framework in the environment with the market impact, we conduct experiments on two different tasks that are closely tied to market impact: (1) minute-level single-asset strategy, and (2) large-volume portfolio rebalancing. In minute-level trading, even small trade sizes can noticeably move prices, leading to slippage. Similarly, large-scale portfolio rebalancing, often performed by large financial institutions, can significantly affect asset prices due to the large order volumes involved.

## 4.1 Performance Comparison on Single-Asset Intra-Day Trading

We start with the single-asset minute-level trading. The non-RL baseline is chosen as the momentum strategy [76], which is designed based on the empirical observation where assets that have performed well in the recent past are more likely to continue performing well in the near future.

Training and Evaluation of RL Agents We implement a Gym-like RL environment [77, 78] constructed on historical data, with full environment details provided in Appendix C.3. All RL agents are trained on one year of earlier historical data (from May 9th, 2021 to May 9th, 2022 as the nominal environment) without accounting for market impact. Their performances are then evaluated over the period from June 9 to December 9, 2022, with the market impact included. To simulate market impact, we reconstruct LOB dynamics using a short period of real trading orders and determine the execution price via the volume-weighted average price (VWAP). A simple example illustrating this estimation process is shown in Table 2, Appendix C.

Results As shown in Table 1, our proposed method consistently outperforms the momentum strategy, the non-robust RL, and the symmetric robust RL baselines (based on ℓ p -norm balls) in terms of the

ηγv

J

˚

u

`

s

t

η

to

p

r

p

s

`

t

1

'

s, a q `

P

0

γv p¨|

p

s

s

1

t

, a q ´

t

q

v

;

p

s

qq

;

Table 1: Performance comparison of different RL agents on selected assets under the simulated market impact from June 9 to December 9, 2022. Robust RL with ℓ p -ellipse uncertainty set consistently achieves the highest Sharpe ratio.

| Asset   | Method                    | Final Value ($)   | Annualized Return (%)   | Sharpe Ratio   | Max Drawdown (%)   |
|---------|---------------------------|-------------------|-------------------------|----------------|--------------------|
| META    | Momentum                  | 96 334            | ´ 7 . 3%                | ´ 0 . 95       | ´ 4 . 2%           |
| META    | Non-Robust RL             | 120 347           | 44 . 8%                 | 1 . 74         | ´ 11 . 3%          |
| META    | Robust RL ( ℓ p -Ball)    | 97 103            | ´ 5 . 7%                | ´ 0 . 28       | ´ 13 . 4%          |
| META    | Robust RL ( ℓ p -Ellipse) | 138 011           | 90 . 8%                 | 2 . 48         | ´ 9 . 9%           |
| MSFT    | Momentum                  | 105 163           | 10 . 6%                 | 1 . 10         | ´ 5 . 3%           |
| MSFT    | Non-Robust RL             | 87 440            | ´ 23 . 5%               | ´ 1 . 75       | ´ 16 . 9%          |
| MSFT    | Robust RL ( ℓ p -Ball)    | 92 159            | ´ 15 . 1%               | ´ 0 . 82       | ´ 11 . 2%          |
| MSFT    | Robust RL ( ℓ p -Ellipse) | 111 485           | 24 . 4%                 | 1 . 20         | ´ 10 . 1%          |
| SPY     | Momentum                  | 107 333           | 15 . 2%                 | 1 . 69         | ´ 3 . 2%           |
| SPY     | Non-Robust RL             | 91 947            | ´ 15 . 5%               | ´ 1 . 64       | ´ 11 . 9%          |
| SPY     | Robust RL ( ℓ p -Ball)    | 100 560           | 1 . 1%                  | 0 . 17         | ´ 6 . 4%           |
| SPY     | Robust RL ( ℓ p -Ellipse) | 109 272           | 19 . 4%                 | 1 . 60         | ´ 5 . 8%           |

Figure 3: Performance comparison of trading strategies on the META stock from June 9 to December 9, 2022, under simulated market impact. The left panel compares the final portfolio values with (in red) and without market impact (in green), illustrating the robustness of each method to executionrelated slippage. The right panel shows the cumulative returns over the evaluation period, tracking the performance of the four strategies in Table 1, alongside the baseline performance of the META stock.

<!-- image -->

risk-adjusted return (Sharpe Ratio). These experiments validate the following key understandings: (i) While robust RL with symmetric uncertainty sets significantly mitigates the effects of market impact (as illustrated in the left panel of Figure 3), it often produces overly conservative strategies that compromise profitability by taking implausible perturbations into consideration; (ii) the nonrobust RL usually suffers greater risk exposure, resulting in the highest Max Drawdown among all methods; (iii) in contrast, the proposed ℓ p -ellipse uncertainty set effectively captures the directional non-symmetry of market impact, allowing the agent to achieve a more favorable trade-off between robustness and return.

## 4.2 Robustness to the Market Impact Scaling in the Trading Volume

In this subsection, we show that a policy trained in a low-volume environment continues to mitigate market impact when transferred to portfolios with significantly larger volumes. We consider a multi-asset portfolio allocation task, modeling the realistic setting where large volumes are traded over short periods to maintain a low-variance portfolio. The same Gym-like RL environment and evaluation period from the previous experiment are used. We evaluate the robustness to the market impact using the relative portfolio gap, the normalized absolute difference in final portfolio value with and without market impact:

<!-- formula-not-decoded -->

where MI represents the market impact. Additional experimental details are provided in Appendix C.

Figure 4: Robustness of RL agents to market impact under increasing trading volumes. The left three panels show normalized portfolio values over time across initial cash levels, with dashed and solid lines indicating performance with and without market impact, respectively. The right panel shows the relative portfolio gap, which increases sharply for the non-robust agent but remains small and stable for the robust RL agent with ℓ p -elliptic uncertainty sets.

<!-- image -->

Results As shown in Figure 4, the robust RL agent with ℓ p -ellipse uncertainty set consistently outperforms the non-robust RL method both in return and in mitigating the effects of market impact. While the performance gap is small at low volume, the non-robust agent degrades rapidly as volume increases, suffering from instability and larger drawdowns. In contrast, the robust agent remains stable and profitable even at high volume ( ' 200 M), demonstrating strong scalability.

## 5 Conclusion &amp; Broader Impact

This paper focuses on the market impact appearing in quantitative trading, where an agent's actions affect prices. By modeling the training environment as the nominal transition kernel, the proposed novel ℓ p -ellipse uncertainty sets better captures the non-symmetric nature of price responses compared to traditional symmetric sets. We established the theoretical tractability of this approach by deriving implicit and explicit closed-form solutions for robust policy evaluation within this framework, enabling efficient robust TD-learning algorithms that account for the market impact during training on the nominal historical environment. Experiments on real historical data demonstrated that our method significantly improves robustness and risk-adjusted returns over non-RL, non-robust RL, and symmetric robust RL baselines. This work broadens the applicability of tractable robust RL and offers a more faithful modeling approach for market impact. The broader impact involves potentially more stable and profitable automated trading strategies.

## References

- [1] Ben Hambly, Renyuan Xu, and Huining Yang. Recent advances in reinforcement learning in finance. Mathematical Finance , 33(3):437-503, 2023.
- [2] Nikolaos Pippas, Cagatay Turkay, and Elliot A Ludvig. The evolution of reinforcement learning in quantitative finance. arXiv preprint arXiv:2408.10932 , 2024.
- [3] Sebastian Jaimungal, Silvana M Pesenti, Ye Sheng Wang, and Hariom Tatsat. Robust risk-aware reinforcement learning. SIAM Journal on Financial Mathematics , 13(1):213-226, 2022.
- [4] Pengqian Yu, Joon Sern Lee, Ilya Kulyatin, Zekun Shi, and Sakyasingha Dasgupta. Model-based deep reinforcement learning for dynamic portfolio optimization. arXiv preprint arXiv:1901.08740 , 2019.
- [5] Xiaochuan Shi, Yihua Zhou, and Lei Wu. Robust reinforcement learning for portfolio management via competition and cooperation strategies, 2023. ICLR 2024 Conference Withdrawn Submission.
- [6] Philip Ndikum and Serge Ndikum. Advancing investment frontiers: Industry-grade deep reinforcement learning for portfolio optimization. arXiv preprint arXiv:2403.07916 , 2024.
- [7] Carlos Betancourt and Wen-Hui Chen. Deep reinforcement learning for portfolio management of markets with a dynamic number of assets. Expert Systems with Applications , 164:114002, 2021.
- [8] Amine Mohamed Aboussalah and Chi-Guhn Lee. Continuous control with stacked deep dynamic recurrent reinforcement learning for portfolio optimization. Expert Systems with Applications , 140:112891, 2020.

- [9] Min-Yuh Day, Ching-Ying Yang, and Yensen Ni. Portfolio dynamic trading strategies using deep reinforcement learning. Soft Computing , 28(15):8715-8730, 2024.
- [10] Yuh-Jong Hu and Shang-Jen Lin. Deep reinforcement learning for optimizing finance portfolio management. In 2019 amity international conference on artificial intelligence (AICAI) , pages 14-20. IEEE, 2019.
- [11] Angelos Filos. Reinforcement learning for portfolio management. arXiv preprint arXiv:1909.09571 , 2019.
- [12] WARAMETH NUIPIAN, PHAYUNG MEESAD, et al. Dynamic Portfolio Management with Deep Reinforcement Learning . PhD thesis, King Mongkut's University of Technology North Bangkok, 2025.
- [13] Farzan Soleymani and Eric Paquet. Financial portfolio optimization with online deep reinforcement learning and restricted stacked autoencoder-deepbreath. Expert Systems with Applications , 156:113456, 2020.
- [14] Hyunmin Cho and Hyun Joon Shin. Trading strategies using reinforcement learning. Journal of the Korea Academia-Industrial cooperation Society , 22(1):123-130, 2021.
- [15] John Moody and Matthew Saffell. Learning to trade via direct reinforcement. IEEE transactions on neural Networks , 12(4):875-889, 2001.
- [16] Yang Li, Wanshan Zheng, and Zibin Zheng. Deep robust reinforcement learning for practical algorithmic trading. IEEE Access , 7:108014-108022, 2019.
- [17] Ji-Heon Park, Jae-Hwan Kim, and Jun-Ho Huh. Deep reinforcement learning robots for algorithmic trading: Considering stock market conditions and us interest rates. IEEE Access , 12:20705-20725, 2024.
- [18] Yasmeen Ansari, Sadaf Yasmin, Sheneela Naz, Hira Zaffar, Zeeshan Ali, Jihoon Moon, and Seungmin Rho. A deep reinforcement learning-based decision support system for automated stock market trading. IEEE Access , 10:127469-127501, 2022.
- [19] Xing Wu, Haolei Chen, Jianjia Wang, Luigi Troiano, Vincenzo Loia, and Hamido Fujita. Adaptive stock trading strategies with deep reinforcement learning methods. Information Sciences , 538:142-158, 2020.
- [20] Olivier Guéant, Charles-Albert Lehalle, and Joaquin Fernandez-Tapia. Dealing with the inventory risk: a solution to the market making problem. Mathematics and Financial Economics , 7(4):477-507, 2013.
- [21] Joel Hasbrouck. Empirical Market Microstructure: The Institutions, Economics, and Econometrics of Securities Trading . Oxford University Press, Oxford, UK, 2007.
- [22] Zihao Zhang, Stefan Zohren, and Stephen Roberts. Deep reinforcement learning for trading. arXiv preprint arXiv:1911.10107 , 2019.
- [23] Hans Buehler, Lukas Gonon, Josef Teichmann, and Ben Wood. Deep hedging. Quantitative Finance , 19(8):1271-1291, 2019.
- [24] David Wu and Sebastian Jaimungal. Robust risk-aware option hedging. Applied Mathematical Finance , 30(3):153-174, 2023.
- [25] Robert Almgren and Neil Chriss. Optimal execution of portfolio transactions. Journal of Risk , 3:5-40, 2001.
- [26] Anna A Obizhaeva and Jiang Wang. Optimal trading strategy and supply/demand dynamics. Journal of Financial markets , 16(1):1-32, 2013.
- [27] Bence Tóth, Zoltán Eisler, and J-P Bouchaud. The square-root impace law also holds for option markets. Wilmott , 2016(85):70-73, 2016.
- [28] Yan Li, Guanghui Lan, and Tuo Zhao. First-order policy optimization for robust markov decision process. arXiv preprint arXiv:2209.10579 , 2022.
- [29] Yike Li, Yunzhe Tian, Endong Tong, Wenjia Niu, and Jiqiang Liu. Robust reinforcement learning via progressive task sequence. In IJCAI , pages 455-463, 2023.
- [30] Guanlin Liu, Zhihan Zhou, Han Liu, and Lifeng Lai. Efficient action robust reinforcement learning with probabilistic policy execution uncertainty. Transactions on Machine Learning Research , 2024.
- [31] Yue Wang and Shaofeng Zou. Online robust reinforcement learning with model uncertainty. In Advances in Neural Information Processing Systems , volume 34, 2021.

- [32] Shangding Gu, Laixi Shi, Muning Wen, Ming Jin, Eric Mazumdar, Yuejie Chi, Adam Wierman, and Costas Spanos. Robust gymnasium: A unified modular benchmark for robust reinforcement learning. In International Conference on Learning Representations , 2025.
- [33] Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust adversarial reinforcement learning. arXiv preprint arXiv:1703.02702 , 2017.
- [34] Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust reinforcement learning using offline data. In Advances in Neural Information Processing Systems , volume 35, 2022.
- [35] Minghong Fang, Xilong Wang, and Neil Zhenqiang Gong. Provably robust federated reinforcement learning. arXiv preprint arXiv:2502.08123 , 2025.
- [36] Wei Shen, Xiaoying Zhang, Yuanshun Yao, Rui Zheng, Hongyi Guo, and Yang Liu. Robust rlhf with noisy rewards. In International Conference on Learning Representations , 2025. Withdrawn submission.
- [37] Shaocong Ma, Ziyi Chen, Shaofeng Zou, and Yi Zhou. Decentralized robust v-learning for solving markov games with model uncertainty. Journal of Machine Learning Research , 24(371):1-40, 2023.
- [38] Pierre Clavier, Laixi Shi, Erwan Le Pennec, Eric Mazumdar, Adam Wierman, and Matthieu Geist. Nearoptimal distributionally robust reinforcement learning with general l \_ p norms. Advances in Neural Information Processing Systems , 37:1750-1810, 2024.
- [39] Navdeep Kumar, Kfir Levy, Kaixin Wang, and Shie Mannor. An efficient solution to s-rectangular robust markov decision processes. arXiv preprint arXiv:2301.13642 , 2023.
- [40] Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Y Levy, and Shie Mannor. Policy gradient for rectangular robust markov decision processes. Advances in Neural Information Processing Systems , 36:59477-59501, 2023.
- [41] Runyu Zhang, Yang Hu, and Na Li. Soft robust MDPs and risk-sensitive MDPs: Equivalence, policy gradient, and sample complexity. In The Twelfth International Conference on Learning Representations , 2024.
- [42] Zifan Wu, Chao Yu, Chen Chen, Jianye Hao, and Hankz Hankui Zhuo. Plan to predict: Learning an uncertainty-foreseeing model for model-based reinforcement learning. Advances in Neural Information Processing Systems , 35:15849-15861, 2022.
- [43] Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, PR Kumar, and Chao Tian. Natural actor-critic for robust reinforcement learning with function approximation. Advances in neural information processing systems , 36:97-133, 2023.
- [44] Qiuhao Wang, Chin Pang Ho, and Marek Petrik. Policy gradient in robust mdps with global convergence guarantee. In International Conference on Machine Learning , pages 35763-35797. PMLR, 2023.
- [45] Zhongchang Sun, Sihong He, Fei Miao, and Shaofeng Zou. Policy optimization for robust average reward mdps. Advances in Neural Information Processing Systems , 37:17348-17372, 2024.
- [46] Shaocong Ma, Ziyi Chen, Yi Zhou, and Heng Huang. Rectified robust policy optimization for modeluncertain constrained reinforcement learning without strong duality. Transactions on Machine Learning Research , 2025.
- [47] Ruihong Huang and Tomas Polak. Lobster: Limit order book reconstruction system. Available at SSRN 1977207 , 2011.
- [48] Jean-Philippe Bouchaud, Julien Kockelkoren, and Zoltan Eisler. The price impact of order book events: Market orders, limit orders and cancellations. Quantitative Finance , 9(3):283-297, 2009.
- [49] Jim Gatheral, Alexander Schied, and Aleksey Slynko. Transient linear price impact and fredholm integral equations. Mathematical Finance , 22(3):445-474, 2012.
- [50] Leonardo Berti, Bardh Prenkaj, and Paola Velardi. Trades: Generating realistic market simulations with diffusion models. arXiv preprint arXiv:2502.07071 , 2025.
- [51] Robert Almgren, Chee Thum, Emmanuel Hauptmann, and Hong Li. Direct estimation of equity market impact. Risk , 18(7):58-62, 2005.
- [52] Anastasia Bugaenko. Empirical study of market impact conditional on order-flow imbalance. arXiv preprint arXiv:2004.08290 , 2020.

- [53] Michaël Karpe, Jin Fang, Zhongyao Ma, and Chen Wang. Multi-agent reinforcement learning in a realistic limit order book market simulation. In Proceedings of the first ACM international conference on AI in finance , pages 1-7, 2020.
- [54] Andrew Todd, Peter Beling, William Scherer, and Steve Y. Yang. Agent-based financial markets: A review of the methodology and domain. In 2016 IEEE Symposium Series on Computational Intelligence (SSCI) , pages 1-5. IEEE, 2016.
- [55] Shen Gao, Yuntao Wen, Minghang Zhu, Jianing Wei, Yuhan Cheng, Qunzi Zhang, and Shuo Shang. Simulating financial market via large language model based agents. arXiv preprint arXiv:2406.19966 , 2024.
- [56] Junjie Li, Yang Liu, Weiqing Liu, Shikai Fang, Lewen Wang, Chang Xu, and Jiang Bian. Mars: a financial market simulation engine powered by generative foundation model. In The Thirteenth International Conference on Learning Representations , 2025.
- [57] Martin D. Gould, Mason A. Porter, Stacy Williams, Mark McDonald, Daniel J. Fenn, and Sam D. Howison. Limit order books. Quantitative Finance , 13(11):1709-1742, 2013.
- [58] Avraam Tsantekidis, Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis. Forecasting stock prices from limit order book using convolutional neural networks. In 2017 IEEE 19th Conference on Business Informatics (CBI) , volume 1, pages 7-12. IEEE, 2017.
- [59] Vincent Ragel. Reinforcement Learning for systematic market making strategies . PhD thesis, Université Paris-Saclay, 2024.
- [60] Zhenglong Li, Vincent Tam, and Kwan L. Yeung. Developing a multi-agent and self-adaptive framework with deep reinforcement learning for dynamic portfolio risk management. arXiv preprint arXiv:2402.00515 , 2024.
- [61] Mohammed Amin Abdullah, Hang Ren, Haitham Bou Ammar, Vladimir Milenkovic, Rui Luo, Mingtian Zhang, and Jun Wang. Wasserstein robust reinforcement learning. arXiv preprint arXiv:1907.13196 , 2019.
- [62] Yufei Kuang, Miao Lu, Jie Wang, Qi Zhou, Bin Li, and Houqiang Li. Learning robust policy against disturbance in transition dynamics via state-conservative policy optimization. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36-7, pages 7247-7254, 2022.
- [63] John C Duchi and Hongseok Namkoong. Learning models with uniform performance via distributionally robust optimization. The Annals of Statistics , 49(3):1378-1406, 2021.
- [64] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction , volume 1. MIT press Cambridge, 1998.
- [65] Yanwei Jia and Xun Yu Zhou. Policy gradient and actor-critic learning in continuous time and space: Theory and algorithms. Journal of Machine Learning Research , 23(275):1-50, 2022.
- [66] Huaqing Xiong, Tengyu Xu, Lin Zhao, Yingbin Liang, and Wei Zhang. Deterministic policy gradient: Convergence analysis. In Uncertainty in Artificial Intelligence , pages 2159-2169. PMLR, 2022.
- [67] Kaiqing Zhang, Zhuoran Yang, and Tamer Basar. Networked multi-agent reinforcement learning in continuous spaces. In 2018 IEEE conference on decision and control (CDC) , pages 2771-2776. IEEE, 2018.
- [68] Bin Hu, Kaiqing Zhang, Na Li, Mehran Mesbahi, Maryam Fazel, and Tamer Ba¸ sar. Toward a theoretical foundation of policy optimization for learning control policies. Annual Review of Control, Robotics, and Autonomous Systems , 6(1):123-158, 2023.
- [69] Elena Smirnova, Elvis Dohmatob, and Jérémie Mary. Distributionally robust reinforcement learning. arXiv preprint arXiv:1902.08708 , 2019.
- [70] Jose Blanchet, Miao Lu, Tong Zhang, and Han Zhong. Double pessimism is provably efficient for distributionally robust offline reinforcement learning: Generic algorithm and robust partial coverage. Advances in Neural Information Processing Systems , 36:66845-66859, 2023.
- [71] Yiting He, Zhishuai Liu, Weixin Wang, and Pan Xu. Sample complexity of distributionally robust offdynamics reinforcement learning with online interaction. In Forty-second International Conference on Machine Learning , 2025.

- [72] Yan Li and Alexander Shapiro. Rectangularity and duality of distributionally robust markov decision processes. arXiv preprint arXiv:2308.11139 , 2023.
- [73] Zhishuai Liu and Pan Xu. Distributionally robust off-dynamics reinforcement learning: Provable efficiency with linear function approximation. In International Conference on Artificial Intelligence and Statistics , pages 2719-2727. PMLR, 2024.
- [74] Miao Lu, Han Zhong, Tong Zhang, and Jose Blanchet. Distributionally robust reinforcement learning with interactive data collection: Fundamental hardness and near-optimal algorithms. Advances in Neural Information Processing Systems , 37:12528-12580, 2024.
- [75] Wolfram Wiesemann, Daniel Kuhn, and Berç Rustem. Robust markov decision processes. Mathematics of Operations Research , 38(1):153-183, 2013.
- [76] Carlo Zarattini, Andrew Aziz, and Andrea Barbon. Beat the market: An effective intraday momentum strategy for s&amp;p500 etf (spy). Available at SSRN 4824172 , 2024.
- [77] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540 , 2016.
- [78] Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulao, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032 , 2024.
- [79] Stephen Boyd and Lieven Vandenberghe. Convex Optimization . Cambridge University Press, 2004.
- [80] Stephen Roman. Advanced Linear Algebra . Graduate Texts in Mathematics. Springer, third edition, 2008.
- [81] Neal Parikh, Stephen Boyd, et al. Proximal algorithms. Foundations and trends® in Optimization , 1(3):127-239, 2014.
- [82] Jonathan M. Borwein and Adrian S. Lewis. Convex Analysis and Nonlinear Optimization: Theory and Examples . Springer, 2 edition, 2006.
- [83] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- [84] Shaocong Ma, Yi Zhou, and Shaofeng Zou. Variance-reduced off-policy tdc learning: Non-asymptotic convergence analysis. Advances in neural information processing systems , 33:14796-14806, 2020.
- [85] Shaocong Ma, Ziyi Chen, Yi Zhou, and Shaofeng Zou. Greedy-gq with variance reduction: Finite-time analysis and improved complexity. arXiv preprint arXiv:2103.16377 , 2021.
- [86] Yi Zhou, Shaocong Ma, et al. Stochastic optimization methods for policy evaluation in reinforcement learning. Foundations and Trends® in Optimization , 6(3):145-192, 2024.
- [87] Shengbo Wang, Nian Si, Jose Blanchet, and Zhengyuan Zhou. Sample complexity of variance-reduced distributionally robust q-learning. Journal of Machine Learning Research , 25(341):1-77, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, we have clearly listed the main contributions in the introduction section.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have discussed the limitation in the appendix.

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

Justification: We have provided the full proof in the appendix.

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

Justification: We include the source code and reproducing instructions in the supplementary material.

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

Justification: We include the source code and reproducing instructions in the supplementary material. The data is accessed using the third-party API.

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

Answer:[Yes]

Justification: We explicitly specify the details in the code and supplementary material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Our evaluation environment is deterministic; that is, there is no randomness.

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

Justification: Included in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed this code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We have included this in the conclusion section.

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

Justification: The source code is for reviewing purpose only and is not considered as the released new asset.

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

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

## Table of Contents

| A   | Backgrounds                                                | Backgrounds                                                |   22 |
|-----|------------------------------------------------------------|------------------------------------------------------------|------|
|     | A.1                                                        | Common Uncertainty Sets in the Literature                  |   22 |
|     | A.2                                                        | The Signed Permutation Group . . . . . .                   |   23 |
| B   | Worst-Case Uncertainty under ℓ p -Ellipse Uncertainty Sets | Worst-Case Uncertainty under ℓ p -Ellipse Uncertainty Sets |   24 |
|     | B.1                                                        | Supporting Lemmas . . . . . . . . . . . .                  |   24 |
|     | B.2                                                        | Implicit Solution . . . . . . . . . . . . . .              |   31 |
|     | B.3                                                        | Explicit Solution . . . . . . . . . . . . . .              |   32 |
| C   | Experiment Details                                         | Experiment Details                                         |   35 |
|     | C.1                                                        | Hardware and System Environment . . . .                    |   35 |
|     | C.2                                                        | Task Descriptions . . . . . . . . . . . . .                |   35 |
|     | C.3                                                        | Reinforcement Learning Framework . . .                     |   36 |
|     | C.4                                                        | Parameter Details . . . . . . . . . . . . .                |   37 |
|     | C.5                                                        | Omitted Visualization . . . . . . . . . . .                |   37 |
|     | C.6                                                        | Other Implementation Details . . . . . . .                 |   37 |
| D   | Limitations                                                | Limitations                                                |   38 |
| E   | Additional Supplementary Materials                         | Additional Supplementary Materials                         |   38 |
|     | E.1                                                        | Convergence of Robust TD-Learning . . .                    |   38 |
|     | E.2                                                        | Parameter Setting and Grid-Search . . . .                  |   39 |
|     | E.3                                                        | Other Implementation Details . . . . . . .                 |   40 |

## A Backgrounds

## A.1 Common Uncertainty Sets in the Literature

When evaluating the robust Bellman operator

<!-- formula-not-decoded -->

the uncertainty set U ' t P s,a u p s,a qP S ˆ A plays a crucial role. Certain structures in U enable efficient robust policy evaluation. Below, we summarize several widely adopted constructions.

f -divergence The f -divergence family [62, 63] generalizes statistical distances between distributions using a convex function f : p 0 , 8q Ñ R with f p 1 q ' 0 . For distributions P and P 0 such that P ! P 0 , the f -divergence is defined as

<!-- formula-not-decoded -->

Special cases include the Kullback-Leibler divergence ( f p t q ' t log t ), total variation distance ( f p t q ' 1 2 | t ´ 1 | ), and χ 2 -divergence ( f p t q ' p t ´ 1 q 2 ). In robust RL, the f -divergence ball around the nominal transition kernel P 0 p¨| s, a q yields the uncertainty set

<!-- formula-not-decoded -->

The inner minimization in Eq. (7) becomes a distributionally robust optimization problem over P 0 . As the result, the robust policy evaluation under the KL-divergence often requires to repeatedly solve an additional convex program.

R -contamination Model The R -contamination model [31] assumes that the true transition kernel lies within a convex mixture of the nominal model P 0 and an arbitrary distribution P 1 :

<!-- formula-not-decoded -->

where R P r 0 , 1 s quantifies the contamination level. This model leads to closed-form solutions for the robust Bellman operator, with the worst-case distribution taking mass at the minimum of the value function v . As a result, this setup enables efficient and model-free learning algorithms, including robust variants of Q-learning, TD learning, and policy gradients. It is particularly well-suited for online learning, where P 0 evolves with the observed data.

ℓ p -norm These sets constrain the deviation from the nominal model P 0 p¨| s, a q using the ℓ p -norm:

<!-- formula-not-decoded -->

When p ' 1 , the constraint corresponds to total variation distance, while p ' 8 bounds the largest single-coordinate deviation. These sets are commonly used due to their interpretability and explicit analytical solution given in [39, 40]. However, their axis-aligned geometry can lead to overly conservative policies in high dimensions.

Integral Probability Metric (IPM) The IPM measures the discrepancy between distributions through expectations over a function class F :

<!-- formula-not-decoded -->

The corresponding uncertainty sets are:

<!-- formula-not-decoded -->

The IPM-based uncertainty sets are particularly useful when the state space is extremely large or continuous, as explicitly solve the minimization problem in Eq. (7) does not requires to access values at all states [43].

Wasserstein Distance The Wasserstein distance [61], grounded in optimal transport theory, accounts for the geometry of the state space. Given a cost function d : S ˆ S Ñ R ` and p ě 1 , the p -Wasserstein distance between P and P 0 is

<!-- formula-not-decoded -->

where Γ p P , P 0 q denotes the set of joint distributions (couplings) with marginals P and P 0 . The uncertainty set is then

<!-- formula-not-decoded -->

Despite their strong theoretical properties, solving the inner minimization often requires dual formulations or approximation techniques.

General Uncertainty Sets There are also many techniques developed to handle the situation where the uncertainty set is general. For example, However, [44] proposes a bilevel approach that iteratively solves the worst-case transition kernel to approximate the robust value function. However, as demonstrated in [28, 44, 75], solving robust RL problems in the general case is NP-hard.

## A.2 The Signed Permutation Group

The signed permutation group plays a central role in characterizing the symmetry structure of uncertainty sets in our robust RL framework. Informally, this group consists of all matrices in R | S |ˆ| S | satisfying the following conditions:

1. Each entry is either 0 , 1 , or ´ 1 .

2. Each row and each column contains exactly one nonzero entry.

In other words, every element of this group is a matrix obtained by permuting the standard basis vectors of R | S | and possibly flipping their signs. Each such matrix can be expressed as the product DP , where D is a diagonal matrix with diagonal entries in t˘ 1 u , and P is a permutation matrix representing an element of the symmetric group S | S | . This leads to the following formal definition:

Definition A.1 (Signed Permutation Group) . Let S | S | denote the permutation group over | S | elements, and let p Z 2 q | S | be the direct sum of | S | copies of the cyclic group of order 2 . Then the signed permutation group , denoted by Signed p S | S | q , is the semidirect product:

<!-- formula-not-decoded -->

where the action of S | S | on p Z 2 q | S | is given by permuting the order.

In this work, we define the 'symmetry' of sets as the invariance under the group action induced by the signed permutation group. Specifically, we say a set B Ă R | S | is symmetric under a group action by G if g ¨ B Ď B for all g P G (or G ¨ B : ' t g ¨ b u g P G,b P B Ď B ). This notion of symmetry leads to the following structural property of ℓ p -norm balls:

Proposition A.2. Let B : ' t B p p β qu p ě 1 ,β ě 0 be the family of ℓ p -norm balls, where B p p β q : ' t u P R d | } u } p ď β u . If there exists a group G such that all elements in B are symmetric under the group action by G , then G must be isomorphic to a subgroup of Signed p S d q .

Proof. All elements in B are symmetric under the group action by G ; that is, for every p ě 1 and every β ě 0 ,

<!-- formula-not-decoded -->

Therefore, the act g : R d Ñ R d is a norm-preserving bijection; by the Mazur-Ulam theorem, it must be affine. Then as it preserves 0 , it must be linear.

In particular, taking p ' 1 and β ' 1 , each g P G is an (invertible) linear isometry of the 1-norm unit ball whose extreme points are exactly

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Because a linear automorphism of a polytope must permute its extreme points, for each i and each g P G there must exist a sign ε i P t˘ 1 u and an index σ p i q P t 1 , . . . , d u such that

<!-- formula-not-decoded -->

Thus in the standard basis g is represented by a signed permutation matrix :

<!-- formula-not-decoded -->

where D ' diag p ε 1 , . . . , ε d q and P is the permutation matrix corresponding to σ P S d . Hence every g P G lies in the signed permutation group Signed p S d q . In other words G Ď Signed p S d q , which equivalently shows G is isomorphic to a subgroup of Signed p S d q .

This result provides a useful insight in designing the ℓ p -ellipse set: the signed permutation group is the largest group under which all ℓ p -norm balls are symmetric. Consequently, to construct a set with less symmetry than the standard ℓ p -norm balls (as we aim to do with our ℓ p -ellipse sets), it is necessary to enlarge the family B to include non-ball shapes.

## B Worst-Case Uncertainty under ℓ p -Ellipse Uncertainty Sets

## B.1 Supporting Lemmas

Definition B.1 (Minkowski sum) . Given two sets A,B Ď R d , the Minkowski sum ` : 2 R d ˆ 2 R d Ñ 2 R d is defined as

<!-- formula-not-decoded -->

Definition B.2 (Fenchel conjugate [79]) . Let g : R d Ñ R be a function over R d . Its Fenchel conjugate is denoted as g ˚ : R d Ñ R and is defined as

<!-- formula-not-decoded -->

We include the following famous Hölder's inequality without providing the proof.

Lemma B.3 (Hölder's inequality) . Let p, q P r 1 , `8s satisfy 1 p ` 1 q ' 1 . For every f, g P R d

<!-- formula-not-decoded -->

Moreover, equality holds if and only if

<!-- formula-not-decoded -->

where J p p f q denotes any ℓ p -unit vector that attains the maximum inner product with f :

<!-- formula-not-decoded -->

Proof. The proof can be found in [80].

Lemma B.4. For any x, y P R d and radii r, s ě 0 , the Minkowski sum of the two ℓ p -norm balls ( p ě 1 )

<!-- formula-not-decoded -->

is again a ball, namely

<!-- formula-not-decoded -->

Proof. For convenience, we omit p at the subscript in this proof. It suffices to show two inclusions.

- B p x, r q ` B p y, s q Ď B p x ` y, r ` s q .

Take any

<!-- formula-not-decoded -->

Then by the triangle inequality,

<!-- formula-not-decoded -->

Hence z P B p x ` y, r ` s q , proving the first inclusion.

- B p x ` y, r ` s q Ď B p x, r q ` B p y, s q .

Let z P B p x ` y, r ` s q , so } z ´p x ` y q} ď r ` s . Set

<!-- formula-not-decoded -->

Define u ' x ` α ` z ´p x ` y q ˘ and v ' y ` β ` z ´p x ` y q ˘ . Then

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus u P B p x, r q and v P B p y, s q , so z ' u ` v P B p x, r q ` B p y, s q . This proves the reverse inclusion.

Combining (1) and (2) gives the desired equality B p x, r q ` B p y, s q ' B ` x ` y, r ` s ˘ .

Lemma B.5. Let g p u q ' ř N i ' 1 } u ´ u i } p . Then the Fenchel conjugate of g : R d Ñ R is

<!-- formula-not-decoded -->

.

Proof. The key is that g p u q ' ř N i ' 1 } u ´ u i } p is a sum of N 'shifted-norms,' and the Fenchel conjugate (Definition B.2) of a sum is the infimal convolution of the conjugates. We proceed in two steps.

- Let f p x q ' } x } p be a single ℓ p -norm mapping and q be the dual of q satisfying 1 p ` 1 q ' 1 By [79], it is standard that

<!-- formula-not-decoded -->

Then we consider its 'shift' by u i . Let f i p u q : ' } u ´ u i } p . By the translation rule for Fenchel conjugates [81, 82],

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- As the Fenchel conjugate of } u ´ u i } p has been evaluated, the Fenchel conjugate of their sum is given by

<!-- formula-not-decoded -->

.

Applying this infimal convolution requires each component f i : R d Ñ R is proper, convex, and lower semicontinuous, which is automatically satisfied by the ℓ p -norm.

Lemma B.6. Let p o , λ o , ˜ ω o P R d and γ o ě 0 be given constants. Let 1 p ` 1 q ' 1 . Then the optimization problem

<!-- formula-not-decoded -->

has the unique minimizer given by

<!-- formula-not-decoded -->

where d represent the coordinate-wise product and | ¨ | is the coordinate-wise absolute value. The optimal value is solved as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To make the Hölder's inequality achieve the equality, we choose w ' ´ tJ q p p o ` λ o q for some t , where J q is the q -unit vector defined in Eq. (8). Then by letting } w } q ' C , we obtain the final result.

Proof. By Hölder's inequality,

Lemma B.7. Suppose that v P R d , t u i u N i ' 1 Ă R d , and µ ě 0 , and the norm exponent 1 p ` 1 q ' 1 (for p ě 1 ). Let w ' v ` µ 1 and C : ' } w } q . Then the optimization problem

<!-- formula-not-decoded -->

is feasible and solves the minimizer

<!-- formula-not-decoded -->

for i ' 1 , 2 , . . . , N , where ˜ λ ˚ P R d is given by

<!-- formula-not-decoded -->

Proof. We consider the constrained Lagrangian function

<!-- formula-not-decoded -->

where } w i } q ď C : ' } w } q . Let the dual function ϑ p ˜ λ q : ' inf t w i u ˜ L pt w i u , ˜ λ q . Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that w ' v ` µ 1 P R d is a given vector. Denote

<!-- formula-not-decoded -->

Put it back to w i, ˚ , we obtain the final result.

Lemma B.8. Suppose that t u i u N i ' 1 Ă R d , β ě 0 , and v P R d . Let

<!-- formula-not-decoded -->

where p P r 1 , `8s and 1 p ` 1 q ' 1 . Then

<!-- formula-not-decoded -->

By Lemma B.6, it solves

As the result,

where

<!-- formula-not-decoded -->

Consequently, the optimal p λ ˚ , µ ˚ q of achieving sup λ,µ φ p λ, µ q is given by

<!-- formula-not-decoded -->

Proof. We take the transformation w : ' v ` µ 1 . Then

<!-- formula-not-decoded -->

For any convex g , by the definition of Fenchel conjugate (Definition B.2), we have

<!-- formula-not-decoded -->

Here g p u q ' ř i } u ´ u i } p . Its conjugate is given by Lemma B.5,

<!-- formula-not-decoded -->

where 1 p ` 1 q ' 1 . We put it back to inf u ' w J u ` λg p u q ‰ ' ´ λg ˚ ` ´ w { λ ˘ , which leads to

<!-- formula-not-decoded -->

where } w λ } q ď 1 . As the result, we take w i ' ´ λz i to obtain

<!-- formula-not-decoded -->

Thus, the full dual becomes

<!-- formula-not-decoded -->

For a fixed µ , we need λ ě } w } q to keep φ finite, and φ p λ, µ q is decreasing in λ . Hence the best choice is

<!-- formula-not-decoded -->

giving

<!-- formula-not-decoded -->

It leads to another optimization problem inf ř i w i ' v ` µ 1 , } w i } q ď} v ` µ 1 } q ř N i ' 1 w J i u i . We construct another Lagrangian function to solve it. Denote w i, ˚ as the minimizer given by Lemma B.7. Then we obtain

<!-- formula-not-decoded -->

It recovers the optimal dual variable z is given by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.9. Suppose that t u i u 2 i ' 1 Ă R , λ ą 0 , β ě 0 , w P R is non-zero, q P r 1 , `8s , and 1 p ` 1 q ' 1 . Define

<!-- formula-not-decoded -->

Then when | w | ď 2 λ , the problem is solved as

<!-- formula-not-decoded -->

Proof. We start from the general case. Define f p u q ' uw ` λ ř N i ' 1 | u ´ u i | . If | w | ď λN , then the sub-gradient is given by

<!-- formula-not-decoded -->

Write t u i u N i ' 1 Ă R in the increasing order:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is the explicit solution. To prove it, we consider u P p u p k ˚ q , u p k ˚ ` 1 q q ; it is larger than exactly k ˚ u i 's. That is,

<!-- formula-not-decoded -->

Whenever u ă u p k ˚ q , the sign of sub-gradient becomes negative. As the result, f p u q is decreasing when u ă u p k ˚ q then increasing when u ą u p k ˚ q . Now we set N ' 2 . The problem gives

<!-- formula-not-decoded -->

When w ě 0 , sign p w q ' ` 1 and

<!-- formula-not-decoded -->

When w ă 0 , sign p w q ' ´ 1 and

<!-- formula-not-decoded -->

Therefore, this formula recovers the original general case solution. Putting it back to φ solves this problem.

The following lemma connects the sum-of-distance description to the quadratic form of an ellipse.

Lemma B.10. Suppose that t u i u 2 i ' 1 Ă R d , β ě 0 , and β ą } u 1 ´ u 2 } 2 . The ellipse set is given by

<!-- formula-not-decoded -->

Then there exists a matrix A such that

<!-- formula-not-decoded -->

where ¯ u : ' u 1 ` u 2 2 . More explicitly, the matrix A has the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define k ˚ : ' r N ´ w { λ 2 s . Then

Proof. Define

<!-- formula-not-decoded -->

and decompose each u P R d by

Then u 1 ' ¯ u ´ fe , u 2 ' ¯ u ` fe , and

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Now decompose x into so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The inequality a p α ` f q 2 ` ξ 2 ` a p α ´ f q 2 ` ξ 2 ď 2 a is equivalent, after two squarings, to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, observe that so

Setting A ' 1 a 2 ee J ` 1 b 2 p I ´ ee J q implies p u ´ ¯ u q J A p u ´ ¯ u q ď 1 . It exactly characterizes t u | } u ´ u 1 } 2 `} u ´ u 2 } 2 ď β u . Simplifying the form of A leads to

<!-- formula-not-decoded -->

This completes the proof.

Lemma B.11. Let v P R d , let A P R d ˆ d be symmetric positive definite, and let ¯ p P R d . Define

<!-- formula-not-decoded -->

Further set

<!-- formula-not-decoded -->

If δ 2 ă α , then F attains a unique maximizer

<!-- formula-not-decoded -->

Proof. We begin by computing the derivative of

b

<!-- formula-not-decoded -->

Using the notation α ' 1 J A ´ 1 1 , β ' v J A ´ 1 1 , γ ' v J A ´ 1 v , and δ ' 1 J ¯ p , one checks

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so

<!-- formula-not-decoded -->

Setting F 1 p µ q ' 0 gives the stationarity condition

<!-- formula-not-decoded -->

which upon squaring yields the quadratic equation

<!-- formula-not-decoded -->

Let δ 2 ă α . Here α ´ δ 2 ą 0 , so dividing by α ´ δ 2 gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

whose two roots are

One verifies by inspecting lim µ Ñ˘8 F 1 p µ q ' δ ¯ ? α that exactly the ' ` ' choice yields a change of sign from ` to ´ , and hence is the unique global maximizer.

## B.2 Implicit Solution

In this subsection, we recap and prove the full version of Theorem 3.4.

Theorem B.12. Let d : ' | S | be the cardinal of state space and 1 p ` 1 q ' 1 for p, q P r 1 , `8s . Suppose that t u i u N i ' 1 Ă R d for each p s, a q P S ˆ A , the uncertainty size β ě 0 , and v P R d . The solution of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is given by where

<!-- formula-not-decoded -->

Remark (The procedure of solving u ˚ ) . To obtain u ˚ , it suffices to solve µ ˚ and λ ˚ as v and all u i 's have been given. The first step is to solve

<!-- formula-not-decoded -->

for i ' 1 , 2 , . . . , N . Both variables depend on the variable µ and other values are known. The next step is to solve

<!-- formula-not-decoded -->

Once p µ ˚ , λ ˚ q is solved, the primal variable µ ˚ is obtained immediately.

Proof. Our goal is to solve the following constrained optimization problem:

<!-- formula-not-decoded -->

As it is a constrained optimization problem, the standard approach of solving this problem is using Lagrangian multipliers. we introduce the Lagrangian multipliers λ ě 0 for the inequality and µ P R for the equality. The Lagrangian function is

<!-- formula-not-decoded -->

with λ ě 0 , µ P R . Because this optimization problem is a standard convex optimization problem with satisfying the Slater's condition, we have the strong duality

<!-- formula-not-decoded -->

Then we turn the original optimization problem into solving its dual-form problem. We let the dual function be φ p λ, µ q : ' inf u L p u, λ, µ q . Then

<!-- formula-not-decoded -->

The above formulation plays the crucial role in our proof. For the implicit solution, we will follow Lemma B.8 to complete the remaining calculation. For the explicit solution, this dual form can be significantly simplified in some cases.

By Lemma B.8, the dual form can be simplified as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The optimal primary variable is given as

<!-- formula-not-decoded -->

## B.3 Explicit Solution

In this subsection, we recap and prove the full version of Theorem 3.5.

Theorem B.13. Let d : ' | S | be the cardinal of state space. Suppose that t u i u N i ' 1 Ă R d for each p s, a q P S ˆ A , the uncertainty size β ě 0 , and v P R d . The optimization problem defined by Eq. (4) is explicitly solved in the following cases:

- (a) Let N ' 1 . then

<!-- formula-not-decoded -->

- (b) Let p ' 1 and N ' 2 . Define ¯ u ' u 1 ` u 2 2 ,

<!-- formula-not-decoded -->

Suppose that β ą } u 2 ´ u 1 } 1 . Then the explicit solution to the optimization problem Eq. (4) is given as

<!-- formula-not-decoded -->

- (c) Let p ' 2 and N ' 2 . Define ¯ u ' u 1 ` u 2 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Suppose that β ą } u 2 ´ u 1 } 2 . Then the explicit solution to the optimization problem Eq. (4) is given as

<!-- formula-not-decoded -->

Proof. We follow the standard routine used in proving Theorem B.12.

- (a) When N ' 1 the objective optimization problem is given by

<!-- formula-not-decoded -->

We take the transformation u 1 ' u ´ u 1 . It still satisfies 1 J u 1 ' 0 . Then the problem become

<!-- formula-not-decoded -->

This transformation has turned this problem into the standard ℓ p -norm structure, which has been explicitly solved in [39, 40]. The optimal u 1 is given for arbitrary p ě 1 as

<!-- formula-not-decoded -->

where µ ˚ ' arg min µ P R } v ` µ 1 } q . As the result,

<!-- formula-not-decoded -->

- (b) When p ' 1 , we define the order

<!-- formula-not-decoded -->

We follow the same routine as Theorem B.12 and derive the dual function φ p λ, µ q :

<!-- formula-not-decoded -->

where p i q decomposes the ℓ 1 -norm by coordinates. When N ' 2 , by Lemma B.9, the dual function is solved as

<!-- formula-not-decoded -->

As the smaller λ is, the larger φ p λ, µ q is. It achieves the supremum at λ ˚ ' max j v j ` µ 2 ' 1 2 } v ` 1 µ } 8 . Then we solve

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

As β ´} u 2 ´ u 1 } 1 ą 0 , it solves

<!-- formula-not-decoded -->

Now we consider the KKT condition of the original Lagrangian function. We solve

<!-- formula-not-decoded -->

For inactive coordinate, the optimal value is attained for arbitrary u j P p u p 1 q j , u p 2 q j q ; in these cases, we simply take u j ' u 1 j ` u 1 j 2 . There are exactly two active coordinates matching the corner-case condition | v j ` µ ˚ | ' 2 λ ˚ : j ˚ ' arg max v j and j ˚ ' arg min v j . In these cases we take u j as u 1 j ` u 1 j 2 subtracting a drift. It finally solves

<!-- formula-not-decoded -->

The magnitude coefficient is used to ensure that u ˚ belongs to } u ˚ } 1 ď β .

- (c) By Lemma B.10, there exists a semi-positive definite matrix A such that

<!-- formula-not-decoded -->

where ¯ u : ' u 1 ` u 2 2 . As the result, the objective optimization problem can be simplified as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We follow the same routine as Theorem B.12 and derive the dual function φ p λ, µ q :

<!-- formula-not-decoded -->

where the infimum is attained at u ˚ ' ¯ u ´ 1 2 λ A ´ 1 p v ` µ 1 q . Then

<!-- formula-not-decoded -->

where (i) applies the optimal choice λ ˚ p µ q ' 1 2 a p v ` µ 1 q J A ´ 1 p v ` µ 1 q and µ ˚ is given by Lemma B.11 to solve this maximization problem. As the result,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the matrix A is determined by t u 1 , u 2 u 2 i ' 1 and the vector v in Lemma B.10. We recall that 1 J ¯ u ' 0 . Therefore, µ ˚ is further simplified as

<!-- formula-not-decoded -->

It completes the proof in the part (c).

## C Experiment Details

In this section, we include the omitted details of Section 4. All source codes and hyper-parameter settings are available in the supplementary materials.

## C.1 Hardware and System Environment

We conducted our experiments on a laptop running Windows 11 Home. The device is equipped with 32GB of RAM, 1TB SSD, an AMD Ryzen 9 7940HS processor and a NVIDIA GeForce RTX 4070 Laptop GPU. Our implementation was tested using Python version 3.10.10. Additional dependencies are listed in the supplementary requirements.txt file.

The actual hardware requirement for running our implementation is significantly lower than the specification listed above. Most experiments can be reproduced on consumer-grade machines with 8-16GB RAM and any CUDA-compatible NVIDIA GPU.

## C.2 Task Descriptions

Multi-Asset Portfolio Rebalancing This task captures the setting where an institutional investor reallocates capital across multiple assets over discrete time intervals to maintain a desired risk-return profile. It reflects strategic portfolio management, such as end-of-day rebalancing or tactical asset allocation. The task emphasizes robustness to market impact under varying capital scales, which is critical for large-volume institutional strategies. In our experiment, we select five representative ETFs, SPY, TLT, GLD, EFA, and VNQ, and use historical data from May 9, 2021 to May 9, 2022 for training. Evaluation is conducted on the out-of-sample period from June 9 to December 9, 2022.

Table 2: Example of ask-side execution from an AMZN order book snapshot (21 June 2012). This table shows how our simulator determines the execution price for a market buy order. Instead of using the last trade price, the simulator consumes liquidity by filling shares at each ask level in order, starting from the best price. It calculates the VWAP based on the prices and quantities filled, and uses this VWAP as the final executed price.

| Level     | Price (USD)   | Depth (sh)   |   Exec. 100 sh |   Exec. 1 000 sh |
|-----------|---------------|--------------|----------------|------------------|
| Level 1   | 223.95        | 100          |         100    |           100    |
| Level 2   | 223.99        | 100          |           0    |           100    |
| Level 3   | 224.00        | 220          |           0    |           220    |
| Level 4   | 224.25        | 100          |           0    |           100    |
| Level 5   | 224.40        | 547          |           0    |           480    |
| VWAP paid | -             | -            |         223.95 |           224.21 |

Single-Asset Intra-Day Trading This task models the decision-making process of an agent that repeatedly buys or sells a single asset over short time intervals, such as minutes or seconds. It reflects the setting of mid-frequency or algorithmic trading, where even small trades can shift market prices. The goal is to maximize the risk-adjusted cumulative return while accounting for execution slippage, making it a standard benchmark for evaluating RL-based trading strategies. We follow the exactly same training and evaluation period as the multi-asset portfolio rebalancing task. In our experiments, we use minute-level historical data for the assets including META, MSFT, and SPY. The observation includes stock price, trading volume, implied volatility, and current portfolio return. During evaluate, we reconstruct the LOB dynamics using the tick-level trade data; the execution price is then simulated as illustrated in Table 2. Same as the multi-asset portfolio rebalancing experiment, the training data covers the period from May 9th, 2021 to May 9th, 2022, and the evaluation period is from June 9 to December 9, 2022. All data are accessed via the Polygon.io Stock Market API.

## C.3 Reinforcement Learning Framework

We implemented a Gym-like RL trading environment [77, 78] using the historical data including the stock price, trading volume, the implied volatility, and the current portfolio return. Instead of using a single timestep data, we set lookback\_period ' 30 to account for 30 previous timestep; as the result, the observed state contains lookback\_period ˆ 4 ' 120 dimension. Given the state described above, we assign the transition probability from the current state s t to the next state s t ` 1 as the probability 1 , independent with the action. The action is implemented as a single float value that determines the position size relative to the maximum possible position size based on the available capital. The reward is a Sharpe-like ratio that consists of two components:

<!-- formula-not-decoded -->

where ε ą 0 is a small constant to ensure numerical stability, and δ is a tunable coefficient controlling the strength of the transaction cost penalty. The first term encourages higher risk-adjusted return, while the second term discourages frequent or drastic position shifts, which aligns with practical trading considerations under market impact or slippage. We also include 0 . 1% transaction cost throughout the experiment.

Uncertainty Restriction to Execution Prices Instead of perturbing the whole transition probability, we restrict the perturbation on the execution price. The formulation of restriction is given as follows: we denote the state s t as two components p p t , f t q , and we hope perturb the transition on p t only. We re-write the transition probability using the conditional probability

<!-- formula-not-decoded -->

Then we set the uncertainty u as the perturbation on the kernel P p p t ` 1 | s t , a t q instead of the whole transition kernel P p s t ` 1 | s t , a t q .

Momentum Strategy The momentum strategy is a classical intra-day minute-level algorithmic trading method that exploits intraday time-series momentum to generate trading signals. We follow the classical implementation presented by [76]. The core idea of the momentum strategy is based on the empirical observation where assets that have performed well in the recent past are more likely to continue performing well in the near future, while poorly performing assets are likely to continue underperforming.

## C.4 Parameter Details

Detailed descriptions of all parameters are provided in the separate supplementary materials.

## C.5 Omitted Visualization

In Figure 3, we have visualized the return curve of the META stock. In this subsection, we include the visualization of the other two assets.

Figure 5: Performance comparison of trading strategies on the MSFT stock and SPY ETF.

<!-- image -->

## C.6 Other Implementation Details

In this subsection, we discuss the practical implementation considerations and provide the details for reproducing our experiments. The source codes are also provided in the supplementary material.

RL Algorithm and Value Function Approximations We use the standard Robust Actor-Critic algorithm [43] to train the RL agent with replacing its IPM uncertainty set or doubly-sampling uncertainty set with the ℓ p -ellipse uncertainty set and the ℓ p -norm uncertainty set. We adopt an Actor-Critic architecture with a shared feature extractor and separate heads for the actor and critic. The shared backbone consists of three fully connected layers with ReLU activations, mapping the (flattened) input state to a high-level representation. The actor head outputs the vector with the same dimension as the action through a two-layer MLP followed by a sigmoid/tanh activation, depending

on the underlying task. The critic head predicts the state value using a similar two-layer MLP. All linear layers are orthogonally initialized to promote stable training. When updating the Actor network, we apply the classical PPO algorithm [83].

Discretization of Execution Prices To apply the robust RL framework, we apply the discretization to the execution price. We introduce two additional hyper-parameter to control the discretization level: 2 N ` 1 represents the number of total discretization in the execution prices and δ represents the strength of each level. Given the execution price p , we have 2 N ` 1 potential execution prices r p ´ Nδ,.. . , p ´ δ, p, p ` δ, . . . , p ` Nδ s in total. This discretization approach allows us to directly apply the closed-form solution to solve the optimal u ˚ .

## D Limitations

Despite the promising results, this work has several limitations. First, the theoretical analysis largely relies on the assumption of finite state and action spaces for tractability, which is primarily due to the limited development of deep learning theory. Second, the market impact simulation during the evaluation stage, while based on trade-level data, remains an approximation and may lack accuracy for extreme volumes. In fact, when the volume is sufficiently high, it often triggers momentum-based strategies deployed by other institutions in the market, resulting in higher market impact than reflected VWAP. Third, although our experiments demonstrate the robustness of the ℓ p -ellipse uncertainty set under increasing volumes, we do not test its robustness under increasing trading frequency. Designing experiments in this setting is more challenging, as the behavior of RL agents varies significantly across different time scales.

## E Additional Supplementary Materials

In this additional supplementary material 3 , we provide several components omitted in our previous technical appendix. It includes: (1) The theoretical analysis of the robust TD-learning. (2) The further breakdown of the parameter setting and the grid-search strategy to optimize the optimal parameter. (3) The other implementation details that are related to our robust reinforcement learning setting.

## E.1 Convergence of Robust TD-Learning

Theorem (Convergence of Robust TD-Learning) . If the uncertainty set is defined as the ℓ p -ellipse uncertainty set, then the worst-case transition probability P ` p¨| s, a q can be represented as

<!-- formula-not-decoded -->

where β is the radius of the uncertainty set and u ˚ is solved in Theorem 3.4 and Theorem 3.5.

Remark. For non-robust policy evaluation, variance-reduction techniques are often employed to accelerate training [84-86]. For robust policy evaluation, the most recent progress can be found in [87]. For simplicity, we only present a naive robust TD-learning approach.

Proof. We apply the following TD-learning update rule:

<!-- formula-not-decoded -->

3 This section was originally submitted as a separate supplementary material. We include it here for the reader's convenience.

It is easy to observe that this update rule is equivalent to the non-robust TD-learning over the worst-case transition probability:

<!-- formula-not-decoded -->

where (i) applies the worst-case transition probability we have derived. Therefore, by applying existing TD-learning convergence analysis, we obtain that the convergence rate is also 1 ? K .

## E.2 Parameter Setting and Grid-Search

We apply grid-search to select hyperparameters to balance learning performance, computational efficiency, and robustness to uncertainty. In this section, we detail the architectural and training hyperparameters used across our three model variants: standard RL, elliptic uncertainty robust RL, and ball-shaped uncertainty robust RL.

## E.2.1 Model Architecture

All models share the same actor-critic network architecture, which consists of:

- Feature Extractor : Three fully-connected layers with ReLU activation functions, each containing 256 neurons.
- Actor Network : Two fully-connected layers (256 and 128 neurons) with the final layer using a tanh activation function to bound actions within r´ 1 , 1 s .
- Critic Network : Two fully-connected layers (256 and 128 neurons) with a linear output layer.
- Weight Initialization : Orthogonal initialization with a gain of ? 2 for linear layers to improve training stability.

This architeture is selected by default and we did not further tune the model architecture.

## E.2.2 Training Parameters

We employ the Proximal Policy Optimization (PPO) algorithm with the following hyperparameters:

Table 3: PPO Training Hyperparameters. The PPO clip parameter ϵ regulates the range of policy updates to ensure stability. Policy update epochs indicate how many times each batch is reused for learning. Batch size is the number of trajectories used per update.

| Parameter                | Standard RL   | Robust RL (Elliptic)   | Robust RL (Ball)   |
|--------------------------|---------------|------------------------|--------------------|
| Learning Rate            | 3 ˆ 10 ´ 4    | 3 ˆ 10 ´ 4             | 3 ˆ 10 ´ 4         |
| Discount Factor ( γ )    | 0.99          | 0.99                   | 0.99               |
| PPO Clip Parameter ( ϵ ) | 0.2           | 0.2                    | 0.2                |
| Policy Update Epochs     | 10            | 10                     | 10                 |
| Batch Size               | 64            | 64                     | 64                 |

We apply the grid-search on the learning rate and the PPO clip parameter; however, the grid-search is not applied to improve the performance. Instead, we apply the grid-search to identify a default

parameter group to ensure the algorithm convergence. In the remaining experiments, we will use the same parameter in the robust RL method for fair comparison.

## E.2.3 Robustness Parameters

For our robust RL implementations, we include another group of hyper-parameters.

Table 4: Robustness Parameters. The robust type is the string used in our codes to determine the different types of uncertainty sets. The parameter β is used to determine the size of uncertainty set.

| Parameter                       | Robust RL (Elliptic)   | Robust RL (Ball)   |
|---------------------------------|------------------------|--------------------|
| Robust Type                     | P1N2                   | P1                 |
| Uncertainty Set Parameter ( β ) | 1 ˆ 10 ´ 4             | 1 ˆ 10 ´ 4         |
| Uncertainty Dimension           | 3                      | 3                  |
| Epsilon                         | 1 ˆ 10 ´ 3             | 1 ˆ 10 ´ 3         |

We grid-search he parameter β from r 0 . 1 , 0 . 01 , 0 . 001 , 0 . 0001 , 0 . 00001 s and use the best result on the training period with adopting the early stopping. Then the best model is evaluated in the testing period to obtain the final performance.

## E.3 Other Implementation Details

The learning rate scheduler employs a ReduceLROnPlateau strategy with a factor of 0.5 and patience of 5 episodes. This adaptive approach reduces the learning rate when performance plateaus. Additionally, gradient clipping with a threshold of 0.5 is applied to prevent exploding gradients and ensure stable training.

The choice of β ' 1 ˆ 10 ´ 4 for both robust methods represents a balance between model performance and robustness. Too large a value would overly emphasize worst-case scenarios, while too small a value would not provide sufficient robustness against uncertainty. Through empirical testing, this value demonstrated the best trade-off between trading performance and resilience to market fluctuations.

Advantage normalization is applied prior to the policy update to stabilize training and improve convergence. The PPO clip parameter ϵ ' 0 . 2 prevents excessively large policy updates, maintaining proximity to the previous policy while allowing sufficient exploration.

Discretization of Execution Prices To apply the robust RL framework, we apply the discretization to the execution price. We introduce two additional hyper-parameter to control the discretization level: 2 N ` 1 represents the number of total discretization in the execution prices and δ represents the strength of each level. Given the execution price p , we have 2 N ` 1 potential execution prices r p ´ Nϵ,.. . , p ´ ϵ, p, p ` ϵ, . . . , p ` Nϵ s in total. This discretization approach allows us to directly apply the closed-form solution to solve the optimal u ˚ . The uncertainty dimension and the epsilon in the robustness parameters are used to determine the discretization level of execution prices. For example, when the discretization level is given by 3 , we have three potential execution prices r p ` ϵ, p, p ´ ϵ s , which also corresponds to the uncertainty dimension. In the nominal transition kernel, we assume the distribution over this discretized set as r 0 . 25 , 0 . 5 , 0 . 25 s , which provides sufficient flexibility for us to choose the uncertainty u .

Uncertainty Restriction to Execution Prices Instead of perturbing the whole transition probability, we restrict the perturbation on the execution price. The formulation of restriction is given as follows: we denote the state s t as two components p p t , f t q , and we hope perturb the transition on p t only. We re-write the transition probability using the conditional probability

<!-- formula-not-decoded -->

Then we set the uncertainty u as the perturbation on the kernel P p p t ` 1 | s t , a t q instead of the whole transition kernel P p s t ` 1 | s t , a t q . Given that the transition of the execution price p t ` 1 is discretized, the transition probability P p p t ` 1 | s t , a t q is a discrete distribution. For example, when taking the uncertainty dimension as 3 , we in fact obtain: P p p t ` 1 ' p ` ϵ | s t , a t q ' 0 . 25 ` u 1 ,

P p p t ` 1 ' p | s t , a t q ' 0 . 25 ` u 2 , and P p p t ` 1 ' p ´ ϵ | s t , a t q ' 0 . 25 ` u 3 . By default, in the P1N2-type robust RL, we choose the shift parameter u 1 ' r u 1 , u 2 , u 3 s ' r 0 . 1 ´ 1 { 3 , ´ 1 { 3 , ´ 1 { 3 s if the action is buying a stock and u 1 ' r u 1 , u 2 , u 3 s ' r´ 1 { 3 , ´ 1 { 3 , 0 . 1 ´ 1 { 3 s if the action is selling a stock.

Solving the Worst-Case Uncertainty After restricting the domain to the discrete execution price, we turn the original worst-case Bellman equation into the following form:

<!-- formula-not-decoded -->

Moreover, we use P p f 1 | p 1 , s, a q as a deterministic transition. It is common when training in the historical data, as (1) the observed feature f 1 comes from the existing dataset and the action a does not change its value (in our implementation, the action a will only affect the execution price), and (2) many features are directly calculated based on the current state, action, and the execution price such as the remaining cash. As the result, we obtain

<!-- formula-not-decoded -->

Here we still write u J V π for convenience, while the value function V π represents the 2 N ` 1 dimensional vector with value taken on s 1 ' p p 1 , f 1 q , where p 1 takes 2 N ` 1 values and f 1 is deterministically determined by p p 1 , s, a q . After making this modification, we can solve min u u J V π using Theorem 3.4 and Theorem 3.5, or existing ℓ p -norm formula [39, 40].
## Constrained Best Arm Identification

Tyron Lardy CWI and Leiden University

## Christina Katsimerou

Booking.com christina.katsimerou@booking.com

## Wouter M. Koolen

CWI and University of Twente wmkoolen@cwi.nl

## Abstract

In real-world decision-making problems, one needs to pick among multiple policies the one that performs best while respecting economic constraints. This motivates the problem of constrained best-arm identification for bandit problems where every arm is a joint distribution of reward and cost. We investigate the general case where reward and cost are dependent. The goal is to accurately identify the arm with the highest mean reward among all arms whose mean cost is below a given threshold. We prove information-theoretic lower bounds on the sample complexity for three models: Gaussian with fixed covariance, Gaussian with unknown covariance, and non-parametric distributions of rectangular support. We propose a combination of a sampling and a stopping rule that correctly identifies the constrained best arm and matches the optimal sample complexities for each of the three models. Simulations demonstrate the performance of our algorithms.

## 1 Introduction

In real-world decision-making systems, identifying the best policy is rarely a matter of optimizing a single performance metric in isolation. Effective policies often involve inherent trade-offs: they achieve desirable outcomes but incur costs. For instance, online platforms routinely deploy promotions to drive customer engagement [Zhao and Harinen, 2019, Zhang et al., 2024]. To ensure sustainability, such incentives must not only be effective but also economically viable-typically captured through feasibility constraints on budget or return on investment (ROI) [Goldenberg et al., 2020]. Similarly, bidding policies in online advertising need to drive traffic or purchases at an acceptable incremental ROI [Chen and Au, 2022]. In healthcare, a treatment needs to improve health outcomes under safety constraints. In all these examples, the goal is not simply to maximize the average benefit, but to do so subject to feasibility constraints-economic, operational, or ethical.

This motivates a constrained exploration problem: given a finite set of policies (arms), confidently identify the best policy with respect to a quality metric, among those that satisfy a feasibility constraint on a cost metric, such as a minimum return on investment (ROI), a risk threshold, or a fairness requirement. That is, each arm is associated with an unknown joint distribution on reward and cost, which may be arbitrarily dependent. During the exploration, the decision-maker chooses an arm in each round and observes a real-valued reward and cost, sampled from that arm. Once there is enough evidence that one arm has the highest quality amongst all feasible arms, or that no arm is feasible, the exploration stops and returns the best feasible arm or no arm respectively. This setting generalizes the classical best arm identification (BAI) problem in multi-armed bandits, where arms are judged only by their quality [Garivier and Kaufmann, 2016], and introduces an additional challenge not captured by standard BAI frameworks: coupled reward and cost metrics: business KPIs are rarely independent-more aggressive policies often yield higher rewards but also incur

greater costs. This interdependence violates the independence assumptions common in prior work on constrained bandits. In addition, reward and cost distributions are typically not parametric, especially in monetary applications.

In this paper, we provide a general strategy for the constrained BAI problem in the fixed-confidence setting. That is, we consider algorithms that are δ -correct , i.e., that output the correct answer (either the best feasible arm or no arm) with probability at least 1 -δ for some fixed confidence level δ . The goal is to minimize the expected number of samples needed by the algorithm, while guaranteeing δ -correctness.

Many variants of constrained BAI have previously been studied. Faizal and Nair [2022] and Yang et al. [2025] studied the same problem in the fixed budget setting, and Kone et al. [2025] in the Pareto set identification problem. An alternative way to incorporate costs in the BAI problem is the multi-fidelity formulation, in which known costs are tied to the desired accuracy level [Poiani et al., 2024], or by minimising the overall cost while still maximising a single reward dimension [Kanarios et al., 2024]. Wang et al. [2022] consider BAI with safety constraints with separate quality and feasibility dimensions. However, they assume independence and a linear or monotonic relation between the cost and reward. Furthermore, the constraints are required to be satisfied throughout the exploration, whereas we focus on pure exploration. David et al. [2018] and Hou et al. [2022] optimise a single performance metric, but with a constraint on some risk measure of the returned arm, such as its variance or a given quantile. Hu and Hu [2024] study a problem that is close to our setting. Each arm is associated with multiple performance metrics and their goal is to find the arm with the highest mean for a given metric, while a certain quantile of the others remains below a threshold. They assume only one of the metrics is sampled per rounds and that the different metrics are independent. Finally, Katz-Samuels and Scott [2019] propose a modification of the LUCB algorithm to solve the constrained BAI problem with multidimensional constraints. While they are the only one among the above to allow dependence, they focus purely on sub-Gaussian distributions, which is not always realistic for business purposes. Overall, most prior works ignore the dependence between reward and cost and/or focus on sub-Gaussian settings. To the best of our knowledge, none consider both dependence and arbitrary models, as we do here.

## 1.1 Contributions

Building on techniques from standard BAI [Garivier and Kaufmann, 2016, Degenne and Koolen, 2019], we derive instance-dependent lower bounds on the sample complexity for generic bandit models. The main difficulty therein lies in the fact that case distinctions arise from possible tradeoffs between cost and reward that do not occur when only considering the reward dimension. As is well-known in the BAI literature, the sample complexity lower bound also gives rise to the proportion of samples that an optimal algorithm should allocate to each arm. We show that these weights and the lower bound can be computed whenever we have numerical access to two transportation functions. In contrast to existing frameworks for BAI, which assume either exponential families [Garivier and Kaufmann, 2016] or nonparametric distributions [Agrawal et al., 2020], this allows us to treat all models in a unified framework.

We show that the transportation functions can be efficiently computed for three bivariate arm models: Gaussian with fixed 2 × 2 covariance matrix, Gaussian with unknown 2 × 2 covariance matrix, and non-parametric distributions of rectangular support. Our proposed algorithm then uses a plugin strategy to track these weights. As the stopping rule, we use a generalized-likelihood-ratio statistic, similar to e.g. Garivier and Kaufmann [2016], Degenne and Koolen [2019]. For proving δ -correctness in the case of Gaussian with fixed covariance and non-parametric distributions of rectangular support, we import known results on the concentration for weighted sums of such statistics [Agrawal et al., 2021, Kaufmann and Koolen, 2021]. For the case of Gaussian with unknown covariance matrix, we prove a concentration result as proof of concept.

## 2 Sample complexity lower bounds

To set the stage, let M be a set of bivariate distributions on R 2 . For any ν ∈ M , we denote its mean reward and cost by m ( ν ) = ( m 1 ( ν ) , m 2 ( ν )) . We consider a K -armed bandit ν = ( ν 1 , . . . , ν K ) ∈ M K . At every time n = 1 , 2 , . . . , one arm I n ∈ [ K ] is chosen and a pair X n = ( R n , C n ) is drawn from ν I n , where R n represents the obtained reward and C n the incurred cost. Given a

threshold γ ∈ R , the objective is to identify the best feasible arm i ∗ ( ν ) = i ∗ ( m ( ν ) ) , where i ∗ ( m ) = arg max i : m i, 2 ≤ γ m i, 1 . Here, the arg max over the empty set is defined to be None , and we assume that the bandit ν has a unique best feasible arm i ∗ ( ν ) ∈ A := [ K ] ∪ { None } . To avoid confusion, it will be crucial to distinguish between arms [ K ] and answers A , especially because for our problem these nearly coincide. Algorithm design must resolve the type conversion: which arm must be pulled to increase evidence in favor of a given answer?

Applying the generic lower bound of Garivier and Kaufmann [2016] results in the following lower bound on the sample complexity of any δ -correct algorithm for constrained BAI (CBAI).

Theorem 2.1 (Garivier and Kaufmann [2016]) . Let δ ∈ (0 , 1) . For any δ -correct strategy with stopping time τ δ and any bandit model ν ∈ M K ,

<!-- formula-not-decoded -->

where KL denotes the Kullback-Leibler divergence and

̸

<!-- formula-not-decoded -->

Their proof reveals that any strategy that matches the lower bound will also match w ∗ ( ν ) as proportion of arm draws, where w ∗ ( ν ) are the weights that achieve (1). To find a strategy with optimal sample complexity, we will compute the characteristic time T ∗ ( ν ) and corresponding oracle weights. To this end, we first present an abstraction, show how it still allows efficient computation, and then implement the abstraction for the following three models:

1. Gaussian with fixed covariance Σ ⪰ 0 : M G, Σ := { N ( µ , Σ) ∣ ∣ µ ∈ R 2 } .
2. Gaussian with unknown covariance: M G := { N ( µ , Σ) ∣ ∣ µ ∈ R 2 , Σ ⪰ 0 } .
3. Non-parametric distributions on the unit square: M B := { P ∣ ∣ P on [0 , 1] 2 } .

Other models can be worthwhile, for example modeling cost and reward as independent, each drawn from some single-parameter exponential family member. We focus on the above three models to highlight the role of dependent rewards and costs.

## 2.1 Solving (1) generically in terms of a transportation cost interface to the model

We analyze the CBAI characteristic time T ∗ ( ν ) from (1), and provide an efficient algorithm for computing w ∗ ( ν ) and hence T ∗ ( ν ) . To start, we introduce a shorthand for the KL projection of an arm ν ∈ M onto the set of distributions { ν ′ ∈ M : m ( ν ′ ) = µ } with a given mean µ ∈ R 2 :

<!-- formula-not-decoded -->

We suppress the dependence on M from the notation. The characteristic time (1) can be rewritten as

̸

<!-- formula-not-decoded -->

̸

At this point we see that we need to know about the mean vectors λ such that i ∗ ( λ ) = i ∗ . We have Proposition 2.1. For each answer i ∈ A , let ¬ i := { λ ∈ R K × 2 ∣ ∣ i ∗ ( λ ) = i } . Then

̸

̸

<!-- formula-not-decoded -->

̸

This shows in particular that the Pure Exploration Rank [Kaufmann and Koolen, 2021, Definition 20] of constrained BAI is two , as ¬ i is a union of parts in which at most two arms are constrained. This

̸

will be useful in obtaining deviation thresholds below. For now, we use the partition above to simplify the characteristic time. Namely, if i ∗ ( ν ) = None , the problem of finding the characteristic time (3) reduces to

̸

<!-- formula-not-decoded -->

and for i ∗ ( ν ) = None , it reduces to

<!-- formula-not-decoded -->

We will argue that numerical access to the following two transportation cost functions, c 1 and c 2 , suffices to implement this characteristic time T ∗ and the corresponding oracle weights w ∗ .

Interface 2.1. The following two functions need to be implemented efficiently:

1. the weighted cost for making arm ν j beat arm ν i (here i can be assumed feasible)

<!-- formula-not-decoded -->

We need separate access to both terms of the sum at the minimum, that is, to KLinf( ν i , λ ∗ i ) and KLinf( ν i , λ ∗ j ) . We will denote these by c 1 ,i ( ν i , ν j , w ) and c 1 ,j ( ν i , ν j , w ) . We will also assume computational access to the limit c 1 ( ν i , ν j , ∞ ) := lim w →∞ c 1 ( ν i , ν j , w ) .

2. the cost for changing the feasibility status of an arm ν

<!-- formula-not-decoded -->

In terms of Interface 2.1, our problem (5) simplifies to

̸

<!-- formula-not-decoded -->

̸

̸

To interpret the revealed structure, note that both cases minimize over precisely K terms; one for each alternative answer different from the correct answer i ∗ . For the i ∗ = None case, we find ourselves in a 1 d thresholding problem Garivier et al. [2017], where the cost, c 2 , is that of discriminating an arm from the threshold γ . For the i ∗ = None case, the inner min j = i ∗ sub-expression matches that of the transport cost for the best arm identification problem Garivier and Kaufmann [2016], where the cost to reverse the quality of two arms there is replaced by our c 1 (which in addition ensures feasibility of the second arm). The outer binary minimum adds one extra case to the range of possibilities to be considered, namely where the best looking arm is rendered infeasible.

̸

We can solve the lower bound problem generically in terms of Interface 2.1:

Theorem 2.2. Let ν be a K -armed bivariate bandit. Let i ∗ := i ∗ ( ν ) . For all i ∈ [ K ] , we have

̸

<!-- formula-not-decoded -->

̸

where ˜ w i ∗ ( ˜ C ) := 1 , and for each sub-optimal j = i ∗ , ˜ w j ( ˜ C ) is the unique solution to w in

<!-- formula-not-decoded -->

and ˜ C ∗ is the unique solution for ˜ C in

̸

<!-- formula-not-decoded -->

if it is attained in the interval [0 , ˜ C max ] and ˜ C ∗ = ˜ C max otherwise, where we abbreviate ˜ C max := min { c 2 ( ν i ∗ ) , min j = i ∗ c 1 ( ν i ∗ , ν j , ∞ ) } .

̸

Efficient Computation Note that this theorem is not only a characterisation, it unlocks a generic computational recipe given oracle access to c 1 and c 2 . To see why, we first observe that the left-hand side of (8) is increasing in w , starting at 0 when w = 0 , and reaching c 1 ( ν i ∗ , ν j , ∞ ) for w → ∞ . Hence w j ( ˜ C ) can be computed by binary search. Moreover, the proof reveals that the left-hand-side of (9) is increasing in ˜ C . So again, we can solve (9) for ˜ C by binary search. All in all, we can compute T ∗ and w ∗ using two nested binary searches. This is the same computational cost as the algorithm of [Garivier and Kaufmann, 2016, below Theorem 5] for the oracle weights in BAI. An efficient implementation for constrained BAI is the basis for the Track-and-Stop algorithm template. It therefore remains to implement c 1 and c 2 for each of our three arm models of interest.

## 2.2 Efficient Implementation of Interface 2.1 for our Three Models for Arms

We implement the interface functions c 1 and c 2 efficiently for our three arm models of interest. We also implement KLinf and discuss the effect of dependence.

Figure 1: (a) shows the bivariate means of an example six-armed bandit ν , with its feasible best arm i ∗ ( ν ) circled green. The vertical axis is reward, while the horizontal axis is cost, with the vertical dashed line indicating the feasibility threshold γ . The four possible types of transports underlying (5a) are illustrated in (b)-(e). We can make the best arm infeasible (b). We can render an arm feasible that was already high (c). We can make a feasible arm exceed the best arm (d). And we can make an arm both feasible and better than the best arm (e). In these diagrams the reward-cost dependence within each arm manifests by the cheapest transports (indicated by red arrows) not being axis aligned.

<!-- image -->

## 2.2.1 Gaussian fixed covariance: KLinf , c 1 and c 2

Proposition 2.2. Let ν = N ( µ , Σ) and consider M = M G, Σ . Then

<!-- formula-not-decoded -->

Moreover, let ν i = N ( µ i , Σ) and ν j = N ( µ j , Σ) with i ∗ ( { µ i , µ j } ) = i , then

<!-- formula-not-decoded -->

Here, the case distinction in c 1 arises by first solving the infimum in (6) while forgetting about one of the constraints at a time. If the resulting minimizer happens to satisfy both constraints, then it must be the solution to the original problem, since its value is at least as low as that of the original problem. If this does not happen for either constraint, both of them must be active.

̸

The Impact of Dependence on Transportation Cost We are interested in the effect of dependence between reward and cost in all three models. The explicit formulas above allow us to highlight its effect explicitly. Here, dependence manifests as a nonzero covariance Σ 12 = 0 . To see its effect, we observe that the minimum cost to move an arm from mean µ to a new location λ such that λ 2 = γ is c 2 ( N ( µ , Σ)) = min λ ∈ R 2 : λ 2 = γ 1 2 ∥ µ -λ ∥ 2 Σ -1 = ( γ -µ 2 ) 2 2Σ 22 , which is attained at λ ∗ = ( µ 1 + Σ 12 Σ 22 ( γ -µ 2 ) , γ ) . So we see that Σ 12 = 0 causes the arm to move diagonally, even

̸

though the objective was to move in dimension two. This is illustrated in Figure 1(b). On the other hand, counter intuitively, the cost of that move does not depend on Σ 12 . These diagonal motions make it subtle to determine the active constraints for the c 1 motion (where we ask for a certain arm to be made feasible and better than another arm). E.g. an arm that starts feasible may be rendered infeasible by making it better. As visualized in Figure 1, the active constraints at the optimal solution can either be feasibility only (c), flipped mean reward only (d) or both (e).

## 2.2.2 Gaussian unknown covariance: KLinf , c 1 and c 2

For unknown covariance a subtlety arises: as we can see below, KLinf( ν, λ ) is not a convex function of λ . As a result, c 1 as specified by (6) is not a convex optimization problem. Fortunately, c 1 can still be computed efficiently.

Proposition 2.3. Let ν = N ( µ , Σ) and consider M = M G . Then

<!-- formula-not-decoded -->

Moreover, let ν i = N ( µ i , Σ i ) and ν j = N ( µ j , Σ j ) , then, abbreviating ℓ ( x ) := 1 2 ln(1 + x ) ,

<!-- formula-not-decoded -->

Notice that the c 2 cost is fully determined by characteristics of the second dimension. In particular, it is independent of the dependence Σ 12 between the cost and reward dimension. This happens because the optimal move will take the covariance structure into account, which cancels its effect. Furthermore, the variable θ that appears in c 1 is introduced as a parameter such that λ a, 1 ≤ θ ≤ λ a, 2 . With this extra parameter, the searches over λ a, 1 and λ a, 2 are straightforward. For the remaining search over θ ∈ R , it is possible to identify the points at which the active case in the second term switches; this will be instance dependent. For example, if µ i, 2 ≤ γ and Σ i, 12 &lt; 0 , the second term will always be active, while for Σ i, 12 &gt; 0 , case 3 takes over whenever θ ≥ µ i, 1 +Σ i, 11 Σ -1 i, 12 ( γ -µ i, 2 ) . The optimal value on each segment can be found by setting the derivative of the objective to zero, which is a matter of finding the roots of a cubic. The global minimizer can then efficiently be found by comparing the minimizers on each segment. So computing c 1 or c 2 takes a constant amount of work.

## 2.2.3 Non-parametric supported on [0 , 1] 2 : KLinf , c 1 and c 2

Proposition 2.4. Let ν be a distribution on [0 , 1] 2 and consider M = M B . Furthermore, let R λ := { ( a 1 , a 2 ) ∈ R 2 | ∀ x ∈ [0 , 1] 2 : 1 + a 1 ( x 1 -λ 1 ) + a 2 ( x 2 -λ 2 ) ≥ 0 } . Then

<!-- formula-not-decoded -->

Finally, let ν i , ν j be distributions on [0 , 1] 2 . With R ′ w := { b ∈ R 3 | b 3 ≥ 0 ≥ b 2 , ∀ x ∈ [0 , 1] 2 : 1 -w ( b 1 + b 2 x 1 ) ≥ 0 and 1 + b 1 + b 2 x 1 + b 3 ( x 2 -γ ) ≥ 0 } , we have

<!-- formula-not-decoded -->

Note that when λ ∈ (0 , 1) 2 , the region R λ is a compact convex set in R 2 . In fact, being the intersection of four half-spaces, it is a quadrilateral with its four vertices on the axes. Moreover, the objective is concave in λ . This means that for ν a distribution of finite support (e.g. an empirical distribution) we can compute KLinf( ν, λ ) using off-the-shelf convex optimisation methods e.g. the ellipsoid method. Similarly, R ′ w for w &gt; 0 is a compact convex subset of R 3 , being an intersection of six half spaces.

The case distinction that we saw for c 1 in the Gaussian fixed covariance case did not disappear. In fact, it manifests in the region for b : b 3 is the Lagrange multiplier for enforcing feasibility of arm j , and -b 2 is the Lagrange multiplier for enforcing the correct order of mean rewards. Either (but not both) can be zero at optimality if the respective constraint is satisfied already.

## 3 Asymptotically Optimal Algorithm

We now develop an asymptotically optimal algorithm. As this is classic, we defer details to the appendix. Throughout, we denote by N i ( n ) = ∑ n s =1 1 I s = i the number of samples taken from arm i in the first n rounds.

Estimates Our approach is based on an estimate ˆ ν ( n ) of the bandit ν after n samples. For the fixed covariance case, we let ˆ ν i ( n ) := N (ˆ µ i ( n ) , Σ) , where ˆ µ i ( n ) = 1 N i ( n ) ∑ n s =1 1 I s = i X s is the empirical mean of arm i after n bivariate outcomes X s = ( R s , C s ) . For the unknown covariance case, we let ˆ ν i ( n ) := N (ˆ µ i ( n ) , ˆ Σ i ( n )) , where ˆ Σ i ( n ) is the empirical covariance of the samples from arm i . Finally, for the non-parametric case, we let ˆ ν i ( n ) be the empirical distribution of the samples { X s | s ≤ n, I s = i } from arm i .

Stopping and recommendation rule The Generalised Likelihood Ratio (GLR) statistic is defined, for ˆ ı = i ∗ (ˆ ν ( n )) , by

̸

<!-- formula-not-decoded -->

̸

We stop at the first time τ δ := inf { n ∈ N : Λ n ≥ β ( δ, n ) } the GLR crosses the exploration threshold β given below, and at that point we will recommend the empirical best feasible arm ˆ ı := i ∗ (ˆ ν ( n )) . We show in Appendix B that

Theorem 3.1. The following exploration thresholds result in a δ -correct recommendation

<!-- formula-not-decoded -->

These thresholds all account for confidence ( ln 1 δ ), a union bound across incorrect answers ( ln K ), and a courser ( ln n ) or finer ( ln ln n ) union bound across time. These bounds are conservative in practice; in the experiments we will use ln 1 δ +lnln n instead (and verify that the rate of incorrect recommendations remains below δ ).

TaS Finally, our sampling rule ensures the asymptotic optimality. We compute a plug-in estimate of the oracle weights w n := w ∗ (ˆ ν ( n )) , and pick I n = arg min i N i ( n -1) -∑ n -1 s =1 w s,i (C-Tracking). We add forced exploration to keep N i ( n ) ≥ √ n . All in all, these ingredients guarantee

<!-- formula-not-decoded -->

## 4 Simulations

In this section, we put our algorithm, TaS , to the test on the four bandits depicted in Figure 2. We treat both the unknown-covariance Gaussian model and the bounded model. For the latter, we clip the Gaussian arms from Figure 2 to [0 , 1] 2 . Since there are no off-the-shelf algorithms to compare to, we adjust a number of sampling strategies used in the BAI literature to the constrained BAI setting.

For the Gaussian model with unknown covariance, the EV-TaS sampling rule uses the empirical covariance to track the weights as if we were in the fixed covariance model. A similar rule was

Figure 2: Each diagram illustrates a 5-arm bandit ν with Gaussian arms. The dots and ellipses give the mean and one standard deviation ring around it in covariance matrix Σ = [0 . 1 0 . 05; 0 . 05 0 . 09] . The two numbers and vector are T ∗ ( ν ) , T ∗ ( ν ) ln 1 δ for δ = 0 . 01 , and w ∗ ( ν ) in percentages.

<!-- image -->

previously considered for regular BAI by Jourdan et al. [2023]. Note that there is no reason to believe that this should work well (and it does not), as the sampling proportions will be sub-optimal. However, it is reasonable to consider if one does not know how to properly handle the unknown covariance. For the bounded case, we introduce the GA-TaS [Ménard, 2019], instead of the original TaS, due to the high cost of computing the optimal sampling ratios per round. Instead we use gradient ascent to solve the optimization problem online, and thus more efficiently.

̸

TopTwo-TCI sampling rule is based on Top Two algorithms in regular BAI, where the arm to sample at each time is randomly chosen between a leader and challenger [Jourdan et al., 2022]. We thus need to define what the leader and challenger mean in our case. The current best answer ˆ ı n = i ∗ (ˆ ν ( n )) will serve as the leader. For the challenger, note that the GLR ( Λ n as in (13)) is defined as a minimum of K terms, each corresponding to an answer different from ˆ ı n . Let us denote each of these terms by Λ n,j . As challenger, we take the answer that minimizes Λ n,j +log( N j ( n )) , where we let N None ( n ) := N ˆ ı n ( n ) . If either the challenger or the leader is None , we sample the other deterministically. This setup resembles the best challenger implementation of Hu and Hu [2024], with the substantial difference that our implementation regards a constrained mean rather than a quantile, and our selection criterion for the challenger accounts for the dependence between reward and cost dimensions. Oracle samples all arms with the optimal weights for the true model. Racing repeatedly samples uniformly all arms and eliminates an answer j (and the corresponding arm if j = None ) once it can be rejected. That is, if ˆ ı = None , we eliminate j when Λ n,j ≥ β ( n, δ ) , i.e. feasibility of arm j is implausible. If ˆ ı = None , we eliminate answer j when min { Λ n,j , Λ n, None } ≥ β ( n, δ ) . The second term ensures that it is implausible that ˆ ı is infeasible. If j = None , the first term in addition ensures that it is implausible that j is better than ˆ ı . We keep sampling until one answer remains. Uniform samples all arms in a round robin fashion. TaS-1d solves the unconstrained BAI problem, ignoring the cost dimension.

̸

All algorithms use the same GLR rule and the stylized stopping threshold log(1 /δ ) + log log( t ) , originally used by Garivier and Kaufmann [2016] and heavily adopted in the literature for allowing shorter runtimes while keeping the errors lower than δ . As initialization, we start by pulling each arm 3 times, which is the minimum required for the covariance matrix estimation. We work in the moderate confidence regime of δ = 0 . 01 . All instances were repeated 1000 times, except the hard one, which we ran 500 times. The results are shown in Table 1 for the unknown covariance Gaussian case and Table 2 for the bounded case. All empirical error rates remain below δ .

Table 1: Gaussian unknown covariance: average runtimes with standard errors

| Instance      | TaS-EV     | TaS-EV   | TaS        | TaS    | Oracle     | Oracle   | Uniform    | Uniform   | TopTwo-TCI   | TopTwo-TCI   | Racing     | Racing   | TaS-1d     | TaS-1d   |
|---------------|------------|----------|------------|--------|------------|----------|------------|-----------|--------------|--------------|------------|----------|------------|----------|
| Easy          | 79 . 4 ±   | 0 . 7    | 76 . 3 ±   | 0 . 7  | 89 . 6 ±   | 0 . 4    | 136 . 4 ±  | 0 . 6     | 68 . 8 ±     | 0 . 6        | 136 . 7 ±  | 1 . 5    | 96 . 6 ±   | 0 . 5    |
| Hard          | 3291 . 1 ± | 70 . 0   | 3218 . 7 ± | 68 . 6 | 4225 . 4 ± | 59 . 4   | 5498 . 9 ± | 129 . 8   | 2859 . 9 ±   | 54 . 9       | 2864 . 6 ± | 51 . 4   | 4815 . 9 ± | 101 . 6  |
| All feasible  | 199 ±      | 2 . 5    | 190 . 4 ±  | 2 . 4  | 229 . 7 ±  | 2 . 3    | 354 . 0 ±  | 4 . 8     | 174 . 6 ±    | 2 . 4        | 271 . 0 ±  | 2 . 6    | 186 . 4 ±  | 2 . 3    |
| None feasible | 234 . 2 ±  | 2 . 7    | 222 . 6 ±  | 2 . 6  | 270 . 1 ±  | 3 . 1    | 576 . 0 ±  | 13 . 4    | 241 . 9 ±    | 5 . 4        | 174 . 6 ±  | 2 . 2    | 3293 . 4 ± | 84 . 4   |

̸

Table 2: Bounded (non-parametric on [0 , 1] 2 ): average runtimes with standard errros

| Instance      | GA-TaS          | Oracle          | Uniform           | TopTwo-TCI      | Racing           |
|---------------|-----------------|-----------------|-------------------|-----------------|------------------|
| Easy          | 98 . 8 ± 0 . 9  | 104 . 7 ± 0 . 9 | 120 . 7 ± 1 . 2   | 96 . 4 ± 0 . 8  | 114 . 5 ± 0 . 9  |
| Hard          | 539 . 7 ± 7 . 6 | 669 . 3 ± 6 . 6 | 1088 . 4 ± 13 . 0 | 457 . 3 ± 4 . 7 | 933 . 4 ± 17 . 8 |
| All feasible  | 241 . 0 ± 3 . 5 | 256 . 7 ± 2 . 8 | 409 . 8 ± 6 . 1   | 191 . 1 ± 2 . 9 | 221 . 7 ± 2 . 8  |
| None feasible | 69 . 6 ± 0 . 7  | 68 . 2 ± 0 . 7  | 116 . 9 ± 1 . 8   | 47 . 5 ± 0 . 5  | 49 . 0 ± 0 . 5   |

Figure 3: Sample complexity and optimal weights as a function of dependence ρ .

<!-- image -->

## 5 The Impact of Dependence on the Sample Complexity

In this section we study the impact of dependence on the sample complexity. We study the following two-arm problem ν ρ in the fixed covariance Gaussian model as a function of correlation ρ ∈ [ -1 , 1] : the feasibility threshold is γ = 2 3 , the arm means are µ 1 = (0 , 0) and µ 2 = ( -1 4 , 1 ) , cost and reward each have variance Σ 11 = Σ 22 = 1 , and the correlation between them is Σ 12 = ρ . These parameters were selected to illustrate the three possible regimes we discuss below. Note that arm 1 is feasible while arm 2 is not, so the correct answer is always arm 1 . Figure 3 shows the characteristic time T ∗ ( ν ρ ) as a function of ρ . It is contrasted with the number of samples needed when sampling uniformly, T w unif ( ν ρ ) , and when using the optimal sample weights ignoring the dependence, T w ind ( ν ρ ) (by assuming ρ = 0 ). That is, the sample complexity for when we sample according to one of these rules, but still stop with the 'correct' GLR rule. Any other δ -correct stopping rule would be slower.

By inspecting (7), we see that the optimal unnormalized weight on the second arm will be chosen to maximize the cost of making arm two better than arm one, as long as that is cheaper than making arm one infeasible. This corresponds to the case that ˜ C ∗ is attained in the interval [0 , ˜ C max ] in Theorem 2.2. Notice that the cost of changing the feasibility status is independent of ρ , as can be seen in the expression for c 2 in Proposition 2.2. The mean reward of arm two is moved to µ 2 , 1 -ρ ( µ 2 , 2 -γ ) , which does depend on ρ . This will result in arm two becoming better than arm one for ρ &lt; µ 1 , 1 -µ 2 , 1 γ -µ 2 , 2 . Therefore, the cost of making arm two feasible and better equals the cost of just making it feasible; see also case 1 of c 1 in Proposition 2.2. This results in the flat regime on the left of the dashed line in Figure 3. As ρ further increases, making arm two feasible and better will involve moving both arms, the cost of which does depend on ρ , as can be seen in the third case of c 1 in Proposition 2.2. At some point, the maximum of this cost over the unnormalized weight becomes larger than the cost of making arm one infeasible. The optimum unnormalized weight will then make the transportation cost equal to the cost of making arm one infeasible, corresponding to the case that ˜ C ∗ = c 2 ( ν i ∗ ) in Theorem (2.2). This causes the change in behavior at the dotted line.

Finally, it is noteworthy that, in this case, uniform sampling sometimes outperforms the strategy assuming independence. This occurs because, for large values of ρ , uniform weights more closely resemble the optimal weights than those derived under the assumption of independence.

## 6 Discussion

We introduced the constrained best arm problem (CBAI), where each arm hides a joint distribution of reward and cost. The goal is to identify from observations the arm of highest mean reward among all arms with mean cost below a given threshold, or to report None if all arms are infeasible. This

model in particular allows us to study the impact of dependence between cost and reward. We characterized optimal sample complexity, and implemented the resulting optimal algorithms for three classes of arm distributions. We analyzed our algorithms theoretically, and showed that they are asymptotically optimal. Finally, we experimentally investigated the performance of a variety of algorithmic templates, including Track-And-Stop, Top-Two and Racing, and show that they perform well. We now discuss questions and future directions.

Can one handle multiple constraints? Let us imagine a multivariate problem with all-but-one dimensions being costs needing to be below respective thresholds. What would change? We could still have a c 1 and c 2 decomposition, where c 1 is the cost to make a designated arm feasible and better, while c 2 is the cost of toggling the feasibility status of an arm. For all our three models, changing the feasibility status of an arm is a solvable problem, even with multiple constraints. We either enforce all of them, or enumerate all constraints to violate, and optimize KLinf . What gets tricky is making an arm feasible and better than another one. In the known Σ and bounded cases, this still results in a convex problem. Yet in the unknown Σ case, KLinf( ν, λ ) is not convex in λ (but it is quasi-convex) and c 1 is not even a quasi-convex problem in λ (as it is a sum of quasi-convex objectives). We currently optimize c 1 in Proposition 2.3 by locating the optimal θ by finding roots of a small number of cubics. With multiple constraints, the number of cases in the right-most function of θ in Proposition 2.3 may equal the (exponential) number of subsets of active constraints.

Is the stylized threshold valid? In Theorem 3.1, we propose GLR thresholds that are of the form ln 1 δ with an added correction factor of ln ln n for the known covariance model and ln n for the unknown covariance and bounded models. In the simulations, we used a stylized threshold of ln ln n for all models. For known covariance, ln ln n is proven valid using mixture martingale techniques Kaufmann and Koolen [2021]. For Gaussians with unknown covariance, it is also possible to relate the GLR statistic to a mixture martingale, as noted in the one-dimensional case by Wang and Ramdas [2025]. However, using this to achieve a ln ln n threshold would require a sophisticated argument about the exact nature of this relation, which we leave for future work.

## References

- S. Agrawal, S. Juneja, and P. Glynn. Optimal δ -correct best-arm selection for heavy-tailed distributions. In Proceedings of the 31st International Conference on Algorithmic Learning Theory , volume 117, pages 61-110, Feb. 2020. Corrected version available https://arxiv.org/abs/ 1908.09094 .
- S. Agrawal, W. M. Koolen, and S. Juneja. Optimal best-arm identification methods for tail-risk measures. In Advances in Neural Information Processing Systems (NeurIPS) 34 , Dec. 2021.
- A. Chen and T. C. Au. Robust causal inference for incremental return on ad spend with randomized paired geo experiments. The Annals of Applied Statistics , 16(1):1-20, 2022.
- Y. David, B. Szörényi, M. Ghavamzadeh, S. Mannor, and N. Shimkin. PAC bandits with risk constraints. In ISAIM , 2018.
- R. Degenne and W. M. Koolen. Pure exploration with multiple correct answers. In Advances in Neural Information Processing Systems (NeurIPS) 32 , pages 14591-14600, Dec. 2019.
- F. Z. Faizal and J. Nair. Constrained pure exploration multi-armed bandits with a fixed budget. arXiv preprint arXiv:2211.14768 , 2022.
- A. Garivier and E. Kaufmann. Optimal best arm identification with fixed confidence. In Conference on Learning Theory , pages 998-1027. PMLR, 2016.
- A. Garivier, P. Ménard, and L. Rossi. Thresholding bandit for dose-ranging: The impact of monotonicity. arXiv preprint arXiv:1711.04454 , 2017.
- D. Goldenberg, J. Albert, L. Bernardi, and P. Estevez. Free lunch! Retrospective uplift modeling for dynamic promotions recommendation within ROI constraints. In Proceedings of the 14th ACM Conference on Recommender Systems , pages 486-491, 2020.

- D. G. Hartig. The Riesz representation theorem revisited. The American Mathematical Monthly , 90 (4):277-280, 1983.
- Y. Hou, V. Y. Tan, and Z. Zhong. Almost optimal variance-constrained best arm identification. IEEE Transactions on Information Theory , 69(4):2603-2634, 2022.
- M. Hu and J. Hu. Multi-task best arm identification with risk constraints. Available at SSRN 5214504 , 2024.
- M. Jourdan, R. Degenne, D. Baudry, R. de Heide, and E. Kaufmann. Top two algorithms revisited. Advances in Neural Information Processing Systems , 35:26791-26803, 2022.
- M. Jourdan, R. Degenne, and E. Kaufmann. Dealing with unknown variances in best-arm identification. In International Conference on Algorithmic Learning Theory , pages 776-849. PMLR, 2023.
- K. Kanarios, Q. Zhang, and L. Ying. Cost aware best arm identification. arXiv preprint arXiv:2402.16710 , 2024.
- J. Katz-Samuels and C. Scott. Top feasible arm identification. In The 22nd international conference on artificial intelligence and statistics , pages 1593-1601. PMLR, 2019.
- E. Kaufmann and W. M. Koolen. Mixture martingales revisited with applications to sequential tests and confidence intervals. Journal of Machine Learning Research , 22(246):1-44, Nov. 2021.
- C. Kone, E. Kaufmann, and L. Richert. Constrained Pareto set identification with bandit feedback. arXiv preprint arXiv:2506.08127 , 2025.
- D. G. Luenberger. Optimization by vector space methods . John Wiley &amp; Sons, 1997.
- P. Ménard. Gradient ascent for active exploration in bandit problems. arXiv preprint arXiv:1905.08165 , 2019.
- M. Naaman. On the tight constant in the multivariate Dvoretzky-Kiefer-Wolfowitz inequality. Statistics &amp; Probability Letters , 173:109088, 2021. ISSN 0167-7152.
- R. Poiani, R. Degenne, E. Kaufmann, A. M. Metelli, and M. Restelli. Optimal multi-fidelity best-arm identification. Advances in Neural Information Processing Systems , 37:121882-121927, 2024.
- H. Wang and A. Ramdas. Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance. Sequential Analysis , 44(1):56-110, 2025.
- Z. Wang, A. J. Wagenmaker, and K. Jamieson. Best arm identification with safety constraints. In International Conference on Artificial Intelligence and Statistics , pages 9114-9146. PMLR, 2022.
- L. Yang, S. Gao, C. Li, and Y. Wang. Stochastically constrained best arm identification with Thompson sampling. Automatica , 176:112223, 2025. ISSN 0005-1098.
- J. S. Zhang, B. Howson, P. Savva, and E. Loh. DISCO: An end-to-end bandit framework for personalised discount allocation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pages 33-49. Springer, 2024.
- Z. Zhao and T. Harinen. Uplift modeling for multiple treatments with cost optimization. In 2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA) , pages 422-431, 2019.

## A Proofs for Section 2

## A.1 Proof of Proposition 2.1

Technically, below we characterize the closure of ¬ i . For the value of optimization problems constrained to ¬ i this difference is immaterial, yet it ensures the optimizers are attained.

̸

First, suppose that i = None . Any λ with i ∗ ( λ ) = i must have an arm j such that λ j, 2 ≤ γ . Conversely, any λ that has an arm j with λ j, 2 ≤ γ has i ∗ ( λ ) = i . So we find that ¬ i = ∪ j ∈ [ K ] { λ | λ j, 2 ≤ γ } for i = None .

̸

̸

̸

For i = None , any λ with i ∗ ( λ ) = i must either have an arm j that is both feasible and better than i , or arm i must not be feasible. That is, either there exists j such that λ j, 1 ≥ λ i, 1 and λ j, 2 ≤ γ or λ i, 2 ≥ γ . Conversely, any λ for which either λ j, 1 ≥ λ i, 1 and λ j, 2 ≤ γ or λ i, 2 ≥ γ must necessarily have i ∗ ( λ ) = i . Therefore, ¬ i = ∪ j = i { λ | λ j, 1 ≥ λ i, 1 and λ j, 2 ≤ γ } ∪ { λ | λ i, 2 ≥ γ } .

## A.2 Proof of Theorem 2.2

We first handle the case that i ∗ ( m ) = None . We are interested, following (5a), in

̸

̸

<!-- formula-not-decoded -->

This can be restructured to

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

̸

̸

Now each w j for j = i ∗ only appears in one term of the minimum. This means that the optimal solution for w -i ∗ is to balance all contributions. We hence need to solve the system

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

for w -i ∗ and C , and then the value is C . Note that C is a concave function of w i ∗ . Resuming from (14), and introducing ˜ C = C/w i ∗ and w j ( C, w i ∗ ) = w i ∗ w j ( C/w i ∗ , 1) = w i ∗ ˜ w j ( ˜ C ) , where ˜ w j ( ˜ C ) is the solution for ˜ w j in

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

̸

where in particular we solved for w i ∗ = 1 1+ ∑ j = i ∗ ˜ w j ( ˜ C ) and w j = ˜ w j C ) 1+ ∑ j = i ∗ ˜ w j ( ˜ C ) . The objective is

̸

decreasing for C ≥ c 2 ( ν i ∗ ) , so the maximum is between 0 and that. To find it, we need to cancel (or, for bisection, compute the sign of) the derivative, i.e.

( ˜ ˜

̸

̸

̸

̸

<!-- formula-not-decoded -->

Differentiating the definition of ˜ w j ( ˜ C ) , we find

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and that is equivalent to

̸

we are left with

̸

and it hence remains to solve for

̸

<!-- formula-not-decoded -->

̸

̸

This is the same as equating F ( ˜ C ) = 1 , where F ( ˜ C ) := ∑ j = i ∗ ˜ C ˜ w ′ j ( ˜ C ) -˜ w j ( ˜ C ) . For ˜ C 1 &gt; ˜ C 2 , we see

̸

<!-- formula-not-decoded -->

̸

where we use a tangent bound on ˜ w j together with the fact that ˜ w ′ j is increasing, since ˜ w j is the inverse of a concave function. It follows that F ( ˜ C ) is increasing, so that F ( ˜ C ) = 1 can be found through bisection.

If i ∗ ( m ) = None , then, following (5b), we are interested in computing

<!-- formula-not-decoded -->

By reasoning similar to the above, the optimal w will balance all contributions. It follows that for every arm i

<!-- formula-not-decoded -->

## A.3 Proofs for the Gaussian transportation functions

In this section, we provide the proofs of Propositions 2.2 and 2.3. For both, we will need the KL between two Gaussians. That is, for ν = N ( µ , Σ) and ν ′ = N ( λ , Σ ′ ) , we have

<!-- formula-not-decoded -->

It furthermore helps to know that, for fixed λ 1 , the minimum 2-dimensional KL is in fact the 1 -dimensional KL (and this insight can symmetrically be applied with dimension 1 and 2 exchanged):

<!-- formula-not-decoded -->

We now proceed with the proofs.

## A.3.1 Proof of proposition 2.2

For ν = N ( µ , Σ) and M = M G, Σ we have, by (16),

<!-- formula-not-decoded -->

where ν ′ = N ( λ , Σ) . Using (17), we find that this means that

<!-- formula-not-decoded -->

Furthermore, for ν i = N ( µ i , Σ) and ν j = N ( µ j , Σ) with i ∗ ( { µ i , µ j } ) = i , we have

<!-- formula-not-decoded -->

The solution for λ i , λ j falls in three cases, depending on which of the two constraints are active at the solution.

̸

1. λ i, 1 ≤ λ j, 1 active and λ j, 2 ≤ γ inactive. Using (17), we need to find

<!-- formula-not-decoded -->

where we have used

<!-- formula-not-decoded -->

We then make the two means in the first coordinate equal, and get

<!-- formula-not-decoded -->

with value

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and optimal second coordinates

<!-- formula-not-decoded -->

2. λ i, 1 ≤ λ j, 1 inactive and λ j, 2 ≤ γ active. Using (17), we find that then λ i = µ i and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the cost is

3. both λ i, 1 ≤ λ j, 1 and λ j, 2 ≤ γ active. Here we need to optimize

<!-- formula-not-decoded -->

Cancelling the θ derivative gives

<!-- formula-not-decoded -->

With that, the value and optimizers become

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

## A.3.2 Proof of proposition 2.3

Let ν = N ( µ , Σ) , ν ′ = N ( λ , Σ ′ ) and M = M G . To derive the KLinf , we will use the following well-known facts about matrix derivatives:

<!-- formula-not-decoded -->

Together with the fact that Σ and Σ ′ are symmetric (and so are their inverses), it follows that

<!-- formula-not-decoded -->

Setting to zero and multiplying by Σ ′ from both left and right, gives

<!-- formula-not-decoded -->

By the matrix determinant lemma, we have | Σ ′ | = | Σ | ( 1 + ( µ -λ ) T Σ -1 ( µ -λ ) ) . Substituting everything back in gives

<!-- formula-not-decoded -->

Using (17), it immediately follows that

<!-- formula-not-decoded -->

Next, let ν i = N ( µ i , Σ i ) and ν j = N ( µ j , Σ j ) . Then

<!-- formula-not-decoded -->

We have essentially already computed the first term inside the parantheses above; it equals 1 2 ln ( 1 + ( µ i, 1 -θ ) 2 + Σ i, 11 ) . For the second term, we see

<!-- formula-not-decoded -->

The values in the second and third case are the result of ignoring one of the two constraints. If the optimizer of this less constrained problem is a feasible solution, the optimizer of the entire problem has been found. If this does not happen for either of them, the optimum must be in the point λ j = ( θ, γ ) (i.e. both constraints must be active). We'll refer to these cases as case 0 -3 respectively.

To get a further handle on this quantity, we will proceed by case distinctions.

1. Let's first consider µ j, 2 ≤ γ . If Σ j, 12 &lt; 0 , then µ j, 2 + Σ j, 12 Σ j, 11 ( µ j, 1 -θ ) -≤ γ for all θ , so that the conditions to case 2 will always be satisfied (case 2 coincides with case 0 for θ &lt; µ j, 1 ). The optimum value of θ will then be in [ µ j, 1 , µ i, 1 ] , since the first term in (18) is decreasing and the second term is zero on ( -∞ , µ j, 1 ] , and reversed for [ µ i, 1 , ∞ ) . It follows that the optimum θ can be found by solving

<!-- formula-not-decoded -->

so it is a matter of finding the roots of a cubic (and pruning to [ µ j, 1 , µ i, 1 ] ). If Σ j, 12 &gt; 0 , then for θ ≤ µ j, 1 + Σ j, 11 Σ j, 12 ( γ -µ j, 2 ) , we will still be in case 2 , but we end up in case 3 for θ larger than that. This does not affect the analysis if µ i, 1 &lt; µ j, 1 + Σ j, 11 Σ j, 12 ( γ -µ j, 2 ) . If µ i, 1 is

larger than that, we can separately find the minimizer for θ ∈ [ µ j, 1 + Σ j, 11 Σ j, 12 ( γ -µ j, 2 ) , µ i, 1 ] . This can be done by solving

<!-- formula-not-decoded -->

which again comes down to solving a cubic (and clipping to the right interval). We can then minimize over the two minima to find the global minimizer.

2. Now we will consider the case µ j, 2 &gt; γ . Then case 0 can never hold and we will be in case 1 for all θ ≤ µ j, 1 -Σ j, 12 Σ j, 22 ( µ j, 2 -γ ) . If Σ j, 12 &gt; 0 , case 2 can also never be satisfied, so we will be in case 3 for all θ ≥ µ j, 1 -Σ j, 12 Σ j, 22 ( µ j, 2 -γ ) . If this bound is larger than µ i, 1 , then each θ ∈ [ µ i, 1 , µ j, 1 -Σ j, 12 Σ j, 22 ( µ j, 2 -γ )] is a minimizer. If the bound is smaller than µ i, 1 , we can use (20) to find the optimal value.

If Σ j, 12 &lt; 0 , then we will only be in case 3 until θ ≥ µ j, 1 -Σ j, 11 Σ j, 12 ( µ j, 2 -γ ) , at which point we enter case 2 . As a sanity check, notice that -Σ j, 11 Σ j, 12 &gt; -Σ j, 12 Σ j, 22 , since Σ j, 11 Σ j, 22 &gt; Σ 2 j, 12 by positive semi-definiteness (so case 2 happens after 3 ). So we can use (19) to find the minimum over all large θ , and again find the global minimizer by minimizing over the two cases.

## A.4 Proof of Proposition 2.4

First, let ν ∈ M B , then

<!-- formula-not-decoded -->

Introducing Lagrange multipliers d 1 , d 2 , d 3 , the constraints can be included in the objective as

<!-- formula-not-decoded -->

Here and in the following, E Q [ · ] is meant to be read as E ( X,Y ) ∼ Q [ · ] . This becomes more tractable by (as is usual for Lagrange multipliers) swapping the max and min, that is,

<!-- formula-not-decoded -->

We show that this swap does not change the value of the problem after further simplifying. The inner minimum has optimizer

<!-- formula-not-decoded -->

so the dual problem becomes

<!-- formula-not-decoded -->

At this point, one can reparameterize by d ′ 2 = d 2 /d 1 and d ′ 3 = d 3 /d 1 and set the derivative with respect to d 1 to zero. Then, reparameterizing once more to a 1 = d ′ 2 1+ d ′ 2 λ 1 + d ′ 3 λ 2 and a 2 = d ′ 3 1+ d ′ 2 λ + d ′ 3 λ 2 gives the desired form

<!-- formula-not-decoded -->

It remains to show that the min-max swap was allowed. To this end, work backwards from (24) (the max over the dual function) and first use the Lagrange Duality Theorem [Luenberger, 1997, Theorem 1, Section 8.6] to relate the max of the dual function to a minimum of the original function. In doing so, the domain of optimization changes to the dual of the constraint space, that is, the dual of the set of bounded linear functionals on the compact unit square [0 , 1] 2 . By Riesz' Representation

Theorem [see e.g. Hartig, 1983], this dual space is equal to the set of finite signed measures on [0 , 1] 2 , so that we recover (22).

Next, fix input arm distributions ν i , ν j ∈ M B and positive weight w . The quantity of interest is

<!-- formula-not-decoded -->

Introducing Lagrange multipliers d 1 , d 2 , d 3 , d 4 , we can write this as

<!-- formula-not-decoded -->

where the implicit min-max swap is allowed by the same argument as before. We then find optimizers

<!-- formula-not-decoded -->

and dual problem

<!-- formula-not-decoded -->

where the constraint is that both of the arguments in the logarithm are positive on the unit square, i.e., d 1 + d 3 x ≥ 0 and d 2 -d 3 x + d 4 y ≥ 0 for all ( x, y ) ∈ [0 , 1] 2 . The constraint is homogeneous in the (vector of) Lagrange multipliers. At the optimum, unconstrained optimality (i.e. zero derivative) in the ( d 1 , d 2 , d 3 , d 4 ) direction requires

<!-- formula-not-decoded -->

We can solve this for d 2 and end up with

<!-- formula-not-decoded -->

If we reparameterize by b 1 = -( d 1 -1) /w , b 2 = -d 3 /w and b 3 = d 4 /w , we get

<!-- formula-not-decoded -->

For completeness, let us remark that the inner minimizers (23) and (26) are defined as densities w.r.t. the original arm distributions ν, ν i , ν j . We now discuss how to recover the outer optimal Q , Q 1 , Q 2 for the primal problem (21) or (25). The above densities (when plugging in the optimal values of the Lagrange multipliers a or d ) are part of the answer. However, these densities themselves may not yet sum to one. The reason is that the primal solutions sometimes put mass outside of the support of their corresponding arm distribution. In some cases this is the only way to satisfy the constraints, in other cases it may be driven by optimality. Solving for satisfaction of the primal constraints (i.e. normalization and means) then resolves how the missing mass must be allocated to recover the primal feasible solutions. It is always possible to do so adding mass in a single point on the boundary of the unit square.

## B Proofs for Section 3

## B.1 Proof of Theorem 3.1 (Thresholds)

Let ν be a bandit with answer i ∗ = i ∗ ( ν ) and true means m i = m ( ν i ) . Following the proof of [Kaufmann and Koolen, 2021, Proposition 21], we have

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

where the first case is for i ∗ = None and the second is for i ∗ = None . Furthermore, the last step uses that ¬ i ∗ is covered by motions of at most two arms, one being i ∗ whenever i ∗ = None , by Proposition 2.1. Note that for i = None the set ¬ i ∗ has a motion of just arm i ∗ as well, but the cost of that is subsumed by that of any two-arm deviation. To find β that ensures δ -correctness, we need an anytime deviation inequality for a sum of two statistics of the form N i ( n ) KLinf(ˆ ν i ( n ) , m ( ν i )) .

For the fixed covarance Gaussian case, we can leverage [Kaufmann and Koolen, 2021, Theorem 9]. Even though that theorem is formulated for arbitrarily many 1d Gaussian arms, we note that for i.i.d. X i ∼ N ( µ , Σ) , our n KLinf , n 2 ∥ ˆ µ ( n ) -µ ∥ 2 Σ -1 , is a sum of two independent 1 d Gaussian contributions. To be more precise, let Y i = Σ -1 / 2 X i , where Σ -1 / 2 is the inverse of the positive definite and symmetric square root of Σ . Then Y i ∼ N (Σ -1 / 2 µ , I (2)) , so that Y i,j ∼ N ((Σ -1 / 2 µ ) j , 1) for j ∈ { 1 , 2 } independently from one another. Furthermore, define ˆ Y n = 1 n ∑ n i =1 Y i . We then see that

<!-- formula-not-decoded -->

The concentration result by Kaufmann and Koolen [2021, Theorem 9] is stated for sums of univariate KLs, as in the left-hand side; this equality allows us to also use it in the fixed covariance setting. We thus find that taking β ( δ, n ) = ln K δ +4ln ln K δ 4 +8ln(4+ln n/ 2) suffices once ln 1 δ ≥ 6 . We chose that latter threshold for readability, see Kaufmann and Koolen [2021] for a more involved threshold that works for any δ ∈ (0 , 1) .

For the unknown covariance, we exploit the expression for KLinf from (11). Recall that for the unknown covariance case we use as our estimate ˆ ν ( n ) = N ( ˆ µ ( n ) , ˆ Σ( n )) , that is, a Gaussian with the maximum likelihood mean and covariance. Then for i.i.d. X i ∼ ν = N ( µ , Σ) ,

<!-- formula-not-decoded -->

Under ν , the statistic ( n -1) ∥ ˆ µ ( n ) -µ ∥ 2 ˆ Σ( n ) -1 has a Hotelling t 2 distribution with n -1 degrees of freedom in 2 dimensions, and therefore ( n -2) KLinf(ˆ ν ( n ) , µ ) ∼ 1 2 χ 2 2 . To show this, we will (1) rely on known results on the relations between different distributions and (2) slightly abuse notation to denote distributions in equations instead of random variable. Up to scaling, the statistic we are concerned with is ln ( 1 + 1 n -1 T 2 (2 , n -2) ) . It can be shown that this is the same as ln ( 1 + 2 n -2 F (2 , n -2) ) (F-distribution). This, in turn, is equivalent to ln ( 1 + β ′ ( 1 , n -2 2 )) (beta-prime). This can be written as -ln ( ( 1 + β ′ ( 1 , n -2 2 )) -1 ) = -ln ( β ′ ( n -2 2 , 1 ) ( β ′ ( n -2 2 , 1 ) +1 ) -1 ) , which equals -ln ( β ( n -2 2 , 1 )) (beta), which is known to be exp ( n -2 2 ) (exponential). The desired result follows by reintroducing the scaling factor n -2 2 .

Hence, for fixed sample size n and λ ∈ [0 , 1] the MGF evaluates to E ν [ e λ ( n -2) KLinf(ˆ ν ( n ) , µ ) ] = 1 1 -λ (alternatively, the above reasoning can be sidestepped by computing this integral directly). We then find that for fixed sample sizes n i and n j , and threshold C ≥ 2 , we get, by a Chernoff bound,

<!-- formula-not-decoded -->

We have, with W -1 denoting the negative branch of the Lambert function,

<!-- formula-not-decoded -->

A weighted union bound over all possible values of n i and n j with each prior π over N gives

<!-- formula-not-decoded -->

Picking π ( n ) ∝ 1 n (ln n ) 2 motivates the choice β ( δ, n ) = ln K δ +2ln n +4lnln n +2ln(ln K δ +2ln t + 4 ln ln n ) . Notice that there both factors N k ( n ) are now replaced by N k ( n ) -2 . Technically, this could be compensated for by adjusting the threshold. However, this compensation would disappear as the sample sizes grow large, so that it does not matter for any asymptotic arguments. Furthermore,

̸

this is the same as simply using the statistic with factor N k ( n ) -2 and noting that, in the limit, it is indistuingishable from our original statistic. In our experiments, we choose to do the latter.

For the bounded case we can use our dual expression from (12) as a maximum over parameters in the compact set R λ ⊆ R 2 . Using the technique of [Agrawal et al., 2021, Lemma F.1] based on worst-case regret bounds for online learning with exp-concave losses, we find that for each arm i there is a ν i martingale ( M i,n ) n ≥ 0 (of mixture form) satisfying M i,n ≥ e N i ( n ) KLinf(ˆ ν i ( n ) , m ( ν i )) -1 -2 ln(1+ N i ( n )) . Taking the product over two arms, applying Ville's inequality, and using concavity of the ln and N i ( n ) + N j ( n ) ≤ n yields δ -correctness for the choice β ( δ, n ) = ln K δ +2+4ln(1 + n/ 2) .

## B.2 Proof of Theorem 3.2 (Asymptotic Optimality)

We show that our algorithm, which consists of (a) the sampling rule (Track-and-Stop with C-tracking) combined with (b) the GLR stopping rule and (c) the empirical-best recommendation rule ensures asymptotic optimality, in the sense that for every bandit ν with a unique best answer i ∗ = i ∗ ( ν ) , our algorithm is δ -correct on ν and ensures

<!-- formula-not-decoded -->

This argument was pioneered by Garivier and Kaufmann [2016] for BAI in exponential families, and extended to general answers by [Kaufmann and Koolen, 2021, Theorem 17]. [Degenne and Koolen, 2019, Theorem 7] proved the (upper-hemi) continuity assumption which was assumed before, and [Agrawal et al., 2020, Section 6] generalized to non-parametric arms.

To apply the argument for constrained BAI in the three models, we need to check two things. (1) the estimates ˆ ν ( t ) concentrate sufficiently fast around the true bandit ν . (2) the oracle weights ν ↦→ w ∗ ( ν ) are a continuous function of the bandit.

Sufficiently fast concentration of empirical mean and variance in sup norm is argued by [Jourdan et al., 2023, Section H.1.1] for one dimension, and it generalizes to our Gaussian cases in two dimensions. For the bounded model, sufficiently fast concentration of the empirical distribution in Lévy metric is given by the multivariate DKW inequality Naaman [2021].

Continuity of the oracle weights w ∗ ( ν ) as a function of the bandit ν follows from two nested applications of Berge's Theorem, which bottom out in continuity of KLinf . For the Gaussian cases (10) and (11) this holds by inspection. The bounded case (12) requires a small argument.

Proposition B.1. Let M = [0 , 1] 2 be the unit square, and let M 0 = (0 , 1) 2 be its interior. Let L be the set of all probability distributions on M . The function KLinf from (2) is jointly continuous on L× M 0 , where we equip L with the Lévy metric.

Proof. For joint lower semi-continuity we can directly apply [Agrawal et al., 2020, Lemma C.2] after observing that our L is compact. For joint upper semi-continuity we exploit the dual representation (12) and go through [Agrawal et al., 2020, Lemma C.3], noting that Skorokhod's theorem applies to the metric space M .

With those details supplied, the remainder of the classic proof applies. See Kaufmann and Koolen [2021, Appendix D].

Let us sketch the template here for completeness. First,

- Forced exploration ensures that the estimates converge to the true bandit, i.e. ˆ ν ( t ) → ν . Note that convergence is measured in the appropriate metric, which is Euclidean distance between parameters for the Gaussian cases and Lèvy for the bounded non-parametric case.
- Continuity of the oracle weight map ν ↦→ w ∗ ( ν ) ensures that the sampling weights converge to the oracle weights, i.e. w t = w ∗ (ˆ ν ( t )) → w ∗ ( ν ) . Here continuity is in the upperhemicontinuous sense.
- C-Tracking ensures that the sample counts converge to the sampling weights, i.e. N ( t ) t → ∑ t s =1 w s t . Since the latter average converges to the oracle weights, so do the sampling proportions themselves N ( t ) t → w ∗ ( ν ) .

- The stopping rule involves the empirical sampling proportion N ( t ) t and the estimates ˆ ν ( t ) . If these are close to w ∗ ( ν ) and ν respectively, then the stopping rule kicks in around time T ∗ ( ν ) ln 1 δ .

Without quantifying all these convergences, we are proving P ν { lim δ → 0 τ δ ln 1 δ = T ∗ ( ν ) } = 1 . The crux of the in-expectation argument is to invoke sufficiently fast concentration of the estimates, to ensure that the contribution to the expected sample complexity due to failures of any of the above convergences is of lower-order in ln 1 δ .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims made in the abstract are contextualized in the introduction. All of them correspond to either propositions or theorems.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide a discussion with two main limitations of our work, namely the stylized threshold used in experiments and the single constraint.

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

Justification: Correct proofs are given in the supplementary material. Wherever possible, we reference existing literature that we build upon.

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

Justification: We provide a detailed discussion of the set-up of our experiments in Section 4, as well as a figure to further clarify.

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

Answer: [No]

Justification: We aim to make the code publicly available at a later date.

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

Justification: Similar to the reproducibility, all details on the implementation are given in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide a standard error for the runtime estimates.

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

Justification: This is not included, because the experiments were run on a personal laptop. We therefore do not expect anyone to run into problems in this regard.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: There were no questionable research practices.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The societal relevance and impact of the research are discussed at the start of the introduction.

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

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
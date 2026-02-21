## Strategic Costs of Perceived Bias in Fair Selection

L. Elisa Celis Yale University

Lingxiao Huang Nanjing University

Milind Sohoni IIT Bombay

## Abstract

Meritocratic systems, from admissions to hiring, aim to impartially reward skill and effort. Yet persistent disparities across race, gender, and class challenge this ideal. Some attribute these gaps to structural inequality; others to individual choice. We develop a game-theoretic model in which candidates from different socioeconomic groups differ in their perceived post-selection value-shaped by social context and, increasingly, by AI-powered tools offering personalized career or salary guidance. Each candidate strategically chooses effort, balancing its cost against expected reward; effort translates into observable merit, and selection is based solely on merit. We characterize the unique Nash equilibrium in the large-agent limit and derive explicit formulas showing how valuation disparities and institutional selectivity jointly determine effort, representation, social welfare, and utility. We further propose a cost-sensitive optimization framework that quantifies how modifying selectivity or perceived value can reduce disparities without compromising institutional goals. Our analysis reveals a perception-driven bias: when perceptions of post-selection value differ across groups, these differences translate into rational differences in effort, propagating disparities backward through otherwise 'fair' selection processes. While the model is static, it captures one stage of a broader feedback cycle linking perceptions, incentives, and outcomes-bridging rationalchoice and structural explanations of inequality by showing how techno-social environments shape individual incentives in meritocratic systems.

## 1 Introduction

Meritocratic selection systems, used by institutions and firms for admissions, hiring, and content curation, aim to allocate opportunities based on observable indicators of ability and effort rather than wealth, identity, or social status. They are widely viewed as promoting fairness and efficiency [42, 74, 37]. Examples include standardized tests such as the SAT and JEE [52, 8], structured interviews and assessments [72, 13], and algorithmic ratings on online platforms [24, 73, 84].

Yet, despite their formal neutrality, these systems often produce significant disparities in representation and outcomes. Women, racial minorities, and lower-income groups are consistently underrepresented in elite universities, leadership roles, and high-paying industries [64, 45, 75]. These gaps persist even when evaluation procedures are blind to group identity, suggesting that there are additional factors that drive inequality in merit-based processes.

One set of explanations points to structural barriers: unequal access to resources that enhance merit (e.g., quality education, extracurricular activities), implicit biases in selection processes, and limited opportunities due to privileged networks [87, 40, 56, 61, 13, 63, 24, 73, 84]. Others suggest that individuals who face the same selection rules may simply make different choices, investing less effort because they perceive lower returns to success due to cultural preferences, opportunity costs, or labor market sorting [31, 11, 39, 66, 12]. These disparities under-utilize talent, reducing innovation, diversity of ideas, and social progress [43, 58, 67]. This raises a central question: how can differences in perceived opportunity translate into systematic behavioral disparities even when evaluation is symmetric?

Nisheeth K. Vishnoi Yale University

Expectations about what selection yields-admission, employment, mobility-are shaped not only by historical inequalities [76, 59, 51], but increasingly by algorithmic tools that mediate labor market signals [46]. Although meritocratic ideals suggest that pay should correlate with skills, productivity, and achievements, empirical studies reveal persistent wage disparities even after controlling for factors such as occupation, education, experience, and hours worked [38, 89, 86, 64, 83, 78, 68, 41]. For example, in the U.S., women earned just 83.1% of what men earned in 2021, despite outnumbering men in the college-educated labor force [83, 26]. Similar wage gaps persist across racial, class, caste, and ethnic lines, with Black, Hispanic, and Indigenous workers earning less than White and Asian peers in comparable roles [68]. Large language models (LLMs) may further exacerbate these disparities. Recent studies show that when asked for job or salary recommendations, LLMs return systematically different responses across demographic groups, even when qualifications are held constant [2, 46]. Such signals can distort perceived opportunity and disincentivize effort long before any selection decision occurs. Taken together, these findings suggest a perception-driven bias: social and algorithmic cues about post-selection value shape pre-selection investment, reinforcing group-level disparities even under ostensibly meritocratic systems.

Our contributions. We introduce a game-theoretic model of meritocratic selection in which candidates from two groups differ in their perceived value of being selected. This model integrates contest theory with models of structural bias from algorithmic fairness and captures how valuation disparities influence effort, merit, and selection outcomes. While the model is static, it represents one stage of a broader feedback process linking perceptions, incentives, and outcomes. Our main contributions are:

1. Modeling. We formulate a two-group contest in which n rational agents, divided into groups G 1 and G 2 (with proportion α ∈ (0 , 1) ), compete for c · n positions. Group-specific valuations follow distributions p 1 and p 2 , with p 2 modeled as a ρ -biased version of p 1 for ρ ∈ (0 , 1] [47], representing structural disparities. Each candidate chooses effort based on their valuation to maximize their expected payoff, which is then converted into observable merit used for selection.
2. Equilibrium characterization. We prove the existence and uniqueness of a symmetric Nash equilibrium in the large-agent limit, and express the equilibrium thresholds for both groups in terms of ( c, α, p 1 , p 2 ) (Theorem 3.1). We further show that the equilibrium in the finiten setting converges to this solution at rate O ( √ log n/n ) .
3. Micro to macro analysis of metrics. Using the equilibrium solution, we derive closed-form expressions for key performance metrics-group-wise representation ratio r R , social welfare ratio r S , and institutional revenue in the case where p 1 is uniform and p 2 is ρ -biased (Proposition 4.1). These expressions reveal how small changes in ρ , c or α can produce non-linear shifts in outcomes.
4. Fairness-aware interventions. We formulate a constrained optimization problem (Problem (6)) that allows institutions to trade off between increasing selectivity ( c ) and reducing valuation bias ( ρ ) under fairness constraints (e.g., 80%-rule). We solve this problem in closed form for linear cost functions and characterize when each intervention is most cost-effective (Figure 4).

Taken together, our framework provides a quantitative lens on how structural or algorithmic biases in perceived value can rationally produce effort and outcome disparities in meritocratic systems, and offers tools to design interventions that enhance both representation and efficiency.

Related work. Our work connects three areas: economic theories of meritocracy, game-theoretic models of contests, and algorithmic fairness. Social scientists have long examined the tension between meritocratic ideals and persistent disparities in outcomes, including gender and racial pay gaps [59, 76, 38, 15]. While prior models of statistical discrimination explain disparities through institutional uncertainty or group-dependent beliefs-often despite equal agent quality [3, 69, 21, 23, 5, 49]-our model assumes perfect institutional information and symmetric evaluation. Instead, we show how valuation asymmetries alone can induce disparities in effort and selection outcomes. From a game-theoretic perspective, our setting builds on all-pay auctions and Tullock contests, which model competitive effort under asymmetry [79, 29, 27]. However, these models rarely consider group-level valuation differences. To our knowledge, we are the first to study equilibrium behavior in large-agent contests with asymmetric valuation distributions across groups. Finally, our bias model extends work in algorithmic fairness that explores valuation gaps and signal noise [48, 28, 18], but previous strategic classification models typically lack inter-agent competition or valuation-based disparities [14, 29]. Thus, our work offers a novel integration of asymmetric group valuations into competitive contest frameworks, with implications for equilibrium behavior, fairness, and institutional design. See Section A for further discussion and detailed comparisons.

## 2 Model and metrics

We consider a population of n agents competing for k = cn indistinguishable spots, where c ∈ (0 , 1) denotes the selection fraction. Agents are partitioned into two groups: an advantaged group G 1 of size (1 -α ) n and a disadvantaged group G 2 of size αn , where α ∈ (0 , 1) . Each agent i ∈ G ℓ ( ℓ ∈ { 1 , 2 } ) has a valuation v i ∼ p ℓ supported on Ω ℓ ⊆ R ≥ 0 . We model systemic disadvantage via a scaling of valuations: if p 1 is the valuation distribution for G 1 , then G 2 has valuations drawn from p 2 ( v ) = 1 ρ p 1 ( v ρ ) , where ρ ∈ (0 , 1] captures the degree of bias, implying E v ∼ p 2 [ v ] = ρ E v ∼ p 1 [ v ] . For instance, if p 1 is uniform on [0 , 1] , then p 2 is uniform on [0 , ρ ] . Such a bias model has been widely studied in the fairness literature [48, 16, 19] and serves as a benchmark for understanding systemic disparities. Section D.1 discusses extensions where the valuation distributions p 1 and p 2 are truncated Gaussians, and the bias parameter ρ may also be drawn from a distribution, introducing stochastic heterogeneity across candidates. These structural disparities across groups may stem from unequal access to opportunity, differences in marginal returns, labor market discrimination, or broader societal narratives about value; see also Remark 2.1 for practical scenarios.

Each agent also has an initial ability a i ∼ p a supported on Ω a ⊆ R ≥ 0 , drawn independently. We assume that p a is identical across groups. Agents choose policies A i : Ω ℓ × Ω a → R ≥ 0 that map their type θ i = ( v i , a i ) to an exerted effort e i = A i ( θ i ) . The agent's score is s i = e i + a i . A strictly increasing merit function m : R ≥ 0 → R ≥ 0 maps scores to merit. The institution selects the k agents with the highest merit values. Each agent's payoff is

<!-- formula-not-decoded -->

where the second equality uses the strict monotonicity of m . Agents know n , k , p 1 , p 2 , p a , their group identity, and their own type θ i = ( v i , a i ) , but not others' types. Let A = ( A 1 , . . . , A n ) denote the joint policy profile. The probability that agent i is selected after exerting effort e is

̸

<!-- formula-not-decoded -->

where θ j = ( v j , a j ) are drawn i.i.d. from the respective group distributions. The expected payoff is

<!-- formula-not-decoded -->

A policy profile A is a Nash equilibrium (NE) if, for all i , v , a , and e ,

<!-- formula-not-decoded -->

We implicitly assume that agents act rationally and strategically to maximize their expected payoffs, using their knowledge of the contest structure to compute the NE policy A [77]. For simplicity, we sometimes assume that p a is a point mass at 0 , so that policies depend only on valuations.

A special case of our model generalizes the classical undifferentiated contest (where p 1 = p 2 ), which has been extensively studied [79, 55, 85]. To the best of our knowledge, our work is the first to study strategic asymmetries arising from valuation differences in settings where group sizes are known and fixed, a structure commonly seen in admissions and hiring. We provide detailed comparisons with prior works [1, 29, 27] in Section A.1. Remark E.7 discusses extensions to multi-group settings and to heterogeneous effort-to-merit mappings, where each individual may convert effort into merit at a different (non-linear) rate.

Metrics. We study three metrics to evaluate fairness, efficiency, and institutional outcomes under a given policy A . Define R ℓ ( A ) as the (random) fraction of agents selected from group G ℓ . The representation ratio is r R ( A ) := E [ min { R 1 ( A ) R 2 ( A ) , R 2 ( A ) R 1 ( A ) }] , 1 a metric commonly used in the fairness literature [22, 6, 17]. r R ( A ) ∈ [0 , 1] , and low values indicate underrepresentation of one group. Building on standard notions of allocative efficiency [70], define groupwise social welfare as S ℓ ( A ) := 1 | G ℓ | ∑ i ∈ G ℓ ( I ( i selected ) · v i -e i ) . The social welfare ratio is r S ( A ) := E [ min { S 1 ( A ) S 2 ( A ) , S 2 ( A ) S 1 ( A ) }] . This metric measures disparities in average payoffs between groups. Define the average revenue as RV ( A,m ) := E [ 1 k ∑ i selected m ( s i ) ] , capturing the average merit of selected agents and aligns with institutional objectives [33, 34].

1 We consider the min operator since randomness can occasionally lead to equal or even higher representation for the disadvantaged group G 2 . This becomes vanishingly rare as n →∞ .

We analyze how the NE policy A and associated metrics vary with the bias parameter ρ and the selection fraction c . These parameters capture systemic disparities and selection competitiveness, respectively. Even for simple instances, deriving closed-form NE strategies under asymmetric valuations is significantly more complex than in the undifferentiated case. A worked example illustrating these challenges is provided in Section B.

Remark 2.1 ( Practical settings with group-based valuation bias). Our model captures environments where disadvantaged groups anticipate lower returns from being selected-due to structural barriers, social context, or biased algorithmic feedback (see Section 1). For example, as discussed in Section 1, persistent wage gaps across gender and race-even after accounting for qualifications-as well as biased algorithmic recommendations can diminish expectations about the benefits of selection. These lower expectations can rationally reduce pre-selection effort, even under formally fair rules, and represent the main regime we study. That said, in domains such as credit, housing, or education, disadvantaged groups may instead face higher marginal returns due to limited outside options; this can be modeled by reversing which group has the compressed valuation distribution.

## 3 Theoretical results: Nash equilibrium and metrics for large n

The first question we address is whether a Nash equilibrium (NE) policy exists for the two-group contest and how it can be computed. While characterizing NE policies for finite n is challenging, the large-population limit ( n → ∞ ) reveals an interesting and tractable structure. The following result shows that in this limit, it is possible to describe how the strategies of the two groups, G 1 and G 2 , converge. However, the absence of an explicit policy formulation for finite n complicates the interpretation of convergence, which we address by adopting the notion of an approximate NE policy.

Definition 3.1 ( ε -Nash equilibrium [54]). For an ε &gt; 0 , a policy A is said to be an ε -NE policy if for any ℓ ∈ { 1 , 2 } , agent i ∈ G ℓ , type ( v, a ) ∈ Ω ℓ × Ω a , and effort e ≥ 0 , the following condition is met: π i ( v, a, e ; A -i ) ≤ π i ( v, a, A i ( v, a ); A -i ) + ε.

An ε -NE permits stability violations up to ε , with exact NE recovered when ε = 0 . This notion allows us to formalize the convergence of NE policies in the following theorem.

Theorem 3.1 ( The two-group contest: Large n limit). Let α, c ∈ (0 , 1) . For ℓ = 1 , 2 , let p ℓ be a density supported on a domain Ω ℓ ⊆ R ≥ 0 . Let p a be a density supported on a domain Ω a ⊆ R ≥ 0 . Let m : R ≥ 0 → R ≥ 0 be a merit function that is strictly increasing. For ℓ = 1 , 2 , let F ℓ be a cumulative density function (CDF) of the sum of valuation and initial ability such that for any ζ ∈ R ≥ 0 , F ℓ ( ζ ) = Pr v ∼ p ℓ ,a ∼ p a [ v + a ≤ ζ ] . Suppose (Ω 1 ∪ Ω 2 ) + Ω a is connected 2 and densities p 1 , p 2 , p a are positive at any point of their own domains. Let t be the unique solution to the equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and let policy A be: each agent i ∈ G 1 uses the restriction A i = s | Ω 1 × Ω a , while each agent j ∈ G 2 uses the restriction A j = s | Ω 2 × Ω a . Moreover, this solution t is monotonically decreasing with c .

This A is the unique policy such that there exists an infinite sequence A (1) , . . . , A ( n ) , . . . , where A ( n ) is a policy for the two-group contest with n agents characterized by a threshold function s ( n ) : (Ω 1 ∪ Ω 2 ) × Ω a → R ≥ 0 , such that the followings hold: (1) For every integer n ≥ 1 , agent i ∈ G 1 uses the restriction A ( n ) i = s ( n ) | Ω 1 × Ω a , while each agent j ∈ G 2 uses the restriction A ( n ) j = s ( n ) | Ω 2 × Ω a and lim n →∞ s ( n ) = s ; (2) Every A ( n ) is an ε n -NE policy with lim n →∞ ε n = 0 .

This theorem characterizes the policy A through a threshold function s parameterized by t , establishing that there exists a sequence of policies A (1) , . . . , A ( n ) , . . . that converge towards A , progressively approximating it. In Section E, we provide the explicit form of policies A ( n ) characterized by s ( n ) (in Theorem E.2) and a complete proof. Note that the value t defines the threshold function s , and consequently, the policy A . Therefore, we focus below on analyzing t . We first remark on the uniqueness of t guaranteed by Equation (2) under certain assumptions on the domains and densities (see Lemma E.1), which are natural in real-world contexts and satisfied by the distributions discussed in Section 2. These assumptions ensure that each F ℓ , being a CDF, is strictly monotonic over its

2 Here, symbol + represents the Minkowski sum of domains, where A + B = { a + b : a ∈ A,b ∈ B } .

domain Ω ℓ +Ω a . Consequently, the combined CDF (1 -α ) F 1 + αF 2 must also be strictly monotonic over the connected domain (Ω 1 ∪ Ω 2 ) + Ω a , which guarantees the uniqueness of the solution t .

A key takeaway from Theorem 3.1 is that, in the largen limit, each agent makes a binary decision based on their combined valuation and initial ability v + a : either exert effort max { t -a, 0 } to ensure a score of at least t , or put in no effort at all. The threshold t , determined by Equation (2), plays a central role in this decision. It is chosen such that a fraction 1 -c of the agent population has v + a ≤ t , meaning that exactly a fraction c is expected to exert effort and compete. Thus, t implicitly encodes the level of competition: higher values of t reflect more intense competition, requiring higher effective scores for selection.

Computing the threshold in the NE policy. Equation (2) is crucial for applying Theorem 3.1, as it enables the explicit computation of t , facilitating analysis in Section 4. Let F a be the CDF of the initial ability. 3 Note that F ℓ ( ζ ) = Pr v ∼ p ℓ ,a ∼ p a [ v + a ≤ ζ ] = ∫ Ω ℓ p ℓ ( v ) F a ( ζ -v ) dv . Thus, Equation (2) is equivalent to (1 -α ) ∫ Ω 1 p 1 ( v ) F a ( ζ -v ) dv + α ∫ Ω 2 αp 2 ( v ) F a ( ζ -v ) dv = 1 -c.

Specifically, when p 2 ( v ) = 1 ρ p 1 ( v ρ ) for some ρ ∈ (0 , 1] , this equation becomes becomes

<!-- formula-not-decoded -->

We illustrate how to use this equation to compute the explicit form of t . Let p 1 be uniform on Ω 1 = [0 , 1] , p 2 be uniform on Ω 2 = [0 , ρ ] , and p a be uniform on Ω a = [0 , 1] . Such uniform densities are often used in studies and analyses [48, 16, 19], serves as a fundamental benchmark for insights into decision-making, allocation mechanisms, and strategic behavior. Moreover, domain (Ω 1 ∪ Ω 2 )+Ω a = [0 , 2] is connected for any value of ρ ∈ (0 , 1] , satisfying assumptions in Theorem 3.1. Since p 1 ( v ) = 1 for v ∈ [0 , 1] , p 2 ( v ) = 1 ρ for v ∈ [0 , ρ ] and F a ( ζ -v ) = min { 1 , ( ζ -v ) + } (Here, x + = max { 0 , x } ),

Equation (2) reduces to ∫ 1 0 (1 -α ) · min { 1 , ( ζ -v ) + } dv + ∫ ρ 0 α ρ · min { 1 , ( ζ -v ) + } dv = 1 -c . Consequently, the solution t is a piecewise function of parameters ρ , c , and α ; see Proposition F.4 for its explicit form. Here, we illustrate the behavior of t over a representative range where α = 0 . 5 (equal-sized groups) and 0 &lt; c ≤ 1 4 (high selectivity).

<!-- formula-not-decoded -->

Note that, while t is the same for both groups, it may happen that t &gt; 1 + ρ (when ρ &lt; 1 -2 √ c ), implying that no agent in G 2 exerts any effort. We note that for other densities, such as piecewise linear and polynomial, including Pareto, explicit forms of the solution t are achievable. For instance, consider a Pareto distribution defined by p 1 ( v ) = 2 v 3 for v ≥ 1 , a ρ -biased density p 2 ( v ) = 1 ρ p ( v ρ ) , and p a is a point mass at 0. Here, t can be explicitly calculated: If α + c -1 &gt; 0 and ρ &lt; √ ( α + c -1) /α , then t = ρ √ α/ ( α + c -1) otherwise, t = √ (1 -α + αρ 2 ) /c .

Computing the metrics. We next ask whether the key metrics associated with the NE policy A from Section 2 can be computed in closed form. Given the simple threshold structure of A , these metrics can indeed be expressed as functions of the scalar threshold t . However, for general densities p 1 and p 2 , the expressions for the representation ratio r R ( A ) and social welfare ratio r S ( A ) become more complex due to the presence of the min operator and the convolution involved between p ℓ and p a (see Theorem F.3). For clarity, we focus on the special case where p 2 is a ρ -biased version of p 1 and p a is a point mass at 0, which admits more tractable expressions. Since t depends on the parameters ρ , c , and α , the resulting metrics are also functions of these parameters. The following theorem characterizes both the explicit forms and their monotonicity behavior.

Theorem 3.2 ( Metrics and their monotonicity). Assume p 2 ( v ) = 1 ρ p 1 ( v ρ ) for some ρ ∈ (0 , 1] and p a is a mass point at 0. Let policy A be defined as in Theorem 3.1, characterized by t being the unique solution of Equation (4) . Then for any density p 1 ,

<!-- formula-not-decoded -->

Moreover, r R ( A ) and r S ( A ) are monotonically increasing w.r.t. ρ , while RV ( A,m ) is monotonically increasing w.r.t. ρ and monotonically decreasing w.r.t. c and α , for any merit function m .

3 Throughout this paper, we extend the domain of a CDF F to the entire real line R such that F is monotonically non-decreasing, with F ( -∞ ) = 0 and F ( ∞ ) = 1 .

Figure 1: Evolution of group effort policies at iteration 500 for various n with ρ = 0 . 8 and c = 0 . 2 .

<!-- image -->

The proof is deferred to Section F. Combined with the closed-form expression for t , this result enables direct computation of the metrics, which we use for contest analysis in Section 4. Notably, r R ( A ) and r S ( A ) increase with c , while RV ( A,m ) decreases, highlighting both benefits and trade-offs for the institute. The qualitative behavior of r R ( A ) and r S ( A ) w.r.t. c and α , however, depends on the underlying densities; see Remark F.2.

Discussion of NE policies for finite n . To assess the robustness and practical relevance of our theoretical results, we study the closeness between the finiten NEpolicy and the infinite-population threshold policy s defined in Equation (3) (see Section C). We propose a dynamic procedure (Algorithm 1) that initializes with s 1 = s 2 = s for groups G 1 and G 2 , and iteratively updates them.

We simulate this dynamics under the setting p 1 = Unif[0 , 1] , p 2 = Unif[0 , ρ = 0 . 8] , p a = δ 0 , with c = 0 . 2 , α = 0 . 5 , and n = 20 , 200 , 600 , 1200 , running 500 iterations in each case. Figure 1 shows representative results; full plots are in Figures 5-8.

The simulations show that even moderate population sizes ( n ≥ 600 ) yield policies closely tracking the infinite NE, validating its use as a practical approximation. We also observe group-level differences in convergence speed and stability (Figure 9), with smoother and faster stabilization as n increases.

Finally, using the proof of Theorem 3.1, we establish that the finiten NE policy is O (log n/n ) -close in value and yields an O ( √ log n/n ) -NE. Concretely, this means that for large n , the finite policy takes the form s ( n ) = 0 for v &lt; t -O ( √ log n/n ) and s ( n ) = t otherwise. For general distributions, closeness depends on the density structure and is more involved. Aligning with our empirical observations, these results reinforce the practical relevance of the largen analysis, which captures the incentive-aligned baseline under strategic behavior. See Section C.2 for details.

Key ideas in the proof of Theorem 3.1. The proof involves two main steps: hypothesizing the NE policy structure and verifying that it is indeed an equilibrium. We sketch the core ideas below; a full overview appears in Section E.1. For clarity, we focus on the case where p a is a point mass at 0.

Hypothesizing the structure of the NE policy. The key challenge in characterizing the NE policy lies in the absence of its explicit form for finite n . Drawing intuition from the undifferentiated case with density p , where the NE policy converges to a threshold function s ( v ) = F -1 p (1 -c ) if v ≥ F -1 p (1 -c ) and 0 otherwise, the first idea is to hypothesize that in the two-group case, the NE policies s 1 , s 2 also take threshold forms with group-dependent thresholds t 1 , t 2 . However, asymmetry in p 1 , p 2 complicates the expression of winning probabilities P i and prevents a straightforward computation of t 1 , t 2 . Focusing on the uniform case where p 1 is Unif[0 , 1] and p 2 is the ρ -biased version supported on [0 , ρ ] , the second idea is that in the limit n →∞ , NE stability requires t 1 = t 2 = t . If t 1 &gt; t 2 , then some agents in G 1 benefit by reducing their effort to slightly above t 2 , contradicting NE; a symmetric argument holds if t 1 &lt; t 2 . Although s 1 and s 2 are defined on different domains, they can be viewed as restrictions of the same threshold function s characterized by t . This justifies setting s 1 = s 2 = s with a shared threshold t , and modeling the two-group contest using an effective mixture density p = (1 -α ) p 1 + αp 2 supported on Ω 1 ∪ Ω 2 . As n → ∞ , the contest becomes indistinguishable from the undifferentiated case with p , yielding the same threshold t = F -1 p (1 -c ) as in Equation (2). Thus, the NE policy A defined in Theorem 3.1 is a natural candidate for the limiting equilibrium. Note that this p is only used for hypothesizing NE rather than demonstrating the convergence; see discussion in Section E.4. Moreover, the above argument suggests that, in the limit, the strategic environment becomes uniform across all agents, motivating us to develop an infinite contest (Definition E.5) and provide an alternative proof; see Section E.5.

Figure 2: Plots of the representation ratio r R ( A ) and the social welfare ratio r S ( A ) as parameters ρ and c vary for Proposition 4.1, with default settings of ( ρ, c, α ) = (0 . 8 , 0 . 1 , 0 . 5) . A dotted line in these plots indicates the threshold at which r R ( A ) = 0 . 8 or r S ( A ) = 0 . 8 .

<!-- image -->

Showing A is an NE. Although we have a solid guess for the NE policy A , a key challenge arises: the NE policy for finite n lacks an explicit form, making it unclear how to define convergence to A . Towards this end, the third key idea is to provide a different proof of convergence to a threshold function for an undifferentiated case with density p , without using the explicit NE formulations. The first attempt to prove that A is an ε -NE for sufficiently large n fails-even in the simple case with p = Unif[0 , 1] and c = 0 . 5 . One can construct a valuation v = 0 . 2 such that the agent benefits by exerting a small effort e = 0 . 01 , achieving a winning probability P i ( e ; A -i ) ≈ 0 . 5 and obtaining a payoff of approximately 0 . 09 . To bypass this, the idea is to define a proxy sequence A ( n ) with threshold policies s ( n ) that 1) converge to A as n → ∞ and 2) ensure that P i ( e ; A ( n ) -i ) → 0 for e &lt; t , making A ( n ) an ε n -NE with ε n → 0 (see Section E.1.3). To this end, we define s ( n ) ( v ) = t if v ≥ t -√ log n/n and 0 otherwise, so that a ( c + √ log n/n ) -fraction of agents put in effort t . This yields P i ( e ; A ( n ) -i ) ≤ √ 1 /n by concentration, ensuring the payoff from deviation is at most √ 1 /n , and A ( n ) is an O ( √ log n/n ) -NE. Finally, we adapt this new proof technique to the two-group case with general p 1 , p 2 by carefully selecting the following threshold shift (Definitions E.1 and E.2):

<!-- formula-not-decoded -->

and set s ( n ) ( v ) = t if v ≥ t -∆ n , 0 otherwise (see Theorem E.2). We find that lim n →∞ ∆ n = t , indicating that s ( n ) converges to s . Crucially, such a ∆ n ensures at least a ( c + √ log n/n ) -fraction of agents, in expectation, putting in effort t . Using this property, we establish a concentration tail bound for the winning probability P i analogous to the undifferentiated contest: P i ( e ; A ( n ) -i ) → 0 if e &lt; t (see Lemma E.4). Furthermore, this minimal winning probability guarantees that A ( n ) is an ε n -NE with lim n →∞ ε n = 0 (Lemma E.6). This analysis concludes Theorem 3.1.

̸

Uniqueness of the NE guaranteed by Theorem 3.1. First, if Ω 1 ∪ Ω 2 is not connected, the CDF (1 -α ) F 1 ( v ) + αF 2 ( v ) may not be strictly monotonic over Ω 1 ∪ Ω 2 . For instance, if α = c = 0 . 5 , p 1 is uniform on Ω 1 = [0 , 1] and p 2 is uniform on Ω 2 = [2 , 3] , then (1 -α ) F 1 (1) + αF 2 (1) = (1 -α ) F 1 (2) + αF 2 (2) = 0 . 5 = 1 -c . Consequently, there may exist two distinct points t 1 = 1 and t 2 = 2 in Ω 1 ∪ Ω 2 , leading to non-unique NEs. In contrast, when Ω 1 ∪ Ω 2 is connected and each p ℓ is positive on Ω ℓ , the unique solution t to Equation (3) ensures a unique NE policy. To see this, by Corollary 3.2 of [20], symmetric agents use symmetric policies in NE, so we hypothesize a policy pair ( s 1 , s 2 ) for G 1 and G 2 , respectively. As n →∞ , both s 1 and s 2 manifest as threshold functions, leading to s 1 = s 2 = s , ensuring the uniqueness of NE. Additionally, it arises because if s 1 = s 2 , the NE would destabilize, as agents from one group would adjust their thresholds to gain higher payoffs. This proof can be easily extended to multiple groups and non-identical cost of effort; see Remark E.7.

## 4 Analysis: metric behavior and intervention design

Variation of metrics with ρ and c . Using Theorems 3.1 and 3.2, we analyze how the metrics-representation ratio r R ( A ) , social welfare ratio r S ( A ) , and average revenue RV ( A,m ) -respond to changes in the bias parameter ρ and the selectivity parameter c . We study whether these effects are linear or non-linear, and whether they exhibit sharp thresholds.

Setup. We adopt the setting from Section 3: p 1 is uniform on [0 , 1] , p 2 is its ρ -biased variant, uniform on [0 , ρ ] , and p a is a point mass at 0. This isolates the effect of asymmetric valuations while simplifying calculations. The corresponding CDFs are F 1 ( v ) = v and F 2 ( v ) = v/ρ , with both saturating to 1 outside their support. Unless varied explicitly, we use default values ρ = 0 . 8 , c = 0 . 1 ,

and α = 0 . 5 , representing moderate bias, high selectivity, and balanced group sizes. We refer to Section D.2 for analogous analysis under truncated Gaussian distributions.

Closed-form metrics. Fixing the density setup above, we apply Theorems 3.1 and 3.2 to derive closed-form expressions for t , r R ( A ) , and r S ( A ) . These are summarized below (proof in Section F).

Proposition 4.1 ( Metrics for uniform densities). Let p 1 be uniform on [0 , 1] , p 2 uniform on [0 , ρ ] , and p a a point mass at 0. Let A be the NE policy as n →∞ . Let ρ c := 1 -c 1 -α . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, RV ( A,m ) = m ( t ) for any merit function m ( · ) ; r R ( A ) and r S ( A ) are monotonically increasing functions of parameters ρ, c, α .

As in Equation (5), Proposition 4.1 reveals a sharp threshold at ρ = 1 -c 1 -α . When ρ &lt; 1 -c 1 -α , t lies above the maximum valuation in G 2 , implying that no agents from that group participate. Consequently, r R ( A ) and r S ( A ) are zero and independent of ρ . When ρ crosses this threshold, these metrics become positive and increase monotonically with ρ , reaching 1 at ρ = 1 , the symmetric case.

Metric behavior. Figure 2 plots the representation ratio r R ( A ) and the social welfare ratio r S ( A ) as functions of the bias parameter ρ and selectivity parameter c . The corresponding threshold values t are shown in Appendix Figure 13(b). Both r R ( A ) and r S ( A ) exhibit non-linear growth with increasing ρ and c , and drop sharply-super-linearly-when these parameters decrease. For instance, with c = 0 . 1 , α = 0 . 5 , and ρ ≤ 0 . 85 , we observe that r R ( A ) ≤ 0 . 2 , indicating notably low representation for group G 2 . This highlights a key practical insight: in highly selective environments, such as contests with a 1-in-10 selection rate, strategic behavior amplifies disparities. These trends echo empirical findings on under-representation in competitive domains [80, 64]. Moreover, reductions in c (i.e., increased selectivity) lead to pronounced declines in both representation and social welfare.

These trends of metrics offer designers of meritocratic selection processes critical insights into strategies for countering under-representation and elevating r R ( A ) (or mitigating disparities in average payoffs and elevating r S ( A ) ). We recall the two main criteria for identifying representation bias: 1) Ensuring the selection of at least one agent from every group, and 2) adhering to the 80% rule, which serves as a guideline for identifying potential adverse impact if the hiring rate for G 2 falls below 80% of that for G 1 , i.e., r R ( A ) ≥ 0 . 8 . Given the fixed nature of α within the population structure, the main avenues for interventions aimed at improving r R ( A ) focus on adjusting the parameters ρ or c . Below, we explore potential interventions for both approaches:

(1) Increasing ρ effectively means increasing the valuation of agents in G 2 . Various strategies have been proposed and implemented to achieve this goal. For instance, [39] highlights several approaches to narrow the pay gap, including enhancing workplace flexibility, decreasing the cost associated with temporal flexibility, and improving the availability of high-quality, affordable childcare. These interventions aim to increase job valuation for women, analogous to increasing ρ . Figure 2(a) quantifies the required increase in ρ : to ensure at least one agent from G 2 is selected, ρ must exceed 0.8; to adhere to the 80%-rule, it should be at least 0.976.

(2) As shown in Figure 2(b), raising c above 0.1 satisfies the criterion for selecting at least one agent from G 2 , while elevating it to 0.5 meets the 80%-rule requirement. Increasing c represents a more straightforward approach than boosting ρ and might be more feasible for institutions. This could involve pre-selecting a larger subset of candidates and applying a distinct selection process to this subset, based on institutional priorities and the likelihood of successful candidates following the expected trajectories.

Finally, regarding the average revenue RV ( A,m ) = m ( t ) , it immediately follows from Proposition 4.1 that RV ( A,m ) significantly decreases as competition within the entire population intensifieseither through a decrease in ρ or an increase in c . Given that average revenue is indicative of the benefit of the institute, this trend underscores the critical need for contest designers to mitigate systemic biases in valuations. The decline in average revenue with increasing bias compromises not

only the fairness of the contest, but also the overall quality of the outcomes it produces. This aligns with research that has explored the losses attributed to systemic biases [43, 58, 67].

Optimizing interventions under cost and fairness constraints. Having identified two policy levers-reducing bias ( ρ ) and increasing selectivity ( c )-a natural question arises: how should institutions choose between these interventions to improve outcomes such as the representation ratio or social welfare ratio? To address this, we formulate a constrained optimization problem for cost-effective intervention design. We allow two interventions: increasing ρ by ∆ ρ ∈ [0 , 1 -ρ ] , and increasing c by ∆ c ∈ [0 , 1 -c ] . Let r R (∆ ρ , ∆ c ) denote the representation ratio under the NE policy with updated parameters ρ +∆ ρ , c +∆ c . The goal is to ensure r R (∆ ρ , ∆ c ) ≥ τ while minimizing intervention cost. We define two components of the cost function:

(1) Resource cost of increasing ρ . Let f : [0 , 1 -ρ ] → R ≥ 0 be monotonic, modeling the institutional cost of boosting valuation. A simple form is linear: f (∆ ρ ) = a ∆ ρ , justified by first-order Taylor approximation when ∆ ρ is small. Other variants include f (∆ ρ ) = a ∆ β ρ for β &gt; 1 , representing the increase in the marginal cost of continuously improving bias.

(2) Cost via revenue loss. Increasing c reduces average revenue RV ( A,m ) = m ( t ) , as it lowers the score threshold t . Let g (∆ ρ , ∆ c ) = m ( t ( ρ, c )) -m ( t ( ρ +∆ ρ , c +∆ c )) , represent the revenue decline. Since the institution seeks to maximize value, this loss contributes to total intervention cost.

Optimization problem. We formalize the intervention design as:

<!-- formula-not-decoded -->

This framework also applies to reducing welfare disparities by replacing r R with r S in the constraint.

Empirical calibration. To demonstrate real-world applicability, we calibrate the model using genderdisaggregated data from JEE Advanced 2024 , a highly competitive entrance exam for India's IITs. Of 180,200 candidates, 139,180 were male and 41,020 female; 40,284 males and 7,964 females qualified. This yields admit rates of 28.9% for males and 19.4% for females, giving an observed representation ratio of r obs ≈ 0 . 671 . The overall selection rate is c ≈ 0 . 268 , and the female applicant fraction is α ≈ 0 . 228 . These values anchor our analysis of strategic disparities and potential interventions. In Section G.1, under the uniform density setup in Proposition 4.1, we compute ρ ≈ 0 . 882 using the explicit form of r = r ( A ) = 1 -(1 -c )(1 -ρ ) .

<!-- formula-not-decoded -->

Explicit solution under uniform densities. We now solve the optimization problem under this uniform density setup with ( ρ, c, α ) = (0 . 882 , 0 . 268 , 0 . 228) , setting m ( e ) = e , f (∆ ρ ) = 5∆ 1 . 1 ρ , and g (∆ ρ , ∆ c ) = t ( ρ, c ) -t ( ρ + ∆ ρ , c + ∆ c ) . This yields the objective: 5∆ 1 . 1 ρ -[ t ( ρ +∆ ρ , c +∆ c ) -t ( ρ, c )] , subject to the condition r R (∆ ρ , ∆ c ) ≥ τ and feasibility constraints.

Insights. Figure 4 shows the optimal intervention as a function of the target threshold τ . For τ ≤ 0 . 92 , increasing c (lowering selectivity) is more cost-effective. For τ &gt; 0 . 92 , increasing ρ (mitigating bias) becomes preferable. This suggests that expanding access is more impactful under high disparity, while improving group valuation is better when gaps are narrower.

We conducted additional simulations by varying α and c beyond the default values (see Section G.2). The results remain consistent with our main findings, confirming the robustness of the above key insights. In Section G.3, we also offer a concrete example to illustrate how our model supports interpretable predictions and can inform data-grounded interventions-while also noting what is required to operationalize it in practice.

<!-- formula-not-decoded -->

Figure 3: Explicit form of Problem (6).

Figure 4: Plot of optimal interventions (∆ ρ , ∆ c ) for various τ ∈ (0 . 671 , 1] .

<!-- image -->

Alternative potential interventions. Having discussed interventions based on adjusting ρ and c , we next consider alternative approaches that modify the contest structure itself. These interventions can further reduce disparities in representation or social welfare ratios, though they depart from the baseline two-group contest formulation. A detailed analysis of these extensions is provided in Section G.4.

Introducing preference heterogeneity. One approach is to apply group-specific merit mappings of the form m ℓ ( s ) = x ℓ s + y ℓ for group G ℓ ( ℓ = 1 , 2 ), with parameters x ℓ , y ℓ ≥ 0 . Here, x ℓ acts as a scaling factor or 'handicap,' and y ℓ as an offset or 'head start' (see also Appendix A). With this intervention, we can still compute the Nash equilibrium for infinite n (Theorem G.1), which implies:

- r R ( A ) and r S ( A ) increase with ρ , c , and α , as well as with x 2 , y 2 , and decrease with x 1 , y 1 ;
- Choosing merit parameters with x 2 &gt; x 1 and y 2 &gt; y 1 can sustain high representation and welfare ratios (e.g., r R ( A ) , r S ( A ) ≥ 0 . 8 ) even in highly selective settings;
- Increasing x 2 or y 2 can thus serve as an effective disparity-reducing intervention.

Incorporating outside options. Another possibility is to assign each agent in group G ℓ a reservation payoff λ ℓ ≥ 0 if not selected. Because this payoff is earned only upon losing, a higher λ ℓ lowers the marginal benefit of effort, acting opposite to the merit parameters x ℓ and y ℓ . Hence, increasing λ 1 (the outside option for the advantaged group) reduces their effort incentives and can help narrow representation and welfare gaps.

Setting group-specific selection rates. Finally, the institution can set separate capacity constraints for each group-for instance, selecting a c -fraction of agents from G 1 and G 2 independently. This decomposes the overall model into two within-group contests, fixing r R ( A ) = 1 under equal selection rates. Compared to the combined contest, agents in G 2 now face a lower bar for selection and exert more effort on average.

## 5 Conclusion, limitations, and future work

This work highlights a central tension in modern meritocratic systems: even when selection mechanisms are formally unbiased, systemic disparities in how groups perceive value can lead rational agents to behave in ways that perpetuate inequality. Our model captures this dynamic through a strategic contest framework that extends all-pay auctions to multi-group settings. By analyzing Nash equilibria in the large population limit, we characterize how group-level biases ( ρ ) and selectivity ( c ) affect fairness and institutional metrics such as representation, social welfare, and revenue. A central contribution is Theorem 3.1, which provides an explicit form for equilibrium strategies under broad conditions. Our framework enables interpretable predictions and supports data-grounded policy interventions.

Our model makes simplifying assumptions to enable analytical tractability. Most notably, it assumes agents are fully rational and that merit is captured by a single-dimensional notion of effort. In practice, decision-making is shaped by uncertainty, cultural context, and multifaceted criteria for merit. Extending the framework to incorporate bounded rationality, noisy information, or multidimensional effort remains an important direction for future work. Several application-driven extensions are also promising. One involves modeling university admissions systems with external incentives (e.g., brand-based free-riding), which may result in over-representation of certain groups ( ρ &gt; 1 ). Another is to study how affirmative action or group-dependent costs reshape equilibrium behavior. These variants would help bridge theory with institutional design. Beyond these extensions, an important avenue is to embed this static framework within dynamic feedback environments where perceptions evolve over time in response to outcomes and institutional signals. Such models could capture how bias propagates or attenuates across repeated selection cycles. Finally, while our model isolates a tractable facet of systemic inequality, real-world disparities-especially in AI-mediated evaluations-demand broader integration with social and historical context. As algorithmic tools shape hiring, admissions, and promotion, our framework helps explain how group-level differences in perceived value can interact with selection to amplify or mitigate bias. More broadly, we view this work as a step toward unifying rational-choice and structural perspectives on inequality through formal, data-driven modeling. We hope this work informs the design of more equitable, data-driven decision systems.

## Acknowledgments

LH acknowledges support from the State Key Laboratory of Novel Software Technology, the New Cornerstone Science Foundation, and NSFC Grant No. 625707396. LEC was supported in part by NSF Award IIS-2045951. NKV was supported in part by NSF Grant CCF-2112665 and by grants from Tata Sons Private Limited, Tata Consultancy Services Limited, and Titan.

## References

- [1] Erwin Amann and Wolfgang Leininger. Asymmetric all-pay auctions with incomplete information: The two-player case. Games and Economic Behavior , 14:1-18, 1996.
- [2] Jiafu An, Difang Huang, Chen Lin, and Mingzhu Tai. Measuring gender and racial biases in large language models: Intersectional evidence from automated resume evaluation. PNAS Nexus , 4(3):pgaf089, 03 2025.
- [3] Kenneth Arrow. The theory of discrimination. Technical report, Princeton University, Department of Economics, Industrial Relations Section., 1971.
- [4] Kenneth J. Arrow. What has economics to say about racial discrimination? Journal of Economic Perspectives , 12(2):91-100, June 1998.
- [5] Jackie Baek and Ali Makhdoumi. The feedback loop of statistical discrimination. SSRN Electronic Journal , 2023.
- [6] Solon Barocas, Moritz Hardt, and Arvind Narayanan. Fairness and Machine Learning . fairmlbook.org, 2019. http://www.fairmlbook.org .
- [7] Yasar Barut and Dan Kovenock. The symmetric multiple prize all-pay auction with complete information. European Journal of Political Economy , 14(4):627-644, 1998.
- [8] Surender Baswana, Partha P Chakrabarti, V Kamakoti, Yash Kanoria, Ashok Kumar, Utkarsh Patange, and Sharat Chandran. Joint seat allocation: An algorithmic perspective, 2015.
- [9] Yahav Bechavod, Katrina Ligett, Aaron Roth, Bo Waggoner, and Zhiwei Steven Wu. Equal Opportunity in Online Classification with Partial Feedback . Curran Associates Inc., Red Hook, NY, USA, 2019.
- [10] Gary S. Becker. The Economics of Discrimination . Economic Research Studies. University of Chicago Press, 2010.
- [11] Francine D. Blau, Marianne A. Ferber, and Anne E. Winkler. The Economics of Women, Men, and Work . Pearson, Boston, MA, 7 edition, 2014.
- [12] Francine D. Blau and Lawrence M. Kahn. The gender wage gap: Extent, trends, and explanations. Journal of Economic Literature , 55(3):789-865, 2016.
- [13] Iris Bohnet. What Works: Gender Equality by Design . Harvard University Press, 2016.
- [14] Yang Cai, Constantinos Daskalakis, and Christos H. Papadimitriou. Optimum statistical estimation with strategic data sources. In COLT , volume 40 of JMLR Workshop and Conference Proceedings , pages 280-296. JMLR.org, 2015.
- [15] Emilio J. Castilla and Stephen Benard. The paradox of meritocracy in organizations. Administrative Science Quarterly , December 2010.
- [16] L. Elisa Celis, Chris Hays, Anay Mehrotra, and Nisheeth K. Vishnoi. The Effect of the Rooney Rule on Implicit Bias in the Long Term. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency , FAccT '21, page 678-689, New York, NY, USA, 2021. Association for Computing Machinery.
- [17] L. Elisa Celis, Lingxiao Huang, and Nisheeth K. Vishnoi. Multiwinner voting with fairness constraints. In IJCAI , pages 144-151. ijcai.org, 2018.

- [18] L. Elisa Celis, Amit Kumar, Anay Mehrotra, and Nisheeth K. Vishnoi. Bias in evaluation processes: An optimization-based model. CoRR , abs/2310.17489, 2023.
- [19] L. Elisa Celis, Anay Mehrotra, and Nisheeth K. Vishnoi. Interventions for ranking in the presence of implicit bias. In FAT* , pages 369-380. ACM, 2020.
- [20] Shuchi Chawla and Jason D. Hartline. Auctions with unique equilibria. In EC , pages 181-196. ACM, 2013.
- [21] Stephen Coate and Glenn C Loury. Will affirmative-action policies eliminate negative stereotypes? The American Economic Review , pages 1220-1240, 1993.
- [22] Joanne McGrath Cohoon, James P. Cohoon, Seth Reichelson, and Selwyn Lawrence. Effective recruiting for diversity. In IEEE, FIE 2013 , pages 1123-1124, 2013.
- [23] Ashley C Craig, Roland G Fryer, et al. Complementary bias: A model of two-sided statistical discrimination. Technical report, National Bureau of Economic Research, 2017.
- [24] Jeffrey Dastin. Amazon scraps secret AI recruiting tool that showed bias against women, October 2019. https://reut.rs/2N1dzRJ .
- [25] Shanglyu Deng, Qiang Fu, and Zenan Wu. Optimally biased tullock contests. Journal of Mathematical Economics , 92:10-21, 2021.
- [26] Drew DeSilver. A majority of u.s. colleges admit most students who apply. Pew Research Center, April 2019. https://www.pewresearch.org/fact-tank/2019/04/09/a-major ity-of-u-s-colleges-admit-most-students-who-apply/ .
- [27] Edith Elkind, Abheek Ghosh, and Paul W. Goldberg. Contests to incentivize a target group. In IJCAI , pages 279-285. ijcai.org, 2022.
- [28] Vitalii Emelianov, Nicolas Gast, Krishna P. Gummadi, and Patrick Loiseau. On fair selection in the presence of implicit variance. In EC , pages 649-675. ACM, 2020.
- [29] Vitalii Emelianov, Nicolas Gast, and Patrick Loiseau. Fairness in selection problems with strategic candidates. In EC , pages 375-403. ACM, 2022.
- [30] Gil S Epstein, Yosef Mealem, and Shmuel Nitzan. Lotteries vs. all-pay auctions in fair and biased contests. Economics &amp; Politics , 25(1):48-60, 2013.
- [31] Warren Farrell. Why Men Earn More: The Startling Truth Behind the Pay Gap - and What Women Can Do About It . AMACOM, New York, NY, 2005.
- [32] Jörg Franke, Christian Kanzow, Wolfgang Leininger, and Alexandra Schwartz. Effort maximization in asymmetric contest games with heterogeneous contestants. Economic Theory , 52:589-630, 2013.
- [33] Jörg Franke, Wolfgang Leininger, and Cédric Wasser. Revenue maximizing head starts in contests. Ruhr Economic Paper , (524), 2014.
- [34] Jörg Franke, Wolfgang Leininger, and Cédric Wasser. Optimal favoritism in all-pay auctions and lottery contests. European Economic Review , 104:22-37, 2018.
- [35] Jörg Franke, Christian Kanzow, Wolfgang Leininger, and Alexandra Schwartz. Lottery versus all-pay auction contests: A revenue dominance theorem. Games and Economic Behavior , 83:116-126, 2014.
- [36] Qiang Fu and Zenan Wu. On the optimal design of all-pay auctions. Technical report, Working Paper, 2021.
- [37] Malcolm Gladwell. Outliers: The Story of Success . Little, Brown and Company, New York, 2008.
- [38] Claudia Goldin. A grand gender convergence: Its last chapter. American Economic Review , 104(4):1091-1119, April 2014.

- [39] Claudia Goldin. Occupational choices and the gender wage gap. American Economic Review , 104(5):348-353, 2014.
- [40] Anthony G Greenwald and Linda Hamilton Krieger. Implicit bias: Scientific foundations. California Law Review , 94(4):945-967, 2006.
- [41] Ariane Hegewisch and Asha DuMonthier. The gender wage gap: 2015: Annual earnings differences by gender, race, and ethnicity. Report, Institute for Women's Policy Research, 2016. Available at Institute for Women's Policy Research website.
- [42] Richard J. Herrnstein and Charles Murray. The Bell Curve: Intelligence and Class Structure in American Life . Free Press, New York, 1994.
- [43] Sylvia Ann Hewlett, Laura Marshall, and Lauren Sherbin. Diversity and innovation: The impact of diversity on innovation processes and outputs. Technical report, Center for Talent Innovation, New York, NY, 2013.
- [44] Lily Hu, Nicole Immorlica, and Jennifer Wortman Vaughan. The disparate effects of strategic manipulation. In FAT , pages 259-268. ACM, 2019.
- [45] Information Technology &amp; Innovation Foundation. Diversity in tech: The unspoken bias in tech workplaces. Technical report, Information Technology &amp; Innovation Foundation, 2020.
- [46] Conrad Kaiser, Rebecca Yu, David Madras, Rebecca Eynon, Michael Veale, Vidushi Marda, Eun Seo Jo, Reuben Binns, Shakir Mohamed, Ziad Obermeyer, et al. Measuring gender and racial biases in large language models: Intersectional evidence from automated resume evaluation. Patterns , 5(4):100974, 2024.
- [47] Jon Kleinberg, Himabindu Lakkaraju, Jure Leskovec, Jens Ludwig, and Sendhil Mullainathan. Human decisions and machine predictions. The quarterly journal of economics , 133(1):237-293, 2018.
- [48] Jon M. Kleinberg and Manish Raghavan. Selection problems in the presence of implicit bias. In ITCS , volume 94 of LIPIcs , pages 33:1-33:17. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2018.
- [49] Junpei Komiyama and Shunya Noda. On statistical discrimination as a failure of social learning: A multi-armed bandit approach. Management Science , 69(6):3331-3350, 2023. Accessed: 2025-05-06.
- [50] Ratul Lahkar and Saptarshi Mukherjee. Optimal large population Tullock contests. Oxford Open Economics , 2:odad003, 05 2023.
- [51] Khen Lampert. Meritocratic Education and Social Worthlessness . Palgrave Macmillan, 2012.
- [52] Tamar Lewin. A New SAT Aims to Realign With Schoolwork, March 2014. https://www.ny times.com/2014/03/06/education/major-changes-in-sat-announced-by-colle ge-board.html .
- [53] Bo Li, Zenan Wu, and Zeyu Xing. Optimally biased contests with draws. Economics Letters , 226:111076, 2023.
- [54] Richard J. Lipton, Evangelos Markakis, and Aranyak Mehta. Playing large games using simple strategies. In EC , pages 36-41. ACM, 2003.
- [55] Xuyuan Liu and Jingfeng Lu. Optimal prize-rationing strategy in all-pay contests with incomplete information. International Journal of Industrial Organization , 50:57-90, 2017.
- [56] Karen S Lyness and Madeline E Heilman. When fit is fundamental: performance evaluations and promotions of upper-level female and male managers. Journal of Applied Psychology , 91(4):777, 2006.
- [57] Daniel Markovits. The Meritocracy Trap . Penguin Press, New York, 2019.

- [58] McKinsey Global Institute. Unlocking the full potential of women in the u.s. economy. Technical report, McKinsey &amp; Company, New York, NY, 2011.
- [59] Stephen J. McNamee and Robert K. Miller Jr. The Meritocracy Myth . Rowman &amp; Littlefield, 2004.
- [60] Smitha Milli, John Miller, Anca D. Dragan, and Moritz Hardt. The social cost of strategic classification. In FAT , pages 230-239. ACM, 2019.
- [61] Corinne A. Moss-Racusin, John F. Dovidio, Victoria L. Brescoll, Mark J. Graham, and Jo Handelsman. Science faculty's subtle gender biases favor male students. Proceedings of the National Academy of Sciences , 109(41):16474-16479, 2012.
- [62] Eunmi Mun and Naomi Kodama. Meritocracy at work?: Merit-based reward systems and gender wage inequality. Social Forces , 100(4):1561-1591, June 2022. Publisher Copyright: © 2021 The Author(s) 2021. Published by Oxford University Press on behalf of the University of North Carolina at Chapel Hill. All rights reserved.
- [63] Audrey Murrell. The privilege, bias, and diversity challenges in college admissions, 5 2019.
- [64] National Center for Education Statistics. The condition of education 2020. Technical report, U.S. Department of Education, 2020.
- [65] Wojciech Olszewski and Ron Siegel. Large contests. Econometrica , 84(2):835-854, 2016.
- [66] June O'Neill. Explaining the gender wage gap. Economic Policy Review , 9(2):22-30, 2003.
- [67] Scott E. Page. The Diversity Bonus: How Great Teams Pay Off in the Knowledge Economy . Princeton University Press, Princeton, NJ, 2017.
- [68] Pew Research Center. The enduring grip of the gender pay gap, 3 2023.
- [69] Edmund S Phelps. The statistical theory of racism and sexism. The american economic review , 62(4):659-661, 1972.
- [70] Robert S Pindyck and Daniel L Rubinfeld. Microeconomics . Prentice Hall, 1997.
- [71] Press Information Bureau, Government of India. JEE (Advanced) 2024: Gender-wise statistics of candidates registered and qualified. https://pib.gov.in/PressReleasePage.aspx?P RID=2023653 , 2024. Accessed: 2025-05-07.
- [72] Elaine D Pulakos. Selection assessment methods. Society for Human Resource Management (SHRM) Foundation, 2005. United States of America.
- [73] Manish Raghavan, Solon Barocas, Jon M. Kleinberg, and Karen Levy. Mitigating bias in algorithmic hiring: evaluating claims and practices. In Mireille Hildebrandt, Carlos Castillo, Elisa Celis, Salvatore Ruggieri, Linnet Taylor, and Gabriela Zanfir-Fortuna, editors, FAT* '20: Conference on Fairness, Accountability, and Transparency, Barcelona, Spain, January 27-30, 2020 , pages 469-481, Barcelona, Spain, 2020. ACM.
- [74] Robert B. Reich. The Future of Success . Knopf, New York, 2000.
- [75] Lauren A. Rivera. Pedigree: How Elite Students Get Elite Jobs . Princeton University Press, Princeton, NJ, 2015.
- [76] Michael J. Sandel. The Tyranny of Merit: What's Become of the Common Good? Farrar, Straus and Giroux, 2020.
- [77] Thomas C Schelling. The Strategy of Conflict . Harvard University Press, Cambridge, MA, 1981.
- [78] Jessica Schieder and Elise Gould. Women's work and the gender pay gap: How discrimination, societal norms, and other forces affect women's occupational choices-and their pay, 2016.
- [79] Christian Seel and Cédric Wasser. On optimal head starts in all-pay auctions. Economics Letters , 124(2):211-214, 2014.

- [80] The Journal of Blacks in Higher Education. Black students remain underrepresented at the nation's most selective colleges and universities, 2023.
- [81] Robert D. Tollison. Rent-seeking: A survey. Kyklos , 35(4):575-602, 1982.
- [82] Gordon Tullock. The welfare costs of tariffs, monopolies, and theft. Western Economic Journal , 5(3):224-232, 1967.
- [83] U.S. Bureau of Labor Statistics. Median earnings for women in 2021 were 83.1 percent of the median for men, 2022.
- [84] Joseph Walker. Meet the New Boss: Big Data, September 2012. https://www.wsj.com/ar ticles/SB10000872396390443890304578006252019616768 .
- [85] Cédric Wasser and Mengxi Zhang. Differential treatment and the winner's effort in contests with incomplete information. Games and Economic Behavior , 138:90-111, 2023.
- [86] Lyss Welding. Women in higher education: Facts &amp; statistics, 2023.
- [87] Christine Wennerås and Agnes Wold. Nepotism and Sexism in Peer-Review. Nature , 387(6631):341-343, May 1997.
- [88] Rick Wicklin. The truncated normal distribution in sas. SAS Blogs, 2013. Available online: https://blogs.sas.com/content/iml/2013/07/24/the-truncated-normal-in-s as.html .
- [89] Jazzlin Yee. Opinion: Females are underrepresented in computer science, 2022.
- [90] Feng Zhu. On optimal favoritism in all-pay contests. Journal of Mathematical Economics , 95:102472, 2021.

## Contents

| 1                  | Introduction                                                                           | Introduction                                                                           | 1   |
|--------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-----|
| 2                  | Model and metrics                                                                      | Model and metrics                                                                      | 3   |
| 3                  | Theoretical results: Nash equilibrium and metrics for large n                          | Theoretical results: Nash equilibrium and metrics for large n                          | 4   |
| 4                  | Analysis: metric behavior and intervention design                                      | Analysis: metric behavior and intervention design                                      | 7   |
| 5                  | Conclusion, limitations, and future work                                               | Conclusion, limitations, and future work                                               | 10  |
| A Detailed related | work                                                                                   | work                                                                                   | 18  |
|                    | A.1 Comparison of the two-group contest with relevant models .                         | . . . . .                                                                              | 19  |
| B                  | Illustrative examples for the two-group case                                           | Illustrative examples for the two-group case                                           | 20  |
| C                  | Analysis of finite NE policies in the uniform distribution case                        | Analysis of finite NE policies in the uniform distribution case                        | 21  |
|                    | C.1 Empirical analysis                                                                 | . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                | 21  |
|                    | C.2 Theoretical analysis .                                                             | . . . . . . . . . . . . . . . . . . . . . . . . . .                                    | 22  |
| D                  | Other bias models and analysis of metrics for their Nash equilibrium                   | Other bias models and analysis of metrics for their Nash equilibrium                   | 23  |
|                    | D.1 Other bias models . .                                                              | . . . . . . . . . . . . . . . . . . . . . . . . . .                                    | 23  |
|                    | D.2 Analysis of metrics for Nash equilibrium in the truncated normal distribution case | D.2 Analysis of metrics for Nash equilibrium in the truncated normal distribution case | 27  |
| E                  | Proof of Theorem 3.1: two-group contest                                                | Proof of Theorem 3.1: two-group contest                                                | 28  |
|                    | E.1 Technical overview .                                                               | . . . . . . . . . . . . . . . . . . . . . . . . . .                                    | 28  |
| ing it             | E.1.1 NE policy for the undifferentiated contest for finite n and obstacle in extend-  | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                            | 29  |
|                    | E.1.2                                                                                  | A conjectured NE policy in two-group contests for large n . .                          | 29  |
|                    | E.1.3                                                                                  | Proving convergence to the conjectured NE policy . . . . . .                           | 31  |
| E.2                | A more                                                                                 | comprehensive version of Theorem 3.1: convergence form . .                             | 32  |
| E.3                | Proof of Theorem E.2                                                                   | . . . . . . . . . . . . . . . . . . . . . . . . . .                                    | 33  |
|                    | Proof of Lemma E.1: solution uniqueness                                                | . . . . . . . . . . .                                                                  | 33  |
| E.3.2              | E.3.1                                                                                  | Bounding winning probability . . . . . . . . . . . . . . . . .                         | 34  |
|                    | E.3.3                                                                                  | Proof that A ( n ) is approximate NE . . . . . . . . . . . . . . .                     | 36  |
|                    | E.3.4                                                                                  | Completing the proof of Theorem E.2 . . . . . . . . . . . . .                          | 36  |
|                    | E.4                                                                                    | Comparing with a distributional two-group contest . . . . . . . . . .                  | 37  |
|                    | E.5                                                                                    |                                                                                        | 38  |
| F                  | An alternative proof using an infinite contest . . . . . . . . . . . . . .             | An alternative proof using an infinite contest . . . . . . . . . . . . . .             | 40  |
|                    | Omitted details for uniform distribution analysis from Sections 3 and 4                | Omitted details for uniform distribution analysis from Sections 3 and 4                |     |
| G                  | Additional details to Section 4                                                        | Additional details to Section 4                                                        | 44  |
|                    | G.1 Case study -estimating perceived bias from JEE Advanced 2024                       | . .                                                                                    | 44  |

| G.2   | Robustness analysis for findings in Section 4 . . . . . . . . . . . . . . . . . . . . . 45   |
|-------|----------------------------------------------------------------------------------------------|
| G.3   | An illustrative example: interpreting and applying the model . . . . . . . . . . . . 46      |
| G.4   | Details for alternative interventions . . . . . . . . . . . . . . . . . . . . . . . . . . 46 |

## A Detailed related work

Meritocratic selection process, pay gap, and statistical discrimination feedback loop. In the social sciences, there is a large body of work that studies meritocratic selection processes and their limitations; see [59, 57, 76] and the references therein. [15, 62] discuss the pay gap in meritocratic systems, shedding light on how merit-based reward systems and gender wage gaps intersect. [38], in an extensive line of work, discusses the gender pay gap and addresses the economic and social factors contributing to wage disparities between men and women. Another line of research focuses on studying statistical discrimination feedback loops, which model how firms update their beliefs about group quality over time, reinforcing disparities [3, 69, 21, 23, 5, 49]. For instance, [3] emphasizes how the cost of individualized assessment incentivizes reliance on priors, which can become self-fulfilling and reinforce structural inequality. [3] models a profit-maximizing employer who faces noisy signals of productivity and rationally uses group-level statistics, leading to persistent wage gaps even with equal underlying abilities. [21] show that pessimistic beliefs about a group's productivity can result in tougher standards, reduced investment incentives, and discriminatory equilibria. [23] extend this to two-sided settings where firms and workers both act on noisy beliefs, reinforcing low-investment, low-opportunity equilibria. A key distinction, as we understand it, is that classical models of statistical discrimination typically generate disparities through imperfect and group-dependent beliefs about identical underlying abilities. In contrast, our framework allows perfect, unbiased information at the institutional level and identical selection criteria for all candidates. We focus instead on valuation asymmetries -that is, differences in the perceived benefit of success across groups-and show that these differences alone can lead to disparities in effort and representation, even under meritocratic selection.

All-pay auctions and Tullock contests. In game theory, there is a significant body of literature that investigates all-pay auctions. For instance, [79, 55, 85] study the setting in which every agent knows their private valuations and the distribution of other agents. Specifically, [79] study a 'biased' 2-agent contest in which the designer is allowed to give a 'headstart' to the effort of one agent. This headstart can be interpreted as differing merits of the agents, which corresponds to the initial abilities in our model. They characterize the optimal design for maximizing the expected highest effort or total effort of agents. In this case, the bias is introduced by the designer, rather than inherent in the system. [85] study the undifferentiated case for a single winner, while the contest designer is allowed to select the contest success function (CSF) based on agents' efforts. Their main focus is on studying the optimal design of the CSF that maximizes the total expected effort. The main difference from our model is that they consider bias in the efforts instead of in the valuations.

The all-pay auction with complete information has also been well-studied. Unlike the setting in this paper, these works assume that the valuations of all agents are known. [7] initiated the study of an n -agent k -winner all-pay auction and provided a complete characterization of the NE distribution. A line of research investigates the optimal design for maximizing the total expected effort/revenue, including imposing a multiplicative bias on the effort of agents [32, 36] or introducing an additional headstart [35, 33, 34, 36, 90].

Tullock contests [82, 81, 30, 34, 25, 53, 50] model the probability of winning based on relative effort without direct costs for participation, whereas an all-pay auction requires all agents to pay their bid amounts regardless of winning, with only the highest bidder(s) securing the prize. [65] study the dynamics of large contests, where a significant number of agents compete. Such contests pose unique analytical challenges and offer insights into the behavior of agents in mass competition scenarios. The works of [32, 33, 34] also investigate how the design of contests can be optimized to maximize revenue, considering factors like bias in efforts, headstarts, and the structure of the CSF. Across these studies, a common theme is the characterization of Nash equilibrium strategies within the context of different contest models, and identifying designs that encourage maximal effort or revenue.

Strategic classification and ranking. Another related direction is strategic learning, which mainly includes strategic classification [14, 44, 60, 9] and strategic ranking [29]. In strategic classification, agents can exert effort to alter their features to achieve higher values according to the published classifier. The designer's aim is to select a classifier that is robust to the manipulation of inputs by strategic agents. However, in this setting, agents' efforts are influenced solely by the published classifier, with no competition among them. In strategic ranking problems, agents' payoffs depend on their post-ranking, which is determined by a combination of their prior rankings and efforts. While

there is competition in this problem, all agents have the same valuation, which is different from our model.

Models of bias in valuations. Several works have modeled group-level biases based on empirical observations [4, 10, 48, 28, 18]. Additive and multiplicative skews in the valuations have also been modeled [48, 10]. [48] consider valuations v &gt; 0 of the advantaged group distributed according to the uniform or Pareto density and, for the disadvantaged group, they model the output as v/β for some fixed β ≥ 1 . We consider a class of bias models inspired by this model, the ρ in our case corresponds exactly to 1 /β . The implicit variance model of [28] models differences in the amount of noise in the valuations for individuals in different groups. Here, the output estimate is drawn from a Gaussian density whose mean is the valuation e (which can take any real value) and whose variance depends on the group of the individual being evaluated: The variance is higher for individuals in the disadvantaged group compared to individuals in the advantaged group. [18] proposes an optimization-based approach to model how group-wise valuation distributions can be obtained by tuning parameters such 'information constraints' or 'risk aversion'.

## A.1 Comparison of the two-group contest with relevant models

To the best of our knowledge, our model is novel and has not been studied in the literature. Below, we compare our model with the most relevant models. Firstly, [1] examines a specific case of our model with n = 2 , k = 1 , and α = 0 . 5 , demonstrating the existence of a unique NE under certain conditions. While their analysis is limited to a two-player scenario, our model generalizes this by considering any number of players and allowing for multiple winners. [29] propose another two-group contest model. However, a key distinction in our model is the consideration of asymmetric valuation distributions across groups, whereas [29] introduces bias through the cost of effort, assuming symmetric valuations for all agents. This asymmetry in valuation distributions in our model adds complexity to the analysis.

Another related work is [27], which explores an all-pay auction with two groups. In their model, agents' abilities are symmetrically distributed, and those in the advantaged group may receive additional rewards with equivalent bids. Despite the symmetric strategic environment in [27], our model features asymmetric valuation distributions between groups, resulting in an asymmetric strategic environment. This asymmetry introduces further computational challenges for deriving the NE; details can be found below.

A detailed comparison with [27]. We provide a detailed comparison between our two-group model and that in [27]. The primary distinction is that their model results in a symmetric strategic environment, while ours creates an asymmetric one. Below, we provide further details on this difference.

In the model of [27], each agent belongs to the target group with probability µ or to the non-target group with probability 1 -µ , independently of the other agents. The ability of an agent is then drawn i.i.d. from distribution F if they belong to the target group, and from G if they belong to the non-target group. As a result, the ability of each agent is drawn identically and independently from the joint distribution µF +(1 -µ ) G . Let H be the CDF of this joint distribution. The probability that a given ability v is among the top k abilities is then given by ∑ n -1 i = n -k ( n -1 i ) H ( v ) i (1 -H ( v )) n -1 -i , which is the same for each agent, thereby resulting in a symmetric strategic environment.

In our model, consider a simplified case where p a is a point mass at 0. Then, F 1 and F 2 correspond to the cumulative distribution functions (CDFs) of p 1 and p 2 , which represent the valuation densities of G 1 and G 2 , respectively. The probability that, for an agent in G 1 , a given valuation v is among the top k valuations is given by:

<!-- formula-not-decoded -->

In contrast, the probability for an agent in G 2 is given by

<!-- formula-not-decoded -->

̸

These two expressions differ whenever p 1 = p 2 , leading to asymmetry in the strategic environment. This asymmetry significantly complicates the computation of the order statistics for the ( k -1) -th

effort compared to the symmetric ones. E.g., for strategies s 1 and s 2 , let F s ℓ ( v ) denote the CDF of efforts s ℓ ( v ) when v ∼ p ℓ . The cumulative distribution of the ( k -1) -th effort e ⋆ from an agent in G 1 is then given by:

<!-- formula-not-decoded -->

In contrast, the cumulative distribution of the ( k -1) -th effort e ⋆ from an agent in G 2 is given by:

<!-- formula-not-decoded -->

In the symmetric ones ( s 1 = s 2 = s ), the computation simplifies to:

<!-- formula-not-decoded -->

Thus, the calculus and approximations for the two-group contest is significantly more difficult, making it harder to arrive at the equilibrium policies than in the contest with a symmetric strategic environment.

## B Illustrative examples for the two-group case

In this section, we present a two-agent example with a biased valuation distribution to illustrate both the difficulty of computing the Nash equilibrium (NE) policy and the significant impact of valuation bias on the contest outcome. Let c = 0 . 5 . Let the density p 1 of agent 1 be the uniform distribution on Ω 1 = [0 , 1] and p 2 of agent 2 be the ρ -biased version of p 1 supported on Ω 2 = [0 , ρ ] . Let the density p a be a point mass at 0. Let A ℓ : Ω ℓ → R ≥ 0 be the NE policy that maps valuation v ℓ to effort A ℓ ( v ℓ ) . We assume A ℓ is monotonically increasing on the domain Ω ℓ .

In this example, if agent 1 puts in effort e , it wins if the effort of agent 2 is smaller than e . Thus, its winning probability P 1 = A -1 2 ( e ) ρ 4 and its payoff is π 1 ( v, e ; A 2 ) = A -1 2 ( e ) ρ v -e . Similarly, if agent 2 puts in effort e , its winning probability P 2 = A -1 1 ( e ) and payoff is π 2 ( v, e ; A 1 ) = A -1 1 ( e ) v -e . Then, by the stability condition (1), we have ∂π ℓ ( v,e ; A 3 -ℓ ) ∂e | e = A ℓ ( v ) = 0 , implying that

<!-- formula-not-decoded -->

Solving this gives us the following explicit forms:

<!-- formula-not-decoded -->

Specifically, when ρ = 1 (the unbiased case), we have A 1 = A 2 = A , which simplifies the stability condition to A ′ ( v ) = v , yielding the NE policy A ( v ) = v 2 2 . Also note that for v ∈ Ω 2 ,

<!-- formula-not-decoded -->

which implies that A 1 ( v ) ≤ A 2 ( v ) . Thus, agent 2 is more inclined to put in greater effort than agent 1 for identical valuations.

4 Here, we assume A -1 2 ( e ) = ρ if A 2 ( ρ ) &lt; e .

Imagine an institute that is unaware of the bias in valuations across two agents and thus applies the unbiased NE policy A ( v ) = v 2 2 to predict the contest outcome. For instance, it would predict the average revenue

<!-- formula-not-decoded -->

where the merit function is m ( t ) = t . However, under a ρ -biased valuation distribution, the true average revenue is RV ρ = ρ 2( ρ +1) , which decreases monotonically with ρ . This implies that the institute could overestimate its expected benefit RV by a fraction of

<!-- formula-not-decoded -->

which amounts to approximately 13% when ρ = 0 . 8 . This example underscores the importance of studying asymmetric valuations and highlights the relevance of our proposed metrics for analyzing their impacts.

We also observe that even for this simple two-agent example, the stability condition is considerably more complex than in the undifferentiated case. In more general settings-such as those involving multiple spots, non-uniform valuation densities, or non-trivial ability densities-the explicit forms of NE policies for a two-group contest become even more complicated, making direct computation and explicit analysis impractical.

## C Analysis of finite NE policies in the uniform distribution case

In this section, we use the uniform distribution example from Section 3 as a running example, introduce a dynamic algorithm (Algorithm 1) to approximate the finite NE policies, and perform a statistical comparison between the finite and infinite cases. Additionally, we provide a theoretical analysis of the closeness between the NE policies and associated metrics in the finite and infinite cases.

## C.1 Empirical analysis

Dynamics for computing finite NE policies. Recall that we consider p 1 = p = Unif[0 , 1] , p 2 = p 2 = Unif[0 , ρ ] for ρ ∈ [0 , 1] , and p a ≡ 0 . Algorithm 1 presents a dynamic procedure to approximately compute the finite-population NE policies.

We initialize each group's policy s (0) ℓ with a smoothness variant of the infinite NE policy (Lines 3-4), then iteratively update these policies over N steps and return the final output as an approximation of the finite NE (Lines 5-38). At each iteration t :

1. We first update the effort set E t based on the policies s ( t -1) ℓ from the previous iteration (Line 6). Since the action space is continuous, we restrict agents to choose efforts only from this finite set E t .
2. Next, we update the policy for group G 1 using the policy s ( t -1) 2 from the previous iteration (Lines 7-21). The computation is performed over a finite set V (1) of discrete valuation levels (Line 1). For each valuation v , we determine the best-response effort that maximizes the agent's expected payoff by computing winning probabilities through a convolution of binomials (Lines 9-19). Specifically, we set p 1 = 1 -v in Lines 9 and 12, consistent with the monotonicity constraint enforced in Line 11. Finally, Line 21 updates the policy using a carefully chosen step size a ( t ) ℓ to ensure convergence.
3. We then update the policy for group G 2 based on the policy s ( t -1) 1 from the previous iteration (Lines 7-21). This process mirrors that of G 1 , with the main difference lying in the computation of winning probabilities for each effort in E t due to the asymmetric valuation distributions.

The resulting policies s ℓ = s ( T ) ℓ are defined on discrete valuation grids. To obtain continuous policies, we interpolate them by connecting adjacent valuation points with straight lines, resulting in piecewise-linear approximations.

Choice of hyperparameters. In our simulations, we set the valuation resolution m v = 101 , effort resolution m e = 101 , total number of iterations T = 500 , and step sizes a ( t ) 1 = a ( t ) 2 = 1 10 T . We always set n 1 = n 2 = n 2 , which means α = 0 . 5 .

Metrics. For each iteration t , we compute the following metric to evaluate the updated policies s ( t ) ℓ :

<!-- formula-not-decoded -->

which quantifies the average policy update for group G ℓ at iteration t . Intuitively, a decreasing ∆ ( ℓ,t ) indicates convergence of the policy sequence s ( t ) ℓ . However, since we work with discretized valuation and effort sets, we do not expect ∆ ( ℓ,t ) to vanish entirely.

Results. Figures 5, 6, 7, and 8 present the evolution of equilibrium policies for population sizes n ∈ { 20 , 200 , 600 , 1200 } across four time snapshots t ∈ { 50 , 150 , 300 , 500 } , with fixed parameters ρ = 0 . 8 and c = 0 . 2 . Although all runs begin with a smoothed version of the infinite-population NE, the dynamics vary significantly with population size. For small n (e.g., n = 20 ), we observe noticeable fluctuations in early iterations, particularly in group G 2 , whose valuation distribution is more concentrated. By t = 500 , both policies stabilize, though they retain visible irregularities due to stochasticity in rank-based feedback. Even though all runs begin with a smooth initialization based on the infinite-population NE, the dynamics unfold differently depending on population size. For small n (e.g., n = 20 ), we observe noticeable fluctuations in the early iterations, especially in group G 2 , whose valuation distribution is more concentrated. At t = 500 , the policies stabilize but retain some irregularity, reflecting noise in the agent-level ranking and feedback structure.

As n increases, both groups' policies become smoother and stabilize more quickly. By n = 600 , the effort policies align closely with the infinite NE, and further updates beyond t = 300 are negligible. These trends are confirmed by the convergence plots in Figure 9, which show a sharp reduction in the ℓ 1 -norm policy update ∆ ( ℓ,t ) with increasing n . Group G 1 consistently converges faster than G 2 , a pattern attributable to its broader valuation support and greater flexibility in effort choice. Overall, the results illustrate that the infinite-population equilibrium is a good predictor even for moderately sized finite systems, while also quantifying the transient effects and instability that emerge in lown regimes.

Interestingly, we also observe from these plots that when n is small ( n = 20 , 200 ), s 2 ( v ) &gt; s 1 ( v ) , while for larger values of n ( n = 600 , 1200 ), s 2 ( v ) &lt; s 1 ( v ) . In the subsequent subsection, we will provide a theoretical analysis to explain the underlying reasons for this behavior.

## C.2 Theoretical analysis

We begin by presenting theoretical evidence for the alignment between finite and infinite NEs, a relationship that is observed empirically. In the proof of Theorem 3.1 (see Section E), we show that for any finite n , ε n -NE policy s ( n ) stated in Theorem 3.1 is ' O ( √ log n/n ) -close' to the policy s for infinite n and is ' O ( √ log n/n ) -close' to an NE policy. Specifically, when p 1 , p 2 are uniform distributions and p a is a point mass at 0, for any constant α , the closeness between s ( n ) and s can be directly translated into the policy form: s ( n ) = 0 for v &lt; t -O ( √ log n/n ) and s ( n ) = t for v ≥ t -O ( √ log n/n ) . For general p 1 , p 2 , p a , we note that the O ( √ log n/n ) -closeness depends on the concept of densities, which is more complex. Corollary 3.2 from [20] implies that the NE policy must be symmetric within each group. Let s 1 represent the policy for G 1 and s 2 for G 2 . The above analysis indicates that the closeness between s 1 , s 2 , and s (from Equation (3)) is expected to be bounded by O ( √ log n/n ) .

In the following, we analyze the empirical observations regarding the scaling of s 1 and s 2 . Intuitively, s 1 ( v ) increases from approximately 0 to approximately t as v increases from t -√ log n/n to t + √ log n/n . If an agent in G 2 exerts an effort of t (1 -√ log n/n ) , the agent's winning probability could exceed 95%. This observation motivates the choice of setting s 2 ( v ) = t (1 -√ log n/n ) , rather than 0, to generate positive profits when v &gt; t (1 -√ log n/n ) / 95% . As a result, if t (1 -

√ log n/n ) / 95% ≤ t , i.e., √ log n n ≥ 0 . 05 , then s 2 ( t ) ≥ t (1 -√ log n/n ) ≥ s 1 ( v ) . Therefore, when n is small and √ log n n ≥ 0 . 05 , it holds that s 2 ( v ) &gt; s 1 ( v ) for v ∈ Ω 2 . Conversely, when n is large, √ log n n becomes small, and s 1 ( v ) &gt; s 2 ( v ) , as agents in G 2 consistently receive lower payoffs than those in G 1 when s 1 = s 2 . This behavior explains the observed scaling between s 1 and s 2 , as discussed in Section C.1.

Finally, we discuss the impact on metrics when the number n of agents is finite. Recall that in the finite n case, we can assume NE policies s 1 and s 2 for group G 1 and G 2 , respectively. As discussed above, when n is not too small, we have 1) s 1 &gt; s 2 and 2) s 1 , s 2 , and s are O ( √ log n/n ) -close. Since s 1 &gt; s 2 and they converge to the same policy as n grows, the representation ratio R 1 ( A ) = E ( | S ∩ G 1 | | G 1 | ) decreases with n , while R 2 ( A ) = E ( | S ∩ G 2 | | G 2 | ) increases with n , where S is the (random) winning set. Consequently, the representation ratio r R ( A ) = R 2 ( A ) R 1 ( A ) increases as n grows. Since the gap between s 1 and s 2 is bounded by O ( √ log n/n ) , this results in an increase of O ( √ log n/n ) in R 2 ( A ) and a similar decrease of O ( √ log n/n ) in R 1 ( A ) compared to the infinite case. Thus, the increase in r R ( A ) should be bounded by O ( √ log n/n ) . A similar quantitative analysis applies to the social welfare ratio r S ( A ) and average revenue RV ( A,m ) .

Figure 5: Evolution of group effort policies over time for n = 20 , ρ = 0 . 8 , and c = 0 . 2 .

<!-- image -->

## D Other bias models and analysis of metrics for their Nash equilibrium

## D.1 Other bias models

A natural extension of p 1 = Unif[0 , 1] in Section 2 is when p 1 is the density of the uniform distribution on an interval [ a, b ] ( 0 &lt; a &lt; b ≤ ∞ ) and p 2 is the density of the uniform distribution on Ω 2 = [ ρa, ρb ] . Then p 2 ( v ) = 1 ρ ( b -a ) and again, E v ∼ p 2 [ v ] = ρ · a + b 2 = ρ · E v ∼ p 1 [ v ] . More generally, one might consider a density p 1 that is supported on a domain Ω 1 = [ a, ∞ ] , along with a ρ -biased density defined as p 2 ( v ) = 1 ρ p 1 ( v ρ ) for ρ ∈ (0 , 1] and v ∈ [ ρa, ∞ ] .

Besides the uniform distribution case, we consider valuations coming from a truncated normal distribution supported on [0 , 1] . Formally, let p 1 be the density of a truncated normal distribution N ( µ, σ 2 ) on the interval Ω 1 = [0 , 1] , where µ lies within (0 , 1) and σ &gt; 0 . Let p 2 be the density of a

<!-- formula-not-decoded -->

Figure 6: Evolution of group effort policies over time for n = 200 , ρ = 0 . 8 , and c = 0 . 2 .

<!-- image -->

Figure 7: Evolution of group effort policies over time for n = 600 , ρ = 0 . 8 , and c = 0 . 2 .

<!-- image -->

Figure 8: Evolution of group effort policies over time for n = 1200 , ρ = 0 . 8 , and c = 0 . 2 .

<!-- image -->

Figure 9: Convergence of group-wise policy updates ∆ ( ℓ,t ) for different population sizes n , with fixed parameters ρ = 0 . 8 and c = 0 . 2 .

<!-- image -->

Figure 10: Statistics for truncated normal distribution with µ = 0 . 5 , σ = 0 . 1 .

<!-- image -->

Figure 11: Plots of t versus ρ for various α with c = 0 . 1 for the truncated normal distribution. The dotted line t 1 = 0 . 9 corresponds to the undifferentiated contest with density p = p 1 .

<!-- image -->

truncated normal distribution N ( ρµ, σ 2 ) on the interval Ω 2 = [0 , 1] . Since the bias is multiplicative, the domain of p 1 , Ω 1 = [0 , 1] , does not influence the assessment of the contest's results. Note that E v ∼ p 2 [ v ] = ρµ + ϕ ( -ρµ σ ) -ϕ ( 1 -ρµ σ ) Φ( 1 -ρµ σ ) -Φ( -ρµ σ ) , where ϕ ( x ) is the probability density function of the standard normal distribution N (0 , 1) and Φ( x ) is its cumulative distribution function. The expectation of p 2 does not decrease linearly with ρ as in the uniform case, but it closely approximates a linear function and monotonically decreases with ρ . This is motivated by real-world settings where the valuations (such as pay or SAT scores) exhibit a truncated normal distribution [88]. Other variants of distributions include piecewise-linear, polynomial (such as Pareto), and log-normal distributions, along with their biased versions.

We implicitly assume that the bias parameter ρ is fixed and identical for all agents in G 2 above. However, ρ could be noisy and non-identical to agents. For instance, let p ρ be a density supported on [0 , 1] . We assume each agent i ∈ G 2 has an individual bias ρ i i.i.d. drawn from p ρ , and its valuation is drawn from the ρ i -biased density of p 1 . Then p 2 is supported on Ω 2 = Ω 1 , and satisfies that for any v ∈ Ω 1 ,

<!-- formula-not-decoded -->

## D.2 Analysis of metrics for Nash equilibrium in the truncated normal distribution case

In this section, we do a similar analysis as in Section 4 for the case that p 1 is a truncated normal distribution N ( µ, σ 2 ) supported on [0 , 1] , p 2 is a ρ -biased truncated normal distribution N ( ρµ, σ 2 ) supported on [0 , 1] , and p a is a point mass at 0. We choose µ = 0 . 5 and σ = 0 . 1 . This selection ensures that the density function is narrowly focused around the mean and the expected value of p 2 is approximately ρµ ; see Figure 10 for illustration. Note that t analogues to Proposition 4.1 is the

Figure 12: Plots illustrating the group-wise social welfare S 1 ( A ) , S 2 ( A ) , and the social welfare ratio r S ( A ) as functions of the parameters ρ , c , and α for the truncated normal distribution. By default, we set ( ρ, c, α ) = (0 . 8 , 0 . 1 , 0 . 5) . A dotted line within these plots indicates the threshold at which r S ( A ) = 0 . 8 .

<!-- image -->

solution of the following equation:

<!-- formula-not-decoded -->

Unlike the uniform distribution case, it is hard to derive closed-form expressions for metrics on the outcomes of the contest. However, one can do numerical computations and we plot solution t , representation ratio r R ( A ) and group-wise social welfare S ℓ ( A ) together with social welfare ratio r S ( A ) in Figures 11 and 12 respectively. All plots exhibit a monotonic behavior similar to that observed with the uniform distribution. Next, we highlight some distinctions with the uniform distribution.

No inflection point. Anotable feature of the truncated normal distribution is its lack of an inflection point. This trait is observed not only for t , but also in the behaviors of r R ( A ) and r S ( A ) . This difference arises because the domain Ω 2 = [0 , 1] remains consistent across all values of ρ .

Representation ratio. Figure 12 shows that to achieve a representation ratio r R ( A ) ≥ 0 . 8 , it is necessary to adjust ρ to a minimum of 0.979 or increase c to at least 0.862. The need to elevate c is more pronounced than the required 0.5 observed with the uniform distribution in Figure 2(b). This difference arises because the truncated normal distribution tends to be more focused around its mean, leading to a higher number of agents in G 1 possessing valuations greater than the expected value ≈ 0 . 4 of p 2 .

## E Proof of Theorem 3.1: two-group contest

In this section, we begin by providing a more detailed technical overview of the proof of Theorem 3.1 (Section E.1). Next, we present a more comprehensive version of Theorem 3.1, including the explicit form of A ( n ) (Theorem E.2 in Section E.2). Finally, we provide the proof of Theorem E.2 (Section E.3).

## E.1 Technical overview

We present an overview of the proof of Theorem 3.1, which characterizes an NE policy for the two-group contest. Recall that, there are n agents belonging to one of the two disjoint groups G 1 , G 2 with | G 1 | = (1 -α ) n and | G 2 | = αn . The valuations of agents in G 1 come from the density p 1 supported on Ω 1 and those of G 2 come from the density p 2 supported on Ω 2 . Each agent has an initial ability drawn from the density p a . The selectivity of the contest is a constant 0 &lt; c &lt; 1 . Theorem 3.1 asserts that, as n →∞ , there is an NE policy for the agents which is determined by a threshold t ∈ (Ω 1 ∪ Ω 2 ) + Ω a when the (Ω 1 ∪ Ω 2 ) + Ω a is a connected subset of R ≥ 0 .

For ease of analysis, we first consider the simple case where p a is a point mass at 0 , so that agents' policies only depend on their valuations. We then show that the extension to a general p a is straightforward. We start by first quickly showing how to compute the NE policy in the special case when p 1 = p 2 for finite n and why this approach does not extend to the two-group setting of interest (Section E.1.1). In Section E.1.2 we show that even though there are major challenges in extending the one-group case, it leads us to the right form of the NE policy for the two-group case as n →∞ : a single threshold function that defines the strategies of agents in both G 1 and G 2 . This also explains how we arrive at Equation (2) that characterizes the threshold t . Finally, in Section E.1.3, we present the approach to formally argue about and prove the convergence of the finite n two-group contest to

this pair of NE policies. This analysis also reveals why the conjectured policy is a Nash equilibrium. With all the background, we conclude Theorem 3.1.

## E.1.1 NE policy for the undifferentiated contest for finite n and obstacle in extending it

Here, we consider the special case when p 1 = p 2 = p is the density of the uniform distribution on [0 , 1] , and p a is a point mass at 0. The argument for other densities is similar. First note that by symmetry among agents, it is reasonable to assume that, at equilibrium, each agent will follow the same policy s . Moreover, since the domain of p 1 is nonnegative reals, it is reasonable to assume that the NE policy is monotone in an agent's valuation. The following calculations show that both symmetry and monotonicity hold.

Recall that an agent i with valuation v is selected if the effort s ( v ) is among the top c fraction of efforts. Thus, assuming that all agents follow s , agent i is selected if and only if there are at least (1 -c ) n distinct agents j with s ( v j ) &lt; s ( v ) . (We ignore the issue that there may be ties for this discussion.) Since s is a monotone function, s ( v j ) &lt; s ( v ) holds if and only if v j &lt; v . Thus, the probability of selection of this agent is

<!-- formula-not-decoded -->

Hence, its expected payoff is π i ( v, s ( v ); A -i ) = P i ( s ( v ); A -i ) · v -s ( v ) . (See Section 2 for notation). A key observation is that the calculation of P i ( s ( v ); A -i ) only depends on density p 1 and v , and is independent of the choice of s , under the monotonicity and symmetry assumptions on s . If s is to be an NE, then it must satisfy that for any other effort e , π i ( v, s ( v ); A -i ) ≥ π i ( v, e ; A -i ) . This follows from the condition that the derivative with respect to v , π ′ i ( v, s ( v ); A -i ) = P ′ i ( s ( v ); A -i ) · v -s ′ ( v ) = 0 . As noted above, P i ( s ( v ); A -i ) does not depend on s , so we get a simple differential equation involving the derivative of s ′ ( v ) = ( ∑ n -1 j =(1 -c ) n ( n -1 j ) v j (1 -v ) n -1 -j ) ′ · v . Thus,

<!-- formula-not-decoded -->

is the unique NE for the undifferentiated contest and it that this s is monotone. For a general density p , we can apply a similar analysis to obtain that

<!-- formula-not-decoded -->

where Q p ( v ) = ∑ n -1 i = n -k ( n -1 i ) · F p ( v ) i · (1 -F p ( v )) n -i -1 for any v ∈ Ω . We also note that the computation of s can become significantly more complicated for a general density p a , as it requires considering the stability condition for two partial derivatives: ∂s ( v,a ) ∂v and ∂s ( v,a ) ∂a .

̸

We now attempt to extend the analysis above to the two-group contest. Since each agent in each group uses the same valuation density, we can still hope for a symmetric and monotone NE policy for each group; say s 1 for G 1 and s 2 for G 2 . However, we can no longer assume that s 1 = s 2 . There are simple examples (see Section B) for which it can be shown that s 1 = s 2 . This considerably complicates the calculation of probability P i for the i th agent getting selected since the order of agents' valuations may differ from that of agents' efforts. For instance, it is now possible that for agent i ∈ G 1 and agent j ∈ G 2 , v i &lt; v j but s 1 ( v i ) &gt; s 2 ( v j ) . Thus, P i must depend on functions s 1 and s 2 , instead of only depending on density p = p 1 and v as in the undifferentiated case. Thus, it is no longer possible to write a simple differential equation as in the undifferentiated case.

## E.1.2 A conjectured NE policy in two-group contests for large n

Since it seems intractable to find an NE policy for the two-group case, we study whether the situation becomes easier when n is large. This hope is rooted in the observation that for the undifferentiated contest when n is large, the NE policy s ( v ) defined in Equation (9) converges to a threshold function. To see this, recall that for the uniform distribution, s ( v ) = (1 -c ) · ∑ n j =(1 -c ) n +1 ( n j ) v j (1 -v ) n -j . Since s ( v ) is the probability associated with a sum of i.i.d. random variables, it follows from the

Chernoff bound that as n →∞ , s ( v ) → 1 -c for any v &gt; 1 -c and s ( v ) → 0 for any v &lt; 1 -c . This argument is not specific to the uniform distribution and extends to any density p 1 with NE policy defined in Equation (10). In particular, if F 1 denotes the CDF of p 1 , the limiting NE is given by s ( v ) = F -1 1 (1 -c ) if v ≥ F -1 1 (1 -c ) and s ( v ) = 0 otherwise. The threshold F -1 1 (1 -c ) guarantees that the expected fraction of agents that put in a nonzero effort is 1 -F 1 ( F -1 1 (1 -c )) = c . The rationale for s ( v ) = F -1 1 (1 -c ) when v is above the threshold is twofold: 1) agents would not exert effort beyond their valuation, ensuring s ( F -1 1 (1 -c )) ≤ F -1 1 (1 -c ) , and 2) agents with valuations below F -1 1 (1 -c ) are disincentivized from participating, leading to s ( F -1 1 (1 -c )) ≥ F -1 1 (1 -c ) .

Thus, one may hope that in a two-group contest, as n →∞ , the NE policy might similarly converge to two threshold functions s 1 and s 2 , each with a corresponding threshold t ℓ . While this assumption allows us to give an explicit form for the probability P i of agent i getting selected, this expression is quite complicated and, importantly, depends on t 1 and t 2 . Thus, we are unable to obtain conditions that determine t 1 and t 2 from the NE condition.

Going back to the setting when p 1 is the density of the uniform distribution over [0 , 1] and p 2 = p 2 is the density of the uniform distribution over [0 , ρ ] , first observe that as ρ → 0 , t 2 → 0 . Thus, one would expect t 1 to be more than t 2 . We now argue that counterintuitive to the above observation, t 1 &gt; t 2 cannot lead to an NE. To see this, first observe that when t 1 &gt; t 2 , if an agent puts in effort t 1 , then it will get selected. Thus, the probability of an agent in G 1 getting selected is 1 -F 1 ( t 1 ) . Hence, if 1 -F 1 ( t 1 ) &lt; c 1 -α , fewer than cn agents in G 1 get selected. Thus, agents in G 1 getting selected will find that putting in effort slightly larger than t 2 instead of t 1 suffices to ensure their effort is larger than all agents in G 2 , and consequently, they will still be selected. Through this reduction in effort, they can gain an additional payoff of t 1 -t 2 , which violates the stability condition. A similar argument holds for G 2 when 1 -F 1 ( t 1 ) &gt; c 1 -α . Thus, t 1 &gt; t 2 leads to instability. Similarly, we can argue that t 1 &lt; t 2 also leads to instability. Thus, in case the NE policies for G 1 and G 2 are thresholds, it must be the case that t 1 = t 2 when n is large. However, it is not clear how this can hold given that the domains of p 1 and p 2 are different.

To explore this, we consider a scenario when α = 0 . 5 , p 1 is the density of the uniform distribution over [0 , 1] and p 2 is the density of the uniform distribution over [0 , 0 . 5] ( ρ = 0 . 5 ). With high probability, there would be more than 0 . 1 n agents from G 1 whose valuation is larger than 0 . 5 . Hence, if we set c = 0 . 1 , no agent from G 2 will have any incentive to put in an effort while for agents in G 1 a threshold of t 1 = 0 . 8 suffices. The key observation is that even though the two policies are different, the policy of G 2 is just 0 . This suggests that both policies can be seen as restrictions of the same threshold function to their respective domains.

Now we show how to compute the threshold t . The idea is to reduce the two-group contest to an undifferentiated one whose density p = (1 -α ) p 1 + αp 2 , supported on the domain Ω 1 ∪ Ω 2 . In this undifferentiated contest, as n →∞ , it is likely that (1 -α ) -fraction of agents with valuation come from p 1 and α -fraction of agents come from p 2 . This suggests that these two contests are increasingly indistinguishable as n grows, leading to the same limiting threshold t = F -1 p (1 -c ) . Thus, threshold t is the solution of the equation (1 -α ) F 1 ( v ) + αF 2 ( v ) = 1 -c - the one denoted in Equation (2). This argument can be extended to general p a , resulting in the following lemma.

Lemma E.1 ( Unique solution). Let α, c ∈ (0 , 1) , p 1 be a density supported on the domain Ω 1 ⊆ R ≥ 0 , p 2 be a density supported on the domain Ω 2 ⊆ R ≥ 0 , and p a is a density supported on the domain Ω a ⊆ R ≥ 0 . If (Ω 1 ∪ Ω 2 ) + Ω a is connected and each density p 1 , p 2 , p a is positive at any point of its domain, then there exists a unique solution t ∈ Ω 1 ∪ Ω 2 for the following equation: (1 -α ) F 1 ( ζ ) + αF 2 ( ζ ) = 1 -c, where for any ζ ∈ R ≥ 0 , F ℓ ( ζ ) = Pr v ∼ p ℓ ,a ∼ p a [ v + a ≤ ζ ] .

The assumption is naturally met in cases such as the uniform distribution and the truncated normal distribution discussed in Section 2. Since F ℓ is a CDF, it must be strictly monotonic across its domain Ω ℓ +Ω a . For any ζ, ζ ′ ∈ (Ω 1 ∪ Ω 2 ) +Ω a with ζ &lt; ζ ′ , if (1 -α ) F 1 ( ζ ) + αF 2 ( ζ ) = (1 -α ) F 1 ( ζ ′ ) + αF 2 ( ζ ′ ) , we must have both F ℓ ( ζ ) = F ℓ ( ζ ′ ) holds. Then ( ζ, ζ ′ ) ∩ ((Ω 1 ∪ Ω 2 ) + Ω) = ∅ , which contradicts the connected domain assumption. Hence, (1 -α ) F 1 ( ζ ) + αF 2 ( ζ ) is strictly monotonic across the domain (Ω 1 ∪ Ω 2 ) + Ω a , which ensures the uniqueness of solution t . The proof can be found in Section E.3.1.

## E.1.3 Proving convergence to the conjectured NE policy

Now we outline how to prove that, as n →∞ , the NE policy for the two-group contest converges to the threshold policy corresponding to t as guaranteed by Lemma E.1. To do so, first, we have to make it precise what convergence means. Towards this, we revisit the undifferentiated contest. While we argued that in this case, as n →∞ , the NE policy tends to a threshold function, recall that we used the explicit form of the NE policy for finite n . Unfortunately, since we do not have an explicit form for the two-group contest (for finite n ), we need a strategy for the undifferentiated case that works without the knowledge of the explicit NE for finite n .

Let A be the NE policy of the undifferentiated contest as n → ∞ , characterized by a function s which is a threshold with parameter t . It suffices to prove that for policy A and every ε &gt; 0 there is an n ε such that for n ≥ n ε , A is an ε -NE. However, we find that there exists an ε &gt; 0 such that for any n ≥ 1 , A is not an ε -NE policy. We revisit the simple example of uniform distribution discussed above, in which c = 0 . 5 , p 1 is the density of the uniform distribution on [0 , 1] , and p a is a point mass at 0. Recall that the threshold t = 1 -c = 0 . 5 . Then, in expectation, 0 . 5 n agents have valuations at least 0 . 5 and put in effort 0 . 5 . As n →∞ , it follows from symmetry that the probability that fewer than 0 . 5 n agents with valuation ≥ t approaches approximately 0.5. Thus, an agent i with a valuation v = 0 . 2 and putting in an effort e = 0 . 01 would have about a 0.5 probability of being selected, i.e., P i ( e ; A -i ) ≈ 0 . 5 . Since s ( v ) = 0 , the probability P i ( A i ( v ); A -i ) = 0 . Thus, we have

<!-- formula-not-decoded -->

This inequality implies that when ε = 0 . 08 , for any n ≥ 1 , A is not an ε -NE policy.

To bypass this, we consider a sequence of proxies for A , denoted by A ( n ) , and characterized by threshold functions s ( n ) . These proxies aim to ensure that the winning probability P i ( e ; A -i ) ≈ 0 and hence, serve as an approximate NE policy (see Definition 3.1). Therefore, we need A ( n ) to satisfy two conditions:

1. A ( n ) converges to A as n approaches infinity, i.e., lim n →∞ s ( n ) = s , and
2. The winning probability under A ( n ) approaches zero in the limit, i.e., lim n →∞ P i ( e ; A ( n ) -i ) → 0 .

Ensuring P i ( e ; A ( n ) -i ) ≈ 0 essentially involves guaranteeing that the probability of having fewer than cn agents with valuation ≥ t is negligible. Specifically, for the uniform case, by adjusting the threshold by √ log n/n = o (1) , we define the policy s ( n ) ( v ) as follows: s ( n ) ( v ) = t if v ≥ t -√ log n/n and s ( n ) ( v ) = 0 otherwise. By concentration, this s ( n ) ensures that the probability P i ( e ; A ( n ) -i ) is bounded above by √ 1 /n (Lemma E.4). This bounded probability ensures A ( n ) to be an √ 1 /n -NE policy (Lemma E.6). This concludes the proof that the policy A is an NE in the large n limit for the uniform distribution case for one group.

Finally, we adapt this new proof technique of constructing A ( n ) to the two-group case with general densities p 1 and p 2 . Mirroring the strategy employed in the uniform distribution case, we would like to shift the threshold of s ( n ) to ensure that, in expectation, a ( c + √ log n/n ) -fraction of agents put in effort t . To satisfy this, we define the following threshold ∆ n (Definitions E.1 and E.2) for the policy A ( n ) :

<!-- formula-not-decoded -->

Consequently, we define the policy s ( n ) as follows: s ( n ) ( v ) = t if v ≥ t -∆ n and s ( n ) ( v ) = 0 otherwise (see Theorem E.2). We find that lim n →∞ ∆ n = t , indicating that s ( n ) converges to s . Crucially, such a ∆ n ensures that (1 -α ) F 1 (∆ n ) + αF 2 (∆ n ) ≤ 1 -( c + √ log n/n ) , thereby maintaining at least a ( c + √ log n/n ) -fraction of agents, in expectation, putting in effort t . Using this property, we establish a bound for the winning probability P i analogous to the undifferentiated contest: P i ( e ; A ( n ) -i ) ≤ n -(1 -α ) + n -α if e &lt; t (see Lemma E.4). The factors n -(1 -α ) and n -α derive from concentration bounds applicable to group G 1 and G 2 , respectively. Furthermore, this minimal winning probability guarantees that A ( n ) is an ε n -NE with lim n →∞ ε n = 0 (Lemma E.6). The extension to the general p a is straightforward. The main difference is that the initial ability a

may already surpass the threshold t , in which case the agent does not need to exert any effort to be selected. This is characterized by the amount of effort s ( n ) ( v, a ) = max { t -a, 0 } if v ≥ t -∆ n .

To summarize, we first arrived at a policy A that is characterized by a function s which is parameterized by a threshold t defined by Lemma E.1 and then we constructed a sequence of 'proxies' A ( n ) that converge to A as n →∞ . Moreover, we construct a sequence ε 1 , . . . , ε n , . . . with limit 0 such that A ( n ) is an ε n -NE policy for every n . This implies that A is an NE policy as n →∞ . Thus, the overview above allows us to prove Theorem 3.1.

## E.2 A more comprehensive version of Theorem 3.1: convergence form

Now we show how to construct a series of policies { A ( n ) } n that approach A , the NE policy from Equation (3) in Theorem 3.1, as n →∞ . The most technical part will be to prove that A ( n ) acts as an ε n -NE policy, where ε n → 0 with increasing n .

Suppose (Ω 1 ∪ Ω 2 ) + Ω a is connected and let t ∈ (Ω 1 ∪ Ω 2 ) + Ω a be a unique solution of the equation (1 -α ) F 1 ( ζ ) + αF 2 ( ζ ) = 1 -c (the uniqueness of t is ensured by Lemma E.1). Since c ∈ (0 , 1) , we have that either F 1 ( t ) &gt; 0 or F 2 ( t ) &gt; 0 . Accordingly, we define a threshold n t as follows.

Definition E.1 ( Threshold n t ). Given a value t ∈ Ω 1 ∪ Ω 2 , we define a threshold n t as follows:

- If both F 1 ( t ) &gt; 0 and F 2 ( t ) &gt; 0 , let n t be the smallest integer such that F ℓ ( t ) -√ log n t n t &gt; 0 for ℓ = 1 , 2 .
- If F 1 ( t ) &gt; 0 and F 2 ( t ) = 0 , let n t be the smallest integer such that F 1 ( t ) -√ log n t n t &gt; 0 .
- If F 1 ( t ) = 0 and F 2 ( t ) &gt; 0 , let n t be the smallest integer such that F 2 ( t ) -√ log n t n t &gt; 0 .

Note that such n t is finite and always exists since √ log n n → 0 as n → ∞ . Also note that n t is monotonically decreasing to t across the domain Ω 1 ∪ Ω 2 . This value of n t is useful for defining ∆ n , which is essential for the construction of policy A ( n ) .

Definition E.2 ( Threshold ∆ n ). Let n ≥ n t be an integer. We define a threshold ∆ n as follows:

- If both F 1 ( t ) &gt; 0 and F 2 ( t ) &gt; 0 , let 5

<!-- formula-not-decoded -->

- If F 1 ( t ) &gt; 0 and F 2 ( t ) = 0 , let ∆ n := F -1 1 ( F 1 ( t ) -√ log n n ) .
- Otherwise if F 1 ( t ) = 0 and F 2 ( t ) &gt; 0 , let ∆ n := F -1 2 ( F 2 ( t ) -√ log n n ) .

The requirement that n ≥ n t ensures the proper definition of ∆ n . As the number of agents n grows indefinitely, the term √ log n n approaches 0, leading ∆ n to converge towards t . The threshold function s ( n ) defined as in Equation (11), designed as a threshold function, incorporates ∆ n as its threshold. The convergence of ∆ n to t as n →∞ is crucial for ensuring that A ( n ) gradually aligns with the policy A over large populations.

We are ready to provide the formal statement of Theorem 3.1.

Theorem E.2 ( Two-group contest: Large n limit). Let α, c ∈ (0 , 1) . For ℓ = 1 , 2 , let p ℓ be a density supported on a domain Ω ℓ ⊆ R ≥ 0 . Let m : R ≥ 0 → R ≥ 0 be a merit function that is strictly increasing. Suppose Ω 1 ∪ Ω 2 is connected and each density p ℓ is positive at any point of domain

5 If the inverse function F -1 ℓ ( F ℓ ( t ) -√ log n n ) yields multiple values, it is defined to be the maximum of these values.

Ω ℓ . Let t ∈ Ω 1 ∪ Ω 2 be a unique solution of the equation (1 -α ) F 1 ( v ) + αF 2 ( v ) = 1 -c (the uniqueness of t is ensured by Lemma E.1). Let n t be defined as in Definition E.1. Let n ≥ n t be an integer and let ∆ n be defined as in Definition E.2. Define

<!-- formula-not-decoded -->

gives rise to a policy for the two-group contest: Under this policy, agent i ∈ G 1 uses the restriction A ( n ) i = s ( n ) | Ω 1 , while each agent j ∈ G 2 uses the restriction A ( n ) j = s ( n ) | Ω 2 . We have lim n →∞ s ( n ) = s , where s is the threshold function defined as in Equation (3) . Moreover, the sequence of policies A ( n t ) , A ( n t +1) , . . . satisfies the following property:

<!-- formula-not-decoded -->

This theorem establishes how an NE policy for the two-group contest approaches a limit as the number of agents, n , grows indefinitely. It reveals that the sequence of policies { A ( n ) } n not only converges to A but also aligns with an NE policy for the two-group contest. Thus, it validates the assertion made in Theorem 3.1 that A serves as an NE policy for the two-group contest in the limit as n →∞ .

## E.3 Proof of Theorem E.2

We provide an overview of the proof, summarized as follows.

1. In Section E.3.1, we prove Lemma E.1 for the uniqueness of solution t that decides the threshold function s .
2. In Section E.3.2, we bound the winning probabilities P i ( e ; A ( n ) -i ) under policy A ( n ) ; summarized by Lemma E.4. Its proof relies on the winning probability for the undifferentiated contest (Lemma E.5), whose computation is via an auxiliary function defined in Definition E.3.
3. In Section E.3.3, we apply Lemma E.4 to prove that A ( n ) is approximate NE (Lemma E.6).
4. Finally in Section E.3.4, we show that Theorem E.2 is a corollary of Lemma E.6.

For simplicity, we first assume that p a is a point mass at 0, such that policies depend solely on valuations. In this case, s ( v, a ) , P i ( e ; a, A -i ) , π ( v, a, e ; A -i ) are simplified to s ( v ) , P i ( e ; A -i ) , π ( v, e ; A -i ) respectively. At the end, we will show how to extend this to a general p a .

## E.3.1 Proof of Lemma E.1: solution uniqueness

Instead of proving Lemma E.1, we directly prove for the general multi-group case. Let G 1 , . . . , G m be m ≥ 2 groups where each G ℓ has size n ℓ = α ℓ n and valuation distribution p ℓ on the domain Ω ℓ ⊆ R ≥ 0 . We have α ℓ ∈ (0 , 1) for every ℓ ∈ [ m ] and ∑ ℓ ∈ [ m ] α ℓ = 1 . We have the following lemma that generalizes Lemma E.1.

Lemma E.3 ( Unique solution for multiple groups). Suppose ( ∪ ℓ ∈ [ m ] Ω ℓ ) + Ω a is connected and each density p ℓ and p a is positive at any point of its domain. There exists a unique solution t ∈ ∪ ℓ ∈ [ m ] Ω ℓ for the equation ∑ ℓ ∈ [ m ] α ℓ F ℓ ( ζ ) = 1 -c , where for any ζ ∈ R ≥ 0 , F ℓ ( ζ ) = Pr v ∼ p ℓ ,a ∼ p a [ v + a ≤ ζ ] .

Proof: Fix ℓ ∈ [ m ] . Recall that we expand the domain of every CDF F ℓ to R ≥ 0 . We have the following properties for F ℓ :

1. F ℓ ( · ) is non-decreasing across the domain R ≥ 0 , i.e., for any ζ, ζ ′ ∈ R ≥ 0 with ζ &lt; ζ ′ , F ℓ ( ζ ) ≤ F ℓ ( ζ ′ ) holds.

̸

2. F ℓ ( · ) is strictly monotonous across the domain Ω ℓ +Ω a , i.e., for any ζ, ζ ′ ∈ Ω ℓ +Ω a with ζ &lt; ζ ′ and ( ζ, ζ ′ ) ∩ (Ω ℓ +Ω a ) = ∅ , we have F ℓ ( v ) &lt; F ℓ ( v ′ ) .

Define a function g : R ≥ 0 → R ≥ 0 such that for any ζ ∈ R ≥ 0 , g ( ζ ) = ∑ ℓ ∈ [ m ] α ℓ F ℓ ( ζ ) . Since g ( · ) is a convex combination of F ℓ ( · ) 's, we know that g ( · ) is also non-decreasing across the domain R ≥ 0 . Moreover, since ( ⋃ ℓ ∈ [ m ] Ω ℓ ) + Ω a is connected, for any ζ, ζ ′ ∈ ( ⋃ ℓ ∈ [ m ] Ω ℓ ) + Ω a with ζ &lt; ζ ′ , there must exist at least one ℓ ∈ [ m ] such that F ℓ ( ζ ) &lt; F ℓ ( ζ ′ ) and ( ζ, ζ ′ ) ∩ (Ω ℓ +Ω a ) . This implies that g ( · ) is strictly monotonous across the domain ( ⋃ ℓ ∈ [ m ] Ω ℓ ) + Ω a .

Now let L and U denote the infimum and the supremum of domain ( ⋃ ℓ ∈ [ m ] Ω ℓ ) + Ω a respectively. We have 0 = g ( L ) &lt; 1 -c &lt; g ( U ) = 1 . Thus, there must exist a unique point t ∈ ( ⋃ ℓ ∈ [ m ] Ω ℓ )+Ω a such that g ( t ) = 1 -c . This completes the proof. □

## E.3.2 Bounding winning probability

We first have the following lemma that bounds the winning probability under policy A ( n ) .

Lemma E.4 ( Bounding winning probability). For every integer n ≥ n t , we have

<!-- formula-not-decoded -->

For preparation, we define the following function that is useful for computing the winning probability P i ( e ; A -i ) for the undifferentiated contest.

Definition E.3 ( Function for computing winning probability). Given integers n, k ≥ 1 and a density p 1 supported on Ω ⊆ R ≥ 0 , we denote a function Q ( n,k ) p : Ω → R ≥ 0 to be for any v ∈ Ω ,

<!-- formula-not-decoded -->

where B ( n, k, x ) = ( n k ) x k (1 -x ) n -k is the Bernstein polynomial.

By definition, Q ( n,k ) p ( v ) represents the probability that, when sampling n -1 independent and identically distributed (i.i.d.) values v 1 , . . . , v n -1 from distribution p 1 , the value v ranks among the top k values in the set { v 1 , . . . , v n -1 , v } . Given its algebraic significance, the function Q ( n,k ) p ( · ) is monotonically increasing to v across the domain Ω . This means that as v increases, the probability of v being in the top k also increases. Also note that for any integers n, n ′ ≥ 1 with n &lt; n ′ ,

<!-- formula-not-decoded -->

This means that as the number of agents n increases, v is less likely to be in the top k . This function can be used to compute P i ( e ; A -i ) for the undifferentiated contest in the following sense.

Lemma E.5 ( Computation of winning probability for the undifferentiated contest). Let n, k ≥ 1 be integers and p 1 be a density supported on Ω ⊆ R ≥ 0 . Let A = ( A 1 , . . . , A n ) be a symmetric policy for the undifferentiated contest satisfying that every A i is strictly monotonically increasing to v across the domain Ω . Then for every i ∈ [ n ] and v ∈ Ω , we have P i ( A i ( v ); A -i ) = Q ( n,k ) p ( v ) .

Proof: By symmetric, we only need to prove the lemma for i = n , i.e., proving P n ( A n ( v ); A -n ) = Q ( n,k ) p ( v ) . Let v 1 , . . . , v n -1 be i.i.d. samples from p 1 . Since A i is strictly monotonically increasing to v across the domain Ω , we note that the sequence v, v 1 , . . . , v n -1 should have the same order as the sequence A n ( v ) , A 1 ( v 1 ) , . . . , A n -1 ( v n -1 ) . Hence, A n ( v ) is among the top k of { A 1 ( v 1 ) , . . . , A n -1 ( v n -1 ) , A n ( v ) } if and only if v is among the top k of { v 1 , . . . , v n -1 , v } . By the definition of winning probabilities and Definition E.3, this implies that P n ( A n ( v ); A -n ) = Q ( n,k ) p ( v ) . This completes the proof of Lemma E.5. □

Now we are ready to prove Lemma E.4.

Proof [: of Lemma E.4] It suffices to prove for the case that both F 1 ( t ) &gt; 0 and F 2 ( t ) &gt; 0 . Proof for the other two cases is identical. By Definition E.2, we have

<!-- formula-not-decoded -->

Let event E ( n,e ) 1 be that there are at least (1 -F 1 ( t ))(1 -α ) n agents in G 1 \ { i } that put in effort larger than e ; and let E ( n,e ) 2 be that there are at least (1 -F 2 ( t )) · αn agents in G 2 \ { i } that put in effort larger than e . Note that

<!-- formula-not-decoded -->

When e ≥ t , we have

<!-- formula-not-decoded -->

Then to prove P i ( e ; A ( n ) -i ) = 1 , it suffices to show that Pr [ E ( n,e ) 1 ] = Pr [ E ( n,e ) 2 ] = 0 . Note that by policy A ( n ) , the maximum effort put in by an agent is t ≤ v . Hence, no agent can put in effort larger than e , which implies that Pr [ E ( n,e ) 1 ] = Pr [ E ( n,e ) 2 ] = 0 . This completes the proof of P i ( e ; A ( n ) -i ) = 1 when e ≥ t .

When e &lt; t , we note that if both E ( n,e ) 1 and E ( n,e ) 2 happen, there are at least k agents that put in effort t . Since events E ( n,e ) 1 and E ( n,e ) 2 are independent, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then to prove P i ( e ; A ( n ) -i ) ≤ n -α + n -(1 -α ) , it suffices to show that Pr [ E ( n,e ) 1 ] ≥ 1 -n -(1 -α ) and Pr [ E ( n,e ) 2 ] ≥ 1 -n -α .

We first bound Pr [ E ( n,e ) 1 ] . Note that there are at least (1 -α ) n agents in G 1 ∪ { i } . Also, note that an agent j ∈ G 1 \ { i } puts in effort A j ( v j ) &gt; e if and only if their valuation v j ≥ ∆ n holds. Now consider the undifferentiated contest among G 1 ∪ { i } with k 1 = (1 -F 1 ( t ))(1 -α ) n and density p 1 . By Lemma E.5, we have

<!-- formula-not-decoded -->

Let X 1 , . . . , X n be (1 -α ) n -1 i.i.d. random variables where each X i = 0 with probability F 1 ( t ) -√ log n n and otherwise X i = 1 . We note that ∑ (1 -α ) n -1 j =(1 -α ) n -1 -k 1 B ((1 -α ) n -1 , j, F 1 ( t ) -√ log n n ) is equivalent to the probability that ∑ i ∈ [ n -1] X i ≤ k 1 -1 . Also note that

<!-- formula-not-decoded -->

Then by the Chernoff bound, we have

<!-- formula-not-decoded -->

Combining with Inequality (15), we prove that Pr [ E ( n,e ) 1 ] ≥ 1 -n -(1 -α ) . By a similar argument, we can also prove Pr [ E ( n,e ) 2 ] ≥ 1 -n -α . Overall, we prove that P i ( e ; A ( n ) -i ) ≤ n -α + n -(1 -α ) when e &lt; t . This completes the proof of Lemma E.4.

## E.3.3 Proof that A ( n ) is approximate NE

Based on Lemma E.4, we are now ready to prove the approximate degree of A ( n ) to be an NE policy. Lemma E.6 ( A ( n ) is approximate NE). For any n ≥ n t , A ( n ) is an ε n -NE policy, where ε n = ( n -α + n -(1 -α ) )∆ n + t -∆ n .

Proof: Fix ℓ = 1 , 2 , i ∈ G ℓ , and v, e ∈ Ω ℓ . We discuss the value π i ( v, e ; A ( n ) -i ) -π i ( v, A ( n ) i ( v ); A ( n ) -i ) . By Lemma E.4, we know that

<!-- formula-not-decoded -->

Then if v &lt; ∆ n , we have π i ( v, A ( n ) i ( v ); A ( n ) -i ) = π i ( v, 0; A ( n ) -i ) = 0 , which implies that

<!-- formula-not-decoded -->

Otherwise if v ≥ ∆ n , we have π i ( v, A ( n ) i ( v ); A ( n ) -i ) = π i ( v, t ; A ( n ) -i ) = v -t , which implies that

<!-- formula-not-decoded -->

Note that when v ≥ ∆ n and e &lt; t ,

<!-- formula-not-decoded -->

Overall, we conclude that the following inequality always holds:

<!-- formula-not-decoded -->

This verifies that A ( n ) is an ε n -NE policy for the two-group contest.

## E.3.4 Completing the proof of Theorem E.2

Proof [: of Theorem E.2] Assume p a is a point mass at 0. We first prove that lim n →∞ s ( n ) = s . This is a direct corollary of the fact that lim n →∞ ∆ n = t . Consequently, for any v ∈ R ≥ 0 , there exists n v such that for any integer n ≥ n v , s ( n ) ( v ) = s ( v ) holds.

By Lemma E.6, A ( n ) is a ε n -NE policy, where ε n = ( n -α + n -(1 -α ) )∆ n + t -∆ n . Since lim n →∞ ∆ n = t , we have

<!-- formula-not-decoded -->

This completes the proof of Equation (12).

Uniqueness of A . To prove that A is the unique NE, we first recall Corollary 3.2 of [20] that says that a subset of symmetric agents should have the same policy in an NE. Thus, assuming A ′ is an NE policy for the two-group contest as n → R ≥ 0 , all agents i ∈ G 1 use a common threshold policy s 1 , and those in G 2 use s 2 . Suppose the threshold for s ℓ is t ℓ . We next prove that t 1 = t 2 , which implies that s 1 = s 2 . When t 1 &gt; t 2 , if an agent puts in effort t 1 , then it will get selected. Thus, the probability of an agent in G 1 getting selected is 1 -F 1 ( t 1 ) . Hence, if 1 -F 1 ( t 1 ) &lt; c 1 -α , fewer than cn agents in G 1 get selected. Thus, agents in G 1 getting selected will find that putting in effort slightly larger than t 2 instead of t 1 suffices to ensure their effort is larger than all agents in G 2 , and consequently, they will still be selected. Through this reduction in effort, they can gain an additional payoff of t 1 -t 2 , which violates the stability condition. A similar argument holds for G 2 when

□

1 -F 1 ( t 1 ) &gt; c 1 -α . Thus, A ′ is not an NE when t 1 &gt; t 2 . Similarly, we can prove that A ′ is not an NE when t 1 &lt; t 2 . Thus, we must have t 1 = t 2 = t and s 1 = s 2 = s .

If (1 -α ) F 1 ( t ) + αF 2 ( t ) &gt; 1 -c , then fewer than cn agents put in a non-zero effort and get selected. Thus, an agent with valuation t ′ &lt; t , has a willingness to put in an effort ε slightly larger than 0 instead of 0. Through this increase in effort, it can gain an additional payoff of t ′ -ε . Thus, A ′ is not an NE. Similarly, we can prove that A ′ is not an NE if (1 -α ) F 1 ( t ) + αF 2 ( t ) &lt; 1 -c . Thus, for an NE policy, t must be the solution of (1 -α ) F 1 ( v ) + αF 2 ( v ) = 1 -c . By Lemma E.1, the equation (1 -α ) F 1 ( v ) + αF 2 ( v ) = 1 -c has a unique solution. Thus, A ′ = A , which proves the uniqueness.

Extension to general p a . For general p a , the solution t of Equation (3) represents a score that a c -fraction of agents can achieve without making their expected payoff negative ( v + a ≥ t ). The proof is almost identical to that when p a is a point mass at 0, except that the effort an agent with v + a ≥ t is willing to put in should be s ( v, a ) = max { t -a, 0 } instead of t . This is because t represents the score that the agent aims to reach, rather than the effort itself.

Overall, we complete the proof of Theorem E.2.

Remark E.7 ( Extension of Theorem 3.1). Using the same proof technique, Theorem 3.1 can be extended to handle multiple groups and non-identical effort costs. Let G 1 , . . . , G m represent m ≥ 2 groups, where | G ℓ | = α ℓ n and the valuation density for each group is p ℓ over the domain Ω ℓ ⊆ R ≥ 0 . Each α ℓ ∈ (0 , 1) satisfies the condition ∑ ℓ ∈ [ m ] α ℓ = 1 . Recall that p a represents the initial ability density over the domain Ω a ⊆ R ≥ 0 , and we introduce an additional effort cost density p κ over the domain Ω κ ⊆ R &gt; 0 . Each agent i ∈ [ n ] knows its type θ i = ( v i , a i , κ i ) and selects an effort level e i ≥ 0 . The agent's score is given by m ( e i + a i ) , and their expected payoff is P i v i -κ i e i . It is important to note that agents' costs of effort κ i may vary and affect only their expected payoff, not their score.

In this extended multi-group contest, for ℓ ∈ [ m ] , we extend CDF F ℓ to be F ℓ ( ζ ) = Pr v ∼ p ℓ ,a ∼ p a ,κ ∼ p κ [ v κ + a ≤ ζ ] . Now suppose domains ⋃ ℓ ∈ [ m ] Ω ℓ , Ω a , Ω κ are connected and densities p ℓ , p a , p κ are positive at any point of their own domains. Let t be the unique solution to the equation ∑ ℓ ∈ [ m ] α ℓ F ℓ ( ζ ) = 1 -c . The infinite NE policy (3) extends to:

<!-- formula-not-decoded -->

## E.4 Comparing with a distributional two-group contest

Recall that the technical challenges for Theorem 3.1 are mainly caused by the asymmetric strategic environment across groups. To avoid asymmetry, one may consider the following variant of the two-group contest. Note that for simplicity, we also assume that p a is a point mass at 0.

Definition E.4 ( Distributional two-group contest). Let n ≥ k ≥ 1 be integers, α ∈ (0 , 1) , ρ ∈ (0 , 1] , and p ℓ be a density supported on a domain Ω ℓ ⊆ R ≥ 0 for ℓ = 1 , 2 . Let each agent i ∈ [ n ] belong to G 1 with probability 1 -α and belong to G 2 with probability α independently. Let the valuation of each agent in G 1 be drawn i.i.d. from p 1 , and the valuation of each agent in G 2 be drawn i.i.d. from p 2 . Assume that each agent i ∈ G ℓ ( ℓ = 1 , 2 ) knows n 1 , n 2 , k , p 1 , p 2 , the group it belongs to and its valuation, and has to choose a policy A i : Ω ℓ → R ≥ 0 to maximize its expected payoff. The goal of the distributional two-group contest is to compute the NE policy satisfying Equation (1) .

The main difference from the two-group contest is that this distributional variant's group identity is random and the valuation density of each agent is identical, say p = (1 -α ) p 1 + αp 2 . Thus, using a similar argument as in an undifferentiated contest, it is easy to verify that the NE policy of the distributional two-group contest is identical to that of an undifferentiated contest with density p 1 . Consequently, in the infinite n case, the NE policy of the distributional two-group contest is identical to that of the two-group contest, say A in Theorem 3.1. Then one may wonder whether this distributional two-group contest can also be used to simplify the proof of Theorem 3.1, as the infinite contest does. Below, we show that this is not the case and discuss the essential differences between the two models. For simplicity, we call the two-group contest Model I and call the distributional two-group contest Model II .

Model distinction. Firstly, Model I itself is of relevant interest as it captures real-world examples in which group sizes are well understood, while in Model II the group sizes are random variables.

Convergence distinction. Though Model I and Model II share the same NE policy A in the infinite case, we would like to clarify that our main convergence result (Theorem E.2) cannot be inferred simply from knowing that the limit of the NEs of the two models is the same. To put it in simplest terms, consider two sequences a 1 , . . . , a n , . . . and b 1 , . . . , b n , . . . that converge to the same limit point z . The proof of convergence for the first sequence does not necessarily provide any information about the convergence of the second sequence. Therefore, the convergence result for our model cannot be simply inferred from prior work.

Analysis distinction. Moreover, the analysis of Model II relies heavily on symmetric policies for all agents (enabled precisely by the fact that group sizes are random), allowing the use of order statistics of p 1 . In contrast, in Model I, we expect asymmetric policies across groups. For example, consider a two-agent case with k = 1 : Agent 1's valuation follows a uniform distribution on [0 , 0 . 5] , while Agent 2's valuation follows a uniform distribution on [0 . 5 , 1] . In Section B, we show that the NE policy for this example must be asymmetric. Any symmetric policy A would result in a near-zero winning probability for Agent 1, leading to a negative expected payoff and implying that A is not an NE. This negative payoff arises from the asymmetric strategic environment faced by Agents 1 and 2, where the density of the highest valuation among other agents differs for each agent. Consequently, the order of winning probabilities of agents ( P i ) can differ from the order of valuations ( v i ), posing a significant mathematical challenge for determining the NE. E.g., for strategies s 1 and s 2 , let F s ℓ ( v ) denote the cumulative distribution of efforts s ℓ ( v ) when v ∼ p ℓ . The cumulative distribution of the ( k -1) -th effort e ⋆ from an agent in G 1 is then given by:

<!-- formula-not-decoded -->

Compare this to the expression for the symmetric case

<!-- formula-not-decoded -->

Thus, the calculus and approximations for the expression for the two-group contest are significantly more difficult, making it much harder to arrive at the equilibrium policies than for Model II.

## E.5 An alternative proof using an infinite contest

Recall that Theorem 3.1 studies the case of n → ∞ for the two-group contest. To increase the understanding of the infinite case, we propose the following infinite version of the two-group contest, where every real number in the interval [0 , 1 -α ] corresponds to an agent in G 1 and in the interval (1 -α, 1] corresponds to an agent in G 2 . For simplicity, we still assume that p a is a point mass at 0. Formally, we provide the following definition.

Definition E.5 ( Infinite contest). Let k ≥ 1 be integers, α ∈ (0 , 1) , ρ ∈ (0 , 1] , and p ℓ be a density supported on a domain Ω ℓ ⊆ R ≥ 0 for ℓ = 1 , 2 . Let every real number in the interval [0 , 1 -α ] correspond to an agent in group G 1 , and in the interval (1 -α, 1] correspond to an agent in group G 2 . For ℓ ∈ { 1 , 2 } , let the valuation of every agent in G ℓ draw i.i.d. from p ℓ . Assume that each agent i ∈ G ℓ ( ℓ = 1 , 2 ) knows α , k , p 1 , p 2 , the group it belongs to and its valuation, and has to choose a policy A i : Ω ℓ → R ≥ 0 to maximize its expected payoff.

There are countless agents in this infinite contest. Also, note that G 1 contains (1 -α ) -fraction of agents while G 2 contains the remaining α -fraction. Below, we show how to use this infinite contest to hypothesize the NE policy A for the two-group contest defined in Theorem 3.1. It mainly consists of two steps: Showing that two-group contests converge to the infinite contest as n →∞ and computing NE for the infinite contest.

Showing two-group contests converge to the infinite contest as n →∞ . We first show that the infinite contest is the limit of two-group contests. Let g n denote a two-group contest as defined in

the two-group contest with an NE policy A n . Let g denote the infinite game as defined in Problem E.5. We view g n as a collection of n density functions of agents, with the i -th agent represented by the real number i -1 n -1 . Agent i belongs to group G 1 if 1 ≤ i ≤ (1 -α ) n and to group G 2 otherwise. From this viewpoint, we propose the following theorem.

Theorem E.8 ( Two-group contests converge to the infinite contest). g n converges to g in the following sense: For any ε &gt; 0 , there exists a sufficiently large n 0 such that for all n ≥ n 0 ,

- For any t ∈ [0 , 1 -α ] , | ∫ t 0 dx -| { i ∈ G 1 : i -1 n -1 ≤ t } | n | ≤ ε , i.e., the difference in the fraction of agents in G 1 associated with real number at most t between g and g n , is at most ε .
- For any t ∈ (1 -α, 1] , | ∫ 1 t dx -| { i ∈ G 2 : i -1 n -1 ≥ t } | n | ≤ ε , i.e., the difference in the fraction of agents in G 2 associated with real number at least t between g and g n , is at most ε .

Note that the agents in g n can be captured by a uniform distribution µ n over real numbers i -1 n -1 ( i ∈ [ n ] ). The convergence conditions in the theorem state that the limit of µ n is the uniform density µ over [0 , 1] , where µ represents the density of agents in g .

Proof: [of Theorem E.8] Let n 0 = ⌈ ε -1 ⌉ . Then we have n ≥ ε -1 . For any t ∈ [0 , 1 -α ] , we have

<!-- formula-not-decoded -->

and

We conclude that

<!-- formula-not-decoded -->

Similarly, for any t ∈ (1 -α, 1] , we can prove that ∣ ∣ ∣ ∣ ∫ 1 t dx -| { i ∈ G 2 : i -1 n -1 ≥ t } | n ∣ ∣ ∣ ∣ ≤ ε . This completes the proof of Theorem E.8. □

Computing NE for the infinite contest. It follows from Theorem E.8 that the limit of the two-group contests g n is the infinite contest g . Then, assuming the NE policy of g n is A ( n ) , we can infer that the limit of A ( n ) is the NE policy of the infinite contest. Thus, to hypothesize the NE policy for g n as n →∞ , it suffices to compute the NE policy for the infinite contest.

We first observe that the strategic environment for all agents in the infinite contest is the same, i.e., the probability that a given valuation v is among the top c -fraction is the same for all agents. This property reduces the infinite contest to an undifferentiated contest (except for the different domains of valuation densities), leading to a symmetric NE policy. Formally, we provide the following lemma that shows that A is exactly the unique NE policy for the infinite contest.

Lemma E.9 ( The infinite contest). Suppose Ω 1 ∪ Ω 2 is connected and each density p ℓ is positive at any point of domain Ω ℓ . Then policy A defined in Equation (3) is the unique NE policy for the infinite contest.

Proof: We first note that for any agent (whether in G 1 or G 2 ), the probability that a given valuation v is among the top c -fraction is given by:

<!-- formula-not-decoded -->

where F is the CDF of the joint density (1 -α ) p 1 + αp 2 . Recall that t is the unique solution to the equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i.e., t = F -1 (1 -c ) . Then through a straightforward calculation, it follows that p 1 ( v ) = 1 for v &gt; t and p 1 ( v ) = 0 for v &lt; t . Since the winning probability function p 1 is identical for all agents, the strategic environment for all agents in the infinite contest is the same. Recall that by Corollary 3.2 of [20], symmetric agents will use a symmetric policy in an NE. Thus, we can assume an increasing symmetric policy s : Ω 1 ∪ Ω 2 → R ≥ 0 for all agents.

By the equilibrium condition, for any valuation v and effort e ,

<!-- formula-not-decoded -->

By a similar argument as for Equation (10) (undifferentiated contest), we can compute that s ( v ) = t for v &gt; t and s ( v ) = 0 for v &lt; t . This turns out to be the threshold function defined in Equation (3). Thus, the policy A , where each agent restricts s to its valuation the domain, is indeed the unique NE for the infinite contest. This completes the proof of Lemma E.9. □

Using the infinite game to provide an alternative proof of Theorem E.2. As shown in Lemma E.9, instead of relying on observations from the finite case as in Section E.1.2, we can use this infinite contest to hypothesize the NE policy A for the two-group contest in the infinite n case.

Once we have a solid guess for the NE policy A using the infinite contest, we need to show that A remains an NE as n →∞ . While this approach simplifies the initial hypothesis, the challenge remains in proving that A is indeed an NE. Similar to our current proof of Theorem 3.1, we must construct a series of proxies that converge to A and increasingly approximate an NE policy. As detailed in Section E.2, this step remains technically challenging.

Overall, using the infinite contest could provide an alternative proof of Theorem E.2. Moreover, the symmetric strategic environment of this infinite contest can provide deeper insights into why the NE policy remains symmetric, even when valuations are asymmetric across groups.

## F Omitted details for uniform distribution analysis from Sections 3 and 4

Theorem F.1 ( Restatement of Theorem 3.2). Assume p 2 ( v ) = 1 ρ p 1 ( v ρ ) for some ρ ∈ (0 , 1] and p a is a mass point at 0. Let policy A be defined as in Theorem 3.1, characterized by t being the unique solution of Equation (4) . Then for any density p 1 ,

<!-- formula-not-decoded -->

Moreover, r R ( A ) is monotonically increasing w.r.t. ρ , c , and α , while RV ( A,m ) is monotonically increasing w.r.t. ρ and monotonically decreasing w.r.t. c and α , for any merit function m .

Proof: We discuss three metrics separately.

Metric RV ( A,m ) . Recall that A is a threshold function characterized by t . Thus, agents with the sum of valuation and initial ability v + a &gt; t get spots. Since p a is a point mass at 0, we know that the score of each selected agent is exactly t . Thus, the average revenue RV ( A,m ) = m ( t ) .

Next, we prove the monotonicity of RV ( A,m ) . Since m is monotonically increasing, we only need to prove the monotonicity of t with respect to ρ, c, α . Recall that t is the solution of Equation (4), which can be rewritten as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since F 1 ( ζ ρ ) is a monotonically decreasing function of ρ , we know that f ( ζ ) is also a monotonically decreasing function of ρ . Thus, the solution t increases with ρ . Since F a ( ζ -ρv ) ≥ F a ( ζ -v ) , f ( ζ ) is an increasing function of α . Also note that f ( ζ ) is an increasing function of c . Thus, as c or α increase, solution t decreases.

Overall, we prove that RV ( A,m ) is monotonically increasing w.r.t. ρ and monotonically decreasing w.r.t. c and α , for any merit function m .

Metric r R ( A ) . Recall that F ℓ is a cumulative density function (CDF) of the sum of valuation and initial ability such that for any ζ ∈ R ≥ 0 , F ℓ ( ζ ) = Pr v ∼ p ℓ ,a ∼ p a [ v + a ≤ ζ ] . Thus, F ℓ is the CDF of p ℓ when p a is a point mass at 0. Then the linearity of expectation yields:

<!-- formula-not-decoded -->

This translates to:

<!-- formula-not-decoded -->

Since p 2 ( v ) = 1 ρ p 1 ( v ρ ) , we know that F 2 ( t ) = F 1 ( t/ρ ) and F 1 ( t ) ≤ F 2 ( t ) . Thus, we have E [ R 2 ( A )] ≤ E [ R 1 ( A )] .

As n →∞ , it suffices to prove that

<!-- formula-not-decoded -->

We first note that | S ∩ G ℓ | is highly concentrated at E v i ,a i [ | S ∩ G 1 | ] since all agents in G ℓ are i.i.d. Concretely, the following inequality holds for any t &gt; 0 by the Chernoff bound:

<!-- formula-not-decoded -->

Hence, for t = o ( √ 1 n ) , we have Pr[ || S ∩ G ℓ | -E v i ,a i [ | S ∩ G ℓ | ] | ≥ t · E v i ,a i [ | S ∩ G ℓ | ]] → 0 . This implies that as n →∞ ,

<!-- formula-not-decoded -->

which completes the proof of the formula of r R ( A ) .

Next, we prove the monotonicity of r R ( A ) with respect to ρ . Recall that t is monotonically increasing with ρ . We know that 1 -F 1 ( t ) is a monotonically decreasing function of ρ . Since 1 -F 2 ( t ) = 1 -c -(1 -α )(1 -F 1 ( t )) α , we know that 1 -F 2 ( t ) is monotonically increasing with ρ . Thus, r R ( A ) = 1 -F 1 ( t/ρ ) 1 -F 1 ( t ) is an increasing function of ρ .

Metric r S ( A ) . By a similar argument as for r R ( A ) , we first have that as n →∞ ,

<!-- formula-not-decoded -->

Also note that

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

Combining the above all, we obtain that r S ( A ) = ∫ ∞ t ( v -t ) p 2 ( v ) dv ∫ ∞ t ( v -t ) p 1 ( v ) dv . Since p 2 ( v ) = 1 ρ p 1 ( v ρ ) , we have

<!-- formula-not-decoded -->

Next, we analyze the monotonicity of r S ( A ) with respect to ρ . Let g ( x ) = ∫ ∞ x ( v -x ) p 1 ( x ) dx , which is monotonically decreasing of x . We have r S ( A ) = ρ · g ( t/ρ ) g ( t ) . Since t is monotonically increasing with ρ , g ( t ) is also monotonically increasing with ρ . Thus, to prove that r S ( A ) is monotonically increasing with ρ , it suffices to prove that t/ρ is monotonically decreasing with ρ . Recall that we have shown that F 1 ( t/ρ ) = F 2 ( t ) is monotonically decreasing with ρ . This implies that t/ρ is indeed monotonically decreasing with ρ , completing the proof.

Overall, we have completed the proof of the theorem.

□

Figure 13: Plots of t versus ρ for various c with α = 0 . 5 for the uniform distribution.

<!-- image -->

Remark F.2 ( Monotonicity of r R ( A ) and r S ( A ) w.r.t. c, α ). By Theorem 3.2, we note that when fixing ρ , both r R ( A ) and r S ( A ) are functions of t . Since t is monotonically decreasing with respect to c and α , we only need to investigate the monotonicity of r R ( A ) and r S ( A ) with respect to t . Proposition 4.1 demonstrates that r R ( A ) and r S ( A ) are monotonically decreasing with t , and hence, monotonically increasing with c and α . Below, we provide an example with p 1 to show that this monotonicity does not always hold.

Let ε &gt; 0 be a sufficiently small number and ρ = 0 . 5 . Let p 1 be supported on Ω 1 = [0 , 2] such that p 1 ( v ) = 1 -ε for v ∈ [0 , 0 . 5] ∪ [1 . 5 , 2] and p 1 ( v ) = ε for v ∈ (0 . 5 , 1 . 5) . Then F 1 ( v ) = (1 -ε ) v for v ∈ [0 , 0 . 5] , 0 . 5 -ε + εv for v ∈ (0 . 5 , 1 . 5) , and (1 -ε ) v + 2 ε -1 . By Theorem 3.2, we have r R ( A ) = 1 -F 1 ( t/ρ ) 1 -F 1 ( t ) . Then in this case, r R ( A ) = 1 -(1 -ε ) / 2 1 -(1 -ε ) / 4 ≈ 2 3 when t = 0 . 25 ; while r R ( A ) = 1 -(1 -ε ) / 2 -0 . 5 ε 1 -(1 -ε ) / 2 ≈ 1 . Thus, r R ( A ) is not monotonically decreasing with t . A similar computation can be done for r S ( A ) .

By a similar argument as for Theorem 3.2, we provide the following formulas of metrics for general densities. The computation is straightforward and we omit here.

Theorem F.3 ( Metrics in general). Let policy A be defined as in Theorem 3.1, characterized by t being the unique solution of Equation (2) . Then for any densities p 1 , p 2 , and p a ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the following, we complete the analysis from Sections 3 and 4 for the case where p 1 and p 2 are uniform densities. The visualization of t for them can be found in Figure 13. We first give the proof of Equation (5) for the case that p a is uniform.

Proposition F.4 ( Complete version of Equation (5) ). Let α, c ∈ (0 , 1) and ρ ∈ (0 , 1] . Let p 1 be uniform on [0 , 1] , p 2 be uniform on [0 , ρ ] , and p a be uniform on [0 , 1] . Let t ∈ [0 , 2] be the solution to the equation ∫ 1 0 (1 -α ) · min { 1 , ( ζ -v ) + } dv + ∫ ρ 0 α ρ · min { 1 , ( ζ -v ) + } dv = 1 -c . Then if 0 &lt; c ≤ 1 -α 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

if 1 &lt; c ≤ 1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: Let f ( ζ ) = ∫ 1 0 (1 -α ) · min { 1 , ( ζ -v ) + } dv + ∫ ρ 0 α ρ · min { 1 , ( ζ -v ) + } dv . We first note that f ( ζ ) is a monotone increasing function with f (0) = 0 and f (2) = 1 . By analyzing the value of min { 1 , ( ζ -v ) + } , we also have the following equation:

<!-- formula-not-decoded -->

Accordingly, we know that

<!-- formula-not-decoded -->

Thus, f is a piecewise-polynomial function of ζ . Solving f ( ζ ) = 1 -c results in Corollary F.4. □

Proposition F.5 ( Restatement of Proposition 4.1). Let p 1 be uniform on [0 , 1] , p 2 be uniform on [0 , ρ ] , and p a be a point mass at 0. Let A be the NE policy for the two-group contest as n →∞ . Then

<!-- formula-not-decoded -->

Moreover, RV ( A,m ) = m ( t ) for any merit function m ( · ) ; r R ( A ) and r S ( A ) are monotonically increasing functions of parameters ρ, c, α .

Proof: Note that RV ( A,m ) = m ( t ) is directly from Theorem 3.2.

{ } +

Computation of t . Note that F 1 ( v ) = v and F 2 ( v ) = min 1 , v ρ . Let g ( v ) = (1 -α ) v α min { 1 , v ρ } . We note that g ( · ) is a piece-wise linear function of v with an inflection point v = ρ . Plugging v = ρ into the equation, we obtain that ρ = 1 -c 1 -α which is an inflection point of t . Then if solution t &gt; ρ , we have that t is the solution of the equation (1 -α ) v + α = 1 -c , implying that t = 1 -c 1 -α . The condition for this case is ρ &lt; 1 -c 1 -α = t . Otherwise if solution t ≤ ρ , we have that t is the solution of the equation (1 -α ) v + α v ρ = 1 -c , implying that t = ρ (1 -c ) ρ -αρ + α . The condition for this case is ρ ≥ 1 -c 1 -α . This completes the proof for t .

Analysis for r R ( A ) . By Theorem 3.2, we have r R ( A ) = 1 -min { 1 , t ρ } 1 -t . By the form of t , we can verify the explicit form of r R ( A ) .

Note that when ρ ≥ 1 -c 1 -α , we have

<!-- formula-not-decoded -->

Let

Define the auxiliary functions:

<!-- formula-not-decoded -->

Then when ρ ≥ 1 -c 1 -α , we have

<!-- formula-not-decoded -->

The partial derivatives w.r.t. ρ, c, α are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, r R ( A ) is monotonically increasing with ρ, c, α when ρ ≥ 1 -c 1 -α . Moreover, the threshold 1 -c 1 -α is monotonically decreasing with c, α . Thus, we conclude that r R ( A ) is a monotonically increasing function of parameters ρ, c, α .

Analysis for r S ( A ) . By a similar argument as for r R ( A ) , we can obtain the formulas of r S ( A ) using Theorem 3.2 and the form of t . Note that r S ( A ) = ρr R ( A ) 2 . Thus, r S ( A ) is monotonically increasing with ρ, c, α .

Overall, we have completed the proof of the proposition.

## G Additional details to Section 4

In this section, we first illustrate details for how to estimate perceived bias from JEE Advanced 2024 (Section G.1). Then we provide a robustness analysis for key findings in Section 4 by varying α and c (Section G.2). Next, we give an illustrative example for the practical use of our model, including how to make interpretable predictions and policy interventions (Section G.3). Finally, we provide omitted details for alternative interventions in Section 4.

## G.1 Case study - estimating perceived bias from JEE Advanced 2024

To illustrate our framework in a high-stakes meritocratic setting, we calibrate the model using data from JEE Advanced 2024, the entrance examination for admission to the Indian Institutes of Technology (IITs). The gender-disaggregated statistics were published by the Government of India's Press Information Bureau [71]:

| Group   | Candidates Appeared   | Qualified   |
|---------|-----------------------|-------------|
| Male    | 139,180               | 40,284      |
| Female  | 41,020                | 7,964       |
| Total   | 180,200               | 48,248      |

Model calibration. We define the disadvantaged group as female candidates and the advantaged group as male candidates . From the data:

- Proportion of disadvantaged applicants:

<!-- formula-not-decoded -->

□

<!-- formula-not-decoded -->

- Selection rate for the entire applicant pool:

<!-- formula-not-decoded -->

- Admit rate for each group:

<!-- formula-not-decoded -->

- Observed representation ratio:

Simplifying both sides:

Compute both sides:

Bring all terms to one side:

<!-- formula-not-decoded -->

The inferred bias parameter is:

<!-- formula-not-decoded -->

Solving for the bias parameter ρ . Using the closed-form expression for the representation ratio in the uniform-valuations model:

<!-- formula-not-decoded -->

we plug in r R = 0 . 671 , c = 0 . 268 , and α = 0 . 228 to solve for ρ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which reflects a perceived disadvantage for female candidates: they value qualification outcomes at roughly 88.2% of their male counterparts' valuation, consistent with the observed gender gap in qualification rates. This example demonstrates how our model can be applied to quantify bias in selection systems using real-world statistics.

## G.2 Robustness analysis for findings in Section 4

In this section, we assess whether our core conclusions in Section 4 depend on the specific parameter settings. To verify robustness, we conducted additional simulations varying α and c beyond the default values. Below we summarize our findings:

Metric robustness across group sizes. We varied α from 0.5 to 0.3 (to represent smaller disadvantaged groups) and recalculated the representation ratio r R ( A ) and welfare ratio r S ( A ) across a grid of disparity levels ( ρ ) and selection rates ( c ); see Figure 14. The overall trends remain consistent: for example, r R ( A ) ≤ 0 . 2 still holds for c = 0 . 1 and ρ ≤ 0 . 85 , confirming that strategic behavior amplifies underrepresentation in highly selective settings.

Robustness of intervention takeaways. We varied c from 0.228 (derived from the JEE Advanced data) to 0.1 and α from 0.268 to 0.5 and re-evaluated intervention strategies. Figure 15 plots optimal interventions for various threshold τ . The overall trends remain consistent. For instance, in Figure 15(a), when τ ≤ 0 . 87 , increasing access (raising c ) remains more cost-effective. In contrast, when τ &gt; 0 . 87 , improving group valuation (increasing ρ ) becomes more impactful. This confirms that the recommendation to prioritize access vs. valuation interventions depending on the disparity level remains valid across reasonable choices of α and c .

Figure 14: Plots of the representation ratio r R ( A ) and the social welfare ratio r S ( A ) as parameters ρ and c vary for Proposition 4.1, with default settings of ( ρ, c, α ) = (0 . 8 , 0 . 1 , 0 . 3) . A dotted line in these plots indicates the threshold at which r R ( A ) = 0 . 8 or r S ( A ) = 0 . 8 .

<!-- image -->

Figure 15: Plot of optimal interventions (∆ ρ , ∆ c ) for various τ

<!-- image -->

## G.3 An illustrative example: interpreting and applying the model

We provide a concrete example to illustrate how our theoretical framework can be used to diagnose and compare policy interventions.

Interpretable diagnostics. Suppose a policymaker observes persistent underrepresentation of a disadvantaged group (e.g., female students) in a selective admissions process. Given data on the overall selection rate c , group size α , and the observed representation ratio r R ( A ) , the policymaker can use our model to infer the implied valuation gap parameter ρ (as shown in Section G.1). This parameter summarizes how much lower the disadvantaged group perceives the value of success relative to the advantaged group.

Although ρ is not directly observable, its interpretation is transparent: it attributes behavioral disparities (such as lower effort investment) to structural differences in perceived incentives rather than to innate ability. In this way, the model provides a normative reading of observed disparities-as equilibrium responses to valuation asymmetries.

Policy interventions. Once the implied parameters are estimated, the policymaker can consider two classes of interventions:

- Valuation-based interventions: improving the perceived value of success (e.g., through mentorship programs or financial aid), which effectively increases ρ ;
- Access-based interventions: expanding the number of available slots, thereby increasing c .

Our framework allows simulation of the effects of each intervention on representation, welfare, and efficiency, enabling counterfactual comparisons under a fixed behavioral model. For example, when ρ is low, expanding access may yield larger gains in representation, while when ρ is already high, improving valuation can be more cost-effective.

Implementation challenges. Estimating parameters such as ρ empirically is nontrivial and remains an open direction. It would require auxiliary data sources (e.g., surveys, longitudinal effort-outcome data) or structural assumptions about the effort cost function. Nonetheless, once such estimates are available, our framework provides a transparent scaffold for interpreting behavioral disparities and evaluating the relative effectiveness of competing policy interventions.

## G.4 Details for alternative interventions

Below we provide more detailed theoretical analysis for alternative potential interventions discussed in Section 4.

Introducing preference heterogeneity. Recall that the institution applies group-specific merit mappings of the form: for group G ℓ ( ℓ = 1 , 2 ) and for score s , m ℓ ( s ) = x ℓ · s + y ℓ for group-specific parameters x ℓ , y ℓ ≥ 0 . We have the following generalized theorem of Theorem 3.1

Theorem G.1 ( Generalization of Theorem 3.1: Introducing preference heterogeneity). Let α, c ∈ (0 , 1) . For ℓ = 1 , 2 , let p ℓ be a density supported on a domain Ω ℓ ⊆ R ≥ 0 . Let p a be a density supported on a domain Ω a ⊆ R ≥ 0 . For ℓ = 1 , 2 , let m ℓ be a merit function defined above. For ℓ = 1 , 2 , let F ℓ be a cumulative density function (CDF) of the sum of valuation and initial ability such that for any ζ ∈ R ≥ 0 , F ℓ ( ζ ) = Pr v ∼ p ℓ ,a ∼ p a [ x ℓ v + y ℓ + a ≤ ζ ] . Suppose ( x 1 Ω 1 + y 1 ) ∪ ( x 2 Ω 2 + y 2 )) + Ω a is connected and densities p 1 , p 2 , p a are positive at any point of their own domains. Let t be the unique solution to the equation

<!-- formula-not-decoded -->

Then the threshold function of the NE policy defined in Eq. (3) extends to be: for ℓ = 1 , 2 ,

<!-- formula-not-decoded -->

Moreover, the threshold t -y 2 x 2 for G 2 is monotonically decreasing with x 2 , y 2 .

Proof: The proof for s ℓ is almost identical to that of Theorem 3.1. We only need to note that for any agent i ∈ G ℓ ( ℓ = 1 , 2 ) with valuation-ability pair ( v i , a i ) ∈ Ω ℓ × Ω a , if its valuation v i ≥ t -y ℓ x ℓ -a i , then its merit must be

<!-- formula-not-decoded -->

which is within the top c -fraction and makes the agent get selected.

Regarding the monotonicity of t -y 2 x 2 , note that as x 2 or y 2 increases, F ℓ ( ζ ) decreases. Then the solution t must increase, resulting in a higher threshold t -y 1 x 1 for group G 1 . This reduces the fraction of agents in G 1 to get selected, and consequently, increases the fraction of agents in G 2 to get selected. Then the threshold t -y 2 x 2 must decrease, which completes the proof. □

Note that when x 1 = x 2 = 1 and y 1 = y 2 = 0 , this theorem is exactly Theorem 3.1, and hence, is a generalization. This theorem implies that by increasing x 2 , y 2 , more agents in G 2 are willing to put in efforts due to lower valuation threshold t -y 2 x 2 . This supports the properties discussed in Section 4.

Setting group-specific selection rates. Assume that the institution selects a c -fraction of agents from G 1 and G 2 independently. The model decomposes into two independent within-group contests, each with its own Nash equilibrium.

For the disadvantaged group G 2 , let F 2 denote the CDF of its combined signal v + a . The equilibrium threshold t 2 under group-specific capacity c is the unique solution to:

<!-- formula-not-decoded -->

In contrast, under a uniform selection rate c applied to the full population (original two-group contest), the common threshold t solves:

<!-- formula-not-decoded -->

Since G 2 has lower valuations by assumption, we typically have F 2 ( ζ ) ≥ F 1 ( ζ ) for all ζ , which implies t 2 &lt; t . That is, the disadvantaged group faces a lower selection bar under group-specific quotas.

As a result, agents in G 2 exert more effort on average under per-group capacities compared to the uniformc case. This is because effort is increasing in the probability of selection, which improves when the threshold is lowered.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Contributions are shown in Sections 2, 3, and 4.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Section 5.

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

Justification: See Section E and F.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in Section or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: See Algorithm 1 in Section C for the implementation of dynamics.

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

Justification: See supplemental material.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/pu blic/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in Section, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: All submissions are anonymous.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Section 4 demonstrates how our results can be used to mitigate the impact of bias. Section B discusses the detrimental effects caused by biased valuations.

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
## On the sample complexity of semi-supervised multi-objective learning

Tobias Wegel 1

Geelon So 2

Junhyung Park 1

Fanny Yang 1

1 Department of Computer Science, ETH Zurich 2 Department of Computer Science and Engineering, UC San Diego

## Abstract

In multi-objective learning (MOL), several possibly competing prediction tasks must be solved jointly by a single model. Achieving good trade-offs may require a model class G with larger capacity than what is necessary for solving the individual tasks. This, in turn, increases the statistical cost, as reflected in known MOL bounds that depend on the complexity of G . We show that this cost is unavoidable for some losses, even in an idealized semi-supervised setting, where the learner has access to the Bayes-optimal solutions for the individual tasks as well as the marginal distributions over the covariates. On the other hand, for objectives defined with Bregman losses, we prove that the complexity of G may come into play only in terms of unlabeled data. Concretely, we establish sample complexity upper bounds, showing precisely when and how unlabeled data can significantly alleviate the need for labeled data. This is achieved by a simple pseudo-labeling algorithm.

## 1 Introduction

The multi-objective learning (MOL) paradigm has recently emerged to extend the classical problem of risk minimization from statistical learning to settings with multiple notions of risk [32, 19, 59, 27]. Multi-objective learning problems are ubiquitous in practice, as it often matters how our models behave with respect to multiple metrics and across different populations. For example, consider designing a policy for a self-driving car: the risks could measure different notions of safety (e.g., safety of passengers or pedestrians), or safety under various conditions (e.g., different locations).

More formally, we study the MOL setting with K population risk functionals R 1 , . . . , R K , each quantifying an average, possibly different, loss ℓ k incurred by a prediction model g over the data distribution P k . The aim is to learn models from a class G that minimize all K excess risks E k ( g ) := R k ( g ) -inf R k jointly, using only finite-sample access to the distributions. Here, inf R k is the Bayes risk of the k th task, which is the smallest achievable risk over all measurable functions. Specifically, we study the sample complexity of learning the set of Pareto optimal models in G . Recall that a model is Pareto optimal in G if any alternative model in G that reduces one risk necessarily increases another (see Definition 1); we often simply say that such a model makes an optimal trade-off. Under mild conditions, the set of Pareto optimal models can be recovered by minimizing a family of scalarized objectives T s that we call the s -trade-offs : 1

<!-- formula-not-decoded -->

where the map s : R K → R is from some family S of scalarization functions that aggregates the excess risks into a single statistic. Notice that if the excess risks were known, the problem in Eq. (1) would reduce to a family of classical (multi-objective) optimization problems [44, 54, 38]. However, because the objectives in Eq. (1) depend on distributional quantities that are unknown, we need to

learn the solutions from data. Specifically, we study the sample complexity of achieving Eq. (1) up to errors ε s &gt; 0 , a problem we call S -multi-objective learning , S -MOL for short (see Definition 2).

Two main lines of work have studied the sample complexity of MOL, both predominantly in the supervised framework. In the multi-distribution learning (MDL) literature [27, 4, 51, 73], the goal of the learner is to recover a solution to Eq. (1) for one specific s -trade-off induced by the scalarization s ( v ) = max k ∈ [ K ] v k . This yields the familiar min-max formulation of MOL. 2 And in the literature for the general S -MOL setting [19, 59], the learner aims to solve Eq. (1) for multiple scalarizations. Both lines establish sample complexity bounds in terms of capacity measures of G , which can be shown to be tight in the worst case. However, solutions with good trade-offs may only be found in a complex model class G even when individual tasks are easy to solve in smaller classes H k . In such cases, previous worst-case results do not address whether it is really necessary to pay the full, supervised statistical cost of S -MOL over G . This motivates our consideration of semi-supervised multi-objective learning, in which the learner has access to both labeled and (cheaper) unlabeled data for each of the K tasks. In the single-objective setting, it is well-known that access to unlabeled data can, at times, significantly reduce the amount of labeled data required [17, 70]. But for multiobjective learning, the sample complexity in a semi-supervised setting is largely unexplored, with only a few exceptions [5, 65] that rely on additional assumptions for unlabeled data to be helpful (see Appendix A for a discussion of related works). Thus, the question we aim to address in this paper is

Given that each task k ∈ [ K ] is solvable in a hypothesis class H k , how much labeled and unlabeled data is needed to achieve trade-offs available in a larger function class G ?

In this paper, we give a holistic characterization of the conditions when unlabeled data can help and by how much. In terms of sample complexity upper bounds, we show that for a large class of losses, the capacity of G comes into play only in the amount of unlabeled data required, while the amount of labeled data merely depends on that of H 1 , . . . , H K . Moreover, we show that these rates are achieved by a simple, pseudo-labeling-based algorithm. Concretely, our contributions are as follows:

- We first show hardness of S -MOL under uninformative losses via a minimax sample complexity lower bound that holds even when the learner knows the Bayes-optimal models for each task and has access to the marginal distributions over unlabeled data, i.e., infinite unlabeled data (Section 3.1).
- We then prove that risks induced by Bregman divergence losses -which include the square and cross-entropy losses-effectively disentangle the multi-objective learning problem. For Bregman losses, information about individual risk minimizers can significantly reduce labeled sample complexity in the semi-supervised setting via a simple pseudo-labeling algorithm (Section 3.2).
- Specifically, for S -MOL with Bregman losses, we first provide a uniform bound over the excess s -trade-offs of the pseudo-labeling algorithm for bounded, Lipschitz losses via uniform convergence (Section 4.1). Our major technical contribution then lies in proving localized rates that are distribution-specific under stronger assumptions (Section 4.2). Crucially, the labeled sample complexity in both bounds only depends on the classes {H k } K k =1 , while G only appears in the unlabeled sample complexity.

Our analysis reveals an interesting insight: even though the pseudo-labeling algorithm is reminiscent of single-objective semi-supervised learning procedures, the reason behind the benefits of unlabeled data turns out to be fundamentally different. In single-objective learning, unlabeled data can only help if, roughly speaking, the marginal carries information about the labels [13, 26, 76]. Our results, in contrast, hold without any such assumptions. In multi-objective learning, unlabeled data helps the learner determine the relative importance of each test instance to each task: if the likelihood of an input is higher under one task than another, a model can accordingly prioritize the more relevant risk to achieve better trade-offs. This may be completely independent of the labels assigned by each task.

1 We scalarize the excess risks E k because they each capture the suboptimality with respect to what is theoretically attainable when optimizing R k without consideration for other risks. Notice however that the Pareto set of the excess risks E k is the same as the Pareto set of the risks R k , as they are equal up to the constants inf R k . We further motivate this in Section 2.2 and Fig. 1a.

2 Contrary to Eq. (1), in the MDL literature, the risks are usually directly scalarized.

## 2 Semi-supervised multi-objective learning

In this section, we formally introduce the semi-supervised S -multi-objective learning problem. For ease of reference, an overview of notation is provided in Table 2 of Appendix F.

## 2.1 Preliminaries and the individual tasks

Let X be the feature space , and Y ⊆ R q the label space . We are interested in K prediction tasks, indexed by k ∈ [ K ] := { 1 , . . . , K } , over joint distributions P k of ( X k , Y k ) on the product space X × Y . We denote the underlying joint probability measure by P . From each task, we observe n k i.i.d. labeled samples { ( X k i , Y k i ) } n k i =1 from P k , and N k i.i.d. unlabeled samples { ˜ X k i } N k i =1 from the marginal of P k on X , denoted P k X . Let D denote the combined dataset of both labeled and unlabeled data. For each task k ∈ [ K ] , we define the population and empirical risks of a function f : X → Y as

<!-- formula-not-decoded -->

where ℓ k : Y × Y → R is a (not necessarily symmetric) loss function with ℓ k ( y, ̂ y ) being the loss incurred by predicting ̂ y when the true label is y . Further, we write F all for the set of all functions f : X → Y for which all integrals in this paper are well-defined.For each k ∈ [ K ] , we assume access to a function class H k ⊆ F all that contains a population risk minimizer of R k , that is,

<!-- formula-not-decoded -->

The risk that any model f : X → Y incurs is at least R k ( f ⋆ k ) , so we focus our attention on achieving small excess risk with respect to the Bayes optimal predictor, defined as E k ( f ) := R k ( f ) -R k ( f ⋆ k ) .

## 2.2 Pareto optimality and scalarization

In multi-objective learning, our aim is to learn models g from some function class G that, ideally, achieve low excess risk on all tasks simultaneously . Since, by assumption, the individual tasks are optimally solved in H k , we only consider hypothesis classes G ⊂ F all that satisfy G ⊇ ⋃ k ∈ [ K ] H k . But even if G is very large, minimizing all excess risks may not be possible. In particular, in this work we do not assume that there exists one f : X → Y that performs well across objectives (as opposed to the collaborative learning framework [14] or the setting in [5]). Instead, the aim is to recover the set of Pareto optimal solutions in the class G for the K objectives, formally defined as follows.

Definition 1 (Pareto optimality) . Let E 1 , . . . , E K be a collection of excess risk functionals. We say that a function g ∈ G is Pareto optimal in G if there is no other g ′ ∈ G such that

<!-- formula-not-decoded -->

The subset of G containing all Pareto optimal functions is called the Pareto set . The subset of R K containing the excess risk vectors of the Pareto set is called the Pareto front , defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In words, any model in G that reduces one risk over a Pareto optimal model must increase another risk. Every Pareto-optimal model corresponds to a distinct 'preference' or 'trade-off', all of which are equally valid from a decision-theoretic perspective [29]. We can quantify such trade-offs using scalarization functions s : R K → R that, for all f ∈ F all , map the excess risks into a scalar objective see also Eq. (1). We call T s ( f ) the s -trade-off achieved by f . It has a natural interpretation: recall that E k ( g s ) is the cost incurred by the k -th task to make this particular type of trade-off over myopically optimizing E k . Then, T s ( g s ) aggregates these costs (see Fig. 1). This interpretation also further motivates scalarizing the excess risks instead of the risks: if one task were to have much higher Bayes risk than another, scalarizing the risks would not aggregate the additional cost, cf. Fig. 1a and [1].

<!-- image -->

R

- (a) The attainable risks in each function class and one Tchebycheff scalarization of the risks and of the excess risks.

E

1

1

F

(

)

λ

2

F

F

(

H

(

1

2

)

F

(b) The larger G , the smaller the minimal s -trade-off within G due to low 'bias'.

E

1

F

(

)

G

F

(c) The larger G , the larger the excess s -trade-off on the function class G due to high 'variance'.

Figure 1: (a) All attainable risks for the function classes F all , G , H 1 , H 2 . Using scalarizations on the risks directly can be misleading if the Bayes risk of one task is much larger than that of another: even if both tasks have equal weight, the Tchebycheff scalarization (5) may inadvertently only solve one task (triangles); cf. [1]. Scalarizing the excess risks avoids this (dots). (b) The Pareto fronts for the classes H 1 , H 2 , G , F all in the space of excess risks, and two Tchebycheff scalarizations s max λ (5) with different λ (gray dashed line), with the corresponding trade-off minimizers (dots), and the gap to the minimizers in F all (bi-directed red arrows). (c) The population and empirical Pareto fronts F ( G ) and ̂ F ( G ) , and the excess s -trade-off of the estimated Pareto points (bi-directed red arrows) on the same two Tchebycheff scalarizations.

Two popular examples of scalarization families are Tchebycheff and linear scalarizations, defined as

<!-- formula-not-decoded -->

where ∆ K -1 is the ( K -1) -probability simplex. They represent the worst-case and averaged notions of excess risks, respectively (see Fig. 1a for a visualization of the Tchebycheff scalarization). Minimizing these families of scalarizations recovers the Pareto set under some conditions (e.g., convexity for linear scalarization), see the detailed discussions in [47, 23, 44]. But of course, other scalarizations also exist [23]. Our most general result (Section 4.1) holds for monotonic scalarizations that satisfy the reverse triangle inequality and positive homogeneity, defined as

<!-- formula-not-decoded -->

Both the linear and Tchebycheff scalarizations from Eq. (5) satisfy the properties in Eq. (6).

## 2.3 Multi-objective learning

Because inf g ∈G T s ( g ) may be arbitrarily large, we evaluate our empirical estimates of g s using the excess s -trade-off , defined, for f ∈ F all , as T s ( f ) -inf g ∈G T s ( g ) . The S -MOL problem is then to achieve small excess s -trade-off across scalarizations with high probability.

Definition 2 ( S -MOL) . Let S be a family of scalarizations s : R K → R , ( ε s ) s ∈S a family of positive real numbers, and δ ∈ (0 , 1) . Let A be an algorithm that, provided with a dataset D and the function classes {H k } K k =1 and G , returns a family of functions { ̂ g s : s ∈ S} ⊂ G . Then A solves the S -multi-objective learning ( S -MOL) problem with parameters (( ε s ) s ∈S , δ ) , if where the probability is taken with respect to draws of the training dataset D .

<!-- formula-not-decoded -->

From the population-level optimization perspective in Eq. (1), better trade-offs become possible as the class G grows. This is visualized in Fig. 1b, showing the Pareto fronts achieved by the function classes F all , G , H 1 , H 2 in a two-objective setting. The separation between the Pareto front F ( G ) and the theoretical optimum F ( F all ) can be seen as the 'multi-objective bias' incurred in S -MOL

)

H

)

all

E

2

F

λ

(

G

)

E

2

F

λ

1

(

all

G

λ

2

F

̂

(

)

due to a conservative choice of G . For two Tchebycheff scalarizations, the red bi-directed arrows in Fig. 1b reflect this point-wise 'bias'. However, because the Pareto front needs to be learned from finite samples, we would also expect from classical learning theory that as G grows, so does the 'multi-objective variance' of an empirical Pareto front ̂ F ( G ) . Fig. 1c illustrates this by the gap between F ( G ) and ̂ F ( G ) , and for the same two Tchebycheff scalarizations, the red bi-directed arrows reflect the excess s -trade-off. In the next section we first address how much excess trade-off any algorithm necessarily incurs when learning F ( G ) from data.

## 3 Motivating Bregman losses: A hardness result

To answer this in the context of a semi-supervised setting, we now argue that for the unlabeled data to help solve S -MOL, the structure of the loss functions is key.

## 3.1 A sample complexity lower bound for ideal semi-supervised S -MOL

Let us consider the class of PAC-learners for S -MOL, which are learners that achieve S -MOL for all distributions over X × Y . For concreteness, consider multi-objective binary classification with zero-one loss, where S is the entire family of linear scalarizations S lin :

̸

Definition 3 (Binary classification) . Let G be a hypothesis class with VC dimension d G ∈ N on a data domain X × Y where Y = { 0 , 1 } . For each task k ∈ [ K ] , define ℓ k ( y, y ) = 1 { y = y } .

̂ ̂ For supervised S -MOL with ε s ≡ ε for all s ∈ S lin , prior works achieve a sample complexity upper bound of O ( Kd G /ε 2 ) , up to logarithmic terms, see [19, 59] and Corollary A.1. In fact, a matching lower bound of Ω( Kd G /ε 2 ) holds as well: after all, the set of s -trade-offs {T s : s ∈ S lin } contains the individual excess risk functionals E k , and hence solving S -MOL requires the learner to solve the K original tasks as well. The lower bound then follows from standard agnostic PAC-learning [57, Theorem 6.8]. In short, previous upper bounds are tight and the sample complexity of supervised S -MOL is Θ( Kd G /ε 2 ) , which also coincides with the sample complexity of MDL under non-adaptive sampling [73]. In the semi-supervised S -MOL setting, the question now becomes: can the unlabeled data reduce the label complexity of this problem? Perhaps surprisingly, we now show that the same lower bound holds, even if the learner has additional access to Bayes optimal classifiers f ⋆ k and the marginal distributions P k X .

Proposition 1 (Hardness of semi-supervised multi-objective binary classification) . Fix any K &gt; 1 and any ε ∈ (0 , 1 / 12) . For a given tuple ( P 1 , . . . , P K ) , denote by S k a labeled dataset consisting of i.i.d. samples from P k , let f ⋆ k be a Bayes optimal classifier of P k , and let P k X be the marginal distribution on X . Denote by A any algorithm that, given { S k , f ⋆ k , P k X } K k =1 , returns a set of classifiers { ̂ g s ∈ G : s ∈ S lin } . If A achieves ( S -MOL) with ε s ≡ ε for all linear scalarizations s ∈ S lin , δ ≤ 1 / 6 , and for all distributions ( P 1 , . . . , P K ) in the multi-objective binary classification setting (Definition 3), then the total number of labeled samples it requires is at least | S 1 | + · · · + | S K | ≥ CKd G /ε 2 where C &gt; 0 is a universal constant.

See Appendix D.1 for the proof. Proposition 1 shows that the label sample complexity lower bounds for supervised S -MOL cannot be improved for the problem in Definition 3-even in an idealized semi-supervised S -MOL setting where the learner has infinite unlabeled data and can perfectly solve the individual learning tasks. This effect is due to the zero-one loss not being a proper scoring rule , which is necessary for weighing the risks of two different tasks against each other. And indeed, other losses, such as the hinge or absolute deviation loss, suffer from the same problem. See also [62] for a discussion of calibration in multi-objective learning. In our main results, we show that this lower bound can be circumvented in learning settings where the loss functions are proper in this sense.

## 3.2 Bregman divergence losses and a pseudo-labeling algorithm

In this section, we introduce Bregman losses and their key property that allows us to leverage unlabeled data and alleviate labeled sample complexity via a pseudo-labeling algorithm.

Definition 4 (Bregman loss) . Let Y be convex. A loss ℓ : Y × Y → [0 , ∞ ] is called a Bregman loss if there is a strictly convex and differentiable potential ϕ : Y → R such that ℓ ( y, ̂ y ) = ϕ ( y ) -ϕ ( ̂ y ) -⟨∇ ϕ ( ̂ y ) , y -̂ y ⟩ .

Many standard prediction losses are Bregman losses. For example, the squared loss can be obtained by setting ϕ ( y ) = ∥ y ∥ 2 2 , the logistic loss by choosing ϕ ( y ) = y log y +(1 -y ) log(1 -y ) , and the Kullback-Leibler divergence using ϕ ( y ) = ∑ q k =1 y j log y j . As we now show, an important fact that we will leverage about learning with a Bregman loss is that the associated excess risk functional can be expressed in terms of its minimizer. To state it precisely, we introduce the notions of population and empirical risk discrepancies of a function f ∈ F all from some h ∈ H k , defined as:

We further define for some h 1 ∈ H 1 , . . . , h K ∈ H K and h = ( h 1 , . . . , h K ) the population and empirical scalarized risk discrepancies of a function f ∈ F all from h as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We are now ready to state Lemma 1, proved in Appendix D.2.

Lemma 1 (Properties of Bregman losses, based on [7]) . For each k ∈ [ K ] , let ℓ k be a Bregman loss with potential ϕ k . If both E [ Y k ] and E [ ϕ k ( Y k )] are finite, then up to almost sure equivalence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

{

| Algorithm 1 Pseudo-labeling (PL-MOL)   | Algorithm 1 Pseudo-labeling (PL-MOL)        |
|----------------------------------------|---------------------------------------------|
| 1:                                     | for k ∈ [ K ] do                            |
| 2:                                     | Compute ̂ h k = argmin h ∈H k ̂ R k ( h )   |
| 3:                                     | end for                                     |
| 4:                                     | for s ∈ S do                                |
| 5:                                     | Compute g s = argmin g ∈G ̂ d s ( g ; ̂ h ) |
| 6:                                     | end for                                     |
| 7:                                     | ̂ Return g s : s .                          |

∈ S}

Along with Eqs. (4) and (8), Lemma 1 implies T s ( f ) = d s ( f ; f ⋆ ) for f ⋆ = ( f ⋆ 1 , . . . , f ⋆ K ) . Note that the second part of Lemma 1 decomposes the risk into a task-specific intrinsic noise and a discrepancy term; R k ( f ) = R k ( f ⋆ k ) + d k ( f ; f ⋆ k ) . It turns out that Bregman divergences are, up to transformation of the label space, the only loss functions that enjoy such a decomposition (see [28] and Appendix B). This decomposition helps justify the following pseudo-labeling multi-objective learning algorithm (Algorithm 1).

Notice that the second step (Line 5) is equivalent to first pseudo-labeling all unlabeled data using the ERMs, and then passing it to the supervised S -MOL algorithm from [59] ('ERM-MOL', Algorithm 2 discussed in Appendix A.1). Finally, note that from a computational perspective, even if S is not finite, Algorithm 1 can be implemented, e.g., using hypernetworks, see Appendix A.2.

̂ First, we minimize the individual empirical risks ̂ R k over H k to obtain ̂ h = ( ̂ h 1 , . . . , ̂ h K ) , the set of empirical risk minimizers; thus, we estimate the task-wise Bayes-optimal models f ⋆ k (Line 2). Given Lemma 1, we can then approximate the excess risks E k via the empirical risk discrepancies ̂ d k ( · ; ̂ h k ) using unlabeled data. And so, the s -trade-off T s ( · ) can accordingly be approximated by ̂ d s ( · ; ̂ h ) . The empirical estimate of the Pareto set in G is then given as the minimizer of ̂ d s ( · ; ̂ h ) in G (Line 5). Note that reusing the covariates of the labeled data in this second step would yield at most a constant gain.

## 3.3 Characterization of models with optimal trade-offs: A variational inequality

The pseudo-labeling method illustrates how one can estimate the s -trade-off solutions g s ∈ G from both labeled and unlabeled data when the losses are Bregman divergences. We now show that Bregman losses also enable characterizing the minimizers g s via a variational inequality in some cases. From this inequality, in turn, we can derive conditions for g s to have a particularly simple representation which sheds some light on why unlabeled data can help. Specifically, under linear scalarizations and convexity, we can show the following result.

Lemma 2 (Variational characterization of minimizers) . For each k ∈ [ K ] , let ℓ k be a Bregman loss with potential ϕ k . Suppose s = s lin λ is linear with weights λ , and µ s := ∑ K k =1 λ k P k X . Denote by ⟨· , ·⟩ s the inner product defined as ⟨ f, f ′ ⟩ s = ∫ ⟨ f ( x ) , f ′ ( x ) ⟩ dµ s ( x ) . Then for every non-empty, convex and closed set G ⊆ F all and convex g ↦→T s ( g ) , where ∇ 2 ϕ k ( g ) denotes the function x ↦→∇ 2 ϕ k ( g ( x )) . If G is bounded, such a g ∈ G exists.

<!-- formula-not-decoded -->

Lemma 2 is a direct consequence of Lemma D.6 and Theorem 46 in [69]. From this lemma, we can derive the s -trade-off solutions g s analytically in the special case where G = F all and all potentials are shared ϕ k = ϕ . In that case, since the set of feasible models is unconstrained, the variational inequality in Lemma 2 holds with equality. In particular, the first argument of ⟨· , ·⟩ s must vanish, up to µ s -equivalence. Thus, we can deduce that the s -trade-off solution is µ s -a.s. of the form

<!-- formula-not-decoded -->

so that x ↦→ w k ( x ) is non-negative and ∑ k ∈ [ K ] w k ( x ) ≡ 1 . In short, the optimal prediction with respect to the s -trade-off on the instance x ∈ X is a convex combination of the individual Bayes optimal labels, cf. [40]. Additionally, if the marginals are shared P k X = P X , then each dP k X /dµ s = 1 , so the weights w k are independent of x . However, these are specific settings, and g s does not generally need to take this form . We will later make use of this specific form in Section 4.2.

## 4 Sample complexity upper bounds for pseudo-labeling

We now present uniform and localized upper bounds for Algorithm 1 for Bregman losses in terms of Rademacher complexities . Specifically, we use the coordinate-wise Rademacher complexity of a Y -valued function class H ⊆ F all under distribution P k X with n samples, which is defined as

We discuss the choice and properties of this Rademacher complexity in Appendix E.2.

<!-- formula-not-decoded -->

## 4.1 A uniform learning bound

We start with some assumptions on the loss functions ℓ k that we require for our bounds.

Assumption 1 (Regularity of the losses) . For each k ∈ [ K ] , let ℓ k be a Bregman loss satisfying: 3

- The loss is L k -Lipschitz continuous in both arguments with ℓ 2 -norm in R q , that is, for all y, y ′ , y ′′ ∈ Y it holds that | ℓ ( y, y ′ ) -ℓ ( y, y ′′ ) | ≤ L k ∥ y ′ -y ′′ ∥ 2 and | ℓ ( y ′ , y ) -ℓ ( y ′′ , y ) | ≤ L k ∥ y ′ -y ′′ ∥ 2 .
- Its associated potential function ϕ k is µ k -strongly convex in Y with respect to ℓ 2 -norm, so that for all y, y ′ ∈ Y , it holds that ℓ k ( y, y ′ ) = ϕ k ( y ) -ϕ k ( y ′ ) -⟨∇ ϕ k ( y ′ ) , y -y ′ ⟩ ≥ µ k 2 ∥ y -y ′ ∥ 2 2 .
- The loss is bounded by some constant B k &lt; ∞ as ℓ k ≤ B k .

The boundedness enables the concentration bounds used in our results and is a common assumption, and the strong convexity and Lipschitz continuity enable using a vector contraction inequality from [41], as well as establishing a uniform approximation of the excess risks. Most Bregman losses satisfy Assumption 1 on bounded domains Y , while some (like the logistic loss) require careful treatment of Lipschitz continuity if the gradient is unbounded at the boundary of Y (cf. Corollary C.1 and Lemma E.1). We now state our first main result.

Theorem 1. Suppose that Assumption 1 holds. Let S be any class of monotone scalarizations that satisfy the reverse triangle inequality and positive homogeneity in Eq. (6) , and let { ̂ g s : s ∈ S} be the class of solutions returned by Algorithm 1. Then ( S -MOL) holds for any δ ∈ (0 , 1) and ε s = s ( ε 1 , . . . , ε K ) , where each ε k is bounded by with C k = max { 4 L k , √ 2 B k , L k √ 24 L k /µ k , L k √ 6 B k /µ k } .

<!-- formula-not-decoded -->

3 The norm of the strong convexity and Lipschitz continuity in the first argument can be replaced by an arbitrary norm in Theorem 1. Replacing the norm of the Lipschitz continuity in the second argument in Theorem 1 entails using other vector Rademacher complexities, cf. Appendix E.2. They cannot be replaced in Theorem 2. Moreover, for our results it is sufficient for the Lipschitz property to hold on the range of G .

Theorem 1 is proved in Appendix D.3. Using VC bounds on the Rademacher complexity (see Lemma E.6), Theorem 1 implies that for VC (subgraph) classes G and H k = H with VC dimensions d G , d H , only ˜ O ( Kd H /ε 4 ) labeled and ˜ O ( Kd G /ε 2 ) unlabeled samples are necessary to achieve ε -excess s -trade-off uniformly for all scalarizations. Comparing this with the sample complexity Θ( Kd G /ε 2 ) from Proposition 1, it is apparent that for Bregman losses, Algorithm 1 can alleviate the label complexity of S -MOL significantly when d G ≫ d H , and completely eradicates its dependence on G . It shows that a large complexity of G can be compensated by a large amount of unlabeled data N k , as long as R k N k ( G ) → 0 for N k →∞ .

Also notice that, under Assumption 1, the map g ↦→E k ( g ) or its domain G can be non-convex, in which case non-linear scalarizations are necessary to reach the entire Pareto front. Theorem 1 applies to many such scalarizations, and in particular, the Tchebycheff scalarizations from Eq. (5).

## 4.2 A localized learning bound

The analysis in Theorem 1 is crude: it estimates the excess risks on all of H k and G , which is why the global Rademacher complexities appear in the bound and the unusual extra square-root appears. Such an analysis can be overly conservative, and a localized bound can provide much tighter statistical guarantees [9, 36]. To facilitate a localized analysis for Algorithm 1, we require some additional assumptions. First of all, we only consider linear scalarizations, that is, S ⊆ S lin from Eq. (5), mostly for the following norms to be Hilbert norms: for all k ∈ [ K ] , s ∈ S lin , and f ∈ F all , define

We also require the following shape, strong convexity and smoothness assumptions.

<!-- formula-not-decoded -->

Assumption 2 (Shape, strong convexity and smoothness) . Recall that f ⋆ k ∈ arg min f ∈F all R k ( f ) .

- For γ &gt; 0 and all s ∈ S , h ∈ H 1 ×··· × H K , the map g ↦→ d s ( g ; h ) -γ ∥ g ∥ 2 s is convex on G .
- For all k ∈ [ K ] , the function classes H k -f ⋆ k are star-shaped around the origin; for all α ∈ [0 , 1] , if h ∈ H k -f ⋆ k , then αh ∈ H k -f ⋆ k . Moreover, the function class G is convex and closed.
- For some ν ∈ (0 , ∞ ) , the second and third derivatives of the potentials ϕ k are bounded on Y as sup y ∈Y ∥ ∇ 2 ϕ k ( y ) ∥ 2 ≤ ν and sup y ∈Y ∥ ∇ 3 ϕ k ( y ) ∥ 2 ≤ ν in the ℓ 2 -operator norms.

∥ ∥ ∥ ∥ The shape constraints are commonly used in local Rademacher complexity proofs [9], and the strong convexity acts as a 'multi-objective Bernstein condition' [9, 35]. Moreover, the convexity of G , strong convexity, and smoothness also enable a variational argument that is integral to the bound based on Lemma 2. To operationalize the smoothness, we assume a well-specified setting:

Assumption 3. The minimizer of f ↦→ d s ( f ; f ⋆ ) over F all is contained in G for all s ∈ S .

Assumption 4 (Norm equivalence) . All covariate distributions P k X are absolutely continuous with respect to the mixture distributions ∑ K k =1 λ k P k X for all s lin λ ∈ S , and there is a constant η so that

Finally, for a refined version of our result, we also require the following norm-equivalence assumption that allows relating errors in ∥·∥ s -norm to errors in ∥·∥ k -norm.

<!-- formula-not-decoded -->

We are now ready to state the localized bound. Recall that f ⋆ k is the Bayes model for the k th task, cf. Eq. (3), and for any h = ( h 1 , . . . , h K ) ∈ H 1 ×··· × H K , define g h s := arg min g ∈G d s ( g ; h ) , so that g s = g f ⋆ s , cf. Eq. (4). The result depends on the Rademacher complexities of the following sets of functions, defined using the balls B ∥·∥ k = { f ∈ F all : ∥ f ∥ k ≤ 1 } as

Specifically, as proved in Lemma D.7, Assumption 4 is equivalent to imposing ∥·∥ k ≤ η ∥·∥ s for all k ∈ [ K ] and s ∈ S . Sufficient conditions for Assumption 4 are that all weights of the scalarizations in S are bounded away from zero, or P k X ≪ P j X and ess sup dP k X /dP j X ≤ η 2 for all k, j ∈ [ K ] .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Critical radii like these are the key quantities of localized generalization bounds [9, 36]. They can be bounded using VC dimension (Lemma E.6) or with (generic) chaining [22, 61]. We define the worstcase critical radius in G by replacing f ⋆ in the definition of u k with a supremum over ground-truth functions h , ¯ u k := sup h ∈H 1 ×···×H K inf { r ≥ 0 : r 2 ≥ R k N k ( G k ( r ; h )) } , and then clearly u k ≤ ¯ u k .

<!-- formula-not-decoded -->

Theorem 2. Let S ⊆ S lin be a set of linear scalarizations, and let Assumptions 1 to 3 hold. Then, if δ &gt; 0 is sufficiently small, the output { ̂ g s : s ∈ S} from Algorithm 1 satisfies ( S -MOL) with probability 1 -δ and ε s = s ( ε 1 , . . . , ε K ) , where and C k = ( ν 3 (1+diam ∥·∥ 2 ( Y )) / γ 2 ) max { L 2 k / γ 2 + B k / γ , L 2 k / µ 2 k + B k / µ k } . 4 If additionally Assumption 4 holds, then for l 2 S = sup s ∈S s ( l 2 1 , . . . , l 2 K ) and n S = (sup s ∈S s (1 /n 1 , . . . , 1 /n K )) -1 we have

ε k ≲ ˜ C k ( u 2 k + l 2 S +( N -1 k + n -1 S ) log(4 K/δ ) ) , (14) with ˜ C k = C k · ( ην / γ ) 2 max k ∈ [ K ] ( B k / µ k + L 2 k / µ 2 k ) . The proof of Theorem 2 can be found in Appendix D.4. By comparing Eq. (10) with Eq. (13), we can see that, under the additional assumptions, Theorem 2 yields much better rates than Theorem 1, whenever the critical radii are (much) smaller than the global Rademacher complexities. Effectively, Theorem 1 provides a 'slow rate' analysis, while Theorem 2 provides a 'fast rate' analysis. Additionally, Theorem 2 avoids the 'doubly slow rate' R k n k ( H k ) 1 / 2 that appears in Theorem 1, and hence can potentially yield a speed-up of power 4 over Theorem 1; e.g., if H has VC (subgraph) dimension d H , the label complexity reduces to order O ( Kd H /ε ) compared to the O ( Kd H /ε 4 ) from Theorem 1.

Adaptivity and weakening Assumption 4. While Eq. (13) depends on the worst location of the true Pareto set g s in G (through ¯ u k ), Eq. (14) refines this bound by also showing the adaptivity of the algorithm to the specific location of the true Pareto set g s in G . Depending on the geometry of G , this set may lie in a 'low complexity region' of G . If that is the case, then the radii u k can be smaller than ¯ u k , and the bound adapts to this low complexity. But this comes at a cost: to prove Eq. (14), we require the norm equivalence from Assumption 4, and have to replace l k by l S . Intuitively, the distance of ̂ g s to g s can only be controlled in the norm ∥·∥ s ; in particular, if λ k = 0 , then there is no reason that ̂ g s should be close to g s in the norm ∥·∥ k . But u k , defined through ∥·∥ k , has to bound the k th coordinate for all scalarizations s ∈ S , making the norm equivalence from Assumption 4 necessary. For finite sets of scalarizations, on the other hand, this can be avoided (but replaced by a union bound), see Corollary A.2. Hence, there seems to be an inherent tension between controlling the error for all scalarizations simultaneously and proper adaptivity to the local complexity of the problem. It is interesting to explore this tension further.

˜ ˜ In the setting where the algorithm has access to the marginals { P k X } K k =1 , called the ideal semisupervised setting [70], the proof of Theorem 2 also yields a slightly tighter bound than (13) (by combining Eqs. (32) and (36)). Under Assumptions 1 and 2, we obtain ε s = s ( ε 1 , . . . , ε K ) with ε k ≲ C k ( l 2 k + n -1 k log(2 K/δ ) ) , where C k = ν 3 γ 2 ( 1 + diam ∥·∥ 2 ( Y ) ) ( B k / µ k + L 2 k / µ 2 k ) .

## 5 Example: non-parametric regression with Lipschitz functions

We now exemplify the benefit of Theorem 2 in an example where the localized rates are much faster than unlocalized ones. More examples are presented in Appendix C.

Let X = [0 , 1] , Y = [0 , 1] and let ℓ k be the square loss. Define for 0 &lt; L H &lt; L G the function classes H = { h : [0 , 1] → [0 , 1] : h is L H -Lipschitz } and G = { g : [0 , 1] → [0 , 1] : g is L G -Lipschitz } . Furthermore, let K = 2 and P k X have a density p k on [0 , 1] with respect to the Lebesgue measure. For Eq. (3) to hold, assume that there exist two functions f ⋆ 1 , f ⋆ 2 ∈ H for which E [ Y k | X k = x ] = f ⋆ k ( x ) for all x ∈ [0 , 1] . We now apply Theorem 2 to obtain upper bounds for S -MOL in this setting.

4 We assume min { L k , µ k } /γ ≥ 1 for all k ∈ [ K ] ; otherwise, remove the squares on each ratio.

Corollary 1. Let S ⊆ S lin be a set of linear scalarizations and assume the functions from Eq. (9) are L G -Lipschitz. Then the output { ̂ g s : s ∈ S} of Algorithm 1 satisfies ( S -MOL) with probability 0 . 99 and ε s = s ( ε 1 , . . . , ε K ) where ε k ≲ ( L H / n k ) 2 / 3 +( L G / N k ) 2 / 3 for all s ∈ S .

Figure 2: On the left: one fit of the methods on 5 labeled and 100 unlabeled samples with weights λ = (1 / 2 , 1 / 2) . In the center : excess s -trade-off as a function of labeled and unlabeled sample sizes for fixed weights λ = (1 / 2 , 1 / 2) . We fix the unlabeled and labeled sample sizes to 2 12 and 2 5 , respectively. On the right: the excess s -trade-off of PL-MOL as a function of unlabeled sample size N 1 while n 1 = n 2 = N 2 = 2 5 are fixed, and for varying weights. We repeat each experiment 10 times and show median, 20% and 80% quantiles.

<!-- image -->

The proof of Corollary 1 can be found in Appendix C.3. Note that we recover the familiar minimax rate n -2 / 3 of Lipschitz regression. In comparison, the crude, unlocalized bound from Theorem 1 would yield the potentially much slower rates L 1 / 4 H n -1 / 4 k + L 1 / 2 G N -1 / 2 k .

We illustrate Corollary 1 in Fig. 2 on the following example: Let H be a set of almost constant functions (that is, L H = 0 . 2 ), and let f ⋆ 1 ≡ a and f ⋆ 2 ≡ b for two constants a, b ∈ [0 , 1] . Minimizing T s ( h ) for the weights λ = (1 / 2 , 1 / 2) over H yields the solution h s ≈ ( a + b ) / 2 while for large enough L G , the solution in G becomes g s = ( p 1 a + p 2 b ) / ( p 1 + p 2 ) . On the left of Fig. 2, we show one data instance and the resulting models from Algorithms 1 and 2 when the densities are p 1 ( x ) = 0 . 7 sin(20 x ) + 1 and p 2 = 2 -p 1 . In the center and on the right, we show the excess s -trade-off in this setting as a function of sample size. We can see the rates predicted by Corollary 1: when we fix the unlabeled sample sizes as large enough ( N 1 = N 2 = 2 12 ), PL-MOL achieves a small excess s -trade-off already for small labeled sample sizes. Meanwhile, ERM-MOL requires a labeled sample size to be of the same order 2 12 before it achieves a similar excess s -trade-off. At the same time, if we fix the labeled sample size sufficiently large to learn the almost constant functions in H , only PL-MOL improves with an increasing number of unlabeled data. In both cases, the familiar n -2 / 3 -rate from Lipschitz regression is observable, as also predicted by Corollary 1. Finally, on the right of Fig. 2, we see that if we keep all sample sizes fixed-except for N 1 -, then the rates are eventually bottlenecked by the harder task for all scalarizations; the risks stagnate at λ 2 N -2 / 3 2 ≍ λ 2 .

## 6 Discussion

This work studies when it is possible to mitigate the statistical cost of multi-objective learning, in which we illuminate the roles of unlabeled data and of the loss functions. This need arises because the function classes that contain models achieving good trade-offs may need to be much larger than those that are well-suited for any one task. We show that for general losses, the label complexity of learning multiple trade-offs simultaneously in a class G is determined solely by the complexity of G , even when the learner has full access to marginal distributions and the Bayes optimal models for each task (Proposition 1). But for Bregman losses, a simple pseudo-labeling algorithm can significantly reduce the label complexity (Theorem 1), where unlabeled data can fully absorb the statistical cost of the expressive model class. Our analysis with local Rademacher complexities further refines these bounds (Theorem 2) and shows adaptivity of the algorithm under some conditions.

The key property that the pseudo-labeling algorithm exploits is the risk decomposition from Lemma 1, which is unique to (generalized) Bregman losses [28]. Nevertheless, it is interesting to determine for exactly which losses the semi-supervised setting can improve upon the supervised one beyond Bregman losses. Under stronger assumptions, we provide a first result of this kind in Appendix B.

Future work may also investigate the tension between controlling the errors of all scalarizations and adaptive rates, and in this context, whether Assumption 4 is really necessary (see discussion in Section 4.2). Moreover, it would be interesting to relax structural assumptions in Theorem 2, e.g., by generalizing it to non-linear scalarizations, and to apply our framework to generative models.

## Acknowledgements

We thank Konstantin Donhauser for helpful discussions. TW was supported by SNSF Grant 204439 and JP by SNSF Grant 218343. GS was partially supported by the NSF award CCF-2112665 (TILOS), DARPA AIE program, the U.S. Department of Energy, Office of Science, the Facebook Research Award, as well as CDC-RFA-FT-23-0069 from the CDC's Center for Forecasting and Outbreak Analytics. This work was done in part while the authors were visiting the Simons Institute for the Theory of Computing.

## References

- [1] A. Agarwal and T. Zhang. Minimax regret optimization for robust machine learning under distribution shift. In Proceedings of the Conference on Learning Theory (COLT) , pages 2704-2729, 2022.
- [2] C. D. Aliprantis and K. C. Border. Infinite dimensional analysis: A hitchhiker's guide . Springer Science &amp; Business Media, 2006.
- [3] R. K. Ando, T. Zhang, and P. Bartlett. A framework for learning predictive structures from multiple tasks and unlabeled data. Journal of Machine Learning Research (JMLR) , 6(11), 2005.
- [4] P. Awasthi, N. Haghtalab, and E. Zhao. Open Problem: The Sample Complexity of Multi-Distribution Learning for VC Classes. In Proceedings of the Conference on Learning Theory (COLT) , pages 5943-5949, 2023.
- [5] P. Awasthi, S. Kale, and A. Pensia. Semi-supervised group DRO: Combating sparsity with unlabeled data. In Proceedings of the Conference on Algorithmic Learning Theory (ALT) , pages 125-160, 2024.
- [6] M.-F. Balcan and A. Blum. A discriminative model for semi-supervised learning. Journal of the ACM , 57(3):1-46, 2010.
- [7] A. Banerjee, X. Guo, and H. Wang. On the optimality of conditional expectation as a Bregman predictor. IEEE Transactions on Information Theory , 51(7):2664-2669, 2005.
- [8] A. Banerjee, S. Merugu, I. S. Dhillon, and J. Ghosh. Clustering with Bregman divergences. Journal of Machine Learning Research (JMLR) , 6:1705-1749, 2005.
- [9] P. L. Bartlett, O. Bousquet, and S. Mendelson. Local Rademacher complexities. Annals of Statistics , 33(4):1497-1537, 2005.
- [10] P. L. Bartlett and S. Mendelson. Rademacher and Gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research (JMLR) , 3:463-482, 2002.
- [11] P. L. Bartlett and S. Mendelson. Empirical minimization. Probability theory and related fields , 135(3):311334, 2006.
- [12] H. H. Bauschke and P. L. Combettes. Convex Analysis and Monotone Operator Theory in Hilbert Spaces . Springer Cham, 2 edition, 2017.
- [13] S. Ben-David, T. Lu, and D. Pál. Does Unlabeled Data Provably Help? Worst-case Analysis of the Sample Complexity of Semi-Supervised Learning. In Proceedings of the Conference on Learning Theory (COLT) , pages 33-44, 2008.
- [14] A. Blum, N. Haghtalab, A. D. Procaccia, and M. Qiao. Collaborative PAC learning. Advances in Neural Information Processing Systems (NeurIPS) , 30, 2017.
- [15] G. Brown and R. Ali. Bias/Variance is not the same as Approximation/Estimation. Transactions on Machine Learning Research (TMLR) , 2024.
- [16] C. L. Canonne. A short note on an inequality between KL and TV. arXiv preprint arXiv:2202.07198 , 2022.
- [17] O. Chapelle, B. Schölkopf, and A. Zien. Semi-Supervised Learning . MIT Press, 2006.
- [18] L. Chen, H. Fernando, Y. Ying, and T. Chen. Three-way trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance. Advances in Neural Information Processing Systems (NeurIPS) , 36:70045-70093, 2023.

- [19] C. Cortes, M. Mohri, J. Gonzalvo, and D. Storcheus. Agnostic learning with multiple objectives. Advances in Neural Information Processing Systems (NeurIPS) , 33:20485-20495, 2020.
- [20] L. Devroye, L. Györfi, and G. Lugosi. A Probabilistic Theory of Pattern Recognition . Springer New York, 1 edition, 1996.
- [21] K. Dowd and D. Blake. After VaR: The theory, estimation, and insurance applications of quantile-based risk measures. Journal of Risk and Insurance , 73(2):193-229, 2006.
- [22] R. M. Dudley. The sizes of compact subsets of Hilbert space and continuity of Gaussian processes. Journal of Functional Analysis , 1(3):290-330, 1967.
- [23] M. Ehrgott. Multicriteria Optimization , volume 491. Springer Science &amp; Business Media, 2005.
- [24] D. J. Foster and A. Rakhlin. ℓ ∞ Vector Contraction for Rademacher Complexity. arXiv preprint arXiv:1911.06468 , 2019.
- [25] Y. Freund and R. E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences , 55(1):119-139, 1997.
- [26] C. Göpfert, S. Ben-David, O. Bousquet, S. Gelly, I. Tolstikhin, and R. Urner. When can unlabeled data improve the learning rate? In Proceedings of the Conference on Learning Theory (COLT) , pages 1500-1518, 2019.
- [27] N. Haghtalab, M. Jordan, and E. Zhao. On-demand sampling: Learning optimally from multiple distributions. Advances in Neural Information Processing Systems (NeurIPS) , 35:406-419, 2022.
- [28] T. Heskes. Bias-variance decompositions: The exclusive privilege of Bregman divergences. arXiv preprint arXiv:2501.18581 , 2025.
- [29] C.-L. Hwang and A. S. M. Masud. Multiple objective decision making-methods and applications: A state-of-the-art survey , volume 164. Springer Science &amp; Business Media, 2012.
- [30] T. Hytönen, J. Van Neerven, M. Veraar, and L. Weis. Analysis in Banach spaces , volume 12. Springer, 2016.
- [31] K. M. Jablonka, G. M. Jothiappan, S. Wang, B. Smit, and B. Yoo. Bias free multiobjective active learning for materials design and discovery. Nature communications , 12(1):2312, 2021.
- [32] Y. Jin. Multi-objective machine learning , volume 16. Springer Science &amp; Business Media, 2007.
- [33] D. F. Jones and H. O. Florentino. Multi-objective optimization: Methods and applications. In The Palgrave handbook of Operations Research , pages 181-207. Springer, 2022.
- [34] J. D. Kalbfleisch and R. L. Prentice. The statistical analysis of failure time data . John Wiley &amp; Sons, 2002.
- [35] V. Kanade, P. Rebeschini, and T. Vaskevicius. Exponential tail local Rademacher complexity risk bounds without the Bernstein condition. Journal of Machine Learning Research (JMLR) , 25(388):1-43, 2024.
- [36] V. Koltchinskii. Local Rademacher Complexities and Oracle Inequalities in Risk Minimization. Annals of Statistics , pages 2593-2656, 2006.
- [37] J. Lee, S. Park, and J. Shin. Learning bounds for risk-sensitive learning. Advances in Neural Information Processing Systems (NeurIPS) , 33:13867-13879, 2020.
- [38] X. Lin, Z. Yang, X. Zhang, and Q. Zhang. Pareto set learning for expensive multi-objective optimization. Advances in Neural Information Processing Systems (NeurIPS) , 35:19231-19247, 2022.
- [39] Q. Liu, X. Liao, and L. Carin. Semi-supervised multitask learning. Advances in Neural Information Processing Systems (NeurIPS) , 20, 2007.
- [40] Y. Mansour, M. Mohri, and A. Rostamizadeh. Domain adaptation with multiple sources. Advances in Neural Information Processing Systems (NeurIPS) , 21, 2008.
- [41] A. Maurer. Bounds for linear multi-task learning. Journal of Machine Learning Research (JMLR) , 7:117-139, 2006.
- [42] A. Maurer. A vector-contraction inequality for Rademacher complexities. In Proceedings of the Conference on Algorithmic Learning Theory (ALT) , pages 3-17, 2016.

- [43] C. McDiarmid. On the method of bounded differences. Surveys in combinatorics , 141(1):148-188, 1989.
- [44] K. Miettinen. Nonlinear multiobjective optimization , volume 12. Springer Science &amp; Business Media, 1999.
- [45] S. Mu and S. Lin. A comprehensive survey of mixture-of-experts: Algorithms, theory, and applications. arXiv preprint arXiv:2503.07137 , 2025.
- [46] A. I. Naimi and L. B. Balzer. Stacked generalization: An introduction to super learning. European journal of epidemiology , 33:459-464, 2018.
- [47] H. Nakayama, Y. Yun, and M. Yoon. Sequential Approximate Multiobjective Optimization using Computational Intelligence . Springer Science &amp; Business Media, 2009.
- [48] A. Navon, A. Shamsian, G. Chechik, and E. Fetaya. Learning the Pareto Front with Hypernetworks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2021.
- [49] K. Nikodem and Z. Pales. Characterizations of inner product spaces by strongly convex functions. Banach Journal of Mathematical Analysis , 5(1):83-87, 2011.
- [50] J. Park and K. Muandet. Towards empirical process theory for vector-valued functions: Metric entropy of smooth function classes. In Proceedings of the Conference on Algorithmic Learning Theory (ALT) , pages 1216-1260, 2023.
- [51] B. Peng. The sample complexity of multi-distribution learning. In Proceedings of the Conference on Learning Theory (COLT) , pages 4185-4204, 2024.
- [52] P. Rigollet. Generalization error bounds in semi-supervised classification under the cluster assumption. Journal of Machine Learning Research (JMLR) , 8(7), 2007.
- [53] A. Roy, G. So, and Y.-A. Ma. Optimization on Pareto sets: On a theory of multi-objective optimization. arXiv preprint arXiv:2308.02145 , 2023.
- [54] M. Ruchte and J. Grabocka. Scalable Pareto front approximation for deep multi-objective learning. In International Conference on Data Mining (ICDM) , pages 1306-1311, 2021.
- [55] M. Seeger. Learning with labeled and unlabeled data. Technical report, University of Edinburgh , 2001.
- [56] B. Sen. A gentle introduction to empirical process theory and applications. Lecture Notes, Columbia University , 2018.
- [57] S. Shalev-Shwartz and S. Ben-David. Understanding machine learning: From theory to algorithms . Cambridge University Press, 2014.
- [58] J. Snell, T. P. Zollo, Z. Deng, T. Pitassi, and R. Zemel. Quantile risk control: A flexible framework for bounding the probability of high-loss predictions. In Proceedings of the International Conference on Learning Representations (ICLR) , 2023.
- [59] P. Súkeník and C. Lampert. Generalization in Multi-Objective Machine Learning. Neural Computing and Applications , pages 1-15, 2024.
- [60] M. Talagrand. Sharper bounds for Gaussian and empirical processes. Annals of Probability , pages 28-76, 1994.
- [61] M. Talagrand. The generic chaining: Upper and lower bounds of stochastic processes . Springer Science &amp; Business Media, 2005.
- [62] R. Verma, V. Fischer, and E. Nalisnick. On calibration in multi-distribution learning. In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAccT) , pages 938-950, 2025.
- [63] M. J. Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge University Press, 2019.
- [64] K. Wang, R. Kidambi, R. Sullivan, A. Agarwal, C. Dann, A. Michi, M. Gelmi, Y. Li, R. Gupta, A. Dubey, et al. Conditional language policy: A general framework for steerable multi-objective finetuning. arXiv preprint arXiv:2407.15762 , 2024.
- [65] T. Wegel, F. Kovaˇ cevi´ c, A. ¸ Tifrea, and F. Yang. Learning Pareto manifolds in high dimensions: How can regularization help? In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS) , 2025.

- [66] D. H. Wolpert. Stacked generalization. Neural networks , 5(2):241-259, 1992.
- [67] N. Yousefi, Y. Lei, M. Kloft, M. Mollaghasemi, and G. C. Anagnostopoulos. Local Rademacher complexitybased learning guarantees for multi-task learning. Journal of Machine Learning Research (JMLR) , 19(38):1-47, 2018.
- [68] B. Yu. Assouad, Fano, and Le Cam. In Festschrift for Lucien Le Cam: research papers in probability and statistics , pages 423-435. Springer, 1997.
- [69] E. Zeidler. Nonlinear Functional Analysis and its Applications . Springer New York, 1 edition, 1985.
- [70] A. Zhang, L. D. Brown, and T. T. Cai. Semi-supervised inference: General theory and estimation of means. Annals of Statistics , 47(5):2538-2566, 2019.
- [71] R. Zhang and D. Golovin. Random hypervolume scalarizations for provable multi-objective black box optimization. In Proceedings of the International Conference on Machine Learning (ICML) , 2020.
- [72] Y. Zhang and Q. Yang. An overview of multi-task learning. National Science Review , 5(1):30-43, 2018.
- [73] Z. Zhang, W. Zhan, Y. Chen, S. S. Du, and J. D. Lee. Optimal multi-distribution learning. In Proceedings of the Conference on Learning Theory (COLT) , pages 5220-5223, 2024.
- [74] M. Zuluaga, A. Krause, and M. Püschel. e-PAL: An active learning approach to the multi-objective optimization problem. Journal of Machine Learning Research (JMLR) , 17(104):1-32, 2016.
- [75] M. Zuluaga, G. Sergent, A. Krause, and M. Püschel. Active learning for multi-objective optimization. In Proceedings of the International Conference on Machine Learning (ICML) , pages 462-470, 2013.
- [76] A. ¸ Tifrea, G. Yüce, A. Sanyal, and F. Yang. Can semi-supervised learning use all the data effectively? A lower bound perspective. Advances in Neural Information Processing Systems (NeurIPS) , 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Each statement in the abstract has a corresponding part in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The negative result (Proposition 1) positions the limitations of our approach, together with the discussion of assumptions and results in Section 4 and in Section 6.

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

Justification: We explicitly state all assumptions, that is, Assumptions 1, 2 and 4

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

Justification: Our work is theoretical, and we only conduct toy experiments that are easily reproducible from the problem setups.

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

Justification: The paper is theoretical. The only experiments that we run are detailed in Section 5 and Appendix C, and do not present a main contribution of this work. They can easily be reproduced from the description in these sections.

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

Justification: We provide full details on the one toy experiments we run in Section 5 and Appendix C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The only figure where this is necessary is Fig. 3, where we report the interquartile ranges as a measure of error.

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

Justification: We only have have toy experiments (Figs. 2 and 3) which execute within a minutes on a standard laptop, as described in Section 5 and Appendix C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We reviewed the code of conduct.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The work is mostly theoretical, and does not have any direct societal impacts.

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

Justification: The core method development in this research does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

## Table of Contents

| A   | Related work                                                 | Related work                                                 |   22 |
|-----|--------------------------------------------------------------|--------------------------------------------------------------|------|
|     | A.1                                                          | A posteriori multi-objective learning                        |   23 |
|     | A.2                                                          | Adjacent related works . . . . . . .                         |   25 |
|     | A.3                                                          | Proofs for ERM-MOL . . . . . . . .                           |   26 |
| B   | Beyond Bregman losses: Pseudo-labeling for the zero-one loss | Beyond Bregman losses: Pseudo-labeling for the zero-one loss |   27 |
|     | B.1                                                          | Proof of Proposition B.1 . . . . . . .                       |   28 |
| C   | More examples                                                | More examples                                                |   29 |
|     | C.1                                                          | Binary classification . . . . . . . . .                      |   29 |
|     | C.2                                                          | ℓ 2 -constrained linear regression . . .                     |   31 |
|     | C.3                                                          | Proofs for the Examples . . . . . . .                        |   31 |
| D   | Proofs of main results                                       | Proofs of main results                                       |   34 |
|     | D.1                                                          | Proof of Proposition 1 . . . . . . . .                       |   34 |
|     | D.2                                                          | Proof of Lemma 1 . . . . . . . . . .                         |   37 |
|     | D.3                                                          | Proof of Theorem 1 . . . . . . . . .                         |   38 |
| D.4 | Proof of Theorem 2                                           | . . . . . . . . .                                            |   40 |
|     | D.4.1                                                        | Preliminaries . . . . . . . .                                |   40 |
|     | D.4.2                                                        | Main proof of Theorem 2 . .                                  |   41 |
|     | D.4.3                                                        | Proof of preliminary lemmata                                 |   45 |
|     | D.4.4                                                        | Proof of Lemma D.8 . . . .                                   |   47 |
|     | D.4.5                                                        | Proof of Lemma D.9 . . . .                                   |   47 |
|     | D.4.6                                                        | Proof of Proposition D.1 . .                                 |   49 |
|     | D.4.7                                                        | Proof of Proposition D.2 . .                                 |   50 |
|     | D.4.8                                                        | Proof of Proposition D.3 . .                                 |   53 |
| E   | Auxiliary results                                            | Auxiliary results                                            |   54 |
|     | E.1                                                          | . . . . .                                                    |   54 |
|     | Concentration inequalities . Rademacher complexities .       | . . . . .                                                    |   55 |
| F   | E.2 Table of Notations                                       | E.2 Table of Notations                                       |   56 |

## A Related work

We review related work; for S -MOL in Appendix A.1 and more adjacent literature in Appendix A.2.

Semi-supervised learning. Semi-supervised single-objective learning is a well-established field of research, and the question of when and how unlabeled data can help in a single learning problem is rather subtle [17, 26, 76, 6]. Interestingly, the reason that unlabeled data helps in our setting is quite different from how it can help in single-objective learning. Contrary to our setup, results that demonstrate a benefit of semi-supervised settings in single-objective learning usually require the marginals to carry some form of information about the labels (such as clusterability, manifold structure, low-density separation, smoothness, compatibility, etc.) [55, 52, 17], without which semisupervised learners are no better than ones that discard the unlabeled data altogether [26, 13]. Our results, on the other hand, hold regardless of such assumptions: if the likelihood of a sample is higher under one task than another, a model with a good trade-off prioritizes that task, and unlabeled data enables (implicitly) estimating that likelihood. This is true, even if that likelihood carries no additional information about the labels.

- Algorithm 2 ERM for Multi-objective Learning (ERM-MOL) Input: Labeled data { ( X k i , Y k i ) } n k i =1 , hypothesis space G , scalarization set S . 1: for k ∈ [ K ] do 2: Define the empirical risk functional: ̂ R k ( g ) := 1 n k ∑ n k i =1 ℓ k ( Y k i , g ( X k i ) ) . 3: end for 4: for s ∈ S do 5: Minimize the empirical s -trade-off: ̂ g s = arg min g ∈G s ( ̂ R 1 ( g ) , . . . , ̂ R K ( g ) ) . 6: end for 7: Return { ̂ g s : s ∈ S} .

A priori and a posteriori decision making. In multi-objective optimization, decision makers can be broadly categorized based on whether they have an a priori or an a posteriori preference over Pareto solutions [29, 33]. An a priori decision maker aims to recover a specific Pareto model, which is the solution to a trade-off T s that is known beforehand. In contrast, an a posteriori decision maker will first recover the whole Pareto set. Recall from Section 2.3 or [23] that, under mild conditions, this entails solving a family of optimization problems ∀ s ∈ S , min g ∈G T s ( g ) . The preference of such a decision maker is then informed by the set of trade-offs that are possible.

The learning version of the problem has been studied for both types of decision makers, where the trade-off functionals need to be estimated from data. This leads to two types of algorithms and generalization bounds. The a priori approach has been especially developed in the context of learning with fairness or multi-group constraints [27, 4, 51, 73]. The a posteriori approach, to which our work mostly belongs, has been studied by [19, 59].

Multi-distribution learning. In the (supervised) multi-distribution learning (MDL) setting, the goal is to learn only one s -trade-off for the scalarization s ( v ) = max v k , which belongs to the family of a priori decision making. Then, for VC-classes of dimension d G , the label complexity to achieve excess s -trade-off ε &gt; 0 is ˜ Θ(( K + d G ) /ε 2 ) using an on-demand sampling framework, in which the algorithm is allowed to decide which distribution to sample from sequentially [27, 4, 51, 73]. Importantly, this adaptive sampling improves upon the 'trivial' rate Θ( Kd G /ε 2 ) (see [73] and Appendix A.1) by removing the multiplicative dependence on the number of objectives. For nonadaptive sampling, the rate Θ( Kd G /ε 2 ) is tight, that is, the fact that the algorithm has to solve only one scalarization does not improve upon the sample complexity of solving all scalarizations, cf. Corollary A.1. Of course, the statistical complexity under adaptive sampling must fail to hold for S -MOL with multiple scalarizations, because it includes all individual learning tasks. MDL is also related to collaborative (where the tasks are assumed to share a ground-truth), federated, and group DRO frameworks, for which we refer the readers to the discussions in [27, 73, 14].

In [5], the authors propose a semi-supervised framework for group DRO (a problem related to MDL). The underlying assumption in [5] is that for each label-scarce group, there exists a group with sufficiently much labeled data and which is 'related enough' for cross-group pseudo-labeling to be effective (similar to the collaborative learning setup).

## A.1 A posteriori multi-objective learning

Empirical risk minimization for S -MOL. Applying empirical risk minimization (ERM) on labeled data to solve S -MOL was analyzed in [19, 59] and in [18] through algorithmic stability. ERM, or perhaps more aptly empirical trade-off minimization, is a natural approach to learning all Pareto solutions. The idea is simply to use labeled data sampled for each of the K tasks to empirically estimate the s -trade-off functional T s of any model. The Pareto set can then be found by minimizing the estimated trade-offs. This algorithm, that we call empirical risk minimization for multi-objective learning (ERM-MOL), is formalized in Algorithm 2.

Learning the Pareto set through ERM has been described and analyzed by [19], where S is a family of linear scalarizations. In particular, they provide a sample complexity upper bound that depends on the complexity of S through a covering number of the weights that appear in S . Later, [59] extended the ERM framework to go beyond the empirical estimator of the risk functionals, allowing

for any 'statistically valid' estimator based on uniform convergence. They further improve the sample complexity upper bound by removing dependency on S in [59, Theorem 2]. Their result can be used to derive bounds for ERM in S -MOL: we now instantiate their bound in our setting (see Section 2.1), making the following assumption to enable comparison with our results:

Assumption 5 (Regularity conditions for ERM-MOL) . The risk and excess risk functionals are equal,

<!-- formula-not-decoded -->

Proposition A.1 (Sample complexity of ERM-MOL) . Suppose that Assumption 5 holds and that the loss ℓ k ( · , · ) is bounded by B and L k -Lipschitz continuous in the second argument for each k ∈ [ K ] . Let S be any class of monotone scalarizations satisfying reverse triangle inequality and positive homogeneity (6) . Then, for any δ ∈ (0 , 1) , the class of solutions returned by Algorithm 2, { ̂ g s : s ∈ S} , satisfies ( S -MOL) with probability 1 -δ and ε s = s ( ε 1 , . . . , ε K ) , where for each k ∈ [ K ] , ε k is given by:

<!-- formula-not-decoded -->

We demonstrate the implications of this bound for a VC class and for linear scalarizations below, using the VC bound on the Rademacher complexity (Lemma E.6), see Appendix A.3 for proofs.

̸

Corollary A.1. Let G be any hypothesis class with VC dimension d G ∈ N on data domain X × Y where Y = { 0 , 1 } . For each task k ∈ [ K ] , define ℓ k ( y, y ′ ) = 1 { y = y ′ } be the zero-one loss (cf. Definition 3). Let ( P 1 , . . . , P K ) be any tuple of data distributions over X ×Y . Then, for any δ, ε &gt; 0 , the output of Algorithm 2 { ̂ g s : s ∈ S lin } satisfies ( S -MOL) with probability 1 -δ and ε s = ε for all s ∈ S lin whenever the number of samples is at least n k = Ω ( d G +log( K/δ ) ε 2 ) for each k ∈ [ K ] .

Dependence on the size of S in our results. The authors in [59] noted that the dependence on S in [19] is sub-optimal, in the worst case by a factor of K log n k , and that such a dependence could be removed. Here, we should add that this is only true because in [59] the learning bounds are globally uniform-no localization bounds were derived. Similarly, the bound from our Theorem 1 is also independent of the size of S . Theorem 2, on the other hand, paints a more nuanced picture: the size of the sets G k ( r ; f ⋆ ) from Eq. (11) depends on the size of S through a union: if all local neighborhoods of the g s are 'similar,' then S does not affect the bound at all. However, if the local neighborhoods are very different, then the union may be larger than any of the individual neighborhoods and hence the bound will grow with the size of S ; see the right side of Fig. 5. See also the discussion in Section 4.2. Nonetheless, if S is finite, Theorem 2 also yields the following bound.

<!-- formula-not-decoded -->

Corollary A.2. Let S ⊆ S lin be finite and let Assumptions 1 and 2 hold. Define

Then, if δ &gt; 0 is sufficiently small, the output { ̂ g s : s ∈ S} from Algorithm 1 satisfies ( S -MOL) with probability 1 -δ and ε s = s ( ε 1 , . . . , ε K ) , where with ˜ C k = ˜ C k ( s ) from Eq. (14) where η 2 = η 2 ( s ) := max { 1 /λ k : k ∈ [ K ] , λ k &gt; 0 } for s = s lin λ .

<!-- formula-not-decoded -->

Proof of Corollary A.2. Consider the setting where S = { s lin λ } is a singleton. Because we only consider this one scalarization, we can make the following case distinction for each k ∈ [ K ] : either λ k = 0 , so we can ignore index k completely, or λ k &gt; 0 and so ess sup dP k X /d ( ∑ K j =1 P j X ) ≤ 1 /λ k . Hence, Assumption 4 is satisfied for S = { s lin λ } with η 2 ( s lin λ ) = max { 1 /λ k : k ∈ [ K ] , λ k &gt; 0 } . The bound for S = { s lin λ } follows from Theorem 2, and the corollary for a finite S follows from a union bound.

Semi-supervised S -MOL. As far as we are aware, we are the first to study the S -MOL problem in the general semi-supervised setting. The closest work to ours is [65], where the question of learning Pareto manifolds in high-dimensional Euclidean space was studied in a semi-supervised setting. They assume that 1) the ground-truths exhibit a sufficiently sparse structure and 2) the objectives have

a benign parametrization (their Assumptions 1,2, and 3): the paper considers parametric function classes, and the algorithm that achieves the bounds requires knowledge about a parameter θ k ∈ R q so that R k depends on distribution P k only through θ k . Estimating these parameters and then performing standard multi-objective optimization can enable learning in high dimensions. The resulting two-stage estimator is similar to our pseudo-labeling algorithm, and they can coincide, e.g., for linear regression with square loss. Moreover, in [65] the necessity of unlabeled data in high-dimensional linear regression is shown. While we borrow an idea for the stability argument in our Proposition D.1, our results apply to far more general settings.

Comparison of label sample complexities. In order to compare the label sample complexity of our results with prior work, we summarize the resulting bounds in Table 1 for VC (subgraph)-classes G and H k = H with VC dimensions d G , d H . Recall that in the ideal setting, the marginals are known.

Table 1: Label complexities up to logarithmic factors from this (gray) and prior work for VC (subgraph) classes. It holds that d H ≤ d G and potentially d H ≪ d G . A definition of d Θ is in Appendix B; both d Θ ≪ d G and d Θ ≫ d G are possible. Note that these results are not strictly comparable, as they depend on varying technical assumptions.

|                                                                                                            | zero-one loss                                                                             | zero-one loss                                        | Bregman loss                                                             |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------|
| problem class                                                                                              | upper bound                                                                               | lower bound                                          | upper bound                                                              |
| supervised MDL supervised S -MOL ideal semi-sup. S -MOL ideal semi-sup. S -MOL (with stronger assumptions) | d G + K ε 2 [73, 51] Kd G ε 2 [59] / Cor. A.1 Kd G ε 2 [59] / Cor. A.1 Kd Θ ε 4 Prop. B.1 | d G + K ε 2 [27] Kd G ε 2 Prop. 1 Kd G ε 2 Prop. 1 - | d G + K ε 2 [73] Kd G ε 2 [59] / Prop. A.1 Kd H ε 4 Thm. 1 Kd H ε Thm. 2 |

## A.2 Adjacent related works

There are many works considering multi-risk settings in different contexts (not to be confused with the multiple competing risks in survival analysis, cf. [34]), for example, in fairness or insurance mathematics through the lens of multiple quantile risk measures [21, 37, 58]. Our work specifically is related to the fields of ensembling, multi-task learning, and Pareto set learning.

Learning multiple models for one task. Recall that in our Algorithm 1, we first learn multiple models (one per task), and combine them into a family of models that trade off the different risks. In comparison, there are many different ways in which combining multiple models can also help on a single task , usually by using some sort of ensembling. For instance:

- Stacked generalization combines multiple base models via a meta-model that takes their predictions as input features and outputs the final prediction [66, 46].
- Mixture-of-Experts models maintain a collection of expert predictors, and use a routing mechanism to select one or more experts based on the new input. This routing is often done through a direct soft gating or a weighted combination of the models [45].
- Boosting aggregates multiple weak learners to form a single strong predictor for one task, typically through sequential training where each model corrects the errors of the previous ensemble [25].

In contrast to any of these methods, our pseudo-labeling algorithm uses the predictions from individual models (in our work ERMs for simplicity) as training targets and fits a new model (or family of models) from scratch using the unlabeled inputs. This distinction is essential: unlike the aforementioned methods, our algorithm does not aggregate existing models to solve a single task, but instead leverages them as a supervisory signal to reduce the statistical cost of learning trade-offs in a richer function class. In particular, the described methods do not address the core challenge in MOL: the need to reconcile conflicting objectives within a single model. Our method explicitly constructs a family of joint predictors that trades off competing risks and can-or sometimes even must-deviate significantly from any of the base models.

Learning multiple models for multiple tasks. Multi-task learning (MTL), including semisupervised MTL, is a problem that is related to MOL in that both are used in settings where multiple

learning problems need to be solved. However, in MTL, the aim is to learn multiple models, one per task, and exploit relatedness between tasks to improve sample complexity [39, 72]. As such, the problem of striking a trade-off, which is at the heart of MOL, is not present in MTL. For example, suppose a new instance x ∈ X is observed. In MTL, we can make multiple different predictions, one per task, in the hope that each prediction is good for the corresponding task. In MOL, on the other hand, we have to commit to one prediction for all tasks. Aside from these differences, as mentioned, if there is a relationship between the different learning problems, we could employ off-the-shelf MTL algorithms to adapt our pseudo-labeling algorithm by learning the task-specific models in the first part of the algorithm (Line 2 in Algorithm 1). Finally, from a technical perspective, it is worth mentioning that (localized) Rademacher complexities have been used for MTL in [67, 3, 41, 50].

Learning for multi-objective optimization. A recent line of research has introduced the socalled Pareto set learning (PSL) framework [54, 38, 48], which has found various applications, e.g., in finetuning language models on multiple objectives [64]. PSL is an approach to making learning algorithms such as Algorithms 1 and 2 computationally tractable: instead of producing a family of models, one for each trade-off, PSL approximates this family with one fixed function that takes both weights of the objectives and covariates as input (often called a hypernetwork [48]). However, importantly, there is no direct connection of PSL to the learning part of the MOL problem: it is actually purely a computational technique . Specifically, if one approximates the outputs of Algorithms 1 and 2 with PSL, then it inherits their statistical guarantees up to the approximation errors. The name Pareto set learning has its origin in the fact that to find such a PSL function, it is common to minimize some expected scalarization, where the expectation is taken with respect to weights of the objectives [71]. A standard way to make this tractable is to sample the weights [48]. Generalization is then usually discussed in terms of the number of sampled weights, not the data. See also [65] for a discussion. Finally, beyond hypernetworks, various other learning techniques have been deployed for multi-objective optimization when evaluating the objectives is expensive, such as active learning in [75, 74, 31].

## A.3 Proofs for ERM-MOL

Proof of Proposition A.1. The proof is analogous to the proof of [59, Theorem 2], additionally using Rademacher complexity and McDiarmid's bound to bound the supremum (denoted C N in [59]) and slightly different assumptions on the scalarizations. We repeat the proof here for completeness.

Fix δ &gt; 0 . By a standard Rademacher bound (also see Appendix D.3), the following generalization guarantee holds for each task k ∈ [ K ] ,

For any scalarization s , let ̂ T s denote the empirical s -trade-off

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ T s ( g ) = s ( ̂ R 1 ( g ) , . . . , ̂ R K ( g ) ) . Then, T s is well-approximated by ̂ T s . By Assumption 5, R k = E k , so that when the event Eq. (16) holds for all k ∈ [ K ] , and this occurs with probability at least 1 -δ by a union bound, we obtain that:

where the first inequality used the reverse triangle inequality, and the second inequality used the above claim and the monotonicity of the scalarization. In particular, this will allow us to bound the excess s -trade-off of g s as follows:

<!-- formula-not-decoded -->

where the (a) and (c) terms both contribute at most s ( ε 1 / 2 , . . . , ε K / 2) error from Equation (17), while the (b) term is non-positive, since ̂ g s minimizes the empirical s -trade-off ̂ T s . The last inequality follows from the positive homogeneity of the scalarizations.

Proof of Corollary A.1. From Proposition A.1 and the VC bound on Rademacher complexity (Lemma E.6), there exists a constant C &gt; 0 such that for each k ∈ [ K ] we have ε k ≤ ε for whenever n k is sufficiently large:

<!-- formula-not-decoded -->

The result follows, since we have that for any s ∈ S lin :

which concludes the proof.

<!-- formula-not-decoded -->

## B Beyond Bregman losses: Pseudo-labeling for the zero-one loss

Recall that Proposition 1 is a worst-case negative result that rules out any benefit of unlabeled data for MOL, at least in the absence of any additional structure. Indeed, our main results focus on overcoming this hardness for more structured losses-namely, Bregman losses. But, there are other natural forms of structure to consider; in this section, we take an alternative approach.

To help motivate this next approach, let us revisit the reason that Bregman losses are amenable to the semi-supervised approach. As we discuss in Section 3.2, the crux is Lemma 1. It expresses the excess risk functionals E k in terms of a discrepancy operator d k ( · ; · ) and the Bayes-optimal model f ⋆ k , as follows:

<!-- formula-not-decoded -->

Estimating the discrepancy operator over G only makes use of unlabeled data. And even though learning the Bayes-optimal model requires labeled data, its statistical cost is mitigated by the knowledge that H k contains the optimal model. Algorithm 1 precisely constructs estimators of E k in this way, before solving for the s -trade-offs over the learned approximations. The close relationship between E k and f ⋆ k described by Eq. (18) is specific to Bregman losses, see [62, 15, 28]. Nevertheless, the excess risk functional for other losses may have decompositions that are similar in spirit.

Excess risk decomposition for the zero-one loss. To see another instance of excess-risk decomposition, let us revisit the setting of the worst-case examples in Proposition 1: the multi-objective binary classification setting with the zero-one loss (Definition 3). In this case, it is a standard result that the excess risk can be expressed in terms of the conditional mean of the labels θ k ( x ) := E [ Y k | X k = x ] , as in [20, Section 2.1]. In particular, the excess risks for the zero-one loss is given by

E k ( f ) = E [∣ ∣ 2 θ k ( X k ) -1 ∣ ∣ · ∣ ∣ f ( X k ) -1 { θ k ( X k ) ≥ 1 / 2 }∣ ∣ ] . (19) Thus, for the zero-one loss, the form of its corresponding excess risk functional is E k ( · ) = d 0 / 1 k ( · ; θ k ) , where let d 0 / 1 k ( · ; · ) be the 'zero-one discrepancy' operator, given by where θ : X → [0 , 1] is a conditional mean. Notice that the operator d 0 / 1 k ( · ; · ) depends only the marginal distribution P k X , as was the case for Bregman losses (cf. Eq. (18)). This suggests that a semi-supervised approach analogous to Algorithm 1 becomes possible if the regression problem of learning θ k is easy. Note that this is potentially much harder than solving the individual classification tasks, and hence different from the original premise of this work.

<!-- formula-not-decoded -->

We now formalize this intuition. For each k ∈ [ K ] , let Θ k be a class of functions θ : X → [0 , 1] . In lieu of assuming that f ⋆ k ∈ H k , we now assume that the true conditional mean θ k is contained in Θ k (we do not assume that Θ k ⊆ G , especially as Θ k consists of regression functions and G

consists of classifiers). Define the empirical regression map ̂ θ k ( · ) and empirical zero-one discrepancy ̂ d 0 / 1 k ( g ; ̂ θ k ) as follows:

And finally, define the analogous s -scalarization as ̂ d 0 / 1 s ( g ; ̂ θ ) := s ( ̂ d 0 / 1 1 ( g ; ̂ θ 1 ) , . . . , ̂ d 0 / 1 K ( g ; ̂ θ K ) ) . Proposition B.1. In the multi-objective binary classification setting (Definition 3), let S be a class of scalarizations satisfying the reverse triangle inequality and positive-homogeneity (6) . Assume that E [ Y k | X = · ] ∈ Θ k . Then, { ̂ g s ∈ arg min g ∈G ̂ d 0 / 1 s ( g ; ̂ θ ) : s ∈ S} satisfies ( S -MOL) with probability 1 -δ and ε s = s ( ε 1 , . . . , ε k ) , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Proposition B.1 is in Appendix B.1 and is analogous to that of Theorem 1. Here, we can also give a data-independent sample complexity bound. Let d Θ k denote the VC subgraph dimension (a.k.a. pseudo dimension) of Θ k and d G the VC dimension of G . Then, Proposition B.1 implies a label sample complexity on the order of O ( Kd Θ k /ε 4 ) and an unlabeled sample complexity of O ( Kd G /ε 2 ) (see Lemma E.6).

In summary, the hardness result Proposition 1 shows that, in the worst-case, the Bayes classifiers f ⋆ k are uninformative for making appropriate trade-offs over zero-one losses. Instead, Eq. (19) suggests that the relevant information is actually captured by the Bayes regressors θ k . Proposition B.1 makes this intuition rigorous: if we have additional structure, given here in the form of Θ k , we can expect benefits of semi-supervision even for the zero-one loss, as long as the Rademacher complexity of Θ k is manageable.

## B.1 Proof of Proposition B.1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in ( a ) we used that the indicators can only disagree if | θ ′ ( x ) -θ ( x ) | ≥ | θ ( x ) -1 / 2 | . It follows that for any g ∈ G ,

∣ ∣ ∣ ∣ We bound each term separately. First, analogous to Eq. (30) in the proof of Theorem 1, since the square loss is 2 -Lipschitz on [0 , 1] , with probability at least 1 -δ/ (2 K )

For the second term, we use that g ↦→ f θ ( x, g ) is 1 -Lipschitz continuous. Let c ( x ) = 1 { θ ( x ) ≥ 1 / 2 } ; then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, using contraction again, we get the analogous bound to Eq. (29) with probability 1 -δ

<!-- formula-not-decoded -->

In conclusion, we have proved that with probability 1 -δ , for all k ∈ [ K ] , we have

The rest of the proof then follows analogously to the proof of Theorem 1 provided in Appendix D.3 by replacing the 'claim' with the uniform bound in Equation (20).

<!-- formula-not-decoded -->

## C More examples

In this section, we discuss two more examples: classification with logistic loss and another example applying Theorem 2 to linear regression.

## C.1 Binary classification

Denote for any q ∈ [1 , ∞ ] the norm balls B d q = { w ∈ R d : ∥ w ∥ q ≤ 1 } . Suppose that the covariates lie in the space X = B d ∞ ⊂ R d , and that the labels in Y = [0 , 1] for each task follow the Bernoulli distribution where σ ( x ) = 1 / (1 + exp ( -x )) denotes the sigmoid function and we assume that w ⋆ k ∈ B d 1 . This is the standard logistic regression setup. The Bayes-optimal models with respect to the logistic loss ℓ ( y, ˆ y ) = -( y log(ˆ y ) + (1 -y ) log(1 -ˆ y )) are given by f ⋆ k ( · ) = σ ( ⟨· , w ⋆ k ⟩ ) ∈ H = { h ( x ) = σ ( ⟨ x, w ⟩ ) : w ∈ B d 1 } . However, striking a good trade-off between the tasks within H can be impossible (see Fig. 3 for a simple example). To circumvent this issue, we may want to use some feature map Φ : B d ∞ →B p ∞ with p ≫ d , and then learn in the larger function class G = { g ( x ) = σ ( ⟨ Φ( x ) , w ⟩ ) : w ∈ B p 1 } . For example, p = O ( d κ ) and H ⊆ G whenever Φ maps to the set of all polynomial features up to degree κ . In this setting, Algorithm 1 effectively exploits unlabeled data to achieve good trade-offs in the larger function class G , as we show in Corollary C.1. This straightforwardly follows from Theorem 1; the proof is given in Appendix C.3.

<!-- formula-not-decoded -->

Corollary C.1 (Logistic regression) . In the setting described above, let S be some class of scalarizations satisfying reverse triangle inequality and positive homogeneity. Suppose that min k ∈ [ K ] N k ≥ log( p + K ) and min k ∈ [ K ] n k ≥ log( d + K ) . Then, the output of Algorithm 1 { ̂ g s : s ∈ S} satisfies ( S -MOL) with probability at least 0 . 99 and ε s = s ( ε 1 , . . . , ε K ) where ε k ≲ ( log( dK ) / n k ) 1 / 4 +( log( pK ) / N k ) 1 / 2 .

We can also empirically observe the benefits of the semi-supervised method, Algorithm 1 (PL-MOL), over purely supervised approaches-namely, over running Algorithm 2 to learn models from either H (ERM-MOL linear) or G (ERM-MOL polynomial). Fig. 3 visualizes a toy classification problem with linear scalarization, and it compares the resulting decision boundaries, Pareto fronts, and excess s -trade-off across the different approaches.

Specifically, consider the data from Fig. 3a: the support of task 2 is completely contained within the support of task 1 , and in particular, there is an area where the labels of the two tasks disagree (the bottom left 'striped rectangle'). Both tasks are optimally solvable by linear models, but trying to solve both tasks at the same time is impossible, even in F all . Meanwhile, better trade-offs still become available using, e.g., polynomial features.

Figure 3: Learning trade-offs in the classification problem visualized in Fig. 3a. We show 1) supervised linear models, 2) supervised polynomial kernels, and 3) the mixture through the semi-supervised PL-MOL algorithm. (b) The training data and decision boundaries of the three methods, with a score threshold of 1 / 2 , for varying trade-off parameters λ 1 . (c) The Pareto fronts for logistic loss and the s -trade-off as a function of the parameter λ 1 in the linear scalarization. (d) The Pareto fronts for zero-one loss and the s -trade-off as a function of the parameter λ 1 . We repeat the experiment 10 times and show corresponding interquartile ranges.

<!-- image -->

We sample n 1 = n 2 = 25 data points uniformly from the regions in Fig. 3a and, in Figs. 3b and 3c, label them according to the linear logistic model Y k | X k = x ∼ Ber( f ⋆ k ( x )) , that is, with noise and in accordance with Eq. (3). Again, we run the three different algorithms on the logistic loss using linear scalarization: ERM-MOL (Algorithm 2) on the function class H of linear models, ERM-MOL on the function class G of linear models on polynomial features up to degree 5 , and PL-MOL (Algorithm 1) using H in the first stage for all tasks, and G in the second stage with an additional number of N 1 = N 2 = 300 unlabeled data points. PL-MOL fits linear models to the labeled data and uses these to predict (soft) pseudo-labels for the unlabeled data, resulting in Fig. 4. Some resulting decision boundaries of each method are shown in Fig. 3b, and the Pareto fronts (on the test data) as well as excess s -trade-offs are shown in Fig. 3c.

Figure 4: Pseudo-labeled data using PL-MOL with the logistic loss.

<!-- image -->

The expected bias-variance trade-off arises here. In this case, the individual tasks can be perfectly solved over the family of linear classifiers H . However, ERM-MOL over H necessarily fails to find good trade-offs, as this model class is insufficiently expressive for the multi-objective learning problem-it has large bias. On the other hand, the ERM-MOL over G yields large estimation error, since there is not enough labeled data to solve for trade-offs over the much larger family of polynomial classifiers-the learned trade-offs have high variance. In contrast, the PL-MOL algorithm reduces this variance using only additional unlabeled data.

In this experiment, we can also corroborate the importance of the loss function. Fig. 3d shows that PL-MOL can be inconsistent when the losses are not Bregman divergences. While the Pareto front found by PL-MOL dominates the other methods, it incorrectly weighs the different objectives per linear scalarization, resulting in a sub-optimal excess s -trade-off. To amplify this effect, in Fig. 3d

we generate the labels in task 1 without any noise, and in task 2 according to this model:

<!-- formula-not-decoded -->

where recall the different regions from Fig. 3a. Merely changing the loss to the zero-one loss then breaks Algorithm 1. Specifically, in Fig. 3d, we show how the same algorithms perform in a large sample regime ( n 1 = n 2 = 400 ). PL-MOL does not attain the best-possible trade-off within G , even when it recovers the Pareto front of G .

## C.2 ℓ 2 -constrained linear regression

We now discuss another example where the localization can yield much tighter results than Theorem 1. To that end, we consider the following problem of constrained linear regression with squared loss.

Let X = B d 2 , Y = [ -1 , 1] , and ℓ k be the squared loss. For R ∈ [0 , 1] , define the hypothesis spaces H = { h ( x ) = ⟨ x, w ⟩ : ∥ w ∥ 2 ≤ R } and G = { g ( x ) = ⟨ x, w ⟩ : ∥ w ∥ 2 ≤ 1 } . We consider distributions that satisfy E [ Y k | X k = x ] = ⟨ w ⋆ k , x ⟩ , that is, a (possibly heteroscedastic) zero-mean noise model where f ⋆ k = ⟨ w ⋆ k , ·⟩ ∈ H ⊂ F all for all k . Suppose that the covariance matrices of X k have smallest eigenvalue bounded from below by κ ∈ [ R, 1] (which is easily satisfied). Theorem 2 then yields the following corollary, proven in Appendix C.3.

<!-- formula-not-decoded -->

Corollary C.2 ( ℓ 2 -constrained linear regression) . In the setting described above, the output of Algorithm 1 satisfies ( S -MOL) with probability 0 . 99 and ε s = s ( ε 1 , . . . , ε K ) for all s ∈ S lin (Eq. (5) ), where

Here 1 /n k and 1 /N k are the localized rates, where Theorem 1 would yield 1 / √ n k and 1 / √ N k instead. Notice that if H is very small (i.e., R &lt; 1 / (2 κn k ) ), then the first term is small due to the smaller complexity of H , while the second term may only become small due to larger unlabeled sample size N k .

<!-- formula-not-decoded -->

## C.3 Proofs for the Examples

In this section we provide the proofs of Corollaries 1, C.1 and C.2.

Proof of Corollary 1. We verify the assumptions of Theorem 2: Eq. (3) holds by definition of the data generating model. Assumption 1 and the smoothness from Assumption 2 hold, because the square loss ℓ is 2 -Lipschitz and 1 -bounded on Y = [0 , 1] , and induced by ϕ ( y ) = y 2 which is 1 -strongly convex, and max { ϕ ′′ , ϕ ′′′ } ≤ 2 . The other parts of Assumption 2 holds because the function classes G and H are convex, and the strong convexity holds with γ s = 1 : For every s ≡ s lin λ ∈ S lin a quick calculation shows that

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

and hence the strong convexity follows from the convexity of g ↦→ ( g -a ) 2 -g 2 for any a ∈ R . Assumption 3 holds because the functions from Eq. (9) are assumed to be L G -Lipschitz and hence by Lemma 2 the global minimizers are contained in G .

To apply Theorem 2, denote the space of 2 L G -Lipschitz functions [0 , 1] → [ -1 , 1] as ˜ G , and note that for any function g ∈ G we have that G g ⊂ G . Hence, we can see that

Denote by N ( t, A , ∥·∥ ) the covering number of a set A with norm ∥·∥ at radius t &gt; 0 , see e.g., [63, Chapter 5] for a definition. It is a standard fact [63, Example 5.10] that the metric entropy of { g ∈ ˜ G : ∥ g ∥ k ≤ r } is bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, using standard bounds with Dudley's entropy integral [22], we can bound the Rademacher complexity of this function class by

<!-- formula-not-decoded -->

and, similarly for H k ( r ) we get that

Solving the corresponding inequalities r 2 ≥ √ L H r n k and r 2 ≥ √ L G r N k yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging this into Eq. (13) from Theorem 2 and noting that 1) for any fixed confidence 1 -δ (such as 0 . 99 ) the confidence term goes to zero faster than the main terms, and 2) the constants C k are universal constants in this example, yields the result.

Proof of Corollary C.1. We apply Theorem 1 to the setting. First, note that f ⋆ k ( x ) = E [ Y k | X k = x ] = σ ( ⟨ x, w ⋆ k ⟩ ) is contained in H , so that Eq. (3) holds. Also note that the loss ℓ ( y, ˆ y ) = -( y log(ˆ y ) + (1 -y ) log(1 -ˆ y )) is a Bregman loss (Definition 4) induced by the potential ϕ ( y ) = y log y +(1 -y ) log(1 -y ) .

We can then check Assumption 1 (using the remark that only the range of G needs to be considered):

Moreover, for all g ∈ G we have that g ( X ) ⊂ [ σ ( -1) , σ (1)] , since for all w ∈ B p 1 and Φ( x ) ∈ B p ∞ we have that |⟨ w, Φ( x ) ⟩| ≤ ∥ w ∥ 1 ∥ Φ( x ) ∥ ∞ ≤ 1 .

1. Because d 2 dy 2 ϕ ( y ) = 1 / ( y (1 -y )) ≥ 4 for all y ∈ [0 , 1] , we have that ϕ is 4 -strongly convex.
2. Making use of the fact that the range of functions in G lies in [ σ ( -1) , σ (1)] , we get that ℓ is L -Lipschitz in both arguments with L = 3 2 σ ( -1) σ (1) . To see that, employ Lemma E.1 with diam |·| ( Y ) = 1 and d 2 dy 2 ϕ ( y ) ≤ 1 σ ( -1) σ (1) on the range of G .
3. Similarly, because the range of functions in G lies in [ σ ( -1) , σ (1)] , the loss is bounded by ℓ ≤ B = -log( σ ( -1)) .

Hence, we may apply Theorem 1. Standard bounds on the Rademacher complexities yield

<!-- formula-not-decoded -->

This can be proven using Lipschitz contraction with respect to the sigmoid (which is 1 / 4 -Lipschitz continuous). For both bounds, there exist distributions so that the bound is tight. Plugging this into Theorem 1 yields (for some fixed high probability, such as 0 . 99 )

<!-- formula-not-decoded -->

Proof of Corollary C.2. Denote Σ k = E [ X k ( X k ) ⊤ ] , s ≡ s lin λ and g w = ⟨· , w ⟩ ∈ G , so that for any w,w ′ ∈ R d where the last inequality holds if max { log p N k , log K N k , log d n k , log K n k } ≤ 1 , which we assumed.

<!-- formula-not-decoded -->

and by an identical argument d k ( g w ; g w ′ ) = ( w -w ′ ) ⊤ Σ k ( w -w ′ ) = ∥ g w -g w ′ ∥ 2 k . It follows that

<!-- formula-not-decoded -->

where we defined the minimizers

<!-- formula-not-decoded -->

This holds because the unconstrained minimizer coincides with the constrained one, ensured by the bounded norms ∥ w ⋆ k ∥ 2 ≤ R ≤ κ and bounded smallest eigenvalue µ min (Σ k ) ≥ κ -note that because ∥ ∥ X k ∥ 2 ≤ 1 we have that µ max (Σ k ) ≤ 1 -which implies

We verify the assumptions of Theorem 2: Eq. (3) holds by definition of the data generating model. Assumption 1 and the smoothness from Assumption 2 hold, because ℓ is 4 -Lipschitz and 4 -bounded on Y = [ -1 , 1] , and induced by ϕ ( y ) = y 2 which is 1 -strongly convex, and max { ϕ ′′ , ϕ ′′′ } ≤ 2 . The convexity of d s ( g ; h ) -∥ g ∥ 2 s in Assumption 2 holds with constants γ s = 1 by inspecting Eq. (21), and G and H are clearly convex.

<!-- formula-not-decoded -->

We now bound the critical radius r R := inf { r ≥ 0 : r 2 ≥ R k n ( F R ( r )) } of the following function class F R ( r ) := { ⟨· , w ⟩ : ∥ w ∥ 2 ≤ 2 R,w ⊤ Σ k w ≤ r 2 } . Note that because µ max (Σ -1 k ) = 1 /µ min (Σ k ) ≤ 1 /κ and X k i ∈ B d 2 , it holds that and thus we get by Jensen's inequality that for any R,r ≥ 0

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and, by standard Rademacher complexity bounds (applying Cauchy-Schwartz and Jensen's inequality),

Hence, we can solve r 2 ≥ r/ √ κn and r 2 ≥ 2 R/ √ n to get, taking the minimum of the two,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We are now ready to apply Theorem 2. Noting that and that the set G k ( r ; f ⋆ ) is included in F 1 ( r ) ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging this into Theorem 2 yields the result.

## D Proofs of main results

## D.1 Proof of Proposition 1

To prove a sample complexity lower bound, we show a reduction from a statistical estimation problem to the semi-supervised multi-objective binary classification problem.

We start by constructing a statistical estimation problem, defining a family of distributions parametrized by the set of Boolean vectors σ ∈ { 0 , 1 } d . We aim to use samples from a distribution to estimate its associated parameter; the distributions will be designed so that any estimator given insufficiently many samples will fail to estimate the underlying parameter well for some σ . Then, we show that any learner that solves the multi-objective learning problem ( S -MOL) with ε s = ε and δ ≥ 5 / 6 can be used to solve the parametric estimation problem, implying a sample complexity lower bound for S -MOL. For convenience, we reproduce the PAC version of S -MOL here:

where P is the set of all distributions over X × Y .

<!-- formula-not-decoded -->

Let's consider the K = 2 case first. We show that n ≥ d/ 1024 ε 2 samples are necessary.

Defining the statistical estimation problem. Let X 0 := { x 1 , . . . , x d } ⊂ X be a set shattered by G . For each σ ∈ { 0 , 1 } d , define the distributions P 1 σ and P 2 σ over X × Y where (i) the marginal distributions on X is uniform over the shattered set { x 1 , . . . , x d } , and (ii) their conditional distributions on Y = { 0 , 1 } given x i are Bernoulli distributions. Let c = 4 ε and define:

<!-- formula-not-decoded -->

Fix a sample size n ∈ N . Define the family Q = {Q σ : σ ∈ { 0 , 1 } d } , where:

<!-- formula-not-decoded -->

Let Z σ ∼ Q σ consist of n i.i.d. draws from P 1 σ and P 2 σ each. The statistical estimation problem will be to construct an estimator ̂ σ ( Z σ ) for σ that recovers at least 3 / 4 of the coordinates of σ :

Reduction to multi-objective learning. Suppose that a learner can solve the S -MOL problem (22) for K = 2 using at most n samples. The reduction from estimating σ is as follows:

<!-- formula-not-decoded -->

1. Given any instance σ ∈ { 0 , 1 } d of the above statistical estimation problem, have the learner solve the MOL problem over ( P 1 σ , P 2 σ ) and linear scalarization S lin using data Z σ ∼ Q σ and zero-one loss.
2. Query the learner for the solution to the linear scalarization s 1 / 2 ≡ s lin λ with weights λ = ( 1 2 , 1 2 ) . Denote this solution by ̂ g s 1 / 2 ( · ; Z σ ) ∈ G and construct the estimator ̂ σ MOL for σ as:

Correctness of reduction. Before proving correctness, we make a few observations:

<!-- formula-not-decoded -->

1. For P 1 σ , the conditional label distribution associated to x i ∈ X 0 is either biased toward 1 or uniform over { 0 , 1 } . In either case, under the zero-one loss, the label 1 is Bayes optimal, and so the constant function f ⋆ 1 , σ ≡ 1 is a Bayes-optimal classifier for P 1 σ . Likewise, f ⋆ 2 , σ ≡ 0 is Bayes optimal for P 2 σ .
2. A function g : X → Y only incurs excess risk from an instance x i drawn from P 1 σ when σ i = 1 and g ( x i ) = 0 . Similarly, it accumulates excess risk from instances x i from P 2 σ when σ i = 0 and g ( x i ) = 1 . The total excess risks of g is given by:

<!-- formula-not-decoded -->

For the linear scalarization s 1 / 2 , we have:

<!-- formula-not-decoded -->

̸

3. Since G shatters X 0 , it contains a function g σ that satisfies g σ ( x i ) = σ i for all i ∈ [ d ] . Thus:

<!-- formula-not-decoded -->

4. Given g : X → Y , define the Boolean vector σ g ∈ { 0 , 1 } d by σ g,i = g ( x i ) . Then, by our choice of c , the excess s 1 / 2 -trade-off of g is related to the Hamming distance between σ g and σ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This last point implies the correctness of the reduction. That is, if a learner can solve (22), then we can use it to construct σ MOL that achieves (23). In particular, (24) shows that:

We now show that this statistical estimation problem requires at least n ≥ / 1024 ε 2 samples across P 1 σ or P 2 σ . This holds for any estimator including those that knows that f ⋆ 1 , σ and f ⋆ 2 , σ are Bayes-optimal classifiers and that the marginal distribution over X for both P 1 σ and P 2 σ are uniform over the shattered set. In particular, the lower bound applies to the semi-supervised MOL learner, which is given access to these Bayes-optimal classifiers and marginal distributions over X .

<!-- formula-not-decoded -->

Minimax lower bound. We now show that if n &lt; d/ 1024 ε 2 , the following bound holds:

where ̂ σ : ( X × Y × X × Y ) n →{ 0 , 1 } d ranges over all estimators using n samples from P 1 σ and P 2 σ each.

<!-- formula-not-decoded -->

For every ̂ σ and σ it follows from Markov's inequality that

We lower bound max σ E [ ∥ σ ( Z σ ) -σ ∥ 1 ] for any estimator σ using Assouad's lemma:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ ̂ Lemma D.1 (Assouad's lemma, [68]) . Let d ≥ 1 be an integer and let Q = { Q σ : σ ∈ { 0 , 1 } d } contain 2 d probability measures. Given σ , σ ′ ∈ { 0 , 1 } d , write σ ∼ σ ′ if they differ only in one coordinate. Let σ be any estimator. Then where KL( ·∥· ) measures the Kullback-Leibler divergence between two distributions.

When σ and σ ′ differ only in one coordinate, the KL divergence between Q σ and Q σ ′ is bounded:

<!-- formula-not-decoded -->

where the last inequality holds when c = 4 ε ≤ 1 / 3 by Lemma D.2. Indeed, we've assumed ε &lt; 1 / 12 .

By Assouad's lemma (Lemma D.1) and the above bound on the KL divergence, in the worst-case setting, any algorithm using n samples will have expected error at least

Plugging into Equation (25), we finally obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Generalization to all K &gt; 1 . The MOL problem with K tasks is at least as hard as M = ⌊ K/ 2 ⌋ separate MOL problems each with two tasks. This leads to a total sample complexity lower bound M · d/ 1024 ε 2 . We obtain the lower bound CKd/ε 2 in the statement of the result by setting C = 1 / 3072 and using Lemma D.3, which shows that ⌊ K/ 2 ⌋ ≥ K/ 3 for all K &gt; 1 .

where the last inequality holds whenever √ 4 nc 2 /d ≤ 1 / 4 , which holds when n &lt; d/ 1024 ε 2 .

More explicitly, we can reduce M separate copies of the statistical estimation problem for σ 1 , . . . , σ M into a single MOL problem over the distributions:

<!-- formula-not-decoded -->

where k = 1 , . . . , M . Define s k 1 / 2 to be the linear scalarization that equally divides all weight across the 2 k -1 and 2 k components:

Then, an estimator ̂ σ k for σ k can be obtained from the by defining as before: ̂ σ k,i = ̂ g s k 1 / 2 ( x i ) . The analysis from the K = 2 setting now holds for each k = 1 , . . . , M . This implies that at least d/ 1024 ε 2 samples must be drawn across each pair of the 2 k -1 and 2 k th distributions. This concludes the proof.

<!-- formula-not-decoded -->

Lemma D.2 (KL-divergence bound, e.g. [16]) . Let x ∈ ( -1 / 3 , 1 / 3) . Then:

Proof. By a direct computation, we have that whenever 4 x 2 ≤ 1 / 2 , which is satisfied when x ∈ [ -1 / 3 , 1 / 3] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality holds from the fact that 1 2 log 1 1 -z ≤ z whenever z ∈ [0 , 1 / 2] .

For the second inequality, we show that the function ϕ ( x ) = KL(1 / 2 + x ∥ 1 / 2) is L -smooth on ( -1 / 3 , 1 / 3) where L ≤ 8 and has zero derivative at x = 0 . This implies that it is upper bounded by L 2 x 2 . In particular, the first and second derivatives are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that ϕ ′′ ≤ 8 whenever x 2 ≤ 1 / 9 .

Lemma D.3. Let K &gt; 1 be a natural number. Then, ⌊ K/ 2 ⌋ ≥ K/ 3 .

Proof. There are two cases:

- When K is even, then ⌊ K/ 2 ⌋ = K/ 2 ≥ K/ 3 .
- When K is odd, then ⌊ K/ 2 ⌋ = ( K -1) / 2 ≥ K/ 3 , where the last inequality is equivalent to 3( K -1) ≥ 2 K , which is further equivalent to K ≥ 3 .

## D.2 Proof of Lemma 1

Let ℓ k be a Bregman loss associated with the potential ϕ k . The first part is proven in [7, Theorem 1]: for any Y k such that E [ Y k ] and E [ ϕ k ( Y k )] are finite, it holds that

Then, by definition of Bregman divergences, we have the following generalized Pythagorean identity [8, Equation (26)]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that by the tower property (see also [7, Equation (1)])

<!-- formula-not-decoded -->

Rearranging yields that E k ( f ) = R k ( f ) -R k ( f ⋆ k ) = d k ( f ; f ⋆ k ) , which is the second claim.

## D.3 Proof of Theorem 1

The proof of Theorem 1 relies on the following lemma on estimating the excess risk functionals E k with the risk discrepancies d k ( f ; h k ) under Assumption 1.

<!-- formula-not-decoded -->

̂ Lemma D.4 (Excess risk functional estimation) . Suppose that Assumption 1 holds and that a function ̂ h k achieves excess risk E k ( ̂ h k ) . Let c k = L k √ 2 /µ k . Then, the risk discrepancy functional d k ( · ; ̂ h k ) defined in Equation (7) approximates E k ( · ) uniformly on F all , that is,

Proof. Recall that f ⋆ k ∈ H k is the minimizer of E k over F all . Then for all f ∈ F all :

which is the claim.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem 1. For k ∈ [ K ] , let ̂ h k be the empirical risk minimizer obtained in Line 2 of Algorithm 1 for the k th objective. Let us recall that we use the empirical risk discrepancy ̂ d k ( · ; ̂ h k ) as an estimate for the excess risk E k ( · ) = d k ( · ; f ⋆ k ) , following the properties of Bregman losses in Lemma 1. We now prove the theorem assuming that the following claim holds.

Claim. With probability at least 1 -δ , each estimate ̂ d k ( · ; ̂ h k ) approximates the population excess risk functional E k up to error ε k / 2 :

where ε k is bounded as in Eq. (10).

<!-- formula-not-decoded -->

Then, for any scalarization s that satisfies the reverse triangle inequality, the s -trade-off T s is also well-approximated by empirical scalarized discrepancy d s ( · ; h ) . In particular, we obtain where the first inequality used the reverse triangle inequality of s , and the second inequality used the monotonicity of s and Eq. (26). In particular, this allows us to bound the excess s -trade-off of ̂ g s , the minimizer of the empirical scalarized discrepancy in G obtained in Line 5 of Algorithm 1, as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the (a) and (c) terms both contribute at most s ( ε 1 / 2 , . . . , ε k / 2) error from Eq. (27), while the (b) term is non-positive, since ̂ g s minimizes the empirical scalarized discrepancy ̂ d s ( · ; ̂ h ) . The last equality follows from positive homogeneity of the scalarization. Then, the result follows for all such scalarizations simultaneously . It remains to prove Eq. (26).

Proof of claim. Recall that ̂ d k ( g ; ̂ h k ) is an empirical estimator of d k ( g ; ̂ h k ) based on the unlabeled samples:

where these were defined in Eq. (7). Moreover, d k ( g ; ̂ h k ) itself is an estimator of the excess risk functional E k ( g ) = d k ( g ; f ⋆ k ) , where f ⋆ k is the Bayes optimal regression function (Lemma 1). Thus, we have the decomposition:

<!-- formula-not-decoded -->

We can bound T a,k and T b,k separately:

<!-- formula-not-decoded -->

(a) For each k ∈ [ K ] , we condition on the labeled samples (i.e., on ̂ h k ) and employ a standard Rademacher bound on the function class:

With probability at least 1 -δ/ (2 K ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality applies symmetrization (Lemma E.5) and the bounded difference inequality (Lemma E.2), and the second inequality follows by contraction (Lemma E.4).

(b) For each k ∈ [ K ] , we apply Lemma D.4 to bound T b,k in terms of the excess risk of ̂ h k , which is a minimizer of the empirical risk R k ( · ) defined in Eq. (2):

In order to use the lemma, we need to show that the excess risk of ̂ h k is indeed upper bounded by ε k . First, observe that the excess risk can be upper bounded as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Again by symmetrization (Lemma E.5), bounded difference (Lemma E.2), and contraction (Lemma E.4) for the function class ℓ k ◦ H k := { ( x, y ) ↦→ ℓ k ( y, h ( x )) : h ∈ H k } , we obtain that with probability at least 1 -δ/ (2 K ) ,

And so, by Lemma D.4, we obtain that with probability at least 1 -δ/ (2 K ) , where c k = L k √ 2 /µ k .

<!-- formula-not-decoded -->

The claim in Eq. (26) follows by a union bound. By combining Equations (28) to (30), we obtain that with probability at least 1 -δ , for all k ∈ [ K ] :

where we can set ε k as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof of Theorem 1.

## D.4 Proof of Theorem 2

In this section, we provide the proof of Theorem 2, but leave some of the technical details to auxiliary results that we prove after the main proof. See also Fig. 5 for a visualization of the proof.

## D.4.1 Preliminaries

Recall that s is a linear scalarization. We begin by introducing the notation µ k = P k X as well as the mixture distribution µ s = s ( µ 1 , . . . , µ K ) . Recall the definitions of the (semi-)Hilbert norms

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which correspond to the inner products (denoting ⟨· , ·⟩ the inner product on R q )

We first verify that these are indeed (semi-)Hilbert norms and inner products.

Lemma D.5. The functions ⟨· , ·⟩ k , ⟨· , ·⟩ s and ∥·∥ k , ∥·∥ s defined above are the inner products and norms of the L 2 ( µ k ) and L 2 ( µ s ) -Bochner spaces of functions X → ( R q , ∥·∥ 2 ) .

See [30, Definition 1.2.15] for a definition. We prove Lemma D.5 in Appendix D.4.3 and use it throughout without explicitly referring to it. Note that we have implicitly assumed that F all ⊆ ⋂ k ∈ [ K ] L 2 ( µ k ) . In order to use first-order calculus throughout the proof, we derive the gradient and smoothness of the map g ↦→ d s ( g ; h ) below. We also prove Lemma D.6 in Appendix D.4.3.

Lemma D.6 (Gradients and smoothness) . For any h ∈ H 1 ×··· × H K and linear scalarization s , denote by ∇ g d s ( g ; h ) : X → R the gradient of the map g ↦→ d s ( g ; h ) induced by the Fréchet derivatives on L 2 ( µ s ) and the inner product ⟨· , ·⟩ s . Then it holds that 5

<!-- formula-not-decoded -->

Moreover, if Assumption 2 holds and we denote D = diam ∥·∥ 2 ( Y ) , then the map g ↦→ d s ( g ; h ) is C sm := ν (1 + D ) -smooth in ∥·∥ s , that is, the gradient from above is C sm -Lipschitz continuous in g with respect to ∥·∥ s . If additionally Assumption 3 holds, then for g s = arg min g ∈G d s ( g ; f ⋆ ) and all g ∈ G

<!-- formula-not-decoded -->

5 Note that whenever λ k &gt; 0 , the Radon-Nikodym derivative dµ k /dµ s is well-defined.

gs

˜

G

<!-- image -->

̂

Figure 5: Informal visualization of the proof of Theorem 2. We first localize the set of estimators { ̂ g s : s ∈ S} (dotted red line) around the random 'helper' set { g ′ s : s ∈ S} (dashed blue line) within the set G k ( ˜ u k ; ̂ h ) . We then expand a set G k ( u k , f ⋆ ) centered at the 'true' set { g s : s ∈ S} (solid green line) to include the set G k ( ˜ u k ; ̂ h ) ⊂ G k ( u k + l k ; f ⋆ ) where l k bounds the maximal deviation of g s to g ′ s . This way, we may bound the critical random critical radius u of G k ( ˜ u k , ̂ h ) in terms of the deterministic critical radius u of G k ( u ; f ⋆ ) and l k as ˜ u 2 k ≲ u 2 k + l 2 k .

Lemma 2 is a direct consequence of the gradient characterization in Lemma D.6 together with Theorem 46 in [69]. Finally, we show that if Assumption 4 holds, the norms ∥·∥ k and ∥·∥ s are equivalent.

Lemma D.7. Let S ⊂ S lin be in the set of linear scalarizations (5) . Then, for any η ∈ [0 , ∞ )

<!-- formula-not-decoded -->

We also prove Lemma D.7 in Appendix D.4.3.

## D.4.2 Main proof of Theorem 2

Recall the empirical and population minimizers of the corresponding risk discrepancies from Eq. (7)

<!-- formula-not-decoded -->

̂ ̂ The basic decomposition of our proof is a triangle inequality with a helper set of minimizers of the population risk discrepancy, defined with respect to pseudo-labeled data as

Our goal is to bound T s ( ̂ g s ) -inf g ∈G T s ( g ) simultaneously for all s ∈ S . By Lemma 1, we have T s ( g s ) -inf g ∈G T s ( g ) = d s ( g s ; f ⋆ ) -d s ( g s ; f ⋆ ) so that we focus on bounding this expression.

<!-- formula-not-decoded -->

Specifically, by the smoothness from Lemma D.6, we can bound the excess trade-off as

<!-- formula-not-decoded -->

Here T lab s quantifies the error from having a finite amount of labeled data to estimate f ⋆ k with ̂ h k and how that error propagates to g ′ s , and T un s quantifies how close to g ′ s we can get with the finite amount of unlabeled data. Our goal will be to bound the terms T lab s and T un s using localization, simultaneously for all s ∈ S . For the general proof technique of localization, we take inspiration from the approaches outlined in [63, 56, 35, 9, 11, 36].

We proceed in three main steps. See also Fig. 5.

1. To bound T lab s , we first use standard localization bounds for the ERMs in each task separately, using uniform bound on the local sets H k ( r ) = ( H k -f ⋆ k ) ∩ r B ∥·∥ k from Eq. (11). We then show how their errors translate to g ′ s through a deterministic stability argument.

h

2

̂

2. To bound T un s , we condition on ̂ h and simultaneously localize around the (random) functions g ′ s for all s ∈ S , resulting in a uniform learning bound on local sets

that are 'centered' at the helper set { g ′ s : s ∈ S} .

<!-- formula-not-decoded -->

3. The resulting bound on T un s from the previous step is random, because g ′ s depends on ̂ h , so we need to further bound it. We prove two ways of doing that, so that the bound takes the minimum of the two: the critical radius of G k ( r, f ⋆ ) = r B ∥·∥ k ∩ ⋃ s ∈S ( G g s ) from Eq. (11) together with the bound on T lab s , or a worst-case bound taking the supremum over such { g ′ s : s ∈ S} .

See also Fig. 5 for a visualization of the corresponding sets.

Throughout, we heavily use the following monotonicity property of the Rademacher complexity, analogous to the usual localization proofs. The proof can be found in Appendix D.4.4.

Lemma D.8. Consider the sets from Eqs. (11) and (33) . Under Assumption 2, the functions

<!-- formula-not-decoded -->

are non-increasing on (0 , ∞ ) for all h ∈ H 1 ×··· × H K .

Step 1: Localization for ERMs in H k and bounding T lab s . In this step, we first bound the error of learning f ⋆ k with the ERMs ̂ h k (or, in fact, any other estimator that satisfies the basic inequality ̂ R k ( ̂ h k ) ≤ ̂ R k ( f ⋆ k ) ). Recall the definition H k ( r ) = ( H k -f ⋆ k ) ∩ r B ∥·∥ k from Eq. (11), and the corresponding critical radii l k = inf { r ≥ 0 : r 2 ≥ R k n k ( H k ( r )) } . Using the non-increasing property from Lemma D.8, we can summarize the bound in the following Lemma.

<!-- formula-not-decoded -->

Lemma D.9. Under Assumptions 1 and 2, and if δ &gt; 0 is sufficiently small, we have that P ( E lab δ ) ≥ 1 -δ , where we define the event

The proof of Lemma D.9 can be found in Appendix D.4.5, and it essentially follows the localization technique from [9]: We bound the suprema of the empirical process over H k ( r ) using Talagrand's inequality (Lemma E.3) in terms of the Rademacher complexity and variance. Using Lemma D.8 and a peeling argument, we get the bound in terms of the critical radius.

Next, we show that the bound from Eq. (35) directly translates into a bound on the helper set { g ′ s : s ∈ S} with respect to labels from ̂ h but known covariate distributions. To do so, we prove the following stability result. Effectively, it removes the square-root from Lemma D.4 that appears in Theorem 1.

Proposition D.1 (Quadratic stability of minimizers) . Denote g h s = arg min g ∈G d s ( g ; h ) and C st := ν 2 / 4 γ 2 . Under Assumptions 1 and 2, we have for any h , h ′ ∈ H 1 ×··· × H K , any s = s lin λ ∈ S , that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We prove Proposition D.1 in Appendix D.4.6. Note that a linear bound would directly follow from Lipschitz continuity of the losses. However, this would yield much slower statistical rates than the stability argument from Proposition D.1. Recalling the definition of ζ k from (35), we can now use Proposition D.1 with g ̂ h s = g ′ s , g f ⋆ s = g s , to conclude that on E lab δ , it holds that for all s ∈ S ,

Eq. (36) describes how well our estimators would approximate the true set { g s : s ∈ S} if we had an infinite amount of unlabeled data. In that sense, this can be seen as an intermediate result in the ideal semi-supervised setting by combining Eqs. (32) and (36).

Step 2: Localization along helper Pareto set in G to bound T un s . Wenowneed to take into account the finite sample effect of having only N k unlabeled samples to estimate the risk discrepancies.

To perform localization around the helper set, we again rely on Talagrand's concentration inequality (Lemma E.3). The benefit of Talagrand's inequality in standard localization usually comes from the fact that it accounts for the variance of the losses when centered at the ground truth , which can usually be controlled by its radius of the local function class. We also used this in Step 1. Now, however, we need to simultaneously localize for all scalarizations s ∈ S . Hence, recall G k ( r ; ̂ h ) from Eq. (33) where, intuitively, r uniformly controls the deviations ̂ g s -g ′ s for all s ∈ S . To keep track of which g ′ s any g ∈ G k ( r ; h ) 'belongs to', we also define the set

Lifting the set G k ( r ; ̂ h ) to S × G is inspired by a similar trick from multi-objective optimization, where the Pareto set is often lifted to this larger space to obtain its manifold structure, cf. [53].

<!-- formula-not-decoded -->

We apply Talagrand's concentration inequality on M k ( r ) and use a localization argument, summarized in the following lemma. Define the radii ˜ u k = inf { r ≥ 0 : r 2 ≥ R k N k ( G k ( r ; ̂ h )) } , and note that these radii are deterministic with respect to the unlabeled data, but are random with respect to the labeled data through the ERMs, a point revisited in the next section. Recall that ̂ g s is the minimizer of ̂ d s ( g ; ̂ h ) and g ′ s is the minimizer of d s ( g ; ̂ h ) (Eq. (7)). The next proposition bounds T un s = ∥ ̂ g s -g ′ s ∥ 2 s (or, in fact, the deviation of any estimator satisfying the basic inequality d s ( g s ; h ) ≤ d s ( g ′ s ; h ) ).

<!-- formula-not-decoded -->

̂ ̂ ̂ ̂ ̂ Proposition D.2 (Localization along helper Pareto set) . Under Assumptions 1 and 2, and for sufficiently small δ &gt; 0 , we have that P ( E un δ ) ≥ 1 -δ , where we define the event

The proof of Proposition D.2 can be found in Appendix D.4.7.

Step 3: Bounding the random critical radii ˜ u k . Recall that ˜ u k is deterministic with respect to the unlabeled data, but random with respect to the labeled data. To make the bound fully deterministic, we prove two bounds, so that their minimum appears in Theorem 2.

Option 1 is taking the trivial approach: recall from Eq. (11) that we define for the function g h s = arg min g ∈G d s ( g ; h ) the set

<!-- formula-not-decoded -->

Then the following deterministic worst-case localized radii bounds ˜ u 2 k ≤ ¯ u 2 k (and u 2 k ≤ ¯ u 2 k ) deterministically (i.e., also almost surely). Option 2 is more nuanced: If Assumption 4 holds, we can combine Eq. (36) with an expansion argument to bound ˜ u k in terms of the l k and the u k . To relate them, we employ the following key proposition.

<!-- formula-not-decoded -->

Proposition D.3 (Critical radius shift) . Let G be any class of functions that is convex (Assumption 2), and let n ∈ N . Let ∥·∥ be any norm on F all and let B = { f ∈ F all : ∥ f ∥ ≤ 1 } be its unit ball. Define as well as the critical radii (for R n defined w.r.t. an arbitrary distribution)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ Let ∆ = sup s ∈S ∥ g s -g ′ s ∥ . Then it holds that ˜ u ≤ 5( u +∆) .

The proof of Proposition D.3 can be found in Appendix D.4.8. We can apply Proposition D.3 to our setting: recall the definitions of G ′ k ( r ) from Eq. (33) and G k ( r ) from Eq. (11), and the definitions ˜ u k = inf { r ≥ 0 : r 2 ≥ G k ( r ; ̂ h ) } and u k = inf { r ≥ 0 : r 2 ≥ R k N k ( G k ( r ; f ⋆ )) } . From Assumption 4, Lemma D.7, and Eq. (36), we know that on E lab δ from Eq. (35), for ζ 2 S = sup s ∈S s ( ζ 2 1 , . . . , ζ 2 K )

Employing Proposition D.3 with this ∆ yields ˜ u 2 k ≲ u 2 k + η 2 C st · ζ 2 S . We define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that in general, either bound can be tighter. For practical purposes, it may be easier to bound ¯ u k anyways, so the detour through u k may be unnecessary.

Putting everything together. From Eq. (32), we see that on E lab δ/ 2 ∩ E un δ/ 2 , which holds with probability at least 1 -δ by union bound, for all s ∈ S , the excess s -trade-off T s ( ̂ g s ) -inf g ∈G T s ( g ) is bounded by C

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ≲ only hides universal constants. From the two options of bounding ˜ u k we obtain: 1. The first bound, valid without Assumption 4: Recalling C sm = ν (1 + D ) , C st = ν 2 4 γ 2

<!-- formula-not-decoded -->

2. The second bound, valid under Assumption 4, by plugging in Eq. (40) and C add from Eq. (39)

That concludes the proof of Theorem 2, with the proofs of the auxiliary results presented next.

<!-- formula-not-decoded -->

## D.4.3 Proof of preliminary lemmata

Proof of Lemma D.5. The claim for ⟨· , ·⟩ k , ∥·∥ k , k ∈ [ K ] is true by definition, but also as a special case of the scalarized form: for any s = s lin λ ∈ S lin and f, f ′ ∈ L 2 ( µ s ) we have where ⟨· , ·⟩ is the Euclidean inner product. This is exactly the inner product of the Bochner L 2 ( µ s ) space (e.g., [30]). Further, plugging in f ′ = f we obtain directly that ⟨ f, f ⟩ s = ∥ f ∥ 2 s , verifying that the norm is induced by this inner product.

<!-- formula-not-decoded -->

Proof of Lemma D.6. Recall that ⟨· , ·⟩ s denotes the inner product of the norm ∥·∥ 2 s = ∑ K k =1 λ k ∥·∥ 2 k . In this proof, we use Fréchet derivatives (denoted D ) and the corresponding gradients ∇ g induced by the inner product ⟨· , ·⟩ s . Background on Fréchet derivatives can be found in [2, 12]. From Lemma D.5 we know that ∥·∥ s actually is the (semi-)Hilbert norm that corresponds to the Bochner L 2 ( µ s ) space with respect to the space ( R q , ∥·∥ 2 ) , where recall that µ s denotes the mixture distribution

<!-- formula-not-decoded -->

It is then easily shown that the gradient of ∑ K k =1 λ k E [ Q k ( g ( X k )) ] for any differentiable functions Q k : R q ⊃ Y → R with sup y ∈Y ∥∇ Q ( y ) ∥ 2 ≤ M &lt; ∞ , induced by ⟨· , ·⟩ s , is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Indeed, for any f ∈ F all ⊂ L 2 ( µ s ) , we can write the Fréchet derivative as the limit where we could use dominated convergence thanks to sup y ∈Y ∥∇ Q ( y ) ∥ 2 ≤ M &lt; ∞ . This implies the claimed form of the gradient.

Since ℓ k is Lipschitz and differentiable, its gradient in g is bounded. Further,

<!-- formula-not-decoded -->

and the gradient of a Bregman divergence in its second argument is given by

<!-- formula-not-decoded -->

so that the previous derivations imply for the Bregman losses that

<!-- formula-not-decoded -->

which is the first claim of the lemma.

<!-- formula-not-decoded -->

Note that µ s -almost surely ∑ K k =1 λ k dµ k dµ s = 1 . Hence, for every fixed h , and g, g ′ ,

To bound the first term, we use from Assumption 2 that the ℓ 2 -operator norm of ∇ 2 ϕ ( g ( x )) is bounded by ν &gt; 0 , so that

To bound the second term, we use that ∥ h k -g ′ ∥ 2 ≤ diam ∥·∥ 2 ( Y ) =: D , and so

<!-- formula-not-decoded -->

which together with the smoothness from Assumption 2 implies

<!-- formula-not-decoded -->

Plugging both into Eq. (41) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by equivalent characterizations of smoothness (e.g., [12, Corollary 18.14]) it follows that

<!-- formula-not-decoded -->

By Assumption 3, we know that the global minimizer of f ↦→ d s ( f ; f ⋆ ) is contained in G . By definition, therefore g s coincides with it, and hence for g ′ = g s and h = f ⋆ , we know that ⟨∇ d s ( g s ; f ⋆ ) , g -g s ⟩ s = 0 and obtain the bound

<!-- formula-not-decoded -->

This concludes the proof.

Proof of Lemma D.7. Denote µ k = P k X and µ s = ∑ k λ k µ k . Recall the definition of the essential supremum of a function f : X → R (with respect to µ s ):

We start with ' ⇐ ': Since µ s ( { x ∈ X : dµ k /dµ s ( x ) ≥ η 2 } ) = 0 , for any k ∈ [ K ] , s ∈ S and f ∈ F all ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Now we show ' ⇒ ': Choose an arbitrary y = 0 ∈ Y and measurable A ⊂ X , and let f = ( y/ ∥ y ∥ 2 )1 A . Note that ∥ f ∥ 2 2 = 1 A . Then for all s = s lin λ ∈ S

This implies the bound α := ess sup dµ k /dµ s ≤ η 2 , since for any ε &gt; 0 we can choose the measurable event A ε := { x : dµ k /dµ s ( x ) ≥ α -ε } which satisfies (by definition) µ s ( A ε ) &gt; 0 and so

Taking ε → 0 concludes the proof.

<!-- formula-not-decoded -->

## D.4.4 Proof of Lemma D.8

For the first function, the argument is standard, we repeat it here for completeness. Let 0 &lt; r &lt; r ′ and consider some h ∈ H k ( r ′ ) . Then ∥ h ∥ k ≤ r ′ and hence ∥ ( r/r ′ ) h ∥ k ≤ r , so that ( r/r ′ ) h ∈ H k ( r ) by the star-shape of H k from Assumption 2. Therefore, we have that which is the claim.

<!-- formula-not-decoded -->

For the other function the proof is identical once we realize that the convexity of G from Assumption 2 implies that G k ( r ; h ) is star-shaped around the origin. Indeed, for any h ∈ H 1 × · · · × H K and g h s = arg min g ∈G d s ( g ; h ) , since ( G g h s ) ∩ r B k is convex and contains the origin,

<!-- formula-not-decoded -->

We require this star-shapedness for all h ∈ H 1 ×···×H K , because we also localize around g ′ s = g ̂ h s that are random elements and may be anywhere in G .

## D.4.5 Proof of Lemma D.9

The proof of this Lemma is a mixture of Corollary 5.3 in [9] and Theorem 14.20 in [63]; see also [35] for an exposition. We repeat it here for completeness and because we make slightly different assumptions from [9, 63], see Remark 1 below. Recall the definition of the sets for any r ≥ 0 ,

<!-- formula-not-decoded -->

and the random variables which are the suprema of empirical processes indexed by the function classes defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Assumption 1, these function classes are uniformly bounded by B k ≥ 0 . Hence, by Talagrand's concentration inequality (Lemma E.3), for any choice of deterministic radii r 1 , . . . , r K ≥ 0 , the event holds with probability at least 1 -δ . Here τ 2 k ( r ) is a short-hand for the variance proxy from Lemma E.3, defined as

<!-- formula-not-decoded -->

We now bound E [ T k ( r k )] and τ 2 k ( r k ) . Using symmetrization (Lemma E.5) and vector contraction (Lemma E.4), recalling that ℓ k is L k -Lipschitz w.r.t. the ℓ 2 -norm in its second argument, we can bound

<!-- formula-not-decoded -->

Therefore, we get on the event Q lab δ ( r 1 , . . . , r K ) that for all k ∈ [ K ]

<!-- formula-not-decoded -->

Now recall the definition

By (34), we get that for any r ≥ l k

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and therefore, if r k ≥ l k for all k , on the event Q lab δ ( r 1 , . . . , r K ) (which holds with probability at least 1 -δ ), it holds that

<!-- formula-not-decoded -->

We now choose r k := ∥ ∥ ∥ ̂ h k -f ⋆ k ∥ ∥ ∥ k , which are random radii, so we have to perform a peeling argument. Define the event

<!-- formula-not-decoded -->

Because ∥ h -f ⋆ k ∥ k ≤ diam ∥·∥ 2 ( Y ) =: D , we know that for any M satisfying 2 M l k ≥ D ⇐⇒ M ≥ log( D/ l k ) / log(2) , for any ∥ h -f ⋆ k ∥ k ≥ l k there must be at least one 0 ≤ m ≤ M so that 2 m -1 l k ≤ ∥ h -f ⋆ k ∥ k ≤ 2 m l k . Moreover, a calculation shows that the functions Φ k satisfy

<!-- formula-not-decoded -->

for sufficiently small δ , and so P ( W lab δ ) is bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in ( a ) we used that P ( Q lab δ/ 2 m (2 m l 1 , . . . , 2 m l K )) ≥ 1 -δ/ 2 m .

Now, by the standard risk decomposition, we have that and we can make a case distinction.

<!-- formula-not-decoded -->

Remark 1 . Many localization proofs for general loss functions only assume strong convexity and Lipschitz continuity (see, e.g., Section 14.3 in [63]), and therefore one needs to handle the case where the L 2 -radius is bounded but the excess loss is not (tightly) bounded, which would occur in the first case below. In our setting, by the smoothness (Lemma D.6), a bounded radius directly implies bounded excess risk, so this case cannot occur and no separate treatment is required.

Either r k = ∥ ∥ ∥ ̂ h k -f ⋆ k ∥ ∥ ∥ k ≤ l k and we are done, or r k = ∥ ∥ ∥ ̂ h k -f ⋆ k ∥ ∥ ∥ k &gt; l k , and so, because P ( W lab δ ) ≤ δ , we have with probability at least 1 -δ

∥ ∥ Recall that by Assumption 1, ϕ k is µ k -strongly convex w.r.t. ∥·∥ 2 , so that ℓ k ( y, y ′ ) ≥ µ k 2 ∥ y -y ′ ∥ 2 2 . Hence, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used Lemma 1 in the second equality. Solving r 2 k ≤ 12 µ k Φ k ( r k , δ ) for r k , we get that

Hence, in either case we have

<!-- formula-not-decoded -->

Therefore, because P ( W lab δ ) ≤ δ , we have that P ( E lab δ ) ≥ 1 -δ , where which concludes the proof for localization in H k .

<!-- formula-not-decoded -->

## D.4.6 Proof of Proposition D.1

Recall the form of the gradient ∇ g d s ( g ; h ) from Lemma D.7. For every fixed g , and any h , h ′ ,

<!-- formula-not-decoded -->

This is what we call 'cross-smoothness'.

Denote g = g h s and g ′ = g h ′ s . We may now use a generalization of the stability argument used in the proof of Theorem 1 in [65], where the following argument was used in R m and for unconstrained

optimization: By the convexity of G (Assumption 2), and the optimality of g, g ′ we get these two variational inequalities see Lemma 2 and [69, Theorem 46]. Combining both, and subtracting 〈 ∇ g d s ( g ; h ′ ) , g ′ -g 〉 on both sides we see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From the second item in Assumption 2, and the main results in [49], we get that the right-hand side of (43) is lower bounded as and from the cross-smoothness in (42), we get that the left-hand side of (43) is upper bounded by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the two, we can see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This is the claimed quadratic bound.

## D.4.7 Proof of Proposition D.2

Throughout this proof, condition on the ̂ h k . In particular, all expectations and variances are conditioned on ̂ h k . Recall from Eq. (33) that for any r ≥ 0

and from Eq. (37) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first part of this proof is mostly standard and follows the same proof structure as Lemma D.9. Define the random variables

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Assumption 1, these function classes are uniformly bounded by B k ≥ 0 . Hence, by Talagrand's concentration inequality (Lemma E.3), for any choice of deterministic radii r 1 , . . . , r K ≥ 0 , the event

holds with probability at least 1 -δ . Here σ 2 k ( r k ) is a short-hand for the variance proxy from Lemma E.3, defined in this section as

We now bound E [ Z k ( r )] and σ 2 k ( r ) . Using symmetrization (Lemma E.5) in addition to vector contraction (Lemma E.4), recalling that ℓ k is L k -Lipschitz w.r.t. ℓ 2 -norm in its second argument, we can bound

<!-- formula-not-decoded -->

Therefore, we get on the event Q un δ ( r 1 , . . . , r K ) that

Define

<!-- formula-not-decoded -->

By Lemma D.8, which holds under Assumption 2, we get that for any r ≥ ˜ u k

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ Therefore, if r k ≥ ˜ u k for all k , on the event Q un δ ( r 1 , . . . , r K ) (which holds with probability at least 1 -δ ), it holds that for all k ∈ [ K ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now come to the part of the proof that is less standard. Consider the family of random radii

<!-- formula-not-decoded -->

We perform a peeling argument to bound the probabilities of the two events

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ Remark 2 . Contrary to Remark 1, here we include the case where the radii are small, because we have to control all K radii simultaneously. One could also adapt the following proof without this case, but the resulting bound would be the same (up to constants).

By the previous derivations, P ( W un δ, 0 ) ≤ δ , and for W un δ, 1 we apply a peeling argument. Because ∥ g -g ′ s ∥ k ≤ diam ∥·∥ 2 ( Y ) =: D , we know that for any M satisfying for small enough δ , which yields that

2 M ˜ u k ≥ D ⇐⇒ M ≥ log( D/ ˜ u k ) / log(2) , and for any ∥ g -g ′ s ∥ k ≥ ˜ u k there must be at least one 0 ≤ m ≤ M so that 2 m -1 ˜ u k ≤ ∥ g -g ′ s ∥ k ≤ 2 m ˜ u k . Moreover, a calculation shows that the functions Ψ k satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in ( a ) we used that P ( Q un δ/ 2 m (2 m ˜ u 1 , . . . , 2 m ˜ u K )) ≥ 1 -δ/ 2 m . Combining the two with a union bound yields P ( ( W un δ, 0 ) c ∩ ( W un δ, 1 ) c ) ≥ 1 -2 δ . Condition on ( W un δ, 0 ) c ∩ ( W un δ, 1 ) c .

By the standard risk decomposition, we get for all s ∈ S

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, by the 'multi-objective Bernstein condition' ∥ ̂ g s -g ′ s ∥ 2 s ≤ 1 γ ( d s ( ̂ g s ; ̂ h ) -d s ( g s ; ̂ h )) , implied by the second item from Assumption 2, and Eq. (44),

On the event ( W un δ, 0 ) c ∩ ( W un δ, 1 ) c , we thus get for every s = s lin λ ∈ S

We can simplify the first term using ab ≤ 1 2 ( a 2 + b 2 ) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging this into the bound on r 2 s yields

<!-- formula-not-decoded -->

where the last inequality is Cauchy-Schwarz. Some algebra shows that r 2 s ≤ r s a s + b s implies r 2 s ≤ 2( a 2 s + b s ) , and so

<!-- formula-not-decoded -->

where in the last line we used L k /γ ≥ 1 . Therefore, because P ( ( W un δ, 0 ) c ∩ ( W un δ, 1 ) c ) ≥ 1 -2 δ , we have that P ( E un δ ) ≥ 1 -δ , where

<!-- formula-not-decoded -->

E un δ := { ∀ s = s lin λ ∈ S : ∥ ̂ g s -g ′ s ∥ 2 s ≲ K ∑ k =1 λ k ( L 2 k γ 2 ˜ u 2 k + ( L 2 k γ 2 + B k γ ) log(2 K/δ ) N k ) } , which concludes the proof of this part.

## D.4.8 Proof of Proposition D.3

Recall that ∆ = sup s ∈S ∥ g s -g ′ s ∥ . For every r ≥ 0 , we have the following key inclusion

To see that, let

h

=

g

g

′

s

(

r

)

. Then

h

+(

g

<!-- formula-not-decoded -->

′

s

g

s

) =

g

g

s

and

h

Because Rademacher complexity is sub-additive, we get that for all r ≥ 0

-

∈ G

-

-

∥

+(

g

<!-- formula-not-decoded -->

where in the last step we used that

<!-- formula-not-decoded -->

Using that for all r ≥ u we have R n ( G ( r )) ≤ r 2 , we get that and using that r ↦→ R n ( G ( r )) /r is non-increasing (by Assumption 2 and Lemma D.8), we get that

which together yields

Again, by the fact that r ↦→ R n ( G ′ ( r )) /r is non-increasing (Lemma D.8), we get that for all r ≥ u +∆

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, for the choice r = 5( u +∆)

<!-- formula-not-decoded -->

which implies that ˜ u ≤ 5( u +∆) , completing the proof.

′

′

s

-

g

s

)

∥ ≤

r

+∆

.

## E Auxiliary results

Lemma E.1 (Lipschitz continuity of Bregman divergences) . Assume diam ∥·∥ ( Y ) = sup y,y ∈Y ∥ y -y ′ ∥ &lt; ∞ and that ϕ is ν -smooth w.r.t. ∥·∥ and ∥∇ ϕ ( x ) -∇ ϕ ( z ) ∥ ∗ ≤ M ∥ x -z ∥ . Then D ϕ is Lipschitz continuous in both of its arguments separately, that is, for all x, y, z we have

<!-- formula-not-decoded -->

Proof. This follows from the three-point identity: First,

<!-- formula-not-decoded -->

and second, by the same argument,

<!-- formula-not-decoded -->

which concludes the proof.

## E.1 Concentration inequalities

Lemma E.2 (Consequence of McDiarmid's inequality [43]) . Let F be a function class of measurable functions X → R that is B -bounded, sup x ∈X | f ( x ) | ≤ B for all f ∈ F , and X,X 1 , . . . , X n be i.i.d. random elements in X . Define

Then it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof can be found, for instance, in [63, 56]. A significant improvement over Lemma E.2 is Talagrand's concentration inequality, stated next.

Lemma E.3 (Talagrand's concentration inequality [60]) . Let F be a countable function class of measurable functions X → R that is B/ 2 -bounded, sup x ∈X | f ( x ) | ≤ B/ 2 for all f ∈ F , and X 1 , . . . , X n be i.i.d. random elements in X . Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.2 Rademacher complexities

While there are multiple notions of Rademacher complexity for vector-valued functions [50], our choice of Rademacher complexity in this work is motivated by the following contraction inequality, which is used multiple times in our proofs.

Lemma E.4 (Vector contraction, Theorem 3 in [42] adapted for Rademacher complexities with absolute values) . Let H be a class of functions X → Y ⊂ R q . Assume that ℓ : Y × Y → R is L -Lipschitz continuous in its second argument with ℓ 2 -norm in R q , that is,

<!-- formula-not-decoded -->

Then it holds for ℓ ◦ H := { ( x, y ) ↦→ ℓ ( y, h ( x )) : h ∈ H} that

<!-- formula-not-decoded -->

where R n denotes the coordinate-wise Rademacher complexity.

This contraction inequality crucially relies on the ℓ 2 -Lipschitz continuity. If the loss exhibits more favourable Lipschitz continuity, e.g., with respect to an ℓ p -norm with p &gt; 2 , then our results can readily be adapted to use other contraction inequalities [24].

We now state two more well-known results from learning theory appearing throughout the manuscript, solely for convenience purposes.

Lemma E.5 (Symmetrization in expectation, e.g., Theorem 4.10 in [63]) . Let F be a class of functions X → R and n ∈ N . Let X 1 , . . . , X n be i.i.d. samples in X . Then

∣ ∣ Lemma E.6 (VC Bounds, [10, 9]) . Suppose that H consists of functions X → { 0 , 1 } and that H has VC dimension d H ∈ N . Let n ≥ d H . Then there exists a constant C &gt; 0 so that the Rademacher complexity of H with respect to any distribution on X is bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If H consists of functions X → [ -B,B ] and has VC-subgraph dimension (a.k.a. pseudo-dimension) d H ∈ N , then there exists a constant C &gt; 0 such that the Rademacher complexity of H with respect to any distribution on X is bounded as

<!-- formula-not-decoded -->

Moreover, for the L 2 -norm ball of functions with the same distribution µ as the Rademacher complexity, B = { f ∈ L 2 ( µ ) : ∥ f ∥ L 2 ( µ ) ≤ 1 } , let ρ := inf { r &gt; 0 : r 2 ≥ R n ( H∩ r B ) } . Then there exists a constant C &gt; 0 so that

<!-- formula-not-decoded -->

## F Table of Notations

Table 2: Notation

| Symbol                                                                                                                                                                                                                                               | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ℓ k , D ϕ R k ( f ) ̂ R k ( f ) E k ( f ) T s ( f ) - d k ( f ; h ) ̂ d k ( f ; h ) d s ( f ; h ) ̂ d s ( f ; h ) ⋆ = ( f ⋆ k ) k ∈ [ K ] ̂ h = ( ̂ h k ) k ∈ [ K ] g s g ′ s ̂ g s s lin λ s max λ B d 1 B d 2 B d ∞ diam ∥·∥ ( A ) R k n l k , u k | loss / Bregman divergence: Y ×Y → R risk: E [ ℓ k ( Y k , f ( X k ))] empirical risk: 1 n k ∑ n k i =1 ℓ k ( Y k i , f ( X k i )) excess risk: R k ( f ) - inf f ∈F all R k ( f ) s -trade-off: s ( E 1 ( f ) , . . . , E K ( f )) excess s -trade-off: T s ( f ) - inf g ∈G T s ( g ) risk discrepancy: E [ ℓ k ( h ( X k ) , f ( X k ))] empirical risk discrepancy: 1 N k ∑ N k i =1 ℓ k ( h ( ˜ X k i ) , f ( ˜ X k i )) scalarized discrepancy: s ( d 1 ( f ; h 1 ) , . . . ,d K ( f ; h K )) empirical scalarized discrepancy: s ( ̂ d 1 ( f ; h 1 ) , . . . , ̂ d K ( f ; h K )) Bayes-optimal models: f ⋆ k = argmin f ∈F all R k ( f ) ERMs in H k : ̂ h k = argmin h ∈H k ̂ R k ( h ) Pareto set in G : argmin g ∈G d s ( g ; f ⋆ ) helper Pareto set in G : argmin g ∈G d s ( g ; ̂ h ) our estimator: argmin g ∈G ̂ d s ( g ; ̂ h ) linear scalarization: ∑ K k =1 λ k v k Tchebycheff scalarization: max k ∈ [ K ] λ k v k ℓ 1 -ball: { v ∈ R d : ∥ v ∥ 1 ≤ 1 } ℓ 2 -ball: { v ∈ R d : ∥ v ∥ 2 ≤ 1 } ℓ ∞ -ball: { v ∈ R d : ∥ v ∥ ∞ ≤ 1 } diameter of the set A ⊂ R d : sup {∥ x - y ∥ : x, y ∈ A} Rademacher complexity w.r.t. distribution k and n samples critial radii from Eq. (12) |
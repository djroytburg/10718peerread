## Consistency Conditions for Differentiable Surrogate Losses

## Drona Khurana

University of Colorado Boulder drona.khurana@colorado.edu

## Dhamma Kimpara ∗

NSF National Center for Atmospheric Research Boulder, Colorado

dkimpara@ucar.edu

## Abstract

The statistical consistency of surrogate losses for discrete prediction tasks is often checked via the condition of calibration. However, directly verifying calibration can be arduous. Recent work shows that for polyhedral surrogates, a less arduous condition, indirect elicitation (IE), is still equivalent to calibration. We give the first results of this type for non-polyhedral surrogates, specifically the class of convex differentiable losses. We first prove that under mild conditions, IE and calibration are equivalent for one-dimensional losses in this class. We construct a counter-example that shows that this equivalence fails in higher dimensions. This motivates the introduction of strong IE, a strengthened form of IE that is equally easy to verify. We establish that strong IE implies calibration for differentiable surrogates and is both necessary and sufficient for strongly convex, differentiable surrogates. Finally, we apply these results to a range of problems to demonstrate the power of IE and strong IE for designing and analyzing consistent differentiable surrogates.

## 1 Introduction

In supervised learning problems, the goal of the learner is to output a model that accurately predicts labels on unseen feature vectors. These problems are specified by target losses , metrics intended to reflect model error. Natural choices for target losses arise in discrete prediction tasks like classification, ranking, and structured prediction. As minimizing discrete target losses directly is generally NP-hard, a convex surrogate loss is typically used instead. A link function maps surrogate reports (predictions) to target reports. Beyond ease of optimization, the surrogate and link must be statistically consistent with respect to the target loss, meaning that minimizing the surrogate should closely approximate minimizing the target given sufficient training data. In the finite-outcome setting, consistency turns out to be equivalent to a simpler condition called calibration [Bartlett et al., 2006, Tewari and Bartlett, 2007, Ramaswamy and Agarwal, 2016]. In particular, calibration has been central to the design of new consistent surrogates, serving as the key condition which must be satisfied.

While simpler than consistency, directly verifying calibration is often cumbersome. In particular, calibration requires that all sequences of reports converging to the surrogate minimizers (i.e., minimizers of the expected surrogate loss), eventually link to target minimizers (Figure 1). This complexity of verifying calibration in turn impedes the design of consistent surrogates. What easily

∗ Work done while at University of Colorado Boulder.

## Anish Thilagar

University of Colorado Boulder anish@colorado.edu

Rafael Frongillo

University of Colorado Boulder raf@colorado.edu

<!-- image -->

̸

Figure 1: Calibration vs. Indirect Elicitation. Calibration requires surrogate minimizers, as well as all sequences converging to surrogate minimizers link to target minimizers. In general, it is not trivial to choose a universal threshold past which the sequences link as desired. Determining such a threshold requires careful reasoning about the relative positions of surrogate minimizers across different outcome distributions, i.e., Γ( p ) relative to Γ( q ) for q = p . IE is analytically easier to verify, as it only requires that surrogate minimizers link to target minimizers. Moreover, IE can be thought of as a geometric condition on the probability simplex, which can directly lead to design insights (§4).

verifiable conditions still imply calibration for important classes of surrogate losses? One promising candidate is indirect elicitation (IE), which only requires that surrogate minimizers be linked to target minimizers (Figure 1). Finocchiaro et al. [2019] established an equivalence between calibration and IE for polyhedral surrogates, which paved the way for the design of novel, consistent surrogates for several open target losses of interest [Wang and Scott, 2020, Thilagar et al., 2022, Finocchiaro et al., 2022b]. Whether or not this equivalence extends to other classes of surrogates has remained open. More generally, beyond the polyhedral case, we still lack simpler conditions for calibration that support the design of new surrogates. We give the first such results-IE-like conditions which are easier to verify than calibration-for the broad and practically relevant class of convex, differentiable losses. We then demonstrate the power of these conditions to streamline the design process.

Theoretical Contributions. We first show that IE and calibration are equivalent for 1 -d convex, differentiable losses (§ 3). 2 In higher dimensions, however, IE no longer implies calibration, even for strongly convex surrogates (Example 2). To address this disparity, we propose a novel strengthening of IE we call strong indirect elicitation (strong IE; see Definition 6 in § 3.4). Strong IE is as easy to verify as IE, as it only depends on surrogate minimizers and not on the surrounding sequences (§ 4.1). We prove that under mild technical assumptions, strong IE implies calibration for differentiable surrogates (Theorem 2). Moreover, for the important class of strongly convex, differentiable surrogates, we show that strong IE is both necessary and sufficient for calibration (Theorem 3). All our calibration proofs are constructive, providing explicit link functions as part of the argument. Taken together, our results deepen our understanding of the conditions required for statistical consistency of surrogate losses.

Significance for Design. We illustrate with two examples that proving IE or strong IE is strictly simpler than directly proving calibration, thus drastically shortening the pathway to establishing consistency (§4). We then demonstrate how the geometric insights from these simpler conditions enable the construction of consistent, 1 -d differentiable surrogates for any orderable target loss (Theorem 4). As an application of Theorem 4, we construct a novel 1 -d surrogate that is convex, differentiable and consistent with respect to the ordinal regression loss. Together, these applications offer instructive proofs of concept and highlight how IE and strong IE can guide efficient surrogate design. We conclude with important future directions (§5).

Related Work. Previous works have also studied easier-to-verify conditions that imply calibration for certain classes of surrogate losses; our work is unique in proposing conditions for arbitrary discrete targets that are broadly applicable to the class of differentiable surrogates. The first conditions for calibration were studied for the important case of multi-class classification, where the target loss is the 0-1 loss. For binary classification, Bartlett et al. [2006] study 1-d surrogates and show that these are calibrated if and only if the loss, in margin form ϕ ( uy ) , is differentiable at 0 and ϕ ′ (0) &lt; 0 . In multi-class classification, Tewari and Bartlett [2007] study higher dimensional surrogates with symmetric superprediction sets [Williamson and Cranko, 2023]. They establish a technical condition on these sets that allows for easier conditions for calibration such as ensuring that the optimizer sets are singletons. The first to formally study the relationship between IE and calibration were Agarwal and Agarwal [2015]. However, their results do not apply to general discrete targets. For polyhedral

2 The statement does not hold if one relaxes differentiability; see Example 1.

losses, Ramaswamy and Agarwal [2016] showed that an IE-like condition is sufficient for calibration and Finocchiaro et al. [2024] showed that IE is equivalent to calibration for arbitrary discrete targets. Specific applications where IE influenced the design and analysis of surrogates/calibration include Finocchiaro et al. [2022b,a], Wang and Scott [2020], Nueve et al. [2024]. Our study of 1-d surrogates is heavily informed by the structure and elicitation of the target properties elucidated in Lambert et al. [2008], Lambert [2011], Steinwart et al. [2014], Finocchiaro and Frongillo [2018].

## 2 Background and Preliminaries

Table 1: Summary of key notation

| Symbol      | Description                   | Symbol        | Description             |
|-------------|-------------------------------|---------------|-------------------------|
| Y = [ n ]   | Set of labels                 | R = [ k ]     | Set of target reports   |
| ℓ : R→ R n  | Discrete target loss function | L : R d → R n | Surrogate loss function |
| ψ : R d →R  | Link function                 | ∆ n           | Probability simplex     |
| γ : ∆ n ⇒ R | Property elicited by ℓ        | Γ : ∆ n ⇒ R d | Property elicited by L  |
| γ r         | Level-set for γ at r          | Γ u           | Level-set for Γ at u    |

## 2.1 Targets, Surrogates and Link Functions

Given a finite label space Y and a finite report space R , let ℓ : R → R |Y| be a discrete target loss associated with some prediction task. ℓ ( · ) y represents the loss when the label is y ∈ Y . Unless specified otherwise, we assume Y = [ n ] and R = [ k ] , for n, k ≥ 2 . We denote the probability simplex over Y by ∆ n := { p ∈ R n | p i ≥ 0 , ∀ i ∈ [ n ] , p ⊤ 1 n = 1 } . As in Finocchiaro et al. [2020, 2024], we assume that the target loss under consideration is non-redundant , i.e., every report r ∈ R uniquely minimizes the expected loss for some distribution, i.e., ∀ r ∈ R , ∃ p ∈ ∆ n such that argmin r ′ ∈R ⟨ p, ℓ ( r ′ ) ⟩ = { r } .

Since ℓ is discrete and non-convex, it is hard to optimize. Our objective then, is to replace ℓ with a surrogate loss , defined over a continuous prediction space , say R d . Denote the surrogate by: L : R d → R n . For y ∈ [ n ] , let L ( · ) y : R d → R denote the y th component of L . To enable optimization of the surrogate, we assume that each component of the surrogate is convex , i.e., L ( · ) y : R d → R is convex for each y ∈ Y . Furthermore, since we are interested in analyzing differentiable surrogate losses, we will also assume that L ( · ) y is differentiable for each y ∈ Y . For all our theoretical results, we will make the following assumption:

Assumption 1. argmin u ∈ R d L ( u ) y is non-empty and compact for each y ∈ [ n ] .

Within the class of differentiable functions, Assumption 1 encompasses an important range of surrogates-including those with strongly convex components and strictly convex minimizable components. It also covers more nuanced cases, such as the surrogate described in Example 7 in Appendix A, which features two Huber-like components. Although each component is uniquely minimizable, not all their convex combinations are. We demonstrate in Section 4 that a one-dimensional instantiation of this loss is calibrated with respect to the ordinal regression target loss.

As the objective is to minimize ℓ , we must systematically map predictions in R d back to R . To do so, we introduce a link function ψ : R d →R .

## 2.2 Property Elicitation, Calibration and Indirect Elicitation

Any set-valued function defined on ∆ n is called a property . We say a loss elicits a property, if it maps each distribution to the minimizer of the expected loss under said distribution. We work with two key properties, denoted γ and Γ , which we define below:

Definition 1 (Target Property, Elicits, Level Sets) . The target loss ℓ : R→ R n is said to elicit the property γ : ∆ n ⇒ R , or in short-hand γ := prop [ ℓ ] if

<!-- formula-not-decoded -->

For any r ∈ R , denote γ r ⊆ ∆ n as the level-set for γ at r , i.e., γ r := { p ∈ ∆ n | r ∈ γ ( p ) } . Since R is a finite set, we say γ is a finite property.

Definition 2 (Surrogate Property, Elicits, Level Sets) . The surrogate loss L : R d → R n is said to elicit the property Γ : ∆ n ⇒ R d , or in short-hand Γ := prop [ L ] if

<!-- formula-not-decoded -->

For any u ∈ R d , denote Γ u ⊆ ∆ n as the level-set for Γ at u , i.e., Γ u := { p ∈ ∆ n | u ∈ Γ( p ) } .

In order to ensure that a surrogate-link pair is actually solving the target problem, we need to ensure that statistical consistency holds. In the finite outcome setting, it is well known that consistency reduces to the simpler notion of calibration [Bartlett et al., 2006, Tewari and Bartlett, 2007, Ramaswamy and Agarwal, 2016]. We thus focus on calibration throughout this paper. Given some distribution p ∈ ∆ n , and the corresponding surrogate minimizer(s) Γ( p ) , calibration roughly requires that all sequences of approximate minimizers link to the optimal target report.

Definition 3 (Calibration) . Given a discrete target ℓ , a surrogate-link pair ( L, ψ ) is calibrated if ∀ p ∈ ∆ n :

<!-- formula-not-decoded -->

We also say L is calibrated, if there exists a link ψ , such that ( L, ψ ) is calibrated.

We next define indirect elicitation, a condition even weaker than calibration (see Theorem 6 in Appendix B for a proof).

Definition 4 (Indirect Elicitation) . A surrogate-link pair, ( L, ψ ) indirectly elicits a discrete target ℓ , if ∀ u ∈ R d , Γ u ⊆ γ ψ ( u ) . We also say L indirectly elicits ℓ , if ∀ u ∈ R d , ∃ r ∈ R such that Γ u ⊆ γ r .

For polyhedral surrogates, Finocchiaro et al. [2024] established that indirect elicitation and calibration are equivalent. This result is striking, as indirect elicitation is significantly easier to verify than calibration. The latter requires ensuring that, for each distribution p ∈ ∆ n , any sequence of reports minimizing ⟨ p, L ( · ) ⟩ in the limit, eventually links to γ ( p ) . Equivalently, it demands that any sequence converging to Γ( p ) ultimately links to γ ( p ) (see Lemma 17 in Appendix C). In contrast, verifying indirect elicitation only necessitates linking optimal surrogate reports to optimal target reports. Specifically, if Γ u = ∅ for some report u then IE holds trivially and there is nothing to check. Otherwise, if p ∈ Γ( u ) , IE demands that ψ ( u ) ∈ γ ( p ) . Thus, IE is fully determined by the structure of the minimizing reports, i.e., Γ(∆ n ) := { Γ( p ) | p ∈ ∆ n } , whereas calibration requires analyzing Γ( p ) along with the local behavior of reports around it.

## 3 Motivating Counterexamples and Main Results

Our primary aim is to identify simpler conditions that yield calibration for differentiable surrogates. IE seems to be an ideal candidate: it is substantially easier to verify, is known to be equivalent for polyhedral losses, and all previously studied calibrated convex surrogates satisfy IE. However, it is not quite strong enough in general. We present two novel counterexamples (Example 1 in 3.1, Example 2 in 3.3) that demonstrate why IE is insufficient for calibration. These examples are far from pathological, and thus demonstrate exactly why IE is too weak for this setting. Example 2 motivates a new condition, strong IE that we go onto show implies calibration for convex, differentiable surrogates, and is both necessary and sufficient if the surrogate has strongly convex components.

## 3.1 Indirect elicitation and calibration are not equivalent

All previously studied convex surrogates that are known to indirectly elicit a target loss are calibrated for some link function ψ . 3 The literature has therefore treated both conditions as roughly equivalent to each other. Yet, this is not generally true. We identify the first known example of a loss that satisfies IE but cannot be calibrated for any choice of link function.

3 For example, consider the hinge surrogate for 0-1 loss, Y = R = {-1 , 1 } and L ( u ) y = max(0 , 1 -uy ) . If the link boundary is at u = 1 , ψ ( u ) = 1 ⇐⇒ u ≥ 1 , ( L, ψ ) satisfies IE but is not calibrated. However, if the link boundary is moved to u = 0 , then ( L, ψ ) is calibrated.

̸

Figure 2: The expected loss for two surrogates for abstain loss. Left: E p L cusp at p = (0 . 5 , 0 . 5) , it is clear that no link yields calibration for the abstain target (Example 1). Right: E p [ L smooth ] , a smoothed version of L cusp that is calibrated, again depicted at p = (0 . 5 , 0 . 5) (Example 6, Appendix A).

<!-- image -->

Example 1 (Cusp) . Let ℓ abs : {-1 , ⊥ , 1 } → R 2 be the target loss for binary classification with abstain level 1 4 using the label space Y = {-1 , 1 } [Bartlett and Wegkamp, 2008].

<!-- formula-not-decoded -->

Let L cusp : R → R 2 be the surrogate loss with L cusp ( u ) y = (1 -uy ) 2 + | u | , and ψ ( u ) = sign( u ) for u = 0 and ψ (0) = ⊥ (abstain). The expected loss of L cusp , E p [ L ( u )] at p = (0 . 5 , 0 . 5) is plotted in Figure 2 (left).

It is target-optimal to abstain whenever the most likely outcome occurs with a probability of at most 3 / 4 , i.e., γ ⊥ := { p ∈ ∆ 2 : max { p 1 , p 2 } ≤ 3 / 4 } = { p ∈ ∆ 2 | p 1 ∈ [1 / 4 , 3 / 4] } . Now, ( L cusp , ψ ) indirectly elicits ℓ as Γ 0 = γ ⊥ . Note that for any link to satisfy IE, it must agree with ψ in [ -0 . 5 , 0 . 5] . However, calibration is not satisfied for any p 1 ∈ (1 / 4 , 3 / 4) , for example: set p = (0 . 5 , 0 . 5) . Consider any positive sequence { u t &gt; 0 } t ≥ 0 with u t → 0 . Then, lim t →∞ ⟨ p, u t ⟩ = inf u ∈ R ⟨ p, L ( u ) ⟩ . However, each u t links to 1 and never the correct report, ⊥ . Indeed any link that satisfies IE exhibits this behavior (the sequence { u t } eventually links to 1 ). Thus, there is no other choice that could yield calibration. To restore calibration, it suffices to 'smooth out' the non-differentiable cusp at u = 0 , to get a differentiable surrogate as in Figure 2. See Example 6 in Appendix A.

L cusp is a remarkably simple non-polyhedral loss: it is strongly convex, one-dimensional, and differentiable everywhere except for a single cusp at u = 0 . It is as 'nice' as a non-differentiable loss can be. Yet, despite indirectly eliciting ℓ abs, it still fails to satisfy calibration. Since smoothing out the cusp yields a calibrated loss, a natural question arises: does differentiability, combined with indirect elicitation, always imply calibration? Differentiable losses are well-structured and extensively studied in machine learning as they are optimization-friendly and enjoy fast convergence rates. This makes the question of understanding the connection between IE and calibration under differentiability all the more compelling.

## 3.2 Differentiability and IE imply calibration for d = 1

In 1 -dimension, we answer the above question affirmatively: IE does imply calibration for differentiable real-valued surrogates. We provide a proof sketch of our theorem in this section.

To set the stage, we recall that a target loss ℓ that is indirectly elicited by a 1 -d surrogate (differentiable or not) possesses special structure. In particular, Finocchiaro et al. [2020] showed that the property γ := prop [ ℓ ] corresponding to such a target satisfies a condition known as orderability , which roughly states that there exists a connected, 1 -dimensional path that crosses each of the target level-sets.

Definition 5 (Orderable [Lambert, 2011]) . A finite property γ : ∆ n ⇒ R is orderable , if there is an enumeration of R = { r 1 , r 2 , ..., r k } such that for all i ≤ k -1 , we have that γ r j ∩ γ r j +1 is a hyperplane intersected with ∆ n .

Theorem 1. Let L : R → R n be a convex, differentiable surrogate that indirectly elicits ℓ . Under Assumption 1, L is calibrated with respect to ℓ .

Proof sketch: We first show that for each j ∈ [ k -1] , the boundary between adjacent target cells, i.e., γ r j ∩ γ r j +1 overlaps completely with some surrogate level-set. So, for some u j ∈ R , Γ u j = γ r j ∩ γ r j +1

Figure 3: Let Y = { 1 , 2 , 3 } , R = { 1 , 2 } . Three candidate target losses ℓ 1 , ℓ 2 , ℓ 3 : R → R 3 , that L CE (Example 2) could be a surrogate for. For each i ∈ { 1 , 2 , 3 } , ℓ i (1) = (1 , 1 , 1) . Whereas, ℓ 1 (2) = (5 / 2 , 5 / 4 , 0) , ℓ 2 (2) = (2 , 1 , 0) and ℓ 3 (2) = (5 / 3 , 5 / 6 , 0) . The target boundary elicited by ℓ 1 (resp. ℓ 3 ) is the red line segment joining q a 1 and q b 1 (resp. q a 3 and q b 3 ). The target boundary elicited by ℓ 2 is the red line segment joining p and (0 , 1 , 0) . The level sets of L CE are the blue points . All level sets of L CE are single points, barring Γ (0 , 0) , which is the entire segment spanning from p = (1 / 2 , 0 , 1 / 2) to (0 , 1 / 2 , 1 / 2) ( blue line segment ). Left: no IE. The segment level set crosses the target boundary, so L CE cannot indirectly elicit ℓ 1 . Center: IE. The segment level set does not cross, but just touches the target boundary, so IE holds, however, strong IE does not hold. Right: strong IE. The segment level set lies entirely within the target cell, so strong IE holds. Note: q a 1 = (0 , 0 . 8 , 0 . 2) , q b 1 = (0 . 4 , 0 , 0 . 6) , q a 3 = (0 . 2 , 0 . 8 , 0) , q b 3 = (0 . 6 , 0 , 0 . 4) .

<!-- image -->

(Lemma 20). We then establish that for any two distributions p, q ∈ ∆ n that lie on either side of the target boundary γ r j ∩ γ r j +1 , optimal reports Γ( p ) , Γ( q ) must lie on either side of u j (Lemma 21). Together, these results establish the existence of a connected, 1 -dimensional path through surrogate minimizers that faithfully mirrors any connected 1 -dimensional path traversing the target level sets. This naturally induces a link ψ that tracks the paths by mapping u j to either of { r j , r j +1 } , and mapping Γ( p ) and Γ( q ) to r j and r j +1 (Theorem 10).

The full proof and constructive link ψ are presented in Appendix D. It differs significantly from the polyhedral case, since barring convexity, differentiable and polyhedral losses have no commonality in their underlying structure.

## 3.3 Differentiability and IE do not imply calibration for d &gt; 1

Unfortunately, there is no direct analogue of Theorem 1 in higher dimensions. In particular, the following 2-dimensional surrogate is differentiable and satisfies IE for a target, but is not calibrated.

Example 2 (Counterexample: IE without calibration) . Let Y = { 1 , 2 , 3 } , R = { 1 , 2 } and consider L CE : R 2 → R Y , where

.

Each component of L CE is differentiable and strongly convex - and so L CE is minimizable. L CE indirectly elicits the target ℓ 2 : R → R 3 shown in Figure 3 (center, red). However, there is no link function ψ : R 2 →R , such that the pair ( L CE , ψ ) is calibrated with respect to ℓ 2 . In particular, there exists a sequence of reports that uniformly link to 1 , but converge to (0 , 0) , which has to link to 2 .

<!-- formula-not-decoded -->

More formally, define for 0 &lt; ϵ ≤ 1 , p ϵ := (1 / 2+ ϵ/ 2 , 0 , 1 / 2 -ϵ/ 2) . Then Γ( p ϵ ) = {( -ϵ 5 -ϵ , -2 ϵ 3+ ϵ )} . Notice that γ ( p ϵ ) = { 1 } = ⇒ ψ (Γ( p ϵ )) = 1 necessarily. Simultaneously, Γ (0 , 0) ⊆ γ 2 and Γ (0 , 0) ̸⊆ γ 1 , = ⇒ ψ ((0 , 0)) = 2 . Denote p ∗ := (0 , 1 / 2 , 1 / 2) ∈ Γ (0 , 0) and observe that γ ( p ∗ ) = {2}. Then, since Γ( p ϵ ) → (0 , 0) as ϵ → 0 , it follows by continuity that ⟨ p ∗ , L CE (Γ( p ϵ )) ⟩ → ⟨ p ∗ , L CE ((0 , 0)) ⟩ = inf u ∈ R 2 ⟨ p ∗ , L ( u ) ⟩ . Thus, L CE violates calibration for any choice of link ψ .

Similarly to L cusp, the surrogate L CE is extremely well-behaved. Each of its components are differentiable, strongly convex, minimizable, and the minimizing reports are always compact sets. Yet, L CE violates calibration despite satisfying IE with respect to ℓ 2 .

Turning our attention again to Figure 3 (center), where level-sets are depicted in blue: we see that geometrically, the violation stems from the location of Γ (0 , 0) , where Γ = prop [ L CE ] . Every surrogate

level-set is a singleton, except Γ (0 , 0) (blue line segment). Notice that Γ (0 , 0) just touches the (red) target boundary γ 1 ∩ γ 2 at the distribution (1 / 2 , 0 , 1 / 2) . Shifting γ 1 ∩ γ 2 to the right to get ℓ 1 immediately violates IE, since Γ u ̸⊆ γ r for any r ∈ { 1 , 2 } , when γ = prop [ ℓ 1 ] (Figure 3, left). On the other hand, shifting the boundary γ 1 ∩ γ 2 by any amount to the left yields a target loss of form ℓ 3 , for which L CE is calibrated (Figure 3, right). So, while indirect elicitation requires that the segment level set be contained within γ 2 , calibration is only achieved for L CE when Γ (0 , 0) is bounded away from the target boundary.

## 3.4 Strong indirect elicitation

Example 2 suggests that while calibration fails under IE, bounding the level set away from the target boundary resolves the problem. We formalize this idea with a new condition, strong indirect elicitation , which is a strengthening of indirect elicitation (see Theorem 5 in Appendix B).

Definition 6 (Strong Indirect Elicitation) . Given a target loss ℓ , let γ ∗ S = { p : γ ( p ) = S } . A surrogate L strongly indirectly elicits ℓ if ∀ u , ∃ S ⊆ R such that Γ u ⊆ γ ∗ S ; equivalently, if for every u ∈ R d and every p, q ∈ Γ u , γ ( p ) = γ ( q ) .

Revisiting Example 2: Notice that L CE does not satisfy strong IE with respect to ℓ 2 , since γ ( p ) = { 1 , 2 } , while γ ((0 , 1 / 2 , 1 / 2)) = { 2 } and both p, (0 , 1 / 2 , 1 / 2) ∈ Γ (0 , 0) . However, γ ( p ) = γ ((0 , 1 / 2 , 1 / 2)) = { 2 } , when γ = prop [ ℓ 3 ] . Thus, L CE satisfies strong IE with respect to ℓ 3 .

Though close to IE in definition, strong IE turns out to be much more powerful for differentiable surrogates, in that it implies calibration. 4

Theorem 2. Let L be a convex, differentiable surrogate that strongly indirectly elicits ℓ . Under Assumption 1, L is calibrated with respect to ℓ .

Proof sketch: Fix p ∈ ∆ n . Key to our proof is establishing that the minimizers 'surrounding' Γ( p ) link to γ ( p ) . Define the level-set bundle at p to be the collection of all level-sets passing through p , i.e., Γ Γ( p ) := ∪ u ∈ Γ( p ) Γ u . Repeated applications of strong IE establish the following: for a sufficiently small ϵ p &gt; 0 , the surrogate minimizers of distributions in ' ϵ p -proximity' to the level-set bundle at p link to γ ( p ) (Lemma 29). For simplicity, denote this set of minimizers as the ϵ p -minimizers. We have thus far that any valid link ψ must ensure that ψ ( ϵ p -minimizers) ∈ γ ( p ) . By establishing upperhemicontinuity of the set-valued map Γ ( · ) : R d ⇒ R (Lemma 26), we show that for some δ p &gt; 0 there exists a δ p -neighborhood around Γ( p ) wherein all minimizers are ϵ p -minimizers (Lemma 27). Thus, all minimizers surrounding Γ( p ) link to γ ( p ) . In effect, this means that surrogate minimizers that link to different target reports are well-separated in space which is imperative for calibration. We conclude via an explicit construction to extend this link ψ to a calibrated link defined for all surrogate reports, including the nowhere-optimal ones (Theorem 11). The reader may refer to Figure 7 in Appendix F for visual intuition of the proof.

Finally, we show that restricting to surrogates with strongly convex components makes strong IE necessary for calibration, and thus strong IE and calibration are equivalent for these surrogates.

Theorem 3. Let L : R d → R n be a surrogate, such that L ( · ) y : R d → R is strongly convex and differentiable for each y ∈ [ n ] . Then, L is calibrated with respect to ℓ if and only if it strongly indirectly elicits ℓ .

Proof sketch: Strong convexity and differentiability together imply Assumption 1. The sufficiency of strong IE thus follows by Theorem 2. For necessity, we show that violating strong IE implies violating calibration. If IE is violated, calibration is violated immediately. So let us assume we have IE but not strong IE. We show that under strong convexity, Γ is continuous and single-valued (see Lemma 35 for a proof). Next, we show the existence of a report u ∈ R d and a pair of distributions p, q ∈ Γ u , such that γ ( p ) ⊂ γ ( q ) (see Lemma 36). Thus, ∃ r ∈ R : r ∈ γ ( q ) , however, r / ∈ γ ( p ) . We then show that there exists a sequence of reports q t → q , such that γ ( q t ) = r . As Γ is single-valued there exists u t = Γ( q t ) , ∀ t . By continuity of Γ , q t → q = ⇒ Γ( q t ) → Γ( q ) ⇐⇒ u t → u . Further, by the continuity of ⟨ p, L ( · ) ⟩ , ⟨ p, L ( u t ) ⟩ → ⟨ p, L ( u ) ⟩ = inf v ∈ R d ⟨ p, L ( v ) ⟩ since p ∈ Γ u . However, since γ ( q t ) = { r } , ψ ( u t ) = r necessarily. At the same time, r / ∈ γ ( p ) . Hence, calibration is violated at p . See Theorem 12 in Appendix G for a full proof.

4 Interestingly, no polyhedral surrogate satisfies strong IE; see Theorem 7 in Appendix B

## 4 Applications

As IE and strong IE are easier to verify than calibration (Figure 1), our main results above lead to improved analytical methods to analyze and design consistent surrogates, which we now demonstrate.

## 4.1 Ease of verification

IE and strong IE are both completely characterized by the relation of optimal surrogate reports to optimal target reports. Importantly, neither condition requires analyzing sequences converging to optimal reports. Thus both conditions are strictly simpler to verify than calibration. While strong IE is a more stringent requirement than IE, checking strong IE is just as easy as checking IE at the individual-report level (see Proposition 1 in Appendix B).

We now present two examples illustrating how concluding calibration via IE or strong IE can significantly simplify the analysis: whereas direct calibration proofs require characterizing minimizers and analyzing nearby sequences, establishing IE or strong IE only requires reasoning about the minimizers themselves. (see also Figure 1 for visual intuition)

Example 3 (Universally calibrated surrogate) . Lemma 11 of Ramaswamy and Agarwal [2016] proposes a n -1 -dimensional, strongly convex, differentiable surrogate that is calibrated for all discrete targets. After the first claim in their proof (see pages 29-30 Ramaswamy and Agarwal [2016]):

## Proof via strong IE

Fix p ∈ ∆ n . Minimizing ⟨ p, L ( u ) ⟩ = ∑ n -1 j =1 ( p j ( u j -1) 2 +(1 -p j ) u 2 j ) yields the unique minimizer u ∗ = ( p 1 , . . . , p n -1 ) ⊤ . Hence | Γ( p ) | = 1 and Γ u = { p } . Immediately, L satisfies strong IE, and thus L is calibrated by Theorem 2.

Our approach shortens the proof from an entire page to a few lines. We also obviate the need for subtle arguments regarding the convergence of sequences that were required in the original proof.

Example 4 (Subset-ranking surrogates) . Theorem 3 of Ramaswamy et al. [2013] proposes a lowdimensional calibrated surrogate for subset-ranking targets common in information retrieval. Our results significantly shorten their calibration proof (see pages 3-4, Ramaswamy et al. [2013]):

<!-- image -->

This bypasses all subsequent proof steps (25 lines) following the first line of page 4 wherein intricate reasoning to show all sequences converging to minimizer sets are appropriately linked.

## 4.2 Design of 1-dimensional surrogates

Example 3 demonstrates the existence of an n -1 dimensional surrogate that is calibrated for any target loss with n outcomes. However, the complexity of several optimization algorithms is often linear, or even quadratic in the domain dimension. Thus, a major research goal of the surrogate loss literature is the design of dimension-efficient surrogates (ideally d &lt;&lt; n -1 for large n ) when possible [Ramaswamy and Agarwal, 2012, Ramaswamy et al., 2013, 2015, Finocchiaro et al., 2019, Blondel, 2019, Finocchiaro et al., 2021]. Recall that Theorem 1 established the equivalence between IE and calibration for 1 -d differentiable surrogates. This equivalence enables us to construct a 1 -d surrogate that is convex and differentiable, for any orderable target loss. We formalize this statement below in Theorem 4 and provide a proof sketch that highlights the key ideas behind the construction.

Theorem 4. Given an orderable target ℓ : R→ R n , there exists a convex, differentiable surrogate L : R → R n satisfying Assumption 1, which is calibrated with respect to ℓ .

Figure 4: Let Y = R = { 1 , 2 , 3 } . The solid-peach colored lines depict the target boundaries elicited by the ordinal regression loss ℓ ord : R → R 3 . The dotted-blue lines depict the level-sets of the surrogate L H : R → R 3 defined in Example 5. Since no level-set of L H : R → R 3 crosses from one target cell to another, IE holds. By Theorem 1 calibration follows.

<!-- image -->

Proof sketch. Since γ := prop[ ℓ ] is orderable, there is an orderable enumeration ( r 1 , . . . , r k ) of reports (Def. 5). By Theorem 11 of Finocchiaro et al. [2020], there exist vectors v 1 , . . . , v k -1 ∈ R n such that (i) for each j ∈ [ k -1] , ⟨ p, v j ⟩ = 0 for all p ∈ γ r j ∩ γ r j +1 , and (ii) the coordinates are monotone, i.e., v i,y ≤ v i +1 ,y for every i ∈ [ k -2] and y ∈ [ n ] .

Let V ∈ R n × ( k -1) have columns v 1 , . . . , v k -1 , and write V j for the j -th row. For each j ∈ [ n ] , define L ( · ) j := LinIntGrad ( V j ) (Subroutine 1, Appendix H). The subroutine first specifies a map g j : R → R on [1 , k -1] by linear interpolation of the values V j [1] , . . . , V j [ k -1] , so that g j ( i ) = V j [ i ] for all integers i ∈ { 1 , . . . , k -1 } ; hence g j is continuous and nondecreasing on (1 , k -1) . It then extends g j outside [1 , k -1] . In particular, for x ≤ 1 , g j ( x ) = X [1] + ( x -1) . And for x ≥ k -1 , g j ( x ) = X [ k -1] + ( x -( k -1)) . The construction ensures that continuity and monotonicity of g j are preserved across R . Furthermore, g j crosses 0 either at a singleton, or at a compact interval. Finally, the subroutine sets L ( u ) j = ∫ u 1 g j ( t ) dt , so ( L ( · ) j ) ′ = g j . Lemma 37 proves that each L ( · ) j is convex, belongs to C 1 ( R ) , has nonempty compact minimizers (the sets { g -1 j (0) | j ∈ [ n ] } ), and so L satisfies Assumption 1. Moreover, at integers i ∈ { 1 , . . . , k -1 } we have ∇ L ( i ) = v i . In Theorem 13 of Appendix H, we show that these properties imply L indirectly elicits ℓ . Hence L is calibrated with respect to ℓ by Theorem 1.

As an application of Theorem 4, we present a novel surrogate for the ordinal regression target loss in Example 5. While previous works have proposed surrogates for ordinal regression [Ramaswamy and Agarwal, 2016, Pedregosa et al., 2017, Finocchiaro et al., 2019], none of the surrogates therein are simultaneously convex, differentiable, minimizable and 1 -dimensional.

Example 5 (Huber-like surrogate for ordinal regression) . Here Y = R = { 1 , 2 , 3 } . Predictions farther away from the true outcome are more heavily penalized. The 3 -class ordinal regression loss is ℓ ord ( y, r ) := | y -r | , for y, r ∈ { 1 , 2 , 3 } . Then an application of Theorem 4 yields the surrogate

<!-- formula-not-decoded -->

where h ( x ) = x 2 2 and f ( x ) = x 2 2 for -1 ≤ x ≤ 1 and f ( x ) = | x | -0 . 5 otherwise.

L H indirectly elicits ℓ ord and is therefore calibrated with respect to it. Figure 4 depicts the target (peach colored lines) and surrogate (blue dotted lines) level-sets for ℓ ord and L H. The target ℓ ord poses a non-trivial challenge. In particular, for a 1 -d convex, differentiable surrogate L to indirectly elicit ℓ ord , it must admit a non-unique minimizer at (0 , 1 / 2 , 0) since the two target-boundaries intersect at this point. On the other hand, the minimizers elsewhere must be unique. L H is precisely such a minimizer. For L H, Γ((1 / 2 , 0 , 1 / 2)) = [ -1 , 1] , whereas for every other p ∈ ∆ 3 , | Γ( p ) | = 1 .

## 5 Discussion and Future Directions

Our results are the first to establish general calibration conditions on the widely used class of convex differentiable surrogate losses in relation to arbitrary discrete target losses. We anticipate that the generality of our results will aid further advances in application and theory. Our conditions are inspired by the equivalence of IE and calibration for polyhedral surrogates. Like IE, strong IE is

substantially easier to verify than checking calibration directly. Hence, strong IE for differentiable losses could play a similar role to IE for polyhedral losses, where IE has been used to establish convex calibration dimension bounds [Ramaswamy and Agarwal, 2016] and to design and analyze numerous surrogates [Finocchiaro et al., 2022b,a, Wang and Scott, 2020, Nueve et al., 2024]. Indeed, we already make first steps in regards to design, by proposing a generalized construction for designing differentiable 1 -dimensional surrogates for orderable targets.

Lower bounds. A promising direction for future work is to use strong IE to study prediction dimension. We believe it can establish lower bounds on the prediction dimension of calibrated surrogates for important target losses. Finocchiaro et al. [2021] leverage IE as a tool to establish such lower bounds. Recall that strong IE is necessary for calibration for the class of strongly convex, differentiable surrogate. At the same time, strong IE imposes more stringent constraints on surrogates than IE. We therefore believe strong IE offers promise to establish novel lower bounds in this setting.

Relaxing Assumption 1. Theorems 1 and 2 assume that arg min u ∈ R d L ( · ) y is non-empty and compact for each y ∈ [ n ] . Theorem 8 in Appendix C shows that Assumption 1 is equivalent to the condition that Γ( p ) is non-empty and compact for every p ∈ ∆ n . The non-emptiness, i.e., minimizability of the functions {⟨ p, L ( · ) ⟩| p ∈ ∆ n } is mathematically well-motivated. Indeed, if minimizability fails for some distribution p ∈ ∆ n , then Γ( p ) is empty. In this case, checking calibration at p necessitates analyzing sequences of form { u t } t ∈ N + such that lim t →∞ ⟨ p, L ( u t ) ⟩ = inf u ∈ R d ⟨ p, L ( u ) ⟩ . Thus, while understanding calibration for non-minimizable losses is an important and interesting direction in its own right, IE and strong IE are not the appropriate tools to do so. We speculate instead that the recently developed theory on astral spaces [Dudík et al., 2022] can be leveraged for this direction. Our assumption of compactness on Γ( p ) is technical and necessary for our proof approach, but may not be strictly necessary for strong IE to yield calibration. Differentiable surrogates with unbounded (and thus non-compact) minimizers are common in practice (for example: the squared hinge loss, the modified Huber loss, etc.) Relaxing this assumption is therefore a valuable direction for future research and could potentially enhance the practical appeal of strong IE.

## Acknowledgments and Disclosure of Funding

We thank Mabel Cluff for early discussions and insights on this project. We thank Stephen Becker for discussions on convergence properties of set-valued functions. We thank Jessie Finocchiaro and Enrique Nueve for discussions on 1 d surrogate losses. This material is based upon work supported by the National Science Foundation under Grant No. IIS-2045347.

## References

- A. Agarwal and S. Agarwal. On consistent surrogate risk minimization and property elicitation. In Conference on Learning Theory , pages 4-22. PMLR, 2015. 2
- F. Aurenhammer. Power diagrams: properties, algorithms and applications. SIAM journal on computing , 16(1):78-96, 1987. 15
- P. L. Bartlett and M. H. Wegkamp. Classification with a reject option using a hinge loss. Journal of Machine Learning Research , 9(8), 2008. 5
- P. L. Bartlett, M. I. Jordan, and J. D. McAuliffe. Convexity, classification, and risk bounds. Journal of the American Statistical Association , 101(473):138-156, 2006. 1, 2, 4
- M. Blondel. Structured prediction with projection oracles. Advances in neural information processing systems , 32, 2019. 8
- K. C. Border. Introduction to correspondences, 2013. 21, 22
- S. Boyd. Convex optimization. Cambridge UP , 2004. 26
- M. Dudík, R. E. Schapire, and M. Telgarsky. Convex analysis at infinity: An introduction to astral space. arXiv preprint arXiv:2205.03260 , 2022. 10

- J. Finocchiaro and R. Frongillo. Convex elicitation of continuous properties. Advances in Neural Information Processing Systems , 31, 2018. 3
- J. Finocchiaro, R. Frongillo, and B. Waggoner. An embedding framework for consistent polyhedral surrogates. Advances in neural information processing systems , 32, 2019. 2, 8, 9
- J. Finocchiaro, R. Frongillo, and B. Waggoner. Embedding dimension of polyhedral losses. In Conference on Learning Theory , pages 1558-1585. PMLR, 2020. 3, 5, 9, 19, 30
- J. Finocchiaro, R. Frongillo, and B. Waggoner. Unifying lower bounds on prediction dimension of convex surrogates. Advances in Neural Information Processing Systems , 34:22046-22057, 2021. 8, 10
- J. Finocchiaro, R. M. Frongillo, and B. Waggoner. An embedding framework for the design and analysis of consistent polyhedral surrogates. Journal of Machine Learning Research , 25(63):1-60, 2024. 3, 4, 14
- J. J. Finocchiaro, R. Frongillo, E. Goodwill, and A. Thilagar. Consistent polyhedral surrogates for top-k classification and variants. In International Conference on Machine Learning , pages 21329-21359. PMLR, 2022a. 3, 10
- J. J. Finocchiaro, R. Frongillo, and E. B. Nueve. The structured abstain problem and the lovász hinge. In Conference on Learning Theory , pages 3718-3740. PMLR, 2022b. 2, 3, 10
- M. Henk, J. Richter-Gebert, and G. M. Ziegler. Basic properties of convex polytopes. In Handbook of discrete and computational geometry , pages 383-413. Chapman and Hall/CRC, 2017. 14
9. J.-B. Hiriart-Urruty and C. Lemaréchal. Convex analysis and minimization algorithms I: Fundamentals , volume 305. Springer science &amp; business media, 1996. 16
10. S.-T. Hu. Introduction to general topology. (No Title) , 1966. 22
- N. S. Lambert. Elicitation and evaluation of statistical forecasts. Preprint , 2011. 3, 5, 18
- N. S. Lambert, D. M. Pennock, and Y. Shoham. Eliciting properties of probability distributions. In Proceedings of the 9th ACM Conference on Electronic Commerce , pages 129-138, 2008. 3, 15
- E. Nueve, D. Kimpara, B. Waggoner, and J. Finocchiaro. Trading off consistency and dimensionality of convex surrogates for multiclass classification. In Advances in Neural Information Processing Systems , 2024. 3, 10
- F. Pedregosa, F. Bach, and A. Gramfort. On the consistency of ordinal regression methods. Journal of Machine Learning Research , 18(55):1-35, 2017. 9
- H. G. Ramaswamy and S. Agarwal. Classification calibration dimension for general multiclass losses. Advances in Neural Information Processing Systems , 25, 2012. 8
- H. G. Ramaswamy and S. Agarwal. Convex calibration dimension for multiclass loss matrices. Journal of Machine Learning Research , 17(14):1-45, 2016. 1, 3, 4, 8, 9, 10
- H. G. Ramaswamy, S. Agarwal, and A. Tewari. Convex calibrated surrogates for low-rank loss matrices with applications to subset ranking losses. Advances in Neural Information Processing Systems , 26, 2013. 8
- H. G. Ramaswamy, A. Tewari, and S. Agarwal. Consistent algorithms for multiclass classification with a reject option. arXiv preprint arXiv:1505.04137 , 2015. 8
- R. Rockafellar. Convex analysis. Princeton Mathematical Series , 28, 1970. 16
- I. Steinwart, C. Pasin, R. Williamson, and S. Zhang. Elicitation and identification of properties. In Conference on Learning Theory , pages 482-526. PMLR, 2014. 3
- A. Tewari and P. L. Bartlett. On the consistency of multiclass classification methods. Journal of Machine Learning Research , 8(5), 2007. 1, 2, 4

- A. Thilagar, R. Frongillo, J. J. Finocchiaro, and E. Goodwill. Consistent polyhedral surrogates for top-k classification and variants. In International Conference on Machine Learning , pages 21329-21359. PMLR, 2022. 2
- Y. Wang and C. Scott. Weston-watkins hinge loss and ordered partitions. Advances in neural information processing systems , 33:19873-19883, 2020. 2, 3, 10
- R. C. Williamson and Z. Cranko. The geometry and calculus of losses. Journal of Machine Learning Research , 24(342):1-72, 2023. 2

## A Examples

4

3

2

1

)

u

(

1

-

L

0

p

v

⊥

= [0

.

5

,

1

0

.

5]

⊤

2

L

<!-- image -->

+1

(

u

)

Figure 5: The figure on the left plots the superprediction set, { L ( u ) : u ∈ R + } , of a surrogate loss that IEs but is not calibrated. To see non-calibration, notice that there is only one possible link. Fix p = [0 . 5 , 0 . 5] ⊤ . The optimal loss is achieved by arg inf v ∈{ L ( u ): u ∈ R + } ⟨ v, p ⟩ . Thus the loss is optimized by v ⊥ which must link to the abstain report ⊥ . However, consider the points to the left of v ⊥ , which link to +1 . The infimum of the loss over these points for p is equal to the loss of v ⊥ , thus violating calibration.

Example 6. Let the target loss be binary classification with abstain level 1 4 , L smooth : R → R 2 .

<!-- formula-not-decoded -->

This is the smooth loss plotted on the right side of Figure 5. For any p ∈ [0 , 1] , Γ(( p, 1 -p )) = 2 p -1 . Hence Γ u = { ( 1+ u 2 , 1 -u 2 ) } . Clearly, each component of L smooth is strongly convex and differentiable. Thus, if L smooth strongly indirectly elicits ℓ , then L smooth is calibrated with respect to ℓ by Theorem 3. Since Γ u is a singleton, strong IE follows trivially by definition, and so L smooth is calibrated with respect to ℓ . "Smoothening out" L cusp from Example 1 thus resolves calibration.

Example 7 (Convex Combinations of Huber losses) . For u ∈ R d , let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

be the sum of two Huber losses. Then, for p = 1 / 2 , Γ( p ) = [ -1 , 1] ×{ 0 } d -1 . For p &gt; 1 / 2 , Γ( p ) is a single point in (1 , 2] ×{ 0 } d -1 . For p &lt; 1 / 2 , Γ( p ) is a single point in [ -2 , -1) ×{ 0 } d -1 . We can see this visually for d = 1 in Figure 6.

3

be the Huber loss in R d . Define

4

<!-- image -->

2

0

6

2

0

N-

Figure 6: The plot of ⟨ p, L H ( u ) ⟩ for 3 different values of p with d = 1 . In higher dimensions, the loss is minimized by exactly the same points, since by construction the minima will always lie on the u 1 -axis. Left: p = 1 / 4 , the ⟨ p, L H ( u ) ⟩ has a unique minimum at u = -5 / 3 . Center: p = 1 / 2 , the ⟨ p, L H ( u ) ⟩ is minimized by any choice of u ∈ [ -1 , 1] . Left: p = 3 / 4 , the ⟨ p, L H ( u ) ⟩ has a unique minimum at u = 5 / 3 .

Lemma 1. rank( ∇ L CE ( u )) is 1 when u = (0 , 0) and 2 otherwise.

Proof.

Let v 1 and v 2 denote the first and second column of ∇ L CE ( u ) respectively. rank( ∇ L CE ( u )) = 0 if and only if v 1 = v 2 = 0 . However, there is no choice of u 1 or u 2 such that either v 1 or v 2 are 0, so rank( ∇ L CE ( u )) ≥ 1 everywhere.

<!-- formula-not-decoded -->

rank( ∇ L CE ( u )) = 1 if and only if λv 1 = v 2 for some λ ∈ R . For this to hold for the first row of ∇ L CE ( u ) we must have 2 λu 1 + λ = 2 u 2 +2 , so u 2 = λu 1 + λ 2 -1 . Similarly, using the second row of ∇ L CE ( u ) implies u 2 = λu 1 + λ 4 -1 2 . This gives two equivalent expressions for u 2 , so we must have λ 2 -1 = λ 4 -1 2 , so λ = 2 . Plugging this into either of the expressions for u 2 yields u 2 = 2 u 1 . Finally, using these values of λ and u 2 the last row of ∇ L CE ( u ) becomes 12 u 1 -2 = 4 u 1 -2 , so u 1 = 0 , and thus u 2 = 2 u 1 = 0 as well. Therefore, rank( ∇ L CE ( u )) = 1 only when u = (0 , 0) .

̸

For any other value of u , we then have rank( ∇ L CE ( u )) ≥ 2 , but since u ∈ R 2 , rank( ∇ L CE ( u )) ≤ 2 everywhere. Therefore, for all u = (0 , 0) , rank( ∇ L CE ( u )) = 2 .

## B Property Elicitation, Level Sets and Minimizing Sets

Lemma 2. Let A,B ⊆ ∆ n : A ⊂ B . Then Γ( A ) ⊆ Γ( B )

Proof. Since A ⊆ B , we have that ∀ a ∈ A,a ∈ B . So, ∀ a ∈ A, Γ( a ) ⊆ ∪ b ∈ B Γ( b ) = Γ( B ) . Thus, Γ( A ) = ∪ a ∈ A Γ( a ) ⊆ Γ( B )

Lemma 3. Consider any convex, differentiable L : R d → R n . Let p ∈ ∆ n and u ∈ R d . Then, u ∈ Γ( p ) ⇐⇒ ∇ L ( u ) ⊤ p = 0 d . Equivalently, Γ u = { p ⊆ ∆ n |∇ L ( u ) ⊤ p = 0 d } .

Proof. Since L y is convex for every y ∈ [ n ] , it follows that the function ⟨ p, L ( · ) ⟩ : R d → R is convex for any p ∈ ∆ n . Now, since the domain of ⟨ p, L ( · ) ⟩ is open and the minimum is attained at some u ∈ R d , it follows that ∇⟨ p, L ( u ) ⟩ = 0 d = ⇒ ∇ L ( u ) ⊤ p = 0 d . Conversely, if ∇⟨ p, L ( u ) ⟩ = 0 d , then u ∈ Γ( p ) by convexity of ⟨ p, L ( · ) ⟩ .

Lemma 4. Any convex, differentiable surrogate L : R d → R n that indirectly elicits a target ℓ : R→ R n satisfies rank ( ∇ L ( u ) ⊤ ) &gt; 0 for every u ∈ R d .

Proof. Assume to the contrary, i.e., ∃ u ∈ R d , such that, rank ( ∇ L ( u ) ⊤ ) = 0 . By the rank-nullity theorem, this means null ( ∇ L ( u ) ⊤ ) = n . So ∇ L ( u ) ⊤ p = 0 d is satisfied by any p ∈ ∆ n . From Lemma 3, we get that Γ u = ∆ n . We assume k = |R| ≥ 2 . If not, the prediction problem is trivial, so { 1 , 2 } ⊆ R for any problem of interest. First consider the pair of discrete reports (1 , 2) . By the non-redundancy of discrete reports, ∃ p 1 ∈ γ 1 , p 2 ∈ γ 2 , such that ⟨ p 1 , ℓ ( · ) ⟩ is uniquely optimized by the discrete prediction 1 and ⟨ p 2 , ℓ ( · ) ⟩ is uniquely optimized by the discrete prediction 2 . Since

Γ u = ∆ n , it follows that p 1 , p 2 ∈ Γ u . This means Γ u ̸⊆ γ 1 , since p 2 / ∈ γ 1 , and similarly Γ u ̸⊆ γ 2 . Similarly, for any j ∈ R : j / ∈ { 1 , 2 } , consider the reports (1 , j ) and repeat the same rationale to establish that Γ u ̸⊆ γ j . Thus, ∃ u ∈ R d such that Γ u ̸⊆ γ r , ∀ r ∈ R , implying L does not indirectly elicit ℓ by the definition of indirect elicitation.

Lemma 5. Let L : R d → R n be a convex, but not necessarily differentiable surrogate. Then for any u ∈ R d , Γ u is compact.

Proof. Pick any u ∈ R d . Since Γ u ⊆ ∆ n and ∆ n is compact, it is clear that Γ u is bounded. So it suffices to show that Γ u is closed. Let { p t } t ∈ N + ⊆ Γ u and suppose that p t → p . Wewant to show that p ∈ Γ u . First note that since p t ∈ ∆ n for each t , it follows that p ∈ ∆ n by compactness of ∆ n . Now, pick any v ∈ R d . It holds for each t ∈ N + that ⟨ p t , L ( u ) ⟩ ≤ ⟨ p t , L ( v ) ⟩ . Taking the limit as t →∞ on both sides, it holds that ⟨ p, L ( u ) ⟩ ≤ ⟨ p, L ( v ) ⟩ for any v ∈ R d . Thus, u ∈ Γ( p ) = ⇒ p ∈ Γ u . Hence, Γ u is closed.

Theorem 5. Let L : R d → R n be a surrogate that strongly indirectly elicits a target loss ℓ : R→ R n . Then L indirectly elicits ℓ .

Proof. Pick any u ∈ R d . By definition, there exists some S ⊆ R , such that γ ( p ) = S for every p ∈ Γ u . Equivalently, p ∈ ∩ r ∈ S γ r for every p ∈ Γ u . Thus, Γ u ⊆ ∩ r ∈ S γ r = ⇒ ∃ r ∈ R , such that Γ u ⊆ γ r .

Theorem 6. Let L : R d → R n be a surrogate that is calibrated with respect to ℓ : R→ R n . Then L indirectly elicits ℓ .

Proof. Since L is calibrated with respect to ℓ , there exists a link function ψ : R d →R , such that, ( L, ψ ) is calibrated with respect to ℓ . Suppose u ∈ R d . If u / ∈ Γ(∆ n ) , then Γ u = ∅ = ⇒ Γ u ⊆ γ r , ∀ r ∈ R yielding indirect elicitation. Now, suppose u ∈ Γ(∆ n ) . We show that Γ u ⊆ γ ψ ( u ) . Assume to the contrary. Then, there exists some p ∈ Γ u , such that p / ∈ γ ψ ( u ) = ⇒ ψ ( u ) / ∈ γ ( p ) . Then, inf v ∈ R d : ψ ( v ) / ∈ γ ( p ) ⟨ p, L ( v ) ⟩ ≤ ⟨ p, L ( u ) ⟩ = inf v ∈ R d ⟨ p, L ( v ) ⟩ since p ∈ Γ u , hence violating calibration. Thus, Γ u ⊆ γ ψ ( u ) and so ( L, ψ ) indirectly elicit ℓ = ⇒ L indirectly elicits ℓ .

Theorem 7. Let L : R d → R n be a polyhedral surrogate that indirectly elicits some target loss ℓ : R→ R n . Then L does not strongly indirectly elicit ℓ .

̸

Proof. We know from [Finocchiaro et al., 2024] that any polyhedral surrogate has a finite representative set S , i.e., S ⊂ R d , such that S has a finite number of elements and that for any p ∈ ∆ n , there exists some u ∈ S , such that p ∈ Γ u . We leverage this fact to prove our claim. Assume by contradiction that L strongly indirectly elicits ℓ . Suppose WLOG that S = { u 1 , u 2 , ..., u m } . Since ∪ i ∈ [ m ] Γ u i = ∆ n , it follows that there exists some S ′ ⊆ S , such that relint ( γ 1 ) ⊆ ∪ v ∈ S ′ Γ v . In particular, S ′ ⊆ S and the level-sets of the reports in S ′ cover relint ( γ 1 ) . Since S is finite, there must exist a minimal subset of S , the level sets of whose elements cover relint ( γ 1 ) . Assume S ′ is such a minimal covering subset. First, we claim that for any v ∈ S ′ , p ∈ Γ v = ⇒ γ ( p ) ∩ { 1 } ̸ = ∅ . Assume not. Then ∃ v ∈ S ′ , such that γ ( p ) ∩ { 1 } = ∅ for some p ∈ Γ v . By strong IE, it holds that γ ( p ) ∩ { 1 } = ∅ , ∀ p ∈ Γ v . Thus, if v ∈ S ′ , S ′ can't be minimal. Next, we claim that for any v ∈ S ′ , p ∈ Γ v = ⇒ γ ( p ) = { 1 } . Assume not, then ∃ v ∈ S ′ : { 1 } ⊂ γ ( p ) for some p ∈ Γ v . By strong IE, it follows that ∀ p ∈ Γ v , { 1 } ⊂ γ ( p ) = ⇒ ∀ p ∈ Γ v , p / ∈ relint ( γ 1 ) . Thus, if v ∈ S ′ , S ′ can't be minimal. Hence, for every v ∈ S ′ , p ∈ Γ v = ⇒ γ ( p ) = { 1 } . Therefore, relint ( γ 1 ) ̸⊂ ∪ v ∈ S ′ Γ v = ⇒ relint ( γ 1 ) = ∪ v ∈ S ′ Γ v . However, S ′ being a subset of finite S is itself finite, and by Lemma 5, ∪ v ∈ S ′ Γ v is a finite union of closed sets implying that ∪ v ∈ S ′ Γ v itself must be closed. On the other hand, relint ( γ 1 ) is not closed by definition and so relint ( γ 1 ) = ∪ v ∈ S ′ Γ v . Hence, L strongly indirectly eliciting ℓ yields a contradiction.

Lemma 6. Consider any convex, differentiable L : R d → R n . Suppose u ∈ R d . Then Γ u is a polytope. If L : R d → R n indirectly elicits any target ℓ : R→ R n , then affdim (Γ u ) ≤ n -2 .

Proof. Recall by Lemma 3 that Γ u = { p ∈ ∆ n |∇ L ( u ) ⊤ p = 0 d } . In other words, Γ u = ∆ n ∩ ker ( ∇ L ( u ) ⊤ ) . So Γ u is the intersection of a polytope and a subspace, implying that Γ u is itself a polytope [Henk et al., 2017].

Next, suppose L indirectly elicits some target ℓ . Then by Lemma 4, it holds that d ≥ rank ( ∇ L ( u ) ⊤ ) &gt; 0 . Thus, 1 ≤ nullity ( L ( u ) ⊤ ) &lt; n . So, Γ u is the intersection of a set of affine dimension n -1 (i.e., ∆ n ) and a subspace of dimension at least 1 (i.e., ker ( ∇ L ( u )) ⊤ ). Thus, affdim (Γ u ) ≤ n -2 .

Lemma 7. Suppose ℓ : R→ R n is an elicitable target loss, and that Y and R are finite sets. Suppose further that each r ∈ R is non-redundant. Then for any r ∈ R , γ r is a convex polytope, such that affdim ( γ r ) = n -1 .

Proof. This can be observed directly from the fact that any finite target is elicitable if and only its cells γ r (where, r ∈ R ) form a power diagram [Lambert et al., 2008]. Power diagrams are essentially weighted Voronoi diagrams. For more details on power diagram, we refer the reader to [Aurenhammer, 1987].

̸

Lemma 8. Let L : R d → R n be a convex, differentiable surrogate. Let ℓ : R → R n . Let u ∈ R d . Suppose p, p ′ ∈ Γ u and that ∃ S ⊆ R such that γ ( p ) ∩ γ ( p ′ ) = S = ∅ . Then, γ ( q ) ⊆ S , where q := p + p ′ 2 .

Proof. First note that by convexity of Γ u , q ∈ Γ u = ⇒ q ∈ ∆ n . For i ∈ R , denote ℓ i := ( ℓ (1 , i ) , ℓ (2 , i ) , ..., ℓ ( n, i )) as the loss vector corresponding to prediction i . Say j ∈ γ ( q ) = ⇒ q ⊤ ℓ j ≤ q ⊤ ℓ i , ∀ i ∈ R . Now, suppose j / ∈ S . Let t ∈ S = ⇒ t ∈ γ ( p ) ∩ γ ( p ′ ) . So, p ′⊤ ℓ t ≤ p ′⊤ ℓ j and p ⊤ ℓ t ≤ p ⊤ ℓ j , with at least one inequality strict (as if both were equalities, then j ∈ S ). So, summing the strict inequality with the other inequality, we get that ( p ′ + p ) ⊤ ℓ t &lt; ( p ′ + p ) ⊤ ℓ j = ⇒ ( p ′ + p ) 2 ⊤ ℓ t &lt; ( p ′ + p ) 2 ⊤ ℓ j = ⇒ q ⊤ ℓ t &lt; q ⊤ ℓ j , which contradicts our supposition that j ∈ γ ( q ) . Thus, j ∈ S = ⇒ γ ( q ) ⊆ S .

̸

Lemma 9. Let L : R d → R n be a convex, differentiable surrogate. Let ℓ : R → R n . Let u ∈ R d , such that, γ ( p ) ∩ γ ( p ′ ) = ∅ for any p, p ′ ∈ Γ u . Let p m ∈ Γ u : | γ ( p m ) | ≤ | γ ( p ) | for every p ∈ Γ u . Then, γ ( p m ) ⊆ γ ( p ) , ∀ p ∈ Γ u .

̸

Proof. Suppose not. We know that γ ( p m ) ∩ γ ( p ) = ∅ , so ∃ S ⊆ R : γ ( p m ) ∩ γ ( p ) = S = ∅ . Clearly, S ⊂ γ ( p m ) and S ⊂ γ ( p ) , since S = γ ( p m ) . Now, pick q = p + p m 2 . Since Γ u is convex, q ∈ Γ u . Now, we know from Lemma 8 that γ ( q ) ⊆ S = ⇒ | γ ( q ) | ≤ | S | &lt; | γ ( p m ) | = ⇒ | γ ( q ) | &lt; | γ ( p m ) | . However, since q ∈ Γ u , this yields a contradiction.

̸

̸

Lemma 10. Let L : R d → R n be a convex, differentiable surrogate. Let ℓ : R→ R n . L indirectly elicits ℓ if and only if ∀ u ∈ R d , it holds that γ ( p ) ∩ γ ( p ′ ) = ∅ , for any p, p ′ ∈ Γ u .

̸

Proof. We first show the = ⇒ direction. Since L indirectly elicits ℓ , it holds that ∀ u ∈ R d , ∃ r ∈ R , such that Γ u ⊆ γ r . Pick any p, p ′ ∈ Γ u . Clearly, p, p ′ ∈ γ r = ⇒ r ∈ γ ( p ) ∩ γ ( p ′ ) = ⇒ γ ( p ) ∩ γ ( p ′ ) = ∅ , for any p, p ′ ∈ Γ u .

̸

We now prove the ⇐ = direction. Suppose that for any u ∈ R d , it holds that γ ( p ) ∩ γ ( p ′ ) = ∅ for any p, p ′ ∈ Γ u . Let p m ∈ Γ u : | γ ( p m ) | ≤ | γ ( p ) | , ∀ p ∈ Γ u . We know from Lemma 9, that γ ( p m ) ⊆ γ ( p ) , ∀ p ∈ Γ u . This implies that ∃ r ∈ R : r ∈ γ ( p m ) = ⇒ r ∈ γ ( p ) , ∀ p ∈ Γ u = ⇒ ∃ r ∈ R : p ∈ γ r , ∀ p ∈ Γ u = ⇒ ∃ r ∈ R : Γ u ⊆ γ r .

Lemma 11. Let L : R d → R n be a convex, differentiable surrogate. Let C u ⊆ Γ u be the set of corners of Γ u . Let ℓ : R→ R n . Suppose r ∈ R . Then, p ∈ γ r for every p ∈ C u ⇐⇒ Γ u ⊆ γ r .

Proof. Recall from Lemma 6 that for any u ∈ R d , Γ u is a polytope. Thus, a finite set C u ⊆ Γ u exists such that C u are the corners of Γ u . Say ∃ r ∈ R , such that p ∈ γ r , ∀ p ∈ C u . Pick any q ∈ Γ u . Clearly, q ∈ conv ( C u ) ⊆ γ r , as γ r is convex by Lemma 7. Thus, q ∈ γ r = ⇒ Γ u ⊆ γ r . For the reverse direction, suppose Γ u ⊆ γ r . Let p ∈ C u ⊆ Γ u . Then p ∈ γ r .

Lemma 12. Let L : R d → R n be a convex, differentiable surrogate. Let C u ⊆ Γ u be the set of corners of Γ u for any u ∈ R d . Let ℓ : R → R n . If ∃ S ⊆ R : γ ( p ) = S for every p ∈ C u then L strongly indirectly elicits ℓ .

̸

Proof. Suppose γ ( p ) = S ⊆ R , for every p ∈ C u . This implies that p ∈ relint ( ∩ r ∈ S γ r ) for each p ∈ C u = ⇒ C u ⊆ relint ( ∩ r ∈ S γ r ) . Since γ r is a convex polytope for each r ∈ S , the set ∩ r ∈ S γ r is convex and so is the set relint ( ∩ r ∈ S γ r ) . Hence, conv ( C u ) ⊆ relint ( ∩ r ∈ S γ r ) = ⇒ Γ u ⊆ relint ( ∩ r ∈ S γ r ) = ⇒ p ∈ relint ( ∩ r ∈ S γ r ) , ∀ p ∈ Γ u = ⇒ γ ( p ) = S, ∀ p ∈ Γ u = ⇒ γ ( p ) = γ ( p ′ ) , ∀ p, p ′ ∈ Γ u = ⇒ strong indirect elicitation is satisfied.

Proposition 1. Let L : R d → R n be a convex, differentiable surrogate and let ℓ : R → R n be a target. Let u ∈ R d be a report and suppose C u is the finite set of corners for Γ u . Then the set { γ ( p ) | p ∈ C u } suffices to check both indirect elicitation and strong indirect elicitation at u .

Proof. The proof follows by Lemmas 11 and 12.

## C Properties of convex, differentiable functions

Lemma 13. Let f : R d → R be a convex, differentiable function. Then argmin u ∈ R d f ( u ) is convex.

Proof. If argmin u ∈ R d f ( u ) = ∅ , the result follows vacuously. Else, suppose f ∗ = min u ∈ R d f ( u ) and that x, y ∈ argmin u ∈ R d f ( u ) . Then for any λ ∈ [0 , 1] , f ( λx +(1 -λ ) y ) ≤ λ · f ( x )+(1 -λ ) · f ( y ) = λ · f ∗ +(1 -λ ) · f ∗ = f ∗ = ⇒ λ · x +(1 -λ ) · y ∈ argmin u ∈ R d f ( u ) , ∀ λ ∈ [0 , 1] .

Lemma 14. Let f : R d → R be a convex finite function on R d . Then argmin u ∈ R d f ( u ) is closed. If argmin u ∈ R d f ( u ) is bounded, argmin u ∈ R d f ( u ) is compact.

Proof. By [Rockafellar, 1970, Corollary 10.1.1] f is continuous. By [Hiriart-Urruty and Lemaréchal, 1996, Prop 3.2.2] every sublevel-set of f is closed.

Lemma 15. Let f : R d → R be a convex, differentiable function. Then f is continuously differentiable.

<!-- formula-not-decoded -->

See Corollary 25.5.1 of [Rockafellar, 1970].

Lemma 16. Let f : R d → R be a convex, differentiable function. Suppose also that f is minimizable and that the set argmin u ∈ R d f ( u ) is bounded. Let U ∗ := argmin u ∈ R d f ( u ) and let f ∗ := min u ∈ R d f ( u ) . Then, for δ &gt; 0 , it holds that:

<!-- formula-not-decoded -->

Proof. First, notice that since U ∗ is bounded, it is compact by Lemma 14. Thus, ¯ B δ ( U ∗ ) \ B δ ( U ∗ ) := ∂ ¯ B δ ( U ∗ ) is also compact. Since f is differentiable, it is continuous everywhere, and thus f attains its infimum over the compact set ∂ ¯ B δ ( U ∗ ) . This proves that

<!-- formula-not-decoded -->

where the final inequality holds as ∂ ¯ B δ ( U ∗ ) ∩ U ∗ = ∅ . We are left to show that

<!-- formula-not-decoded -->

Clearly, inf u ∈ R d \ B δ ( U ∗ ) f ( u ) ≤ min u ∈ ∂ ¯ B δ ( U ∗ ) f ( u ) , so for the equality to fail, we would need inf u ∈ R d \ ¯ B δ ( U ∗ ) f ( u ) &lt; min u ∈ ∂ ¯ B δ ( U ∗ ) f ( u ) . This, in turn, requires that there exist u ′ ∈ R d \ ¯ B δ ( U ∗ ) such that

<!-- formula-not-decoded -->

Pick any u ∗ ∈ U ∗ and consider the line segment conv( u ∗ , u ′ ) , connecting u ∗ and u ′ . There exists some v ∈ ∂ ¯ B δ ( U ∗ ) such that v ∈ conv( u ∗ , u ′ ) . It holds that

<!-- formula-not-decoded -->

which violates convexity of f since v ∈ conv( u ∗ , u ′ ) , completing the proof.

Lemma 17. Let f : R d → R be a convex, differentiable function. Suppose also that f is minimizable and that the set U ∗ := argmin u ∈ R d f ( u ) is bounded. Let f ∗ := min u ∈ R d f ( u ) . Let { u t } t ∈ N + be a sequence in R d . Then:

<!-- formula-not-decoded -->

Proof. We first show that:

<!-- formula-not-decoded -->

Since U ∗ is bounded, it is compact by Lemma 14. Pick δ &gt; 0 . There exists T δ ∈ N + such that for every t ≥ T δ ,

In particular, for each t ≥ T δ , there exists u ∗ t ∈ U ∗ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies u t ∈ B δ ( U ∗ ) ⊆ ¯ B δ ( U ∗ ) for every t ≥ T δ . Since U ∗ is compact, ¯ B δ ( U ∗ ) is also compact.

Now, f is differentiable and therefore continuous everywhere. Since ¯ B δ ( U ∗ ) is compact, f is uniformly continuous within ¯ B δ ( U ∗ ) . Pick ϵ &gt; 0 . It suffices to show the existence of some T ∈ N + such that

By uniform continuity, there exists δ ϵ &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If δ ϵ ≥ δ , then for any u t with t ≥ T δ , where u ∗ t ∈ U ∗ , implying u t , u ∗ t ∈ ¯ B δ ( U ∗ ) . Thus, for any t ≥ T δ ,

Otherwise, if δ ϵ &lt; δ , pick T δ ϵ ∈ N + such that d ( u t , U ∗ ) &lt; δ ϵ for every t ≥ T δ ϵ . Then, for each t ≥ T δ ϵ , there exists u ∗ t ∈ U ∗ such that

Thus, for all t ≥ T δ ϵ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now prove the reverse direction

<!-- formula-not-decoded -->

Assume to the contrary that this implication does not hold. Then, there exists some δ &gt; 0 such that for every T ∈ N + , there exists t ≥ T such that

This implies the existence of a subsequence { u t j } j ∈ N + such that

For every j ∈ N + ,

By Lemma 16,

Thus, for every j ∈ N + ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This contradicts the assumption that f ( u t ) → f ∗ as t →∞ , completing the proof.

Lemma 18. Let f 1 , f 2 : R d → R be convex finite functions such that S 1 = arg min x ∈ R d f 1 ( x ) and S 2 = arg min x ∈ R d f 2 ( x ) are compact and nonempty. Let g ( x ) = f 1 ( x ) + f 2 ( x ) . Then, S = arg min x ∈ R d g ( x ) is also compact and nonempty.

Proof. By Lemma 14, S is closed, so it suffices to show it is bounded and nonempty. Fix any x 1 ∈ S 1 and x 2 ∈ S 2 . Let y 1 = f 1 ( x 1 ) and y 2 = f 2 ( x 2 ) be the minimia achieved by f 1 and f 2 . Now, choose any x g ∈ R d and let y = g ( x g ) . Note that by construction we must have y ≥ y 1 + y 2 . Therefore, let δ := y -( y 1 + y 2 ) ≥ 0 .

Now, let d 1 = diam( S 1 ) be the maximum distance between any two points in S 1 . Then ∂B 2 d 1 ( x 1 ) , the set of points distance 2 d 1 from x 1 , must be disjoint from S 1 , so f 1 does not achieve its minimum on this set. However, since the set is compact, we can still minimize f 1 on it. Let x ∗ 1 = arg min x ∈ ∂B 2 d 1 ( x 1 ) f 1 ( x ) , and y ∗ 1 = f 1 ( x ∗ 1 ) &gt; y 1 . By convexity, the segment between ( x 1 , y 1 ) and any other point in the epigraph of f 1 must be entirely contained within the epigraph. In particular, for any x outside the ball of radius 2 d 1 , the line connecting ( x 1 , y 1 ) and ( x, f ( x )) must pass through or above ( x ′ , y ∗ 1 ) for some x ′ ∈ ∂B 2 d 1 ( x 1 ) . Essentially, this tells us that outside the 2 d 1 -ball the epigraph of f 1 lies above the cone of slope y ∗ 1 -y 1 2 d 1 centered at x 1 . Algebraically, this means that for any x ̸∈ B 2 d 1 ( x 1 ) , f 1 ( x ) ≥ y ∗ 1 -y 1 2 d 1 ∥ x -x 1 ∥ + y 1 . In particular, if we let r 1 = max(2 d 1 , δ 2 d 1 y ∗ 1 -y 1 ) , then for any x ̸∈ B r 1 ( x 1 ) ,

<!-- formula-not-decoded -->

We can repeat the same process for x 2 and f 2 , letting d 2 = diam( S 2 ) , x ∗ 2 = arg min x ∈ ∂B 2 d 2 ( x 2 ) f 2 ( x ) , and y 2 = f 2 ( x ∗ 2 ) , and r 2 = max(2 d 2 , δ 2 d 2 y ∗ 2 -y 2 ) , we have for any x ̸∈ B r 2 ( x 2 ) ,

<!-- formula-not-decoded -->

Let B = B r 1 ( x 1 ) ∪ B r 2 ( x 2 ) . Combining the previous two equations, we have that for any x ̸∈ B ,

<!-- formula-not-decoded -->

Recall that y = g ( x g ) was chosen arbitrarily. In particular, we must have inf x ∈ R d g ( x ) ≤ y . Therefore, g can achieve its minimum only on B , so we can equivalently define S = arg min x ∈ B g ( x ) . Finally, since B is bounded, S must be as well, and since it is closed the argmin of g must be achieved somewhere, so S is nonempty.

Theorem 8. Let L : R d → R n . If for each y ∈ [ n ] , L ( · ) y : R d → R is convex, and arg min u ∈ R d L ( u ) y is non-empty and compact, then Γ( p ) is non-empty and compact for each p ∈ ∆ n .

Proof. Pick any p ∈ ∆ n . Then, for each y ∈ [ n ] , p y · L ( · ) y is convex. Further, notice that since p y · L ( · ) y is just L ( · ) y scaled by a positive scalar, it follows that arg min u ∈ R d p y · L ( · ) y = arg min u ∈ R d L ( · ) y , and thus, arg min u ∈ R d p y · L ( · ) y is non-empty and compact for each y ∈ [ n ] . Applying Lemma 18 inductively, it follows that arg min u ∈ R d ⟨ p, L ( · ) ⟩ is non-empty and compact. Since we picked p arbitrarily, it follows that Γ( p ) is non-empty and compact for each p ∈ ∆ n .

## D One-Dimensional Surrogate Losses

Definition 7. (Orderable) [Lambert, 2011] A finite property γ : ∆ n ⇒ R is orderable , if there is an enumeration of R = { r 1 , r 2 , ..., r k } such that for all i ≤ k -1 , we have γ r j ∩ γ r j +1 is a hyperplane intersected with ∆ n . We say that the ordered tuple E γ := ( r 1 , r 2 , ..., r k ) is the enumeration associated with R .

Without loss of generality, we assume for the rest of this section that for any finite orderable property γ , it holds that, γ j ∩ γ j +1 is a hyperplane intersected with ∆ n , ∀ j ∈ [ k -1] . In particular, the enumeration associated with R will always assumed to be E γ = (1 , 2 , ..., k -1 , k ) .

Theorem 9. [Finocchiaro et al., 2020] If a convex surrogate loss L : R → R n indirectly elicits a target loss ℓ : R→ R n , then the property γ = prop[ ℓ ] is orderable.

̸

Definition 8. (Intersection Graph) [Finocchiaro et al., 2020] Given a discrete loss ℓ : R → R n and associated finite property γ = prop [ ℓ ] , the intersection graph has vertices R and edges ( r, r ′ ) if γ r ∩ γ r ′ ∩ relint (∆ n ) = ∅ .

Lemma 19. [Finocchiaro et al., 2020] A finite property γ is orderable if and only if its intersection graph is a path, i.e., a connected graph where two nodes have degree 1 and all other nodes have degree 2 .

Lemma 20. Let L : R → R n be a convex, differentiable surrogate and suppose ℓ : R → R n . If L indirectly elicits ℓ , then there exist disjoint sets I 1 , I 2 , ..., I k -1 ⊂ R d , where for each j ∈ [ k -1] , I j := { u ∗ ∈ R | Γ u ∗ = γ j ∩ γ j +1 } . For each j ∈ [ k -1] , the set I j is either a singleton { u j } or a closed compact interval [ u j, 1 , u j, 2 ] .

̸

For the reverse inclusion, assume to the contrary that Γ u j ⊂ T j . This means that Γ u j must have an extremal point p ∈ relint ( T j ) , which in turn means that p ∗ ∈ relint (∆ n ) . However, Γ u j is the intersection of a subspace with ∆ n , so its extremal points must be on the boundary of ∆ n . Thus, Γ u j = T j .

Proof. Since L indirectly elicits ℓ , it follows by Theorem 9 that γ is orderable. Thus, for each j ∈ [ k -1] , γ j ∩ γ j +1 is a hyperplane intersected with ∆ n . Denote T j := γ j ∩ γ j +1 . By the nonredundancy of target reports, it holds for any j ∈ [ k -1] that affdim ( γ j ) = n -1 , affdim ( T j ) = n -2 and that relint ( T j ) ⊂ relint(∆ n ) . Fix j ∈ [ k -1] . Suppose p ∈ relint ( T j ) = ⇒ p ∈ relint(∆ n ) . By minimizability of ⟨ p, L ( · ) ⟩ , there exists a u j ∈ R , such that p ∈ Γ u j . Since L indirectly elicits ℓ , it follows from Lemma 4 that rank ( ∇ L ( u j )) = 1 . Further, since p ∈ Γ u j ∩ relint (∆ n ) , it follows that affdim (Γ u j ) = n -2 . Now, we claim that Γ u j = T j . We first show that Γ u j ⊆ T j . Assume to the contrary. Then, ∃ p ′ ∈ Γ u j , such that p ′ / ∈ T j . Since Γ u j is convex, it follows that conv ( p ′ , p ) ⊆ Γ u j . Since γ is orderable, we know from Lemma 19 that no 3 target cells intersect in relint (∆ n ) . Then, since p lies on the interior of the common boundary between ( n -2 dimensional) polytopes γ j and γ j +1 , there must be a sufficiently small ϵ &gt; 0 , such that B ϵ ( p ) ∩ ∆ n is fully contained in relint ( γ j ) in one halfspace defined by the hyperplane T j and is fully contained in relint ( γ j +1 ) in the other halfspace defined by T j . It follows that ∃ q ∈ conv ( p ′ , p ) , such that q ∈ relint ( γ j ) or q ∈ relint ( γ j +1 ) . Suppose WLOG that q ∈ relint ( γ j ) . This means that ∃ q ∈ Γ u j : γ ( q ) = { j } . So, by indirect elicitation, it must hold that Γ u j ⊆ γ j and that Γ u j ̸⊆ γ r for any r = j . However, since Γ u j is the intersection of the subspace ker ( ∇ L ( u j ) ⊤ ) with ∆ n , Γ u j must have extremal points at the boundaries of ∆ n . This means that Γ u j cannot terminate at p and must extend beyond p into γ j +1 . Thus, ∃ q ′ ∈ relint ( γ j +1 ) , such that q ′ ∈ Γ u j . This violates Γ u j ⊆ γ j and hence violates indirect elicitation. So, Γ u j ⊆ T j .

̸

Now, define I j := { u ∗ ∈ R | Γ u ∗ = γ j ∩ γ j +1 } . It follows that u ∗ ∈ I j = ⇒ u ∗ ∈ Γ( p ) since p ∈ γ j ∩ γ j +1 . Thus, I j ⊆ Γ( p ) . However, choosing u j ∈ Γ( p ) arbitrarily, we proved that Γ u j = γ j ∩ γ j +1 . Thus, Γ( p ) ⊆ I j . Therefore, I j = Γ( p ) which always exists by minimizability of ⟨ p, L ( · ) ⟩ . Further, since Γ( p ) is compact (by assumption), it follows that I j is either a singleton, or a compact interval in R . Similarly, by picking j ′ ∈ [ k -1] : j ′ = j , we can establish the existence of (a singleton or compact interval set) I j ′ , such that Γ u j ′ = γ j ′ ∩ γ j ′ +1 . To see that, I j ∩ I j ′ = ∅ , ∀ j, j ′ ∈ [ k -1] : j = j ′ , assume to the contrary that there exists some v ∈ I j ∩ I j ′ . Then, Γ v = γ j ∩ γ j +1 = γ j ′ ∩ γ j ′ +1 . However, γ j ∩ γ j +1 = γ j ′ ∩ γ j ′ +1 for any j = j ′ and so the sets I j and I ′ j must be disjoint.

̸

̸

̸

Throughout the rest of this section, we will inherit from Lemma 20, the notation I j for the set { u ∗ ∈ R | Γ u ∗ = γ j ∩ γ j +1 }

Lemma 21. Let L : R → R n be a convex, differentiable surrogate and suppose ℓ : R → R n . Suppose L indirectly elicits ℓ . Let p, q ∈ ∆ n : γ ( p ) = { j } and γ ( q ) = { j +1 } , where j ∈ [ k -1] . Let u p ∈ Γ( p ) and u q ∈ Γ( q ) . Then, u p &lt; min ( I j ) ≤ max ( I j ) &lt; u q , or u q &lt; min ( I j ) ≤ max ( I j ) &lt; u p .

Proof. Fix j ∈ [ k -1] . Pick any u j ∈ I j . For p : γ ( p ) = { j } , q : γ ( q ) = { j + 1 } , and u p ∈ Γ( p ) , u q ∈ Γ( q ) , we will show that exactly one of the following holds: u p &lt; u j &lt; u q or

u q &lt; u j &lt; u p . Now, from the definition of I j and from Lemma 3, we know that Γ u j = γ j ∩ γ j +1 = { p ′ ⊆ ∆ n |∇ L ( u j ) ⊤ p ′ = 0 } . Since p ∈ relint ( γ j ) and q ∈ relint ( γ j +1 ) , it follows that p and q are on different sides of the hyperplane { x ∈ R n |∇ L ( u j ) ⊤ x = 0 } . So, it must hold that ∇ L ( u j ) ⊤ p &lt; 0 and ∇ L ( u j ) ⊤ q &gt; 0 , or that ∇ L ( u j ) ⊤ p &gt; 0 and ∇ L ( u j ) ⊤ q &lt; 0 .

̸

If instead ∇ L ( u j ) ⊤ p &gt; 0 and ∇ L ( u j ) ⊤ q &lt; 0 , it would follow that u p &lt; min ( I j ) ≤ max ( I j ) &lt; u q by the same argument.

Assume WLOG that ∇ L ( u j ) ⊤ p &lt; 0 and ∇ L ( u j ) ⊤ q &gt; 0 . Now, notice that ∇⟨ p, L ( · ) ⟩ = ∇ L ( · ) ⊤ p and similarly, ∇⟨ q, L ( · ) ⟩ = ∇ L ( · ) ⊤ q . Since the function ⟨ p, L ( · ) ⟩ is convex and differentiable, it holds from the monotonicity of gradients of convex functions that, ⟨∇ L ( u j ) ⊤ p -∇ L ( u p ) ⊤ p, u j -u p ⟩ ≥ 0 = ⇒ ⟨∇ L ( u j ) ⊤ p, u j -u p ⟩ ≥ 0 = ⇒ u j -u p ≤ 0 , since ∇ L ( u j ) ⊤ p &lt; 0 . Clearly, u j = u p and so, u j -u p &lt; 0 . Similarly, we can show that u j -u q &gt; 0 . We have thus shown that u q &lt; u j &lt; u p . Since u j was arbitrarily chosen from I j , it holds that u q &lt; min ( I j ) ≤ max ( I j ) &lt; u p . Now, for j ∈ R , denote U j := { u ∈ R : u ∈ Γ( p ) , γ ( p ) = { j }} .

Denote by U j := { u ∈ R | u ∈ Γ( p ) , γ ( p ) = { j }} . Let u 1 ∈ U 1 , u 2 ∈ U 2 , ..., u k ∈ U k . Then by Lemma 21, we know that, u 1 &lt; min( I 1 ) ≤ max( I 1 ) &lt; u 2 &lt; min( I 2 ) ≤ max( I 2 ) &lt; u 3 ... &lt; u k -2 &lt; min( I k -2 ) ≤ max( I k -2 ) &lt; u k -1 &lt; min( I k -1 ) ≤ max( I k -1 ) &lt; u k , or, u 1 &gt; max( I 1 ) ≥ min( I 1 ) &gt; u 2 &gt; max( I 2 ) ≥ min( I 2 ) &gt; u 3 ... &gt; u k -2 &gt; max( I k -2 ) ≥ min( I k -2 ) &gt; u k -1 &gt; max( I k -1 ) ≥ min( I k -1 ) &gt; u k . This means, that either max( I 1 ) &lt; min( I k -1 ) , or max( I k -1 ) &lt; min( I 1 ) . We assume WLOG from hereon, that max( I 1 ) &lt; min( I k -1 ) . Thus, we have shown that min( I j ) is a uniform, strict upper bound on U j and that max( I j -1 ) is uniform, strict lower bound on U j .

Theorem 10. Let L : R → R n be a convex, differentiable surrogate and suppose ℓ : R → R n . Under Assumption 1, L is calibrated with respect to ℓ . Then ( L, ψ ) is calibrated with respect to ℓ , for any link ψ : R →R , that satisfies

<!-- formula-not-decoded -->

̸

Proof. For the entirety of this proof, we define γ 0 = γ k +1 = I 0 = I k +1 = ∅ . Let ψ : R →R be any link of the form proposed in the theorem statement. We first show calibration for p ∈ ∆ n : γ ( p ) = { j } for some j ∈ R . We know from Lemma 21, that max( I j -1 ) &lt; u p &lt; min( I j ) for any u p ∈ Γ( p ) . We have that ψ ( u ) = j for any u &lt; min ( I j -1 ) and any u &gt; max ( I j ) . Whereas, u ∈ I j -1 can be linked to either j -1 or j , and u ∈ I j can be linked to either j or j +1 . The remaining reports always link to j .

̸

<!-- formula-not-decoded -->

The final equality follows from convexity of the function ⟨ p, L ( · ) ⟩ . The final strict inequality follows from the fact that max( I j -1 ) &lt; u p &lt; min( I j ) for any u p ∈ Γ( p ) . The same argument holds for any other distribution p ′ : γ ( p ′ ) = { j } . In fact, since j was picked arbitrarily from R , the argument extends to any q : γ ( q ) = { j ′ } , for any j ′ ∈ R . So, we have established calibration at all distribution lying in the relative interiors of target cells. We still need to prove calibration for distributions lying on target boundaries.

Again, start by fixing some j ∈ [ k -1] . Suppose p ∈ relint ( γ j ∩ γ j +1 ) . Then, since γ is orderable, we know from Lemma 19 that no 3 target cells can intersect in the relative interior of the simplex. Thus, γ ( p ) = { j, j +1 } . We have that ψ ( u ) / ∈ { j, j +1 } for any u &lt; min ( I j -1 ) and any u &gt; max ( I j +1 ) . Whereas, u ∈ I j -1 can be linked to either j -1 or j and u ∈ ∪ I j +1 can be linked to either j or

j +1 . The remaining reports always link to one of j -1 or j .

The final equality follows from convexity of the function ⟨ p, L ( · ) ⟩ . The final strict inequality follows by noting that γ j -1 ∩ γ j and γ j +1 ∩ γ j +2 are disjoint from relint ( γ j ∩ γ j +1 ) and hence max ( I j -1 ) and min ( I j +1 ) are suboptimal for p ∈ relint ( γ j ∩ γ j +1 ) . The same argument extends to all distributions lying in the relative interiors of target boundaries. Thus, we have established calibration for all distributions, barring distributions on target boundaries that are not inside the relative interiors of the boundaries.

<!-- formula-not-decoded -->

Suppose p ∈ ∆ n , such that p lies on some target boundary, however, p does not lie in the relative interior of the target boundary. Such points can lie within the intersection of 2 or more target cells. In case, p ∈ ∩ j ∈R γ j , γ ( p ) = R and calibration follows trivially, since the set u ∈ R : ψ ( u ) / ∈ γ ( p ) is empty. Otherwise, suppose, γ ( p ) = S , for some S ⊂ R . It must be that discrete reports within S are consecutive integers. That is, suppose j ∈ S and j ′ ∈ S . Then, either | j -j ′ | ≤ 1 , or else, if | j -j ′ | &gt; 1 , then j ∗ ∈ S for any j ∗ : min { j, j ′ } &lt; j ∗ &lt; max { j, j ′ } . This follows from the fact that γ is orderable, and we are assuming the enumeration associated with R is E γ = (1 , 2 , ..., k -1 , k ) . Thus, suppose that γ ( p ) = S ⊂ R : S = { j, j +1 , ..., j + t } , where t ≥ 1 . We have that ψ ( u ) / ∈ S for any u &lt; min( I j -1 ) and for any u &gt; max( I j + t ) . Whereas, u ∈ I j -1 may be linked to either j -1 / ∈ S or j ∈ S , and u ∈ I j + t may be linked to either j + t ∈ S or j + t +1 / ∈ S . The remaining surrogate reports always link to some discrete report in S .

<!-- formula-not-decoded -->

The final equality follows from convexity of the function ⟨ p, L ( · ) ⟩ . The final strict inequality follows by noting that p / ∈ γ j -1 ∩ γ j and p / ∈ γ j + t ∩ γ j + t +1 by construction. Thus, ( L, ψ ) is calibrated at p . The same argument extends to any distribution that lies on some target boundary, but not its relative interior.

## E Correspondences

We consolidate basic definitions and results about correspondences in this section. Note that different authors can have slightly differing terminology and conventions related to correspondences. We direct the reader to Border [2013] and references therein for a more detailed discussion on different conventions. We adopt the conventions, definitions and terminology used in Border [2013].

Definition 9. (Correspondence, graph, image, domain) [Border, 2013] A correspondence φ from X to Y associates to each point in X a subset φ ( x ) of Y . We write this as φ : X ⇒ Y . For a correspondence φ : X ⇒ Y , let gr φ denote the graph of φ , which we define to be

<!-- formula-not-decoded -->

Let φ : X ⇒ Y , and let F ⊂ X . The image φ ( F ) of F under φ is defined to be

<!-- formula-not-decoded -->

The value φ ( x ) is allowed to be the empty set, but we call { x ∈ X : φ ( x ) = ∅} the domain of φ , denoted dom φ .

̸

The terms multifunction, point-to-set mapping , and set-valued function are also used for a correspondence.

Definition 10. (Metric upper hemicontinuity) [Border, 2013] Let X be a metric space equipped with the metric d X : X × X → R . A correspondence φ : X ⇒ Y is said to satisfy metric upper hemicontinuity at a point x ∈ X if for every ϵ &gt; 0 , ∃ δ &gt; 0 , such that

<!-- formula-not-decoded -->

Metric upper hemicontinuity is a special case of the more general, topological notion of upper hemicontinuity which requires that the pre-image of open neighborhoods of φ ( x ) be open sets (see [Border, 2013] for a formal definition). However, metric upper hemicontinuity at x and the topological notion of upper hemicontinuity at x are equivalent when φ ( x ) is compact (see Proposition 11 of [Border, 2013]). For our purposes, this will always be the case. So, we simply work with the simpler, metric based notion of upper hemicontinuity.

Definition 11. (Closed at x , Closed correspondence) [Border, 2013] The correspondence φ : X ⇒ Y is closed at x ∈ X if whenever x n → x , y n ∈ φ ( x n ) , and y n → y , then y ∈ φ ( x ) . A correspondence is closed if it is closed at every point of its domain, that is, if its graph is closed.

Lemma 22. [Border, 2013] If the correspondence φ : X ⇒ Y is closed at x ∈ X , then φ ( x ) is a closed set.

Lemma 23. [Border, 2013] Suppose Y is compact and φ : X ⇒ Y is closed at x ∈ X , then φ is upper hemicontinuous at x .

We say a correspondence φ : X ⇒ Y is compact-valued if φ ( x ) is compact for every x ∈ X .

Lemma 24. [Border, 2013] Let K ⊂ X be a compact set and suppose φ : X ⇒ Y is upper hemicontinuous and compact-valued. Then φ ( K ) is compact.

## F Sufficiency of Strong Indirect Elicitation

We start by presenting a couple of helper results, that we will leverage in our main proofs.

Lemma 25. Lebesgue's Number Lemma : (see Thm. IV.5.4 of Hu [1966]) Let ( X , d ) be a compact metric space. Let A be an arbitrary index set, and U = ∪ α ∈A U α be an open cover of X . Then, there exists a number δ L &gt; 0 , such that for any x ∈ X , B δ L ( x ) ⊂ U α x , where α x ∈ A

Given an open cover U compact set X , a constant δ L &gt; 0 that satisfies the condition of Lemma 25 is known as a Lebesgue Number for the cover [Hu, 1966]. We state and prove a simply corollary of Lemma 25, which we will make use of later.

Corollary 1. Let m ∈ Z + , δ L &gt; 0 be a Lebesgue number for some open cover U = ∪ α ∈A U α of a compact set X ⊆ R m . Then, B δ L / 2 ( X ) ⊆ U

Proof. Let v ∈ B δ L / 2 ( X ) . This means, there exist u ∈ X and b ∈ B δ L / 2 ( 0 m ) , such that, v = u + b , i.e., v ∈ B δ L / 2 ( u ) ⊂ B δ L ( u ) . Now, by Lemma 25, ∃ α ∈ A : B δ L ( u ) ⊂ U α ⊆ U . So, v ∈ U . Since v was arbitrarily chosen from B δ L / 2 ( X ) , it follows that B δ L / 2 ( X ) ⊆ U

Figure 7: Visual intuition for the proof of sufficiency of strong IE. Let n = 3 , Y = { 1 , 2 , 3 } and R = { 1 , 2 , 3 } ( white block with blue border - right ). Let p ∈ ∆ n ( black point within triangle left ). Γ Γ( p ) is the level-set bundle at p ( dark purple region - left ). Here, γ ( p ) = { 2 } . Lemma 29 ensures the existence of ϵ p &gt; 0 : The image of Γ( B ϵ p (Γ Γ( p ) )) ( gray region - center ) under Γ ( · ) is fully contained within γ 2 ( gray region - left ). By Lemma 27, ∃ δ p &gt; 0 : Γ(∆ n ) ∩ B δ p (Γ( p )) ( light green region - center ) is contained within Γ( B ϵ p (Γ Γ( p ) )) (gray region - center). Thus, every report within the light green region necessarily links to γ ( p ) ( red block - right ). So any sequence of reports { u t } t ∈ N + ⊆ Γ(∆ n ) ( light pink region - center ), for which lim t →∞ u t → Γ( p ) , must eventually link to γ ( p ) . Further, Theorem 11 shows how to link nowhere optimal reports without violating calibration. Thus, Γ( p ) is 'protected' in the calibration sense.

<!-- image -->

Suppose L : R d → R n is a convex, differentiable surrogate loss. We assume throughout that for every p ∈ ∆ n , Γ( p ) := arg min u ∈ R d ⟨ p, L ( u ) ⟩ exists and that Γ( p ) is compact. Note that, since the function ⟨ p, L ( · ) ⟩ is convex, Γ( p ) is closed by Lemma 14 . Thus, it suffices to say Γ( p ) is bounded, though we will usually say compact for clarity.

To prove the sufficiency of strong indirect elicitation, we first establish basic properties about the surrogate level sets { Γ u | u ∈ R d } . To do so, we adjust our lens slightly, and view Γ u as the image of a correspondence at a point u in its domain. In particular, we denote Γ ( · ) : R d ⇒ ∆ n as the correspondence that maps surrogate reports to the set of distributions they optimize.

Lemma 26. The correspondence Γ ( · ) : R d ⇒ ∆ n is closed at every u ∈ Γ(∆ n ) . In fact, Γ ( · ) is upper hemicontinuous at every u ∈ Γ(∆ n ) .

̸

Proof. Suppose u ∈ Γ(∆ n ) = ⇒ Γ u = ∅ . Let { u i } i ∈ N be a sequence in R d and let { p i } i ∈ N be a sequence in ∆ n such that p i ∈ Γ u i ∀ i ∈ N and suppose p i → p ∈ ∆ n and u i → u ∈ R d . To prove Γ ( · ) is closed, we need to show p ∈ Γ u (see Definition 11). Let i ∈ N . Since p i ∈ Γ u i , ∇ L ( u i ) T p i = 0 d . Now, notice that:

<!-- formula-not-decoded -->

Observe that lim i →∞ p i = p (by construction), and lim i →∞ ∇ L ( u i ) = ∇ L ( u ) (since u i → u by construction, and since ∇ L ( · ) is continuous). Thus,

<!-- formula-not-decoded -->

We have thus shown that Γ ( · ) is closed at u . Since the target space, ∆ n is compact, the fact that Γ ( · ) is closed at u implies that Γ ( · ) is upper hemicontinuous at u (by Lemma 23).

Lemma 27. Let p ∈ ∆ n . For every ϵ &gt; 0 , there exists δ &gt; 0 , such that:

<!-- formula-not-decoded -->

Proof. Pick any p ∈ ∆ n . Since Γ ( · ) is upper hemicontinuous at u ∈ Γ( p ) by Lemma 26, we have that for every ϵ &gt; 0 , ∃ δ u &gt; 0 for each u ∈ Γ( p ) such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So, we have shown that ∪ u ∈ Γ( p ) Γ B δu ( u ) ⊆ B ϵ (Γ Γ( p ) ) . It follows by Lemma 2 that:

<!-- formula-not-decoded -->

Now, suppose u ′ ∈ Γ(∆ n ) ∩ B δ u ( u ) . Let p ′ ∈ Γ u ′ . Then, p ′ ∈ Γ u ′ ⊆ Γ B δu ( u ) . So u ′ ∈ Γ( p ′ ) ⊆ Γ(Γ B δu ( u ) ) . Since u ′ was picked arbitrarily in Γ(∆ n ) ∩ B δ u ( u ) , it follows that:

<!-- formula-not-decoded -->

The final inclusion, i.e., ∪ u ∈ Γ( p ) Γ(Γ B δu ( u ) ) ⊆ Γ( ∪ u ∈ Γ( p ) Γ B δu ( u ) ) , holds by the following rationale: Suppose v ∈ ∪ u ∈ Γ( p ) Γ(Γ B δu ( u ) ) . This means ∃ u ′ ∈ Γ( p ) : v ∈ Γ(Γ B δ u ′ ( u ′ ) ) = ⇒ ∃ p v ∈ Γ B δ u ′ ( u ′ ) : v ∈ Γ( p v ) . Now, since p v ∈ Γ B δ u ′ ( u ′ ) , p v ∈ ∪ u ∈ Γ( p ) Γ B δu ( u ) as u ′ ∈ Γ( p ) . Thus, v ∈ Γ( ∪ u ∈ Γ( p ) Γ B δu ( u ) ) . So, we have shown that:

<!-- formula-not-decoded -->

Observing that the RHS of inclusion (4) and the LHS of inclusion (3) are the same, we combine the inclusions to get that:

<!-- formula-not-decoded -->

Let us denote ∪ u ∈ Γ( p ) B δ u ( u ) by U . Now, notice that Γ( p ) ⊆ U . Further, U is a union of open sets and is thus open itself. This means, U is an open cover of Γ( p ) . Since Γ( p ) is compact, due to Lemma 25, there exists a Lebesgue number δ L &gt; 0 for U . Set δ := δ L / 2 . Then, by Corollary 1, B δ (Γ( p )) ⊆ U .

where the final inclusion follows from inclusion (5), thus concluding our proof.

<!-- formula-not-decoded -->

Lemma 28. For any p ∈ ∆ n , suppose Γ( p ) is a non-empty, compact set. Then, the set Γ Γ( p ) = ∪ u ∈ Γ( p ) Γ u is closed.

Proof. Fix p ∈ ∆ n . We know from Lemma 26 that Γ ( · ) is closed at any u ∈ Γ( p ) . This implies that for any u ∈ Γ( p ) , Γ u is a closed set. Since the target space ∆ n is compact, Γ u must be bounded, and thus Γ u is compact for each u ∈ Γ( p ) . So the restriction of Γ ( · ) to Γ( p ) , i.e., Γ ( · ) | Γ( p ) : Γ( p ) ⇒ ∆ n is upper hemicontinuous and compact-valued, and so by Lemma 24, Γ Γ( p ) is compact since Γ( p ) is compact by assumption. Thus, Γ Γ( p ) is closed.

Lemma 29. Let L : R d → R n be a convex, differentiable surrogate. Suppose Assumption 1 holds . Let ℓ : R→ R n be a discrete target loss, with finite property γ := prop[ L ] . If L strongly indirectly elicits ℓ , then the following holds: For each p ∈ ∆ n :

<!-- formula-not-decoded -->

Proof. Let p ∈ ∆ n . Define S := ∪ r ′ ∈R\ γ ( p ) γ r ′ . If S is empty, then γ ( p ) = R and γ ( p ′ ) ⊆ γ ( p ) , ∀ p ∈ ∆ n and then the result follows trivially. Otherwise, notice that γ ( q ) = γ ( p ) for any q ∈ Γ Γ( p ) , since ∃ v ∈ Γ( p ) : q ∈ Γ v and then γ ( p ) = γ ( q ) by strong indirect elicitation. Thus, γ ( q ) ∩ S = ∅ , ∀ q ∈ Γ Γ( p ) , and so S ∩ Γ Γ( p ) = ∅ . Recall that the target cells, i.e., γ r : r ∈ R are closed by virtue of being convex polytopes as shown in Lemma 7. So, S is a finite union of closed sets, and is thus closed. In fact S is compact since S ⊆ ∆ n . Also, Γ Γ( p ) is closed by Lemma 28. Similarly, Γ Γ( p ) is compact since Γ Γ( p ) ⊆ ∆ n . Since S ∩ Γ Γ( p ) = ∅ , and both S and Γ Γ( p ) are compact and , it holds that d ( S, Γ Γ( p ) ) &gt; 0 . Thus, ∃ ϵ p &gt; 0 : ∀ p ′ ∈ B ϵ p (Γ Γ( p ) ) , p ′ / ∈ S and so:

<!-- formula-not-decoded -->

Now, suppose v ∈ Γ( B ϵ p (Γ Γ( p ) )) . So, there exists p ′ ∈ Γ v , such that p ′ ∈ B ϵ p (Γ Γ( p ) ) = ⇒ γ ( p ′ ) ⊆ γ ( p ) due to (6). Now, for any q ∈ Γ v , it must hold that γ ( q ) = γ ( p ′ ) by strong indirect elicitation. Therefore, γ ( q ) ⊆ γ ( p ) , ∀ q ∈ Γ v . The result follows since v was arbitrarily chosen from Γ( B ϵ p (Γ Γ( p ) )) .

Lemma 30. Let L : R d → R n be a convex, differentiable surrogate. Suppose Assumption 1 holds. Then, Γ(∆ n ) is closed.

Proof. We show that for any sequence { u t } t ∈ N + such that u t ∈ Γ(∆ n ) for each t ∈ N + , if u t → u , then it holds that u ∈ Γ(∆ n ) . Since u t ∈ Γ(∆ n ) , ∃ p t ∈ ∆ n , such that p t ∈ Γ u t for each t ∈ N + . Now, since ∆ n is compact, we can extract a subsequence p t j , such that p t j → p ∈ ∆ n and that u t j → u . We will show that u ∈ Γ( p ) . Consider:

<!-- formula-not-decoded -->

Now, as j → ∞ , ∥∇ L ( u t j ) - ∇ L ( u ) ∥ → 0 as u t j → u (by the continuity of ∇ L ( · ) ). Also, ∥ p t j -p ∥ → 0 by construction. Hence, ∇ L ( u t j ) ⊤ p t j → ∇ L ( u ) ⊤ p . However, since p t j ∈ Γ u t j , it holds that ∇ L ( u t j ) ⊤ p t j = 0 d = ⇒ ∇ L ( u ) ⊤ p = 0 d = ⇒ u ∈ Γ( p ) = ⇒ u ∈ Γ(∆ n ) , thus concluding our proof.

We are now ready to state and prove our main link construction, which in turn proves the sufficiency of strong IE for calibration under our assumptions. Throughout the proof, let dist ( a, B ) := inf b ∈ B ∥ a -b ∥ 2 , and when the infimum is attained, let proj B ( a ) := { b ∈ B | ∥ a -b ∥ 2 = dist ( a, B ) } , for any point a and any set B .

Theorem 11. Let L : R d → R n be a convex, differentiable surrogate. Let ℓ : R → R n be a discrete target loss, with finite property γ := prop[ L ] . Suppose L strongly indirectly elicits ℓ . Under Assumption 1, ( L, ψ ) is calibrated with respect to ℓ , for any link ψ : R d → R , that satisfies the following:

<!-- formula-not-decoded -->

Proof. First observe that a link of form ψ exists since for any u ∈ R d , proj Γ(∆ n ) ( u ) is non-empty and well-defined as Γ(∆ n ) is closed by Lemma 30. Now, we show that for any such link ψ , ( L, ψ ) satisfies calibration with respect to ℓ .

For any p ∈ ∆ n , we know from Lemma 29 that there exists some ϵ p &gt; 0 , such that for any v ∈ Γ( B ϵ p (Γ Γ( p ) )) , it holds that γ ( q ) ⊆ γ ( p ) , ∀ q ∈ Γ v . Then, we know from Lemma 27, that there exists δ p &gt; 0 such that Γ(∆ n ) ∩ B δ p (Γ( p )) ⊆ Γ( B ϵ p (Γ Γ( p ) )) . So,

<!-- formula-not-decoded -->

Now, suppose there exists a link ψ : R d →R of the form proposed in the theorem statement, such that ( L, ψ ) is not calibrated. This means, there exists p ∈ ∆ n , such that

<!-- formula-not-decoded -->

Thus, there exists some sequence { u t } t ∈ N + such that ψ ( u t ) / ∈ γ ( p ) but lim t →∞ ⟨ p, L ( u t ) ⟩ → inf u ∈ R d ⟨ p, L ( u ) ⟩ . It follows from Lemma 17 in Appendix C, that lim t →∞ dist ( u t , Γ( p )) → 0 .

Thus, for some t ∈ N + , it holds that dist ( u t , Γ( p )) &lt; δ p / 4 . Since Γ( p ) is compact, there must be some u ∗ ∈ Γ( p ) , such that ∥ u t -u ∗ ∥ 2 &lt; δ p / 4 . However, since ψ ( u t ) / ∈ γ ( p ) , there exists some v ∈ proj Γ(∆ n ) ( u t ) , such that ∥ u t -v ∥ 2 ≤ ∥ u t -u ∗ ∥ 2 &lt; δ p / 4 . Further, by the link definition, ψ ( u t ) ∈ γ ( q ) , and since ψ ( u t ) / ∈ γ ( p ) , it holds that γ ( q ) ̸⊆ γ ( p ) , for q ∈ Γ v . However, this contradicts condition (7) as ∥ u ∗ -v ∥ 2 ≤ ∥ u ∗ -u t ∥ 2 + ∥ u t -v ∥ 2 &lt; δ p / 2 . Thus, no such sequence exists and so

<!-- formula-not-decoded -->

Hence ( L, ψ ) is calibrated with respect to ℓ .

## G Equivalence of Strong Indirect Elicitation and Calibration under Strong Convexity

Lemma 31. Let L : R d → R n be a convex, differentiable surrogate. Consider the function, F L : ∆ n × R d → R , where F L ( p, u ) = ⟨ p, L ( u ) ⟩ . F L is continuous.

Proof. Let { ( p t , u t ) } t ∈ N + be a sequence in ∆ n × R d , such that lim t →∞ ( p t , u t ) → ( p, u ) , for some p ∈ ∆ n , u ∈ R d . We need to show that lim t →∞ ⟨ p t , L ( u t ) ⟩ → ⟨ p, L ( u ) ⟩ .

<!-- formula-not-decoded -->

Taking the limit as t → ∞ , ∥ L ( u t ) -L ( u ) ∥ + ∥ L ( u ) ∥ · ∥ p t -p ∥ → 0 since ∥ p t -p ∥ → 0 by construction, and ∥ L ( u t ) -L ( u ) ∥ → 0 as u t → u and L y ( · ) is continuous for each y ∈ [ n ] , and thus L y ( u t ) → L y ( u ) , ∀ y ∈ [ n ] = ⇒ ∥ L ( u t ) -L ( u ) ∥ → 0 . Thus, lim t →∞ ⟨ p t , L ( u t ) ⟩ = ⟨ p, L ( u ) ⟩ , and so lim t →∞ F L ( p t , u t ) → F L ( p, u ) , whenever lim t →∞ ( p t , u t ) → ( p, u ) . Hence, F L is continuous.

Lemma 32. Let f : R d → R be a differentiable, strongly convex function. Then arg min u ∈ R d f ( u ) exists and is a singleton.

Lemma 33. [Boyd, 2004] Let f : R d → R be µ f -strongly convex and let g : R d → R be µ g -strongly convex. Then f + g is ( µ f + µ g ) -strongly convex. For any α &gt; 0 , the function α · f is α · µ f -strongly convex. Also, f is µ -strongly convex for every 0 &lt; µ ≤ µ f .

Lemma 34. Let L : R d → R n be a convex, differentiable surrogate. Suppose for each y ∈ [ n ] L y : R d → R is µ y -strongly convex, where µ y &gt; 0 . Let µ m := min { µ i } n i =1 . Then ⟨ p, L ( · ) ⟩ is µ m -strongly convex, for every p ∈ ∆ n .

Proof. Let p ∈ ∆ n . For any y ∈ [ n ] , p y · L y is p y · µ y -strongly convex by Lemma 33. Also, ⟨ p, L ( · ) ⟩ = Σ y ∈ [ n ] p y · L y ( · ) is Σ y ∈ [ n ] p y · µ y - strongly convex by Lemma 33. Notice that, Σ y ∈ [ n ] p y · µ y ≤ Σ y ∈ [ n ] p y · µ m = µ m . Thus, ⟨ p, L ( · ) ⟩ is µ m -strongly convex by Lemma 33.

Lemma 35. Let L : R d → R n be a surrogate loss, with strongly convex, differentiable components. Then, Γ : ∆ n ⇒ R d is single-valued and continuous.

Proof. Suppose for each y ∈ [ n ] L y : R d → R is µ y -strongly convex, where µ y &gt; 0 . Let µ m := min { µ i } n i =1 . Then we know by Lemma 34, that ⟨ p, L ( · ) ⟩ is µ m -strongly convex for every p ∈ ∆ n . Fix some p ∈ ∆ n . Let { p t } t ∈ N + be a sequence of distributions in ∆ n , such that lim t →∞ p t → p . We know from Lemma 32 that Γ( p t ) exists and is single-valued for each p t since ⟨ p t , L ( · ) ⟩ is differentiable and strongly convex. Suppose u t ∈ R d such that u t = Γ( p t ) , for each t ∈ N + . We need to show that lim t →∞ u t → u ∗ where u ∗ = Γ( p ) (again, Γ( p ) exists and is single-valued).

Suppose v ∈ R d . We know from Lemma 31 that the function F : ∆ n × R d , where F L ( · , · ) = ⟨· , L ( · ) ⟩ is continuous. Then define m v := min u ∈ ∂B 1 ( v ) ,q ∈ ∆ n ⟨ q, L ( u ) ⟩ and M v := max q ∈ ∆ n ⟨ q, L ( v ) ⟩ . Both m v , M v exist since F L is continuous, ∂B 1 ( v ) × ∆ n and ∆ n × { v } are compact. Now, pick any

w ∈ R d : ∥ w ∥ = 1 . Let β &gt; max { 1 , 1+ 2 · ( M v -m v ) µ m } . Let v 1 := v + w and v 2 := v + β · w . Notice, v 1 = β -1 β v + 1 β v 2 . Next, for any q ∈ ∆ n , we have that:

<!-- formula-not-decoded -->

So we have established that,

<!-- formula-not-decoded -->

The first inequality holds due to the definition of m v . The second inequality follows by µ m -strong convexity and the final equality follows from the definition of v 2 . Now, we claim that, ⟨ q, L ( v 2 ) ⟩ &gt; ⟨ q, L ( v ) ⟩ . Assume to the contrary that ⟨ q, L ( v 2 ) ⟩ ≤ ⟨ q, L ( v ) ⟩ . Since, M v ≥ ⟨ q, L ( v ) ⟩ , we get by (8) that, m v ≤ M v -1 2 · µ m · ( β -1) = ⇒ m v -M v ≤ -1 2 · µ v · ( β -1) = ⇒ β ≤ 1 + 2 · ( M v -m v ) µ m , which violates the condition for choosing β &gt; max { 1 , 1+ 2 · ( M v -m v ) µ m } . Thus, ⟨ q, L ( u ) ⟩ &gt; ⟨ q, L ( v ) ⟩ for any u : ∥ v -u ∥ &gt; max { 1 , 1 + 2 · ( M v -m v ) µ m } and any q ∈ ∆ n , since v 2 = v + β · w for arbitrary w ∈ R d : ∥ w ∥ = 1 and β &gt; max { 1 , 1 + 2 · ( M v -m v ) µ m } was chosen arbitrarily, followed by which q ∈ ∆ n was also arbitrarily picked. Thus, for any q ∈ ∆ n , Γ( q ) must be such that ∥ v -Γ( q ) ∥ ≤ max { 1 , 1 + 2 · ( M v -m v ) µ m } since ⟨ q, L (Γ( q )) ⟩ ≤ ⟨ q, L ( v ) ⟩ . Thus, Γ(∆ n ) is uniformly bounded in a ball around v . And so, the sequence { u t } t ∈ N + = Γ( p t ) t ∈ N + must be bounded as well. Thus, there exists a u ′ ∈ R d and a subsequence { u t j } j ∈ N + such that lim j →∞ u t j → u ′ . By definition, ⟨ p t j , L ( u t j ) ⟩ ≤ ⟨ p t j , L ( u ) ⟩ for every u ∈ R d . Thus, by continuity of F L , we have that ⟨ p, L ( u ′ ) ⟩ ≤ ⟨ p, L ( u ) ⟩ , ∀ u ∈ R d . Since ⟨ p, L ( · ) ⟩ admits a unique minimizer, it follows that u ′ = u ∗ = Γ( p ) . Thus, every convergent subsequence of { u t } t ∈ N + must converge to the same limit u ∗ , and as { u t } t ∈ N + is bounded, lim t →∞ u t = u ∗ = Γ( p ) .

Thus, we have shown that for any p ∈ ∆ n and any sequence of distributions { p t } t ∈ N + , such that lim t →∞ p t → p , it follows that Γ( p t ) → Γ( p ) . Thus, Γ is continuous.

Lemma 36. Let L : R d → R n be a surrogate loss with strongly convex, differentiable components. Let ℓ : R → R n . If L indirectly elicits ℓ , but does not strongly indirectly elicit ℓ , there exists some report u ∈ R d , such that γ ( p m ) ⊂ γ ( p ) , for some p m , p ∈ Γ u .

̸

̸

Proof. We claim that ∃ p ′ , q ′ ∈ Γ u , such that | γ ( p ′ ) | = | γ ( q ′ ) | . Assume to the contrary that for every p, q ∈ Γ u , | γ ( p ) | = | γ ( q ) | . So, we have that ∃ u ∈ R d , such that γ ( p ∗ ) = γ ( q ∗ ) , but | γ ( p ∗ ) | = | γ ( q ∗ ) | . Sine L indirectly elicits ℓ , we have by Lemma 10 that γ ( p ∗ ) ∩ γ ( q ∗ ) = ∅ . So ∃ S ⊆ R : S = ∅ and S = γ ( p ∗ ) ∩ γ ( q ∗ ) , while S ⊂ γ ( p ∗ ) and S ⊂ γ ( q ∗ ) . In particular, this means | S | &lt; | γ ( p ∗ | . Now, we know by Lemma 8 that p ∗ + q ∗ 2 is such that γ ( p ∗ + q ∗ 2 ) ⊆ S . This means | γ ( p ∗ + q ∗ 2 ) | ≤ | S | &lt; | γ ( p ∗ ) | = ⇒ | γ ( p ∗ + q ∗ 2 ) | &lt; γ ( p ) which contradicts our assumption since p ∗ + q ∗ 2 ∈ Γ u by convexity of Γ u . Thus, ∃ p ′ , q ′ ∈ Γ u , such that | γ ( p ′ ) | = | γ ( q ′ ) | .

̸

̸

̸

Let p m ∈ Γ u : | γ ( p m ) | ≤ | γ ( p ) | , ∀ p ∈ Γ u . We know by Lemma 9 that γ ( p m ) ⊆ γ ( p ) , ∀ p ∈ Γ u . In fact, ∃ p ∈ Γ u : | γ ( p ) | = | γ ( p m ) | = ⇒ | γ ( p m ) | &lt; | γ ( p ) | and so γ ( p m ) ⊂ γ ( p ) .

Theorem 12. Let L : R d → R n be a surrogate loss, with strongly convex, differentiable components. Let ℓ : R→ R n . If L does not strongly indirectly elicit ℓ , then there is no link function ψ : R d →R , such that ( L, ψ ) satisfies calibration with respect to ℓ .

Proof. First, suppose L does not indirectly elicit ℓ . Then straight away, calibration fails since calibration implies indirect elicitation by Theorem 6. Now, suppose L indirectly elicits ℓ , but does not strongly indirectly elicit ℓ . We know from Lemma 36, that for some u ∈ R d , γ ( p m ) ⊂ γ ( p ) , where p m , p ∈ Γ u . Assume WLOG that γ ( p m ) = { 1 , 2 , ..., t } and that γ ( p ) = { 1 , 2 , ..., t, ..., t + j }

̸

where j ≥ 1 . In particular, p lies on the boundary of the cell γ t +1 , which is a convex polytope. Thus, we can pick a sequence { p i } i ∈ N + , such that γ ( p i ) = { t +1 } , ∀ i ∈ N + , and lim i →∞ p i → p . Define v i := Γ( p i ) , for every i ∈ N + . Then, we have by Lemma 35 that since each of the components of L are differentiable and strongly convex, it holds that Γ is continuous and hence lim i →∞ v i = u . To ensure calibration at p m , it is necessary ψ ( u ) ∈ γ ( p m ) , since u = Γ( p m ) and γ ( p m ) ⊆ γ ( p ) , ∀ p ∈ Γ u . Also, to ensure calibration at p i it is necessary that ψ ( v i ) = t +1 , since γ ( v i ) = { t +1 } for every i ∈ N + . However, despite this, we show calibration fails:

<!-- formula-not-decoded -->

Hence, we have shown that inf v ∈ R d : ψ ( v ) / ∈ γ ( p m ) ⟨ p m , L ( v ) ⟩ = inf v ∈ R d ⟨ p, L ( v ) ⟩ , thus violating calibration at p m .

## H Constructing 1d surrogates for orderable properties

In this section, we provide an explicit construction of a consistent, convex, differentiable surrogate with domain dimension 1 for a given orderable target loss. Formally, given an orderable target, ℓ : R→ R n , we prove constructively the existence of a convex, differentiable surrogate L : R → R n that is consistent with respect to ℓ .

Our construction hinges on a subroutine which we will call LinIntGrad ( X ) . Given a vector X ∈ R k -1 , for some integer k ≥ 2 , LinIntGrad constructs a function f : R → R that is convex, differentiable and whose gradients match X at inputs { 1 , 2 , .., k -1 } , i.e., f ′ ( i ) = X [ i ] , ∀ i ∈ [ k -1] . See Subroutine 1 for the detailed construction.

## Subroutine 1 LinIntGrad ( X )

- 1: Input: X [1] , . . . , X [ k -1] (with k ≥ 2 )
- 3: (A) Define g on (1 , k -1) by linear interpolation
- 2: Goal: Define a gradient map g : R → R and f ( x ) = ∫ x 1 g ( t ) dt
- 4: for j = 1 , 2 , . . . , k -2 do
- 5: For x ∈ ( j, j +1] \ { k -1 } , set

<!-- formula-not-decoded -->

- 6: (B) Left extrapolation on ( -∞ , 1]

<!-- formula-not-decoded -->

- 7: For x ≤ 1 , set
- 8: (C) Right extrapolation on [ k -1 , ∞ )

<!-- formula-not-decoded -->

- 9: For x ≥ k -1 , set

<!-- formula-not-decoded -->

Lemma 37. Given some X ∈ R k -1 , where k ≥ 2 is an integer. Following the notation of Subroutine 1, let g be the gradient-map constructed and let f = LinIntGrad ( X ) . Then:

- (i) g is continuous and nondecreasing on R .
- (ii) f is convex and C 1 on R , with f ′ ( x ) = g ( x ) for all x .

- (iii) f ′ ( i ) = X [ i ] for each i ∈ { 1 , . . . , k -1 } .
- (iv) Minimizer location:
- In Case 1 ( X [1] ≤ 0 ≤ X [ k -1] ), one has arg min f ⊆ [1 , k -1] . If X ≡ 0 on { 1 , . . . , k -1 } then arg min f = [1 , k -1] ; otherwise g crosses 0 inside [1 , k -1] and the minimizer lies there (unique if the crossing is strict).
- In Case 2, when X [1] &gt; 0 , arg min f = { 1 -X [1] }
- In Case 3, when X [ k -1] &lt; 0 , arg min f = { k -1 -X [ k -1] } .
- (v) arg min f is nonempty and compact.

Proof. (i) Continuity and monotonicity of g . First, consider the behavior of g on the interval (1 , k -1) - see (A) in Subroutine 1. On each sub-interval, i.e., on each element of the set { ( j, j +1] \ { k -1 }| j ∈ [ k -2] } , g is either affine with nonnegative slope or constant. Thus, g is continuous on ( j, j + 1) , and nondecreasing on ( j, j + 1] \ { k -1 } , ∀ j ∈ [ k -2] . Let j ∈ { 2 , 3 , ..., k -2 } . We have that g ( j -) := lim x → j -X [ j -1] + ( x -( j -1))( X [ j ] -X [ j -1]) = X [ j ] , while g ( j + ) := lim x → j + X [ j ] + ( x -j )( X [ j +1] -X [ j ]) = X [ j ] . Thus, g ( j -) = g ( j + ) = X [ j ] = g ( j ) . Hence g is continuous on (1 , j -1) . We already know g is non-decreasing on each subinterval. Notice also that, g ( j ) = X [ j ] ≤ X [ j +1] = g ( j +1) . Hence, g is non-decreasing and continuous on (1 , j -1) .

Wenowanalyze continuity at x = 1 ; see (B) in Subroutine 1: for x ≤ 1 , we set g ( x ) = X [1]+( x -1) . For x ∈ (1 , 2) , we have g ( x ) = X [1] + ( x -1) ( X [2] -X [1] ) . Clearly, lim x → 1 -g ( x ) = X [1] = lim x → 1 + g ( x ) . Thus g is continuous at x = 1 .

A similar check at x = k -1 ; see (C) in Subroutine 1: For x ∈ ( k -2 , k -1) , g ( x ) = X [ k -2] + ( x -( k -2) )( X [ k -1] -X [ k -2] ) . So, lim x → ( k -1) -g ( x ) = X [ k -1] . Whereas, for x ≥ k -1 , g ( x ) = X [ k -1] + ( x -( k -1)) . So, lim x → ( k -1) + g ( x ) = X [ k -1] . Hence g is continuous at 1 and k -1 , which means g is continuous on [1 , k -1] . On ( -∞ , 1) , as well as ( k -1 , ∞ ) , g is affine with non-negative slope. Thus, g is continuous and non-decreasing on R .

(ii) Convexity and differentiability of f . Since f ( x ) = ∫ x 1 g ( t ) dt , f is C 1 ( R ) with f ′ = g everywhere. So f ′ is monotone (since g is nondecreasing), and hence f is convex.

(iii) Gradients match X in [ k -1] . For each j ∈ { 1 , . . . , k -1 } , the interpolation g ( x ) = X [ j ] + ( X [ j +1] -X [ j ] ) ( x -j ) gives g ( j ) = X [ j ] . Hence f ′ ( j ) = g ( j ) = X [ j ] .

(iv) Minimizer location. Recall that for differentiable convex f , any minimizer x ⋆ satisfies f ′ ( x ⋆ ) = 0 , and conversely if f ′ changes sign from negative to positive at x ⋆ , then x ⋆ is the unique minimizer.

̸

On [1 , k -1] , g is continuous and nondecreasing with g (1) = X [1] ≤ 0 and g ( k -1) = X [ k -1] ≥ 0 , hence any zero of g lies in [1 , k -1] . If X ≡ 0 on { 1 , . . . , k -1 } , we have g ≡ 0 on [1 , k -1] . If not, g must cross 0 somewhere in [1 , k -1] due to continuity. Thus, arg min f ⊆ [1 , k -1] and arg min f = ∅ . So arg min f is non-empty and bounded. Since f is convex, arg min f is closed, and hence compact.

Case 1 ( X [1] ≤ 0 ≤ X [ k -1] ): If X [1] &lt; 0 , then g ( x ) = X [1] &lt; 0 , ∀ x ≤ 1 . Whereas, if X [1] = 0 , then g ( x ) = x -1 &lt; 0 , ∀ x &lt; 1 . Either way, g ( x ) = 0 for any x &lt; 0 . Similarly, if X [ k -1] &gt; 0 , then g ( x ) = X [ k -1] &gt; 0 , ∀ x ≥ k -1 . Whereas, if X [ k -1] = 0 , g ( x ) = x -( k -1) &gt; 0 , ∀ x &gt; k -1 . Either way, g ( x ) = 0 for any x &gt; k -1 . Thus, g = 0 on ( -∞ , 1) ∪ (1 , ∞ ) .

̸

Case 2 ( X [1] &gt; 0 ): First notice that in this case, since X [1] &gt; 0 , g (1) &gt; 0 and since g is nondecreasing it follows that g ( x ) &gt; 0 for every x ≥ 1 . So any zero g attains must be in ( -∞ , 1) . For x &lt; 1 , g ( x ) = X [1] + ( x -1) = 0 ⇐⇒ x = 1 -X [1] &lt; 1 . Since g is strictly increasing on ( -∞ , 1) , it crosses 0 exactly once at 1 -X [1] . Thus, arg min f = { 1 -X [1] } which is non-empty and compact.

Case 2 ( X [ k -1] &lt; 0 ): First notice that in this case, since X [ k -1] &lt; 0 , g ( k -1) &lt; 0 and since g is nondecreasing it follows that g ( x ) &lt; 0 for every x ≤ k -1 . So any zero g attains must be in ( k -1 , ∞ ) . For x &gt; k -1 , g ( x ) = X [ k -1]+ ( x -( k -1) ) = 0 ⇐⇒ x = k -1 -X [ k -1] &gt; k -1 . Since g is strictly increasing on ( k -1 , ∞ ) , it crosses 0 exactly once at k -1 -X [ k -1] . Thus, arg min f = { k -1 -X [ k -1] } which is non-empty and compact.

̸

̸

## Construction 2 SURROGATE CONSTRUCTION

- 1: Inputs: V 1 , . . . , V n ∈ R k -1 (rows of matrix V ) 2: Output: Consistent, convex and differentiable surrogate loss L : R → R n 3: for all j ∈ [ n ] do 4: L ( · ) j ← LinIntGrad ( V j ) 5: Define L : L ( u ) ← [ L ( u ) 1 , L ( u ) 2 , ..., L ( u ) n ] 6: return L

Theorem 13. Given an orderable target ℓ : R→ R n , there exists a convex, differentiable surrogate L : R → R n satisfying Assumption 1, which is calibrated with respect to ℓ . In particular, Construction 2 yields such a surrogate loss L .

Proof. Since γ = prop [ ℓ ] is orderable, there exists an orderable enumeration of R , i.e., E γ = ( r 1 , r 2 , ..., r k ) (see Definition 7). We know from Theorem 11 of Finocchiaro et al. [2020] that there exists a set { v 1 , v 2 , ..., v k -1 } ⊂ R n such that:

1. The set satisfies coordinate-wise monotonicity. That is, ∀ i ∈ [ k -2] , y ∈ [ n ] it holds that, v i,y ≤ v i + ,y .
2. The set of vectors are normal to target boundaries, i.e., ∀ p ∈ γ r i ∩ γ r i +1 , ⟨ p, v i ⟩ = 0 .

Let us denote by V ∈ R n × k -1 the matrix with column vectors v 1 , v 2 , ..., v k -1 in that order. Let V j , j ∈ [ n ] denote the j th row of V . These row-vectors are set as inputs to the surrogate construction described in Construction 2. We now show that the output of L = SURROGATE CONSTRUCTION ( V 1 , V 2 , ..., V n ) is convex, differentiable, satisfies Assumption 1 and is consistent w.r.t. ℓ .

Each component L ( · ) j , j ∈ [ n ] defined in Construction 2, is obtained via Subroutine 1. Thus, for each j ∈ [ n ] , L ( · ) j is convex, differentiable and satisfies Assumption 1 by Lemma 37. We show that L indirectly elicits ℓ and the result then follows by Theorem 1.

̸

̸

Assume L does not indirectly elicit ℓ . This means, there exists some u ∈ R , such that Γ u ̸⊆ γ r , ∀ r ∈ R . In particular, this means that the level-set Γ u crosses from one target cell's relative interior into another target cell's relative interior. By convexity of Γ u , there exists some j ∈ [ k -1] and some p ∈ γ r j ∩ γ r j +1 such that p ∈ Γ u . By construction, ∇ L ( j ) = v j . Clearly, u = j as ∇ L ( u ) = v j . Assume WLOG that u &lt; j . This means ∇ L ( u ) = ∇ L ( j ) -δ , where δ i ≥ 0 , ∀ i ∈ [ n ] and δ i ∗ &gt; 0 for some i ∗ ∈ [ n ] . So, ⟨ p, ∇ L ( u ) ⟩ = ⟨ p, ∇ L ( j ) -δ ⟩ = -⟨ p, δ ⟩ &lt; 0 by the condition on δ . Thus, p / ∈ Γ u yielding a contradiction. Hence L must indirectly elicit ℓ .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: [Yes]

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: [Yes]

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

Justification: [Yes]

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

Justification: There are no experiments.

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

Justification: There is no data or code.

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

Justification: We use no data.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: We have no data.

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

Justification: We just used our brains.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: [Yes]

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: How could theorists possibly affect society let's be honest.

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

Justification: We don't have data or models. We are theorists.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: [Yes]

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

Justification: We are tragically asset-less.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We didn't do any experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We didn't do human trials.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We only used them for editing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.
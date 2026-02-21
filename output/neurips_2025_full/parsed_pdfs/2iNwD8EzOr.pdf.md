## Online Inverse Linear Optimization: Efficient Logarithmic-Regret Algorithm, Robustness to Suboptimality, and Lower Bound

## Shinsaku Sakaue ∗

CyberAgent

Tokyo, Japan shinsaku.sakaue@gmail.com

## Han Bao ∗

The Institute of Statistical Mathematics Tokyo, Japan bao.han@ism.ac.jp

## Abstract

In online inverse linear optimization, a learner observes time-varying sets of feasible actions and an agent's optimal actions, selected by solving linear optimization over the feasible actions. The learner sequentially makes predictions of the agent's true linear objective function, and their quality is measured by the regret , the cumulative gap between optimal objective values and those achieved by following the learner's predictions. A seminal work by Bärmann et al. (2017) obtained a regret bound of O ( √ T ) , where T is the time horizon. Subsequently, the regret bound has been improved to O ( n 4 ln T ) by Besbes et al. (2021, 2025) and to O ( n ln T ) by Gollapudi et al. (2021), where n is the dimension of the ambient space of objective vectors. However, these logarithmic-regret methods are highly inefficient when T is large, as they need to maintain regions specified by O ( T ) constraints, which represent possible locations of the true objective vector. In this paper, we present the first logarithmic-regret method whose per-round complexity is independent of T ; indeed, it achieves the best-known bound of O ( n ln T ) . Our method is strikingly simple: it applies the online Newton step (ONS) to appropriate exp-concave loss functions. Moreover, for the case where the agent's actions are possibly suboptimal, we establish a regret bound of O ( n ln T + √ ∆ T n ln T ) , where ∆ T is the cumulative suboptimality of the agent's actions. This bound is achieved by using MetaGrad, which runs ONS with Θ(ln T ) different learning rates in parallel. We also present a lower bound of Ω( n ) , showing that the O ( n ln T ) bound is tight up to an O (ln T ) factor.

## 1 Introduction

Optimization problems serve as forward models of various processes and systems, ranging from human decision-making to natural phenomena. In real-world applications, the true objective function of such models is rarely known a priori. This motivates the problem of inferring the objective function from observed optimal solutions, or inverse optimization . Early work in this area emerged from geophysics, aiming at estimating subsurface structure from seismic wave data [11, 53]. Subsequently, inverse optimization has been extensively studied [2, 13, 14, 28], applied across various domains, such as

* This work was primarily conducted during the period when SS was affiliated with the University of Tokyo and RIKEN AIP, and HB with Kyoto University and OIST.

## Taira Tsuchiya

The University of Tokyo and RIKEN AIP Tokyo, Japan tsuchiya@mist.i.u-tokyo.ac.jp

## Taihei Oki

Hokkaido University Hokkaido, Japan oki@icredd.hokudai.ac.jp

transportation [6], power systems [9], and healthcare [12], and have laid the foundation for various machine learning methods, including inverse reinforcement learning [44] and contrastive learning [50].

This study focuses on an elementary yet fundamental case where the objective function of forward optimization is linear. We consider an agent who repeatedly selects an action from a set of feasible actions by solving forward linear optimization. 1 Let n be a positive integer and R n the ambient space where forward optimization is defined. For t = 1 , . . . , T , given a set X t ⊆ R n of feasible actions, the agent selects an action x t ∈ X t that maximizes x ↦→⟨ c ∗ , x ⟩ over X t , where c ∗ ∈ R n is the agent's internal objective vector and ⟨· , ·⟩ denotes the standard inner product on R n . We want to infer c ∗ from observations consisting of the feasible sets and the agent's optimal actions, i.e., { ( X t , x t ) } T t =1 .

For this problem, Bärmann et al. [4, 5] have shown that online learning methods are effective for inferring the agent's underlying objective vector c ∗ . Consider a learner who aims to infer c ∗ . For t = 1 , . . . , T , the learner makes a prediction ˆ c t of c ∗ based on the past observations { ( X i , x i ) } t -1 i =1 and receives ( X t , x t ) as feedback. Let ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ represent an optimal action induced by the learner's t th prediction. The regret of choosing ˆ x t instead of x t is defined as ∑ T t =1 ⟨ c ∗ , x t -ˆ x t ⟩ . 2 Their idea is to regard R n ∋ c ↦→⟨ c, ˆ x t -x t ⟩ as a cost function and apply online learning methods, such as the online gradient descent (OGD). Then, ∑ T t =1 ⟨ c ∗ , x t -ˆ x t ⟩ ≤ ∑ T t =1 ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ = O ( √ T ) follows from the standard guarantee of online learning methods. As such, online learning methods with sublinear regret bounds can make the average regret converge to zero as T →∞ .

While the regret bound of O ( √ T ) is optimal in general online linear optimization (e.g., Hazan [26, Section 3.2]), the above online inverse linear optimization has special problem structures that could allow for better regret bounds; intuitively, feedback ( X t , x t ) is more informative about c ∗ , which defines the regret, due to the optimality of x t ∈ X t for c ∗ . Besbes et al. [7, 8] indeed showed that a logarithmic regret bound of O ( n 4 ln T ) is possible, and Gollapudi et al. [25] further improved the bound to O ( n ln T ) . 3 While these methods significantly improve the dependence on T in the regret bounds, their per-round computation cost is prohibitively high when T is large. Specifically, these methods iteratively update (appropriately inflated) regions that indicate possible locations of the true objective vector c ∗ and set prediction ˆ c t to the 'center' of the regions (the circumcenter in Besbes et al. [7, 8] and the centroid in Gollapudi et al. [25]). Since those regions are represented by O ( T ) constraints, their per-round complexity grows polynomially in T , at least in a straightforward implementation. Indeed, Besbes et al. [7, 8] and Gollapudi et al. [25] only claim that their methods run in poly( n, T ) time. This is in stark contrast to the earlier online-learning approach of Bärmann et al. [4, 5], whose per-round complexity is independent of T ; however, its O ( √ T ) -regret bound is much worse in terms of T . Is it then possible to design a logarithmic-regret method whose per-round complexity is independent of T ?

## 1.1 Our contributions

In this paper, we first present an O ( n ln T ) -regret method whose per-round complexity is independent of T (Theorem 3.1), answering the above question affirmatively. Table 1 summarizes the comparisons of our result with prior work. Our method is very simple: we apply the online Newton step (ONS) [27] to appropriately designed exp-concave loss functions. We believe this simplicity is a strength of our method, which makes it accessible to a wider audience and easier to implement.

We then address more realistic situations where the agent's actions can be suboptimal. We establish a regret bound of O ( n ln T + √ ∆ T n ln T ) , where ∆ T denotes the cumulative suboptimality of the agent's actions over T rounds (Theorem 4.1). We also apply this result to the offline setting via the online-to-batch conversion (Corollary 4.2). This bound is achieved by applying MetaGrad [55, 56], a universal online learning method that runs ONS with Θ(ln T ) different learning rates in parallel, to the suboptimality loss [43], a loss function commonly used in inverse optimization. While universal online learning is originally intended to adapt to unknown types of loss functions, our result shows that it is useful for adapting to unknown suboptimality levels in online inverse linear optimization. At

1 An 'agent' is sometimes called an 'expert,' which we do not use to avoid confusion with the expert in universal online learning (see Section 2.3). Additionally, our results could potentially be extended to nonlinear settings based on kernel inverse optimization [6, 39], although we focus on the linear setting for simplicity.

2 In the online setting, the learner's goal subtly deviates from inferring c ∗ directly. Instead, the learner aims to make predictions ˆ c t such that the induced actions ˆ x t are good for the true objective c ∗ .

3 Gollapudi et al. [25] studied the same problem under the name of contextual recommendation .

Table 1: Comparisons of the regret bound under optimal feedback and per-round/total complexity. Here, τ solve is the time for computing ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ , and τ E-proj / τ G-proj is the time for the Euclidean/generalized projection; typically, τ E-proj = O ( n ) and τ G-proj = O ( n 3 ) (see Section 3 and Appendix A for details). Regarding the regret bound of Bärmann et al. [4, 5], the dependence on n varies depending on the problem setting, which we discuss in Appendix A (here, set sizes are regarded as constants). Besbes et al. [7, 8] and Gollapudi et al. [25] only claim that the total complexity is poly( n, T ) . Our inspection in Appendix A estimates the per-round complexity of Gollapudi et al. [25] as O ( τ solve + n 5 T 3 ) or higher.

|                                                                                        | Regret bound                                       | Per-round complexity                                                                | Total complexity                                    |
|----------------------------------------------------------------------------------------|----------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------|
| Bärmann et al. [4, 5] Besbes et al. [7, 8] Gollapudi et al. [25] This work (Section 3) | O ( √ T ) O ( n 4 ln T ) O ( n ln T ) O ( n ln T ) | O ( τ solve + τ E-proj + n ) Not claimed Not claimed O ( τ solve + τ G-proj + n 2 ) | Per-round × T poly( n,T ) poly( n,T ) Per-round × T |

a high level, our important contribution lies in uncovering the deeper connection between inverse optimization and online learning, thereby enabling the former to leverage the powerful toolkit of the latter.

Finally, we present a regret lower bound of Ω( n ) (Theorem 5.1). Thus, the upper bound of O ( n ln T ) achieved by the method of Gollapudi et al. [25] and ours is tight up to an O (ln T ) factor. While the proof idea is somewhat straightforward, this lower bound clarifies the optimal dependence on n in the regret bound, thereby resolving a question raised in Besbes et al. [8, Section 7].

## 1.2 Related work

Classic studies on inverse optimization explored formulations for identifying parameters of forward optimization from a single observation [2, 29]. Recently, data-driven inverse optimization, which is intended to infer parameters of forward optimization from multiple noisy (possibly suboptimal) observations, has drawn significant interest [3, 6, 10, 35, 39, 42, 43, 52, 63]. This body of work has addressed offline settings with other criteria than the regret, which we formally define in (2). The suboptimality loss was introduced by Mohajerin Esfahani et al. [43] in this context.

The line of work by Bärmann et al. [4, 5], Besbes et al. [7, 8], and Gollapudi et al. [25], mentioned in Section 1, is the most relevant to our work; we present the detailed comparisons with them in Appendix A. It is worth mentioning here that Gollapudi et al. [25] obtained an exp( O ( n ln n )) -regret bound, in addition to the O ( n ln T ) bound in Table 1; therefore, it is possible to achieve a finite regret bound, although the dependence on n is exponential. Recently, Sakaue et al. [49] obtained a finite regret bound by assuming a gap between the optimal and suboptimal objective values. Unlike their work, we do not require such gap assumptions. Online inverse linear optimization can also be viewed as a variant of stochastic linear bandits [1, 18], where noisy objective values are given as feedback, instead of optimal actions. Intuitively, the optimal-action feedback is more informative and allows for the O ( n ln T ) regret upper bound, while there is a lower bound of Ω( n √ T ) in stochastic linear bandits [18, Theorem 3]. Online-learning approaches to other related settings have also been studied [20, 30, 59]; see Besbes et al. [8, Section 1.2] for an extensive discussion on the relation to these studies. Additionally, Chen and Kılınç-Karzan [15] and Sun et al. [51] studied online-learning methods for related settings with different criteria.

ONS [27] is a well-known online convex optimization (OCO) method that achieves a logarithmic regret bound for exp-concave loss functions. While ONS requires the prior knowledge of the expconcavity, universal online learning methods, including MetaGrad, can automatically adapt to the unknown curvatures of loss functions, such as the strong convexity and exp-concavity [55, 56, 58, 64]. Our strategy for achieving robustness to suboptimal feedback is to combine the regret bound of MetaGrad (Proposition 2.6) with the self-bounding technique (see Section 4 for details), which is widely adopted in the online learning literature [23, 60, 66].

Contextual search [38, 48] is a related problem of inferring the value of ⟨ c ∗ , x t ⟩ for an underlying vector c ∗ given context vectors x t . The method of Gollapudi et al. [25] is based on techniques developed in this context. Robustness to corrupted feedback is also studied in contextual search [36, 46, 47].

However, note that the problem setting is different from ours. Also, the regret in contextual search is defined with optimal choices even under corrupted feedback, and the regret bounds scale linearly with the cumulative corruption level. By contrast, our regret is defined with the agent's possibly suboptimal actions and our regret bound grows only at the rate of √ ∆ T for the cumulative suboptimality ∆ T .

Improving the per-round complexity is crucial. This topic has gathered particular attention in online portfolio selection [17, 34, 57]. There exists a trade-off between the per-round complexity and regret bounds among known methods for this problem, and advancing this frontier is recognized as important research [32, 54, 65]. When it comes to online inverse linear optimization, logarithmic regret bounds had only been achieved by the somewhat inefficient methods of Besbes et al. [7, 8] and Gollapudi et al. [25], while the efficient online-learning approach of Bärmann et al. [4, 5] only enjoys the O ( √ T ) -regret bound. This background highlights the significance of our efficient O ( n ln T ) -regret method, which realizes the benefits of both approaches that previously existed in a trade-off relationship.

## 2 Preliminaries

## 2.1 Problem setting

Weconsider an online learning setting with two players, a learner and an agent . The agent sequentially solves linear optimization problems of the following form for t = 1 , . . . , T :

<!-- formula-not-decoded -->

where c ∗ is the agent's objective vector, which is unknown to the learner. Every feasible set X t ⊆ R n is non-empty and compact, and the agent's action x t always belongs to X t . Weassume that the agent's action is optimal for (1), i.e., x t ∈ arg max x ∈ X t ⟨ c ∗ , x ⟩ , except in Section 4, where we discuss the case where x t can be suboptimal. The set, X t , is not necessarily convex; we only assume access to an oracle that returns an optimal solution x ∈ arg max x ′ ∈ X t ⟨ c, x ′ ⟩ for any c ∈ R n . If X t is a polyhedron, any solver for linear programs (LPs) of the form (1) can serve as the oracle. Even if (1) is, for example, an integer LP, we may use empirically efficient solvers, such as Gurobi, to obtain an optimal solution.

The learner sequentially makes a prediction of c ∗ for t = 1 , . . . , T . Let Θ ⊆ R n denote a set of linear objective vectors, from which the learner picks predictions. We assume that Θ is a closed convex set and that the true objective vector c ∗ is contained in Θ . For t = 1 , . . . , T , the learner outputs a prediction ˆ c t of c ∗ based on past observations { ( X i , x i ) } t -1 i =1 and then receives ( X t , x t ) as feedback from the agent. Let ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ denote an optimal action induced by the learner's t th prediction. 4 We consider the following two measures of the quality of predictions ˆ c 1 , . . . , ˆ c T ∈ Θ :

<!-- formula-not-decoded -->

Following prior work [7, 8, 25], we call R c ∗ T the regret , which is the cumulative gap between the optimal objective values and the objective values achieved by following the learner's predictions. Note that we have ⟨ c ∗ , x t -ˆ x t ⟩ ≥ 0 as long as x t is optimal for c ∗ . While the regret is a natural performance measure, the second one, ˜ R c ∗ T , in (2) is convenient when considering the online-learning approach [4, 5]. We always have R c ∗ T ≤ ˜ R c ∗ T since the additional term consisting of ⟨ ˆ c t , ˆ x t -x t ⟩ is non-negative due to the optimality of ˆ x t for ˆ c t ; intuitively, this term quantifies how well ˆ c t explains the agent's choice x t . Our upper bounds in Theorems 3.1 and 4.1 apply to ˜ R c ∗ T , and our lower bound in Theorem 5.1 applies to R c ∗ T .

Remark 2.1. The problem setting of Besbes et al. [7, 8] involves context functions and initial knowledge sets , which might make their setting appear more general than ours. However, it is not difficult to confirm that our methods are applicable to their setting. See Appendix A for details.

## 2.2 Boundedness assumptions and suboptimality loss

We introduce the following bounds on the sizes of X t and Θ .

4 We may break ties, if any, arbitrarily. Our results remain true as long as ˆ x t is optimal for ˆ c t .

Assumption 2.2. The ℓ 2 -diameter of Θ is bounded by D &gt; 0 , i.e., max { ∥ c -c ′ ∥ 2 : c, c ′ ∈ Θ } ≤ D . Similarly, the ℓ 2 -diameter of X t is bounded by K &gt; 0 for t = 1 , . . . , T . Furthermore, there exists B &gt; 0 satisfying the following condition:

<!-- formula-not-decoded -->

Assuming bounds on the diameters is common in the previous studies [4, 5, 7, 8, 25]. We additionally introduce B &gt; 0 to measure the sizes of X t and Θ taking their mutual relationship into account. Note that the choice of B = DK is always valid due to the Cauchy-Schwarz inequality. This quantity is inspired by a semi-norm of gradients used in Van Erven et al. [56] and enables sharper analysis than that conducted by simply setting B = DK .

We also define the suboptimality loss for later use.

Definition 2.3. For t = 1 , . . . , T , for any action set X t and the agent's possibly suboptimal action x t , the suboptimality loss is defined by ℓ t ( c ) := max x ∈ X t ⟨ c, x ⟩ - ⟨ c, x t ⟩ for all c ∈ Θ .

That is, ℓ t ( c ) is the suboptimality of x t ∈ X t for c . Mohajerin Esfahani et al. [43] introduced this as a loss function that enjoys desirable computational properties in the context of inverse optimization. Specifically, the suboptimality loss is convex, and there is a convenient expression of a subgradient.

Proposition 2.4 (cf. Bärmann et al. [4, Proposition 3.1]) . The suboptimality loss, ℓ t : Θ → R , is convex. Moreover, for any ˆ c t ∈ Θ and ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ , it holds that ˆ x t -x t ∈ ∂ℓ t (ˆ c t ) .

Confirming these properties is not difficult: the convexity is due to the fact that ℓ t is the pointwise maximum of linear functions c ↦→⟨ c, x ⟩-⟨ c, x t ⟩ , and the subgradient expression is a consequence of Danskin's theorem [19] (or one can directly prove this as in Bärmann et al. [4, Proposition 3.1]). It is worth mentioning that, as pointed out by Sakaue et al. [49], ˜ R c ∗ T appears as the linearized upper bound on the regret with respect to the suboptimality loss, i.e., ∑ T t =1 ( ℓ t (ˆ c t ) -ℓ t ( c ∗ )) ≤ ∑ T t =1 ⟨ ˆ c t -c ∗ , g t ⟩ = ˜ R c ∗ T , where g t = ˆ x t -x t ∈ ∂ℓ t (ˆ c t ) . This enables the online-to-batch conversion for the suboptimality loss, as discussed in Section 4.1. Additionally, we have ˜ R c ∗ T = R c ∗ T + ∑ T t =1 ℓ t (ˆ c t ) in (2).

## 2.3 ONS and MetaGrad

Webriefly describe ONS and MetaGrad, based on Hazan [26, Section 4.4] and Van Erven et al. [56], to aid understanding of our methods. Appendix B shows the details for completeness. Readers who wish to proceed directly to our results may skip this subsection, taking Propositions 2.5 and 2.6 as given.

For convenience, we first state a specific form of ONS's O ( n ln T ) regret bound, which is later used in MetaGrad and in our analysis. See Algorithm 1 in Appendix B.1 for the pseudocode of ONS.

Proposition 2.5. Let W ⊆ R n be a closed convex set whose ℓ 2 -diameter is at most W &gt; 0 . Let w 1 , . . . , w T and g 1 , . . . , g T be vectors in R n satisfying the following conditions for some G,H &gt; 0 :

<!-- formula-not-decoded -->

Take any η ∈ ( 0 , 1 5 H ] and define loss functions f η t : W → R for t = 1 , . . . , T as follows:

<!-- formula-not-decoded -->

Let w η 1 , . . . , w η T ∈ W be the outputs of ONS applied to f η 1 , . . . , f η T . Then, for any u ∈ W , it holds that

<!-- formula-not-decoded -->

Next, we describe MetaGrad (see Algorithm 2 in Appendix B.3), which we apply to the following general OCO problem on a closed convex set, W ⊆ R n . For t = 1 , . . . , T , we select w t ∈ W based on information obtained up to the end of round t -1 ; then, we incur f t ( w t ) and observe a subgradient, g t ∈ ∂f t ( w t ) , where f t : W → R denotes the t th convex loss function. We assume that W and g t for t = 1 , . . . , T satisfy the conditions in (3). Our goal is to make the regret with respect to f t , i.e., ∑ T t =1 ( f t ( w t ) -f t ( u )) , as small as possible for any comparator u ∈ W .

MetaGrad maintains η -experts , each of whom is associated with one of Θ(ln T ) different learning rates η ∈ ( 0 , 1 5 H ] . Each η -expert applies ONS to loss functions f η t of the form (4), where w t ∈ W

is the t th output of MetaGrad and g t ∈ ∂f t ( w t ) is given as feedback. In each round t , given the outputs w η t of η -experts (which are computed based on information up to round t -1 ), MetaGrad computes w t ∈ W by aggregating them via the exponentially weighted average (EWA).

For any comparator u ∈ W , define ˜ R u T := ∑ T t =1 ⟨ w t -u, g t ⟩ and V u T := ∑ T t =1 ⟨ w t -u, g t ⟩ 2 . Since all functions f t are convex, the regret with respect to f t , or ∑ T t =1 ( f t ( w t ) -f t ( u )) , is bounded by ˜ R u T from above. Furthermore, from the definition of f η t , we can decompose ˜ R u T as follows:

<!-- formula-not-decoded -->

which simultaneously holds for all η &gt; 0 . The first summation on the right-hand side, i.e., the regret of EWA compared to w η t , is indeed as small as O (ln ln T ) , while Proposition 2.5 ensures that the second summation is O ( n ln T ) . Thus, the right-hand side is O ( n ln T η + ηV u T ) . If we knew the true V u T value, we could choose η ≃ √ n ln T/V u T to achieve O (√ n ln T · V u T ) . This might seem impossible as we do not know any of u , g t , and w t beforehand. However, we can show that at least one of Θ(ln T ) values of η leads to almost the same regret, eschewing the need for knowing V u T . Formally, MetaGrad achieves the following regret bound (cf. Van Erven et al. [56, Corollary 8]). 5

Proposition 2.6. Let W ⊆ R n be given as in Proposition 2.5. Let w 1 , . . . , w T ∈ W be the outputs of MetaGrad applied to convex loss functions f 1 , . . . , f T : W → R . Assume that for every t = 1 , . . . , T , subgradient g t ∈ ∂f t ( w t ) satisfies the conditions (3) in Proposition 2.5. Then, it holds that

<!-- formula-not-decoded -->

We outline how this result applies to exp-concave losses. Taking W , G , and H to be constants and ignoring the additive term of O ( n ln( T/n )) for simplicity, we have ˜ R u T = O ( √ n ln T · V u T ) . If all f t are α -exp-concave for some α ≤ 1 / ( GW ) , then f t ( w t ) -f t ( u ) ≤ ⟨ w t -u, g t ⟩ -α 2 ⟨ w t -u, g t ⟩ 2 holds (e.g., Hazan [26, Lemma 4.3]). Summing this over t and using Proposition 2.6 yield

<!-- formula-not-decoded -->

where the last inequality is due to √ ax -bx ≤ a 4 b for any a ≥ 0 , b &gt; 0 , and x ≥ 0 . Remarkably, MetaGrad achieves the O ( n α ln T ) regret bound without prior knowledge of α , whereas ONS achieves this regret bound by using the α value. Furthermore, even when some f t are not exp-concave, MetaGrad still enjoys a regret bound of O ( √ T ln ln T ) [56, Corollary 8]. As such, MetaGrad can automatically adapt to the unknown curvature of loss functions (at the cost of the negligible ln ln T factor), which is the key feature of universal online learning methods.

## 3 An efficient O ( n ln T ) -regret method based on ONS

This section presents an efficient logarithmic-regret method for online inverse linear optimization. Our method is remarkably simple: we apply ONS to exp-concave loss functions defined similarly to the η -experts' losses (4) used in MetaGrad. The proof is very short given the ONS's regret bound in Proposition 2.5. Despite this simplicity, we can achieve the regret bound of O ( n ln T ) , which matches the best-known regret upper bound of Gollapudi et al. [25], with far lower per-round complexity.

Theorem 3.1. Assume that for every t = 1 , . . . , T , action x t ∈ X t is optimal for c ∗ ∈ Θ . Let ˆ c 1 , . . . , ˆ c T ∈ Θ be the outputs of ONS applied to loss functions defined as follows for t = 1 , . . . , T :

<!-- formula-not-decoded -->

where ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ and we set η = 1 5 B . 6 Then, for R c ∗ T and ˜ R c ∗ T in (2) , it holds that

<!-- formula-not-decoded -->

5 In Van Erven et al. [56, Corollary 8], the multiplicative factor of H in the second term and the denominators of Hn in ln are replaced with WG and n , respectively. We modify it to obtain the above bound; see Appendix B.

6 This is equivalent to MetaGrad with a single 1 5 B -expert applied to the suboptimality losses, ℓ 1 , . . . , ℓ T .

Proof. Consider using Proposition 2.5 in the current setting with W = Θ , w η t = w t = ˆ c t , g t = ˆ x t -x t , u = c ∗ , W = D , G = K , and H = B . Since the optimality of x t and ˆ x t for c ∗ and ˆ c t , respectively, ensures ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ ≥ 0 , we have ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ 2 ≤ B ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ due to Assumption 2.2. Therefore, ˜ R c ∗ T = ∑ T t =1 ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ and V c ∗ T := ∑ T t =1 ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ 2 satisfy V c ∗ T ≤ B ˜ R c ∗ T . By using this and Proposition 2.5 with η = 1 5 B , for some constant C ONS &gt; 0 , it holds that

<!-- formula-not-decoded -->

and rearranging the terms yields ˜ R c ∗ T = O ( Bn ln ( DKT Bn )) . 7 This also applies to R c ∗ T ≤ ˜ R c ∗ T .

Time complexity. We discuss the time complexity of the method. Let τ solve be the time for solving linear optimization to find ˆ x t and τ G-proj the time for the generalized projection onto Θ used in ONS (see Appendix B.1). In each round t , we compute ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ in τ solve time; after that, the ONS update takes O ( n 2 + τ G-proj ) time. Therefore, it runs in O ( τ solve + n 2 + τ G-proj ) time per round, which is independent of T . If problem (1) is an LP, τ solve equals the time for solving the LP (cf. Cohen et al. [16] and Jiang et al. [33]). Also, τ G-proj is often affordable as Θ is usually specified by the learner and hence has a simple structure. For example, if Θ is the unit Euclidean ball, the generalized projection can be computed in O ( n 3 ) time by singular value decomposition (e.g., Mhammedi et al. [41, Section 4.1]). We may also use the quasi-Newton-type method for further efficiency [40].

## 4 Robustness to suboptimal feedback with MetaGrad

In practice, assuming that the agent's actions are always optimal is unrealistic. This section discusses how to handle suboptimal feedback effectively. Here, we let x t ∈ X t denote an arbitrary action taken by the agent, which the learner observes. Now that x t may have nothing to do with c ∗ , we can no longer ensure meaningful bounds on the regret that compares ˆ x t with optimal actions. For example, if revealed actions x t remain all zeros for t = 1 , . . . , T , we can learn nothing about c ∗ , and hence the regret that compares ˆ x t with optimal actions grows linearly in T in the worst case. Considering this issue, we highlight that the regret, R c ∗ T = ∑ T t =1 ⟨ c ∗ , x t -ˆ x t ⟩ , used here is defined with the agent's possibly suboptimal actions x t , not with those optimal for c ∗ . Small upper bounds on this regret ensure that, if the agent's actions x t are nearly optimal for c ∗ , so are ˆ x t . This regret still satisfies R c ∗ T ≤ ˜ R c ∗ T = ∑ T t =1 ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ since ˆ x t is optimal for ˆ c t . Additionally, recall that the suboptimality loss, ℓ t , in Definition 2.3 can be defined for any action x t ∈ X t and that ℓ t ( c ∗ ) = max x ∈ X t ⟨ c ∗ , x ⟩ - ⟨ c ∗ , x t ⟩ ≥ 0 indicates the suboptimality of x t for c ∗ . Below, we use ∆ T := ∑ T t =1 ℓ t ( c ∗ ) to denote the cumulative suboptimality of the agent's actions x t .

In this setting, it is not difficult to show that ONS used in Theorem 3.1 enjoys a regret bound that scales linearly with ∆ T . However, the linear dependence on ∆ T is not satisfactory, as it results in a regret bound of O ( T ) even for small suboptimality that persists across all rounds. The following theorem ensures that by applying MetaGrad to the suboptimality losses, we can obtain a regret bound that scales with √ ∆ T .

Theorem 4.1. Let ˆ c 1 , . . . , ˆ c T ∈ Θ be the outputs of MetaGrad applied to the suboptimality losses, ℓ 1 , . . . , ℓ T , given in Definition 2.3. Let ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ for t = 1 , . . . , T . Then, it holds that

<!-- formula-not-decoded -->

Proof. Similar to the proof of Theorem 3.1, we apply Proposition 2.6 with W = Θ , w t = ˆ c t , g t = ˆ x t -x t , u = c ∗ , W = D , G = K , and H = B ; in addition, g t = ˆ x t -x t ∈ ∂ℓ t (ˆ c t ) holds due

7 We may use any η as long as ηB &lt; 1 holds; η = 1 5 B is for consistency with MetaGrad in Appendix B.

to Proposition 2.4. Thus, Proposition 2.6 ensures the following bound for some constant C MG &gt; 0 : 8

<!-- formula-not-decoded -->

where ˜ R c ∗ T = ∑ T t =1 ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ and V c ∗ T = ∑ T t =1 ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ 2 . In contrast to the case of Theorem 3.1, ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ 2 ≤ B ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ is not ensured since ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ can be negative due to the suboptimality of x t . Instead, we will show that the following inequality holds:

<!-- formula-not-decoded -->

If ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ ≥ 0 , (7) is immediate from ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ 2 ≤ B ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ and ℓ t ( c ∗ ) ≥ 0 . If ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ &lt; 0 , ⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ 2 ≤ B ( -⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ ) holds. In addition, we have ℓ t ( c ∗ ) = max x ∈ X t ⟨ c ∗ , x ⟩-⟨ c ∗ , x t ⟩ ≥ ⟨ c ∗ , ˆ x t -x t ⟩ ≥ ⟨ c ∗ , ˆ x t -x t ⟩-⟨ ˆ c t , ˆ x t -x t ⟩ = -⟨ ˆ c t -c ∗ , ˆ x t -x t ⟩ , where the second inequality follows from ⟨ ˆ c t , ˆ x t -x t ⟩ ≥ 0 . Multiplying both sides by 2 yields

<!-- formula-not-decoded -->

Thus, (7) holds in any case, and hence V c ∗ T ≤ B ˜ R c ∗ T +2 B ∆ T . Substituting this into (6), we obtain

<!-- formula-not-decoded -->

We assume ˜ R c ∗ T &gt; 0 ; otherwise, the trivial bound of ˜ R c ∗ T ≤ 0 holds. By the subadditivity of x ↦→ √ x for x ≥ 0 , we have ˜ R c ∗ T ≤ √ a ˜ R c ∗ T + b , where a = C 2 MG Bn ln ( DKT Bn ) and b = √ 2 a ∆ T + a C MG . Since x ≤ √ ax + b implies x = 4 3 x -x 3 ≤ 4 3 ( √ ax + b ) -x 3 = -1 3 ( √ x -2 √ a ) 2 + 4 3 ( a + b ) ≤ 4 3 ( a + b ) for any a, b, x ≥ 0 , we obtain ˜ R c ∗ T ≤ 4 3 ( a + b ) = O ( Bn ln ( DKT Bn ) + √ ∆ T Bn ln ( DKT Bn ) ) .

If every x t is optimal, i.e., ∆ T = 0 , the bound recovers that in Theorem 3.1. Note that MetaGrad requires no prior knowledge of ∆ T ; it automatically achieves the bound that scales with √ ∆ T , analogous to the original bound in Proposition 2.6 that scales with √ V u T . Moreover, a refined version of MetaGrad [56] enables us to achieve a similar bound without prior knowledge of K , B , or T (see Appendix B.4). Universal online learning methods shine in such scenarios where adaptivity to unknown quantities is desired. Another noteworthy point is that the last part of the proof uses the self-bounding technique [23, 60, 66]. Specifically, we derived ˜ R c ∗ T ≲ a + b from ˜ R c ∗ T ≤ √ a ˜ R c ∗ T + b , where the latter means that ˜ R c ∗ T is upper bounded by a term of lower order in ˜ R c ∗ T itself, hence the name self-bounding. We expect that the combination of universal online learning methods and self-bounding, through relations like V c ∗ T ≲ ˜ R c ∗ T +∆ T used above, will be a useful technique for deriving meaningful guarantees in online inverse linear optimization.

Time complexity. The use of MetaGrad comes with a slight increase in time complexity. First, as with the case of ONS, ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ is computed in each round, taking τ solve time. Then, each η -expert performs the ONS update, taking O ( n 2 + τ G-proj ) time. Since MetaGrad maintains Θ(ln T ) distinct η values, the total per-round time complexity is O ( τ solve +( n 2 + τ G-proj ) ln T ) . If the O ( τ G-proj ln T ) factor is a bottleneck, we can use more efficient universal algorithms [41, 61] to reduce the number of projections from Θ(ln T ) to 1 . Moreover, the O ( n 2 ) factor can also be reduced by sketching techniques (see Van Erven et al. [56, Section 5]).

## 4.1 Online-to-batch conversion

We briefly discuss the implication of Theorem 4.1 in the offline setting, where feedback follows some underlying distribution. As noted in Section 2.2, the bound in Theorem 4.1 applies to the regret with respect to the suboptimality loss, ∑ T t =1 ( ℓ t (ˆ c t ) -ℓ t ( c ∗ )) , since it is bounded by ˜ R c ∗ T from above. Therefore, the standard online-to-batch conversion (e.g., Orabona [45, Theorem 3.1]) implies the following convergence of the average prediction in terms of the suboptimality loss.

8 Here, we can simultaneously achieve ˜ R c ∗ T = O ( DK √ T ln ln T ) thanks to MetaGrad's guarantee [56, Corollary 8], which can yield a stronger bound when n is huge.

Corollary 4.2. For any non-empty and compact X ⊆ R n , x ∈ X , and c ∈ Θ , define the corresponding suboptimality loss as ℓ X,x ( c ) := max x ′ ∈ X ⟨ c, x ′ ⟩ - ⟨ c, x ⟩ . Let ∆ &gt; 0 and define X ∆ as the set of observations ( X,x ) with bounded suboptimality, ℓ X,x ( c ∗ ) ≤ ∆ . Assume that { ( X t , x t ) } T t =1 are drawn i.i.d. from some distribution on X ∆ (hence ∆ T ≤ ∆ T ). Let ˆ c 1 , . . . , ˆ c T ∈ Θ be the outputs of MetaGrad applied to the suboptimality losses ℓ t = ℓ X t ,x t for t = 1 , . . . , T . Then, it holds that

<!-- formula-not-decoded -->

Bärmann et al. [4, Theorem 3.14] also obtained a similar offline guarantee via the online-to-batch conversion. Their convergence rate is O ( 1 √ T ) even when ∆ = 0 , whereas our Corollary 4.2 offers the faster rate of O ( ln T T ) if ∆ = 0 . It also applies to the case of ∆ &gt; 0 , which is important in practice because stochastic feedback is rarely optimal at all times. We emphasize that if regret bounds scale linearly with ∆ T , the above online-to-batch conversion cannot ensure that the excess suboptimality loss (the left-hand side) converge to zero as T → 0 . This observation lends support to the importance of the √ ∆ T -dependent regret bound we established in Theorem 4.1.

## 5 Ω( n ) lower bound

We construct an instance where any online learner incurs an Ω( n ) regret, implying that the O ( n ln T ) upper bound is tight up to an O (ln T ) factor. More strongly, the following Theorem 5.1 shows that, for any B &gt; 0 that gives the tight upper bound in Assumption 2.2, no learner can achieve a regret smaller than Bn 4 , which means that the Bn factor in our Theorem 3.1 is inevitable.

Theorem 5.1. Let n be a positive integer and Θ = [ -1 √ n , + 1 √ n ] n . For any T ≥ n , B &gt; 0 , and the learner's outputs ˆ c 1 , . . . , ˆ c T ∈ Θ , there exist c ∗ ∈ Θ and X 1 , . . . , X T ⊆ R n such that

<!-- formula-not-decoded -->

hold, where R c ∗ T = ∑ T t =1 ⟨ c ∗ , x t -ˆ x t ⟩ , x t ∈ arg max x ∈ X t ⟨ c ∗ , x ⟩ , ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ , and the expectation is taken over the learner's possible randomness.

̸

Proof. We focus on the first n rounds and show that any learner must incur Bn 4 in these rounds; in the remaining rounds, we may use any instance since the optimality of x t for c ∗ ensures ⟨ c ∗ , x t -ˆ x t ⟩ ≥ 0 . For t = 1 , . . . , n , let X t = { x ∈ R n : -B 4 √ n ≤ x ( t ) ≤ B 4 √ n, x ( i ) = 0 for i = t } , where x ( i ) denotes the i th element of x . That is, X t is the line segment on the t th axis from -B 4 √ n to B 4 √ n . Then, max { ⟨ c -c ′ , x -x ′ ⟩ : c, c ′ ∈ Θ , x, x ′ ∈ X t } = B holds for each t = 1 , . . . , n . Let c ∗ ∈ Θ be a random vector such that each entry is -1 √ n or 1 √ n with probability 1 2 , which is drawn independently of any other randomness. Then, the optimal action, x t ∈ X t , which is zero everywhere except that its t th coordinate equals c ∗ ( t ) | c ∗ ( t ) | · B 4 √ n , achieves ⟨ c ∗ , x t ⟩ = B 4 . Note that the learner's t th prediction ˆ c t is independent of c ∗ ( t ) since it depends only on past observations, { ( X i , x i ) } t -1 i =1 , which have no information about c ∗ ( t ) . Thus, ˆ x t ∈ arg max x ∈ X t ⟨ ˆ c t , x ⟩ is also independent of c ∗ ( t ) , and hence

<!-- formula-not-decoded -->

where the expectation is taken over the randomness of c ∗ . This implies that any deterministic learner incurs Bn 4 in the first n rounds in expectation. Thanks to Yao's minimax principle [62], we can conclude that for any randomized learner, there exists c ∗ ∈ Θ such that E [ R c ∗ T ] ≥ Bn 4 holds.

In the above proof, we restricted X 1 , . . . , X T to line segments so that each x t ∈ arg max x ∈ X t ⟨ c ∗ , x ⟩ reveals nothing about c ∗ ( t +1) , . . . , c ∗ ( n ) . Whether a similar lower bound holds when all X t are fulldimensional remains an open question. Another side note is that the Ω( n ) lower bound does not contradict the O ( √ T ) upper bound of Bärmann et al. [4]. Their OGD-based method indeed achieves a regret bound of O ( DK √ T ) , where D and K are upper bounds on the ℓ 2 -diameters of Θ and X t , respectively. In the above proof, T ≥ n , D ≥ 1 , and K ≥ B 2 √ n hold, implying that their regret upper bound is DK √ T ≳ Bn . Hence, the Ω( n ) -lower bound and their O ( DK √ T ) -upper bound are compatible.

## 6 Conclusion and discussion

We have presented an efficient ONS-based method that achieves an O ( n ln T ) -regret bound for online inverse linear optimization. Then, we have extended the method to deal with suboptimal feedback based on MetaGrad, achieving an O ( n ln T + √ ∆ T n ln T ) -regret bound, where ∆ T is the cumulative suboptimality of the agent's actions. Finally, we have presented a lower bound of Ω( n ) , which shows that the O ( n ln T ) upper bound is tight up to an O (ln T ) factor. Regarding limitations, our work is restricted to the case where the agent's optimization problem is linear, as mentioned in Footnote 1; how to deal with non-linearity is an important direction for future work. In online portfolio selection, ONS is efficient but inferior to the universal portfolio algorithm regarding the dependence on the gradient norm [57]. Exploring possible similar relationships in online inverse linear optimization is left for future work. Last but not least, closing the O (ln T ) gap between the upper and lower bounds is an important open problem. Interestingly, if all X t are line segments as in Section 5 and the learner can observe X t in the beginning of round t , the algorithm of Gollapudi et al. [25, Theorem 5.2] offers a regret upper bound of O ( n 5 log 2 n ) , which is finite and polynomial in n . We also provide an additional discussion on a finite regret bound for the case of n = 2 in Appendix C.

## Acknowledgments and Disclosure of Funding

SS was supported by JST ERATO Grant Number JPMJER1903. TT was supported by JST ACT-X Grant Number JPMJAX210E and JSPS KAKENHI Grant Number JP24K23852. HB was supported by JST PRESTO Grant Number JPMJPR24K6. TO was supported by JST FOREST Grant Number JPMJFR232L and JSPS KAKENHI Grant Number JP22K17853.

## References

- [1] Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems , volume 24, pages 2312-2320. Curran Associates, Inc., 2011 (cited on page 3).
- [2] R. K. Ahuja and J. B. Orlin. Inverse optimization. Operations Research , 49(5):771-783, 2001 (cited on pages 1, 3).
- [3] A. Aswani, Z.-J. (Max) Shen, and A. Siddiq. Inverse optimization with noisy data. Operations Research , 66(3):870-892, 2018 (cited on page 3).
- [4] A. Bärmann, A. Martin, S. Pokutta, and O. Schneider. An online-learning approach to inverse optimization. arXiv:1810.12997 , 2020 (cited on pages 2-5, 9, 21, 28).
- [5] A. Bärmann, S. Pokutta, and O. Schneider. Emulating the expert: Inverse optimization through online learning. In Proceedings of the 34th International Conference on Machine Learning , volume 70, pages 400-410. PMLR, 2017 (cited on pages 2-5, 21).
- [6] D. Bertsimas, V. Gupta, and I. C. Paschalidis. Data-driven estimation in equilibrium using inverse optimization. Mathematical Programming , 153(2):595-633, 2015 (cited on pages 2, 3).
- [7] O. Besbes, Y. Fonseca, and I. Lobel. Online learning from optimal actions. In Proceedings of the 34th Conference on Learning Theory , volume 134, pages 586-586. PMLR, 2021 (cited on pages 2-5, 21, 26).
- [8] O. Besbes, Y. Fonseca, and I. Lobel. Contextual inverse optimization: Offline and online learning. Operations Research , 73(1):424-443, 2025 (cited on pages 2-5, 21, 26-28).
- [9] J. R. Birge, A. Hortaçsu, and J. M. Pavlin. Inverse optimization for the recovery of market structure from market outcomes: An application to the MISO electricity market. Operations Research , 65(4):837-855, 2017 (cited on page 2).
- [10] J. R. Birge, X. Li, and C. Sun. Learning from stochastically revealed preference. In Advances in Neural Information Processing Systems , volume 35, pages 35061-35071. Curran Associates, Inc., 2022 (cited on page 3).
- [11] D. Burton and P. L. Toint. On an instance of the inverse shortest paths problem. Mathematical Programming , 53(1):45-61, 1992 (cited on page 1).

- [12] T. C. Y. Chan, M. Eberg, K. Forster, C. Holloway, L. Ieraci, Y. Shalaby, and N. Yousefi. An inverse optimization approach to measuring clinical pathway concordance. Management Science , 68(3):1882-1903, 2022 (cited on page 2).
- [13] T. C. Y. Chan, T. Lee, and D. Terekhov. Inverse optimization: Closed-form solutions, geometry, and goodness of fit. Management Science , 65(3):1115-1135, 2019 (cited on page 1).
- [14] T. C. Y. Chan, R. Mahmood, and I. Y. Zhu. Inverse optimization: Theory and applications. Operations Research , 73(2):1046-1074, 2025 (cited on page 1).
- [15] V. X. Chen and F. Kılınç-Karzan. Online convex optimization perspective for learning from dynamically revealed preferences. arXiv:2008.10460 , 2020 (cited on page 3).
- [16] M. B. Cohen, Y. T. Lee, and Z. Song. Solving linear programs in the current matrix multiplication time. Journal of the ACM , 68(1):1-39, 2021 (cited on page 7).
- [17] T. M. Cover and E. Ordentlich. Universal portfolios with side information. IEEE Transactions on Information Theory , 42(2):348-363, 1996 (cited on page 4).
- [18] V. Dani, T. P. Hayes, and S. M. Kakade. Stochastic linear optimization under bandit feedback. In Proceedings of the 21st Conference on Learning Theory , pages 355-366. PMLR, 2008 (cited on page 3).
- [19] J. M. Danskin. The theory of max-min, with applications. SIAM Journal on Applied Mathematics , 14(4):641-664, 1966 (cited on page 5).
- [20] C. Dong, Y. Chen, and B. Zeng. Generalized inverse optimization through online learning. In Advances in Neural Information Processing Systems , volume 31, pages 86-95. Curran Associates, Inc., 2018 (cited on page 3).
- [21] V. Feldman, C. Guzman, and S. Vempala. Statistical query algorithms for mean vector estimation and stochastic convex optimization. arXiv:1512.09170 , 2015 (cited on page 22).
- [22] M. Frank and P. Wolfe. An algorithm for quadratic programming. Naval Research Logistics Quarterly , 3(1-2):95-110, 1956 (cited on page 22).
- [23] P. Gaillard, G. Stoltz, and T. van Erven. A second-order bound with excess losses. In Proceedings of the 27th Conference on Learning Theory , volume 35, pages 176-196. PMLR, 2014 (cited on pages 3, 8).
- [24] D. Garber and N. Wolf. Frank-Wolfe with a nearest extreme point oracle. In Proceedings of the 34th Conference on Learning Theory , volume 134, pages 2103-2132. PMLR, 2021 (cited on page 22).
- [25] S. Gollapudi, G. Guruganesh, K. Kollias, P. Manurangsi, R. Paes Leme, and J. Schneider. Contextual recommendations and low-regret cutting-plane algorithms. In Advances in Neural Information Processing Systems , volume 34, pages 22498-22508. Curran Associates, Inc., 2021 (cited on pages 2-6, 10, 21, 22, 26, 28).
- [26] E. Hazan. Introduction to online convex optimization. arXiv:1909.05207 , 2023. https://arxiv. org/abs/1909.05207v3 (cited on pages 2, 5, 6, 22-24).
- [27] E. Hazan, A. Agarwal, and S. Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning , 69(2):169-192, 2007 (cited on pages 2, 3).
- [28] C. Heuberger. Inverse combinatorial optimization: A survey on problems, methods, and results. Journal of Combinatorial Optimization , 8(3):329-361, 2004 (cited on page 1).
- [29] G. Iyengar and W. Kang. Inverse conic programming with applications. Operations Research Letters , 33(3):319-330, 2005 (cited on page 3).
- [30] S. Jabbari, R. M. Rogers, A. Roth, and S. Z. Wu. Learning from rational behavior: Predicting solutions to unknown linear programs. In Advances in Neural Information Processing Systems , volume 29, pages 1570-1578. Curran Associates, Inc., 2016 (cited on page 3).
- [31] M. Jaggi. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In Proceedings of the 30th International Conference on Machine Learning , volume 28, pages 427-435. PMLR, 2013 (cited on page 22).
- [32] R. Jézéquel, D. M. Ostrovskii, and P. Gaillard. Efficient and near-optimal online portfolio selection. arXiv:2209.13932 , 2022 (cited on page 4).
- [33] S. Jiang, Z. Song, O. Weinstein, and H. Zhang. A faster algorithm for solving general LPs. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing , pages 823832. ACM, 2021 (cited on page 7).
- [34] A. Kalai and S. Vempala. Efficient algorithms for universal portfolios. Journal of Machine Learning Research , 3(Nov):423-440, 2002 (cited on page 4).

- [35] A. Keshavarz, Y. Wang, and S. Boyd. Imputing a convex objective function. In Proceedings of the 2011 IEEE International Symposium on Intelligent Control , pages 613-619. IEEE, 2011 (cited on page 3).
- [36] A. Krishnamurthy, T. Lykouris, C. Podimata, and R. Schapire. Contextual search in the presence of irrational agents. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing , pages 910-918. ACM, 2021 (cited on page 3).
- [37] S. Lacoste-Julien and M. Jaggi. On the global linear convergence of Frank-Wolfe optimization variants. In Advances in Neural Information Processing Systems , volume 28, pages 496-504. Curran Associates, Inc., 2015 (cited on page 22).
- [38] A. Liu, R. Paes Leme, and J. Schneider. Optimal contextual pricing and extensions. In Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms , pages 1059-1078. SIAM, 2021 (cited on page 3).
- [39] Y. Long, T. Ok, P. Zattoni Scroccaro, and P. Mohajerin Esfahani. Scalable kernel inverse optimization. In Advances in Neural Information Processing Systems , volume 37, pages 9946499487. Curran Associates, Inc., 2024 (cited on pages 2, 3).
- [40] Z. Mhammedi and K. Gatmiry. Quasi-Newton steps for efficient online exp-concave optimization. In Proceedings of the 36th Conference on Learning Theory , volume 195, pages 44734503. PMLR, 2023 (cited on page 7).
- [41] Z. Mhammedi, W. M. Koolen, and T. van Erven. Lipschitz adaptivity with multiple learning rates in online learning. In Proceedings of the 32nd Conference on Learning Theory , volume 99, pages 2490-2511. PMLR, 2019 (cited on pages 7, 8, 26).
- [42] S. K. Mishra, A. Raj, and S. Vaswani. From inverse optimization to feasibility to ERM. In Proceedings of the 41st International Conference on Machine Learning , volume 235, pages 35805-35828. PMLR, 2024 (cited on page 3).
- [43] P. Mohajerin Esfahani, S. Shafieezadeh-Abadeh, G. A. Hanasusanto, and D. Kuhn. Data-driven inverse optimization with imperfect information. Mathematical Programming , 167:191-234, 2018 (cited on pages 2, 3, 5).
- [44] A. Y. Ng and S. J. Russell. Algorithms for inverse reinforcement learning. In Proceedings of the 17th International Conference on Machine Learning , pages 663-670. Morgan Kaufmann Publishers Inc., 2000 (cited on page 2).
- [45] F. Orabona. A modern introduction to online learning. arXiv:1912.13213 , 2023. https://arxiv. org/abs/1912.13213v6 (cited on page 8).
- [46] R. Paes Leme, C. Podimata, and J. Schneider. Corruption-robust contextual search through density updates. In Proceedings of the 35th Conference on Learning Theory , volume 178, pages 3504-3505. PMLR, 2022 (cited on page 3).
- [47] R. Paes Leme, C. Podimata, and J. Schneider. Density-based algorithms for corruption-robust contextual search and convex optimization. arXiv:2206.07528 , 2025 (cited on page 3).
- [48] R. Paes Leme and J. Schneider. Contextual search via intrinsic volumes. SIAM Journal on Computing , 51(4):1096-1125, 2022 (cited on page 3).
- [49] S. Sakaue, H. Bao, and T. Tsuchiya. Revisiting online learning approach to inverse linear optimization: A Fenchel-Young loss perspective and gap-dependent regret analysis. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics , volume 258, pages 46-54. PMLR, 2025 (cited on pages 3, 5).
- [50] L. Shi, G. Zhang, H. Zhen, J. Fan, and J. Yan. Understanding and generalizing contrastive learning from the inverse optimal transport perspective. In Proceedings of the 40th International Conference on Machine Learning , volume 202, pages 31408-31421. PMLR, 2023 (cited on page 2).
- [51] C. Sun, S. Liu, and X. Li. Maximum optimality margin: A unified approach for contextual linear programming and inverse linear programming. In Proceedings of the 40th International Conference on Machine Learning , volume 202, pages 32886-32912. PMLR, 2023 (cited on page 3).
- [52] Y. Tan, D. Terekhov, and A. Delong. Learning linear programs from optimal decisions. In Advances in Neural Information Processing Systems , volume 33, pages 19738-19749. Curran Associates, Inc., 2020 (cited on page 3).
- [53] A. Tarantola. Inverse problem theory: Methods for data fitting and model parameter estimation. Geophysical Journal International , 94(1):167-167, 1988 (cited on page 1).

- [54] C.-E. Tsai, H.-C. Cheng, and Y.-H. Li. Online self-concordant and relatively smooth minimization, with applications to online portfolio selection and learning quantum states. In Proceedings of the 34th International Conference on Algorithmic Learning Theory , volume 201, pages 1481-1483. PMLR, 2023 (cited on page 4).
- [55] T. van Erven and W. M. Koolen. MetaGrad: Multiple learning rates in online learning. In Advances in Neural Information Processing Systems , volume 29, pages 3666-3674. Curran Associates, Inc., 2016 (cited on pages 2, 3, 25).
- [56] T. van Erven, W. M. Koolen, and D. van der Hoeven. MetaGrad: Adaptation using multiple learning rates in online learning. Journal of Machine Learning Research , 22(161):1-61, 2021 (cited on pages 2, 3, 5, 6, 8, 22, 26).
- [57] T. van Erven, D. van der Hoeven, W. Kotłowski, and W. M. Koolen. Open problem: Fast and optimal online portfolio selection. In Proceedings of the 33rd Conference on Learning Theory , volume 125, pages 3864-3869. PMLR, 2020 (cited on pages 4, 10).
- [58] G. Wang, S. Lu, and L. Zhang. Adaptivity and optimality: A universal algorithm for online convex optimization. In Proceedings of the 35th Uncertainty in Artificial Intelligence Conference , volume 115, pages 659-668. PMLR, 2020 (cited on pages 3, 25).
- [59] A. Ward, N. Master, and N. Bambos. Learning to emulate an expert projective cone scheduler. In Proceedings of the 2019 American Control Conference , pages 292-297. IEEE, 2019 (cited on page 3).
- [60] C.-Y. Wei and H. Luo. More adaptive algorithms for adversarial bandits. In Proceedings of the 31st Conference On Learning Theory , volume 75, pages 1263-1291. PMLR, 2018 (cited on pages 3, 8).
- [61] W. Yang, Y. Wang, P. Zhao, and L. Zhang. Universal online convex optimization with 1 projection per round. In Advances in Neural Information Processing Systems , volume 37, pages 31438-31472. Curran Associates, Inc., 2024 (cited on page 8).
- [62] A. C.-C. Yao. Probabilistic computations: Toward a unified measure of complexity. In Proceedings of the 18th Annual Symposium on Foundations of Computer Science , pages 222-227. IEEE, 1977 (cited on page 9).
- [63] P. Zattoni Scroccaro, B. Atasoy, and P. Mohajerin Esfahani. Learning in inverse optimization: Incenter cost, augmented suboptimality loss, and algorithms. Operations Research , 0(0):1-19, 2024 (cited on page 3).
- [64] L. Zhang, G. Wang, J. Yi, and T. Yang. A simple yet universal strategy for online convex optimization. In Proceedings of the 39th International Conference on Machine Learning , volume 162, pages 26605-26623. PMLR, 2022 (cited on page 3).
- [65] J. Zimmert, N. Agarwal, and S. Kale. Pushing the efficiency-regret Pareto frontier for online learning of portfolios and quantum states. In Proceedings of the 35th Conference on Learning Theory , volume 178, pages 182-226. PMLR, 2022 (cited on page 4).
- [66] J. Zimmert and Y. Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research , 22(28):1-49, 2021 (cited on pages 3, 8).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: See Abstract, Section 1, and Section 1.1.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Section 2.2 for assumptions, Sections 3 and 4 for the computational complexity, Section 5 for limitations regarding the lower bound, and Section 6 for additional discussions.

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

Justification: See Section 2.2 for assumptions. All theorems are followed by proofs.

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

Justification: Appendix D provides a detailed description of our preliminary experiments.

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

Justification: The source code and its readme file are available at https://github.com/ssakaue/ online-inverse-linear-optimization-code.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/ guides/CodeSubmissionPolicy) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: See Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Table 2 reports the results with the mean and standard deviation.

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

Justification: See Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

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
- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

## A Detailed comparisons with previous results

Below we compare our results with Bärmann et al. [4, 5], Besbes et al. [7, 8], and Gollapudi et al. [25].

Bärmann et al. [4, 5] used ˜ R c ∗ T as the performance measure, as with our Theorems 3.1 and 4.1, and provided two specific methods. The first one, based on the multiplicative weights update (MWU), is tailored for the case where Θ is the probability simplex, i.e., Θ = { c ∈ R n | c ≥ 0 , ∥ c ∥ 1 = 1 } . The authors assumed a bound of K ∞ &gt; 0 on the ℓ ∞ -diameters of X t and obtained a regret bound of O ( K ∞ √ T ln n ) . The second one is based on the online gradient descent (OGD) and applies to general convex sets Θ . The authors assumed that the ℓ 2 -diameters of Θ and X t are bounded by D &gt; 0 and K &gt; 0 , respectively, and obtained a regret bound of O ( DK √ T ) . In the first case, our Theorem 3.1 with B = K ∞ , D = √ 2 , and K ≤ 2 √ nK ∞ offers a bound of O ( K ∞ n ln( T/ √ n )) ; in the second case, we obtain a bound of O ( DKn ln( T/n )) by setting B = DK . In both cases, our bounds improve the dependence on T from √ T to ln T , while scaled up by a factor of n , up to logarithmic terms. Regarding the computation time, their MWU and OGD methods run in O ( τ solve + τ E-proj + n ) time per round, where τ E-proj is the time for the Euclidean projection onto Θ , hence faster than our method. Also, suboptimal feedback is discussed in Bärmann et al. [4, Sections 3.1]. However, their bound does not achieve the logarithmic dependence on T even when ∆ T = 0 , unlike our Theorem 4.1.

Besbes et al. [7, 8] used R c ∗ T as the performance measure, which is upper bounded by ˜ R c ∗ T . They assumed that c ∗ lies in the unit Euclidean sphere and that the ℓ 2 -diameters of X t are at most one. Under these conditions, they obtained the first logarithmic regret bound of O ( n 4 ln T ) . By applying Theorem 3.1 to this case, we obtain a bound of O ( n ln( T/n )) , which is better than their bound by a factor of n 3 . As discussed in Section 1, their method relies on the idea of narrowing down regions represented with O ( T ) constraints, and hence it seems inefficient for large T ; indeed Besbes et al. [8, Theorem 4] only claims that the total time complexity is polynomial in n and T . Considering this, our ONS-based method is arguably much faster while achieving the better regret bound.

On the problem setting of Besbes et al. [7, 8]. As mentioned in Remark 2.1, the problem setting of Besbes et al. [7, 8] is seemingly different from ours. In their setting, in each round t , the learner first observes ( X t , f t ) , where f t : X t → R n is called a context function . Then, the learner chooses ˆ x t ∈ X t and receives an optimal action x t ∈ arg max x ∈ X t ⟨ c ∗ , f t ( x ) ⟩ as feedback. It is assumed that the learner can solve max x ∈ X t ⟨ c, f t ( x ) ⟩ for any c ∈ R n and that all f t are 1 -Lipschitz, i.e., ∥ f t ( x ) -f t ( x ′ ) ∥ 2 ≤ ∥ x -x ′ ∥ 2 for all x, x ′ ∈ X t . We note that our methods work in this setting, while the presence of f t might make their setting appear more general. Specifically, we redefine X t as the image of f t , i.e., { f t ( x ) : x ∈ X t } . Then, their assumption ensures that we can find f t (ˆ x t ) ∈ X t that maximizes X t ∋ ξ ↦→⟨ ˆ c t , ξ ⟩ , and the ℓ 2 -diameter of the newly defined X t is bounded by 1 due to the 1 -Lipschitzness of f t . Therefore, by defining g t = f t (ˆ x t ) -f t ( x t ) and applying it in Theorems 3.1 and 4.1, we recover the bounds therein on ∑ T t =1 ⟨ ˆ c t -c ∗ , f t (ˆ x t ) -f t ( x t ) ⟩ , with D , K , and B being constants. The bounds also apply to the regret, ∑ T t =1 ⟨ c ∗ , f t ( x t ) -f t (ˆ x t ) ⟩ , used in Besbes et al. [7, 8]. Additionally, Besbes et al. [7, 8] consider a (possibly non-convex) initial knowledge set C 0 ⊆ R n that contains c ∗ . We note, however, that they do not care about whether predictions ˆ c t lie in C 0 or not since the regret, their performance measure, does not explicitly involve ˆ c t . Indeed, predictions ˆ c t that appear in their method are chosen from ellipsoidal cones that properly contain C 0 in general. Therefore, our methods carried out on a convex set Θ ⊇ C 0 work similarly in their setting.

Gollapudi et al. [25] studied essentially the same problem as online inverse linear optimization under the name of contextual recommendation (where they and Besbes et al. [7, 8] appear to have been unaware of each other's work). As with Besbes et al. [7, 8], Gollapudi et al. [25] assumed that c ∗ and X 1 , . . . , X T lie in the unit Euclidean ball, denoted by B n . Similar to Besbes et al. [7, 8], their method maintains the region K t , which is the intersection of hyperplanes { c ∈ R d : ⟨ c -ˆ c s , x s -ˆ x s ⟩ ≥ 0 } for s = 1 , . . . , t -1 , and sets ˆ c t to the centroid of K t + 1 T B n , where + is the Minkowski sum. As regards the regret analysis, their key idea is to use the approximate Grünbaum theorem: whenever the learner incurs ⟨ c ∗ , x t -ˆ x t ⟩ ≥ 1 T , Vol ( K t + 1 T B n ) decreases by a constant factor, where Vol denotes the volume. Consequently, Vol ( K 1 + 1 T B n ) / Vol ( K T + 1 T B n ) ≲ T n implies the regret bound of R c ∗ T = ∑ t ⟨ c ∗ , x t -ˆ x t ⟩ = O ( n ln T ) . As such, the per-round complexity of their method also inherently depends on T , and Gollapudi et al. [25, Section 1.2] only claims the total time complexity of poly( n, T ) . In this setting, our ONS-based method achieves a regret bound of O ( n ln( T/n )) and is arguably more efficient since the per-round complexity is independent of T .

## Algorithm 1 Online Newton Step

```
1: Set γ = 1 2 min { 1 β , α } , ε = n W 2 γ 2 , A 0 = εI n , and w 1 ∈ W . 2: for t = 1 , . . . , T : 3: Play w t and observe q t . 4: A t ← A t -1 + ∇ q t ( w t ) ∇ q t ( w t ) ⊤ . 5: w t +1 ← arg min { ∥ ∥ ∥ w t -1 γ A -1 t ∇ q t ( w t ) -w ∥ ∥ ∥ 2 A t : w ∈ W } . ▷
```

<!-- formula-not-decoded -->

Estimating the per-round complexity of Gollapudi et al. [25]. As described above, the method of Gollapudi et al. [25] requires ˆ x t for each t , and hence the per-round complexity involves τ solve , the time to solve max x ∈ X t ⟨ ˆ c t , x ⟩ . Aside from this, its per-round complexity is dominated by the cost for computing the centroid of K t + 1 T B n , where K t is represented by O ( T ) hyperplanes. It is known that the problem of exactly computing the centroid is #P-hard in general, but we can approximate it via sampling with a membership oracle of K t + 1 T B n . To the best of our knowledge, computing a point that is ε -close to the centroid takes O ( n 4 /ε 2 ) membership queries, up to logarithmic factors [21, Theorem 5.7], and it is natural to set ε = 1 /T to make the approximation error negligible. Thus, it takes O ( n 4 T 2 ) membership queries. Regarding the complexity of the membership oracle, naively checking whether a given point satisfies all the O ( T ) linear constraints of K t takes O ( nT ) time. Handling the Minkowski sum with 1 T B would complicates the procedure, though it can be done in poly( n, T ) time by using, for example, Frank-Wolfe-type algorithms [22, 24, 31, 37]. For now, O ( nT ) would be a reasonable (optimistic) estimate of the complexity of the membership oracle. Consequently, the total per-round complexity of their method is estimated to be O ( τ solve + n 5 T 3 ) (or higher).

## B Details of ONS and MetaGrad

We present the details of ONS and MetaGrad. The main purpose of this section is to provide simple descriptions and analyses of those algorithms, thereby assisting readers who are not familiar with them. As in Appendix B.4, we can also derive a regret bound of MetaGrad that yields a similar result to Theorem 4.1 directly from the results of Van Erven et al. [56].

First, we discuss the regret bound of ONS used by η -experts in MetaGrad, proving Proposition 2.5. Then, we establish the regret bound of MetaGrad in Proposition 2.6.

## B.1 Regret bound of ONS

Let I n ∈ R n × n denote the identity matrix. For any A,B ∈ R n × n , A ⪰ B means that A -B is positive semidefinite. For positive semidefinite A ∈ R n × n , let ∥ x ∥ A = √ x ⊤ Ax for x ∈ R n . Let W ⊆ R n be a closed convex set. A function q : W → R is α -exp-concave for some α &gt; 0 if W ∋ w ↦→ e -αq ( w ) is concave. For twice differentiable q , this is equivalent to ∇ 2 q ( w ) ⪰ α ∇ q ( w ) ∇ q ( w ) ⊤ . The following regret bound of ONS mostly comes from the standard analysis [26, Section 4.4], and hence readers familiar with it can skip the subsequent proof. The only modification lies in the use of β instead of Wλ (defined below), where β ≤ Wλ always holds and hence slightly tighter. This leads to the multiplicative factor of B , rather than DK , in Theorems 3.1 and 4.1.

Proposition B.1. Let W ⊆ R n be a closed convex set with the ℓ 2 -diameter of at most W &gt; 0 . Assume that q 1 , . . . , q T : W → R are twice differentiable and α -exp-concave for some α &gt; 0 . Additionally, assume that there exist β, λ &gt; 0 such that max w ∈W ∣ ∣ ∇ q t ( w t ) ⊤ ( w -w t ) ∣ ∣ ≤ β and ∥∇ q t ( w t ) ∥ 2 ≤ λ hold. Let w 1 , . . . , w T ∈ W be the outputs of ONS (Algorithm 1). Then, for any u ∈ W , it holds that

<!-- formula-not-decoded -->

where γ = 1 2 min { 1 β , α } is the parameter used in ONS.

Proof. We first give a useful inequality that follows from the α -exp-concavity. By the same analysis as the proof of Hazan [26, Lemma 4.3], for γ ≤ α 2 , we have

<!-- formula-not-decoded -->

Note that we also have ∣ ∣ 2 γ ∇ q t ( w t ) ⊤ ( u -w t ) ∣ ∣ ≤ 2 γβ ≤ 1 . Since ln(1 -x ) ≤ -x -x 2 / 4 holds for x ≥ -1 , applying this with x = 2 γ ∇ q t ( w t ) ⊤ ( u -w t ) yields

<!-- formula-not-decoded -->

We turn to the iterates of ONS. Since w t +1 is the projection of w t -1 γ A -1 t ∇ q t ( w t ) onto W with respect to the norm ∥·∥ A t , we have ∥ w t +1 -u ∥ 2 A t ≤ ∥ ∥ ∥ w t -1 γ A -1 t ∇ q t ( w t ) -u ∥ ∥ ∥ 2 A t for u ∈ W due to the Pythagorean theorem, hence

<!-- formula-not-decoded -->

Rearranging the terms, we obtain

<!-- formula-not-decoded -->

From A t = A t -1 + ∇ q t ( w t ) ∇ q t ( w t ) ⊤ , summing over t and ignoring γ 2 ( w T +1 -u ) ⊤ A T ( w T +1 -u ) ≥ 0 , we obtain

<!-- formula-not-decoded -->

Since we have A 1 -∇ q 1 ( w 1 ) ∇ q 1 ( w 1 ) ⊤ = A 0 = εI n and ε = n W 2 γ 2 , the above inequality implies

<!-- formula-not-decoded -->

The first term in the right-hand side is bounded as follows due to the celebrated elliptical potential lemma (e.g., Hazan [26, proof of Theorem 4.5]):

<!-- formula-not-decoded -->

where we used det A 0 = ε n and det A T = det ( ∑ T t =1 ∇ q t ( w t ) ∇ q t ( w t ) ⊤ + εI n ) ≤ ( Tλ 2 + ε ) n , which follows from the fact that eigenvalues of ∑ T t =1 ∇ q t ( w t ) ∇ q t ( w t ) ⊤ are at most Tλ 2 . Combining (8), (9), and (10), we obtain

<!-- formula-not-decoded -->

as desired.

## B.2 Regret bound of η -expert

We now establish the regret bound of ONS in Proposition 2.5, which is used by η -experts in MetaGrad. Let η ∈ ( 0 , 1 5 H ] and consider applying ONS to the following loss functions, which are defined in (4):

<!-- formula-not-decoded -->

As in Proposition 2.5, the ℓ 2 -diameter of W is at most W &gt; 0 , and the following conditions hold:

<!-- formula-not-decoded -->

Therefore, f η t satisfies the conditions in Proposition B.1 with α = 2 (1+2 ηH ) 2 , β = ηH +2 η 2 H 2 , and λ = η (1 + 2 ηH ) G . Since 1 α = 1 2 +2 ηH +2 η 2 H 2 ≥ β holds, we have γ = 1 2 min { 1 β , α } = α 2 . Thus, for any η ∈ ( 0 , 1 5 H ] , we have γ ∈ [ 25 49 , 1 ) ⊆ [ 1 2 , 1 ] and γλ = ηG 1+2 ηH ≤ G 7 H . Consequently, Proposition B.1 implies that for any u ∈ W , the regret of the η -expert's ONS is bounded as follows:

<!-- formula-not-decoded -->

## B.3 Regret bound of MetaGrad

We turn to MetaGrad applied to convex loss functions f 1 , . . . , f T : W → R . We here use w t ∈ W and g t ∈ ∂f t ( w t ) to denote the t th output of MetaGrad and a subgradient of f t at w t , respectively, for t = 1 , . . . , T . We assume that these satisfy the conditions in (3), as stated in Proposition 2.6.

Algorithm 2 describes the procedure of MetaGrad. Define η i = 2 -i 5 H for i = 0 , 1 , . . . , ⌈ 1 2 log 2 T ⌉ , called grid points , and let G ⊆ ( 0 , 1 5 H ] denote the set of all grid points. For each η ∈ G , η -expert runs ONS with loss functions f η 1 , . . . , f η T to compute w η 1 , . . . , w η T . In each round t , we obtain w t by

## Algorithm 2 MetaGrad

<!-- formula-not-decoded -->

- 2: for t = 1 , . . . , T :
- 4: Play w t = ∑ η ∈G ηp t w t ∑ η ∈G ηp η t .
- 3: Fetch w η t from η -experts for all η ∈ G .
- η η
- 5: Observe g t ∈ ∂f t ( w t ) and send ( w t , g t ) to η -experts for all η ∈ G .
- 6: p η t +1 ← p η t exp( -f η t ( w η t )) /Z t for all η ∈ G , where Z t = ∑ η ∈G p η t exp( -f η t ( w η t )) .

aggregating the η -experts' outputs w η t based on the exponentially weighted average method (EWA). We set the prior as p η i 1 = C ( i +1)( i +2) for all η i ∈ G , where C = 1 + 1 1+ ⌈ 1 2 log 2 T ⌉ . Then, it is known that for every η ∈ G , the regret of EWA relative to the η -expert's choice w η t is bounded as follows:

<!-- formula-not-decoded -->

where we used C ≥ 1 in the second inequality. We here omit the proof as it is completely the same as that of Van Erven and Koolen [55, Lemma 4] (see also Wang et al. [58, Lemma 1]).

We are ready to prove Proposition 2.6. Let V u T = ∑ T t =1 ⟨ w t -u, g t ⟩ 2 . By using f η t ( w t ) = 0 , (11), and (12), it holds that

<!-- formula-not-decoded -->

for all η ∈ G . For brevity, let

<!-- formula-not-decoded -->

If we knew V u T , we could set η to η ∗ := √ A V u T ≥ 1 5 H √ T to minimize the above regret bound, A η + ηV u T . Actually, we can do almost the same without knowing V u T thanks to the fact that the regret bound holds for all η ∈ G . If η ∗ ≤ 1 5 H , by construction we have a grid point η ∈ G such that η ∗ ∈ [ η 2 , η ] , hence

<!-- formula-not-decoded -->

Otherwise, η ∗ = √ A V u T ≥ 1 5 H holds, which implies V u T ≤ 25 H 2 A . Thus, for η 0 = 1 5 H ∈ G , we have

<!-- formula-not-decoded -->

Therefore, in any case, we have

<!-- formula-not-decoded -->

obtaining the regret bound in Proposition 2.6.

## Algorithm 3 O (1) -Regret Algorithm for n = 2 .

- 1: Set C 1 to S 1 .
- 2: for t = 1 , . . . , T :
- 3: Draw ˆ c t uniformly at random from C t .
- 4: Observe ( X t , x t ) .
- 5: C t +1 ←C t ∩ N t . ▷ N t is the normal cone.

## B.4 Lipschitz adaptivity and anytime guarantee

Recent studies [41, 56] have shown that MetaGrad can be further made Lipschitz adaptive and agnostic to the number of rounds. Specifically, MetaGrad given in Van Erven et al. [56, Algorithms 1 and 2] works without knowing G , H , or T in advance, while using (a guess of) W . By expanding the proofs of Van Erven et al. [56, Theorem 7 and Corollary 8], we can confirm that the refined version of MetaGrad enjoys the following regret bound:

<!-- formula-not-decoded -->

By using this in the proof of Theorem 4.1, we obtain

<!-- formula-not-decoded -->

and the algorithm does not require knowing K , B , T , or ∆ T in advance.

## C On removing the ln T factor: the case of n = 2

This section provides an additional discussion on closing the ln T gap in the upper and lower bounds on the regret. Specifically, focusing on the case of n = 2 , we provide a simple algorithm that achieves a regret bound of O (1) in expectation, removing the ln T factor. We also observe that extending the algorithm to general n ≥ 2 might be challenging. Note that Gollapudi et al. [25, Theorem 4.1] has already established a regret bound of exp( O ( n ln n )) as mentioned in Section 1.2, which implies an O (1) -regret bound for n = 2 . The purpose of this section is simply to stimulate discussions on closing the ln T gap by presenting another simple analysis. Below, let B n and S n -1 denote the unit Euclidean ball and sphere in R n , respectively, for any integer n &gt; 1 .

## C.1 An O (1) -regret method for n = 2

We focus on the case of n = 2 and present an algorithm that achieves a regret bound of O (1) in expectation. We assume that all x t ∈ X t are optimal for c ∗ for t = 1 , . . . , T . For simplicity, we additionally assume that all X t are contained in 1 2 B 2 and that c ∗ lies in S 1 . For any non-zero vectors c, c ′ ∈ R n , let θ ( c, c ′ ) denote the angle between the two vectors. The following lemma from Besbes et al. [8], which holds for general n ≥ 2 , is useful in the subsequent analysis.

<!-- formula-not-decoded -->

Our algorithm, given in Algorithm 3, is a randomized variant of the one investigated by Besbes et al. [7, 8]. The procedure is intuitive: we maintain a set C t ⊆ S 1 that contains c ∗ , from which we draw ˆ c t uniformly at random, and update C t by excluding the area that is ensured not to contain c ∗ based on the t th feedback ( X t , x t ) . Formally, the last step takes the intersection of C t and the normal cone N t = { c ∈ R n : ⟨ c, x t -x ⟩ ≥ 0 , ∀ x ∈ X t } of X t at x t , which is a convex cone containing c ∗ . Therefore, every C t is a connected arc on S 1 and is non-empty due to c ∗ ∈ C t (see Figure 1).

Theorem C.2. For the above setting of n = 2 , Algorithm 3 achieves E [ R c ∗ T ] ≤ 2 π .

Figure 1: Illustration of c ∗ , C t , N t , and C t +1 .

<!-- image -->

Figure 2: An example of C t on S 2 . The darker area, A ( C t ) , becomes arbitrarily small as ε → 0 , while θ ( c ∗ , ˆ c t ) does not.

<!-- image -->

Proof. For any connected arc C ⊆ S 1 , let A ( C ) ∈ [0 , 2 π ] denote its central angle, which equals its length. Fix C t . If ˆ c t ∈ C t ∩ int( N t ) , where int( · ) denotes the interior, ˆ x t = x t is the unique optimal solution for ˆ c t , hence ⟨ c ∗ , x t -ˆ x t ⟩ = 0 . Taking the expectation about the randomness of ˆ c t , we have

<!-- formula-not-decoded -->

where we used Pr[ˆ c t ∈ C t \ int( N t )] = Pr[ˆ c t ∈ C t \ N t ] = A ( C t \ N t ) /A ( C t ) (since the boundary of N t has zero measure). If A ( C t ) ≥ π/ 2 , from ⟨ c ∗ , x t -ˆ x t ⟩ ≤ ∥ c ∗ ∥ 2 ∥ x t -ˆ x t ∥ 2 ≤ 1 , we have

<!-- formula-not-decoded -->

If A ( C t ) &lt; π/ 2 , Lemma C.1 and ˆ c t , c ∗ ∈ C t imply ⟨ c ∗ , x t -ˆ x t ⟩ ≤ sin θ ( c ∗ , ˆ c t ) ≤ sin A ( C t ) . Thus, by using 1 x sin x ≤ 1 ( x ∈ R ), we obtain

<!-- formula-not-decoded -->

Therefore, we have E [ ⟨ c ∗ , x t -ˆ x t ⟩ ] ≤ A ( C t \ N t ) in any case. Consequently, we obtain

<!-- formula-not-decoded -->

where the last inequality is due to C t +1 = C t ∩ N t , which implies C s ⊆ C t and C s ∩ ( C t \ N t ) = ∅ for any s &gt; t , and hence no double counting occurs in the above summation.

## C.2 Discussion on higher-dimensional cases

Algorithm 3 might appear applicable to general n ≥ 2 by replacing S 1 with S n -1 and defining A ( C t ) as the area of C t ⊆ S n -1 . However, this idea faces a challenge in bounding the regret when extending the above proof to general n ≥ 2 . 9

As suggested in the proof of Theorem C.2, bounding E [ ⟨ c ∗ , x t -ˆ x t ⟩ ] is trickier when A ( C t ) is small (cf. the case of A ( C t ) &lt; π/ 2 ). Luckily, when n = 2 , we can bound it thanks to Lemma C.1 and sin θ ( c ∗ , ˆ c t ) ≤ sin A ( C t ) , where the latter roughly means the angle, θ ( c ∗ , ˆ c t ) , is bounded by the area, A ( C t ) , from above. Importantly, when n = 2 , both the central angle and the area of an arc are identified with the length of the arc, which is the key to establishing sin θ ( c ∗ , ˆ c t ) ≤ sin A ( C t ) . This is no longer true for n ≥ 3 . As in Figure 2, the area, A ( C t ) , can be arbitrarily small even if the angle

9 We note that a hardness result given in Besbes et al. [8, Theorem 2] is different from what we encounter here. They showed that their greedy circumcenter policy fails to achieve a sublinear regret, which stems from the shape of the initial knowledge set and the behavior of the greedy rule for selecting ˆ c t ; this differs from the issue discussed above.

within there, or the maximum θ ( c ∗ , ˆ c t ) for c ∗ , ˆ c t ∈ C t , is large. 10 This is why the proof for the case of n = 2 does not directly extend to higher dimensions. We leave closing the O (ln T ) gap for n ≥ 3 as an important open problem for future research.

## D Numerical experiments

We conducted numerical experiments to complement our theoretical results. Experiments were conducted on Google Colab equipped with an Intel ® Xeon ® CPU @ 2.20GHz, 12 GB RAM, running Ubuntu 22.04.4 LTS with Python 3.12.11. The code is available at https://github.com/ssakaue/ online-inverse-linear-optimization-code.

We use a setup based on the hard instance considered in our lower bound analysis (Section 5). Let c ∗ be a random vector with ∥ c ∗ ∥ 2 = 1 . The learner's prediction set Θ is the n -dimensional Euclidean unit ball. At each round t , we sample an endpoint v uniformly at random from the unit sphere in R n , and set X t = {-v, + v } . We report results for T = 10 , 000 rounds in dimensions n = 2 , 20 , and 200 . To mitigate randomness, we repeat each experiment 10 times independently and report mean and standard deviation.

Compared methods. We compare the following three methods:

- ONS : our proposed method based on ONS,
- OGD : the OGD-based method of Bärmann et al. [4],
- CP : a cutting-plane style method inspired by Gollapudi et al. [25].

For CP, computing the centroid of the feasible region is #P-hard. Therefore, we adopt a randomized heuristic: we pre-sample n × 10 4 candidate points from the unit ball, eliminate those violating accumulated cuts, and approximate the centroid by averaging the remaining points.

Results. Table 2 summarizes the cumulative regret and runtime over T = 10 , 000 rounds (mean ± standard deviation across 10 trials).

Table 2: Experimental results over T = 10 , 000 rounds (mean ± standard deviation across 10 independent runs).

(a) Cumulative Regret

n

= 20

52

07

= 2

±

2

.

±

±

Method

ONS

OGD

CP

7

8

2

.

.

.

n

81

46

77

4

0

.

31

33

.

55

.

±

0

.

80

±

43

0

.

50

.

68

829

.

91

±

287

.

00

n

= 200

46

.

19

67

515

.

.

±

0

.

88

41

50

±

±

1

.

36

43

.

75

(b) Cumulative Runtime (s)

| Method   | n = 2             | n = 20            | n = 200             |
|----------|-------------------|-------------------|---------------------|
| ONS      | 0 . 648 ± 0 . 115 | 0 . 705 ± 0 . 086 | 9 . 073 ± 0 . 727   |
| OGD      | 0 . 150 ± 0 . 024 | 0 . 150 ± 0 . 017 | 0 . 233 ± 0 . 019   |
| CP       | 4 . 670 ± 3 . 205 | 5 . 700 ± 1 . 748 | 71 . 252 ± 12 . 152 |

When the dimension is small ( n = 2 ), CP achieves the lowest cumulative regret. However, its cumulative regret deteriorates significantly for n = 20 and n = 200 . This degradation stems from the limited number of pre-sampled candidate points: in principle, about T n samples would be required for an accurate approximation, which is infeasible even for moderate n . Moreover, although the above randomized centroid approximation substantially reduces computation by averaging over the

10 A similar issue, though leading to different challenges, is noted in Besbes et al. [8, Section 4.4], where their method encounters ill-conditioned (or elongated) ellipsoids. They addressed this by appropriately determining when to update the ellipsoidal cone. The ln T factor arises as a result of balancing being ill-conditioned with the instantaneous regret.

surviving candidates, it remains less scalable than OGD and ONS. These results highlight practical limitations of CP in moderate to high-dimensional settings.

In contrast, ONS consistently achieves low regret values across all dimensions while remaining computationally feasible. It outperforms OGD in terms of the regret and scales reasonably well with increasing n . Note that our ONS implementation uses a straightforward projection subroutine that repeatedly solves similar linear systems-an overhead that could be reduced by more sophisticated implementation techniques. Further speedups could also be achieved via quasi-Newton-type updates or sketching-based techniques, as discussed in Sections 3 and 4.

Taken together, these findings affirm that ONS provides a strong and scalable alternative to existing methods in online inverse linear optimization, especially when CP is computationally infeasible and OGD's regret performance is unsatisfactory.
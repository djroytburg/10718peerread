## Tight Lower Bounds and Improved Convergence in Performative Prediction

Pedram Khorsandi Mila, Quebec AI Institute Université de Montréal

Rushil Gupta Mila, Quebec AI Institute Université de Montréal

Simon Lacoste-Julien ∗ Mila, Quebec AI Institute Université de Montréal Canada CIFAR AI Chair

Mehrnaz Mofakhami Mila, Quebec AI Institute Université de Montréal

Gauthier Gidel ∗ Mila, Quebec AI Institute Université de Montréal Canada CIFAR AI Chair

## Abstract

Performative prediction is a framework accounting for the shift in the data distribution induced by the prediction of a model deployed in the real world. Ensuring convergence to a stable solution-one at which the post-deployment data distribution no longer changes-is crucial in settings where model predictions can influence future data. This paper, for the first time, extends the Repeated Risk Minimization (RRM) algorithm class by utilizing historical datasets from previous retraining snapshots, yielding a class of algorithms that we call Affine Risk Minimizers that converges to a performatively stable point for a broader class of problems. We introduce a new upper bound for methods that use only the final iteration of the dataset and prove for the first time the tightness of both this new bound and the previous existing bounds within the same regime. We also prove that our new algorithm class can surpass the lower bound for standard RRM, thus breaking the prior lower bound, and empirically observe faster convergence to the stable point on various performative prediction benchmarks. We offer at the same time the first lower bound analysis for RRM within the class of Affine Risk Minimizers, quantifying the potential improvements in convergence speed that could be achieved with other variants in our scheme.

## 1 Introduction

Decision-making systems are increasingly integral to critical judgments in sectors such as public policy [Fire and Guestrin, 2019], healthcare [Bevan and Hood, 2006], and education [Nichols and Berliner, 2007]. However, as these systems become more reliant on quantitative indicators, they become vulnerable to the effects described by Goodhart's Law: 'When a measure becomes a target, it ceases to be a good measure' [Goodhart, 1984]. This principle is particularly relevant when predictive models not only forecast outcomes but also influence the behavior of individuals and organizations, leading to performative effects that can subvert the original goals of these systems.

For example, in environmental regulation, companies might manipulate emissions data to meet regulatory targets without truly reducing pollution, thus distorting the intended environmental protection efforts [Fowlie et al., 2012]. In healthcare, hospitals may modify patient care practices to improve performance metrics, potentially prioritizing score improvements over actual patient health outcomes [Bevan and Hood, 2006]. Similarly, in education, the emphasis on standardized test scores can lead

∗ Equal advising. Correspondence to Pedram Khorsandi: &lt;pedram.khorsandi@mila.quebec&gt;

schools to focus narrowly on test preparation, compromising the broader educational experience [Nichols and Berliner, 2007]. These examples demonstrate how decision systems, when overly focused on specific indicators, can be manipulated, resulting in the corruption of the very processes they aim to enhance.

Given these challenges, it is essential to develop predictive models that are not only accurate but also robust against the performative shifts they may provoke. The work by Perdomo et al. [2020] addresses this challenge within the framework of Repeated Risk Minimization (RRM) , where they explore the dynamics of model retraining in the presence of performative feedback loops. In their approach, the authors propose an iterative method that adjusts the predictive model based on the distributional shifts caused by prior model deployments, aiming to stabilize the model performance despite the continuous evolution of the underlying data distribution. By characterizing the convergence properties of their method, they provide a theoretical guarantee for the stability of the model at a performative equilibrium.

Our work extends this framework by leveraging the datasets collected at each snapshot during the retraining process, introducing a new class of algorithms called Affine Risk Minimizers (ARM). By utilizing historical data from previous updates, we show that it is possible to converge to a stable point for a broader class of problems that

Figure 1: An example showing that using older snapshots (purple) speeds up convergence to the stable point (orange star) compared to only the latest snapshot (red). The implementation is provided in the code.

<!-- image -->

were previously unsolvable, extending beyond the bounds established in prior analyses [Mofakhami et al., 2023]. We derive a new upper bound under less restrictive assumptions than Mofakhami et al. [2023] and provide the first tightness analysis for the framework in Perdomo et al. [2020] as well as for our newly established rate. Our method, which incorporates historical datasets, demonstrates superior convergence properties both theoretically and experimentally.

Converging to a stable point is essential in decision-dependent learning systems. Without stability, iterative retraining may lead to persistent fluctuations, preventing reliable long-term predictions. Prior work has examined convergence rates and the range of problem classes in which iterative schemes can achieve stability [Li and Wai, 2024, Narang et al., 2024, Perdomo et al., 2020]. This paper provides the first analytical techniques for examining the tightness of the upper bounds for these methods. In addition, we show that the ARM framework ensures convergence to a stable solution for a broader class of problems, mitigating the limitations of existing approaches.

Contributions. 1 We establish a new upper bound, enhancing the convergence rate of RRM under less restrictive conditions than [Mofakhami et al., 2023]; 2 We establish the tightness of the analysis in both the framework proposed by [Perdomo et al., 2020] and our modification of the framework from Mofakhami et al. [2023]; 3 We introduce a new class of algorithms, named Affine Risk Minimizers (ARM), that provides convergence for a wider class of problems by utilizing linear combinations of datasets from earlier training snapshots; 4 We provide both theoretical and experimental enhancements, showcasing scenarios where ARM improves convergence; 5 Finally, we present the first lower bound techniques for iterative retraining schemes and apply it to both Perdomo et al. [2020] and our modified framework to establish theoretical lower bounds for ARM, detailing the maximum potential improvement in convergence rates achievable through the use of past datasets.

## 2 Related Work

Performative prediction introduces a framework for learning under decision-dependent data [Perdomo et al., 2020], and has been widely studied in various aspects, from stochastic optimization methods to

find stable classifiers [Li and Wai, 2022, Mendler-Dünner et al., 2020] to approaches that focus on performative optimal solutions, the minimizer of performative risk [Miller et al., 2021, Jagadeesan et al., 2022, Lin and Zrnic, 2024]. In this work, we focus our analysis on performative stable solutions, whose deployment removes the need for repeated retraining in changing environments [Perdomo et al., 2020, Mendler-Dünner et al., 2020, Jagadeesan et al., 2021, Brown et al., 2022, Mofakhami et al., 2023].

One of the main applications of this framework is strategic classification [Hardt et al., 2016] which involves deploying a classifier interacting with agents who strategically adapt their features to alter the classifier's predictions and achieve their favorable outcomes. Strategic Classification has been widely used in the literature of performative prediction [Perdomo et al., 2020, Mendler-Dünner et al., 2020, Miller et al., 2021, Hardt et al., 2022, Mofakhami et al., 2023, Narang et al., 2024, Góis et al., 2025], and we adopt this setting in our experiments to empirically demonstrate our theoretical contributions.

Prior work in performative prediction either assumes the data distribution is a function of the parameters modeled as D ( θ ) [Perdomo et al., 2020, Izzo et al., 2021, Drusvyatskiy and Xiao, 2022, Dong et al., 2023], or more realistically dependent on the predictions as in D ( f θ ) [Mofakhami et al., 2023, Mendler-Dünner et al., 2022]. Although existing work only assumes one of these settings, our work adheres to both, by providing a tightness analysis of the rates proposed in Perdomo et al. [2020] and Mofakhami et al. [2023] and showcasing scenarios where we can provide an improved convergence by considering the history of distributions. To the best of our knowledge, we are first to provide a lower bound on the convergence rates achievable using any such affine combination of previous snapshots.

For the assumptions, we adopt those in Mofakhami et al. [2023] and Perdomo et al. [2020]. While several works pursue performatively optimal solutions by assuming convexity over the performative risk itself, recent efforts such as Zheng et al. [2024], Cyffers et al. [2024] relax these requirements by considering non-convex objectives or weaker regularity conditions. In contrast, our analysis imposes no convexity assumption on the performative risk, relying only on convexity of the loss function to establish convergence guarantees.

Most related to our idea of using previous distributions are works that study gradually shifting environments considering history dependence [Brown et al., 2022, Li and Wai, 2022, Rank et al., 2024]. Brown et al. [2022] brought up the notion of stateful performative prediction studying problems where the distribution depends on the classifier and the previous state of the population. This is modeled by a transition function that is fixed but a priori unknown and they show that by imposing a Lipschitz continuity assumption similar to ϵ -sensitivity to the transition map, they can prove the convergence of RRM to an equilibrium distribution-classifier pair. In our work, we consider a specific dependence on history, by using an affine combination of previous distributions, and show that this can lead to an improved convergence than prior work without imposing any additional assumption.

## 3 Performative Stability and RRM

In the context of performative prediction, two primary frameworks are commonly used to address the challenge of shifting distributions due to the model's influence: Repeated Risk Minimization (RRM) and Repeated Gradient Descent (RGD). Throughout this paper, we focus on RRM.

RRM iteratively retrains the model on the distribution it induces until it converges to a performatively stable classifier. Formally, consider a model f θ ∈ F with parameters θ ∈ Θ , and a distribution D ( f θ ) that depends on these parameters. The performative risk is defined as:

<!-- formula-not-decoded -->

where ℓ ( f θ ( x ) , y ) is the loss function for a data point z = ( x, y ) . Aclassifier is performatively stable if it minimizes the performative risk on the distribution it induces:

<!-- formula-not-decoded -->

The RRM framework updates the model parameters by solving:

<!-- formula-not-decoded -->

until convergence, i.e., θ t +1 ≈ θ t .

## 4 Improved Rates and Optimality of Analysis

Both Perdomo et al. [2020] and Mofakhami et al. [2023] derive convergence rates for RRM under distinct assumptions. The assumptions made in these studies reflect the sensitivity of the distribution map D ( . ) to changes in the model and the structural properties of the loss function. Specifically, Perdomo et al. [2020] focuses on Wasserstein-based sensitivity and convexity with respect to the model parameters, while Mofakhami et al. [2023] introduces a framework with Pearson χ 2 -based sensitivity and strong convexity with respect to the predictions. Building on these foundations and motivated by Mofakhami et al. [2023], we now outline the assumptions for our framework, which departs from Mofakhami et al. [2023] only in Assumption 1 below.

Assumption 1 ϵ -sensitivity w.r.t. Pearson χ 2 divergence: The distribution map D ( f θ ) , with pdf p f θ , maintains ϵ -sensitivity with respect to Pearson χ 2 divergence. Formally, for any f θ , f θ ′ ∈ F :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

This assumption, inspired by prior work, is Lipschitz continuity on D ( . ) , implying that if two models with similar prediction functions are deployed, the distributions they induce should also be similar.

Assumption 2 Norm equivalency: The distribution map D ( f θ ) satisfies norm equivalency with parameters C ≥ 1 and c ≤ C . For all f θ , f θ ′ , f θ ∗ ∈ F :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and p ( x ) is the initial distribution, referred to as the base distribution.

The base distribution p ( x ) , following prior formulations in the literature [Perdomo et al., 2020, Mofakhami et al., 2023, Brown et al., 2022], corresponds to the pre-deployment data distribution. This interpretation of p ( x ) as an intervention-free or organic distribution is also consistent with other areas of the literature. Schnabel et al. [2016] demonstrate this in the context of recommender systems, where unbiased test sets like Yahoo! R3 are constructed to reflect user behavior prior to any algorithmic influence.

This assumption holds whenever the distribution map satisfies the bounded density ratio property, i.e., c p f θ ( x ) ≤ p ( x ) ≤ C p f θ ( x ) for all f θ ∈ F . In such cases, one can define small constants c := inf θ,x p f θ ( x ) p ( x ) and C := sup θ,x p f θ ( x ) p ( x ) . We measure these constants in Appendix I for our experimental setup in Section 7.

Assumption 3 Strong convexity w.r.t. predictions: The loss function ˆ y ↦→ ℓ (ˆ y, y ) is γ -strongly convex. For any differentiable function ℓ , and for all y, ˆ y 1 , ˆ y 2 ∈ Y :

<!-- formula-not-decoded -->

Assumption 4 Bounded gradient norm: The loss function ℓ ( f θ ( x ) , y ) has a bounded gradient norm, with an upper bound M = sup x,y,θ ∥∇ ˆ y ℓ ( f θ ( x ) , y ) ∥ .

Building upon Mofakhami et al. [2023], we introduce a new theorem that demonstrates faster linear convergence for RRM, showing that stability can be achieved under less restrictive conditions.

where

Theorem 1 (RRM convergence modified Mofakhami's framework) Suppose the loss ℓ ( f θ ( x ) , y ) is γ -strongly convex with respect to f θ ( x ) (A3) and that the gradient norm with respect to f θ ( x ) is bounded by M = sup x,y,θ ∥∇ ˆ y ℓ ( f θ ( x ) , y ) ∥ (A4). Let the distribution map D ( · ) be ϵ -sensitive with respect to the Pearson χ 2 divergence (A1), satisfy norm equivalency with parameters C ≥ 1 and c ≤ C (A2), and the function space F be convex and compact under the norm ∥ · ∥ .

Then, for G ( θ t ) = arg min θ ∈ Θ E z ∼D ( f θ t ) ℓ ( f θ ( x ) , y ) , with z = ( x, y ) , we have 2 :

<!-- formula-not-decoded -->

By the Schauder fixed-point theorem, a stable classifier f θ PS exists, and if √ ϵM γ &lt; 1 , RRM converges to a unique stable point f θ PS at a linear rate:

<!-- formula-not-decoded -->

This shows that RRM achieves linear convergence to a stable classifier, provided that √ ϵM γ &lt; 1 , ensuring that the mapping is contractive and guarantees convergence. This result improves upon Mofakhami et al. [2023] by eliminating the constant C from the rate, as defined in Assumption 2. Additionally, this approach can achieve improved rates of convergence, as discussed in Theorem 8, where we show how the new definition of ϵ -sensitivity leads to faster convergence.

Despite these improvements, the following theorem establishes for the first time a lower bound under the given assumptions, indicating that the convergence rate cannot be further improved without additional conditions:

Theorem 2 (Tight lower bound modified Mofakhami's framework) Suppose that Assumptions 1-4 hold, with parameters ϵ , M , and γ such that √ ϵM γ ≤ 1 . Under these conditions, there exists a problem instance such that, utilizing RRM, the following holds:

<!-- formula-not-decoded -->

If instead √ ϵM γ &gt; 1 , the bound is Ω(1) , indicating non-convergence.

This result establishes the tightness of the convergence rate under the specific assumptions outlined earlier, demonstrating that the bound cannot be improved without imposing more restrictive assumptions. The full proof of this theorem can be found in Appendix D. A similar tightness analysis for the framework proposed by Perdomo et al. [2020] is provided in the following section, confirming that both frameworks achieve optimal convergence guarantees given their respective conditions.

## 4.1 Tightness Analysis in Perdomo et al. [2020]'s Framework

In their work, Perdomo et al. [2020] make a set of assumptions that differ from Assumption 1-4. Their ϵ -sensitivity assumption is with respect to the Wasserstein distance, and their strong convexity assumption is with respect to the parameters. Formally, they make the following set of assumptions to show the convergence of RRM

Assumption 5 The distribution map θ ↦→ D ( θ ) is ϵ -sensitive w.r.t W 1 :

<!-- formula-not-decoded -->

the loss function θ ↦→ ℓ ( z : θ ) of the performative risk (1) is γ -strongly convex for any z ∈ Z and z ↦→∇ θ ℓ ( z : θ ) is β -Lipschitz for any θ ∈ Θ .

2 Throughout this work, whenever we refer to f G ( θ ) , it denotes f ˆ θ , where ˆ θ ∈ G ( θ ) .

Under these assumptions and for βϵ γ &lt; 1 , Perdomo et al. [2020] showed that RRM does converge to a performatively stable point at a rate: 3

<!-- formula-not-decoded -->

We note that ϵ -sensitivity with respect to the χ 2 divergence (Assumption 1) is generally a stronger condition than ϵ -sensitivity with respect to the Wasserstein distance W 1 , particularly when the input space has small diameter [Mofakhami et al., 2023]. While χ 2 sensitivity implies tighter control over the induced distributional shifts, the two notions are not equivalent, and one does not necessarily imply the other. Hence, each framework is analyzed under its respective assumption.

Theorem 3 (Tight lower bound Perdomo's framework) There exists a problem instance and an initialization θ 0 following Assumption 5 such that employing RRM, we have:

<!-- formula-not-decoded -->

The proof uses a quadratic loss ℓ ( z, θ ) = γ 2 ∥ θ -β γ z ∥ 2 and a performative distribution z ∼ N ( ϵθ, σ 2 ) satisfying ϵ -sensitivity under W 1 . Hence, the RRM update θ t +1 = ϵ β γ θ t matches the contraction factor ( ϵ β γ ) t in Perdomo et al. [2020], confirming the bound's tightness. More detailed proof of this result is provided in Appendix C.

Our theorems show that given the assumptions in either framework, the convergence rate for RRM reaches a fundamental lower bound. This implies that further improvements in convergence speed would require either more restrictive assumptions or a novel optimization framework.

In the next section, we present, for the first time, an approach that breaks the RRM algorithm class by exploiting data from earlier training snapshots, and thereby surpasses the established lower bound, providing improved convergence guarantees.

## 5 Usage of Old Snapshots: Affine Risk Minimizers

Instead of relying solely on the current data distribution induced by D ( f θ t ) , we leverage datasets from previous training snapshots { D ( f θ i ) } t -1 i =0 . The new scheme optimizes model parameters over an aggregated distribution:

<!-- formula-not-decoded -->

where D t is an affine combination of previous distributions, formulated as:

<!-- formula-not-decoded -->

We refer to this class of algorithms as Affine Risk Minimizers . As demonstrated in Appendix A (Lemma 9), the set of stable points for this class of algorithms coincides with those obtained through standard RRM. The following lemma formalizes the convergence of ARM under the stated assumptions, using only the average of the final two training snapshots.

Lemma 1 (2-Snapshots ARM recurrence) Consider the class of problems for which Assumptions 1-4 are satisfied, and let the distribution map D ( . ) be ϵ C -sensitive with respect to the base distribution within the convex function space F . Formally, for any f θ , f θ ′ ∈ F ,

<!-- formula-not-decoded -->

3 Note that if βϵ γ ≥ 1 the convergence rate is vacuous. In that case, a performatively stable point may not even exist.

where ∥ f θ -f θ ′ ∥ 2 is defined in Equation 8. The distribution at iteration t is given by

<!-- formula-not-decoded -->

Under these conditions, the following convergence property holds for the iterative sequence generated by Equation 12: √ √

<!-- formula-not-decoded -->

where m t = max {∥ f θ t -f θ t -1 ∥ , ∥ f θ t -1 -f θ t -2 ∥} .

The problem class defined here aligns with that in Theorem 2 for the case C ≈ 1 . Now, the following theorem provides theoretical evidence of improved convergence, which will be further supported by experiments in Section 7.

Theorem 4 (2-Snapshots ARM convergence) If ( √ 3 2 √ ϵM γ ) &lt; 1 , the sequence described in Lemma 1 forms a Cauchy sequence, converging to a stable point.

Relaxing the earlier condition √ ϵM γ ≤ 1 to the threshold 2 √ 3 ≈ 1 . 155 , this theorem demonstrates a modest but tangible improvement allowing ARM to breach the lower bound of Theorem 2 and shows that when √ ϵM γ &lt; 2 √ 3 ≈ 1 . 155 with C ≈ 1 , convergence to a stable point remains possible under the same conditions, in contrast to standard RRM, which does not converge. A detailed proof of this theorem, along with Lemma 1, is provided in Appendix E, where we show that the algorithm generates a Cauchy sequence and converges to the stable point. We prove the convergence for schemes that use the average of the last n snapshots, for any n , but the best result is obtained for n = 2 so far.

In the following section, we explore the lower bound for the convergence rates achievable using any affine combination of previous snapshots.

## 6 Lower Bounds for Affine Risk Minimizers

We established the potential for convergence across a wider class of problems using ARMs. This prompts the question of how much the convergence class can be improved, which we address in this section.

We propose the first distinct lower bounds for the framework described in Section 4 and that of Perdomo et al. [2020] for the class of Affine Risk Minimizers. The lower bound for our framework is presented in this section, while the corresponding result for Perdomo et al. [2020] is detailed in the following.

Theorem 5 (ARM lower bound modified Mofakhami's framework) Suppose that Assumptions 1-4 hold. Then, there exists a problem instance in this regime, and for any algorithm in the Affine Risk Minimizers class, such that:

<!-- formula-not-decoded -->

This demonstrates that the convergence rate for the class of problems satisfying Assumptions 1-4 cannot exceed the given lower bound.

## 6.1 Lower Bound with Perdomo et al. [2020]'s Assumption

We show that the convergence rate for RRM provided in Equation 10 is optimal among the class of Affine Risk Minimizers up to a factor 2 .

Theorem 6 (ARM lower bound Perdomo's framework) There exists a problem instance and an initialization θ 0 following Assumption 5 such that for any algorithm in the Affine Risk Minimizers class, we have:

<!-- formula-not-decoded -->

Proof Sketch. The proofs of Theorems 5 and 6 are inspired by the idea of introducing a new dimension at each iteration by Nesterov's lower bound for convex smooth functions [Nesterov, 2014]. We construct a problem instance satisfying Assumption 5 and derive the iteration dynamics of RRM to establish a lower bound on its convergence rate.

We introduce a structured transformation of the parameter space, different from the one introduced in Nesterov [2014], using the matrix

<!-- formula-not-decoded -->

This matrix ensures that if a vector v is in the span of { e 1 , ..., e i } , where e i is the standard basis vector in R d , then Av extends the span to include e i +1 . This property allows us to control the iterative exploration of dimensions in the distribution mapping.

We define the loss function as ℓ ( f θ ( x ) , y ) = γ 2 ∥ θ -β γ z ∥ 2 , and set the distribution map to D ( θ ) = N ( ϵ 2 Aθ + e 1 , σ 2 ) . This choice ensures that each RRM iteration follows the update rule

<!-- formula-not-decoded -->

From Lemma 9 (Appendix A), we know that any Affine Risk Minimizer converges to the same set of stable points as RRM. Since the problem instance we constructed has a unique stable point, it remains to explicitly compute this point and derive a lower bound by summing the contributions from undiscovered dimensions, completing the proof. A detailed proof is provided in Appendix F. □

To further illustrate this, Figure 2 provides empirical evidence supporting the theoretical lower bound derived for Perdomo et al. [2020]. The figure shows the convergence of ∥ θ -θ PS ∥ over multiple iterations for various combinations of previous snapshots. As indicated by the dotted line, the lower bound is never violated, demonstrating that the theoretical result holds in practice. The experimental setup for these results is also detailed in Appendix F. We also extend our result to a more general case for ARM in the following theorem (with proof in Appendix F.1).

Theorem 7 (Proximal ARM lower bound) Let Assumption 5 hold and generate the iterates via

<!-- formula-not-decoded -->

where D t is any mixture distribution defined in Equation 13. Then there exists a problem instance and an initialization θ 0 such that every algorithm in the Affine Risk Minimizers class satisfies

<!-- formula-not-decoded -->

The proximal ARM update in Equation 18 is the ARM version of the proximal formulation of RRM introduced by Drusvyatskiy and Xiao [2022]. In fact, the lower bound of Theorem 7 continues to hold for the standard RRM method, since RRM corresponds to the special case of ARM in which the mixture distribution D t places all its weight on the most recent iterate. Thus, this result demonstrates that the proof technique we have developed for establishing exponential lower bounds is not limited to the ARM family but extends naturally to other iterative decision-dependent optimization procedures. In the supplementary material, we also present convergence rates for Proximal RRM for the first time on the framework of Perdomo et al. [2020], filling a gap in the literature, though these rates offer no improvement over standard RRM.

Figure 2: Convergence of ∥ θ t -θ PS ∥ over iterations t for different values of τ , which defines the aggregation of datasets from training snapshots: D t = ∑ t i = t -τ +1 1 τ D ( θ i ) . The dotted line shows our lower bound from Perdomo et al. [2020], with ϵ = 2 . 49 , β = 1 , and γ = 5 . 0 . The experiment, consistent across all methods, validates the bound by showing that ∥ θ t -θ PS ∥ does not drop below it, supporting our theory.

<!-- image -->

## 7 Experiments

We conduct experiments in two semi-synthetic environments to evaluate whether aggregating past snapshots improves convergence to the performatively stable point. We present an empirical comparison of different averaging windows for prior snapshots. At each time step t , we form D t by aggregating the datasets from the training snapshots as

<!-- formula-not-decoded -->

where we compare methods using various values of τ , including τ = 1 , 2 , 4 , t 2 , and 'all' (which includes all snapshots up to time t ).

We first discuss our evaluation metric, then present detailed case studies on the credit scoring environment Mofakhami et al. [2023] (Section 7.1) and the rideshare markets Narang et al. [2024] (Appendix J).

Evaluation Metric. Throughout our experiments, we focus on changes in loss as a result of performativity. We define ∆ R t , i.e. the loss shift due to performativity at time t , as the absolute difference in loss observed by a model before and after the data distribution has changed due to performative effects while keeping the model's state constant.

<!-- formula-not-decoded -->

This metric allows for clearer comparisons between methods by minimizing overlap in the plots, unlike the performative risk (Equation 1).

## 7.1 Credit Scoring

Setup. Inspired by Mofakhami et al. [2023], we use the Resample-if-Rejected (RIR) procedure to model distribution shifts in a controlled experimental setting. This methodology involves users strategically altering their data to influence the classification outcome.

Let us consider a base distribution with probability density function p and a function g : f θ ( x ) ↦→ g ( f θ ( x )) indicating the probability of rejection based on the prediction f θ ( x ) ∈ R . The modified distribution p f θ , under the RIR mechanism, evolves as follows:

- Sample x from p .
- With probability 1 -g ( f θ ( x )) , accept and output x . Otherwise, resample from p .

Figure 3: Loss shift due to performativity for the credit-scoring environment. To accurately measure Performative Risk, we average over 500 runs per method. Increasing the aggregation window τ ( 1 → 2 → 4 → t/ 2 → all) reduces the loss shifts and, consequently, reaches the stable point faster.

<!-- image -->

Our data comes from Kaggle's Give Me Some Credit dataset 4 , which includes features x ∈ R 11 and labels y ∈ { 0 , 1 } , where y = 1 indicates a defaulting applicant. We partition the features into two sets: strategic and non-strategic. We assume independence between strategic and non-strategic features. While non-strategic features remain fixed, the strategic features are resampled using the RIR procedure with a rejection probability g ( f θ ( x )) = f θ ( x ) + δ . We use a scaled sigmoid function after the second layer. This scales f θ ( x ) to the interval [0 , 1 -δ ] , ensuring that g ( f θ ( x )) ∈ [ δ, 1] remains a valid probability. Further implementation details are available in Appendix I.

Theorem 8 Let f θ ( x ) ∈ [0 , 1 -δ ] for all θ ∈ Θ , where 0 &lt; δ &lt; 1 is fixed. Then, for g ( f θ ( x )) = f θ ( x ) + δ , RIR is ϵ -sensitive as defined in Assumption 1 with ϵ = O ( δ -3 2 ) .

This result provides an example where our rate surpasses the rate previously derived in Mofakhami et al. [2023] ( O ( δ -2 ) within the same framework). In addition, Mofakhami et al. [2023] derived the remaining constants in the rate for this setup, and these constants remain unchanged in our result. Furthermore, for any value of M and γ , our rate can guarantee convergence for a wider class of problems. The proof of this theorem, along with justifications for the improved rate, is presented in Appendix H. Mofakhami et al. [2023] derive the

Results. The outcomes of this case study are shown in Figure 3. For larger window sizes ( τ ), we omit the initial iterations in the figure because they follow the same update rule as smaller τ methods, leading to identical values. Figure 3 demonstrates the advantage of using older snapshots in the optimization process. As the window size increases from 1 to 2 , we observe a near-half reduction in the loss shift, particularly in the early iterations, with the improvement persisting even after 50 iterations. While larger windows continue to reduce the loss shift, the marginal gains decrease as window size increases. This is evident from the similarity between the curves for window sizes t/ 2 , and 'all'. The decreasing marginal gains elicit a trade-off against the time, memory, and resource consumption. As the window size increases, both time per iteration and the memory consumption increase linearly. Thus, the user has to pick the right aggregation window τ based on the application and the resources available to achieve the desired convergence speed while respecting the logistical constraints. The corresponding performative risk plot can also be found in Appendix I.

## 8 Conclusion

In this paper, we introduced a new class of algorithms for improved convergence in performative prediction by utilizing historical datasets from previous retraining snapshots. Our theoretical contributions include establishing a new upper bound for last-iterate methods, demonstrating the tightness of this bound, and surpassing existing lower bounds through the aggregation of historical datasets. We have also presented the first lower bound analysis for Repeated Risk Minimization (RRM) within the class of Affine Risk Minimizers. Our empirical results validate the theoretical findings, showing that using prior snapshots leads to more effective convergence to a stable point. These contributions provide new insights into performative prediction and offer an alternative approach to enhancing learning in dynamic environments.

## Acknowledgments

This research was supported by the Canada CIFAR AI Chair Program and the NSERC Discovery Grants RGPIN-2023-04373 and RGPIN-2025-05123. We gratefully acknowledge the computational resources provided by Calcul Quebec ( calculquebec.ca ) and the Digital Research Alliance of Canada ( alliancecan.ca ), which enabled part of the experiments. Simon Lacoste-Julien is a CIFAR Associate Fellow in the Learning in Machines &amp; Brains program. We also thank António Góis and Alan Milligan for their valuable feedback, which notably contributed to this work.

4 Give me Some Credit Dataset, 2011: https://www.kaggle.com/c/GiveMeSomeCredit

## References

- Gwyn Bevan and Christopher Hood. What's measured is what matters: Targets and gaming in the english public health care system. Public Administration , 2006. https://doi.org/10.1111/j. 1467-9299.2006.00600.x .
- Gavin Brown, Shlomi Hod, and Iden Kalemaj. Performative prediction in a stateful world. In AISTATS , 2022. https://proceedings.mlr.press/v151/brown22a.html .
- Edwige Cyffers, Muni Sreenivas Pydi, Jamal Atif, and Olivier Cappé. Optimal classification under performative distribution shift. In NeurIPS , 2024. https://openreview.net/forum?id= 3J5hvO5UaW .
- Roy Dong, Heling Zhang, and Lillian Ratliff. Approximate regions of attraction in learning with decision-dependent distributions. In AISTATS , 2023. https://proceedings.mlr.press/ v206/dong23b.html .
- D.C. Dowson and B.V. Landau. The fréchet distance between multivariate normal distributions. Journal of Multivariate Analysis , 12(3):450-455, 1982. https://www.sciencedirect.com/ science/article/pii/0047259X8290077X .
- Dmitriy Drusvyatskiy and Lin Xiao. Stochastic optimization with decision-dependent distributions. Mathematics of Operations Research , 2022. https://doi.org/10.1287/moor.2022.1287 .
- Michael Fire and Carlos Guestrin. Over-optimization of academic publishing metrics: observing Goodhart's Law in action. GigaScience , 8(6):giz053, 2019. https://doi.org/10.1093/ gigascience/giz053 .
- Meredith Fowlie, Stephen P. Holland, and Erin T. Mansur. What do emissions markets deliver and to whom? evidence from southern california's nox trading program. American Economic Review , 2012. https://www.aeaweb.org/articles?id=10.1257/aer.102.2.965 .
- António Góis, Mehrnaz Mofakhami, Fernando P. Santos, Gauthier Gidel, and Simon LacosteJulien. Performative prediction on games and mechanism design. In AISTATS , 2025. https: //proceedings.mlr.press/v258/gois25a.html .
- Ziv Goldfeld, Rami Pellumbi, and Kia Khezeli. Lecture 6: f-divergences. In ECE 5630: Information Theory for Data Transmission, Security and Machine Learning , 2020. https://people.ece. cornell.edu/zivg/ECE\_5630\_Lectures6.pdf .
- C. A. E. Goodhart. Problems of Monetary Management: The UK Experience , pages 91-121. Macmillan Education UK, 1984. https://doi.org/10.1007/978-1-349-17295-5\_4 .
- Moritz Hardt, Nimrod Megiddo, Christos Papadimitriou, and Mary Wootters. Strategic classification. In ITCS . Association for Computing Machinery, 2016. https://doi.org/10.1145/2840728. 2840730 .
- Moritz Hardt, Meena Jagadeesan, and Celestine Mendler-Dünner. Performative power. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, NeuIPS . Curran Associates, Inc., 2022. https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 90e73f3cf1a6c84c723a2e8b7fb2b2c1-Paper-Conference.pdf .
- Zachary Izzo, Lexing Ying, and James Zou. How to learn when data reacts to your model: Performative gradient descent. In ICML , 2021. https://proceedings.mlr.press/v139/izzo21a. html .
- Meena Jagadeesan, Celestine Mendler-Dünner, and Moritz Hardt. Alternative microfoundations for strategic classification. In ICML , 2021. https://proceedings.mlr.press/v139/ jagadeesan21a.html .
- Meena Jagadeesan, Tijana Zrnic, and Celestine Mendler-Dünner. Regret minimization with performative feedback. In ICML , 2022. https://proceedings.mlr.press/v162/jagadeesan22a. html .

- Qiang Li and Hoi-To Wai. State dependent performative prediction with stochastic approximation. In AISTATS , 2022. https://proceedings.mlr.press/v151/li22c.html .
- Qiang Li and Hoi To Wai. Stochastic optimization schemes for performative prediction with nonconvex loss. In NeurIPS , 2024. https://neurips.cc/virtual/2024/poster/94252 .
- Licong Lin and Tijana Zrnic. Plug-in performative optimization. In ICML , 2024. https:// proceedings.mlr.press/v235/lin24ab.html .
- Celestine Mendler-Dünner, Juan Perdomo, Tijana Zrnic, and Moritz Hardt. Stochastic optimization for performative prediction. In NeurIPS , 2020. https://proceedings.neurips.cc/paper\_ files/paper/2020/file/33e75ff09dd601bbe69f351039152189-Paper.pdf .
- Celestine Mendler-Dünner, Frances Ding, and Yixin Wang. Anticipating performativity by predicting from predictions. In NeurIPS , 2022. https://dl.acm.org/doi/10.5555/3600270.3602530 .
- John P Miller, Juan C Perdomo, and Tijana Zrnic. Outside the echo chamber: Optimizing the performative risk. In ICML , 2021. https://proceedings.mlr.press/v139/miller21a. html .
- Mehrnaz Mofakhami, Ioannis Mitliagkas, and Gauthier Gidel. Performative prediction with neural networks. In arxiv , 2023. https://arxiv.org/abs/2304.06879 .
- Adhyyan Narang, Evan Faulkner, Dmitriy Drusvyatskiy, Maryam Fazel, and Lillian J. Ratliff. Multiplayer performative prediction: learning in decision-dependent games. JMLR , 2024. https: //www.jmlr.org/papers/volume24/22-0131/22-0131.pdf .
- Yurii Nesterov. Introductory Lectures on Convex Optimization: A Basic Course . Springer, 1 edition, 2014. https://dl.acm.org/doi/10.5555/2670022 .
- Sharon L. Nichols and David C. Berliner. Collateral damage: How high-stakes testing corrupts America's schools . Harvard Education Press, 2007. https://psycnet.apa.org/record/ 2007-03254-000 .
- Frank Nielsen and Kazuki Okamura. On the f-divergences between densities of a multivariate location or scale family. Statistics and Computing , 2024. https://doi.org/10.1007/ s11222-023-10373-6 .
- Juan Perdomo, Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt. Performative prediction. In ICML , 2020. https://proceedings.mlr.press/v119/perdomo20a.html .
- Ben Rank, Stelios Triantafyllou, Debmalya Mandal, and Goran Radanovic. Performative reinforcement learning in gradually shifting environments, 2024. https://arxiv.org/abs/2402. 09838 .
- Juliusz Schauder. Der fixpunktsatz in funktionalraümen. Studia Mathematica , 2(1):171-180, 1930. https://doi.org/10.4064/sm-2-1-171-180 .
- Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. Recommendations as treatments: Debiasing learning and evaluation. In ICML , 2016. https: //proceedings.mlr.press/v48/schnabel16.html .
- Xue Zheng, Tian Xie, Xuwei Tan, Aylin Yener, Xueru Zhang, Ali Payani, and Myungjin Lee. Profl: Performative robust optimal federated learning, 2024. https://arxiv.org/abs/2410.18075 .

## A Auxiliary Lemmas and Technical Results

Lemma 2 (Expectation of a Gaussian-Weighted Exponential Function) Let x ∼ N ( µ , σ 2 I ) . Then the expected value of x exp ( -1 2 e ∥ x ∥ 2 ) is given by:

<!-- formula-not-decoded -->

Proof: The expected value is expressed as:

<!-- formula-not-decoded -->

Merging the exponentials:

<!-- formula-not-decoded -->

Completing the square yields:

<!-- formula-not-decoded -->

Since the integral is over a Gaussian distribution with mean µ /σ 2 1 e + 1 σ 2 , after multiplying by the constant term, we obtain:

<!-- formula-not-decoded -->

Lemma 3 (Young's Product Inequality) Let a, b ≥ 0 and let p, q &gt; 1 be conjugate exponents, i.e. 1 p + 1 q = 1 . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 4 (Bound on Chi-Square Divergence for Convex Combinations) Let P and Q and R be probability distributions on R n . For any α ∈ [0 , 1] , the following inequality holds:

<!-- formula-not-decoded -->

Proof: We begin by expanding the chi-square divergence using its definition, followed by applying Young's inequality.

<!-- formula-not-decoded -->

which, as a result, one can derive,

Lemma 5 (Inverse of an antisymmetric of a Jordan Normal Form Matrix) Let A ∈ R d × d be defined as:

<!-- formula-not-decoded -->

and let bI -cA be an invertible matrix where c b ≤ 1 2 and A is as defined above. Then the inverse of ( bI -cA ) applied to e 1 , the first standard basis vector, has the following form for large d :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for d ≥ 2 T when T is large, and t ≤ T .

Proof: The matrix A has the following form:

<!-- formula-not-decoded -->

Moreover, the sum below is:

Thus, ( bI -cA ) takes the form:

<!-- formula-not-decoded -->

We continue by computing the inverse of the lower triangular matrix with diagonal entries λ 1 , λ 2 , . . . , λ d and subdiagonal entries of -1 as shown below:

<!-- formula-not-decoded -->

Using the formula above (diagonal entries λ 1 = b c -1 , λ 2 = b c -1 , . . . , λ d = b c -1 and subdiagonal entries of -1 ) the inverse of bI -cA will have the form:

<!-- formula-not-decoded -->

Now, applying this inverse to the vector e 1 L , where e 1 = [1 0 . . . 0] T , we get the following:

<!-- formula-not-decoded -->

Sum of the entries from index t to d is:

<!-- formula-not-decoded -->

This is a geometric series. The closed form of the sum is:

<!-- formula-not-decoded -->

For large d ≥ 2 t and b c -1 ≥ 1 , this sum can be approximated by the leading term:

<!-- formula-not-decoded -->

Thus, applying the inequality 1 1 x -1 ≤ x for all x &lt; 1 , we obtain the following lower bound for the sum:

<!-- formula-not-decoded -->

Lemma 6 Let N ( µ 1 , Σ 1 ) and N ( µ 2 , Σ 2 ) be two multivariate normal distributions with means µ 1 , µ 2 ∈ R d and covariance matrices Σ 1 , Σ 2 ∈ R d × d . The squared 1-Wasserstein distance between these distributions is bounded by:

<!-- formula-not-decoded -->

This expression bounds the Wasserstein distance between two multivariate normal distributions, as shown in Dowson and Landau [1982].

Lemma 7 Let N ( µ 1 , Σ) and N ( µ 2 , Σ) be two multivariate normal distributions with means µ 1 , µ 2 ∈ R d and a shared covariance matrix Σ ∈ R d × d . The χ 2 -divergence between these distributions is bounded by:

<!-- formula-not-decoded -->

This provides the χ 2 -divergence between two multivariate normal distributions, as shown in Nielsen and Okamura [2024].

Lemma 8 Any projection proj ( . ) from R d into any convex set C ∈ R d is a continuous function.

Proof: To prove that the projection is continuous, we need to show that if x n → x in R d , then proj C ( x n ) → proj C ( x ) .

Let y n = proj C ( x n ) and y = proj C ( x ) . Since y n ∈ C and y n minimizes the distance to x n , we have:

<!-- formula-not-decoded -->

As x n → x , the right-hand side ∥ x n -y ∥ → ∥ x -y ∥ , and thus ∥ x n -y n ∥ is bounded. Since the sequence { y n } is bounded and lies in the compact set C , it has a convergent subsequence y n k → ¯ y ∈ C . By the continuity of the distance function, we have:

<!-- formula-not-decoded -->

As y = proj C ( x ) minimizes the distance from x to C , it follows that ¯ y = y , and thus y n → y . Therefore, proj C ( x n ) → proj C ( x ) , proving continuity.

Lemma 9 The set of stable points for any method in the class of Affine Risk Minimizers is equivalent to the set of stable points for standard RRM.

Proof: Consider the mapping for an affine risk minimizer using the last τ iterates, defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At a stable point, the mapping satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now show that every stable point for this mapping is also a stable point for the standard RRM mapping, defined as:

<!-- formula-not-decoded -->

where which implies that:

From the definition of D t , we have:

<!-- formula-not-decoded -->

since ∑ t -1 i = t -τ α ( t ) i = 1 . Therefore:

<!-- formula-not-decoded -->

implying that any stable point for G τ is also a stable point for G .

Conversely, if θ t = θ t -1 at a stable point of G , then iterating the mapping G τ τ times yields the sequence:

<!-- formula-not-decoded -->

A similar argument shows that this stable point satisfies:

<!-- formula-not-decoded -->

because:

<!-- formula-not-decoded -->

Which leads to,

<!-- formula-not-decoded -->

showing that this stable point is also stable for G τ .

Thus, the set of stable points is equivalent for both mappings.

Lemma 10 Let a be a real number with 0 &lt; a &lt; 1 . Then for every integer t ≥ 0 ,

<!-- formula-not-decoded -->

Lemma 11 Let A 1 and A 2 be two probability distributions and let B 1 and B 2 be another two probability distributions. Then, we have

<!-- formula-not-decoded -->

Proof: To compute the χ 2 divergence between the averages A 1 + A 2 2 and B 1 + B 2 2 , we start with the definition:

<!-- formula-not-decoded -->

Simplifying the numerator, we get:

<!-- formula-not-decoded -->

Applying the inequality ( a + b ) 2 ≤ 2 a 2 +2 b 2 , we can further bound this as follows:

<!-- formula-not-decoded -->

By distributing the terms, this becomes:

<!-- formula-not-decoded -->

Now, since 1 p B 1 ( x )+ p B 2 ( x ) ≤ 1 p B 1 ( x ) and 1 p B 1 ( x )+ p B 2 ( x ) ≤ 1 p B 2 ( x ) , we can split the integral as follows:

<!-- formula-not-decoded -->

By definition of the χ 2 divergence, this final expression is equivalent to:

<!-- formula-not-decoded -->

Thus, we have shown that

<!-- formula-not-decoded -->

which completes the proof.

Lemma 12 Let η = f G ( θ ′ ) -f G ( θ ) . Suppose the function space F is convex, and

<!-- formula-not-decoded -->

where z = ( x, y ) ∼ p f θ represents the distribution induced by the model f θ , and ℓ is a continuously differentiable loss function. Then the following inequality holds:

<!-- formula-not-decoded -->

Refer to Mofakhami et al. [2023] for the proof.

Lemma 13 Suppose a nonnegative sequence ( a t ) t ≥ 0 satisfies the recurrence

<!-- formula-not-decoded -->

for all t ≥ 1 and some 0 &lt; ϵ ≤ 1 . Then for every integer t ≥ 0 , one has

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: Set A := max { a 0 , a 1 } . We proceed by unrolling the recursion in pairs:

<!-- formula-not-decoded -->

and in general, each two steps introduce at least one additional factor of ϵ , yielding the claimed bound.

## B Proof of Theorem 1

The proof of Theorem 1 largely follows the approach in Mofakhami et al. [2023]. To facilitate readability, we have restated the common parts from the proof in Mofakhami et al. [2023].

Fix θ and θ ′ in Θ . Let h : F → R and h ′ : F → R be two functionals defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where each data point z is a pair of features x and label y .

For a fixed z = ( x, y ) , due to strong convexity of ℓ ( f θ ( x ) , y ) in f θ ( x ) we have:

<!-- formula-not-decoded -->

Taking an integral over z w.r.t p f θ ( z ) , and knowing that ∥ f G ( θ ) -f G ( θ ′ ) ∥ 2 f θ = ∫ ∥ f G ( θ ) ( x ) -f G ( θ ′ ) ( x ) ∥ 2 p f θ ( z ) dz , we get the following:

<!-- formula-not-decoded -->

Similarly:

<!-- formula-not-decoded -->

Since f G ( θ ) minimizes h , the following result can be achieved through the convexity of the function space, (Lemma 12):

<!-- formula-not-decoded -->

Adding (24) and (25) and using the above inequality, we conclude:

<!-- formula-not-decoded -->

This is a key inequality that will be used later in the proof.

Now recall that there exists M such that M = sup x,y,θ ∥∇ ˆ y ℓ ( f θ ( x ) , y ) ∥ and the distribution map over data is ϵ -sensitive w.r.t Pearson χ 2 divergence, i.e.

<!-- formula-not-decoded -->

With this in mind, we do the following calculations:

<!-- formula-not-decoded -->

( ∗ ) comes from the fact that ∣ ∣ ∫ f ( x ) dx ∣ ∣ ≤ ∫ | f ( x ) | dx , and the Cauchy-Schwarz inequality states that | E [ XY ] | ≤ √ E [ X 2 ] E [ Y 2 ] .

We conclude from the above derivations that:

<!-- formula-not-decoded -->

Similar to inequality (26), since f G ( θ ′ ) minimizes h ′ , one can prove:

<!-- formula-not-decoded -->

From (27) we know that ∫ ( f G ( θ ) ( x ) -f G ( θ ′ ) ( x ) ) ⊤ ∇ ˆ y ℓ ( f G ( θ ′ ) ( x ) , y ) p f θ ( z ) dz is negative, so with this fact alongside (29) and (30), we can write:

<!-- formula-not-decoded -->

Combining (27) and (31), we obtain:

<!-- formula-not-decoded -->

To prove the existence of a fixed point, we use the Schauder fixed point theorem [Schauder, 1930]. Define

<!-- formula-not-decoded -->

For this function, U ( f θ ) = f G ( θ ) . So instead of Equation 32, we can write:

<!-- formula-not-decoded -->

Using Assumption 2, we derive the following bound,

<!-- formula-not-decoded -->

This inequality shows that for any f θ 0 ∈ F , if lim n →∞ ∥ f θ n -f θ 0 ∥ = 0 , then lim n →∞ ∥U ( f θ n ) -U ( f θ 0 ) ∥ = 0 , which proves the continuity of U with respect to the norm ∥ . ∥ . Thus, since U is a continuous function from the convex and compact set F to itself, the Schauder fixed point theorem ensures that U has a fixed point. Therefore, f θ PS exists such that f G ( θ PS ) = f θ PS .

If we set θ = θ PS and θ ′ = θ t -1 for θ PS being any sample in the set of stable classifiers, we know that G ( θ ) = θ PS and G ( θ ′ ) = θ t . So we will have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

Note that Equation 36 applies to any stable point. Suppose there are two distinct stable points, f θ 1 PS and f θ 2 PS . By the definition of stable points and using Equation 33, we have:

<!-- formula-not-decoded -->

√

Under the assumption that ϵM γ &lt; 1 , the inequality above ensures that f θ 1 PS = f θ 2 PS 5 and the stable point must be unique. Thus, Equation 36 confirms that RRM converges to a unique stable classifier at a linear rate when √ ϵM γ &lt; 1 .

5 It is important to clarify that f θ 1 PS = f θ 2 PS does not imply ∀ x f θ 1 PS ( x ) = f θ 2 PS ( x ) . Instead, it indicates that ∥ f θ 1 PS -f θ 2 PS ∥ = 0 .

## C Proof of Theorem 3

In this section, we examine the tightness of the analysis presented in Perdomo et al. [2020] by considering a specific loss function and designing a particular performativity framework. We focus on the loss function ℓ ( z, θ ) = γ 2 ∥ θ -β γ z ∥ 2 , which is γ -strongly convex with respect to the parameter θ and its gradient w.r.t. θ is β -Lipschitz, aligning with the assumptions stipulated in Perdomo et al. [2020].

We model performativity through the following distribution: z ∼ N ( ϵθ, σ 2 ) . According to Lemma 6 the 1 -Wasserstein distance between two normal distributions is upper bounded by:

<!-- formula-not-decoded -->

it follows that the distribution mapping specified is ϵ -sensitive, as described in Perdomo et al. [2020]. Under these conditions, the RRM process results in the following update mechanism:

<!-- formula-not-decoded -->

This arises because:

<!-- formula-not-decoded -->

This progression directly corresponds to the upper bound suggested by Perdomo et al. [2020], confirming that the analysis is tight. No further refinement of the analytical model would mean a faster convergence rate for the given set of assumptions as detailed in Perdomo et al. [2020].

## D Proof of Theorem 2

We define the model fitting function as f θ ( x ) = θ , and the corresponding loss function is:

<!-- formula-not-decoded -->

where proj ∥ . ∥ =0 . 95 denotes the projection onto the surface of a ball with radius 0 . 95 . By setting θ ∈ Θ = { z | ∥ z ∥ ≤ 0 . 05 min { M γ , 1 √ ϵ }} , we ensure that the gradient norm remains smaller than M . Since the loss function is γ -strongly convex, it satisfies both Assumptions 4 and 3.

Throughout this proof ∥ θ 1 -θ 2 ∥ = ∥ f θ 1 -f θ 2 ∥ f θ ′ for any choice of θ ′ .

We define the distribution mapping as follows:

<!-- formula-not-decoded -->

The χ 2 -divergence between two distributions D ( θ 1 ) = N ( µ 1 , σ ) and D ( θ 2 ) = N ( µ 2 , σ ) , where µ 1 = √ ϵθ 1 and µ 2 = √ ϵθ 2 , with σ = 1 2 , is given by (Lemma 7):

<!-- formula-not-decoded -->

Thus, the χ 2 -divergence between the distributions is bounded by ϵ ∥ θ 1 -θ 2 ∥ 2 , making it ϵ -sensitive according to Assumption 1. Note that

With this set up one would derive the update rule:

<!-- formula-not-decoded -->

Using,

<!-- formula-not-decoded -->

and given that E [ x ] = √ ϵθ ≤ 0 . 05 min { √ ϵM γ , 1 } by the definition of Θ , the condition holds.

<!-- formula-not-decoded -->

Assuming we start with θ 0 in the feasible set and operate in the regime where M √ ϵ γ ≤ 1 , the projection into the feasible set can be omitted. Therefore, we have:

<!-- formula-not-decoded -->

It is clear that θ = 0 is the stable point in this setup, so:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In other words:

For the case where M √ ϵ γ &gt; 1 , the projection remains constrained to the surface of the ball Θ , preventing convergence to the stable point.

## E Proof of Lemma 1 and Theorem 4

This proof is heavily inspired by the proof of Theorem 1 in Appendix B. We start by restating the stronger assumption that was added in Lemma 1 and that implies other standard assumptions for this paper.

Assumption 6 ϵ -sensitivity with respect to Pearson χ 2 divergence (version 2): The distribution map D ( f θ ) maintains ϵ -sensitivity with respect to Pearson χ 2 divergence. For all f θ , f θ ′ ∈ F :

<!-- formula-not-decoded -->

where ∥ f θ -f θ ′ ∥ 2 is defined in Equation 8.

Note that, Assumptions 2 and 6, imply Assumption 1:

<!-- formula-not-decoded -->

Following the methodology described for Theorem 2 in Mofakhami et al. [2023], we begin by defining the functional evaluations at consecutive time steps as follows:

<!-- formula-not-decoded -->

where p t ( z ) denotes the probability density function of sample z from the distribution D t .

Utilizing the convexity of ℓ and Lemma 1 from Mofakhami et al. [2023], following the line of argument in Equation 17 of Mofakhami et al. [2023], we establish the following inequality:

<!-- formula-not-decoded -->

where ∥ f θ t +1 -f θ t ∥ 2 p t represents the squared norm, calculated as:

<!-- formula-not-decoded -->

and p t ( x ) = 1 n ∑ n -1 i =0 p f θ t -i ( x ) , Using the bounded gradient assumption, we deduce:

<!-- formula-not-decoded -->

Now combining equations 38 and 39 we get,

<!-- formula-not-decoded -->

Note that Equation 40, is a direct consequence of Assumptions 3-4, 6, and doesn't rely on definition of D t (refer to Equation 32 for the proof). In other words, if we define our method as the mapping

<!-- formula-not-decoded -->

where D ( f θ 1 . . . f θ n ) = ∑ n i =1 D ( f θ i ) n , then,

<!-- formula-not-decoded -->

where, p d is probability density function of distribution D ( f θ 1 . . . f θ n ) . We use this information further on in the proof.

The remaining task is to bound the χ 2 divergence. Start by defining:

<!-- formula-not-decoded -->

Which implies:

<!-- formula-not-decoded -->

Now, using this one can derive:

<!-- formula-not-decoded -->

(Proposition 6.1 of Goldfeld et al. [2020], convexity of f -divergence w.r.t. its arguments)

<!-- formula-not-decoded -->

where m 2 t = max 0 ≤ i&lt;n {∥ f θ t -i -1 -f θ t -i ∥ 2 } . Which further gives us,

<!-- formula-not-decoded -->

And in conclusion, we derive the following bound:

<!-- formula-not-decoded -->

Using Assumption 2, we further obtain:

<!-- formula-not-decoded -->

Substituting this back into the previous inequality, we finally get:

<!-- formula-not-decoded -->

Note the n = 2 minimizes the bracket above amongst the integers. 6 Continuing with n = 2 , we have:

<!-- formula-not-decoded -->

Convergence to a Stable Point. By expanding the max term in Equation 44, using Lemma 13, we establish the following bound:

<!-- formula-not-decoded -->

with f 0 = max {∥ f θ 2 -f θ 1 ∥ , ∥ f θ 1 -f θ 0 ∥} . Combining this inequality with Lemma 10 and assuming √ ϵM γ ≤ 4 √ 3 2 , we obtain:

<!-- formula-not-decoded -->

Where c = ( √ 3 2 √ ϵM γ ) -1 2 . For clarity, let α = ( √ 3 2 √ ϵM γ ) 1 2 , resulting in:

<!-- formula-not-decoded -->

Notice that the right-hand side of this inequality is independent of k . With α = ( √ 3 2 √ ϵM γ ) 1 2 &lt; 1 , for any δ &gt; 0 , there exists t &gt; 1 such that for all m &gt; t , ∥ f θ m -f θ t ∥ ≤ δ . Thus, the sequence is Cauchy with respect to the norm ∥ · ∥ ; and by the compactness (and therefore completeness) of F , it converges to a point f ∗ .

To show that f ∗ is a stable point, we start by showing the continuity of the mapping

<!-- formula-not-decoded -->

where D ( f θ 1 , f θ 2 ) = D ( f θ 1 )+ D ( f θ 2 ) 2 . Applying Lemma 11 and Assumption 6, we obtain:

<!-- formula-not-decoded -->

Combining this with equations 41 and 42, we derive:

<!-- formula-not-decoded -->

Thus, for any sequence lim n →∞ ( f θ 1 n , f θ 2 n ) = ( f θ 1 , f θ 2 ) ,

<!-- formula-not-decoded -->

This implies that if lim n →∞ ( f θ 1 n , f θ 2 n ) = ( f θ 1 , f θ 2 ) , then lim n →∞ ∥U ( f θ 1 n , f θ 2 n ) -U ( f θ 1 , f θ 2 ) ∥ = 0 . By the continuity of U , we conclude:

<!-- formula-not-decoded -->

This establishes that f ∗ = f θ PS is a stable point.

6 We can derive a convergence guarantee for any n . The values of n ≤ 5 yields a bracket of ≤ 1 and thus can match or improve the class of convergence compared to just the last iterate. For larger n 's, we still have convergence, but on a smaller class of functions as the bracket &gt; 1 .

## F Proof of Lower bound in Perdomo et al. [2020] Framework

In this proof, we begin by considering a loss function defined as follows:

<!-- formula-not-decoded -->

This function is γ -strongly convex for the parameter θ and its gradient with respect to θ is β -Lipschitz in sample space. The necessary assumptions on the loss function, as outlined in Perdomo et al. [2020], are satisfied by this formulation.

We define the matrix A within R d × d as:

<!-- formula-not-decoded -->

The critical property of this matrix is that if a vector b The key property of this matrix is that if a vector b ∈ span { e i | i ≤ t } , then Ab ∈ span { e i | i ≤ t + 1 } , where each e i ∈ R d is a standard basis vector with all coordinates zero except for the i -th coordinate, which is 1. This structure enables the introduction of a new dimension only at the end of each RRM iteration. With the correct initialization, this ensures that the updates remain within a minimum distance from the stable point due to undiscovered dimensions.

We define D ( θ ) as the distribution of z given by:

<!-- formula-not-decoded -->

Note that since spectral radius A is 2 , the mapping D ( . ) defined as above would be ϵ -sensitive. Under this setting, the first-order Repeated Risk Minimization (RRM) update, starting with θ 0 = e 1 , is described by:

<!-- formula-not-decoded -->

Due to the properties of matrix A , we conclude that θ t +1 ∈ span { e i | ∀ i ≤ t +1 } .

The stationary point θ PS of this setup is located at:

<!-- formula-not-decoded -->

Note that at time step t the best model within the feasible set is θ t ∈ span { e i |∀ i ≤ t } . Given that one can conclude that the best L 1 -distance to stationary point achievable at time step t is lower bounded by the sum over the last d -t entries of θ PS . Setting d = 2 T and using Lemma 5 we get

<!-- formula-not-decoded -->

Similar to Repeated Risk Minimization (RRM), the Repeated Gradient Descent (RGD) method introduces a new dimension in each iteration step. Specifically, the gradient update rule in RGD is given by:

<!-- formula-not-decoded -->

This formulation ensures that each step effectively augments the dimensionality of the parameter space being explored only by a single dimension. Consequently, the lower bound established for RRM also applies to these RGD settings.

## F.1 Lower Bound for Proximal RRM

We proceed to use the exact same setup as Appendix F. Keeping in mind that the only changing factor here would be the optimization oracle at each step of RRM. Consider the general case of

<!-- formula-not-decoded -->

with D t defined as:

̸

<!-- formula-not-decoded -->

Now given that the loss is strongly convex, the minimizer retrieves a unique point according to the mapping:

<!-- formula-not-decoded -->

Same as before, using Lemma 9, we search for the stable point using the standard RRM update rule, which in this setup would be the following:

<!-- formula-not-decoded -->

This update rule for λ = ∞ would retrieve the stationary point θ PS ,

<!-- formula-not-decoded -->

Note that again, because of the structure of matrix A , Equation 50 can introduce a new dimension only after taking one step of RRM. Therefore, the best method any method in the ARM class can do is to only match the first t coordinates of the stationary point. So one can conclude that the best L 1 -distance to stationary point achievable at time step t is lower bounded by the sum over the last d -t entries of θ PS . Setting d = 2 T and using Lemma 5 we get

<!-- formula-not-decoded -->

## G Proof of Lower Bound for Modified Mofakhami et al. [2023] Framework

We define the model fitting function as f θ ( x ) = θ , and the corresponding loss function is:

<!-- formula-not-decoded -->

This loss is γ -strongly convex, ensuring unique minimizers and stable convergence properties. Additionally, we assume θ ∈ Θ = { z |∥ z ∥ ≤ δM γ } , ensuring that the gradient norm ∥ γθ -M (1 -δ ) xe -1 2 e ∥ x ∥ 2 ∥ remains bounded by M . This holds because the mapping f ( x ) = xe -1 2 e ∥ x ∥ 2 is chosen such that, f : R → [0 , 1] .

Observe that, for all f θ ∗ , f θ , f θ ′ ∈ F , we have ∥ θ -θ ′ ∥ = ∥ f θ -f θ ′ ∥ f θ ∗ :

<!-- formula-not-decoded -->

We define the distribution mapping as follows:

<!-- formula-not-decoded -->

where A is a lower triangular matrix:

<!-- formula-not-decoded -->

Matrix A has the property that if b is in the span of { e 1 , . . . , e i } , then Ab will be in the span of { e 1 , . . . , e i +1 } . Here, e i denotes the standard basis vector, where its i -th element is 1 and all other elements are 0. This makes A crucial for ensuring that each update step involves interactions that span progressively larger subspaces.

The χ 2 -divergence between two distributions D ( θ ) = N ( µ , Σ) and D ( θ ) = N ( µ , Σ) , where

1 1 2 2 µ 1 = √ σ 2 ϵ 2 Aθ + e 1 L and µ 2 = √ σ 2 ϵ 2 Aθ ′ + e 1 L , with Σ = σ 2 I , according to Lemma 7:

<!-- formula-not-decoded -->

Since the spectral norm of matrix A is 2, we have:

<!-- formula-not-decoded -->

Thus, the χ 2 -divergence between the distributions is bounded by ϵ ∥ θ -θ ′ ∥ 2 , ensuring that the divergence scales with the difference between θ and θ ′ .

The update rule for θ is:

<!-- formula-not-decoded -->

This is the unique minimizer of the loss function due to the γ -strong convexity. Additionally, this is a continuous mapping from a compact convex set Θ to itself. By Schauder 's fixed-point theorem, there exists a stable fixed point, denoted as θ PS , satisfying:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Choosing L ≥ 2 M (1 -δ ) γ ( δ + δ 2 ) one can guarantee the term in the projection operation would have a norm smaller than δ , i.e. it would be in Θ . So you can drop the projection operation from the equality above.

Thus, the stable point would hold true in the following equality:

<!-- formula-not-decoded -->

The same assumptions stated above would allow us to use Lemma 5:

<!-- formula-not-decoded -->

To lower bound c σ 2 ,θ PS , we note that ∥ E [ x ] ∥ ∈ [0 , ϵ 2 δ + γ ( δ + δ 2 ) 2 M (1 -δ ) ] and minimize the exponential term with respect to σ 2 :

<!-- formula-not-decoded -->

√

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where c &gt; 0 is a constant independent of δ . Setting σ = 2 2 to maximise σ σ 2 e +1 , and lim δ → 0 , we achieve:

Hence,

<!-- formula-not-decoded -->

## H Proof of Theorem 8

Consider a feature vector x divided into strategic features x s and non-strategic features x f , so that x = ( x s , x f ) . We resample only the strategic features with probability g ( f θ ( x )) , representing the probability of rejection for x . The pdf of the modified distribution p f θ is:

<!-- formula-not-decoded -->

where the integral is over all possible values of x ′ s with x f held constant, since only the strategic features are resampled. The first term represents the option that we accept the first sample at x ; the second term represents the possibility that we reject the first sample at x ′ = ( x ′ s , x f ) and then resample at x s to obtain x as well.

Assuming the strategic and non-strategic features are independent, we can rewrite this expression as:

<!-- formula-not-decoded -->

where p X s and p X f are the marginal distributions of the strategic and non-strategic features, respectively, and we define:

<!-- formula-not-decoded -->

Since 0 ≤ f θ ( x ) ≤ 1 -δ for some δ &gt; 0 , it follows that δ ≤ g ( f θ ( x )) ≤ 1 for every x . Therefore, δ ≤ C θ ( x f ) ≤ 1 .

In the RIR procedure, the distribution of the label y given x is not affected by the predictions so for every z = ( x, y ) we have p f θ ( z ) = p f θ ( x ) p ( y | x ) for any f θ . This results in the following equality:

<!-- formula-not-decoded -->

We prove that this mapping is ϵ -sensitive with respect to χ 2 divergence, where ϵ = 1 δ ( 1 + 1 -δ 2 √ δ ) .

<!-- formula-not-decoded -->

This inequality follows from the fact that δ ≤ C θ ( x f ) and 1 -g ( f θ ( x )) ≥ 0 , therefore 1 1 -g ( f θ ( x ))+ C θ ( x f ) ≤ 1 δ .

Continuing, we have:

<!-- formula-not-decoded -->

This comes from the fact that ∫ x ′ s p X s ( x ′ s )( f θ (( x ′ s , x f )) -f θ ′ (( x ′ s , x f ))) dx ′ s = C θ ( x f ) -C θ ′ ( x f ) . We use equation 51 to replace p ( x ) .

<!-- formula-not-decoded -->

Since g ( f θ ( x )) is a bounded random variable in [ δ, 1] , its variance is less than (1 -δ ) 2 4 , according to Popoviciu's inequality. Also since for any θ ∈ Θ we have f θ ( x ) ≤ 1 we can infer | f θ ( x ) -f θ ′ ( x ) | ≤ 1

<!-- formula-not-decoded -->

From Appendix A.3 in Mofakhami et al. [2023], we know that ∥ f θ -f θ ′ ∥ 2 ≤ 1 δ ∥ f θ -f θ ′ ∥ 2 f θ . Hence,

<!-- formula-not-decoded -->

Rate Improvement Arguments: By using Assumptions 4 and 6 from Mofakhami et al. [2023], it can be shown that the method is Cϵ -sensitive as defined in Assumption 1. Specifically,

<!-- formula-not-decoded -->

In this case, our rate aligns with the rate from Mofakhami et al. [2023], demonstrating that in all cases where their rate holds, our approach offers at least an equivalent or faster rate. However, there are instances where our rate results in a smaller constant than Cϵ . As outlined in Appendix A.3 of Mofakhami et al. [2023], the same RIR framework derives C = 1 δ under Assumption 2 and ϵ = 1 δ with respect to Assumption 6, yielding Cϵ = 1 δ 2 . We show that instead of Cϵ = 1 δ 2 , we obtain 1 δ ( 1 + 1 -δ 2 √ δ ) , which is strictly smaller for any 0 ≤ δ &lt; 1 . This shows that this rate is a strict improvement over Mofakhami et al. [2023].

Figure 4: Log performative risk for the credit scoring environment across the RRM iterations. The numbers in the plot are averaged over 500 runs. Increasing the size of aggregation window τ from 1 → 2 → 4 → t/ 2 → all reduces the oscillations in the risk and converges to the same point. Note that the plot starts from iteration 5 for better readability as the initial risk values were very high.

<!-- image -->

## I Performative Risk for Credit-Scoring

Figure 4 shows the log performative risk for the credit-scoring environment. This metric has been adapted from Mofakhami et al. [2023]. Figure 4 further substantiates our claims as we see lower oscillations in the risk for larger aggregation windows. We also note that although larger windows yield more stable trajectories, they can incur worse performative risk along the way before ultimately settling. Furthermore, methods converge to roughly the same performatively stable point-as predicted by the theory of a unique stable point in Lemma 9-since the difference in log performative risk at the end of 50 iterations is negligible between all methods. However, as pointed out in Section 7, all methods oscillate in a similar range, thus hindering the readability of the plot.

Hyper-parameters. For our experiments, we fix the value of δ = 0 . 55 . The RRM procedure is carried out for a maximum of 50 iterations with a learning rate of 3 e -4 and Adam optimizer run over A100-40G GPUs. Each run only took a few minutes. Further, all the experimental results and plots are averaged over 500 runs, where each run for each method has the same model initialization. Thus, the only source of randomization is the sampling under RIR mechanism, where the sampling changes across different runs but is the same for all the methods given a specific run.

Figure 5: The plot shows loss shift due to performativity and performative risk across the iterations for player 1 in the game between two firms. The values in the plot are means over 200 runs. Increasing the aggregation window size τ leads to lower loss shifts even in this simple game and hence, faster convergence than just relying on the dataset from the current timestamp.

<!-- image -->

## J Revenue Maximization in Ride Share Market

Setup. This is a two-player semi-synthetic game between two ride-share providers, Uber and Lyft, both trying to maximize their respective revenues. Each player takes an action in this game by setting their price for the riders across 11 different locations in the same city of Boston, MA. The price set by one firm directly influences the demand observed by both firms. The demand constitutes the data distribution and at each time step, a total of 25 demand samples are sampled for a firm i , and the optimal response is found by minimizing Equation 53 for a maximum of 40 retraining steps on CPU. Each run took only a few seconds. The simulations use the publicly available Uber and Lyft dataset from Boston, MA on Kaggle 7 .

Notations and Equations. Let i = 1 , 2 denote the two firms in the game. Inspired by Narang et al. [2024], each firm i observes a demand z i that depends linearly on the firm's price x i and its opponent's price x -i as follows:

<!-- formula-not-decoded -->

where z base is the mean demand observed at each of the 11 locations, as measured in the kaggle dataset. Each demand sample is a vector of dimension 11 .

A i and A -i are fixed matrices representing the price elasticity of demand, i.e. the change in demand due to a unit change in price for the player i and -i (opponent) respectively. We introduce interactions between the ride prices in a location and the demand in a different location within the same city by making A matrices non-diagonal. Additionally, note that the price elasticities A i will always be negative as the firm will experience less demand if it increases its price. Similarly, the price elasticities A -i will be positive.

Each player observes a revenue of z T i x i . Thus, the loss function that each player i seeks to minimize in the RRM framework can be described as:

<!-- formula-not-decoded -->

where λ is a hyperparameter for the regularization term (= 70 for our experiments). For any player i , each element of x i is clipped to be between the range of [ -30 , 30] and the initial price x 0 i is sampled randomly from a uniform distribution on [0 , 1] .

Results. Figure 5 shows the plot for loss shift due to performativity and the performative risk versus the iterations averaged over 200 runs. For this plot, we assume player 1 is the player who makes the predictions and adjusts to the performative effects introduced due to the actions of player 2. It can be clearly observed that as we increase the aggregation window from 1 → 2 → 4 → t/ 2 → all , we

7 Uber and Lyft dataset from Boston, MA, 2019: https://www.kaggle.com/datasets/brllrb/ uber-and-lyft-dataset-boston-ma

get mostly lower loss shifts and hence, an improvement in the convergence rate. Since we start at random price value, taking the past into account in the beginning makes the algorithm worse but the effect is neutralized as the data from more time steps is observed. Given the simple linear nature of the problem, this is a significant improvement and provides evidence for our claims about using the data from the previous snapshots. Secondly, performative risk plot in figure 5 also highlights that all methods converge to points having very close values of performative risk, with the methods having larger τ showing oscillations with smaller amplitude.

## K Proximal RRM Convergence in Perdomo et al. [2020] Framework

Theorem 9 (Proximal RRM Convergence) Suppose the loss ℓ ( z ; θ ) is β -jointly smooth and γ -strongly convex. If the distribution map D ( · ) is ϵ -sensitive, then for

<!-- formula-not-decoded -->

we have, for all θ, θ ′ ∈ Θ ,

Proof. Fix θ and θ ′ . Let

<!-- formula-not-decoded -->

Because ℓ ( z ; ϕ ) is γ -strongly convex, both f and f ′ are ( γ + λ ) -strongly convex. Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since f is minimized at G ( θ ) , the inner product in 56 is non-negative. Combining 55 and 56 yields

<!-- formula-not-decoded -->

Define the regularized loss

The map

<!-- formula-not-decoded -->

Furthermore, if ϵβ + λ γ + λ &lt; 1 , then G is a contraction, possesses a unique fixed point θ PS , and the proximal RRM iterates θ t +1 = G ( θ t ) converge linearly:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is 1 -Lipschitz in z because of the β -joint smoothness of ℓ . The ϵ -sensitivity of D ( · ) then implies

<!-- formula-not-decoded -->

Using the 1 -Lipschitz function above in 58 gives

∣

∣

∣

<!-- formula-not-decoded -->

Unfolding ℓ θ and rearranging, we obtain

<!-- formula-not-decoded -->

Since G ( θ ′ ) minimizes f ′ , we have ( G ( θ ) -G ( θ ′ ) ) ⊤ ∇ ϕ f ′ ( G ( θ ′ ) ) ≥ 0 . Thus

<!-- formula-not-decoded -->

by the Cauchy-Schwarz inequality. Combining 59 and 60 gives

<!-- formula-not-decoded -->

and substituting the upper bound 57 for the right-hand side yields

<!-- formula-not-decoded -->

Since ∥ G ( θ ) -G ( θ ′ ) ∥ ≥ 0 , dividing both sides of 62 by ( γ + λ ) ∥ G ( θ ) -G ( θ ′ ) ∥ gives

<!-- formula-not-decoded -->

## Stability of the Proximal RRM Solution in Standard RRM

We examine whether the fixed point of the Proximal RRM introduced in Theorem 9 coincides with the performatively stable point of RRM.

Assume that ϵβ γ &lt; 1 , Under this condition, a performatively stable point of RRM exists and it's unique. Because the proximal term vanishes at ϕ = θ PS , optimality of θ PS yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so θ PS is also a fixed point of the Proximal RRM.

Note that, for any λ &gt; 0 ,

<!-- formula-not-decoded -->

By Theorem 9, inequality 65 guarantees that the Proximal RRM admits a unique fixed point, denoted by θ λ PS . Since θ PS satisfies 64 and the minimiser of 64 is unique, we conclude

<!-- formula-not-decoded -->

Thus, under eβ γ &lt; 1 , the performatively stable solution of RRM is identical to the unique fixed point of the Proximal RRM.

## L Limitations

Despite our contributions, several limitations warrant discussion. First, although we prove that ARM achieves performative stability under a broader set of sensitivity conditions than standard RRM, this does not necessarily translate into faster convergence rates. Second, while the underlying techniques can extend to other iterative schemes, such as Repeated Gradient Descent (RGD), and more general gradient-based methods, we do not provide explicit convergence analyses or lowerbound characterizations for these alternatives. Finally, our entire treatment assumes deterministic, non-stochastic setups with exact access to full distributions; we leave the problem of accommodating sampling noise and stochastic gradients for future work.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide support for all the claims made in the submission.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in Appendix L.

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

Justification: All the assumptions required are prescribed in the main text.

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

Justification: The experimental setup to reproduce the results for both experiments is provided in the Appendix J and I.

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

Justification: Code will be provided in the supplementary material. Link to the public datasets used throughout the experiments is provided in the text.

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

Justification: Details are provided in the Appendix J and I.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Because the methodology requires linear memory overhead, we couldn't retain the intermediate data needed for error bars, and since variance grows with more steps, demanding an impractically large number of runs to mitigate, we report only mean trends.

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

Justification: Details provided in the Appendix J and I.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our paper follows NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: [NA]

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Original owners of the experimental setups used in the paper are referenced.

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
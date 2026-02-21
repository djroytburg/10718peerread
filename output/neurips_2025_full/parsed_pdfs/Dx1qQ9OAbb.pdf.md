## On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts

Fanqi Yan ⋆, 1 Huy Nguyen ⋆, 2 Dung Le ⋆, 2 Pedram Akbarian Nhat Ho 2 Alessandro Rinaldo 2

3

1 Department of Computer Science, 2 Department of Statistics and Data Sciences, 3 Department of Electrical and Computer Engineering, The University of Texas at Austin {fanqi.yan, huynm, quangdung0110, akbarian, minhnhat}@utexas.edu , alessandro.rinaldo@austin.utexas.edu

## Abstract

The softmax-contaminated mixture of experts (MoE) model is deployed when a large-scale pre-trained model, which plays the role of a fixed expert, is fine-tuned for learning downstream tasks by including a new contamination part, or prompt, functioning as a new, trainable expert. Despite its popularity and relevance, the theoretical properties of the softmax-contaminated MoE have remained unexplored in the literature. In the paper, we study the convergence rates of the maximum likelihood estimator of gating and prompt parameters in order to gain insights into the statistical properties and potential challenges of fine-tuning with a new prompt. We find that the estimability of these parameters is compromised when the prompt acquires overlapping knowledge with the pre-trained model, in the sense that we make precise by formulating a novel analytic notion of distinguishability. Under distinguishability of the pre-trained and prompt models, we derive minimax optimal estimation rates for all the gating and prompt parameters. By contrast, when the distinguishability condition is violated, these estimation rates become significantly slower due to their dependence on the prompt convergence rate to the pre-trained model. Finally, we empirically corroborate our theoretical findings through several numerical experiments.

## 1 Introduction

Mixture of experts (MoE) [14, 16] has emerged as a statistical machine learning model that aggregates the power of multiple sub-models. This model consists of two primary components: expert function (or, simply, expert) and a gating network. Experts can be, for example, a feed-forward network (FFN) [33, 4], a classifier [2, 27], or a regression model [7, 17]. The gating network softly divides the input space into multiple regions where the opinions of some experts are deemed to be more trustworthy than others. This is done by dynamically allocating higher input-dependent weights instead of constant weights to the various experts, making MoE more flexible and adaptive than traditional mixture models [25]. As a consequence, MoE has been leveraged in a wide range of fields, including natural language processing [5, 15, 10, 8, 21, 33], computer vision [32, 24], speech recognition [36, 37], multimodal learning [11, 38, 28], continual learning [20, 22], and reinforcement learning [1, 3].

Unlike these applications where all experts are trainable, parameter-efficient fine-tuning methods such as prefix tuning [23, 19, 18] can be interpreted as a mixture of a frozen or pre-trained expert and

⋆ Co-first authors.

a trainable prompt expert responsible for learning downstream or more specialized tasks, which we refer to as contaminated MoE throughout this paper. Despite the empirical success of this fine-tuning approach, there is a very limited theoretical understanding of their properties and limitations in the literature. To the best of our knowledge, contaminated MoE has only been previously studied in [35] to characterize expert structures achieving the optimal parameter estimation rates. However, the analysis in that work is conducted under a simplified setting where the gating (mixture weight) is independent of the input value, which is a very impractical assumption. To close this gap, we undertake a thorough theoretical analysis of the more commonly used softmax-contaminated MoE model, specified in equation (1) below, a contaminated MoE model whose gating function takes the form of a soft-maxed linear network.We analyze the issue of identifiability and the convergence properties of the maximum likelihood estimator of the prompt parameters to shed light on the understanding of prompt behavior in prefix tuning methods. A main take-away of our analysis is the potential for the prompt to be exceedingly similar to - and thus to acquire the same knowledge as - the pre-trained model, a situation greatly impacting the estimability of the prompt parameter. To overcome this issue, in Definition 1 we formulate analytical properties of the pre-trained and prompt models, which we refer to as distinguishability, that are guaranteed to rule out excessive overlap between the models and ensure good estimation rates. We make the following contributions.

(i) Distinguishability of the prompt model from the pre-trained model. In Section 2, we propose a novel notion of distinguishability between the pre-trained and prompt models and then illustrate its properties.

(ii) When the distinguishability condition is satisfied, we show in Section 3.1 that the prompt does not converge to the pre-trained model - intuitively, these two models have distinct expertise. In fact, we demonstrate that the convergence rates of the MLE of all the prompt and gating parameters are of parametric order in the sample size n , that is, ˜ O ( n -1 / 2 ) . Furthermore, we establish minimax lower bounds on the estimation errors with matching rates, thus showing that the convergence rate of MLE is minimax optimal.

(iii) When the distinguishability condition is violated, the prompt will converge to the pre-trained model, that is, both models employ the same expert structure and thus will gain similar expertise. In Section 3.2, we show that, under this setting, the estimation rates for prompt and gating parameters are negatively affected by the prompt convergence to the pre-trained model and, therefore, become substantially slower than the parametric rate ˜ O ( n -1 / 2 ) . We confirm that these slower rates are tight by deriving matching minimax lower bounds. See Table 1 for a summary of our results.

Lastly, in Section 4, we carry out several numerical experiments to empirically justify our theoretical results, and then conclude the paper in Section 5. Rigorous proofs are provided in the Appendices.

A major technical innovation in our contribution that sets it apart from existing theoretical analyses of MoE models is the fact that we let the parameters of the prompt model to vary with the sample size n , thus potentially allowing for a more challenging estimation task as the sample size increases. This approach is necessary to carry out a minimax analysis.

Notation. For any n ∈ N , we let [ n ] := { 1 , 2 , . . . , n } . For a vector u we denote with ∥ u ∥ its Euclidean norm value. Given any two positive sequences ( a n ) n ≥ 1 and ( b n ) n ≥ 1 , we write a n = O ( b n ) or a n ≲ b n if a n ≤ Cb n for all n ∈ N and some C &gt; 0 . We further write a n = ˜ O ( b n ) to denote a n ≲ b n polylog( b n ) , where polylog( b n ) indicate any term that is polylogarithmic in b n . Lastly, for any two densities p and q (dominated by the Lebesgue measure), their squared Hellinger distance is computed as d 2 H ( p, q ) := 1 2 ∫ [ √ p ( x ) -√ q ( x )] 2 dx , while the total variation distance is given by d V ( p, q ) := 1 2 ∫ | p ( x ) -q ( x ) | dx .

Table 1: Summary of parameter estimation rates in the softmax-contaminated MoE model. Notice that the rates are in expectation. For the notation, please refer to equations (1) and (2). In addition, we also denote ∆ η ∗ := η ∗ -η 0 and ∆ ν ∗ := ν ∗ -ν 0 .

| Setting             | &#124; exp( ̂ τ n ) - exp( τ ∗ ) &#124;    | ∥ ̂ β n - β ∗ ∥                            | ∥ ̂ η n - η ∗ ∥                            | &#124; ̂ ν n - ν ∗ &#124;                  |
|---------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
| Distinguishable     | ˜ O ( n - 1 2 )                            | ˜ O ( n - 1 2 )                            | ˜ O ( n - 1 2 )                            | ˜ O ( n - 1 2 )                            |
| Non-distinguishable | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 2 ) | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 1 ) | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 1 ) | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 1 ) |

## 2 Preliminaries

In this section, we begin with setting up the problem, followed by a discussion on related works in Section 2.1. Then, in Section 2.2, we introduce the distinguishability condition and provide an investigation into the fundamental properties of the softmax-contaminated MoE, including the model identifiability and the model convergence.

## 2.1 Problem Setup

Problem setting. Suppose that ( X 1 , Y 1 ) , ( X 2 , Y 2 ) , . . . , ( X n , Y n ) ∈ X × Y ⊂ R d × R are i.i.d. samples of covariate-response pairs of size n . We assume that the input covariates X 1 , X 2 , . . . , X n are drown in an i.i.d. manner from some known continuous probability distribution on R d and that the responses are generated according to a softmax-contaminated MoE model, which postulates that the conditional density function of the response given the covariates is given by

<!-- formula-not-decoded -->

Above, the pre-trained model corresponds to as a fixed and known conditional probability density function f 0 ( ·| h 0 ( · , η 0 ) , ν 0 ) , parametrized by the pre-trained mean expert function x ↦→ h 0 ( x, η 0 ) and variance ν 0 . Meanwhile, the prompt model, denoted as f ( ·| h ( · , η ∗ ) , ν ∗ ) is modeled as an unknown Gaussian density function with the prompt mean expert x ↦→ h ( x, η ∗ ) and variance ν ∗ . We collect all the unknown parameters of the prompt model into the vector G ∗ = ( β ∗ , τ ∗ , η ∗ , ν ∗ ) , belonging to some parameter space Ξ ⊆ R d × R × R q × R + . Note that we allow the values of these parameters to vary with the sample size n . However, for notational convenience, we suppress the dependence of G ∗ on n throughout the paper. In addition, it should also be noted that the 'probabilistic" MoE model (1) can be related to 'deterministic" MoE models used in deep learning [33] by taking the expectation of the response given the covariate, that is,

<!-- formula-not-decoded -->

Maximum likelihood estimation (MLE). We utilize the maximum likelihood method [34] to estimate the unknown parameters G ∗ = ( β ∗ , τ ∗ , η ∗ , ν ∗ ) of the softmax-contaminated MoE model (1) as follows:

<!-- formula-not-decoded -->

For the sake of theory, we assume that the input space X is bounded, whereas the parameter space Ξ is compact. In addition, we assume that the prompt expert function x ↦→ h ( x, η ) is differentiable with respect to η ∈ R q for almost all x ∈ X . Note that these assumptions are mild and have been used in previous works [13, 30, 35].

Related work. Mendes et al. [26] considered an MoE model where each expert was formulated as a polynomial regression model. Their objective was to address the trade-off between the number of experts and the expert size to obtain the optimal parameter estimation rates. Next, Ho et al. [13] took into account the parameter estimation problem for Gaussian MoE models with input-free gating. They demonstrated that when expert functions satisfied an algebraic independence condition, the convergence rates of MLE were optimal of parametric order on the sample size. Conversely, if the expert functions are not algebraic independent, then the parameter estimation rates became inversely proportional to the number of fitted experts. These results were then extended to more practical settings of input-dependent gatings, including softmax gating [31] and sigmoid gating [29], revealing that the latter was more sample-efficient than former in terms of expert estimation.

It was not until 2024 that Nguyen et al. [30] investigated a contaminated MoE where a frozen pretrained model was fine-tuned by a mixture of prompts rather than a single prompt model. However, they imposed two unrealistic assumptions on their model of interest: they equipped the contaminated MoE with input-free gating and kept the ground-truth parameters unchanged with the sample size.

Then, Yan et al. [35] overcame the second limitation by allowing ground-truth parameters to hinge on the sample size as in the case of traditional mixture models [6], while the first limitation remained unsolved. Therefore, in this work, our goal is to completely address both limitations by studying the softmax-contaminated MoE in equation (1).

Challenges. There are three fundamental challenges of our analysis compared to previous work.

1. Uniform convergence rates. We allow ground-truth parameters G ∗ to change with sample size n , which is challenging yet closer to practice than the settings in previous works on MoE [31, 29], where G ∗ does not change with n . Thus, the convergence rates of parameter estimations in our work are uniform rather than point-wise as in those works.
2. Minimax lower bounds. We determine minimax lower bounds under both distinguishable and non-distinguishable settings. Based on these lower bounds, we can claim that our derived convergence rates are optimal. However, no minimax lower bounds are provided in [31, 29].
3. Input-dependent gating. The latest work on understanding the contaminated MoE model is [35], but it considers input-free gating in the analysis. On the other hand, in this paper, we take into account softmax gating, which hinges upon the input value. This input-dependence yields several challenges on the convergence of density estimation and parameter estimation.

## 2.2 Fundamental Properties of the Softmax-Contaminated MoE

As mentioned above, when the prompt's learned skills overlap with those of the pre-trained model, estimating the prompt parameters becomes challenging due to potential non-identifiability. To capture that issue accurately, we introduce an analytic condition called distinguishability in Definition 1.

Definition 1 (Distinguishability) . We say that f 0 is distinguishable from f if the following hold: for any distinct pairs of parameters ( η 1 , ν 1 ) , ( η 2 , ν 2 ) ∈ Θ , if there exist measurable real-valued functions x ∈ X ↦→ b 0 ( x ) , x ∈ X ↦→ b 1 ( x ) , and x ∈ X ↦→ { c α ( x ) } 0 ≤| α |≤ 1 , where α = ( α 1 , α 2 ) ∈ N q × N with | α | = | α 1 | + α 2 ≤ 1 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for almost every ( x, y ) ∈ X × Y , then it must be the case that

<!-- formula-not-decoded -->

To help understand the notion of distinguishability better, in our next result we characterize the class of pre-trained models distinguishable from the prompt f . The proof can be found in Appendix B.1.

Proposition 1. If a pre-trained model f 0 does not belong to the family of Gaussian densities, then f 0 is distinguishable from the prompt model f in the sense of Definition 1.

On the other hand, if f 0 belongs to the family of Gaussian distributions and the pre-trained expert shares the same structure as the prompt expert, that is, h 0 = h , then the above condition is violated. It should be noted that the distinguishability condition ensures that the prompt does not acquire overlapping knowledge with the pre-trained model since the equation f 0 ( y | h ( x, η 0 ) , ν 0 ) = f ( y | h ( x, η ) , ν ) cannot hold for almost all ( x, y ) ∈ X × Y . Moreover, we illustrate in the following proposition that the distinguishability condition also implies that the softmax-contaminated MoE is identifiable.

Proposition 2 (Identifiability) . Let G,G ′ be two components in Ξ . Suppose that f is distinguishable from f 0 , then if the identifiability equation p G ( y | x ) = p G ′ ( y | x ) holds for almost all ( x, y ) ∈ X × Y , then we obtain G = G ′ .

The proof of Proposition 2 is provided in Appendix B.2. Given the consistency of the softmaxcontaminated MoE, we continue to investigate the convergence behavior of density estimation under this model in Proposition 3 whose proof can be found in Appendix B.3. We conclude this section with a consistency guarantee for the contaminate density itself, which under mild tail conditions on f 0 , can be estimated at a parametric rate in the Hellinger distance, regardless of the distinguishability between f 0 and f . Below and throughout the paper, E p G ∗ ,n denotes the expectation operator with respect

to the joint distribution of the data ( X 1 , Y 1 ) , . . . , ( X n , Y n ) and assuming the softmax-contaminated MoE model (1) parametrized by G ∗ ∈ Ξ , i.e. Y i | X i ∼ p G ∗ for all i . Instead, E X indicates the expectation with respect to the input distribution.

Proposition 3 (Model Convergence) . Suppose that the pre-trained model f 0 is bounded and, for some p &gt; 0 ,

<!-- formula-not-decoded -->

Then, for the MLE ̂ G n defined in equation (2) , it holds, for almost all x ∈ X ,

<!-- formula-not-decoded -->

The above result shows that the density estimator p ̂ G n converges to the true density p G ∗ under the Hellinger distance at the near-parametric rate of order ˜ O ( n -1 / 2 ) . To extract from this result a convergence guarantee for the MLE ̂ G n itself, we follow a by-now-standard approach in the latest analysis of MoEs; see, e.g., [31]. The main idea is that, if one can exhibit a loss function among parameters, say D ( ̂ G n , G ∗ ) , such that E p G ∗ ,n [ D ( ̂ G n , G ∗ )] ≲ E p G ∗ ,n [ E X [ d H ( p ̂ G n ( ·| X ) , p G ∗ ( ·| X ) )] , then convergence of ̂ G n in the expected D ( · , · ) loss, as well potentially information on the rate of convergence, will follow. See Appendix A for further details. Throughout the rest of the paper, we assume that the tail condition (3) on f 0 and the distribution of X used in Proposition 3 is in effect.

## 3 Convergence Analysis of Parameter Estimation

In this section, we present various convergence rates for the MLE estimator of the model prompt and gating parameters. In Sections 3.1 and 3.2 we provide separate minimax analyses, depending on whether the distinguishability condition of Definition 1 holds or not, respectively.

## 3.1 Distinguishable Setting

To start with, we consider a scenario in which the pre-trained model f 0 is distinguishable from the prompt model f . Recall that given the density estimation rate in Proposition 3, we need to construct a loss function between the MLE ̂ G n and the ground-truth parameters G ∗ , which should be bounded by the Hellinger distance between the two corresponding densities, in order to capture the parameter estimation rates. Tailored to the distinguishable setting, we measure the discrepancy between two arbitrary parameters G and G ∗ in Ξ via the loss

<!-- formula-not-decoded -->

We are ready to determine the convergence behavior of the MLE under distinguishable settings.

̸

Theorem 1. Suppose that the pre-trained model f 0 is distinguishable from the prompt model f . For almost every x ∈ X , and for any η ∈ R q , we assume that the Jacobian of the prompt expert function does not vanish, i.e., ∂h ∂η ( x, η ) = 0 . Then, there exists a positive constant C 1 that depends on Ξ and f 0 such that the Hellinger lower bound E X [ d H ( p G ( ·| X ) , p G ∗ ( ·| X ))] ≥ C 1 D 1 ( G,G ∗ ) holds for all parameters G ∈ Ξ . As a result, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Theorem 1 is deferred to Appendix A.1. The bound in equation (6) reveals that the gating parameter estimator exp( ̂ τ n ) converges to its ground-truth counterpart exp( τ ∗ ) at a rate of order ˜ O ( n -1 / 2 ) . Analogously, looking at the bound in equation (7), since the terms exp( τ ∗ ) cannot go to zero due to the compactness of the parameter space Ξ , it follows that the convergence rates of the parameter estimators ̂ β n , ̂ η n , and ̂ ν n to β ∗ , η ∗ and ν ∗ are also of order ˜ O ( n -1 / 2 ) . Meanwhile, in the contaminated MoE with input-free gating in [35], the estimation rates for prompt parameters

η ∗ , ν ∗ are slower than ˜ O ( n -1 / 2 ) as they depend on the convergence rate of the gating parameter to zero. Therefore, replacing the input-free gating with the softmax gating in the contaminated MoE helps reduce the sample complexity of parameter estimation.

Given the near-parametric convergence rates in Theorem 1, it is natural to wonder if they are optimal. To answer this question in the affermative, below we derive minimax lower bounds.

Theorem 2. If the pre-trained model f 0 is distinguishable from the prompt model f , then the following minimax lower bounds hold for any 0 &lt; r &lt; 1 :

<!-- formula-not-decoded -->

where the infimum is over all estimators G n := ( β n , τ n , η n , ν n ) taking values in Ξ .

The proof of Theorem 2 can be found in Appendix A.2. The above minimax lower bounds imply that, under distinguishability, the convergence rates of the MLE, of order ˜ O ( n -1 / 2 ) is nearly minimax optimal, save for a logarithmic factor.

## 3.2 Non-distinguishable Setting

We now turn to the much subtler case in which the distinguishability condition is violated. Since we assume a Gaussian prompt, it follows from Proposition 2 that the pre-trained model f 0 necessarily belongs to the family of Gaussian densities. Furthermore, if the pre-trained and prompt model use the same expert function, i.e. h 0 = h , then f 0 is not distinguishable from the prompt model f . We will thus focus on this challenging scenario.

Under this setting, the prompt model may converge to the pre-trained model. In particular, if the pair of prompt parameters ( η ∗ , ν ∗ ) converge to the pair of pre-trained parameters ( η 0 , ν 0 ) as n →∞ , then it follows that f ( ·| h ( · , η ∗ ) , ν ∗ ) converges to f 0 ( ·| h ( · , η 0 ) , ν 0 ) , indicating that the prompt learns the same expertise as the pre-trained model. Therefore, it becomes difficult for the gating network to assign higher weight to either the pre-trained model or the prompt than the other as they have similar expertise. As a result, one may expect the estimation rates of the gating parameters to be substantially slower. To formalize these setttings precisely, we need to pay more attention to the expert structure.

It should be noted that a key step in obtaining the MLE convergence rates in Theorem 1 is to decompose the density discrepancy p ̂ G n -p G ∗ into a combination of linearly independent terms through an appropriate Taylor series expansion of the function g ( y | x ; β, η, ν ) := exp( β ⊤ x ) · f ( y | h ( x, η ) , ν ) with respect to its parameters β, η, ν . This process involves, in particular, higher derivatives of the expert function h with respect to η , which may not be algebraically independent. To ensure the linear independence of the terms in the Taylor expansion, we formulate a strong identifiability condition that is indeed sufficient for these purposes.

Definition 2 (Strong Identifiability) . The expert function x ↦→ h ( x, η ) is strongly identifiable if it is twice differentiable with respect to η ∈ R q for almost all x ∈ X , and if, for any fixed β ∈ R d and η ∈ R q , each of the following sets of real-valued functions (of x ) consists of linearly independent functions over R . For notational simplicity, we write h ( · ) in place of h ( · , η ) below.

## 1. The first-order gating independence set:

<!-- formula-not-decoded -->

2. The gradient product independence set:

<!-- formula-not-decoded -->

3. The mixed and second-order independence set:

<!-- formula-not-decoded -->

Here, the First-order gating independence condition guarantees that changes in h with respect to η remain distinguishable, even after modulation by the gating weights exp( β ⊤ X ) . This is a minimal requirement to ensure that the expert and gating mechanisms interact in a structurally non-degenerate way. The Gradient product independence condition guarantees that the products of directional derivatives of h are distinguishable from each other (even under modulation by gating terms) and cannot be expressed as a linear combination of basic functions. This prevents higher-order interactions among gradients from collapsing into lower-order structures. Finally, the Mixed and second-order independence condition is stronger than the first-order one. It rules out first-order interactions between expert and gating parameters of the form ∂h/∂η ( w ) = x ( w ) · ∂h/∂η ( v ) , which would imply ∂g/∂η ( w ) = ∂ 2 g/ ( ∂β ( w ) ∂η ( v ) ) . It also requires that second-order derivatives remain linearly independent, even accounting for the effect of the gating function. This guarantees that both first- and second-order directional changes in h convey distinct, non-redundant information, and that higher-order structure in h cannot be reduced to or absorbed by lower-order terms. This is essential when handling second-order Taylor expansions of the model.

Examples. The expert functions h ( x, η ) = GELU( η ⊤ x ) , h ( x, η ) = sigmoid( η ⊤ x ) , and h ( x, η ) = tanh( η ⊤ x ) satisfy the strong identifiability condition, as their nonlinearities avoid degeneracies. In contrast, h ( x, η ) = ReLU( η ⊤ x ) fails the second-order independence condition, as the second-order derivatives vanish almost everywhere. Another failure case arises when h ( x, η ) = σ ( a ⊤ x + b ) , where η = ( a, b ) and σ is any scalar activation function. This leads to ∂h/∂a = x · ∂h/∂b , directly violating Condition 3.

To determine the convergence rates for the MLE in these settings, we construct the following loss function between parameters G and G ∗ , carefully tailored to the non-distinguishable setting:

<!-- formula-not-decoded -->

Theorem 3. Suppose that f 0 belongs to the family of Gaussian densities and h 0 = h . Then, there exists a positive constant C 2 that depends on Ξ , η 0 , ν 0 such that E X [ d H ( p G ( ·| X ) , p G ∗ ( ·| X ))] ≥ C 2 D 2 ( G,G ∗ ) holds for all parameters G . As a result, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any sequence ( l n ) n ≥ 1 such that l n / log n →∞ as n →∞ where we denote

<!-- formula-not-decoded -->

The proof of Theorem 3 is in Appendix A.3. Note that under the setting of Theorem 3, the softmaxcontaminated MoE model is not identifiable, that is, the equation p G ( y | x ) = p G ∗ ( y | x ) for almost all ( x, y ) does not imply G = G ∗ . For that reason, we restrict the parameter space to the set Ξ( l n ) to guarantee the consistency of the MLE. Compared to Theorem 1, the above rates exhibit differ in several aspects.

(i) From equation (8), we observe that the convergence rate of exp( ̂ τ n ) to exp( τ ∗ ) becomes slower than the parametric order ˜ O ( n -1 / 2 ) as they depend on the vanishing rate of (∆ η ∗ , ∆ ν ∗ ) to zero. For example, if the pair of prompt parameters ( η ∗ , ν ∗ ) approach ( η 0 , ν 0 ) at the rate of ˜ O ( n -1 / 8 ) , then the bound (8) implies that exp( ̂ τ n ) goes to exp( τ ∗ ) at the rate of ˜ O ( n -1 / 4 ) . This toy example is indeed confirmed by our numerical experiments in the next section.

(ii) Likewise, the convergence rates of the estimators ( ̂ β n , ̂ η n , ̂ ν n ) are also impacted by the convergence rates of the prompt parameters and therefore slower than ˜ O ( n -1 / 2 ) . For example, if

(∆ η ∗ , ∆ ν ∗ ) go to zero at the rate of ˜ O ( n -1 / 8 ) , then the bound (9) indicates that ̂ β n , ̂ η n , ̂ ν n converges to β ∗ , η ∗ , ν ∗ at the rate of ˜ O ( n -3 / 8 ) , respectively. Again, in our numerical experiments below we empirically verify this behavior.

In our final result, whose proof can be found in Appendix A.4, we show that the slower converge rates for the MLE under non-distinguishability are in fact essentially minimax optimal.

Theorem 4. Suppose that f 0 belongs to the family of Gaussian densities and h 0 = h . Then, the minimax lower bounds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hold for any sequence ( l n ) n ≥ 1 and any 0 &lt; r &lt; 1 , , where the infimum is over all estimators G n taking values in Ξ .

## 3.3 Practical Implications

There are two important practical implications for the design of a contaminated MoE model from our theoretical results.

1. Softmax gating is more sample-efficient than input-free gating. We observe that softmax gating yields faster convergence rates of prompt parameter estimation in contaminated MoE than input-free gating in [35]. In particular, when using input-free gating, Table 2 reveals that the rates for estimating expert parameters and variance depend on the convergence rate of the gating parameter to zero. By contrast, when using softmax gating, estimation rates for expert parameters and variance become significantly faster as the previous rate dependence disappears. Therefore, our theories encourage the use of softmax gating over input-free gating when tuning contaminated-MoE-based models.
2. Prompt models should have different expertise from pre-trained models. It can be seen from Table 2 that when the prompt model acquires overlapping knowledge with the pre-trained model (non-distinguishable setting), the convergence rates of parameter estimation are slower than when these models have distinct knowledge (distinguishable setting). Thus, our theories advocate using prompt models with different expertise from the pre-trained model.

Table 2: Comparison of parameter estimation rates in input-free-contaminated MoE [35] and softmaxcontaminated MoE (Ours). Below, we consider gating parameters exp( β ∗ 0 ) , expert parameters η ∗ , and variance ν ∗ . In addition, λ ∗ denotes the constant weight in input-free-contaminated MoE.

| Distinguishable Setting     | Distinguishable Setting                    | Distinguishable Setting                                |
|-----------------------------|--------------------------------------------|--------------------------------------------------------|
|                             | Gating parameters                          | Expert parameters and Variance                         |
| Input-free gating [35]      | ˜ O ( n - 1 / 2 )                          | ˜ O ( n - 1 / 2 ( λ ∗ ) - 1 )                          |
| Softmax gating (Ours)       | ˜ O ( n - 1 / 2 )                          | ˜ O ( n - 1 / 2 )                                      |
| Non-distinguishable Setting | Non-distinguishable Setting                | Non-distinguishable Setting                            |
|                             | Gating parameters                          | Expert parameters and Variance                         |
| Input-free gating [35]      | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 2 ) | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 1 ( λ ∗ ) - 1 ) |
| Softmax gating (Ours)       | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 2 ) | ˜ O ( n - 1 2 · ∥ (∆ η ∗ , ∆ ν ∗ ) ∥ - 1 )             |

## 4 Numerical Experiments

In this section, we present several numerical experiments to verify our theoretical findings.

Experimental setup. Recall that, in the distinguishable setting, the pre-trained model f 0 does not belong to the Gaussian density family. Thus, we let f 0 be the density of a Laplace distribution, with mean function h 0 ( x, η 0 ) = tanh( η ⊤ 0 x ) and variance ν 0 . Here, η 0 is a d -dimensional vector defined as e 1 := (1 , 0 , . . . , 0) , and ν 0 = 0 . 001 . Meanwhile, the prompt f is formulated as a Gaussian density,

with the same tanh mean function but a different parameter η ∗ -i.e., h ( x, η ∗ ) = tanh(( η ∗ ) ⊤ x ) -and variance ν ∗ .

On the other hand, in the non-distinguishable setting , both f and f 0 belong to the Gaussian density family, and h and h 0 are expert functions of the same form (albeit parameterized by different values of η 0 and η ∗ ). As in the previous case, we let the expert function be the tanh function: in the pre-trained model, the expert is h ( x, η 0 ) = tanh( η ⊤ 0 x ) , and in the prompt model, it is h ( x, η ∗ ) = tanh(( η ∗ ) ⊤ x ) .

Synthetic data generation. We create synthetic datasets following the model outlined in equation (1). Specifically, we generate data pairs { ( X i , Y i ) } n i =1 ∈ X ×Y ⊂ R d × R by first drawing each covariate X i independently from a standard Gaussian distribution, for i = 1 , . . . , n , and consistently set d = 8 across all trials. The responses Y i are drawn from the density p G ∗ ( y | x ) , where G ∗ = ( β ∗ , τ ∗ , η ∗ , ν ∗ ) :

(a) In the distinguishable setting, we let β ∗ = 1 / d · 1 d , τ ∗ = 1 , η ∗ = -e 1 = -η 0 and ν ∗ = ν 0 = 0 . 001 .

√

(b) In the non-distinguishable setting, we examine two cases to study the MLE convergence behavior as either η ∗ or ν ∗ varies with n : in the first, η ∗ is an O ( n -1 / 8 ) perturbation of η 0 with ν ∗ fixed at ν 0 ; in the second, η ∗ = -η 0 while ν ∗ is perturbed around ν 0 at the same rate. In detail, we set:

- (i) In the first case, β ∗ = 1 / √ d · 1 d , τ ∗ = 1 , η ∗ = e 1 (1 + n -1 / 8 ) = η 0 (1 + n -1 / 8 ) , and ν ∗ = ν 0 = 0 . 001 .
- (ii) In the second case, β ∗ = 1 / √ d · 1 d , τ ∗ = 1 , η ∗ = -e 1 = -η 0 , and ν ∗ = 0 . 001(1 + n -1 / 8 ) = ν 0 (1 + n -1 / 8 ) .

Training procedure. We conduct 40 experiments and, for each of them, consider 20 different sample sizes n , ranging from 10 3 to 10 5 . In computing the MLEs, the initialization is set relatively close to the true parameter values to mitigate potential optimization instabilities. We use an EM algorithm [16] to compute the MLE, employing an off-the-shelf BFGS optimizer for the M-step due to the absence of a universal closed-form solution. All the numerical experiments are performed on a MacBook Air with an Apple M4 chip.

Results. The experimental results are presented in Figure 1 and Figure 2, where the x-axis displays varying sample sizes n , and the y-axis shows the parameter estimation error. We now present a detailed analysis of the results shown in each figure:

(a) Figure 1 displays the results for Theorem 1. We observe that the convergence rates of ( ̂ β n , ̂ τ n , ̂ η n , ̂ ν n ) are O ( n -0 . 45 ) , O ( n -0 . 52 ) , O ( n -0 . 50 ) , O ( n -0 . 54 ) , respectively, aligning with the theoretical rates of order O ( n -1 / 2 ) in Theorem 1.

(b) On the other hand, Figure 2 illustrates the parameter estimation errors for the simulations conducted in the non-distinguishable setting as Theorem 3.

- (i) In the first case, η ∗ converges to η 0 at the rate of O ( n -1 / 8 ) , while ν ∗ remains fixed, Figure 2a shows that the convergence rate of exp( ̂ τ n ) to exp( τ ∗ ) is O ( n -0 . 23 ) , which is consistent with the expected rate of O ( n -1 / 4 ) . The convergence rates for ̂ β n , ̂ η n , and ̂ ν n are O ( n -0 . 37 ) , O ( n -0 . 39 ) , and O ( n -0 . 35 ) , respectively, all of which are approximately O ( n -0 . 375 ) , as they hinge on the vanishing rate O ( n -3 / 8 ) . These empirical rates are consistent with the theoretical rates in Theorem 3.
- (ii) In the alternative setting, η ∗ is held fixed, while ν ∗ converges to ν 0 at the rate of O ( n -1 / 8 ) . Figure 2b reveals that the convergence rate of exp( ̂ τ n ) to exp( τ ∗ ) is of order O ( n -0 . 22 ) , again close to O ( n -1 / 4 ) . Meanwhile, the MLEs ̂ β n , ̂ η n , and ̂ ν n still empirically converge to β ∗ , η ∗ , and ν ∗ at rates of O ( n -0 . 39 ) , O ( n -0 . 37 ) , and O ( n -0 . 39 ) , respectively, which align well with the theoretical rates ˜ O ( n -3 / 8 ) . This observation is consistent with the theoretical convergence rates in Theorem 3.

## 5 Conclusion

In this paper, we characterize the convergence behavior of maximum likelihood estimators for parameters in the softmax-contaminated MoE model formulated as a mixture of a frozen pre-trained

Figure 1: ( Distinguishable Setting: f 0 is the density of a Laplace distribution.) Log-log graphs depicting the empirical convergence rates of the MLE ( ̂ β n , ̂ τ n , ̂ η n , ̂ ν n ) to the ground-truth values ( β ∗ , τ ∗ , η ∗ , ν ∗ ) . The blue lines display the parameter estimation errors, while the orange dashed dotted lines are the fitted lines, highlighting the empirical MLE convergence rates.

<!-- image -->

Figure 2: ( Non-distinguishable Setting: f 0 is a Gaussian density.) Log-log graphs depicting the empirical convergence rates of the MLE ( ̂ β n , ̂ τ n , ̂ η n , ̂ ν n ) to the ground-truth values ( β ∗ , τ ∗ , η ∗ , ν ∗ ) . The blue lines display the parameter estimation errors, while the orange dashed dotted lines are the fitted lines, highlighting the empirical MLE convergence rates. Figure 2a and Figure 2b illustrates results for Case (i) and Case (ii), respectively.

<!-- image -->

model and a trainable prompt model. To capture the challenge in which the prompt model admits the same expertise as the pre-trained model, we propose a novel analytic distinguishability condition and divide our analysis based on that condition. When the distinguishability condition is satisfied, we obtain minimax optimal parameter estimation rates of parametric order in the sample size, which are faster than those under the contaminated MoE with input-free gating. Conversely, when the distinguishability condition is violated, these rates become substantially slower than the parametric rates as they hinge on the convergence rates of prompt parameters to pre-trained parameters.

Based on our theoretical analysis, we make the following observations. First, the softmax gating helps to improve the sample efficiency for estimating the parameters in the contaminated MoE compared to the input-free gating. Second, the convergence rates for parameter estimation will be negatively affected if the prompt model acquires overlapping knowledge with the pre-trained model, thereby increasing the sample complexity of parameter estimation.

In future work, we plan to consider a more challenging setting of the contaminated MoE where the pre-trained model is fine-tuned by multiple prompt models rather than a single prompt as in the current setting. Furthermore, we can also generalize the analysis to the scenario where the prompt models belong to various families of distributions, rather than being restricted to Gaussian distributions.

## References

- [1] J. S. O. Ceron, G. Sokar, T. Willi, C. Lyle, J. Farebrother, J. N. Foerster, G. K. Dziugaite, D. Precup, and P. S. Castro. Mixtures of experts unlock parameter scaling for deep RL. In Forty-first International Conference on Machine Learning , 2024. (Cited on page 1.)
- [2] Z. Chen, Y. Deng, Y. Wu, Q. Gu, and Y. Li. Towards understanding the mixture-of-experts layer in deep learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 23049-23062. Curran Associates, Inc., 2022. (Cited on page 1.)
- [3] Y. Chow, A. Tulepbergenov, O. Nachum, D. Gupta, M. Ryu, M. Ghavamzadeh, and C. Boutilier. A Mixture-of-Expert Approach to RL-based Dialogue Management. In The Eleventh International Conference on Learning Representations , 2023. (Cited on page 1.)
- [4] D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. K. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.04088 , 2024. (Cited on page 1.)
- [5] DeepSeek-AI et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 , 2024. (Cited on page 1.)
- [6] D. Do, H. Nguyen, K. Nguyen, and N. Ho. Minimax optimal rate for parameter estimation in multivariate deviated models. In Advances in Neural Information Processing Systems , volume 36, pages 30096-30133. Curran Associates, Inc., 2023. (Cited on page 4.)
- [7] S. Faria and G. Soromenho. Fitting mixtures of linear regressions. Journal of Statistical Computation and Simulation , 80(2):201-225, 2010. (Cited on page 1.)
- [8] W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research , 23(120):1-39, 2022. (Cited on page 1.)
- [9] S. Gadat, J. Kahn, C. Marteau, and C. Maugis-Rabusseau. Parameter recovery in two-component contamination mixtures: The lˆ2 strategy. 2020. (Cited on page 24.)
- [10] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024. (Cited on page 1.)
- [11] X. Han, H. Nguyen, C. Harris, N. Ho, and S. Saria. Fusemoe: Mixture-of-experts transformers for fleximodal fusion. In Advances in Neural Information Processing Systems , 2024. (Cited on page 1.)
- [12] N. Ho and X. Nguyen. On strong identifiability and convergence rates of parameter estimation in finite mixtures. Electronic Journal of Statistics , 10:271-307, 2016. (Cited on page 20.)
- [13] N. Ho, C.-Y. Yang, and M. I. Jordan. Convergence rates for Gaussian mixtures of experts. Journal of Machine Learning Research , 23(323):1-81, 2022. (Cited on pages 3 and 40.)
- [14] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton. Adaptive mixtures of local experts. Neural Computation , 3, 1991. (Cited on page 1.)
- [15] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mixtral of experts. arxiv preprint arxiv 2401.04088 , 2024. (Cited on page 1.)
- [16] M. I. Jordan and R. A. Jacobs. Hierarchical mixtures of experts and the EM algorithm. Neural Computation , 6:181-214, 1994. (Cited on pages 1 and 9.)

- [17] J. Kwon and C. Caramanis. EM Converges for a Mixture of Many Linear Regressions. In S. Chiappa and R. Calandra, editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics , volume 108 of Proceedings of Machine Learning Research , pages 1727-1736. PMLR, Aug. 2020. (Cited on page 1.)
- [18] M. Le, A. Nguyen, H. Nguyen, C. Nguyen, A. Tran, and N. Ho. On the expressiveness of visual prompt experts. arxiv preprint arxiv 2501.18936 , 2025. (Cited on page 1.)
- [19] M. Le, C. Nguyen, H. Nguyen, Q. Tran, T. Le, and N. Ho. Revisiting prefix-tuning: Statistical benefits of reparameterization among prompts. In The Thirteenth International Conference on Learning Representations , 2025. (Cited on page 1.)
- [20] M. Le, A. N. The, H. Nguyen, T. T. N. Vu, H. T. Pham, L. N. Van, and N. Ho. Mixture of experts meets prompt-based continual learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. (Cited on page 1.)
- [21] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In International Conference on Learning Representations , 2021. (Cited on page 1.)
- [22] H. Li, S. Lin, L. Duan, Y. Liang, and N. Shroff. Theory on mixture-of-experts in continual learning. In The Thirteenth International Conference on Learning Representations , 2025. (Cited on page 1.)
- [23] X. L. Li and P. Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , Online, Aug. 2021. Association for Computational Linguistics. (Cited on page 1.)
- [24] H. Liang, Z. Fan, R. Sarkar, Z. Jiang, T. Chen, K. Zou, Y. Cheng, C. Hao, and Z. Wang. M 3 ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with ModelAccelerator Co-design. In NeurIPS , 2022. (Cited on page 1.)
- [25] B. Lindsay. Mixture models: Theory, geometry and applications . In NSF-CBMS Regional Conference Series in Probability and Statistics. IMS, Hayward, CA., 1995. (Cited on page 1.)
- [26] E. F. Mendes and W. Jiang. Convergence rates for mixture-of-experts. arXiv preprint arxiv 1110.2058 , 2011. (Cited on page 3.)
- [27] H. Nguyen, P. Akbarian, T. Nguyen, and N. Ho. A general theory for softmax gating multinomial logistic mixture of experts. In Proceedings of the 41st International Conference on Machine Learning , pages 37617-37648, 2024. (Cited on page 1.)
- [28] H. Nguyen, X. Han, C. W. Harris, S. Saria, and N. Ho. On expert estimation in hierarchical mixture of experts: Beyond softmax gating functions. arxiv preprint arxiv 2410.02935 , 2024. (Cited on page 1.)
- [29] H. Nguyen, N. Ho, and A. Rinaldo. Sigmoid gating is more sample efficient than softmax gating in mixture of experts. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. (Cited on pages 3 and 4.)
- [30] H. Nguyen, K. Nguyen, and N. Ho. On parameter estimation in deviated Gaussian mixture of experts. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics , 2024. (Cited on page 3.)
- [31] H. Nguyen, T. Nguyen, and N. Ho. Demystifying softmax gating function in Gaussian mixture of experts. In Advances in Neural Information Processing Systems , 2023. (Cited on pages 3, 4, and 5.)
- [32] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. S. Pint, D. Keysers, and N. Houlsby. Scaling vision with sparse mixture of experts. In Advances in Neural Information Processing Systems , volume 34, pages 8583-8595. Curran Associates, Inc., 2021. (Cited on page 1.)

- [33] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In In International Conference on Learning Representations , 2017. (Cited on pages 1 and 3.)
- [34] S. van de Geer. Empirical Processes in M-estimation . Cambridge University Press, 2000. (Cited on pages 3, 38, and 39.)
- [35] F. Yan, H. Nguyen, D. Le, P. Akbarian, and N. Ho. Understanding expert structures on minimax parameter estimation in contaminated mixture of experts. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics , 2025. (Cited on pages 2, 3, 4, 5, and 8.)
- [36] Z. You, S. Feng, D. Su, and D. Yu. Speechmoe: Scaling to large acoustic models with dynamic routing mixture of experts. In Interspeech , 2021. (Cited on page 1.)
- [37] Z. You, S. Feng, D. Su, and D. Yu. Speechmoe2: Mixture-of-experts model with improved routing. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 7217-7221, 2022. (Cited on page 1.)
- [38] S. Yun, I. Choi, J. Peng, Y. Wu, J. Bao, Q. Zhang, J. Xin, Q. Long, and T. Chen. Flexmoe: Modeling arbitrary modality combination via the flexible mixture-of-experts. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. (Cited on page 1.)

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The paper's contributions and scope are reflected accurately in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In Section 5, we identify the limitations and present the future development of our analysis.

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

Justification: Each theoretical result contains the full set of assumptions. All the proofs are deferred to the appendices.

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

Justification: All the experimental details are provided in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often

one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: We use synthetic data for our experiments. We will consider releasing the code upon the acceptance of our work.

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

Justification: All the experimental details are provided in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The error bars are reported in Section 4.

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

Justification: The computer resource is reported in Section 4.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have read and followed all the NeurIPS code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Given the theoretical nature of the paper, we do not think there are any positive or negative societal impacts of the work performed.

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

Justification: This is a theoretical work, and we do not release any data or models that have a high risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use any existing assets.

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

Justification: We do not release any new assets in this work.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are not used as any original, important, or non-standard component in the development of the core methods in this paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Supplementary Material for 'On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts'

In this supplementary material, we provide the theoretical proofs omitted from the main text. Appendix A presents the proofs of our main results, including the theorems on convergence rates for parameter estimation and the minimax lower bounds stated in Section 3. Proofs of auxiliary results concerning the fundamental properties of the softmax-contaminated MoE model, introduced in Section 2, are deferred to Appendix B.

## A Proof of Main Results

In this section, we present the proofs of the MLE rate theorem and the minimax lower bound theorem from Section 3, covering both distinguishable and non-distinguishable settings.

## A.1 Proof of Theorem 1

We begin by proving Theorem 1 under the distinguishable setting.

Proof of Theorem 1. Let G = ( ¯ β, ¯ τ, ¯ η, ¯ ν ) , we need to demonstrate that

<!-- formula-not-decoded -->

Using the argument with Fatou's lemma as in Theorem 3.1, [12] , it is sufficient to show that

<!-- formula-not-decoded -->

Assume by contrary that the above claim is not true. Then, there exist two sequences G n = ( β n , τ n , η n , ν n ) and G ∗ ,n = ( β ∗ n , τ ∗ n , η ∗ n , ν ∗ n ) , such that when n tends to infinity, we get

<!-- formula-not-decoded -->

In this proof, we will take into account only the most challenging setting of ( β n , η n , ν n ) and ( β ∗ n , η ∗ n , ν ∗ n ) when they converge to the same limit point ( β ′ , η ′ , ν ′ ) , where ( β ′ , η ′ , ν ′ ) is not necessarily equal to ( ¯ β, ¯ η, ¯ ν ) .

Step 1: Density Decomposition. Subsequently, we consider Q n ( Y | X ) = [1 + exp(( β n ) ⊤ X + τ n )] · [ p G n ( Y | X ) -p G ∗ ,n ( Y | X )] , which can decomposed as

<!-- formula-not-decoded -->

Based on the first order Taylor expansion, I n and II n could be denoted as

<!-- formula-not-decoded -->

where ℓ 1 = α 1 , ℓ 2 = | α 2 | +2 α 3 , and

<!-- formula-not-decoded -->

for all ( ℓ 1 , ℓ 2 ) ∈ N d × N such that 1 ≤ 2 | ℓ 1 | + ℓ 2 ≤ 2 .

Similarly, II n can be expressed as:

<!-- formula-not-decoded -->

Here R p ( Y | X ) /D 1 ( G n , G ∗ ,n ) → 0 as n → ∞ , where R p ( X,Y ) , p ∈ [2] are Taylor remainders . Consequently, Q n can be expressed as:

<!-- formula-not-decoded -->

with coefficients T n ℓ 1 ,ℓ 2 and S n γ are defined for any 0 ≤ 2 | ℓ 1 | + ℓ 2 ≤ 2 , and 0 ≤ | γ | ≤ 1 as:

̸

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

̸

where Q n can be viewed as linear combinations of elements of the set H 1 defined as

<!-- formula-not-decoded -->

Step 2: Non-vanishing coefficients. In this step, we will use a contradiction argument to demonstrate that not all the coefficients in the set

<!-- formula-not-decoded -->

vanish as n → ∞ where D 1 n := D 1 ( G n , G ∗ ,n ) . Specifically, suppose that all these coefficients converge to zero, when n →∞ , then we get,

<!-- formula-not-decoded -->

Similarly, by analyzing the limits of T n ℓ 1 ,ℓ 2 /D 1 n s.t. 1 ≤ 2 | ℓ 1 | + ℓ 2 ≤ 2 , we conclude that:

<!-- formula-not-decoded -->

as n → ∞ for all u ∈ [ d ] , v ∈ [ q ] . Given that our parameter lies in a compact set, there exists a positive constant C such that | exp( τ ∗ n ) / exp( τ n ) | ≤ C . Thus, we have

<!-- formula-not-decoded -->

The limits imply that

<!-- formula-not-decoded -->

Combining the results in equations (16) and (17) with the formulation of D 1 n , we deduce that

<!-- formula-not-decoded -->

which is a contradiction. Thus, not all the coefficients in the set S 1 tend to 0 as n →∞ .

Step 3 - Application of Fatou's lemma. Let us denote by m n the maximum of the absolute values of those coefficients. It follows from the previous result that 1 /m n ̸→∞ . Then | T n ℓ 1 ,ℓ 2 | / ( m n D 1 n ) and | S n γ | / ( m n D 1 n ) remain bounded, we can consider subsequences of these terms, ensuring that: | T n ℓ 1 ,ℓ 2 | /m n D 1 n → η ℓ 1 ,ℓ 2 , | S n γ | /m n D 1 n → ω γ , as n →∞ for all 0 ≤ 2 | ℓ 1 | + ℓ 2 ≤ 2 , 0 ≤ | γ | ≤ 1 . Here, at least one among η ℓ 1 ,ℓ 2 ( j ) and ω γ ( j ) is different from zero. By applying the Fatou's lemma, we get

<!-- formula-not-decoded -->

Under the given assumption, the left-hand side of the equation (18) is zero. Consequently, the integrand on the right-hand side of the equation (18) must also be zero almost surely with respect to ( X,Y ) . This results in:

<!-- formula-not-decoded -->

for almost surely ( X,Y ) . Furthermore, by Lemma 1, the collection

<!-- formula-not-decoded -->

is linearly independent with respect to ( X,Y ) . Consequently, it follows that η ℓ 1 ,ℓ 2 = ω γ = 0 , for all 0 ≤ 2 | ℓ 1 | + ℓ 2 ≤ 2 , 0 ≤ | γ | ≤ 1 . But this contradicts that from the definition, at least one among η ℓ 1 ,ℓ 2 , ω γ is nonzero. Hence, we reach the desired conclusion.

Lemma 1. Suppose that f 0 is distinguishable with f , then the set W 1 defined in equation (19) is linearly independent w.r.t. ( X,Y ) .

Proof of Lemma 1. Recall the set

<!-- formula-not-decoded -->

and the density

<!-- formula-not-decoded -->

In words, p G ∗ is a convex combination (depending on X ) of

<!-- formula-not-decoded -->

Noting that the term in set W 1 can be divided as the density function or its first and second derivatives

<!-- formula-not-decoded -->

along with the factor involving only X .

## Step 1: Distinguishable property with respect to Y .

First, fix X . Suppose for contradiction that there exist real numbers c 0 , c 1 , c 2 , d (may depend on X ), not all zero, such that

<!-- formula-not-decoded -->

Note that ∂ 0 f ∂h 0 = f . Hence we have

<!-- formula-not-decoded -->

Since

<!-- formula-not-decoded -->

the above can be rewritten as

<!-- formula-not-decoded -->

Using the hypothesis about the distinguishable property of f 0 with respect to f as well as the Gaussian property of f , which implies ∂ 2 f/∂h 2 = 1 / 2 · ∂f/∂ν , we have

<!-- formula-not-decoded -->

̸

and simultaneously c 1 = c 2 = 0 . But ϕ ( X ) = 0 on a set of X -values of positive measure , so d = 0 . Plugging d = 0 into c 0 + d (1 -ϕ ( X )) = 0 yields c 0 = 0 . Hence c 0 = c 1 = c 2 = d = 0 . Since no nontrivial linear combination of { f, ∂f ∂σ , ∂ 2 f ∂σ 2 , p G ∗ } can vanish almost everywhere, these four functions are linearly independent when X is fixed. This completes the proof of step 1.

## Step 2: Distinguishable property with respect to X .

Let us consider coefficients appear in each density factor.

- Term related to p G ∗ ( Y | X ) : The factor appearing along with p G ∗ ( Y | X ) are exp(( β ∗ ) ⊤ X ) , and X ( i ) exp(( β ∗ ) ⊤ X ) , where 1 ≤ i ≤ d . Suppose there exists constants c, a 1 , . . . , a d such that

<!-- formula-not-decoded -->

This equation means that c + ∑ d i =1 a i X ( i ) = 0 , a.s. Given that X has non-vanish almost everywhere density function, this relation implies that c = 0 , a i = 0 , 1 ≤ i ≤ d .

- Terms related to f ( Y | h ( X,η ∗ ) , ν ∗ ) : The factors appearing along with f ( Y | h ( X,η ∗ ) , ν ∗ ) are exp(( β ∗ ) ⊤ X ) , and X ( i ) exp(( β ∗ ) ⊤ X ) , where 1 ≤ i ≤ d . The identical argument as in the case for p G ∗ ( Y | X ) also gives us the independency.
- Terms related to ∂f ∂h ( Y | h ( X,η ∗ ) , ν ∗ ) : The factors appearing along with p G ∗ ( Y | X ) are

<!-- formula-not-decoded -->

Suppose there exists constants a 1 , . . . , a d not all equal to zero such that

<!-- formula-not-decoded -->

This equation means that

<!-- formula-not-decoded -->

where a = ( a 1 , . . . , a d ) . This is a contradiction.

- Terms related to ∂ 2 f ∂h 2 ( Y | h ( X,η ∗ )) : There is only one such term is

<!-- formula-not-decoded -->

Its coefficient obviously vanishes from the independent property with respect to Y .

This completes the proof of Lemma 1.

## A.2 Proof of Theorem 2

As a first step in proving the minimax lower bounds for the distinguishable setting (Theorem 2), we define two distances:

<!-- formula-not-decoded -->

for any G 1 = ( β 1 , τ 1 , η 1 , ν 1 ) ∈ Ξ and G 2 = ( β 2 , τ 2 , η 2 , ν 2 ) ∈ Ξ . Obviously d 2 ( G 1 , G 2 ) is a proper distance. The structure for d 1 ( G 1 , G 2 ) tells us that it is not symmetric. Only when τ 1 = τ 2 = τ , d 1 ( G 1 , G 2 ) is symmetric. Also d 1 ( G 1 , G 2 ) still satisfies a weak triangle inequality:

<!-- formula-not-decoded -->

Therefore, we will apply the modified Le Cam method for nonsymmetric loss, as outlined in Lemma C.1 of [9], to handle this distance. For f satisfies all assumptions in Theorem 2, based on the Taylor expansion, we have the following results:

Lemma 2. Given f in Theorem 2, we denote

<!-- formula-not-decoded -->

we achieve for any r &lt; 1 that

<!-- formula-not-decoded -->

We will prove this lemma later.

Proof of Theorem 2. Denote G ∗ = ( β ∗ , τ ∗ , η ∗ , ν ∗ ) and assume r &lt; 1 . Given Lemma 2 part (i) , for any sufficiently small ϵ &gt; 0 , there exists G ′ ∗ = ( β ∗ 1 , τ ∗ , η ∗ 1 , ν ∗ 1 ) such that d 1 ( G ∗ , G ′ ∗ ) = d 1 ( G ′ ∗ , G ∗ ) = ϵ , there exists a constant C 0 , s.t.

<!-- formula-not-decoded -->

Now we will denote p n G ∗ as the density of the n -i.i.d. sample ( X 1 , Y 1 ) , · · · , ( X n , Y n ) . Lemma C.1 in [9] tells us that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Last inequality is from the definition of the Total Variation distance and Hellinger distance and equation (20). Let ϵ 2 r = 1 C 2 0 n , then for any r &lt; 1 we have

<!-- formula-not-decoded -->

where c 1 is some positive constant. Following a similar reasoning and using Lemma 2 part (ii) , we will obtain

<!-- formula-not-decoded -->

for some positive constant c 2 . Consequently, we establish all of the results for Theorem 2.

Proof of Lemma 2 (i) . Consider two sequences

<!-- formula-not-decoded -->

with the same τ n . By the contaminated MoE model definition, we have

<!-- formula-not-decoded -->

for j = 1 , 2 . Since ( τ n , β j,n ) lie in a compact set, and both f 0 and f are non-negative. Hence, the squared Hellinger distance satisfies

<!-- formula-not-decoded -->

for some constants C, C ′ depending on the compactness bounds.

Consider the Taylor expansion of the map

<!-- formula-not-decoded -->

at the point ( β 2 ,n , η 2 ,n , ν 2 ,n ) , expanded up to first order with integral remainder. Let α = ( α 1 , α 2 , α 3 ) denote a multi-index where α 1 ∈ N d , α 2 ∈ N q , and α 3 ∈ N index components of β , η , and ν , respectively. Then we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So it follows that

̸

<!-- formula-not-decoded -->

since τ n lies in a compact set. This establishes part (i) of the lemma.

Proof of Lemma 2 (ii). We consider two sequences

<!-- formula-not-decoded -->

with different τ 1 ,n = τ 2 ,n but the same ( β n , η n , ν n ) .

Using the contaminated MoE definition, the difference in conditional densities is:

<!-- formula-not-decoded -->

By the standard bound for squared Hellinger distance,

<!-- formula-not-decoded -->

Since ( β n , η n , ν n ) lie in a compact set, and both f and f 0 are bounded away from zero, we have p S ′ 2 ,n ( Y | X ) ≥ c &gt; 0 . So the denominator is lower bounded.

Then there exists a constant C ′ such that:

<!-- formula-not-decoded -->

Now recall the definition of the distance:

<!-- formula-not-decoded -->

So we conclude:

<!-- formula-not-decoded -->

as long as e τ 1 ,n -e τ 2 ,n → 0 , and r &lt; 1 . Hence,

<!-- formula-not-decoded -->

which proves part (ii).

## A.3 Proof of Theorem 3

We proceed to prove Theorem 3 for the non-distinguishable setting.

Proof. Let G = ( ¯ β, ¯ τ, ¯ η, ¯ ν ) and (¯ η, ¯ ν ) can be identical to ( η 0 , ν 0 ) . Then, we will show that

̸

- (i) When ( η 0 , ν 0 ) = (¯ η, ¯ ν ) ,

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Part (i) can be proved by using the same arguments as in the proof A.1. Thus, we will consider only part (ii) in this section, specifically the most challenging setting that ( η 0 , ν 0 ) = (¯ η, ¯ ν ) . Under this assumption, we know that h 0 and h are the same expert function, s.t. f 0 ( Y | h 0 ( X,η 0 ) , ν 0 ) = f ( Y | h ( X,η 0 ) , ν 0 ) for almost surely ( X,Y ) ∈ X × Y . Assume that the above claim in equation (21) does not hold, then there exist two sequences G n = ( β n , τ n , η n , ν n ) and G ∗ ,n = ( β ∗ n , τ ∗ n , η ∗ n , ν ∗ n ) , such that

<!-- formula-not-decoded -->

We now analyze the limiting behavior of the sequences ( λ n , G n ) and ( λ ∗ n , G ∗ n ) as they approach ( ¯ λ, ¯ G ) . In particular, we distinguish between three asymptotic regimes based on how the expert parameters ς n = ( η n , ν n ) and ς ∗ n = ( η ∗ n , ν ∗ n ) converge.

̸

First, it may occur that both ς n and ς ∗ n converge to the same limit ς 0 = ( η 0 , ν 0 ) . Alternatively, both sequences may converge to a common limit ς ′ = ς 0 , which is distinct from the true expert. Finally, it is also possible that one sequence converges to ς 0 while the other converges to a different point ς ′ = ς 0 .

In the following, we analyze each of these cases and demonstrate that in all scenarios, the assumption that the normalized difference vanishes leads to a contradiction when f 0 = f .

## Case 1:

At first we consider that ( η n , ν n ) and ( η ∗ n , ν ∗ n ) share the same limit of ( η 0 , ν 0 ) . Without loss of generality, we can suppose that τ ∗ n ≥ τ n . Subsequently, we consider W n := [ p G n ( Y | X ) -p G ∗ ,n ( Y | X )] · [1 + exp(( β ∗ n ) ⊤ X + τ ∗ n )] · [1 + exp(( β n ) ⊤ X + τ n )] , which can decomposed as

<!-- formula-not-decoded -->

where we denote g ( Y | X ; β, η, ν ) = e ( X ; β ) f ( Y | X ; η, ν ) = exp ( β ⊤ X ) f ( Y | h ( X,η ) , ν ) .

We expand around the reference parameters β ∗ n , η ∗ n , ν ∗ n , where the parameter differences are given by ∆ η n = η n -η 0 , ∆ ν n = ν n -ν 0 , and ∆ η ∗ n = η ∗ n -η 0 , ∆ ν ∗ n = ν ∗ n -ν 0 . Applying a second-order Taylor expansion, then we obtain:

<!-- formula-not-decoded -->

where R 1 ( X,Y ) is the remainder term containing higher-order terms, and the second equality is due to ∂f ∂ν = 1 2 ∂ 2 f ∂h 2 . Similarly, we will have that

<!-- formula-not-decoded -->

Then, grouping the terms according to the order of derivative γ := | α 2 | +2 α 3 and the monomial degree ζ := | α 1 | , we can rewrite the expansion in the compact form:

<!-- formula-not-decoded -->

where each coefficient I n,γ,ζ ( X ) depends on the parameter differences and derivatives of h with respect to η . More specifically we have that

<!-- formula-not-decoded -->

Similarly, we can rewrite II n in the same fashion as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

̸

In the same way, we can rewrite III n in the same fashion as follows, here the difference for β ∗ n is zero, so all the coefficients with ζ = 0 is zero, but in order for the alignment of the expression, we will still express III n as follows

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Nowweconsider IV n = exp ( ( β ∗ n + β n ) ⊤ X + τ ∗ n + τ n ) · [ f ( Y | σ ( X,η n ) , ν n ) -f ( Y | σ ( X,η ∗ n ) , ν ∗ n )] , which is equivalent to

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Then we could conclude that

<!-- formula-not-decoded -->

Therefore, we can view the quantity W n /D 2 ( G n , G ∗ ,n )) as a linear combination of elements of the set L∪K , and L = ∪ 4 γ =0 ∪ 2 ζ =0 L γ,ζ , K = ∪ 4 γ =1 K γ , where

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume by contrary that all the coefficients of these elements vanish when n →∞ . Looking at the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) X ∂f ∂h ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all w ∈ [ d ] , u ∈ [ q ]

<!-- formula-not-decoded -->

Looking at the coefficients of X ∂ 2 f ∂h 2 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all w ∈ [ d ]

<!-- formula-not-decoded -->

Looking at the coefficients of ∂ 2 h ∂η ( u ) ∂η ( v ) ( X,η ∗ n ) ∂f ∂h ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u, v ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) ∂f ∂h ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of ∂ 2 f ∂h 2 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get

<!-- formula-not-decoded -->

Looking at the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) ∂h ∂η ( v ) ( X,η ∗ n ) ∂ 2 f ∂h 2 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u, v ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) ∂ 3 f ∂h 3 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of ∂ 4 f ∂h 4 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get

<!-- formula-not-decoded -->

Looking at the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) exp(( β n ) ⊤ X ) ∂f ∂h ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of ∂ 2 h ∂η ( u ) ∂η ( v ) ( X,η ∗ n ) exp(( β n ) ⊤ X ) ∂f ∂h ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u, v ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of exp(( β n ) ⊤ X ) ∂ 2 f ∂h 2 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get

<!-- formula-not-decoded -->

Looking at the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) ∂h ∂η ( v ) ( X,η ∗ n ) exp(( β n ) ⊤ X ) ∂ 2 f ∂h 2 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u, v ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) exp(( β n ) ⊤ X ) ∂ 3 f ∂h 3 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u ∈ [ q ] ,

<!-- formula-not-decoded -->

Looking at the coefficients of exp(( β n ) ⊤ X ) ∂ 4 f ∂h 4 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we get for all u ∈ [ q ] ,

<!-- formula-not-decoded -->

Now, combining (23) and (24), recall that all the gating parameters are in compact sets, and applying the Cauchy-Schwarz inequality followed by summation over coordinates, we got that

<!-- formula-not-decoded -->

While it is intuitive that the similar result holds for ∥ (∆ η ∗ n , ∆ ν ∗ n ) ∥ , a slightly tricky handle should be employed here. Suppose that

<!-- formula-not-decoded -->

By combining this assumption with equation (23), we have there are at least one coordinate u such that | (∆ η ∗ n ) ( u ) / (∆ η n ) ( u ) |→∞ , which implies that (∆ η ∗ n ) / (∆ η ∗ n -∆ η n ) ( u ) → 1 . Thus, by multiplying equation (31) with (∆ η ∗ n ) / (∆ η ∗ n -∆ η n ) ( u ) → 1 , we have

<!-- formula-not-decoded -->

Also noting that ∥ β n -β ∗ n ∥ is bounded as the parameters belongs to a compact set, we have

<!-- formula-not-decoded -->

which is a contradiction here. Thus, we have

<!-- formula-not-decoded -->

Similarly, also by combining equation (24) and (33), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a result, we have

In a similar manner, by considering equations (31) through (36), we obtain that

<!-- formula-not-decoded -->

Let u = v in the first equation in equation (25), we achieve that for all u ∈ [ d ] ,

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

We also have each term inside equation (43) is non-negative, thus

<!-- formula-not-decoded -->

Applying the AM-GM inequality, we have for all u, v ∈ [ d ] ,

<!-- formula-not-decoded -->

Next, by considering the coefficients of ∂h ∂η ( u ) ( X,η ∗ n ) ∂f ∂h ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , and

∂ 2 f ∂h 2 ( Y | h ( X,η ∗ n ) , ν ∗ n )) exp(( β ∗ n ) ⊤ X ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Noting that for u, v ∈ [ d ] ,

<!-- formula-not-decoded -->

Thus, from equation (45) and equation (46), we achieve that for u, v ∈ [ d ] ,

<!-- formula-not-decoded -->

By using the same arguments we will derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By using the same arguments to derive equation (42), equation (44) and equation (45), we can point out that

<!-- formula-not-decoded -->

Collecting results in equation (37), (40) and (41), and equations (44) to (50), we obtain that

<!-- formula-not-decoded -->

which is a contradiction.

Therefore, not all the coefficients in the representation of W n /D 2 ( G n , G ∗ ,n ) tend to 0 as n →∞ . Let us denote by m n the maximum of the absolute values of those coefficients. Based on the previous result, 1 /m n ̸→∞ . Additionally, we define

<!-- formula-not-decoded -->

when n → ∞ for all w ∈ [ d ] , u, v ∈ [ q ] . Note that at least one among α γζ,wuv , β γζ,wuv and ρ γ,uv , π γ,uv where γ ∈ [4] , ζ ∈ { 0 , 1 } must be different from zero. By applying the Fatou's lemma, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is worth noting that for almost surely ( X,Y ) , the set L ∪ K is linearly independent under nondistinguishable setting , which leads to the fact that E τζ ( X ) = K τ ( X ) = 0 for almost surely X for any τ ∈ [4] , ζ ∈ { 0 , 1 } .

Similar to the proof of Theorem 1, and recalling that the experts are strongly identifiable, we conclude that all the coefficients in Equation (51) must be zero for all w,u,v .

This contradicts the fact that not all coefficients vanish. Thus, we obtain the conclusion for this case.

## Case 2:

In this case, we consider that ( η n , ν n ) and ( η ∗ n , ν ∗ n ) share the same limit, but different from ( η 0 , ν 0 ) . From the formulation of the metric D 1 in the proof A.1, it is clear that D 2 ≲ D 1 . Therefore, we get W n ( X,Y ) /D 1 ( G n , G ∗ ,n ) → 0 as n → ∞ . Noting that ( η n , ν n ) and ( η ∗ n , ν ∗ n ) share the limit ( η ∗ , ν ∗ ) = ( η 0 , ν 0 ) , we have f 0 = f ( Y | h ( X,η 0 ) , ν 0 ) and f ( Y | h ( X,η ∗ ) , ν ∗ ) satisfying f 0 and f independent up to second order as in Lemma 1. Thus, we can process in a similar way as in Theorem 1 to draw a contradiction.

## Case 3:

Lastly, we consider that one of G n or G ∗ n converges to G 0 , while the other converges to G ′ = G 0 . Without loss of generality, suppose that G n → G ′ and G ∗ n → G 0 . By passing through the limit for

<!-- formula-not-decoded -->

̸

̸

noting that

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

This equation implies that

<!-- formula-not-decoded -->

which further implies that

<!-- formula-not-decoded -->

and hence

<!-- formula-not-decoded -->

̸

This equation means that G ′ = G 0 , which is a contradiction.

̸

## A.4 Proof of Theorem 4

In what follows, we present the proof of Theorem 4 for the non-distinguishable setting.

Proof of Theorem 4. The proof follows similar steps to the arguments in the previous two sections. Concretely, define for S 1 = ( τ 1 , β 1 , η 1 , ν 1 ) , S 2 = ( τ 2 , β 2 , η 2 , ν 2 ) :

<!-- formula-not-decoded -->

It is straightforward that d ′ and d ′′ satisfy the weak triangle inequality. Following the same schema as in Lemma 2, we can demonstrate two subsequent results for any r &gt; 1 :

- (i) Two sequences can be found

<!-- formula-not-decoded -->

such that d ′ ( S 1 ,n , S 2 ,n ) → 0 and E X [ h H ( p S 1 ,n ( ·| X ) , p S 2 ,n ( ·| X ))] /d r ′ ( S 1 ,n , S 2 ,n ) → 0 as n →∞ .

- (ii) Two sequences can be found

<!-- formula-not-decoded -->

such that d ′′ ( S 1 ,n , S 2 ,n ) → 0 and E X [ h H ( p S 1 ,n ( ·| X ) , p S 2 ,n ( ·| X ))] /d r ′′ ( S 1 ,n , S 2 ,n ) → 0 as n →∞ .

We can omit the justification for the above results as it can follow a similar approach as in Lemma 2. This leads to the conclusion of the theorem.

## B Proof of Auxiliary Results

## B.1 Proof of Proposition 1

Proof. Fix an arbitrary x ∈ X and abbreviate

<!-- formula-not-decoded -->

Because f is Gaussian in its argument, there exist µ 1 , µ 2 ∈ R and σ 2 1 , σ 2 2 &gt; 0 such that g j ( y ) = 1 √ 2 πσ 2 j exp ( -( y -µ j ) 2 / (2 σ 2 j ) ) for j = 1 , 2 .

Set

<!-- formula-not-decoded -->

With these notations the assumed identity becomes

<!-- formula-not-decoded -->

1. b 0 ( x ) = 0 . Because g 0 is not Gaussian by assumption, while g 1 , g 2 , H 1 , H 2 all belong to the finite-dimensional linear span G := span { y ↦→ g 1 ( y ) , y ↦→ ( y -µ 2 ) k g 2 ( y ) : k = 0 , 1 , 2 } , we have g 0 / ∈ G . Hence the only way (52) can hold on a set of positive measure is with b 0 ( x ) = 0 .

2. Linear independence inside G . Divide (52) (now with b 0 ( x ) = 0 ) by g 2 ( y ) ; we obtain the polynomial identity

<!-- formula-not-decoded -->

The ratio g 1 /g 2 is the analytic (non-polynomial) function

<!-- formula-not-decoded -->

̸

̸

̸

with K = 0 . Since µ 1 = µ 2 or σ 2 1 = σ 2 2 , this exponential term cannot be expressed as a quadratic polynomial in y . Consequently the set of functions { g 1 /g 2 , 1 , y -µ 2 , ( y -µ 2 ) 2 } is linearly independent on any interval. Hence every coefficient in the polynomial identity must vanish:

<!-- formula-not-decoded -->

3. Conclusion. We have shown that b 0 ( x ) = b 1 ( x ) = c 0 ( x ) = c 1 ( x ) = c 2 ( x ) = 0 for the fixed x . Because the same argument works for almost every x ∈ X , all coefficients vanish almost surely. Thus the unified distinguishability condition of Definition 1 is satisfied, completing the proof.

## B.2 Proof of Proposition 2

Proof. Write the two (single-expert) conditional densities

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume the identifiability equality p G ( y | x ) = p G ′ ( y | x ) holds for almost every ( x, y ) ∈ X × Y . Subtracting the two representations gives

<!-- formula-not-decoded -->

̸

Step 1. If λ ( x ) = λ ′ ( x ) . Suppose on a set of positive x -measure, λ ( x ) = λ ′ ( x ) . Divide (53) by λ ( x ) -λ ′ ( x ) ; then for those x

̸

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Since f is distinguishable from f 0 , the only possibility is b ( x ) = c ( x ) = 0 , hence λ ( x ) = λ ′ ( x ) a.e.-contradiction. Therefore

<!-- formula-not-decoded -->

Because the soft-max map ( β, τ ) ↦→ λ ( · ) is injective, we conclude

<!-- formula-not-decoded -->

Step 2. Equality of expert parameters. With λ ( x ) = λ ′ ( x ) , equation (53) reduces to

<!-- formula-not-decoded -->

̸

Definition 1 forces the situation ( η, ν ) = ( η ′ , ν ′ ) impossible. Hence the only consistent solution is ( η, ν ) = ( η ′ , ν ′ ) .

Step 3. Conclusion. We have shown β = β ′ , τ = τ ′ , η = η ′ , and ν = ν ′ ; hence G = G ′ .

## B.3 Proof of Proposition 3

We begin by introducing several standard notations used throughout this proof. Let ( P , d ) be a metric space, where d is a metric on P . An ϵ -net of ( P , d ) is a collection of balls of radius ϵ whose union covers P . The covering number N ( ϵ, P , d ) denotes the minimal cardinality of such a covering, and the entropy number is defined as H ( ϵ, P , d ) := log N ( ϵ, P , d ) .

The bracketing number N B ( ϵ, P , d ) is the minimal number of pairs { ( f i , f i ) } n i =1 such that f i &lt; f i , d ( f i , f i ) &lt; ϵ , and P is covered by the union of the brackets. The corresponding bracketing entropy is denoted by H B ( ϵ, P , d ) := log N B ( ϵ, P , d ) .

When P is a family of densities, we take d to be the L 2 ( m ) distance, where m denotes the Lebesgue measure.

In particular, let P (Ξ) := { p λ : λ ∈ Ξ } , and define the symmetrized density ¯ p λ := 1 2 ( p ∗ + p λ ) , where p ∗ denotes the true density. We then define the following sets: P (Ξ) := { ¯ p λ : λ ∈ Ξ } and P 1 / 2 (Ξ) := { ¯ p 1 / 2 λ : ¯ p λ ∈ P (Ξ) } . To study convergence rates, we consider the localized version of the symmetrized class: P 1 / 2 (Ξ , ϵ ) := { ¯ p 1 / 2 λ ∈ P 1 / 2 (Ξ) : d H (¯ p λ , p ∗ ) ≤ ϵ } , where d H ( · , · ) denotes the Hellinger distance. Then we assess the complexity of this class via the bracketing entropy integral defined in [34]: J B ( ϵ, P 1 / 2 (Ξ , ϵ ) , m ) := ∫ ϵ ϵ 2 / 2 13 √ H B ( u, P 1 / 2 (Ξ , ϵ ) , m ) du ∨ ϵ, where a ∨ b := max { a, b } . For brevity, we may omit the dependence on m when it is clear from context.

For the proof at first we consider a general lemma that provides the desired convergence rate, provided that a bracketing entropy condition is satisfied.

Lemma 3. Assume the following assumption hold: Given a universal constant J &gt; 0 , there exists N &gt; 0 , possibly depending on Ξ , such that for all n ≥ N and all ϵ &gt; (log( n ) /n ) 1 / 2 , we have

<!-- formula-not-decoded -->

Then, there exists a constant C &gt; 0 depending only on Ξ such that for all n ≥ 1 ,

<!-- formula-not-decoded -->

This lemma indicates that it suffices to verify the entropy condition in Equation (54) in order to obtain the convergence rate. However, this condition is often technically difficult to establish directly. As a workaround, we may instead prove the following sufficient condition:

Lemma 4. If the distribution satisfies

<!-- formula-not-decoded -->

it will meet the assumption in Equation (54).

Although we have simplified the condition in Equation (54) to Equation (55), verifying Equation (55) is still nontrivial. Fortunately, for the contaminated model defined in Equation (1),

<!-- formula-not-decoded -->

we assume that f 0 is bounded with light tails and that f is a univariate Gaussian density. Under these assumptions, we can verify Equation (55) via the following lemma:

Lemma 5. Let Γ be a compact subsets of R d × R and Θ be a bounded subsets of R q × R + , f is a univariate Gaussian density and f 0 is bounded with tail E X ( -log f 0 ( Y | h ( X,η 0 ) , ν 0 )) ≳ Y q for almost surely Y ∈ Y for some q &gt; 0 . Then, for any 0 &lt; ε &lt; 1 2 , the following results hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above results, we obtain the desired conclusion for Theorem 3.

Now we will prove Lemma 3, Lemma 4 and Lemma 5 in order. At first we need to introduce another Lemma 6 before we prove Lemma 3. Lemma 6 is Theorem 5.11 in [34] and its proof can also be found in [34].

Lemma 6. Let R &gt; 0 , k ≥ 1 and G is a subset in Ξ where G ∗ ∈ G ⊂ Ξ . Given C 1 &lt; ∞ , for all C sufficiently large, and for n ∈ N and t &gt; 0 is in the following range

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then we will have

<!-- formula-not-decoded -->

Proof of Lemma 3. Firstly, by Lemma 4.1 and 4.2 in [34], we have

<!-- formula-not-decoded -->

here µ n ( ̂ G n ) is an empirical process defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S is a smallest number such that 2 S δ/ 4 &gt; 1 .

Now we will use Lemma 6: choose R = 2 s +1 δ, C 1 = 15 and t = √ n 2 2 s ( δ/ 4) 2 . We can confirm that condition (i) in Lemma 3 is met since 2 s -1 δ/ 4 ≤ 1 for all s ≤ S . For the condition (ii), it is still satisfied since

<!-- formula-not-decoded -->

Now since the two conditions in Lemma 6 are all satisfied, we could conclude that

<!-- formula-not-decoded -->

here constant c is a large constant that does not depend on G ∗ . Now we could derive the bound on supremum of expectation:

<!-- formula-not-decoded -->

here ˜ c is independent from G ∗ and δ n := √ log n/n . So we can conclude that

<!-- formula-not-decoded -->

Proof of Lemma 4. Because P 1 / 2 (Ξ , δ ) ⊂ P 1 / 2 (Ξ) and from the definition of Hellinger distance, we have

<!-- formula-not-decoded -->

Now, using the fact that for densities f ∗ , f 1 , f 2 , we have h 2 ( f 1 + f ∗ 2 , f 2 + f ∗ 2 ) ≤ h 2 ( f 1 ,f 2 ) 2 , it is easy to verify that H B ( δ/ √ 2 , P (Ξ) , d H ) ≤ H B ( δ, P (Ξ) , d H ) . Hence, if equation (55) holds true, then

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

Proof of Lemma 5. Proof for (i): Let E ϵ ( S ) denote an ϵ -net of a set S under the ∥ · ∥ ∞ norm. Then log |E ϵ ( S ) | = log N ( ϵ, S, ∥ · ∥ ∞ ) .

Let P (Θ) := { p Υ : Υ ∈ Θ } , where p Υ ( Y | X ) := f ( Y | h ( X,η ) , ν ) . By Lemma 6 in [13], we have log N ( ϵ, P (Θ) , ∥ · ∥ ∞ ) ≲ log(1 /ϵ ) .

We now consider the contaminated model p Υ as a composition of smooth components indexed by ( β, τ, η, ν ) ∈ Ξ := Γ × Θ , where Γ ⊂ R d +1 and Θ ⊂ R q × R + are compact.

Since σ ( β ⊤ X + τ ) := exp( β ⊤ X + τ ) / (1+exp( β ⊤ X + τ )) is infinitely differentiable and Lipschitz over compact Γ , it follows that for any λ = ( β, τ ) ∈ Γ , there exists ˜ λ = ( ˜ β, ˜ τ ) ∈ E ϵ (Γ) such that

<!-- formula-not-decoded -->

Likewise, for any Υ = ( η, ν ) ∈ Θ , there exists ˜ Υ ∈ E ϵ (Θ) such that

∥

p

Υ

-

p

Υ

˜

∥

∞

≤

ϵ.

Now, consider the difference

<!-- formula-not-decoded -->

so that by the triangle inequality and boundedness of f 0 and f ,

<!-- formula-not-decoded -->

Hence, the covering number of P (Ξ) satisfies

<!-- formula-not-decoded -->

Proof for (ii): First, let η ≤ ε be a positive number, which will be chosen later. We consider f is the density function of an univariate Gaussian distribution, so f is light tail: for any | Y | ≥ 2 a and X ∈ X ,

<!-- formula-not-decoded -->

Also f 0 is bounded with tail log f 0 ( Y | h ( X,η 0 ) , ν 0 ) ≲ -Y q and f 0 ( Y | h ( X,η 0 )) , ν 0 ) ≤ M for almost surely Y ∈ Y for some M,q &gt; 0 . Now let q = min { p, 2 } and C 2 = max { M, 1 / √ 2 πℓ } , we will have

<!-- formula-not-decoded -->

here C 1 is a positive constant depending on ℓ and f 0 . Moreover H ( X,Y ) is an envelope of P (Ξ) . Next, let g 1 , . . . , g N represent an η -net over P k (Ξ) . Then, we construct the brackets [ p L i ( X,Y ) , p U i ( X,Y )] as follows:

<!-- formula-not-decoded -->

for i = 1 , · · · , N . As a result, P k (Ξ) ⊂ ⋃ N i =1 [ p L i ( X,Y ) , p U i ( X,Y )] and p U i ( X,Y ) -p L i ( X,Y ) ≤ min { 2 η, H ( X,Y ) } . Consequently,

<!-- formula-not-decoded -->

This shows that

Setting η = ϵ/c , we find

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since h 2 ≤ ∥ · ∥ 1 holds between the Hellinger distance and the total variation distance, we conclude the bracketing entropy bound.
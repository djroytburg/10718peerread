## Rescaled Influence Functions: Accurate Data Attribution in High Dimension

Ittai Rubinstein EECS and CSAIL MIT Cambridge, MA

ittair@mit.edu

Samuel B. Hopkins EECS and CSAIL MIT Cambridge, MA samhop@mit.edu

## Abstract

How does the training data affect a model's behavior? This is the question we seek to answer with data attribution . The leading practical approaches to data attribution are based on influence functions (IF). IFs utilize a first-order Taylor approximation to efficiently predict the effect of removing a set of samples from the training set without retraining the model, and are used in a wide variety of machine learning applications. However, especially in the high-dimensional regime (# params ≥ Ω( # samples ) ), they are often imprecise and tend to underestimate the effect of sample removals, even for simple models such as logistic regression. We present rescaled influence functions (RIF), a tool for data attribution which can be used as a dropin replacement for influence functions, with little computational overhead but significant improvement in accuracy. We compare IF and RIF on a range of realworld datasets, showing that RIFs offer significantly better predictions in practice, and present a theoretical analysis explaining this improvement. Finally, we present a simple class of data poisoning attacks that would fool IF-based detections but would be detected by RIF.

## 1 Introduction

Data attribution aims to explain the behavior of a machine learning model in terms of its training data. If θ is a model trained on a dataset { ( x i , y i ) } i ∈ [ n ] , the fundamental algorithmic task in data attribution is to answer the question:

LeaveT -Out Effect: How would θ have been different if some subset T ⊆ [ n ] of the training set had been missing?

The ability to quickly and accurately predict a leaveT -out (LTO) effect, or to search for subsets producing a large leave-out effect, unlocks extensive capabilities from classical statistical inference to modern machine learning. For example, the jackknife, leavek -out cross-validation, and bootstrap are all widely used to quantify uncertainty and estimate generalization error or confidence intervals, and all rely on the ability to quickly estimate LTO effects [Efr92, GSL + 19, Jae72]. Machine learning has seen an explosion of applications of data attribution, for dataset curation [KL17, KATL19], explainability [KATL19, GBA + 23], crafting and detection of data poisoning attacks [EIC + 25, KSL22, SS19], machine unlearning [SAKS21, GGHVDM19, IASCZ21], credit attribution [JDW + 19, GZ19], bias detection [BAHAZ19], and more.

Ascertaining the ground truth leaveT -out effect in general requires a full retrain of a model for each T of interest, which is computationally intractable in all but the simplest settings. Consequently, approximations to the leaveT -out effect are widely used. Key desiderata for such approximations

are (1) accuracy , (2) computational efficiency even for large-scale models, and (3) additivity : the predicted effect of removing T should be the sum of predicted effects of removing each element of T individually. Additivity enables another important capability: search for the subset T of a given size with the greatest predicted effect according to a given metric, by taking the k training data points with largest predicted leave-one-out (LOO) effects [BGM20, IPE + 22, HBN + 24].

Influence functions (IF) [Ham74] are by far the most widely used and studied data attribution method. The IF is a first-order approximation to the change in model parameters when infinitesimally downweighting an individual sample. IF approximations are well studied in classical, under-parameterized settings, where they are typically accurate and enjoy solid theoretical foundations [GSL + 19]. But, despite widespread adoption for data attribution in high-dimensional/overparameterized models, IF's accuracy in the high-dimensional setting is comparatively poor. Empirical studies show that IFs often underestimate the true magnitude of parameter changes, leading to potentially misleading conclusions about data importance or model robustness [BPF21, KL17]. And, existing theoretical analyses justifying IF approximations break down for overparametrized models. But, thus far, more accurate alternatives to IFs have proved too computationally expensive to be practical.

We study a simple and fast-to-compute modification of the influence function, which we term the rescaled influence function (RIF). RIFs improve accuracy by incorporating a limited amount of higher-order information about the change in model parameters from sample removal, but retain the additivity and in many settings also the computational efficiency of IFs. We show via experiments and theoretical analysis that RIFs are accurate for data attribution in overparameterized models where IFs struggle. Like IFs, RIFs are model and task agnostic, meaning that they can be applied to any empirical risk minimization-based training method with smooth losses, and they can estimate the leaveT -out effect according to any (smooth) measure of change to model parameters. We therefore advocate using RIFs as a drop-in replacement for IFs across data attribution applications.

Organization In Section 1.1, we introduce RIFs formally. Section 2 presents our experimental results, and Section 3 presents our theoretical analysis of RIF. We discuss context and conclusions in Sections 4 and 5

## 1.1 Influence Functions, Newton Steps, and Rescaled Influence Functions

We now introduce the rescaled influence function formally. Suppose that { ( x i , y i ) } i ∈ [ n ] is a training data set, Θ ⊆ R d is a class of models, and ℓ ( x, y, θ ) is a twice-differentiable loss function; ℓ may include a regularizer. For simplicity, we imagine that ℓ is convex, although the definition of RIFs can be extended to the non-convex case. Let ˆ θ = arg min θ ∈ Θ ∑ i ≤ n ℓ ( x i , y i , θ ) be the empirical loss minimizer (or, in the non-convex setting, any local minimum of the empirical loss).

Influence Functions The influence function IF i ∈ R d associated to the i -th training sample is a first-order estimate of the effect of dropping that sample. 1 Introducing a weight w i ∈ [0 , 1] associated to each sample i and allowing ˆ θ to depend on w via ˆ θ ( w ) = arg min θ ∈ Θ ∑ i ≤ n w i · ℓ ( x i , y i , θ ) ,

<!-- formula-not-decoded -->

Here, H is the Hessian of ∑ i ≤ n ℓ ( x i , y i , θ ) evaluated at ˆ θ (see e.g., [RHRS86] for a derivation). For T ⊆ [ n ] , the IF estimate of the leaveT -out model is

<!-- formula-not-decoded -->

We can obtain all the single-sample IF estimates IF i at the cost of a single Hessian inversion and n gradient computations, which then suffice to obtain ˆ θ IF ,T for any T via additivity.

Newton Steps IFs are additive and efficiently computable, but their accuracy suffers when n and d are comparable, or, worse still, if d significantly exceeds n as in the overparameterized setting

1 Some treatments replace dropping with up-weighting, with a resulting difference of sign compared to our convention.

([KATL19]; see also Section 2). A much more accurate approximation to the leaveT -out effect is given by taking a single Newton step (NS) to optimize the leaveT -out loss ∑ i/ ∈ T ℓ ( x i , y i , θ ) , starting from ˆ θ . The NS approximation to the leaveT -out effect is given by

<!-- formula-not-decoded -->

Here, H [ n ] \ T is the Hessian of the leaveT -out loss, evaluated at ˆ θ , and the second equality follows from the fact that θ is a local optimum of ℓ .

As early as 1981, Pregibon [Pre81] observes in the context of leave-one-out estimation for logistic regression that the Newton step approximation is remarkably accurate. At a high level this is because, unlike the IF approximation, the NS approximation takes into account the change to the Hessian from removing the samples in T . For convex losses, the true leaveT -out effect can often be obtained by Newton iteration - taking multiple Newton steps initialized with ˆ θ . The only differences we expect to see between the one-step NS approximation and the result of Newton iteration would arise because the Hessian may change from its value at ˆ θ . Thus, for problems with Lipschitz Hessians, we expect NS to be a very accurate approximation to the true leaveT -out effect; [KATL19] offers experimental validation of this idea for leavek -out estimation in logistic regression, and some formal justification.

Rescaled Influence Functions The accuracy of the NS approximation comes at significant cost, since each fresh T requires a Hessian inversion, and additivity is lost. The RIF recovers additivity and much of the computational efficiency of IF, but retains much of the accuracy of the NS approximation. For sample i ∈ [ n ] , let RIF i be the NS approximation to the leavei -out effect, given by RIF i = H -1 [ n ] \{ i } · ∇ ℓ i ( x i , y i , ˆ θ ) . Then for T ⊆ [ n ] , we define the RIF approximation to the leaveT -out effect to be

RIF is additive by definition.

The computational overhead of RIF compared to IF depends in general on the cost of computing the n leave-one-out Hessian inversions - once these are obtained, no fresh Hessian inversion is needed to compute ˆ θ RIF ,T for any T . RIF is especially attractive in generalized linear models and neural networks with a ReLU activation function, where RIF i can be obtained from IF i by multiplying by a rescale factor (1 -h i ) -1 , where h i is a (generalized) leverage score associated to the i -th sample, which can be computed via a single matrix-vector product with H -1 . Thus, for generalized linear models, no additional Hessian inversion is needed. For example, in logistic regression, the formula for RIF i uses the rescaling (1 -h i ) -1 , where h i = ˆ y i (1 -ˆ y i ) · x ⊤ i H -1 x i ; here ˆ y i ∈ [0 , 1] is the logistic predicted label of the i -th sample according to ˆ θ .

Beyond generalized linear models and ReLU neural networks, whenever each sample makes a lowrank contribution the Hessian, the n leave-one-out Hessian inversions can be computed quickly via the Sherman-Morrison/Woodbury formula. In all of our experiments, the running time overhead to compute RIF is negligible (see Table 2).

In underparameterized settings, it is reasonable to expect that removing a single sample has a negligible effect on the Hessian, and so IF i ≈ RIF i . But for high-dimensional or overparameterized models, a single sample removal can have a significant effect on the Hessian. Our experiments and theory demonstrate the significant accuracy improvement of RIF compared to IF in high-dimensional and overparameterized models.

We note that the idea of summing over estimates of leave-one-out effects to estimate the leaveT -out effect is not new, and has been a central component of many previous data models [IPE + 22]. In their seminal TRAK paper, Park et al. separately consider both the idea of combining LOO effects additively [PGI + 23a][Definition 2.3] and the idea of using a Newton step to estimate LOO effects of a logistic regression [PGI + 23a][Definition 3.1] but do not explicitly combine the rescaling effect in their estimator except to note that the rescaling correction has little to no effect in their setting.

A similar approach that has been the focus of recent research is the Additive-One-Exact data model, which estimates the LTO effect by summing over the exact LOO effects. This data model was

<!-- formula-not-decoded -->

Figure 1: Accuracy of IF versus RIF compared across datasets from image classification (DogFish, Cat vs Dog, Truck vs Automobile), natural language (Spam vs Ham), and audio (ESC-50). In each dataset, we study a binary classification task solved via logistic regression with frozen-embedding features. Each point represents a single choice of subset T . The horizontal axis represents ground truth leaveT -out effect as measured by changes to test predictions, test losses, and self-loss, computed via refitting the logistic model. The vertical axis represents the prediction of this effect made by IF/RIF/NS. A perfectly accurate prediction falls along the black diagonal line. In essentially every case, the RIF prediction falls nicely along this 'ground truth' line, agreeing with the NS prediction, while IF typically underestimates the leaveT -out effect.

<!-- image -->

introduced by Kuschnig et al. [KZCC21] and further analyzed by Hu et al. [HHZM24] and by Huang et al. [HBN + 24]. Kuschnig et al., Hu et al. and Huang et al. study the accuracy of this method for identifying sets of highly influential samples in ordinary least squares (OLS) regressions. Moreover, Huang et al. also note that because a single Newton step is equivalent to a full retrain for the case of OLS, a natural extension of the Additive-One-Exact data model is to sum over the single-Newton step attributions of the individual samples [HBN + 24][Appendix C.2], and Hu et al. [HHZM24][Section 3.1] hypothesize that an NS-like rescaling might explain some of the inaccuracy of the IF estimates in Koh et al.'s experiments. However, the experiments and theoretical analyses of these previous works focus on the case of OLS linear regression where RIF is equivalent to Additive-One-Exact and to the best of our knowledge, no prior work offers quantitative experimental or theoretical comparisons between RIF and other data attribution methods in the high-dimensional beyond this setting. 2

## 2 Empirical Results

We now present empirical findings on the accuracy of RIF estimates for leaveT -out effects. Our experimental setup is inspired by the seminal work of [KL17, KATL19], who assess the accuracy of influence function estimates using logistic regression as a testbed.

We compare IF, NS, and RIF estimates across the first five datasets in Table 1, spanning vision, NLP, and audio classification tasks. Each dataset is processed using a domain-specific embedding, and we train a logistic regression model to solve a binary classification task on the embedded data. We compare the actual vs predicted effect of removing a given set of samples T from the training set, while varying:

2 We are grateful to Tamara Broderick, Jenny Huang, Yuzheng Hu, and Jiaqi Ma for making us aware of these prior works via personal communication.

- Sample-removal strategy: Following [KATL19], we evaluate both random subsets and more structured sets of training points, selected using heuristics such as clustering by a random feature or by Euclidean distance in feature space.
- Accuracy metric: As in [KATL19], we assess accuracy by comparing predicted and actual changes in three scalar quantities when a set T is removed: (1) the total predicted probability for a target class over a subset of test samples, (2) the total test loss on this subset, and (3) the loss on the training set including the removed samples ('self-loss'). The test subset is selected to include a balanced mix of high-loss and randomly chosen test points.
- Size of removed subset: We consider values of | T | ranging from 0 . 1% to 5% of the training set.

We illustrate our main findings in Figure 1. Across every dataset, fraction of sample removals, and accuracy metric, we find that RIF significantly outperforms IF. For more details on our experimental setup, see the supplemental material.

Table 1: Summary of datasets used in our experiments. Each dataset involves a binary classification task which we solve using a regularized logistic regression with mild L 2 regularization. We include both datasets used in the [KATL19] benchmark (DogFish and Enron), as well as several new datasets spanning a wide range of domains, including vision, natural language processing, and audio. For more details about these datasets, see the supplementary material.

| Name      |    d |     n | Test Accuracy   | Description                                                                                      |
|-----------|------|-------|-----------------|--------------------------------------------------------------------------------------------------|
| ESC-50    |  512 |  1600 | 83.0%           | ESC-50 dataset embedded using OpenL3; 'artifi- cial' vs 'natural' classification [Pic15, CWSB19] |
| CatDog    | 2048 |  9600 | 80.9%           | ResNet-50 embeddings of CIFAR-10 cat and dog classes [Kri09, Tor16]                              |
| AutoTruck | 2048 |  9600 | 92.7%           | ResNet-50 embeddings of CIFAR-10 truck and au- tomobile classes [Kri09, Tor16]                   |
| DogFish   | 2048 |  1800 | 98.3%           | Inception v3 embeddings of dog and fish images from ImageNet [SVI + 16, RDS + 15]                |
| Enron     | 3294 |  4137 | 96.1%           | Bag-of-words embeddings of the standard spam vs ham dataset [KATL19, MAP06]                      |
| IMDB      |  512 | 40000 | 87.7%           | BERT embeddings of the IMDB sentiment dataset [MDP + 11, DCLT19]                                 |

Tradeoff: Dimension and Regularization As the number of samples n decreases compared to the model dimension d , we expect the higher-order effect captured by RIF to be stronger. Figure 2 shows this tradeoff, comparing the IF and RIF accuracy while varying the ratio of n and d by sub-sampling a fixed dataset. A similar tradeoff appears when we add an L 2 regularization term of 1 2 λ ∥ θ ∥ 2 to the loss for different values of λ &gt; 0 . Increasing λ dampens the higher-order effects captured by RIF in the limit λ →∞ the Hessian does not vary as samples are removed. In Figure 2 we illustrate this tradeoff by varying λ for a fixed dataset (DogFish), observing that IF and RIF agree for large λ but not for small λ .

Detecting Data Poisonings with RIF One common use of additive data attributions such as influence functions is to detect potential outliers contaminating a dataset [KL17, BGM20, RH25, KLM + 23]. We conduct a simple experiment to demonstrate the advantages of RIF over IF for this task. We take a binary image classification problem (Truck vs Automobile), add an incorrectly-labeled test sample to the training set, and train a logistic regression model on the resulting poisoned dataset. We then compare the accuracy of IF and RIF estimates of the effect that removing the poisoned sample would have on the model's prediction for that test sample. RIF significantly outperforms IF. See Figure 3.

## 3 Theoretical Results

We turn to a theoretical explanation of the effectiveness of RIF to estimate leaveT -out effects in high dimensions. Prior work [KATL19] shows that under reasonable assumptions, the NS approximation

Figure 2: First row: accuracy of IF versus RIF compared across differing ratios of n and d , for the IMDB dataset, subsampled randomly to obtain datasets of varying sizes. IF and RIF are similar when n ≫ d , but as n decreases, RIF remains accurate while IF degrades. Second row: A similar comparison for the overparameterized DogFish dataset, where we vary the regularization strength λ . IF becomes accurate only under strong regularization, while RIF remains robust across settings. In all plots, we compare the predicted versus actual values of the self-loss metric. Blue points show the RIF estimate, green points the IF estimate, and cyan points the Newton step. Point shapes indicate different strategies for selecting training samples to remove, as in Figure 1.

<!-- image -->

Figure 3: On the right we plot the actual vs predicted effect on a test samples logits from removing a 'poisoned' sample from the train set using both IF and RIF. On the left we show the poisoned image corresponding to the leftmost point in the plot - an image of an automobile mislabeled as "Truck". RIF predictions (blue) align much more closely with the actual effects, while IF predictions (green) tend to underestimate these effects.

<!-- image -->

provides a very accurate approximation of the true leaveT -out effect; this is also easily visible in the experiments we reproduced above. Importantly, the NS approximation remains accurate even when the IF estimate is poor. Motivated by this, we focus our analysis on the gap between our RIF estimate and the NS estimate. This leads to a comparatively simple theorem statement, avoiding too many assumptions.

Our setting is as follows. We assume that a model is trained via minimization of a convex empirical risk of the form:

<!-- formula-not-decoded -->

We think of each ℓ i as a per-sample loss from the i -th sample in an underlying training set, although we do not actually need to assume such a training set underlies the optimization problem. Let g i := ∇ ℓ i ( ˆ θ ) and H i := ∇ 2 ℓ i ( ˆ θ ) denote the gradient and Hessian of the i th sample at the solution ˆ θ , and define the total Hessian H := ∑ n i =1 H i .

We make the following set of assumptions on the loss functions. Most of the assumptions are parameterized quantitatively, and our final theorem bounding the quality of the RIF approximation depends on these parameters. Crucially, these assumptions allow for n ≈ d (or even n ≪ d , if regularization is added), so that our main theorem captures how RIF remains accurate for highdimensional barely-underparameterized or even overparameterized models. We discuss after our main theorem statement how to interpret these assumptions quantitatively.

Assumption 1 (Positive Semidefiniteness/Convexity) . We assume that each H i is positive semidefinite, or equivalently, that ℓ i is convex.

The next two assumptions are the key quantitative ones. We offer some discussion now and more after we state our main theorem.

Assumption 2 (No Single-Sample Gradient or Hessian Too Large) . For all i ∈ { 1 , . . . , n } , we assume

<!-- formula-not-decoded -->

for some C ℓ , C R &gt; 0 . Here ∥·∥ op is the operator norm/maximum singular value.

̸

The second clause of Assumption 2 can be rewritten as H i ⪯ C R (1 -C -1 R ) ∑ j = i H ⊤ j . This just captures that no single-sample Hessian H i is too much larger in any direction than the sum of all the others. This is the key condition allowing for large dimension d : even if n ≈ d , this condition can be satisfied (and indeed will be satisfied for, e.g., random low-rank H i ) without taking C R = ω (1) .

̸

Assumption 3 (Cross-Sample Incoherence) . For some ε, δ &gt; 0 , and for all i = j , ∥ ∥ ∥ H 1 / 2 i H -1 H 1 / 2 j ∥ ∥ ∥ op ≤ δ and ∥ ∥ ∥ H 1 / 2 i H -1 g j ∥ ∥ ∥ 2 ≤ ε .

We expect ε, δ to be small because in high dimensions gradients and Hessians of distinct samples are likely to point in close-to-orthogonal directions. We carry this intuition out in more detail below.

Ultimately, we use IF/RIF/NS to estimate the change to f ( ˆ θ ) for some evaluation function f . For instance, in our experiments, f is typically test loss or a test prediction. To show that the RIF and NS estimates are close, we require our evaluation function f to have bounded gradients:

Assumption 4 (Evaluation Gradient Projection Control) . Let ∇ f ( θ ) denote the gradient of an evaluation function f : R d → R . For all i , ∥ ∥ ∥ H 1 / 2 i H -1 ∇ f ( ˆ θ ) ∥ ∥ ∥ 2 ≤ η for some η &gt; 0 .

Let w ∈ [0 , 1] n be a weight change vector. We study the NS and RIF approximations to the optimum of the weighted loss ∑ i ≤ n w i ℓ i ( θ ) . (So, to capture leaveT -out, we set w i = 1 for i ∈ T and otherwise w i = 0 .) We define ˆ θ RIF , w and ˆ θ NS , w analogously to ˆ θ RIF ,T , ˆ θ NS ,T , respectively. We are now ready to state our main theorem:

Theorem 3.1 (Accuracy of Rescaled Influence Function) . Under Assumptions 1-4, for any k ≤ 1 2 δC R ,

<!-- formula-not-decoded -->

̸

The proof of Theorem 3.1 proceeds via a matrix-perturbation analysis which shows that the Hessian inversion in the NS approximation can itself be approximated well without considering the contributions to the inverse from ∇ 2 ℓ i 's interaction with ∇ 2 ℓ j when i = j . We defer the proof to supplemental material, and focus instead on interpreting Theorem 3.1, to illustrate how it captures the improvement of RIF compared to IF.

Interpreting Assumptions and Theorem 3.1 Prior works [GSL + 19, KATL19] prove similar-inspirit results to Theorem 3.1, but concerning IF rather than RIF. A direct comparison of Theorem 3.1 to those results in prior work is challenging, as each result is derived under different assumptions. So, to better understand the practical significance of our bounds compared to those in prior work, and see

why they capture the accuracy of RIF for overparameterized models, we analyze their asymptotic behavior in a simplified setting. Since this is for illustration purposes only, we keep the analysis informal.

Consider linear regression with square loss (ordinary least squares), where the data vectors are drawn i.i.d. from a standard Gaussian distribution, x i ∼ N ( 0 , I ) . And suppose n ≥ (1 + Ω(1)) d , i.e., n and d are comparable. In this case, we know that:

- Each individual Hessian contribution H i = x i x ⊤ i is low rank with rk ( H i ) = 1 and ∥ H i ∥ op = O ( d ) ,
- The total Hessian is approximately isotropic: H ≈ n I ,
- Gradient vectors are bounded in norm: ∥ g i ∥ 2 ≈ √ d .

We can apply the heuristic that random vectors u, v ∈ R d are likely to have |⟨ u, v ⟩| ≈ ∥ u ∥∥ v ∥ / √ d , and so long as n ≥ (1 + Ω(1)) d , we expect the key variables in Theorem 3.1 to scale as:

- C ℓ := max i ∈ [ n ] ∥ ∥ H -1 / 2 g i ∥ ∥ 2 ≈ √ d √ n = O (1)
- C R := max i ∈ [ n ] 1 1 -∥ H -1 / 2 H i H -1 / 2 ∥ op ≈ n n -d = O (1) ,

̸

- δ := max i = j ∥ ∥ ∥ H 1 / 2 i H -1 H 1 / 2 j ∥ ∥ ∥ op = ˜ O ( d n ) ,

̸

- ε := max i = j ∥ ∥ ∥ H 1 / 2 i H -1 g j ∥ ∥ ∥ = ˜ O ( d n ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under these conditions, Theorem 3.1 guarantees that for any set of at most k ≤ k threshold = ˜ Ω ( n √ d ) removed samples, the discrepancy between the RIF and Newton step estimates is bounded by:

<!-- formula-not-decoded -->

The scaling rate n -2 in the denominator matches what we expect for influence functions, as established in [GSL + 19]. But influence function approximations incur significantly worse dimension dependence in the numerator, meaning that n must be much larger than d (indeed, quadratic in d or even larger) to obtain nontrivial guarantees. For comparison, in supplemental material, we analyze the bounds proved by [GSL + 19, KATL19] for influence functions to the same random-design ordinary-least-squares setting and show that they guarantee influence function accuracy only for much larger n or smaller d . For example, the bounds of [GSL + 19] are only applicable for k ≤ ˜ O ( n d 2 ) , and yield an error bound that scales as ˜ O ( k 2 d 4 ∥∇ θ f ∥ 2 n 2 ) .

Finally, to assess the tightness of our result relative to the RIF magnitude itself, we note that under the same random-design least-squares setup and the same heuristics about inner products of highdimensional random vectors, the RIF estimate for the removal of the topk most influential samples scales as

<!-- formula-not-decoded -->

Hence, the ratio of the RIF estimate ("signal") to the RIF-NS error ("noise") is

<!-- formula-not-decoded -->

This implies that RIF provides a good relative-error approximation to NS even in high dimensions, provided k ≪ n √ d .

## 4 Related Work

Influence functions were introduced by Hampel in the context of robust statistics [Ham74], and in the context of estimation of standard errors via the infinitesimal jackknife by Jaeckel [Jae72], with a broad ensuing literature in statistics; see e.g., [Law86, GSL + 19]. Recent work in econometrics [BGM20] uses influence functions to uncover robustness issues in large empirical studies.

The seminal work [KL17] introduced the modern use of influence functions to study the relationship between training data and model behavior in modern machine learning. Ensuing works [BNL + 22, BPF21, GBA + 23, FZ20] study influence functions for neural networks, and use them as a tool to study and interpret model behavior. [GJB19, BYF20] propose second and higher-order approximations to leave-one-out and leaveT -out effects, but these approximations sacrifice linearity and efficiency. Many applications of influence functions have appeared recently, e.g., machine unlearning [GGHVDM19, SAKS21, SW22], data valuation [JDW + 19], robustness quantification [SS19], and fairness [LL22]. To scale influence functions up to very large models and datasets, where Hessian inversion becomes infeasible, several works develop sketching/random projection techniques to approximate influence functions, e.g., [WCZ + 16, PGI + 23b, SZVS22].

Data attribution - tracing model behavior back to subsets of training data - has become a major industry in machine learning; see the recent survey [HL24] and extensive citations therein, as well as the NeurIPS 2024 workshop [NMI + 24] and ICML 2024 tutorial [MIE + 24].

Newton-step approximations to the leave-1-out error have been studied since at least 1981 [Pre81]. Cross-validation is an especially important application [RM18, WKM20]. Additionally, several recent works consider data models that additively combine estimates of leave-one-out effects to compute a leaveT -out effect [KZCC21, IPE + 22, PGI + 23a, HHZM24, HBN + 24]. However, to the best of our knowledge no previous work provides an empirical or theoretical evaluation of the RIF method beyond low-dimensional least-squares regression.

## 5 Discussion and Conclusion

IFs and Importance-Ordering: Revisiting the Common Wisdom Common wisdom regarding IF approximations to leaveT -out effects for high-dimensional models holds that the approximations typically underestimate the true leaveT -out effect, but that there is a strong correlation between the influence-function approximation to the leaveT -out effects and the true leaveT -out effects, especially measured in terms of the ordering of subsets based on their predicted/actual leaveT -out effect. The seminal [KATL19] even phrases this as an outstanding open question, writing that their work 'opens up the intriguing question of why we observe [correlation and underestimation] across a wide range of empirical settings'.

Our work sheds significant light on this question. First of all, it explains why we see such correlation in a great many cases - if most samples have a similar 'rescale factor' relating IF and RIF (which we would expect to happen for e.g., random data), this induces a linear relationship between RIF and IF estimates. Since RIF is an excellent approximation to the true leaveT -out effect, this explains the correlation between IF and the ground truth, and explains why IF typically underestimates the truth the rescale factors are always larger than 1 .

[KATL19] also note that this IF/ground-truth correlation phenomenon need not be universal, and indeed we observe several experiments where it does not hold. For instance, in the first row of Figure 1, in the Cat vs Dog dataset, we see a dramatically non-linear and even non-monotone relationship between IF and ground truth, since different subset-selection strategies yield very different relationships between IF and ground truth. Even the ordering of subsets by IF-predicted effect is not accurate in this example, but RIF remains accurate.

Limitations Although much more accurate than IFs, RIFs are still imperfect predictors of groundtruth - see e.g., the ESC-50 dataset in Figure 1 or the rightmost variants of the IMBD dataset in Figure 2. We expect high-dimensional logistic regression to be a good 'model organism' for high-dimensional machine learning, so our experiments are limited to that setting. RIF also still requires inverting the Hessian; as discussed in related work for very large-scale models this can be computationally infeasible, and approximate techniques are required. While we show that RIFs are

preferable to IFs for detecting certain simple data-poisoning attacks, we do not expect that RIFs are a secure general defense against data poisoning.

Conclusion We show that RIFs are an appealing drop-in replacement for IFs, with little computational overhead in generalized linear models (or whenever individual training samples contribute low-rank terms to the Hessian), but dramatically improved accuracy. Both experiments and theory support this conclusion. Furthermore, the fact that RIFs and IFs differ by a per-sample scaling factor helps to resolve an open question from prior work, showing that the correlation between IF and ground truth leaveT -out occurs when the per-sample scalings all (approximately) agree.

## Acknowledgments and Disclosure of Funding

We would like to thank Jenny Y. Huang, David R. Burt, Yunyi Shen, Tin D. Nguyen, Vishwak Srinivasan, Tamara Broderick, Yuzheng Hu and Jiaqi Ma for helpful conversations and correspondences. This work was supported by NSF Award No. 2238080 and CSAIL Alliances.

## Compute Resources

All experiments were conducted on a server equipped with 64GB RAM, 2 IBM POWER9 CPU cores, and 4 NVIDIA Tesla V100 SXM2 GPUs (each with 32GB memory).

Table 2 details the computational cost of training the base models and computing their IF and RIF data attribution. Another major computational overhead was in retraining the model to obtain ground-truth values for the retrain effect. Despite this, compute resources were not a bottleneck for our work. The total wall-clock time for all experiments reported in the paper was under 100 hours.

Table 2: Comparison of runtime components across datasets. The rescaling step consistently added negligible overhead across all experiments.

| Dataset      | Training   | Hessian   | Inversion   | Influence   | Rescaling       |
|--------------|------------|-----------|-------------|-------------|-----------------|
| ESC50        | 1.8 s      | 0.056 s   | 0.0005 s    | 0.051 s     | 0.0033 s (0.2%) |
| CatDog       | 76 s       | 4.9 s     | 0.010 s     | 4.8 s       | 0.087 s (0.1%)  |
| AutoTruck    | 48 s       | 4.9 s     | 0.0094 s    | 4.8 s       | 0.087 s (0.2%)  |
| DogFish      | 0.43 s     | 0.92 s    | 0.0095 s    | 0.89 s      | 0.015 s (0.7%)  |
| Enron        | 6.7 s      | 15 s      | 0.065 s     | 15 s        | 0.095 s (0.3%)  |
| IMDB (n=16d) | 20 s       | 0.92 s    | 0.0012 s    | 0.87 s      | 0.044 s (0.2%)  |

## References

- [BAHAZ19] Marc-Etienne Brunet, Colleen Alkalay-Houlihan, Ashton Anderson, and Richard Zemel. Understanding the origins of bias in word embeddings. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 803-811. PMLR, 09-15 Jun 2019.
- [BGM20] Tamara Broderick, Ryan Giordano, and Rachael Meager. An automatic finite-sample robustness metric: when can dropping a little data make a big difference? arXiv preprint arXiv:2011.14999 , 2020.
- [BNL + 22] Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, and Roger B. Grosse. If influence functions are the answer, then what is the question? In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , 2022.

- [BPF21] Samyadeep Basu, Phillip Pope, and Soheil Feizi. Influence functions in deep learning are fragile. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021.
- [BYF20] Samyadeep Basu, Xuchen You, and Soheil Feizi. On second-order group influence functions for black-box predictions. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event , volume 119 of Proceedings of Machine Learning Research , pages 715-724. PMLR, 2020.
- [CWSB19] Aurora Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello. Look, listen and learn more: Design choices for deep audio embeddings. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 3852-3856, 2019.
- [DCLT19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pretraining of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) , pages 4171-4186, 2019.
- [Efr92] Bradley Efron. Bootstrap methods: another look at the jackknife. In Breakthroughs in statistics: Methodology and distribution , pages 569-593. Springer, 1992.
- [EIC + 25] Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, and Aleksander Madry. Optimizing ml training with metagradient descent. arXiv preprint arXiv:2503.13751 , 2025.
- [FZ20] Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. Advances in Neural Information Processing Systems , 33:2881-2891, 2020.
- [GBA + 23] Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296 , 2023.
- [GGHVDM19] Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens Van Der Maaten. Certified data removal from machine learning models. arXiv preprint arXiv:1911.03030 , 2019.
- [GJB19] Ryan Giordano, Michael I Jordan, and Tamara Broderick. A higher-order swiss army infinitesimal jackknife. arXiv preprint arXiv:1907.12116 , 2019.
- [GSL + 19] Ryan Giordano, William Stephenson, Runjing Liu, Michael Jordan, and Tamara Broderick. A swiss army infinitesimal jackknife. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1139-1147. PMLR, 2019.
- [GZ19] Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International conference on machine learning , pages 22422251. PMLR, 2019.
- [Ham74] Frank R Hampel. The influence curve and its role in robust estimation. Journal of the american statistical association , 69(346):383-393, 1974.
- [HBN + 24] Jenny Y Huang, David R Burt, Tin D Nguyen, Yunyi Shen, and Tamara Broderick. Approximations to worst-case data dropping: unmasking failure modes. arXiv preprint arXiv:2408.09008 , 2024.
- [HHZM24] Yuzheng Hu, Pingbang Hu, Han Zhao, and Jiaqi Ma. Most influential subset selection: Challenges, promises, and beyond. Advances in Neural Information Processing Systems , 37:119778-119810, 2024.
- [HL24] Zayd Hammoudeh and Daniel Lowd. Training data influence analysis and estimation: A survey. Machine Learning , 113(5):2351-2403, 2024.

- [IASCZ21] Zachary Izzo, Mary Anne Smart, Kamalika Chaudhuri, and James Zou. Approximate data deletion from machine learning models. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics , volume 130 of Proceedings of Machine Learning Research , pages 2008-2016. PMLR, 13-15 Apr 2021.
- [IPE + 22] Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Understanding predictions with data and data with predictions. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA , volume 162 of Proceedings of Machine Learning Research , pages 9525-9587. PMLR, 2022.
- [Jae72] L. Jaeckel. The infinitesimal jackknife, memorandum. Technical Report MM 72-1215-11, Bell Laboratories, Murray Hill, NJ, 1972.
- [JDW + 19] Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas J Spanos. Towards efficient data valuation based on the shapley value. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1167-1176. PMLR, 2019.
- [KATL19] Pang Wei Koh, Kai-Siang Ang, Hubert H. K. Teo, and Percy Liang. On the accuracy of influence functions for measuring group effects. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada , pages 5255-5265, 2019.
- [KL17] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017 , volume 70 of Proceedings of Machine Learning Research , pages 1885-1894. PMLR, 2017.
- [KLM + 23] Alaa Khaddaj, Guillaume Leclerc, Aleksandar Makelov, Kristian Georgiev, Hadi Salman, Andrew Ilyas, and Aleksander Madry. Rethinking backdoor attacks. In International Conference on Machine Learning , pages 16216-16236. PMLR, 2023.
- [Kri09] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.
- [KSL22] Pang Wei Koh, Jacob Steinhardt, and Percy Liang. Stronger data poisoning attacks break data sanitization defenses. Machine Learning , pages 1-47, 2022.
- [KZCC21] Nikolas Kuschnig, Gregor Zens, and Jesús Crespo Cuaresma. Hidden in plain sight: Influential sets in linear models. Technical report, CESifo Working Paper, 2021.
- [Law86] John Law. Robust statistics-the approach based on influence functions, 1986.
- [LL22] Peizhao Li and Hongfu Liu. Achieving fairness at no utility cost via data reweighing with influence. In International conference on machine learning , pages 1291712930. PMLR, 2022.
- [MAP06] Vangelis Metsis, Ion Androutsopoulos, and Georgios Paliouras. Spam filtering with naive bayes-which naive bayes? In CEAS , volume 17, pages 28-69. Mountain View, CA, 2006.
- [MDP + 11] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies , pages 142-150, 2011.

- [MIE + 24] Aleksander Madry, Andrew Ilyas, Logan Engstrom, Sung Min (Sam) Park, and Kristian Georgiev. Data attribution at scale. https://icml.cc/virtual/2024/ tutorial/35228 , 2024. Tutorial presented at the 41st International Conference on Machine Learning (ICML 2024), Vienna, Austria, July 22, 2024.
- [NMI + 24] Elisa Nguyen, Sadhika Malladi, Andrew Ilyas, Logan Engstrom, Sam Park, and Tolga Bolukbasi. Attributing model behavior at scale (attrib). https://neurips. cc/virtual/2024/workshop/84704 , 2024. Workshop at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024), December 14, 2024.
- [PGI + 23a] Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. TRAK: attributing model behavior at scale. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pages 27074-27113. PMLR, 2023.
- [PGI + 23b] Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. Trak: Attributing model behavior at scale. arXiv preprint arXiv:2303.14186 , 2023.
- [Pic15] Karol J. Piczak. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd Annual ACM Conference on Multimedia , pages 1015-1018. ACM, 2015.
- [Pre81] Daryl Pregibon. Logistic regression diagnostics. The annals of statistics , 9(4):705724, 1981.
- [RDS + 15] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision , 115:211-252, 2015.
- [RH25] Ittai Rubinstein and Samuel B. Hopkins. Robustness auditing for linear regression: To singularity and beyond. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025) , 2025.
- [RHRS86] Peter J Rousseeuw, Frank R Hampel, Elvezio M Ronchetti, and Werner A Stahel. Robust statistics: the approach based on influence functions, 1986.
- [RM18] Kamiar Rahnama Rad and Arian Maleki. A scalable estimate of the extra-sample prediction error via approximate leave-one-out. arXiv preprint arXiv:1801.10243 , 2018.
- [SAKS21] Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems , 34:18075-18086, 2021.
- [SS19] Peter Schulam and Suchi Saria. Can you trust this prediction? auditing pointwise reliability after learning. In The 22nd international conference on artificial intelligence and statistics , pages 1022-1031. PMLR, 2019.
- [SVI + 16] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2818-2826, 2016.
- [SW22] Vinith Suriyakumar and Ashia C Wilson. Algorithms that approximate data removal: New results and limitations. Advances in Neural Information Processing Systems , 35:18892-18903, 2022.
- [SZVS22] Andrea Schioppa, Polina Zablotskaia, David Vilar, and Artem Sokolov. Scaling up influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 8179-8186, 2022.

Then,

- [Tor16] TorchVision Contributors. ResNet-50 Pretrained Model. https: //pytorch.org/vision/stable/models/generated/torchvision. models.resnet50.html , 2016. Accessed: 2025-05-14.

[WCZ + 16] Mike Wojnowicz, Ben Cruz, Xuan Zhao, Brian Wallace, Matt Wolff, Jay Luan, and Caleb Crable. 'influence sketching': Finding influential samples in large-scale regressions. In 2016 IEEE International Conference on Big Data (Big Data) , pages 3601-3612. IEEE, 2016.

[WKM20] Ashia Wilson, Maximilian Kasy, and Lester Mackey. Approximate cross-validation: Guarantees for model assessment and selection. In International conference on artificial intelligence and statistics , pages 4530-4540. PMLR, 2020.

## A Proof of Theorem 3.1

Recall our main theoretical result from Section 3:

Theorem A.1 (Theorem 3.1 (restated)) . Under Assumptions 1-4, for any k ≤ 1 2 δC R ,

<!-- formula-not-decoded -->

Before delving into the proof of Theorem 3.1, we introduce a useful technical lemma:

Lemma A.2. Let A 1 , . . . , A k ∈ R d × d and let H ∈ R d × d be positive semidefinite. Suppose:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

Proof of Theorem 3.1. We begin by analyzing the difference between the Newton step and the rescaled influence function (RIF) approximation.

Recall that the Newton step is defined as:

<!-- formula-not-decoded -->

where each g i ∈ R d is the i th gradient component, and H i is the i th contribution to the Hessian. Define the weighted Hessian:

<!-- formula-not-decoded -->

For each i ∈ { 1 , . . . , n } , define w ( i ) := w · 1 { i } to isolate the i -th coordinate. The RIF estimator is given by:

<!-- formula-not-decoded -->

Our goal is to bound the difference between the Newton step and RIF estimators and we do this by bounding the contribution of each individual sample. That is, for each i ∈ [ n ] , we will try to bound

<!-- formula-not-decoded -->

To do so, we begin by expressing each matrix in terms of H and its perturbations. Observe:

<!-- formula-not-decoded -->

Moreover, we define R := ( I -G w ( i ) ) -1 , where G w ( i ) = H -1 / 2 w i H i H -1 / 2 . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with A = H w ( i ) , B = H w ( i ) -H w , we obtain:

<!-- formula-not-decoded -->

We now expand the correction term on the right-hand side further by applying the same identity again, this time expanding H w = H -( H -H w ) ,

<!-- formula-not-decoded -->

where the second term reflects higher-order correction contributions due to recursive matrix inversion.

To bound the full error

<!-- formula-not-decoded -->

It suffices to control the size of each of these terms separately. In other words, we will proceed to bound:

1. The first order correction ( ∇ f ) ⊤ H -1 ( H w ( i ) -H w ) H -1 w ( i ) g i ,
2. The higher order terms ( ∇ f ) ⊤ H -1 ( H -H w ) H -1 w ( H w ( i ) -H w ) H -1 w ( i ) g i .

## Bounding the First Order Correction

To bound the first order correction, we use the same formula above to split H -1 w ( i ) into a leading term and higher order terms. The goal of this separation is to show that this update to the Hessian does not rotate too much of the weight of g i onto the eigenspace of H j for any j = i

<!-- formula-not-decoded -->

̸

Therefore, for any j = i ,

<!-- formula-not-decoded -->

Therefore, this first order correction is at most

̸

<!-- formula-not-decoded -->

̸

## Bounding the Higher Order Corrections

We next bound the second (higher-order) term using the Cauchy-Schwarz inequality.

<!-- formula-not-decoded -->

Using the matrix identity:

̸

We will bound each of these terms independently.

The right-most multiplicand is bounded using the analysis of the first order term

̸

<!-- formula-not-decoded -->

From the triangle inequality,

̸

<!-- formula-not-decoded -->

̸

Using the assumption ∥ ∥ H -1 / 2 H j H -1 / 2 ∥ ∥ op ≤ 1 , it follows that

<!-- formula-not-decoded -->

and from Assumption 5, we also have

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

Next, define A j = w j H -1 / 2 H j H -1 / 2 . Then for all j ,

<!-- formula-not-decoded -->

since ∥ w ∥ ∞ ≤ 1 and by Assumption 2 ∥ ∥ H -1 / 2 H j H -1 / 2 ∥ ∥ op ≤ 1 -1 C R .

̸

Moreover, for all i = j , we have

<!-- formula-not-decoded -->

̸

So,

<!-- formula-not-decoded -->

̸

Applying Lemma A.2 to the collection { A j } , we conclude that

<!-- formula-not-decoded -->

For any k &lt; 1 2 δC R , it follows that I -G w is PSD and ∥ G w ∥ op &lt; 1 , so we have

<!-- formula-not-decoded -->

Therefore,

## Summary:

So far, we have show that for all i ∈ [ n ] ,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

̸

Proof of Lemma A.2. We define the linear operator C : R k × k × d × d → R d × d to be

<!-- formula-not-decoded -->

where M ∈ R k × k × d × d is a rank-4 tensor with M ij ∈ R d × d .

For tensors M , N , define their contraction:

<!-- formula-not-decoded -->

Define Σ : R k × k × d × d → R k × k as Σ( M ) ij := ∥ M ij ∥ op , and define ∆ ∈ R k × k with entries ∆ ij = ∥ ∥ √ A i H -1 √ A j ∥ ∥ op . Then by the triangle inequality and submultiplicativity of the operator norm, we have the point-wise inequality

<!-- formula-not-decoded -->

Applying this iteratively for a sequence M 1 , . . . , M ℓ , we obtain:

<!-- formula-not-decoded -->

̸

Now consider the identity tensor M with M ii = I d and M ij = 0 for i = j . Then:

<!-- formula-not-decoded -->

Let C := C ( M ) . Then:

<!-- formula-not-decoded -->

By triangle inequality and bounding each tensor entry:

<!-- formula-not-decoded -->

Taking ℓ -th roots:

<!-- formula-not-decoded -->

Letting ℓ →∞ , the prefactor tends to 1, giving:

<!-- formula-not-decoded -->

Now bound ∥ ∆ ∥ op . Each diagonal entry ∆ ii = ∥ ∥ √ A i H -1 √ A i ∥ ∥ op = ∥ ∥ H -1 / 2 A i H -1 / 2 ∥ ∥ op ≤ σ . Thus,

<!-- formula-not-decoded -->

Then:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where R is the off-diagonal part of ∆ and ∥ R ∥ 2 F = ∑ i = j δ 2 ij .

Hence:

̸

## B Asymptotic Analyses of the Bounds of [KATL19] and [GSL + 19]

## B.1 Analysis of [KATL19]

Koh et al. [KATL19] present two main theoretical results. The first bounds the difference between a single Newton step and a full retrain, and the second bounds the difference between the Newton step and the influence function estimate. We focus on the latter, since that is more directly comparable to the guarantees of Theorem 3.1. To facilitate a direct comparison, we restate their Proposition 2 with all assumptions made explicit below.

Proposition B.1 (Proposition 2 of [KATL19], rephrased) . Assume the evaluation function f ( θ ) is C f -Lipschitz, the Hessian ∇ 2 θ ℓ ( x, y, θ ) is C H -Lipschitz, and the third derivative of f ( θ ) exists and is bounded in norm by C f, 3 . Let σ min and σ max be the smallest and largest eigenvalues of H 1 , respectively, and define

<!-- formula-not-decoded -->

Then the Newton-influence error Err Nt -inf ( w ) is

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The matrix D ( w ) has eigenvalues between 0 and σ max /λ . The residual term Err f, 3 ( w ) captures the error due to third-order derivatives and is bounded by

<!-- formula-not-decoded -->

To compare this guarantee with Theorem 3.1, which bounds the inner product between the data attribution error and ∇ f , we focus on the first term in the bound from Proposition B.1. This term quantifies the error in estimating the linear evaluation function f using influence functions.

Recall that in the simple linear regression setting we define for our simplified asymptotic analysis, we have H ≈ n I , and this is also the case with H λ, 1 . Using the bound D ( w ) ⪯ σ max λ I from Proposition B.1, the Cauchy-Schwarz inequality gives:

<!-- formula-not-decoded -->

The scaling of σ max /λ depends on the regime. Under strong regularization (e.g., bottom-right of Figure 2), it may be O (1) . However, as Koh et al. observe, this rarely happens in practice, suggesting that it would be more reasonable to assume that σ max /λ = ω (1) .

Let g denote the per-sample gradient, so that g ( w ) = ∑ i w i g i represents the total gradient over removed samples. Following Koh et al.'s approach in Proposition 1, we apply the triangle inequality to bound √

<!-- formula-not-decoded -->

Altogether, the Koh et al. bound on the difference between the IF and the NS estimations for the 1st order change in f comes out to

<!-- formula-not-decoded -->

To get a sense for the scaling of this bound, as with the bound of Theorem 3.1, we compare it to the actual IF estimate to obtain an estimate of signal-to-noise-ratio between IF and its distance from NS

<!-- formula-not-decoded -->

Therefore, the guarantee of Koh et al. do not rule out the possibility of the difference between the NS estimate and the IF estimate completely dominating the removal effects even in simple scenarios (regardless of how k, d may scale with n ).

## B.2 Analysis of [GSL + 19]

## B.2.1 Assumptions and Statement

We now summarize the theoretical guarantees provided by Giordano et al., which underlie their infinitesimal jackknife approximation for estimating the effect of data perturbations.

Assumption 5 (Smoothness; Assumption 1 of [GSL + 19]) . For all θ ∈ Ω θ , each g n ( θ ) is continuously differentiable in θ .

Assumption 6 (Non-degeneracy; Assumption 2 of [GSL + 19]) . For all θ ∈ Ω θ , the Hessian H ( θ, 1 w ) is non-singular, with

<!-- formula-not-decoded -->

Assumption 7 (Bounded averages; Assumption 3 of [GSL + 19]) . There exist finite constants C g and C h such that

<!-- formula-not-decoded -->

Assumption 8 (Local smoothness; Assumption 4 of [GSL + 19]) . There exists ∆ θ &gt; 0 and a finite constant L h such that for all θ with ∥ θ -ˆ θ 1 ∥ 2 ≤ ∆ θ ,

<!-- formula-not-decoded -->

Assumption 9 (Bounded weight averages; Assumption 5 of [GSL + 19]) . The weighted norm 1 √ N ∥ w ∥ 2 is uniformly bounded for w ∈ W by a constant C w &lt; ∞ .

Condition 1 (Set complexity; Condition 1 of [GSL + 19]) . There exists a δ ≥ 0 and a corresponding subset W δ ⊆ W such that:

<!-- formula-not-decoded -->

Definition 1 (Constants from Assumptions) . Define

<!-- formula-not-decoded -->

Theorem B.2 (Error bound for the approximation; Theorem 1 of [GSL + 19]) . Under Assumptions 5-9, if δ ≤ ∆ δ , then

<!-- formula-not-decoded -->

## B.2.2 Analysis

We now analyze the guarantees provided by Giordano et al. [GSL + 19] in the context of our linear regression setting.

In our setup with squared loss and a linear model, the first- and second-order statistics become:

<!-- formula-not-decoded -->

Note that h i ( θ ) does not depend on θ , and thus the local smoothness constant L h (Assumption 8) is zero. Further, the Hessian takes the form

<!-- formula-not-decoded -->

so assuming the data is appropriately scaled, we expect the spectrum of its Hessian to be somewhat clustered and hence C op = O (1) (Assumption 6).

Assumption 7 requires bounds on ∥ g ( θ ) ∥ 2 and ∥ h ( θ ) ∥ 2 . In general, linear regression does not admit uniform convergence over θ due to unbounded gradients as θ → ∞ , but if we fix ∥ θ ∥ to a moderate scale by limiting the scope of Ω θ , we can reasonably assume that ∥ g i ( θ ) ∥ 2 ≈ σ √ d , giving C g ≈ σ √ d = O ( √ d ) and C h ≈ d .

We now turn to Condition 1, which controls how large the weighted deviations can be. In particular, we focus on the second half of this condition, which requires that

<!-- formula-not-decoded -->

When removing a set of k points (i.e., w = 1 -1 T ), the deviation includes k terms of magnitude ∥ h i ( θ ) ∥ 1 ≈ d 2 , resulting in

<!-- formula-not-decoded -->

The bound in Theorem B.2 requires this to be at most ∆ δ = O (1) , so we obtain the constraint:

<!-- formula-not-decoded -->

This represents the main constraint required for Theorem B.2 to apply.

Finally, recall that in the main result of Theorem B.2, the error is bounded by

<!-- formula-not-decoded -->

Given δ ≈ kd 2 n , and C op = C IJ = O (1) , we conclude:

<!-- formula-not-decoded -->

## C Experimental Details

We based our experimental design on that of Koh et al. [KATL19] who evaluate standard influence functions in a similar setting in order to have a clearer benchmark for comparison.

## C.1 Model Training

We fit all the logistic regression models using the scipy.optimize.minimize function to train the model using L-BFGS-B , and set a very strict stopping criterion to ensure that we converge to the global optimum and suppress dependencies on the initial weights when using a warm-start retrain.

For the DogFish and Enron datasets also considered by Koh et al., we used the same L 2 regularization parameter, and for all new datasets, we set the regularization to 1 E -5 .

## C.2 Removal Set Construction

Similar to Koh et al., we evaluate our data attribution methods on removals of 'correlated' sets of samples from every regression. We focus on relatively fewer sample removals, varying the number of samples linearly along the range from 0 . 1% to 5% of the training set. For each dataset and each group construction strategy, we select 40 such sets of samples (1 for each size).

For each such size k , we construct removal sets of size k using the following strategies

1. Clustered Samples: we construct sets of samples clustered either by a single feature or by L 2 distance. When clustering by a single feature, for each set of samples to remove, we select a random sample i ∈ [ n ] and a random feature j ∈ [ d ] , and output the k samples for which X i ′ ,j is closest to X i,j . When clustering by L 2 distance, we select the center sample i ∈ [ n ] uniformly at random and output the k samples closest to it in L 2 norm.
2. Top Percentile Samples: For each of the metrics, we construct a top-percentile set of samples of size k , by selecting first selecting the top 2 k samples and outputting a random subset of half of them. We consider the metrics of: high positive / negative influence on test loss and high positive / negative influence on test predictions, both computed using the standard influence function to keep our benchmark comparable with that of Koh et al.
3. Random Subsets: k samples selected uniformly at random.

## C.3 Datasets and Embeddings

We consider several classification tasks in this paper. For each, we extract features from a particular modality (vision, NLP, or audio), embed them into a d -dimensional representation using a frozen pretrained model, and train a logistic regression classifier on a relevant 2-class classification problem.

For the Enron and DogFish datasets, we try to keep to the same conventions as Koh et al. [KATL19] for a clean comparison.

ESC-50 embedded using OpenL3 ESC-50 is a dataset of ≈ 5 second audio clips each corresponding to one of 50 categories with 40 samples from each category [Pic15]. We convert this to a 2 class classification problem by dividing the categories into 'natural' sounds ( breathing , cat , cow , etc.) and 'artificial' sounds ( airplane , chainsaw , clapping etc.).

We embed these audio samples using last-layer embeddings of the OpenL3 python library [CWSB19]. This produces d = 512 dimensional embeddings, and we separate them into train and test samples using a random 80 -20 train-test split.

CIFAR-2 embedded using ResNet-50 We consider 2 CIFAR-2 datasets generated by limiting the CIFAR-10 dataset [Kri09] to 2 classes (Cat vs Dog, and Automobile vs Truck).

The photos from both train and test sets are embedded using the last-layer embeddings of the default pretrained ResNet-50 model in the torchvision python library [Tor16].

DogFish embedded with Inception v3 We reproduce the DogFish dataset from Koh et al. [KATL19].

This dataset contains photos of dogs and fish from the ImageNet dataset [RDS + 15] embedded using frozen last-layer embeddings of the Inception v3 network [SVI + 16].

Enron embedded with Spacy We reproduce the Enron dataset from Koh et al. [KATL19].

This NLP dataset consists of Spam vs Ham emails [MAP06] embedded using a bag-of-words embedding with the spacy python library using the 'en\_core\_web\_sm' dictionary. We note that our embeddings for the Enron dataset may differ slightly from those of Koh et al. [KATL19], likely due to version differences in the spacy library. However, our empirical results are consistent with theirs.

IMDB embedded with BERT We consider the NLP IMDB sentiment analysis dataset consisting of 50000 movie reviews classified into positive and negative [MDP + 11]. We embed the text reviews using the BERT model [DCLT19].

## C.4 Experiments

An implementation of our experiments is available at github.com/ittai-rubinstein/rescaled-influencefunctions. This appendix provides a concise overview of the procedures implemented in the accompanying code.

## C.4.1 Comparison of Influence and Actual Effect

To produce Figure 1, we select sets of samples to remove based on the methods described in Appendix C.2. For each set of samples we retrain the logistic regression model without these samples to obtain the ground truth effect on the change in the metric f , and compare to the application of the same metric f to the models predicted by each of the data attribution techniques.

Removal effect vs influence One minor distinction considered in the appendix of Koh et al. [KATL19] is between the influence on a metric and the 'parameter influence' on a metric. They define the influence on a metric to be the inner product between the gradient of the metric and the estimated change in model parameters

<!-- formula-not-decoded -->

and the parameter influence of a set of removals (which we simply call the 'removal effect') to be

<!-- formula-not-decoded -->

Figure 4: Accuracy of IF versus RIF compared across datasets from image classification (DogFish, Cat vs Dog, Truck vs Automobile), natural language (Spam vs Ham), and audio (ESC-50). Each datapoint in this experiment is generated as its equivalent in Figure 1, except that instead of evaluating the metric f (e.g., test-loss) on the retrained model or the data model prediction of the retrain effect, we use the leading order Taylor approximation of the change in this metric. There is no major qualitative difference between the results of this experiment and the ones reported in Figure 1, so we decided to keep the original evaluation for a clearer apples-to-apples comparison.

<!-- image -->

We use the latter method to produce all the data points in Figures 1 and 2 (the metric considered in Figure 3 is linear so it is not affected by this distinction). However, similar to Koh et al., we observe very little effect to using the linear method instead.

## C.4.2 Varying n and λ

In these experiments we repeated the same experimental procedure as the one used to generate Figure 1, but with varying levels of L 2 regularization for the DogFish dataset and subsampling the IMDB dataset to different numbers of samples (via uniformly random draws). We report the effect of these removals on self-loss.

## C.4.3 Data Poisoning

To ground our results we consider a particular application of data attribution for detecting data poisoning attacks. We consider the simple data poisoning attack, where an adversary trying to flip our models prediction on some test sample (selected uniformly at random) and adds this sample with a flipped label to the train set. We then run IF and RIF data attributions on the poisoned dataset and use them to predict the effect of the poisoned sample on its own logit ( z i = ⟨ θ , x i ⟩ ) and compare this to the ground truth of a full retrain.

## C.5 Licensing of External Assets

We summarize the license information for all datasets and pretrained models used in our experiments. All assets are cited in the main text.

## Notes

Assets without explicit licenses (e.g., CIFAR-10, Enron, IMDB) are used strictly for non-commercial research purposes. We do not redistribute any datasets or pretrained weights.

Table 3: License summary for datasets used in our experiments. All assets are cited and used in accordance with their respective terms.

| Asset        | Source     | License       | Use / Notes                                                                                  |
|--------------|------------|---------------|----------------------------------------------------------------------------------------------|
| ESC-50       | [Pic15]    | CC BY-NC 3.0  | Freely available for non-commercial re- search use                                           |
| CIFAR-10     | [Kri09]    | Not specified | Widely used in academic settings; original authors affiliated with the University of Toronto |
| ImageNet     | [RDS + 15] | Custom terms  | Access requires agreement to ImageNet's non-commercial license                               |
| Enron Spam   | [MAP06]    | Not specified | Used under standard academic fair use; available via public research repositories            |
| IMDB Reviews | [MDP + 11] | Not specified | Publicly downloadable from the Stanford AI Lab; used for academic research                   |

Table 4: License summary for pretrained models and libraries. All tools are used under compatible terms for non-commercial research.

| Model                   | Version                      | License      | Use / Notes                                                                                                                      |
|-------------------------|------------------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------|
| OpenL3                  | v0.4.2                       | MIT          | Permissive open-source license; commercial use allowed                                                                           |
| ResNet-50 (TorchVision) | v0.13.1                      | BSD 3-Clause | Standard pretrained model from TorchVision; license is permissive, but pretrained weights originate from ImageNet                |
| Inception v3            | -                            | Apache 2.0   | Model license is permissive; weights trained on ImageNet, which restricts downstream use                                         |
| spaCy                   | v3.8.2                       | MIT          | Freely usable model provided by spaCy; li- cense allows commercial and academic use                                              |
| BERT (Trans- formers)   | bert-base- uncased (v4.36.2) | Apache 2.0   | Hugging Face model with a permissive li- cense; trained on BookCorpus and Wikipedia, which may have unclear redistribution terms |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We present the RIF method in Section 1.1. We compare IF and RIF and also present our data poisoning example in Section 2. Finally, we present a theoretical analysis of this comparison in Section 3.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

## Answer: [Yes]

Justification: We discuss the limitations of our work in Section 5.

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

Justification: Section 3 contains a full list of the assumptions needed for our main theoretical result (Theorem 3.1) as well as a detailed discussion of their meaning and the asymptotic scaling of our bounds in a simple setting. Due to space limitations, we moved the proofs of this theorem and its asymptotic analysis to the supplemental material.

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

Justification: In the supplemental material we give a detailed explanation of all of our experimental procedures and also include a library that can be used to reproduce all the figures and tables in our paper.

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

Justification: In the supplemental material, we include a library that can be used to reproduce all the experimental results in our paper and we plan to include a link to a public git repository with the same library in the camera ready version of the paper.

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

Justification: We give a high-level overview of our experimental procedures in Section 2 and a more detailed explanation of all of our methods as well as an implementation of our experimental procedures in the supplemental material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: None of the experiments reported in the paper require error bars, as all of the reported datapoints are computed exactly.

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

Justification: We include a paragraph detailing the compute resources used for our experiments at the end of the main text of our submission.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: Our submission is purely foundational and to the best of our knowledge there is no clear path to any negative applications.

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

Justification: Our submission uses only existing public datasets and models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Our submission utilizes some existing datasets and pretrained embeddings. We cite the relevant sources in the main text and give additional details on licensing in the supplemental material.

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
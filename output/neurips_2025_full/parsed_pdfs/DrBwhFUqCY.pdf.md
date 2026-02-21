## When Models Don't Collapse: On the Consistency of Iterative MLE

## Daniel Barzilai

Weizmann Institute of Science daniel.barzilai@weizmann.ac.il

## Abstract

The widespread use of generative models has created a feedback loop, in which each version of a model is trained on data partially produced by its predecessors. This process has raised concerns about model collapse : A critical degradation in performance caused by repeated training on synthetic data. However, different analyses in the literature have reached different conclusions as to the severity of model collapse. As such, it remains unclear how concerning this phenomenon is, and under which assumptions it can be avoided. To address this, we theoretically study model collapse for maximum likelihood estimation (MLE), in a natural setting where synthetic data is gradually added to the original data set. Under standard assumptions (similar to those long used for proving asymptotic consistency and normality of MLE), we establish non-asymptotic bounds showing that collapse can be avoided even as the fraction of real data vanishes. On the other hand, we prove that some assumptions (beyond MLE consistency) are indeed necessary: Without them, model collapse can occur arbitrarily quickly, even when the original data is still present in the training set. To the best of our knowledge, these are the first rigorous examples of iterative generative modeling with accumulating data that rapidly leads to model collapse.

## 1 Introduction

Generative models such as large language models (LLMs) and diffusion models are increasingly filling the internet with synthetic data. At the same time, web-scraped content remains a common source for training newer models. This creates a feedback loop in which each version of a model is trained partly on outputs of all past models, and synthetic data gradually dominates future training sets. Because identifying and filtering artificially generated content at scale is not always feasible, existing biases and artifacts in the models can become increasingly embedded in the training data and amplified from model to model. This phenomenon has recently been termed Model Collapse : A critical degradation in performance caused by repeated training on synthetic data [Shumailov et al., 2024, Bertrand et al., 2024, Dohmatob et al., 2024a, Gerstgrasser et al., 2025].

The effects of training on synthetic data appear to vary widely across settings. In some cases, even a small amount of synthetic data has been shown to significantly degrade performance [Dohmatob et al., 2025]; in others, model collapse can be avoided altogether [Gerstgrasser et al., 2025, Dohmatob et al., 2024a] or synthetic data may even be beneficial [Jain et al., 2024, Dohmatob et al., 2024b].

To help clarify this picture, we theoretically study the behavior of maximum likelihood estimation (MLE) under iterative training with synthetic data. We focus on a natural and practically motivated setting (also studied by Alemohammad et al. [2024], Gerstgrasser et al. [2025], Dey and Donoho [2024]), where we initially have n samples from some ground-truth distribution, and at each iteration T , the latest model generates n new samples that are then accumulated with all previous data and used to train the next model.

## Ohad Shamir

Weizmann Institute of Science ohad.shamir@weizmann.ac.il

Recently, Dey and Donoho [2024] analyzed a similar setting specifically for distributions arising from exponential families, and showed that for any fixed T and in the limit as n →∞ , the error of the iterationT model is degraded by at most a universal multiplicative constant compared to the initial estimator trained solely on real data. However, their guarantees are asymptotic in n while T is fixed. Therefore, they do not quantify how the performance depends on the proportion of synthetic versus real data, since the fraction of training data that is real is a constant (given by 1 /T ) while n grows to infinity. Therefore, it is difficult to deduce from their results how concerning model collapse may be as the fraction of real data decreases.

In contrast, in this work, we prove in Theorem 4.1 non-asymptotic guarantees that remain valid even as the fraction of real data approaches zero. Under standard regularity and smoothness assumptions (of the kind long used to prove asymptotic consistency and normality of MLE), we show that as long as the number of samples per iteration is at least polylogarithmic in the number of iterations, iterative MLE is consistent (meaning that it converges to the ground truth model as the sample size increases). These findings offer a sharper theoretical understanding of when model collapse can be avoided.

We complement this result with negative ones, that illustrate what can go wrong when these assumptions are violated. In particular, we construct families of distributions for which MLE is consistent when trained on real data, but nonetheless suffers from collapse when synthetic data is iteratively accumulated. Our negative results come in two flavors. In Theorem 5.1, for any fixed sample size n , we construct a family of distributions such that the first iteration gives an excellent approximation to the real distribution, but even the second does not with constant probability. Next, in Theorem 5.2, we show that there exists a family of distributions such that for any n , model collapse will eventually occur, after a number of iterations which grows arbitrarily slowly with n .

To the best of our knowledge, Theorem 5.1 and Theorem 5.2 are the first rigorous examples of iterative generative modeling with accumulating data that rapidly lead to model collapse. Recently, it has been suggested that model collapse does not occur when data accumulates across iterations [Gerstgrasser et al., 2025, Dey and Donoho, 2024, Schaeffer et al., 2025]. Our results show that such claims can only be true under structural assumptions beyond MLE consistency.

## 2 Related Work

Model collapse has recently drawn considerable attention, driven in part by the realization that many datasets are already contaminated with synthetic samples [Alemohammad et al., 2024]. A growing number of empirical studies have reported at least some level of performance degradation in models trained on such data [Shumailov et al., 2024, Alemohammad et al., 2024, Hataya et al., 2023, Bohacek and Farid, 2023, Briesch et al., 2023, Guo et al., 2023].

Several types of synthetic data contamination settings have been considered. Shumailov et al. [2024] considered a fully-synthetic setting, meaning that each model trains only on data produced by the previous model, without any real data. In such a setting, even simple Gaussian distributions can be shown to suffer from severe model collapse. In addition to a fully-synthetic setting, Alemohammad et al. [2024] considered an accumulating-data setting, where data is mixed between real and synthetic data. They observed empirically that in such cases, model collapse may either occur slowly or be avoided altogether, depending on how much real data is added at each iteration. Since then, a few works have theoretically considered accumulating-data settings [Gerstgrasser et al., 2025, Dey and Donoho, 2024]. These works suggest that data accumulation plays a significant role and can mitigate model collapse. Our results show that this is partially true: MLE in a data accumulation setting can avoid model collapse if the models are sufficiently well-behaved, but there exist models that can suffer from severe model collapse even in such a setting.

Among theoretical works, several settings have been studied that differ somewhat from the setting of this paper. A notable line of works considers linear regression, taking advantage of the closed-form expression for the least squares estimate [Dohmatob et al., 2024a, Gerstgrasser et al., 2025, Dohmatob et al., 2025]. These works focus on discriminative models, where previous models are used to label new data, not synthetically generate new data as in this paper. Moreover, the setting of [Dohmatob et al., 2025] is non-iterative, and they analyze the test error when the training data contains some samples that are labeled from a linear predictor drawn from a 'bad' synthetic distribution that differs from the real one. This is a key difference from works such as Dey and Donoho [2024] that analyzed a setting where data gradually accumulates.

There are quite a few works that analyze a specific family of generative models. For example, Gaussians and kernel density estimators have been analyzed in Shumailov et al. [2024], Kazdan et al. [2024], He et al. [2025]. Fu et al. [2024] analyzed model collapse for simplified one-hidden-layer diffusion model. Dohmatob et al. [2024b] analyzed simplified token generators, including Hutter LLMs [Hutter, 2021] and associative memories [Cabannes et al., 2023]. Fu et al. [2025] analyzed several architectures under a framework they called recursive stability, which bears similarities to algorithmic stability. In contrast to all of these works, our work applies to general families of distributions.

A few works characterize iterative generative modeling by analyzing MLE as we do here. Marchi et al. [2024] assume that the differences between distributions in subsequent generations form a martingale difference sequence. However, this assumption is difficult to verify and somewhat unlikely in general. Seddik et al. [2024], Bertrand et al. [2024] analyze a setting where data is mixed from the ground truth model as well as the latest generative model. Our work focuses on a more natural setting of data accumulating over time. There is also the work of Dey and Donoho [2024], which was discussed in the introduction.

Lastly, we note that some works have proposed mechanisms to mitigate collapse through supervision or intervention. For instance, Ferbach et al. [2024], Feng et al. [2025], Amin et al. [2025] show that even minimal forms of ground-truth feedback can substantially reduce the risk of collapse. In contrast, our work focuses on the unsupervised case, where synthetic data accumulates and no corrective signal is available.

## 3 Setting and Notation

## 3.1 Notation

We use bold-faced font to denote vectors, e.g. x ∈ R d , and denote by ∥ x ∥ the Euclidean norm. We let [ n ] := { 1 , . . . , n } . Unless otherwise stated, ∥·∥ denotes the operator norm for matrices and 3 rd-order tensors, where the latter is defined for a 3 rd-order tensor A as ∥ A ∥ = sup v 1 , v 2 , v 3 = 0 A ( v 1 , v 2 , v 3 ) ∥ v 1 ∥∥ v 2 ∥∥ v 3 ∥ . We use the standard big-O notation, with O ( · ) , Θ( · ) and Ω( · ) hiding absolute constants that do not depend on problem parameters. To specify constants that depend only on certain quantities, we may put these quantities in parentheses. For example, C ( K 1 , K 2 ) would denote a constant that depends only on K 1 , K 2 . For a given vector v and radius r &gt; 0 , we let B r ( v ) := { u : ∥ u -v ∥ ≤ r } be the closed ball of radius r centered at v . For a function f ( θ ) , we write ∇ 2 f ( θ ) for its Hessian, and ∇ 3 f ( θ ) for its 3 rd-order derivative tensor, meaning [ ∇ 3 f ( θ )] i,j,k = ∂ 3 ∂θ i ∂θ j ∂θ k f ( θ ) . For a matrix A , we denote by λ min ( A ) , λ max ( A ) its minimal and maximal eigenvalues respectively.

## 3.2 Iterative Maximum Likelihood Estimation

In this section, we formalize the iterative MLE setting that will be studied throughout the paper. Let Θ be a set of parameters and consider a corresponding family of probability density functions (PDFs) over an input space X , given by P Θ := { p θ ( · ) | θ ∈ Θ } . Generative modeling aims to approximate unknown ground truth parameters θ ⋆ using some θ ∈ Θ . Perhaps the most fundamental way to do this is through MLE (throughout the paper, we will also use this acronym to refer to the maximum likelihood estimator - the meaning should be clear from context).

Definition 3.1. Let X be a dataset with elements belonging to X , the MLE trained on X is given by

<!-- formula-not-decoded -->

In the above definition, it is not immediately clear why the MLE exists or if it is unique. Existence is known to hold under mild assumptions, and throughout the proofs, we will explicitly show existence whenever necessary. Regarding uniqueness, the MLE may not be unique in general. However, under mild assumptions, it is known that the MLE converges to the real parameters θ ⋆ (e.g. [Wald, 1949]), and that given sufficiently many samples, the log-likelihood is strictly concave in a neighborhood of θ ⋆ . As such, asymptotically, the MLE is expected to be unique. Nevertheless, formally treating this typically introduces unnecessary and undesired complications to the analysis. It is therefore standard to simply assume that the MLE is unique whenever it exists (e.g. [Lehmann and Casella, 2006]). We

̸

## Algorithm 1 Iterative Maximum Likelihood Estimation

Require: Parameter space Θ ⊆ R d ; family of distributions { p θ } θ ∈ Θ over input space X ; number of samples per iteration n ; target parameters θ ⋆ ∈ Θ .

- 1: Set θ (0) := θ ⋆
- 2: for T = 0 , 1 , 2 , . . . do
- 3: sample X ( T ) := { x ( T ) 1 , . . . , x ( T ) n } ∼ p θ ( T ) ( · ) i.i.d.
- 4: Define cumulative dataset: X ( ≤ T ) := ⋃ T t =0 X ( t )
- 5: Train model on X ( ≤ T ) :

<!-- formula-not-decoded -->

## 6: end for

follow Bertrand et al. [2024] in making a similar, but slightly milder assumption that if there are multiple parameter vectors maximizing the log-likelihood, the argmax may choose the one that is closest to a given reference point. This is, of course, made explicit in the proofs.

Throughout the paper, we will be mostly interested in what happens when MLEs are iteratively re-trained. We will be analyzing a setting where synthetic data accumulates over time, as this is what one naturally expects to occur with web data (see the Related Works, Sec. 2, for a discussion on this). Let θ ⋆ ∈ Θ denote the parameters of the real underlying distribution, and set θ (0) := θ ⋆ . For each iteration T = 0 , 1 , . . . , sample X ( T ) := { x ( T ) 1 , . . . , x ( T ) n } ∼ p θ ( T ) ( · ) i.i.d. and add these to the existing dataset, giving X ( ≤ T ) := ⋃ T t =0 X ( t ) . Then, obtain θ ( T +1) as the MLE given the training data X ( ≤ T ) . We refer the reader to Algorithm 1 for a complete description of iterative MLE. Note that for convenience, the algorithm is written as a minimization problem using the negative log likelihood (or cross-entropy loss).

We are now ready to state our assumptions. They are minor variants of those long used to study MLE in classical statistical literature (since at least Cramér [1946], see also [Le Cam, 1956, van der Vaart, 2000, Lehmann and Casella, 2006]). The first set of assumptions consists of standard regularity conditions (see e.g. [Lehmann, 1999]).

Assumption 1 (Regularity Conditions) .

- (A) There exists some r &gt; 0 such that the closed ball B r ( θ ⋆ ) is contained in Θ .
- (B) The probability density functions p θ are distinct.
- (C) The set of points for which p θ is positive does not depend on θ .

Assumption 1. B is necessary to quantify the distance between distributions p θ , p θ ′ using ∥ θ -θ ′ ∥ . Note that one can always satisfy Assumption 1. B by removing duplicates from P Θ , or by considering the quotient topology as in Redner [1981]. Assumption 1. C avoids pathologies and ensures that log p θ ( x ) is well-defined throughout the iterative sampling process. In distributions modeled using neural networks, probabilities are usually given by applying a softmax, ensuring that they are always positive and thus satisfying Assumption 1. C.

Classical analysis of MLE often require various smoothness assumptions on log p θ ( x ) such as bounded third derivatives (see for example [Cramér, 1946, Lehmann and Casella, 2006]). We will use the following (where r &gt; 0 is the radius from Assumption 1. A):

Assumption 2 (Smoothness) . For any x ∈ X and θ ∈ Θ , log( p θ ( x )) is 3 times continuously differentiable in θ , the partial derivatives support differentiation under the integral sign 1 , and

1 Meaning that we can exchange the order of differentiation and integration. This is a mild assumption that is implicit in many papers.

- (A) Sub-Gaussian gradients: There exists some K 1 &gt; 0 such that for any θ ∈ B r ( θ ⋆ )

<!-- formula-not-decoded -->

- (B) Bounded Hessian: There exists some K 2 &gt; 0 such that for any x ∈ X and θ ∈ B r ( θ ⋆ ) , ∥ ∥ ∇ 2 θ log ( p θ ( x )) ∥ ∥ ≤ K 2 .
- (C) Bounded Third Derivatives: There exists some K 3 &gt; 0 such that for any x ∈ X and θ ∈ B r ( θ ⋆ ) , ∥ ∥ ∇ 3 θ log ( p θ ( x )) ∥ ∥ ≤ K 3 .

Assumptions 2. A, 2. B allow us to bound the difference between sums of random variables and their expected values. Since our bounds are non-asymptotic, one cannot avoid some assumptions to bound these differences. Sub-Gaussianity is a standard assumption in non-asymptotic works, and holds (for example) for bounded random vectors. Nevertheless, our assumptions need to hold only in a small neighborhood of θ ⋆ , making them relatively mild. It is possible to relax these assumptions further, but we do not pursue such generalizations, as it is not the focus of our paper.

Before stating our next assumption, we recall that the Fisher information matrix at some θ is defined as

<!-- formula-not-decoded -->

The Fisher information matrix is well-known to play a central role in the analysis of MLE. Under our other assumptions, it is straightforward to show that the Fisher information matrix is always positive semidefinite (see Appendix A for more information). In fact, standard analyses of MLE (say, to establish asymptotic normality) require the matrix to be positive definite at θ ⋆ [van der Vaart, 2000, Lehmann and Casella, 2006]. Thus, to get our non-asymptotic bounds, it is reasonable to assume the following (where again, r &gt; 0 is the value from Assumption 1. A):

Assumption 3. There exists some λ 0 &gt; 0 such that for any θ ∈ B r ( θ ⋆ ) , λ min ( I ( θ )) ≥ λ 0 .

We note that one can equivalently assume that I ( θ ) is positive definite only at θ ⋆ , and pick r small enough such that by the smoothness assumption, this holds for the neighborhood. However, the formulation above is more convenient for our purposes.

We end by noting that the assumptions above are mostly satisfied (at least approximately) by neural networks. For example, the constant support assumption (Assumption 1. C) is trivially satisfied in standard architectures, since the softmax function widely used to assign probabilities is always non-zero. The smoothness assumptions can be satisfied in various settings, especially when using techniques such as weight decay, which are standard in modern LLM training. Of course, the exact bounds would depend on the architecture and the setting. In general, we believe that weakening these assumptions is quite feasible and is an interesting direction for future work.

## 4 Consistency of Iterative MLE

In this section, we formally show that iterative MLE remains consistent under the conditions from the previous section. In particular, we provide a non-asymptotic bound, which establishes that as long as the number of samples n is at least polylogarithmic in the number of iterations T , then with high probability, all models remain close to the ground-truth parameters. This result highlights that model collapse is not inevitable, even when T →∞ and the fraction of real data vanishes.

Theorem 4.1. Under Assumptions 1 - 3, there exist constants c, C &gt; 0 which depend only on K 1 , K 2 , K 3 , λ 0 and r , such that for any T ∈ N , δ &gt; 0 and any n ≥ c (log( T ) + 1) 2 log 2 ( 7 dT δ ) , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

For sufficiently large n , the bound in Eq. (1) is independent of T , and has only a logarithmic dependence on the dimension d . The theorem is stated for a specific T , but a union bound can easily provide a similar result holding simultaneously for all t ∈ [ T ] , at the cost of a log( T ) factor.

Under the same assumptions as in Theorem 4.1, convergence of parameters also implies convergence in KL-Divergence and convergence in total variation (TV) distance. We refer the reader to Appendix A.1 for background and details. In particular, for a suitable absolute constant C &gt; 0 , Theorem 4.1 implies

<!-- formula-not-decoded -->

We now detail some ways in which Theorem 4.1 differs from past results on model (non)-collapse. In Bertrand et al. [2024], synthetic data does not accumulate across iterations, and for each iteration t ∈ [ T ] , most of the training data used to train θ ( t ) is real. The maximal fraction of synthetic data was increased in the follow-up work Ferbach et al. [2024] when assuming access to the full distribution (i.e. n = ∞ ). Similarly, Dey and Donoho [2024] first fix T and then analyze the limit of n →∞ . They do not provide finite sample guarantees that quantify the dependence between T and n . Seddik et al. [2024] bounded the expected value of the TV distance for distributions over finite vocabularies. When the amount of synthetic data is sufficiently large relative to the vocabulary size, their bound scales as O ( √ k/n ) where k is the total amount of synthetic data. In the data accumulation setting, k = ( T -1) n , in which case the bound becomes O ( √ T/n ) .

We note that while Theorem 4.1 considers a setting where the exact MLE is computable, we believe the theorem can naturally be extended to also accommodate an optimization error, where only an approximate MLE is available. Indeed, we empirically observe in Appendix G that for families of distributions for which an exact formula for the MLE is known, the results are robust to mild optimization error.

## 4.1 Proof Sketch of Theorem 4.1

We provide here the proof intuition for Theorem 4.1, and refer the reader to Appendix D for the rigorous proof.

As a preliminary stage, we first show using Proposition C.1 that given enough samples, for any t ∈ [ T ] , ∥ ∥ θ ( t +1) -θ ( t ) ∥ ∥ is small with high probability. The challenges of this step are that this is done in a non-asymptotic way and takes into account data arising from all previous iterations.

We note that Theorem 4.1 cannot be obtained naively as a direct consequence of the Proposition C.1. Extending Proposition C.1 to a bound on ∥ ∥ θ ( T ) -θ (0) ∥ ∥ using the triangle inequality leads to a suboptimal dependence on T , since it doesn't take into account cancellations from iteration to iteration. Instead, as we will show in the following paragraph, Proposition C.1 will be used to ensure that for large n , θ ( t +1) will be sufficiently close to θ ( t ) to enable Taylor expanding the log likelihood around it. This idea draws inspiration from the asymptotic normality analysis of MLE [Cramér, 1946, Lehmann and Casella, 2006].

To that end, fix some t ∈ [ T ] and observe that for any such t , since θ ( t +1) is the MLE on X ( ≤ t ) , it is a stationary point of the log-likelihood function. As such, Taylor expanding, we show that there exists a matrix R t ∈ R m × m with ∥ R t ∥ ≤ tϵ such that

<!-- formula-not-decoded -->

By definition, θ ( t ) is the MLE for X ( ≤ t -1) , so it is a stationary point for the corresponding log-likelihood function and thus ∑ t -1 j =0 ∇ ℓ j ( θ ( t ) ) = 0 . For notational simplicity, let H t := ( ∑ t j =0 ∇ 2 ℓ j ( θ ( t ) ) ) + R t , then the above simplifies to

<!-- formula-not-decoded -->

In the full proof, we show that H t is invertible. In such a case, we can rearrange the above equation to obtain

<!-- formula-not-decoded -->

Importantly, this allows us to express how the parameters evolve over many iterations by taking a telescopic sum as follows.

<!-- formula-not-decoded -->

The expected value of ∇ ℓ t ( θ ( t ) ) (conditioned on θ ( t ) ) can be shown to be zero, so that the first term forms a martingale, which allows us to bound the norm essentially as if all samples were independent. Since each ∇ ℓ t is scaled by 1 t +1 , the variance scales as 1 ( t +1) 2 . So the variance of the sum can be upper bounded as ∑ T t =1 1 t 2 ≤ ∑ ∞ t =1 1 t 2 ≤ π 2 6 . In summary, we show that with high probability

<!-- formula-not-decoded -->

The second term in Eq. (2) has to be treated differently, as correlations between H t -1 and ∇ ℓ t imply that each term is not necessarily mean-zero, and so the sum should be expected to have some dependence on T . This term somewhat complicates the proof, as bounding it requires knowing that ∥ ∥ θ ( t ) -θ (0) ∥ ∥ is sufficiently small for all t &lt; T . The proof thus works inductively, bounding this term from iteration to iteration. Roughly speaking, in the end, we show that with high probability

<!-- formula-not-decoded -->

where the last inequality follows from the assumption that n is sufficiently large.

## 5 Necessity of Structural Assumptions

Theorem 4.1 provides conditions under which the iterative MLE retains good performance, even if the proportion of synthetic data approaches 1 . Clearly, this cannot always be true. In particular, there are well-known examples of families of distributions on which even standard MLE is inconsistent: Namely it will not converge to the ground-truth parameters as the sample size increases, even when trained purely on real data (e.g. [Bahadur, 1958, Ferguson, 1982, Le Cam, 1990]). In such situations, the whole question of model collapse is rather meaningless. Thus, a natural (informal) follow-up question is the following: In the setting where synthetic data is added to the real dataset in each iteration, is there a family of distributions that is sufficiently well-behaved for MLE to be asymptotically consistent (when trained on real data), but still exhibits rapid model collapse? In other words, do there exist cases where the MLE can learn the real distribution, and yet model collapse still occurs when applying MLE iteratively?

In this section, we show that the answer is yes, and demonstrate different settings in which model collapse can occur when the conditions of Theorem 4.1 are not satisfied. To the best of our knowledge, these are the first rigorous examples of iterative generative modeling with accumulating data that rapidly leads to model collapse.

We emphasize that, following the rest of the paper, we focus here on a setting where synthetic data iteratively accumulates on top of the real data. A different model collapse setting studied in some previous works is when at each iteration, MLE is performed purely on synthetic data generated by the latest model. In such a setting, the real training data disappears already after a single iteration, and it

has been shown to lead to model collapse even for very well-behaved distributions such as Gaussians [Shumailov et al., 2024]. It has recently been suggested that if data is added rather than replaced (as in our setting), the extent to which iterative MLE performance degrades is limited [Gerstgrasser et al., 2025, Dey and Donoho, 2024, Schaeffer et al., 2025]. We show here that this can be true only if further assumptions are made, beyond just MLE consistency (as we do in Theorem 4.1).

To formalize our results, we will require the following consistency definition for MLE:

Definition 5.1. We will say a family of distributions P Θ is TV-consistent , if for any θ ⋆ ∈ Θ and n ∈ N , the MLE ˆ θ trained on n i.i.d. samples from p θ ⋆ exists, and

<!-- formula-not-decoded -->

Note that we use here convergence in total variation, rather than convergence in parameters as in Theorem 4.1. The reason is that to establish our negative results, we have to make use of distributions that do not follow the assumptions of Theorem 4.1, and in particular do not satisfy the smoothness assumptions there. Without smoothness, parametric convergence and convergence of distributions are no longer equivalent in general. Thus, using a probability metric such as total variation is more natural in our setting, as we are ultimately interested in approximating the ground-truth distribution.

## 5.1 Models Can Collapse Immediately

By definition, for a TV-consistent family of distributions, p θ (1) is a good approximation of the ground truth distribution p θ ⋆ , assuming the number of samples n is sufficiently large. Our first negative result shows that, perhaps surprisingly, one cannot hope to show the same even for p θ (2) without further assumptions. Specifically, for any n there is some family of distributions (that may depend on n ), such that MLE on n samples from the ground-truth distribution will perform well, but if we now augment the data with n synthetic samples from the MLE solution, and re-run MLE, then the resulting distribution p θ (2) will exhibit model collapse with constant probability.

Theorem 5.1. There exists Θ ⊆ R 2 and θ ⋆ ∈ Θ , such that for any n ∈ N , there is a TV-consistent family of distributions { p θ } θ ∈ Θ (that may depend on n ) such that

1. with probability at least 1 -1 n ,

<!-- formula-not-decoded -->

2. For some absolute constants c, C &gt; 0 , it holds with probability at least c that

<!-- formula-not-decoded -->

In the above theorem, as the number of samples grows, we can find a family of distributions such that p θ (1) is very close to p θ ⋆ with high probability, but there is some constant probability that p θ (2) will be far from p θ (1) . This implies that statements similar to Theorem 4.1 are not possible for general TV-consistent families without further assumptions. Indeed, Theorem 5.1 implies that the relative gap in total variation between iterations t = 1 and t = 2 can be arbitrarily large, since

<!-- formula-not-decoded -->

We now provide some intuition for the proof of Theorem 5.1, with the full rigorous proof appearing in Appendix E. We consider a family of distributions, given by the following parameterized mixture of uniform distributions on R :

<!-- formula-not-decoded -->

where U ( · ) is the uniform distribution on an interval, Θ = { ( α, µ ) | α ∈ [ 0 , 1 4 ] µ ∈ [2 , 3 -f ( α )] } are the parameters, and f is a positive function that decays very quickly with α , so that the PDF of U ([ µ, µ + f ( α )]) approaches a delta function (the exact form of f depends on n ). Let θ (0) := θ ⋆ := ( α (0) = 0 , µ (0) = 0) such that p θ (0) = U ([0 , 1]) . We show that the MLE θ (1) = ( α (1) , µ (1) ) satisfies

<!-- formula-not-decoded -->

As such, α (1) converges very quickly to α (0) as n increases, and we prove that this implies a rapid convergence of TV( p θ (0) , p θ (1) ) , regardless of the value of µ (1) .

We now move on to analyzing the second iteration. Because α (1) ≈ 1 2 n , then with some constant probability (over the sampling of n new samples from p θ (1) ), at least one of these samples x (1) i will be inside the interval [2 , 3] . When this happens, because f ( α ) is tiny for larger values of α (leading to a high likelihood in the interval [ µ, µ + f ( α )] ), the MLE solution θ (2) = ( α (2) , µ (2) ) will be such that x (1) i ∈ [ µ (2) , µ (2) + f ( α (2) )] and α (2) will be sufficiently large so that f ( α (2) ) is very small. In particular, α (2) will be considerably larger than the ground truth α (0) = 0 , leading to model collapse.

## 5.2 Arbitrarily Fast Model Collapse

Theorem 5.1 shows that without further assumptions, model collapse can occur already after a single iteration. However, the construction requires picking the distribution according to the sample size n ∈ N , which is arguably unnatural. Below, we show that this requirement can be removed: Namely, there exists a family of distributions where model collapse will occur for any sample size n . On the flip side, the model collapse no longer occurs after a single iteration, but rather after a number of iterations that grows with n (although the growth rate can be arbitrarily slow):

Theorem 5.2. Let ϕ : (0 , ∞ ) → (0 , ∞ ) be any strictly monotonically increasing function such that lim n →∞ ϕ ( n ) = ∞ . Then there exists an absolute constant C &gt; 0 , a set Θ , θ ⋆ ∈ Θ and a TV-consistent family of distributions P Θ (which depends on ϕ ), such that for any δ ∈ (0 , 1) , n ∈ N , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Importantly, ϕ can be chosen to grow arbitrarily slowly. For example, taking ϕ ( n ) = log log( n +1) , Theorem 5.2 implies that one can exhibit model collapse in as few as O (log log( n +1)) iterations.

The proof of Theorem 5.2 draws inspiration from the proof of Theorem 5.1, but the construction is more involved, as the distribution can no longer depend on the number of samples n . The family will consist of two types of distributions. The first, which we denote as h α , has the form

<!-- formula-not-decoded -->

where α i ∈ [0 , 1 4 ] , and α has a finite number of non-zero indices. One way to think of these distributions is as sampling using an iterative process, where starting from j = 0 , one flips a coin with bias α j , and either samples a point from U ([ j, j +1 -2 α j ]) (with probability 1 -α j ), or with probability α j , increase j by one and repeat the process, until some point is sampled. We also include a family of distributions g β,J corresponding to

<!-- formula-not-decoded -->

where J ∈ N \ { 1 } , β ∈ [0 , 1] and f is a function that decays very quickly as J increases.

Now, consider the ground truth distribution to be h 0 (meaning α (0) j = 0 for all j ), which is actually just U ([0 , 1]) . We show that at any iteration t , the density h α ( t ) that maximizes the likelihood out of functions of the form h α is given by taking α ( t +1) j = 1 2 (1 -max X ( ≤ t ) ∩ [ j, j +1] -j ) .

Thus, the general procedure is as follows: For any J ∈ N , once α ( t ) j &gt; 0 for every j ≤ J -1 , there is a non-zero chance that a new sample x ( t ) i will reach interval [ J, J +1] , ensuring α ( t +1) J &gt; 0 . We choose the function f so that for any N ∈ N , there is some J N such that if n ≤ N and if there is some sample in [ J N , J N +1] , then the MLE will be of the form g β,J (as f ( J N ) is sufficiently small, leading to a high likelihood of the sample). We show that once this happens, the total variation distance will be large, and the proof will be complete.

The difficult part is showing that for any J ∈ N , there is some time T ∈ N such that with high probability, there will be a sample in [ J, J +1] . Moreover, this T can be chosen to be essentially

independent of n . Since we may let J N grow arbitrarily slowly in N , and the number of iterations needed to obtain a sample in [ J N , J N +1] can be upper bounded independently of N , the number of iterations needed for model collapse can grow arbitrarily slowly with N (and thus with n ).

## 5.3 Implications and Relation to Theorem 4.1

The results of this section inform us how in the absence of the assumptions of Theorem 4.1, model collapse can occur arbitrarily quickly. Even though the constructions of Theorems 5.1 and 5.2 are artificial, they highlight more general phenomena that are needed for model collapse to occur or to be avoided. In particular, in our view, the main difference from Theorem 4.1 is the smoothness assumption. Theorems 5.1 and 5.2 crucially use a highly non-smooth construction, in which slight perturbations of the parameters can induce huge differences in the resulting model, and we find it unlikely that a negative example would be possible without this behavior.

## 6 Discussion

We studied model collapse in a setting that has recently gained interest in the literature, where synthetic data accumulates over time. Focusing on MLE, we showed that collapse can be avoided under standard assumptions even as the proportion of real data vanishes, provided that the number of samples is polylogarithmic in the number of iterations. At the same time, when these assumptions are not satisfied, we construct scenarios where the MLE is consistent, yet collapse occurs arbitrarily quickly with synthetic data. These examples show that MLE consistency alone is not sufficient for preventing model collapse even in the accumulating-data setting.

While the assumptions in this work are rather classic, they may not be the mildest possible while still allowing for positive results. Moving forward, it would be interesting to bridge the gap still present in this work between the assumptions in the negative and positive results and characterize assumptions that are both necessary and sufficient for avoiding model collapse. Our hope is that these results contribute to a clearer theoretical understanding of model collapse, and lead to a more fine-grained perspective on when it does or does not occur.

## Acknowledgments and Disclosure of Funding

This research is supported in part by European Research Council (ERC) grant 754705, by the Israeli Council for Higher Education (CHE) via the Weizmann Data Science Research Center and by research grants from the Estate of Harry Schutzman and the Anita James Rosen Foundation.

## References

- Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard Baraniuk. Self-consuming generative models go mad. In The Twelfth International Conference on Learning Representations , 2024.
- Kareem Amin, Sara Babakniya, Alex Bie, Weiwei Kong, Umar Syed, and Sergei Vassilvitskii. Escaping collapse: The strength of weak data for large language model training. arXiv preprint arXiv:2502.08924 , 2025.
- RR Bahadur. Examples of inconsistency of maximum likelihood estimates. Sankhy¯ a: The Indian Journal of Statistics , pages 207-210, 1958.
- Quentin Bertrand, Avishek Joey Bose, Alexandre Duplessis, Marco Jiralerspong, and Gauthier Gidel. On the stability of iterative retraining of generative models on their own data. In ICLR , 2024.
- Matyas Bohacek and Hany Farid. Nepotistically trained generative-ai models collapse. arXiv e-prints , pages arXiv-2311, 2023.
- Martin Briesch, Dominik Sobania, and Franz Rothlauf. Large language models suffer from their own output: An analysis of the self-consuming training loop. CoRR , 2023.
- Vivien Cabannes, Elvis Dohmatob, and Alberto Bietti. Scaling laws for associative memories. arXiv preprint arXiv:2310.02984 , 2023.
- Harald Cramér. Mathematical Methods of Statistics . Princeton University Press, 1946.
- Apratim Dey and David Donoho. Universality of the π 2 / 6 pathway in avoiding model collapse. arXiv preprint arXiv:2410.22812 , 2024.
- Elvis Dohmatob, Yunzhen Feng, and Julia Kempe. Model collapse demystified: The case of regression. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024a.
- Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, and Julia Kempe. A tale of tails: Model collapse as a change of scaling laws. In Forty-first International Conference on Machine Learning , 2024b.
- Elvis Dohmatob, Yunzhen Feng, Arjun Subramonian, and Julia Kempe. Strong model collapse. In The Thirteenth International Conference on Learning Representations , 2025.
- Yunzhen Feng, Elvis Dohmatob, Pu Yang, Francois Charton, and Julia Kempe. Beyond model collapse: Scaling up with synthesized data requires verification. In The Thirteenth International Conference on Learning Representations , 2025.
- Damien Ferbach, Quentin Bertrand, Joey Bose, and Gauthier Gidel. Self-consuming generative models with curated data provably optimize human preferences. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=cyv0LkIaoH .
- Thomas S Ferguson. An inconsistent maximum likelihood estimate. Journal of the American Statistical Association , 77(380):831-834, 1982.
- Shi Fu, Sen Zhang, Yingjie Wang, Xinmei Tian, and Dacheng Tao. Towards theoretical understandings of self-consuming generative models. In International Conference on Machine Learning , pages 14228-14255. PMLR, 2024.
- Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian, and Dacheng Tao. A theoretical perspective: How to prevent model collapse in self-consuming training loops. In The Thirteenth International Conference on Learning Representations , 2025.
- Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Tomasz Korbak, Henry Sleight, Rajashree Agrawal, John Hughes, Dhruv Bhandarkar Pai, Andrey Gromov, et al. Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data. In First Conference on Language Modeling , 2025.
- Yanzhu Guo, Guokan Shang, Michalis Vazirgiannis, and Chloé Clavel. The curious decline of linguistic diversity: Training language models on synthetic text. arXiv preprint arXiv:2311.09807 , 2023.
- Ryuichiro Hataya, Han Bao, and Hiromi Arai. Will large-scale generative models corrupt future datasets? In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 20555-20565, 2023.
- Hengzhi He, Shirong Xu, and Guang Cheng. Golden ratio weighting prevents model collapse. arXiv preprint arXiv:2502.18049 , 2025.

Marcus Hutter. Learning curve theory. arXiv preprint arXiv:2102.04074 , 2021.

- Ayush Jain, Andrea Montanari, and Eren Sasoglu. Scaling laws for learning with real and surrogate data. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M Kakade, and Michael I Jordan. A short note on concentration inequalities for random vectors with subgaussian norm. arXiv preprint arXiv:1902.03736 , 2019.
- Joshua Kazdan, Rylan Schaeffer, Apratim Dey, Matthias Gerstgrasser, Rafael Rafailov, David L Donoho, and Sanmi Koyejo. Collapse or thrive? perils and promises of synthetic data in a self-generating world. arXiv preprint arXiv:2410.16713 , 2024.
- Lucien Le Cam. On the asymptotic theory of estimation and testing hypotheses. In Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Contributions to the Theory of Statistics , volume 3, pages 129-157. University of California Press, 1956.
- Lucien Le Cam. Maximum likelihood: an introduction. International Statistical Review/Revue Internationale de Statistique , pages 153-171, 1990.
- Erich L Lehmann and George Casella. Theory of point estimation . Springer Science &amp; Business Media, 2006.
- Erich Leo Lehmann. Elements of large-sample theory . Springer, 1999.
- Matteo Marchi, Stefano Soatto, Pratik Chaudhari, and Paulo Tabuada. Heat death of generative models in closed-loop learning. arXiv preprint arXiv:2404.02325 , 2024.
- Whitney K Newey and Daniel McFadden. Large sample estimation and hypothesis testing. Handbook of econometrics , 4:2111-2245, 1994.
- Richard Redner. Note on the consistency of the maximum likelihood estimate for nonidentifiable distributions. The Annals of Statistics , pages 225-228, 1981.
- Rylan Schaeffer, Joshua Kazdan, Alvan Caleb Arulandu, and Sanmi Koyejo. Position: Model collapse does not mean what you think. arXiv preprint arXiv:2503.03150 , 2025.
- Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, and Merouane Debbah. How bad is training on synthetic data? a statistical analysis of language model collapse. arXiv preprint arXiv:2404.05090 , 2024.
- Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. Ai models collapse when trained on recursively generated data. Nature , 631(8022):755-759, 2024.
- George Tauchen. Diagnostic testing and evaluation of maximum likelihood models. Journal of Econometrics , 30(1-2):415-443, 1985.
- Joel A Tropp. User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics , 12:389-434, 2012.
- AWvan der Vaart. Asymptotic statistics , volume 3. Cambridge university press, 2000.
- Roman Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- Abraham Wald. Note on the consistency of the maximum likelihood estimate. The Annals of Mathematical Statistics , 20(4):595-601, 1949.

## A Background on Likelihood Estimation

For any θ ∈ Θ , the Fisher information matrix is defined as

<!-- formula-not-decoded -->

We state here some well-known results regarding the Fisher information matrix that will be used throughout the proofs (e.g. [Lehmann, 1999][Section 7.5]).

Theorem A.1. If Assumptions 1, 2 hold, then for any θ ∈ B r ( θ (0) ) ,

<!-- formula-not-decoded -->

Note that in particular, this implies that I ( θ ) is the covariance matrix of the random vector ∇ θ log p θ ( x ) and is therefore p.s.d.

We will also need the following.

Theorem A.2. If Assumptions 1, 2 hold, then for any θ ∈ B r ( θ (0) ) ,

<!-- formula-not-decoded -->

Note that the above theorem also implies ∥I ( θ ) ∥ ≤ sup x ∥ ∥ ∇ 2 θ log p θ ( x ) ∥ ∥ ≤ K 2 . We state this formally as the following corollary.

Corollary A.1. If Assumptions 1, 2 hold, then for any θ ∈ B r ( θ (0) ) ,

<!-- formula-not-decoded -->

## A.1 Parametric Convergence vs. KL vs. TV

Two common ways to compare PDFs p, q over an input space X are the KL divergence:

<!-- formula-not-decoded -->

and the total variation distance

<!-- formula-not-decoded -->

We note that while the TV is a proper metric, the KL divergence is not, as it is not symmetric. Nevertheless, the two can be related by the well-known Pinsker's inequality:

<!-- formula-not-decoded -->

It is well known that under sufficient smoothness assumptions, convergence in parameters implies convergence in KL and total variation. Indeed, for a fixed x ∈ X , consider a second-order Taylor expansion of log ( p θ ( x )) around θ (0) , which gives

<!-- formula-not-decoded -->

where the remainder R ( x ) can be shown to satisfy | R ( x ) | ≤ K 3 6 ∥ ∥ θ -θ (0) ∥ ∥ 3 under Assumption 2. By Theorem A.1 the expected value of the gradient term is 0 and by Theorem A.2 the expected value of the hessian term is -I ( θ (0) ) .

As such, Taylor expanding at every point x together with Theorem A.1 and Theorem A.2, the KL divergence can be approximated as

<!-- formula-not-decoded -->

By Pinsker's inequality, this implies

<!-- formula-not-decoded -->

## B Concentration

We start with a couple of known results that will be useful for approximating the gradient and hessian of the log-likelihood.

Theorem B.1 (Jin et al. [2019] Corollary 7) . Let z 1 , . . . , z T ∈ R d be random vectors and assume there exist fixed σ 1 , . . . , σ t such that for all t ∈ [ T ] , E [ z t | z 1 , . . . , z t -1 ] = 0 and

<!-- formula-not-decoded -->

Then there exists an absolute constant C &gt; 0 such that for any δ &gt; 0 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Theorem B.2 (Tropp [2012] Theorem 7.1) . Let { M t } be a finite sequence of random symmetric d × d matrices such that E [ M t | M 1 , . . . , M t -1 ] = 0 . Assume further that there exists a fixed sequence of symmetric d × d matrices { A t } such that M 2 t ⪯ A 2 t almost surely. Let σ 2 := ∥ ∥ ∑ t A 2 t ∥ ∥ , then for all u ≥ 0 ,

<!-- formula-not-decoded -->

We bring Theorem B.3 to a slightly more convenient form for our uses.

Theorem B.3. Let { M t } T t =1 be a finite sequence of random symmetric d × d matrices such that E [ M t | M 1 , . . . , M t -1 ] = 0 . Assume further that there exists some K &gt; 0 such that ∥ M t ∥ ≤ K almost surely. Then for any δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Proof. Set A 2 t := K 2 I d , σ 2 := TK 2 , apply Theorem B.2 once to bound ∑ t M t and again to bound -∑ t M t . The corollary follows from the union bound.

Lemma B.1. Under Assumptions 1, 2, if θ (1) , . . . , θ ( T -1) ∈ B r ( θ (0) ) then there exists an absolute constant C &gt; 0 such that for any δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Proof. For t ∈ { 0 , . . . , T -1 } , i ∈ [ n ] let z t,i := 1 t +1 ∇ log ( p θ ( t ) ( x ( t ) i ) ) . We order these Tn random vectors z t,i first by t and then by i . Specifically, let ρ : [ Tn ] →{ 0 , . . . , T -1 } × [ n ] be this mapping of indices, such that

<!-- formula-not-decoded -->

By Theorem A.1, for all k ∈ [ Tn ] , E [ z ρ ( k ) | z ρ (1) , . . . , z ρ ( k -1) ] = 0 . Furthermore, by Assumption 2. A, for any t, i and any u ≥ 0

<!-- formula-not-decoded -->

In particular, for all k ∈ [ Tn ] letting σ k := K 1 / ( ρ ( k ) 1 +1) (where ρ ( k ) 1 is the t ∈ { 0 , . . . , T -1 } that corresponds to ρ ( k ) ) we have

<!-- formula-not-decoded -->

As such, by Theorem B.1 there exists an absolute constant C &gt; 0 such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Note that since ∑ T t =1 1 t 2 ≤ π 2 6 , we have

<!-- formula-not-decoded -->

We obtain with the same probability that for a suitable altered constant C &gt; 0 ,

<!-- formula-not-decoded -->

We can also obtain concentration for a single ¯ θ ∈ B r ( θ (0) ) . We omit the proof as it is a simplified version of Lemma B.1 (specifically, the assumptions and Theorem A.1 imply that the conditions of Theorem B.1 are satisfied, which gives the following result).

Lemma B.2. Let ¯ θ ∈ B r ( θ (0) ) and x 1 , . . . , x n ∼ p ¯ θ i.i.d. Under Assumptions 1, 2, there exists an absolute constant C &gt; 0 such that for any θ ∈ B r ( θ (0) ) , δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

We will also need the following result for the Hessian of the log-likelihood:

Lemma B.3. Under Assumptions 1, 2, if θ (1) , . . . , θ ( T -1) ∈ B r ( θ (0) ) then there exists an absolute constant C &gt; 0 such that for any δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Proof. For t ∈ { 0 , . . . , T -1 } , i ∈ [ n ] let M t,i := -∇ 2 log ( p θ ( t ) ( x ( t ) i ) ) -I ( θ ( t ) ) . We order these Tn random matrices M t,i first by t and then by i . Specifically, let ρ : [ Tn ] →{ 0 , . . . , T -1 } × [ n ] be this mapping of indices, such that

<!-- formula-not-decoded -->

By Theorem A.2, for all k ∈ [ Tn ] , E [ M ρ ( k ) | M ρ (1) , . . . , M ρ ( k -1) ] = 0 . Furthermore, by Assumption 2. B and Corollary A.1, for any k ∈ [ Tn ] ,

<!-- formula-not-decoded -->

As such, by Theorem B.3 there exists an absolute constant C &gt; 0 such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Once again, we can also obtain an analogous result for a single ¯ θ ∈ B r ( θ (0) ) . The proof is also analogous to Lemma B.3.

Lemma B.4. Let ¯ θ ∈ B r ( θ (0) ) and x 1 , . . . , x n ∼ p ¯ θ i.i.d. Under Assumptions 1, 2, there exists an absolute constant C &gt; 0 such that for any θ ∈ B r ( θ (0) ) , δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

## C Preparatory Results

## C.1 Non-Asymptotic Consistency

LemmaC.1. If Assumption 2 holds, then for every x ∈ X , ∇ 2 θ log p θ ( x ) is K 3 -Lipschitz on B r ( θ (0) ) ; that is,

<!-- formula-not-decoded -->

Proof. Fix x ∈ X and θ, θ ′ ∈ B r ( θ (0) ) . Consider the line segment γ : [0 , 1] → B r ( θ (0) ) given by γ ( t ) = θ + t ( θ ′ -θ ) . Note that the convexity of B r ( θ (0) ) implies that γ ( t ) ∈ B r ( θ (0) ) for all t ∈ [0 , 1] . From the fundamental theorem of calculus,

<!-- formula-not-decoded -->

where [ ∇ 3 θ log p γ ( t ) ( x )[ θ ′ -θ ] ] ij = ∑ d k =1 ∂ 3 ∂θ i ∂θ j ∂θ k log p γ ( t ) ( x )[ θ ′ -θ ] k .

Applying the operator norm and Assumption 2,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma C.2. Under Assumptions 1, 2, for any t ∈ N , if θ (0) , . . . , θ ( t ) ∈ B r ( θ (0) ) , then there exists an absolute constant C &gt; 0 such that for any δ &gt; 0 , with probability at least 1 -δ

<!-- formula-not-decoded -->

Proof. By the triangle inequality,

<!-- formula-not-decoded -->

By Lemma C.1, ∇ 2 ℓ j ( θ ) is K 3 Lipschitz in θ . Using this and the triangle inequality, the first term is bounded by

<!-- formula-not-decoded -->

By Lemma B.3, there exists an absolute constant C &gt; 0 such that with probability at least 1 -δ the second term is at most CK 2 √ ( t +1)log ( 2 d δ ) n , concluding the proof.

Lemma C.3. Under Assumptions 1, 2, for any t ∈ N , if θ (0) , . . . , θ ( t ) ∈ B r ( θ (0) ) , then there exists an absolute constant C &gt; 0 such that for any δ &gt; 0 , with probability at least 1 -δ

<!-- formula-not-decoded -->

Proof. By C.1, I ( θ ) is K 3 Lipschitz in θ , so

<!-- formula-not-decoded -->

The proof now follows immediately from Lemma C.2.

We now prove the following proposition, which will serve a substantial role in the proof of Theorem 4.1.

Proposition C.1. Under Assumptions 1 - 3, there exist constants c := c ( K 1 , K 2 , K 3 , λ 0 , r ) &gt; 0 and C := C ( K 1 , λ 0 ) &gt; 0 and a constant C 2 := C 2 ( K 2 , K 3 ) given by Lemma C.2 such that for any t ∈ N , if max j ≤ t -1 ∥ ∥ θ ( j ) -θ (0) ∥ ∥ ≤ max ( λ 0 4 C 2 , r/ 2 ) , then for any δ &gt; 0 , and n ≥ c log ( 4 d δ ) , with probability at least 1 -δ

<!-- formula-not-decoded -->

Proof. Fix some a &gt; 0 that will be specified later, and let S a := S a ( θ ( t -1) ) be the sphere of radius a with center at θ ( t -1) . We will show that for sufficiently small a , with high probability it will hold simultaneously for all θ on the sphere S a that ∑ t -1 j =0 ℓ j ( θ ) &gt; ∑ t -1 j =0 ℓ j ( θ ( t -1) ) . As a result, with high probability, there must be a local minimum of ∑ t -1 j =0 ℓ j ( θ ) within the ball of radius a centered at θ ( t -1) . This implies 2 that ∥ ∥ θ ( t +1) -θ ( t ) ∥ ∥ ≤ a .

Assume for now that a is small enough such that S a ⊆ B r ( θ (0) ) . We will later ensure this explicitly by picking a &lt; r/ 2 (which is sufficient due to the assumption that ∥ ∥ θ ( t -1) -θ (0) ∥ ∥ ≤ r/ 2 ).

We first Taylor expand the normalized negative log-likelihood around θ ( t -1) ,

<!-- formula-not-decoded -->

where Q ( θ ) is the quadratic term, given by

<!-- formula-not-decoded -->

and R ( θ ) is the remainder term, which for some ˜ θ between θ and θ ( t -1) satisfies

<!-- formula-not-decoded -->

where the last inequality follows from 2. C, and by the convexity of B r ( θ (0) ) which implies that ˜ θ ∈ B r ( θ (0) ) .

For the linear term, first note that if t ≥ 2 then θ ( t -1) is a stationary point of ∑ t -2 j =0 ℓ j ( · ) , so ∑ t -2 j =0 ∇ ℓ j ( θ ( t -1) ) ⊤ = 0 . So for any t ∈ N , ∑ t -1 j =0 ∇ ℓ j ( θ ( t -1) ) = ∇ ℓ t -1 ( θ ( t -1) ) . Using this and Lemma B.2, there exists a constant C 1 := C 1 ( K 1 ) &gt; 0 such that with probability at least 1 -δ/ 2 ,

<!-- formula-not-decoded -->

For the quadratic term, since the matrix ∇ 2 ℓ j ( θ t -1 ) and the Fisher information matrices are symmetric, we have by Weyl's inequality, Assumption 3 and Lemma C.2 that for C 2 = C 2 ( K 2 , K 3 ) &gt; 0 it holds with probability at least 1 -δ/ 2 that

<!-- formula-not-decoded -->

2 Here we use that if the argmax in the definition of MLE is not unique, it chooses the parameters closest to θ ( t -1) . See the discussion following Def. (3.1) for more details.

Plugging Eq. (10) back into the quadratic term given by Eq. (7) and using the assumption that max j ≤ t -1 ∥ ∥ θ ( j ) -θ (0) ∥ ∥ ≤ λ 0 / (4 C 2 ) we have

<!-- formula-not-decoded -->

where the last inequality follows by assumption.

Now take a = 8 C 1 tλ 0 √ log ( 4 d δ ) n . We can choose some constant c := c ( K 1 , K 2 , K 3 , λ 0 , r ) &gt; 0 (independent of t ) such that for any n ≥ c log ( 4 d δ ) , all of the following hold:

<!-- formula-not-decoded -->

The first condition was needed at the beginning of the proof. The second condition will allow us to bound Eq. (11), since together with the choice of a it ensures that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The third condition on a ensures that the remainder term from Eq. (8) is negligible, as

<!-- formula-not-decoded -->

Notice that the choice of a ensures that the bound for the linear term in Eq. (9) becomes

<!-- formula-not-decoded -->

So overall, the Taylor expansion Eq. (6) satisfies

<!-- formula-not-decoded -->

So we have shown that for a = 8 C 1 tλ 0 √ log ( 4 d δ ) n and n ≥ c log ( 4 d δ ) , it holds with probability at least 1 -δ that for all θ ∈ S a , ∑ t -1 j =0 ℓ j ( θ ) &gt; ∑ t -1 j =0 ℓ j ( θ ( t -1) ) . This implies the desired result as discussed at the beginning of the proof.

## C.2 Lemmas for Theorem 4.1

Lemma C.4. Under Assumption 2, for any t ∈ N , if there exists some open ball B ⊆ Θ such that θ ( t ) , θ ( t +1) ∈ B , then there exists a matrix R t ∈ R d × d with ∥ R t ∥ ≤ t +1 2 K 3 ∥ ∥ θ ( t +1) -θ ( t ) ∥ ∥ such that

<!-- formula-not-decoded -->

As a result, Eq. (11) becomes

Proof. Fix some coordinate i ∈ [ d ] and consider the Taylor expansion of ∂ ∂θ i ∑ t j =0 ℓ j around θ ( t ) , which gives that for some z i ∈ R d that lies in the line segment between θ ( t ) and θ ( t +1) ,

<!-- formula-not-decoded -->

where z i ∈ B (and in particular, z i ∈ Θ ). Let R t ∈ R d × d be the matrix whose coordinates are given by [ R t ] i,k := 1 2 ∑ t j =0 ∑ d r =1 ∂ 3 ∂θ r ∂θ k ∂θ i ℓ j ( z i )( θ ( t +1) -θ ( t ) ) r . Then Eq. (13) implies

<!-- formula-not-decoded -->

It remains to bound ∥ R t ∥ . By Assumption 2. C, we have

̸

<!-- formula-not-decoded -->

2

Lemma C.5. Let A,B ∈ R d × d be positive definite matrices, then

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

## D Proof of Theorem 4.1

Theorem 4.1. Under Assumptions 1 - 3, there exist constants c, C &gt; 0 which depend only on K 1 , K 2 , K 3 , λ 0 and r , such that for any T ∈ N , δ &gt; 0 and any n ≥ c (log( T ) + 1) 2 log 2 ( 7 dT δ ) , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Proof. Let C 1 := C 1 ( K 1 , K 2 , K 3 , λ 0 , r ) &gt; 0 denote the maximum of the constants appearing in the statements of Lemmas B.1, B.2, B.3, B.4, C.2, C.3 and Proposition C.1, and let δ 0 , . . . , δ T &gt; 0 be given by δ t := δ/ (2 T ) for t &lt; T and δ T = δ/ 2 . Let C and c be constants as in the theorem statement, whose values will be determined throughout the proof, and set

<!-- formula-not-decoded -->

We will show inductively on t = 0 , . . . , T that for any n ≥ N , it holds with probability at least 1 -1 2 tδ 0 -1 2 ∑ t j =1 δ j that

<!-- formula-not-decoded -->

Note that in the case of t = T , the probability of Eq. (15) holding becomes 1 -1 2 Tδ 0 -1 2 ∑ T j =1 δ j ≥ 1 -δ and the theorem follows.

For t = 0 the claim is trivial. Now, assume Eq. (15) holds for t -1 , and we will prove it holds for t . By Eq. (15), for sufficiently large c and the assumption that n ≥ N , the conditions of Proposition C.1 are satisfied (if Eq. (15) is not &lt; λ 0 4 C 2 as Proposition C.1 requires, one can replace c by a suitable larger constant that depends on the same parameters), so it implies that with probability at least 1 -δ 0 / 6 (using the union bound),

<!-- formula-not-decoded -->

We let A 1 denote the event that Eq. (15) and Eq. (16) indeed hold. By the union bound, P ( A 1 ) ≥ 1 -1 2 ( t -1) δ 0 -1 2 ∑ t -1 j =1 δ j -δ 0 / 6 .

Consider some τ ∈ { 0 , . . . , t -1 } . θ ( τ +1) is defined as the MLE on X ( ≤ τ ) , which in particular means that it is a stationary point of the log-likelihood function, so ∑ τ j =0 ∇ ℓ j ( θ ( τ +1) ) = 0 . When A 1 occurs, the conditions of Lemma C.4 are satisfied, which gives us a Taylor expansion for ∑ τ j =0 ℓ j ( θ ( τ +1) ) as

<!-- formula-not-decoded -->

where R τ is a matrix that satisfies by Eq. (16)

<!-- formula-not-decoded -->

(where again the last inequality assumes c is sufficiently large; if not, increase it).

By definition, for any τ &gt; 0 , θ ( τ ) is the MLE for X ( ≤ τ -1) , so it is a stationary point satisfying ∑ τ -1 j =0 ∇ ℓ j ( θ ( τ ) ) = 0 . For notational simplicity, let H τ := ( ∑ τ j =0 ∇ 2 ℓ j ( θ ( τ ) ) ) + R τ , then Eq. (17) simplifies to

<!-- formula-not-decoded -->

To isolate θ ( τ +1) -θ ( τ ) we first show that H τ is invertible. By Lemma C.3, with probability at least 1 -δ 0 / (6 t ) ,

<!-- formula-not-decoded -->

where the second inequality follows from Eq. (15) and that δ t = δ 0 for τ &lt; T . Let A 2 denote the even that Eq. (20) is indeed satisfied for all τ ∈ { 0 , . . . , t -1 } , which by the union bound satisfies P ( A 2 ) ≥ 1 -δ 0 / 6 . When both A 1 and A 2 occur, using Weyl's inequality, Eq. (20) and Eq. (18) we have,

<!-- formula-not-decoded -->

where the last inequality follows for sufficiently large c and the condition that n ≥ N . In particular, under these events, every H τ is invertible so Eq. (19) implies

<!-- formula-not-decoded -->

Taking a telescopic sum, we obtain

<!-- formula-not-decoded -->

It remains to bound the terms in Eq. (22). We will first employ an additional probabilistic bound for the gradient terms. By Lemma B.1, with probability at least 1 -δ t / 2

<!-- formula-not-decoded -->

Similarly, by Lemma B.2 and the union bound, it holds with probability at least 1 -δ 0 / 6 that

<!-- formula-not-decoded -->

Let A 3 denote the event that Eq. (23) and Eq. (24) are satisfied. Then letting A := A 1 ∩ A 2 ∩ A 3 be the intersection of the desired events in this proof, we have P ( A ) ≥ 1 -1 2 tδ 0 -1 2 ∑ t j =1 δ j as desired.

Under the event A , from Eq. (18, 20, 21) and Lemma C.5, it holds for all τ ∈ { 0 , . . . , t -1 } that

<!-- formula-not-decoded -->

Combining Eq. (24), Eq. (25) and the fact that ∑ t τ =1 1 τ ≤ 1 + ∫ t 1 1 x dx ≤ 1 + log( t ) , we have for a suitable C ′ = C ′ ( K 1 , K 2 , K 3 , λ 0 , r )

<!-- formula-not-decoded -->

where ( ⋆ ) follows whenever √ c ≥ C ′ by the assumption that

<!-- formula-not-decoded -->

Using this and Eq. (23), Eq. (22) reduces to

<!-- formula-not-decoded -->

Taking a suitable C gives the desired bound. Lastly, for the induction we also need ∥ ∥ θ ( t ) -θ (0) ∥ ∥ ≤ r 2 . This is indeed the case, taking sufficiently large c .

## E Proof of Theorem 5.1

Construction 1. Consider a fixed N ∈ N and let

<!-- formula-not-decoded -->

Let X = R , Θ = { ( α, µ ) | α ∈ [ 0 , 1 4 ] µ ∈ [2 , 3 -f ( α )] } . Letting U denote the uniform distribution, we define the family of distributions given by:

<!-- formula-not-decoded -->

Equivalently, letting I denote the indicator function (where for any set A , I A ( x ) is 1 if x ∈ A and 0 otherwise), the PDFs p θ are given by:

<!-- formula-not-decoded -->

As such,

<!-- formula-not-decoded -->

Lemma E.1. Under Construction 1, P Θ is a TV-consistent family of distributions and θ ( t ) exist.

Proof. Consider some dataset X ⊆ X of size k ∈ N , there is a finite number of values that p θ ( x ) can take, depending on the interval x lies in. This means,

<!-- formula-not-decoded -->

As such, there must be some θ that achieves this maximum.

Consistency of the MLEs follows from Lemma F.10

Theorem 5.1. There exists Θ ⊆ R 2 and θ ⋆ ∈ Θ , such that for any n ∈ N , there is a TV-consistent family of distributions { p θ } θ ∈ Θ (that may depend on n ) such that

1. with probability at least 1 -1 n ,

<!-- formula-not-decoded -->

2. For some absolute constants c, C &gt; 0 , it holds with probability at least c that

<!-- formula-not-decoded -->

Proof. Consider the setting given by Construction 1 with N = n and let θ (0) = ( α (0) = 0 , µ (0) = 2) . Existence of θ (1) and θ (2) as well as TV-consistency of P Θ are given by Lemma E.1.

Because α (0) = 0 , p θ (0) is supported on [0 , 1] , meaning that x (0) i ∈ [0 , 1] for every i ∈ [ n ] . As such,

<!-- formula-not-decoded -->

where the last equality used log ( 1 2 + 1 -α 2(1 -2 α ) ) = log ( 1 2 ( 1 + 1 -α 1 -2 α )) = log ( 1 2 ) + ( 1 + 1 -α 1 -2 α ) , and that x (0) i ∈ [0 , 1] .

Let x max := max i ∈ [ n ] x (0) i . Note that whenever α ≤ 1 -x max 2 , then every x (0) i is inside the interval [0 , 1 -2 α ] . Consequently, for all α ∈ [ 0 , 1 -x max 2 ] , ℓ 0 ( θ ) = -log ( 1 2 ) -log ( 1 + 1 -α 1 -2 α ) . Since the function -log ( 1 + 1 -α 1 -2 α ) is monotonically decreasing in α for all α &lt; 1 2 , ℓ 0 ( θ ) is also monotonically decreasing on [ 0 , 1 -x max 2 ] . As such, the MLE θ (1) = ( α (1) , µ (1) ) which minimizes ℓ 0 ( θ ) must satisfy

<!-- formula-not-decoded -->

Consistency of θ (1) : By Eq. (28) and Lemma F.8 for any δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Now using this and that p θ (0) = I [0 , 1] ( x ) , the total variation can be bounded as

<!-- formula-not-decoded -->

Inconsistency of θ (2) : We will now show that with some constant probability, there will be some x (1) i ∈ [2 , 3] . Let A denote the event that x max ≤ 1 -1 n . Since x (0) i ∼ U ([0 , 1]) i.i.d, we have

<!-- formula-not-decoded -->

Conditioned on A , we have α (1) ≥ 1 -x max 2 ≥ 1 2 n , so for each x (1) i ∼ p θ (1) ,

<!-- formula-not-decoded -->

Therefore, the probability that none of the x (1) i fall in [2 , 3] is at most

<!-- formula-not-decoded -->

Applying the law of total probability,

<!-- formula-not-decoded -->

Thus, with constant probability, one of the samples x (1) i lies in [2 , 3] . The remainder of the proof is conditioned on this occurring. We will now show that the existence of x (1) i ∈ [2 , 3] implies that α (2) will be far from α (0) = 0 .

Now consider any α ∈ [0 , 1 / 10] . The function f (defined in Eq. (26)) satisfies f ( α ) = 1 39 for any such α . As such, the term α 4 ( 1 + 1 f ( α ) ) is at most 1 for any α ∈ [0 , 1 / 10] . Consequently, for any α ∈ [0 , 1 / 10] , the only term in Eq. (27) that is negative is the first one, meaning for any x we have

<!-- formula-not-decoded -->

Since this bound is monotonically decreasing in α , we have for any α ∈ [0 , 1 / 10] ,

<!-- formula-not-decoded -->

Now let ¯ α = 1 / 8 and fix ¯ µ such that there is at least one sample in [¯ µ, ¯ µ + f (¯ α )] (which we know exists as there is some x (1) i ∈ [2 , 3] ). Plugging this ¯ θ = (¯ α, ¯ µ ) into Eq. (27), using that the first term

is negative, and that there is at least one x (1) i ∈ [2 , 3] , we have:

<!-- formula-not-decoded -->

As such, we have shown that α (2) / ∈ [0 , 1 / 10] . As a result, the TV distance can be lower bounded as

<!-- formula-not-decoded -->

## F Proof of Theorem 5.2

Construction 2. Let

<!-- formula-not-decoded -->

namely, the set of all countable tuples in [0 , 1 / 4] N ∪{ 0 } that have a finite number of non zero entries. For any α ∈ A , let

<!-- formula-not-decoded -->

where we use the notational convention that ∏ -1 k =0 α k = 1 .

To see that this is a valid PDF , first note that ∫ ∞ -∞ h α ( x ) dx = ∑ ∞ j =0 (1 -α j ) ( ∏ j -1 k =0 α k ) . Now consider any fixed M ∈ N , then

<!-- formula-not-decoded -->

In particular, since α k ∈ [0 , 1 / 4] , this converges to 1 as M →∞ .

Let f : [2 , ∞ ) → (0 , 1 / 2) be a monotonically decreasing function that will be specified later in the proof. We also define for any β ∈ [0 , 1] and J ∈ N ,

<!-- formula-not-decoded -->

The parameters θ will consist of tuples ( α , β, J, s ) where s ∈ { 0 , 1 } is a "selector" which tells us if we should choose the PDF h α or the PDF g β,J . Specifically, the parameter space is Θ = A × [0 , 1] × ( N \ { 1 } ) ×{ 0 , 1 } . And the distributions P Θ are given by

<!-- formula-not-decoded -->

Consider the ground truth distribution θ (0) to be such that

<!-- formula-not-decoded -->

For each t , θ ( t ) is an MLE given the data X ( ≤ t ) . Existence will be guaranteed in Lemma F.1. Regarding uniqueness, we do not use the fact that θ ( t ) is the closest maximizer of the log likelihood to θ ( t -1) . This is completely unimportant to the proof.

Lastly, for convenience, let

<!-- formula-not-decoded -->

where the maximum exists because the set is finite. In words, M t,j denotes the maximal observed offset within the j 'th interval [ j, j +1] up to time t .

Remark 1. Under Construction 2, for any x ∈ X , if there is some non-negative integer j ( x ) such that x ∈ [ j ( x ) , j ( x ) + 1 -2 α j ( x ) ] , then

<!-- formula-not-decoded -->

If no such j ( x ) exists, then h α ( x ) = 0 and log ( h α ( x )) is undefined.

Lemma F.1. Under Construction 2, P Θ is a TV-consistent family of distributions and θ ( t ) exist.

Proof. Consider some dataset X of size k and fix b ∈ N such that X ⊆ [0 , b ] Following Remark 1 as well as the definition of g β,J , it is straightforward to see that for any x i , there is a finite number of values that p θ ( x ) can take, depending on the interval x lies in. This means,

<!-- formula-not-decoded -->

As such, there must be some θ that achieves this maximum.

Note that any PDF in P Θ has finite support. So w.l.o.g we may assume that supp ( p θ ( ⋆ ) ) ⊆ [0 , b ] so that for any n , samples x 1 , . . . , x n from p θ ( ⋆ ) will all be in [0 , b ] .

By Remark 1, for any θ ∈ Θ and j ≥ b , the parameters α j do not affect the log likelihood. Furthermore, since g β,J = 1 2 J I [0 ,J ] ( x ) the log likelihood is strictly decreasing in J for ∀ J ≥ b +1 . As such, for the purpose of showing TV-consistency, we may "discard" all values of J ≥ b +1 and all indices ≥ J +1 in α , treating Θ as [0 , 1 4 ] J +1 × [0 , 1] × 2 , . . . , J +1 ×{ 0 , 1 } . This is a closed and bounded subset of a Euclidean space and is therefore compact. Furthermore, log p θ (0) ( x ) are uniformly bounded. So by Lemma F.10, the MLE is consistent.

Lemma F.2. For any t ∈ N ∪ { 0 } with s ( t ) = 0 , and any j ∈ N ∪ { 0 } , if M t,j &gt; 0 then

<!-- formula-not-decoded -->

Proof. This is a direct consequence of Remark 1. Specifically, from Eq. (29) it follows that the log likelihood is strictly increasing in α ( t +1) j , and is subject to the constraint that for all x ∈ [ j, j +1] it holds that x ∈ [ j, j +1 -2 α ( t +1) j ] . In particular, this implies that any maximizer must satisfy

<!-- formula-not-decoded -->

which is equivalent to what we needed to show.

The following lemma will be used throughout. It shows that for any interval [ j, j +1] , once there is some x ( t ) i ∈ [ j, j +1] , the values of M t,j and α ( t +1) j will remain the same in future iterations, as long as the MLE takes the form h α .

Lemma F.3. Under Construction 2, for any j ∈ N ∪ { 0 } if there exists some t j ∈ N ∪ { 0 } with M t j ,j &gt; 0 , then ∀ t &gt; t j , if s ( t j +1) , . . . , s ( t ) = 0 ,

<!-- formula-not-decoded -->

Proof. We prove the claim by induction on t . The case of t = t j is trivial.

Now, assume the claim holds for some time t -1 . Then α ( t ) j = α ( t j +1) j , so following Remark 1, all new samples X ( t ) that are inside the interval [ j, j +1] must also be inside the interval

<!-- formula-not-decoded -->

where the last equality follows from Lemma F.2. Hence, no new sample in [ j, j +1] can exceed j + M t j ,j , which by the induction hypothesis was already the maximum. Thus

<!-- formula-not-decoded -->

and applying Lemma F.2 again gives

<!-- formula-not-decoded -->

This completes the induction.

Lemma F.4. Under Construction 2, let j, t ∈ N and u := (1 -α ( t ) j ) ∏ j -1 k =0 α ( t ) k &gt; 0 . For any δ ∈ (0 , 1) let

<!-- formula-not-decoded -->

let B denote the event that ∃ i ∈ [ n ] s.t x ( t ) i ∈ [ j, j +1] and for any q ∈ N , let A q denote the event that ∣ ∣ ∣ { i ∈ [ n ] | x ( t ) i ∈ [ j, j +1] } ∣ ∣ ∣ ≥ q . Then if s ( t ) = 0 ,

<!-- formula-not-decoded -->

Proof. By construction, for any i ∈ [ n ] , if s ( t ) = 0 (so that p θ ( t ) is of the form h α ( t ) ) it holds that

<!-- formula-not-decoded -->

Let b i be 1 if x ( t ) i ∈ [ j, j + 1] and 0 otherwise. Then conditioned on θ (1) , . . . , θ ( t ) , b i are i.i.d. Bernoulli random variables with parameter u , so applying Lemma F.7 completes the proof.

Lemma F.5. Under Construction 2, let j ∈ N ∪ { 0 } and suppose that there exists some t j such that M t j , 0 , . . . , M t j ,j &gt; 0 . Let u := (1 -α ( t j +1) j +1 ) ∏ j k =0 α ( t j +1) k . For any δ ∈ (0 , 1) , letting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then it holds with probability at least 1 -δ that either s ( t ) = 1 for some t ∈ { t j +1 , . . . , t j +1 } or

<!-- formula-not-decoded -->

and

Proof. For any k ∈ { 0 , . . . , j } , by the assumptions that M t j ,k &gt; 0 , Lemma F.3 states that for all t ≥ t j , if s ( t j +1) , . . . , s ( t +1) = 0 then α ( t +1) k = α ( t j +1) k = 1 2 (1 -M t j ,k ) &gt; 0 .

For any t, i let b t,i be the bernoulli random variables that take the value of 1 if x ( t ) i ∈ [ j +1 , j +2] and 0 else. By Remark 1, b t,i are Ber ( u ) random variables. Let A t denote the event that ∀ b t,i = 0 . For any t ≥ t j +1 , x ( t ) i are i.i.d. when conditioned on θ (1) , . . . , θ ( t ) , and when s ( t ) = 0 , we get

<!-- formula-not-decoded -->

Notice in particular that A t depends only on θ (1) , . . . , θ ( t j +1) and s ( t j +1) , . . . , s ( t ) . So applying this argument inductively for each t ∈ { t j +1 , . . . , t j +1 } we get that

<!-- formula-not-decoded -->

where the last inequality follows from the choice of t j +1 and the fact that α ( t j +1) j +1 ≤ 1 / 4 .

By Lemma F.3, if M t,j +1 &gt; 0 for some t ∈ { t j + 1 , . . . , t j +1 } and if s ( t ) , . . . , s ( t j +1 = 0 then M t j +1 ,j +1 &gt; 0 . In summary, we have given the lower bound on M t j +1 ,j +1 needed for the lemma with probability at least 1 -δ/ 2 .

We now move on to the upper bound of the lemma. Suppose that there exists a τ which is the first timestep for which M τ,j +1 &gt; 0 or s ( τ ) = 1 . If s ( τ ) = 1 we are done, so assume it is 0 . Let B denote the event that ∃ i ∈ [ n ] s.t x ( τ ) i ∈ [ j + 1 , j + 2] and let A denote the event that ∣ ∣ ∣ { i ∈ [ n ] | x ( τ ) i ∈ [ j +1 , j +2] } ∣ ∣ ∣ ≤ q . We want to bound M τ,j +1 , where we must condition on the fact there is at least one sample at time τ that reached interval j . Recall that by Lemma F.3, α ( τ ) k = α ( t j +1) k for all k ≤ j . By Lemma F.4, using our choice of q we obtain

<!-- formula-not-decoded -->

Now suppose that this event indeed holds, so there are at most q samples that land inside the interval [ j +1 , j +2] at time τ . Since x ( τ ) i are i.i.d. (when conditioned on θ (1) , . . . , θ ( τ ) ) those that land in interval [ j + 1 , j + 2] are distributed within the interval as i.i.d. uniform random variables on [ 0 , 1 -2 α ( τ ) j +1 ] (which is included in [0 , 1] ), so letting z 1 . . . , z q be i.i.d. uniform random variables on [0 , 1] , by Lemma F.8 it holds that

<!-- formula-not-decoded -->

So overall, the desired bounds hold with probability at least 1 -δ .

Lemma F.6. Under Construction 2, for any J ∈ N and for any δ ∈ (0 , 1) , let

<!-- formula-not-decoded -->

where C &gt; 0 is some absolute constant. Then with probability at least 1 -δ , there exists some t ≤ t J for which at least one of the following holds:

1. s ( t ) = 1 .
2. M t J ,J &gt; 0 .

Proof. We begin by analyzing θ (1) . By Lemma F.2,

<!-- formula-not-decoded -->

By construction, p θ (0) is the uniform distribution on [0 , 1] , so M 0 , 0 is the maximum of n i.i.d. standard uniform random variables. We thus use Lemma F.8 to bound M 0 , 0 ; so it holds with probability at least 1 -δ/ 2 that

<!-- formula-not-decoded -->

By Lemma F.3, this also means that for every t ≥ 0 ,

<!-- formula-not-decoded -->

If s (1) = 1 we are done. Assume not. We now move on to bounding α j for j &gt; 0 . Set t 0 := 0 , and for every j ∈ [ J ] we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the q j defined here is slightly larger than the one defined in Lemma F.5 as (1 -α ( t j +1) j +1 ) &lt; 1 . By Lemma F.5 and Lemma F.3 for any j ∈ [ J ] , if M t j -1 , 0 , . . . , M t j -1 ,j -1 &gt; 0 , with probability at least 1 -δ/ (2 J ) , either there exists some t ≤ t j with s ( t ) = 1 or

<!-- formula-not-decoded -->

Note that by Lemma F.3, the same bound holds for any t ≥ t j such that s ( t j ) , . . . , s ( t ) = 0 . It was already shown for j = 0 that M t 0 , 0 &gt; 0 , so for each j ∈ [ J ] , conditioning on t 0 , . . . , t j -1 it holds with probability at least 1 -jδ/ (2 J ) that either there is some t ≤ t j with s ( t ) = 1 , or the bound on M t j ,j given in Eq. (32) holds. Applying the union bound, this is true for all j ∈ [ J ] with probability at least 1 -δ/ 2 . From now suppose that for all t ≤ t J , s ( t ) = 0 (otherwise we are done).

We now move to bounding the q j terms. Using the bounds on α ( t j +1) 0 from Eq. (30), we have

<!-- formula-not-decoded -->

for some suitable constant C ′ &gt; 0 , where we used that α ( t j +1) k ∈ [0 , 1 / 4] .

As such, by Lemma F.2 and Lemma F.3, it holds for all j ∈ [ J ] that

<!-- formula-not-decoded -->

So using this bound, Eq. (30), and taking C = max(8 C ′ , 1) , for any j ∈ [ J ] , the product can be bounded as

<!-- formula-not-decoded -->

and

Overall, Eq. (31) leads to

<!-- formula-not-decoded -->

The right-hand side is a geometric series of the form ∑ J j =1 r j for r &gt; 2 . Furthermore, for any r ≥ 2 a geometric series satisfies ∑ J j =1 r j ≤ 2 r J . Using this, we obtain

<!-- formula-not-decoded -->

Replacing C by a suitable larger constant C , this can be upper bounded as

<!-- formula-not-decoded -->

Theorem 5.2. Let ϕ : (0 , ∞ ) → (0 , ∞ ) be any strictly monotonically increasing function such that lim n →∞ ϕ ( n ) = ∞ . Then there exists an absolute constant C &gt; 0 , a set Θ , θ ⋆ ∈ Θ and a TV-consistent family of distributions P Θ (which depends on ϕ ), such that for any δ ∈ (0 , 1) , n ∈ N , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Proof. TV-consistency of P Θ and existence of θ ( t ) for any t ∈ N are given by Lemma F.1.

For any J &gt; 1 , and x ∈ [0 , 1] , g β,J ( x ) = 1 2 J ≤ 1 4 but p θ (0) ( x ) = 1 . So the TV distance between any g β,J ( x ) and p θ (0) ( x ) is lower bounded as

<!-- formula-not-decoded -->

As such, it suffices to show that there exists some time T such that with the desired probability, g β,J is chosen as the MLE.

As mentioned in Remark 1, if there is some non-negative integer j ( x ) such that x ∈ [ j ( x ) , j ( x ) + 1 -2 α j ( x ) ] , then

<!-- formula-not-decoded -->

If no such j ( x ) exists, then log ( h α ( x )) is undefined. Note that the first term is increasing in α j ( x ) and the second is negative (because α k ≤ 1 / 4 ). As such,

<!-- formula-not-decoded -->

For the PDFs g β,J , we have

<!-- formula-not-decoded -->

We now show that if T and J are such that T is sufficiently large and there is some sample in the interval [ J -1 , J ] , then a function of the form g β,J will be the MLE. We will show that there exist some g β,J for which the log likelihood is bigger than for all PDFs of the form h α . The existence of the MLE implies that there must be some function of the form g β,J that is the MLE.

Now fix any J which will be specified later, and suppose momentarily that for some T ∈ N , M T,J -1 &gt; 0 (meaning that there exists some sample in [ J -1 , J ] ).

Let β J := 1 -M T,J -1 + 1 2 f ( J ) such that J -β J = J -1 + M T,J -1 -1 2 f ( J ) and as such, by the definition of M T,J -1 there must be some point in [ J -β J , J -β J + f ( J )] . Note that for any J ∈ N , since f is assumed to satisfy f ( J ) ≤ 1 2 it holds that log ( 1 2 f ( J ) ) ≥ 0 , and thus

<!-- formula-not-decoded -->

In particular, to ensure the log likelihood of g β J ,J is bigger than for any h α , it suffices for the right-hand side of Eq. (35) upper bound the right-hand side of Eq. (34). So we want:

<!-- formula-not-decoded -->

Taking the exponential of both sides and rearranging, the above is equivalent to

<!-- formula-not-decoded -->

To that end, by Lemma F.6, for some absolute constant C &gt; 0 , letting

<!-- formula-not-decoded -->

it holds with probability at least 1 -δ , that either s ( t ) = 1 for some t ≤ T or M T,J -1 &gt; 0 . If the first holds, we are done, so assume the latter.

Now, for any strictly monotonically increasing function ϕ : (0 , ∞ ) → (0 , ∞ ) with lim n →∞ ϕ ( n ) = ∞ , let

<!-- formula-not-decoded -->

Then to ensure Eq. (36) is satisfied, we need ϕ -1 ( ψ ( J )) ≥ n , or equivalently, J ≥ max ( ψ -1 ( ϕ ( n )) , 2 ) (where the 2 is because our domain includes only J ≥ 2 ). In particular, we take J := max ( ⌈ ψ -1 ( ϕ ( n )) ⌉ , 2 ) . If J = 2 , T = ψ (2) = C log ( 4 δ ) δ , and otherwise we can bound T as

<!-- formula-not-decoded -->

In summary, we have shown that at some timestep up to T , it holds with probability at least 1 -δ that the TV distance is at least 3 / 8 .

## F.1 Auxiliary Lemmas

Lemma F.7. For all i ∈ [ n ] let b i ∼ Ber ( u ) be i.i.d. Bernoulli random variables with parameter u . Then for any δ ∈ (0 , 1) ,

<!-- formula-not-decoded -->

Proof. Since b i are i.i.d., using the inequality 1 -x ≤ exp( -x ) for any x ,

<!-- formula-not-decoded -->

By Chernoff's inequality (c.f. [Vershynin, 2018]) and the chain rule of probability for any q &gt; un ,

<!-- formula-not-decoded -->

where the last inequality uses that e -x 1 -e -x ≤ 1 /x for any x &gt; 0 . Now we split into two cases depending on un . First, if un ≤ 4 e 2 δ &lt; 1 , for any q ≥ 2 , Eq. (38) becomes

<!-- formula-not-decoded -->

On the other hand, if un &gt; 4 e 2 δ , taking q ≥ 2 + e 2 un +2log ( 1 δ ) (which in particular ensures that ( eun q ) ≤ 1 /e and q ≥ 2 + 2 log ( 1 δ ) ), Eq. (38) becomes

<!-- formula-not-decoded -->

In either case, q ≥ 2 + e 2 un +2log ( 1 ) suffices to ensure that desired bound.

δ

Lemma F.8. For n ∈ N let x 1 , . . . , x n ∼ U ([0 , 1]) be i.i.d. uniform [0 , 1] random variables.

1. For any δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

2. For any δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

As a result, for any δ &gt; 0 , it holds with probability at least 1 -δ that

<!-- formula-not-decoded -->

Proof. Since x i are i.i.d., the CDF of max i ∈ [ n ] x i is given by

<!-- formula-not-decoded -->

To prove Eq. (39), by Bernoulli's inequality (1 -u ) n ≥ 1 -un , so it suffices to take u = δ n .

To prove Eq. (40), we use the well known inequality 1 -u ≤ exp( -u ) to obtain

<!-- formula-not-decoded -->

Taking u = 1 n log ( 1 δ ) completes the proof.

Eq. (41) follows from Eq. (40) and Eq. (39) with δ/ 2 and the union bound.

The following lemma gives a version of the uniform law of large numbers that is suited for TVconsistency. We note that the conditions can be made even milder (c.f. [Tauchen, 1985]), and are relatively similar to those of [Wald, 1949, Redner, 1981].

Lemma F.9 (Newey and McFadden [1994] Lemma 2.4) . Let Θ ⊆ R d be compact, θ (0) ∈ Θ , let { x i } n i =1 ∼ p θ (0) be i.i.d. and let f ( x, θ ) be a function which for any θ ∈ Θ is measurable, continuous for almost all x s, and satisfies | f ( x, θ ) | ≤ ϕ ( x ) for some function ϕ ( x ) with E x ∼ p θ (0) [ ϕ ( x )] &lt; ∞ . Then E [ f ( x, θ )] is continuous in θ and

<!-- formula-not-decoded -->

Lemma F.10. Let Θ ⊆ R d be compact, ¯ θ ∈ Θ , and assume that for any θ ∈ Θ , log( p θ ( x )) is measurable, continuous for almost all x , and satisfies | log( p θ ( x )) | ≤ ϕ ( x ) for some function ϕ ( x ) with E x ∼ p ¯ θ [ ϕ ( x )] &lt; ∞ . Then if for any n , there exists an MLE ˆ θ ( n ) with respect to n i.i.d. samples from ¯ θ , it holds that

<!-- formula-not-decoded -->

Proof. By Lemma F.9, for any δ, ϵ &gt; 0 , there is some n 0 ∈ N such that for all n ≥ n 0 ,

<!-- formula-not-decoded -->

ˆ θ ( n ) minimizes ℓ , implying ℓ ( ˆ θ ( n ) ) ≤ ℓ ( ¯ θ ) and thus with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Rearranging and using Pinsker's inequality,

<!-- formula-not-decoded -->

## G Experiments

(a) Exact MLE

<!-- image -->

<!-- formula-not-decoded -->

(b) Optimized MLE

Figure 1: MLE for a one-dimensional Gaussian distribution.

There are by now many experiments in the literature that support our finding from Theorem 4.1 [Alemohammad et al., 2024, Gerstgrasser et al., 2025, Dey and Donoho, 2024]. In particular, in those papers, the error does not increase much from iteration to iteration when synthetic data is added gradually.

Rather than repeating experiments, we analyze the difference between exact MLE solutions and those that are obtained via optimization. To this end, we pick several families whose MLE has a known closed form. These include a Gaussian (where the parameters are the mean and std) in Fig. 1, Exponential distribution in Fig. 2, and a family of Beta distributions with PDFs given by

Figure 2: MLE for a one-dimensional Exponential distribution.

<!-- image -->

Figure 3: MLE with respect to a Beta distribution family with PDFs given by p ( x ; θ ) = θx θ -1 for θ &gt; 0 and x ∈ (0 , 1) .

<!-- image -->

p ( x ; θ ) = θx θ -1 for θ &gt; 0 and x ∈ (0 , 1) in Fig. 3. The real parameters are θ 0 = ( µ = 0 , σ = 1) for the Gaussian and θ 0 = 1 for the other distributions. When optimizing numerically for the MLE, we use scipy.optimize.minimize on the negative log likelihood to find the parameters. We opt for this built-in function to remove any uncertainty regarding the quality of the optimization code itself. We take the number of samples to be one of 20 , 50 , or 100 . We run the iterative MLE algorithm as specified in the paper for up to T = 100 . All values are averaged over 50 runs. The error is measured by the norm relative to the real parameters, meaning ∥ ∥ θ ( t ) -θ 0 ∥ ∥ . In all cases, the error at all timesteps is similar to the error at time 1 , as our theory would suggest. Furthermore, we observe the model (non)-collapse behavior between the exact MLE and the optimized one to be similar.

We also consider various θ 0 going from 0 . 1 to 1 for the Beta distribution, where a smaller θ 0 corresponds to a neighborhood of the parameters that are less smooth. In all cases, we plot the ratio between the error at time T to the error at time 1 . For θ = 1 , the error increases by a factor of only 1.25 across 100 iterations, but for the 'less smooth' θ 0 = 0 . 1 , the error increases by a factor of 3.27. These confirm that our negative results hint at a more general phenomenon and support our results.

Figure 4: MLE with respect to a Beta distribution family with PDFs given by p ( x ; θ ) = θx θ -1 for θ &gt; 0 and x ∈ (0 , 1) , for various choices of real parameter θ 0 .

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] .

Justification: The abstract and introduction summarize the theorems and reference them. All claims are supported by the theorems.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in the "Discussion" section.

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

Justification: The setting and notation are discussed in detail in the "Setting and Notation" section. All theorems are rigorously proved in the appendix. We also include proof sketches in the main paper to accompany the proofs in the appendix.

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

Justification: [NA]

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

Justification: [NA]

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: [NA]

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms in every respect to the code of ethics, and we do not foresee any negative implications of our work.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Our work is completely theoretical; We do address the potential societal impacts of model collapse (e.g. in the introduction), but we do not expect any such impacts to be directly influenced by the current work.

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

Answer: [NA]

Justification: [NA]

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

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
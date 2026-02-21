## Understanding the Gain from Data Filtering in Multimodal Contrastive Learning

## Divyansh Pareek Sewoong Oh Simon S. Du

Paul G. Allen School of Computer Science and Engineering University of Washington, Seattle, WA

{dpareek,sewoong,ssdu}@cs.washington.edu

## Abstract

The success of modern multimodal representation learning relies on internet-scale datasets. Due to the low quality of a large fraction of raw web data, data curation has become a critical step in the training pipeline. Filtering using a trained model (i.e., teacher-based filtering) has emerged as a successful solution, leveraging a pre-trained model to compute quality scores. To explain the empirical success of teacher-based filtering, we characterize the performance of filtered contrastive learning under the standard bimodal data generation model. Denoting η ∈ (0 , 1] as the fraction of data with correctly matched modalities among n paired samples, we utilize a linear contrastive learning setup to show a provable benefit of data filtering: ( i ) the error without filtering is upper and lower bounded by 1 / η √ n , and ( ii ) the error with teacher-based filtering is upper bounded by 1 / √ ηn in the large η regime, and by 1 / √ n in the small η regime.

## 1 Introduction

The seminal work of Radford et al. [29] introduced CLIP, a large-scale multimodal training paradigm that leverages contrastive learning on image and language modalities. This marked a significant advancement in general purpose representation learning that enabled unprecedented zero-shot downstream performance. A crucial factor in the success of CLIP and other vision-language models (VLMs) was the shift towards training on massive datasets [39], often comprising billions of imagetext pairs scraped from the internet (e.g., LAION-5B [30] and DataComp-1B [11]). The sheer quantity of data unlocks the capability to learn robust representations [9]. However, due to the inherently noisy nature of web data, this introduces significant challenges regarding the quality , resulting in the need for data curation. Smaller but higher quality subsets of the data have been observed to result in better models than larger but noisier datasets [39, 25, 11]. Gadre et al. [11, Figure 2] observe that training on only a selected 30% of the dataset results in a better performing model than training on the full corpus. To handle such a significant fraction of low-quality data, data curation has become a critical step in modern internet-scale pretraining pipeline of foundation models [1].

For vision-language datasets, a number of methods have been introduced for data filtering [10, 35, 18, 8, 32, 22]. Among these, teacher-based filtering , where a pre-trained model is used to score samples and retain high-quality ones, has emerged as a particularly effective strategy [10, 35]. This approach marks a progression from earlier efforts which relied on heuristic-based filtering (e.g., the WIT400M dataset used in CLIP [29]). Subsequent and ongoing curation efforts have increasingly leveraged strong existing models, like CLIP itself, to refine datasets further [30, 11].

In the theory community, the success of CLIP models has been attributed to two factors: the choice of using a contrastive loss and the use of multimodal datasets. A series of modeling and analyses followed to explain the benefits from these two factors under various scenarios [24, 14, 33, 27, 17, 13, 6, 7].

However, despite the empirical successes of data filtering in the CLIP training pipeline, a theoretical understanding of this phenomenon has been lacking. Our goal is to provide a deeper understanding of the benefits of using teacher-based data filtering in the CLIP training pipeline, i.e., multimodal representation learning with a contrastive loss. In particular, we aim to understand the benefits of data filtering against the baseline of contrastive learning without filtering, by focusing on one key parameter of interest: the fraction of high-quality data present. Using η ∈ (0 , 1] to denote the fraction of high-quality data pairs within the dataset, for both the filtering and no-filtering approaches, we ask the question: How does the quality of the learned representation behave as a function of η ?

The choice of the data corruption model is crucial. In the related field of robust statistics, similar questions have been studied under adversarial corruptions. However, for large multimodal datasets, we posit that a stochastic corruption model is more relevant in capturing the nature of real data. For instance, in vision-language data, a significant portion of the misalignment arises randomly: images paired with irrelevant or tangentially related captions due to the processes of automated web scraping and the uncontrolled nature of internet data (see, e.g., [26, Figure 1] for examples). We adopt such a model (detailed in Section 3.1), where a fraction η of pairs are correctly aligned, while the remaining 1 -η fraction has mismatched modalities. Under the stochastic corruption model of Section 3.1 and the contrastive learning setup of Section 3.2, we analyze the performance of teacher-based filtering (Figure 1c) and compare against the baseline of no filtering (Figure 1a).

Contributions. We demonstrate a provable benefit of data filtering. The error of the unfiltered contrastive learning with n samples and η clean fraction depends as 1 / η √ n , as shown by an upper bound in Corollary 1 (result from Nakada et al. [24, Theorem 3.1]) and a lower bound in Proposition 1. On the other hand, for teacher-based filtering (Theorem 1, main result), the dependency on η is improved to 1 / √ ηn when η is large, and to 1 / √ n when η is small. Note that our result includes the training of the teacher model on the given dataset, i.e., we do not assume the existence of any strong pre-trained model. In Section 7, we empirically demonstrate the benefit of teacher-based data filtering in a synthetic experimental setting. Figure 3a verifies the 1 / η dependence of the unfiltered contrastive learning, and the improved dependence achieved by the teacher-based filtering in two regimes, namely 1 / √ η for large η and independent of η for small η . Figure 3b restates the finding of Fang et al. [10, Figure 4] to show that the qualitative observation of improved η dependence via filtering holds true even with real data.

## 2 Related work

Our theoretical investigation of data filtering builds upon existing analyses of multimodal contrastive learning [24, 14, 33]. In particular, Nakada et al. [24, Theorem 3.1] gives the rate for the unfiltered contrastive learning, and we study the rate with data filtering. The theory of contrastive learning (CL) has been studied in many other contexts [13, 6, 7, 17, 27]. Chen et al. [6] build a theoretical understanding for zero-shot transfer in CLIP-style models. Huang et al. [13] theoretically compare unimodal and multimodal CL, and Daunhawer et al. [7] study identifiability of the latent factors with the CL objective. We remark that the assumptions on the data generative model across these works are related but sometimes subtly different.

The practical need for data curation arises from the inherent noise in web-scale datasets used for training vision-language models [39, 25, 11] and increasingly, large language models [1, 20, 38, 34, 37]. In the multimodal context, numerous empirical techniques have been developed [10, 35, 18, 22, 8, 32], with community benchmarks like DataComp [11] facilitating systematic evaluation. Teacher-based filtering, the focus of our work, is a widely adopted and effective empirical strategy [11, 35, 39], but we note that other approaches have also been explored, in particular, editing bad data [26] (with some theoretical explanations [28, 41]). However, theoretical studies of data filtering are limited. Some works include the study of data selection under weak supervision in general statistical models [19], and selecting data during training [31].

## 3 Setup

Section 3.1 describes our model for multimodal data and the assumptions on the related parameters. Section 3.2 formulates the contrastive learning objective on data pairs from the model.

## 3.1 Bimodal data model

Building on recent theoretical work in multimodal contrastive learning [36, 14, 24], we assume the signal has a low-rank structure, while the noise is unstructured and dense. Adopting a linear generative model, the paired bimodal data, x ∈ R d and ˜ x ∈ R ˜ d , is expressed as:

<!-- formula-not-decoded -->

representing, for example, image and text in the case of vision-language data. Here z, ˜ z ∈ R r denote the latent variables lying in a shared r -dimensional space that captures the common underlying concept. The first terms U z and ˜ U ˜ z represent the signals of interest, residing in r -dimensional subspaces spanned by the columns of U and ˜ U , and the terms ξ and ˜ ξ represent the dense noise. For simplicity, we assume that the maps U ∈ R d × r and ˜ U ∈ R ˜ d × r are composed of unit-norm orthogonal columns, fixing the scale of this problem.

We say a bimodal paired example is corrupted if the individual modalities do not correspond to the same latent concept. This models how a large fraction of image-text pairs found on the internet are corrupted by arbitrary captions that are unrelated to the content of the image. We formalize this in Assumption 1, with η denoting the clean fraction. Figure 4 in Appendix A provides an illustration. For the noise, we assume a Gaussian distribution with a diagonal covariance (Assumption 2).

Assumption 1 (Corruption model) . Let z 1 , z 2 ∼ N (0 , I r ) be two independent draws from the r -dimensional standard Gaussian. For an η ∈ (0 , 1] , the joint distribution on ( z, ˜ z ) is induced by

<!-- formula-not-decoded -->

Assumption 2 (Noise model) . The noise { ξ, ˜ ξ } are mutually independent and independent of { z, ˜ z } , and are zero-mean Gaussian variables given by ξ ∼ N ( 0 , γ -1 I d ) and ˜ ξ ∼ N ( 0 , ˜ γ -1 I ˜ d ) .

The signal is unit-scale in r -dimensions since ∥ U ∥ = 1 and Cov( z ) = I r , hence the signal-to-noise ratios (SNRs) for the two modalities are γ ( r/d ) and ˜ γ ( r/ ˜ d ) respectively. This model is parametrized by ( η, U , ˜ U , γ, ˜ γ, r, d, ˜ d ) , and the aim is to recover U and ˜ U , given paired samples. This is a standard model in bimodal contrastive learning [36, 14, 24] and is inspired by the spiked covariance model [3, 40]. Consider an extreme case where the images are matched to randomly shuffled captions. This corresponds to η = 0 , and recovering the subspaces U and ˜ U becomes akin to two separate unimodal estimation problems, whose optimal (up to constants) error rate is known with tight upper and lower bounds [5, Eq. (9)]:

<!-- formula-not-decoded -->

where ERR is defined via the chordal distance between two subspaces in Eq. (4). This follows from the fact that E [ xx ⊤ ] = UU ⊤ + γ -1 I d . The √ d / n dependence is expected from the concentration, the √ r dependence comes from the error metric being chordal (frobenius norm) as opposed to projection (spectral norm), and γ -1 / 2 dependence captures how the error vanishes with high SNR. Refer to Appendix B.2 for a description of how to arrive at Eq. (2) using the result from Cai et al. [5, Eq. (9)]. When η &gt; 0 fraction of data is correctly matched, our goal is to characterize the error rate achieved by the contrastive learning on the paired data and show that data filtering can improve the error rate compared to the baseline of no filtering.

Notation. For a matrix Q = USV ⊤ and an integer a , let SVD a ( Q ) = U a S a V ⊤ a denote the projection of Q onto its topa components. Let lsv ( Q ) denote the left singular vectors of Q , and lsv a ( Q ) denote its top a left singular vectors. Similarly, let rsv ( Q ) and rsv a ( Q ) be defined for the right singular vectors. We use O ( . ) to denote asymptotic upper bounds, and ˜ O ( . ) to denote upper bounds with only η, n factors (omitting the dimension and SNR parameters). Similarly, we use the standard notation Ω( . ) , ω ( . ) to denote asymptotic lower bounds. The notation ≳ , ≲ hides absolute constants, and we write a ≍ b when a ≲ b and a ≳ b holds simultaneously. Additionally, we will sometimes use the random variable c ∼ Ber( η ) ∈ { 0 , 1 } to denote the (hidden) 'coin toss' in accordance with Assumption 1, with c = 1 denoting to the clean case.

## 3.2 Contrastive learning formulation

We utilize a linear contrastive learning framework from [24, 14]. By linear we mean ( i ) the encoders that map the data, x and ˜ x , to the shared embedding space are linear, and ( ii ) the contrastive loss computed on the embeddings is linear. This setting corresponds to the choice of ϵ = 0 , ψ = ϕ = Id maps in Nakada et al. [24, eq 2.1], an equation that captures a more general contrastive loss framework. Werefer the reader to Tian [33, Figure 1] for different contrastive learning setups achieved by different choices of ψ and ϕ .

Let G ∈ R r × d and ˜ G ∈ R r × ˜ d denote the learnable encoders for the input, x and ˜ x respectively. Figure 5 in Appendix A provides a helpful visualization. The similarity score of a pair ( x, ˜ x ) is computed as the inner product ⟨ G x, ˜ G ˜ x ⟩ , which is widely used theoretically [24, 14, 15, 33] and empirically [29, 12]. The multimodal contrastive loss maximizes the similarity of observed pairs, while minimizing the similarity of 'generated' pairs. Given n paired samples { ( x i , ˜ x i ) } n i =1 , the parameters G , ˜ G are learned by minimizing the ρ -regularized objective given by:

̸

<!-- formula-not-decoded -->

̸

where s ij := ⟨ G x i , ˜ G ˜ x j ⟩ for i, j ∈ [ n ] is the similarity score, and R ρ ( G , ˜ G ) := ( ρ/ 2) ∥ G ⊤ ˜ G ∥ 2 F is the regularizer with strength ρ &gt; 0 . The regularizer ensures that the learned parameters have finite norms. Indeed, Eq. (6) shows that this objective has a closed-form solution with a 1 / ρ multiplier, which becomes infinite if ρ = 0 . Note that CLIP [29] does not need a regularizer since the inner product is taken with normalized vectors (i.e. ( 1 / ∥ G x ∥ ) G x instead of G x ). The parameters G and ˜ G assume the knowledge of the latent dimension r (since they are of sizes r × d and r × ˜ d ). In practice, the latent dimension is typically a design choice and is therefore known at training time. Theoretically, assuming the latent dimension is known allows us to isolate the effects of data filtering from the separate, well-studied problem of subspace rank estimation (for e.g., in Cai et al. [5]).

Also note that this objective is in a full-batch setting, i.e. the entire n × n grid of similarities is computed to maximize the diagonals and minimize the off-diagonals. This does not cause computational issues since the objective has a closed-form solution, given by Eq. (6).

To measure the quality of a solution, we use the chordal distance between two subspaces in Definition 1. This is a standard measure of how well G , ˜ G recover U , ˜ U respectively [24, 14].

Definition 1. The error metric for a learned embedding G , ˜ G is defined as

<!-- formula-not-decoded -->

We note two points. First, the metric only considers the right singular vectors . This is because the essential information in G , ˜ G is contained in the right subspaces. Indeed, the loss in Eq. (3) is only affected by G ⊤ ˜ G , which is preserved under the transformation G ← A G , ˜ G ← A ˜ G for any orthonormal matrix A . Second, the metric uses the sin Θ distance, which is a geometrically intuitive way to measure closeness between two subspaces (refer to Appendix B.1 for a background).

## 4 Baseline: unfiltered contrastive learning

We study the error rate of the unfiltered contrastive learning (Figure 1a). We show that the error is upper and lower bounded by ˜ O ( 1 / η √ n ) . The upper bound is given in Corollary 1, which is a result from Nakada et al. [24]. We show a matching lower bound in Proposition 1.

Corollary 1 (Corollary of [24, Theorem 3.1]) . Given a dataset of pairs { ( x i , ˜ x i ) } n i =1 generated i.i.d. according to the bimodal data model in Eq. (1) satisfying Assumptions 1 and 2, the solution of minimizing the contrastive loss in Eq. (3) satisfies with probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

provided the number of samples n ≳ (1 /η 2 ) max { d, ˜ d } ( 1 + γ -1 ) ( 1 + ˜ γ -1 ) .

Figure 1: Our goal is to analyze the Train-Filter-Train approach illustrated in (c) and show that it improves upon the no filtering approach of (a). Here ϕ ⋆ denotes the ground-truth parameters and ˆ ϕ denotes the learned version. In our setting, ϕ ∗ ≡ { U , ˜ U } and ˆ ϕ ≡ { G , ˜ G } .

<!-- image -->

Remark 4.1 (Looseness in SNR parameters compared to Eq. (2)) . The dependence on SNR parameters ( γ, ˜ γ ) in Corollary 1 is looser than the unimodal estimation counterpart in Eq. (2) . As stated, the error upper bound in Corollary 1 does not become zero when γ →∞ . We remark that this is an artifact of the analysis. Indeed a tighter analysis is possible that recovers a √ γ -1 ˜ γ -1 term also in the upper bound, for instance, using the ideas in Cai et al. [5, Section 7], in particular [5, Eq. (39)].

A complete proof of Corollary 1 is presented in Appendix D, which is largely a reconstruction from Nakada et al. [24] with some minor corrections. The analysis has three parts. First, the unregularized term of the contrastive loss in Eq. (3) simplifies to L 0 ( G , ˜ G ) = -Tr ( GS n ˜ G ⊤ ) , where S n ∈ R d × ˜ d denotes the cross-covariance matrix of the data, defined as

<!-- formula-not-decoded -->

Second, the regularized contrastive loss, albeit nonconvex, admits a closed-form solution as the SVD of S n , given in Eq. (6). Due to this, we can directly analyze the solution without the need for optimization analysis.

<!-- formula-not-decoded -->

The third key piece is concentration of S n . We show finite sample concentration of S n in operator norm, namely w.h.p. ∥ ∥ S n -S ∥ ∥ ≲ 1 / ρ √ n , for the limiting quantity S = ( η / ρ ) U ˜ U ⊤ . Using a DavisKahan like result, we can translate the operator norm concentration to a distance between the angles of subspaces, for both left and right singular vectors, yielding w.h.p. ERR( G , ˜ G ) ≲ 1 / η √ n . Note the dependence on the regularization strength ρ vanishes (as long as ρ &gt; 0 ) due to its appearance in both the numerator (via op-norm concentration) and denominator (since the singular values of S scale as η / ρ ). This sketch describes the 1 / η dependence of the unfiltered contrastive learning. In Proposition 1, we show that this dependence is tight. We present a proof of Proposition 1 in Appendix E by constructing a hard problem instance (parameterized by η ).

Proposition 1. Under the setting of Corollary 1, there is a class of problem instances with latent dimension r = 1 such that the error achieved by the minimizer of Eq. (3) is lower bounded (up to absolute constants) with probability 1 -exp( -Ω(max { d, ˜ d } )) as:

<!-- formula-not-decoded -->

## 5 Our approach: teacher-based filtering

In the previous section, we concluded that the unfiltered contrastive learning achieves a tight error dependence of 1 / η . In this section, we ask: can filtering algorithms improve upon the η dependency? Intuitively, we expect the answer to be yes, since filtering can identify corrupted samples and remove

them (increasing the clean fraction η ). Indeed, if the filter could perfectly identify all clean samples, it would achieve a dependence of 1 / √ η (since this would be akin to the unfiltered contrastive learning with η ← 1 and n ← ηn ). We will now study the η dependence of teacher-based filtering.

Teacher-based filtering, which follows a Train-Filter-Train approach, has proven to be a successful method in practice [10, 35]. In the first training step, a teacher model is trained on (potentially a part of) the dataset. In the filter step, (the remaining part of) the dataset is filtered by using the teacher to compute a similarity score to evaluate the quality of each sample. The filtering usually happens by selecting samples with score above a certain threshold θ ∈ R . In the second training step, a student model is trained on the filtered dataset. Refer to Figure 1c for an illustration. The student can be initialized at the teacher's solution, or even at a fresh random initialization. The intuition is that the teacher can extract useful signal from the dataset despite the presence of corrupted samples, which can help in identifying and discarding corrupted samples. Algorithm 1 describes this process in the setup of Section 3. The split of the dataset into two halves is for the convenience of analysis, by ensuring the filtering rule (which depends on the first half of samples and θ ) is independent of the samples being filtered (the second n/ 2 samples). We now state our main result.

## Algorithm 1 Teacher-based filtering in the setup of Section 3.

Input: Dataset D = { ( x i , ˜ x i ) } n i =1 , Threshold θ ∈ R .

Step 1 (Train): Obtain G T , ˜ G T by minimizing Eq. (3) on the first n/ 2 samples { ( x i , ˜ x i ) } i ≤ n/ 2 .

Step 2 (Filter): Create D filt ( θ ) from { ( x i , ˜ x i ) } i&gt;n/ 2 by retaining sample i iff ⟨ G T x i , ˜ G T ˜ x i ⟩ &gt; θ .

Step 3 (Train): Output G ( θ ) , ˜ G ( θ ) by minimizing Eq. (3) on D filt ( θ ) .

Theorem 1. Under the model in Eq. (1) satisfying Assumptions 1 and 2 with r ≥ 2 , there exists a threshold θ ∗ ∈ R such that, given a dataset of pairs { ( x i , ˜ x i ) } n i =1 generated i.i.d. according to the model, the output of Algorithm 1 satisfies with probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

provided n ≳ (1 /η 2 ) max { d, ˜ d } ( 1 + γ -1 ) ( 1 + ˜ γ -1 ) . Here T 0 . 5 , T 0 are defined as

<!-- formula-not-decoded -->

We provide a full proof in Appendix G, and discuss the sketch in Section 6. Certain observations are in order. First, we see two regimes of behavior. The error behaves as 1 / √ η for large values of η , and becomes independent of η for small values of η (note that η still needs to large enough to satisfy the requirement of n ≳ 1 / η 2 for theorem to be valid). Both these regimes exhibit a better dependence on η than the unfiltered contrastive learning's rate of 1 / η . From the expressions, we note that the switch between the regimes happens at η = 1 / r 2 (up to constants). Second, this result is stated for the optimal filtering threshold θ ∗ . The optimal choice of this hyperparameter depends on the problem quantities, particularly n and η . Understanding this dependence is an interesting direction of research, but outside the scope of the current work. Our analysis considers two fixed choices of θ that recover each of the regimes. We also present a small experiment on varying the filtering threshold θ in the vicinity of θ ∗ in Appendix H. Third, we remark that it remains an interesting research question to study whether an improved dependence on η (at least something better than 1 / η ) can be achieved with a single training loop on the data (as the teacher-based filtering is a two-step training process).

It is perhaps surprising that the error can become independent of the clean fraction η , which is better than the oracle rate of 1 / √ η . This counter intuitive benefit stems from the use of the inner product to compute similarities (Section 3.2) on the corruption model given by Assumption 1. Owing to this, the distribution of the similarity scores before filtering follows a very typical structure, explained in Figure 2. Filtering can retain samples from the right tail of the noisy score distribution D 0 , and these samples provide useful signal to recover the ground-truth U parameter. Finally, we remark on the assumptions needed for this result. Assumption 2 makes this setting somewhat special, since Nakada

et al. [24] allow for a general covariance Σ ξ , Σ ˜ ξ (with bounded norms) on the noise. Handling a more general noise covariance is trivial for unfiltered contrastive learning, but significantly more challenging in the case of filtering. We argue that Assumption 2 preserves the essential characteristics of the problem though, while simplifying the analysis of filtering. In the following section, we discuss the proof ideas in more detail.

## 6 Analysis of the filtering algorithm

In this section, we describe the main ideas behind the proof of Theorem 1. In Section 6.1, we study the distribution of the scalar score used for filtering samples. In Section 6.2, we use the score characterization to understand filtering by thresholding on the scores.

## 6.1 The score used for filtering

For a sample ( x, ˜ x ) , let S ( x, ˜ x ; A ) for a matrix A ∈ R d × ˜ d denote the score of the sample, defined in Eq. (7). This scalar score is meant to capture the quality of the sample ( x, ˜ x ) . Treating ( x, ˜ x ) as a random i.i.d. sample from the model in Section 3.1, we characterize the distribution of the score. Note that the teacher-based filtering is simply using A := G ⊤ T ˜ G T to score the data (the subscript is used to denote the teacher's parameters). To understand teacher-based filtering, an intermediate step will be to understand filtering using an 'oracle' which has access to the ground-truth problem parameters (refer to Figure 1b). The oracle scores data using A := U ˜ U ⊤ , given in Eq. (8). Since G ⊤ T ˜ G T → ( η / ρ ) U ˜ U ⊤ as the number of samples n →∞ , we expect the teacher filtering to resemble the oracle filtering in the large n regime. The positive scaling factor of η / ρ does not affect threshold-based filtering, as the ordering of samples remains unchanged.

<!-- formula-not-decoded -->

Remark 6.1 (Two versions of oracle) . There are two possibilities for an 'oracle' in this setup. The first kind has access to the ground-truth problem parameters, which is what we study. The second kind has access to the clean/corrupted status of each sample. The second kind can trivially achieve an error dependence of 1 / √ ηn by choosing to only use the clean samples.

Recalling Assumption 1, since ˜ z = z for clean samples, the score in Eq. (8) is defined through the independent randomness in z, ξ, ˜ ξ . For corrupted samples, it is defined via the independent randomness in all z, ˜ z, ξ, ˜ ξ . Wecharacterize the distribution in both cases, detailed in Appendix F. The main observations are illustrated in Figure 2a. D 0 denotes the distribution of the score in the corrupted case, with mean µ ( D 0 ) = 0 (since z, ˜ z are independent), and variance σ 2 0 = r ( 1 + γ -1 ) ( 1 + ˜ γ -1 ) . Similarly, D 1 denotes the distribution in the clean case, with mean µ ( D 1 ) = r (since z = ˜ z leading to a squared term), and variance σ 2 1 = r + r ( 1 + γ -1 ) ( 1 + ˜ γ -1 ) . Note that σ 2 0 ≤ σ 2 1 ≤ 2 σ 2 0 .

Since clean and corrupted data are mixed with η, 1 -η proportions, the score of a generic sample from the population is given by the mixture distribution D := η D 1 +(1 -η ) D 0 . Figure 2 provides an illustration of the score distribution D . Due to i.i.d. data, the oracle filtering algorithm's scores are n i.i.d. draws from D . The filtering threshold θ can be picked in various ways, leading to various algorithms for filtering. The threshold θ →-∞ corresponds to no filtering.

Remark 6.2. Since σ 0 ≤ σ 1 = √ 2 r (1 + γ -1 ) (1 + ˜ γ -1 ) , the condition γ, ˜ γ = ω ( √ r ) ensures that r / σ 1 = ω (1) , leading to a separation between the modes of D 0 and D 1 . In this case, the clean and corrupted data become well-separated via the oracle score S ( x, ˜ x ; U ˜ U ⊤ ) .

## 6.2 Analysis of thresholding on the score distribution

In this section, we discuss an analysis for the oracle filtering algorithm (Figure 1b), which captures the main conceptual ideas of data filtering in the setup of Section 3. The proof for the teacher-based filtering (Theorem 1) is given in Appendix G, which uses the ideas from this section, along with the

Figure 2: Distribution of the oracle score S ( x, ˜ x ; U ˜ U ⊤ ) is given by the mixture of D 0 , D 1 with weights (1 -η ) , η respectively. Here σ 2 0 , σ 2 1 depend on parameters r, γ, ˜ γ . The threshold θ ∈ R is used to filter the datapoints (score &gt; θ are retained, others are discarded). Subfigure (b) shows the observed histogram in a synthetic setting for n = 50000 samples with r = 16 , γ = ˜ γ = 10 4 .

<!-- image -->

operator norm concentration in Corollary 1 to bound the deviation caused by the difference between the teacher scores and the oracle scores. Given a dataset { ( x i , ˜ x i ) } n i =1 , let n sel ( θ ) denote the number of samples retained after oracle filtering, and let I sel ( θ ) ⊆ [ n ] denote the indices of the samples selected, defined by the condition i ∈ I sel ( θ ) ⇐⇒ S ( x i , ˜ x i ; U ˜ U ⊤ ) &gt; θ . Analogous to Eq. (5), we define S n ( θ ) to be the empirical cross-covariance of the filtered data, given by Eq. (9).

<!-- formula-not-decoded -->

Observe that similar to Eq. (6), the closed-form solution of the optimization holds even on the filtered dataset. The step that changes is the concentration, namely, the characterization of how S n ( θ ) concentrates as n increases, according to the distributions of the involved random quantities. In the following, we argue that S n ( θ ) concentrates to S ( θ ) , given by Eq. (10), and characterize the behavior of S ( θ ) to recover a guarantee akin to Theorem 1.

Notation . We set up some useful notation on the score distributions D 0 , D 1 from Figure 2a. For any a ∈ R , let P 0 ( a ) = P Z ∼D 0 ( Z &gt; a ) and P 1 ( a ) = P Z ∼D 1 ( Z &gt; a ) denote the probabilities of the upper tails of the corrupted and clean parts respectively, and let P ( a ) = P Z ∼D ( Z &gt; a ) = ηP 1 ( a ) + (1 -η ) P 0 ( a ) denote the probability of selection from the mixture distribution. Similarly for expectations, define E 0 ( a ) := E Z ∼D 0 [ Z | Z &gt; a ] , E 1 ( a ) := E Z ∼D 1 [ Z | Z &gt; a ] .

Concentration of S n ( θ ) to S ( θ ) . We claim that S n ( θ ) ≈ S n ( θ ) by using two approximations. First, the un-centered version in S n ( θ ) approximates the centered version in S n ( θ ) . Second, although n sel ( θ ) is a random quantity, it concentrates around nP ( θ ) . We formally bound the error due to both these approximations in the full proof. Since the filtering threshold θ is chosen independent of the samples being filtered, the selected samples satisfy the i.i.d property under the conditional law of the score being above θ . This allows us to show that the approximate version, S n ( θ ) , concentrates around its expectation, S ( θ ) , by bounding the spectral norm of the difference via a Matrix-Bernstein type inequality. Overall, we get

<!-- formula-not-decoded -->

Analysis of S ( θ ) and P ( θ ) . Simplifying S ( θ ) reveals that it is simply a scaled version of U ˜ U ⊤ , with the scaling coefficient depending on θ described by the conditional expectations E 0 ( θ ) and E 1 ( θ ) . Concretely, S ( θ ) = 1 / r ( η E 1 ( θ ) + (1 -η ) E 0 ( θ )) U ˜ U ⊤ . Owing to this, the application of a Davis-Kahan result on Eq. (11) will dictate the guarantee of recovering U , ˜ U for the filtering algorithm. The error behaves as:

<!-- formula-not-decoded -->

The behavior of the functions E 0 ( θ ) , E 1 ( θ ) and P 0 ( θ ) , P 1 ( θ ) precisely quantify this rate. As a sanity check, setting θ = -∞ recovers the 1 / η behavior of the unfiltered contrastive learning, as E 1 ( -∞ ) = r, E 0 ( -∞ ) = 0 and P 1 ( -∞ ) = 1 , P 0 ( -∞ ) = 1 . Since E 0 ( θ ) , E 1 ( θ ) are increasing functions in θ , whereas P 0 ( θ ) , P 1 ( θ ) are decreasing, we observe a tradeoff. A larger threshold θ results in larger conditional expectations E 0 ( θ ) , E 1 ( θ ) , but smaller probabilities of selection P 0 ( θ ) , P 1 ( θ ) . In the Appendix, we formally characterize this behavior, involving calculations on the conditional expectations and probabilities of the Gaussian distribution. Here, we discuss the two choices of θ that recover the two regimes of the filtering behavior. The threshold θ = 0 results in E 1 (0) ≥ r , E 0 (0) ≥ 2 / π and P 1 (0) ≥ 0 . 5 , P 0 (0) = 0 . 5 , recovering the independent of η regime. And the threshold θ = r / 2 results in E 1 ( r / 2 ) ≥ r , E 0 ( r / 2 ) ≥ r / 2 (using a trivial lower bound for the conditional expectation), and P 1 ( r / 2 ) ≥ 0 . 5 (but P 0 ( r / 2 ) is small), recovering the 1 / √ η regime. The optimal θ ∗ will achieve a rate better than the above two special points, hence the upper bound on the error is given by the min of these two regimes, recovering the upper bound in Theorem 1.

## 7 Experiments

<!-- image -->

(a) Synthetic experiment.

(b) Real experiment, Fang et al. [10, Figure 4].

Figure 3: (a). Observed dependence of ERR( G , ˜ G ) on η for a synthetic experiment. The error of the unfiltered contrastive learning follows a 1 / η dependence, but deviates for small η since the requirement of n ≳ 1 / η 2 in Corollary 1 gets violated. The error of the filtering algorithm follows a 1 / √ η dependence in the large η (or small 1 / η ) regime, and an independent of η dependence in the small η regime. Going beyond to even smaller η causes deviations since Theorem 1 also requires n ≳ 1 / η 2 . The teacher-based filtering is with the threshold θ = 0 . (b). A similar trend on real data observed by Fang et al. [10]. The y-axis shows 1 -Accuracy, which is different than the error metric in (a). However, we note that the qualitative trend of the orange line having a smaller slope than the blue line still holds. Numbers from Fang et al. [10] are reproduced with permission.

In this section, we validate our theoretical results with a synthetic setup. With parameters d = 10 , ˜ d = 8 , r = 4 , and SNR γ = ˜ γ = 10 4 , and with randomly generated U , ˜ U , we generate n = 10 M samples according to the model in Section 3.1, and vary the clean fraction η . We experiment over 10 values of η geometrically decreasing from 1 to 10 -3 . This experiment was run on a cluster of 50 CPUs with 500G memory, and required less than 10 minutes. Figure 3a shows the result and discusses the observations, which validate Corollary 1 and Theorem 1. To extend these observations to real settings, the main limitations are posed by the modeling assumptions in Section 3. Despite the limitations, Figure 3b shows evidence that the qualitative conclusions drawn from the theory hold with real image-text data too. Concretely, it shows that the downstream model performance on reducing the clean fraction η degrades more steeply without data filtering.

## 8 Conclusion and Broader Impacts

This paper presents a theoretical investigation into teacher-based data filtering for multimodal contrastive learning with stochastically corrupted data. We rigorously establish its benefit, demonstrating

that filtering improves the error dependence on the clean data fraction, η , from 1 / η (no filtering) to 1 / √ η in the large η regime, and perhaps surprisingly, to independent of η in the small η regime. The latter finding suggests that teacher-based filtering can be particularly beneficial when data quality is low, achieving performance independent of the initial clean fraction. Our results provide a formal basis for the empirical success of teacher-based data filtering. The main limitations are posed by the assumption of linearity in Section 3, and the model of stochastic corruptions in Assumption 1. Future work could explore the optimal selection of filtering thresholds and investigate whether similar gains can be achieved with one-step filtering algorithms.

Our contributions are largely on the theoretical understanding of data filtering, and its potential benefits. At a high-level, effective data filtering can reduce the compute cost needed to train models, which has positive potential impacts through more judicious use of energy resources. On the other hand, data filtering can exacerbate the biases present in a dataset by selecting certain subpopulations more than the others. If this goes unchecked, it has potential negative impacts to society.

## Acknowledgements

SSD acknowledges the support of NSF DMS 2134106, NSF IIS 2143493, the Sloan Fellowship, and the AI2050 program at Schmidt Sciences. SO acknowledges the support of NSF grants no. 2112471, 2229876, and 2505865.

## References

- [1] A. Albalak, Y. Elazar, S. M. Xie, S. Longpre, N. Lambert, X. Wang, N. Muennighoff, B. Hou, L. Pan, H. Jeong, et al. A survey on data selection for language models. arXiv preprint arXiv:2402.16827 , 2024.
- [2] M. Bagnoli and T. Bergstrom. Log-concave probability and its applications. Economic Theory , 26(2), 2005.
- [3] Z. Bai and J. Yao. On sample eigenvalues in a generalized spiked population model. Journal of Multivariate Analysis , 106, 2012. ISSN 0047-259X.
- [4] F. Bunea and L. Xiao. On the sample covariance matrix estimator of reduced effective rank population matrices, with applications to fpca. Bernoulli , 21(2), 2015.
- [5] T. Cai, Z. Ma, and Y. Wu. Optimal estimation and rank detection for sparse spiked covariance matrices, 2016. URL https://arxiv.org/abs/1305.3235 .
- [6] Z. Chen, Y. Deng, Y. Li, and Q. Gu. Understanding transferable representation learning and zero-shot transfer in CLIP, 2024.
- [7] I. Daunhawer, A. Bizeul, E. Palumbo, A. Marx, and J. E. Vogt. Identifiability results for multimodal contrastive learning, 2023.
- [8] L. Engstrom, A. Ilyas, B. Chen, A. Feldmann, W. Moses, and A. Madry. Optimizing ml training with metagradient descent. arXiv preprint arXiv:2503.13751 , 2025.
- [9] A. Fang, G. Ilharco, M. Wortsman, Y. Wan, V. Shankar, A. Dave, and L. Schmidt. Data determines distributional robustness in contrastive language image pre-training (clip). In International Conference on Machine Learning , pages 6216-6234. PMLR, 2022.
- [10] A. Fang, A. M. Jose, A. Jain, L. Schmidt, A. Toshev, and V. Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425 , 2023.
- [11] S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems , 36, 2023.
- [12] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 9729-9738, 2020.
- [13] W. Huang, A. Han, Y. Chen, Y. Cao, Z. Xu, and T. Suzuki. On the comparison between multi-modal and single-modal contrastive learning, 2024.
- [14] W. Ji, Z. Deng, R. Nakada, J. Zou, and L. Zhang. The power of contrast for feature learning: A theoretical analysis. Journal of Machine Learning Research , 24(330):1-78, 2023.

- [15] L. Jing, P. Vincent, Y . LeCun, and Y. Tian. Understanding dimensional collapse in contrastive self-supervised learning. arXiv preprint arXiv:2110.09348 , 2021.
- [16] I. M. Johnstone. On the distribution of the largest eigenvalue in principal components analysis. The Annals of Statistics , 2001.
- [17] S. Joshi, A. Jain, A. Payani, and B. Mirzasoleiman. Data-efficient contrastive language-image pretraining: Prioritizing data quality over quantity, 2024.
- [18] W. Kim, S. Chun, T. Kim, D. Han, and S. Yun. Hype: Hyperbolic entailment filtering for underspecified images and texts. In European Conference on Computer Vision . Springer, 2024.
- [19] G. Kolossov, A. Montanari, and P. Tandon. Towards a statistical theory of data selection under weak supervision, 2023.
- [20] Z. Lin, Z. Gou, Y. Gong, X. Liu, Y. Shen, R. Xu, C. Lin, Y. Yang, J. Jiao, N. Duan, and W. Chen. Rho-1: Not all tokens are what you need, 2024.
- [21] L. Lovász and S. Vempala. The geometry of logconcave functions and sampling algorithms. Random Structures &amp; Algorithms , 30(3), 2007.
- [22] P. Maini, S. Goyal, Z. C. Lipton, J. Z. Kolter, and A. Raghunathan. T-mars: Improving visual representations by circumventing text feature learning. arXiv preprint arXiv:2307.03132 , 2023.
- [23] A. M. Mathai and S. B. Provost. Quadratic forms in random variables: theory and applications.
- [24] R. Nakada, H. I. Gulluk, Z. Deng, W. Ji, J. Zou, and L. Zhang. Understanding multimodal contrastive learning and incorporating unpaired data, 2023.
- [25] T. Nguyen, G. Ilharco, M. Wortsman, S. Oh, and L. Schmidt. Quality not quantity: On the interaction between dataset design and robustness of clip. Advances in Neural Information Processing Systems , 35:21455-21469, 2022.
- [26] T. Nguyen, S. Y. Gadre, G. Ilharco, S. Oh, and L. Schmidt. Improving multimodal datasets with image captioning. Advances in Neural Information Processing Systems , 36, 2023.
- [27] K. Oko, L. Lin, Y. Cai, and S. Mei. A statistical theory of contrastive pre-training and multimodal generative ai. arXiv preprint arXiv:2501.04641 , 2025.
- [28] D. Pareek, S. S. Du, and S. Oh. Understanding the gains from repeated self-distillation, 2024.
- [29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning . PmLR, 2021.
- [30] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems , 35: 25278-25294, 2022.
- [31] V. Shah, X. Wu, and S. Sanghavi. Choosing the sample with lowest loss makes sgd robust. In International Conference on Artificial Intelligence and Statistics . PMLR, 2020.
- [32] M. Shechter and Y. Carmon. Filter like you test: Data-driven data filtering for clip pretraining. arXiv preprint arXiv:2503.08805 , 2025.
- [33] Y. Tian. Understanding deep contrastive learning via coordinate-wise optimization. Advances in Neural Information Processing Systems , 35:19511-19522, 2022.
- [34] J. T. Wang, T. Wu, D. Song, P. Mittal, and R. Jia. Greats: Online selection of high-quality data for llm training in every iteration. Advances in Neural Information Processing Systems , 37, 2024.
- [35] Y. Wang, Y. Chen, W. Yan, A. Fang, W. Zhou, K. G. Jamieson, and S. S. Du. Cliploss and norm-based data selection methods for multimodal contrastive learning. Advances in Neural Information Processing Systems , 37, 2024.
- [36] Z. Wen and Y. Li. Toward understanding the feature learning process of self-supervised contrastive learning. In International Conference on Machine Learning , pages 11112-11122. PMLR, 2021.
- [37] A. Wettig, A. Gupta, S. Malik, and D. Chen. Qurating: Selecting high-quality data for training language models. arXiv preprint arXiv:2402.09739 , 2024.

- [38] S. M. Xie, S. Santurkar, T. Ma, and P. S. Liang. Data selection for language models via importance resampling. Advances in Neural Information Processing Systems , 36:34201-34227, 2023.
- [39] H. Xu, S. Xie, X. E. Tan, P.-Y . Huang, R. Howes, V . Sharma, S.-W. Li, G. Ghosh, L. Zettlemoyer, and C. Feichtenhofer. Demystifying clip data. arXiv preprint arXiv:2309.16671 , 2023.
- [40] A. R. Zhang, T. T. Cai, and Y. Wu. Heteroskedastic pca: Algorithm, optimality, and applications, 2021.
- [41] B. Zhu, M. I. Jordan, and J. Jiao. Iterative data smoothing: Mitigating reward overfitting and overoptimization in RLHF, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction are written to summarize the sections that follow. The main contributions are theoretical, and their key takeaways are mentioned in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations of the work include assumptions and settings, and are discussed with the text (e.g. section 3.1 discusses the modeling assumptions and their limitations).

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

Justification: The theoretical results contributed are Proposition 1 and Theorem 1. The assumptions are discussed in section 3.1, and the proofs are provided in the Appendix (sections E and G).

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: The experiments (synthetic) are discussed in section 7 and necessary details are provided (e.g. parameter settings).

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

Justification: We do not publish code, primarily because the synthetic experiments serve for the verification of the theory and are relatively simple to implement.

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

Justification: There is just one hyperparameter, the filtering threshold θ . Section 7 (Figure 3a) includes the necessary details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Figure 3a includes error bars.

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

Justification: Section 7 includes these details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We adhere to the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Section 8 discusses the broader impacts of this work.

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

Justification: This is a theoretical study and no real-world data/models have been used.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No existing assets (codebases/datasets/etc) have been used.

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

Justification: No new assets have been introduced.

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

Justification: The paper does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Additional Illustrations

In this section, we provide some useful illustrations. Figure 4 illustrates the corruption model described in Assumption 1. Figure 5 illustrates the linear maps G , ˜ G used to generate the embeddings from observed data (according to the model in Fig 4). Figure 6 accompanies Remark A.1.

<!-- image -->

˜

˜

.

Figure 4: Model for stochastic corruptions. In this work, the forward maps f, ˜ f are linear (refer to Eq. (1)) and the latent distributions are Gaussians.

<!-- image -->

Figure 5: On seeing multimodal data ( x, ˜ x ) , linear maps G , ˜ G (learnable parameters) create the embeddings that lie in R r (the knowledge of r , the true latent dimension, is assumed). The similarity is measured with the inner product ⟨ G x, ˜ G ˜ x ⟩ .

)

Figure 6: Illustration of the joint distribution of ( x, ˜ x ) . The overall distribution is a mixture of two zero mean Gaussians: the independent case (w.p. 1 -η ) and the correlated case (w.p. η ).

<!-- image -->

Remark A.1. The distribution of ( x, ˜ x ) ∈ R d + ˜ d from Section 3.1 is a mixture of two zero-mean Gaussians. With weight η , the covariance matrix is Σ 1 (for c = 1 , i.e. the clean case). With weight 1 -η , the covariance is Σ 0 (for c = 0 ). Figure 6 provides an illustration.

<!-- formula-not-decoded -->

## B Background

This section covers some useful background concepts.

## B.1 Measuring the distance between subspaces

The concept of principal angles provides a geometrically intuitive way to measure the closeness between two subspaces. Let X and Y be two r -dimensional subspaces within a larger Euclidean space R d . There exist r principal angles 0 ≤ θ 1 ≤ θ 2 ≤ · · · ≤ θ r ≤ π/ 2 that describe the relative orientation of these subspaces.

- θ 1 represents the smallest possible angle between any two unit vectors x ∈ X and y ∈ Y .
- Subsequent angles θ k capture the minimum angles within directions orthogonal to those defining the previous angles θ 1 , . . . , θ k -1 .
- The cosines cos( θ i ) measure the alignment (1 means aligned, 0 means orthogonal within that principal direction), while the sines sin( θ i ) measure the separation or angle.

To aggregate this information into a single distance metric, we often use the frobenius norm of the sine of the principal angles, denoted ∥ sin Θ( X , Y ) ∥ F . It is defined as

<!-- formula-not-decoded -->

This metric provides an overall measure of the difference between the subspaces. It's zero if and only if X = Y (since all θ i = 0 ), and it increases as the subspaces diverge.

Computing this metric relies on matrix operations involving orthonormal bases for the subspaces. Let X ∈ R d × r be a matrix whose columns form an orthonormal basis for X (so X ⊤ X = I r ). Similarly, let Y ∈ R d × r be a matrix with orthonormal columns forming a basis for Y . The distance metric ∥ sin Θ( X , Y ) ∥ F can be computed using X and Y via the following formula

<!-- formula-not-decoded -->

Here, X ⊥ is any d × ( d -r ) matrix such that its columns form an orthonormal basis for the orthogonal complement of X , denoted X ⊥ . This means that the combined matrix [ X X ⊥ ] must be a d × d orthogonal matrix. Notationally, we often just write ∥ sin Θ( X , Y ) ∥ F instead of using X , Y .

## B.2 Optimal unimodal estimation rates in the spiked covariance model

Eq. (1) uses the well-known spiked covariance model for each of the two modalities, originally introduced by Johnstone [16] and well-studied in the literature [3, 40, 5]. Cai et al. [5] establish optimal (minimax) estimation rates for the covariance matrix (i.e. UU ⊤ + γ -1 I d ) and the principal subspace (i.e. U ) in a more general sparse spiked covariance model. In particular, [5, Eq. (7)] describes the minimax rate for covariance estimation, and [5, Eq. (9)] describes the minimax rate for subspace estimation. We use the latter result to get Eq. (2). Since the problem of subspace estimation is invariant to scaling, we instantiate [5, Eq. (9)] for the estimation of data with covariance γ UU ⊤ + I d (since σ = 1 is assumed in [5, Eq. (1)] to fix the problem scaling). With this, the paramaters map as λ = γ , p = d and k = d (since our model is not sparse). This establishes a rate (up to constants) of √ dγ -1 (1+ γ -1 ) n for the estimation in 2 -norm. An additional factor of √ r appears since we use the Frobenius-norm (i.e. the chordal distance in Definition 1), and Eq. (2) follows.

## C Lemmas

This section presents Lemmas used in the proofs. The first three Lemmas are standard results in the literature, and we include them without proof.

Lemma 1 (Weyl's Inequality) . For matrices A,B ∈ R m × n , let p = min( m,n ) and let σ 1 ( M ) ≥ σ 2 ( M ) ≥ · · · ≥ σ p ( M ) ≥ 0 denote the singular values for M ∈ { A,B } . Then, for all j = 1 , . . . , p , it holds that

<!-- formula-not-decoded -->

Lemma 2 (Wedin's Theorem) . Let A, ˆ A ∈ R m × n be matrices of the same size. Let r ≤ min( m,n ) be the rank of both A, ˆ A , and let the SVDs be A = U Σ V ⊤ and ˆ A = ˆ U ˆ Σ ˆ V ⊤ . Let σ r ( A ) &gt; 0 denote the r th singular value of A , and assume σ r ( A ) &gt; ∥ ˆ A -A ∥ 2 . Then it holds that:

<!-- formula-not-decoded -->

Lemma 3 (Whittle's Inequality) . Let X 1 , X 2 , . . . be a sequence of independent random variables such that: ( i ) E [ X k ] = 0 for all k ≥ 1 , and ( ii ) the distribution of each X k is symmetric about zero (i.e., X k and -X k have the same distribution). Let S n = ∑ n k =1 X k be the partial sum (with S 0 = 0 ). If ϕ : R → R is a convex function such that ϕ (0) = 0 , then the sequence E [ ϕ ( S n )] is non-decreasing in n . That is, for all n ≥ 1 :

<!-- formula-not-decoded -->

Lemma 4. Let A,B ∈ R m × n with rank( A ) = r ≥ 1 . If ∥ A -B ∥ 2 &lt; σ r ( A ) , then for every t ∈ [0 , 1] , it holds that rank ( (1 -t ) A + tB ) ≥ r .

Proof. Let X t = (1 -t ) A + tB . For any matrices M,N and any k ,

<!-- formula-not-decoded -->

which follows Lemma 1. Applying this with M = X t , N = A , and k = r ,

<!-- formula-not-decoded -->

Lemma 5. Let X be a random variable with a log-concave density, mean µ X , and variance σ 2 X . It holds that

<!-- formula-not-decoded -->

Proof. Let m ( x ) = E [ X -x | X &gt; x ] be the mean residual life function. We want to bound E [ X | X &gt; θ ] = θ + m ( θ ) for θ ≥ µ X . Due to log-concavity of X , m ( x ) is non-increasing (see, eg, Bagnoli and Bergstrom [2, Theorem 6]). Since m ( x ) is non-increasing, m ( θ ) ≤ m ( µ X ) = E [ X -µ X | X &gt; µ X ] . We will now bound the conditional expectation for this case of θ = µ X .

Let Y = X -µ X . Then E [ Y ] = 0 and V ( Y ) = σ 2 X . m ( µ X ) = E [ Y | Y &gt; 0] = E [ Y + ] P ( Y &gt; 0) , where Y + = max(0 , Y ) . We know E [ Y + ] ≤ √ E [( Y + ) 2 ] ≤ √ E [ Y 2 ] = σ X . As for the denominator, we know that for any random variable X with a log-concave density and mean µ X , P ( X ≥ µ X ) ≥ 1 / e (see, eg, Lovász and Vempala [21, Lemma 5.4]). Thus, m ( µ X ) ≤ σ X 1 /e = e σ X .

Lemma 6. Let x, y ∈ R d and ˜ x, ˜ y ∈ R ˜ d be random vectors. Assume that the pair ( x, ˜ x ) is independent of the pair ( y, ˜ y ) . Let A be a fixed d × ˜ d matrix and let θ ∈ R be a scalar threshold. Define the events C x = { x ⊤ A ˜ x &gt; θ } and C y = { y ⊤ A ˜ y &gt; θ } . Assume that these events have non-zero probability, i.e., P ( C x ) &gt; 0 and P ( C y ) &gt; 0 . Then the conditional expectation of the outer product x ˜ y ⊤ given both events C x and C y factorizes as follows:

<!-- formula-not-decoded -->

Proof. The definition of conditional expectation given multiple events is conditioning on their intersection. Here I denotes the indicator function.

<!-- formula-not-decoded -->

The event C x is determined solely by the random variables x and ˜ x . The event C y is determined solely by the random variables y and ˜ y . By the initial assumption, the pair ( x, ˜ x ) is independent of

the pair ( y, ˜ y ) . Therefore, the event C x is independent of the event C y . This implies P ( C x ∩ C y ) = P ( C x ) P ( C y ) . Hence the denominator factorizes (and is non-zero since P ( C x ) &gt; 0 and P ( C y ) &gt; 0 ).

Now consider the numerator. Since C x and C y are independent, I C x ∩ C y = I C x I C y , which implies

<!-- formula-not-decoded -->

again, due to independence of the pairs. Hence the numerator also factorizes.

Lemma 7. Let x ∈ R d and ˜ x ∈ R ˜ d be random vectors such that their joint distribution is a multivariate normal distribution with zero mean. Let A be a fixed d × ˜ d matrix, and consider the conditioning event R = { ( x, ˜ x ) | x ⊤ A ˜ x &gt; θ } for some threshold θ ∈ R . Assume that the probability of this event is non-zero, i.e., P ( R ) &gt; 0 . Then

<!-- formula-not-decoded -->

Proof. Let Z = ( x, ˜ x ) ∈ R d + ˜ d . The joint probability density function of Z , denoted by p ( Z ) , corresponds to the N (0 , Σ joint ) distribution for some covariance matrix Σ joint . The conditional expectation is defined as:

<!-- formula-not-decoded -->

We focus on the numerator integral and show that it is zero owing to symmetry. First note that p ( Z ) is symmetric around the origin. That is, p ( Z ) = p ( -Z ) for all Z ∈ R d + ˜ d . Second, observe that under the transformation Z ↦→-Z , the condition becomes ( -u ) ⊤ A ( -˜ u ) &gt; θ , which simplifies to u ⊤ A ˜ u &gt; θ . Thus, the region R is symmetric with respect to the origin: Z ∈ R ⇐⇒ Z ∈ R .

Lemma 8. Consider the random variable z := uv , where u, v are jointly Gaussian as

<!-- formula-not-decoded -->

Let { z k } r k =1 be r independent copies. The conditional expectation is upper and lower bounded as

<!-- formula-not-decoded -->

For the specific case of γ = 0 (i.e. u, v independent) and θ = 0 , a stronger lower bound is

<!-- formula-not-decoded -->

Proof. Simplify the expression. Observe that z k are i.i.d. random variables. The expectation is E [ z k ] = E [ uv ] = γ (since E [ u ] = 0 = E [ v ] ). Let S = ∑ r k =1 z k , and let p S ( . ) denote the PDF of S . The expectation is E [ S ] = rγ , and the variance is V [ S ] = r ( σ 2 u σ 2 v + γ 2 ) .

Due to the symmetry among the i.i.d. variables z k , the conditional expectation E [ z i | S &gt; θ ] is the same for all i ∈ { 1 , . . . , r } . Let Q ( θ ) = E [ z i | S &gt; θ ] . By linearity of expectation, we have

<!-- formula-not-decoded -->

Proof of lower bounds: general case lower bound θ r . Observe that

<!-- formula-not-decoded -->

Combining this with Eq. (12) shows the θ / r lower bound.

Proof of lower bounds: general case lower bound γ . For this, we show E [ S | S &gt; θ ] is nondecreasing in θ . Let h ( θ ) = E [ S | S &gt; θ ] . Using Eq. (13), its derivative is given by

<!-- formula-not-decoded -->

Thus E [ S | S &gt; θ ] is non-decreasing in θ . In particular, E [ S | S &gt; θ ] ≥ E [ S ] (i.e. the unconditional limit in the limit θ →-∞ ). Since E [ S ] = rγ , using this in Eq. (12) shows the lower bound of γ .

Proof of lower bounds: the specific case of γ = 0 and θ = 0 . Since the distribution of z k is symmetric around zero, the distribution of S = ∑ k z k is also symmetric around zero. Therefore, P ( S &gt; 0) = 1 / 2 . Using this, we get

<!-- formula-not-decoded -->

Also, the expectation of the absolute value is E [ | S | ] = ∫ ∞ -∞ | s | p S ( s ) ds . Due to symmetry (i.e. p S ( -s ) = p S ( s ) ), we get

<!-- formula-not-decoded -->

Using Eq. (15) and Eq. (16), we get

<!-- formula-not-decoded -->

Eq (†) holds intuitively. To formally show it, we invoke Lemma 3 (Whittle's inequality) on the convex function ϕ ( x ) = | x | . Using this with Eq. (12) gives the desired result.

Proof of the upper bound. The probability density function of z = uv is given by

<!-- formula-not-decoded -->

where ρ = γ/ ( σ u σ v ) denotes the correlation factor. Note that | ρ | &lt; 1 is ensured via γ &lt; σ u σ v in the lemma statement. The function K 0 ( a | x | ) is log-concave for a &gt; 0 . The term exp( bx ) is log-linear (hence log-concave). The product of log-concave functions is log-concave. Thus, f z ( x ) is log-concave. Since S is a sum of r i.i.d. random variables with log-concave densities, S also has a log-concave density. We use Lemma 5 to get that E [ S | S &gt; θ ] ≤ θ + e √ r ( σ 2 u σ 2 v + γ 2 ) for θ ≥ rγ . For θ ∈ [0 , rγ ] , we use the non-decreasing property of E [ S | S &gt; θ ] from Eq. (14). Plugging into Eq. (12) concludes the argument.

Lemma 9. Consider Gaussian random variables x, y ∈ R r , such that

<!-- formula-not-decoded -->

For θ ∈ R , define A ( θ ) := E [ xy ⊤ | x ⊤ y &gt; θ ] . It holds that A ( θ ) satisfies

<!-- formula-not-decoded -->

where f ( θ ) is a scalar function of θ ∈ R , such that

<!-- formula-not-decoded -->

In the special case of a xy = 0 , it further holds that f (0) ≥ 2 √ a x a y / πr .

Proof. We first build an intuition for the quantity A ( θ ) ∈ R r × r . For θ = -∞ , A ( θ ) becomes the unconditional expectation, which is a xy I r according to the given covariance structure. As θ increases in R , we expect A ( θ ) to increase.

̸

A ( θ ) is diagonal. We first show that A ( θ ) is a diagonal matrix. The ( i, j ) -th entry is A ( θ ) ij = E [ x i y j | Z &gt; θ ] , where Z = x ⊤ y = ∑ r l =1 x l y l . Consider the transformation T i : R 2 r → R 2 r that maps ( x, y ) to ( x ′ , y ′ ) where x ′ l = x l for l = i , x ′ i = -x i , and y ′ l = y l for l = i , y ′ i = -y i .

̸

̸

First, note that Z ′ = ∑ l = i x l y l + ( -x i )( -y i ) = Z . Hence the condition Z &gt; θ is invariant under the transformation T i . Second, due to independence and the block diagonal structure of the covariance, the overall joint density is a product of univariate Gaussians centered around zero. Due to the symmetry of a univariate Gaussian, the overall density is also invariant under T i . Third, the entry x i y j becomes -x i y j under the transformation T i . Due to this symmetry, we conclude that the off-diagonal entries are zero.

All the diagonal entries of A ( θ ) are equal by symmetry. The diagonal entries are A ( θ ) ii = E [ x i y i | Z &gt; θ ] . Let Z i = x i y i , meaning Z = ∑ r l =1 Z l . Due to the block diagonal structure on ( x, y ) , each Z i is independent and identically distributed. Hence, A ( θ ) ii = A ( θ ) jj for any i, j ∈ [ r ] .

Properties of f ( θ ) . From the above two steps, we conclude that A ( θ ) = f ( θ ) I r for some scalar function f : R → R . Using the trace trick, we see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the covariances of x, y are scaled identity, each x i y i , i ∈ [ r ] is identically distributed. This distribution is akin to uv for u ∼ N (0 , a x ) , v ∼ N (0 , a y ) with Cov ( u, v ) = a xy . Hence

<!-- formula-not-decoded -->

for u i , v i i.i.d. according to the described distribution. Lemma 8 shows the required properties on this conditional expectation, showing the desired inequalities in the statement of this lemma.

Lemma 10. Let x ∈ R d and ˜ x ∈ R ˜ d be jointly Gaussian vectors with mean zero and joint covariance matrix Σ full which is positive definite. Consider M O , M T ∈ R d × ˜ d satisfying rank( M O ) ≥ 2 and ∥ M T -M O ∥ &lt; σ rank( M O ) ( M O ) . For any A ∈ R d × ˜ d , let Y A := x ⊤ A ˜ x . For a real θ ≥ 0 , define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the randomness is over the Gaussian ( x, ˜ x ) . Then, there exist constants C P ( θ, Σ full , M O ) &gt; 0 and C E ( θ, Σ full , M O ) &gt; 0 that depend on θ , the covariance Σ full , and M O, such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We prove the two bounds using differentiability arguments. Define ∆ M := M T -M O, and define the scalar Y t := x ⊤ ( M O + t ∆ M ) ˜ x . Note that we have overloaded notation by reusing Y ; it shall be clear from the context that Y t for a scalar t and Y A for a matrix A mean different things.

Using Lemma 4 with the given condition on ∥ M T -M O ∥ , we conclude that rank( M O + t ∆ M ) ≥ rank( M O ) ≥ 2 for all t ∈ [0 , 1] . Since ( x, ˜ x ) is jointly Gaussian, rank ≥ 2 ensures that Y t for all t ∈ [0 , 1] have a smooth and bounded density everywhere. This is because the random variable Y A is equivalent to the quadratic form on a Gaussian, ( 1 / 2 ) z ⊤ H z with

<!-- formula-not-decoded -->

This quadratic form has a known characteristic function as below (Mathai and Provost [23, Sec 3.2])

<!-- formula-not-decoded -->

One can see that rank( H ) = 2 · rank( A ) and | ϕ ( t ) | decays as | t | -rank( H ) / 2 as | t |→∞ . This shows that rank( H ) ≥ 4 ensures at least a | t | -2 decay, which ensures boundedness everywhere.

(i) Probability Difference Bound (eq. (19) ). Define the path h ( t ) := P { Y t &gt; θ } for t ∈ [0 , 1] . Then by the Mean Value Theorem, it holds that

<!-- formula-not-decoded -->

Since Y t has a finite and bounded density everywhere, h ( t ) is differentiable and its derivative is

<!-- formula-not-decoded -->

where δ is the Dirac delta function. Using the Cauchy-Schwarz inequality, we can write

<!-- formula-not-decoded -->

where f Y t ( θ ) is the density of Y t at θ . Because Y t is non-degenerate for t ∈ [0 , 1] , both f Y t ( θ ) and the conditional expectation are finite and bounded over t . Thus the linear dependence on ∥ ∆ M ∥ in Eq. (19) follows, since any ξ ∈ (0 , 1) satisfies the above conditions.

(ii) Expectation Difference Bound (eq. (20) ). Define H ( t ) := E [ x ˜ x ⊤ · I { Y t &gt; θ } ] . Then by the Mean Value Theorem, we have

<!-- formula-not-decoded -->

Differentiating under the expectation gives

<!-- formula-not-decoded -->

For any matrix norm, we have

<!-- formula-not-decoded -->

Again, all terms other than ∥ ∆ M ∥ are bounded for t ∈ [0 , 1] , yielding the desired Eq. (20).

Lemma 11. Let x 1 , . . . , x n ∈ R d be n i.i.d. random vectors drawn from a Gaussian distribution N (0 , Σ) , where Σ is a d × d positive definite covariance matrix, d ≥ 1 , n ≥ 1 . Let S be a random subset of indices { 1 , . . . , n } generated by including each index j ∈ { 1 , . . . , n } independently with probability p ∈ (0 , 1] . Let n c = | S | denote the number of selected samples, and define the sample covariance matrix for n c &gt; 0 as ˆ Σ n c = (1 /n c ) ∑ i ∈ S x i x ⊤ i . For a failure probability δ ∈ (0 , 1) , assume that np &gt; 8 log(2 /δ ) holds. Then, with probability at least 1 -δ , both n c ≥ np/ 2 and the sample covariance matrix of the selected data satisfies:

<!-- formula-not-decoded -->

Proof. Define k min := ⌈ np -√ 2 np log(2 /δ ) ⌉ . Note that k min ≥ np/ 2 due to the assumption. Let

<!-- formula-not-decoded -->

denote the failure events. A union bound over the two failure probabilities will give the desired result. Below we bound the individual failure probabilities.

Bounding P ( F 1 ) : Define ∆ 0 := √ 2 log(2 /δ ) / ( np ) , so that k min = ⌈ (1 -∆ 0 ) np ⌉ . Since we assumed np &gt; 8 log(2 /δ ) , ∆ 0 &lt; 0 . 5 . By a standard Chernoff bound for binomial distributions, P ( n c &lt; (1 -∆ 0 ) np ) ≤ exp( -np ∆ 2 0 / 2) = exp( -log(2 /δ )) = δ/ 2 . Since k min ≥ (1 -∆ 0 ) np (due to the ceil operation), it follows that P ( F 1 ) = P ( n c &lt; k min ) ≤ P ( n c ≤ (1 -∆ 0 ) np ) ≤ δ/ 2 .

Bounding P ( F 2 ) : Using the law of total probability, we write

<!-- formula-not-decoded -->

For any k ≥ k min , we have 1 / √ k ≤ 1 / √ k min . Thus, for k ≥ k min :

<!-- formula-not-decoded -->

And the right hand side is bounded by δ/ 2 owing to standard matrix concentration results. So, P ( F 2 ) ≤ ∑ n k = k min ( δ/ 2) P ( n c = k ) ≤ δ/ 2 .

## D A proof of Corollary 1

We present a proof of Corollary 1, which follows the proof presented in Nakada et al. [24] while fixing some typos. Before diving into the proof, we make some remarks.

First , the result stated in Corollary 1 is tighter than its counterpart Nakada et al. [24, Theorem 3.1] by a dimension factor. This is because we use tighter concentration, as detailed in the explanation between Eqs (28) and (29). Second , as remarked in Remark 4.1, Corollary 1 is not tight in the SNR parameters γ, ˜ γ . Third , the result in Nakada et al. [24] is for a general covariance on the signal, Σ z , and the noise, Σ ξ , whereas our setting is more restricted from Assumptions 1 and 2. This restriction is required for the analysis of filtering in Theorem 1.

Fourth , the result in [24] is stated with probability 1 -O ( 1 / n ) , whereas we state it with probability 1 -exp( -d ) . Due to this, Corollary 1 as stated does not have a log n factor inside the square root, unlike Nakada et al. [24, Theorem 3.1]. Fifth , there is a small subtle difference in the setting of [24] and ours. We use η to denote the fixed probability of clean samples in Assumption 1, whereas Nakada et al. [24] use η to denote the fraction of clean samples in the sampled dataset, which is a random quantity. Using n c to denote the number of clean samples, we go through the additional step of controlling the error in | n c / n -η | , which scales as 1 / √ n , since this source of error is 1-dimensional. Sixth , the result in Nakada et al. [24, Theorem 3.1] is stated as min { √ r, . } . While it is true that the sin Θ metric can be at most √ r , the final step in the proof is the application Lemma 2, which requires a condition that translates to n ≳ ( 1 / η 2 ) max { d, ˜ d } (1 + γ -1 )(1 + ˜ γ -1 ) . And so this is how we state the result in Corollary 1, which makes the stated upper bound always smaller than √ r .

For clarity, we write the algorithm:

<!-- formula-not-decoded -->

Output. G ⊤ ˜ G ∈ R d × ˜ d (with rank = r , since G ∈ R r × d , ˜ G ∈ R r × ˜ d ) by minimizing Eq. (3).

Step 1: Reduction of loss. We show that

<!-- formula-not-decoded -->

where S n denotes the cross covariance matrix of the data, given by (Eq. (5) rewritten)

<!-- formula-not-decoded -->

Proof. Expand the LHS as

<!-- formula-not-decoded -->

̸

where eq ( a ) holds because the overall sum over the n × n similarity matrix is the same whether done over rows or columns.

For the RHS, we first rewrite S n as

<!-- formula-not-decoded -->

Using the above, we rewrite the RHS as

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(Cyclic nature of Trace)

<!-- formula-not-decoded -->

̸

Comparing the above to eq (X) concludes the proof.

Step 2: Closed-form solution. We show that (Eq. (6) rewritten)

<!-- formula-not-decoded -->

Hence, even though the optimization problem is non-convex, there is a closed-form solution, and no optimization analysis is needed. In particular, the right singular vectors of G , ˜ G are determined independent of the choice of ρ . This result is from Nakada et al. [24, Lemma 2.1].

Proof. Using Step 1's result, we can write

<!-- formula-not-decoded -->

The objective can be rewritten as

<!-- formula-not-decoded -->

The optimization variables appear only in the second term. Since rank ( G ⊤ ˜ G ) = r , by the Eckart-Young-Minsky Theorem, the solution is given by the best rank r approximation of S n / ρ .

Step 3: Relating error to op-norm concentration of S n . Weshow the below, where S n concentrates to S = η U ˜ U ⊤ .

<!-- formula-not-decoded -->

Proof. By triangle inequality, we have

<!-- formula-not-decoded -->

And for the first term on the right hand side, we use

<!-- formula-not-decoded -->

In Eq. (†), we used Lemma 1, and Eq. (††) holds because σ r +1 ( S ) = 0 , since S is rank r .

Step 4: Concentration of S n . We show that with probability 1 -exp ( -Ω(max { d, ˜ d } ) ) ,

<!-- formula-not-decoded -->

Before we prove this, we remark that the condition of n ≳ n 0 (for the appropriate n 0 stated in the statement of Corollary 1) ensures that the ˜ O (1 /n ) term is at ˜ O ( η 2 ) whereas the first term is ˜ O ( η ) . This ensures that we are in the regime where the 1 / √ n term dominates.

Proof. We start with the expansion of S n ,

<!-- formula-not-decoded -->

̸

̸

The main term that dictates the convergence is S (1) n . The term S (2) n concentrates around zero (since samples i = j, i, j ∈ [ n ] are independent), and the rate of convergence is ˜ O ( 1 / n ) due to averaging over n 2 terms, which is a higher order term. Let n c be a random variable that denotes the number of clean data points. We expand the sum in S (1) n below.

<!-- formula-not-decoded -->

We control the error in each term separately. For terms J 2 , K 1:3 , we need a result like Nakada et al. [24, Proposition C.1] in the simple case of X ⊥ ˜ X . For term J 1 , we need it for X = ˜ X .

The following two facts are going to be used multiple times. Here X,Y denote random quantities, and all others are fixed quantities (matrices/vectors).

<!-- formula-not-decoded -->

For the independent terms ( J 2 , K 1:3 ), we will use the below generic result. For R d x ∋ x ∼ N (0 , Σ x ) and R d y ∋ y ∼ N (0 , Σ y ) and N i.i.d. draws from both, we have the below result from the application of a Matrix-Bernstein result.

<!-- formula-not-decoded -->

For the dependent term ( J 1 ), we will use the below. Let R d x ∋ x ∼ N (0 , Σ x ) and N i.i.d. draws from this. This is also known in the literature, for e.g., Bunea and Xiao [4, Theorem 2.2].

<!-- formula-not-decoded -->

Note that the above two concentration results are tighter than Nakada et al. [24, Proposition C.1] by a factor of dimension, since the proposition has trace terms too, whereas only operator norms appear in the above two equations. This manifests in Corollary 1 as stated being tighter than Nakada et al. [24, Theorem 3.1] by a dimension factor inside the square root (since we avoided log n but did not incur an additional dimension due to the failure probability of exp( -d ) ). Finally, since n c = Bin ( n, η ) , the ratio n c / n concentrates to η , with the error described by Hoeffding's inequality as

<!-- formula-not-decoded -->

Using these results, we bound the individual terms of deviation. We first bound the independent terms using Eq. (27) with t := max { d, ˜ d } . The choice of N is given with each setting. With probability 1 -exp( -Ω(max { d, ˜ d } )) , the following hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now bound the dependent term using Eq. (28). We need some additional machinery to deal with the random denominator, which we capture in Lemma 11. The requirement of np ≳ log(1 /δ ) in the lemma translates to n ≳ max { d, ˜ d } / η , since we have p := η and δ := exp( -max { d, ˜ d } ) . As we will see later, step 5 of the proof requires n ≳ max { d, ˜ d } / η 2 , hence this requirement is already satisfied. With probability 1 -exp( -Ω(max { d, ˜ d } )) , it holds:

<!-- formula-not-decoded -->

For the concentration of n c / n , we use Eq. (29) to get that with probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

We now add all the error bounds. For the combined error from terms J 1 and J 2 , we note that √ 1 -n c / n ≤ 1 , and ( n c / n √ η ) ≤ 2 with high probability (since n c / n concentrates around η ). The failure probability of this can be absorbed into the overall failure probability. Eq. (24) follows.

Step 5: Relating singular vector recovery error to operator norm concentration. We will apply Lemma 2 (a Davis-Kahan type result) to relate the sin Θ metric to the operator norm. Combining Eqs. (24), (23) and (6), we get that with probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

The instantiation for Lemma 2 is as follows: A = η ρ U ˜ U ⊤ , ˆ A = G ⊤ ˜ G . Note that both A, ˆ A are rankr , and σ r ( A ) = η / ρ . We get

<!-- formula-not-decoded -->

Now we will use three things. First, for the numerator, we use ∥ M ∥ F ≤ √ rank( M ) · ∥ M ∥ 2 for any matrix M . Second, for the denominator, we will need the additional condition of n ≳ ( 1 / η 2 ) max { d, ˜ d } (1 + γ -1 )(1 + ˜ γ -1 ) to ensure the second term is at most half of the first term. This also ensures that the ˜ O ( 1 / n ) does not dominate the 1 / √ n term. Third, triangle inequality with the fact that ∥ ∥ ∥ sin Θ ( lsv ( G ⊤ ˜ G ) , rsv ( G ) )∥ ∥ ∥ F = 0 gives the final result. To see this fact, write

<!-- formula-not-decoded -->

(Using SVD of the middle component)

Using the uniqueness of SVD, we get that lsv ( G ⊤ ˜ G ) = V G P and rsv ( G ⊤ ˜ G ) = V ˜ G Q . Since P, Q are just orthogonal transforms, the subspace spanned by V G and V G P are the same, implying ∥ sin Θ( V G , V G P ) ∥ F = 0 (and analogously for V ˜ G and V ˜ G Q ).

Combining Eqs. (32) and (33) gives the desired result. Since the upper bound is valid for recovery of both U and ˜ U , Corollary 1 as stated follows.

## E A proof of Proposition 1

Consider the following construction for the hard problem instance (lower bound): ( i ) the latent dimension r = 1 , and ( ii ) the noise ˜ ξ = 0 (i.e. ˜ γ = ∞ ), but ξ = 0 (i.e. γ is finite). This means the following proof recovers the dγ -1 part from the max { dγ -1 , ˜ d ˜ γ -1 } term in Proposition 1. A similar argument can be made for the case when ξ = 0 , ˜ ξ = 0 , leading to the max over both errors.

̸

Owing to r = 1 , this becomes a 1-dimensional vector recovery problem. Let u , ˜ u ∈ R d denote the vectors to recover. Upon seeing S n , there is no error in estimating ˜ u since ˜ ξ = 0 , but there is error in estimating u . To calculate this error, define u n to be the top-left singular vector of S n . Note that S n has only one non-zero singular value, since it fully lies on ˜ u in the right singular vector space (i.e. S n v = 0 for any v ⊥ ˜ u ). Hence

Step 0. Writing down S n .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

We expand S (1) n below, using n c to denote the random variable denoting the clean samples. Note that E n c = ηn . Similarly one can expand S (2) n , however, the error of S (2) n will behave as O ( 1 / n ) due to averaging over n 2 samples, which is a higher order term in the overall rate. That is, the behavior (in the large n regime) will be largely dictated by S (1) n .

<!-- formula-not-decoded -->

As for the expectations, they are given by:

<!-- formula-not-decoded -->

E [ S (2) n ] = 0 (since all random quantities are zero-mean and independent) .

Step 1. Decompose sin θ metric. Our goal is a high probability lower bound on | sin θ ( u n , u ) | , where u n is the random quantity. Note that

<!-- formula-not-decoded -->

To see this, note that LHS = √ 1 -( u ⊤ u n ) 2 . Squaring both sides and expanding suffices.

Step 2. Compute the metric for this case. Using Eq. (34) in Eq. (35), we can write

<!-- formula-not-decoded -->

̸

Step 3. Computing the high probability bound. We will give high probability lower bound on the numerator and denominator of Eq. (36) separately.

Step 3.1. For the numerator: We first expand S (1) n as

<!-- formula-not-decoded -->

Similarly, for S (2) n we have

<!-- formula-not-decoded -->

Combining the two, we get

<!-- formula-not-decoded -->

Now we want to compute a high confidence lower bound on the norm of the above. We first relate ∥ ∥ ( I d -uu ⊤ ) w n ∥ ∥ to ∥ w n ∥ . This is because w n is spherically symmetric, and ( I d -uu ⊤ ) is a rank-( d -1) matrix with all non-zero eigenvalues equal to one. We get

<!-- formula-not-decoded -->

Now due to w n being spherically symmetric, ∥ w n ∥ (the magnitude) and ˆ w n (the direction) are independent random quantities. Further, ˆ w n is uniformly distributed on S d -1 .

For ∥ w n ∥ , we will use sharp Gaussian concentration. The intuition is that ∥ w n ∥ cannot be too smaller than √ dγ -1 / n , for large d . Concretely, it holds that

<!-- formula-not-decoded -->

An appropriate choice of δ = exp( -d/ 4) , which results in

<!-- formula-not-decoded -->

For the second term (with the direction ˆ w n ), this will be at least Ω(1) with high probability, since u ⊤ ˆ w n will be large only with very small probability when then dimension d is big enough. Concretely, it holds that

<!-- formula-not-decoded -->

Overall, for the numerator, we conclude that

<!-- formula-not-decoded -->

̸

̸

Step 3.2. For the denominator: We need a high confidence upper bound on ∥ S n ∥ . We can use Matrix-Bernstein type analysis. Note that E [ S n ] = η u ˜ u ⊤ . And the deviation is dominated by

<!-- formula-not-decoded -->

Again, the dominating term is the first one. This means that we only have to show high confidence upper bound on ∥ (1 /n ) ∑ i z i ξ i ∥ , and hence the problem has reduced to vector concentration instead of matrix concentration. Analogous to Eq. (37), one can show

<!-- formula-not-decoded -->

Overall, using the triangle inequality, we have

<!-- formula-not-decoded -->

Step 4. Combined result: From 3.1 and 3.2, for n ≥ 4 dγ -1 / η 2 (so the high-conf UB for ∥ S n ∥ is 2 η ),

<!-- formula-not-decoded -->

## F Characterizing the score distribution of the oracle

The Bernoulli variable c ∈ { 0 , 1 } captures the status of clean/corrupted nature of a sample. We first characterize the score distribution in both cases separately, and then create the relevant mixture distribution using the proportions η, 1 -η for clean, corrupted samples respectively.

Before the calculations, we state some Lemmas that will be used.

Lemma 12. Let X be distributed as N (0 , Ω) . For a fixed matrix A , it holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 13. Let X be distributed as N (0 , Ω) , and ˜ X be distributed as N (0 , ˜ Ω) . Let X, ˜ X be independent of each other. For a fixed matrix A , it holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consider a block matrix X given as below

<!-- formula-not-decoded -->

Lemma 14. For a block matrix X given as above, it holds that

<!-- formula-not-decoded -->

Lemma 15. For a block matrix X given as above, with A , D are square matrices, it holds that

<!-- formula-not-decoded -->

Case 0: Corrupted samples ( c = 0 case). Let Z 0 d = { S ( x, ˜ x ; U ˜ U ⊤ ) | c = 0 } , with distribution D 0 . This (scalar) random variable is equivalent to X ⊤ U ˜ U ⊤ ˜ X , where X, ˜ X are independent and follow X ∼ N ( 0 , UU ⊤ + γ -1 I d ) , ˜ X ∼ N ( 0 , ˜ U ˜ U ⊤ + ˜ γ -1 I ˜ d ) . This is in-line with Remark A.1. We invoke Lemma 13 to get the first two moments.

1. Mean: 0 .
2. Variance: r (1 + γ -1 )(1 + γ -1 ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

3. Tails: Since X, ˜ X are independent, the tails are described by the quadratic form on two independent Gaussians. This random variable is ( i ) symmetric, and ( ii ) uni-modal, and the tails decay exponentially.
2. Case 1: Clean samples ( c = 1 case). Let Z 1 d = { S ( x, ˜ x ; U ˜ U ⊤ ) | c = 1 } , with distribution D 1 . This random variable is equivalent to X ⊤ B X , where X = [ x, ˜ x ] ⊤ follows X ∼ N (0 , Σ 1 ) (refer to Remark A.1); and B is a block matrix given as below. We invoke Lemma 12 to get the first two moments.

<!-- formula-not-decoded -->

1. Mean: r .

<!-- formula-not-decoded -->

2. Variance: r + r (1 + γ -1 )(1 + ˜ γ -1 ) .

<!-- formula-not-decoded -->

3. Tails: Since X, ˜ X are dependent, the tails are described by the quadratic form on two dependent Gaussians. The tails decay exponentially, and are described by the HansonWright inequality. A similar calculation as the variance provides the exact parameters, and the inequality becomes:

<!-- formula-not-decoded -->

## G A proof of Theorem 1

In this section, we present a proof of Theorem 1. We first define one additional piece of notation. For U , let U ⊥ ∈ R d × ( d -r ) denote the completion of the orthonormal basis. That is, the matrix U full = [ U U ⊥ ] ∈ R d × d is such that U ⊤ full U full = I d = U full U ⊤ full . Similarly define ˜ U ⊥ ∈ R ˜ d × ( ˜ d -r ) .

Recall that we have n samples of the form { ( x i , ˜ x i ) } n i =1 , i.i.d from the mixture distribution (with η, 1 -η ratios for clean, corrupted respectively). Let n T samples be used to train the teacher, and let N = n T -n samples be used to train the student. Let ρ T , ρ be the respective regularization parameters, and let ( G T , ˜ G T ) , ( G , ˜ G ) denote the respective embedding matrices at the solution of Eq. (3). Consider a general threshold θ ∈ R that is used to filter the dataset based on the teacher scores. Note that we have ensured that θ is independent of the N samples to be filtered, since it depends only on the n T samples used for teacher training. For the teacher, from Corollary 1, we know that with probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

Here ( G T , ˜ G T ) are random quantities that depend on the n T samples used. For the rest of the analysis, we will assume them to be fixed (since they don't depend on the randomness of the remaining N samples). Finally, we will give a high probability guarantee that will use the confidence bound in Eq. (45) as one of the terms in the combined error bound, with an appropriate choice of n T and ρ T. We now study the student with data filtering. It is useful to define

<!-- formula-not-decoded -->

These are the matrices used for scoring the samples by the teacher and its oracle version, respectively. Note that rank( M O ) = r since both U , ˜ U are rankr matrices. From the teacher guarantee in Eq. (45), it holds that M T → M O as n T →∞ . Recall that the scoring function is S ( x, ˜ x ; M ) = x ⊤ M ˜ x , and a sample ( x, ˜ x ) is selected/retained iff S ( x, ˜ x ; M T ) &gt; θ .

We define certain quantities that will be central to the analysis. Akin to Eq. (5), we define the empirical cross-covariance matrix of the data after selection in Eq. (47). Let n sel , T ( θ ) be the number of samples selected, which is a random variable with E [ n sel , T ( θ )] = N P T ( θ ) . Let I sel , T ( θ ) ⊆ [ N ] denote the indices of the points selected. That is, i ∈ I sel , T ( θ ) ⇐⇒ S ( x i , ˜ x i ; M T ) &gt; θ . Similarly, define n sel , O ( θ ) and I sel , O ( θ ) . Construct the empirical cross-covariance matrix for the filtered dataset:

<!-- formula-not-decoded -->

To analyze its asymptotic limit, we define S ( θ ) as the limit of the cross-covariance, for both the teacher and the oracle. Similarly, let P ( θ ) denote the probability mass of data that is retained (also in the limit of n →∞ ), for both the teacher and the oracle. These are described in Eqs (48), (49).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that S T ( θ ) , S O ( θ ) are the limits of S N, T ( θ ) , S N, O ( θ ) as N → ∞ . The threshold θ → -∞ recovers the no filtering case, i.e. both S N, T ( θ ) , S N, O ( θ ) approach S N . We will now follow proof steps similar to Section D. Steps 1 and 2 hold for a general cross covariance matrix, and can be used directly. Steps 3 and 4 are concerned with the limit of S n ( θ ) as n →∞ , and how it concentrates around the limit. These steps will change significantly. Finally, we will be able to reuse Lemma 2 for step 5. We detail each of these proof steps below.

Step 1. Following the exact same proof steps as in Section D, the unregularized contrastive loss objective on the n sel , T ( θ ) samples is equivalent to

<!-- formula-not-decoded -->

Step 2. Again, following the exact same proof steps as in Section D, the solution to the ρ -regularized minimization problem is given by

<!-- formula-not-decoded -->

Step 3. This step changes from Section D. We use the following:

<!-- formula-not-decoded -->

By triangle inequality, we have

<!-- formula-not-decoded -->

And for the first term on the right hand side, we use

<!-- formula-not-decoded -->

where we used Lemma 1 in Eq (†).

Step 3'. Analysis of S O ( θ ) : The main difference in Eq. (23) and Eq. (52) is the term σ r +1 ( S O ( θ )) . This additional step of the proof analyzes the properties of S O ( θ ) . In particular, we will show that S O ( θ ) is rankr , and hence σ r +1 ( S O ( θ )) = 0 . Additionally, we establish upper and lower bounds on the singular values of S O ( θ ) that will be used later in the proof. From Eq. (49), we simplify to write

<!-- formula-not-decoded -->

where ( x, ˜ x ) is drawn from the mixture model: η · N (0 , Σ 1 ) + (1 -η ) · N (0 , Σ 0 ) . To simplify notation, define ¨ θ := ( θρ T ) / η . From the conditioning event, it seems that U ⊤ x and ˜ U ⊤ ˜ x is a good 'basis' for a decomposition. Pre-multiply and post-multiply to recover this basis for the x ˜ x ⊤ term inside the expectation as

<!-- formula-not-decoded -->

Call the top left entry in this decomposition to be the 'dominant', and the other three as 'nondominant'. We will show the non-dominant entries will be zero. The following reparametrization makes things cleaner.

<!-- formula-not-decoded -->

Let's further simplify the expressions with another transformation. The subscripts S, N denote the signal (containing some noise) and noise part.

<!-- formula-not-decoded -->

Due to the diagonal structure of Σ ξ , Σ ˜ ξ , we infer the distributions as

<!-- formula-not-decoded -->

And crucially, due to the diagonal structure of Σ ξ , Σ ˜ ξ , we infer that { ε, ε ⊥ , ˜ ε, ˜ ε ⊥ } are all mutually independent , and independent of z, ˜ z . This entails that the transformed vector is Gaussian with mean zero and covariance given as below.

<!-- formula-not-decoded -->

The above is for the corrupted case (w.p. 1 -η ). In the clean case (w.p. η ), the blue entries change to I r due to the relation of z = ˜ z . Our E [ . ] notation includes the expectation over this randomness along with the randomness of x, ˜ x . Denote by Ω 0 and Ω 1 the covariances of the signal part, i.e. ( x S , ˜ x S ) in these two cases:

<!-- formula-not-decoded -->

Overall, under the transformation, the expectation simplifies to

<!-- formula-not-decoded -->

Due to x N , ˜ x N being independent of all other entries via Eq. (53), and since the conditioning event in Eq. (55) only involves x S , ˜ x S , we conclude that the non-dominant entries in the expectation will be zero . Hence we are left with the simplified rankr form for the d × ˜ d matrix:

<!-- formula-not-decoded -->

We will now use Lemma 9 to simplify both the terms above. Note that Ω 1 , Ω 0 satisfy the lemma's requirement of the block diagonal covariance.

<!-- formula-not-decoded -->

where the following conditions hold on f 1 , f 0 (converting back from ¨ θ to θ ):

<!-- formula-not-decoded -->

Using the above equations, and the special case of θ = 0 in Lemma 9, we conclude:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will use these inequalities in step 5. In particular, since ∥ S O ( θ ) ∥ = η f 1 ( θ ) + (1 -η ) f 0 ( θ ) ,

<!-- formula-not-decoded -->

Step 4. Concentration of S N, T ( θ ) to S O ( θ ) : We break this into subparts as below.

Step 4.1. Concentration of S N, T ( θ ) to S T ( θ ) : Using the below substeps, we show that with probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

Step 4.1.1. Replacing the random denominator: Recall n sel , T ( θ ) = ∑ N i =1 I { S ( x i , ˜ x i ; M T ) &gt; θ } is the (random) number of selected samples. Since the teacher's score matrix M T and threshold θ are fixed independently of these N samples, the indicators are i.i.d. Bernoulli random variables with mean P T ( θ ) . By a standard Chernoff bound for sums of independent Bernoulli variables, n sel , T ( θ ) concentrates sharply around its expectation: for any 0 &lt; δ &lt; 1 ,

<!-- formula-not-decoded -->

In particular, choosing δ = (√ max { d, ˜ d } / N P T ( θ ) ) , we conclude that

<!-- formula-not-decoded -->

On this high-probability event, the following holds (recall the definition of Q N, T ( θ ) from Eq. (47)).

<!-- formula-not-decoded -->

In (†), we used Eq. (61), which implies that 0 . 5 N P T ( θ ) ≤ n sel , T ( θ ) ≤ 1 . 5 N P T ( θ ) when NP T ( θ ) ≳ max { d, ˜ d } (which is indeed true, since in Step 5 we set N = n/ 2 &amp; n ≳ max { d, ˜ d } is assumed in Theorem 1, and in Step 4.3 we ensure that P T ( θ ) ≳ 1 ). In (††), we again used Eq. (61) directly. In (†††), we used that ∥ Q N, T ( θ ) ∥ grows on the order of NP T ( θ ) (since it is the sum of n sel , T ( θ ) i.i.d. outer products each with bounded expectation). Thus, overall, replacing the random n sel , T ( θ ) by

NP T ( θ ) in the normalization incurs an error of order √ max { d, ˜ d } / NP T ( θ ) with high probability. In the subsequent analysis, we may therefore work with the fixed denominator NP T ( θ ) for convenience.

Step 4.1.2. The centered vs un-centered version: We have that

<!-- formula-not-decoded -->

̸

The second term on the right hand side concentrates to E [ x ˜ y ⊤ | x ⊤ M T ˜ x &gt; θ, y ⊤ M T ˜ y &gt; θ ] , where ( x, ˜ x ) and ( y, ˜ y ) are i.i.d. from the joint mixture distribution. This expectation is zero , which we formally characterize in Lemmas 6 and 7. The rate of concentration is ˜ O ( 1 N P T ( θ ) ) , due to averaging over ( N P T ( θ )) 2 terms, and is hence a higher order term.

Step 4.1.3. Analysis of the fixed-denominator un-centered version: The selected samples satisfy the property of being i.i.d from the conditional law of the selection rule. In particular, for each i ∈ I sel , T ( θ ) the matrix X i := x i ˜ x ⊤ i has expectation E [ X i ] = S T ( θ ) and these matrices { X i : i ∈ I sel , T ( θ ) } are independent. Using a Matrix-Bernstein concentration result (Eqs. (27) and (28)), it follows that with

probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

Step 4.2. Error between teacher and oracle: We show that ∥ S T ( θ ) -S O ( θ ) ∥ scales proportionally to ∥ M T -M O ∥ , and the latter is precisely bounded by Eq. (45). To show this, we first simplify the conditional expectation in S O ( θ ) , S T ( θ ) , define E O ( θ ) , E T ( θ ) as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I ( . ) denotes the indicator. Let ∆ E ( θ ) := E T ( θ ) -E O ( θ ) and ∆ P ( θ ) := P T ( θ ) -P O ( θ ) . Also define ∆ I ( θ ; x, ˜ x ) := I ( x ⊤ M T ˜ x &gt; θ ) -I ( x ⊤ M O ˜ x &gt; θ ) . Then, we write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will now bound ∥ ∆ E ( θ ) ∥ 2 and | ∆ P ( θ ) | in terms of ∥ M T -M O ∥ 2 . Recall that ( x, ˜ x ) follow the mixture distribution (Remark A.1). Decomposing the expectations and probabilities into respective mixtures, we get

<!-- formula-not-decoded -->

From the above, since both η, 1 -η are smaller than 1 , we get that

<!-- formula-not-decoded -->

where the subscripts 1 , 0 denote the fully clean, corrupted cases respectively (i.e. η = 1 , η = 0 respectively). Lemma 10 captures the general form of this, and we invoke this lemma on both the clean data (with covariance Σ 1 ) and the noisy data (with covariance Σ 0 ). Note that rank( M O ) ≥ 2 is satisfied since rank( M O ) = r and we assumed r ≥ 2 in the statement of Theorem 1. Further, the condition of ∥ M T -M O ∥ &lt; σ r ( M O ) is satisfied due to n ≳ (1 /η 2 ) max { d, ˜ d } ( 1 + γ -1 ) ( 1 + ˜ γ -1 ) , since M O has r non-zero singular values all equal to η / ρ T and Eq. (45) with the condition on n implies that ∥ M T -M O ∥ ≲ η / ρ T (note that implicitly the condition also ensures that the contribution of the ˜ O (1 /n ) term is bounded). The appropriate constants inside the ≳ notation will ensure the required condition. Overall, we get

<!-- formula-not-decoded -->

Step 4.3. Analysis of P T ( θ ) and P O ( θ ) : In this part, we show that both P T ( θ ) and P O ( θ ) can be lower bounded by an absolute constant (say, 1 / 10 ) for the relevant regime of filtering threshold θ .

Argument for P T ( θ ) : Using Step 4.2, we have P T ( θ ) ≥ P O ( θ ) -| ∆ P ( θ ) | , and the deviation is small since | ∆ P ( θ ) | ≲ ∥ M T -M O ∥ . Using Eq. (45), we note that a large ρ T can make ∥ M T -M O ∥ arbitrarily small. Indeed in Step 5, we will set ρ T to a large value. Since the deviation is small, we can use, for instance, P T ( θ ) ≥ ( 1 / 2 ) P O ( θ ) . Hence, arguing P O ( θ ) is large suffices, which we do below.

Argument for P O ( θ ) : Next, we show that P O ( θ ) is 'large enough' for the choices of θ ∈ { 0 , rη / 2 ρ T } , and we will use these fixed points in Step 5. Recall from Section 6.2, due to the mixture distribution, the below holds. Here we have accounted for the scaling factor in the definition of M O.

<!-- formula-not-decoded -->

In Step 5, we will consider the fixed points θ ∈ { 0 , rη / 2 ρ T } , and so we need lower bounds on P 0 (0) , P 0 ( r / 2 ) and P 1 (0) , P 1 ( r / 2 ) . We state them below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c &gt; 0 is an absolute constant. For P 0 ( . ) , we have lower bounds 0 . 5 (due to symmetry) and 0 (trivially). For P 1 ( . ) , we simply invoke the observation that both { 0 , r / 2 } are below the mean of the distribution (refer to Figure 2a), and so an appropriate constant c exists satisfying the above. Overall, we conclude that P O (0) = Ω(1) and P O ( rη / 2 ρ T ) = Ω( η ) .

Step 5. Final guarantee via application of Lemma 2: Using Eqs. (60) and (64) in Eq. (52) with Eq. (51), and combining the guarantee from Eq. (45), with probability 1 -exp( -Ω(max { d, ˜ d } )) :

<!-- formula-not-decoded -->

We set n T = n/ 2 , and so N = n -n T = n/ 2 (as in Algorithm 1). For ρ T, we note that it can be chosen arbitrarily large to reduce the second term in the error above. This is because any ρ T &gt; 0 will allow the teacher parameters G T , ˜ G T to recover the subspace spanned by U , ˜ U respectively, but a large choice of ρ T will make the operator norm small. This does not cause the filtering to change, since the threshold θ changes multiplicatively with ρ T (effectively scaling the picture in Figure 2).

The condition of n ≳ 1 η 2 max { d, ˜ d } (1 + γ -1 )(1 + ˜ γ -1 ) is inherited from Corollary 1 (to be able to use eq (45)). The additional condition on n , from the application of Lemma 2 to the above equation (similar to Eq. (33)), results in a larger factor than 1 / η 2 , hence is already satisfied.

Now we apply Lemma 2 on the above equation, and follow the argument similar to step 5 in Section D. An additional factor of √ r appears due to the norm being the chordal distance (frobenius norm). Using Eq. (56) and Eq. (65), we get that with probability 1 -exp( -Ω(max { d, ˜ d } )) , the error ERR ( G , ˜ G ) is upper bounded (up to constants) by:

<!-- formula-not-decoded -->

Finally, we plug in the values θ ∈ { 0 , ηr / 2 ρ T } to recover the terms T 0 , T 0 . 5 as stated in Theorem 1. Using Eq. (57) and (66), the scaling term of the error above becomes

<!-- formula-not-decoded -->

Using Eq. (58) and (67), the scaling term of the error above becomes

<!-- formula-not-decoded -->

The above describes both regimes of behavior, and why an extra factor of r appears in the term T 0 , compared to the term T 0 . 5 , in Theorem 1. This concludes the argument.

## H Discussion on robustness of the choice of filtering threshold

We note that the error achieved by teacher-based filtering can be fairly robust to the choice of θ , the filtering threshold. Our synthetic experiment in Figure 3a was conducted with a fixed, untuned threshold of θ = 0 . Further, we conduct an experiment measuring the sensitivity of the final error with respect to the choice of θ . In the setting of Figure 3a with n = 10000 samples, we fix η = 0 . 3

(in-line with the empirically observed clean fraction in CLIP data [11]) and (implicitly) vary the filtering threshold θ of the teacher-based filtering (by explicitly varying the fraction of data retained in the filtering step). The below table shows that the error of teacher-based filtering is relatively flat for values of θ in the vicinity of the optimal threshold θ ∗ . An analogous experiment on real data [11, Figure 2] makes a similar observation.

Table 1: Mean error vs. fraction of data retained.

| Fraction of data retained   | Mean error ( ± 1 σ ) ( × 10 - 4 )   |
|-----------------------------|-------------------------------------|
| 1%                          | 28 . 76 ± 4 . 00                    |
| 10%                         | 11 . 79 ± 1 . 20                    |
| 20%                         | 9 . 85 ± 1 . 39                     |
| 30%                         | 9 . 08 ± 1 . 15                     |
| 40%                         | 8 . 97 ± 1 . 09                     |
| 50%                         | 8 . 71 ± 1 . 05                     |
| 100%                        | 16 . 51 ± 2 . 03                    |

## I Discussion on the potential of robust statistics for the analysis of filtering

An initial instinct based on Figure 2 is to use ideas from robust statistics. As discussed in Remark 6.2, we can expect D 0 and D 1 to be well-separated, which means there will exist some θ ∈ R (a reasonable guess is θ ≈ r / 2 ) such that the selected data is mostly clean. After filtering, the picture resembles the robust statistics setting: an α corruption on the clean distribution for some small α . This is a reasonable approach overall, but has two shortcomings. First , this approach will not achieve zero error as n →∞ . We are shooting for f ( η ) · 1 / √ n which is better than 1 / √ n + g ( η ) , since the latter is non-zero even when n → ∞ . This approach will end up getting the latter. This is because the canonical rate in robust statistics is √ d / n sel + α . Under filtering, n sel and α are functions of θ . One can determine the optimal θ to balance the tradeoff, but to get a final rate of the form f ( η ) · 1 / √ n , this will require some conditions on n, η (possibly η bigger than a threshold, and n smaller than a threshold). Since our case has stochastic corruption which is weaker than adversarial corruption, we can expect to prove something for all n and all η . Second , this approach performs a 'reductive" operation of treating data as only clean v/s corrupted, and assuming the corrupted part provides no signal. This is a closely linked argument to the first one above. The crucial observation is that the right tail of the corrupted data (i.e. D 0 in Figure 2) actually provides 'close to clean' samples. This is because these just happened to be samples such that the z, ˜ z - albeit independently sampled in a high-dimensional space - happened to have a high inner product (small angle). Our adopted approach, based on the conditional properties of the Gaussian distribution, formalizes this intuition that the right tail of D 0 also provides signal.
## Knowledge Distillation of Uncertainty using Deep Latent Factor Model

Sehyun Park Department of Statistics Seoul National University ps\_hyen@snu.ac.kr

Jongjin Lee Samsung Research ga0408@snu.ac.kr

Ilsang Ohn Department of Statistics Inha University ilsang.ohn@inha.ac.kr

Yunseop Shin Department of Statistics Seoul National University dbstjq48@snu.ac.kr

Yongdai Kim ∗ Seoul National University

Department of Statistics ydkim0903@gmail.com

## Abstract

Deep ensembles deliver state-of-the-art, reliable uncertainty quantification, but their heavy computational and memory requirements hinder their practical deployments to real applications such as on-device AI. Knowledge distillation compresses an ensemble into small student models, but existing techniques struggle to preserve uncertainty partly because reducing the size of DNNs typically results in variation reduction. To resolve this limitation, we introduce a new method of distribution distillation (i.e. compressing a teacher ensemble into a student distribution instead of a student ensemble) called Gaussian distillation, which estimates the distribution of a teacher ensemble through a special Gaussian process called the deep latent factor model (DLF) 2 by treating each member of the teacher ensemble as a realization of a certain stochastic process. The mean and covariance functions in the DLF model are estimated stably by using the expectation-maximization (EM) algorithm. By using multiple benchmark datasets, we demonstrate that the proposed Gaussian distillation outperforms existing baselines. In addition, we illustrate that Gaussian distillation works well for fine-tuning of language models and distribution shift problems.

## 1 Introduction

While DNNs have succeeded tremendously in various AI tasks, the rapid increase in their model sizes has raised a concern about high computational resource demands, which limits their applications to real world applications such as on-device AI [1], and thus developers have increasingly compressed large-scale language models into much smaller models [1, 2, 3, 4]. A representative tool for compression is knowledge distillation (KD), which constructs a smaller DNN that mimics a given large-scale DNN [5].

Another concern is that the inherent over-parameterization of a single DNN makes them susceptible to overfitting, leading to overconfident predictions. When training predictive models, it is essential to learn models not only accurate but also reliable. For reliable prediction, proper quantification of uncertainty has become an important topic in AI research [6, 7, 8]. Deep ensemble (an ensemble of multiple DNNs) has received much attention not only for its strong predictive performance but

∗ Corresponding author.

2 The source code of DLF is publicly available at https://github.com/sehyun1094/DLF

also for its ability to quantify prediction uncertainty [9, 10]. An ensemble of DNNs can mitigate overconfident predictions by reflecting the uncertainty (i.e., the variation of multiple predictions made by members of an ensemble) when making a final decision.

Since deep ensemble requires even more computational resources than DNNs, KD of deep ensemble is necessary for improving its applicability. Several works have focused on distilling a given teacher ensemble to a student ensemble instead of distilling a teacher ensemble to a single student DNN to keep the uncertainty as much as possible [8, 11, 12, 13, 14, 15]. Algorithms for KD of deep ensemble can be roughly categorized into two approaches: one-to-one distillation and distribution distillation. One-to-one distillation compresses each member in a teacher ensemble to a smaller DNN, which becomes a member of a student ensemble. Various weight-sharing architectures for student DNNs, along with their corresponding learning algorithms, have been proposed [11, 13, 14, 16]. On the other hand, distribution distillation treats each ensemble member in a teacher ensemble as an independent realization of a certain distribution whose parameters are modeled by a student DNN. [17] and [18] assume that the conditional class probability vector of each member in a teacher ensemble follows a Dirichlet distribution and devise a method to estimate the parameters in the Dirichlet distribution using a student DNN.

There are still limitations in existing KD methods for deep ensemble. One-to-one distillation methods tend to lose a significant amount of uncertainty in a teacher ensemble when they compress large DNNs into smaller ones, while performance of Dirichlet distillation (the distribution distillation with a Dirichlet distribution) is inferior to one-to-one distillation partly [11, 14] because of instability in learning the parameters in the Dirichlet distribution.

The aim of this paper is to propose a new distribution distillation method that is numerically stable in learning and superior to other baselines in uncertainty quantification. In our proposed method, we treat each member in a teacher ensemble as an independent realization of a Gaussian process and estimate the mean and covariance functions of the Gaussian process based on observed predictions of members in a teacher ensemble. For this purpose, we propose the deep latent factor (DLF) model where the mean and covariance functions are modeled by a student DNN and implement an EM algorithm to estimate the maximum likelihood estimator (MLE) of the student DNN. We call our method Gaussian distillation .

Our contributions are summarized as follows.

- We propose a new distribution distillation method based on a specially designed Gaussian process called the DLF model that achieves superior performance in uncertainty quantification to other baselines.
- We develop an EM algorithm to estimate the student DNN in the DLF model. In particular, we propose a way of finding a good initial solution by maximizing the penalized complete log-likelihood.
- We do numerical experiments to show that Gaussian distillation outperforms other baselines for both regression and classification. We also illustrate that Gaussian distribution is a useful tool for fine-tuning language models.
- We apply the pre-trained DLF to distribution shift problems and show numerically that it outperforms baselines.

## 2 Preliminaries

## 2.1 Prediction uncertainty

In a nutshell, quantifying prediction uncertainty in supervised tasks involves efficiently estimating the predictive distribution of the output y given a new input denoted as p ( y | x n ew ) . The variation in the predictive distribution can be used as a measure of uncertainty in prediction.

A typical way of estimating the predictive distribution begins with a parametric generative model for the input and output pair. Let p ( y | x , θ ) be the conditional distribution of the output y ∈ Y given an input x ∈ X ⊂ R d , where θ ∈ Θ is an unknown parameter. Then, we try to estimate θ based on training data ( x 1 , y 1 ) , . . . , ( x m , y m ) such that p ( y | x , ˆ θ ) is as close as possible to p ∗ ( y | x ) , where ˆ θ

is an estimate of θ and p ∗ ( y | x ) is the true conditional distribution. For example, the MLE minimizes the empirical KL divergence between p ( y | x , θ ) and p ∗ ( y | x ) .

It is well known, however, that the variation in p ( y | x , ˆ θ ) is smaller than that in p ∗ ( y | x ) because p ( y | x , ˆ θ ) does not take into account the uncertainty in estimating ˆ θ . Thus, making a decision solely with p ( y | x , ˆ θ ) would lead in overconfident results. A proper uncertainty quantification in prediction should consider not only uncertainty in p ∗ ( y | x ) (aleatory) [6, 7] but also uncertainty in ˆ θ (epistemic).

A popular way of considering both aleatory and epistemic uncertainties in prediction is to use an ensemble. We construct multiple estimates ˆ θ 1 , . . . , ˆ θ n of θ and then estimate the predictive distribution as ˆ p ( y | x ) = ∑ n i =1 p ( y | x , ˆ θ i ) /n, which we call the averaged prediction model. For deep learning, the two most representative methods of constructing multiple estimates are deep ensemble [9, 19, 20] and Bayesian DNNs [21, 22, 23, 24, 25, 26, 27]. Deep ensemble generates multiple estimates by learning a DNN with different initial parameter, while Bayesian DNNs generate ˆ θ s from the posterior distribution. In this paper, we focus on deep ensemble, but our proposed method can be applied to Bayesian DNNs without modification.

## 2.2 Review of ensemble distillation

As mentioned in Introduction, deep ensemble has an intrinsic limitation in its practical applications due to high computational costs and times along with demands for substantial memory to store and process multiple prediction models. To resolve this problem, KD of deep ensemble has received much attention [5, 8, 11, 12, 13, 14, 15, 17]. A basic idea of KD of deep ensemble is to approximate large DNNs in a teacher ensemble by smaller student DNNs. A naive approach of KD of deep ensemble is to approximate the averaged prediction model ˆ p ( y | x ) of a teacher ensemble by a small single DNN [15, 28, 29]. This naive approach, however, does not perform well since it is hard to distill the uncertainty in ˆ p ( y | x ) into a single student DNN.

Aremedy is to distill a teacher ensemble into a student ensemble. Several methods have been proposed for this purpose, which can be roughly divided into two categories that are explained in the subsequent subsections.

## 2.2.1 One-to-one distillation

The main idea of one-to-one distillation is to construct multiple student models, each of which corresponds to each teacher model. That is for given n many teacher models p ( t ) i ( y | x ) , i = 1 , . . . , n, n many student models p ( s ) i ( y | x ) are constructed. To save computation time and memory further, various special neural network architectures for n student models p ( s ) i ( y | x ) , i = 1 , . . . , n have been proposed. Examples are Hydra [11], Batch Ensembles (BE) and Latent Batch Ensemble (LBE) [12, 14]. See Appendix A.1 for details.

## 2.2.2 Distribution distillation

Distribution distillation assumes that teacher models are independent realizations of a stochastic model with unknown parameters modeled by a student DNN and estimates the student DNN based on the prediction values of the teacher models [17, 18]. To be more specific, for classification problems, we assume that ( p ( t ) i ( y | x ) , y = 1 , . . . , c ) , i = 1 , . . . , n for a given x are independently generated from the Dirichlet distribution with parameters α 1 ( x ) , . . . , α c ( x ) and model these parameters by a student DNN. Once the student DNN is learned, ensemble members are generated from the learned Dirichlet distribution and aggregated in the prediction phase. We call this method Dirichlet distillation . See Appendix A.2 for details.

## 3 The Proposed Method

We propose a new method of distribution distillation. The main idea of the proposed method is that we treat members in a teacher ensemble as independent realizations of a Gaussian process and estimate the mean and covariance functions of the Gaussian process by a student DNN. Then, in the inference

phase, we generate ensemble members from the estimated Gaussian process. We call our proposed method Gaussian distillation . See Figure 1 for the overall process of Gaussian distillation.

Figure 1: Overall process of Gaussian distillation

<!-- image -->

A technical difficulty of this idea is to model and estimate the covariance function. To resolve this problem, we use the DLF, which is an extension of the standard linear factor model [30], where the mean and factor loading are modeled by a student DNN.

For the probabilistic model of data, we consider y = f ( x ) + ϵ, where ϵ ∼ N (0 , σ 2 ϵ ) for regression problems and p ( y | x ) = exp( f y ( x )) / ∑ c v =1 exp( f v ( x )) for classification problems. Thus, a teacher ensemble for regression problems consists of multiple teacher models for f as well as multiple estimates of σ 2 ϵ , while a teacher ensemble for classification problems consists of multiple teacher models for multivariate functions f ( · ) = ( f 1 ( · ) , . . . , f c ( · )) .

## 3.1 Deep Latent Factor model

In this subsection, we introduce special Gaussian processes for f ( · ) and f ( · ) , respectively.

Univariate case The DLF model for a univariate random function f : X → R is defined as

<!-- formula-not-decoded -->

where µ θ ( · ) : X → R is the mean function, Φ θ ( · ) : X → R q is the factor loading function and Z ∼ N q ( 0 , I q ) is the latent factor. Here, N q is the q -dimensional Gaussian distribution and I q is the q -dimensional identity matrix. In the DLF model, we set ( µ θ ( · ) , Φ θ ( · ) ⊤ ) by a student DNN parameterized by θ which has q +1 output nodes.

It is easy to see that the DLF is a Gaussian process with mean function µ θ ( · ) and covariance function Σ θ ( · , · ) = Φ θ ( · ) ⊤ Φ θ ( · ) . Once we have n many teacher models f 1 ( · ) , . . . , f n ( · ) , we assume them to be independent realizations of the DLF model and estimate the mean and factor loading functions.

Multivariate case The DLF model for a multivariate function f ( · ) = ( f 1 ( · ) , . . . , f c ( · )) ⊤ is defined as

<!-- formula-not-decoded -->

where µ θ ( · ) : X → R c is the mean function, Φ θ ( · ) : X → R q is the factor loading function, L ∈ R c × c is a lower-triangular matrix and Z ∼ MN c,q (0 , I c , I q ) . Here, MN c,q (0 , I c , I q ) is a matrix-variate Gaussian distribution. It can be shown that the DLF model is a multivariate Gaussian process MGP c ( µ θ , Σ , Λ) with the mean function µ θ ( · ) , covariance function Σ( · , · ) = Φ θ ( · ) ⊤ Φ θ ( · ) and parameter matrix Λ = LL ⊤ . For the definition of multivariate Gaussian process, see [31].

## 3.2 Estimation of the mean and factor loading

The main idea of Gaussian distillation is to estimate the mean and factor loading functions by maximizing the corresponding log-likelihood, assuming that teacher models are independent realizations

of the DLF. For optimization, we use the EM algorithm [32]. In this section, we explain the EM algorithm for Gaussian distillation. For ease of notation, we only consider the univariate DLF model, and refer to Appendix B.2.2 for the multivariate DLF.

Suppose that n many teacher models f 1 ( · ) , . . . , f n ( · ) are given. Gaussian distillation consists of three steps. The first step is to choose m -many design points D design = { x ( d ) 1 , . . . , x ( d ) m } . We will discuss how to choose the design points in Section 3.4. The second step is to calculate the vectors of prediction values of each teacher model at the design points to have f i = ( f i ( x ( d ) j ) , j = 1 , . . . , m ) ⊤ for i = 1 , . . . , n. The final step is to estimate the parameter θ in the DLF model assuming that f 1 ( · ) , . . . , f n ( · ) are independent realizations of a random function following the DLF model. Since f i s are independent Gaussian random vectors, the MLE can be obtained by use of the EM algorithm as follows.

To make the EM algorithm numerically stable, we consider the noisy DLF model which assumes that f i = ˜ f i + v i , where v i ∼ N m ( 0 , σ 2 f I m ) and ˜ f i = ( ˜ f i ( x 1 ) , . . . , ˜ f i ( x m )) ⊤ with ˜ f i ( · ) s following the DLF model. Specifically, each ˜ f i is expressed as ˜ f i ( · ) = µ θ ( · ) + Φ θ ( · ) ⊤ z i , where z i ∼ N q ( 0 , I q ) denotes the latent factor corresponding to the i -th function realization. Then, we obtain the MLE of the parameter θ in the mean and factor loading functions as well as σ 2 f . We abuse the notation to write θ = ( θ, σ 2 f ) unless there is any confusion.

The complete log-likelihood is given as

<!-- formula-not-decoded -->

where f 1: n = { f 1 , . . . , f n } , z 1: n = { z 1 , . . . , z n } , µ θ = ( µ θ ( x ( d ) 1 ) , . . . , µ θ ( x ( d ) m )) ⊤ and Φ θ = (Φ θ ( x ( d ) 1 ) , . . . , Φ θ ( x ( d ) m )) ⊤ is an m × q matrix.

For a given parameter θ ( t -1) at time t -1 , the E-step is to calculate the conditional expectation of the complete log-likelihood Q ( θ | θ ( t -1) ) = E z 1: n | f 1: n ,θ ( t -1) [ ℓ com ( θ | f 1: n , z 1: n )] , whose formula is given in Appendix B. In the M-step, we update θ ( t ) by a stochastic gradient descent algorithm on mini-batches. The EM algorithm is summarized in Algorithm 1 in Appendix B.

Choice of the initial parameter Note that the EM algorithm may converge to a local optimum, and the choice of an initial solution significantly impacts the final estimate. For the DLF model, where the factor loading involves a complex DNN structure, this issue becomes even worse. Moreover, the identifiability issue of the factor loading makes initializing the EM algorithm from a well-chosen starting point become even more crucial.

For searching a good initial solution, we pretrain the DLF model by maximizing the following penalized complete log-likelihood with respect to θ and z i s where

<!-- formula-not-decoded -->

for λ &gt; 0 , where z ′ 1: n are samples generated from the standard Gaussian distribution and D MMD is the Maximum Mean Discrepancy (MMD) with the RBF kernel. The term MMD is introduced to make the distribution of the estimated z 1: n similar to the standard Gaussian distribution.

KD for σ 2 ϵ For regression problems, we need a KD method for σ 2 ϵ . Let σ 2 ϵ, 1 , . . . , σ 2 ϵ,n be estimators of σ 2 ϵ provided by each teacher model. We assume that σ 2 ϵ,i , i = 1 , . . . , n are independently generated from the inverse gamma distribution and estimate the parameter in the distribution accordingly. In the inference phase, we generate ensemble members of σ 2 ϵ from the estimated inverse gamma distribution.

## 3.3 Comparison with other baselines

Comparison with Hydra Recall that we model ( µ θ ( · ) , Φ θ ( · ) ⊤ ) ⊤ by a DNN with q + 1 many heads. Note that the DLF model assumes f i ( · ) = µ θ ( · ) + Φ θ ( · ) ⊤ z i for i = 1 , . . . , n, and thus we

can interpret ( µ θ ( · ) , Φ θ ( · ) ⊤ ) ⊤ as the body and z i s as the model-specific weights at the head. In view of sharing the body, the DLF model is quite similar to the conventional multi-head structure used in Hydra [11]. The main difference is that the DLF model treats z i s as random quantities and thus integrates out before estimating the MLE while Hydra treats z i s as fixed effects and estimates θ and z i s simultaneously by minimizing a given loss. It is well-known that treating random effects as fixed effects is highly susceptible to bias [33, 34, 35, 36]. Our experimental results in Section 5 amply demonstrate that treating z i s as random is better than treating z i s as fixed effects.

Comparison with Dirichlet distillation At least, there are two advantages of Gaussian distillation compared to Dirichlet distillation. Gaussian distillation can be applied to both regression and classification models while Dirichlet distillation is only applicable to classification problems. The second advantage is that estimation of the mean and factor loading in the DLF model is easier than estimation of the parameter in the Dirichlet distribution owing to the nice EM algorithm. This stability makes Gaussian distillation perform better than Dirichlet distillation. The inferior performance of Dirichlet distillation compared to one-to-one distillation, as reported in [11, 13], is confirmed by our experiments in Section 5.

## 3.4 Choice of design points

Let ˆ µ ( · ) and ˆ Σ( · , · ) be the estimate of µ ∗ ( · ) and Σ ∗ ( · , · ) by the DLF model with design points D design . For a given x ∈ X , let ˆ p x and p ∗ , x be the distributions of f ( x ) under the assumption that f ( · ) is a Gaussian process with the parameters (ˆ µ, ˆ Σ) and ( µ ∗ , Σ ∗ ) , respectively. In Theorem G.3 in Appendix G.2, we prove that sup x ∈D design d 1 (ˆ p x , p ∗ , x ) converges to 0 as n →∞ if we choose the architecture of a student DNN for ( µ θ ( · ) , Φ θ ( · ) ⊤ ) appropriately, where d 1 is the ℓ 1 metric.

For x ̸∈ D design , if ˆ µ and µ ∗ , as well as ˆ Σ and Σ ∗ are (coordinate-wise) Lipschitz, it can be shown that d 1 (ˆ p x , p ∗ , x ) ≤ d 1 (ˆ p x (1) , p ∗ , x (1) ) + C ∥ x -x (1) ∥ for a positive constant C, where x (1) is the nearest point in D design to x . See Theorem G.3 in Appendix G.2. Note that the term ∥ x -x (1) ∥ is affected by the choice of design points. Suppose that x is a realization of a random vector X ∼ P . Then, the expected nearest-neighbor distance E X ∼ P ∥ X -X (1) ∥ becomes smaller when the design points are located in a higher-density region of P . This observation suggests that design points similar to test data would be better. Validation data (dataset whose distribution is the same as the training data used for learning a teacher ensemble) would be a promising candidate for the design points. See Appendix D.1.1 for numerical experiments.

## 4 Application to distribution shift problems

The pre-trained DLF can be applied to distribution shift problems. We say that given new data is shifted in distribution if the distribution of new data is different from that of training data. Distribution shift problems, whose aim is to efficiently learn a prediction model on new data when the size of new data is small, have been studied extensively [37, 38, 39, 40, 41, 42]. A popular method is to learn a DNN on training data first and retrain the head of the DNN on new data while the body is fixed [43, 44].

Note that the DLF model is given as

<!-- formula-not-decoded -->

for j = 1 , . . . , c, where z jkl s are independent standard Gaussian random variables. For distribution shift problems, we can treat (ˆ µ ( · ) , ˆ Φ( · ) ⊤ , ˆ L ) as a learned body and z j s are the weights in the prediction head. Then we learn only the weights of the head on new data while fixing the body. In Section 5.3, we show empirically that this method outperforms its competitors. The superior performance of the DLF model for distribution shift problems indicates that Gaussian distillation is good at not only uncertainty quantification but estimating the feature vector (ˆ µ ( · ) , ˆ Φ( · ) ⊤ ) .

## 5 Experiments

In this section, we investigate Gaussian distillation by analyzing multiple benchmark datasets. We compare Gaussian distillation with existing baselines including the naive distillation (one-to-one distillation without sharing weights between student DNNs, small-Ens), Hydra [11] and BE [12] for regression and classification problems as well as fine-tuning of language models in view of uncertainty quantification. For classification, we also evaluate Proxy-Dirichlet Distillation (Proxy-End 2 ) [18] and Ensemble Distillation via Flow Matching (EDFM) [45]. In addition, we show that a pre-trained DLF outperforms its competitors for distribution shift problems.

## 5.1 Uncertainty quantification for regression and classification problems

## 5.1.1 Regression case

Datasets We analyze six benchmark datasets from the UCI repository [46] including Boston housing, Concrete, Energy, Wine, Power Plant, and Kin8nm. Each dataset is randomly split into 90% training and 10% testing, and teacher models are trained following the experimental protocol of [47]. We repeat this procedure 10 times to obtain 10 measures of the evaluation metrics of each methods and report the averages (with the standard errors). See Appendix C.1 for details of implementation.

Results Table 1 presents the results of the four evaluation metrics (see Appendix C.1.2 for the definitions) for performance and uncertainty quantification. DLF outperforms Hydra and BE in most cases. Even when it is not the best, DLF is at least the second best. The coverage probabilities of Hydra and BE are sometimes much lower than those of DLF (Boston housing and Concrete for Hydra, and Boston housing and kim8nm for BE), which suggests that deterministic distillation methods fail to fully preserve the uncertainty in a teacher ensemble. This observation is not surprising since variation of smaller models is in general smaller than that of larger models and weight sharing would reduce the variation further. Thus, a way of adding additional uncertainty to a student ensemble is needed, and distribution distillation is such a solution.

Table 1: Results on UCI benchmark datasets. ( ∗ : closer to the coverage probability of a teacher ensemble is better)

| Metric                     | Method                          | Datasets                                                               | Datasets                                                               | Datasets                                                               | Datasets                                                               | Datasets                                                               | Datasets                                                                   |
|----------------------------|---------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------|
|                            |                                 | Boston housing                                                         | Concrete                                                               | Energy                                                                 | Wine                                                                   | Power                                                                  | Kin8nm                                                                     |
| RMSE ↓                     | Teachers small-Ens Hydra BE DLF | 2.5786 2.7280 (0.0184) 2.8346 (0.0835) 2.8375 (0.0729) 2.6687 (0.1700) | 5.6191 5.6952 (0.0494) 6.0558 (0.1366) 5.9777 (0.1475) 5.6047 (0.1771) | 0.5692 0.6367 (0.0182) 0.6549 (0.0360) 0.6661 (0.0354) 0.5659 (0.0239) | 0.5497 0.6002 (0.0083) 0.5689 (0.0114) 0.556 (0.0187) 0.5506 (0.0104)  | 4.2197 4.2430 (0.0864) 4.2284 (0.0074) 4.2367 (0.0041) 4.2211 (0.0010) | 0.0794 0.0865 (0.0007) 0.0932 (0.0023) 0.0962 (0.0059) 0.0825 (0.0010)     |
| NLL ↓                      | Teachers small-Ens Hydra BE DLF | 2.3850 2.4150 (0.0085) 2.4843 (0.0478) 2.4892 (0.0408) 2.4346 (0.1147) | 3.1134 3.1672 (0.0167) 3.2586 (0.0293) 3.2241 (0.0332) 3.1584 (0.0463) | 0.8533 0.9829 (0.0263) 0.9914 (0.0660) 0.9879 (0.0511) 0.8525 (0.0471) | 0.7980 0.8861 (0.0160) 0.8322 (0.0166) 0.8065 (0.0138) 0.8230 (0.0199) | 2.8586 2.8622 (0.0191) 2.8604 (0.0017) 2.8628 (0.0010) 2.8591 (0.0010) | -1.1109 -1.032 (0.0108) -0.9434 (0.0307) -0.9115 (0.0808) -1.0754 (0.0126) |
| CRPS ↓                     | Teachers small-Ens Hydra BE DLF | 1.4425 1.5233 (0.0087) 1.6041 (0.0494) 1.6158 (0.0490) 1.4317 (0.1029) | 2.9926 2.9953 (0.0304) 3.3084 (0.0639) 3.2320 (0.0894) 3.0622 (0.0954) | 0.3137 0.3461 (0.0077) 0.3622 (0.0216) 0.3617 (0.0168) 0.3163 (0.0119) | 0.2962 0.3307 (0.0067) 0.3075 (0.0040) 0.3020 (0.0059) 0.2980 (0.0043) | 2.3360 2.3392 (0.0321) 2.3405 (0.0048) 2.3483 (0.0024) 2.3364 (0.0010) | 0.0443 0.0475 (0.0005) 0.0518 (0.0012) 0.0532 (0.0035) 0.0458 (0.0005)     |
| 95% Coverage Probability ∗ | Teachers small-Ens Hydra BE DLF | 0.9608 0.9408 (0.0016) 0.8995 (0.0109) 0.9093 (0.0090) 0.9240 (0.0154) | 0.9515 0.9431 (0.0041) 0.9097 (0.0115) 0.9282 (0.0099) 0.9291 (0.0125) | 1.0000 0.9921 (0.0011) 0.9948 (0.0091) 0.9922 (0.0110) 1.0000 (0.0000) | 0.9750 0.9569 (0.0042) 0.9594 (0.0053) 0.9612 (0.0092) 0.9681 (0.0055) | 0.9697 0.9669 (0.0012) 0.9782 (0.0003) 0.9778 (0.0014) 0.9711 (0.0053) | 0.9610 0.9649 (0.0016) 0.9407 (0.0046) 0.9080 (0.0213) 0.9761 (0.0036)     |

## 5.1.2 Classification case

Datasets CIFAR-10 and CIFAR-100 consist of 50,000 training and 10,000 test images. In this experiment, the training data are further split into 80% training and 20% validation, and teacher models are trained on the training data and the number of epochs is determined by the validation

data. Implementation details of the distillation methods are given in Appendix C.2. Experiments are repeated with 5 different random initializations for each method.

Results As shown in Table 2, DLF outperforms the other baselines consistently in terms of not only uncertainty quantification but also accuracy. In particular, improvements of DLF with respect to ECE are noticeable. The definitions of the evaluation metrics are given in Appendix C.2.2.

Table 2: Results on CIFAR-10 and CIFAR-100.

| dataset   | method                                        | Acc(%) ↑                                                               | NLL ↓                                                                                  | ECE(%) ↓                                                        |
|-----------|-----------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| CIFAR-10  | Teachers small-Ens Hydra LBE Proxy-EnD 2 EDFM | 94.24 92.87 (0.35) 93.16 (0.06) 93.25 (0.26) 90.92 (0.28) 90.62 (0.23) | 0.1539 0.2377 (0.0019) 0.2660 (0.0063) 0.2480 (0.0053) 0.2861 (0.0027) 0.2858 (0.0025) | 0.9 3.93 (0.14) 4.15 (0.01) 4.11 (0.10) 2.08 (0.19) 2.78 (0.17) |
| CIFAR-10  | DLF                                           | 93.40 (0.14)                                                           | 0.2246 (0.0023)                                                                        | 2.79 (0.20)                                                     |
| CIFAR-10  | Teachers                                      | 81.36                                                                  | 0.7167                                                                                 | 1.41                                                            |
| CIFAR-10  | small-Ens                                     | 79.29 (0.26)                                                           | 1.0413 (0.0145)                                                                        | 12.90 (0.24)                                                    |
| CIFAR-10  | Hydra                                         | 77.42 (0.15)                                                           | 1.2912 (0.0272)                                                                        | 12.70 (0.37)                                                    |
| CIFAR-100 | LBE                                           | 79.58 (0.40)                                                           | 1.0110 (0.0087)                                                                        | 13.42 (0.42)                                                    |
| CIFAR-10  | Proxy-EnD 2                                   | 67.62 (0.22)                                                           | 1.2355 (0.0151)                                                                        | 7.35 (0.25)                                                     |
| CIFAR-10  | EDFM                                          | 64.17 (0.32)                                                           | 1.6741 (0.0242)                                                                        | 11.35 0.47)                                                     |
| CIFAR-10  | DLF                                           | 79.68 (0.23)                                                           | 0.8974 (0.0042)                                                                        | 9.45 (0.31)                                                     |
| CIFAR-10  |                                               |                                                                        |                                                                                        |                                                                 |
| CIFAR-10  |                                               |                                                                        |                                                                                        |                                                                 |
| CIFAR-10  |                                               |                                                                        |                                                                                        |                                                                 |
| CIFAR-10  |                                               |                                                                        |                                                                                        |                                                                 |
| CIFAR-10  |                                               |                                                                        |                                                                                        |                                                                 |

## 5.2 Application to fine-tuning of language models

In this section, we apply the proposed distillation framework to downstream binary classification tasks using pretrained language models. Given pre-trained teacher and student language models, fine-tuned teacher and student models are obtained using Low-Rank Adaptation (LoRA) [48]. For the teacher and student pre-trained language models, " RoBERTa " [49] and " DistilRoBERTa " [1] 3 are used. As a teacher ensemble, we obtain four fine-tuned models by combining LoRA and RoBERTa with randomly selected initializations for each task. Then, an ensemble of fine-tuned student language models is constructed by applying LoRA to DistilRoBERTa for each distillation method.

Datasets We analyze three GLUE [50] and SuperGLUE [51] sub-tasks: RTE, MRPC, and WiC. All three datasets are binary classification tasks. Implementation details of the distillation methods are given in Appendix C.3.

Results As shown in Table 3, Gaussian distillation outperforms Hydra and LBE with large margins. We conjecture that the performance gap of Gaussian distillation to Hydra and LBE would become larger when the complexity gap between teacher and student models becomes larger. This is a reasonable conjecture since smaller models could preserve less variations in teacher models. Gaussian distillation would add additional variations to the student models through variations of the latent vector Z .

## 5.3 Application to distribution shift problems

We compare DLF with two baselines which fine-tune only the head on new data while the body is learned by either (1) a standard DNN or (2) applying Hydra on training data of CIFAR-10. For distribution-shifted new data, we swap the labels of CIFAR-10 as is done by [44]. See implementation details in Appendix C.4. The results are given in Figure 2 which amply show that DLF is superior. It is interesting to see that DLF outperforms even when the sample size of the new data is large, which implies that the learned body by DLF is qualitatively different from those by DNN and Hydra. We do not know the reason but an implication is that Gaussian distillation is good at learning not only quantifying uncertainty but also learning the feature vector (i.e. the body).

3 https://www.huggingface.co/distilroberta-base

Table 3: Results on GLUE and SuperGLUE benchmark datasets

| dataset   | method                           | Acc (%) ↑                                                           | NLL ↓                                                                  | ECE (%) ↓                                                        |
|-----------|----------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------|
| RTE       | Teachers small-Ens Hydra LBE DLF | 75.09 67.15 (0.0057) 62.82 (0.1650) 65.97 (0.1406) 67.06 (0.1040)   | 0.8401 0.6739 (0.0124) 0.9034 (0.1733) 0.9235 (0.0664) 0.6658 (0.0762) | 17.70 13.09 (0.0148) 23.31 (0.4907) 25.63 (0.1909) 9.74 (0.5050) |
| MRPC      | Teachers small-Ens Hydra LBE DLF | 87.25 83.092 (0.0172) 82.19 (0.0357) 82.23 (0.0153) 83.094 (0.0035) | 0.3435 0.4596 (0.0396) 0.6429 (0.1274) 0.6527 (0.0544) 0.4526 (0.0113) | 4.77 9.84 (0.0164) 11.79 (0.0136) 13.6 (0.0114) 10.72 (0.0392)   |
| WiC       | Teachers small-Ens Hydra LBE DLF | 68.03 65.02 (0.0062) 65.05 (0.0210) 65.36 (0.1209) 66.18 (0.0146)   | 0.6395 0.7674 (0.0206) 1.1628 (0.0355) 0.8809 (0.0950) 0.7706 (0.0543) | 9.14 16.69 (0.0116) 26.26 (0.0191) 20.28 (0.0300) 15.99 (0.0198) |

Figure 2: Comparison of performances of the three learning algorithms (DNN, Hydra and DLF) pretrained on CIFAR-10 and fine-tuned on CIFAR10-Flip as the sample size of CIFAR 10-Flip data varies. The solid curves are the means and the shaded bands are the min-max spreads obtained from 5 training models on 5 randomly selected new data of CIFAR10-Flip.

<!-- image -->

## 5.4 Ablation studies

In Appendix D, we present the results of ablation studies including the sensitivity of Gaussian distillation to the choice of design points, the dimension of latent factor, the architecture size of student DNNs and the number of ensemble members used in a teacher and student ensemble. In addition, we compare the results when the initial solution is randomly selected in Gaussian distillation.

## 6 Conclusion

We proposed a novel method for distilling deep ensembles, specifically addressing the challenges associated with computational costs, inference time, and storage capacities inherent in traditional deep ensemble approaches. The key innovation lies in modeling the covariance structure of deep ensembles through the DLF model, enabling efficient preservation of uncertainty in a teacher ensemble with significantly reduced inference costs.

There are several future research topics. First, in this paper, we only focused on deep ensembles. It would be valuable to consider Bayesian DNNs, as they provide a framework for uncertainty quantification [25, 26, 27] and can potentially serve as a prior for on-device posterior updates. Second, for distillation of a fine-tuned language model, we used DistilRoBERTa [1], a pretrained distilled language model. It would be promising to distill the pretrained language model and the model for LoRA simultaneously. Third, the DLF could be used for online Bayesian learning by approximating the posterior with respect to old data by the DLF and using it for the prior of new data. We will pursue this idea in a near future.

## Acknowledgements

This work was partly supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. 2022R1A5A7083908), the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2025-00556079), and by Institute of Information &amp; communications Technology Planning &amp; Evaluation (IITP) grant funded by the Korea government(MSIT) [NO.RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)].

## References

- [1] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 , 2019.
- [2] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543 , 2023.
- [3] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023.
- [4] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219 , 2024.
- [5] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 , 2015.
- [6] Yarin Gal et al. Uncertainty in deep learning. 2016.
- [7] Andrey Malinin and Mark Gales. Predictive uncertainty estimation via prior networks. Advances in neural information processing systems , 31, 2018.
- [8] Zelda E Mariet, Rodolphe Jenatton, Florian Wenzel, and Dustin Tran. Distilling ensembles improves uncertainty estimates. In Third symposium on advances in approximate bayesian inference , 2020.
- [9] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems , 30, 2017.
- [10] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261 , 2019.
- [11] Linh Tran, Bastiaan S Veeling, Kevin Roth, Jakub Swiatkowski, Joshua V Dillon, Jasper Snoek, Stephan Mandt, Tim Salimans, Sebastian Nowozin, and Rodolphe Jenatton. Hydra: Preserving ensemble diversity for model distillation. arXiv preprint arXiv:2001.04694 , 2020.
- [12] Yeming Wen, Dustin Tran, and Jimmy Ba. Batchensemble: an alternative approach to efficient ensemble and lifelong learning. arXiv preprint arXiv:2002.06715 , 2020.
- [13] Giung Nam, Jongmin Yoon, Yoonho Lee, and Juho Lee. Diversity matters when learning from ensembles. Advances in neural information processing systems , 34:8367-8377, 2021.
- [14] Giung Nam, Hyungi Lee, Byeongho Heo, and Juho Lee. Improving ensemble distillation with weight averaging and diversifying perturbation. arXiv preprint arXiv:2206.15047 , 2022.
- [15] Hailin Zhang, Defang Chen, and Can Wang. Adaptive multi-teacher knowledge distillation with meta-learning. In 2023 IEEE International Conference on Multimedia and Expo (ICME) , pages 1943-1948. IEEE, 2023.
- [16] Martin Ferianc and Miguel Rodrigues. Simple regularisation for uncertainty-aware knowledge distillation. arXiv preprint arXiv:2205.09526 , 2022.

- [17] Andrey Malinin, Bruno Mlodozeniec, and Mark Gales. Ensemble distribution distillation. arXiv preprint arXiv:1905.00076 , 2019.
- [18] Max Ryabinin, Andrey Malinin, and Mark Gales. Scaling ensemble distribution distillation to many classes with proxy targets. Advances in Neural Information Processing Systems , 34:6023-6035, 2021.
- [19] Thomas G Dietterich. Ensemble methods in machine learning. In International workshop on multiple classifier systems , pages 1-15. Springer, 2000.
- [20] Olivier Laurent, Adrien Lafage, Enzo Tartaglione, Geoffrey Daniel, Jean-Marc Martinez, Andrei Bursuc, and Gianni Franchi. Packed-ensembles for efficient uncertainty estimation. arXiv preprint arXiv:2210.09184 , 2022.
- [21] Radford M Neal and Radford M Neal. Monte carlo implementation. Bayesian learning for neural networks , pages 55-98, 1996.
- [22] Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11) , pages 681-688. Citeseer, 2011.
- [23] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning , pages 1050-1059. PMLR, 2016.
- [24] Andrew G Wilson and Pavel Izmailov. Bayesian deep learning and a probabilistic perspective of generalization. Advances in neural information processing systems , 33:4697-4708, 2020.
- [25] Pavel Izmailov, Sharad Vikram, Matthew D Hoffman, and Andrew Gordon Gordon Wilson. What are bayesian neural network posteriors really like? In International conference on machine learning , pages 4629-4640. PMLR, 2021.
- [26] Mrinank Sharma, Sebastian Farquhar, Eric Nalisnick, and Tom Rainforth. Do bayesian neural networks need to be fully stochastic? In International Conference on Artificial Intelligence and Statistics , pages 7694-7722. PMLR, 2023.
- [27] Insung Kong, Dongyoon Yang, Jongjin Lee, Ilsang Ohn, Gyuseung Baek, and Yongdai Kim. Masked bayesian neural networks: Theoretical guarantee and its posterior inference. In International conference on machine learning , pages 17462-17491. PMLR, 2023.
- [28] Takashi Fukuda, Masayuki Suzuki, Gakuto Kurata, Samuel Thomas, Jia Cui, and Bhuvana Ramabhadran. Efficient knowledge distillation from an ensemble of teachers. In Interspeech , pages 3697-3701, 2017.
- [29] Kisoo Kwon, Hwidong Na, Hoshik Lee, and Nam Soo Kim. Adaptive knowledge distillation based on entropy. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 7409-7413. IEEE, 2020.
- [30] David J Bartholomew, Martin Knott, and Irini Moustaki. Latent variable models and factor analysis: A unified approach , volume 904. John Wiley &amp; Sons, 2011.
- [31] Zexun Chen, Jun Fan, and Kuo Wang. Multivariate gaussian processes: definitions, examples and applications. Metron , 81(2):181-191, 2023.
- [32] Donald B Rubin and Dorothy T Thayer. Em algorithms for ml factor analysis. Psychometrika , 47:69-76, 1982.
- [33] Norman E Breslow and Xihong Lin. Bias correction in generalised linear mixed models with a single component of dispersion. Biometrika , 82(1):81-91, 1995.
- [34] Xihong Lin and Norman E Breslow. Bias correction in generalized linear mixed models with multiple components of dispersion. Journal of the American Statistical Association , 91(435):1007-1016, 1996.

- [35] Bas Engel. A simple illustration of the failure of pql, irreml and aphl as approximate ml methods for mixed models for binary data. Biometrical Journal: Journal of Mathematical Methods in Biosciences , 40(2):141-154, 1998.
- [36] Anders Skrondal and Sophia Rabe-Hesketh. Generalized latent variable modeling: Multilevel, longitudinal, and structural equation models . Crc Press, 2004.
- [37] Qi Lei, Wei Hu, and Jason Lee. Near-optimal linear regression under distribution shift. In International Conference on Machine Learning , pages 6164-6174. PMLR, 2021.
- [38] Ruihan Wu, Chuan Guo, Yi Su, and Kilian Q Weinberger. Online adaptation to label distribution shift. Advances in Neural Information Processing Systems , 34:11340-11351, 2021.
- [39] Yong Bai, Yu-Jie Zhang, Peng Zhao, Masashi Sugiyama, and Zhi-Hua Zhou. Adapting to online label shift with provable guarantees. Advances in Neural Information Processing Systems , 35:29960-29974, 2022.
- [40] Dheeraj Baby, Saurabh Garg, Tzu-Ching Yen, Sivaraman Balakrishnan, Zachary Lipton, and YuXiang Wang. Online label shift: Optimal dynamic regret meets practical algorithms. Advances in Neural Information Processing Systems , 36:65703-65742, 2023.
- [41] Saurabh Garg, Nick Erickson, James Sharpnack, Alex Smola, Sivaraman Balakrishnan, and Zachary Chase Lipton. Rlsbench: Domain adaptation under relaxed label shift. In International Conference on Machine Learning , pages 10879-10928. PMLR, 2023.
- [42] Elan Rosenfeld and Saurabh Garg. (almost) provable error bounds under distribution shift via disagreement discrepancy. Advances in Neural Information Processing Systems , 36:2876128784, 2023.
- [43] Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Finetuning can distort pretrained features and underperform out-of-distribution. arXiv preprint arXiv:2202.10054 , 2022.
- [44] Yoonho Lee, Annie S Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical fine-tuning improves adaptation to distribution shifts. arXiv preprint arXiv:2210.11466 , 2022.
- [45] Jonggeon Park, Giung Nam, Hyunsu Kim, Jongmin Yoon, and Juho Lee. Ensemble distribution distillation via flow matching. In Forty-second International Conference on Machine Learning .
- [46] Arthur Asuncion, David Newman, et al. Uci machine learning repository, 2007.
- [47] Thang Bui, Daniel Hernández-Lobato, Jose Hernandez-Lobato, Yingzhen Li, and Richard Turner. Deep gaussian processes for regression using approximate expectation propagation. In International conference on machine learning , pages 1472-1481. PMLR, 2016.
- [48] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR , 1(2):3, 2022.
- [49] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 , 2019.
- [50] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461 , 2018.
- [51] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems , 32, 2019.
- [52] Kaare Brandt Petersen, Michael Syskind Pedersen, et al. The matrix cookbook. Technical University of Denmark , 7(15):510, 2008.

- [53] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [54] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [55] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146 , 2016.
- [56] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning , pages 1321-1330. PMLR, 2017.
- [57] Adam X Yang, Maxime Robeyns, Xi Wang, and Laurence Aitchison. Bayesian low-rank adaptation for large language models. arXiv preprint arXiv:2308.13111 , 2023.
- [58] Jose G Moreno-Torres, Troy Raeder, Rocío Alaiz-Rodríguez, Nitesh V Chawla, and Francisco Herrera. A unifying view on dataset shift in classification. Pattern recognition , 45(1):521-530, 2012.
- [59] Yann Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N , 7(7):3, 2015.
- [60] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [61] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning , pages 6105-6114. PMLR, 2019.
- [62] Minwoo Chae, Dongha Kim, Yongdai Kim, and Lizhen Lin. A likelihood approach to nonparametric estimation of a singular distribution using deep generative models. Journal of Machine Learning Research , 24(77):1-42, 2023.
- [63] Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with relu activation function. 2020.
- [64] Wing Hung Wong and Xiaotong Shen. Probability inequalities for likelihood ratios and convergence rates of sieve mles. The Annals of Statistics , pages 339-362, 1995.
- [65] Yongdai Kim, Ilsang Ohn, and Dongha Kim. Fast convergence rates of deep neural networks for classification. Neural Networks , 138:179-197, 2021.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In Introduction, we provide the method of this paper and its key contributions. Abstract also briefly mentions these points.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss a potential limitation of our work in Section 6.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: In Appendix G, we provide the full set of assumption and a complete proof of the main theorem in this paper. Also, for the theoretical claims in Section 3.4, we provide complete mathematical proofs in Appendix G.2.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: In Appendix C, we provide details of our experimental setups.

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

Justification: We provide the source codes of our proposed algorithm in the supplementary material. Furthermore, we will publicly upload them after acceptance.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: In Appendix C, we provide details of our experimental setups.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In Section 5, we report the mean and standard deviation from multiple runs. And in Section 5.3 and Appendix D, we also plot the max-min area in figures.

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

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: In Appendix C, we report the number of parameters for each regression and classification case. And we provide the used computing resources in Appendix C.5

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We thoroughly and carefully checked the Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We give several societal broader impacts of our work in Section 6.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: We give several societal broader impacts of our work in Section 6.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We mention all the owners/references/urls of methods/codes/datasets used in Section 5 and Appendix C.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We provide the source codes of our proposed algorithm in the supplementary material. Furthermore, we will publicly upload them after acceptance.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This study does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This study does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## APPENDIX

## A Review of Ensemble Distillation

## A.1 One-to-one Distillation

For given teacher members p ( t ) i , i = 1 , . . . , n , student p ( s ) i , i = 1 , . . . , n are trained to minimize

<!-- formula-not-decoded -->

where E x is the expectation operator of a certain probability distribution on the input space X . The following two methods use specially designed DNNs for student models which share the weights between student models.

## A.1.1 Hydra

Hydra [11] employs a multi-head architecture in which a single shared body network extracts common features and a set of n distinct linear heads generates predictions for each ensemble member. Formally, the student DNN is parameterized by

<!-- formula-not-decoded -->

where θ body is used across all heads. At inference, the body is evaluated once, and each head h θ head ,i ( · ) produces a member-specific output p ( s ) i ( y | x ) = h θ head ,i ( b θ body ( x ) ) . This design captures ensemble diversity while reducing both computation and memory compared to maintaining n independent networks.

## A.1.2 Batch Ensemble and Latent Batch Ensemble

[12] introduces Batch Ensemble (BE) to reduce the memory and computational burden of deep ensembles. In this architecture, all student networks share a core weight matrix W l at each layer, while individual ensemble members are differentiated by rank-one perturbations. Specifically, for the i -th student at layer l ,

<!-- formula-not-decoded -->

where r l i ∈ R d out and s l i ∈ R d in modulate the rows and columns of W l , respectively. This construction enables each sub-network to maintain member-specific behaviors while reusing the majority of parameters, yielding significant savings in both storage and inference costs compared to training n independent models.

Building on BE, [14] proposes Latent Batch Ensemble (LBE), which further compresses the ensemble at inference time. Instead of maintaining n distinct perturbations, LBE computes the average rank-one mask across all students:

<!-- formula-not-decoded -->

The resulting single student network requires only one forward pass per input while capturing the ensemble's mean perturbation in weight space. Empirical results demonstrate that LBE matches or exceeds the calibration performance of standard BE, with inference cost reduced by a factor of n . However, the proposed Latent Batch Ensemble is a specialized method designed for ensemble distillation for classification problems.

## A.2 Distribution Distillation

Distribution distillation frames the output of an ensemble at the input x as samples from an inputdependent Dirichlet distribution [17, 18]. Let f i , i = 1 , . . . , n be given teacher models for a classification task. For a given x , let π i ( x ) = softmax ( f i ( x )) , and we assume that π 1 ( x ) , . . . , π n ( x )

independently follow the Dirichlet distribution with parameter α ( x ) = ( α 1 ( x ) , . . . , α c ( x )) . Then, we model α ( x ) by a student DNN Ψ ( x | θ ) and estimate θ by maximizing

<!-- formula-not-decoded -->

where p ( π | α ) is the density of the Dirichlet distribution with parameter α . See [17, 18] for details.

## A.2.1 Proxy-Dirichlet Distillation

[18] proposed Proxy-Dirichlet Distillation (Proxy-EnD 2 ) to mitigate the convergence difficulties of standard Dirichlet distillation [17] when the number of classes is very high. The method constructs a proxy Dirichlet distribution from the teacher models and trains the student DNN Ψ ( x | θ ) by minimizing the reverse KL divergence between p ( π | Ψ ( x | θ )) and the proxy distribution.

The proxy distribution is obtained from teacher models as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, estimate θ by minimizing

<!-- formula-not-decoded -->

where ˆ α ( · ) = [ˆ α 1 ( · ) , . . . , ˆ α c ( · )] .

## A.2.2 Ensemble Distribution Distillation via Flow Matching

Ensemble Distribution Distillation via Flow Matching (EDFM) [45] models a conditional distribution of logits h 1 given x over R c . The method constructs the target distribution p 1 ( h | x ) as an empirical distribution using f i ( x ) .

They model using Flow matching such that

<!-- formula-not-decoded -->

where Ψ θ is student DNN.

Then, the network is trained by minimizing the conditional flow matching loss

<!-- formula-not-decoded -->

where h 0 ∼ N c (0 , σ 2 I c ) , h 1 ( x ( d ) j ) ∼ p 1 ( ·| x ( d ) j ) , t ∼ U [0 , 1] , h t ( x ( d ) j ) = t h 1 ( x ( d ) j ) + (1 -t ) h 0

and λ denotes a time dependent weighting function.

Sampling proceeds by numerically integrating the learned flow Ψ θ ( t, h t ( x ) , x ) with a standard ordinary differential equation (ODE) solver. For more detailed information, see [45].

## B Details of Gaussian distillation

## B.1 EMalgorithm for the univariate DLF model

The main idea of Gaussian distillation is to estimate the mean and factor loading functions based on given teacher models assuming that teacher models are independent realizations of the DLF and estimate the parameter in the student DNNs by maximizing the corresponding log-likelihood. For optimization, we use the EM algorithm [32].

<!-- formula-not-decoded -->

Suppose that n many teacher models f 1 ( · ) , . . . , f n ( · ) are given. Gaussian distillation consists of three steps. The first step is to choose m -many design points D design = { x ( d ) 1 , . . . , x ( d ) m } . The second step is to calculate the vectors of prediction values of each teacher model at the design points to have f i = ( f i ( x ( d ) j ) , j = 1 , . . . , m ) ⊤ for i = 1 , . . . , n. The final step is to estimate the parameter θ in the DLF by the MLE assuming that f 1 ( · ) , . . . , f n ( · ) are independent realizations of a random function following the DLF model. Since f i s are independent Gaussian random vectors, the MLE can be obtained by use of the EM algorithm as follows.

To make the EM algorithm numerically stable, we consider the noisy DLF model which assumes that f i = ˜ f i + v i , where v i ∼ N m ( 0 , σ 2 f I m ) and ˜ f i = ( ˜ f i ( x 1 ) , . . . , ˜ f i ( x m )) ⊤ with ˜ f i ( · ) s following the DLF model. Specifically, each ˜ f i is expressed as ˜ f i ( · ) = µ θ ( · ) + Φ θ ( · ) ⊤ z i , where z i ∼ N q ( 0 , I q ) denotes the latent factor corresponding to the i -th function realization. Then, we obtain the MLE of the parameter θ in the mean and factor loading functions as well as σ 2 f . We abuse the notation to write θ = ( θ, σ 2 f ) unless there is any confusion.

The complete log-likelihood is given as

<!-- formula-not-decoded -->

where f 1: n = { f 1 , . . . , f n } , z 1: n = { z 1 , . . . , z n } , µ θ = ( µ θ ( x ( d ) 1 ) , . . . , µ θ ( x ( d ) m )) ⊤ and Φ θ = (Φ θ ( x ( d ) 1 ) , . . . , Φ θ ( x ( d ) m )) ⊤ is an m × q matrix.

Thus, for a given parameter θ ( t -1) at time t -1 , the E-step is to calculate the conditional expectation of the complete log-likelihood which is given as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the M-step, we usually update θ ( t ) by maximizing Q ( θ | θ ( t -1) ) . Instead of maximizing Q ( θ | θ ( t -1) ) , we update θ by using a stochastic gradient descent algorithm (i.e., a gradient descent algorithm on a given mini-batches). The EM algorithm is summarized in Algorithm 1.

Algorithm 1: EM algorithm for the univariate DLF model

Input: Design points D design = { x ( d ) 1 , . . . , x ( d ) m } , Teacher ensemble members f 1 ( · ) , . . . , f n ( · ) , learning rate η &gt; 0 Output: Estimated parameter θ Initialize parameter: θ (0) for t = 1 , 2 . . . T do Shuffle the dataset D design and divide into mini-batches {B 1 , . . . , B M } for l = 1 , 2 , . . . , M do Calculate prediction values of teacher model f 1: n, B l , where f 1: n, B l = ( f 1: n ( x design j ) , x design j ∈ B l ) . Calculate E [ z | θ ( t -1) ] and E [ zz ⊤ | θ ( t -1) ] using f 1: n, B l Q ( θ | θ ( t -1) ) := E z 1: n | f 1: n, B l ,θ ( t -1) [ ℓ com ( θ ; f 1: n, B l , z 1: n )] Calculate gradient g ( t ) := ∇ θ ( -Q ( θ | θ ( t -1) ) ) Update θ ( t ) ← θ ( t -1) -η · g ( t ) end end

## B.2 Multivariate DLF model

In this section, we provide details of the multivariate DLF model introduced in Section 3.1. With given the design points { x ( d ) 1 , . . . , x ( d ) m } , let f = ( f ( x ( d ) j ) , j = 1 , . . . , m ) ⊤ and µ θ = ( µ θ ( x ( d ) j ) , j = 1 , . . . , m ) ⊤ be m × c matrices, and let Φ θ = ( Φ θ ( x ( d ) j ) , j = 1 , . . . , m ) ⊤ be an m × q matrix. Then, we assume that f follows the matrix-variate Gaussian distribution

<!-- formula-not-decoded -->

Using the properties of the matrix-variate Gaussian distribution, we can vectorize f , thereby allowing the multivariate DLF model to be handled the same as the univariate case.

## B.2.1 Vectorization

According to Definition 4 in [31], the vectorization of f can be expressed as follows:

<!-- formula-not-decoded -->

where ⊗ denotes the Kronecker product. This association arises from a factorization of the covariance matrix of the multivariate Gaussian distribution into the Kronecker product of matrices Φ ⊤ θ Φ θ and LL ⊤ .

## B.2.2 EMalgorithm for the multivariate DLF model

We explain the EM algorithm for training the multivariate DLF model in the same way as in Appendix B.1. In the multivariate case, matrix vectorization and its associated properties serve as the central tools, as detailed in Section 10.2.2 of [52].

Suppose that n teacher ensemble members f 1 ( · ) , . . . , f n ( · ) are given, where each f i ( · ): X → R c is a multivariate function. As in the univariate case, KD via the multivariate DLF involves three steps. First, we choose m design points D design = { x ( d ) 1 , . . . , x ( d ) m } . The second step is to calculate the prediction values of n many ensemble members at the design points to have f i = ( f i ( x ( d ) j ) , j = 1 , . . . , m ) ⊤ for i = 1 , . . . , n. Here, unlike in the univariate case, each f i should be regarded as an m × c matrix. Similarly to the univariate case, the final step employs the EM algorithm to obtain the MLE.

Similarly to the univariate case, we assume that f i = ˜ f i + v i , where v i ∼ MN m,c ( 0 , σ 2 f I m , I c ) and ˜ f i = ( ˜ f i ( x 1 ) , . . . , ˜ f i ( x m )) ⊤ with ˜ f i ( · ) s following the multivariate DLF model. From the property of the matrix-variate Gaussian distribution, we can rewrite the following vec( f i ) = vec( ˜ f i ) +

vec( v i ) , where vec( v i ) ∼ N mc ( 0 , σ 2 f I mc ) . And in the case of multivariate case, we abuse the notation to write θ ( t ) := ( θ ( t ) , L ( t ) , σ f ( t ) ) like the univariate case.

From this, the complete log-likelihood is given as

<!-- formula-not-decoded -->

where B i = vec( f i ) -vec( µ θ ) -( Φ θ ⊗ L ) vec( z i ) , f 1: n = { f 1 , . . . , f n } , z 1: n = { z 1 , . . . , z n } , µ θ = ( µ θ ( x ( d ) 1 ) , . . . , µ θ ( x ( d ) m )) ⊤ is an m × c matrix and Φ θ = (Φ θ ( x ( d ) 1 ) , . . . , Φ θ ( x ( d ) m )) ⊤ is an m × q matrix. Thus, for given parameter θ ( t -1) at time t -1 , the E-step is to calculate the conditional expectation of the complete log-likelihood which is given as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

In the M-step, instead of maximizing Q ( θ | θ ( t -1) ) , we update θ by use of a stochastic gradient descent algorithm.

## C Experimental details

In this section, we describe the overall setup of our experiments in detail, focusing on the datasets, model architectures, training procedures, and evaluation metrics used in regression and classification problems. Four baselines are considered: (i) a small ensemble of lightweight networks ('small-Ens'), (ii) Hydra, and (iii) BE for regression and LBE for classification.

## C.1 Regression case

We consider the standard regression problem

<!-- formula-not-decoded -->

Suppose that there are n many teacher models f ( t ) i ( · ) and σ ( t ) ϵ,i . Then, the predictive distribution of y given x is constructed as p ( t ) ( y | x ) = 1 n ∑ n i =1 N ( y | f ( t ) i ( x ) , σ ( t ) ϵ,i 2 ) , where N ( y | µ, σ 2 ) is the density of the Gaussian distribution with mean µ and variance σ 2 The predictive distribution p ( s ) ( y | x ) based on student ensemble members is defined similarly.

We obtain 50 teacher models of DNNs with two hidden layers and 100 nodes at each layer which are learned by minimizing the sum of squared residuals of the training data with 50 randomly selected initial solutions. For the design points used in the distillation, we use the training data themselves. The architecture of student models comprises of an one-hidden-layer MLP with 50 units. The number of parameters in the student ensemble of each method is summarized in Table 4. Note that the number of parameters of DLF is smaller than those of the other baselines because the dimension of the latent factor is 10 instead of 50.

Table 4: The number of parameters in the student ensemble.

|                 | Method               | Datasets       | Datasets   | Datasets   | Datasets   | Datasets   | Datasets   |
|-----------------|----------------------|----------------|------------|------------|------------|------------|------------|
|                 |                      | Boston housing | Concrete   | Energy     | Wine       | Power      | Kin8nm     |
| # of parameters | Teachers             | 90,000         | 65,000     | 65,000     | 80,000     | 45,000     | 65,000     |
| # of parameters | small-Ens            | 45,000         | 32,500     | 32,500     | 40,000     | 22,500     | 32,500     |
| # of parameters | Hydra                | 6,650          | 6,150      | 6,150      | 6,450      | 5,750      | 6,150      |
| # of parameters | BE                   | 7,351          | 6,601      | 6,601      | 7,051      | 6,001      | 6,001      |
| # of parameters | DLF(factor dim = 10) | 1,862          | 1,612      | 1,612      | 1,762      | 1,412      | 1,612      |

The Adam [53] is used for the optimization.

## C.1.1 Dataset

We consider the following 6 UCI datasets (Boston housing, Concrete, Energy, Wine, Power Plant, Kin8nm) [46]. We divide each dataset into a 9:1 ratio randomly for the training and test data. The experiment is repeated with 10 random split to have 10 evaluation metrics.

Table 5: Description of UCI benchmark datasets used in the experiment

| Dataset        |   size |   # of features |
|----------------|--------|-----------------|
| Boston housing |    506 |              13 |
| Concrete       |   1030 |               8 |
| Energy         |   1030 |               8 |
| Wine           |   9568 |              11 |
| Power          |    768 |               4 |
| Kin8nm         |   8192 |               8 |

## C.1.2 Evaluation metric

Let { ( x 1 , y 1 ) , · · · , ( x m test , y m test ) } be given test data.

Root Mean Square Error Root Mean Square Error (RMSE) is defined as

<!-- formula-not-decoded -->

where ˆ E ( y | x ) = ∫ y ˆ p ( y | x ) dy and ˆ p ( y | x ) is an estimated predictive distribution.

Negative Log Likelihood Negative Log Likelihood (NLL) is defined as

<!-- formula-not-decoded -->

Continuous Ranked Probability Score Continuous Ranked Probability Score (CRPS) is defined as

<!-- formula-not-decoded -->

where ̂ F j ( v ) = ∫ v -∞ ˆ p ( y | x j ) dy .

## C.2 Classification case

We consider a c -class classification problem where

<!-- formula-not-decoded -->

for a given (vector-valued) function f ( · ) = ( f 1 ( · ) , . . . , f c ( · )) . For a given teacher ensemble f ( t ) i , i = 1 , . . . , n, the teacher predictive distribution is estimated by p ( t ) ( y | x ) = ∑ n i =1 p ( y | x , f ( t ) i ) /n. The student predictive distribution for a given student ensemble is defined similarly.

## C.2.1 Dataset

In classification settings, we analyze two CIFAR datasets [54]. Each dataset contains 50,000 training and 10,000 test images of natural scenes, sized 32 × 32 pixels.

Table 6: Description of CIFAR-10 and CIFAR-100

| Dataset   |   Train size |   Test size |   # of labels |
|-----------|--------------|-------------|---------------|
| CIFAR-10  |        50000 |       10000 |            10 |
| CIFAR-100 |        50000 |       10000 |           100 |

We follow the set-up of experiments in [14]. As a teacher, we use an ensemble of four neural networks, where each model is a Wide-ResNet (WRN) [55]. Specifically, WRN-28-1 is used for CIFAR-10 and WRN-28-4 is used for CIFAR-100. And, the student model uses the WRN-16-1 network for CIFAR-10 and the WRN-28-1 network for CIFAR-100.

Training lasts 200 epochs on a single GPU using SGD with Nesterov momentum of 0.9, weight decay of 5 × 10 -4 , and batch size of 128. A one-cycle cosine annealing schedule with a five-epoch linear warm-up (from 0.001 to 0.1) is employed. The number of parameters in each ensemble is summarized in Table 7.

Table 7: Number of model parameters in each ensemble.

|                 | Method              | Datasets   | Datasets   |
|-----------------|---------------------|------------|------------|
|                 |                     | CIFAR-10   | CIFAR-100  |
| # of parameters | Teachers            | 1.48M      | 23.488M    |
| # of parameters | small-Ens           | 0.70M      | 1.50M      |
|                 | Hydra               | 0.18M      | 0.39M      |
|                 | LBE                 | 0.18M      | 0.38M      |
|                 | DLF(factor dim = 8) | 0.18M      | 0.38M      |

## C.2.2 Evaluation metric

Accuracy Accuracy (ACC) is defined as

<!-- formula-not-decoded -->

where ˆ y j = arg max y ˆ p ( y | x j ) .

Negative Log-Likelihood Negative log-likelihood (NLL) is defined as

<!-- formula-not-decoded -->

Expected Calibration Error Expected Calibration Error (ECE) [56] is defined as

<!-- formula-not-decoded -->

where { B 1 , . . . , B M } is a partition of the test data D test such that B l = { ( x , y ) ∈ D test | p (ˆ y | x ) ∈ ( ( l -1) /M,l/M ]} . In this work, M is set to be 15.

## C.3 Fine-tuning of language models

The experiments are conducted on three datasets from the GLUE [50] and SuperGLUE [51] benchmark : RTE, MRPC, and WiC. Table 8 summarizes the details of each dataset.

Table 8: Description of GLUE and SuperGLUE benchmark datasets used in the experiments

| Dataset   | Task Type                      | Description                                                                                                                                       |
|-----------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| RTE       | Recognizing Textual Entailment | Determines whether a given hypothesis can be inferred from a given premise sentence.                                                              |
| MRPC      | Paraphrase Detection           | Identifies whether two sentences are semantically equivalent. The dataset consists of sentence pairs from news sources.                           |
| WiC       | Word Sense Disam- biguation    | Determines whether a specific word used in two differ- ent contexts has the same meaning. Contextual under- standing of word senses is essential. |

Model Construction For teacher models, RoBERTa [49] is fine-tuned with the Low-Rank Adaptation (LoRA) method [48] on each task. For each dataset, four fine-tuned models are trained using different random initializations to construct teacher ensemble.

Student models are based on DistilRoBERTa [1], which is also fine-tuned with LoRA. Features are extracted through the shared backbone, and task-specific prediction heads are constructed depending on the design of each distillation method.

Training and Evaluation Settings The experimental settings, including learning rate, batch size, number of epochs, and the rank of LoRA, follow the configuration used in [57]. Model performances are evaluated using the three metrics described in Appendix C.2: Acc, NLL and ECE.

## C.4 Application to distribution shift

To apply the framework introduced in Section 5.3 to classification tasks, we consider the following model:

<!-- formula-not-decoded -->

where W is the weight matrix. For new data D new , we estimate W by minimizing the cross-entropy on new data while θ and L are fixed at ˆ θ and ˆ L estimated on training data.

For numerical study, we consider a distribution shift scenario where the conditional distribution P ( X | Y ) changes [58]. To generate synthetic data, we use CIFAR-10 and flip the labels by y ↦→ 9 -y to construct CIFAR 10-Flip dataset [44]. We first learn the body by a DNN, Hydra, and DLF on the training data of 50,000 CIFAR-10 images under the WRN-16-1 architecture for the teacher and student ensembles. Then, we train the weights of the linear head on new data while the body is fixed.

## C.5 Hardwares

All our experiments are done through Python 3.9.16 with Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz, NVIDIA TITAN Xp GPU and 128GB RAM.

## D Ablation Study

We do the following ablation studies on Boston housing data.

- We investigate how the choice of design points affects distillation performance.
- The effect of the choice of latent factor dimension affects distillation performance.
- The effect of the MMD-based initialization is investigated to assess its role in stabilizing the EM algorithm during training.
- The capacity of the student models is varied by adjusting the network width, and the effect of this change is analyzed for each baseline.
- We investigate the effect of ensemble size by varying the number of ensemble members used in the teacher and student ensembles.

Quantitative results are presented using RMSE, NLL, and CRPS, aggregated over ten independent runs.

## D.1 Choice of design points

## D.1.1 Comparison of different design points selection methods

We investigate how different strategies of selection design points influence the performance of the Gaussian distillation. The entire dataset is partitioned into three disjoint subsets: D = D train teacher ⊕ D train new ⊕D test , with a fixed ratio of 4 . 5 : 4 . 5 : 1 . The teacher ensemble is trained on D train teacher , and the student model is distilled using various forms of design points D design . To analyze the impact of the choice of design points, the four distinct strategies for selecting D design are considered:

- Design 1: Directly using the teacher training data, D design = D train teacher .
- Design 2: Using mixup samples from D train teacher . D design ⊂ mixup {D train teacher } .
- Design 3: Using a training data not used in training teacher ensemble, D design = D train new .
- Design 4: Using mixup samples from D train new . D design ⊂ mixup {D train new }

Here, mixup {D train teacher } denotes the set of samples generated by linearly combining two randomly selected samples from D train teacher . For each index j , we randomly draw two data pairs ( x j , y j ) and ( x c j , y c j ) from D train teacher and form the mixed sample

<!-- formula-not-decoded -->

The set mixup {D train teacher } consists of all mixed pairs ( x j m , y j m ) , with the number of generated design points in Designs 2 and 4 matched to the size of D train teacher . These strategies are designed to cover both scenarios where design points are reused from the teacher training data and where additional or perturbed data are incorporated. All methods are evaluated on the reserved test data D test .

Table 9: Performances of Gaussian distillation for different strategies of the design point selection

| Design Strategy                     | RMSE                                                            | NLL                                                             | CRPS                                                           |
|-------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------|
| Design 1 Design 2 Design 3 Design 4 | 3.2316 (0.0587) 3.5856 (0.1263) 2.8786 (0.0322) 3.1787 (0.2498) | 2.6317 (0.0267) 2.8003 (0.0638) 2.4816 (0.0129) 2.6122 (0.1144) | 1.7896 (0.0271) 1.9648 (0.0681) 1.6427 (0.0165) 1.792 (0.1310) |

The results summarized in Table 9 and visualized in Figure 3 illustrate the effect of different choices of design points on the performance of the Gaussian distillation.. Among the four strategies, the use

Figure 3: Box plot of evaluation metrics (RMSE, NLL, CRPS) for different strategies of the design point selection

<!-- image -->

of new training data (Design 3) for design points consistently yields the best performance across all metrics. In contrast, strategies involving mixup consistently yield inferior performances regardless of whether teacher training data (Design 2) or new data (Design 4) are used.

To sum up, the results suggest that using new training data is the best for Gaussian distillation. However, the size of the teacher training data becomes smaller, which might lead to performance degradation. In practice, we could find the optimal partition of teacher training data and new training data based on additional validation data.

## D.1.2 Comparison for different selection of the number of design points

We also investigate the effect of the number of design points on the student model. As in the previous experimental setup, the teacher ensemble is trained on D train teacher , and the student model is distilled using various sizes of design sets D design . We only consider the design type D train teacher (Design 1) and D train new (Design 3). The number of design points is controlled by the design ratio

<!-- formula-not-decoded -->

meaning that Gaussian distillation uses r ×|D design | samples for distillation.

Table 10: Performance for Gaussian distillation for the number of design points

| Design Type   | Design 1        | Design 1        | Design 1        | Design 3        | Design 3        | Design 3        |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| design_ratio  | RMSE            | NLL             | CRPS            | RMSE            | NLL             | CRPS            |
| 0.2           | 3.9829 (0.4823) | 3.0244 (0.2738) | 2.1773 (0.1704) | 3.8643 (0.3277) | 2.951 (0.1728)  | 2.145 (0.1349)  |
| 0.4           | 3.576 (0.3231)  | 2.8017 (0.1612) | 1.9818 (0.1349) | 3.0703 (0.0735) | 2.5612 (0.0316) | 1.7464 (0.0396) |
| 0.6           | 3.3996 (0.1750) | 2.711 (0.0834)  | 1.9122 (0.0573) | 3.0055 (0.0634) | 2.5337 (0.0267) | 1.7161 (0.0243) |
| 0.8           | 3.2548 (0.1407) | 2.6433 (0.0642) | 1.8444 (0.0663) | 3.0229 (0.0945) | 2.5413 (0.0395) | 1.7197 (0.0444) |
| 1             | 3.2316 (0.0587) | 2.6317 (0.0267) | 1.7896 (0.0271) | 2.8786 (0.0322) | 2.4816 (0.0129) | 1.6427 (0.0165) |

The results summarized in Table 10 and Figure 5 show that increasing the design ratio consistently improves the performance of the Gaussian distillation. That is, the larger the number of design points, the better the performance of Gaussian distillation is.

## D.2 Dimension of the latent factor

We investigate the influence of the dimension of the latent factor on the performance of Gaussian distillation. The result visualized in Figure 5 indicates that the performance of Gaussian distillation is not sensitive to the dimension of the latent factor unless the dimension is too small.

Figure 4: Evaluation metrics (RMSE, NLL, CRPS) versus the design ratio.

<!-- image -->

Figure 5: Evaluation metrics (RMSE, NLL, CRPS) versus latent factor dimension.

<!-- image -->

## D.3 MMDvs random initial

We compare the proposed MMD initialization in Section 3.2 with the random initialization. As shown in Figure 6, the MMD initialization strategy outperforms the random initialization unless the dimension of the latent factor is too small. In addition, the variations of the evaluation metrics for the MMD initialization are much smaller than those of the random initialization. That is, the MMD initialization is indispensable for the superior performance of Gaussian distillation.

## D.4 Capacity of student models

The effect of the capacity of student models is examined by varying the number of nodes in the one-layer DNN architecture. We increase the number of nodes gradually from 50 to 60, 70, 80, 90, and 100, and obtain the evaluation metrics of the distillation methods. The results in Figure 7 show that the performances of DLF and small-Ens keep improving as the number of nodes increases, while

Figure 6: Evaluation metrics (RMSE, NLL, CRPS) versus latent factor size with or without the MMD initialization

<!-- image -->

the performances of Hydra and BE are saturated. Apparently, DLF behaves similarly to small-Ens, which is interesting since small-Ens demands heavier computation and much more storage.

Figure 7: Evaluation metrics (RMSE, NLL, CRPS) versus the number of hidden nodes.

<!-- image -->

## D.5 Number of ensemble members

To evaluate the effect of ensemble size, we evaluate the performance of the KD methods by varying the number of ensemble members from 10 to 50. As we can see from Figure 8, for all methods, the performances keep improving as the ensemble size increases. Note that the number of weights in the DLF model is not proportional to the ensemble sizes (instead, it is proportional to the dimension of the latent factor), while it is proportional for the other baselines. This is an additional advantage of Gaussian distillation.

Figure 8: Evaluation metrics (RMSE, NLL, CRPS) versus the number of ensemble members.

<!-- image -->

## D.6 Application to large dataset

To further examine the scalability and robustness of our proposed model, we conducted an additional experiment on a larger and more complex dataset, Tiny ImageNet [59]. This dataset extends the original ImageNet hierarchy but contains a reduced subset of classes and image resolutions, making it a challenging yet computationally manageable benchmark for evaluating model generalization on large-scale visual domains.

Tiny ImageNet consists of 200 classes with 100,000 training images and 10,000 test images, where each image is resized to 64×64 pixels. To ensure fair evaluation and avoid overfitting, we randomly split the training set into 80% for training and 20% for validation.

We compared our method against Small Ensemble, Hydra, LBE, and Proxy-EnD 2 . The teacher network was implemented using WRN-28-4, while the student network employed WRN-16-4. Both teacher and student models were trained under an ensemble size of four, following the same distillation pipeline described in Appendix C.2.

Table 11: Results on Tiny ImageNet.

| dataset       | method                                    | Acc(%) ↑                                     | NLL ↓                                                                               | ECE(%) ↓                                                                   |
|---------------|-------------------------------------------|----------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Tiny ImageNet | Teacher Small-Ens Hydra LBE Proxy-EnD DLF | 68.74 57.11 56.80 46.08 51.64 58.34 (0.2121) | 1.2806 1.854 (0.0032) 1.7503 (0.0092) 2.24 (0.0407) 2.0195 (0.0069) 1.6737 (0.0043) | 3.22 13.51 (0.0032) 2.75 (0.092) 2.22 (0.0053) 2.77 (0.0038) 1.92 (0.1098) |
| Tiny ImageNet |                                           | (0.0027)                                     |                                                                                     |                                                                            |
| Tiny ImageNet |                                           | (0.0037)                                     |                                                                                     |                                                                            |
| Tiny ImageNet |                                           | (0.0119)                                     |                                                                                     |                                                                            |
| Tiny ImageNet | 2                                         | (0.0022)                                     |                                                                                     |                                                                            |
| Tiny ImageNet |                                           |                                              |                                                                                     |                                                                            |

Table 11 shows that our approach outperforms all other methods across all measures, even when the number of classes and samples is large, suggesting that the proposed framework generalizes well beyond small-scale datasets. In contrast, both LBE and Proxy-EnD 2 exhibit a significant drop in performance when applied to complex data, confirming that our latent factor modeling remains effective even in such challenging settings.

## E Uncertainty quantification

We conduct experiments to evaluate whether each model appropriately quantifies uncertainty by detecting out-of-distribution (OOD) data in a classification problem. We adopt predictive mutual

information, which estimates epistemic uncertainty [6], as the OOD detection score :

<!-- formula-not-decoded -->

Specifically, we denote the in-distribution (ID) dataset by { x in 1 , ..., x in m 1 } , the OOD dataset by { x out 1 , ..., x out m 2 } and y is the output of the model with corresponding predictive mutual information ̂ I [ y, θ | x in i ] for i = 1 , ..., m 1 and ̂ I [ y, θ | x out i ] for i = 1 , ..., m 2 .

For evaluation, we assign label 0 to ID data and label 1 to OOD data, meaning that lower predictive mutual information indicates ID whereas higher values indicate OOD. We then compute the AUROC between these labels and the predictive mutual information scores.

We use CIFAR-10 as the ID and SVHN, CIFAR-100, and Tiny ImageNet as OOD. All models are trained on the CIFAR-10 training set with 50,000 images, and OOD detection is evaluated on the CIFAR-10 test set with 10,000 images versus the test sets of the OOD datasets, where SVHN has 26,032 images, CIFAR-100 has 10,000 images, and Tiny ImageNet has 10,000 images. We repeat the entire procedure five times with different random seeds and report the mean of AUROC and its standard error. Table 12 show that DLF performs best on SVHN and remains competitive across CIFAR-100 and Tiny ImageNet.

Table 12: AUROC Results on in-distribution and out-of-distribution detection.

| Method      | Out-Of-Distribution Datasets   | Out-Of-Distribution Datasets   | Out-Of-Distribution Datasets   |
|-------------|--------------------------------|--------------------------------|--------------------------------|
| Method      | SVHN                           | CIFAR-100                      | Tiny ImageNet                  |
| Teachers    | 0.9403                         | 0.8604                         | 0.9400                         |
| small-Ens   | 0.9093 (0.0037)                | 0.8000 (0.0036)                | 0.7956 (0.0023)                |
| Hydra       | 0.7178 (0.0107)                | 0.6644 (0.0083)                | 0.6709 (0.0059)                |
| LBE         | 0.8329 (0.0249)                | 0.8107 (0.0130)                | 0.8152 (0.0141)                |
| Proxy-EnD 2 | 0.8427 (0.0344)                | 0.8427 (0.0183)                | 0.8456 (0.0175)                |
| DLF         | 0.9359 (0.0212)                | 0.8357 (0.0062)                | 0.8291 (0.0067)                |

## F Computational Cost

In this section, we report a comparison of the computational costs between our proposed method and baseline method during training.

## F.1 Training time

First, we evaluate the training time on the CIFAR-10 and CIFAR-100 datasets. For a fair comparison, all experiments are conducted under the same hardware environment. We follow the experimental setup described in Appendix C.2. Each experiment is repeated four times, and the average training times are reported in hours.

Table 13: Comparison of training time on CIFAR-10 and CIFAR-100 datasets.

| Method             |   CIFAR-10 Training Time |   CIFAR-100 Training Time |
|--------------------|--------------------------|---------------------------|
| Teachers small-Ens |                     10.2 |                     17.4  |
|                    |                      4.6 |                     12.5  |
| Hydra              |                      3.1 |                     10.28 |
| LBE                |                      6.5 |                     21.5  |
| Proxy-EnD 2        |                      4.6 |                      9.2  |
| DLF (Ours)         |                      4.2 |                     11.28 |

As shown in Table 13, our proposed DLF model also exhibits a relatively short training time compared to small ensemble, Hydra, and Proxy-EnD 2 . In the DLF model, µ θ and Φ θ are local networks whose

output sizes depend only on the target and latent dimensions; hence, they do not scale with the student model size. Moreover, a key advantage of DLF is that it can be effectively trained using EM algorithm. As detailed in Appendix B, the E-step has a closed-form solution, making the overall training process computationally efficient. In contrast, LBE requires more computational resources and longer training time as the model becomes larger, which will be discussed further in the following subsection.

## F.2 Floating-point operations

For a further analyze computational complexity, we compute Floating-point operations (FLOPs) per training for Hydra, Batch Ensemble and DLF. FLOPs provide a hardware-independent measure of computational cost. For example, ResNet [60] and EfficientNet [61] use FLOPs to evaluate and compare model efficiency across architectures.

Let c denote the number of classes, B the batch size, and n the number of ensemble members. The latent dimension in DLF is denoted by q . F body and F head represent the FLOPs for one forward pass through the share body (e.g., Wide-ResNet [55]) and one Hydra head, respectively. In the DLF model, F µ, Φ corresponds to the FLOPs for the small fully connected layers that produce µ θ and Φ θ . Finally, α indicates the multiplier accounting for both forward and backward passes. Then, the FLOPs per training step for Hydra, Batch Ensemble, and DLF are formulated as follows:

- Hydra:
- Batch Ensemble :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϵ (extra cost from rank-1 matrix) is usually ≤ 0 . 05 .

- DLF:

<!-- formula-not-decoded -->

Note that when the shared backbone is a Wide-ResNet, the corresponding F body is typically on the order of 10 9 FLOPs. In Batch Ensemble, the additional cost comes from the overhead factor (1 + ϵ ) applied to the full backbone. Although ϵ is usually small (typically less than 0.05), its effect is not negligible due to the large scale of F body.

In contrast, Hydra shares the backbone and only adds a small cost from the lightweight head modules. Since F head ≪ F body, the total complexity remains dominated by the shared body, even with large ensemble sizes n .

Similarly, although DLF requires additional computations such as F µ, Φ , and matrix operations such as Bq 2 , q 3 , and n ( Bcq + q 2 ) , their contributions are negligible since q is typically small (often less than 20 in practice) and F µ, Φ , Bq 2 , q 3 , n ( Bcq + q 2 ) ≪ F body. Consequently, the dominant cost in DLF also comes from the shared backbone.

## G Theoretical results

We intend to similarly investigate the convergence of the sieve maximum likelihood estimation (MLE), as discussed by [62], using the results of estimating smooth functions within the sparse deep neural network (DNN) function class proposed by [63] and sieve MLE's convergence rates by [64]. We will investigate the convergence rate in terms of decaying rates of the eigenvalues.

Notation For a natural number m , we define [ m ] = { 1 , 2 , . . . , m } . For a m × q -dimensional matrix A , we denote the spectral norm of the matrix A by ∥ A ∥ 2 and the Frobenius norm by ∥ A ∥ F , that is, ∥ A ∥ 2 = sup z ∈ R q : ∥ z ∥ 2 =1 ∥ Az ∥ 2 and ∥ A ∥ F = √ Tr ( A ⊤ A ) . For a square matrix A , let λ min ( A ) denote the smallest eigenvalue of A . For two positive sequences ( a n ) n ∈ N and ( b n ) n ∈ N , we write a n ≲ b n or b n ≳ a n , if there exists a positive constant C &gt; 0 such that a n ≤ Cb n for any n ∈ N . We write a n ≍ b n if both a n ≳ b n and a n ≲ b n hold. For a vector-valued function f = ( f 1 , . . . , f m ) ⊤ defined on a domain X , we denote the 'elementwise-maximum' supremum norm by ∥ f ∥ ∞ = max 1 ≤ j ≤ m ∥ f j ∥ ∞ = max 1 ≤ j ≤ m sup x ∈X | f j ( x ) | . The Hölder space of smoothness

β &gt; 0 with domain [0 , 1] d and radius K &gt; 0 is defined as, letting s be the smallest integer larger than or equal to β -1 ,

<!-- formula-not-decoded -->

where C s d denotes the set of s -times differentiable functions on [0 , 1] d and ∥ · ∥ H β d denotes the Hölder norm defined by

̸

<!-- formula-not-decoded -->

where N 0 = { 0 , 1 , 2 , . . . } . Let F be a given class of functions defined on X . A collection { f i : i ∈ [ N ] } is called a δ -covering set of F with respect to a certain norm ∥ · ∥ defined on X , if, for all f ∈ F , there exists f i in the collection such that ∥ f -f i ∥ ≤ δ . The cardinality of the minimal δ -covering set is called the δ -covering number of F with respect to the norm ∥ · ∥ , and is denoted by N ( δ, F , ∥ · ∥ ) . A collection { ( l i , u i ) : i ∈ [ N ] } of pairs of functions with l i ≤ u i is called a δ -bracketing set of F with respect to a norm ∥·∥ if, for all f ∈ F , there exists ( l i , u i ) in the collection such that l i ≤ f ≤ u i and ∥ l i -u i ∥ &lt; δ . The cardinality of the minimal δ -bracketing set is called the δ -bracketing number of F with respect to the norm ∥ · ∥ , and is denoted by N [] ( δ, F , ∥ · ∥ ) . For two probability density functions p 1 and p 2 , let us denote the Hellinger distance between them by h ( p 1 , p 2 ) = [ ∫ ( p 1 / 2 1 ( x ) -p 1 / 2 2 ( x )) 2 d x ] 1 / 2 .

## G.1 Problem formulation

Let ˜ f 1 , . . . , ˜ f n be n independent realizations of the Gaussian process with mean function µ ∗ ( · ) and covariance kernel Σ ∗ ( · , · ) . Suppose that we have m many d -dimensional design points D = { x ( d ) 1 , . . . , x ( d ) m } (where we omit the superscript 'design' unlike the main body of the manuscript for simplicity). We assume that without loss of generality, x ( d ) i ∈ [0 , 1] d for every j ∈ [ m ] by appropriate normalization. Then we observe f i |D = ( ˜ f i ( x ( d ) 1 ) + v i 1 , . . . , ˜ f i ( x ( d ) m ) + v im ) ⊤ , where v im s are independent Gaussian random variables with mean 0 and variance σ 2 ∗ . Note that f i |D follows the multivariate normal distribution p ∗ := N ( µ ∗|D , Σ ∗|D ) , where

<!-- formula-not-decoded -->

Let λ ∗ ,j and ψ ∗ ,j ( · ) be the j -th eigenvalues and eigenfunctions of the kernel Σ ∗ , ordered by their magnitude and ϕ ∗ ,j ( · ) = √ λ j ψ ∗ ,j ( · ) be the scaled eigenfunctions. Then, the covariance matrix can be decomposed into the q -leading part and the low-rank part

<!-- formula-not-decoded -->

where Φ ∗|D = ( ϕ ∗ ( x ( d ) 1 ) , . . . , ϕ ∗ ( x ( d ) m )) ⊤ ∈ R m × q with ϕ ∗ ( x ) = ( ϕ ∗ j ( x )) j ∈ [ q ] and Σ ∗ &gt;q |D = ( ∑ j&gt;q ϕ ∗ j ( x ( d ) u ) ϕ ∗ j ( x ( d ) v ) ) u ∈ [ m ] ,v ∈ [ m ] . For the sake of notational simplicity, we consider the case where σ 2 ∗ is known. The proof can be extended easily for the case of unknown σ 2 ∗ .

Our aim is to estimate p ∗ based on observations f 1 | m , . . . , f n | m by modeling µ ∗ and ϕ ∗ by a specially design DNN. For given µ ∈ R m and Σ ∈ R m × m , let p µ , Σ be the density of the Gaussian distribution with mean µ and covariance matrix Σ . For give mean function µ θ ( · ) and a vector of q many scaled eigenfunctions ϕ θ ( · ) parameterized by θ, we consider p µ θ |D , Σ θ |D , where

<!-- formula-not-decoded -->

with Φ θ |D = ( ϕ θ ( x ( d ) 1 ) , . . . , ϕ θ ( x ( d ) m )) ⊤ . We model ( µ θ ( · ) , ϕ θ ( · )) by a DNN with q + 1 many outputs and estimate θ by a (sieve) maximum likelihood estimator (MLE) that is defined as ˆ θ =

argmax θ ∈ Θ n ℓ n ( θ ) , where

<!-- formula-not-decoded -->

Here, the sieve Θ n depends on the architecture of DNNs. We will prove that the estimated Gaussian distribution converges to the true Gaussian distribution p ∗ in probability as n →∞ under regularity conditions while D is fixed, provided that the sieve Θ n is selected appropriately.

## G.2 Results

For the sieve Θ n , we consider a set of parameters, whose elements are in [ -1 , 1] (following [63]), of sparse DNNs with L n many layers, r n many nodes at each hidden layers, S n many nonzero elements and q n +1 output nodes. When we would like to clarify such architectural choices, we will sometimes use the notation Θ n = Θ( L n , r n , S n , q n ) . For a DNN parameter θ ∈ Θ n , we let g θ be the corresponding realized DNN function, but for a technical reason, the outputs of this function are truncated at [ -B,B ] , so that g θ is a function from R d to [ -B,B ] q +1 . We denote by G (Θ n ) = { g θ : θ ∈ Θ n } . Such sparse DNNs have been considered in many previous studies [e.g., 62, 63, 65] to investigate the asymptotic properties of DNNs.

Given the design points D = { x ( d ) 1 , . . . , x ( d ) m } , for each DNN parameter θ ∈ Θ n with the realized DNN g θ = ( g θ, 1 , . . . , g θ,q n +1 ) ⊤ , we define the m -dimensional vector µ θ |D = ( g θ, 1 ( x ( d ) u )) u ∈ [ m ] and m × m symmetric matrix Σ θ |D = Φ θ |D Φ ⊤ θ |D + σ 2 ∗ I m , where Φ θ |D = ( g θ,j +1 ( x ( d ) u )) u ∈ [ m ] ,j ∈ [ q n ] . For notational simplicity, we write p θ |D = p µ θ |D , Σ θ |D , the density of the Gaussian distribution with mean µ θ |D and covariance matrix Σ θ |D . We let the class of such Gaussian distributions P (Θ n ; D ) = { p θ |D : θ ∈ Θ n } .

Lemma G.1. Let D be an arbitrary set of m design points. There exists an absolute constant C 1 &gt; 0 such that for any δ ∈ (0 , C 1 /q n ) , the following holds

<!-- formula-not-decoded -->

Theorem G.2. Suppose that µ ∗ and ϕ ∗ ,j , j = 1 , . . . belong to H β d ( K ) . Consider the sieve MLE ˆ p = p ˆ θ over Θ n = Θ( L n , r n , S n , q n ) with L n ≍ log n and r n , S n , q n ≲ n . Define

<!-- formula-not-decoded -->

Assume that q n →∞ , ϵ ∗ n q n → 0 and n ( ϵ ∗ n ) 2 →∞ as n →∞ . Then, we have

<!-- formula-not-decoded -->

as n →∞ for some absolute constant C 2 &gt; 0 .

We can make ϵ ∗ n converge to 0 by letting q n and S n diverge with a appropriate speed provided the eigenvalues λ j , j ≥ q n converge to 0 sufficiently fast (e.g. λ j ≍ exp( -j ) ).

The upper bound of Theorem G.2 is about the estimated Gaussian distribution at the design points D design . For prediction, we need an upper bound of the estimated Gaussian distribution at a new point x . The following theorem, whose proof is given in Appendix G.4, provides an upper bound.

Theorem G.3 (Upper bound at a new input) . If ˆ µ and ˆ Φ j , j = 1 , . . . , q n are Lipschitz, the probability of

<!-- formula-not-decoded -->

for a certain positive constant C 3 and

<!-- formula-not-decoded -->

converges to 1 as n →∞ , where d 1 ( g, h ) = ∫ z | g ( z ) -h ( z ) | dz for given two probability densities on R .

## G.3 Auxiliary Lemmas

Before proving Lemma G.1, we introduce the following two lemmas.

Lemma G.4. If Σ 2 -Σ 1 is positive definite, then

<!-- formula-not-decoded -->

Proof. Define µ ∗ = ( Σ -1 1 -Σ -1 2 ) -1 ( Σ -1 1 µ 1 -Σ -1 2 µ 2 ) . Note that by assumption Σ 2 -Σ 1 is invertible, and thus we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The sum of the second and third terms is further simplified as

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

which completes the proof.

LemmaG.5. Let σ 2 be the lower bound of the minimum eigenvalues of Σ 1 and Σ 2 . If ∥ Σ 2 -Σ 1 ∥ 2 ≤ cσ 2 for a given c &gt; 0 , the following inequalities hold:

and where ζ = 3 σ 2 c.

Proof. The first inequality holds because

<!-- formula-not-decoded -->

The second inequality follows similarly.

<!-- formula-not-decoded -->

## G.4 Proofs

## Proof of Lemma G.1

Proof. Fix ϵ &gt; 0 . Let { g 1 , . . . , g N } with N = N ( ϵ, G (Θ n ) , ∥ · ∥ ∞ ) be a ϵ -covering of G (Θ n ) . For each i ∈ [ N ] , let θ i be the parameter of g i and let µ i = µ θ i |D and Σ i = Σ θ i |D . Then for any θ ∈ Θ n , letting µ = µ θ |D and Σ = Σ θ |D for simplicity, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third line follows from that ∥ Φ ∥ 2 , ∥ Φ i ∥ 2 ≤ √ mq n B . Now, let δ = ϵ max { 2 mq n B, √ mσ ∗ } /σ 2 ∗ so that ∥ µ -µ i ∥ 2 ≤ σ ∗ δ and ∥ Σ -Σ i ∥ 2 ≤ σ 2 ∗ δ . Let ζ = 3 δ . Then we will show that [ l i , u i ] is a Hellinger bracket of the density p µ , Σ ( x ) when we define

<!-- formula-not-decoded -->

Then by Lemma G.5, (1 + ζ ) Σ i -Σ and Σ -(1 + ζ ) -1 Σ i are both positive definite. So by Lemma G.4, we have for any x ∈ R m

<!-- formula-not-decoded -->

By Lemma G.5 again, we have ∥ ((1+ ζ ) Σ i -Σ ) -1 ∥ 2 ≤ ( σ 2 ∗ ( ζ -(1+ ζ ) δ )) -1 = ( σ 2 ∗ (2 -ζ ) δ ) -1 ≤ ( σ 2 ∗ δ ) -1 for any sufficiently small ϵ . Moreover, by Weyl's inequality,

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

Using the inequality log(1 + z ) ≥ z/ 2 for z ∈ [0 , 2] , we have

<!-- formula-not-decoded -->

which implies p µ , Σ ( x ) ≤ u i ( x ) . Similarly, we also have

<!-- formula-not-decoded -->

and so l i ( x ) ≤ p µ , Σ ( x ) for any x ∈ R m .

We now bound the size of the bracket. Note that

<!-- formula-not-decoded -->

and

Due to the inequality z 2 / 2 ≥ z -log(1 + z ) for any z ≥ 0 ,

<!-- formula-not-decoded -->

Moreover, by taking ϵ sufficiently small so that ζ &lt; 3 /m , we have

<!-- formula-not-decoded -->

Thus, we have h 2 ( l i , u i ) ≤ (9 / 4 mζ +4 e 2 / 3 ) ζ ≤ (3 / 4 + 4 e 2 / 3 ) ζ ≤ 26 δ . Hence, redefining constant as 26 δ → δ , we complete the proof.

## Proof of Theorem G.2

Proof. The proof follows a similar reasoning in the proof of Theorem 3 in [62], which is based on Theorem 4 in [64] with α = 0+ . We divide the proof into the following four steps.

Bounding the estimation error: Check Eq. (3.1) of [64] For the class of DNN parameters Θ n = Θ( L n , r n , S n , q n ) , by Lemma 5 in [63], we can get the following covering number bound

<!-- formula-not-decoded -->

for any δ &gt; 0 . Applying Lemma G.1, for 0 &lt; δ &lt; C 1 /q n , we have

<!-- formula-not-decoded -->

Moreover, for a positive constant ϵ such that √ 2 ϵ ≤ C 1 /q n , we have

<!-- formula-not-decoded -->

Then the above display is bounded by n 1 / 2 ϵ 2 up to an absolute constant when we take ϵ = ϵ n = √ S n (log n ) 2 /n as L n ≍ log n . Thus, Eq. (3.1) of [64] is satisfied.

Bounding the Kullback-Liebler approximation error We first note that for any θ ∈ Θ n , we have

<!-- formula-not-decoded -->

To bound the two terms in Eq. (6), we use the well-known results about the approximation ability of sparse DNNs to Hölder smooth functions [e.g., Theorem 5 of 63]. Namely, there exists θ † ∈ Θ n such that

<!-- formula-not-decoded -->

For the first term in Eq. (6), we define

<!-- formula-not-decoded -->

and establish the upper bound

<!-- formula-not-decoded -->

But by Eq. (7), ∥ Φ θ † |D -Φ ∗|D ∥ F ≲ √ q n ( S n /L n ) -β/d which converges to 0 as n → ∞ . This implies that, as ∥ Φ ∗|D ∥ 2 is bounded, ∥ Φ θ † |D -Φ ∗|D ∥ 2 F is smaller than 2 ∥ Φ ∗|D ∥ 2 ∥ Φ θ † |D -Φ ∗|D ∥ F eventually. Therefore, we have

<!-- formula-not-decoded -->

Moreover, by Eq. (7), it is immediate that ∥ µ θ † |D -µ ∗|D ∥ 2 ≲ ( S n /L n ) -2 β/d . Adopting the notation of [64], we have δ n = q n ( S n /L n ) -2 β/d + κ n ≍ q n ( S n / log n ) -2 β/d + κ n .

Bounding the Kullback-Liebler variation The last ingredient of the proof is to bound the so-called Kullback-Liebler variation defined as

<!-- formula-not-decoded -->

We will find a suitable network parameter to get a manageable upper bound of the above, which is denoted by τ n in [64]. We use Lemma G.5 for this purpose. As in the argument used in the previous step, we can find a network parameter θ † such that ∥ Σ θ † |D -Σ ∗|D ∥ 2 F ≤ C ′ δ n for some absolute constant C ′ &gt; 0 . As q n →∞ , we have ξ = λ min ( Φ θ † |D Φ ⊤ θ † |D ) &gt; 0 . We then construct the network θ ‡ satisfying µ θ ‡ |D = µ θ † |D and Φ θ ‡ |D = (1+(1+ C ′ ) δ n /ξ ) 1 / 2 Φ θ † |D . Then, by Weyl's inequality,

<!-- formula-not-decoded -->

which implies that Σ θ ‡ |D -Σ ∗|D is positive definite. Using Lemma G.5, we have

<!-- formula-not-decoded -->

where we use Weyl's inequality for the third inequality. Therefore, we have

<!-- formula-not-decoded -->

Adopting the notation of [64], we set τ n = log(1 + ( q n ) 1 / 2 δ n ) .

Combining the pieces together Let ϵ ∗ n = ϵ n ∨ √ δ n . Then by Theorem 4 of [64], there exists an absolute constant C ′′ &gt; 0 such that

<!-- formula-not-decoded -->

which tends to zero as n →∞ by the assumptions ϵ ∗ n q n → 0 and n ( ϵ ∗ n ) 2 →∞ .

## Proof of Theorem G.3

Proof. Since the total variation norm is upper bounded by the Hellinger distance, the total variation norm between ˆ p and p ∗ is also upper bounded by C 2 ϵ ∗ with probability converging to 1. In turn, by the definition of the total variation norm, sup x ∈D design d 1 (ˆ p x , p ∗ , x ) is upper bounded by C 2 ϵ ∗ with probability converging to 1. Due to the Lipschitz condition of ˆ µ, ˆ Σ as well as µ ∗ , Σ ∗ , there exists a constant L &gt; 0 such that d 1 (ˆ p x , ˆ p x ′ ) ≤ L ∥ x -x ′ ∥ and d 1 ( p ∗ , x , p ∗ , x ′ ) ≤ L ∥ x -x ′ ∥ for any x and x ′ in R d . Finally, we have

<!-- formula-not-decoded -->

with probability converging to 1. The proof is complete by letting C 3 = C 2 and C 4 = 2 L.
## Quantifying Uncertainty in the Presence of Distribution Shifts

## Yuli Slavutsky

Department of Statistics Columbia University New York, NY 10027, USA yuli.slavutsky@columbia.edu

## David M. Blei

Departments of Statistics, Computer Science

Columbia University New York, NY 10027, USA

david.blei@columbia.edu

## Abstract

Neural networks make accurate predictions but often fail to provide reliable uncertainty estimates, especially under covariate distribution shifts between training and testing. To address this problem, we propose a Bayesian framework for uncertainty estimation that explicitly accounts for covariate shifts. While conventional approaches rely on fixed priors, the key idea of our method is an adaptive prior, conditioned on both training and new covariates. This prior naturally increases uncertainty for inputs that lie far from the training distribution in regions where predictive performance is likely to degrade. To efficiently approximate the resulting posterior predictive distribution, we employ amortized variational inference. Finally, we construct synthetic environments by drawing small bootstrap samples from the training data, simulating a range of plausible covariate shift using only the original dataset. We evaluate our method on both synthetic and real-world data. It yields substantially improved uncertainty estimates under distribution shifts.

## 1 Introduction

Neural networks are powerful predictive models, capable of capturing complex relationships from data [Raghu et al., 2017]. Despite this capability, they struggle to provide reliable measures of predictive uncertainty. This issue is particularly relevant when the distribution of covariates shifts between training and test data. Such shifts frequently occur in real-world settings, including high-stakes applications like medicine, where inaccurate uncertainty estimates can lead to harmful outcomes [Topol, 2019, Rajkomar et al., 2019].

Consider a dataset { ( x i , y i ) } n i =1 and a new test point x ∗ . In a classical Bayesian neural network [Neal, 2012], the posterior predictive distribution is

<!-- formula-not-decoded -->

where p ( y | x ∗ , θ ) comes from the neural network with weights θ and p ( θ | x 1: n , y 1: n ) is the posterior over those weights. This expression reveals that, in the classical model, predictive uncertainty arises entirely from uncertainty about model parameters θ . But intuitively, if the new covariate vector x ∗ lies far from the training covariates x 1: n , then we should become more uncertain about our prediction.

To capture this intuition, we propose a Bayesian approach that better reflects uncertainty due to covariate shifts. The central idea is to adapt the prior distribution to explicitly depend on covariates, i.e., replacing the classical prior p ( θ ) with p ( θ | x 1: n , x ∗ ) . This leads to a posterior predictive distribution of the form

<!-- formula-not-decoded -->

Figure 1: (a) In training one covariate is fixed; the data lies on a one-dimensional subspace. All predictors intersecting the fixed axis at the same point are equivalent. (b) At test time, variation along the second dimension reveals that some predictors may fit better the new data, prompting a prior shift. (c) Possible labeling where only the solid line separates the test data.

<!-- image -->

As we will discuss, conditioning the prior on the test covariates x ∗ allows the posterior to adjust its uncertainty in accordance with the proximity of x ∗ to the training distribution. Thus, the model captures the impact of covariate shift on predictive performance, and delivers increased uncertainty for inputs that lie far from the training data.

The intuition behind this prior is that predictions become more uncertain at covariates far from the training data because the learned relationship may no longer hold. Consider a logistic regression with two covariates, where one covariate varies substantially and the other hardly varies. The classical posterior of the coefficients can assign both low and high value to the less variable covariate since in the training data it is almost indistinguishable from the intercept. But if the test data includes a previously unseen value of this covariate, our predictive uncertainty should increase. The test data point, even without its response, indicates that the coefficient could differ significantly from what the training data alone suggests. For an illustration see Figure 1.

Implementing this idea presents three key challenges:

The first challenge is to specify the prior p ( θ | x 1: n , x ∗ ) . We propose an energy-based prior that spreads its mass on the plausible values of the weights, given the covariates.

The second challenge is to compute Equation (2), which involves the posterior distribution p ( θ | x ∗ , x 1: n , y 1: n ) . Unlike a classical posterior distribution, this posterior is located in the context of a prediction about x ∗ , which comes into the adaptive prior. To approximate it, we use the idea of amortized variational inference [Kingma et al., 2015, Margossian and Blei, 2024]. We learn a family of approximate posteriors that take test covariates x ∗ as input and produce an approximate posterior tailored to its prediction.

A final challenge is that fitting our amortized variational family requires both training data from the training distribution and new data from a covariate-shifted distribution. In practice, however, we often only have one training set, without access to data drawn from a shifted distribution. We use small bootstrap samples to form synthetic environments [Slavutsky and Benjamini, 2024] and prove that they contain covariate-shifted distributions able to approximate unseen shifts. We adapt the variational objective to match all of these environments.

Together, these ideas form Variational Inference under Distribution Shif (VIDS). On both real and synthetic data, we show that VIDS outperforms existing methods in terms of predictive accuracy, calibration of uncertainty, and robustness under covariate shifts. VIDS provides accurate estimates of posterior predictive uncertainty in the face of distribution shift.

Related work. Forming predictions under covariate shift is important to many applications. Examples from the medical domain include imaging data from different hospitals [Zech et al., 2018, AlBadawy et al., 2018, Perone et al., 2019, Castro et al., 2020], and failure to provide reliable predictions when applied to different cell types. In image classification, cross-dataset generalization remains challenging [Torralba and Efros, 2011], including cases where shifts are introduced by variations in cameras [Beery et al., 2018], or by temporal and geographic differences. Other examples include person re-identification, where the training data often includes biases with respect to to gender [Grother et al., 2019, Klare et al., 2012], age [Best-Rowden and Jain, 2017, Michalski et al., 2018, Srinivas et al., 2019], and race [Raji and Buolamwini, 2019, Wang et al., 2019]. All of these settings that can benefit benefit from the use of VIDS.

Approaches for uncertainty estimation in neural networks include ensemble-based methods [Lakshminarayanan et al., 2017, Valdenegro-Toro, 2019], and methods based on noise addition [Dusenberry et al., 2020, Maddox et al., 2019, Wen et al., 2020]. A prominent line of work focuses on Bayesian neural networks [Tishby et al., 1989, Denker and LeCun, 1990], which offer a principled framework for uncertainty quantification and have been widely adopted for this purpose [Ovadia et al., 2019]. These approaches (e.g., [Ha et al., 2016, Yoon et al., 2018]) typically place a single prior distribution

over the model parameters that is shared across all inputs. In contrast, our method defines a prior that adapts to individual covariates.

Several techniques have been given Bayesian interpretations, including regularization methods such as dropout [Kingma et al., 2015, Gal and Ghahramani, 2016], stochastic gradient-based approximations [Welling and Teh, 2011, Dubey et al., 2016, Li et al., 2016], and variational inference methods that approximate the posterior distribution [Graves, 2011, Neal, 2012, Blundell et al., 2015, Louizos and Welling, 2016, Malinin and Gales, 2018].

Aprevalent strategy for handling uncertainty under distribution shift involves distance-aware methods, which estimate uncertainty based on the distance between new inputs and the training data. These include approaches that rely on training data density estimation [Sensoy et al., 2018], often implemented via kernel density methods [Ramalho and Miranda, 2020, Van Amersfoort et al., 2020] or Gaussian Processes (GPs) [Williams and Rasmussen, 2006]. More recent examples include Spectralnormalized Neural Gaussian Processes (SNGP) [Liu et al., 2020], which apply spectral normalization to stabilize network weights, and Deterministic Uncertainty Estimation (DUE) [Van Amersfoort et al., 2021], which integrates a GP with a deep feature extractor trained to preserve distance information in the representation space.

Most relevant to our work are methods that combine Bayesian and distance-based approaches, such as Park and Blei [2024], which incorporate an energy-based criterion into the training objective to increase predictive uncertainty for inputs that are unlikely under the training distribution.

The key idea behind the aforementioned distance-based approaches is to detect shifts by measuring how far test inputs lie from the training inputs, typically using fixed metrics in the input space. But these measures can be misleading because not all shifts affect predictive performance equally. Consider a univariate logistic regression. Suppose there are two types of shifts: (i) training data concentrate near the decision boundary, and (ii) training data appear only at extreme covariate values. At test time, covariates cover the entire range. Under shift (i), predictive performance remains strong because the critical region near the boundary was covered during training. Under shift (ii), performance deteriorates because the model never saw data near the decision boundary.

The important factor is not simply the difference in covariates, but how shifts influence predictive accuracy. Unlike predefined distance measures, the adaptive prior in VIDS essentially learns from data how covariate shifts affect predictive performance. It uses its adaptive prior to directly model the change in predictive uncertainty that is induced by newly observed covariates.

## 2 Predictive uncertainty under distribution shifts

We address uncertainty estimation under covariate shift, where the distribution of test-time inputs x ∗ may differ from that of the training data x 1: N . To account for such shifts, we extend the classical Bayesian framework by treating covariates as random variables and modeling their dependence on the model parameters θ . This leads to a formulation in which the prior over θ is conditioned on the newly observed covariate x ∗ , resulting in a predictive posterior that explicitly reflects this dependence. Consequently, the predictive uncertainty, defined through this posterior, can adapt to reflect greater uncertainty for covariates x ∗ that are unlikely under the training distribution, thereby capturing their potential impact on predictive performance. We begin by revisiting the classical Bayesian model.

## 2.1 Background on the classical Bayesian model

In the classical Bayesian framework the parameters θ are treated as a random variable drawn from a prior distribution θ ∼ p ( θ ) , while the covariates x 1: N are considered fixed. Under the standard conditional independence assumption, each outcome y i is independent of all other pairs ( x j , y j ) given θ and x i . The posterior distribution over θ is

<!-- formula-not-decoded -->

New test points x ∗ are likewise treated as fixed. Thus, the introduction of a new test input x ∗ does not alter the posterior, and consequently does not affect the predictive uncertainty. Again, see Equation 1.

## 2.2 A Bayesian approach to covariate shift

Our goal in this work is to explicitly model distributional differences between the training covariates x 1 , . . . , x N ∼ p x and a new covariate x ∗ ∼ p x ∗ . To effectively capture how such shifts impact predictive performance, the posterior predictive at x ∗ should adapt to the observed change in the covariate distribution. To this end, we propose a model in which the model parameters θ depend on both the training and test covariates, x 1: N and x ∗ . This dependence is illustrated in the probabilistic graphical model in Figure 2, which indicates the structure of the posterior predictive for x ∗ .

As in the classical framework, our model assumes that y ∗ is conditionally independent of all other variables given x ∗ and θ . However, by allowing θ to depend on the covariates, the prior p ( θ | x 1: N , x ∗ ) can now adjust the plausibility of parameter values based both on training and test inputs, and thus capture distribution shifts . As a result, the posterior p ( θ | x 1: N , y 1: N , x ∗ ) is explicitly dependent on x ∗ through the covariate-dependent prior.

Let f θ denote the predictive model parametrized by θ . The predictive uncertainty for x ∗ in our model is defined through its predictive posterior in Equation 2, which is our primary quantity of interest. Our goal is to approximate this predictive uncertainty. Since the likelihood p ( y ∗ | x ∗ , θ ) can be evaluated directly by the predictive model f θ ( x ∗ ) , the central challenge lies in accurately estimating the covariate-dependent posterior p ( θ | x ∗ , x 1: N , y 1: N ) .

In what follows, we first define a concrete adaptive prior p ( θ | x ∗ , x 1: N ) that conditions on both the observed training covariates and the test-time covariate. We then develop a variational inference scheme to approximate the posterior under this adaptive prior.

Figure 2: Graphical model. Thick red arrows denote additional dependencies introduced by our model. Observed variables shown in gray.

<!-- image -->

Finally, since test-time covariates x ∗ from a shifted distribution are typically unavailable during training, we approximate this setting by constructing synthetic environments designed to simulate diverse covariate distributions by subsampling the training data. Using these environments, we design an algorithm that approximates a posterior capable of anticipating predictive degradation under a range of potential test-time shifts.

## 2.3 The adaptive prior

Our prior aims to capture the plausibility of θ given both the training inputs x 1: N and a new test covariate x ∗ . We propose an adaptive prior conditioned on both x 1: N and x ∗ , defined by the following energy function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Z ( θ ) := ∫ exp( E ( θ ; x 1: N , x ∗ )) dθ is the normalizing factor. 1

This formulation allows for a smooth adaptation to test-time shifts. When only a small number of test covariates x ∗ are introduced, or if the inputs are similar to the training data, the prior remains close to a distribution conditioned on the training covariates alone. However, as more test inputs are observed, especially if they differ substantially from the training distribution, the prior adjusts more significantly to reflect the new covariate distribution.

We illustrate this adaptivity in the simple example described in §1. We consider a logistic model for y ∈ { 0 , 1 } , where covariates x consist of two features: one of them remains constant in all training examples, but in new test examples, both features vary. Figure 3 shows how the prior distribution changes when the shifted examples are introduced. See Appendix A for additional details and an illustration that lightly shifted examples lead to minimal adaptations.

1 This definition requires integrability of exp( E ( θ ; x 1: N , x ∗ )) , and thus we assume that lim ∥ θ ∥→∞ E ( θ ; x 1: N , x ∗ ) = -∞ with at least linear decay.

Figure 3: Changes in the prior due to the introduction of test covariates drawn from a shifted distribution x ∗ ∼ N ( 1 2 , 1) , where both features vary.

<!-- image -->

## 2.4 VIDS: Variational inference under distribution shifts

Given this prior, our goal is to estimate the posterior p ( θ | x 1: N , y 1: N , x ∗ ) , which enables approximate computation of predictive uncertainty for the test input x ∗ according to Equation (2). For this, we fit a variational distribution q ϕ ( θ ; x ∗ ) ≈ p ( θ | x 1: N , y 1: N , x ∗ ) . This variational distribution is an amortized posterior approximation , parametrized as a function of x ∗ , allowing approximation of the posterior across multiple test-time covariates x ∗ .

We model the amortized posterior as a multivariate Gaussian distribution, parametrized by its mean µ and diagonal covariance matrix Σ . Let Q d denote the family of d -dimensional Gaussian distributions with diagonal covariance. Thus, we seek q ϕ ( θ ; x ∗ ) with ϕ = ( µ, Σ) , that minimizes the KullbackLeibler divergence to the true posterior:

<!-- formula-not-decoded -->

Specifically, we optimize the evidence lower bound (ELBO) on the log-likelihood [Blei et al., 2017, Kingma and Welling, 2014, Rezende and Mohamed, 2015]

<!-- formula-not-decoded -->

which is equivalent to solving Equation (6).

We train a neural network h γ with weights γ , to output ϕ from x ∗ , and optimize γ rather than ϕ directly. Given the training set D = { ( x i , y i ) } N i =1 and M test covariates { x ∗ j } j M =1 , we define the following objective to fit our amortized posterior

<!-- formula-not-decoded -->

Note that evaluation of this objective requires estimation of the prior

<!-- formula-not-decoded -->

which involves integration over the outcome space Y . For discrete Y the integration is simply summation, and thus can be easily computed. If Y is continuous, we apply Monte Carlo integration: we sample r target values uniformly from an integration range [ y min , y max ] , and for each sample compute the log-likelihood under a unit-variance Gaussian centered at each predicted value.

## 2.5 The variational family and stochastic optimization of the variational objective

To fully describe VIDS, it remains to define the amortized variational family and the optimization procedure to maximize Equation 8. So far, the variational posterior has been amortized with respect to x ∗ . Now, we will amortize also with respect to the training set, in order to better optimize across multiple environments, as described in the next section.

Figure 4: Optimization mechanism for a single test example in a single synthetic environment.

<!-- image -->

The variational family is q ϕ ( θ ; x ∗ , x 1: n ) where the parameters ϕ come from an inference network. We decompose the inference network into an embedding network g : R d → R k , parametrized by ξ , and a prediction layer parametrized by θ . We assume that g has been pre-trained to maximize the likelihood p ( y | x, θ ) = f θ ( g ξ ( x )) on the training dataset. We focus on the prediction layer θ .

Given the training data x 1: n and test covariate x ∗ , we define the inference network as follows:

1. We compute the embeddings of the training set, ˆ g ( x 1 ) , . . . , ˆ g ( x n ) , and aggregate them into a single summary statistic (e.g., the mean) 2 denoted by g ( x 1 , . . . , x n ) .
2. We compute the embedding of the test covariate, ˆ g ( x ∗ ) .
3. The aggregated training embedding and the test embedding are concatenated and passed through a network h γ : R 2 k → Φ , which outputs the parameters ϕ = ( µ, Σ) of the variational distribution q ϕ ( θ | x 1: n , x ∗ ) .

In practice, we optimize a single-sample Monte Carlo estimate of the expectations in Equation 7. To enable differentiable sampling from q ϕ , we use the reparametrization trick [Kingma and Welling, 2014], where we sample θ ∼ q ϕ by first sampling ϵ from a standard Gaussian and then calculating θ = µ +Σ ϵ . The full procedure is summarized in Algorithm 1 and illustrated in Figure 4.

## 2.6 Uncertainty estimation

With training completed using Algorithm 1, we finally turn to estimating predictive uncertainty for a new test input x ∗ , i.e., from the posterior predictive distribution of Equation (2).

Given the learned parameters ˆ γ , we evaluate the representation of the test point g ˆ ξ ( x ∗ ) and use it with the combined representation of the training data to compute the variational posterior parameters via ˆ ϕ = h ˆ γ ( g ˆ ξ ( x 1: N ) . Wethen approximate the posterior predictive by drawing samples θ (1) , . . . , θ ( S ) ∼ q ˆ ϕ and calculating the corresponding predictions f θ ( s ) ( g ˆ ξ ( x ∗ )) .

## 3 Posterior estimation across multiple distribution shifts

So far, we have developed a method to approximate the posterior given training data D = { ( x i , y i ) } N i =1 and a set of test covariates { x ∗ j } j M =1 , drawn from a shifted distribution. However, a key challenge in real-world settings is that such test covariates are typically unavailable in advance. To address this limitation, similarly to [Slavutsky and Benjamini, 2024], we generate synthetic environments via subsampling from the training data.

Specifically, we construct L environments by sampling L pairs of datasets

<!-- formula-not-decoded -->

where each dataset pair is constructed by sampling data pairs ( x i , y i ) uniformly at random with replacement from the original training set D . We refer to each resulting pair of datasets as a synthetic

2 The summary embedding g ( x 1 , . . . , x n ) is a permutation-invariant aggregation of learned element-wise representations, and thus follows the Deep Sets setting [Zaheer et al., 2017].

## Algorithm 1 Variational posterior

- 1: Input: Training data D , covariates { x ∗ j } j M =1 , pre-trained embedding g ˆ ξ , predictor f ( · ; θ ) , iterations K , learning rate η , initialization γ (0) .
- 2: Compute train embeddings g ˆ ξ ( x 1 ) , . . . g ˆ ξ ( x N ) and test embeddings g ˆ ξ ( x ∗ 1 ) , . . . , g ˆ ξ ( x ∗ M )
- 3: Aggregate train embeddings to obtain g ˆ ξ ( x 1 , . . . , x N )
- 4: for 1 ≤ k ≤ K do
- 5: for 1 ≤ j ≤ M do
- 6: Compute ϕ ( k ) j = h ( g ˆ ξ ( x 1: N ) , g ˆ ξ ( x ∗ j ); γ ( k -1) )
- 7: Sample ϵ ( k ) j and compute θ ( k ) j = µ ( k ) j +Σ ( k ) j · ϵ ( k ) j for ( µ ( k ) j , Σ ( k ) j ) = ϕ ( k ) j
- 8: Compute

<!-- formula-not-decoded -->

- 9: end for
- 10: Compute L ( k ) = ∑ m j =1 L D ( ϕ ( k ) 1: m )
- 11: Update the parameters of h by performing a gradient ascent step:
- 12: end for Return: ˆ γ := γ ( K )

<!-- formula-not-decoded -->

environment, denoted e ( ℓ ) = { D ( ℓ ) tr , D ( ℓ ) te } . Each such synthetic test set D ( ℓ ) te is likely to exhibit a different empirical distribution, thereby simulating a potential covariate shift.

The core idea of our approach can viewed as an inverse bootstrap sampling: while bootstrap sampling relies on subsamples of the original dataset being highly likely to resemble the population distribution, we instead focus on the low-probability cases where the subsample deviates from the original dataset's distribution. These deviations simulate potential distribution shifts that may arise at test time.

In the following proposition we show that drawing enough subsamples guarantees that with high probability, at least one of them will be close to the true unknown test distribution.

Proposition (informal) 3.1. Let p and p ∗ be binned distributions of the training data and the unobserved test set, respectively. Assume that supp( p ∗ ) ⊆ supp( p ) . Then, for any ϵ &gt; 0 and 0 ≤ α &lt; 1 , there exist m and L such that, with probability at least 1 -α , the empirical distribution of at least one of L randomly drawn subsamples of size m from the training data satisfies ∥ ˆ p ( ℓ ) -p ∗ ∥ 1 ≤ ϵ .

Appendix B gives a formal proposition, the proof, an application to the case where supp p ∗ ̸⊆ supp p , and analysis of relationships between the number of required synthetic environments L to ϵ and α .

While the proposition guarantees that, given enough sampled environments, at least one will approximate the true (unknown) test-time shift, it remains unclear which one that is. To address this, we aim to ensure that the learned posterior performs well across all synthetic environments. This motivates the use of environment-level penalties, inspired by the out-of-distribution (OOD) generalization literature [Arjovsky et al., 2019, Wald et al., 2021, Krueger et al., 2021].

Thus, we introduce the following cross-environment objective:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, we set the penalty to the variance across the environments, as proposed by Krueger et al. [2021]. In Algorithm 2 we summarize the complete procedure of variational posterior estimation with synthetic environments (see Appendix C).

<!-- formula-not-decoded -->

<!-- image -->

0.0

0.4

1.0

0.0

0.4

0.2

0.4

0.0

0.2

0.4

0.8

Figure 5: Simulation results. Red crosses represent training data, and gray dots test data. Black lines depict predictions, gray shaded area spanning ± 1 standard deviation. Top: Heteroskedastic linear regression for a = 0 . 5 ; Bottom: Binary classification with missing data for t = 0 . 3 . VIDS is the only one to capture correct variance structures and thus achieves the best results.

## 4 Experiments

We evaluate VIDS on both synthetic and real-world datasets, across classification and regression tasks. We compare the uncertainty estimates produced by VIDS (ours) with previous distance aware methods: SNGP [Liu et al., 2020], DUE [Van Amersfoort et al., 2021], and distance uncertainty layers (DUL) [Park and Blei, 2024] (see §1 for more details). In all experiments, the same neural network architecture is used as the prediction model. Hyper-parameters of our and competing methods were optimized via grid search to maximize average performance (accuracy for classification, RMSE for regression) on a single sample of J = 50 synthetic test environments of size m = 10 , which was discarded from the analysis. For the corresponding values, and additional implementation details see Appendix E.

## 4.1 Synthetic data

We begin by examining two synthetic examples exhibiting covariate shifts. Each experiment uses N = M = 500 training and test points, and VIDS constructs synthetic environments of size m = 20 . All models use a fully connected neural network with one hidden layer of width d = 8 .

4.1.1 Regression: heteroscedastic linear model We sample x ∼ U [0 , a ] , and x ∗ ∼ U [0 , b ] for a &lt; b . Outcomes for train and test follow y = βx + ϵ ( x ) where ϵ ( x ) ∼ N (0 , x 10 ) . Results for a = 0 . 5 , b = 1 and β = 1 are shown in figure 5, and results for additional settings in Appendix D.

Results: VIDS consistently achieves the lowest RMSE and is the only one to capture the correct variance structure. DUE exhibits low variance but underestimates uncertainty at higher x values. SNGP overfits, yielding poor test performance and excessive variance. DUL provides the best posterior mean among competitors, but overestimates uncertainty due to incorrect variance modeling.

4.1.2 Classification: logistic regression with missing data For both the training and test sets, we sample x ∼ Beta ( 1 / 2 , 1 / 2 ) and compute ρ ( x ) = σ ( -5 + 10 x ) where σ denotes the standard sigmoid function. Outcomes are then generated according to y | x ∼ Ber ( ρ ( x )) . However, in the training data, we exclude middle values within the range ( t, 1 -t ) . Results for t = 0 . 3 are shown in Figure 5.

Results: VIDS achieves the highest accuracy and lowest calibration error. In addition, as can be seen in Figure 5, VIDS exhibits lowest variance while correctly modeling uncertainty: higher in the unseen middle and lower at the edges. DUE and, to a greater extent, DUL predict well but misrepresent uncertainty, increasing variance at the edges, with DUL overestimating variance overall. SNGP estimates higher variance in the middle but consistently under-predicts.

Figure 6: Classification accuracy over 10 repetitions. Celeb-A titles formatted as Target - Shift Attr. VIDS achieves highest accuracy in all experiments.

<!-- image -->

Figure 7: RMSE results for three experiments on three UCI regression datasets over 10 repetitions. VIDS achieves lowest or comparable RMSE in all experiments.

<!-- image -->

## 4.2 Real data

## 4.2.1 Classification

For the classification experiments we use a simple convolutional neural-network with two convolution blocks (see details in Appendix E).

Corrupted CIFAR-10 We evaluate model performance under corruption-induced distribution shifts using the CIFAR-10-C dataset [Hendrycks and Dietterich, 2019]. We perform experiments on three corruption types: defocus blur, glass blur, and contrast. We construct the training set of 5000 images, 90% clean from the original CIFAR-10 dataset [Krizhevsky et al., 2009] and 10% corrupted images from CIFAR-10-C, while the test set is constructed with 5000 images, 90% corrupted and 10% clean.

Celeb-A For each experiment, we choose one annotated attribute as the target, and another attribute A to induce a distribution shift in the CelebA dataset [Liu et al., 2015]. The training set contains 500 images with 90% having A = 1 and 10% with A = 0 ; the test set reverses this ratio: 90% images with A = 0 and 10% with A = 1 . We run three such experiments with the following shift-target pairs: (i) Pale Skin → Blond Hair, (ii) Heavy Makeup → Male, and (iii) Gray Hair → Blond Hair.

Results: Figure 6 shows that in all 6 classification experiments VIDS achieves better accuracy.

## 4.2.2 Regression

We conduct experiments on three UCI regression datasets-Boston, Concrete, and Wine. For each, we designate a prediction target and apply K-Means clustering ( K = 2 ) on all numeric covariates. To simulate a non-trivial covariate shift, we use the cluster with the higher average within-cluster Euclidean distance primarily for training (90% of training data), and the lower-distance cluster primarily for testing (90% of test data). In all these experiments we use a simple linear regression (one-layer network) as the base model. For additional details see Appendix E.

Results: Figure 7 shows that VIDS achieves the lowest average RMSE across all three datasets and consistently low variance, while DUE under-performs on the Concrete and Boston datasets and DUL considerably under-performs on the Wine dataset.

## 5 Conclusion

We introduced VIDS, a Bayesian method for quantifying uncertainty under potential distribution shifts. VIDS leverages two central ideas: incorporating dependencies between network parameters and the test covariates, and balancing performance across synthetic environments to simulate covariate shifts. VIDS consistently outperforms existing methods, particularly when assumptions like homoscedasticity or continuity are violated.

## 6 Acknowledgments

We are grateful to members of the Blei Lab for fruitful discussions and feedback. In particular, to Sebastian Salazar, Eli N. Weinstein, Andrew Jesson, Nicolas Beltran, and Sweta Karlekar. We thank Andrew Jesson for verifying the DUE implementation. This work is supported by NSF IIS2127869, NSF DMS-2311108, ONR N000142412243, the Simons Foundation and DoD OUSD (R&amp;E) under Cooperative Agreement PHY-2229929 (The NSF AI Institute for Artificial and Natural Intelligence). YS is supported by a Founder's Postdoctoral Fellowship, Department of Statistics, Columbia University.

## References

- [1] Ehab A AlBadawy, Ashirbani Saha, and Maciej A Mazurowski. Deep learning for segmentation of brain tumors: Impact of cross-institutional training and testing. Medical Physics , 45(3): 1150-1158, 2018.
- [2] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv Preprint arXiv:1907.02893 , 2019.
- [3] Sara Beery, Grant Van Horn, and Pietro Perona. Recognition in terra incognita. In Proceedings of the European Conference on Computer Vision (ECCV) , pages 456-473, 2018.
- [4] Lacey Best-Rowden and Anil K Jain. Longitudinal study of automatic face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence , 40(1):148-162, 2017.
- [5] David M. Blei, Alp Kucukelbir, and Jon McAuliffe. Variational inference: A review for statisticians. Journal of American Statistical Association , 112(518):859-877, 2017.
- [6] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In International Conference on Machine Learning , pages 1613-1622. PMLR, 2015.
- [7] Daniel C Castro, Ian Walker, and Ben Glocker. Causality matters in medical imaging. Nature Communications , 11(1):3673, 2020.
- [8] John Denker and Yann LeCun. Transforming neural-net output levels to probability distributions. Advances in Neural Information Processing Systems , 3, 1990.
- [9] Kumar Avinava Dubey, Sashank J Reddi, Sinead A Williamson, Barnabas Poczos, Alexander J Smola, and Eric P Xing. Variance reduction in stochastic gradient langevin dynamics. Advances in Neural Information Processing Systems , 29, 2016.
- [10] Michael Dusenberry, Ghassen Jerfel, Yeming Wen, Yian Ma, Jasper Snoek, Katherine Heller, Balaji Lakshminarayanan, and Dustin Tran. Efficient and scalable bayesian neural nets with rank-1 factors. In International Conference on Machine Learning , pages 2782-2792. PMLR, 2020.
- [11] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning , pages 1050-1059. PMLR, 2016.
- [12] Alex Graves. Practical variational inference for neural networks. Advances in Neural Information Processing Systems , 24, 2011.
- [13] Patrick Grother, Mei Ngan, Kayee Hanaoka, et al. Ongoing face recognition vendor test (frvt) part 3: Demographic effects. Nat. Inst. Stand. Technol., Gaithersburg, MA, USA, Rep. NISTIR , 8280, 2019.
- [14] David Ha, Andrew Dai, and Quoc V Le. Hypernetworks. arXiv preprint arXiv:1609.09106 , 2016.
- [15] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. International Conference on Learning Representations (ICLR) , 2019.
- [16] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. International Conference on Learning Representations , 2014.
- [17] Durk P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. Advances in Neural Information Processing Systems , 28, 2015.
- [18] Brendan F Klare, Mark J Burge, Joshua C Klontz, Richard W Vorder Bruegge, and Anil K Jain. Face recognition performance: Role of demographic information. IEEE Transactions on Information Forensics and Security , 7(6):1789-1801, 2012.

- [19] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [20] David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. Out-of-distribution generalization via risk extrapolation (rex). In International Conference on Machine Learning , pages 5815-5826. PMLR, 2021.
- [21] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems , 30, 2017.
- [22] Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin. Preconditioned stochastic gradient langevin dynamics for deep neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 30, 2016.
- [23] Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax Weiss, and Balaji Lakshminarayanan. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. Advances in Neural Information Processing Systems , 33:7498-7512, 2020.
- [24] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV) , December 2015.
- [25] Christos Louizos and Max Welling. Structured and efficient variational deep learning with matrix gaussian posteriors. In International Conference on Machine Learning , pages 1708-1716. PMLR, 2016.
- [26] Wesley J Maddox, Pavel Izmailov, Timur Garipov, Dmitry P Vetrov, and Andrew Gordon Wilson. A simple baseline for bayesian uncertainty in deep learning. Advances in Neural Information Processing Systems , 32, 2019.
- [27] Andrey Malinin and Mark Gales. Predictive uncertainty estimation via prior networks. Advances in Neural Information Processing Systems , 31, 2018.
- [28] Charles Margossian and David M. Blei. Amortized variational inference: When and why. In Uncertainty in Artificial Intelligence , 2024.
- [29] Dana Michalski, Sau Yee Yiu, and Chris Malec. The impact of age and threshold variation on facial recognition algorithm performance using images of children. In 2018 International Conference on Biometrics (ICB) , pages 217-224. IEEE, 2018.
- [30] Radford M Neal. Bayesian Learning for Neural Networks , volume 118. Springer Science &amp; Business Media, 2012.
- [31] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. Advances in Neural Information Processing Systems , 32, 2019.
- [32] Yookoon Park and David Blei. Density uncertainty layers for reliable uncertainty estimation. In International Conference on Artificial Intelligence and Statistics , pages 163-171. PMLR, 2024.
- [33] Christian S Perone, Pedro Ballester, Rodrigo C Barros, and Julien Cohen-Adad. Unsupervised domain adaptation for medical imaging segmentation with self-ensembling. NeuroImage , 194: 1-11, 2019.
- [34] Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the expressive power of deep neural networks. In International Conference on Machine Learning , pages 2847-2854. PMLR, 2017.
- [35] Inioluwa Deborah Raji and Joy Buolamwini. Actionable auditing: Investigating the impact of publicly naming biased performance results of commercial ai products. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society , pages 429-435, 2019.

- [36] Alvin Rajkomar, Jeffrey Dean, and Isaac Kohane. Machine learning in medicine. New England Journal of Medicine , 380(14):1347-1358, 2019.
- [37] Tiago Ramalho and Miguel Miranda. Density estimation in representation space to predict model uncertainty. In Engineering Dependable and Secure Machine Learning Systems: Third International Workshop, EDSMLS 2020, New York City, NY, USA, February 7, 2020, Revised Selected Papers 3 , pages 84-96. Springer, 2020.
- [38] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning , pages 1530-1538. PMLR, 2015.
- [39] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. Advances in Neural Information Processing Systems , 31, 2018.
- [40] Yuli Slavutsky and Yuval Benjamini. Class distribution shifts in zero-shot learning: Learning robust representations. Advances in Neural Information Processing Systems , 2024.
- [41] Nisha Srinivas, Karl Ricanek, Dana Michalski, David S Bolme, and Michael King. Face recognition algorithm bias: Performance differences on images of children and adults. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops , pages 0-0, 2019.
- [42] Joy A. Thomas Thomas M. Cover. In Elements of Information Theory , pages 347-355. John Wiley &amp; Sons, Ltd, 2005.
- [43] Tishby, Levin, and Solla. Consistent inference of probabilities in layered networks: Predictions and generalizations. In International 1989 Joint Conference on Neural Networks , pages 403-409. IEEE, 1989.
- [44] Eric J Topol. High-performance medicine: The convergence of human and artificial intelligence. Nature Medicine , 25(1):44-56, 2019.
- [45] Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In CVPR 2011 , pages 1521-1528. IEEE, 2011.
- [46] Matias Valdenegro-Toro. Deep sub-ensembles for fast uncertainty estimation in image classification. Bayesian Deep Learning Workshop at Neural Information Processing Systems , 2019.
- [47] Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. Uncertainty estimation using a single deep deterministic neural network. In International Conference on Machine Learning , pages 9690-9700. PMLR, 2020.
- [48] Joost Van Amersfoort, Lewis Smith, Andrew Jesson, Oscar Key, and Yarin Gal. On feature collapse and deep kernel learning for single forward pass uncertainty. Bayesian Deep Learning Workshop at Neural Information Processing Systems , 2021.
- [49] Yoav Wald, Amir Feder, Daniel Greenfeld, and Uri Shalit. On calibration and out-of-domain generalization. Advances in Neural Information Processing Systems , 34:2215-2227, 2021.
- [50] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, and Yaohai Huang. Racial faces in the wild: Reducing racial bias by information maximization adaptation network. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 692-702, 2019.
- [51] Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning (ICML-11) , pages 681-688. Citeseer, 2011.
- [52] Yeming Wen, Dustin Tran, and Jimmy Ba. Batchensemble: An alternative approach to efficient ensemble and lifelong learning. International Conference on Learning Representations , 2020.
- [53] Christopher K Williams and Carl Edward Rasmussen. Gaussian processes for machine learning, vol. 2, no. 3. Cambridge, MA, USA: Massachusetts Institute of Technology Publishing , 2006.

- [54] Jaesik Yoon, Taesup Kim, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin Ahn. Bayesian model-agnostic meta-learning. Advances in neural information processing systems , 31, 2018.
- [55] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. Advances in neural information processing systems , 30, 2017.
- [56] John R Zech, Marcus A Badgeley, Manway Liu, Anthony B Costa, Joseph J Titano, and Eric Karl Oermann. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine , 15(11):e1002683, 2018.

## A Example for the adaptive prior

We revisit the simple example from §1, considering a binary prediction setting where y ∈ { 0 , 1 } and the covariates x ∈ R 2 consist of two features, x (1) and x (2) . In the training data x (2) remains constant across all examples. However, at test time, new subtypes may be encountered where both features exhibit variability.

Assume the following logistic model

<!-- formula-not-decoded -->

Due to the exchangeability between β 0 and β 2 in the training data, all combinations with a fixed value of β 0 + β 2 are equally plausible when observing only the training data. However, this symmetry is broken when test data from new subtypes is observed.

As a concrete example, assume that for all datapoints (train and test) the first feature is distributed as x (1) ∼ N (1 , 1) . In the training data, x (2) i = 1 2 for all 1 ≤ i ≤ N ; however, for new test examples x ∗ , the second feature is drawn from N ( 1 2 , 1 ) . Let θ ′ := ( β 2 , β 1 , β 0 ) ⊺ . While for any of the first N datapoints, p ( y | x i , θ ) = p ( y | x i , θ ′ ) , this no longer holds for x ∗ , thus leading to changes in the prior distribution upon arrival of a new test example x ∗ .

Figure 3 illustrates how the prior distribution, which for y ∈ { 0 , 1 } corresponds to

<!-- formula-not-decoded -->

adapts when new test examples are observed.

For the training data, any combination of β 0 and β 2 with a fixed sum β 0 + β 2 results in the same probability. However, as new examples with varying values of x (2) are introduced, the prior distribution begins to depend on how the weight is distributed between β 0 and β 2 . Consequently, the equivalence regions concentrate around specific combinations of β 0 and β 2 .

If, instead, the second feature of the test examples x ∗ , is drawn from a distribution closely aligned with the training data, N ( 1 2 , 1 10 2 ) , i.e., a distribution closely similar to the training data, the resulting changes to the prior are minimal. This is illustrated in Figure 8.

Figure 8: Changes in the prior due to introduction of test covariates drawn from a distribution similar to the training data, x ∗ ∼ N ( 1 2 , 1 10 2 ) . In the top row, β 0 is fixed at -1 , while in the bottom row, β 1 is fixed at 1 .

<!-- image -->

## B Theoretical analysis of synthetic-environment sampling

The statement of Proposition 3.1 holds for any pair of finite distributions p, p ∗ ∈ ∆ k , not just those arising from our specific application-namely, the empirical distribution of the training set and the binned test distribution. In what follows, we formally state and prove this more general result, and then return to its application in our setting.

Proposition B.1. Let B 1 , . . . , B k be a partition of X . Denote the binned empirical distribution of the train set x 1: N as p ∈ ∆ k , where p ( i ) = 1 N ∑ N j =1 1 { x j ∈ B i }∀ 1 ≤ i ≤ k . Similarly, define the binned test distribution induced by the partition as p ∗ ( i ) = ∫ B i p x ∗ ( x ) dx . Assume that supp p ∗ ⊆ supp p . Then for any ϵ &gt; 0 and α ∈ (0 , 1) , there exist integers m,L ∈ N , such that if L independent samples of size m are drawn according to p , then with probability at least 1 -α , at least one sample will induce a binned empirical distribution ˆ p ( ℓ ) satisfying ∥ ˆ p ( ℓ ) -p ∗ ∥ 1 ≤ ϵ .

Proof. Let m := ⌈ 2( k -1) ϵ ⌉ and define q ∈ ∆ k as follows

<!-- formula-not-decoded -->

For 1 ≤ i ≤ k -1 , we have | q ( i ) -p ∗ ( i ) | ≤ 1 m , and | q ( k ) -p ∗ ( k ) | ≤ k -1 m . Hence, ∥ q -p ∗ ∥ 1 ≤ 2( k -1) m ≤ ϵ .

Define

<!-- formula-not-decoded -->

the set of all samples of size m drawn according to p , such that their empirical distribution p z = q . This is the type-set of q . Denote the probability of drawing a given sample z = z 1: m by p m ( z ) := ∏ m j =1 p ( z j ) . Then, by the method of types (see, e.g., Theorem 11.1.4 in [42]), the probability of drawing a sample whose empirical distribution equals q is

<!-- formula-not-decoded -->

Consequently, the probability of drawing L samples of size m according to p , such that the empirical distribution of at least one of them is q satisfies

<!-- formula-not-decoded -->

Taking L ≥ log α log(1 -ξ ) completes the proof.

Remark B.2. Using the inequality log(1 -ξ ) ≥ -ξ 1 -ξ , we have

<!-- formula-not-decoded -->

revealing the relationship of L to the tolerance ϵ and to α .

Application to our setting We now return to the setting in which p ∈ ∆ k denotes the binned empirical distribution of the train set x 1: N , induced by the partition of X . Specifically,

<!-- formula-not-decoded -->

for all 1 ≤ i ≤ k . Similarly, let p ∗ ∈ ∆ k denote the binned test distribution induced by the partition, given by

<!-- formula-not-decoded -->

If supp p ∗ ⊆ supp p , then the conditions of Proposition 3.1 are satisfied, and thus drawing enough subsamples from x 1: N guarantees that, with high probability, in the binned space at least one of them will be close to the test distribution.

However, if there exists a bin B i , such that x j / ∈ B i for all 1 ≤ j ≤ N but ∫ B i p x ∗ ( x ) dx &gt; 0 , then if ∥ p -p ∗ ∥ 1 = ϵ ′ &lt; ϵ , by discarding any bins B i such that B i ∈ supp p ∗ and B i / ∈ supp p , and re-normalizing the probabilities on the remaining bins to sum to 1, we obtain a reduced distribution for which we can require a distance to p ∗ that is at most ϵ -ϵ ′ . Consequently, Proposition 3.1 can be applied directly to this re-normalized distribution, guaranteeing with high probability that at least one subsample will be close to the test distribution within the reduced tolerance ϵ -ϵ ′ .

## C Multi-environment algorithm

Algorithm 2 Variational posterior with synthetic environments

- 1: Input: Data D , no. synthetic environments L , train and test environment sizes n and m , pretrained embedding g ˆ ξ , predictor f ( · ; θ ) , iterations K , learning rate η , initialization γ (0) .
- 2: for 1 ≤ k ≤ K do
- 3: for 1 ≤ ℓ ≤ L do
- 4: Sample synthetic datasets: D ( ℓ ) tr = { ( x ( ℓ ) i , y ( ℓ ) i ) } n i =1 , D ( ℓ ) te = { ( x ∗ ( ℓ ) j , y ∗ ( ℓ ) j ) } j m =1
- 5: Compute and test embeddings: g ˆ ξ ( x ( ℓ ) 1 ) , . . . g ˆ ξ ( x ( ℓ ) n ) and g ˆ ξ ( x ∗ ( ℓ ) 1 ) , . . . , g ˆ ξ ( x ∗ ( ℓ ) m )
- 6: Aggregate train embeddings to obtain g ˆ ξ ( x ( ℓ ) 1 , . . . , x ( ℓ ) n )
- 7: for 1 ≤ j ≤ m do
- 8: Compute ϕ ( k,ℓ ) j = h ( g ˆ ξ ( x ( ℓ ) 1: n ) , g ˆ ξ ( x ∗ ( ℓ ) j ); γ ( k -1) )
- 9: Sample ϵ ( k,ℓ ) j and compute θ ( k,ℓ ) j = µ ( k,ℓ ) j +Σ ( k,ℓ ) j · ϵ ( k,ℓ ) j for ( µ ( k,ℓ ) j , Σ ( k,ℓ ) j ) = ϕ ( k,ℓ ) j 10: Compute:

<!-- formula-not-decoded -->

- 11: end for
- 12: end for
- 13: Compute L ( k ) = 1 ∑ L ℓ =1 ∑ m j =1 L D ( ϕ ( k,ℓ ) 1: m )
- 14: Update the parameters of h by gradient ascent: γ ( k ) ← γ ( k -1) + η ∇ γ L ( k )
- 15: end for

## D Additional experimental results

## D.1 Synthetic data

Full results for the synthetic experiments presented in Figure 5 are provided in Table 1.

Table 1: Results for synthetic data linear regression and binary classification

| Experiment   | Metric       | VIDS (ours)                 | DUE                         | SNGP                        | DUL                         |
|--------------|--------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
| Linear       | RMSE         | 0.068 (0.002)               | 0.110 (0.008)               | 0.488 (0.004)               | 0.081 (0.010)               |
| Logistic     | Accuracy ACE | 0.900 (0.009) 0.011 (0.003) | 0.880 (0.001) 0.034 (0.007) | 0.844 (0.027) 0.099 (0.038) | 0.900 (0.008) 0.018 (0.000) |

In Figure 9 we provide results for the heteroscedastic regression experiment, for additional values of the parameter a , controlling how much the test data is shifted with respect to the observed training data, and thus the difficulty of generalization on providing accurate uncertainty measures.

<!-- image -->

0.4

0.2

0.4

0.2

80

Figure 9: Simulation results for heteroskedastic regression for a = 0 . 25 (top) and a = 0 . 75 (bottom). Red crosses represent training data, while gray dots indicate test data. The black line depicts the predictions, with the gray shaded area spanning ± 1 standard deviation. A single repetition is depicted; RMSE values are averaged over 10 repetitions.

## E Implementation Details

## E.1 General implementation details

Code The code to reproduce our results is attached to the submission and upon acceptance a link to a permanent repository will be included in the main text.

All the code in this work was implemented in Python 3.11. We used Numpy 2.0, TensorFlow 2.13 and TensorFlow Addons 0.21 packages. The UCI datasets were loaded through sklearn 1.6. CIFAR-C dataset was obtained from Zenodo 3 . CIFAR-10 and Celeb-A datasets were loaded through torchvision 0.21. All figures were generated using Matplotlib 3.10.

For DUE and DUL implementation was adapted from the source code of the original papers 45 . For SNGP implementation was adapted from the implementation provided in the source code of DUE.

We ran all synthetic data and UCI experiments on 2 CPUs. Each repetition of these experiments lasted less than 7 minutes. For real data classification experiments (on CIFAR-10 and Celeb-A datasets) we used a single A100 cloud GPU. Each repetition lasted less than 18 minutes.

Hyper-parameters Our setting deals with an unknown covariate shift. Thus, hyperparameters for all methods were chosen via a grid-search in a single experiment repetition, (excluded from the analysis). Some of the hyperparameters are method specific. Our variational method uses environment related parameters: number of environments J , size of each test environment m , and size of each train environment n . Below we refer to the KL penalty λ as 'penalty', and to the variance penalty as τ . The DUE method employs inducing points to approximate the Gaussian process component. Both DUE and SNGP use GP features, random Fourier features used for approximating the kernel, and scaling of the input features. In DUL the number of steps corresponds to epochs. The hyperparameters of all methods are specified in the corresponding tables below.

Normalizing factor For continuous response variables, our implementation approximates the log of the normalizing factor z using the log of the mean of exponential log-likelihoods: log v ≈ log( 1 n ∼ i exp ( v i )) where v i are the integrated log likelihoods.

3 https://zenodo.org/record/2535967/files/CIFAR-10-C.tar

4 https://github.com/y0ast/DUE

5 https://github.com/yookoon/density\_uncertainty\_layers

Table 2: Hyperparameters for heteroskedastic linear and logistic regression.

|                   | Linear      | Linear   | Linear   | Linear   | Logistic    | Logistic   | Logistic   | Logistic   |
|-------------------|-------------|----------|----------|----------|-------------|------------|------------|------------|
| Parameter         | VIDS (ours) | DUE      | SNGP     | DUL      | VIDS (ours) | DUE        | SNGP       | DUL        |
| J                 | 30          | -        | -        | -        | 30          | -          | -          | -          |
| m                 | 20          | -        | -        | -        | 20          | -          | -          | -          |
| n                 | 500         | -        | -        | -        | 500         | -          | -          | -          |
| τ                 | 0.001       | -        | -        | -        | 0.001       | -          | -          | -          |
| Penalty           | 0.005       | -        | 1        | 1        | 0.005       | -          | 1          | 1          |
| Batch size        | 520         | 100      | 64       | 50       | 520         | 100        | 32         | 50         |
| Steps             | 30          | 1500     | 7000     | 500      | 30          | 1500       | 1000       | 300        |
| Learning rate     | 10 - 2      | 10 - 2   | 10 - 3   | 10 - 3   | 10 - 3      | 10 - 2     | 10 - 3     | 10 - 2     |
| n_inducing_points | -           | 20       | -        | -        | -           | 20         | -          | -          |
| GP features       | -           | -        | 128      | -        | -           | -          | 3          | -          |
| Random features   | -           | -        | 1024     | -        | -           | -          | 128        | -          |
| Feature scale     | -           | -        | 2        | -        | -           | -          | 2          | -          |

## E.2 Implementation details for synthetic data experiments

The data-related parameters of the synthetic experiments are described in the main text. The hyperparameters used for the heteroskedastic linear regression and logistic regression with missing data are detailed in Table 2.

For VIDS we specify h γ as a fully-connected neural network with 6 layers of sizes 64 d, 32 d, 16 d, 8 d, 4 d, 2 d and ReLU activation between the layers, for d = 8 .

## E.3 Implementation details for real data classification experiments

For the classification experiments (both for CIFAR and Celeb-A datasets) and all methods, we specify the base model as a convolutional neural network with two convolutional blocks, each with a 3×3 convolution with 32 filters, followed by a ReLU activation and a 2×2 max-pooling. These are followed by a fully connected layer with 64 units, ReLU activation and a final fully connected layer of dimension d = 16 .

We performed a grid search for hyperparameters for each method on a single repetition of the experiment on an excluded setting. We chose the "Pixelate" corruption for the search on CIFAR, and the Target - Shift Attribute pair of Male - Blurry for Celeb-A. The resulting hyper-parameters are reported in Table 3.

CIFAR For VIDS, we specify h γ as a fully connected neural network with 4 layers of sizes layers of sizes 32 d · 10 , 16 d · 10 , 4 d · 10 , 2 d · 10 and ReLU activations between them, for d = 16 .

Celeb-A For VIDS, we specify h γ as a fully connected neural network with 5 layers of sizes 32 d, 16 d, 8 d, 4 d, 2 d and ReLU activations between them, for d = 16 .

## E.4 Implementation details for real data regression experiments

We use the standard target variables from the UCI datasets: MEDV for Boston, Compressive Strength for Concrete, and Quality for Wine.

To evaluate model performance under distribution shifts, we split each dataset into two groups by applying the K-Means algorithm with K = 2 on all numerical columns, excluding the target variable. We calculate the average Euclidean within-cluster distance for each cluster as the mean distance from each point to its centroid. The cluster with the larger average within-cluster distance is designated as the majority in training.

We then sample data from both clusters uniformly at random to form the training and test sets. The training set contains 90% of the high-variance cluster, while the test set consists of 90% of the

Table 3: Hyperparameters for classification experiments.

|                   | CIFAR       | CIFAR   | CIFAR   | CIFAR   | Celeb-A     | Celeb-A   | Celeb-A   | Celeb-A   |
|-------------------|-------------|---------|---------|---------|-------------|-----------|-----------|-----------|
| Parameter         | VIDS (ours) | DUE     | SNGP    | DUL     | VIDS (ours) | DUE       | SNGP      | DUL       |
| J                 | 10          | -       | -       | -       | 10          | -         | -         | -         |
| m                 | 20          | -       | -       | -       | 20          | -         | -         | -         |
| n                 | 5000        | -       | -       | -       | 1000        | -         | -         | -         |
| τ                 | 0.001       | -       | -       | -       | 0.001       | -         | -         | -         |
| Penalty           | 0.001       | -       | 1       | 1       | 0.005       | -         | 1         | 1         |
| Batch size        | 5020        | 100     | 100     | 50      | 1020        | 100       | 100       | 50        |
| Steps             | 25          | 25,000  | 100,000 | 500     | 50          | 1500      | 10,000    | 500       |
| Learning rate     | 10 - 4      | 10 - 4  | 10 - 4  | 10 - 3  | 10 - 3      | 10 - 2    | 10 - 3    | 10 - 3    |
| n_inducing_points | -           | 100     | -       | -       | -           | 20        | -         | -         |
| GP features       | -           | -       | 64      | -       | -           | -         | 16        | -         |
| Random features   | -           | -       | 512     | -       | -           | -         | 64        | -         |
| Feature scale     | -           | -       | 2       | -       | -           | -         | 2         | -         |

low-variance cluster. Both the training and test features and target variables are standardized using the median and standard deviation of the training set.

For all experiments we specified the base model as a fully connected neural network with 4 layers of sizes 8 d, 4 d, 2 d, 4 d, 2 d and ReLU activations between them, for d = 32 .

The hyper-parameters for the experiments are detailed in Table 4.

Table 4: Hyperparameters for UCI regression experiments.

| Parameter         | VIDS (ours)   | DUE    | SNGP   | DUL    |
|-------------------|---------------|--------|--------|--------|
| J                 | 10            | -      | -      | -      |
| m                 | 50            | -      | -      | -      |
| n                 | 1000          | -      | -      | -      |
| τ                 | 0.001         | -      | -      | -      |
| Penalty           | 0.01          | -      | 1      | 1      |
| Batch size        | 520           | 100    | 64     | 50     |
| Steps             | 150           | 1000   | 7000   | 500    |
| Learning rate     | 10 - 3        | 10 - 2 | 10 - 3 | 10 - 3 |
| n_inducing_points | -             | 20     | -      | -      |
| GP features       | -             | -      | 16     | -      |
| Random features   | -             | -      | 16     | -      |
| Feature scale     | -             | -      | 2      | -      |

## E.5 Running times

Our method differs from others in that it pre-trains a representation and performs variational inference only on the prediction layer. As a result, for smaller models (e.g., linear regression), our approach can be more computationally expensive due to per-environment optimization. However, for larger models, our method has an advantage since competing methods must optimize many more parameters. In the following table, we specify the hardware and execution times for all methods across our experiments.

Table 5: Runtime comparison across methods.

| Dataset                          | Hardware       | Unit    |   Vids |    DUE |   SNGP |    DUL |
|----------------------------------|----------------|---------|--------|--------|--------|--------|
| Linear regression                | 2 CPUs         | seconds | 67.16  | 39.9   | 54.8   | 16.4   |
| Logistic regression              | 2 CPUs         | seconds | 61.25  | 40.3   | 16.5   |  6.29  |
| Concrete                         | 2 CPUs         | minutes |  3.237 |  2.07  |  0.7   |  0.4   |
| Boston                           | 2 CPUs         | minutes |  3.112 |  2.017 |  0.717 |  0.267 |
| Wine                             | 2 CPUs         | minutes |  2.523 |  1.583 |  1.567 |  0.895 |
| CIFAR-C (max across corruptions) | A100 Cloud GPU | minutes |  2.516 |  7.447 |  3.527 |  4.04  |
| Celeb-A (max across tasks)       | A100 Cloud GPU | minutes |  2.42  |  4.417 |  4.22  |  3.318 |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims and contributions of the paper are listed in the final paragraph of the Introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations and possible future extensions are discussed at the end of the Conclusion section.

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

Justification: Modeling assumptions are explicitly stated in Section 2.2, and proofs in Appendix B.

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

Justification: The algorithms are described in Algorithm 1 and 2. Details for all empirical results are provided in Appendix E.

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

Justification: Source code to reproduce in experimental results is provided with the submission, and will be made public upon acceptance.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an accep answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All details are specified in Appendix E.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Mean performance and standard deviation across replicates are reported in tables throughout the study. All relevant plots include error bars.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in s or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in s or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or s in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Specified in Appendix E.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is not related to any particular application that might have societal impacts.

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

Justification: Not applicable for the current study.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Datasets and previous approaches discussed in the paper have all been properly attributed.

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

Answer:[Yes]

Justification: We provide documented source code.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The study does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The study does not involve human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
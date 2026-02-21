## Uncertainty Quantification for Physics-Informed Neural Networks with Extended Fiducial Inference

## Frank Shih

Department of Epidemiology and Biostatistics Memorial Sloan Kettering Cancer Center New York, NY 10017

shihf@mskcc.org

## Faming Liang

Department of Statistics Purdue University West Lafayette, IN 47907 fmliang@purdue.edu

## Abstract

Uncertainty quantification (UQ) in scientific machine learning is increasingly critical as neural networks are widely adopted to tackle complex problems across diverse scientific disciplines. For physics-informed neural networks (PINNs), a prominent model in scientific machine learning, uncertainty is typically quantified using Bayesian or dropout methods. However, both approaches suffer from a fundamental limitation: the prior distribution or dropout rate required to construct honest confidence sets cannot be determined without additional information. In this paper, we propose a novel method within the framework of extended fiducial inference (EFI) to provide rigorous uncertainty quantification for PINNs. The proposed method leverages a narrow-neck hyper-network to learn the parameters of the PINN and quantify their uncertainty based on imputed random errors in the observations. This approach overcomes the limitations of Bayesian and dropout methods, enabling the construction of honest confidence sets based solely on observed data. This advancement represents a significant breakthrough for PINNs, greatly enhancing their reliability, interpretability, and applicability to real-world scientific and engineering challenges. Moreover, it establishes a new theoretical framework for EFI, extending its application to large-scale models, eliminating the need for sparse hyper-networks, and significantly improving the automaticity and robustness of statistical inference.

Keywords : Adaptive Stochastic Gradient MCMC, Deep Learning, Black-Scholes Model, Partial Differential Equation, Porous-FKPP Model

## 1 Introduction

Physics-informed neural networks (PINNs) (Raissi et al., 2019) are a class of scientific machine learning models that integrate physical principles directly into the training process of the DNN models. They achieve this by incorporating terms derived from ordinary differential equations (ODEs) or partial differential equations (PDEs) into the DNN's loss function. PINNs take spatial-temporal coordinates as input and produce functions that approximate the solutions to the differential equations. Because they embed physics-based constraints, PINNs can typically address problems that are described by few data while ensuring adherence to the given physical laws (Zou et al., 2025; Cuomo

## Zhenghao Jiang

Department of Statistics Purdue University West Lafayette, IN 47907 jiang976@purdue.edu

et al., 2022), paving the ways for the use of neural networks in out-of-distribution (OOD) prediction (see e.g., Yao et al. (2024)).

From a Bayesian viewpoint, PINNs can be seen as using 'informative priors' drawn from mathematical models, thus requiring less data for system identification (Zhang et al., 2019). However, this naturally raises a challenge in uncertainty quantification: How should one balance the prior information and the observed data to ensure faithful inference about the underlying physical system? As emphasized in Zou et al. (2025), uncertainty quantification is becoming increasingly critical as neural networks are widely employed to tackle complex scientific problems, particularly in high-stake application scenarios. However, addressing this issue within the Bayesian framework is difficult, as the prior information and data information are essentially exchangeable under Bayesian formulations. The use of 'informative priors' can conflict with the spirit of 'posterior consistency' (see e.g., Ghosal et al. (2000)), a foundational principle in Bayesian inference. This conflict creates a dilemma: using an informative prior risks overshadowing the data, while using a weak prior may lead to violations of the underlying physical law. In practice, this makes it difficult, if not impossible, to properly calibrate the resulting credible intervals without additional information.

In addition to Bayesian methods (Yang et al., 2021), dropout (Srivastava et al., 2014) has also been employed to quantify the uncertainty of PINNs, as demonstrated by Zhang et al. (2019). Dropout is primarily used as a regularization technique to reduce overfitting during DNN training. Gal and Ghahramani (Gal and Ghahramani, 2016) showed that dropout training in DNNs can be interpreted as approximate Bayesian inference in deep Gaussian processes, allowing model uncertainty to be estimated from dropout-trained DNN models. However, this approach shares a limitation similar to Bayesian methods: The dropout rate, which directly influences the magnitude of the estimated model uncertainty, cannot be determined without additional information, making it challenging to ensure consistent and reliable uncertainty quantification.

This paper introduces an EFI (Liang et al., 2025) approach to quantify the uncertainty in PINNs. EFI provides a rigorous theoretical framework that addresses the limitations of Bayesian and dropout methods by formulating the problem as a structural equation-solving task. In this framework, each observation is expressed as a data-generating equation, with the random errors (contained in observations) and DNN parameters treated as unknowns (see Section 2). EFI jointly imputes the random errors and estimates the inverse function that maps the observations and imputed random errors to DNN parameters. Consequently, the imputed random errors are propagated to the DNN parameters through the estimated inverse function, allowing the model uncertainty to be accurately quantified without the need for additional information. Our contribution in this paper is two-fold:

- A new theoretical framework for EFI: We develop a new theoretical framework for EFI that significantly enhances the automaticity of statistical inference. Originally, EFI was developed in Liang et al. (2025) under a Bayesian framework, where a sparse prior is imposeStochasticd on the hyper-network (referred to as the w -network in Section 2) to ensure consistent estimation of the inverse function. However, due to the limitations of existing sparse deep learning theory (Sun et al., 2022), this Bayesian approach could only be applied to models with dimensions fixed or increasing at a very low rate with the sample size. In this paper, we propose learning the inverse function using a narrow-neck w -network, which ensures consistent estimation of the inverse function without relying on the use of sparse priors. Moreover, it enables EFI to work for large-scale models, such as PINNs, where the number of model parameters can far exceed the sample size. By avoiding the need for Bayesian sparse priors, our framework allows EFI to fulfil the original goal of fiducial inference: Inferring the uncertainty of model parameters based solely on observations .
- Open-source software for uncertainty quantification in PINNs: We provide an open source software package for uncertainty quantification in PINNs, which can be easily extended to conventional DNNs and other high-dimensional statistical models.

Related Work The proposed method belongs to the class of imprecise probabilistic techniques (Augustin et al., 2014). However, compared to other methods in the class, such as credal Bayesian deep learning (Caprio et al., 2023a), imprecise Bayesian neural networks (Caprio et al., 2023b), and other Bayesian neural network-based methods, the key advantage of EFI is that it avoids the need for prior specification while ensuring accurate calibration of predictions.

Another related line of work concerns uncertainty quantification for machine learning models. Beyond the Bayesian and dropout methods noted above, this line includes conformal prediction

(Vovk et al., 2005), deep ensembles (Lakshminarayanan et al., 2016), and stochastic deep learning (Sun and Liang, 2022; Liang et al., 2022), among others. These methods primarily target predictive uncertainty and are often ineffective or inapplicable for quantifying uncertainty in model parameters. In contrast, EFI addresses both predictive uncertainty and parameter uncertainty, and further provides theoretical guarantees for the validity of the resulting prediction and confidence intervals. The ability to accurately quantify uncertainty in deep neural network parameters is a distinctive advantage of EFI.

## 2 A Brief Review of EFI

While fiducial inference was widely considered as a big blunder by R.A. Fisher, the goal he initially set -inferring the uncertainty of model parameters based solely on observations - has been continually pursued by many statisticians, see e.g. structural inference (Fraser, 1966, 1968), generalized fiducial inference (Hannig, 2009; Hannig et al., 2016; Murph et al., 2022), and inferential models (Martin and Liu, 2013, 2015; Martin, 2023). To this end, Liang et al. (2025) developed the EFI method based on the fundamental concept of structural inference.

Consider a regression model: Y = f ( X , Z, θ ) , where Y ∈ R and X ∈ R d represent the response and explanatory variables, respectively; θ ∈ R p represents the vector of parameters; and Z ∈ R represents a scaled random error following a known distribution π 0 ( · ) . Suppose that a random sample of size n , denoted by { ( y 1 , x 1 ) , ( y 2 , x 2 ) , . . . , ( y n , x n ) } , has been collected from the model. In structural inference, the observations can be expressed in data-generating equations as follows:

<!-- formula-not-decoded -->

This system of equations consists of n + p unknowns, namely, { θ , z 1 , z 2 , . . . , z n } , while there are only n equations. Therefore, the values of θ cannot be uniquely determined by the data-generating equations, and this lack of uniqueness of unknowns introduces uncertainty in θ .

Let Z n = { z 1 , z 2 , . . . , z n } denote the unobservable random errors contained in the data, which are also called latent variables in EFI. Let G ( · ) denote an inverse function/mapping for θ , i.e.,

<!-- formula-not-decoded -->

It is worth noting that the inverse function is generally non-unique. For example, it can be constructed by solving any p equations in (1) for θ . As noted by Liang et al. (2025), this non-uniqueness of inverse function mirrors the flexibility of frequentist methods, where different estimators of θ can be constructed to achieve desired properties such as efficiency, unbiasedness, and robustness.

Since the inverse function G ( · ) is generally unknown, Liang et al. (2025) proposed to approximate it using a sparse DNN, see Figure A1 in the Appendix for illustration. They also introduced an adaptive stochastic gradient Langevin dynamics (SGLD) algorithm, which facilitates the simultaneous training of the sparse DNN and simulation of the latent variables Z n . See Algorithm 1 for the pseudo-code. Refer to Section A1 of the Appendix for the mathematical formulation of the method. Briefly, they let w n denote the weights of w -network and define an energy function U n ( Y n , X n , Z n , w n ) . subsequently, they define a posterior distribution π ϵ ( w n | X n , Y n , Z n ) for w n and a predictive distribution π ϵ ( Z n | X n , Y n , w n ) for Z n , where ϵ can be read as a temperature. They treat Z n as missing data and learn w n through solving the following equation:

<!-- formula-not-decoded -->

using Algorithm 1.

Under mild conditions for the adaptive SGLD algorithm, it can be shown that

<!-- formula-not-decoded -->

where w ∗ n denotes a solution to equation (3) and p → denotes convergence in probability, and that

<!-- formula-not-decoded -->

in 2-Wasserstein distance, where d ⇝ denotes weak convergence. To study the limit of (6) as ϵ decays to 0, i.e., p ∗ n ( z | Y n , X n , w ∗ n ) = lim ϵ ↓ 0 π ϵ ( Z n | X n , Y n , w ∗ n ) , where p ∗ n ( z | Y n , X n , w ∗ n ) is referred

## Algorithm 1: Adaptive SGLD for EFI computation

(i) (Initialization) Initialize w (0) n , Z (0) n , M (the number of fiducial samples to collect), and K (burn-in iterations).

for k=1,2,. . . , K + M do

(ii) (Latent variable imputation) Given w ( k ) n , simulate Z ( k +1) n using the SGLD algorithm:

<!-- formula-not-decoded -->

where υ k +1 is the learning rate, and e ( k +1) ∼ N (0 , I d z ) . (iii) (Parameter updating) Draw a minibatch { ( y 1 , x 1 , z ( k ) 1 ) , . . . , ( y m , x m , z ( k ) m ) } and update the network weights by the SGD algorithm:

<!-- formula-not-decoded -->

where γ k +1 is the step size, and log π ϵ ( y i | x i , z ( k ) i , w ( k ) n ) can be appropriately defined according to (A6).

(iv) (Fiducial sample collection) If k +1 &gt; K , calculate ˆ θ ( k +1) i = ˆ g ( y i , x i , z ( k +1) i , w ( k +1) n ) for each i ∈ { 1 , 2 , . . . , n } and average them to get a fiducial ¯ θ -sample as calculated in (A5). end

(v) (Statistical Inference) Conducting statistical inference for the model based on the collected fiducial samples.

to as the extended fiducial density (EFD) of Z n , Liang et al. (2025) impose specific conditions on the structure of the w -network, including that the w -network is sparse and that the output layer width (i.e., the dimension of θ ) is either fixed or grows very slowly with the sample size n . Under these assumptions, they prove the consistency of w ∗ n based on the sparse deep learning theory developed in Sun et al. (2022). This consistency further implies that

<!-- formula-not-decoded -->

serves as a consistent estimator for the inverse function/mapping θ = G ( Y n , X n , Z n ) , where ˆ g ( · ) denotes the learned neural network function. Refer to Appendix A2 for the expression of p ∗ n ( z | Y n , X n , w ∗ n ) .

Let Z n = { z ∈ R n : U n ( Y n , X n , Z n , w ∗ n ) = 0 } denote the zero-energy set. Under some regularity conditions on the energy function, Liang et al. (2025) proved that Z n is invariant to the choice of G ( · ) . Let Θ := { θ ∈ R p : θ = G ∗ ( Y n , X n , z ) , z ∈ Z n } denote the parameter space of the target model, which represents the set of all possible values of θ that G ∗ ( · ) takes when z runs over Z n . Then, for any function b ( θ ) of interest, its EFD µ ∗ n ( ·| Y n , X n ) associated with G ∗ ( · ) is given by

<!-- formula-not-decoded -->

for any measurable set B ⊂ Θ , where Z n ( B ) = { z ∈ Z n : b ( G ∗ ( Y n , X n , z )) ∈ B } , and P ∗ n ( z | X n , Y n , w ∗ n ) denote the cumulative distribution function (CDF) corresponding to p ∗ n ( z | X n , Y n , w ∗ n ) . The EFD provides an uncertainty measure for b ( θ ) . Practically, it can be constructed based on the samples { b ( ¯ θ 1 ) , b ( ¯ θ 2 ) , . . . , b ( ¯ θ M ) } , where { ¯ θ 1 , ¯ θ 2 , . . . , ¯ θ M } denotes the fiducial ¯ θ -samples collected at step (iv) of Algorithm 1. As a practical application, Kim and Liang (2025) applied EFI to quantify the uncertainty of individual treatment effects in causal inference.

Finally, we note that for a neural network model, its parameters are only unique up to certain loss-invariant transformations, such as reordering hidden neurons within the same hidden layer or simultaneously altering the sign or scale of certain connection weights (Sun et al., 2022). Therefore, for the w -network, the consistency of w ∗ n refers to its consistency with respect to one of the equivalent solutions to (3), while mathematically w ∗ n can still be treated as unique.

## 3 EFI for Uncertainty Quantification in PINNs

## 3.1 EFI Formulation for PDEs

Consider a multidimensional dynamic process, u ( x ) , defined on a domain Ω ⊂ R d through a PDE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x = ( x 1 , x 2 , . . . , x d -1 , t ) T ∈ R d indicates the space-time coordinate vector, u ( · ) represents the unknown solution, β are the parameters related to the physics, and f and b are called the physics term and initial/boundary term, respectively. The observations are given in the forms { x u i , u i } n u i =1 , { x f i , f i } n f i =1 , and { x b i , b i } n b i =1 . Let u ϑ ( x ) denote the DNN approximation to the solution u ( x ) , where ϑ denotes the DNN parameters. In data-generating equations, the observations can be expressed as

<!-- formula-not-decoded -->

where z u i , z f i , and z b i are independent Gaussian random errors with zero mean. Our objective is to infer u and/or β , as well as to quantify their uncertainty, given the data and governing physical law. In physics, inferring u with β known is termed the forward problem , while inferring u when β is unknown is referred to as the inverse problem .

When applying EFI to address this problem, the EFI network comprises two DNNs. The first, referred to as the data modeling network, is used to approximate u ( x ) . The second, called the w -network, is used to approximate the parameters of the data modeling network as well as other parameters in equation (9). For an illustration, see Figure 1. Specifically, for the inverse problem, the output of the w -network corresponds to θ = { ϑ , β } ; for the forward problem, its output corresponds to θ = { ϑ } .

Figure 1: An EFI network with a double neural network (double-NN) structure.

<!-- image -->

Given the data-generating equations, the energy function for EFI can be defined as follows:

<!-- formula-not-decoded -->

̸

̸

̸

where u n = ( u 1 , . . . , u n ) T , f = ( f 1 , . . . , f n ) T , b n = ( b 1 , . . . , b n ) T , z n = ( z u 1 , . . . , z u n ; z f 1 , . . . , z f n ; z b n , . . . , z b n ) T ; n = # { z u i = 0 , z f i = 0 , z b i = 0 } denotes the total number of noisy observations in the dataset; and η θ , η u , η f , and η b are belief weights for balancing

different terms. Fortunately, as shown in Liang et al. (2025), the choices for these terms will not affect much the performance of the algorithm as long as ϵ → 0 .

Using EFI to solve PINNs, if we can correctly impute Z n and consistently estimate the inverse function G ( · ) , the uncertainty of θ can be accurately quantified according to (8). However, consistent estimation of the inverse function is unattainable under the current EFI theoretical framework due to the high dimensionality of θ , which often far exceeds the sample size n . This limitation arises from the existing sparse deep learning theory (Sun et al., 2022), which constrains the dimension of θ to remain fixed or grow very slowly with n . To address this challenge, we propose a new theoretical framework for EFI, as detailed in Section 3.2, which extends EFI to accommodate large-scale models and addresses the constraints of the current framework.

## 3.2 A New Theoretical Framework of EFI for Large-Scale Models

EFI treats the random errors in observations as latent variables. Consequently, training the w -network is reduced to a problem of parameter estimation with missing data. Under a Bayesian setting, Liang et al. (2025) addressed the problem by solving equation (3) and employ Algorithm 1 for the solution. By imposing regularity conditions such as smoothness and dissipativity (Raginsky et al., 2017), Liang et al. (2025) established the following convergence results for Algorithm 1:

Lemma 3.1. (Theorem 4.1 and Theorem 4.2, (Liang et al., 2025)) Suppose the regularity conditions in Liang et al. (2025) hold, and the learning rate sequence { υ k : k = 1 , 2 , . . . } and the step size sequence { γ k : k = 1 , 2 , . . . } are set as: υ k = C υ c υ + k α and γ k = C γ c γ + k β for some constants C υ &gt; 0 , c υ &gt; 0 , C γ &gt; 0 and c γ &gt; 0 , and α, β ∈ (0 , 1] satisfying β ≤ α ≤ min { 1 , 2 β } .

- (i) (Root Consistency) There exists a root w ∗ n ∈ { w : ∇ w log π ( w | X n , Y n ) = 0 } such that

<!-- formula-not-decoded -->

for some constant ζ &gt; 0 and iteration number k 0 &gt; 0 .

- (ii) (Weak Convergence of Latent Variables) Let π ∗ z = π ( Z n | X n , Y n , w ∗ n ) and T k = ∑ k -1 i =0 υ i +1 , and let π ( T k ) z denote the probability law of Z ( k ) n . Then,

<!-- formula-not-decoded -->

for any k ∈ N , where W 2 ( · , · ) is the 2-Wasserstein distance, c 0 , c 1 , and c 2 are some positive constants, c LS is the logarithmic Sobolev constant of π ∗ z , and δ g is a coefficient reflecting the variation of the stochastic gradient used in latent variable imputation step.

In our implementation, the latent variable imputation step is done for each observation separately, ensuring δ g = 0 . We choose α ∈ (0 , 1] and set γ 1 ≺ 1 T 4 k for any T k , which ensures W 2 ( π ( T k ) z , π ∗ z ) → 0 as k →∞ . It is worth noting that Lemma 3.1 holds regardless of the size of the w -network. Thus, it remains valid for large-scale models. However, in this work, we do not impose any priors on w n , which corresponds to the non-informative prior setting π ( w n ) ∝ 1 .

Given the root consistency result, it is still necessary to establish that the resulting inverse function estimator (7) is consistent with respect to the true parameter θ ∗ to ensure valid downstream inference. Liang et al. (2025) established this consistency using sparse deep learning theory (Sun et al., 2022), but their approach was limited to settings where the dimension of θ is fixed or increases very slowly with the sample size n . In this work, we achieve the consistency by employing a narrow-neck w -network, which overcomes the limitation of the current framework of EFI.

To motivate the development of the narrow-neck w -network, we first note an important mathematical fact: As implied by (A6), each ˆ θ i ∈ R p in the EFI network tends to converge to a constant vector as ϵ → 0 . Consequently, different components of ˆ θ become highly correlated across n observations and ˆ θ can be effectively represented in a much lower-dimensional space, even though the dimension of θ may be very high. A straightforward solution to address this issue is to incorporate a restricted Boltzmann machine (RBM) (Hinton and Salakhutdinov, 2006) into the w -network, as illustrated in Figure A2, where the neck layer is binary and the last two layers form a Gaussian-binary RBM (Gu et al., 2022; Chu et al., 2017). As discussed in Liang et al. (2025), ˆ θ can be treated as Gaussian

random variables in the EFI network. The binary layer of the RBM serves as a dimensionality reducer for ˆ θ . Such a RBM-embedded w -network can be trained using the imputation-regularized optimization (IRO) algorithm (Liang et al., 2018a) in a similar way to that used in Wu et al. (2019).

Since the primary role of the w -network is forward learning of θ , it can also be formulated as a stochastic neural network (StoNet) (Liang et al., 2022; Sun and Liang, 2022), where the binary layer can be extended to be continuous. This StoNet-formulation leads to the following hierarchical model:

<!-- formula-not-decoded -->

where i = 1 , 2 , . . . , n , j = 1 , 2 , . . . , p , e i,j ∼ N (0 , σ 2 ˆ θ ) , v i ∈ R d h denotes a vector of random errors following a known distribution, d h denotes the width of the stochastic neck layer, h denotes the number of hidden layers of the w -network, µ i = g ( X i , Y i , Z i ; w (1) n ) is the mean of the feeding vector to the stochastic neck layer, and R ( · ) represents a transformation. With a slight abuse of notation, we assume that m i has been augmented with a constant component to account for the intercept term in the regression model for ˆ θ ( j ) i . For R ( · ) , we recommend the setting:

- (*) The neck layer is stochastic with: R ( µ i , v i ) = Ψ( µ i + v i ) , where v i ∼ N (0 , σ 2 v ) with a pre-specified value of σ 2 v , and Ψ is an (element-wise) activation function.

Under this setting, the w (1) -network forms a nonlinear Gaussian regression with response µ i + v i for i = 1 , 2 , . . . , n . Conceptually, the StoNet can be trained using the stochastic EM algorithm (Celeux et al., 1996) by iterating between the steps: (i) Latent variable imputation : Impute the latent variables { v i : i = 1 , . . . , n } and Z n = { z 1 , . . . , z n } conditioned on the current estimates of w n = { w (1) n , w (2) n } , where w (2) n = { ξ j : j = 1 , 2 , . . . , p } . (ii) Optimization : Conditioned on the imputed latent variables, update the estimates of w (1) n and w (2) n separately. Specifically, w (1) n can be estimated by training the w (1) n -network using SGD, and w (2) n can be estimated by performing p linear regressions as specified in (11).

Following the standard theory of the stochastic EM algorithm (Nielsen, 2000; Liang et al., 2018a), { w (1) n , w (2) n } will converge to a solution to the equation: ∇ w n log π ϵ ( w n | X n , Y n ) = ∫ [ ∇ w n log π ϵ ( w n | X n , Y n , Z n , V n ) π ϵ ( Z n , V n | X n , Y n , w n ) ] d Z n d V n = 0 , as the sample size n and the number of iterations of the algorithm become large, where the prior π ( w n ) ∝ 1 and V n = ( v 1 , v 2 , . . . , v n ) T . Denote the converged solution by w ∗ n = { w ∗ (1) n , w ∗ (2) n } . The consistency of the resulting inverse function estimator can be established by leveraging the sufficient dimension reduction property of the StoNet (Liang et al., 2022) and the prediction property of linear regressions. Specifically, { m i : i = 1 , . . . , n } serves as a sufficient dimension reduction for { ( X i , Y i , Z i ) : i = 1 , . . . , n } . The consistency of the inverse mapping can thus be ensured by the linear relationship ˆ θ i ∼ m i , as described in (11), along with the prediction property of linear regression. Note that the linear relationship ˆ θ i ∼ m i can be generally ensured by the universal approximation ability of the DNN.

Toward a rigorous mathematical development, we impose the following conditions:

<!-- formula-not-decoded -->

where E ( · ) denotes expectation; λ min ( · ) and λ max ( · ) denote the minimum and maximum eigenvalue of a matrix, respectively; and c ′ &gt; 0 , δ &gt; 0 , and ρ max &gt; 0 are some constants. Condition (i) is generally satisfied by choosing a sufficiently narrow neck layer, in particular, we set d h ≺ n . Condition (ii) is justified in Appendix A4.2.

By the asymptotic equivalence between the StoNet and conventional DNN (Liang et al., 2022) (see also Appendix A4.1), the StoNet can be trained by directly training the DNN. Based on this asymptotic equivalence, Theorem 3.2 establishes the consistency of the inverse mapping learned by Algorithm 1 for large-scale models, see Appendix A4.3 for the proof.

Theorem 3.2. Suppose that the narrow neck layer is set as in (*) with σ 2 v ≺ ϵ ηhd h p , the activation function Ψ( · ) is c -Lipschitz continuous, and d h ≺ n is sufficiently small such that the conditions in

(12) are satisfied while admitting a non-empty zero energy set Z n . Additionally, assume that the other regularity conditions (Assumptions A4.1-A4.2 in Appendix A4) hold. If ϵ ≺ min { n p 2 d 2 h , h pd h } , then the inverse mapping θ = G ( Y n , X n , Z n ) , learned by Algorithm 1 with a narrow neck w -network, is consistent.

This narrow neck setting for the w -network eliminates the need to specify a prior for w (1) n . Notably, under this setting, the resulting estimate θ is not necessarily sparse, introducing a new research paradigm for high-dimensional problems; moreover, the w (1) n -network does not need to be excessively large, keeping the overall size of the w -network manageable. This facilitates the application of EFI to high-dimensional and complex models.

## 4 Simulation Studies

We compared EFI with Bayesian and dropout methods across multiple simulation studies, including the Poisson equation under various settings and the Black-Scholes model. Below, we present the results for the 1-D Poisson equation, with results from other studies provided in Appendix A5. The experimental settings for all simulation studies are detailed in Appendix A6.

## 4.1 1-D Poisson Equation

Consider a 1-D Poisson equation as in Yang et al. (2021):

<!-- formula-not-decoded -->

where Ω = [ -0 . 7 , 0 . 7] , β = 0 . 01 , u = sin 3 (6 x ) , and f can be derived from (13). Here, we assume the analytical expression of f is unavailable; instead, 200 sensor measurements of f are available with the sensors equidistantly distributed across Ω . Additionally, there are two sensors at x = -0 . 7 and x = 0 . 7 to provide the left/right Dirichlet boundary conditions for u . We model the boundary noise as z u i ∼ N (0 , 0 . 05 2 ) for i = 1 , 2 , . . . , 20 , with 10 observations drawn from each boundary sensor. For the interior domain, we assume noise-free measurements of f . This simulation is repeated over 100 independent datasets to evaluate the robustness and reliability of the proposed approach.

The Bayesian method was first applied to this example as in Yang et al. (2021) using a two-hiddenlayer DNN, with 50 hidden units in each layer, to approximate u ( x ) . The same network architecture was also used for this example by other methods as shown in Table 1. The negative log-posterior is given as follows:

<!-- formula-not-decoded -->

where C denotes the log-normalizing constant, n u = 20 , n f = 200 , n b = 0 , and σ u = 0 . 05 . The prior π ( ϑ ) is specified as in Yang et al. (2021), where each unknown parameter is assigned an independent standard Gaussian distribution. The same prior is used across all Bayesian simulations in this paper. We implement the method using the Python package hamiltorch (Cobb and Jalaian, 2021). Since f is observed exactly, the corresponding variance σ f should theoretically be zero. However, the formulation in (14) does not allow σ f = 0 , necessitating experimentation with different non-zero values of σ f . These variations resulted in different widths of confidence intervals, as illustrated in Figure A4. Notably, there is no clear guideline for selecting σ f to achieve the appropriate interval width necessary for the desired coverage rate. This ambiguity underscores the dilemma (mentioned in Introduction) inherent in the Bayesian method. Specifically, the f -term in (14) functions as part of the prior for the parameter ϑ .

A similar issue arises when selecting the dropout rate for PINNs. As illustrated in Figure A3, the uncertainty estimation in PINNs is highly sensitive to the choice of the dropout rate. When the dropout rate approaches zero, the confidence interval collapses into a point estimate, failing to capture uncertainty. Conversely, excessively large dropout rates introduce significant bias, leading to unreliable interval estimates. This highlights the challenge of determining an appropriate dropout rate to balance bias and variability in uncertainty quantification. As a possible way to address this issue,

Concrete Dropout (Gal et al., 2017) was implemented for the example with the code provided at https://github.com/yaringal/ConcreteDropout . Unlike the conventional dropout method using fixed dropout rate, Concrete Dropout treats the dropout rate as a hyperparameter and learns it simultaneously when training the model.

Figure 2: EFI-PINN diagnostic for 1D-Poisson

<!-- image -->

We then applied EFI to this example, using the same DNN structure for the data-modeling network. Figure 2 summarizes the imputed random errors by EFI for 100 datasets, showing that EFI correctly imputes the realized random errors from the observations. Consequently, EFI achieves the correct recovery of the underlying physical law.

Table 1: Metrics for 1D-Poisson, averaged over 100 runs.

| Method                                                                                                                                                           | MSE                                                                                                                                                                                 | Coverage Rate                                                                                                                                                     | CI-Width                                                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PINN (no dropout) Dropout (0.5%) Dropout (1%) Dropout (5%) Concrete Dropout Bayesian ( σ f = 0 . 05 ) Bayesian ( σ f = 0 . 005 ) Bayesian ( σ f = 0 . 0005 ) EFI | 0.000121 (0.000012) 0.000228 (0.000024) 0.000851 (0.000111) 0.006276 (0.000982) 0.000184 (0.000020) 0.000826 (0.000102) 0.000170 (0.000020) 0.001460 (0.000219) 0.000148 (0.000016) | 0.0757 (0.011795) 1.0000 (0.000000) 0.9999 (0.000104) 0.9893 (0.004024) 0.1185 (0.018002) 0.9997 (0.000302) 0.9914 (0.008600) 0.5884 (0.032390) 0.9517 (0.014186) | 0.002139 (0.000024) 0.191712 (0.004367) 0.274651 (0.006734) 0.660927 (0.030789) 0.003706 (0.000073) 0.372381 (0.002635) 0.081557 (0.000692) 0.055191 (0.001123) 0.050437 (0.000462) |

We evaluated the accuracy and robustness of each method by recording three metrics, as summarized in Table 1, based on an average across 100 repeated experiments. The 'MSE' measures the mean squared distance between the network prediction and the true solution. The coverage rate represents the percentage of the true solution contained within the 95% confidence interval, while the CI-Width quantifies the corresponding interval width. As discussed above, the dropout method produces inflated coverage rates and excessively wide confidence intervals, with CI widths ranging from 0.1917 to 0.6609. Similarly, the Bayesian method is sensitive to the choice of σ f , with larger σ f values yielding inflated coverage rates, while smaller σ f values result in underestimated coverage rates. In contrast, EFI is free of hyperparameter tuning, making inference on the model parameters based on the observations and the embedded physical law. It produces the most balanced results, with a mean squared error (MSE) of 1 . 48 × 10 -4 and a coverage rate of 95.17%, closely aligning with the nominal 95% target. Moreover, it provides the shortest CI-width 0.0504. Concrete Dropout achieves a comparable MSE as EFI, indicating similar model estimation accuracy. However, it performs significantly worse in terms of uncertainty quantification, with a coverage rate of only 11.85%, far below the nominal 95%. This substantial under-coverage is due to an underestimation of predictive uncertainty, as evidenced by the markedly narrower confidence interval width.

## 5 Real Data Examples

This section demonstrates the abiliy of the EFI-PINN framework to quantify uncertainty for models learned from real data. We considered two models: the Montroll growth model,

<!-- formula-not-decoded -->

where k , C , and θ are unknown parameters; and the reaction-diffusion model governed by the generalized Porous-Fisher-Kolmogoriv-Petrovsky-Piskunov (P-FKPP) equation:

<!-- formula-not-decoded -->

where D , m , and r are unknown parameters. We modeled the Chinese hamster V79 fibroblast tumor cell growth data (Rodrigues, 2024) using equation (15), and the scratch assay data (Jin and Cai, 2006) using equation (16). Additional details about the datasets are provided in the Appendix. Figure 3 illustrates that the proposed EFI-PINN framework is not only capable of learning the PDE models from data but also effectively quantifying the uncertainty of the models. Further results and discussions can be found in the Appendix. The prediction uncertainty can also be quantified with the proposed method.

Figure 3: Confidence bands (shaded areas) for the learned models: (left): Montroll growth model; (right): generalized P-FKPP model.

<!-- image -->

## 6 Conclusion

This paper presents a novel theoretical framework for EFI, enabling effective uncertainty quantification for PINNs. EFI addresses the challenge of uncertainty quantification through a unique approach of 'solving data-generating equations,' transforming it into an objective process that facilitates the construction of honest confidence sets. In contrast, existing methods such as dropout and Bayesian approaches rely on subjective hyperparameters (e.g., dropout rates or priors), undermining the honesty of the resulting confidence sets. This establishes a new research paradigm for statistical inference of complex models, with the potential to significantly impact the advancement of modern data science.

Although Algorithm 1 performs well in our examples, its efficiency can be further improved through several enhancements. For instance, in the latent variable imputation step, the SGLD algorithm (Welling and Teh, 2011) can be replaced with the Stochastic Gradient Hamiltonian Monte Carlo (Chen et al., 2014), which offers better sampling efficiency. Similarly, in the parameter updating step, the SGD algorithm can be accelerated with momentum. Momentum-based algorithms facilitate faster convergence to a good local minimum during the early stages of training. As training progresses, the momentum can be gradually reduced to zero, ensuring alignment with the convergence theory of EFI. This approach balances computational efficiency with theoretical rigor, enhancing the overall performance of EFI.

While the examples presented focus on relatively small-scale models, this is not a limitation of the approach. In principle, EFI can be extended to large-scale models, such as ResNets and CNNs, through transfer learning-a direction we plan to explore in future work.

## Acknowledgments

Liang's research is supported in part by the NSF grant DMS-2210819 and the NIH grant R01GM152717.

Shih's research is partially supported by MSK Cancer Center Support Grant/Core Grant (P30 CA008748).

## References

- Augustin, T., Coolen, F. P. A., de Cooman, G., and Troffaes, M. C. M. (2014), Introduction to imprecise probabilities , John Wiley &amp; Sons.
- Black, F. and Scholes, M. S. (1973), 'The Pricing of Options and Corporate Liabilities,' Journal of Political Economy , 81, 637 - 654.
- Caprio, M., Dutta, S., Jang, K. J., Lin, V., Ivanov, R., Sokolsky, O., and Lee, I. (2023a), 'Credal Bayesian Deep Learning,' .
- -(2023b), 'Imprecise Bayesian Neural Networks,' ArXiv , abs/2302.09656.
- Celeux, G., Chauveau, D., and Diebolt, J. (1996), 'Stochastic versions of the EM algorithm: an experimental study in the mixture case,' Journal of Statistical Computation and Simulation , 55, 287-314.
- Chen, T., Fox, E., and Guestrin, C. (2014), 'Stochastic gradient hamiltonian monte carlo,' in International conference on machine learning , pp. 1683-1691.
- Chu, J., Wang, H., Meng, H., Jin, P., and Li, T. (2017), 'Restricted Boltzmann Machines With Gaussian Visible Units Guided by Pairwise Constraints,' IEEE Transactions on Cybernetics , 49, 4321-4334.
- Cobb, A. D. and Jalaian, B. (2021), 'Scaling Hamiltonian Monte Carlo Inference for Bayesian Neural Networks with Symmetric Splitting,' Uncertainty in Artificial Intelligence .
- Cuomo, S., Cola, V. S. D., Giampaolo, F., Rozza, G., Raissi, M., and Piccialli, F. (2022), 'Scientific Machine Learning Through Physics-Informed Neural Networks: Where we are and What's Next,' Journal of Scientific Computing , 92.
- Deng, W., Zhang, X., Liang, F., and Lin, G. (2019), 'An adaptive empirical Bayesian method for sparse deep learning,' Advances in neural information processing systems , 32.
- Dittmer, S., King, E. J., and Maass, P. (2018), 'Singular Values for ReLU Layers,' IEEE Transactions on Neural Networks and Learning Systems , 31, 3594-3605.
- Fraser, D. A. S. (1966), 'Structural probability and a generalization,' Biometrika , 53, 1-9.
- -(1968), The Structure of Inference , New York-London-Sydney: John Wiley &amp; Sons.
- Gal, Y. and Ghahramani, Z. (2016), 'Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,' in International Conference on Machine Learning , vol. PMLR 48, pp. 1050-1059.
- Gal, Y., Hron, J., and Kendall, A. (2017), 'Concrete dropout,' Advances in neural information processing systems , 30.
- Ghosal, S., Ghosh, J. K., and van der Vaart, A. (2000), 'Convergence rates of posterior distributions,' Annals of Statistics , 28, 500-531.
- Gu, L., Yang, L., and Zhou, F. (2022), 'Approximation properties of Gaussian-binary restricted Boltzmann machines and Gaussian-binary deep belief networks,' Neural networks : the official journal of the International Neural Network Society , 153, 49-63.
- Hannig, J. (2009), 'On generalized fiducial inference,' Statistica Sinica , 19, 491-544.
- Hannig, J., Iyer, H., Lai, R. C. S., and Lee, T. C. M. (2016), 'Generalized Fiducial Inference: A Review and New Results,' Journal of the American Statistical Association , 111, 1346-1361.
- Higham, N. J. and Cheng, S. H. (1998), 'Modifying the inertia of matrices arising in optimization,' Linear Algebra and its Applications , 261-279.
- Hinton, G. E. and Salakhutdinov, R. (2006), 'Reducing the Dimensionality of Data with Neural Networks,' Science , 313, 504 - 507.

- Jin, J. and Cai, T. T. (2006), 'Estimating the Null and the Proportion of Nonnull Effects in Large-Scale Multiple Comparisons,' Journal of the American Statistical Association , 102, 495 - 506.
- Kim, S. and Liang, F. (2025), 'Extended fiducial inference for individual treatment effects via deep neural networks,' Statistics and Computing , 35.
- Lagergren, J. H., Nardini, J. T., Baker, R. E., Simpson, M. J., and Flores, K. B. (2020), 'Biologicallyinformed neural networks guide mechanistic modeling from sparse experimental data,' PLoS Computational Biology , 16.
- Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2016), 'Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,' in Neural Information Processing Systems .
- Liang, F., Jia, B., Xue, J., Li, Q., and Luo, Y. (2018a), 'An imputation-regularized optimization algorithm for high-dimensional missing data problems and beyond,' Journal of the Royal Statistical Society, Series B , 80, 899-926.
- Liang, F., Kim, S., and Sun, Y. (2025), 'Exended Fiducial Inference: Toward an Automated Process of Statistical Inference,' Journal of the Royal Statistical Society Series B , 87, 98-131.
- Liang, F., Li, Q., and Zhou, L. (2018b), 'Bayesian Neural Networks for Selection of Drug Sensitive Genes,' Journal of the American Statistical Association , 113, 955-972.
- Liang, S., Sun, Y., and Liang, F. (2022), 'Nonlinear Sufficient Dimension Reduction with a Stochastic Neural Network,' NeurIPS 2022 .
- Martin, R. (2023), 'Fiducial inference viewed through a possibility-theoretic inferential model lens,' Journal of Machine Learning Research , 215, 299-310.
- Martin, R. and Liu, C. (2013), 'Inferential Models: A Framework for Prior-Free Posterior Probabilistic Inference,' Journal of the American Statistical Association , 108, 301 - 313.
- -(2015), Inferential Models: Reasoning with Uncertainty , CRC Press.
- Marusic, M., Bajzer, Z., Freyer, J. P., and Vuk-Pavlovi´ c, S. (1994), 'Analysis of growth of multicellular tumour spheroids by mathematical models,' Cell Proliferation , 27.
- Milnor, J. and Stasheff, J. D. (1974), Characteristic Classes , Princeton University Press.
- Murph, A. C., Hannig, J., and Williams, J. P. (2022), 'Generalized Fiducial Inference on Differentiable Manifolds,' arXiv:2209.15473 .
- Nielsen, S. (2000), 'The stochastic EM algorithm: Estimation and asymptotic results,' Bernoulli , 6, 457-489.
- Raginsky, M., Rakhlin, A., and Telgarsky, M. (2017), 'Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis,' in Conference on Learning Theory , PMLR, pp. 1674-1703.
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019), 'Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,' J. Comput. Phys. , 378, 686-707.
- Rodrigues, J. A. (2024), 'Using Physics-Informed Neural Networks (PINNs) for Tumor Cell Growth Modeling,' Mathematics , 12.
- Song, Q., Sun, Y., Ye, M., and Liang, F. (2020), 'Extended Stochastic Gradient MCMC for LargeScale Bayesian Variable Selection,' Biometrika , 107, 997-1004.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014), 'Dropout: a simple way to prevent neural networks from overfitting,' The journal of machine learning research , 15, 1929-1958.
- Sun, Y. and Liang, F. (2022), 'A kernel-expanded stochastic neural network,' Journal of the Royal Statistical Society Series B , 84, 547-578.

- Sun, Y., Song, Q., and Liang, F. (2022), 'Consistent Sparse Deep Learning: Theory and Computation,' Journal of the American Statistical Association , 117, 1981-1995.
- Vovk, V., Gammerman, A., and Shafer, G. (2005), Algorithmic Learning in a Random World , Springer.
- Welling, M. and Teh, Y. W. (2011), 'Bayesian learning via stochastic gradient Langevin dynamics,' in Proceedings of the 28th international conference on machine learning (ICML-11) , pp. 681-688.
- Wu, M., Luo, Y., and Liang, F. (2019), 'Accelerate Training of Restricted Boltzmann Machines via Iterative Conditional Maximum Likelihood Estimation.' Statistics and its interface , 12, 377-385.
- Yang, L., Meng, X., and Karniadakis, G. E. (2021), 'B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data,' Journal of Computational Physics , 425, 109913.
- Yao, Y., Yan, S., Goehring, D., Burgard, W., and Reichardt, J. (2024), 'Improving Out-of-Distribution Generalization of Trajectory Prediction for Autonomous Driving via Polynomial Representations,' ArXiv , abs/2407.13431.
- Zhang, D., Lu, L., Guo, L., and Karniadakis, G. E. (2019), 'Quantifying total uncertainty in physicsinformed neural networks for solving forward and inverse stochastic problems,' J. Comput. Phys. , 397, 108850.
- Zou, Z., Meng, X., and Karniadakis, G. E. (2025), 'Uncertainty quantification for noisy inputs-outputs in physics-informed neural networks and neural operators,' Computer Methods in Applied Mechanics and Engineering .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract states that we provide theoretical proof for the proposed framework and numerical experiment for overcoming limitations for Bayesian PINN and Dropout PINN.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the conclusion, we mentioned that the examples we presented are relatively small scale models, however, the proposed algorithm can be extended easily to large scale models through transfer learning.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: In this paper, we provide convergence theory for proposed algorithm.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide our source code and hyperparameters in the supplementary material.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide our source code and hyperparameters in the supplementary mate- rial.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide our source code and hyperparameters in the supplementary material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All our experiments are replicated 100 times with randomly chosen random seeds, each metic is averaged over all 100 runs to ensure statistical significance.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide the wall-clock time for the Poisson-1D experiment with different algorithms in the supplementary material.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research relies solely on simulated and publicly available open-source data. No human, animal, or private data were used. The methods developed and evaluated pose no foreseeable ethical concerns and comply fully with the NeurIPS Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This work develops a foundational algorithm for uncertainty quantification in PINNs. While it does not involve sensitive data or direct deployment, it may support future applications in scientific modeling and engineering.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper does not release any pretrained models or datasets with potential for misuse. All experiments are conducted using simulation or publicly available open-source data, and the developed methods pose no foreseeable dual-use concerns.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All external assets used in this work-such as open-source code and datasets-are properly credited in the paper. We ensured that their licenses (e.g., MIT, Apache 2.0) were respected, and we include relevant citations and links where applicable.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets yet.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve any crowdsourcing or research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve research with human subjects or any activity requiring IRB or equivalent approval.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This paper does not involve the use of large language models (LLMs) as part of the core methodology. Any use of LLMs was limited to minor writing or editing support and did not influence the scientific content or originality of the research.

## Appendix

This appendix is organized as follows. Section A1 provides a brief review of the EFI method and its self-diagnostic property. Section A2 presents the analytical expression of the extended fiducial density (EFD) function of Z n . Section A3 includes an illustrative plot of the proposed narrow-neck w -network. Section A4 gives the proof of Theorem 3.2. Section A5 presents additional numerical results. Finally, Section A6 details the parameter settings used in our numerical experiments.

## A1 A Brief Review of the EFI Method and Its Self-Diagnosis

## A1.1 The EFI Method

To ensure a smooth presentation of the EFI method, this section partially overlaps with Section 2 of the main text. Consider a regression model:

<!-- formula-not-decoded -->

where Y ∈ R and X ∈ R d represent the response and explanatory variables, respectively; θ ∈ R p represents the vector of parameters; and Z ∈ R represents a scaled random error following a known distribution π 0 ( · ) . Suppose that a random sample of size n , denoted by { ( y 1 , x 1 ) , ( y 2 , x 2 ) , . . . , ( y n , x n ) } , has been collected from the model. In structural inference, the observations can be expressed in data-generating equations as follows:

<!-- formula-not-decoded -->

This system of equations consists of n + p unknowns, namely, { θ , z 1 , z 2 , . . . , z n } , while there are only n equations. Therefore, the values of θ cannot be uniquely determined by the data-generating equations, and this lack of uniqueness of unknowns introduces uncertainty in θ .

Let Z n = { z 1 , z 2 , . . . , z n } denote the unobservable random errors contained in the data, which are also called latent variables in EFI. Let G ( · ) denote an inverse function/mapping for θ , i.e.,

<!-- formula-not-decoded -->

It is worth noting that the inverse function is generally non-unique. For example, it can be constructed by solving any p equations in (1) for θ . As noted by Liang et al. (2025), this non-uniqueness of inverse function mirrors the flexibility of frequentist methods, where different estimators of θ can be constructed to achieve desired properties such as efficiency, unbiasedness, and robustness.

Figure A1: Diagram of EFI given in Liang et al. (2025): The orange nodes and orange links form a deep neural network (DNN), referred to as the w -network, which is parameterized by w n (with the subscript n indicating its dependence on the training sample size n ); the green node represents the latent variable to impute; and the black lines represent deterministic functions.

<!-- image -->

Since the inverse function G ( · ) is generally unknown, Liang et al. (2025) proposed to approximate it using a sparse DNN, see Figure A1 for illustration. The EFI network has two output nodes defined,

respectively, by

<!-- formula-not-decoded -->

where ˜ y i = f ( x i , z i , ¯ θ ) , f ( · ) is as specified in (1), and d ( · ) is a function measuring the difference between y i and ˜ y i . That is, the node e i, 1 quantifies the variation of ˆ θ i , while the node e i, 2 represents the fitting error. For a normal linear/nonlinear regression, d ( · ) can be defined as

<!-- formula-not-decoded -->

For logistic regression, it is defined as a ReLU function, see Liang et al. (2025) for details.

Let ˆ θ i := ˆ g ( y i , x i , z i , w n ) denote the DNN prediction function parameterized by the weights w n in the EFI network, and let

<!-- formula-not-decoded -->

which serves as an estimator of the inverse function G ( · ) .

EFI defines an energy function

<!-- formula-not-decoded -->

where η θ &gt; 0 is a regularization parameter, ˆ θ i 's and ¯ θ can be expressed as functions of ( Y n , X n , Z n , w n ) , and d ( · ) is a function measuring the difference between y i and ˜ y i . The likelihood function is given by

<!-- formula-not-decoded -->

for some constant ϵ close to 0. As discussed in Liang et al. (2025), the choice of η θ does not affect much on the performance of EFI as long as ϵ is sufficiently small. Subsequently, the posterior of w n is given by

<!-- formula-not-decoded -->

where π ( w n ) denotes the prior of w n ; and the predictive distribution of Z n is given by

<!-- formula-not-decoded -->

where π ⊗ n 0 ( Z n ) = ∏ n i =1 π 0 ( z i ) under the assumption that z i 's are independently identically distributed (i.i.d.). In EFI, w n is estimated through maximizing the posterior π ϵ ( w n | X n , Y n ) given the observations { X n , Y n } . By the Bayesian version of Fisher's identity (Song et al., 2020), the gradient equation ∇ w n log π ϵ ( w n | X n , Y n ) = 0 can be re-expressed as

<!-- formula-not-decoded -->

which can be solved using an adaptive stochastic gradient MCMC algorithm (Liang et al., 2022; Deng et al., 2019). The algorithm works by iterating between the latent variable imputation and parameter updating steps, see Algorithm 1 for the pseudo-code. This algorithm is termed 'adaptive' because the transition kernel in the latent variable imputation step changes with the working parameter estimate of w n . The parameter updating step can be implemented using mini-batch SGD, and the latent variable imputation step can be executed in parallel for each observation ( y i , x i ) . Hence, the algorithm is scalable with respect to large datasets.

Under mild conditions for the adaptive SGLD algorithm, it can be shown that

<!-- formula-not-decoded -->

where w ∗ n denotes a solution to equation (3) and p → denotes convergence in probability, and that

<!-- formula-not-decoded -->

in 2-Wasserstein distance, where d ⇝ denotes weak convergence. To study the limit of (6) as ϵ decays to 0, i.e.,

<!-- formula-not-decoded -->

where p ∗ n ( z | Y n , X n , w ∗ n ) is referred to as the extended fiducial density (EFD) of Z n , Liang et al. (2025) impose specific conditions on the structure of the w -network, including that the w -network is sparse and that the output layer width (i.e., the dimension of θ ) is either fixed or grows very slowly with the sample size n . Under these assumptions, they prove the consistency of w ∗ n based on the sparse deep learning theory developed in Sun et al. (2022). This consistency further implies that

<!-- formula-not-decoded -->

serves as a consistent estimator for the inverse function/mapping θ = G ( Y n , X n , Z n ) . Refer to Appendix A2 for the analytic expression of p ∗ n ( z | Y n , X n , w ∗ n ) .

Let Z n = { z ∈ R n : U n ( Y n , X n , Z n , w ∗ n ) = 0 } denote the zero-energy set. Under some regularity conditions on the energy function, Liang et al. (2025) proved that Z n is invariant to the choice of G ( · ) . Let Θ := { θ ∈ R p : θ = G ∗ ( Y n , X n , z ) , z ∈ Z n } denote the parameter space of the target model, which represents the set of all possible values of θ that G ∗ ( · ) takes when z runs over Z n . Then, for any function b ( θ ) of interest, its EFD µ ∗ n ( ·| Y n , X n ) associated with G ∗ ( · ) is given by

<!-- formula-not-decoded -->

for any measurable set B ⊂ Θ , where Z n ( B ) = { z ∈ Z n : b ( G ∗ ( Y n , X n , z )) ∈ B } , and P ∗ n ( z | X n , Y n , w ∗ n ) denote the cumulative distribution function (CDF) corresponding to p ∗ n ( z | X n , Y n , w ∗ n ) . The EFD provides an uncertainty measure for b ( θ ) . Practically, it can be constructed based on the samples { b ( ¯ θ 1 ) , b ( ¯ θ 2 ) , . . . , b ( ¯ θ M ) } , where { ¯ θ 1 , ¯ θ 2 , . . . , ¯ θ M } denotes the fiducial ¯ θ -samples collected at step (iv) of Algorithm 1.

Finally, we note that for a neural network model, its parameters are only unique up to certain loss-invariant transformations, such as reordering hidden neurons within the same hidden layer or simultaneously altering the sign or scale of certain connection weights (Sun et al., 2022). Therefore, for the w -network, the consistency of w ∗ n refers to its consistency with respect to one of the equivalent solutions to (3), while mathematically w ∗ n can still be treated as unique.

## A1.2 Self-Diagnosis in EFI

Given the flexibility of DNN models, reliable diagnostics are crucial in deep learning to ensure model robustness and accuracy while identifying potential issues during training. Unlike dropout and Bayesian methods, which lack self-diagnostic capabilities, EFI includes a built-in mechanism for self-diagnosis. Specifically, this can be achieved through (i) analyzing the QQ-plot of the imputed random errors, and (ii) verifying that the energy function U n converges to zero.

According to Lemma 3.1, the imputed random errors ˆ Z n should follow the same distribution as the true random errors Z n . Since the theoretical distribution of Z n is known, the convergence of ˆ Z n can be assessed using QQ-plot as shown in Figure 2(b). When Z n has been correctly imputed, the energy function must converge to zero to ensure the consistency of the inverse function estimator. In practice, we can check whether U n ( Y n , X n , Z n , w n ) = o ( ϵ ) as ϵ → 0 . The validity of inference for the model uncertainty can thus be ensured if both diagnostic tests are satisfied. This diagnostic method is entirely data-driven, offering a simple way for validating the EFI results.

If the diagnostic tests are not satisfied, the hyperparameters of EFI can be adjusted to ensure both tests are met for valid inference. These adjustments may include modifying the width of the neck layer, the size of the w (1) n -network, the size of the data-modeling network, as well as tuning the learning rates and iteration numbers used in Algorithm 1.

## A2 Extended Fiducial Density Function of Z n

Let Z n = { z ∈ R n : U n ( Y n , X n , Z n , w ∗ n ) = 0 } denote the zero-energy set, and let P ∗ n ( z | X n , Y n , w ∗ n ) denote the cumulative distribution function (CDF) corresponding to p ∗ n ( z | X n , Y n , w ∗ n ) . Under

some regularity conditions on the energy function, Liang et al. (2025) proved that Z n is invariant to the choice of G ( · ) . Furthermore, they studied the convergence of lim ϵ ↓ 0 π ϵ ( z | X n , Y n , w ∗ n ) in two cases: Π n ( Z n ) &gt; 0 and Π n ( Z n ) = 0 , where Π n ( · ) denotes the probability measure corresponding to the density function π ⊗ 0 ( z ) on R n . Specifically,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is invariant to the choices of the inverse function G ( · ) and energy function U n ( · ) . For example, the logistic regression belongs to this case as shown in Liang et al. (2025).

- (b) ( Π n ( Z n ) = 0 ) : In this case, Z n forms a manifold in R n with the highest dimension p , and p ∗ n ( z | Y n , X n , w ∗ n ) concentrates on the highest dimensional manifold and is given by

<!-- formula-not-decoded -->

where ν is the sum of intrinsic measures on the p -dimensional manifold in Z n , and t ∈ R n -p denotes the coefficients of the normalized smooth normal vectors in the tubular neighborhood decomposition (Milnor and Stasheff, 1974) of z .

By Theorem 3.2 and Lemma 4.2 in Liang et al. (2025), if the target model is noise-additive and d ( · ) is specified as in (A4), then P ∗ n ( z | X n , Y n , w ∗ n ) in (A14) can be reduced to

<!-- formula-not-decoded -->

That is, under the consistency of w ∗ n , p ∗ n ( z | X n , Y n , w ∗ n ) is reduced to a truncated density function of π ⊗ n 0 ( z ) on the manifold Z n , while Z n itself is also invariant to the choice of the inverse function. In other words, for noise-additive models, the EFD of Z n is asymptotically invariant to the inverse function we learned given its consistency.

## A3 Narrow-Neck w -Networks

Figure A2: A conceptual structure of narrow neck w -networks.

<!-- image -->

## A4 Proof of Theorem 3.2

## A4.1 Asymptotic Equivalence between StoNets and DNNs

We first provide a brief review of the theory regarding asymptotic equivalence between the StoNet and DNN models, which was originally established in Liang et al. (2022).

Consider a StoNet model:

<!-- formula-not-decoded -->

where X ∈ R p and Y ∈ R d h +1 represent the input and response variables, respectively; Y i ∈ R d i are latent variables; e i ∈ R d i are introduced noise variables; b i ∈ R d i and w i ∈ R d i × d i -1 are model parameters, d 0 = p , and Ψ( Y i -1 ) = ( ψ ( Y i -1 , 1 ) , ψ ( Y i -1 , 2 ) , . . . , ψ ( Y i -1 ,d i -1 )) T is an element-wise activation function for i = 1 , . . . , h +1 . The StoNet defines a latent variable model that reformulates the DNN as a composition of many simple regressions. In this context, we assume that e i ∼ N (0 , σ 2 i I d i ) for i = 1 , 2 , . . . , h, h +1 , though other distributions can also be considered for e i 's (Sun and Liang, 2022).

The DNN model corresponding to (A16) is given as follows:

<!-- formula-not-decoded -->

Let ϑ = { b 1 , w 1 , . . . , b h +1 , w h +1 } denote the parameter set of the DNN model. Let π DNN ( Y | X , ϑ ) denote the likelihood function of the DNN model (A17), and let π ( Y , Y mis | X , ϑ ) denote the likelihood function of the StoNet (A16). Let Q ∗ ( ϑ ) = E (log π DNN ( Y | X , ϑ )) , where the expectation is taken with respect to the joint distribution π ( X , Y ) , Liang et al. (2022) made the following assumption regarding the network structure, activation function, and the variance of the latent variables of StoNet:

Assumption A4.1. (i) Parameter space ˜ Θ (of ϑ ) is compact; (ii) For any ϑ ∈ ˜ Θ , E (log π ( Y , Y mis | X , ϑ )) 2 &lt; ∞ ; (iii) The activation function ψ ( · ) is c -Lipschitz continuous; (iv) The network's widths d l 's and depth h are allowed to increase with n ; (v) The noise introduced in StoNet satisfies the following condition: σ 1 ≤ σ 2 ≤ · · · ≤ σ h +1 , and d h +1 ( ∏ h i = k +1 d 2 i ) d k σ 2 k ≺ σ 2 h +1 h for any k ∈ { 1 , 2 , . . . , h } .

Assumption A4.2. (i) Q ∗ ( ϑ ) is continuous in ϑ and uniquely maximized at ϑ ∗ ; (ii) for any ϵ &gt; 0 , sup ϑ ∈ Θ \ B ( ϵ ) Q ∗ ( ϑ ) exists, where B ( ϵ ) = { ϑ : ∥ ϑ -ϑ ∗ ∥ &lt; ϵ } , and δ = Q ∗ ( ϑ ∗ ) -sup ϑ ∈ Θ \ B ( ϵ ) Q ∗ ( ϑ ) &gt; 0 .

Under Assumptions A4.1 and A4.2, Liang et al. (2022) proved the following lemma.

Lemma A4.3. (Liang et al., 2022) Suppose that Assumptions A4.1-A4.2 hold, and π ( Y , Y mis | X , ϑ ) is continuous in ϑ . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϑ ∗ = arg max ϑ ∈ ˜ Θ E (log π DNN ( Y | X , ϑ )) denotes the true parameters of the DNN model as specified in (A17), and ˆ ϑ n = arg max ϑ ∈ ˜ Θ { 1 n ∑ n i =1 log π ( Y ( i ) , Y ( i ) mis | X ( i ) , ϑ ) } denotes the maximum likelihood estimator of the StoNet model (A16) with the pseudo-complete data.

Lemma A4.3 implies that the StoNet and DNN are asymptotically equivalent as the training sample size n becomes large, and it forms the basis for the bridging the StoNet and the DNN. The asymptotic equivalence can be elaborated from two perspectives. First, suppose the DNN model (A17) is true. Lemma A4.3 implies that when n becomes large, the weights of the DNN can be learned by training a StoNet of the same structure with σ 2 i 's satisfying Assumption A4.1-(v). On the other hand, suppose that the StoNet (A16) is true, and then Lemma A4.3 implies that for any StoNet satisfying Assumptions A4.1 &amp; A4.2, the weights ϑ can be learned by training a DNN of the same structure when the training sample size is large.

## A4.2 Justification for Condition (12)-(ii)

To justify this condition, we first introduce the following lemma:

Lemma A4.4. Consider a random matrix M ∈ R n × d with n ≥ d . Suppose that the eigenvalues of M T M are upper bounded, i.e., λ max ( M T M ) ≤ κ max for some constant κ max &gt; 0 . Let Ψ( M ) denote an elementwise transformation of M . Then λ max ( (Ψ( M )) T (Ψ( M )) ) ≤ κ max for the tanh, sigmoid and ReLU transformations.

Proof. For ReLU, the result follows from Lemma 5 of Dittmer et al. (2018). For tanh and sigmoid , since they are Lipschitz continuous with a Lipschitz constant of 1, Lemma 5 of Dittmer et al. (2018) also applies.

Since the connection weights take values in a compact space ˜ Θ , there exists a constant 0 &lt; τ max &lt; ∞ such that

<!-- formula-not-decoded -->

for any l = 1 , 2 , . . . , h +1 , where w l ∈ R d l × d l -1 is the weight matrix of the DNN at layer l .

Let M l ∈ R n × d l denote the output of hidden layer l ∈ { 1 , 2 , . . . , h } of the StoNet; that is,

<!-- formula-not-decoded -->

Consider the case that the activation functions are bounded, such as sigmoid and tanh . Then the matrix E ( M T h M h ) has the eigenvalues upper bounded by nκ max for some constant 0 &lt; κ max &lt; ∞ .

For the case that the activation functions are unbounded, such as ReLU or leaky ReLu , we can employ the layer-normalization method in training. In this case, by Lemma A4.4, the matrix Ψ( M h -1 ) T Ψ( M h -1 ) has the eigenvalues upper bounded by nκ max , provided that the eigenvalues of the matrix M T h -1 M h -1 is bounded by nκ max after layer-normalization.

Let ˜ M h = Ψ( M h -1 ) w T h . Then, by an extension of Ostrowski's theorem, see Theorem 3.2 of Higham and Cheng (1998), the eigenvalues of the matrix ˜ M T h ˜ M h = w h (Ψ( M h -1 )) T Ψ( M h -1 ) w T h is bounded by

<!-- formula-not-decoded -->

Also, we have 1 n λ max ( E ( V T V )) = σ 2 v under the assumption (*). Then, with the use of Lemma A4.4 and the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

as the sample size n becomes large. This concludes the proof for condition (12)-(ii).

## A4.3 Proof of Theorem 3.2

Proof. First, we note that σ 2 ˆ θ , the variance of each component of ˆ θ i ( ∈ R p ) , is of the order O ( ϵ/η ) under the setting of the energy function (A6). By setting σ 2 u ≺ ϵ ηhd h p , it is easy to verify that the w -network satisfies Assumption A4.1. Therefore, under the additional regularity conditions given in Assumption A4.2, the proposed stochastic w -network is asymptotically equivalent to the original w -network. Furthermore, the stochastic w -network can be trained by training the original w -network using Algorithm 1.

Next, we prove that the inverse mapping learned through training the stochastic w -network is consistent. For Lemma 3.1, Liang et al. (2025) first established the convergence of the weights to w ∗ n , and subsequently established the weak convergence of the latent variables. Therefore, due to the Markov chain nature of Algorithm 1, it suffices to prove that the inverse mapping produced by the StoNet is consistent, assuming the latent variables have been correctly imputed (i.e., the true values of Z n are known). Once the inverse mapping's consistency is established, the latent variables will be correctly imputed in the next iteration, owing to the algorithm's equation-solving nature, which is ensured by setting ϵ → 0 upon convergence to the zero-energy region. This ensures that algorithm remains in its equilibrium, enabling accurate inference of model uncertainty.

Under the StoNet setting, estimation of w (2) n involves solving p low-dimensional regressions. In particular, given M = ( m 1 , m 2 , . . . , m n ) T ∈ R n × d l , solving each of the regressions contributes a parameter estimation error:

<!-- formula-not-decoded -->

where ξ ∗ i denotes the i th column of w ∗ (2) n , i.e., w ∗ (2) n := ( ξ ∗ 1 , . . . , ξ ∗ p ) T ∈ R p × d h ; ˆ ξ m j = ( M T M ) -1 M T ˆ θ ( j ) is the OLS estimator for the regression coefficients; and ˆ θ ( j ) = ( ˆ θ ( j ) 1 , . . . , ˆ θ ( j ) n ) T .

Let ˆ w m (2) = ( ˆ ξ m 1 , . . . , ˆ ξ m p ) T ∈ R p × d h . By a fundamental property of linear regression, the mean prediction ˆ θ ∗ i = ˆ w m (2) m i is consistent with respect to θ ∗ = E ( ˆ θ i ) , provided that d h ≺ n , the m i 's extract all ˆ θ -relevant information from the input variables, and the w -network has sufficient capacity to establish the linear relationship ˆ θ i ∼ m i for i = 1 , 2 , . . . , n . The equality θ ∗ = E ( ˆ θ i ) is guaranteed by the setting of ϵ → 0 and by the construction of the energy function, which approaches to zero if and only if the empirical mean 1 n ∑ n i =1 ˆ θ i converges to θ ∗ and the variance of ˆ θ i approaches to zero. Furthermore, as shown in Liang et al. (2022), the stochastic layer effectively provides a sufficient dimension reduction for the input variables.

Finally, we prove that the inverse mapping obtained by Algorithm 1 during the training of the nonstochastic w -network is also consistent. Let ˜ θ ∗ i = w ∗ (2) Ψ( µ i ) . Let ˜ θ ∗ ( j ) i and ˆ θ ∗ ( j ) i denote the j th element of ˜ θ ∗ i and ˆ θ ∗ i , respectively. From equation (*), we have

<!-- formula-not-decoded -->

where d ˜ Θ denotes the radius of the parameter space ˜ Θ (centered at 0), the second inequality follows from Cauchy-Schwarz inequality, the third inequality follows from the Taylor expansion for Ψ( µ i + v i ) (at the point µ i ), and the last inequality follows from (A20), condition (12)-(ii), and the boundedness of ξ ∗ j as stated in Assumption A4.1-(i).

Substituting σ v ≺ √ ϵ ηhd h p and ignoring some constant factors in (A21), we obtain

<!-- formula-not-decoded -->

where ∥ · ∥ 1 denotes the l 1 -norm of a vector.

By setting ϵ ≺ min { n p 2 d 2 h , h pd h } , we have E ∥ ˆ θ ∗ i -˜ θ ∗ i ∥ 1 = o (1) , which implies ˜ θ ∗ i is also consistent with respect to θ ∗ = E ( ˆ θ i ) . Consequently, the inverse mapping 1 n ∑ n i =1 ˜ θ ∗ i produced by Algorithm 1 in training the non-stochastic w -network is also consistent with respect to θ ∗ . This concludes the proof of the theorem.

## A5 Additional Numerical Results

Regarding uncertainty quantification, we note that there are two types of uncertainties:

1. Aleatoric uncertainty: This refers to the irreducible noise inherent in the data-generating process. It can be modeled as

<!-- formula-not-decoded -->

- where ϵ i ∼ N (0 , σ 2 ) . Estimating the unknown variance σ 2 corresponds to quantifying the aleatoric uncertainty (system random error). This is precisely what we addressed in the last experiment included in our previous rebuttal, where σ 2 = 0 . 1 was treated as unknown.
2. Epistemic uncertainty: This refers to the reducible estimation error due to limited data or incomplete knowledge of the true model (see, for example, model comparison in Section A5.8). In classical statistics, confidence intervals quantify epistemic uncertainty: as the dataset size increases, epistemic uncertainty-and thus the width of the confidence interval-decreases.

In the following section, we demonstrate through different examples that EFI is able to accurately quantity both types of uncertainties. We use the coverage rate as the key metric to quantify epistemic uncertainty. The coverage rate can reach the nominal level only when the parameter estimates are unbiased and the uncertainty estimation is accurate.

## A5.1 1-D Poisson Equation

Figure A3 and Figure A4 provide typical trajectories learned for the 1-D Poisson model (13) using the methods: PINN, Dropout, and Bayesian PINN. For the ablation study, we vary the noise standard deviation from 0 . 01 to 0 . 1 to further assess the validity and accuracy of the proposed method in uncertainty quantification. The results, presented in Tables A1, A2, and A3, show that under different noise levels, the EFI algorithm consistently achieves a 95% coverage rate for the 95% confidence intervals.

To further investigate the relationship between confidence intervals and epistemic uncertainty, we conducted an additional experiment using two different sample sizes: 20 and 80. The results, presented in Table A4, clearly show that as the sample size increases, the width of the confidence interval decreases while the coverage rate remains consistent. This demonstrates that the confidence interval effectively captures the reducible nature of epistemic uncertainty.

Figure A3: 1-D Poisson model (13): (a) Trajectory learned by PINN (without uncertainties); (b)-(d) Trajectories learned by Dropout with different dropout rates.

<!-- image -->

## A5.2 1-D Poisson Equation with f -measurement error

We revisit the same 1-D Poisson model as defined in (13), but now incorporate measurement errors in both u and f . Specifically, we consider 4 sensors for u and 40 sensors for f , with each sensor

Figure A4: 1-D Poisson model (13): Typical trajectories learned by Bayesian PINN with different σ f values.

<!-- image -->

Table A1: Comparison of different methods for 1D-Poisson with σ = 0 . 01

| Method                                                                                                                                          | MSE                                                                                                                                                             | Coverage Rate                                                                                                                                   | CI-Width                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PINN (no dropout) Dropout (0.5%) Dropout (1%) Dropout (5%) Bayesian ( σ f = 0 . 05 ) Bayesian ( σ f = 0 . 005 ) Bayesian ( σ f = 0 . 0005 ) EFI | 0.000008 (0.000001) 0.000160 (0.000037) 0.000676 (0.000082) 0.009080 (0.001516) 0.000181 (0.000025) 0.000007 (0.000001) 0.000007 (0.000001) 0.000007 (0.000001) | 0.2808 (0.028721) 1.0000 (0.000000) 1.0000 (0.000000) 0.9639 (0.011218) 0.9984 (0.000400) 0.9915 (0.003220) 0.9544 (0.013751) 0.9423 (0.015229) | 0.002079 (0.000026) 0.199798 (0.004135) 0.276534 (0.006642) 0.593541 (0.025812) 0.254861 (0.001554) 0.028386 (0.000201) 0.010437 (0.000046) 0.010759 (0.000117) |

recording 10 replicate measurements. Measurement errors are modeled as z u i ∼ N (0 , 0 . 05 2 ) and z f i ∼ N (0 , 0 . 05 2 ) , and we simulate 100 independent datasets under this setting. The experimental results are summarized in Table A5.

For the Bayesian PINN (B-PINN) method (Yang et al., 2021), we set σ u = σ f = 0 . 05 , ensuring the likelihood function is correctly specified. However, as shown in Table A5, B-PINN produces excessively wide confidence intervals, resulting in an inflated and inaccurate coverage rate. This highlights another issue inherent to Bayesian DNNs, as noted in Liang et al. (2018b); Sun et al. (2022): their performance can be significantly affected by the choice of prior in smalln -largep settings. Here, p refers to the number of parameters in the DNN used to approximate the solution u ( x ) . Similarly, the dropout method continues to produce overly wide confidence intervals and inflated coverage rates.

In contrast, EFI achieves a coverage rate of 94.88% with the smallest confidence interval width. Notably, this experiment involves noise in both u and f observations. To evaluate the imputed errors, the QQ-plot of ˆ z u i and ˆ z f i across 100 experiments is shown in Figure A5. The Q-Q plot confirms that the distribution of imputed errors closely follows its theoretical distribution, supporting the validity of the EFI approach in handling measurement noise from different sources.

Table A2: Comparison of different methods for 1D-Poisson with σ = 0 . 025

| Method                                                                                                                                          | MSE                                                                                                                                                             | Coverage Rate                                                                                                                                   | CI-Width                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PINN (no dropout) Dropout (0.5%) Dropout (1%) Dropout (5%) Bayesian ( σ f = 0 . 05 ) Bayesian ( σ f = 0 . 005 ) Bayesian ( σ f = 0 . 0005 ) EFI | 0.000046 (0.000006) 0.000157 (0.000028) 0.000666 (0.000084) 0.004573 (0.000654) 0.000161 (0.000019) 0.000060 (0.000013) 0.000090 (0.000014) 0.000048 (0.000006) | 0.1529 (0.019175) 1.0000 (0.000000) 1.0000 (0.000000) 0.9979 (0.001157) 0.9992 (0.000307) 0.9630 (0.012690) 0.8504 (0.023270) 0.9577 (0.014487) | 0.002110 (0.000025) 0.195658 (0.003892) 0.267936 (0.005751) 0.643227 (0.031514) 0.257184 (0.001612) 0.037341 (0.000418) 0.024676 (0.000366) 0.027845 (0.000115) |

Table A3: Comparison of different methods for 1D-Poisson with σ = 0 . 1

| Method                                                                                                                                          | MSE                                                                                                                                                             | Coverage Rate                                                                                                                                   | CI-Width                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PINN (no dropout) Dropout (0.5%) Dropout (1%) Dropout (5%) Bayesian ( σ f = 0 . 05 ) Bayesian ( σ f = 0 . 005 ) Bayesian ( σ f = 0 . 0005 ) EFI | 0.000581 (0.000066) 0.000935 (0.000106) 0.001413 (0.000175) 0.006840 (0.001027) 0.000966 (0.000106) 0.000674 (0.000080) 0.004944 (0.000788) 0.000624 (0.000058) | 0.0320 (0.006518) 0.9905 (0.002706) 0.9976 (0.001929) 0.9955 (0.002439) 0.9936 (0.002068) 0.9582 (0.015014) 0.3827 (0.030245) 0.9493 (0.013526) | 0.002088 (0.000024) 0.196226 (0.003748) 0.274188 (0.007653) 0.658607 (0.031770) 0.285660 (0.001405) 0.104743 (0.000325) 0.066296 (0.002515) 0.099543 (0.000775) |

Table A4: Numerical results for 1D-Poisson with σ = 0 . 1 , where the confidence interval width shrinks as sample size increases. The results are computed based on 100 independent simulations.

| Method   |   n b (sample size) | MSE                 | Coverage Rate     | CI-Width            |
|----------|---------------------|---------------------|-------------------|---------------------|
| EFI      |                  20 | 0.000624 (0.000058) | 0.9493 (0.013526) | 0.099543 (0.000775) |
| EFI      |                  80 | 0.000173 (0.000017) | 0.9501 (0.013459) | 0.054256 (0.000723) |

Table A5: Comparison of different methods for the 1-D Poisson model (13) (with f -measurement error), averaged over 100 runs.

| Method         | hidden layers   | MSE                 | Coverage Rate     | CI-Width            |
|----------------|-----------------|---------------------|-------------------|---------------------|
| PINN           | [50, 50]        | 0.000271 (0.000019) | 0.0921 (0.008076) | 0.003625 (0.000085) |
| Dropout (0.5%) | [50, 50]        | 0.000310 (0.000022) | 1.0000 (0.000000) | 0.240585 (0.002282) |
| Dropout (1.0%) | [50, 50]        | 0.000530 (0.000046) | 1.0000 (0.000000) | 0.357782 (0.005141) |
| Dropout (5.0%) | [50, 50]        | 0.003527 (0.000193) | 1.0000 (0.000000) | 0.669799 (0.006158) |
| Bayesian       | [50, 50]        | 0.000233 (0.000017) | 0.9960 (0.002035) | 0.086291 (0.000068) |
| EFI            | [50, 50]        | 0.000238 (0.000017) | 0.9488 (0.009419) | 0.060269 (0.000365) |

## A5.3 Computating Time

Table A6 reports the wall-clock time for the Poisson-1D experiment with different algorithms. As shown by the table, EFI is slower than PINN (with dropout) but much faster than B-PINN (with the HMCsampler). Importantly, we have demonstrated that EFI is the only existing method capable of correctly constructing confidence intervals in a statistically rigorous manner. For larger networks, we can adopt transfer learning techniques by constructing EFI hyper-networks only for the last few layers of the θ -network, which would significantly reduce the computational cost.

## A5.4 Non-linear Poisson Equation

We extend our study to a non-linear Poisson equation given by:

<!-- formula-not-decoded -->

where Ω = [ -0 . 7 , 0 . 7] , β = 0 . 01 , k = 0 . 7 , u = sin 3 (6 x ) , and f can be derived from (A22). For this scenario, we use 4 sensors located at x ∈ {-0 . 7 , -0 . 47 , 0 . 47 , 0 . 7 } to provide noisy observations of the solution u . Additionally, we employ 40 sensors, equally spaced within [ -0 . 7 , 0 . 7] , to measure f . Both u and f measurements are assumed to contain noise. In the simulation, measurement errors are modeled as z u i ∼ N (0 , 0 . 05 2 ) for i = 1 , 2 , . . . , 40 , with each solution sensor providing 10 replicate measurements, and z f i ∼ N (0 , 0 . 05 2 ) for i = 1 , 2 , . . . , 400 .

The experimental results are summarized in Table A7. The findings exhibit a similar pattern to those in Table A5: The Bayesian and dropout methods yield inflated coverage rates and overly wide confidence intervals, whereas the EFI method achieves an accurate coverage rate and the narrowest confidence interval. For a fair comparison, we exclude cases where B-PINN converged to incorrect

<!-- image -->

Figure A5: EFI-PINN diagnostic for the 1-D Poisson model (13) (with f -measurement error)

Table A6: Wall-clock time for PINN, B-PINN and EFI-PINN

| Algorithm      | Hypernetwork   |   Epoch ( × 10 3 ) |   Wall-clock time (s) |   Time per epoch (ms) |
|----------------|----------------|--------------------|-----------------------|-----------------------|
| PINN (dropout) | -              |                200 |                   133 |                 0.665 |
| B-PINN         | -              |                100 |                  1330 |                13.3   |
| EFI            | [16,16,16]     |                200 |                   391 |                 1.955 |
| EFI            | [16,16,4]      |                200 |                   385 |                 1.925 |

solutions, as these represent instances of failure in the optimization process, see Figure A6 for an instance.

Table A7: Comparison of different methods for the nonlinear 1-D Poisson model (A22) (with f -measurement error), averaged over 100 runs.

| Method                      | hidden layers   | MSE                 | Coverage Rate     | CI-Width            |
|-----------------------------|-----------------|---------------------|-------------------|---------------------|
| PINN                        | [50, 50]        | 0.000507 (0.000044) | 0.0947 (0.007564) | 0.005154 (0.000565) |
| Dropout (0.5%)              | [50, 50]        | 0.001050 (0.000123) | 0.9962 (0.002068) | 0.246367 (0.002182) |
| Dropout (1%)                | [50, 50]        | 0.002807 (0.000309) | 0.9861 (0.003829) | 0.358088 (0.005178) |
| Dropout (5%)                | [50, 50]        | 0.007010 (0.000394) | 0.9956 (0.001566) | 0.543565 (0.003096) |
| Bayesian (unstable removed) | [50, 50]        | 0.000376 (0.000033) | 0.9938 (0.002618) | 0.104673 (0.000394) |
| EFI                         | [50, 50]        | 0.000385 (0.000039) | 0.9483 (0.009191) | 0.099880 (0.002853) |

## A5.5 Non-linear Poisson Inverse Problem

In this section, we consider the same non-linear Poisson equation as in (A22), but with k = 0 . 7 treated as an unknown parameter to be estimated. For this setup, we utilize 8 sensors, evenly distributed across Ω = [ -0 . 7 , 0 . 7] to measure u , with measurement noise modeled as z u i ∼ N (0 , 0 . 05 2 ) . Additionally, 200 sensors are employed to measure f , and these measurements are assumed to be noise-free.

To apply EFI framework to inverse problem, we extend the output of the w -network by adding an additional dimension dedicated to estimating k , as depicted in Figure A7. This modification enables the EFI framework to simultaneously estimate the solution u and the parameter k , along with their respective uncertainties. The results are presented in Table A8, demonstrating the capability of EFI to provide accurate uncertainty quantification for both u and k . In contrast, B-PINN consistently produces excessively large confidence intervals for both the solution u and the parameter k. Notably, the confidence interval for k estimated by B-PINN is approximately twice as wide as that produced by EFI, indicating a significant overestimation of uncertainty. This highlights the superior precision and robustness of the EFI framework in inverse problems.

For this problem, we also tested EFI with a larger data modeling network, consisting of two hidden layers and each hidden layer consisting of 100 hidden units. As expected, EFI produced results similar to those obtained with a much smaller data modeling network. As noted earlier, EFI can accurately quantify model uncertainty as long as the random errors are correctly imputed and the

Figure A6: Successful and failed optimization results of Bayesian PINN for the nonlinear 1-D Poisson model (A22) (with f -measurement error)

<!-- image -->

Figure A7: Diagram of EFI for inverse problems, where the orange links indicates the contribution of ¯ θ to both θ -network and k estimation.

<!-- image -->

inverse function is consistently estimated. This capability is independent of the specific configurations of the w -network and the data modeling network, highlighting EFI's flexibility and robustness.

Table A8: Comparison of different methods for the nonlinear 1-D Poisson model (A22) with parameter estimation, averaged over 100 runs.

| Method         | hidden layers   | MSE( × 10 - 4 )   | Coverage Rate   | CI-Width        | k-Mean          | k-Coverage Rate   | k-CI-Width      |
|----------------|-----------------|-------------------|-----------------|-----------------|-----------------|-------------------|-----------------|
| PINN           | [50, 50]        | 1.79 (0.13)       | 0.1464 (0.0097) | 0.0045 (0.0001) | 0.6998 (0.0006) | 0.00 (0.0000)     | 7.9e-5 (5e-6)   |
| Dropout (0.5%) | [50, 50]        | 1.84 (0.10)       | 1.0000 (0.0000) | 0.2393 (0.0035) | 0.6912 (0.0007) | 0.09 (0.0288)     | 0.0045 (0.0002) |
| Dropout (1.0%) | [50, 50]        | 2.72 (0.24)       | 1.0000 (0.0000) | 0.3032 (0.0071) | 0.6874 (0.0008) | 0.11 (0.0314)     | 0.0091 (0.0007) |
| Dropout (5.0%) | [50, 50]        | 27.08 (2.12)      | 1.0000 (0.0000) | 0.5363 (0.0131) | 0.6493 (0.0023) | 0.06 (0.0239)     | 0.0252 (0.0014) |
| Bayesian       | [50, 50]        | 1.31 (0.08)       | 0.9752 (0.0055) | 0.0517 (3e-5)   | 0.6994 (0.0005) | 1.00 (0.0000)     | 0.0411 (0.0002) |
| EFI            | [50, 50]        | 1.03 (0.08)       | 0.9473 (0.0099) | 0.0396 (0.0002) | 0.6985 (0.0004) | 0.94 (0.0239)     | 0.0179 (0.0002) |
| EFI            | [100, 100]      | 0.98 (0.07)       | 0.9560 (0.0099) | 0.0395 (0.0002) | 0.6995 (0.0004) | 0.96 (0.0197)     | 0.0168 (0.0002) |

## A5.6 Poisson equation with unknown noise standard deviation

We now consider the case where the noise standard deviation is treated as an unknown parameter. In this setting, the variability in the observations due to the inherent randomness of the data-generating process-often referred to as systematic error-can be interpreted as aleatoric uncertainty.

As shown in Table A9, the EFI algorithm successfully recovers accurate estimates for both the solution u and the noise standard deviation σ , accompanied by well-calibrated confidence intervals.

Table A9: Numerical results for 1D-Poisson with unknown σ = 0 . 1 and sample size n b = 60 , where the number in the parentheses represents the standard error of the estimator.

| Method   | MSE                 | Coverage Rate     | CI-Width            | σ -mean             | σ -CR             | σ -CI-Width         |
|----------|---------------------|-------------------|---------------------|---------------------|-------------------|---------------------|
| EFI      | 0.000220 (0.000025) | 0.9559 (0.014349) | 0.061347 (0.000848) | 0.097269 (0.001158) | 0.9400 (0.023868) | 0.056768 (0.001284) |

## A5.7 Black-Scholes Model

As a practical example, we consider the classical option pricing model in finance - the Black-Scholes model (Black and Scholes, 1973):

<!-- formula-not-decoded -->

which describes the price V ( S, t ) of an option. Here, S is the price of the underlying asset (e.g., a stock), t is time, σ represents the volatility of the asset, r is the risk-free interest rate, K is the strike price, and T is the expiration time of the option. The boundary conditions reflect specific financial constraints.

This model has been widely used to calculate the price of European call and put options. Specifically, the analytic solution for the call option price C ( S t , t ) is given by

<!-- formula-not-decoded -->

where Φ( · ) denotes the standard normal cumulative distribution function. However, the uncertainty of the model has not yet been well studied in the literature. Accurately quantifying model uncertainty can significantly benefit decision-making, providing investors with a scientific foundation for making safer and more informed choices.

In this simulation experiment, we set T = 1 , σ = 0 . 5 , r = 0 . 05 and K = 1 . The domain is defined on Ω = [0 , T ] × [0 , S max ] , where S max = 2 . We assume the availability of 5 sensors at t = 0 for the price levels S ∈ { 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0 } , with each sensor providing 10 replicate measurements. Measurement errors are modeled as z u i ∼ N (0 , 0 . 05 2 ) for i = 1 , . . . , 50 , representing noisy observations. For the boundaries at { S = 0 } and { t = T } , we use 50 sensors with noise-free measurements. For physical domain, we randomly pick 800 points from Ω to satisfy the BlackScholes equation. The results of the simulation are presented in Table A10, where the metrics are evaluated at t = 0 and t = 0 . 5 . At t = 0 , the evaluation reflects the model's performance using noisy observed data, while at t = 0 . 5 , the solutions are extended from the boundaries using the Black-Scholes equation. This setup highlights the model's ability to handle noisy observations and accurately propagate solutions over time through the governing equation. EFI demonstrates superior performance by providing not only the most accurate solutions, as evidenced by the lowest MSE, but also the most reliable confidence intervals.

Table A10: Comparison of different methods for the Black-Scholes Model, averaged over 100 runs: 'CR' refers to the coverage rate with a nominal value of 95%.

| Method                     | hidden layers   | MSE( t =0)( × 10 - 4 )   | CR( t =0)       | CI-Width( t =0)   | MSE( t =0.5) ( × 10 - 4   | CR( t =0.5)     | CI-Width( t =0.5)   |
|----------------------------|-----------------|--------------------------|-----------------|-------------------|---------------------------|-----------------|---------------------|
| PINN                       | [50, 50]        | 3.08 (0.44)              | 0.1410 (0.0102) | 0.0046 (0.0002)   | 15.73 (2.26)              | 0.2427 (0.0192) | 0.0080 (0.0006)     |
| Dropout (0.5%)             | [50, 50]        | 1.37 (0.20)              | 0.5897 (0.0216) | 0.0190 (0.0002)   | 2.19 (0.37)               | 0.6303 (0.0252) | 0.0197 (0.0004)     |
| Dropout (1.0%)             | [50, 50]        | 4.30 (2.59)              | 0.6743 (0.0219) | 0.0244 (0.0005)   | 1.83 (0.26)               | 0.6983 (0.0255) | 0.0234 (0.0004)     |
| Dropout (5.0%)             | [50, 50]        | 1.71 (0.70)              | 0.9137 (0.0122) | 0.0538 (0.0004)   | 1.70 (0.21)               | 0.9387 (0.0110) | 0.0510 (0.0002)     |
| Bayesian ( σ f = 0 . 05 )  | [50, 50]        | 1.59 (0.41)              | 0.9637 (0.0175) | 0.0516 (0.0010)   | 12.61 (7.14)              | 0.9413 (0.0187) | 0.0658 (0.0015)     |
| Bayesian ( σ f = 0 . 005 ) | [50, 50]        | 10.75 (1.46)             | 0.5437 (0.0255) | 0.0388 (0.0007)   | 39.39 (8.82)              | 0.4807 (0.0225) | 0.0426 (0.0010)     |
| EFI                        | [50, 50]        | 0.38 (0.05)              | 0.9440 (0.0133) | 0.0158 (0.0001)   | 0.17 (0.02)               | 0.9600 (0.0082) | 0.0123 (0.0001)     |

To further illustrate these findings, we visualize the prediction surface in Figure A8. The figure reveals that B-PINN extends the solution poorly toward the edge at S = 2 , where no data points are available,

relying solely on physical laws for extrapolation. In contrast, EFI provides a smoother and more accurate extension. The dropout method performs reasonably well for this example with a dropout rate of 5%; however, its confidence interval remains significantly wider than that of EFI. As previously noted, determining an appropriate dropout rate is not feasible without additional information. Figure A9 highlights EFI's ability to correctly quantify uncertainties. Near the boundary at S = 0 , where boundary information is available, the confidence interval is appropriately narrow. As the stock price S increases, and boundary information becomes scarce, the confidence interval widens, reflecting the growing uncertainty. In comparison, dropout and Bayesian methods fail to capture this behavior accurately. They produce overly broad or inconsistent intervals, particularly near the boundaries and regions with limited data, underscoring their limitations in handling uncertainty quantification for this problem.

Figure A8: European Call Option Price.

<!-- image -->

Figure A9: European Call Option Price at t = 0 .

<!-- image -->

## A5.8 Real Data: Montroll Growth Model

Consider the Montroll growth model:

<!-- formula-not-decoded -->

where k , C , and θ are unknown parameters. We applied this model to published data on the growth of Chinese hamster V79 fibroblast tumor cells (Marusic et al., 1994), which also appears in Rodrigues (2024). The dataset comprises 45 measurements of tumor volumes ( 10 9 vm 3 ) collected over a 60-day period. Table A11 shows the parameter estimates obtained using PINN, and Figure A10(a) shows the learned growth curve.

Table A11: Parameter estimates obtained with PINN for the model (A25).

| Parameter   |   Estimated Value |
|-------------|-------------------|
| k           |            0.8311 |
| C           |            7.3327 |
| θ           |            0.1694 |

b

Figure A10: Montroll growth Model for Chinese hamster V79 fibroblast tumor cells.

<!-- image -->

However, the standard PINN method does not provide confidence intervals for p ( t ) or for the parameters k , C , and θ . To apply EFI-PINN to this dataset, we assume a heteroscedastic noise structure given by σ t = √ t · σ , where σ is treated as an additional unknown parameter. Thus, we estimate four parameters in total under the EFI framework. Using EFI-PINN, we detected that k and θ are nearly non-identifiable, see Figure A10(b), which shows their joint distribution by plotting their samples throughout training.

To address this non-identifiability issue, we fix θ = 0 . 1 and reduce the model to:

<!-- formula-not-decoded -->

The corresponding confidence intervals for p ( t ) are shown in the left plot of Figure 3. The confidence intervals of k , C , and σ are given in Table A12.

Table A12: Parameter estimates (with 95% confidence intervals) obtained by EFI-PINN for the Montroll growth model.

| Parameter   | 95% Confidence Interval   |   Mean |
|-------------|---------------------------|--------|
| θ           | -                         |   0.1  |
| k           | (1.25, 1.40)              |   1.25 |
| C           | (7.30, 7.69)              |   7.44 |
| σ           | (0.27, 0.47)              |   0.36 |

The Montroll experiment highlights the strength of EFI in quantifying uncertainty for all parameters of interest. Moreover, it demonstrates EFI's ability to detect model identifiability issues, underscoring its utility in the statistical inference of scientific models. Additionally, EFI produces an estimate of σ , which enables the quantification of predictive uncertainty.

## A5.9 Real Data: FKPP Model and Porous-FKPP Model

Consider the Fisher-Kolmogorov-Petrovsky-Piskunov (FKPP) model and the porous FKPP (PFKPP) model, which are governed by the following reaction-diffusion equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D , r , and m are unknown parameters, and K denotes the carrying capacity. This equation has been used to model a wide range of growth and transport of biological processes. We applied it to scratch assay data (Jin and Cai, 2006). The biological experiments were conducted under varying initial cell densities - specifically, 10,000, 12,000, 14,000, 16,000, 18,000, and 20,000 cells per well. Cell densities were recorded at 37 equally-spaced spatial positions across five equally-spaced time points - specifically, 0.0 days, 0.5 days, 1.0 days, 1.5 days, and 2.0 days. See also Lagergren et al. (2020) for additional descriptions of the dataset. In addition to the dataset, we partitioned the space-time domain [0 , 2] × [0 , 2] , where the first interval corresponds to the spatial domain and the second to the temporal domain, into a 50 × 10 grid for computing the PDE loss (i.e., the f -term in equation (10)). The fitting curves of the EFI algorithm for different models and initial density values are shown in Figures A11 and A12. Notably, beyond point estimation, EFI also constructs confidence intervals. Additionally, we note that the right plot of Figure 3 was generated using a parameter setting different from that listed in Table A20. In this setting, fewer sample points were allocated for computing the PDE loss, which resulted in smaller fitting errors but larger deviations from the assumed PDE model. Essentially, the two settings correspond to different datasets, as the number of sample points used to evaluate the energy function differs between them.

In Table A13, the root mean squared errors (RMSEs) are computed between the predicted solution u and the observed data, reflecting a combination of epistemic and aleatoric uncertainty. Based on the model formulations in (A26) and (A27), the P-FKPP model is more flexible and is therefore expected to exhibit reduced epistemic uncertainty, leading to smaller RMSE values. Consistent with this expectation, Table A13 shows that the P-FKPP model achieves lower RMSEs compared to the standard FKPP model.

Regarding parameter uncertainty, we note that EFI is able to quantify the uncertainty associated each parameter. However, due to the transformation applied to the first term of (A27), the values of D are no longer on the same scale across the two models, whereas the values of r remain comparable in scale. The results are reported in Table A13.

Summary Through both simulation and real data experiments, we have demonstrated that the proposed EFI algorithm effectively quantifies uncertainties associated with the model and the datagenerating process, resulting in accurate estimation of both epistemic and aleatoric uncertainties.

Table A13: RMSE and estimated parameters with 95% confidence intervals for FKPP and PorousFKPP models.

| Model   |   Initial Cell Density |   RMSE |          D | Interval           |     R | Interval       | M     | Interval       |
|---------|------------------------|--------|------------|--------------------|-------|----------------|-------|----------------|
| FKPP    |                  10000 |  58.04 |    0.00936 | (0.00754, 0.01195) | 0.829 | (0.797, 0.877) | -     | -              |
| FKPP    |                  12000 |  82.09 |    0.00378 | (0.00281, 0.00461) | 0.632 | (0.603, 0.658) | -     | -              |
| FKPP    |                  14000 |  82.93 |    0.02929 | (0.02739, 0.03268) | 0.534 | (0.505, 0.585) | -     | -              |
| FKPP    |                  16000 |  99.14 |    0.02636 | (0.02503, 0.02789) | 0.608 | (0.585, 0.633) | -     | -              |
| FKPP    |                  18000 | 115.27 |    0.03784 | (0.03541, 0.04032) | 0.549 | (0.524, 0.575) | -     | -              |
| FKPP    |                  20000 | 136.67 |    0.05471 | (0.05007, 0.05817) | 0.492 | (0.458, 0.520) | -     | -              |
| P-FKPP  |                  10000 |  46.9  | 1167.67    | (72.48, 2719.97)   | 0.846 | (0.832, 0.856) | 1.335 | (1.037, 1.490) |
| P-FKPP  |                  12000 |  67.88 | 1825.54    | (32.84, 4416.53)   | 0.674 | (0.649, 0.696) | 1.433 | (1.033, 1.603) |
| P-FKPP  |                  14000 |  73.59 |  289.37    | (79.08, 552.13)    | 0.625 | (0.600, 0.649) | 1.096 | (0.951, 1.199) |
| P-FKPP  |                  16000 |  70.83 |   57.36    | (19.21, 97.22)     | 0.628 | (0.608, 0.650) | 0.920 | (0.804, 0.999) |
| P-FKPP  |                  18000 |  96.5  |   21.58    | (9.07, 35.78)      | 0.563 | (0.536, 0.587) | 0.780 | (0.683, 0.863) |
| P-FKPP  |                  20000 | 123.34 |    1.472   | (1.058,2.020)      | 0.496 | (0.464, 0.530) | 0.408 | (0.370, 0.452) |

Figure A11: FKPP model

<!-- image -->

## A6 Experimental Settings

In all the simulation experiments, we begin by generating data from a specific physics model. Using the simulated data, we iteratively run the algorithm to estimate the model parameters. To enhance convergence of Algorithm 1, some algorithmic parameters (such as learning rate, SGD momentum, λ = 1 /ϵ , and etc.) are adjusted during the initial iterations, referred to as the annealing period. To tune different parameters, we use three different annealing schemes, including linear, exponential and polynomial. Their specific forms are in Table A14. For the nonlinear Poisson inverse problem, both the DNN parameters ϑ and the unknown parameter k are estimated; while for all other problems, only the DNN parameters ϑ are estimated. At the end of the simulation, the samples collected in the burn-in period are discarded, and the samples collected in the remaining iterations are used for inference. The burn-in period is set to be at least as long as the annealing period across all experiments.

In our simulations, to ensure the weights of the w -network remain within a compact space as required in Assumption A4.1-(i), we impose a Gaussian prior, N(0,100), on each connection weight. However, due to the large variance, this prior has minimal impact on the algorithm's performance, serving primarily to ensure its stability.

Figure A12: Porous-FKPP model

<!-- image -->

Table A14: Notations

| Notation                                                                                                                                                    | Meaning                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| epochs burn-in period annealing period linear_x_y (with progress ρ ∈ [0 , 1] ) exp_x_y (with progress ρ ∈ [0 , 1] ) poly_x_c_k (with progress ρ ∈ [0 , 1] ) | total number of sampling/optimization iterations the proportion of total iterations allocated to the burn-in process the proportion of total iterations allocated for parameter adjustment to enhance convergence x +( y - x ) ρ x 1 - ρ y ρ x 1+( cρ ) k |

Table A15: Parameter settings for 1D-Poisson

| Parameter Name               | PINN      | Dropout    | Bayesian PINN                       | EFI                  |
|------------------------------|-----------|------------|-------------------------------------|----------------------|
| t start                      | -0.7      | -0.7       | -0.7                                | -0.7                 |
| t end                        | 0.7       | 0.7        | 0.7                                 | 0.7                  |
| noise sd in u                | 0.05      | 0.05       | 0.05                                | 0.05                 |
| noise sd in f                | 0.0       | 0.0        | 0.0                                 | 0.0                  |
| # of solution sensors        | 2         | 2          | 2                                   | 2                    |
| # of solution replicates     | 10        | 10         | 10                                  | 10                   |
| # of differential sensors    | 200       | 200        | 200                                 | 200                  |
| # of differential replicates | 1         | 1          | 1                                   | 1                    |
| epochs                       | 500000    | 500000     | 100000                              | 200000               |
| burn-in period               | 0.5       | 0.5        | 0.4                                 | 0.1                  |
| hidden layers                | [50 , 50] | [50 , 50]  | [50 , 50]                           | [50 , 50]            |
| activation function          | tanh      | tanh       | tanh                                | tanh                 |
| learning rate                | 3e-4      | 3e-4       | 1e-4/1e-4/linear_1e-4_3e-6          | poly_5e-6_100.0_0.55 |
| η f                          | 1.0       | 1.0        | /                                   | 1.0                  |
| dropout rate                 | /         | 0.5%/1%/5% | /                                   | /                    |
| L                            | /         | /          | 6                                   | /                    |
| σ f                          | /         | /          | 0.05/exp_0.05_0.005/exp_0.05_0.0005 | /                    |
| σ u                          | /         | /          | 0.05                                | /                    |
| annealing period             | /         | /          | 0.3                                 | 0.1                  |
| sgd momentum                 | /         | /          | /                                   | linear_0.9_0.0       |
| sgld learning rate           | /         | /          | /                                   | poly_5e-6_10.0_0.55  |
| λ                            | /         | /          | /                                   | linear_50.0_500.0    |
| η θ                          | /         | /          | /                                   | 1.0                  |
| encoder hidden layers        | /         | /          | /                                   | [16 , 16 , 16]       |
| encoder activation function  | /         | /          | /                                   | leaky relu           |

Table A16: Parameter settings for 1D-Poisson (with f error)

| Parameter Name               | PINN      | Dropout    | Bayesian PINN   | EFI                   |
|------------------------------|-----------|------------|-----------------|-----------------------|
| t start                      | -0.7      | -0.7       | -0.7            | -0.7                  |
| t end                        | 0.7       | 0.7        | 0.7             | 0.7                   |
| noise sd in u                | 0.05      | 0.05       | 0.05            | 0.05                  |
| noise sd in f                | 0.05      | 0.05       | 0.05            | 0.05                  |
| # of solution sensors        | 4         | 4          | 4               | 4                     |
| # of solution replicates     | 10        | 10         | 10              | 10                    |
| # of differential sensors    | 40        | 40         | 40              | 40                    |
| # of differential replicates | 10        | 10         | 10              | 10                    |
| epochs                       | 100000    | 100000     | 50000           | 200000                |
| burn-in period               | 0.5       | 0.5        | 0.4             | 0.1                   |
| hidden layers                | [50 , 50] | [50 , 50]  | [50 , 50]       | [50 , 50]             |
| activation function          | tanh      | tanh       | tanh            | tanh                  |
| learning rate                | 3e-4      | 3e-4       | 1e-4            | poly_2.5e-6_50.0_0.55 |
| η f                          | 1.0       | 1.0        | /               | 1.0                   |
| dropout rate                 | /         | 0.5%/1%/5% | /               | /                     |
| L                            | /         | /          | 6               | /                     |
| σ f                          | /         | /          | 0.05            | /                     |
| σ u                          | /         | /          | 0.05            | /                     |
| annealing period             | /         | /          | /               | 0.1                   |
| sgd momentum                 | /         | /          | /               | linear_0.9_0.0        |
| sgld learning rate           | /         | /          | /               | poly_5e-6_100.0_0.55  |
| λ                            | /         | /          | /               | linear_50.0_1000.0    |
| η θ                          | /         | /          | /               | 1.0                   |
| encoder hidden layers        | /         | /          | /               | [64 , 64 , 16]        |
| encoder activation function  | /         | /          | /               | leaky relu            |

Table A17: Parameter settings for nonlinear 1D-Poisson (with f error)

| Parameter Name               | PINN      | Dropout    | Bayesian PINN   | EFI                  |
|------------------------------|-----------|------------|-----------------|----------------------|
| t start                      | -0.7      | -0.7       | -0.7            | -0.7                 |
| t end                        | 0.7       | 0.7        | 0.7             | 0.7                  |
| noise sd in u                | 0.05      | 0.05       | 0.05            | 0.05                 |
| noise sd in f                | 0.05      | 0.05       | 0.05            | 0.05                 |
| # of solution sensors        | 4         | 4          | 4               | 4                    |
| # of solution replicates     | 10        | 10         | 10              | 10                   |
| # of differential sensors    | 40        | 40         | 40              | 40                   |
| # of differential replicates | 10        | 10         | 10              | 10                   |
| k                            | 0.7       | 0.7        | 0.7             | 0.7                  |
| epochs                       | 100000    | 100000     | 100000          | 200000               |
| burn-in period               | 0.5       | 0.5        | 0.4             | 0.1                  |
| hidden layers                | [50 , 50] | [50 , 50]  | [50 , 50]       | [50 , 50]            |
| activation function          | tanh      | tanh       | tanh            | tanh                 |
| learning rate                | 3e-4      | 3e-4       | 1e-4            | poly_5e-6_100.0_0.55 |
| η f                          | 1.0       | 1.0        | /               | 1.0                  |
| dropout rate                 | /         | 0.5%/1%/5% | /               | /                    |
| L                            | /         | /          | 6               | /                    |
| σ f                          | /         | /          | exp_0.2_0.05    | /                    |
| σ u                          | /         | /          | 0.05            | /                    |
| annealing period             | /         | /          | 0.3             | 0.1                  |
| sgd momentum                 | /         | /          | /               | linear_0.9_0.0       |
| sgld learning rate           | /         | /          | /               | poly_5e-6_100.0_0.55 |
| λ                            | /         | /          | /               | exp_50.0_1000.0      |
| η θ                          | /         | /          | /               | 1.0                  |
| encoder hidden layers        | /         | /          | /               | [64 , 64 , 16]       |
| encoder activation function  | /         | /          | /               | leaky relu           |

Table A18: Parameter settings for nonlinear 1D-Poisson with parameter estimation

| Parameter Name               | PINN      | Dropout    | Bayesian PINN   | EFI                  |
|------------------------------|-----------|------------|-----------------|----------------------|
| t start                      | -0.7      | -0.7       | -0.7            | -0.7                 |
| t end                        | 0.7       | 0.7        | 0.7             | 0.7                  |
| noise sd in u                | 0.05      | 0.05       | 0.05            | 0.05                 |
| noise sd in f                | 0.0       | 0.0        | 0.0             | 0.0                  |
| # of solution sensors        | 8         | 8          | 8               | 8                    |
| # of solution replicates     | 10        | 10         | 10              | 10                   |
| # of differential sensors    | 200       | 200        | 200             | 200                  |
| # of differential replicates | 1         | 1          | 1               | 1                    |
| k                            | 0.7       | 0.7        | 0.7             | 0.7                  |
| epochs                       | 50000     | 500000     | 50000           | 350000               |
| burn-in period               | 0.5       | 0.5        | 0.4             | 0.1                  |
| hidden layers                | [50 , 50] | [50 , 50]  | [50 , 50]       | [50 , 50]            |
| activation function          | tanh      | tanh       | tanh            | tanh                 |
| learning rate                | 3e-4      | 3e-4       | 1e-4            | poly_5e-6_100.0_0.55 |
| η f                          | 1.0       | 1.0        | /               | 1.0                  |
| dropout rate                 | /         | 0.5%/1%/5% | /               | /                    |
| L                            | /         | /          | 10              | /                    |
| σ f                          | /         | /          | 0.05            | /                    |
| σ u                          | /         | /          | 0.05            | /                    |
| annealing period             | /         | /          | /               | 0.1                  |
| sgd momentum                 | /         | /          | /               | linear_0.9_0.0       |
| sgld learning rate           | /         | /          | /               | poly_5e-6_100.0_0.55 |
| λ                            | /         | /          | /               | linear_50.0_1000.0   |
| η θ                          | /         | /          | /               | 1.0                  |
| encoder hidden layers        | /         | /          | /               | [128 , 128 , 12]     |
| encoder activation function  | /         | /          | /               | leaky relu           |

Table A19: Parameter settings for Black-Scholes Model

| Parameter Name                                                                                                                    | PINN                                 | Dropout                        | Bayesian PINN                                           | EFI                                                 |
|-----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|--------------------------------|---------------------------------------------------------|-----------------------------------------------------|
| S range                                                                                                                           | [0 . 0 , 2 . 0]                      | [0 . 0 , 2 . 0] 0]             | [0 . 0 , 2 . 0] [0 . 0 , 1 . 0]                         | [0 . 0 , 2 . 0]                                     |
| t range                                                                                                                           | [0 . 0 , 1 . 0]                      | [0 . 0 , 1 .                   |                                                         | [0 . 0 , 1 . 0]                                     |
| σ                                                                                                                                 | 0.5                                  | 0.5                            | 0.5                                                     | 0.5                                                 |
| r                                                                                                                                 | 0.05                                 | 0.05                           | 0.05                                                    | 0.05                                                |
| K                                                                                                                                 | 1.0                                  | 1.0                            | 1.0                                                     | 1.0                                                 |
| noise sd                                                                                                                          | 0.05                                 | 0.05                           | 0.05                                                    | 0.05                                                |
| # of price sensors                                                                                                                | 5                                    | 5                              | 5                                                       | 5                                                   |
| # of price replicates                                                                                                             | 10                                   | 10                             | 10                                                      | 10                                                  |
| # of boundary samples                                                                                                             | 50                                   | 50                             | 50                                                      | 50                                                  |
| # of differential samples                                                                                                         | 800                                  | 800                            | 800                                                     | 800                                                 |
| epochs burn-in period hidden layers activation function learning rate η f dropout rate L σ f σ u pretrain epochs annealing period | 200000 0.5                           | 200000 0.5 [50 , 50] β 3e-4    | 100000 0.4 [50 , 50] softplus( β = 5 ) / / 6 0.05 5000  | 300000 0.1 [50 , 50] softplus( β = 10 ) 1.0 / / / / |
|                                                                                                                                   | [50 , 50] softplus( β = 5 ) 3e-4 1.0 | softplus( = 5 ) 1.0 0.5%/1%/5% | 1e-4/linear_1e-4_1e-5 0.05/exp_0.05_0.005 0.3 / / / / / | poly_5e-6_100.0_0.55                                |
|                                                                                                                                   | /                                    |                                |                                                         |                                                     |
|                                                                                                                                   | /                                    | /                              |                                                         |                                                     |
|                                                                                                                                   | /                                    | /                              |                                                         |                                                     |
|                                                                                                                                   | /                                    | /                              |                                                         |                                                     |
|                                                                                                                                   | /                                    | /                              |                                                         | /                                                   |
|                                                                                                                                   | /                                    | /                              |                                                         | 0.1                                                 |
| sgd momentum                                                                                                                      | /                                    | /                              |                                                         | linear_0.9_0.0                                      |
| sgld learning rate                                                                                                                | /                                    | /                              |                                                         | poly_5e-6_100.0_0.55                                |
| sgld alpha                                                                                                                        | /                                    | /                              |                                                         | 1.0                                                 |
| λ                                                                                                                                 | /                                    | /                              |                                                         | linear_50.0_1000.0                                  |
| η θ                                                                                                                               | /                                    | /                              |                                                         | 1.0                                                 |
| encoder hidden layers                                                                                                             | /                                    |                                | /                                                       | [64 , 64 , 16]                                      |
| encoder activation function                                                                                                       | /                                    | / /                            | /                                                       | leaky relu                                          |

Table A20: Parameter settings for real data

| Parameter Name                                                                                                                                                               | Montroll growth                                                                                                                          | FKPP                                                                                                                       | P-FKPP                                                                                                                     |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| epochs burn-in period hidden layers activation function learning rate η f sgd momentum sgld learning rate sgld alpha λ η θ encoder hidden layers encoder activation function | 200000 0.1 [50 , 50] softplus( β = 10 ) poly_1e-6_10.0_0.55 1.0 0.9 poly_1e-4_10.0_0.95 1.0 log_50.0_500.0 1.0 [32 , 32 , 16] leaky relu | 200000 0.1 [50 , 50] tanh poly_1e-6_10.0_0.55 1.0 0.9 poly_1e-6_10.0_0.95 1.0 log_50.0_500.0 1.0 [64 , 64 , 16] leaky relu | 200000 0.1 [50 , 50] tanh poly_1e-6_10.0_0.55 1.0 0.9 poly_1e-6_10.0_0.95 1.0 log_50.0_500.0 1.0 [64 , 64 , 16] leaky relu |
## Recurrent Memory for Online Interdomain Gaussian Processes

Wenlong Chen 1 , ∗ , Naoki Kiyohara 1 , 2 , ∗ , Harrison Bo Hua Zhu 3 , 1 , ∗ , Jacob Curran-Sebastian 3 , Samir Bhatt 3 , 1 , Yingzhen Li 1 ,

1 Imperial College London 2 Canon Inc. 3

University of Copenhagen

wenlong.chen21@imperial.ac.uk n.kiyohara23@imperial.ac.uk

harrison.zhu@sund.ku.dk yingzhen.li@imperial.ac.uk

## Abstract

Wepropose a novel online Gaussian process (GP) model that is capable of capturing long-term memory in sequential data in an online learning setting. Our model, Online HiPPO Sparse Variational Gaussian Process (OHSVGP), leverages the HiPPO (High-order Polynomial Projection Operators) framework, which is popularized in the RNN domain due to its long-range memory modeling capabilities. We interpret the HiPPO time-varying orthogonal projections as inducing variables with timedependent orthogonal polynomial basis functions, which allows the SVGP inducing variables to memorize the process history. We show that the HiPPO framework fits naturally into the interdomain GP framework and demonstrate that the kernel matrices can also be updated online in a recurrence form based on the ODE evolution of HiPPO. We evaluate OHSVGP with online prediction for 1D time series, continual learning in discriminative GP model for data with multidimensional inputs, and deep generative modeling with sparse Gaussian process variational autoencoder, showing that it outperforms existing online GP methods in terms of predictive performance, long-term memory preservation, and computational efficiency.

## 1 Introduction

Gaussian processes (GPs) are popular choices for modeling time series due to their functional expressiveness and uncertainty quantification abilities [Roberts et al., 2013, Fortuin et al., 2020]. However, GPs are computationally expensive and memory intensive, with cubic and quadratic complexities, respectively. In online regression settings, such as weather modeling, the number of time steps can be very large, quickly making GPs infeasible. Although variational approximations, such as utilizing sparse inducing points (SGPR [Titsias, 2009]; SVGP [Hensman et al., 2013, 2015a]) and Markovian GPs [S¨ arkk¨ a and Solin, 2019, Wilkinson et al., 2021], have been proposed to address the computational complexity, it would still be prohibitive to re-fit the GP model from scratch every time new data arrives. Bui et al. [2017] proposed an online sparse variational GP (OSVGP) learning method that sequentially updates the GP posterior distribution only based on the newly arrived data. However, as indicated in their paper, their models may not maintain the memory of the previous data, as the inducing points will inevitably shift as new data arrive. This is a major drawback, as their models may not model long-term memory unless using a growing number of inducing points.

In deep learning, as an alternative to Transformers [Vaswani et al., 2017], significant works on state space models (SSMs) have been proposed to model long-term memory in sequential data. Originally proposed to instill long-term memory in recurrent neural networks, the HiPPO (High-order Polynomial Projection Operators) framework [Gu et al., 2020] provides mathematical foundations for

*Equal contribution.

Source Code: https://github.com/harrisonzhu508/HIPPOSVGP .

compressing continuous-time signals into memory states through orthogonal polynomial projections. HiPPO is computationally efficient and exhibits strong performance in long-range memory tasks, and forms the basis for the state-of-the-art SSMs, e.g., structured state space sequential (S4) model [Gu et al., 2022] and Mamba [Gu and Dao, 2023, Dao and Gu, 2024].

Inspired by HiPPO, we propose Online HiPPO SVGP (OHSVGP), by applying the HiPPO framework to SVGP in order to leverage the long-range memory modeling capabilities. Our method interprets the HiPPO time-varying orthogonal projections as inducing variables of an interdomain SVGP [L´ azaroGredilla and Figueiras-Vidal, 2009, Leibfried et al., 2020, Van der Wilk et al., 2020], where the basis functions are time-dependent orthogonal polynomials. We show that we are able to significantly resolve the memory-loss issue in OSVGP, thereby opening up the possibility of applying GPs to long-term online learning tasks. In summary, our contributions include:

- (Section 3) We demonstrate that HiPPO integrates into the interdomain GPs by interpreting the HiPPO projections as inducing variables with time-dependent orthogonal polynomial basis functions. This allows the inducing variables to compress historical data, capturing long-term information.
- (Section 3.2 &amp; 5.1) We show that the kernel matrices can leverage the efficient ODE evolution of the HiPPO framework, bringing an extra layer of computational efficiency to OHSVGP.
- (Section 5) We demonstrate OHSVGP on a variety of online/continual learning tasks including time series prediction, continual learning on UCI benchmarks, and continual learning in Gaussian process variational autoencoder, showing that it outperforms other online sparse GP baselines in terms of predictive performance, long-term memory preservation, and computational efficiency.

## 2 Background

In this section, we provide a brief overview of GPs, inducing point methods, online learning with GPs, and Gaussian process variational autoencoders. In addition, we review the HiPPO method, which is the basis of our proposed method.

## 2.1 Gaussian processes

Let X be the input space. For time series data, X = [0 , ∞ ) , the set of non-negative real numbers. A Gaussian process (GP) f ∼ GP (0 , k ) is defined with a covariance function k : X × X → R . It has the property that for any finite set of input points X = [ x 1 , . . . , x n ] ⊺ , the random vector f ≡ f ( X ) = [ f ( x 1 ) , . . . , f ( x n )] ⊺ ∼ N (0 , K ff ) , where K ff is the kernel matrix with entries [ k ( X , X )] ij ≡ [ K ff ] ij = k ( x i , x j ) . For notational convenience and different sets of input points X 1 and X 2 , we denote the kernel matrix as K f 1 f 2 or k ( X 1 , X 2 ) . The computational and memory complexities of obtaining the GP posterior on X 1 scale cubically and quadratically respectively, according to n 1 = | X 1 | . Given responses y and inputs X , a probabilistic model can be defined as y i ∼ p ( y i | f ( x i )) with a GP prior f ∼ GP (0 , k ) , where p ( y i | f ( x i )) is the likelihood distribution. However, for non-conjugate likelihoods, the posterior distribution is intractable, and approximate inference methods are required, such as, but not limited to, variational inference [Titsias, 2009, Hensman et al., 2013, 2015a] and Markov chain Monte Carlo (MCMC) [Hensman et al., 2015b].

## 2.2 Variational inference and interdomain Gaussian processes

To address the intractability and cubic complexity of GPs, Sparse Variational Gaussian Processes (SVGP; [Titsias, 2009, Hensman et al., 2013, 2015b]) cast the problem as an optimization problem. By introducing M inducing points Z ∈ X M that correspond to M inducing variables u = [ f ( z 1 ) , . . . , f ( z M )] ⊺ , the variational distribution q ( f , u ) is defined as q ( f , u ) := p ( f | u ) q θ ( u ) , where q θ ( u ) is the variational distribution of the inducing variables with parameters θ . Then, the evidence lower bound (ELBO) is defined as

<!-- formula-not-decoded -->

where q ( f i ) = ∫ p ( f i | u ) q θ ( u )d u is the posterior distribution of f i ≡ f ( x i ) . Typical choices for the variational distribution are q θ ( u ) = N ( u ; m u , S u ) , where m u and S u are the free-form mean

and covariance of the inducing variables, and yields the posterior distribution:

<!-- formula-not-decoded -->

When the likelihood is conjugate Gaussian, the ELBO can be optimized in closed form and m u and S u can be obtained in closed form (SGPR; Titsias [2009]). In addition to setting the inducing variables as the function values, interdomain GPs [L´ azaro-Gredilla and Figueiras-Vidal, 2009] propose to generalize the inducing variables to u m := ∫ f ( x ) ϕ m ( x )d x , where ϕ m ( x ) are basis functions, to allow for further flexibility. This yields [ K f u ] m = ∫ k ( x, x ′ ) ϕ m ( x ′ )d x ′ and [ K uu ] nm = s k ( x, x ′ ) ϕ n ( x ) ϕ m ( x ′ )d x d x ′ .

We see that the interdomain SVGP bypasses the selection of the inducing points Z ∈ R M , and reformulates it with the selection of the basis functions ϕ i . The basis functions dictate the structure of the kernel matrices, which in turn modulate the function space of the GP approximation. In contrast, SVGP relies on the inducing points Z , which can shift locations according to the training data. Some examples of basis functions include Fourier basis functions [Hensman et al., 2018] and the Dirac delta function δ z m , the latter recovering the standard SVGP inducing variables.

## 2.3 Online Gaussian processes

In this paper, we focus on online learning with GPs, where data arrives sequentially in batches ( X t 1 , y t 1 ) , ( X t 2 , y t 2 ) , . . . etc. For example, in the time series prediction setting, the data arrives in intervals of (0 , t 1 ) , ( t 1 , t 2 ) , . . . etc. The online GP learning problem is to sequentially update the GP posterior distribution as data arrives. Suppose that we have already obtained p t 1 ( y | f ) p t 1 ( f | u t 1 ) q t 1 ( u t 1 ) of the likelihood and variational approximation (with inducing points Z t 1 ), from the first data batch ( X t 1 , y t 1 ) . Online SVGP (OSVGP; [Bui et al., 2017]) utilizes the online learning ELBO

<!-- formula-not-decoded -->

where y i ∈ y t 2 for i = 1 , . . . , n t 2 and ˜ q t 2 ( u t 1 ) := ∫ p t 2 ( u t 1 | u t 2 ) q t 2 ( u t 2 )d u t 2 . Unfortunately, with more and more tasks, OSVGP may not capture the long-term memory in the data since as new data arrives, it is not guaranteed that the inducing points after optimization can sufficiently cover all the previous tasks' data domains.

## 2.4 Gaussian process variational autoencoders

Gaussian processes can be embedded within a variational autoencoder (VAE; [Kingma and Welling, 2014]) framework, giving rise to the Gaussian process variational autoencoder (GPV AE; [Casale et al., 2018, Fortuin et al., 2020, Ashman et al., 2020, Jazbec et al., 2021, Zhu et al., 2023]). For sparse GPs with inducing variables, Jazbec et al. [2021] introduced the SVGPVAE, which combines the sparse variational GP (SVGP) with the VAE formulation. The likelihood p ( y | φ θ ( f )) is parameterized by a decoder network φ θ , which takes GP latent draws f as input, together with the variational inducing posterior q θ ( u | y ) . This posterior, q θ ( u | ϕ ( y )) , is parameterized by the encoder network ϕ . Finally, the latent GP f is typically modeled as a multi-output GP with independent components. GPVAEs have been shown to successfully model high-dimensional time series such as weather data and videos [Zhu et al., 2023, Fortuin et al., 2020]. In this work, we consider the SVGPV AE model defined in Jazbec et al. [2021] for one set of our experiments, and the detailed specification of the model and training objective can be found in Appendix D.2.

## 2.5 HiPPO: recurrent memory with optimal polynomial projections

The HiPPO framework [Gu et al., 2020] provides mathematical foundations for compressing continuous-time signals into finite-dimensional memory states through optimal polynomial projections. Given a time series y ( t ) , HiPPO maintains a memory state c ( t ) ∈ R M that optimally approximates the historical signal { y ( x ) } x ≤ t . The framework consists of a time-dependent measure ω ( t ) ( x ) over ( -∞ , t ] that defines input importance, along with normalized polynomial basis functions { g ( t ) n ( x ) } M -1 n =0 that are orthonormal under ω ( t ) ( x ) , satisfying ∫ t -∞ g ( t ) m ( x ) g ( t ) n ( x ) ω ( t ) ( x )d x =

δ mn . The historical signal is encoded through projection coefficients given by c n ( t ) = ∫ t -∞ y ( x ) g ( t ) n ( x ) ω ( t ) ( x )d x . This yields the approximation y ( x ) ≈ ∑ M -1 n =0 c n ( t ) g n ( x ) for x ∈ ( -∞ , t ] , minimizing the L 2 -error ∫ t -∞ ∥ y ( x ) -∑ n c n ( t ) g n ( x ) ∥ 2 ω ( t ) ( x )d x . Differentiating c ( t ) := [ c 0 ( t ) , . . . , c M -1 ( t )] ⊺ induces a linear ordinary differential equation d d t c ( t ) = A ( t ) c ( t ) + B ( t ) y ( t ) with matrices A ( t ) , B ( t ) encoding measure-basis dynamics. Discretization yields the recurrence of the form c t = A t c t -1 + B t y t enabling online updates. The structured state space sequential (S4) model [Gu et al., 2022] extends HiPPO with trainable parameters and convolutional kernels, while Mamba [Gu and Dao, 2023, Dao and Gu, 2024] introduces hardware-aware selective state mechanisms, both leveraging HiPPO for efficient long-range memory modeling. HiPPO supports various measure-basis configurations [Gu et al., 2020, 2023]. A canonical instantiation, HiPPO-LegS, uses a uniform measure ω ( t ) ( x ) = 1 t 1 [0 ,t ] ( x ) with scaled Legendre polynomials adapted to [0 , t ] , g ( t ) n ( x ) = (2 m +1) 1 / 2 P m ( 2 x t -1 ) . This uniform measure encourages HiPPO-LegS to keep the whole past in memory.

## 3 Interdomain inducing point Gaussian processes with HiPPO

We bridge the HiPPO framework with interdomain Gaussian processes by interpreting HiPPO's state vector defined by time-varying orthogonal projections as interdomain inducing points. This enables adaptive compression of the history of a GP while preserving long-term memory.

## 3.1 HiPPO as interdomain inducing variables

Recall that in an interdomain setting in Section 2.2, inducing variables are defined through an integral transform against a set of basis functions. Let f ∼ GP (0 , k ) , and consider time-dependent basis functions ϕ ( t ) m ( x ) = g ( t ) m ( x ) ω ( t ) ( x ) , where g ( t ) m are the orthogonal functions of HiPPO and ω ( t ) is the associated measure. We define the corresponding interdomain inducing variables as u ( t ) m = ∫ f ( x ) ϕ ( t ) m ( x )d x , which is not a standard random variable as in Section 2.2. Rather, it is a random functions (i.e. stochastic processes) over time ( u ( t ) m ≡ u m ( t ) ) due to time-dependent basis functions. These inducing variables adapt in time, capturing long-range historical information in a compact form via HiPPO's principled polynomial projections.

## 3.2 Adapting the kernel matrices over time

When new observations arrive at later times in a streaming scenario, we must adapt both the prior cross-covariance K fu and the prior covariance of the inducing variables K uu . In particular, the basis functions in our HiPPO construction evolve with time, so the corresponding kernel quantities also require updates. Below, we describe how to compute and update these matrices at a new time t 2 given their values at time t 1 . For clarity, we first discuss K fu , then K uu .

Prior cross-covariance K ( t ) fu . Recall that for a single input x n , the prior cross-covariance with the m -th inducing variable is [ K ( t ) fu ] nm = ∫ k ( x n , x ) ϕ ( t ) m ( x )d x . We can compute the temporal evolution of K ( t ) fu in a manner consistent with the HiPPO approach, leveraging the same parameters A ( t ) and B ( t ) . Specifically,

<!-- formula-not-decoded -->

where [ K ( t ) fu ] n, : is the n -th row of K ( t ) fu . The matrices A ( t ) and B ( t ) depend on the specific choice of the HiPPO measure and basis functions. In our experiments, we employ HiPPO-LegS, whose explicit matrix forms are provided in Appendix A. One then discretizes in t (e.g. using an Euler method or a bilinear transform) to obtain a recurrence update rule.

Prior inducing covariance K ( t ) uu . The ℓm -th element of the prior covariance matrix for the inducing variables is given by [ K ( t ) uu ] ℓm = s k ( x, x ′ ) ϕ ( t ) ℓ ( x ) ϕ ( t ) m ( x ′ )d x d x ′ . Since k ( x, x ′ ) depends on both

Figure 1: Online HiPPO Sparse Variational Gaussian Process (OHSVGP) on a toy time series with 2 tasks. Here x is used to denote arbitrary time index. (a) Time-dependent basis functions with end time index x = t 1 and x = t 2 . (b) Evolution of optimal approximate posterior of inducing variables (mean ± 2 marginal standard deviation). (c) , (d) , (e) illustrate predictive mean ± 2 standard deviation of posterior online GP, OHSVGP and finite basis reconstruction of posterior OHSVGP, respectively.

<!-- image -->

x and x ′ , a recurrence update rule based on the original HiPPO formulation, which is designed for single integral, can not be obtained directly for K ( t ) uu . Fortunately, for stationary kernels, Bochner Theorem [Rudin, 1994] can be applied to factorize the double integrals into two separate single integrals, which gives rise to Random Fourier Features (RFF) approximation [Rahimi and Recht, 2007]: for a stationary kernel k ( x, x ′ ) = k ( | x -x ′ | ) , RFF approximates it as follows: k ( x, x ′ ) ≈ 1 N ∑ N n =1 [cos ( w n x ) cos ( w n x ′ ) + sin ( w n x ) sin ( w n x ′ )] , where w n ∼ p ( w ) is the spectral density of the kernel. Substituting this into the double integral factorizes the dependency on x and x ′ , reducing [ K ( t ) uu ] ℓm to addition of products of one-dimensional integrals. Each integral, with the form of either ∫ cos( w d x ) ϕ ( t ) ℓ ( x )d x or ∫ sin( w d x ) ϕ ( t ) ℓ ( x )d x , again corresponds to a HiPPO-ODE in time. By sampling multiple random features, updating them recurrently to time t , and averaging, we obtain RFF approximation of K ( t ) uu . In addition, more advanced Fourier feature approximation techniques (e.g., [Ton et al., 2018]) can be leveraged for non-stationary kernels. The details of the ODE for recurrent updates of the RFF samples appear in Appendix B.1. Alternatively, one may differentiate K ( t ) uu directly with respect to t . This yields a matrix ODE of the form different from the original HiPPO formulation. For details, see Appendix B.2. Empirically, a vanilla implementation of this approach shows numerical unstability. Hence, we conduct our experiments based on RFF approximation.

Sequential variational updates. Having obtained K ( t 2 ) fu , K ( t 2 ) uu at a new time t 2 &gt; t 1 , we perform variational updates following the online GP framework described in Section 2.3. This ensures the posterior at time t 2 remains consistent with both the new data and the previous posterior at time t 1 , based on K ( t 1 ) fu , K ( t 1 ) uu . Overall, this procedure endows interdomain HiPPO-based GPs with the ability to capture long-term memory online. By viewing the induced kernel transforms as ODEs in time, we efficiently preserve the memory of past observations while adapting our variational posterior in an online fashion. Figure 1b illustrates the evolution of the optimal posterior q ( u ( x ) ) as time x increases on a toy online time series regression problem with two tasks, where x determines the end of the recurrent update for the prior cross and inducing covariance matrices (evolved up to K ( x ) fu and K ( x ) uu , respectively). Furthermore, when x &gt; t 1 , we will update q ( u ( x ) ) online with the two data points from the second task by optimizing the online ELBO (Eq. 3), which gives the discrete jump at x = t 1 . Figure 1d shows the posterior OHSVGP compared with the fit of the gold-standard online GP in Figure 1c. Notably, if f ∼ q t ( f ) , then q ( u ( t ) m ) d = ∫ f ( x ) ϕ ( t ) m ( x )d x (detailed derivation in Appendix C). Therefore, our framework also provides a finite basis approximation of the posterior OHSVGP as a byproduct: f = ∑ M m =1 u ( t ) m g ( t ) m ( x ) , u ( t ) m ∼ q ( u ( t ) m ) . Figure 1e plots the finite basis approximation/reconstruction and it is close to the posterior OHSVGP for this simple example.

## 3.3 Extending OHSVGP to multidimensional input

For multidimensional input data, suppose there is a time order for the first batch of training points with inputs { x (1) n } N 1 n =1 , such that x (1) i appears after x (1) j if i &gt; j , and we further assume x i appears at time index i ∆ t (i.e., x ( i ∆ t ) = x (1) i ), where ∆ t is a user-specified constant step size. In this case, we can again obtain interdomain prior covariance matrices via HiPPO recurrence. For example, a forward Euler method applied to the ODE in Eq. 4 for K t fu yields

<!-- formula-not-decoded -->

The equation above can be viewed as a discretization (with step size ∆ t ) of an ODE solving path integrals of the form ∫ N 1 ∆ t 0 k ( x (1) n , x ( s ) ) ϕ ( t ) m ( s ) ds . The i -th training input x (1) i is assumed to be x (1) i := x ( i ∆ t ) and thus the path integral is approximately solved with discretized recurrence based on the training inputs corresponding to { x ( i ∆ t ) } N i =1 . We continue the recurrence for the second task with ordered training inputs { x (2) n } N 2 n =1 by assigning time index ( N 1 + i )∆ t to its i -th instance. and keep the recurrence until we learn all the tasks continually. In practice, one may use a multiple of ∆ t as the step size to accelerate the recurrence, e.g., instead of using all the training inputs, one can compute the recurrence based on { x 1 , x 3 , x 5 , · · · } only by using step size 2∆ t . When there is no natural time order for training instances in each task, such as in standard continual learning applications, we need to sort the instances with some criterion to create pseudo time order to fit OHSVGP, similar to the practice of applying SSMs to non-sequence data modalities, e.g., SSMs, when applied to vision tasks, assign order to patches in an image for recurrence update of the memory [Zhu et al., 2024]. In our experiments, we show that the performance of OHSVGP, when applied to continual learning, depends on the sorting criterion used.

## 4 Related work

Online sparse GPs. Previous works mainly focus on reducing the sparse approximation error with different approximate inference techniques, such as variational inference [Bui et al., 2017, Maddox et al., 2021], expectation propagation [Csat´ o and Opper, 2002, Bui et al., 2017], Laplace approximation [Maddox et al., 2021], and approximation enhanced with replay buffer [Chang et al., 2023]. The orthogonal research problem of online update of inducing points remains relatively underexplored, and pivoted-Cholesky [Burt et al., 2019] as deployed in Maddox et al. [2021], Chang et al. [2023] is one of the most effective approaches for online update of inducing points up to date. We tackle this problem by taking advantage of the long-term memory capability of HiPPO to design an interdomain inducing variable based method and the associated recurrence based online update rules. Notably, our HiPPO inducing variables in principle are compatible with all the aforementioned approximate inference frameworks since only the way of computing prior covariance matrices will be different from standard online sparse GPs.

Interdomain GPs. To our knowledge, OHSVGP is the first interdomain GP method in the context of online learning. Previous interdomain GPs typically construct inducing variables via integration based on a predefined measure (e.g., a uniform measure over a fixed interval [Hensman et al., 2018] or a fixed Gaussian measure [L´ azaro-Gredilla and Figueiras-Vidal, 2009]) to prevent diverging covariances, and this predefined measure may not cover all regions where the time indices from future tasks are, making them unsuitable for online learning. In contrast, OHSVGP bypasses this limitation by utilizing adaptive basis functions constructed based on time-dependent measure which keeps extending to the new time region as more tasks arrive.

Markovian GPs. Markovian GPs [S¨ arkk¨ a and Solin, 2019, Wilkinson et al., 2021] have similar recurrence structure during inference and training due to their state space SDE representation. However, Markovian GPs are popularized due to their O ( n ) computational complexity and is not explicitly designed for online learning.

## 5 Experiments

Applications &amp; datasets. We evaluate OHSVGP against baselines in the following tasks.

- Time series prediction. We consider regression benchmarks, Solar Irradiance [Lean, 2004], and Audio Signal [Bui and Turner, 2014] produced from the TIMIT database [Garifolo et al., 1993]. We preprocess the two datasets following similar procedures described in Gal and Turner [2015] and Bui et al. [2017], respectively (the train-test split is different due to random splitting). In addition, we consider a daily death-count time series from Santa Catarina State, Southern Brazil spanning the March 2020 to February 2021 COVID-19 pandemic, obtained from Hawryluk et al. [2021]. We construct online learning tasks by splitting each dataset into 10 (5 for COVID) sequential partitions with an equal number of training instances.
- Continual learning. We consider continual learning on two UCI datasets with multi-dim inputs, Skillcraft [Blair et al., 2013] and Powerplant [Tfekci and Kaya, 2014], using the same data preprocessing procedure as in Stanton et al. [2021]. We construct two types of continual learning problems by first sorting the data points based on either the values in their first dimension or their L2 distance from the origin, and then splitting the sorted datasets into 10 sequential tasks with an equal number of training instances.
- High dimensional time series prediction. We evaluate GPVAEs on hourly climate data from ERA5 [Copernicus Climate Change Service, Climate Data Store, 2023, Hersbach et al., 2023], comprising 17 variables across randomly scattered locations around the UK from January 2020 onward. The dataset is split into 10 sequential tasks of 186 hourly time steps each.

Baseline. We compare OHSVGP with OSVGP [Bui et al., 2017] and OVC (Online Variational Conditioning; [Maddox et al., 2021]). At the beginning of each task, OSVGP initialize the induing points by sampling from the old inducing points and the new data points, while OVC initializes them via pivoted-Cholesky [Burt et al., 2019] and we consider both fixing the initialized inducing points as in Chang et al. [2023] (OVC) or keep training them as in Maddox et al. [2021] (OVC-optZ). For time series regression with Gaussian likelihood, we consider OHSGR and OSGPR (OHSVGP and OSVGP based on closed form ELBO), and we further consider OVFF (OSGPR based on variational Fourier feature (VFF), an interdomain inducing point approach from Hensman et al. [2018]).

Hyperparameters. Within each set of experiments, all the models are trained using Adam [Kingma and Ba, 2015] with the same learning rate and number of iterations. For OHSVGP, we construct inducing variables based on HiPPO-LegS [Gu et al., 2020] (see Appendix F.4 for visualizations of using other HiPPO variants) and use 1000 RFF samples. We use ARD-RBF kernel, except for OVFF, tailored specifically to Mat´ ern kernels, where we use Mat´ ern5 2 kernel instead. Similar to Maddox et al. [2021], we do not observe performance gain by keeping updating kernel hyperparameters online, and we include results with trainable kernel hyperparameters in Appendix F.2 for time series regression, but the performance becomes unstable when number of tasks is large. Thus, we either only train the kernel hyperparameters during the initial task and keep them fixed thereafter (Section 5.3) or obtain them from a full GP model trained over the initial task. It is also worth noting that OVFF requires computing covariances as integrals over a predefined interval covering the whole range of the time indices from all tasks (including unobserved ones), which is impractical in real online learning scenarios. For our experiments, we set the two edges of this interval to be the minimum and maximum time index among the data points from all the tasks, respectively.

Evaluations &amp; metrics. We report results in Negative Log Predictive Density (NLPD) in the main text, and Root Mean Squared Error (RMSE) in Appendix F.1 (expected calibration error (ECE; [Guo et al., 2017]) for COVID data instead), which shows consistent conclusions as NLPD. We report the mean and the associated 95% confidence interval obtained from 5 (3 for experiments on ERA5) independent runs.

## 5.1 Online time series prediction

Time series regression. Figure 2 shows NLPD (over the past tasks) of different methods during online learning through the 10 tasks for Solar Irradiance and Audio dataset. Overall, OHSGPR consistently achieves the best performance with OVC performing competitively, especially as we learn more and more tasks, suggesting OHSGPR effectively preserves long-term memory through its HiPPO-based memory mechanism. OSGPR shows catastrophic forgetting starting around task

Figure 2: Test set NLPD over the learned tasks vs. number of learned tasks for Solar Irradiance and Audio signal prediction dataset.

<!-- image -->

Figure 3: Predictive mean ± 2 standard deviation of OSGPR, OVC, OVFF, and OHSGPR after task 10 of the Solar dataset. M = 50 inducing variables are used.

<!-- image -->

5, especially when the number of inducing points M is small. Although OVC-optZ also initializes inducing points with pivoted-Cholesky as OVC, with further optimization, its performance starts to degrade starting from task 6 for the audio dataset when M = 100 , which suggests the online ELBO objective cannot guarantee optimal online update of inducing points that preserve memory. OVFF tends to perform well at the later stage. However, during the first few tasks, it underfits the data significantly compared with other methods since its inducing variables are computed via integration over a predefined interval capturing the region of all the tasks, which is unnecessarily long and suboptimal for learning at the early stage.

In Figure 3, we compare the final predictive distributions for different methods after finishing online learning all 10 tasks of Solar Irradiance. The inducing points Z for OSGPR tend to move to the regions where the later tasks live after online training, and the prediction of OSGPR in the initial regions without sufficient inducing points becomes close to the uninformative prior GP. In contrast, OHSGPR maintains consistent performance across both early and recent time periods.

Infectious disease modeling We replace the Gaussian likelihood with a non-conjugate Negative Binomial likelihood to capture the over-dispersion in COVID-19 death counts. All methods use M ∈ { 15 , 30 } inducing points and are trained for 5000 iterations per task with a learning rate of 0.01. Figure 4 reports the change of NLPD through online learning for the first four out of five tasks. The wide metric variance reflects the noisy nature of death-count data as it is difficult to accurately track down COVID-19 death counts. OHSVGP achieves the best performance overall while OSVGP forgets Task 1 with small M .

Runtime comparison. Table 1 shows the accumulated wall-clock runtime for different methods to learn all the tasks. Unlike OSVGP and OVC-optZ, which must iteratively optimize inducing points (for which we train e.g., 1000 iterations for time-series regression tasks), OHSVGP, OVFF (both based on interdomain inducing points), and OVC (based on one-time pivoted-Cholesky update of inducing points for each task) bypass this cumbersome optimization. In particular,

OHSVGP recurrently evolves K fu and K uu for each new task. For regression problems where closedform posterior can be obtained, OHSGPR requires no training at all. As a result, OHSGPR, OVC and OVFF run significantly faster, adapting to all tasks within a couple of seconds for Solar Irradiance and Audio data. For COVID data, even when free-form variational parameters of

Table 1: Wall-clock accumulated runtime for learning all the tasks on a single NVIDIA RTX3090 GPU in seconds, of different models for time series prediction experiments.

|               | Solar Irradiance   | Solar Irradiance   | Audio Data   | Audio Data   | COVID   | COVID   |
|---------------|--------------------|--------------------|--------------|--------------|---------|---------|
| Method        | M                  | M                  | M            | M            | M       | M       |
|               | 50                 | 150                | 100          | 200          | 15      | 30      |
| OSGPR/OSVGP   | 140                | 149                | 144          | 199          | 525     | 530     |
| OVC           | 0.450              | 0.620              | 0.558        | 0.863        | 345     | 360     |
| OVFF          | 0.327              | 0.354              | 0.295        | 0.356        | -       | -       |
| OHSGPR/OHSVGP | 0.297              | 0.394              | 0.392        | 0.655        | 370     | 380     |

Figure 4: Test set NLPD on COVID dataset right after learning Task i and after learning all the tasks.

<!-- image -->

Figure 5: Test set NLPD after continually learning Task i and after learning all the tasks for i = 1, 2, 4, 8. Tasks are created by splitting Powerplant and Skillcraft datasets with inputs sorted either according to the 1st input dimension or L2 distance to the origin).

<!-- image -->

inducing variables are learned using uncollapsed ELBO, OHSVGP and OVC are still significantly faster than OSVGP since no gradient computation is required for the inducing points.

## 5.2 Continual learning on UCI datasets

We use 256 inducing variables for all methods, and for each task, we train each method for 2000 iterations with a learning rate of 0.005. We only consider OVC here since initial trials show OVC-optZ give worse results on these two datasets. As described in Section 3.3, within each task, OHSVGP requires sorting the data points to compute prior covariance matrices via recurrence. We consider two sorting criteria. The first one, which we call OHSVGP-o, uses the oracle order compatible with how the tasks are created (e.g., sort with L2-distance to the origin if the tasks are initially splitted based on it). In real-world problems, we typically do not have the information on how the distribution shifts from task to task. Hence, we also consider OHSVGP-k, which uses a heuristic sorting method based on kernel similarity: we select the i -th point in task j to be x ( j ) i = arg max x ∈ X ( j ) k ( x , x ( j ) i -1 ) for i &gt; 1 , and the first point in first task is set to be x (1) 1 = arg max x ∈ X (1) k ( x , 0 ) . Figure 5 compares the two variants of OHSVGP with OSVGP and OVC. Overall, OSVGP achieves the worst performance and is again prone to forgetting the older tasks, especially in Figure 5c. OVC performs decently for Skillcraft but it also demonstrates catastrophic forgetting in Figure 5c. While OHSVGP-k achieves similar performance as OSVGP on Skillcraft, OHSVGP-o consistently outperforms the other methods across all 4 scenarios, suggesting the importance of a sensible sorting method when applying OHSVGP for continual learning. Here, we only report the results for Task 1, 2, 4, and 8 for concise presentation, and in Appendix F.1, we include the complete results for all the tasks (the overall conclusion is the same). In Appendix F.3, we further visualize how different sorting methods impact OHSVGP's performance in continual learning with a 2D continual classification problem.

## 5.3 Continual learning for high dimensional time series prediction

All models share a two-layer MLP encoder-decoder, a 20-dimensional latent space, and a multi-output GP with independent components; we use M ∈ { 50 , 100 } and train each task for 20 epochs with learning rate 0.005 on single NVIDIA A6000 GPU. The continual learning in SVGPVAE is achieved

Figure 6: Test set NLPD after continually learning Task i and after learning all the tasks for i = 1, 2, 4, 8, on ERA5 dataset.

<!-- image -->

by further imposing Elastic Weight Consolidation (EWC; [Kirkpatrick et al., 2017]) loss on the encoder and decoder, which yields the vanilla baseline, Online SVGPVAE (OSVGP). Since EWC alone leaves inducing locations non-regularized, a principled online placement rule for the inducing points will improve the model. Thus, we further consider OVC-SVGPVAE (OVC) which adjusts inducing points online via Pivoted-Cholesky, and OVC-SVGPVAE per dimension (OVC-per-dim), which makes OVC more flexible by allocating a separate set of M inducing points to every latent dimension. Our method, Online HiPPO SVGPVAE (OHSVGP), replaces standard inducing points in SVGPVAE with HiPPO inducing variables and updates them online via recurrence. Figure 6 plots the change of NLPD during continual learning for Task 1, 2, 4, and 8 (full results in Appendix F.1). The performance of OHSVGP remains stable throughout, while the other methods all demonstrate obvious catastrophic forgetting shown by the large gaps between performances after learning current task i and after learning final task 10. Two factors plausibly explain the gap: first, standard inducing points cannot adequately cover the long time axis, whereas OHSVGP ties its inducing variables to basis functions rather than time locations; second, the added encoder-decoder complexity makes optimization harder for models that must reuse a limited inducing set. Increasing M narrows the gap but scales at O ( M 3 ) computational and O ( M 2 ) memory cost respectively, underscoring OHSVGP's superior efficiency.

## 6 Conclusion

We introduce OHSVGP, a novel online Gaussian process model that leverages the HiPPO framework for robust long-range memory in online/continual learning. By interpreting HiPPO's time-varying orthogonal projections as adaptive interdomain GP basis functions, we leverage SSM for improved online GP. This connection allows OHSVGP to harness HiPPO's efficient ODE-based recurrent updates while preserving GP-based uncertainty-aware prediction. Empirical results on a suite of online and continual learning tasks show that OHSVGP outperforms existing online sparse GP methods, especially in scenarios requiring long-term memory. Moreover, its recurrence-based covariance updates yield far lower computational overhead than OSVGP's sequential inducing point optimization. This efficient streaming capability and preservation of historical information make OHSVGP well-suited for real-world applications demanding both speed and accuracy.

Broader impact. This paper presents work whose goal is to advance machine learning research. There may exist potential societal consequences of our work, however, none of which we feel must be specifically highlighted here.

## Acknowledgments and Disclosure of Funding

Samir Bhatt acknowledges funding from the MRC Centre for Global Infectious Disease Analysis (reference MR/X020258/1), funded by the UK Medical Research Council (MRC). This UK funded award is carried out in the frame of the Global Health EDCTP3 Joint Undertaking. Samir Bhatt acknowledges support from the Danish National Research Foundation via a chair grant (DNRF160) which also supports Jacob Curran-Sebastian. Samir Bhatt acknowledges support from The Eric and Wendy Schmidt Fund For Strategic Innovation via the Schmidt Polymath Award (G-22-63345) which also supports Harrison Bo Hua Zhu. Samir Bhatt acknowledges support from the Novo Nordisk Foundation via The Novo Nordisk Young Investigator Award (NNF20OC0059309).

## References

- Matthew Ashman, Jonathan So, Will Tebbutt, Vincent Fortuin, Michael Pearce, and Richard E Turner. Sparse Gaussian process variational autoencoders. arXiv preprint arXiv:2010.10177 , 2020.
- Mark Blair, Joe Thompson, Andrew Henrey, and Bill Chen. SkillCraft1 Master Table Dataset. UCI Machine Learning Repository, 2013. DOI: https://doi.org/10.24432/C5161N.
- Thang D. Bui and Richard E. Turner. Tree-structured Gaussian process approximations. In Advances in Neural Information Processing Systems , 2014.
- Thang D. Bui, Cuong V. Nguyen, and Richard E. Turner. Streaming sparse Gaussian process approximations. In Advances in Neural Information Processing Systems , 2017.
- David R. Burt, Carl E. Rasmussen, and Mark van der Wilk. Rates of convergence for sparse variational Gaussian process regression. In International Conference on Machine Learning (ICML) , 2019.
- Francesco Paolo Casale, Adrian Dalca, Luca Saglietti, Jennifer Listgarten, and Nicolo Fusi. Gaussian process prior variational autoencoders. Advances in neural information processing systems , 31, 2018.
- Paul E. Chang, Prakhar Verma, S.T. John, Arno Solin, and Mohammad Emtiyaz Khan. Memory-based dual Gaussian processes for sequential learning. In International Conference on Machine Learning , 2023.
- Copernicus Climate Change Service, Climate Data Store. Era5 hourly data on single levels from 1940 to present, 2023. URL https://doi.org/10.24381/cds.adbb2d47 . Accessed: DD-MMMYYYY.
- Lehel Csat´ o and Manfred Opper. Sparse on-line Gaussian processes. Neural Computation , 14(3): 641-668, 2002.
- Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning (ICML) , 2024.
- Seth Flaxman, Swapnil Mishra, Axel Gandy, H Juliette T Unwin, Thomas A Mellan, Helen Coupland, Charles Whittaker, Harrison Zhu, Tresnia Berah, Jeffrey W Eaton, et al. Estimating the effects of non-pharmaceutical interventions on covid-19 in europe. Nature , 584(7820):257-261, 2020.
- Vincent Fortuin, Dmitry Baranchuk, Gunnar R¨ atsch, and Stephan Mandt. GP-VAE: Deep probabilistic time series imputation. In International Conference on Artificial Intelligence and Statistics , pages 1651-1661. PMLR, 2020.
- Yarin Gal and Richard E. Turner. Improving the Gaussian process sparse spectrum approximation by representing uncertainty in frequency inputs. In International Conference on Machine Learning (ICML) , 2015.
- Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Franc ¸ois Laviolette, Mario March, and Victor Lempitsky. Domain-adversarial training of neural networks. Journal of Machine Learning Research , 17(59):1-35, 2016. URL http://jmlr.org/papers/ v17/15-239.html .
- J. Garifolo, L. Lamel, W. Fisher, J. Fiscus, D. Pallett, N. Dahlgren, and V. Zue. TIMIT acousticphonetic continuous speech corpus LDC93S1. In Philadelphia: Linguistic Data Consortium , 1993.
- Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 , 2023.
- Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher R´ e. HiPPO: Recurrent memory with optimal polynomial projections. In Advances in Neural Information Processing Systems , 2020.
- Albert Gu, Karan Goel, and Christopher R´ e. Efficiently modeling long sequences with structured state spaces. In The International Conference on Learning Representations (ICLR) , 2022.

- Albert Gu, Isys Johnson, Aman Timalsina, Atri Rudra, and Christopher Re. How to train your HIPPO: State space models with generalized orthogonal basis projections. In International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=klK17OQ3KB .
- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning , 2017.
- Iwona Hawryluk, Henrique Hoeltgebaum, Swapnil Mishra, Xenia Miscouridou, Ricardo P Schnekenberg, Charles Whittaker, Michaela Vollmer, Seth Flaxman, Samir Bhatt, and Thomas A Mellan. Gaussian process nowcasting: application to covid-19 mortality reporting. In Uncertainty in Artificial Intelligence , pages 1258-1268. PMLR, 2021.
- James Hensman, Nicol` o Fusi, and Neil D. Lawrence. Gaussian processes for big data. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence , UAI'13, pages 282-290, Arlington, Virginia, USA, 2013. AUAI Press.
- James Hensman, Alexander Matthews, and Zoubin Ghahramani. Scalable variational Gaussian process classification. In Artificial Intelligence and Statistics , pages 351-360. PMLR, 2015a.
- James Hensman, Alexander G Matthews, Maurizio Filippone, and Zoubin Ghahramani. MCMC for variationally sparse Gaussian processes. Advances in Neural Information Processing Systems , 28, 2015b.
- James Hensman, Nicolas Durrande, and Arno Solin. Variational Fourier features for Gaussian processes. Journal of Machine Learning Research , 18(151):1-52, 2018.
- H. Hersbach, B. Bell, P. Berrisford, G. Biavati, A. Hor´ anyi, J. Mu˜ noz Sabater, J. Nicolas, C. Peubey, R. Radu, I. Rozum, D. Schepers, A. Simmons, C. Soci, D. Dee, and J.-N. Th´ epaut. Era5 hourly data on single levels from 1940 to present, 2023. Accessed: DD-MMM-YYYY.
- Roger A. Horn and Charles R. Johnson. Topics in Matrix Analysis . Cambridge University Press, 1991.
- Metod Jazbec, Matt Ashman, Vincent Fortuin, Michael Pearce, Stephan Mandt, and Gunnar R¨ atsch. Scalable Gaussian process variational autoencoders. In International Conference on Artificial Intelligence and Statistics , pages 3511-3519. PMLR, 2021.
- Sanyam Kapoor, Theofanis Karaletsos, and Thang D. Bui. Variational auto-regressive Gaussian processes for continual learning. In International Conference on Machine Learning , 2021.
- D. P. Kingma and M. Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations , 2014.
- Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations , 2015.
- James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences , 114 (13):3521-3526, 2017.
- Judith Lean. Solar irradiance reconstruction. In Data contribution series # 2004-035, IGBP PAGES/World Data Center for Paleoclimatology NOAA/NGDC Paleoclimatology Program, Boulder, CO, USA , 2004.
- Felix Leibfried, Vincent Dutordoir, ST John, and Nicolas Durrande. A tutorial on sparse Gaussian processes and variational inference. arXiv preprint arXiv:2012.13962 , 2020.
- Miguel L´ azaro-Gredilla and Anibal Figueiras-Vidal. Inter-domain Gaussian processes for sparse inference using inducing features. In Advances in Neural Information Processing Systems , 2009.
- Wesley J. Maddox, Samuel Stanton, and Andrew Gordon Wilson. Conditioning sparse variational Gaussian processes for online decision-making. In Advances in Neural Information Processing Systems , 2021.

- M´ elodie Monod, Alexandra Blenkinsop, Xiaoyue Xi, Daniel Hebert, Sivan Bershan, Simon Tietze, Marc Baguelin, Valerie C Bradley, Yu Chen, Helen Coupland, et al. Age groups that sustain resurging covid-19 epidemics in the united states. Science , 371(6536):eabe8372, 2021.
- Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems , 2007.
- Stephen Roberts, Michael Osborne, Mark Ebden, Steven Reece, Neale Gibson, and Suzanne Aigrain. Gaussian processes for time-series modelling. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 371(1984):20110550 , 2013.
- W. Rudin. Fourier analysis on groups. Wiley Classics Library. Wiley-Interscience New York, reprint edition , 1994.
- Simo S¨ arkk¨ a and Arno Solin. Applied stochastic differential equations , volume 10. Cambridge University Press, 2019.
- Samuel Stanton, Wesley J. Maddox, Ian Delbridge, and Andrew Gordon Wilson. Kernel interpolation for scalable online Gaussian processes. In International Conference on Artificial Intelligence and Statistics . PMLR, 2021.
- Pnar Tfekci and Heysem Kaya. Combined Cycle Power Plant. UCI Machine Learning Repository, 2014. DOI: https://doi.org/10.24432/C5002N.
- Michalis Titsias. Variational learning of inducing variables in sparse Gaussian processes. In Artificial intelligence and statistics , pages 567-574. PMLR, 2009.
- Jean-Francois Ton, Seth Flaxman, Dino Sejdinovic, and Samir Bhatt. Spatial mapping with gaussian processes and nonstationary fourier features. Journal of Spatial Statistics , 28:59-78, 2018.
- H Juliette T Unwin, Swapnil Mishra, Valerie C Bradley, Axel Gandy, Thomas A Mellan, Helen Coupland, Jonathan Ish-Horowicz, Michaela AC Vollmer, Charles Whittaker, Sarah L Filippi, et al. State-level tracking of covid-19 in the united states. Nature communications , 11(1):6189, 2020.
- Mark Van der Wilk, Vincent Dutordoir, ST John, Artem Artemev, Vincent Adam, and James Hensman. A framework for interdomain and multioutput Gaussian processes. arXiv preprint arXiv:2003.01115 , 2020.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems , 2017.
- William Wilkinson, Arno Solin, and Vincent Adam. Sparse algorithms for Markovian Gaussian processes. In International Conference on Artificial Intelligence and Statistics , pages 1747-1755. PMLR, 2021.
- Harrison Zhu, Carles Balsells Rodas, and Yingzhen Li. Markovian Gaussian process variational autoencoders. In International Conference on Machine Learning , 2023.
- Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision Mamba: Efficient visual representation learning with bidirectional state space model. In International Conference on Machine Learning (ICML) , 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We accurately describe the paper's contributions and scope, and include the necessary motivations and backgrounds required to understand our contributions. The claims are verified with our experiments.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We discuss the effect of a sub-optimal sorting method when applying our method to continual learning in our experiments (Section 3.3 &amp; 5.2 &amp; Appendix F.3).

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: No theoretical results.

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

Justification: We include the detailed dataset composition and hyperparameter information.

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

Justification: We have released our code at https://github.com/harrisonzhu508/ HIPPOSVGP/tree/main .

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

Justification: We fully describe the training and test details, including data splits, random seeds, hyperparameter tuning, optimizer type and any necessary information needed for reproducibility in our experiments section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We include the error bars of the performance metrics over several random seed runs.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer 'Yes' if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

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

Justification: We indicate which GPUs were used to run the experiments. We also include wallclock time information (Table 1).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: In this paper, we introduce work designed to push the boundaries of machine learning. While our methods could carry ethical implications, we do not believe any require specific discussion at this point in the submission process.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We include a statement for broader impacts at the end of the paper.

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

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Whenever we use any assets, we always cite the original asset source.

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

Answer: [Yes]

Justification: We fully describe our contributions and any assets used in the paper. We will also be releasing code after the publishing of the paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLM was only used for editing the writing, helping with plotting and assisted coding.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A HiPPO-LegS matrices

Here we provide the explicit form of matrices used in our implementation of HiPPO-LegS [Gu et al., 2020]. For a given time t , the measure ω ( t ) ( x ) = 1 t 1 [0 ,t ] ( x ) and basis functions ϕ ( t ) m ( x ) = g ( t ) m ( x ) ω ( t ) ( x ) = √ 2 m +1 t P m ( 2 x t -1 ) 1 [0 ,t ] ( x ) are used, where P m ( · ) is the m -th Legendre polynomial and 1 [0 ,t ] ( x ) is the indicator function on the interval [0 , t ] . These basis functions are orthonormal, i.e.,

<!-- formula-not-decoded -->

Following Gu et al. [2020], the HiPPO-LegS framework maintains a coefficient vector c ( t ) ∈ R M × 1 that evolves according to the ODE:

<!-- formula-not-decoded -->

where f ( t ) is the input signal at time t . The matrices A ( t ) ∈ R M × M and B ( t ) ∈ R M × 1 are given by:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

These matrices govern the evolution of the basis function coefficients over time, where the factor 1 /t reflects the time-dependent scaling of the basis functions to the adaptive interval [0 , t ] . When discretized, this ODE yields the recurrence update used in our implementation.

## B Computing prior covariance of the inducing variables K ( t ) uu

We provide the detailed derivation for the following two approaches when the inducing functions are defined via HiPPO-LegS. Recall that the ℓm -th element of the prior covariance matrix for the inducing variables is given by

<!-- formula-not-decoded -->

where ϕ ( t ) ℓ ( x ) = g ( t ) ℓ ( x ) ω ( t ) ( x ) are the time-varying basis functions under the HiPPO-LegS framework.

## B.1 RFF approximation

Since k ( x, x ′ ) depends on both x and x ′ , a recurrence update rule based on the original HiPPO formulation, which is designed for single integral, can not be obtained directly for K ( t ) uu . Fortunately, for stationary kernels, Bochner Theorem [Rudin, 1994] can be applied to factorize the above double integrals into two separate single integrals, which gives rise to Random Fourier Features (RFF) approximation [Rahimi and Recht, 2007]: for a stationary kernel k ( x, x ′ ) = k ( | x -x ′ | ) , RFF approximates it as follows:

<!-- formula-not-decoded -->

where w n ∼ p ( w ) is the spectral density of the kernel. Substituting this into the double integral (Eq. 10) factorizes the dependency on x and x ′ , reducing [ K ( t ) uu ] ℓm to addition of products of onedimensional integrals. Each integral based on a Monte Carlo sample w has the form of either

<!-- formula-not-decoded -->

which corresponds to a standard projection coefficient in the HiPPO framework. We further stack these integrals based on M basis functions and define

<!-- formula-not-decoded -->

Collecting N Monte Carlo samples { w n } N n =1 , we form the feature matrix

<!-- formula-not-decoded -->

and the RFF approximation of the covariance is

<!-- formula-not-decoded -->

Since Z ( t ) w n and Z ′ ( t ) w n are standard HiPPO projection coefficient, their computation is governed by the HiPPO ODE evolution as before

<!-- formula-not-decoded -->

with h n ( t ) = cos( w n t ) and h ′ n ( t ) = sin( w n t ) and these ODEs can be solved in parallel across different Monte Carlo samples. In summary, the procedure involves sampling multiple random features, updating them recurrently to time t , and averaging across samples to obtain RFF approximation of K uu ( t ) .

For non-stationary kernels, more advanced Fourier feature approximation techniques (e.g., [Ton et al., 2018]) can be applied.

## B.2 Direct ODE evolution

Differentiating [ K ( t ) uu ] ℓ,m with respect to t gives

<!-- formula-not-decoded -->

Applying the product rule:

<!-- formula-not-decoded -->

In HiPPO-LegS, each ϕ ( t ) ℓ ( x ) obeys an ODE governed by lower-order scaled Legendre polynomials on [0 , t ] and a Dirac delta boundary term at x = t . Concretely,

<!-- formula-not-decoded -->

where δ t ( x ) is the Dirac delta at x = t (see Appendix D.3 in Gu et al. [2020] for details).

Substituting this expression into the integrals yields the boundary terms of the form ∫ k ( t, x ′ ) ϕ ( t ) m ( x ′ ) d x ′ , along with lower-order terms involving { [ K ( t ) uu ] ℓ,m , [ K ( t ) uu ] ℓ -1 ,m , · · · } , etc. Summarizing in matrix form leads to

<!-- formula-not-decoded -->

where the lm -th entry of K ( t ) uu ∈ R M × M is [ K ( t ) uu ] ℓ,m , A ( t ) ∈ R M × M is the same lower-triangular matrix from the HiPPO-LegS framework defined in Eq. 8, and ˜ B ( t ) ∈ R M × M is built from the boundary contributions as

<!-- formula-not-decoded -->

where 1 M ∈ R 1 × M is a row vector of ones of size M and c ( t ) ∈ R M × 1 is the coefficient vector with each element being

<!-- formula-not-decoded -->

After discretizing in t (e.g. an Euler scheme), one repeatedly updates K ( t ) uu and the boundary vector c ( t ) over time.

## B.2.1 Efficient computation of ˜ B ( t )

Computing ˜ B ( t ) directly at each time step requires evaluating M integrals, which can be computationally intensive, especially when t changes incrementally and we need to update the matrix ˜ B ( t ) repeatedly.

To overcome this inefficiency, we propose an approach that leverages the HiPPO framework to compute ˜ B ( t ) recursively as s evolves. This method utilizes the properties of stationary kernels and the structure of the Legendre polynomials to enable efficient updates.

Leveraging Stationary Kernels Assuming that the kernel k ( x, t ) is stationary, it depends only on the difference d = | x -t | , so k ( x, t ) = k ( d ) . In our context, since we integrate over x ∈ [ t start , t ] with x ≤ t , we have d = t -x ≥ 0 . Therefore, we can express k ( x, t ) as a function of d over the interval [0 , t -t start ] :

<!-- formula-not-decoded -->

Our goal is to approximate k ( d ) over the interval [0 , t -t start ] using the orthonormal Legendre basis functions scaled to this interval. Specifically, we can represent k ( d ) as

<!-- formula-not-decoded -->

where g ( t ) m ( d ) are the Legendre polynomials rescaled to the interval [0 , t -t start ] .

Recursive Computation via HiPPO-LegS To efficiently compute the coefficients ˜ c m ( t ) , we utilize the HiPPO-LegS framework, which provides a method for recursively updating the coefficients of a function projected onto an orthogonal basis as the interval expands. In our case, as t increases, the interval [ t start , t ] over which k ( d ) is defined also expands, and we can update ˜ c m ( t ) recursively.

Discretizing time with step size ∆ t and indexing t k = t start + k ∆ t , the update rule using the Euler method is:

<!-- formula-not-decoded -->

where ˜ c k = [˜ c 0 ( t k ) , ˜ c 1 ( t k ) , . . . , ˜ c M -1 ( t k )] ⊺ , and A ∈ R M × M and B ∈ R M are again matrices defined by the HiPPO-LegS operator as in Eq. 8 and 9.

Accounting for Variable Transformation and Parity The change of variables from x to d = t -x introduces a reflection in the function domain. Since the Legendre polynomials have definite parity, specifically,

<!-- formula-not-decoded -->

we need to adjust the coefficients accordingly when considering the reflected function.

As a result of this reflection, when projecting k ( d ) onto the Legendre basis, the coefficients ˜ c m ( t ) computed via the HiPPO-LegS updates will correspond to a reflected version of the function. To

account for this, we apply a parity correction to the coefficients. Specifically, the corrected coefficients c m ( t ) are related to ˜ c m ( t ) by a sign change determined by the degree m :

<!-- formula-not-decoded -->

This parity correction ensures that the computed coefficients properly represent the function over the interval [ t start , t ] without the effect of the reflection.

By computing c ( t ) recursively as t evolves, we can efficiently update ˜ B ( t ) = c ( t )[1 , . . . , 1] at each time step without the need to evaluate the integrals directly. This approach significantly reduces the computational burden associated with updating ˜ B ( t ) and allows for efficient computation of K ( t ) uu via the ODE.

## B.2.2 Unstability of directly evolving K ( t ) uu as ODE.

Empirically, we find that the direct ODE approach is less stable compared with RFF approach. Intuitively, it can be seen from the difference in the forms of their evolutions, especially in the first term. In RFF approach, the first term of the evolution of Fourier feature is of the form A ( t ) Z ( t ) w , which includes evolving vectors with the operator L 1 : X → A ( t ) X . In direct ODE approach, the first term in the direct evolution of K ( t ) uu is of the form A ( t ) K ( t ) uu + K ( t ) uu A ( t ) ⊤ , which requires the Lyapunov operator L 2 : X → A ( t ) X + XA ( t ) ⊤ . The critical difference is that L 2 has eigenvalues { λ i + λ j } (where λ i and λ j are eigenvalues of A ( t ) ) [Horn and Johnson, 1991], while L 1 has eigenvalues { λ i } . Since HiPPO-LegS uses a lower-triangular A ( t ) with negative diagonal entries, the eigenvalues are all negative λ i &lt; 0 . Hence, the eigenvalues of the Lyapunov operator L 2 are approximately as twice negative as the eigenvalues of L 1 , leading to a stiff ODE system with poorer numerical conditioning.

## C Finite basis approximation of posterior OHSVGP

Here, we show that q ( u ( t ) ) is the distribution of HiPPO coefficients of the posterior OHSVGP q t ( f ) . From Eq. 2, the posterior of the function values evaluated at arbitrary indices X is q t ( f X ) = N ( f X ; K ( t ) f X u K ( t ) -1 uu m ( t ) u , K f X f X -K ( t ) f X u K ( t ) -1 uu [ K ( t ) uu -S ( t ) u ] K ( t ) -1 uu K ( t ) uf X ) .

Based on this, we compute the mean of the m -th HiPPO coefficient for q t ( f ) as follows,

<!-- formula-not-decoded -->

which is exactly the variational mean of q ( u ( t ) m ) . Similarly, the covariance between the l -th and the m -th HiPPO coefficient for q t ( f ) can be computed as

<!-- formula-not-decoded -->

which is exactly the variational covariance between u ( t ) l and u ( t ) m in q ( u ( t ) ) .

Hence, if f ∼ q t ( f ) , then q ( u ( t ) m ) d = ∫ f ( x ) ϕ ( t ) m ( x )d x , which implies that we can approximate the posterior OHSVGP with finite basis: f = ∑ M m =1 u ( t ) m ϕ ( t ) m ( x ) , u ( t ) m ∼ q ( u ( t ) m ) .

## D Additional experimental details

## D.1 Infectious disease modeling

For a sanity check, we also fit a weekly renewal-equation model [Flaxman et al., 2020, Unwin et al., 2020, Monod et al., 2021], trained offline on the full history. This is a well-established infectious disease model that has been widely utilized by scientists during the COVID-19 pandemic, and we include its results in section F.1 (denoted as AR(2) Renewal in the legend). Interestingly, OHSVGP achieves better predictive performance than this traditional infectious disease model. Although this may be partly due to the strong inductive biases of the renewal equations, it nevertheless highlights OHSVGP's suitability for long infectious-disease time series.

The details of renewal-equation model are as follows. Let y t,a be the number of deaths on day t = 1 , . . . , T in state a = 1 , . . . , A (in our experiments, a = 1 since we only fit on 1 state), The probabilistic model is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where σ is the sigmoid function, N a is the population of country a , NumDeathsInit a is the initial number of deaths in country a , the IFR is the infection fatality rate and SI is the serial interval. The choice of f a ( τ ) requires a stochastic process and here we choose to model it as weekly random

effect on day τ using AR(2) process as in Monod et al. [2021], Unwin et al. [2020]. Typically, Markov chain Monte Carlo (MCMC) methods are used to infer the posterior distribution of f a and the other parameters, so it is less scalable than the online sparse GP based models shown in main text, especially when the number of states a and time steps T increase.

## D.2 SVGPVAE model details

With SVGPVAE, we utilize Jazbec et al. [2021] and notation from Zhu et al. [2023], and we have the following encoder-decoder model:

<!-- formula-not-decoded -->

with likelihood p ( y t | f t ) = N ( y t | φ ( f t ) , σ 2 I ) and decoder network φ : R L → R d y . The encoder ϕ : R d y → R 2 L yields (˜ y 1: L t , ˜ v t 1: L ) = ϕ ( y t ) . f t follows an L -dimensional multi-output GP and its approximate posterior is given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p ( f l 1: T | u l m ) is the prior conditional distribution.

Following Jazbec et al. [2021], the objective function is defined as:

<!-- formula-not-decoded -->

where L l H is the 'Hensman' ELBO described in Equation 7 of Jazbec et al. [2021]. Since the variational parameters m l and S l , and the likelihood are all amortized by neural networks, we further add EWC (Elastic Weight Consolidation; [Kirkpatrick et al., 2017]) regularization for both encoder and decoder networks to the loss above for continual learning.

## E Algorithmic Breakdown of OHSVGP

Algorithm 1 The HIPPO-SVGP ELBO for a single task of data. Differences with SVGP in blue.

Require: · X = { x 1 , . . . x n = t 1 } (training time steps up to time t 1 ),

- { y 1 , . . . , y n } (training targets),
- Z ∈ R M × 1 (inducing points) A ( t ) ∈ R M × M , B ( t ) ∈ R M × 1 (HIPPO matrices) ,
- m u ∈ R M × 1 , S u ∈ R M × M (variational params)
- 1: K fu = k ( X , Z ) , K uu = k ( Z , Z ) , K t 1 fu , K t 1 uu from HIPPO ODEs evolved from 0 to the final time step t 1 with HIPPO matrices A ( t ) , B ( t )
- 2: µ ( x i ) = K t 1 f i u ( K t 1 uu ) -1 m u
- ▷ Variational Posterior Mean
- 3: σ 2 ( x i ) = K t 1 f i f i -K t 1 f i u ( K t 1 uu ) -1 [ K t 1 uu -S u ]( K t 1 uu ) -1 K t 1 u f i ▷ Variational Posterior Variance
- 5: KL ← KL ( N ( m u , S u ) ||N (0 , K t 1 uu ))
- 4: ℓ varexp ← ∑ n i =1 E N ( µ ( x i ) ,σ 2 ( x i )) [ log p ( y i | f i ) ] ▷ closed form or quadrature/MC
- 6: return ℓ varexp -KL

Algorithm 2 The OHSVGP ELBO on the second task after learning the first task. Differences with OSVGP in blue.

Require: · X ′ = { t 1 &lt; x ′ 1 , . . . x ′ n ′ = t 2 } (training time steps up to time t 2 ),

- { y ′ 1 , . . . , y ′ n ′ } (training targets),
- Z t 1 ∈ R M × 1 (frozen and learned inducing points from task 1) with inducing variables u t 1 = f ( Z t 1 ) ,
- Z t 2 ∈ R M × 1 (new inducing points for task 2) with inducing variables u t 2 = f ( Z t 2 ) , A ( t ) ∈ R M × M , B ( t ) ∈ R M × 1 (HIPPO matrices),
- m u t 1 ∈ R M × 1 , S u t 1 ∈ R M × M (learned variational params from task 1),
- m u t 2 ∈ R M × 1 , S u t 2 ∈ R M × M (new variational params for task 2),
- K t 1 u t 1 u t 1

<!-- formula-not-decoded -->

- 1: K f ′ u t 2 = k ( X ′ , Z t 2 ) , K u t 2 u t 2 = k ( Z t 2 , Z t 2 ) , K t 2 f ′ u t 2 evolved from 0 to the final time step t 2 , K t 2 u t 2 u t 2 evolved from t 1 to the final time step t 2 with HIPPO matrices A ( t ) , B ( t )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 3: σ 2 t 2 ( x ′ i ) = K t 2 f ′ i f ′ i -K t 2 f ′ i u t 2 ( K t 2 u t 2 u t 2 ) -1 [ K t 2 u t 2 u t 2 -S u t 2 ]( K t 2 u t 2 u t 2 ) -1 K t 1 u t 2 f ′ i ▷ Variational Posterior Variance

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 9: return ℓ varexp -KL + CorrectionTerm ( t 1 , t 2 )

## F Additional results

## F.1 Full results including RMSE and ECE

We also include the full results of test NLPD and RMSE (ECE for experiments on COVID due to non-Gaussian likelihood) containing evaluation of Task i after learning tasks j = i, i +1 , · · · , 10 (5 for experiments on COVID) for all i .

We define the Root Mean Squared Error (RMSE) and Negative Log Predictive Density (NLPD) as the following:

<!-- formula-not-decoded -->

(41)

The Expected Calibration Error ([Guo et al., 2017]; ECE) is defined as

<!-- formula-not-decoded -->

where K = 10 , c k ∈ { 0 . 05 , 0 . 15 , . . . , 0 . 95 } N is the number of test points ( x i , y i ) , ˆ q p ( x i ) = the empirical p -quantile of the S predictive samples { ˆ y ( s ) i } S s =1 , S = 100 if the argument is true,

<!-- formula-not-decoded -->

Figure 7 shows the RMSE results for time series regression experiments (results of NLPD are in Figure 2 in the main text). Figure 8 and 9 show NLPD and ECE results on COVID dataset with an additional sanity check baseline described in Appendix D.1. Figure 10 - 13 show full results of RMSE and NLPD for UCI datasets, and Figure 14 and 15 show full results of RMSE and NLPD for ERA5 dataset.

Figure 7: Test set RMSE over the learned tasks vs. number of learned tasks for Solar Irradiance and Audio signal prediction dataset.

<!-- image -->

Figure 8: Test set NLPD per task after continually learning each task for all the 5 tasks on COVID dataset.

<!-- image -->

Figure 9: Test set ECE per task after continually learning each task for all the 5 tasks on COVID dataset.

<!-- image -->

Figure 10: Test set RMSE per task after continually learning each task for all the 10 tasks on UCI Skillcraft dataset.

<!-- image -->

<!-- image -->

(b) Skillcraft (L2)

Figure 11: Test set NLPD per task after continually learning each task for all the 10 tasks on UCI Skillcraft dataset.

Figure 12: Test set RMSE per task after continually learning each task for all the 10 tasks on UCI Powerplant dataset.

<!-- image -->

Figure 13: Test set NLPD per task after continually learning each task for all the 10 tasks on UCI Powerplant dataset.

<!-- image -->

Figure 14: Test set RMSE per task after continually learning each task for all the 10 tasks on ERA5 dataset.

<!-- image -->

Figure 15: Test set NLPD per task after continually learning each task for all the 10 tasks on ERA5 dataset.

<!-- image -->

## F.2 Results for time series regression with trainable kernel hyperparameters

Figure 16 and 17 show RMSE and NLPD results for time series regression experiments based on trainable kernel hyperparameters (i.e., keep optimizing kernel hyperparameters online in all the tasks). Notice that OVC is only compatible with fixed kernel [Maddox et al., 2021], so we don't consider it here. Compared with the results based on fixed kernel in the main text, here all methods show less stable performance. Previous works either find a well-performed fixed kernel [Maddox et al., 2021] or scale the KL terms in the online ELBO objective with a positive factor requiring careful tuning to mitigate the unstable online optimization of kernel hyperparameters [Stanton et al., 2021, Kapoor et al., 2021].

Figure 16: Test set RMSE over the learned tasks vs. number of learned tasks for Solar Irradiance and Audio signal prediction dataset (keep updating kernel hyperparameters).

<!-- image -->

Figure 17: Test set NLPD over the learned tasks vs. number of learned tasks for Solar Irradiance and Audio signal prediction dataset (keep updating kernel hyperparameters).

<!-- image -->

## F.3 Visualization of impacts of sorting criterion for OHSVGP in continual learning

Here, we consider fitting OHSVGPs with 20 inducing variables for a continual binary classification problem on the Two-moon dataset [Ganin et al., 2016]. The data is splitted into three task and we use a Bernoulli likelihood to model binary labels. We consider three different sorting criteria:

- Random, denoted as OHSVGP-rand. The order of data points in each task is obtained via random permutation.
- Kernel similarity maximization, denoted as OHSVGP-k-max. We select the i -th point in task j to be x ( j ) i = arg max x ∈ X ( j ) k ( x , x ( j ) i -1 ) for i &gt; 1 , and the first point in first task is set to be x (1) 1 = arg max x ∈ X (1) k ( x , 0 ) . The intuition is that the signals to memorize, when computing the prior covariance matrices, tend to be more smooth if the consecutive x 's are close to each other.
- Kernel similarity minimization, denoted as OHSVGP-k-min. We select the i -th point in task j to be x ( j ) i = arg min x ∈ X ( j ) k ( x , x ( j ) i -1 ) for i &gt; 1 , and the first point in first task is set to be x (1) 1 = arg min x ∈ X (1) k ( x , 0 ) . In this case, we deliberately make it difficult to memorize the signals in the recurrent computation for the prior covariance matrices.

Figure 18 show the decision boundaries after each task for different OHSVGPs based on different sorting criteria. We also include the decision boundaries of an OSVGP model for reference. Both OHSVGP-k-max and OHSVGP-rand return decision boundaries achieving 100% accuracy, while OHSVGP-k-min show catastrophic forgetting after Task 3, which suggests OHSVGP requires a sensible sorting criterion to perform well in continual learning tasks.

Figure 18: Decision boundaries of OSVGP, and OHSVGP models with different sorting criteria after each task (3 in total) on the Two-moon dataset. For OSVGP, we visualize the inducing points with red color.

<!-- image -->

## F.4 Comparison of basis-measure variants

Figure 19 shows the results of OHSGPR applied to a toy time-series regression dataset, where the data is split chronologically into three equal segments, used as Tasks 1-3 in an online learning setup. The figure compares the effect of several variants of the HiPPO operators [Gu et al., 2020, 2023] when used for OHSGPR. Subfigures (a-c) correspond to HiPPO-LegS as used in all of our main experiments. Subfigures (d-f) apply HiPPO-LegT, (g-i) apply HiPPO-LagT, based on the Laguerre polynomial basis, and (j-l) apply HiPPO-FouT, based on Fourier basis functions. While OHSGPR-LegS successfully memorizes all the past tasks, OHSGPR-LegT, OHSGPR-LagT and OHSGPR-FouT all demonstrate catastrophic forgetting to certain degree since instead of the uniform measure over the past (as is used in HiPPO-LegS), they are based on measures which place more mass over the recent history. LegT and FouT use a fixed-length sliding window measure, while LagT uses exponentially decaying measure, which assigns more importance to recent history.

Figure 19: Comparison of OHSGPR based on different HiPPO variants on a toy online regression dataset.

<!-- image -->
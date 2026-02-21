## State Size Independent Statistical Error Bound for Discrete Diffusion Models

Shintaro Wakasugi 1 , Taiji Suzuki 1,2

1 The University of Tokyo, 2 RIKEN AIP

wakasugi-shintaro362@g.ecc.u-tokyo.ac.jp, taiji@mist.i.u-tokyo.ac.jp

## Abstract

Diffusion models operating in discrete state spaces have emerged as powerful approaches, demonstrating remarkable efficacy across diverse domains, including reasoning tasks and molecular design. Despite their promising applications, the theoretical foundations of these models remain substantially underdeveloped, with the existing literature predominantly focusing on continuous-state diffusion models. A critical gap persists in the theoretical understanding of discrete diffusion modeling: the absence of a rigorous framework for quantifying estimation error with finite data. Consequently, the fundamental question of how precisely one can reconstruct the true underlying distribution from a limited training set remains unresolved. In this work, we analyze the estimation error induced by a score estimation of the discrete diffusion models. One of the main difficulties in the analysis stems from the fact that the cardinality of the state space can be exponentially large with respect to its dimension, which results in an intractable error bound by a naive approach. To overcome this difficulty, we make use of a property that the state space can be smoothly embedded in a continuous Euclidean space that enables us to derive a cardinality independent bound, which is more practical in real applications. In particular, we consider a setting where the state space is structured as a hypercube graph, and another where the induced graph Laplacian can be asymptotically well approximated by the ordinary Laplacian defined on the continuous space, and then derive state space size independent bounds.

## 1 Introduction

Diffusion modeling has demonstrated state-of-the-art performance in learning problems such as creating images (Song et al., 2021; Dhariwal and Nichol, 2021), videos (Ho et al., 2022), and audios (Chen et al., 2020; Kong et al., 2020), drawing significant attention to their applications.

Theoretical studies on diffusion modeling in continuous state spaces have been conducted within the framework of score-based generative modeling (Sohl-Dickstein et al., 2015; Song and Ermon, 2019; Song et al., 2021; Ho et al., 2020; Vahdat et al., 2021). One of the most important characterizations of diffusion modeling is the formulation using stochastic differential equations (SDEs) proposed by Song et al. (2021).

With the advent of SDE formulation, significant efforts have been made to analyze the estimation error between the true distribution and the generated distribution. Lee et al. (2022b) showed that the total variation distance between the two distributions can be bounded by the polynomial order of the score estimation error and the step size with the time discretization of the reverse process. Lee et al. (2022b) assumed the smoothness of the score function and the validity of the log-Sobolev inequality (LSI) for the true distribution, while Chen et al. (2023b) and Lee et al. (2022a) derived error bounds without the LSI condition, Chen et al. (2023a) further relaxed the smoothness assumption. Moreover, Song et al. (2021), Pidstrigach (2022) evaluated error rates under the manifold hypothesis,

which assumes that the true distribution is concentrated on a low-dimensional manifold. De Bortoli et al. (2021) and De Bortoli (2022) derived error bounds under the assumptions of score estimation error bounds at each time step and each point, considering dissipative structures and the manifold hypothesis, respectively.

While the above studies assumed the accuracy for score approximation, Oko et al. (2023) developed a theoretical framework that derives the approximation error of the score function when the true density belongs to a Besov space. Their work combined function approximation theory from deep learning with the approximation theory of diffusion modeling and employed concentration inequalities to establish score estimation error bounds. Building upon this seminal work, several refined theoretical analyses have been proposed, including relaxation of the lower bound condition for the density (Zhang et al., 2024), analyses under the manifold assumption on the support of the data distribution (Azangulov et al., 2024), statistical guarantees for reflected diffusion models (Holk et al., 2024), and minimax optimality of the probability flow ODE (Cai and Li, 2025).

Recently, diffusion modeling for discrete states has also gained attention (Hoogeboom et al., 2021; Austin et al., 2021; Richemond et al., 2022; Meng et al., 2022; Sun et al., 2023; Santos et al., 2023; Lou et al., 2024). Notable advancements have been made in learning problems with discrete structures, such as natural language processing (Austin et al., 2021; He et al., 2023; Wu et al., 2023), molecular design (Zhang et al., 2023; Gruver et al., 2023; Campbell et al., 2024; Lee et al., 2025), graph generation (Niu et al., 2020; Shi et al., 2020; Vignac et al., 2023), and segmentation (Zbinden et al., 2023). In addition, in areas where continuous diffusion modeling performs well, such as image (Hu et al., 2022; Zhu et al., 2023) and audio generation (Yang et al., 2023), discrete diffusion models have been shown to efficiently infer multimodal generation problems conditioned on discrete structures like text.

As for the theoretical analysis of discrete diffusion modeling, Campbell et al. (2022) analyzed total variation distance based on Markov chains, and Chen and Ying (2024) reduced the error under the condition that the discrete state space is restricted to the vertices of a hypercube. Ren et al. (2025) proposed a more general approximation theory for discrete diffusion characterized by a Poisson random measure with evolving intensity, allowing discrete diffusion modeling to be formulated as stochastic integrals similar to the theory of continuous diffusion modeling.

However, the theoretical analysis of score estimation error in discrete diffusion modeling remains unexplored. In this study, we derive score estimation error bounds for discrete diffusion models by applying the function approximation theory of neural networks and concentration inequalities, which were previously used in the score estimation theory of continuous diffusion modeling (Oko et al., 2023). Our main contributions are summarized as follows:

- First, we develop a theoretical framework for bounding the score estimation error in discrete diffusion models. Unlike continuous diffusion models that rely on L 2 loss in score matching, our analysis handles the Bregman divergence loss that naturally arises in discrete diffusion, and we rigorously control the estimation error using the Hellinger distance.
- Second, we introduce a novel approach to achieve state-size independent error bounds by embedding the discrete space X into R d and approximating the eigenvectors of the graph Laplacian by functions in an anisotropic Besov space. This enables the use of advanced function approximation results for deep ReLU networks. Under mild regularity assumptions, we show that the error bound depends only polylogarithmically on the number of discrete states M , which nearly achieves the optimal rate conjectured in Ren et al. (2025).
- Third, we demonstrate that this framework can be instantiated in concrete settings such as the hypercube [0 , 1] D and graph-based diffusion processes on smooth manifolds. In both examples, we show that the eigenvectors of the transition matrix admit efficient approximations.

## 2 Preliminary

Here, we introduce discrete diffusion models and prepare some technical matters for our theoretical analysis. Before introducing the discrete diffusion models, we briefly review the continuous state diffusion models. The continuous diffusion models consist of two stochastic processes, the forward process and the reverse process, so that the model can generate data whose distribution is sufficiently close to the true distribution p 0 on R d . The forward process { X t } t ≥ 0 on R d is formulated as the

following Ornstein-Uhlenbeck (OU) process:

<!-- formula-not-decoded -->

where { B t } t ≥ 0 represents a d -dimensional standard Brownian motion. Under certain assumptions on the initial distribution p 0 (Haussmann and Pardoux, 1986; Cattiaux et al., 2023), the distribution of X t , denoted by p t , converges exponentially to the standard normal distribution as t →∞ . The reverse process { Y t } 0 ≤ t ≤ T ( T ≥ 0 ) defined by the following stochastic process trace-backs the forward process:

<!-- formula-not-decoded -->

That is, the distribution of Y T -t coincides with p t for 0 ≤ t ≤ T , and thus we can generate samples from the target distribution p 0 by sampling Y T via the reverse process. However, we do not know the initial distribution of the reverse process p T and the score function ∇ log p t because both of them are dependent on the unknown true distribution p 0 . Here, the initial distribution p T can be replaced by the standard normal distribution for sufficiently large T since p t converges to the standard normal exponentially fast, and the score function ∇ log p t ( Y t ) can be estimated by least squares score matching using finite-size training data. These approximations, along with the time discretization of the process, induce errors in the resulting distribution generated by the model. Chen et al. (2023a) obtained an upper bound on this error in terms of Kullback-Leibler divergence (KL-divergence). Their result implies that the main part of the error is the estimation error of the score function ∇ log p t . Based on this observation, Oko et al. (2023) derived the following bound on the score estimation error.

Proposition 1 (Oko et al. (2023), informal) . Under some smoothness assumptions on p 0 , a score matching estimator ̂ s t obtained on a deep neural network model with an appropriate network size, can achieve

<!-- formula-not-decoded -->

Here, s represents the smoothness parameter of the Besov space to which p 0 belongs, and δ &gt; 0 is a sufficiently small end-point time.

This theorem implies that diffusion models with an appropriately designed score-function estimator can achieve the minimax optimal rate n -2 s d +2 s to estimate the target distribution with smoothness s . In this work, we establish an analogue of this result for discrete diffusion models.

## 2.1 Discrete diffusion modeling

Discrete diffusion modeling is defined over a finite set X instead of R d . Let M := | X | , and consider estimating the probability mass vector p 0 ∈ ∆ M , where ∆ M := { p | ∑ x ∈ X p ( x ) = 1 , p ∈ R M ≥ 0 } . We assume that each x ∈ X has a vector representation ι ( x ) ∈ R D . We identify this vector representation with x and use the same notation x for both meanings. A typical situation is X = { 0 , 1 } D where ι ( x ) = ( x 1 , . . . , x D ) ∈ R D , and another example is one-hot-vector representation X = { x ∈ { 0 , 1 } M | ∑ M i =1 x i = 1 } .

Forward process: In the forward process of discrete diffusion modeling, the distribution { p t } t ≥ 0 at each time step follows the master equation of the following Markov process:

<!-- formula-not-decoded -->

̸

where Q t ∈ R M × M is the transition rate matrix satisfying Q t ( x, x ) = -∑ y = x Q t ( y, x ) ( ∀ x ∈ X ) and Q t ( x, y ) ≥ 0 ( ∀ x = y ∈ X ) . If π denotes the stationary distribution of the Markov process (2), the following equation analogous to the continuous diffusion model holds (Bobkov and Tetali, 2006):

<!-- formula-not-decoded -->

Here, ρ ( Q ) is the modified log-Sobolev constant(Bobkov and Tetali, 2006) defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Ent π ( f ) := E π [ f log f ] -E π [ f ] log E π [ f ] and E π ( f, g ) = E π [ fQ ⊤ g ] .

̸

Reverse process: The reverse process { q t } 0 ≤ t ≤ T = { p T -t } 0 ≤ t ≤ T can be formulated by using another transition rate matrix Q t as follows (Kelly, 2011):

<!-- formula-not-decoded -->

̸

where Q t ( y, x ) = p t ( y ) p t ( x ) Q t ( x, y ) and Q t ( x, x ) = -∑ y = x Q t ( y, x ) . Then, it is known that q t = p T -t (0 ≤ t ≤ T ) . Although this reverse process only gives the ODE of the probability mass function q t , its particle implementation can be given by the τ -leaping algorithm (Campbell et al., 2022; Ren et al., 2025). This algorithm expresses the reverse process as a stochastic integral and applies the Euler-Maruyama scheme, similar to the continuous case. The transformation into a stochastic integral is performed using a Poisson random measure with evolving intensity (Protter, 1983). For simplicity, we consider the case where Q t is time-homogeneous ( Q t = Q holds for all t ≥ 0 ).

Proposition 2 (Ren et al. (2025)) . The reverse process (4) can be expressed as the following stochastic integral defined by a Poisson process N [ µ ] with respect to an intensity function µ (see Definition 2 in Appendix B for its definition):

<!-- formula-not-decoded -->

where X t -denotes the left limit of X t and ˜ Q denotes the matrix Q with the diagonal elements set to 0.

Score estimation: However, to implement the τ -leaping algorithm, we need to know the score function s ◦ t ( x, y ) := p t ( y ) p t ( x ) . We approximate it using a score network s : X 2 × R → R (( x, y, t ) ↦→ s t ( x, y )) in a deep neural network model F . The score network s can be estimated via score matching analogous to the continuous state diffusion models, but we need to account for the non-negativity constraint of the score function. For that purpose, the denoising score entropy is employed instead of L 2 loss (Lou et al., 2024):

̸

<!-- formula-not-decoded -->

̸

Here s ◦ t ( x, y | x 0 ) := p t ( y | x 0 ) p t ( x | x 0 ) and BR denotes the Bregman divergence defined by

<!-- formula-not-decoded -->

for a strictly convex function f , where we employ a particular choice K ( x ) := x -log x for the convex function f . In training the neural network, analogous to continuous diffusion models, we approximate the expectation over x 0 ∼ p 0 by empirical distribution defined by the training data D n := { x i } n i =1 ( x i i.i.d. ∼ p 0 ) with size n , and we seek ̂ s ∈ F that minimizes this empirical loss. We define the loss function ℓ as

̸

<!-- formula-not-decoded -->

Then, the empirical loss ̂ L ( s ) can be written as ̂ L ( s ) := 1 n n i =1 l s ( x i ) . As we have stated above, we find the empirical risk minimizer ̂ s in the set of deep neural networks that is defined as

<!-- formula-not-decoded -->

where L represents the depth, W = ( W i ) L i =1 represents the width with W L +1 = 1 , B represents the sparsity, and B is a bound on the norm of parameters. Here, the activation function η ( · ) is given by η ( x ) = ReLU( x ) := max( x, 0) . The function class to which the score network belongs is defined

Algorithm 1 Implementation of discrete diffusion modeling by τ -leaping

- 1: Input: ̂ y 0 ∼ π , time discretization { t k } k ∈ [0 ,K ] ( t 0 = 0 , t K = T -δ ), intensity function ˆ µ t , score network s t
- 3: for n = 0 to K -1 do
- ̂ 2: Output: sample from ̂ y t K ∼ ̂ q T -δ
- 4: ̂ y t n +1 ← ∑ y ∈ X ( y -ˆ y t n ) P ( ̂ µ t n ( y )( t n +1 -t n )) ;
- 5: end for

as F := { s ∈ Φ( L, W, S, B ) | s t ( x, y ) ∈ [1 /R,R ] ( ∀ t, x, y ) } with a hyper-parameter R ≥ 1 (see Assumption 3) 1 .

Wedenote the expectation of a measurable function f with respect to x 0 ∼ p 0 as Pf , and its empirical distribution as P n f , i.e., Pf := E x 0 ∼ p 0 [ f ] and P n f := 1 n ∑ n i =1 f ( x i ) . Accordingly, the expected loss L ( s ) and empirical loss ̂ L ( s ) can be expressed as

<!-- formula-not-decoded -->

Then, the empirical risk minimizer on the deep neural network model F is given by

<!-- formula-not-decoded -->

Once we have obtained an estimator ̂ s , we can define the corresponding intensity function ̂ µ t with time discretization ( t k ) K k =0 , with t 0 = 0 and t K = T -δ , as

<!-- formula-not-decoded -->

where ⌊ t ⌋ = t k for t ∈ [ t k , t k +1 ) . Moreover, due to Eq. (3), the initial distribution of the reverse process q 0 can be replaced by π ( ≃ p T ) . Then, the τ -leaping algorithm with our estimate ̂ s can be implemented as in Algorithm 1.

## 2.2 Technical assumptions and theoretical tools for the error analysis

To derive the discrepancy between the generated distribution and the true target distribution for the estimated discrete diffusion model implemented by τ -leaping algorithm (Alg. 1), we prepare some technical tools.

Assumption 1. The transition rate matrix Q is symmetric, and there exist positive constants C, D, and D such that Q ( x, y ) &lt; C , D &lt; -Q ( x, x ) &lt; D .

Assumption 2. There exists a lower bound ρ &gt; 0 for the modified log-Sobolev constant ρ ( Q ) .

Assumption 3. There exists R ≥ 1 such that the true score function and our model satisfy s ◦ t ( x, y ) ∈ [1 /R,R ] and ̂ s t ( x, y ) ∈ [1 /R,R ] .

Assumption 4. There exists γ ∈ [0 , 1] such that for any y ∈ X satisfying Q ( x t -, y ) &gt; 0 , the following holds: ∣ ∣ ∣ p t ( x t -) Q ( x t ,y ) p t ( x t ) Q ( x t -,y ) -1 ∣ ∣ ∣ ≲ 1 ∨ t -γ .

Assumption 1 is a natural condition for discrete diffusion models which ensures the regularity of the rate matrix (Ren et al., 2025). Assumption 2 guarantees the exponential convergence of the forward process in discrete diffusion models, serving a role analogous to functional inequalities in the continuous case (Bakry et al., 2014). Assumption 3 imposes boundedness on the score function, which mirrors common assumptions made in continuous-state diffusion processes (Chen et al., 2023a). Assumption 4 corresponds to the Lipschitz continuity of the score function in continuous diffusion models (Chen et al., 2023b,a), ensuring sufficient regularity for the analysis. Under these assumptions, the following result holds.

1 The function value restriction can be practically implemented as Φ ′ = max { min { Φ , R } , R -1 } where min and max can be realized by ReLU activation.

Proposition 3 (Ren et al. (2025)) . 2 In the τ -leaping algorithm, suppose the time discretization { t k } k ∈ [0 ,K ] satisfies t k +1 -t k ≤ κ (1 ∨ ( T -t k +1 ) 1+ γ -η ) for some η &gt; 0 and assume that

̸

<!-- formula-not-decoded -->

for the score network ̂ s t ( x, y ) . Here, we suppose that γ &lt; η ≲ 1 -T -1 for γ &lt; 1 , and η = 1 for γ = 1 , and it holds that

<!-- formula-not-decoded -->

Then, under Assumption 1 to 4, the following error bound holds

<!-- formula-not-decoded -->

Similar to continuous diffusion modeling, the upper bound on the error consists of three terms. (i) corresponds to the error from the convergence rate of the forward process, as shown in Eq. (3). (ii) arises from time discretization. (iii) represents the estimation error of the score network. Although the Girsanov's theorem cannot be applied to discrete diffusion models, another similar proposition can be derived from the characterization of a Poisson random measure with evolving intensity (Ren et al., 2025), which allows for the evaluation of the estimation error. However, unlike continuous diffusion models, the sample size or neural network parameter size required to satisfy the estimation error assumption has not been examined. Hence, the aim of this paper is to show an upper bound of ε sc analogous to Eq. (1).

## 3 Naive estimation error bound of discrete diffusion models

Here, we derive an estimation error bound of the score estimation network in a rather naive way. In our analysis, the key step is to derive an upper bound on the Hellinger distance between the empirical risk minimizer ̂ s and the true score function s ◦ , defined by

̸

<!-- formula-not-decoded -->

First, we give a bound on this Hellinger distance under a naive setting.

Theorem 1. Under Assumption 1 to 4, and the same parameter settings as Proposition 3, if the network size is set as L = O (log 2 ( mM )) , W = ˜ O ( M ) , S = ˜ O ( M ) , B = ˜ O ( M 2 ) and T = O (log( M ∨ n )) , then with probability at least 1 -2 e -t for t ≥ 1 , it holds that

<!-- formula-not-decoded -->

The formal proof of Theorem 1 is provided in Appendix C . By Proposition 3 and Theorem 1, we obtain the following result showing an estimation error bound for the estimated distribution.

Theorem 2. Suppose that the same condition as that of Theorem 1 holds. Then, the following estimation error bound holds with probability 1 -2 e -t for t ≥ 1 :

<!-- formula-not-decoded -->

One of the biggest difficulties in deriving this bound is the requirement to carefully treat the Bregman divergence in contrast to the continuous diffusion models where we could directly deal with the L 2 -norm. The difficulty can be overcome by noticing that the Bregman divergence can be lower bounded by the Hellinger distance. By combining this notion with the so-called peeling device

2 It was later pointed out that the τ -leaping update can become ill-defined for non-ordinal data when multiplejumps appear in a single time-window, which we were not aware of at the time of writing. Our theoretical results do not rely on this aspect and therefore remain valid.

van de Geer (2000), we achieve O (1 /n ) rate of convergence with respect to the sample size n , which significantly improves upon the standard Rademacher complexity bound that only provides an O (1 / √ n ) bound. An additional technical contribution is to show how ReLU neural networks can effectively approximate the score function s t .

This bound holds under minimal assumptions. On the other hand, if the state space is a product space such as X = { 0 , 1 } D (which often happens in real applications), then M can be exponentially large with respect to the dimension D . To overcome this difficulty, we consider a situation where the discrete state space X can be embedded into a Euclidean space R d and the eigenvectors corresponding to the Markov transition operator Q can be well approximated by a smooth function on R d . By doing so, we obtain a significantly improved bound as seen in the next section.

## 4 State size independent error analysis with continuous space embedding

As we mentioned above, we aim to improve the estimation error by explicitly considering an embedding from the discrete space X into a Euclidean space R D . Through the embedding, we may approximate the eigenvectors of Q by functions defined on R D enabling the application of function approximation theory for smooth functions defined on Euclidean spaces. Especially, the anisotropic Besov space is a useful function class that covers a wide range of functions with smoothness.

Anisotropic Besov space. Here, we begin by defining anisotropic Besov spaces. Let Ω = [0 , 1] d . For a function f : Ω → R , we define its L p -norm as

<!-- formula-not-decoded -->

For β ∈ R d ++ , let | β | := ∑ d j =1 | β j | . The r -th order finite difference in the direction h ∈ R d is defined as: ∆ r h ( f )( x ) := ∆ r -1 h ( f )( x + h ) -∆ r -1 h ( f )( x ) and ∆ r h ( f )( x ) := f ( x ) , for x + rh ∈ Ω ; otherwise, we define ∆ r h ( f )( x ) = 0 .

Definition 1 (Anisotropic Besov Space) . Let 0 &lt; p, q ≤ ∞ , β = ( β 1 , . . . , β d ) ⊤ ∈ R d ++ , r := max i ⌊ β i ⌋ +1 . Then the Besov semi-norm is defined as

<!-- formula-not-decoded -->

where w r,p is the r -th order modulus of smoothness defined by w r,p ( f, t ) := sup h ∈ R d : | h i |≤ t i ∥ ∆ r h ( f ) ∥ p . The anisotropic Besov space B β p,q (Ω) is defined as B β p,q (Ω) := { f ∈ L p (Ω) | ∥ f ∥ B β p,q := ∥ f ∥ p + | f | B β p,q &lt; ∞} . The unit ball of the anisotropic Besov space is denoted by U ( B β p,q ) .

The harmonic mean of the components of β , which is given by ˜ β := ( ∑ d j =1 1 /β j ) -1 , plays an important role in evaluating the approximation error by ReLU deep neural networks. It is known that the Hölder class and the Sobolev class with p = 2 are special cases of anisotropic Besov spaces Triebel (1983, 2011). In the following, we consider a situation where the eigenvectors of Q can be well approximated by a function in a Besov space on the continuous space.

## 4.1 Assumpitons

We now summarize the additional assumptions required for the analysis.

Assumption 5. Let 0 &lt; ε &lt; 1 . Suppose the orthonormal eigenvectors U = ( u 1 , . . . , u M ) of the graph Laplacian L = -Q satisfy u j ( x ) = O (1 / √ M ) for all x ∈ X and the initial distribution p 0 can be expanded as:

<!-- formula-not-decoded -->

Assume that for each j = 1 , . . . , M , there exists a function √ Mu ∗ j : X → [0 , 1] satisfying | u ∗ j ( x ) -u j ( x ) | ≤ ε/ √ M and representable as u ∗ j ( x ) = h j ( Px ) , where h j ∈ γ j U ( B β p,q ) with γ j &gt; 0 and P ∈ R d × D are projection matrices for all j . Moreover, assume ∥ P ∥ ∞ = O (1) and ˜ β &gt; 1 /p .

Assumption 6. For each j = 1 , . . . , M , the expansion coefficient satisfies | c j | ≲ | c 1 | · j -s and the Besov norms of h j satisfy | γ j | ≲ j γ where s &gt; 0 and γ ≥ 0 .

Assumption 5 is a technical condition that enables the application of function approximation theories in the anisotropic Besov space (Suzuki and Nitanda, 2021) (see also Suzuki (2019)). The factor O (1 / √ M ) in u j stems from the normalization of orthonormal eigenvectors, i.e., ∑ x ∈ X u j ( x ) 2 = 1 . Assumption 6 imposes a polynomial decay condition on c j , which is a standard regularity assumption in nonparametric statistics, particularly in the analysis of kernel methods (Caponnetto and De Vito, 2007; Ying and Pontil, 2008; Dieuleveut and Bach, 2016). The condition on γ j reflects the increasing complexity of the basis functions h j . Similar assumptions are common in the analysis of eigenfunctions of the Laplacian operator in continuous settings.

## 4.2 State space size independent error bound

Theorem 3. Assume that Assumption 5 and 6 as well as Assumption 1 to 4 hold, and if the network size is set as L = O (log 2 ( M )) , W = ˜ O ( M ) , S = ˜ O ( M ) , B = ˜ O ( M 2 ) and T = O (log( M ∨ n )) , the following estimation error bound holds with probability 1 -2 e -t :

<!-- formula-not-decoded -->

The proof of Theorem 3 is given in Appendix D. The estimation error bound in Theorem 3 shows that for s ≥ 1 , the dependence on M is only polylogarithmic, successfully removing any polynomial dependence on M . This aligns with the optimal rate conjectured in Ren et al. (2025) for discrete diffusion models. Even when γ +1 / 2 &lt; s &lt; 1 , the exponent on M remains below 1, yielding an improved convergence rate over Theorem 1.

As for the dependence on the sample size n , the bound does not explicitly depend on the embedded dimension d , which is desirable. Moreover, when s &lt; 1 , the rate recovers the optimal rate n -2 ˜ β/ (1+2 ˜ β ) derived by Suzuki and Nitanda (2021), but can be dependent on M polynomially. This is because estimation errors for O ( M ) -basis functions affect the final results when the decrease of coefficients is slow. On the other hand, when s is large, we may 'cut-off' redundant basis functions so that we mitigate the dependency on M to poly-log order while the rate with respect to n becomes a bit slower instead.

Finally, combining Proposition 3 and theorem 3 yields an upper bound on the distribution estimation error. Here, we let the right hand side of the bound in Theorem 3 be denoted by Ξ n,t .

Theorem 4. Suppose that the same condition as that of Theorem 3 holds. Then, the following error bound holds with probability 1 -2 e -t :

<!-- formula-not-decoded -->

Example 1: Hypercube { 0 , 1 } D . As an example, we consider the hypercube setting X = { 0 , 1 } D similar to Chen and Ying (2024). This setting is natural in practice, as the general discrete space X = [ S ] D which is commonly assumed in many works such as Campbell et al. (2022); Lou et al. (2024); Zhang et al. (2025) can be encoded as a hypercube structure { 0 , 1 } D log | S | . We let the eigenvalues of L = -Q be ordered as 0 = λ 1 &lt; λ 2 ≤ · · · ≤ λ M .

̸

Assumption 7. Let the discrete state space be X = { 0 , 1 } D . For any pair of distinct states x = y , assume the rate matrix Q ( x, y ) satisfies:

<!-- formula-not-decoded -->

where d ( x, y ) denotes the Hamming distance between x and y .

Under Assumption 7, since the diagonal term satisfies -Q ( x, x ) = D , we obtain D = O ( D ) . Moreover, the following spectral property holds. √

Lemma 4. Under Assumption 7, for every w ∈ X , h w ( x ) := cos( πw ⊤ x ) / M is an eigenvector corresponding to the eigenvalue 2 | w | , where | w | is the number of ones in w . In particular, λ 2 = 2 , which is independent of the dimension D .

Based on this lemma, we can derive the convergence of the discrete diffusion model as follows.

Corollary 1. Under the assumptions of Proposition 3 and Assumption 5 to 7, for arbitrary δ 0 &gt; 0 , the following error bound holds with probability 1 -2 e -t :

<!-- formula-not-decoded -->

The proof can be found in Appendix E. We observe that when s ≥ 1 , the convergence rate is essentially ˜ O ( n -(2( s -1)) / (2 s -1) ) , which is independent of the state space size M . This rate could be achieved by showing that the eigenvectors in this setting can be represented by cosine functions on the continuous space, and that such trigonometric functions can be efficiently approximated by deep neural networks. On the other hand, when s &lt; 1 , the convergence rate is ˜ O ( M/n ) , matching the error bound of the naive estimator in Theorem 1. In this case, accurately approximating the score function requires aggregating contributions across the entire state space, which leads to higher estimation complexity.

Example 2: Discrete graph diffusion process. Finally, we consider a diffusion process defined on a graph. In this setting, each point of X is randomly generated on a d -dimensional, smooth, closed and connected Riemannian manifold M⊂ R D isometrically embedded in R d through ι : M→ R D , where each point x obeys the uniform distribution p on M independently. On this point cloud X , we can define the transition rate matrix as the ordinary graph Laplacian : First, let the affinity matrix be W ( x, y ) := k σ ( x,y ) p σ ( x ) p σ ( y ) where k σ ( x, y ) = exp ( -∥ x -y ∥ 2 2 σ 2 ) and p σ ( x ) = ∑ y ∈ X k σ ( x, y ) , second, using a diagonal matrix D ∈ R X × X defined as D ( x, x ) = ∑ y ∈ X W ( x, y ) , let the normalized weight matrix as A = D -1 W , and finally we define the normalized Graph Laplacian as the transition matrix Q = 1 σ 2 ( A -I ) . The stochastic process corresponding to Q is known as a diffusion process on the graph with the normalized weight matrix A . In this setting, the graph Laplacian Q can be considered as a discrete approximation of the Laplace-Beltrami operator -∆ M defined on M , and thus the eigenfunctions and eigenvalues of the Laplace-Beltrami operator provide a good approximation of those of Q (i.e., bounding ε in Assumption 5) (Dunson et al., 2021). Since the eigenfunctions of ∆ M are included in the Sobolev space W β 2 ( M ) with arbitrary β , we can apply our theorem to derive the following bound. The proof is given in Appendix F.

Corollary 2. Suppose that the same assumptions as Theorem 3 and Assumption 5 and 6 hold, and s &gt; 1 . Then, if M is sufficiently large so that M = Ω( n max { 1+10 /d, 5 / 2+4 /d } (8 d +26) ( d/β +2)( s -1)+1 ) and σ = ( log( M ) M ) 1 4 d +13 , then there exists an event with high probability on the realization of X such that, for arbitrary β such that s -2 β/d &gt; 1 , the following error bound holds with probability 1 -2 e -t :

<!-- formula-not-decoded -->

Therefore, a distribution on a point cloud X on a smooth manifold M can be well approximated by the discrete diffusion models that utilize the graph diffusion process induced by the graph Laplacian as the forward process.

## 5 Numerical experiment

To complement our theoretical analysis, we conducted a score-matching experiment that exactly instantiates Example 1 on the hypercube { 0 , 1 } D with M := 2 D . For each D = 6 , 8 , 10 we constructed two distributions using the Hadamard basis H ∈ {± M -1 / 2 } M × M :

<!-- formula-not-decoded -->

Figure 1: Score matching results on hypercube toy examples. Each bar shows the mean denoising score entropy (DSE) over 10 trials; error bars are ± 1 standard deviation ( ≈ 0 . 4 -0 . 7 × 10 -4 ).

<!-- image -->

The coefficient vector c ∈ R M was chosen in two regimes:

Low-frequency: c 0 , . . . , c 4 ∼ N (0 , 1) , c k = 0 for k ≥ 5 .

High-frequency: c k ∼ N (0 , 1) for all k ≥ 2 (DC and first harmonic set to zero).

Activating only the first few modes yields a smooth distribution, whereas using the whole spectrum (minus the DC term) creates a highly oscillatory one. Each distribution was evolved for t = 1 under the rate matrix Q shown in Assumption 7. We then train a one-hidden-layer score network s θ ( x, y ) (input 2 D , 256 ReLU units, Softplus output) using the denoising score entropy (DSE) loss. Training used 5000 random Hamming-1 pairs, ADAM (lr = 10 -3 ), and 20 epochs. Performance was evaluated as the mean DSE over all Hamming-1 pairs. Each setting was repeated 10 times.

The mild increase from D = 6 to D = 10 matches the logarithmic dependence on M = 2 D predicted by Theorem 4 and corollary 1, while the higher errors for high-frequency mixtures reflect the bounds' sensitivity to the smoothness parameter. These observations provide concrete evidence that the theoretical guarantees translate directly to practice without any hyper-parameter tuning or architectural changes.

## 6 Conclusion

In this study, we established the first theoretical framework for estimating score functions in discrete diffusion models. We proved that directly approximating the score function for each discrete state using a neural network yields an estimation error rate of ˜ O ( M/n ) under a naive analysis and further improved this bound to ˜ O ( n -2 ˜ β ( s -1) / ( s -1+2 s ˜ β -˜ β ) ) independent of the state space size M by assuming a polynomial decay condition on the spectral decomposition of the target distribution. Our analysis made use of a decomposition of the target distribution by eigenvectors of the transition matrix Q . Then, we utilized the fact that the eigenfunctions can be well approximated by smooth functions (i.e., Besov spaces) on an embedded continuous space in order to reduce the model complexity. We also demonstrated concrete bounds for the hypercube settings and the graph diffusion processes.

One of the main drawbacks of our analysis is that we assumed a mild condition on the score function s ◦ t such that it is bounded above by R and below by 1 /R . This condition inherits the same condition assumed in the continuous space (Oko et al., 2023). Relaxing this condition to the case where there is no uniform lower bound on the density as performed in Zhang et al. (2024) is an important direction for future work.

## Acknowledgment

SW was partially supported by JST CREST (JPMJCR2115). TS was partially supported by JSPS KAKENHI (24K02905) and JST CREST (JPMJCR2015). This research is supported by the National Research Foundation, Singapore, Infocomm Media Development Authority under its Trust Tech Funding Initiative, and the Ministry of Digital Development and Information under the AI Visiting Professorship Programme (award number AIVP-2024-004). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views

of National Research Foundation, Singapore, Infocomm Media Development Authority, and the Ministry of Digital Development and Information.

The authors are grateful to Satoshi Hayakawa for insightful comments, particularly for drawing our attention to the potential ill-definedness of the τ -leaping scheme in the SDE form.

## References

- J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems , volume 34, pages 17981-17993, 2021.
- I. Azangulov, G. Deligiannidis, and J. Rousseau. Convergence of diffusion models under the manifold hypothesis in high-dimensions. arXiv:2409.18804 , 2024.
- D. Bakry, I. Gentil, and M. Ledoux. Analysis and Geometry of Markov Diffusion Operators . Springer, 2014.
- P. Bartlett, O. Bousquet, and S. Mendelson. Local Rademacher complexities. The Annals of Statistics , 33:1487-1537, 2005.
- S. Bobkov and P. Tetali. Modified logarithmic Sobolev inequalities in discrete settings. Journal of Theoretical Probability , 19:289-336, 2006.
- O. Bousquet. A Bennett concentration inequality and its application to suprema of empirical process. Comptes Rendus de l'Académie des Sciences Paris, Series I - Mathematics , 334:495-500, 2002.
- C. Cai and G. Li. Minimax optimality of the probability flow ODE for diffusion models. arXiv:2503.09583 , 2025.
- A. Campbell, J. Benton, V. De Bortoli, T. Rainforth, G. Deligiannidis, and A. Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems , volume 35, pages 28266-28279. Curran Associates, Inc., 2022.
- A. Campbell, J. Yim, R. Barzilay, T. Rainforth, and T. Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 5453-5512. PMLR, 2024.
- A. Caponnetto and E. De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics , 7:331-368, 2007.
- P. Cattiaux, G. Conforti, I. Gentil, and C. Léonard. Time reversal of diffusion processes under a finite entropy condition. Annales de l'Institut Henri Poincaré, Probabilités et Statistiques , 59(4):1844 1881, 2023.
- H. Chen and L. Ying. Convergence analysis of discrete diffusion model: Exact implementation through uniformization. arXiv:2402.08095 , 2024.
- H. Chen, H. Lee, and J. Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In International Conference on Machine Learning , pages 4735-4763. PMLR, 2023a.
- N. Chen, Y. Zhang, H. Zen, R. J. Weiss, M. Norouzi, and W. Chan. Wavegrad: Estimating gradients for waveform generation. In International Conference on Learning Representations , 2020.
- S. Chen, S. Chewi, J. Li, Y. Li, A. Salim, and A. Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In The Eleventh International Conference on Learning Representations , 2023b.
- V. De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis. Transactions on Machine Learning Research , 2022. ISSN 2835-8856. URL https://openreview. net/forum?id=MhK5aXo3gB .

- V. De Bortoli, J. Thornton, J. Heng, and A. Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. Advances in Neural Information Processing Systems , 34: 17695-17709, 2021.
- P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems , 34:8780-8794, 2021.
- A. Dieuleveut and F. Bach. Nonparametric stochastic approximation with large step-sizes. The Annals of Statistics , 44(4):1363-1399, 2016.
- D. B. Dunson, H.-T. Wu, and N. Wu. Spectral convergence of graph Laplacian and heat kernel reconstruction in L ∞ from random samples. Applied and Computational Harmonic Analysis , 55: 282-336, 2021.
- N. Gruver, S. D. Stanton, N. C. Frey, T. G. J. Rudner, I. Hotzel, J. Lafrance-Vanasse, A. Rajpal, K. Cho, and A. G. Wilson. Protein design with guided discrete diffusion. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. URL https://openreview.net/ forum?id=MfiK69Ga6p .
- A. Hassannezhad, G. Kokarev, and I. Polterovich. Eigenvalue inequalities on riemannian manifolds with a lower ricci curvature bound. Journal of Spectral Theory , 6(4):807-835, 2016.
- U. G. Haussmann and E. Pardoux. Time reversal of diffusions. The Annals of Probability , 14(4): 1188-1205, 1986.
- Z. He, T. Sun, Q. Tang, K. Wang, X. Huang, and X. Qiu. DiffusionBERT: Improving generative masked language models with diffusion models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 4521-4534, Toronto, Canada, 2023. Association for Computational Linguistics.
- J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems , volume 33, pages 6840-6851, 2020.
- J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet. Video diffusion models. arXiv:2204.03458 , 2022.
- A. Holk, C. Strauch, and L. Trottner. Statistical guarantees for denoising reflected diffusion models. arXiv:2411.01563 , 2024.
- E. Hoogeboom, D. Nielsen, P. Jaini, P. Forré, and M. Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems , volume 34, pages 12454-12465, 2021.
- M. Hu, Y. Wang, T.-J. Cham, J. Yang, and P. Suganthan. Global context with discrete diffusion in vector quantised modelling for image generation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 11492-11501, 2022.
- F. P. Kelly. Reversibility and Stochastic Networks . Cambridge University Press, USA, 2011.
- V. Koltchinskii. Local Rademacher complexities and oracle inequalities in risk minimization. The Annals of Statistics , 34:2593-2656, 2006.
- Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations , 2020.
- H. Lee, J. Lu, and Y. Tan. Convergence of score-based generative modeling for general data distributions. In NeurIPS 2022 Workshop on Score-Based Methods , 2022a.
- H. Lee, J. Lu, and Y. Tan. Convergence for score-based generative modeling with polynomial complexity. In Advances in Neural Information Processing Systems , 2022b.
- S. Lee, K. Kreis, S. P. Veccham, M. Liu, D. Reidenbach, Y. Peng, S. Paliwal, W. Nie, and A. Vahdat. GenMol: A drug discovery generalist with discrete diffusion. arXiv:2501.06158 , 2025.

- A. Lou, C. Meng, and S. Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 32819-32848. PMLR, 2024.
- C. Meng, K. Choi, J. Song, and S. Ermon. Concrete score matching: Generalized score matching for discrete data. In Advances in Neural Information Processing Systems , volume 35, pages 34532-34545. Curran Associates, Inc., 2022.
- R. Nakada and M. Imaizumi. Adaptive approximation and generalization of deep neural network with intrinsic dimensionality. Journal of Machine Learning Research , 21(174):1-38, 2020.
- C. Niu, Y. Song, J. Song, S. Zhao, A. Grover, and S. Ermon. Permutation invariant graph generation via score-based generative modeling. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics , volume 108 of Proceedings of Machine Learning Research , pages 4474-4484. PMLR, 2020.
- K. Oko, S. Akiyama, and T. Suzuki. Diffusion models are minimax optimal distribution estimators. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 26517-26582. PMLR, 2023.
- J. Pidstrigach. Score-based generative models detect manifolds. In Advances in Neural Information Processing Systems , volume 35, pages 35852-35865. Curran Associates, Inc., 2022.
- P. Protter. Point Process Differentials with Evolving Intensities , pages 467-472. Springer Netherlands, Dordrecht, 1983.
- Y. Ren, H. Chen, G. M. Rotskoff, and L. Ying. How discrete and continuous diffusion meet: Comprehensive analysis of discrete diffusion models via a stochastic integral framework. In The Thirteenth International Conference on Learning Representations , 2025.
- P. H. Richemond, S. Dieleman, and A. Doucet. Categorical sdes with simplex diffusion. arXiv:2210.14784 , 2022.
- J. E. Santos, Z. R. Fox, N. Lubbers, and Y. T. Lin. Blackout diffusion: Generative diffusion models in discrete-state spaces. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 9034-9059. PMLR, 2023.
- J. Schmidt-Hieber. Nonparametric regression using deep neural networks with relu activation function. The Annals of Statistics , 48(4):1916-1921, 2020.
- C. Shi, M. Xu, Z. Zhu, W. Zhang, M. Zhang, and J. Tang. Graphaf: a flow-based autoregressive model for molecular graph generation. In International Conference on Learning Representations , 2020. URL https://openreview.net/forum?id=S1esMkHYPr .
- J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning , pages 2256-2265. PMLR, 2015.
- Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019.
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations , 2021.
- H. Sun, L. Yu, B. Dai, D. Schuurmans, and H. Dai. Score-based continuous-time discrete diffusion models. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=BYWWwSY2G5s .
- T. Suzuki. Adaptivity of deep reLU network for learning in Besov and mixed smooth Besov spaces: optimal rate and curse of dimensionality. In International Conference on Learning Representations , 2019. URL https://openreview.net/forum?id=H1ebTsActm .

- T. Suzuki and A. Nitanda. Deep learning is adaptive to intrinsic dimensionality of model smoothness in anisotropic Besov space. In Advances in Neural Information Processing Systems , volume 34, pages 3609-3621. Curran Associates, Inc., 2021.
- M. Talagrand. New concentration inequalities in product spaces. Inventiones Mathematicae , 126: 505-563, 1996.
- H. Triebel. Theory of Function Spaces . Monographs in Mathematics. Birkhäuser Verlag, 1983.
- H. Triebel. Entropy numbers in function spaces with mixed integrability. Revista matemática complutense , 24(1):169-188, 2011.
- A. Vahdat, K. Kreis, and J. Kautz. Score-based generative modeling in latent space. In Advances in Neural Information Processing Systems , volume 34, pages 11287-11302. Curran Associates, Inc., 2021.
- S. van de Geer. Empirical Processes in M-Estimation . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2000.
- C. Vignac, I. Krawczuk, A. Siraudin, B. Wang, V. Cevher, and P. Frossard. DiGress: Discrete denoising diffusion for graph generation. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=UaAD-Nu86WX .
- M. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.
- T. Wu, Z. Fan, X. Liu, H.-T. Zheng, Y. Gong, y. shen, J. Jiao, J. Li, z. wei, J. Guo, N. Duan, and W. Chen. AR-Diffusion: Auto-regressive diffusion model for text generation. In Advances in Neural Information Processing Systems , volume 36, pages 39957-39974. Curran Associates, Inc., 2023.
- D. Yang, J. Yu, H. Wang, W. Wang, C. Weng, Y. Zou, and D. Yu. Diffsound: Discrete diffusion model for text-to-sound generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing , 31:1720-1733, 2023.
- Y. Ying and M. Pontil. Online gradient descent learning algorithms. Foundations of Computational Mathematics , 8:561-596, 01 2008.
- L. Zbinden, L. Doorenbos, T. Pissas, A. T. Huber, R. Sznitman, and P. Marquez-Neila. Stochastic segmentation with conditional categorical diffusion models. In IEEE/CVF International Conference on Computer Vision (ICCV) , pages 1119-1129, Los Alamitos, CA, USA, 2023. IEEE Computer Society.
- K. Zhang, H. Yin, F. Liang, and J. Liu. Minimax optimality of score-based diffusion models: Beyond the density lower bound assumptions. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 6013460178. PMLR, 2024.
- W. Zhang, X. Wang, J. Smith, J. Eaton, B. Rees, and Q. Gu. DiffMol: 3d structured molecule generation with discrete denoising diffusion probabilistic models. In ICML 2023 Workshop on Structured Probabilistic Inference &amp; Generative Modeling , 2023.
- Z. Zhang, Z. Chen, and Q. Gu. Convergence of score-based discrete diffusion models: A discrete-time analysis. In The Thirteenth International Conference on Learning Representations , 2025. URL https://openreview.net/forum?id=pq1WUegkza .
- Y. Zhu, Y. Wu, K. Olszewski, J. Ren, S. Tulyakov, and Y. Yan. Discrete contrastive diffusion for cross-modal music and image generation. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=1-MBdJssZ-S .

## A Construction of Neural Networks

This section collects fundamental tools on function approximation using neural networks. These results play a central role in our approximation error analysis. Many of the lemmas below are based on Oko et al. (2023).

## A.1 Composition and Combination Lemmas

We begin with lemmas that describe how to combine multiple neural networks. These lemmas are essential to realize a large neural network that approximates complicated functions.

Lemma 5 (Nakada and Imaizumi (2020)) . For any neural networks ϕ 1 , . . . , ϕ k with ϕ i : R d i → R d i +1 and ϕ i ∈ Φ( L i , W i , S i , B i ) , there exists a network ϕ ∈ Φ( L, W, S, B ) such that ϕ ( x ) = ϕ k ◦ ϕ k -1 ◦ · · · ◦ ϕ 1 ( x ) for x ∈ R d 1 with

<!-- formula-not-decoded -->

Lemma 6 (Oko et al. (2023)) . For any networks ϕ 1 , . . . , ϕ k with ϕ i : R d i → R d ′ i and ϕ i ∈ Φ( L i , W i , S i , B i ) , there exists a network ϕ ∈ Φ( L, W, S, B ) such that ϕ ( x ) = [ ϕ 1 ( x 1 ) ⊤ , . . . , ϕ k ( x k ) ⊤ ] ⊤ : R d 1 + ··· + d k → R d ′ 1 + ··· + d ′ k for x = ( x ⊤ 1 · · · x ⊤ k ) ⊤ with

<!-- formula-not-decoded -->

Lemma 7 (Oko et al. (2023)) . For any networks ϕ 1 , . . . , ϕ k with ϕ i : R d i → R d and ϕ i ∈ Φ( L i , W i , S i , B i ) , there exists a network ϕ ∈ Φ( L, W, S, B ) such that ϕ ( x ) = ∑ k i =1 ϕ i ( x i ) : R d 1 + ··· + d k → R d for x = ( x ⊤ 1 · · · x ⊤ k ) ⊤ with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 8 (Oko et al. (2023)) . For t 1 &lt; t 2 &lt; t 1 &lt; t 2 and f ( x, t ) : R d × R → R , assume that ϕ i ( x, t ) approximates f ( x, t ) up to ε &gt; 0 within [ t i , t i ] ( i = 1 , 2) . There exist networks ϕ 1 swit ( t ; t 2 , t 1 ) , ϕ 2 swit ( t ; t 2 , t 1 ) ∈ Φ(3 , (1 , 2 , 1 , 1) ⊤ , 8 , max { t 1 , ( t 1 -t 2 ) -1 } ) with

<!-- formula-not-decoded -->

## A.2 Approximations of rational functions

The following lemmas describe how to approximate elementary operations such as multiplication and reciprocal using neural networks. These are particularly important because the score function in discrete diffusion models involves the ratio of probabilities between two discrete states, which can be approximated by neural networks designed to emulate rational functions.

Lemma 9 (Oko et al. (2023)) . For k ≥ 2 , C ≥ 1 , 0 &lt; ε error ≤ 1 and ε &gt; 0 , there exists a network ϕ mult ( x 1 , . . . , x k ) ∈ Φ( L, W, S, B ) with L = O (log k (log ε -1 + k log C )) , ∥ W ∥ = 48 k, S = O ( k log ε -1 + k log C ) , B = C k such that | ϕ mult ( x ) | ≤ C k , ϕ mult ( x 1 , . . . , x k ) = 0 if at least one of x i is equal to zero, and

<!-- formula-not-decoded -->

Lemma 10 (Oko et al. (2023)) . For a constant R &gt; 1 and any 0 &lt; ε &lt; R -1 , there exists a network ϕ rec ( x 1 , . . . , x k ) ∈ Φ( L, W, S, B ) with L = O (log 2 ε -1 ) , ∥ W ∥ = O (log 3 ε -1 ) , S = O (log 4 ε -1 ) , B = O ( ε -2 ) such that

<!-- formula-not-decoded -->

## B Poisson Random Measures with evolving intensity

In this section, we introduce the formal definition of Poisson random measures with evolving intensity, which is a foundational concept required for implementing the τ -leaping algorithm. Although this concept is not central to the main theoretical results of this paper, we include the definition here for completeness and reference.

Definition 2 (Protter (1983); Ren et al. (2025)) . Consider a probability space (Ω , F , P ) and a measure space ( X, B , ν ) . A non-negative predictable process λ t ( y ) on R + × X × Ω is assumed to satisfy for any T &gt; 0 :

<!-- formula-not-decoded -->

where ν is the counting measure. probability measure N [ λ ](d t, d y ) on R + × X is called a Poisson random measure with evolving intensity λ t ( y ) if:

1. For any B ∈ B and 0 ≤ s &lt; t , N [ λ ](( s, t ] × B ) ∼ P ( ∫ t s ∫ B λ τ ( y ) ν ( dy ) dτ ) . where P ( · ) represents a Poisson distribution with the given expectation.
2. For any t ≥ 0 , B and disjoint sets { B i } n i =1 , the processes { N t [ λ ]( B i ) := N [ λ ]((0 , t ] × B i ) } n i =1 are independent.

## C Proof of Theorem 1

To prove Theorem 1, we decompose the total estimation error into approximation and generalization components . We begin by analyzing the score approximation error with a neural network. This part captures the model bias, and the resulting bound depends on the smoothness of the score function and the expressive power of the network class. The subsequent subsection will then address the generalization error using tools from statistical learning theory. Together, these analyses yield the desired estimation error bound.

## C.1 Approximation error analysis

In this subsection, we construct a neural network that approximates the true score function s ◦ t ( x, y ) in a compositional manner by separately approximating p t ( x ) and its reciprocal 1 /p t ( x ) , and then combining them in a multiplicative way.

Lemma 11. For any ε 1 &gt; 0 , there exists a neural network ϕ 1 ( x, t ) ∈ Φ( L, W, S, B ) such that

<!-- formula-not-decoded -->

The parameters of ϕ 1 ( x, t ) are bounded as follows:

<!-- formula-not-decoded -->

Proof. We begin by approximating the function e -λ j t using a neural network. Define A := log 3 ε -1 .

For j = 0 , we set ϕ j = 1 so that e -λ j t is approximated without an error. For j &gt; 0 , using the Taylor expansion, we obtain

<!-- formula-not-decoded -->

where θ ∈ [0 , 1] . Setting k = max {⌈ 2 e 2 D ⌉ , ⌈ log 3 ε -1 ⌉} , we obtain the bound

<!-- formula-not-decoded -->

Here, we used ( n/e ) n &lt; n ! in the second inequality. By Lemma 9, for each ( -λ j ) i i ! ( t -s ) i , there exists a neural network ϕ ( t ; i ) that approximates it within an error of ε/ 3( k +1) over the interval s ≤ t ≤ s +2 /λ j . The network parameters satisfy:

<!-- formula-not-decoded -->

Similarly, by lemma 7, there exists a neural network ϕ s ( t ) that approximates e -λ j t within an error of ε/ 3 over s ≤ t ≤ s +2 /λ j , with the following parameter bounds:

<!-- formula-not-decoded -->

Note that | c j u j ( x ) | ≤ 1 . Indeed,

<!-- formula-not-decoded -->

Thus, we can construct a neural network ϕ ∗ j ( x, t ) that approximates c j u j ( x ) e -tλ j as follows:

<!-- formula-not-decoded -->

Using Lemma 9, the approximation error from each ϕ mult is bounded by ε/ log ε -1 . We set the parameters of ϕ mult to L = O (log ε -1 ) , ∥ W ∥ ∞ = O (1) , S = O (log ε -1 ) , B = O (1) . To restrict the input t of ϕ ∗ j ( x, t ) to the range [0 , A ] , we define ϕ j ( x, t ) := ReLU( ϕ ∗ j ( x, t ) -ϕ ∗ j ( x, A )) + ϕ ∗ j ( x, A ) . This ensures that ϕ j ( x, t ) approximates e -λ j t with an error at most ε for all t ≥ 0 . For t ≤ A , the following inequality holds: | ϕ j ( x, t ) -c j u j ( x ) e -λ j t | ≤ ε/ 3 + ε/ 3 &lt; ε . For t &gt; A , we have | ϕ j ( x, t ) -c j u j ( x ) e -λ j t | ≤ | ϕ j ( x, t ) -ϕ j ( x, A ) | + | ϕ j ( x, A ) -c j u j ( x ) e -λ j A | + | c j u j ( x )( e -λ j t -e -λ j A ) | ≤ 0 + 2 ε/ 3 + ε/ 3 = ε . Thus, the parameters of ϕ ( x, t ) are evaluated as follows:

<!-- formula-not-decoded -->

Finally, setting ε := ε 1 /M 2 and summing ϕ j ( x, t ) over all j = 1 , . . . , M , we construct a neural network ϕ 1 ( x, t ) that approximates Mp t ( x ) within an error of ε 1 . By Lemmas 7 and 9, the parameter bounds are given by

<!-- formula-not-decoded -->

Building on this result, we next construct a neural network that approximates the reciprocal 1 /p ( x ) .

Lemma 12. For any 0 &lt; ε 0 ≤ R and ε 1 &gt; 0 there exists a neural network ϕ ( x, t ) ∈ Φ( L, W, S, B satisfying the following inequality:

t 1 2 )

<!-- formula-not-decoded -->

with the parameters of ϕ 2 are bounded as follows:

<!-- formula-not-decoded -->

Proof. From Lemmas 5 and 10, there exists ϕ 2 ( x, t ) = ϕ rec ◦ ϕ 1 ( x, t ) satisfying

<!-- formula-not-decoded -->

Here, the network parameters coincide with those stated above. Regarding p t , the boundedness of the score function ensures that Mp t ( x ) = Mp t ( x ) ∑ y ∈ X p t ( y ) = M ∑ y ∈ X s t ( y,x ) ≥ 1 R which satisfies the conditions of Lemma 10

Combining the approximations of p t ( x ) and its reciprocal, we are now ready to construct a neural network that directly approximates the score function s ◦ t ( x, y ) = p t ( y ) /p t ( x ) . The following lemma formalizes this construction and provides the approximation error bound.

Lemma 13. For any 0 &lt; ε 0 ≤ 1 R and ε 1 , ε 2 &gt; 0 , there exists a network ϕ score ( x, y, t ) ∈ F such that

<!-- formula-not-decoded -->

Here, the parameters of ϕ score are evaluated as:

<!-- formula-not-decoded -->

In particular, the following holds:

̸

<!-- formula-not-decoded -->

Proof. Consider the neural network ϕ 3 ( x, y, t ) := [ ϕ 1 ( y, t ) , ϕ 2 ( x, t )] ⊤ , which parallelizes the estimation of p t ( y ) and 1 /p t ( x ) . The function ϕ score ( x, y, t ) := ϕ mult ◦ ϕ 3 ( x, y, t ) is a neural network that estimates s ◦ t ( x, y ) = p t ( y ) p t ( x ) , and its error is evaluated as follows:

<!-- formula-not-decoded -->

From Lemmas 5, 9, 11 and 12, the parameters are bounded as showed in above:

<!-- formula-not-decoded -->

Considering ¯ ϕ score ( x, y, t ) which restricts the output of ϕ score ( x, y, t ) to [1 /R,R ] , Assumption 3 ensures that this restriction does not increase the error, and the order of the parameters remains unchanged. Thus, we can conclude ϕ score ∈ F .

Since 1 /R ≤ ϕ score , s ◦ ≤ R and 0 ≤ x -log( x ) -1 ≤ 4 log( R )( x -1) 2 for x ∈ [ -1 /R 2 , R 2 ] , we have that

<!-- formula-not-decoded -->

Therefore, we obtain that

̸

<!-- formula-not-decoded -->

which achieves the assertion.

## C.2 Generalization Error Analysis

This subsection focuses on bounding the generalization error. The analysis proceeds in three steps:

- First, we relate the score estimation error to a conditional form via a denoising representation.
- Second, we define a loss class and control its generalization error using local Rademacher complexity and Peeling device.
- Third, we derive explicit complexity bounds using covering number estimates for neural networks.

̸

̸

We begin by expressing the explicit score matching entropy as the denoising score entropy. This representation justifies interpreting the score entropy as the empirical loss.

Lemma 14 (Lou et al. (2024)) . For any s t ( x t , y ) and t &gt; 0 , the following holds:

̸

<!-- formula-not-decoded -->

̸

where C is a constant independent of s t .

Proof.

<!-- formula-not-decoded -->

̸

where C is a constant depending only on p t , Q .

Next, We relate the Hellinger distance to the expected excess risk.

Lemma 15. For any g ∈ G := { g := ℓ ̂ s -ℓ s ◦ | ̂ s ∈ F} , we have that ∫ T -δ 0 h 2 ( ̂ s t , s ◦ t )d t ≲ Pg . Proof. From Lemma 14, we have:

̸

<!-- formula-not-decoded -->

̸

which concludes the assertion.

Here, we aim to bound the generalization error using the local Rademacher complexity R n ( G r ) . We define G r := { g := ℓ s -ℓ s ◦ | g ∈ G,Pg ≤ r } .

̸

̸

̸

̸

̸

̸

̸

Lemma 16. For any g ∈ G := { g := ℓ s -ℓ s ◦ | s ∈ F} , it holds that ∥ g ∥ ∞ ≲ TRD which also indicates that Pg 2 ≲ TRDPg.

Proof. Define ℓ ∗ s ( x, y, x 0 , t ) := BR K ( s t ( x, y ) ∥ s ◦ t ( x, y | x 0 )) · s ◦ t ( x, y | x 0 ) . Then, for any s ∈ F , x, y ∈ X , it holds that

<!-- formula-not-decoded -->

Therefore, we arrive at

<!-- formula-not-decoded -->

̸

where we used the symmetricity of Q and R ≥ 1 in the last equality. Therefore, we have Pg 2 ≤ ∥ g ∥ ∞ Pg ≲ TRDPg .

We introduce two classical tools for controlling the supremum of the empirical loss.

Proposition 17 (Peeling device (van de Geer, 2000; Bartlett et al., 2005; Koltchinskii, 2006)) . Suppose there exists a function ϕ : [0 , ∞ ) → [0 , ∞ ) and ̂ r ∗ &gt; 0 such that ∀ r &gt; ̂ r ∗ ,

<!-- formula-not-decoded -->

Then, ∀ r &gt; ̂ r ∗ , the following holds:

<!-- formula-not-decoded -->

Proposition 18 (Talagrand's concentration inequality) . (Talagrand, 1996; Bousquet, 2002)] Let ˜ G be a separable set of measurable functions on the probability space ( X, A , P ) , and suppose that ∀ g ∈ ˜ G ,

<!-- formula-not-decoded -->

Then, ∀ t &gt; 0 ,

<!-- formula-not-decoded -->

By using these theorems, the prediction error can be decomposed into the approximation error (model bias term) and the generalization error (variance term).

̸

̸

Lemma 19. Define s ∗ := arg min f ∈F L ( f ) , ̂ s := arg min f ∈F ̂ L ( f ) , ̂ g := ℓ ̂ s -ℓ s ◦ , and r ∗ := L ( s ∗ ) -L ( s ◦ ) . For the function ϕ ( r ) defined in Proposition 17, there exists ̂ r ≳ max { ϕ ( ̂ r ) , r ∗ , tTD n } such that the following holds with probability 1 -2 e -t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define ˜ G := { ˜ g := Pg -g ( x ) Pg + r | g ∈ G } . Then, the following holds:

<!-- formula-not-decoded -->

Using Proposition 17 to bound the term E { x ′ i } [ sup g ∈ ˜ G 1 n ∑ n i =1 g ( x ′ i ) ] in Proposition 18, we have

<!-- formula-not-decoded -->

The first inequality follows from a standard property of the Rademacher complexity, and the final bound is derived using Proposition 17. Therefore, from Proposition 18, it follows that with probability at least 1 -e -t ,

<!-- formula-not-decoded -->

Let us define

<!-- formula-not-decoded -->

Since Pg ∗ 2 ≤ TRDPg ∗ , the second term can be bounded via Bernstein's inequality (Wainwright, 2019, Proposition 2.14) as

<!-- formula-not-decoded -->

with probability 1 -e -t for t &gt; 0 . We now choose ̂ r ≳ max { ϕ ( ̂ r ) , r ∗ , tTRD n } . If necessary, we can scale ̂ r so that ψ n ( ̂ r ) ≤ 1 / 2 . Under this condition, the following inequality holds:

<!-- formula-not-decoded -->

Proof. By Lemma 16,

To evaluate the complexity function ϕ ( r ) , we use known bounds on the covering number of ReLU neural networks.

Lemma 20 (Schmidt-Hieber (2020)) . For S ⊂ Φ( L, W, S, B ) , the covering number log N ( ε, S , ∥ · ∥ ∞ ) satisfies

<!-- formula-not-decoded -->

Combining Lemma 20 with the Dudley integral gives the following bound on the local Rademacher complexity.

Lemma 21. For ϕ ( r ) defined in Proposition 17,

<!-- formula-not-decoded -->

By combining these local Rademacher complexity controls, we obtain the proof of Theorem 1.

Proof of Theorem 1. From Lemmas 15 and 19, with probability at least 1 -2 e -t the following inequality holds:

<!-- formula-not-decoded -->

According to Lemma 13, the model bias term r ∗ can be bounded as

<!-- formula-not-decoded -->

Furthermore, from Lemma 21, the generalization error ̂ r is bounded as

<!-- formula-not-decoded -->

The network parameters satisfy the following upper bounds: L = O (log 2 ε -1 0 ∨ log 2 Mε -1 1 ∨ log ε -1 2 ) , ∥ W ∥ ∞ = O (log 3 ε -1 0 ∨ M log 3 Mε -1 1 ) , S = O (log 4 ε -1 0 ∨ M log 5 Mε -1 1 ∨ log ε -1 2 ) , B = O ( M 2 ∨ ε -2 0 ∨ log Mε -1 1 ) . To balance the size of each term, we set ε 0 = M nD ∧ 1 R , ε 1 = M 3 n 3 D 3 ∨ M nD , ε 2 = M nD . Given that D = O ( M ) , the desired bound follows.

## D Proof of Theorem 3

In this section, we provide the proof of Theorem 3. While the subsequent analysis closely parallels the arguments in Appendix C, the key distinction here lies in the use of function approximation theory in anisotropic Besov spaces.

## D.1 Approximation thoery in anisotropic Besov spaces

Here, we give the function approximation method in an anisotropic Besov class by deep neural networks. We define the affine composition model, which composes affine transformations with functions in an anisotropic Besov space:

<!-- formula-not-decoded -->

Under this model, the following bound on the approximation error is known.

Proposition 22 (Suzuki and Nitanda (2021)) . Suppose x follows the uniform distribution on Ω = [0 , 1] D , and define ˜ x = Ax + b ∈ R , which is assumed to have a bounded density supported on [0 , 1] d . Suppose there exists a constant C such that ∥ A ∥ ∞ ∨∥ b ∥ ∞ ≤ C . Then for 0 &lt; p, q, r ≤ ∞ and β ∈ R d ++ satisfying ˜ β &gt; (1 /p -1 /r ) + , the following approximation error bound holds:

<!-- formula-not-decoded -->

where R r ( F , H ) := sup f ∗ ∈H inf f ∈F ∥ f ∗ -f ∥ L r (Ω) denotes the worst-case approximation error. Here, the network parameters are set as

<!-- formula-not-decoded -->

where ϵ = N -˜ β log( N ) -1 , and c ( d,m ) is a constant depending only on d and m .

## D.2 Approximation error bound

Under Assumptions Assumption 5 to 7, we construct a neural network that approximates the score function s ◦ t ( x, y ) .

Lemma 23. For every k ∈ [1 , . . . , M ] , there exists a neural network ϕ 1 cont ( x, t ) ∈ Φ( L, W, S, B ) such that

<!-- formula-not-decoded -->

The network parameters satisfy:

<!-- formula-not-decoded -->

where the functions f 1 ( k ) and f 2 ( k ) are defined as:

<!-- formula-not-decoded -->

for 1 ≤ k ≤ M and f 2 ( M ) = 0 .

Proof. We aim to approximate the function Mp t ( x ) = M ∑ M j =1 c j u j ( x ) e -tλ j using a neural network by truncating the sum to a fixed number of terms. Specifically, we consider p ∗ t ( x ) = ∑ k j =1 c j u ∗ j ( x ) e -tλ j , where each u ∗ j ∈ H aff . By Proposition 22, there exists a neural network ϕ 0 j ∈ Φ( L 1 ( d ) , W 1 ( d ) , S 1 ( d ) , γ j B 1 ( d )) such that

<!-- formula-not-decoded -->

Following the same strategy as in Lemma 11, we can construct a neural network to approximate the exponential decay e -tλ j . We define ϕ ∗ j ( t ) as follows:

<!-- formula-not-decoded -->

The product u ∗ j ( x ) · e -tλ j can be approximated using ϕ ∗ j ( t ) and ϕ 0 j , and by summing over j = 1 , . . . , k and multiplying by M , we can construct the neural network ϕ 1 cont ∈ Φ( L, W, S, B ) satisfying ∥ ϕ 1 cont -Mp ∗ t ∥ ≤ N -˜ β ∑ k j =1 γ j j -s . Here, we set the approximation error of ϕ ∗ j ( t ) and ϕ mult to ε = M -2 N -˜ β . According to Lemmas 7 and 9, the parameters of ϕ 1 cont can be bounded as follows:

<!-- formula-not-decoded -->

Now, the total error in approximating p t ( x ) is given by

<!-- formula-not-decoded -->

For the sum ∑ k j =1 j -( s -γ ) and ∑ M j = k +1 j -s , we have

<!-- formula-not-decoded -->

Since c 1 = 1 / √ M , we can achieve the desired upper bound.

Lemma 24. For any 0 &lt; ε 0 ≤ R -1 , there exists a neural network ϕ 2 cont ( x, t ) ∈ Φ( L, W, S, B ) such that

<!-- formula-not-decoded -->

with network parameters bounded as:

<!-- formula-not-decoded -->

Here, f 1 ( k ) and f 2 ( k ) are defined in Lemma 23.

Proof. As in Lemma 12, we can construct ϕ 2 cont ( x, t ) = ϕ rec ◦ ϕ 1 cont ( x, t ) to achieve the desired approximation.

Lemma 25. Let 0 &lt; ε 0 ≲ R -1 and N ∈ N . Then, there exists a neural network ϕ cont ( x, y, t ) ∈ F such that

<!-- formula-not-decoded -->

The parameters of ϕ cont are bounded as follows:

<!-- formula-not-decoded -->

Here, f 1 ( k ) and f 2 ( k ) are defined in Lemma 23. Moreover, the following bound holds:

̸

<!-- formula-not-decoded -->

Proof. The proof proceeds in a similar manner to Lemma 13. We construct a network ϕ 3 cont ( x, y, t ) := [ ϕ 1 cont ( y, t ) , ϕ 2 cont ( x, t )] ⊤ by combining the neural networks that approximate Mp t ( y ) and 1 /Mp t ( x ) in parallel. Then, define the overall network as ϕ cont ( x, y, t ) := ϕ mult ◦ ϕ 3 ( x, y, t ) . This network estimates the true score s ◦ t ( x, y ) = p t ( y ) p t ( x ) , and the error can be bounded as follows:

<!-- formula-not-decoded -->

Setting ε 1 = ε 0 gives the desired approximation rate.

By the same argument as Lemma 13, we can bound the divergence as

̸

̸

<!-- formula-not-decoded -->

## D.3 Proof of the statement

By combining the results from Appendices C.2 and D.2, we are now ready to prove Theorem 3.

Proof of Theorem 3. By applying Lemmas 15 and 19, we obtain

<!-- formula-not-decoded -->

with probability 1 -2 e -t . By Lemmas 21 and 25, the model bias term r ∗ and the generalization error ̂ r can be bounded as:

<!-- formula-not-decoded -->

By setting N = O ( nk -1 f 1 ( k ) 2 ) 1 1+2 ˜ β and ε 0 = O ( n -1 / 2 ) , we have

<!-- formula-not-decoded -->

We consider balancing the first and second terms as follows:

1. When s ≤ 1 (and thus s -γ ≤ 1 ): By setting k = M , we obtain

<!-- formula-not-decoded -->

2. When 1 &lt; s and s -γ ≤ 1 : By setting k = ( λ 1+2 ˜ β 2 n -˜ β ) -1 (1+2 ˜ β )( s -1)+(1 -( s -γ )+ ˜ β ) , we have that

<!-- formula-not-decoded -->

3. When 1 &lt; s and 1 &lt; s -γ : By setting k = ( n -˜ β λ 1+2 ˜ β 2 ) 1 1+ ˜ β -s -2 s ˜ β , we have

<!-- formula-not-decoded -->

## E Proof of Corollary 1

## E.1 Proof of Lemma 4

Before proceeding to the main proof, we present a preparatory result on the spectral properties of the graph Laplacian in Lemma 4. Since h w ( x ) = ( -1) w ⊤ x , for every w ∈ { 0 , 1 } D , the function h w : { 0 , 1 } D →{-1 , 1 } is an eigenfunction of the adjacency matrix A defined on { 0 , 1 } D . Let the state v i ∈ { 0 , 1 } D satisfy d ( v i , x ) = 1 in terms of the Hamming distance, where the index i indicates the coordinate at which v i and x differ. Then, we obtain

<!-- formula-not-decoded -->

which implies h w is an eigenfunction of A with the eigenvalue D -2 | w | . Since L = DI -A , where I denotes the identity matrix, it follows that h w is also an eigenfunction of L with eigenvalue 2 | w | , and in particular, λ 2 = 2 .

## E.2 Main proof

First, we show h w ( x ) := cos( πw ⊤ x ) / √ M belongs to the Sobolev class H β ([0 , 1] D ) for any constant β ∈ N . Sobolev spaces are defined by

<!-- formula-not-decoded -->

Here α = ( α 1 , . . . α D ) ∈ N D 0 . Using the chain rule, we obtain

<!-- formula-not-decoded -->

The term p α is a finite linear combination of trigonometric functions and | p α | = O (1) . In particular, we have the bound | D α h w | ≲ O ( | w | | α | / √ M ) . Thus, the Sobolev norm of h w is bounded as

<!-- formula-not-decoded -->

This implies that h w ∈ H β ([0 , 1] D ) and, in particular, h w lies in the scaled Sobolev unit ball: there exists a constant γ w &gt; 0 such that h w ∈ γ w U ( H β ) , where U ( H β ) denotes the unit ball in H β ( R D ) . Since H β = B β 2 , 2 in the sense of Besov spaces, Theorem 4 can be applied directly. Specifically, for the function u ∗ j ( x ) = cos( πw ⊤ j x ) , Assumptions 5 and 6 hold with parameters ε = 0 , γ j = O (1) , and γ = 0 . Moreover, Lemma 4 ensures that λ 2 = O (1) . By choosing β sufficiently large, the desired bound follows from Theorem 4.

## F Proof of Corollary 2

Here, let ϕ j ( x ) be the eigenfunctions of the Laplace-Beltrami operator -∆ M normalized in L 2 ( M ) , and their corresponding eigenvalues are denoted by 0 = µ 1 &lt; µ 2 ≤ · · · , that is, -∆ M ϕ j = µ j ϕ j , and let σ ( -∆ M ) := { µ i } ∞ i =1 be the set of eigenvalues. For a vector u = ( u ( x )) x ∈ X , we define

<!-- formula-not-decoded -->

where N ( x ) := { y ∈ X | ∥ x -y ∥ ≤ σ } . Then, the following result is known:

Proposition 26 (Dunson et al. (2021)) . Suppose that M is sufficiently large such that

<!-- formula-not-decoded -->

where X 2 , X 3 &gt; 1 are constants depending on d and the volume, the radius, the curvature and the second fundamental form of the manifold M , and Γ k = min 1 ≤ j ≤ k dist( µ j , σ ( -∆ M ) \{ µ j } ) . Let σ = ( log( M ) M ) 1 4 d +13 . Then for 1 ≤ j ≤ K , it holds that

<!-- formula-not-decoded -->

Since -∆ M ϕ j = µ j ϕ j , the Sobolev norm of ϕ j for β ∈ N can be evaluated as

<!-- formula-not-decoded -->

Moreover, since it is known that µ j = Θ( j 2 /d ) (Hassannezhad et al., 2016), we obtain that

<!-- formula-not-decoded -->

This implies that we may choose γ = 2 β/d . Now suppose that β is chosen so that s -γ &gt; 1 . More precisely, we can approximate ϕ j by a composite function ˜ ϕ j ◦ ψ M where ψ M : R D → R d

represents a smooth function giving the local coordinate of M and ˜ ϕ j is a function on this local coordinate that corresponds to ϕ j . Then, we may think ˜ ϕ j is a function in a Besov space B β 2 , 2 on a compact domain of R d , and thus ˜ ϕ j ◦ ψ M can be approximated by ReLU deep neural networks with an error O ( j γ N -β/d ) .

Since p is the uniform distribution, a measure concentration inequality (Wainwright, 2019) implies

<!-- formula-not-decoded -->

uniformly over x ∈ X , with high probability, provided that σ ≥ ( log( M ) M ) 1 4 d +13 . In this event, we see that ∥ u ∥ ℓ 2 (ˆ p ) = Θ( ∥ u ∥ / √ M ) = Θ(1 / √ M ) . Then, by Eq. (6), Assumption 5 is satisfied for u ∗ j ( x ) = ∥ u j ∥ ℓ 2 (ˆ p ) ϕ j ( x ) ≃ ϕ j ( x ) / √ M with the approximation error

<!-- formula-not-decoded -->

To achieve the assertion in Theorem 4 with ˜ β = β/d , we choose k in its proof as

<!-- formula-not-decoded -->

which yields µ k = O ( n 2 /d ( d/β +2)( s -1)+1 ) . Hence, ( µ d/ 2+5 k ) 8 d +26 = O ( n (1+10 /d )(8 d +26) ( d/β +2)( s -1)+1 ) and ( µ (5 d +7) / 4 k ) 8 d +26 = O ( n (5 / 2+4 /d )(8 d +26) ( d/β +2)( s -1)+1 ) . We assume that M is sufficiently large to satisfy Eq. (5) with this choice of k , which ensures the approximation error bound above holds.

Based on these arguments, Theorem 4, with ˜ β = β/d and p = q = 2 , gives that

<!-- formula-not-decoded -->

which yields the assertion.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The target of the analysis (discrete diffusion models) is detailed in Section 2. The convergence analysis is given in Sections 3 and 4. Proof details are provided throughout the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See the conclusion section (Sec. 6).

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

Justification: The assumptions of the theoretical analyses are given in Sections 2.2 and Section 4 before we state our results. All complete proofs are provided throughout the appendix.

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

Justification: There is no numerical experiment in the paper.

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

Justification: There is no numerical experiment conducted in the paper.

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

Justification: This is a purely theoretical paper. There is no numerical experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: There is no numerical experiment.

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

Justification: There is no numerical experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors have checked that the research conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper is purely theoretical paper and no immediate societal impact is expected.

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

Justification: This is a purely theoretical paper.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

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

Justification: The core method development and theoretical analyses in this research do not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
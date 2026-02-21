## Optimal Spectral Transitions in High-Dimensional Multi-Index Models

Leonardo Defilippis 1 , Yatin Dandi 2,3 , Pierre Mergny 2 , Florent Krzakala 2 , and Bruno Loureiro 1

1 Département d'Informatique, École Normale Supérieure, PSL &amp; CNRS, Paris, France 2 Information, Learning and Physics Laboratory. EPFL, CH-1015 Lausanne, Switzerland. 3 Sloan School of Management, MIT, United States.

## Abstract

We consider the problem of how many samples from a Gaussian multi-index model are required to weakly reconstruct the relevant index subspace. Despite its increasing popularity as a testbed for investigating the computational complexity of neural networks, results beyond the single-index setting remain elusive. In this work, we introduce spectral algorithms based on the linearization of a message passing scheme tailored to this problem. Our main contribution is to show that the proposed methods achieve the optimal reconstruction threshold. Leveraging a high-dimensional characterization of the algorithms, we show that above the critical threshold the leading eigenvector correlates with the relevant index subspace, a phenomenon reminiscent of the Baik-Ben Arous-Peche (BBP) transition in spiked models arising in random matrix theory. Supported by numerical experiments and a rigorous theoretical framework, our work bridges critical gaps in the computational limits of weak learnability in multi-index model.

A popular model to study learning problems in statistics, computer science and machine learning is the multi-index model , where the target function depends on a low-dimensional subspace of the covariates. In this problem, one aims at identifying the p -dimensional linear subspace spanned by a family of orthonormal vectors w ⋆, 1 , . . . , w ⋆,p ∈ R d from n independent observations ( x i , y i ) from the model:

<!-- formula-not-decoded -->

This formulation encompasses several fundamental problems in machine learning, signal processing, and theoretical computer science, including: (i) Linear estimation, where p = 1 and g ( z ) = z , (ii) Phase retrieval, where p = 1 and g ( z ) = | z | . (iii) Learning Two-layer neural networks, where p is the width and g ( z ) = ∑ j ∈ J p K a j σ ( z j ) for some non-linear activation function σ : R → R , or (iv) learning Sparse parity functions, where g ( z ) = sign ( ∏ j ∈ J p K z j ) .

A classical problem in statistics [1, 2, 3], the multi-index model has recently gained in popularity in the machine learning community as a generative model for supervised learning data where the labels only depend on an underlying low-dimensional latent subspace of the covariates [4, 5, 6].

Of particular interest to this work are spectral methods, which play a fundamental role in machine learning by offering an efficient and computationally tractable approach to extracting meaningful structure from high-dimensional noisy data. A paradigmatic example is the Baik-Ben Arous-Péché (BBP) transition [7], where the leading eigenvalue of a matrix correlates with the hidden signal - a phenomenon that is ubiquitous in machine learning theory. Beyond their practical utility, spectral methods often serve as a starting point for more advanced approaches, including iterative and nonlinear techniques. This leads us to the central question of this paper:

Can one design optimal spectral methods that minimizes the amount of data required for identifying the hidden subspace in multi-index models?

While the optimal spectral method for single-index models are well understood [8, 9, 10, 11], and their optimality in terms of weak recovery threshold established [12, 9] their counterparts for multi-index models remain largely unexplored. This gap is particularly important, as multi-index models serve as a natural testbed for studying the computational complexity of feature learning in modern machine learning, and have attracted much attention recently [13, 5, 6, 14, 15, 16, 17].

Main contributions In this work, we step up to this challenge by constructing optimal spectral methods for multi-index models. We introduce and analyze spectral algorithms based on a linearized message-passing framework, specifically tailored to this setting. Our main contribution is to present two such constructions, establish the reconstruction threshold for these methods and to show that they achieve the provably optimal threshold for weak recovery among efficient algorithms [18].

Other Related works Recently, multi-index models have become a proxy model for studying non-convex optimization [19, 14, 15]. [20] has shown that the sample complexity of one-pass SGD for single-index model is governed by the information exponent of the link function. This analysis was generalized in several directions, such as to other architectures [16], larger batch sizes [21] and to overparametrised models [22]. A similar notion, known as the leap exponent , was introduced for multi-index models, where it was shown that different index subspaces might be learned hierarchically according to their interaction [23, 13, 24, 25]. The picture was found to be different for batch algorithms exploiting correlations in the data [26, 27, 28], achieving a sample complexity closer to optimal first order methods [12, 29, 18]. [18] in particular, provided optimal asymptotic thresholds for weak recovery within the class of first order methods.

Spectral methods are widely employed as a warm start strategy for initializing other algorithms, in particular for iterative schemes (such as gradient descent) for which random initialization is a fixed point. Relevant to this work is the class of approximate message passing (AMP) algorithms, which have garnered significant interest in the high-dimensional statistics and machine learning communities over the past decade [30, 31, 32, 33]. AMP for multi-index models was discussed in [4, 18, 34, 35]. Spectral initialization for AMP in the context of single-index models has been studied by [36].

The interplay between AMP and spectral methods has been extensively studied in the literature, see for example [37, 38, 39, 9, 11, 10, 40]. In particular the idea of using message passing algorithm to derive spectral method has been very successful, leading to the non-backtracking matrix [41] and more recently to the Kikuchi hierarchy [42, 43], and to non-linear optimal spectral methods for matrix and tensor factorization [38, 44, 45, 46].

During the finalization of this work, an independent paper analysing a family of spectral estimators indexed by a pre-processing function T for Gaussian multi-index models [47] appeared. Their results leverage tools from random matrix theory to characterize the asymptotic spectral distribution of this family, as well as the correlation of the top eigenvectors with the indices. They prove that a tailored T asymptotically achieves the optimal weak recovery threshold of [18] in the particular case where G ( y ) is jointly diagonalizable for all y , providing an alternative proof of Conjecture 2.10 and Theorem 2.11 for this particular case.

## 1 Setting and Notations

Notations To enhance readability, we adopt the following consistent notations through the paper: scalar quantities are denoted using non-bold font (e.g., a or A ), vectors are represented in bold lowercase letters (e.g., a ), and matrices are written in bold uppercase letters (e.g., A ). We further differentiate random variables depending on the noise with non-italic font (e.g, a , a , A ). We denote by ⟨ a , b ⟩ the standard Euclidean scalar product, ∥ a ∥ the ℓ 2 -norm of a vector a , ∥ A ∥ op the operator norm of a matrix A and ∥ A ∥ F its Frobenius norm. Given n ∈ N , we use the shorthand J n K = { 1 , . . . , n } . We denote S p + the cone of positive semi-definite p × p matrices. ( x ) + := max(0 , x ) .

The Gaussian Multi Index Model We consider the supervised learning setting with n i.i.d. samples ( x i , y i ) i ∈ J n K with covariates x i ∼ N ( 0 , d -1 I d ) ∈ R d and labels y drawn conditionally from

the Gaussian multi-index model defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where W ⋆ := ( w ⋆, 1 , . . . , w ⋆,p ) ∈ R d × p is a weight matrix with independent columns with w ⋆,j ∼ N ( 0 , I d ) , j ∈ J p K and P ( ·|· ) is a conditional probability distribution. Additionally, we define the link function g : R p ∋ z ↦→ g ( z ) ∈ R as the conditional mean and the labels' marginal distribution Z ( y ) = E W ⋆ E x ∼ N ( 0 ,d -1 I d ) [ P (y = y | W T ⋆ x )] . Note that, in the limit d →∞ , for x ∼ N ( 0 , d -1 I ) , the Central Limit Theorem implies W T ⋆ x ∼ N ( 0 , I p ) and Z ( y ) = E z ∼ N ( 0 , I p ) [ P (y = y | z )] .

We investigate the problem of reconstructing W ⋆ in the proportional high-dimensional limit

<!-- formula-not-decoded -->

In particular we are interested in the existence of an estimator ˆ W that correlates with the weight matrix W ⋆ better than a random estimator. This is formalized as follows.

Definition 1.1. (Weak subspace recovery) Given an estimator ˆ W of W ⋆ with ∥ ˆ W ∥ 2 F = Θ( d ) , we say we have weak recovery of a subspace V ⊂ R p , if with high probability.

<!-- formula-not-decoded -->

Computational bottlenecks for weak recovery in the Gaussian multi-index models have been studied by [18] using an optimal generalized approximate message passing (GAMP) scheme, see Appendix A for a detailed discussion of the algorithm. In particular, for the appropriate choice of denoiser functions, given in eq. (43), AMP is provably optimal among first-order methods [48, 49]. Their work provides a classification of the directions in R p in terms of computational complexity of their weak learnability. In particular, if and only if

E z ∼ N ( 0 , I p ) [ z P (y | z )] ∝ E [ z ∣ ∣ y] = 0 (6) almost surely over y ∼ Z , the subspace of directions that can be learned in a finite number of iterations is empty, for AMP randomly initialized. Nonetheless, if the initialization contains an arbitrarily small (but finite) amount of side-information about the ground truth weights W ⋆ , AMP can weakly recover a subspace of R p , provided that α &gt; α c , where the critical sample complexity is characterized in Lemma 1.2.

Lemma 1.2. [18], Stability of the uninformed fixed point and critical sample complexity] If M = 0 ∈ R p × p is a fixed point of the state evolution associated to the optimal GAMP (43), then it is an unstable fixed point if and only if ∥F ( M ) ∥ F &gt; 0 and n &gt; α c d , where the critical sample complexity α c is:

with and G (y) := E z ∼ N ( 0 , I p ) [ zz T -I p | y] . Moreover, if ∥F ( M ) ∥ F = 0 , then M = 0 is a stable fixed point for any n = Θ( d ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The aim of our work is to close this gap, providing an estimation procedure that achieves weak recovery at the same critical threshold α c defined in eq. (7), but crucially does not require an informed initialization . In what follows we restrict the problem defined in (3) to the set of link functions satisfying eq. (6). These functions (said to have a generative exponent 2 in the terminology of [29]) covers a large class of the relevant problems, with the exception of the really hard functions such as sparse parities, which cannot be solved efficiently with a linear in d number of samples [18].

Throughout this manuscript we adopt the notation a = vec( A ∈ R b × p ) , b = n, d , for the vector in R bp with components a ( iµ ) = A iµ , i ∈ J b K , µ ∈ J p K , where the double index ( iµ ) ∈ J b K × J p K is a shorthand for the scalar index i +( µ -1) b . Similarly, we say that mat( a ) = A ⇐⇒ vec( A ) = a .

<!-- formula-not-decoded -->

We can now introduce the Spectral Methods we aim to investigate. Given a matrix X ∈ R n × d with rows x i ∼ N ( 0 , d -1 I d ) and a vector of labels y ∈ R n with elements sampled as in (3), define ˆ G ∈ R np × np as and the following two spectral estimators ˆ W L , ˆ W T of the weight matrix W ⋆ as

## 1. Asymmetric spectral method:

where ω 1 ∈ R np is the eigenvector corresponding to the eigenvalue γ L 1 with largest real part of the matrix L ∈ R np × np defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. Symmetric spectral method:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where w 1 ∈ R dp is the eigenvector associated to the largest eigenvalue γ T 1 of the symmetric matrix T ∈ R dp × dp defined as

Note that the constant N is arbitrary and can be chosen to fix d -1 ∥ ˆ W L ∥ 2 F = d -1 ∥ ˆ W T ∥ 2 F = N .

At first glance, these two spectral estimators may appear ad hoc or lacking a clear theoretical justification. However, they can be motivated by a linearization of the optimal GAMP algorithm [18] around the non-informative fixed point, which reads:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting the second equation into the first, this is equivalent to vec ( Ω t +1 ) = L vec ( Ω t ) . Moreover, assuming convergence, 1 one ends up with vec ( ˆ W ) = T vec ( ˆ W ) .

This suggests that at first leading order in the estimates, the dynamics of the algorithm is governed by power-iteration (and respectively a variant of it) on the matrix L (respectively the symmetric matrix T ) defined previously. As power iteration converges under mild assumptions to the top (matrix) eigenvector of the tensor, this further suggests to use the top eigenvectors ˆ W L , ˆ W T of the corresponding matrices as an estimate for the weight matrix W ⋆ .

Additional details on the linearized GAMP are reported in Appendix A.1. Similar approaches have been thoroughly investigated in the context of single-index models [9, 10], community detection [41, 37], spiked matrix estimation [38, 39], where they have been provably shown to provide a non-vanishing correlation with the ground truth exactly at the optimal weak recovery threshold. It is interesting to notice that the spectral estimators ˆ W L and ˆ W T correspond to the generalization for multi-index models of the spectral methods derived in [10], respectively from the linearization of the optimal Vector Approximate Message Passing [50, 51] and the Hessian of the TAP free energy associated to the posterior distribution for the weights [37]. In particular, for p =1 , the matrices proposed in our manuscript exactly reduce to the two ones (called respectively "TAP" and "LAMP") investigated in [10].

1 Here and in the rest of the paper, we drop the time index t to refer to the quantities at convergence.

<!-- image -->

Re

λ

Re

λ

Figure 1: Distribution of the eigenvalues (dots) λ ∈ C of L at finite n = 10 4 , for g ( z 1 , z 2 ) = z 1 z 2 , α c ≈ 0 . 59375 . ( Left ) α = 0 . 4 &lt; α c . ( Right ) α = 1 &gt; α c . The dashed blue circle has radius equal to √ α / α c , i.e. the value γ b predicted in Theorem 2.4. The dashed orange vertical line corresponds to Re λ = α / α c , the eigenvalue γ s defined in Theorem 2.3. As predicted by the state evolution equations for this problem, two significant eigenvalues (highlighted in orange) are observed near this vertical line.

## 2 Main Technical Results

In order to characterize the weak recovery of the proposed spectral methods, we define two message passing schemes tailored to respectively have eigenvectors of L and T as fixed points. Similarly to these previous works, we leverage the state evolution associated to the algorithms in order to quantify the alignment between the spectral estimators and the weight matrix W ⋆ , tracking the overlap matrices and their value M , Q , at convergence. The state evolution equations for generic linear GAMP algorithms are presented in Appendix A.2.

<!-- formula-not-decoded -->

## 2.1 Asymmetric spectral method

Definition 2.1. For γ &gt; 0 , consider the linear GAMP algorithm (40,41)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When γ is chosen as an eigenvalue of L , the correspondent eigenvector ω ∈ R np is a fixed point of eq. (17) for Ω t = mat( ω ) . In the high-dimensional limit, the asymptotic overlaps of the estimator ˆ W can be tracked thanks to the state evolution equations, which follow from an immediate application of the general result of [52] to the GAMP algorithm (2.6).

Proposition 2.2 (State evolution [52]) . Let M t and Q t denote the overlaps defined in eq. (16) for the iterative algorithm (2.1) . Then, in the proportional high-dimensional limit n, d → ∞ at fixed α = n / d , they satisfy the following state evolution :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

With the state evolution equations in hand, one can derive a sharp characterization of the asymptotic weak recovery threshold in terms of the spectral properties of the estimator by a linear stability argument [53]

Theorem 2.3. For α &gt; α c , γ s = α / α c is the largest value of γ such that the state evolution 2.2 has a stable fixed point ( M , Q ) with M = 0 , Q ∈ S p + \ { 0 } . Additionally, M ∈ S p + \ { 0 } .

̸

Theorem 2.4. For all α ∈ R , γ b = √ α / α c is the largest value for γ such that the state evolution 2.2 has a fixed point ( M , Q ) with M = 0 , Q ∈ S p + \ { 0 } . The fixed point is stable for α &lt; α c and unstable otherwise.

The derivation of the Theorems is outlined in Appendix B, where we further show that the iterations of Algorithm 2.1 correspond to the power-iteration on the operator L , normalized by γ . Assuming the convergence of Algorithm 2.1 in O (log d ) iterations, Theorem 2.3 thus implies that for α &gt; α c the top-most eigenvector of L achieves a non-vanishing overlap along W ⋆ , with the eigenvalues converging to γ s . Analogously, Theorem 2.4 implies that for α &lt; α c , the top-most eigenvector of L has a vanishing overlap along W ⋆ , with the eigenvalues converging to γ b .

Thus, the above two results indicate a change of behavior of the operator norm of L at the critical value α c , leading to the following conjecture (a detailed sketch can be found in Appendix D):

Conjecture 2.5. In the high-dimensional limit n, d →∞ , n / d → α , the empirical spectral distribution associated to the pn eigenvalues of L converges weakly almost surely to a density whose support is strictly contained in a disk of radius γ b = √ α / α c centered at the origin. Moreover

- for α &lt; α c , ∥ L ∥ op a.s. - - - - → n →∞ γ b and the associated eigenvector is not correlated with W ⋆ ;
- for α &gt; α c , ∥ L ∥ op a.s. - - - - → n →∞ γ s &gt; γ b and the associated eigenvector defined in (10), weakly recovers the signal.

This conjecture, motivated by the results (2.3-2.4) is perfectly supported by extensive simulations, as illustrated in Fig. 1, 2. An entire rigorous proof requires, however, a fine control of the spectral norm of these operators, which is a notably difficult problem in random matrix theory. We emphasize that the asymptotic characterization of the asymmetric spectral estimator is a novelty aspect of this work, even in the single-index model setting p = 1 .

## 2.2 Symmetric spectral method

In order to simplify the notation, define the symmetric matrices T ( y ) ∈ R p × p and ˆ G t T ∈ R np × np

Definition 2.6. Consider the linear GAMP algorithm (40,41)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

a t a parameter to be fixed and γ t chosen, a posteriori, such that ∥ V ∥ op = 1 . Note that V t is symmetric at all times given a symmetric initialization V 0 .

In Appendix C we show that, for properly chosen a t , the fixed point for the iterate ˆ W t V t is the eigenvector of T with eigenvalue lim t →∞ a t γ t . The denoiser functions chosen for this GAMP algorithm are derived as a generalization of the ones in [54], where a similar approach has been used to characterize the recovery properties of spectral algorithms for structured single-index models.

<!-- image -->

α

Figure 2: Largest eigenvalues (magnitude, if complex) of the matrices L (triangles), n = 5000 , and T (circles), d = 5000 , for the function g ( z 1 , z 2 ) = z 1 z 2 , versus the sample complexity α . The orange and blue lines, respectively represent the values of the largest eigenvector α / α c and the edge of the bulk √ α / α c in Conj. 2.5 for the asymmetric spectral method. The green and purple line correspond to the values of the largest eigenvalue λ s and the edge of the bulk λ b in Conj. 2.10 for the symmetric spectral method.

Proposition 2.7 (State evolution [52, 9]) . Let M t and Q t denote the overlaps defined in eq. (16) for the iterative algorithm (2.6) . Then, in the proportional high-dimensional limit n, d →∞ at fixed α = n / d , they satisfy the following state evolution equations:

<!-- formula-not-decoded -->

where we have defined the operators

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The complete set of state evolution equations is displayed in Appendix C.1. For a = 1 , the fixed point for eq. (26) is V = I p and the operators F ( M ; 1 , I p ) and ˜ F ( M ; 1 , I p ) coincide with F ( M ) defined in (8), having the largest eigenvalue corresponding to α -1 c . As a → ∞ , V → I p and F ( · ; a, V ( a )) → 0 . Therefore, since the operator depends continuously on a ∈ [1 , ∞ ) , there exists a continuous function ν F 1 ( a ) for the largest eigenvalue of the operator as a function of the parameter a , with ν F 1 (1) = α -1 c and ν F 1 ( a →∞ ) → 0 . Hence, for all α &gt; α c , there exists a &gt; 1 such that ν F 1 ( a ) = α -1 . A similar argument applies to ˜ F and its largest eigenvalue ν ˜ F 1 ( a ) .

̸

Theorem 2.8. For α &gt; α c , consider a t = a &gt; 1 solution of ν F 1 ( a t ) = α -1 and γ t such that ∥ V ∥ op = 1 , with V t +1 given by eq. (26). Then, the state evolution has a stable fixed point ( M , Q ) with M = 0 , Q ∈ S p + \ { 0 } , corresponding to the eigenvalue λ s = aγ .

Theorem 2.9. For α ≥ α c , a t = a ≥ 1 solution to ν ˜ F 1 ( a ) = α -1 , γ t such that ∥ V ∥ op = 1 , with V t +1 given by eq. (26), the state evolution has an unstable fixed point ( M , Q ) with M = 0 , Q ∈ S p + \{ 0 } , corresponding to the eigenvalue λ b = aγ .

The derivation of the Theorems is outlined in Appendix C. Analogously to how Theorems 2.3, 2.4 motivate Conjecture 2.5, we show in Appendix C that Theorems 2.8, 2.9, along with a mapping to power-iteration, lead to the following conjecture:

Conjecture 2.10. In the high-dimensional limit n, d → ∞ , n / d → α , for α &gt; α c , the largest eigenvalue of T converges to λ s , defined in Theorem 2.8. In this regime, the symmetric spectral method, defined in (12), weakly recovers the signal. Moreover, the empirical spectral distribution of the pd eigenvalues of T converges weakly almost surely to a density upper bounded by λ b &lt; λ s defined in Theorem 2.9.

Figure 3: Distribution of the eigenvalues of T , d = 10 4 , for the link function g ( z ) = p -1 ∥ z ∥ 2 , p = 4 . The critical threshold in α c = 2 . The distribution is truncated on the left. ( Left ) α = 1 &lt; α c . ( Center ) α = α c . ( Right ) α = 6 &gt; α c . As predicted by the state evolution, we observe four eigenvalues (in green) separated from the main bulk, centered around λ s = 1 (green vertical line) obtained in Theorem 2.8. The vertical purple line correspond to the value λ b provided in Theorem 2.9 as a bound for the bulk.

<!-- image -->

We emphasize that, when p = 1 , the proposed symmetric estimator specializes to the method in [9, 55, 12] for the single-index model, and therefore benefits from a fully rigorous characterization of its weak recovery properties and spectral phase transitions. However, in multi-index models ( p &gt; 1 ), the matrix T exhibits by construction a highly structured form due to the presence of repeated entries from the measurement matrix X . This intrinsic redundancy complicates its analysis using standard random matrix theory tools. Conjecture 2.10 based on results (2.8-2.9) and further supported by numerical simulations (see Fig. 2, 3), offers a novel framework for understanding the spectral properties of such matrices.

<!-- formula-not-decoded -->

Moreover, for any model P ( ·| z ) such that G ( y ) = E [ zz T -I p ∣ ∣ y = y ] admits a common basis for all y , with real eigenvalues { λ k ( y ) } p k =1 , the analysis of spectrum of T can be simplified following the arguments in Appendix C.3.1. Indeed, the symmetric spectral method reduces to the diagonalization of p matrices

Their structure allows the use of the techniques in [9, 55] to analyze the spectrum, and supports the formalization of the results in Conjecture 2.10 for this subset of problems. In Appendix C.3 we prove the following result

Theorem 2.11. Assume that the matrix E [ zz T | y ] admits a basis of orthonormal eigenvectors independent of y . Then, in the high-dimensional limit n, d →∞ , n / d → α , for α &gt; α c , the largest eigenvalue of T converges to λ s = 1 . Moreover, the empirical spectral distribution of the pd eigenvalues of T converges weakly almost surely to a density upper bounded by λ b &lt; 1 defined in Theorem 2.9.

This class of models P ( ·| z ) includes several examples of interest, such as those depending on z through a quadratic form z ⊤ Az with deterministic A (e.g., the norm ∥ z ∥ 2 ), or through the product z 1 z 2 . . . z p (e.g., the embedded sparse parity).

## 2.3 Relation between the two spectral methods

As in the single-index setting, L and T are related by the following Proposition (proven in App. G).

Proposition 2.12. Define ˆ G R np × np such that ˆ G := δ G (y ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 4: Overlap ∥ M ∥ 2 F / Tr( Q ) as a function of the sample complexity α . The dots represent numerical simulation results, computed for n = 5000 (for the asymmetric method) or d = 5000 (for the symmetric method) and averaging over 10 instances. ( Left ) Link function g ( z 1 , z 2 ) = z 1 z 2 . Solid lines are obtained from state evolution predictions (Section 3). Dashed line at α c ≈ 0 . 59375 . ( Right ) Link function g ( z ) = p -1 ∥ z ∥ 2 , p = 4 . Solid lines are obtained from state evolution predictions (Section 3). Dashed line at α c = 2 .

<!-- image -->

2. Given an eigenpair γ T , w R dp of T and defining ω := ( I np + ˆ G ) -1 vec ( X mat( w )) , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, if there exists an eigenvector w of T with eigenvalue γ T = 1 , then ω = ( I np + ˆ G ) -1 vec ( X mat( w )) is an eigenvector of L with eigenvalue γ L = γ T = 1 .

In the setting of Theorem 2.11, where the largest eigenvalue of T is λ s = 1 , Proposition 2.12 implies that the principal eigenvector of L is not the only one correlated with the signal. There exists at least one additional informative eigenvector, whose eigenvalue is one and lies hidden within the bulk of radius √ α/α c &gt; 1 . This phenomenon is reminiscent of what has been observed for single-index models in [10].

## 3 Numerical illustrations

In this section we illustrate the framework introduced in Section 2 to predict the asymptotic performance of the spectral estimators (10,12) for specific examples of link functions, providing a comparison between our asymptotic analytical results and finite size numerical simulations for the overlap between the spectral estimators and the weights W ⋆ , defined as m := ∥ M ∥ F / √ Tr( Q ) , where M and Q are the overlap matrices defined in eq. (16) correspondent to the fixed points in Theorems 2.3, 2.4, 2.8, 2.9. In Figure 4 we compare these theoretical predictions to numerical simulations at finite dimensions, respectively for the link functions g ( z 1 , z 2 ) = z 1 z 2 and g ( z ) = p -1 ∥ z ∥ 2 . Additional numerical experiments are presented in Appendix E.

Asymmetric spectral method. We provide closed-form expressions for the overlap parameter m := ∥ M ∥ F / √ Tr( Q ) of the spectral estimator ˆ W L (10), for a selection of examples of link functions. The details of the derivation are given in Appendix E.

- g ( z ∈ R ) (single-index model):
- g ( z ) = p -1 || z || 2 :

<!-- formula-not-decoded -->

- g ( z ) = sign( z 1 z 2 ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- g ( z ) = ∏ p k =1 z k :

<!-- formula-not-decoded -->

where we have used and the previous expression are written in terms of the modified Bessel function of the second kind and Meijer G -function, with the notations 0 p ∈ R p = (0 , . . . , 0) T and e p ∈ R p = (0 , . . . , 0 , 1) T .

<!-- formula-not-decoded -->

- g ( z 1 , z 2 ) = z 1 z -1 2 :

<!-- formula-not-decoded -->

Symmetric spectral method. We provide expressions for the overlap parameter m := ∥ M ∥ F / √ Tr( Q ) of the spectral estimator ˆ W T (12), for a selection of examples of link functions. In all the following cases, the state evolution equations simplify, allowing to write the results as functionals of λ : R → R , specific to each problem:

- g ( z ∈ R ) (single-index model): λ ( y ) = Var [ z | y ] -1 ;
- g ( z ) = sign( z 1 z 2 ) : λ ( y ) = 2 y/π
- g ( z ) = p -1 ∥ z ∥ 2 : λ ( y ) = y -1 ;

For all these examples, the value α c has been reported in the previous paragraph. For α &gt; α c , consider a and γ solutions of

- g ( z ) = ∏ p k =1 z k : λ ( y ) defined in eq. (37).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for any α , the overlap m := ∥ M ∥ F / √ Tr( Q ) is given by which is strictly positive ∀ α &gt; α c . In all the example the principal eigenvector is λ s = 1 . Additional details can be found in Appendix F.

## 4 Conclusion and Perspectives

In this work, we tackled weak recovery in high-dimensional multi-index models via spectral methods, deriving two estimators inspired by a linearization of AMP. We showed they achieve the optimal reconstruction threshold, closing a key gap in prior approaches that required additional side information. Our analysis establishes that above the critical sample complexity, the leading eigenvectors of the proposed spectral operators align with the ground-truth subspace, echoing the BBP transition in random matrix theory. This work advances our understanding of weak subspace recovery in multi-index models and provides a principled framework for designing optimal spectral estimators. It bridges ideas from random matrix theory, approximate message passing, and neural feature learning.

Several directions remain open. A random matrix theory analysis of our spectral analysis - which requires a challenging control of the spectral norms - could be used to prove the two conjectures. Extending our AMP linearization to higher-order schemes, such as the Kikuchi hierarchy, may unlock insights into harder generative exponent problems, including the notorious sparse parity function. We hope this work sparks further research at the intersection of spectral methods and high-dimensional inference.

## Acknowledgements

The authors would like to thank Antoine Maillard, Benjamin Aubin and Lenka Zdeborova for early discussions on this problems in KITP Santa Barbara. We would also like to thank Filip Kovaˇ cevi´ c, Yihan Zhang and Marco Mondelli for the discussions on the relationship with their work following the first version of this manuscript. This research was supported in part by grant NSF PHY-2309135 to the Kavli Institute for Theoretical Physics (KITP), by the Swiss National Science Foundation under grant SNSF OperaGOST (grant number 200390) and DGIANGO (grant number 225837), and by the French government, managed by the National Research Agency (ANR), under the France 2030 program with the reference "ANR-23-IACL-0008" and the Choose France - CNRS AI Rising Talents program.

## References

- [1] Jerome H Friedman and Werner Stuetzle. Projection pursuit regression. Journal of the American statistical Association , 76(376):817-823, 1981.
- [2] Ming Yuan. On the identifiability of additive index models. Statistica Sinica , pages 1901-1911, 2011.
- [3] Dmitry Babichev and Francis Bach. Slice inverse regression with score functions. Electronic Journal of Statistics , 12(1):1507 - 1543, 2018.
- [4] Benjamin Aubin, Antoine Maillard, Florent Krzakala, Nicolas Macris, Lenka Zdeborová, et al. The committee machine: Computational to statistical gaps in learning a two-layers neural network. Advances in Neural Information Processing Systems , 31, 2018.
- [5] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory , pages 5413-5452. PMLR, 2022.
- [6] Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. How twolayer neural networks learn, one (giant) step at a time. Journal of Machine Learning Research , 25(349):1-65, 2024.
- [7] Jinho Baik, Gérard Ben Arous, and Sandrine Péché. Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. The Annals of Probability , 33(5):1643 - 1697, 2005.
- [8] Wangyu Luo, Wael Alghamdi, and Yue M. Lu. Optimal spectral initialization for signal recovery with applications to phase retrieval. IEEE Transactions on Signal Processing , 67(9):2347-2356, 2019.
- [9] Marco Mondelli and Andrea Montanari. Fundamental limits of weak recovery with applications to phase retrieval. In Proceedings of the 31st Conference On Learning Theory , volume 75 of Proceedings of Machine Learning Research , pages 1445-1450. PMLR, 06-09 Jul 2018.
- [10] Antoine Maillard, Florent Krzakala, Yue M Lu, and Lenka Zdeborová. Construction of optimal spectral methods in phase retrieval. In Mathematical and Scientific Machine Learning , pages 693-720. PMLR, 2022.
- [11] Marco Mondelli, Christos Thrampoulidis, and Ramji Venkataramanan. Optimal combination of linear and spectral estimators for generalized linear models. Foundations of Computational Mathematics , 22(5):1513-1566, Oct 2022.
- [12] Jean Barbier, Florent Krzakala, Nicolas Macris, Léo Miolane, and Lenka Zdeborová. Optimal errors and phase transitions in high-dimensional generalized linear models. Proceedings of the National Academy of Sciences , 116(12):5451-5460, 2019.
- [13] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics. In The Thirty Sixth Annual Conference on Learning Theory , pages 2552-2623. PMLR, 2023.

- [14] Luca Arnaboldi, Ludovic Stephan, Florent Krzakala, and Bruno Loureiro. From highdimensional &amp; mean-field dynamics to dimensionless odes: A unifying approach to sgd in two-layers networks. In The Thirty Sixth Annual Conference on Learning Theory , pages 1199-1227. PMLR, 2023.
- [15] Elizabeth Collins-Woodfin, Courtney Paquette, Elliot Paquette, and Inbar Seroussi. Hitting the high-dimensional notes: An ode for sgd learning dynamics on glms and multi-index models. Information and Inference: A Journal of the IMA , 13(4):iaae028, 2024.
- [16] Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. Foundations of Computational Mathematics , pages 1-84, 2024.
- [17] Berfin Simsek, Amire Bendjeddou, and Daniel Hsu. Learning gaussian multi-index models with gradient flow: Time complexity and directional convergence. arXiv preprint arXiv:2411.08798 , 2024.
- [18] Emanuele Troiani, Yatin Dandi, Leonardo Defilippis, Lenka Zdeborova, Bruno Loureiro, and Florent Krzakala. Fundamental computational limits of weak learnability in high-dimensional multi-index models. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics , volume 258 of Proceedings of Machine Learning Research , pages 2467-2475. PMLR, 03-05 May 2025.
- [19] Rodrigo Veiga, Ludovic Stephan, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborová. Phase diagram of stochastic gradient descent in high-dimensional two-layer neural networks. Advances in Neural Information Processing Systems , 35:23244-23255, 2022.
- [20] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. Journal of Machine Learning Research , 22(106):1-51, 2021.
- [21] Luca Arnaboldi, Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. Online learning and information exponents: The importance of batch size &amp; Time/Complexity tradeoffs. In Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 1730-1762. PMLR, 21-27 Jul 2024.
- [22] Luca Arnaboldi, Florent Krzakala, Bruno Loureiro, and Ludovic Stephan. Escaping mediocrity: how two-layer networks learn hard generalized linear models with sgd. arXiv preprint arXiv:2305.18502 , 2024.
- [23] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks. In Conference on Learning Theory , pages 4782-4887. PMLR, 2022.
- [24] Alberto Bietti, Joan Bruna, and Loucas Pillaud-Vivien. On learning gaussian multi-index models with gradient flow. arXiv preprint arXiv:2310.19793 , 2023.
- [25] Alireza Mousavi-Hosseini, Denny Wu, and Murat A Erdogdu. Learning multi-index models with neural networks via mean-field langevin dynamics. arXiv preprint arXiv:2408.07254 , 2024.
- [26] Yatin Dandi, Emanuele Troiani, Luca Arnaboldi, Luca Pesce, Lenka Zdeborova, and Florent Krzakala. The benefits of reusing batches for gradient descent in two-layer networks: Breaking the curse of information and leap exponents. In Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 999110016. PMLR, 21-27 Jul 2024.
- [27] Luca Arnaboldi, Yatin Dandi, Florent Krzakala, Luca Pesce, and Ludovic Stephan. Repetita iuvant: Data repetition allows sgd to learn high-dimensional multi-index functions. arXiv preprint arXiv:2405.15459 , 2024.
- [28] Jason D Lee, Kazusato Oko, Taiji Suzuki, and Denny Wu. Neural network learns lowdimensional polynomials with sgd near the information-theoretic limit. Advances in Neural Information Processing Systems , 37:58716-58756, 2024.

- [29] Alex Damian, Loucas Pillaud-Vivien, Jason D Lee, and Joan Bruna. Computational-statistical gaps in gaussian single-index models. arXiv preprint arXiv:2403.05529 , 2024.
- [30] David L Donoho, Arian Maleki, and Andrea Montanari. Message-passing algorithms for compressed sensing. Proceedings of the National Academy of Sciences , 106(45):18914-18919, 2009.
- [31] Mohsen Bayati and Andrea Montanari. The dynamics of message passing on dense graphs, with applications to compressed sensing. IEEE Transactions on Information Theory , 57(2):764-785, 2011.
- [32] Sundeep Rangan. Generalized approximate message passing for estimation with random linear mixing. In 2011 IEEE International Symposium on Information Theory Proceedings , pages 2168-2172. IEEE, 2011.
- [33] Alyson K Fletcher and Sundeep Rangan. Iterative reconstruction of rank-one matrices in noise. Information and Inference: A Journal of the IMA , 7(3):531-562, 2018.
- [34] Parthe Pandit, Mojtaba Sahraee-Ardakan, Sundeep Rangan, Philip Schniter, and Alyson K Fletcher. Matrix inference and estimation in multi-layer models. Journal of Statistical Mechanics: Theory and Experiment , 2021(12):124004, dec 2021.
- [35] Nelvin Tan and Ramji Venkataramanan. Mixed regression via approximate message passing. Journal of Machine Learning Research , 24(317):1-44, 2023.
- [36] Marco Mondelli and Ramji Venkataramanan. Approximate message passing with spectral initialization for generalized linear models. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics , volume 130 of Proceedings of Machine Learning Research , pages 397-405. PMLR, 13-15 Apr 2021.
- [37] Alaa Saade, Florent Krzakala, and Lenka Zdeborová. Spectral density of the non-backtracking operator on random graphs. Europhysics Letters , 107(5):50005, 2014.
- [38] Thibault Lesieur, Florent Krzakala, and Lenka Zdeborová. Constrained low-rank matrix estimation: Phase transitions, approximate message passing and applications. Journal of Statistical Mechanics: Theory and Experiment , 2017(7):073403, 2017.
- [39] Benjamin Aubin, Bruno Loureiro, Antoine Maillard, Florent Krzakala, and Lenka Zdeborová. The spiked matrix model with generative priors. Advances in Neural Information Processing Systems , 32, 2019.
- [40] Ramji Venkataramanan, Kevin Kögler, and Marco Mondelli. Estimation in rotationally invariant generalized linear models via approximate message passing. In International Conference on Machine Learning , pages 22120-22144. PMLR, 2022.
- [41] Florent Krzakala, Cristopher Moore, Elchanan Mossel, Joe Neeman, Allan Sly, Lenka Zdeborová, and Pan Zhang. Spectral redemption in clustering sparse networks. Proceedings of the National Academy of Sciences , 110(52):20935-20940, 2013.
- [42] Alexander S Wein, Ahmed El Alaoui, and Cristopher Moore. The kikuchi hierarchy and tensor pca. In 2019 IEEE 60th Annual Symposium on Foundations of Computer Science (FOCS) , pages 1446-1468. IEEE, 2019.
- [43] Jun-Ting Hsieh, Pravesh K Kothari, and Sidhanth Mohanty. A simple and sharper proof of the hypergraph moore bound. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 2324-2344. SIAM, 2023.
- [44] Amelia Perry, Alexander S Wein, Afonso S Bandeira, and Ankur Moitra. Optimality and suboptimality of pca i: Spiked random matrix models. The Annals of Statistics , 46(5):2416-2451, 2018.
- [45] Alice Guionnet, Justin Ko, Florent Krzakala, Pierre Mergny, and Lenka Zdeborová. Spectral phase transitions in non-linear wigner spiked models. arXiv preprint arXiv:2310.14055 , 2023.

- [46] Aleksandr Pak, Justin Ko, and Florent Krzakala. Optimal algorithms for the inhomogeneous spiked wigner model. Advances in Neural Information Processing Systems , 36, 2024.
- [47] Filip Kovaˇ cevi´ c, Yihan Zhang, and Marco Mondelli. Spectral estimators for multi-index models: Precise asymptotics and optimal weak recovery. arXiv preprint arXiv:2502.01583 , 2025.
- [48] Michael Celentano, Andrea Montanari, and Yuchen Wu. The estimation error of general first order methods. In Conference on Learning Theory , pages 1078-1141. PMLR, 2020.
- [49] Andrea Montanari and Yuchen Wu. Statistically optimal firstorder algorithms: a proof via orthogonalization. Information and Inference: A Journal of the IMA , 13(4):iaae027, 2024.
- [50] Philip Schniter, Sundeep Rangan, and Alyson K. Fletcher. Vector approximate message passing for the generalized linear model. In 2016 50th Asilomar Conference on Signals, Systems and Computers , pages 1525-1529, 2016.
- [51] Sundeep Rangan, Philip Schniter, and Alyson K. Fletcher. Vector approximate message passing. In 2017 IEEE International Symposium on Information Theory (ISIT) , page 1588-1592. IEEE Press, 2017.
- [52] Adel Javanmard and Andrea Montanari. State evolution for general approximate message passing algorithms, with applications to spatial coupling. Information and Inference: A Journal of the IMA , 2(2):115-144, 2013.
- [53] Steven H Strogatz. Nonlinear dynamics and chaos: with applications to physics, biology, chemistry, and engineering (studies in nonlinearity). Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering (Studies in Nonlinearity) , 2001.
- [54] Yihan Zhang, Hong Chang Ji, Ramji Venkataramanan, and Marco Mondelli. Spectral estimators for structured generalized linear models via approximate message passing (extended abstract). In Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 5224-5230. PMLR, 30 Jun-03 Jul 2024.
- [55] Yue M Lu and Gen Li. Phase transitions of spectral initialization for high-dimensional nonconvex estimation. Information and Inference: A Journal of the IMA , 9(3):507-541, 11 2019.
- [56] Cynthia Rush and Ramji Venkataramanan. Finite sample analysis of approximate message passing algorithms. IEEE Transactions on Information Theory , 64(11):7264-7286, 2018.
- [57] Gen Li and Yuting Wei. A non-asymptotic framework for approximate message passing in spiked models. arXiv preprint arXiv:2208.03313 , 2022.
- [58] Gen Li, Wei Fan, and Yuting Wei. Approximate message passing from random initialization with applications to z 2 synchronization. Proceedings of the National Academy of Sciences , 120(31):e2302930120, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims in the abstract and introduction are supported by mathematical proofs or numerical results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses the main limitations of the work, particularly the lack of a full random matrix theory analysis and the technical challenges that prevent such a treatment.

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

Justification: Each theoretical result is supported by a complete and correct proof, and all necessary assumptions are properly stated.

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

Justification: The paper provides all the information needed to reproduce the experimental results. In particular, the spectral methods are presented in Section 1, while the theoretical predictions are presented in Section 3.

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

Justification: We judge the code too simple to be released, and we provide enough information for the reproducibility of the numerical plots. All data sets used in the experiments are synthetic.

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

Justification: The experimental details are discussed in the captions.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The numerical experiments serve as illustrations of the theoretical results. The agreement is so good that error bars are unnecessary.

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

Justification: All experiments are simple enough to be run on a standard laptop in few hours.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This work is of theoretical nature, and therefore has no major ethical implications.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is of theoretical nature, and therefore has no relevant societal implications.

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

Justification: This work is of theoretical nature, and therefore has no risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All resources used from other works are properly acknowledged.

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

Justification: No new assets are introduced in the paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This work does not involve crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This work is of theoretical nature, and does not have potential risks for the people involved.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research does not involve LLMs as any important, original or non-standard component.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard component.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Generalized Approximate Message Passing algorithms

In this section we present a general version of the multi-dimensional Generalized Approximate Message Passing (GAMP) algorithm [32], defined as the iterations

<!-- formula-not-decoded -->

and ˆ W t +1 = f t +1 in ( B t +1 ) . The denoiser functions f t in : R p → R p and g t out : R p × R → R p are vector-valued mappings acting row-wise respectively on b j ∈ R p = B j. and ω i ∈ R p = Ω i. and the Onsager terms are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, the algorithm is uniquely determined by the choice of denoisers. For instance, the optimal GAMP for Gaussian multi-index models, derived in [4], is given by

<!-- formula-not-decoded -->

## A.1 Linear GAMP

In this manuscript we focus on a special type of GAMP algorithms that have linear denoiser functions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

namely where

A particular example of Linear GAMP is the one obtained linearizing the denoiser functions (43) around the uninformed fixed point of the algorithm b = 0 , ω = 0 and V = I p . We obtain a Linear GAMP with and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 State Evolution of Linear GAMP

<!-- formula-not-decoded -->

One of the main advantages provided by the Approximate Message Passing algorithm is the possibility to track the value of low-dimensional functions of the iterates at all finite times, in the high-dimensional limit, through a set of iterative equations denoted as state evolution [52]. In particular, we are interested to the following overlap matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

that characterize respectively the alignment between ˆ W t and the weights W ⋆ and the norm of ˆ W t . In this appendix we present the state evolution equations for the GAMP algorithm (45, 47) with linear denoiser functions, while we refer to [4] for a complete derivation in more general settings:

with the auxiliary matrices given by

<!-- formula-not-decoded -->

## B Proof Outlines for Theorems 2.3 and 2.4 - Asymmetric Spectral Method

Consider the following generalized power iteration algorithm

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

A t n →∞ - - - - → α E y ∼ Z [ E [ zz T -I ∣ ∣ y] ] = 0 (57) (as shown in (50)), and V t = γ -1 I . In fact, we can rewrite the generalized power iteration algorithm as in Definition 2.1

with ˆ G defined in eq. (9) and γ a parameter to be fixed. A pair ω = 0 , ˆ W = γ -1 X T mat ( ˆ Gω ) , is a fixed point of the above algorithm if and only if Lω = γ Ω . Thus, given the largest real γ such that the algorithm has a non-zero fixed point, the latter will correspond to the asymmetric spectral estimator in Definition 10. Interestingly, the above algorithm is also linear GAMP A.1, with denoiser functions g t out ( ω ∈ R p , y ) = E [ zz T -I | y = y ] ω and f t in ( b ) = γ -1 b , ∀ t , and Onsager terms

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.1 State Evolution

As an Approximate Message Passing algorithm, this algorithm enables to track low-dimensional functions of the iterate ˆ W t via the associated state evolution. Specifically, we will analyze the weak recovery properties of the asymmetric spectral method by studying the convergence of the state evolution equations A.2:

<!-- formula-not-decoded -->

where, recalling the notation G (y) = E [ zz T -I | y] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The linear operator F : R p × p → R p × p is symmetric, therefore it admits p 2 eigenpairs ( ν k ∈ R , M k ) k ∈ [ p 2 ] such that F ( M k ) = ν k M k and the (matrix) eigenvectors are an orthonormal basis of R p × p . In particular, Lemma 1.2 implies that ν 1 := max k ν k &gt; 0 and M 1 ∈ S p + . This eigenvalue corresponds to the inverse of the critical sample complexity α c , defined in 1.2. We can distinguish between two kind of fixed points:

̸

1. Informed fixed points. ( Theorem 2.3 ) ∀ α , for γ = αν k ( ν k = 0 ), M ∝ M k , is a non-zero fixed point for eq. (61). In particular, since the asymmetric spectral estimator is the fixed point correspondent to the largest γ , we are interested to the case M ∝ M 1 . We show now that ∥ M ∥ F = 0 ( i.e. the fixed point is actually informative) for α &gt; α c . Given

̸

γ = α / α c &gt; 1 , the largest eigenvalue of the operator γ -1 α F ( · ) is equal to 1 and any M ∝ M 1 is a stable fixed point. Moreover, eq. (62) at convergence:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in eq. (66) we used

<!-- formula-not-decoded -->

Therefore, the correspondent estimator eq. (60) weakly recovers W ⋆ . Note that, although it is beyond the scope of this work, the presented framework enables the identification of additional informative eigenvectors (with real eigenvalues smaller than α/α c ), emerging from the bulk at larger sample complexities.

ˆ W

2. Uninformed fixed point. ( Theorem 2.4 ) Initializing GAMP in a subspace orthogonal to the signal, i.e. M 0 = 0 , we have that M t = 0 at all times and eq. (62) at convergence

<!-- formula-not-decoded -->

Therefore, for γ = √ αν 1 , M = 0 , Q = M 1 ∈ S p + \ { 0 } is a fixed point of the state evolution. Note that, since the largest eigenvalue of γ -1 α F ( · ) in eq. (61) is √ α/α c , M = 0 is a stable fixed point for α ≤ α c , and unstable otherwise. Since the proposed GAMP is a generalized power iteration algorithm, normalized by the constant γ ∈ R , it converges only if γ corresponds to the absolute value of the eigenvalue with largest magnitude in the subspace of initialization.

## C Proof Outlines for Theorems 2.8 and 2.9 - Symmetric Spectral Method

Similarly to what we have done in the previous section, we introduce a GAMP algorithm that will serve as a framework to study the properties of the spectral estimator defined in (12). We stress that this algorithm does not offer any particular advantage for the practical computation of the spectral estimator compared to other spectral algorithms.

Consider the Generalized Approximate Message Passing algorithm defined by the denoiser functions

<!-- formula-not-decoded -->

where T ( y ) is the preprocessing function defined in (22) and a t , γ t parameters to be fixed.

We first show that, for suitable choiches of a t and γ t , the non-zero fixed point of this algorithm correspond to an eigenvector of T . Dropping the time index for the fixed-point variables and parameters, and defining ˆ G T ∈ R np × np as for i ∈ J n K , µ ∈ J p K , the fixed points satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used

Therefore, ˆ W = ˆ W T V -1 is a fixed point of the algorithm, for a t and γ t appropriately chosen, with eigenvalue given by a t γ t at convergence. 2

## C.1 State Evolution

The state evolution equations of the overlap matrices are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Note that F ( · ; a, V ) is a symmetric linear operator on the space of p × p matrices, with respect to the inner product ⟨ M , M ′ ⟩ := Tr( M T M ′ ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies that, for a, V fixed, F ( · ; a, V ) has p 2 real eigenvalues { ν k ( a, V ) } k ∈ J p 2 K and admits an orthonormal basis { M k ( a, V ) } k ∈ J p 2 K of eigenvectors in R p × p . Moreover, note that from the state evolution iterations, we can verify that V t = V T t = ⇒ V t +1 = V T t +1 , therefore, we consider the matrix V t to be symmetric at all times. From the state evolution equations at convergence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, in order to bound the operator norm of V , we choose so that ∥ V t →∞ ∥ op = 1 . We can distinguish the following cases:

2 Note that the overlap matrices for this algorithm refer to ˆ W T V -1 and not directly to the spectral estimator itself. However, if ∥ M ∥ F &gt; 0 , the weak recovery condition is satisfied.

- Informed fixed points. ( Theorem 2.8 ) Choosing a t such that

then ∃ M = 0 stable fixed point of the state evolution, and, for α &gt; α c ( a &gt; 1)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

- Uninformed fixed points. ( Theorem 2.9 ) Initializing GAMP with M 0 = 0 , which is a fixed point of the state evolution,

<!-- formula-not-decoded -->

Similarly to F ( · ; a, V ) , the symmetric operator ˜ F ( · ; a, V ) has p 2 real eigenvalues. Defining ν ˜ F 1 ( a ) as its largest one, we notice that ν ˜ F 1 (1) = α -1 c and ν ˜ F 1 ( a → ∞ ) → 0 . Therefore, for α &gt; α c , ∃ a &gt; 1 such that ν ˜ F 1 = α -1 and the state evolution has a fixed point M = 0 , Q ∈ S p + \ { 0 } . Moreover, for such a , the largest eigenvalue of F ( · ; a, V ) is larger than α -1 , hence the uninformed fixed point is unstable for α &gt; α c . This can be shown repeating a similar argument as the one we have applied in eq. (88). Since the GAMP convergence equations correspond to a generalized power iteration of T , normalized by the eigenvalue aγ , the instability of the uninformed fixed point implies that it is associated to an eigenvector smaller than λ s defined in Theorem 2.8.

## C.2 Sketch of derivation for Conjecture 2.10

These results for the fixed points of the state evolution justify Conjecture 2.10 on the weak recovery properties of the symmetric spectral estimator. The following argument is analogous to the one given for the asymmetric spectral method given in Appendix D. As a consequence of eq. 74, the Algorithm in Definition 2.6 behaves as the power-iteration on the matrix T . Suppose that λ 1 ( T ) -λ 2 ( T ) = ω ( d -κ ) for some κ &gt; 0 . Then, we obtain that w t = ( aγ ) -1 Tw t -1 converges to the top-most eigenvector of 1 aγ T in O (log d ) iterations.

Moreover, based on extensive confirmations in the literature [56, 57, 58], we conjecture that the asymetric spectral method is described by the state-evolution equations up to O (log d ) iterations.

̸

Theorem 2.8 predicts the convergence of the state evolution iterate M t to a fixed point corresponding to M = 0 for α &gt; α c . Since the fixed point conditions for the algorithm stipulate that w t = 1 λ s Tw t , we obtain under the validity of the state-evolution description:

and consequently

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, the equivalence to power-iteration implies that the LHS must converge to the top-most eigenvalue of T . We thus conclude that:

<!-- formula-not-decoded -->

## C.3 Random Matrix Theory analysis for the case where G ( y ) is jointly diagonalizable ∀ y

In this section, we will consider the setting of Thm. 2.11 which will analyzed using random matrix theory tools. We first describe in Sec. C.3.1 the setting and how the constants in Conjecture 2.10 simplified in that case and then sketch the outline of the proof using random matrix theory tools in Sec. C.3.2.

## C.3.1 Introduction to the model

For l ∈ J p K we denote by z l = λ l (y) / ( λ l (y) + 1) where y ∼ Z and for y ∈ Supp( Z ) , λ l ( y ) ≡ λ l ( G ( y )) is the l -th eigenvalue of G ( y ) = E z ∼ N (0 , I p ) [ zz T -I | y = y ] . Given the sample ( x i , y i ) i ∈ J n K , we let D l = Diag( { λ l (y i )( λ l (y i ) + 1) -1 } i ∈ J n K ) for l ∈ J p K . In the rest of this section we will restrict to a particular setting for the model introduced in the main text. We first recall the definition.

Definition C.1 (Jointly diagonalizable) . Let M ( . ) : R ⊃ I ∋ t ↦→ M ( t ) be a symmetric matrixvalued function. We say that M ( . ) is jointly diagonalizable if for all t ∈ I , we have M ( t ) = U Λ ( t ) U T , with Λ ( t ) diagonal and the orthogonal matrix U is constant with respect to t .

We will restrict to a subclass of our model such that one has

Assumption C.2. G ( . ) : Supp( Z ) ∋ y ↦→ G ( y ) is jointly diagonalizable.

Note that this subset of problems includes many cases of interest, including the ones considered in this manuscript, such as the monomial g ( z ) = ∏ k ∈ J p K z k , the norm g ( z ) = p -1 ∥ z ∥ 2 and the embedded sparse parity g ( z 1 , z 2 ) = sign( z 1 , z 2 ) .

By rotationally invariance of the hidden directions ( W ⋆ ( d ) = OW ⋆ for any orthogonal matrix O ) two jointly diagonalizable multi-index models specified by the conditional distributions P ( ·| z ) and P O ( ·| z ) := P ( ·| Oz ) are equivalent up to a change basis and in particular share the same α c . Indeed with Z O ( y ) := E z ∼ N (0 , I p ) P O ( y | z ) one can immediately check that Z O ( y ) = Z ( y ) and E z ∼ N (0 , I p ) [ z P O ( y | z )] = O T E z ∼ N (0 , I p ) [ z P ( y | z )] and E z [ zz T P O ( y | z )] = O T Cov[ z ∣ ∣ y ] O . As a consequence, we can set G ( y ) to be diagonal without loss of generality.

Next, we describe how the constants in Conjecture 2.10 simplifies under this jointly diagonalizable setting. To this end, we introduce the convex functions

<!-- formula-not-decoded -->

and let a α,l = arg min ψ α,l ( a ) be the minima, that is a α,l solves

<!-- formula-not-decoded -->

For later use, we also define we have the following result.

<!-- formula-not-decoded -->

Lemma C.3. Under Assumption C.2, the constants ( α c , λ b , λ c ) in Conjecture 2.10 further simplifies to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For Part-(i) since G ( y ) is diagonal, the critical threshold described in Lem. 1.2 is maximized over rank-one matrices, that is

<!-- formula-not-decoded -->

which gives by Courant-Fisher theorem the desired result for the threshold.

We next turn to the analysis of SE of Prop. 2.2 under the jointly diagonalizable assumption C.2. Setting T l (y) := λ l (y)( λ l (y) + 1) -1 , the fixed point equations read for any ( µ, ν ) ∈ J p K 2 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and γ = max µ ( 1 + α E y [ T µ (y)( a/V 2 µ -T µ (y)) -1 ]) . The candidated for the informative eigenvalues λ T µν = a µν γ of T correspond to the solutions of

<!-- formula-not-decoded -->

and are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that, for µ = ν , eq. 100 has always a solution for α &gt; α c,µ := E y [ λ µ (y) 2 ] , and ∀ µ ∈ J p K

<!-- formula-not-decoded -->

These eigenvalues are informative as

<!-- formula-not-decoded -->

Analogously, the edge of the bulk for each block of T can be obtained solving, for α &gt; α c,µ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the correspondent eigenvalues λ T ,b µ are given by

<!-- formula-not-decoded -->

from which we conclude Part-(ii) and (iii) of the Lemma.

## C.3.2 Outline of the proof using RMT

We give a proof of Conjecture 2.10 with the values for the threshold α c , the top outlier λ s = 1 and the edge λ b computed in Lem. C.3.

Assuming C.2, the symmetric spectral estimator introduced in Eq. (13) has the block-diagonal structure and thus its spectral properties can be immediately obtained from its diagonal block since we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, the top eigenvalue of T appearing in Conjecture 2.10 is simply the max of the top eigenvalue of each block ˜ T l and one can restrict to the study of the spectral properties of the ˜ T l , following closely the derivation of [55] which tackles the case p = 1 .

For each ˜ T l , we will partition the column of the sensing vector x i into parts that align with the hidden components w ⋆ and part that lives in the orthogonal complement, the only difference with the setting considered in [55] being in that now one has to deal with p hidden directions instead of one. To do so, we first use the Gram-Schmidt decomposition of the hidden matrix 1 √ d W ⋆ :

<!-- formula-not-decoded -->

where V = ( v 1 , . . . , v p ) is a semi-orthogonal matrix of dimension ( d × p ) and R ij = ⟨ u i , w j ) δ i ≤ j ∈ R p × p is upper triangular. Note that as w i,⋆ are iid standard Gaussian, as d →∞ we have R ii → 1 and R ij → 0 for i = j , exponentially fast. By rotationally invariance one has X ( d ) = O X for any O orthogonal matrix, hence one can fix v i = e i (the i -th canonical vector) without loss of generality. We decompose each vectors x i in this basis as follows

<!-- formula-not-decoded -->

or equivalently

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where K is by construction a matrix with iid Gaussian entries. From this partition, we can re-write each ˜ T l as

Lemma C.4 (Edge and existence of outliers) . As d → ∞ , the empirical spectral distribution of T l converges weakly almost surely to a distribution µ l with rightmost edge τ l := ψ α,l ( a α,l ) . Furthermore, for each l ∈ J p K , there exists up to p outliers in the spectrum of T l above this edge τ l .

Proof. This follows from the same proof of Proposition 3.1 in [55] (see Appendix A.4): by eigenvalue interlacing theorem, if put the eigenvalues in increasing order, we have for any i ∈ J ( d -p ) p K : λ ( ↗ ) i ( ˜ T l ) ≤ λ ( ↗ ) i ( P l ) ≤ λ ( ↗ ) i + p ( ˜ T l ) . Furthermore the empirical distribution of P l converges strongly to a limiting distribution µ l with rightmost edge τ l , since all but the top p eigenvalues of T l are trapped between eigenvalues of P l , one gets the desired result.

From Eq. (114), one immediately obtains that the empirical distribution of the full matrix T converges weakly almost surely to the distribution µ := 1 p ∑ l µ l whose rightmost edge is given by λ b in Lem. C.3. Next, following again [55], we map the position of the top outliers in each block T l to an additive spiked matrix model.

̸

Lemma C.5. Let ˜ P l ( µ ) = P l + Q l ( R l -µ I ) -1 Q T l denotes the family of spiked matrices indexed by a parameter µ ∈ R \ Spec( R l ) and set the function L l ( µ ) := λ 1 ( ˜ P l ( µ )) , where λ 1 denotes the highest eigenvalue, then we have λ 1 ( ˜ T l ) = L l ( µ ⋆ ) , where µ ⋆ is the unique fixed point of the equation L l ( µ ) = µ .

Proof. This follows from the determinant lemma in the same proof of Proposition 3.1 of [55] (see Section 3.2) with their (1 × 1) upper block replaced now by the ( p × p ) block R l .

Lemma C.6. Let ˜ P l ( µ ) as in Lem.C.5, then as d →∞ if the largest real solution (in a ) of

<!-- formula-not-decoded -->

exists and we denote it by a 1 ,l , we have L l ( µ ) → a 1 ,l and otherwise L l ( µ ) → τ l .

Proof. Applying the determinant lemma to the matrix ˜ P l ( µ ) , one finds that L l ( µ ) is characterized as the largest solution of the equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, by the law of large numbers, we have the following almost sure limits as d →∞ :

<!-- formula-not-decoded -->

where κ ∼ N ( 0 , I p ) . Substituting these limits into the determinant equation yields the asymptotic equation for the position of the top outlier L ( µ ) .

Combing Lemma C.5 with Lemma C.6, we can characterize the limiting position of the top outlier of T l in terms of ζ α,l : If the largest real solution (in a ) of exists and call it by a ∗ α,l then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Otherwise, if there is no such solution to Eq.(123), we have no outliers, that is

<!-- formula-not-decoded -->

As a consequence, the critical threshold α c for the appearance of an outlier in the full matrix T is given as the minimal value of α for which there exists a solution (in a ) of Eq. (123) as l ∈ J p K . To obtain explicitly this threshold, we first express Eq. (123) in terms of the G ( y ) = E z { κκ T -I | y = y } :

where we have replaced z l by its expression in the first expectation. Assuming now that α is such that there exists a solution a ∗ α,l &gt; a α,l of Eq. (126), by definition (95) of ζ α,l ( . ) , the latter can be replaced by ψ α,l . Next from Assumption C.2, G ( y ) is diagonal with eigenvalues { λ l ( y ) } l ∈ J p K and thus, each solution of Eq. (126) must solve for l ′ ∈ J p K :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the largest one is given for l ′ = l that is

and always exists for as long as α &gt; α c,l := ( E [ λ l (y) 2 ] ) -1 , from which one gets the desired result since α c = min l ∈ J p K α c,l .

To conclude, we need to show that for α &gt; α c , the asymptotic of max l ∈ J p K λ 1 ( T l ) given by Eq. (124), further simplifies to max l ∈ J p K λ 1 ( T l ) → 1 . Replacing ζ α,l by its expression, we have :

<!-- formula-not-decoded -->

since by assumption of our model, we must have E y λ l (y) = 0 , the latter can also be expressed as

<!-- formula-not-decoded -->

and since for α &gt; α c , the largest solution a ⋆ must solve Eq.(128), the expectation reduces to 1 /α such that one has

<!-- formula-not-decoded -->

which concludes the proof.

## D Sketch of the derivation of Conjecture 2.5

As we saw in Equations 55, the Algorithm in Definition 2.1 is equivalent to power-iteration on the matrix L . We may further suppose that Re ( λ 1 ( L )) -Re ( λ 2 ( L )) = ω ( d -κ ) for some κ &gt; 0 . For instance, κ = 2 3 under the Tracy-Widom scalings.

Assuming such a scaling for the spectral gap, we obtain that ω t = γ -1 Lω t -1 converges to the top-most eigenvector of 1 γ L in O (log d ) iterations.

Moreover, based on extensive confirmations in the literature [56, 57, 58], we conjecture that the asymetric spectral method is described by the state-evolution equations up to O (log d ) iterations.

̸

Theorems 2.3 and 2.4 predict the convergence of the state evolution iterate M t to a fixed point corresponding to M = 0 for α &lt; α c and M = 0 for α &gt; α c . Since the fixed point conditions for the algorithm stipulate that ω t = 1 γ s L ω t , we obtain under the validity of the state-evolution description:

and consequently

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, the equivalence to power-iteration implies that the LHS must converge to the top-most eigenvalue of L . We thus conclude that:

<!-- formula-not-decoded -->

## E Details on examples - Asymmetric spectral method

## E.1 Single-index models

The case of single-index models ( p = 1 ) allows for significant simplifications, as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This leads to the well known expression for the critical weak recovery threshold [12, 9, 55, 10, 29]

and to the following result for the overlap parameter m = M / √ Q

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E.2 G ( y ) jointly diagonalizable ∀ y

Following the arguments at the beginning of Appendix C.3, whenever we are in this setting, we can consider without loss of generality G ( y ) = E [ zz T -I | y = y ] = diag( λ 1 ( y ) , . . . , λ p ( y )) . The eigenpairs of the operator F eq. (63) are given by

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

with the critical sample complexity α -1 c = ν 1 := max k ∈ J p K ν ( k,k ) 3 . If the maximum is achieved by more than one pair of indices, we expect that the matrix L principal eigenvalue is degenerate, with degeneracy given by the cardinality of the set I = { ( k, h ) ∈ { 1 , ..., p } 2 | ν ( k,h ) = α -1 c } . Note that ∀ k, h , ν ( k,h ) = ν ( h,k ) and if ν ( k,h ) = max µ ∈{ k,h } ν µµ = ⇒ ν ( k,k ) = ν ( h,h ) . 4 The generic principal eigenvector of F is given by M = || M || F ∑ ( k,h ) ∈I c ( k,h ) M ( k,h ) with ∑ c 2 ( k,h ) = 1 . We introduce the ansatz: Q s.t. Q kk = 0 iff ( k, k ) ∈ I ; this implies Tr( F ( Q )) = ∑ k E [ λ k ( y ) 2 ] Q kk = α -1 c Tr( Q ) . Eq. (66) becomes

<!-- formula-not-decoded -->

As a special case, we consider the example λ k ( y ) = λ h ( y ) for all h, k such that ν ( k,k ) = ν ( h,h ) = ν 1 . One instance for this case is given by the link function g ( z ) = p -1 ∑ k ∈ J p K z 2 k . Then, defining λ (3) = E y [ λ k (y) 3 ] for any k | ( k, k ) ∈ I , the solution for m 2 does not depend on the coefficients c ( k,h ) and simplifies to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We consider now specific cases of link functions that are such that Cov[ z ∣ ∣ y ] is jointly diagonalizable ∀ y . We refer to [18] for the derivation of the expressions of Z ( y ) and Cov[ z ∣ ∣ y ] in all the examples contained in this Appendix E.2.

The above expression does not depend on the coefficients c ( k,h ) , therefore it is valid for all the degenerate directions in the principal eigenspace.

Note that, in general, if P ( ·| z ) = P ( ·| Qz ) , for Q orthogonal, then G ( · ) = QG ( · ) Q T . We use this property to find examples of jointly diagonalizable G ( · ) . Given any P ( ·| z ) depending on z through a quadratic form z T Az = 1 / 2 z T ( A + A T ) z , with A deterministic, and U , whose columns u µ are

3 Note that ∀ k, h ∈ p 2 , ν ( k,h ) ≤ max( ν ( k,k ) , ν ( h,h ) ) .

J K 4 Without loss of generality ν ( h,h ) ≤ ν ( k,k ) and ν ( k,k ) = ν ( k,h ) ≤ √ ν ( k,k ) ν ( h,h ) = ⇒ ν ( k,k ) ≤ ν ( h,h ) . Therefore ν ( k,k ) = ν ( h,h ) .

orthonormal eigenvectors of 1 / 2 ( A + A T ) , such models are invariant with respect to the orthogonal matrix Q ν = U diag ( ( ( -1) 2 -δ µ,ν ) µ ∈ J p K ) ) U T , for any ν ∈ J p K . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

̸

̸

Hence, G ( y ) is jointly diagonalizable with respect to the basis of eigenvectors U . Similarly, we can show that any model P ( ·| z ) depending on z through ∏ µ ∈ J p K z µ correspond to a jointly diagonalizable G ( · ) . In fact, if p = 2 , it depends on a quadratic form, while if p &gt; 3 it is invariant with respect to Q νκ = diag ( ( ( -1) 2 -δ µ,ν -δ µ,κ ) µ ∈ J p K ) ) , for any ν = κ . As in the previous example, we can show that G µν ( y ) = 0 , for any µ = ν and G ( y ) is diagonal.

E.2.1 g ( z 1 , . . . , z p ) = p -1 ∑ k ∈ J p K z 2 k

For a generic M ∈ R p × p

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

therefore α c = p / 2 and λ (3) = 8 /p 3 . Plugging these quantities in eq. (143), the overlap matrices at convergence satisfy

<!-- formula-not-decoded -->

E.2.2 g ( z 1 , z 2 ) = sign( z 1 z 2 )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The matrix E [ zz T -I | y] is jointly diagonalizable ∀ y , with eigenvalues λ 1 ( y ) = 2y π -1 and λ 2 (y) = -2y π -1 . Therefore, the eigenvalues of F are given by and

Leveraging eq. (142), the overlap matrices M and Q at convergence satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

E.2.3 g ( z 1 , . . . , z p ) = ∏ p k =1 z k For p = 2

<!-- formula-not-decoded -->

<!-- image -->

Re

λ

Re

λ

Figure 5: Distribution of the eigenvalues (dots) λ ∈ C of L at finite n = 5 · 10 3 , for g ( z 1 , z 2 ) = sign( z 1 z 2 ) , α c = π 2 / 4 . ( Left ) α = 1 . 4 &lt; α c . ( Right ) α = 7 &gt; α c . The dashed blue circle has radius equal to √ α / α c , i.e. the value γ b predicted in Theorem 2.4. The dashed orange vertical line corresponds to Re λ = α / α c , the eigenvalue γ s defined in Theorem 2.3. As predicted by the state evolution equations for this problem, two significant eigenvalues (highlighted in orange) are observed near this vertical line. Additionally, one can observe that our framework predicts other two degenerate eigenvalues at -γ s , here highlighted in cyan.

<!-- image -->

Figure 6: Distribution of the eigenvalues of T , d = 10 4 , for the link function g ( z 1 , z 2 ) = sign( z 1 z 2 ) . The critical threshold in α c = π 2 / 4 . The distribution is truncated on the left. ( Left ) α = α c . ( Right ) α = 7 &gt; α c . As predicted by the state evolution framework, in this regime we observe two eigenvalues separated from the main bulk, centered around λ s (green vertical line) obtained in Theorem 2.8. The vertical purple line correspond to the value λ b provided in Theorem 2.9 as a bound for the bulk.

where K n (y) is the modified Bessel function of the second kind. The matrix G ( y ) is jointly diagonalizable for all y , with eigenvalues λ 1 ( y ) = | y | K 1 ( | y | ) K 0 ( | y | ) +y -1 and λ 2 (y) = | y | K 1 ( | y | ) K 0 ( | y | ) -y -1 . Therefore, the eigenvalues of are given by 5

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Leveraging eq. (142), the overlap matrices M and Q at convergence satisfy

<!-- formula-not-decoded -->

If instead p ≥ 3 , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

F

and the previous expression are written in terms of the Meijer G -function. Therefore α -1 c = E y [ λ (y) 2 ] and, leveraging eq. (143), we obtain

<!-- formula-not-decoded -->

## E.3 A non-jointly diagonalizable case: g ( z 1 , z 2 ) = z 1 /z 2

If the matrix G ( y ) is not jointly diagonalizable ∀ y , there is not a general simplification for equations

(61,62), and each example needs to be treated separately.

In this section we consider the Gaussian multi-index model with link function g ( z 1 , z 2 ) = z 1 / z 2 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order to verify that both directions are not trivial , we need to compute E [ z | y] and verify that is zero almost surely over y ∼ Z :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality is the result of the integral of an odd function over a symmetric domain. In order to study the perfomance of the spectral method, we compute

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The eigenpairs of G ( y ) are λ 1 ( y ) = 1 , with eigenvector ( y, 1) T , and λ 2 ( y ) = -1 with eigenvector ( -1 , y ) T , which depends on y . Considering a generic M = ( m 1 m 2 m 3 m 4 ) , we have that

<!-- formula-not-decoded -->

therefore, the eigenpairs of F are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and α c = 1 . Moreover, one could easily verify that G ( M 1 ) = 0 . The overlap of the spectral estimator with the signal is therefore M ∝ I , and, from the state evolution eq. (62) at convergence, we have that where we leverage the symmetry of Q in eq. (171) to write F ( Q ) = 2 -1 Tr( Q ) I = ⇒ Tr( F ( Q )) = Tr( Q ) .

<!-- formula-not-decoded -->

Figure 7: Overlap ∥ M ∥ 2 F / Tr( Q ) as a function of the sample complexity α . The dots represent numerical simulation results, computed for n = 5000 (for the asymmetric method) or d = 5000 (for the symmetric method) and averaging over 10 instances. The link function is g ( z 1 , z 2 ) = z 1 /z 2 . Solid lines are obtained from state evolution predictions. Dashed vertical line at α c = 1 .

<!-- image -->

## F Details on examples - Symmetric spectral method

In all the considered examples with p ≥ 2 , the matrix E [ zz T | y = y ] admits a unique orthonormal basis of eigenvectors independent of y . Therefore, the state evolution equations can be significantly simplified following the same considerations applied in Appendices C.3 and E.2, to which we refer for the notation adopted in this appendix. Additionally, for all these examples, the eigenvalues λ k ( y ) of G ( y ) satisfy the additional conditions

<!-- formula-not-decoded -->

so that

J K It is easy to verify that these conditions implies that V = I , and the state evolution admits stable fixed point M , Q ∝ I , where the proportionality constants can be numerically computed through one-dimensional integrals, expressed in terms of λ 1 ( y ) (the choice of the eigenvalue is arbitrary in this setting) and given in eq. (39). Additionally, the largest eigenvalue of T is always equal to one and degenerate (see App. C.3).

<!-- formula-not-decoded -->

Proposition 2.12 readily implies that, for all these examples, the matrix L has a correspondent informative subspace of eigenvectors that can be computed from the subspace of leading eigenvectors of T with eigenvalue equal to 1 and hidden in the bulk.

## G Proof of Proposition 2.12

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->
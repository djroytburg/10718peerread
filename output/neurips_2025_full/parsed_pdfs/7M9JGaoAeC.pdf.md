## Infinite Neural Operators: Gaussian processes on functions

| Daniel Augusto de Souza ∗ University College London   | Yuchen Zhu College                          | Harry Jake Cunningham University College        |
|-------------------------------------------------------|---------------------------------------------|-------------------------------------------------|
|                                                       | University London                           | London                                          |
| Yuri Saporito Fundac ¸ ˜ ao Getulio Vargas            | Diego Mesquita Fundac ¸ ˜ ao Getulio Vargas | Marc Peter Deisenroth University College London |

## Abstract

A variety of infinitely wide neural architectures (e.g., dense NNs, CNNs, and transformers) induce Gaussian process (GP) priors over their outputs. These relationships provide both an accurate characterization of the prior predictive distribution and enable the use of GP machinery to improve the uncertainty quantification of deep neural networks. In this work, we extend this connection to neural operators (NOs), a class of models designed to learn mappings between function spaces. Specifically, we show conditions for when arbitrary-depth NOs with Gaussiandistributed convolution kernels converge to function-valued GPs. Based on this result, we show how to compute the covariance functions of these NO-GPs for two NO parametrizations, including the popular Fourier neural operator (FNO). With this, we compute the posteriors of these GPs in regression scenarios, including PDE solution operators. This work is an important step towards uncovering the inductive biases of current FNO architectures and opens a path to incorporate novel inductive biases for use in kernel-based operator learning methods.

## 1 Introduction

Neural Operators (NOs, Kovachki et al., 2023) are deep learning architectures designed to learn mappings between function spaces-with direct applications in many areas of science and engineering (Pathak et al., 2022; Li et al., 2024). NOs generalize conventional convolutional neural networks using kernel integral operators , which integrate the input function against a learnable kernel at each layer. Importantly, unlike CNNs, NOs can be trained with inputs of mixed, arbitrary resolutions and output predictions in discretizations of arbitrary granularity.

Despite their growing adoption, most works on NOs are primarily empirical, and most of the theoretical properties of NOs are still unexplored. In contrast, the convergence of Bayesian neural networks to Gaussian processes as their width goes to infinity has been amply studied (Neal, 1995; Novak et al., 2019; Yang, 2019). However, due to the infinite dimensionality of function spaces, it is unclear whether GPs are a limiting case for NOs and, if this is the case, how to characterize them.

In this work, we elucidate this question and present a set of assumptions that guarantee the existence of the infinite limit of NOs as Gaussian elements in the space of operators. Additionally, we present how to derive the covariance function for infinite-width NOs in an analogous fashion to the covariance functions of infinitely wide, densely connected NNs. Finally, we characterize the infinite-width limit of Fourier neural operators (FNOs) and propose a novel Bayesian NO architecture based on Mat´ ern GP-distributed integral kernels.

∗ Corresponding author: daniel.souza.21@ucl.ac.uk

Our experiments reinforce our theoretical results, showcasing the agreement between increasingly wide NOs and our derived expressions for the infinite limit at initialization. Additionally, we compare the performance of these models in a regression setting.

## 2 Background

This section provides a brief background on NOs (Section 2.1), along with basic notions of probability in Hilbert spaces (Section 2.2) and Gaussian processes on functions.

## 2.1 Operator learning and neural operators

Kovachki et al. (2023) propose neural operators, a family of parametrized operators. Recall that multilayer perceptrons transform vectors using successive layers of sums of linear transformations followed by element-wise non-linear activation functions. Analogously, Kovachki et al. (2023) define the building layers of neural operators (NOs) as sums of both point-wise linear operations and kernel integral operators, possibly followed by point-wise element-wise non-linear activation functions.

Well-defined dot products in function spaces are central to coherently defining NOs. Thus, we will often assume functions lie in a vector space in which their dot product is finite wrt some measure µ X over their domain X . We define the Lebesgue space L 2 ( X , µ X ; R d ) as the equivalence classes of functions in this vector space that agree almost everywhere in X with respect to µ X . When clear from context, we will simply denote this vector space by L 2 ( X ) . Whenever needed to evaluate functions point-wise, we further assume the function lies in an appropriate Reproducing Kernel Hilbert Space (RKHSs). In this work, we will be using both the Lebesgue space L 2 ( X ) and RKHSs, when adequate.

Point-wise operators. These operations are carried over from standard neural networks. Thus, given a function f : X → R d , we consider dense layer-operations, with parameters W ∈ R b × d , defined as ( W f )( x ): X → R b := W f ( x ) , and element-wise activations, with a given σ : R → R , to be defined as σ [ f ] j ( x ) := σ (f j ( x )) . By composing and adding results between layers, we can build neural operators that basically act just on the output of the functions.

Kernel integral operator. The majority of interesting behaviors require expanding the receptive field and aggregate results from different function evaluations into one. The kernel integral operator A K : ( X → R d ) → ( Y → R b ) , parametrized by a matrix-valued kernel function K : Y×X → R b × d together with a measure µ X on X , is defined as:

<!-- formula-not-decoded -->

Under this operation, the function evaluated at a single evaluating point y linearly aggregates information on all evaluating points in the domain X as modulated by the kernel K and the measure µ X . Note, that this function may not converge for all values of y , but, for any kernel K ∈ L 2 ( Y × X , µ Y × µ X ; R b × d ) , the operator A K : L 2 ( X , µ X ) → L 2 ( Y , µ Y ) is well-defined.

Constructing neural operators. Given these building blocks, Kovachki et al. (2023) describe a neural operator as a three-part layered model. Firstly, a sequence of point-wise operators are applied to preprocess the function and change the dimension of its output. This is called the Lift layer . The second component is a combination of point-wise and kernel integral operators, in the so-called Neural Operator layer . Finally, the Projection layer , a sequence of point-wise operators is applied to the final result.

Specifically, a neural operator layer combines a matrix-valued kernel K and matrix W into

<!-- formula-not-decoded -->

Setting the matrix-valued kernel to zero recovers the lift and projection layers. Therefore, a neural operator with depth d and scalar output can be written succinctly as the composition:

<!-- formula-not-decoded -->

## 2.2 Probability in Hilbert spaces

Given a probability space (Ω , Σ , P ) , and a Hilbert space H , random elements in H are functions x : Ω →H , such that the inner product ω ∈ Ω ↦→⟨ y, x ( ω ) ⟩ H is a real-valued random variable, for

any y ∈ H . As usual, we follow the standard notation of denoting the random elements/variables not as functions x but as elements x . Likewise, expectation is defined in terms of the random variables ⟨ y, x ( ω ) ⟩ , for each y ∈ H . We say that the expectation of x , when it exists, is the element of H , denoted by E [ x ] , such that E [ ⟨ y, x ⟩ ] = ⟨ y, E [ x ] ⟩ , for any y ∈ H .

We denote the space of Hilbert-Schmidt (HS) operators mapping elements from a Hilbert space A to B by HS( A ; B ) . This space is the completion of the span of rank-one operators of the form a ⊗ b : A → B , defined as ( a ⊗ b )( x ) = ⟨ x, a ⟩ A b for all a ∈ A and b ∈ B . For L 2 spaces, we have the isomorphism HS ( L 2 ( X ; R d ) , L 2 ( Y ; R b )) ∼ = L 2 ( X × Y ; R b × d ) , under which ( f ⊗ g )[ h ]( · ) = ∫ X gf ⊺ ( · , x ) h ( x ) d µ X ( x ) , where f ∈ L 2 ( X ; R d ) , g ∈ L 2 ( Y ; R b ) , and gf ⊺ ∈ L 2 ( X × Y ; R b × d ) .

Moreover, the (cross-)covariance operator of two centered variables x and y is defined as the expectation of the tensor product E [ x ⊗ y ] . When this expectation exists, it is also a HS operator denoted as Cov( x, y ) . From these definitions, we have that ⟨ z 2 , Cov( x, y )[ z 1 ] ⟩ = cov( ⟨ z 2 , y ⟩⟨ z 1 , x ⟩ ) , for any z 1 , z 2 ∈ H . In L 2 spaces, we will make use of the isomorphism above and represent the covariance operator by its integration kernel. So, for any random elements f ∈ L 2 ( X ; R d ) and g ∈ L 2 ( Y ; R b ) , we introduce the function C [ f , g ] : X × Y → R b × d such that Cov( f , g )[ h ]( · ) = ∫ X C [ f , g ]( · , x ) h ( x ) d µ X ( x ) .

In this work, we will make use of an extension of the strong law of large numbers to random elements: Theorem 2.1 (Strong law of large numbers (Mourier, 1956)) . Let H be a separable Hilbert space and { x j } j ∈ N be a countable sequence of identically distributed random elements. Consider the sample average y N = (1 /N ) ∑ N j =1 x j . If, for any j , the expected norm E [ ∥ x j ∥ ] exists, then, the sequence { y N } N ∈ N converges almost surely to the constant random element y ∞ = E [ x j ] .

## 2.3 Operator valued kernels and Hilbert space valued Gaussian processes

Now, given a set X and a separable Hilbert space H , an operator-valued valued kernel C: X × X → HS( H ; H ) is any Hermitian positive-definite function, i.e., for all x , x ′ ∈ X , C( x , x ′ ) = C( x ′ , x ) ⊺ , and, for any n &gt; 0 , { ( x i , y i ) } n i =1 ⊂ X × H and { α ij } n i,j =1 ⊂ R , we have that ∑ n i,j =1 α ij ⟨ y j , C( x i , x j )[ y i ] ⟩ &gt; 0 (Kadri et al., 2016).x

Consider an operator-valued kernel C: X ×X → HS( H ; H ) such that x ↦→ C( x , x ) is of trace-class. We say f : X × Ω →H is a centered Gaussian process with covariance function C if, for any n &gt; 0 and { ( x i , y i ) } n i =1 ⊂ X × H , the vector ( ⟨ y 1 , f( x 1 , · ) ⟩ , . . . , ⟨ y n , f( x n , · ) ⟩ ) is a random element distributed as an n -dimensional Gaussian with covariance

<!-- formula-not-decoded -->

We denote this by f ∼ GP(0 , C) . For simplicity, we also define f( x ) := f( x , · ) .

## 3 Infinite-width neural operators as Gaussian processes

It is well known that infinite-width limits of various Bayesian neural networks are Gaussian processes (Neal, 1995; Matthews et al., 2018). We generalize this connection and show that infinite-width neural operators are function-valued Gaussian processes.

Analogous to Novak et al. (2019), who place Gaussian priors on the convolution kernels of a CNN, the natural step towards function-valued GPs is to put independent GP priors on the component operators. Similarly, we require the weights and kernel for any component operator to be i.i.d. and with covariance shrinking with width. Theorem 3.1 states the main result of this work.

Theorem 3.1 (Infinite-width neural operators are Gaussian processes) . Let X ⊆ R d x be a measurable space and let H ( X ; R J ) ⊂ L 2 ( X ; R J ) be an RKHS for any J ∈ N + . Then, for a given depth D ∈ N + , consider a vector of positive integers J = [ J 0 , J 1 , . . . , J D -1 , 1] ⊺ ∈ N D +1 and a J -indexed neural operators Z ( D ) J of depth D :

<!-- formula-not-decoded -->

where,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and σ : R → R such that ( σ ◦ f) ∈ L 2 ( X ) for any f ∈ L 2 ( X ) .

When all parameters are independently distributed a priori according to

<!-- formula-not-decoded -->

then, the iterated limit lim J →∞ · · · lim J →∞ Z ( D ) J

D -1 1 , in the sense of Definition B.1, is equal to a function- valued GP Z ( D ) ∞ ∼ GP(0 , c ∞ ) , where c ∞ [ f , g ] is available in closed-form.

An outline of the proof is presented in Section 3.2, where we present the explicit formula for c ∞ , which depends on the conditional covariance function between layers. Before delving into these details, we introduce the compositionality property of covariance functions in Section 3.1. This property enables the closed-form computation of the conditional covariances, thereby fully characterizing the limiting covariance function c ∞ .

## 3.1 Operator-valued covariance functions

We realize the following crucial points: i) The covariance function only depends on the inner product of the values of the input functions, and ii) Using the strong law of large numbers, the covariance of the composition of operators can be described by composing its covariance functions. This is presented in the next lemma, with proof postponed to Appendix B.2.

Lemma 3.2 (Compositionality of covariance functions) . Let B 1 : L 2 ( X ; R d ) → L 2 ( X ; R J ) be a random operator and B 2 : L 2 ( X ; R J ) → L 2 ( X ) be a centered function-valued Gaussian process. If the following assumptions hold:

- For all f ∈ L 2 ( X ; R d ) and x ∈ X , each component of B 1 [ f ]( x ) ∈ R J is independent and identically distributed such that the covariance function C B 1 [ f , g ] = c B 1 [ f , g ] I J ;
- The covariance function of B 2 can be expressed, for all f , g ∈ L 2 ( X ; R J ) as c B 2 [ f , g ] = c B 2 [ 1 J g ⊺ f ] and the function h ↦→ c B 2 [h] is a continuous map from L 2 ( X × X ) to itself.

Then, B 2 ◦ B 1 converges in distribution to a function-valued Gaussian process as J →∞ , and

<!-- formula-not-decoded -->

For each operator discussed in Section 2.1, below we state the conditions under which they are function-valued Gaussian processes, and derive their covariance functions.

Point-wise linear operator. Given a vector w ∈ R d and a function f : X → R d , then, define the linear operator ( w ⊺ f ): X → R such that ( w ⊺ f )( x ) = ∑ d p =1 w p f p ( x ) . If the entries of the weight vector follow an i.i.d. Gaussian distribution, i.e. w ∼ N ( 0 , σ 2 I ) , then, this is a centered Gaussian process taking values from L 2 ( X ; R d ) to L 2 ( X ; R ) with covariance function:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that c w only depends on the function f ⊺ 2 f 1 : X × X → R , so we abuse notation and write c w [ f 1 , f 2 ] = c w [ f ⊺ 2 f 1 ] . Moreover, this function is homogeneous: α c w [ f ⊺ 2 f 1 ] = c w [ α f ⊺ 2 f 1 ] , for α &gt; 0 .

Kernel integral operator. As defined in Section 2.1, given a function k : Y × X → R d and an input function f : X → R d , we consider the linear operator A k ⊺ [ f ] : Y → R . If k follows an i.i.d. GP such that k ∈ L 2 ( Y × X ) ∼ GP(0 , c k ) , then we have that A k ⊺ is a centered function-valued GP mapping from L 2 ( X ; R d ) to L 2 ( Y ) with covariance function, denoted here by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note c A k ⊺ also only depends on the inner product of f 2 and f 1 and is homogeneous. Thus, we denote c A k ⊺ [ f 1 , f 2 ] = c A k ⊺ [ f ⊺ 2 f 1 ] .

Point-wise element-wise activation. Given a non-linear function σ : R → R , we abuse the notation and define the non-linear operator σ [ · ] : L 2 ( X ) → L 2 ( X ) as

<!-- formula-not-decoded -->

Note that some restrictions on σ need to be placed for this to be a well-defined operator in L 2 ( X ) . As an example of such condition, for their theoretical analysis, Kovachki et al. (2023) restricts activations to measurable linearly bounded functions, noting that the popular ReLU, ELU, tanh, and sigmoid activations satisfy this condition. In Appendix B.1, we provide a proof that this condition is sufficient for finite measure domains.

Consider a centered Gaussian operator B : L 2 ( X ) →H ( X ) with covariance function c B such that H ( X ) ⊂ L 2 ( X ) is an RKHS with reproducing kernel k H . When σ [ · ] is a well-defined operator, the operator ( σ ◦ B ) is a random operator in L 2 ( X ) → L 2 ( X ) with covariance function:

<!-- formula-not-decoded -->

Now, since B [f 1 ] and B [f 2 ] are Gaussian processes with outputs in an RKHS H , we can consider the following bivariate Gaussian r.v. b [f 1 , f 2 ] = [ B [f 1 ]( x 1 ) , B [f 2 ]( x 2 )] ⊺ :

<!-- formula-not-decoded -->

This random variable is well-defined due to the reproducing property, B [f]( x ) = ⟨ k H ( · , x ) , B [f] ⟩ .

Thus, we can continue to conclude

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where L is a square-root of the covariance matrix of b [f 1 , f 2 ] and l i is the i -th row of this matrix.

The expected value c σ as a function of l 1 and l 2 in Eq. (20) is known as the dual kernel of σ . The dual kernels for many activation functions have closed-form solutions (e.g., sigmoid (Williams, 1996, Eq. 10) and ReLU (Cho and Saul, 2009, Eq. 1)) or can be efficiently approximated (Han et al., 2022). Any of these solutions can be directly used in our context by computing the covariance matrix of b [f 1 , f 2 ] and applying the rows of its square-root as arguments.

In conclusion, we construct an covariance function c σ : H ( X × X ) → L 2 ( X × X ) such that, for a given covariance function c B : L 2 ( X × X ) →H ( X × X ) :

<!-- formula-not-decoded -->

for all f 1 , f 2 , h 1 , h 2 ∈ L 2 ( X )

## 3.2 Outline of the proof for Theorem 3.1

We now describe a sketch for the proof, we refer the readers to Appendix B.3 for the complete proof.

Step 1. We start by showing that, under the conditions of Theorem 3.1, each linear layer in a neural operator layer is a function-valued Gaussian process when conditioned on its inputs. Moreover, as discussed in Section 3.1, the conditional covariance function of each node on each layer only depends on the empirical covariance function of its inputs c[ f , g ]( x ′ , x ) = (1 /J ) ∑ J j =1 g j ( x ) f j ( x ′ ) . We denote this dependency by writing the conditional covariance function as c ( ℓ | ℓ -1) [ · ]( x , x ′ ) .

Step 2. Due to the chosen prior distribution of each layer, we know that each node in H ℓ [ · ] ∈ R J ℓ is i.i.d. and, therefore, we can apply Lemma 3.2 to conclude that, as J ℓ -1 → ∞ , the covariance c ( ℓ | ℓ -1) [ H ℓ -1 [ g ] ⊺ H ℓ -1 [ f ] /J ℓ -1 ] converges almost surely to c ( ℓ | ℓ -1) [c H ℓ -1 [ f , g ]] .

Step 3. Combining both steps, we show, by induction on ℓ up until ℓ = d , that, as J → ∞ , the covariance function of Z J [ f ] is simply the composition of all the previous covariances as denoted in Step 1. So, we have that the covariance function of Z ∞ is:

c ( d ) [ f , g ] = c ( d | d -1) [c ( d -1 | d -2) [ · · · c (2 | 1) [c H 1 [ f , g ]] · · · ] . (23) Finally, denote c ( d ) as c ∞ .

## 4 Parametrizations and computations

To apply the results of Theorem 3.1, we must specify a covariance function for the integral kernel operators A K . This choice corresponds to selecting a particular neural operator parameterization, following the approach of Kovachki et al. (2023).

In this section, we derive the operator-valued covariance functions for A K under two parametrizations of the integral operator. The first is based on the band-limited Fourier Neural Operator (Section 4.1); the second models the kernel as a non-stationary process, with a prior distribution derived from the classical Mat´ ern family of covariance functions (Section 4.2).

A common assumption for both cases is that the input domain is compact. This ensures that samples of the kernel components k j reside in a L 2 space. By further choosing the domain to be the d x -dimensional flat torus T d x = R d x / 2 π Z d x , we are able to exploit Fourier analysis tools. In particular, by assuming that the input functions are band-limited enables tractable computations through the connection of Fourier series with discrete Fourier transforms for evaluations in regular grids.

## 4.1 Fourier neural operator

Out of the parametrizations proposed by Kovachki et al. (2023), the Fourier neural operator is the most popular due to its computational benefits. By imposing three assumptions into the convolutions kernel - periodicity, shift-invariance, and band-limitedness - we can use the convolution theorem to compute the integrals using sums up to the chosen band-limit of the kernel in the Fourier space.

Concretely, assuming periodicity is equivalent to choosing the domain to be some d x -dimensional flat torus X = T d x , and shift-invariance means kernels satisfy k j ( w , x ) = k j ( w -x ) , where we abuse notation and represent the kernel as a univariate function of the same name k j : T d x → R d . Under these conditions, any k j admits a Fourier series representation:

<!-- formula-not-decoded -->

where FS s is the ( s 1 , . . . , s d x ) -th coefficient of the Fourier series and ψ s ( x ) = exp[ -i · s ⊺ x ] , with i = √ -1 being the imaginary unit. Moreover, to have a band-limited kernel implies that only finitely many Fourier coefficients are non-zero, i.e. there is some B j ∈ N , 1 ≤ j ≤ d x , such that FS s [ k j ] = 0 , if | s j | &gt; B j , for all 1 ≤ j ≤ d x .

Under these conditions, despite all input functions f being represented with a (potentially infinite) Fourier series, by the convolution theorem, the NO layer H j [ f ] is band-limited and its Fourier series coefficients can be computed directly from the product of Fourier coefficients of the kernel function k and the input function f . Thus, we have that:

<!-- formula-not-decoded -->

Parameterization of an FNO. Following Section 3.1, when k is a R d -valued GP, the kernel integral operator A k is a function-valued Gaussian process with covariance function of A k in terms of the covariance function of k , C k :

<!-- formula-not-decoded -->

The most popular choice proposed by Kovachki et al. (2023) is to directly parametrize the Fourier coefficients of the kernel. Thus, we let these 2 B +1 Fourier coefficients follow i.i.d. centered complex Gaussian distributions with variance σ 2 k (Appendix A.1), obtaining the covariance function C k :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where B is a hyperparameter of the model controlling the band-limit of the integral kernel.

This allows us to derive a finite-sum representation of the covariance of A k parameterized by σ 2 k .

<!-- formula-not-decoded -->

## 4.2 Toroidal Mat´ ern operator

In this section, we propose a model in which the kernel does not admit a shift-invariant decomposition. Another popular decomposition used in the Gaussian process literature is the tensor-product factorization, where the covariance function of a GP factorizes over the input dimension. That is, f : R d x → R ∼ GP(0 , c) , where c( a , b ) = ∏ d x j c j ( a j , b j ) ; although the covariance factorizes over the input dimensions, in general, samples from f do not.

Our proposal will make use of the ubiquitous Mat´ ern family of covariance functions, which are characterized by the smoothness parameter ν . Following Borovitskiy et al. (2020), we define the Mat´ ern covariance functions in the d x -dimensional flat torus T d x = T ⊗··· ⊗ T as:

<!-- formula-not-decoded -->

where ℓ is the lengthscale hyper-parameter and the spectral density ˆ c is defined as:

<!-- formula-not-decoded -->

In general, this kernel is not tensor-product factorized, but for the special case of ν = ∞ , the squared exponential covariance function, the factorization holds (Appendix A.2.1). Thus, in general, we enforce the tensor-product factorization:

<!-- formula-not-decoded -->

where ℓ is the automatic relevance determination (ARD) lengthscale hyper-parameter.

Parameterization of a toroidal Mat´ ern operator. So, we consider a convolution kernel k : X × X → R d defined as the product of Mat´ ern covariance functions:

<!-- formula-not-decoded -->

where c( · , · ; ν, ℓ ) : X × X → R is the Mat´ ern covariance functions with smoothness parameter ν and length-scale ℓ .

Again, following Section 3.1, we express the covariance of the operator as:

<!-- formula-not-decoded -->

Thus, by using the identity Eq. (33), we can derive:

<!-- formula-not-decoded -->

## 5 Experimental validation

In this section, we empirically show i) the agreement between finite width neural operators with increasing width and their corresponding infinite-width neural operator, and ii) evaluate our model against FNO in a regression task.

Figure 1: A density estimation of the empirical distribution of the output of increasing channel dimension compared to the infinite width distribution. On top of each plot we show the total variation distance of the empirical distribution against the infinite width distribution.

<!-- image -->

Section 5.1 explores the distribution of untrained randomly initialized Fourier neural operators of varying width and the distribution of the infinite-width FNO ( ∞ -FNO). As expected from the theoretical results, these distributions should eventually match as the hidden dimension increases.

Section 5.2 considers two tasks: a synthetic regression example, where the task is to predict the output of a non-linear operator, and the task of predicting the final evolution of Burgers' equation given the initial state. This situation is not covered in our theory, since it only applies to the distribution of the neural operators at initialization, but our experiments show the behavior of the posteriors of infinite-width neural operators against Adam trained finite-width neural operators of increasing width.

Throughout this section, our stating point is a single hidden-layer neural operator Z [f] : T → R := ( w ⊺ 2 ◦ ReLU ◦ ( A K + W 1 ))[f] . More details for each experiment can be found in Appendix C. All experiments were implemented in Python, mainly based on the GPyTorch (Gardner et al., 2018) library, and run in a desktop computer using a Titan RTX. Code is avaliable at https://github.com/spectraldani/infinite-neural-operator .

## 5.1 Empirical demonstration of results

In this experiment, we demonstrate that our analytical computation of the variance for a neural operator layer H agrees with empirical estimates, and we validate Theorem 3.1 by showing that the output of a neural operator Z converges to a Gaussian distribution as the number of hidden channels J increases.

Throughout all experiments, the input function f : T → R has band-limit B = 3 , with its output values f( x ) sampled from a uniform distribution U ( -1 , 1) . Both operators are evaluated at x = 0 , so we analyze the empirical distributions of H [f](0) and Z [f](0) , respectively.

Following Section 4.1, we parametrize the integral kernel operators using band-limited functions. The band-limit of the kernel is set equal to that of the input f , and the kernel coefficients are drawn from a Gaussian distribution with unit variance scaled by the inverse of the number of hidden channels.

As shown in Fig. 2, the empirical estimate of the variance converges to the theoretical value as the number of Monte Carlo samples increases, supporting the correctness of our variance computation. Furthermore, Fig. 1 shows that, as the number of hidden channels grows, the total variation distance (TVD) between the empirical distribution and a Gaussian distribution approaches zero, thereby further verifying the validity of Theorem 3.1.

## 5.2 Regression tasks

In this task, we're given n pairs of 1D functions { f i , g i } n i =1 evaluated in a grid with m = 2 B m +1 points. We consider FNOs of increasing width J ∈ N + , as well as ∞ -FNOs, both with increasing kernel band-limits B ∈ { 1 , 5 , 20 } . These models will be trained on two datasets: (a) A operator

Figure 2: Plot of the MC estimate for the variance of H [f](0) against our analytical computation (Sec. 3.1).

<!-- image -->

∞

-

)

Figure 3: Results for the regression experiments. Mean and std. of test L 2 loss as a function of width J for different band-limits B .

<!-- image -->

generated by a randomly-initialized ground truth FNO Z true with band-limit B = 5 and width J = 1 . We sample n = 100 input functions f i : T → R with uniformily-distributed outputs U ( -1 , 1) and band-limit B m = 5 . (b) 1D Burgers' equation dataset from Takamoto et al. (2022) with ν = 0 . 002 . The task is to predict the end state ( t = 2 ) given the initial condition ( t = 0 ). Due to memory constraints, we subsample the total dataset data to n = 100 functions and a grid size of m = 103 .

The hyperparameters of the ∞ -FNO are estimated using L-BFGS, while the parameters of the FNOs are optimized with Adam using a step size of 0 . 001 . We evaluate all models using 5-fold crossvalidation and report the average and standard deviation of the empirical L 2 norm of the prediction error. For ∞ -FNOs, we use the posterior mean as the prediction.

In general, we do not expect close agreement between the predictive performance of ∞ -FNOs and finite-width FNOs, as the former corresponds to a Bayesian estimate while the latter are trained by minimizing an empirical risk, nonetheless, as observed in Figs. 3a and 3b, there is consistency between the gap of hyperparameters in the same model class.

In the synthetic case, as we know the band-limits of the ground truth operator, Fig. 3a shows that the models are only able to accurately predict the output when their band-limits exceed that of the ground truth.

## 6 Related works

Infinite limits of stochastic NNs. The study of infinite-width Bayesian neural networks began with the seminal work of Neal (1995) and was later extended to deep architectures (Lee et al., 2018; Yang, 2019; Matthews et al., 2018). Our analysis builds on the ideas developed by Matthews et al. (2018). From the outset, these infinite-width models were considered 'disappointing' (Neal, 1995), a view reinforced by findings that neither the Bayesian limit nor the neural tangent kernel limit learns features from data (Aitchison, 2020). However, recent work shows these models still reflect the different inductive biases of their finite-width counterparts (Novak et al., 2019), and that alternative initialization distributions can enable feature learning in the infinite-width setting (Yang and Hu, 2021).

Bayesian neural operators. Several works have investigated approximate Bayesian uncertainty quantification in finite-width neural operators using function-valued Gaussian processes. Magnani et al. (2022, 2024) both employ last-layer Laplace approximations to construct GP approximations of the posterior distribution. In addition, Magnani et al. (2022) considers the case where the kernel K of the integral operator A K follows a Mat´ ern GP prior. However, their analysis is restricted to the finite-width regime on compact subsets of Euclidean space, whereas our work focuses on the flat torus.

Kernel methods for operator learning. Batlle et al. (2024) propose the use of kernel methods for operator learning, leveraging operator-valued kernels and the representer theorem in their corresponding RKHS. Their results are promising and highlight the potential of kernel-based approaches in this

domain. Our contribution introduces an additional way to construct operator-valued kernels based on neural operators, enabling new kernel-based models for operator learning.

## 7 Discussion

In this work, we formalized the concept of infinite-width Bayesian neural operators, established their existence (Theorem 3.1), and described how to compute their associated covariance functions (Section 4). We validated these results empirically (Section 5.1) and further assessed the performance of these models in a regression setting (Section 5.2).

Our contributions lay a foundation for future investigations, particularly in bridging the gap between SGD-trained finite-width neural operators and their infinite-width counterparts. Addressing this challenge will require extending the neural tangent kernel framework (Jacot et al., 2018; Lee et al., 2019) to settings involving Hilbert space-valued functions. Moreover, while we focused on the ubiquitous FNO architecture, deriving covariance functions for other architectures, such as the graph neural operator (Kovachki et al., 2023), remains an open direction.

Limitations. Our current implementation for computing the required kernel quantities scales with cubically in both the evaluation grid size and the number of training functions. We anticipate that future work can improve computational efficiency by leveraging advances from the Gaussian process literature to improve scalability and efficiency (Borovitskiy et al., 2020; Gilboa et al., 2015).

## Acknowledgments and Disclosure of Funding

YZ acknowledges support by the Engineering and Physical Sciences Research Council with grant number EP/S021566/1. YS was supported by Fundac ¸˜ ao Carlos Chagas Filho de Amparo ` a Pesquisa do Estado do Rio de Janeiro (FAPERJ) through the Jovem Cientista do Nosso Estado Program (E26/201.375/2022 (272760)) and by Conselho Nacional de Desenvolvimento Cient´ ıfico e Tecnol´ ogico (CNPq) through the Productivity in Research Scholarship (306695/2021-9, 305159/2025-9). DM was supported by the Fundac ¸˜ ao Carlos Chagas Filho de Amparo ` a Pesquisa do Estado do Rio de Janeiro (FAPERJ) (SEI-260003/000709/2023) and the Conselho Nacional de Desenvolvimento Cient´ ıfico e Tecnol´ ogico (CNPq) (404336/2023-0, 305692/2025-9).

## References

- Laurence Aitchison. Why bigger is not always better: on finite and infinite neural networks. In Proceedings of the 37th International Conference on Machine Learning (ICML) , 2020. URL https://proceedings.mlr. press/v119/aitchison20a.html .
- Pau Batlle, Matthieu Darcy, Bamdad Hosseini, and Houman Owhadi. Kernel methods are competitive for operator learning. Journal of Computational Physics , 496, 2024. URL https://doi.org/10.1016/j. jcp.2023.112549 .
- Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, and Marc Deisenroth. Mat´ ern Gaussian processes on Riemannian manifolds. In Advances in Neural Information Processing Systems (NeurIPS) , 2020. URL https://papers.nips.cc/paper/2020/hash/92bf5e6240737e0326ea59846a83e076-Abstract. html .
- Youngmin Cho and Lawrence Saul. Kernel methods for deep learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2009. URL https://proceedings.neurips.cc/paper/2009/hash/ 5751ec3e9a4feab575962e78e006250d-Abstract.html .
- Jacob R. Gardner, Geoff Pleiss, David Bindel, Kilian Q. Weinberger, and Andrew Gordon Wilson. GPyTorch: Blackbox matrix-matrix Gaussian process inference with gpu acceleration. In Advances in Neural Information Processing Systems (NeurIPS) , 2018. URL https://papers.nips.cc/paper\_files/paper/2018/ hash/27e8e17134dd7083b050476733207ea1-Abstract.html .
- Elad Gilboa, Yunus Saatc ¸i, and John P. Cunningham. Scaling multidimensional inference for structured Gaussian processes. IEEE Transactions on Pattern Analysis and Machine Intelligence , 37, 2015. URL http://dx.doi.org/10.1109/TPAMI.2013.192 .

- Insu Han, Amir Zandieh, Jaehoon Lee, Roman Novak, Lechao Xiao, and Amin Karbasi. Fast neural kernel embeddings for general activations. In Advances in Neural Information Processing Systems (NeurIPS) , 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ e7be1f4c6212c24919cd743512477c13-Abstract-Conference.html .
- Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems (NeurIPS) , 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ 5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html .
- Hachem Kadri, Emmanuel Duflos, Philippe Preux, St´ ephane Canu, Alain Rakotomamonjy, and Julien Audiffren. Operator-valued kernels for learning from functional response data. Journal of Machine Learning Research (JMLR) , 17, 2016. URL http://jmlr.org/papers/v17/11-315.html .
- Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research (JMLR) , 24, 2023. URL http://jmlr.org/papers/v24/ 21-1524.html .
- Jaehoon Lee, Jascha Sohl-Dickstein, Jeffrey Pennington, Roman Novak, Sam Schoenholz, and Yasaman Bahri. Deep neural networks as Gaussian processes. In International Conference on Learning Representations (ICLR) , 2018. URL https://openreview.net/forum?id=B1EA-M-0Z .
- Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In Advances in Neural Information Processing Systems (NeurIPS) , 2019. URL https://proceedings. neurips.cc/paper/2019/hash/0d1a9651497a38d8b1c3871c84528bd4-Abstract.html .
- Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. ACM / IMS Journal of Data Science , 1(3), 2024. URL https://doi.org/10.1145/3648506 .
- Emilia Magnani, Nicholas Kr¨ amer, Runa Eschenhagen, Lorenzo Rosasco, and Philipp Hennig. Approximate Bayesian neural operators: Uncertainty quantification for parametric PDEs. 2022. URL https://arxiv. org/abs/2208.01565 .
- Emilia Magnani, Marvin Pf¨ ortner, Tobias Weber, and Philipp Hennig. Linearization turns neural operators into function-valued Gaussian processes. 2024. URL https://arxiv.org/abs/2406.05072 .
- Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, and Zoubin Ghahramani. Gaussian process behaviour in wide deep neural networks. 2018. URL https://arxiv.org/abs/1804.11271 .
- Edith Mourier. L-random elements and l'-random elements in Banach spaces. In Contributions to Probability Theory , pages 231-242. University of California Press, December 1956. URL https://doi.org/10.1525/ 9780520350670-017 .
- Radford M. Neal. Bayesian Learning for Neural Networks . PhD thesis, University of Toronto, 1995.
- Roman Novak, Lechao Xiao, Yasaman Bahri, Jaehoon Lee, Greg Yang, Jiri Hron, Daniel A. Abolafia, Jeffrey Pennington, and Jascha Sohl-Dickstein. Bayesian deep convolutional networks with many channels are Gaussian processes. In International Conference on Learning Representations (ICLR) , 2019. URL https: //openreview.net/forum?id=B1g30j0qF7 .
- Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, and Animashree Anandkumar. Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators. 2022. URL https://arxiv.org/abs/2202.11214 .
- Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pfl¨ uger, and Mathias Niepert. PDEBench: An extensive benchmark for scientific machine learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2022. URL https://papers.neurips.cc/paper\_files/ paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets\_and\_Benchmarks. html .
- Christopher Williams. Computing with infinite networks. In Advances in Neural Information Processing Systems (NeurIPS) , 1996. URL https://proceedings.neurips.cc/paper/1996/hash/ ae5e3ce40e0404a45ecacaaf05e5f735-Abstract.html .

- Greg Yang. Wide feedforward or recurrent neural networks of any architecture are Gaussian processes. In Advances in Neural Information Processing Systems (NeurIPS) , 2019. URL https://papers.neurips. cc/paper\_files/paper/2019/hash/5e69fda38cda2060819766569fd93aa5-Abstract.html .
- Greg Yang and Edward J. Hu. Tensor programs IV: Feature learning in infinite-width neural networks. In Proceedings of the 38th International Conference on Machine Learning (ICML) , 2021. URL https: //proceedings.mlr.press/v139/yang21c.html .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our contributions consist of i) a major theorem (Theorem 3.1) characterizing the infinite-width limit of Neural Operators (NOs), and ii) closed-form equations (Section 4) for computing the kernel induced by NO architectures, which allow for practical implementations.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have included the limitations of our work in the Discussion section.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: The main assumptions are placed inside the theorems, the main text includes a proof sketch of our main result, and the full proof for it and related lemmas is provided in the supplemental material.

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

Justification: The main text include a summary of how the dataset was generated and how to compute the necessary quantities for implementation. The supplementary material include further details on hyper-parameters and initializations alongside the full source code for the experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in

some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We will provide code as part of the supplemental material.

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

Justification: We describe this in the experimental section and include more detail in the supplemental material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We include mean and standard error for experiment 5.1, representing the different sample used for estimation and experiment 5.2 contain mean and standard deviation of representing each of the 5-fold splits.

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

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We have included the specs of the machine used for experiments in the main text and in the supplemental material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This work deals with theoretical results and synthetic data, thus, we believe that it conforms to the Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work deals with theoretical results and synthetic data, we believe there is no direct path for direct negative application of this work.

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

Justification: The models used in this work do not pose such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The authors of frameworks used for implementation are credited.

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

Answer: [NA]

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## Infinite Neural Operators: Gaussian processes on functions

## (Supplemental Materials)

## A Covariance Function Computation

In this section, we will work out in detail the computations of Section 4, first for the Fourier neural operator (FNO) case and later for the toroidal Mat´ ern operator.

## A.1 Fourier neural operator

Under the direct parametrization of the integral kernel operator, the coefficients of the kernel's Fourier series (FS) are parametrized and randomly sampled at initialization. Therefore, our first step is to derive what the Gaussian process distribution of a band-limited function with i.i.d. Gaussian FS coefficients is.

Fourier series. Given a function on the d x -dimensional torus f ( · ): T d x → R d , f = (f 1 , . . . , f d ) , it can be represented in terms of a Fourier series:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and i = √ -1 is the imaginary unit.

Note that, as f p is a real-valued function, we also have that FS s [f p ] = FS -s [f p ] , where ¯ z is the complex conjugate.

Gaussian distributed band-limited functions. Consider the sequence ˆ f : [ -B,.. . , B ] d x → C defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

where ℜ z and ℑ z are the real and imaginary parts of the complex number z , respectively, and all random variables are independent of each other, except the conjugate duals ˆ f s and ˆ f -s . For s = 0 , the equations above can also be expressed as:

<!-- formula-not-decoded -->

With this in mind, the expectation of the product of two elements is:

<!-- formula-not-decoded -->

for p ∈ { 1 , . . . , d } , where,

<!-- formula-not-decoded -->

Thus, we can define the Gaussian process f : T d x → R through a Fourier series representation:

<!-- formula-not-decoded -->

We compute the covariance function of f as:

<!-- formula-not-decoded -->

## A.1.1 Covariance after convolution c A k ⊺

Let us place a centered Gaussian distribution on the Fourier series of the band-limited convolution kernel k : X × X → R d , so that:

<!-- formula-not-decoded -->

So, let us consider the quantity c A k [ f 1 , f 2 ] for arbitrary functions f 1 and f 2 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 Toroidal Mat´ ern operator

Definition A.1 (Mat´ ern family of kernels on a closed manifold) . The Mat´ ern family of kernels c with lengthscale ℓ in a d -dimensional closed manifold M are described as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where, λ k and ϕ k are the k -th eigenvalues and eigenfunctions, respectively, of the Laplace-Beltrami operator of the manifold M .

For a 1-dimensional flat torus, an orthonormal eigensystem for the Laplace-Beltrami operator is:

<!-- formula-not-decoded -->

Additionally, for a d x -dimensional flat torus, an orthonormal eigensystem for the Laplace-Beltrami operator is given by the sum and product of the 1-dimensional eigensystem such that, given an index k = [ k 1 , . . . , k d x ] , we have that λ k = ∑ d x j =1 λ k j and ϕ k ( x ) = ∏ d x j =1 ϕ k j ( x j ) .

Expression of 1-dimensional toroidal kernel using complex exponentials. For convenience, we will rewrite the series expansion of this kernel to use exponentials of complex numbers.

Start by noting that:

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, we can rewrite the Mat´ ern kernel expression as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Product kernel for T d x In order to have one lengthscale per dimension, we will make a tensor product kernel where the kernel of a d x -dimensional torus T d x is the product of the 1-d toroidal kernel for each dimension:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2.1 ν = ∞ lets Mat´ ern kernel be a product kernel

Notice that when ν = ∞ and ℓ j = ℓ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With the proper rearrangement, we can see that this fits the definition of a Mat´ ern kernel in the T d x , as the eigenvalues of its Beltrami-Laplace operator can be expressed as the sum of the eigenvalues for the 1-dimensional flat torus T .

## A.2.2 Covariance after convolution c A k

Let us place a centered factored Mat´ ern prior in the convolution kernel k : Z × X → R d , so that:

<!-- formula-not-decoded -->

When clear from context, we will suppress the dependency on the hyper-parameters of the Mat´ ern kernel.

So, let us consider the quantity c A k [ f 1 , f 2 ] for arbitrary functions f 1 and f 2 :

<!-- formula-not-decoded -->

## B Proofs

In this section, we include the proofs for Lemma 3.2, Theorem 3.1, and a short lemma on the well-defined-ness of the activation operator.

## B.1 Well-defined-ness of the point-wise element-wise activation operator

Lemma B.1 Let ( X , Σ , µ X ) be a finite measure space, i.e. µ X ( X ) ≤ ∞ , and σ : R → R a Borel measurable function such that

<!-- formula-not-decoded -->

for some constant C ∈ R . Then, the operator σ [f] : L 2 ( X ) → L 2 ( X ) = σ ◦ f is well defined.

Proof. Remember that a function f is in L 2 ( X ) if, and only if,

<!-- formula-not-decoded -->

Now, from the linear boundedness condition, we know that for any f ∈ L 2 ( X ) and any x ∈ X :

<!-- formula-not-decoded -->

by squaring both sides and taking integrals,

<!-- formula-not-decoded -->

Now, note that the constant function 1 is in L 2 ( X ) since ∫ X 1 d µ X ( x ) = µ X ( X ) &lt; ∞ and that | f( · ) | is in L 2 X . Thus, from linearity, 1+ | f( · ) | is also in L 2 X and ∫ X 1+ | f( x ) | d µ X ( x ) &lt; ∞ . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.2 Compositionality of covariance functions

Lemma 3.2. Let B 1 : L 2 ( X ; R d ) → L 2 ( X ; R J ) be a random operator and B 2 : L 2 ( X ; R J ) → L 2 ( X ) be a centered function-valued Gaussian process. If the following assumptions hold:

- For all f ∈ L 2 ( X ; R d ) and x ∈ X , each component of B 1 [ f ]( x ) ∈ R J is independent and identically distributed such that the covariance function C B 1 [ f , g ] = c B 1 [ f , g ] I J ;
- The covariance function of B 2 can expressed, for all f , g ∈ L 2 ( X ; R J ) as c B 2 [ f , g ] = c B 2 [ 1 J g ⊺ f ] and the function h ↦→ c B 2 [h] is a continuous map from L 2 ( X × X ) to itself.

Then, B 2 ◦ B 1 converges in distribution to a function-valued Gaussian process as J →∞ , and

<!-- formula-not-decoded -->

Proof. Consider a set of size N ∈ N + , { ( f n , h n ) } N n =1 ⊂ L 2 ( X ; R d ) × L 2 ( X ) , then define the N -dimensional vector:

<!-- formula-not-decoded -->

Additionally, define the function:

<!-- formula-not-decoded -->

Then, the conditional random variable z | { ¯ c B 1 [ f i , f j ] } N i,j =1 is Gaussian distributed with zero mean and covariance:

<!-- formula-not-decoded -->

We want to show that every z converges in distribution to a Gaussian distribution when J →∞ , thus, it is useful to remember the following facts:

- Multivariate Levy's continuity theorem. A sequence of random variables { x j } ∞ j =1 converges to another one x ∞ if and only if the sequence of characteristic functions ϕ x j ( t ) = E [exp( i · t ⊺ x j )] , where i = √ -1 , converges point-wise to ϕ x ∞ .
- Characteristic function of a N -dimensional Gaussian distribution. If x ∼ N ( 0 , Σ ) , then ϕ x ( t ) = exp( t ⊺ Σ t ) and ϕ x ( t ) ≤ 1 , for all t ∈ R N .
- Strong law of large numbers. As J →∞ , the random element K [ f i , f j ] converges strongly to the constant c B 1 [ f i , f j ] , for all f i .
- Portmanteau theorem. Given a sequence of random elements in H converging in distribution { x i } ∞ i =1 → x ∞ , then, lim i →∞ E [ f ( x i )] = E [ f ( x ∞ )] , for all bounded and continuous functions f : H → R .

Thus, we begin with the characteristic function of the variable z :

<!-- formula-not-decoded -->

by the tower rule, we can write

<!-- formula-not-decoded -->

where, [ Σ ] jk = cov( z j , z k | ¯ c B 1 [ f j , f k ]) is a random variable.

Now, because of the continuity of inner products and the assumption that c B 2 is continuous, we know that the mapping h ↦→ cov( z j , z k | ¯ c B 1 [ f j , f k ] = h) is continuous. With this we take the limit:

<!-- formula-not-decoded -->

using the portmanteau theorem, we get that:

<!-- formula-not-decoded -->

and, finally, by the strong law of large numbers, we write the expectation as:

<!-- formula-not-decoded -->

Therefore, we have shown that z converges to a centered Gaussian distribution with:

<!-- formula-not-decoded -->

Since the set { f n , h n } N n =1 is arbitrary, we have shown that B 2 ◦ B 1 is a centered function-valued Gaussian process with covariance function c B 2 [c B 1 [ · , · ]] .

## B.3 Infinite-width neural operators are Gaussian processes

Definition B.1 (Iterated convergence in distribution) . Let X i be a random variable for each i = [ i 1 , · · · , i k ] ⊂ N + . The iterated limit

<!-- formula-not-decoded -->

whenever exists, is defined as the iterated limit in distribution. That is, suppose there are random variables X [ ∞ ,i 2 , ··· ,i k ] , X [ ∞ , ∞ , ··· ,i k ] , · · · , X [ ∞ , ∞ , ··· , ∞ ] such that for every i 2 , · · · , i k

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

.

.

.

<!-- formula-not-decoded -->

Then we define the iterated limit of X i as

<!-- formula-not-decoded -->

Theorem 3.1. Let X ⊆ R d x be a measurable space and let H ( X ; R J ) ⊂ L 2 ( X ; R J ) be an RKHS for any J ∈ N + . Then, for a given depth D ∈ N + , consider a vector positive integers J = [ J 0 , J 1 , . . . , J D -1 , 1] ⊺ ∈ N D +1 and a J -indexed neural operators Z ( D ) J of depth D :

<!-- formula-not-decoded -->

where,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with W ( ℓ ) ∈ R J ℓ × J ℓ -1 , and K ( ℓ ) ∈ H ( X × X ; R J ℓ × J ℓ -1 ) .

When all parameters are independently distributed a priori according to

<!-- formula-not-decoded -->

then, the iterated limit lim J D -1 →∞ · · · lim J 1 →∞ Z ( D ) J , in the sense of Definition B.1, is equal to a functionvalued GP Z ( D ) ∞ ∼ GP(0 , c ∞ ) , where c ∞ [ f , g ] is available in closed-form.

Proof. First, we note from Section 3.1 that the covariances c W ( ℓ ) [ f , g ] and c A K ( ℓ ) [ f , g ] are equal to:

<!-- formula-not-decoded -->

such that both depend on the empirical covariance 1 J ℓ -1 g ⊺ f , for all f , g ∈ L 2 ( X ; R J ℓ -1 ) and ℓ ∈ N + . Therefore, since H ( ℓ ) is the sum of these two independent function-valued Gaussian processes, we have that H ( ℓ ) ∼ GP ( 0 , c ( ℓ | ℓ -1) I J ℓ ) such that:

<!-- formula-not-decoded -->

With this in mind, we proceed the proof by induction on the depth D .

Base case. For the base case D = 1 , we consider the operator Z (1) J . Therefore, there are no limits to consider in this step. Nonetheless, as discussed in the previous paragraph, this quantity is a function-valued GP with covariance:

<!-- formula-not-decoded -->

Therefore, our claim is proven.

Inductive step. Our inductive hypothesis says that, for a specific ℓ ∈ N + , we have that the iterated limit lim J ℓ -1 →∞ · · · lim J 1 →∞ Z ( ℓ ) J converges in distribution to a Z ( ℓ ) ∞ ∼ GP ( 0 , c ( ℓ ) I J ℓ ) .

As a first step, we would like to prove that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consider an arbitrary set of size N ∈ N + ,

<!-- formula-not-decoded -->

and define the variables z [ F , H ] ∈ R N and Z ( ℓ ) J [ F ] ∈ L 2 ( X ; R N × J ℓ ) such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition, z [ F , H ] conditioned on Z ( ℓ ) J [ F ] follows a multivariate centered Gaussian distribution N ( 0 , Σ ( Z ( ℓ ) J [ F ] )) with covariance matrix:

<!-- formula-not-decoded -->

Thus, by the tower rule, the characteristic function of the marginal distribution of z [ F , H ] is:

<!-- formula-not-decoded -->

Now, consider the point-wise convergence of the characteristic function:

<!-- formula-not-decoded -->

Using the portmanteau theorem and continuity of Σ ( · ) , we have that:

<!-- formula-not-decoded -->

Now, our inductive hypothesis says that Z ( ℓ ) J converges in distribution to a function-valued Gaussian process Z ( ℓ ) ∞ with each output being i.i.d. With this fact, we can conclude that Z ( ℓ ) J [ F ] also converges in distribution to the corresponding variable: [ Z ( ℓ ) ∞ [ F ]] n := Z ( ℓ ) ∞ [ f n ] . This means that:

<!-- formula-not-decoded -->

converges in distribution to

which is the characteristic function of a variable defined as:

<!-- formula-not-decoded -->

Therefore, z [ F , H ] iteratively converges in distribution to ˜ z [ F , H ] , as J ℓ →∞ for every ℓ ≤ ℓ . Since the set ( F , H ) is arbitrary, we can conclude that ( H ( ℓ +1) ◦ σ ◦ Z ( ℓ ) J ) also converges in distribution to ( H ( ℓ +1) ◦ σ ◦ Z ( ℓ ) ∞ ) as a random operator.

From the induction step, we know that the entries in ( σ ◦ Z ( ℓ ) ∞ ) are i.i.d. since the entries of σ ◦ Z ( ℓ ) ∞ are also i.i.d. Therefore, we use Lemma 3.2 to show that lim J ℓ →∞ ( H ( ℓ +1) ◦ σ ◦ Z ( ℓ ) ∞ ) converges in distribution to a function-valued Gaussian process with covariance function

<!-- formula-not-decoded -->

Therefore, we just proved by induction that the iterated limit lim J D -1 →∞ · · · lim J 1 →∞ Z ( D ) J converges in distribution to a Z ( D ) ∞ ∼ GP( 0 , c ∞ I J ℓ ) and this covariance function is equal to:

<!-- formula-not-decoded -->

## C Experimental details

In this section, we describe the setup for our experiments. As previously mentioned, all experiments were run in a desktop machine with a 3.8GHz Intel Core i7-9800X CPU and a 24GB NVIDIA Titan RTX (TU102) GPU. More details for each experiment can be found below.

## C.1 Empirical demonstration of results

For both experiments, the input function f : T → R has band-limit B = 3 , with its output values f( x ) sampled from a uniform distribution U ( -1 , 1) . In other words, we can express this band-limited function as:

<!-- formula-not-decoded -->

where each f s ∼ U ( -1 , 1) is independent and identically distributed.

In the first experiment of Fig. 2, we construct the operator layer H under the usual formulation:

<!-- formula-not-decoded -->

where w ∼ N (0 , 1) and k follows the band-limited Gaussian process distribution (Section 4.1 and Appendix A.1) with with band-limit B = 3 and variance σ 2 = 1 / 7 . Then, the operator on f is evaluated at zero H [f](0) with increasing sample sizes.

For the second experiment of Fig. 1, we construct the single-layer neural operator:

<!-- formula-not-decoded -->

where J is the width of the hidden layer, and w 2 ∼ N (0 , 1 /J ) , w 2 ∼ N (0 , 1) , and k follows an i.i.d. band-limited Gaussian process distribution (Section 4.1 and Appendix A.1) with band-limit B = 3 and variance σ 2 = 1 / 7 . For varying widths J ∈ { 1 , 10 , 100 , 1000 } , we evaluate 10,000 samples of the operator on f at zero Z [f](0) and show the density of the empirical distribution using kernel density estimation (KDE) with a Gaussian kernel.

These experiments are implemented in the file experiments/fno limit.ipynb .

## C.2 Regression

We consider FNOs of increasing width, J ∈ { 1 , 10 , 100 } and J ∈ { 1 , 3 , 10 , 100 , 500 } for the synthetic and 1D Burgers' respectively, as well as ∞ -FNOs, both with increasing kernel band-limits B ∈ { 1 , 5 , 20 } . These single-layer neural operators are constructed as:

<!-- formula-not-decoded -->

where J is the width of the hidden layer, and w 2 ∼ N (0 , 1 /J ) , w 2 ∼ N (0 , 1) , and k follow an i.i.d. band-limited Gaussian process distribution (Section 4.1 and Appendix A.1) with variance σ 2 = 1 / (2 B +1) .

The hyperparameters of the ∞ -FNO are estimated using L-BFGS, while the parameters of the FNOs are optimized with Adam using a step size of 0 . 001 . We evaluate all models using 5-fold crossvalidation and report the average and standard deviation of the empirical L 2 norm of the prediction error. For ∞ -FNOs, we use the posterior mean as the prediction.

This experiment is implemented in the file experiments/train.py .

## Synthetic regression

We start by defining the ground truth Fourier neural operator (FNO) which will generate our training and test data:

<!-- formula-not-decoded -->

where the hidden layer's width is 1 and the band-limit of k is equal to 5. Next, we sample n = 100 input functions f i : T → R with the same band-limit B = 5 and uniformily-distributed outputs U ( -1 , 1) , so that we have:

<!-- formula-not-decoded -->

where each f is ∼ U ( -1 , 1) is independent and identically distributed. We then compute Z true [f i ] on an equally spaced grid given by {-5 2 π 11 , . . . , 5 2 π 11 } ⊂ R 11 .

## 1D Burgers' equation

This dataset is provided from PDEBench (Takamoto et al., 2022), which includes solutions to the 1D Burgers' equation:

<!-- formula-not-decoded -->

where x ∈ (0 , 1) and t ∈ (0 , 2] are independent variables and ν is the diffusion coefficient.

The regression task is set up with ν = 0 . 002 and a collection of initial conditions { u(0 , · ) = f i } n i =1 and their respective end states { u(2 , · ) = g i } n i =1 . Due to memory constraints when creating the covariance matrices for ∞ -FNO, we subsample the original dataset to n = 100 functions and a grid size of m = 103 . The original data can be downloaded at https://darus.uni-stuttgart.de/ api/access/datafile/268193 .
## Information-Computation Tradeoffs for Noiseless Linear Regression with Oblivious Contamination

## Ilias Diakonikolas

University of Wisconsin, Madison ilias@cs.wisc.edu

## Daniel M. Kane

University of California, San Diego dakane@ucsd.edu

## Ankit Pensia ∗

Carnegie Mellon University ankitp@cmu.edu

## Abstract

We study the task of noiseless linear regression under Gaussian covariates in the presence of additive oblivious contamination. Specifically, we are given i.i.d. samples from a distribution ( x, y ) on R d × R with x ∼ N (0 , I d ) and y = x ⊤ β + z , where z is drawn independently of x from an unknown distribution E . Moreover, z satisfies P E [ z = 0] = α &gt; 0 . The goal is to accurately recover the regressor β to small ℓ 2 -error. Ignoring computational considerations, this problem is known to be solvable using O ( d/α ) samples. On the other hand, the best known polynomialtime algorithms require Ω( d/α 2 ) samples. Here we provide formal evidence that the quadratic dependence in 1 /α is inherent for efficient algorithms. Specifically, we show that any efficient Statistical Query algorithm for this task requires VSTAT complexity at least ˜ Ω( d 1 / 2 /α 2 ) .

## 1 Introduction

Linear regression is a prototypical supervised learning task with a wide range of applications [RL87; Die01; McD09]. In the vanilla setting, we are given labeled samples ( x ( i ) , y ( i ) ) , where the covariates x ( i ) are drawn i.i.d. from a distribution on R d and the labels y ( i ) are (potentially noisy) evaluations of a linear function. The goal of the learner is to approximately recover the hidden regression vector. In this standard setting, linear regression is well-understood both statistically and computationally. Specifically, under Gaussian covariates with additive Gaussian noise, the least-squares estimator is computationally efficient and statistically optimal.

In many real-world scenarios, the input data is subject to some form of contamination, e.g., errors due to skewed and corrupted measurements, making even simple statistical estimation tasks algorithmically challenging. In the context of linear regression, classical computationally efficient estimators inherently fail in the presence of data contamination. An important goal in this context is to understand the possibilities and limitations of computationally efficient estimation in the presence of contaminated data.

∗ The majority of this work was done while the author was at the Simons Institute, UC Berkeley.

## Chao Gao

University of Chicago chaogao@uchicago.edu

## John Lafferty

Yale University john.lafferty@yale.edu

In this work, we study the fundamental problem of linear regression with Gaussian covariates in the presence of oblivious additive contamination in the responses (see Definition 1.1). In the oblivious contamination model, an adversary is allowed to corrupt a (1 -α ) -fraction of the labels (by adding an adversarially selected value to the label), for some parameter α &gt; 0 , and is limited in their capability by requiring the contamination be independent of the samples. Interestingly, the oblivious model information-theoretically allows for consistent estimation even for α → 0 . This stands in contrast to the more challenging model of adversarial contamination [Hub64; DK23], where non-trivial guarantees are impossible if more than half of the labels are corrupted.

To facilitate the subsequent discussion, we define our learning task below.

Definition 1.1 (Noiseless Linear Regression with Oblivious Contamination in Responses) . Let α ∈ (0 , 1) be the probability of inliers. Let E be a univariate distribution with P Z ∼ E ( Z = 0) ≥ α . For β ∈ R d , we denote by P β,E the distribution on labeled examples ( x, y ) ∈ R d × R defined as follows:

<!-- formula-not-decoded -->

Given i.i.d. samples { ( x i , y i ) } n i =1 from an unknown P β ∗ ,E , the goal is to construct an estimate ̂ β such that ∥ ̂ β -β ∗ ∥ 2 is small.

The model of Definition 1.1 goes back to the work of Candes and Tao [CT05], who studied it (for more general design matrices) as a classical example of error correction. It is also a standard model in face recognition [WM10], image inpainting [NT13], privacy-preserving data analysis [DMT07], and model repair [GL20]. A basic result in this area is that the true β can be recovered exactly, as long as the design matrix satisfies restricted isometry (therefore, for Gaussian design) and the number of nonzero entries of the noise is not too large (detailed below) [CT05; CRTV05; WM10; NT13; GL20]. Interestingly, Candes and Tao [CT05] noted that the model can also be recast as compressed sensing.

The statistical task of linear regression with Gaussian covariates under oblivious contamination has been extensively studied over the past decade [TJSO14; JTK14; BJK15; BJKK17; SBRJ19; PF20; DT19; dNS21]. The oblivious model has also been explored for other natural tasks, including PCA, sparse recovery [PF20; dLNNST21], and estimating a signal with additive oblivious contamination [dNNS22]. While most prior work has focused on Gaussian or subgaussian design matrices, a more recent line of investigation has developed efficient estimators in the distribution-free setting under mild assumptions [DKPT23a; DKPT23b].

Let us return to Definition 1.1 and discuss the precise quantitative aspects. Ignoring computational constraints, the sample complexity n required to obtain any non-trivial estimate of β ∗ for the problem of Definition 1.1 is n = d/α ; in fact, n = Θ( d/α ) samples suffice to estimate β ∗ exactly . In contrast, the best known computationally efficient algorithms require sample complexity of n = Ω( d/α 2 ) samples [GL20; dNS21]. Interestingly, known polynomial-time algorithms using n = O ( d/α 2 ) samples succeed even for the (more challenging) noisy version of the estimation task-where (in addition to oblivious contamination) the clean labels are perturbed by random observation noise (e.g., Gaussian noise). 2

While noisy linear regression with oblivious contamination information-theoretically requires Ω( d/α 2 ) samples, this is not the case for the noiseless version considered in this work-where, as mentioned above, O ( d/α ) samples suffice. This quadratic gap in 1 /α between the informationtheoretic optimum and the sample complexity of known polynomial-time algorithms can be significant in applications where the fraction of inliers α is small. Beyond practical considerations, given the fundamental nature of this estimation problem, it is natural to ask whether a computationally efficient algorithm with (near-)optimal sample complexity (i.e., within logarithmic factors of the optimal) exists. This leads to the central question motivating our work:

Question 1.2. Does there exist a constant c &gt; 0 so that for all d ∈ N , α ∈ (0 , 1) , there exists an algorithm, using O ( poly( d ) α 2 -c ) samples and running in poly( d, n ) time, that computes an estimate ̂ β such that ∥ ̂ β -β ∗ ∥ 2 is small?

2 To be precise, in the noisy version of the problem, the labels are of the form y = x ⊤ β + ξ + Z , where ξ ∼ N (0 , σ 2 ) . Definition 1.1 corresponds to the important special case of σ = 0 . For the noisy case of σ &gt; 0 , the information-theoretic error rate is ∥ ̂ β -β ∗ ∥ 2 = Θ ( σ · √ d nα 2 ) .

Our main result answers this question in the negative for efficient Statistical Query (SQ) algorithms-a broad and well-studied family of algorithms.

## 1.1 Main Result

To establish our negative result, we shall show that even the following (easier) testing task is computationally hard for SQ algorithms:

Testing Problem 1.3 (Testing Version of Linear Regression with Oblivious Contamination) . Let ρ &gt; 0 be the signal strength and α ∈ (0 , 1) be the inlier probability. Let E be a (known) univariate distribution on R that assigns at least α probability to 0 . Let R ∗ ρ,E be the univariate distribution of G + z , where G ∼ N (0 , ρ 2 ) and z ∼ E independently. The algorithm gets sample access to a distribution ( x, y ) ∼ Θ with the goal of distinguishing:

- 'Null': Θ = P , where under P : x ∼ N (0 , I d ) and y ∼ R ∗ ρ,E independently.
- 'Alternate': First a unit vector v is sampled uniformly, and then conditioned on v , Θ = Q v , where under Q v : x ∼ N (0 , I d ) and y = ρv ⊤ x + z , where z ∼ E is independent of x .

We say that an algorithm A succeeds if the failure probability of A is less than 1 / 10 under both the 'null' and the 'alternate'.

Note that, under the null hypothesis, the features x and the responses y are independent of each other; while under the alternate hypothesis, they follow the distribution P β,E of Definition 1.1 with ∥ β ∥ 2 = ρ . We show in Appendix C that a (computationally-efficient) estimation algorithm for the task of estimating β with error ρ/ 4 suffices to (computationally-efficiently) solve the testing problem above.

Proposition 1.4 (Efficient Reduction of Testing to Estimation; Informal) . If there exists a computationally-efficient algorithm to compute ̂ β with ∥ ̂ β -β ∗ ∥ ≤ ρ/ 4 with high probability, then it can be transformed into a computationally-efficient algorithm for Testing Problem 1.3.

Basics on SQ Algorithms. Instead of getting sample access, SQ algorithms [Kea98; FGRVX17] interact with the underlying distribution D through the following oracle.

Definition 1.5 (VSTAT Oracle) . Let D be a distribution on X . A statistical query is a bounded function f : X → [0 , 1] . For a 'simulation complexity' m ∈ N , a VSTAT(m) oracle for the distribution D on the input f returns a value v such that | v -E D [ f ] | ≤ max { 1 /m, √ ( E D [ f ](1 -E D [ f ])) /m } .

That is, the VSTAT ( m ) oracle returns an estimate of E D [ f ] with error comparable to the deviation in Bernstein's inequality for high-probability estimates of taking m i.i.d. samples from the Bernoulli distribution with bias E D [ f ] . We thus refer to m as the simulation complexity.

A Statistical Query (SQ) algorithm is an algorithm whose objective is to learn some information about an unknown distribution D by making adaptive calls to the corresponding VSTAT oracle. The complexity of an SQ algorithm is quantified by the total number of queries to the VSTAT oracle (viewed as a measure of the algorithm's running time) and the maximum simulation complexity of any such query (viewed as a measure of the algorithm's sample complexity).

In the context of our learning problem (Definition 1.1), it is worth pointing out the following. First, there exists an inefficient SQ algorithm with small simulation complexity, which in particular can be simulated using ˜ O ( d/α ) many i.i.d. samples (see Appendix D). Second, there exist efficient SQ algorithm whose simulation complexity matches the sample complexity ˜ O ( d/α 2 ) of known efficient algorithms (see Appendix E).

With this context, our main result is the following:

Theorem 1.6 (SQ Hardness of Testing Problem 1.3; informal) . Consider the Testing Problem 1.3. Suppose that (i) α ≫ 1 d polylog( d ) (i.e., the fraction of inliers is not too tiny) and (ii) ρ = ˜ Θ( α ) . Then there exists a distribution E satisfying P Z ∼ E ( Z = 0) ≥ α such that any SQ algorithm that solves Testing Problem 1.3 either

- uses d Ω(log 2 ( d/α )) many queries, or
- uses at least one query to VSTAT ( m ) for m = ˜ Ω( √ d/α 2 ) .

Informally speaking, Theorem 1.6 shows that no SQ algorithm can solve the testing problem (and, via Proposition 1.4, the estimation problem of approximating β ∗ ) with less than super-polynomial in d many queries, unless using queries whose simulation complexity is at least ˜ Ω( √ d/α 2 ) . That is, either the algorithm 'uses' ˜ Ω( √ d/α 2 ) many 'samples' (in the sense of simulation complexity mentioned above) or it takes super-polynomial 'time' (in the sense of number of queries). We thus obtain evidence that the quadratic dependence in 1 /α on the sample size is required for computationally efficient algorithms.

It is worth noting that the SQ-hard instances that we construct for the testing problem are efficiently solvable with ˜ O ( √ d/α 2 ) samples. We conjecture that the correct dependence on d is in fact linear (i.e., an Ω( d/α 2 ) lower bound on the computational sample complexity). This is left as an interesting question for future work (see Section 4).

Finally, while the focus of this work is on the SQ model, SQ-hardness results typically translate to quantitatively similar hardness for low-degree polynomial tests [Hop18; KWB19], via the work of [BBHLS21]. While we do not establish a formal theorem in this regard, we believe that our SQ-hard instances are also hard for low-degree polynomials.

## 1.2 Overview of Techniques

We wish to show that it is hard to solve Testing Problem 1.3 with fewer than √ d/ρ 2 samples (we will ultimately set ρ = ˜ Θ( α ) ). The first question we face is to make a judicious choice of the contamination distribution E that (I) satisfies our noise model, namely P Z ∼ E ( Z = 0) ≥ α ; and (II) it is SQ-hard to distinguish the null and alternate hypotheses.

Choice of Contamination Distribution: Intuition. A natural first step to consider is what happens if we select the contamination distribution E to be the standard Gaussian, i.e., E = N (0 , 1) . In this case, the testing task corresponding to Testing Problem 1.3 is information-theoretically impossible with o ( √ d/ρ 2 ) samples. Unfortunately, this choice does not fit our criterion (I), requiring that the contamination distribution must be exactly 0 with probability at least α .

Inspired from the information-theoretic sample complexity lower bound for the Gaussian contamination setting, we instead consider a scenario where the contamination is given by a distribution E , which is a discrete Gaussian with spacing s (see Definition 2.9). Heuristically, the discrete Gaussian approximately matches its low-degree moments with the continuous Gaussian case, and thus, prior work [DKS17] hints that it is SQ-hard to distinguish between the cases of discrete Gaussian and continuous Gaussian contamination. Since the case of continuous Gaussian contamination information-theoretically requires Ω( √ d/ρ 2 ) samples, intuitively we are moving in the right direction. Note that the aforementioned discrete Gaussian E assigns probability Ω( s ) to 0 . Taking s = Θ( α ) , we simultaneously satisfy criterion (I) above and have a reasonable chance of satisfying (II).

The above is the key intuitive idea underlying our proof. However, there are a number of important technical steps required to make the analysis work towards satisfying (II).

Discrete Noise and Non-Gaussian Component Analysis. For a unit vector v , let Q v be the distribution over ( x, y ) such that y = ρv ⊤ x + Z , where Z ∼ E independently of x and E is the suitable discrete Gaussian distribution. Let P be the distribution over ( x, y ) corresponding to the null hypothesis, namely x and y are independent with correct marginals (i.e., x ∼ N (0 , I d ) and y ∼ ρ 2 G + Z , where G ∼ N (0 , 1) and Z ∼ E are independent). We wish to show that it is SQ-hard to distinguish between Q v , for random v , and P . Note that conditioning on the value of y , Q v is a standard Gaussian in the directions orthogonal to v and is given by some known distribution, A y , in the v -direction. This means that the testing problem we are considering is effectively a conditional Non-Gaussian Component Analysis (NGCA) problem (Testing Problem 2.8). Unfortunately, there are several technical obstacles preventing us from applying existing tools [DKS17; DKS19].

The first technical hurdle arises from the fact that A y is a discrete distribution, and in particular has infinite chi-squared norm with respect to the standard Gaussian. In particular, this means that the standard SQ-dimension related techniques for proving lower bounds will not work here. Instead, we need to leverage and adapt the recent work of [DKRS23] that directly uses Gaussian Fourier analysis to establish SQ lower bounds even when the chi-squared distance is infinite. Unfortunately, the latter

work [DKRS23] does not give SQ-lower bounds for conditional Non-Gaussian Component Analysis tasks (as the one we are dealing with here). Consequently, we will require a careful adaptation of their techniques in our context.

Connection with Continuous Gaussian Contamination. A key requirement for the Gaussian Fourier analysis to go through in [DKRS23] is that A y 's have well-behaved moments. Unfortunately, an additional technical challenge arising in our context is that directly bounding the relevant moments of A y (which belongs to the family of discrete Gaussians) is challenging.

Instead, for the purpose of the analysis, we again leverage the connection with continuous Gaussian contamination. Specifically, we choose B y to be a continuous Gaussian counterpart of the discrete Gaussian A y . Let the resulting distribution on ( x, y ) be T v (which is a continuous counterpart of Q v ). Note that this is again an instance of conditional NGCA. Since the B y 's are now (continuous) Gaussians (and hence satisfy many desirable properties, e.g., continuity), it can be shown that if v and w are nearly orthogonal vectors, T v and T w will have small chi-squared inner product with respect to P .

Hardness of Continuous Noise Contamination. The fact that two random unit vectors have small inner product with high probability can be used to show that the task of testing between P and { T v } v ∼S d -1 has large SQ dimension. This implies SQ-hardness of this basic testing problem. In fact, it will imply the more powerful result that for any bounded function f , with high probability over v , the expectations E T v [ f ] and E P [ f ] cannot be distinguished by a VSTAT ( o ( m 0 )) query for m 0 := √ d/ ( ρ 2 · log 4 d ) ; see Proposition 3.6.

Quantitative Relationship between Discrete and Continuous Gaussian Noise. We now return to the challenge of computing moments of discrete Gaussians A y (for performing Gaussian Fourier analysis). We resolve this issue by comparing these moments to the moments of B y . As A y will be a discrete version of the Gaussian B y , this relationship will be relatively manageable to prove. We then combine this ingredient with techniques involving Hermite analysis from [DKRS23] to show the following: for any bounded test function f , with high probability over the choice of a random v , it holds that | E Q v [ f ] -E T v [ f ] | is tiny (inverse super-polynomial in m 0 ) as long as s ≪ ρ polylog( d ) (Theorem 3.7).

Putting Everything Together. Combining the above, we obtain the following: for any f , with high probability over random v , it holds that (i) | E Q v [ f ] -E T v [ f ] | is inverse super-polynomially small in m 0 , and (ii) | E T v [ f ] -E P [ f ] | is smaller than the threshold for VSTAT ( o ( m 0 )) . Therefore, by a union bound and a triangle inequality, it follows that with high probability | E Q v [ f ] -E P [ f ] | is also smaller than the threshold for VSTAT ( o ( m 0 )) , implying SQ-hardness (Proposition 2.6).

## 1.3 Related Work

Our work is broadly situated in the field of robust statistics, which has a long history dating back to Huber and Tukey [Hub64; Tuk60]. Robust statistics aims to design estimators that are tolerant to data contamination. Focusing on high-dimensional data, our work studies the statistical and computational aspects of robust estimation, which has seen a flurry of work in the last decade since [DKKLMS16; LRV16]; see [DK23] for a recent book on this topic. For designing robust estimators, the choice of contamination model naturally plays a crucial role. This work is part of a broader effort to understand computational and statistical aspects of natural, not fully adversarial, contamination models; see, e.g., [BJK15; BJKK17; ZJS19; DGT19; DK22; DKMR22; DKRS22; DKKTZ22; DKPT23a; DDKWZ23b; DDKWZ23a; MVBWS24; NGS24; PP24; DZ24; KG25; DIKP25].

Historically, the prototypical contamination model in robust statistics has been Huber's contamination model [Hub64], which was strengthened to total variation distance [Hub65] and strong contamination models [DKKLMS16]. The task of linear regression under these contamination models is now well understood both statistically [CGR16] and computationally [DKS19; PJL20; DKPP23]. As mentioned earlier, it is information-theoretically impossible to achieve consistency in these models if the proportion of contamination is bounded away from zero. Thus, an important direction is to understand the possibilities and limitations in other, less adversarial, contamination models. The oblivious adversary studied here is one such model, and indeed it does lead to consistent estimation even when the oblivious outliers constitute the majority of the observed data; see the discussion below Defini-

tion 1.1. Our work shows that while this weaker contamination model is benign from the perspective of information-theoretic rates, it does present surprising information-computation tradeoffs.

## 2 Preliminaries

For a univariate distribution E , we define R ∗ ρ,E to be the univariate distribution of G + z , where G ∼ N (0 , ρ 2 ) and z ∼ E independently. For two vectors v and w in R d , we use ⟨ v, w ⟩ to denote the standard inner product ∑ i ∈ [ d ] v i w i . A degreek tensor in d -dimensions v is an element in ( R d ) ⊗ k . with entries ( v i 1 ,...,i k ) i 1 ∈ [ d ] ,...,i k ∈ [ d ] . For a vector v , we use v ⊗ k to denote the k -tensor with entries ∏ d ℓ =1 v i ℓ . For two k -tensors v and w , we use ⟨ v , w ⟩ to denote the inner product ∑ i 1 ,...,i k v i 1 ,...,i k w i 1 ,...,i k and use ∥ v ∥ 2 := √ ⟨ v , v ⟩ . A k -tensor function F : X → ( R d ) ⊗ k maps each x ∈ X to a k -tensor.

Hermite Polynomials For a k ∈ N , we use h k : R → R to denote the k -th normalized probabilist's polynomial (which is a degreek polynomial with definition h k ( x ) := 1 √ k ! ( -1) k e x 2 / 2 d k dx k e -x 2 / 2 ). We shall also use the k -th Hermite tensor H k as defined in [DKRS23, Definition 2.2].

Fourier Analysis For a distribution P on a domain X , we use L 2 ( X , P ) to denote the space of all functions f : X → R with E x ∼ P [ f 2 ( x )] &lt; ∞ . For two functions f, g ∈ L 2 ( X , P ) , we use ⟨ f, g ⟩ P to denote the inner product E x ∼ P [ f ( x ) g ( x )] and ∥ f ∥ L 2 ( P ) to denote ⟨ f, f ⟩ P . For a function f : R d → R and an ℓ ∈ N , we define f ≤ ℓ to be the degreeℓ Hermite approximation function f ≤ ℓ ( x ) := ∑ ℓ k =0 ⟨ A k , H k ⟩ where A k := E x ∼ P [ f ( x ) H k ( x )] . We extend this definition to f : R d × R as follows: First, for each y ∈ R , we define f y : R d → R as x ↦→ f ( x, y ) and then define f ≤ ℓ ( x, y ) := f y ( x ) ≤ ℓ , that is, for each y , we perform degreeℓ approximation of f y . We use f &gt;ℓ := f -f ≤ ℓ to denote the residual.

Fact 2.1. For every function f : R d → [ -1 , 1] , ∥ f &gt;ℓ ∥ L 2 ( N (0 , I d )) → 0 as ℓ →∞ . Furthermore, for all f : R d × R → [ -1 , 1] and univariate measures R , ∥ f &gt;ℓ ∥ L 2 ( N (0 , I d ) × R ) → 0 as ℓ →∞ .

We use ˜ Ω , ˜ Θ notation to hide polylog( d, 1 /α ) factors. For two non-negative functions, a and b , we use a ≲ b (similarly a ≳ b ) to say that there exists a constant (independent of other problem parameters) such that a ≤ Cb (respectively, a ≥ Cb ) ; if a ≲ b and b ≳ a , then we say a ≍ b .

SQ Algorithms We state the preliminaries of SQ for the following generic testing problem.

Testing Problem 2.2 (Generic Testing Problem) . Let P and {Q v } v ∈S d -1 be distributions over a domain Z , which correspond to 'null' and 'alternate', respectively.

- First sample Γ ∼ Ber(1 / 2) and v ∼ S d -1 independently (unknown to the statistician).
- Then set Θ = P ('null') if Γ = 0 and Θ = Q v ('alternate') otherwise.

̸

- The statistician gets (either sample/oracle) access to the distribution Θ and generates ̂ Γ ∈ { 0 , 1 } using an algorithm A . We say A solves the testing problem if P ( ̂ Γ = Γ) ≤ 0 . 1 .

We say an SQ algorithm A solves a problem with query complexity q and accuracy complexity m if it iteratively (potentially also adaptively and randomly) makes queries f 1 , . . . , f q (each f i is bounded in [0 , 1] and could depend on the previous queries and their responses) on the underlying distribution ( Θ above) to a VSTAT ( m ) oracle. An SQ lower bound is an information-theoretic lower bound of the following form: any successful SQ algorithm A must have either q ≥ q 0 or m ≥ m 0 . In the remainder of this section, we detail the technical results for proving such lower bounds.

Definition 2.3 (Pairwise Correlation) . For a reference distribution P , and candidate distributions Q 1 and Q 2 , the pairwise correlation between Q 1 and Q 2 with respect to P is defined as χ P ( Q 1 , Q 2 ) := E Z ∼P [ q 1 ( Z ) q 2 ( Z ) p 2 ( Z ) -1 ] , where q 1 ( · ) , q 2 ( · ) , p ( · ) denote the densities of Q 1 , Q 2 , and P with respect to a common measure, respectively. When Q 1 = Q 2 , the pairwise correlation becomes the same as the χ 2 -divergence between Q 1 and P , i.e., χ 2 ( Q 1 , P ) = ∫ Z q 2 1 ( x ) p ( x ) dx -1 .

Statistical dimension is then defined using these pairwise correlations:

Definition 2.4 (Statistical dimension from [BBHLS21]) . The statistical dimension of Testing Problem 2.2 at accuracy complexity m is defined as:

<!-- formula-not-decoded -->

where (i) v, v ′ iid ∼ S d -1 independently and (ii) the inner supremum is taken over events E ⊂ S d -1 ×S d -1 on v and v ′ (i.i.d. from unit sphere) that have probability at least 1 /q 2 .

We now define the notion of success of a query f that will be useful to us:

Definition 2.5 (Success of a query on a distribution) . We say that a query f : Z → [0 , 1] succeeds on distinguishing Q v and P with accuracy complexity m , denoted by the event E f,v,m , if | E Q [ f ( Z )] -

<!-- formula-not-decoded -->

v .

We are now equipped to state the SQ lower bounds that we will use repeatedly.

Proposition 2.6 (SQ Lower Bound) . Consider Testing Problem 2.2. Then

- (C.I) For any query f : Z → [0 , 1] , P v ∼S d -1 ( E f,v.m ) ≤ 1 SDA(7 m ) .
- (C.II) Suppose for all queries f : Z → [0 , 1] , it holds that P v ∼S d -1 ( E f,v,m ) ≤ 1 q . Then any SQ algorithm A that solves Testing Problem 2.2 must use either Ω( q ) queries in expectation or at least one query as powerful as VSTAT ( m +1) .

Non-Gaussian Component Analysis Wewill primarily consider Testing Problem 2.2 of a particular form called Non-Gaussian Component Analysis (NGCA). We begin by defining High-Dimensional Hidden Direction Distribution:

Definition 2.7 (High-Dimensional Hidden Direction Distribution) . For a unit vector v ∈ R d and a distribution H on the real line, we define P H v to be the distribution over R d , where P H v is the product distribution whose orthogonal projection onto the direction of v is H , and onto the subspace perpendicular to v is the standard ( d -1) -dimensional normal distribution. In particular, if H is a continuous distribution with probability density function (pdf) H ( x ) , then P H v ( x ) has the pdf H ( v ⊤ x ) ϕ ⊥ v ( x ) , where ϕ ⊥ v ( x ) = exp ( -∥ x -( v ⊤ x ) v ∥ 2 2 / 2 ) / (2 π ) ( d -1) / 2 .

[DKS17] established SQ lower bounds for the NGCA problem, where the null and the alternate are N (0 , I d ) and { P H v } v ∼S d -1 , respectively and (i) H (nearly) matches many moments with N (0 , 1) and (ii) has finite χ 2 ( H , N (0 , 1)) . For linear regression, we would need the following generalization:

Testing Problem 2.8 (Conditional NGCA) . Let {H y } y ∈ R be a family of univariate distributions and R be a univariate distribution. Consider Testing Problem 2.2 over ( x, y ) on the domain R d × R with

- ('Null') Under P : x ∼ N (0 , I d ) and y ∼ R independently.
- ('Alternate') Under Q v : y ∼ R and conditioned on y = y 0 , X | y = y 0 ∼ P H y 0 v .

Building on [DKS17], [DKS19] showed SQ-hardness for the problem above if (i) H y matches moments with N (0 , 1) for (nearly) all y ∈ R and (ii) χ 2 ( Q v , P ) &lt; ∞ . Unfortunately, neither of these conditions holds for us, and we need more flexible and powerful tools to bypass these limitations.

Discrete Gaussian We define Discrete Gaussian distributions that are central in our analysis. Definition 2.9. For a center µ ∈ R , deviation σ &gt; 0 , base θ , and spacing s &gt; 0 , define the distribution DG ′ [ µ, σ, θ, s ] to be the positive measure over θ + s Z that assigns mass sϕ µ,σ ( θ + si ) for all i ∈ Z ; here, ϕ µ,σ denotes the pdf of the Gaussian distribution with mean µ and standard deviation σ . We use DG [ µ, σ, θ, s ] to denote the normalized probability distribution.

Discrete Gaussians behave similarly to Gaussians with respect to low-degree polynomials:

Fact 2.10 ([DKRS23, Fact C.3] and [DK22, Lemma 3.12]) . For any polynomial p of degree at most k and θ ∈ R and s &gt; 0 , we have that ∣ ∣ E G ∼N (0 , 1) [ p ( G )] -E Y ∼ DG [0 , 1 ,θ,s ] [ p ( Y )] ∣ ∣ ≲ √ E G ∼N (0 , 1) [ p 2 ( G )] k !2 O ( k ) exp( -Ω(1 /s 2 )) .

## 3 Proof of Theorem 1.6

In this section, we will prove our main result Theorem 1.6. The first step is to make a judicious choice of the noise distribution E . For reasons outlined in Section 1.2, we choose E to be a discrete Gaussian with σ 2 ≈ 1 and spacing s (eventually set to ˜ Θ( α ) ).

As the second step, we note that the resulting testing problem is an instance of conditional NGCA.

Testing Problem 3.1 (NGCA with Discrete Gaussian) . For y ∈ R , define the distribution A y := DG [ µ y , ˜ σ, θ y , s ′ ] with parameter values in Definition 3.2. Consider Testing Problem 2.8 with

- (Marginal of y ) R := R ∗ ρ,E with E = DG [ 0 , σ, 0 , s ] . 3
- (Conditional NGCA) For each y ∈ R , H y is equal to A y .

We denote the corresponding null by P and the alternate for direction v by Q v .

We mention the parameter choices that we shall enforce from now on:

Definition 3.2. Let signal strength ρ ∈ (0 , ρ 0 ) for sufficiently small ρ 0 &gt; 0 , standard deviation σ ∈ (0 . 5 , 1) , spacing s ∈ (0 , 1) satisfy the following values:

- σ = √ 1 -ρ 2 ,
- s ′ = s/ρ ≤ 0 . 001 ,
- µ y := ρy ,
- θ y = y/ρ .

That is, for each y , the conditional distribution of the covariates in the hidden direction is a discrete Gaussian with mean µ y (scaling linearly with y ) and standard deviation σ (slightly smaller than 1 ). While these parameters might look a bit obscure, they perfectly resemble the typical setting of E = N (0 , σ 2 ) . 4 The next result, proved in Appendix B.1, shows that Testing Problem 1.3 is equivalent to Testing Problem 3.1.

Proposition 3.3. Testing Problem 3.1 is equivalent to Testing Problem 1.3 when E = DG [ 0 , σ, 0 , s ] .

Thus, to prove Theorem 1.6, it suffices to consider Testing Problem 3.1, which is a conditional NGCA instance. Since the distributions { A y } y ∈ R are (necessarily) degenerate, the lower bound machinery of SDA and pairwise correlations developed in [DKS17; DKS19] for (conditional) NGCA lead only to vacuous bounds. To bypass this degeneracy, we will instead use Proposition 2.6 (C.II) and will show that for any bounded query f : U → [0 , 1] ,

<!-- formula-not-decoded -->

where the notion of being 'large' is according to Definition 2.5 for m = ˜ o ( ρ 2 / √ d ) . However, it is unwieldy to compute (or upper bound) this probability. Hence, we first take a detour to a related testing problem with the more usual continuous Gaussian noise in the next section.

## 3.1 Conditional NGCA with Continuous Gaussian

As mentioned in the introduction, we use the similarity of discrete Gaussian with continuous Gaussian (with respect to polynomials) as an analysis tool. We define the analogous testing problem with continuous Gaussian noise below.

Testing Problem 3.4. For y ∈ R , let B y denote the distribution N ( µ y , σ 2 ) with parameters as in Definition 3.2. Consider Testing Problem 2.8 with

- (Marginal of y ) R := R ∗ ρ,E with E = DG [ 0 , σ, 0 , s ] .
- (Conditional NGCA) For each y ∈ R , H y is equal to B y .

We denote the corresponding null by P (same as 3.1) and the alternate for direction v by T v .

Remark 3.5. Observe that the alternate above T v does not correspond to the following (Gaussian) linear model: y = ρv ⊤ x + z for x ∼ N (0 , I d ) and z ∼ N (0 , σ 2 ) independently of x . This is because the marginal of Y under the aforementioned linear model would have been Gaussian N (0 , 1) , while it is R in Testing Problem 3.4 (which is not Gaussian).

Before establishing similarity with discrete Gaussian quantitatively, we first establish that Testing Problem 3.4 is SQ-hard. In fact, we show the stronger result that the associated SDA is large.

Proposition 3.6 (SQ Hardness of Continuous Noise) . Consider Testing Problem 3.4. Then for any m ∈ N and q ∈ N satisfying ρ 2 √ log(1 /q ) √ d ≲ 1 m , we have that SDA( m ) ≳ q .

3 Recall that R ∗ ρ,E is the distribution of x + z for x ∼ N (0 , ρ 2 ) and z ∼ E independent of each other.

4 The conditional distribution of x | y in the hidden direction v would be N ( µ y , σ 2 ) [DKPPS21, Fact 3.3].

Proof Sketch. We prove a bound on SDA by calculating an analytic upper bound on the pairwise correlation χ P ( T v , T v ′ ) . Since the marginal of y is identical ( R ) under P, T v , and T v ′ , the pairwise correlation is equal to E y ∼ R [ χ N (0 , I d ) ( P B y v , P B y v ′ )] . Since P B y v and P B y v ′ are Gaussians, there is a closed-form expression (in terms of y ), which we integrate out using nice properties of R .

As a consequence, Proposition 2.6 (C.I) implies that for any f : U → [0 , 1] and m = o ( ρ 2 log 4 d √ d ) ,

<!-- formula-not-decoded -->

## 3.2 Hardness of Distinguishing Discrete and Gaussian Noise

Towards establishing (1), a natural step after proving (2) is to argue that, with high probability, | E T v [ f ( Z )] -E Q v [ f ( Z )] | is small. This is exactly what we establish in the next result, which is our main technical result:

Theorem 3.7. Suppose that (i) α ≫ 1 d polylog( d ) and ρ 2 ≥ s 2 log C ( d/α ) for a large constant C &gt; 0 . Then for any f : U → [0 , 1] , it holds that P v ∼S d -1 [∣ ∣ ∣ E Q v [ f ] -E T v [ f ] ∣ ∣ ∣ ≳ ( α d ) log 2 ( d/α ) ] ≤ 1 d log 2 d .

In the remainder of this section, we detail the proofs and intuition for the above result.

As a first step, we do a Hermite expansion of the function f as in [DKRS23], but generalized to the setting of conditional NGCA. However, for technical reasons due to the degeneracy of A y and hence Q v , we would need to perform another truncation operation.

Definition 3.8. Define ˜ A y to be the univariate distribution A y conditioned on { z : | z | ≤ d } and let ˜ Q v to be analogous to Q v but with ˜ A y instead of A y .

We now use the Hermite expansion to obtain the following result:

<!-- formula-not-decoded -->

Proof Sketch. Since R ρ,E has very light tails, we can replace f with ˜ f which leads to a difference of at most P ( | y | ≥ L ) ≲ e -Ω( L 2 ) .

Next, we decompose ˜ f as ˜ f ≤ ℓ and ˜ f &gt;ℓ , where the ˜ f &gt;ℓ term appears as is in (3) and can be ignored momentarily. Then, using law of total expectation, we can write E ( x,y ) ∼ TQ v [ ˜ f ≤ ℓ ] = E y [ E x [ ˜ f ≤ ℓ y ( x )]] . The result in [DKRS23, Lemma 3.3] implies that E P Ay v [ ˜ f ≤ ℓ y ( x )] = ∑ ℓ k =0 A k,y 〈 v ⊗ k , T k,y 〉 for A k,y := E x ∼ A y [ h k ( x )] . A similar argument holds for B y . Taking the difference and integrating over y , we obtain E Q v [ ˜ f ≤ ℓ ] -E T v [ ˜ f ≤ ℓ ] = E y [∑ ℓ k =1 ( A ′ k,y -B k,y ) ⟨ v ⊗ k , T k,y ⟩ ] .

Since ˜ f is zero for | y | ≥ L , T k,y is also zero for large y and we can take the maximum only over | y | ≤ L , yielding (3) roughly. However, later on, we would still need to control E Q [ ˜ f &gt;ℓ ] , which could potentially be large because of degeneracy and unboundedness of A y s. Therefore, we replace A y s with ˜ Q y s to make it bounded; using concentration of A y s, this leads to an additional e -Ω( d 2 ) term.

Thus, we crucially need to control | ˜ A k,y -B k,y | and obtain high-probability estimates (over randomness in v ) on |⟨ v ⊗ k , T k,y ⟩| . We begin with the former, whose proof is deferred to Appendix B.5; we note that the key ingredient in proving this result is Fact 2.10.

Lemma 3.10 (Closeness of Hermite Coefficients) . For any y ∈ R and k ∈ N , we have

- (Tighter for small k ) | ˜ A k,y -B k,y | ≲ max ( 1 , | µ y | k ) k O ( k ) · ( e -Ω ( ρ 2 s 2 ) + e -Ω( d ) ) .
- (Tighter for larger k ) | ˜ A k,y -B k,y | ≲ e O ( µ 2 y ) .

Thus, Lemma 3.10 implies that (i) for small k , the difference is inverse super-polynomially small if ρ 2 ≫ s 2 polylog( d ) ≍ α 2 polylog( d ) and (ii) it stays bounded by O (1) e L 2 for | y | ≤ L for any k .

We now turn to computing high-probability estimates on E y |⟨ v ⊗ k , T k,y ⟩| . Here, we reparameterize the arguments in [DKRS23] and obtain the following result, whose proof is deferred to Appendix B.6.

Proposition 3.11. Let { T k,y } k ∈ N ,y ∈ R be tensors with ∥ T k,y ∥ 2 ≤ 1 for all k ∈ N , y ∈ R , and let t ∈ N be arbitrary . Then for any δ ∈ (0 , 1) , it holds with probability 1 -δ over a random unit vector v that

<!-- formula-not-decoded -->

The result above is applicable to our setting because for each y ∈ R : ∑ ∞ k =1 ∥ T k,y ∥ 2 2 = ∥ ˜ f y ∥ 2 L 2 ( N (0 , I d ) ≤ 1 , where the equality uses the orthonormality of Hermite tensors under N (0 , I d ) and the inequality uses that ˜ f is bounded by 1 .

## 3.2.1 Proof sketch of Theorem 3.7

We are now ready to present a proof sketch of Theorem 3.7. Combining Proposition 3.9 with Lemma 3.10 and Proposition 3.11 and the fact that | µ y | ≤ L for L ≥ 1 , we obtain that for any t ∈ N with probability at least 1 -d -log 2 d ,

<!-- formula-not-decoded -->

For L = log 5 d , t = L 6 and ρ = st 2 , the sum of the first four terms is at most O ( e -L 2 ) ≤ d -log 2 ( d/α ) . For the last term, we show that taking ℓ large enough suffices-this argument uses Fact 2.1 and the truncation of A y as per [DKRS23]; see Appendix B.8 for details.

## 3.3 Proof Sketch of Theorem 1.6

Since E = DG [ 0 , σ, 0 , s ] , we have that P E ( z = 0) = Θ( s/σ ) up to normalization 1 ± e -1 /s 2 ; see Fact 2.10. Taking s = Θ( α ) , we get that P ( z = 0) ≥ α satisfying our model. To establish SQ lower bound, it suffices to show that the probability of success of f on distinguishing Q v and P with m ≪ m 0 := ˜ Θ( √ d/α 2 ) accuracy complexity is at most 1 /q 0 := d -Ω(log 2 d ) (cf. Proposition 2.6). Let this event be E f,v,m . We now define the following events:

- E ′ f,v,m := { v : ∣ ∣ E Q v [ f ( x, y )] -E T v [ f ( x, y )] ∣ ∣ ≥ 1 4 m 2 } ; Theorem 3.7 implies that P ( E ′ f,v,m ) ≤ 1 2 q 0 .
- E ′′ f,v,m is defined analogous to E f,v,m but with (i) T instead of Q and (ii) Cm accuracy complexity as opposed to m for a large constant C . Proposition 3.6 implies P ( E ′′ f,v,m ) ≤ 1 2 q 0 .

Since E f,v ⊂ E ′ f,v,m ∪ E ′′ f,v,m (Claim B.7), the desired result follows by a union bound.

## 4 Conclusions and Open Problems

In this work, we studied the fundamental problem of noiseless linear regression under Gaussian marginals with additive oblivious contamination. Our main result is an information-computation tradeoff for SQ algorithms, suggesting that efficient learners require sample complexity at least quadratic in 1 /α , where α is the fraction of inliers, while linear dependence in 1 /α informationtheoretically suffices. An immediate open problem concerns the dependence on d in the lower bound. Specifically, it is a plausible conjecture that there exists a lower bound of Ω( d/α 2 ) on the computational sample complexity of the problem (thus, exactly matching the sample complexity of known algorithms). We note that such a lower bound would require a new hardness construction, as our hard testing instance is efficiently solvable with O ( d 1 / 2 /α 2 ) samples.

## Acknowledgements

ID was supported by NSF Medium Award CCF-2107079 and an H.I. Romnes Faculty Fellowship. CG was supported by NSF Grants ECCS-2216912 and DMS-2310769 and an Alfred Sloan fellowship. DK was supported by NSF Medium Award CCF-2107547. AP was supported by Research Pod on Resilience in Brain, Natural, and Algorithmic Systems at the Simons Institute, UC Berkeley.

## References

| [BBHLS21]   | M. Brennan, G. Bresler, S. B. Hopkins, J. Li, and T. Schramm. 'Statistical query algorithms and low-degree tests are almost equivalent'. In: Proc. 34th Annual Conference on Learning Theory (COLT) . 2021.                                                                                                                        |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [BJK15]     | K. Bhatia, P. Jain, and P. Kar. 'Robust Regression via Hard Thresholding'. In: Advances in Neural Information Processing Systems 28 (NeurIPS) . 2015, pp. 721- 729.                                                                                                                                                                |
| [BJKK17]    | K. Bhatia, P. Jain, P. Kamalaruban, and P. Kar. 'Consistent Robust Regression'. In: Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017 . 2017, pp. 2107-2116.                                                                                                   |
| [BLM13]     | S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: A Nonasymp- totic Theory of Independence . Oxford University Press, 2013.                                                                                                                                                                                     |
| [CGR16]     | M. Chen, C. Gao, and Z. Ren. 'A General Decision Theory for Huber's $\epsilon$- Contamination Model'. In: Electronic Journal of Statistics 10.2 (2016), pp. 3752- 3774.                                                                                                                                                            |
| [CRTV05]    | E. Candes, M. Rudelson, T. Tao, and R. Vershynin. 'Error correction via linear programming'. In: 46th Annual IEEE Symposium on Foundations of Computer Science (FOCS'05) . IEEE. 2005, pp. 668-681.                                                                                                                                |
| [CT05]      | E. J. Candes and T. Tao. 'Decoding by linear programming'. In: IEEE transactions on information theory 51.12 (2005), pp. 4203-4215.                                                                                                                                                                                                |
| [DDKWZ23a]  | I. Diakonikolas, J. Diakonikolas, D. Kane, P. Wang, and N. Zarifis. 'Near-Optimal Bounds for Learning Gaussian Halfspaces with Random Classification Noise'. In: Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, 2023 . 2023.                 |
| [DDKWZ23b]  | I. Diakonikolas, J. Diakonikolas, D. M. Kane, P. Wang, and N. Zarifis. 'Information- Computation Tradeoffs for Learning Margin Halfspaces with Random Classification Noise'. In: The Thirty Sixth Annual Conference on Learning Theory, COLT 2023 . Vol. 195. Proceedings of Machine Learning Research. PMLR, 2023, pp. 2211-2239. |
| [DGT19]     | I. Diakonikolas, T. Gouleakis, and C. Tzamos. 'Distribution-Independent PAC Learning of Halfspaces with Massart Noise'. In: Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019 . 2019, pp. 4751-4762.                                             |
| [Die01]     | T. E. Dielman. Applied Regression Analysis for Business and Economics . Duxbury/Thomson Learning, 2001.                                                                                                                                                                                                                            |
| [DIKP25]    | I. Diakonikolas, G. Iakovidis, D. M. Kane, and T. Pittas. 'Efficient Multivariate Robust Mean Estimation Under Mean-Shift Contamination'. In: Proc. 42nd Inter- national Conference on Machine Learning (ICML) . 2025.                                                                                                             |
| [DK22]      | I. Diakonikolas and D. M. Kane. 'Near-Optimal Statistical Query Hardness of Learning Halfspaces with Massart Noise'. In: Proc. 35th Annual Conference on Learning Theory (COLT) . 2022.                                                                                                                                            |
| [DK23]      | I. Diakonikolas and D. M. Kane. Algorithmic High-Dimensional Robust Statistics . Cambridge University Press, 2023.                                                                                                                                                                                                                 |
| [DKKLMS16]  | I. Diakonikolas, G. Kamath, D. M. Kane, J. Li, A. Moitra, and A. Stewart. 'Robust Estimators in High Dimensions without the Computational Intractability'. In: Proc. 57th IEEE Symposium on Foundations of Computer Science (FOCS) . 2016.                                                                                         |
| [DKKTZ22]   | I. Diakonikolas, D. M. Kane, V. Kontonis, C. Tzamos, and N. Zarifis. 'Learning general halfspaces with general Massart noise under the Gaussian distribution'. In: STOC '22: 54th Annual ACM SIGACT Symposium on Theory of Computing, 2022 . ACM, 2022, pp. 874-885.                                                               |

| [DKMR22]   | I. Diakonikolas, D. Kane, P. Manurangsi, and L. Ren. 'Cryptographic Hardness of Learning Halfspaces with Massart Noise'. In: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, 2022 . 2022.   |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [DKPP23]   | I. Diakonikolas, D. M. Kane, A. Pensia, and T. Pittas. 'Near-Optimal Algorithms for Gaussians with Huber Contamination: Mean Estimation and Linear Regression'. In: Advances in Neural Information Processing Systems 36 (NeurIPS) . 2023.                                       |
| [DKPPS21]  | I. Diakonikolas, D. M. Kane, A. Pensia, T. Pittas, and A. Stewart. 'Statistical query lower bounds for list-decodable linear regression'. In: Advances in Neural Information Processing Systems 34 (NeurIPS) . 2021.                                                             |
| [DKPT23a]  | I. Diakonikolas, S. Karmalkar, J. Park, and C. Tzamos. 'Distribution-Independent Regression for Generalized Linear Models with Oblivious Corruptions'. In: Proc. 36th Annual Conference on Learning Theory (COLT) . 2023.                                                        |
| [DKPT23b]  | I. Diakonikolas, S. Karmalkar, J. Park, and C. Tzamos. 'First Order Stochastic Optimization with Oblivious Noise'. In: Advances in Neural Information Processing Systems 36 (NeurIPS) . 2023.                                                                                    |
| [DKRS22]   | I. Diakonikolas, D. Kane, L. Ren, and Y. Sun. 'SQ Lower Bounds for Learning Single Neurons with Massart Noise'. In: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022 . 2022.                  |
| [DKRS23]   | I. Diakonikolas, D. Kane, L. Ren, and Y. Sun. 'SQ Lower Bounds for Non-Gaussian Component Analysis with Weaker Assumptions'. In: Advances in Neural Informa- tion Processing Systems 36 (NeurIPS) . 2023.                                                                        |
| [DKS17]    | I. Diakonikolas, D. M. Kane, and A. Stewart. 'Statistical Query Lower Bounds for Robust Estimation of High-Dimensional Gaussians and Gaussian Mixtures'. In: Proc. 58th IEEE Symposium on Foundations of Computer Science (FOCS) . 2017.                                         |
| [DKS19]    | I. Diakonikolas, W. Kong, and A. Stewart. 'Efficient Algorithms and Lower Bounds for Robust Linear Regression'. In: Proc. 30th Annual Symposium on Discrete Algorithms (SODA) . 2019.                                                                                            |
| [dLNNST21] | T. d'Orsi, C. H. Liu, R. Nasser, G. Novikov, D. Steurer, and S. Tiegel. 'Consistent Estimation for PCA and Sparse Regression with Oblivious Outliers'. In: Advances in Neural Information Processing Systems 34 (NeurIPS) . 2021.                                                |
| [DMT07]    | C. Dwork, F. McSherry, and K. Talwar. 'The price of privacy and the limits of LP decoding'. In: Proceedings of the thirty-ninth annual ACM symposium on Theory of computing . 2007, pp. 85-94.                                                                                   |
| [dNNS22]   | T. d'Orsi, R. Nasser, G. Novikov, and D. Steurer. 'Higher degree sum-of-squares relaxations robust against oblivious outliers'. In: CoRR abs/2211.07327 (2022).                                                                                                                  |
| [dNS21]    | T. d'Orsi, G. Novikov, and D. Steurer. 'Consistent regression when oblivious outliers overwhelm'. In: Proc. 38th International Conference on Machine Learning (ICML) . Ed. by Marina Meila and Tong Zhang. 2021.                                                                 |
| [DT19]     | A. Dalalyan and P. Thompson. 'Outlier-robust estimation of a sparse linear model using ℓ 1 -penalized Huber's M -estimator'. In: Advances in neural information processing systems 32 (2019).                                                                                    |
| [DZ24]     | I. Diakonikolas and N. Zarifis. 'A Near-optimal Algorithm for Learning Margin Halfspaces with Massart Noise'. In: Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024 . 2024.              |
| [FGRVX17]  | V. Feldman, E. Grigorescu, L. Reyzin, S. S. Vempala, and Y. Xiao. 'Statistical Algorithms and a Lower Bound for Detecting Planted Cliques'. In: Journal of the ACM 64.2 (2017).                                                                                                  |
| [FGV17]    | V. Feldman, C. Guzman, and S. S. Vempala. 'Statistical Query Algorithms for Mean Vector Estimation and Stochastic Convex Optimization'. In: Proc. 28th Annual Symposium on Discrete Algorithms (SODA) . 2017.                                                                    |
| [GL20]     | C. Gao and J. Lafferty. 'Model repair: Robust recovery of over-parameterized statistical models'. In: arXiv preprint arXiv:2005.09912 (2020).                                                                                                                                    |

| [Hop18]   | S. B. Hopkins. 'Statistical inference and the sum of squares method'. PhD thesis. Cornell University, 2018.                                                                                                                 |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Hub64]   | P. J. Huber. 'Robust Estimation of a Location Parameter'. In: The Annals of Mathe- matical Statistics 35.1 (Mar. 1964), pp. 73-101.                                                                                         |
| [Hub65]   | P. J. Huber. 'A Robust Version of the Probability Ratio Test'. In: The Annals of Mathematical Statistics 36.6 (1965), pp. 1753-1758.                                                                                        |
| [JTK14]   | P. Jain, A. Tewari, and P. Kar. 'On Iterative Hard Thresholding Methods for High- Dimensional M-Estimation'. In: Advances in Neural Information Processing Sys- tems 27 (NeurIPS) . 2014.                                   |
| [Kea98]   | M. J. Kearns. 'Efficient noise-tolerant Learning from Statistical Queries'. In: Jour- nal of the ACM 45.6 (1998), pp. 983-1006.                                                                                             |
| [KG25]    | S. Kotekal and C. Gao. 'Optimal Estimation of the Null Distribution in Large-Scale Inference'. In: IEEE Transactions on Information Theory (2025).                                                                          |
| [Kra04]   | I. Krasikov. 'New Bounds on the Hermite Polynomials'. In: arXiv preprint math/0401310 (2004).                                                                                                                               |
| [KWB19]   | D. Kunisky, A. S. Wein, and A. S. Bandeira. 'Notes on Computational Hardness of Hypothesis Testing: Predictions Using the Low-Degree Likelihood Ratio'. In: Mathematical Analysis, its Applications and Computation . 2019. |
| [LRV16]   | K. A. Lai, A. B. Rao, and S. Vempala. 'Agnostic Estimation of Mean and Co- variance'. In: Proc. 57th IEEE Symposium on Foundations of Computer Science (FOCS) . 2016.                                                       |
| [McD09]   | J. H. McDonald. Handbook of Biological Statistics, volume 2 . Sparky House Pub- lishing, Baltimore, MD, 2009.                                                                                                               |
| [MVBWS24] | T. Ma, K. A. Verchand, T. B. Berrett, T. Wang, and R. J. Samworth. 'Estimation beyond Missing (Completely) at Random'. In: arXiv 2410.10704 (2024).                                                                         |
| [NGS24]   | S. Nietert, Z. Goldfeld, and S. Shafiee. 'Robust Distribution Learning with Local and Global Adversarial Corruptions (extended abstract)'. In: Proc. 37th Annual Conference on Learning Theory (COLT) . 2024.               |
| [NT13]    | N. H. Nguyen and T. D. Tran. 'Exact Recoverability From Dense Corrupted Ob- servations via ℓ 1 -Minimization'. In: IEEE transactions on information theory 59.4 (2013), pp. 2017-2035.                                      |
| [PF20]    | S. Pesme and N. Flammarion. 'Online robust regression via sgd on the l1 loss'. In: Advances in Neural Information Processing Systems 33 (2020), pp. 2540-2552.                                                              |
| [PJL20]   | A. Pensia, V. Jog, and P. Loh. 'Robust Regression with Covariate Filtering: Heavy Tails and Adversarial Contamination'. In: CoRR abs/2009.12976 (Sept. 2020).                                                               |
| [PP24]    | A. Pensia and T. Pittas. 'Optimal Robust Estimation under Local and Global Cor- ruptions: Stronger Adversary and Smaller Error'. In: Proc. 38th Annual Conference on Learning Theory (COLT) . 2024.                         |
| [RL87]    | P. J. Rousseeuw and A. M. Leroy. Robust Regression and Outlier Detection . New York, NY, USA: John Wiley &Sons, Inc., 1987.                                                                                                 |
| [SBRJ19]  | A. S. Suggala, K. Bhatia, P. Ravikumar, and P. Jain. 'Adaptive hard thresholding for near-optimal consistent robust regression'. In: Proc. 32nd Annual Conference on Learning Theory (COLT) . 2019.                         |
| [TJSO14]  | E. Tsakonas, J. Jaldén, N. D. Sidiropoulos, and B. Ottersten. 'Convergence of the Huber Regression M-Estimate in the Presence of Dense Outliers'. In: IEEE Signal Processing Letters 21.10 (2014), pp. 1211-1214.           |
| [Tuk60]   | J. W. Tukey. 'A survey of sampling from contaminated distributions'. In: Contribu- tions to probability and statistics 2 (1960), pp. 448-485.                                                                               |
| [Ver18]   | R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science . Cambridge University Press, 2018.                                                                                           |
| [WM10]    | J. Wright and Y. Ma. 'Dense error correction via ℓ 1 -minimization'. In: IEEE Transactions on Information Theory 56.7 (2010), pp. 3540-3560.                                                                                |
| [ZJS19]   | B. Zhu, J. Jiao, and J. Steinhardt. 'Generalized Resilience and Robust Statistics'. In: The Annals of Statistics 50.4 (2019), pp. 2256-2283.                                                                                |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All of the assumptions and limitations are mentioned in the main theorem.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, the limitations of the SQ framework is mentioned.

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

Justification: Yes, all of our assumptions and the distributional assumptions are mentioned clearly.

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

Justification: No experimental results.

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
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: No experiments.

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

Justification: No experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: No experiments.

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

Justification: No experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Yes, the research follows the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is theoretical where computational lower bounds of a statistical problem is studied.

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

Justification: No such risks exist for this paper.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No such assets are used.

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

Justification: No such assets are released.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No such experiments are involved.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No such experiments were conducted.

Guidelines:

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

## Supplementary Material

The Appendix is organized as follows: Appendix A contains additional preliminaries and background on SQ algorithms. Appendix B contains proofs deferred from Section 3. Appendix C gives a computationally-efficient reduction from testing to estimation. Appendix D gives an inefficient SQ algorithm that uses VSTAT oracle with accuracy complexity linear in 1 α , whereas Appendix E gives an efficient SQ algorithm that uses a VSTAT oracle with accuracy complexity quadratic in 1 α .

## A Additional Preliminaries

We say a random variable X or a distribution P is σ -subgaussian if P ( | X | ≥ t ) ≤ 2 exp( -ct 2 /σ 2 ) for all t &gt; 0 ; here c is an absolute constant.

Fact A.1. There exists a finite constant a 0 &gt; 0 such that if X is σ -subgaussian then ∣ ∣ ∣ E [ e a X 2 σ 2 ] -1 ∣ ∣ ∣ ≲ | a | for | a | ≤ a 0 .

Proof. We use expansion of e x and the fact that E [ | X | p ] ≤ ( Cσ √ p ) p for a σ -subgaussian distribution [Ver18, Proposition 2.5.2] to get

<!-- formula-not-decoded -->

which is of order O ( a ) for small enough a because it then converges as a geometric sequence.

For completeness, we provide the proof of Fact 2.1.

Fact 2.1. For every function f : R d → [ -1 , 1] , ∥ f &gt;ℓ ∥ L 2 ( N (0 , I d )) → 0 as ℓ →∞ . Furthermore, for all f : R d × R → [ -1 , 1] and univariate measures R , ∥ f &gt;ℓ ∥ L 2 ( N (0 , I d ) × R ) → 0 as ℓ →∞ .

Proof. The first statement is a simple consequence of the fact that Hermite polynomials are a complete orthonormal system of L 2 ( R d , N (0 , I d )) .

For the second statement, we shall use dominated convergence theorem. Define the residue f &gt;ℓ y ( x ) := f ( x, y ) -f y ( x ) &gt;ℓ and J ℓ ( y ) := ∥ f &gt;ℓ y ∥ 2 L 2 ( N (0 , I d )) . Observe that E y ∼ R [ J ℓ ( y )] = ∥ f &gt;ℓ ∥ L 2 ( N (0 , I d ) × R ) . The first statement implies that for each y ∈ R , J ℓ ( · ) → 0 as ℓ → ∞ . Furthermore, J ℓ is uniformly bounded by 4 as follows:

<!-- formula-not-decoded -->

where we use Parseval's identity to say ∥ f ≤ ℓ y ∥ 2 L 2 ( N (0 , I d )) ≤ ∥ f y ∥ 2 L 2 ( N (0 , I d )) and that | f y | ≤ 1. Since J ℓ → 0 pointwise as ℓ →∞ and 0 ≤ J ℓ ≤ 4 uniformly, by the dominated convergence theorem, E y [ J ℓ ( y )] → 0 as ℓ →∞ .

## A.1 Statistical Query Algorithms

Instead of getting sample access, SQ algorithms interact with the underlying distribution through an oracle. Observe that there are many ways of implementing a VSTAT ( m ) oracle, especially when the SQ algorithm A makes multiple requires-all we require is that each response is a valid VSTAT ( m ) response to each query.

Recall the notion of success from Definition 2.5. The notion of success is intimately tied to the SDA as shown by the following result:

Proposition A.2 (SQ lower bounds using SDA; Proposition 2.6 (C.I)) . For any query f : Z → [0 , 1] , the following holds: P v ∼S d -1 ( E f,v.m ) ≤ 1 SDA(7 m ) .

We use the arguments implicit in [FGRVX17; BBHLS21; DKRS23].

Proof. Here, we assume that Q v has a valid density with respect to P . For a v ∈ S d -1 and z ∈ Z , we use q v ( z ) to denote the Radon-Nikodym derivative of Q v with respect to P . Observe that

<!-- formula-not-decoded -->

Fix a query f and assume that a 1 := E P [ f ] ≤ 1 2 , otherwise apply the following arguments to 1 -f . We shall show that P ( E f,v,m ) ≤ 1 / SDA(7 m ) by contradiction. Suppose P ( E f,v,m ) &gt; 1 / SDA(7 m ) .

Lemma 3.5 in [FGRVX17] implies that for any m ≥ 1 , 0 ≤ a 1 , a 2 ≤ 1 ,

<!-- formula-not-decoded -->

Applying this with a 1 = E P [ f ] and a 2 = E Q v [ f ] , then if f succeeds on v , then

Taking expectation over v and squaring gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since P ( E ) = P ( E f,v,m ) 2 ≥ 1 SDA(7 m ) 2 , the definition of SDA implies the RHS is &lt; 1 7 m , a contradiction. Hence, P ( E f,v,m ) ≤ 1 SDA(7 m ) .

Proposition A.3 (Query Complexity Lower Bound; Proposition 2.6 (C.II)) . Fix a m ∈ N and q ∈ N . Suppose that for all bounded queries f : Z → [0 , 1] , the probability of success is small as follows:

<!-- formula-not-decoded -->

Then any (potentially randomized and adaptive to the responses of the previous queries) SQ algorithm A for solving Testing Problem 2.2 (with failure probability less than 0 . 25 ) must use either Ω( q ) = Ω(SDA(7 m )) queries or at least one query as powerful as VSTAT ( m +1) .

Again, we use ideas implicit in [FGRVX17; BBHLS21; DKRS23].

Proof. We will fix the VSTAT oracle to be a deterministic oracle V ∗ defined below (independent of the algorithm A ). Since the oracle is fixed, it suffices to show lower bounds against deterministic algorithms.

Consider the following oracle V ∗ :

- If Θ = P , then for any query f , it returns E P [ f ] .
- If Θ = Q v , then it answers differently based on the 'niceness' of f :
- -('a good query for V ∗ on v ') If E P [ f ] is a valid VSTAT ( m ) response, then answer E P [ f ] .
- -(otherwise) Return E Q v [ f ] .

Observe that the oracle V above is a valid VSTAT ( m ) oracle for all both null and all alternate.

Now, let A ′ be any deterministic SQ algorithm (deterministic as a function of the answers of the oracle) that solves the testing problem with queries f 1 , . . . , f q ′ for q ′ = aq for some a &lt; 1 to be decided soon.

̸

Consider the case when Θ = P . Recall that the adversary returns E P [ f i ] for i ∈ [ q ′ ] , which is a valid response. Then, the accuracy guarantee of A implies that A must output 'null' on these instances (because it is deterministic); otherwise the failure probability P ( ̂ Γ = Γ) ≥ 0 . 5 .

Now, consider the alternate case where Θ = Q v for v ∼ S d -1 . Observe that if E ∁ f,v,m holds for a query f , then it is a 'a good query for V ∗ on v '. By assumption, the probability (over v ) that any fixed query f is not good is at most P Θ ( E f,v,m ) ≤ 1 /q . Thus, by a union bound, the probability (over v ) that all the queries { f i } q ′ i =1 are good for V ∗ is at least 1 -a . When all the queries are good, the algorithm's input is the same as in the null case, and hence the algorithm must answer 'null'. Therefore, the overall failure probability is at least 0 . 5(1 -a ) ≥ 0 . 25 , which is a contradiction.

## A.2 Pairwise Correlation

We will use the following closed-form expression for the pairwise correlations between Gaussians: Lemma A.4. Let unit vectors u, v ∈ R d and scalars a ∈ R and γ ∈ (0 , 1) . Let cos θ = u ⊤ v . Then

<!-- formula-not-decoded -->

Proof. For any two Gaussians A = N ( µ 1 , Σ 1 ) and B = N ( µ 2 , Σ 2 ) , the average correlation with respect to the standard Gaussian can be calculated as follows:

<!-- formula-not-decoded -->

- A = (Σ -1 1 + Σ -1 2 -I ) -1 · s 1 = det(Σ 1 ) · s 2 = det(Σ 1 ) · s 1 , 2 = det( A -1 ) · h = µ ⊤ 1 Σ -1 1 µ 1 + µ ⊤ 2 Σ -1 2 µ 2 · y = Σ -1 1 µ 1 +Σ -1 2 µ 2 · h ′ = y ⊤ Ay .

We will now instantiate the formula above in our context below.

- (Calculating s 1 and s 2 ) s 1 = s 2 = 1 -γ . Also define b := 1 -γ .
- (Calculating h ) Moreover, Σ -1 1 = ( I -vv ⊤ ) + b -1 vv ⊤ and Σ -1 2 = ( I -uu ⊤ ) + b -1 uu ⊤ . Therefore, h = µ ⊤ 1 Σ -1 1 µ 1 + µ ⊤ 1 Σ -1 1 µ 1 = 2 a 2 b -1 .
- (Calculating y ) The same calculations as above give y = Σ -1 1 µ 1 +Σ -1 1 µ 1 = ab -1 ( v + u ) .
- (Calculating s 1 , 2 ) We begin by calculating A -1 :

<!-- formula-not-decoded -->

Therefore, the determinant of A -1 is 1 -γ 2 cos 2 θ (1 -γ ) 2 for α = γ 1 -γ . This can be seen as follows by considering 2 × 2 matrices:

<!-- formula-not-decoded -->

which equals the expression above.

- (Calculating A and h ′ ) Letting U = [ u ; v ] ∈ R d × 2 and C = α I 2 , then

<!-- formula-not-decoded -->

First, observe that ( u + v ) ⊤ J uu ⊤ ( u + v ) = (1 + cos θ ) 2 for J ∈ { uu ⊤ , vv ⊤ , vu ⊤ , uv ⊤ } . Therefore, ( u + v ) ⊤ M ( u + v ) equals for M := I -A

<!-- formula-not-decoded -->

Let M be I -A . Then

<!-- formula-not-decoded -->

Overall, we get that s 1 s 2 s 1 , 2 = 1 -γ 2 cos 2 θ and h ′ -h = 2 a 2 b -1 (1 -γ ) cos θ 1+ γ cos θ = 2 a 2 cos θ 1+ γ cos θ .

## A.3 Discrete Gaussian

Fact A.5 (Translation of Discrete Gaussian) . For any θ ∈ R , s &gt; 0 , µ ∈ R , σ ∈ R + , the random variables X ∼ DG [ µ, σ, θ, s ] and X ′ := σY + µ for Y ∼ DG [ 0 , 1 , θ ′ , s ′ ] with θ ′ = ( θ -µ ) /σ and s ′ = s/σ have the same law.

Proof. First the support of both X and X ′ are equal to θ + s Z (indeed the support of Y is θ ′ + s ′ Z , which when multiplied by σ , yields ( θ -µ ) + s Z , and further shifting by µ yields θ + s Z .

Starting with X , for any i ∈ Z , P ( X = θ + si ) ∝ sϕ µ,σ ( θ + si ) ∝ s σ exp( -0 . 5( θ + si -µ ) 2 /σ 2 ) ∝ exp( -0 . 5( θ + si -µ ) 2 /σ 2 ) , where we use that s and σ can be absorbed into the normalizing constant. ′

Turning to the random variable X ,

<!-- formula-not-decoded -->

using the definitions of θ ′ and s ′ . Since the support is equal and the two distributions are equal up to constant, they must be equal.

Fact A.6 ([DKRS23, Fact C.3] and [DK22, Lemma 3.12]) . We have the following:

- For any polynomial p of degree at most k and θ ∈ R and s &gt; 0 , we have that ∣ ∣ ∣ ∣ E G ∼N (0 , 1) [ p ( G )] -E Y ∼ DG [ 0 , 1 ,θ,s ] [ p ( Y )] ∣ ∣ ∣ ∣ ≲ √ E G ∼N (0 , 1) [ p 2 ( G )] k !2 O ( k ) exp( -Ω(1 /s 2 )) .
- (Monomials and for the unnormalized measure) For any k ∈ N and s ≥ 0 : ∣ ∣ ∣ ∣ E G ∼N (0 , 1) [ G k ] -E Y ∼ DG ′ [ 0 , 1 ,θ,s ] [ Y k ] ∣ ∣ ∣ ∣ ≲ k !( O ( s )) k exp( -Ω(1 /s 2 )) . In particular, the total mass of DG ′ [ 0 , 1 , θ, s ] is 1 ± exp( -Ω(1 /s 2 )) .

## B Proofs Deferred from Section 3

## B.1 Proof of Proposition 3.3

Proposition 3.3. Testing Problem 3.1 is equivalent to Testing Problem 1.3 when E = DG [ 0 , σ, 0 , s ] .

Proof. First, by definition the distributions P under Testing Problem 1.3 and Testing Problem 3.1 are the same. For Q , we shall do the calculations explicitly.

As a starting point, it is easy to see that the conditional distribution of X given y under Testing Problem 1.3 is an instance of NGCA, as in Testing Problem 3.1. To see this, define x ′ = v ⊤ x to be the projection of x along v , and define x ⊥ = x -( v ⊤ x ) v to be its orthogonal projection. Observe that x ′ and x ⊥ are distributed as standard (multivariate) Gaussian and are independent of each other (because X ∼ N (0 , I d ) ). Hence, the conditional distribution of y given X ≡ ( x ′ , x ⊥ ) can be written as y = ρx ′ + Z , implying that y is independent of x ⊥ . Therefore, the conditional distribution of X given y = y 0 follows like a standard (multivariate) Gaussian in subspace orthogonal to v . Along the direction v , the distribution of X is equivalent to the conditional distribution of x ′ given y , which we denote by ˜ J y . Our goal is to show that ˜ J y is equal to DG [ µ y , σ, θ y , s ′ ] as in Testing Problem 3.1.

Observe that marginal distribution of Y is a Gaussian mixture with countable components, given by

<!-- formula-not-decoded -->

with w ( i ) = cs (2 π ) -1 / 2 exp( -s 2 i 2 / (2 σ 2 )) , where c denotes the normalization constant. Since Z is discrete over the domain s Z , the conditional distribution of X given Y = y 0 is discrete with support ( y 0 -s Z ) /ρ = θ y 0 -s ′ Z , which is the same support as DG [ µ y 0 , σ, θ y 0 , s ′ ] . For any x 0 in this discrete set, the conditional probability of X = x 0 given y = y 0 is given by the following (where we hide multiplicative terms that do not depend on x 0 under the normalization constant):

<!-- formula-not-decoded -->

which is the mass assigned by DG [ µ y 0 , σ, θ y 0 , s ′ ] ; Here we repeatedly use that ρ 2 + σ 2 = 1 .

## B.2 Concentration Properties of Distributions

In this section, we state the concentration properties of various distributions that appear in our analysis.

Lemma B.1. Let the parameters be as in Definition 3.2. Then we have the following:

1. The distributions A y , ˜ A y , and B y are O ( | y | + σ ) -subgaussian and if X follows either one of these distributions, then P ( | X -µ y | &gt; t ) ≲ e -t 2 / 2 σ 2 .
2. The distribution DG [ 0 , σ, 0 , s ] is an O ( σ ) -subgaussian distribution, and R ρ,E is an O ( σ + ρ ) -subgaussian distribution.
3. The distributions P A y v , P ˜ A y v , and P B y v are a O ( | y | + σ +1) -subgaussian distributions. 5

Proof. We do it case-by-case.

- 1.

<!-- formula-not-decoded -->

This tail also implies O ( | y | + σ ) -subgaussianity as follows: we claim that P ( | X | &gt; t ) ≲ e -ct 2 max( | µy | ,σ ) 2 . Observe that it suffices to consider t ≳ max( | µ y | , σ ) ; otherwise, the bound is trivially true. For t ≫| µ y | , P ( | X | &gt; t ) ≤ P ( | X -µ y | ≥ t/ 2) and we can then use (6). The same arguments hold for B y . The claim for the tails of | X | under ˜ A y follows from that of A y because ˜ A y is obtained from conditioning on an event of probability at least 0 . 5 .

2. The claim for DG [ 0 , σ, 0 , s ] follows from (6). For R ρ,E , we use the fact that if x 1 and x 2 are two independent σ 1 and σ 2 -subgaussian random variables, then their sum is O ( √ σ 1 + σ 2 ) -subgaussian [Ver18, Proposition 2.6.1].
3. After rotating appropriately, P A y v and P B y are vectors of independent coordinates and thus follow a multivariate subgaussian distribution with variance proxy bounded by the subgaussian parameter of any individual coordinate [Ver18, Lemma 3.4.2]. The subgaussian proxy for the v direction is established in the first item, while for the other coordinates it is O (1) .

## B.3 Proof of Proposition 3.6

Proposition 3.6 (SQ Hardness of Continuous Noise) . Consider Testing Problem 3.4. Then for any m ∈ N and q ∈ N satisfying ρ 2 √ log(1 /q ) √ d ≲ 1 m , we have that SDA( m ) ≳ q .

5 A multivariate random vector X is termed σ -subgaussian if, for all unit vectors v , the real-valued random variable v ⊤ X is σ -subgaussian.

Proof. To calculate the average SQ correlation between T v and T v ′ , we can first calculate the average correlation between the conditional distributions and then take the average marginal over y to obtain the following expression:

<!-- formula-not-decoded -->

Here, we crucially used that the marginal distribution of y under P , T v and T v ′ is identical.

Observe that the distribution P B y v is equal to N ( µ y v, ( I d -vv ⊤ ) + σ 2 vv ⊤ ) . Using Lemma A.4 with a = µ y = ρy , γ = 1 -σ 2 = ρ 2 , and cos θ = v ⊤ v ′ to calculate χ N (0 , I d ) ( P B y v , P B y v ′ ) , we obtain

<!-- formula-not-decoded -->

for appropriately defined f ( θ ) := 1 √ 1 -ρ 4 cos 2 θ -1 and g ( θ ) := ρ 2 cos θ 1+ ρ 2 cos θ . Therefore, the average correlation over y ∈ R is equal to

<!-- formula-not-decoded -->

Now, observe that | g ( θ ) | ≤ ρ 2 ≤ ρ 2 0 by assumption for a sufficiently small ρ 0 . Therefore, if we define r ( θ ) := E y ∼ R [ exp ( g ( θ ) y 2 )] -1 , then Fact A.1 implies that

<!-- formula-not-decoded -->

Combining this with (8), we obtain

<!-- formula-not-decoded -->

where we use that ρ 2 | cos θ | ≤ 0 . 1 . In particular,

<!-- formula-not-decoded -->

We are now ready to show that SDA( m ) ≥ q , for which we need to show the following:

<!-- formula-not-decoded -->

Using (9), it suffices to show that

<!-- formula-not-decoded -->

If v and v ′ are two independent random unit vectors, then W := v ⊤ v ′ is a centered Θ(1 / √ d ) -subgaussian random variable [Ver18, Theorem 3.4.6]. For subgaussian random variables, we use the simple inequality (a simple consequence of Hölder's inequality) E [ | W | ∣ ∣ E ] ≤ ∥ W ∥ ψ 2 √ log(1 / P ( E )) ≲ 1 √ d √ log( q ) . We obtain that the left hand side in (11) is less than ρ 2 ( O (1) √ d · √ log q ) and hence (11) holds if

<!-- formula-not-decoded -->

which is the desired conclusion.

## B.4 Proof of Proposition 3.9

Observe that Proposition 3.9 follows from the result below because of Lemma B.1. Indeed, Lemma B.1 implies that (i) P y ( | y | ≥ L ) ≲ e -Ω( L 2 ) as σ ≲ 1 and ρ ≲ 1 and (ii) for any y with | y | ≤ d/ 2 , P A y ( | z | &gt; d ) ≤ P A y ( | z -µ y | &gt; d/ 2) ≲ e -Ω( d 2 ) .

Proposition B.2. Let f : U → [0 , 1] . For L ≥ 1 , define the set C : { y : | y | ≤ L ˜ σ } and the function ˜ f := f y ∈C . Then for any ℓ ∈ N , we have that for y ∼ R ρ,E :

<!-- formula-not-decoded -->

where T k,y := E x ∼N (0 , I d ) [ ˜ f y ( x ) H k ( x )] , A k,y := E x ∼ ˜ A y [ h k ( x )] and B k,y := E x ∼ B y [ ˜ f ( x )] for A y , B y defined in Testing Problems 3.1 and 3.4.

Proof. We start by replacing E Q v [ f ] with E ˜ Q v [ f ] at the cost of additive TV( Q v , ˜ Q v ) . This total variation distance is O ( P Q v ( | v ⊤ x | &gt; d )) , which can be upper bounded by P ( | y | ≥ L ) + max y : | y |≤ L P ( | v ⊤ x | ≥ d ) . Hence, in the rest of this proof, we shall use ˜ Q everywhere.

Next we decompose f = ˜ f + f ′ for f ′ := f ✶ y ̸∈C . Then we further decompose ˜ f as ˜ f ≤ ℓ + ˜ f &gt;ℓ y . By triangle inequality, it suffices to show that the expectations of ˜ f ≤ ℓ , ˜ f &gt;ℓ ) , and f ′ are close. Observe that the term for ˜ f &gt;ℓ is already present in the final conclusion. Next, for f ′ , the boundedness of f and the same marginals of Q v and T v imply that

<!-- formula-not-decoded -->

In the remainder, we focus on the terms corresponding to ˜ f ≤ ℓ . By the law of total expectation (whose validity for ˜ f ≤ ℓ is justified below), we have that

<!-- formula-not-decoded -->

To compute the inner expectation, which is an instance of the unsupervised NGCA, we will use [DKRS23, Lemma 3.3]:

Lemma B.3 (Fourier Decomposition Lemma of [DKRS23]) . Let A ′ be any distribution supported on R and v a unit vector. Then for any ℓ ∈ N and g : R d → [0 , 1] ,

<!-- formula-not-decoded -->

where A k = E x ∼ A ′ [ h k ( x )] and T k = E x ∼N (0 , I d ) [ g ( x ) H k ( x )] .

Consider a fixed y 0 ∈ R and apply the above result to A ′ := ˜ A y 0 and g ( x ) := ˜ f y ( x ) := f ( x, y 0 ) ✶ y 0 ∈C . Define ˜ A k,y := E x ∼ ˜ A y [ h k ( x )] and B k,y := E x ∼ B y [ h k ( x )] and T k,y := E x ∼N (0 , I d ) [ ˜ f ( x, y ) H k ( x )] . We obtain that

<!-- formula-not-decoded -->

Observe that the term k = 0 corresponds to E N (0 , I d ) [ f ( x, y 0 )] for each y 0 , implying that the expectation of the k = 0 term (over y ) is exactly E P [ ˜ f ( x, y )] . Thus, we get the following decomposition:

<!-- formula-not-decoded -->

Similarly, the decomposition for the continuous Gaussian noise is as follows:

<!-- formula-not-decoded -->

The claim follows by combining Equations (15) and (16).

Justifying (14). It suffices to show that E ˜ Q v [ | ˜ f ≤ ℓ | ] &lt; ∞ , which we will establish below. By Fubini's theorem, we have that

<!-- formula-not-decoded -->

This can be further upper bounded by finite sum of the terms (at most d O ( ℓ ) ) involving

<!-- formula-not-decoded -->

for some polynomials p ( · ) . Since | f | is upper bounded by 1 , the term above is further upper bounded by E y E x ∼ P ˜ Ay v [ | p ( x ) | 2 ] using Jensen's inequality. Using Lemma B.1, E x ∼ P ˜ Ay v [ | p ( x ) | 2 ] is upper bounded by poly( | µ y | , d, ∥ p ∥ ℓ 2 ) and since µ y is linear in y , E [poly( µ y )] is also finite because R is O (1) -subgaussian.

A similar reasoning justifies (14) for T v .

## B.5 Proof of Lemma 3.10

Lemma 3.10 (Closeness of Hermite Coefficients) . For any y ∈ R and k ∈ N , we have

- (Tighter for small k ) | ˜ A k,y -B k,y | ≲ max ( 1 , | µ y | k ) k O ( k ) · ( e -Ω ( ρ 2 s 2 ) + e -Ω( d ) ) . 2
- (Tighter for larger k ) | ˜ A k,y -B k,y | ≲ e O ( µ y ) .

Proof. We first consider the case for large k .

Large k . For large k , we shall use the fact that | h k ( x ) | ≤ exp( x 2 / 4) for all x ∈ R [Kra04]. Lemma B.1 implies that for both B y and ˜ A y ,

<!-- formula-not-decoded -->

where we use that σ ≤ 1 . Therefore, under the both X ∼ ˜ A y and X ∼ B y , we have that

<!-- formula-not-decoded -->

Indeed for t ≤ 10 µ y , the upper bound is bigger than 1 and hence holds; for t ≥ 10 µ y , P ( | X | &gt; t ) ≤ P ( | X -µ y | ≥ 0 . 9 t ) ≲ exp( -0 . 4 t 2 ) .

Therefore, we can upper bound E [ | h k ( X ) | ] for both distributions as follows:

<!-- formula-not-decoded -->

Smaller k . We first define C k,y := E x ∼ A y [ h k ( x )] . Since ˜ A y is A y conditioned on E := { z : | z | ≤ d } and satisfies P ( E ) ≥ 1 -τ for τ ≲ e -Ω( d ) (see Lemma B.1), we have that for any function g :

<!-- formula-not-decoded -->

The above inequality follows by noting that the left hand side above is exactly equal to E Ay [ g ] -E By [ g ] 1 -τ + τ E By [ g ] 1 -τ + E Ay [ g I E ] 1 -τ and then applying Cauchy-Schwarz inequality. In our context, the above display equation yields:

<!-- formula-not-decoded -->

We will now upper bound this difference. We first claim that for ˜ θ y = ( θ y -µ y ) /σ and ˜ s = s ′ /σ , we have that

<!-- formula-not-decoded -->

To see this, recall that B k,y = E x ∼ B y [ h k ( x )] = E x ∼N ( µ y ,σ 2 ) h k ( x ) , which implies that it is equal to E x ∼N (0 , 1) h k ( σx + µ y ) . For A k,y , the claim follows analogously from Fact A.5.

Lemma B.4. Let k ∈ N , q ∈ R , a ∈ R , b ∈ R and s ′′ ≪ 1 . Let G ∼ N (0 , 1) and Y ∼ DG [ 0 , 1 , q, s ′ ] .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying this result on (19) with b = µ y , a = σ ≤ 1 and s ′′ = ˜ s = s ′ /σ = s/ρσ and plugging it in (18) in combination with τ ≲ e -Ω( d ) , we get Lemma 3.10.

We now provide the proof of Lemma B.4

Proof. Defining the polynomial p k ( x ) := h k ( b + ax ) , we can apply Fact 2.10 to p k ( · ) to conclude that the deviation in the first item is at most

<!-- formula-not-decoded -->

Hence, to establish both the first and the second items, it remains to show the upper bound √ E G ∼N (0 , 1) [ h 2 k ( b + aG )] ≲ k O ( k ) max(1 , | b | k ) max(1 , | a | k ) . To that effect, we use the explicit form of the Hermite polynomials:

<!-- formula-not-decoded -->

which gives the following expression:

<!-- formula-not-decoded -->

There are Θ( k 2 ) terms in the expression above and by linearity of the expectation, it suffices to control the maximum term above:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves the desired result.

We now focus on the third item. Here, we again apply Fact 2.10 but this time to the polynomial p 2 k , which would then imply that

<!-- formula-not-decoded -->

To upper bound E [ | h k ( b + aG ) 4 ] , we can use a similar series of arguments as in (21) to get the desired result, wherein we replace the use of Cauchy-Schwarz inequality with the inequality E [ X 1 X 2 X 3 X 4 ] ≤ ∏ 4 i =1 ( E [ X 4 i ]) 1 / 4 .

We now provide the proof of Lemma B.4

Proof. Defining the polynomial p k ( x ) := h k ( b + ax ) , we can apply Fact 2.10 to p k ( · ) to conclude that the deviation in the first item is at most

<!-- formula-not-decoded -->

Hence, to establish both the first and the second items, it remains to show the upper bound √ E G ∼N (0 , 1) [ h 2 k ( b + aG )] ≲ k O ( k ) max(1 , | b | k ) max(1 , | a | k ) . To that effect, we use the explicit form of the Hermite polynomials:

<!-- formula-not-decoded -->

which gives the following expression:

<!-- formula-not-decoded -->

There are Θ( k 2 ) terms in the expression above. Moreover, By linearity of the expectation, it suffices to control the maximum term above:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves the desired result.

We now focus on the third item. Here, we again apply Fact 2.10 but this time to the polynomial p 2 k , which would then imply that

<!-- formula-not-decoded -->

To upper bound E [ | h k ( b + aG ) 4 ] , we can use a similar series of arguments as in (21) with the inequality E [ X 1 X 2 X 3 X 4 ] ≤ ∏ 4 i =1 ( E [ X 4 i ]) 1 / 4 instead of Cauchy-Schwarz inequality to get the desired result.

## B.6 Proof of Proposition 3.11

Proposition 3.11. Let { T k,y } k ∈ N ,y ∈ R be tensors with ∥ T k,y ∥ 2 ≤ 1 for all k ∈ N , y ∈ R , and let t ∈ N be arbitrary . Then for any δ ∈ (0 , 1) , it holds with probability 1 -δ over a random unit vector v that

<!-- formula-not-decoded -->

The result for k ≤ t follows by the claim that T k,y has norm at most 1 almost surely. Hence, we will focus on the second claim, for which we shall crucially use the concentration results from [DKRS23], which we state in a different formulation below.

Lemma B.5 (Lemma 3.7 and Corollary 3.9 in [DKRS23]) . Let T k,y be a random k -tensor supported with randomness y . For a random unit vector v independent of y , let W k denote the random variable E y ∣ ∣ ⟨ v ⊗ k , T k,y ⟩ ∣ ∣ . Let W ′ be the random variable v ⊤ w for a unit vector w ∈ S d -1 . Then for any even p ∈ N ,

<!-- formula-not-decoded -->

In particular, if ⟨ T k,y , T k,y ⟩ ≤ 1 almost surely, then there exists a constant C &gt; 0 such that the following conclusion holds for any even p ∈ N and k ∈ N :

<!-- formula-not-decoded -->

While (22) is established in [DKRS23, Lemma 3.7] for a fixed tensor T , the desired follows by Jensen's inequality: for any even p and error bound g ( v, y ) , we have that E v ( E y [ g ( v, y )])] p ≤ E v E y [ g ( v, y ) p ] = E y ( E v [ g ( v, y ) p ]) , where one can now use [DKRS23, Lemma 3.7].

We now couple Lemma B.5 with the simple fact that for any random variable X , with probability 1 -δ , | X | ≤ (1 /δ ) 1 /p ∥ X ∥ L p for any p ≥ 1 . Therefore, for any k ≥ t , with probability 1 -δ/k 2 , | W k | is less than c min( ∥ W k ∥ L log( k/δ ) , √ k/δ ∥ W k ∥ L 2 ) . We now calculate these bounds separately for different k .

- (Small k and large p ) Define p k ≍ log( k/δ ) . For any k such that k ≥ t and kp k ≤ C ′ dp k , we have

<!-- formula-not-decoded -->

This follows by considering the following two regimes separately:

- -( c ′ p k k ≤ d for a tiny enough constant c ′ ) In this regime, the bound ( Ckp k d ) k/ 4 is decreasing in k and thus the maximum is achieved at k = t .
- -( C ′ dp k ≥ c ′ p k k ≥ d for a large constant c ′ ) In this regime, the second bound gives the desired result by noting max( d, p k k ) ≤ C ′ d p k and k ≥ d/p k ≥ d/p d .
- (Large k and p = 2 ) Moreover, if k ≥ C ′ d , then ∥ W k ∥ 2 ≤ ( d 2 k ) d/ 2 . Therefore, with probability 1 -δ/k 2 , | W k | ≤ √ ( k 2 /δ )( d/ 2 k ) d/ 2 ≤ √ d/δ ( d/ 2 k ) d/ 4 .

Taking a union bound, we obtain the following bound that holds with probability at least 1 -δ ,

<!-- formula-not-decoded -->

The summation ∑ k ≥ C ′ d ( d 2 k ) d/ 2 can be upper bounded by a constant factor multiple of the first expression in the sum (this can be seen by integrating ∫ x ≥ x 0 x -a dx for a &gt; 2 ), and the first expression is at most e -Ω( d ) because C ′ is large enough.

## B.7 Proof of Theorem 3.7

We are now ready to present the proof of Theorem 3.7.

Proof of Theorem 3.7. Combining Proposition 3.9 with Lemma 3.10 and Proposition 3.11 and the fact that | µ y | ≤ L for L ≥ 1 , we obtain that for any t ∈ N and ℓ ∈ N with probability at least 1 -d -log 2 d ,

<!-- formula-not-decoded -->

For L = log 5 d , t = L 6 and ρ = st 2 , the sum of all but the last term is at most O ( e -L 2 ) ≤ d -log 2 ( d/α ) . For the last term, we show in Appendix B.8 that taking ℓ large enough suffices-this argument uses Fact 2.1 and the truncation of A y as per [DKRS23].

## B.8 Handling ˜ f &gt;ℓ

We now show that for any f : R d × R → [0 , 1] and any δ ∈ (0 , 1) , there exists ℓ ∈ N , depending only on ( f, d, δ, L, σ, ρ, s, α ) such that with 1 -δ , | E Q v [ ˜ f &gt;ℓ ] -E T v [ ˜ f &gt;ℓ ] | is smaller than γ for a γ appropriately small.

First by Fact 2.1, we know that there exists an ℓ ( γ ′ ) so that ∥ ˜ f &gt;ℓ ∥ L 2 ( P ) ≤ γ ′ . Since χ 2 ( P, T v ) is finite (as established in (10)), this implies that ∥ ˜ f &gt;ℓ ∥ L 2 ( T v ) is also sufficiently small. By CauchySchwarz, we get that for every γ and v ∈ S d -1 , there exists an ℓ ′ ( δ, d, γ ) so that | E T v [ ˜ f &gt;ℓ ] | ≤ γ .

Thus, it remains to argue about E Q v [ ˜ f &gt;ℓ ] . By a Markov inequality, it suffices to show that E v [ | E Q v [ ˜ f &gt;ℓ ] | ] ≤ E v E Q v [ | ˜ f &gt;ℓ | ] ≤ γ/δ . Let D be the distribution of ( x, y ) obtained over ( x, y ) as follows: first y ∼ R and then v ∼ S d -1 and x ∼ P ˜ A y v . Let D y be the conditional distribution of x given y under D . Thus E v E Q v [ | ˜ f &gt;ℓ | ] ≤ γ/δ = E D [ | ˜ f &gt;ℓ | ] . We will now show that χ 2 ( D,P ) &lt; ∞ , which would suffice for our result. Observe that

<!-- formula-not-decoded -->

Observe that D y is obtained from P ˜ A y v where ˜ A is supported only on { x : | x | ≤ d } . [DKRS23, Lemma 3.1] implies that χ 2 ( D v , N (0 , I d )) is uniformly upper bounded by O d (1) . Integrating this uniform upper bound by O d (1) , we get the desired conclusion of χ 2 ( D,P ) &lt; ∞ .

## B.9 Formal version of Theorem 1.6

We are now ready to state and prove the formal version of Theorem 1.6.

Theorem B.6 (SQ Hardness of Testing Problem 1.3) . Consider the testing problem in Testing Problem 1.3 with E = DG [ 0 , σ, 0 , s ] for s ≍ α and σ = 1 . Furthermore, assume that

- α ≫ 1 d polylog( d ) (i.e., it is not too tiny)
- ρ 2 ≍ α 2 polylog( d/α ) and ρ ≤ ρ 0 for a sufficiently small absolute constant ρ 0 .

Then we have the following guarantees:

1. P E ( z = 0) ≥ α (i.e., it is a valid instance).
2. Any SQ algorithm that solves the testing problem with probability at least 2 / 3 either uses q ≳ q 0 := d log 2 ( d/α ) many queries or uses a single query which is as powerful as VSTAT ( m ) for m ≳ √ d α 2 polylog( d, 1 /α ) .

Proof. Since σ ≥ 1 / 2 , we get that P ( z = 0) ≥ α (recall that P Z ∼ DG [ 0 ,σ, 0 ,s ] ( z = 0) = Θ( s/σ ) ), which satisfies the first claim of Theorem B.6.

To establish the second claim about the SQ complexity, using Proposition 2.6, it suffices to show that the probability of success of f on distinguishing Q v and P with m simulation complexity is at most 1 /q 0 . Recall that the success event E f,v,m is defined as the following event:

<!-- formula-not-decoded -->

and our goal is to show that for any fixed bounded query f : Z → [0 , 1] , we have P v ∼S d -1 [ E f,v,m ] ≤ 1 q for q ≍ d log 2 ( d/α ) and m ≳ m 0 := σ 2 √ d ρ 2 polylog( d/α ) .

We now define the following events:

- First, E ′ f,v,m is defined as: ∣ ∣ E Q v [ f ( x, y )] -E T v [ f ( x, y )] ∣ ∣ ≥ 1 4 m 2 .
- Next, the event E ′′ f,v,m is defined as: for a large constant C (which can be deduced from the proof of Claim B.7),

<!-- formula-not-decoded -->

Next, we show in Claim B.7 that E f,v,m ⊂ E ′ f,v,m ∪ E ′′ f,v,m . By the union bound and Claim B.7, it suffices to establish that the probabilities of these events individually is at most 1 2 q .

- ( E ′ f,v,m ) Theorem 3.7 implies the desired bound for any m ≤ ( d/α ) log 2 ( d/α ) and q ≤ d log 2 ( d/α ) .
- ( E ′ f,v,m ) This inequality was established in Proposition 3.6 for any m ≲ m 0 with m 0 ≍ σ 2 √ d ρ 2 √ log(1 /q ) . Taking q = d log 2 ( d/α ) and σ = Θ(1) leads to m 0 ≍ √ d ρ 2 polylog( d/α ) .

This completes the proof of Theorem B.6.

We now provide the statement and the proof of Claim B.7.

Claim B.7. We have that E f,v,m ⊂ E ′ f,v,m ∪ E ′′ f,v,m .

Proof of Claim B.7. Indeed, we have that

<!-- formula-not-decoded -->

Observe that on E ′ f,v,m , | E T v [ f ] -E T v [ f ] | ≤ τ for τ = O (1 /m 2 ) . Since the expectations are close, the standard deviations are also close: | √ ( E T v [ f ])(1 -E T v [ f ]) -( E Q v [ f ])(1 -E Q v [ f ]) | = O ( √ τ ) . Therefore, the second term above is at most

<!-- formula-not-decoded -->

Since τ ≳ 1 /m 2 , the overall term is at most

<!-- formula-not-decoded -->

We now claim that this is less than the threshold for E f,v,m in (23). Towards that goal, define a = √ E P [ f ] · E P [1 -f ] and b for the corresponding term with Q v . Consider the case when min( a, b ) / √ Cm ≤ 1 /Cm . Then the left hand side above is 2 Cm , which is less than the quantity in E f,v,m , which is at least 1 /m . Suppose now that min( a, b ) ≥ 1 / ( Cm ) . Then the term above is at most 1 Cm + min( a,b ) √ Cm ≤ 2 min( a,b ) Cm , which is less than the quantity in E f,v,m , which is at least min( a,b ) m .

Thus, we have shown that E f,v,m ⊂ E ′ f,v,m ∪ E ′′ f,v,m .

## C Computationally-Efficient Reduction from Testing to Estimation

Suppose there is an algorithm A with the following guarantees: given n i.i.d. samples ( x 1 , y 1 ) , . . . , ( x n , y n ) in R d × R from Definition 1.1 with inlier probability α and regressor β ∈ R d , computes an estimate ̂ β such that ∥ ̂ β -β ∥ 2 ≲ τ .

Consider the following (randomized) algorithm A ′ that takes 2 n samples S = { ( x 1 , y 1 ) , . . . , ( x n , y n ) } and S ′ = { ( x ′ 1 , y ′ 1 ) , . . . , ( x ′ n , y ′ n ) } and perform the following operation:

- Sample a random rotation matrix U ∈ R d × d .
- Let ̂ β 1 be the output of A on S .
- Define S ′′ := { ( U x ′ 1 , y ′ 1 ) , . . . , ( U x ′ n , y ′ n ) }
- Let w be the output of A on S ′′ .
- Let ̂ β 2 = U ⊤ w .
- Let W = 〈 ̂ β 1 ∥ ̂ β 1 ∥ , ̂ β 2 ∥ ̂ β 2 ∥ 2 〉 . If | W | &gt; 1 / 9 , output 'alternate', otherwise output 'null'.

Theorem C.1. If τ ≤ ρ/ 4 and d ≳ log(1 /δ ) , then A ′ solves the testing problem in Testing Problem 1.3 with probability at least 1 -2 δ .

Proof. We will argue the success probabilities separately.

Alternate Distribution Consider the case when the underlying distribution is alternate and let the latent hidden direction be v . Conditioned on v and U , the samples S and S ′′ satisfy the conditions of Definition 1.1 with the underlying regressor β and U β , where β := ρv ; here we use that Gaussian distribution is rotationally invariant and U x is again distributed as isotropic Gaussian. Thus, the guarantees of A imply that with probability 1 -2 δ , we have that ∥ ̂ β 1 -β ∥ 2 ≤ τ and ∥ ̂ β 2 -β ∥ 2 = ∥ w -U β ∥ 2 ≤ τ . Since τ ≤ ρ/ 4 , we have that ̂ β 1 ∥ 2 ≤ 1 . 5 ρ and the same for ∥ ̂ β 2 ∥ . Since

<!-- formula-not-decoded -->

the closeness guarantee implies that

<!-- formula-not-decoded -->

Hence, with probability 1 -2 δ , we have that | W | ≥ ( ρ 2 / 4) / (3 ρ/ 2) 2 ≥ 1 / 9 , and hence the algorithm would correctly output 'alternate'.

Null Distribution We will argue that w is independent of U . Indeed, for any U , the distribution of the samples in S ′′ is i.i.d. from N (0 , I d ) × R , where R is the marginal distribution of y (recall that y is independent of X under the null). Hence, S ′′ and w are independent of U . Therefore, ̂ β 2 ∥ ̂ β 2 ∥ 2 is distributed uniformly over the unit sphere (independent of β 1 ). Hence, W is distributed as the product of two unit vectors, implying that with probability 1 -δ , | W | ≲ √ log(1 /δ ) d , and hence the algorithm correctly outputs 'null' for d large enough.

## D Inefficient SQ Algorithm with Correct Sample Complexity

In this section, we mention an SQ algorithm that uses q = exp( ˜ O ( d/τα )) queries from VSTAT ( m ) with m = Θ(1 /α ) and outputs an estimate ˜ β such that ∥ ̂ β -β ∥ 2 ≲ τ . Furthermore, this SQ algorithm can be simulated from O ( d log(1 /α ) α ) i.i.d. samples from distribution P β ∗ ,E .

Theorem D.1. Let ∥ β ∗ ∥ 2 ≤ 1 and α ∈ (0 , 1) and let the underlying distribution be P β ∗ ,E for an unknown E and known α . There exists an SQ algorithm that uses q ≤ exp( O ( d log(1 /τα )) many queries to VSTAT ( m ) for m ≲ 1 /α and outputs an estimate ˜ β such that ∥ ̂ β -β ∗ ∥ ≲ τ .

Furthermore, with high probability, the VSTAT ( m ) oracle for this SQ algorithm can be simulated using m ′ = ˜ O ( d α ) many i.i.d. samples from P β ∗ ,E .

Proof. Let C be a τ ′ -cover of { x : ∥ x ∥ 2 ≤ 1 } with respect to the Euclidean norm for τ ′ = 0 . 01 τα . We know such a cover exists with log |C| ≲ d log(1 /τ ′ ) . Furthermore, let β ′ ∈ C be τ ′ -close to β ∗ . For each β ∈ C , define the query f β ( x, y ) = ✶ | x ⊤ β -y |≤ τ ′ . The SQ algorithm is as follows:

- For each β ∈ C , let v β ← VSTAT ( f β , m ) .
- Output ̂ β = argmax β ∈C v β .

Correctness. To show correctness, we shall show that for β that is τ -far from β ∗ , it must be the case that v β &lt; v β ′ , which would imply that any such β can not be the output.

Let us start by analyzing E [ f β ] . Let the distribution E be αδ 0 +(1 -α ) E ′ for an arbitrary distribution E ′ , where δ 0 is the point mass at origin. Then observe that for G ∼ N (0 , 1) :

<!-- formula-not-decoded -->

In particular, for β ′ , E [ f β ′ ] ≥ 0 . 5 α because ∥ β ′ -β ∗ ∥ ≤ τ ′ and P ( | G | ≤ 1) ≥ 0 . 5 . It can then be checked max β ∈C ≥ v β ′ ≥ E [ f β ′ ] -1 m -√ E [ f β ′ ] m , which is bigger than E [ f β ′ ] / 2 if m ≳ 1 E [ f β ′ ] , which is satisfied since m ≥ 1 α and E [ q β ] ≥ 0 . 5 α .

Now consider any ∗ ′

<!-- formula-not-decoded -->

Therefore, for any such β , v β ≤ E [ f β ′ ] + 1 m + √ E [ f β ′ ] m ≤ 0 . 02 α if m ≳ 1 /α . Therefore, any such β can not be ̂ β and hence ∥ ̂ β -β ∗ ∥ 2 ≤ τ .

Simulation with samples We implement the VSTAT ( m ) oracle by taking a set S of i.i.d. samples and returning the empirical mean of q β over S . Observe that all of the queries f β are halfspaces and hence have VC Dimension O ( d ) . For i ∈ { 1 , . . . , log(1 /α 0 ) } , let A i = { β : E [ q β ] ∈ [2 i α, 2 i +1 α ] ∪ [1 -2 i +1 α, 1 -2 i α ] } . Let A 0 = { β : E [ q β ] ∈ [0 , α ] ∪ [1 -α, 1] } .

By uniform concentration [BLM13, Theorem 13.7] and [BLM13, Theorem 12.5], if n ≥ d log(1 / 2 i +1 α ) 2 i +1 α , then with probability 1 -δ/J for J = log(1 /α ) , for all β ∈ A i for i ∈ N ∪ { 0 } , we have

<!-- formula-not-decoded -->

By a union bound over A i 's, this uniform concentration holds for all β ∈ R d . That is, if n ≥ dJ +log( J/δ ) α , then with probability 1 -δ , for all β ∈ R d , we have

<!-- formula-not-decoded -->

if n ≳ ( dJ +log( J/δ )) · ( m + αm 2 ) + m log( J/δ ) . On this event, we get that the empirical approximation is a VSTAT ( m/ 4) oracle. Since we need m = Θ(1 /α ) , the required sample complexity for failure probability δ is at most d log(1 /α )+log(log(1 /α ) /δ α .

## E Efficient SQ Algorithm with Matching Accuracy

We now show that there exists an efficient SQ algorithm that solves Definition 1.1 and the hard instance in Theorem 1.6 with polynomially number of VSTAT ( d/α 2 ) queries. Let β ∗ be the unknown regressor with ∥ β ∗ ∥ 2 ≤ 1 . In this section, we use u as a shorthand for ( x, y ) .

Theorem E.1. Let α ∈ (0 , 1) and β ∗ ∈ B , where B := { β : ∥ β ∥ 2 ≤ 1 } . For any ϵ ∈ (0 , 1) , there is an SQ algorithm that takes these α, ϵ as input, makes poly( d ) number of queries to VSTAT ( d ϵα 2 ) on P β ∗ ,E , and (iii) computes an estimate ̂ β ∈ R d such that ∥ ̂ β -β ∗ ∥ 2 ≲ ϵα .

Observe that we do not need ϵ to be very small to solve the hard instance of Testing Problem 1.3, i.e., we can set ϵ = ρ/α = ˜ Θ(1) and still solve Testing Problem 1.3 with polynomial number of queries to VSTAT ( ˜ Θ( d/α 2 )) .

Proof. Define the function g ( x ) : X → { 0 , 1 } to be function such that g ( x ) = 0 if and only if ∥ x ∥ 2 ≥ L √ d for L = polylog( d/αϵ ) . Consider the loss function ℓ ( β, u ) := g ( x ) · ℓ Huber ( y -x ⊤ β ) ; here ℓ Huber ( · ) is the Huber loss with the gradient h ( z ) = z ✶ z ∈ [ -1 , 1] + sgn ( z ) ✶ | z | &gt; 1 . Consider the averaged loss L ( β ) := E u ∼ P β ∗ ,E [ ℓ ( β, u )] . We claim the following:

1. L is κ -strongly convex on B for κ = Θ( α ) .
2. L is L 1 smooth (Lipschitz continuous gradient) on B for L 1 = O (1) .
3. For every z ∈ Z , the function ℓ ( · , z ) is convex, and it is L 0 -Lipschitz for L 0 ≲ L √ d .
4. β ∗ is the unique minimizer of L .

Therefore, we can apply [FGV17, Corollary 4.12] with parameters L 0 , L 1 , and κ to find an αϵ -close estimate ̂ β such that ∥ ̂ β -argmin L ( β ) ∥ 2 ≲ αϵ with O ( dL 1 log( L 1 diam( B ) /αϵ ) κ ) = O ( d log(1 /αϵ ) α ) many queries to VSTAT ( O ( L 2 0 αϵκ )) = VSTAT ( d · polylog( d/α ) α 2 ϵ ) . We get the desired conclusion by noting that β ∗ uniquely minimizes L ( β ) .

We now give the details omitted earlier:

1. For any unit vector v , v ⊤ ∇ 2 L v is equal to E u [ g ( x ) ∇ 2 ℓ Huber ( y -x ⊤ β )( x ⊤ u ) 2 ] . The convexity follows by non-negativity of the Huber loss.

<!-- formula-not-decoded -->

The last inequality follows because g ( x ) I | x ⊤ β ∗ -x ⊤ β |≤ 1 ( x ⊤ v ) 2 ≳ ✶ ∥ x ∥ 2 ≤ L √ d ✶ | x ⊤ w |≤ 1 ✶ | x ⊤ v |≥ 0 . 5 for some unit vector w . Using triangle inequality, we obtain that its probability is lower bounded by P [ ✶ | x ⊤ w |≤ 1 ✶ | x ⊤ v |≥ 0 . 5 ] -P ((1 -g ( x ))) ≳ 1 -d -100 ≳ 1 .

2. The smoothness follows from the same arguments as above by upper bounding g ( x ) and ∇ 2 ℓ Huber by 1 .
3. Observe that the gradient satisfies ∇ ℓ ( β, z ) = g ( x ) h ( y -x ⊤ β ) x and therefore ∥∇ ℓ ( β, z ) x ∥ 2 ≤ L √ d , where we use that ∥ xg ( x ) ∥ ≤ √ Ld and the gradient of Huber loss is bounded by 1 .
4. By strong convexity on B , it suffices to show that β ∗ has zero gradient.

<!-- formula-not-decoded -->

where we use that xg ( x ) is a symmetric random variable and independent of z .
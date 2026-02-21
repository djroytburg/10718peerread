## Guarantees for Alternating Least Squares in Overparameterized Tensor Decomposition

## Dionysis Arvanitakis

Department of Computer Science Northwestern University

## Vaidehi Srinivas

Department of Computer Science Northwestern University Evanston, IL 60208

Evanston, IL 60208 dionarva@u.northwestern.edu

vaidehi@u.northwestern.edu

## Aravindan Vijayaraghavan

Department of Computer Science

Northwestern University Evanston, IL 60208

aravindv@northwestern.edu

## Abstract

Tensor decomposition is a canonical non-convex optimization problem that is computationally challenging, and yet important due to applications in factor analysis and parameter estimation of latent variable models. In practice, scalable iterative methods, particularly Alternating Least Squares (ALS), remain the workhorse for tensor decomposition despite the lack of global convergence guarantees. A popular approach to tackle challenging non-convex optimization problems is overparameterization- on input an n × n × n tensor of rank r , the algorithm can output a decomposition of potentially rank k (potentially larger than r ). On the theoretical side, overparameterization for iterative methods is challenging to reason about and requires new techniques. The work of Wang et al., (NeurIPS 2020) makes progress by showing that a variant of gradient descent globally converges when overparameterized to k = O ( r 7 . 5 log n ) . Our main result shows that overparameterization provably enables global convergence of ALS: on input a third order n × n × n tensor with a decomposition of rank r ≪ n , ALS overparameterized with rank k = O ( r 2 ) achieves global convergence with high probability under random initialization. Moreover our analysis also gives guarantees for the more general low-rank approximation problem. The analysis introduces new techniques for understanding iterative methods in the overparameterized regime based on new matrix anticoncentration arguments.

## 1 Introduction

Iterative heuristics like alternating least squares (ALS), alternate minimization, and gradient descent are the workhorse for many computational tasks in machine learning and high-dimensional data analysis. Their simplicity, scalability and empirical success have led to their widespread use, even for highly non-convex problems. Yet rigorous guarantees have been hard to establish due to the non-convex nature of these problems.

Tensor decomposition is a prime example of a well-studied non-convex problem where there is a disconnect between the practical performance of iterative heuristics and known theoretical guarantees. We are given a third-order tensor T ∈ R n × n × n , and the goal is to decompose the tensor as a sum of

a few rank1 tensors when possible, i.e.,

<!-- formula-not-decoded -->

where the { x i ⊗ y i ⊗ z i : i ∈ [ k ] } are the rank1 terms of the decomposition, and the vectors { x i , y i , z i : i ∈ [ k ] } ∈ R n are called the factors of the decomposition. This is sometimes called the CP decomposition of the tensor, and is an important tool in factor analysis and parameter estimation of many latent variable models in machine learning [see e.g., KB09, Moi18, JGKA19]. Finding the smallest r for which a rankr decomposition of T exists is NP-hard in the worst case [Hås90, HL13] . Algorithmic guarantees are known under additional genericity or smoothed analysis assumptions for more sophisticated but less scalable algorithms like simultaneous diagonalization and other spectral methods [Har72, LRA93], algebraic methods [DLCC07] and the sum-of-squares hierarchy [BKS15] (see [Vij20] for more detailed comparisons).

The most popular algorithms in practice are iterative methods for the optimization problem given by the least squares objective

<!-- formula-not-decoded -->

In particular, the Alternating Least Squares (ALS) algorithm is an iterative algorithm that alternately updates one set of variables, say x 1 , . . . , x k ∈ R n , while keeping the rest of them ( y 1 , . . . , y k and z 1 , . . . , z k ) fixed. Note that each update step is a least squares problem (e.g., in the variables x 1 , . . . , x k ) when the remaining variables are fixed (see Section 2 for a more detailed description of the algorithm). The optimization landscape of (1) is highly non-convex, and such iterative algorithms can potentially converge to bad local optima. This has even inspired new variants of ALS like the Orthogonalized ALS algorithm of [SV17], that come with rigorous guarantees under strong assumptions like orthogonality or incoherence of the factors. Yet, the ALS algorithm remains the most popular method for tensor decompositions in practice, despite our poor understanding of when ALS succeeds [KB09, BK25].

Overparameterization has recently emerged as a powerful approach to mitigating non-convexity. Introducing more parameters than those of the ground-truth model often improves optimization dynamics in practice, even in complex settings like training deep neural networks. In our setting, the given tensor T has a rank r decomposition of the form T = ∑ r i =1 a i ⊗ b i ⊗ c i , and the goal is to find a decomposition of potentially larger rank k .

It is challenging to reason about overparameterization with iterative methods for tensor decomposition. Existing approaches based on lazy training and standard mean-field analysis requires an overparameterization of rank that depends polynomially or even exponentially on the ambient dimension. Surprisingly, the work of [WWL + 20] makes progress by showing global convergence for a variant of gradient descent with moderate overparameterization of k = O ( r 7 . 5 log n ) that is nearly independent of the ambient dimension n . The main question we address in this paper is:

Question. Does ALS admit a polynomial time global convergence guarantee with moderate overparameterization (a function of r and not n )?

Our main result answers this question in the affirmative. Concretely, our main contributions are the following:

- We prove that the ALS algorithm on a tensor with a mildly conditioned rankr decomposition 1 when overparameterized with k = O ( r 2 ) and with random initialization of { ( x i , y i , z i ) : i ∈ [ r ] } , converges with high probability to a global minimum (i.e., objective value 0 ).
- We also provide rigorous guarantees for ALS under overparameterization for the more general low-rank approximation problem

<!-- formula-not-decoded -->

1 By midly conditioned we mean that the condition numbers of factor matrices are bounded by poly( r )

We prove that ALS when overparameterized with k = O ( r 2 ) and initialized randomly, finds a solution whose objective value is competitive with the OPT r up to a multiplicative factor that is polynomial in only r (and independent of n ).

For both our results, a moderate overparameterization of k = O ( r 2 ) suffices. We suspect it may be challenging to improve the amount of overparameterization necessary. We remark that even for more computationally-intensive algorithms based on spectral methods, the best polynomial time guarantees require an overparameterization of k = O ( r 2 ) [BCV14, SWZ19]. We leave it as a direction for future work to investigate whether one can prove better upper or lower bounds on the amount of overparameterization necessary to recover a provable guarantee for ALS.

Recent work has developed a few different techniques for analyzing iterative methods with overparameterization. Techniques based on the lazy training approach argue that when the model is sufficiently overparameterized, the optimization problem is locally convex and the method will converge to a good solution near the initialization [COB19]. Lazy training analyses incur overparameterization that is polynomial in the ambient dimension n , which can be very large compared to the rank r .

The work of [WWL + 20] makes progress by instead adopting the framework of mean field analysis . While previous work that introduced this technique analyzes problems in the case of very large, or even infinite overparameterization [MMN18], [WWL + 20] was able to show global convergence for a variant of gradient descent with only moderate overparameterization of k = O ( r 7 . 5 log n ) 2 , achieving an exponentially better dependence on n than lazy training analyses for small r .

Our work develops an analysis that is significantly different from previous approaches, based on new matrix anticoncentration statements. At a high level, we observe that if the iterates X,Y,Z are sufficiently random and in the span of the components of the true tensor, then due to overparameterization they will form a basis for an appropriate space related to the components. If this occurs, then the next iteration of ALS will find a near-exact solution and converge. At initialization, X,Y,Z are independently fully random. However, they are random in the ambient space, and not restricted to the subspace of interest. The first iteration of ALS should in fact update X,Y,Z to be within the correct space. However, the updated X,Y,Z now exhibit significant dependencies on each other. Our analysis shows that despite this, the updated X,Y,Z are still random enough to form the appropriate basis. Thus, the crux of our argument is in showing that this iteration preserves enough randomness from the initialization. Quantitatively, this requires arguing about the least singular value of various structured random matrices that arise in the algorithm, along with careful matrix perturbation analyses. Our techniques are applied to a version of ALS that updates all the factor matrices in parallel. Similar guarantees, based on these techniques, for the standard ALS will appear in the arxiv version of the paper.

## 1.1 Related Work

We now describe related work on tensor decompositions and overparameterization, and place our work in the context of these prior works.

Tensor decomposition has a rich history going back to at least [Har70, Har72]. This decomposition into a sum of rank-one tensors is also referred to as CP decomposition or PARAFAC decomposition. See also [KB09] for other decomposition notions for tensors including Tucker decompositions. There are several iterative algorithms that are popular in practice like alternating least squares, alternate minimization, gradient descent and tensor power method [see KB09, AGH + 14, JGKA19, for more details]. In particular, the ALS algorithm was first introduced by [Har70, CC70], and has been the workhorse algorithm for tensor decomposition in practice [BK25]. While ALS is popular for its efficiency and steps towards understanding convergence of the iterates have been made [Usc12, WC14], we do not have a good understanding of when it converges to a global optimum solution.

The two results on tensor decompositions that is most relevant to our work are the work of Sharan and Valiant [SV17] who introduced and gave guarantees for an orthogonalized version of ALS, and the work of Wang, Wu, Lee, Ma and Ge [WWL + 20] on analyzing gradient descent in the overparameterized regime.

2 For orderℓ tensors the bound on overparameterization is O ( r 2 . 5 ℓ log( n )) . Our work only focuses on the order3 case.

Comparison to the Orthogonalized ALS algorithm. The work of Sharan and Valiant [SV17] introduced a variant of the ALS algorithm that orthogonalizes the factors in each step, in addition to the ALS update. As described in [SV17] this allows the ALS algorithm to avoid issues where multiple components of the decomposition capture the same factor of the tensor, when the rank1 terms have different magnitudes. Their work also proves guarantees under the assumption that the factors { a i : i ∈ [ r ] } of the decomposition are orthogonal or incoherent. The decomposition computed by the algorithm is not overparameterized i.e., k = r . However, the algorithm is more suited for settings where the target decomposition has near-orthogonal factors. Our analysis is for the ALS algorithm in the overparameterized regime. We do not make incoherence or orthogonality assumptions on the factors; we just need mild conditioning of the factor matrix (condition number that is polynomial in r ). One can interpret our results as proving that in the overparameterized setting, ALS does not face some fo the earlier issues pointed out in [SV17].

Overparameterized tensor decompositions using iterative algorithms. The work of [WWL + 20] analyzed a variant of gradient descent in the overparameterized regime of tensor decompositions. Their techniques were able to go beyond lazy-training analyses, and the standard mean-field analysis bounds that require overparameterization of polynomial or even exponential in the ambient dimension n . They were surprisingly able to provide guarantees for overparameterized rank k that is almost independent of the ambient dimension. Concretely, their guarantees hold for third-order tensors when the overparameterization is k = O ( r 7 . 5 log n ) ; for general orderℓ tensors they need k = O ( r 2 . 5 ℓ log n ) . In this work, we instead analyze ALS in the overparameterized regime. We can get guarantees for smaller overparameterized rank k = O ( r 2 ) . Moreover, we also approximation guarantees for overparameterized ALS even when the tensor is not exactly of rank r . We get guarantees for the more general low-rank approximation problem that incurs a loss that is within a multiplicative factor (depending polynomially only on r , and independent of k ) of the optimum value. To the best of our knowledge, we are unaware of any such guarantees for gradient descent and other iterative algorithms.

Analysis of other iterative algorithms for tensor decompositions. There are several other works that try to provide guarantees for iterative methods including alternating minimization, the tensor power method, and other gradient descent based algorithms. The work of [AGH + 14] analyzes the tensor power method, and provides guarantees that are specialized to the setting when the factors are orthogonal or near orthogonal [AGH + 14, AGJ14, AGJ17]. The works of [JO14, JGKA19] also analyze a variant of the alternating minimization algorithm, and provides convergence guarantees under nearby initialization. These works are not in the overparameterized regime ( k = r ). They find components one at a time but either require stronger assumptions, or provide local convergence guarantees. Finally, it is known that for certain matrix factorization problems and special settings of tensor decomposition (e.g., orthogonal factors), the non-convex optimization landscape is benign i.e., it does not have any local optima that are not globally optimal [Ma21]. However, the general tensor decomposition is highly non-convex with bad local minima as shown in [WWL + 20].

Other tensor decomposition algorithms. Theoretical guarantees have been established for the simultaneuous diagonalization algorithm [Har72, LRA93, Moi18] and its variants, algebraic methods [DLCC07, JLV23, Koi24, KMW24], sum-of-squares algorithms [BKS15, HSSS16, MSS16]. In the overparameterized setting, spectral methods and algorithms based on subspace embeddings can also find decompositions of rank k = r 2 in polynomial time even for the more general low-rank approximation problem with an error that is constant factor competitive with the best rankr decomposition [SWZ19, BCV14]. The focus of our paper is to prove rigorous guarantees for the ALS algorithm, which is the most popular algorithm in practice.

## 2 Algorithm, Results, and Preliminaries

The Alternating Least Squares (ALS) algorithm for tensor decomposition has many variants. The version that we analyze is given in Algorithm 1 3 . Given a tensor T , the algorithm randomly initializes

3 The version that we analyze is a parallel version of the commonly used ALS method, it however seems like our techniques can be extended to give guarantees for the standard, sequential version of ALS. Our analysis for

"sequential" ALS will appear in the arxiv version of the paper.

the three modes X,Y,Z ∈ R n × k of a decomposition, corresponding to a model tensor

<!-- formula-not-decoded -->

On each iteration, the algorithm updates each mode individually in parallel to minimize the least squares objective:

<!-- formula-not-decoded -->

Since ̂ T is multilinear in X,Y,Z , this is a least squares problem with respect to each mode. The least squares problem could have multiple optima, in fact due to the overparameterization, we expect this to be the case. Typically, ALS is implemented using a linear system solver for each of these subproblems [BK25]; this is also what we will analyze. The updates (Algorithm 1, Lines 8, 12, 16) hence correspond to

<!-- formula-not-decoded -->

Here we use the shorthand flatten( T, mode A, modes B ⊗ C ) to mean that the order-3 tensor T is reshaped into a matrix, by taking each n × n slice in the B,C modes and vectorizing it into an n 2 -dimensional row of the flattened matrix. There will be n such rows corresponding to mode A . (This is explained in more detail in Section 2.1.) Also, M † refers to the Moore-Penrose pseudoinverse of the matrix M .

## Algorithm 1 Alternating Least Squares (ALS) for order-3 tensor decomposition

```
Require: Tensor T ∈ R n × n × n , rank r of T , error tolerance ε 1: k ← Θ( r 2 ) // rank of overparameterized model 2: X (0) ∈ R n × k ←N (0 , 1) n × k , Y (0) ∈ R n × k ←N (0 , 1) n × k , Z (0) ∈ R n × k ←N (0 , 1) n × k 3: // randomly initialize model 4: t ← 0 5: while true do 6: // X,Y,Z updates can be evaluated in parallel 7: // X update 8: err X , X ( t +1) ← min X ∈ R n × k and arg min X ∈ R n × k of ∥ ∥ ∥ T -∑ k i =1 X i ⊗ Y ( t ) i ⊗ Z ( t ) i ∥ ∥ ∥ 2 F 9: if err X ≤ ε then 10: return X ( t +1) , Y ( t ) , Z ( t ) 11: // Y update 12: err Y , Y ( t +1) ← min Y ∈ R n × k and arg min Y ∈ R n × k of ∥ ∥ ∥ T -∑ k i =1 X ( t ) i ⊗ Y i ⊗ Z ( t ) i ∥ ∥ ∥ 2 F 13: if err Y ≤ ε then 14: return X ( t ) , Y ( t +1) , Z ( t ) 15: // Z update 16: err Z , Z ( t +1) ← min Z ∈ R n × k and arg min Z ∈ R n × k of ∥ ∥ ∥ T -∑ k i =1 X ( t ) i ⊗ Y ( t ) i ⊗ Z i ∥ ∥ ∥ 2 F 17: if err Z ≤ ε then 18: return X ( t ) , Y ( t ) , Z ( t +1) 19: t ← t +1
```

We give the following guarantee for Algorithm 1. In what follows, we will assume that ALS uses a sub-routine for solving the linear system in polynomial time; concretely, it computes the pseudo-inverse solution up to arbitrary precision ε &gt; 0 in Frobenius norm in time polynomial in n, log(1 /ε ) .

Theorem 2.1 (Guarantee for overparameterized ALS) . For any constant c 0 &gt; 0 , there exists constants c = c ( c 0 ) ≥ 1 and γ 0 ∈ (0 , 1) , such that the following holds. Let A,B,C ∈ R n × r be the factor

matrices of the decomposition of a rankr tensor T ,

<!-- formula-not-decoded -->

and suppose the condition numbers κ ( A ) , κ ( B ) , κ ( C ) ≤ r c 0 . Then, given T , an error parameter ε , and a k ∈ N satisfying cr 2 ≤ k ≤ n γ 0 , with probability at least 1 -o (1) , Algorithm 1 runs in polynomial time and in O (1) steps finds a rankk decomposition X,Y,Z ∈ R n × k of T . That is, X,Y,Z satisfy

<!-- formula-not-decoded -->

The above theorem shows that ALS succeeds from random initialization with overparameterized rank k = O ( r 2 ) . For the theorem, we analyze standard Gaussian initializiation, the scale of the random initialization does not matter much. For the above theorem, we assume that the factor matrices A,B and C have condition numbers upper bounded by some large polynomial in r . This assumption on the condition numbers is quite mild: for example, it is satisfied w.h.p. for a natural smoothed analysis model. 4 It is weaker than incoherence or orthogonality assumptions, as the vectors in our setting can be quite correlated. Moreover, we believe the assumption to be an artifact of our analysis, and may not be necessary.

Finally, our analysis also implies approximation guarantees for overparameterized ALS with k = O ( r 2 ) under random initialization, in the more general low-rank approximation problem, where T = ∑ r i =1 a i ⊗ b i ⊗ c i + E , where ∥ E ∥ F is the error.

Theorem 2.2 (Low-rank tensor approximation using overparameterized ALS) . For any constant c 0 &gt; 0 , there exists constants c = c ( c 0 ) ≥ 1 and γ 0 ∈ (0 , 1) , such that the following holds. Let A,B,C ∈ R n × r be the decomposition of a rankr tensor T ,

<!-- formula-not-decoded -->

and suppose the condition numbers κ ( A ) , κ ( B ) , κ ( C ) ≤ r c 0 . Then, given T , r , and an error parameter ε , for cr 2 ≤ k ≤ n γ 0 , with probability at least 1 -o (1) , Algorithm 1 runs in polynomial time and in O (1) steps finds a rankk decomposition X,Y,Z ∈ R n × k of T . That is, X,Y,Z satisfy

<!-- formula-not-decoded -->

The above theorem gives an ALS guarantee under overparameterization for the optimization problem in (2) in the general setting when OPT r &gt; 0 , and generalizes Theorem 2.1 (special case when OPT r = 0 ). The multiplicative factor loss in the objective compared to OPT r is polynomial in r and independent of the ambient dimension n . To the best of our knowledge, such an approximation guarantee was not known previously for ALS or other iterative algorithms like gradient descent.

## 2.1 Notation and Preliminaries

We now introduce some notation and preliminaries that will be used in the rest of the paper. We refer to a tensor T by its decomposition into factor matrices . That is, we associate T with the matrices X,Y,Z ∈ R n × r , where r is the rank of T and

<!-- formula-not-decoded -->

where x i , y i , z i refer to columns i of X,Y,Z respectively, and ⊗ when applied to vectors is the outer product. That is, each component x i ⊗ y i ⊗ z i is an n × n × n tensor. Since each component is an

4 E.g., in the smoothed analysis model, you have an arbitrary matrix which is normalized to have columns of at most unit length, with a random perturbation of length 1 / poly( r ) .

outer product of 3 vectors, this is an order -3 tensor. Each direction of the outer product is referred to as a mode , and we will sometimes refer to the modes of the tensor by the corresponding factor matrix, i.e., the X mode, Y mode, and Z mode. (These are the analogues of the rows and columns of a matrix/order-2 tensor.) The squared Frobenius norm of a tensor T is the sum of the squares of its entries.

It will also be useful to interact with flattened forms of the tensors we analyze. We use ⊙ to refer to the Khatri-Rao product of two matrices Y, Z ∈ R n × r , which has columns given by

<!-- formula-not-decoded -->

where vec( · ) reformats an n × n matrix as an n 2 -dimensional vector. This is useful to flatten tensors into matrices. In particular, we have

<!-- formula-not-decoded -->

This reshaping into a matrix is exactly what arises in the least squares problem in Algorithm 1 (Lines 8, 12, 16). In Section 2 we describe the flattening operation for a tensor T with factor matrices X,Y,Z . The flattening is just a reformatting of the entries of T , so computing it does not require explicit access to the factor matrices X,Y,Z . However, it is indeed the case that

<!-- formula-not-decoded -->

which will be useful for our analysis.

Another useful tensor product on matrices is the Kronecker product which we refer to as ⊗ . 5 The Kronecker product of two matrices X ∈ R n × r , Y ∈ R m × k is an nm × rk matrix that satisfies

<!-- formula-not-decoded -->

(The entries can also be written explicitly as ( X ⊗ Y ) n ( i 1 -1)+ i 2 ,k ( j 1 -1)+ j 2 = X i 1 ,j 1 Y i 2 ,j 2 .) Since the columns of a Khatri-Rao product are flattenings of rank-1 matrices, this gives the following identity. For A ∈ R n × r , B ∈ R m × k , and X ∈ R r × ℓ , Y ∈ R k × q , we have

<!-- formula-not-decoded -->

We will use the Moore-Penrose pseudoinverse M † ∈ R m × n of a matrix M ∈ R n × m . This is defined as a matrix such that

<!-- formula-not-decoded -->

where the nullspace of M , Null( M ) , is the subspace of vectors mapped to 0 by M : ⟨{ x | Mx = 0 }⟩ , and the image of M , Im( M ) is the subspace of vectors that can be realized by the linear transformation M : ⟨{ y | ∃ x, Mx = y }⟩ . Here and elsewhere we use ⟨·⟩ to denote the linear span. There are also a few direct expressions for the pseudoinverse that will be useful to us. For M ∈ R n × r , where r ≤ n , if M is rank r ( M is full column rank), we have

<!-- formula-not-decoded -->

For M ∈ R r × n , where r ≤ n , if M is rank r ( M is full row rank), we have

<!-- formula-not-decoded -->

## 3 Analysis of the Algorithm

Fix a tensor T = ∑ r i =1 a i ⊗ b i ⊗ c i for some rankr decomposition A,B,C ∈ R n × r . Consider iteration t of ALS on T . Without loss of generality, we focus on the update to mode Z (Algorithm 1, Line 8). ALS will converge on this step if X ( t ) , Y ( t ) , Z ( t +1) fits T , i.e., the error between the model tensor and the true tensor is below ε . For the purposes of the overview, we will ignore the ε error term, and refer to this as perfectly fitting the tensor.

5 This overloads the notation that we use for the outer product on vectors. When applied to vectors we will always mean the outer product. When applied to matrices with dimensions larger than 1 we will always mean the Kronecker product.

The least-squares problem for Z at time t +1 is to reconstruct the slices of T as linear combinations of the X ( t ) i ⊗ Y ( t ) i . That is, slice j of T is given by

<!-- formula-not-decoded -->

The least-squares problem along mode Z is then

<!-- formula-not-decoded -->

which is n independent least-squares problems, one for each slice of T or row of Z . Thus, ALS will fit T and converge if, for every slice j , T : , : ,j is realizable as a linear combination of the { x ( t ) i ⊗ y ( t ) i : i ∈ [ k ] } . Since every slice j of T is a linear combination of { a i ⊗ b i : i ∈ [ r ] } , a sufficient condition for convergence is that

<!-- formula-not-decoded -->

The two lines are reshapings of the same statement. Now suppose for a moment that the columns of Y ( t ) and X ( t ) were each drawn independently and randomly from colspan( A ) and colspan( B ) respectively (for example, from standard Gaussians over the r -dimensional spaces). Then we would have that since k ∈ Ω( r 2 ) , with high probability

<!-- formula-not-decoded -->

where A ⊗ B denotes the Kronecker product of A with itself. Since colspan( A ⊙ B ) ⊆ colspan( A ⊗ B ) , because the Khatri-Rao product is a subset of the columns of the Kronecker product, (5) is sufficient to ensure convergence.

Of course, the columns of X and Y are not initialized randomly in the span of A and B , instead they are random in the whole n -dimensional space. This means that at initialization, each column of X ( Y ) can be thought of as the sum of a random vector in the span of A ( B ), and a component orthogonal to A ( B ). Components orthogonal to the span of A ( B ) only make the Frobenius error of the decomposition higher, since they contribute terms that are orthogonal to T . That is, denote X = X + X ⊥ , where the columns of X are in the column span of A , and the columns of X ⊥ are orthogonal to the column span of A . Then

<!-- formula-not-decoded -->

The first step of ALS (Algorithm 1, Line 8) will set X so that the second term is 0, which since Y and Z are randomly initialized means setting X ⊥ = 0 . If the first step of ALS only set X ⊥ , Y ⊥ , Z ⊥ = 0 and did not modify X,Y , Z , then on the second step (4) would hold, and ALS would converge.

This is however not the case as ALS updates X as the minimizer of the least squares objective. Thus X (1) = X (1) no longer has independent random columns. Instead it is a function of X (0) , Y (0) , Z (0) , and A,B,C . Our main technical insight is that, despite X (1) , Y (1) , Z (1) having this complex dependence on each other and A , B and C , (4) holds with high probability. It is straightforward to show that after the first iteration, each of the factor matrices will be in the span of A , B and C respectively, which means that X (1) ⊙ Y (1) will be in the span of A ⊗ B . Thus, proving that X (1) ⊙ Y (1) has rank r 2 implies that colspan ( X (1) ⊙ Y (1) ) = colspan ( A ⊗ B ) , then condition (4) follows since colspan( A ⊙ B ) ⊆ colspan( A ⊗ B ) . This is captured by Theorem 4.1, which is the main technical component of our proof, and the focus of the next section.

## 4 Least Singular Value Bound through Anti-Concentration

For the proofs, we will refer to X (1) , Y (1) , Z (1) as ˆ X, ˆ Y , ˆ Z . In this section we give an overview of the proof of our claim that ˆ X ⊙ ˆ Y spans A ⊗ A when X,Y and Z are initialized randomly with their

entries being i.i.d. standard Gaussians and, in particular, show that this statement is true in a robust sense. Formal statements of this claim as well as detailed proofs can be found in Appendix C. We give an inverse polynomial in n lower bound on the least singular value of the matrix. Our main technical contribution is proving the following theorem.

Theorem 4.1. Under the assumptions of Theorem 2.1 with probability at least 1 -o (1) we have that:

<!-- formula-not-decoded -->

Assuming that A and B are mildly conditioned we can in fact turn our attention to showing least singular value bounds for the following matrix:

<!-- formula-not-decoded -->

One main challenge is that the entries of the matrices ( Y ⊙ Z ) † and ( X ⊙ Z ) † are random but highly dependent. While there have been powerful techniques developed recently for proving least singular value bounds of matrices with polynomial entries [BESV24], the random matrix in our setting does not exhibit such structure and is therefore difficult to reason about directly.

We proceed by first arguing about the matrix pseudoinverse by showing in Lemma C.1 that with probability at least 1 -o (1) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The high level intuition is that if the columns of the matrix that we are taking the pseudoinverse of were Gaussian vectors then they would be mostly orthogonal, meaning that the pseudoinverse would be close to the transpose of the matrix. Our analysis shows that the same intuition translates to matrices whose columns are tensor products of Gaussian vectors.

Using that, it suffices, up to poly( n ) factors, to analyze the least singular value of the matrix:

<!-- formula-not-decoded -->

Furthermore, let:

<!-- formula-not-decoded -->

Assume for now that we have a guarantee stating that σ r 2 ( ˆ L ) ≥ 1 poly( k ) . As we will discuss later, this is challenging, and much of our technical work is devoted to proving this. We can now use this to prove the existence of a matrix M such that:

<!-- formula-not-decoded -->

Matrix M expresses the columns of the Kronecker product of matrices ( B ⊙ C ) ⊤ ( Y ⊙ Z ) and ( A ⊙ C ) ⊤ ( X ⊙ Z ) as linear combinations of the columns of their Khatri-Rao product. Such a matrix is guaranteed to exist only because we have assumed that ˆ L spans R r 2 . Using (3) we can express L as:

<!-- formula-not-decoded -->

where E = I ⊙ E 2 + E 1 ⊙ I + E 1 ⊙ E 2 ; further by Lemma C.8, ∥ E ∥ ≤ O (√ log( k ) n + log( k ) k n ) . We can now leverage the existence of matrix M to get that:

<!-- formula-not-decoded -->

where we have crucially used that, by definition of M , M ( I ⊙ I ) = I . In Lemma C.3, we use that σ r 2 ( ˆ L ) ≥ 1 / poly( k ) to prove that the spectral norm of M is also bounded by a polynomial in k . Hence

<!-- formula-not-decoded -->

when n is a sufficiently large compared to k . We now have that:

<!-- formula-not-decoded -->

thus establishing the main least singular value claim.

In the above proof overview, we assumed a lower bound on the least singular value of ˆ L . The proof of this claim is quite technical and involves a careful net argument along with anti-concentration of low-degree polynomials of independent random variables [CW01]. We first express the columns of the matrix in a convenient way that factors the dependency of having Z on both sides of the Khatri-Rao product. We then argue by applying an ε -net argument and showing that for every fixed vector in R r 2 , the probability that the inner products between the fixed vector and all the columns of ˆ L is negligible is exponentially small in k . The formal statement with a detailed proof of it can be found in Lemma C.2.

## 5 Conclusion

Our work proves rigorous polynomial-time global convergence guarantees of the popular ALS method for tensor decomposition with moderate overparameterization of O ( r 2 ) . It has been challenging to establish rigorous guarantees for iterative heuristics that are the state-of-the-art in practice. Our analysis is based on new matrix anticoncentration techniques to argue about the iterates, that differs significantly from previous approaches. Our theoretical results on overparameterization are also supported by empirical evaluations in Appendix D. It would be compelling to use these techniques to analyze gradient descent, or prove global convergence guarantees for other non-convex optimization heuristics.

## Acknowledgments

The authors were supported by the NSF-funded Institute for Data, Econometrics, Algorithms and Learning (IDEAL) through the grant NSF ECCS-2216970 and the NSF via grant CCF-2154100. In addition, Vaidehi Srinivas was also partly supported by the Northwestern Presidential fellowship.

## References

- [AGH + 14] Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. Journal of Machine Learning Research , 15, 2014.
- [AGJ14] Animashree Anandkumar, Rong Ge, and Majid Janzamin. Guaranteed non-orthogonal tensor decomposition via alternating rank1 updates. arXiv preprint arXiv:1402.5180 , 2014.
- [AGJ17] Animashree Anandkumar, Rong Ge, and Majid Janzamin. Analyzing tensor power method dynamics in overcomplete regime. Journal of Machine Learning Research , 18, 2017.
- [BCV14] Aditya Bhaskara, Moses Charikar, and Aravindan Vijayaraghavan. Uniqueness of tensor decompositions with applications to polynomial identifiability. Conference on Learning Theory , 2014.
- [BESV24] Aditya Bhaskara, Eric Evert, Vaidehi Srinivas, and Aravindan Vijayaraghavan. New tools for smoothed analysis: Least singular value bounds for random matrices with dependent entries. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing , STOC 2024, page 375-386, New York, NY, USA, 2024. Association for Computing Machinery.
- [Bha97] Rajendra Bhatia. Matrix Analysis , volume 169. Springer, 1997.

- [BK25] Grey Ballard and Tamara G. Kolda. Tensor Decompositions for Data Science . Cambridge University Press, 2025.
- [BKS15] Boaz Barak, Jonathan A. Kelner, and David Steurer. Dictionary learning and tensor decomposition via the sum-of-squares method. In Proceedings of the Symposium on Theory of Computing , 2015.
- [CC70] J. Douglas Carroll and Jih-Jie Chang. Analysis of individual differences in multidimensional scaling via an n-way generalization of 'eckart-young' decomposition. Psychometrika , 35(3):283-319, 1970.
- [COB19] Lénaïc Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019.
- [CW01] Anthony Carbery and James Wright. Distributional and L q norm inequalities for polynomials over convex bodies in R n . Math. Res. Lett. , 8(3):233-248, 2001.
- [DLCC07] Lieven De Lathauwer, Josphine Castaing, and Jean-François Cardoso. Fourth-order cumulant-based blind identification of underdetermined mixtures. IEEE Trans. on Signal Processing , 55, 2007.
- [Har70] Richard A. Harshman. Foundations of the parafac procedure: Models and conditions for an 'explanatory' multimodal factor analysis. UCLA Working Papers in Phonetics , 16:1-84, 1970.
- [Har72] Richard A. Harshman. Determination and proof of minimum uniqueness conditions for PARAFAC1. UCLA Working Papers in Phonetics , 22:111-117, 1972.
- [Hås90] Johan Håstad. Tensor rank is np-complete. Journal of algorithms , 11(4):644-654, 1990.
- [HJ12] Roger A Horn and Charles R Johnson. Matrix analysis . Cambridge university press, 2012.
- [HL13] Christopher J Hillar and Lek-Heng Lim. Most tensor problems are np-hard. Journal of the ACM (JACM) , 60(6):1-39, 2013.
- [HSSS16] Samuel B. Hopkins, Tselil Schramm, Jonathan Shi, and David Steurer. Fast spectral algorithms from sum-of-squares proofs: Tensor decomposition and planted sparse vectors. In STOC , 2016.
- [JGKA19] Majid Janzamin, Rong Ge, Jean Kossaifi, and Animashree Anandkumar. Spectral learning on matrices and tensors. Foundations and Trends in Machine Learning , 12, 11 2019.
- [JLV23] Nathaniel Johnston, Benjamin Lovitz, and Aravindan Vijayaraghavan. Computing linear sections of varieties: quantum entanglement, tensor decompositions and beyond . In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS) , pages 1316-1336, Los Alamitos, CA, USA, November 2023. IEEE Computer Society.
- [JO14] Prateek Jain and Sewoong Oh. Provable tensor factorization with missing data. Advances in Neural Information Processing Systems , 27, 2014.
- [KB09] Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. SIAM review , 51(3), 2009.
- [KMW24] Pravesh K. Kothari, Ankur Moitra, and Alexander S. Wein. Overcomplete tensor decomposition via koszul-young flattenings, 2024.
- [Koi24] Pascal Koiran. On the uniqueness and computation of commuting extensions. Linear Algebra and its Applications , 703:645-666, 2024.

- [KPAP19] Jean Kossaifi, Yannis Panagakis, Anima Anandkumar, and Maja Pantic. Tensorly: Tensor learning in python. Journal of Machine Learning Research(JMLR) , 20(26), 2019.
- [Lov10] Shachar Lovett. An elementary proof of anti-concentration of polynomials in gaussian variables. In Electronic Colloquium on Computational Complexity (ECCC) , volume 17, page 182, 2010.
- [LRA93] S. E. Leurgans, R. T. Ross, and R. B. Abel. A decomposition for three-way arrays. SIAM Journal on Matrix Analysis and Applications , 14(4):1064-1083, 1993.
- [Ma21] Tengyu Ma. Why Do Local Methods Solve Nonconvex Problems? , page 465-485. Cambridge University Press, 2021.
- [MMN18] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences , 115(33):E7665-E7671, 2018.
- [Moi18] Ankur Moitra. Algorithmic Aspects of Machine Learning . Cambridge University Press, 2018.
- [MSS16] Tengyu Ma, Jonathan Shi, and David Steurer. Polynomial-time tensor decompositions with sum-of-squares. In Proceedings of the 57th Annual IEEE Symposium on the Foundations of Computer Science (FOCS) , 2016.
- [SV17] Vatsal Sharan and Gregory Valiant. Orthogonalized ALS: A theoretically principled tensor decomposition algorithm for practical use. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning , volume 70 of Proceedings of Machine Learning Research , pages 3095-3104. PMLR, 06-11 Aug 2017.
- [SWZ19] Zhao Song, David P. Woodruff, and Peilin Zhong. Relative error tensor low rank approximation. In Symposium on Discrete Algorithms (SODA) , 2019.
- [Usc12] André Uschmajew. Local convergence of the alternating least squares algorithm for canonical tensor approximation. SIAM Journal on Matrix Analysis and Applications , 33(2):639-652, 2012.
- [Ver18] Roman Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- [Vij20] Aravindan Vijayaraghavan. Beyond the Worst-Case Analysis of Algorithms , chapter 19 Efficient Tensor Decomposition. Cambridge University Press, 2020.
- [WC14] Liqi Wang and Moody T Chu. On the global convergence of the alternating least squares method for rank-one approximation to generic tensors. SIAM Journal on Matrix Analysis and Applications , 35(3):1058-1072, 2014.
- [WWL + 20] Xiang Wang, Chenwei Wu, Jason D Lee, Tengyu Ma, and Rong Ge. Beyond lazy training for over-parameterized tensor decomposition. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 21934-21944. Curran Associates, Inc., 2020.

## A More Preliminaries

Our analysis involves various operations involving Khatri-Rao products, which are quite challenging to reason about. We also make extensive use of the Kronecker product defined in subsection 2.1, which are more natural products for matrices, and easier to reason about. We have the following simple facts about the Kronecker product:

Fact A.1. Let A,B ∈ R n × k , then for the Kronecker product of A and B , it holds that:

<!-- formula-not-decoded -->

2.

3.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof follows easily by noting that if U a Σ a V a and U b Σ b V b are the singular value decompositions of A and B respectively then ( U a ⊗ U b )(Σ a ⊗ Σ b )( V a ⊗ V b ) is a singular value decomposition of matrix A ⊗ B .

We now introduce some additional notation that we use in our analysis. For a matrix A we use Π A to denote the projection matrix onto the subspace spanned by the columns of A . We use λ i ( A ) to denote the i -th eigenvalue of matrix A , we also use σ i ( A ) to denote the i -th singular value of matrix A and κ ( A ) to denote the condition number of A . For the factor matrices A , B and C of the ground truth we use κ = max( κ ( A ) , κ ( B ) , κ ( C )) .

For X,Y and Z being the random initializations of our algorithm we use:

<!-- formula-not-decoded -->

and also:

<!-- formula-not-decoded -->

notice that ˆ L is the Khatri-Rao product of matrices ( B ⊙ C ) ⊤ ( Y ⊙ Z ) and ( A ⊙ C ) ⊤ ( X ⊙ Z ) while ˆ L K is their Kronecker product. Furthermore, we use D z to denote the:

<!-- formula-not-decoded -->

The following expression is useful in our analysis:

Lemma A.2. For the columns of matrices ( B ⊙ C ) ⊤ ( Y ⊙ Z ) and ( A ⊙ C ) ⊤ ( X ⊙ Z ) we have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We first observe that the j -th entry of vector ( B ⊙ C ) ⊤ ( Y ⊙ Z ) i is given by ⟨ b j , y i ⟩⟨ c j , z i ⟩ . The vector whose j -th entry is given by ⟨ b j , y i ⟩ is B ⊤ y i . Left multiplying this by D z i gives the result.

We now prove some useful claims.

Claim A.3. Given matrices A ∈ R n 1 × m , B ∈ R n 2 × m , we have

<!-- formula-not-decoded -->

where ∥ A ∥ = σ 1 ( A ) is the spectral norm and σ r ( A ) is the least singular value of the matrix A .

Proof. We have that

<!-- formula-not-decoded -->

For the upper bound, from (9)

<!-- formula-not-decoded -->

A similar proof shows that ∥ A ⊙ B ∥ 2 F ≤ ∥ B ∥ 2 F · ∥ A ∥ 2 .

For the lower bound, from (9) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following claim shows that for a mildly conditioned decomposition, the Frobenius norm of the ground-truth tensor T can be sandwiched up to poly( r ) factors by the corresponding norms of the factor matrices.

Claim A.4. Let T = ∑ r i =1 a i ⊗ b i ⊗ c i . Then

<!-- formula-not-decoded -->

where ∥ A ∥ = σ 1 ( A ) is the spectral norm and σ r ( A ) is the least singular value of the matrix A . A symmetric statement also holds with the Frobenius norm of A or B instead of C being used.

Proof. First, we note that by considering an appropriate flattening of the tensor,

<!-- formula-not-decoded -->

We prove the lower bound first. Note that A has full column rank. Hence

<!-- formula-not-decoded -->

by using lower bound in Claim A.3. Similarly for the upper bound, we use the upper bound in Claim A.3 to conclude

<!-- formula-not-decoded -->

as required.

We will use the classic perturbation bound for topk singular space of a matrix due to Davis and Kahan. The following is a consequence that we use in our robust analysis [see Theorem VII.3.2 Bha97].

Fact A.5. Let M, ˜ M ∈ R d × m , and let Π denote the projection matrix onto the column space of M , and let ˜ Π denote the left singular space of ˜ M corresponding to the singular values larger than δ respectively. Then for a universal constant c &gt; 0

<!-- formula-not-decoded -->

where ∥ M ∥ refers to the operator norm of M .

## B Formal statements of the theorems

We give the formal versions of the main theorems below.

Theorem B.1 (Guarantee for General decompositions) . For any constant c 0 &gt; 0 , there exists a constant c = c ( c 0 ) ≥ 1 such that the following holds. Let A,B,C ∈ R n × r be the decomposition of a rankr tensor T ,

<!-- formula-not-decoded -->

and suppose the condition numbers κ ( A ) , κ ( B ) , κ ( C ) ≤ r c 0 ≤ n γ 0 where γ 0 = γ 0 ( c 0 ) is a constant. Then, given T , r , and an error parameter ε , for k = C op · r 2 , C op = C op ( c 0 ) is a constnant, Algorithm 1 runs in polynomial time and in O (1) steps finds a rankk decomposition X,Y,Z ∈ R n × k of T , i.e., ˜ X, ˜ Y , ˜ Z satisfy

<!-- formula-not-decoded -->

As discussed in the proof overview, Theorem B.1 uses the following theorem, generalized for asymmetric decompositions

Theorem B.2 (Generalization of Theorem 4.1 for asymmetric decompositions) . Under the assumptions of Theorem B.1 with probability at least 1 -o (1) , we have that:

<!-- formula-not-decoded -->

We give a proof of this Theorem in C.1

The following theorem is a robust version of Theorem B.1 and gives a guarantee for low rank approximations

Theorem B.3 (Guarantee for low-rank approximations) . For any constant c 0 &gt; 0 , there exists a constant c = c ( c 0 ) ≥ 1 such that the following holds. Let A,B,C ∈ R n × r be matrices such that for tensor tensor T :

<!-- formula-not-decoded -->

and suppose the condition number κ ( A ) , κ ( B ) , κ ( C ) ≤ r c 0 ≤ n γ 0 , where γ 0 = γ 0 ( c 0 ) is a constant. Then, given T , r , and an error parameter ε , for k = C op r 2 with C op = C op ( c 0 ) being a constant, Algorithm 1 runs in polynomial time (in n, k, log(1 /ε ) ) and in O (1) steps finds a rankk decomposition ˜ X, ˜ Y , ˜ Z ∈ R n × k of T , i.e., ˜ X, ˜ Y , ˜ Z satisfy

<!-- formula-not-decoded -->

We note that Theorem B.1 is a special case of Theorem B.3. We give a proof of B.3 in subsection C.2; this also implies the correctness of the more specialized Theorem B.1.

## C Analysis and Proofs

## C.1 Least singular value bound for ˆ X ⊙ ˆ Y

In this subsection we give a proof of Theorem B.2 . In particular we show that under the assumptions of Theorem B.1, with probability 1 -o (1) :

<!-- formula-not-decoded -->

We use the following lemmas in our analysis.

Lemma C.1 allows us to go from the ( X ⊙ Z ) † ⊤ to matrix X ⊙ Z times 1 /n 2 multiplied by identity plus an error matrix E whose spectral norm we bound. It is much easier to argue about matrix ˆ X ⊙ ˆ Y when we express ( X ⊙ Z ) † ⊤ and ( Y ⊙ Z ) † ⊤ like that.

Lemma C.1 (Pseudoinverse transpose simplification) . Let X,Z ∈ R n × k be matrices with i.i.d. standard Gaussian random variables as entries, then there exists an absolute constant C such that, with probability at least 1 -1 k :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We have that, with probability 1 , the columns of matrix ( X ⊙ Z ) are linearly independent, meaning that we can express the pseudoinverse as :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It therefore suffices to show that:

<!-- formula-not-decoded -->

where E has small spectral norm. Let for convenience:

<!-- formula-not-decoded -->

We show that all the eigenvalues of matrix W are close to 1 and use that to show that the inverse is close to identity in spectral norm. Specifically, let for convenience:

<!-- formula-not-decoded -->

By Lemma C.4 we have that with probability at least 1 -1 k , for every eigenvalue of W :

<!-- formula-not-decoded -->

We condition on that event. The eigenvalues of matrix W -1 are the reciprocals of the eigenvalues W , meaning that assuming that α ( n, k ) ≤ 1 2 , we get that:

<!-- formula-not-decoded -->

Now let E = W -1 -I and note that:

<!-- formula-not-decoded -->

Furthermore, E is a symmetric matrix meaning that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking transpose:

which in turn implies that:

the lemma follows.

Lemma C.2 is the main technical lemma for the proof, it gives us a least singular value bound for matrix ˆ L and a bound on the Frobenius norm of matrix ˆ L K . This allows us to bound matrix M by poly( k ) . Putting everything together we show that L = ˆ L ( I + ME ) where ˆ L has non-negligible least singular value and ∥ ME ∥ = o (1) giving us a non-negligible bound on the least singular value of L which in turn gives us the bound on σ r 2 ( ˆ X ⊙ ˆ Y )

Lemma C.2 (Bound on least singular value of ˆ L and Frob. norm of ˆ L K ) . With probability at least 1 -2 k -exp( -4 r ) -( 1 k ) r 2 , the following hold:

1. For the least singular value of matrix ˆ L , we have that:

<!-- formula-not-decoded -->

where C CW is an absolute constant coming from the Carbery-Wright inequality, Theorem C.7

2. For the matrix:

<!-- formula-not-decoded -->

we have that its Frobenius norm is bounded by:

<!-- formula-not-decoded -->

Proof. We first focus on item 1. We condition on the event of Lemma C.5 which happens with probability at least 1 -2 k and we have that for every i, j :

<!-- formula-not-decoded -->

We will prove the claim using the variational characterization of singular values, in particular we will show that for very v ∈ R r 2 , we have that:

<!-- formula-not-decoded -->

We consider a fine enough ε -net, N , of S r 2 -1 , where ε = ε ( k ) is an inverse polynomial to be determined later. Note that by Corollary 4.2.13 in [Ver18], we have that |N| ≤ ( 3 ε ) r 2 . Let u ∈ N , we will bound the probability that:

<!-- formula-not-decoded -->

where δ is a parameter to be specified later. We have that:

<!-- formula-not-decoded -->

where we have used that all the columns of matrix ˆ L are independent. W now fix an i and bound the probability that ∣ ∣ ∣ ⟨ ˆ L i , u ⟩ ∣ ∣ ∣ ≤ δ σ r ( A ) σ r ( B ) σ r ( C ) 2 k 4 r 2 . We use that, by Lemma A.2, we can express the i -th column of matrix ˆ L as:

<!-- formula-not-decoded -->

where D z i = diag ( {⟨ a j , z i ⟩} ) , we can write the inner product with u as:

<!-- formula-not-decoded -->

For the least singular value of matrix BD z i ⊗ AD z i we have that:

<!-- formula-not-decoded -->

let for convenience v = ( AD z i ) ⊗ ( AD z i ) u , for the norm of v we have that:

<!-- formula-not-decoded -->

We are interested in bounding the probability that |⟨ y i ⊗ x i , v ⟩| ≤ δ σ r ( A ) σ r ( B ) σ r ( C ) 2 k 4 r 2 . We will use the Carbery-Wright inequality ([CW01], [Lov10]) to bound the probability. We have that f ( x, y ) = 1 ∥ v ∥ ⟨ y ⊗ x, v ⟩ is a degree 2 polynomial, satisfying:

<!-- formula-not-decoded -->

Applying Theorem C.7 we get that there exists a constant C CW :

<!-- formula-not-decoded -->

which in turn gives us that:

<!-- formula-not-decoded -->

We can now use equation 14 to get that:

<!-- formula-not-decoded -->

We now take a union bound over all elements of the ε -net N , to get that with probability at least 1 -( C CW · δ 1 / 2 ) k ( 3 ε ) r 2 , the bound holds. By Lemma C.6 we have that with probability at least 1 -exp( -4 r ) , for every i :

<!-- formula-not-decoded -->

We use that to bound the spectral norm of ˆ L . We first bound the norm of every column, we have that:

<!-- formula-not-decoded -->

We now use that the spectral norm is bounded by the Frobenius norm which is in turn bounded by √ r times the largest norm of a column to get that:

<!-- formula-not-decoded -->

Again applying the union-bound, we get that with probability at least 1 -exp( -4 r ) -( C CW · δ 1 / 2 ) k ( 3 ε ) r 2 both of these events happen. We now take an arbitrary vector v ∈ S r 2 -1 , we have that there exists a vector u ∈ N such that ∥ u -v ∥ ≤ ε :

<!-- formula-not-decoded -->

We take δ = 1 C 2 CW · k and ε so that we can take ∥ ˆ Lv ∥ to be at least half as large as what we get on the ε -net. In particular, we take:

<!-- formula-not-decoded -->

we get that:

<!-- formula-not-decoded -->

The failure probability is upper bounded by:

<!-- formula-not-decoded -->

We analyze the third term. For convenience, let C ′ = 18 C CW ˜ c , we have that:

<!-- formula-not-decoded -->

We can now take C op = 2( C ′ +2 c o +6 . 75 + 1) to get that:

<!-- formula-not-decoded -->

The first item of the claim follows.

For the second item, recall that we have conditioned on the event that for every i :

<!-- formula-not-decoded -->

We have that:

and similarly:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can now easily bound the Kronecker product of the two matrices:

<!-- formula-not-decoded -->

Assuming that the events of Lemma C.2 hold, this lemma bounds the Frobenius norm of matrix M : Lemma C.3 (Existence and properties of M ) . Assuming that 12 and 13 of Lemma C.2 hold, then there exists a matrix M such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We first observe that ˆ L , by Equation 12, spans R r 2 which in particular implies that it spans the columns of matrix ˆ L K which lie in this space. This gives us the existence of a matrix M satisfying 15. We now observe that the i -th column of matrix ˆ L is equal to the (( i -1) k + i ) -th column of matrix ˆ L K . In other words we have that:

<!-- formula-not-decoded -->

where e i is the i -th standard basis vector. This implies that we can take:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

and 16 also holds. The rest of the columns of M we select them to be the minimum norm vectors such that M satisfies equation 15. In other words, for j = ( i -1) k + i for every i , we let

<!-- formula-not-decoded -->

We now analyze the Frobenius norm of matrix M , defined as above. For j for which there exists an i such that j = ( i -1) k + i , we have that:

<!-- formula-not-decoded -->

For j such that no such i exists, we have that:

<!-- formula-not-decoded -->

which in turn gives us that:

We can now bound the Frobenius norm of matrix M :

<!-- formula-not-decoded -->

where we have used that the square root is a subadditive function. Using the bounds from 12 and 13, we get the result:

<!-- formula-not-decoded -->

In Lemma C.4 we argue that the eigenvalues of matrix ( X ⊙ Z ) ⊤ ( X ⊙ Z ) are all close to 1 with high probability, we use this to show in Lemma C.1 that we can replace the matrix ( X ⊙ Y ) † ⊤ by 1 n 2 ( X ⊙ Y )( I + E )

Lemma C.4 (Eigenvalues close to 1 ) . Let X,Z ∈ R n × k be random matrices with i.i.d standard Gaussian random variables as entries, then with probability at least 1 -1 k , we have that, for every i :

<!-- formula-not-decoded -->

where C is an absolute constant

Proof. We will analyze the diagonal and non diagonal entries of the matrix 1 n 2 ( X ⊙ Z ) ⊤ ( X ⊙ Z ) and will show that the diagonal entries are concentrated very close to 1 while the off-diagonal entries are concentrated very close to 0 . We will then use Gersgorin disk theorem (Theorem 6.1.1 in [HJ12]) to get the result. For convenience, we let W = 1 n 2 ( X ⊙ Z ) ⊤ ( X ⊙ Z ) . For the off-diagonal entries we observe that:

<!-- formula-not-decoded -->

For the inner product ⟨ x i , x j ⟩ we have:

<!-- formula-not-decoded -->

By Lemma 2.2.7 in [Ver18] each summand is a subexponential random variable with subexponential norm:

<!-- formula-not-decoded -->

where K is the sub-Gaussian norm of the standard Gaussian. Using Bernstein's inequality, Theorem 2.8.1 in [Ver18], (the inner product is a sum of sub-exponential random variables) we get that:

<!-- formula-not-decoded -->

for some absolute constant c . Setting t = √ C 1 n log( k ) , for C 1 being a large enough constant we have that with probability at least 1 -1 k 3 :

<!-- formula-not-decoded -->

Similarly, we have that, with probability at least 1 -1 k 3 :

<!-- formula-not-decoded -->

For the diagonal entries of the matrix M we have that:

<!-- formula-not-decoded -->

We write ∥ x i ∥ 2 = ∑ n l =1 x 2 i ( l ) and use the fact that there exists an absolute constant ˜ C such that x i ( l ) 2 -1 is subexponential random variable with subexponential norm:

<!-- formula-not-decoded -->

We apply Bernstein's inequality again, to get that:

<!-- formula-not-decoded -->

We set t = √ C 2 n log( k ) for large enough constant C 2 , to get that with probability at least 1 -1 k 3 , we have that:

<!-- formula-not-decoded -->

Similarly, with probability at least 1 -1 k 3 , we have that:

<!-- formula-not-decoded -->

We get that, by the union bound, with probability at least 1 -1 k equations 18, 19, 20 and 21 hold for every i, j . Setting C = max( C 1 , C 2 ) , with probability at least 1 -1 k , for the off-diagonal entries of matrix ( Z ⊙ X ) ⊤ ( Z ⊙ X ) we get that:

<!-- formula-not-decoded -->

And for all diagonal entries, assuming that √ C log( k ) n ≤ 1 , we get that:

<!-- formula-not-decoded -->

We now apply Gershgorin disc theorem (Theorem 6.1.1 in [HJ12]) and the the fact that 1 n 2 ( Z ⊙ X ) ⊤ ( Z ⊙ X ) is symmetric and therefore all its eigenvalues are real to get that, for every i :

̸

<!-- formula-not-decoded -->

where from the first to the second line we use the triangle inequality.

In Lemma C.5 we give bounds upper and lower bounds for the inner products ⟨ c j , z i ⟩ , needed to bound the least singular value of matrix ˆ L and the Frobenius norm of matrix ˆ L K .

Lemma C.5. With probability at least 1 -2 k over the randomness in Z , we have that for every i and for every j :

<!-- formula-not-decoded -->

Proof. For fixed i, j , the inner product between c j and z i has distribution ⟨ c j , z i ⟩ ∼ N (0 , ∥ c j ∥ 2 ) . For the lower bound, we have that:

<!-- formula-not-decoded -->

where we have used that 1 ∥ c j ∥ ⟨ c j , z i ⟩ ∼ N (0 , 1) and that the density of the standard Gaussian is upper bounded by 1 √ 2 π . We now use the union bound to get that:

<!-- formula-not-decoded -->

where we have used that for every j , ∥ c j ∥ ≥ σ r ( C ) . For the upper bound we again fix i, j and use that 1 ∥ c j ∥ ⟨ c j , x i ⟩ ∼ N (0 , 1) . Using Proposition 2.1.2 in [Ver18], we have that:

<!-- formula-not-decoded -->

Letting t = √ 2 log( k 3 ) ≥ 1 , we get that:

<!-- formula-not-decoded -->

By the union bound and the fact that for every j , ∥ c j ∥ ≤ σ 1 ( C ) :

<!-- formula-not-decoded -->

Using again the union bound we have that with probability at least 1 -2 k for every i, j :

<!-- formula-not-decoded -->

Lemma C.6 (Bound on columns) . Assume that for every i, j , we have that |⟨ c j , z i ⟩| ≤ √ 6 log( k ) σ 1 ( C ) , then with probability at least 1 -exp( -4 r ) we have that for every i :

<!-- formula-not-decoded -->

where ˜ c is a large enough constant.

Proof. Fix an i and recall that by Lemma A.2:

<!-- formula-not-decoded -->

We focus on the norm of D z i A ⊤ x i , we have that:

<!-- formula-not-decoded -->

where we have used that Π A A = A and that any projection matrix is symmetric. We have that:

<!-- formula-not-decoded -->

where from line 2 to line 3 we have used that the operator norm is submultiplicative. From line 3 to line 4 that the operator norm of a diagonal matrix is equal to the largest entry in absolute value and from line 4 to line 5 our assumption on |⟨ c j , z i ⟩| . Similarly, we get that:

<!-- formula-not-decoded -->

We now use that ∥ Π A x i ∥ has the same distribution as that of a norm of an r dimensional random vector with i.i.d. standard Gaussian entries, by Theorem 3.1.1 in [Ver18], we get that:

<!-- formula-not-decoded -->

Applying this bound with t = ( √ ˜ c/ 2 -1) √ r (we will specify ˜ c later) as well as the union bound, we get that:

<!-- formula-not-decoded -->

We take ˜ c to be a large enough constant so that 4 k exp( -( √ ˜ c/ 2 -1) 2 r ) ≤ exp( -4 r ) . The definition of k in Lemma C.2 depends on ˜ c , because C op = 2(18 C CW · ˜ c +2 c 0 +7 . 75) , we can nevertheless select ˜ c large enough so that, for every r :

<!-- formula-not-decoded -->

We conclude that, with probability at least 1 -exp( -4 r ) , for every i :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The claim follows.

Theorem C.7. Let f ( x ) = f ( x 1 , x 2 , . . . , x n ) be a degree d polynomial such that Var [ f ] = 1 when x is a standard n -dimensional Gaussian vector, then for every t ∈ R and for every ε &gt; 0 , we have that:

<!-- formula-not-decoded -->

Claim C.8. Let A,B ∈ R n × k , then:

we now use that:

<!-- formula-not-decoded -->

Proof. We first observe that the columns of A ⊙ B are a subset of the columns of the matrix A ⊗ B , meaning that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem B.2 We analyze the least singular value of matrix ˆ X ⊙ ˆ Y . Using Equation 3 we can rewrite the matrix ˆ X ⊙ ˆ Y as:

<!-- formula-not-decoded -->

We now have that:

<!-- formula-not-decoded -->

,

We can, without loss of generality, using our assumption on the condition numbers of A , B and C assume that for the least singular values of A , B and C , we have that σ r ( A ) ≥ 1 poly( r ) , σ r ( B ) ≥ 1 poly( r ) and σ r ( C ) ≥ 1 poly( r ) (otherwise we can rescale the tensor T so that this holds):

<!-- formula-not-decoded -->

It therefore suffices to analyze the least singular value of matrix:

<!-- formula-not-decoded -->

By Lemma C.1, we have that with probability at least 1 -2 k :

<!-- formula-not-decoded -->

where ∥ E 1 ∥ , ∥ E 2 ∥ ≤ 6 √ C log( k ) n +2 C k n log( k ) ( C is an absolute constant). We now have that:

<!-- formula-not-decoded -->

where we have used L to denote:

<!-- formula-not-decoded -->

we therefore, have that:

<!-- formula-not-decoded -->

Hence, it suffices, in order to prove the claim, to bound the least singular value of matrix L by 1 poly( k ) . We first analyze the matrix:

<!-- formula-not-decoded -->

6 By Lemma C.2 we have that with probability at least 1 -2 k -exp( -4 r ) -( 1 k ) r 2 :

<!-- formula-not-decoded -->

and that for matrix:

<!-- formula-not-decoded -->

Its Frobenius norm is bounded:

<!-- formula-not-decoded -->

6 ˆ L would be equal to L if E 1 = E 2 = 0

By Lemma C.3, assuming that equations 22 and 23 hold, there exists a matrix M such that 15, 16 and 17 hold. By the union bound, with probability at least 1 -4 k -exp( -4 r ) -( 1 k ) r 2 = 1 -o (1) the events of Lemma C.1 and C.2 both hold. We now have that, using Equation 3:

<!-- formula-not-decoded -->

where we have used E to denote the matrix I ⊙ E 2 + E 1 ⊙ I + E 1 ⊙ E 2 . By Claim C.8 and assuming that n is large enough compared to k , it follows that:

<!-- formula-not-decoded -->

We can now use matrix M :

This concludes the proof.

<!-- formula-not-decoded -->

where we have used that M is such that M ( I ⊙ I ) = I . We will use this expression to analyze the least singular value of matrix L . We use the variational characterization of singular values:

<!-- formula-not-decoded -->

We have the bound on the least singular value of ˆ L , we only have to analyze the least singular value of matrix I + ME . We argue by showing that ME has small spectral norm. We have that:

<!-- formula-not-decoded -->

Assuming n γ 0 ≥ k for a small enough constant γ 0 , we get that:

<!-- formula-not-decoded -->

Using the variational characterization of the singular values, we have that:

<!-- formula-not-decoded -->

## C.2 Robust Analysis

Recall that the tensor T has a decomposition of rank r given by factor matrices A,B,C ∈ R n × r that approximates T i.e., T = ∑ r i =1 a i ⊗ b i ⊗ c i + Err with OPT = ∥ T -∑ r i =1 a i ⊗ b i ⊗ c i ∥ 2 F = ∥ Err ∥ 2 F .

The following lemma relates the objective value to singular values of different matrices related to the updates.

Lemma C.9. [Objective value in the second iteration] Suppose X (1) , Y (1) be the iterates of Algorithm 1 after the updates of the first iteration. Let ˜ Φ = X (1) ⊙ Y (1) and Φ = ̂ X ⊙ ̂ Y where ̂ X, ̂ Y are the updates after the first iteration when there is no error. Then we have that the loss objective value in the second iteration is at most

<!-- formula-not-decoded -->

Proof. We prove this statement using the above lemmas, and using Davis-Kahan theorem for perturbations of top singular spaces.

To bound the objective value in the second iteration as in (24), we use the characterization of least squares value being the squared perpendicular distance of the target vector from the span of the columns i.e., if Π ⊥ ˜ Φ is the projection matrix onto the subspace orthogonal to the column span of ˜ Φ = X (1) ⊙ Y (1) , then

<!-- formula-not-decoded -->

where E 3 ∈ R n 2 × n is the flattening of the tensor Err . Let Π Φ be the projection matrix on to the span of the columns of ̂ X ⊙ ̂ Y , and Π ⊥ Φ be the projection matrix for the subspace orthogonal to it.

<!-- formula-not-decoded -->

where we have used that Π ⊥ Φ ( A ⊙ B ) C ⊤ = 0 , since Φ = ̂ X ⊙ ̂ Y contains A ⊗ B w.h.p. from the previous non-robust analysis. Furthermore, since the top r 2 singular values of M are separated from the least singular value of ˜ Φ corresponding to Π ⊥ ˜ Φ (i.e., 0 , since this corresponds to the nullspace of ˜ Φ ), we have by the Davis-Kahan theorem (see Fact A.5)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some constant

<!-- formula-not-decoded -->

The following claims bound the different terms in (24) Lemma C.11.

Lemma C.10. [Projections onto the column space of Y ⊙ Z ] Let Q ∈ R n 2 × r be an arbitrary matrix and Y, Z ∈ R n × k be random matrices with i.i.d N (0 , 1) entries. There exists a universal constant c &gt; 0 such that with probability at least 1 -o (1)

<!-- formula-not-decoded -->

Proof. From Lemma C.1 we have that

<!-- formula-not-decoded -->

for our choice of parameters. Hence, it will suffice to upper bound ∥ Q ⊤ ( Y ⊙ Z ) ∥ F .

Each of the k columns of Y ⊙ Z is an i.i.d. random vector distributed identically to y ⊗ z where y, z ∼ N (0 , I n × n ) . Consider a fixed j ∈ [ r ] , i ∈ [ k ] . For any t ≥ 1 , from concentration of quadratic multivariate polynomials due to Hanson-Wright inequality [see e.g., Ver18], we have that the ( j, i ) th entry of Q ⊤ ( Y ⊙ Z ) can be written as

<!-- formula-not-decoded -->

for t = O (log( kr )) . The lemma follows after a union bound to get an upper bound on the magnitude of each of the kr entries of the matrix.

This in turns leads to the following claim.

Lemma C.11. [Perturbation of the spectrum with noise] Let X (1) , Y (1) be the iterates of Algorithm 1 after the updates of the first iteration. Let ˜ Φ = X (1) ⊙ Y (1) and ̂ X, ̂ Y denote the updates after the first iteration when there is no error. Then with high probability 1 -o (1) , we have for Φ = ̂ X ⊙ ̂ Y that

<!-- formula-not-decoded -->

A similar claim holds for Y (1) ⊙ Z (1) and Z (1) ⊙ X (1) as well.

In the above expression √ OPT ≪∥ A ∥∥ B ∥∥ C ∥ F .

Proof. Recall that T = ∑ r i =1 a i ⊗ b i ⊗ c i + Err , where ∥ Err ∥ F = √ OPT. Moreover, for the flattening E 1 , E 2 ∈ R n × n 2 along the first and second modes of Err , we have with probability 1 -o (1) from Lemma C.10 that that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore,

<!-- formula-not-decoded -->

by Claim A.3. We have a similar bound for ∥ ̂ Y ∥ F . Hence, we have

<!-- formula-not-decoded -->

We now finish the proof of Theorem B.3.

Proof of Theorem B.3 Our analysis will work with any algorithm that computes the pseudoinverse solution. For any M , we find ̂ M such that

<!-- formula-not-decoded -->

Without loss of generality, we can assume that ∥ T ∥ F = 1 (for scaling), and ∥ A ∥ F = ∥ B ∥ F = ∥ C ∥ F (since we can redistribute the mass among the factors of any decomposition arbitrarily). Let us denote by OPT = ∥ Err ∥ 2 F , and κ = max { κ ( A ) , κ ( B ) , κ ( C ) } . By Claim A.4, we have that 1 ≤ ∥ A ∥ 3 F ≤ rκ 2 ≤ r 1+2 c 0 . For our purposes, we can think of OPT ≥ ε , since the guarantee and proof works up any upper bound on ∥ Err ∥ 2 F . The algorithm in every iteration solves a least squares problem up to precision ε in time polynomial in n, d, log(1 /ε ) . We can ignore this ε in this robust analysis, since it is dominated by the error terms in the tensor, and the intermediate steps. Also note that OPT ≤ 1 ; in fact, it will be useful to think of OPT &lt; 1 / poly( k, r ) , since otherwise the trivial bound suffices.

We can bound the objective value in the second iteration using Lemma C.9 and combine it with Lemma C.11 to get

<!-- formula-not-decoded -->

by using the bound of σ r 2 (Φ) ≥ ( n 4 poly( k, r )) -1 from Theorem B.2, and the bound on κ in the assumptions on Theorem B.3. This concludes the proof.

## D Experimental Evaluation

Our theoretical results guarantee convergence of Algorithm 1 when the overparameterization k = O ( r 2 ) . In our experiments we investigate whether this overparameterization factor is also observed in practice, and what the leading constant in the dependence is. The second question is how much overparameterization is needed i.e., does ALS require k = Ω( r 2 ) to succeed? Our experiments suggests that both these questions are true.

Algorithm 1 is a non-standard version of the ALS algorithm because in each iteration, it performs the updates to each mode in parallel. That is, X ( t +1) , Y ( t +1) , Z ( t +1) are all a function of X ( t ) , Y ( t ) , Z ( t ) . This is in contrast to the standard version of ALS which updates the modes sequentially. That is, X ( t +1) will depend on Y ( t ) , Z ( t ) as in the parallel version, but then Y ( t +1) will depend on X ( t +1) , Z ( t ) , and Z ( t +1) will depend on X ( t +1) , Y ( t +1) . Our theoretical results focus on the parallel update version (Algorithm 1) because it is easier to analyze. In our experiments, we evaluate both the parallel-update and the standard sequential versions of ALS, to see what the effect of overparameterization is.

## D.1 Experimental setup

To evaluate the parallel-update version of ALS (Algorithm 1), we implemented a non-optimized version using the scipy least squares solver. For each n, r, k that we analyze, we run 20 trials. For each trial we generate 3 random n × r factor matrices (each entry is an independent Gaussian) to make up our ground truth tensor. We initialize the factors of our model to be fully random n × k matrices. For all n = 200 and n = 500 we set the maximum number of iterations to be 20, due to computational constraints.

To evaluate standard ALS, we used the parafac method from the TensorLy library [KPAP19], which provides an optimized version of the standard (sequential) ALS method. As for parallel-update ALS, for each n, r, k that we analyze we run 20 trials, and for each trial we generate 3 random (Gaussian) n × r factor matrices to make up our ground truth tensor. We then call parafac on this tensor with random initialization. 7 For n = 500 we set the maximum number of iterations to be 100, and for n = 1000 we set the maximum number of iterations to be 20, due to computational constraints.

We provide python code to run both experimental setups as part of the supplementary material.

## D.2 Parallel-updated ALS discussion

k = r 2 suffices. Our theoretical results guarantee that parallel-update ALS (Algorithm 1) should converge in O (1) steps as long as k = Ω( r 2 ) . Our experiments validate that this holds for k = r 2 , with no leading constant. In Figure 1, we plot the errors of running Algorithm 1 for n = 200 , r ∈ { 8 , 11 , 14 , 17 , 20 } , and various values of k that depend on the setting of r . We see that consistently across all settings of r , Algorithm 1 consistently fails to converge for any value of k &lt; r 2 and consistently converges for k ≥ r 2 .

Our theoretical results guarantee that, once n is sufficiently larger than k , the overparameterized rank necessary for parallel-update ALS (Algorithm 1) to succeed has no dependence on n . In Figure 2 we plot the errors of running Algorithm 1 for r = 8 , and n ∈ { 200 , 500 } . We observe that there is indeed no apparent difference in the results.

## D.3 Standard ALS discussion

k ≤ r 2 suffices for standard ALS. While our theoretical results apply to the parallel-update version of ALS, we observe that overparameterization k = r 2 seems to suffice to ensure convergence for the standard sequential version of ALS as well. We run this experiment for n = 500 and various values of r and k , and the results can be found in Figure 3, Figure 4, Figure 5, Figure 6, Figure 7, Figure 8, Figure 9. In all of these experiments, we see that ALS starts to converge for values of k that are less than r 2 . Our theoretical result for the parallel-update ALS guarantees that the overparameterization necessary to ensure convergence should have no dependence on n . To evaluate whether this is true for

7 We use fully random initialization as opposed to the default SVD initialization, which deviates significantly from what we analyze in this work.

standard ALS, we provide Figure 9 and Figure 10, which both evaluate r = 20 and the same values of k for two different values of n ( n = 500 and n = 1000 ). We observe that the different choices of n do not appear to have any significant impact on the error of standard ALS as a function of the overparameterization.

Comparison to parallel-update ALS. We see that in comparison to the parallel-update version of ALS, the standard version has a more graceful degradation of error as a function of k . While we do not have a theoretical result that proves that standard ALS performs only better than the parallel-update version we analyze, it does appear in our experiments that this is the case. Even though standard ALS converges for smaller values of k than the parallel-update ALS, we note that many of our experiments, including Figure 7, Figure 8, Figure 9, and Figure 10, seem to display that standard ALS experiences instability at values of k very close to r 2 , that it does not experience for other nearby values of k . We view this as an interesting phenomenon to investigate in future work.

Necessary overparameterization. Our theoretical results guarantee that parallel-update ALS converges for k = Ω( r 2 ) . Our experimental results suggest that parallel-update ALS converges exactly when k ≥ r 2 (with no leading constant). Our experiments also suggest that standard (sequential) ALS converges for values of k &lt; r 2 . However, we observe that even though the input tensors in our experiment are chosen randomly from a nicely-behaved distribution , standard ALS still requires k significantly larger than r to converge. Our results are inconclusive as to what dependency k must have on r to ensure convergence. We view it as an exciting future direction of both theoretical and experimental work to understand the overparameterization necessary to ensure convergence of standard ALS.

## D.4 Data

<!-- image -->

Overparameterizationk

Figure 1: Results of running the parallel-update version of ALS (Algorithm 1) for n = 200 , various values of r , and various values of k that depend on r . We see that this method consistently fails to converge for k &lt; r 2 and consistently converges for k ≥ r 2 . For this experiment we run ALS for a maximum of 20 iterations per trial. For trials where the method converged, it always converged in 2 iterations, which is consistent with our theoretical result. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 11.

Figure 2: Results of running the parallel-update version of ALS (Algorithm 1) for r = 8 , two values of n , and various values of k that depend on r . We see that this method consistently fails to converge for k &lt; r 2 and consistently converges for k ≥ r 2 . For this experiment we run ALS for a maximum of 20 iterations per trial. For trials where the method converged, it always converged in 2 iterations, which is consistent with our theoretical result. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 12.

<!-- image -->

Figure 3: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 500 , r = 8 and various values of k . We observe that the error degrades gracefully as a function of k . The minimum k necessary to ensure convergence seems to be significantly larger than r = 8 , but smaller than r 2 = 64 . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 13.

<!-- image -->

Figure 4: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 500 , r = 10 and various values of k . We observe that the error degrades gracefully as a function of k . The minimum k necessary to ensure convergence seems to be significantly larger than r = 10 , but smaller than r 2 = 100 . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 14.

<!-- image -->

Figure 5: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 500 , r = 12 and various values of k . We observe that the error degrades gracefully as a function of k . The minimum k necessary to ensure convergence seems to be significantly larger than r = 12 , but smaller than r 2 = 144 . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 15.

<!-- image -->

Figure 6: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 500 , r = 14 and various values of k . We observe that the error degrades gracefully as a function of k . The minimum k necessary to ensure convergence seems to be significantly larger than r = 14 , but smaller than r 2 = 196 . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 16.

<!-- image -->

Figure 7: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 500 , r = 16 and various values of k . We observe that the error degrades gracefully as a function of k . The minimum k necessary to ensure convergence seems to be significantly larger than r = 16 , but smaller than r 2 = 256 . We also observe that standard ALS seems to experience some instability for values of k very close to r 2 = 256 . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 13.

<!-- image -->

Figure 8: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 500 , r = 18 and various values of k . We observe that the error degrades gracefully as a function of k . The minimum k necessary to ensure convergence seems to be significantly larger than r = 18 , but smaller than r 2 = 324 . We also observe that standard ALS seems to experience some instability for values of k very close to r 2 = 324 . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 18.

<!-- image -->

Figure 9: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 500 , r = 20 and various values of k . We observe that the error degrades gracefully as a function of k . The minimum k necessary to ensure convergence seems to be significantly larger than r = 20 , but smaller than r 2 = 400 . We also observe that standard ALS seems to experience some instability for values of k very close to r 2 = 400 . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 19.

<!-- image -->

Figure 10: Results of running standard ALS ( parafac by TensorLy [KPAP19]) for n = 1000 , r = 20 and various values of k . We observe the results of this experiment are very similar to Figure 9, suggesting that convergence of standard ALS is a function of k and not n . For this experiment we run ALS for a maximum of 100 iterations per trial. The reported values are aggregated over 20 independent trials, with error bars corresponding to one standard deviation. The data for this plot can be found in Figure 19.

<!-- image -->

Figure 11: Data used to generate Figure 1. For these experiments n = 200 , and the maximum number of iterations of ALS is 20. The reported values are aggregated over 20 independent trials.

|   r | k         |     mean |   std. dev |
|-----|-----------|----------|------------|
|   8 | r 2 - 2   | 1.35     |   1.07     |
|   8 | 2 - 1     | 1.3      |   0.855    |
|   8 | r r 2     | 5.99e-13 |   9.46e-13 |
|   8 | r 2 +1    | 3.21e-14 |   1.85e-14 |
|   8 | r 2 +2    | 2.86e-14 |   1.89e-14 |
|   8 | r 2 + r   | 4.92e-14 |   1.66e-14 |
|   8 | r 2 +2 r  | 3.09e-14 |   4.6e-15  |
|  11 | r 2 - 2   | 1.18     |   0.813    |
|  11 | r 2 - 1   | 1.21     |   0.511    |
|  11 | r 2       | 3.38e-13 |   8.14e-13 |
|  11 | r 2 +1    | 4.66e-14 |   4.06e-14 |
|  11 | r 2 +2    | 2.81e-14 |   1.21e-14 |
|  11 | r 2 + r   | 1.25e-14 |   2.1e-15  |
|  11 | r 2 +2 r  | 9.46e-15 |   1.23e-15 |
|  14 | r 2 - 2   | 1.03     |   0.659    |
|  14 | 2 - 1     | 1.73     |   1.69     |
|  14 | r r 2     | 9.44e-13 |   3.14e-12 |
|  14 | r 2 +1    | 6.93e-14 |   1.03e-13 |
|  14 | r 2 +2    | 3.96e-14 |   1.49e-14 |
|  14 | r 2 + r   | 1.54e-14 |   1.53e-15 |
|  14 | r 2 +2 r  | 3.67e-14 |   4.32e-15 |
|  17 | r 2 - 2   | 1.22     |   0.628    |
|  17 | r 2 - 1   | 1.18     |   0.731    |
|  17 | r 2       | 2.55e-13 |   5.84e-13 |
|  17 | r 2 +1    | 5.83e-14 |   3.82e-14 |
|  17 | 2 +2      | 4.44e-14 |   1.36e-14 |
|  17 | r r 2 + r | 1.56e-14 |   1.29e-15 |
|  17 | r 2 +2 r  | 9.45e-15 |   6.06e-16 |
|  20 | r 2 - 2   | 1.02     |   0.376    |
|  20 | r 2 - 1   | 1.43     |   1.07     |
|  20 | r 2       | 9.33e-13 |   1.04e-12 |
|  20 | r 2 +1    | 8.34e-14 |   3.87e-14 |
|  20 | r 2 +2    | 7.58e-14 |   2.77e-14 |
|  20 | r 2 + r   | 1.49e-14 |   8.28e-16 |
|  20 | r 2 +2 r  | 3.83e-14 |   3.45e-15 |

|   n |   r | k        |     mean |   std. dev |
|-----|-----|----------|----------|------------|
| 200 |   8 | r 2 - 2  | 1.35     |   1.07     |
| 200 |   8 | r 2 - 1  | 1.3      |   0.855    |
| 200 |   8 | r 2      | 5.99e-13 |   9.46e-13 |
| 200 |   8 | r 2 +1   | 3.21e-14 |   1.85e-14 |
| 200 |   8 | r 2 +2   | 2.86e-14 |   1.89e-14 |
| 200 |   8 | r 2 + r  | 4.92e-14 |   1.66e-14 |
| 200 |   8 | r 2 +2 r | 3.09e-14 |   4.6e-15  |
| 500 |   8 | r 2 - 2  | 1.5      |   1.31     |
| 500 |   8 | r 2 - 1  | 1.81     |   1.88     |
| 500 |   8 | r 2      | 2.67e-12 |   1.11e-11 |
| 500 |   8 | r 2 +1   | 3.36e-14 |   1.91e-14 |
| 500 |   8 | r 2 +2   | 2.72e-14 |   1.25e-14 |
| 500 |   8 | r 2 + r  | 1.11e-14 |   2.17e-15 |
| 500 |   8 | r 2 +2 r | 6.7e-15  |   1.29e-15 |

Figure 12: Data used to generate Figure 2. For these experiments the maximum number of iterations of ALS is 20. The reported values are aggregated over 20 independent trials.

Figure 13: Data used to generate Figure 3. For this experiment n = 500 , and the maximum number of iterations of ALS was 100 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|   8 |   8 | 0.279915 |    0.111048 |
|   8 |  16 | 0.106255 |    0.071116 |
|   8 |  32 | 0.0293   |    0.032489 |
|   8 |  48 | 0.00011  |    0.000177 |
|   8 |  56 | 0        |    0        |
|   8 |  60 | 0        |    0        |
|   8 |  63 | 0        |    0        |
|   8 |  64 | 0        |    0        |
|   8 |  65 | 0        |    0        |
|   8 |  72 | 0        |    0        |
|   8 |  80 | 5e-06    |    2.2e-05  |

Figure 14: Data used to generate Figure 4. For this experiment n = 500 , and the maximum number of iterations of ALS was 100 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|  10 |  10 | 0.237    |    0.100775 |
|  10 |  20 | 0.105275 |    0.073401 |
|  10 |  40 | 0.03214  |    0.025882 |
|  10 |  50 | 0.01725  |    0.014379 |
|  10 |  60 | 0.009715 |    0.01181  |
|  10 |  80 | 5e-05    |    6.1e-05  |
|  10 |  90 | 0        |    0        |
|  10 |  95 | 0        |    0        |
|  10 |  99 | 3.5e-05  |    0.000157 |
|  10 | 100 | 4.6e-05  |    0.000113 |

Figure 15: Data used to generate Figure 5. For this experiment n = 500 , and the maximum number of iterations of ALS was 100 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|  12 |  12 | 0.241995 |    0.075478 |
|  12 |  24 | 0.10385  |    0.057524 |
|  12 |  48 | 0.052115 |    0.037644 |
|  12 |  72 | 0.020825 |    0.015514 |
|  12 |  96 | 0.006375 |    0.006497 |
|  12 | 120 | 1.5e-05  |    3.7e-05  |
|  12 | 132 | 0        |    0        |
|  12 | 138 | 0        |    0        |
|  12 | 143 | 0        |    0        |
|  12 | 144 | 0        |    0        |
|  12 | 145 | 0        |    0        |
|  12 | 156 | 2e-05    |    8.9e-05  |
|  12 | 168 | 0        |    0        |

Figure 16: Data used to generate Figure 6. For this experiment n = 500 , and the maximum number of iterations of ALS was 100 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|  14 |  14 | 0.1976   |    0.08427  |
|  14 |  28 | 0.081565 |    0.065547 |
|  14 |  56 | 0.054265 |    0.03741  |
|  14 |  86 | 0.037515 |    0.023547 |
|  14 |  98 | 0.02151  |    0.015892 |
|  14 | 112 | 0.01377  |    0.011933 |
|  14 | 140 | 0.001525 |    0.002253 |
|  14 | 168 | 1e-05    |    3.1e-05  |
|  14 | 182 | 0        |    0        |
|  14 | 189 | 0        |    0        |
|  14 | 195 | 0        |    0        |
|  14 | 196 | 5e-06    |    2.2e-05  |
|  14 | 197 | 1e-05    |    3.1e-05  |
|  14 | 210 | 0        |    0        |
|  14 | 224 | 0.00027  |    0.001207 |

Figure 17: Data used to generate Figure 7. For this experiment n = 500 , and the maximum number of iterations of ALS was 100 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|  16 |  16 | 0.20394  |    0.078098 |
|  16 |  32 | 0.07173  |    0.045191 |
|  16 |  64 | 0.065095 |    0.039075 |
|  16 |  96 | 0.04665  |    0.02052  |
|  16 | 128 | 0.023405 |    0.015258 |
|  16 | 160 | 0.008025 |    0.007481 |
|  16 | 192 | 0.000725 |    0.001254 |
|  16 | 224 | 5e-06    |    2.2e-05  |
|  16 | 240 | 0        |    0        |
|  16 | 248 | 0        |    0        |
|  16 | 255 | 0.00012  |    0.000537 |
|  16 | 256 | 3.5e-05  |    7.5e-05  |
|  16 | 257 | 0.00272  |    0.011766 |
|  16 | 272 | 5e-06    |    2.2e-05  |
|  16 | 288 | 0.000575 |    0.001772 |

Figure 18: Data used to generate Figure 8. For this experiment n = 500 , and the maximum number of iterations of ALS was 100 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|  18 |  18 | 0.201895 |    0.075775 |
|  18 |  36 | 0.04922  |    0.060002 |
|  18 |  72 | 0.058525 |    0.041778 |
|  18 | 108 | 0.045035 |    0.020357 |
|  18 | 144 | 0.02728  |    0.016545 |
|  18 | 162 | 0.02057  |    0.011974 |
|  18 | 180 | 0.01624  |    0.009968 |
|  18 | 216 | 0.004095 |    0.004536 |
|  18 | 252 | 0.000635 |    0.001181 |
|  18 | 288 | 5e-06    |    2.2e-05  |
|  18 | 306 | 0        |    0        |
|  18 | 315 | 0        |    0        |
|  18 | 323 | 0.000965 |    0.004269 |
|  18 | 324 | 3.5e-05  |    6.7e-05  |
|  18 | 325 | 0.00613  |    0.027297 |
|  18 | 342 | 5e-06    |    2.2e-05  |
|  18 | 360 | 3e-05    |    0.000134 |

Figure 19: Data used to generate Figure 9. For this experiment n = 500 , and the maximum number of iterations of ALS was 100 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|  20 |  20 | 0.222785 |    0.087655 |
|  20 |  40 | 0.103795 |    0.062048 |
|  20 |  80 | 0.085645 |    0.040937 |
|  20 | 120 | 0.06505  |    0.023917 |
|  20 | 160 | 0.037505 |    0.014094 |
|  20 | 200 | 0.027705 |    0.013622 |
|  20 | 220 | 0.02015  |    0.009954 |
|  20 | 240 | 0.0129   |    0.007192 |
|  20 | 260 | 0.00727  |    0.004    |
|  20 | 280 | 0.005775 |    0.003874 |
|  20 | 300 | 0.001783 |    0.001685 |
|  20 | 340 | 0.000275 |    0.000197 |
|  20 | 360 | 4.5e-05  |    5.1e-05  |
|  20 | 380 | 0        |    0        |
|  20 | 390 | 0        |    0        |
|  20 | 399 | 6.5e-05  |    0.000208 |
|  20 | 400 | 0.005205 |    0.023042 |
|  20 | 401 | 0.000145 |    0.000417 |
|  20 | 420 | 0        |    0        |
|  20 | 440 | 0.00019  |    0.00085  |

Figure 20: Data used to generate Figure 10. For this experiment n = 1000 , and the maximum number of iterations of ALS was 20 . The reported values are aggregated over 20 independent trials.

|   r |   k |     mean |   std. dev. |
|-----|-----|----------|-------------|
|  20 |  20 | 0.239725 |    0.102431 |
|  20 |  40 | 0.13363  |    0.059243 |
|  20 |  80 | 0.096775 |    0.033897 |
|  20 | 120 | 0.0835   |    0.029776 |
|  20 | 160 | 0.044465 |    0.01832  |
|  20 | 200 | 0.02991  |    0.01427  |
|  20 | 220 | 0.024735 |    0.011906 |
|  20 | 240 | 0.01598  |    0.00924  |
|  20 | 260 | 0.011645 |    0.008376 |
|  20 | 280 | 0.00604  |    0.004469 |
|  20 | 300 | 0.002445 |    0.002042 |
|  20 | 320 | 0.00157  |    0.001794 |
|  20 | 340 | 0.00025  |    0.000173 |
|  20 | 360 | 5e-05    |    5.1e-05  |
|  20 | 380 | 0        |    0        |
|  20 | 390 | 0        |    0        |
|  20 | 399 | 7.5e-05  |    0.000251 |
|  20 | 400 | 0.001155 |    0.004456 |
|  20 | 401 | 0.0001   |    0.000296 |
|  20 | 420 | 2e-05    |    8.9e-05  |
|  20 | 440 | 0        |    0        |

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide rigorous proofs of all the statements made in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [NA]

Justification: We clearly describe the assumptions under which our conclusions hold in all theorem statements.

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

Justification: We provide the assumptions under which our conclusions hold and provide rigorous proofs.

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

Justification: We clearly describe how the experiments can be reproduced.

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

Justification: We provide python code to run both experimental setups as part of the supplementary material.

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

Justification: We don't have training/test experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The error bars clearly describe the statistical significance using standard visualization techniques

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

Justification: We used a regular laptop.

Guidelines:

- The answer NA means that the paper does not include experiments.

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work has no foreseeable societal impact.

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

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly credit original owners of assets.

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

Justification: We don't releas new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

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
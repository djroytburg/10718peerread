## How to Learn a Star: Binary Classification with Starshaped Polyhedral Sets

## Marie-Charlotte Brandenburg

Ruhr Universität Bochum Universitätsstr. 150, 44801 Bochum, Germany marie-charlotte.brandenburg@rub.de

## Abstract

We consider binary classification restricted to a class of continuous piecewise linear functions whose decision boundaries are (possibly nonconvex) starshaped polyhedral sets, supported on a fixed polyhedral simplicial fan. We investigate the expressivity of these function classes and describe the combinatorial and geometric structure of the loss landscape, most prominently the sublevel sets, for two loss-functions: the 0/1-loss (discrete loss) and a log-likelihood loss function. In particular, we give explicit bounds on the VC dimension of this model, and concretely describe the sublevel sets of the discrete loss as chambers in a hyperplane arrangement. For the log-likelihood loss, we give sufficient conditions for the optimum to be unique, and describe the geometry of the optimum when varying the rate parameter of the underlying exponential probability distribution.

## 1 Introduction

We study the problem of binary classification from a geometric and combinatorial perspective. Given a finite labeled data-set and a prescribed loss-function, we focus on characterizing the structure of those parameters that yield perfect classification - namely, the set of global minimizers of the loss function. More generally, we investigate the geometry and combinatorics of the entire loss landscape in parameter space. Understanding the geometry is central to analyze the behavior of learning algorithms, as, for example, the arrangement of critical points and the connectivity of minimizers influence optimization efficiency and generalization. Combinatorial structures, such as polyhedral decompositions, provide insights into how parameter spaces partition into regions of similar behavior. Naturally, these subdivisions interact with the subdivision into sets of classifiers which induce the same classification on the data, and is therefore intimately related to the VC dimension (the Vapnik-Chervonenkis dimension) of binary classifiers [Vapnik and Chervonenkis, 1971].

In order to make rigorous statements, we fix the function class used for classification to be a class which is suitable for the specific learning task. Fixing a large set of classifiers can lead to practical difficulties due to the complexity of the space of allowed classifiers, while a small set of classifiers may not be able to capture the nature of the underlying problem. A natural function class consists of those functions whose decision boundary - the geometric object separating the two classes - is the boundary of a convex polyhedron. Such function classes have been previously considered, for example, in Astorino and Gaudioso [2002], Manwani and Sastry [2010] and Kantchelian et al. [2014], where the optimal separating convex polyhedron is found through iteratively solving LPs, minimizing a logistic loss function and finding a large margin convex separator, respectively.

While polyhedral classifiers form a well-structured function class, they are also highly restrictive. In particular, the region enclosed by the decision boundary is necessarily convex, which may not always align with the structure of the underlying classification problem. Maintaining the piecewise

## Katharina Jochemko

KTH Royal Institute of Technology 100 44 Stockholm, Sweden jochemko@kth.se

Figure 1: Two identical classifications of 3 points by different starshaped polyhedral sets, supported on the same polyhedral fan with 8 generators in R 2 .

<!-- image -->

linear nature, but allowing non-convex functions, we consider a class of piecewise linear functions whose decision boundaries are (possibly nonconvex) star-shaped polyhedral sets, supported on a fixed polyhedral fan. Fixing the polyhedral fan makes this a tractable and learnable class of functions whose space of parameters exhibits nice geometric structures as we show. Considering these function classes generalizes the approach in Cevikalp and Triggs [2017], where kites are used for solving visual object detection and multi-class discrimination.

Our family of star-shaped classifiers falls within the broader class of continuous piecewise-linear functions, and as such can be represented by a suitably structured ReLU network. However, the powerful flexibility of ReLU networks makes it challenging to enforce specific geometric properties, such as ensuring that the decision region satisfies star-convexity. Moreover, the parameter space of general ReLU neural networks admits undesirable combinatorial and geometric properties such as disconnectedness and local non-global minima even for separable data Brandenburg et al. [2024]. In contrast, by building our classifiers directly on a fixed simplicial fan, we retain the ability to model non-convex boundaries and yet maintain high control over the shape and connectivity of the decision regions.

We focus on classification with polyhedral starshaped sets with respect to two loss functions: the 0/1-loss (or discrete loss ) allows us to study the parameter space in a combinatorial fashion using polyhedral methods. Additionally, we consider a log-likelihood loss function , which is amenable to numerical optimization methods, and whose level sets carry convex-geometric structures.

## 1.1 Our contributions

In this article, we consider binary classification restricted to a class of continuous piecewise linear functions whose decision boundaries are (possibly nonconvex) starshaped polyhedral sets. More precisely, for a fixed simplicial polyhedral fan we consider the class of functions whose restrictions to each cone in the fan is linear.

Geometry of the parameter space. We initiate the geometric study of the space of parameters defining the classification by polyhedral starshaped sets, and show that binary classifications with this model correspond to chambers in the data arrangement , a hyperplane arrangement within the parameter space. Investigating the expressivity of such starshaped polyhedral classifiers, we show that the VC dimension equals the number of rays in the fan, quantifying how the number of linear regions of the classifier impacts sample complexity.

Geometry of sublevel sets. We examine the effect of the choice of the loss function explicitly on two concrete loss functions, and contrast how the geometry induced by these different losses governs optimization. We show that the sublevel sets of false positives and false negatives are starshaped sets

in the parameter space, and, in case of perfect separability, sublevel sets with respect to the discrete loss are starconvex sets. For the log-likelihood loss we show concavity, and consequently convexity holds for its superlevel sets, implying that an optimum can be found in polynomial time. We give sufficient conditions for the optimum to be unique, and describe the geometry of the optimum when varying the rate parameter of the underlying exponential probability distribution.

Extended parameter spaces. While most of our results focus on starshaped sets with fixed origin, we also consider starshaped classifiers where we allow translation of the origin, and consider the translation vector as an extra parameter. For a fixed starshaped set and varying translation vectors, we show that the discrete loss is constant on chambers in an arrangement of stars , but the sublevel sets are generally no longer starconvex. While this setup combined with the log-likelihood loss does in general not lead to a convex program, we show that the log-likelihood function is piecewise concave on the underlying data fan arrangement .

Finally, we allow to simultaneously vary the shape of the star and the position of the origin. In this case, the sublevel sets of the discrete loss are semialgebraic sets , i.e., finite unions and intersections of solutions to polynomial inequalities, and we show that they are not necessarily path connected. We also explore the expressivity of this larger family of translated starshaped classifiers and show that the VC dimension is O ( d 2 log 2 ( d ) k log 2 ( k )) if d is the dimension of the ambient space and k the number of maximal cells in the fan.

## 1.2 Limitations

Throughout the article, we assume a fixed simplicial fan as given. In practical applications, an appropriate fan suitable for the specific task needs to be chosen prior to the analysis. We emphasize that this paper is a purely theoretical contribution. The presented framework has not been tested on large-scale synthetic or real-world data, only small-scale experiments as presented in the end of this article have been conducted. Developing a systematic or heuristic approach for selecting parameters such as the simplicial fan, a translation vector or the rate parameter, is beyond the scope of this work.

## 2 Description of the model

## 2.1 Polyhedral geometry and stars

We begin by introducing essential notions from polyhedral geometry, and the class of classifiers we consider. For a thorough background on polyhedral geometry we refer the reader to [Ziegler, 2012, Chapters 1-2]. Examples and visualizations of polyhedral fans are given in Appendix A

Definition 2.1. A set S ⊆ R d is star-convex with respect to a center o ∈ S if for every s ∈ S , the line segment [ o , s ] = { µ o +(1 -µ ) s : 0 ≤ µ ≤ 1 } is contained in S . In particular, a set is convex it is star-convex with respect to every o ∈ S .

Definition 2.2. A set C ⊆ R d is a polyhedral cone if

<!-- formula-not-decoded -->

for vectors v 1 , . . . , v k ∈ R d , k &gt; 0 . The vectors { v i } 1 ≤ i ≤ k are called generators of C . We use the notation C = cone( v 1 , . . . , v k ) for a cone with these generators. If C is generated by linearly independent vectors then C is called simplicial .

Definition 2.3. A hyperplane { x ∈ R d : 〈 h , x 〉 = a } is a supporting hyperplane of the cone C if 〈 h , v 〉 ≥ a for all points v ∈ C . A subset F ⊆ C is called a (proper) face if F = C ∩ H for some supporting hyperplane H .

Definition 2.4. A collection ∆ of polyhedral cones is called a polyhedral fan if the following two conditions are both satisfied.

- (i) If C ∈ ∆ then also every face of C is in ∆ .
- (ii) If C 1 , C 2 ∈ ∆ then C 1 ∩ C 2 is a face of C 1 and C 2 .

Moreover, ∆ is called a simplicial fan if it contains only simplicial cones. It is further called complete if the union of all cones it contains is R d . A full-dimensional cone of a complete fan is a maximal cone . The collection of generators of all cones in the fan are called the generators of the fan .

Intuitively speaking, a fan is a collection of cones that fit together nicely. Examples of two well-known classes of simplicial fans, namely kite fans and Coxeter fans of type B , are given in Appendix A.1.

In the following, let ∆ always be a complete, simplicial fan with generators { v i } 1 ≤ i ≤ n . Further below we will also consider affine translates of ∆ consisting of translated cones of the form C + t where C is a cone and t ∈ R d is a fixed translation vector.

Proposition 2.5. For every vector a = ( a 1 , . . . , a n ) ∈ R n there is a unique function f ∆ a : R d → R such that f ∆ a ( v i ) = a i for 1 ≤ i ≤ n and the restriction f ∆ a | C is linear for any cone C ∈ ∆ .

Indeed, for any x ∈ R d there is a unique cone C ∈ ∆ with generators v i 1 . . . , v i k such that x = µ i 1 v i 1 + . . . + µ i k v i k ∈ C and µ i j &gt; 0 for all 1 ≤ j ≤ k . If C is a full-dimensional cone, then k = d and V C = ( v i 1 . . . v i d ) is an invertible square matrix such that V C ( µ i 1 . . . µ i d ) T = x , so V -1 C x = ( µ i 1 . . . µ i d ) T . Define µ j := 0 for j ∈ { 1 , . . . , n } \ { i 1 , . . . , i k } . We write [ x ] ∆ ∈ R n for the vector ( µ 1 , . . . , µ n ) expressing x as a positive linear combination of the generators of the cone of ∆ that it lies in. We will sometimes simply write [ x ] when the fan ∆ is clear. For exemplifying computations of [ x ] ∆ when ∆ is the kite fan or the Coxeter fan of type B, we refer to Appendix A.1. Since f ∆ a ( v i j ) = a i j for 1 ≤ j ≤ k , the linearity of f ∆ a | C implies

<!-- formula-not-decoded -->

Let X = { ( x ( i ) , y ( i ) ) } i m =1 ⊂ R d ×{ 0 , 1 } be a binary labeled dataset. Define the ( m × n ) -matrix A X to be such that the i th row is [ x ( i ) ] ∆ . Then evaluating A X a results in a vector whose i th entry is f ∆ a ( x ( i ) ) . Observe that since ∆ is simplicial, the matrix A X is sparse in the sense that there are at most d non-zero entries in every row.

We consider the task of finding a classifier c : R d →{ 0 , 1 } that predicts y ( i ) well given x ( i ) . Given a complete, simplicial fan ∆ , we consider the set of functions

<!-- formula-not-decoded -->

Each function f ∆ a , a &gt; 0 , defines a classifier c a : R d →{ 0 , 1 } by setting

<!-- formula-not-decoded -->

The classification according to c a is the vector ( c a ( x (1) ) , . . . , c a ( x ( m ) )) . By slight abuse of notation we also denote the set of all classifiers c a , a ≥ 0 by S ∆ . The 0 -class c -1 a (0) is enclosed in a starshaped set. Indeed, it is the union of simplices with vertex sets of the form { 0 , 1 a i 1 v i 1 , . . . , 1 a i k v i k } where v i 1 , . . . , v i k are the generators of a cone C of ∆ . We call this a star and denote it as star( a ) = c -1 a (0) . See Figure 1 for examples.

A data point ( x ( i ) , y ( i ) ) has a positive label if y ( i ) = 1 and a negative label if y ( i ) = 0 . A point x ( i ) is a false positive with respect to a if f ∆ a ( x ( i ) ) &gt; 1 and y ( i ) = 0 . Similarly, the data point x ( i ) is a false negative with respect to a if f ∆ a ( x ( i ) ) ≤ 1 and y ( i ) = 1 . We denote the number of false positives and false negatives by FP( a ) and FN( a ) , respectively.

## 2.2 Loss functions

In this article, we consider minimization with respect to two distinct loss functions: the 0 / 1 -loss and a log-likelihood loss function. For the 0 / 1 -loss (or discrete loss ), we seek to minimize the number of misclassifications, counting both the false positives and false negatives, i.e.

<!-- formula-not-decoded -->

For the log-likelihood loss function , let y be the random variable giving the class label of the random vector x ∈ R d . Weapproximate the probability that x is not in the star with the cumulative distribution function of the exponential probability distribution,

<!-- formula-not-decoded -->

where λ &gt; 0 is the rate parameter of the exponential distribution. The task is to find a ∈ R &gt; 0 that maximizes the log-likelihood function

<!-- formula-not-decoded -->

We observe that P ( y = 1 | x , ∆ , a ) approaches 0 when x approaches 0 , i.e. is in the set star( a ) defined by c a , and 1 when x approaches infinity, i.e. is outside the star.

Note that, in principle, one can choose to approximate P ( y = 1 | x , ∆ , a ) with any function F ( f ∆ a ( x )) were F is a cumulative distributive function on R ≥ 0 . The choice above will be justified by its desirable properties as shown in the following sections.

## 3 Geometry of the parameter space

In this section we study the set of optimal parameters a ∈ R n &gt; 0 as well as the sublevel sets of the 0 / 1 -loss (1) and the loss function given by the log-likelihood function (2) from a combinatorial and geometric point of view. We begin by analyzing the expressivity of the classifier, i.e., we determine the VC dimension of the set of classifiers S ∆ . Recall that a dataset is shattered by a class of binary classifiers if for any possible labeling of the data there is a classifier in the class that produces the same labeling. The VC dimension of the class of classifiers is the maximal size of a dataset that can be shattered by the class of functions.

Theorem 3.1. Let ∆ be a simplicial fan with n generators. Then the VC dimension of the set of classifiers S ∆ is equal to n .

## 3.1 Geometry of the 0/1-loss

We seek to understand the geometry inside the parameter space R n &gt; 0 = { a : a &gt; 0 } . For an example which illustrates all definitions and results stated in this and the following subsection (Sections 3.1 and 3.2), we refer to Example A.3 in Appendix A.2.

Given an unlabeled data point x ( i ) , we associate the classification hyperplane

<!-- formula-not-decoded -->

which separates parameters a inducing a classifier c a with c a ( x ( i ) ) = 0 from the ones with c a ( x ( i ) ) = 1 . These hyperplanes define the data arrangement

<!-- formula-not-decoded -->

Figure 2: An example of a 1 -dimensional dataset, perfectly classified by a star supported on a fan with n = 2 rays, and the level sets of the two loss functions in parameter space R 2 &gt; 0 . This example is explained in detail in Appendix A.2.

<!-- image -->

The data arrangement subdivides the ambient space R n &gt; 0 into (possibly empty) half-open chambers , i.e., subsets of the form

<!-- formula-not-decoded -->

where X 0 , X 1 is any partition of X . By construction, the data arrangement has the following properties.

Proposition 3.2. The half-open chambers of the data arrangement are in bijection with classifications of the dataset. More precisely, each half-open chamber is the set of vectors a whose induced classifiers agree on the dataset.

A direct consequence of Proposition 3.2 is that the false positives, the false negatives and thus also the discrete loss is constant on the half-open chambers. In general, since every chamber is convex and corresponds to a unique classification, the set of all parameters a that perfectly separate the data points is convex.

Corollary 3.3. The discrete loss function err( a ) is constant on the half-open chambers of the data arrangement. The parameters a for which c a perfectly separate X form a convex set.

For any function g : R n → Z ≥ 0 and k ∈ Z ≥ 0 , the k -th level set of g , denoted L ( g, k ) , consists of all parameters a ∈ R n with g ( a ) = k . Further, the k -th sublevel set , denoted S ( g, k ) is defined as S ( g, k ) = ⋃ k i =0 L ( g, i ) . In particular, all the (sub)level sets of the discrete loss function, L (err , k ) , are unions of half-open chambers. Distinguishing further between false positive and false negatives, we obtain the following geometric structure of their sublevel sets.

Theorem 3.4. The sublevel sets of FP and FN are star-convex sets with star center 0 and ∞ , respectively. That is, for all a , t ∈ R n &gt; 0 ,

- (i) FP( a ) ≤ FP( a + t ) .
- (ii) FN( a ) ≥ FN( a + t ) .

Corollary 3.3 implies that if the dataset is separable, i.e., if L (err , 0) = ∅ , then the set of perfect classifiers L (err , 0) is a convex set. Under the same assumption, we can make a similar statement as Theorem 3.4 about the sublevel sets of the discrete loss function.

glyph[negationslash]

Theorem 3.5. Let c a be a classifier that perfectly separates the dataset X , i.e., let c a ∈ L (err , 0) , and let c b ∈ L (err , k ) . Then for every d ∈ [ a , b ] holds c d ∈ S (err , k ) . In particular, the sublevel sets S (err , k ) are star-convex and connected through walls of co-dimension 1 for every k .

Theorem 3.5 shows that if the data is separable, then the sublevel sets of err are star-convex, but not necessarily convex in the usual sense (see also Example A.3 in Appendix A.2). Much stronger, in the case of non-separable data, sublevel sets and even the set of minimizers of the discrete loss can be disconnected. One example for this scenario is the point configuration from Example A.3, with labeling 0 , 0 , 1 , 1 , 1 , 1 , 0 , 0 . Here, the minimum of the discrete loss-function is 4 , attained in four non-neighboring half-open chambers.

To summarize the results in this subsection, our geometric analysis reveals that sublevel sets under the 0/1-loss decompose into convex chambers within a hyperplane arrangement (Corollary 3.3), while simultaneously exhibiting star-convexity with respect to any optimal point in the chamber which minimizes the loss (Theorems 3.4 and 3.5). These theorems establish that while global optimization over the 0-1 loss is hard due to the combinatorial complexity of the overall non-convex landscape, the local landscape is well-connected. In particular, for separable data these results show that local optimization methods can between neighboring chambers along straight lines, improving in each step without getting stuck at local minima. For non-separable data, a similar behavior is still exhibited by false positives and false negatives.

## 3.2 Log-likelihood loss

In Section 3.1 we have observed that the 0/1-loss or discrete loss admits discrete geometric structures in the parameter space which are governed by an affine hyperplane arrangement. However, the discrete loss function is difficult to compute in practice. We thus propose an alternative loss-function

for practical purposes. The log-likelihood loss function turns out to be well-suited for optimization procedures due to concavity. The definitions and results stated in this subsection are exemplified in Example A.3 in Appendix A.2.

Theorem 3.6. The log-likelihood function L ( a ) is concave. In particular, any local maximum is a global maximum.

The computation of the maximum likelihood estimator, the maximizer of the log-likelihood lossfunction, can be summarized by the following algorithm.

Algorithm 1 Computation of the maximum likelihood estimator

<!-- formula-not-decoded -->

1: determine A

```
( i ) ( i ) i m X i =1
```

<!-- formula-not-decoded -->

Convex (and thus also concave) functions can be optimized in polynomial time. Moreover, the description in Section 2 implies that A X can be determined in polynomial time. This implies the following.

Corollary 3.7. Algorithm 1 can be computed in polynomial time in the size of the input data.

Strictly speaking, for Algorithm 1 to be a convex program, we need to consider the closed positive orthant. Any solution on the boundary corresponds to a degenerate star, where star-defining points on rays move to infinity. This degenerate case will be treated in Theorem 3.9.

Similarly to Section 3.1, we now consider the superlevel sets of the log-likelihood loss. For given t ∈ R , the superlevel S L ( t ) of L is defined as

<!-- formula-not-decoded -->

The following is a direct consequence of the fact that L is concave.

Corollary 3.8. The superlevel sets S L ( t ) are convex sets for all t ∈ R .

We now analyze cases in which the maximum of the log-likelihood loss is unique. For this, consider the positively labeled subdataset X 1 = { x ( i ) : y ( i ) = 1 } ⊂ X and let A X 1 be the submatrix of A X composed of all rows [ x ( i ) ] corresponding to x ( i ) ∈ X 1 ; similarly we define A X 0 for X 0 = X \ X 1 . Then we have the following sufficient condition for a unique maximum of L ( a ) .

Theorem 3.9. The log-likelihood loss function L ( a ) is strictly concave if the matrix A X 1 has rank n . Furthermore, if also the rank of A X 0 equals n , then L ( a ) has a unique (possibly degenerate) maximum in the closed positive orthant R n ≥ 0 .

Matrices of the form A X also appear in the study of reconstruction of polytopes with fixed facet directions from support function evaluations. Dostert and Jochemko [2023] showed that such a reconstruction of a polytope is unique if and only if rank A X = n . In other words, A X ∈ ( R n × m ) \ V where V is the algebraic variety encoding that rank A X &lt; n . In particular, it is sufficient that each interior of a maximal cell of ∆ contains a data point, possibly after adding minimal noise. From Theorem 3.9 together with Dostert and Jochemko [2023, Corollary 3.14] we obtain the following sufficient condition on the uniqueness of the maximum of L ( a ) .

Theorem 3.10. Let X 0 and X 1 be sets of noisy data labeled with 0 and 1 respectively, and such that for every maximal cell σ ∈ ∆ there is at least one data point from X 0 and X 1 in the interior of σ . Then L ( a ) has a unique (possibly degenerate) maximum in the closed positive orthant R n ≥ 0 .

Observe that the assumptions in the preceding statement can be considered mild: Given enough data points in generic position it is reasonable to assume that the center of the star can always be chosen in a way such that the condition is satisfied.

Note that the maximizer of L , as well as the number of false positives and false negatives also depends on the choice of the rate parameter λ . We now treat λ as an additional variable and consider the log-likelihood function L ( λ, a ) .

Theorem 3.11. Let X be a dataset and λ 0 &gt; 0 be a rate parameter such that L ( λ 0 , a ) has a unique maximum a ∗ ( λ 0 ) . Then L ( λ, a ) has a unique maximum for all λ &gt; 0 , denoted a ∗ ( λ ) , and the function λ ↦→ a ∗ ( λ ) is a straight line inside R n &gt; 0 approaching the origin.

Corollary 3.12. Let X be a dataset such that L ( λ, a ) has a unique maximum at a ∗ ( λ ) for all λ &gt; 0 . Then

- (i) FP( a ∗ ( λ )) is monotone decreasing in λ , and
- (ii) FN( a ∗ ( λ )) is monotone increasing in λ .

Given the monotonicity of the number of false positives and false negatives in the optimal solution, it is natural to ask if the 0 / 1 -loss, which is given by their sum, is convex in the sense that it first decreases and then increases with varying λ , without any 'ups' and 'downs'. However, this is in general not the case. A counterexample is given by Example A.3.

## 4 Geometry of the generalized parameter space

In the previous section, we have considered polyhedral fans whose cones have their apex at the origin, and varied the shapes of the stars defined on this fixed fan. In this section, we extend this framework by allowing translations. In Section 4.1 we first fix the shape of the star and only vary it by translation, whereas in Section 4.2 we investigate the space of both operations at the same time.

## 4.1 Translations of a fixed star

Recall that the star of a ∈ R n &gt; 0 is defined as star( a ) = { x ∈ R d : f ∆ a ( x ) ≤ 1 } , where f ∆ a = 〈 [ x ] ∆ , a 〉 . Given a translation vector t ∈ R d , we have

<!-- formula-not-decoded -->

where [ x -t ] ∆ captures the nonzero coefficients µ i such that x = t + µ i 1 v i 1 + . . . + µ i k v i k ∈ C + t , and C + t ∈ ∆+ t is the unique cone of the translated fan containing x . On the other hand, we have

<!-- formula-not-decoded -->

where x -C = { x -c | c ∈ C } is the reflected cone -C translated by the vector x . We first analyze the behavior of [ x -t ] ∆ when varying t .

Proposition 4.1. For any x ( i ) , the function t ↦→ [ x ( i ) -t ] ∆ is piecewise-linear, with linear pieces supported on the closed cones of the fan x ( i ) -∆ . For a fixed a ∈ R n &gt; 0 holds

<!-- formula-not-decoded -->

Given a fixed classifier a ∈ R n &gt; 0 , we can ask about the nature of the translational 0 / 1 -loss function

<!-- formula-not-decoded -->

For this, we define an arrangement of stars S 1 , . . . , S n ⊂ R d to be the union of all (possibly non-convex) polyhedral sets of the form S c 1 1 ∩ · · · ∩ S c n n , where S c j j ∈ { S j , R d \ S j } .

Definition 4.2. Given a fixed a ∈ R n &gt; 0 , the data star arrangement of a dataset X = { ( x ( i ) , y ( i ) ) } i m =1 is the arrangement of stars -star( a ) + x ( i ) for i = 1 , . . . , m .

An example of the data star arrangement of a 2 -dimensional dataset, together with the level sets of the translational 0/1-loss is given in Example A.4 in Appendix A.3.

Theorem 4.3. The translational 0 / 1 -loss err a ( t ) is constant on half-open cells of the data star arrangement of X .

For a fixed classifier a ∈ R n &gt; 0 we may also consider maximizing the translational log-likelihood function

<!-- formula-not-decoded -->

To describe the behavior of this function, we define an arrangement of (translated) polyhedral fans ∆ 1 + t 1 , . . . , ∆ k + t k to be the polyhedral complex consisting of all (necessarily convex) intersections of the form ( C 1 + t 1 ) ∩ ( C 2 + t 2 ) ∩ · · · ∩ ( C k + t k ) where C i + t i is a cone in ∆ i + t i .

Definition 4.4. Given a fixed a ∈ R n &gt; 0 , the data fan arrangement of a dataset X = { ( x ( i ) , y ( i ) ) } i m =1 is the arrangement of fans x ( i ) -∆ for i = 1 , . . . , m .

Theorem 4.5. The translational log-likelihood function is concave on any maximal cell of the data fan arrangement. In particular, t ↦→L a ( t ) is a piecewise concave function on R d .

## 4.2 Translations and transformations of the star together

In Section 3 we have considered classification by a starshaped polyhedral set in which t ∈ R d is fixed (and assumed to be the origin), and varied a ∈ R n &gt; 0 . In Section 4.1 we have fixed a ∈ R n &gt; 0 and varied t ∈ R d . As a final step, we will now vary the tuple ( a , t ) ∈ R n &gt; 0 × R d , i.e., both parameters simultaneously, and consider the classifiers

<!-- formula-not-decoded -->

For a fixed cone C ∈ ∆ and a data point x ( i ) , we consider the sets which contain those tuples ( a , t ) such that x ( i ) ∈ C + t , and such that x ( i ) lies inside or outside star( a ) + t , respectively:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 4.6. The sets S 0 ( C, x ( i ) ) and S 1 ( C, x ( i ) ) are basic semialgebraic sets, i.e., finite intersections of solutions to polynomial inequalities. More precisely, each of them is the intersection of a polyhedral cone with a single quadratic inequality.

We extend the 0/1-loss to be viewed as a function err( a , t ) in variables ( a , t ) ∈ R n &gt; 0 × R d . In contrast to Theorem 3.5 and Theorem 4.3, the (sub)level sets in this extended product of both parameter spaces are neither polyhedral nor do they have piecewise-linear boundary, but they are semialgebraic.

Theorem 4.7. The level sets and sublevel sets of the extended 0/1-loss on R n &gt; 0 × R d are semialgebraic sets, i.e., finite unions and intersections of solutions to polynomial inequalities. The defining polynomials have degree at most 2 .

In the previous sections, the shape of the (sub)level sets immediately implied path-connectedness. However, in this more general framework, this property does not necessary hold.

Theorem 4.8. The (sub)level sets of the extended 0/1-loss are in general not path-connected.

We end this section by considering the expressivity of our starshaped classifiers, when translation is allowed. We give the following upper bound.

Theorem 4.9. For a fixed simplicial polyhedral fan in dimension d with k maximal cones, the VC dimension of the class of functions { c a , t : ( a , t ) ∈ R n &gt; 0 × R d } is in O ( d 2 log 2 ( d ) k log 2 ( k )) .

## 5 Experiments

We conducted small-scale experiments where we tested Algorithm 1, implemented in SageMath 10.5 [The Sage Developers, 2024], on two-dimensional synthetic data. The computations were done on a MacBook Pro equipped with an M2 Pro chip and 32 GB of RAM. For comparison, we also applied several standard binary classification methods leading to convex optimization problems on the same dataset, as well as a ReLU neural network. The computation running time ranged from few seconds to one hour.

## 5.1 Data

Figure 3a illustrates 500 data points sampled from a given star-shaped region (in green) defined on eight rays. The data was generated as follows: we randomly selected the x - and y -coordinates of all points from the interval [ -1 , 1] using a uniform distribution and discarded any resulting points ( x, y ) lying outside the unit circle. This was done to achieve a near rotational symmetry of the data set. For each remaining point, we then checked whether it lies inside or outside the star-shaped region. The corresponding label was assigned accordingly, with a 90% probability of being correct.

| Model        |   Accuracy |
|--------------|------------|
| SVM sigmoid  |      0.534 |
| SVM linear   |      0.718 |
| SVM poly     |      0.718 |
| Logistic reg |      0.718 |
| SVM RBF      |      0.78  |
| Neural net   |      0.824 |
| Algorithm 1  |      0.852 |

Figure 3: Synthetic data and accuracy of classification for tested models.

<!-- image -->

## 5.2 Results

In the first experiment, we used Algorithm 1 together with the same eight-ray fan structure to predict the labels of the synthetic data described above. We compared binary classification of our algorithm with multiple standard classification models. In the following we give a description of our results. Illustrations can be found in Appendix B

Running Algorithm 1 on the synthetic data set, the optimal value of the regularization parameter was found to be approximately λ = 0 . 83 , yielding an accuracy of 0 . 852 . The resulting optimal star classifier is shown in Figure 10a. For comparison, we also tested standard implementations of SVMs (with linear, polynomial, RBF, and sigmoid kernels), logistic regression, and a ReLU neural network with two hidden layers of sizes 5 and 2 , respectively. The SVMs with linear and polynomial kernels, as well as logistic regression, performed poorly, assigning all points to the same class, thereby achieving an accuracy of 0 . 718 . The SVM with a sigmoid kernel performed even worse. In contrast, the SVM with an RBF kernel and the neural network achieved better results, with accuracies of 0 . 78 and 0 . 824 , respectively. See Figure 3b for a summary of the results.

In a further experiment, we ran Algorithm 1 on the same dataset but with different underlying fans as input. Specifically, we considered both a refinement and a coarsening of the original fan with eight rays, as depicted in Figure 11. In the case of the refined fan, the decision boundary remained almost unchanged and the accuracy improved marginally. For the coarsened fan, the shape of the decision boundary changed considerably and the accuracy became significantly worse depending on which ray was removed; see Figure 12 for an illustration if the starshaped sets. These results support the following heuristic for fan selection in two dimensions: start with a small number of rays and iteratively refine the fan by adding more rays. If the additional rays do not produce significant new dents in the boundary, they can be safely discarded.

## 6 Conclusion

This article demonstrates that polyhedral starshaped sets constitute a promising family of classifiers, striking a balance between convex polyhedral classifiers and general piecewise linear functions - the latter corresponding to the class of functions representable by ReLU neural networks. The results on VC dimensions highlight that this family remains tractable from a statistical learning perspective. This is further supported by the properties of the proposed loss functions, notably convexity and star-convexity of their (sub)level sets. Moreover, the presented framework provides a high level of flexibility, particularly due to the ability to freely choose the rate parameter λ , which enables manual adjustment of the trade-off between false positives and false negatives as needed.

The presented framework has been tested only on very few example data sets in two dimensions. It remains an open question how to optimally select the parameters, such the underlying fan, the translation vector and the rate parameter, in a manner tailored to the specific problem at hand.

## Acknowledgements

We would like to thank Maria Dostert and Mariel Supina for many fruitful discussions. We also want to thank Roland Púˇ cek for insightful conversations. MB and KJ were supported by the Wallenberg AI, Autonomous Systems and Software Program funded by the Knut and Alice Wallenberg Foundation. KJ was furthermore supported by grant nr 2018-03968 and nr 2023-04063 of the Swedish Research Council as well as the Göran Gustafsson Foundation. MB was furthermore supported by the SPP 2458 "Combinatorial Synergies", funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation).

## References

- A. Astorino and M. Gaudioso. Polyhedral separability through successive LP. Journal of Optimization Theory and Applications , 112(2):265-293, Feb. 2002. doi:10.1023/a:1013649822153.
- A. Blumer, A. Ehrenfeucht, D. Haussler, and M. K. Warmuth. Learnability and the Vapnik-Chervonenkis dimension. Journal of the ACM (JACM) , 36(4):929-965, 1989. doi:10.1145/76359.76371.
3. M.-C. Brandenburg, G. Loho, and G. Montúfar. The real tropical geometry of neural networks for binary classification. Transactions on Machine Learning Research , 2024. URL https: //openreview.net/forum?id=I7JWf8XA2w .
- H. Cevikalp and B. Triggs. Polyhedral conic classifiers for visual object detection and classification. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 4114-4122, 2017. doi:10.1109/CVPR.2017.438.
- M. Dostert and K. Jochemko. Learning polytopes with fixed facet directions. SIAM Journal on Applied Algebra and Geometry , 7(2):440-469, 2023. doi:10.1137/22M1481695.
- A. Kantchelian, M. C. Tschantz, L. Huang, P. L. Bartlett, A. D. Joseph, and J. D. Tygar. Largemargin convex polytope machine. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, editors, Advances in Neural Information Processing Systems , volume 27. Curran Associates, Inc., 2014. URL https://proceedings.neurips.cc/paper\_files/paper/ 2014/file/320f39caebd792d18483222f92c4498e-Paper.pdf .
- A. Kupavskii. The VC-dimension of k -vertex d -polytopes. Combinatorica , 40(6):869-874, Nov. 2020. doi:10.1007/s00493-020-4475-4.
- N. Manwani and P. S. Sastry. Learning polyhedral classifiers using logistic function. In M. Sugiyama and Q. Yang, editors, Proceedings of 2nd Asian Conference on Machine Learning , volume 13 of Proceedings of Machine Learning Research , pages 17-30, Tokyo, Japan, 08-10 Nov 2010. PMLR. URL https://proceedings.mlr.press/v13/manwani10a.html .
9. The Sage Developers. SageMath, the Sage Mathematics Software System (Version 10.5) , 2024. URL https://www.sagemath.org .
- V. N. Vapnik and A. Y. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability &amp; Its Applications , 16(2):264-280, 1971. doi:10.1137/1116025.
- G. M. Ziegler. Lectures on Polytopes . Springer New York, May 2012. ISBN 038794365X. doi:10.1007/978-1-4613-8431-1.

## A Examples

## A.1 Examples of polyhedral fans (Section 2)

Example A.1 (Kites) . For 1 ≤ i ≤ d , let e i ∈ R d be the vector with 1 in coordinate i and 0 in all other coordinates. The coordinate hyperplanes { x ∈ R d : 〈 e i , x 〉 = 0 } divide R d into chambers, and these chambers along with all of their faces form a simplicial fan ♦ with generators {± e i } 1 ≤ i ≤ d . Every star arising from this fan is necessarily convex, and such stars are called kites . Consider the ( d × 2 d ) -matrix A glyph[diamondmath] = ( e 1 , -e 1 , e 2 , -e 2 , . . . , e d , -e d ) . Given x ∈ R d , the vector [ x ] ♦ is obtained as

<!-- formula-not-decoded -->

where the maximum is taken coordinatewise. The 2 -dimensional fan glyph[diamondmath] and the function [ x ] glyph[diamondmath] restricted to each maximal cone is depicted in Figure 4a.

<!-- image -->

(a) The kite fan glyph[diamondmath] .

(b) The Coxeter fan of type B.

Figure 4: The functions [ x ] glyph[diamondmath] and [ x ] B restricted to the full-dimensional cones of the 2 -dimensional fans from Examples A.1 and A.2. 0 k denotes the k -dimensional 0 -vector.

Example A.2 (Type B stars) . The coordinate hyperplanes along with the hyperplanes x i = ± x j for pairs 1 ≤ i &lt; j ≤ d divide R d into chambers, which along with their faces form a simplicial fan B with generators { 0 , ± 1 } d \ { 0 } . The fan B is known as the Coxeter fan of type B . Let x = ( x 1 , . . . , x d ) ∈ R d and let σ : { 1 , . . . , d } → { 1 , . . . , d } be a permutation such that | x σ (1) | ≤ | x σ (2) | ≤ · · · ≤ | x σ ( d ) | . Then we have that

<!-- formula-not-decoded -->

where v 1 , . . . , v d generate a cone of B containing x and

<!-- formula-not-decoded -->

The vector [ x ] B can be recovered from (4). The 2 -dimensional fan B and the function [ x ] B restricted to each maximal cone is depicted in Figure 4b.

## A.2 Examples of the geometry of the parameter space (Section 3)

Example A.3 (Classifications of 1-dimensional dataset) . Consider the 1 -dimensional polyhedral fan ∆ with generators v 1 = -e 1 , v 2 = e 1 , and the 1 -dimensional labeled dataset X consisting of 8 distinct points

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For z ∈ R ≥ 0 we have

<!-- formula-not-decoded -->

The associated data arrangement is depicted in Figure 5a, and Figure 5b shows the 1 -dimensional dataset together with the stars associated to the points

<!-- formula-not-decoded -->

<!-- image -->

- (a) Data arrangement H X .
- (b) The dataset and the stars (thick black line) associated to a 1 , a 2 , a 3 (top to bottom).

<!-- image -->

Figure 5: The data arrangement and dataset from Example A.3.

By Proposition 3.2, each half-open chamber of the data arrangement is the set of classification vectors a whose induced classifiers agree on the dataset. Thus, the number of false positives FP( a ) , the number of false negatives FN( a ) and the discrete loss function err( a ) are constant on each of the half-open chambers (cf. Corollary 3.3). Figure 6 shows the values of these functions and it can be verified that the sublevel sets of FP( a ) and FN( a ) are star-convex with centers 0 and ∞ , respectively (cf. Theorem 3.4). As the data is perfectly separable, Theorem 3.5 implies that the sublevel sets of err( a ) are star-convex and connected through walls of codimension 1 , as depicted in Figure 6c.

Figure 7 shows the level sets of the log-likelihood loss function for the same example, for two choices of the rate parameter λ . In accordance to Corollary 3.8, the plots illustrate that the superlevel sets of

Figure 6: The values of FP( a ) , FN( a ) and err( a ) on the half-open chambers of the data arrangement for the 1 -dimensional dataset from Example A.3. The diagonal lines in all three images show the function λ ↦→ a ∗ ( λ ) defined in Section 3.2.

<!-- image -->

these functions are convex. Moreover, the rate parameter has an influence on the (unique) maximizer of these functions, but they lie on a line λ ↦→ a ∗ ( λ ) , as shown in Theorem 3.11.

The same line is drawn in Figures 6a and 6b, illustrating that the numbers of false positives and false negative are monotone increasing and decreasing, respectively (cf. Corollary 3.12). In Figure 6c, the line crosses regions where the 0 / 1 -loss has values 3 , 2 , 3 , 2 , 3 , 2 , 3 , 4 , respectively. This shows that the 0 / 1 -loss is not 'convex' in the sense that the sequence of values of the 0 / 1 -loss along this line is unimodal, but goes up and down repeatedly.

(a) Level sets of L ( λ, a ) for λ = 0 . 5 . The maximum is attained at a = (0 . 93 , 0 . 48) .

<!-- image -->

(b) Level sets of L ( λ, a ) for λ = 2 . The maximum is attained at a = (0 . 23 , 0 . 12) .

<!-- image -->

Figure 7: Level sets of log-likelihood loss functions for different choices of λ on the dataset from Example A.3. The black dot depicts the minimum a ∗ ( λ ) , lying on the line of maxima when varying the choice of λ .

## A.3 Examples of the geometry of the generalized parameter space (Section 4)

Example A.4 (Star arrangement) . Consider the 2 -dimensional labeled dataset X consisting of 3 distinct points

<!-- formula-not-decoded -->

and let ∆ be the 2 -dimensional Coxeter fan of type B (cf. Example A.2), which has 8 rays. We fix the star through values

<!-- formula-not-decoded -->

Figure 8: The data points from Example A.4, with fixed a and two perfect classifiers.

<!-- image -->

Figure 8 shows two different translations star( a ) + t that are perfect classifiers. Figure 9 shows the associated star arrangement in R 2 , and the sublevel sets of the translational 0/1-loss. In particular, in contrast to Theorem 3.5, it can be observed that the 0 th level set L (err a , 0) consists of two fulldimensional connected components. The translated stars in Figure 8 show one example from each of the connected components.

Figure 9: The star arrangement from Example A.4, and the level sets of the translational 0/1-loss function.

<!-- image -->

## B Experiments

In this section we collect illustrations of the results of our experiments described in Section 5.

8

Figure 10: Decision boundaries for Algorithm 1, a neural network and SVM with RBF kernel.

<!-- image -->

Figure 11: Left: Fan on 8 rays (blue) and the supported star (green) used in the generation of the synthetic data; Center: Refinement of the fan with one extra ray, and an example of a supported star; Right: Coarsening of the fan with one ray removed, and an example of a supported star.

<!-- image -->

Figure 12: Optimal starshaped decision boundaries for Algorithm 1, with the same underlying fan as the synthetic data (left), a refinement of the fan (center) and a coarsening of the fan (right). The underlying fans are depicted in Figure 11.

<!-- image -->

## C Proofs

## C.1 Proof of Theorem 3.1

Proof. Let v 1 , . . . , v n be the generators of ∆ and let glyph[lscript] ∈ { 0 , 1 } n be an arbitrary assignment of 0 / 1 -labels to these generators. Then there exists a classifier c a ∈ S ∆ that assigns the same labels to v 1 , . . . , v n than glyph[lscript] , namely, for glyph[epsilon1] &gt; 0 , we set

<!-- formula-not-decoded -->

Then f ∆ a ( v i ) = a i ≥ 1 if and only if glyph[lscript] ( v i ) = 1 , and thus c a ( v i ) = glyph[lscript] ( v i ) as claimed. It follows that the set of classifiers shatters the set of generators and thus the VC dimension of S ∆ is at least n .

To see that the VC dimension is at most n , we assume that there is a set x (1) , . . . , x ( n +1) ∈ R d of n + 1 points that can be shattered by S ∆ . By construction, for all 1 ≤ i ≤ n + 1 , c a ( x ( i ) ) = 0 if and only if 〈 [ x ( i ) ] ∆ , a 〉 ≤ 1 . In particular, if x (1) , . . . , x ( n +1) can be shattered by S ∆ then [ x (1) ] ∆ , . . . , [ x ( n +1) ] ∆ can be shattered by the set of half-spaces of the form { x ∈ R n : a T x ≤ b, a 1 , . . . , a n ≥ 0 } . This is a subset of the set of halfspaces considered in Proposition C.1 below, and thus we obtain a contradiction by Proposition C.1. This completes the proof.

Proposition C.1. Let H be the set of halfspaces in R n of the form

<!-- formula-not-decoded -->

Then the VC dimension of H equals n .

Proof. We need to show that there is no set of n +1 that can be shattered by H . Let y 1 , . . . , y n +1 be points in R n and let ˜ y 1 , . . . , ˜ y n +1 be their projections on the first n -1 coordinates. Then ˜ y 1 , . . . , ˜ y n +1 are affinely dependent, that is, there exist µ 1 , . . . , µ n , not all equal to zero, such that ∑ i µ i = 0 and ∑ i µ i ˜ y i = 0 . Let

<!-- formula-not-decoded -->

Then A &gt; 0 and we have that

<!-- formula-not-decoded -->

lies in the convex hull of both the sets { ˜ y i : µ i &gt; 0 } and { ˜ y i : µ i &lt; 0 } . We consider the vectors

<!-- formula-not-decoded -->

Then both w + and w -agree with ˜ w on the first n -1 coordinates. W.l.o.g we may assume that ( w -) n ≤ ( w + ) n , that is, the last coordinate of w -is not bigger than the last coordinate of w + . Then we claim that there exists no half-space { x ∈ R n : a T x ≤ b, a n ≥ 0 } in H such that a T y i ≤ b for all i such that µ i &gt; 0 and a T y i &gt; b for all i such that µ i &lt; 0 . To see this, we assume to the contrary that such a hyperplane exists. We then have

<!-- formula-not-decoded -->

and similarly a T w -&gt; b . In particular, a T w -&gt; a T w + . Since w + and w -agree on the first n -1 coordinates it therefore follows that a n ( w + ) n &lt; a n ( w -) and thus w + &lt; w -since a n &gt; 0 , a contradiction. Thus, no hyperplane in H satisfies the claim and thus H does not shatter any set of n +1 points. The VC dimension is thus at most n . To see that it is fact equal to n we observe that the set of unit vectors e 1 , . . . , e n can be shattered.

## C.2 Proof of Theorem 3.4

Proof. For any labeled data point ( x ( i ) , y ( i ) ) and t ≥ 0 ,

<!-- formula-not-decoded -->

In particular, if x ( i ) is in the 1 -class of c a then it is also in the 1 -class of c a + t , so any false positive of c a is also a false positive of c a + t . This shows the first claim. The second claim follows analogously.

## C.3 Proof of Theorem 3.5

Proof. Let c a ∈ L (err , 0) , c b ∈ L (err , k ) and d ( t ) = t b + (1 -t ) a for t ∈ [0 , 1] . Let further X 0 = { x ( i ) : y ( i ) = 0 } and X 1 = X \ X 0 . Since for fixed x the function f ∆ d ( t ) ( x ) = 〈 [ x ] ∆ , d ( t ) 〉 is linear in t , for x ∈ X 0 holds 0 = c a ( x ) ≤ c d ( t ) ( x ) ≤ c b ( x ) ∈ { 0 , 1 } , and therefore 0 = FP( c a ) ≤ FP( c d ( t ) ) ≤ FP( c b ) . For x ∈ X 1 holds 1 = c a ( x ) ≥ c d ( t ) ( x ) ≥ c b ( x ) ∈ { 0 , 1 } , and therefore 0 = FN( c a ) ≤ FN( c d ( t ) ) ≤ FN( c b ) . Since err( c d ( t ) ) = FP( c d ( t ) ) + FN( c d ( t ) ) it follows that c d ( t ) ∈ S (err , k ) for all t ∈ [0 , 1] . This implies that the sublevel sets are star-convex. Furthermore, observe that star-convexity holds with respect to any a for which c a perfectly separates the data. The set of all such a is a full-dimensional cell in the data arrangement. It thus follows that S (err , k ) must consist of cells in the hyperplane arrangement that are connected through walls of codimension 1 , and L (err , k ) and L (err , k +1) are connected through walls of codimension 1 .

## C.4 Proof of Theorem 3.6

Proof. It is sufficient to show that all summands of L ( a ) in the expression above are concave. To that end, we observe that for given training data ( x ( i ) , y ( i ) ) i =1 ,...,m , each summand ( -λ ) f ∆ a ( x ( i ) ) = -λ 〈 [ x ( i ) ] ∆ , a 〉 is linear in a and thus concave. To see that log ( 1 -e -λf ∆ a ( x ( i ) ) ) is concave we calculate

<!-- formula-not-decoded -->

In particular, the Hessian of log ( 1 -e -λf ∆ a ( x ( i ) ) ) ,

<!-- formula-not-decoded -->

is negative semi-definite and thus the likelihood function L ( a ) is concave.

## C.5 Proof if Theorem 3.9

Proof. From the proof of Theorem 3.6 we see that the Hessian of L ( a ) is a negative linear combination of the rank1 matrices ( [ x ( i ) ] ∆ ) T [ x ( i ) ] ∆ for x ( i ) ∈ X 1 , that is, Hess L ( a ) = ∑ i λ i ( [ x ( i ) ] ∆ ) T [ x ( i ) ] ∆ for some λ i &lt; 0 where the sum is over all i such that x ( i ) ∈ X 1 . Now let v be an eigenvector of Hess L ( A ) with eigenvalue µ . Since A X 1 has rank n , there exists an x ( i 0 ) ∈ X 1 with 〈 [ x ( i 0 ) ] T , v 〉 glyph[negationslash] = 0 . Then

<!-- formula-not-decoded -->

glyph[negationslash]

It follows that µ = 0 and thus µ &lt; 0 . Thus, Hess L ( A ) is negative definite and thus L ( A ) is strictly concave. Since the parameter space R n &gt; 0 = { a : a &gt; 0 } is convex, Hess L ( A ) has a unique maximum on the extended positive orthant ( R ≥ 0 ∪ {∞} ) n . If furthermore the rank of A X 0 is n then for each j ∈ { 1 , . . . , n } there exists an x ( i ) ∈ X 0 such that [ x ( i ) ] j &gt; 0 . Therefore, we see that L ( a ) →-∞ whenever a j →∞ . Thus, the unique maximum must be attained in the closed positive orthant ( R ≥ 0 ) n .

## C.6 Proof of Theorem 3.11

Proof. We observe that for all t &gt; 0 holds L ( tλ 0 , a ) = L ( λ 0 , t a ) . Therefore, for fixed t &gt; 0 holds

<!-- formula-not-decoded -->

for all a ∈ R n &gt; 0 . Thus, a tλ 0 = 1 /t · a ∗ ( λ 0 ) is the unique maximum of L ( tλ 0 , a ) , and all maxima lie on the ray { a ∗ ( λ ) : λ &gt; 0 } = { a ∗ ( tλ 0 ) : t &gt; 0 } = { 1 t a ∗ ( λ 0 ) : t &gt; 0 } .

## C.7 Proof of Corollary 3.12

Proof. From the proof of Theorem 3.11 we see that for any 0 &lt; λ &lt; λ ′ holds a ∗ ( λ ′ ) = λ/λ ′ · a ∗ ( λ ) &lt; a ∗ ( λ ) . Since a ∗ ( λ ) ∈ R n &gt; 0 both claims follow from Theorem 3.4.

## C.8 Proof of Proposition 4.1

Proof. We begin with the first statement. First, let C = cone( v i 1 , . . . , v i d ) be a full-dimensional cone of ∆ such that t is contained in the interior of the cone x ( i ) -C of the fan x ( i ) -∆ . Equivalently, x ( i ) -t ∈ C . With V C = ( v i 1 . . . v i d ) , we can thus compute [ x ( i ) -t ] ∆ = V -1 C ( x ( i ) -t ) , which is a linear function in t . The statement for lower-dimensional cones follows by taking limits. The second statement follows from substitution of the variables x ↦→-t and t ↦→ x ( i ) in (3).

## C.9 Proof of Theorem 4.3

Proof. Proposition 4.1 implies that

<!-- formula-not-decoded -->

and is thus constant on each cell of the data star arrangement.

## C.10 Proof of Theorem 4.5

Proof. By Proposition 4.1, for any i , t ↦→ f ∆ a ( x ( i ) -t ) = 〈 a , [ x ( i ) -t ] 〉 is linear in t on every maximal cell F of the fan arrangement of x ( i ) -∆ , i = 1 , . . . , m . Let g F ( t ) = 〈 a , [ x ( i ) -t ] 〉 be this linear function. Then the Hessian of L a ( t ) is the sum of matrices of the form

<!-- formula-not-decoded -->

Since these are negative semi-definite, this proofs the claim.

## C.11 Proof of Proposition 4.6

Proof. First note that

<!-- formula-not-decoded -->

The first set equals R n &gt; 0 × ( -C + x ( i ) ) . Restricted to ( a , t ) such that t ∈ x ( i ) -C , the function t ↦→ [ x ( i ) -t ] is a linear map by Proposition 4.1, so the expression f ∆ a ( x ( i ) -t ) = 〈 [ x ( i ) -t ] ∆ , a 〉 is a quadratic polynomial in variables t 1 , . . . , t d , a 1 , . . . , a n . Therefore, S 0 ( C, x ( i ) ) it the intersection of solutions to the linear inequalities defining the polyhedral cone R n &gt; 0 × ( -C + x ( i ) ) , and the quadratic inequality 〈 [ x ( i ) -t ] ∆ , a 〉 ≤ 1 . Similarly, S 1 ( C, x ( i ) ) is the intersection of R n &gt; 0 × ( -C + x ( i ) ) with the set of solutions to the inequality 〈 [ x ( i ) -t ] ∆ , a 〉 &gt; 1 .

## C.12 Proof of Theorem 4.7

Proof. We consider the subdivision of R n &gt; 0 × R d into sets

<!-- formula-not-decoded -->

where we range over all possible b ( C, i ) ∈ { 0 , 1 } for all C ∈ ∆ , i ∈ { 1 , . . . , m } . By Proposition 4.6, each S b ( C,i ) ( C, x ( i ) ) is basic semialgebraic with defining polynomials of degree at most 2 , and hence the same holds for the above finite intersection. By construction, the extended 0 / 1 -loss is constant on each ⋂ C ∈ ∆ ⋂ m i =1 S b ( C,i ) ( C, x ( i ) ) . Fix k ∈ Z ≥ 0 . Then the k th level set L (err , k ) of the extended 0 / 1 -loss is the (finite) union over all sets ⋂ C ∈ ∆ ⋂ m i =1 S b ( C,i ) ( C, x ( i ) ) on which the extended 0 / 1 -loss is equal to k . Thus, the level set is a semialgebraic set. Since sublevel sets are finite unions of level sets, the same holds for sublevel sets.

## C.13 Proof of Theorem 4.8

Proof. We show this statement by giving an example of a data-set with a disconnected level set L (err , 0) . For this, we continue with Example A.4, a configuration of 3 data points in R 2 . The 2 -dimensional Coxeter fan of type B has 8 rays and 8 maximal cones. Thus, the parameter space R 8 &gt; 0 × R 2 is subdivided into cells of the form ⋂ C ∈ ∆ ⋂ i ∈{ 1 , 2 , 3 } S b ( C,i ) ( C, x ( i ) ) , many of these are empty or lower dimensional. The extended 0/1-loss attains the value 0 on 16 maximal cells. To describe them, we use the indexing of the rays as depicted in Figure 4b, and denote C i = cone( v i , v i +1 ) for i = 1 , . . . , 7 , C 8 = cone( v 8 , v 1 ) . One example of a valid configuration is a tuple ( a , t ) such that x (1) ∈ (star( a ) + t ) ∩ ( C 4 + t ) , x (2) ∈ ( R 2 \ (star( a ) + t )) ∩ ( C 3 + t ) and x (3) ∈ (star( a ) + t ) ∩ ( C 2 + t ) , as depicted in Figure 8a. The set of ( a , t ) satisfying these conditions

is S 0 ( C 4 , x (1) ) ∩S 1 ( C 3 , x (2) ) ∩S 0 ( C 2 , x (3) ) . In total, the set of perfect classifiers, i.e., the 0 th level set is the union of the following 16 nonempty cells:

<!-- formula-not-decoded -->

It can be checked that the union of the sets in each of the above columns is path connected. However, there is no path from a point in the left column to a point in the right column. If these sets were path connected, then there was a path from the configuration depicted in Figure 8a to the configuration in Figure 8b by a continuous translation of the star and by shifting the points 1 a i v i along the rays continuously. But since x (2) lies in the convex hull of x (1) and x (3) and the line segment through any of these point is parallel to the rays v 2 , v 6 , any such continuous transformation necessarily increases the number of mistakes at a certain point.

## C.14 Proof of Theorem 4.9

Proof. By Kupavskii [2020], the set of all simplices has VC dimension O ( d 2 log 2 ( d )) . Any starshaped classifier c a , t is a union of k simplices. Thus, the result follows from Blumer et al. [1989].

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract briefly summarizes the content of the article. The introduction positions the results of the paper within the literature to our best knowledge, and gives a more detailed overview over the results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: A discussion of the limitations is given in the introduction under 'Limitations'. Guidelines:

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

Justification: All assumptions are either clearly stated in the description of the model, or explicitly stated in the theorems. Rigorous proofs are given in Appendix B. Due to the amount of theoretical results we have not included informal proofs for the statements in the core of the paper.

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

Justification: The paper provides all necessary information to reproduce the experiments. The experimental results are independent of the main claims and main results of the article, which is theoretical in nature.

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

Answer: [No]

Justification: The small experiments in this article are not central to the contribution and easily reproducible with the description provided in the article.

## Guidelines:

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

Justification: The paper specifies all necessary details to fully understand and reproduce the experiments conducted for this article.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Statistical significance is not relevant for the conducted experiments.

## Guidelines:

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

Justification: All of this information is provided.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in this article fully complies with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Since this is a purely theoretical work, there is no immediate societal impact.

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

Justification: This paper does not involve the release of any data or models, so the paper poses no such risks.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

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
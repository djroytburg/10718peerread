## Depth-Bounds for Neural Networks via the Braid Arrangement

## Moritz Grillo

Max Planck Institute for Mathematics in the Sciences moritz.grillo@mis.mpg.de

## Georg Loho

University of Technology Nuremberg christoph.hertrich@utn.de

Freie Universität Berlin University of Twente georg.loho@math.fu-berlin.de

## Abstract

We contribute towards resolving the open question of how many hidden layers are required in ReLU networks for exactly representing all continuous and piecewise linear functions on R d . While the question has been resolved in special cases, the best known lower bound in general is still 2. We focus on neural networks that are compatible with certain polyhedral complexes, more precisely with the braid fan. For such neural networks, we prove a non-constant lower bound of Ω (log log d ) hidden layers required to exactly represent the maximum of d numbers. Additionally, we provide a combinatorial proof that neural networks satisfying this assumption require three hidden layers to compute the maximum of 5 numbers; this had only been veri fi ed with an excessive computation so far. Finally, we show that a natural generalization of the best known upper bound to maxout networks is not tight, by demonstrating that a rank-3 maxout layer followed by a rank-2 maxout layer is suf fi cient to represent the maximum of 7 numbers.

## 1 Introduction

Among the various types of neural networks, ReLU networks have become particularly prominent [Glorot et al., 2011, Goodfellow et al., 2016]. For a thorough theoretical understanding of such neural networks, it is important to analyze which classes of functions we can represent with which depth. Classical universal approximation theorems [Cybenko, 1989, Hornik, 1991] ensure that just one hidden layer can approximate any continuous function on a bounded domain with arbitrary precision. However, establishing an analogous result for exact representations remains an open question and is the subject of ongoing research [Arora et al., 2018, Hertrich et al., 2023, Haase et al., 2023, Valerdi, 2024, Averkov et al., 2025].

While in practical settings approximate representations are often suf fi cient, studying the exact piecewise linear structure of neural network representations enabled deep connections between neural networks and fi elds like tropical and polyhedral geometry [Huchette et al., 2023]. These connections, in turn, are important for algorithmic tasks like neural network training [Arora et al., 2018, Goel et al., 2021, Khalife and Basu, 2022, Froese et al., 2022, Froese and Hertrich, 2023, Bertschinger et al., 2023] and veri fi cation [Li et al., 2019, Katz et al., 2017, Froese et al., 2025b,a, Stargalla et al., 2025], including understanding the computational complexity of the respective tasks.

Arora et al. [2018] initiate the study of exact representations by showing that the class of functions exactly representable by ReLU networks is the class of continuous piecewise linear (CPWL) functions.

## Christoph Hertrich

Speci fi cally, they demonstrate that every CPWL function de fi ned on R d can be represented by a ReLU network with ⌈ log 2 ( d + 1) ⌉ hidden layers. This result is based on Wang and Sun [2005], who reduce the representation of a general CPWL function to the representation of maxima of d +1 af fi ne terms. By computing pairwise maxima in each layer, such a maximum of d +1 terms can be computed with logarithmic depth overall in the manner of a binary tree. Very recently, Bakaev et al. [2025b] improved the upper bound by proving that every CPWL function can be represented with ⌈ log 3 ( d -1) ⌉ +1 hidden layers. Their results refute the conjecture of Hertrich et al. [2023] that ⌈ log 2 ( d +1) ⌉ hidden layers are indeed necessary to compute all CPWL functions.

Based on the result by Wang and Sun [2005], Hertrich et al. [2023] deduced that it suf fi ces to determine the minimum depth representation of the maximum function. While it is easy to show that max  0 , x 1 , x 2  cannot be represented with one hidden layer [Mukherjee and Basu, 2017], Bakaev et al. [2025b] showed that two hidden layers are suf fi cient to represent max  0 , x 1 , x 2 , x 3 , x 4  . However, it remains open if there exists a CPWL function on R d that really needs logarithmic many hidden layers to be represented. In particular, it is already open whether there is a function that needs more than two hidden layers to be represented.

Understanding depth lower bounds is important for clarifying the potential advantages of architectural choices. In particular, proving depth lower bounds on computing the max function helps formally explain why elements like max-pooling layers are powerful and cannot be easily replaced by shallow stacks of standard ReLU layers, regardless of their width.

In order to identify tractable special cases to prove lower bounds on the necessary number of hidden layers to compute the max function, two approaches have been pursued so far. The fi rst restricts the possible breakpoints of all neurons in a network computing x → max  0 , x 1 ,    , x d  . A breakpoint of a neuron is an input for which the function computed by the neuron is non-differentiable. A neural network is called B 0 d -conforming if breakpoints only appear where the ordering of some pair of coordinates changes (i.e., all breakpoints lie on hyperplanes x i = x j or x i = 0 ). While B 0 d -conforming networks can compute the max function with ⌈ log 2 ( d +1) ⌉ hidden layers, Hertrich et al. [2023] show that 2 hidden layers are insuf fi cient to compute the function max  0 , x 1 , x 2 , x 3 , x 4  , using a computational proof via a mixed integer programming formulation of the problem. The second approach restricts the weights of the network. Averkov et al. [2025] show that, if all weights are N -ary fractions, the max function can only be represented by neural network with depth Ω ( log d log log N ) by extending an approach of Haase et al. [2023]. Furthermore, Bakaev et al. [2025a] proved lower bounds for the case when some or all weights are restricted to be nonnegative. To the best of our knowledge, the two approaches of restricting either the breakpoints or the weights are incomparable.

Our contributions We follow the approach from Hertrich et al. [2023] and prove lower bounds on B 0 d -conforming networks. On one hand, following Hertrich et al. [2023], we believe that understanding B 0 d -conforming networks might also shed light on the expressivity of general networks, for example, by studying different underlying fans instead of focusing on the braid fan as an intermediate step. On the other hand, B 0 d -conforming also appears in Brandenburg et al. [2025] and Froese et al. [2025b] due to the connection to submodular functions and graphs.

In Section 4 we prove for d = 2 2 ℓ -1 that the function x → max  0 , x 1 ,    , x d  is not representable with a B 0 d -conforming ReLU network with ℓ hidden layers. This means that depth Ω (log log d ) is necessary for computing all CPWL functions, yielding the fi rst conditional non-constant lower bound without restricting the weights of the neural networks.

To prove our results, the fi rst observation is that the set of functions that are representable by a B 0 d -conforming network forms a fi nite-dimensional vector space (Proposition 2.2). While one would like to identify subspaces of this vector space representable with a certain number of layers, taking the maximum of two functions does not behave well with the structure of linear subspaces. To remedy this, we identify a suitable sequence of subspaces F L ( k ) for k = 1 , 2 ,    that can be controlled through an inductive construction. These auxiliary subspaces arise from the correspondence between B 0 d -conforming functions and set functions. This allows us to employ the combinatorial structure of the collection of all subsets of a fi nite ground set. This is also re fl ected in the structure of the breakpoints of B 0 d -conforming functions. Hence, we are able to show that applying a rank2 -maxoutlayer to functions in F L ( k ) yields a function in F L ( k 2 + k ) . Iterating this argument yields the desired bounds.

In Section 5, we focus on the case d = 4 . We provide a combinatorial proof of the result of Hertrich et al. [2023] showing that the function x → max  0 , x 1 , x 2 , x 3 , x 4  is not representable by a B 0 d -conforming ReLU network with two hidden layers.

Finally, in Section 6, we study maxout networks as natural generalization of ReLU networks. A straightforward generalization of the upper bound of Arora et al. [2018] shows that B 0 d -conforming maxout network with ranks r i in the hidden layers i = 1 ,    , ℓ can compute the maximum of  ℓ i =1 r i numbers. We prove that this upper bound is not tight: a maxout network with one rank3 layer and one rank2 layer can compute the maximum of 7 numbers, that is, the function x → max  0 , x 1 ,    , x 6  .

Further Related Work In light of the prominent role of the max function for neural network expressivity, Safran et al. [2024] studied ef fi cient neural network approximations of the max function.

In an extensive line of research, tradeoffs between depth and size of neural networks have been explored, demonstrating that deep networks can be exponentially more compact than shallow ones [Montúfar et al., 2014, Telgarsky, 2016, Eldan and Shamir, 2016, Arora et al., 2018, Ergen and Grillo, 2024]. While most of these works also involve lower bounds on the depth, they are usually proven under assumptions on the width. In contrast, we aim towards proving lower bounds on the depth for unrestricted width. The opposite perspective, namely studying bounds on the size of neural networks irrespective of the depth, has been subject to some research using methods from combinatorial optimization [Hertrich and Skutella, 2023, Hertrich and Sering, 2024, Hertrich and Loho, 2024].

One of the crucial techniques in expressivity questions lies in connections to tropical geometry via Newton polytopes of functions computed by neural networks. This was initiated by Zhang et al. [2018], see also Maragos et al. [2021], and subsequently used to understand decision boundaries, bounds on the depth, size, or number of linear pieces, and approximation capabilities [Montúfar et al., 2022, Misiakos et al., 2022, Haase et al., 2023, Brandenburg et al., 2024, Valerdi, 2024, Hertrich and Loho, 2024].

## 2 Preliminaries

In Appendix A, the reader can fi nd an overview of the notation used in the paper and in Appendix B detailed proofs of all the statements.

Polyhedra We review basic de fi nitions from polyhedral geometry; see Schrijver [1986], Ziegler [2012] for more details.

̸

A polyhedral complex P is a fi nite collection of polyhedra such that (i) ∅ ∈ P , (ii) if P ∈ P then all faces of P are in P , and (iii) if P, P ′ ∈ P , then P  P ′ is a face both of P and P ′ . A polyhedral fan is a polyhedral complex where all polyhedra are cones. The lineality space of a polyhedron P is de fi ned as  v ∈ R d  x + v ∈ P for all x ∈ P  . The lineality space of a polyhedral complex P is the lineality space of one (and therefore all) P ∈ P .

A polyhedron P is the intersection of fi nitely many closed halfspaces and a polytope is a bounded polyhedron. A hyperplane supports P if it bounds a closed halfspace containing P , and any intersection of P with such a supporting hyperplane yields a face F of P . A face is a proper face if F ⊊ P and F = ∅ and inclusion-maximal proper faces are referred to as facets . A (polyhedral) cone C ⊆ R n is a polyhedron such that λ u + µv ∈ C for every u, v ∈ C and λ , µ ∈ R ≥ 0 . A cone is pointed if it does not contain a line. A cone C is simplicial , if there are linearly independent vectors v 1 ,    , v k ∈ R n such that C =   k i =1 λ i v i  λ i ≥ 0  .

Neural networks and CPWL functions A continuous function f : R n → R is called continuous and piecewise linear (CPWL), if there exists a polyhedral complex P such that the restriction of f to each full-dimensional polyhedron P ∈ P n is an af fi ne function. If this condition is satis fi ed, we say that f and P are compatible with each other. We denote the set of all CPWL functions from R d to R by CPWL d .

For a number of hidden layers ℓ ≥ 0 , a neural network with recti  ed linear unit (ReLU) activation is de fi ned by a sequence of ℓ + 1 af fi ne maps T i : R n i -1 → R n i , i ∈ [ ℓ + 1] . We assume that n 0 = d and n ℓ +1 = 1 . If σ denotes the function that computes the ReLU function x → max  x, 0 

in each component, the neural network is said to compute the CPWL function f : R d → R given by f = T ℓ +1 ◦ σ ◦ T ℓ ◦ σ ◦ · · · ◦ σ ◦ T 1 .

A rankr -maxout layer is de fi ned by r af fi ne maps T ( q ) : R d → R n for q ∈ [ r ] and computes the function x → (max  ( T (1) x ) j ,    , ( T ( r ) x ) j  ) j ∈ [ n ] . For a number of hidden layers ℓ ≥ 0 and a rank vector r = ( r 1 ,    , r ℓ ) ∈ N ℓ , a rankr -maxout neural network is de fi ned by maxout layers f i : R n i -1 → R n i of rank r i for i ∈ [ ℓ ] respectively and an af fi ne transformation T out : R n ℓ → R . The rankr -maxout neural network computes the function f : R d → R given by f = T out ◦ f ℓ ◦ · · · ◦ f 1 . Let M r d be the set of functions representable by a rankr -maxout neural network with input dimension d . Moreover, let M 2 d ( ℓ ) be the set of functions representable with networks with ℓ rank2 -maxout layers.

## The braid arrangement and set functions

De  nition 2.1. The braid arrangement in R d is the hyperplane arrangement consisting of the  d 2  hyperplanes x i = x j , with 1 ≤ i &lt; j ≤ d . The braid fan B d is the polyhedral fan induced by the braid arrangement.

Sometimes we will also refer to the fan given by the  d +1 2  hyperplanes x i = x j and x i = 0 for 1 ≤ i &lt; j ≤ d , which we denote by B 0 d .

<!-- formula-not-decoded -->

We summarize the properties of the braid fan that are relevant for this work. For more details see Stanley [2007]. The k -dimensional cones of B d are given by where 1 S =  i ∈ S e i . The braid fan has span( 1 [ d ] ) as lineality space. Dividing out the lineality space of B d yields B 0 d -1 . See Figure 1a for an illustration of B 0 d .

Using the speci fi c structure of the cones of B d in terms of subsets of [ d ] allows to relate the vector space V B d of CPWL functions compatible with the braid fan B d with the vector space of set functions F d := R 2 [ d ] : restricting to the values on  1 S  S ⊆ [ d ] yields a vector space isomorphism Φ : V B d → F d whose inverse map is given by interpolating the values on  1 S  S ⊆ [ d ] to the interior of the cones of the braid fan. Detailed proofs of all statements can be found in Appendix B.

Proposition 2.2. The linear map Φ : V B d → F d given by F ( S ) := Φ ( f )( S ) = f ( 1 S ) is an isomorphism.

This implies that V B d has dimension 2 d . Another basis for V B d is given by  σ M  M ∈ 2 [ d ]  , where the function σ M : R d → R is de fi ned by σ M ( x ) = max i ∈ M x i [Danilov and Koshevoy, 2000, Jochemko and Ravichandran, 2022]. We have the following strict containment of linear subspaces:

<!-- formula-not-decoded -->

Let X and Y be fi nite sets such that X ⊆ Y , then the interval [ X,Y ] :=  S ⊆ [ Y ]  X ⊆ S  is a Boolean lattice with the partial order given by inclusion. The rank of [ X,Y ] is given by  Y \ X  . Sometimes we also write x 1 · · · x n for the set  x 1 ,    , x n  ∈ L and x 1 · · · x n for the set X  ( Y \  x 1 ,    , x n  ) . For a Boolean lattice L = [ X,Y ] of rank n , the rank function r : L → [ n ] 0 is given by r ( S ) =  S  -  X  and r ( S ) is called the rank of S . Moreover, we de fi ne the levels of a Boolean lattice by L i := r -1 ( i ) and introduce the notation L ≤ i :=  j ≤ i L j for the set of elements whose rank is bounded by i . For S, T ∈ L with S ⊆ T , we call [ S, T ] a sublattice of L and de fi ne the vector α S,T ∈ R L by α S,T :=  S ⊆ Q ⊆ T ( -1) r ( Q ) -r ( S ) 1 Q . The set F L := ( R L ) ∗ of real-valued functions on L is a vector space, and for any fi xed S, T ∈ L , the map F → ⟨ α S,T , F ⟩ is a linear functional of F L . Furthermore, let where V B d ( k ) := span  σ M  M ⊆ [ d ] ,  M  ≤ k  . In order to describe the linear subspaces Φ ( V B d ( k )) , we now describe the isomorphism Φ with respect to the basis  σ M  M ∈ 2 [ d ]  .

<!-- formula-not-decoded -->

and F L ( k ) := ( R L ( k )) ⊥ =  F ∈ F L  ⟨ α S,T , F ⟩ = 0 for all α S,T ∈ R L ( k )  be a linear subspace of F L . To simplify notation, we also set F d ( k ) := F 2 [ d ] ( k ) .

Proposition 2.3. The isomorphism Φ : V B d → F d maps the function f =  M ⊆ [ d ] λ M · σ M to the set function de  ned by F ( S ) :=  M ⊆ [ d ] M  S = ∅ λ M · σ M  The inverse Φ -1 : F d → V B d of Φ is

̸

given by the Möbius inversion formula F →  M ⊆ [ d ] -⟨ α [ d ] \ M, [ d ] , F ⟩ . In particular, it holds that Φ ( V B d ( k )) = F d ( k ) for all k ≤ d and dim( F d ( k )) = dim( V B d ( k )) =  k i =1  d i  . See also Figure 1b for an illustration of Proposition 2.3.

## 3 Neural networks conforming with the braid fan

For a polyhedral complex P , we call a maxout neural network P -conforming , if the functions at all neurons are compatible with P . By this we mean that for all i ∈ [ ℓ ] and all coordinates j of the codomain of f i , the function π j ◦ f i ◦    ◦ f 1 is compatible with P , where π j is the projection on the coordinate j . We denote by M r P the set of all functions representable by P -conforming rankr -maxout networks. For the remainder of this article, we only consider the cases M r B d and M r B 0 d

By computing r i maxima in each layer, we can compute the basis functions of V B d (  ℓ i =1 r i ) with a B d -conforming rankr -maxout network.

Lemma 3.1. The function x → max  0 , x 1 ,    , x d -1  can be represented by a B 0 d -1 -conforming rankr -maxout network if and only if the function x → max  x 1 ,    , x d  can be represented by a B d -conforming rankr -maxout network.

Proposition 3.2. For any rank vector r ∈ N ℓ , it holds that all functions in V B d (  ℓ i =1 r i ) are representable by a B d -conforming rankr -maxout network.

For f 1 ,    , f r ∈ V B d , the function max  f 1 ,    , f r  is B d -compatible if taking the maximum does not create breakpoints that do not lie on the braid arrangement, that is, on every cone C of the braid arrangement, it holds that max  f 1 ,    , f r  = f q for a q ∈ [ r ] . Next, we aim to model the compatibility with the braid arrangement for set functions. We call a tuple ( F 1 ,    , F r ) ∈ F r L conforming if for every chain ∅ = S 0 ⊊ S 1 ⊊    ⊊ S n ⊆ [ n ] there is a j ∈ [ r ] such that F j ( S i ) = max  F 1 ,    , F r  ( S i ) for all i ∈ [ n ] 0 . Then, the set C r L ⊆ F r L of conforming tuples are exactly those tuples of CPWL functions such that applying the maxout activation function yields a function that is still compatible with the braid fan as stated in the next lemma. Again, to simplify notation, we also set C r d := C r 2 [ d ] .

Most of the paper is concerned with proving that M r B d is contained in certain subspaces of V B d . Let F r L =  i ∈ [ r ] F L be the r -fold direct sum of F L with itself. In order to model the application of the rankr -maxout activation function for a set function under the isomorphism Φ , we de fi ne for ( F 1 ,    , F r ) ∈ F r L the function max  F 1 ,    , F r  ∈ F L given by max  F 1 ,    , F r  ( S ) = max  F 1 ( S ) ,    , F r ( S )  .

Lemma 3.3. For ( F 1 ,    , F r ) ∈ ( F d ) r , the function max  Φ -1 ( F 1 ) ,    , Φ -1 ( F r )  is B d -conforming if and only if ( F 1 ,    , F r ) ∈ C r d . In this case,

<!-- formula-not-decoded -->

The statement ensures that taking the maximum of the set functions is the same as taking the maximum of the piecewise-linear functions exactly for compatible tuples.

## 4 Doubly-logarithmic lower bound

In this section, we prove that for any number of layers ℓ ∈ N , the function max  0 , x 1 ,    , x 2 2 ℓ -1  is not computable by a B 0 d -conforming rank2 -maxout neural network (or equivalently ReLU neural network) with ℓ hidden layers. Due to the equivalence of B d and B 0 d , we will prove that M 2 B d ( ℓ ) ⊆ (2 2 ℓ 1 ) for d 2 2 ℓ 1 +1 .

First, we de fi ne an operation A on subspaces of V B d that describes rank2 -maxout layers that maintain compatibility with B d . For any subspace U ⊆ V B d , let A ( U ) ⊆ V B d be the subspace containing all

<!-- formula-not-decoded -->

<!-- image -->

the functions computable by a B d -conforming rank 2 -maxout layer that takes functions from U as input. Formally,

<!-- formula-not-decoded -->

Lemma 4.1. It holds that (1) M 2 B d (1) = A ( V B d (1)) = V B d (2) , and (2) for all ℓ ∈ N , M 2 B d ( ℓ ) = A ( M 2 B d ( ℓ -1)) = A ℓ ( V B d (1)) .

Clearly, A ( U 1 ) is a subspace of A ( U 2 ) whenever U 1 is a subspace of U 2 . We recursively de fi ne A ℓ ( U ) = A ( A ℓ -1 ( U )) . This recursive de fi nition allows to describe the set of B d -conforming network with ℓ rank2 -maxout layers M 2 B d ( ℓ ) .

Since it holds that max  f 1 , f 2  = max  0 , f 1 -f 2  + f 2 , we can assume wlog that one of the functions is the zero map, as stated in the following lemma.

Lemma 4.2. It holds that A ( U ) = span  max  0 , f   f ∈ U, max  0 , f  ∈ V B d  .

To prove that M 2 B d ( ℓ ) = A ℓ ( V B d (1)) is a proper subspace of V B d for d ≥ 2 2 ℓ -1 +1 , we perform a layerwise analysis and inductively bound n k depending on k such that A ( V B d ( k )) ⊆ V B d ( n k ) for all k ∈ N . In this attempt, we translate this task to the setting of set functions on Boolean lattices using the isomorphism Φ . Recall that the pairs ( F 1 , F 2 ) ∈ C 2 L are precisely the functions such that the maximum of the corresponding CPWL functions f 1 and f 2 is still compatible with B d . Moreover, it is easy to observe, that the pair (0 , F ) ∈ F 2 L is conforming if and only if F is contained in the set

Again, to simplify notation, we also set C d := C 2 [ d ] and use the notation F + = max  0 , F  . By slightly overloading notation, for any subspace U ⊆ F L , let A ( U ) = span  F +  F ∈ U  C L  . Lemma 3.3 justi fi es this notation and allows us to carry out the argumentation to the world of set functions on Boolean lattices, as we conclude in the following lemma.

<!-- formula-not-decoded -->

Lemma 4.3. It holds that A ( Φ ( U )) = Φ ( A ( U )) for all subspaces U ⊆ V B d . In particular, for any lattice L = [ X,Y ] , it holds that A ( F L (1)) = F L (2) .

In the following, we prove that A ( F L ( k )) ⊆ F L ( k 2 + k ) by an induction on k and Lemma 4.3 serves as the base case.

Next, we describe properties of the vector space R L that will be useful for the induction step. Every sublattice of L of rank k + 1 is of the form [ S, S  T ] , where S  T = ∅ and  T  = k + 1 . For any T ⊆ Y \ X , one can decompose L = [ X,Y ] into the sublattices [ S, S  T ] for all S ⊆ Y \ T , resulting in the following lemma.

Lemma 4.4. Let L = [ X,Y ] be a lattice of rank n . Then, (1) for every T ⊆ Y \ X , it holds that α X,Y ∈ span  α S,S  T  S ⊆ Y \ T  , and (2) for every T ⊆ Y \ X with  T  = k , it holds that α S,S  T -α S ′ ,S ′  T ∈ R L ( k ) for all S, S ′ ∈ [ X,Y \ T ] .

See Figure 2 for a visualization of Lemma 4.4. Lemma 4.4 implies that it suf fi ces to fi nd a T ⊆ Y such that ⟨ α S,S  T , F + ⟩ = 0 for all S ⊆ Y \ T , in order to prove that F L ( n -1) . The idea of the

<!-- image -->

∅

∅

Figure 2: Illustration of Lemma 4.4. The solid line in Figure 2a, decomposes the lattice in [ ∅ , abc ]  [ d, abcd ] , which implies that α ∅ ,abcd = α ∅ ,abc -α d,abcd . The dashed line further decomposes [ ∅ , abc ] = [ ∅ , bc ]  [ a, abc ] . The 3 fi gures illustrate that α S,S  { b,c } -α S ′ ,S ′  { b,c } ∈ R L (2) for all S, S ′ ⊆  a, d  .

-

Figure 3: An illustration of the induction step. Let Y =  a, b, c, d, e  , X = ∅ , L = [ X,Y ] and F ∈ F L (2)  C L . If F ( a ) &lt; 0 and F ( b ) &gt; 0 , then it follows that F ( R ) for all R ∈ [ S, S  T ] for S = ab and T = cde (Figure 3a). In particular, F ∈ F S,S  T (1) and thus, by Lemma 4.4, it holds that F ∈ F S ′ ,S ′  T (1) for all S ′ ⊆ Y \ T . Figure 3b shows the decomposition of the lattice L = [ X,Y ] for T =  c, d, e  into the sublattices [ S, S  T ] for all S ⊆ Y \ T . For every such sublattice we have that F ∈ F [ S,S  T ] (1)  C [ S,S  T ] and thus by induction ⟨ α S,S  T , F + ⟩ = 0 .

<!-- image -->

induction step is to fi nd a T of cardinality at least ( k -1) 2 +( k -1)+1 such that F ∈ F [ S,S  T ] ( k -1) for all S ⊆ Y \ T . Then, applying the induction hypothesis to each sublattice [ S, S  T ] yields ⟨ α S,S  T , F + ⟩ = 0 and hence F + ∈ F L ( n -1) .

If F ∈ F L ( k ) , Lemma 4.4 implies that for any T ′ ⊆ Y \ X of cardinality k , the value ⟨ α S ′ ,S ′  T ′ , F ⟩ is independent of S ′ ⊆ Y \ T ′ . Hence, in this case, it suf fi ces to fi nd a T such that F ∈ F [ S,S  T ] ( k -1) for only one S ⊆ Y \ T , since it is equivalent to F ∈ F [ S,S  T ] ( k -1) for all S ⊆ Y \ T .

̸

Lemma 4.5 says that, given that the positive and negative support are not empty, we can always 'push the elements X + and X -in the support down in the lattice', that is, we can fi nd elements in the supports that are of relatively low rank. See Figure 1c for an illustration.

Given F ∈ F L ( k )  C L , it remains to fi nd such S and T . We de fi ne the support of F ∈ F L by supp( F ) =  S ∈ L  F ( S ) = 0  and the positive and negative support by supp + ( F ) =  S ∈ L  F ( S ) &gt; 0  respectively supp -( F ) =  S ∈ L  F ( S ) &lt; 0  . In particular, F ∈ C L implies that for X + ∈ supp + ( F ) and X -∈ supp -( F ) , it holds that F ( R ) = 0 for all R ⊇ X +  X -.

Lemma 4.5. Let L = [ X,Y ] be a lattice of rank n . Let F ∈ F L ( k )  C L such that F ̸≥ 0 and F ̸≤ 0 . Then, there are X -∈ L ≤ k  supp -( F ) and X + ∈ L ≤ k  supp + as well as Y -∈ L ≥ n -k  supp -( F ) and Y + ∈ L ≥ n -k  supp + ( F ) .

Let S = X +  X -, then F ∈ C L implies that for T = Y \ S , we have that F ( R ) = 0 for all R ∈ [ S, S  T ] . In particular, it holds that F ∈ F [ S,S  T ] ( k -1) . Thus, by Lemma 4.4, if F ∈ F L ( k ) ,

Figure 4: An illustration of Lemma 5.1 (left) and Lemma B.1 (right). If supp( F ) ⊆ L 2  L 3 , then we can match every S ∈ L 2 with a T ∈ L 3 such that F ( T ) = F ( S ) which implies ⟨ α ∅ ,abcde , F + ⟩ =  S ∈ L 2 F + ( S ) - T ∈ L 3 F + ( T ) = 0 . If F ( a ) &lt; 0 and F ( bcde ) &gt; 0 , then it holds that ⟨ α ∅ ,abcde , F ⟩ = ⟨ α ∅ ,bcde , F ⟩ = 0 .

<!-- image -->

it follows that F ∈ F [ S ′ ,S ′  T ] ( k -1) for all S ′ ⊆ Y \ T ′ . Since  S  is at most 2 k it follows by counting that if n ≥ ( k 2 + k +1) , the cardinality of T is at least ( k -1) 2 +( k -1) + 1 . This allows to apply the inductions hypothesis to all sublattices [ S ′ , S ′  T ] for S ′ ⊆ Y \ T , resulting in the following proposition. See also Figure 3b for an illustration of the induction.

Proposition 4.6. For k ∈ N , let L = [ X,Y ] be a lattice of rank n ≥ k 2 + k +1 and F ∈ F L ( k )  C L . Then it holds that ⟨ α X,Y , F + ⟩ = 0

Applying Proposition 4.6 to every sublattice of rank k 2 + k +1 allows to sharpen the bound.

Proposition 4.7. Let L be a lattice and k ∈ N , then it holds that A ( F L ( k )) ⊆ F L ( k 2 + k ) .

Translating this result back to the CPWL functions and applying the argument iteratively for a rank2 -maxout network, layer by layer, we obtain the following theorem.

Theorem 4.8. For a number of layers ℓ ∈ N , it holds that M 2 B d ( ℓ ) ⊆ V B d (2 2 ℓ -1 ) .

Corollary 4.9. The function x → 0 , x 1 ,    , x 2 2 ℓ -1  is not computable by a B 0 d -conforming ReLU neural network with ℓ hidden layers.

## 5 Combinatorial proof for dimension four

In this section, we prove that the function max  0 , x 1 ,    , x 4  cannot be computed by a B 0 d -conforming rank(2 , 2) -maxout networks or equivalently ReLU neural networks with 2 hidden layers. This completely class fi es the set of functions computable by B d -conforming ReLU neural networks with 2 hidden layers.

If L is a lattice of rank 5 and F ∈ F L (2)  C L , we know by Lemma 4.5, given that the supports of F are not empty, that there are X + ∈ L 2  supp + ( F ) and X -∈ L 2  supp -( F ) . We fi rst argue that in the special case of rank 5 we can even assume that there are X + ∈ L 1  supp + ( F ) and X -∈ L 1  supp -( F ) . Then, with analogous arguments as in Section 4, we prove that F + ∈ F L (4) , resulting in the sharp bound for rank(2 , 2) -maxout networks.

If the positive support of a function F ∈ F L (2)  C L is contained in the levels L 2 and L 3 , then for every S ∈ supp + ( F )  L 2 there must be a T ∈ supp + ( F )  L 3 such that T ⊇ S and F ( S ) ≤ F ( T ) since ⟨ α S,Y , F ⟩ = 0 . Applying the same argument to T , we conclude that F ( S ) = F ( T ) and that there are no further subsets in supp + ( F ) that are comparable to S or T . Thus, we can match the subsets S ∈ L 2 with the subsets T ∈ L 3 such that F ( S ) = F ( T ) and hence it follows that ⟨ α X,Y , F + ⟩ =  S ∈ L 2 F + ( S ) - T ∈ L 3 F + ( T ) = 0 . By symmetry, the same holds if supp -( F ) ⊆ L 2  L 3 . See Figure 4 for an illustration. Following this idea, we state the lemma for a more general case.

Lemma 5.1. Let L = [ X,Y ] be a lattice of rank n and F ∈ F L ( k )  C L with n ≥ 2 k + 1 . If there are i, j ∈ [ n ] 0 such that supp + ( F ) ⊆ L i  L j or supp -( F ) ⊆ L i  L j , then it holds that F + ∈ F L ( n -1) .

If there is a X + ∈ L 1  supp + ( F ) and a X -∈ L 4  supp -( F ) , then it holds that ⟨ α X,Y , F + ⟩ = ⟨ α X + ,Y , F ⟩ = 0 (Figure 4 and Lemma B.1 in the appendix). Thus we can assume that there are X + ∈ L 1  supp + ( F ) and X -∈ L 1  supp -( F ) . By proceeding analogously as in Section 4, we prove the following theorem.

Theorem 5.2. It holds that M 2 B d (2) = V B d (4) . In particular, the function x → 0 , x 1 ,    , x 4  is not computable by a B 0 d -conforming ReLU neural network with 2 hidden layers.

## 6 The unimaginable power of maxouts

By Proposition 3.2, all functions in V B d (  ℓ i =1 r i ) are representable by a B d -conforming rankr -maxout network. In Section 5, we have seen that this bound is tight for the rank vector (2 , 2) . In this section, we prove that this bound in general is not tight by demonstrating that the function x → 0 , x 1 ,    , x 6  is computable by a B 0 d -conforming rank(3 , 2) -maxout network.

Proposition 6.1. Let f 1 , f 2 ∈ V B 7 (3) be the functions given by

<!-- formula-not-decoded -->

Then it holds that max  f 1 , f 2  ∈ V B 7 (7) \ V B 7 (6) .

Proof Sketch. Let F 1 = Φ ( f 1 ) and F 2 = Φ ( f 2 ) . We write i 1 · · · i n for  i 1 ,    , i n  and i 1 · · · i n for [7] \  i 1 ,    , i n  and note that the sublattices [12 , 3] , [13 , 2] , [23 , 1] , [3 , 12] , [2 , 13] , [1 , 23] , [ ∅ , 123] , [123 , [7]] form a partition of [ ∅ , [7]] .

We fi rst show that on any of the above sublattices except [1 , 23] , either F 1 or F 2 attains the maximum on all elements of the sublattice and that for F := F 1 -F 2 it holds that supp + ( F ) ⊆ [1 , 23]  146  167 and ⟨ α [ ∅ , [7]] , F + ⟩ = ⟨ α 12 , 3 , F ⟩ -F (146) -F (167) = -2 and thus F + ∈ F L \ F L (6) . Then by looking at the partition into sublattices, we argue that F ∈ C L and thus by Lemma 3.3, we conclude that max  f 1 , f 2  ∈ V B 7 \ V B 7 (6) .

̸

Hence max  f 1 , f 2  =  M ⊆ [7] λ m σ M with λ [7] = 0 and since all functions in V B d (6) are computable by a rank(3 , 2) -maxout network, we conclude that x → x 1 ,    , x 7  is computable by a rank(3 , 2) -maxout network or equivalently:

Theorem 6.2. The function x → 0 , x 1 ,    , x 6  is computable by a rank(3 , 2) -maxout network.

Remark 6.3. One can check (e.g., with a computer) that x → 0 , x 1 ,    , x 6  is computable by a rank(3 , 2) -maxout network with integral weights. This is particularly interesting in light of Haase et al. [2023], who prove a ⌈ log 2 ( d +1) ⌉ lower bound for the case of integral weights and ReLU networks.

## 7 Conclusion and Limitations

Characterizing the set of functions that a ReLU network with a fi xed number of layers can compute remains an open problem. We established a doubly-logarithmic lower bound under the assumption that breakpoints lie on the braid fan. This assumption allowed us to exploit speci fi c combinatorial properties of the braid arrangement. In the speci fi c case of four dimensions, we reprove the tight bound for B 0 d -conforming networks of Hertrich et al. [2023] with combinatorial arguments. Given that Bakaev et al. [2025b] showed that one can compute the maximum of 5 numbers with 2 -layers, this implies that considering B 0 d -conforming networks is a real restriction. While this indicates that the doubly-logarithmic lower bound may not extend to all networks, our approach provides a foundation for adapting these techniques toward more general depth lower bounds, for example, by looking at different underlying fans instead of just the braid fan.

Acknowledgments Moritz Grillo was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project 464109215 within the priority programme SPP 2298 'Theoretical Foundations of Deep Learning,' and by Germany's Excellence Strategy - MATH+: The Berlin Mathematics Research Center (EXC-2046/1, project ID: 390685689). Part of this work

was completed while Christoph Hertrich was af fi liated with Université Libre de Bruxelles, Belgium, and received support by the European Union's Horizon Europe research and innovation program under the Marie Sk ł odowska-Curie grant agreement No 101153187-NeurExCo.

## References

- R. Arora, A. Basu, P. Mianjy, and A. Mukherjee. Understanding deep neural networks with recti fi ed linear units. In International Conference on Learning Representations , 2018.
- G. Averkov, C. Hojny, and M. Merkert. On the expressiveness of rational ReLU neural networks with bounded depth. In The Thirteenth International Conference on Learning Representations , 2025. URL https://openreview.net/forum?id=uREg3OHjLL .
- E. Bakaev, F. Brunck, C. Hertrich, D. Reichman, and A. Yehudayoff. On the depth of monotone relu neural networks and icnns. arXiv preprint arXiv:2505.06169 , 2025a.
- E. Bakaev, F. Brunck, C. Hertrich, J. Stade, and A. Yehudayoff. Better neural network expressivity: Subdividing the simplex, 2025b. URL https://arxiv.org/abs/2505.14338 .
- D. Bertschinger, C. Hertrich, P. Jungeblut, T. Miltzow, and S. Weber. Training fully connected neural networks is ∃ R-complete. In NeurIPS , 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html .
6. M.-C. Brandenburg, G. Loho, and G. Montufar. The real tropical geometry of neural networks for binary classi fi cation. Transactions on Machine Learning Research , 2024.
7. M.-C. Brandenburg, M. Grillo, and C. Hertrich. Decomposition polyhedra of piecewise linear functions. In The Thirteenth International Conference on Learning Representations , 2025. URL https://openreview.net/forum?id=vVCHWVBsLH .
- G. V. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems , 2:303-314, 1989.
- V. I. Danilov and G. A. Koshevoy. Cores of cooperative games, superdifferentials of functions, and the minkowski difference of sets. Journal of Mathematical Analysis and Applications , 247(1):1-14, 2000. ISSN 0022-247X. doi: 10.1006/jmaa.2000.6756. URL https: //www.sciencedirect.com/science/article/pii/S0022247X00967568 .
- R. Eldan and O. Shamir. The power of depth for feedforward neural networks. In V. Feldman, A. Rakhlin, and O. Shamir, editors, 29th Annual Conference on Learning Theory , volume 49 of Proceedings of Machine Learning Research , pages 907-940, Columbia University, New York, New York, USA, 23-26 Jun 2016. PMLR. URL https://proceedings.mlr.press/v49/ eldan16.html .
- E. Ergen and M. Grillo. Topological expressivity of relu neural networks. In S. Agrawal and A. Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 1599-1642. PMLR, 30 Jun-03 Jul 2024. URL https://proceedings.mlr.press/v247/ergen24a.html .
- V. Froese and C. Hertrich. Training neural networks is np-hard in fi xed dimension. In Proceedings of the 37th International Conference on Neural Information Processing Systems , NIPS '23, Red Hook, NY, USA, 2023. Curran Associates Inc.
- V. Froese, C. Hertrich, and R. Niedermeier. The computational complexity of relu network training parameterized by data dimensionality. Journal of Arti  cial Intelligence Research , 74:1775-1790, 2022.
- V. Froese, M. Grillo, C. Hertrich, and M. Stargalla. Parameterized hardness of zonotope containment and neural network veri fi cation. arXiv preprint arXiv:2509.22849 , 2025a.

- V. Froese, M. Grillo, and M. Skutella. Complexity of injectivity and veri fi cation of relu neural networks (extended abstract). In N. Haghtalab and A. Moitra, editors, Proceedings of Thirty Eighth Conference on Learning Theory , volume 291 of Proceedings of Machine Learning Research , pages 2188-2189. PMLR, 30 Jun-04 Jul 2025b. URL https://proceedings.mlr.press/ v291/froese25a.html .
- X. Glorot, A. Bordes, and Y. Bengio. Deep sparse recti fi er neural networks. In G. Gordon, D. Dunson, and M. Dudík, editors, Proceedings of the Fourteenth International Conference on Arti  cial Intelligence and Statistics , volume 15 of Proceedings of Machine Learning Research , pages 315323, Fort Lauderdale, FL, USA, 11-13 Apr 2011. PMLR. URL https://proceedings. mlr.press/v15/glorot11a.html .
- S. Goel, A. Klivans, P. Manurangsi, and D. Reichman. Tight Hardness Results for Training Depth-2 ReLU Networks. In J. R. Lee, editor, 12th Innovations in Theoretical Computer Science Conference (ITCS 2021) , volume 185 of Leibniz International Proceedings in Informatics (LIPIcs) , pages 22:1-22:14, Dagstuhl, Germany, 2021. Schloss Dagstuhl - Leibniz-Zentrum für Informatik. ISBN 978-3-95977-177-1. doi: 10.4230/LIPIcs.ITCS.2021.22. URL https://drops.dagstuhl. de/entities/document/10.4230/LIPIcs.ITCS.2021.22 .
- I. J. Goodfellow, Y. Bengio, and A. Courville. Deep Learning . MIT Press, Cambridge, MA, USA, 2016. http://www.deeplearningbook.org .
- C. A. Haase, C. Hertrich, and G. Loho. Lower bounds on the depth of integral ReLU neural networks via lattice polytopes. In International Conference on Learning Representations , 2023.
- C. Hertrich and G. Loho. Neural networks and (virtual) extended formulations. arXiv preprint arXiv:2411.03006 , 2024.
- C. Hertrich and L. Sering. Relu neural networks of polynomial size for exact maximum fl ow computation. Mathematical Programming , pages 1-30, 2024.
- C. Hertrich and M. Skutella. Provably good solutions to the knapsack problem via neural networks of bounded size. INFORMS journal on computing , 35(5):1079-1097, 2023.
- C. Hertrich, A. Basu, M. Di Summa, and M. Skutella. Towards lower bounds on the depth of relu neural networks. SIAM Journal on Discrete Mathematics , 37(2):997-1029, 2023. doi: 10.1137/22M1489332. URL https://doi.org/10.1137/22M1489332 .
- K. Hornik. Approximation capabilities of multilayer feedforward networks. Neural Networks , 4: 251-257, 1991.
- J. Huchette, G. Muñoz, T. Serra, and C. Tsay. When deep learning meets polyhedral theory: A survey. arXiv preprint arXiv:2305.00241 , 2023.
- K. Jochemko and M. Ravichandran. Generalized permutahedra: Minkowski linear functionals and ehrhart positivity. Mathematika , 68(1):217-236, 2022. doi: 10.1112/mtk.12122. URL https: //londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/mtk.12122 .
- G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer. Reluplex: An ef fi cient smt solver for verifying deep neural networks. In R. Majumdar and V. Kun ˇ cak, editors, Computer Aided Veri  cation , page 97-117, Cham, 2017. Springer International Publishing. ISBN 978-3-319-633879.
- S. Khalife and A. Basu. Neural networks with linear threshold activations: Structure and algorithms. In K. Aardal and L. Sanità, editors, Integer Programming and Combinatorial Optimization , page 347-360, Cham, 2022. Springer International Publishing. ISBN 978-3-031-06901-7.
- J. Li, J. Liu, P. Yang, L. Chen, X. Huang, and L. Zhang. Analyzing deep neural networks with symbolic propagation: Towards higher precision and faster veri fi cation. In B.-Y. E. Chang, editor, Static Analysis , page 296-319, Cham, 2019. Springer International Publishing. ISBN 978-3-030-32304-2.
- P. Maragos, V. Charisopoulos, and E. Theodosis. Tropical geometry and machine learning. Proceedings of the IEEE , 109(5):728-755, 2021.

- P. Misiakos, G. Smyrnis, G. Retsinas, and P. Maragos. Neural network approximation based on hausdorff distance of tropical zonotopes. In International Conference on Learning Representations , 2022.
- G. Montúfar, R. Pascanu, K. Cho, and Y. Bengio. On the number of linear regions of deep neural networks. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2 , NIPS'14, page 2924-2932, Cambridge, MA, USA, 2014. MIT Press.
- G. Montúfar, Y. Ren, and L. Zhang. Sharp bounds for the number of regions of maxout networks and vertices of minkowski sums. SIAM Journal on Applied Algebra and Geometry , 6(4):618-649, 2022.
- A. Mukherjee and A. Basu. Lower bounds over boolean inputs for deep neural networks with relu gates. arXiv preprint arXiv:1711.03073 , 2017.
- I. Safran, D. Reichman, and P. Valiant. How many neurons does it take to approximate the maximum? In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 3156-3183. SIAM, 2024.
- A. Schrijver. Theory of Linear and Integer programming . Wiley-Interscience, 1986.
- R. Stanley. An introduction to hyperplane arrangements , pages 389-496. 10 2007. ISBN 9780821837368. doi: 10.1090/pcms/013/08.
- M. Stargalla, C. Hertrich, and D. Reichman. The computational complexity of counting linear regions in relu neural networks. arXiv preprint arXiv:2505.16716 , 2025.
- M. Telgarsky. bene fi ts of depth in neural networks. In V. Feldman, A. Rakhlin, and O. Shamir, editors, 29th Annual Conference on Learning Theory , volume 49 of Proceedings of Machine Learning Research , pages 1517-1539, Columbia University, New York, New York, USA, 23-26 Jun 2016. PMLR. URL https://proceedings.mlr.press/v49/telgarsky16.html .
- J. L. Valerdi. On minimal depth in neural networks. arXiv preprint arXiv:2402.15315 , 2024.
- S. Wang and X. Sun. Generalization of hinging hyperplanes. IEEE Transactions on Information Theory , 51(12):4425-4431, 2005.
- L. Zhang, G. Naitzat, and L.-H. Lim. Tropical geometry of deep neural networks. In International Conference on Machine Learning , pages 5824-5832. PMLR, 2018.
- G. M. Ziegler. Lectures on Polytopes . Springer New York, May 2012. ISBN 038794365X. URL https://www.ebook.de/de/product/3716808/guenter\_m\_ ziegler\_lectures\_on\_polytopes.html .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately re fl ect the paper's contributions and scope?

Answer: [Yes]

Justi fi cation: All claimed results are proven in the main part or appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and re fl ect how much the results can be expected to generalize to other settings.
- It is fi ne to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justi fi cation: We clearly and openly discuss the assumptions and limitations of our theorems in the introduction and the theorem statements.

## Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-speci fi cation, asymptotic approximations only holding locally). The authors should re fl ect on how these assumptions might be violated in practice and what the implications would be.
- The authors should re fl ect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should re fl ect on the factors that in fl uence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational ef fi ciency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be speci fi cally instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justi fi cation: All assumptions are mentioned in the statements. All proofs are given in the main text or appendix.

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

Justi fi cation: No experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or veri fi able.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suf fi ce, or if the contribution is a speci fi c model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
5. Open access to data and code

Question: Does the paper provide open access to the data and code, with suf fi cient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justi fi cation: No experiments.

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

Justi fi cation: No experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical signi  cance

Question: Does the paper report error bars suitably and correctly de fi ned or other appropriate information about the statistical signi fi cance of the experiments?

Answer: [NA]

Justi fi cation: No experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, con fi -dence intervals, or statistical signi fi cance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not veri fi ed.
- For asymmetric distributions, the authors should be careful not to show in tables or fi gures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding fi gures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide suf fi cient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justi fi cation: No experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justi fi cation: Purely theoretical research.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justi fi cation: Purely theoretical research.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake pro fi les, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact speci fi c groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the ef fi ciency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justi fi cation: Purely theoretical research.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety fi lters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justi fi cation: No assets used.

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

Justi fi cation: No new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip fi le.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justi fi cation: No crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fi ne, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justi fi cation: No crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary signi fi cantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scienti fi c rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justi fi cation: No non-standard LLM usage.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.
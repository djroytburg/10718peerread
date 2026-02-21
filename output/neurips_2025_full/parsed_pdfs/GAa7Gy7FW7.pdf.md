## Optimal Minimum Width for the Universal Approximation of Continuously Differentiable Functions by Deep Narrow MLPs

## Geonho Hwang

Department of Mathematical Sciences Gwangju Institute of Science and Technology Gwangju, Buk-gu 61005

hgh2134@gist.ac.kr

## Abstract

In this paper, we investigate the universal approximation property of deep, narrow multilayer perceptrons (MLPs) for C 1 functions under the Sobolev norm, specifically the W 1 , ∞ norm. Although the optimal width of deep, narrow MLPs for approximating continuous functions has been extensively studied, significantly less is known about the corresponding optimal width for C 1 functions. We demonstrate that the optimal width can be determined in a wide range of cases within the C 1 setting. Our approach consists of two main steps. First, leveraging control theory, we show that any diffeomorphism can be approximated by deep, narrow MLPs. Second, using the Borsuk-Ulam theorem and various results from differential geometry, we prove that the optimal width for approximating arbitrary C 1 functions via diffeomorphisms is min( n + m, max(2 n +1 , m )) in certain cases, including ( n, m ) = (8 , 8) and (16 , 8) , where n and m denote the input and output dimensions, respectively. Our results apply to a broad class of activation functions.

## 1 Introduction

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The choice of neural network architecture plays a crucial role in determining performance. However, in practice, architectural decisions are often made through trial and error. It is therefore important to provide theoretical guidance on what should be avoided and how to select appropriate width and depth based on the input space, target function, and specific tasks. The universal approximation property (UAP) refers to the ability of deep learning models to approximate a given class of functions. Since deep networks must approximate general functions to perform specific tasks, the UAP has received considerable attention as a theoretical foundation. While various forms of universal approximation theorems exist depending on the network type and its characteristics, one actively studied setting is the universal approximation property of deep, narrow multilayer perceptrons (deep, narrow MLPs), which reflects the practical scenario where networks are deep but relatively narrow in width.

MLPs with fixed width and arbitrarily large depth exhibit different universal approximation behavior depending on whether their width exceeds a critical threshold (Johnson, 2018; Kidger &amp; Lyons, 2020). This threshold is called the minimum width , and numerous studies have investigated upper and lower bounds for this threshold based on the input dimension n , output dimension m , and the choice of activation function.

The most intensively studied case involves the approximation of continuous functions under the uniform norm. Notable results include the upper bound n + m + c ( σ ) , where c ( σ ) is a constant depending on the activation function, shown by Hanin &amp; Sellke (2017); Kidger &amp; Lyons (2020). More recently, Hwang (2023) improved this upper bound to max(2 n +1 , n ) .

For lower bounds, Johnson (2018); Cai (2022); Kim et al. (2023) proved that the minimum width must be at least n +1 or m + 1 d&lt;m ≤ 2 n , depending on the setting. However, few studies have succeeded in narrowing the gap between known lower and upper bounds. Among the few, Park et al. (2020); Hwang (2023) proved optimality in specific cases: the minimum width is 3 for ( n, m ) = (1 , 2) and 4 for (2 , 2) .

Beyond the uniform norm, there has also been research under other norms. Park et al. (2020) established the optimal minimum width of deep, narrow MLPs with ReLU activation in the L p norm. However, research on general norms beyond the L p and uniform norms remains limited.

However, there has been a scarce number of papers that study norms involving derivatives of functions in the setting of deep narrow MLPs. Many deep learning techniques directly penalize the difference between the derivative of the target function and that of the network. These include Sobolev Training (Czarnecki et al., 2017), Physics-Informed Neural Networks (PINNs) (Raissi et al., 2019), and Generative Adversarial Networks with gradient penalty (Gulrajani et al., 2017; Arbel et al., 2018).

In this work, we determine the minimum width required to approximate continuously differentiable functions in Sobolev spaces. Specifically, we focus on approximation with respect to the W 1 , ∞ norm. Compared to the uniform norm, the topology of the Sobolev norm W 1 , ∞ is finer, enabling tighter lower bound estimates. For the upper bound, tools from control theory allow us to match the upper bounds known in the uniform norm setting. Using these ideas, we compute both upper and lower bounds for approximation in Sobolev spaces. In some cases, the lower bound coincides with the upper bound, thus identifying the optimal minimum width. This includes interesting cases such as (8 , 8) and (16 , 8) . The exact pairs to which our result applies can be found in Theorem 5.9.

Our contributions are as follows:

- We show that deep, narrow MLPs can approximate arbitrary diffeomorphisms with respect to the Sobolev norm W 1 , ∞ . (Theorem 4.1)
- Weprecisely characterize the additional width required to approximate arbitrary continuously differentiable functions as compositions of diffeomorphisms and linear transformations. (Definition 4.3 and Theorem 4.6)
- Using these results, we prove that the known upper bounds n + m and max(2 n +1 , m ) under the uniform norm also hold under the Sobolev norm W 1 , ∞ . (Theorem 5.1)
- We prove that these upper bounds are also lower bounds for infinitely many combinations of n and m , and therefore, these values represent the optimal minimum width in those cases. (Theorem 5.9)

## 2 Related Words

In this section, we review previous studies on the universal approximation property (UAP). Cybenko (1989) proved that a two-layer MLP possesses the UAP in the space of continuous functions. This result was extended by Leshno et al. (1993) to more general activation functions.

While these early results focus on two-layer networks, subsequent research has investigated the UAP of deep, narrow MLPs. Hanin &amp; Sellke (2017) established a universal approximation theorem for deep, narrow MLPs with ReLU activation, providing both lower and upper bounds on the minimum width. Johnson (2018) showed that a width of at least n +1 is required for networks with monotonic activation functions. Kidger &amp; Lyons (2020) proved that a width of n + m +1 suffices for general non-polynomial activation functions, while n + m +2 is sufficient for polynomial activations. Park et al. (2020) demonstrated that the optimal minimum width is three when n = 1 and m = 2 with ReLU. Cai (2022) showed that a width of at least max( n, m ) is necessary for general activation functions. Kim et al. (2023) proved a lower bound of m + 1 n&lt;m ≤ 2 m . Hwang (2023) established an upper bound of max(2 n +1 , m ) for networks using the Leaky-ReLU activation and showed that the optimal minimum width is four when n = m = 2 .

There have also been investigations of the UAP under norms other than the uniform norm. Park et al. (2020) showed that the optimal minimum width is max( n + 1 , m ) in the L p ( R n , R m ) space for ReLU networks. Additionally, Kim et al. (2024) demonstrated that in the L p ([0 , 1] n , R m ) setting, the optimal minimum width becomes min( n, m, 2) .

In addition to studies on MLPs, there has been significant progress in understanding the universal approximation property of residual networks (ResNets). Lin &amp; Jegelka (2018) demonstrated that even ResNets with one-neuron hidden layers can serve as universal approximators, highlighting the expressive power that arises from their residual structure. Aizawa et al. (2020) extended this line of research by analyzing both ResNets and ODENets, providing rigorous mathematical results along with supporting numerical experiments. More recently, Tabuada &amp; Gharesifard (2022) investigated ResNets from a control-theoretic perspective.

Beyond function value approximation, some universal approximation theorems also consider derivatives. Li (1996) proved that a two-layer MLP can approximate arbitrary derivatives of a function, provided the activation function is sufficiently smooth.

However, these results do not cover the Sobolev norm in the context of deep narrow MLPs. In this paper, we provide a partial answer to this open question by establishing results under the W 1 , ∞ norm.

## 3 Notation and Definition

In this section, we introduce the notations and definitions used throughout this paper: N denotes the set of natural numbers, and N 0 = N ∪ { 0 } . B n ( r ) denotes the open ball in R n centered at the origin with radius r . For a set A ⊂ R d , A denotes the closure of A with respect to the Euclidean norm.

For two open sets V ⊂ U ⊂ R d , we say that V is a precompact subset of U if V ⊂ R d is compact and V ⊂ U . We denote this as V ⋐ U . For sets A,B ⊂ R d , the Minkowski sum is defined as A + B = { x + y ∈ R d | x ∈ A, y ∈ B } .

For a d -dimensional vector x ∈ R d , we denote by x i the i -th component of x ; in other words, x = ( x 1 , x 2 , . . . , x d ) . Similarly, for a function f : X → R n , we write f i to denote the i -th component function, so that f ( x ) = ( f 1 ( x ) , . . . , f n ( x )) . We use x i : j to represent the ( j -i +1) -dimensional subvector ( x i , x i +1 , . . . , x j ) . For vectors x, y ∈ R d , the dot product is denoted by x · y ∈ R and defined as x · y := ∑ d i =1 x i y i . For vectors x = ( x 1 , . . . , x d 1 ) ∈ R d 1 and y = ( y 1 , . . . , y d 2 ) ∈ R d 2 , we define the operation ⊕ as x ⊕ y := ( x 1 , . . . , x d 1 , y 1 , . . . , y d 2 ) ∈ R d 1 + d 2 . Similarly, for functions f : X → R d 1 and g : X → R d 2 , we define f ⊕ g : X → R d 1 + d 2 by ( f ⊕ g )( x ) := f ( x ) ⊕ g ( x ) .

Let Aff n,m denote the set of affine transformations from R n to R m . For a function f : X → Y and a subset X ′ ⊂ X , we write f | X ′ to denote the restriction of f to the domain X ′ . For r ∈ N 0 , the space C r ( X ; Y ) denotes the set of functions that are r -times continuously differentiable. For U ⊂ R n and r = ( r 1 , . . . , r n ) ∈ N n 0 , the space C r ( U ; R m ) consists of functions f such that the mixed partial derivative ∂ r 1 + ··· + rn f ∂x r 1 1 ...∂x rn n exists and is continuous. For k ∈ N 0 and r ∈ [0 , 1] , the space C k,r ( U ; R m ) consists of functions whose k -th order partial derivatives are Hölder continuous with exponent r . In particular, C 0 , 1 ( U ; R m ) denotes the space of Lipschitz continuous functions. We define C 0 , 1 loc ( U ; R m ) as the space of locally Lipschitz continuous functions: that is, f ∈ C 0 , 1 loc ( U ; R m ) if for every precompact set V ⋐ U , there exists a constant L V such that ∥ f ( x ) -f ( y ) ∥ ≤ L V ∥ x -y ∥ for all x, y ∈ V . We denote the Lipschitz constant of f on V by L V ( f ) .

## 3.1 Sobolev Space

We define the Sobolev space as follows: We denote the weak derivative of u by Du .

Definition 3.1 (Sobolev Space) . Let n, k ∈ N , p ∈ N ∪{∞} , and let U ⊂ R n be an open set. The Sobolev space W k,p ( U ) is defined by

<!-- formula-not-decoded -->

equipped with the norm

<!-- formula-not-decoded -->

The vector-valued Sobolev space W k,p ( U ; R m ) for m,k ∈ N is defined as

<!-- formula-not-decoded -->

with the norm

<!-- formula-not-decoded -->

More specifically, we focus on the following local Sobolev space, considering compact domains: Definition 3.2 (Local Sobolev Space) . Let U ⊂ R m , r ∈ N 0 , and p ∈ [1 , ∞ ] . The local Sobolev space W r,p loc ( U ; R n ) is defined as the projective limit:

<!-- formula-not-decoded -->

where the right-hand side is given explicitly as

<!-- formula-not-decoded -->

The local Sobolev space is equipped with the relative topology inherited from the product topology of the spaces W r,p ( V ; R n ) .

In this paper, we focus on the Sobolev norm W 1 , ∞ . It is well known that W 1 , ∞ loc = C 0 , 1 loc . See Theorem 4.5, p.155 in Evans (2018) for details. It is also known that for convex domains, the Sobolev and Lipschitz spaces coincide (Theorem 4.1 in Heinonen (2005)): if V ⊂ R d is convex, then W 1 , ∞ ( V ) = C 0 , 1 ( V ) . Moreover, there exist constants C 1 , C 2 &gt; 0 depending only on d and n such that (see Theorem 4, p.279 and Theorem 6, p.281 in Evans (2022)):

<!-- formula-not-decoded -->

For convenience, we will always take the continuous representative among functions that differ only on a set of Lebesgue measure zero.

For a set of functions A ⊂ W 1 ,p ( U ; R m ) , we denote by A W 1 , ∞ = A the closure of A with respect to the norm ∥ · ∥ W 1 ,p ( U ; R m ) . Similarly, for a set of functions A ⊂ W 1 ,p loc ( U ; R m ) , we denote the closure in the local Sobolev topology by A W 1 , ∞ loc = A loc .

## 3.2 Activation Function

We adopt the commonly used condition on activation functions, as proposed by Kidger &amp; Lyons (2020).

̸

Condition 1. There exist constants α ∈ R and ϵ ∈ R + such that a nonlinear activation function σ is a C 1 function on the interval ( α -ϵ, α + ϵ ) , and σ ′ ( α ) = 0 .

The ReLU activation function is defined as

<!-- formula-not-decoded -->

and the Leaky-ReLU activation function is defined as

<!-- formula-not-decoded -->

We consider MLPs with sets of activation functions. For example, MLPs with the Leaky-ReLU activation function select an activation function from the following set at each layer:

<!-- formula-not-decoded -->

We use the symbols σ and Σ to denote an activation function and a set of activation functions, respectively. We define Leaky-ReLU-like activation functions as follows:

Definition 3.3 (Leaky-ReLU-like) . A set of activation functions Σ is called Leaky-ReLU-like if and only if for each β ∈ R + , there exists a C 1 activation function σ β ∈ Σ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also denote the identity function by id :

and

<!-- formula-not-decoded -->

Activation functions applied to vectors operate componentwise. For a set of activation functions Σ , define Σ d as

## 3.3 Deep Narrow MLP

We define the set of deep, narrow MLPs with a set of activation functions Σ , arbitrary depth, input dimension n , output dimension m , and at most w intermediate dimensions as ∆ Σ n,m,w . (The exact definition is provided in Appendix A.1.) For a singleton activation function σ , we define:

<!-- formula-not-decoded -->

For natural numbers n ≥ m ∈ N , we define the natural projection p n,m : R n → R m and the zero-padding inclusion q m,n : R m → R n as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Any function f ∈ ∆ Σ n,m,w can be decomposed as:

<!-- formula-not-decoded -->

where g ∈ ∆ Σ w,w,w . Note that if g 1 , g 2 ∈ ∆ Σ w,w,w , then their composition g 1 ◦ g 2 also belongs to ∆ Σ w,w,w .

From this point on, we will use the notation σ to refer to either a single activation function or a set of activation functions Σ , depending on the context.

## 3.4 Subsets of Diffeomorphisms

We define the sets of diffeomorphisms. For definitions of concepts from differential geometry, see Appendix A.2.

Definition 3.4 (Diffeomorphism: D r ( U ) ) . Let U ⊂ R d be an open subset, and let r be a non-negative integer or infinity. Then D r ( U ) denotes the set of C r -diffeomorphisms from U to R d .

## 4 Universal Approximation

## 4.1 Problem Formulation

Our primary goal is to identify the minimum width w W 1 , ∞ min ∈ N such that any continuously differentiable function f ∈ C 1 ( R n ; R m ) can be approximated by elements of ∆ σ n,m,w W 1 , ∞ min in the topology of W 1 , ∞ loc ( R n ; R m ) . In other words, our aim is to determine the value of w W 1 , ∞ min ( n, m, σ ) such that

<!-- formula-not-decoded -->

w W 1 , ∞ min ( n, m, σ ) denotes the minimum width for which MLPs of this width and arbitrary depth can approximate C 1 functions to any accuracy in the W 1 , ∞ norm.

<!-- formula-not-decoded -->

## 4.2 Diffeomorphisms and Continuously Differentiable Functions

Our proof strategy is divided into two parts. First, we approximate a diffeomorphism using deep narrow MLPs with a small additional width. Next, we show that any continuously differentiable function can be approximated by a composition of affine transformations and diffeomorphisms, and we rigorously estimate the required width. In this subsection, we aim to prove the following theorem, which asserts that a diffeomorphism can be approximated by deep narrow MLPs.

Theorem 4.1. Let σ be one of a non-polynomial C 1 , 1 -function, LR, ReLU, or Leaky-ReLU-like activation functions. Then, for any natural number d ∈ N , the following relation holds:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The theorem states that deep, narrow MLPs with a small additional width can approximate arbitrary diffeomorphisms. The proof relies on techniques from control theory. Approximating an entire diffeomorphism directly using a neural network is challenging. To address this, we interpret a diffeomorphism as the solution of an ordinary differential equation evolving over time. In other words, the existence of a diffeomorphism implies the existence of a continuous flow connecting the identity map to the diffeomorphism. The direction and magnitude of this flow are determined by a vector field. The problem then reduces to approximating this continuous flow step by step, which is equivalent to approximating a vector field. Deep, narrow MLPs can approximate such flows over sufficiently small time intervals by leveraging the universal approximation property. Then, by approximating the flow generated by this two-layer MLP using a deep narrow MLP, we complete the argument. The full proof is provided in Appendix C.1.

Now, we introduce a quantity Ω( n, m ) such that any continuously differentiable function from [0 , 1] n to R m can be approximated by a composition of affine transformations and Ω( n, m ) -dimensional diffeomorphisms. We further show that this width is optimal. To this end, we begin with the following lemma.

Lemma4.2 (Theorem C of Palais (1960)) . Let n, m ∈ N with n ≤ m , and let f : K = [0 , 1] n → R m be a smooth embedding. Then, there exists a smooth diffeomorphism F : R m → R m such that the following equation holds:

<!-- formula-not-decoded -->

The lemma implies that any smooth embedding can be decomposed into an affine transformation followed by a diffeomorphism. Now, let Emb( X,Y ) denote the set of smooth embeddings from X to Y . We define the quantity Ω( n, m ) as follows:

Definition 4.3 ( Ω( n, m ) ) .

<!-- formula-not-decoded -->

where the closure is taken with respect to the C 1 -norm.

Using the lemma above and the definition of Ω( n, m ) , we state the following theorem:

Theorem 4.4. Let σ be one of a non-polynomial C 1 , 1 -function, LR, ReLU, or Leaky-ReLU-like activation functions. Then, for any natural numbers n and m , the following relation holds:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The proof of the theorem is provided in Appendix D.1. The preceding theorem shows that Ω( n, m ) is a sufficient width for approximating functions with n -dimensional input and m -dimensional output. Conversely, the following proposition demonstrates that Ω( n, m ) is also a necessary width for such approximation.

Proposition 4.5. Let σ be a set of non-decreasing, C 1 activation functions. Then, for natural numbers n and m , the following relation holds:

<!-- formula-not-decoded -->

The proof of this proposition is provided in Appendix D.2. By combining the previous theorems with this proposition, we derive the following result. This theorem demonstrates that the purely geometrically defined quantity Ω( n, m ) has a fundamental connection to the minimum width of deep narrow MLPs.

Theorem 4.6. The following relation holds:

<!-- formula-not-decoded -->

for a Leaky-ReLU-like σ in which every element is increasing, and

<!-- formula-not-decoded -->

for a set of C 1 , 1 increasing activation functions σ .

## 5 Calculation of Ω( n, m )

In the previous section, we showed that Ω( n, m ) nearly determines the minimum width required for universal approximation. In this section, we provide general bounds for Ω( n, m ) and compute exact values for specific cases.

## 5.1 Upper Bound of Ω( n, m )

We begin by establishing the following general upper bound.

Theorem 5.1. The following relation holds:

<!-- formula-not-decoded -->

Proof. The inequality Ω( n, m ) ≤ n + m follows directly from the definition of Ω( n, m ) . Ω( n, m ) ≤ max(2 n +1 , m ) is by Lemma 5.2.

Lemma 5.2. Consider natural numbers n and m where m &gt; 2 n . Let f ∈ C 1 ( R n ; R m ) be a continuously differentiable function. Then, for a bounded open set U ⊂ R n and a positive number ϵ ∈ R + , there exists a smooth embedding g : U → R m such that

<!-- formula-not-decoded -->

Proof. This is a direct consequence of the transversality theorem. (See Chapter 3, Theorem 2.1 of Hirsch (2012) for details)

In some cases, we can improve the general bound established above.

Lemma 5.3. For even k , the following eqaution holds:

<!-- formula-not-decoded -->

Proof. By Kim et al. (2023), we have Ω( k, 2 k -1) ≥ 2 k . Thus, it suffices to prove that Ω( k, 2 k -1) ≤ 2 k . As immersions are dense in C 1 ( R k , R 2 k -1 ) , it is enough to approximate an immersion f . By Corollary 3.2 of Lashof &amp; Smale (1959), there exists a smooth embedding g such that ∥ p 2 k, 2 k -1 ◦ g -f ∥ W 1 , ∞ ( U ; R m ) &lt; ϵ . Note that while the original result is stated for the uniform norm, the same proof applies directly in the C 1 norm setting.

## 5.2 Lower Bound of Ω( n, m )

In this subsection, we present a lower bound for certain cases, which coincides with the upper bound established in the previous section, thereby yielding the optimal minimum width. To prove the lower bound, we require an argument of the following form: Given a function f : R n → R m , there exists ϵ &gt; 0 such that if the codomain dimension of another function g is small, then the concatenation f ⊕ g cannot be an embedding. To this end, we construct a function f whose self-intersection is transversal and has the structure of a sphere S r . If all antipodal points on the sphere are mapped to the same value by f , then we can apply the Borsuk-Ulam theorem.

Lemma 5.4 (Borsuk-Ulam Theorem) . Let h : S n → R n be a continuous function. Then there exists a point x ∈ S n such that

<!-- formula-not-decoded -->

The Borsuk-Ulam theorem states that every continuous map from an n -dimensional sphere to R n maps some pair of antipodal points to the same point. Now, suppose we have an embedding S r ↪ → R n and a map f such that f ( x ) = f ( -x ) for all antipodal points x ∈ S r . Then, for any function g : R r → R r , there exists a pair of antipodal points on S r that are mapped to the same value by g . Therefore, the map G = f ⊕ g cannot be injective, and hence cannot be an embedding. This leads to the conclusion Ω( n, m ) ≥ m + r +1 .

The difficulty, however, lies in the fact that we must consider a map G such that ∥ p m + r,n ◦ G -f ∥ is small, rather than requiring exact equality p m + r,n ◦ G = f . The following lemma guarantees that the diffeomorphic structure of the self-intersection is preserved under small perturbations in the C 1 norm.

Lemma 5.5 (Ehresmann's Lemma for Intersection) . For n, m ∈ N with 2 n &gt; m , consider a precompact set U ⊂ R n and a C 1 function f : U → R m in transversal position. Then there exists ϵ &gt; 0 such that the following holds: Consider arbitrary g ∈ C 1 ( U ; R m ) satisfying

<!-- formula-not-decoded -->

Define the diagonal ∆ of U × U as

<!-- formula-not-decoded -->

Define ˜ f : U × U -∆ → R m as

Similarly, define ˜ g as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, there exists a C 1 diffeomorphism Φ : ˜ f -1 (0) → ˜ g -1 (0) such that

<!-- formula-not-decoded -->

where T denotes the involution ( x, y ) ↦→ ( y, x ) .

The proof of Lemma 5.5 is provided in Appendix E.1. Now, the only remaining task is to construct a function with such a self-intersection structure. The following two lemmas provide results for specific cases.

Lemma 5.6. Assume that there exists a submersion f : RP n -1 × ( -1 , 1) → R m . Then, the following relation holds:

<!-- formula-not-decoded -->

Proof. Let f : RP n -1 × ( -1 , 1) → R m be a submersion. Then, there exists a lifting ˜ f : S n -1 × ( -1 , 1) → R m such that for a canonical two-to-one covering p : S n -1 × ( -1 , 1) → RP n -1 × ( -1 , 1) , we have p ◦ ˜ f = f . Then, as all antipodal points have the same ˜ f values and the intersection is transversal, it follows from the previous arguments that Ω( n, m ) ≥ n + m . This completes the proof.

Lemma 5.7 (Projective Space Submersion Lemma) . For n ∈ N , consider a, b, c ∈ N 0 such that n +1 = 2 4 a + b × c , where 0 ≤ b ≤ 3 and c is an odd number. Then, for any natural number m ∈ N satisfying m ≤ 8 a +2 b , RP n × ( -1 , 1) can be submerged into R m .

Proof. By Theorem B of Phillips (1967), there exists a submersion M → R m if and only if there exists a section in F m ( M ) where F m ( M ) is the bundle of m -frames tangent to M . This condition is equivalent to the existence of m linearly independent vector fields. By Theorem 1.1 of Davis (2012), the maximum number of linearly independent vector fields on RP n equals 8 a +2 b -1 where n +1 = 2 4 a + b × c for 0 ≤ b ≤ 3 and an odd number c ∈ N . Therefore, RP n × ( -ϵ, ϵ ) has 8 a +2 b independent vector fields, and thus can be submerged into R 8 a +2 b .

We can also verify that an immersion with an RP n -structure intersection exists when 3 n +1 &lt; 2 m ≤ 4 n , which implies that Ω( n, m ) = 2 n +1 .

Lemma 5.8 (Theorem 3 of Miller (1969)) . Given n and m, 3 n + 1 &lt; 2 m ≤ 4 n -2 . Then if n +1 ∼ = 0(mod c 2 n -m ) there exists a transversal immersion S n → R m +1 with self-intersection RP 2 n -m -1 . Here, c m is defined as

<!-- formula-not-decoded -->

By combining all the results, we obtain the following theorem.

Theorem 5.9. If 2 4 a + b | n for a, b ∈ N 0 satisfying 0 ≤ b ≤ 3 and m ≤ 8 a +2 b ,

<!-- formula-not-decoded -->

If 3 n +3 2 &lt; m ≤ 2 n and n +1 ∼ = 0(mod c 2 n -m +1 ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 5.10. At first glance, the dependence of the optimal minimum width on the parity of the input and output dimensions may appear somewhat artificial. However, Lemma 5.3 and Theorem 5.9 together yield the relation

<!-- formula-not-decoded -->

which provides strong evidence that this parity dependence may be a fundamental property.

## 6 Limitation

Although our results yield optimal values in many cases-such as Ω(8 , 8) = 16 and Ω(16 , 8) = 24 -they do not apply to all combinations of n and m . Our analysis is asymptotically valid primarily when m is either much smaller than n or significantly larger, specifically in the regime where 3 n &lt; 2 m . Developing a theoretical framework that addresses the intermediate regime not covered by our theory would be a compelling direction for future research. Furthermore, determining the exact lower bound in cases where n +1 is not divisible by a power of 2 remains an open and intriguing problem.

Also, our analysis is non-constructive and asymptotic in nature. In particular, we do not provide explicit rates of approximation or quantitative bounds on the depth required for a network to achieve a given precision. As a result, our results establish existence guarantees but leave open the practical question of how deep a network must be to approximate a target function within a prescribed accuracy. This limitation stands in contrast to constructive approximation results that do provide explicit dependence on approximation error.

Furthermore, our work does not characterize the role of the smoothness of the target function in the approximation behavior. It is natural to expect that smoother functions should be easier to approximate, and indeed prior studies have demonstrated this by analyzing approximation rates in terms of Sobolev smoothness classes (Schmidt-Hieber, 2020). Incorporating such smoothnessdependent considerations into the analysis of deep, narrow networks remains an important direction for future work.

If 2 n +1 ≤ m ,

## 7 Conclusion

In this study, we investigated the minimum width of deep narrow MLPs required to approximate continuously differentiable functions under the Sobolev norm. Our analysis established optimality in a broad range of cases. However, our proof techniques rely on the robustness of the topological structure under small perturbations in the derivatives of the target functions and therefore do not directly extend to the uniform norm. Nonetheless, the structure of the proofs suggests that similar bounds may still hold under the uniform norm. Developing more refined algebraic topological tools to rigorously bridge this gap presents an interesting direction for future research.

## Acknowledgments and Disclosure of Funding

This work was supported by National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) [RS-2025-00515264, RS-2024-00406127], and Global University Project grant funded by the GIST in 2025.

## References

- Aizawa, Y., Kimura, M., and Matsui, K. Universal approximation properties for an odenet and a resnet: Mathematical analysis and numerical experiments. arXiv preprint arXiv:2101.10229 , 2020.
- Arbel, M., Sutherland, D. J., Bi´ nkowski, M., and Gretton, A. On gradient regularizers for mmd gans. Advances in neural information processing systems , 31, 2018.
- Biagi, S. and Bonfiglioli, A. An Introduction to the Geometrical Analysis of Vector Fields: with Applications to Maximum Principles and Lie Groups . World Scientific, 2019.
- Cai, Y. Achieve the minimum width of neural networks for universal approximation. In The Eleventh International Conference on Learning Representations , 2022.
- Caponigro, M. Orientation preserving diffeomorphisms and flows of control-affine systems. IFAC Proceedings Volumes , 44(1):8016-8021, 2011.
- Cybenko, G. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems , 2(4):303-314, 1989.
- Czarnecki, W. M., Osindero, S., Jaderberg, M., Swirszcz, G., and Pascanu, R. Sobolev training for neural networks. Advances in neural information processing systems , 30, 2017.
- Davis, D. Vector fields on rp n × rp m . Proceedings of the American Mathematical Society , 140(12): 4381-4388, 2012.
- Evans, L. Measure theory and fine properties of functions . Routledge, 2018.
- Evans, L. C. Partial differential equations , volume 19. American Mathematical Society, 2022.
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., and Courville, A. C. Improved training of wasserstein gans. Advances in neural information processing systems , 30, 2017.
- Hanin, B. and Sellke, M. Approximating continuous functions by relu nets of minimal width. arXiv preprint arXiv:1710.11278 , 2017.
- Heinonen, J. Lectures on Lipschitz analysis . Number 100. University of Jyväskylä, 2005.
- Hirsch, M. W. Differential topology , volume 33. Springer Science &amp; Business Media, 2012.
- Hwang, G. Minimum width for deep, narrow mlp: A diffeomorphism and the whitney embedding theorem approach. arXiv preprint arXiv:2308.15873 , 2023.
- Johnson, J. Deep, skinny neural networks are not universal approximators. arXiv preprint arXiv:1810.00393 , 2018.

- Kidger, P. and Lyons, T. Universal approximation with deep narrow networks. In Conference on learning theory , pp. 2306-2327. PMLR, 2020.
- Kim, N., Min, C., and Park, S. Minimum width for universal approximation using relu networks on compact domain. arXiv preprint arXiv:2309.10402 , 2023.
- Kim, N., Min, C., and Park, S. Minimum width for universal approximation using relu networks on compact domain. In The Twelfth International Conference on Learning Representations , 2024.
- Lashof, R. and Smale, S. Self-intersections of immersed manifolds. Journal of Mathematics and Mechanics , pp. 143-157, 1959.
- Leshno, M., Lin, V. Y., Pinkus, A., and Schocken, S. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural networks , 6(6):861-867, 1993.
- Li, X. Simultaneous approximations of multivariate functions and their derivatives by neural networks with one hidden layer. Neurocomputing , 12(4):327-343, 1996.
- Lin, H. and Jegelka, S. Resnet with one-neuron hidden layers is a universal approximator. Advances in neural information processing systems , 31, 2018.
- Miller, J. G. Self-intersections of some immersed manifolds. Transactions of the American Mathematical Society , 136:329-338, 1969.
- Palais, R. S. Extending diffeomorphisms. Proceedings of the American Mathematical Society , 11(2): 274-277, 1960.
- Park, S., Yun, C., Lee, J., and Shin, J. Minimum width for universal approximation. arXiv preprint arXiv:2006.08859 , 2020.
- Phillips, A. Submersions of open manifolds. Topology , 6(2):171-206, 1967.
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics , 378:686-707, 2019.
- Schmidt-Hieber, J. Nonparametric regression using deep neural networks with relu activation function. 2020.
- Tabuada, P. and Gharesifard, B. Universal approximation power of deep residual neural networks through the lens of control. IEEE Transactions on Automatic Control , 2022.

## A Definitions and Notations

## A.1 Sets of Neural Networks

For a set of activation functions Σ , the set of MLPs denoted by N Σ d 0 ,d 1 ,...,d N is defined as:

<!-- formula-not-decoded -->

Note that, in general, an MLP can have different activation functions in each layer. If the set Σ is a singleton, i.e., Σ = { σ } , we omit the set notation and simply write:

<!-- formula-not-decoded -->

We define the set of deep, narrow MLPs with input dimension n , output dimension m , and at most w intermediate dimensions as:

<!-- formula-not-decoded -->

## A.2 Some Definitions from Differential Geometry

Definition A.1 (Diffeomorphism) . For natural numbers d, r ∈ N and open sets U 1 , U 2 ⊂ R d , a function f : U 1 → U 2 is a C r -diffeomorphism if and only if it is bijective, r -times continuously differentiable, and its inverse f -1 is r -times continuously differentiable.

Definition A.2 (Immersion) . Let M and N be smooth manifolds, and let f : M → N be a C 1 -map. The map f is called an immersion if for every point p ∈ M , the differential

<!-- formula-not-decoded -->

is injective.

Definition A.3 (Submersion) . Let M and N be smooth manifolds, and let f : M → N be a C 1 -map. The map f is called a submersion if, for every point p ∈ M , the differential

<!-- formula-not-decoded -->

is surjective.

Definition A.4 (Embedding) . Let M and N be smooth manifolds, and let f : M → N be a C 1 -map. The map f is called an embedding if it is an immersion and a homeomorphism onto its image f ( M ) , where f ( M ) is equipped with the subspace topology from N .

Definition A.5 (Transversality) . Let M,N,P be smooth manifolds and let f : M → P , g : N → P be smooth maps. We say that f and g are transverse (written f ⋔ g ) if for every pair of points p ∈ M , q ∈ N with f ( p ) = g ( q ) , we have

<!-- formula-not-decoded -->

That is, the images of the differentials at p and q together span the tangent space of P at f ( p ) = g ( q ) .

## B Practical Lemmas

In this section, we present several useful lemmas that are employed throughout the paper. The composition of functions is addressed by the following lemma.

Lemma B.1. Let f i → f in the W 1 , ∞ loc ( R m ; R l ) topology and g i → g in the W 1 , ∞ loc ( R n ; R m ) topology. Then f i ◦ g i → f ◦ g in the W 1 , ∞ loc ( R n ; R l ) topology.

Proof. It is sufficient to prove that, for each V ⋐ R n , the Lipschitz constant of f ◦ g -f i ◦ g i on V converges to zero as i increases. Choose a sufficiently large number i 0 such that for any i ≥ i 0 , we have ∥ g -g i ∥ L ∞ ( V ; R m ) &lt; 1 . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where g ( V ) + B n (1) is the Minkowski sums.

This lemma implies that if each function can be approximated by neural networks in the local Sobolev topology, then their composition can also be approximated in the same topology.

We can apply a partial activation function using the following lemma.

Lemma B.2. For natural numbers n, m, w ∈ N , and an activation function σ satisfying Condition 1, the following relation holds:

<!-- formula-not-decoded -->

Proof. For each f ∈ ∆ { σ, id } n,m,w , f can be represented as:

<!-- formula-not-decoded -->

where g ∈ ∆ { σ, id } w,w,w . By the definition of ∆ { σ, id } w,w,w , there exists a natural number N ∈ N such that the following equation holds:

<!-- formula-not-decoded -->

̸

where W i ∈ Aff w,w , g i ∈ { σ, id } w for each i ∈ [1 , N ] N . By Lemma B.1, if W i , g i ∈ ∆ σ w,w,w loc for each i ∈ [1 , N ] N , the composition g is also in ∆ σ w,w,w loc , again, leading to f ∈ ∆ σ n,m,w loc . Obviously, W i ∈ ∆ σ w,w,w loc , and it is sufficient to prove that ∆ σ w,w,w loc ⊃ { σ, id } w . For g ∈ { σ, id } w , consider I ⊂ [1 , w ] N such that g i ( x ) = σ ( x ) if i ∈ I and g i ( x ) = x if i / ∈ I . By Condition 1, there exists α ∈ R and ϵ ∈ R + such that σ ′ ( α ) = 0 and σ is C 1 function in ( α -ϵ, α + ϵ ) . For an arbitrary precompact set V ⋐ R and sufficiently large M so that α + x M ⊂ ( α -ϵ, α + ϵ ) for any x ∈ V , the following relation holds:

<!-- formula-not-decoded -->

Because M ( σ ( α + x M ) -σ ( α ) ) σ ′ ( α ) ∈ N σ 1 , 1 , 1 , the identity function x ↦→ x ∈ N σ 1 , 1 , 1 loc . Define f i ∈ N σ 1 , 1 , 1 as

<!-- formula-not-decoded -->

and concatenation f M ∈ N σ w,w,w as f M ( x ) := ( f 1 ( x 1 ) , . . . , f w ( x w )) . Then, for arbitrary precompact set V ⋐ R w ,

<!-- formula-not-decoded -->

Therefore, g ∈ ∆ σ loc , and this completes the proof.

w,w,w

## C Proof of Theorem 4.1

## C.1 Main Proof of Theorem 4.1

The theorem is proved using the following lemma, which states that any continuously differentiable function can be approximated by a two-layer neural network.

Lemma C.1 (Theorem 2.1. of Li (1996)) . Let K be a compact subset of R s , s ≥ 1 , and f ∈ C m 1 ( K ) ∩·· ·∩ C m q ( K ) , where m i ∈ N s 0 for 1 ≤ i ≤ q . Also, let σ be any non-polynomial function in C n ( R ) , where n = max {| m i | : 1 ≤ i ≤ q } . Then for any ϵ &gt; 0 , there is a network

<!-- formula-not-decoded -->

where c i ∈ R , w i ∈ R s , and θ i ∈ R , 0 ≤ i ≤ v , such that

<!-- formula-not-decoded -->

The following two lemmas state that any arbitrary increasing function can be approximated using Leaky-ReLU and Leaky-ReLU-like activation functions, respectively.

Lemma C.2 (Increasing Functions to Leaky-ReLU) . For any increasing C 1 function f ,

<!-- formula-not-decoded -->

The proof of Lemma C.2 is provided in Appendix C.2.

Lemma C.3. Let Σ = { σ β | β ∈ R + } be a set of Leaky-ReLU-like activation functions. Then, for any increasing C 1 function f ,

<!-- formula-not-decoded -->

The proof of Lemma C.3 is provided in Appendix C.3. The two lemmas yield the following corollary.

Corollary C.4 (Generalization of Activation) . For a natural number d ∈ N and any increasing, C 1 activation function ρ , the following relation holds:

<!-- formula-not-decoded -->

where σ is the Leaky-ReLU or a set of Leaky-ReLU-like activation functions.

The following lemma is a technical result used to approximate a vector field with deep narrow MLPs. Lemma C.5. For t, b ∈ R , and w ∈ R d , define f t : R d → R d as:

<!-- formula-not-decoded -->

Let σ be the Leaky-ReLU or a set of Leaky-ReLU-like activation functions. Then, there exists a positive real number δ ∈ R + such that, for | t | &lt; δ , the following relation holds:

<!-- formula-not-decoded -->

The proof of Lemma C.5 is provided in Appendix C.4. The following lemma states that any smooth diffeomorphism can be approximated by flows generated by (time-dependent) vector fields. The definition of a vector field is as follows:

Definition C.6 (Flow of a Vector Field) . Let f : R d × R → R d be a function that is Lipschitz continuous with respect to x and a piecewise continuous with respect to t . For each f ∈ A , consider a ODE system

<!-- formula-not-decoded -->

where x : R → R d . We define a flow map ϕ t,s f : R d → R d , corresponding to f as follows:

<!-- formula-not-decoded -->

For t = 0 , we omit t and just denote it as ϕ s f :

<!-- formula-not-decoded -->

We define the maximal domain M f ⊂ R d × R as the set which satisfies ( x, t ) ∈ M f if and only if the solution ϕ t f ( x ) is well-defined. It is well known that M f is an open set. It is also well known that if f is a C k -function with respect to x and t , then ϕ t f ( x ) is also C k -function with respect to x and t . (See Theorem B.41 of Biagi &amp; Bonfiglioli (2019) for example.)

When we consider Df , we only consider a Jacobian with respect to x :

<!-- formula-not-decoded -->

Lemma C.7 (Theorem 5 of Caponigro (2011)) . Any orientation preserving diffeomorphism can be represented by a flow map: For any diffeomorphism f ∈ D ∞ ( R d ) with det ( Df ) &gt; 0 , there exists a flow map ϕ t F generated by a ODE system ˙ x = F ( x, t ) with a smooth vector field F : R d × R → R d such that the following equation holds:

<!-- formula-not-decoded -->

If two vector fields are close, then the flows they generate are also close.

Lemma C.8. Consider C 1 , 1 functions f 1 , f 2 : R d × R → R d . Define two ODE systems ˙ x = f i ( x, t ) for i = 1 , 2 and let ϕ t i := ϕ t f i be a flow map defined by each f i . Assume that V × [0 , τ ] ⊂ M f 1 for a precompact set V ⋐ R d and τ ∈ R + . Define ˜ V as

<!-- formula-not-decoded -->

Then, for any ϵ ∈ R + , there exists a positive number δ ∈ (0 , 1) such that if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Lemma C.8 is provided in Appendix C.5.

Now, we approximate a two-layered-MLP-like vector field using deep narrow MLPs.

Lemma C.9. For i ∈ [1 , N ] N and C 1 , 1 -functions v i : R d × R → R d , let

<!-- formula-not-decoded -->

For a real number τ ∈ R + and a precompact set U ⋐ R d , assume that U × [0 , τ ] ⊂ M f .

Consider n ∈ N , t k := kτ n , ∆ t := τ n ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, there exists a natural number n 0 ∈ N such that if n ≥ n 0 , the following relation holds:

<!-- formula-not-decoded -->

The proof of Lemma C.9 is provided in Appendix C.6.

By combining all the lemmas, we prove the theorem.

Proof of Theorem 4.1. By Theorem 2.7, p.50 in Hirsch (2012), we only have to consider D ∞ ( R d ) . If f is an orientation reversing diffeomorphism, g ◦ f is orientation preserving where g ∈ ∆ LR d,d,d is defined as:

<!-- formula-not-decoded -->

Therefore, we only consider an orientation preserving diffeomorphism f ∈ D ∞ ( R d ) . Consider an arbitrary precompact set V ⋐ R d and ϵ ∈ R + . By Lemma C.7, there exists an ODE flow induced by F ∈ C ∞ ( R d × R ; R d ) such that

<!-- formula-not-decoded -->

By Lemma C.8, there exists δ ∈ R + and ˜ V ⋐ R d such that if ∥ F ( · , t ) -F 2 ( · , t ) ∥ W 1 , ∞ ( ˜ V ; R d ) &lt; δ for all t ∈ [0 , 1] , then, ∥ ϕ 1 F -ϕ 1 F 2 ∥ W 1 , ∞ ( V ; R d ) &lt; ϵ . Consider a compact set K such that ˜ V ⊂ K . By Lemma C.1, there exists a F 2 : R d × R → R d such that

<!-- formula-not-decoded -->

where F 2 is represented as for all t ∈ [0 , τ ] , then,

and

<!-- formula-not-decoded -->

where a i , b i , c i ∈ R , and w i ∈ R d . Here, ρ = tanh if σ is Leaky-ReLU or Leaky-ReLU-like, and ρ equals to σ if σ is an activation function satisfying Condition 1. As both F and F 2 are C 1 , 1 functions,

<!-- formula-not-decoded -->

for all t ∈ [0 , τ ] . Thus, ∥ ϕ 1 F -ϕ 1 F 2 ∥ W 1 , ∞ ( V ; R d ) &lt; ϵ 2 . Then, by Lemma C.9, there exists a natural number n 0 ∈ N so that if n ≥ n 0 , then, for t k := kτ n , ∆ t = τ n ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the Leaky-ReLU or Leaky-ReLU like σ , by Lemma C.5, there exists i ∈ [1 , N ] N , k ∈ [1 , n ] N , and δ i,k ∈ R + such that if | t | &lt; δ i , then, f i,k ∈ ∆ σ d,d,d loc . Choose sufficiently large n so that | ∆ tc i | &lt; δ i for all , each f i,k ∈ ∆ σ d,d,d loc . Then, S n ∈ ∆ σ d,d,d loc . For σ satisfying Condition 1, f i,k ∈ ∆ σ d,d,d +1 loc , thus, S n ∈ ∆ σ d,d,d +1 loc . Thus, ∥ ϕ 1 F -S n ∥ W 1 , ∞ ( V ; R d ) &lt; ϵ for S n ∈ ∆ σ d,d,d + α ( σ ) loc . This completes the proof.

## C.2 Proof of Lemma C.2

Proof. ∆ LR 1 , 1 , 1 is the set of strictly increasing piecewise linear functions with finite segments. Consider any increasing C 1 function σ : R → R , a compact set K ⊂ R , and a positive real number ϵ ∈ R + . We will construct a function f ∈ U such that ∥ σ -f ∥ W 1 , ∞ &lt; ϵ . Consider a closed interval [ a, b ] ⊃ K . Then, there exists a natural number n ∈ N such that ∥ f ( x ) -f ( y ) ∥ &lt; ϵ 4 and ∥ Df ( x ) -Df ( y ) ∥ &lt; ϵ 4 for ∥ x -y ∥ &lt; 1 n . Define f ∈ U as a piecewise linear function with breaking points x = a +( b -a ) i/n for 0 ≤ i ≤ n , which has the same values with f in each breaking point. For all x ∈ K and the closest breaking point y ∈ [ a, b ] , | x -y | &lt; ϵ . Then,

<!-- formula-not-decoded -->

And for two adjacent breaking points y 0 , y 1 such that x ∈ [ y 0 , y 1 ] ,

<!-- formula-not-decoded -->

for a c ∈ ( y 0 , y 1 ) by mean value theorem, almost everywhere. Therefore, ∥ σ ( x ) -f ( x ) ∥ W 1 , ∞ ( K ) &lt; ϵ . Because the selection of a compact set K ⊂ R is arbitrary, σ ∈ ∆ LR 1 , 1 , 1 loc , and this completes the proof.

## C.3 Proof of Lemma C.3

Proof. As strictly increasing C 1 functions are dense in the set of increasing C 1 functions in C 1 topology, we only need to approximate a strictly increasing C 1 function f . Consider an arbitrarily small error ϵ ∈ R + and an open interval ( a, b ) . It is sufficient to prove that there exists g ∈ ∆ σ 1 , 1 , 1 loc such that

<!-- formula-not-decoded -->

Define L 1 , L 2 ∈ R + as uniquely determined value as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Choose a sufficiently small ϵ ′ ∈ R + so that (6 L 2 +4) b (1+ ϵ ′ )+2 ϵ ′ &lt; ϵ . There exists a natural number N ∈ N such that if ∥ x -y ∥ &lt; 1 N , then, ∥ f ( x ) -f ( y ) ∥ &lt; ϵ 4 and ∥ Df ( x ) -Df ( y ) ∥ &lt; min ( ϵ 4 , ϵ ′ ) . Define h as a piecewise linear function with breaking points α i = a +( b -a ) i/N for 0 ≤ i ≤ N , which has the same values with σ in each breaking point. Then,

<!-- formula-not-decoded -->

and the following inequality holds:

Define b : R + → R as

Now, it is sufficient to prove that there exists a function h ′ ∈ ∆ σ 1 , 1 , 1 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that γ i be the slope of h in ( α i , α i +1 ) . We use mathematical induction on n to prove the following: There exists a f n,m ∈ ∆ σ 1 , 1 , 1 such that

<!-- formula-not-decoded -->

2. there exists a natural number M such that if m ≥ M , then, ∥ Dh -Df n,m ∥ L ∞ (( a,α n +1 )) &lt; ϵ 2 ,

<!-- formula-not-decoded -->

For n = 0 , there is nothing to prove. Assume that the induction hypothesis is satisfied for n . Define f n +1 ,m ∈ ∆ σ 1 , 1 , 1 as

<!-- formula-not-decoded -->

As σ β ( mx ) m m →∞ - - - - → LR β ( x ) in C 0 -topology,

<!-- formula-not-decoded -->

with C 0 -topology. Now, it is sufficient to prove that the derivate-related assumptions. Df n +1 ,m can be calculated as

<!-- formula-not-decoded -->

Then, for any δ ∈ R + and x ∈ [ a, α n +1 -δ ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

And, for x ∈ [ α n +1 + δ, b ] , lim m →∞ sup x ∈ [ α n +1 + δ,b ] ∥ Df n,m ( x ) -γ n ∥ = 0 , and therefore,

<!-- formula-not-decoded -->

Therefore, the induction hypothesis 3 is satisfied. As h ( x ) = γ n +1 ( x -α n +2 ) + f ( α n +2 ) for x ∈ [ α n +1 , α n +2 ] ,

<!-- formula-not-decoded -->

Now, it remains to prove that there exists a natural number M ′ such that if m ≥ M ′ , then ∥ Dh -Df n +1 ,m ∥ L ∞ (( α n +1 -δ,α n +1 + δ )) &lt; ϵ . Choose sufficiently large M ′ so that if m ≥ M ′ , then,

<!-- formula-not-decoded -->

Define γ i as

Then, for x ∈ ( α n +1 -δ, α n +1 + δ ) ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore, for x ∈ ( α n +1 -δ, α n +1 + δ ) ,

<!-- formula-not-decoded -->

By mathematical induction, we conclude that there exists f N,m ∈ ∆ σ 1 , 1 , 1 and M ∈ N such that if m ≥ M , then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the proof.

## C.4 Proof of Lemma C.5

Proof. For w = ( w 1 , . . . , w d ) , if w i = 0 for i ∈ [1 , d -1] N , the last term of f t can be calculated as:

<!-- formula-not-decoded -->

For sufficiently small δ ∈ R + and | t | &lt; δ , this function is increasing. Thus, by Corollary C.4, the following relations holds:

<!-- formula-not-decoded -->

Also, the following relation holds:

<!-- formula-not-decoded -->

̸

Now assume that there exists i ∈ [1 , d -1] N such that w i = 0 . Further, without loss of generality, assume that w 1 = 0 . Then, the following functions are elements of ∆ σ d,d,d loc

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Then, the composition f 6 ◦ f 5 ◦ f 4 ◦ f 3 ◦ f 2 ◦ f 1 becomes

<!-- formula-not-decoded -->

̸

## C.5 Proof of Lemma C.8

Proof. Let L and L ′ be a Lipschitz constant of f 1 and Df 1 with respect to x in ˜ V , respectively. Restrict δ to

<!-- formula-not-decoded -->

We first prove that, for all x ∈ V and t ∈ [0 , τ ] , ϕ t 2 ∈ ˜ V . Define T as

<!-- formula-not-decoded -->

1. Obviously, 0 ∈ T .
2. And T is an open set relative to [0 , τ ] : Assume that t ∈ T . Then, as ϕ t 2 ( x ) ∈ ˜ V for all x ∈ V , and M f 2 and ˜ V is open, there exist ϵ 1 ,x , ϵ 2 ,x ∈ R + such that if ∥ y -x ∥ &lt; ϵ 1 ,x and | t ′ -t | ≤ ϵ 2 ,x , then, ϕ t ′ 2 ( y ) ∈ ˜ V . Because V is compact, we can choose a finite cover {{ x } + B d ( ϵ 1 ,x ) } x ∈ S of V . Then, [ t, t +min x ∈ S ϵ 2 ,x ) ⊂ T , and T becomes an open set.
3. T is closed relative to [0 , τ ] : Assume that T = [0 , t ) for t ∈ (0 , τ ] . It is sufficient to prove that

<!-- formula-not-decoded -->

is finite and in ˜ V . Define e ( x, t ) as

<!-- formula-not-decoded -->

Then, the following equation holds:

<!-- formula-not-decoded -->

Then, as ϕ s 1 ( x ) , ϕ s 1 ( x ) ∈ ˜ V for s ∈ [0 , t ) , the following inequalities hold:

<!-- formula-not-decoded -->

where the last inequality is by Gronwall's inequality. As δte Lt &lt; δτe Lτ &lt; 1 ,

<!-- formula-not-decoded -->

for all x ∈ V , which leads to t ∈ T .

4. We conclude that T = [0 , τ ] .

Next, we prove that ∥ e ( x, t ) ∥ can be bounded. It is already proven by setting δ &lt; ϵ 2 τe Lτ . Then, ∥ e ( x, t ) ∥ &lt; ϵ 2 .

Finally, we will prove that ∥ De ( x, t ) ∥ can be bounded.

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

where the last inequality is by Gronwall's inequality again. By setting sufficiently small δ , we get

<!-- formula-not-decoded -->

for all x ∈ V and t ∈ [0 , τ ] . This completes the proof.

## C.6 Proof of Lemma C.9

Proof. Define V 0 ⊂ R d × R and V 0 t ⊂ R d as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

As V 0 is compact, there exists a positive number δ ∈ R + such that

<!-- formula-not-decoded -->

for all t ∈ [0 , τ ] . Define V ⊂ R d × R as

<!-- formula-not-decoded -->

We will conduct all our discussions on V where ϕ t v is well-defined. Denote the supremum and the Lipschitz constant of v in V as C and L , respectively. Also, denote the supremum and the Lipschitz constant (as operator norm) of Dv in V as C ′ and L ′ .

In this proof, we will use a big-O notation with respect to ∆ t ; that is, a function f : R → R is denoted as if and only if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c is a constant independent of ∆ t and polynomially dependent on N,L,C,L ′ , C ′ .

We will check that

<!-- formula-not-decoded -->

We define U l,k ∈ C 1 ( R d ; R d ) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, it is sufficient to bound two terms:

<!-- formula-not-decoded -->

And define U k as

The first term can be calculated as

<!-- formula-not-decoded -->

where the last equality is by the following arguments: for any k and x ∈ V k ,

<!-- formula-not-decoded -->

where the second last inequality is by Gronwall's inequality. Therefore,

<!-- formula-not-decoded -->

and the bound is independent of k . To calculate the second term ∥ T k +1 -U k +1 ∥ W 1 , ∞ ( V t k ; R d ) , for l ∈ [1 , N ] N , define T l,k ∈ C 1 ( R d ; R d ) as

<!-- formula-not-decoded -->

Then, T N,k = T k . We inductively bound

<!-- formula-not-decoded -->

When l = 1 , T 1 ,k = U 1 ,k = f 1 ,k , and there is nothing to prove. Assume that the above induction hypothesis is satisfied for l . Then,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Therefore,

And thus,

<!-- formula-not-decoded -->

Now, define e k : R d → R d as

We restrict ∆ t sufficiently small so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under this assumption, we use the mathematical induction on k to prove that S k ( x ) ∈ V t k for an arbitrary k ∈ [1 , n ] N . It is obvious when k = 0 . Assume that the induction hypothesis is satisfied for k = k 0 . For x ∈ U and k ≤ k 0 ,

<!-- formula-not-decoded -->

Then, for any k ≤ τ ∆ t ,

<!-- formula-not-decoded -->

As S k +1 ( x ) = ϕ t k +1 v ( x ) + e k +1 ( x ) ∈ V t k +1 , the induction hypothesis is satisfied. Also, ∥ e k ∥ L ∞ ( U ; R d ) &lt; ϵ 2 .

Now, we bound Dϕ t v ( x ) . First, we bound a derivative D ( ϕ t k ,t k +1 v -I d ) . For arbitrary s, t ∈ [0 , τ ] and x ∈ V s , consider the following equation.

<!-- formula-not-decoded -->

Apply derivative to both sides, and we get

<!-- formula-not-decoded -->

where the last inequality is by the Gronwall's inequality. Denote the last constant as L ′ 1 := dL ′ e L ′ τ ; that is,

<!-- formula-not-decoded -->

Calculate the Lipschitz constant of Dϕ t k ,t k +1 v . For s, t ∈ [ t k , t k +1 ] , and x, y ∈ V t k .

<!-- formula-not-decoded -->

where the second last inequality is by Gronwall's inequality. Denote the last constant as L ′ 2 := 4 L ′ C ′ e 2 ; that is,

<!-- formula-not-decoded -->

Now we calculate the followings:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use the mathematical induction on l to prove that

<!-- formula-not-decoded -->

When l = 1 , T 1 ,k = U 1 ,k = f 1 ,k , and there is nothing to prove. Assume that the above induction hypothesis is satisfied for l . Then,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Therefore, the induction hypothesis is satisfied.

For any x ∈ U and k , we have

<!-- formula-not-decoded -->

For the first term, we have

<!-- formula-not-decoded -->

For the second term, there exists a constant c 2 ∈ R + satisfying

<!-- formula-not-decoded -->

For the last term, we have

<!-- formula-not-decoded -->

Then, by selecting a sufficiently small ∆ t , we have

<!-- formula-not-decoded -->

for a constant c 3 ∈ R + . We conclude that

<!-- formula-not-decoded -->

and this completes the proof.

## D Proofs of Approximation Lemmas

## D.1 Proof of Theorem 4.4

Proof. Consider a function f ∈ C 1 ( R n ; R m ) and a precompact set V ⋐ R n . It is sufficient to prove that, for any ϵ ∈ R + , there exists a function ˜ f ∈ ∆ σ n,m, Ω( n,m )+ α ( σ ) such that ∥ f -˜ f ∥ W 1 , ∞ ( V ; R m ) &lt; ϵ . Because ∆ σ n,m, Ω( n,m )+ α ( σ ) is closed under affine transformation composition, we only need to consider V satisfying V ⋐ (0 , 1) n . By the definition of Ω( n, m ) , for any ϵ ∈ R + , there exists an embedding g ∈ Emb([0 , 1] n , R Ω( n,m ) ) such that

<!-- formula-not-decoded -->

Because Ω( n, m ) ≥ n , by Lemma 4.2, for q n, Ω( n,m ) : ( x 1 , . . . , x n ) ↦→ ( x 1 , . . . , x n , 0 , . . . , 0) , there exists a smooth diffeomorphism G such that g = G ◦ q n, Ω( n,m ) . By Theorem 4.1, there exists an MLP H ∈ ∆ σ Ω( n,m ) , Ω( n,m ) , Ω( n,m )+ α ( σ ) such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then,

Therefore,

<!-- formula-not-decoded -->

p Ω( n,m ) ,m ◦ H ◦ q n, Ω( n,m ) ∈ ∆ σ n,m, Ω( n,m )+ α ( σ ) . This completes the proof.

## D.2 Proof of Proposition 4.5

Proof. For a non-decreasing C 1 activation function σ , there exist smooth, strictly increasing activation functions σ n that converge to σ in W 1 , ∞ loc topology. Therefore, ∆ σ d,d,d loc ⊂ ∆ { σ n | n ∈ N } d,d,d loc , making it sufficient to consider only a smooth, strictly increasing activation function σ .

For f ∈ ∆ σ n,m, Ω( n,m ) -1 , it can be decomposed as:

<!-- formula-not-decoded -->

where g ∈ ∆ σ Ω( n,m ) -1 , Ω( n,m ) -1 , Ω( n,m ) -1 . As ∆ σ Ω( n,m ) -1 , Ω( n,m ) -1 , Ω( n,m ) -1 ⊂ D ∞ ( R Ω( n,m ) -1 ) loc , g ◦ q n, Ω( n,m ) -1 ∣ ∣ (0 , 1) n ∈ Emb((0 , 1) n , R Ω( n,m ) -1 ) loc . Therefore, we have:

<!-- formula-not-decoded -->

and as the selection of f ∈ ∆ σ n,m, Ω( n,m ) -1 is arbitrary, we get the following:

<!-- formula-not-decoded -->

As Ω( n, m ) -1 &lt; Ω( n, m ) , by the definition of Ω( n, m ) :

<!-- formula-not-decoded -->

and thus,

<!-- formula-not-decoded -->

Therefore, we have C 1 ( R n , R m ) ⊈ ∆ σ n,m, Ω( n,m ) -1 loc . This completes the proof.

## E Proofs of Topological Lemmas

## E.1 Proof of Lemma 5.5

Proof. Define F : ( -δ, 1 + δ ) × U × U → R m +1 as

<!-- formula-not-decoded -->

Then F is a proper submersion for sufficiently small ϵ . Then, DF ( α, x, y ) can be calculated as

<!-- formula-not-decoded -->

Consider a vector field X i in ( -δ, 1 + δ ) × U × U for i ∈ [1 , m +1] N which satisfy the following:

<!-- formula-not-decoded -->

where e i is the i -th coordinate vector. Then, define G : F -1 ( R m +1 ) → R m +1 × F -1 (0) as

<!-- formula-not-decoded -->

Then, G has a inverse G -1 : R m +1 × F -1 (0) → F -1 ( R m +1 ) which can be calculated as

<!-- formula-not-decoded -->

for x ∈ R and x ∈ F -1 (0) . Then, for the projection p : R m +1 × F -1 (0) → R m +1 , the following equation holds:

<!-- formula-not-decoded -->

Therefore, F -1 ( c 1 ) is diffeomorphic to F -1 ( c 2 ) for c 1 , c 2 ∈ R m +1 .

Note that the above diffeomorphism G can be defined for all X i that satisfy Equation (176).

We set X 1 as

<!-- formula-not-decoded -->

Then, ϕ 1 X 1 is the diffeomorphism between F -1 (0 , 0) = { 0 }× ˜ f -1 (0) and F -1 (1 , 0) = { 1 }× ˜ g -1 (0) . Let X 1 be represented as

<!-- formula-not-decoded -->

It is enough to prove that M 1 ( α, y, x ) = M 2 ( α, x, y ) . Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, A ( y, x ) = -A ( x, y ) .

( DF ) T DF can be represented as

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

and

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: No justification

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: No justification

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

Justification: No justification

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: No justification

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

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
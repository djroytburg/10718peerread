<!-- image -->

## O ( √ T ) Static Regret and Instance Dependent Constraint Violation for Constrained Online Convex Optimization

## Rahul Vaze ∗

School of Technology and Computer Science

Tata Institute of Fundamental Research, Mumbai rahul.vaze@gmail.com

## Abhishek Sinha

School of Technology and Computer Science Tata Institute of Fundamental Research, Mumbai abhishek.sinha@tifr.res.in

## Abstract

The constrained version of the standard online convex optimization (OCO) framework, called COCO is considered, where on every round, a convex cost function and a convex constraint function are revealed to the learner after it chooses the action for that round. The objective is to simultaneously minimize the static regret and cumulative constraint violation (CCV). An algorithm is proposed that guarantees a static regret of O ( √ T ) and a CCV of min { V , O ( √ T log T ) } , where V depends on the distance between the consecutively revealed constraint sets, the shape of constraint sets, dimension of action space and the diameter of the action space. When constraint sets have additional structure, V = O (1) . Compared to the state of the art results, static regret of O ( √ T ) and CCV of O ( √ T log T ) , that were universal, the new result on CCV is instance dependent, which is derived by exploiting the geometric properties of the constraint sets.

## 1 Introduction

In this paper, we consider the constrained version of the standard online convex optimization (OCO) framework, called constrained OCO or COCO. In COCO, on every round t, the online algorithm first chooses an admissible action x t ∈ X ⊂ R d , and then the adversary chooses a convex loss/cost function f t : X → R and a constraint function of the form g t ( x ) ≤ 0 , where g t : X → R is a convex function. Since g t 's are revealed after the action x t is chosen, an online algorithm need not necessarily take feasible actions on each round, and in addition to the static regret

<!-- formula-not-decoded -->

an additional metric of interest is the total cumulative constraint violation (CCV) defined as CCV [1: T ] ≡ ∑ T t =1 max( g t ( x t ) , 0) . Let X glyph[star] be the feasible set consisting of all admissible actions that satisfy all constraints g t ( x ) ≤ 0 , t ∈ [ T ] . Under the standard assumption that X glyph[star] is not

∗ Both authors acknowledge support of the Department of Atomic Energy, Government of India, under project no. RTI4001 and Google India Research Award 2023.

empty (called the feasibility assumption ), the goal is to design an online algorithm to simultaneously achieve a small regret (1) with respect to any admissible benchmark x glyph[star] ∈ X glyph[star] and a small CCV.

glyph[negationslash]

With constraint sets G t = { x ∈ X : g t ( x ) ≤ 0 } being convex for all t , and the assumption X glyph[star] = ∩ t G t = ∅ implies that sets S t = ∩ t τ =1 G τ are convex and are nested, i.e. S t ⊆ S t -1 and X glyph[star] ∈ S t for all t . Essentially, set S t 's are sufficient to quantify the CCV.

## 1.1 Prior Work

Constrained OCO (COCO): (A) Time-invariant constraints: COCO with time-invariant constraints, i.e., g t = g, ∀ t [Yuan and Lamperski, 2018, Jenatton et al., 2016, Mahdavi et al., 2012, Yi et al., 2021] has been considered extensively, where functions g are assumed to be known to the algorithm a priori . The algorithm is allowed to take actions that are infeasible at any time to avoid the costly projection step of the vanilla projected OGD algorithm and the main objective was to design an efficient algorithm with a small regret and CCV while avoiding the explicit projection step.

(B) Time-varying constraints: The more difficult question is solving COCO problem when the constraint functions, i.e. , g t 's, change arbitrarily with time t . In this setting, all prior work on COCO made the feasibility assumption. One popular algorithm for solving COCO considered a Lagrangian function optimization that is updated using the primal and dual variables [Yu et al., 2017, Sun et al., 2017, Yi et al., 2023]. Alternatively, [Neely and Yu, 2017] and [Liakopoulos et al., 2019] used the drift-plus-penalty (DPP) framework [Neely, 2010] to solve the COCO, but which needed additional assumption, e.g. the Slater's condition in [Neely and Yu, 2017] and with weaker form of the feasibility assumption [Neely and Yu, 2017]'s.

[Guo et al., 2022] obtained the bounds similar to [Neely and Yu, 2017] but without assuming Slater's condition. However, the algorithm [Guo et al., 2022] was quite computationally intensive since it requires solving a convex optimization problem on each round. Finally, very recently, the state of the art guarantees on simultaneous bounds on regret O ( √ T ) and CCV O ( √ T log T ) for COCO were derived in [Sinha and Vaze, 2024] with a very simple algorithm that combines the loss function at time t and the CCV accrued till time t in a single loss function, and then executes the online gradient descent (OGD) algorithm on the single loss function with an adaptive step-size. Another extension of [Sinha and Vaze, 2024] can be found in [Lekeufack and Jordan, 2025] that considers COCO problem under predictions about f t 's and g t 's. See Remark 6 for comparison of this work with [Lekeufack and Jordan, 2025]. Please refer to Table 1 for a brief summary of the prior results.

The COCO problem has been considered in the dynamic setting as well [Chen and Giannakis, 2018, Cao and Liu, 2018, Vaze, 2022, Liu et al., 2022] where the benchmark x glyph[star] in (1) is replaced by x glyph[star] t ( x glyph[star] t = arg min x f t ( x ) ) that is also allowed to change its actions over time. However, in this paper, we focus our entire attention on the static version. A special case of COCO is the online constraint satisfaction (OCS) problem that does not involve any cost function, i.e., f t = 0 , ∀ t, and the only object of interest is minimizing the CCV. The algorithm with state of the art guarantee for COCO [Sinha and Vaze, 2024] was shown to have a CCV of O ( √ T log T ) for the OCS.

## 1.2 Convex Body Chasing Problem

A well-studied problem related to the COCO is the nested convex body chasing (NCBC) problem [Bansal et al., 2018, Argue et al., 2019, Bubeck et al., 2020], where at each round t , a convex set χ t ⊆ χ is revealed such that χ t ⊆ χ t -1 , and χ 0 = χ ⊆ R d is a convex, compact, and bounded set. The objective is to choose action x t ∈ χ t so as to minimize the total movement cost C = ∑ T t =1 || x t -x t -1 || , where x 0 ∈ χ is some fixed action. Best known-algorithms for NCBC [Bansal et al., 2018, Argue et al., 2019, Bubeck et al., 2020] choose x t to be the centroid or Steiner point of χ t , essentially well inside the newly revealed convex set in order to reduce the future movement cost. With COCO, such an approach does not appear useful because of the presence of cost functions f t 's whose minima could be towards the boundary of convex sets χ t 's.

## 1.3 Limitations of Prior Work

We explicitly show in Lemma 6 that the best known algorithm [Sinha and Vaze, 2024] (in terms of regret and up to log factors for CCV) for solving COCO suffers a CCV of Ω( √ T log T ) even for 'simple' problem instances where f t = f and g t = g for all t and d = 1 dimension, for which ideally the CCV should be O (1) . The same is true for most other algorithms, where the main reason for their large CCV for simple instances is that all these algorithms treat minimizing the CCV as

a regret minimization problem for functions g t . What they fail to exploit is the geometry of the underlying nested convex sets S t 's that control the CCV.

## 1.4 Main open question

In comparison to the above discussed upper bounds, the best known simultaneous lower bound [Sinha and Vaze, 2024] for COCO is R [1: T ] = Ω( √ d ) and CCV [1: T ] = Ω( √ d ) , where d is the dimension of the action space X . Without constraints, i.e., g t ≡ 0 for all t , the lower bound on R [1: T ] = Ω( √ T ) [Hazan, 2012]. Thus, there is a fundamental gap between the lower and upper bound for the CCV, and the main open question for COCO is : Is it possible to simultaneously achieve R [1: T ] = O ( √ T ) and CCV [1: T ] = o ( √ T ) or CCV [1: T ] = O (1) for COCO? Even though we do not fully resolve this question, in this paper, we make some meaningful progress by proposing an algorithm that exploits the geometry of the nested sets S t 's and show that it is possible to simultaneously achieve R [1: T ] = O ( √ T ) and CCV [1: T ] = O (1) in certain cases, and for general case, give a bound on the CCV that depends on the shape of the convex sets S t 's while achieving R [1: T ] = O ( √ T ) . In particular, the contributions of this paper are as follows.

## 1.5 Our Contributions

In this paper, we propose an algorithm (Algorithm 2) that tries to exploit the geometry of the nested convex sets S t 's. In particular, Algorithm 2 at time t , first takes an OGD step from the previous action x t -1 with respect to the most recently revealed loss function f t -1 with appropriate step-size to reach y t -1 , and then projects y t -1 onto the most recently revealed set S t -1 to get x t , the action to be played at time t . Let F t be the 'projection' hyperplane passing through x t that is perpendicular to x t -y t -1 . For Algorithm 2, we derive the following guarantees.

- The regret of the Algorithm 2 is O ( √ T ) .
- The CCV for the Algorithm 2 takes the following form
- -When sets S t 's are structured, e.g. are spheres, or axis parallel cuboids/regular polygons, CCV is O (1) .
- -For the special case of d = 2 , when projection hyperplanes F t 's progressively make increasing angles with respect to the first projection hyperplane F 1 , the CCV is O (1) .
- -For general S t 's, the CCV is upper bounded by a quantity V that is a function of the distance between the consecutive sets S t and S t +1 for all t , the shape of S t 's, dimension d and the diameter D . Since V depends on the shape of S t 's, there is no universal bound on V , and the derived bound is instance dependent.
- As pointed out above, for general S t 's, there is no universal bound on the CCV of Algorithm 2. Thus, we propose an algorithm Switch that combines Algorithm 2 and the algorithm from [Sinha and Vaze, 2024] to provide a regret bound of O ( √ T ) and a CCV that is minimum of V and O ( √ T log T ) . Thus, Switch provides a best of two worlds CCV guarantee, which is small if the sets S t 's are 'nice', while in the worst case it is at most O ( √ T log T ) .
- For the OCS problem, where f t = 0 , ∀ t , we show that the CCV of Algorithm 2 is O (1) compared to the CCV of O ( √ T log T ) [Sinha and Vaze, 2024].

## 2 COCO Problem

On round t, the online policy first chooses an admissible action x t ∈ X ⊂ R d , and then the adversary chooses a convex cost function f t : X → R and a constraint of the form g t ( x ) ≤ 0 , where g t : X → R is a convex function. Once the action x t has been chosen, we let ∇ f t ( x t ) and full function g t or the set { x : g t ( x ) ≤ 0 } to be revealed, as is standard in the literature. We now state the standard assumptions made in the literature while studying the COCO problem [Guo et al., 2022, Yi et al., 2021, Neely and Yu, 2017, Sinha and Vaze, 2024].

Assumption 1 (Convexity) X ⊂ R d is the admissible set that is closed, convex and has a finite Euclidean diameter D . The cost function f t : X ↦→ R and the constraint function g t : X ↦→ R are convex for all t ≥ 1 .

Table 1: Summary of the results on COCO for arbitrary time-varying convex constraints and convex cost functions. In the above table, 0 ≤ β ≤ 1 is an adjustable parameter. Conv-OPT refers to solving a constrained convex optimization problem on each round. Projection refers to the Euclidean projection operation on the convex set X . The CCV bound for this paper is stated in terms of V which can be O (1) or depends on the shape of convex sets S t 's.

| Reference                                                                                                               | Regret                                                                    | CCV                                                                                         | Complexity per round                                                                              |
|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [Neely and Yu, 2017], [Liakopoulos et al., 2019] [Guo et al., 2022] [Yi et al., 2023] [Sinha and Vaze, 2024] This paper | O ( √ T ) O ( √ T ) O ( √ T ) O ( T max( β, 1 - β ) ) O ( √ T ) O ( √ T ) | O ( √ T ) O ( √ T ) O ( T 3 4 ) O ( T 1 - β/ 2 ) O ( √ T log T ) O (min { V , √ T log T } ) | Conv-OPT, Slater's condition Conv-OPT, Slater's condition Conv-OPT Conv-OPT Projection Projection |

Assumption 2 (Lipschitzness) All cost functions { f t } t ≥ 1 and the constraint functions { g t } t ≥ 1 's are G -Lipschitz, i.e., for any x, y ∈ X , we have | f t ( x ) -f t ( y ) | ≤ G || x -y || , | g t ( x ) -g t ( y ) | ≤ G || x -y || , ∀ t ≥ 1 .

glyph[negationslash]

Assumption 3 (Feasibility) With G t = { x ∈ X : g t ( x ) ≤ 0 } , we assume that X glyph[star] = ∩ T t =1 G t = ∅ . Any action x glyph[star] ∈ X glyph[star] is defined to be feasible.

The feasibility assumption distinguishes the cost functions from the constraint functions and is common across all previous literature on COCO [Guo et al., 2022, Neely and Yu, 2017, Yu and Neely, 2016, Yuan and Lamperski, 2018, Yi et al., 2023, Liakopoulos et al., 2019, Sinha and Vaze, 2024].

For any real number z , we define ( z ) + ≡ max(0 , z ) . Since g t 's are revealed after the action x t is chosen, any online policy need not necessarily take feasible actions on each round. Thus in addition to the static 2 regret defined below

<!-- formula-not-decoded -->

where an additional obvious metric of interest is the total cumulative constraint violation (CCV) defined as CCV [1: T ] = ∑ T t =1 ( g t ( x t )) + . Under the standard assumption (Assumption 3) that X glyph[star] is not empty, the goal is to design an online policy to simultaneously achieve a small regret with x glyph[star] ∈ X glyph[star] and a small CCV.

For simplicity, we define set

<!-- formula-not-decoded -->

where G t is as defined in Assumption 3. All G t 's are convex and consequently, all S t 's are convex and are nested, i.e. S t ⊆ S t -1 . Moreover, because of Assumption 3, each S t is non-empty and in particular X glyph[star] ∈ S t for all t . After action x t has been chosen, set S t controls the constraint violation, which can be used to write an upper bound on the CCV [1: T ] as follows.

Definition 4 For a convex set χ and a point x / ∈ χ , dist ( x, χ ) = min y ∈ χ || x -y || .

With G being the common Lipschitz constants for all g t 's, the constraint violation at time t ,

<!-- formula-not-decoded -->

## 3 Algorithm from Sinha and Vaze [2024]

The best known algorithm (Algorithm 1) to solve COCO Sinha and Vaze [2024] (in terms of regret and up to log factors for CCV) was shown to have the following guarantee.

2 The static-ness refers to the fixed benchmark using only one action x glyph[star] throughout the horizon of length T

## Algorithm 1 Online Algorithm from Sinha and Vaze [2024]

- 1: Input: Sequence of convex cost functions { f t } T t =1 and constraint functions { g t } T t =1 , G = a common Lipschitz constant, T = Horizon length, D = Euclidean diameter of the admissible set X , P X ( · ) = Euclidean projection oracle on the set X
- 2: Let β = (2 GD ) -1 , V = 1 , λ = 1 2 √ T , Φ( x ) = exp( λx ) -1 .
- 3: Initialization: Set x 1 = 0 , CCV (0) = 0 .
- 4: For t = 1 : T
- 5: Play x t , observe f t , g t , incur a cost of f t ( x t ) and constraint violation of ( g t ( x t )) +
- 6: ˜ f t ← βf t , ˜ g t ← β max(0 , g t ) .
- 7: CCV ( t ) = CCV ( t -1) + ˜ g t ( x t ) .
- 8: Compute ∇ t = ∇ ˆ f t ( x t ) , where ˆ f t ( x ) := V ˜ f t ( x ) + Φ ′ ( CCV ( t ))˜ g t ( x ) , t ≥ 1 . √
- 9: x t +1 = P X ( x t -η t ∇ t ) , where η t = 2 D 2 √ ∑ t τ =1 ||∇ τ || 2 2
- 10: EndFor

Theorem 5 [Sinha and Vaze [2024]] Algorithm 1's Regret [1: T ] = O ( √ T ) and CCV [1: T ] = O ( √ T log T ) when f t , g t are convex.

We next show that in fact the analysis of Sinha and Vaze [2024] is tight for the CCV even when d = 1 and f t ( x ) = f ( x ) and g t ( x ) = g ( x ) for all t . With finite diameter D and the fact that any x glyph[star] ∈ X glyph[star] belongs to all nested convex bodies S t 's, when d = 1 , one expects that the CCV for any algorithm in this case will be O ( D ) . However, as we show next, Algorithm 1 does not effectively make use of geometric constraints imposed by nested convex bodies S t 's.

Lemma 6 Even when d = 1 and f t ( x ) = f ( x ) and g t ( x ) = g ( x ) for all t , for Algorithm 1, its CCV [1: T ] = Ω( √ T log T ) .

Proof: Input: Consider d = 1 , and let X = [1 , a ] , a &gt; 2 . Moreover, let f t ( x ) = f ( x ) and g t ( x ) = g ( x ) for all t . Let f ( x ) = cx 2 for some (large) c &gt; 0 and g ( x ) be such that G = { x : g ( x ) ≤ 0 } ⊆ [ a/ 2 , a ] and let |∇ g ( x ) | ≤ 1 for all x .

Let 1 &lt; x 1 &lt; a/ 2 . Note that CCV ( t ) (defined in Algorithm 1) is a non-decreasing function, and let t glyph[star] be the earliest time t such that Φ ′ ( CCV ( t )) ∇ g ( x ) &lt; -c . For f ( x ) = cx 2 , ∇ f ( x ) ≥ c for all x &gt; 1 . Thus, using Algorithm 1's definition, it follows that for all t ≤ t glyph[star] , x t &lt; a/ 2 , since the derivative of f dominates the derivative of Φ ′ ( CCV ( t )) g ( x ) until then.

<!-- formula-not-decoded -->

Essentially, Algorithm 1 is treating minimizing the CCV problem as regret minimization for function g similar to function f and this leads to its CCV of Ω( √ T log T ) . For any given input instance with d = 1 , an alternate algorithm that chooses its actions following online gradient descent (OGD) projected on to the most recently revealed feasible set S t achieves O ( √ T ) regret (irrespective of the starting action x 1 ) and O ( D ) CCV (since any x glyph[star] ∈ S t for all t ). We extend this intuition in the next section, and present an algorithm that exploits the geometry of the nested convex sets S t for any d .

## 4 New Algorithm for solving COCO

In this section, we present a simple algorithm (Algorithm 2) for solving COCO. Algorithm 2 is essentially an online projected gradient algorithm (OGD), which first takes an OGD step from the previous action x t -1 with respect to the most recently revealed loss function f t -1 with appropriate step-size which is then projected onto S t -2 to reach y t -1 , and then projects y t -1 onto the most recently revealed set S t -1 to get x t , the action to be played at time t . (3).

Remark 1 Step 6 of Algorithm 2 might appear unnecessary, however, its useful for proving Theorem 12.

Since Algorithm 2 is essentially an online projected gradient algorithm, similar to the classical result on OGD, next, we show that the regret of Algorithm 2 is O ( √ T ) .

## Algorithm 2 Online Algorithm for COCO

- 1: Input: Sequence of convex cost functions { f t } T t =1 and constraint functions { g t } T t =1 , G = a common Lipschitz constant, d dimension of the admissible set X , step size η t = D G √ t . D = Euclidean diameter of the admissible set , ( · ) = Euclidean projection on the set ,
- X P X X 2: Initialization: Set x 1 ∈ X arbitrarily, CCV (0) = 0 . 3: For t = 1 : T 4: Play x t , observe f t , g t , incur a cost of f t ( x t ) and constraint violation of ( g t ( x t )) + 5: Set S t as defined in (3) 6: y t = P S t -1 ( x t -η t ∇ f t ( x t )) 7: x t +1 = P S t ( y t ) 8: EndFor

## Lemma 7 The Regret [1: T ] for Algorithm 2 is O ( √ T ) .

Extension of Lemma 7 when f t 's are strongly convex which results in Regret [1: T ] = O (log T ) for Algorithm 2 follows standard arguments Hazan [2012] and is omitted.

The real challenge is to bound the total CCV for Algorithm 2. Let x t be the action played by Algorithm 2. Then by definition, x t ∈ S t -1 . Moreover, from (4), the constraint violation at time t , CCV ( t ) ≤ G dist ( x t , S t ) . The next action x t +1 chosen by Algorithm 2 belongs to S t , however, it is obtained by first taking an OGD step from x t to reach y t and then projects y t onto S t . Since f t 's are arbitrary, the OGD step could be towards any direction, and thus, there is no direct relationship between x t +1 and x t . Informally, ( x 1 , x 2 , . . . , x T ) is not a connected curve with any useful property. Thus, we take recourse in upper bounding the CCV via upper bounding the total movement cost M (defined below) between nested convex sets using projections. The total constraint violation for Algorithm 2 is

<!-- formula-not-decoded -->

where in ( a ) b t is the projection of x t onto S t , i.e., b t = P S t ( x t ) and in ( b ) M t = ∑ t τ =1 || x τ -b τ || is defined to be the total movement cost on the instance S 1 , . . . , S t . The object of interest is M T .

## 5 Bounding the Total Movement Cost M T for Algorithm 2

We start by considering structured problem instances where CCV of Algorithm 2 is O (1) , i.e., independent of T .

Lemma 8 If all nested convex bodies S 1 ⊇ S 2 ⊇ · · · ⊇ S T are spheres then M T ≤ d 3 / 2 D = O (1) .

Lemma 9 If all nested convex bodies S 1 ⊇ S 2 ⊇ · · · ⊇ S T are cuboids/regular polygons that are axis parallel to each other, then M T ≤ d 3 / 2 D = O (1) .

Interestingly, input instance where S t 's are axis-parallel cuboids has been used to derive the only known lower bound for COCO of Regret [1: T ] = O ( √ d ) and CCV [1: T ] = O ( √ d ) [Sinha and Vaze, 2024].

Remark 2 Lemma 8 and 9 are first results of its kind in COCO, where even for nicely structured instances the previous best known guarantee is CCV [1: T ] = O ( √ T log T ) [Sinha and Vaze, 2024] or CCV [1: T ] = O ( √ T ) [Ferreira and Soares, 2025].

Next, we show that similar O (1) CCVguarantee can be obtained for Algorithm 2 with less structured input, however, only when d = 2 .

## 5.1 Special case of d = 2

In this section, we show that if d = 2 (all convex sets S t 's lie in a plane) and the projections satisfy a monotonicity property depending on the problem instance, then we can bound the total CCV for Algorithm 2 independent of the time horizon T and consequently getting a O (1) CCV.

Figure 1: Definition of F t 's.

<!-- image -->

Figure 2: Figure representing the cone C w t ( c t ) that contains the convex hull of m t and S t with unit vector w t ..

<!-- image -->

Definition 10 Recall from the definition of Algorithm 2, y t = P S t -1 ( x t -η t ∇ f t ( x t )) and x t +1 = P S t ( y t ) . Let the hyperplane perpendicular to line segment ( y t , x t +1 ) passing through x t +1 be F t . Without loss of generality, we let y t / ∈ S t , since otherwise the projection is trivial. Essentially F t is the projection hyperplane at time t . Let H + t denote the positive half plane corresponding to F t , i.e., H + t = { z : z T ( y t -x t +1 ) ≥ 0 } . Refer to Fig. 1. Let the angle between F 1 and F t be θ t .

Definition 11 The instance S 1 ⊇ S 2 ⊇ · · · ⊇ S T is defined to be monotonic if θ 2 ≤ θ 3 ≤ · · · ≤ θ T .

Theorem 12 For d = 2 when the instance is monotonic, CCV [1: T ] = O ( GD ) for Algorithm 2.

Theorem 12 shows that CCV of Algorithm 2 is independent of T as long as the instance is monotonic when d = 2 . It is worth noting that even under the monotonicity assumption it is non-trivial to upper bound the CCV since the successive angles made by F t 's with F 1 can increase arbitrarily slowly, making it difficult to control the total CCV. The proof is derived by using basic convex geometry results from Manselli and Pucci [1991] in combination with exploiting the definition of Algorithm 2 and the monotonicity condition.

Finally, in the next subsection, we upper bound M T , and consequently the CCV for Algorithm 2, when the input has no structure other than S t 's being nested.

## 5.2 General Guarantee on CCV

In this subsection, we give a general bound on M T of Algorithm 2 for any sequence of nested convex bodies which depends on the geometry of the nested convex bodies (instance dependent). To state the result we need the following preliminaries.

Following (5), b t = P S t ( x t ) where x t ∈ ∂S t -1 , where ∂S is the boundary of convex set S . Without loss of generality, x t / ∈ S t since otherwise the distance || x t -b t || = 0 . Let m t be the mid-point of x t and b t , i.e. m t = x t + b t 2 .

Definition 13 Let the convex hull of m t ∪ S t be C t . Let w t be a unit vector such that there exists c t &gt; 0 such that the cone

<!-- formula-not-decoded -->

contains C t . Since S t is convex, such w t , c t &gt; 0 exist. For example, w t = b t -x t is one such choice for which c t &gt; 0 since m t / ∈ S t . See Fig. 2 for a pictorial representation.

Let c glyph[star] w t ,t = arg max c t C w t ( c t ) , c glyph[star] t = max w t c glyph[star] w t ,t , and w glyph[star] t = arg max w t c glyph[star] w t ,t . Moreover, let c glyph[star] = min t c glyph[star] t , where by definition, c glyph[star] &lt; 1 .

Essentially, 2 cos -1 ( c glyph[star] t ) is the angle width of C t with respect to w glyph[star] t , i.e. each element of C t makes an angle of at most cos -1 ( c glyph[star] t ) with w glyph[star] t .

Remark 3 Note that c glyph[star] t is only a function of the distance || x t -b t || and the shape of S t 's, in particular, the maximum width of S t along the directions perpendicular to vector x t -b t ∀ t which

can be at most the diameter D . c glyph[star] t decreases (increasing the 'width' of cone C w glyph[star] t ( c glyph[star] t ) ) as || x t -b t || decreases, but small || x t -b t || also implies small violation at time t from (5) .

Remark 4 c glyph[star] is instance dependent or algorithm dependent? For notational simplicity, we have defined c glyph[star] using x t 's (Algorithm 2 specific quantity) and its projection b t on S t . However, since x t and x t -1 have no useful relation between them, x t can be any arbitrary point on the boundary of S t -1 , and c glyph[star] is in effect defined with respect to arbitrary x t ∈ S t -1 making it an instance-dependent quantity.

Lemma 14 M T for Algorithm 2 is at most 2 V d ( d -1) V d -1 ( 1 c glyph[star] ) d D, where V d is the ( d -1) -dimensional Lebesgue measure of the unit sphere in d dimensions.

Proof Idea Projecting x t ∈ ∂S t -1 onto S t to get b t = P S t ( x t ) , the diameter of S t is at most diameter of S t -1 -|| x t -b t || , however, only along the direction b t -x t . Since the shape of S t is arbitrary, as a result, the diameter of S t need not be smaller than the diameter of S t -1 along any pre-specified direction, which was the main idea used to derive Lemma 8. Thus, to prove Lemma 14 we relate the distance || x t -b t || with the decrease in mean width of a convex body, that is defined as the expected width of the convex body along all the directions that are chosen uniformly randomly (formal definition is provided in Definition 34).

√

Note that V d /V d -1 = O (1 / d ) . Thus, from Lemma 14 we get the following main result of the paper for Algorithm 2 combining Lemma 7 and Lemma 14.

<!-- formula-not-decoded -->

Theorem 15 is an instance dependent result for the CCV, compared to the prior universal guarantees of ˜ O ( √ T ) on the CCV. In particular, it exploits the geometric structure of the nested convex sets S t 's and derives an upper bound on the CCV that only depends on the 'shape' of S t 's via c glyph[star] . Moreover, c glyph[star] is only a dimension ( d ) dependent quantity (independent of T ) as long as the minimum distance between consecutive constraint sets is not function of T , since the diameter D is constant, whereas all existing algorithms will suffer from CCV of Ω( √ T ) even in this case.

Remark 5 One pertinent question at this time is: What is c glyph[star] and why should the CCV for a problem instance necessarily depend on it? c glyph[star] corresponds to the minimum angle width (via) of the problem instance, the angular width of the 'smallest' cone containing the newly revealed constraint sets. Angle width essentially depends on the width of the convex sets in directions perpendicular to the direction of projection, and controls the total CCV , since successive convex constraint sets are nested (lie inside each other), the smaller the angle width smaller is the room that an algorithm has to violate the constraints in future steps. Angle width also depends on the distance between x t and S t and is potentially large when d ( x t , S t ) is small and the diameter along the direction perpendicular to x t -b t is large.

c glyph[star] is a fundamental natural object that inherently captures the geometric difficulty in bounding the CCV. The core contribution of this paper is to formalize this by bringing in the novel concept of connecting the reduction of average width of the convex constraint set to the total constraint violation, that entails non-trivial convex analysis. If c glyph[star] is in fact small (e.g. total CCV is Ω( √ T ) ) for a problem instance then that problem instance does not have enough geometric features to extract via projections. To cover for such instances, we propose the Switch algorithm next to cap the CCV by ˜ O ( √ T ) .

## 6 Algorithm Switch

Theorem 15 provides an instance dependent bound on the CCV, that is a function of c glyph[star] . If c glyph[star] is small, CCV can be larger than O ( √ T log T ) , the CCV guarantee of Algorithm 1 [Sinha and Vaze, 2024]. Thus, next, we marry the two algorithms, Algorithm 1 and Algorithm 2, in Algorithm 3 to provide a best of both results as follows.

√

Theorem 16 Switch (Algorithm 3) has regret Regret [1: T ] = O ( T ) , while CCV [1: T ] = min { O ( √ d ( 1 c glyph[star] ) d D ) , O ( √ T log T ) } .

## Algorithm 3 Switch

- 1: Input: Sequence of convex cost functions { f t } T t =1 and constraint functions { g t } T t =1 , G = a common Lipschitz constant, d dimension of the admissible set X , D = Euclidean diameter of the admissible set X , P X ( · ) = Euclidean projection operator on the set X , 2: Initialization: Set x 1 ∈ X arbitrarily, CCV (0) = 0 . 3: For t = 1 : T 4: If CCV ( t -1) ≤ √ T log T 5: Follow Algorithm 2 and update CCV ( t ) = CCV ( t -1) + max { g t ( x t ) , 0 } . 6: Else 7: Follow Algorithm 1 with resetting CCV ( t -1) = 0 8: EndIf 9: EndFor

Algorithm Switch should be understood as the best of two worlds algorithm, where the two worlds correspond to one having nice convex sets S t 's that have CCV independent of T or o ( √ T ) for Algorithm 2, while in the other, CCV of Algorithm 2 is large on its own, and the overall CCV is controlled by discontinuing the use of Algorithm 2 once its CCV reaches √ T log T and switching to Algorithm 1 thereafter that has universal guarantee of O ( √ T log T ) on its CCV.

## 7 OCS Problem

In [Sinha and Vaze, 2024], a special case of COCO, called the OCS problem, was introduced where f t ≡ 0 for all t . Essentially, with OCS, only constraint satisfaction is the objective. In [Sinha and Vaze, 2024], Algorithm 1 was shown to have CCV of O ( √ T log T ) . Next, we show that Algorithm 2 has CCV of O (1) for the OCS, a remarkable improvement.

<!-- formula-not-decoded -->

As discussed in [Sinha and Vaze, 2024], there are important applications of OCS, and it is important to find tight bounds on its CCV. Theorem 17 achieves this by showing that CCV of O (1) can be achieved, where the constant depends only on the dimension of the action space and the diameter. This is a fundamental improvement compared to the CCV bound of O ( √ T log T ) from [Sinha and Vaze, 2024]. Theorem 17 is derived by using the connection between the curve obtained by successive projections on nested convex sets and self-expanded curves (Definition 20) and then using a classical result on self-expanded curves from [Manselli and Pucci, 1991].

## 8 Experimental Results

In this section, we compare the performance of Algorithm 1 and Algorithm 2 experimentally. We start by simulating the performance of Algorithm 1 and Algorithm 2 on the input that was used to prove Lemma 6. Fig. 3 numerically verifies the claim of Lemma 6 that the CCV of Algorithm 1 is Ω( √ T log T ) , while the CCV of Algorithm 2 remains constant.

## 8.1 Synthetic Data

Next, we consider a more reasonable data setup to compare the performance of Algorithm 1 and Algorithm 2, where with d = 10 , we let f t ( x ) = || x -a t || 1 , and a t is a d -dimensional vector that is coordinate-wise uniformly distributed between [ -1 , 1] and is independent across t . Similarly, we consider g t ( x ) = max(0 , w T t · x -0 . 1) where w t is a d -dimensional vector that also is coordinatewise uniformly distributed between [ -1 , 1] and is independent across t . This choice ensures that x = 0 is feasible for all constraints, i.e., Assumption 3 is satisfied. In Figs. 4a and 4b, we plot the regret and CCV, respectively, for Algorithm 1 and Algorithm 2, and see that Algorithm 2 outperforms Algorithm 1 in both the regret and the CCV.

Figure 3: Regret and CCV comparison for input described in Lemma 6.

<!-- image -->

(a) Regret comparison of Algorithm 1 and Algorithm 2

<!-- image -->

## 9 Conclusions

One fundamental open question for COCO is: whether it is possible to simultaneously achieve R [1: T ] = O ( √ T ) and CCV [1: T ] = o ( √ T ) or CCV [1: T ] = O (1) . In this paper, we have made substantial progress towards answering this question by proposing an algorithm that exploits the geometric properties of the nested convex sets S t 's that effectively control the CCV. The state of the art algorithms [Sinha and Vaze, 2024, Ferreira and Soares, 2025] achieve a CCV of ˜ Ω( √ T ) even for very simple instances as shown in Lemma 6, and conceptually different algorithms are needed to achieve CCV of o ( √ T ) . We propose one such algorithm and show that when the nested convex constraint sets are well structured, achieving a CCV of O (1) is possible without losing out on O ( √ T ) regret guarantee. We also derived a bound on the CCV for general problem instances, that is as a function of the shape of nested convex constraint sets and the distance between them, and the diameter.

In the absence of good lower bounds, the open question remains unresolved in general, however, this paper significantly improves the conceptual understanding of COCO problem by demonstrating that good algorithms need to exploit the geometry of the nested convex constraint sets.

## References

Jianjun Yuan and Andrew Lamperski. Online convex optimization for cumulative constraints. Advances in Neural Information Processing Systems , 31, 2018.

Rodolphe Jenatton, Jim Huang, and C´ edric Archambeau. Adaptive algorithms for online convex optimization with long-term constraints. In International Conference on Machine Learning , pages

(b) CCV comparison of Algorithm 1 and Algorithm 2

<!-- image -->

402-411. PMLR, 2016.

- Mehrdad Mahdavi, Rong Jin, and Tianbao Yang. Trading regret for efficiency: online convex optimization with long term constraints. The Journal of Machine Learning Research , 13(1):25032528, 2012.
- Xinlei Yi, Xiuxian Li, Tao Yang, Lihua Xie, Tianyou Chai, and Karl Johansson. Regret and cumulative constraint violation analysis for online convex optimization with long term constraints. In International Conference on Machine Learning , pages 11998-12008. PMLR, 2021.
- Hao Yu, Michael Neely, and Xiaohan Wei. Online convex optimization with stochastic constraints. Advances in Neural Information Processing Systems , 30, 2017.
- Wen Sun, Debadeepta Dey, and Ashish Kapoor. Safety-aware algorithms for adversarial contextual bandit. In International Conference on Machine Learning , pages 3280-3288. PMLR, 2017.
- Xinlei Yi, Xiuxian Li, Tao Yang, Lihua Xie, Yiguang Hong, Tianyou Chai, and Karl H Johansson. Distributed online convex optimization with adversarial constraints: Reduced cumulative constraint violation bounds under slater's condition. arXiv preprint arXiv:2306.00149 , 2023.
- Michael J Neely and Hao Yu. Online convex optimization with time-varying constraints. arXiv preprint arXiv:1702.04783 , 2017.
- Nikolaos Liakopoulos, Apostolos Destounis, Georgios Paschos, Thrasyvoulos Spyropoulos, and Panayotis Mertikopoulos. Cautious regret minimization: Online optimization with long-term budget constraints. In International Conference on Machine Learning , pages 3944-3952. PMLR, 2019.
- Michael J Neely. Stochastic network optimization with application to communication and queueing systems. Synthesis Lectures on Communication Networks , 3(1):1-211, 2010.
- Hengquan Guo, Xin Liu, Honghao Wei, and Lei Ying. Online convex optimization with hard constraints: Towards the best of two worlds and beyond. Advances in Neural Information Processing Systems , 35:36426-36439, 2022.
- Abhishek Sinha and Rahul Vaze. Optimal algorithms for online convex optimization with adversarial constraints. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=TxffvJMnBy .
- Ricardo N. Ferreira and Cl´ audia Soares. Optimal bounds for adversarial constrained online convex optimization, 2025. URL https://arxiv.org/abs/2503.13366 .
- Jordan Lekeufack and Michael I. Jordan. An optimistic algorithm for online convex optimization with adversarial constraints, 2025. URL https://arxiv.org/abs/2412.08060 .
- Tianyi Chen and Georgios B Giannakis. Bandit convex optimization for scalable and dynamic iot management. IEEE Internet of Things Journal , 6(1):1276-1286, 2018.
- Xuanyu Cao and KJ Ray Liu. Online convex optimization with time-varying constraints and bandit feedback. IEEE Transactions on automatic control , 64(7):2665-2680, 2018.
- Rahul Vaze. On dynamic regret and constraint violations in constrained online convex optimization. In 2022 20th International Symposium on Modeling and Optimization in Mobile, Ad hoc, and Wireless Networks (WiOpt) , pages 9-16, 2022. doi: 10.23919/WiOpt56218.2022.9930613.
- Qingsong Liu, Wenfei Wu, Longbo Huang, and Zhixuan Fang. Simultaneously achieving sublinear regret and constraint violations for online convex optimization with time-varying constraints. ACM SIGMETRICS Performance Evaluation Review , 49(3):4-5, 2022.
- Nikhil Bansal, Martin B¨ ohm, Marek Eli´ aˇ s, Grigorios Koumoutsos, and Seeun William Umboh. Nested convex bodies are chaseable. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 1253-1260. SIAM, 2018.

- C.J. Argue, S´ ebastien Bubeck, Michael B Cohen, Anupam Gupta, and Yin Tat Lee. A nearly-linear bound for chasing nested convex bodies. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 117-122. SIAM, 2019.
- S´ ebastien Bubeck, Bo'az Klartag, Yin Tat Lee, Yuanzhi Li, and Mark Sellke. Chasing nested convex bodies nearly optimally. In Proceedings of the Fourteenth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 1496-1508. SIAM, 2020.
- Elad Hazan. The convex optimization approach to regret minimization. Optimization for machine learning , page 287, 2012.

√

- Hao Yu and Michael J Neely. A low complexity algorithm with o ( T ) regret and o (1) constraint violations for online convex optimization with long term constraints. arXiv preprint arXiv:1604.02218 , 2016.
- Paolo Manselli and Carlo Pucci. Maximum length of steepest descent curves for quasi-convex functions. Geometriae Dedicata , 38(2):211-227, 1991.

Harold Gordon Eggleston. Convexity, 1966.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide complete theorem statements and proofs of all claims.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Our result crucially makes use of feasibility assumption (Assumption 3) which is universally used in the COCO literature. In the absence of good lower bounds, the problem considered in the paper question remains open in full generality.

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

Answer: [Yes]

Justification: We clearly state the assumptions under which our theoretical results hold.

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

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

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

Answer: [NA]

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: This paper deals with fundamental optimization theory and conform with NeurIPS Code of Ethics

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a theoretical paper and the authors do not see any immediate direct societal impact of this paper.

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

Justification: This theoretical paper does not pose any such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

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

Guidelines:

- The answer NA means that the paper does not release new assets.

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

## 10 Comparison with [Lekeufack and Jordan, 2025]

Remark 6 [Lekeufack and Jordan, 2025] consider the COCO problem when predictions about both cost functions f t 's and constraint functions g t 's are available. With predictions, they show that if predictions are perfect, O (1) regret and CCV is achievable, while if the predictions are totally wrong, in the worst case the regret and CCV are at most as bad as the result of [Sinha and Vaze, 2024]. Intermediate range of results is also obtained depending on the quality of prediction. Essentially [Lekeufack and Jordan, 2025] use the prediction wrapper over the algorithm of [Sinha and Vaze, 2024] to derive their guarantee.

In this paper, however, we are not assuming any predictions , and are solving the COCO problem with the worst case input, similar to all the prior work listed in Table 1. Moreover, the presented algorithm is conceptually different than [Sinha and Vaze, 2024], and for the first time shows that O (1) or instance dependent CCV while having O ( √ T ) regret is possible, which is not the case with prior work even for d = 1 .

Thus, the setting of [Lekeufack and Jordan, 2025] is completely different and not really comparable with our results.

## 11 Proof of Lemma 7

Proof: From the convexity of f t 's, for x glyph[star] satisfying Assumption (3), we have

<!-- formula-not-decoded -->

From the choice of Algorithm 2 for x t +1 , we have

<!-- formula-not-decoded -->

where inequalities ( a ) and ( b ) follow since x glyph[star] ∈ S t for all t . Hence

<!-- formula-not-decoded -->

Summing this over t = 1 to T , we get

<!-- formula-not-decoded -->

where the final inequality follows by choosing η t = D G √ t .

## 12 Proof of Lemma 8 and Lemma 9.

Proof: [Proof of Lemma 8] Recall the definition that x t ∈ ∂S t -1 , b t = P S t ( x t ) ∈ S t from (5). Let || x t -b t || = r , then since all S t 's are spheres, at least along one of the d -orthogonal canonical basis vectors, diameter ( S t ) ≤ diameter ( S t -1 ) -r √ d . Since the diameter along any of the d -axis is D , we get the answer. ✷ We would like to remark that the proof is short and elementary that should be seen as a strength. Proof: [Proof of Lemma 9] Proof is identical to Lemma 8. ✷

✷

## 13 Preliminaries for Bounding the CCV in Theorem 12 and Theorem 17

Let K 1 , . . . , K T be nested (i.e., K 1 ⊇ K 2 ⊃ K 3 ⊇ · · · ⊇ K T ) bounded convex subsets of R d .

Definition 18 If σ 1 ∈ K 1 , and σ t +1 = P K t +1 ( σ t ) , for t = 1 , . . . , T . Then the curve

<!-- formula-not-decoded -->

is called the projection curve on K 1 , . . . , K T .

We are interested in upper bounding the quantity

<!-- formula-not-decoded -->

Lemma 19 For a projection curve σ , Σ ≤ d d/ 2 diameter ( K 1 ) .

To prove the result we need the following definition.

Definition 20 A curve γ : I → R d is called self-expanded, if for every t where γ ′ ( t ) exists, we have

<!-- formula-not-decoded -->

for all u ∈ I with u ≤ t , where &lt; ., . &gt; represents the inner product. In words, what this means is that γ starting in a point x 0 is self expanded, if for every x ∈ γ for which there exists the tangent line T , the arc (sub-curve) ( x 0 , x ) is contained in one of the two half-spaces, bounded by the hyperplane through x and orthogonal to T .

For self-expanded curves the following classical result is known.

Theorem 21 Manselli and Pucci [1991] For any self-expanded curve γ belonging to a closed bounded convex set of R d with diameter D , its total length is at most O ( d d/ 2 D ) .

Proof: [Proof of Lemma 19] From Definition 18, the projection curve is

<!-- formula-not-decoded -->

Let the reverse curve be r = { r t } t =0 ,...,T -2 , where r t = ( σ T -t , σ T -t -1 ) . Thus we are reading σ backwards and calling it r . Note that since σ t is the projection of σ t -1 on K t , each piece-wise linear segment ( σ t , σ t +1 ) is a straight line and hence differentiable except at the end points. Moreover, since each σ t is obtained by projecting σ t -1 onto K t and K t +1 ⊆ K t , we have that the projection hyperplane F t that passes through σ t = P K t ( σ t -1 ) and is perpendicular to σ t -σ t -1 separates the two sub curves { ( σ 1 , σ 2 ) , ( σ 2 , σ 3 ) , . . . , ( σ t -1 , σ t ) } and { ( σ t , σ t +1 ) , ( σ t +1 , σ t +2 ) , . . . , ( σ T -1 , σ T ) } .

Thus, we have that for each segment r τ , at each point where it is differentiable, the curve r 1 , . . . r τ -1 lies on one side of the hyperplane that passes through the point and is perpendicular to r τ +1 . Thus, we conclude that curve r is self-expanded.

As a result, Theorem 21 implies that the length of r is at most O ( d d/ 2 diameter ( K 1 )) , and the result follows since the length of r is same as that of σ which is Σ . ✷

## 14 Proof of Theorem 12

Proof: Recall that d = 2 , and the definition of F t from Definition 10. Let the center be c = P S 1 ( x 1 ) . Let t orth be the earliest t for which ∠ ( F t , F 1 ) = π .

Initialize κ = 1 , s (1) = 1 , τ (1) = 1 .

BeginProcedure Step 1:Definition of Phase κ . Consider

<!-- formula-not-decoded -->

If there is no such τ ( κ ) ,

Phase κ ends, define Phase κ as Empty , s ( κ +1) = τ ( κ ) + 1 .

## Else If

<!-- formula-not-decoded -->

## Else If

<!-- formula-not-decoded -->

## End If

Increment κ = κ +1 , and Go to Step 1.

## EndProcedure

Example 22 To better understand the definition of phases, consider Fig. 5, where the largest t for which the angle between F t and F 1 is at most π/ 4 is 3 . Thus, τ (1) = 3 , i.e., phase 1 explores till time t = 3 and phase 1 ends. The starting hyperplane to consider in phase 2 is s (2) = 3 and given that angle between F 3 and and the next hyperplane F 4 is more than π/ 4 , phase 2 is empty and phase 2 ends by exploring till t = 4 . The starting hyperplane to consider in phase 3 is s (3) = 4 and the process goes on. The first time t such that the angle between F 1 and F t is π is t = 6 , and thus t orth = 6 , and the process stops at time t = 6 . This also implies that S 6 ⊂ F 1 . Since S t 's are nested, for all t ≥ 6 , S t ⊂ F 1 . Hence the total CCV after t ≥ t orth is at most GD .

The main idea with defining phases, is to partition the whole space into empty and non-empty regions, where in each non-empty region, the starting and ending hyperplanes have an angle to at most π/ 4 , while in an empty phase the starting and ending hyperplanes have an angle of at least π/ 4 . Thus, we get the following simple result.

Lemma 23 For d = 2 , there can be at most 4 non-empty and 4 empty phases.

Proof is immediate from the definition of the phases, since any consecutively occurring non-empty and empty phase exhausts an angle of at least π/ 4 .

Remark 7 Since we are in d = 2 dimensions, for all t ≥ t orth, the movement is along the hyperplane F 1 and thus the resulting constraint violation after time t ≥ t orth is at most GD . Thus, in the phase definition above, we have only considered time till t orth and we only need to upper bound the CCV till time t orth .

Figure 5: Figure corresponding to Example 22.

<!-- image -->

We next define the following required quantities.

Definition 24 With respect to the quantities defined for Algorithm 2, let for a non-empty phase κ

<!-- formula-not-decoded -->

t glyph[star] ( κ ) is the time index belonging to phase κ for which y t is the farthest.

Definition 25 A non-empty phase κ consists of time slots T ( κ ) = [ τ ( κ -1) , τ ( κ )] and the angle ∠ ( F t 1 , F t 2 ) ≤ π/ 4 for all t 1 , t 2 ∈ T ( κ ) . Using Definition 24, we partition T ( κ ) as T ( κ ) = T -( κ ) ∪ T + ( κ ) , where T -( κ ) = [ τ ( κ -1) + 1 , t glyph[star] ( κ ) + 1] and T + ( κ ) = [ t glyph[star] ( κ ) + 2 , τ ( κ )] .

Thus, T ( κ ) and T ( κ +1) have one common time slot.

Definition 26 [Definition of z t ( κ ) for t ∈ T -( κ ) ]. Let z t glyph[star] ( κ )+1 = x t glyph[star] ( κ )+1 . For t ∈ T -( κ ) \ t glyph[star] ( κ ) + 1 , define z t ( κ ) inductively as follows. z t ( κ ) is the pre-image of z t +1 ( κ ) on F t -1 such that the projection of z t ( κ ) on F t is z t +1 ( κ ) .

Definition 27 [Definition of z t ( κ ) for t ∈ T + ( κ ) ]. For t ∈ T + ( κ ) , define z t ( κ ) inductively as follows. z t ( κ ) is the projection of z t -1 ( κ ) on F t -1 .

See Fig. 6 for a visual illustration of t glyph[star] ( κ ) and z t ( κ ) .

Figure 6: Illustration of definition of z t ( κ ) for t ∈ T ( κ ) . In this example, for phase 1 , t glyph[star] (1) = 3 since the distance of y 3 from c is the farthest for phase 1 that consists of time slots T (1) = { 2 , 3 } . Hence z t glyph[star] (1)+1 (1) = x 4 . For t ∈ T (1) \ t glyph[star] (1) + 1 , z t (1) are such z t +1 (1) is a projection of z t (1) onto F t .

<!-- image -->

The main idea behind defining z t ( κ ) 's is as follows. For each non-empty phase, we will construct a projection curve (Definition 18) using points z k such that the length of the projection curve upper bounds the CCV of Algorithm 2 (shown in Lemma 33), and then use Lemma 19 to upper bound the length of the projection curve.

Definition 28 [Definition of S ′ t for a non-empty phase κ :] S ′ t glyph[star] ( κ )+1 = S t glyph[star] ( κ )+1 . For t ∈ T -( κ ) \ t glyph[star] ( κ ) + 1 , S ′ t is the convex hull of z t +1 ( κ ) ∪ S t ∪ S ′ t +1 ( κ ) . For t ∈ T + ( κ ) , S ′ t = S t . See Fig. 7.

Lemma 29 For a non-empty phase κ , for any t ∈ T ( κ ) , S ′ t +1 ⊆ S ′ t , i.e. they are nested.

Definition 30 For a non-empty phase, χ ( κ ) = S ′ τ ( κ -1) ∩ H + τ ( κ ) , where H + τ ( κ ) has been defined in Definition 10.

Definition 31 [New Violations for t ∈ T ( κ ) :] For a non-empty phase κ , for t ∈ T ( κ ) \ τ ( κ -1) , let

<!-- formula-not-decoded -->

Figure 7: Definition of S t 's where U t are the extra regions that are added to S t to get S ′ t .

<!-- image -->

Lemma 32 For each non-empty phase κ , all z t ( κ ) 's for t ∈ T ( κ ) belongs to B ( c , √ 2 D ) , where B ( c, r ) is a ball with radius r centered at c . In other words, χ ( κ ) ⊆ B ( c , √ 2 D ) .

Proof: Recall that for a non-empty phase κ , T ( κ ) = T -( κ ) ∪ T + ( κ ) . We first argue about t ∈ T -( κ ) . By definition, z t glyph[star] ( κ )+1 = x t glyph[star] ( κ )+1 and x t glyph[star] ( κ )+1 ∈ S t glyph[star] ( κ ) . Thus, z t glyph[star] ( κ )+1 ∈ B ( c , √ 2 D ) . Next we argue for t ∈ T -( κ ) \ t glyph[star] ( κ ) + 1 . Recall that the diameter of X is D , and the fact that y t ∈ S t -1 from Algorithm 2. Thus, for any non-empty phase κ , the distance from c to the farthest y t belonging to the phase κ is at most D , i.e., r max ( κ ) ≤ D . Let the pre-image of z t glyph[star] ( κ )+1 ( κ ) onto F s ( κ ) (the base hyperplane with respect to which all hyperplanes have an angle of at most π/ 4 in phase κ ) be p ( κ ) such that projection of p ( κ ) onto F s ( κ ) is z t glyph[star] ( κ )+1 ( κ ) . From the definition of any non-empty phase, the angle between F s ( κ ) and F t for t ∈ T ( κ ) is at most π/ 4 . Thus, the distance of p ( κ ) from c is at most √ 2 D .

Consider the 'triangle' Π( κ ) that is the convex hull of c , z t glyph[star] ( κ )+1 ( κ ) and p ( κ ) . Given that the angle between F t glyph[star] ( κ ) and F t glyph[star] ( κ ) -1 is at most π/ 4 , the argument above implies that z t ( κ ) ∈ Π( κ ) for t = t glyph[star] ( κ ) . For t = t glyph[star] ( κ ) -1 , z t ( κ ) ∈ F t -1 is the projection of z t -1 ( κ ) onto S ′ t -1 . This implies that the distance of z t ( κ ) (for t = t glyph[star] ( κ ) -1 ) from c is at most

<!-- formula-not-decoded -->

where α t 1 ,t 2 is the angle between F t 1 and F t 2 . From the monotonicity of angles θ t (Definition 11), and the definition of a non-empty phase, we have that α t,t glyph[star] ( κ ) + α t glyph[star] ( κ ) ,t glyph[star] ( κ )+1 ≤ π/ 4 and α t,t glyph[star] ( κ ) ≥ 0 , α t glyph[star] ( κ ) ,t glyph[star] ( κ )+1 ≥ 0 . Next, we appeal to the identity

<!-- formula-not-decoded -->

where A + B ≤ π/ 4 , to claim that z t ( κ ) ∈ Π( κ ) for t = t ( κ ) -1 .

Iteratively using this argument while invoking the identity (7) gives the result that for any t ∈ T -( κ ) , we have that z t ( κ ) belongs to Π( κ ) . Since Π( κ ) ⊆ B ( c , √ 2 D ) , we have the claim for all t ∈ T -( κ ) .

By definition z t ( κ ) for t ∈ T + ( κ ) belong to S t -1 ⊆ S 1 . Thus, their distance from c is at most D . ✷

Lemma 33 For each non-empty phase κ , and for t ∈ T ( κ ) the violation v t ( κ ) ≥ dist ( x t , S t ) , where dist ( x t , S t ) is the original violation.

Proof: By construction of any non-empty phase κ , for t ∈ T ( κ ) both x t ( κ ) and z t ( κ ) belong to F t -1 . Moreover, by construction, the distance of z t ( κ ) from c is at least as much as the distance of x t from c . Thus, using the monotonicity property of angles θ t (Definition 11) we get the result. See Fig. 6 for a visual illustration. ✷

For each non-empty phase κ , by definition, the curve defined by sequence z t ( κ ) for t ∈ T ( κ ) is a projection curve (Definition 18) on sets S ′ t ( κ ) (note that S ′ t ( κ ) 's are nested from Lemma 29). Moreover, for all t ∈ T ( κ ) , set S ′ t ( κ ) ⊂ χ ( κ ) which is a bounded convex set. Thus, for d = 2 from Lemma 19 the length of curve z ( κ ) = { ( z t ( κ ) , z t +1 ( κ )) } t ∈ T ( κ )

<!-- formula-not-decoded -->

By definition, the number of non-empty phases till time t orth is at most 4 . Moreover, in each nonempty phase χ ( κ ) ⊆ B ( c , √ 2 D ) from Lemma 32 .

Thus, from (8), we have that

<!-- formula-not-decoded -->

Using Lemma 33, we get

<!-- formula-not-decoded -->

For any empty phase, the constraint violation is the length of line segment ( x t , P S t ( x t )) (Algorithm 2) crossing it is a straight line whose length is at most O ( D ) . Moreover, the total number of empty phases (Lemma 23) is a constant. Thus, the length of the curve ( x t , P S t ( x t )) for Algorithm 2 corresponding to all empty phases is at O ( D ) .

Recall from (4) that the CCV is at most G times dist ( x t , S t ) . Thus, from (10) we get that the total violation incurred by Algorithm 2 corresponding to non-empty phases is at most O ( GD ) , while corresponding to empty phases is at O ( GD ) . Finally, accounting for the very first violation dist ( x 1 , S 1 ) ≤ D and the fact that the CCV after time t ≥ t orth (Remark 7) is at most GD , we get that the total constraint violation CCV [1: T ] for Algorithm 2 is at most O ( GD ) .

✷

## 15 Proof of Theorem 14

Proof: We need the following preliminaries.

Definition 34 Let K be a non-empty convex bounded set in R d . Let u be a unit vector, and glyph[lscript] u a line through the origin parallel to u . Let K u be the orthogonal projection of K onto glyph[lscript] u , with length | K u | . The mean width of K is defined as

<!-- formula-not-decoded -->

where S d 1 is the unit sphere in d dimensions and V d its ( d -1) -dimensional Lebesgue measure.

The following is immediate.

<!-- formula-not-decoded -->

Lemma 35 Eggleston [1966] For d = 2 ,

<!-- formula-not-decoded -->

<!-- image -->

Hu

H'

Figure 8: Figure representing the cone C w t ( c t ) that contains the convex hull of m t and S t with respect to the unit vector w t . u is a unit vector perpendicular to H u an hyperplane that is a supporting hyperplane C t at m t such that C t ∩ H u = { m t } and u T ( x t -m t ) ≥ 0

glyph[negationslash]

Lemma 35 implies that W ( K ) = W ( K 1 ) + W ( K 2 ) even if K 1 ∪ K 2 = K and K 1 ∩ K 2 = φ .

Recall from (5) that x t ∈ ∂S t -1 and b t is the projection of x t onto S t , and m t is the mid-point of x t and b t , i.e. m t = x t + b t 2 . Moreover, the convex sets S t 's are nested, i.e., S 1 ⊇ S 2 ⊇ · · · ⊇ S T . To prove Theorem 14 we will bound the rate at which W ( S t ) (Definition 34) decreases as a function of the length || x t -b t || .

From Definition 13, recall that C t is the convex hull of m t ∪ S t . We also need to define C -t as the convex hull of x t ∪ S t . Since S t ⊆ C t and C -t ⊆ S t -1 (since S t -1 is convex and x t ∈ S t -1 ), we have

<!-- formula-not-decoded -->

Definition 36 ∆ t = W ( C t ) -W ( C -t ) .

The main ingredient of the proof is the following Lemma that bounds ∆ t whose proof is provided after completing the proof of Theorem 14.

## Lemma 37

<!-- formula-not-decoded -->

where c glyph[star] t has been defined in Definition 13.

Recalling that c glyph[star] = min t c glyph[star] t from Definition 13, and combining Lemma 37 with (12) and (13), we get that

<!-- formula-not-decoded -->

since S 1 ⊇ S 2 ⊇ · · · ⊇ S T . Recalling that diameter ( S 1 ) ≤ D , Theorem 14 follows. ✷

Proof: [Proof of Lemma 37]

Let H u be the hyperplane perpendicular to vector u . Let U 0 be the set of unit vectors u such that hyperplanes H u are supporting hyperplanes to C t at point m t such that C t ∩ H u = { m t } and u T ( x t -m t ) ≥ 0 . See Fig. 8 for reference.

Since b t is a projection of x t onto S t , and m t is the mid-point of x t , b t , for u ∈ U 0 , the hyperplane H ′ u containing x t and parallel to H u is a supporting hyperplane for C -t .

Thus, using the definition of K u from (11),

<!-- formula-not-decoded -->

since || x t -m t || = || x t -b t || / 2 .

Recall the definition of C w glyph[star] t ( c glyph[star] t ) from Definition 13 which implies that the convex hull of m t and S t , C t is contained in C w glyph[star] t ( c glyph[star] t ) . Next, we consider U 1 the set of unit vectors u such that hyperplanes H u are supporting hyperplanes to C w glyph[star] t ( c glyph[star] t ) at point m t such that u T ( x t -m t ) ≥ 0 . By definition C t ⊆ C w glyph[star] t ( c glyph[star] t ) , it follows that U 1 ⊂ U 0 .

Thus, from (14)

<!-- formula-not-decoded -->

Recalling the definition of w glyph[star] t (Definition 13), vector u ∈ U 1 can be written as

<!-- formula-not-decoded -->

where u T ⊥ w glyph[star] t = 0 , | u ⊥ | = 1 and since u ∈ U 1

<!-- formula-not-decoded -->

Let S ⊥ = { u ⊥ : | u ⊥ | = 1 , u T ⊥ w glyph[star] t = 0 } . Let du ⊥ be the ( n -2) -dimensional Lebesgue measure of S ⊥ .

It is easy to verify that du = λ d -2 (1 -λ 2 ) -1 / 2 dλdu ⊥ and hence from (15)

<!-- formula-not-decoded -->

Note that ∫ du ⊥ u ⊥ du ⊥ = 0 . Thus,

<!-- formula-not-decoded -->

where ( a ) follows since ∫ S ⊥ du ⊥ = V d -1 by definition, ( b ) follows since ( w glyph[star] t ) T ( x t -m t ) || x t -m t || ≥ c glyph[star] t from Definition 13.

✷

## 16 Proof of Theorem 16

Proof: Since CCV ( t ) is a monotone non-decreasing function, let t min be the largest time until which Algorithm 2 is followed by Switch . The regret guarantee is easy to prove. From Theorem 15, regret until time t min is at most O ( √ t min ) . Moreover, starting from time t min till T , from Theorem 5, the regret of Algorithm 1 is at most O ( √ T -t min ) . Thus, the overall regret for Switch is at most O ( √ T ) .

For the CCV, with Switch , until time t min , CCV ( t min ) ≤ √ T log T . At time t min , Switch starts to use Algorithm 1 which has the following appealing property from (8) Sinha and Vaze [2024] that for

any t ≥ t min where at time t min Algorithm 1 was started to be used with resetting CCV ( t min ) = 0 . For any t ≥ t min

<!-- formula-not-decoded -->

where β = (2 GD ) -1 , V = 1 , λ = 1 2 √ T , Φ( x ) = exp( λx ) -1 , and λ = 1 2 √ T . We trivially have Regret t ( x glyph[star] ) ≥ -Dt 2 D ≥ -t 2 . Hence, from (18), we have that for any λ = 1 2 √ T and any t ≥ t min

<!-- formula-not-decoded -->

Since as argued before, with Switch , CCV ( t min ) ≤ √ T log T , we get that CCV [1: T ] ≤ O ( √ T log T ) . ✷

## 17 Proof of Theorem 17

Clearly, with f t ≡ 0 for all t , with Algorithm 2, y t = x t and the successive x t 's are such that x t +1 = P S t ( x t ) . Thus, essentially, the curve x = ( x 1 , x 2 ) , ( x 2 , x 3 ) , . . . , ( x T -1 , x T ) formed by Algorithm 2 for OCS is a projection curve (Definition 18) on S 1 ⊇ , . . . , ⊇ S T and the result follows from Lemma 19 and the fact that diameter ( S 1 ) ≤ D .
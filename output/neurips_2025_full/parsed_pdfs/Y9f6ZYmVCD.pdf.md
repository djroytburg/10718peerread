## Strategic Classification with Non-Linear Classifiers

## Benyamin Trachtenberg, Nir Rosenfeld

Technion - Israel Institute of Technology Haifa, Israel

{benyamint, nirr} @cs.technion.ac.il

## Abstract

In strategic classification, the standard supervised learning setting is extended to support the notion of strategic user behavior in the form of costly feature manipulations made in response to a classifier. While standard learning supports a broad range of model classes, the study of strategic classification has, so far, been dedicated mostly to linear classifiers. This work aims to expand the horizon by exploring how strategic behavior manifests under non-linear classifiers and what this implies for learning. We take a bottom-up approach showing how non-linearity affects decision boundary points, classifier expressivity, and model class complexity. Our results show how, unlike the linear case, strategic behavior may either increase or decrease effective class complexity, and that the complexity decrease may be arbitrarily large. Another key finding is that universal approximators (e.g., neural nets) are no longer universal once the environment is strategic. We demonstrate empirically how this can create performance gaps even on an unrestricted model class.

## 1 Introduction

Strategic classification considers learning in a setting where users modify their features in response to a deployed classifier to obtain positive predictions [3, 4, 16]. This setting aims to capture a natural tension that can arise between a system aiming to maximize accuracy and its users who seek favorable outcomes. Such a tension is well-motivated across diverse applications, including loan approval, university admission, job hiring, welfare and education programs, and medical services. The field has drawn much attention since its introduction, leading to advances along multiple learning fronts such as generalization [6, 33, 36], optimization [25], loss functions [24, 26], and extensions [11, 14, 18, 30].

However, despite over a decade of effort, research in strategic classification remains predominantly focused on linear classification. Due to the inherent challenges of strategic learning, this focus has been reasonable, given that linearity has the benefit of inducing a simple form of strategic behavior that permits tractable analysis. Yet, since many realistic applications-including those mentioned above-make regular use of non-linear models, we believe that it is important to establish a better understanding of how non-linearity shapes strategic behavior, and how behavior in turn affects learning. Our paper therefore aims to extend the study of strategic classification to the non-linear regime.

Our first observation is that non-linear classifiers induce qualitatively different behavior than linear ones in the strategic setting. Consider first the structure induced by standard linear classifiers. Generally, users in strategic classification are assumed to best-respond to the classifier by modifying their features in a way that maximizes utility (from prediction) minus costs (from modification). For a linear classifier h coupled with the common choice of an ℓ 2 cost, modifications x ↦→ x h manifest as a projection of x onto the linear decision boundary of h as long as it incurs less than a unit cost. This simple form has several implications. First, note that all points 'move' in the same direction, making strategic behavior, in effect, one-dimensional. Second, the region of points that move forms a band of unit distance on the negative side of h . This means that the set of points classified as positive

forms a halfspace, which implies that the 'effective classifier', given by h ∆ ( x ) := h ( x h ) , is also linear. Third, and as a result, the induced effective model class remains linear.

Together, the above suggest that for linear classifiers coupled with ℓ 2 costs, the transition from the standard, non-strategic setting to a strategic one requires anticipating which points move, but otherwise remains similar. One known implication is that the strategic and non-strategic VC dimension of linear classifiers are the same [33]. Furthermore, the Bayes-optimal strategic classifier can be derived by taking the optimal non-strategic classifier and simply increasing its bias term by one [29]. Note that this means that any linearly separable problem remains separable, and that strategic classifiers can, in principle, attain the same level of accuracy as their non-strategic counterparts. Thus, while perhaps challenging to solve in practice, the intrinsic difficulty of strategic linear classification remains unchanged.

The above, however, is unlikely to hold for non-linear classifiers: once the decision boundary is non-linear, different points will move in different directions. This lends to the possible distortion of the shape of a given classifier, which in turn can change the effective (i.e., strategic) model class. Such changes can have concrete implications on expressivity, generalization, optimization, and attainable accuracy. This motivates the need to explore how strategic responses shape outcomes in non-linear strategic classification, which we believe is crucial as a first step towards developing future methods. Our general approach is to examine-through analysis, examples, and experiments-how objects in the original non-strategic task map to corresponding objects in the induced strategic problem instance. Initially, our conjecture was that strategic behavior acts as a 'smoothing' operator on the original decision boundary, but as we show, the actual effect of strategic behavior turns out to be more involved.

To build an intuition of the mechanisms at play, we take a bottom-up approach and study three mappings between (i) individual points, (ii) classifiers, and (iii) model classes. For points, we begin by showing that the mapping is not necessarily bijective. As a result, some points on the original decision boundary become 'wiped out', while others are 'expanded' to become intervals (arcs for ℓ 2 costs) on the induced boundary. We demonstrate several mechanisms that can give rise to both phenomena. For classifiers, we show how the effects on individual points aggregate to modify the entire decision boundary. A key finding is that certain decision boundaries are impossible to express under strategic behavior. We give several examples of reasonable classifier types that cannot exist under strategic behavior, and show that some effective classifiers h ∆ can be traced back to (infinitely) many original classifiers h . The main takeaway is that universal approximators, such as neural networks or RBF kernels, are no longer universal in a strategic environment.

Our main contribution lies in our analysis of model classes, where our results build on the observation that the mapping from original to effective classifiers within a class is not necessarily bijective. In particular, because many h can be mapped to the same h ∆ , the VC dimension of the induced model class can shrink even from ∞ down to 0, causing model classes that were not learnable in the standard setting to become learnable in the strategic one. That said, the VC dimension of the induced class can generally either grow or shrink. We highlight some conditions that guarantee either of these outcomes, and, in particular, provide an upper bound on the strategic complexity increase for the expressive family of piecewise-linear model classes (e.g., ReLU networks). Our analysis yields several practical takeaways for learning, model selection, and future method development (see Appendix C.10). We conclude with experiments that shed light on how strategic behavior may alter the effective complexity and maximum accuracy of the learned model class in practice.

## 2 Related work

Our work focuses on the common setting of supervised binary classification, where non-linear models are highly prevalent in general. This follows the original formulation of strategic classification [3, 16], which has since been extended also to online settings [1, 5, 8] and regression [2, 31]. As noted, linear models have thus far been at the forefront of the field, with many works relying (explicitly or implicitly) on their properties or induced structure [5, 10, 18, 25, 29]. Even for works with results that are agnostic to model class selection, linear classifiers remain the focus of special-case analyses, examples, and empirical evaluation [20, 26, 37], and their generality does not shed light on the impact of non-linearity. Other works offer only tangential results, such as welfare analysis in the non-linear setting [35], but do not paint a picture of the general learning problem. Our low-level analysis makes use of ideas from the field of offset curves from computational geometry [12, 21]. Whereas this field is mostly concerned with the approximation and analytical properties of individual offset curves, we are

Figure 1: (Left) Original non-linear classifier h . (Middle) Strategic response to original classifier. (Right) Resulting effective classifier h ∆ (different from h ) after accounting for strategic responses.

<!-- image -->

interested in the effects on class complexity and what this implies for learning. In terms of theory, both [33, 36] apply PAC analysis to strategic classification, the latter providing VC bounds for the linear class. Most relevant to us is [6], who study the general learnability of strategic learning under different feedback models. While not restricted to particular classes, their analysis considers graph costs, where users can move only to a finite set of possible points, and their results depend crucially on the cardinality of this set. Our work tackles the more pervasive setting of continuous cost functions, which imply continuous modifications. The above works focus primarily on statistical hardness; while this is also one of our aims, we are interested in establishing a broader understanding of the non-linear regime.

## 3 Preliminaries

We work in the original supervised setting of strategic classification [3, 16]. Let x ∈ X = R d denote user features, y ∈ Y = { 0 , 1 } their corresponding label, and assume pairs ( x, y ) are drawn iid from some unknown joint distribution D . Given a training set T = { ( x i , y i ) } i m =1 ∼ D m , learning aims for the common goal of finding a classifier h ∈ H that maximizes the expected predictive accuracy, where H is some chosen model class. The unique aspect of strategic classification is that at test time, users modify their features to maximize gains in response to the learned h via the best-response mapping

<!-- formula-not-decoded -->

where α determines their utility from obtaining a positive prediction ˆ y = h ( x ) , and c is a function governing modification costs. We assume w.l.o.g. that the user gains no utility from a negative prediction. Like most works, we focus mainly on the ℓ 2 cost, i.e., c ( x, x ′ ) = ∥ x -x ′ ∥ 2 , but discuss other costs in a subset of our results (see Appendix C.1 for discussion on how our results generalize to other cost functions). Because no rational user will pay more than α to obtain a positive prediction, x h will always be contained in the ball of radius α around x , denoted B α ( x ) = { x ′ : c ( x, x ′ ) ≤ α } . It will also be useful to define the corresponding sphere, S α ( x ) = { x ′ : c ( x, x ′ ) = α } .

Given the above, the learning objective is to maximize the expected strategic accuracy , defined as:

<!-- formula-not-decoded -->

Levels of analysis. We assume w.l.o.g. that classifiers are of the form h ( x ) = 1 { f ( x ) ≥ 0 } for some scalar score function f , and that the decision boundary is then the surface of points for which f ( x ) = 0 . Our analysis takes a 'bottom-up' approach in which we first consider changes to individual points on the boundary, then ask how such local changes aggregate to affect a given classifier h , and finally, how these changes impact the entire model class H . It will therefore be useful to consider three mappings between original and corresponding strategic objects. For points, the mapping is z ↦→ ∇ h ( z ) = { x : ∆ h ( x ) = z ∧ c ( x, z ) = α } , i.e., the set of points x that, in response to h , move to z at maximal cost. For classifiers, we define the effective classifier as h ∆ ( x ) := h (∆ h ( x )) (see Fig. 1), and consider the mapping h ↦→ h ∆ . Since both h and h ∆ are defined w.r.t. raw (unmodified) inputs x , this allows us to compare the original decision boundary of h to the 'strategic decision boundary' of its induced h ∆ . For model classes, we consider the mapping H ↦→ H ∆ := { h ∆ : h ∈ H } , and refer to H ∆ as the effective model class .

The linear case. As noted, most works in strategic classification consider H to be the class of linear classifiers, h ( x ) = 1 { w ⊤ x ≥ b } . A known result is that ∆ h ( x ) can be expressed in closed form as a

Figure 2: Types of point mappings. From left to right: one-to-one (case #1), direct wipeout (case #2), indirect wipeout (case #2), expansion (case #3), and collision (case #4). Black lines show original h , dashed gray lines indicate h ∆ , gold arrows show z ↦→∇ h ( z ) , and dotted circles show S α ( x )

<!-- image -->

conditional projection operator: x moves to x proj = x -w ⊤ x -b ∥ w ∥ 2 2 w if h ( x ) = 0 and c ( x, x proj ) ≤ α , and otherwise remains at x [9]. Note that all points that move do so in the same direction, i.e., orthogonally to w . Furthermore, each x on the decision boundary of h is mapped to a single, unique point z on the decision boundary of h ∆ at an orthogonal distance (or cost) of exactly α . Considering this for all boundary points, we get that h ∆ simply shifts the decision boundary by α , which gives h ∆ ( x ) = 1 { w ⊤ x ≥ b -α } . Thus, H ∆ remains the set of all linear classifiers, and so H ∆ = H .

The non-linear case: first steps. Informally stated, the main difference in the non-linear case is that points no longer move in the same direction (see Fig. 1). Note that while points still move to the 'closest' point on the decision boundary, the projection operator is no longer nicely defined. As a result, the shape of the decision boundary of h may not be preserved when transitioning to h ∆ . Moreover, as we will show, some points on the decision boundary of h will 'disappear' and have no effect on the induced h ∆ , while other points may be expanded to affect a continuum of points on h ∆ . A final observation is that directionality is important in the strategic setting: because points always move towards the positive region, the shape of h ∆ for some h may be very different than that of the same classifier but with "inverted" predictions, h ′ ( x ) = 1 -h ( x ) . This lack of symmetry means that we cannot take for granted that the induced H ∆ will include both h ∆ and 1 -h ∆ , even if H does.

## 4 Underlying Mechanisms

Due to the added complexity of the non-linear setting, we first present some basic results and observations related to the underlying mechanisms of the non-linear setting in order to build the necessary intuition for our higher-level results. We begin by examining the point mapping z ↦→∇ h ( z ) and then turn to investigate how strategic behavior affects the classifier mapping h ↦→ h ∆ .

## 4.1 Point mapping

To gain an understanding of what determines the shape of h ∆ of a general h , we begin by considering a single point z on the decision boundary of h and examining how (and if) it maps to corresponding point(s) on the decision boundary of h ∆ through the point mapping z ↦→∇ h ( z ) . We identify and discuss four general cases, illustrated in Fig. 2:

Case #1: One-to-one. For linear h , the point mapping is always one-to-one. For non-linear h , the mapping remains one-to-one as long as h is 'well-behaved' around z . We consider this the 'typical' case and note that each z is mapped to its α -offset point x = z -α ˆ n z , where ˆ n z is the unit normal at z w.r.t. h (see Appendix B.1). A necessary condition for one-to-one mapping is that h is smooth at z , but smoothness is not sufficient, as the following cases depict.

Case #2: Wipeout. Even if h is smooth at z , in some cases the point mapping will be null: z ↦→ ∅ . We refer to this phenomenon as wipeout , and identify two types. Direct wipeout occurs when the signed curvature κ at z is negative and of sufficiently large magnitude; this is defined formally in Sec. 4.2. Intuitively, this results from strategic behavior obfuscating any rapid changes in the decision boundary of h . Indirect wipeout occurs when the set of points ∇ h ( z ) is contained in B α ( x ′ ) for some other x ′ on the decision boundary. For both types, the result is that B α ( z ) is entirely contained in the positive region of h ∆ and such z have no effect on h ∆ (see Appendix B.2).

Case #3: Expansion. When the decision boundary of h at z is not smooth (i.e., a kink or corner), z can be mapped to a continuum of points on the effective boundary of h ∆ . This, however, occurs only when the decision boundary is locally convex towards the positive region. The set of mapped

points ∇ h ( z ) are those on S α ( z ) which do not intersect with any other B α ( x ′ ) for any other x ′ on the boundary of h . Thus, the shape of the induced 'artifact' is partial to a ball. Partial expansion may occur if some of the expansion artifact is discarded through indirect wipeout (Appendix C.2).

Case #4: Collision. A final case is when a set of points { z 1 ...z n } on the boundary of h are all mapped to the same single point x on h ∆ , i.e., ∇ h ( z ) = ∇ h ( z ′ ) = x for all z, z ′ in the set. Typically, this happens at the transitions between well-behaved decision boundary segments and wiped-out regions. Collision often results in non-smooth kinks on h ∆ , even if all source points were smooth on h .

## 4.2 Curvature

Building up to the level of classifiers, it is first useful to examine what happens to boundary points z in terms of how h behaves around them. We make use of the notion of signed curvature , denoted κ , which quantifies how much the decision boundary deviates from being linear at a certain point x . For d = 2 , curvature is formally defined as κ = 1 /r , where r is the radius of the circle that best approximates the curve at x . For d &gt; 2 , we use κ ∗ = sup( K ) , where K includes all directional curvatures at x (see Appendix A.1). The sign of κ measures whether the decision boundary bends (i.e., is convex) towards the positive region ( κ &gt; 0 ), the negative region ( κ &lt; 0 ), or is locally linear ( κ = 0 ).

Let z be a typical point on the decision boundary of h which satisfies the one-to-one mapping detailed in Sec. 4.1. We borrow a classic result from the field of offset curves [12] to calculate the effective signed curvature of h ∆ at x , denoted κ ∆ , as a function of the curvature κ of h at z .

Proposition 1. Let z be a point on the decision boundary of h with signed curvature κ ≥ -1 /α . The effective curvature of the corresponding x on the boundary of h ∆ is given by:

<!-- formula-not-decoded -->

When d &gt; 2 , each κ ∈ K is mapped as in Eq. 3 to form K ∆ = { κ/ (1 + ακ ) : κ ∈ K } , and κ ∗ ∆ follows. See Appendix B.3 for proof and an illustration of how this impacts the shape of h ∆ . Prop. 1 entails several interesting properties. First, strategic behavior preserves the direction of curvature, i.e., κ · κ ∆ ≥ 0 always. In particular, κ = 0 entails κ ∆ = 0 , which explains why linear functions remain linear. Second, the effective curvature κ ∆ is monotonically increasing in κ , concave, and sub-linear. This implies that for κ &gt; 0 the effective positive curvature decreases , while for κ &lt; 0 the effective negative curvature increases . Thus, strategic behavior acts as a directional smoothing operator, i.e., the boundaries' rate of change is expected to decrease but only in the positive direction.

Finally, in the positive direction, κ ∆ is bounded from above, approaching 1 /α as κ →∞ . This gives:

Observation 1. No effective classifier h ∆ can have positive curvature κ ∆ greater than 1 /α .

When d &gt; 2 , Obs. 1 holds for each curvature κ ∈ K and can be summarized as κ ∗ ∆ ≤ 1 /α . Though Prop. 1 only implies Obs. 1 at points on h ∆ that are typical, we will show in Sec. 4.4 that it is also true at any point on h ∆ . The above holds irrespective of the shape and curvature of the original h . In particular, non-smooth points on h (i.e., with κ →∞ ), such as an edge or vertex at the intersection of halfspaces, become smooth on h ∆ . Conversely, in the negative direction, κ ∆ approaches -∞ as κ → 1 /α . Thus, negative curvature can grow arbitrarily large, and at κ = -1 /α maps to a non-smooth "kink" in h ∆ . κ ∆ is no longer defined when κ crosses the asymptote at -1 /α because such points are wiped out and have no corresponding x on the boundary of h ∆ . This is the exact case of indirect wipeout in Sec. 4.1 (see Appendix B.2 for proof and Appendix C.10 for practical use in learning).

Observation 2. Any point z on the decision boundary of h with curvature κ &lt; -1 /α has no influence on h ∆ . In higher dimensions, this is true for any point x with inf( K ) &lt; -1 /α .

## 4.3 Containment

Because the decision boundary is a level set of some (non-linear) score function f , contained regions are not only possible, but realistic. Consider a given classifier h where the decision boundary forms some bounded connected negative region C ⊂ R d where h ( x ) = 0 ∀ x ∈ C . Strategic behavior dictates that negative areas shrink, and so only a subset of C will still be classified as negative in h ∆ . If C is sufficiently small, then it will disappear completely leaving h ∆ ( x ) = 1 ∀ x ∈ C .

Observation 3. Any negative region C fully contained in some B α ( x ′ ) will become positive in h ∆ .

Figure 3: Types of impossible effective classifiers. From left to right: small positive region, narrow positive strip, large positive curvature, piecewise (hyper)linear convex towards the positive region.

<!-- image -->

The fact that positive areas grow and negative areas shrink also implies that there is a minimum positive region size that can exist in any h ∆ . This idea will be formalized and generalized later in Sec. 4.4. One implication of Obs. 3 is that when negative regions are wiped out, two or more positive regions may merge, leading to a less expressive decision boundary. On the other hand, the strategic setting can also split a negative region, creating a more expressive decision boundary (see Appendix C.3).

Observation 4. Strategic behavior can merge positive regions and split negative regions thereby either increasing or decreasing the number of connected regions (either positive or negative).

## 4.4 Impossible Effective Classifiers

Based on the above, we derive two general statements about effective classifiers that cannot exist. Each statement considers some candidate classifier g , and shows that if g satisfies a certain property, then there is no classifier h for which g is its induced effective classifier, i.e., g = h ∆ for any h . We first require some additional notation. Given h , define the set of negatively classified points as X 0 ( h ) = { x : h ( x ) = 0 } . For a set A ⊆ X , we say a point x is reachable from A if ∃ x ′ ∈ A : c ( x, x ′ ) ≤ α . Proposition 2. Let g be a candidate classifier. If there exists a point x on the decision boundary of g such that all points x ′ ∈ S α ( x ) are reachable from some point in X 0 ( g ) , then there is no h such that g = h ∆ .

̸

The intuition behind Prop. 2 is that no h can simultaneously strategically classify x as positive while all y ∈ X 0 ( g ) as negative. Full proof in Appendix B.4. As stated in Prop. 3, when g is smooth at x , Prop. 2 can be simplified to require that only a single point be checked instead of all of S α ( x ) .

Proposition 3. Let g be a candidate classifier. If there exists a point x on the decision boundary of g where (i) g is smooth, and (ii) the offset point ˆ x = x + α ˆ n x is reachable by some other x ′ ∈ X 0 ( g ) , then there is no h such that g = h ∆ .

The intuition behind Prop. 3 is the same as for Prop. 2, except that when g is smooth at x , the only point in S α ( x ) that may not be reachable by some x ′ ∈ X 0 ( g ) is the α -offset (see Appendix B.5).

Examples. The implication of Props. 2 and 3 is that there exist decision boundaries that are generally realizable, but are no longer realizable as effective classifiers in the strategic setting. Note that because it only takes a single decision boundary point to render an entire effective classifier infeasible, each of these examples are broad categories and pose meaningful restrictions on the set of possible effective classifiers. As illustrated in Fig. 3, we highlight four types of impossible effective decision boundaries:

1. Classifiers with a small positive region that can be enclosed by a ball of radius α .
2. Classifiers with a narrow positive region in which points are α -close to the decision boundary.
3. Classifiers with large positive signed curvature anywhere on the decision boundary.
4. Classifiers with any locally convex piecewise linear or hyperlinear segments.

All proofs are in Appendix C.4, and either build on Obs. 1-4, or show the conditions of Props. 2 or 3 hold. From these four examples, we note the following observation.

Observation 5. The mapping h ↦→ h ∆ is not bijective, with many candidate h ∆ having no corresponding h and many h mapped to the same h ∆ .

In Appendix C.5 we show that there are actually infinite h that map to nearly all possible h ∆ . Furthermore, Obs. 5 will be important for proving Thm. 1 and Cor. 1.

<!-- image -->

Figure 4: (Left) Example of increasing VC. (Right) Example of a dataset with limited strategic accuracy. Any h ∆ which correctly classifies the gold-rimmed point must err on a negative point.

<!-- image -->

## 5 Class-level analysis

On the classifier level, the strategic setting is clearly more restrictive than the regular setting. Indeed, in Sec. 5.2 we extend this idea into the model class level by analyzing the impacts on universality and optimal accuracy. However, as we show in Sec. 5.1, the model class mapping is more nuanced than the classifier mapping since complexity can actually increase as a result of strategic behavior.

## 5.1 Class complexity

Recall that for linear classes, H ∆ remains linear, and so H ↦→ H ∆ = H . Although this equivalence is not unique to the class of linear classifiers (see examples in Appendix C.6), typically, we can expect strategic behavior to alter the complexity of the induced class. Here, we use VC analysis to shed light on this effect by examining the possible relations between VC( H ) and VC( H ∆ ) . 1

Complexity reduction. Since our results so far point out that effective classifiers are inherently constrained, a plausible guess is that effective complexity should decrease. Indeed, there are some natural classes where the VC cannot increase. For example:

Proposition 4. Let Q k s be the class of negative-inside k -vertex polytopes bounded by a radius of s , then VC( Q k s ) ≥ VC( Q k s, ∆ ) . Moreover, if s = ∞ , then VC( Q k s ) = VC( Q k s, ∆ ) as implied by Thm. 2.

The proof for Prop. 4 can be found in Appendix B.7. Curiously, in some extreme cases, the VC reduction spans the maximum extent.

Theorem 1. There exist several model classes H where VC( H ) = ∞ , but VC( H ∆ ) = 0 .

The proof for Thm. 1 can be found in Appendix B.8. See Appendix C.7 for other examples where VC strictly decreases but does not diminish completely. Thm. 1 carries concrete implications for strategic learning:

Corollary 1. Some model classes that are not learnable become learnable in strategic environments.

Complexity increase. Though a lower effective VC is indeed a possibility, a more plausible outcome is for it to stay the same or increase. As an intuition for how the VC might increase, consider the simple example in Fig. 4 (left). Here, H includes four non-overlapping classifiers (each color is a separate classifier) with VC( H ) = 1 . Yet, with strategic behavior, the effective classifiers (dashed curves) overlap, increasing the shattering capacity to 2. This case can be generalized to give V C ( H ∆ ) = θ ( d · V C ( H )) (see Appendix C.8). We next identify a simple generic sufficient condition for the effective complexity of any learnable class to be non-decreasing.

Theorem 2. If H is closed under input scaling and VC( H ) &lt; ∞ , then VC( H ) ≤ VC( H ∆ ) .

The proof for Thm. 2 can be found in Appendix. B.9. Thm. 2 applies to many common classes, including polynomials, piecewise linear functions, and most neural networks. See Appx. C.10 for implications on model selection.

Because Thm. 2 suggests that the effective VC can increase, but does not state by how much, one concern would be that this increase can be unbounded, which would deem the effective class unlearnable. Our next result shows that this is not the case even for the common and highly expressive class of piecewise-linear classifiers (with a bounded number of segments), which includes neural

1 This aligns with the notion of strategic VC used in [22, 33], i.e., SVC( H ) ≡ VC( H ∆ ) .

networks with ReLU activations (of fixed size) as a special case, and is often used in studies on neural network approximation [e.g., 19, 28].

Theorem 3. Let H m,k be a class of piecewise-linear classifiers with at most m segments and k intersections of linear segments. Then VC( H m,k ∆ ) = O ( dmlog ( m ) + ν p k log ( k ) ) , where ν p is the VC of the class of ℓ p norm balls, and is Θ( d ) for p = 1 , 2 , ∞ .

The proof for Thm. 3 is in Appendix B.10. We also prove that VC( H m,k ) = O ( dmlog ( m ) ) , suggesting that the effective VC dimension can increase, but only to a limited extent. The main implication of Thm. 3 is therefore that under these conditions, strategic behavior maintains learnability: for any H that can be expressed as (or approximated by) piecewise-linear functions, if H (or its approximation) is learnable, then H ∆ is also learnable. We conjecture that this relation remains true for general H ; see Appendix A.3 for discussion on the challenges of the general case and connections to [6]. For the special case of positive-inside polytopes - a subset of the piecewise-linear class the effective VC dimension even maintains the same order of magnitude.

Theorem 4. Consider either the ℓ 1 , ℓ 2 , or ℓ ∞ costs and let H ⊆ P k , where P k is the set of all k -vertex polytopes in R d and has VC( P k ) = O ( d 2 k log k )[ 23 ] . Then VC( H ∆ ) = O ( d 2 k log k ) .

The proof for Thm. 4 is in Appendix. B.11. Note that for the ℓ 1 and ℓ ∞ costs, the effective class remains within the polytope family, but potentially with additional vertices (see Appendix B.6). For the ℓ 2 cost, the bound still holds despite H ∆ no longer including only polytopes (since corners in h can become "rounded" in h ∆ ).

## 5.2 Universality and approximation

Our results in Sec. 4.4 show four 'types' of decision boundaries that cannot be attributed to any effective classifier, though others may exist. This implies that several basic classifiers are not realizable in the strategic setting and cannot be approximated well. This drives our principal conclusion:

Corollary 2. Any universal approximator class H is no longer universal under strategic behavior.

In other words, there exists a classifier g for which no effective class H ∆ can include g as a member, even for H that are universal approximators in the non-strategic setting. Common examples of universal approximator classes include polynomial thresholds [32], RBF kernel machines [27], many classes of neural networks [e.g., 17], and gradient boosting machines [13]. In the strategic setting, these classes are no longer all-encompassing like in the standard setting.

Approximation gaps. Cor. 2 implies that even if the data is in itself realizable, and even without any limitations on the hypothesis class, the learner may already start off with a bounded maximum training accuracy simply by learning in the strategic setting. Fig. 4 (right) depicts one such example dataset where the maximum attainable accuracy drops from 1 to 0.96. This is because strategic behavior makes it impossible to generally approximate intricate functions, e.g., by using fast-changing curvature (via increased depth) or many piecewise-linear segments (with ReLUs). Another result is that the common practice of interpolating the data using highly expressive over-parametrized models (e.g., as in deep learning) is no longer a viable approach. On the upside, it may be possible for the strategic responses to prevent unintentional overfitting, allowing for better generalization.

In extreme cases, the gap between the maximum standard accuracy and the maximum strategic accuracy can be quite large. Our next result demonstrates that there are distributions in which, despite a maximum standard accuracy of 1, the maximum strategic accuracy falls to the majority class rate.

Proposition 5. For all d , there exist (non-degenerate) distributions on R d that are realizable in the standard setting, but whose maximum strategic accuracy is the majority class rate, max y p ( y ) .

In one dimension, the distribution can be constructed by placing a negative point δ to the left and right of each positive point. In higher dimensions, negative points can be arranged in an R d simplex around each positive point. The full proof is in Appendix B.12. Prop. 6 extends this idea to show that there is a distribution whose maximum strategic accuracy approaches 1 / 2 as α increases. The general proof is given in Appendix B.13.

Proposition 6. There exists a distribution that is realizable in the standard setting, but whose accuracy under any h ∆ is at most 0 . 5 + 1 ⌊ 2 α +1 ⌋ .

Figure 5: (Left) Expressivity. For random polynomial classifiers of degree k , results show the smallest degree k ′ that captures the effective decision boundary. (Center) An instance showing positive curvature decreasing (A), negative curvature increasing (B), and wipeout (C). (Right) Approximation. As data becomes more entangled (low separation), strategic approximation degrades.

<!-- image -->

In practice, we expect the accuracy gap to depend on the distribution. We explore this gap experimentally in Sec. 6. An interesting observation is that in certain non-realizable settings, the maximum strategic accuracy can exceed the maximum standard accuracy. (see Appendix C.9).

## 6 Experiments

To complement our theoretical results, we present two experiments that empirically demonstrate the effects of strategic behavior on non-linear classifiers. The code for both experiments is available at https://github.com/BML-Technion/scnonlin .

## 6.1 Expressivity

Following our results from Sec. 5.1 on the change in hypothesis class expressivity, we experimentally test whether the h ∆ of a given random h belongs to a more or less expressive class than h itself. In particular, we focus on degreek polynomial classifiers and compare the degree k of h to the degree k ∆ of the polynomial approximation of h ∆ . 2 Because user features are often naturally bounded, we assume that x ∈ X = R 2 and that x is bounded by | x | ∞ ≤ 100 . See Appendix D.1 for full details.

Fig. 5 (left) shows results for varying k ∈ [1 , 10] and for α ∈ [2 , 40] . When α is small, we find that k ∆ ≈ k , meaning that strategic behavior has little impact on expressivity. However, as α increases, k ∆ becomes larger than k , suggesting that h ∆ is more complex than h . This is due to the fact that as α increases, point mapping collisions (see Sec. 4.1) become more likely, causing non-smooth cusps which are more complex than basic polynomials - to form in the decision boundary (Fig. 5 (center)). When α is quite large, we see that k ∆ is still larger for low k , but drops considerably for higher k . This can be attributed to the fact that tightly-embedded higher-dimension polynomials and increased strategic reach from large αs are both causes of indirect wipeout (see Sec. 4.1) because they increase the reachability of ∇ h ( z ) by other decision boundary points. As such, much of the original decision boundary is wiped out, leaving behind a lower complexity effective decision boundary.

## 6.2 Approximation

As seen in Sec. 5.2, the lack of universality in the strategic setting can impose a limitation on the maximal attainable strategic accuracy. To demonstrate the practical implications of this effect, we upper bound the maximum strategic accuracy of any H ∆ on a set of synthetic data by experimentally calculating the maximum strategic accuracy of the unrestricted effective hypothesis class H ∗ ∆ . We compare the results to two benchmarks: (i) the standard accuracy of the regular unrestricted hypothesis class H ∗ (which is always 1), and (ii) the strategic (and standard) accuracy of the linear effective hypothesis class H lin ∆ = H lin . The first benchmark measures the extent to which the strategic setting

2 Though the h ∆ of a polynomial h is not necessarily a polynomial, we can compare the expressivity of h and h ∆ by finding the lowest degreek ∆ polynomial g that well-approximates h ∆ .

hinders the learner, while the second measures the extent to which the learner can benefit from non-linearity, even in the strategic world. We generate data by sampling points from class-conditional Gaussians x ∼ N ( yµ, 1) , and aim to find an h that obtains an optimal fit. The parameter µ serves to show how H ∆ compares to our two baselines under data that ranges from well-separated (large µ ) to more "interleaved" and harder to classify by restricted classes (small µ ). Details in Appendix D.2.

Fig. 5 (right) shows results for increasing class separation ( µ ). As µ decreases and the classification problem becomes more difficult, the maximum strategic accuracy of H ∗ ∆ diverges from the upper H ∗ baseline, with performance deteriorating as strategic behavior intensifies (i.e., larger α ). Because any actual H ∆ can only have worse performance than the ideal H ∗ ∆ , this indicates that strategic behavior can become a significant burden in more difficult tasks, such as classification in a non linearlyseparable setting. Nonetheless, when µ is small, H ∗ ∆ significantly outperforms the lower H lin ∆ baseline, signaling that non-linearities still improve accuracy in the strategic setting despite any strategic effects.

## 7 Discussion

This work sets out to explore the interplay between non-linear classifiers and strategic user behavior. Our analysis demonstrates that non-linearity induces behavior that is both qualitatively different than the linear case and also non-apparent by simply studying it. Simply put, the strategic setting is fundamentally limited, though has a few potential advantages. Our results show how such behavior can impact classifiers by morphing the decision boundary and model classes by increasing or decreasing complexity. Although prior work has made clear the importance of accounting for strategic behavior in the learning objective, our work suggests that there is a need for additional broader considerations throughout the entire learning pipeline from the choices we make initially (such as which model class to use) to setting our final expectations (such as the accuracy levels we may aspire to achieve). We detail practical takeaways for learning and model class selection in Appendix C.10. This motivates future theoretical questions, as well as complementary work on practical aspects such as optimization (i.e., how to solve the learning problem), modeling (i.e., how to capture true human responses), and evaluation (i.e., on real humans and in the wild).

## Acknowledgments

The authors are grateful to Shay Moran and Nadav Dym for thoughtful discussions and suggestions. This work is supported by the Israel Science Foundation grant no. 278/22.

## References

- [1] Saba Ahmadi, Hedyeh Beyhaghi, Avrim Blum, and Keziah Naggita. The strategic perceptron. In Proceedings of the 22nd ACM Conference on Economics and Computation , pages 6-25, 2021.
- [2] Yahav Bechavod, Chara Podimata, Steven Wu, and Juba Ziani. Information discrepancy in strategic learning. In International Conference on Machine Learning , pages 1691-1715. PMLR, 2022.
- [3] Michael Brückner, Christian Kanzow, and Tobias Scheffer. Static prediction games for adversarial learning problems. The Journal of Machine Learning Research , 13(1):2617-2654, 2012.
- [4] Michael Brückner and Tobias Scheffer. Nash equilibria of static prediction games. In Advances in neural information processing systems , pages 171-179, 2009.
- [5] Yiling Chen, Yang Liu, and Chara Podimata. Learning strategy-aware linear classifiers. Advances in Neural Information Processing Systems , 33:15265-15276, 2020.
- [6] Lee Cohen, Yishay Mansour, Shay Moran, and Han Shao. Learnability gaps of strategic classification. 247:1223-1259, 2024. Publisher Copyright: © 2024 L. Cohen, Y. Mansour, S. Moran and H. Shao.; 37th Annual Conference on Learning Theory, COLT 2024 ; Conference date: 30-06-2024 Through 03-07-2024.

- [7] Daniel Cullina, Arjun Nitin Bhagoji, and Prateek Mittal. Pac-learning in the presence of evasion adversaries. In Proceedings of the 32nd International Conference on Neural Information Processing Systems , NIPS'18, page 228-239, Red Hook, NY, USA, 2018. Curran Associates Inc.
- [8] Jinshuo Dong, Aaron Roth, Zachary Schutzman, Bo Waggoner, and Zhiwei Steven Wu. Strategic classification from revealed preferences. In Proceedings of the 2018 ACM Conference on Economics and Computation , pages 55-70, 2018.
- [9] Itay Eilat, Ben Finkelshtein, Chaim Baskin, and Nir Rosenfeld. Strategic classification with graph neural networks. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023.
- [10] Itay Eilat and Nir Rosenfeld. Performative recommendation: diversifying content via strategic incentives. In International Conference on Machine Learning , pages 9082-9103. PMLR, 2023.
- [11] Andrew Estornell, Sanmay Das, Yang Liu, and Yevgeniy Vorobeychik. Group-fair classification with strategic agents. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency , FAccT '23, page 389-399, New York, NY, USA, 2023. Association for Computing Machinery.
- [12] R. T. Farouki and C. A. Neff. Analytic properties of plane offset curves. Comput. Aided Geom. Des. , 7(1-4):83-99, June 1990.
- [13] Jerome H Friedman. Greedy function approximation: a gradient boosting machine. Annals of statistics , pages 1189-1232, 2001.
- [14] Ganesh Ghalme, Vineet Nair, Itay Eilat, Inbal Talgam-Cohen, and Nir Rosenfeld. Strategic classification in the dark. In International Conference on Machine Learning , pages 3672-3681. PMLR, 2021.
- [15] Alirio Gómez Gómez and Pedro L. Kaufmann. On the Vapnik-Chervonenkis dimension of products of intervals in R d . arXiv e-prints , page arXiv:2104.07136, April 2021.
- [16] Moritz Hardt, Nimrod Megiddo, Christos Papadimitriou, and Mary Wootters. Strategic classification. In Proceedings of the 2016 ACM conference on innovations in theoretical computer science , pages 111-122, 2016.
- [17] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks , 2(5):359-366, 1989.
- [18] Safwan Hossain, Evi Micha, Yiling Chen, and Ariel Procaccia. Strategic classification with externalities. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2023 , 2025.
- [19] Changcun Huang. Relu networks are universal approximators via piecewise linear or constant functions. Neural Computation , 32(11):2249-2278, 2020.
- [20] Meena Jagadeesan, Celestine Mendler-Dünner, and Moritz Hardt. Alternative microfoundations for strategic classification. In International Conference on Machine Learning , pages 4687-4697. PMLR, 2021.
- [21] Soo Won Kim, Ryeong Lee, and Young Joon Ahn. A new method approximating offset curve by bézier curve using parallel derivative curves. Computational and Applied Mathematics , 37(2):2053 - 2064, 2018. Cited by: 3.
- [22] Anilesh K Krishnaswamy, Haoming Li, David Rein, Hanrui Zhang, and Vincent Conitzer. Classification with strategically withheld data. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 5514-5522, 2021.
- [23] Andrey Kupavskii. The vc-dimension of k-vertex d-polytopes. Combinatorica , 40(6):869-874, December 2020.
- [24] Tosca Lechner and Ruth Urner. Learning losses for strategic classification. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI , pages 7337-7344. AAAI Press, 2022.

- [25] Sagi Levanon and Nir Rosenfeld. Strategic classification made practical. In International Conference on Machine Learning , pages 6243-6253. PMLR, 2021.
- [26] Sagi Levanon and Nir Rosenfeld. Generalized strategic classification and the case of aligned incentives. In Proceedings of the 39th International Conference on Machine Learning (ICML) , 2022.
- [27] J. Park and I. W. Sandberg. Universal approximation using radial-basis-function networks. Neural Computation , 3(2):246-257, 1991.
- [28] Philipp Petersen and Felix Voigtlaender. Optimal approximation of piecewise smooth functions using deep relu neural networks. Neural Networks , 108:296-330, 2018.
- [29] Elan Rosenfeld and Nir Rosenfeld. One-shot strategic classification under unknown costs. In Forty-first International Conference on Machine Learning , 2024.
- [30] Han Shao, Avrim Blum, and Omar Montasser. Strategic classification under unknown personalized manipulation. In Advances in Neural Information Processing Systems , volume 36, 2023.
- [31] Yonadav Shavit, Benjamin Edelman, and Brian Axelrod. Causal strategic linear regression. In International Conference on Machine Learning , pages 8676-8686. PMLR, 2020.
- [32] Marshall H Stone. The generalized weierstrass approximation theorem. Mathematics Magazine , 21(5):237-254, 1948.
- [33] Ravi Sundaram, Anil Vullikanti, Haifeng Xu, and Fan Yao. PAC-learning for strategic classification. In International Conference on Machine Learning , pages 9978-9988. PMLR, 2021.
- [34] Aad Vaart and Jon Wellner. A note on bounds for vc dimensions. Institute of Mathematical Statistics collections , 5:103-107, 01 2009.
- [35] Tian Xie and Xueru Zhang. Non-linear welfare-aware strategic learning. Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society , 7:1660-1671, 10 2024.
- [36] Hanrui Zhang and Vincent Conitzer. Incentive-aware PAC learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 5797-5804, 2021.
- [37] Tijana Zrnic, Eric Mazumdar, Shankar Sastry, and Michael Jordan. Who leads and who follows in strategic classification? Advances in Neural Information Processing Systems , 34:1525715269, 2021.

## A Background

## A.1 Signed Curvature

Our analysis makes use of the notion of signed curvature -asigned version of the normal curvature -toquantify shape alteration. Because there is an inherent notion of direction due to the binary nature of class labels, we can define curvature as being positive if the curvature normal vector points in the direction of the positive region and negative if it points in the direction of the negative region. Though in R 2 this defines a single value, we must instead define a curvature value for each 'direction' in higher dimensions. More formally, for each tangent vector ⃗ t in the tangent hyperplane T x at x , we define the curvature value κ ⃗ t as the signed curvature at x on the curve obtained by intersecting the decision boundary with the plane L containing ⃗ t and the normal vector ˆ n x . Defining K = { κ ⃗ t : ⃗ t ∈ T } as the set of all such curvatures, our analysis will focus on κ ∗ = sup( K ) in R 3 and above.

## A.2 Bounded/Unbounded Input Domain

At a high level, our results in Sec. 5 do not require an unbounded input space, but rather some input space that is larger than X . In particular, if we assume that raw inputs, x , are bounded by | x | ≤ B , then for the problem to generally be well-defined, we must allow this space to expand at the minimum to B + α to accommodate for strategic input modifications. More generally, we must ensure that | ∆ h ( x ) | ≤ B ( α ) holds for all x and any h , for some expansion B ( α ) ≥ B , where the exact form of B ( α ) depends both on the cost function and on H .

Other than Thm. 2, our results simply require the minimal expansion be satisfied and are therefore not dependent on an unbounded input domain. Likewise, Thm. 2 does not require an unbounded input domain since for any given learnable H , there exists some finite B ( α ) for which the result holds. However, since the statement is generic (i.e., applies to all learnable H that are closed to scaling), we do not have a closed-form expression for B ( α ) that applies simultaneously to all such H .

## A.3 General Upper Bound on VC Increase

In Thm. 3, we provide an upper bound on the strategic VC dimension of classes of piecewise-linear classifiers, though note that a general upper bound remains an open question. A key difficulty in finding a general upper bound in the strategic setting is that points will move differently when facing different h from the same H . Thus, reasoning about the shattering of an arbitrary set must account for how points can move under all possible h ∈ H , though not simultaneously, which proved to be highly challenging in the general, unstructured case. This is why we have included results that exploit common structures like the upper bound based on piecewise linearity (Thm. 3) and lower bound based on H that are closed to scaling (Thm. 2).

That said, we conjecture that Thm. 3 holds for general H , as proven by Cohen et al. [6] in the special case of finite manipulation graphs. In particular, they show that VC( H ∆ ) = ˜ Θ(VC( H ) · log k ) where k is the degree of the manipulation graph. Note that the bound is not transferable to our case since it is vacuous for k →∞ , and the proof technique does not apply to continuous graphs.

## B Proofs

Lemma 1. In R 2 , if the curvature κ at x is positive, then the radius r insc of the maximum inscribed circle at x is no bigger than the radius r osc = 1 κ of the osculating circle at x (the circle that best approximates the curve at x and has the same curvature). As a corollary, if curvature κ at x is negative, then the radius r esc of the maximum escribed circle at x is no bigger than the radius r osc = -1 κ of the osculating circle at x .

Proof. Note that both the osculating circle and the inscribed circle have their centers along the normal line to the curve at x (and on the same side of h ), and that the osculating circle may cross the curve at x . If it does, then any circle which has a larger radius and a center on the same side of h along the normal line to the curve will also cross the decision boundary at x , so the radius of the inscribed circle (which strictly does not cross the curve) is strictly smaller than the radius of the osculating circle. If the osculating circle does not cross the curve, then it must be the largest strictly tangent

circle at x , or else it would not be the best circular approximation to the curve at x . As such, the largest inscribed circle, which must be both tangent to the curve at x also not intersect it anywhere else, must have a radius that is no larger than r osc .

## B.1 One-to-One Mapping

Claim: In the one-to-one mapping, the point z on the decision boundary of h is mapped to the point x = z -α ˆ n z on the decision boundary of h ∆ .

Proof. Note that all points on the decision boundary of h ∆ must be at the maximum cost away from h 3 , so ∇ h ( z ) ⊂ S α ( z ) . Additionally, for each x ∈ ∇ h ( z ) , the ball B α ( x ) , which is the set of points that x can move to, must not intersect the decision boundary of h (or else it would not be at maximum cost away) and must furthermore be strictly tangent to the decision boundary of h at z . Given that the decision boundary of h is smooth at z , this means that the normal to the curve at z must run along the radius of B α ( x ) from z to x , implying that x = z ± α ˆ n z . Because of the inherent directionality of the setting, the normal vector at z is defined to point in the positive direction. This means that only the point x = z -α ˆ n z would seek to move to z to gain a positive classification, as x = z + α ˆ n z is either already labeled positive or can move to a closer point. Therefore, ∇ h ( z ) = { z -α ˆ n z } in the one-to-one case.

## B.2 Wipeout Mapping

Claim: Points on the decision boundary of h that are 'wiped out' do not affect the effective decision boundary of h ∆ .

Proof. In the case of indirect wipeout , this follows directly from the definition: if ∇ h ( z ) is contained in B α ( x ′ ) for some other x ′ on the decision boundary of h (or in the union of several B α ( x ′ ) ), then none of the points in ∇ h ( z ) are at a maximum cost away from h , and are therefore not on the decision boundary of h ∆ .

In the case of direct wipeout , the normal one-to-one mapping fails because the point x = z -α ˆ n z is no longer at the maximum cost away from h . Specifically, because the curvature κ &lt; -1 /α 4 , the ball B α ( x ) now intersects h , which means that x is less than α away from h and therefore not on the decision boundary of h ∆ . For R 2 , Lemma 1 proves that the radius of the maximum escribed circle at z is less than the radius of the osculating circle, so r esc ≤ r osc = -1 κ &lt; α . Therefore, there is no circle of radius α that is strictly tangent to h (on the negative side), so B α ( x ) must intersect h .

For higher dimensions (see Appendix A.1), there is no longer a single normal curvature. Instead, we define the set K of curvatures in each plane p ∈ P defined by the normal vector and some tangent vector at z . Note that B α ( x ) will intersect h if any of its projections onto each p ∈ P (a circle) intersects the curve obtained by intersecting the decision boundary with the plane p . Since each of these cases are in R 2 , B α ( x ) will intersect h if inf( K ) &lt; -1 /α .

## B.3 Proof of Proposition 1

Proposition 1: Let z be a point on the decision boundary of h with signed curvature κ ≥ -1 /α . The effective curvature of the corresponding x on the boundary of h ∆ is given by:

<!-- formula-not-decoded -->

Proof. While an equivalent form of Prop. 1 has already been proven in the field offset curves [12], we nonetheless reprove it here to offer some intuition behind it. In the normal, one-to-one mapping, because both the center of curvature and ∇ h ( z ) lie on the normal line to h at z (see Appendix B.1),

3 Otherwise, x would not be on the boundary of positive and negative classification by h ∆ because all of the points adjacent to it can also strategically move to achieve a positive classification.

4 or inf( K ) &lt; -1 /α for d &gt; 2 . See Sec. 4.2 and Appendix A.1 for curvature definitions

D

Figure 6: (Top) Graph of Eq. 3 with asymptotes represented as dashed lines. (Bottom) Depictions of curvature mappings for increasing κ .

<!-- image -->

strategic movement can be seen as increasing the signed radius of curvature, r osc , by α (where we will define the radius of curvature as negative if the curvature is negative at z ). In higher dimensions, this is equivalent to increasing each of the radii of curvature by α (see Appendix A.1). We therefore have:

<!-- formula-not-decoded -->

Fig. 6 shows the graph of Eq. 3 as well as examples of curvature mappings for various κ . Note that Eq. 3 has a horizontal asymptote at κ ∆ = 1 /α because the effective curvature cannot exceed 1/ α (see Sec. 4.4). Additionally, Eq. 3 has a vertical asymptote at κ = -1 /α because any point with lower curvature has no effect on h ∆ (see Appendix B.2). When κ = -1 /α , x is mapped to a non-smooth kink of infinite curvature.

## B.4 Proof of Proposition 2

Proposition 2: Let g be a candidate classifier. If there exists a point x on the decision boundary of g such that all points x ′ ∈ S α ( x ) are reachable from some point in X 0 ( g ) , then there is no h such that g = h ∆ .

Proof. In order for g to be the effective classifier of the regular classifier h , we must have:

1. For each point q that is positively classified by g , there must be a point q ′ ∈ B α ( q ) which is classified as positive by h . In other words, q must be able to acquire a positive label by staying in place or strategically moving within B α ( q ) .
2. For each point q that is negatively classified by g , there cannot be any points q ′ ∈ B α ( q ) which are classified as positive by h . In other words, q must not be able to acquire a positive label by staying in place or strategically moving within B α ( q ) .

Assume there exists a point x on the decision boundary of g such that all points x ′ ∈ S α ( x ) are reachable from X 0 ( g ) and that there exists h such that g = h ∆ . Since x is on the decision boundary

Figure 7: (Left) Each face of a negative-inside polytope is mapped to a new linear face, though the resulting h ∆ may be a polytope with fewer faces and vertices due to wipeout. (Right) Each face of a positive-inside polytope is also mapped to a new linear face, though each vertex will have to expand to fill the gap between these faces.

<!-- image -->

of g , there exists a point x + δ, δ → 0 that is negatively classified by g . For requirements 1 and 2 to hold, there must be a point x ′ ∈ S = { q : q ∈ B α ( x ) , q ̸∈ B α ( x + δ ) } that is positively classified by h . Since S ⊂ S α ( x ) , we can expand the claim to say that there must be a point x ′ ∈ S α ( x ) that is positively classified by h . However, if all points x ′ ∈ S α ( x ) are reachable by some point q that is negatively classified by g , then none of the points x ′ ∈ S α ( x ) can be positively classified by h (by requirement 2). Therefore, if there exists a point x on the decision boundary of g for which every point on the sphere S α ( x ) is reachable by some point that is negatively classified by g , then g cannot be a possible effective classifier.

## B.5 Proof of Proposition 3

Proposition 3: Let g be a candidate classifier. If there exists a point x on the decision boundary of g where (i) g is smooth, and (ii) the offset point ˆ x = x + α ˆ n x is reachable by some other x ′ ∈ X 0 ( g ) , then there is no h such that g = h ∆ .

Proof. By Prop. 2, for each point x on the decision boundary of g , there must be a point x ′ on S α ( x ) that is classified positively by h and also not reachable by any point negatively classified by g . As a result, the ball B α ( x ′ ) , which is the set of points that can move to x ′ , must not intersect the decision boundary and must furthermore be strictly tangent to the decision boundary at x . Given that the decision boundary is smooth at x , this means that the normal to the curve at x must run along the radius of B α ( x ′ ) from x to x ′ , implying that x ′ = x ± α ˆ n x . Because the strategic movement is in the direction of a positive classification, the only possible option is for x to move to x ′ = x + α ˆ n x . However, if x ′ = x + α ˆ n x is reachable by some point that is negatively classified by h ∆ , then the associated h would not have classified x ′ as positive, so x would not have been a positively classified point by h ∆ .

## B.6 Effective Class of Polytope Family

Each polytope is simply a combination of n linear faces. As such, when a polytope has a negative inside, each face will be mapped to a new linear face (with some wipeout of part or all of the face) leading to an h ∆ that is a smaller-radius polytope, albeit with potentially fewer faces and edges (see Fig. 7 (Left)). Although the faces of a positive-inside polytope also map to new linear faces, each vertex will have to expand to fill the gap between these new faces. As such, h ∆ is now a larger polytope whose corners are partial to a ball (see Fig. 7 (Right)). For the ℓ 1 and ℓ ∞ costs, h ∆ therefore remains a polytope since the ℓ 1 and ℓ ∞ balls are piecewise-linear themselves. For the ℓ 2 cost, h ∆ is now a polytope with rounded corners. In either case, the decision boundary of each p k ∆ ∈ P k ∆ can be seen as the intersection of a radius-scaled p k and k balls centered at each vertex of p k .

## B.7 Proof of Proposition 4

Proposition 4: Let Q k s be the class of negative-inside k -vertex polytopes bounded by a radius of s , then VC( Q k s ) ≥ VC( Q k s, ∆ ) . Moreover, if s = ∞ , then VC( Q k s ) = VC( Q k s, ∆ ) as implied by Thm. 2.

Proof. As noted in Appendix B.6, negative-inside polytopes map to smaller-radius negative-inside polytopes with potentially fewer faces and vertices. As such Q k s, ∆ ⊂ Q k s , so VC( Q k s, ∆ ) ≤ VC( Q k s ) .

Figure 8: Example h from a class H where VC( H ) = ∞ , but VC( H ∆ ) = 0 . The gray dashed curve is the h ∆ of any h like the one above. Each such h has the same small gap where the decision boundary remains above the dotted purple curve.

<!-- image -->

However, if s = ∞ , the class Q k s is closed to input scaling leading to VC( Q k s, ∆ ) ≥ VC( Q k s ) by Thm. 2, so we must have that VC( Q k s, ∆ ) = VC( Q k s ) .

## B.8 Proof of Theorem 1

Theorem 1: There exist several model classes H where VC( H ) = ∞ , but VC( H ∆ ) = 0 .

Proof. We provide two examples. For the first example, fix α &lt; 1 and consider the class of R 2 classifiers H = { h S : S ⊆ Z 2 } where h S = 1 { x ̸∈ B α -δ ( s ) ∀ s ∈ S } . Each classifier in this model class contains small, non-overlapping negative regions centered at lattice points. Thus, the negative regions of all h ∈ H will disappear under strategic movement, leaving h ∆ ( x ) = 1 ∀ h ∈ H . Despite the fact that H shatters Z 2 (and therefore has VC( H ) = ∞ ), H ∆ cannot shatter a single point (and therefore has VC( H ∆ ) = 0 ).

Another example where h ∆ is perhaps more natural can be seen in Fig. 8. In this case, H is the class of classifiers with narrow gaps whose decision boundaries can be written as

<!-- formula-not-decoded -->

where k is some constant less than α and each function f i ( x 1 ) satisfies f i ( x 1 ) ≥ √ α 2 -x 2 1 + √ α 2 -k 2 for all -k ≤ x 1 ≤ k (i.e., f i ( x ) lies above the radiusα arc between ( -k, 0) and ( k, 0) ). In this case, the decision boundary points in the gap will get wiped out on all h ∈ H , leaving the exact same h ∆ . Because H is nearly unrestricted in the range -k ≤ x 1 ≤ k , it is capable of shattering infinite points that lie in this range and has infinite VC dimension. On the other hand, because | H ∆ | = 1 , we get VC( H ∆ ) = 0 .

## B.9 Proof of Theorem 2

Definition 1. A hypothesis class H is said to be closed to input scaling if for all h ( x ) ∈ H and for all a &gt; 0 , we have that h a ( x ) = h ( ax ) ∈ H as well.

Note that the families of polynomials and polytopes are clearly closed to input scaling. Furthermore, most classes of neural networks are as well since h ( ax ) can be realized by scaling the first weight vector by a .

Theorem 2: If H is closed under input scaling and VC( H ) &lt; ∞ , then VC( H ) ≤ VC( H ∆ ) .

Proof. Intuitively, the reason that the VC dimension cannot decrease is that for any H with a VC of dimension k that is closed to input scaling, we can find a set S of k points that can be shattered by H , where none of the points s ∈ S strategically move under any of the h ∈ H used to shatter S . As a result, H ∆ will also shatter S , so its VC dimension is at least k .

Let VC( H ) = k and G ⊂ H be a set of 2 k classifiers that shatters some set S of k points. Define c DB ( s, g ) as the minimum strategic cost necessary for point s to obtain a positive classification from classifier g and define c neg ( S, G ) as the minimum strategic cost for any s ∈ S to obtain a positive classification from a classifier g ∈ G that classifies s negatively:

<!-- formula-not-decoded -->

Since both S and G are finite sets and negative points face a strictly positive cost to obtain a positive classification, we get that there exists a unique value c neg ( S, G ) &gt; 0 . Additionally, because the VC dimension is indifferent to input scaling 5 , if we scale both the classifiers and the points, then G ′ = { g ( ax ) : g ( x ) ∈ G } will shatter S ′ = { ax : x ∈ S } with c neg ( S ′ , G ′ ) = a · c neg ( S, G ) . Note that because H is closed to input scaling, G ′ ⊂ H as well. Furthermore, if we choose a = α +1 c neg ( S,G ) , then c neg ( S ′ , G ′ ) = α +1 . Note that if c neg ( S ′ , G ′ ) &gt; α , no point s ′ ∈ S ′ will move strategically under any g ′ ∈ G ′ , meaning that g ′ ∆ ( s ′ ) = g ′ ( s ′ ) ∀ g ′ ∈ G, s ′ ∈ S ′ . As a result, G ′ ∆ will also shatter S ′ so VC( G ′ ∆ ) ≥ | S ′ | = k . Since G ′ ∆ ⊂ H ∆ , we get that

<!-- formula-not-decoded -->

## B.10 Proof of Theorem 3

Theorem 3: Let H m,k be a class of piecewise-linear classifiers with at most m segments and k intersections of linear segments. Then VC( H m,k ∆ ) = O ( dmlog ( m ) + ν p k log ( k ) ) , where ν p is the VC of the class of ℓ p norm balls, and is Θ( d ) for p = 1 , 2 , ∞ .

Proof. We will make use of the following known results:

Lemma 2 (Gómez and Kaufmann (2021)) . Let H Ball be the class of all ℓ p balls in R d . Then H Ball has VC dimension VC( H Ball ) = Θ( d ) for p = 1,2, or ∞ [15].

Lemma 3 (Vaart and Wellner (2009)) . Let classes C 1 , C 2 , ..C k have VC dimensions V 1 , V 2 , ...V k and V = ∑ k i =1 V i . Let C U = ⊔ m j =1 C j ≡ { ⋃ m j =1 h j : h j ∈ C j } , and let C I = ⊔ m j =1 C j ≡ { ⋂ m j =1 h j : h j ∈ C j } . Then both C U and C I have VC dimension upper bounded by O ( V · log ( k ) ) [34].

We note that by combining Lemmas 2 and 3, the class H Ball k of the union of up to k ℓ p balls has a VC dimension of O ( ν p k log ( k ) ) , which becomes O ( d k log ( k ) ) in the case of either the ℓ 1 , ℓ 2 , or ℓ ∞ norm cost functions. Additionally, the class H Lin m of piecewise linear models with up to m segments has a VC dimension of O ( d m log ( m ) ) . 6

Because linear classifiers map to new shifted linear classifiers, all we have to do to determine the effective classifier of a piecewise-linear h is to combine (take the union of) the effective segments of each linear segment in h and the potential ℓ p ball artifacts around each intersection. Note that, as seen in Fig. 7 (Left), some of the effective segments may be wiped out in this process, and, as seen in Fig. 7 (Right), only intersections that open to the positive region will induce ℓ p ball artifacts in h ∆ . Therefore, the h ∆ of a piecewise linear h with m segments and k intersections will be the intersection of a piecewise linear classifier g with up to m segments and up to k ℓ p ball artifacts. As a result, we can write H ∆ ⊆ G ∩ H Ball k , where G ⊆ H Lin m . Therefore:

5 Input scaling only changes the distance between points, but does not change their relative positions, so the class H and its input-scaled class H ′ will have the same VC dimension.

6 Note that combining the two technically proves the bounds for the classes of the union of exactly k balls and exactly m segments, respectively, though we can easily expand them to the classes of up to k balls and up to m segments by recognizing that some of the balls/segments from classes C 1 , C 2 , ...C k may be the exact same.

<!-- formula-not-decoded -->

Consequently, if H is learnable, then H ∆ is too.

## B.11 Proof of Theorem 4

Theorem 4: Consider either the ℓ 1 , ℓ 2 , or ℓ ∞ costs and let H ⊆ P k , where P k is the set of all k -vertex polytopes in R d and has VC( P k ) = O ( d 2 k log k )[ 23 ] . Then VC( H ∆ ) = O ( d 2 k log k ) .

Proof. As seen in Appendix B.6, the effective classifier of a k-vertex polytope is the intersection of a larger radius (or smaller for negative inside polytopes) k-vertex polytope and k ℓ p balls centered at each of the vertices. Therefore, H ∆ ⊂ P k ∩ H ball k , so by Lemmas 2 and 3 (and the VC dimension of H ball k derived in Appendix B.10):

<!-- formula-not-decoded -->

## B.12 Proof of Proposition 5

Proposition 5: For all d , there exist (non-degenerate) distributions on R d that are realizable in the standard setting, but whose maximum strategic accuracy is the majority class rate, max y p ( y ) .

Proof. Fix d and choose any arbitrary set of positive points X 1 . When d = 1 , the set of negative points X 0 can be constructed by placing a negative point δ to the left and right of each positive point. When d &gt; 1 , this generalizes to selecting d +1 points around each positive point x ∈ X 1 that form an R d simplex T d δ ( x ) 7 . In this case, the classifier h ( x ) = 1 { x ∈ X 1 } achieves perfect standard accuracy, and the classifier that classifies all points as negative will achieve a strategic accuracy of d +1 d +2 . Note that for each point q in space to which x ∈ X 1 can strategically move, there exists at least one x ′ ∈ T d δ ( x ) that can as well. Therefore, for each positive point x ∈ X 1 that achieves a correct positive classification by strategically moving to q ( q can be the same as x if it doesn't move), there is a unique negative point x ′ ∈ T d δ ( x ) that achieves an incorrect negative classification by strategically moving to q , so the strategic accuracy does not benefit from correctly classifying any positive points. As a result, the maximum strategic accuracy is achieved by correctly classifying each negative point and incorrectly classifying each positive point, which gives the majority class rate.

## B.13 Proof of Proposition 6

Proposition 6: There exists a distribution that is realizable in the standard setting, but whose accuracy under any h ∆ is at most 0 . 5 + 1 ⌊ 2 α +1 ⌋ .

7 We can assume without loss of generality that ⋂ x 1 ,x 2 ∈X 1 T d δ ( x ) = ∅ ∀ x 1 , x 2 ∈ X 1 since there are infinite options for each T d δ , and we can just pick ones that do not overlap.

<!-- formula-not-decoded -->

Proof. Consider the set of lattice points in R 1 with alternating labels. Although f ( x ) = sin ( πx ) achieves perfect standard accuracy on this distribution, we will show that the maximum strategic accuracy for any model class is 0 . 5 + 1 ⌊ 2 α +1 ⌋ , which approaches 0.5 as α →∞ . Note that in R 1 , the minimum length of a positive interval is 2 α . Additionally, for each positive interval I of length k , I will contain either ⌊ k ⌋ or ⌊ k ⌋ + 1 points and include at most one more positive point than negative points. Therefore, each interval I will, at best, be correct on ⌊ k ⌋ +1 2 out of ⌊ k ⌋ points, which is optimized for k = 2 α (i.e., the minimum possible value of k ). Furthermore, because the negative intervals may also only have at most one more negative point than positive points, the negative intervals are optimized when they include only a single negative point. As such, an optimal strategic classifier h consists of repeated positive intervals that include ⌊ 2 α ⌋ points, followed by a negative interval that includes one negative point. For each positive and negative interval group, h correctly classifies at most ⌊ 2 α ⌋ +1 2 +1 out of ⌊ 2 α +1 ⌋ points correctly for an overall maximum accuracy of ⌊ 2 α +1 ⌋ 2 +1 ⌊ 2 α +1 ⌋ = 0 . 5 + 1 ⌊ 2 α +1 ⌋ .

## C Additional results

## C.1 Other Cost functions

Our low-level analysis (Sec. 4) begins with a focus on the ℓ 2 -norm cost, both because it is the most popular choice in the literature, and because we found it to offer the best intuition. Nonetheless, most of our results hold more broadly, and in particular, all theorems and propositions in Sec. 5 generalize to any ℓ p ( p ≥ 1 ) norm:

- Prop. 4 applies generally, since negative-inside polytopes are still mapped to smaller negative-inside polytopes under general ℓ p cost functions.
- Thm. 1, Cor. 1, and the example in Fig. 4 of increasing VC dimension remain under minor tweaks to the constructions that take into account the differences in shapes of ℓ p balls for different values of p ≥ 1 .
- Thm. 2 holds for any cost function under which the maximum Euclidean distance any user can strategically move is bounded by a constant. This holds not only for ℓ p norms, but also for most reasonable cost functions.
- Thm. 3 is already stated for general ℓ p norms.
- Thm. 4 is stated for ℓ 1 , ℓ 2 , and ℓ ∞ . However, a more general result can be stated with dependence on the VC dimension of the cost function balls (i.e., the set C = { c ( x, x ′ ) ≤ α ∀ x ∈ R d } ). The bound would then be O ( d 2 k log ( k ) + VC( C ) ) , and would support any plug-in results for VC( C ) .
- Prop. 5 and Prop. 6 also remain under minor tweaks to the constructions that take into account the differences in shapes of ℓ p balls for different values of p ≥ 1 .
- Obs. 3, 4, 5, and Cor. 2 hold for ℓ p norms in general as well, since they are not shape dependent, but rather results of movement within a ball.

Thus, our claims regarding the fact that non-learnable classes can become learnable, the loss of universality, and the limits to approximation all hold more generally.

In terms of asymmetric norms such as the Mahalanobis norm, while these norms would complicate the curvature analysis (and subsequent impossibility results that build off them), all results in Sec. 5 would still hold. This is because while the shape of balls changes (like with other ℓ p balls), fundamental properties like lines mapping to other lines do not. As such, the results in Sec. 5 would still hold for the exact same reasons as delineated above for ℓ p balls.

As for feature-dependent (also known as 'instance-wise') costs, these are qualitatively different from the conventional global costs, in a way that can make the claims become irrelevant or even degenerate. For example, Sundaram et al. [33] show that on the one hand, even for linear H , instance-wise cost function can cause VC( H ∆ ) = ∞ , and on the other hand, for separable costs (which are instance-wise) it holds that VC( H ∆ ) ≤ 2 for any H .

Figure 9: (Left) : Partial expansion. (Right) : Regions merging/splitting

<!-- image -->

## C.2 Partial Expansion

As seen in Fig. 9 (Left), partial expansion may occur if part of the expansion artifact is discarded through indirect wipeout . In the figure, part of the artifact (gold) is within α of the upper positive region and therefore does not become part of the decision boundary.

## C.3 Merging/Splitting Regions

As seen in Fig. 9 (Right), two positive regions can merge together when they are separated by a narrow pass in h . If the narrow pass is part of a larger negative region, the region will be split into two, which can potentially increase the complexity of the decision boundary (for example, in Fig. 9 (Left)).

## C.4 Types of Impossible Effective Classifiers

As stated in Sec. 4.4, we identify four "types" of classifiers g that cannot exist as effective classifiers. Formally, these classifiers are ones which include either:

1. A small continuous region of positively classified points C where ∃ q ∈ R d such that x ∈ B α ( q ) ∀ x ∈ C .
2. A narrow continuous region C where all points are classified as positive and are α -close to the decision boundary.
3. Asmooth point on the decision boundary which has signed curvature κ &gt; 1 /α (or κ ∗ &gt; 1 /α when d &gt; 2 ).
4. A locally convex intersection of piecewise linear or hyperlinear segments.

Note that the first case is just a special case of the second case. Nonetheless, we have separated them to emphasize that the impossible positive regions can either be fully or partially contained.

## Cases 1 and 2: Small/Narrow Positive Regions

Let x ∈ C be some point in a small positive region in g . For x to be classified as positive, there must be a point x ′ ∈ B α ( x ) which is classified positively by h and not reachable by any q ∈ X 0 ( h ) (see requirements 1 and 2 in Appendix B.4). However, any point x ′ ∈ B α ( x ) ∩ C is reachable by X 0 ( h ) and any point x ′ ∈ B α ( x ) ∩ C c is also reachable by X 0 ( h ) since it is closer to some q ∈ X 0 ( h ) than to x . Therefore, there are no points x ′ ∈ B α ( x ) which are not reachable by any q ∈ X 0 ( h ) , so g cannot be an effective classifier.

## Case 3: Large positive curvature

We will show that any g with directional curvature greater than 1 /α anywhere on the decision boundary will fail Prop. 3. In other words, if the directional curvature at point x is greater than 1 /α , then the point x + α ˆ n x will be reachable by some point that is negatively classified by g . In order for the point x + α ˆ n x to be reachable by x but not any point negatively classified by g , the

Figure 10: Diagram for proof of case 4.

<!-- image -->

ball B α ( x + α ˆ n x ) -which is the set of points that can strategically move to x + α ˆ n x -must not intersect g and must also be strictly tangent to it at x . However, if the curvature at point x is greater than 1 /α , then the maximum radius r insc of a ball that is strictly tangent to x is less than α , so g does not meet Prop. 3.

In R 2 , Lemma 1 proves that the radius r insc of the maximum inscribed circle at x is no bigger than the radius r osc = 1 κ of the osculating circle at x . We therefore have that r insc ≤ r osc = 1 κ &lt; α . In higher dimensions (see Appendix A.1), there is no longer a single normal curvature. Instead, we define the set K of curvatures in each plane p ∈ P defined by the normal vector and some tangent vector at x . Note that B α ( x + α ˆ n x ) is only strictly tangent to the decision boundary at x if it's intersection with each p ∈ P (a circle) is strictly tangent at x to the curve obtained by intersecting the decision boundary with the plane p . Since each of these cases are in R 2 , each one will only hold if the curvature at x in p is no more than 1 /α . Therefore, if κ ∗ = sup( K ) &gt; 1 /α for any x on the decision boundary, then the maximum radius of a ball that is strictly tangent to x is less than α , so g does not meet Prop. 3.

## Case 4: Piecewise Linear Segments

Lemma 4. The set of points that are reachable by point x but not point x + δ⃗ v for unit vector ⃗ v and δ → 0 all satisfy ⃗ v · x ≤ 0 .

Proof (Lemma 4). Define an orthonormal basis v 1 , v 2 ...v d with v 1 = v . For any point q such that ⃗ v · q &gt; 0 , the vector connecting x and q can be written as a 1 v 1 + a 2 v 2 ... + a d v d while the vector connecting x + δ⃗ v and q can be written as ( a 1 -δ ) v 1 + a 2 v 2 ... + a d v d . Since 0 &lt; δ &lt; a 1 (note that ⃗ v · q &gt; 0 so a 1 &gt; 0 ), q is closer to x + δ⃗ v than x . Therefore, any point that is reachable by x and not by x + δ⃗ v must satisfy ⃗ v · x ≤ 0 .

Proof (Case 4). As depicted in Fig. 10, let L 1 and L 2 be two R d hyperplane segments that intersect on the decision boundary of g at the R d -1 hyperplane segment I , and let x be some point on I. Let 0 &lt; θ &lt; π be the angle between L 1 and L 2 on the positive side. Let ⃗ v 1 be the unit vector in L 1 that runs through x and is orthogonal to I ; ⃗ v 2 be the unit vector in L 2 that runs through x and is orthogonal to I ; and ⃗ v C = -⃗ v 1 + ⃗ v 2 || ⃗ v 1 + ⃗ v 2 || . Note that for δ → 0 the points x 1 = x + δ⃗ v 1 and x 2 = x + δ⃗ v 2 are on the decision boundary while the point x 12 = x + δ⃗ v C is classified as positive and x C = x -δ⃗ v C is classified as negative. Additionally, if we define ⃗ v A as the the result of ⃗ v 1 being rotated 0 &lt; ϵ &lt; π -θ 2 around I away from the positive region and ⃗ v B as the result of ⃗ v 2 being rotated ϵ around I away from the positive region, then both x A = x + δ⃗ v A and x B = x + δ⃗ v B are classified as negative.

Assume that there exists a classifier h associated with the effective classifier g . Therefore, there exists a point q that is reachable by x but not by x A , x B , or x C (which are classified as negative by g ). By Lemma 4, any point q that is reachable under strategic movement by point x but not x A , x B , or x C must at least satisfy q · ⃗ v A ≤ 0 , q · ⃗ v B ≤ 0 , and q · ⃗ v C ≤ 0 . However, because ⃗ v A + ⃗ v B points in the same direction as ⃗ v 1 + ⃗ v 2 -and therefore the opposite direction of ⃗ v C -we must have that q · ⃗ v C = 0 and q · ( ⃗ v A + ⃗ v B ) = 0 , which implies that both q · ⃗ v A = 0 and q · ⃗ v B = 0 . However, this implies that the vector ⃗ q is orthogonal to both ⃗ v A and ⃗ v B , which can only happen if q lies on I or if the angle between ⃗ v A and ⃗ v B is either 0 or π . In the former case, q is then reachable

Figure 11: Example of VC decreasing from 2 to 1. Each color represents a different classifier (drawn together to illustrate shattering capacity), and the dashed curves represent the corresponding effective classifiers. Though the black points can be shattered by H , none of the h ∆ overlap so VC( H ∆ ) = 1 .

<!-- image -->

by the negatively classified point that is just over the decision boundary from it. The latter case is also not an option because the angle between ⃗ v A and ⃗ v B is θ AB = θ +2 ϵ , and we have ensured that 0 &lt; θ +2 ϵ &lt; θ +2 π -θ 2 = π . In either case, it is impossible for g to classify x as positive while classifying x A , x B , and x C as negative, so there is no h such that g = h ∆ .

## C.5 Non-bijectiveness of Classifier Mapping

In Sec. 4.4, we show that the mapping h ↦→ h ∆ is not surjective by showing that there are many potential h ∆ with no associated h . Here, we will show that h ↦→ h ∆ is also not injective since many classifiers map to the same effective classifier. Furthermore, we show that there are actually infinite h that map to nearly all possible h ∆ . Though there are many more interesting examples, the simplest case of non-injectiveness can be seen by considering an h that includes a continuous positive region C with an infinite number of points. Define the classifier h x as the classifier that is identical to h except that h x classifies x ∈ C as negative instead of positive. Because the point x will achieve a positive classification through strategic movement, h x ∆ is unaffected by x and is the same as h ∆ . Moreover, h x ∆ = h ∆ is true of any x ∈ C , so as long as any h that maps to h ∆ contains a positive region with infinite points, then h ∆ will have infinite h that all map to it. Because positive points in h map to positive radiusα balls in h ∆ , any h ∆ with a positive region C such that ∃ x : B α ( x ) ⊂ C will have infinite h that all map to it. Note that since all positive regions in h ∆ have a minimum size requirement ∃ x : B α ( x ) ⊆ C , the aforementioned requirement covers all but the h ∆ with absolute minimum-size positive regions.

## C.6 Classes with H = H ∆

̸

Though H = H ∆ for most classes H , there are a few rather simple H other than the class of linear functions where H = H ∆ . For example, under any ℓ p norm cost, an ℓ p ball with a positive inside will grow by a radius of α , while an ℓ p ball with a negative inside will shrink by a radius of α (at least until the radius hits 0). Therefore, many model classes of the subsets of the ℓ p balls - including the class of all negative-inside balls and the class of balls with a radius that is a multiple of α -all have H = H ∆ for any fixed α .

Additionally, since each line segment and ℓ p arc maps to another line segment or ℓ p arc, many piecewise-linear or piecewise ℓ p arc classes will also have H = H ∆ . For example, any concave linear intersection will be mapped to a shifted concave linear intersection, so the class of concave linear intersections satisfies H = H ∆ .

## C.7 Classes with VC( H ) &gt; VC( H ∆ ) &gt; 0

There are a couple of mechanisms that can lead to a decrease in VC dimension. In the first mechanism, multiple different h have the same h ∆ leading to a smaller and less expressive H ∆ . Two such cases are shown in Appendix B.8 to prove that the VC may decrease even from ∞ to 0. In a similar fashion,

we can build an H where only some (instead of all) of the h ∈ H have the same h ∆ , which would decrease the VC dimension to a non-zero effective VC dimension.

In the second mechanism, the VC can change due to effective classifiers gaining or losing overlap from strategic movement. As the dual to Fig. 4 (Left) where the VC increases from 1 to 2 due to this mechanism, Fig. 11 presents an example where the VC decreases from 2 to 1. Here, H includes four overlapping classifiers with VC( H ) = 2 . Yet, under strategic behavior, the effective classifiers cease to overlap, decreasing the shattering capacity to 1.

C.8 Example of H ∆ with VC( H ∆ ) = Θ( d · (VC( H ))

Claim: There exists a class H with VC( H ∆ ) = Θ( d · (VC( H )) .

Proof. For simplification of notation, we build on the case shown in Fig. 4 (Left) using classifiers h that classify some ball as positive and all other points as negative. Note that a similar proof can be built based on a class H of point classifiers [7], but we have decided to include the following construction since classifiers that classify only a single point as positive are quite unrealistic. Let set S = Z d 2 -{ 0 d } + {-δ d } be the set of Z d 2 lattice points with the point { 0 d } replaced by the point {-δ d } . Consider the case where α = 0 . 75 and H = { B 0 . 25 ( x ) : x ∈ S } is the class of radius-0.25 balls around the points in S . Because the radius of positive balls expand by α under strategic movement, we get that H ∆ = { B 1 ( x ) : x ∈ S } . Note that because none of the balls in H overlap, VC( H ) = 1 . On the other hand, the balls in H ∆ do overlap, allowing H ∆ to shatter the set of d one-hot points E = { e i : i ∈ [1 ...d ] } , where e i denotes the point with a 1 in the ith coordinate and 0's elsewhere. Note that for each x ∈ S -{-δ d } , the ball B 1 ( x ) ∈ H ∆ classifies E according to the sign pattern x . Additionally, the ball B 1 ( -δ d ) does not classify any point in E as positive, so H ∆ covers all 2 d sign patterns of E . As such, VC( H ∆ ) ≥ d . However, since H ∆ ⊂ B = { B 1 ( x ) : x ∈ R d } , we get that VC( H ∆ ) ≤ VC( B ) = d +1 [15] giving VC( H ∆ ) = Θ( d · ( V C ( H )) .

## C.9 Example of Strategic Accuracy Exceeding Standard Accuracy

Claim: In the unrealizable setting, the maximum strategic accuracy can exceed the maximum standard accuracy.

Proof. In the unrealizable setting, the maximum strategic accuracy can exceed the maximum standard accuracy when strategic movement helps correct the incorrect classification of positive points. For example, consider a basic setup where α = 1 , X = { ( -1 , 0) , (2 , 0) , ( -3 , 0) } , Y = { 1 , 1 , -1 } , and H = { h 1 ( x ) = sign ( x 1 + x 2 ) , h 2 ( x ) = sign ( x 1 -x 2 -1) } . In the standard case, both h 1 and h 2 correctly classify the second and third points, but not the first point, so the maximum accuracy is 2 3 . However, in the strategic setting, the first point is able to obtain a positive classification under h 1 by moving to ( -1 3 , 2 3 ) (and the third point still cannot get a positive prediction), so the maximum strategic accuracy is 1.

## C.10 Practical Takeaways

Given the challenge of optimization in strategic batch learning (even in simpler settings), our paper aims to first establish an understanding of the challenges inherent to non-linear strategic learning (in particular in relation to the linear case), with the hope that our results and conclusions can help guide the future design of learning algorithms, as well as set expectations for what is achievable and what is not. The difficulty of designing a principled algorithm for the general non-linear case is underscored by the fact that existing works rely on strong assumptions to enable optimization. For example, the original paper of Hardt et al. [16] provides an algorithm, but requires assumptions on the cost function that reduce the problem to a one-dimensional learning task, and their algorithm is essentially a line search. Levanon and Rosenfeld [25] differentiate through ∆ , but this relies strongly on this operation being a linear projection, which applies only to linear classifiers. Neither of the above naturally extend to the non-linear case.

Towards the goal of practical learning in the general strategic setting, we provide here some examples of how our results can be used as practical takeaways for effective learning:

1. Learning via inversion: One implication of our function-level analysis is that for any classifier h with strategic accuracy acc strat ( h ) , the classifier h ′ such that h ↦→ h ∆ = h ′ has regular (i.e., non-strategic) accuracy acc( h ′ ) = acc strat ( h ) . Thus, one approach to optimizing strategic accuracy is via reduction: (i) train h ′ ∈ H ′ to maximize non-strategic accuracy using any conventional approach for some choice of H ′ , and then (ii) apply the inverse function mapping (which may not be 1-to-1) to obtain a non-strategic classifier h , whose effective decision boundary is that of h ′ . This approach requires the ability to solve the 'inverse' problem (which may not be 1-to-1) to find the effective classifier. We discuss cases where this would and wouldn't work in Sec. 4.4.
2. Regularizing curvature: A sufficient condition for a pre-image strategic h ′ to exist for a given non-strategic h is that the function mapping is 1-to-1. Rather than restrict learning to only invertible classes H ′ a priori, an alternative approach is to work with general H ′ but regularize against those h ′ ∈ H ′ that do not have an inverse. Prop. 1 implies that one way to promote this is by encouraging h ′ that are smooth, since low-curvature classifiers are less prone to direct wipeout , and therefore more likely to permit inversion.
3. Finding the interpolation threshold: The interpolation threshold - the point in which the number of model parameters (or model complexity more generally) attains minimal training error - is key for learning in practice: it marks the extreme point of overfitting (in the classic underparametrized regime) and the beginning of benign overfitting (in the modern overparametrized regime, e.g., of deep neural nets). In non-strategic learning, this point is easily identified as that where the training error is zero. In contrast, our results show that in strategic learning, the minimal training error can be strictly positive. This makes it unclear when (and if) the threshold has been reached. Fortunately, our procedure for our approximation experiment in Sec. 6.2 can be used generally to compute the minimal attainable strategic training error, independently of the chosen model class. This provides a tool that can be used in practice to measure interpolation.

In terms of model class selection, our results in Sec. 5 offer insight into potential upsides and downsides to certain classes. One implication of our results is that strategic behavior can generally either increase or decrease the complexity of the chosen model class. However, Thm. 2 ensures that the VC dimension can only increase for any class that is closed to input scaling, which includes many common classes used in practice. Thus, a learner who chooses such classes is guaranteed that the potential expressivity of the learned classifier will not be deteriorated by strategic behavior. Additionally, Prop. 4, Thm. 4, and Thm. 3 all give upper bounds on the SVC of particular classes (e.g., ReLU neural networks). The implication is that a learner choosing to work with such H is guaranteed that if H is learnable, then the induced H ∆ is also learnable. Finally, Obs. 5 and Thm. 1 both point to the fact that several non-strategic classifiers h can be mapped to the exact same effective classifier h ∆ . If a class H includes many such cases, then it has significant redundancy, which can have negative implications on the process of choosing a good h ∈ H via learning (e.g., in terms of generalization, optimization, or the effectiveness of proxy losses). Ideally, the learner should be cautious of classes in which this is prevalent, for example, classes that permit high curvature or small negative areas.

## D Experimental details

## D.1 Experiment #1: Expressivity

In Sec. 6.1, we experimentally demonstrate how the strategic setting affects classifier expressivity. In this experiment, we first randomly sample a degreek polynomial h , then determine the set of points S that make up the decision boundary of h ∆ (i.e., are directly adjacent to it), and finally find the best approximate polynomial fit of S 8 . Because user features are often naturally bounded, we assume that x ∈ X = R 2 and that all x are bounded by | x | ∞ ≤ 100 (see Appendix A.2 for a discussion on how our theoretical results hold under bounded input domains). We report average values and

8 Because we are concerned specifically with the polynomial fit of h ∆ , we label the points in a fine grid G both regularly and then strategically, and use the strategic labels to get a polynomial fit. In this way, we do not directly compute the effective classifier (which would be difficult), but approximate it well enough to get the polynomial fit as well as visualize it.

standard error confidence intervals of 100 instances of the setup for each α ∈ { 2 , 4 , 8 , 16 , 24 , 32 , 40 } and k ∈ [1 , 10] . The instances were divided among 100 CPUs to speed up computation.

To randomly sample polynomials of degree k , we leverage the fact that the number of points needed to uniquely determine a degreek polynomial over R 2 is ( k +2 2 ) . Therefore, for each instance, we randomly choose ( k +2 2 ) points and labels, and use an SVM polynomial fitter to find the best fitting degreek polynomial fit to these points. To ensure that the generated polynomial is indeed a fulldegree polynomial, we construct a grid G of points over [ -100 , 100] 2 , label each point g ∈ G using h to get labels Y , and verify that the SVM best-fitting degreek polynomial achieves near-perfect accuracy on ( G,Y ), while the best ( k -1) -degree polynomial achieves low accuracy on ( G,Y ).

To find the set of points S that make up the decision boundary of h ∆ , we first strategically label each g ∈ G by checking if g can reach any point g ′ that is classified as positive by h . To determine the points that make up the decision boundary of h ∆ , we select only the points g ∈ G where g and at least one of its direct neighbors are labeled differently by h ∆ .

Finally, we again use an SVM polynomial fit to determine the best approximate polynomial degree of h ∆ . Because the h ∆ of a polynomial is not necessarily a polynomial itself, we set k ′ to be the lowest degree of the polynomial whose fit on S passes a set tolerance threshold. Empirical tests showed that a tolerance of 0.9 was high enough to ensure a good fit on all of G , but not so high that the SVM fit algorithm needlessly increased the degree just to get a complete overfit to S .

## D.2 Experiment #2: Approximation

In Sec. 6.2, we experimentally demonstrate the effects of non-universality on the maximum strategic accuracy. This experiment consists of (i) sampling synthetic data and (ii) calculating the maximum linear accuracy and strategic accuracy on each dataset instance. We report average values and standard error confidence intervals of 20 instances of the setup for each α ∈ { 0 . 5 , 1 , 2 } and µ ∈ [0 , 5] . The instances were divided among 100 CPUs to speed up computation.

For each instance, we generate synthetic R 2 data by sampling points from class-conditional Gaussians x ∼ N ( yµ, 1) , where µ is the separation between the centers of each class. Because we were unable to find a polynomial-time algorithm to calculate the maximum accuracy of H ∗ ∆ , exactly 25 points were drawn from each class (denote the full dataset X ). Though this setup may not reflect all possible data distributions, it does represent how the strategic environment behaves under increasing data separability.

To calculate the maximum linear accuracy in each instance, we first note that there exists an optimal linear classifier that runs through (or infinitesimally close to for negative points) exactly two dataset points. This is because for any optimal classifier h that runs through exactly one dataset point x ∈ X , h may be rotated until it runs through another x ′ ∈ X without changing any of the labels. For any optimal classifier h that runs through no x ∈ X , h can be shifted until it runs through one (or two collinear) x ∈ X and then rotated appropriately. Because all data points are random, we assume that no three points are collinear. Therefore, the maximum linear accuracy on X can be calculated by taking the maximum accuracy over the ( n 2 ) classifiers that run through each pair of x, x ′ ∈ X 9 .

When calculating the maximum strategic accuracy, we note that the minimum size of a positive region in h ∆ is the ball B α , so any strategic classifier can be represented by the intersection of multiple B α . We therefore calculate the set S of all possible subsets s ⊆ X that any B α classifies as positive, and find the intersection ⋂ s ∈ S ′ ⊆ S s with the highest accuracy. To find all s ∈ S , we iterate through the circles centered at each point on a fine-grained grid over the range of the dataset (padded by α on all sides) and check which points x ∈ X each circle includes. To find the best intersection of these circles, we iterate through each of the subsets of Z ⊆ X 0 10 and then s ∈ S to find the maximum number of positive points that can be classified as positive if we allow ourselves to mistakenly classify the negative points in Z .

9 Because points on the decision boundary are labeled as positive, we manually define that negative points used to define the linear classifier are accurately labeled to reflect the fact that the optimal classifier does not run through the point, but infinitely close to it.

10 Subsets are traversed in size order with early stopping once the size of the subset is large enough that the accuracy will be worse than the current best accuracy, even if all positive points are correctly classified.

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims in the abstract and introduction are based on the results in the main body of the paper and put into perspective based on the most recent results in the field.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Results that are based on strong assumptions are clearly noted and extensions to more general cases are discussed.

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

Justification: All assumptions and proofs are clearly set forth in either the main body of the paper or in referenced supplemental material.

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

Justification: All information needed to reproduce the experiments are clearly stated in the corresponding appendix sections of each experiment.

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

Justification: The full set of code used to run the experiments is attached in the supplemental works.

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

Justification: All hyperparameter details are included in the appropriate appendix sections discussing the experimental setup.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Standard error confidence intervals are reported on all figures.

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

Justification: Computer resources are delineated along with experimental setups.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research was conducted using ethical practices in every respect.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is fully theoretical and as such does not pose any positive or negative societal impacts.

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

Justification: No real data or models were used in this work.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No outside assets were used in this paper.

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

Justification: The paper does not release any new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing or human subjects.

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

Justification: Core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
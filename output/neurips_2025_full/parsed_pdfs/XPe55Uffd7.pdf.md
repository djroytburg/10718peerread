## Agnostic Active Learning Is Always Better Than Passive Learning

## Steve Hanneke

Department of Computer Science Purdue University steve.hanneke@gmail.com

## Abstract

This work resolves a long-standing open question of central importance to the theory of active learning, closing a qualitative and quantitative gap in our understanding of active learning in the non-realizable case. We provide the first sharp characterization of the optimal first-order query complexity of agnostic active learning, and propose a new general active learning algorithm which achieves it. Remarkably, the optimal query complexity admits a leading term which is always strictly smaller than the sample complexity of passive supervised learning (by a factor proportional to the best-in-class error rate). This was not previously known to be possible. For comparison, in all previous general analyses, the leading term exhibits an additional factor, such as the disagreement coefficient or related complexity measures, and therefore only provides improvements over passive learning in restricted cases. The present work completely removes such factors from the leading term, implying that every concept class benefits from active learning in the non-realizable case . Whether such benefits are possible has been the driving question underlying the past two decades of research on the theory of agnostic active learning. This work finally settles this fundamental question.

## 1 Introduction

Active learning is a well-known powerful variant of supervised learning, in which the learning algorithm interactively participates in the process of labeling the training examples. In this setting, there is a pool (or stream) of unlabeled examples, and the learning algorithm selects individual examples and queries an oracle (typically a human labeler) to observe their labels. This happens sequentially, so that the learner has observed previously-queried labels before deciding which example to query next. The intended purpose of active learning is to reduce the overall number of labels necessary for learning to a given accuracy, called the query complexity . We are therefore particularly interested in using active learning in scenarios where its query complexity is significantly smaller than the number of randomly-sampled training examples which would be needed to achieve the same accuracy, called the sample complexity of passive supervised learning.

Active learning has not only been incredibly useful for many practical machine learning problems (e.g., Cohn et al., 1996; Tong and Koller, 2001; Zhu et al., 2003; Olsson, 2009; Settles, 2012; Ren et al., 2021; Mosqueira-Rey et al., 2023) but has also given rise to a rich and nuanced theoretical literature (e.g., Dasgupta, 2005, 2011; Balcan et al., 2009; Hanneke, 2007b, 2014; Zhang and Chaudhuri, 2014; Hanneke and Yang, 2015; see Appendix A for a detailed survey). Moreover, the insights and techniques discovered in this literature have had tremendous influence on other branches of the learning theory literature (e.g., Awasthi et al., 2014; Foster et al., 2021; Hanneke, 2009b, 2016a,b, 2024; Zhivotovskiy and Hanneke, 2018; Simon, 2015; Balcan and Long, 2013; El-Yaniv and Wiener, 2010; Balcan et al., 2022).

Within the literature on the theory of active learning, a central topic which has garnered by-far the most interest is that of agnostic active learning: that is, the study of active learning algorithms capable of providing performance guarantees even in noisy or otherwise non-realizable learning problems, without assumptions on the form of the noise. This line of work was initiated by the groundbreaking A 2 algorithm (Agnostic Active) of Balcan, Beygelzimer, and Langford (2005, 2006, 2009) (with its general analysis later given by Hanneke, 2007b) and concurrently a lower bound analysis of Kääriäinen (2005, 2006) (later strengthened by Beygelzimer, Dasgupta, and Langford, 2009). These results were later refined and extended in numerous ways. However, throughout this two-decades long history, there has persisted a significant gap between the sharpest known upper and lower bounds on the optimal query complexity. Moreover, this gap represents an important qualitative distinction: while the lower bound is always smaller than the sample complexity of passive learning, the existing upper bounds only reflect such improvements under further restrictive conditions (e.g., bounded disagreement coefficient). Thus, the issue of resolving this gap is of central importance to this subject, since it has implications for answering the question:

Does every concept class admit benefits from using active learning instead of passive learning?

The main contribution of the present work is to establish that this is indeed true, and in fact the known lower bound is always attainable. To achieve this, we introduce new algorithmic principles for active learning (the AVID principle), improving concentration of error estimates via adaptively isolating regions where the error estimates have high variance and allocating more queries to such regions.

## 2 Background and Summary of the Main Result

̸

Let C be any concept class 1 (a set of functions X → { 0 , 1 } on a set X called the instance space ) and denote by d = VC( C ) the VC dimension of C (Vapnik and Chervonenkis, 1971; see Definition 4). Let P be an (unknown) joint distribution on X × { 0 , 1 } , and define the error rate of any classifier h : X → { 0 , 1 } as er P ( h ) := P (( x, y ) : h ( x ) = y ) . In the active learning problem, there is a sequence ( X 1 , Y 1 ) , . . . , ( X m , Y m ) of i.i.d. samples from P , but the learner initially only observes the X i values (the unlabeled examples). It then has the capability to query any example X i , which reveals the corresponding true label Y i , in a sequential manner (i.e., it chooses its next query X i ′ after observing the label Y i of its previous query point X i ). After a number of such queries, the learner returns a classifier ˆ h . The goal is to achieve a small excess error rate er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε while making as few queries as possible. We are particularly interested in quantifying the number of queries sufficient to achieve this, as a function of ε and the value of the best-in-class error rate inf h ∈ C er P ( h ) , known as a first-order query complexity bound.

Specifically, for any ε, δ, β ∈ (0 , 1) , the optimal query complexity , QC a ( ε, δ ; β, C ) , is defined as the minimal Q ∈ N for which there exists an active learner A a such that (for a sufficiently large number m of unlabeled examples), for every P with inf h ∈ C er P ( h ) ≤ β , with probability at least 1 -δ , A a makes at most Q queries and returns ˆ h satisfying er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε . The main quantity for comparison is the sample complexity of supervised passive learning . A passive learner A p simply trains on n labeled training examples ( X 1 , Y 1 ) , . . . , ( X n , Y n ) sampled i.i.d. from P to produce a classifier ˆ h . For ε, δ, β ∈ (0 , 1) , the optimal sample complexity of passive learning, M p ( ε, δ ; β, C ) , is defined as the minimal size n ∈ N of such a training sample for which there exists a passive learner A p such that, for every P with inf h ∈ C er P ( h ) ≤ β , with probability at least 1 -δ , A p returns ˆ h satisfying er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε . We remark that, in both the active and passive cases, these definitions place no restrictions on the computational efficiency of the learning algorithms, but rather focus on the data efficiency , which is our primary interest in this work (see Section G).

Since both the query complexity and sample complexity concern the number of labels sufficient for learning, it is natural to compare QC a ( ε, δ ; β, C ) with M p ( ε, δ ; β, C ) to quantify the benefits of active learning. Thus, the primary interest in the theory of agnostic active learning is quantifying how much smaller QC a ( ε, δ ; β, C ) is compared to M p ( ε, δ ; β, C ) . Since our interest is agnostic learning, it is most interesting to focus on the regime where P is far-from-realizable: that is, where β is much larger than ε . In this regime, it is well known from the works of Vapnik and Chervonenkis (1974); Devroye and Lugosi (1995); Hanneke, Larsen, and Zhivotovskiy (2024b) that the optimal sample

1 To focus on non-trivial cases, we suppose | C | ≥ 3 . We also suppose X is equipped with a σ -algebra specifying its measurable subsets, and we adopt the standard mild measure-theoretic restrictions on the σ -algebra and the class C from empirical process theory: namely, the image-admissible Suslin property (Dudley, 1999).

complexity of passive learning satisfies M p ( ε, δ ; β, C ) = Θ ( β ε 2 ( d +log ( 1 δ )) ) . In comparison, the known lower bound for active learning is QC a ( ε, δ ; β, C ) = Ω ( β 2 ε 2 ( d +log ( 1 δ )) ) (Kääriäinen, 2006; Beygelzimer, Dasgupta, and Langford, 2009). Thus, the strongest improvement we might hope from active learning is a factor of β (representing the best-in-class error rate).

However, in the prior literature, this β -factor improvement has only been demonstrated in upper bounds under restrictions to C or P . Specifically, every general upper bound on QC a ( ε, δ ; β, C ) in the literature has the form c ( β ) d β 2 ε 2 (ignoring logs), where c ( β ) is a ( C , P ) -dependent quantity. For instance, one commonly appearing such quantity c ( β ) is the disagreement coefficient θ ( β ) of Hanneke (2007b). We refer the reader to Appendix A for a detailed survey of such quantities c ( β ) which have appeared in the literature. Importantly, for all such upper bounds in the literature, the corresponding factor c ( β ) has the property that there exist simple classes C and distributions P for which c ( β ) ≥ 1 β (see Hanneke and Yang, 2015; Hanneke, 2016b, 2024): for instance, even for linear classifiers on R 2 or singletons on N . Note that when c ( β ) ≥ 1 β , a query complexity c ( β ) d β 2 ε 2 becomes no smaller than d β ε 2 , the sample complexity of passive learning. Moreover, one can show that avoiding such d β ε 2 query complexities would require new algorithmic techniques (see Appendix A). Naturally, the question of refining such c ( β ) factors has been a subject of much interest for many years. In particular, it has remained open whether such factors might even be avoided entirely , so that the β -factor improvement might always be achievable. In a series of talks, I conjectured that the lower bound Ω ( β 2 ε 2 ( d +log ( 1 δ )) ) is always sharp (in the far-from-realizable regime), and even offered a sizable prize for a solution (along with lower-order terms) (e.g., Hanneke and Nowak, 2019).

Contributions of this Work: In the present work, we completely resolve this question. We prove that (in the above regime) QC a ( ε, δ ; β, C ) = Θ ( β 2 ε 2 ( d +log ( 1 δ )) ) . In other words, the β -factor improvement is always achievable, the known lower bound is sharp , and there is no need for restrictions on ( C , P ) or additional factors c ( β ) as appear in all prior works.

Extending to the full range of β , the more-general form of the bound we prove also includes an additive lower-order term to account for the smallβ regime. In the simplest such bound (Theorem 1), this lower-order term is simply ˜ O ( d ε ) , so that the general form is QC a ( ε, δ ; β, C ) = ˜ O ( d β 2 ε 2 + d ε ) (Theorem 3 and Appendix F refine this lower-order term for some classes). For comparison, the general form of the passive sample complexity is M p ( ε, δ ; β, C ) = ˜ Θ ( d β ε 2 + d ε ) . We note that, even in the nearly-realizable regime ( β = ˜ O ( ε ) ), it is known that d ε is a lower bound on the query complexity for many classes C (Dasgupta, 2005; Hanneke, 2014; see Appendix D of Hanneke and Yang, 2015), so that this term is sometimes unavoidable, and hence the benefits of active learning can wane in the nearly-realizable regime. Likewise, the lower bound d β 2 ε 2 implies the benefits can also diminish in the very-high-noise regime ( β = Ω(1) ). In contrast, as discussed above, in the far-from-realizable regime ( √ ε ≤ β ≪ 1 ), the bound is of order d β 2 ε 2 , reflecting a β -factor improvement over the sample complexity of passive learning d β ε 2 . Additionally, the intermediate regime of moderate-size β (i.e., ε ≪ β &lt; √ ε ) also exhibits improvements over passive learning for all C : in this regime, M p ( ε, δ ; β, C ) = Ω ( d β ε 2 ) , whereas QC a ( ε, δ ; β, C ) = ˜ O ( d ε ) ≪ d β ε 2 , reflecting an improvement by a factor ˜ O ( ε β ) . Altogether, this result reveals a previously-unknown and truly remarkable fact: QC a ( ε, δ ; β, C ) ≪M p ( ε, δ ; β, C ) in all regimes ε ≪ β ≪ 1 , or in other words, in all regimes outside the nearly-realizable and very-high-noise cases, the following is true:

For every concept class C , the optimal query complexity of agnostic active learning is strictly smaller than the optimal sample complexity of agnostic passive learning.

This result resolves an important long-standing open question central to the past two decades of research on the theory of agnostic active learning.

## 3 Main Results

Formally, the following theorem expresses the new upper bound, together with known lower bounds for comparison (Kääriäinen, 2006; Beygelzimer, Dasgupta, and Langford, 2009; Hanneke, 2014; Hanneke and Yang, 2015). A more-detailed version of the result appears in Theorem 5 (Appendix C).

Theorem 1. For every concept class C , letting d = VC( C ) , ∀ ε, δ ∈ (0 , 1 / 8) , ∀ β ∈ [0 , 1] ,

<!-- formula-not-decoded -->

and QC a ( ε, δ ; β, C ) = Ω ( β 2 ε 2 ( d +log ( 1 δ )) ) . Moreover, for every d ∈ N there exists C with VC( C ) = d such that QC a ( ε, δ ; β, C ) = Ω ( β 2 ε 2 ( d +log ( 1 δ )) + d ε ) .

We provide a new general active learning algorithm A avid achieving this upper bound in Section 4. Importantly, the algorithm does not need to know β (or anything else about P ) to achieve this guarantee: i.e., it is completely adaptive to the value β . Moreover, the number of unlabeled examples the algorithm requires is only ˜ Θ ( d β ε 2 + d ε ) , of the same order as the sample complexity of passive learning; it can also adaptively determine how many unlabeled examples to use without knowing β .

The AVID Principle: The main innovation underlying the algorithm, which enables it to achieve this query complexity, represents a new principle for the design of active learning learning algorithms, which we call Adaptive Variance Isolation by Disagreements (AVID). The algorithm adaptively partitions the instance space X into regions , with the aim of isolating a region ∆ ⊆ X where it is most challenging to learn, due to exceptionally high variance in the error estimation problem in the ∆ region (where ∆ will be defined as a union of pairwise disagreement regions witnessing the high variance, carefully selected to ensure PX (∆) = O ( β ) ). It then allocates disproportionately more queries to this challenging region ∆ compared to the (considerably-easier) remaining region X \ ∆ . This idea has interesting connections to techniques explored in other branches of the literature (e.g., Hanneke, Larsen, and Zhivotovskiy, 2024b; Bousquet and Zhivotovskiy, 2021; Puchkin and Zhivotovskiy, 2022), discussed in Appendix A.

## 3.1 Refinement of the Lower-order Term for Some Classes

The AVID principle already suffices to achieve the query complexity bound in Theorem 1. Moreover, for most concept classes of interest, the query complexity bound in Theorem 1 is already optimal , matching a lower bound (up to log factors in the lower-order term): e.g., linear classifiers in R k , k ≥ 2 (Dasgupta, 2005; Hanneke, 2014; Hanneke and Yang, 2015). However, while the lead term β 2 ε 2 ( d +log ( 1 δ )) is already optimal for every concept class C , there do exist some special classes C for which a further refinement of the lower-order term d ε is possible (e.g., threshold classifiers 1 [ a, ∞ ) on R ). As our second main result, we provide a refinement of the upper bound in Theorem 1 to capture such special classes, thereby establishing a query complexity bound which is nearly optimal for every concept class.

Since such refinements are only possible for some concept classes, the expression of this refinement necessarily depends on an additional complexity measure of the class C . We prove that the optimal lower-order term in the query complexity is well-captured by a quantity known as the star number of C , introduced by Hanneke and Yang (2015). In particular, Hanneke and Yang (2015) showed that the star number precisely characterizes the optimal query complexity in the realizable case ( β = 0 ); since this is a limiting case of agnostic learning, it is natural that this quantity plays a crucial role in characterizing the optimal lower-order term. The formal definition is as follows.

̸

Definition 2. For any concept class C , the star number s = s ( C ) is the supremum n ∈ N for which ∃ x 1 , . . . , x n ∈ X and h 0 , h 1 , . . . , h n ∈ C such that ∀ i, j ∈ { 1 , . . . , n } , h i ( x j ) = h 0 ( x j ) ⇔ i = j .

The star number essentially describes a scenario which is intuitively challenging for active learners in the realizable case, wherein there is a set of instances x j and a default labeling h 0 ( x j ) , but the target concept is some h i which differs from h 0 at just one instance x i , unknown to the learner (which

must therefore query nearly all of these x j instances, searching for the special point x i , in order to identify the target concept h i ). Hanneke and Yang (2015) provide numerous examples calculating s for various concept classes. For instance, thresholds on R have s = 2 and decision stumps on R k have s = 2 k . However, it is worth noting that s is typically large (or infinite) for most concept classes of interest in learning theory (e.g., s = ∞ for linear classifiers on R k , k ≥ 2 ). This fact is important to the present work, since Hanneke and Yang (2015); Hanneke (2016b, 2024) have shown that the c ( β ) factors (discussed in Section 2 above) appearing in all previous general upper bounds all become no smaller than s ∧ 1 β in the worst case over distributions (subject to the β constraint). Thus, all general upper bounds c ( β ) d β 2 ε 2 from the prior literature become no smaller than d β ε 2 in the worst case when s = ∞ . In a sense, this means Theorem 1 is actually most interesting in the (typical) case of s = ∞ , since no previously known upper bounds offer any improvements over passive learning in this case (without further restrictions to P ), in stark contrast to Theorem 1 which has no dependence on s and provides improvements over passive learning in the lead term for every concept class.

Nonetheless, the special structure of classes with s &lt; ∞ turns out to provide some additional advantages for active learning, so that in order to state a general query complexity bound which is optimal for every concept class C , we need to account for this structure, via a dependence on s in the lower-order term. Specifically, by combining the AVID principle with existing principles for active learning (namely, disagreement-based queries), we can take further advantage of the power of active learning, thereby enabling a refinement of the lower-order term for classes with s &lt; ∞ . The following result presents a new general query complexity bound reflecting such refinements, together with a known lower bound for comparison (due to Kääriäinen, 2006; Beygelzimer, Dasgupta, and Langford, 2009; Hanneke and Yang, 2015). The implication is that this new upper bound is nearly optimal for every concept class C (including the lower-order term, up to a factor of d , which we discuss below). A more-detailed version of the result appears in Theorem 5 of Appendix C (and distribution-dependent variants are presented in Appendix F, replacing s with variants of the disagreement coefficient ).

Theorem 3. For every C , letting d = VC( C ) and s = s ( C ) , ∀ ε, δ ∈ (0 , 1 / 8) , ∀ β ∈ [0 , 1]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We may note that the upper bound in Theorem 1 is an immediate implication of Theorem 3 (we have stated Theorem 1 separately merely to emphasize that the improvements over passive learning are available without any special properties of C such as finite star number). Theorem 3 provides a refinement in the lower-order term compared to Theorem 1 when s &lt; 1 ε . In particular, for s &lt; ∞ , the asymptotic dependence on ε in the lower-order term is log 2 ( 1 ε ) . We leave open the question of whether this can be further refined to log ( 1 ε ) , which would match a known lower bound on this dependence for all infinite classes (Kulkarni, Mitter, and Tsitsiklis, 1993; Hanneke and Yang, 2015). The only significant difference between the upper and lower bounds in Theorem 3 is the factor of d in the lower-order term. I conjecture this term can be further refined to ˜ O ( s ∧ d ε ) , which is known to be sharp for some classes (Hanneke and Yang, 2015), and would fully answer a question posed by Hanneke and Nowak (2019). Beyond this, it is known that a gap between such lower-order terms in general upper and lower bounds is unavoidable if the only dependence on C is via d and s . Specifically, it follows from arguments in Appendix D of Hanneke and Yang (2015) that for some classes C this term should be ˜ Θ ( s ∧ d ε ) while for other classes C the term should be ˜ Θ ( s ∧ 1 ε + d ) . Thus, obtaining matching (bigΘ ) upper and lower bounds would require introducing a new complexity measure reflecting the distinctions between these types of classes, which we leave as an open question.

## 4 Algorithm and Outline of the Analysis

We next present the algorithm achieving Theorems 1 and 3 and a sketch of its analysis (the complete formal proof is given in Appendix E). Before stating the algorithm, we first introduce a few additional definitions and convenient notational conventions.

Error and disagreement regions: For any function h : X → { 0 , 1 } , define its error region

̸

<!-- formula-not-decoded -->

̸

In particular, note that er P ( h ) = P (ER( h )) . For any set V ⊆ C define the region of disagreement : DIS( V ) := { x ∈ X : ∃ f, g ∈ V, f ( x ) = g ( x ) } . For any two functions f, g : X → { 0 , 1 } , abbreviate by { f = g } := { x ∈ X : f ( x ) = g ( x ) } their pairwise disagreement region .

̸

̸

̸

Overloaded set notation: For convenience, we adopt a convention of treating sets A ⊆ X as notationally interchangeable with their labeled extension A ×{ 0 , 1 } ⊆ X × { 0 , 1 } . For instance, for functions f, g, h : X → { 0 , 1 } , we may write ER( h ) ∩ { f = g } , which, by the above convention, is interpreted as ER( h ) ∩ ( { f = g }×{ 0 , 1 } ) . Wealso overload notation for set intersections to allow for intersections of sets with sequences : that is, for any set Z , sequence S = { z 1 , . . . , z m } ∈ Z m , and set A ⊆ Z , we define S ∩ A as the subsequence { z i : i ≤ m,z i ∈ A } , and likewise S \ A := S ∩ ( Z\ A ) . We also apply these conventions in combination: i.e., for a sequence S ∈ ( X × { 0 , 1 } ) m and a set ∆ ⊆ X , we define S ∩ ∆ := S ∩ (∆ ×{ 0 , 1 } ) and S \ ∆ := S ∩ (( X \ ∆) ×{ 0 , 1 } ) .

̸

̸

Empirical estimates: We will make use of empirical estimates of quantities such as er P ( h ) and P X ( f = g ) . For any set Z and sequence S = { z 1 , . . . , z m } ∈ Z m , for any set A ⊆ Z , define the empirical measure : ˆ P S ( A ) := 1 m | S ∩ A | = 1 m ∑ m i =1 1 [ z i ∈ A ] . Again, we also apply these conventions in combination: i.e., for S ∈ ( X × { 0 , 1 } ) ∗ and ∆ ⊆ X , we define ˆ P S (∆) := ˆ P S (∆ ×{ 0 , 1 } ) . For any sequence S ∈ ( X × { 0 , 1 } ) ∗ and function h : X → { 0 , 1 } , define its empirical error rate (or empirical risk ): ˆ er S ( h ) := ˆ P S (ER( h )) .

Decision lists: We will often express decision-list aggregations of functions f, g : X → { 0 , 1 } . For instance, for any set ∆ ⊆ X , we may write h = f 1 X\ ∆ + g 1 ∆ to express a function h with h ( x ) = f ( x ) for x / ∈ ∆ and h ( x ) = g ( x ) for x ∈ ∆ .

## 4.1 The AVID Agnostic Algorithm: Adaptive Variance Isolation by Disagreements

We are now ready to describe the algorithm achieving the upper bounds in Theorems 1 and 3 (for full formality, some additional technical minutiae for the definition are given in Section C). Fix any values ε, δ ∈ (0 , 1) (the error and confidence parameters input to the learner). Fix any distribution P (unknown to the learner) and let ( X 1 , Y 1 ) , . . . , ( X m , Y m ) be independent P -distributed random variables (for any sufficiently large m , quantified explicitly in Theorem 5). The algorithm is stated in Figure 1, expressed in terms of certain quantities and data subsets defined as follows. 2 Let C := 11 10 , N := ⌈ log C ( 2 ε )⌉ , and for each k ∈ N define ε k := C 1 -k and m k := Θ ( 1 ε k ( d log ( 1 ε k ) +log ( 1 δ ) )) (see Section C for the precise constants). In Step 3, C ′ denotes an appropriate universal constant (see Section C). As defined in Figure 1, the algorithm makes use of different portions of the data ( S 1 k , S 2 k , S 3 k,i , S 4 k ) for different purposes, and to complete the definition of the algorithm we next specify how these data subsets are defined in the algorithm. We first split the initial 2 M 1 := 2 ∑ N +1 k =1 m k examples { ( X 1 , Y 1 ) , . . . , ( X 2 M 1 , Y 2 M 1 ) } into consecutive disjoint contiguous segments S 1 1 , . . . , S 1 N +1 , S 4 1 , . . . , S 4 N +1 , with the segments S 1 k and S 4 k being of size m k . The algorithm also allocates disjoint segments ( S 2 k , S 3 k,i ) of the remaining data { ( X i , Y i ) : 2 M 1 &lt; i ≤ m } , but does so adaptively during its execution. Specifically, if and when the algorithm reaches Step 2 with a value k , or reaches Step 9 (in which case let k = N +1 ), for the value i k and the set ∆ i k as defined at that time in the algorithm, it constructs a data subset S 2 k , allocating to S 2 k the next m ′ k consecutive examples which have not yet been allocated to any data subset S 1 k ′ , S 2 k ′ , S 3 k ′ ,i ′ , S 4 k ′ (i.e., fresh , previously-unused, examples), where, letting ˆ p k := 2 ˆ P S 4 k (∆ i k ) , we define m ′ k := Θ ( ˆ p k ε 2 k ( d +log ( 3+ N -k δ )) ) (see Section C for the precise constants). Similarly, if and when the algorithm reaches Step 5 with some values of ( k, i ) , it constructs a data subset S 3 k,i , allocating to S 3 k,i the next m k consecutive examples which have not yet been allocated.

2 For simplicity, we have expressed the algorithm as representing a set of surviving concepts V k ⊆ C . However, it should be clear from the definition that running the algorithm does not require explicitly storing V k . Rather, the various uses of this set can be implemented as constrained optimization problems (in Steps 4-6 and ˆ h k ), where the constraints are merely the inequalities which would define the sets V k ′ , k ′ ≤ k , and Step 3 is then replaced by simply adding one more constraint to the constraint set.

̸

```
Algorithm A avid Input: Error parameter ε , Confidence parameter δ , Unlabeled data X 1 , . . . , X m Output: Classifier ˆ h 0. Initialize i = i 1 = 0 , ∆ 0 = ∅ , V 0 = C 1. For k = 1 , . . . , N 2. Query all examples in S 1 k ∩ D k -1 \ ∆ i k and S 2 k ∩ ∆ i k 3. V k ← { h ∈ V k -1 : ˆ er 1,2 k ( h ) ≤ ˆ er 1,2 k ( ˆ h k ) + ε k C ′ } 4. If V k = ∅ or ˆ er 1,2 k ( ˆ h k ) < min h ∈ V k ˆ er 1,2 k ( h ) -ε k 4 C ′ , Then Return ˆ h := ˆ h k 5. While max f,g ∈ V k ˆ P S 3 k,i ( { f = g } \ ∆ i ) > ε k +2 6. ( f, g ) ← argmax ( f ′ ,g ′ ) ∈ V 2 k ˆ P S 3 k,i ( { f ′ = g ′ } \ ∆ i ) 7. ∆ i +1 ← ∆ i ∪ { f = g } , and update i ← i +1 8. i k +1 ← i 9. Query all examples in S 1 N +1 ∩ DN \ ∆ i N +1 and S 2 N +1 ∩ ∆ i N +1 and Return ˆ h := ˆ hN +1
```

Figure 1: The AVID Agnostic algorithm. Notations N , D k -1 , ε k , ˆ h k , S 1 k , S 2 k , S 3 k,i , ˆ er 1,2 k defined in the text.

To complete the definition of the algorithm, we define D k -1 , ˆ er 1,2 k , and ˆ h k , appearing in the algorithm, as follows. For each value of k encountered in the ' For ' loop, as well as for k = N +1 in the case the algorithm reaches Step 9, define (where V k -1 and ∆ i k are as defined in the algorithm):

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the definition of the A avid algorithm.

We remark that the examples in S 3 k,i and S 4 k are never queried in the algorithm, and thus the algorithm (necessarily) only uses the unlabeled X i values in these data subsets (to estimate certain marginal P X probabilities), so in fact these can be regarded as unlabeled data subsets. Similarly, the algorithm only queries a portion of S 1 k and S 2 k , and the remaining unqueried portions are in fact never used by the algorithm. For notational simplicity, we do not make these facts explicit in the notation.

Description of the algorithm: We briefly summarize the behavior of the algorithm (with explanations following in Section 4.2). As the algorithm iterates over rounds k of the ' For ' loop, it maintains a partition of the space into a region ∆ i k and its complement X \ ∆ i k . In each round, the algorithm refines a set V k of surviving concepts from C , aiming to prune out suboptimal concepts (Step 3). There are two crucial aspects of this, both in how the estimates of er P ( h ) are defined, and in the choice of function ˆ h k to which we compare. For the purpose of error estimation, in Step 2 it queries a number of random examples in X \ ∆ i k (or rather, the slightly smaller region D k -1 \ ∆ i k , since examples in X \ D k -1 are uninformative for estimating error differences ) and a number of random examples in ∆ i k . It uses the examples from each of the two regions to estimate the error rate of each h in that region, and combines these two estimates into an overall error estimate ˆ er 1,2 k ( h ) as in (1). It then prunes suboptimal concepts from V k -1 , removing all h ∈ V k -1 having estimated error ˆ er 1,2 k ( h ) &gt; ˆ er 1,2 k ( ˆ h k ) + ε k C ′ . The reason ˆ er 1,2 k ( h ) estimates error rates in the two regions separately is that, as it will turn out, we require a disproportionately larger number of samples to accurately estimate the error rates in the region ∆ i k compared to the complement X \ ∆ i k : for the latter, we use the samples in S 1 k ∩ D k -1 \ ∆ i k (queried in Step 2), where S 1 k has a modest size m k = ˜ Θ ( d ε k ) , while for the former we use the samples in S 2 k ∩ ∆ i k (also queried in Step 2), where S 2 k has a potentially larger size m ′ k roughly ˜ Θ ( P X (∆ i k ) d ε 2 k ) . The other crucial aspect in Step 3 is how we define the function ˆ h k to which we compare. For this, rather than (the seemingly-natural idea of) simply comparing to the smallest ˆ er 1,2 k ( h ) among h ∈ V k -1 , we instead compare to an even smaller value: the smallest ˆ er 1,2 k ( h ) among a more-complex class V (4) k -1 defined in (2), comprised of decision list

̸

̸

̸

functions which use one concept h 2 for predictions in ∆ i k , and use (equivalently) a majority vote of three concepts f, g, h 1 for predictions in X \ ∆ i k . ˆ h k is defined as a minimizer of ˆ er 1,2 k in V (4) k -1 , as in (3). This use of a more-complex comparator function is critical for certain parts of the proof (namely, keeping PX (∆ i k ) small). However, given that ˆ h k is chosen from a more-complex class, it becomes possible that ˆ h k may be substantially better than all h ∈ V k . In this event, the algorithm terminates early and returns ˆ h k (Step 4). Otherwise, if it makes it past this early-stopping case, its next objective is to define the region ∆ i k +1 for use in the next iteration. This occurs in the ' While ' loop (Steps 5-7). On each round of this loop, it uses a fresh data set S 3 k,i of size m k = ˜ Θ ( d ε k ) to check whether there exist f, g ∈ V k significantly distant from each other in the region X \ ∆ i (Step 5). If so, it adds their pairwise disagreement region { f = g } to the ∆ i region to define ∆ i +1 and increments i (Step 7). It repeats this until no such pair f, g exists, at which time it defines i k +1 = i (Step 8) and proceeds to the next iteration of the ' For ' loop. After N = O (log(1 /ε )) such iterations, it returns ˆ hN +1 (Step 9).

̸

We note that the algorithm's returned classifier ˆ h might not be an element of C (known as an improper learner), but rather can be represented as a (shallow) decision list of concepts from C . This aspect is quite important to certain parts of the proof, and we leave open the question of whether Theorems 1 and 3 are achievable by a proper learner (see Appendix G). We also remark that the D k -1 set is only needed for establishing Theorem 3: the algorithm achieves the query complexity bound in Theorem 1 even if we replace D k -1 with the full space X everywhere.

## 4.2 Principles and Outline of the Proof

Next we explain the high-level principles underlying the design of the algorithm, highlighting the two key innovations compared to previous approaches, which enable the improved query complexity guarantee (namely, separating out the ∆ i k regions, and the definition of ˆ h k ).

̸

̸

Empirical localization: The principles underlying the design of the algorithm begin with a familiar principle from statistical learning: empirical localization (Koltchinskii, 2006; Bartlett, Bousquet, and Mendelson, 2005). Specifically, the uniform Bernstein inequality (Lemma 7) implies that for an i.i.d. data set S , the sample complexity of uniform concentration of differences | ( ˆ er S ( f ) -ˆ er S ( g )) -(er P ( f ) -er P ( g )) | becomes smaller when the diameter diam( C ) = sup f,g ∈ C PX ( f = g ) of the concept class is small, noting that PX ( f = g ) bounds the variance of loss differences 1 [ f ( x ) = y ] -1 [ g ( x ) = y ] . Quantitatively, for any 0 &lt; ε ′ &lt; diam( C ) , ˜ Θ ( d diam( C ) ( ε ′ ) 2 ) samples S suffice to guarantee | ( ˆ er S ( f ) -ˆ er S ( g )) -(er P ( f ) -er P ( g )) | ≤ ε ′ . This fact leads to a natural well-known algorithmic principle, wherein we can prune from C concepts h having ˆ er S ( h ) -min h ′ ∈ C ˆ er S ( h ′ ) &gt; ε ′ (as the above inequality implies these verifiably have suboptimal error rates), leaving a subset V ′ 1 of surviving concepts, while preserving h ⋆ ∈ V ′ 1 , where h ⋆ := argmin h ∈ C er P ( h ) . Moreover, if these surviving concepts V ′ 1 have diam( V ′ 1 ) &lt; diam( C ) , we get an improved concentration guarantee for ˆ er S ( f ) -ˆ er S ( g ) among f, g ∈ V ′ 1 from the uniform Bernstein inequality, which enables us to prune even more concepts from V ′ 1 , leaving a set V ′ 2 of surviving concepts, and so on for V ′ 3 , V ′ 4 , . . . . Quantitatively, we can combine this with a schedule of resolutions ε k , so that as long as h ⋆ ∈ V ′ k -1 and diam( V ′ k -1 ) ≤ ε k , an i.i.d. data set S 1 k of size m k = ˜ Θ ( d ε k ) = ˜ Ω ( d diam( V ′ k -1 ) ε 2 k ) suffices to guarantee ∣ ∣ ( ˆ er S 1 k ( f ) -ˆ er S 1 k ( g ) ) -( er P ( f ) -er P ( g ) )∣ ∣ ≤ ε k C ′ , enabling us to further reduce to a subset V ′ k = { h ∈ V ′ k -1 : ˆ er S 1 k ( h ) ≤ min h ′ ∈ V ′ k -1 ˆ er S 1 k ( h ′ ) + ε k C ′ } for which all h ∈ V ′ k have er P ( h ) -er P ( h ⋆ ) ≤ 2 ε k C ′ , while preserving h ⋆ ∈ V ′ k . Iterating this N = Θ ( log C ( 1 ε )) times (recalling ε k = C 1 -k ) results in a subset V ′ N of concepts h with er P ( h ) -er P ( h ⋆ ) ≤ ε .

̸

Disagreement-based active learning: An additional observation, underlying many active learning algorithms ( disagreement-based methods), is that the above argument still holds while replacing ˆ er S 1 k ( h ) with ˆ P S 1 k (ER( h ) ∩ D ′ k -1 ) , where D ′ k -1 := DIS( V ′ k -1 ) . To see this, note that ∀ h, h ′ ∈ V ′ k -1 , ˆ P S 1 k (ER( h ) ∩ D ′ k -1 ) -ˆ P S 1 k (ER( h ′ ) ∩ D ′ k -1 ) = ˆ er S 1 k ( h ) -ˆ er S 1 k ( h ′ ) . Thus, we may equivalently define V ′ k = { h ∈ V ′ k -1 : ˆ P S 1 k (ER( h ) ∩ D ′ k -1 ) ≤ min h ′ ∈ V ′ k -1 ˆ P S 1 k (ER( h ′ ) ∩ D ′ k -1 ) + ε k C ′ } . Moreover, as long as diam( V ′ k -1 ) ≤ ε k , we have P X ( D ′ k -1 ) ≤ s ε k (Hanneke and Yang, 2015). Since the quantities in V ′ k only rely on the labels of examples in D ′ k -1 ∩ S 1 k , constructing V ′ k only requires

a number of queries O ( s ε k m k ) ∧ m k . Summing over k , these queries total to at most the claimed lower-order term in Theorem 3 (though note that even without this D ′ k -1 refinement we still recover the lower-order term from Theorem 1). So far, this is all essentially standard reasoning commonly followed in the prior literature on active learning (e.g., Hanneke, 2009b, 2014; Koltchinskii, 2010).

̸

Handling non-shrinking diameter: However, the above algorithmic principle breaks down if we reach a k with diam( V ′ k -1 ) = O ( ε k ) . This failure can easily occur in the agnostic setting, where it is possible for the set V ′ k -1 above to contain multiple relatively-good functions f, g which are nevertheless far from each other. 3 This is the motivation for the first key innovation in A avid : namely, if we ever reach such a k , where the V k set does not naturally have diam( V k ) ≤ ε k +1 (as tested in Step 5), the algorithm removes a portion of the space X to artificially reduce the diameter. Specifically, it identifies a pair f, g ∈ V k with P X ( f = g ) &gt; ε k +1 (intuitively, an obstruction to having low diameter) and separates out their pairwise disagreement region { f = g } from the region of focus of the algorithm (Steps 5-7). 4 Having set aside this region, the algorithm continues, focusing on the remaining set X \ { f = g } . This step is repeated, and these set-aside regions { f = g } are altogether captured in the set ∆ i (Step 7). Thus, we repeatedly find pairs f, g ∈ V k with P X ( { f = g } \ ∆ i ) &gt; ε k +1 (Steps 5-6) and add { f = g } to ∆ i (Step 7) until the diameter of V k on X \ ∆ i is reduced below ε k +1 . At that point, the algorithm proceeds to the next round ( k ← k +1 ). On the next round k , since we have (artificially) ensured the diameter of V k -1 is at most ε k in the region X \ ∆ i k , the uniform Bernstein argument implies m k examples S 1 k suffice to guarantee every f, g ∈ V k -1 have ˆ P S 1 k (ER( f ) ∩ D k -1 \ ∆ i k ) -ˆ P S 1 k (ER( g ) ∩ D k -1 \ ∆ i k ) within ± ε k 2 C ′ of P (ER( f ) \ ∆ i k ) -P (ER( g ) \ ∆ i k ) ) .

̸

̸

Error in the ∆ i k region: There remains the issue of estimating error rates in the ∆ i k isolated region. For this, the algorithm uses a data set S 2 k of size m ′ k ≈ d P X (∆ i k ) ε 2 k , queries all examples in S 2 k ∩ ∆ i k , and uses these to estimate the error rates P (ER( h ) ∩ ∆ i k ) in the ∆ i k region. By a refinement of the uniform convergence bound of Talagrand (1994) accounting for an envelope set ∆ i k (Lemma 8), this number m ′ k of examples suffices to ensure ∣ ∣ ∣ ˆ P S 2 k (ER( h ) ∩ ∆ i k ) -P (ER( h ) ∩ ∆ i k ) ∣ ∣ ∣ ≤ ε k 4 C ′ for every h ∈ C . Combining this with the above error-differences estimates in the X \ ∆ i k region, we can guarantee that the functions f, g ∈ V k -1 have ∣ ∣ ( ˆ er 1,2 k ( f ) -ˆ er 1,2 k ( g ) ) -(er P ( f ) -er P ( g )) ∣ ∣ ≤ ε k C ′ , recalling the definition of ˆ er 1,2 k from (1). Altogether, we conclude that, as long as h ⋆ ∈ V k -1 , a set V ′′ k := { h ∈ V k -1 : ˆ er 1,2 k ( h ) ≤ min h ′ ∈ V k -1 ˆ er 1,2 k ( h ′ ) + ε k C ′ } would contain only functions h satisfying er P ( h ) -er P ( h ⋆ ) ≤ 2 ε k C ′ while preserving h ⋆ ∈ V ′′ k . The actual definition of V k in Step 3 is only slightly different from this, for reasons we discuss next.

̸

Bounding the size of ∆ i k : Since the number of queries in S 2 k ∩ ∆ i k is ≈ d PX (∆ i k ) 2 /ε 2 k , if we hope to achieve a query complexity with lead term ˜ O ( d β 2 ε 2 ) it is crucial to guarantee PX (∆ i k ) = O ( β ) . This is the motivation for the second key innovation in A avid : defining the update in V k by comparison to the function ˆ h k in (3), rather than the best h ′ ∈ V k -1 . This turns out to be the most subtle part of the argument, requiring precise choices in the design of the algorithm. The essential argument is as follows. Suppose the algorithm reaches Step 6 for some ( k, i ) , so that it will add { f = g } to the ∆ i region. We then want to argue that P (ER( h ⋆ ) ∩ { f = g } \ ∆ i ) = Ω( P ( { f = g } \ ∆ i )) : that is, each time we add to ∆ i , we chop off a portion of ER( h ⋆ ) of size (under P ) proportional to the increase in PX (∆ i ) . Clearly if we can show this is always the case, we will inductively maintain PX (∆ i ) = O ( β ) , resulting in the claimed leading term in the query complexity. Now, to show this indeed occurs, we first note that one of f, g must err on at least half of { f = g } \ ∆ i k ; w.l.o.g. suppose it is f : that is, P (ER( f ) ∩ { f = g } \ ∆ i k ) ≥ 1 2 P X ( { f = g } \ ∆ i k ) . Now consider a function f ⋆ = f 1 { f = g }\ ∆ i k + h ⋆ 1 { f = g }\ ∆ i k + f 1 ∆ i k which replaces f by h ⋆ in the region { f = g } \ ∆ i k . Note that, if h ⋆ ∈ V k -1 , then f ⋆ ∈ V (4) k -1 defined in (2). Since ˆ h k has minimal ˆ er 1,2 k

̸

3 For instance, for C the class of intervals 1 [ a,b ] on R , with P X = Uniform([0 , 1]) and P ( Y = 1 | X ) = 1 [0 , 1 / 4] ∪ [3 / 4 , 1] ( X ) , the concepts 1 [0 , 1 / 4] and 1 [3 / 4 , 1] are both optimal among C , yet distance 1 / 2 apart.

4 This reasoning is somewhat reminiscent of the motivation for the splitting approach to active learning (Dasgupta, 2005), differing only in how we resolve the obstruction: whereas splitting would resolve it with queries to eliminate one element from each obstructing pair, here we resolve it by subtracting the pairwise disagreement region from the region of focus X\ ∆ i (see Appendix A.2.3). This idea is also related to a technique of Hanneke, Larsen, and Zhivotovskiy (2024b) for agnostic passive learning, discussed in Appendix A.3.

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

among V (4) k -1 , and f ∈ V k implies ˆ er 1,2 k ( f ) ≤ ˆ er 1,2 k ( ˆ h k ) + ε k C ′ , extending the above concentration of ˆ er 1,2 k differences to functions in V (4) k -1 (with appropriate adjustment of constants in m k , m ′ k ) implies er P ( f ) -er P ( f ⋆ ) ≤ 2 ε k C ′ . Thus, since f ⋆ and f only disagree on { f = g } \ ∆ i k , we have

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

In other words, P (ER( h ⋆ ) ∩ { f = g } \ ∆ i k ) ≥ 1 2 P X ( { f = g } \ ∆ i k ) -2 ε k C ′ . This is almost what we wanted, aside from having ∆ i k in place of ∆ i . We then argue P (ER( h ⋆ ) ∩ { f = g } \ ∆ i ) ≥ P (ER( h ⋆ ) ∩ { f = g } \ ∆ i k ) -P X ( { f = g } \ ∆ i k ) + P X ( { f = g } \ ∆ i ) , which (by the above) is at least P X ( { f = g } \ ∆ i ) -1 2 P X ( { f = g } \ ∆ i k ) -2 ε k C ′ . Since both f, g ∈ V k -1 , we know P X ( { f = g }\ ∆ i k ) ≤ ε k , so that this lower-bound is at least P X ( { f = g }\ ∆ i ) -ε k 2 -2 ε k C ′ . On the other hand, for appropriate constants in m k , the condition in Step 5 allows us to upper-bound ε k in terms of P X ( { f = g }\ ∆ i ) : namely, P X ( { f = g }\ ∆ i ) ≥ ε k c for c with C 2 &lt; c ≤ 3 2 ∧ C ′ 9 . Thus, we have P (ER( h ⋆ ) ∩{ f = g }\ ∆ i ) ≥ ( 1 -c 2 -2 c C ′ ) P X ( { f = g }\ ∆ i ) . Since each ∆ i k is a union of such (disjoint) { f = g } \ ∆ i regions ( i &lt; i k ), β ≥ P (ER( h ⋆ ) ∩ ∆ i k ) ≥ ( 1 -c 2 -2 c C ′ ) P X (∆ i k ) .

The early stopping case: The above argument for P X (∆ i k ) = O ( β ) hinges on having h ⋆ ∈ V k -1 . However, since ˆ h k is a more-complex function than h ⋆ , there is a chance that h ⋆ / ∈ V k after Step 3. For this reason, we have added the early stopping case in Step 4. By using slightly tighter concentration inequalities than used to update V k , this step effectively tests that ˆ h k is not so much better than all concepts in V k -1 that h ⋆ might have been removed. Thus, if we make it past Step 4, we maintain h ⋆ ∈ V k so that the above argument applies on the next round. On the other hand, in the event that this test fails, we have effectively verified that ˆ h k is at least slightly better than all concepts in V k -1 (including h ⋆ ), and we can safely return ˆ h k in this case.

̸

Overall behavior: The effective overall behavior of the algorithm is to isolate in the region ∆ i k the most-challenging part of the error estimation problem, due to the high variance (diameter) of the error differences in that region. It then allocates a disproportionately larger number of queries S 2 k ∩ ∆ i k to this region, toward estimating the error rates there. By comparing with the function ˆ h k (which separately optimizes errors in pairwise difference regions { f = g } \ ∆ i k ) in the definition of V k , we can maintain that ∆ i k never grows larger than O ( β ) , so that the number of queries in S 2 k ∩ ∆ i k does not grow excessively large. The remaining region X \ ∆ i k enjoys the property that the set V k -1 has diameter ≤ ε k , so that we can easily estimate error differences in this region by a uniform Bernstein inequality. Altogether, after at most N = O ( log ( 1 ε )) rounds, this achieves the objective of ε excess error rate, while using a number of queries as stated in the query complexity bound in Theorem 3. The formal proof is given in Appendix E.

## 5 Conclusions and Summary of the Appendices

This work resolves a long-standing open question of central importance to the theory of active learning, proving that every concept class benefits from active learning in the non-realizable case. Quantitatively, we establish a new sharp upper bound on the optimal query complexity, with leading term that is smaller than that of passive learning by a factor proportional to the best-in-class error rate.

The appendices include the formal proofs, along with additional contents. Appendix A presents a thorough summary of related work and background on the theory of active learning, as well as other works with techniques related to those used here. Appendix C presents remaining minutiae for the definition of A avid , along with a more-detailed version of Theorem 3, including formal claims regarding the number of unlabeled examples. Appendix E presents the formal proof of Theorem 3. Appendix F presents distribution-dependent refinements of Theorem 3, which replace the star number s with certain P -dependent complexity measures: variants of the disagreement coefficient. We further argue that the disagreement coefficient θ P ( ε ) , as originally defined by Hanneke (2007b), provably cannot be attained as a replacement for s in the lower-order term (by any algorithm), while on the other hand A avid does achieve a lower-order term ˜ O ( θ P ( β + ε ) 2 d ) . We also present subregion-based refinements of the algorithm and analysis, based on techniques of Zhang and Chaudhuri (2014). Appendix G presents extensions ( multiclass classification, stream-based active learning), along with several open questions and future directions.

̸

̸

̸

̸

̸

̸

̸

̸

̸

## References

- N. Ailon, R. Begleiter, and E. Ezra. Active learning using smooth relative regret approximations with applications. Journal of Machine Learning Research , 15(3):885-920, 2014.
- D. Angluin. Queries and concept learning. Machine Learning , 2(4):319-342, 1987. doi: 10.1007/ BF00116828. URL https://doi.org/10.1007/BF00116828 .
- J. Asilis, S. Devic, S. Dughmi V. Sharan, and S.-H. Teng. Proper learnability and the role of unlabeled data. In Proceedings of the 36 th International Conference on Algorithmic Learning Theory , 2025a.
- J. Asilis, M. M. Høgsgaard, and G. Velegkas. Understanding aggregations of proper learners in multiclass classification. In Proceedings of the 36 th International Conference on Algorithmic Learning Theory , 2025b.
- A. C. Atkinson and A. N. Donev. Optimum Experimental Designs . Clarendon Press, 1992.
- P. Awasthi, V. Feldman, and V. Kanade. Learning using local membership queries. In Proceedings of the 26 th Conference on Learning Theory , 2013.
- P. Awasthi, M.-F. Balcan, and P. M. Long. The power of localization for efficiently learning linear separators with noise. In Proceedings of the 46 th ACM Symposium on the Theory of Computing , 2014.
8. M.-F. Balcan and A. Blum. A discriminative model for semi-supervised learning. Journal of the ACM , 57(3):1-46, 2010.
9. M.-F. Balcan and S. Hanneke. Robust interactive learning. In Proceedings of the 25 th Conference on Learning Theory , 2012.
10. M.-F. Balcan and P. M. Long. Active and passive learning of linear separators under log-concave distributions. In Proceedings of the 26 th Conference on Learning Theory , 2013.
11. M.-F. Balcan and H. Zhang. Sample and computationally efficient learning algorithms under sconcave distributions. 2017.
12. M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. In NIPS Workshop on Foundations of Active Learning , 2005.
13. M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. In Proceedings of the 23 rd International Conference on Machine Learning , 2006.
14. M.-F. Balcan, A. Broder, and T. Zhang. Margin based active learning. In Proceedings of the 20 th Conference on Learning Theory , 2007.
15. M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. Journal of Computer and System Sciences , 75(1):78-89, 2009.
16. M.-F. Balcan, S. Hanneke, and J. Wortman Vaughan. The true sample complexity of active learning. Machine Learning , 80(2-3):111-139, 2010.
17. M.-F. Balcan, A. Blum, S. Hanneke, and D. Sharma. Robustly-reliable learners under poisoning attacks. In Proceedings of the 35 th Conference on Learning Theory , 2022.
- P. Bartlett, M. I. Jordan, and J. McAuliffe. Convexity, classification, and risk bounds. Journal of the American Statistical Association , 101(473):138-156, 2006.
- P. L. Bartlett, O. Bousquet, and S. Mendelson. Local rademacher complexities. The Annals of Statistics , 33(4):1497-1537, 2005.
- E. Baum. Neural net algorithms that learn in polynomial time from examples and queries. IEEE Transactions on Neural Networks , 2(1):5-19, 1991.
- E. Baum and K. Lang. Query learning can work poorly when a human oracle is used. In Proceedings of the International Joint Conference in Neural Networks , 1992.

- G. Bennett. Probability inequalities for the sum of independent random variables. Journal of the American Statistical Association , 57(297):33-45, 1962.
- S. Bernstein. On a modification of Chebyshev's inequality and of the error formula of Laplace. Annales Scientifiques de l'Institut de la Société des Savants d'Ukraine, Section de Mathématiques , 1(4):38-49, 1924.
- A. Beygelzimer, S. Dasgupta, and J. Langford. Importance weighted active learning. In Proceedings of the 26 th International Conference on Machine Learning , 2009.
- A. Beygelzimer, D. Hsu, J. Langford, and T. Zhang. Agnostic active learning without constraints. In Advances in Neural Information Processing Systems 23 , 2010.
- S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence . Oxford University Press, 2013.
- O. Bousquet. A Bennett concentration inequality and its application to suprema of empirical processes. Comptes Rendus Mathematique , 334(6):495-500, 2002.
- O. Bousquet and N. Zhivotovskiy. Fast classification rates without standard margin assumptions. Information and Inference: A Journal of the IMA , 10(4):1389-1421, 2021.
- O. Bousquet, S. Hanneke, S. Moran, and N. Zhivotovskiy. Proper learning, Helly number, and an optimal SVM bound. In Proceedings of the 33 rd Conference on Learning Theory , 2020.
- N. Brukhim, D. Carmon, I. Dinur, S. Moran, and A. Yehudayoff. A characterization of multiclass learnability. In Proceedings of the 63 rd Annual IEEE Symposium on Foundations of Computer Science , 2022.
- B. G. Buchanan. Scientific theory formation by computer. In Computer Oriented Learning Processes , pages 515-534. 1976.
- A. D. Bull. Spatially-adaptive sensing in nonparametric regression. The Annals of Statistics , 1:41-62, 2013.
- G. Cavallanti, N. Cesa-Bianchi, and C. Gentile. Learning noisy linear classifiers via adaptive and selective sampling. Machine Learning , 83:71-102, 2011.
- O. Chapelle, B. Scholkopf, and A. Zien. Semi-Supervised Learning . Adaptive Computation and Machine Learning Series. MIT Press, 2006.
- K. Chaudhuri, S. M. Kakade, P. Netrapalli, and S. Sanghavi. Convergence rates of active learning for maximum likelihood estimation. In Advances in Neural Information Processing Systems 28 , 2015.
- H. Chernoff. A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. The Annals of Mathematical Statistics , pages 493-507, 1952.
- D. Cohn, L. Atlas, and R. Ladner. Improving generalization with active learning. Machine Learning , 15(2):201-221, 1994.
- D. A. Cohn, Z. Ghahramani, and M. I. Jordan. Active learning with statistical models. Journal of Artificial Intelligence Research , 4:129-145, 1996.
- C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Active learning with region graphs. In Proceedings of the 36 th International Conference on Machine Learning , 2019a.
- C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Region-based active learning. In Proceedings of the 22 nd International Conference on Artificial Intelligence and Statistics , 2019b.
- C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Active learning with disagreement graphs. In Proceedings of the 36 th International Conference on Machine Learning , 2019c.
- C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Adaptive region-based active learning. In Proceedings of the 37 th International Conference on Machine Learning , 2020.

- A. Daniely and S. Shalev-Shwartz. Optimal learners for multiclass problems. In Proceedings of the 27 th Conference on Learning Theory , 2014.
- S. Dasgupta. Analysis of a greedy active learning strategy. In Advances in Neural Information Processing Systems 17 , 2004.
- S. Dasgupta. Coarse sample complexity bounds for active learning. In Advances in Neural Information Processing Systems 18 , 2005.
- S. Dasgupta. The two faces of active learning. Theoretical Computer Science , 412(19), 2011.
- S. Dasgupta, A. T. Kalai, and C. Monteleoni. Analysis of perceptron-based active learning. In Proceedings of the 18 th Conference on Learning Theory , 2005.
- S. Dasgupta, D. Hsu, and C. Monteleoni. A general agnostic active learning algorithm. In Advances in Neural Information Processing Systems 20 , 2007.
- M. H. DeGroot. Uncertainty, information, and sequential experiments. The Annals of Mathematical Statistics , 33(2):404-419, 1962.
- O. Dekel, C. Gentile, and K. Sridharan. Selective sampling and active learning from single and multiple teachers. Journal of Machine Learning Research , 13(9):2655-2697, 2012.
- G. DeSalvo, C. Gentile, and T. S. Thune. Online active learning with surrogate loss functions. Advances in Neural Information Processing Systems 34 , 2021.
- L. Devroye and G. Lugosi. Lower bounds in pattern recognition and learning. Pattern Recognition , 28:1011-1018, 1995.
- I. Diakonikolas, D. Kane, and M. Ma. Active learning of general halfspaces: Label queries vs membership queries. In Advances in Neural Information Processing Systems 37 , 2024.
- R. M. Dudley. Uniform Central Limit Theorems . Cambridge University Press, 1999.
- S. Efromovich. Sequential design and estimation in heteroscedastic nonparametric regression. Sequential Analysis , 26(1):3-25, 2007.
- B. Eisenberg. On the Sample Complexity of PAC-Learning using Random and Chosen Examples . PhD thesis, Massachusetts Institute of Technology, 1992.
- B. Eisenberg and R. Rivest. On the sample complexity of PAC-learning using random and chosen examples. In Proceedings of the 3 rd Annual Workshop on Computational Learning Theory , 1990.
- R. El-Yaniv and Y. Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research , 11(5):1605-1641, 2010.
- R. El-Yaniv and Y. Wiener. Active learning via perfect selective classification. Journal of Machine Learning Research , 13(2):255-279, 2012.
- V. V. Fedorov. Theory of Optimal Experiments . Academic Press, 1972.
- R. A. Fisher. The Design of Experiments . Oliver and Boyd, 1935.
- D. J. Foster, A. Rakhlin, D. Simchi-Levi, and Y. Xu. Instance-dependent complexity of contextual bandits and reinforcement learning: A disagreement-based perspective. In Proceedings of the 34 th Conference on Learning Theory , 2021.
- Y. Freund, H. S. Seung, E. Shamir, and N. Tishby. Selective sampling using the query by committee algorithm. Machine Learning , 28:133-168, 1997.
- E. Friedman. Active learning for smooth problems. In Proceedings of the 22 nd Conference on Learning Theory , 2009.
- R. Gelbhart and R. El-Yaniv. The relationship between agnostic selective classification, active learning and the disagreement coefficient. Journal of Machine Learning Research , 20(33):1-38, 2019.

- E. Giné and V. Koltchinskii. Concentration inequalities and asymptotic results for ratio type empirical processes. The Annals of Probability , 34(3):1143-1216, 2006.
- S. A. Goldman and M. J. Kearns. On the complexity of teaching. Journal of Computer and System Sciences , 50:20-31, 1995.
- A. Gonen, S. Sabato, and S. Shalev-Shwartz. Efficient active learning of halfspaces: An aggressive approach. The Journal of Machine Learning Research , 14(1):2583-2615, 2013.
- S. Hanneke. Teaching dimension and the complexity of active learning. In Proceedings of the 20 th Conference on Learning Theory , 2007a.
- S. Hanneke. A bound on the label complexity of agnostic active learning. In Proceedings of the 24 th International Conference on Machine Learning , 2007b.
- S. Hanneke. Adaptive rates of convergence in active learning. In Proceedings of the 22 nd Conference on Learning Theory , 2009a.
- S. Hanneke. Theoretical Foundations of Active Learning . PhD thesis, Machine Learning Department, School of Computer Science, Carnegie Mellon University, 2009b.
- S. Hanneke. Rates of convergence in active learning. The Annals of Statistics , 39(1):333-361, 2011.
- S. Hanneke. Activized learning: Transforming passive to active with improved label complexity. Journal of Machine Learning Research , 13(5):1469-1587, 2012.
- S. Hanneke. Theory of disagreement-based active learning. Foundations and Trends in Machine Learning , 7(2-3):131-309, 2014.
- S. Hanneke. The optimal sample complexity of PAC learning. Journal of Machine Learning Research , 17(38):1-15, 2016a.
- S. Hanneke. Refined error bounds for several learning algorithms. Journal of Machine Learning Research , 17(135):1-55, 2016b.
- S. Hanneke. The star number and eluder dimension: Elementary observations about the dimensions of disagreement. In Proceedings of the 37 th Conference on Learning Theory , 2024.
- S. Hanneke and A. Kontorovich. Stable sample compression schemes: New applications and an optimal SVM margin bound. In Proceedings of the 32 nd International Conference on Algorithmic Learning Theory , 2021.
- S. Hanneke and S. Kpotufe. A no-free-lunch theorem for multitask learning. The Annals of Statistics , 50(6):3119-3143, 2022.
- S. Hanneke and R. Nowak. Tutorial on Active Learning: From Theory to Practice. In The 36 th International Conference on Machine Learning , 2019. URL https://youtu.be/0TADiY7iPAc? t=5865 .
- S. Hanneke and L. Yang. Negative results for active learning with convex losses. In Proceedings of the 13 th International Conference on Artificial Intelligence and Statistics , 2010.
- S. Hanneke and L. Yang. Minimax analysis of active learning. Journal of Machine Learning Research , 16(12):3487-3602, 2015.
- S. Hanneke and L. Yang. Surrogate losses in passive and active learning. Electronic Journal of Statistics , 13(2):4646-4708, 2019.
- S. Hanneke, A. Karbasi, S. Moran, and G. Velegkas. Universal rates for active learning. In Advances in Neural Information Processing Systems 37 , 2024a.
- S. Hanneke, K. G. Larsen, and N. Zhivotovskiy. Revisiting agnostic PAC learning. In Proceedings of the 65 th IEEE Symposium on Foundations of Computer Science , 2024b.
- S. Har-Peled, D. Roth, and D. Zimak. Maximum margin coresets for active and noise tolerant learning. In Proceedings of the 35 th International Joint Conference on Artificial Intelligence , 2007.

- D. Haussler. Decision theoretic generalizations of the PAC model for neural net and other learning applications. Information and Computation , 100:78-150, 1992.
- D. Haussler and P. M. Long. A generalization of Sauer's lemma. Journal of Combinatorial Theory, Series A , 71(2):219-240, 1995.
- T. Hegedüs. Generalized teaching dimensions and the query complexity of learning. In Proceedings of the 8th Conference on Computational Learning Theory , 1995.
- L. Hellerstein, K. Pillaipakkamnatt, V. Raghavan, and D. Wilkins. How many queries are needed to learn? Journal of the Association for Computing Machinery , 43(5):840-862, 1996.
- M. Hopkins, D. Kane, S. Lovett, and G. Mahajan. Point location and active learning: Learning halfspaces almost optimally. In Proceedings of the 61 st Annual IEEE Symposium on Foundations of Computer Science , 2020.
- D. Hsu. Algorithms for Active Learning . PhD thesis, Department of Computer Science and Engineering, School of Engineering, University of California, San Diego, 2010.
7. T.-K. Huang, A. Agarwal, D. J. Hsu, J. Langford, and R. E. Schapire. Efficient and parsimonious agnostic active learning. In Advances in Neural Information Processing Systems 28 , 2015.
- J. C. Jackson. An efficient membership-query algorithm for learning DNF with respect to the uniform distribution. Journal of Computer and System Sciences , 55(3):414-440, 1997.
- M. Kääriäinen. On active learning in the non-realizable case. In NIPS Workshop on Foundations of Active Learning , 2005.
- M. Kääriäinen. Active learning in the non-realizable case. In Proceedings of the 17th International Conference on Algorithmic Learning Theory , 2006.
- M. J. Kearns, R. E. Schapire, and L. M. Sellie. Toward efficient agnostic learning. Machine Learning , 17:115-141, 1994.
- V. Koltchinskii. Local Rademacher complexities and oracle inequalities in risk minimization. The Annals of Statistics , 34(6):2593-2656, 2006.
- V. Koltchinskii. Rademacher complexities and bounding the excess risk in active learning. Journal of Machine Learning Research , 11(9):2457-2485, 2010.
- S. R. Kulkarni, S. K. Mitter, and J. N. Tsitsiklis. Active learning using arbitrary binary valued queries. Machine Learning , 11:23-35, 1993.
- J. Lewi, R. Butera, and L. Paninski. Sequential optimal design of neurophysiology experiments. Neural Computation , 21(3):619--687, 2009.
- S. Mahalanabis. A note on active learning for smooth problems. arXiv :1103.3095 , 2011.
- E. Mammen and A.B. Tsybakov. Smooth discrimination analysis. The Annals of Statistics , 27(6): 1808-1829, 1999.
- P. Massart and E. Nédélec. Risk bounds for statistical learning. The Annals of Statistics , 34(5): 2326-2366, 2006.
- T. Mitchell. Version Spaces: An Approach to Concept Learning . PhD thesis, Stanford University, 1979.
- O. Montasser, S. Hanneke, and N. Srebro. VC classes are adversarially robustly learnable, but only improperly. In Proceedings of the 32 nd Conference on Learning Theory , 2019.
- E. Mosqueira-Rey, E. Hernández-Pereira, D. Alonso-Ríos, J. Bobes-Bascarán, and Á. Fernández-Leal. Human-in-the-loop machine learning: a state of the art. Artificial Intelligence Review , 56(4): 3005-3054, 2023.

- M. Naghshvar and T. Javidi. Active sequential hypothesis testing. The Annals of Statistics , 41(6): 2703-2738, 2013.
- B. K. Natarajan. On learning sets and functions. Machine Learning , 4:67-97, 1989.
- R. D. Nowak. Generalized binary search. In Proceedings of the 46 th Allerton Conference on Communication, Control, and Computing , 2008.
- R. D. Nowak. The geometry of generalized binary search. IEEE Transactions on Information Theory , 57(12), 2011.
- F. Olsson. A literature survey of active machine learning in the context of natural language processing. 2009.
- L. Paninski. Asymptotic theory of information-theoretic experimental design. Neural Computation , 17(7):1480-1507, 2005.
- C. S. Peirce. A note on the theory of the economy of research. In Report of the Superintendent of the United States Coast Survey Showing the Progress of the Work for the Fiscal Year Ending June 30, 1876 , 1879.
- R. J. Popplestone. An experiment in automatic induction. In Proceedings of the Fifth Annual Machine Intelligence Workshop, Edinburgh , pages 203-215, 1969.
- N. Puchkin and N. Zhivotovskiy. Exponential savings in agnostic active learning through abstention. IEEE Transactions on Information Theory , 68(7):4651-4665, 2022.
- M. Raginsky and A. Rakhlin. Lower bounds for passive and active learning. In Advances in Neural Information Processing Systems 24 , 2011.
- P. Ren, Y. Xiao, X. Chang, P.-Y. Huang, Z. Li, B. B. Gupta, X. Chen, and X. Wang. A survey of deep active learning. ACM Computing Surveys (CSUR) , 54(9):1-40, 2021.
- B. Settles. Active Learning . Synthesis Lectures on Artificial Intelligence and Machine Learning, Morgan &amp; Claypool Publishers, 2012.
- H. S. Seung, M. Opper, and H. Sompolinsky. Query by committee. In Proceedings of the 5 th Annual Workshop on Computational Learning Theory , 1992.
- H. Shayestehmanesh. Active learning under the Bernstein condition for general losses. Master's thesis, University of Victoria, 2020.
- H. Simon. An almost optimal PAC algorithm. In Proceedings of the 28 th Conference on Learning Theory , 2015.
- H. A. Simon and G. Lea. Problem solving and rule induction: A unified view. In Knowledge and Cognition , pages 105-129. Lawrence Erlbaum Associates, 1974.
- R. G. Smith, T. M. Mitchell, R. A. Chestek, and B. G. Buchanan. A model for learning systems. In Proceedings of the 5 th International Joint Conference on Artificial Intelligence , pages 338-343, 1977.
- M. Talagrand. Sharper bounds for gaussian and empirical processes. The Annals of Probability , 22: 28-76, 1994.
- S. Tong and D. Koller. Support vector machine active learning with applications to text classification. Journal of Machine Learning Research , 2(11):45-66, 2001.
- C. Tosh and D. Hsu. Diameter-based interactive structure discovery. In Proceedings of the 23 rd International Conference on Artificial Intelligence and Statistics , 2020.
- A. B. Tsybakov. Optimal aggregation of classifiers in statistical learning. The Annals of Statistics , 32 (1):135-166, 2004.
- G. Turán. Lower bounds for PAC learning with queries. In Proceedings of the 6 th Annual Conference on Computational Learning Theory , 1993.

- L. G. Valiant. A theory of the learnable. Communications of the ACM , 27(11):1134-1142, November 1984.
- A. W. van der Vaart and J. A. Wellner. Weak Convergence and Empirical Processes . Springer, 1996.
- V. Vapnik and A. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability and its Applications , 16(2):264-280, 1971.
- V. Vapnik and A. Chervonenkis. Theory of Pattern Recognition . Nauka, Moscow, 1974.
- M. Vidyasagar. Learning and Generalization with Applications to Neural Networks . Springer-Verlag, 2 nd edition, 2003.
- A. Wald. Sequential Analysis . John Wiley and Sons, New York, 1947.
- L. Wang. Smoothness, disagreement coefficient, and the label complexity of agnostic active learning. Journal of Machine Learning Research , 12(7):2269-2292, 2011.
- Y. Wang and A. Singh. Noise-adaptive margin-based active learning and lower bounds under Tsybakov noise condition. In Proceedings of the 30 th AAAI Conference on Artificial Intelligence , 2016.
- Y. Wiener, S. Hanneke, and R. El-Yaniv. A compression technique for analyzing disagreement-based active learning. Journal of Machine Learning Research , 16(4):713-745, 2015.
- S. Yan, K. Chaudhuri, and T. Javidi. Active learning with logged data. In Proceedings of the 35 th International Conference on Machine Learning , 2018.
- S. Yan, K. Chaudhuri, and T. Javidi. The label complexity of active learning from observational data. In Advances in Neural Information Processing Systems 32 , 2019.
- C. Zhang and K. Chaudhuri. Beyond disagreement-based agnostic active learning. In Advances in Neural Information Processing Systems 27 , 2014.
- T. Zhang. Statistical behavior and consistency of classification methods based on convex risk minimization. The Annals of Statistics , 32(1):56-85, 2004.
- T. Zhang. Mathematical Analysis of Machine Learning Algorithms . Cambridge University Press, 2023.
- T. Zhang and F. Oles. A probability analysis on the value of unlabeled data for classification problems. In International Conference on Machine Learning , 2000.
- N. Zhivotovskiy and S. Hanneke. Localization of VC classes: Beyond local Rademacher complexities. Theoretical Computer Science , 742:27-49, 2018.
- X. Zhu, J. Lafferty, and Z. Ghahramani. Combining active learning and semi-supervised learning using Gaussian fields and harmonic functions. In ICML workshop on the Continuum from Labeled to Unlabeled Data in Machine Learning and Data Mining , 2003.
- Y. Zhu and R. Nowak. Efficient active learning with abstention. In Advances in Neural Information Processing Systems 35 , 2022.

## A Survey of the Theory of Active Learning and Other Related Work

There is at this time quite an extensive literature on the theory of active learning. We refer the interested reader to the surveys of Hanneke (2014), Dasgupta (2011), and the 2019 ICML tutorial of Hanneke and Nowak (2019) for detailed discussions of classic works in this literature. In this section, we present a brief survey of the subject, with particular emphasis on the parts most-closely related to the present work.

## A.1 A Brief Historical Overview

The literature on active learning has a long history, dating back at least to the classical works on experiment design in statistics (Peirce, 1879; Fisher, 1935), wherein the analogous setting to active learning is referred to as sequential design (e.g., Wald, 1947; DeGroot, 1962; Fedorov, 1972; Atkinson and Donev, 1992; Efromovich, 2007; Zhang and Oles, 2000; Paninski, 2005; Lewi, Butera, and Paninski, 2009; Bull, 2013; Naghshvar and Javidi, 2013; Chaudhuri, Kakade, Netrapalli, and Sanghavi, 2015). Active learning has also been an important subject within the machine learning literature from the very beginning (e.g., Popplestone, 1969; Simon and Lea, 1974; Buchanan, 1976; Smith, Mitchell, Chestek, and Buchanan, 1977; Mitchell, 1979). Below we briefly mention some of the background of the subject in the learning theory literature, before giving detailed background of the literature on agnostic active learning.

Membership Queries: In the learning theory literature, the idea of active learning also appeared as a natural variant of the problem of Exact learning with queries . Specifically, in this setting, supposing there is an unknown target concept h ⋆ ∈ C , the objective of the learner is to exactly identify h ⋆ . To achieve this goal, the learner has access to an oracle (who knows h ⋆ ), to which it may pose queries of a given type. The most relevant such queries (to the present work) are membership queries : namely, it may construct any x ∈ X and query for the value h ⋆ ( x ) (in later works in machine learning, this is sometimes known as query synthesis ). Early discussion of this framework and corresponding algorithmic principles appear in the seminal work of Mitchell (1979). General analyses of the number of queries necessary and sufficient to identify h ⋆ (i.e., the query complexity of Exact learning) were developed in the works of Angluin (1987); Hegedüs (1995); Hellerstein, Pillaipakkamnatt, Raghavan, and Wilkins (1996); Nowak (2008, 2011); Hopkins, Kane, Lovett, and Mahajan (2020), and a related average-case analysis was developed by Dasgupta (2004).

Closer to the setting considered in the present work, the idea of learning with membership queries has also been extensively studied in the context of PAC learning in the realizable case. In that setting, the learner observes i.i.d. samples ( X i , Y i ) with unknown distribution P , under the assumption that there exists an unknown target concept h ⋆ ∈ C with er P ( h ⋆ ) = 0 . The learner is additionally permitted to make membership queries for this concept h ⋆ , with the goal of producing a predictor ˆ h having er P ( ˆ h ) ≤ ε with high probability 1 -δ . While most of the literature on PAC learning with membership queries has focused on the benefits of such queries for the computational complexity of learning (e.g., Valiant, 1984; Baum, 1991; Jackson, 1997), the literature also contains several works on the number of samples and queries for learning in this setting (e.g., Eisenberg and Rivest, 1990; Eisenberg, 1992; Seung, Opper, and Sompolinsky, 1992; Turán, 1993; Kulkarni, Mitter, and Tsitsiklis, 1993; Diakonikolas, Kane, and Ma, 2024).

Modern Active Learning with Label Queries: While the early literature on PAC learning with membership queries included several strong positive results (exhibiting advantages in both query complexity and computational complexity compared to learning from i.i.d. samples alone), when researchers implemented these algorithms and tried to use them for practical machine learning with a human labeler as the oracle, they found that the instances x ∈ X queried by the learner often turned out to be rather nonsensical, unnatural, or borderline cases between two labels (e.g., Baum and Lang, 1992). As such, human labelers were unable to provide useful answers to the queries, leading to poor performance of the learning algorithm. To address this issue, researchers turned to studying algorithms whose queries are restricted to only natural instances x ∈ X , which in most works (with a few notable exceptions, e.g., Awasthi, Feldman, and Kanade, 2013) essentially means x in the support of the marginal distribution PX : i.e., the types of examples that might occur naturally in the population. To actualize this restriction, researchers proposed a simple variant of active learning (which has become the standard framework in the literature, and is now essentially synonymous

with the term active learning ), in which there are i.i.d. samples ( X 1 , Y 1 ) , . . . , ( X m , Y m ) from an unknown distribution P , but the learner initially only observes the unlabeled examples X i , and can query to observe individual labels Y i (in a sequential fashion, so that it observes the label Y i of its previous query before selecting the next query X i ′ ) (Cohn, Atlas, and Ladner, 1994; Freund, Seung, Shamir, and Tishby, 1997; Tong and Koller, 2001). Such queries can typically be answered by human experts, being of the same type as used for data annotation in standard supervised machine learning. In this scenario, the unlabeled examples X i are typically assumed to be available in abundance, while obtaining the labels Y i is considered comparably more expensive (relying on the effort of a human expert), so that the primary objective is to minimize the number of label queries needed to achieve a given accuracy of a learned predictor ˆ h . This is the setting studied in the present work.

The theoretical literature on this subject has origins in early works discussing algorithmic principles based on version spaces (Mitchell, 1979; Cohn, Atlas, and Ladner, 1994). Many of the early works providing actual bounds on the query complexity focused on showing improvements over passive learning for special scenarios, such as linear classifiers under distribution assumptions (e.g., Freund, Seung, Shamir, and Tishby, 1997; Dasgupta, Kalai, and Monteleoni, 2005; Har-Peled, Roth, and Zimak, 2007; Balcan, Beygelzimer, and Langford, 2006; Balcan, Broder, and Zhang, 2007; Balcan and Long, 2013; Gonen, Sabato, and Shalev-Shwartz, 2013; Wang and Singh, 2016; Cavallanti, Cesa-Bianchi, and Gentile, 2011; Dekel, Gentile, and Sridharan, 2012). This was followed by a boom of general-case analyses, providing general theories analyzing the query complexity for any concept class (e.g., Dasgupta, 2005; Hanneke, 2007a,b, 2009b,a, 2011, 2012, 2014; Dasgupta, Hsu, and Monteleoni, 2007; Balcan, Hanneke, and Vaughan, 2010; Beygelzimer, Dasgupta, and Langford, 2009; Koltchinskii, 2010; Zhang and Chaudhuri, 2014; El-Yaniv and Wiener, 2012; Wiener, Hanneke, and El-Yaniv, 2015; Hanneke and Yang, 2015; Hanneke, Karbasi, Moran, and Velegkas, 2024a), some of which are discussed in more detail below.

Agnostic PAC Learning: The PAC learning framework has also been extended to allow nonrealizable distributions P , that is, removing the restriction that inf h ∈ C er P ( h ) = 0 . This framework was abstractly formulated in the classic work of Vapnik and Chervonenkis (1974), with interest in the computer science literature initiated by the works of Haussler (1992); Kearns, Schapire, and Sellie (1994). Since such non-realizable distributions P might not allow for predictors ˆ h with er P ( ˆ h ) ≤ ε , the objective in this framework changes to merely achieving a relatively low error rate compared to the best error rate achievable by concepts in the class C . More precisely, we aim to produce a predictor ˆ h which, with probability at least 1 -δ , satisfies er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε . The goal is to achieve this objective, for every distribution P , without any restrictions. This framework is termed agnostic PAC learning , to emphasize that we do not assume any special knowledge of P when designing such a learning algorithm (Kearns, Schapire, and Sellie, 1994).

While an agnostic learning algorithm should achieve this objective for every distribution P , this need not restrict the analysis of such learners to consider only the worst case over all P . In particular, in the present work, we are primarily interested in analyzing the number of queries necessary and sufficient for agnostic active learning, as a function of the best-in-class error rate inf h ∈ C er P ( h ) , known as a first-order query complexity bound. Precisely, as introduced in Section 2, for every ε, δ, β ∈ (0 , 1) , we denote by QC a ( ε, δ ; β, C ) the minimax optimal first-order query complexity: that is, the minimal Q ∈ N for which there exists an active learning algorithm A a such that (for a sufficiently large number m of unlabeled examples), for every distribution P with inf h ∈ C er P ( h ) ≤ β , with probability at least 1 -δ , A a makes at most Q queries and returns a predictor ˆ h satisfying er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε . While, in principle, this definition of QC a ( ε, δ ; β, C ) admits learners which explicitly depend on knowledge of β , we will find that the optimal query complexity is achievable (up to constant factors and lower-order terms) simultaneously for all β by an active learner which does not require knowledge of β . Such a learner is said to be adaptive to β . In particular, such a learner is therefore an agnostic PAC learner, and the β restriction only enters in its analysis.

The Passive Learning Baseline: Since the predictor ˆ h produced by an active learning algorithm is based on its queried subset of a given set of i.i.d. examples ( X i , Y i ) , the natural quantity for comparison is the number of i.i.d. labeled examples necessary to obtain the same accuracy: i.e., the sample complexity of standard supervised learning, which in this literature is termed passive

learning . 5 Recall from Section 2 that we denote by M p ( ε, δ ; β, C ) the minimax optimal sample complexity of passive learning: i.e., the minimal n such that there exists a passive learning algorithm A p that, for every P with inf h ∈ C er P ( h ) ≤ β , for S ∼ P n and ˆ h n = A p ( S ) , guarantees with probability at least 1 -δ that er P ( ˆ h n ) ≤ inf h ∈ C er P ( h ) + ε . Since we can always design an active learner that simply queries the first n examples and runs a passive learner A p , we clearly always have QC a ( ε, δ ; β, C ) ≤ M p ( ε, δ ; β, C ) . Thus, the main question of interest is whether QC a ( ε, δ ; β, C ) is strictly smaller than M p ( ε, δ ; β, C ) , and if so, by how much. Lower bounds of Vapnik and Chervonenkis (1974); Devroye and Lugosi (1995) establish that

<!-- formula-not-decoded -->

recalling that d denotes the VC dimension of C (Vapnik and Chervonenkis, 1971; see Definition 4 of Appendix B). The classic analysis of Vapnik and Chervonenkis (1974) further established this lower bound can nearly be achieved by the simple method of empirical risk minimization , i.e., ˆ h n = argmin h ∈ C ˆ er S ( h ) , providing an upper bound M p ( ε, δ ; β, C ) = O ( β ε 2 ( d log ( 1 ε ) +log ( 1 δ )) + 1 ε ( d log ( 1 ε ) +log ( 1 δ )) ) . This has since been refined in various ways, such as via localized chaining arguments (e.g., Giné and Koltchinskii, 2006). Most recently, Hanneke, Larsen, and Zhivotovskiy (2024b) proved an upper bound M p ( ε, δ ; β, C ) = O ( β ε 2 ( d +log ( 1 δ )) ) + ˜ O ( 1 ε ( d +log ( 1 δ ))) , matching the lower bound (4) up to log factors in the lower-order term (the problem of removing these remaining log factors remains open at this time). The algorithm achieving this is improper , meaning its returned ˆ h n is not necessarily an element of C , and Hanneke, Larsen, and Zhivotovskiy (2024b) in fact show that for some concept classes C improperness is necessary to match the lower bound (4) in the lead term, as all proper learners incur an extra log ( 1 β ) factor. In the special case of β = 0 (the realizable case ), the lower bound (4) was shown to be achievable by Hanneke (2016a) (also necessarily via an improper learner), so that M p ( ε, δ ; 0 , C ) = Θ ( 1 ε ( d +log ( 1 δ ))) . The lower bound (4) will therefore serve as a suitable baseline for gauging whether the query complexity QC a ( ε, δ ; β, C ) of active learning is smaller than the sample complexity M p ( ε, δ ; β, C ) of passive learning.

The Need for Distribution-dependent Analysis in Realizable Active Learning: Much of the early work on active learning focused on the realizable case , i.e., the special case β = 0 . In this special case, it was quickly observed by Dasgupta (2004, 2005) that there are some concept classes (e.g., thresholds 1 [ a, ∞ ) on R ) where active learning offers strong improvements over passive learning, and other concept classes (e.g., intervals 1 [ a,b ] on R ) where the (distribution-free) minimax query complexity QC a ( ε, δ ; 0 , C ) offers no significant improvements over passive learning. The essential advantage in the former case arises from a kind of 'binary search' behavior, where the 'uncertainty' is being sequentially reduced by a careful choice of queries. In contrast, the essential challenge in the latter case is the problem of 'searching in the dark' for a small-but-important region: e.g., the optimal concept is 1 for a single unknown x i among some x 1 , . . . , x 1 /ε , and PX = Uniform( { x 1 , . . . , x 1 /ε } ) . It turns out this hard scenario is embedded in many concept classes of interest, a fact which was formalized and quantified by Hanneke and Yang (2015) in the star number complexity measure (Definition 2) discussed below. Such concept classes C naturally exhibit a lower bound QC a ( ε, δ ; 0 , C ) = Ω ( 1 ε ) . Even worse, consider a scenarios where the optimal concept can be 1 for any d points x i among x 1 , . . . , x d / (2 ε ) , and PX = Uniform( { x 1 , . . . , x d / (2 ε ) } ) . Hanneke and Yang (2015) show this scenario has QC a ( ε, δ ; 0 , C ) = Ω ( d ε ) , so that QC a ( ε, δ ; 0 , C ) has the same joint dependence on ( d , ε ) as passive learning M p ( ε, δ ; 0 , C ) = Θ ( 1 ε ( d +log ( 1 δ ))) , only offering

5 Since the active learner also has access to the remaining (unqueried) i.i.d. unlabeled examples X i , it is also natural to compare to the related framework of semi-supervised learning, in which a learner has access to some number n of i.i.d. labeled examples with distribution P and additionally some larger number m of i.i.d. unlabeled examples with distribution PX (Chapelle, Scholkopf, and Zien, 2006). While, under some favorable conditions, the labeled sample complexity n of semi-supervised learning can be smaller than that of strictly-supervised passive learning (see Balcan and Blum, 2010), the lower bounds on the (distribution-free) sample complexity of passive learning discussed in this work remain valid for the labeled sample complexity of semi-supervised learning (regardless of how many unlabeled examples are available), so that for the purpose of comparison in the present work, the distinction between supervised and semi-supervised passive learning as a baseline is not important, and we will simply compare to passive supervised learning for simplicity.

an improvement in the dependence on δ (where δ thus only affects the unlabeled sample complexity). Moreover, as noted by Hanneke (2014), similar scenarios 6 are embedded (with d = Θ(VC( C )) ) in most concept classes C of interest in learning theory (e.g., linear classifiers in R 3 d , and axis-aligned rectangles in R 2 d ), so that such classes also exhibit no significant improvements over passive learning in their (distribution-free minimax) realizable-case query complexity QC a ( ε, δ ; 0 , C ) .

Motivated by the fact that such hard scenarios are embedded in many concept classes of interest, Dasgupta (2005) suggested that, for such concept classes, the only viable way to understand the potential advantages of active learning is to focus on distribution-dependent analysis, toward identifying special scenarios where active learning algorithms offer improvements over passive learning, by formulating appropriate assumptions on the distribution P . This narrative quickly caught on in the literature, with a variety of distribution-dependent analyses and general P -dependent complexity measures being proposed to analyze certain active learning strategies under various restrictions on the realizable distribution P (Dasgupta, 2005; Hanneke, 2007a,b, 2014; Balcan, Broder, and Zhang, 2007; Balcan and Long, 2013; Zhang and Chaudhuri, 2014; El-Yaniv and Wiener, 2012; Wiener, Hanneke, and El-Yaniv, 2015). We discuss several of these in detail below.

Active Learning in the Non-realizable Case: Given the above narrative, when approaching the analysis of active learning in the non-realizable case ( β &gt; 0 ), it might at first seem perfectly reasonable to expect that for many concept classes C the query complexity QC a ( ε, δ ; β, C ) might not be much smaller than the sample complexity of passive learning M p ( ε, δ ; β, C ) . As such, the literature on agnostic active learning has largely focused on extending the distribution-dependent analyses from the realizable case to the agnostic setting (e.g., Balcan, Beygelzimer, and Langford, 2006, 2009; Hanneke, 2007b,a, 2009b, 2011, 2014; Balcan and Hanneke, 2012; Zhang and Chaudhuri, 2014; Wiener, Hanneke, and El-Yaniv, 2015). These upper bounds conformed to the accepted narrative, in that they offer improvements under some distributions, but for most classes C , in the worst case over distributions P (with inf h ∈ C er P ( h ) ≤ β ) they essentially revert to the passive sample complexity M p ( ε, δ ; β, C ) .

However, since this narrative was born from analysis of the realizable case , there was no actual reason to believe it should remain valid in the non-realizable case. In particular, there remained an intriguing possibility that there could perhaps be other advantages of active learning specific to the non-realizable case : that is, beyond the 'binary search' type advantages known from the realizable case (as captured by the complexity measures proposed for realizable-case analysis). Some hints that such additional advantages may exist appear in the works of Efromovich (2007) and Hanneke and Yang (2015) studying certain special scenarios (noise models more-restrictive than the agnostic setting), which found that active learning can also be useful for adaptively identifying noisy regions (i.e., regions of X 's where P ( Y | X ) is close to 1 2 ) and allocating queries appropriately to compensate for this noisiness without wasting excessive queries in less-noisy regions (as passive learning would). This additional advantage, specific to the non-realizable case, offered quantitative advantages over passive learning under the specific conditions studied in those works (e.g., Hanneke and Yang, 2015 showed improvements in query complexity under certain noise models, namely Tsybakov noise and Benign noise, for all classes C , including those with the 'searching in the dark' scenario embedded in them). However, these works left open the question of whether such advantages can be observed also in the more-challenging agnostic setting. The extension to the agnostic case is not at all clear, since in this setting (unlike the special noise models in the works above) the source of non-realizability is not only the noisiness of the P ( Y | X ) label distribution, but also model misspecification : i.e., it is possible to have β &gt; 0 even when P ( Y | X ) ∈ { 0 , 1 } , if the Bayes classifier is not in C , in which

6 The hard scenario embedded in these classes is slightly more structured. Namely, for any d ∈ N and infinite X , partition X into disjoint infinite subsets X 1 , . . . , X d , and define C d = { h : ∀ i ≤ d , ∑ x ∈X i h ( x ) ≤ 1 } ,

which has VC( C d ) = d . For such classes C d , Hanneke and Yang (2015) show that QC a ( ε, δ ; 0 , C d ) = Ω ( d ε ) (as an aside, it is a straightforward exercise to show a matching upper bound for this class). For homogeneous linear classifiers in R 3 d , we can construct the X i sets as circles in disjoint subspaces: i.e., X i is all ( z 1 , . . . , z 3 d ) ∈ R 3 d s.t. all coordinates z j = 0 except z 2 3( i -1)+1 + z 2 3( i -1)+2 = 1 and z 3( i -1)+3 = 1 (to allow for non-homogeneous linear classifiers on these circles, each controlled using 3 distinct weights in the 3 d -dimensional linear classifier); the classifiers with boundary tangent to these circles witness the d singleton problems in C d . For axis-aligned rectangles in R 2 d , we can construct the X i sets as diagonal lines in disjoint 2 -dimensional subspaces: i.e., X i is all ( z 1 , . . . , z 2 d ) ∈ R 2 d with all z j = 0 except z 2( i -1)+1 ∈ [0 , 1] and z 2( i -1)+2 = 1 -z 2( i -1)+1 ; as each X i can be classified by a 2 -dimensional rectangle based on 2 distinct coordinates, the classifiers with a single corner intersecting each of these diagonal lines then witness the d singleton problems in C d .

case the idea of adapting to 'noisiness' of labels is no longer a useful framing of the problem. (The present work finds the appropriate re-framing of this capability, replacing the notion of 'noisiness' by the variance of the excess-error estimation problem, and identifies algorithmic principles for isolating such high-variance regions, constituting the A VID principle).

Quantifying the Query Complexity (Realizable Case): As discussed above, motivated by negative results for distribution-free analysis in the realizable case, numerous works have studied the query complexity of active learning under restrictions on the distribution P . These distribution-dependent analyses are often expressed in terms of abstract P -dependent complexity measures intended to capture favorable conditions for active learning, compared to passive learning. Quantitatively, in the realizable case ( β = 0 ), such P -dependent query complexity bounds are typically expressed in the form c P ( ε ) · d · polylog ( 1 εδ ) for some P -dependent complexity measure c P ( ε ) . Examples of such complexity measures include the splitting index (Dasgupta, 2005), disagreement coefficient (Hanneke, 2007b, 2009b, 2011), empirical extended teaching dimension (Hanneke, 2007a), and a subregion variant of the disagreement coefficient (Zhang and Chaudhuri, 2014), among others (e.g., El-Yaniv and Wiener, 2012; Hanneke, 2012, 2014; Wiener, Hanneke, and El-Yaniv, 2015; Hanneke and Yang, 2015). Some of these were accompanied by related minimax lower bounds holding for any fixed PX marginal distribution (Dasgupta, 2005; Hanneke, 2007a; Balcan and Hanneke, 2012). We discuss several of these P -dependent analyses and c P ( ε ) complexity measures in detail in Appendix A.2.

The works of Hanneke and Yang (2015); Hanneke (2016b, 2024) later showed that all of these proposed complexity measures c P ( ε ) have worst-case values (i.e., sup P c P ( ε ) over realizable P ) precisely equal a complexity measure s therein termed the star number of C (Definition 2), a quantity which abstractly formalizes and quantifies the extent to which the aforementioned 'searching in the dark' scenario is embedded in a given concept class C . Thus, the star number unifies all of these complexity measures in the case of distribution-free analysis. Moreover, Hanneke and Yang (2015) also show the star number sharply characterizes the optimal distribution-free query complexity in the realizable case: namely,

<!-- formula-not-decoded -->

(see Hanneke and Yang, 2015, for more-detailed bounds). Hanneke and Yang (2015) also show the upper and lower bounds in (5) represent a nearly-sharp dependence on ( d , s ) : that is, there exist concept classes C d , s and C d , s of any given VC dimension d and star number s ≥ d for which QC a ( ε, δ ; 0 , C d , s ) = Θ ( s ∧ ( 1 ε + d )) and QC a ( ε, δ ; 0 , C d , s ) = Θ ( s ∧ d ε ) . Thus, the bounds in (5) are essentially unimprovable (up to a log factor) without introducing additional complexity measures for the class C . In particular, this also means any upper bound on QC a ( ε, δ ; 0 , C ) depending on C only via d and s can be no smaller than Ω ( s ∧ d ε ) .

The bounds in (5) imply that QC a ( ε, δ ; 0 , C ) admits an improved dependence on ε compared to M p ( ε, δ ; 0 , C ) = Θ ( 1 ε ( d +log ( 1 δ ))) if and only if s &lt; ∞ . While there exist some interesting concept classes C with finite star number (e.g., threshold classifiers on R , decision stumps on R p ), it turns out most concept classes of interest in learning theory have infinite (or very large) star number (e.g., s = ∞ for linear classifiers on R p , p ≥ 2 ). Thus, the general bounds in (5) quantitatively reflect the fact (already observed in several cases by Dasgupta, 2004, 2005) that we should typically not expect any significant benefits of active learning in the realizable case to be reflected in the (distribution-free minimax) query complexity QC a ( ε, δ ; 0 , C ) . Moreover, as mentioned above, the concept classes C d , s with s = ∞ witnessing near-sharpness of the upper bound in (5) are also embedded in many concept classes of interest in learning theory (see footnote 6), further strengthening the lower bound to Ω ( d ε ) for such classes, thus matching the sample complexity M p ( ε, δ ; 0 , C ) of passive learning in all dependencies (except δ ).

Quantifying the Query Complexity (Agnostic Case): Turning to the agnostic case ( β ≥ 0 ), Kääriäinen (2006) established a general lower bound QC a ( ε, δ ; β, C ) = Ω ( β 2 ε 2 log ( 1 δ ) ) , later strengthened by Beygelzimer, Dasgupta, and Langford (2009) to

<!-- formula-not-decoded -->

Comparing this to the sample complexity of passive learning (discussed above), namely M p ( ε, δ ; β, C ) = Θ ( β ε 2 ( d +log ( 1 δ )) ) + ˜ Θ ( 1 ε ( d +log ( 1 δ ))) , we see that in the regime β ≫ √ ε , the best improvement we can hope for from active learning would be to replace the factor β with β 2 : i.e., squaring the dependence on the best-in-class error rate inf h ∈ C er P ( h ) . In the regime β ≲ √ ε , the realizable-case lower bound from (5) becomes relevant (the realizable case being a special case, since clearly inf h ∈ C er P = 0 satisfies the condition inf h ∈ C er P ( h ) ≤ β ), which may be thought of as a lower-order additive term.

The work of Balcan, Beygelzimer, and Langford (2006) initiated the study of upper bounds on the query complexity in the agnostic case, showing that the lower bound (6) can be matched in the special cases of threshold classifiers (concepts 1 [ a, ∞ ) on X = R ), and (in the regime β ≲ ε/ √ d ) matched up to a factor d for homogeneous linear classifiers under PX uniform in an origin-centered ball, extending these well-known examples from the realizable case. This analysis was generalized to all concept classes by Hanneke (2007b), expressing a query complexity bound of the form ˜ O ( c P ( β + ε ) d ( β 2 ε 2 +1 )) , where the factor c P ( β + ε ) is based on a P -dependent complexity measure θ P ( β + ε ) therein termed the disagreement coefficient (Definition 25). In particular, the bound of Hanneke (2007b) matches the lower bound of Kääriäinen (2006); Beygelzimer, Dasgupta, and Langford (2009) up to logs only when θ P ( β + ε ) = ˜ O (1) . The latter holds for threshold classifiers, and for other classes under restrictions on P , but in many other cases θ P ( β + ε ) can be as large as 1 β + ε due to the 'searching in the dark' problem discussed above; in such cases, the query complexity upper bound of Hanneke (2007b) is no smaller than the sample complexity of passive learning M p ( ε, δ ; β, C ) . Numerous later works (some discussed in detail below) discovered refinements and alternative P -dependent complexity measures used to express upper bounds on the query complexity (Hanneke, 2007a; Dasgupta, Hsu, and Monteleoni, 2007; Hanneke, 2009b, 2011, 2014; Zhang and Chaudhuri, 2014; Wiener, Hanneke, and El-Yaniv, 2015). However, like the bound of Hanneke (2007b), all of these results establish query complexity upper bounds of the form ˜ O ( c P ( β + ε ) d ( β 2 ε 2 +1 )) for some P -dependent complexity measure c P ( β + ε ) , all of which have the property that, in the 'searching in the dark' type scenarios discussed above, the value c P ( β + ε ) ≥ 1 β + ε , so that in such scenarios these upper bounds are all no smaller than the sample complexity of passive learning M p ( ε, δ ; β, C ) .

As in the realizable case, these various analysis were later all unified under worst-case analysis over P by the star number in the work of Hanneke and Yang (2015). Indeed, these complexity measures c P ( β + ε ) are in fact the same family of complexity measures alluded to above for the realizable case. As such, by the aforementioned result of Hanneke and Yang (2015), they all satisfy sup P c P ( β + ε ) = s ∧ 1 β + ε . Thus, the upper bounds established by these works, all being of the form ˜ O ( c P ( β + ε ) d ( β 2 ε 2 +1 )) , unify to a single upper bound of the form ˜ O (( s ∧ 1 β + ε ) d ( β 2 ε 2 +1 )) in the worst case over distributions P (satisfying inf h ∈ C er P ( h ) ≤ β ). In particular, this also means they all fail to imply any improvements over the sample complexity of passive learning M p ( ε, δ ; β, C ) in the worst case over such distributions P , for any concept class C with s = ∞ . 7 This is significant, since (as discussed above) most commonly-studied concept classes have s = ∞ , including, for instance, linear classifiers in R p , p ≥ 2 . On the other hand, the lower bound (6), of the form Ω ( β 2 ε 2 ( d +log ( 1 δ )) ) , has no such factor s ∧ 1 β + ε . The natural question is therefore which of these can be strengthened: the upper bound or lower bound.

The above gap has a qualitative significance. If the lower bound could be strengthened to match the upper bound, it would mean that (as in the realizable case) there are classes where active learning offers no advantage in its minimax query complexity compared to passive learning. On the other hand, if the upper bound can be strengthened to match the lower bound, it would mean that (unlike the realizable case) the query complexity of active learning is always smaller than the sample complexity of passive learning in the agnostic setting: i.e., for every concept class. The problem of resolving this gap has remained open until now. In the present work, we completely resolve this question,

7 One can also show that this is not merely a result of loose analysis. The algorithms (prior to the present work) can be made to behave similarly to passive learners (meaning they query almost indiscriminately) in some scenarios constructed on large star sets, resulting in a number of queries β ε 2 .

strengthening the upper bound to match the lower bound (6), thereby establishing that active learning is always better than passive learning in the agnostic case, providing an improvement by squaring the dependence on the best-in-class error rate in the leading term: i.e., replacing β with β 2 . Establishing this upper bound requires a new principle for active learning, specific to the agnostic setting, which we develop in this work (termed AVID , for Adaptive Variance Isolation by Disagreements ).

Before proceeding with the presentation of our results, we first provide, in the next subsection, a detailed survey of several of the prior works mentioned in the above brief historical summary.

## A.2 Detailed Description of Relevant Techniques in the Prior Literature

In this subsection, we provide further details of relevant works in the literature. Due to the vastness and diversity of the literature on the theory of active learning, we will not provide an exhaustive survey here, instead focusing on the techniques and results most-relevant to the present work.

## A.2.1 Disagreement-based Active Learning

̸

By-far the most well-studied technique in the literature on the theory of active learning is disagreementbased active learning. A disagreement-based active learner is given as input the sequence X 1 , X 2 , . . . , X m of unlabeled examples. It maintains (either explicitly or implicitly) a set V ⊆ C of surviving concepts (known as a version space ), with a guarantee that the best-in-class concept 8 h ⋆ is always retained in V . To choose its query points, it finds the next unlabeled example X i in the sequence for which ∃ f, g ∈ V with f ( X i ) = g ( X i ) , and queries for the label Y i : or more succinctly, it queries the next X i ∈ DIS( V ) , where

̸

<!-- formula-not-decoded -->

denotes the region of disagreement of V . It then updates the set V of surviving concepts based on this new information (or, in some variants, it performs this update only periodically, rather than after every query). This is abstractly summarized in the following outline (where Step 4 can be instantiated in various ways, as discussed below).

```
Algorithm Outline: Disagreement-based Active Learning Input: Unlabeled data X 1 , . . . , X m Output: Classifier ˆ h 0. Initialize V = C 1. For i = 1 , 2 , . . . , m 2. If X i ∈ DIS( V ) 3. Query for label Y i 4. Update V 5. Return any ˆ h ∈ V
```

̸

The idea is that, if we seek to return a concept ˆ h ∈ V with small er P ( ˆ h ) -er P ( h ⋆ ) , then for any X i / ∈ DIS( V ) , since all surviving concepts agree on the classification of X i , the label Y i would provide no information that would help with this goal, so we do not bother querying for this label: that is, such a Y i cannot help to estimate the relative performances er P ( f ) -er P ( g ) of concepts f, g ∈ V , since regardless of Y i , we have 1 [ f ( X i ) = Y i ] -1 [ g ( X i ) = Y i ] = 0 . In contrast, the next X i ∈ DIS( V ) in the sequence is a random sample from PX ( ·| DIS( V )) , so that for any f, g ∈ V , 1 [ f ( X i ) = Y i ] -1 [ g ( X i ) = Y i ] is an unbiased estimate of the difference of error rates under the conditional distribution P ( ·| DIS( V ) × { 0 , 1 } ) , which (again since f, g agree outside DIS( V ) ) is proportional to er P ( f ) -er P ( g ) . By reasoning about uniform concentration of these estimates, we can define an update rule for V in Step 4 that never removes the best-in-class concept h ⋆ while pruning sub-optimal concepts from V (where the resolution of this pruning improves as i grows).

̸

̸

In the realizable case , since we always have h ⋆ ( X i ) = Y i , the updates to V in Step 4 can simply remove all concepts incorrect on a queried example ( X i , Y i ) : that is, V ←{ h ∈ V : h ( X i ) = Y i } (called the version space ), which always retains h ⋆ ∈ V . The algorithmic principle of disagreementbased queries, and corresponding reasoning about correctness and potential advantages, was already

8 The theory easily generalizes to cases where the infimum inf h ∈ C er P ( h ) is not attained, in which case define h ⋆ ∈ C to have er P ( h ⋆ ) sufficiently close to the infimum, e.g., as in (10) below.

̸

identified in the early work of Mitchell (1979) (for the membership queries model). 9 The precise form expressed above (sequentially checking unlabeled examples for disagreements) was first explicitly studied by Cohn, Atlas, and Ladner (1994), and in their honor, this realizable-case technique is referred to as CAL in the literature. As for its theoretical analysis, the original works of Mitchell (1979); Cohn, Atlas, and Ladner (1994) include the observation that h ⋆ is retained in V , and Cohn, Atlas, and Ladner (1994) additionally include some discussion of generalization. However, the formal analysis of the query complexity of this technique in the PAC framework only began with the later work of Balcan, Beygelzimer, and Langford (2006) (bounding the query complexity for some specific concept classes), and the general analysis of the technique (applicable to any concept class) began with the works of Hanneke (2007b, 2009b, 2011); Dasgupta, Hsu, and Monteleoni (2007).

The idea of disagreement-based active learning was first extended to the agnostic setting ( β ≥ 0 ) by Balcan, Beygelzimer, and Langford (2006), with an instantiation of the above outline they called the A 2 algorithm (for Agnostic Active ). The main idea in A 2 is to instantiate the update to V in Step 4 using uniform concentration inequalities. In their original version, they specifically define UB( h ) and LB( h ) as high-probability uniform upper and lower bounds on er P ( ·| DIS( V ) ×{ 0 , 1 } ) ( h ) based on the queries from DIS( V ) since the last update to V (where they only update V periodically in their algorithm, so that the queried examples since the last update are i.i.d. samples from P ( ·| DIS( V ) × { 0 , 1 } ) ). They then define the update as V ←{ h ∈ V : LB( h ) ≤ min h ′ ∈ V UB( h ′ ) } . The idea is that they wish to remove a concept h from V if there is another concept h ′ ∈ V whose upper bound UB( h ′ ) on its error rate is smaller than the lower bound LB( h ) on the error rate of h . In particular, since h, h ′ agree on all x / ∈ DIS( V ) , we have

<!-- formula-not-decoded -->

and hence a concept h can be removed from V only if er P ( h ) -er P ( h ′ ) &gt; 0 for some h ′ ∈ V , meaning that h is verifiably suboptimal, guaranteeing that the algorithm always retains h ⋆ ∈ V . Conversely, by querying the examples in DIS( V ) , we are improving the concentration inequalities UB( h ) , LB( h ) , so that suboptimal concepts are removed from V , which has two benefits: (1) we are converging to a set of relatively low-error concepts (important for the final error guarantee), and (2) by reducing V we are potentially also reducing DIS( V ) , focusing the algorithm's queries to more-informative samples and decreasing the query complexity.

The original analysis of Balcan, Beygelzimer, and Langford (2006) included the above correctness guarantee (i.e., the algorithm maintains h ⋆ ∈ V ), along with a general guarantee that the A 2 algorithm returns an ˆ h with er P ( ˆ h ) ≤ er P ( h ⋆ ) + ε , with a number of queries never significantly worse than that of passive learning. Also, as a sort of proof of concept illustrating the potential benefits of A 2 in a simple example, they also quantified the query complexity advantages in the special case of threshold classifiers (concepts 1 [ a, ∞ ) on X = R ), showing a bound ˜ O ( β 2 ε 2 +1 ) for that class (matching the lower bound of Kääriäinen, 2006). They also studied the special case of learning homogeneous linear classifiers under a uniform distribution in an origin-centered ball in R d , focusing on the regime β ≲ ε/ √ d , for which they showed the query complexity is ˜ O ( d 2 log ( 1 ε ) log ( 1 δ )) .

̸

The first general analyses (i.e., applicable to any concept class) of the query complexity of active learning in the agnostic setting were given in the works of Hanneke (2007b,a). In particular, Hanneke (2007b) analyzed the A 2 disagreement-based active learning algorithm, providing a general query complexity bound expressed in terms of a new complexity measure therein termed the disagreement coefficient . Specifically, for r &gt; 0 , denoting by B P X ( h ⋆ , r ) = { h ∈ C : PX ( x : h ( x ) = h ⋆ ( x )) ≤ r } the h ⋆ -centered r -ball (under L 1 ( PX ) ), the disagreement coefficient is defined as

<!-- formula-not-decoded -->

The intuitive interpretation of the relevance of this quantity is that, as the algorithm progresses, the set V of surviving concepts will become closer and closer to h ⋆ (up until a distance O ( β + ε ) ), so that the probability of querying decreases as PX (DIS( V )) ≤ PX (DIS(B P X ( h ⋆ , r ))) for an appropriate r decreasing as the number of queries grows.

Hanneke (2007b) proves that, for any C and P , for β = er P ( h ⋆ ) , the A 2 algorithm succeeds after a number of queries ˜ O ( θ P ( β + ε ) 2 d ( β 2 ε 2 +1 )) . This matches the lower bound (6) of Kääriäinen

9 Mitchell (1979) also discusses some reasonable extensions of version spaces to the non-realizable case.

(2006); Beygelzimer, Dasgupta, and Langford (2009) up to logs whenever θ P ( β + ε ) = ˜ O (1) . In particular, Hanneke (2007b) bounds θ P ( β + ε ) for a number of scenarios ( C , P ) , including showing that this general query complexity upper bound recovers the examples of Balcan, Beygelzimer, and Langford (2006): θ P ( β + ε ) ≤ 2 for threshold classifiers, and θ P ( β + ε ) = O ( √ d ) for homogeneous linear classifiers under a uniform distribution on an origin-centered sphere (thus also removing the constraints on β, ε from the result of Balcan, Beygelzimer, and Langford, 2006). However, Hanneke (2007b) also found θ P ( β + ε ) can sometimes be as large as 1 β + ε , particularly for the 'searching in the dark' scenarios discussed above, in which case the query complexity bound is no smaller than the sample complexity of passive learning.

Subsequently, Dasgupta, Hsu, and Monteleoni (2007) refined the dependence on θ P ( β + ε ) in this bound (analyzing a different disagreement-based algorithm), replacing θ P ( β + ε ) 2 with θ P ( β + ε ) . Their technique also identifies a principle enabling more-practical implementation of disagreementbased active learning, expressing the algorithm as a reduction to empirical risk minimization (ERM). In general (even with A 2 ), we can always maintain the set V implicitly (i.e., without storing this large object V explicitly), by simply maintaining the set of constraints that define it, which then enable us to perform the various operations (e.g., checking whether X i ∈ DIS( V ) , or computing min h ′ ∈ V UB( h ) ) via constraint satisfaction or constrained optimization problems. The algorithm of Dasgupta, Hsu, and Monteleoni (2007) takes this a step further, expressing such operations as (effectively) unconstrained optimization problems, or in other words, calls to an ERM subroutine (i.e., an algorithm which returns a concept in C minimizing the number of mistakes on any given labeled data set). Specifically, they store two labeled data sets Q i , L i , where Q i are the queried examples so far (up to round i ) and L i are the unqueried examples so far (with inferred labels). On round i , they consider the concepts h 1 , h 0 ∈ C of minimal ˆ er Q i -1 ( h ) subject to ˆ er L i -1 ( h ) = 0 and h 1 ( X i ) = 1 , h 0 ( X i ) = 0 , if they exist (noting that such concepts can each be obtained by a single call to an ERM oracle with appropriately high weight, or repetition, of the L i -1 and ( X i , y ) examples). If ˆ er Q i -1 ( h 1 ) and ˆ er Q i -1 ( h 0 ) are of similar sizes, they query for Y i and add it to Q i -1 to get Q i (letting L i = L i -1 ), and otherwise they take an inferred label ˆ y i = argmin y ˆ er Q i -1 ( h y ) and add ( X i , ˆ y i ) to L i -1 to get L i (letting Q i = Q i -1 ). Note that this is equivalent to maintaining a set V of all concepts h ∈ C having ˆ er L i -1 ( h ) = 0 and ˆ er Q i -1 ( h ) of similar size to min h ′ ˆ er Q i -1 ( h ′ ) among all h ′ ∈ C with ˆ er L i -1 ( h ′ ) = 0 , and querying X i iff X i ∈ DIS( V ) . Thus, this algorithm can also be viewed as a disagreement-based active learner (and this connection is made explicit in the analysis of Dasgupta, Hsu, and Monteleoni, 2007). (Notably, subsequent works of Beygelzimer, Hsu, Langford, and Zhang, 2010; Hsu, 2010 even further simplified this technique by dropping the L i -1 constraints, obtaining similar query complexity bounds).

The specific quantification of the 'similar sizes' criterion for the difference of empirical error rates comes from uniform Bernstein-style concentration inequalities (related to the uniform Bernstein inequality stated in Lemma 7 of Appendix D below). In particular, letting S i -1 = { ( X j , Y j ) : j &lt; i } , among all pairs of concepts h, h ′ ∈ C with ˆ er L i -1 ( h ) = ˆ er L i -1 ( h ′ ) = 0 , since they agree on examples in L i -1 , we have

<!-- formula-not-decoded -->

̸

so that we can make use of concentration inequalities for estimating differences of error rates from the i.i.d. data set S i -1 (e.g., as in Lemma 7). Reasoning inductively, if ˆ er L i -1 ( h ⋆ ) = 0 , such a concentration inequality guarantees that among all h ∈ C with ˆ er L i -1 ( h ) = 0 , ˆ er Q i -1 ( h ⋆ ) can never be too much larger than ˆ er Q i -1 ( h ) , so that if the algorithm does not query X i (meaning ˆ er Q i -1 ( h 1 -ˆ y i ) is much larger than ˆ er Q i -1 ( h ˆ y i ) ), the corresponding label ˆ y i must be h ⋆ ( X i ) , so that adding ( X i , ˆ y i ) to L i retains that ˆ er L i ( h ⋆ ) = 0 . Conversely, if ˆ er Q i -1 ( h 0 ) and ˆ er Q i -1 ( h 1 ) are of similar sizes, then the algorithm has effectively verified that there exist concepts h, h ′ ∈ C with er P ( h ) and er P ( h ′ ) rather close to er P ( h ⋆ ) which nevertheless disagree on X i ( h ( X i ) = h ′ ( X i ) ), and querying for Y i and adding ( X i , Y i ) to Q i then strengthens the concentration of the ˆ er Q i estimates, to help further distinguish among such small-error concepts in subsequent rounds.

Dasgupta, Hsu, and Monteleoni (2007) analyzed the query complexity of this algorithm, showing that it guarantees er P ( ˆ h ) ≤ er P ( h ⋆ )+ ε after a number of queries ˜ O ( θ P ( β + ε ) d ( β 2 ε 2 +1 )) . Compared to the original analysis of Hanneke (2007b), this improves the bound in its dependence on θ P ( β + ε ) , reducing from quadratic θ P ( β + ε ) 2 to linear θ P ( β + ε ) . Again, the conclusion is that the algorithm's query complexity matches the lower bound (6) of Kääriäinen (2006); Beygelzimer, Dasgupta, and

Langford (2009) up to logs whenever θ P ( β + ε ) = ˜ O (1) ; however, again, in scenarios ( C , P ) with θ P ( β + ε ) = Ω ( 1 β + ε ) , the bound offers no improvements over the sample complexity of passive learning.

The above techniques, and corresponding analysis in terms of the disagreement coefficient, seeded a vast literature, with many variations on the technique, analysis, and complexity measures, and many examples of scenarios ( C , P ) for which θ P ( β + ε ) can be favorably bounded. This branch of the literature is collectively referred to as disagreement-based active learning (see e.g., the works of Hanneke, 2009b, 2011, 2012, 2014, 2016b; Balcan, Hanneke, and Vaughan, 2010; Hsu, 2010; El-Yaniv and Wiener, 2012; Friedman, 2009; Mahalanabis, 2011; Koltchinskii, 2010; Wang, 2011; Beygelzimer, Dasgupta, and Langford, 2009; Beygelzimer, Hsu, Langford, and Zhang, 2010; Raginsky and Rakhlin, 2011; Ailon, Begleiter, and Ezra, 2014; Huang, Agarwal, Hsu, Langford, and Schapire, 2015; Wiener, Hanneke, and El-Yaniv, 2015; Hanneke and Yang, 2010, 2015, 2019; Yan, Chaudhuri, and Javidi, 2018, 2019; Gelbhart and El-Yaniv, 2019; Cortes, DeSalvo, Gentile, Mohri, and Zhang, 2019a; Cortes, DeSalvo, Gentile, Mohri, and Zhang, 2019b,c, 2020; DeSalvo, Gentile, and Thune, 2021; Shayestehmanesh, 2020; Puchkin and Zhivotovskiy, 2022). A detailed summary of this line of work is presented in the survey of Hanneke (2014).

In the context of distribution-free analysis, Hanneke and Yang (2015) showed that sup P θ P ( β + ε ) = s ∧ 1 β + ε , where s is the star number of C (Definition 2), and where the sup is over realizable distributions P (so that, in particular, they satisfy the condition er P ( h ⋆ ) ≤ β ). Thus, in terms of their implications for the distribution-free query complexity QC a ( ε, δ ; β, C ) , these P -dependent analyses of disagreement-based active learning simplify to a bound of the form QC a ( ε, δ ; β, C ) = ˜ O (( s ∧ 1 β + ε ) d ( β 2 ε 2 +1 )) . In particular, such bounds are capable of providing improvements in distribution-free query complexity over the sample complexity of passive learning M p ( ε, δ ; β, C ) if and only if s &lt; ∞ (which, as discussed above, is a rather strong restriction). This contrasts with Theorems 1, 3, which provide improvements for all concept classes C , regardless of whether s is finite or infinite. (The role of s in Theorem 3 is merely in refining the lower-order term in the special case that s &lt; ∞ ). In this sense, Theorems 1, 3 are most interesting for classes C with s = ∞ , since no previous techniques provide improvements over passive learning for such classes.

We note that, while the AVID Agnostic algorithm (Figure 1) itself should not be regarded as a disagreement-based active learner (as its primary advantage over passive learning is not based on the restriction of queries to DIS( V ) ), elements of disagreement-based learning have been incorporated into it for the purpose of the refined lower-order term in the upper bound in Theorem 3. Specifically, the choice to query examples in S 1 k ∩ D k -1 \ ∆ i k in Step 2 (and similarly Step 9) restricts to queries in D k -1 = DIS( V k -1 ) . This restriction is directly responsible for the lower-order term in Theorem 3 being of the form ˜ O (( s ∧ 1 ε ) d ) rather than ˜ O ( d ε ) as in Theorem 1. On the other hand, for the purpose of the lead term, this incorporation of disagreement-based queries is unnecessary, and indeed Theorem 1 remains valid without this aspect of the algorithm: that is, in Steps 2 and 9, if we simply query all of S 1 k \ ∆ i k , the algorithm still achieves the query complexity bound stated in Theorem 1 with its lower-order term ˜ O ( d ε ) .

The argument leading to the refined lower-order term in Theorem 3 makes use of reasoning directly rooted in the analysis of disagreement-based methods via the disagreement coefficient (Lemma 22), and indeed we present P -dependent refinements of this lower-order term directly expressed in terms of θ P ( β + ε ) in Appendix F.2. In particular, we show (Corollary 28) the lower-order term ˜ O (( s ∧ 1 ε ) d ) can be replaced by ˜ O ( θ P ( β + ε ) 2 d ) , yielding an overall P -dependent query complexity bound O ( β 2 ε 2 ( d +log ( 1 δ )) ) + ˜ O ( θ P ( β + ε ) 2 d ) . we further argue, in Appendix F.1, that it is not possible (by any algorithm) to reduce this lower-order term to ˜ O ( θ P ( β + ε ) d ) or even ˜ O ( θ P (0) d ) , though we do show that other intermediate forms of the term are achievable, such as ˜ O ( θ P ( β + ε ) d ( β + ε ε )) .

## A.2.2 Subregion-based (Margin-based) Active Learning

Shortly after the work of Balcan, Beygelzimer, and Langford (2006), which included an analysis of homogeneous linear classifiers under a uniform distribution, Balcan, Broder, and Zhang (2007) proposed a refinement of disagreement-based active learning specific to linear classifiers. Rather than querying every example in the region of disagreement DIS( V ) , they identified a subregion

̸

R ⊆ DIS( V ) which suffices for the purpose of estimating differences of error rates er P ( f ) -er P ( g ) among f, g ∈ V . The key idea is to choose R so that any f, g ∈ V have PX ( { f = g } \ R ) small, so that R captures most of the disagreements between concepts f, g ∈ V that are far apart. In their case, since they were specifically focusing on homogeneous linear classifiers (i.e., concepts h w ( x ) = 1 [ ⟨ w,x ⟩ ≥ 0] on X = R d ) under PX uniform in an origin-centered ball, they could describe this region R as a slab around the boundary of a current hypothesis h ˆ w : that is, R = { x ∈ R d : |⟨ ˆ w,x ⟩| ≤ b } for an appropriate width b (which decreases over time as the algorithm progresses). In other words, the algorithm queries examples X i with low margin under the current hypothesis ˆ w . As such, this technique is referred to as margin-based active learning . They analyzed this technique for the realizable case and under a specialized noise condition (Tsybakov noise), and found it provides advantages over disagreement-based learning: in the realizable case, improving the query complexity from d 3 / 2 · polylog ( 1 εδ ) to d · polylog ( 1 εδ ) (matching the query complexities achieved by earlier works Freund, Seung, Shamir, and Tishby, 1997; Dasgupta, Kalai, and Monteleoni, 2005; Dasgupta, 2005), while allowing for some robustness to non-realizable distributions P (albeit not fully agnostic). The technique was later extended in various ways, including studying adaptivity to certain noise parameters (Wang and Singh, 2016) and generalizing beyond the uniform distribution, to general isotropic log-concave or s-concave distributions (Balcan and Long, 2013; Balcan and Zhang, 2017).

̸

This idea was extended to general concept classes C and distributions P , including the agnostic setting, in the work of Zhang and Chaudhuri (2014). Again the idea is to identify a region R ⊆ DIS( V ) for which concepts f, g ∈ V have only small disagreements outside R : PX ( { f = g } \ R ) ≤ η , for a small η . Rather than an explicit region R (as in margin-based active learning), they simply choose a subset of the unlabeled examples via a linear program, which they show (in the analysis) can be related to an optimal choice of such a region. We discuss this technique in detail in Appendix F.3.

̸

The implication of this refinement of disagreement-based learning is a P -dependent query complexity bound, stated in terms of a subregion-based refinement of the disagreement coefficient, defined as follows (adopting some simplifications from Hanneke, 2016b). As above, define the r -ball B P X ( h ⋆ , r ) = { h ∈ C : PX ( x : h ( x ) = h ⋆ ( x )) ≤ r } for r &gt; 0 . Also, for η ≥ 0 , define

̸

<!-- formula-not-decoded -->

where R and f are restricted to be measurable. Finally, for ε ≥ 0 , define the subregion disagreement coefficient (Definition 31) as

<!-- formula-not-decoded -->

for an appropriate universal constant c &gt; 1 . The technique of Zhang and Chaudhuri (2014) provides a P -dependent query complexity bound of the form ˜ O ( φ P ( ε, 2 β ) d ( β 2 ε 2 +1 )) . In particular, it follows from the definitions that φ P ( ε, 2 β ) ≤ θ P (2 β + ε ) (see Appendix F.3), and Zhang and Chaudhuri (2014) discuss some examples where the gap is large. Thus, this represents a refinement of the query complexity bounds for disagreement-based active learning discussed above.

As a primary example where φ P ( ε, 2 β ) ≪ θ P ( β + ε ) , consider again the scenario of homogeneous linear classifiers on R d under PX an isotropic log-concave distribution (as considered in the marginbased active learning works of Balcan, Broder, and Zhang, 2007; Balcan and Long, 2013 discussed above). In this scenario, Zhang and Chaudhuri (2014) show that φ P ( ε, 2 β ) = O ( log ( β ε )) (based on concentration arguments from Balcan and Long, 2013). Thus, in this scenario, the query complexity bound of Zhang and Chaudhuri (2014) is ˜ O ( d ( β 2 ε 2 +1 )) . In contrast, Hanneke (2007b) showed θ P ( β + ε ) = Ω ( √ d ∧ 1 β + ε ) for PX the uniform distribution on an origin-centered sphere (a special case of isotropic log-concave), so that the query complexity bounds for disagreement-based active learning are roughly d 3 / 2 ( β 2 ε 2 +1 ) , hence are suboptimal by a factor √ d .

That said, in the context of distribution-free analysis, it is unclear whether there are advantages from this subregion technique. Specifically, Hanneke (2016b) showed that sup P φ P ( ε, 0) = s ∧ 1 ε (where the sup is restricted to realizable distributions P ), which matches the worst-case value of θ P ( ε )

(established by Hanneke and Yang, 2015). In (54) of Appendix F.3, we further extend this to φ P ( ε, η ) (using the fact that φ P ( ε, η ) ≥ φ P ( η + ε, 0) ), establishing that (for ε, η ≥ 0 with η + ε ≤ 1 )

<!-- formula-not-decoded -->

where again the sup is restricted realizable distributions P . Thus, the implication of the P -dependent query complexity bound of Zhang and Chaudhuri (2014) for bounding the distributionfree query complexity QC a ( ε, δ ; β, C ) is merely to recover the same query complexity bound ˜ O (( s ∧ 1 β + ε ) d ( β 2 ε 2 +1 )) already known to hold for disagreement-based active learning. In particular, this means that the above query complexity bound of Zhang and Chaudhuri (2014) is capable of providing improvements in the distribution-free query complexity of active learning, compared to the sample complexity M p ( ε, δ ; β, C ) of passive learning, if and only if s &lt; ∞ (again, a rather strong restriction). Again, this contrasts with Theorems 1, 3, which provide improvements for all concept classes C , regardless of s , with s merely influencing refinements in the lower-order term in Theorem 3.

In Appendix F.3, we give a refinement of the A VID Agnostic algorithm, which adopts this subregion technique (in combination with the AVID principle). We show this Subregion-AVID Agnostic algorithm achieves a P -dependent refinement of the lower-order term compared to the original AVID Agnostic algorithm. For instance, one implication of this refinement is replacing the term ˜ O (( s ∧ 1 ε ) d ) in Theorem 3 with ˜ O ( φ P ( ε, 5 β ) 2 d ) , yielding a P -dependent query complexity bound O ( β 2 ε 2 ( d +log ( 1 δ )) ) + ˜ O ( φ P ( ε, 5 β ) 2 d ) . It follows from an example in Appendix F.1 that the above quadratic dependence φ P ( ε, 5 β ) 2 cannot be reduced to φ P ( ε, 5 β ) (or even φ P (0 , 0) ) without introducing additional factors, though we also establish intermediate forms of the term, such as ˜ O ( φ P ( ε, 5 β ) d ( β + ε ε )) .

## A.2.3 Other Topics and Techniques in the Theory of Active Learning

In addition to disagreement-based active learning and its subregion-based refinement, and query complexity bounds for the agnostic setting, the active learning literature also contains numerous other techniques and topics. Though these ideas are not directly used in the present work, we briefly survey them here for completeness, and in some cases, to discuss connections to the results and techniques of the present work. For brevity, we omit most of the formal definitions, algorithms, and precise statements of the results, rather summarizing the essential ideas, and referring the interested reader to the original works for the precise results and details (some of which are also surveyed by Hanneke, 2014 in detail).

̸

The Splitting Index: The earliest general theory of active learning, providing query complexity bounds applicable to any concept class and realizable-case distribution, was proposed by Dasgupta (2005). That work proposes a ( C , P ) -dependent complexity measure called the splitting index ρ ∈ [0 , 1] , based on the property that, for any γ &gt; ε and any finite set of pairs ( f, g ) ∈ C 2 with PX ( f = g ) &gt; γ , there will likely be an unlabeled example X i for which, regardless of whether Y i is 0 or 1 , we are guaranteed that at least a ρ fraction of the pairs ( f, g ) will have at least one function incorrect on ( X i , Y i ) (see Dasgupta, 2005 for the precise definition). The idea is that ρ measures a notion of progress , from querying such an X i , toward reducing the diameter of the version space below γ . The pairs ( f, g ) in the version space having PX ( f = g ) &gt; γ are the obstructions to reducing the diameter of the version space. The above definition guarantees there will be an example X i we can query such that, regardless of which label Y i is returned, discarding all inconsistent concepts from the version space results in a reduction in the set of such obstructing pairs, leaving at most a 1 -ρ fraction of them. If we start with an α -cover V of the concept class C of size roughly α -d (with α &lt; ε small enough to guarantee all queried labels agree with some h ∈ V ), we would require at most O ( d ρ log ( 1 α )) such queries to eliminate all obstructing pairs, and thus reduce the diameter of the version space below γ . We can then decrease γ by a constant factor and repeat, until the diameter is below ε , at which time we can return any surviving concept, yielding a query complexity roughly O ( d ρ log ( 1 α ) log ( 1 ε )) .

̸

The splitting index also provides a lower bound Ω ( 1 ρ ) on the realizable-case query complexity (where, in this case, ρ can be PX -dependent, but should be h ⋆ -independent; see Dasgupta, 2005; Balcan

and Hanneke, 2012; Hanneke, 2014 for precise statements). This is particularly interesting due to being PX -dependent, and yet still providing near-matching upper and lower bounds (in contrast, other quantities such as the disagreement coefficient and subregion-based refinement only provide upper bounds, and provably cannot yield general PX -dependent lower bounds). Since the above results reveal the PX -dependent query complexity can be well-captured by the splitting index, whereas Hanneke and Yang (2015) have shown the optimal distribution-free realizable-case query complexity is characterized by the star number (Definition 2), it is also natural to study the relation between these quantities. Toward this end, Hanneke and Yang (2015) have in fact shown that these quantities are equivalent in the context of distribution-free analysis: namely, sup P ⌊ 1 ρ ⌋ = min { s , ⌊ 1 ε ⌋} .

The splitting index analysis also provides another interesting feature, which is perhaps missing from other works on active learning in the realizable case: it quantifies a trade-off between the query complexity and the number of unlabeled examples. This is reflected in the above (imprecise) definition in the part that requires that such a ρ -splitting example X i is likely to exist in the unlabeled data (this is made precise in Dasgupta, 2005 by another parameter τ reflecting the probability of obtaining such an example). Using a larger number of unlabeled examples can increase the likelihood of including an example X i that eliminates a larger fraction of pairs, so that the splitting index ρ can grow larger (hence decreasing the query complexity) for larger unlabeled sample sizes. This improvement from having larger unlabeled sample sizes is not reflected in other complexity measures proposed in the literature, and in such cases the splitting-based query complexity bounds can be substantially smaller than those based on these other complexities, such as the disagreement coefficient (see Hanneke, 2014 for explicit comparisons). It is worth noting that such trade-offs are not known to be possible in the agnostic setting.

While the original work of Dasgupta (2005) was developed for the realizable case, subsequent works have explored extensions to non-realizable settings under restrictions on the types of non-realizability. Specifically, Balcan and Hanneke (2012); Hanneke (2014); Tosh and Hsu (2020) have extended the theory to allow for so-called Massart noise , wherein it is assumed P ( Y = h ⋆ ( X ) | X ) -1 2 is everywhere positive and bounded away from 0 . The extension of the splitting technique to that setting merely requires that we only remove a concept from consideration upon having sufficiently many errors on queried examples. The resulting query complexity bounds are then similar to the above.

To date, the splitting technique has not been extended to the agnostic setting in any meaningful way (e.g., to obtain query complexity bounds which could not also be obtained by, say, running disagreement-based active learning with an ( ε/ 2) -cover of the concept class). The agnostic setting presents significant challenges for this technique, due to the ρ -splitting examples X i being possibly in ER( h ⋆ ) for the best-in-class concept h ⋆ , meaning such examples cannot be trusted as the sole source of information for pruning suboptimal concepts; see the scenario in Appendix F.1 (which is constructed therein for a different reason, but also illustrates this issue).

̸

We may remark that the AVID principle, developed in the present work and employed in A avid , has certain aspects that are intriguingly reminiscent of the splitting technique of Dasgupta (2005). As in splitting, A avid aims to reduce the diameter of a set V of surviving concepts. Toward this end, again as in splitting, it identifies obstructing pairs : f, g ∈ V with PX ( f = g ) &gt; ε k , where ε k is the desired diameter guarantee at that stage. However, the main difference is in how such obstructions are addressed in the algorithm. While the splitting technique would attempt to resolve this obstruction by querying to eliminate at least one of f, g for many such obstructing pairs, A avid instead simply removes (isolates) the region { f = g } from the space X (adding it to the ∆ region), and estimates error rates separately in ∆ and X \ ∆ . Thus, in this aspect, the algorithmic principle underlying A avid is considerably different from splitting. Nevertheless, this common focus on addressing pairs ( f, g ) obstructing the reduction of the diameter presents an intriguing connection, which might potentially warrant further exploration.

̸

Empirical Teaching Dimension: Another approach to agnostic active learning, based on principles seemingly distinct from disagreement-based methods, was proposed in the work of Hanneke (2007a). The technique there is inspired by early work on Exact learning with membership queries in the realizable case by Hegedüs (1995); Hellerstein, Pillaipakkamnatt, Raghavan, and Wilkins (1996), which found interesting connections between active learning and the complexity of machine teaching (Goldman and Kearns, 1995). Hanneke (2007a) extends those ideas to the PAC setting, starting with an upper bound for the realizable case, based on a PX -dependent complexity measure τ ( ε ) therein

termed the extended teaching dimension growth function : for an i.i.d.PX data set S of size 1 ε , τ ( ε ) (roughly) represents the minimal size of a subsample which induces the same version space (for any fixed target concept h ⋆ ). The main technique is to find sets of τ ( ε ) unlabeled examples for which the labels are guaranteed to significantly reduce the number of concepts in (a finite cover of) the version space. The work also presents a realizable-case lower bound based on a modified variant of this complexity measure.

Hanneke (2007a) further extends the upper bound to the non-realizable case, establishing an upper bound ˜ O ( τ ( β + ε ) d ( β 2 ε 2 +1 )) . In particular, this matches the lower bound (6) of Kääriäinen (2006); Beygelzimer, Dasgupta, and Langford (2009) up to logs when τ ( β + ε ) = ˜ O (1) . Hanneke (2007a) provides examples where τ ( β + ε ) is bounded, including the class of thresholds (concepts 1 [ a, ∞ ) on X = R ) and axis-aligned rectangles (of at least some volume) under restrictions on PX ; however, as with the previously discussed complexity measures, τ ( β + ε ) can be as large as 1 β + ε for the 'searching in the dark' type scenarios discussed above, in which case the above query complexity bound is no smaller than the sample complexity of passive learning. Indeed, as with the complexity measures discussed above, results of Hanneke and Yang (2015) imply that, taking the worst case over distributions, τ ( ε ) becomes equivalent to the star number (Definition 2): sup P X τ ( ε ) ≈ min { s , 1 ε } .

Variants of this τ ( ε ) complexity measure were later further analyzed (for several example scenarios, and more-generally, in relation to the disagreement coefficient), under the name version space compression set size , and (interestingly) have also been found useful for studying disagreement-based active learning, by El-Yaniv and Wiener (2010, 2012); Wiener, Hanneke, and El-Yaniv (2015); Hanneke and Yang (2015); Hanneke (2016b); Hanneke and Kontorovich (2021).

Restricted Noise Models: Besides the study of first-order agnostic query complexity guarantees QC a ( ε, δ ; β, C ) (the subject of the present work), the theory of active learning additionally includes many works on query complexity guarantees holding under other conditions or parameterizations, or in other words, under various noise models . Here we briefly survey some of this literature.

The most-similar noise model to that studied in the present work is the benign noise setting (Hanneke, 2009b), which differs from the agnostic setting only in that it makes the additional assumption that inf h ∈ C er P ( h ) = inf h er P ( h ) , where the infimum on the right hand side is over all measurable functions h : X → { 0 , 1 } (not necessarily in C ). In other words, the benign noise setting assumes the best-in-class error β = inf h ∈ C er P ( h ) is also the Bayes risk of P : i.e., the error rate of the function x ↦→ 1 [ P ( Y = 1 | X = x ) ≥ 1 2 ] . Since the distributions used to establish the lower bound (4) for passive learning satisfy the benign noise condition, this still serves as a suitable comparison point for the query complexity of active learning. Similarly, the distributions used to establish the lower bound (6) for active learning also satisfy benign noise, and therefore the lower bound (6) also holds in the benign noise setting. Notably, in the special case of benign noise, Hanneke and Yang (2015) have shown a result analogous to the present work: the optimal first-order query complexity of active learning is ˜ O ( d β 2 ε 2 +min { s , d ε } ) . In particular, comparing to (4), this means, under the benign noise assumption, the query complexity of active learning is always better than the sample complexity of passive learning. That work posed the question of whether such improvements are also attainable in the more-challenging agnostic setting, a question which the present work answers positively. Notably, the above result for benign noise is even slightly sharper in the lower-order term, compared to our Theorem 3 (which has d s rather than s ); I conjecture this d s can also be reduced to s in the agnostic setting. There are interesting connections or analogies between the algorithm used by Hanneke and Yang (2015) and the AVID principle developed in the present work, and we discuss these connections in Appendix A.3 below. However, one noteworthy point is that A avid requires vastly fewer unlabeled examples to obtain the query complexity guarantee, compared to the method of Hanneke and Yang (2015), so that the present work also offers some benefits over the known techniques for the benign noise setting as well.

## A.3 Background of the AVID Principle

Having surveyed much of the related work on agnostic active learning above, we conclude our discussion of related work by discussing previous works in the learning theory literature containing ideas related to our main technique (the A VID principle).

̸

Arguably the main innovation involved in this work is the decomposition of the space X into regions X \ ∆ i k and ∆ i k , and augmenting the predictor ˆ h k to be a (shallow) decision list of concepts from C . One key inspiration for the main idea underlying the technique is rooted in the works of Bousquet and Zhivotovskiy (2021); Puchkin and Zhivotovskiy (2022) on prediction with an abstention option (evaluated with the Chow loss ). Interestingly, this continues a long precedent of finding useful connections and cross-inspirations between active learning and prediction with abstentions (Mitchell, 1979; El-Yaniv and Wiener, 2010, 2012; Zhang and Chaudhuri, 2014, e.g.,). Specifically, Bousquet and Zhivotovskiy (2021); Puchkin and Zhivotovskiy (2022) consider methods exhibiting a kind of transition time , in which they determine that, for some f, g ∈ C , abstaining in a the pairwise disagreement region { x : f ( x ) = g ( x ) } , and predicting with f in its complement, comes out to have smaller Chow loss than the overall loss of the best h ∈ C . Some reasoning very much analogous to this (and directly inspired by it) can be found in one of the base cases of the arguments in the present paper (namely, concerning the 'early stopping' case in the algorithm), in which we find that in the case of early stopping (Step 4), we can find f, g ∈ V k -1 and h 1 , h 2 ∈ C , such that predicting with h 1 in { f = g } \ ∆ i k (rather than abstaining), with f in { f = g } \ ∆ i k , and with h 2 in ∆ i k , produces a smaller overall error rate in compared to the best concept h ⋆ ∈ C . Of course, the algorithm and analysis here contain many additional pieces on top of this, but it is interesting that this connection to learning with abstentions still remains present at the core (though it is noteworthy that this connection is qualitatively different from the usual one, in that here we are not replacing abstentions with queries, but rather that a part of the analysis inspires part of our analysis). We remark that this analysis of learning with abstentions by Bousquet and Zhivotovskiy (2021); Puchkin and Zhivotovskiy (2022) was also inspirational for an active learning method in the work of Zhu and Nowak (2022) (though the aim in that work is different from the present work, and the setting is generally not comparable to ours).

At a high level, we can view the technique as also analogous to an idea of Hanneke and Yang (2015) developed for the benign noise model: namely, the restriction of the agnostic setting to the case the Bayes classifier h ⋆ Bayes ( x ) ↦→ 1 [ P ( Y = 1 | X = x ) ≥ 1 / 2] is in the concept class C . Hanneke and Yang (2015) prove a query complexity bound for this special case which matches Theorem 3 (and indeed, refines the lower-order term's s d dependence to simply s ). In that context, since the h ⋆ Bayes ∈ C , the only source of non-realizability is in the noisiness of the conditional label distribution Y | X . Thus, if an active learner could repeatedly query a given X t to receive multiple conditionally independent samples of Y t given X t , it could use the majority vote of these samples to effectively de-noise the label of X t , thereby identifying h ⋆ Bayes ( X t ) . This strategy only fails if P ( Y = 1 | X = X t ) is very close to 1 2 , in which case this de-noising would require too many queries to be worthwhile, particularly since such noisy examples have very little effect on the excess error rate er P ( ˆ h ) -er P ( h ⋆ Bayes ) . As such, if the active learner cannot identify the optimal label within some number of queries, it should abandon the example X t and move on. Of course, in the model of active learning studied in this work, and in the work of Hanneke and Yang (2015), an active learner cannot actually obtain multiple conditionally independent copies of the label Y t . However, by appropriate discretization of the space X based on the structure of the concept class C , Hanneke and Yang (2015) are able to approximate this idealized behavior. The resulting algorithm effectively adapts to the noisiness of the labels of examples X t within the equivalence classes induced by this discretization, allocating more queries to the noisier (high-label-variance) regions (and abandoning the regions it finds to be too noisy). In that sense, the high-level idea behind the A VID principle is similar in nature. The goal is to isolate the regions where learning is more challenging, due to higher variance in error difference estimation, and allocate disproportionately more queries to these regions. Of course, in the agnostic case, this is made much more challenging, since the source of non-realizabilityy is not merely label noise, but also model misspecification (i.e., h ⋆ Bayes / ∈ C ) so that de-noising the examples may sometimes have little benefit (e.g., it is even possible to have β &gt; 0 while P ( Y = 1 | X ) ∈ { 0 , 1 } ). As such, the AVID principle necessarily makes greater use of the structure of the concept class to isolate such regions of high variance in error difference estimation.

It is worth mentioning that other works on active learning have also considered decomposing the space X into subregions and learning separately in each region (e.g., Cortes, DeSalvo, Gentile, Mohri, and Zhang, 2019a; Cortes, DeSalvo, Gentile, Mohri, and Zhang, 2019b, 2020). However, we note that these works retain the above issue of having a query complexity of the form c ( β ) d β 2 ε 2 for a complexity measure c ( β ) (as discussed in Section 2) such that, in the worst case over distributions P

̸

(respecting the β constraint) the results become ultimately no smaller than the sample complexity of passive learning.

The idea of decomposing a predictor into a decision list based on pairwise disagreement regions has an even closer parallel in the recent work of Hanneke, Larsen, and Zhivotovskiy (2024b), which removes a log factor from the lead term in the (first-order) sample complexity of passive learning, thereby obtaining an optimal lead term of Θ ( β ε 2 ( d +log ( 1 δ )) ) . The overall approach in that work is in many ways similar to the technique in the present work, though with some important differences in the actual algorithms. In particular, since the interest in that work is merely removing a factor log ( 1 β ) , it essentially suffices for the algorithm to reduce the best-in-class error rate in a region X \ ∆ down to β log(1 /β ) (for PX (∆) = O ( β ) ), so that a uniform Bernstein inequality for the error rate of ERM implies the desired result in that region X \ ∆ , and a uniform convergence analysis of ERM under the conditional distribution given ∆ implies the desired result in the region ∆ . In contrast, our interest in the present work is a factor of β in the lead term, with a lower-order term of size ˜ O ( d ε ) , and to achieve this our algorithm aims to reduce (below ε ) the diameter of a set V k of surviving concepts, in a region X \ ∆ (with PX (∆) = O ( β ) ). We achieve this via uniform estimation of error differences, using an appropriate number of samples from these two regions, while precisely controlling the schedule of decreases of this diameter in the algorithm (in part by increasing the ∆ region as needed to maintain this schedule of diameter decreases). Nevertheless, the essential inspiration and strategy behind these two algorithms are notably related, perhaps indicating that the AVID principle might in fact be a widely useful idea.

## B Additional Definitions and Notation

We provide additional definitions and notation required for the formal analysis. A fundamental quantity in statistical learning theory is the VC dimension (Vapnik and Chervonenkis, 1971), which plays an important role in characterizing the optimal query complexity (and optimal sample complexity of passive learning). It is defined as follows.

Definition 4. For any concept class C , the VC dimension of C , denoted by VC( C ) , is defined as the supremum n ∈ N ∪ { 0 } for which there exists a sequence { x 1 , . . . , x n } ∈ X n such that { ( h ( x 1 ) , . . . , h ( x n )) : h ∈ C } = { 0 , 1 } n (i.e., all 2 n classifications are realizable by C ).

For brevity, in all results, proofs, and discussion below (where C is clear from the context), we will simply denote by d := VC( C ) . In all statements below, we suppose d &lt; ∞ (see Appendix G). Also note that, by our assumption that | C | ≥ 3 (see footnote 1), we always have d ≥ 1 .

̸

Additional Notation and Conventions: For any distribution P on X × { 0 , 1 } , denote by P X the marginal distribution on X . Throughout, we refer to any sequence S ∈ ( X × { 0 , 1 } ) ∗ as a data set . For any x ∈ R , it will be convenient to define log( x ) = ln(max { x, e } ) , and for x &gt; 0 we define log( x/ 0) = x/ 0 = ∞ and 0 log( x/ 0) = 0 . For a, b ∈ R ∪{∞} , we use a ∧ b or min { a, b } to denote the minimum of a and b , and a ∨ b or max { a, b } to denote the maximum of a and b . We will make use of standard bigO notation ( O , Ω , Θ effectively hide universal constant factors, while ˜ O , ˜ Θ effectively hide log factors) to simplify theorem statements. The precise constant and log factors will always be made explicit in the formal proofs. We also adopt a convention regarding conditional probabilities: all claims involving conditional probabilities given a random variable should be interpreted as holding almost surely (i.e., for a version of the conditional probability), such as when claiming that an event holds with conditional probability at least 1 -δ given a random variable X . We also continue the notational conventions introduced in Section 4, such as ER( h ) , DIS( C ′ ) , { f = g } , overloading set notation to treat A ⊆ X as notationally interchangeable with its labeled extension A ×{ 0 , 1 } , extending notation for set-intersection to allow intersections with sequences, and defining empirical estimates ˆ P S ( A ) = | S ∩ A | / | S | . See Section 4 for details of these conventions.

Measurability: We remark that, formally speaking, an active learning algorithm can be defined simply as a measurable function A : ( X × { 0 , 1 } ) m × X → { 0 , 1 } : that is, taking as input an i.i.d. data set S = { ( X i , Y i ) } i ≤ m and an independent test point X and evaluating to a prediction A ( S, X ) ∈ { 0 , 1 } . In this view, the number of queries is merely bookkeeping, keeping track

̸

of the dependences of this function on the labels Y i . For simplicity of presentation, we have adopted the common colloquialism of referring to the function ˆ h returned by A ( S ) , which in this view simply refers to the function A ( S, · ) , so that er P ( ˆ h ) is simply the conditional expectation E [ 1 [ A ( S, X ) = Y ] | S ] . The measurability of the algorithms A defined in this work follows from measurability of the individual operations involved in their execution 10 under the standard measuretheoretic assumptions on ( X , C ) specified in footnote 1. To simplify the presentation, we do not explicitly discuss this in the proofs.

## C The Query Complexity of the AVID Agnostic Algorithm

This section presents a detailed version of Theorem 3, bounding the query complexity of the AVID Agnostic algorithm. Recall the definition of the algorithm and notation from Section 4. Before stating the theorem, we first discuss a few additional technical aspects of the algorithm omitted from the high-level description in Section 4, starting with an explicit specification of the quantities involved. Let c , c be universal constants, defined by Lemmas 7 and 8 of Appendix D. We define 11

0 1 C = 11 10 , C ′′ = ( 200 C 3 8 -5 C 3 ) 2 , and C ′ = √ C ′′ 16 . For a given ε, δ ∈ (0 , 1) (arguments to A avid ), as in Section 4 we let N = ⌈ log C ( 2 ε )⌉ , and for k ∈ N , let ε k = C 1 -k , and we then define m k := ⌈ 300 C ′′ c 0 ε k ( d log ( C ′′ c 0 ε k ) +log ( 1 δ ) )⌉ . The algorithm adaptively allocates data subsets S 1 k , S 2 k , S 3 k,i , S 4 k during its execution, as described in Section 4.1. Recall that S 1 k , S 3 k,i , and S 4 k are all of size m k (for any k, i for which they exist). The data subset S 2 k is of size m ′ k , formally defined as follows. For the value i k and the set ∆ i k as defined in the algorithm at the time that S 2 k is allocated (either in Step 2 for some value of k , or in Step 9, in which case let k = N +1 ), letting ˆ p k := 2 ˆ P S 4 k (∆ i k ) , define m ′ k := ⌈ C ′′ c 2 1 ˆ p k ε 2 k ( d +log ( 4(3+ N -k ) 2 δ ))⌉ .

We remark that, for simplicity of presentation, we have described the algorithm without explicitly discussing what happens if the algorithm runs out of unlabeled examples while allocating examples to subsets S 2 k , S 3 k,i . In this event, the algorithm can simply halt and return an arbitrary predictor ˆ h , as the analysis will account for this event in the δ failure probability. To avoid excessive clutter, we do not explicitly mention this case in the description of the algorithm or allocation of data subsets used therein (i.e., we explicitly discuss this only in the analysis, and indeed only in the final part of the proof; see the discussion at the start of Appendix E).

The following theorem provides a bound on the query complexity achieved by A avid along with a bound on the unlabeled data set size sufficient to achieve it. This result represents a detailed version of the upper bound in Theorem 3 of Section 4 (in particular, Theorems 1 and 3 are immediate implications of this result). The constant factors in the bigO will be made explicit in the formal proof. The proof is given in Appendix E.

Theorem 5 (Query Complexity of AVID Agnostic) . For any concept class C with VC( C ) &lt; ∞ , letting d = VC( C ) , for every distribution P on X ×{ 0 , 1 } , letting β = inf h ∈ C er P ( h ) , for any ε, δ ∈ (0 , 1) , if the algorithm A avid is executed with parameters ( ε, δ ) , with any number m ≥ M ( ε, δ ; β ) of

10 The only part requiring some care in this regard is the definition of ˆ h k in (3), where formally we require that, given V k -1 , ∆ i k , m ′ k , the function ( S 1 k , S 2 k , x ) ↦→ ˆ h k ( x ) should be a measurable function; such a measurable function can be shown to exist assuming C (and therefore V k -1 ) satisfies the conditions of footnote 1, following straightforwardly from arguments of (Dudley, 1999).

11 For simplicity of presentation, the constant C plays two major roles in the algorithm. First, it controls the schedule of diameter guarantees ε k in the algorithm. Second, it controls certain constant factors in uniform concentration guarantees employed in the proof (Lemma 10). If we were to separate these roles, into C 1 and C 2 , respectively, the two values exhibit a trade-off. In particular, we can admit a schedule ε k = C 1 -k 1 for any choice of 1 &lt; C 1 &lt; 2 by an appropriately large choice of C ′′ (diverging as C 1 → 2 ) and corresponding C 2 &gt; 1 sufficiently close to 1 . The source of this 2 limitation is the multiplicative factor in Lemma 20 which, in the limit as C ′′ →∞ and C 2 → 1 , becomes 2 2 -C 1 . We also remark that we have defined constants that enable the cleanest presentation of the algorithm and analysis. We leave the issue of optimizing the constants to minimize the query complexity for future work.

i.i.d.P examples, for a value M ( ε, δ ; β ) (defined in Lemma 24) satisfying

<!-- formula-not-decoded -->

then with probability at least 1 -δ , the returned predictor ˆ h satisfies er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε and the algorithm makes a number of queries at most Q ( ε, δ ; β ) (defined in Lemma 23) satisfying

<!-- formula-not-decoded -->

Remark on adaptivity to β : We emphasize that the algorithm does not need to know β in its execution (i.e., it adaptively achieves the above query complexity bound for all β ). A more subtle point worth noting is that we can also run the algorithm without ourselves knowing β (to choose m ), since the guarantee on query complexity holds for any unlabeled sample size m ≥ M ( ε, δ ; β ) . For instance, if we run the algorithm with a β -independent number of unlabeled examples m = ˜ Θ ( 1 ε 2 ( d log ( 1 ε ) +log ( 1 δ ))) , the query complexity bound Q ( ε, δ ; β ) would remain valid as stated in Theorem 5. Additionally, in the proof (see Lemma 24), we show that, in a sense, even the unlabeled sample complexity M ( ε, δ ; β ) is achieved adaptively , since the algorithm (with no knowledge of β ) only actually uses the first (at most) M ( ε, δ ; β ) unlabeled examples in the sequence. This is itself an interesting feature. In particular, if we consider an alternative setting where, rather than getting the unlabeled data altogether at the start, the algorithm can adaptively sample new unlabeled examples X i ∼ P X one-at-a-time during execution (i.e., it has access to an unlabeled example oracle , which it can use to construct the data subsets S 1 k , S 2 k , S 3 k,i , S 4 k , during execution), the analysis establishes that the algorithm will succeed while adaptively sampling at most M ( ε, δ ; β ) unlabeled examples (and querying at most Q ( ε, δ ; β ) of them), all without knowing β (or anything else about P ).

## D Concentration Inequalities

This section presents a number of useful concentration inequalities, essential to the analysis. We begin with the classic multiplicative Chernoff bound (Chernoff, 1952; Bernstein, 1924). We will find the following particular form to be useful; since this is slightly different from the more-typical statements of Chernoff bounds, we include a brief explanation of how this result is derived from the more-standard exponential form.

Lemma 6 (Multiplicative Chernoff bound) . Fix any p ∈ [0 , 1] and n ∈ N , and let B 1 , . . . , B n be i.i.d. Bernoulli( p ) random variables. Let ¯ B := 1 n ∑ n i =1 B i . For any δ ∈ (0 , 1) , with probability at least 1 -δ , the following both hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We include a brief explanation, based on more well-known exponential forms of the Chernoff bound: namely, P ( ¯ B &lt; (1 / 2) p ) ≤ e -np/ 8 and P ( ¯ B &gt; 2 p ) ≤ e -np/ 3 (see e.g., Zhang, 2023).

For the first inequality in the lemma, we note that it trivially holds if p &lt; 8 n ln ( 2 δ ) , and otherwise, if p ≥ 8 n ln ( 2 δ ) , then by the above exponential tail bound, we have P ( ¯ B &lt; (1 / 2) p ) ≤ e -np/ 8 ≤ δ 2 . For the second claimed inequality, note that it trivially holds if 6 n ln ( 2 δ ) ≥ 1 , so let us focus on the case 6 n ln ( 2 δ ) &lt; 1 . Note that for p ′ ∈ [0 , 1] and B ′ 1 , . . . , B ′ n i.i.d. Bernoulli( p ′ ) , and ¯ B ′ = 1 n ∑ n i =1 B ′ i , for any x ∈ R the value of P ( ¯ B ′ &gt; x ) is non-decreasing in p ′ . Thus, letting p ′ = max { p, 3 n ln ( 2 δ )} ≥ p , this monotonicity (together with the second exponential tail bound above) implies P ( ¯ B &gt; 2 p ′ ) ≤ P ( ¯ B ′ &gt; 2 p ′ ) ≤ e -np ′ / 3 ≤ δ 2 . The lemma then follows by the union bound, so that both of these inequalities hold simultaneously with probability at least 1 -δ . ■

<!-- formula-not-decoded -->

We will also rely heavily on uniform concentration inequalities. Toward stating these, we first introduce additional useful notation.

VC dimension of collections of sets: As is standard in the literature, we overload the definition of VC dimension (Vapnik and Chervonenkis, 1971) to also allow for collections of sets. Formally, for any non-empty set Z and any non-empty A ⊆ 2 Z (i.e., a collection of subsets of Z ), the VC dimension of A , denoted by VC( A ) , is the supremum n ∈ N ∪ { 0 } for which there exists Z ⊆ Z with | Z | = n such that { Z ∩ A : A ∈ A} = 2 Z (i.e., it is possible to pick out any subset of Z by intersection with an appropriate A ∈ A ). Equivalently, VC( A ) is the VC dimension (Definition 4) of the indicator functions { 1 A : A ∈ A} .

Uniform concentration term: For any non-empty set Z and any non-empty A ⊆ 2 Z , for any n ∈ N and δ ∈ (0 , 1) , define (for a universal constant c 0 defined by Lemma 7 below)

<!-- formula-not-decoded -->

The following result represents a uniform variant of the classic Bernstein inequality (or Bennett inequality) (Bernstein, 1924; Bennett, 1962). It can be derived from results proven by Vapnik and Chervonenkis (1974) (see Hanneke and Kpotufe, 2022 for an explicit derivation, via a layered application of Massart's lemma and Bousquet's inequality). We additionally include implications providing a uniform variant of multiplicative Chernoff bounds, which are easily derived from the stated uniform Bernstein inequality (taking B = ∅ ).

Lemma 7 (Uniform Bernstein and multiplicative Chernoff bounds) . There is a finite universal constant c 0 &gt; 1 for which the following holds. Fix any n ∈ N , δ ∈ (0 , 1) , any non-empty set Z , and any set A ⊆ 2 Z with VC( A ) &lt; ∞ . 12 Define ε ( n, δ ; A ) as in (7) . Fix any distribution P on Z and let Z = { Z 1 , . . . , Z n } ∼ P n (i.i.d. P random variables). For any measurable set A ⊆ Z , define its empirical probability ˆ PZ ( A ) := 1 n ∑ n i =1 1 [ Z i ∈ A ] . With probability at least 1 -δ , every A,B ∈ A ∪ {∅} satisfy the following (where A ⊕ B := ( A \ B ) ∪ ( B \ A ) denotes the symmetric difference)

<!-- formula-not-decoded -->

Moreover, for any ε &gt; 0 and α ∈ (0 , 1) satisfying ε ( n, δ ; A ) ≤ α 2 4 ε , the above inequality immediately yields the following implications: ∀ A ∈ A ,

<!-- formula-not-decoded -->

We also make use of a uniform concentration inequality which refines the classic uniform convergence bound √ 1 n ( VC( A ) + log ( 1 δ )) of Talagrand (1994) in the case that ⋃ A has small measure under P . The lemma is well-known in the literature, and follows immediately from expectation bounds based on chaining involving an envelope function (e.g., Theorem 2.14.1 of van der Vaart and Wellner, 1996) together with Bousquet's inequality (Bousquet, 2002) to achieve high probability. For completeness, we provide a brief direct proof, by simply applying the uniform convergence bound of Talagrand (1994) to the samples from the conditional distribution given a set D ⊇ ⋃ A .

Lemma 8. There is a finite universal constant c 1 ≥ 1 for which the following holds. Let A be as in Lemma 7, and suppose D ⊆ Z is a measurable set such that ∀ A ∈ A , A ⊆ D . Then for the same quantities as Lemma 7, if P ( D ) ≥ 9 n ln ( 4 δ ) , then with probability at least 1 -δ , ∀ A ∈ A

<!-- formula-not-decoded -->

12 We suppose standard mild measure-theoretic restrictions on A and the σ -algebra of Z , from empirical process theory: namely, the image-admissible Suslin condition (Dudley, 1999).

Proof. Note that the samples in Z ∩ D are conditionally i.i.d. P ( ·| D ) given | Z ∩ D | . For each A ∈ A , denote by ˆ PZ ( A | D ) := ˆ P Z ∩ D ( A ) (or 0 if | Z ∩ D | = 0 ). Applying the uniform convergence bound of Talagrand (1994) to the samples in Z ∩ D under the conditional distribution given | Z ∩ D | , together with the law of total probability, yields that, with probability at least 1 -δ 2 , ∀ A ∈ A ,

<!-- formula-not-decoded -->

for a finite universal constant c ′ 1 ≥ 1 . Moreover, by Bernstein's inequality (see Theorem 2.10 of Boucheron, Lugosi, and Massart, 2013), with probability at least 1 -δ 2 ,

<!-- formula-not-decoded -->

where the last inequality is due to the assumption that P ( D ) ≥ 9 n ln ( 4 δ ) . By the union bound, these two events occur simultaneously with probability at least 1 -δ . Suppose this occurs. In particular, by the assumption that P ( D ) ≥ 9 n ln ( 4 δ ) , (9) further implies

<!-- formula-not-decoded -->

so that the right hand side of (8) is at most

<!-- formula-not-decoded -->

Combining this with (8) and (9) implies that ∀ A ∈ A , since A ⊆ D ,

<!-- formula-not-decoded -->

where c 1 := c ′ 1 √ 6 + 2 √ ln(4 e ) (recalling log( x ) := ln( x ∨ e ) ).

■

## E Proof of Theorem 5: Query Complexity of the AVID Agnostic Algorithm

The formal proof of Theorem 5, given at the end of this section, will be built up from a sequence of lemmas, roughly following the outline presented in Section 4.2.

Throughout this section, we fix an arbitrary concept class C (with d := VC( C ) &lt; ∞ ) and distribution P on X × { 0 , 1 } , let β = inf h ∈ C er P ( h ) , fix any ε, δ ∈ (0 , 1) (where ε, δ are inputs to the AVID algorithm), let ( X 1 , Y 1 ) , ( X 2 , Y 2 ) , . . . be independent P -distributed examples, and we let all values ( N , ε k , m k , m ′ k , etc.) be defined as in Appendix C, based on these values ε, δ , and the examples ( X 1 , Y 1 ) , ( X 2 , Y 2 ) , . . . . Also let h ⋆ ∈ C denote any concept with

<!-- formula-not-decoded -->

For full generality, we do not assume there exists a minimizer achieving the infimum on the right hand side; rather, any choice of h ⋆ satisfying this near -minimality property will suffice for our purposes in the analysis below.

To simplify the proof, we will establish the sequence of lemmas under a scenario where the algorithm is executed with an inexhaustible source of examples (for the adaptive allocation of data subsets): i.e., an infinite sequence ( X 1 , Y 1 ) , ( X 2 , Y 2 ) , . . . of independent P -distributed examples. However, it will follow from these lemmas that, with high probability, the algorithm only depends on a finite prefix ( X 1 , Y 1 ) , . . . , ( X m , Y m ) , for a sufficiently large m = M ( ε, δ ; β ) as in Theorem 5 (see Lemma 24). At the end of the section, when combining the lemmas into a formal proof of Theorem 5, we will return to the standard setting where the algorithm has access only to such a finite prefix. In that context, the event that the algorithm attempts to access any examples ( X t , Y t ) with t &gt; m will be accounted for as part of the allowed δ -probability failure event, and thus (as mentioned in Appendix C) in such a case the algorithm can simply halt and return an arbitrary predictor ˆ h . As mentioned in the remark following Theorem 5, the fact that the algorithm adaptively decides how many unlabeled examples to use is itself an interesting feature, as it means the algorithm can be considered adaptive to β even in its use of unlabeled examples.

Before proceeding with the proof, we first introduce some convenient notation regarding the values of k and i encountered in the algorithm. If the algorithm returns in Step 9, denote by K := N +1 , and otherwise, let K be the maximum value of k reached in the ' For ' loop in the algorithm; we argue in Lemma 10 below that the algorithm terminates eventually, with high probability, so that this latter case coincides with the case of returning in Step 4, with K being the value of k on which this occurs. Let K := { 1 , . . . , K ∧ N } : that is, the set of values of k encountered in the ' For ' loop in the algorithm. Also, for each k ∈ K , denote by I k the values of i encountered by the algorithm on round k ; in particular, for k &lt; K , I k = { i k , . . . , i k +1 } . In the case K = N +1 , for convenience also denote by I N +1 := { i N +1 } .

We begin with a lemma which motivates our choice of sample size m k for S 1 k , S 3 k,i , S 4 k . Recall m k := ⌈ 300 C ′′ c 0 ε k ( d log ( C ′′ c 0 ε k ) +log ( 1 δ ) )⌉ . Also recall our convention (adopted throughout this work) of treating sets D ⊆ X as notationally interchangeable with their labeled extension D ×{ 0 , 1 } , such as in A ∩ D or A \ D for A ⊆ X × { 0 , 1 } .

Lemma 9. Fix any set D ⊆ X and define a family of subsets of X × { 0 , 1 } :

̸

<!-- formula-not-decoded -->

̸

For any n ∈ N and δ ′ ∈ (0 , 1) , let ε ( n, δ ′ ; A ) be defined as in (7) . For each k ∈ { 1 , . . . , N +1 } , letting δ k := δε 2 k +3 72 , it holds that

<!-- formula-not-decoded -->

Proof. We begin by bounding VC( A ) , as needed to evaluate ε ( m k , δ k ; A ) . Define the following families of subsets of X × { 0 , 1 } :

̸

<!-- formula-not-decoded -->

̸

̸

First note that A ⊆ A 2 . To see this, note that for any f, g, h ∈ C , taking A = ER( f ) , B = ER( h ) , C = { f = g }×{ 0 , 1 } , we have that ((ER( f ) ∩{ f = g } ) ∪ (ER( h ) ∩{ f = g } )) \ D = (( A \ C ) ∪ ( B ∩ C )) \ D ∈ A 2 . Similarly, for any f, g ∈ C , taking A = ∅ , B = X × { 0 , 1 } , C = { f = g } × { 0 , 1 } reveals ( { f = g } × { 0 , 1 } ) \ D = (( A \ C ) ∪ ( B ∩ C )) \ D ∈ A 2 . Finally, for h, f ∈ C , taking A = ER( h ) , B = ∅ , C = { f = f }×{ 0 , 1 } = ∅ reveals ER( h ) \ D = (( A \ C ) ∪ ( B ∩ C )) \ D ∈ A 2 .

̸

̸

Next we bound VC( A 2 ) . It is immediate from the definition that VC( { ER( h ) : h ∈ C } ) = d . Moreover, this implies VC( A 0 ) ≤ d +2 (Vidyasagar, 2003, Lemma 4.11). Also note that A 1 ⊆ { A ⊕ B : A,B ∈ A 0 } , where A ⊕ B := ( A \ B ) ∪ ( B \ A ) is the symmetric difference: that is, trivially ( X ×{ 0 , 1 } ) ⊕∅ = X ×{ 0 , 1 } , and for any f, g ∈ C , { f = g }×{ 0 , 1 } = ER( f ) ⊕ ER( g ) . Thus, any element of A 2 can be expressed as a fixed function of four sets A,B,A ′ , B ′ ∈ A 0 : namely ( A,B,A ′ , B ′ ) ↦→ (( A \ ( A ′ ⊕ B ′ )) ∪ ( B ∩ ( A ′ ⊕ B ′ ))) \ D . Based on this fact, well-known results about the effect of such combinations on the VC dimension imply VC( A 2 ) = O (VC( A 0 )) : explicitly, Theorem 4.5 of Vidyasagar (2003) implies VC( A 2 ) ≤ 25VC( A 0 ) ≤ 25( d +2) . By the

̸

̸

assumption that | C | ≥ 3 (footnote 1) we know d ≥ 1 , so that 25( d +2) ≤ 75 d . Altogether, we have VC( A ) ≤ 75 d .

With this in mind, we may note that (also using that d ≥ 1 and C ′′ ≥ 9 C 3 )

<!-- formula-not-decoded -->

In particular, if VC( A ) ≥ 1 , then by Corollary 4.1 of Vidyasagar (2003), (12) implies

<!-- formula-not-decoded -->

Moreover, if VC( A ) = 0 , then recalling we define 0 log(1 / 0) = 0 , (12) trivially implies (13) in this case as well. Thus, regardless of the value of VC( A ) , by definition of ε ( m k , δ k ; A ) , the claim in (11) follows from (13). ■

We continue the proof with a lemma conveniently summarizing several uniform concentration bounds which are useful in various places throughout the rest of the proof. In particular, the lemma focuses on concentration inequalities in the X \ ∆ i k region of focus of the learning algorithm. It will therefore be convenient to explicitly define the portion of the functions in V (4) k -1 specific to this region: namely, for every k ∈ { 1 , . . . , K } , define 13

̸

<!-- formula-not-decoded -->

Lemma 10. On an event E 0 of probability at least 1 -δ 4 , for every k ∈ { 1 , . . . , K } , it holds that

̸

<!-- formula-not-decoded -->

and for every k ∈ K and every i ∈ I k , ∀ f, g ∈ C ,

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

and moreover, max I k ≤ 1 ε k +3 . In particular, the latter implies the algorithm eventually terminates (in Step 9 if K = N +1 , or in Step 4 if K ≤ N ).

Proof. Consider any k ∈ { 1 , . . . , N +1 } having a non-zero probability of k ≤ K . Let δ k be as in Lemma 9. Recall that the data set S 1 k is independent of all data involved in rounds k ′ &lt; k in the algorithm, whereas the event k ≤ K and (in this event) the set ∆ i k are entirely determined by data involved in rounds k ′ &lt; k . Thus, even conditioned on the event that k ≤ K and and the set ∆ i k , the data set S 1 k remains conditionally i.i.d.P . Therefore, letting A k denote the set A as defined in Lemma 9 with D = ∆ i k , applying the uniform Bernstein inequality (Lemma 7 in Appendix D) with this A k under the conditional distribution given the event k ≤ K and the set ∆ i k implies that, with conditional probability at least 1 -δ k given the event k ≤ K and the set ∆ i k , it holds that ∀ A,B ∈ A k ,

<!-- formula-not-decoded -->

13 Since this work focuses on binary classification, V (3) k -1 can equivalently be stated as { Maj( f, g, h ) : f, g ∈ V k -1 , h ∈ C } , where Maj( f, g, h )( x ) = 1 [ f ( x ) + g ( x ) + h ( x ) ≥ 2] is the majority vote function. The definition of V (3) k -1 above expresses a more-general form, which, as we discuss in Section G, also extends to multiclass classification.

By the law of total probability, on an event E 0 ,k of probability at least 1 -δ k , if k ≤ K (and thus ∆ i k and A k are defined) then (17) holds ∀ A,B ∈ A k .

̸

In particular, on the event E 0 ,k , supposing k ≤ K , if we consider any h, h ′ ∈ V (3) k -1 , then for the sets A = ER( h ) \ ∆ i k ∈ A k and B = ER( h ′ ) \ ∆ i k ∈ A k , we may note that the symmetric difference A ⊕ B = (ER( h ) ⊕ ER( h ′ )) \ ∆ i k = ( { h = h ′ } × { 0 , 1 } ) \ ∆ i k , so that together with (11) of Lemma 9, (17) implies

̸

<!-- formula-not-decoded -->

̸

To arrive at the claim in (14), we merely note that for any f, g, f ′ , g ′ ∈ V k -1 and h, h ′ ∈ C , letting DL( f, g, h ) := f 1 { f = g } + h 1 { f = g } and DL( f ′ , g ′ , h ′ ) := f ′ 1 { f ′ = g ′ } + h ′ 1 { f ′ = g ′ } , for any x ∈ X \ D k -1 , we have g ( x ) = f ( x ) = f ′ ( x ) = g ′ ( x ) , so that DL( f, g, h )( x ) = f ( x ) = f ′ ( x ) = DL( f ′ , g ′ , h ′ )( x ) . Thus, any h, h ′ ∈ V (3) k -1 have h ( x ) = h ′ ( x ) for all x / ∈ D k -1 , and therefore

̸

<!-- formula-not-decoded -->

so that (14) follows from (18). To unify the discussion below, for any k ∈ { 1 , . . . , N + 1 } with probability zero of k ≤ K , also denote by E 0 ,k the event (of probability one) that k &gt; K .

Turning now to the claims in (15) and (16), consider any ( k, i ) having non-zero probability that k ∈ K and i ∈ I k . Note that, since S 3 k,i is a data set of size m k , allocated from the remaining unused unlabeled data upon reaching Step 5 with values ( k, i ) (noting this can happen at most once in the algorithm), the samples in S 3 k,i are conditionally i.i.d.P given k ∈ K and i ∈ I k , and moreover, S 3 k,i is conditionally independent of ∆ i given the events that k ∈ K and i ∈ I k . In the event that k ∈ K and i ∈ I k , let A k,i denote the set A as defined in Lemma 9 with D = ∆ i . Recalling again our definition of C = 11 10 and C ′′ ≥ 32 C 5 ( C C -1 ) 2 , note that for α = 1 -1 C , (11) of Lemma 9 implies 2 2

̸

ε ( m k , δ k ; A k,i ) &lt; ε k C ′′ &lt; α 4 ε k +2 &lt; α 4 ε k +1 . Therefore, applying Lemma 7 of Appendix D under the conditional distribution given the events that k ∈ K and i ∈ I k and the set ∆ i , we have that with conditional probability at least 1 -δ k , ∀ f, g ∈ C , the set ( { f = g } \ ∆ i ) ×{ 0 , 1 } ∈ A k,i satisfies

̸

<!-- formula-not-decoded -->

and

̸

<!-- formula-not-decoded -->

̸

̸

By the law of total probability, there is an event E 0 ,k,i of probability at least 1 -δ k , on which, if k ∈ K and i ∈ I k , then the above inequalities hold ∀ f, g ∈ C . To unify cases, for any ( k, i ) with k ≤ N and i ≤ 1 /ε k +3 having probability zero of satisfying k ∈ K and i ∈ I k , also define E 0 ,k,i as the event (of probability one) that either k / ∈ K or i / ∈ I k .

We have thus established (14) for all k ≤ K and (15 - 16) for all k ∈ K and i ∈ I k with i ≤ 1 /ε k +3 , on the event E 0 := ( ⋂ k ≤ N +1 E 0 ,k ) ∩ ⋂ k ≤ N ⋂ i ≤ 1 /ε k +3 E 0 ,k,i . By the union bound, E 0 fails with probability at most

<!-- formula-not-decoded -->

where the equality follows from our definition of δ k = δε 2 k +3 72 (from Lemma 9) and the last inequality follows from our choice of C = 11 10 .

Finally, we argue that, on the event E 0 , for any k ∈ K , the maximum value of i ∈ I k satisfies i ≤ 1 /ε k +3 . We argue this by induction. Specifically, we will argue that, for any k ∈ K and i ∈ I k , P X ( X \ ∆ i ) ≤ 1 -iε k +3 . For the purpose of induction, suppose that for some k ∈ K , we have P X ( X \ ∆ i k ) ≤ 1 -i k ε k +3 (which is trivially satisfied for k = 1 , since i k = 0 , which can therefore serve as a base case for induction). Taking this i k as a base case for a further nested

̸

̸

induction on i ∈ I k (noting that i k is the minimum element of I k ), suppose that for some i ∈ I k we have P X ( X \ ∆ i ) ≤ 1 -iε k +3 . Since probabilities are non-negative, this necessarily implies i ≤ 1 /ε k +3 . Then note that, if i is not the maximal element of I k , the algorithm augments ∆ i in Step 7, so that ∆ i +1 = ∆ i ∪ { f = g } for ( f, g ) defined in Step 6. By the criterion in Step 5, we further know that ˆ P S 3 k,i ( { f = g } \ ∆ i ) &gt; ε k +2 . Since i ≤ 1 /ε k +3 , the event E 0 implies (15) holds, which therefore implies P X (∆ i +1 \ ∆ i ) = P X ( { f = g } \ ∆ i ) &gt; ε k +3 , so that P X ( X \ ∆ i +1 ) = P X ( X \ ∆ i ) -P X (∆ i +1 \ ∆ i ) &lt; 1 -( i +1) ε k +3 , thus extending the inductive hypothesis. By the principle of induction, this establishes that P X ( X \ ∆ i ) ≤ 1 -iε k +3 for every i ∈ I k . In particular, returning to the induction on k , in the event that this k is not the maximal element of K , we have i k +1 ∈ I k , so that P X ( X \ ∆ i k +1 ) ≤ 1 -i k +1 ε k +3 ≤ 1 -i k +1 ε ( k +1)+3 , which therefore extends the inductive hypothesis for k . By the principle of induction, we have thus established that every k ∈ K and i ∈ I k satisfy P X ( X \ ∆ i ) ≤ 1 -iε k +3 . In particular, since probabilities are non-negative, this immediately implies any such ( k, i ) satisfy i ≤ 1 /ε k +3 , as claimed. Thus, on the event E 0 , we have established all of the claimed inequalities: (14) for all k ∈ { 1 , . . . , K } , and (15 - 16) for all k ∈ K and i ∈ I k , which further satisfy max I k ≤ 1 ε k +3 . ■

̸

The following is an obvious implication of Lemma 10, which will be useful to state explicitly for later reference.

Lemma 11. On the event E 0 , for every k ∈ K and i ∈ I k , if the algorithm reaches Step 6 with these values ( k, i ) , then for f, g as defined there,

̸

<!-- formula-not-decoded -->

̸

Moreover, on the event E 0 , every k ∈ { 1 , . . . , K } with ∆ i k = ∅ satisfies P X (∆ i k ) &gt; ε k +2 .

̸

Proof. By the condition in Step 5, if the algorithm reaches Step 6 then ˆ P S 3 k,i ( { f = g } \ ∆ i ) &gt; ε k +2 . By (15) of Lemma 10, on the event E 0 , this implies P X ( { f = g } \ ∆ i ) &gt; ε k +3 .

̸

̸

Turning now to the second claim, suppose again that E 0 occurs, and first note that this claim is trivially satisfied if ∆ i K = ∅ . To address the remaining case, suppose ∆ i K = ∅ , and consider the minimum value k ′ ∈ { 1 , . . . , K } for which ∆ i k ′ = ∅ . By definition we have ∆ i 1 = ∆ 0 = ∅ , which implies we must have k ′ ≥ 2 . By minimality of k ′ , we also know that the algorithm reaches Step 6 at least once during round k = k ′ -1 of the ' For ' loop, in particular with i = i k . Thus, letting ( f, g ) be as defined in Step 6 for these values ( k, i ) = ( k ′ -1 , i k ′ -1 ) , by the first claim in the lemma, we have P X (∆ i k ′ ) ≥ P X ( f = g ) = P X ( { f = g } \ ∆ i k ′ -1 ) &gt; ε k ′ +2 . Thus, since ∆ i k ′′ is non-decreasing in k ′′ , and minimality of k ′ implies all k ′′ with ∆ i k ′′ = ∅ have k ′′ ≥ k ′ , we conclude that every k ′′ ∈ { 1 , . . . , K } with ∆ i k ′′ = ∅ satisfies P X (∆ i k ′′ ) ≥ P X (∆ i k ′ ) &gt; ε k ′ +2 ≥ ε k ′′ +2 . ■

̸

Next we state a bound on the diameters of V k -1 and V (3) k -1 , useful for Lemmas 15, 20, and 22.

Lemma 12. On the event E 0 , for every k ∈ { 1 , . . . , K } ,

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Proof. Throughout this proof, we suppose the event E 0 holds. The inequality (19) is trivially satisfied for k = 1 , recalling that V 0 = C and ε 1 = C 0 = 1 . For the remaining case, fix any k ′ ∈ { 2 , . . . , K } and consider the round k = k ′ -1 in the ' For ' loop (noting that, by definition of K , we have k = k ′ -1 ∈ K regardless of whether K = N +1 or K ≤ N ). Since k +1 = k ′ ≤ K , we know the algorithm reaches Step 8 in round k (i.e., it does not terminate early in Step 4 during round k ). In particular, this means the condition in Step 5 fails for the value i = i k +1 = max I k : that is, max f,g ∈ V k ˆ P S 3 k,i ( { f = g } \ ∆ i k +1 ) ≤ ε k +2 . By (16) of Lemma 10, this implies

̸

<!-- formula-not-decoded -->

̸

This completes the proof of (19) for every k ∈ { 1 , . . . , K } .

̸

̸

̸

̸

̸

̸

To show (20), let k ∈ { 1 , . . . , K } , and for any f, g ∈ V k -1 and h ∈ C , denote by DL( f, g, h ) := f 1 { f = g } + h 1 { f = g } ∈ V (3) k -1 . Note that for any f, g, f ′ , g ′ ∈ V k -1 , h, h ′ ∈ C , and x ∈ X , if g ( x ) = f ( x ) = f ′ ( x ) = g ′ ( x ) , then DL( f, g, h )( x ) = DL( f ′ , g ′ , h ′ )( x ) . Therefore,

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

where the last inequality is by (19). This completes the proof of the lemma.

<!-- formula-not-decoded -->

The following Lemmas 13 and 14 concern concentration of empirical errors in the set S 2 k ∩ ∆ i k , which will be useful in establishing guarantees on the quality of ˆ h k (in Lemma 15) and of the functions in V k (in Lemmas 16 and 17) below. We first need to argue that the ˆ p k quantities approximate P X (∆ i k ) , which leads to the data sets S 2 k being of appropriate size for concentration of empirical error rates.

Lemma13. There is an event E 1 of probability at least 1 -δ 4 such that, on E 0 ∩ E 1 , ∀ k ∈ { 1 , . . . , K } , the quantity ˆ p k := 2 ˆ P S 4 k (∆ i k ) (as defined above) satisfies

<!-- formula-not-decoded -->

Proof. Consider any k ∈ { 1 , . . . , N +1 } having non-zero probability that k ≤ K . Note that the execution of the algorithm does not depend on S 4 k at any time prior to Step 2 of round k (or Step 9 if k = N +1 ), supposing this step is even reached in the algorithm (i.e., k ≤ K ). Thus, since the event that k ≤ K and the set ∆ i k are both completely determined by events occurring prior to this first time the examples in S 4 k are used by the algorithm, we have that S 4 k is independent of these. Thus, conditioned on the event that k ≤ K and on the random variable ∆ i k , we have that for the sequence of m k examples ( X t , Y t ) comprising S 4 k , the corresponding sequence of indicator random variables 1 [ X t ∈ ∆ i k ] are conditionally independent Bernoulli( P X (∆ i k )) random variables. Therefore, applying a multiplicative Chernoff bound (Lemma 6 of Appendix D) under the conditional distribution given the event k ≤ K and the random variable ∆ i k , together with the law of total probability, we have that on an event E 1 ,k of probability at least 1 -δε k 44 , if k ≤ K , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For simplicity, for any k ∈ { 1 , . . . , N +1 } having probability zero of k ≤ K , simply define E 1 ,k as the event of probability one that k &gt; K , so that the above claim also holds (vacuously) for such values k . Define an event E 1 = ⋂ N +1 k =1 E 1 ,k , and note that, by the union bound, E 1 occurs with probability at least 1 -∑ N +1 k =1 δε k 44 ≥ 1 -δ 4 .

̸

k k 2 P X (∆ i k ) &gt; 2 ε k +2 &gt; 6 m k ln ( 88 δε k ) , so that the right hand side of (23) equals 2 P X (∆ i k ) and hence ( )

We now argue these inequalities further imply the simpler inequalities stated in (21), on the additional event E 0 . Suppose the event E 0 ∩ E 1 holds, and let k ∈ { 1 , . . . , K } . If ∆ i k = ∅ , (21) trivially holds since ˆ p k = 0 = P X (∆ i k ) . To address the remaining case, suppose ∆ i k = ∅ . By the final claim in Lemma 11, we have P X (∆ i k ) &gt; ε k +2 . Also note that, by definition of m k (and recalling | C | ≥ 2 , which implies d ≥ 1 ), we have 8 m ln ( 88 δε ) &lt; ε k +2 . In particular, these imply

ˆ p k ≤ 4 P X (∆ i k ) . Moreover, since 8 m k ln 88 δε k &lt; ε k +2 &lt; P X (∆ i k ) , the ' max ' on the right hand side of (22) cannot be achieved by the second term (as it is smaller than the quantity on the left hand side), so it must be achieved by the first term. Therefore, P X (∆ i k ) ≤ 2 ˆ P S 4 k (∆ i k ) = ˆ p k . ■

Using Lemma 13 to bound the size of the data set S 2 k (which is based on ˆ p k ), we are now ready to establish a concentration inequality for the error rates in the ∆ i k region in the following lemma.

Lemma 14. There is an event E 2 of probability at least 1 -δ 4 such that, on the event E 0 ∩ E 1 ∩ E 2 , ∀ k ∈ { 1 , . . . , K } ,

<!-- formula-not-decoded -->

Proof. Consider any k ∈ { 1 , . . . , N +1 } having non-zero probability of k ≤ K . Supposing k ≤ K occurs, define a collection of sets A ′ k := { ER( h ) ∩ ∆ i k : h ∈ C } . Note that VC( A ′ k ) ≤ d (which is immediate from the definition of VC dimension). We aim to apply Lemma 8 of Appendix D, a refinement of the uniform convergence bound of Talagrand (1994), which accounts for an envelope set D ⊇ ⋃ A ′ k ; specifically, we instantiate the various sets and variables in Lemma 8 to be Z = X × { 0 , 1 } , n = m ′ k , A = A ′ k , envelope set D = ∆ i k , data set Z = S 2 k , and confidence parameter δ/ (4(3+ N -k ) 2 ) , and we apply the lemma under the conditional distribution given the event k ≤ K and given the random variables ∆ i k and m ′ k . Since the event that k ≤ K , and the random variables ∆ i k and m ′ k , are all completely determined by examples allocated to data sets before allocating examples to the data set S 2 k , we may note that the m ′ k examples comprising S 2 k are conditionally independent P -distributed random variables given the event that k ≤ K and given the random variables ∆ i k and m ′ k . Thus, applying Lemma 8 of Appendix D under the conditional distribution given the event that k ≤ K and given the random variables ∆ i k and m ′ k , together with the law of total probability, we have that on an event E 2 ,k of probability at least 1 -δ 4(3+ N -k ) 2 , if k ≤ K and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For simplicity, for any k ∈ { 1 , . . . , N +1 } having zero probability of k ≤ K , let E 2 ,k denote the event of probability one that k &gt; K . Finally, define E 2 = ⋂ N +1 k =1 E 2 ,k , and note that, by the union bound, E 2 holds with probability at least 1 -∑ N +1 k =1 δ 4(3+ N -k ) 2 ≥ 1 -δ 4 ∑ ∞ j =2 1 j 2 ≥ 1 -δ 4 .

̸

Now suppose the event E 0 ∩ E 1 ∩ E 2 occurs, and consider any k ∈ { 1 , . . . , K } . If ∆ i k = ∅ then (24) holds trivially since the left hand side of (24) is then zero. To address the remaining case, suppose ∆ i k = ∅ . By the final claim in Lemma 11, we have P X (∆ i k ) &gt; ε k +2 . Moreover, by Lemma 13 we have ˆ p k ≥ P X (∆ i k ) . Recalling m ′ k := ⌈ C ′′ c 2 1 ˆ p k ε 2 k ( d +log ( 4(3+ N -k ) 2 δ ))⌉ , these imply

<!-- formula-not-decoded -->

where the last inequality follows from c 1 ≥ 1 and C ′′ ≥ 18 C 4 . Thus, 9 m ′ k ln ( 16(3+ N -k ) 2 δ ) &lt; ε k +2 &lt; P X (∆ i k ) . By the definition of E 2 , it follows that (25) holds. Moreover, since ˆ p k ≥ P X (∆ i k ) , we have that m ′ k ≥ C ′′ c 2 1 P X (∆ i k ) ε 2 k ( d +log ( 4(3+ N -k ) 2 δ )) , so that the right hand side of (25) is at most ε k √ C ′′ , thus establishing (24). ■

Combining the concentration inequality from Lemma 14 with (14) of Lemma 10 together with (20) of Lemma 12 yields a concentration inequality for the differences ˆ er 1,2 k ( h ) -ˆ er 1,2 k ( h ′ ) among h, h ′ ∈ V k -1 , recalling the definition from (1):

<!-- formula-not-decoded -->

In fact, the implication is stronger than this, admitting functions h, h ′ ∈ V (4) k -1 . In particular, for any k ∈ { 1 , . . . , K } , note that V (4) k -1 in (2) can equivalently be defined as

<!-- formula-not-decoded -->

The following lemma provides a concentration inequality for ˆ er 1,2 k ( h ) -ˆ er 1,2 k ( h ′ ) among functions h, h ′ ∈ V (4) k -1 .

Lemma 15. On the event E 0 ∩ E 1 ∩ E 2 , for every k ∈ { 1 , . . . , K } we have

<!-- formula-not-decoded -->

recalling that C ′ := √ C ′′ 16 . Moreover, (26) implies

<!-- formula-not-decoded -->

Proof. Suppose the event E 0 ∩ E 1 ∩ E 2 holds and consider any k ∈ { 1 , . . . , K } . Note that for any h = h 1 1 X\ ∆ i k + h 2 1 ∆ i k ∈ V (4) k -1 we have er P ( h ) = P (ER( h 1 ) \ ∆ i k ) + P (ER( h 2 ) ∩ ∆ i k ) and ˆ er 1,2 k ( h ) = ˆ P S 1 k (ER( h 1 ) ∩ D k -1 \ ∆ i k ) + ˆ P S 2 k (ER( h 2 ) ∩ ∆ i k ) .

Consider any h 1 , h ′ 1 ∈ V (3) k -1 and any h 2 , h ′ 2 ∈ C , and let h = h 1 1 X\ ∆ i k + h 2 1 ∆ i k and h ′ = h ′ 1 1 X\ ∆ i k + h ′ 2 1 ∆ i k . By Lemma 14, we have ∀ h ′′ 2 ∈ { h 2 , h ′ 2 } ,

<!-- formula-not-decoded -->

̸

Additionally, since (20) of Lemma 12 implies P X ( { h 1 = h ′ 1 } \ ∆ i k ) ≤ 3 ε k , the inequality (14) of Lemma 10 implies

<!-- formula-not-decoded -->

where the last inequality follows from C ′′ ≥ 14 . Combining these with the triangle inequality (namely, | ((ˆ a + ˆ b ) -(ˆ a ′ + ˆ b ′ )) -(( a + b ) -( a ′ + b ′ )) | ≤ | (ˆ a -ˆ a ′ ) -( a -a ′ ) | + | ˆ b -b | + | ˆ b ′ -b ′ | ) yields that

<!-- formula-not-decoded -->

To see that (26) implies (27), note that, by the definition of ˆ h k in (3), h = ˆ h k has minimal ˆ er 1,2 k ( h ) among all h ∈ V (4) k -1 : that is, ∀ h ∈ V (4) k -1 , ˆ er 1,2 k ( ˆ h k ) -ˆ er 1,2 k ( h ) ≤ 0 . Together with (26), this implies that ∀ h ∈ V (4) k -1 , er P ( ˆ h k ) -er P ( h ) ≤ ˆ er 1,2 k ( ˆ h k ) -ˆ er 1,2 k ( h ) + ε k 4 C ′ ≤ ε k 4 C ′ . ■

In particular, Lemma 15 immediately implies the following lemma concerning the quality of the functions h in V k .

Lemma 16. On the event E 0 ∩ E 1 ∩ E 2 , ∀ k ∈ K , ∀ h ∈ V k -1 , the following implications hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Proof. Suppose the event E 0 ∩ E 1 ∩ E 2 occurs and consider any k ∈ K and h ∈ V k -1 . In particular, note that we also have h ∈ V (4) k -1 , since letting f = g = h , we have h = f 1 { f = g } + h 1 { f = g } ∈ V (3) k -1 , and thus h = h 1 X\ ∆ i k + h 1 ∆ i k ∈ V (4) k -1 .

If h ∈ V k , then by definition of V k in Step 3 we have ˆ er 1,2 k ( h ) -ˆ er 1,2 k ( ˆ h k ) ≤ ε k C ′ . Together with (26) of Lemma 15, this implies er P ( h ) -er P ( ˆ h k ) ≤ ε k C ′ + ε k 4 C ′ = 5 ε k 4 C ′ , which establishes (29).

On the other hand, if er P ( h ) -er P ( ˆ h k ) ≤ 3 ε k 4 C ′ , then (26) of Lemma 15 implies ˆ er 1,2 k ( h ) -ˆ er 1,2 k ( ˆ h k ) ≤ 3 ε k 4 C ′ + ε k 4 C ′ = ε k C ′ . Thus, any such h is retained in V k , which establishes (30). ■

The main implication of Lemma 16 pertains to the early stopping case in Step 4, which we turn to next. Recall h ⋆ denotes an (arbitrary) concept in C with er P ( h ⋆ ) &lt; inf h ∈ C er P ( h ) + ε 10 4 . In the following lemma, in addition to arguing that the predictor ˆ h returned in Step 4 has low excess error rate compared to h ⋆ (in fact, negative ), this lemma also reveals a second major role of this early stopping case: it ensures that on all rounds k in which the algorithm does not terminate in Step 4, we retain h ⋆ ∈ V k .

Lemma 17. On the event E 0 ∩ E 1 ∩ E 2 , the following implications hold for every k ∈ K :

- If A avid does not return in Step 4 on round k , then h ⋆ ∈ V k .
- If A avid returns in Step 4 on round k , then er P ( ˆ h k ) &lt; er P ( h ⋆ ) .

̸

Proof. Suppose the event E 0 ∩ E 1 ∩ E 2 occurs. We will prove the first claim by induction on k . As a base case, we trivially have h ⋆ ∈ C = V 0 . Now, for the purpose of induction, let k ∈ K be such that h ⋆ ∈ V k -1 . Also note (as discussed in the proof of Lemma 16) that this also implies h ⋆ ∈ V (4) k -1 . If h ⋆ / ∈ V k , then by (30) of Lemma 16, we have er P ( h ⋆ ) -er P ( ˆ h k ) &gt; 3 ε k 4 C ′ . In particular, this implies that if V k = ∅ , then together with (26) of Lemma 15, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows from ε 10 4 &lt; 2 ε N 10 4 ≤ ε k 4 C ′ . Thus, either V k = ∅ or min h ∈ V k ˆ er 1,2 k ( h ) -ˆ er 1,2 k ( ˆ h k ) &gt; ε k 4 C ′ , so that either way the algorithm will return in Step 4 in this case. Therefore, if the algorithm does not return in Step 4 on round k , it must be that h ⋆ ∈ V k . This completes the proof of the first claim, by the principle of induction.

Finally, we turn to the second claim. Suppose, for some k ∈ K , the algorithm returns in Step 4 on round k . In particular, either k = 1 , in which case h ⋆ ∈ C = V k -1 , or k &gt; 1 , in which case (since the algorithm did not return in Step 4 on round k -1 ) the first claim in the lemma implies h ⋆ ∈ V k -1 . Again note that this also implies h ⋆ ∈ V (4) k -1 . If h ⋆ / ∈ V k , then (30) of Lemma 16 implies er P ( h ⋆ ) &gt; er P ( ˆ h k ) + 3 ε k 4 C ′ . Otherwise, if h ⋆ ∈ V k , the condition in Step 4 implies ˆ er 1,2 k ( h ⋆ ) -ˆ er 1,2 k ( ˆ h k ) &gt; ε k 4 C ′ . Together with (26) of Lemma 15, this implies

<!-- formula-not-decoded -->

Thus, in either case, we have er P ( h ⋆ ) &gt; er P ( ˆ h k ) , which establishes the second claim.

■

Lemmas 15, 16, and 17 together have a particularly nice implication, which, although not strictly needed for the proof of Theorem 5, is worth noting (and will be useful in Appendix F). Specifically, we have the following corollary.

Corollary 18. On the event E 0 ∩ E 1 ∩ E 2 , ∀ k ∈ K ,

<!-- formula-not-decoded -->

Proof. Suppose the event E 0 ∩ E 1 ∩ E 2 occurs and consider any k ∈ K . Since k -1 &lt; K , Lemma 17 implies h ⋆ ∈ V k -1 in the case k ≥ 2 , while the case k = 1 has h ⋆ ∈ V 0 by definition of V 0 = C . Together with (27) of Lemma 15, this implies er P ( ˆ h k ) ≤ er P ( h ⋆ ) + ε k 4 C ′ . Combined with (29) of Lemma 16, we have that every h ∈ V k satisfies er P ( h ) ≤ er P ( ˆ h k ) + 5 ε k 4 C ′ ≤ er P ( h ⋆ ) + 3 ε k 2 C ′ . ■

At this point, we may note that Lemmas 15 and 17 together completely address the error guarantee for the ˆ h returned by A avid , on the event E 0 ∩ E 1 ∩ E 2 . as summarized in the following lemma.

Lemma 19. On the event E 0 ∩ E 1 ∩ E 2 , A avid eventually terminates, and the function ˆ h it returns satisfies er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε .

Proof. Suppose the event E 0 ∩ E 1 ∩ E 2 occurs. If the algorithm terminates in Step 4 in some round k ∈ K , by definition we have ˆ h = ˆ h k , and thus Lemma 17 implies er P ( ˆ h ) &lt; er P ( h ⋆ ) &lt; inf h ∈ C er P ( h ) + ε 10 4 . On the other hand, if the algorithm does not return in Step 4 on any round k ∈ K , then we have K = N + 1 (recalling Lemma 10 implies the algorithm eventually terminates), so that by definition ˆ h = ˆ hN +1 . Since in this case Lemma 17 implies h ⋆ ∈ V N (and hence h ⋆ ∈ V (4) N ), (27) of Lemma 15 implies er P ( ˆ h ) = er P ( ˆ hN +1 ) ≤ er P ( h ⋆ ) + ε N +1 4 C ′ &lt; inf h ∈ C er P ( h ) + ε 10 4 + ε 8 C ′ &lt; inf h ∈ C er P ( h ) + ε . ■

With the analysis of error guarantees complete, we turn now to establishing the bound Q ( ε, δ ; β ) on the number of queries, as claimed in Theorem 5. This will be comprised of two main parts. First, we

argue that the set ∆ i k never grows too large: specifically, recalling β := inf h ∈ C er P ( h ) , Lemma 20 will establish that P X (∆ i k ) = O ( β ) , which in turn allows us to bound the number of queries in S 2 k ∩ ∆ i k on each round (in the proof of Lemma 23). Second, in the proof of Lemma 22, we bound P X ( D k -1 \ ∆ i k ) ≤ s ε k , by reasoning in terms of the disagreement coefficient (Hanneke, 2007b), relating the latter to the star number via a result of Hanneke and Yang (2015). This in turn allows us to bound the number of queries in S 1 k ∩ D k -1 \ ∆ i k on each round (in the proof of Lemma 23).

We begin with the first of these parts, stated in the following lemma. We remark that this lemma plays a special role in constraining the allowed values of the constant C , as the argument breaks down if C is taken too large. On the other hand, the proof also reveals that it is possible to decrease the factor ' 5 ' in this lemma to any value c &gt; 2 by taking C &gt; 1 appropriately close to 1 and by an appropriately large choice of the constant C ′′ (and hence C ′ ). See footnote 11 for further discussion.

Lemma 20. On the event E 0 ∩ E 1 ∩ E 2 , for all k ∈ { 1 , . . . , K } and i ∈ I k ,

<!-- formula-not-decoded -->

Proof. We will argue that, on E 0 ∩ E 1 ∩ E 2 , for any h 0 ∈ C , each region ∆ i +1 \ ∆ i (defined in Step 7) satisfies

<!-- formula-not-decoded -->

so that each addition to ∆ i 'chops off' a piece of ER( h 0 ) of measure proportional to the increase in measure P X (∆ i +1 ) -P X (∆ i ) = P X (∆ i +1 \ ∆ i ) . The claim in the lemma then follows immediately from (31), since it holds trivially for i = 0 (recalling ∆ 0 = ∅ ), and if any k ∈ { 1 , . . . , K } and i ∈ I k has i ≥ 1 , then applying (31) inductively yields

<!-- formula-not-decoded -->

Taking the infimum over all h 0 ∈ C then implies the lemma.

We proceed now with the formal proof of (31). Suppose the event E 0 ∩ E 1 ∩ E 2 occurs, and for the purpose of analyzing the increases ∆ i +1 \ ∆ i of the ∆ i set (which only occur in Step 7), consider any k ∈ K and any i ∈ I k with i &lt; max I k (equivalently, the algorithm reaches Step 7 with this ( k, i ) ). Let ( f, g ) be as defined in Step 6 for this ( k, i ) so that ∆ i +1 \ ∆ i = { f = g } \ ∆ i .

̸

̸

Note that { f = g } × { 0 , 1 } = (ER( f ) ∩ { f = g } ) ∪ (ER( g ) ∩ { f = g } ) , so that (lower-bounding 'max' by 'average')

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

̸

where in fact the last inequality holds with equality (since, for { 0 , 1 } labels, (ER( f ) ∩{ f = g } ) and (ER( g ) ∩ { f = g } ) are disjoint). Thus, ∃ f ′ ∈ { f, g } with

̸

̸

<!-- formula-not-decoded -->

̸

̸

Let h ′ = f ′ 1 { f = g }\ ∆ i k + h 0 1 { f = g }\ ∆ i k + f ′ 1 ∆ i k and note that h ′ ∈ V (4) k -1 . Also recall that ˆ er 1,2 k ( ˆ h k ) = min h ∈ V (4) k -1 ˆ er 1,2 k ( h ) , and hence ˆ er 1,2 k ( h ′ ) ≥ ˆ er 1,2 k ( ˆ h k ) . Since we also have f ′ ∈ V (4) k -1 (as discussed in the proof of Lemma 16), Lemma 15 implies

<!-- formula-not-decoded -->

where the last inequality is due to f ′ ∈ { f, g } ⊆ V k , recalling the definition of V k in Step 3. Moreover, by definition of f ′ and h ′ , we have

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

Equivalently: P (ER( h 0 ) ∩ { f = g } \ ∆ i k ) ≥ 1 2 P X ( { f = g } \ ∆ i k ) -( er P ( f ′ ) -er P ( h ′ )) . Combining this with (32), we conclude that

̸

̸

̸

<!-- formula-not-decoded -->

Also note that, since ∆ i ⊇ ∆ i k , we have

̸

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

Moreover, again since ∆ i ⊇ ∆ i k , we have

̸

<!-- formula-not-decoded -->

Combining (34), (35), and (33) yields that

̸

<!-- formula-not-decoded -->

̸

̸

Lemma 12 and the fact that f, g ∈ V k ⊆ V k -1 imply P X ( { f = g } \ ∆ i k ) ≤ ε k , and Lemma 11 implies P X ( { f = g } \ ∆ i ) &gt; ε k +3 . Together, we have

̸

<!-- formula-not-decoded -->

̸

̸

Additionally, again since P X ( { f = g } \ ∆ i ) &gt; ε k +3 , we have that 5 ε k 4 C ′ &lt; 5 C 3 4 C ′ P X ( { f = g } \ ∆ i ) . Combining these inequalities with (36), and recalling ∆ i +1 \ ∆ i = { f = g } \ ∆ i , yields that

̸

<!-- formula-not-decoded -->

Recalling that C ′ = √ C ′′ 16 = 25 C 3 16 -10 C 3 , we have C 3 2 + 5 C 3 4 C ′ = 4 5 , so that the right hand side above equals 1 5 P X (∆ i +1 \ ∆ i ) , which establishes (31). ■

Next we turn to the second part of the argument outlined above: bounding the number of queries in the sets S 1 k ∩ D k -1 \ ∆ i k . We begin by stating a known fact, due to Hanneke and Yang (2015, Theorem 10): namely, that the disagreement coefficient (Hanneke, 2007b) is upper bounded by the star number (Hanneke and Yang, 2015) (indeed, Theorem 10 of Hanneke and Yang, 2015 shows the relation is even sharp in the worst case over h and distributions P ′ X ).

̸

Lemma 21 (Hanneke and Yang, 2015) . For any measurable h : X → { 0 , 1 } , any distribution P ′ X on X , and any r &gt; 0 , defining the r -ball centered at h as B P ′ X ( h, r ) := { h ′ ∈ C : P ′ X ( h ′ = h ) ≤ r } , it holds that

<!-- formula-not-decoded -->

Toward bounding the number of queries in the sets S 1 k ∩ D k -1 \ ∆ i k in the algorithm, the following lemma establishes a bound on P X ( D k -1 \ ∆ i k ) by a straightforward application of Lemma 21 to the conditional probabilities P X ( D k -1 |X \ ∆ i k ) , in combination with a diameter bound supplied by (19) of Lemma 12.

Lemma 22. On the event E 0 ∩ E 1 ∩ E 2 , for every k ∈ { 1 , . . . , K } , P X ( D k -1 \ ∆ i k ) ≤ s ε k .

Proof. Suppose the event E 0 ∩ E 1 ∩ E 2 holds and consider any k ∈ { 1 , . . . , K } . If P X ( X\ ∆ i k ) = 0 , we trivially have that P X ( D k -1 \ ∆ i k ) = 0 ≤ s ε k . To address the remaining case, suppose P X ( X \ ∆ i k ) &gt; 0 , and denote by P k := P X ( ·|X \ ∆ i k ) . By (19) of Lemma 12 we have

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

In particular, since k -1 &lt; K , Lemma 17 implies h ⋆ ∈ V k -1 in the case k ≥ 2 , while the case k = 1 has h ⋆ ∈ V 0 by definition of V 0 = C . Thus, the above inequality implies

<!-- formula-not-decoded -->

Together with Lemma 21, this implies

<!-- formula-not-decoded -->

We therefore have that

<!-- formula-not-decoded -->

We are now ready to state a lemma bounding the total number of queries in the algorithm, by a combination of Lemmas 13, 20, and 22 together with a multiplicative Chernoff bound argument. For convenience, this lemma also supplies an upper bound on the sizes m ′ k of the data sets S 2 k , which will be of further use when establishing the bound M ( ε, δ ; β ) on the total number of unlabeled examples sufficient for the execution of the algorithm (Lemma 24 below). Specifically, in the following lemma, for any k ∈ { 1 , . . . , N +1 } , denote by

<!-- formula-not-decoded -->

Lemma 23. There is an event E 3 of probability at least 1 -δ 4 , such that on ⋂ 3 j =0 E j , ∀ k ∈ { 1 , . . . , K } , the following claims hold: m ′ k ≤ m ′ k ,

<!-- formula-not-decoded -->

Moreover, we have ∑ N +1 k =1 m ′ k ≤ M 2 , and the total number of queries by A avid is at most Q ( ε, δ ; β ) , where

<!-- formula-not-decoded -->

̸

Proof. By Lemma 13, on E 0 ∩ E 1 ∩ E 2 , ∀ k ∈ { 1 , . . . , K } , P X (∆ i k ) ≤ ˆ p k ≤ 4 P X (∆ i k ) . Recall the definition of m ′ k := ⌈ C ′′ c 2 1 ˆ p k ε 2 k ( d +log ( 4(3+ N -k ) 2 δ ))⌉ . Thus, if ∆ i k = ∅ , we have ˆ p k = 0 , hence m ′ k = 0 , and the implication m ′ k ≤ m ′ k trivially follows. Otherwise, on E 0 ∩ E 1 ∩ E 2 , if ∆ i k = ∅ , the final claim in Lemma 11 implies P X (∆ i k ) &gt; ε k +2 , so that together m ′ k is at most

<!-- formula-not-decoded -->

Since Lemma 20 implies P X (∆ i k ) ≤ 5 β on E 0 ∩ E 1 ∩ E 2 , we conclude that m ′ k ≤ m ′ k .

We next turn to establishing (39). Consider any k ∈ { 1 , . . . , N +1 } having non-zero probability that k ≤ K . Given that k ≤ K , note that V k -1 and ∆ i k have no dependence on S 1 k , so that the samples in S 1 k are conditionally i.i.d.P given the event that k ≤ K and given the random variables V k -1 and ∆ i k . Therefore, applying a multiplicative Chernoff bound (Lemma 6 of Appendix D) under the conditional distribution given the event k ≤ K and the random variables V k -1 and ∆ i k , with conditional probability at least 1 -δ 8 k ( k +1) ,

<!-- formula-not-decoded -->

In particular, by the law of total probability, this implies that for every k ∈ { 1 , . . . , N +1 } , with probability at least 1 -δ 8 k ( k +1) , if k ≤ K then (42) holds. Letting E ′ 3 denote the event that (42) holds for every k ∈ { 1 , . . . , K } , by the union bound, E ′ 3 holds with probability at least 1 -δ 8 . Combining (42) with Lemma 22, we have that on the event E 0 ∩ E 1 ∩ E 2 ∩ E ′ 3 , ∀ k ∈ { 1 , . . . , K } ,

<!-- formula-not-decoded -->

where the rightmost inequality follows from recalling m k := ⌈ 300 C ′′ c 0 ε k ( d log ( C ′′ c 0 ε k ) +log ( 1 δ ) )⌉ , which satisfies ε k m k ≥ 6 ln ( 16 k ( k +1) δ ) . Thus, we have established (39).

We argue the left inequality in (40) similarly. Consider any k ∈ { 1 , . . . , N +1 } having non-zero probability of k ≤ K . Given k ≤ K , note that ∆ i k has no dependence on S 2 k or m ′ k , so that the m ′ k samples in S 2 k are conditionally i.i.d.P given the event k ≤ K and given the random variables ∆ i k and m ′ k . Therefore, applying a multiplicative Chernoff bound (Lemma 6 of Appendix D) under the conditional distribution given the event k ≤ K and the random variables ∆ i k and m ′ k , with conditional probability at least 1 -δ 8(3+ N -k ) 2 ,

<!-- formula-not-decoded -->

By the law of total probability, we have that for every k ∈ { 1 , . . . , N +1 } , with probability at least 1 -δ 8(3+ N -k ) 2 , if k ≤ K then (43) holds. Letting E ′′ 3 denote the event that (43) holds for every k ∈ { 1 , . . . , K } , by the union bound, E ′′ 3 holds with probability at least 1 -∑ N +1 k =1 δ 8(3+ N -k ) 2 ≥ 1 -δ 8 . Let E 3 = E ′ 3 ∩ E ′′ 3 , and note that, by the union bound, E 3 holds with probability at least 1 -δ 4 . For the remainder of the proof, let us suppose the event ⋂ 3 j =0 E j occurs.

̸

To arrive at the simpler claimed inequalities in (40), we follow a similar argument to the final part of the proof of Lemma 14. Explicitly, we first note that for any k ∈ { 1 , . . . , K } , if ∆ i k = ∅ , we trivially have | S 2 k ∩ ∆ i k | = 0 = 2 P X (∆ i k ) m ′ k ≤ 10 βm ′ k . On the other hand, if ∆ i k = ∅ , the final claim in Lemma 11 implies P X (∆ i k ) &gt; ε k +2 , and combined with Lemma 13 this further implies ˆ p k ≥ P X (∆ i k ) &gt; ε k +2 . Therefore, in this case,

<!-- formula-not-decoded -->

where the rightmost inequality follows from c 1 ≥ 1 and C ′′ ≥ 6 C 4 . Thus, the left inequality in (40) follows from (43). The right inequality in (40) follows immediately from the fact (established above) that m ′ k ≤ m ′ k , together with the fact (from Lemma 20) that P X (∆ i k ) ≤ 5 β .

The remaining claims in the lemma follow from reasoning about convergence of the relevant series. Specifically, recalling that ε k = C 1 -k , N = ⌈ log C ( 2 ε )⌉ , and C = 11 10 , we note that ∑ N +1 k =1 1 ε 2 k = 1 C 2 -1 ( C 2( N +1) -1 ) ≤ 28 ε 2 and

<!-- formula-not-decoded -->

Recalling m ′ k = 25 C ′′ c 2 1 β ε 2 k ( d +2ln(3 + N -k ) + ln ( 4 δ )) , we have

<!-- formula-not-decoded -->

To obtain the query bound Q ( ε, δ ; β ) in (41), note that the total number of queries is precisely

<!-- formula-not-decoded -->

By (40), the first term in (45) is upper bounded by 10 β · ∑ N +1 k =1 m ′ k , and (44) implies this is at most 10 βM 2 . The second term in (45) is trivially upper bounded by M 1 := ∑ N +1 k =1 m k . Moreover, noting that ε k m k is increasing in k , (39) implies the second term in (45) is also upper bounded by 3 s ε N +1 · mN +1 · ( N +1) ≤ (3 / 2) s ε ( N +1) mN +1 . Together with the definition of Q ( ε, δ ; β ) from (41), we have that the total number of queries (45) is at most Q ( ε, δ ; β ) .

The bound on the asymptotic form of Q ( ε, δ ; β ) in (41) follows immediately from the definitions. Specifically, by definition of M 2 , we have 10 βM 2 = O ( β 2 ε 2 ( d +log ( 1 δ )) ) . Moreover, since ε N +1 ≥ ε 2 C , we have (3 / 2) s ε ( N +1) mN +1 = O ( s log ( 1 ε ) ( d log ( 1 ε ) +log ( 1 δ ))) , while (since each k ≤ N + 1 has ε k ≥ ε 2 C ), M 1 = ∑ N +1 k =1 m k ≤ ∑ N +1 k =1 301 C ′′ c 0 ε k ( d log ( 2 CC ′′ c 0 ε ) +log ( 1 δ ) ) = O ( 1 ε ( d log ( 1 ε ) +log ( 1 δ ))) by evaluating the geometric series. ■

As a final step before composing these lemmas into a proof of Theorem 5, we state an explicit bound on the number of unlabeled examples used by the algorithm. Much of this analysis is already implied by the above lemmas: namely, by definition, the number of examples allocated to data sets S 1 k and S 4 k is precisely 2 M 1 = 2 ∑ N +1 k =1 m k , and Lemma 23 implies the number of examples allocated to data sets S 2 k is at most M 2 . What remains is to bound the number of examples allocated to the data sets S 3 k,i , which hinges on bounding the number of iterations of the ' While ' loop for each k . We have already noted, in Lemma 10, that max I k ≤ 1 ε k +3 on the event E 0 , which already suffices to establish a coarse bound ˜ O ( d ε 2 ) . However, we will need a slight refinement to obtain the claimed upper bound, which will follow from a combination of Lemmas 11 and 20.

Lemma 24. On the event ⋂ 3 j =0 E j , the total number of examples allocated to data sets S 1 k , S 4 k ( k ≤ N +1 ), S 2 k ( k ≤ K ), and S 3 k,i ( k ∈ K , i ∈ I k ) is at most

<!-- formula-not-decoded -->

Proof. Suppose the event ⋂ 3 j =0 E j occurs. By definition, the number of examples allocated to data sets S 1 k and S 4 k is m k each, for k ∈ { 1 , . . . , N +1 } , so that the total number of such examples is ∑ N +1 k =1 2 m k = 2 M 1 . Also, by the first claim in Lemma 23, the number m ′ k of examples allocated to each S 2 k data set (for k ∈ { 1 , . . . , K } ) satisfies m ′ k ≤ m ′ k . Moreover, Lemma 23 also establishes that ∑ N +1 k =1 m ′ k ≤ M 2 . Together, we have that the total number of examples allocated to data sets S 2 k is ∑ K k =1 m ′ k ≤ M 2 . Thus, to complete the proof of Lemma 24, it suffices to bound the total number of examples allocated to data sets S 3 k,i ( k ∈ K , i ∈ I k ).

Toward this end, recall that for each k ∈ K , each S 3 k,i is of size m k , and is allocated if and when the algorithm reaches Step 5 with values ( k, i ) . Thus, if k = K (which, by the final claim in Lemma 10, occurs only if the algorithm returns in Step 4 in round k ), then no examples are allocated to any S 3 k,i sets in round k , whereas if k &lt; K , then the number of S 3 k,i data sets allocated during round k is precisely the number of distinct values of i encountered in round k : that is, |I k | . Moreover, note that since each time through the ' While ' loop increments i , each k ∈ K with k &lt; K has |I k | = i k +1 -i k +1 . It follows that the total number of examples allocated to data sets S 3 k,i in the algorithm is precisely ∑ k ∈K : k&lt;K m k ( i k +1 -i k +1) .

Next we upper bound i k +1 -i k for each k ∈ K with k &lt; K . Specifically, for any such k , note that ∆ i k +1 \ ∆ i k = ⋃ i k +1 -1 i = i k (∆ i +1 \ ∆ i ) , and by definition the sets ∆ i +1 \ ∆ i are disjoint over i . Moreover, by Lemma 11 and the definition of ∆ i +1 in Step 7, any i ∈ { i k , . . . , i k +1 -1 } has P X (∆ i +1 \ ∆ i ) &gt; ε k +3 . Therefore,

<!-- formula-not-decoded -->

On the other hand, by Lemma 20, P X (∆ i k +1 \ ∆ i k ) ≤ P X (∆ i k +1 ) ≤ 5 β . Combining these inequalities, we conclude that ( i k +1 -i k ) ≤ 5 C 3 β ε k . Combined with the facts that m k ≤ mN and

∑ k ∈K : k&lt;K m k ≤ M 1 , altogether we have

<!-- formula-not-decoded -->

where the last inequality follows by evaluating the geometric series and recalling ε N ≥ ε 2 C . This completes the proof that the total number of examples allocated to data sets S 1 k , S 3 k,i , S 2 k , S 4 k is at most M ( ε, δ ; β ) . The claimed asymptotic form of M ( ε, δ ; β ) follows immediately from the definitions of the quantities involved: namely, by definition, 3 M 1 = Θ ( 1 ε ( d log ( 1 ε ) +log ( 1 δ ))) , M 2 = Θ ( β ε 2 ( d +log ( 1 δ )) ) , and (since ε N ≥ ε 2 C ) 100 C 4 βm N ε = Θ ( β ε 2 ( d log ( 1 ε ) +log ( 1 δ )) ) . ■

We are now ready to combine the above lemmas into a complete proof of Theorem 5.

Proof of Theorem 5. By the union bound, the event ⋂ 3 j =0 E j has probability at least 1 -δ . By Lemma 24, on ⋂ 3 j =0 E j , A avid uses at most M ( ε, δ ; β ) (as defined in the lemma) of the examples in the sequence; in particular, this means that if we were to run the algorithm with a finite sequence ( X 1 , Y 1 ) , . . . , ( X m , Y m ) , for any m ≥ M ( ε, δ ; β ) , then on the event ⋂ 3 j =0 E j , the behavior of the algorithm (e.g., queries, returned ˆ h ) is identical to the idealized setting the above lemmas were established under (where there is an unlimited supply of examples), and hence the claims in the above lemmas remain valid. Thus, for any sample size m ≥ M ( ε, δ ; β ) , on the event ⋂ 3 j =0 E j , by Lemma 19 we have er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε , and by Lemma 23 the total number of queries is at most Q ( ε, δ ; β ) as defined therein. ■

A remark on intersecting with D k -1 in ∆ i k : We remark that Theorem 5 remains valid if we restrict either (or both) h 1 , h 2 to be in V k -1 in the definition (2) of V (4) k -1 . The entire proof remains valid (applying the same change to V (3) k -1 ), with the only exception being the first inequality in Lemma 20, which should then replace C by V k -1 . This change is of no consequence to the second inequality in the lemma since the proof of Lemma 17 in fact implies that h ⋆ ∈ V k -1 holds simultaneously (on E 0 ∩ E 1 ∩ E 2 ) for all functions h ⋆ ∈ C satisfying (10) (and hence inf h ∈ V k -1 er P ( h ) = β ). Moreover, with this restriction to require h 2 ∈ V k -1 in V (4) k -1 , we can extend the intersection with D k -1 to the ∆ i k region: that is, instead of querying all of S 2 k ∩ ∆ i k (in Step 2) or S 2 N +1 ∩ ∆ i N +1 (in Step 9), we can instead merely query the subset S 2 k ∩ D k -1 ∩ ∆ i k (in Step 2) or S 2 N +1 ∩ DN ∩ ∆ i N +1 (in Step 9). With this change, we must then also modify the definition of ˆ er 1,2 k ( h ) in (1) to ˆ er 1,2 k ( h ) := ˆ P S 1 k (ER( h ) ∩ D k -1 \ ∆ i k ) + ˆ P S 2 k (ER( h ) ∩ D k -1 ∩ ∆ i k ) . The argument in Lemma 15 extends to this modified definition of ˆ er 1,2 k ( h ) , since (as in the proof of (14) in Lemma 10) we are only interested in error differences , which, for h, h ′ ∈ V k -1 , satisfy ˆ P S 2 k (ER( h ) ∩ D k -1 ∩ ∆ i k ) -ˆ P S 2 k (ER( h ′ ) ∩ D k -1 ∩ ∆ i k ) = ˆ P S 2 k (ER( h ) ∩ ∆ i k ) -ˆ P S 2 k (ER( h ′ ) ∩ ∆ i k ) . Indeed, with additional modifications to the proof, we can then even slightly refine the query complexity analysis, since if we replace the sets ER( h ) ∩ ∆ i k with ER( h ) ∩ D k -1 ∩ ∆ i k in Lemma 14, the envelope set in the application of Lemma 8 in the proof of Lemma 14 can be chosen as D k -1 ∩ ∆ i k , so that we can refine the definition of ˆ p k to 2 ˆ P S 4 k ( D k -1 ∩ ∆ i k )+ O ( ε k ) . However, since these changes concern only the leading term β 2 ε 2 ( d +log ( 1 δ )) in Theorem 5, which is already optimal (perfectly matching the lower bounds of Kääriäinen, 2006; Beygelzimer, Dasgupta, and Langford, 2009), they are completely inconsequential to the theorem. We have therefore stated the algorithm without these modifications, for simplicity. However, this modified variant would be interesting in the context of P -dependent analysis, where it can lead to refinements to the leading term in the upper bound under certain favorable distributions. We leave the investigation of such refinements as an interesting direction for future work (focusing our P -dependent analysis in Appendix F on refining the lower-order term).

## F Distribution-Dependent Analysis

In addition to analysis based on the star number (Hanneke and Yang, 2015), the active learning literature includes a variety of distribution-dependent complexity measures which have been used to analyze the query complexity in various contexts (see Appendix A). In this section, we will add to this line of work a distribution-dependent analysis of A avid which replaces the star number s in Theorem 3 by a (never-larger) distribution-dependent quantity (Theorem 27), which can be further upper-bounded in terms of a simpler and more-familiar quantity: namely, a quadratic θ 2 dependence in the disagreement coefficient (Hanneke, 2007b; Definition 25 below). We also show (in Appendix F.1) that it is not possible (by any algorithm) to obtain a lower-order term which replaces the star number in Theorem 3 with the disagreement coefficient θ itself, so that the aforementioned θ 2 quadratic dependence generally cannot be reduced to linear (without introducing other factors). We will also present (in Appendix F.3) a slight refinement of A avid , which replaces the region of disagreement D k -1 by a carefully-chosen subregion , following the technique of Zhang and Chaudhuri (2014); Balcan, Broder, and Zhang (2007), which yields a corresponding refinement of the distribution-dependent query complexity bound. For instance, in the case of learning homogeneous linear classifiers under a uniform (or isotropic log-concave) distribution, this recovers a known query complexity bound ˜ O ( d β 2 ε 2 + d ) (and indeed, improves log factors in the lead term compared to prior works).

The Disagreement Coefficient: In the context of agnostic active learning, the most commonly-used P -dependent complexity measure is the disagreement coefficient , introduced by Hanneke (2007b), defined as follows.

Definition 25. For any concept class C and distribution PX on X , for any measurable function f : X → { 0 , 1 } , for any ε ≥ 0 , the disagreement coefficient , denoted by θ P X ,f ( ε ) , is defined as

<!-- formula-not-decoded -->

̸

where B P X ( f, r ) := { h ∈ C : PX ( h = f ) ≤ r } denotes the r -ball centered at f , and DIS( C ′ ) := { x ∈ X : ∃ h, h ′ ∈ C ′ , h ( x ) = h ′ ( x ) } denotes the region of disagreement (as in Section 4). For any distribution P on X × { 0 , 1 } and ε &gt; 0 , for h ⋆ as in (10) , 14 define θ P ( ε ) := θ P X ,h ⋆ ( ε ) .

̸

There are many works establishing bounds on the disagreement coefficient for commonly-studied classes C under various restrictions on the distribution P (see Hanneke, 2014, for a detailed summary). As discussed in Appendix A, the disagreement coefficient commonly appears in analyses of the query complexity of disagreement-based active learning methods (e.g., Hanneke, 2007b, 2009b, 2011, 2014; Dasgupta, Hsu, and Monteleoni, 2007). Since the lower-order term in Theorem 3 arises from the analysis of queries in the region of disagreement DIS( V k -1 ) of V k -1 , one might naturally wonder whether we can replace s with θ P ( ε ) in the upper bound in Theorem 3. Hanneke and Yang (2015) have shown that sup P θ P ( ε ) = s ∧ 1 ε for ε ∈ (0 , 1] , which implies that if we could replace s by θ P ( ε ) it would indeed represent a distribution-dependent refinement of the upper bound in Theorem 3. However, it turns out this is not possible (by any algorithm) for some classes C , as we demonstrate by an example in Appendix F.1. Following this, in Appendix F.2, we find that it is possible to achieve a lower-order term ˜ O ( d θ P ( β + ε ) 2 ) , and indeed this is achieved by A avid . This quadratic dependence unfortunately means the upper bound is sometimes loose (i.e., sometimes larger than that in Theorem 3). However, as an intermediate step, we also establish a query complexity bound (Theorem 27) expressed in terms of a modified disagreement coefficient which is never larger than the s -dependent query complexity bound in Theorem 3 (though which is more difficult to evaluate due to a more-involved definition).

14 When h ⋆ is not uniquely defined, in principle we can define θ P ( ε ) as the infimum value among all choices of such h ⋆ . It is also possible to define h ⋆ as an ε -independent fixed function, even when er P ( h ) does not have a minimizer in C , by choosing it as an element of the L 1 ( PX ) -closure of C having er P ( h ⋆ ) = inf h ∈ C er P ( h ) : see (Hanneke, 2012) for a proof that such an h ⋆ always exists when VC( C ) &lt; ∞ . In particular, with such an h ⋆ , the limiting value θ P (0) := θ P X ,h ⋆ (0) is also well-defined.

## F.1 Impossibility of Replacing s with θ P ( ε )

In this section, we present an example demonstrating that no algorithm can achieve a P -dependent query complexity bound which replaces s by θ P ( ε ) in Theorem 3.

An Example: Consider the following concept class (see Hanneke, 2007b, for a related construction). Let X = Z (the integers) and define a concept class

<!-- formula-not-decoded -->

In other words, each h ∈ C ts defines a threshold classifier on the positive integers and a singleton classifier on the negative integers, and the position -t of the singleton point mirrors the position t of the threshold boundary point.

Fix any ε, β ∈ (0 , 1 / 3) , denote by n = 1 -2 β 2 ε , and for simplicity suppose n ∈ N . Define a marginal distribution PX on X as follows: ∀ x ∈ { 1 , . . . , n } , PX ( { x } ) = 2 β n and PX ( {-x } ) = 2 ε . Note that this completely specifies PX . Now define a family of probability distributions P 1 , . . . , P n : for each t ∈ { 1 , . . . , n } , P t has marginal distribution PX on X and conditional distribution ∀ x ∈ X

<!-- formula-not-decoded -->

In particular, note that each P t satisfies inf h ∈ C ts er P t ( h ) = β and h ⋆ is uniquely equal 1 {-t }∪ [ t, ∞ ) . 15

̸

Upper-bounding θ P t ( ε ) : For any P t , we will argue θ P t ( ε ) ≤ θ P X ,h ⋆ (0) = O ( 1 β ) . For any r &gt; 0 , if h ∈ B P X ( h ⋆ , r ) , then letting k h := |{ h = h ⋆ } ∩ { 1 , . . . , n }| , since each x ∈ { 1 , . . . , n } has PX ( { x } ) = 2 β n , we must have k h ≤ k r := ⌊ rn 2 β ⌋ . Since h and h ⋆ both implement threshold functions in this region { 1 , . . . , n } , the k h elements in { h = h ⋆ } ∩ { 1 , . . . , n } are a contiguous segment: either { t, . . . , t + k h -1 } or { t -k h , . . . , t -1 } . In either case, we have { h = h ⋆ } ∩ { 1 , . . . , n } ⊆ { t -k r , . . . , t + k r -1 } ∩ { 1 , . . . , n } . Moreover, since h = 1 {-t ′ }∪ [ t ′ , ∞ ) for some t ′ ∈ N , this further implies { h = h ⋆ } ∩ {-1 , . . . , -n } ⊆ {-( t -k r ) , . . . , -( t + k r ) } ∩ {-1 , . . . , -n } . Since DIS(B P X ( h ⋆ , r )) is just the union of these { h = h ⋆ } regions among all h ∈ B P X ( h ⋆ , r ) , we have

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We also note that any r &lt; 2 ε has B P X ( h ⋆ , r ) = { h ⋆ } . Altogether,

<!-- formula-not-decoded -->

̸

Lower-bounding the query complexity: On the other hand, we will argue that the query complexity is Ω( 1 ε ) under the assumption P ∈ { P 1 , . . . , P n } . Note that every h : X → { 0 , 1 } has the same value of P t (ER( h ) \ {-1 , . . . , -n } ) = β . Together with the definition of P t in {-1 , . . . , -n } , this implies any h : X → { 0 , 1 } with { h = h ⋆ }∩{-1 , . . . , -n } ̸ = ∅ has er P t ( h ) -er P t ( h ⋆ ) ≥ 2 ε . Thus, the problem of learning, to an excess error ε (under the assumption that P ∈ { P t : t ∈ { 1 , . . . , n }} ) is equivalent to the problem of identifying the value t ∈ { 1 , . . . , n } for which the distribution P = P t : that is, if an algorithm returns ˆ h with er P t ( ˆ h ) -er P t ( h ⋆ ) ≤ ε , the unique x ∈ { 1 , . . . , n } for which ˆ h ( -x ) = 1 satisfies x = t . Moreover, in the active learning problem defined by these distributions, for every x / ∈ {-1 , . . . , -n } , the conditional distribution P t ( Y = 1 | X = x ) of responses to queries for examples at x is invariant to t , so that such queries reveal no information about which

15 Indeed, these distributions P t even satisfy the stronger benign noise property: i.e., inf h ∈ C ts er P t ( h ) = Bayes risk (a setting studied by Hanneke, 2009b; Hanneke and Yang, 2015). Thus, the argument in this section further implies the impossibility of using θ P ( ε ) in the lower-order term under benign noise.

̸

̸

̸

t has P = P t , and hence without loss of generality we can restrict to active learning algorithms that do not query outside {-1 , . . . , -n } . The unlabeled examples also reveal no such information, since all P t have the same marginal distribution PX . Altogether, the active learning problem for this set of distributions is information-theoretically no easier (in terms of query complexity) than the problem of actively identifying a singleton classifier on {-1 , . . . , -n } in the realizable case under marginal Uniform( {-1 , . . . , -n } ) . 16 It is well known that the minimax query complexity of this latter problem (with confidence parameter δ = 1 / 3 ) is Ω( n ) = Ω( 1 ε ) (Dasgupta, 2004, 2005; Hanneke, 2014; Hanneke and Yang, 2015), which therefore serves as a lower bound on the minimax query complexity for P ∈ { P t : t ∈ { 1 , . . . , n }} : that is, for every active learning algorithm, there exists P ∈ { P t : t ∈ { 1 , . . . , n }} for which, with probability at least δ , it either makes Ω( 1 ε ) queries or returns ˆ h with er P ( ˆ h ) &gt; inf h ∈ C ts er P ( h ) + ε .

Conclusion that θ P t ( ε ) is not achievable in the lower-order term: From the above arguments, we can conclude that for the class C ts , it is not possible to replace s by θ P ( ε ) (or indeed θ P (0) ) in the upper bound of Theorem 3 to obtain a P -dependent refinement of the upper bound. Formally, for every active learning algorithm guaranteeing that, for every P with inf h ∈ C ts er P ( h ) ≤ β , with probability at least 2 / 3 (i.e., δ = 1 / 3 ), it returns ˆ h with er P ( ˆ h ) ≤ inf h ∈ C ts er P ( h ) + ε , there exists a distribution P satisfying this for which θ P ( ε ) ≤ 1 -2 β β +3 = O ( 1 β ) , yet with probability at least 1 / 3 , the algorithm makes a number of queries Ω( 1 ε ) . We have argued this conclusion for any choices of ε, β ∈ (0 , 1 / 3) (with n ∈ N for simplicity). In particular, for ε ≪ β ≪ √ ε (e.g., β ≈ ε 2 / 3 ), such a distribution P has β 2 ε 2 + θ P ( ε ) = β 2 ε 2 + O ( 1 β ) ≪ 1 ε , so that replacing s with θ P ( ε ) in Theorem 3 cannot yield a valid query complexity bound (holding for all P ) for any active learning algorithm. Indeed, we have established that this conclusion also holds for θ P (0) (as defined in footnote 14).

Wewill see in Corollary 28 of Appendix F.2 that A avid does achieve an upper-bound ˜ O ( d θ P ( β + ε ) 2 ) on the lower-order term: a quadratic dependence on the disagreement coefficient. This conclusion is compatible with the above scenario, since β 2 ε 2 + 1 β 2 = Ω ( 1 ε ) for the full range of β, ε .

## F.2 Replacing s with θ P ( β + ε ) 2

Appendix F.1 implies the disagreement coefficient θ P ( ε ) , as defined in Definition 25, cannot be used as a P -dependent substitute for the star number s in Theorem 3 (at least, not with a linear dependence). In this section, we will argue that the AVID Agnostic algorithm A avid does achieve a P -dependent lower-order term which is at most quadratic in the disagreement coefficient: namely, ˜ O ( d θ P ( β + ε ) 2 ) . We will argue this by first establishing a P -dependent refinement of Theorem 5 based on a modified disagreement coefficient (Definition 26) which is never larger than the star number. While this quantity itself is often more-difficult to calculate, compared to the original disagreement coefficient θ P ( ε ) , fortunately it is always upper bounded by O ( β 2 ε 2 + θ P ( β + ε ) 2 ) . In particular, this means that for any P with θ P (0) &lt; ∞ , the asymptotic dependence on ε, δ in the lower-order term in Theorem 3 can be reduced to polylog ( 1 εδ ) .

̸

Specifically, the modified disagreement coefficient we consider can be expressed as the value θ P ∆ ,h ⋆ ( ε ) produced under a restriction of PX to a subregion X \ ∆ of size at least 1 -O ( β ) . Toward stating the definition, we first extend Definition 25 to allow for general measures µ : that is, for any measure µ on X and measurable f : X → { 0 , 1 } , define B µ ( f, r ) := { h ∈ C : µ ( h = f ) ≤ r } ,

16 Formally, for any active learning algorithm A , under distributions P ∈ { P t : t ∈ { 1 , . . . , n }} , we can convert A into an active learner A ′ for realizable-case singletons under Uniform( {-1 , . . . , -n } ) with at most the query complexity of A under such distributions P . Specifically, given any number m of i.i.d. unlabeled examples X 1 , . . . , X m ∼ Uniform( {-1 , . . . , -n } ) , define independent random variables (also independent of X 1 , . . . , X m ) B 1 , . . . , B m ∼ Bernoulli(2 β ) , X ′ 1 , . . . , X ′ m ∼ Uniform( { 1 , . . . , n } ) , and Y ′ 1 , . . . , Y ′ m ∼ Bernoulli( 1 2 ) . For each i ≤ m , let X ′′ i = X i if B i = 0 and X ′′ i = X ′ i if B i = 1 . Then A ′ runs A with unlabeled data X ′′ 1 , . . . , X ′′ m ; whenever A queries an X ′′ i with B i = 0 , A ′ queries for the label Y i of X i and gives this as a response to the query, and whenever A queries an X ′′ i with B i = 1 , A ′ gives Y ′ i as a response to the query. Note that the corresponding data sequence and responses observed by A are indeed identical to running A under P = P t , where -t is the singleton location for the realizable-case singleton problem P t ( ·|{-1 , . . . , -n } ) . Thus, the query complexity of A ′ identifying the t for the realizable-case singletons distribution P t ( ·|{-1 , . . . , -n } ) is at most that of A identifying this t when P = P t .

and for ε ≥ 0 define θ µ,f ( ε ) := sup r&gt;ε µ (DIS(B µ ( f,r ))) r ∨ 1 . Wethen consider the following definition: a region-excluded disagreement coefficient .

Definition 26. For any distribution P on X × { 0 , 1 } and any measurable ∆ ⊆ X , define a measure A ↦→ P ∆ ( A ) := PX ( A \ ∆) . For any ε, τ ≥ 0 , for h ⋆ ∈ C as in (10) (under P ), 17 define

<!-- formula-not-decoded -->

We can equivalently define θ P ( ε ; τ ) as the disagreement coefficient under a worst-case conditional distribution PX ( ·|X \ ∆) : that is,

<!-- formula-not-decoded -->

where we define θ P X ( ·|X\ ∆) ,h ⋆ ( ε/PX ( X \ ∆)) = 1 in the case PX ( X \ ∆) = 0 (which coincides with the value θ P ∆ ,h ⋆ ( ε ) for such ∆ ).

We may note that θ P ( ε ; τ ) indeed provides a refinement of the star number, in that it is never larger . Specifically, since Hanneke and Yang (2015) have shown

<!-- formula-not-decoded -->

for every ε ∈ (0 , 1] , the expression in (46) of θ P ( ε ; τ ) as the disagreement coefficient under conditional distributions immediately implies

<!-- formula-not-decoded -->

Thus, replacing s in Theorem 3 by θ P ( ε ; τ ) would indeed yield a (never-larger) P -dependent refinement.

We give examples below (Appendix F.2.1) of calculating and upper-bounding θ P ( ε ; τ ) under various scenarios ( C , P ) . We remark that, due to the supremum over regions ∆ , the quantity θ P ( ε ; τ ) is often much more involved to calculate or bound compared to the original disagreement coefficient θ P ( ε ) in Definition 25. We might therefore think of θ P ( ε ; τ ) as a kind of intermediate complexity measure, which is useful in that it provides a P -dependent refinement of s , while also admitting general upper bounds which are more accessible than directly calculating θ P ( ε ; τ ) . Concretely, there are at least weak relations between θ P ( ε ; τ ) and the more-familiar disagreement coefficient from Definition 25: namely, θ P ( ε ) ≤ θ P ( ε ; τ ) and

<!-- formula-not-decoded -->

These upper bounds on θ P ( ε ; τ ) are noteworthy since θ P ( τ + ε ) is typically significantly easier to calculate compared to directly calculating θ P ( ε ; τ ) (and there are already many works deriving bounds on θ P ( τ + ε ) for various scenarios; see Hanneke, 2014).

The quantity θ P ( ε ; τ ) is particularly well-suited for the analysis of A avid , since the algorithm explicitly maintains low diameter of V k under a region-excluded measure A ↦→ PX ( A \ ∆ i k ) . Specifically, Lemma 12 implies V k -1 ⊆ B P ∆ i k ( h ⋆ , ε k ) , while Lemma 20 implies PX (∆ i k ) ≤ 5 β , so that PX ( D k -1 \ ∆ i k ) ≤ θ P ( ε k ; 5 β ) ε k , and hence the number of queries in S 1 k ∩ D k -1 \ ∆ i k is O ( θ P ( ε k ; 5 β ) ε k m k ) = ˜ O ( θ P ( ε ; 5 β ) d ) . Formally, this leads to the following result, which simply replaces s with θ P ( ε ; 5 β ) in the lower-order term compared to Theorem 5. Due to (48), the query complexity bound in this result is never larger than that of Theorem 5 (and below we discuss scenarios where it is strictly smaller). We remark that, based on the comment preceding Lemma 20, the factor ' 5 ' in θ P ( ε ; 5 β ) in this theorem can be reduced to any value c &gt; 2 by appropriately adjusting the constants C , C ′′ in the algorithm.

17 The remarks concerning the choice of h ⋆ in footnote 14 also apply here, noting that the lemmas concerning h ⋆ in Appendix E actually apply simultaneously to all functions h ⋆ ∈ C satisfying (10).

Theorem 27 (Distribution-dependent Query Complexity of AVID Agnostic) . For any concept class C with VC( C ) &lt; ∞ , letting d = VC( C ) , for every distribution P on X × { 0 , 1 } , letting β = inf h ∈ C er P ( h ) , for any ε, δ ∈ (0 , 1) , if the algorithm A avid is executed with parameters ( ε, δ ) , with any number m ≥ M ( ε, δ ; β ) of i.i.d.P examples (for M ( ε, δ ; β ) as in Theorem 5, defined in Lemma 24), then with probability at least 1 -δ , the returned predictor ˆ h satisfies er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε and the algorithm makes a number of queries at most Q ( ε, δ ; P ) satisfying

<!-- formula-not-decoded -->

Proof. The result follows identically to Theorem 5, with only one minor change: replacing s with 2 Cθ P ( ε ; 5 β ) in Lemma 22. Note that this one change will suffice, since every subsequent appearance of s in the proof is due to its appearance in Lemma 22, and hence changing s to 2 Cθ P ( ε ; 5 β ) in this lemma allows us to make the same change in every subsequent appearance of s in the proof.

̸

To see why Lemma 22 remains valid with this change, first note that its proof establishes that, on the event E 0 ∩ E 1 ∩ E 2 , in the non-trivial case of PX ( X \ ∆ i k ) = 0 , (38) holds. Rather than relaxing the third expression in (38) using the star number, we can instead relax it using θ P ( ε ; 5 β ) : that is, for P k as defined in that context, (38) implies

<!-- formula-not-decoded -->

where the last inequality follows from Definition 26, the fact that ε k &gt; ε 2 C , and the fact (from Lemma 20) that PX (∆ i k ) ≤ 5 β . We then note that (as in Corollary 7.2 of Hanneke, 2014) for any ∆ ⊆ X ,

<!-- formula-not-decoded -->

and therefore θ P ( ε/ (2 C ); 5 β ) ≤ 2 Cθ P ( ε ; 5 β ) . Altogether, we have that Lemma 22 remains valid while replacing s with 2 Cθ P ( ε ; 5 β ) . ■

We emphasize that A avid does not need to know the value θ P ( ε ; 5 β ) (or anything else about P ) to achieve this query complexity: that is, it is adaptive to the value of θ P ( ε ; 5 β ) .

Together with (49), the above result further implies a (sometimes loose) relaxation, in which the lower-order term has a quadratic dependence on θ P ( β + ε ) , as formally stated in the following corollary (compare this with Appendix F.1, which showed it is impossible to generally reduce this θ P ( β + ε ) 2 term to a linear term θ P ( β + ε ) or even θ P (0) ).

Corollary 28. The query complexity bound Q ( ε, δ ; P ) in Theorem 27 (achieved by A avid ) satisfies

<!-- formula-not-decoded -->

Proof. Due to the first two inequalities in (49), and θ P (5 β + ε ) ≤ θ P ( β + ε ) , the second term in the expression of Q ( ε, δ ; P ) in Theorem 27 is at most

<!-- formula-not-decoded -->

Relaxing d log ( 1 ε ) +log ( 1 δ ) ≤ log ( 1 ε ) ( d +log ( 1 δ )) and noting that

<!-- formula-not-decoded -->

and ( β + ε ε ) 2 ≤ 4 β 2 ε 2 +4 , the quantity (50) is at most O ( β 2 ε 2 ( d +log ( 1 δ )) +min { θ P ( β + ε ) 2 log 4 ( 1 ε ) , 1 ε log ( 1 ε )}( d +log ( 1 δ ))) . Adding this to the first term in the expression of Q ( ε, δ ; P ) , the result follows. ■

In particular, Corollary 28 implies that, whenever θ P (0) &lt; ∞ , the dependence on β, ε in the query complexity bound in Theorem 27 is of order β 2 ε 2 +polylog ( 1 ε ) . For instance, see (Hanneke, 2014, Chapter 7) for some general conditions on ( C , P ) under which this occurs. We remark that the first bound in Corollary 28 is at least never larger than the upper bound in Theorem 1, since we always have θ P ( β + ε ) ( β + ε ε ) ≤ 1 ε ; however, we note that this is not the case for the second upper bound in Corollary 28. Beyond these basic observations, there exist scenarios ( C , P ) where both upper bounds in Corollary 28 are loose compared to Theorem 27, to such an extent that they are sometimes even larger than the s -dependent bound in Theorem 5 (see Example 6 below). It is for this reason that we have chosen to express Theorem 27 in terms of the more-complicated quantity θ P ( ε ; τ ) , to provide a starting point for P -dependent analysis that is at least never worse than Theorem 5.

## F.2.1 Examples

We next present some examples illustrating the values of the lower-order terms in Theorem 27 and Corollary 28 by bounding the quantities θ P ( ε ; τ ) and θ P ( β + ε ) 2 . Specifically, Example 1 achieves this via the relation to s , Example 2 provides a simple scenario with s = ∞ where it is possible to directly bound θ P ( ε ; τ ) , Example 3 expresses a bound on θ P ( β + ε ) which is known in the literature, but when combined with Corollary 28 provides an improved P -dependent query complexity bound compared to previous works. Example 5 revisits the example from Appendix F.1 to illustrate that θ P ( ε ; 5 β ) provides a valid lower-order term for this example. In Appendix F.4, we will present additional examples of P -dependent query complexity bounds, for some classes with VC( C ) = ∞ , via PX -dependent covering numbers .

Example 1 (Thresholds) . Due to (48), any C with finite star number s admits a bounded θ P ( ε ; τ ) . A simple example of this is threshold classifiers: namely, X = R and C = { 1 [ t, ∞ ) : t ∈ R } . This class has s = 2 (Hanneke and Yang, 2015), and hence θ P ( ε ; τ ) ≤ 2 for any P .

Example 2 (Linear classifiers under 1 -sparse distributions) . To illustrate a simple example where θ P ( ε ; τ ) = O (1) while s = ∞ , consider the class C of linear classifiers in X = R p , p ≥ 2 : that is, C = { x ↦→ 1 ⟨ w,x ⟩ + b ≥ 0 : w ∈ R p , b ∈ R } . This class has s = ∞ (Hanneke and Yang, 2015). However, if we consider PX as a distribution supported entirely on one axis (e.g., Uniform([0 , 1] × { 0 } p -1 ) ), then it is a simple exercise to show that θ P ( ε ; τ ) ≤ 2 : the concepts in B P ∆ ( h ⋆ , r ) are those that disagree with h ⋆ on at most r measure (under P ∆ ) either to the left or right of where the h ⋆ separator intersects the axis, so that DIS(B P ∆ ( h ⋆ , r )) is simply the union of these two (at most) r -measure regions, hence has P ∆ measure at most 2 r .

While the above examples merely recover known results, the following example derives a previouslyunknown P -dependent query complexity bound, which significantly improves over the best previously-known bound for this scenario.

Example 3 (Rectangles) . Consider the case X = R p , p ≥ 1 , and C = { 1 [ a 1 ,b 1 ] ×···× [ a p ,b p ] : a 1 ≤ b 1 , . . . , a p ≤ b p } : the class of axis-aligned rectangles (Mitchell, 1979). This class is known to have s = ∞ (Hanneke and Yang, 2015). Consider PX = Uniform([0 , 1] p ) (the example trivially extends to any product distribution PX with marginals on each axis having continuous CDFs) and any P with well-defined h ⋆ ∈ argmin h ∈ C er P ( h ) satisfying PX ( { x : h ⋆ ( x ) = 1 } ) =: λ &gt; 0 and El-Yaniv (2015) have shown that θ P ( β + ε ) = O ( d λ log( d ) ∧ 1 β + ε ) for this scenario, and based on this, the best known query complexity upper bound is of the form ˜ O ( min { d 2 λ ( β + ε ) 2 ε 2 , d β + ε ε 2 }) We can derive a bound which improves over this, as follows. We first recall that Theorem 1 provides a query complexity bound ˜ O ( d β 2 ε 2 + d ε ) , which already improves over the query complexity bound

. The optimal firstorder query complexity under these conditions is not yet precisely known. However, Wiener, Hanneke, .

of Wiener, Hanneke, and El-Yaniv (2015) in all regimes with ε ≪ β ≪ 1 (for every λ ). However, we can further refine the lower-order term by introducing a dependence on λ . Specifically, the first bound in Corollary 28 provides a query complexity bound ˜ O ( d β 2 ε 2 + d 2 ( β + ε ) λε ) , which offers a refinement over Theorem 1 whenever λ ≫ d ( β + ε ) . Moreover, the second bound in Corollary 28 provides a query complexity bound ˜ O ( d β 2 ε 2 + d 3 λ 2 ) . In particular, for λ = Θ(1) and β = ˜ O ( √ ε ) , this yields a query complexity bound poly( d )polylog ( 1 εδ ) , which was only available in the bound of Wiener, Hanneke, and El-Yaniv (2015) in the more-restrictive regime β = ˜ O ( ε ) . We leave open the question of identifying the optimal query complexity for this scenario. In particular, one concrete technical question toward that end would be to determine whether, for λ &gt; 2 τ , θ P ( ε ; τ ) = ˜ O ( d λ ) .

Example 4 (Linear Classifiers) . Consider the commonly-studied concept class of linear classifiers , defined as: X = R d -1 ( d ≥ 3 ) and C = { h w,b : w ∈ R d -1 , b ∈ R } , where h w,b ( x ) = 1 [ ⟨ w,x ⟩ + b ≥ 0] . This is perhaps the most well-studied concept class in the active learning literature. Its VC dimension satisfies VC( C ) = d (Vapnik and Chervonenkis, 1974), and while its star number satisfies s = ∞ (Hanneke and Yang, 2015), the disagreement coefficient has been shown to be bounded or sublinear under various distributional conditions (Hanneke, 2007b, 2014; Balcan, Hanneke, and Vaughan, 2010; Friedman, 2009; Mahalanabis, 2011; Wiener, Hanneke, and El-Yaniv, 2015). These results compose directly with Corollary 28 to yield previously-unknown bounds on the query complexity under these same conditions. For instance, if PX is a mixture of a finite number t of multivariate Gaussian distributions with full-rank diagonal covariance matrices, then Wiener, Hanneke, and El-Yaniv (2015) provide a bound θ P ( r ) ≤ c d ,t log d -2 ( 1 r ) for a ( d , t ) -dependent constant c d ,t . Plugging into Corollary 28 (or rather, the explicit bounds in the proof thereof) yields a novel query complexity bound of order β 2 ε 2 ( d +log ( 1 δ )) + c 2 d ,t log 2( d -2) ( 1 β + ε ) log 4 ( 1 ε ) ( d +log ( 1 δ )) . More generally, if PX admits a density with respect to the Lebesgue measure on R d -1 , then (taking h ⋆ as in footnote 14) Hanneke (2014) argues that θ P ( r ) = o ( 1 r ) (where the specific form of this function θ P ( r ) varies depending on P ). Recalling that (as ε → 0 ) the lower-order term becomes relevant only in the regime β ≪ √ ε , combining this with Corollary 28 yields a query complexity bound which often provides refinements over Theorem 1. In particular, under sufficient regularity conditions on PX (see the proof of Hanneke, 2014) to ensure this o ( 1 r ) function further satisfies θ P ( r ) log 2 ( 1 r ) = o ( 1 r ) , the resulting asymptotic dependence on ( ε, β ) is of the form β 2 ε 2 + o ( 1 ε ) . Moreover, if additionally the density of PX is bounded and has finite-diameter support, and if the hyperplane boundary corresponding to h ⋆ passes through a continuity point of this density in its support, then Hanneke (2014) argues θ P ( r ) = O (1) , so that Corollary 28 yields a query complexity bound with asymptotic dependence on ( ε, β ) of the form β 2 ε 2 + log 4 ( 1 ε ) . Moreover, under the further restrictions (density bounded away from 0 , compactness of the support), Friedman (2009); Mahalanabis (2011) argue θ P ( r ) is asymptotically bounded by O ( d ) (for the precise statement, see the original works, or discussion thereof by Hanneke, 2014).

Example 5 (Coupled thresholds and singletons) . Let us revisit the example from Appendix F.1, for which we argued that θ P ( ε ) cannot itself be used to replace the star number in Theorem 3 (for any algorithm). We will here explain how the region-excluded disagreement coefficient θ P ( ε ; 5 β ) explicitly corrects for the issue with θ P ( ε ) in this example. Specifically, consider again X = Z , and C = C ts := { 1 {-t }∪ [ t, ∞ ) : t ∈ N } , the class of coupled thresholds and singletons. Let ε, β ∈ (0 , 1 / 3) , let n = 1 -2 β 2 ε (and assume n ∈ N ), and consider again the distributions P t , t ∈ { 1 , . . . , n } , as defined in Appendix F.1: that is, all P t have marginal PX on X , where for x ∈ { 1 , . . . , n } , PX ( { x } ) = 2 β n , PX ( {-x } ) = 2 ε , and for x ∈ {-1 , . . . , -n } , P t ( Y = 1 | X = x ) = 1 [ x = -t ] , while every x / ∈ {-1 , . . . , -n } has P t ( Y = 1 | X = x ) = 1 2 . Note that, for ∆ = [0 , ∞ ) , we have PX (∆) = 2 β . Moreover, for h ⋆ as defined under P t , we have B P ∆ ( h ⋆ , 4 ε ) = C ts (since only the disagreements on the singleton part are measured by P ∆ ). This implies DIS(B P ∆ ( h ⋆ , 4 ε )) = Z \ { 0 } , so that P ∆ (DIS(B P ∆ ( h ⋆ , 4 ε ))) = 1 -2 β . Therefore, θ P ( ε ; 5 β ) ≥ θ P ( ε ; 2 β ) ≥ P ∆ (DIS(B P ∆ ( h ⋆ , 4 ε ))) 4 ε = 1 -2 β 4 ε . Since we always have θ P ( ε ; 5 β ) ≤ 1 ε , we conclude that θ P ( ε ; 5 β ) = Θ ( 1 ε ) . As argued in Appendix F.1, the minimax optimal query complexity (constraining to P ∈ { P t : t { 1 , . . . , n }} ) is Ω ( 1 ε ) , so that, unlike θ P ( ε ) , the quantity θ P ( ε ; 5 β ) is an appropriate replacement for s in Theorem 3. Of course, Theorem 27 shows this replacement is always valid, so the point here is merely to illustrate how the exclusion of the ∆ region in the definition of

̸

θ P ( ε ; 5 β ) is precisely the right type of correction, compared to θ P ( ε ) , for this example, as it explicitly removes the issue underlying the failure of θ P ( ε ) : namely, the fact that the threshold portion of the concepts 1 {-t ′ }∪ [ t ′ , ∞ ) is irrelevant to the learning problem inherent in the P t distributions. It is also worth noting that the first upper bound θ P ( ε ; 5 β ) ≤ θ P (5 β + ε ) ( 5 β + ε ε ) from (49) also yields a value Θ ( 1 ε ) (since this upper bound is never larger than 1 ε ). However, the second upper bound θ P (5 β + ε ) 2 + ( 5 β + ε ε ) 2 can be significantly looser for this example, in most regimes of ε, β (namely, β = Θ( √ ε ) ).

## F.2.2 The Error Disagreement Coefficient

It is also possible to derive Corollary 28 via another intermediate P -dependent variant of the disagreement coefficient: namely, the error disagreement coefficient , defined as follows.

Definition 29. For any probability distribution P on X × { 0 , 1 } , for any ε ≥ 0 , define

<!-- formula-not-decoded -->

where C P ( r ) := { h ∈ C : er P ( h ) -inf h ′ ∈ C er P ( h ′ ) ≤ r } is known as the r -minimal set .

Similarly to θ P ( ε ; τ ) , the quantity θ er P ( ε ) has direct relations to the original disagreement coefficient from Definition 25. Specifically, for h ⋆ as in (10), 18 since B P X ( h ⋆ , r/ 2) ⊆ C P ( r ) ⊆ B P X ( h ⋆ , 2( β + r )) for any r &gt; ε , we immediately have

<!-- formula-not-decoded -->

By definition, we always have θ er P ( ε ) ≤ 1 ε . However, unlike θ P ( ε ; τ ) in (48), the quantity θ er P ( ε ) is not always upper-bounded by the star number s (see Example 6 below), so that we need be careful when replacing s by θ er P ( ε ) in Theorem 3.

It is also worth noting that θ er P ( ε ) is often not as easy to use for studying specific scenarios, compared to θ P ( ε ) , due to the dependence on the conditional distribution Y | X (whereas θ P ( ε ) depends only on PX and h ⋆ ). Nonetheless, below we will state a query complexity bound in terms of θ er P ( ε ) (Theorem 30) which is sometimes smaller than that in Theorem 27 (as we illustrate in examples below), and moreover (together with (51)) provides another route to proving the query complexity bound in Corollary 28.

̸

The quantity θ er P ( ε ) essentially arises naturally in many existing analyses of disagreement-based active learning (e.g., Hanneke, 2009b, 2011, 2014; Koltchinskii, 2010; Foster, Rakhlin, Simchi-Levi, and Xu, 2021), wherein certain algorithms are shown to makes queries in a subset of DIS( C P ( ε ′ )) for an appropriate ε ′ ≥ ε (decreasing as the algorithm runs). In those contexts, it is traditional to upper bound PX (DIS( C P ( ε ′ ))) by θ P ( r ( ε ′ )) r ( ε ′ ) , where r ( ε ′ ) ≥ sup h ∈ C P ( ε ′ ) PX ( h = h ⋆ ) : for instance, r ( ε ′ ) = 2( β + ε ′ ) suffices in the agnostic setting. However, one can alternatively upper bound PX (DIS( C P ( ε ′ ))) by θ er P ( ε ′ ) ε ′ . Such arguments are also valid in the context of A avid , since Corollary 18 implies PX ( D k -1 ) ≤ PX (DIS( C P ( ε k ))) ≤ θ er P ( ε k ) ε k , so that the number of queries in S 1 k ∩ D k -1 in round k is of order θ er P ( ε k ) ε k m k = ˜ O ( θ er P ( ε k ) d ) , which will lead to a lower-order term ˜ O ( θ er P ( ε ) d ) . Together with reasoning similar to the proof of Theorem 27, this implies the following.

Theorem 30. Under the same conditions as Theorem 27, with probability at least 1 -δ the predictor ˆ h returned by A avid satisfies er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε and the algorithm makes a number of queries at most Q ( ε, δ ; P ) satisfying

<!-- formula-not-decoded -->

18 The relation sharpens to θ P ( ε ) ≤ θ er P ( ε ) ≤ θ P (2 β + ε ) ( 2 β + ε ε ) if we take h ⋆ ∈ argmin h ∈ C er P ( h ) , supposing this exists (or otherwise, taking h ⋆ as discussed in footnote 14).

Since θ er P ( ε ) ≤ 1 ε , the query complexity bound in Theorem 30 is never larger than that in Theorem 1. However, unlike Theorem 27, since the quantity θ er P ( ε ) is not always upper-bounded by the star number s , the query complexity bound in Theorem 30 is sometimes larger than that in Theorem 3 (see Example 6 below). That said, the quantities θ er P ( ε ) and θ P ( ε ; 5 β ) are generally incomparable (see Examples 6 and 7), so that either bound may be useful depending on the scenario being studied. Moreover, in light of (51), Theorem 30 is also useful for providing another route to establishing Corollary 28, which is therefore an immediate corollary of either Theorem 27 or Theorem 30.

As mentioned, depending on ( C , P ) , the quantitative difference between θ er P ( ε ) and θ P ( ε ; 5 β ) can be better or worse. We illustrate this in the following two examples.

Example 6 ( θ er P ( ε ) ≫ s ≥ θ P ( ε ; 5 β ) ) . As mentioned, for some scenarios, θ er P ( ε ) can be quite large, even larger than the star number s , so that the bound in Theorem 30 becomes even worse than the P -independent bound in Theorem 3 (in contrast to Theorem 27, which is never worse than Theorem 3). For instance, consider a singletons class: X = { 1 , . . . , 2 β } , C = { 1 { t } : t ∈ X} , where β ∈ (0 , 1 / 2) satisfies 2 β ∈ N for simplicity. Let PX = Uniform( X ) , P ( Y = 1 | X = x ) = β 2(1 -β ) for all x ∈ X . Note that every h ∈ C has er P ( h ) = β . Moreover, s = |X| 1 = 2 -β β . However, consider 0 &lt; ε ≪ β . Since C P ( ε ) = C and DIS( C ) = X , we have θ er P ( ε ) = 1 ε = Θ ( s β ε ) . For instance, for β = ε 2 / 3 , the bound in Theorem 3 is (ignoring logs) of order β 2 ε 2 + 1 β = 2 ε 2 / 3 ≪ 1 ε = θ er P ( ε ) . In light of (51), this example also witnesses a scenario where both of the query complexity bounds in Corollary 28 are worse than the s -dependent bound in Theorem 3; more directly, in this example, we have θ P ( β + ε ) ( β + ε ε ) = 1 ε ≫ s . In contrast, for any r &lt; β 2 and ∆ ⊂ X , we have DIS(B P ∆ ( h ⋆ , r )) ∈ {∅ , ∆ } (depending whether the x with h ⋆ ( x ) = 1 is in ∆ or not), so that P ∆ (DIS(B P ∆ ( h ⋆ , r ))) = 0 ; this immediately implies θ P ( ε ; τ ) ≤ 2 β (indeed, by careful reasoning, we can observe that any τ ≥ β 2 has θ P ( ε ; τ ) = 2 β -1 = s ). More generally, by (48), the bound in Theorem 27 is never worse than that in Theorem 3.

On the other hand, there are scenarios ( C , P ) where the opposite occurs, so that in general neither quantity θ P ( ε ; 5 β ) nor θ er P ( ε ) dominates the other. This is illustrated in the following example.

Example 7 ( θ P ( ε ; 5 β ) ≫ θ er P ( ε ) ) . Consider again the class from Appendix F.1 (and Example 5): that is, X = Z and C = C ts := { 1 {-t }∪ [ t, ∞ ) : t ∈ N } . Let ε, β ∈ (0 , 1 / 3) with ε ≪ β , and define P with marginal PX on X as defined in Appendix F.1: that is, n = 1 -2 β 2 ε , and for x ∈ { 1 , . . . , n } , PX ( { x } ) = 2 β n , PX ( {-x } ) = 2 ε . However, rather than the distributions P t described there, consider the family P ′ t , t ∈ { 1 , . . . , n } , with P ′ t ( Y = 1 | X = x ) = β +(1 -2 β ) 1 {-t }∪ [ t, ∞ ) ( x ) for every x ∈ X . These distributions represent a scenario with uniform classification noise . For P = P ′ t for any t ∈ { 1 , . . . , n } , letting h ⋆ = 1 {-t }∪ [ t, ∞ ) , it is easy to see that er P ( h ⋆ ) = inf h ∈ C er P ( h ) = β . Moreover, θ er P ( ε ) = θ P ( ε/ (1 -2 β )) ≤ θ P (0) ≤ 1 -2 β β +3 , where the last inequality was established in Appendix F.1. In contrast, we argued in Example 5 that θ P ( ε ; 5 β ) ≥ 1 -2 β 4 ε ≫ 1 -2 β β +3 ≥ θ er P ( ε ) . Thus, in this scenario, θ P ( ε ; 5 β ) is larger than θ er P ( ε ) .

A natural question is whether the gaps between θ P ( ε ; 5 β ) and θ er P ( ε ) in Examples 6 and 7 can also arise in cases where the smaller of the two corresponding query complexity bounds (either Theorem 27 or 30) is actually nearlysharp (in a minimax analysis over a family of distributions). This is straightforward to obtain, by defining a family of possible distributions P each obtained as a uniform mixture of one of the above two scenarios and the simple 2 -point construction of Kääriäinen (2006) giving rise to the β 2 ε 2 lower bound. For brevity, we omit the details of this.

The fact that Theorem 27 at least provides a starting point for P -dependent analysis which is never worse than the P -independent bound in Theorem 3 is a desirable feature. In contrast, the bound in Theorem 30 is sometimes better and sometimes worse than that in Theorem 3, so that one should be careful when using Theorem 30. Nonetheless, as illustrated in Example 7, there are at least some scenarios where θ er P ( ε ) may be useful for describing favorable scenarios (particularly concerning the Y | X conditional distribution).

## F.3 Querying in Subregions of the Region of Disagreement

In the active learning literature, one technique for going beyond disagreement-based queries is to query examples in a carefully selected subregion R ⊆ DIS( V ) of the region of disagreement of the set V of surviving concepts. This idea originates in the work of Balcan, Broder, and Zhang (2007) on margin-based active learning of homogeneous linear classifiers under certain marginals PX in realizable and Tsybakov-noise scenarios, and was extended to a technique for general concept classes and the agnostic case by Zhang and Chaudhuri (2014) (see Appendix A for further discussion of the history). For instance, the most well-known case of this technique providing improvements over disagreement-based queries (see Example 8 below) is homogeneous linear classifiers under a uniform distribution on a sphere (alternatively, any isotropic log-concave distribution), where the query complexity of this technique is ˜ O ( d ( β + ε ) 2 ε 2 ) (Zhang and Chaudhuri, 2014) (minimax optimal up to log factors), compared to disagreement-based active learning for which the best known bound is ˜ O ( d 3 / 2 · ( β + ε ) 2 ε 2 ) (Dasgupta, Hsu, and Monteleoni, 2007).

In this section, we show this technique is also compatible with the AVID principle, and propose a refinement of the AVID Agnostic algorithm which replaces D k -1 = DIS( V k -1 ) in Steps 2 and 9 with a well-chosen subregion R k -1 ⊆ D k -1 . We argue that this change does not affect the validity of Theorem 5, and admits refined P -dependent query complexity bounds compared to those presented in Appendix F.2. In particular, this shows that the AVID principle can recover the optimal query complexity of homogeneous linear classifiers under the uniform distribution, and generally any isotropic log-concave distribution (indeed, with improved log factors compared to prior works).

The basic argument (building from the original ideas of Balcan, Broder, and Zhang, 2007, and Zhang and Chaudhuri, 2014) is that, rather than querying all examples in S 1 k ∩ D k -1 \ ∆ i k in Step 2 of A avid , the algorithm identifies a subset of these examples Q k ⊆ S 1 k ∩ D k -1 \ ∆ i k which suffices for the purpose of updating V k in Step 3. Specifically, we aim to identify a subset Q k ⊆ S 1 k ∩ D k -1 \ ∆ i k for which, for any h, h ′ ∈ V k -1 ,

̸

<!-- formula-not-decoded -->

̸

In other words, most of the significant disagreements in X \ ∆ i k among concepts in V k -1 are captured in the Q k set. In particular, this retains the guarantees of Lemmas 15 and 16 with only minor adjustments to the constants in the bounds (accounting for the potential ε k 8 C ′ probability disagreements that are lost).

Formally, consider the algorithm A sub avid stated in Figure 2, where the Q k data subset is defined below. The values C, C ′ , C ′′ , N, m k and data subsets S 1 k , S 4 k are all as defined in A avid . The data subsets S 2 k , S 3 k,i are defined analogously to A avid except allocated in the corresponding steps of A sub avid : that is, if and when the algorithm reaches Step 2 with a value k , or reaches Step 9 (in which case let k = N +1 ), then for the value i k and the set ∆ i k as defined at that time in the algorithm, letting ˆ p k := 2 ˆ P S 4 k (∆ i k ) , the algorithm allocates to S 2 k the next m ′ k := ⌈ C ′′ c 2 1 ˆ p k ε 2 k ( d +log ( 4(3+ N -k ) 2 δ ))⌉ consecutive examples not previously allocated to any data subset, and likewise, if and when the algorithm reaches Step 5 with values ( k, i ) , it allocates to S 3 k,i the next m k consecutive examples which have not yet been allocated to any data subset.

We define the Q k data subset via a technique analogous to the work of Zhang and Chaudhuri (2014), specified via a discrete linear program with a finite number of constraints imposed by the set of realizable classifications of S 1 k . Let t k = ∑ k -1 k ′ =1 m k ′ , and recall S 1 k := { ( X t k +1 , Y t k +1 ) , . . . , ( X t k + m k , Y t k + m k ) } . The algorithm inductively constructs sets V k ⊆ C in Step 3 (analogous to the V k sets in A avid ). For any given k ∈ { 1 , . . . , N +1 } , denote by

<!-- formula-not-decoded -->

the set of V k -1 -realizable classifications of S 1 k . For the set ∆ i k (which is defined inductively based on previous rounds of the algorithm, analogously to the set ∆ i k in A avid ), consider the following integer linear program with binary variables 19 ζ 1 , 0 , ζ 1 , 1 , q 1 , . . . , ζ m k , 0 , ζ m k , 1 , q m k .

19 For simplicity, in this work, we present a technique based on an integer linear program, to arrive at a deterministic querying strategy. It is straightforward to extend the result to allow for non-integer solutions

̸

```
Algorithm A sub avid Input: Error parameter ε , Confidence parameter δ , Unlabeled data X 1 , . . . , X m Output: Classifier ˆ h 0. Initialize i = i 1 = 0 , ∆ 0 = ∅ , V 0 = C 1. For k = 1 , . . . , N 2. Query all examples in Q k and S 2 k ∩ ∆ i k 3. V k ← { h ∈ V k -1 : ˆ er 1,2 k ( h ) ≤ ˆ er 1,2 k ( ˆ h k ) + ε k C ′ } 4. If V k = ∅ or ˆ er 1,2 k ( ˆ h k ) < min h ∈ V k ˆ er 1,2 k ( h ) -ε k 4 C ′ , Then Return ˆ h := ˆ h k 5. While max f,g ∈ V k ˆ P S 3 k,i ( { f = g } \ ∆ i ) > ε k +2 6. ( f, g ) ← argmax ( f ′ ,g ′ ) ∈ V 2 k ˆ P S 3 k,i ( { f ′ = g ′ } \ ∆ i ) 7. ∆ i +1 ← ∆ i ∪ { f = g } , and update i ← i +1 8. i k +1 ← i 9. Query all examples in QN +1 and S 2 N +1 ∩ ∆ i N +1 and Return ˆ h := ˆ hN +1
```

̸

Figure 2: The SubregionAVID Agnostic algorithm.

<!-- formula-not-decoded -->

In particular, note that the solution only depends on the unlabeled examples X t k +1 , . . . , X t k + m k , and thus the algorithm may use the solution of this optimization problem when determining an appropriate set Q k of queries in Steps 2 and 9. Denote by q k 1 , . . . , q k m k the respective values of the q 1 , . . . , q m k variables at the solution found by LP k . Then define Q k as a subsequence of S 1 k :

<!-- formula-not-decoded -->

Let us generalize the definition of ˆ P S 1 k to involve intersections with the subsequence Q k : for any set A ⊆ X ×{ 0 , 1 } , ˆ P S 1 k ( A ∩ Q k ) := 1 m k ∑ m k t =1 q k t · 1 [( X t k + t , Y t k + t ) ∈ A ] , and ˆ P S 1 k (ER( f ) \ Q k ) := 1 m k ∑ m k t =1 (1 -q k t ) · 1 [( X t k + t , Y t k + t ) ∈ A ] . As usual, we also overload this notation for A ⊆ X , such as sets { f = g } , interpreting such sets A as synonymous with their labeled extension A ×{ 0 , 1 } .

The algorithm also relies on the following modifications to the definition of ˆ er 1,2 k :

<!-- formula-not-decoded -->

The definitions of V (4) k -1 and ˆ h k are then defined as in (2) and (3) based on the set V k -1 defined in A sub avid and the modified definition of ˆ er 1,2 k in (52). This completes the specification of the A sub avid algorithm.

We state a query complexity guarantee for this algorithm, phrased in terms of a variant of a subregion disagreement coefficient . As in Appendix F.2, we first present the known definition from the literature, which serves both as a starting point for the modified version and as a more-accessible quantity useful for upper-bounding the new quantity. Specifically, the following definition (a refinement of the disagreement coefficient from Definition 25) was proposed by Zhang and Chaudhuri (2014) (see also Hanneke, 2016b). 20

ζ t, 0 , ζ t, 1 ∈ [0 , 1] to the LP, resulting in a randomized querying strategy (see Zhang and Chaudhuri, 2014). This makes no significant difference to the query complexity bound (see Hanneke, 2016b, for a related discussion), but may be more attractive from a computational perspective.

20 The variant stated here is phrased slightly differently, to simplify the definition. In particular, φ P ( ε, 0) is equivalent to a quantity φ 01 c ( ε ) studied by Hanneke (2016b), which is only slightly different than the original

̸

̸

Definition 31. For any measure µ on X , any V ⊆ C , and any η ≥ 0 , define

̸

<!-- formula-not-decoded -->

For any distribution PX on X and any measurable h : X → { 0 , 1 } , for any ε, α ≥ 0 , define

<!-- formula-not-decoded -->

In particular, for any distribution P on X × { 0 , 1 } , letting h ⋆ be as in (10) (or see footnote 14), define φ P ( ε, α ) := φ P X ,h ⋆ ( ε, α ) .

̸

The quantity Φ P X ( V, η ) identifies the smallest PX ( R ) among regions R ⊆ X for which functions g ∈ V do not disagree much outside the region R (i.e., they have at most η disagreement with a fixed function f on X \ R ). In particular, we can upper bound Φ P X ( V, η ) by taking R = DIS( V ) and any f ∈ V , which satisfies sup g ∈ V PX ( { g = f } \ R ) = 0 ≤ η , so that Φ P X ( V, η ) ≤ PX (DIS( V )) . It immediately follows that the quantity φ P X ,h ( ε, α ) is never larger than the disagreement coefficient: φ P X ,h ( ε, α ) ≤ θ P X ,h ( α + ε ) . (53)

Indeed, there are several known examples of scenarios ( C , P X, h ⋆ ) where φ P ( ε, α ) is substantially smaller than θ P ( α + ε ) (Zhang and Chaudhuri, 2014). One example (discussed formally in Example 8 below) is the class C of homogeneous linear classifiers in R d under PX a uniform distribution on an origin-centered sphere, where θ P (0) = Θ ( √ d ) and φ P ( ε, α ) = O ( log ( α + ε ε )) (Hanneke, 2007b; Balcan, Broder, and Zhang, 2007; Zhang and Chaudhuri, 2014).

By (53) and (47), we also always have φ P ( ε, α ) ≤ s ∧ 1 α + ε for any ε, α ≥ 0 with α + ε ≤ 1 . Indeed, as with θ P ( ε ) in (47), this inequality turns out to be sharp in the worst case. Specifically, Hanneke (2016b) has shown that sup P X sup h ∈ C φ P X ,h ( ε, 0) = s ∧ 1 ε for ε ∈ (0 , 1] . Additionally, by definition we have φ P X ,h ( ε, α ) ≥ φ P X ,h ( α + ε, 0) . Since (53) implies φ P X ,h ( ε, α ) ≤ θ P X ,h ( α + ε ) , it immediately follows from combining this result of Hanneke (2016b) with (47) that for any ε, α ≥ 0 with α + ε ≤ 1 ,

<!-- formula-not-decoded -->

Due to (53) and the example in Appendix F.1, we know it is not possible to replace s with φ P ( ε, α ) in Theorem 3 for any α ≥ 0 (for any algorithm). However, similarly to the modification θ P ( ε ; τ ) of θ P ( ε ) presented in Appendix F.2, we can modify Definition 31 appropriately to provide a quantity suitable for developing a query complexity bound for A sub avid . Specifically, as in Definition 26, let us first generalize the definition of φ P X ,h ( ε, α ) to general measures µ : that is, for any measure µ on X and measurable function h : X → { 0 , 1 } , for any ε, α ≥ 0 , define φ µ,h ( ε, α ) := sup r&gt;α + ε Φ µ (B µ ( h,r ) , ( r -α ) / (36 CC ′′ )) r ∨ 1 . Then consider the following definition, representing a region-excluded subregion disagreement coefficient .

Definition 32. For any distribution P on X × { 0 , 1 } and any measurable ∆ ⊆ X , define a measure A ↦→ P ∆ ( A ) := PX ( A \ ∆) . For any ε &gt; 0 and α, τ ≥ 0 , for h ⋆ ∈ C as in (10) (or see footnotes 14, 17), define

<!-- formula-not-decoded -->

As was the case for θ P ( ε ; τ ) , we can equivalently define φ P ( ε, α ; τ ) as the subregion disagreement coefficient under a worst-case conditional distribution PX ( ·|X \ ∆) : that is,

<!-- formula-not-decoded -->

where we define φ P X ( ·|X\ ∆) ,h ⋆ ( ε/PX ( X \ ∆) , α/PX ( X \ ∆)) = 1 in the case PX ( X \ ∆) = 0 (which coincides with the value φ P ∆ ,h ⋆ ( ε, α ) for such ∆ ). In particular, combining this equivalent definition with (54) yields that

<!-- formula-not-decoded -->

quantity studied by Zhang and Chaudhuri (2014) in that it considers binary functions rather than fractional values in [0 , 1] . Hanneke (2016b) has shown this change to binary values makes little quantitative difference compared to the quantity of Zhang and Chaudhuri (2014).

Thus, replacing s in Theorem 3 by φ P ( ε, α ; τ ) would yield a (never-larger) P -dependent refinement.

As with θ P ( ε ; τ ) , the quantity φ P ( ε, α ; τ ) itself may often be challenging to calculate. Fortunately, again as with θ P ( ε ; τ ) , it can be upper-bounded by expressions that are more-easily calculated (though at the expense of some slack, so that they are no longer upper-bounded by s ). We might therefore think of φ P ( ε, α ; τ ) as an intermediate complexity measure (analogous to θ P ( ε ; τ ) ), which is useful in providing a starting point for a P -dependent refinement of s which is never larger than s , and which admits general upper bounds which are more accessible than directly calculating φ P ( ε, α ; τ ) . Specifically, it follows immediately from Definition 32 that we always have a lower bound φ P ( ε, α ) ≤ φ P ( ε, α ; τ ) , and an upper bound

<!-- formula-not-decoded -->

Making use of the quantity φ P ( ε, α ; τ ) to analyze A sub avid analogously to the analysis of A avid based on θ P ( ε ; τ ) in Theorem 27, we arrive at the following theorem.

Theorem 33 (Distribution-dependent Query Complexity of Subregion AVID Agnostic) . For any concept class C with VC( C ) &lt; ∞ , letting d = VC( C ) , for every distribution P on X × { 0 , 1 } , letting β = inf h ∈ C er P ( h ) , for any ε, δ ∈ (0 , 1) , if the algorithm A sub avid is executed with parameters ( ε, δ ) , with any number m ≥ M ( ε, δ ; β ) of i.i.d.P examples (for M ( ε, δ ; β ) as in Theorem 5, defined in Lemma 24), then with probability at least 1 -δ , the returned predictor ˆ h satisfies er P ( ˆ h ) ≤ inf h ∈ C er P ( h ) + ε and the algorithm makes a number of queries at most Q ( ε, δ ; P ) satisfying

<!-- formula-not-decoded -->

Proof Sketch. The proof of this theorem follows nearly identically to the proof of Theorem 5. We will merely highlight the changes compared to the original proof. Specifically, throughout the proof, we first replace all definitions from A avid with the corresponding definitions from A sub avid (e.g., ∆ i , i k , ˆ h k , V k , D k -1 = DIS( V k -1 ) , S 2 k , S 3 k,i , K , K , etc.) so that all definitions in the proof refer to the respective quantities in the A sub avid algorithm. Since the only definitional change in A sub avid compared to A avid is in the use of Q k rather than S 1 k ∩ D k -1 \ ∆ i k , to provide the ε error guarantee it will suffice to argue that the inequality (14) of Lemma 10 remains valid (only slightly larger) with this change: namely, on the event E 0 ,

̸

<!-- formula-not-decoded -->

Note that this is only larger than the bound in (14) by an additive ε k C ′′ (which we will argue below is inconsequential to the proof). As was true of (14), we argue that (58) in fact follows immediately from (18), as follows. Consider the values ζ k 1 , 0 , ζ k 1 , 1 , q k 1 , . . . , ζ k m k , 0 , ζ k m k , 1 , q k m k at the solution of the LP k optimization. Due to the first constraint in LP k , we know every f ∈ V k -1 has

<!-- formula-not-decoded -->

Moreover, due to the second constraint in LP k , for every f, g ∈ V k -1 , any X t k + t ∈ { f = g } has ζ k t, 1 -f ( X t k + t ) + ζ k t, 1 -g ( X t k + t ) + q k t = 1 , so that q k t = 0 = ⇒ ζ k t, 1 -f ( X t k + t ) + ζ k t, 1 -g ( X t k + t ) = 1 . Together, we have

̸

̸

<!-- formula-not-decoded -->

̸

Recall that every h ∈ V (3) k -1 is of the form h = DL( f ′ , g ′ , h ′ ) := f ′ 1 { f ′ = g ′ } + h ′ 1 { f ′ = g ′ } for some f ′ , g ′ ∈ V k -1 and h ′ ∈ C . Consider any two such functions h, h ′ ∈ V (3) k -1 , where h = DL( f 1 , g 1 , h 1 ) and h ′ = DL( f 2 , g 2 , h 2 ) for f 1 , g 1 , f 2 , g 2 ∈ V k -1 and h 1 , h 2 ∈ C . Note that

̸

<!-- formula-not-decoded -->

Therefore, the union bound implies

̸

<!-- formula-not-decoded -->

̸

̸

Also note that, due to the indicator 1 [ X t k + t / ∈ ∆ i k ] in the first constraint of LP k , at the solution to LP k , every X t k + t / ∈ ∆ i k has q k t = 0 , so that any h ∈ V (3) k -1 has ER( h ) ∩ Q k = (ER( h ) \ ∆ i k ) ∩ Q k . Altogether, we have that every h, h ′ ∈ V (3) k -1 satisfy

̸

<!-- formula-not-decoded -->

Together with (18) we arrive at the claimed inequality (58).

̸

In the context of the rest of the proof of Theorem 5, the only place (14) is used is in (28) in the proof of Lemma 15. In that context, substituting (58) yields the same conclusion: namely, for h 1 , h ′ 1 ∈ V (3) k -1 , since (20) of Lemma 12 implies P X ( { h 1 = h ′ 1 } \ ∆ i k ) ≤ 3 ε k , (58) implies

<!-- formula-not-decoded -->

where the last inequality follows from C ′′ ≥ 100 . Therefore, the conclusion of Lemma 15 remains valid (with the modified definition of ˆ er 1,2 k from (52)). The rest of the proof of the error bound (Lemma 19), and unlabeled sample size M ( ε, δ ; β ) (Lemma 24), and size of PX (∆ i k ) (Lemma 20) follow verbatim from this fact.

It remains only to establish the claimed bound Q ( ε, δ ; P ) on the number of queries. In the context of the proof of Theorem 5, this effectively means replacing (39) of Lemma 23 with a bound on | Q k | based on φ P ( ε, 0; 5 β ) , on an event E ′ 3 of probability at least 1 -δ 8 (which replaces the event E ′ 3 defined in the proof of Lemma 23).

Toward this end, consider any k ∈ { 1 , . . . , N +1 } having non-zero probability of k ≤ K . Given the event that k ≤ K and the random variables V k -1 and ∆ i k , fix a measurable function h k : X → { 0 , 1 } and a measurable set R k ⊆ X (dependent on V k -1 and ∆ i k but not on S 1 k ) such that

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Such a pair ( h k , R k ) is guaranteed to exist by the definition of Φ µ ( · , · ) in Definition 31.

We aim to argue that the constraints in LP k are satisfied by taking ζ t,h k ( X t k + t ) = 1 [ X t k + t / ∈ R k ] and q t = 1 [ X t k + t ∈ R k ] , via a uniform multiplicative Chernoff bound (Lemma 7 of Appendix D). Toward this end, define a collection ˜ A k of subsets of X :

̸

<!-- formula-not-decoded -->

Note that VC( ˜ A k ) ≤ d . Let ˜ δ k := δε k +3 144 . We bound ε ( m k , ˜ δ k ; ˜ A k ) by reasoning similar to the proof of Lemma 9. Specifically, we have

<!-- formula-not-decoded -->

̸

̸

̸

̸

where the last inequality is by c 0 ≥ 1 , C ′′ &gt; 144 C 3 , and ( C ′′ ) 150 / 108 &gt; 54 C ′′ . By Corollary 4.1 of Vidyasagar (2003), this implies so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Letting α = 2 3 , we therefore have ε ( m k , ˜ δ k ; ˜ A k ) &lt; α 2 4 ε k 6 C ′′ . Together with (60) and Lemma 7 of Appendix D, we have that with conditional probability at least 1 -˜ δ k given the event that k ≤ K and the random variables V k -1 , R k , and ∆ i k ,

̸

<!-- formula-not-decoded -->

By the law of total probability, there is an event E ′ 3 ,k of probability at least 1 -˜ δ k such that, on E ′ 3 ,k , if k ≤ K , then (63) holds. To unify notation, for any k ∈ { 1 , . . . , N +1 } having probability zero of k ≤ K , define E ′ 3 ,k as the event (of probability one) that k &gt; K , so that this conclusion also vacuously holds for such values k .

In particular, for any k ∈ { 1 , . . . , N + 1 } , suppose the events E ′ 3 ,k and k ≤ K occur. For each t ∈ { 1 , . . . , m k } , let ζ ′ t, 0 = 1 [ h k ( X t k + t ) = 0] 1 [ X t k + t / ∈ ( R k \ ∆ i k )] , ζ ′ t, 1 = 1 [ h k ( X t k + t ) = 1] 1 [ X t k + t / ∈ ( R k \ ∆ i k )] , q ′ t = 1 [ X t k + t ∈ R k \ ∆ i k ] . Note that these values satisfy the second and third constraints on ζ t, 0 , ζ t, 1 , q t in LP k . Moreover, (63) implies that ∀ g ∈ V k -1 ,

̸

<!-- formula-not-decoded -->

so that the first constraint in LP k is also satisfied by this choice of ζ t, 0 , ζ t, 1 , q t . Since the values ζ k t, 0 , ζ k t, 1 , q k t at the solution of LP k minimize ∑ m k t =1 q t among all choices of ζ t, 0 , ζ t, 1 , q t satisfying the constraints, we conclude that the above values of q ′ t satisfy

<!-- formula-not-decoded -->

Next we upper bound the right hand side of (64) via a multiplicative Chernoff bound (Lemma 6 of Appendix D). Consider again any k ∈ { 1 , . . . , N +1 } having non-zero probability of k ≤ K . Given the event k ≤ K and the random variables R k and ∆ k -1 , Lemma 6 of Appendix D implies that, with conditional probability at least 1 -˜ δ k ,

<!-- formula-not-decoded -->

where the last inequality follows from (62) and straightforward reasoning about numerical constant factors. By the law of total probability, there is an event E ′′ 3 ,k of probability at least 1 -˜ δ k on which, if k ≤ K , then (65) holds. To unify notation, for any k ∈ { 1 , . . . , N +1 } having probability zero of k ≤ K , also define E ′′ 3 ,k as the event (of probability one) that k &gt; K , so that this conclusion also vacuously holds for such values k .

Define E ′ 3 = ⋂ N +1 k =1 E ′ 3 ,k ∩ E ′′ 3 ,k . By the union bound, the event E ′ 3 fails with probability at most

<!-- formula-not-decoded -->

where the last inequality follows from our choice of C = 11 10 .

Altogether, on the event E ′ 3 , for every k ∈ { 1 , . . . , K } , (64), (65), and (61) together imply

<!-- formula-not-decoded -->

It remains to relate the right hand side of (66) to the quantity φ P ( ε, 0; 5 β ) . For the remainder of the proof, suppose the event E 0 ∩ E 1 ∩ E 2 ∩ E 3 holds (with E ′ 3 in the definition of E 3 from the proof of Lemma 23 replaced by the above definition of E ′ 3 ). Consider any k ∈ { 1 , . . . , K } . Recall that Lemma 17 implies h ⋆ ∈ V k -1 , which together with Lemma 12 implies V k -1 ⊆ B P ∆ i k ( h ⋆ , ε k ) . Thus, since the definition of Φ µ ( · , · ) is non-decreasing in its first argument, we have

<!-- formula-not-decoded -->

Also recall that Lemma 20 implies PX (∆ i k ) ≤ 5 β , so that ∆ i k is among the sets ∆ considered in the supremum in the definition of φ P ( ε k , 0; 5 β ) . Additionally, note that ε k ≥ ε N +1 &gt; ε 2 C . It follows that

<!-- formula-not-decoded -->

Altogether, we have that every k ∈ { 1 , . . . , K } satisfies

<!-- formula-not-decoded -->

Substituting | Q k | in place of | S 1 k ∩ D k -1 \ ∆ i k | in the proof of Lemma 23, and using the above bound on | Q k | in place of (39), we arrive at a bound Q ( ε, δ ; P ) on the total number of queries

<!-- formula-not-decoded -->

As with A avid , the algorithm A sub avid does not need to know the value φ P ( ε, 0; 5 β ) (or anything else about P ) to achieve this query complexity: that is, it is adaptive to the value φ P ( ε, 0; 5 β ) .

Together with (57), Theorem 33 further implies a (sometimes loose) relaxation in terms of φ P ( ε, 5 β ) , which is often easier to evaluate for given scenarios ( C , P ) . This is stated formally in the following corollary. As mentioned above, the example in Appendix F.1 shows that it is not generally possible to reduce the φ P ( ε, 5 β ) 2 dependence to a linear φ P ( ε, 5 β ) (or even any φ P (0 , α ) ).

Corollary 34. The query complexity bound Q ( ε, δ ; P ) in Theorem 33 (achieved by A sub avid ) satisfies

<!-- formula-not-decoded -->

Proof. Due to the first two inequalities in (57), the second term in the expression of Q ( ε, δ ; P ) in Theorem 27 is at most

<!-- formula-not-decoded -->

Relaxing d log ( 1 ε ) +log ( 1 δ ) ≤ log ( 1 ε ) ( d +log ( 1 δ )) and noting that

<!-- formula-not-decoded -->

and ( β + ε ε ) 2 ≤ 4 β 2 ε 2 +4 , the quantity (67) is at most

<!-- formula-not-decoded -->

Adding this to the first term in the expression of Q ( ε, δ ; P ) , the result follows.

<!-- formula-not-decoded -->

An immediate consequence of Corollary 34 is that, whenever sup α ∈ [0 , 5] φ P ( ε, α ) = polylog ( 1 ε ) , the dependence on β, ε in the query complexity bound in Theorem 33 is of order β 2 ε 2 +polylog ( 1 ε ) . As was true of Corollary 28, (56) implies the first upper bound in Corollary 34 is never larger than the upper bound in Theorem 1; however, this is not always the case for the second upper bound in Corollary 34. Moreover, unlike Theorem 33, both upper bounds in Corollary 34 can sometimes be loose compared to the s -dependent bound in Theorem 5 (e.g., Example 6 in Appendix F.2.2). For this reason, as with Theorem 27, Theorem 33 is useful despite having a quantity φ P ( ε, 0; 5 β ) that is more challenging to calculate, as it provides a starting point for P -dependent analysis that is at least never worse than Theorem 5.

To illustrate a well-known scenario where the technique presented in this subsection provides improvements over the basic A avid algorithm, consider the following example.

Example 8 (Homogeneous linear classifiers, uniform distribution) . As an implication of Corollary 34, we find that A sub avid recovers a near-optimal query complexity bound for learning homogeneous linear classifiers under any marginal PX that is isotropic log-concave. Let d ≥ 2 . For any x, w ∈ R d , denote by h w ( x ) = 1 [ ⟨ w,x ⟩ ≥ 0] . In this scenario, we suppose X = R d , C = { h w : w ∈ R d , ∥ w ∥ = 1 } (for which VC( C ) = d ), and PX is any isotropic log-concave distribution (Balcan and Long, 2013) (for instance, PX = Uniform( { x : ∥ x ∥ = 1 } ) is one such distribution). In other words, C is the class of linear classifiers whose hyperplane decision boundary passes through the origin. This scenario has a long history of interest in the active learning literature (see Section A), featuring prominently (with PX a uniform distribution) in the original A 2 paper of Balcan, Beygelzimer, and Langford (2005, 2006, 2009), which studied the case β ≲ ε/ √ d and showed a query complexity bound ˜ O ( d 2 log ( 1 ε ) log ( 1 δ )) in this regime. Later works refined this, via subregion-based techniques. Building on the works of Balcan, Broder, and Zhang (2007); Balcan and Long (2013) (which studied more-restrictive noise models), Zhang and Chaudhuri (2014) obtain a query complexity bound ˜ O ( d β 2 ε 2 + d ) . Here we argue this query complexity bound can be recovered from Corollary 34 (indeed, with improvements by log factors in the lead term). Specifically, Zhang and Chaudhuri (2014) show (based on results of Balcan and Long, 2013) that φ P ( ε, 5 β ) = O ( log ( β ε )) . Plugging into Corollary 34 (rather, the expression obtained in the proof thereof), we obtain a query complexity bound

<!-- formula-not-decoded -->

Compared to the result of Zhang and Chaudhuri (2014), this improves the lead term by a factor log 2 ( β ε ) (though at the expense of additional log factors in the lower-order term). We also note that this query complexity bound represents a refinement of what would be obtained from Corollary 28, since even for the special case of PX uniform on an origin-centered sphere, θ P ( β + ε ) = Θ ( √ d ∧ 1 β + ε ) (Hanneke, 2007b).

## F.4 Classes with Infinite VC Dimension via Covering Numbers

As one final remark about P -dependent query complexity bounds, we note that it is also possible to derive interesting query complexity improvements over passive learning even for classes with VC( C ) = ∞ under conditions on P commonly studied in the nonparametric passive learning literature: namely, bounded covering numbers .

̸

Denote by N ( ε, C , L 1 ( PX )) the minimal size of a proper ε -cover: that is, the size of the smallest C ′ ⊆ C for which sup h ∈ C min h ′ ∈ C ′ P X ( h = h ′ ) ≤ ε . Being able to construct such a cover from unlabeled examples requires some additional structure beyond finite covering numbers under PX (e.g.,

finite expected empirical covering numbers suffices; see e.g., van der Vaart and Wellner, 1996). Let us suppose such conditions are satisfied by ( C , PX ) , so that (since an active learner can be assumed to have access to an abundant supply of unlabeled examples) we may assume we have access to a valid ( ε/ 2) -proper-cover C ε/ 2 under L 1 ( PX ) of size O ( N ( ε/ 2 , C , L 1 ( PX ))) . Constructing this cover C ε/ 2 does not affect the query complexity, since it only requires the use of unlabeled examples.

We can then run A avid using C ε/ 2 in place of C . Since VC( C ε/ 2 ) = O (log( N ( ε/ 2 , C , L 1 ( PX )))) , and inf h ∈ C ε/ 2 er P ( h ) ≤ inf h ∈ C er P ( h ) + ε 2 =: β + ε 2 , we thereby obtain from Theorem 1 a PX -dependent query complexity bound

<!-- formula-not-decoded -->

This result can then be composed with bounds on the covering numbers N ( ε/ 2 , C , L 1 ( PX )) of various classes C under various conditions on PX known from the literature. For instance, this provides an improved query complexity for boundary fragment classes (a class defined by smoothness conditions on the decision boundaries of concepts in C ) under near-uniform distributions PX on [0 , 1] k +1 compared to the results established by Wang (2011) (see Wang, 2011; Tsybakov, 2004 for the precise definitions and covering numbers).

## G Extensions and Future Directions

We conclude with some extensions and several interesting open questions and future directions.

̸

Extension to Multiclass Classification: We can easily generalize the result to hold for multiclass classification : that is, where Y is a general label space, C is a family of measurable functions h : X → Y , P is a distribution on X × Y , and we still define er P ( h ) := P (( x, y ) : h ( x ) = y ) = P (ER( h )) . The exact same upper bound extends to this setting if we replace d with max { VC( A ) , d G } where A is as in Lemma 9 (replacing { 0 , 1 } with Y there) and d G denotes the graph dimension of C (Natarajan, 1989). The star number s is still defined as in Definition 2 (see Hanneke, 2024). The proof holds with only superficial modifications to rely solely on VC( A ) (for the X \ ∆ i k concentration) and d G (for concentration in ∆ i k ). We further note that this dimension max { VC( A ) , d G } is at most O ( d N ( C ) log( |Y| )) , where d N ( C ) is the Natarajan dimension of C (Natarajan, 1989); this follows by a similar argument as used to bound VC( A ) in the proof of Lemma 9, using a generalization of Sauer's lemma for the multiclass setting proven by Haussler and Long (1995).

For a bounded number of labels |Y| , this again leads to essentially optimal query complexity, as a lower bound Ω ( d N ( C ) β 2 ε 2 ) based on the Natarajan dimension d N ( C ) can be shown (similarly to the lower bound for binary classification).

However, for unbounded label spaces ( |Y| = ∞ ) the learnability and optimal sample complexity of passive learning in the realizable case are known to depend on a dimension called the DS dimension (Brukhim, Carmon, Dinur, Moran, and Yehudayoff, 2022; Daniely and Shalev-Shwartz, 2014) which is between the Natarajan dimension and graph dimension. This raises an important question: What is the optimal query complexity for multiclass agnostic active learning?

Extension to Stream-based Active Learning: For simplicity, we have defined the learning model as so-called pool-based active learning, in that the learning algorithm was given the entire sequence X 1 , . . . , X m of unlabeled examples as input, and can query any example, in any order. However, it is also common to consider an alternative protocol called stream-based active learning (or selective sampling ): namely, where the active learner observes the unlabeled examples X t one-at-a-time in sequence , and for each, decides whether or not to query, and can never revisit that decision later. In the literature on stream-based active learning, it is common to express the guarantees of the active learning in two parts: (1) a bound on the error guarantee expressed as a function of the number m of unlabeled examples processed, and (2) a bound on the number of queries it makes among the first m examples (e.g., Dasgupta, Hsu, and Monteleoni, 2007).

We note that A avid can easily be re-expressed as a stream-based active learner. Specifically, rather than limiting the ' For ' loop in Step 1 to N = O ( log ( 1 ε )) rounds, we can simply let the algorithm run until it has allocated as many unlabeled examples m as we wish. Rather than allocating all of the

S 1 k , S 4 k data subsets at the start, we can simply allocate these sets if and when the algorithm reaches the k th iteration of the ' For ' loop, at which point the algorithm collects the next m k examples to allocate to S 1 k , querying each of these examples X t iff X t ∈ D k -1 \ ∆ i k . Likewise, it then collects the next m k examples to allocate to S 4 k (without making any queries), to calculate the value m ′ k (where, in this case, we should suitably replace the value 3 + N -k in the log term in m ′ k to remove the dependence on ε : for instance, replacing it with k +2 would suffice for the present discussion). It then collects the next m ′ k examples to allocate to the data subset S 2 k , querying each of these examples X t iff X t ∈ ∆ i k . It then moves on to execute Steps 3-4. Similarly, upon each time it reaches Step 5, it simply collects the next m k unlabeled examples to construct S 3 k,i (without making any queries), which then enables it to execute Steps 5-7. We can execute this until any number m of unlabeled examples have been processed, and define the predictor at such a time as the ˆ h k for the last iteration k for which Step 2 was able to completely execute. If the algorithm ever satisfies the early stopping criterion in Step 4 for some iteration k , we can simply take ˆ h k as its final predictor. We can then derive the corresponding excess error bound and query bound from the above analysis of the query complexity and unlabeled sample complexity: namely, with probability at least 1 -δ , the predictor ˆ h produced after m unlabeled examples satisfies

<!-- formula-not-decoded -->

and its number of queries is bounded by

<!-- formula-not-decoded -->

Here the βm term is where the improvements over passive learning provided by the A VID principle are reflected in the query bound (as the above excess error bound is nearly as small as the best achievable excess error guarantees for passive learning with m labeled examples; Vapnik and Chervonenkis, 1974; Devroye and Lugosi, 1995; Hanneke, Larsen, and Zhivotovskiy, 2024b). In particular, the above guarantees compare favorably to previous analyses of stream-based active learning (e.g., Dasgupta, Hsu, and Monteleoni, 2007) in the regime of moderate-size β , where, for the same excess error guarantee, the bounds on the number of queries include a term such as ˜ O ( θ P ( β ) βm ) , which becomes of order ( s ∧ 1 β ) βm in the worst case over P , and hence is no better than m when s = ∞ . In contrast, in this regime of moderate-size β , where the βm term dominates, we obtain a factor β improvement in the number of queries.

We also remark that the analysis above also supplies an 'anytime' guarantee, where the algorithm can simply be executed indefinitely, and the above excess error bound and query bound hold simultaneously for every m (where, again, if the algorithm ever satisfies the condition in Step 4, its predictor should simply be defined as the corresponding ˆ h k forevermore, and it need not query any further examples in the sequence).

The Optimal Lower-Order Term: As discussed above, while the leading term in Theorem 3 is exacty optimal (perfectly matching a lower bound), the lower-order term in the upper bound in Theorem 3 presents a small gap (in the dependence on d ) compared to the best known lower bound (Hanneke and Yang, 2015). As discussed, some aspects of this gap (concerning 1 ε + d vs d ε ) cannot be improved if the dependence on C is only expressed via d and s : that is, without introducing new complexity measures. We leave open the question of formulating such an alwayssharp complexity measure, that is, the question: What is the optimal form of the query complexity Θ(QC a ( ε, δ ; β, C )) for all classes C ? However, aside from this gap, there is a gap which might be improvable even in expressions of the bound purely in terms of d and s : namely, the term s d in the upper bound. I conjecture this can be reduced to simply s : that is, QC a ( ε, δ ; β, C ) = O ( β 2 ε 2 ( d +log ( 1 δ )) ) + ˜ O ( min { s , d ε }) for every concept class C .

Proper Learning: As noted above, the A avid algorithm is an improper learner, meaning its returned predictor ˆ h might not be an element of the concept class C (rather, it is a shallow decision list built from concepts in C ). It is an interesting open question to determine whether there exist proper active learners achieving the query complexity bound in either Theorem 1 or 3 for every concept class C . It follows from Corollary 18 that, in the return case in Step 9, it would suffice to return ˆ h equal any element of VN . Thus, the main challenge in obtaining a proper learner is in the early-stopping case in Step 4. In this return case, we have effectively verified that er P ( ˆ h k ) is better than er P ( h ⋆ ) (Lemma 17). However, the resolution of the error estimates ˆ er 1,2 k at this stage might not yet be sufficient to find an h ∈ V k -1 nearly as good. Indeed, for this reason, any such early return case in an active learning algorithm may be problematic for proper learning.

On the other hand, we remark that, for all previous known separations between proper and improper sample complexities, the respective proofs break down if the learner is given access to the marginal distribution PX or a sufficiently large unlabeled data set (Bousquet, Hanneke, Moran, and Zhivotovskiy, 2020; Hanneke, Larsen, and Zhivotovskiy, 2024b; Daniely and Shalev-Shwartz, 2014; Montasser, Hanneke, and Srebro, 2019; Asilis, Devic, Sharan, and Teng, 2025a; Asilis, Høgsgaard, and Velegkas, 2025b). Since, for the purpose of merely bounding the query complexity , we may suppose an active learner has access to a large unlabeled data set, this hints that such improvements might indeed be achievable by proper active learners, or otherwise, a novel technique is needed for establishing such a separation between proper and improper active learning.

Computational Efficiency: The focus of this work has been solely on the information-theoretic query complexity of agnostic active learning, without any computational or resource constraints beyond the number of queries and unlabeled examples. However, computational considerations are of course also important to consider. To actually achieve the agnostic learning guarantee of ε excess error is typically thought to be computationally intractable for many concept classes, without distribution restrictions. Nonetheless, it would be interesting to determine whether, at least at some level, the improvements in the leading term reflected in Theorems 1 and 3 might also be reflected in a computationally efficient method, for some classes C (e.g., linear classifiers) under some restrictions on the distribution P which enable computational tractability yet for which such query complexity bounds are not captured by prior results (e.g., by θ P ( ε ) ).

Beyond this, a classical approach to obtaining computationally efficient algorithms in practice is to introduce convex relaxations of the various optimization problems involved in a given algorithm. In the literature on passive learning, the theory of error bounds for empirical risk minimization has been extended to allow for convex relaxations of the 0 -1 loss, called a surrogate loss , while still guaranteeing bounds on the excess error rate under appropriate assumptions on P relating excess surrogate risks to excess error rates (Bartlett, Jordan, and McAuliffe, 2006; Zhang, 2004). Prior work on disagreement-based active learning has been found to compose well with this theory of surrogate losses. Specifically, Hanneke and Yang (2019); Hanneke (2014) express disagreement-based active learning algorithms, in which the optimization problems defining the query criterion and the learner's final predictor are relaxed to convex programs expressed in terms of any given surrogate loss. For such algorithms, they derive query complexity bounds (based on the disagreement coefficient θ P ( ε ) ) holding under the same conditions studied by the passive learning works (Bartlett, Jordan, and McAuliffe, 2006; Zhang, 2004). It is thus a natural question to determine whether such a theory can be made to work for the algorithmic principles underlying A avid (i.e., the A VID principle), leading to an algorithm only requiring computationally tractable convex optimization problems based on a given surrogate loss, and expressing query complexity improvements over passive learning (of the type found in Theorems 1 and 3) under these same conditions on P relating excess surrogate risks to the excess error rates. This approach is made challenging in the context of A avid , due to its use of improper predictors ˆ h k , and even more-so due to the maximization in Steps 5 and 6 (whereas convex surrogate losses would typically only allow tractability of minimization problems).

As a step toward such a technique, an interesting intermediate question is whether Theorems 1 and 3 can be achieved by an active learning algorithm expressed as a reduction to an empirical risk minimization (ERM) oracle : that is, where the access to the concept class C is restricted to solving optimization problems of the form argmin h ∈ C ˆ er S ( h ) for data sets S (or possibly a weighted ERM). This would be particularly interesting if these data sets S are only constructed from subsets of the labeled examples ( X t , Y t ) queried by the algorithm (perhaps plus one additional example ( X t , y ) with an artificial label y , which may be needed when deciding whether to query X t ). Previous works

by Beygelzimer, Hsu, Langford, and Zhang (2010); Hsu (2010) have expressed disagreement-based active learning algorithms as reductions to such ERM oracles. It is therefore a natural question to consider whether the AVID principle can also be implemented based only on such oracles (and such an implementation could also be an important step toward enabling the above composition with the theory of surrogate losses).

Unlabeled Sample Complexity: Theorem 5 reveals that, to achieve the stated query complexity bound with A avid , it suffices to have access to a number of unlabeled examples M ( ε, δ ; β ) = O ( β + ε ε 2 ( d log ( 1 ε ) +log ( 1 δ )) ) . In comparison, we can obtain an obvious lower bound on the number of unlabeled examples necessary to achieve any query complexity bound by a lower bound on the sample complexity of fully-supervised passive learning (Devroye and Lugosi, 1995): i.e., Ω ( β + ε ε 2 ( d +log ( 1 δ )) ) . Thus, the upper bound M ( ε, δ ; β ) in Theorem 5 can be improved by at most a log ( 1 ε ) factor. This naturally raises the question: Is it possible to achieve a near-optimal query complexity Θ(QC a ( ε, δ ; β, C )) with an algorithm which uses a number of unlabeled examples O ( β + ε ε 2 ( d +log ( 1 δ )) ) ? Such a result would then be optimal simultaneously in both the number of queries and the number of unlabeled examples. To date, this is not even known to be achievable by fully-supervised passive learning, the best known upper bound having an additive ˜ O ( d ε ) term (Hanneke, Larsen, and Zhivotovskiy, 2024b). Thus, for now, a more-approachable question would be whether it is possible to match the query complexity bound in Theorem 3 using a number of unlabeled examples suboptimal only in log factors in the lower-order term, that is: Is there an algorithm achieving a query complexity upper bound O ( β 2 ε 2 ( d +log ( 1 δ )) ) + ˜ O (( s ∧ 1 ε ) d ) which uses a number of unlabeled examples at most O ( β ε 2 ( d +log ( 1 δ )) ) + ˜ O ( d ε ) ? As an intermediate step, it would already be interesting to determine whether this many unlabeled examples suffices to achieve the query complexity bound in Theorem 1.

Tsybakov Noise: Beyond the above directions, there are a number of further extensions of this work that seem ripe for exploration. One natural direction is extending the techniques in this work to the case of Tsybakov noise (Mammen and Tsybakov, 1999; Tsybakov, 2004; Massart and Nédélec, 2006). The optimal query complexity under Tsybakov noise was already identified by Hanneke and Yang (2015) (aside from similar gaps to the d ε vs 1 ε + d issue discussed above, which require introducing a new complexity measure to resolve). However, the algorithmic techniques in the present work are significantly simpler, and moreover, have the potential to dramatically reduce the number of unlabeled examples required for learning, compared to the technique of Hanneke and Yang (2015). I conjecture that the A VID principle is capable of yielding near-optimal query complexity guarantees under Tsybakov noise (with a number of unlabeled examples of the same order as the sample complexity of supervised learning, up to log factors); however, obtaining such guarantees may require a more-sophisticated usage of the principle, such as by the creation of multiple different regions ∆ , coinciding with different levels of variance of excess error estimates. Indeed, an analogous tiered allocation of queries was key to the original analysis of the query complexity under Tsybakov noise by Hanneke and Yang (2015).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The paper formally proves (appropriate formalizations of) the claims made in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper explicitly mentions a number of questions left open by this work (e.g., refining a lower-order term, proper learning, expression as a reduction to ERM, efficient relaxations via surrogate losses).

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

## Answer: [Yes]

Justification: The main body includes formal definitions and a proof outline, and a complete formal proof is provided in the supplemental material.

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

Justification: This is a purely theoretical work, and as such does not include experimental results.

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

Justification: This is a purely theoretical work, and as such does not include an experimental component relying on data or code.

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

Justification: This is a purely theoretical work, and as such does not include an experimental component.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This is a purely theoretical work, and as such does not include experimental results.

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

Justification: This is a purely theoretical work, and as such does not include an experimental component relying on computer resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed the NeurIPS Code of Ethics and fully conform to the guidelines.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a purely theoretical work, and as such is not likely to have broader societal impacts.

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

Justification: This is a purely theoretical work, and as such there is no associated data or model being released.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not make use of such assets. All relevant literature is properly cited.

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

Justification: The work in this paper does not involve crowdsourcing or human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The work in this paper does not involve human subjects or crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs were not involved in any aspect of this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
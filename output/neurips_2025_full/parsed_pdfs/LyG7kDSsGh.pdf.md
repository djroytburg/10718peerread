## On Traceability in /lscript p Stochastic Convex Optimization

Sasha Voitovych ∗

Mahdi Haghifam †

Idan Attias ‡

Gintare Karolina Dziugaite §

Roi Livni ¶

Daniel M. Roy ‖

## Abstract

In this paper, we investigate the necessity of traceability for accurate learning in stochastic convex optimization (SCO) under /lscript p geometries. Informally, we say a learning algorithm is m -traceable if, by analyzing its output, it is possible to identify at least m of its training samples. Our main results uncover a fundamental tradeoff between traceability and excess risk in SCO. For every p ∈ [1 , ∞ ), we establish the existence of an excess risk threshold below which every sample-efficient learner is traceable with the number of samples which is a constant fraction of its training sample. For p ∈ [1 , 2], this threshold coincides with the best excess risk of differentially private (DP) algorithms, i.e., above this threshold, there exist algorithms that are not traceable, which corresponds to a sharp phase transition. For p ∈ (2 , ∞ ), this threshold instead gives novel lower bounds for DP learning, partially closing an open problem in this setup. En route to establishing these results, we prove a sparse variant of the fingerprinting lemma, which is of independent interest to the community.

## 1 Introduction

Tracing or membership inference informally asks whether it is possible, using only the output of a learning algorithm, to distinguish samples in the training set from held-out samples. The existence of a tracer that identifies training examples reveals that the model has memorized specific examples rather than purely captured the underlying distribution [SSSS17; CCNSTT22]. In particular, understanding tracing has an important role in generalization theory, where an algorithm that is not traceable is known to generalize well beyond its training data [SZ20]. Tracing is also an important technical tool in, e.g., differential privacy (DP), where tracing attacks are the workhorse behind tight lower bounds for the risk-privacy trade-offs [BUV14]. From a privacy standpoint, even the leakage of a single example is viewed as catastrophic. However, from a generalization theory standpoint, we want to understand the exact relationship between an algorithm's generalization performance and the number of traceable examples.

To reason rigorously about tracing, following [DSSUV15], we define the problem of tracing as follows. Let A n be a learning algorithm that, given a training set S n = ( Z 1 , . . . , Z n ) of

∗ SV and MH are equal contribution authors.

∗ Institute for Data, Systems, and Society, Massachusetts Institute of Technology

† Khoury College of Computer Sciences, Northeastern University

‡ University of Illinois at Chicago; Toyota Technological Institute at Chicago

§ Google DeepMind

¶ Department of Electrical Engineering, Tel Aviv University

‖ Department of Statistical Sciences, University of Toronto; Vector Institute

n i.i.d. samples from some underlying distribution D , outputs a learned model ˆ θ . Then, a tracer T is a hypothesis tester that, given the model ˆ θ and a candidate point Z , outputs In if it believes Z was in S n , or Out otherwise. Formally, for some small soundness parameter ξ ∈ (0 , 1) and m ≤ n , we require T to satisfy:

<!-- formula-not-decoded -->

When such a tracer exists, we say that A n is ( ξ, m )-traceable. Equivalently, m is the expected number of samples in the training set S n for which the tracer outputs IN , and we refer to m as recall . (See Definitions 2.3 and 2.4.)

A fundamental problem in learning theory is investigating how an algorithm's generalization ability interacts with the information it retains about the training samples (including, in our language, its traceability). The common wisdom is that any information about the training set in a learned model is in tension with generalization [XR17; BMNSY18; SZ20]. On the other hand, non-traceable algorithms, such as differentially private algorithms, are often unable to reach optimal excess risk. The central question we study in this paper is: what is the exact tradeoff between the number of traceable examples and achievable excess risk?

This question was considered, first, in the context of mean estimation of a d -dimensional vector, in the seminal work of [BUV14; DSSUV15]. It was also studied in the context of Stochastic Convex Optimization (SCO) [SSSS09] in the work of [ADHLR24]. The work of [DSSUV15] studied the tradeoff between excess risk and the number of traceable examples, when a mechanism publishes an estimate of the mean that is accurate in every coordinate (i.e., the output of the algorithm has error of α with respect to /lscript ∞ norm to the true mean). At a high level, they showed that for every algorithm that has accuracy better than that achievable by a private algorithm, Ω(1 /α 2 ) examples are traceable on some hard instance. Notice that for the task of mean estimation in /lscript ∞ norm, the statistical sample complexity is Θ(log( d ) /α 2 ). Thus, for every algorithm, the preceding result only shows that it is possible to trace out a 1 / log( d ) fraction of the input samples. In contrast, [ADHLR24] exhibited an SCO problem in /lscript 2 geometry for which a constant fraction of the training samples are traceable. An important open problem, then, is to further explore and understand in which setups we expect a constant fraction of the training sample to be traceable.

In this work, we investigate this question, of traceability, in the fundamental learning setup of Stochastic Convex Optimization for general /lscript p geometries. We show that, in this general learning setup, when private learning is not possible, there is no meaningful gap between sample complexity and traceability. That is, in every geometry, there exist a hard problem for which every (sample-efficient) algorithm is traceable with a recall which is a constant fraction of its sample size. Due to connections between SCO and mean estimation problems, our results also extend to the latter settings; in particular, we close the log( d ) gap in the setting of [DSSUV15] and show that optimal traceability is dimension-dependent .

SCO is an ideal testbed for this problem: (1) as in modern machine learning practices, first-order methods are known to achieve optimal sample-complexity rates in this setting [Fel16; HRS16; AKL21], and (2) within this framework, we can design provable methods that mitigate tracing, such as DP algorithms [CMS11; BST14; BFTG19]. Therefore, by studying the problem of traceability in SCO, we also deepen our understanding of the interaction between privacy risks and sample-optimal learning.

To present our results, we recall the basic setup of SCO. An SCO problem is characterized via a triple P = ( Z , Θ , f ), where Z is the data space, Θ ⊂ R d is the parameter space, which must be convex, and f : Θ ×Z → R is a loss function such that f ( · , z ) is convex for all z ∈ Z . In SCO, data points are drawn from an underlying distribution D over Z , unknown to the learner. The objective of the learner is to minimize the expected risk based on observed samples. Then, a learning algorithm A n : Z n → Θ receives a sample S n = ( Z 1 , . . . , Z n ) of n data points from Z n and returns a (perhaps randomized) output in Θ. Then, for D ∈ M 1 ( Z ), expected risk is defined as F D ( θ ) := E Z ∼D [ f ( Z, θ )] . For an SCO problem to be learnable, one often assumes that the loss function f is Lipschitz and the diameter of the space Θ is bounded, both of which can be measured w.r.t. different norms. These bounds govern the behavior of learnability, but they can be measured in different geometries. A

canonical class of SCO problems is induced by the /lscript p norms, in which case we assume that Θ has bounded /lscript p -diameter and f ( · , z ) is /lscript p -Lipschitz, for a fixed p ∈ [1 , ∞ ].

## 1.1 Contributions

In this paper, we establish a fundamental tradeoff between traceability and excess risk for algorithms in the context of SCO in general geometries. Some settings in which tracing is not possible are already well-understood: in the excess risk regime where DP [DMNS06] is possible, no samples can be traced. Due to this observation, the problem of traceability is only meaningful outside the DP risk regime. More formally, let us define minimax statistical and DP excess risks in /lscript p SCO. Specifically, for a family of /lscript p Lipschitz problems L d p , we let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where, for concreteness, we take ε = 0 . 01 and δ = 1 /n 2 in the above.

Main Contribution: We show that every sample-efficient algorithm that achieves an excess risk outside the DP regime (that is, α = o ( α DP )) is traceable with recall proportional to the number of samples. The precise statement of our main contribution varies based on the geometry of SCO:

Tracing when p = 1 . For the case p = 1, we show that any learner whose excess risk is better, by a small polynomial factor, than the best risk attainable by a DP algorithm with constant ε and δ = 1 /n 2 must be traceable. Moreover, we give an essentially optimal lower bound on the number of samples that can be traced. In more detail, we show that there exists an /lscript 1 -SCO problem such that, if an algorithm achieves risk of

<!-- formula-not-decoded -->

then Ω(log( d ) /α 2 )) of the training samples can be traced (see Theorem 2.6). We note that the choice of the constant 0 . 01 above is arbitrary. It is instructive to compare our results to [DSSUV15]. While the settings of mean estimation and SCO are generally different, our lower bound for /lscript 1 geometry also extends to mean estimation in /lscript ∞ norm (see Corollary 2.9 for a formal argument). In both settings, the sample complexity scales like log( d ) /α 2 , however, [DSSUV15] showed traceability of only 1 / log( d ) fraction of the samples. On the other hand, in our work we show that there is no meaningful gap between sample complexity and traceability, and every sample-efficient algorithm outside the DP regime must memorize a constant fraction of its sample. Notably, our results also imply that traceability is dimension-dependent in this setup.

Tracing when p ∈ (2 , ∞ ) . For /lscript p SCO with p ≥ 2, in Theorem 2.7, we show that for

<!-- formula-not-decoded -->

where α DP is set as α DP = Θ ( d/n 2 ) 1 /p we can construct an SCO problem such that, if a learner achieves a risk of α , then Ω(1 /α p ) of its samples are traceable. Note that, the non-private sample complexity of learning for p &gt; 2 is precisely Θ(1 /α p ) in the relevant parameter regime, ∗∗ i.e., the number of traced out samples is of the order of the sample complexity. Note that, the optimal DP risk in this setup constitutes an open problem, and the quantity α DP above need not be the optimal DP risk in this setting. Nevertheless, this quantity can be shown to be a lower bound on the optimal DP risk (in the regime ε ∈ Θ(1)). We extend this result to other regimes of ε , and another important contribution of our work is proving such DP lower bounds.

∗∗ We point out that, in general, the sample complexity for p ≥ 2 scales as 1 /n 1 /p ∧ d 1 / 2 -1 /p / √ n , however, α stat /lessmuch α DP only when d ≳ n . Thus, the question of traceability is only non-vacuous in the overparameterized regime.

Table 1: Summary of traceability results. All results are stated up to constants. The sample complexity bounds are implied by Theorem A.1. Minimax DP rates are known due to [BFTG19; BGN21; AFKT21; GLLST23] and are displayed up to log factors and with δ = 1 /n 2 . ( † ) Although, in general, the sample complexity in this setting is a minimum of two terms, within the stated range of α , the term 1 /α p dominates.

| p        | Recall       | Range of α                                                            | Sample complexity   | Minimax DP rate       | Refs.    |
|----------|--------------|-----------------------------------------------------------------------|---------------------|-----------------------|----------|
| 1        | log( d ) α 2 | (√ log( d ) n , d 0 . 49 n √ log(1 /ξ ) )                             | log( d ) α 2        | √ log( d ) n + √ d εn | Thm. 2.6 |
| (1 , 2]  | 1 α 2        | (√ 1 n , √ d n √ log(1 /ξ ) )                                         | 1 α 2               | 1 √ n + √ d εn        | Thm. 2.5 |
| [2 , ∞ ) | 1 α p        | ( min { 1 n 1 /p , d 1 / 2 - 1 /p √ n } , ( d n 2 log(1 /ξ ) ) 1 /p ) | 1 α p †             | Open                  | Thm. 2.7 |

In particular, we provide an improved lower bound on DP-SCO under /lscript p geometries for p &gt; 2 in the high dimensional regime, i.e., d ≥ εn , which is arguably the most interesting regime as it is more relevant for the modern ML applications. Specifically, we show, in Theorem 2.8, that for all ε &lt; 1 and small δ we can construct a problem such that for every ( ε, δ )-DP algorithm, A n , there exists a data distribution such that:

<!-- formula-not-decoded -->

In particular, the above implies that when d ≥ εn , the risk due to privacy dominates the statistical risk. This result improves upon all previous best bounds in the literature when d ≥ εn [ABGMU22; LLL24]. In particular, Theorem 3.1 of [LLL24] gives a lower bound √ d/n 2 ε 2 , which is weaker than our lower bound for every p &gt; 2. Corollary 4 of [ABGMU22] gives a lower bound of min { ( 1 εn ) 1 p , d 1 -1 /p εn } which is weaker than our lower bound for d ≥ εn .

Tracing when p ∈ (1 , 2] . For each p ∈ (1 , 2], we show that there exists an /lscript p SCO problem such that, if an algorithm achieves excess risk of α ≲ α DP log 2 ( n ) , then Ω(1 /α 2 ) of its samples can be traced (see Theorem 2.5). This result uncovers a fundamental dichotomy between traceability and privacy in /lscript p SCO. It is known that p ∈ (1 , 2], Θ(1 /α 2 ) is precisely the sample complexity of learning /lscript p -Lipschitz problems [AWBR09]. We note that α DP for p = 2 is known due to [BST14]. However, as we discuss in Appendix B.1, combining [BST14] with the tracing results of [DSSUV15] does not give the optimal tracing of Θ(1 /α 2 ) samples.

## 1.1.1 Traceability beyond SCO: PAC Learning

A natural question is whether a similar phenomenon holds true for other learning setups. Consider the setting of binary classification PAC learning. We show that, for every class with VC dimension bounded by d vc , the recall of every tracer is in O ( d vc log 2 ( n )), i.e., it is at most a small fraction of the training sample provided n /greatermuch d vc . Since many such classes, including the class of thresholds, are not privately learnable [BNSV15; ALMM19; BLM20], the sharp transitions between privacy and traceability does not hold in PAC classification. We also point out that for the class of thresholds, we can remove the log 2 ( n ) factor from the recall upper bound. See Appendix H.

## 1.2 Technical contributions

Our technical contributions are elaborated on in Section 2.3. In essence, our technical novelties are twofold. First, we present a novel sparse fingerprinting lemma that, intuitively, shows that learners over sparse domains must be correlated to their samples. The key novelty of this result is that the correlation is inversely proportional to the sparsity parameter. This feature is not present in prior work, since fingerprinting lemmas are most often applied for learners/estimators over a hypercube domain.

Second, armed with this new fingerprinting result, we present a generic conversion result using a notion of a subgaussian trace value , which converts any lower bound on correlation

with the samples into a number of samples that can be traced. While it is well-appreciated by prior work that, conceptually, a fingerprinting lower bound implies a traceability lower bound, proving results for our setting of SCO involves complicated sparse domains embedded into /lscript p balls. This makes it more technically challenging to prove the necessary concentration phenomena holds for a tracer over the corresponding domain, which motivates us to restrict our attention to tracers that induce a subgaussian process over the domain.

## 1.3 Related Work

Our work is most similar in spirit to [DSSUV15; ADHLR24]. Our work builds on top of these results on a number of fronts. A key distinct aspect of our approach is the difference in the structure of hard problems and the new sparse fingerprinting lemma. Also, our generic traceability theory of subgussian trace value (Section 2.3) provides an abstract treatment of the approach in [DSSUV15]. Our approach allows to seamlessly convert fingerprinting lemmas into traceability results and even non-private sample complexity lower bound.

Our work also makes progress towards closing the gap regarding the optimal excess error for /lscript p DP-SCO for p &gt; 2. The best known upper bounds for DP-SCO in /lscript p geometry for p &gt; 2 are due to [BGN21; GLLST23], and Theorem 2.8 is the best lower bound.

To put our sparse fingerprinting lemma into the context of prior work, it can be seen to generalize the results of [SU17] to sparse sets. Another 'sparse fingerprinting lemma' in the literature is given by [CWZ23]. Our results are distinct by the way sparsity enters the lemmas: in [CWZ23] the mean vector is sparse (and data is dense), and in our case, the mean is dense and the data vectors are sparse. The proof techniques also differ substantially. Our sparse fingerprinting lemma is also an example of a fingerprinting lemma for the setting where the coordinates of the data vector are not independent, similar to [KMS22; LT24]. Additional related work is discussed in Appendix B.

## 2 Problem Setup and Main Results

We begin by some definitions. For a (measurable) space R , M 1 ( R ) denotes the set of all probability measures on R . In SCO, an α -learner is defined to be a learner whose expected excess risk is bounded by α . A formal definition is given below.

Definition 2.1 ( α -learner) . Fix α &gt; 0, n ∈ N and SCO problem (Θ , Z , f ). We say A n : Z n → M 1 (Θ) is an α -learner for (Θ , Z , f ) iff for every D ∈ M 1 ( Z ), we have E S n ∼D ⊗ n , ˆ θ ∼A n ( S n ) [ F D ( ˆ θ ) ] -inf θ ∈ Θ F D ( θ ) ≤ α.

In our work, we focus on learning Lipschitz-bounded families of problems, which are defined below. For every p ∈ [1 , ∞ ], let B p ( r ) = { θ ∈ R d : ‖ θ ‖ p ≤ r } be the unit ball in /lscript p norm.

Definition 2.2 (Lipschitz-bounded problems) . Fix p ∈ [1 , ∞ ], and let d &lt; ∞ be a natural number. We let L d p denote the set of all /lscript p -Lipschitz-bounded SCO problems in d dimensions. Namely, P = (Θ , Z , f ) ∈ L d p iff (i) Θ ⊂ B p (1), and (ii) for every θ 1 , θ 2 ∈ Θ and z ∈ Z , we have | f ( z, θ 1 ) -f ( z, θ 2 ) | ≤ ‖ θ 1 -θ 2 ‖ p .

## 2.1 Tracing

The key notion we study here is tracing , and we next introduce our framework for traceability. We consider families of tracers that assign each candidate point a real-valued score capturing how likely it is to have been seen during training. Then, the tracer converts these scores into binary In or Out decisions by thresholding the score. Intuitively the score corresponds to the likelihood of the event that the learner saw a data point during training.

Definition 2.3 (Tracer) . Fix data space Z and parameter space Θ. A tracer's strategy is a tuple of T = ( φ, D ) where φ : Θ ×Z → R and D ∈ M 1 ( Z ).

Definition 2.4 (( ξ, m )-traceability) . Let n ∈ N , ξ ∈ (0 , 1), and m ∈ N . We say a learning algorithm A n is ( ξ, m )-traceable if there exists a tracer ( φ, D ) and λ ∈ R such that, if ( Z 0 , Z 1 , . . . , Z n ) ∼ D ⊗ ( n +1) and ˆ θ ∼ A n ( Z 1 , . . . , Z n ), we have (i) Soundness: Pr ( φ ( ˆ θ, Z 0 ) ≥ λ ) ≤ ξ , and (ii) Recall: E [∣ ∣ { i ∈ [ n ] : φ ( ˆ θ, Z i ) ≥ λ } ∣ ∣ ] ≥ m .

## 2.2 Main Results

## 2.2.1 Traceability of α -Learners

In this section, we discuss our traceability results for accurate learners in /lscript p geometries. First, we will state a result that applies to p ∈ [1 , 2), and then present its slight refinement for p = 1. We will then present our result for p ≥ 2. See Appendices F.1 to F.3 for proofs.

Theorem 2.5. There exists a universal constant c &gt; 0 such that, for all p ∈ [1 , 2) , if d , n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 are such that

<!-- formula-not-decoded -->

then there exist an /lscript p SCO problem that every α -learner is ( ξ, m ) -traceable with m ∈ Ω ( α -2 ) .

Note that the upper bound on α in Equation (3) is precisely the optimal DP excess risk for ε ∈ Θ(1) and p ∈ [1 , 2] [AFKT21; BGN21], and the lower bound is precisely the optimal non-private risk (except p = 1; see Theorem A.1). Moreover, for p ∈ (1 , 2], the lower bound on m exactly matches the statistical sample complexity.

As mentioned above, for p = 1, the lower bound on recall in Theorem 2.5 is less than sample complexity by a factor of log( d ). This prompts us to establish the following refinement

Theorem 2.6. There exists a universal constant c &gt; 0 such that, if d is large enough and n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 are such that

<!-- formula-not-decoded -->

there exists a /lscript 1 SCO problem that every α -learner is ( ξ, m ) -traceable with m ∈ Ω ( log( d ) /α 2 ) .

Note that the upper bound in Equation (4) is slightly stronger than in Equation (3); however, the lower bound on recall now matches the sample complexity of learning in /lscript 1 geometry. We now present a result for p ≥ 2.

Theorem 2.7. There exists a universal constant c &gt; 0 such that, for all p ∈ [2 , ∞ ) , if d , n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 are such that

<!-- formula-not-decoded -->

there exist an /lscript p SCO problem such that every α -learner is ( ξ, m ) -traceable with m ∈ Ω(1 / (6 α ) p ) .

For p ∈ (2 , ∞ ), our results have a different implication, showing that all sufficiently accurate learners need to memorize a number of samples on the order of the sample complexity. However, in this case, the upper bound in Equation (5) need not be the optimal DP risk. Instead, it provides a lower bound on the optimal DP risk, as we will see next.

## 2.2.2 Improved DP-SCO Lower Bound for p &gt; 2

Theorem 2.8. Let p ∈ [2 , ∞ ) . There exist a universal constant c &gt; 0 and an /lscript p SCO problem P = (Θ , Z , f ) such that every ( ε, δ ) -DP learner of P with ε ≤ 1 and δ ≤ c/n satisfies,

<!-- formula-not-decoded -->

## 2.2.3 Consequences for mean estimation

Consider the setting of mean estimation in /lscript ∞ norm as in [DSSUV15]. Our results in Theorem 2.6 extend almost verbatim to this setting.

Corollary 2.9. Let Z = {± 1 } d , and suppose an estimator is given such that, given access to i.i.d. samples Z 1 , . . . , Z n ∈ Z , outputs ˆ µ with E ‖ ˆ µ -E [ Z 1 ] ‖ ∞ ≤ α/ 2 . Then, there exists a universal constant c &gt; 0 such that, if d is large enough and n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 satisfy Equation (4) , then the estimator ˆ µ is ( ξ, m ) -traceable with m ∈ Ω ( log( d ) /α 2 ) .

## 2.3 Roadmap of the proof

Our proofs rely on introducing two key technical elements that allow us to generalize tracing techniques to general /lscript p setups. The first element is a generic conversion result involving a complexity notion which we term the subgaussian trace value of a problem. As we show in the proof of Theorem 2.8, we can use the subgaussian trace value to prove traceability results over general domains, establish DP sample complexity lower bounds, and, even to recover non-private sample complexity lower bounds. While the connection between the first two aspects is well-known [FS17], we find the ability of trace value to recover non-private lower bounds surprising.

Our second technical contribution concerns techniques for lower bounding the subgaussian trace value, which we accomplish through several novel fingerprinting lemmas . Previous works used the standard fingerprinting lemma, where the learner observes points on a hypercube, to lower bound DP and traceability in /lscript 2 geometry. However, when moving to general /lscript p geometries, this setup no longer captures the hardest settings to learn. For instance, for p &gt; 2, canonical instances of hard problems involve data drawn from sparse sets [AWBR09]. We thus prove new fingerprinting lemmas that enable us to leverage our framework in such settings. These fingerprinting lemmas are then applied to carefully constructed instances of hard problems, and we show that every accurate learner of these problems is traceable.

## 3 General framework: subgaussian trace value

We next describe more formally the framework of subgaussian tracers. For a random variable X , the subgaussian norm of X is the quantity ‖ X ‖ ψ 2 := inf { t : E [ exp( X 2 /t 2 ) ] ≤ 2 } [Ver18]. We use the following definition of a subgaussian process:

Definition 3.1 (Subgaussian process) . We call an indexed collection of random variables { X θ } a σ -subgaussian process w.r.t a metric space (Θ , ‖·‖ ) if for every θ, θ ′ ∈ Θ, we have (i) ‖ X θ -X θ ′ ‖ ψ 2 ≤ σ ‖ θ -θ ′ ‖ , and (ii) ‖ X θ ‖ ψ 2 ≤ σ diam ‖·‖ (Θ) .

For origin symmetric convex body Θ, let ‖·‖ Θ denote the Minkowski norm w.r.t. Θ, that is ‖ x ‖ Θ := inf { λ &gt; 0: x ∈ λ Θ } . If Θ is not convex or not origin symmetric, we let ‖·‖ Θ be the Minkowski norm w.r.t. convex hull of (Θ ∪ -Θ). Note that ‖·‖ Θ is the minimal norm to contain Θ in its unit ball.

Definition 3.2 (Subgaussian tracer) . Fix κ ∈ R to be a constant, and let Θ be a convex body. We let T κ be the class of subgaussian tracers at scale κ &gt; 0, that is, a tracer ( φ, D ) ∈ T κ iff

- (i) { φ ( θ, Z ) } θ ∈ Θ where Z ∼ D is a 1-subgaussian process w.r.t. (Θ , ‖·‖ Θ ).
- (ii) | φ ( θ, z ) | ≤ κ for all θ ∈ Θ and z ∈ Z .

Definition 3.3 (Subgaussian trace value) . Fix n ∈ N , α ∈ [0 , 1], and κ ∈ R . Consider an arbitrary SCO problem P = (Θ , Z , f ). Let T κ be as in Definition 3.2. Then, we define the subgaussian trace value of problem P by

<!-- formula-not-decoded -->

where the inf is taken over all A n that achieve excess risk ≤ α on P with n samples.

Traceability via subgaussian trace value. The subgaussian trace value characterizes the average score the pair ( φ, D ) assigns to the data points in the training set. However, the definition of recall in Definition 2.3 requires characterizing the number of samples in the training set that takes a large value. The former can be converted into the latter, provided the sum of squared scores of samples is not too large. A formal statement, which is a

consequence of Paley-Zygmund inequality, can be found in Lemma A.11. In the next lemma, we show how to control the sum of squares of the φ ( ˆ θ, Z i ) using the subgaussian assumption.

Lemma 3.4. Fix n, d ∈ N . Suppose Θ ⊂ R d is a subset of a unit ball in some norm ‖·‖ . Let φ : Θ ×Z → R and D ∈ M 1 ( Z ) be such that, as Z ∼ D , { φ ( θ, Z ) } is a σ -subgaussian process w.r.t. (Θ , ‖·‖ ) . Let ( Z 1 , . . . , Z n ) ∼ D ⊗ n . Then, there is a constant C &gt; 0 , such that

<!-- formula-not-decoded -->

Equipped with this lemma, in the next theorem, we show that if, the subgaussian trace value of a problem is large, then every α -learner is traceable.

Theorem 3.5. Fix n ∈ N , d ∈ N , κ &gt; 0 and α ∈ [0 , 1] . Consider an arbitrary SCO problem P = (Θ , Z , f ) . Let T = Tr κ ( P ; n, α ) be the subgaussian trace value of P . Then, for some constant c &gt; 0 , every α -learner A n is ( ξ, m ) -traceable with

<!-- formula-not-decoded -->

Privacy lower bounds via subgaussian trace value. In the next theorem, we show that the notion of subgaussian trace value directly lower bounds the best privacy parameters achievable by a DP algorithm. The proof is based on [FS17].

Theorem 3.6. There exists a universal constant c &gt; 0 , such that the following holds. Fix p ∈ [1 , ∞ ) , n ∈ N , d ∈ N , α ∈ [0 , 1] , κ &gt; 0 ε &gt; 0 , and δ ∈ [0 , 1] . Consider an arbitrary SCO problem P = (Θ , Z , f ) in R d . Let T = Tr κ ( P ; n, α ) be the subgaussian trace value of problem P . Then, for every ( ε, δ ) -DP α -learner A n , we have exp( ε ) -1 ≥ c ( T -2 δκ ) .

Non-private sample complexity via subgaussian trace value. Surprisingly, if we directly use subgaussian trace value , we can recover optimal sample complexity bounds for all p ∈ [1 , ∞ ) and all regimes of ( d, α ), thus unifying traceability with private and non-private sample complexity lower bounds. While we detail the argument formally in Appendix G, we consider here a helpful example of /lscript 2 geometry. First, it can be shown that we always have Tr( P ; n, α ) ≲ √ d/n for arbitrary problem P (see Proposition G.1). Also, we will later show that, for every α &gt; 0, there exist an /lscript 2 problem P with Tr( P ; n, α ) ≳ √ d/nα (see Theorem 5.1). Combining these two inequalities gives n ≳ 1 /α 2 , which is optimal.

## 4 The sparse fingerprinting lemma

By introducing the notion of subgaussian trace value, we have reduced the problems of traceability and privacy lower bounds to the question of lower bounding the subgaussian trace value. Now, we discuss the techniques to lower bound subgaussian trace value. The proofs can be found in Appendix D. Due to space limitation, we only discuss the details for the case of p &gt; 1 and present the details of p = 1 in Appendix E.2.

For /lscript 2 geometry, one can lower bound subgaussian trace value using the classical fingerprinting lemma in [DSSUV15]. While this strategy leads to traceability results in /lscript 2 geometry, examples of hard problems for /lscript p geometry with p &gt; 2 are those with sparse sets Z (e.g., as in [AWBR09]). This motivates us to prove the following sparse fingerprinting lemma , which is another important contribution of our work. For a vector x ∈ R d , let supp( x ) be the set of its non-zero coordinates and denote ‖ x ‖ 0 = | supp( x ) | .

<!-- formula-not-decoded -->

Definition 4.1 (Sparse distributions family) . Fix d ∈ N , k ∈ [ d ] and µ ∈ [ -k/d, k/d ] d . Consider the mixture distribution on Z k = { z ∈ { 0 , ± 1 } d : ‖ z ‖ 0 = k } given by, for all z ∈ Z k , D µ,k ( z ) = E J ∼ unif (( [ d ] k )) [ P µ,k,J ( z )] , where

✶ j ∈ J Note that, in particular, E Z ∼D µ,k [ Z ] = µ . Intuitively, one can think of sampling from D µ,k using the following procedure: (i) sample the support coordinates J ∼ unif ( [ d ] k ) , (ii) for each j ∈ J , sample Z j from {± 1 } with mean d k µ j independently, (iii) for each j /negationslash∈ J , set Z j = 0.

With this distribution family at hand, we may state the sparse fingerprinting lemma. For x, y ∈ R d and a subset R ⊆ [ d ] of coordinates, we use 〈· , ·〉 S to denote the inner product 〈 x, y 〉 R := ∑ i ∈ R x i y i . Also, for α, β, γ &gt; 0, let s -beta [ -γ,γ ] ( α, β ) be the symmetric betadistribution, i.e., beta distribution with parameters α, β scaled and shifted to have support [ -γ, γ ] (see Definition A.13).

Lemma 4.2 (Sparse fingerprinting) . Fix d, n ∈ N and let k ∈ [ d ] . For each µ ∈ [ -k/d, k/d ] d , let Z k and D µ,k be as in Definition 4.1 . Let π = s -beta [ -k/d,k/d ] ( β, β ) ⊗ d be a prior and set

<!-- formula-not-decoded -->

Then, for every learning algorithm A n : Z n →M 1 ( R d ) with sample S n = ( Z 1 , . . . , Z n ) ,

<!-- formula-not-decoded -->

The key novelty of this lemma is that it provides a way to study the correlation between a learner's output and training samples on sparse sets Z k . An important and distinctive feature of this result is that the right-hand side scales by a factor of d/k , highlighting the fact that sparse problems correspond to greater subgaussian trace values. Intuitively, this stems from the fact that each coordinate is seen fewer times by the learning algorithm, meaning it must retain more information from each training sample in order to learn accurately. Additionally, for the special case k = d , the result precisely recovers the fingerprinting lemma from [SU17].

## 5 Final steps: bounding the subgaussian trace value for hard problems

Finally, we go over the construction of hard problems. To illustrate the difficulty of problem constructions, we give an example of a problem that requires many samples to learn but nevertheless is not traceable. Consider learning over /lscript 1 ball with linear loss. Let

<!-- formula-not-decoded -->

Consider a difficult set of distributions {D i } d i =1 where D i is a product distribution on Z and has mean α on coordinate i and mean zero on all other coordinates. It can be shown this problem requires Θ(log( d ) /α 2 ) samples to learn up to risk of α/ 3, and ERM is an optimal learner. However, after seeing Θ(log( d ) /α 2 ) samples from D i , the ERM takes the value ˆ θ = e i w.h.p., which is also the population risk minimizer. In other words, it becomes impossible to trace out any specific samples on which ˆ θ was trained.

Generic construction for p ∈ (1 , ∞ ) . As mentioned above, to obtain optimal results for p &gt; 2, problems constructed need to be sparse, and the main subtlety in our constructions is choosing the sparsity parameter. For some k ∈ [ d ] to be chosen later, consider the following /lscript p -Lipschitz problem P k,p .

<!-- formula-not-decoded -->

Here, the parameter space Θ is the largest /lscript ∞ ball inscribed into the unit /lscript p ball, and q is the Hölder conjugate of p , i.e., 1 p + 1 q = 1. The next step is to show that α -learners for the above problem must be correlated with the mean of the unknown data distribution. Let D be a distribution with mean µ , and suppose A n is an α -learner for Equation (7). Then,

<!-- formula-not-decoded -->

Now, we apply the sparse fingerprinting lemma (Lemma 4.2). A key step is choosing the scale β ≥ 1 of the beta-prior. On the one hand, β should be small enough to guarantee E d -1 /p ‖ µ ‖ 1 &gt; k 1 /q α , so that the above lower bound is non-vacuous. On the other hand, taking β too small decreases the sample complexity of learning the problem, thus, disallowing

the desired level of recall. The optimal choice is β ∝ α -2 · ( k/d ) 1 /p , as long as this quantity is ≥ 1. This choice yields

<!-- formula-not-decoded -->

where κ ∈ Θ(1), and, for some universal constant c &gt; 0, we let

<!-- formula-not-decoded -->

Note that the d 1 /p / √ k scaling ensures φ induces a 1-subgaussian process. Finally, it remains to choose a suitable value for k , for each pair ( p, α ). Recall the definition of P k,p from Equation (7).

Theorem 5.1. Let P k,p be the family of problems described in Equation (7) . There exist universal constants c 1 , c 2 &gt; 0 such that, for all α ∈ (0 , 1 / 6] and d ∈ N , the following subgaussian trace value lower bounds hold for all p ∈ [1 , ∞ ) and κ ≤ c 1 √ d :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the reduction Theorem 3.5, the above establishes Theorems 2.5 and 2.7.

Refinement for p = 1 . While the above construction also yields a traceability result for p = 1, it is suboptimal for the following simple reason: for k = d , the problem in Equation (7) only requires Θ(1 /α 2 ) samples to learn, thus, it is impossible to trace out Ω(log( d ) /α 2 ) samples. On the other hand, the problem in Equation (6) requires Θ(log( d ) /α 2 ) samples to learn but is not traceable. The intuition we follow here is to modify the construction in Equation (7) to make Θ 'look' more like an /lscript 1 -ball to drive up the sample complexity while still avoiding the counterexample with an ERM learner from the beginning of the section. In particular, we consider the following /lscript 1 -problem,

<!-- formula-not-decoded -->

for a suitably chosen s ∈ [ d ]. Note that, if we choose s /greatermuch 1, Θ above is a polytope with much more vertices (2 s ( d s ) ) than an /lscript 1 ball (2 d ), which would intuitively force a learner like an ERM to reveal more information about the training sample. On a technical level, selecting large s improves the subgaussian constant of a tracer; however, selecting s that is too large shrinks the diameter of the set, and thus, the problem becomes easier to learn. We must trade off these two aspects, and carefully set the value of s . As it turns out, the optimal choice is s ∝ d 1 -c for any small c &gt; 0 in order to establish Theorem 2.6. The remainder of the proof is rather technical and hence is deferred to Appendix F.2.

## 6 Limitations

We conclude by stating an intriguing open problem. We conjecture Theorem 2.8 is tight, and the dichotomy between traceability and SCO also holds for p &gt; 2. In particular, we conjecture that the optimal DP-SCO excess risk for /lscript p with p &gt; 2 scales as

<!-- formula-not-decoded -->

ignoring log(1 /δ ) factors. If the conjecture is true, we have a complete understanding of traceability in SCO. If it is false, it reveals that there is something fundamentally different about settings with p &gt; 2, which would also significantly enrich our understanding of DP-SCO.

## Acknowledgments

The authors would like to thank Mufan Li and Ziyi Liu for their comments on the drafts of this work.

## Funding

This work was completed while Sasha Voitovych was a student at the University of Toronto and supported by an Undergraduate Student Research Award from the Natural Sciences and Engineering Research Council of Canada. Mahdi Haghifam is supported by a Khoury College of Computer Sciences Distinguished Postdoctoral Fellowship. Idan Attias is supported by the National Science Foundation under Grant ECCS-2217023, through the Institute for Data, Econometrics, Algorithms, and Learning (IDEAL). Roi Livni is supported by a Google fellowship, a Vatat grant and the research has been funded, in parts, by an ERC grant (FoG - 101116258). Daniel M. Roy is supported by the funding through NSERC Discovery Grant and Canada CIFAR AI Chair at the Vector Institute.

## References

| [AWBR09]   | A. Agarwal, M. J. Wainwright, P. Bartlett, and P. Ravikumar. 'Information-theoretic lower bounds on the oracle complexity of convex optimization'. Advances in Neural Information Processing Systems 22 (2009).                |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ALMM19]   | N. Alon, R. Livni, M. Malliaris, and S. Moran. 'Private PAC learning implies finite Littlestone dimension'. In: Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing . 2019, pp. 852- 860.               |
| [AKL21]    | I. Amir, T. Koren, and R. Livni. 'SGD generalizes better than GD (and regularization doesn't help)'. In: Conference on Learning Theory . PMLR. 2021, pp. 63-92.                                                                |
| [ABGMU22]  | R. Arora, R. Bassily, C. Guzmán, M. Menart, and E. Ullah. 'Differen- tially private generalized linear models revisited'. Advances in neural information processing systems 35 (2022), pp. 22505-22517.                        |
| [AFKT21]   | H. Asi, V. Feldman, T. Koren, and K. Talwar. 'Private stochastic convex optimization: Optimal rates in l1 geometry'. In: International Conference on Machine Learning . PMLR. 2021, pp. 393-403.                               |
| [ADHLR24]  | I. Attias, G. K. Dziugaite, M. Haghifam, R. Livni, and D. M. Roy. 'Information Complexity of Stochastic Convex Optimization: Applications to Generalization and Memorization'. arXiv preprint arXiv:2402.09327 (2024).         |
| [BFTG19]   | R. Bassily, V. Feldman, K. Talwar, and A. Guha Thakurta. 'Private stochastic convex optimization with optimal rates'. Advances in neural information processing systems 32 (2019).                                             |
| [BGN21]    | R. Bassily, C. Guzmán, and A. Nandi. 'Non-Euclidean differentially private stochastic convex optimization'. In: Conference on Learning Theory . PMLR. 2021, pp. 474-499.                                                       |
| [BMNSY18]  | R. Bassily, S. Moran, I. Nachum, J. Shafer, and A. Yehudayoff. 'Learn- ers that Use Little Information'. In: Algorithmic Learning Theory . 2018, pp. 25-55.                                                                    |
| [BST14]    | R. Bassily, A. Smith, and A. Thakurta. 'Private empirical risk mini- mization: Efficient algorithms and tight error bounds'. In: 2014 IEEE 55th annual symposium on foundations of computer science . IEEE. 2014, pp. 464-473. |
| [Bat08]    | N. Batir. 'Inequalities for the gamma function'. Archiv der Mathematik 91.6 (2008), pp. 554-563.                                                                                                                               |
| [BHMZ20]   | O. Bousquet, S. Hanneke, S. Moran, and N. Zhivotovskiy. 'Proper learning, Helly number, and an optimal SVM bound'. In: Conference on Learning Theory . PMLR. 2020, pp. 582-609.                                                |

[BBFST21] G. Brown, M. Bun, V. Feldman, A. Smith, and K. Talwar. 'When is memorization of irrelevant training data necessary for high-accuracy learning?' In: Proceedings of the 53rd annual ACM SIGACT symposium on theory of computing . 2021, pp. 123-132.

[BNSV15] M. Bun, K. Nissim, U. Stemmer, and S. Vadhan. 'Differentially private release and learning of threshold functions'. In: 2015 IEEE 56th Annual Symposium on Foundations of Computer Science . IEEE. 2015, pp. 634649.

[BLM20] M. Bun, R. Livni, and S. Moran. 'An equivalence between private classification and online prediction'. In: 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS) . IEEE. 2020, pp. 389-402.

[BUV14] M. Bun, J. Ullman, and S. Vadhan. 'Fingerprinting codes and the price of approximate differential privacy'. In: Proceedings of the forty-sixth annual ACM symposium on Theory of computing . 2014, pp. 1-10.

- [CCNSTT22] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramer.

[CWZ23] T. T. Cai, Y. Wang, and L. Zhang. 'Score attack: A lower bound technique for optimal differentially private learning'. arXiv preprint arXiv:2303.07152 (2023).

'Membership inference attacks from first principles'. In: 2022 IEEE Symposium on Security and Privacy (SP) . IEEE. 2022, pp. 1897-1914.

C. Cheng, J. Duchi, and R. Kuditipudi. 'Memorize to generalize: on the necessity of interpolation in high dimensional linear regression'. In: Conference on Learning Theory . PMLR. 2022, pp. 5528-5560.

[CMS11] K. Chaudhuri, C. Monteleoni, and A. D. Sarwate. 'Differentially private empirical risk minimization.' Journal of Machine Learning Research 12.3 (2011).

[CDK22]

[DMNS06] C. Dwork, F. McSherry, K. Nissim, and A. Smith. 'Calibrating noise to sensitivity in private data analysis'. In: Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3 . Springer. 2006, pp. 265-284.

[Fel16] V. Feldman. 'Generalization of ERM in Stochastic Convex Optimization: The Dimension Strikes Back'. In: Advances in Neural Information Processing Systems . Ed. by D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett. Vol. 29. Curran Associates, Inc., 2016.

- [DSSUV15] C. Dwork, A. Smith, T. Steinke, J. Ullman, and S. Vadhan. 'Robust traceability from trace amounts'. In: 2015 IEEE 56th Annual Symposium on Foundations of Computer Science . IEEE. 2015, pp. 650669.

[Fel20] V. Feldman. 'Does learning require memorization? a short tale about a long tail'. In: Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing . 2020, pp. 954-959.

- [FKT20] V. Feldman, T. Koren, and K. Talwar. 'Private stochastic convex

[FS17]

optimization: optimal rates in linear time'. In: Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing . 2020, pp. 439-449.

- [GLLST23] S. Gopi, Y. T. Lee, D. Liu, R. Shen, and K. Tian. 'Private convex

V. Feldman and T. Steinke. 'Generalization for adaptively-chosen estimators via stable median'. In: Conference on learning theory . PMLR. 2017, pp. 728-757.

optimization in general norms'. In: Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) . SIAM. 2023, pp. 5068-5089.

- [HDMR21] M. Haghifam, G. K. Dziugaite, S. Moran, and D. Roy. 'Towards a unified information-theoretic framework for generalization'. Advances in Neural Information Processing Systems 34 (2021), pp. 26370-26381.

[HNKRD20]

|                | 'Sharpened generalization bounds based on conditional mutual infor- mation and an application to noisy, iterative algorithms'. Advances in Neural Information Processing Systems 33 (2020), pp. 9925-9935.                                                                                                                                                                                           |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [HRTSMK23]     | M. Haghifam, B. Rodríguez-Gálvez, R. Thobaben, M. Skoglund, D. M. Roy, and G. Karolina Dziugaite. 'Limitations of Information- Theoretic Generalization Bounds for Gradient Descent Methods in Stochastic Convex Optimization'. In: Proceedings of The 34th In- ternational Conference on Algorithmic Learning Theory . Vol. 201. Proceedings of Machine Learning Research. PMLR, 2023, pp. 663-706. |
| [HRS16]        | M. Hardt, B. Recht, and Y. Singer. 'Train faster, generalize better: Stability of stochastic gradient descent'. In: International conference on machine learning . PMLR. 2016, pp. 1225-1234.                                                                                                                                                                                                        |
| [HSRDTMPSNC08] | N. Homer, S. Szelinger, M. Redman, D. Duggan, W. Tembe, J. Muehling, J. V. Pearson, D. A. Stephan, S. F. Nelson, and D. W. Craig. 'Resolving individuals contributing trace amounts of DNA to highly complex mixtures using high-density SNP genotyping microar- rays'. PLoS genetics 4.8 (2008), e1000167.                                                                                          |
| [KOV17]        | P. Kairouz, S. Oh, and P. Viswanath. 'The Composition Theorem for Differential Privacy'. IEEE Transactions on Information Theory 63.6 (2017), pp. 4037-4049.                                                                                                                                                                                                                                         |
| [KLSU18]       | G. Kamath, J. Li, V. Singhal, and J. Ullman. 'Privately Learning High-Dimensional Distributions'. arXiv preprint arXiv:1805.00216 (2018).                                                                                                                                                                                                                                                            |
| [KLSU19]       | G. Kamath, J. Li, V. Singhal, and J. Ullman. 'Privately learning high-dimensional distributions'. In: Conference on Learning Theory . PMLR. 2019, pp. 1853-1902.                                                                                                                                                                                                                                     |
| [KMS22]        | G. Kamath, A. Mouzakis, and V. Singhal. 'New lower bounds for private estimation and a generalized fingerprinting lemma'. Advances in neural information processing systems 35 (2022), pp. 24405-24418.                                                                                                                                                                                              |
| [LLL24]        | Y. T. Lee, D. Liu, and Z. Lu. The Power of Sampling: Dimension-free Risk Bounds in Private ERM . 2024. arXiv: 2105.13637 [cs.LG] .                                                                                                                                                                                                                                                                   |
| [LW86]         | N. Littlestone and M. Warmuth. 'Relating data compression and learnability' (1986).                                                                                                                                                                                                                                                                                                                  |
| [Liv24]        | R. Livni. 'Information theoretic lower bounds for information theoretic upper bounds'. Advances in Neural Information Processing Systems 36 (2024).                                                                                                                                                                                                                                                  |
| [LT24]         | X. Lyu and K. Talwar. 'Fingerprinting Codes Meet Geometry: Im- proved Lower Bounds for Private Query Release and Adaptive Data Analysis'. arXiv preprint arXiv:2412.14396 (2024).                                                                                                                                                                                                                    |
| [MY16]         | S. Moran and A. Yehudayoff. 'Sample compression schemes for VC classes'. Journal of the ACM (JACM) 63.3 (2016), pp. 1-10.                                                                                                                                                                                                                                                                            |
| [NY83]         | A. S. Nemirovskij and D. B. Yudin. 'Problem complexity and method efficiency in optimization' (1983).                                                                                                                                                                                                                                                                                                |
| [SDSOJ19]      | A. Sablayrolles, M. Douze, C. Schmid, Y. Ollivier, and H. Jégou. 'White-box vs black-box: Bayes optimal strategies for membership inference'. In: International Conference on Machine Learning . PMLR. 2019, pp. 5558-5567.                                                                                                                                                                          |
| [SSSS09]       | S. Shalev-Shwartz, O. Shamir, N. Srebro, and K. Sridharan. 'Stochastic Convex Optimization.' In: COLT . Vol. 2. 4. 2009, p. 5.                                                                                                                                                                                                                                                                       |
| [SSSS17]       | R. Shokri, M. Stronati, C. Song, and V. Shmatikov. 'Membership Infer- ence Attacks against Machine Learning Models'. In: IEEE Symposium on Security and Privacy . 2017.                                                                                                                                                                                                                              |
| [ST10]         | K. Sridharan and A. Tewari. 'Convex Games in Banach Spaces'. In: COLT . 2010, pp. 1-13.                                                                                                                                                                                                                                                                                                              |

|         | selection'. In: 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS) . IEEE. 2017, pp. 552-563.                                                                        |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [SZ20]  | T. Steinke and L. Zakynthinou. 'Reasoning about generalization via conditional mutual information'. In: Conference on Learning Theory . PMLR. 2020, pp. 3437-3452.                          |
| [Ste16] | T. A. Steinke. 'Upper and lower bounds for privacy and adaptivity in algorithmic data analysis'. PhD thesis. 2016.                                                                          |
| [Ver18] | R. Vershynin. High-dimensional probability: An introduction with ap- plications in data science . Vol. 47. Cambridge university press, 2018.                                                |
| [Wai19] | M. J. Wainwright. High-dimensional statistics: A non-asymptotic view- point . Vol. 48. Cambridge university press, 2019.                                                                    |
| [XR17]  | A. Xu and M. Raginsky. 'Information-theoretic analysis of general- ization capability of learning algorithms'. In: Advances in Neural Information Processing Systems . 2017, pp. 2524-2533. |

## Appendix Contents

| Introduction   | Introduction                                                                                   | Introduction                                                                                   | Introduction                                                                                   |   1 |
|----------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-----|
|                | 1.1                                                                                            | Contributions . . . . . . . .                                                                  | . . . . . . . . . . .                                                                          |   3 |
|                |                                                                                                | 1.1.1 Traceability beyond SCO: PAC Learning . . . . . . . . . . .                              | 1.1.1 Traceability beyond SCO: PAC Learning . . . . . . . . . . .                              |   4 |
|                | 1.2                                                                                            | Technical contributions . .                                                                    | . . . . . . . . . . . .                                                                        |   4 |
|                | 1.3                                                                                            | Related Work . . . .                                                                           | . . . . . . . . . . . . . . .                                                                  |   5 |
| 2              | Problem                                                                                        | Setup and                                                                                      | Main Results                                                                                   |   5 |
|                | 2.1                                                                                            | Tracing . . . .                                                                                | . . . . . . . . . . . . . . . . . .                                                            |   5 |
|                | 2.2                                                                                            | Main Results . . . . . . . . .                                                                 | . . . . . . . . . .                                                                            |   6 |
|                |                                                                                                | 2.2.1                                                                                          | Traceability of α -Learners . . . . . . . .                                                    |   6 |
|                |                                                                                                | 2.2.2                                                                                          | Improved DP-SCO Lower Bound for p >                                                            |   6 |
|                |                                                                                                | 2.2.3                                                                                          | Consequences for mean estimation . . .                                                         |   6 |
|                | 2.3                                                                                            | Roadmap of the proof .                                                                         | . . . . . . . . . . . . .                                                                      |   7 |
| 3              | General framework: subgaussian trace value                                                     | General framework: subgaussian trace value                                                     | General framework: subgaussian trace value                                                     |   7 |
| 4              | The                                                                                            | sparse fingerprinting                                                                          | lemma                                                                                          |   8 |
| 5              | Final                                                                                          | steps: bounding                                                                                | the subgaussian trace value                                                                    |   9 |
| 6              | Limitations                                                                                    | Limitations                                                                                    | Limitations                                                                                    |  10 |
| A              | Additional preliminaries                                                                       | Additional preliminaries                                                                       | Additional preliminaries                                                                       |  17 |
|                | A.1                                                                                            | Background on SCO .                                                                            | . . . . . . . . . . . . . .                                                                    |  17 |
|                | A.2                                                                                            | Differential                                                                                   | Privacy . . . . . . . . . . . . . . . .                                                        |  18 |
|                | A.3                                                                                            | Concentration inequalities .                                                                   | . . . . . . . . . . .                                                                          |  18 |
|                | A.4                                                                                            | Beta distributions .                                                                           | . . . . . . . . . . . . . . .                                                                  |  19 |
| B              | Additional Related Work                                                                        | Additional Related Work                                                                        | Additional Related Work                                                                        |  21 |
|                | B.1 Detailed comparison with [DSSUV15; BST14]. . .                                             | B.1 Detailed comparison with [DSSUV15; BST14]. . .                                             | B.1 Detailed comparison with [DSSUV15; BST14]. . .                                             |  21 |
| C              | Proofs from Section 3                                                                          | Proofs from Section 3                                                                          | Proofs from Section 3                                                                          |  22 |
|                | C.1 Proof of Lemma 3.4                                                                         | C.1 Proof of Lemma 3.4                                                                         | . . . . . . . . . . . . . . .                                                                  |  22 |
|                | C.2 Proof of Theorem 3.5 .                                                                     | C.2 Proof of Theorem 3.5 .                                                                     | . . . . . . . . . . . . . .                                                                    |  24 |
|                | C.3 Proof of Theorem 3.6                                                                       | C.3 Proof of Theorem 3.6                                                                       | . . . . . . . . . . . . . . .                                                                  |  26 |
| D              | Proofs of fingerprinting lemmas (Section 4)                                                    | Proofs of fingerprinting lemmas (Section 4)                                                    | Proofs of fingerprinting lemmas (Section 4)                                                    |  26 |
|                | D.1 Proof of Lemma 4.2 . . . . . . . . . . .                                                   | D.1 Proof of Lemma 4.2 . . . . . . . . . . .                                                   | . . . .                                                                                        |  26 |
|                | D.2 Fingerprinting for /lscript 1 setup. . . . . . . . . . . . . . . . . . . . . . . . . . . . | D.2 Fingerprinting for /lscript 1 setup. . . . . . . . . . . . . . . . . . . . . . . . . . . . | D.2 Fingerprinting for /lscript 1 setup. . . . . . . . . . . . . . . . . . . . . . . . . . . . |  28 |

bounds

30

E.1

Proofs for

-geometries (Theorem 5.1) .

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

30

|     | /lscript p E.2 Proofs for /lscript 1 -geometry . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   |   33 |
|-----|------------------------------------------------------------------------------------------------------------|------|
| F   | Proofs of the main results (Section 2.2) . . . . . . . . . . . . . . . . . . . . . . . . .                 |   36 |
| F.1 | Proof of Theorem 2.5 . . . . .                                                                             |   36 |
| F.2 | Proof of Theorem 2.6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                           |   37 |
| F.3 | Proof of Theorem 2.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                           |   38 |
| F.4 | Proof of Theorem 2.8 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                           |   38 |
| F.5 | Proof of Corollary 2.9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                           |   39 |
| G   | Connection between subgaussian trace value and non-private sample com- plexity                             |   40 |
| G.1 | Lower bounds for p ∈ (1 , ∞ ) . . . . . . . . . . . . . . . . . . . . . . . . . .                          |   40 |
| G.2 | Lower bounds for p = 1. . . . . . . . . . . . . . . . . . . . . . . . . . . . .                            |   42 |
| H   | Traceability of VC classes (Section 1.1.1)                                                                 |   44 |

## A Additional preliminaries

## A.1 Background on SCO

The next proposition summarizes the known minimax rates for learning SCO problems in general geometries. A proof can be found in [NY83; AWBR09; ST10].

Theorem A.1. Fix p ∈ [1 , ∞ ] , d ∈ N , and n ∈ N . Let α stat ( L d p , n ) be the minimax excess risk rate of learning /lscript p -Lipschitz-bounded problems, as defined in Equation (1) . Then,

1. For p = 1 , we have
2. For 1 &lt; p ≤ 2 , we have

<!-- formula-not-decoded -->

3. For 2 ≤ p &lt; ∞ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark A.2 . Notice that, in the overparameterized regime ( d ≥ n ), the minimax excess risk for p ≥ 2 is Θ (( 1 n ) 1 /p ) which is dimension-independent. This shows that for d ≥ n , in all geometries except p = { 1 , ∞} , the minimax excess risk is dimension-free. /triangleleft

This proposition implies the following corollary on the minimum number of samples required for α -learners.

Corollary A.3. Fix p ∈ [1 , ∞ ] , d ∈ N , and α ∈ (0 , 1] . Let N stat ( L d p , n ) be the sample complexity of learning problems L d p up to excess risk α , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then,

1. For p = 1 , we have
2. For 1 &lt; p ≤ 2 , we have

<!-- formula-not-decoded -->

3. For 2 ≤ p &lt; ∞ , we have
4. For p = 1 , we have
4. For p = ∞ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 Differential Privacy

Definition A.4. Let ε &gt; 0 and δ ∈ [0 , 1). A randomized mechanism A n : Z n →M 1 (Θ) is ( ε, δ )-DP, iff, for every two neighboring datasets S n ∈ Z n and S ′ n ∈ Z n (that is, S n , S ′ n differ in one element), and for every measurable subset M ⊆ Θ, it holds

<!-- formula-not-decoded -->

Algorithms that satisfy DP are not traceable in the sense of Definition 2.4 [KOV17]. The following simple proposition formalizes this observation.

Proposition A.5. Fix n ∈ N and ε, δ &gt; 0 . Let A n be an ( ε, δ ) -DP algorithm. Then, if A n is ( ξ, m ) -traceable, it holds that

<!-- formula-not-decoded -->

## A.3 Concentration inequalities

First, we collect lemmata on the subgaussian norm, introduced in Section 3, which we use to derive concentration inequalities. The following is Equation (2.14) in [Ver18], and shows that a bound on subgaussian norm immediately leads to concentration inequalities.

Lemma A.6 (Subgaussian concentration) . There exists a universal constant C such that the following holds for every random variable X with ‖ X ‖ ψ 2 &lt; ∞ : for every t ≥ 0 ,

<!-- formula-not-decoded -->

The subgaussian norm behaves nicely under the summation of independent random variables. The following is Proposition 2.6.1 in [Ver18].

Lemma A.7 (Sum of subgaussian variables) . Let C &gt; 0 be a universal constant. Let X 1 , . . . , X n be a collection of arbitrary independent real random variables. Then,

<!-- formula-not-decoded -->

Subgaussian norm also behaves nicely under mixtures. In particular, we have the following proposition.

Proposition A.8 (Subgaussian mixtures) . Let { X α } α ∈ A be σ -subgaussian random variables, and let π be a distribution over the index set A . Then, a mixture of { X α } α ∈ A under α ∼ π is also σ -subgaussian.

Proof. Let Y be such mixture. Then, for every t &gt; 0, we have

<!-- formula-not-decoded -->

Plugging in t = σ into above, and using that ‖ X α ‖ ψ 2 ≤ σ for all α , we have

<!-- formula-not-decoded -->

i.e., ‖ Y ‖ ψ 2 ≤ σ , as desired.

It is well-known that bounded random variables are subgaussian (Equation (2.17) of [Ver18]).

Proposition A.9. Suppose X is a random variable such that X ∈ [ -b, b ] almost surely. Then,

<!-- formula-not-decoded -->

for some universal constant C &gt; 0 .

We will heavily use the following result for the supremum of subgaussian processes (which follows from [Ver18, Theorem 8.1.6]). Let N (Θ , ‖·‖ , ε ) denote the covering number of Θ in norm ‖·‖ at scale ε &gt; 0.

Proposition A.10. Let { X θ } θ ∈ Θ be a σ -subgaussian process w.r.t. a metric space (Θ , ‖·‖ ) as per Definition 3.1, and further assume that Θ is contained in the unit ball of ‖·‖ . Let t ≥ 0 be arbitrary. Then, with probability at least 1 -4 exp( -t 2 )

<!-- formula-not-decoded -->

for some universal constant C &gt; 0 .

Proof. Fix an arbitrary θ 0 ∈ Θ. Using Theorem 8.1.6 [Ver18], we obtain the following bound for the increment of the subgaussian process { X θ } ,

<!-- formula-not-decoded -->

First, note that for ε ≥ 1, N (Θ , ‖·‖ , ε ) = 1, since Θ lies in the unit ball of ‖·‖ . Thus,

<!-- formula-not-decoded -->

Note that, by triangle inequality, we have

<!-- formula-not-decoded -->

Since { X θ } θ ∈ Θ satisfies Definition 3.1, we have

<!-- formula-not-decoded -->

From Lemma A.6, we then have

<!-- formula-not-decoded -->

for some constant c &gt; 0. Combining this with Equation (10) and taking a union bound with Equation (9), we get

<!-- formula-not-decoded -->

for some absolute constant C &gt;

<!-- formula-not-decoded -->

The following lemma is an anti-concentration inequality based on Paley-Zygmund inequality. It shows that if the sum of variables is large, one can conclude that many of them are large given an appropriate control over their sum of squares. It is given as Lemma A.4 in [ADHLR24], and it is also similar to Lemma 25 in [DSSUV15].

Lemma A.11. Fix n ∈ N and ( a 1 , . . . , a n ) ∈ R n . Let A 1 := ∑ i ∈ [ n ] a i and A 2 := ∑ i ∈ [ n ] ( a i ) 2 . Then, for every β ∈ R , ∣ ∣ { i ∈ [ n ] : a i ≥ β/n } ∣ ∣ ≥ (max { A 1 -β, 0 } ) 2 A 2 .

## A.4 Beta distributions

Next definitions are the versions of beta distributions that we use in this paper. Recall that, classically, beta distribution is supported on [0 , 1]. However, in our results, it is convenient to consider the rescaled and centered variants.

Definition A.12. Fix β &gt; 0. A (symmetric) beta distribution denoted by s -beta ( β, β ) is a continuous distribution, such that, if X ∼ s -beta ( β, β ), then, for every a ∈ [ -1 , 1], we have

<!-- formula-not-decoded -->

where B ( β ) = 2 2 β -1 Γ( β ) 2 / Γ(2 β ).

Definition A.13. Fix β &gt; 0 and γ ∈ (0 , 1]. We define rescaled (symmetric) beta distribution, denoted by s -beta [ -γ,γ ] ( β, β ), where for a ∈ [ -γ, γ ], its distribution is given by

<!-- formula-not-decoded -->

where B ( β ) = 2 2 β -1 Γ( β ) 2 / Γ(2 β ).

We have the following result on the first moment of the beta distribution.

Lemma A.14. Fix β &gt; 0 . Let X ∼ s -beta ( β, β ) where β ≥ 1 . Then,

<!-- formula-not-decoded -->

Proof. Let B ( β ) = 2 2 β -1 Γ( β ) 2 / Γ(2 β ) be the normalization constant. We have

<!-- formula-not-decoded -->

It remains to upper bound B ( β ). It follows from Theorem 1.5 of [Bat08] that, for every x ≥ 1, we have where a = √ 2 e and b = 2 π are absolute constants. Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the last line we used β -1 / 4 ≥ 3 4 β which holds as β ≥ 1. Thus,

<!-- formula-not-decoded -->

as desired.

Since the density of the rescaled beta distribution is homogeneous w.r.t. γ , we have the following result.

Corollary A.15. Fix β ≥ 1 and γ ∈ (0 , 1] . Let X ∼ s -beta [ -γ,γ ] ( β, β ) . Then,

<!-- formula-not-decoded -->

## B Additional Related Work

Necessity of memorization in learning. A parallel line of work investigated memorization using the notion of label memorization in supervised setups. As per this definition, a learner is said to memorize its training samples if it 'overfits' at these points. Feldman [Fel20] showed that, in some classification tasks, if the underlying distribution is long-tailed , then a learner is forced to memorize many training labels. Cheng, Duchi, and Kuditipudi [CDK22] showed this phenomenon also occurs in the setting of linear regression. While this framework is suitable to study memorization in supervised tasks, the notion of 'labels' in SCO in not well-defined and thus calls for alternative definitions.

Another line of work studied memorization through the lens of information theoretic measures. Brown, Bun, Feldman, Smith, and Talwar [BBFST21] used input-output mutual information (IOMI) as a memorization metric and showed that IOMI can scale linearly with the training sample's entropy, indicating that a constant fraction of bits is memorized. In the context of SCO in /lscript 2 geometry, lower bounds on IOMI have been studied in [HRTSMK23; Liv24]. Specifically [Liv24] demonstrated that, for every accurate algorithm, its IOMI must scale with dimension d . Our approach to the study of memorization is conceptually different since we focus on the number of samples memorized as opposed to the number of bits . Nevertheless, it can be shown using Lemma H.3 and [HNKRD20, Thm. 2.1] that the recall lower bounds IOMI of an algorithm (provided that soundness parameter ξ is small enough, e.g., ξ = 1 /n 2 ). However, because of the Lipschitzness of loss functions in /lscript p SCO, we can use discretization of Θ and design algorithms with IOMI that is significantly smaller that the entropy of the training set, thus, memorization in the sense of [BBFST21] does not arise here.

Membership inference. Membership inference is an important practical problem [HSRDTMPSNC08; SSSS17; CCNSTT22]. In these works, the focus is on devising strategies for the tracer in modern machine learning settings, particularly neural networks. Our work takes a more fundamental perspective, aiming to determine whether membership inference is inherently unavoidable or simply a byproduct of specific training algorithms. An interesting aspect of our results is that, for 1 &lt; p ≤ 2, the optimal strategy for tracing depends only on the loss function, which is in line with empirical studies [SDSOJ19].

Private Stochastic Convex Optimization. DP-SCO has been extensively studied in /lscript 2 geometry (see, for instance, [CMS11; BST14; BFTG19; FKT20]). For /lscript p with p ∈ [1 , 2), the optimal DP excess risk was established in [AFKT21; BGN21]. The best known upper bounds for DP-SCO in /lscript p geometry for p &gt; 2 are due to [BGN21; GLLST23]. In this setting, there is a long-standing gap between upper and lower bounds, and the best known lower bounds are due to [ABGMU22; LLL24], which our paper improves on.

## B.1 Detailed comparison with [DSSUV15; BST14].

One might hope that existing traceability results (such as [DSSUV15]) and a clever reduction to mean estimation (such as [BST14, Section 5.1]) might yield optimal results for SCO. Here, we will demonstrate rigorously that merely combining results and techniques of [DSSUV15; BST14] yields suboptimal results for the setup of SCO, even in the simple setting of /lscript 2 geometry. [BST14] considers the following /lscript 2 problem:

<!-- formula-not-decoded -->

To apply fingerprinting to establish traceability, we first need to posit a prior distribution over the unknown distribution. [DSSUV15] does so by considering product distributions over Z , and placing a uniform prior over the mean µ ∈ [ -1 / √ d, 1 / √ d ] d . We now show that this (Bayesian) problem requires only O (1 /α ) samples to learn, and thus, tracing Ω(1 /α 2 ) samples is clearly impossible. Consider the ERM learner ˆ θ . It is easy to see that ˆ θ can be written as:

<!-- formula-not-decoded -->

where ˆ µ is the empirical mean of the dataset, that is, ˆ µ = 1 n ∑ n i =1 Z i . Similarly, the population risk minimizer θ /star is µ/ ‖ µ ‖ 2 . The expected excess risk of ˆ θ is then:

<!-- formula-not-decoded -->

where in (a) we used the AM-GM inequality, and the fact that the expression on the preceding line is always bounded by 2. The intuition behind the rest of the argument is that, due to the uniform prior on µ , we have ‖ µ ‖ , ‖ ˆ µ ‖ ∈ Ω(1) with high probability. At the same time

<!-- formula-not-decoded -->

thus, expected risk will be on the order of O (1 /n ). To formalize this, note that E ‖ µ ‖ 2 2 = 1 / 3, and ‖ µ ‖ 2 is a sum of d independent random variables bounded by 1 / √ d in absolute value. Hoeffding's inequality then yields that we have ‖ µ ‖ 2 2 ≥ 1 / 6 with very high probability. Similarly, we can obtain ‖ µ -ˆ µ ‖ 2 2 ≤ 1 / (2 n ) + 1 / 36 ≤ 1 / 12 for large enough n , with high probability. Then, with high probability, event

<!-- formula-not-decoded -->

holds Then, the excess risk is upper bounded by

<!-- formula-not-decoded -->

as desired.

## C Proofs from Section 3

## C.1 Proof of Lemma 3.4

We first prove a slightly more general concentration statement to bound the supremum in Lemma 3.4, which will be useful to reuse in other proofs. Let N (Θ , ‖·‖ , ε ) denote the size of the minimal cover of Θ in norm ‖·‖ at scale ε &gt; 0. Then, the more general statement is given below.

Lemma C.1. Fix n, d ∈ N . Suppose Θ ⊂ R d is a subset of a unit ball in some norm ‖·‖ . Let φ : Θ ×Z → R and D ∈ M 1 ( Z ) be such that, as Z ∼ D , { φ ( θ, Z ) } is a σ -subgaussian process w.r.t. (Θ , ‖·‖ ) and for every θ ∈ Θ , E [ φ ( θ, Z )] = 0 . Let ( Z 1 , . . . , Z n ) ∼ D ⊗ n . Then, there exist a universal constant C &gt; 0 , such that for every t ≥ 0 ,

<!-- formula-not-decoded -->

Proof. Let Φ θ denote the following random vector

<!-- formula-not-decoded -->

Then, observe that, the desired quantity is equal to

<!-- formula-not-decoded -->

Then, 〈 x, Φ θ 〉 can be seen to be a random process parameterized by a pair ( x, θ ). We will show that it is, in fact, a subgaussian process. Indeed, note that, by triangle inequality,

<!-- formula-not-decoded -->

Since Φ i θ is σ -subgaussian for each i , we have by Lemma A.7,

<!-- formula-not-decoded -->

for some universal constant C &gt; 0. Now, for every i , (Φ θ -Φ θ ′ ) i is σ ‖ θ -θ ′ ‖ -subgaussian. Therefore, by Lemma A.7, we have

<!-- formula-not-decoded -->

Combining the two inequalities, we get

<!-- formula-not-decoded -->

Thus, 〈 x, Φ θ 〉 is (2 Cσ )-subgaussian process w.r.t the norm γ , defined as

<!-- formula-not-decoded -->

Moreover, we can see that Θ × S n -1 is a subset of a unit ball in γ . By definition of γ , we have

<!-- formula-not-decoded -->

Then, using Proposition A.10, we have, for some constant K &gt; 0, that with probability 1 -4 exp( -t 2 )

<!-- formula-not-decoded -->

as desired, where in (a) we used Example 5.8 from [Wai19], and K ′ &gt; 0 is some other universal constant.

Using Example 5.8 from [Wai19] once again to upper bound √ log N (Θ; ‖·‖ , ε ), we have the proof of Lemma 3.4.

Lemma 3.4. Fix n, d ∈ N . Suppose Θ ⊂ R d is a subset of a unit ball in some norm ‖·‖ . Let φ : Θ ×Z → R and D ∈ M 1 ( Z ) be such that, as Z ∼ D , { φ ( θ, Z ) } is a σ -subgaussian process w.r.t. (Θ , ‖·‖ ) . Let ( Z 1 , . . . , Z n ) ∼ D ⊗ n . Then, there is a constant C &gt; 0 , such that

<!-- formula-not-decoded -->

Proof. From Example 5.8 in [Wai19], we have

<!-- formula-not-decoded -->

Plugging this into the result of Lemma C.1, with probability at least 1 -4 exp( -t 2 ), we have

<!-- formula-not-decoded -->

for some other universal constant C ′ &gt; 0 .

## C.2 Proof of Theorem 3.5

Theorem 3.5. Fix n ∈ N , d ∈ N , κ &gt; 0 and α ∈ [0 , 1] . Consider an arbitrary SCO problem P = (Θ , Z , f ) . Let T = Tr κ ( P ; n, α ) be the subgaussian trace value of P . Then, for some constant c &gt; 0 , every α -learner A n is ( ξ, m ) -traceable with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We set

First, we show that the soundness condition holds. Since Z and ˆ θ are independent, and using the subgaussian nature of φ ( ˆ θ, Z ), we have by Lemma A.6

<!-- formula-not-decoded -->

where c &gt; 0 is some constant. For recall, let's define the set I as follows

<!-- formula-not-decoded -->

Using Lemma A.11, we have

<!-- formula-not-decoded -->

where for every x ∈ R , we define ( x ) + = max { x, 0 } . Then, Lemma 3.4 tells us that, for t := √ n + d , we have with probability 1 -4 exp( -t 2 ),

<!-- formula-not-decoded -->

for some constant C &gt; 0. Thus,

<!-- formula-not-decoded -->

This implies,

<!-- formula-not-decoded -->

We know that, almost surely,

Thus, almost surely,

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, where ( a ) follows by Jensen's inequality and c = 1 / 4 C .

<!-- formula-not-decoded -->

## C.3 Proof of Theorem 3.6

Theorem 3.6. There exists a universal constant c &gt; 0 , such that the following holds. Fix p ∈ [1 , ∞ ) , n ∈ N , d ∈ N , α ∈ [0 , 1] , κ &gt; 0 ε &gt; 0 , and δ ∈ [0 , 1] . Consider an arbitrary SCO problem P = (Θ , Z , f ) in R d . Let T = Tr κ ( P ; n, α ) be the subgaussian trace value of problem P . Then, for every ( ε, δ ) -DP α -learner A n , we have exp( ε ) -1 ≥ c ( T -2 δκ ) .

Proof. Consider an arbitrary distribution D and a function φ s.t. { φ ( θ, Z ) } θ ∈ Θ is a 1subgaussian process w.r.t. (Θ , ‖·‖ Θ ) and | φ | ≤ κ almost surely. Consider a sample S n = ( Z 1 , . . . , Z n ) and let Z 0 be a freshly sampled point; let S ( i ) n be a sample with Z i substituted by Z 0 . Let ˆ θ be a learner trained on S n and ˆ θ ( i ) be a learner trained on S ( i ) n . Then, since ˆ θ is ( ε, δ )-DP and noting that φ ( θ, Z ) is supported on [ -κ, κ ], we may apply Lemma A.1 of [FS17] and get

<!-- formula-not-decoded -->

By independence of ˆ θ and Z i , we conclude that φ ( ˆ θ, Z i ) is 1-subgaussian random variable. It is well-known that E | X | ≤ Cσ if X is σ -subgaussian for some constant C (see part (ii) of Proposition 2.5.2 of [Ver18] for p = 1), thus the above gives

<!-- formula-not-decoded -->

Then, for every D and φ we get that,

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, after rearranging, implies the desired result.

## D Proofs of fingerprinting lemmas (Section 4)

## D.1 Proof of Lemma 4.2

Lemma 4.2 (Sparse fingerprinting) . Fix d, n ∈ N and let k ∈ [ d ] . For each µ ∈ [ -k/d, k/d ] d , let Z k and D µ,k be as in Definition 4.1 . Let π = s -beta [ -k/d,k/d ] ( β, β ) ⊗ d be a prior and set

<!-- formula-not-decoded -->

Then, for every learning algorithm A n : Z n →M 1 ( R d ) with sample S n = ( Z 1 , . . . , Z n ) ,

<!-- formula-not-decoded -->

/negationslash

Proof. For each j ∈ [ d ], let I j := { i ∈ [ n ] : Z j i = 0 } as the index of the training points such that their j -th coordinate is non-zero. Then, we have

<!-- formula-not-decoded -->

Then, define the following function

<!-- formula-not-decoded -->

/negationslash

The proof is based on the following two observations: 1) conditioned on { Z i m } m = j,i ∈ [ n ] , ˆ θ is a function of { Z j i } i ∈ [ n ] , 2) conditioned on {I r } r ∈ [ d ] the non-zero elements in { Z j i } i ∈ [ n ] are sampled i.i.d from {± 1 } with mean d k µ j . Then, based on these observations Equation (15) follows as an straightforward application of [Ste16, Lemma 4.3.7].

We claim

<!-- formula-not-decoded -->

/negationslash

Recall the definition of π and notice that π is a product measure. Let π j be the distribution on the j -th coordinate. By the definition of the prior distribution, we can write

<!-- formula-not-decoded -->

/negationslash

Therefore, we have

<!-- formula-not-decoded -->

/negationslash where the last step follow from Equations (15) and (16). Then, notice that

/negationslash

<!-- formula-not-decoded -->

Therefore, by the definition of inner product in R d , we have

<!-- formula-not-decoded -->

as was to be shown.

## D.2 Fingerprinting for /lscript 1 setup.

Additionally, to prove Theorem 2.6, we will need the following fingerprinting lemma. It can be seen as a generalization of beta-fingerprinting lemma in [SU17] using the scaling matrix technique of [KLSU18].

Lemma D.1 (Fingerprinting lemma with a scaling matrix) . Fix d ∈ N . Let Z = {± 1 } d and let β &gt; 0 be arbitrary. Consider arbitrary 0 &lt; γ ≤ 1 . For every µ ∈ [ -γ, γ ] d , let D µ be the product distribution on Z with mean µ , i.e., for every z ∈ Z , we have D µ = ∏ d k =1 ( 1+ z k µ k 2 ) let Λ µ be a diagonal matrix of size d where the i -th diagonal element is given by Λ ii µ = 1 -( µ i /γ ) 2 1 -( µ i ) 2 , and let φ µ ( θ, z ) = 〈 θ, Λ µ ( z -µ ) 〉 . Let π = s -beta [ -γ,γ ] ( β, β ) ⊗ d be a prior. Then, for every algorithm A n : Z n →M 1 ( R d ) , we have

<!-- formula-not-decoded -->

This fingerprinting lemma is handy for the following reason. To ensure the problem is hard to learn, entries of µ typically need to inversely scale with α . To achieve this, one can select small γ in the above to shrink the beta-prior to a smaller scale, while simultaneously having the freedom to set β to any value. In particular, this allows us to choose β ∈ Θ(log( d )) in the proof of Theorem 2.6 to leverage the anti-concentration result of [SU17, Prop. 5].

Before we proceed with the proof, we state the necessary lemmata. Throughout this section, for a real number p ∈ [ -1 , 1], we will write X ∼ p to denote the fact that X is a random variable on {± 1 } with mean p . The following is a classical fingerprinting result.

Lemma D.2 (Lemma 5 of [DSSUV15]) . Let f : {± 1 } n → R be arbitrary. Define g : [ -1 , 1] → R by

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Armed with the above result, we proceed to the proof of Lemma D.1. We will first prove a per-coordinate version of Lemma D.1. We make a note that the proofs combine techniques for beta-fingerprinting results of [SU17] and the scaling matrix technique of [KLSU19; ADHLR24].

Lemma D.3 (Per-coordinate version of Lemma D.1) . Let f : {± 1 } n → R be arbitrary. Let π = s -beta [ -γ,γ ] ( β, β ) be a prior distribution. Then,

<!-- formula-not-decoded -->

Proof. Let

<!-- formula-not-decoded -->

Then, by Lemma D.2, we have for every p ∈ [ -1 , 1],

<!-- formula-not-decoded -->

Recalling the definition of scaled symmetric beta distribution from Definition A.13, we have

<!-- formula-not-decoded -->

where in (a) we used integration by parts. This concludes the proof.

Applying the above results to each coordinate and summing the equalities gives Lemma D.1.

Lemma D.1 (Fingerprinting lemma with a scaling matrix) . Fix d ∈ N . Let Z = {± 1 } d and let β &gt; 0 be arbitrary. Consider arbitrary 0 &lt; γ ≤ 1 . For every µ ∈ [ -γ, γ ] d , let D µ be the product distribution on Z with mean µ , i.e., for every z ∈ Z , we have D µ = ∏ d k =1 ( 1+ z k µ k 2 ) let Λ µ be a diagonal matrix of size d where the i -th diagonal element

is given by Λ ii µ = 1 -( µ i /γ ) 2 1 -( µ i ) 2 , and let φ µ ( θ, z ) = 〈 θ, Λ µ ( z -µ ) 〉 . Let π = s -beta [ -γ,γ ] ( β, β ) ⊗ d be a prior. Then, for every algorithm A n : Z n →M 1 ( R d ) , we have

<!-- formula-not-decoded -->

Proof. For a sample S n = ( Z 1 , . . . , Z n ) and j ∈ [ j ], we will use S j n ∈ R n to denote a vector ( Z j 1 , . . . , Z j n ) of j th coordinates. For each coordinate j ∈ [ d ], let f j : {± 1 } n → R the function such that

<!-- formula-not-decoded -->

In other words, f j ( X ) is the expected value of θ j , given that j th coordinates of samples in S n are given by X . Applying the result of Lemma D.1 to f j , we have

<!-- formula-not-decoded -->

By the law of total expectation, we get

<!-- formula-not-decoded -->

Finally, summing the above over all coordinates j ∈ [ d ], we obtain

<!-- formula-not-decoded -->

## E Hard problem constructions and proofs of subgaussian trace value lower bounds

## E.1 Proofs for /lscript p -geometries (Theorem 5.1)

First, recall here the construction of the hard problems P k,p in Equation (7), parameterized by k ∈ [ d ]

<!-- formula-not-decoded -->

Proposition E.1. Let A n be an α -learner for P k,p . Let D ∈ M 1 ( Z ) be a distribution with mean µ = E Z ∼D [ Z ] . Then, we have

First, we show in the simple proposition below that α -learners for linear problems must agree with the distribution mean.

<!-- formula-not-decoded -->

Proof. Since A n is an α -learner, we have

<!-- formula-not-decoded -->

which, after rearranging, becomes

<!-- formula-not-decoded -->

where in the last transition we used duality of /lscript ∞ and /lscript 1 norms and the fact that Θ = B ∞ ( d -1 /p ). This concludes the proof.

In the next lemma, we show that every α -learner for P k,p needs to have a large correlation with the training samples in order to achieve small excess risk. The proof is an application of Lemma 4.2 combined with Proposition E.1.

Lemma E.2. Let α ≤ 1 / 6 , and suppose k ∈ [ d ] is such that k ≥ (6 α ) p d . Then, for every α -learner A n for P k,p , there exists µ ∈ [ -k/d, k/d ] d and distribution D ∈ M 1 ( Z k ) with mean µ such that the following holds: let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then,

Proof. Let and π = s -beta [ -k/d,k/d ] ( β, β ). Then, using Corollary A.15, we have

<!-- formula-not-decoded -->

Then, using Lemma 4.2 we have

<!-- formula-not-decoded -->

Since the above holds in expectation over draws of µ , there exists at least one value of µ for which the above holds; let D = D k,µ . Then, letting

<!-- formula-not-decoded -->

we obtain

<!-- formula-not-decoded -->

as desired.

We now argue that the pair ( φ, D ) from the lemma above (with φ scaled by some constant) constitutes a valid subgaussian tracer. In particular, the lemma below shows that { φ ( θ, Z ) } θ ∈ Θ induces a O (1)-subgaussian process w.r.t. (Θ , ‖·‖ Θ ) norm.

Lemma E.3. Fix d ∈ N . Let µ ∈ [ -k/d, k/d ] d be arbitrary. Let φ : Θ × Z k → R be as in Lemma E.2. Let D µ,k be the data distribution from Definition 4.1 for some µ , and consider Z ∼ D µ,k . Then, { φ ( θ, Z ) } θ ∈ Θ is a C -subg process w.r.t. to (Θ , ‖·‖ Θ ) for some universal constant C &gt; 0 .

Proof. Let J ∈ ( [ d ] k ) be an arbitrary coordinate subset of size k , and, recalling Definition 4.1, let Z J be a random variable with PMF given by P µ,k,J . Then, Z is a uniform mixture of { Z J } J ∈ ( [ d ] k ) .

Fix J ∈ ( [ d ] k ) , and let θ 1 , θ 2 ∈ Θ be two arbitrary points. First, we upper bound a subgaussian norm of φ ( θ 1 , Z J ) -φ ( θ 2 , Z J ). We have

<!-- formula-not-decoded -->

where C 1 , 2 &gt; 0 are universal constants, in (a) we apply Lemma A.7, in (b) we apply Proposition A.9, and in (c) we use that, since Θ = B ∞ ( d -1 /p ), we have ‖·‖ Θ = d 1 /p ‖·‖ ∞ . Thus, letting C = √ C 1 C 2 , we have

<!-- formula-not-decoded -->

Now, note that φ ( θ 1 , Z ) -φ ( θ 2 , Z ) has the same distribution as a uniform mixture of { φ ( θ 1 , Z J ) -φ ( θ 2 , Z J ) } J ∈ ( [ d ] k ) . Then, by Proposition A.8, we also have

<!-- formula-not-decoded -->

which satisfies the first condition in Definition 3.1. Finally, by plugging θ 2 = 0 into the above, we have which satisfies the second condition in Definition 3.1. Thus, { φ ( θ, Z ) } θ ∈ Θ is a C -subgaussian process w.r.t. (Θ , ‖·‖ Θ ), as desired.

<!-- formula-not-decoded -->

Finally, we lower bound the subgaussian trace value of P k,p .

Lemma E.4. Let α ≤ 1 / 6 and d ∈ N be arbitrary, and let k ∈ [ d ] be such that k ≤ (6 α ) p d . Let P k,p be as in Equation (7) . Then, the following subgaussian trace value lower bounds hold for every p ∈ [1 , ∞ ) and some κ ≤ c 1 √ d ,

<!-- formula-not-decoded -->

where c 1 , 2 &gt; 0 are universal constants.

Proof. Let C &gt; 0 be the constant from Lemma E.3, and φ be as in Lemma E.2. Then, { φ ( θ, Z ) /C } θ ∈ Θ is a 1-subgaussian process w.r.t. (Θ , ‖·‖ Θ ). Moreover, φ ( θ, Z ) /C ≤ √ k/C ≤ √ d/C . Then, letting κ = √ d/C and using Lemma E.2, the subgaussian trace value of P k,p can be lower bounded by

<!-- formula-not-decoded -->

as desired.

Now we are ready to prove Theorem 5.1.

Theorem 5.1. Let P k,p be the family of problems described in Equation (7) . There exist universal constants c 1 , c 2 &gt; 0 such that, for all α ∈ (0 , 1 / 6] and d ∈ N , the following subgaussian trace value lower bounds hold for all p ∈ [1 , ∞ ) and κ ≤ c 1 √ d :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The theorem is a direct consequence of Lemma E.4. For p ≤ 2, plug in k = d into the statement of Lemma E.4. We obtain

<!-- formula-not-decoded -->

For p ≥ 2, plug in k = (6 α ) p ∨ 1. We obtain,

<!-- formula-not-decoded -->

as desired.

## E.2 Proofs for /lscript 1 -geometry

## E.2.1 Intuition

Refinement for p = 1 . While the above construction also yields a traceability result for p = 1, it is suboptimal for the following simple reason: for k = d , the problem in Equation (7) only requires Θ(1 /α 2 ) samples to learn, thus, it is impossible to trace out Ω(log( d ) /α 2 ) samples. On the other hand, the problem in Equation (6) requires Θ(log( d ) /α 2 ) samples to learn but is not traceable. The intuition we follow here is to modify the construction in Equation (7) to make Θ 'look' more like an /lscript 1 -ball to drive up the sample complexity while still avoiding the counterexample with an ERM learner from the beginning of the section. In particular, we consider the following /lscript 1 -problem,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for a suitably chosen s ∈ [ d ]. Note that, if we choose s /greatermuch 1, Θ above is a polytope with much more vertices (2 s ( d s ) ) than an /lscript 1 ball (2 d ), which would intuitively force a learner like an ERM to reveal more information about the training sample. On a technical level, selecting large s improves the subgaussian constant of a tracer; however, selecting s that is too large shrinks the diameter of the set, and thus, the problem becomes easier to learn. We must trade off these two aspects, and carefully set the value of s . As it turns out, the optimal choice is s ∝ d 1 -c for an arbitrary small c &gt; 0 in order to establish Theorem 2.6. The remainder of the proof is rather technical and hence is deferred to Appendix F.2.

## E.2.2 Formal Proof

For technical reasons, we will need the following refinement of Lemma 3.4 for the special case when the function φ is convex and Θ is a polytope. In the proof, we use Lemma C.1.

Lemma E.5. Fix n, d ∈ N . Suppose Θ ⊂ R d is (i) a subset of a unit ball in some norm ‖·‖ , and (ii) Θ is a polytope with N vertices. Let φ : Θ ×Z → R be a measurable function that is convex in its first argument. Let D ∈ M 1 ( Z ) be such that φ ( θ, Z ) is a σ -subgaussian process w.r.t. (Θ , ‖·‖ ) .Let ( Z 1 , . . . , Z n ) ∼ D ⊗ n . Then, for every t ≥ 0 ,

<!-- formula-not-decoded -->

where C &gt; 0 is some universal constant.

Proof. Similarly to the proof of Lemma C.1, let Φ θ denote the following random vector

<!-- formula-not-decoded -->

Then, observe that, the desired quantity is equal to

<!-- formula-not-decoded -->

Let V be the set of vertices of Θ with | V | ≤ N . Since φ is convex in its first argument, the supremum above is attained in one of the vertices of Θ. Thus,

<!-- formula-not-decoded -->

Thus, we may apply Lemma C.1 to V instead of Θ. Trivially, we have

<!-- formula-not-decoded -->

Then, we have with probability 1 -2 exp( -t 2 ),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as desired.

Intuitively, in the special case when Θ is a polytope, the log-number of vertices becomes 'effective dimension' instead of d , due to the fact that φ satisfies the convexity requirement. In some cases, we can have d /greatermuch log( N ), in which the above gives a tighter concentration. In particular, this is a case in our construction for /lscript 1 geometry in Equation (17). With the above result, we can also establish the following refinement of Theorem 3.5.

Theorem E.6. Fix n ∈ N , d ∈ N , κ &gt; 0 and α ∈ [0 , 1] . Consider an arbitrary SCO problem P = (Θ , Z , f ) , and suppose Θ is a polytope with N &gt; 0 vertices. Let T be defined as,

<!-- formula-not-decoded -->

Then, for some constant c &gt; 0 , every α -learner A n is ( ξ, m ) -traceable with

<!-- formula-not-decoded -->

Proof. The proof is identical to Theorem 3.5, but using Lemma E.5 instead of Lemma 3.4, and thus, replacing d with log( N ) everywhere. We omit the details.

Now, recall the construction from Equation (17),

<!-- formula-not-decoded -->

It is easy to see that f ( · , Z ) above is 1-Lipschitz w.r.t. /lscript 1 as Z ⊂ B ∞ (1). We have the following claim.

Lemma E.7. Let P s be as in (18) , n ∈ N and 1 / 8 &gt; α &gt; 0 . Then,

<!-- formula-not-decoded -->

where κ ≤ c ′ √ s for some constants c, c ′ &gt; 0

Proof. We aim to use Lemma D.1 to characterize the subgaussian trace value. Consider the construction of the prior in Lemma D.1 with the following parameters: γ = 8 α ≤ 1 and β = 1 + 1 2 log ( d 16( s ∨ 14) ) . Then, by combining Lemma D.1 with and Proposition E.1, there exist a prior π and a family { Λ µ } of diagonal matrices with non-negative diagonal entries bounded by 1 from above, such that

<!-- formula-not-decoded -->

where in (a) we used Proposition 5 of [SU17]. Since this holds in expectation over µ , it holds for at least one choice of µ . Let µ be that value. Now, let φ ( θ, Z ) = C -1 / 2 √ s 〈 ˆ θ, Λ( Z -µ ) 〉 , where C is the absolute constant from Lemma A.7. For all θ, θ ′ ∈ Θ, we have

<!-- formula-not-decoded -->

where in (a) we apply Lemma A.7, and in (b) we use that for every θ ∈ Θ, we have ‖ θ ‖ 2 ≤ 1 / √ s , thus, √ s ‖·‖ 2 ≤ ‖·‖ Θ . Plugging θ ′ = 0 gives

<!-- formula-not-decoded -->

Thus, { φ ( θ, Z ) } θ ∈ Θ is a 1-subg process w.r.t. (Θ , ‖·‖ Θ ). Finally, we have

Therefore, setting κ = C -1 / 2 √ s and noting that φ is linear (and therefore convex) in its first argument, we have as desired.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F Proofs of the main results (Section 2.2)

## F.1 Proof of Theorem 2.5

Theorem 2.5. There exists a universal constant c &gt; 0 such that, for all p ∈ [1 , 2) , if d , n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 are such that

<!-- formula-not-decoded -->

then there exist an /lscript p SCO problem that every α -learner is ( ξ, m ) -traceable with m ∈ Ω ( α -2 ) .

Proof. We begin by noting that the interval for α in Equation (3) is non-empty only if

<!-- formula-not-decoded -->

where we used ξ &lt; 1 /e in the last transition. Via straightforward algebra, the above implies d ≥ n , thus, we may without loss of generality assume d ≥ n in the remainder of the proof.

Let P k,p be as in Equation (7), and set k as in Theorem 5.1. Then, Theorem 5.1 gives the following lower bound on the subgaussian trace value in this case:

<!-- formula-not-decoded -->

for some κ ≤ c 1 √ d . Thus, provided c is small enough ( c ≤ c 2 ), by Theorem 3.5, every α -learner is ( ξ, m )-traceable with m satisfying, for some universal constant c ′ &gt; 0,

<!-- formula-not-decoded -->

where we used d ≥ n in the last inequality. Then, we have

<!-- formula-not-decoded -->

as desired.

## F.2 Proof of Theorem 2.6

Then, the proof of Theorem 2.6 follows.

Theorem 2.6. There exists a universal constant c &gt; 0 such that, if d is large enough and n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 are such that

<!-- formula-not-decoded -->

there exists a /lscript 1 SCO problem that every α -learner is ( ξ, m ) -traceable with m ∈ Ω ( log( d ) /α 2 ) .

Proof. We begin by noting that the interval for α in Equation (4) is non-empty only if

<!-- formula-not-decoded -->

where we used ξ &lt; 1 /e and c &lt; 1 / 6 in the last transition. Via straightforward algebra, the above implies d 0 . 98 / log( d ) ≥ n , thus, we may without loss of generality assume d 0 . 98 / log( d ) ≥ n in the remainder of the proof.

Let s = d 0 . 98 . Note that, letting V be the set of vertices of the polytope Θ = B 1 (1) ∩B ∞ (1 /s ), we have

<!-- formula-not-decoded -->

The proof of this fact is straightforward and it is based on showing that every point in Θ can be written as a convex combination of the points in V . Thus,

<!-- formula-not-decoded -->

By the choice of s and using Lemma E.7, we have, for some κ ≤ c 1 √ s ,

<!-- formula-not-decoded -->

where (a) and (b) hold provided d (and thus s ) is large enough, and (c) holds provided c &gt; 0 in Equation (4) is small enough. By Theorem E.6, every α -learner is ( ξ, m )-traceable, where

<!-- formula-not-decoded -->

Recall that d 0 . 98 ≥ n log( d ) ≥ n (for d ≥ 3). Moreover, log | V | ≤ s log( de 2 /s ) = d 0 . 98 log( e 2 d 0 . 02 ) ≤ Cd 0 . 98 log( d ), for some universal C &gt; 0. Thus, for d large enough,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as desired.

## F.3 Proof of Theorem 2.7

Theorem 2.7. There exists a universal constant c &gt; 0 such that, for all p ∈ [2 , ∞ ) , if d , n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 are such that

<!-- formula-not-decoded -->

there exist an /lscript p SCO problem such that every α -learner is ( ξ, m ) -traceable with m ∈ Ω(1 / (6 α ) p ) .

Proof. Throughout, we assume c is a sufficiently small constant. Assume c &lt; 1 / 6. Then, we begin by noting that the interval for α in Equation (5) is non-empty only if

<!-- formula-not-decoded -->

where we used ξ &lt; 1 /e and c &lt; 1 / 6 in the last transition. Via straightforward algebra, the above implies d ≥ n , thus, we may without loss of generality assume d ≥ n in the remainder of the proof.

Let P k,p be as in Equation (7), and set k as in Theorem 5.1. Then, Theorem 5.1 gives the following lower bound on the subgaussian trace value

<!-- formula-not-decoded -->

for some κ ≤ c 1 √ d . Note that, from (5), we have

<!-- formula-not-decoded -->

Now, note that, the minimum in Equation (20) is achieved in the second term iff α ≥ d -1 /p / 6. Then, the lower bound on the subgaussian trace value becomes

<!-- formula-not-decoded -->

where the second transition follows from Equation (5) and the third transition holds whenever c &gt; 0 is small enough (e.g., when c ≤ c 2 /p 2 / 6). Then, by Theorem 3.5 every α -learner is ( ξ, m )-traceable with m satisfying, for some universal constant c ′ &gt; 0,

<!-- formula-not-decoded -->

where we used d ≥ n in the last transition. Thus,

<!-- formula-not-decoded -->

as desired.

## F.4 Proof of Theorem 2.8

Theorem 2.8. Let p ∈ [2 , ∞ ) . There exist a universal constant c &gt; 0 and an /lscript p SCO problem P = (Θ , Z , f ) such that every ( ε, δ ) -DP learner of P with ε ≤ 1 and δ ≤ c/n satisfies,

<!-- formula-not-decoded -->

Proof. By Theorem 5.1, for some problem P , we have

<!-- formula-not-decoded -->

Then Theorem 3.6 implies

Note that for all ε ≤ 1, we have 2 ε ≥ exp( ε ) -1. Thus, which implies,

Then, for some C &gt; 0,

Rearranging gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that if ε ≥ δκ , the desired bound is immediate. For δκ ≥ ε , we have, since δ ≤ c/n and κ ≤ c ′ √ d ,

<!-- formula-not-decoded -->

for some C ′ &gt; 0, as desired.

## F.5 Proof of Corollary 2.9

Corollary 2.9. Let Z = {± 1 } d , and suppose an estimator is given such that, given access to i.i.d. samples Z 1 , . . . , Z n ∈ Z , outputs ˆ µ with E ‖ ˆ µ -E [ Z 1 ] ‖ ∞ ≤ α/ 2 . Then, there exists a universal constant c &gt; 0 such that, if d is large enough and n , ξ ∈ (0 , 1 /e ) , and α &gt; 0 satisfy Equation (4) , then the estimator ˆ µ is ( ξ, m ) -traceable with m ∈ Ω ( log( d ) /α 2 ) .

Proof. We will first show that we can use the mean estimation algorithm to solve the corresponding hard problem for /lscript 1 -SCO. Specifically, consider the SCO problem as in Equation (17), and define the following learning algorithm based on the mean estimation. Let ˆ µ be the output of mean estimator based on the samples Z 1 , . . . , Z n , and let

Let θ /star be the population risk minimizer, and let µ be the true mean, that is, µ = E [ Z 1 ]. Then, the excess risk of ˆ θ can be upper bounded as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last transition follows since ˆ θ, θ /star both lie inside Θ ⊂ B 1 . Now, by the choice of ˆ θ , the second term is non-positive. Thus,

<!-- formula-not-decoded -->

Taking expectations on both sides, we get

<!-- formula-not-decoded -->

Applying the result of Theorem 2.6 to ˆ θ , we conclude the proof.

## G Connection between subgaussian trace value and non-private sample complexity

From the main part of the paper, we observe that the subgaussian trace value is typically inversely proportional to α . We start by proving two innocuous results (Propositions G.1 and G.3) that establish an absolute upper bound on subgaussian trace value. It will then allow us to extract lower bounds on α by plugging in our lower bounds on subgaussian trace value (Theorems G.2 and G.4). We start with the p ∈ (1 , ∞ ) case, and then consider p = 1.

## G.1 Lower bounds for p ∈ (1 , ∞ )

Proposition G.1. Fix n ∈ N , d ∈ N , and α ∈ [0 , 1] . Consider an arbitrary SCO problem P = (Θ , Z , f ) in R d . Let Tr κ ( P ; n, α ) be the subgaussian trace value of problem P . Then, we have √

<!-- formula-not-decoded -->

for some universal constant C &gt; 0 .

Proof. Let ( φ, D ) be an arbitrary subgaussian tracer. Consider the process { X θ } θ ∈ Θ defined as where S n = ( Z 1 , . . . , Z n ) ∼ D ⊗ n . We will argue that { X θ } θ ∈ Θ is O (1 / √ n )-subgaussian process w.r.t. (Θ , ‖·‖ Θ ). First, consider arbitrary θ 1 , θ 2 ∈ Θ. We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in (a) we applied Lemma A.7, and in (b) we used the fact that { φ ( θ, Z i ) } θ ∈ Θ is 1-subgaussian process w.r.t. (Θ , ‖·‖ Θ ) for every i ∈ [ n ]. Moreover, for every θ ∈ Θ, we similarly have

<!-- formula-not-decoded -->

where in (a) we applied Lemma A.7, and in (b) we used the fact that { φ ( θ, Z i ) } θ ∈ Θ is 1-subgaussian process w.r.t. (Θ , ‖·‖ Θ ) for every i ∈ [ n ]. Thus, { X θ } θ ∈ Θ is C/ √ n -subgaussian

process w.r.t. (Θ , ‖·‖ ) as per Definition 3.1. Therefore, by Proposition A.10, we have, with probability at least 1 -4 exp( -t 2 )

<!-- formula-not-decoded -->

where in second inequality we use [Wai19, Example 5.8]. Hence,

<!-- formula-not-decoded -->

where C ′ &gt; 0 is some universal constant, (a) follows by a change of variables u = Ct/ √ n , and (b) follows from Equation (22). By Definition 2.3, we can write

<!-- formula-not-decoded -->

By rearranging the terms we obtain the desired result.

We now show that Theorem 5.1 implies lower bounds on the sample complexity of learning /lscript p -Lipshitz-bounded problems for every p ∈ [1 , ∞ ). In particular, we show that the problems considered in Theorem 5.1 require many samples to learn (equivalently, we show a lower bound on optimal excess risk α ).

Theorem G.2. Let α &gt; 0 , p ∈ [1 , ∞ ) and n ∈ N be arbitrary. Let P k,p be as in Equation (7) , and set k as in Theorem 5.1. Suppose there exist an α -learner for P k,p . Then,

<!-- formula-not-decoded -->

(ii) for p ∈ (2 , ∞ ) , we have for some universal constant c &gt; 0 .

Proof. First, consider an arbitrary k ∈ [ d ]. We apply Proposition G.1 to the result of Lemma E.4. We then have the following double inequality

<!-- formula-not-decoded -->

Solving for α in the above, we have

<!-- formula-not-decoded -->

First, consider the case p ∈ [1 , 2]. Then k = d , and we have

<!-- formula-not-decoded -->

as desired. Now, consider the case p ∈ (2 , ∞ ). Then k = (6 α ) p d ∨ 1, and we have

<!-- formula-not-decoded -->

Solving for α , we have

Thus, as desired.

## G.2 Lower bounds for p = 1 .

Now, consider the case p = 1. For p = 1, we consider the problem as in Equation (17). We will need the following refinement of Proposition G.1 in a special case when Θ is a polytope with few vertices. Intuitively, d in the statement Proposition G.1 can be replaced by log N where N is the number of vertices of Θ.

Proposition G.3. Fix n ∈ N , d ∈ N , and α ∈ [0 , 1] . Consider an arbitrary SCO problem P = (Θ , Z , f ) in R d , where Θ is a polytope with N vertices. Let Tr κ ( P ; n, α ) be the subgaussian trace value of problem P . Then, we have

<!-- formula-not-decoded -->

for some universal constant C &gt; 0 .

Proof. Let ( φ, D ) be an arbitrary subgaussian tracer. Similarly to the proof of Proposition G.1, consider the process { X θ } θ ∈ Θ defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that, since 2 /p ≤ 1, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S n = ( Z 1 , . . . , Z n ) ∼ D ⊗ n . As in the proof of Proposition G.1, { X θ } θ ∈ Θ is a C/ √ n -subgaussian process w.r.t. (Θ , ‖·‖ Θ ) for some universal constant C &gt; 0.

Let V be the set of vertices of Θ; then | V | = N , as per the proposition statement. Since φ is convex in its first argument, the mapping θ ↦→ X θ is also convex (almost surely). Then,

<!-- formula-not-decoded -->

Therefore, by Proposition A.10, we have, with probability at least 1 -4 exp( -t 2 )

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where C ′ &gt; 0 is some universal constant, (a) follows by a change of variables u = Ct/ √ n , and (b) follows from Equation (22). By Definition 2.3, we can write

<!-- formula-not-decoded -->

By rearranging the terms we obtain the desired result.

Then, sample complexity lower bounds for /lscript 1 geometry follow.

Theorem G.4. Let α &gt; 0 and n ∈ N be arbitrary. Let P s be as in Equation (17) and set s = d 0 . 99 . Suppose there exists an α -learner for P s . Then, for large enough d , we have

<!-- formula-not-decoded -->

for some universal constant c &gt; 0 .

Proof. Lemma E.7 gives the following lower bound on the subgaussian trace value of P s ,

<!-- formula-not-decoded -->

At the same time, noting that Θ is a polytope with vertices given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some universal C &gt; 0. Combining this with the lower bound on subgaussian trace value, we have which gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that s = d 0 . 99 . For large enough d , we have s ∨ 14 = s . Also, note that log( d/s ) ≥ log( d ) / 100. Then, for for large enough d , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which has cardinality we have by Proposition G.3

for some universal c ′ &gt; 0, as desired.

## H Traceability of VC classes (Section 1.1.1)

First, we state the main result.

Theorem H.1. Fix n ∈ N and ξ &lt; 0 . 1 . Let H be an arbitrary VC concept class with VC dimension d vc . Then, there exists an optimal algorithm in terms of number of samples such that it is ( ξ/ ( n log( n ) , m ) -traceable with m ≤ O ( d vc log 2 ( n ) ) . Moreover, when H is the class of thresholds, we have m ∈ O (1) .

To prove it we use an information-theoretic notion that controls the difficulty of tracing from [SZ20].

Definition H.2. Fix n ∈ N . Let D be a data distribution, and A n be a learning algorithm. For every n ∈ N , let Z = ( Z j,i ) j ∈{ 0 , 1 } ,i ∈ [ n ] be an array of i.i.d. samples drawn from D , and U = ( U 1 , . . . , U n ) ∼ Ber ( 1 2 ) ⊗ n , where U and Z are independent. Define training set S n = ( Z U i ,i ) i ∈ [ n ] . Then, define

<!-- formula-not-decoded -->

In the next theorem, we show that the existence of a tracer for a learning algorithm provides a lower bound on the CMI of the algorithm. A similar observation is made in [ADHLR24].

Lemma H.3. Fix n ∈ N such that n ≥ 2 and ξ &lt; 1 / 2 . Let A n be an arbitrary learning algorithm that is ( ξ/ ( n log( n ) , m ) -traceable. Then, it holds sup D CMI D ( A n ) ≥ m -3 ξ.

The following two results from [SZ20] and [HDMR21] provide upper bounds on the CMI of sample compression schemes. We skip the formal definitions of sample compression schemes and refer the reader to [LW86; MY16; BHMZ20].

Lemma H.4 (Thm 4.2. [SZ20]) . Let H be an arbitrary concept class with VC dimension d vc . Then, there exists an algorithm such that for every data distribution D , CMI D ( A n ) ≤ O ( d vc log 2 ( n )) .

Lemma H.5 (Thm 3.4. [HDMR21]) . Let H be the concept class of threshold in R . Then, there exists an algorithm such that for every data distribution D and n ≥ 2 , CMI D ( A n ) ≤ 2 log 2 .

Proof of Theorem H.1. The proof is simply by combining Lemma H.4 with Lemma H.3. For the case of the class of thresholds, we use Lemma H.5.

Proof of Lemma H.3. Assume there exists a tracer with recall of m and soundness parameter of ξ/ ( n log( n ). Let D denote the distribution used by the tracer (see Definition 2.3). Define the following random set

<!-- formula-not-decoded -->

Also, for every i ∈ [ n ], define the following random variable

By the definition of mutual information and U ⊥ ⊥ Z , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, to lower bound CMI D ( A n ), we need to upper bound H( U | Z , ˆ θ ). By the subadditivity of the entropy, we have

<!-- formula-not-decoded -->

Then, by monotonicity of the entropy and the chain rule, we have

<!-- formula-not-decoded -->

In the next step, by the definition of conditional entropy,

<!-- formula-not-decoded -->

Define the random variable Y i = ✶ ( i ∈ V 1 ) . Notice that Y i is ( ˆ θ, Z )-measurable random variable. Then, using the notations for the disintegrated conditional entropy from [HDMR21], we have

The main observation is that under the events Y i = 1 and G i = 1, U i is deterministically known from ( Z , ˆ θ ). It follows because

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, since H Z , ˆ θ, G i =1 ( U i ) ≤ 1 with probability one, by Equation (26)

<!-- formula-not-decoded -->

Thus, from Equations (25) and (26), we obtain the following upper bound

<!-- formula-not-decoded -->

where H b ( · ) : [0 , 1] → [0 , 1] is the binary entropy function defined as H b ( x ) = -x log( x ) -(1 -x ) log(1 -x ). We can lower bound CMI D ( A n ) as follows

<!-- formula-not-decoded -->

where the last step follows because n -∑ n i =1 Pr( Y i = 0) = ∑ n i =1 Pr( Y i = 1) as Y i is an indicator random variable. In the next step, by the definition of soundness from Definition 2.3 and the definition of CMI in Definition H.2, we have

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

By the recall condition from Definition 2.3 and the definition of CMI in Definition H.2, we have

<!-- formula-not-decoded -->

We also use the following well-known inequality, H b ( x ) ≤ -x log( x ) + x for x ∈ [0 , 1]. As a result, we obtain

<!-- formula-not-decoded -->

where the last step follows because -x log( x ) ≤ 1 /e for x ∈ [0 , 1].

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims in the abstract and the introduction capture the precise theoretical claims made in the main part.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: It has been discussed when comparing with prior work.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Every theoretical result is presented in a form of a theorem stating all assumptions made.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Theorems and Lemmas that the proof relies upon should be properly referenced.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer:[NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- Please see the NeurIPS code and data submission guidelines ( https://nips. cc/public/guides/CodeSubmissionPolicy ) for more details.
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not include any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The full details can be provided either with the code, in appendix, or as supplemental material.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- The assumptions made should be given (e.g., Normally distributed errors).
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper does not include any experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors have reviewed the NeurIPS Code of Ethics and found that the research in this paper conforms to the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: It is a theory paper, and we forsee no such impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This is a theory paper, and it poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

Guidelines:

- The answer NA means that the paper does not use existing assets.

- The authors should cite the original paper that produced the code package or dataset.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/ datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer:[NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs. Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
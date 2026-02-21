## Efficient Fairness-Performance Pareto Front Computation

Mark Kozdoba *1 , Binyamin Perets *1 , and Shie Mannor 1,2

1 Technion - Israel Institute of Technology, Haifa, Israel 2 NVIDIA

## Abstract

There is a well known intrinsic trade-off between the fairness of a representation and the performance of classifiers derived from the representation. In this paper we propose a new method to compute the optimal Pareto front of this trade off. In contrast to the existing methods, this approach does not require the training of complex fair representation models.

Our approach is derived through three main steps: We analyze fair representations theoretically, and derive several structural properties of optimal representations. We then show that these properties enable a reduction of the computation of the Pareto Front to a compact discrete problem. Finally, we show that these compact approximating problems can be efficiently solved via off-the shelf concave-convex programming methods.

In addition to representations, we show that the new methods may also be used to directly compute the Pareto front of fair classification problems. Moreover, the proposed methods may be used with any concave performance measure. This is in contrast to the existing reduction approaches, developed recently in fair classification, which rely explicitly on the structure of the non-differentiable accuracy measure, and are thus unlikely to be extendable.

The approach was evaluated on several real world benchmark datasets and compares favorably to a number of recent state of the art fair representation and classification methods.

## 1 Introduction

Fair representations are a central topic in the field of Fair Machine Learning, Mehrabi et al. (2021), Pessach and Shmueli (2022),Chouldechova and Roth (2018). Since their introduction in Zemel et al. (2013), Fair representations have been extensively studied, giving rise to a variety of approaches based on a wide range of modern machine learning methods, such GANs, variational auto encoders, numerous variants of Optimal Transport methods, and direct variational formulations. See the papers Feldman et al. (2015), Madras et al. (2018), Gordaliza et al. (2019); Zehlike et al. (2020), Song et al. (2019), Du et al. (2020), Zhao and Gordon (2022), Jovanovi´ c et al. (2023), Dehdashtian et al. (2024), for a sample of existing methods.

For a given representation learning problem and a target classification problem, since the fairness constraints reduce the space of feasible classifiers, the best possible classification performance will usually be lower as the fairness constraint becomes stronger. This phenomenon is known as the Fairness-Performance trade-off. Assume that we have fixed a way to measure fairness. Then for a given representation learning method, one is often interested in the fairness-performance curve ( γ, E ( γ )) . Here, γ is the fairness level, and E ( γ ) is the classification performance of the method at

Figure 1: Fair Representation Problem Setting.

<!-- image -->

that level. The curve ( γ, E ( γ )) where E ( γ ) is the best possible performance over all representations and classifiers under the constraint is known as the Fairness-Performance Pareto Front.

As indicated by the above discussion, representation learning methods typically involve models with high dimensional parameter spaces, and complex, possibly constrained non-convex optimisation algorithms. As such, these methods may be prone to local minima and sensitivity to a variety of hyper-parameters, such as architecture details, learning rates, and even initializations. While the representations produced by such methods may often be useful, it nevertheless may be difficult to decide whether their associated Fairness-Performance curve is close to the true Pareto Front.

In this paper we propose a new method to compute the optimal Pareto front, which does not require the training of complex fair representation models. In other words, we show that, perhaps somewhat surprisingly, the computation of the Pareto Front can be decoupled from that of the representation, and only relies on learning of much simpler, unconstrained classifiers on the data. To achieve this, we first show that the optimal fair representations satisfy a number of structural properties. While these properties may be of independent interest, here we use them to express the points on the Pareto Front as the solutions of small discrete optimisation problems. These problems, known as concave minimisation problems Benson (1995), have been extensively studied and can be efficiently solved using modern dedicated optimisation frameworks, Shen et al. (2016).

We now describe the results in more detail. Let X ∈ X and A ∈ A denote the data features and the sensitive attribute, respectively. We assume that A is binary, while X may take values in an arbitrary space X , typically with X = R d . In addition, we have a target variable Y , taking values in a finite set Y . We are then interested in representations that maximise the performance of prediction of Y , under the fairness constraint. The representation is denoted by Z , and is typically expressed by constructing the conditional distributions P ( Z | X = x, A = a ) , for a ∈ { 0 , 1 } . The problem setting is illustrated in Figure 1.

Our approach consists of three main steps: We first observe that one can map the data features ( x, a ) ∈ X × A to a much smaller space ∆ Y of distributions on the set of label values Y , without loosing any information necessary for the computation of the Pareto front. The mapping is done by means of the optimal Bayes classifier. This result is referred to as Factorization Lemma, Section 3.2, where the mapping is done via the optimal Bayes classifier. Similar arguments were recently implicitly used in the study of fair classification tradeoffs, (Xian et al., 2023; Wang et al., 2023), but were restricted to classification and to accuracy loss (Section 2).

The advantage of working with a small space such as ∆ Y is that it can be easily discretized. For instance, if Y = 0 , 1 , then we essentially have ∆ Y = [0 , 1] , which is descretised trivially. We note that other, data dependent discretisation schemes, such as clustering, maybe possible for problems involving highly multi label targets. Alternatively, one can also consider the dataset itself as a grid, a view that is typically taken by transportation based approaches, e.x. Gordaliza et al. (2019), Xian et al. (2023).

Next, assuming the data is discretized and finite, we ask how large the represntation space Z should be, in order to support both optimal performance and fairness? For instance, we believe the answer to the following question is not apriori obvious: Can representations on infinite spaces be approximated, in terms of performance and fairness, by representations on finite and bounded spaces Z ? These questions are addressed by the Invertibility Theorem, Section 3.3, which asserts that all optimal representations may be taken in a certain canonical form, which we term invertible . This

result, in conjunction with an additional approximation lemma, is used in Section 4.1 to construct representations with any desired degree of approximation.

Finally, based on these results in Section 4.1 we also introduce the MIFPO (Model Independent Fairness-Performance Optimization), a discrete optimisation problem that is essentially equivalent to a computation of the fairness-performance tradeoff on a discrete set. We show that in this situation MIFPO is a concave minimisation problem with linear constraints, and we solve it using the disciplined convex-concave programming framework, DCCP, Shen et al. (2016).

We evaluate our approach on standard fairness benchmark datasets and compare its fairnessperformance curve to multiple state-of-the-art fair representation methods. We also compare MIFPO to the fairness-performance Pareto front of fair classifiers 1 . As expected, MIFPO effectively serves as an upper bound on almost all other algorithms in both cases.

To summarise, the contributions of this paper are as follows: (a) We derive several new structural properties of optimal fair representations. (b) We use these properties to construct a model independent problem, MIFPO, which can approximate the Pareto Front of arbitrary high dimensional data distributions, but is much simpler to solve than direct representation learning for such distributions. (c) We illustrate the approach on real world fairness benchmarks.

The rest of this paper is organised as follows: Section 2 discusses the literature and related work. In Section 3 we discuss the theoretical results, including factorization and the Invertibility Theorem. The MIFPO problem construction and the full Pareto Front computation algorithm are provided in Section 4. Experimental results are presented in Section 5, and we conclude the paper in Section 6. All proofs are provided in the Supplementary Material.

## 2 Literature and Prior Work

We refer to the book Barocas et al. (2023), and surveys Mehrabi et al. (2021),Du et al. (2020), for a general overview of representations. Tradeoffs in particular where explicitly studied in Song et al. (2019), Balunovi´ c et al. (2022a), Zhao and Gordon (2022), Jovanovi´ c et al. (2023),Dehdashtian et al. (2024), among others.

In this paper we use the total variation based fairness constraints, similarly to the line of work in Madras et al. (2018), Zhao and Gordon (2022), Balunovi´ c et al. (2022a), Jovanovi´ c et al. (2023). Other constraints used in the literature include entropy based constraints, Song et al. (2019), or RKHS based independence test constraints, Dehdashtian et al. (2024).

As discussed earlier, the vast majority of the work above concentrates on finding neural network based fair representations via involved optimization schemes with possible local minima, which may be hard to analyze. This highlights the usefulness of our approach direct approach to the computation of the Pareto front, which has clear theoretical grounding, and in which sources of approximation error are well understood and may be controlled.

Relations between fairness and performance were studied in Zhao and Gordon (2022). In particular, for perfectly fair representations, they derived lower bounds on the accuracy in terms of the difference of the base rates between the groups. However, this work did not introduce new algorithms for the computation of fair representations or of the associated Pareto front. The extension of the considerations in this paper to the full front was carried in Xian et al. (2023) for classification (see below).

The accuracy fairness tradeoff has also been extensively studied in the context of fair classification (without representations), see for instance Agarwal et al. (2018), Kim et al. (2020) , Alghamdi et al. (2022), Xian et al. (2023), Wang et al. (2023), for a sample of recent approaches. In particular, the papers Xian et al. (2023), Wang et al. (2023) are state of the art, and are also the most closely related to our methods, among the existing work.

Similarly to our approach, the analysis in these two papers starts with the estimation of the probabilities P ( Y | X,A ) , which are then used to compute the constrained performance. However, the subsequent steps are different. Crucially, the analysis in both Xian et al. (2023) and Wang et al. (2023) relies critically on the properties of the accuracy as the performance metric. Consequently,

1 See Sections 2 and 3.2 for the relation between our representation framework and fair classification results.

it can not be extended to general concave performance measures, such as the standard (minus) log loss, for instance. Roughly speaking, in the appropriate sense, accuracy largely ignores classification probabilities. This allows the simple description of classifiers as small confusion matrices in Wang et al. (2023) (extending the approach of Kim et al. (2020)), and the restriction of the distributions to the vertices of the simplex in Xian et al. (2023). The special structure of accuracy is highlighted also in our Lemma 3.1, where we show that classifiers with accuracy may be effectively described by representations using only 2 points. We conclude that even when restricted to classification, our approach analyses a fundamentally more complex situation compared to previous work. On the other hand, Xian et al. (2023) and Wang et al. (2023) support non binary sensitive attributes and group labels, while such an extension for our methods is out of scope for this paper. A comparison of computational complexities for these algorithms may be found in Supplementary J.

## 3 Structure of Fair Representations

In this Section we describe several theoretical properties of fair representations. In Section 3.1 we introduce the problem setup and the necessary notation. In Section 3.2 we discuss relations to classification with accuracy loss and the factorization result, which allows to reduce the size of the representation space. The Invertibility Theorem is introduced in Section 3.3.

## 3.1 Problem Setting

Let A be a binary sensitive variable, and let X be an additional feature random variable, with values in a set X , typically with X = R d . Assume also that there is a target variable Y with finitely many values in a set Y , jointly distributed with X,A .

A representation Z of ( X,A ) is defined as a random variable taking values in some space Z , with (i) distribution given through P θ ( Z | X,A ) , where θ are the parameters of the representation , and (ii) such that conditioned on ( X,A ) , Z is independent of the rest of the variables of the problem. In particular, we have

<!-- formula-not-decoded -->

where ⊥ ⊥ denotes statistical independence.

Fairness in this paper will be measured by the Total Variation distance. For two distributions, µ, ν on R d , with densities f µ , f ν , respectively, this distance is defined as

<!-- formula-not-decoded -->

Note that ∫ | f µ ( x ) -f ν ( x ) | dx is in fact the L 1 distance, and the equivalence ∥·∥ TV = 1 2 ∥·∥ L 1 is well known, see Cover and Thomas (2012).

For a ∈ { 0 , 1 } , let µ a be the distribution of Z given A = a , i.e. µ a ( · ) := P ( Z = ·| A = a ) . We denote the distance induced by the representation as D TV ( Z ) = ∥ µ 0 -µ 1 ∥ TV , and for γ ≥ 0 , we say that the representation Z is γ -fair iff

<!-- formula-not-decoded -->

Note that (3) is a quantitative relaxation of the 'perfect fairness' condition in the sense of statistical parity, which requires Z ⊥ ⊥ A . Specifically, observe that by definition, Z ⊥ ⊥ A iff (3) holds with γ = 0 (i.e. µ 0 = µ 1 ). In addition, as shown in Madras et al. (2018), (3) implies several other common fairness criteria, in particular, bounds on demographic parity and equalized odds metrics for any downstream classifier built on top of Z .

Next, we describe the measurement of information loss in Y due to the representation. Let h : ∆ Y → R be a continuous and concave function on the set of probability distributions on Y , ∆ Y . The quantity h ( P ( Y | X = x )) will measure the best possible prediction accuracy of Y conditioned on X = x , for varying x . As an example, consider the case of binary Y , Y = { 0 , 1 } . Every point in ∆ Y can be written as ( p, 1 -p ) for p ∈ [0 , 1] , and we may choose h to be the optimal binary classification error,

<!-- formula-not-decoded -->

Another possibility it to use the entropy, h ((1 -p, p )) = p log p +(1 -p ) log (1 -p ) . The average uncertainty of Y is given by E x ∼ X h ( P ( Y | X = x )) . Note that this notion does not depend on a

particular classifier, but reflects the performance the best classifier can possibly achieve (under appropriate cost).

The goal of fair representation learning is then to find representations Z that for a given γ ≥ 0 satisfy the constraint (3), and under that constraint minimize the objective E = E θ given by

<!-- formula-not-decoded -->

That is, the representation should minimise the optimal Y prediction error (using Z ) under the fairness constraint.

The curve that associates to every 0 ≤ γ ≤ 1 the minimum of (5) over all representations Z which satisfy (3) with γ is referred to as the Pareto Front of the Fairness-Performance trade-off.

In supplementary material Section A we show that for any representation, E z ∼ Z h ( P ( Y | Z = z )) ≥ E x ∼ X h ( P ( Y | X = x )) , i.e. representations generally decrease or maintain the performance.

## 3.2 Classification and Factorization

In this section we show that the Pareto front of binary classifiers, with accuracy performance and statistical parity fairness measure, can be computed from the Pareto front of representations with total variation fairness measure. In fact, Lemma 3.1 below states that both Pareto fronts amount to the same curve. As discussed in Section 1, this equivalence implies that MIFPO can be used to evaluate fair classifiers, in addition to fair representations.

̸

For a binary classifier ˆ Y of Y , with ( X,A ) as features. The prediction error is defined as usual by ϵ ( ˆ Y ) := P ( ˆ Y = Y ) . The statistical parity of ˆ Y is defined as

<!-- formula-not-decoded -->

Lemma 3.1. Let ˆ Y be a classifier of Y , let the representation uncertainty measure be given by (4) . Then there is a representation given by a random variable Z on a set Z with |Z| = 2 , such that

<!-- formula-not-decoded -->

Conversely, for any given representation Z , there is a classifier ˆ Y of Y as a function of Z (and thus of ( X,A ) ), such that

<!-- formula-not-decoded -->

The Proof of Lemma 3.1 is presented in Supplementary Material Section I.

We now describe the Factorization result. Let f ∗ : X × A → ∆ Y be the Bayes optimal classifier of Y given X,A . That is, for every x ∈ X , a ∈ A , f ∗ ( x, a ) is the conditional distribution of Y given x, a , i.e. f ∗ ( x, a ) = P ( Y = ·| X = x, A = a ) . Denote by ( X ′ , A ) a new pair of random variables, taking values in ∆ Y × A , given by ( X ′ , A ) = ( f ∗ ( X,A ) , A ) .

Lemma 3.2 (Factorization) . For any representation Z of ( X,A ) , there is a representation Z ′ of ( X ′ , A ) , such that

<!-- formula-not-decoded -->

In words, for every representation Z , we can find a representation Z ′ that only accesses ( x, a ) through the value f ∗ ( x, a ) , and is at least as good in terms of both fairness and performance. Equivalently, this means that any two points ( x 1 , a ) and ( x 2 , a ) with coinciding conditional Y distribution may be treated as identical for the purposes of constructing optimal representations. As a result, to find optimal tradeoffs, we can only consider the representations Z ′ on the small space ∆ Y ×A , rather than Z on the much bigger space X × A .

Observations related to Lemma 3.2 were made in the context of classification in Kim et al. (2020),Xian et al. (2023), and Wang et al. (2023), which also start from the Bayes optimal classifier. Lemma 3.2 generalizes these observations to representations and to general losses. The proof may be found in Supplementary K.

Figure 2: (a) The MIFPO Setting (b) Distribution of P ( Y = 1 | X,A ) for each group across datasets.

<!-- image -->

## 3.3 The Invertibility Theorem

In this section we define the notion of invertibility for representations, and show that considering invertible representations is sufficient for computing the Pareto front.

A representation Z on a set Z is invertible if for every z ∈ Z and every a ∈ { 0 , 1 } , there is at most one x ∈ X such that P ( Z = z | X = x, A = a ) &gt; 0 . In words, a representation is invertible, if any given z can be produced by at most two original features ( x, a ) , and at most one for each value a . For z ∈ Z , we say that an ( x, a ) is a parent of z if P ( Z = z | X = x, A = a ) &gt; 0 .

Theorem 3.1. Let Z be any representation of ( X,A ) on a set Z . Then there exists an invertible representation Z ′ of ( X,A ) , on some set Z ′ , such that

<!-- formula-not-decoded -->

In words, for every representation, we can find an invertible representation of the same data which satisfies at least as good a fairness constraint, and has at least as good performance as the original. In particular, this implies that when one searches for optimal performance representations, it suffices to only search among the invertible ones.

The proof proceeds by observing that if an atom z ∈ Z has more than one parent for a fixed a , then one can split this atom into two, with each having less parents. However, the details of this construction are somewhat intricate and the full argument can be found in Section C.

Although in this paper we concentrate on the case of binary sensitive variable, we note that Theorem 3.1 may be extended to multi valued attributes, with a similar argument. In that case, invertibility would mean that every z ∈ Z would still have at most two parents, u, v , corresponding to different values a, a ′ of A .

## 4 The Model Independent Optimization Problem

In this Section we motivate and introduce the MIFPO optimisation problem, and then discuss the full Pareto front computation procedure starting from the raw data.

## 4.1 MIFPO Definition

For the purposes of this Section, we assume that the x feature space X is finite. In the next section, Section 4.2, we describe how we obtain such finite spaces by using the factorization result and discretizing ∆ Y . Note, however, that the full original, possibly high dimensional feature space X , is never discretized.

Write S 0 = { ( x, 0) | x ∈ X} = X × { 0 } , and similarly S 1 = X × { 1 } , for the two halves of the full feature space, X × A = S 0 ∪ S 1 .

Parameters: The MIFPO parameters model the data distribution and are as follows: (a) the probability distributions β 0 ∈ ∆ S 0 and β 1 ∈ ∆ S 1 , on S 0 and S 1 respectively, modeling P (( X,A ) | A = 0) and P (( X,A ) | A = 1) respectively, i.e. the distribution of the data features on each sensitive subgroup. (b) The subgroup proportions α a = P ( A = a ) , and (c) the conditional Y distributions,

ρ u , ρ v ∈ ∆ Y , modeling ρ u = P ( Y = ·| ( X,A ) = u ) when a = 0 or ρ v = P ( Y = ·| ( X,A ) = v ) when a = 1 .

Representation Space: Perhaps the first question one can ask when constructing a representation of the data as above is: How large the representation space should be? We now answer this question using the theory of Section 3.

Fix an integer k ≥ 2 . The representation space Z will be a finite set which can be written as

<!-- formula-not-decoded -->

where [ k ] := { 1 , 2 , . . . , k } . That is, every point z ∈ Z corresponds to some triplet ( u, v, j ) , with u ∈ S 0 , v ∈ S 1 , j ∈ [ k ] . To explain this choice, recall that by the Invertibility result, we know that we may consider only invertible representations. In such representations, every point z ∈ Z is indexed by a pair of parents ( u, v ) ∈ S 0 × S 1 , suggesting that we may index the points by S 0 × S 1 to begin with. Next, for a given such pair ( u, v ) , we may ask how many points z should have the same pair ( u, v ) as their parents? In Supplementary Section D, we show that using k points for every pair, we can obtain uniform approximation over all representations. That is, given a degree of approximation ε , Lemma D.1 provides a bound on k which is sufficient to obtain such approximation. While such a bound would clearly depend on ε , we not that it does note depend on the sizes | S 0 | , | S 1 | . These considerations explain the choice of (11) as the representation space. We have used k = 5 in all experiments.

Variables: The variables of the problem model the representation itself. They will be denoted by r a u,v,j for ( u, v, j ) ∈ Z and a ∈ A , and model the probabilities r a u,v,j = P ( Z = ( u, v, j ) | ( X,A ) = s ) , where either a = 0 and s = u ∈ S 0 , or a = 1 and s = v ∈ S 1 for some v ∈ X . That is, for a = 0 , points u transition to ( u, v, j ) for some v ∈ S 1 , j ∈ [ k ] , and similarly for a = 1 , points v transition to ( u, v, j ) for some u ∈ X , j ∈ [ k ] . This notation preserves our convention that ( u, v, j ) ∈ Z has u and v as its only parents. The situation is illustrated in Figure 2(a).

Note that the variables represent probabilities, and thus satisfy the following constraints:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Performance Objective and Fairness Constraints: With these preparations, we are ready to write the performance cost (5) in the new notation:

<!-- formula-not-decoded -->

Indeed, observe that due to the structure of our representations, every z has two parents, and we have P ( Z = z ) = ( α 0 β 0 ( u ) r 0 u,v,j + α 1 β 1 ( v ) r 1 u,v,j ) . Similarly, P ( Y | Z = z ) is computed via

<!-- formula-not-decoded -->

and substituted inside h to obtain (14). As we show in Supplementary E, the cost (14) is a concave function of the variables r .

We now proceed to discuss the fairness constraint. Recall that we define µ a ( z ) = P ( Z = z | A = a ) , for a ∈ { 0 , 1 } . For z = ( u, v, j ) we have then µ a (( u, v, j )) = β a ( u ) r a u,v,j , for a ∈ { 0 , 1 } , and we can write

<!-- formula-not-decoded -->

and the Fairness constraint, for a given γ ∈ [0 , 1] , is thus simply

<!-- formula-not-decoded -->

We now summarise the full MIFPO problem.

Definition 4.1 (MIFPO) . For a fixed finite ground set X × A = S 0 ∪ S 1 , the problem parameters are the weight α 0 , the distributions β 0 , β 1 , on S 0 and S 1 respectively, and the distributions ρ x ∈ ∆ Y for every x ∈ S 0 ∪ S 1 . The problem variables are { r 0 u,v,j , r 1 u,v,j } ( u,v,j ) ∈Z as defined above. We are interested in minimizing the concave function (14) , subject to the constraints (12) , (13) , and (16) .

The relationship between MIFPO and the Optimal Transport problem is detailed in Supplementary B.

Finally, observe that the MIFPO constraints above linear , with (16) being equivalent to two linear inequality constraints. We note that these constraints may be replace by equivalent linear equality constraints, via appropriate slack variables, which is more convenient in practice. See Supplementary F for details.

## 4.2 The Full Algorithm

In this Section we summarize the full Pareto front computation algorithm, including the estimation of the MIFPO parameters α , β and ρ as discussed above.

Let D = { (( x i , a i ) , y i ) } i ≤ N be the dataset, and write D a = { (( x i , a i ) , y i ) ∈ D | a i = a } , so that D = D 0 ∪ D 1 .

The algorithm proceeds in the following steps: Step 1: we learn the probability estimators c 0 , c 1 : R d → ∆ Y , separately on D 0 and D 1 . These estimators should approximate the optimal Bayes classifier (Section 3.2). Note that such estimation of probabilities is well studied, and is known as calibration , see Niculescu-Mizil and Caruana (2005), Kumar et al. (2019), (Berta et al., 2024).

Step 2: (Discretization and Parameter estimation) For a given integer L &gt; 0 , the space ∆ Y is discretized into L bins. This corresponds to taking the ground sets S 0 , S 1 in MIFPO to be of size L . The data D a is then mapped into the L bins using c a . The distribution β a ( w ) is then simply measures the proportion of points { c a ( x ) } ( x,a ) ∈ D a that fall into bin w ≤ L . Finally, for every bin w , we choose an arbitrary point inside that bin as the representative distribution, ρ w . The parameters α a are estimated simply by α a = | D a | / | D | . Note that for binary Y , ∆ Y is simply the interval [0 , 1] , which is trivial to discretize. See Figure 2(b) for an example of such histograms on real data. We note that one could easily consider more complex discretization schemes, such as clustering, which could be applied efficiently to multi label problems. See Supplementary J for a discussion.

Step 3: For a given a fairness threshold γ &gt; 0 , we can now construct the MIFPO instance, Definition 4.1, with | S 0 | = | S 1 | = L , the additional approximation parameter k , and α, β, ρ as discussed above. As discussed in Section 4.1, we found it sufficient to use k = 5 throughout the paper. The MIFPO is then solved using the existing methods, as detailed in Section 5. The full algorithm is schematically show as Algorithm 1, Supplementary H.1.

## 5 Experiments

Our approach requires two main computational components: building calibrated classifiers to evaluate c a , and solving the discrete optimization problem described in Section 4.2. For the calibrated classifier, we have used XGBoost (Chen et al., 2015), with Isotonic Regression calibration, as implemented in sklearn, Pedregosa et al. (2011). Next, as discussed in Sections 1, 4.1, MIFPO is a concave minimisation problem, under linear constraints. To solve its, we have used the DCCP framework and the associated solver (Shen et al., 2016, 2024), which are based on the combination of convexconcave programming (CCP) Lipp and Boyd (2016) and disciplined convex programming, Grant et al. (2006). We note that although local minima are theoretically possible, the above framework is well-established, and the concave structure can be exploited to allow finding optimal solutions in most practical cases, (Shen et al., 2016). In particular, our results do not indicate local minima issues. However, it may also be worth noting that MIFPO could in principle be also solved with the classical branch-and-bound methods, Benson (1995), which may be slower but do guarantee the global optimum solution.

Throughout the experiments, we use the missclassification error loss h given by (4). Additional implementation details may be found in Supplementary Section H.

Our experimental validation of MIFPO encompasses three standard fairness benchmarks: the Health dataset alongside two variants of ACSIncome-one restricted to California (ACSIncome-CA) and

Figure 3: Comparison of fairness-accuracy trade-offs across three benchmark datasets: Health (left), ACSIncome-CA (middle), and ACSIncome-US (right). MIFPO's Pareto front is represented as a solid line with markers. The horizontal axis represents the fairness constraint (statistical parity distance), while the vertical axis shows prediction accuracy.

<!-- image -->

another spanning the entire United States (ACSIncome-US). In Figure 3 we evaluate MIFPO against five state-of-the-art fair representation techniques: CVIB (Moyer et al., 2019), FCRL (Gupta et al., 2021), FNF (Balunovi´ c et al., 2022b), sIPM (Kim et al., 2023), and Fare (Jovanovi´ c et al., 2023). Each competitive approach was tuned across diverse hyperparameter settings to generate a spectrum of representations balancing fairness and accuracy. Moreover, we evaluated MIFPO against several fair-classification methods on multiple datasets as presented in the Supplementary Section H.

The empirical results presented in Figure 3 demonstrate MIFPO's effectiveness relative to prior approaches. MIFPO consistently achieves performance equal to or superior than the baseline methods across almost all operating points. Furthermore, MIFPO provides a significant methodological advantage through its ability to characterize the complete Pareto frontier. In the figure, MIFPO's performance is visualized as a solid line with points that trace the entire Pareto front, while competing algorithms are represented as individual points corresponding to different hyperparameter configurations.

## 5.1 Implementation

All evaluations can be found at https://github.com/bp6725/ Efficient-Fair-Pareto-Paper . The MIFPO algorithm's source code is available in the https://github.com/bp6725/FairPareto repository. The algorithm is also implemented as the "FairPareto" Python package on PyPI, which provides a scikit-learn compatible API for computing optimal fairness-performance Pareto fronts. The package supports two usage modes: a tabular mode with automatic classifier training and calibration given a sensitive attribute column, and a second mode for any data type (images, text) where users provide pre-trained classifiers for each sensitive group. This enables researchers to benchmark their fair classification methods against theoretical optimality with minimal code and make informed decisions about fairness-performance trade-offs. The package is open-source, available on PyPI, and includes comprehensive documentation with examples for both tabular and image data.

## 6 Conclusions, Limitations, And Future Work

In this paper we have introduced new fundamental properties of optimal fair representations. In particular, these are the first theoretical results that allow approximation of the Pareto front for arbitrary concave performance measures. We have used these results to develop a model independent procedure for the computation of Fairness-Performance Pareto front from data, demonstrated the procedure on real datasets, and have shown that it may be used as a benchmark for other representation learning algorithms.

We now discuss limitations and a few possible directions for future work. This work primarily concentrated on binary sensitive attribute A and binary Y , with the aim to develop the underlying new principles in the simplest case first. As discussed earlier (Sections 1, J), the multi-label case may be treated by more elaborate discretizations. We also noted that the Invertibility Theorem holds for multi valued sensitive attributes as well, which allows to extend the approximation analysis to that case too. Both of these steps, however, would increase the MIFPO problem size. On the other hand, it is also worth noting that this size does not depend directly neither on the feature dimension d , nor on the sample size N and thus the problem scales well in that sense.

In view of these observations, we believe it would be of interest to study the following question on the true complexity of the tradeoff evaluation: Suppose we are given access to the Bayes optimal classifier of the data, f ∗ . This encapsulates, in a sense, most of the 'continuous' information of the problem. Then, how scalable can Pareto estimation methods be made theoretically, in terms of |Y| , |A| , while still maintaining controllable approximation bounds?

## Acknowledgments and Disclosure of Funding

This work has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. 101070568.

## Bibliography

- Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., and Wallach, H. (2018). A reductions approach to fair classification. In International conference on machine learning , pages 60-69. PMLR.
- Alghamdi, W., Hsu, H., Jeong, H., Wang, H., Michalak, P., Asoodeh, S., and Calmon, F. (2022). Beyond adult and compas: Fair multi-class prediction via information projection. Advances in Neural Information Processing Systems , 35:38747-38760.
- Balunovi´ c, M., Ruoss, A., and Vechev, M. (2022a). Fair normalizing flows. In The Tenth International Conference on Learning Representations (ICLR 2022) .
- Balunovi´ c, M., Ruoss, A., and Vechev, M. (2022b). Github for fair normalizing flows. https: //github.com/eth-sri/fnf ,.
- Barocas, S., Hardt, M., and Narayanan, A. (2023). Fairness and machine learning: Limitations and opportunities . MIT Press.
- Benson, H. P. (1995). Concave minimization: theory, applications and algorithms. In Handbook of global optimization , pages 43-148. Springer.
- Berta, E., Bach, F., and Jordan, M. (2024). Classifier calibration with roc-regularized isotonic regression. In International Conference on Artificial Intelligence and Statistics , pages 1972-1980. PMLR.
- Chen, T., He, T., Benesty, M., Khotilovich, V., Tang, Y., Cho, H., Chen, K., Mitchell, R., Cano, I., Zhou, T., et al. (2015). Xgboost: extreme gradient boosting. R package version 0.4-2 , 1(4):1-4.
- Chouldechova, A. and Roth, A. (2018). The frontiers of fairness in machine learning. arXiv preprint arXiv:1810.08810 .
- Cover, T. M. and Thomas, J. A. (2012). Elements of Information Theory . John Wiley &amp; Sons.
- Cruz, A., Belém, C. G., Bravo, J., Saleiro, P., and Bizarro, P. (2023). Fairgbm: Gradient boosting with fairness constraints. In The Eleventh International Conference on Learning Representations .
- Dehdashtian, S., Sadeghi, B., and Boddeti, V. N. (2024). Utility-fairness trade-offs and how to find them. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 12037-12046.

- Du, M., Yang, F., Zou, N., and Hu, X. (2020). Fairness in deep learning: A computational perspective. IEEE Intelligent Systems , 36(4):25-34.
- Fefferman, C., Mitter, S., and Narayanan, H. (2016). Testing the manifold hypothesis. Journal of the American Mathematical Society , 29(4):983-1049.
- Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., and Venkatasubramanian, S. (2015). Certifying and removing disparate impact. In proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining , pages 259-268.
- Gordaliza, P., Del Barrio, E., Fabrice, G., and Loubes, J.-M. (2019). Obtaining fairness using optimal transport theory. In International Conference on Machine Learning , pages 2357-2365. PMLR.
- Grant, M., Boyd, S., and Ye, Y. (2006). Disciplined convex programming . Springer.
- Gupta, U., Ferber, A. M., Dilkina, B., and Steeg, G. V. (2021). Controllable guarantees for fair outcomes via contrastive information estimation.
- Jang, T., Shi, P., and Wang, X. (2021). Group-aware threshold adaptation for fair classification. CoRR , abs/2111.04271.
- Jovanovi´ c, N., Balunovic, M., Dimitrov, D. I., and Vechev, M. (2023). Fare: Provably fair representation learning with practical certificates. In International Conference on Machine Learning , pages 15401-15420. PMLR.
- Kim, D., Kim, K., Kong, I., Ohn, I., and Kim, Y. (2023). Learning fair representation with a parametric integral probability metric.
- Kim, J. S., Chen, J., and Talwalkar, A. (2020). Fact: A diagnostic for group fairness trade-offs. In International Conference on Machine Learning , pages 5264-5274. PMLR.
- Kumar, A., Liang, P. S., and Ma, T. (2019). Verified uncertainty calibration. Advances in Neural Information Processing Systems , 32.
- Lipp, T. and Boyd, S. (2016). Variations and extension of the convex-concave procedure. Optimization and Engineering , 17:263-287.
- Madras, D., Creager, E., Pitassi, T., and Zemel, R. (2018). Learning adversarially fair and transferable representations. In International Conference on Machine Learning , pages 3384-3393. PMLR.
- Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., and Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM computing surveys (CSUR) , 54(6):1-35.
- Moyer, D., Gao, S., Brekelmans, R., Steeg, G. V., and Galstyan, A. (2019). Invariant representations without adversarial training.
- Niculescu-Mizil, A. and Caruana, R. (2005). Predicting good probabilities with supervised learning. In Proceedings of the 22nd international conference on Machine learning , pages 625-632.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research , 12:2825-2830.
- Pessach, D. and Shmueli, E. (2022). A review on fairness in machine learning. ACM Computing Surveys (CSUR) , 55(3):1-44.
- Peyré, G., Cuturi, M., et al. (2019). Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning , 11(5-6):355-607.
- Saleiro, P., Kuester, B., Stevens, A., Anisfeld, A., Hinkson, L., London, J., and Ghani, R. (2018). Aequitas: A bias and fairness audit toolkit. arXiv preprint arXiv:1811.05577 .
- Shen, X., Diamond, S., Gu, Y., and Boyd, S. (2016). Disciplined convex-concave programming. In 2016 IEEE 55th conference on decision and control (CDC) , pages 1009-1014. IEEE.

- Shen, X., Diamond, S., Gu, Y., and Boyd, S. (2024). Github for disciplined convex-concave programming. https://github.com/cvxgrp/dccp/ ,.
- Song, J., Kalluri, P., Grover, A., Zhao, S., and Ermon, S. (2019). Learning controllable fair representations. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 2164-2173. PMLR.
- Wang, H., He, L., Gao, R., and Calmon, F. (2023). Aleatoric and epistemic discrimination: Fundamental limits of fairness interventions. Advances in Neural Information Processing Systems , 36:27040-27062.
- Wang, H., He, L. L., Gao, R., and Calmon, F. P. (2024). Aleatoric and epistemic discrimination: fundamental limits of fairness interventions. NIPS '23, Red Hook, NY, USA. Curran Associates Inc.
- Xian, R., Yin, L., and Zhao, H. (2023). Fair and optimal classification via post-processing. In International conference on machine learning , pages 37977-38012. PMLR.
- Zehlike, M., Hacker, P., and Wiedemann, E. (2020). Matching code and law: achieving algorithmic fairness with optimal transport. Data Mining and Knowledge Discovery , 34(1):163-200.
- Zemel, R., Wu, Y., Swersky, K., Pitassi, T., and Dwork, C. (2013). Learning fair representations. In Dasgupta, S. and McAllester, D., editors, Proceedings of the 30th International Conference on Machine Learning , volume 28 of Proceedings of Machine Learning Research , pages 325-333, Atlanta, Georgia, USA. PMLR.
- Zhao, H. and Gordon, G. J. (2022). Inherent tradeoffs in learning fair representations. Journal of Machine Learning Research , 23(57):1-26.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims are supported in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: A discussion may be found in Section 6, as well as in Section 2 in context of comparison to other methods.

## Guidelines:

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

Justification: All results are formally stated and have proofs in the Supplementary.

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

Justification: All information on reproducibility is included in Section 5 and corresponding Supplementary sections. The full code will be published with the final version of the paper. Guidelines:

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

## Answer: [No]

Justification: While we describe all reproducibility details, the full code will be published with the final version of the paper. The datasets used are publicly available.

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

Justification: We supply all the necessary details for the reproduction of the results. Furthermore, the full details will be included as part of the published code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The methods discussed in this paper produce identical results over multiple runs (std &lt;= 1e-5). We note that for some benchmark methods we have used existing evaluation results; however, the authors of those evaluations claim to have conducted multiple runs to validate stability, and the results presented on the graphs are from different hyperparameters as an integral part of the analysis.

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

Justification: Yes, relevant information is provided in the Supplementary. We note that all experiments of this paper together run in under day on a standard desktop.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have examined the Code of Ethics and verified that the research conforms with the code.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is in Fairness, with obvious societal impact implications. However, while our approach is algorithmically new and can provide practically and empirically better evaluation of fairness-performance tradeoffs, the general societal implications of such tradeoffs are well understood in the field and we believe require no special new consideration in this particular work.

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

Justification: We do not see reasonable scenarios in which release of our code may be risky.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All assets are precisely referenced.

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

Justification: We do not release new assets this time. As mentioned earlier, the implementation of the methods in this paper will be released with the final version of the paper.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing or reesearch with humans.

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

Justification: This research does not involve LLMs as any important, original, or nonstandard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Supplementary Material

| A   | Monotonicity of Loss Under Representations                                                 |   20 |
|-----|--------------------------------------------------------------------------------------------|------|
| B   | MIFPO and Optimal Transport                                                                |   21 |
| C   | Proof Of Theorem 3.1                                                                       |   21 |
| D   | Uniform Approximation and Two Point Representations                                        |   25 |
| E   | Concavity Of E r                                                                           |   27 |
| F   | MIFPO Equality Constraint                                                                  |   28 |
| G   | Additional Proofs                                                                          |   28 |
| H   | Experiments                                                                                |   29 |
|     | H.1 Implementations and computational details . . . . . . . . . . . . . . . . . . . . .    |   29 |
|     | H.2 Additional Evaluations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   30 |
|     | H.3 Minimization of Concave Functions under Convex Constraints and the Entropy Loss        |   31 |
| I   | Fair Classifiers As Fair Representations                                                   |   33 |
| J   | Computational Complexities                                                                 |   34 |
| K   | Factorization                                                                              |   34 |

## A Monotonicity of Loss Under Representations

As discussed in Section 3.1, we observe that representations can not increase the performance of the classifier (i.e decrease the loss).

Lemma A.1. For every ( Y, X, A ) , every representation Z as above, and concave h ,

<!-- formula-not-decoded -->

Note that the right hand-side above can be considered a 'trivial" representation, Z = ( X,A ) .

In what follows, to simplify the notation we use expressions of the form P ( x, a | z ) to denote the formal expressions P ( X = x, A = a | Z = z ) , whenever the precise interpretation is clear from context.

Proof. For every value y ∈ Y , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, on line (18), ∂ P ( x | a,z ) ∂x is the density of P ( x | a, z ) with respect to dx . Crucially, the transition from (18) to (19) is using the property (1). The transition (19) to (20) is a change of notation. Using

(20) and the concavity of h , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B MIFPO and Optimal Transport

In this Section we discuss the relation between the MIFPO minimisation problem, Definition 4.1, and the problem of Optimal Transport (OT). General background on OT may be found in Peyré et al. (2019). We discuss the similarity between OT and the minimisation of (14) under the constraint (16) with γ = 0 , i.e. the perfectly fair case. In this case, (16) is equivalent to the condition β 0 ( u ) r u,v,j = β 1 ( v ) r v,u,j , for all ( u, v, j ) ∈ Z . Next, note that thus in is case the expression for P ( Y | Z = ( u, v, j )) is

<!-- formula-not-decoded -->

and this is independent of the variables r ! Therefore we can write the cost (14) as

<!-- formula-not-decoded -->

Note further that for fixed u, v , the different j 's in this expression play similar roles and could be effectively merged as a single point.

The cost (24) has several similarities with OT. First, in both problems we have two sides, S 0 and S 1 , and we have a certain fixed loss associated with "matching" u and v . In case of (24), this loss is ( α 0 + α 1 ) h ( ρ u α 0 + ρ v α 1 α 0 + α 1 ) , which describes the information loss incurred by colliding u and v in the representation. And second, similarly to OT, (24) it is linear in the variables r . Linear programs are conceptually considerably simpler than minimisation of the concave objective (14).

## C Proof Of Theorem 3.1

In this Section we prove Theorem 3.1.

To keep the notation and the main argument concise, we prove the result under the assumption that ( X,A ) is finitely supported. Since no assumptions are made on the cardinalities of the supports, the general measurable case follows by standard approximation arguments.

We now introduce the additional notation necessary for the proof. Let S 0 , S 1 be finite disjoint sets, where S a represents the values of ( X,A ) when A = a , for a ∈ { 0 , 1 } . Denote S = S 0 ∪ S 1 . We are assuming that there is a probability distribution ζ on S , and A is the random variable A = ✶ { s ∈ S 1 } . X is defined as taking the values s ∈ S , with P ( X = s ) = ζ ( s ) . Further, the variable Y is defined to take values in a finite set Y , and for every s ∈ S , its conditional distribution is given by ρ s ∈ ∆ Y . That is, P ( Y = y | X = s, A = a ) = ρ s ( y ) = ρ s,a ( y ) . 2 This completes the description of the data model.

For a ∈ { 0 , 1 } we denote α a = P ( A = a ) = ζ ( S a ) , and β a ( s ) = P ( X = s | A = a ) . Observe that β a ( s ) = 0 if s / ∈ S a , and β a ( s ) = ζ ( s ) /ζ ( S a ) if s ∈ S a .

We now describe the representation. The representation will take values in a finite set Z . For every s ∈ S and z ∈ Z , let T a ( z, s ) = P ( Z = z | X = s, A = a ) be the conditional probability of representing s as z . T a are sometimes referred as the transition kernels of the representation. For

2 Note that there is a slight redundancy in the notation P ( Y = y | X = s, A = a ) here, since a is determined by s . However, to retain compatibility with the standard notation, literature, we specify them both. This is similar to the continuous situation, in which although A is technically part of the features, X and A are specified separately.

fixed ( X,A ) , the T a 's fully define the distribution of the representation Z and we shall refer to the representation as T or as Z in interchangeably. Finally, for a ∈ { 0 , 1 } denote

<!-- formula-not-decoded -->

With the new notation, a representation T is invertible if for every z ∈ Z and every a ∈ { 0 , 1 } , there is at most one s ∈ S a such that T a ( z, s ) &gt; 0 . In words, a representation is invertible, if any given z can be produced by at most two original features s , and at most one in each of S 0 and S 1 .

Given a representation T and z ∈ Z , we say that an s ∈ S is a parent of z if T a ( z, s ) &gt; 0 for the appropriate a .

Proof Of Theorem 3.1. Assume T is not invertible. Then there is a z ∈ Z which has at least two parents in either S 0 or S 1 . Assume without loss of generality that z has two parents in S 0 . Let

<!-- formula-not-decoded -->

be the sets of parents of z in S 0 and S 1 respectively. Chose a point x ∈ U , and denote by U r = U \{ x } the remainder of U . By assumption we have | U r | ≥ 1 . We also assume that | V | &gt; 0 . The easier case | V | = 0 will be discussed later.

Now, we construct a new representation, T ′ . The range of T ′ will be Z ′ = Z \ z ∪ { z ′ , z ′′ } . That is, we remove z and add two new points. Denote

<!-- formula-not-decoded -->

Then T ′ is defined as follows:

<!-- formula-not-decoded -->

All values of T ′ that were not explicitly defined in (28) are set to 0 . In words, on the side of S 0 , we move all the parents of z except x to be the parents of z ′′ , while z ′ will have a single parent, x . On the S 1 side, both z ′ and z ′′ will have the same parents as z , with transitions multiplied by κ and 1 -κ respectively. The multiplication by κ is crucial for showing both inequlaities in (10).

Note that T ′ can be though of as splitting z into z ′ and z ′′ , such that z ′ has one parent on the S 0 side, and z ′′ has strictly less parents than z had. Once we show that T ′ satisfies (10), it is clear that by induction we can continue splitting T ′ until we arrive at an invertible representation which can no longer be split, thus proving the Lemma.

In order to show (10) for T ′ , we will sequentially show the following claims:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here the probabilities involving z ′ , z ′′ refer to the representation T ′ . Observe that the left hand side of (33) is the contribution of z to the performance cost E t ∼ Z h ( P ( Y | Z = t )) of T , while the right

hand side of (33) is the contribution of z ′ , z ′′ to the performance cost of T ′ . Since all other elements t ∈ Z have identical contributions, this shows the first inequality in (10). Similarly, recall that

<!-- formula-not-decoded -->

and thus the left hand side of (34) is the contribution of z to ∥ µ 0 -µ 1 ∥ TV , with the right hand side being the contribution of z ′ , z ′′ to ∥ µ ′ 0 -µ ′ 1 ∥ TV , therefore yielding the claim ∥ µ ′ 0 -µ ′ 1 ∥ TV = ∥ µ 0 -µ 1 ∥ TV .

Claim (29) : By definition,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, by definition we have

<!-- formula-not-decoded -->

and summing this with (37), we obtain P ( z ) = P ( z ′ ) + P ( z ′′ ) .

Claim (30) : Note that it is sufficient to prove the claim for a = 0 since the probabilities sum to 1. Write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly,

Claim (31) : For a = 1 , let us show P ( s | z, a ) = P ( s | z ′ , a ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The statement P ( s | z, a ) = P ( s | z ′′ , a ) is shown similarly. Next, for a = 0 , we have P ( x | z ′ , a ) = 1 and P ( x | z ′′ , a ) = 0 by the definition of the coupling T ′ . Moreover, for s ∈ U r , P ( s | z ′ , a ) = 0 also follows by the definition of T ′ . Finally, write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim (32) : We first observe that for any representation (and any z),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the property (1) for the transition (59)-(60). Now, using (31), for a = 1 we have

<!-- formula-not-decoded -->

For a = 0 , we have for z ′ using (31):

<!-- formula-not-decoded -->

For a = 0 and z ′′ we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used (31) again on the last line.

Combining (62),(63),(65), and using (30) and the general expression (61), we obtain the claim (32). Claim (33) : This follows immediately from (32) by using (29) and the concavity of h .

Claim (34) : By definition, for every representation, µ a ( z ) = P ( Z = z | a ) = P ( z | a ) P ( z ) P ( a ) . Thus, using (29),(30) we have for a ∈ { 0 , 1 } ,

<!-- formula-not-decoded -->

which in turn yields (34).

It remains only to recall that we have derived (33),(34) under the assumption that | V | &gt; 0 . That is, we assumed that the point z which fails invertability on S 0 has some parents in S 1 . The case when | V | = 0 , i.e. there are no parents in S 1 can be treated using a similar argument, but is much simpler. Indeed, in this case one can simply split z into z ′ and z ′′ and splitting the S 0 weight between them as before, without the need to carefully balance the interaction of probabilities with S 1 via κ .

## D Uniform Approximation and Two Point Representations

As discussed in Sections 1,4.1, we are interested in showing that all optimal invertible representations, no matter which, and no matter on which set Z ′ , can be approximated using a representation with the following property: For every u ∈ S 0 , v ∈ S 1 , there are at most k points z ∈ Z that have ( u, v ) as parents, see Figure 2(a). Here k would depend only on the desired approximation degree, but not on Z ′ , or on the exact representation we are approximating. We therefore refer to this result as the Uniform Approximation result. Its implications for practical use were discussed in Section 4.1.

The notation used in this Section was introduced in the beginning of Section C.

To proceed with the analysis, in what follows we introduce the notion of two-point representation. The main result is given as Lemma D.1 below.

This Section uses the notation of Section 3.3. Let T be an invertible representation, let u ∈ S 0 , v ∈ S 1 be some points, and denote by Z uv = { z j } k 1 the set of all points z ∈ Z which have u and v as parents. Denote by

<!-- formula-not-decoded -->

the total weights of β 0 and β 1 transferred by the representation from u and v respectively to Z uv . Recall that ρ u , ρ v denote the distributions of Y conditioned on u, v . We call the situation above, i.e. the collection of numbers ( { β 0 ( u ) T 0 ( z j , u ) } j ≤ k , { β 1 ( v ) T 1 ( Z j , v ) } j ≤ k ) , a two point representation , since it describes how the weight from the points u, v is distributed in the representation, independently of the rest of the representation. The contribution of Z uv to the global performance cost is

<!-- formula-not-decoded -->

while its contribution to the fairness condition is

<!-- formula-not-decoded -->

Let us now consider two extreme cases of two-point representations. Assume that the total amounts of weight to be represented, w u , w v are fixed. The first case is when k = 1 , and this is the maximum fairness case, since in this case the weights w u , w v overlap as much as possible. Indeed, the contributions to the fairness penalty and performance cost in this case are

<!-- formula-not-decoded -->

respectively. The other extreme case is when w u and w v do not overlap at all. This case be realised with k = 2 , by sending all w u to z 1 and all w v to z 2 . The fairness and performance contributions would be

<!-- formula-not-decoded -->

respectively. Note that the fairness penalty is the maximum possible, while the performance cost is the minimum possible (indeed, this is the cost before the representation, and any representation can only increase it, by Lemma A.1). We thus observed that each two points u, v , with fixed total weight w u , w v , can have their own Pareto front of performance-fairness. One could, in principle, fix a threshold γ uv , | w u -w v | ≤ γ uv ≤ w u + w v for the fairness penalty (70), and obtain a performance cost between that in (71) and (72). However, it is not clear how large the number of points k should be in order to realise such intermediate representations. In the following Lemma we show that one can uniformly approximate all the points on the two-point Pareto front using a fixed number of points, that depends only on the function h . This means that in practice one can choose a certain number n of z points, and have guaranteed bounds on the possible amount of loss incurred with respect to all representations of all other sizes.

Lemma D.1. For every ε &gt; 0 , there a number n = n ε depending only on the function h , with the following property: For every two-point representation ( { β 0 ( u ) T 0 ( z j , u ) } j ≤ k , { β 1 ( v ) T 1 ( Z j , v ) } j ≤ k ) , with total weights w u , w v , there is a two point representation T ′ on a set Z ′ u,v , with the same total weights, such that |Z ′ uv | ≤ n , and such that

<!-- formula-not-decoded -->

Proof. To aid with brevity of notation, define for j ≤ k

<!-- formula-not-decoded -->

Then we can write

<!-- formula-not-decoded -->

where Λ is the vector Λ = ( α -1 0 , -α -1 1 ) , c j = ( c j 0 , c j 1 ) , and Λ c j is the inner product of the two.

Observe that the cost E uv,T depends on c j mainly through the fractions c j 0 c j 0 + c j 1 . Our strategy thus would be to approximate all k of such fractions by a δ -net of a size independent of k . To this end, set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since h is continuous (by assumption), and defined on a compact set, it is uniformly continuous, and so is h uv . By definition, this means there is a δ &gt; 0 such that for all p, p ′ with | p -p ′ | ≤ δ , it holds that | h uv ( p ) -h uv ( p ′ ) | ≤ ε . Let us now choose { x i } n i =1 to be a δ net on [0 , 1] . For every i ≤ n set

<!-- formula-not-decoded -->

That is, Γ i is the set of indices j such that p j is approximated by x i . Using x i we construct the representation T ′ as follows: For a ∈ { 0 , 1 } set

<!-- formula-not-decoded -->

For n new points, z i ∈ Z ′ uv , set T ′ 0 ( z ′ i , u ) = c ′ i 0 /β 0 ( u ) , T ′ 1 ( z ′ i , v ) = c ′ i 1 /β 1 ( v ) . Note that the total weights are preserved, ∑ i ≤ n c ′ i 0 = w u and ∑ i ≤ n c ′ i 1 = w v .

Next, for every j ∈ Γ i we have

<!-- formula-not-decoded -->

and define h uv : [0 , 1] → R by

Thus

<!-- formula-not-decoded -->

Next, observe that by the construction of x i ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition,

<!-- formula-not-decoded -->

where we have used (81) in the last transition.

Combining the two inequalities yields the second part of (73),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

yielding the first part of, and thus completing the proof of, statement (73).

It remains to observe that above we have used a δ net for h uv , which depends on ρ u , ρ v . However, we can directly build an appropriate δ -net in full range of h , the simplex ∆ Y , which would produce bounds valid for all u, v . Indeed, let δ ′ be such that | h ( ν ) -h ( ν ) | ≤ ε for all µ, ν ∈ ∆ Y with ∥ u -v ∥ 1 ≤ δ ′ . Observe that the map p ↦→ pρ v +(1 -p ) ρ u is 2-Lipschitz from R to ∆ Y equipped with the ∥·∥ 1 norm, for any u, v ∈ ∆ Y . Thus, choosing δ = 1 2 δ ′ , we have | h uv ( p ) -h uv ( p ′ ) | ≤ ε if | p -p ′ | ≤ δ . This completes the proof of the Lemma.

## E Concavity Of E r

Note that the variables r appear in (14) both as coefficients multiplying h and inside the arguments of h , in a fairly involved manner. Nevertheless, the cost turns out to still retain an interesting structure, as it is concave , if h is. We record this in the following Lemma.

Lemma E.1. If h : ∆ Y → R is concave, then of every ρ 1 , ρ 2 ∈ ∆ Y the function g : R 2 → R , given by g (( c 1 , c 2 )) = ( c 1 + c 2 ) h ( c 1 ρ 1 + c 2 ρ 2 c 1 + c 2 ) is concave.

Proof. It is sufficient to show that for every c, c ′ ∈ R 2 , we have g (( c + c ′ ) / 2) ≥ 1 2 ( g ( c ) + g ( c ′ ) . To this end, define the map F : R 2 → ∆ Y by

<!-- formula-not-decoded -->

Finally, note that

and note that

<!-- formula-not-decoded -->

It then follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F MIFPO Equality Constraint

As noted in the main text, although the inequality constraint (16) is convex in the variables r , and can be incorporated directly into most optimisation frameworks, it may be significantly more convenient to work with equality constraints. Using the following Lemma, we can find equivalent equality constraints in a particularly simple form.

Lemma F.1. Let µ 0 , µ 1 ∈ ∆ Z be two probability distributions over Z and fix some γ ≥ 0 . If ∥ µ 0 -µ 1 ∥ TV = γ then there exist ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 . In the other direction, if there exist ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 , then ∥ µ 0 -µ 1 ∥ TV ≤ γ .

The proof may be found in Section G.

As consequence of this result, if we find distributions ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 holds, then we know that (16) also holds, and conversely, if (16) holds, then distributions as above exist.

Using this observation, we introduce new variables, ϕ 0 u,v,j and ϕ 1 u,v,j , for every ( u, v, j ) ∈ Z , which correspond to ϕ 0 (( u, v, j )) and ϕ 1 (( u, v, j )) respectively. These variables will be required to satisfy the following constraints:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here the first two lines encode the fact that ϕ 0 , ϕ 1 are probabilities, while the third line encodes the fairness constraint, as discussed above.

## G Additional Proofs

Proof Of Lemma F.1

Proof. For this proof it is more convenient to work with the ℓ 1 norm ∥·∥ 1 directly. Recall that ∥ µ 0 -µ 1 ∥ TV = 1 2 ∥ µ 0 -µ 1 ∥ 1 and that

<!-- formula-not-decoded -->

Assume that ∥ µ 0 -µ 1 ∥ 1 = 2 γ . Define the functions ¯ ϕ 0 ( z ) = ✶ { µ 1 ≥ µ 0 } ( z ) · ( µ 1 ( z ) -µ 0 ( z )) and ¯ ϕ 1 ( z ) = ✶ { µ 0 ≥ µ 1 } ( z ) · ( µ 0 ( z ) -µ 1 ( z )) . Note that we then have

Indeed, define

<!-- formula-not-decoded -->

Note also that we can write

<!-- formula-not-decoded -->

which combined with (103) yields (101).

Next, we can also directly verify that

<!-- formula-not-decoded -->

and thus setting ϕ 0 = γ -1 ¯ ϕ 0 , ϕ 1 = γ -1 ¯ ϕ 1 completes the proof of this direction.

In the other direction, given ϕ 0 , ϕ 1 ∈ ∆ Z such that µ 0 + γϕ 0 = µ 1 + γϕ 1 , we have

<!-- formula-not-decoded -->

thus completing the proof.

## H Experiments

This section describes additional evaluation details and experiments with fair classifiers. In SectionH.1 we provide the main algorithm figure, and discuss technical implementation details. Section H.2 contains the comparison to a number of fair classifiers, and Section H.3 discusses implementation of the entropy cost h within the DCCP framework.

## H.1 Implementations and computational details

## Algorithm 1 MIFPO Implementation

<!-- formula-not-decoded -->

The Pareto front evaluation requires two main parts - building a calibrated classifier required for evaluating c a = P ( Y | X,A = a ) , and later solving the optimization problem MIFPO (see Algorithm 1).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Comparison to FGBM and Post-processed LGBM across Datasets

Figure 4: Comparing common fair classification pipelines to the MIFPO Pareto front, for LSAC, COMPAS, and ADULT datasets. For FGBM and LGBM+Post Process methods, each point represents a trade-off obtained at a single hyper-parameter configuration.

<!-- image -->

For a calibrated classifier, we are using standard model calibration. Model calibration is a wellstudied problem where we fit a monotonic function to the probabilities of some base model so that the probabilities will reflect real probabilities, that is, P ( Y | X ) . Here, we used Isotonic regression (Berta et al., 2024) for model calibration with XGBoost (Chen et al., 2015) as the base model. For training the XGBoost model, a GridSearchCV approach is employed to find the best hyperparameters from a specified parameter grid, using 3-fold cross-validation.

The experiments were conducted on a system with an Intel Core i9-12900KS CPU (16 cores, 24 threads), 64 GB of RAM, and an NVIDIA GeForce RTX 3090 GPU.

## H.2 Additional Evaluations

Following the equivalence between the Pareto fronts of fair binary classification and representations for the accuracy cost (Section 3.1), we evaluated the accuracy-fairness trade-off for some common fair classification algorithms. For our evaluation, we selected the most widely used algorithms based on GitHub repository stars and citation counts in the literature, demonstrating the importance of our proposed method in comparison to common approaches. We also evaluate FairFront, a more recent approach introduced in Wang et al. (2023).

Fair classifiers generally fall into pre-processing, in-processing, and post-processing categories. Pre- and post-processing types often utilize standard classifiers as part of their fair classification pipeline. We specifically evaluated two widely adopted algorithms, representing different categories: FairGBM Cruz et al. (2023): An in-processing method where a boosting trees algorithm (LightGBM) is subjected to pre-defined fairness constraints. Balanced-Group-Threshold Jang et al. (2021): A post-processing method which adjusts the threshold per group to obtain a fairness criterion. For FairGBM, we used the original implementation provided by the authors. For Balanced-GroupThreshold post-processing, we utilized implementations available via Aequitas Saleiro et al. (2018), a popular bias and fairness audit toolkit.

We conducted our evaluation using three of the most common datasets in this field, which are known to have relevance to real-world decision-making processes: the Adult dataset (income prediction), COMPAS (recidivism prediction), and LSAC (law school admission).

It is important to note that, as a rule, common fairness classification methods are not designed to control the fairness-accuracy trade-off explicitly. Instead, in most cases, these methods rely on rerunning the algorithm for a wide range of hyperparameter settings, in the hope that different hyperparameters would result in different fairness-accuracy trade-off points. However, there typically is no direct known and controlled relation between hyperpatameters and the obtained fairness-accuracy trade-off. For FairGBM, we utilized the hyperparameter ranges specified in the original paper, Cruz et al. (2023). In the case of the balancing post-processing method, we conducted a grid search over the full range of all possible hyperparameters to ensure a comprehensive analysis.

Figure 4 shows the MIFPO computed Pareto front, and all hyperparameter runs of the two algorithms above, with accuracy evaluated on the test set. These experiments demonstrate the following two points: (a) The standard classifiers achieve a considerably lower accuracy than what is theoretically possible at a given fairness level. (b) the existing methods are also unable to present solutions for

Figure 5: Comparing FairFront to MIFPO accuracy-fairness tradeoff on two curated datasets.

<!-- image -->

the full range of the statistical parity values. The values from the FGBM and the post-processing algorithms all have statistical parity ≤ 0 . 2 . Similarly to to the case of fair representations, these results emphasize the limitations of current fair classifiers in achieving optimal trade-offs between accuracy and fairness across the full range of fairness values.

Additionally, we add a comparison to FairFront (Wang et al., 2024), which, similarly to our work, depends on estimates of P ( Y | X ) . See the discussion in Section 2. We note, however, that the utility of this comparison is limited due to the very strict setup defined in that paper, which cannot be applied to natural datasets. Specifically: 1) The implementation in Wang et al. (2023) requires creating a finite discrete space by binning over the full R d feature space 3 , which allows for perfect modeling of the distribution P ( Y | X ) (by simple counting). This is not normally possible on real datasets in practice. 2) The binning itself was performed manually for each dataset separately, and we were unable to discern the logic behind the selected parameters. 3) To allow reasonable binnig, the true dimensions of the data are manually reduced, again by picking features manually for each dataset. Due to these issues, it was practically impossible to apply the method in Wang et al. (2023) to other standard datasets, or even to the datasets used in Wang et al. (2023) but with standard features.

Nonetheless, for the sake of comparison, we compared MIFPO and FairFront on the same restricted, preprocessed version of the datasets that FairFront used, with the same bin-based probability estimator. The results are shown in Figure 5.

## H.3 Minimization of Concave Functions under Convex Constraints and the Entropy Loss

As described in figure 5, we used the disciplined convex concave programming (DCCP) framework and the associated solver, (Shen et al., 2016, 2024) for solving the concave minimization with convex constraints problem.

Minimizing concave functions under convex constraints is a common problem in optimization theory. Unlike convex optimization where global minima can be readily found, in concave minimization problems we only know that the local minimas lie on the boundaries of the feasible region defined by the convex constraints. While techniques such as branch-and-bound algorithms, cutting plane methods, and heuristic approaches are often employed, here we used the framework of DCCP which gain a lot of popularity in recent years.

The DCCP framework extends disciplined convex programming (DCP) to handle nonconvex problems with objective and constraint functions composed of convex and concave terms. The idea behind a "disciplined" methodology for convex optimization is to impose a set of conventions inspired by basic principles of convex analysis and the practices of convex optimization experts. These "disciplined" conventions, while simple and teachable, allow much of the manipulation and transformation required for analyzing and solving convex programs to be automated. DCCP builds upon this idea, providing an organized heuristic approach for solving a broader class of nonconvex problems by combining DCP principles with convex-concave programming (CCP) methods, and is implemented as an extension to the CVXPY package in Python.

While convenient, the use of the disciplined framework bears some limitations. Mainly, generic operations like element-wise multiplication are not under the allowed set of operations (and for

3 This is completely different from the discretizaton of ∆ Y used in MIFPO. The space R d is the feature space, and is much larger than ∆ Y .

obvious reasons), which limits the usability. Notice, that for the prediction accuracy measure h ( p ) = min( p, 1 -p ) this is not a problem, but for the entropy classification error h ((1 -p, p )) = -p log p -(1 -p ) log(1 -p ) , this is more challenging. Nevertheless, here we show that the standard DCCP framework allows for entropy classification error.

<!-- formula-not-decoded -->

We can write the cost function under the entropy accuracy error as:

<!-- formula-not-decoded -->

Proof.

Thus,

<!-- formula-not-decoded -->

Hence :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, the expression can be written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given the element-wise entropy function x · (1 -x ) is with known characteristics and under the dccp framework, we can use the entropy error for our cost using :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## I Fair Classifiers As Fair Representations

As discussed in Section 3.1, Pareto front of binary classifiers with statistical parity can be computed from the Pareto front of representations with total variation fairness distance. In this Section we provide the proof of this result, Lemma 3.1. The Lemma and the related notation are restated here for convenience.

̸

For a binary classifier ˆ Y of Y , its prediction error is defined as usual by ϵ ( ˆ Y ) := P ( ˆ Y = Y ) . The statistical parity distance of ˆ Y is defined as

<!-- formula-not-decoded -->

Let the uncertainy measure h be defined by (4). Note that the first part of the Lemma does use the special properties of this h and does not necessarily hold for other costs h .

Lemma I.1. Let ˆ Y be a classifier of Y . Then there is a representation given by a random variable Z on a set Z with |Z| = 2 , such that

<!-- formula-not-decoded -->

Conversely, for any given representation Z , there is a classifier ˆ Y of Y as a function of Z (and thus of ( X,A ) ), such that

<!-- formula-not-decoded -->

Proof. Let us begin with the second part of the Lemma, inequalities (109). Given a representation Z , ϵ ( ˆ Y ) ≤ E z ∼ Z h ( P ( Y | Z = z )) follows since E z ∼ Z h ( P ( Y | Z = z )) is the error of the optimal classifier of Y as a function of Z . We choose ˆ Y to be such an optimal classifier and thus satisfy the above inequality, with equality. Next, the second inequality in (109) holds for for any classifier ˆ Y derived from Z . The argument below is a slight generalisation of the argument in Madras et al. (2018). Define f ( z ) = P ( ˆ Y = 1 | Z = z ) . Note that for a ∈ { 0 , 1 } , P ( ˆ Y = 1 | A = a ) = ∫ f ( z ) dµ a ( z )

. Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used | f | ≤ 1 in the second line. Repeating the argument also for P ( ˆ Y = 1 | A = 1 ) -P ( ˆ Y = 1 | A = 0 ) , we obtain the second inequality in (109).

We now turn to the first statement, (108). Let ˆ Y be a classifier of Y as a function of ( X,A ) . Observe that thus by definition P ( ˆ Y | X,A ) induces a distribution on the set { 0 , 1 } , and thus may be considered as a representation Z := ˆ Y of ( X,A ) on that set. We now relate the properties of this Z as a representation to the quantities ϵ ( ˆ Y ) and ∆ SP ( ˆ Y ) . Similarly to the argument above, the first part of (108) follows since E z ∼ Z h ( P ( Y | Z = z )) is the best possible error over all classifiers. For the second part, note that since ˆ Y is binary, we have

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used (113) for the second to third line transition. This completes the proof of the second part of (108).

## J Computational Complexities

In this Section we discuss alternative discretization schemes for the MIFPO algorithm. We also discuss various complexity aspects of the classification algorithms Xian et al. (2023) and Wang et al. (2023) and relate them to the complexity of MIFPO.

In Section 4.2 we described a data independent discretization of ∆ Y by binning. While effective for small label sets Y , larger sets would require a different approach. One alternative is to cluster the data instead of binning ∆ Y itself. Indeed, by choosing cluster centers { η i } M 1 ⊂ ∆ Y , such that each data point { f ∗ ( x, a ) } ( x,a ) ∈ D is well approximated by one the centers (or just most points are approximated), we can guarantee arbitrarily good approximation of the true Pareto front. The cardinality M of such clustering would depend on the intrinsic dimension of the data, which we typically expect to be lower than the full dimension of ∆ Y , due to the Manifold Hypothesis, Fefferman et al. (2016).

Another possibility is to not use any explicit discretization, and instead to use each data point as a separate bin (equivalently, we use the points themselves as the cluster centers { η i } ). In this case, the complexity scales with the size of the dataset, but not with the dimensions of ∆ Y . The MIFPO construction in Section 4.1 implies that MIFPO in this case would have O ( N 2 ) variables for a dataset of size N . While not applicable for large N , this is similar to the complexity of a variety of often used algorithms. Classical example of such complexity is the Spectral Clustering. We also observe that Xian et al. (2023), a recent state of the art fairness classification algorithm mentioned above, also has such complexity.

Indeed, the approach in Xian et al. (2023) involves the computation of a certain transportation plan between data points, and the encoding of such plans also requires O ( N 2 ) variables. Thus problem sizes occuring in MIFPO would be smaller or equal to those in Xian et al. (2023), despite the fact that MIFPO is solving a considerably more general problem (see Section 2).

Finally, the algorithm in Wang et al. (2023) involves optimization in the space of confusion matrices, with dimensions of size |A| · |Y| × |Y| . As discussed in Section 2, the reduction of the problem to confusion matrices of possible due to special properties of the classification problem and the accuracy loss.

The algorithm in Wang et al. (2023), FairFront, is an iterative algorithm, where each iteration involves solving a certain difference-of-convex (DC) program which is constructed from a full dataset. The class of DC programs is equivalent to that convex-concave programs considered in this paper (see Shen et al. (2016)). In fact, similarly to MIFPO, the algorithm in Wang et al. (2023) also uses DCCP, Shen et al. (2016), although applied to a different problem.

In each iteration, the solution of DC program is then used to add new constraints to a certain main convex program. While it is proved that asymptotically this process converges to the optimal front, there are no bounds on the number of iterations. This may lead to the convex solver crashing due to too many constraints, and in fact we have observed such crashes in our evaluation.

To summarize 4 , MIFPO involves solving one convex-concave problem, with size which may be independent of the data size. In contrast, FairFront involves iteratively solving convex-concave problems and a main convex program, where the number of terms in the objective of each convexconcave problem scales with the size of the data, and the number of constraints grows in the convex problem grows with iterations, thus making the iterations progressively harder.

## K Factorization

In this Section we the factorization result, Lemma 3.2. We restate the result for convenience.

Recall that f ∗ ( x, a ) denotes the Bayes optimal classifier of Y , i.e. f ∗ : X × A → ∆ Y is given by

<!-- formula-not-decoded -->

We define a new variable X ′ , with values in ∆ Y , by X ′ := f ∗ ( X,A ) .

4 In this Section we have discussed the theoretical complexity aspects of FairFront. Additional details pertaining to the official implementation of Wang et al. (2023) may be found in Section H.2.

Lemma K.1 (Factorization) . For any representation Z of ( X,A ) , there is a representation Z ′ of ( X ′ , A ) , such that

<!-- formula-not-decoded -->

Proof. The representation Z ′ of X ′ , A will be defined as follows: For σ ∈ ∆ Y , a ∈ A , z ∈ Z set:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second line is the definition and is added for clarity. To see intuition behind this definition note that we have

<!-- formula-not-decoded -->

In words, for a fixed a , to compute P ( Z ′ = z | X ′ = σ, A = a ) we effectively collect all x such that f ∗ ( x, a ) = σ and average all of their representations. Equivalently, all points x with the same σ are merged into one point, and their representations summed up according to their relative weight.

We will now show that neither the performance nor the fairness condition change under this operation.

Since D TV ( Z ) is defined solely in terms of the distributions P ( Z = ·| A = a ) , to show that D TV ( Z ) = D TV ( Z ′ ) it is enough to show that

<!-- formula-not-decoded -->

To this end, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the transition (124) to (125) is by the definition of Z ′ . Thus we have shown that D TV ( Z ) = D TV ( Z ′ ) .

Next, note that the above argument implies also that P ( Z = z ) = P ( Z ′ = z ) . Thus, in order to show the performance equality,

<!-- formula-not-decoded -->

it is enough to show that P ( Y | Z ′ = z ) = P ( Y | Z = z ) for every z ∈ Z . Further, again since P ( Z = · ) = P ( Z ′ = · ) , we can show that

<!-- formula-not-decoded -->

for all z ∈ Z , y ∈ Y . Write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(136)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the transition (132)-(133) is due to the independence condition (1). On line (134) we split the sum over x into sum over subsets { x | f ∗ ( x, a ) = σ } and an outer some over all σ . The transition (136)-(137) is due to the equality (122)-(121). Finally, for the transition (137)-(138), we have used the fact that

<!-- formula-not-decoded -->

which holds by definition of X ′ . Similarly to the earlier discussion on merging of x with similar value of f ∗ , the above argument proceeded by regrouping the summation over x by the value of f ∗ ( x, a ) . The computation thus showed that this process yields the definition of Z ′ . In particular, this regrouping process and equation (141) explain why the space ∆ Y is special and all representations may be factored through it. This completes the proof of the Lemma.

In the above argument we have used the summation over σ , i.e. ∑ σ . . . . This is formally possible when ( X,A ) has a discreet distribution. The full general case may be obtained simply by replacing the summation by integration and conditioning on σ by the general conditional expectation operator.
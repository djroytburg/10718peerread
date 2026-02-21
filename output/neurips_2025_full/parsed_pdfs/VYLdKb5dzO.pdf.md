## Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization

Milad Sefidgaran 1 , Kimia Nadjahi 2 , Abdellatif Zaidi 1 , 3

1 Paris Research Center, Huawei Technologies France

2 CNRS, ENS Paris, France 3 Universit´ e Gustave Eiffel, France milad.sefidgaran2@huawei.com, kimia.nadjahi@ens.fr, abdellatif.zaidi@univ-eiffel.fr

## Abstract

In this paper, we leverage stochastic projection and lossy compression to establish new conditional mutual information (CMI) bounds on the generalization error of statistical learning algorithms. It is shown that these bounds are generally tighter than the existing ones. In particular, we prove that for certain problem instances for which existing MI and CMI bounds were recently shown in Attias et al. [2024] and Livni [2023] to become vacuous or fail to describe the right generalization behavior, our bounds yield suitable generalization guarantees of the order of O (1 / √ n ) , where n is the size of the training dataset. Furthermore, we use our bounds to investigate the problem of data 'memorization' raised in those works, and which asserts that there are learning problem instances for which any learning algorithm that has good prediction there exist distributions under which the algorithm must 'memorize' a big fraction of the training dataset. We show that for every learning algorithm, there exists an auxiliary algorithm that does not memorize and which yields comparable generalization error for any data distribution. In part, this shows that memorization is not necessary for good generalization.

## 1 Introduction

One of the major problems in statistical learning theory consists in understanding what really drives the generalization error of learning algorithms. That is, what makes an algorithm trained on a given dataset continue to perform well on unseen data samples. Historically, this fundamental question has been studied independently in various lines of work, using seemingly unconnected tools. This includes VC-dimension theory [1], Rademacher complexity approaches [2], stability-based analysis [3] and, more recently, intrinsic-dimension [4-8] and information-theoretic approaches [9-21]. It is only until recently that the above various approaches were shown to be possibly unified [22, 23] using a variable-length compressibility technique, which is rate-distortion-theoretic in nature.

In the context of statistical learning theory perhaps one can date back information-theoretic approaches to the PAC-Bayes bounds of McAllester [24, 25], which were then followed by various extensions and ramifications [26-39]. The mutual information (MI) bounds of [9] and [10] have the advantages to be relatively simpler comparatively and of offering somewhat clearer insights into the question of generalization. Roughly, such bounds suggest that a learning algorithm generalizes better as its output model reveals less information about the training data samples, where the amount of revealed information is measured in terms of the Shannon mutual information.

However, MI-based bounds are also known to sometimes take large (infinite) values and become vacuous, such as for continuous data and deterministic models. This shortcoming has been identified in a number of works, including [40, 41]. The issue was believed to be resolved by the introduction in [12] of the important framework of conditional mutual information (CMI). The CMI setting introduces a 'super-sample' construction in which an auxiliary 'ghost sample' is used in conjunction with the training sample; and a sequence of Bernoulli random variables determines which data samples among the super-sample were

used for the training. It is shown that a bound on the generalization error involves the mutual information between the Bernoulli random variables and the hypothesis (e.g., model parameters), conditionally given the super-sample [12, Theorem 2]. Because the entropy of Bernoulli random variables is bounded, the resulting bound is bounded. Many follow-up works have proposed extensions and improvements of the original CMI bounds, including using randomized subset and individual sample techniques, disintegration, and fast-rate variations in regimes in which the empirical risk is small - See [42] for more on this.

CMI-type bounds were largely believed to be exempt from the aforementioned limitations of MI bounds until it was recently reported that examples can be constructed for which the standard 1 CMI-based bound and its individual-sample variant fail [14, 43, 46]. The (counter-) examples of [46] are in the context of Stochastic Convex Optimization (SCO) problems; and those of [43] involve carefully constructed Convex-Lipschitz-bounded (CLB) and Convex-set-Strongly convex-Lipschitz (CSL) instance problems. These limitations were sometimes extrapolated to the extent of even questioning the utility of informationtheoretic bounds for the analysis of the generalization error of statistical learning algorithms more generally [47]. In this context, we mention [23, Appendix A] in which it was shown that, when applied to the counter-example of [47], a lossy version of MI bounds yields generalization bounds that are of order O (1 /n ) , instead of Ω(1) in the case of standard (lossless) MI bounds. 2 The idea of lossy compression was also used in [49].

In this paper, essentially, we show that the aforementioned limitations are in fact not inherent to the CMI framework; and, actually, the CMI framework can be adjusted slightly by the incorporation of a suitable stochastic projection and a suitable lossy compression to cope with those issues. Also, leveraging the utility of CMI and membership inference to study the problem of memorization and its relationship to generalization in machine learning, we use our results to revisit the necessity of memorization for SCO problems claimed in [43]. We show that memorization is not necessary for good generalization; and, as such, the result contributes to a better understanding of what role memorization plays in machine learning, a problem which is yet to be fully understood. Specifically, our contributions are as follows.

- We introduce stochastic projection in conjunction with lossy compression in the CMI framework, and we use them to establish a new CMI-based bound that is generally tighter than the CMI bounds of [12].
- We show that, in sharp contrast with classic CMI-based bounds which fail when applied to the aforementioned CLB, CSL and SCO problem instances of [43, 46] and may not even decay with the number of training samples, our new CMI bound yields meaningful results and decays with the number of training samples as O (1 / √ n ) .
- By applying them to generalized linear stochastic (non-convex) optimization problems, in the appendices we demonstrate that our bounds remain non-vacuous even beyond the convex case previously studied in [50]. The generalization is shown to come at the expense of a slower decay with n in our case; namely, O (1 / 4 √ n ) instead of O (1 / √ n ) if the functions are convex as in [50].
- We leverage the key ingredients of stochastic projection and lossy compression in the framework of CMI to study the 'memorization' issue identified and studied in [43]. Specifically, [43] has demonstrated that, for a given problem instance and every ε -learner algorithm, there exists a data distribution under which the algorithm 'memorizes' the training samples. We show that for any learning algorithm A that memorizes the training data, one can find (via stochastic projection and lossy compression) an alternate learning algorithm ˜ A with comparable generalization error and that does not memorize the training data for any data distribution. In part, this means that memorization is not necessary for good generalization in SCO.
- In the appendices, we use our general bound to study the generalization error of subspace training algorithms. Specifically, we investigate the setting in which the training is performed using SGD or SGLD; and we derive new bounds based on the differential entropy of Gaussian mixture distributions. This entropy depends on the gradient difference for the training and test datasets, the noise power, the learning rate, and the uncertainty of the index of the training dataset within the super-dataset.

1 The authors of [43] do not evaluate the performance of variants of CMI such as chained CMI [44], evaluated CMI and f -CMI [20, 21, 45] on their counter-example.

2 The counterexample of [47] has also been addressed by Wang and Mao [48] using a different technique called 'Sample-Conditioned Hypothesis Stability'.

## 2 Notation and Background

Let Z be some random variable with unknown distribution µ and taking values in some alphabet Z . Let S n ≜ ( Z 1 , . . . , Z n ) ∈ Z n be a set of n data samples drawn uniformly from the distribution µ , i.e., S n ∼ P S n = µ ⊗ n . In the framework of statistical learning, a (possibly) stochastic learning algorithm A : Z n →W takes the training dataset S n as input and returns a hypothesis W ∈ W ⊆ R D . We assume that A is randomized , in the sense that its output W ≜ A ( S n ) is a random variable distributed according to P W | S n . We denote the distribution induced on ( S n , W ) as P S n ,W = P W | S n ⊗ P S n = P W | S n ⊗ µ ⊗ n .

For a given function ℓ : Z × W → R , the loss incurred by using a hypothesis w ∈ W for a sample z is evaluated as ℓ ( z, w ) . A statistical learning algorithm seeks to find a hypothesis w whose population risk R ( w ) ≜ E Z ∼ µ [ ℓ ( Z, w )] is minimal. However, since the data distribution µ is unknown, direct computation of the population risk R ( w ) is not possible. Instead, one resorts to minimizing the empirical risk ̂ R ( s n , w ) ≜ 1 n ∑ n i =1 ℓ ( z i , w ) or a regularized version of it. Throughout, if s n is known from the context, we will use the shorthand notation ̂ R n ( w ) ≡ ̂ R ( s n , w ) .

The generalization error induced by a specific choice of hypothesis w ∈ W and dataset s n is evaluated as

<!-- formula-not-decoded -->

and the expected generalization error of the learning algorithm A is obtained by taking the expectation over all possible choices of ( s n , w ) , as

<!-- formula-not-decoded -->

## 2.1 Conditional Mutual Information Framework

Let ˜ S ∈ Z n × 2 be a super-sample composed of 2 n data points Z i,j that are drawn uniformly from the distribution µ , where j ∈ { 0 , 1 } and i ∈ [ n ] . Also, let J = ( J 1 , . . . , J n ) ∈ { 0 , 1 } n be a vector of n independent Bernoulli (1 / 2) random variables, all drawn independently from ˜ S . Let ˜ S J = { Z 1 ,J 1 , Z 2 ,J 2 , . . . , Z n,J n } . In what follows, ˜ S J plays the role of the training dataset S n , ˜ S \ ˜ S J plays the role of a test or 'ghost' dataset S ′ n and ˜ S is a shuffled version of the union of the two. For an algorithm A : Z n →W , its CMI with respect to the data distribution µ is defined as

<!-- formula-not-decoded -->

The CMI captures the information that the output hypothesis of the algorithm A trained on ˜ S J provides about the membership vector J given the super-sample ˜ S . Equivalently, the CMI measures the extent to which the training and test datasets are distinguishable given the shuffled version of the union of the two, as well as the trained model. In its simplest form, it is shown in [12] that the generalization error of an algorithm for a bounded loss in the range [0 , 1] can be upper-bounded as

<!-- formula-not-decoded -->

Furthermore, for a Convex-Lipschitz-Bounded (CLB) whose formal definition will follow, the generalization error of A was shown in [47] to be upper-bounded as

<!-- formula-not-decoded -->

Definition 1 (SCO Problem) . A stochastic convex optimization (SCO) problem is a triple ( W , Z , ℓ ) , where W ∈ R D is a convex set and ℓ ( z, · ): W → R is a convex function for every z ∈ Z .

Definition 2 (Convex-Lipschitz-Bounded (CLB)) . An SCO problem is called CLB if i) for every w ∈ W , ∥ w ∥ ≤ R , and ii) the loss function is convex and L -Lipschitz, i.e., ∀ z ∈ Z , ∀ w 1 , w 2 ∈ W : | ℓ ( z, w 1 ) -ℓ ( z, w 2 ) | ≤ L ∥ w 2 -w 1 ∥ . We denote this subclass of SCO problems by C L,R .

## 3 New CMI-based bounds via stochastic projection and lossy compression

While the CMI-based bounds are known to be generally tighter than the corresponding MI ones and even tight in some settings [12, 14], they can become vacuous in some cases. This includes the Stochastic Convex Optimization (SCO) examples constructed in the recent works [43, 46], which we will discuss in

more detail in Section 4. For these (counter-)examples, it was shown in [43, 46] that CMI-type bounds do not vanish, so they fail to accurately describe the generalization error. In this section, we show that such limitations are not inherent to the CMI framework. In fact, by combining stochastic projection with lossy compression (analogously to [49], which addressed the MI case), we derive new CMI-based bounds that do not suffer from such limitations. For instance, when applied to the SCO examples of [43], we show in Section 4 that our new bounds resolve the limitations of other known CMI-based bounds as identified therein. These bounds are also shown in the appendices to apply to the analysis of the generalization error for subspace training algorithms trained with SGD or SGLD.

Our new bounds involve two main ingredients, stochastic projection and lossy compression .

Stochastic projection. Let Θ ∈ R D × d be a random matrix with entries distributed according to some joint distribution P Θ , chosen independently of ˜ S , In our approach, similar to [49], instead of considering the hypothesis W ∈ W ⊆ R D which lies in a D -dimensional space, we consider its projection Θ ⊤ W ∈ R d onto a smaller d -dimensional space, with d ≪ D .

Lossy Compression. Let ϵ ∈ R be given. An ϵ -lossy algorithm is a (possibly) stochastic map ˆ A : Z n × R D × d → ˆ W that maps a pair ( S n , Θ) to a compressed hypothesis or model ˆ W ∈ ˆ W ⊆ R d generated according to some conditional kernel P ˆ W | S n , Θ that satisfies

<!-- formula-not-decoded -->

This constraint guarantees that, when projected back onto the original hypothesis space of dimension D , the compressed model ˆ W has an average generalization error which is within at most ϵ from that of the original model W . In a sense, one works with a compressed model ˆ W which lies in a much smaller dimension space, but with the guarantee that this causes almost no increase in the generalization error. In effect, the auxiliary projected-back model Θ ˆ W substitutes the original model W .

The concept of a lossy algorithm, also referred to as a 'surrogate' or 'compressed' algorithm, was introduced in [37, 51, 52] and shown therein to be key to obtaining tighter, non-vacuous, generalization bounds. In this paper, we consider a particular lossy algorithm that involves a suitable stochastic projection followed by quantization. Specifically, we constrain the general conditional P ˆ W | S n , Θ to take the specific form P ˆ W | Θ ⊤ W , where W = A ( S n ) . Formally, one imposes the Markov chain ( S n , Θ , W ) -Θ ⊤ W -ˆ W or equivalently P ˆ W | S n , Θ ,W = P ˆ W | Θ ⊤ W . In other words, we let ˆ A ( S n , Θ) = ˜ A (Θ ⊤ A ( S n )) , where ˜ A : R d → ˆ W is defined via the Markov kernel P ˆ W | Θ ⊤ A ( S n ) .

Our generalization bounds that will follow are expressed in terms of disintegrated CMI, defined as follows. Let a super-sample ˜ S and a stochastic projection matrix Θ be given. The disintegrated CMIof an algorithm ˆ A : Z n → ˆ W is defined as

<!-- formula-not-decoded -->

where ˆ A ( ˜ S J , Θ) = ˜ A (Θ ⊤ A ( ˜ S J )) = ˆ W and I ˜ S , Θ ( ˆ A ( ˜ S J , Θ); J ) is the CMI given an instance of ˜ S and Θ , computed using the joint distribution P J ⊗ P W | ˜ S J ⊗ P ˆ W | Θ ⊤ W , with P J = Bern (1 / 2) ⊗ n .

The next theorem states our main generalization bound and is proved in Appendix E.

Theorem 1. Let a learning algorithm A : Z n → W where W ⊆ R D be given. Then, for every ϵ ∈ R , every d ∈ N , and every projected model quantization set ˆ W ⊆ R d , we have

<!-- formula-not-decoded -->

where ˆ W ∈ ˆ W , Θ ∈ R D × d , the infima are over all arbitrary choices of Markov kernel P ˆ W | Θ ⊤ W and distribution P Θ that satisfy the following distortion criterion:

<!-- formula-not-decoded -->

and the term ∆ ℓ ˆ w ( ˜ S , Θ) is given by

<!-- formula-not-decoded -->

Observe that P W | ˜ S = E P J [ P W | ˜ S J ] . Also, if ℓ ( · , · ) ∈ [0 , C ] for some non-negative constant C ∈ R + , then it is easy to see that the term ∆ ℓ ˆ w ( ˜ S , Θ) is bounded from the above as ∆ ℓ ˆ w ( ˜ S , Θ) ≤ C 2 .

The result of Theorem 1 essentially means that the generalization error of the original model is upper bounded by a term that depends on the CMI of the auxiliary model ˆ W plus an additional distortion term that quantifies the generalization gap between the auxiliary and original models. The rationale is that, although the (worst-case) CMI term still depends on the dimension d after stochastic projection, this dimension corresponds to a subspace of the original hypothesis space and can be chosen arbitrarily small in order to guarantee that the bound vanishes with n . Also, the term in left-hand-side (LHS) of equation 3 represents the average distortion (measured by the difference of induced generalization errors) between the original model and the one obtained after projecting back the auxiliary compressed model onto the original hypothesis space. The analysis of this term may seem non-easy; but as visible from the proof, it is not. This is because, defined as a difference term, its analysis does not necessitate accounting for statistical dependencies between S and W . Instead, one only needs to account for the effect of the following sources of randomness: i) the stochastic projection matrix, ii) the quantization noise, and iii) discrepancies between the empirical measure of S and the true unknown distribution µ . As shown in the proofs, the analysis of the distortion term involves the use of classic concentration inequalities. Furthermore, the construction of ˆ W allows us to consider the worst-case bound for the CMI-terms of the RHS of equation 2 without losing the order-wise optimality in certain cases.

We close this section by noting that it is well known that CMI-type bounds can be improved by application of suitable techniques such as random-subset or individual sample techniques or in order to get fast rates O (1 /n ) for small empirical risk regimes, see, e.g., [20, 53, 54]. These same techniques can be applied straightforwardly to our bound of Theorem 1 to get improved ones. For the sake of brevity, we do not elaborate on this here; and we refer the reader to the supplements where a single-datum version of Theorem 1 is provided.

## 4 Application to resolving recently raised limitations of classic CMI bounds

Prior works [43, 46] have recently reported carefully constructed counter-example learning problems and have shown that classic MI-based and CMI-based bounds fail to provide meaningful results when applied to them. In this section, we show that the careful addition of our stochastic projection along with our lossy compression resolves those issues, in the sense that the resulting new bound (our Theorem 1), which is still of CMI-type, now yields meaningful results when applied to those counter-examples. In essence, the improvement is brought up by: (i) noticing that the aforementioned negative results for standard CMI-based generalization error bounds rely heavily on that the dimension of the hypothesis space grows fast with n (over-parameterized regime), e.g., as Ω( n 4 log n ) in the considered counter-examples of [43], which calls for suitable projection onto a smaller dimension space in which this does not hold, and (ii) properly accounting for the distortion induced in the generalization error after projection back to the original high dimensional space.

First, we recall briefly the counterexamples mentioned in [43] and [46]; and, for each of them, we show how our bound of Theorem 1 applies successfully to it. Recall the definitions of a stochastic convex optimization (SCO) problem and a Convex-Lipschitz-Bounded (CLB) SCO problem as given, respectively, in Definition 1 and Definition 2.

Definition 3 ( ε -learner for SCO) . Fix ϵ &gt; 0 . For a given SCO problem ( W , Z , ℓ ) , A = {A n } n ≥ 1 is called an ε -learner algorithm with sample complexity N : R × R → N if the following holds: for every δ ∈ (0 , 1] and n ≥ N ( ε, δ ) we have that for every µ ∈ M 1 ( Z ) , where M 1 ( Z ) denotes the set of probability measures on Z , with probability at least 1 -δ over S n ∼ µ ⊗ n and internal randomness of A ,

<!-- formula-not-decoded -->

## 4.1 Counter-example of Attias et al. [2024] for CLB class

Denote by B D ( ν ) the D -dimensional ball of radius ν ∈ R + .

Definition 4 (Problem instance P ( D ) cvx ) . Let L, R ∈ R + , Z ⊆ B D (1) , and W = B D ( R ) . Define the loss function ℓ : Z × W → R as

<!-- formula-not-decoded -->

We denote this SCO problem instance as P ( D ) cvx . It is easy to see that this optimization problem belongs to the subclass C L,R of SCO problems as defined in Definition 2.

For this (counter-) example learning problem, [43] have shown that for every ε -learner there exists a data distribution for which the CMI bound of equation 1 for the optimal sample complexity, which is Θ ( ( LR ε ) 2 ) as shown in [50], scales just as Θ( LR ) . For instance, that CMI-bound on the generalization error does not decay with the size n of the training dataset!

Theorem 2 (CMI-accuracy tradeoff, [43, Theorems 4.1 and 5.2]) . Let ε 0 ∈ (0 , 1) be a universal constant. Consider the above defined P ( D ) cvx problem instance with parameters ( L, R ) . Consider any ϵ ≤ ϵ 0 and for any algorithm A = {A n } n ∈ N that ε -learns P ( D ) cvx with sample complexity N ( · , · ) . Then, the following holds: i. For every δ ≤ ε , n ≥ N ( ε, δ ) , and D = Ω ( n 4 log( n ) ) , 3 there exists a set Z ⊆ B D (1) and a data distribution µ ∈ M 1 ( Z ) , denoted as µ p ∗ , such that CMI ( µ, A n ) = Ω ( ( LR ε ) 2 ) . ii. In particular, considering the optimal sample complexity N ( ε, δ ) = Θ ( L 2 R 2 ε 2 ) , the CMI generalization bound of equation 1 equals LR √ 8 CMI ( µ, A n ) /N ( ε, δ ) = Θ( LR ) .

For this example, it was further shown [43, Corollary 5.6] that application of the individual sample technique of [55, 56] (which is traditionally used to avoid the unbounded-ness issue as instance of so called randomized-subset techniques wherein the linearity of the expectation operator is used to obtain an average bound for the loss on randomly chosen subsets of the training set rather than the loss averaged over the full training set) actually yields the very same bound order-wise; and, thus, it does not resolve the issue for this counter-example.

Furthermore, as shown in [43, Equation 1], the expectation of the LHS of equation 5 can be bounded as

<!-- formula-not-decoded -->

Thus, while the LHS of this inequality is bounded from above by ε by assumption, its right-hand side (RHS) is Θ( LR ) by Theorem 2. This means that the CMI bound of equation 1 fails to describe well the excess error of the LHS. In [43], this was even somewhat extrapolated to negatively answer the question about ' whether the excess error decomposition using CMI can accurately capture the worst-case excess error of optimal algorithms for SCOs '.

The above applies for any ε -learner of the problem instance P ( D ) cvx when Z = {± 1 / √ D } D and µ p ∗ ( z ) = ∏ D k =1 ( 1+ √ Dz k p ∗ k 2 ) , 4 for all z = ( z 1 , . . . , z D ) , where p ∗ = ( p ∗ 1 , . . . , p ∗ D ) ∈ [ -1 , 1] D .

The next theorem shows that when applied to the aforementioned counter-example, our new CMI-bound of Theorem 1 does not suffer from those shortcomings. Also, this holds true for: (i) arbitrary values of the dimension D ∈ N including n -dependent ones, (ii) arbitrary learning algorithms (including the ε -learners of P ( D ) cvx ), (iii) arbitrary choices of Z ⊆ B D (1) and (iv) arbitrary data distributions µ .

Theorem 3. For every learning algorithm A : Z n →W of the instance P ( D ) cvx defined as in Definition 4, the generalization bound of Theorem 1 yields

<!-- formula-not-decoded -->

In particular, setting N ( ε, δ ) = Θ ( L 2 R 2 ε 2 ) for ε -learner algorithms we get

<!-- formula-not-decoded -->

The proof of Theorem 3 is deferred to Appendix F.2.

Some remarks are in order. First, while when applied to the studied counter-example the CMI bound of equation 1 yields a bound of the order Θ( LR ) , i.e., one that does not decay with n , our new CMIbased bound of Theorem 1 yields one that decays with n as O ( LR/ √ n ) . Second, when specialized to the

3 The arXiv version of [43] requires a smaller increase of D with n ; namely, D = Ω ( n 2 log( n ) ) . Here, we consider values of D that are mentioned in the published PMLR version of the document, i.e., D = Ω ( n 4 log( n ) ) ; but the approach and results that will follow also hold for D = Ω ( n 2 log( n ) ) .

4 In the construction of [43], by changing n , the data distribution changes, but, for better readability, we drop such dependence in the notation.

case of ε -learner algorithms and considering the sample complexity Θ ( ( LR ε ) 2 ) , we get a bound on the generalization error of the order O ( ε ) . Using this bound, we can write

<!-- formula-not-decoded -->

Contrasting with equation 6 and noticing that if the second term of the summation of the RHS of equation 7 (optimization error) is small then both sides of equation 7 are O ( ϵ ) , it is clear that now the excess error decomposition using our new CMI-based bound can accurately capture the worst-case excess error. Third, as it can be seen from the proof, stochastic projection onto a one-dimensional space, i.e., d = 1 , is sufficient to get the result of Theorem 3. In essence, this is the main reason why, in sharp contrast with projection- and lossy-compression-free CMI-bounds, ours of Theorem 1 does not become vacuous. That is, one can reduce the effective dimension of the model for the studied example even if the original dimension D is allowed to grow with n as Ω( n 4 log( n )) as judiciously chosen in[43] for the purpose of making classic CMI-based bounds fail. Furthermore, it is worth noting that, for this problem, the projection is performed using the famous Johnson-Lindenstrauss [57] dimension reduction algorithm. Since this dimension reduction technique is 'lossy', controlling the induced distortion is critical. To do so, we introduce an additional lossy compression step by adding independent noise in the lower-dimensional space. This approach is reminiscent of lossy source coding and allows to obtain possibly tighter bounds on the quantized, projected model. Finally, we mention that for bigger class problem instances or for the memorization problem of Section 5, projection onto one-dimensional spaces may not be enough to get the desired order O ( LR/ √ n ) . In Appendix B, it will be shown that for generalized linear stochastic optimization problems, one may need d = Θ( √ n ) . Similarly, in Section 5 and Appendix C, projections with d = n 2 r -1 , r &lt; 1 and d = Θ(log n ) are used.

## 4.2 Counter-example of Attias et al. [2024] for CSL class

The question of whether classic CMI-bounds and individual-sample versions thereof may still fail if one considers more structured subclasses of SCO problems was raised (and answered positively!) in Attias et al. [43]. For convenience, we recall the following two definitions.

Definition 5 (Convex set-Strongly Convex-Lipschitz (CSL)) . An SCO problem is called CSL if i) the loss function is L -Lipschitz, and ii) the loss function is λ -strongly convex, i.e., ∀ z ∈ Z , ∀ w 1 , w 2 ∈ W : ℓ ( z, w 2 ) ≥ ℓ ( z, w 1 ) + ⟨ ∂ℓ ( z, w 1 ) , w 2 -w 1 ⟩ + λ 2 ∥ w 2 -w 1 ∥ 2 , where ∂ℓ ( z, w 1 ) is the subgradient of ℓ ( z, · ) at w 1 . We denote this subclass by C L,λ .

Definition 6 (Problem instance P ( D ) scvx ) . Let λ, R ∈ R + , Z ⊆ B D (1) , and W = B D ( R ) . Define the loss function ℓ : Z × W → R as ℓ sc ( z, w ) = -L c ⟨ w,z ⟩ + λ 2 ∥ w ∥ 2 . We denote this SCO problem as P ( D ) scvx , which belongs to C L,λ , with L = L c + λR .

Setting λ = L c = R = 1 , D = Ω( n 4 log( n )) , δ = O (1 /n 2 ) , Z = {± 1 / √ D } D and for a particular data distribution that is carefully chosen therein (not reproduced here for brevity), [43, Theorem 4.2] states that for any learning algorithm that ε -learns the problem instance P ( D ) scvx ,

<!-- formula-not-decoded -->

Moreover, the application of the individual-sample technique does not result in better decay of the bound order-wise [43, Corollary 5.7].

Noticing that (i) the loss ℓ sc ( z, w ) = -L c ⟨ w,z ⟩ + λ 2 ∥ w ∥ 2 considered in Definition 6 differs from that ℓ sc ( z, w ) = -L ⟨ w,z ⟩ of Definition 4 essentially through the added squared magnitude of the model and (ii) that addition does not alter the generalization error of a given learning algorithm, then it is easy to see that Theorem 3 also applies for the problem P ( D ) scvx at hand; and, in this case, it gives a bound of the order O (1 / √ n ) . This is stated in the next proposition, which is proved in Appendix F.3.

Proposition 1. For every learning algorithm A : Z n →W of the instance P ( D ) scvx defined as in Definition 6 the generalization bound of Theorem 1 yields

<!-- formula-not-decoded -->

In particular, choosing L c = R = λ = 1 and setting N ( ε, δ ) = c ε for some non-negative constant c ∈ R + for the ERM algorithm (which is an ε -learner - see, e.g., [50, Theorem 6]), one gets gen( µ, A ) = O ( √ ε ) .

## 4.3 Counter-example of Livni [2023]

The counter-example of [46] is the same as the problem instance of Definition 4, with the one difference that the loss function is taken to be the squared distance instead of the inner product, i.e., ℓ ( z, w ) = -L ∥ w -x ∥ 2 , for some non-negative constant L ∈ R + . Livni [46] has shown that the MI bound of [11] (which is a single-datum bound) fails and becomes vacuous when evaluated for this particular learning problem. However, since ℓ ( z, w ) = -L ∥ x ∥ 2 -L ∥ w ∥ 2 +2 L ⟨ w,x ⟩ and noticing that the squared norm terms do not alter the generalization error relative to when computed for a loss function given by only the inner-product term, it follows that Theorem 3 still applies and gives a bound of the order O (1 / √ n ) for this problem instance. In addition, for the optimal sample complexity, the bound is O ( ε ) . In essence, this means that unlike the MI bound of [11], our new CMI-based bound of Theorem 1 does not become vacuous when applied to the problem at hand.

In Appendix B, we apply the bound of Theorem 1 to a wider family of generalized linear stochastic optimization problems. In particular, we show that no counter-example could be found for which the bound of Theorem 1 does not vanish, even if one considers the bigger class of generalized linear stochastic optimization problems in place of the SCO class problems of [43].

## 5 Memorization

Loosely speaking, a learning algorithm is said to 'memorize' if by only observing its output model, an adversary can correctly guess elements of the training data among a given super-sample. For the CLB and CSL subclasses of problems studied in Section 4, Attias et al. [43] showed that there are problem instances for which, for any ε -learner algorithm, there exists a data distribution under which the learning algorithm 'memorizes' most of the training data. This is obtained by designing an adversary capable of identifying a significant fraction of the training samples.

In this section, we show that given a learning algorithm A that memorizes the training samples, one can find (via stochastic projection and lossy compression) an alternate learning algorithm ˜ A with comparable generalization error and that does not memorize the training data. 5

Definition 7 (Recall Game [43, Definition 4.3]) . Given A = {A n } n ≥ 1 , let Q : R D ×Z × M 1 ( Z ) → { 0 , 1 } be an adversary for the following game. For i ∈ [ n ] , given a fresh data point Z ′ i ∼ µ independent of ( Z i , W ) , let Z i, 1 = Z i and Z i, 0 = Z ′ i . Then, the adversary is given Z i,K i , where K i ∼ Bern (1 / 2) is independent of other random variables. The adversary declares ˆ K i ≜ Q ( W,Z i,K i , µ ) as its guess of K i .

The game consists of n rounds. At each round i ∈ [ n ] , a pair ( Z i, 0 , Z i, 1 ) is considered and the adversary makes two independent guesses: one for the sample Z i, 0 , the other for Z i, 1 .

Definition 8 (Soundness and recall [43, Definition 4.4]) . Consider the setup of Definition 7. Assume that the adversary plays the game in n rounds. For every round i ∈ [ n ] , the adversary plays two times, independently of each other, using respectively ( W,Z i, 0 , µ ) and ( W,Z i, 1 , µ ) as input. Then, for a given ξ ∈ [0 , 1] , the adversary is said to be ξ -sound if P ( ∃ i ∈ [ n ] : Q ( W,Z i, 0 , µ ) = 1) ≤ ξ . Also, the adversary certifies the recall of m samples with probability q ∈ [0 , 1] if P ( ∑ i ∈ [ n ] Q ( W,Z i, 1 , µ ) ≥ m ) ≥ q . If both conditions are met, we say that the adversary ( m,q,ξ ) -traces the data.

Clearly, the concept of ( m,q,ξ ) -tracing the data by an adversary is most interesting for values of ( m,q,ξ ) that are such that: ξ is small (i.e., the adversary makes accurate predictions), m is large and q is nonnegligible (i.e., the adversary can recall a significant part of the training data). As Lemma 1, which is stated in Appendix C.1, asserts, certain values of ( m,q,ξ ) can be attained even by a 'dummy' adversary that makes guesses without even looking at the given data sample.

For the problem instance P ( D ) cvx , Attias et al. [43] have shown that, for every ϵ -learner algorithm, there exist a distribution and an adversary that is capable of identifying a significant portion of the training data.

Theorem 4 ([43, Theorem 4.5]) . Consider the P ( D ) cvx problem instance of Definition 4 with L = R = 1 . Fix arbitrary ξ ∈ (0 , 1] and let Z = {± 1 / √ D } D . Let ε 0 ∈ (0 , 1) be a universal constant. Let ε &gt; 0 such that ε &lt; ε 0 , δ &lt; ε . Then, given any ε -learner algorithm A with sample complexity N ( ε, δ ) = Θ(log(1 /δ ) /ε 2 ) , there exist a data distribution µ p ∗ and an adversary such that for n = N ( ε, δ ) and D = Ω( n 4 log( n/ξ )) , the adversary ( Ω(1 /ε 2 ) , 1 / 3 , ξ ) -traces the data.

5 The memorization problem has also been studied in [58] via some examples in which the data distribution µ is not fixed and comes from a meta-distribution, i.e. µ ∼ P µ . Instead of using the recall game, [58] measured the amount of memorization by I ( S ; W | µ ) .

A key implication of Theorem 4 is that, for some fixed q &gt; 0 , the result holds even when ξ ∈ (0 , 1] is arbitrarily small and m = Ω( n ) (by choosing ε = O (1 / √ n ) ). In other words, for the considered class of problems P cvx with data drawn from µ p ∗ , the constructed adversary can provably trace an arbitrarily large part of the training dataset.

We show that the stochastic projection and lossy compression techniques used in the CMI framework can partially mitigate this memorization issue, in a sense that will be made precise in Theorem 8. To this end, we first establish a general result on memorization.

Theorem 5. Consider any learning algorithm A = {A n } n ≥ 1 such that CMI ( µ, A n ) = o ( n ) . Then, for any adversary for this learning algorithm that ( m,q,ξ ) -traces the data, the following holds: i) m = o ( n ) or ξ ≥ q , ii) if, for some α ∈ (0 , 1) and n 0 ∈ N ∗ , m ≥ αn for every n ≥ n 0 , then for any ϵ ∈ (0 , α ) it holds that: P ( ∑ i ∈ [ n ] Q ( W,Z i, 0 , µ ) ≥ m ′ ) ≥ ( α -ϵ ) q , where m ′ = ( ϵ 1 /q + ϵ -α ) n -o ( n ) = Ω( n ) .

Theorem 5, whose proof is provided in Appendix G.1, applies to any learning problems. In particular, it is not limited to P ( D ) cvx or the CLB subclass. The argument relies on Fano's inequality for approximate recovery [59, Theorem 2]. We construct a suitable estimator of the index set J based on the adversary's guesses, and we show that if this estimator can correctly recover a fraction c &gt; 1 2 of the membership indices J , then CMI ( µ, A n ) = Θ( n ) .

Theorem 5 i) means that if the CMI of a learning algorithm is of order o ( n ) , then any adversary that recalls a non-negligible fraction of the training dataset with some probability q ( i.e., , m = Θ( n ) ) is q -sound at best. This means that, in this regime, no adversary can do better than a dummy one that makes random guesses independently of the data (See Lemma 1 in Appendix C.1 for what is attainable by a dummy adversary). Theorem 5 ii) means that if an adversary recalls Ω( n ) training samples with some probability, then it must also incorrectly guess the membership of Ω( n ) test samples with some non-negligible probability.

Next, we use the result of Theorem 5 for P ( D ) cvx to show that while the output model W of any ε -learner algorithm must memorize a significant fraction of the data (for some distribution) as asserted in Theorem 4 the auxiliary model Θ ˆ W (which is obtained through suitable stochastic projection and lossy compression), achieves comparable generalization error without memorizing the data!

Theorem 6. Consider the P ( D ) cvx problem instance of Definition 4 with L = R = 1 . For every r &gt; 0 , every Z ⊆ B D (1) and every learning algorithm A : Z n → R D , there exists another (compressed) algorithm A ∗ : Z n → R D , defined as A ∗ ( S n ) ≜ Θ ˜ A (Θ ⊤ A ( S n )) = Θ ˆ W , where the projection matrix Θ ∈ R D × d , d = 500 r log( n ) , is distributed according to some distribution P Θ independent of ( S n , W ) , such that for any data distribution µ , the following conditions are met simultaneously:

- i) the generalization error of the auxiliary model Θ ˆ W satisfies

<!-- formula-not-decoded -->

- ii) if there exists an adversary that by having access to both Θ and ˆ W (and hence Θ ˆ W ) ( m,q,ξ ) -traces the data, then it must be that: a) m = o ( n ) or ξ ≥ q , and b) if, for some α ∈ (0 , 1) and n 0 ∈ N ∗ , m ≥ αn for every n ≥ n 0 , then for any ϵ ∈ (0 , α ) it holds that: P ( ∑ i ∈ [ n ] Q (Θ ˆ W,Z i, 0 , µ ) ≥ m ′ ) ≥ ( α -ϵ ) q , where m ′ = ( ϵ 1 /q + ϵ -α ) n -o ( n ) = Ω( n ) .

Theorem 6, proved in Appendix G.2, holds for Θ being stochastic and shared with the adversary. In essence, it asserts that for any algorithm A ( S ) = W , one can construct a suitable projected-quantized model ˆ A ( S, Θ) = ˆ W from which no adversary would be able to trace the data, for any data distribution µ . It is appealing to contrast this result with that of [43, Theorem 4.5] on the necessity of memorization. Consider the SCO instance problem with O (1) convex-Lipschitz loss defined over the ball of radius one in R D considered in [43, Theorem 4.5] and let an ε -learner algorithm A with output model W and sample complexity N ( ε, δ ) = Θ(log(1 /δ ) /ε 2 ) with D = Ω( n 4 log( n/ξ )) be given. The result of [43, Theorem 4.5] states that there exists a data distribution for which the algorithm A must memorize a big fraction of the training data. Applied to this particular instance problem, Theorem 6 asserts that if a random Θ is chosen and shared with the adversary then the auxiliary model Θ ˆ W has the following guarantees: (i) for any data distribution, no adversary can trace the data, and (ii) on average over Θ the associated generalization error is arbitrarily close to that of the original model W . At first glance, this may seem to contradict the necessity of memorization stated in [43, Theorem 4.5]. It is important to note, however,

that the auxiliary algorithm does not satisfy the conditions required in [43, Theorem 4.5]; and, so, the latter does not apply to Θ ˆ W . In particular, while [43, Theorem 4.5] requires the model to be bounded, in our construction for every w we have E ˆ W, Θ [ Θ ˆ W ] ≈ w but E ˆ W, Θ [ ∥ ∥ Θ ˆ W ∥ ∥ 2 ] increases roughly as D d (see Lemma 2 in Appendix C.4.1). As discussed after Lemma 2, this causes E ˆ W, Θ [ ∥ ∥ Θ ˆ W ∥ ∥ 2 ] to grow as Ω( n 3 ) when D = Ω( n 4 log( n/ξ )) , i.e., it becomes arbitrarily large as n increases. Intuitively, this is what prevents an adversary from guessing correctly whether a sample has (or not) been used for training, and which makes some key proof steps of Attias et al. fail when applied to the auxiliary model Θ ˆ W . These steps are discussed in detail in Appendix C.4.2.

A somewhat weaker version of Theorem 6, which is stated in Theorem 8 in Appendix C.2, holds for the projection matrix Θ being deterministic . In a sense, it provides a stronger guarantee on the generalization error of the auxiliary model, in that the closeness to the performance of the original model holds now for the given Θ and not only in average over Θ as in Theorem 6. However, this comes at the expense of the auxiliary algorithm being dependent on the data distribution. A consequence of this is that the result does not preclude the existence of other distributions for which there would exist adversaries capable of tracing the data. Moreover, in Theorem 9 in Appendix C.3, we show that a similar result holds if one considers the closeness in terms of the population risk, instead of the generalization error.

Summarizing, neither of the results of Theorem 6 and Theorem 8 contradict those of [43]. In essence, they assert that for any learning algorithm A one can find an alternate auxiliary algorithm via stochastic projection combined with lossy compression for which no adversary would be able to trace the data; and, yet, the found auxiliary algorithm has generalization error that is arbitrarily close to that of the original model. Appendix C.3 extends this closeness to the population risk.

## 6 Implications and Concluding Remarks

## Sample-compression schemes

Formally, a learning algorithm is a sample compression scheme of size k ∈ N if there exists a pair of mappings ( ϕ, ψ ) such that for all samples S = ( Z 1 , . . . , Z n ) of size n ≥ k , the map ϕ compresses the sample into a lengthk sequence which the map ψ uses to reconstruct the output of the algorithm, i.e., A ( S ) = ψ ( ϕ ( S )) . Steinke and Zakynthinou [12] establish that if an algorithm A n is a samplecompression scheme ( ϕ, ψ ) of size k , then it must be that the associated CMI is bounded from above as CMI ( A n ) ≤ k log(2 n ) . The finding of [43] that, for certain SCO problem instances, every ε -learner algorithm must have CMI that blows up with n (faster than n ) was used therein to refute the existence of such sample-compression schemes for the studied SCO problems. The results of this paper may constitute a path to obtaining such schemes when the definition is extended to involve approximate reconstruction (in terms of induced generalization error) instead of the strict A n ( · ) = ψ ( ϕ ( · )) of Littlesone and Warmuth [60].

## Fingerprinting codes and privacy attacks

In [61], the authors study the problem of designing privacy attacks on mean estimators that expose a fraction of the training data. They show that a well-designed adversary can guess membership of the training samples from the output of every algorithm that estimates mean with high precision. Our results suggest that stochastic projection and lossy compression might be useful to construct differentially private codes that prevent such fingerprinting type attacks. For instance, while noise would naturally be one constituent of the recipe in this context, its injection in a suitable smaller subspace of the summary statistics might be the key enabler of privacy guarantees in such contexts.

## Concluding remarks

In this work, we revisit recent limitations identified in conditional mutual information-based generalization bounds. By incorporating stochastic projections and lossy compression mechanisms into the CMI framework, we derive bounds that remain informative in stochastic convex optimization, thereby offering a new perspective on the results in [43, 46]. Our approach also provides a constructive resolution to the memorization phenomenon described in [43], by showing that for any algorithm and data distribution, one can construct an alternative model that does not trace training data while achieving comparable generalization.

Like prior work on information-theoretic bounds, our analysis applies to stochastic convex optimization. A natural, open question is whether and how these results can be extended to more general learning settings. Another key direction is to translate our theoretical findings into actionable design principles for learning algorithms with controlled generalization and compressibility.

## Acknowledgments

The authors thank the anonymous reviewers for their many insightful comments and suggestions. Their feedback and the ensuing discussions led to the alternative variants of Theorem 8 ( i.e. , Theorem 6 and Theorem 9), and greatly shaped some of the paper's discussions. Kimia Nadjahi would also like to thank Mahdi Haghifam for the helpful discussions.

## References

- [1] VNVapnik and A Ya Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability and its Applications , 16(2):264, 1971.
- [2] Peter L Bartlett, Olivier Bousquet, and Shahar Mendelson. Local rademacher complexities. Annals of Statistics , pages 1497-1537, 2005.
- [3] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability, stability and uniform convergence. The Journal of Machine Learning Research , 11:2635-2670, 2010.
- [4] Umut S ¸ims ¸ekli, Ozan Sener, George Deligiannidis, and Murat A Erdogdu. Hausdorff dimension, heavy tails, and generalization in neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 5138-5151. Curran Associates, Inc., 2020.
- [5] Tolga Birdal, Aaron Lou, Leonidas Guibas, and Umut S ¸ims ¸ekli. Intrinsic dimension, persistent homology and generalization in neural networks. In Advances in Neural Information Processing Systems (NeurIPS) , 2021.
- [6] Liam Hodgkinson, Umut Simsekli, Rajiv Khanna, and Michael Mahoney. Generalization bounds using lower tail exponents in stochastic optimizers. In International Conference on Machine Learning , pages 8774-8795. PMLR, 2022.
- [7] Soon Hoe Lim, Yijun Wan, and Umut S ¸ims ¸ekli. Chaotic regularization and heavy-tailed limits for deterministic gradient descent. arXiv preprint arXiv:2205.11361 , 2022.
- [8] Yijun Wan, Melih Barsbey, Abdellatif Zaidi, and Umut S ¸ ims ¸ekli. Implicit compressibility of overparametrized neural networks trained with heavy-tailed SGD. In Proceedings of the 41st International Conference on Machine Learning , pages 49845-49866, 2024.
- [9] Daniel Russo and James Zou. Controlling bias in adaptive data analysis using information theory. In Arthur Gretton and Christian C. Robert, editors, Proceedings of the 19th International Conference on Artificial Intelligence and Statistics , volume 51 of Proceedings of Machine Learning Research , pages 1232-1240, Cadiz, Spain, 09-11 May 2016. PMLR.
- [10] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. Advances in Neural Information Processing Systems , 30, 2017.
- [11] Yuheng Bu, Shaofeng Zou, and Venugopal V. Veeravalli. Tightening mutual information-based bounds on generalization error. IEEE Journal on Selected Areas in Information Theory , 1(1): 121-130, May 2020. ISSN 2641-8770.
- [12] Thomas Steinke and Lydia Zakynthinou. Reasoning about generalization via conditional mutual information. In Jacob Abernethy and Shivani Agarwal, editors, Proceedings of Thirty Third Conference on Learning Theory , volume 125 of Proceedings of Machine Learning Research , pages 3437-3452. PMLR, 09-12 Jul 2020.
- [13] Amedeo Roberto Esposito, Michael Gastpar, and Ibrahim Issa. Generalization error bounds via R´ enyi-, f -divergences and maximal leakage, 2020.
- [14] Mahdi Haghifam, Gintare Karolina Dziugaite, Shay Moran, and Daniel M. Roy. Towards a unified information-theoretic framework for generalization. In Thirty-Fifth Conference on Neural Information Processing Systems , 2021.
- [15] Gergely Neu, Gintare Karolina Dziugaite, Mahdi Haghifam, and Daniel M. Roy. Informationtheoretic generalization bounds for stochastic gradient descent, 2021.
- [16] Gholamali Aminian, Yuheng Bu, Laura Toni, Miguel Rodrigues, and Gregory Wornell. An exact characterization of the generalization error for the gibbs algorithm. Advances in Neural Information Processing Systems , 34:8106-8118, 2021.

- [17] Ruida Zhou, Chao Tian, and Tie Liu. Individually conditional individual mutual information bound on generalization error. IEEE Transactions on Information Theory , 68(5):3304-3316, 2022. doi: 10.1109/TIT.2022.3144615.
- [18] G´ abor Lugosi and Gergely Neu. Generalization bounds via convex analysis. In Conference on Learning Theory , pages 3524-3546. PMLR, 2022.
- [19] Saeed Masiha, Amin Gohari, and Mohammad Hossein Yassaee. f-divergences and their applications in lossy compression and bounding generalization error. IEEE Transactions on Information Theory , 2023.
- [20] Hrayr Harutyunyan, Maxim Raginsky, Greg Ver Steeg, and Aram Galstyan. Information-theoretic generalization bounds for black-box learning algorithms. Advances in Neural Information Processing Systems , 34, 2021.
- [21] Fredrik Hellstr¨ om and Giuseppe Durisi. A new family of generalization bounds using samplewise evaluated cmi. Advances in Neural Information Processing Systems , 35:10108-10121, 2022.
- [22] Milad Sefidgaran, Romain Chor, and Abdellatif Zaidi. Rate-distortion theoretic bounds on generalization error for distributed learning. Advances in Neural Information Processing Systems , 35: 19687-19702, 2022.
- [23] Milad Sefidgaran and Abdellatif Zaidi. Data-dependent generalization bounds via variable-size compressibility. IEEE Transactions on Information Theory , 2024.
- [24] David A McAllester. Some PAC-Bayesian theorems. In Proceedings of the eleventh annual conference on Computational learning theory , pages 230-234, 1998.
- [25] David A McAllester. PAC-Bayesian model averaging. In Proceedings of the twelfth annual conference on Computational learning theory , pages 164-170, 1999.
- [26] Matthias Seeger. PAC-Bayesian generalisation error bounds for gaussian process classification. Journal of machine learning research , 3(Oct):233-269, 2002.
- [27] John Langford and Rich Caruana. (not) bounding the true error. Advances in Neural Information Processing Systems , 14, 2001.
- [28] Olivier Catoni. A PAC-Bayesian approach to adaptive classification. preprint , 840, 2003.
- [29] Andreas Maurer. A note on the pac bayesian theorem. arXiv preprint cs/0411099 , 2004.
- [30] Pascal Germain, Alexandre Lacasse, Franc ¸ois Laviolette, and Mario Marchand. PAC-Bayesian learning of linear classifiers. In Proceedings of the 26th Annual International Conference on Machine Learning , pages 353-360, 2009.
- [31] Ilya O Tolstikhin and Yevgeny Seldin. PAC-Bayes-empirical-bernstein inequality. Advances in Neural Information Processing Systems , 26, 2013.
- [32] Luc B´ egin, Pascal Germain, Franc ¸ois Laviolette, and Jean-Francis Roy. PAC-Bayesian bounds based on the r´ enyi divergence. In Artificial Intelligence and Statistics , pages 435-444. PMLR, 2016.
- [33] Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin. A strongly quasiconvex PAC-Bayesian bound. In International Conference on Algorithmic Learning Theory , pages 466-492. PMLR, 2017.
- [34] Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv preprint arXiv:1703.11008 , 2017.
- [35] Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks, 2018.
- [36] Omar Rivasplata, Ilja Kuzborskij, Csaba Szepesv´ ari, and John Shawe-Taylor. PAC-Bayes analysis beyond the usual bounds. Advances in Neural Information Processing Systems , 33:16833-16845, 2020.
- [37] Jeffrey Negrea, Gintare Karolina Dziugaite, and Daniel Roy. In defense of uniform convergence: Generalization via derandomization with an application to interpolating predictors. In International Conference on Machine Learning , pages 7263-7272. PMLR, 2020.
- [38] Jeffrey Negrea, Mahdi Haghifam, Gintare Karolina Dziugaite, Ashish Khisti, and Daniel M. Roy. Information-theoretic generalization bounds for SGLD via data-dependent estimates, 2020.

- [39] Paul Viallard, Pascal Germain, Amaury Habrard, and Emilie Morvant. A general framework for the disintegration of PAC-Bayesian bounds. arXiv preprint arXiv:2102.08649 , 2021.
- [40] Raef Bassily, Shay Moran, Ido Nachum, Jonathan Shafer, and Amir Yehudayoff. Learners that use little information. In Algorithmic Learning Theory , pages 25-55. PMLR, 2018.
- [41] Ido Nachum, Jonathan Shafer, and Amir Yehudayoff. A direct sum result for the information complexity of learning. In Conference On Learning Theory , pages 1547-1568. PMLR, 2018.
- [42] Fredrik Hellstr¨ om, Giuseppe Durisi, Benjamin Guedj, Maxim Raginsky, et al. Generalization bounds: Perspectives from information theory and PAC-Bayes. Foundations and Trends® in Machine Learning , 18(1):1-223, 2025.
- [43] Idan Attias, Gintare Karolina Dziugaite, Mahdi Haghifam, Roi Livni, and Daniel M. Roy. Information complexity of stochastic convex optimization: Applications to generalization, memorization, and tracing. In Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 2035-2068. PMLR, 21-27 Jul 2024.
- [44] Hassan Hafez-Kolahi, Zeinab Golgooni, Shohreh Kasaei, and Mahdieh Soleymani. Conditioning and processing: Techniques to improve information-theoretic generalization bounds. Advances in Neural Information Processing Systems , 33:16457-16467, 2020.
- [45] Ziqiao Wang and Yongyi Mao. Tighter information-theoretic generalization bounds from supersamples. In Proceedings of the 40th International Conference on Machine Learning , pages 36111-36137, 2023.
- [46] Roi Livni. Information theoretic lower bounds for information theoretic upper bounds. In Proceedings of the 37th International Conference on Neural Information Processing Systems , NIPS '23, Red Hook, NY, USA, 2023. Curran Associates Inc.
- [47] Mahdi Haghifam, Borja Rodr´ ıguez-G´ alvez, Ragnar Thobaben, Mikael Skoglund, Daniel M Roy, and Gintare Karolina Dziugaite. Limitations of information-theoretic generalization bounds for gradient descent methods in stochastic convex optimization. In International Conference on Algorithmic Learning Theory , pages 663-706. PMLR, 2023.
- [48] Ziqiao Wang and Yongyi Mao. Sample-conditioned hypothesis stability sharpens informationtheoretic generalization bounds. Advances in Neural Information Processing Systems , 36:4951349541, 2023.
- [49] Kimia Nadjahi, Kristjan Greenewald, Rickard Br¨ uel Gabrielsson, and Justin Solomon. Slicing mutual information generalization bounds for neural networks. In International Conference on Machine Learning , pages 37213-37236. PMLR, 2024.
- [50] Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Stochastic convex optimization. In COLT , volume 2, number 4, page 5, 2009.
- [51] Yuheng Bu, Weihao Gao, Shaofeng Zou, and Venugopal Veeravalli. Information-theoretic understanding of population risk improvement with model compression. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 3300-3307, 2020.
- [52] Milad Sefidgaran, Amin Gohari, Gael Richard, and Umut Simsekli. Rate-distortion theoretic generalization bounds for stochastic learning algorithms. In Conference on Learning Theory , pages 4416-4463. PMLR, 2022.
- [53] Peter Grunwald, Thomas Steinke, and Lydia Zakynthinou. PAC-Bayes, mac-bayes and conditional mutual information: Fast rate bounds that handle general vc classes. In Conference on Learning Theory , pages 2217-2247. PMLR, 2021.
- [54] Milad Sefidgaran, Abdellatif Zaidi, and Piotr Krasnowski. Minimum description length and generalization guarantees for representation learning. In Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) , 2023.
- [55] Borja Rodr´ ıguez-G´ alvez, Germ´ an Bassi, Ragnar Thobaben, and Mikael Skoglund. On random subset generalization error bounds and the stochastic gradient langevin dynamics algorithm. In 2020 IEEE Information Theory Workshop (ITW) , pages 1-5. IEEE, 2021.
- [56] Ruida Zhou, Chao Tian, and Tie Liu. Individually conditional individual mutual information bound on generalization error. IEEE Transactions on Information Theory , 68(5):3304-3316, 2022.
- [57] William B Johnson and Joram Lindenstrauss. Extensions of lipschitz mappings into a hilbert space 26. Contemporary mathematics , 26:28, 1984.

- [58] Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar. When is memorization of irrelevant training data necessary for high-accuracy learning? In Proceedings of the 53rd annual ACM SIGACT symposium on theory of computing , pages 123-132, 2021.
- [59] Jonathan Scarlett and Volkan Cevher. An introductory guide to fano's inequality with applications in statistical estimation. arXiv preprint arXiv:1901.00555 , 2019.
- [60] Nick Littlestone and Manfred Warmuth. Relating data compression and learnability. Citeseer , 1986.
- [61] Cynthia Dwork, Adam Smith, Thomas Steinke, Jonathan Ullman, and Salil Vadhan. Robust traceability from trace amounts. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science , pages 650-669, 2015. doi: 10.1109/FOCS.2015.46.
- [62] Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: isoperimetry and processes . Springer Science &amp; Business Media, 2013.
- [63] Ankit Pensia, Varun Jog, and Po-Ling Loh. Generalization error bounds for noisy, iterative algorithms. 2018 IEEE International Symposium on Information Theory (ISIT) , pages 546-550, 2018.
- [64] Mahdi Haghifam, Jeffrey Negrea, Ashish Khisti, Daniel M Roy, and Gintare Karolina Dziugaite. Sharpened generalization bounds based on conditional mutual information and an application to noisy, iterative algorithms. Advances in Neural Information Processing Systems , 33:9925-9935, 2020.
- [65] Borja Rodr´ ıguez G´ alvez, Germ´ an Bassi, Ragnar Thobaben, and Mikael Skoglund. On random subset generalization error bounds and the stochastic gradient langevin dynamics algorithm. CoRR , abs/2010.10994, 2020.
- [66] Hao Wang, Yizhe Huang, Rui Gao, and Flavio Calmon. Analyzing the generalization capability of SGLDusing properties of gaussian channels. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , volume 34, pages 24222-24234. Curran Associates, Inc., 2021.
- [67] Hao Wang, Rui Gao, and Flavio P Calmon. Generalization bounds for noisy iterative algorithms using properties of additive noise channels. Journal of machine learning research , 24(26):1-43, 2023.
- [68] Sejun Park, Umut Simsekli, and Murat A Erdogdu. Generalization bounds for stochastic gradient descent via localized ε -covers. Advances in Neural Information Processing Systems , 35:2790-2802, 2022.
- [69] Aymeric Dieuleveut, Alain Durmus, and Francis Bach. Bridging the gap between constant step size stochastic gradient descent and Markov chains, 2018.
- [70] Leo Kozachkov, Patrick M Wensing, and Jean-Jacques Slotine. Generalization in supervised learning through riemannian contraction. arXiv preprint arXiv:2201.06656 , 2022.
- [71] Allan Grønlund, Lior Kamma, and Kasper Green Larsen. Near-tight margin-based generalization bounds for support vector machines. In Hal Daum´ e III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 3779-3788. PMLR, 13-18 Jul 2020.
- [72] Jean Gallier. Discrete mathematics . Springer Science &amp; Business Media, 2011.
- [73] Robert G Gallager. Information theory and reliable communication , volume 588. Springer, 1968.
- [74] Ziqiao Wang and Yongyi Mao. On the generalization of models trained with SGD: Informationtheoretic bounds and implications, 2021.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We proved several theoretical results showing the effectiveness of the projection and quantization technique and discussed it in detail. In particular, we showed how this can be used to resolve the recently raised concerns on the information-theoretic bounds.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We clearly stated the problem instances and classes for which we demonstrated that this approach results in good generalization bounds. We also stated all assumptions needed for each result.

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

Justification: In this paper, we stated all results rigorously, along with the assumptions used and detailed proofs in the supplements. The proofs are rigorous with enough details provided for the reader to follow.

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

Justification: Our work is a theoretical paper with rigorously proven claims, and does not involve any experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case

of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: Our work does not involve any experiment.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: Our work does not involve any experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our work does not involve any experiment.

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

Justification:Our work does not involve any experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our work is a theoretical paper on learning theory and does not violate any code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is a theoretical paper on learning theory and does not have any direct negative societal impact.

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

Justification: Our work does not involve any experiment.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Our work does not involve any experiment.

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

Justification: Our work does not involve any experiment or any new asset.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work is a theoretical paper on learning theory.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our work does not involve crowd sourcing nor any research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: We have not used LLMs for this work.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendices

The appendices are organized as follows:

- In Appendix A, we present some extensions of Theorem 1, that are used in the subsequent sections.
- The results of Section 4 have been extended to a wider family of generalized linear stochastic optimization problems in Appendix B.
- Further results on memorization are presented in Appendix C. In particular
- -In Appendix C.1, we discuss what values of ( m,q,ξ ) can be achieved by a 'dummy adversary'.
- -In Appendix C.2, we consider the case where the projection matrix Θ is fixed and shared with the adversary.
- -In Appendix C.3, we discuss how to provide guarantees on the closeness in terms of the population risk between the projected-quantized model to the original model.
- -In Appendix C.4, we provide technical lemmas used in the main text on reconciliation of our results with those of [43].
- The generalization error of subspace training algorithms is investigated in Appendix D. In particular, in Appendix D.1, we develop generalization bounds for the case where iterative optimization algorithms such as SGD and SGLD are used for the optimization of the subspace training algorithms.
- The proof of Theorem 1 is presented in Appendix E.
- In Appendix F, we present the proofs of the results presented in Section 4 and Appendix B regarding the applications of Theorem 1 to resolving recently raised limitations of classic CMI bounds. In particular,
- -a general Johnson-Lindenstrauss projection scheme JL ( d, c w , ν ) is introduced in Appendix F.1, which is used in the following subsections, with different choices of ( d, c w , ν ) ,
- -Theorem 3 is proved in Appendix F.2,
- -Proposition 1 is proved in Appendix F.3,
- -Theorem 7 is proved in Appendix F.4,
- -and Lemma 4 is proved in Appendix F.5.
- Appendix G contains the proofs of the results in Section 5 and Appendix C, about the memorization. More precisely,
- -Theorem 5 is proved in Appendix G.1,
- -Theorem 6 is proved in Appendix G.2,
- -Lemma 1 is proved in Appendix G.3,
- -Theorem 8 is proved in Appendix G.4,
- -Theorem 9 is proved in Appendix G.5,
- -Lemma 2 is proved in Appendix G.6,
- -and Lemma 5 is proved in Appendix G.7.
- Lastly, Appendix H contains the proofs of the results of Appendix D on the generalization error of subspace training algorithms when trained using SGD or SGLD. More precisely,
- -Lemma 3 is proved in Appendix H.1,
- -Theorem 10 is proved in Appendix H.2,
- -Theorem 13 is proved in Appendix H.3,
- -and Lemma 6 is proved Appendix H.4.

## A Extensions of Theorem 1

As mentioned in Section 3, Theorem 1 can be improved in several ways, similar to those proposed in [20, 53, 54]. Here, we state only the single-datum version of Theorem 1, which is used in Appendix D, followed by a remark about extending Theorem 1 and its corollary to more general lossy compression algorithms. Denote

<!-- formula-not-decoded -->

Corollary 1. Consider the setup of Theorem 1. Then,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where the infima are over P ˆ W | Θ ⊤ W and P Θ that satisfy the distortion criterion

<!-- formula-not-decoded -->

and where

<!-- formula-not-decoded -->

To derive inequality 9, first note that by equation 11, it is sufficient to show that

<!-- formula-not-decoded -->

Next, using the linearity of the expectation, we can write

<!-- formula-not-decoded -->

Then applying Theorem 1 for each of the terms E Z i , ˆ W [gen( { Z i } , ˆ W )] yields equation 9.

The inequality 10 can be achieved similarly, by considering

<!-- formula-not-decoded -->

instead of equation 12.

The results of Theorem 1 and, consequently, Corollary 1, are valid for a broader class of learning algorithms, A , and lossy compression algorithms, ˆ A , as discussed in the remark below and shown in the proof of Theorem 1 in Appendix E.

Remark 1. As shown in Appendix E, the bounds of Theorem 1 and consequently Corollary 1 hold if the learning algorithm A is aware of the projection matrix Θ , i.e., if A : Z n × R D × d → W takes both the dataset S and the projection matrix Θ as input in order to learn the model W . Moreover, the results of Theorem 1 and Corollary 1 are valid if the quantization step can also depend on S , Θ and A ( S, Θ) . In this general case, ˆ W = ˆ A ( S, Θ) = ˜ A (Θ , S, A ( S, Θ)) = ˆ W . This setting trivially includes the case in which A : Z n → W and the quantization depends only on Θ ⊤ A ( S, Θ) . For the ease of the exposition, we found it better not to state the result in its most general form.

## B Generalized linear stochastic optimization problems

In this section, we show that our bound of Theorem 1 can be applied successfully to get useful bounds on the generalization error of a family of generalized linear stochastic optimization problems that is wider than the ones considered previously in related prior art.

Definition 9 (Generalized linear stochastic optimization) . Let L, B, R ∈ R + and W = B D ( R ) . Define the loss function ℓ gl : Z × W → R as

<!-- formula-not-decoded -->

where g : R ×Z → R is L -Lipschitz with respect to the first argument, ϕ : Z → B D ( B ) and r : W → R is some arbitrary function. Denote this problem as P ( D ) glso .

This class of problems is larger than the one considered in [50]. For instance, while the results of [50] require the L -Lipschitz function g ( · , · ) and the function r ( · ) to be both convex to hold, our next theorem applies to arbitrary L -Lipschitz functions g ( · , · ) and arbitrary functions r ( · ) .

Theorem 7. For every learning algorithm A : Z n → W of the instance problem P ( D ) glso defined in Definition 9, the generalization bound of Theorem 1 yields

<!-- formula-not-decoded -->

The proof, stated in Appendix F.4, is based on Theorem 1. In order to find a proper stochastic projection and quantization, we use the Johnson-Lindenstrauss (JL) dimensional reduction transformation in a space of dimension d . Then, we apply lossy compression to the projected model. Thanks to the combined projection-quantization, the disintegrated CMI can be bounded easily in the d -dimensional space. However, there are two main caveats to using the JL Lemma directly. First, one needs to bound the term ∆ ℓ ˆ w ( ˜ S , Θ) (see equation 4). This is particularly difficult since the JL Lemma does not guarantee distance preservation in the original space of dimension D after projecting back the quantized model. Second, bounding the distortion term is less easy than in Theorem 3, since using the Lipschitz property requires bounding the absolute value of the difference between inner products of the original and projected-quantized models. In essence, this is the reason why, by opposition to JL transformation for which it suffices to take d = log( n ) , here one needs a higher-dimensional projection space comparatively, with d = √ n .

Theorem 7 shows that no counter-example could be found for which the bound of Theorem 1 does not vanish, even if one considers the bigger class of generalized linear stochastic optimization problems of Definition 9 in place of the SCO class problems of [43]. The convergence rate O (1 / 4 √ n ) of Theorem 7 is, however, not optimal. A better rate, O (1 / √ n ) , seems to be achievable using Rademacher analysis and Talagrand's contraction lemma [62]. Using a more refined analysis, the same rate might be possible to achieve using our Theorem 1. More precisely, in the part of the current proof of Theorem 7 that analyses the distortion term, we do not account for the discrepancy between the empirical measure of S and the true distribution µ ; and, instead, we consider a worst-case scenario. A finer analysis that takes such discrepancy into account should lead to a sharper expected concentration bound for the distortion term, and, so, a better rate.

## C Further results on memorization

In this section, we provide further results on memorization. In Appendix C.1, we show that even a 'dummy' adversary can trace the data for some values of ( m,q,ξ ) . In Appendix C.2, we study the case where the projection matrix Θ is deterministic. In Appendix C.3, we provide another variant of Theorem 8, in which we can guarantee the closeness of the projected-quantized model to the original model in terms of population risk (instead of the generalization error considered in Theorem 8). Finally, in Appendix C.4, we present some technical lemmas used in the discussions of Section 5 on the relation of our results with those established in [43].

## C.1 Dummy adversary

In this section, we show that certain values of ( m,q,ξ ) are attainable by a 'dummy' adversary who makes guesses without even looking at the given data sample.

Lemma 1. Given a learning algorithm A n : Z →W , there exists an adversary that ( m,q,ξ ) -traces the data for some m ∈ [0 , n ] and q, ξ ∈ [0 , 1] if one of the following conditions holds: i) ξ ≥ q , or ii) there

<!-- formula-not-decoded -->

n √ .

This lemma, proved in Appendix G.3, implies in particular that even a dummy adversary can ( m,q,ξ ) -trace the data in several cases: when ξ is small, when q is large, or when ξ is small and q is large, provided that m = o ( n ) .

## C.2 Deterministic projection

In this section, we show that in Theorem 6, one can allow Θ to be deterministic. However, this comes at the cost of being specific to a given data distribution.

Theorem 8. Consider the P ( D ) cvx problem instance of Definition 4 with L = R = 1 . For every r &lt; 1 , every Z ⊆ B D (1) , every data distribution µ , and every learning algorithm A , there exist a projection matrix Θ ∈ R D × d with d = ⌈ n 2 r -1 ⌉ , a Markov Kernel P ˆ W | Θ ⊤ W and a compression algorithm A ∗ Θ : Z n → R d , defined as A ∗ Θ ( S n ) ≜ ˜ A (Θ ⊤ A ( S n )) = ˆ W , such that the following conditions are met simultaneously:

- i) the generalization error of the auxiliary model Θ ˆ W satisfies

<!-- formula-not-decoded -->

where the expectation is taken over ( S n , W, ˆ W ) ∼ P S n ,W P ˆ W | Θ ⊤ W .

- ii) if there exists an adversary that by having access to both Θ and ˆ W (and hence Θ ˆ W ) ( m,q,ξ ) -traces the data, then it must be that: a) m = o ( n ) or ξ ≥ q , and b) if, for some α ∈ (0 , 1) and n 0 ∈ N ∗ , m ≥ αn for every n ≥ n 0 , then for any ϵ ∈ (0 , α ) it holds that: P ( ∑ i ∈ [ n ] Q (Θ ˆ W,Z i, 0 , µ ) ≥ m ′ ) ≥ ( α -ϵ ) q , where m ′ = ( ϵ 1 /q + ϵ -α ) n -o ( n ) = Ω( n ) .

As shown in the proof in Appendix G.4, the constraint on the difference generalization error can be replaced with one with a faster decay with n , namely

<!-- formula-not-decoded -->

for some r ∈ [ R ] and d = 500 r log( n ) . Also, if n = N ( ε, δ ) , then m,m ′ = Ω ( 1 /ε 2 ) , which means that any adversary who ( m,q,ξ ) -traces the training data is deemed to misclassify any arbitrary big part of the test samples.

For the proof of Theorem 8, we first apply the projection-quantization approach of Theorem 3. Then, for a proper Θ that satisfies the distortion criterion of equation 13 and for which the CMI is o ( n ) we apply Theorem 8. Note two important differences with Theorem 3. First, because one now deals with absolute value of the average difference of generalization errors one also needs to lower bound the average distortion. Also, for r &gt; 1 / 2 a faster convergence rate of O ( n -r ) is required. This renders the analysis trickier and requires projection on a space of dimension n 2 r -1 .

## C.3 Guarantees on the population risk

In this section, we demonstrate that the closeness guarantee of the projected-quantized model and the original model can also be provided in terms of population risk.

Theorem 9. Consider the P ( D ) cvx problem instance of Definition 4 with L = R = 1 . For every r &lt; 1 / 2 , every Z ⊆ B D (1) , every data distribution µ , and every learning algorithm A , there exist a projection matrix Θ ∈ R D × d with d = ⌈ n 2 r ⌉ , a Markov Kernel P ˆ W | Θ ⊤ W and a compression algorithm A ∗ Θ : Z n → R d , defined as A ∗ Θ ( S n ) ≜ ˜ A (Θ ⊤ A ( S n )) = ˆ W , such that the following conditions are met simultaneously:

- i) the generalization error of the auxiliary model Θ ˆ W

<!-- formula-not-decoded -->

where the expectation is taken over ( S n , W, W ) ∼ P S ,W P ˆ ⊤

- satisfies ˆ n W | Θ W .
- ii) if there exists an adversary that by having access to both Θ and ˆ W (and hence Θ ˆ W ) ( m,q,ξ ) -traces the data, then it must be that: a) Either m = o ( n ) or ξ ≥ q , and b) if m = Ω( n ) then there exists m ′ = Ω( n ) and q ′ ∈ (0 , 1] such that for sufficiently large n , it holds that P ( ∑ i ∈ [ n ] Q (Θ ˆ W,Z i, 0 , µ ) ≥ m ′ ) ≥ q ′ .

This result is proved in Appendix G.5. Furthermore, similarly to Theorem 8, the constraint on the difference of population risks can be replaced with one with a faster decay with n , namely

<!-- formula-not-decoded -->

for some r ∈ [ R ] and d = 500 r log( n ) .

## C.4 Reconciliation with results of Attias et al. 2024

In this section, we provide the technical lemma showing that the norm two of the projected-quantized model, used in our results, is unbounded. Furthermore, we discuss in detail the steps of the proofs in [43] where this bounded assumption is needed.

## C.4.1 Uboundedness of the norm two of the projected-quantized model

In this section, for the projected-quantized algorithm Θ ˆ W , used in Theorem 8 and Theorem 6, we show that E ˆ W, Θ [ ∥ ∥ ∥ Θ ˆ W ∥ ∥ ∥ 2 ] blows-up with n when D/d grows with n . This lemma is proved in Appendix G.6.

Lemma 2. Consider the JL ( d, c w , ν ) transformation described in Appendix F.1, with some d ∈ N + , c w ∈ [ 1 , √ 5 / 4 ) , and ν ∈ (0 , 1] . Then, for every w ∈ W ,

<!-- formula-not-decoded -->

Consider ∥ w ∥ = 1 and let D = n 4 log( n/ξ ) as considered in [43, Theorem 4.5]. We note that the notation d used in [43] corresponds to the notation D in this paper.

Then, considering the constructions used for Theorem 8 and Theorem 6, we have c w = 1 . 1 and ν = 0 . 4 . Moreover, d is chosen either as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or

Using Lemma 2 with these choices give

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and respectively. Hence, in both cases E Θ ,V ν [ ∥ ∥ ∥ Θ ˆ W ∥ ∥ ∥ 2 ] grows at least as fast as Ω( n 3 ) .

## C.4.2 Details of needed boundedness assumption in Attias et al.

As discussed before, [43, Theorem 4.1] and [43, Theorem 4.5] require the model to be bounded. However, as shown in the previous section, this assumption does not hold for the projected-quantized algorithm Θ ˆ W when D/d and d grow with n . In this section, we discuss precisely where the bounded model assumption is necessary in the proofs of the impossibility results of [43].

- Proof of [43, Theorem 4.1] and recall analysis in the proof of [43, Theorem 4.5], in part, relies on an established upper bound Ω(1 /ε 2 ) on the term E [ |I| ] , where the set I is the subset of columns of supersample such that one of the samples has a large correlation with the output of the algorithm and the other one has small correlation with the output of the algorithm.

- -To establish this upper bound, in the last inequality of Page 19 of [43], it is assumed that the norm of the model is bounded. Now, when working with the model Θ ˆ W , the right-hand side of this inequality needs to be replaced by the D/d -dependent quantity 8 ε 4 n 2 D d + 2 ε 2 = Ω( n 5 ) when D = Ω( n 4 ) and d = o ( n ) . This has to be contrasted with the actual bound 8 ε 4 n 2 +2 ε 2 when the bounded model's norm assumption holds. Thus, one important issue is that, this quantity now being non-negligible, the LHS of (9) can no longer be lowerbounded by the RHS of the inequality (9).
- -Another step, used for establishing the upper bound on E [ |I| ] , is the step that upper bounds P ( E c ) = O (1 /n 2 ) , for the event E defined on top of Page 19 of [43]. In this case again, in the set of equations before equation (12), it is assumed that the norm of the model is bounded to derive ∥ A ˆ θ 2 ∥ ≤ 144 2 ε 4 . However, since norm two of Θ ˆ W is Ω( n 3 ) , then these steps are ot valid and hence the analysis does not give P ( E c ) = O (1 /n 2 ) anymore.
- Another proof step of [43, Theorem 4.1], used also in the soundness analysis in the proof of [43, Theorem 4.5], relies on upper bounds for the error event G c , defined on [43, Page 18] as the probability that the correlation between the model output and the held-out samples is significant. These upper bounds, [43, Equations 11] in the proof of [43, Theorem 4.1] and also on [43, 29] in the soundness analysis in the proof of [43, Theorem 4.5], are based on an application of [43, Lemma B.8] and by assuming that the norm two of the model is bounded by 1 . These steps again fail if the norm two of the model grows as Ω( D/d ) = Ω( n 3 ) .

## D Random subspace training algorithms

The generalization bounds of Theorem 1 and Corollary 1 apply to any arbitrary learning algorithm. In this section, we show how this bound can be applied to random subspace training algorithms. Then, we consider the case where they are trained using an iterative optimization algorithm.

Let St( d, D ) = { Θ ∈ R D × d : Θ ⊤ Θ = I d } be the Stiefel manifold, equipped with the uniform distribution P Θ . Moreover, for a given Θ ∈ R D × d , let W Θ ,d ≜ { w ∈ R D : ∃ w ′ ∈ R d s . t . w = Θ w ′ } . Random subspace training algorithms first randomly generate an instance of Θ according to P Θ , which is kept frozen during training. A random subspace training algorithm A ( d ) : Z n × R D × d → W Θ ,d is a learning algorithm that takes the dataset S and the projection matrix Θ as input, and chooses a model W ∈ W Θ ,d , by choosing a W ′ ∈ R d .

In other words, A ( d ) ( S, Θ) = Θ W ′ , or alternatively, since Θ ⊤ Θ = I d , W ′ = Θ ⊤ A ( d ) ( S, Θ) . Hence, using Corollary 1 and by noting Remark 1, we can obtain the following result.

Corollary 2. Consider a random subspace training algorithm and a loss function ℓ : Z × R D → [0 , C ] . Then, for any ϵ ∈ R and the quantization set ˆ W ⊆ R d , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where ˆ W ∈ ˆ W and the infimum are over all Markov kernels P ˆ W | W ′ ,S, Θ that satisfies the following distortion criterion:

<!-- formula-not-decoded -->

This bound is used in the following subsection, when SGD or SGLD are used for random subspace training. Note that the above bound includes the case of ˆ W = W ′ and ϵ = 0 , which results in the lossless bounds of gen( µ, A ( d ) ) ≤ E P ˜ S P Θ [ C n ∑ i ∈ [ n ] √ 2 CMI Θ i ( ˜ S , W ′ ) ] and gen( µ, A ( d ) ) ≤ E P ˜ S P Θ P J -i [ C n ∑ i ∈ [ n ] √ 2 CMI Θ i, J -i ( ˜ S , W ′ ) ] .

The results presented in the next section are extensions and improvements in some aspects, upon previous work on bounding the generalization error of SGLD without projection [38, 63-67].

## D.1 Generalization bounds for SGD and SGLD Algorithms

In this section, we consider subspace training algorithms that are trained using an iterative optimization algorithm such as mini-batch Stochastic Gradient Descent (SGD) or Stochastic Gradient Langevin dynamics (SGLD).

Let b ∈ N bet the mini-batch size, and let

<!-- formula-not-decoded -->

be the sample indices chosen at time t ∈ [ T ] , i.e., given ˜ S ∈ Z n × 2 and J = ( J 1 , . . . , J n ) , the chosen indices at time t are ˜ S V t , J ≜ ˜ S V t , J Vt ≜ { Z i t, 1 ,J i t, 1 , . . . , Z i t,b ,J i t,b } . Furthermore, denote

<!-- formula-not-decoded -->

We use also the notation V ≜ ( V 1 , . . . , V T ) and recall that J -i ≜ J [ n ] \{ i } .

The considered noisy iterative optimization algorithm consists of the following steps:

- (Initialization) Sample Θ ∈ R D × d and set the initial model's parameters to W 0 = Θ W ′ 0 , where W ′ 0 ∈ R d .
- (Iterate) For t ∈ [ T ] , apply the update rule

<!-- formula-not-decoded -->

with η t &gt; 0 (the learning rate), σ t ≥ 0 (the variance of the Gaussian noise), and ε t ∼ N ( 0 d , I d ) (the isotropic Gaussian noise). Here, the projection is an optional operator often used to keep the norm of the model parameters bounded.

- (Output) Return the final hypothesis W T = Θ W ′ T .

Note that here, we train on a subspace of dimension d &lt; D defined by Θ (randomly picked at initialization and fixed during training). Note also that when σ t = 0 for all t ∈ [ T ] , this algorithm reduces to the minibatch SGD (with projection).

## D.1.1 Mutual information of a mixture of two Gaussians and the component

To state our results, we start by defining two useful functions. Suppose that

<!-- formula-not-decoded -->

where ( J, Y 1 , Y 2 ) are independent real-valued random variables defined as follows: J ∼ Bern ( p ) , Y 1 ∼ N (0 , 1) , and Y 2 ∼ N ( a, 1) , for some a ∈ R . Then, it is easy to show that I ( X ; J ) = f ( a, p ) , where the function f : R × [0 , 1] → [0 , log 2] is defined as 6

<!-- formula-not-decoded -->

Here, g a,p : R × [0 , 1] → R + is defined as a mixture of two scalar Gaussian distributions with probabilities p and 1 -p :

<!-- formula-not-decoded -->

The following lemma, proved in the supplements, establishes some properties of the function f ( a, p ) .

̸

Lemma 3. i) For every p ∈ [0 , 1] , f (0 , p ) = 0 . ii) For every p ∈ [0 , 1] , f ( a, p ) = f ( -a, p ) and f ( a, p ) is an strictly increasing function of a in the range [0 , ∞ ) . iii) lim a →∞ f ( a, p ) = log(2) h b ( p ) . iv) For every a ∈ R , f ( a, p ) = f ( a, 1 -p ) and for a = 0 , f ( a, p ) is strictly increasing with respect to p in the range [0 , 1 / 2] .

6 All logarithms are considered to have the base of e .

## D.1.2 Lossless generalization bound

We start by stating our bound in its simplest form.

Theorem 10. Suppose that ℓ ∈ [0 , C ] . Then, the generalization error of a random subspace training algorithm, optimized using iterations defined in 17, is upper-bounded as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

This result is proved in Appendix H.2.

In the bound of equation 20, the term f ( η t σ t ∆ t,i , p t,i ) is an increasing function with respect to η t σ t , ∆ t,i , and a decreasing function with respect to | p t,i -1 / 2 | . As t increases, the learning algorithm 'memorizes' more of the dataset; therefore, | p t,i -1 / 2 | increases and thus these terms decrease. Furthermore, the learning rate decreases, causing this term to decrease more.

Note that by Lemma 3, f ( · , p ) is maximized for p = 1 2 . Hence, a simpler upper bound from Theorem 10 can be achieved by replacing p t,i by 1 2 .

## D.1.3 Lossy generalization bound

The bound of Theorem 10 has a clear shortcoming; whenever η t σ t is very small, the bound becomes loose. In particular, for SGD where σ t = 0 , the bound becomes vacuous. In this section, to overcome this issue, we consider a lossy version of the above bound. While the lossy bound can be stated without any further assumptions, for a more concrete bound, we make the following assumptions.

Assumption 11 (Lipschitzness) . The loss function is L -Lipschitz, i.e., for any w ′ 1 , w ′ 2 ∈ R d , any z ∈ Z , and any Θ ∈ St( d, D ) , we have | ℓ ( z, Θ w ′ 1 ) -ℓ ( z, Θ w ′ 2 ) | ≤ L ∥ w ′ 1 -w ′ 2 ∥ .

Note that since Θ ⊤ Θ = I d , then ∥ w ′ 1 -w ′ 2 ∥ = ∥ Θ w ′ 1 -Θ w ′ 2 ∥ .

Assumption 12 (Contractivity) . There exists some α ∈ R + , such that for any w ′ 1 , w ′ 2 ∈ W ′ , z ∈ Z , and Θ ∈ St( d, D ) , we have

<!-- formula-not-decoded -->

Whenever α &lt; 1 , we say the projected SGLD is α -contractive.

Similar assumptions have been used in previous works, such as [68]. In fact, the contractivity property of SGD has been theoretically proved under certain conditions, such as when the loss function is smooth and strongly convex [68-70].

In addition to being sensitive to cases where η t σ t is very small, the bound of Theorem 10 does not account for the 'forgetting' effect of the iterative optimization algorithms: the information obtained by W ′ T about J i in the initial iterations will eventually fade out, as T increases. To account for this effect, similar to [66, 67], we assume that W ′ = B D ( R ) , 7 for some R ∈ R + .

Theorem 13. Suppose that ℓ ∈ [0 , C ] , W ′ = B D ( R ) , for some R ∈ R + , and Assumptions 11 and 12 hold with constants L ∈ R + and α ≤ 1 , respectively. Then, for any set of { ν t } t ∈ [ T ] , such that ν t ∈ R + , the generalization error of a random subspace training algorithm, optimized using iterations defined in 17, is upper bounded as

<!-- formula-not-decoded -->

7 In this setup, for w ′ ∈ W ′ , Proj { w ′ } = w ′ and otherwise Proj { w ′ } = R ∥ w ′ ∥ w ′ .

where

<!-- formula-not-decoded -->

where ˆ W t are random variables that satisfy

<!-- formula-not-decoded -->

for ε ′ t ∼ N ( 0 d , I d ) , which is an auxiliary additional noise, independent of all other random variables, and where Φ( x ) ≜ ∫ ∞ x 1 √ 2 π exp( -y 2 / 2)d y is the Gaussian complementary cumulative distribution function (CCDF).

This theorem is proved in Appendix H.3. Here, we discuss some remarks.

First, the 'gained' information from the initial iterations fades as T →∞ , when q t &lt; 1 (note that always q t ≤ 1 ).

Second, we note that, unlike in Theorem 10, where p t,i depends on all past iterations in which sample i is used, in this theorem, ˆ p t,i depends only on the immediate past iteration. It can be shown that a similar result can be achieved for Theorem 13 i.e., allowing ˆ p t,i to depend on all past iterations, at the expense of replacing all { q t } t by 1 .

Third, it can be observed that if ∀ t ∈ [ T ] : ν t = 0 , we recover Theorem 10, except for the definition of p t,i , that can be adjusted at the expense of replacing all { q t } t by 1 , as explained above. Furthermore, by increasing ν t , the second term in equation 21, i.e. the 'distortion' term, increases; but the first 'rate' term decreases since f ( η t √ σ 2 t + ν 2 t ˆ ∆ t,i , ˆ p t,i ) decreases. Therefore, in general, the lossy bound can outperform the lossless bound. In particular, for SGD, i.e., when σ t = 0 , the lossless bound and previous works (for the case of no projection) [38, 63-67] become vacuous, while the lossy bound does not.

Lastly, to achieve this bound, we considered a sequence of parallel 'perturbed' iterations. In each of these auxiliary iterations, we introduced an additional independent noise ν t ε ′ t , where ε ′ t ∼ N ( 0 d , I d ) . It can be seen that for the contractive SGD/SGLD, the effect of added perturbation in the initial iterations vanishes as T →∞ . Therefore, once again, it can be seen that the effect of the increase in mutual information from the initial iterations eventually fades.

## E Proof of Theorem 1

We prove the theorem in its most general form stated in Remark 1. This means that we assume that the learning algorithm A is also aware of the projection matrix Θ , i.e., A : Z n × R D × d → W takes both the dataset S n and the projection matrix Θ as input to learn W . Moreover, we allow the quantization step to depend on S , Θ , and A ( S n , Θ) . In this general case, ˆ W = ˆ A ( S, Θ) = ˜ A (Θ , S n , A ( S, Θ)) . We denote this general compressed algorithm by P ˆ W | S n ,W, Θ . Note that P ˆ W | Θ ⊤ W is a special case of this more general setup.

Fix some ϵ ∈ R and the quantization set ˆ W . Consider any Markov kernel P ˆ W | S n ,W, Θ and P Θ that satisfy the following distortion criterion:

<!-- formula-not-decoded -->

Using this condition, it is sufficient to show that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Denote the marginal distribution of ( S n , Θ , ˆ W ) under P S n P Θ P W, ˆ W | S n , Θ by P S n , Θ , ˆ W and conditional distribution of ˆ W given ( S n , Θ) by P ˆ W | S n , Θ . Hence, P S n , Θ , ˆ W = P S n P Θ P ˆ W | S n , Θ and

<!-- formula-not-decoded -->

It is hence sufficient to show that for any ˜ S and Θ ,

<!-- formula-not-decoded -->

Denote P ˆ W | ˜ S , Θ ≜ E P J [ P ˆ W | ˜ S J , Θ ] and P J , ˆ W | ˜ S , Θ ≜ P J P ˆ W | ˜ S J , Θ ≜ P J | ˜ S , Θ , ˆ W P ˆ W | ˜ S , Θ be the conditional distributions of ( J , ˆ W ) given ( ˜ S , Θ) . Note that the marginal distribution of J under P J , ˆ W | ˜ S , Θ is P J , i.e.,

<!-- formula-not-decoded -->

Now, fix some λ = 0 that will be determined later. We have

̸

<!-- formula-not-decoded -->

where ( a ) follows from Donsker-Varadhan's inequality and ( b ) by the inequality 1 2 ( e -x + e x ) ≤ e x 2 / 2 . Hence,

<!-- formula-not-decoded -->

where the last step is followed by letting

This completes the proof.

<!-- formula-not-decoded -->

## F Proofs of Section 4 and Appendix B: Application to raised limitations of CMI bounds

For the proofs of Section 4 and Appendix B, we always consider the normalized setup, i.e., R = 1 , L = 1 (for Theorem 3), L c = 1 (for Proposition 1), and B = 1 (for Theorem 7). The proof applies for arbitrary values of ( R,L,L c , B ) , by simply scaling the constants.

All proofs are based on Theorem 1, with a particular class of choices of P Θ and P ˆ W | Θ ⊤ W , called the choices from the scheme JL ( d, c w , ν ) for some d ∈ N , c w ∈ [ 1 , √ 5 / 4 ) , and ν ∈ (0 , 1] , described in Appendix F.1. For a given JL ( d, c w , ν ) , we then use Theorem 1 for some suitable ϵ ∈ R :

<!-- formula-not-decoded -->

Recall that the term ∆ ℓ ˆ w ( ˜ S , Θ) is defined as

<!-- formula-not-decoded -->

and the choices of P Θ and P ˆ W | Θ ⊤ W should satisfy the distortion criterion

<!-- formula-not-decoded -->

For brevity, we often use the notation

<!-- formula-not-decoded -->

Furthermore, denote the D -dimensional ball of radius ν ∈ R + and center w ∈ R D by B D ( w,ν ) . If w = 0 D , for simplicity we write B D ( 0 D , ν ) ≡ B D ( ν ) , where 0 D designates the all-zero vector in R D .

## F.1 Johnson-Lindenstrauss projection scheme

Fix some constant c w ∈ [ 1 , √ 5 4 ) and ν ∈ (0 , 1] . Let d ∈ N ∗ and Θ be a matrix of size D × d whose elements are i.i.d. samples from N (0 , 1 /d ) . For a given Θ and W = A ( S n ) , in the scheme JL ( d, c w , ν ) , let

<!-- formula-not-decoded -->

Let V ν be a random variable that takes value uniformly over B d ( ν ) . Let ˆ W ∈ ˆ W = B d ( c w + ν ) be defined as

<!-- formula-not-decoded -->

This means that ˆ W is a random variable that takes value uniformly over B d ( U, ν ) :

<!-- formula-not-decoded -->

In other words, we define ˆ W as a quantization of W ′ = Θ ⊤ W obtained as follows: if ∥ Θ ⊤ W ∥ ≤ c w , then ˆ W is uniformly sampled from B d ( Θ ⊤ W,ν ) ; otherwise, ˆ W is uniformly sampled from B d ( ν ) . Such quantization has been previously used in [22] to establish a generalization bound for the distributed SVM learning algorithm.

Disintegrated CMI bound: The disintegrated CMI bound CMI Θ ( ˜ S , ˆ A ) in the scheme JL ( d, c w , ν ) can be upper bounded as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

- ( a ) follows from the fact that conditioning does not increase the entropy,
- ( b ) yields due to Markov chain ˆ W -Θ ⊤ W -( ˜ S , Θ , J , W ) ,
- and ( c ) holds since i) ∥ ˆ W ∥ ≤ c w + ν by construction and hence h ˜ S , Θ ( ˆ W ) is upper bounded by the differential entropy of a random variable taking value uniformly over B d ( c w + ν ) , and ii) since given Θ ⊤ W , ˆ W is chosen uniformly over a d -dimensional ball either around 0 d or Θ ⊤ W , depending on ∥ Θ ⊤ W ∥ .

## F.2 Proof of Theorem 3

As explained in Appendix F, we consider the case L = R = 1 , and use Theorem 1 using the JL ( d, c w , ν ) transformation described in Appendix F.1, with some d ∈ N + , c w ∈ [ 1 , √ 5 / 4 ) , and ν ∈ (0 , 1] . To do so, we start by bounding CMI Θ ( ˜ S , ˆ A ) , the distortion equation 23, and E ˜ S , Θ [∆ ℓ ˆ w ( ˜ S , Θ)] .

Bound on the disintegrated CMI: It is shown in equation 26 that

<!-- formula-not-decoded -->

Bound on the distortion: Next, we bound the distortion term. By definition, and using the linearity of expectation, we obtain

<!-- formula-not-decoded -->

where ¯ Z ≜ E Z ∼ µ [ Z ] -1 n ∑ n i =1 Z i .

<!-- formula-not-decoded -->

Let E be the event that ∥ Θ ⊤ W ∥ &gt; c w and denote by E c the complementary event of E . By the law of total expectation,

<!-- formula-not-decoded -->

By definition of ˆ W , E [ ˆ W ] = 0 under E , E [ ˆ W ] = Θ ⊤ W otherwise. Therefore, equation 29 can be simplified as

<!-- formula-not-decoded -->

✶ where equation 30 follows from Cauchy-Schwarz inequality, and equation 31 results from H¨ older's inequality.

Now, we bound each of the terms E [ ∥ Θ ⊤ ¯ Z ∥ 2 ] , E [ ∥ Θ ⊤ W ∥ 4 ] , and E [ ✶ {E} ] .

- Since the elements of Θ ∈ R D × d are i.i.d. from N (0 , 1 /d ) , then for any fixed vector x ∈ R D , each entry of √ d Θ ⊤ x ∥ x ∥ is an independent random variable distributed according to N (0 , 1) . Hence, V x = ∥ ∥ ∥ √ d Θ ⊤ x ∥ x ∥ ∥ ∥ ∥ 2 is a chi-squared random variable with d -degrees of freedom, and we

have

This concludes that for any ¯ z ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Moreover, since V x is a chi-squared distribution with d -degrees of freedom, we have that

<!-- formula-not-decoded -->

Hence for every w ∈ W ,

<!-- formula-not-decoded -->

- By [71, Lemma 9], for any w ∈ B D (1) , if c w ∈ [1 , √ 5 2 ) ,

<!-- formula-not-decoded -->

More precisely by [71, Lemma 9] we have for any t ∈ [0 , 1 / 4) and any w ∈ B d (1) ,

<!-- formula-not-decoded -->

Wenote that this inequality is a 'single-sided' tail bound version of [71, Lemma 9] (while therein stated as a 'double-sided' tail bound). This explains why RHS of the inequality in [71, Lemma 9] is 2 e -0 . 21 dt 2 , while here we have e -0 . 21 dt 2 .

Next, note that ( t +1) ∥ w ∥ 2 ≤ ( t +1) , hence

<!-- formula-not-decoded -->

Thus, by letting t = c 2 w -1 for c w ∈ [1 , √ 5 / 4) , we have

<!-- formula-not-decoded -->

Combining the above upper bounds on E [ ∥ Θ ⊤ ¯ Z ∥ 2 ] , E [ ∥ Θ ⊤ W ∥ 4 ] , and E [ {E} ] , we obtain,

<!-- formula-not-decoded -->

where equation 33 follows from assuming that W ⊆ B D (1) .

It remains then to upper bound E [ ∥ ¯ Z ∥ 2 ] . By definition of ¯ Z and the linearity of expectation, we have

̸

<!-- formula-not-decoded -->

̸

̸

where equation 34 results from Cov( Z i , Z j ) = 0 for i = j since Z i , Z j are independent, and equation 35 follows from Z ⊆ B D (1) (thus, for any i , ∥ E [ Z ] -Z i ∥ ≤ 2 ).

Combining equation 33 and equation 34, we conclude that the distortion is bounded by

<!-- formula-not-decoded -->

Bound on E ˜ S , Θ [∆ ℓ ˆ w ( ˜ S , Θ)] : We have

<!-- formula-not-decoded -->

where ( a ) follows by Cauchy-Schwarz inequality, ( b ) is derived since ∥ ˆ w ∥ ≤ ( c w + ν ) , and ( c ) since for any fixed z , each entry of Θ ⊤ z ∥ z ∥ is an independent random variable distributed according to N (0 , 1 d ) and hence

<!-- formula-not-decoded -->

since ∥ ∥ Z i, 0 -Z i, 1 ∥ ∥ ≤ 2 .

## Generalization Bound: Now, let

<!-- formula-not-decoded -->

Inequality 36 shows that the above choices of P Θ and P ˆ W | Θ ⊤ W (according to the scheme JL ( d, c w , ν ) ) satisfy the distortion criterion equation 23. Hence, equation 22 gives

<!-- formula-not-decoded -->

where ( a ) is achieved using equation 27, ( b ) by Jensen inequality and due to the concavity of the function √ x , and ( c ) is derived using equation 37.

The proof is completed by letting

## F.3 Proof of Proposition 1

As explained in Appendix F, it is sufficient to consider the case L c = R = 1 . We have

<!-- formula-not-decoded -->

where ( a ) by definition of ℓ sc ( z, w ) = -⟨ w,z ⟩ + λ 2 ∥ w ∥ 2 by Definition 6, ( b ) holds since by Definition 4, we have ℓ c ( z, w ) = -L ⟨ w,z ⟩ , and ( c ) follows by Theorem 3.

## F.4 Proof of Theorem 7

As explained in Appendix F, we consider the case L = R -B = 1 . First, note that similar to the proof of Proposition 1, the generalization error does not change, if we consider the loss function ℓ glm ( z, w ) ≜ g ( ⟨ w,ϕ ( z ) ⟩ , z ) -g (0 , z ) instead of ℓ gl ( z, w ) = g ( ⟨ w,ϕ ( z ) ⟩ , z ) + r ( w ) . More precisely,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) follows since E Z ∼ µ [ g (0 , Z )] = E Z i ∼ µ [ g (0 , Z i )] .

Hence, for the rest of the proof, we consider the generalization with respect to the following loss function:

<!-- formula-not-decoded -->

Note that due the Lipschitzness of the function g ( · , · ) with respect to its first argument, for every z ∈ Z and w ∈ W , we have

<!-- formula-not-decoded -->

Furthermore since ∥ w ∥ , ∥ ϕ ( z ) ∥ ≤ 1 , using Cauchy-Schwarz inequality yields

<!-- formula-not-decoded -->

Now, we proceed to establish a generalization bound with respect to the loss function ℓ glm ( z, w ) . We use Theorem 1 with the JL ( d, c w , ν ) transformation described in Appendix F.1, for some d ∈ N + , c w ∈ [ 1 , √ 5 / 4 ) , and ν ∈ (0 , 1] . To do so, We start by bounding CMI Θ ( ˜ S , ˆ A ) , the distortion equation 23, and E ˜ S , Θ [∆ ℓ ˆ w ( ˜ S , Θ)] .

Bound on the disintegrated CMI: It is shown in equation 26 that

<!-- formula-not-decoded -->

Bound on the distortion: Next, we bound the distortion term.

<!-- formula-not-decoded -->

where ( a ) holds due to Lipschitzness of the function g with respect to its first argument. Hence,

<!-- formula-not-decoded -->

where the last step follows since by equation 25, ˆ W = U + V ν .

In the rest, we fix z and w and upper bound each of the terms in the right-hand side of equation 40:

<!-- formula-not-decoded -->

Let E be the event that ∥ Θ ⊤ W ∥ &gt; c w and denote by E c the complementary event of E .

- We start by bounding C 1 .

<!-- formula-not-decoded -->

where ( a ) holds since by equation 24, under E , U = 0 d , and under E c , U = Θ ⊤ W , ( b ) is derived since ∥ w ∥ , ∥ ϕ ( z ) ∥ ≤ 1 and hence, Cauchy-Schwarz inequality yields |⟨ w,ϕ ( z ) ⟩| ≤ 1 , and ( c ) derived by equation 32.

Thus, to bound C 1 , it remained to bound E Θ [∣ ∣ ⟨ w,ϕ ( z ) ⟩ -〈 Θ ⊤ w, Θ ⊤ ϕ ( z ) 〉∣ ∣ ] . We use a trick borrowed from [71, Proof of Theorem 9]. Note that ∥ w ∥ , ∥ ϕ ( z ) ∥ ≤ 1 . Hence, to upper bound E Θ [∣ ∣ ⟨ w,ϕ ( z ) ⟩ -〈 Θ ⊤ w, Θ ⊤ ϕ ( z ) 〉∣ ∣ ] , it is sufficient to consider the case where ∥ w ∥ = ∥ ϕ ( z ) ∥ = 1 . Let

<!-- formula-not-decoded -->

It is easy to verify that ⟨ v, ϕ ( z ) ⟩ = 0 . Hence, since ϕ ( z ) ⊥ v , we have

<!-- formula-not-decoded -->

Now, for every r ∈ [ d ] , denote the r 'th row of Θ ⊤ ∈ R d × D by T r and let

<!-- formula-not-decoded -->

Since ϕ ( z ) ⊥ v and since the Gaussian distributions are rotationally invariant, we have that X 1 , . . . , X d , Y 1 , . . . , Y d are i.i.d. Gaussian random variables distributed according to N (0 , 1 /d ) . Hence, using the identity w = v + ⟨ w,ϕ ( z ) ⟩ ϕ ( z ) , we can write

<!-- formula-not-decoded -->

where ( a ) is derived using the inequalities ∥⟨ w,ϕ ( z ) ⟩ ≤ 1 and ∥ v ∥ ≤ 1 . We bound the expectation over Θ of each of these terms, denoted respectively as

<!-- formula-not-decoded -->

- -Note that the distribution of

<!-- formula-not-decoded -->

is a chi-squared distribution χ 2 ( d ) with d -degrees of freedom. Moreover, asymptotically as d → ∞ , χ 2 ( d ) converges to N ( d, 2 d ) . Equivalently, asymptotically, χ 2 ( d ) -d →N (0 , 2 d ) . Combining this asymptotic behavior with the fact that for a Gaussian random variable Z ∼ N (0 , σ 2 ) , with σ ∈ R + , we have that E [ | Z | ] = σ √ 2 π , yield

<!-- formula-not-decoded -->

- -To bound the term C 1 , 2 , notice that ∑ r ∈ [ d ] X r Y r converges to a random variable with Gaussian distribution N (0 , 1 /d ) , as d → ∞ . Hence, once again using the fact that for a Gaussian random variable Z ∼ N (0 , σ 2 ) , E [ | Z | ] = σ √ 2 π , yield

<!-- formula-not-decoded -->

Combining equation 41, equation 42, and equation 43 gives

<!-- formula-not-decoded -->

- Now to bound C 2 , let V ν = ( V ν, 1 , . . . , V ν,d ) .

<!-- formula-not-decoded -->

where ( a ) holds by the symmetry of the distribution of V ν , ( b ) holds since E Θ ∼ P Θ [ ∥ Θ ⊤ ϕ ( z ) ∥ ] ≤ E Θ ∼ P Θ [ ∥ Θ ⊤ ϕ ( z ) ∥ 2 ] 1 / 2 = ∥ ϕ ( z ) ∥ ≤ 1 , ( c ) holds by Lemma 4, proved in Appendix F.5, and ( d ) holds since by using Gautschi's inequality we have Γ( x +1 / 2) Γ( x +1) ≤ 1 √ x .

Lemma 4. Let V ν = ( V ν, 1 , . . . , V ν,d ) ∼ Uniform ( B d ( ν )) . Then, E V ν ∼ Uniform ( B d ( ν )) [ | V ν, 1 | ] = ν Γ ( d +2 2 ) √ π Γ ( d +3 2 ) .

Combining equation 39. equation 45, and equation 46 gives

<!-- formula-not-decoded -->

Bound on E ˜ S , Θ [∆ ℓ ˆ w ( ˜ S , Θ)] : We have

<!-- formula-not-decoded -->

where ( a ) holds by equation 38 and ( b ) since by construction ∥ ˆ w ∥ ≤ c w + ν . Hence,

<!-- formula-not-decoded -->

where

- ( a ) follows from equation 48,
- ( b ) since for any fixed z , each entry of Θ ⊤ z ∥ z ∥ is an independent random variable distributed according to N (0 , 1 d ) and hence

<!-- formula-not-decoded -->

Generalization Bound: Now, using Theorem 1 for the above choices of P Θ and P ˆ W | Θ ⊤ W (according to the scheme JL ( d, c w , ν ) ) gives

<!-- formula-not-decoded -->

where ( a ) is achieved using equation 27 and equation 47, ( b ) by Jensen inequality and due to the concavity of the function √ x , and ( c ) is derived using equation 37.

The proof is completed by letting

## F.5 Proof of Lemma 4

Note that

<!-- formula-not-decoded -->

where X = ( X 1 , . . . , X d ) ∼ Uniform ( B d (1)) . Hence, it is sufficient to show that E X ∼ Uniform ( B d (1)) [ | X 1 | ] = Γ ( d +2 2 ) √ π Γ ( d +3 2 ) .

<!-- formula-not-decoded -->

First, we compute the marginal distribution of X 1 . Note that

<!-- formula-not-decoded -->

Now, we have

<!-- formula-not-decoded -->

where ( a ) is achieved by letting u = x 2 1 and in ( b ) , Beta ( · , · ) is the Beta function.

## G Proofs of Section 5 and Appendix C: Memorization

In this section, we provide the proofs of Section 5 and Appendix C. Recall that for a given K i , the adversary outputs its guess of K i as ˆ K i ≜ Q ( W,Z i,K i , µ ) . Throughout the proofs and for better readability, we sometimes denote ˆ K i = 1 by ˆ K i = 'in' and ˆ K i = 0 by ˆ K i = 'not in', referring to the semantic meaning that the given Z i,K i is part of the training dataset or not.

## G.1 Proof of Theorem 5

We prove each part separately. As stated in the beginning of Appendix G, throughout the proofs and for better readability, we sometimes denote ˆ K i = 1 by ˆ K i = 'in' and ˆ K i = 0 by ˆ K i = 'not in', referring to the semantic meaning that the given Z i,K i is part of the training dataset or not.

## G.1.1 Part i.

We prove the result by contradiction. Suppose that there exists an adversary for the algorithm A that is ξ -sound and certifies a recall of m samples with probability q , where ξ &lt; q and m = Ω( n ) . As before, we denote the output of the learning algorithm by A n ( S n ) = W .

Recall that ˜ S J = { Z 1 ,J 1 , Z 2 ,J 2 , . . . , Z n,J n } is the training dataset S n and ˜ S \ ˜ S J is the test dataset S ′ n .

Define ˆ J i ∈ { 0 , 1 } as follows:

<!-- formula-not-decoded -->

where U i ∼ Bern (1 / 2) is a binary uniform random variable, independent of other random variables.

Recall that given A , a ξ -sound adversary means that,

<!-- formula-not-decoded -->

and an adversary certifying a recall of m samples means that,

<!-- formula-not-decoded -->

Since we assumed m = Ω( n ) , there exists c 1 ∈ (0 , 1] and n 0 ∈ N such that, for all n ≥ n 0 , m ≥ c 1 n . The second condition then yields,

<!-- formula-not-decoded -->

Define the Hamming distance d H : { 0 , 1 } n ×{ 0 , 1 } n → [ n ] between binary vectors J and ˆ J as

̸

Next, we use Fano's inequality with approximate recovery [59, Theorem 2]. Let t = 1 n ⌊ n 2 ( 1 -c 1 2 )⌋ and denote

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that N ˆ j is the same for all ˆ j ∈ { 0 , 1 } n . Indeed, d H ( j , ˆ j ) = d H ( j ⊕ a , ˆ j ⊕ a ) , where ⊕ denotes the modulo two summation, for any a ∈ { 0 , 1 } n , and ∑ j ∈{ 0 , 1 } n ✶ { d H ( j ⊕ a , ˆ j ⊕ a ) ≤ nt } = ∑ j ∈{ 0 , 1 } n ✶ { d H ( j , ˆ j ⊕ a ) ≤ nt } . Hence, N ˆ j = N ˆ j ⊕ a for any a , and the maximum over ˆ j of N ˆ j is equal to N 1 n .

With these notations, we have

<!-- formula-not-decoded -->

where ( a ) follows by the assumption of the theorem, ( b ) results from K is independent of ( W, ˜ S , J ) , ( c ) results from I ( J ; ˆ J | W, ˜ S , K ) = 0 since ˆ J is a function of ( W, ˜ S , K ) , ( d ) results from I ( J ; W, ˆ J | ˜ S , K ) = I ( J ; ˆ J | ˜ S , K ) + I ( J ; W | ˜ S , K , ˆ J ) ≥ I ( J ; ˆ J | ˜ S , K ) (by the positivity of mutual information), ( e ) is due to the identities below,

<!-- formula-not-decoded -->

( f ) results from applying Fano's inequality with approximate recovery [59, Theorem 2], and ( g ) is derived using the claim, proved later below, that N 1 n ≤ c 3 2 nh b ( t ) for some constant c 3 ∈ R + and for n sufficiently large.

Note that t = 1 n ⌊ n 2 ( 1 -c 1 2 )⌋ &lt; 1 / 2 and as n → ∞ , t → 1 -c 1 / 2 2 &lt; 1 / 2 . Hence, since h b ( x ) is a continuous function of x ∈ [0 , 1] , 1 -h b ( t ) converges to the constant 1 -h b ( 1 -c 1 / 2 2 ) &gt; 0 . Hence, if we show that for sufficiently large n , 1 -P e t &gt; 0 , we obtain a contradiction. Since the left-hand side is of order o ( n ) , which is greater than the right-hand side, which is Ω( n ) , and the proof is complete.

Hence, it remains to show for n sufficiently large, Claim i) N 1 n ≤ c 3 2 nh b ( t ) for some constant c 3 ∈ R + , and Claim ii) P e t &lt; 1 .

## Proof of Claim i)

We have

<!-- formula-not-decoded -->

where n ′ = 2 ⌈ n 2 ⌉ and c 3 ∈ R + , ( a ) results from n ′ ≥ n , ( b ) follows from applying [72, Proposition 5.18] 8 ( n ′ is even and nt ≤ n ′ / 2 -1 ), ( c ) is derived using the relation

<!-- formula-not-decoded -->

which is valid for any m ∈ N and 1 ≤ j ≤ m -1 (see [73, Exercise 5.8.a]), and ( d ) holds for sufficiently large n , using n ≤ n ′ ≤ n +1 .

̸

<!-- formula-not-decoded -->

̸

̸

̸

8 See also https://mathoverflow.net/questions/17202/sum-of-the-first-k-binomial-coefficients-for-fixed-n for a reformulation.

̸

<!-- formula-not-decoded -->

and we justify the main steps hereafter:

- ( a ) holds since under the event E c 1 , we have that ∀ i ∈ [ n ] , ✶ {Q ( W,Z i,J c i , µ ) = 'in' } = 0 and also whenever i) both Q ( W,Z i,J c i , µ ) = 'not in' and Q ( W,Z i,J i , µ ) = 'not in', ˆ J i is chosen as U i and hence the Hamming difference of the i 'th coordinate is ✶ { U i = J i } , and ii) when Q ( W,Z i,J c i , µ ) = 'not in' and Q ( W,Z i,J i , µ ) = 'in', ˆ J i is chosen as J i and hence the Hamming difference of the i 'th coordinate is 0.

̸

- ( b ) holds since under the event E c 2 , we have that ∑ i ∈ [ n ] ✶ {Q ( W,Z i,J i , µ ) = 'not in' } ≤ n (1 -c 1 ) , · ( c ) holds since for r &lt; nt , the probability is zero,
- ( d ) holds by Hoeffding's inequality for the independent uniform random variables ✶ { U i = J i } and since nt &gt; n (1 -c 1 / 2) / 2 ≥ r/ 2 for n sufficiently large, · ( e ) holds for n large enough since,

̸

<!-- formula-not-decoded -->

where ( ∗ ) is derived since i) for n sufficiently large, ⌈ n (1 -c 1 ) ⌉ nt = ⌈ n (1 -c 1 ) ⌉ ⌊ n 2 (1 -c 1 2 ) ⌋ which is less than 2 for n large, and ii) since ( 1 x -1 + x 4 ) is decreasing in the range (0 , 2] ,

- ( f ) results from P ( E 1 , E 2 ) ≤ P ( E 1 ) + P ( E 2 ) .

Since for sufficiently large n , e -⌈ n (1 -c 1 ) ⌉ ( nt ⌈ n (1 -c 1 ) ⌉ -1 2 ) 2 (which converges to e -nc 2 1 8(1 -c 1 ) ) gets sufficiently small, hence, if ξ &lt; q , then P e t &lt; 1 . This completes the proof of Claim ii) , and hence of Part i) .

## G.1.2 Part ii.

Similarly to Part i) (Appendix G.1.1), we will prove the result by contradiction: assume that there exists an adversary for A such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

This also gives,

<!-- formula-not-decoded -->

In our proof, we allow the adversary to be stochastic. We denote expectations and probabilities with respect to the adversary's randomness (which is independent of all other random variables) by E Q [ · ] and P Q [ · ] , where needed. The main part of the proof relies on the following lemma, which we state below but prove later (in Appendix G.7) for better readability.

Lemma 5. The following holds.

<!-- formula-not-decoded -->

By Lemma 5, we have

<!-- formula-not-decoded -->

where ( a ) holds using (50) and ∑ i ∈ [ n ] ✶ {Q ( W,Z i,J i , µ ) = 'not in' } ≤ n . Hence, using Markov's inequality, or equivalently,

<!-- formula-not-decoded -->

Hence, for any we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by varying q ′ over the interval (0 , αq ) , the ratio m ′ /n changes asymptotically from 0 to αq . In other words, if n is sufficiently large, then for any

<!-- formula-not-decoded -->

we have

This completes the proof of Part ii) .

<!-- formula-not-decoded -->

## G.2 Proof of Theorem 6

To prove Theorem 6, we show that for any learning algorithm A : Z → R D , the projected-quantized algorithm, defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any distribution µ . Having shown this, applying Theorem 5 completes the proof.

Fix any arbitrary distribution µ . Consider the construction of JL ( d, c w , ν ) , described in Appendix F.1. It is shown in equation 26 that

<!-- formula-not-decoded -->

which, together with the data-processing inequality, yield

<!-- formula-not-decoded -->

Furthermore, similar to equation 36, where it is shown that

<!-- formula-not-decoded -->

it can be shown that

<!-- formula-not-decoded -->

Plugging the choices satisfies equation 8 and

<!-- formula-not-decoded -->

in equation 52 and equation 53 result equation 51 and equation 8, which completes the proof.

## G.3 Proof of Lemma 1

If m = 0 , then consider an adversary that always outputs Q ( W,Z,µ ) = 0 , for any Z ∈ Z .

̸

✶ Consider an adversary that first picks a random V . If V = 0 , then for any Z ∈ Z , it declares Q ( W,Z,µ ) = 0 . Otherwise ( i.e., V = 1 ), it declares Q ( W,Z,µ ) = 0 with probability r n and Q ( W,Z,µ ) = 1 with probability 1 -r n , independently of ( W,Z,µ ) .

In the following, we assume that m = nm ′ = 0 . Let V ∈ { 0 , 1 } be a binary random variable, independent of all other random variables, such that P ( V = 0) = α . For example, if there exists a set B ⊆ W such that P ( W ∈ B ) = α , then the adversary can set V = { W / ∈ B} .

If V = 0 , the adversary never recalls m samples with any positive probability

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

Using the above two relations, this adversary is ξ -sound and recalls m samples with probability q if, restricting to V = 1 , the adversary is ξ (1 -α ) -sound and recalls m samples with probability q (1 -α ) . For the adversary to be ξ (1 -α ) -sound given V = 1 , we should have P ( ∀ i ∈ [ n ] , Q ( W,Z i, 0 , µ ) = 0) ≥ 1 -ξ (1 -α ) . Hence, this adversary is ξ -sound if and only if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, when V = 1 , to find the probability of recalling m = nm ′ samples with probability q (1 -α ) , note that the probability of Q ( W,Z i, 1 , µ ) = 1 is equal to (1 -r n ) . We consider two cases:

<!-- formula-not-decoded -->

ii. If m ′ &lt; 1 -r n , using Hoeffding's inequality, we have

<!-- formula-not-decoded -->

Considering these two cases separately,

- i. We should find a value of α such that q (1 -α ) ≤ 1 and 0 ≥ n √ 1 -ξ (1 -α ) . Both conditions are satisfied for α = 1 -ξ , if ξ ≥ q .
- ii. It is sufficient to find a value for r n such that m ′ &lt; (1 -r n ) , r n ≥ n √ 1 -ξ (1 -α ) and 1 -

<!-- formula-not-decoded -->

It satisfies the first condition and the recall condition. Lastly, the soundness condition is satisfied if for sufficiently large n , we have

<!-- formula-not-decoded -->

## G.4 Proof of Theorem 8

We prove the theorem and the comment after it, separately.

In the first case and to prove Theorem 8, we show that for every r &lt; 1 there exists a projection matrix Θ ∈ R D × d with d = ⌈ n 2 r -1 ⌉ , a Markov Kernel P ˆ W | Θ ⊤ W and a compression algorithm A ∗ Θ ,n : Z n → R d , defined as A ∗ Θ ,n ( S n ) ≜ ˜ A (Θ ⊤ A ( S n )) = ˆ W , such that

<!-- formula-not-decoded -->

and therefore,

<!-- formula-not-decoded -->

Having shown this, then applying Theorem 5 completes the proof.

In the second case, we show that for every r ∈ R , there exist a projection matrix Θ ∈ R D × d with d = ⌈ r log( n ) ⌉ , a Markov Kernel P ˆ W | Θ ⊤ W and a compression algorithm A ∗ Θ ,n : Z n → R d , defined as A ∗ Θ ,n ( S n ) ≜ ˜ A (Θ ⊤ A ( S n )) = ˆ W , such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Having shown this, again applying Theorem 5 completes the proof.

Hence, it remains to show the existence of such projection matrices Θ ∈ R D × d , Markov Kernels P ˆ W | Θ ⊤ W and compression algorithms A ∗ Θ ,n : Z n → R d , for each of the above cases.

## G.4.1 Case i.

Consider the construction of JL ( d, c w , ν ) , described in Appendix F.1. It is shown in equation 26 that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We show that for any r &lt; 1 , letting

<!-- formula-not-decoded -->

Hence, for any fixed Θ ,

Now, let results in

<!-- formula-not-decoded -->

Having shown this, it's easy to see that there exists a Θ , for which simultaneously

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Fix this matrix Θ ∈ R D × d and the Markov Kernel P ˆ W | Θ ⊤ W induced by that. Choosing the overall algorithms as A ∗ Θ ,n : Z n → R d completes the proof.

Hence, it remains to show that equation 54 holds. By equation 28, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining above equation with equation 44 for ϕ ( ¯ Z ) = ¯ Z gives

<!-- formula-not-decoded -->

Next, we know by equation 35 that E [ ∥ ¯ Z ∥ ] ≤ E [ ∥ ¯ Z 2 ∥ ] 1 / 2 ≤ 2 √ n . Hence,

<!-- formula-not-decoded -->

The proof is completed by letting and

<!-- formula-not-decoded -->

## G.4.2 Case ii.

Consider the construction of JL ( d, c w , ν ) , described in Appendix F.1. It is shown in equation 26 that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, it is shown in equation 36 that

<!-- formula-not-decoded -->

Hence, there exists at least one Θ for which

<!-- formula-not-decoded -->

Choose this matrix Θ ∈ R D × d and the Markov Kernel P ˆ W | Θ ⊤ W induced by that. Call the overall algorithms as A ∗ Θ ,n : Z n → R d , with the choices

<!-- formula-not-decoded -->

Plugging these constants in equation 55 and equation 56 completes the proof.

## G.5 Proof of Theorem 9

We first provide the proof of Theorem 9. The proof for the comment after the theorem, i.e., to show equation 14 instead of equation 13, then follows similarly to the below proof, in a similar manner shown in the Case ii part of the proof of Theorem 8.

To prove Theorem 9, we follow the Case i part of the proof of Theorem 8, with a slight modification: ¯ Z is replaced by Z , which results in convergence rates roughly √ n larger than the current ones. For the sake of completeness, we provide the proof.

Let

Hence, for any fixed Θ , results in

<!-- formula-not-decoded -->

Following similarly to the Case i part of the proof of Theorem 8, it is sufficient to show that for any r &lt; 1 / 2 , letting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, it remains to show that equation 57 holds. We have

<!-- formula-not-decoded -->

Denote ˜ z ≜ E Z ∼ µ [ Z ] . Hence, since ˆ W = U + V ν , where E V ν [ V ν ] = 0 , we have

<!-- formula-not-decoded -->

Combining the above equation with equation 44, and by replacing ϕ ( ¯ Z ) by ˜ z , gives

<!-- formula-not-decoded -->

The proof is completed by letting

<!-- formula-not-decoded -->

## G.6 Proof of Lemma 2

Consider the the JL ( d, c w , ν ) transformation described in Appendix F.1 with some d ∈ N + , c w ∈ [ 1 , √ 5 / 4 ) , and ν ∈ (0 , 1] . Recall that ˆ W = U + V ν , where V ν be a random variable that takes value uniformly over B d ( ν ) and

<!-- formula-not-decoded -->

Let E be the event that ∥ Θ ⊤ w ∥ &gt; c w and denote by E c the complementary event of E . We have

<!-- formula-not-decoded -->

where

- ( a ) follows by the triangle inequality,
- ( b ) follows by noting that each element of Θ is i.i.d. with distribution N (0 , 1 /d ) ,
- ( c ) holds since V ν ∈ B d ( ν ) ,
- ( d ) is derived by the definition of U and E ,
- ( e ) follows using Cauchy-Schwarz inequality,
- ( f ) is derived in equation 32,
- and ( g ) followed by following relations

<!-- formula-not-decoded -->

shown below.

Proof of norm two. Note that E Θ [ ∥ ∥ ΘΘ ⊤ w ∥ ∥ 2 ] scales with ∥ w ∥ . Hence, it suffices to assume that ∥ w ∥ = 1 . Next, first we show that E Θ [ ∥ ∥ ΘΘ ⊤ w ∥ ∥ 2 ] is the same for any w with ∥ w ∥ = 1 .

For any w ∈ R D , there exists an orthonormal matrix Q ∈ R D × D such that QQ ⊤ = I D and Qw = e 1 ≜ [1 , 0 , 0 , · · · , 0] ⊤ . This matrix can be constructed by letting the first row as w ⊤ , and choosing the other

rows orthogonal to w ⊤ . Next, by letting Θ ′ = Q Θ , we can write

<!-- formula-not-decoded -->

The result follows by noting that E [ ∥ ∥ Θ ′ Θ ′⊤ e 1 ∥ ∥ 2 ] = E [ ∥ ∥ ΘΘ ⊤ e 1 ∥ ∥ 2 ] , since the distribution of Θ is rotationally invariant.

Hence, it is sufficient to compute E [ ∥ ∥ ΘΘ ⊤ e 1 ∥ ∥ 2 ] . Denote the elements of Θ by θ i,j , where i ∈ [ D ] , j ∈ [ d ] . Then, simple algebra gives

<!-- formula-not-decoded -->

We know that for θ ∼ N (0 , 1 /d ) ,

<!-- formula-not-decoded -->

Then, it suffices to consider terms in the expansions that are non-zero, i.e. the terms where only even norms of each random variable appear. We consider all such cases:

̸

1. i = 1 : D -1 choices

1.1. j = j ′ : d choices and and the expectation of each term equals 1 d 2 .

2. i = 1 : 1 choice
2. 2.1. j = j ′ : d choices and and the expectation of each term equals 3 d 2 .

̸

- 2.2. j = j ′ : d ( d -1) choices and and the expectation of each term equals 1 d 2 .

Summing all terms and factorizing properly gives

<!-- formula-not-decoded -->

Proof of norm four. Note that E Θ [ ∥ ∥ ΘΘ ⊤ w ∥ ∥ 4 ] scales with ∥ w ∥ . Hence, it suffices to assume that ∥ w ∥ = 1 . Next, similar to the proof of norm two, it can be shown that E Θ [ ∥ ∥ ΘΘ ⊤ w ∥ ∥ 4 ] is the same for any w with ∥ w ∥ = 1 . Hence, it is sufficient to compute E [ ∥ ∥ ΘΘ ⊤ e 1 ∥ ∥ 4 ] . Denote the elements of Θ by θ i,j , where i ∈ [ D ] , j ∈ [ d ] . Then, simple algebra gives

<!-- formula-not-decoded -->

We know that for θ ∼ N (0 , 1 /d ) ,

<!-- formula-not-decoded -->

Then, it suffices to consider terms in the expansions that are non-zero, i.e. the terms where only even norms of each random variable appear. We consider all such cases:

̸

1. i = i ′ = 1 : D -1 choices

̸

- 1.1. j 1 = j 2 = j ′ 1 = j ′ 2 : d choices, and the expectation of each term equals 9 d 4 .
- 1.2. Two of ( j 1 , j 2 , j ′ 1 , j ′ 2 ) are the same, and two others as well, with a different value: 3 d ( d -1) choices, and the expectation of each term equals 1 d 4 .

Hence, the sum of the expectation of the terms for this case equals:

<!-- formula-not-decoded -->

2. i, i ′ = 1 and i = i ′ : ( D -1)( D -2) choices

̸

- 2.1. j 1 = j 2 = j ′ 1 = j ′ 2 : d choices, and the expectation of each term equals 3 d 4 .
- 2.2. j 1 = j 2 and different from j ′ 1 = j ′ 2 : d ( d -1) choices and the expectation of each term equals 1 d 4 .

Hence, the sum of the expectation of the terms for this case equals:

<!-- formula-not-decoded -->

3. i = 1 and i ′ = 1 or i ′ = 1 and i = 1 : 2( D -1) choices

̸

̸

- 3.1. j 1 = j 2 = j ′ 1 = j ′ 2 : d choices, and the expectation of each term equals 15 d 4 .
- 3.3. j 1 different from j ′ 1 = j ′ 2 = j 2 : d ( d -1) choices and the expectation of each term equals 3 d 4 .
- 3.2. j 1 = j 2 and different from j ′ 1 = j ′ 2 : d ( d -1) choices and the expectation of each term equals 3 d 4 .
- 3.4. j 2 different from j ′ 1 = j ′ 2 = j 1 : d ( d -1) choices and the expectation of each term equals 3 d 4 .

̸

- 3.5. j 1 = j 2 and both different from j ′ 1 = j ′ 2 : d ( d -1)( d -2) choices and the expectation of each term equals 1 d 4 .

Hence, the sum of the expectation of the terms for this case equals:

<!-- formula-not-decoded -->

4. i = i ′ = 1 : 1 choice
2. 4.1. j 1 = j 2 = j ′ 1 = j ′ 2 : d choices, and the expectation of each term equals 105 d 4 .
3. 4.3. Two of ( j 1 , j 2 , j ′ 1 , j ′ 2 ) are the same, and two others as well, with a different value: 3 d ( d -1) choices, and the expectation of each term equals 9 d 4 .
4. 4.2. Exactly three of the indices among ( j 1 , j 2 , j ′ 1 , j ′ 2 ) are the same: 4 d ( d -1) choices and the expectation of each term equals 15 d 4 .
5. 4.4. There are exactly two same indices among ( j 1 , j 2 , j ′ 1 , j ′ 2 ) : 6 d ( d -1)( d -2) choices and the expectation of each term equals 3 d 4 .
6. 4.5. All indices among ( j 1 , j 2 , j ′ 1 , j ′ 2 ) are different: d ( d -1)( d -2)( d -3) choices and the expectation of each term equals 1 d 4 .

Hence, the sum of the expectation of the terms for this case equals:

<!-- formula-not-decoded -->

Finally, summing all terms and factorizing properly gives

<!-- formula-not-decoded -->

## G.7 Proof of Lemma 5

To prove this lemma, we show the below stronger result:

which results also

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For a given ( W,Z i,j ) , denote

P Q ( Q ( W,Z i,j , µ ) = 'in' ) = E Q [ ✶ {Q ( W,Z i,j , µ ) = 'in' } ] ≜ p ( W,Z i,j ) , where the probability and expectation with respect to Q refer to the stochasticity of the adversary. Note that p ( W,Z i,j ) is a measurable function of ( W,Z i,j ) .

For r ∈ { 0 , 1 , . . . , 2 n -1 } , denote its binary representation as r = ( b r, 1 , . . . , b r,n ) , where b r,i ∈ { 0 , 1 } . Now, consider 2 n auxiliary estimators, indexed by r ∈ { 0 , 1 , . . . , 2 n -1 } and defined as follows. The estimator r , for the i -th sample, by having access to ( W,Z i, 0 , Z i, 1 ) estimates J i as

<!-- formula-not-decoded -->

Note that each of these estimators makes its estimations only by having access to ( W,Z i, 0 , Z i, 1 ) .

Define the Hamming distance d H : { 0 , 1 } n ×{ 0 , 1 } n → [ n ] between binary vectors J and ˆ J as

̸

We now compute the expectation of d H ( J , ˆ J ) for the r-th estimator, i.e., E W, ˜ S , J , ˆ J [ d H ( J , ˆ J ) ] . Note that due to the symmetry of ˜ S , we can only consider the case where J = (1 , 1 , . . . , 1) := 1 n .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, suppose by contradiction that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is not o ( n ) . This means that there exists some b 1 ∈ R + and a sequence { a i } i ∈ N such that lim i →∞ a i = ∞ and limiting n to this subsequence, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we use Fano's inequality with approximate recovery [59, Theorem 2]. Let t = 1 n ⌊ (1 -b 1 ) n 2 ⌋ and denote

<!-- formula-not-decoded -->

It is easy to not that N ˆ j is the same for all ˆ j ∈ { 0 , 1 } n . Hence, the maximum over ˆ j of N ˆ j is equal to N 1 n . With these notations, we have

<!-- formula-not-decoded -->

where ( a ) is by construction of W and as shown in the proof of Theorem 3, ( b ) is derived since ˆ J is a function of ( W, ˜ S ) , ( c ) is derived due to positivity of the mutual information, ( d ) is derived due to the below relations

<!-- formula-not-decoded -->

( e ) is derived using [59, Theorem 2], and ( f ) is derived using the claim, proved later below, that N 1 n ≤ c 3 2 nh b ( t ) for some constant c ∈ R + and for n sufficiently large.

Note that t = 1 n ⌊ (1 -b 1 ) n 2 ⌋ &lt; 1 / 2 and as n →∞ , 1 -h b ( t ) converges to the constant 1 -h b ( 1 -b 1 2 ) &gt; 0 . Hence, if we show that for sufficiently large n , 1 -P e t &gt; 1 -b 2 , for some constant b 2 ∈ (0 , 1) , the contradiction is achieved. Since the left-hand side is of order o ( n ) , which is greater than the right-hand side, which is Ω( n ) , and the proof is complete.

Hence, it remains to show for n sufficiently larg i) N 1 n ≤ c 3 2 nh b ( t ) for some constant c ∈ R + and ii) P e t &lt; b 2 , for some constant b 2 ∈ (0 , 1) .

Proof of Claim i) This is shown in equation 49.

Proof of Claim ii) Using Markov's inequality, we have

<!-- formula-not-decoded -->

Now, we have

<!-- formula-not-decoded -->

for some constant b 2 ∈ (1 / 2 , 1) and n sufficiently large (or a i sufficiently large).

This completes the proof of the lemma.

## H Proofs of Appendix D: Random subspace training algorithms

## H.1 Proof of Lemma 3

Part i. For a = 0 ,

<!-- formula-not-decoded -->

which is a standard Gaussian distribution. Hence, h ( g a,p ( x )) = log( √ 2 πe ) and f ( a, p ) = 0 .

Part ii. The relation f ( a, p ) = f ( -a, p ) is trivial since by the symmetry of the distribution g a,p . To show the increasing behavior with respect to a , consider 0 ≤ a ′ &lt; a and some p ∈ [0 , 1] . We show f ( a ′ , p ) &lt; f ( a, p ) . For a &gt; 0 , let

<!-- formula-not-decoded -->

where Y 1 ∼ N (0 , 1) is independent of J ∼ Bern ( p ) . Then, it is easy to verify that

<!-- formula-not-decoded -->

Now let σ ≜ √ ( a a ′ ) 2 -1 and define

<!-- formula-not-decoded -->

where Y 2 ∼ N ( 0 , σ 2 ) is independent of other random variables. Note that Y 3 ≜ a ′ ( Y 1 + Y 2 ) a is independent of J and distributed according to N (0 , 1) . Hence, we can write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) follows from equation 58, ( b ) from equation 59 and the strong data processing inequality, and ( c ) from equation 60. This completes the proof of the strictly increasing behavior with respect to a in the range [0 , ∞ ) .

Part iii. Denote Q 1 ( x ) := 1 √ 2 π e -x 2 2 and Q 2 ( x ) := 1 √ 2 π e -( x -a ) 2 2 . Note that g a,p ( x ) = pQ 1 ( x ) + (1 -p ) Q 2 ( x ) . Hence, h ( g a,p ( x )) = -p E Q 1 [log( g a,p ( x ))] -(1 -p ) E Q 2 [log( g a,p ( x ))] . Now, considering the limit to infinity, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) is deduced by noting that both Q 1 and Q 2 are Gaussian distributions with variance 1 and hence, their differential entropy is equal to 1 2 log(2 πe ) .

This concludes that lim a →∞ f ( a, p ) = h b ( p ) .

Part iv. f ( a, p ) = f ( a, 1 -p ) is trivial since by the symmetry of the distribution g a,p .

To show the strictly increasing behavior with respect to p , consider 0 ≤ p 1 &lt; p 2 ≤ 1 / 2 . Let

<!-- formula-not-decoded -->

where Y ∼ N (0 , 1) is independent of J 1 ∼ Bern ( p 1 ) . Then, due to Part ii,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let Z ∼ Bern ( q ) be independent of other random variables for some q ∈ (0 , 1) that will be determined later. Let

<!-- formula-not-decoded -->

where V = | J 1 -Z | . Note that V ∼ Bern ( p 1 q +(1 -p 1 )(1 -q )) is independent of Y .

Now, on the one hand, we have

<!-- formula-not-decoded -->

On the other hand,

<!-- formula-not-decoded -->

where ( a ) is derived by strong data processing inequality and since p 1 ∈ [0 , 1 / 2) and q ∈ (0 , 1) , ( b ) is derived for J ′ ∼ Bern (1 -p 1 ) independent of Y , and steps ( c ) , ( d ) , ( e ) are derived using equation 62.

Hence, combining equation 61, equation 63, and equation 63, we have

<!-- formula-not-decoded -->

The proof completes by find a q ∈ [0 , 1] such that p 1 q +(1 -p 1 )(1 -q ) = p 2 . To show that such q exist, first denote e p 1 ( q ) := p 1 q +(1 -p 1 )(1 -q ) . Now, note that e p 1 (1) = p 1 &lt; p 2 and e p 1 (0) = 1 -p 1 &gt; 1 2 ≥ p 2 . Hence, there exists a q ∗ ∈ (0 , 1) such that e p ( q ∗ ) = p 2 . This completes the proof of this part.

## H.2 Proof of Theorem 10

Recall that

Moreover, note that

<!-- formula-not-decoded -->

is the set of sample indices chosen at time t ∈ [ T ] , chosen independently of any other random variables. Hence,

<!-- formula-not-decoded -->

where A ( d ) V is the algorithm A ( d ) where the batch indices V = ( V 1 , . . . , V T ) are used.

The proof consists of bounding each of the conditional mutual information terms

<!-- formula-not-decoded -->

and then using the bound 15 of Corollary 2, with ˆ A ( d ) V = A ( d ) V and ϵ = 0 .

It is sufficient then to show that for a fixed V and every fixed i ∈ [ n ] , we have that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

For a fixed i ∈ [ n ] , if { t : i ∈ V t } is an empty set, then the final model is independent of J i and hence CMI Θ V ,i, J -i ( ˜ S , W ′ ) = 0 , which completes the proof. Now, assume that this set is not empty. For ease of notation, suppose that

<!-- formula-not-decoded -->

where 1 ≤ t 1 &lt; t 2 &lt; · · · &lt; t M ≤ T .

Then, for a fixed V ,

<!-- formula-not-decoded -->

where ( a ) holds since by the data processing inequality I ˜ S , J -i , Θ , V ( W ′ T ; J i ) ≤ I ˜ S , J -i , Θ , V ( W ′ t M ; J i ) and I ˜ S , J -i , Θ , V ( W ′ t M ; J i ) ≤ I ˜ S , J -i , Θ , V ( W ′ t M , W ′ t M -1 , W ′ t M -1 , W ′ t M -1 -1 , · · · , W ′ t 1 , W ′ t 1 -1 ; J i ) by the non-negativity of the mutual information, ( b ) is derived using the chain rule for the mutual information and by using the convention that when m = 1 , the conditioning part { W ′ t m -1 , W ′ t m -1 -1 , · · · , W ′ t 1 , W ′ t 1 -1 } is an empty set, and ( c ) is derived since I ˜ S , J -i , Θ , V ( W ′ t m -1 ; J i | , W ′ t m -1 , W ′ t m -1 -1 , · · · , W ′ t 1 , W ′ t 1 -1 ) = 0 .

Consider a fixed value of ( W ′ t m -1 , W ′ t m -1 , W ′ t m -1 -1 , · · · , W ′ t 1 , W ′ t 1 -1 ) and let

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

Hence, it is sufficient to show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that where ̂ R ( V t m , W ) ≜ 1 b ∑ i ′ ∈ V t m ℓ ( Z i ′ ,J i ′ , W ) . Denote

̸

<!-- formula-not-decoded -->

Furthermore, denote

<!-- formula-not-decoded -->

where the last line holds since by assumption i ∈ V t m .

Using the data processing inequality, we have that

<!-- formula-not-decoded -->

Hence, it is sufficient to show that

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

To compute each of the two terms in right-side of equation 67, first we derive the marginal and conditional distributions of 1 σ t m ˜ W t m .

- Given F m and given J i = 0 ,

<!-- formula-not-decoded -->

Hence, given F m and given J i = 0 , 1 σ t m ˜ W t m is distributed as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

- Similarly, given F m and given J i = 1 , 1 σ t m ˜ W t m is distributed as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

- Lastly, since P ( J i = 0 ∣ ∣ F m ) = p t m ,i , then given F m , 1 σ t m ˜ W t m is distributed as

<!-- formula-not-decoded -->

Now, we compute each of the two terms of h F m ( ˜ W t m /σ t m ) and h F m ( ˜ W t m /σ t m | J i ) :

- The term h F m ( ˜ W t m /σ t m ) equals the differential entropy h ( ˜ P ) . Since the differential entropy is invariant under the shift and since also the Gaussian distributions ˜ P 0 and ˜ P 1 are invariant under the rotation, h ( ˜ P ) is equal to the entropy of the distribution ˜ Q , defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

and

<!-- formula-not-decoded -->

Note that ∥ a d ∥ = ∥ µ 1 -µ 0 ∥ .

Furthermore, we can write

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and for r ∈ { 2 , 3 , . . . , d } ,

Hence,

<!-- formula-not-decoded -->

where ( a ) is derived by equation 70, ( b ) holds since the distributions Q 2 , . . . , Q d are scalar standard Gaussian distributions, ( c ) is derived for a 1 ≜ η t m bσ t m ∆ t m ,i and by the definition of g a,p ( · ) in 19, and ( d ) by the definition of f ( a, p ) in 18.

- To compute h F m ( ˜ W t m /σ t m | J i ) , note that for each value of J i , due to equation 68 and equation 69, the conditional distribution of 1 σ t m ˜ W t m is a multivariate Gaussian distribution with covariance I d . Hence,

<!-- formula-not-decoded -->

Combining equation 71 and equation 72 gives equation 66 which completes the proof.

## H.3 Proof of Theorem 13

Recall that

<!-- formula-not-decoded -->

where ̂ R ( V t , W ) ≜ 1 b ∑ i ∈ V t ℓ ( Z i,J i , W ) .

In the proof, to define the lossy compression algorithm P ˆ W | W ′ , Θ ,S of Corollary 2, we introduce auxiliary optimization iterations { ˆ W t } t ∈ [ T ] , as follows. Let ˆ W 0 = W ′ 0 , and for t ∈ [ T ] , let

<!-- formula-not-decoded -->

where ε ′ t ∼ N ( 0 d , I d ) is an additional noise, independent of all other random variables.

In the following Lemma, proved in Appendix H.4, we show that, this choice of P ˆ W | W ′ , Θ ,S, V satisfies the distortion term equation 16:

for

<!-- formula-not-decoded -->

Q

i

=

N

(0

,

1)

.

Lemma 6. The following inequalities holds:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Hence, it is sufficient to show that

<!-- formula-not-decoded -->

Note that the iterations defined in equation 73 are equivalent in distribution to the following iterations:

<!-- formula-not-decoded -->

where ˜ ε t ∼ N ( 0 d , I d ) is independent of all other random variables and

<!-- formula-not-decoded -->

Similar to the proof of Theorem 10, and by using Corollary 2, it is sufficient to show that for a fixed V and every fixed i ∈ [ n ] , we have that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

For a fixed i ∈ [ n ] , if { t : i ∈ V t } is an empty set, then the final model is independent of J i and hence CMI Θ V ,i, J -i ( ˜ S , ˆ W T ) = 0 , which completes the proof. Now, assume that this set is not empty. For ease of notation, suppose that

<!-- formula-not-decoded -->

where 1 ≤ t 1 &lt; t 2 &lt; · · · &lt; t M ≤ T .

We show by induction on m ∈ [ M ] that, we have

<!-- formula-not-decoded -->

where ˆ ∆ t,i and ˆ p t,i are defined as above and

<!-- formula-not-decoded -->

with the convention that A t ′ t,i = 1 for t ′ ≤ t .

Once this claim is shown, then we have

<!-- formula-not-decoded -->

where

- ( a ) is achieved by repeated using of [74, Lemma 4],
- ( b ) is derived using equation 74,
- and ( c ) holds by definitions of A t k ,i = A T t k ,i and A t M t k ,i .

Hence, it remains to show that equation 74 holds by induction.

Consider the base of the induction m = 1 . Note that A t 1 t 1 ,i = 1 . Hence, the result follows using the proof of Theorem 10; more precisely using equation 64 with W ′ → ˆ W t 1 , ∆ t,i → ˆ ∆ t,i , p t,i → ˆ p t,i , and σ t → ˆ σ t .

Now, suppose that the result holds for m = N ≤ M -1 , i.e., where

<!-- formula-not-decoded -->

We show that it also holds for m = N +1 ≤ M .

We have

<!-- formula-not-decoded -->

where

- ( a ) is derived using the proof of Theorem 10; more precisely using equation 65 with W ′ t m → ˆ W t N +1 , ∆ t m ,i → ˆ ∆ t N +1 ,i , p t m ,i → ˆ p t N +1 ,i , σ t m → ˆ σ t N +1 , and by considering

<!-- formula-not-decoded -->

- ( b ) is derived by repeated using of [74, Lemma 4],
- ( c ) holds by the assumption of the induction 75,
- and ( d ) by definition of A t N t,i and A t N +1 t,i .

This completes the proof of the theorem.

## H.4 Proof of Lemma 6

To prove the result, we show first what

<!-- formula-not-decoded -->

using induction over t ∈ [ T ] . Then, using the Lipschitzness property of the loss function, we have that

<!-- formula-not-decoded -->

where ( a ) is obtained using the fact that if Z ∼ N (0 , I d ) , then ∥ Z ∥ has a chi-distribution, whose mean is equal to √ 2 Γ(( d +1) / 2) Γ( d/ 2) .

For t = 1 ,

<!-- formula-not-decoded -->

where ( a ) is derived since for any w ′ 1 , w ′ 2 ∈ R d , ∥ ∥ Proj { w ′ 1 } -Proj { w ′ 2 }∥ ∥ ≤ ∥ ∥ w ′ 1 -w ′ 2 ∥ ∥ , by the contraction property of the projection. This shows the base of the induction.

Suppose that equation 76 holds for t = t ′ . Now, we show that it also holds for t = t ′ +1 .

<!-- formula-not-decoded -->

where ( a ) is derived using the triangle inequality, ( b ) using the contractility assumption, and ( c ) using the assumption of the induction. This completes the proof of the induction and the proof of the lemma.
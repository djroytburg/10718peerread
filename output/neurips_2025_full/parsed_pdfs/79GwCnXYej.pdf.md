## Tight Bounds for Answering Adaptively Chosen Concentrated Queries

## Emma Rapoport

Tel Aviv University emmarapoport@gmail.com

## Edith Cohen

Google Research and Tel Aviv University edith@cohenwang.com

## Abstract

Most work on adaptive data analysis assumes that samples in the dataset are independent. When correlations are allowed, even the non-adaptive setting can become intractable, unless some structural constraints are imposed. To address this, Bassily and Freund [2016] introduced the elegant framework of concentrated queries , which requires the analyst to restrict itself to queries that are concentrated around their expected value. While this assumption makes the problem trivial in the non-adaptive setting, in the adaptive setting it remains quite challenging. In fact, all known algorithms in this framework support significantly fewer queries than in the independent case: At most O ( n ) queries for a sample of size n , compared to O ( n 2 ) in the independent setting.

In this work, we prove that this utility gap is inherent under the current formulation of the concentrated queries framework, assuming some natural conditions on the algorithm. Additionally, we present a simplified version of the best-known algorithms that match our impossibility result.

## 1 Introduction

Adaptive interaction with data is a central feature of modern analysis pipelines, from scientific exploration to model selection and parameter tuning. However, adaptivity introduces fundamental statistical difficulties, as it creates dependencies between the data and the analysis procedures applied to it, which could quickly lead to overfitting and false discoveries. Motivated by this, following the seminal work of Dwork et al. [2015b], a substantial body of work has established rigorous frameworks for addressing this problem. These works demonstrated that various notions of algorithmic stability, and in particular differential privacy (DP) [Dwork et al., 2006], allow for methods which maintain statistical validity under the adaptive setting. Most of the current work, however, focuses on the case where the underlying data distribution is a product distribution , i.e., the samples in the dataset are independent of each other. Much less is understood about the feasibility of accurate adaptive analysis when the data exhibits correlations. In this work, we examine the extent to which accurate adaptive analysis remains possible under minimal structural conditions on the data distribution.

Before presenting our new results, we describe our setting more precisely. Let X be a data domain. We consider the following game between a data analyst A and a mechanism M .

1. The analyst A chooses a distribution D over tuples in X ∗ (under some restrictions).
3. For k rounds j = 1 , 2 , . . . , k :
2. The mechanism M obtains a sample S ← D . %Wedenote | S | = n .
4. (a) The analyst A chooses a query q j : X → [0 , 1] , possibly as a function of all previous answers given by M (under some restrictions).
5. (b) The mechanism M obtains q j and responds with an answer a j ∈ R , which is given to A .

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Uri Stemmer Tel Aviv University and Google Research u@uri.co.il

Note that the analyst A is adaptive in the sense that it chooses the queries q j based on previous outputs of M , which in turn depend on the sample S . So the queries q j themselves depend on S . If instead the analyst A were to fix all k queries before the game begins, then these queries would be independent of the dataset S . We refer to this variant of the game (where all queries are fixed ahead of time) as the non-adaptive setting .

The goal of M in this game is to produce accurate answers w.r.t. the expectation of the queries over the underlying distribution D . Formally, we say that M is ( α , β ) -statistically accurate if for every analyst A , with probability at least 1 -β , for every j ∈ [ k ] it holds that | a j -q j ( D ) | ≤ α , where q j ( D ) := E T ← D [ 1 | T | ∑ x ∈ T q j ( x ) ] . As a way of dealing with worst-case analysts, the analyst A is assumed to be adversarial in that it tries to cause the mechanism to fail. We therefore sometimes think of A as an attacker .

The main question here is:

Question 1.1. What is the maximal number of queries one can accurately answer, k , as a function of the sample size n , the desired utility parameters α , β , and the type of restrictions we place on the choice of D and the queries q j (in Steps 1 and 3a above)?

The vast majority of the work on adaptive data analysis (ADA) focuses on the case where D is restricted to be a product distribution over n elements (without restricting the choice of the queries q j ). After a decade of research, this setting is relatively well-understood: For constant α , β , there exist computationally efficient mechanisms that can answer Θ ( n 2 ) adaptive queries, and no efficient mechanism can answer more than that. 1

The situation is far less well-understood when correlations in the data are possible. Let us consider the following toy example as a warmup. Suppose that the attacker randomly picks one of the following two distributions:

- D 0 = The distribution that with probability 1 / 2 returns the sample ( 1 2 , 1 2 , . . . , 1 2 ) and w.p. 1 / 2 returns the sample (0 , 0 , . . . , 0) .
- D 1 = The distribution that with probability 1 / 2 returns the sample ( 1 2 , 1 2 , . . . , 1 2 ) and w.p. 1 / 2 returns the sample (1 , 1 , . . . , 1) .

Note that in this scenario, a mechanism holding the sample ( 1 2 , 1 2 , . . . , 1 2 ) cannot accurately answer the query q ( x ) = x , as the true answer could be either 1 / 4 or 3 / 4 . The takeaway from this toy example is that when correlations in the data are possible, then we must impose additional restrictions on our setting in order to make it feasible. There are two main approaches for this in the literature:

1. Explicitly limit dependencies within the sample [Kontorovich et al., 2022]: Intuitively, if we restrict the attacker A to choose only distributions D that adhere to certain 'limited dependencies' assumptions, then the problem becomes feasible. A downside of this approach is that it is typically tied to a specific measure for limiting dependencies, and it is not clear why one should prefer one measure over another.
2. Limit the attacker to concentrated queries [Bassily and Freund, 2016]: Notice that the toy example above cannot be solved even in the non-adaptive setting, because the description of the hard query q ( x ) = x does not depend on the sample S . So in a sense, it is 'unfair' to attempt solving it in the adaptive setting. In other words, if something cannot be solved in the non-adaptive setting, how can we hope to solve it in the adaptive setting? Motivated by this, Bassily and Freund [2016] restricted the attacker A to queries that in the non-adaptive setting are sharply concentrated around their true mean. Specifically, the attacker is restricted to choose queries q j such that if we were to sample a fresh dataset T from the underlying distribution D (where T is independent of the description of q j ), then with high probability it holds that the empirical average of q j on T is close to the true mean of q j over D . Notice that under this restriction, the problem becomes trivial in the non-adaptive setting, as we could simply answer each query using its exact empirical average. In the adaptive setting, however, the problem is quite challenging.

1 See, e.g., Hardt and Ullman [2014], Dwork et al. [2015b,a], Steinke and Ullman [2015], Bassily et al. [2016], Cummings et al. [2016], Rogers et al. [2016], Feldman and Steinke [2017], Nissim et al. [2018], Feldman and Steinke [2018], Shenfeld and Ligett [2019], Steinke and Zakynthinou [2020], Jung et al. [2020], Shenfeld and Ligett [2023], Blanc [2023], Nissim et al. [2023]

In this work we continue the study of this question for concentrated queries. We aim to characterize the largest number of adaptively-chosen concentrated-queries one can accurately answer (without assuming independence in the data). Formally,

Definition 1.2 (Concentrated queries) . Let X be a domain, let D be a distribution over tuples in X ∗ , and let ε , γ ∈ [0 , 1] be parameters. A query q : X → [0 , 1] is ( ε , γ ) -concentrated w.r.t. D if

<!-- formula-not-decoded -->

where q ( S ) = 1 | S | ∑ x ∈ S q ( x ) is the empirical average of q on S and q ( D ) = E T ← D [ q ( T )] is the expected value of q over sampling a fresh dataset from D .

For example, if D is a product distribution over datasets of size n , then, by the Hoeffding bound, every query q : X → [0 , 1] is ( ε , γ ) -concentrated for every ε ≥ √ ln 2 2 n with γ = 2 e -2 n ε 2 . This example motivates the following question:

Question 1.3. How many adaptive queries could we efficiently answer when correlations in the data are allowed, but the analyst is restricted to ( ε , γ ) -concentrated queries for ε , γ that are comparable to what is guaranteed without correlations, say ε = 1 √ n and γ = n -10 ?

Bassily and Freund [2016] introduced this question and presented noise addition mechanisms that can efficiently answer O ( n ) adaptive queries under the conditions of Question 1.3. By noise addition mechanism we mean a mechanism that given a sample S answers every query q with q ( S ) + η , where η is drawn independently from a fixed noise distribution. Note the stark contrast from the i.i.d. case, where it is known that O ( n 2 ) queries can be supported rather than only O ( n ) . To achieve their results, Bassily and Freund [2016] introduced a stability notion called typical stability and showed that (1) noise addition mechanisms with appropriate noise are typically stable; and (2) typical stability guarantees statistical validity in the adaptive setting, even in the face of correlations in the data. More generally, the algorithm of Bassily and Freund [2016] can support roughly ˜ O ( 1 ε 2 ) queries provided that γ is mildly small (polynomially small in k ).

Following that, Kontorovich et al. [2022] showed that a qualitatively similar result could be obtained via compression arguments (instead of typical stability). However, their (computationally efficient) algorithms require γ to be exponentially small in k and thus do not apply to the parameters ε , γ stated in Question 1.3. They do support other ranges for ( ε , γ ) , but at any case their efficient algorithms cannot answer more than O ( n ) queries when the parameters ( ε , γ ) adhere to the behavior of Hoeffding's inequality for i.i.d. samples. For example, for ε = O (1) and γ = 2 -Ω ( n ) their algorithm supports O ( n ) adaptive queries to within constant accuracy (even when there are correlations in the data).

To summarize, currently there are two existing techniques for answering adaptively chosen concentrated queries: Either via typical stability in the small ε regime or via compression arguments in the tiny γ regime . At any case, all known results do not break the O ( n ) queries barrier, even when the concentration parameters reflect the behavior guaranteed in the i.i.d. setting. In contrast, without correlations in the data, answering O ( n 2 ) queries is possible.

## 1.1 Our results

## 1.1.1 An impossibility result for answering concentrated queries

We establish a new negative result providing strong evidence that the linear barrier discussed above is inherent. Our result applies to mechanisms that perturb the empirical mean of a query either by adding independent noise or by evaluating it on a randomly selected subsample of the dataset, which together constitute all known efficient, polynomial-time techniques for answering a super-linear number of queries in the i.i.d. setting. For brevity, we refer to these as Noise and Subsampling (NS) mechanisms. Specifically, we show that NS mechanisms cannot answer more than O ( n ) adaptively chosen concentrated queries, even if the query concentration matches the behavior of Hoeffding's inequality for i.i.d. samples. This constitutes the first negative result for answering adaptively chosen concentrated queries, and stands in sharp contrast to the O ( n 2 ) achievable in the i.i.d. setting. Specifically,

Theorem 1.4 (informal) . Let ε &gt; 0 and γ ∈ (0 , 1] . Then there exists a domain X and a distribution D over X n such that the following holds. For any NS mechanism M there exists an adaptive analyst

issuing ( ε , γ ) -concentrated queries q 1 , . . . , q k , where k = Ω ( min { 1 γ , 1 ε 2 ln ( 1 ε · γ )}) , such that with probability at least 0 . 9 there is a query q i for which the answer provided by M deviates from its true mean q i ( D ) by at least 0.9.

To interpret this result, note for example that when ε = O (1) and γ = 2 -n then our bound on k gives k = O ( n ) . We show that the same is true for all values of ( ε , γ ) that match the behavior of the Hoeffding bound in the i.i.d. setting.

This negative result emphasizes a fundamental limitation. In order to break the linear barrier on the number of supported queries, future work must either impose additional structural assumptions on the problem or introduce new algorithmic techniques beyond noise addition and subsampling mechanisms.

## 1.1.2 A simplified positive result

As we mentioned, Bassily and Freund [2016] introduced the notion of typical stability and leveraged it to design algorithms supporting adaptive concentrated queries. However, their definitions and techniques are quite complex. In particular, bounding the number of supported queries k as a function of the concentration parameters ε and γ is not easily extractable from their theorems.

We present a significantly simpler analysis for their algorithm that does not use typical stability at all. Instead, it relies on techniques from differential privacy Dwork et al. [2006]. In addition to being simpler, our analysis allows us to save logarithmic factors in the resulting bounds on k . Formally, we show the following theorem.

Theorem 1.5 (informal) . Fix parameters ε , γ . There exists a noise addition mechanism M that guarantees ( 1 , 1 ) -statistical accuracy against any analyst issuing at most k queries which are

<!-- formula-not-decoded -->

100 100 A ( { }) .

In retrospect, leveraging differential privacy (DP) to answer concentrated queries (as we do in this work) is a natural approach as it is simpler than prior work on this topic and aligns with other works on other variants of the ADA problem. In a sense, the reason for the additional complexity in the work of Bassily and Freund [2016] steams from their alternative stability notion (typical stability). To the best of our knowledge, our work is the first to derive meaningful positive results for answering adaptively chosen concentrated queries via differential privacy when correlations are present in the data .

## 1.1.3 Technical overview of our negative result (informal)

The key insight underlying our negative result is that query concentration alone does not prevent an attacker from extracting substantial information about correlated data. We consider a domain X partitioned into 1 ε subsets, and define a distribution D over X n in which each sample consists of 1 ε distinct elements, each drawn from a different subset and repeated ε n times.

This structure simultaneously maximizes the information each query can reveal while ensuring that every query remains tightly concentrated. The attacker designs each query to assign nonzero values only within a single targeted subset, keeping the empirical mean within [0 , ε ] and satisfying ( ε , γ ) -concentration by construction. Yet, the responses still leak significant information about the data.

Building on the adaptive attack of Nissim et al. [2018] for the i.i.d. setting, our attacker progressively identifies the repeated elements: each query randomly assigns binary values within the targeted subset and updates an accumulated score to isolate the correct element. We present a simple analysis adapted to our setting, showing that the sample can be recovered with high probability. This breaks the accuracy guarantee of any NS mechanism once the number of queries exceeds our derived lower bound.

Our construction highlights that when correlations are present, concentration alone cannot prevent information leakage. Thus, accurately answering more than a linear number of adaptive, concentrated queries requires stronger structural assumptions on the distribution.

## 1.1.4 Technical overview of our positive result (informal)

We prove Theorem 1.5 by showing that the mechanism that answers queries with their noisy empirical average guarantees statistical accuracy (for an appropriately calibrated noise distribution). To show this, we introduce a thought experiment involving three mechanisms, all initialized with the same sample S ∼ D , all interacting with the same analyst A :

- Real-world mechanism: Answers each query using the empirical mean plus independent noise. This is the mechanism whose accuracy we want to analyze.
- Oracle mechanism: Answers each query using its true mean under the target distribution D , plus independent noise. Note that this mechanism 'knows' the target distribution D . This is not a real mechanism; it only exists as part of our proof. The noise magnitude will be small enough such that this mechanism remain accurate.
- Hybrid mechanism: Initially behaves like the real-world mechanism, but switches permanently to behave like the oracle mechanism if at some point the empirical mean on any query deviates significantly from its true mean. This is also not a real mechanism; it exists only as part of our proof.

Our analysis proceeds in two steps. First, we leverage techniques from differential privacy to demonstrate that the output distributions of the the oracle and hybrid mechanisms are close. This allows us to invoke advanced-composition-like theorems from differential privacy, ensuring that the outcome distributions of these two mechanisms remain close even after k adaptive queries.

Second, we identify a class of good interactions . In these scenarios, the hybrid mechanism never switches to oracle responses, making its behavior identical to the real-world mechanism. We show that these good interactions occur with high probability under the oracle mechanism, and by extension, under the hybrid mechanism. We thus get that the real-world mechanism is also likely to produce these good interactions.

By combining these insights, we conclude that, under suitable concentration assumptions on the queries, the real-world mechanism's outputs closely track those of the oracle mechanism, which is statistically accurate by definition, thus ensuring statistical accuracy even in adaptive settings.

## 2 Preliminaries

We now formally define the two classes of mechanisms for which our negative result holds, which we refer to collectively as Noise and Subsampling (NS) mechanisms.

Definition 2.1 (Noise-Addition Mechanism) . A mechanism M is a noise-addition mechanism if, given a dataset S and a statistical query q , it returns a = q ( S ) + η , where η is a random variable drawn independently from a fixed, zero-mean noise distribution that does not depend on S or q .

Definition 2.2 (Subsampling Mechanism) . A mechanism M is a subsampling mechanism if, given a dataset S of size n , it answers each query q as follows. For each round independently, the mechanism samples a subsample S ′ of size m by drawing m elements from S independently and uniformly at random with replacement , and returns a = q ( S ′ ) = 1 m ∑ x ∈ S ′ q ( x ) , the empirical mean of q on the subsample. The subsample S ′ is freshly resampled for every query, independently of previous rounds.

Differential privacy. Consider an algorithm that operates on a dataset. Differential privacy is a stability notion requiring the (outcome distribution of the) algorithm to be insensitive to changing one example in the dataset. Formally,

Definition 2.3 (Dwork et al. [2006]) . Let M be a randomized algorithm whose input is a dataset. Algorithm M is ( ε , δ ) -differentially private (DP) if for any two datasets S, S ′ that differ in one point (such datasets are called neighboring ) and for any event E it holds that Pr[ M ( S ) ∈ E ] ≤ e ε · Pr[ M ( S ′ ) ∈ E ] + δ .

The most basic constructions of differentially private algorithms are via the Laplace mechanism as follows.

Definition 2.4 (The Laplace Distribution) . A random variable has probability distribution Lap( b ) if its probability density function is f ( x ) = 1 2 b exp ( -| x | b ) , where x ∈ R .

Theorem 2.5 (Dwork et al. [2006]) . Let f be a function that maps datasets to the reals with sensitivity /lscript (i.e., for any neighboring S, S ′ we have | f ( S ) -f ( S ′ ) | ≤ /lscript ). The mechanism M that on input S adds noise with distribution Lap( /lscript ε ) to the output of f ( S ) preserves ( ε , 0) -differential privacy.

Finite precision and bounded outputs Real-world computing devices can only produce finitely many bits of precision. Accordingly, we assume outputs are rounded or truncated, ensuring a discrete output space. In our negative result, we additionally assume that outputs lie within a fixed bounded range. This holds automatically for subsampling mechanisms, since the empirical mean of values in [0 , 1] always lies in [0 , 1] . For noise-addition mechanisms, we assume outputs lie in the interval [ -1 , 2] . Since queries are bounded in [0 , 1] and the accuracy parameter α is also in [0 , 1] , any response outside this range would already violate the accuracy guarantee, meaning the mechanism has failed and the attack has succeeded.

## 3 An impossibility result for answering concentrated queries

We begin by noting that the bound of 1 / γ queries is unavoidable. To see this, consider a distribution D that is uniform over 1 / γ disjoint samples S 1 , . . . , S 1 / γ of size n each. Now consider the analyst that queries (one by one) all 1 / γ queries of the form q i ( x ) = 1 if x ∈ S i and q i ( x ) = 0 otherwise. The 'true mean' of each of these queries over D is exactly γ , and each of them, the probability of deviating from this true mean by more than γ (over sampling S ∼ D ) is at most γ . So for γ &lt; ε and α &lt; 1 -γ these queries are all concentrated, and one of them causes the mechanism to lose accuracy. See Appendix A.1 for the formal details.

The main result of this section gives a stronger impossibility bound. We construct a distribution and domain such that any NS mechanism can be forced to fail with probability 1 -β by an attacker issuing only ( ε , γ ) -concentrated queries after k = Ω ( 1 ε 2 · ln ( 1 ε · β · γ )) rounds.

We then consider the setting where γ is a function of n and ε that corresponds to the concentration of bounded queries in an i.i.d. regime. Specifically, if γ ( n, ε ) = 2exp( -2 ε 2 n ) , as given by the double-sided Hoeffding inequality, then the two combined bounds imply that no NS mechanism can answer more than O ( n ) such queries.

Domain and distribution. We now formalize the domain and sample construction described above. Let ε ∈ (0 , 1] and define r = 1 / ε . Let X be a finite domain of size N = max { 1 ε 2 , 1 ε γ } . Assume for simplicity that r is an integer and that it divides both N and n . Partition X into r disjoint subsets X 1 , . . . , X r , each of size ε N . Label the elements in each subset arbitrarily: X i = { x 1 i , x 2 i , . . . , x ε i N } for all i = 1 , . . . , r .

The distribution D over X n is defined as follows. First, sample an index j ∼ Unif { 1 , 2 , . . . , ε N } . Then, output the sample S = ( x j 1 , . . . , x j 1 ︸ ︷︷ ︸ ε n , x j 2 , . . . , x j 2 ︸ ︷︷ ︸ ε n , . . . , x j r , . . . , x j r ︸ ︷︷ ︸ ε n ) . That is, a single index j determines one point x j i from each subset X i , and the sample consists of each of these points repeated exactly n/r = ε n times. Although D is defined on X n , its support contains only N/r = ε N distinct samples, and each is determined by the shared index j .

Attack Overview: The attack procedure (Algorithm 1) operates over k rounds of information gathering followed by a single final query. During the information gathering rounds, each query q t is constructed using i.i.d. Bernoulli random variables: each element x j 1 in the targeted subset X 1 independently takes the value 1 with probability p t ∼ Unif[0 , 1] , while all other elements in the domain receive value 0. Throughout the interaction, we track an accumulated score Z j for each element x j 1 , defined so that the score increment z j t has positive expectation if x j 1 matches the unique element appearing in the true sample, and zero expectation otherwise. After all k rounds, we identify the element with the highest cumulative score. By standard concentration arguments, this element is likely to be the one present in the actual sample, as it uniquely accumulates a positive expected score. We issue a final query that evaluates to 1 for the elements in the sample we identified and 0 for all other elements, thus pinpointing the true sample. Throughout the analysis, we assume

α &lt; 1 -1 | Supp( D ) | , ensuring that if the final query successfully identifies the true sample, the resulting deviation exceeds α .

## Algorithm 1 Attack Procedure

Initialization: Let M be an NS mechanism initialized with a sample S ∼ D . For each element x j 1 ∈ X 1 , initialize an accumulated score Z j = 0 .

1. Sample p t ∼ Unif[0 , 1] .

Information gathering rounds: For each round t ∈ [ k ] :

2. Define the query q t : X → { 0 , 1 } as:

<!-- formula-not-decoded -->

3. Submit q t to the mechanism and receive the response a t .
4. For each x j 1 ∈ X 1 , define z j t = ( a t -p t /r ) ( q t ( x j 1 ) -p t ) , and update Z j ← Z j + z j t .

<!-- formula-not-decoded -->

Final query: After k rounds, compute j ∗ = arg max j Z j . Submit a final query q ∗ : X → { 0 , 1 } by setting

Analysis of the attack. We now prove that the attack described above succeeds using only ( ε , γ ) -concentrated queries. This analysis establishes three components: (1) all k information-gathering queries are ( ε , γ ) -concentrated, (2) the final attack query q ∗ is also ( ε , γ ) -concentrated, and (3) with high probability, the attack correctly identifies the underlying sample. The proofs for the concentration of the queries are deferred to Appendix A.2.

<!-- formula-not-decoded -->

Proof Sketch. Consider the variables z j t = ( a t -p t r )( q t ( x j 1 ) -p t ) . It can be shown (Appendix A.3.1) that these satisfy E [ z j t ] = 1 6 r if j corresponds to the true sample, and 0 otherwise. Summing these variables, define Z j = ∑ k t =1 z j t . Each Z j thus accumulates a positive expectation only for the true index j s , and zero otherwise. Using standard concentration inequalities, one obtains that the index maximizing Z j coincides with the true sample with probability at least 1 -β , provided k satisfies the stated bound. The detailed argument is deferred to Appendix A.3.

Combining Theorem 3.1 and the attack in Appendix A.1, we get:

Theorem 3.2. Let ε &gt; 0 , γ ∈ (0 , 1] , and β ∈ (0 , 1) . Then there exists a domain X and a distribution D over X n such that for any NS mechanism M , and any α ∈ ( 0 , 1 -1 | Supp( D ) | ) , there exists an analyst issuing k adaptive ( ε , γ ) -concentrated queries with k = Ω ( min { 1 γ , 1 ε 2 ln ( 1 ε · β · γ )}) , such that Pr[ ∃ i ∈ [ k ] such that | M ( q i ) -q i ( D ) | &gt; α ] ≥ 1 -β .

Comparison to the i.i.d. setting We now compare our query bound to the classical i.i.d. setting, where differentially private mechanisms can answer up to O ( n 2 ) adaptive statistical queries with bounded error. In sharp contrast, our results imply a strong negative statement: even if the query concentration matches the behavior of Hoeffding's inequality for i.i.d. samples, the number of accurately answerable queries by NS mechanisms is tightly bounded by O ( n ) . To make this comparison precise, we assume a fixed failure probability (e.g. β = 0 . 01 ), and let the concentration rate γ ( n, ε ) follow the double-sided Hoeffding bound: γ ( n, ε ) = 2 exp( -2 n ε 2 ) . Under this assumption, the bound from theorem 3.2 simplifies to k = O ( n ) for any ε ∈ (0 , 1] . The full derivation is a straightforward asymptotic calculation, deferred to Appendix A.4.

## 4 A simplified positive result

## 4.1 Setup and definitions

We introduce a thought experiment involving three mechanisms, all initialized with the same sample S ∼ D , all interacting with the same adaptive analyst A over k rounds.

Asample S ∼ D is drawn once and remains hidden from the analyst. The analyst A issues a sequence of k statistical queries q 1 , . . . , q k : X → [0 , 1] , where each query may depend adaptively on previous queries and responses; A is assumed to be deterministic without loss of generality, as randomized analysts can be treated by taking expectation over a distribution of deterministic strategies.

The mechanism's responses are based on one of three strategies: the real-world mechanism adds Laplace noise to the empirical mean M S ( q i ) = q i ( S ) + η i , where η i ∼ Laplace(0 , b ) ; the oracle mechanism adds Laplace noise to the true mean M O ( q i ) = q i ( D ) + η ′ i , where η ′ i ∼ Laplace(0 , b ) ; and the hybrid mechanism responds as the real-world mechanism while all past queries are ε -concentrated relative to S , but switches to the oracle mechanism once any empirical mean deviates

To describe the interaction between the analyst and the mechanism, we define the transcript t = ( q 1 , a 1 , . . . , q k , a k ) , which records the sequence of queries and their corresponding responses. Each answer a i is given by either M S ( q i ) , M O ( q i ) , or M H ( q i ) , depending on the mechanism being used in the interaction.

by more than ε from the true mean: M H ( q i ) = { q i ( S ) + η i , if max j ≤ i ∣ ∣ ˆ q j ( S ) -q j ( D ) ∣ ∣ ≤ ε , q i ( D ) + η ′ i , otherwise .

Definition 4.1 (Transcript) . Let A be an analyst interacting with a mechanism over k rounds. The random transcript T is the sequence of queries and responses generated in the interaction. A particular outcome is denoted by t = ( q 1 , a 1 , . . . , q k , a k ) .

Transcript Probability Notation. Let t be a transcript arising from an interaction between an analyst A and a mechanism M . We denote the probability of t arising under mechanism M as Pr M ( T = t ) , where Pr M S ( T = t ) , Pr M O ( T = t ) , and Pr M H ( T = t ) refer to the probabilities under the real-world, oracle, and hybrid mechanisms, respectively.

To analyze the outcome of the interaction, we define two categories of "good" events: (1) Statistical accuracy : This event contains all transcripts t such that all answers in t are close to the true means of their respective queries. (2) Sample concentration : This event contains all pairs of transcripts t and samples S such that the empirical mean on S of each query in t is close to its true mean on D . Note that statistical accuracy is a property of the mechanism's outputs and their deviation from the true means, independent of the sample; sample concentration , by contrast, depends on both transcript and sample as it reflects how well the empirical means align with the true expectations.

Definition 4.2 ( α -accurate transcript) . A transcript t = ( q 1 , a 1 , . . . , q k , a k ) is α -accurate if every response a i is within α of the true mean q i ( D ) ; that is: | a i -q i ( D ) | ≤ α for all i ∈ [ k ] .

Definition 4.3 ( ε -good pair ( S, t ) ) . Let S ∈ Supp( D ) be a sample and let t = ( q 1 , a 1 , . . . , q k , a k ) be a transcript of k queries and responses. The pair ( S, t ) is called ε -good if, for every query q i in t , the empirical mean of q i on S is close to its true mean: ∣ ∣ q i ( S ) -q i ( D ) ∣ ∣ ≤ ε .

Our strategy involves demonstrating that the probability of sample concentration events occurring is similar across the real-world, oracle, and hybrid mechanisms. By establishing this, we can infer that events satisfying both statistical accuracy and sample concentration-which occur with high probability under the oracle mechanism-also occur with high probability under the real-world mechanism. Thus ensuring that the real-world mechanism maintains statistical accuracy despite the adaptivity of the analyst.

## 4.2 Relating the distribution of events under the oracle and hybrid mechanisms

The first component of our analysis shows that the output distributions of the oracle and hybrid mechanisms are closely aligned, similarly to the guarantees provided by ( ε , 0) -differential privacy for neighboring datasets. This is formalized in Lemma 4.4, with the proof deferred to Appendix B.1.

Lemma 4.4. Let S ∈ Supp( D ) be a sample, and let q be any query. Then for every measurable set in the output space E ⊆ Y : Pr [ M H ( q ) ∈ E ] ≤ e ε b Pr [ M O ( q ) ∈ E ] and Pr [ M O ( q ) ∈ E ] ≤ e ε b Pr [ M H ( q ) ∈ E ] .

Extending advanced composition to our setting. The preceding lemma allows us to extend the advanced composition analysis of Dwork et al. [2010b] (see also Dwork and Roth [2014]) to our framework. One of their results shows that if an ( ε , 0) -differentially private mechanism interacts with an analyst over k rounds, then for any δ ′ &gt; 0 , the overall interaction is ( ε ∗ , δ ′ ) -differentially private, where ε ∗ ≈ √ k ε . Although this theorem is framed in terms of differential privacy and neighboring datasets, the proof relies solely on the following: in each round, the conditional distributions of the outputs in two parallel experiments Y and Z given identical histories up to the previous round satisfy that for any E ⊆ Supp( Y ) , it holds that ln ( Pr[ Y ∈ E ] Pr[ Z ∈ E ] ) ≤ ε , and similarly for the reverse ratio. In our setting, Lemma 4.4 implies that this condition holds for any interaction of a fixed analyst with the oracle and hybrid mechanisms once you condition on identical histories. We now formalize the corresponding composition theorem in our framework and, for completeness, supply a full proof in appendix B.2 that mimics the proof of [Dwork et al., 2010b, Theorem III.1] to demonstrate that it applies under our conditions.

Theorem 4.5. Let S ∈ Supp( D ) be a sample, and let A be a fixed analyst. Consider two k -round interactions with A : one with the hybrid mechanism M H , and one with the oracle mechanism M O .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From this, we conclude the following corollary (the proof is deferred to Appendix B.3):

Corollary 4.6. Let A be a fixed analyst, S ∈ Supp( D ) a sample, and let E be any event that can arise in the interaction with the analyst. For any ρ &gt; 0 , define ε ∗ as in Theorem 4.5. Then e -ε ∗ (Pr M O [ E ] -ρ ) ≤ Pr M H [ E ] ≤ e ε ∗ Pr M O [ E ] + ρ .

## 4.3 High-probability accuracy of the Laplace mechanism

The next lemma shows that the real-world and hybrid mechanisms assign equal probability to any event consisting entirely of ε -good sample-transcript pairs. This follows from the fact that both mechanisms operate with the same randomness, so as long as all queries remain ε -good, their responses are identical. Full details are provided in Appendix B.4.

Lemma 4.7. Fix an analyst A , and let G be the set of ε -good sample-transcript pairs. Then, for every measurable subset E ⊆ G , Pr M S [ E ] = Pr M H [ E ] .

The following lemma provides a high-probability guarantee for ε -good and α -accurate transcripts under the oracle mechanism. Since the output is independent of the sample, a union bound over queries and noise yields this result, with the full argument deferred to Appendix B.5.

Lemma 4.8. Let S ∼ D and consider a k -round interaction between an analyst A and the oracle mechanism, producing the transcript t . Define the failure probability of any of the Laplace noises exceeding α as ζ = 1 -Pr[ | η ′ 1 | ≤ α ] k , for η ′ 1 ∼ Laplace(0 , b ) . Then,

<!-- formula-not-decoded -->

We have established that: (1) the transcript distribution under the hybrid mechanism closely approximates that of the oracle mechanism, (2) the probability of any event consisting of ε -good sample-transcript pairs is identical under both the real-world and hybrid mechanisms, and (3) under the oracle mechanism, the joint event of ε -good pairs and α -accuracy occurs with high probability. Combining these facts implies that the real-world mechanism is statistically accurate with high probability. This is formalized in the following theorem, the proof of which is deferred to Appendix B.6.

Theorem 4.9. Let A be an analyst, and M S the real-world Laplace mechanism interacting with A over k rounds. For α &gt; 0 and ρ &gt; 0 , define ε ∗ as in theorem 4.5, and let ζ be as in lemma 4.8. Then, the probability that the real-world mechanism produces an α -accurate transcript satisfies

<!-- formula-not-decoded -->

Theorem 4.10. Let A be any analyst issuing k adaptive ( ε , γ ) -concentrated queries, and fix an accuracy parameter α &gt; 0 and failure probability β &gt; 0 . Then the Laplace mechanism can achieve ( α , β ) -accuracy over all k queries provided k = O ( min { β γ , β ε -2 , α 2 β 2 ε 2 [ln(1 / ε )] 2 ln(1 / β ) }) .

Proof. Run the Laplace mechanism with noise scale b = α 2 ln(1 / ε ) . Theorem 4.9 implies that for any fixed α &gt; 0 and number of queries k , the real-world mechanism satisfies ( α , β ) -accuracy provided e -ε ∗ · ( 1 -k γ -ζ -ρ ) ≥ 1 -β . Requiring each term in the failure probability ρ , ζ , k γ and ε ∗ to be ≤ β / 4 , yields the desired result. See Appendix B.7 for the details of this derivation.

Simplified bound for constant accuracy and failure (example). If we assume constant parameters for failure probability and accuracy with ε &lt; α (e.g., α = β = 0 . 01 ), Theorem 4.10 implies the existence of a noise addition mechanism M that guarantees (0 . 01 , 0 . 01) -statistical accuracy against any analyst A issuing up to k adaptive, ( ε , γ ) -concentrated queries, provided

<!-- formula-not-decoded -->

## Acknowledgments and Disclosure of Funding

Emma Rapoport is partially supported by the Israel Science Foundation (grant 1419/24), the Blavatnik Family Foundation and the Deutsch Foundation. Edith Cohen is partially supported by Israel Science Foundation, (grant 1156/23). Uri Stemmer is partially supported by the Israel Science Foundation (grant 1419/24) and the Blavatnik Family Foundation.

## References

- Kazuoki Azuma. Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, Second Series , 19(3):357-367, 1967.
- Raef Bassily and Yoav Freund. Typicality-based stability and privacy. CoRR , abs/1604.03336, 2016.
- Raef Bassily, Kobbi Nissim, Adam D. Smith, Thomas Steinke, Uri Stemmer, and Jonathan R. Ullman. Algorithmic stability for adaptive data analysis. In STOC , 2016.
- Guy Blanc. Subsampling suffices for adaptive data analysis. In STOC , 2023.
- Rachel Cummings, Katrina Ligett, Kobbi Nissim, Aaron Roth, and Zhiwei Steven Wu. Adaptive learning with robust generalization guarantees. In COLT , 2016.
- Cynthia Dwork and Aaron Roth. The Algorithmic Foundations of Differential Privacy . Now Publishers, 2014.
- Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam D. Smith. Calibrating noise to sensitivity in private data analysis. In TCC , 2006.
- Cynthia Dwork, Guy N Rothblum, and Salil Vadhan. Boosting and differential privacy. In 2010 IEEE 51st annual symposium on foundations of computer science , pages 51-60. IEEE, 2010a.
- Cynthia Dwork, Guy N. Rothblum, and Salil P. Vadhan. Boosting and differential privacy. In FOCS , 2010b.
- Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Roth. Generalization in adaptive data analysis and holdout reuse. In NIPS , 2015a.

- Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Aaron Leon Roth. Preserving statistical validity in adaptive data analysis. In STOC , 2015b.
- Vitaly Feldman and Thomas Steinke. Generalization for adaptively-chosen estimators via stable median. In COLT , 2017.
- Vitaly Feldman and Thomas Steinke. Calibrating noise to variance in adaptive data analysis. In COLT , 2018.
- Moritz Hardt and Jonathan Ullman. Preventing false discovery in interactive data analysis is hard. In FOCS , 2014.
- Christopher Jung, Katrina Ligett, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, and Moshe Shenfeld. A new analysis of differential privacy's generalization guarantees. In ITCS , 2020.
- Aryeh Kontorovich, Menachem Sadigurschi, and Uri Stemmer. Adaptive data analysis with correlated observations. In ICML , 2022.
- Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics , 22(1):79-86, 1951.
- Kobbi Nissim, Adam D. Smith, Thomas Steinke, Uri Stemmer, and Jonathan R. Ullman. The limits of post-selection generalization. In NeurIPS , 2018.
- Kobbi Nissim, Uri Stemmer, and Eliad Tsfadia. Adaptive data analysis in a balanced adversarial model. NeurIPS , 2023.
- Ryan Rogers, Aaron Roth, Adam Smith, and Om Thakkar. Max-information, differential privacy, and post-selection hypothesis testing. In FOCS , 2016.
- Moshe Shenfeld and Katrina Ligett. A necessary and sufficient stability notion for adaptive generalization. NeurIPS , 2019.
- Moshe Shenfeld and Katrina Ligett. Generalization in the face of adaptivity: A bayesian perspective. NeurIPS , 2023.
- Thomas Steinke and Jonathan Ullman. Interactive fingerprinting codes and the hardness of preventing false discovery. In COLT , 2015.
- Thomas Steinke and Lydia Zakynthinou. Reasoning about generalization via conditional mutual information. In COLT , 2020.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction provide accurate summaries of the major claims made in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of the results are captured in the formal theorem statements, and we emphasize the differences between our setting an other settings studied in prior work.

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

Justification: We fully prove all of our theorems.

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

Justification: The paper and its underlying research conform to the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is not expected to have any immediate societal impact.

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

## Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendices

## A Additional proofs for negative result

## A.1 A simple negative result using 1 / γ queries

We present a simple construction showing that for any values of 0 &lt; γ ≤ ε ≤ 1 and any α ≤ 1 -γ , there exists a domain, distribution, and adversary strategy such that after at most k = 1 / γ queries, all of which are ( ε , γ ) -concentrated, the mechanism is forced to return a response that is not statisticallyaccurate.

Claim A.1. Fix parameters 0 &lt; γ ≤ ε ≤ 1 and α ≤ 1 -γ . There exists a distribution D over X n and a set of k = 1 γ queries, each ( ε , γ ) -concentrated, such that an attacker submitting these queries to any mechanism will receive a response that differs from its true expectation by more than α on at least one query.

Proof. Let r = 1 / γ , assumed to be an integer for simplicity. Let the domain be X = { 1 , 2 , . . . , rn } , partitioned into r disjoint subsets S 1 , . . . , S r ⊂ X , each of size n .

Let D be the uniform distribution over these subsets: that is, S ∼ D means S = S i with probability 1 /r = γ for any i ∈ { 1 , . . . , r } .

Define queries q S 1 , . . . , q S r : X → { 0 , 1 } by

<!-- formula-not-decoded -->

Each query has true mean q S i ( D ) = 1 /r = γ . For any sample S = S i , we have q S i ( S ) = 1 , while for all j = i , q S j ( S ) = 0 .

/negationslash

/negationslash

To verify concentration, note that for any S = S i , The empirical mean q S i ( S ) = 0 , so the deviation from the true mean is exactly γ ≤ ε . For the sample S = S i , q S i ( S ) = 1 , and the deviation is 1 -γ ≥ α .

Thus, submitting the k = 1 / γ queries guarantees that one query must yield an error greater than α , violating ( α , β ) -statistical accuracy.

## A.2 All queries in the attack described in algorithm 1 are ( ε , γ ) -concentrated

Lemma A.2. Each query q t in the first k rounds in the attack described in algorithm 1 is ( ε , γ ) -concentrated.

Proof. Fix round t and any S ′ ∈ Supp( D ) . The sample consists of ε n copies of x j 1 of some element x j 1 ∈ X 1 , and ( n -ε n ) elements from X\X 1 . Since q t ( x ) = 0 for x / ∈ X 1 , we have q t ( S ′ ) = ε q t ( x j 1 ) , where q t ( x j 1 ) ∼ Bernoulli( p t ) . Therefore the empirical mean of any sample in Supp( D ) is in { 0 , ε } , and q t ( D ) the true mean is ε p t ∈ [0 , ε ] , so the absolute deviation is at most ε .

Lemma A.3. The final query q ∗ in the attack described in algorithm 1 is ( ε , γ ) -concentrated under D .

Proof. Let T ∼ D . The query q ∗ evaluates to 1 if T is the sample generated by choosing the j ∗ -th element in each subset X i , and 0 otherwise. Thus, the true mean q ∗ ( D ) = 1 ε N . If q ∗ ( T ) = 0 , the deviation is 1 ε N ; if q ∗ ( T ) = 1 , the deviation exceeds ε but this event occurs with probability 1 ε N . Since N = max { 1 ε 2 , 1 εγ } , we have 1 ε N ≤ ε and 1 ε N ≤ γ , so the deviation exceeds ε with probability at most γ , meaning q ∗ is ( ε , γ ) -concentrated.

## A.3 Proof of theorem 3.1

We begin by proving a supporting lemma:

## A.3.1 Lemma A.4

Lemma A.4. Let j s denote the index of the elements that appear in the true sample S that is used by the mechanism. Define

Then for each j ∈ { 1 , . . . , ε N } and t ∈ { 1 , . . . , k } ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let S = ( x j s 1 , . . . , x j s 1 , . . . , x j s r . . . , x j s r ) be the true sample that is used by the mechanism. Case 1: j = j s . Here q t ( x j 1 ) is independent of a t , and E [ q t ( x j 1 )] = p t , so:

<!-- formula-not-decoded -->

/negationslash

Case 2: j = j s . The analysis in this case depends on the type of mechanism used.

If M is a noise addition mechanism: Substitute a t = q t ( x j s 1 ) r + η t into z j s t . Since η t is zero-mean and independent,

<!-- formula-not-decoded -->

Because q t ( x j s 1 ) ∼ Bernoulli( p t ) and p t ∼ Unif[0 , 1] , this gives:

<!-- formula-not-decoded -->

If M is a subsampling mechanism: The output a t is the empirical mean of the query evaluated on a subsample S M of size m which is constructed by drawing elements i.i.d. and uniformly from the original sample S . We can express the output as:

<!-- formula-not-decoded -->

where each Y /lscript is drawn uniformly from S . By construction, the original sample S (of size n ) contains exactly n/r copies of x j s 1 . Therefore, the probability that any given draw Y /lscript is equal to x j s 1 is:

<!-- formula-not-decoded -->

/negationslash

Furthermore, q t ( Y /lscript ) = 0 whenever Y /lscript = x j s 1 . Let C be the number of times x j s 1 is drawn into S M , so C ∼ Bin( m, 1 /r ) . The output can then be simplified to:

<!-- formula-not-decoded -->

as all other points sampled into S M contribute zero to the sum. We first compute the expectation conditioned on p t and C :

<!-- formula-not-decoded -->

Next, taking the expectation over the binomial random variable C yields the expectation conditioned only on p t :

<!-- formula-not-decoded -->

Thus, in the subsampling case as well, the final expectation is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 3.1. For the attack to identify the true sample with probability at least 1 -β , it suffices that

<!-- formula-not-decoded -->

Proof of theorem 3.1. Let j s denote the index of the true sample fed to the mechanism. Define for each index j the cumulative variable Z j = ∑ k t =1 z j t , where

According to lemma A.4:

/negationslash

For each j = j s , define the difference W j t = z j s t -z j t , and let W j = ∑ k t =1 W j t = Z j s -Z j . Note that for a fixed j , the variables z j 1 , . . . z j k are i.i.d., and so are the variables W j 1 , . . . , W j k .

By assumption, mechanism outputs are bounded in a fixed interval. Since p t , q t ( x ) ∈ [0 , 1] , each term | W j t | is bounded. Also, E [ W j t ] = 1 6 r .

Applying Hoeffding's inequality to W j , we get:

for some constant C .

We compare each alternative index j = j s to the true one by checking whether Z j ≥ Z j s , which occurs exactly when W j ≤ 0 . By a union bound over the N/r -1 such indices:

/negationslash

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

To ensure that the attack identifies the true sample with probability of at least 1 -β , it suffices that

<!-- formula-not-decoded -->

Substituting r = 1 / ε and using the definition of N = max { 1 ε 2 , 1 εγ } , we obtain

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

## A.4 Comparison to the i.i.d. setting

Fixed failure probability. For simplicity, we assume the failure probability β is a constant (e.g. β = 0 . 01 ). Under this simplification, the final bound from Section 3 means that no noise-addition mechanism can maintain ( α , β ) -statistical accuracy for:

<!-- formula-not-decoded -->

Comparison under Hoeffding-style concentration To mirror the i.i.d. setting, let γ ( n, ε ) be the double-sided Hoeffding bound: γ ( n, ε ) = 2 exp( -2 n ε 2 ) . This yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Parametrizing the concentration rate To understand how k scales with n , we write ε ( n ) = f ( n ) / √ n , where f ( n ) ∈ (0 , √ n ] . This gives:

Substituting, we get:

<!-- formula-not-decoded -->

We divide the analysis into three regimes based on the value of f ( n ) 2 relative to ln n :

Case 1: f ( n ) 2 &gt; ln n . In this case, the maximum in the second term is attained by f ( n ) 2 , so:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Case 2: f ( n ) 2 ∈ [ 1 2 ln n, ln n ]

Case 3: f ( n ) 2 &lt; 1 2 ln n

<!-- formula-not-decoded -->

Conclusion For any choice of ε ∈ (0 , 1] , whether constant or varying with n (e.g., ε ( n ) = 1 √ n ), if γ ( n, ε ) matches the behavior of the Hoeffding concentration bound, the resulting bound is k = O ( n ) .

## B Additional proofs for positive result

## B.1 Proof of lemma 4.4

Lemma 4.4. Let S ∈ Supp( D ) be a sample, and let q be any query. Then for every measurable set in the output space E ⊆ Y : Pr [ M H ( q ) ∈ E ] ≤ e ε b Pr [ M O ( q ) ∈ E ] and Pr [ M O ( q ) ∈ E ] ≤ e ε b Pr [ M H ( q ) ∈ E ] .

Proof of lemma 4.4. If M H ( q ) = M O ( q ) , the probabilities are equal. Otherwise, since M H ( q ) = M S ( q ) and ∣ ∣ q ( S ) -q ( D ) ∣ ∣ ≤ ε , we have M H ( q ) ∼ Laplace( q ( S ) , b ) and M O ( q ) ∼ Laplace( q ( D ) , b ) . The two distributions differ only in location by at most ε . Therefore, their density ratio is bounded: p M H ( q ) ( y ) p M O ( q ) ( y ) ≤ exp( ε /b ) for all y ∈ R , and similarly for the reverse ratio. Assuming the mechanism outputs are discretized to a finite set Y ⊂ R by rounding to fixed precision, each output value y ∈ Y corresponds to an interval I y ⊂ R . Integrating over these intervals preserves the density ratio bound, yielding the stated probability bounds.

## B.2 Proof of Theorem 4.5

Before presenting the full proof of theorem4.5, we first introduce additional preliminaries, notation, and a supporting lemma that are used throughout the argument.

## B.2.1 Additional preliminaries

Definition B.1 (KL divergence or relative entropy [Kullback and Leibler, 1951]) . For two distributions Y and Z on the same domain, the KL divergence (or relative entropy ) of Y from Z is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now recall several definitions and results from Dwork et al. [2010b] that are instrumental in the proof of advanced composition in differential privacy, and which we will use directly in our analysis. Definition B.2 (Max divergence, e.g., [Dwork et al., 2010a]) . Let Y and Z be distributions on the same domain. Their max divergence is

Lemma B.3 (Lemma III.2 in Dwork et al. [2010b]) . If Y and Z satisfy D ∞ ( Y ‖ Z ) ≤ ε and D ∞ ( Z ‖ Y ) ≤ ε , then D ( Y ‖ Z ) ≤ ε ( e ε -1 ) .

<!-- formula-not-decoded -->

Lemma B.4 (Azuma-Hoeffding inequality [Azuma, 1967]) . Let C 1 , . . . , C k be real-valued random variables with | C i | ≤ a almost surely. Suppose also that E [ C i | C 1 = c 1 , . . . , C i -1 = c i -1 ] ≤ β for every partial sequence ( c 1 , . . . , c i -1 ) ∈ Supp( C 1 , . . . , C i -1 ) . Then, for any z &gt; 0 ,

<!-- formula-not-decoded -->

## B.2.2 Definitions and notations

We recall the definition of a transcript:

Definition 4.1 (Transcript) . Let A be an analyst interacting with a mechanism over k rounds. The random transcript T is the sequence of queries and responses generated in the interaction. A particular realization is denoted by t = ( q 1 , a 1 , . . . , q k , a k ) .

Extended notation. We extend the transcript notation introduced above by letting Q i and A i denote the random variables corresponding to the query issued and response returned at round i , respectively. The full transcript is then the random tuple T = ( Q 1 , A 1 , . . . , Q k , A k ) , and a specific realization is written t = ( q 1 , a 1 , . . . , q k , a k ) . The values of A i depend on the mechanism: in the real-world mechanism, A i = M S ( q i ) ; in the oracle mechanism, A i = M O ( q i ) ; and in the hybrid mechanism, A i = M H ( q i ) .

Definition B.6 (Support of transcripts) . Let D be the data distribution over X n , and let T be the random transcript produced by an interaction (with any of the mechanisms) with an analyst A . We define: T A = { t : Pr[ T = t ] &gt; 0 } .

Definition B.5 (Transcript prefix) . For each round i ∈ [ k ] , the prefix of the transcript up to round i is the random variable T i -1 = ( Q 1 , A 1 , . . . , Q i -1 , A i -1 ) . For a particular realization t = ( q 1 , a 1 , . . . , q k , a k ) , we write the corresponding prefix as t i -1 = ( q 1 , a 1 , . . . , q i -1 , a i -1 ) .

Remark B.7. The support T A depends only on the analyst A , not on the mechanism. This is true because all mechanisms respond by adding independent Laplace noise to either q i ( S ) or q i ( D ) , and, by assumption, the output space Y is finite. Therefore, for any fixed query q i , every output a i ∈ Y occurs with positive probability under all mechanisms. As a result, the transcript t = ( q 1 , a 1 , . . . , q k , a k ) has nonzero probability under each mechanism if and only if it is possible under the analyst's query selection behavior.

## B.2.3 Supporting Lemma

Lemma B.8. Let S ∈ Supp( D ) be a fixed sample, and let q be any query. Then the Kullback-Leibler divergence between the outputs of the hybrid and oracle mechanisms satisfies

<!-- formula-not-decoded -->

Proof. By Lemma 4.4, for every measurable event E ⊆ Y , we have

Taking the supremum over all E ⊆ Y gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying Lemma B.3, yields the stated inequalities:

<!-- formula-not-decoded -->

## B.2.4 Proof of theorem 4.5

Theorem 4.5. Let S ∈ Supp( D ) be a sample, and let A be a fixed analyst. Consider two k -round interactions with A : one with the hybrid mechanism M H , and one with the oracle mechanism M O .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of theorem 4.5. Fix an analyst A and a sample S . To show that

<!-- formula-not-decoded -->

We begin by decomposing the log-likelihood ratio over the k rounds:

<!-- formula-not-decoded -->

Since the analyst is assumed to be deterministic, the query in the i -th round is fully determined by the history up to that round. Therefore for any mechanism M and for any t = ( q 1 , a 1 , . . . , q k , a k ) ∈ T A , it holds that:

<!-- formula-not-decoded -->

Next, we define the random variable, for any t = ( q 1 , a 1 , . . . , q k , a k ) ∈ T A :

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We want to apply Azuma-Hoeffding's inequality B.4 to the sequence C 1 , . . . , C k , to show that for any ρ &gt; 0

which implies that

<!-- formula-not-decoded -->

To apply Azuma-Hoeffding's inequality (Lemma B.4) to the sequence C 1 , . . . , C k , it suffices to verify the following conditions for all i ∈ { 1 , . . . , k } :

1. Pr t ∼ M H ( | C i ( t i ) | ≤ ε b ) = 1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now verify each item in turn. Fix any transcript t = ( q 1 , a 1 , . . . , q k , a k ) ∈ T A .

Verification of 1. From Lemma 4.4, for every round i , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus for any t ∈ T A :

<!-- formula-not-decoded -->

Verification of 2. We begin by bounding the conditional expectation of C i given any fixed transcript prefix t i -1 ∈ T A . Since the analyst is deterministic, fixing t i -1 determines the query q i , and the randomness in round i lies only in the mechanism's response.

<!-- formula-not-decoded -->

where the third equality uses the fact that t ′ i -1 = t i -1 , since t ′ i ∼ Pr M H [ · | T i -1 = t i -1 ] , and the final inequality follows from Lemma B.8.

Note that C 1 , . . . , C k are deterministic functions of the transcript, meaning that the transcript prefix t i -1 fully determines the values c 1 , . . . , c i -1 . Hence by showing the above bound holds for any transcript prefix in the support T A , it implies the desired conditional expectation also holds for any c 1 , . . . , c i -1 ∈ Supp( C 1 , . . . , C i -1 ) . Therefore:

<!-- formula-not-decoded -->

Conclusion. Since both items 1 and 2 hold, Azuma-Hoeffding's inequality implies that for any ρ &gt; 0 , as claimed.

A symmetric argument applies to the reversed log-likelihood ratio, by repeating the analysis with the roles of M H and M O swapped. Hence, for any t ∈ T A , we obtain

<!-- formula-not-decoded -->

## B.3 Proof of corollary 4.6

Corollary 4.6. Let A be a fixed analyst, S ∈ Supp( D ) a sample, and let E be any event that can arise in the interaction with the analyst. For any ρ &gt; 0 , define ε ∗ as in Theorem 4.5. Then e -ε ∗ (Pr M O [ E ] -ρ ) ≤ Pr M H [ E ] ≤ e ε ∗ Pr M O [ E ] + ρ

Proof. Let B = { t ∈ T A : ln Pr M H ( T = t ) Pr M O ( T = t ) &gt; ε ∗ } be the 'bad' event where the likelihood-ratio bound fails. By Theorem 4.5, Pr M H [ B ] is at most ρ . Hence

<!-- formula-not-decoded -->

Exchanging roles of M H and M O yields the lower bound Pr M H [ E ] ≥ e -ε ∗ (Pr M O [ E ] -ρ ).

## B.4 Proof of lemma 4.7

Lemma 4.7. Fix an analyst A , a noise scale b , and let G be the set of ε -good sample-transcript pairs. Then, for every measurable subset E ⊆ G , Pr M S [ E ] = Pr M H [ E ] .

<!-- formula-not-decoded -->

Proof of lemma 4.7. Run both mechanisms by first drawing the sample S ∼ D and then drawing k independent Laplace noises η 1 , . . . , η k ∼ Laplace (0 , b ) . These draws fix all randomness in the interaction. On any transcript t in the event E , every query q i satisfies | q i ( S ) -q i ( D ) | ≤ ε , so by definition, the hybrid mechanism never switches to the oracle mode. Hence for every draw ( S, η 1 , . . . , η k ) that yields t , both M S and M H produce the same t . Since the joint distribution over ( S, η 1 , . . . , η k ) is identical in both mechanisms, the probability of observing any t ∈ E is the same.

## B.5 Proof of lemma 4.8

Lemma 4.8. Let S ∼ D and consider a k -round interaction between an analyst A and the oracle mechanism, producing the transcript t . Define the failure probability of any of the Laplace noises exceeding α as ζ = 1 -Pr[ | η ′ 1 | ≤ α ] k , for η ′ 1 ∼ Laplace(0 , b ) . Then,

<!-- formula-not-decoded -->

Proof of lemma 4.8. Since the oracle mechanism operates independently of the sample, the queries are chosen independently of the sample. By the definition of ( ε , γ ) -concentration and applying a union bound over all k queries, the probability that the sample-transcript pair is ε -good is at least 1 -k · γ . Additionally the probability that for all k rounds the oracle's response is within α of the true mean is 1 -ζ , where ζ represents the failure probability due to the added Laplace noises. Combining these bounds with a union bound yields the desired result.

## B.6 Proof of theorem 4.9

Theorem 4.9. Let A be an analyst, and M S the real-world Laplace mechanism interacting with A over k rounds. For α &gt; 0 and ρ &gt; 0 , define ε ∗ as in theorem 4.5, and let ζ be as in lemma 4.8. Then, the probability that the real-world mechanism produces an α -accurate transcript satisfies

<!-- formula-not-decoded -->

Proof of theorem 4.9. Let E denote the event that a sample-transcript pair ( S, t ) is both ε -good and α -accurate. By Lemma 4.8, we have Pr M O [ E ] ≥ 1 -k γ -ζ . Applying Lemma 4.6, the probability of E under the hybrid mechanism is Pr M H [ E ] ≥ e -ε ∗ (Pr M O [ E ] -ρ ) . By Lemma 4.7, we know the probabilities for ε -good pairs are identical for the real-world and hybrid mechanisms, so Pr M S [ E ] = Pr M H [ E ] . Since E is a subevent of the event that the transcript is α -accurate, we conclude that the probability of an α -accurate transcript is at least e -ε ∗ (1 -k γ -ζ -ρ ) .

## B.7 Detailed derivation of the query bound

Theorem 4.10. Let A be any analyst issuing k adaptive ( ε , γ ) -concentrated queries, and fix an accuracy parameter α &gt; 0 and failure probability β &gt; 0 . Then the Laplace mechanism can achieve ( α , β ) -accuracy over all k queries provided k = O ( min { β γ , β ε -2 , α 2 β 2 ε 2 [ln(1 / ε )] 2 ln(1 / β ) }) .

Full derivation of the bounds in theorem 4.10. Theorem 4.9 implies that for any fixed α &gt; 0 and number of queries k , the real-world mechanism satisfies ( α , β ) -accuracy provided

<!-- formula-not-decoded -->

As in the proof of Theorem 4.10, we set b = α 2 ln(1 / ε ) . The overall failure probability depends on four terms: ρ , k γ , ζ , and ε ∗ , each of which is bounded by β / 4 . Under these conditions, the total success probability is approximated by

Concretely:

<!-- formula-not-decoded -->

## (i) Concentration failure.

<!-- formula-not-decoded -->

- (ii) Noise-exceedance failure. Recall ζ = 1 -(1 -e -α /b ) k ≤ k e -α /b . Requiring

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(iii) Likelihood-ratio failure. The term ε ∗ = √ 2 k ln(1 / ρ ) ε b + k ε b ( e ε /b -1) must satisfy ε ∗ ≤ β / 4 . With ρ = β / 4 and using the bound e ε /b -1 ≤ 2 ε /b for ε b ∈ (0 , 1) , we obtain

Substitute b = α 2 ln(1 / ε ) , so ε b = 2 ε ln(1 / ε ) α . This yields

<!-- formula-not-decoded -->

Conclusion. Taking the minimum over the three derived bounds on k completes the proof of Theorem 4.10.
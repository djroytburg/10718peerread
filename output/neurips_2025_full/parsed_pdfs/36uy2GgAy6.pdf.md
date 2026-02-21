## When Data Can't Meet: Estimating Correlation Across Privacy Barriers

## Abhinav Chakraborty

Columbia University New York, NY 10027 ac4662@columbia.edu

## Arnab Auddy

The Ohio State University Columbus, OH 43210 arnab.auddy@columbia.edu

## Abstract

We consider the problem of estimating the correlation of two random variables X and Y , where the pairs ( X,Y ) are not observed together, but are instead separated co-ordinate-wise at two servers: server 1 contains all the X observations, and server 2 contains the corresponding Y observations. In this vertically distributed setting, we assume that each server has its own privacy constraints, owing to which they can only share suitably privatized statistics of their own component observations. We consider differing privacy budgets ( ε 1 , δ 1 ) and ( ε 2 , δ 2 ) for the two servers and determine the minimax optimal rates for correlation estimation allowing for both noninteractive and interactive mechanisms. We also provide correlation estimators that achieve these rates and further develop inference procedures, namely, confidence intervals, for the estimated correlations. Our results are characterized by an interesting rate in terms of the sample size n , ε 1 , ε 2 , which is strictly slower than the usual central privacy estimation rates. More interestingly, we find that the interactive mechanism is always better than its non-interactive counterpart whenever the two privacy budgets are different. Results from extensive numerical experiments support our theoretical findings.

## 1 Introduction

Federated learning is a popular and extensively studied framework in modern machine learning. In traditional federated learning, due to privacy concerns, the servers are not allowed to pool raw data, but are restricted to sharing only sufficiently privatized statistics derived from the local observations. This method is particularly beneficial when training on sensitive data, such as healthcare or finance. The federated scenario is very systematically studied when the separation occurs horizontally, i.e. observations of the same set of features are binned separately into different servers. See, for example, Kairouz et al. [2021], Li et al. [2020a,b], Zhang et al. [2021] and the references therein.

To encourage collaboration on proprietary data across different organizations, however, it is often more reasonable to assume that the federation occurs 'vertically', or across features. For example, in healthcare data, a hospital and a pharmaceutical company might have different pieces of information on the same patient: the hospital does not share private clinical information such as patient demographics or test results with the company, which instead has its own private information on the same patient's response to certain drugs. This new framework called vertical federated learning has recently seen studied in Chen et al. [2020], Liu et al. [2024], Wu et al. [2020], Wei et al. [2022], Yang et al. [2019], but a theoretical understanding of estimation and inference has largely been missing. This motivates the current work. We study the correlation of bivariate data from n pairs of samples ( X i , Y i )

T. Tony Cai

The Wharton School University of Pennsylvania Philadelphia, PA 19104

tcai@wharton.upenn.edu

which are not observed together, but are instead separated into two servers as { X i : 1 ≤ i ≤ n } and { Y i : 1 ≤ i ≤ n } .

To distinguish our results from the influence of estimating the marginal distributions of X and Y , we assume that E ( X ) = E ( Y ) = 0 and Var( X ) = Var( Y ) = 1, and ( X,Y ) are sub-Gaussian. That is, we assume that our data are pre-normalized to have mean zero and variance one. We revisit the question of normalization in the supplementary material and show both theoretically and in numerical experiments that the rate of correlation estimation is not influenced by this step. In this situation, we consider estimating ρ = E ( XY ) from the statistics shared by the two servers: viz., Server 1 releases T 1 ( X 1 , . . . , X n ), and Server 2 releases T 2 ( Y 1 , . . . , Y n ). To protect user privacy, we impose the differential privacy framework (see, e.g., Abowd et al. [2020], Bassily et al. [2014], Dwork [2006], Karwa and Vadhan [2017]) on T 1 and T 2 ; both of which must satisfy ( ε 1 , δ 1 ) and ( ε 2 , δ 2 )-DP constraints. For ease of reference, we will somewhat loosely denote the above by a server-level ( ε 1 , ε 2 , δ 1 , δ 2 )-DP constraint and introduce specific definitions later. Such distributed privacy requirements are frequently used in federated learning. See, e.g., Auddy et al. [2024], Cai et al. [2024a,b,c], Shen et al. [2022], Wei et al. [2020, 2021] and references therein.

## 1.1 Main results

The key finding in this work is that the complexity of the correlation estimation in the above setup fundamentally depends on whether or not the statistics T 1 and T 2 are allowed to depend on one another. We now present our main results. Throughout this paper, we assume ε 1 , ε 2 ≤ C for a constant C &gt; 0.

## 1.1.1 Non-interactive protocol

In our first set of results, we consider estimating ρ in the non-interactive (NI) framework of stricter privacy requirements, where T 1 and T 2 are constructed independently, i.e., without any interaction or information about one another. In this case, the differential privacy requirements on T 1 and T 2 are as follows. With X = ( X 1 , . . . , X n ), Y = ( Y 1 , . . . , Y n ), and similarly X ′ , Y ′ (with one data point replaced):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let NI( ε 1 , ε 2 , δ 1 , δ 2 ) to be the class of all correlation estimators constructed using T 1 ( X ) and T 2 ( Y ) satisfying the above privacy requirement. The following theorem states the minimax rate for estimating ρ in this scenario.

Theorem 1.1. The minimax rate for estimating correlation ρ via a non-interactive procedure satisfying server level ( ε 1 , ε 2 , δ 1 , δ 2 ) -DP constraints is given by

<!-- formula-not-decoded -->

for a factor L n of order at most O (log( n )) , whenever δ 1 , δ 2 = o ( n -1 ) .

Note that the rate does not depend on δ 's. This implies that our rate matching correlation estimator achieves ( ε 1 , ε 2 , 0 , 0)-DP, and is still rate optimal (up to logarithmic terms) even within NI( ε 1 , ε 2 , δ 1 , δ 2 ), the class of all non-interactive estimators satisfying ( ε 1 , δ 1 ) and ( ε 2 , δ 2 ) DP constraints for δ 1 , δ 2 are small positive numbers. The rate optimal estimator in this case is given by the correlation of privatized batch means from both servers.

It is useful to compare the above rate with the ones existing in the literature. Firstly, when ( X,Y ) are jointly observed, and we impose ( ε, δ )-central DP constraints on ( X i , Y i ), the optimal correlation estimation rate is given by 1 n 2 ε 2 . See, e.g., Biswas et al. [2020], Cai et al. [2021]. As expected, when ε 1 = ε 2 = ε , this is better than the rate we observe in the current feature separated case, thus highlighting the cost of vertical federation. A second comparison can be made with component-wise local privacy rates, studied in Amorino and Gloter [2023]. The authors there show that in the vertically separated scenario, if we impose ( ε 1 , 0) and ( ε 2 , 0) local DP constraints, the minimax estimation rate for correlation is given by 1 nε 2 1 ε 2 2 , which is again strictly worse than the rates we find under the server level DP constraints.

## 1.1.2 Interactive protocol

We next move on to a larger class of estimators in the interactive (INT) framework, where we still require server level privacy, but one of the servers is allowed access to the privatized statistic output from the other. In other words, we allow the functions T 1 and T 2 to have one way interaction with each other. This requires making exactly one out of two possible choices. The first possibility is that when constructing T 2 , Server 2 has access to T 1 ( X ), in addition to its own data Y . The second possibility arises by analogously interchanging the roles of servers 1 and 2. To fix ideas, if we are in the first case, i.e server 2 gets to observe the transcript T 1 , before computing T 2 , the privacy requirements become:

<!-- formula-not-decoded -->

Replacing X with Y and the index 1 with 2 allows one to write the analogous privacy constraint in the second case where Server 1 has access to T 2 ( Y ). Let INT( ε 1 , ε 2 , δ 1 , δ 2 ) to be the class of all correlation estimators constructed using T 1 ( X ) and T 2 ( Y , T 1 ( X )) satisfying the above interactive privacy requirement. The following theorem states the minimax rate for estimating ρ in this scenario.

Theorem 1.2. The minimax rate for estimating correlation ρ via a non-interactive procedure satisfying server level ( ε 1 , ε 2 , δ 1 , δ 2 ) -DP constraints is given by

<!-- formula-not-decoded -->

for a factor L n of order at most O (log( n )) , whenever δ 1 , δ 2 = o ( n -1 ) .

/negationslash

Note that unlike NI, in the INT rate, the dominating term depends on ε 1 ∨ ε 2 , i.e. the less stringent privacy requirement. The stronger privacy requirement i.e., ε 1 ∧ ε 2 appears in the second term, but its effect is mitigated by the better sample size factor n -2 . This leads to INT being a strictly better estimator than NI whenever ε 1 = ε 2 . An interesting special case is when X are public, meaning ε 1 is a constant, in which case we find ( ε 2 , δ 2 )-central DP rates for correlation estimation.

The rate optimal estimator in the interactive case is borne out of a natural idea: the server with a less stringent privacy budget should share their statistics with the other server. That is, if ε 1 &gt; ε 2 , we should allow T 2 to depend on T 1 ( X ) and Y . The situation is reversed if ε 2 &gt; ε 1 .

In addition to point estimates ̂ ρ , we also derive asymptotically valid confidence intervals in both the non-interactive (NI) and interactive (INT) scenarios. That is, we find ( ̂ ρ (NI) L,n , ̂ ρ (NI) U,n ) and ( ̂ ρ (INT) L,n , ̂ ρ (INT) U,n ) such that for fixed α ∈ (0 , 1)

<!-- formula-not-decoded -->

We show that our estimation methods are minimax optimal by proving corresponding lower bounds, which to the best of our knowledge, has not been established previously under central differential privacy in a vertically distributed setting. While we follow the classical Le Cam framework, our main technical contribution is a direct control of KL divergence via Fisher information curvature bounds, yielding sharp lower bounds under both noninteractive and one-way interactive protocols. These bounds match our upper bounds up to constants in the Gaussian case and up to logarithmic factors in the sub-Gaussian case. Prior works, such as Hadar et al. [2019], bound KL via mutual information in communication constraint settings; we take a more direct route tailored to central DP. Unlike local DP lower bounds in Amorino and Gloter [2023], our approach handles the more delicate structure of central privacy with vertical data splitting.

The rest of the paper is organized as follows. In Sections 2 and 3 respectively, we describe non-interactive and interactive correlation estimators for bivariate Gaussian and bivariate sub-Gaussian distributions. Section 4 provides minimax lower bounds showing that our estimation procedures are nearly optimal in all cases. Finally, Section 5 shows numerical experiments to corroborate our theoretical results. All proofs are in the supplementary material.

## 2 Non-interactive estimation methods

We first demonstrate an estimation procedure in the non-interactive paradigm. Here Server 1 and Server 2 construct and share T 1 ( X ) and T 2 ( X ) without knowledge of one another. As mentioned in the introduction T 1 ( X ) must satisfy ( ε 1 , δ 1 )-DP and T 2 ( X ) must satisfy ( ε 2 , δ 2 )-DP constraints. Our estimator is based on sharing privatized batch means. Choosing m ≥ 1 we separate the n observations in each server into batches of size m as follows:

<!-- formula-not-decoded -->

## 2.1 Non-interactive correlation estimation for Gaussian distribution

In this subsection, we assume that ( X,Y ) ∼ N ( 0 , Σ( ρ )) with (Σ( ρ )) 11 = (Σ( ρ )) 22 = 1 and (Σ( ρ )) 12 = ρ , the bivariate Gaussian distribution with E ( X ) = E ( Y ) = 0, Var( X ) = Var( Y ) = 1 and correlation E ( XY ) = ρ .

Our estimation procedure for ρ is through the product of sample means across multiple batches. In order to bound the sensitivity directly, i.e., without clipping, we will use the signs of X i and Y i in place of ( X i , Y i ) themselves, to compute our correlation estimator.

<!-- formula-not-decoded -->

where B j are as defined in (1) for j = 1 , . . . , k . To ensure ( ε 1 , 0)-DP and ( ε 2 , 0)-DP constraints each server adds Laplace noise to each batch mean and outputs the vectors T 1 ( X ) , T 2 ( Y ) ∈ R m with elements:

<!-- formula-not-decoded -->

where Z ( j ) l indep ∼ Laplace ( 0 , 2 mε l ) for l = 1 , 2. We can then compute

<!-- formula-not-decoded -->

Since ( X,Y ) are bivariate Gaussians, the covariance above satisfies

<!-- formula-not-decoded -->

which leads to the method-of-moments based private correlation estimator:

<!-- formula-not-decoded -->

We would like to emphasize that (4) is precisely where we use the assumption of Gaussianity on ( X,Y ). Since the bivariate distribution is completely known once ρ is specified, we can explicitly write P ( XY &gt; 0) as a function of ρ , which in turn enables our sign-based estimation procedure. While this can be extended to other bivariate families which are specified by a single correlation parameter ρ , we do not discuss these details for brevity.

To create confidence intervals for ρ , let us define S 2 η to be the sample variance of { ( T 1 ( X )) j ( T 2 ( Y )) j : 1 ≤ j ≤ k } . Then we can define the confidence interval:

<!-- formula-not-decoded -->

where z 1 -α/ 2 is the (1 -α/ 2)-th quantile of the standard Normal distribution.

## 2.2 Non-interactive correlation estimation for sub-Gaussian distributions

In general, we would deal with non-Gaussian data, and thus the sign-based procedure of the previous section would not be exact anymore. We will use a clipping based estimator for this case. For clipping parameters λ 1 , λ 2 &gt; 0 to be chosen later we replace (2) by

<!-- formula-not-decoded -->

where B j are as defined in (1) for j = 1 , . . . , k . As before, each server adds Laplace noise to each batch mean and shares:

<!-- formula-not-decoded -->

where Z ( j ) l indep ∼ Laplace ( 0 , 2 λ l mε l ) for l = 1 , 2. Then we will estimate ρ by the quantity:

<!-- formula-not-decoded -->

Once again defining S 2 ρ to be the sample variance of { ( T 1 ( X )) j ( T 2 ( Y )) j : 1 ≤ j ≤ k } , we have the confidence interval:

<!-- formula-not-decoded -->

where z 1 -α/ 2 is the (1 -α/ 2)-th quantile of the standard Normal distribution. The following theorem states the results for correlation estimator under non-interactive protocol.

Theorem 2.1. The following results hold on the estimation error of ρ using a noninteractive componentwise privacy constrained estimator.

1. When ( X,Y ) ∼ N ( 0 , Σ( ρ )) with (Σ( ρ )) 11 = (Σ( ρ )) 22 = 1 and (Σ( ρ )) 12 = ρ , the estimator ̂ ρ ( G ) NI described in Section 2.1 satisfies ̂ ρ ( G ) NI ∈ NI( ε 1 , ε 2 , δ 1 , δ 2 ) and

<!-- formula-not-decoded -->

2. When ( X,Y ) have mean zero, variance one, X is η 1 -sub-Gaussian, Y is η 2 -subGaussian, and E [ XY ] = ρ , the estimator ̂ ρ ( SG ) NI described in Section 2.2 satisfies ̂ ρ ( SG ) NI ∈ NI( ε 1 , ε 2 , δ 1 , δ 2 ) and

<!-- formula-not-decoded -->

λ 1 = 2 η 1 √ log( n ) , and λ 2 = 2 η 2 √ log( n ) .

3. For any fixed α ∈ (0 , 1) , the confidence intervals defined in (5) and (8) satisfy P ( ρ ∈ CI ( k ) NI ( α )) → 1 -α as n →∞ , for k ∈ { G,SG } .

## 3 Interactive estimation methods

We now show that the rates in the previous section can be improved if we allow a one-step interactive scheme between the two servers. To fix ideas, suppose ε 1 &gt; ε 2 , i.e., the privacy requirement in the first server are less stringent than that in the second one. We will then share the private transcripts involving X to the second server containing the Y observations. This leads to an estimation error rate that improves over the non-interactive protocol.

## 3.1 Interactive correlation estimation for Gaussian distribution

In this case, our interactive estimator based on signs of ( X,Y ) is as follows. Server 1 first communicates to Server 2 the privatized sign vector T 1 ( X ) with elements:

<!-- formula-not-decoded -->

where S i iid ∼ Bernoulli ( exp( ε 1 ) exp( ε 1 )+1 ) are independent sign flips introduced by Server 1 to protect the privacy of X i . Given T 1 ( X ) the second server first computes the covariance

<!-- formula-not-decoded -->

and then outputs the privatized version

<!-- formula-not-decoded -->

As before we then have the private correlation estimator

<!-- formula-not-decoded -->

Similar to the non-interactive case, defining ̂ σ 2 η := 1 -( exp( ε 1 ) -1 exp( ε 1 )+1 ) 2 ( ̂ η ( P ) XY, int ) 2 allows the confidence interval given by the following.

1. If c ∗ = lim n →∞ 2 √ nσ η ε 2 is finite, then the CI is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where for any x ∈ R we define F ∗ ( x ) := P ( Z XY + ̂ c ∗ Z Lap ≤ x ) for ̂ c ∗ = 2 / ( √ n ̂ σ η ε 2 ) and Z Lap ∼ Laplace(0 , 1).

2. If 1 √ nε 2 diverges as n →∞ , then the CI is

<!-- formula-not-decoded -->

## 3.2 Interactive correlation estimation for sub-Gaussian distributions

Following previous sections, Server 1 will send to Server 2 the vector of privatized clipped observations T 1 ( X ) ∈ R n with elements ( T 1 ( X )) i = [ X i ] λ 1 + Z 1 i for a clipping parameter λ 1 &gt; 0 and Z 1 i iid ∼ Laplace (2 λ 1 /ε 1 ) for i = 1 , . . . , n . Then Server 2 can construct

<!-- formula-not-decoded -->

In the above [ x ] t := sign( x )( | x | ∧ t ) for any x ∈ R and t &gt; 0. Here Z 2 ∼ Laplace (2 λ 2 /nε 2 ) is Laplace noise added to ensure DP requirements. In addition to ̂ ρ ( SG ) INT , Server 2 also outputs a privatized sample variance S 2 ρ of [( T 1 ( X )) i Y i ] λ 2 for i = 1 , . . . , n . Then we have the confidence interval constructed as follows:

1. If c ∗ = lim n →∞ 2 λ 2 √ nσ ρ ε 2 is finite, then the CI is

<!-- formula-not-decoded -->

where for any x ∈ R we define F ∗ ( x ) := P ( Z XY + ̂ c ∗ Z Lap ≤ x ) for ̂ c ∗ = 2 λ 2 / ( √ nS ρ ε 2 ), and Z Lap ∼ Laplace(0 , 1).

2. If λ 2 √ nε 2 diverges as n →∞ , then the CI is

<!-- formula-not-decoded -->

The following theorem states the results for correlation estimator under the interactive protocol.

Theorem 3.1. The following results hold on the estimation error of ρ using the above privacy constrained interactive estimator.

1. When ( X,Y ) ∼ N ( 0 , Σ( ρ )) with (Σ( ρ )) 11 = (Σ( ρ )) 22 = 1 and (Σ( ρ )) 12 = ρ , the estimator ̂ ρ ( G ) INT described in Section 3.1 satisfies ̂ ρ ( G ) INT ∈ INT( ε 1 , ε 2 , δ 1 , δ 2 ) and

<!-- formula-not-decoded -->

2. When ( X,Y ) have mean zero, variance one, X is η 1 -sub-Gaussian, Y is η 2 -subGaussian, and E [ XY ] = ρ , the estimator ̂ ρ ( SG ) INT described in Section 3.2 satisfies ̂ ρ ( SG ) INT ∈ INT( ε 1 , ε 2 , δ 1 , δ 2 ) and

<!-- formula-not-decoded -->

if λ 1 = 2 η 1 √ log( n ) and λ 2 = 4( η 2 ∨ 1)(log( n )) 2 / ( ε 1 ∧ 1) .

3. For any fixed α ∈ (0 , 1) , under their respective assumptions, the confidence intervals defined in (10) , (11) , (12) , and (13) satisfy P ( ρ ∈ CI ( k ) INT ( α )) → 1 -α as n → ∞ , for k ∈ { G,SG } .

## 4 Minimax lower bounds

In this section, we show that the private correlation estimators derived in the previous section are in fact minimax optimal in many cases. Our proof strategy is based on bounding Fisher information of the private transcripts and then using Van Trees inequality. We recall some standard results from parameter estimation theory in the next subsection.

## 4.1 Fisher information and Van Trees inequality

Let θ be a real-valued parameter taking an unknown value in some interval [ a, b ]. We observe some random variable (or vector) X with distribution P ( x | θ ) parameterized by θ .

Assume that P ( ·| θ ) is absolutely continuous with respect to a reference measure µ , for each θ ∈ [ a, b ], and dP ( ·| θ ) dµ ( x ) is differentiable with respect to θ ∈ ( a, b ) for µ -almost all x . Then the Fisher information of θ w.r.t. X , denoted as I F ( X ; θ ), is defined as

<!-- formula-not-decoded -->

The following inequality is well-known. See for example Gill and Levit [1995].

Lemma 1 (Van Trees inequality) . Let θ be a real parameter with prior density ζ supported on [ a, b ] ⊂ R , and let X ∼ P ( · | θ ) with conditional density p ( x | θ ) = dP ( ·| θ ) dµ ( x ) . Under some regularity conditions we have that for every estimator ̂ θ = ̂ θ ( X ) with E [( ̂ θ -θ ) 2 ] &lt; ∞ under the joint law of ( X,θ ) satisfies

<!-- formula-not-decoded -->

where I F ( ζ ) := ∫ b a ( ζ ′ ( θ ) ) 2 ζ ( θ ) dθ is the prior Fisher information.

The 'regularity conditions' in Lemma 1 are to ensure that one can apply the dominated convergence theorem to exchange certain integrals and differentiations in the calculus. See for example Vaart [1998]. Additionally, assume that

<!-- formula-not-decoded -->

where η ( /epsilon1 ) &lt; C η for all | /epsilon1 | &lt; c 0 for some numerical constants c 0 &lt; 1 and C η &gt; 0.

## 4.2 Non interactive

For the non-interactive protocols the servers output transcripts T 1 and T 2 which are ( ε 1 , δ 1 ) and ( ε 2 , δ 2 )-DP respectively. The transcripts are based on;y on the data from their own servers. An estimator ̂ ρ is then calculated after combining T 1 and T 2 .

Our lower bound is shown by the difficulty of correlation estimation when ρ = 0. Let us denote the transcripts by T ≡ ( T 1 , T 2 ). As a first step, the next lemma shows that I F ( T ; 0) is smaller than a quantity involving the sample size n and the privacy parameters ε 1 , ε 2 . Lemma 2. Assume that for k = 1 , 2 , δ k log(1 /δ k ) = O ( ε 2 k ) . Let us denote the Fisher information for the transcripts T by I F ( T ; ρ ) . We have that

<!-- formula-not-decoded -->

The local regularity assumption in (16) at ρ = 0 ensures that up to a constant factor, the bound from the above lemma carries over to I F ( T ; ρ ) for | ρ | ≤ c 0 . For a suitable choice of prior density ζ this in turn implies an upper bound on E 0 [ I F ( T ; 0)] and allows us to complete the proof by using Van-Trees inequality (Lemma 1). We then have the following lower bound on the minimax risk for estimating ρ in the non-interactive setting.

<!-- formula-not-decoded -->

Theorem 4.1. Assume that δ k = o ( n -1 -ω ) for k = 1 , 2 and n ( ε 2 1 ∧ ε 2 2 ) →∞ . Then for non interactive protocols the minimax rate is lower bounded by

Remark 4.1. The assumption n ( ε 2 1 ∧ ε 2 2 ) →∞ assumes that the minimax rate is going to zero ensuring consistent estimation of ρ in the first place.

## 4.3 Interactive

We next allow one way interaction among the servers where either of the server can share its transcripts with the other server. Let us denote the set of protocols which allow allow interaction from server 1 to 2 as Π 1 → 2 , i.e server 2 gets to observe the transcript T 1 , before computing T 2 . We first show the following upper bound on I F (Π 1 → 2 ; 0).

Lemma 3. Assume that δ 1 log(1 /δ 1 ) = o ( ε 2 1 ) and δ 2 log(1 /δ 2 ) 2 = o ( nε 2 1 ε 2 2 ) . Let us denote the Fisher information for the transcripts Π 1 → 2 by I F (Π 1 → 2 ; ρ ) . We have that

<!-- formula-not-decoded -->

If we denote the protocol which allow interaction from server 2 to 1 we can show that I F (Π 2 → 1 ; 0) ≤ nε 2 2 ∧ n 2 ε 2 1 ε 2 2 . Since we allow for either of the protocols Π ≡ (Π 1 → 2 , Π 2 → 1 ) we have that

Similar to the non-interactive case we can then use the local regularity assumption in (16) and a suitable prior density ζ with Van Trees inequality, leading to the following lower bound on the minimax risk in the interactive setting.

<!-- formula-not-decoded -->

Theorem 4.2. Assume that for k = 1 , 2 δ k = o ( n -1 -ω ) for ω &gt; 0 , n ( ε 2 1 ∨ ε 2 2 ) → ∞ and n 2 ε 2 1 ε 2 2 →∞ . Then for interactive protocols the minimax rate is lower bounded by

Remark 4.2. The assumption n ( ε 2 1 ∨ ε 2 2 ) →∞ and n 2 ε 2 1 ε 2 2 →∞ assumes that the minimax rate is going to zero, ensuring consistent estimation of ρ in the first place.

<!-- formula-not-decoded -->

## 5 Numerical experiments

We evaluate our non-interactive sign-batch ( NI ) and interactive sign-flip ( INT ) estimators across different parameter settings. All our codes can be found at https:// github.com/abhinavc3/distributed-correlation .

## 5.1 Simulation experiments

In our experiments we write non-normalized to mean that the mean and variances of the marginal distributions are known, and normalized to mean that they are unknown and estimated. We use two generative models.

- Gaussian: ( X,Y ) ∼ N ( µ, 2Σ( ρ )) with µ = (0 . 5 , 0 . 5) /latticetop , and Σ( ρ ) given by Var( X ) = Var( Y ) = 1 and Corr( X,Y ) = ρ . We run each estimator with and without the private normalization step.
- Bounded-factor (sub-Gaussian): X = U + E 1 , Y = U + E 2 with U ∼ Unif [ - √ 3 ρ, √ 3 ρ ] and E i ∼ Unif [ -√ 3(1 -ρ ) , √ 3(1 -ρ ) ] , so each marginal is centred, variance-one, and bounded hence sub-Gaussian.

For every design point we record mean-squared error (MSE), average confidence-interval (CI) length, empirical coverage (1 -α = 0 . 95) and the mean CI offset band E [CI L -ρ ] → E [CI U -ρ ] where CI L and CI U are the upper and lower confidence bars. In practice it is sufficient to use the confidence intervals from (10) and (12) since (11) and (13) are respectively the limiting versions of the above two.

Parameter Grid. We vary our parameters as below, with 250 replications for each cell:

- Sample size: n ∈ { 1000 , 1500 , 2500 , 4000 , 6000 , 9000 } .
- Correlation: ρ ∈ { 0 , 0 . 15 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 65 , 0 . 8 , 0 . 9 } .
- Privacy budget: ( ε 1 , ε 2 ) ∈ { (0 . 5 , 0 . 5) , (1 , 1) , (1 . 5 , 0 . 5) } . Mean CI offset bands - n = 1500 , e 1 =  1.5, e 2 =  0.5 Mean CI offset bands - n = 1500 , e 1 =  1.5, e 2 =  0.5

Figure 1: Gaussian, n = 1500 , ( ε 1 , ε 2 ) = (1 . 5 , 0 . 5) . Mean CI-offset bands for NI (grey) and INT (blue). Left: without normalization. Right: with private normalization. Curves overlap.

<!-- image -->

Figure 1 compares the mean CI offset bands for n = 1500 and the budget ( ε 1 , ε 2 ) = (1 . 5 , 0 . 5). With and without normalization the ribbons coincide, indicating that private normalization is cost-free . Figure 2 shows CI width and coverage versus n at ρ = 0 . 5; both variants adhere to the nominal 95% band. Figure 3 confirms that INT is uniformly more efficient than NI, while normalization leaves MSE unchanged (largest relative difference &lt; 2%).

Figure 2: Gaussian, ρ = 0 . 5 . Average CI length versus n . Left: no normalization; right: with normalization. Normalization has no discernible effect; INT yields shorter CIs. The coverage probabilities are above 0.91 for all CIs.

<!-- image -->

Figure 3: Gaussian (left) and Bounded Factor (right) MSE, ρ = 0 . 5. MSE versus n (log-log). INT dominates NI We repeat the study with the bounded-factor DGP. The qualitative picture is the same:

<!-- image -->

INT enjoys narrower CIs and lower MSE , and both estimators achieve nominal coverage.

Figure 4: Mean confidence interval bands for non-interactive (left) and interactive (right) methods for estimating the correlation between age and BMI in the Health and Retirement Study (HRS) data. The black dotted line indicates the non-private estimator.

<!-- image -->

For the sake of brevity we only show the MSE plots (Figure 3 right). The CI bands, coverage and width plots are deferred to the supplementary material.

## 5.2 Real data experiments

We illustrate our methods using data from the Health and Retirement Study (HRS) , a longitudinal survey of older adults in the United States. We focus on two variables-age and body mass index (BMI)-from Wave 2 (year 1993-94) corresponding to around 20k individuals. In this demographic, age and BMI are known to exhibit a mild negative correlation.

We consider a distributed scenario in which the two variables reside on separate servers, and the goal is to estimate their Pearson correlation coefficient ρ . Each server first applies a Central differentially private (CDP) normalization so that the privatized features have approximately zero mean and unit variance. Specifically, we allocate ε = 0 . 1 for each of the mean and standard deviation estimates. The clipping bounds are chosen based on domain knowledge-[45, 90] for age and [15, 35] for BMI-demonstrating a setting where the privacy mechanism leverages prior information rather than data-dependent thresholds.

After normalization, we apply both the non-interactive (NI) and interactive (INT) protocols to obtain private confidence intervals for the estimated correlation ̂ ρ . We compare these to the non-private benchmark while varying the privacy budget ε corr , keeping it equal across the two servers. Results are given in Figure 4. As ε corr increases, the private intervals contract and concentrate around the non-private ρ . Moreover, for a fixed ε corr , the INT intervals are consistently shorter than their NI counterparts. Notably, at ε corr = 1, the interactive CI excludes zero while the non-interactive CI includes it-illustrating that privacy noise can increase uncertainty and, in some cases, prevent rejection of the null hypothesis ρ = 0.

## 6 Discussion

Across both distributions and all privacy budgets explored, INT consistently outperforms NI , while the required private normalization step incurs no measurable loss in bias, MSE or interval width. These findings support the theoretical claim that normalization's privacy cost is dominated by the subsequent correlation release.

We discuss two important directions of future work. First, allowing multiple features per server-rather than a single feature-introduces new challenges, particularly in handling inter-feature correlations and maintaining privacy in higher dimensions. Second, extending our methods to heavy-tailed distributions would broaden applicability, as such data often arise in practice and require more robust estimation techniques.

## Acknowledgements

The research was supported in part by NSF grant NSF DMS-2413106 and NIH grants R01GM123056 and R01-GM129781.

## References

- John M Abowd, Ian M Rodriguez, William N Sexton, Phyllis E Singer, and Lars Vilhuber. The modernization of statistical disclosure limitation at the us census bureau. US Census Bureau , 2020.
- Chiara Amorino and Arnaud Gloter. Minimax rate for multivariate data under componentwise local differential privacy constraints. arXiv preprint arXiv:2305.10416 , 2023.
- Arnab Auddy, T Tony Cai, and Abhinav Chakraborty. Minimax and adaptive transfer learning for nonparametric classification under distributed differential privacy constraints. arXiv preprint arXiv:2406.20088 , 2024.
- Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th annual symposium on foundations of computer science , pages 464-473. IEEE, 2014.
- Sourav Biswas, Yihe Dong, Gautam Kamath, and Jonathan Ullman. Coinpress: Practical private mean and covariance estimation. Advances in Neural Information Processing Systems , 33:14475-14485, 2020.
- T Tony Cai, Yichen Wang, and Linjun Zhang. The cost of privacy: Optimal rates of convergence for parameter estimation with differential privacy. The Annals of Statistics , 49 (5):2825-2850, 2021.
- T Tony Cai, Abhinav Chakraborty, and Lasse Vuursteen. Federated nonparametric hypothesis testing with differential privacy constraints: Optimal rates and adaptive tests. arXiv preprint arXiv:2406.06749 , 2024a.
- T Tony Cai, Abhinav Chakraborty, and Lasse Vuursteen. Optimal federated learning for nonparametric regression with heterogeneous distributed differential privacy constraints. arXiv preprint arXiv:2406.06755 , 2024b.
- Tony Cai, Abhinav Chakraborty, and Lasse Vuursteen. Optimal federated learning for functional mean estimation under heterogeneous privacy constraints. arXiv preprint arXiv:2412.18992 , 2024c.
- Tianyi Chen, Xiao Jin, Yuejiao Sun, and Wotao Yin. Vafl: a method of vertical asynchronous federated learning. arXiv preprint arXiv:2007.06081 , 2020.
- Cynthia Dwork. Differential privacy. In International colloquium on automata, languages, and programming , pages 1-12. Springer, 2006.
- Richard D Gill and Boris Y Levit. Applications of the van trees inequality: a bayesian cram´ er-rao bound. Bernoulli , pages 59-79, 1995.
- Uri Hadar, Jingbo Liu, Yury Polyanskiy, and Ofer Shayevitz. Communication complexity of estimating correlations. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing , pages 792-803, 2019.
- Peter Kairouz, H Brendan McMahan, Brendan Avent, Aur´ elien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and trends® in machine learning , 14(1-2):1-210, 2021.
- Vishesh Karwa and Salil Vadhan. Finite sample differentially private confidence intervals. arXiv preprint arXiv:1711.03908 , 2017.
- Li Li, Yuxi Fan, Mike Tse, and Kuo-Yi Lin. A review of applications in federated learning. Computers &amp; Industrial Engineering , 149:106854, 2020a.
- Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. IEEE signal processing magazine , 37(3):5060, 2020b.

- Yang Liu, Yan Kang, Tianyuan Zou, Yanhong Pu, Yuanqin He, Xiaozhou Ye, Ye Ouyang, Ya-Qin Zhang, and Qiang Yang. Vertical federated learning: Concepts, advances, and challenges. IEEE Transactions on Knowledge and Data Engineering , 36(7):3615-3634, 2024.
- Sheng Shen, Tianqing Zhu, Di Wu, Wei Wang, and Wanlei Zhou. From distributed machine learning to federated learning: In the view of data privacy and security. Concurrency and Computation: Practice and Experience , 34(16):e6002, 2022.
- Alexandre B. Tsybakov. Introduction to nonparametric estimation . Springer series in statistics. Springer, New York ; London, 2009. ISBN 978-0-387-79051-0 978-0-387-79052-7. OCLC: ocn300399286.
- A. W. van der Vaart. Asymptotic statistics . Cambridge series in statistical and probabilistic mathematics. Cambridge University Press, Cambridge, UK ; New York, NY, USA, 1998. ISBN 978-0-521-49603-2.
- Kang Wei, Jun Li, Ming Ding, Chuan Ma, Howard H Yang, Farhad Farokhi, Shi Jin, Tony QS Quek, and H Vincent Poor. Federated learning with differential privacy: Algorithms and performance analysis. IEEE transactions on information forensics and security , 15:34543469, 2020.
- Kang Wei, Jun Li, Ming Ding, Chuan Ma, Hang Su, Bo Zhang, and H Vincent Poor. Userlevel privacy-preserving federated learning: Analysis and performance optimization. IEEE Transactions on Mobile Computing , 21(9):3388-3401, 2021.
- Kang Wei, Jun Li, Chuan Ma, Ming Ding, Sha Wei, Fan Wu, Guihai Chen, and Thilina Ranbaduge. Vertical federated learning: Challenges, methodologies and experiments. arXiv preprint arXiv:2202.04309 , 2022.
- Yuncheng Wu, Shaofeng Cai, Xiaokui Xiao, Gang Chen, and Beng Chin Ooi. Privacy preserving vertical federated learning for tree-based models. arXiv preprint arXiv:2008.06170 , 2020.
- Shengwen Yang, Bing Ren, Xuhui Zhou, and Liping Liu. Parallel distributed logistic regression for vertical federated learning without third-party coordinator. arXiv preprint arXiv:1911.09824 , 2019.
- Chen Zhang, Yu Xie, Hang Bai, Bin Yu, Weihong Li, and Yuan Gao. A survey on federated learning. Knowledge-Based Systems , 216:106775, 2021.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly articulate the main contributions of the paper, including the problem setup, the proposed methodology, and the theoretical guarantees. They accurately reflect the scope of the work and are consistent with the results presented in both the theoretical analysis and the simulation study. Any assumptions and limitations are also stated appropriately, ensuring that the claims are well-aligned with the actual contributions of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations and future directions are described in the Discussion section.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly

when image resolution is low or images are taken in low lighting. Or a speechto-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The paper provides a complete and rigorous treatment of each theoretical result, with all necessary assumptions clearly stated alongside the corresponding theorems. Full proofs are included in the supplemental material, and the main paper provides intuitive explanations to aid understanding. All theorems and lemmas are properly numbered, referenced, and supported by either original arguments or citations to well-established results, ensuring the theoretical contributions are transparent and verifiable.

## Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We have provided all experiment details in Section 5 and codes in an anonymized code repository.

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

Justification: We have provided all codes in an anonymized code repository.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips. cc/public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All details are given in Section 5 and the anonymous code repository.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We have reported 95% confidence intervals with all our estimates.

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

Answer: [Yes]

Justification: All experiments were done on a desktop with 32 GB RAM, and were done over the course of 1 hour.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: There are no violations of the code of ethics.

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
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that

unfairly impact specific groups), privacy considerations, and security considerations.

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
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/ datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
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

## Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research has not used LLM as an important, original, or nonstandard component of the core methods.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.

## A Implementation details

Since the mean and variance of each server can be computed under the central differential privacy (CDP) framework, we adopt estimators similar to those proposed in Karwa and Vadhan [2017] for our simulation study. After obtaining these estimators, we standardize the data and use the resulting values for downstream analysis.

Additionally, to improve the stability of our estimators, we incorporate intermediate clipping steps in our simulation study. For example, in the Gaussian case, we clip the mean of the signs to the interval [ -1 , 1] before applying the sin transformation. In the sub-Gaussian case, we clip the final estimator to [ -1 , 1].

## A.1 Additional simulation study

Here we collect the additional plots and results pertaining to our simulation study.

Figure 5: Gaussian Normalized MSE, ρ = 0 . 5 . MSE versus n (log-log).

<!-- image -->

Figure 6: Bounded-factor, n = 6000 , ( ε 1 , ε 2 ) = (1 . 5 , 0 . 5) . Mean CI offset bands for NI and INT.

<!-- image -->

## B Proofs

Proof of Theorem 1.1. The proof of this theorem follows from parts 1 and 2 of Theorem 2.1 and Theorem 4.1.

Proof of Theorem 1.2. The proof of this theorem follows from parts 1 and 2 of Theorem 3.1 and Theorem 4.2.

Figure 7: Bounded-factor, ρ = 0 . 5 . CI length (left) and coverage (right) versus n .

<!-- image -->

Figure 8: Gaussian, ρ = 0 . 5 . Empirical coverage versus n . Left: no normalisation; right: with normalisation. Normalisation has no discernible effect; INT yields shorter CIs. The coverage probabilities are above 0.91 for all CIs.

<!-- image -->

## B.1 Proofs of upper bound results

Proof of Theorem 2.1. We prove the two statements in the theorem separately.

1. It is straightforward to check that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To bound the error in estimating ρ by ̂ ρ ( P ) we therefore note that

<!-- formula-not-decoded -->

In the penultimate equality, we use the choice m = 8 ε 1 ε 2 , which minimizes the expression in the previous line. The privacy constraints are satisfied by the Laplace mechanism and checking the sensitivity of the batch means.

2. It is straightforward to check that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

✶ We can thus bound the bias of the estimator ̂ ρ ( SG ) NI as:

where we use the fact that X is η 1 -subgaussian and Y is η 2 -subgaussian. At the same time,

<!-- formula-not-decoded -->

where in the last step we use the choice of m = λ 1 λ 2 ε 1 ε 2 , which minimizes the expression in the previous step. Thus the MSE of ̂ ρ ( P ) λ in estimating ρ is given by:

<!-- formula-not-decoded -->

We now choose λ 1 = 2 η 1 √ log( n ) and λ 2 = 2 η 2 √ log( n ) for some κ &gt; 0. The bias bound from (20) then becomes:

leading to the MSE bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Once again the privacy constraints are satisfied by the Laplace mechanism and checking the sensitivity of the batch means.

3. We split the proofs for confidence interval coverage into the Gaussian and subGaussian cases.
2. (a) (Gaussian case) Note that ̂ η XY in (3) is an average of k iid observations T j defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality follows by expanding the squares of iid averages in the first term. Thus we have

<!-- formula-not-decoded -->

and thus by delta method, ̂ ρ ( G ) NI = sin( π ̂ η ( P ) XY / 2) satisfies:

<!-- formula-not-decoded -->

Here we used the fact that sin( π E [ ̂ η XY ] / 2) = ρ . To estimate σ 2 η we use the sample variance of T j :

<!-- formula-not-decoded -->

Note that S 2 η is constructed from ( ε 1 , ε 2 )-DP statistics T j , and thus S 2 η is also differentially private. By standard calculations,

<!-- formula-not-decoded -->

where we use our choice = 8 / ( ε 1 ε 2 ) and k = n/m , and thus by Slutsky's theorem, we then have

<!-- formula-not-decoded -->

We thus have asymptotically (1 -α ) coverage confidence intervals:

<!-- formula-not-decoded -->

- (b) (sub-Gaussian case) Identical to what we observed for the case of Gaussian data, note that ̂ ρ ( SG ) NI in (7) is an average of k iid observations T j defined as follows:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where the last equality follows by expanding the squares of iid averages in the first term of the previous line. Thus we have

<!-- formula-not-decoded -->

To estimate σ 2 ρ we use the sample variance of T j :

<!-- formula-not-decoded -->

Note that S 2 ρ is constructed from ( ε 1 , ε 2 )-DP statistics T j , and thus S 2 ρ is also differentially private. By standard calculations,

<!-- formula-not-decoded -->

where we use our choice m = 4 η 1 η 2 (log( n )) / ( ε 1 ε 2 ) and k = n/m , and thus by Slutsky's theorem, along with the asymptotically vanishing bias from (21) we then have √

<!-- formula-not-decoded -->

We thus have an asymptotically (1 -α ) coverage confidence interval:

<!-- formula-not-decoded -->

Proof of Theorem 3.1. We separate the proofs of the two statements as follows.

1. To derive the MSE of the interactive correlation estimator for Gaussian data, we first calculate from (9):

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

whenever n ( ε 1 ∧ 1) is sufficiently large.

2. To derive the MSE of the interactive correlation estimator for sub-Gaussian data, we take the following approach. It is straightforward to check that

We next have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the fact that X is η 1 -subgaussian and Y is η 2 -subgaussian. We can then bound the bias of the estimator ̂ ρ ( SG ) INT as:

<!-- formula-not-decoded -->

where we use (22) with

<!-- formula-not-decoded -->

and hence the variance becomes

<!-- formula-not-decoded -->

3. We split the proofs for confidence interval coverage into the Gaussian and subGaussian cases.
2. (a) (Gaussian case) From (9) we write:

<!-- formula-not-decoded -->

where T i = (2 S i -1) sign( X i ) sign( Y i ) and Z 2 ∼ Laplace ( 0 , 2 nε 2 ) . Let us define:

<!-- formula-not-decoded -->

for which we have the consistent estimator:

<!-- formula-not-decoded -->

We recall that E [ ̂ η XY, int ] = 2 P ( XY &gt; 0) -1 and thus by the Berry Esseen limit theorem on T i ,

<!-- formula-not-decoded -->

for a numerical constant C &gt; 0. Here

<!-- formula-not-decoded -->

and thus by the delta method,

<!-- formula-not-decoded -->

To derive the confidence intervals we make two separate cases:

Case 1: (( √ nε 2 ) -1 → c ) In the first case we consider ( √ nε 2 ) -1 → c for a finite constant c ≥ 0. In this case we have the confidence interval

<!-- formula-not-decoded -->

where for any x ∈ R we define

F ( x ) := P ( Z XY + c ∗ Z Lap ≤ x ) where c ∗ = lim n →∞ 2 √ nσ η ε 2 and Z Lap ∼ Laplace(0 , 1) .

The above is a valid confidence interval when lim n →∞ 2 √ nσ η ε 2 = c ∗ for some finite c ∗ ≥ 0. This is no longer the case when √ nε 2 → 0 as n →∞ .

Case 2: ( √ nε 2 → 0) In this case, (24) and (25) imply that we have the asymptotic convergence:

<!-- formula-not-decoded -->

leading to the asymptotically (1 -α ) coverage confidence interval

<!-- formula-not-decoded -->

where the width of the CI is determined by the α -th quantiles of the Laplace(0 , 1) distribution.

- (b) (sub-Gaussian case) Note that

<!-- formula-not-decoded -->

where T i = [([ X i ] λ 1 + Z 1 i ) Y i ] λ 2 are iid random variables. Thus by the Berry Esseen theorem,

<!-- formula-not-decoded -->

where Z XY ∼ N (0 , 1), Z ′ 2 ∼ Laplace ( 0 , 2 λ 2 √ nσ ρ ε 2 ) and

<!-- formula-not-decoded -->

As before, we now make two cases to derive the confidence intervals.

Case 1: (( √ nε 2 /λ 2 ) -1 → c ) In the first case we consider ( √ nε 2 /λ 2 ) -1 → c for a finite constant c ≥ 0. In this case (23) and (26) imply that we have the confidence interval

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

the sample variance of T i , is an ε 1 -DP consistent estimator for σ 2 ρ . Moreover, as before for any x ∈ R we define

<!-- formula-not-decoded -->

The above is a valid confidence interval when lim n →∞ 2 λ 2 √ nσ ρ ε 2 = c ∗ for some finite c ∗ ≥ 0. This is no longer the case when √ nε 2 /λ 2 → 0 as n →∞ .

Case 2: ( √ nε 2 /λ 2 → 0) In this case, (23) and (26) imply that we have the asymptotic convergence:

<!-- formula-not-decoded -->

leading to the asymptotically (1 -α ) coverage confidence interval

<!-- formula-not-decoded -->

where the width of the CI is determined by the α -th quantiles of the Laplace(0 , 1) distribution.

## B.2 Proofs of lower bound results

Proof of Theorem 4.1 . Fix any non-interactive ( ε 1 , ε 2 , δ 1 , δ 2 )-DP protocol with transcript T = ( T 1 , T 2 ), and let P ρ denote the law of T when ( X i 1 , X i 2 ) n i =1 i.i.d. ∼ N ( 0 , ( 1 ρ ρ 1 )) . We check that f ( x ) = x log(1 /x ) is an increasing function of x whenever x ∈ (0 , exp( -1)). Thus δ k = o ( n -1 -ω ) implies δ k log(1 /δ k ) = o ( n -1 ) = o ( ε 2 k ). The second inequality follows from the fact that nε 2 k →∞ . Invoking Lemma 2 gives, at ρ = 0,

<!-- formula-not-decoded -->

Step 1. Prior supported in a small neighborhood of 0 . Let J = [ -L/ 2 , L/ 2] with L ≤ 2 c 0 and center ρ 0 = 0. Define the cosine-squared prior on J :

<!-- formula-not-decoded -->

This prior satisfies the well-known identity (see, e.g., Tsybakov [2009])

<!-- formula-not-decoded -->

Step 2. Prior-averaged information of the transcript. By (16) with ρ 0 = 0 and | ρ | ≤ L/ 2 ≤ c 0 ,

Therefore

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 3. Van Trees inequality. Applying Lemma 1 with parameter ρ , likelihood P ρ , and prior λ , we obtain the Bayes risk lower bound

<!-- formula-not-decoded -->

Using (27) and writing A := nε 2 1 ∧ nε 2 2 ,

<!-- formula-not-decoded -->

Step 4. Choice of L and consequence. To minimize the denominator in (30) we take the largest admissible support, L = 2 c 0 , yielding

Since the minimax risk dominates the Bayes risk for any prior,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, whenever A →∞ (our standing regime), the constant prior term is negligible and we obtain

Step 5. Classical (1 /n ) term. Additionally the non-private parametric difficulty contributes an additional Ω(1 /n ) term (e.g. by repeating the bound above without privacy constraints and with A /equivasymptotic n near ρ = 0). Combining the terms yields

<!-- formula-not-decoded -->

Proof of Theorem 4.2. Let ( i, j ) be such that ε i ≥ ε j . Since δ i = o ( n -1 -ω ), we have δ i log(1 /δ i ) = o ( n -1 ) = o ( ε 2 i ), where the last equality follows from nε 2 i → ∞ . Similarly, δ j = o ( n -1 -ω ) implies δ j log 2 (1 /δ j ) = o ( n -1 ) = o ( nε 2 1 ε 2 2 ), using that n 2 ε 2 1 ε 2 2 → ∞ . Hence, the conditions of Lemma 3 are met, so that C Π ,n in (17) indeed represents the Fisher-information bound for the (interactive, one-way) protocol.

Throughout, we abbreviate

Let Π ∈ { Π 1 → 2 , Π 2 → 1 } be any fixed one-way interactive DP protocol, and denote by P ρ the law of the full transcript under correlation ρ .

<!-- formula-not-decoded -->

Step 1. Local regularity of information and the prior. By the standing regularity assumption (16), there are numerical constants c 0 ∈ (0 , 1) and C η &gt; 0 such that

<!-- formula-not-decoded -->

In particular, for | ρ | ≤ c 0 ,

We place on ρ the cosine-squared prior supported on J = [ -L/ 2 , L/ 2] with L ≤ 2 c 0 and center 0 that the prior Fisher information is (see proof of Theorem 4.1 for details)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2. Prior-averaged information of the transcript. By (31) and Lemma 3 (or its analogue for Π 2 → 1 ),

<!-- formula-not-decoded -->

Step 3. Van Trees inequality. Applying Lemma 1 with parameter ρ , likelihood P ρ , and prior λ , we obtain where c 1 := 1+ C η is an absolute constant. Choosing the largest admissible support L = 2 c 0 gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the minimax risk dominates the Bayes risk for every prior,

Step 4. Extracting the two interactive terms. By definition, C Π ,n = min { A,B } with

<!-- formula-not-decoded -->

Hence 1 /C Π ,n = max { 1 /A, 1 /B } ≥ 1 2 ( 1 /A + 1 /B ) . Using (33) and the fact that the additive constant ( π/c 0 ) 2 is negligible whenever A ∨ B →∞ , we obtain the privacy-induced contribution

<!-- formula-not-decoded -->

Step 5. Baseline parametric term and conclusion. Even without privacy constraints, estimating a correlation from n i.i.d. Gaussian samples incurs risk Θ(1 /n ); hence

<!-- formula-not-decoded -->

which is the desired bound.

## B.3 Proofs of Lemmas

In this section we provide proofs of lemmas used to prove the lower bound theorems.

Proof of Lemma 2. The main technical ingredient that goes into proving the minimax lower bound is obtaining a upper bound on the Fisher Information under the null i.e ρ = 0. Denote Z = ( X i , Y i ) n i =1 , it can be shown that the score function S ρ ( Z ) for the parameter ρ under the null is given by

<!-- formula-not-decoded -->

The fisher information under the null I F ( T ; 0) is given by

<!-- formula-not-decoded -->

The fisher info under null can be expressed as

<!-- formula-not-decoded -->

where we have used the fact that X i ⊥ Y i | T , X i ⊥ T 2 | T 1 and Y i ⊥ T 1 | T 2 in the second line. We now have that

<!-- formula-not-decoded -->

where in we use the fact that under the null T 1 ⊥ T 2 . Define to matrices M X and M Y such that

<!-- formula-not-decoded -->

then we have that I F ( T ; 0)] = tr( M /latticetop X M Y ). Using Lemma 6 we have that I F ( T ; 0)] ≤ tr( M X ) ‖ M Y ‖ 2 where ‖ . ‖ 2 is the spectral norm . Next let us bound tr( M X ). Note that we can rewrite M X as

<!-- formula-not-decoded -->

where X is the data vector ( X i ) n i =1 and E ( X | T ) is the vector ( E ( X i | T )) n i =1 . Hence

<!-- formula-not-decoded -->

Using Lemma 5 we have that tr( M X ) ≤ n 2 π ( e ε 1 -e -ε 1 2 ) 2 . For bounding ‖ M Y ‖ 2 we can either bound by tr( M Y ) which implies by the previous argument that ‖ M Y ‖ 2 ≤ nε 2 2 or using contraction of the conditional expectation i.e.

<!-- formula-not-decoded -->

which implies ‖ M X ‖ 2 ≤ 1. Putting everything together we have that

<!-- formula-not-decoded -->

Using the fact that e x -1 ≤ 2 x for 0 &lt; x &lt; 1 we have that

<!-- formula-not-decoded -->

Proof of Lemma 3. Denote Z = ( X i , Y i ) n i =1 , it can be shown that the score function S ρ ( Z ) for the parameter ρ under the null is given by

<!-- formula-not-decoded -->

The fisher information under the null I F ( T ; 0) is given by

<!-- formula-not-decoded -->

The fisher info under null can be expressed as

<!-- formula-not-decoded -->

where we have used the fact that X i ⊥ Y i | T , X i ⊥ T 2 | T 1 in the second line. Using the fact that E ( E ( A | B ) 2 ) ≤ E A 2 we have that

<!-- formula-not-decoded -->

Hence expanding the sum of squares we have that

<!-- formula-not-decoded -->

where we used the fact that Y i ⊥ Y j , T 1 and E Y i = 0, E Y 2 i = 1 in the second and third line. The last inequality above follows from Lemma 5.

Following (39) we can write

<!-- formula-not-decoded -->

Let us call G k = E ( X k | T 1 ) Y k ( ∑ n i =1 E ( E ( X i | T 1 ) Y i | T 1 , T 2 )) and G ′ k = E ( X k | T 1 ) Y k ( ∑ n i =1 E ( E ( X i | T 1 ) Y i | T 1 , T ′ 2 )). Also note that E G ′ k = 0 since E Y k = 0 and Y k ⊥ T 1 , T ′ 2 . Now following a similar argument as in (46) we get that

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

The last line follows since ( T 1 , T ′ 2 ) d = ( T 1 , T 2 ) and the fact that E ( E ( X k | T 1 ) 2 ) ≤ EX 2 k = 1 Using the fact that I F ( T ; 0) = ∑ k E G k and putting everything together we have that

<!-- formula-not-decoded -->

Set in Lemma 4, to obtain

<!-- formula-not-decoded -->

We can similarly show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Putting everything together we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If √ I F ( T ; 0) ≤ n √ 2 π ( e ε 2 -e -ε 2 2 )( e ε 1 -e -ε 1 2 ∧ 1 ) we are done else dividing both sides by √ I F ( T ; 0) we have

<!-- formula-not-decoded -->

The second term can be dropped if δ 2 log(1 /δ 2 ) 2 = o ( nε 2 1 ε 2 2 ). The final form is achieved by using the fact that ε 1 , ε 2 ≤ 1.

## B.4 Auxiliary Lemmas

Lemma 4. Define G k = E ( X k | T 1 ) Y k ( ∑ n i =1 E ( E ( X i | T 1 ) Y i | T 1 , T 2 )) then we have that

<!-- formula-not-decoded -->

Proof. Let us denote by Z i = E ( X i | T 1 ) Y i . We begin by bounding E e t | G i | 1 / 2 . By AM-GM and Cauchy-Schwarz inequality, we have that

<!-- formula-not-decoded -->

Using the conditional Jensen's Inequality with the function x → e tx 2 which is convex to obtain that

<!-- formula-not-decoded -->

Hence E e t | G i | 1 / 2 ≤ E ( e t | Z i | ). Bounding the RHS as follows

<!-- formula-not-decoded -->

where we used the AM-GM inequality in the second line and the independence of Y i and E ( X i | T 1 ) in the third line. Using conditional Jensen again, we would have E e 1 2 t ( E X i | T 1 ) 2 ≤ E e 1 2 tX 2 i which implies E ( e t | Z i | ) ≤ E e 1 2 tX 2 i E e 1 2 tY 2 i .

Putting everything together we have that E e t | G i | 1 / 2 ≤ E e 1 2 tX 2 i E e 1 2 tY 2 i ≤ 2 for t ≤ 1 / 2 (since X i , Y i ∼ χ 2 1 ). This implies that

<!-- formula-not-decoded -->

The last inequality follows from Markov. Hence we have that

<!-- formula-not-decoded -->

Lemma 5. Assuming for k = 1 , 2 , δ k log(1 /δ k ) = o ( ε 2 k ) , we have for any 1 ≤ i ≤ n E ( E ( X i | T 1 )) 2 ≤ 2 π ( e ε 1 -e -ε 1 2 ) 2 , similarly we have E ( E ( Y i | T 2 )) 2 ≤ 2 π ( e ε 2 -e -ε 2 2 ) 2 .

Proof of Lemma 5. Note that E ( E ( X i | T 1 )) 2 = E [ X i ( E ( X i | T 1 ))]. Denote A i = X i ( E ( X i | T 1 )) we can write E A i = E A + i -E A -i . Also let us define A ′ i = X i ( E ( X i | T ′ 1 )) where T ′ 1 = T 1 ( X ′ ), where X ′ is the adjacent dataset with its i th data point replaced by X ′ i which is an independent copy.

We can write E A + i as

<!-- formula-not-decoded -->

Similarly we have that

<!-- formula-not-decoded -->

Since E A i = E A + i -E A -i we have that

<!-- formula-not-decoded -->

where we have used the fact that E A ′ i = 0. Observe that

<!-- formula-not-decoded -->

Next we upper bound ∫ ∞ M P ( | A i | ≥ t ) dt in that direction we look at

<!-- formula-not-decoded -->

where we used the AM-GM inequality for the exponent. Next we can apply the CauchySchwarz inequality to obtain that

<!-- formula-not-decoded -->

the second term can further be bounded using the conditional Jensen's Inequality with the function x → e tx 2 which is convex to obtain that

<!-- formula-not-decoded -->

Putting everything together we have that E e t | A i | ≤ E ( e tX 2 i ) ≤ √ 2 for t ≤ 1 / 4 (since X i ∼ χ 2 1 ).This implies that

<!-- formula-not-decoded -->

The last inequality follows from Markov. Hence we have that ∫ ∞ M P ( | A i | ≥ t ) ≤ 4 √ 2 e -M/ 4 , set M = 4log(1 /δ 1 ) to obtain ∫ ∞ M P ( | A i | ≥ t ) ≤ 4 √ 2 δ 1 . we can similarly show that

<!-- formula-not-decoded -->

Putting everything together we have that

<!-- formula-not-decoded -->

If E A i ≤ 2 π ( e ε 1 -e -ε 1 2 ) 2 we are done else dividing both sides by √ E A i we have

<!-- formula-not-decoded -->

The second term can be dropped if δ 1 log(1 /δ 1 ) = o ( ε 2 1 ).

Lemma 6. For square matrices A and B , if B is symmetric, we have

<!-- formula-not-decoded -->

Proof of Lemma 6. The proof follows from von Neumann's trace inequality:

<!-- formula-not-decoded -->

where α i and β i are the singular values of A and B respectively. The proof follows by the definition of /lscript 2 operator norm used on matrix A .
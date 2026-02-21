## Bivariate Matrix-valued Linear Regression (BMLR): Finite-sample performance under Identifiability and Sparsity Assumptions

## Nayel Bettache

Department of Statistics and Data Science, Cornell University, Capital Fund Management, 23 rue de l'Université, 75007 Paris, France nayel.bettache@cfm.com

## Abstract

This paper studies a bilinear matrix-valued regression model where both predictors and responses are matrices. For each observation t , the response Y t ∈ R n × p and predictor X t ∈ R m × q satisfy Y t = A ∗ X t B ∗ + E t , with A ∗ ∈ R n × m + (rowwise ℓ 1 -normalized), B ∗ ∈ R q × p , and E t independent Gaussian noise matrices. The goal is to estimate A ∗ and B ∗ from the observed pairs ( X t , Y t ) . We propose explicit, optimization-free estimators and establish non-asymptotic error bounds, including sparse settings. Simulations confirm the theoretical rates. We illustrate the practical utility of our method through an image denoising application.

## 1 Introduction

Supervised learning is a core task in modern data analysis, often applied to high-dimensional datasets. With recent advances in data acquisition technologies, many real-world datasets now exhibit intrinsic matrix structures. Examples include spatiotemporal measurements, dynamic imaging, and multivariate longitudinal studies, where rows and columns encode distinct dimensions such as time, space, or experimental conditions [8, 27]. In such settings, both the covariates and the responses may be naturally represented as matrices.

Traditional linear regression models are designed to predict a scalar outcome from a vector-valued covariate.. When the response is vector-valued, a naive approach consists in fitting a separate linear model to each coordinate. However, this strategy ignores the multivariate structure of the response. To address this, several works [6, 18, 3] propose modeling the stacked regression coefficients as a low-rank matrix, leading to the multivariate linear regression model. Other works have focused on predicting scalar responses from matrix-valued covariates [38, 28], leading to the trace regression model where the parameter becomes a matrix, usually assumed to have a low-rank structure. In our setting, both the covariates and responses are matrix-valued. It is again not appropriate to fit independent trace regression models to each entry of the output matrix. Thus, we propose a bilinear model that explicitly couples the row and column structures of the input and output matrices.

In this paper, we extend these lines of work by studying a bivariate matrix-valued linear regression (BMLR) model, where both the predictor and the response are matrix-valued. This framework captures richer structural dependencies and arises naturally in applications such as spatiotemporal forecasting and dynamic network modeling. The BMLR model considers T independent observations ( X t , Y t ) T t =1 , where each predictor matrix X t ∈ R m × q and response matrix Y t ∈ R n × p satisfy the relationship

<!-- formula-not-decoded -->

The unknown parameter matrices are A ∗ ∈ R n × m + and B ∗ ∈ R q × p . The non-negativity constraint helps ensuring identifiability and interpretability. In many applications, A ∗ represents mixing

weights, attention coefficients, or interaction strengths that are naturally non-negative. Note that the model in (1) remains unchanged if A ∗ is scaled by a positive constant and B ∗ is divided by the same constant. To resolve this identifiability issue, we impose the constraint that each row of A ∗ has unit ℓ 1 -norm. The noise matrices ( E t ) T t =1 ∈ R n × p have independent and identically distributed entries, each drawn from a centered Gaussian distribution with variance σ 2 . We discuss more general noise structures in Appendix A. The goal is to estimate the parameters A ∗ and B ∗ from the observed data ( X t , Y t ) T t =1 . Once estimated, these parameters can be used to predict the out-of-sample response Y T +1 given a new covariate matrix X T +1 ∈ R m × q . This model builds on a growing literature dedicated to matrix-valued regression with matrix-valued observations [32, 23, 16, 29, 7, 11, 4], and complements recent advances in matrix autoregression and dynamic systems [9, 10, 48, 49, 24, 43, 2, 30, 41, 50]. These approaches leverage the matrix structure of the data to improve interpretability and predictive performance.

Notice that for any t ∈ [ T ] , A ∗ captures the link between the rows of Y t and the rows of X t B ∗ while B ∗ captures the link between the columns of Y t and the columns of A ∗ X t . This bilinear structure encodes interactions along both matrix dimensions and is inherently richer than what standard multivariate regression models capture. If one ignores the matrix structure and vectorizes both sides, the model becomes vec ( Y t ) = ( B ∗ ) ⊤ ⊗ A ∗ · vec ( X t ) + vec ( E t ) , where vec ( · ) denotes the column-wise vectorization of a matrix and ⊗ is the Kronecker product [21]. In this formulation, the regression reduces to a standard multivariate linear model with T vectorized observations and coefficient matrix M ∗ = ( B ∗ ) ⊤ ⊗ A ∗ , which can be estimated directly using ordinary least squares [11]. However, this vectorized approach hides the problem structure. Estimating A ∗ and B ∗ from an estimate ˆ M of M ∗ amounts to solving a Kronecker factorization problem. It reduces to finding the nearest rank-one matrix in the Kronecker product space [33]. This is a non-convex problem that discards the individual matrix roles of A ∗ and B ∗ . Moreover, the vectorized formulation leads to estimators with high variance when nmpq ≫ T , as it fails to exploit the low-dimensional structure induced by the bilinear form.

In contrast, our approach preserves and exploits the bilinear structure directly in the matrix domain, allowing for interpretable and efficient estimation of A ∗ and B ∗ without solving the intractable Kronecker decomposition problem.

## 1.1 Context

Given the observations ( X t , Y t ) T t =1 , the linear regression framework models the relationship between the responses and the predictors using an unknown linear map f : R m × q → R n × p . For each t ∈ [ T ] this relationship is expressed as Y t = f ( X t ) + E t where E t represents the noise term. The goal is to learn the unknown function f only using the given observations. In a parametric setting we assume that f belongs to a predefined class of functions.

A naive approach would be to break down the linear relationship by focusing on each entry of the responses individually. This leads to consider for i ∈ [ n ] and j ∈ [ p ] a linear functional f ij : R m × q → R such that

<!-- formula-not-decoded -->

where [ Y t ] ij denotes the coefficient on the i th row and j th column of the matrix Y t . Riesz representation theorem [5] ensures the existence of a unique matrix M ∗ ij ∈ R m × q such that

<!-- formula-not-decoded -->

where X ⊤ t stands for the transposed of X t and Tr ( X ⊤ t M ∗ ij ) denotes the trace of the square matrix X ⊤ t M ∗ ij . In this model, the objective is to estimate the np matrices ( M ∗ ij ) . This problem is equivalent to considering np independent trace regression models [38]. Hence, this naive approach ignores the multivariate nature of the possibly correlated entries of each response Y t .

To overcome this issue, we consider for i ∈ [ n ] and j ∈ [ p ] the vectors α i ∈ R m + and β j ∈ R q and assume M ∗ ij = α i β ⊤ j . For identifiability issues we assume that for all i ∈ [ n ] , ∥ α i ∥ 1 = 1 . This model ensures that the matrices ( M ∗ ij ) share a common structure. It also implies that they share the same rank, namely one here. Hence this model now accounts for the multivariate nature of the problem. It has n ( m -1) + pq free parameters and rewrites as follows:

<!-- formula-not-decoded -->

Consider A ∗ the matrix obtained by stacking the n row vectors α i and B ∗ the matrix obtained by stacking the p column vectors β j . This leads to the BMLR model (1).

## 1.2 Related works

The BMLR model, first introduced in [23], has gained notable attention as a powerful framework for examining relationships between matrix-structured responses and predictors. In the development of estimation techniques for this model, two principal methods have been explored: alternating minimization and spectral aggregation.

Alternating Minimization , usually presented in a Maximum Likelihood Estimation [16] or a least squares Estimation(LSE) context [23]. As the objective is non convex in both parameters, a two-step iterative algorithm is usually derived to construct the estimators.

Spectral Methods , presented in a factor model framework in [7], offers an alternative that leverages the spectral properties of the target matrices. Authors propose a new estimation method, the α -PCA, that aggregates the information in both first and second moments and extract it via a spectral method. They show that for specific values of the hyperparameter α , namely α = -1 , the α -PCA method corresponds to the least squares estimator. However, the procedure is non convex and they rely on approximate solution by alternating minimization. More specifically they maximize row and column variances respectively after projection.

Kronecker Product Factorization: The Kronecker Product Factorization (KRO-PRO-FAC) method has recently introduced new possibilities for estimation within matrix-valued linear regression. [11] present this approach, which leverages Kronecker products to decompose complex matrices into simpler components. A key advantage of this method is that it circumvents the need to estimate the covariance between individual entries of the response matrix, significantly reducing computational complexity. The KRO-PRO-FAC algorithm is accompanied by non-asymptotic bounds on the estimation of the Kronecker product between the parameters. However, an important limitation of this approach is the lack of direct control over the estimation accuracy of each parameter separately. This restricts its applicability in scenarios where parameter-wise interpretability or precision is critical, leaving a gap that motivates alternative methodologies, such as the optimizationfree estimators proposed in this work.

Autoregressive Frameworks: In autoregressive settings, where X t := Y t -1 in (1), the primary focus is on capturing temporal dependencies by minimizing the residual sum of squares [9, 48, 49, 24, 43]. These frameworks are particularly relevant for modeling dynamic systems and matrix-valued time series. However, parameter estimation in autoregressive frameworks typically relies on computationally intensive procedures, such as iterative optimization or matrix decompositions. These methods become increasingly impractical as the dimensions of the data grow, creating a significant bottleneck in high-dimensional settings. This challenge is especially pronounced in applied studies, where the scale and complexity of datasets continue to expand, underscoring the need for more efficient and scalable estimation techniques.

## 1.3 Summary of Contributions

This work studies the estimation problem of the matrix parameters A ∗ and B ∗ in (1). Our contributions are the following:

Noiseless Case Analysis: In Section 2, we study an oracle case and establish that, in the absence of noise, the true parameters can be exactly retrieved. This analysis highlights the fundamental identifiability properties of the model.

Optimization-Free Estimators: In Section 3, we propose explicit, optimization-free estimators ˆ B and ˆ A defined in (3) and (4) respectively. They significantly simplify the estimation process and are particularly advantageous in high-dimensional settings where traditional optimization-based methods become computationally prohibitive. Theoretical guarantees are provided in Theorems 3.3 and 3.6. We establish non-asymptotic bounds characterizing the dependence of estimation accuracy on the problem dimensions ( n, p, m, q ) and the sample size T . For ˆ A ∈ R n × m , the performance improves with larger sample size T and larger values of p and q , the column dimensions of the response and predictor matrices, respectively. However, the performance deteriorates as n and m , which determine the size of A ∗ , increase. In contrast, for ˆ B ∈ R q × p , the convergence rate improves with increases in T and n , showcasing a "blessing of dimensionality" effect in the row dimension of the target matrices. Nonetheless, the performance decreases with larger values of p , q , and m .

Numerical Validation: In Section 4, we validate our theoretical findings through extensive numerical simulations. On synthetic datasets, we show that the empirical convergence rates align closely with theoretical predictions, see Figure 1. For real-world data from the CIFAR-10 dataset, we demonstrate the practical effectiveness of our procedures by introducing controlled noise, estimating the correction matrices ˆ A and ˆ B on the training set, and evaluating denoising performance on the test set. The results, presented in Figures 2 and 3, highlight the effectiveness of the proposed estimators in practical scenarios.

Sparse Adaptive Estimators: Extending the framework to sparse settings, we introduce in Appendix 3.3 hard-thresholded estimators, ˆ B S and ˆ A S , defined in (5) and (6) respectively. These estimators exploit the sparsity of the true parameters to achieve improved convergence rates, as established in Theorems 3.7 and 3.8. Moreover, we demonstrate that these estimators can recover the exact support of the true parameters with high probability, providing strong practical guarantees. These estimators exhibit the same dependency on the problem dimensions ( n, p, m, q ) and the sample size T as their dense counterpart ˆ B and ˆ A .

## 2 Analysis at the population level

In this oracle case we consider T observations ( X t , M t ) that satisfy the relationship

<!-- formula-not-decoded -->

where A ∗ ∈ R n × m + and B ∗ ∈ R q × p are the unknown parameters to be recovered. The matrix A ∗ is assumed to have rows with unit L 1 -norm, ensuring identifiability of the model.

Understanding this oracle case will allow to establish baseline results that will guide the analysis at the sample level (1) involving noise. The primary goal here is to derive conditions under which the matrices A ∗ and B ∗ can be uniquely recovered, as well as to propose efficient algorithms for their recovery. This involves leveraging the structure of the observed predictors ( X t ) T t =1 and the constraints on the parameters.

In this section, we assume that the matrices ( X t ) T t =1 form a generating family of R m × q , which implies that T is larger than mq . We define two matrices, M := ( vec ( M 1 ) , . . . , vec ( M T )) ⊤ ∈ R T × np and X := ( vec ( X 1 ) , . . . , vec ( X T )) ⊤ ∈ R T × mq .

Remark 2.1 . When the design matrices ( X t ) T t =1 are generated randomly, X ⊤ X is invertible under mild conditions [13, 36, 39, 40].

We note E ( k,l ) the canonical basis matrix with 1 at entry ( k, l ) and define the unobserved matrix C ∈ R mq × np as C := ( vec ( A ∗ E ( k,l ) B ∗ ) ) k ∈ [ m ] , l ∈ [ q ] , where each row of C corresponds to the vectorized form of A ∗ E ( k,l ) B ∗ . The entry of C located at the k +( l -1) m -th row and i +( j -1) n -th column is denoted by [ C ] ( i,j ) ( k,l ) . By construction, each entry of the matrix C is defined as [ C ] ( i,j ) ( k,l ) = [ A ∗ ] ik · [ B ∗ ] lj , for all i ∈ [ n ] , k ∈ [ m ] , l ∈ [ q ] , and j ∈ [ p ] . Moreover, in the model (2), the matrix C can be exactly reconstructed from the observations, as shown in Lemma C.2 in the supplementary material. Corollary 2.4 shows how A ∗ and B ∗ and can be exactly recovered from this quantity

Proposition 2.2. In the model (2) , where the design matrices ( X t ) T t =1 form a generating family of R m × q , the parameter matrices A ∗ ∈ R n × m and B ∗ ∈ R q × p satisfy the following relationships:

̸

<!-- formula-not-decoded -->

Remark 2.3 . For fixed ( l, j ) ∈ [ q ] × [ p ] , the entries ( [ C ] ( i,j ) ( k,l ) ) ( i,k ) share the same sign, as each entry is the product of [ A ∗ ] ik which is non-negative and [ B ∗ ] lj .

The following corollary provides a representation of the entries of A ∗ and B ∗ as averages. This characterization will be particularly useful at the sample level, leading to plug-in estimators.

̸

Corollary 2.4. Let D 0 ⊂ [ p ] × [ q ] denote the set of indices ( j, l ) such that [ B ∗ ] lj = 0 and let F := [ n ] × [ m ] . Then the entries of the matrices A ∗ and B ∗ can be expressed as:

<!-- formula-not-decoded -->

Remark 2.5 . The magnitude of [ B ∗ ] lj can be expressed as | [ B ∗ ] lj | = 1 n n ∑ i =1 n ∑ k =1 ∣ ∣ ∣ [ C ] ( i,j ) ( k,l ) ∣ ∣ ∣ .

## 3 Analysis at the sample level

We now consider T observations ( X t , Y t ) that satisfy (1). Our objective is to estimate A ∗ and B ∗ .

From the observations ( X t , Y t ) T t =1 , we construct Y := ( vec ( Y 1 ) , . . . , vec ( Y T )) ⊤ ∈ R T × np and X := ( vec ( X 1 ) , . . . , vec ( X T )) ⊤ ∈ R T × mq .

We assume that the design matrices ( X t ) T t =1 form a generating family of R m × q . This assumption implies that T ≥ mq and ensures that X ⊤ X ∈ R mq × mq is invertible. Following Remark 2.1, this is a mild assumption. We further define the unobserved noise matrix E := ( vec ( E 1 ) , . . . , vec ( E T )) ⊤ ∈ R T × np and the unobserved signal matrix M := ( vec ( M 1 ) , . . . , vec ( M T )) ⊤ ∈ R T × np where ( E t ) T t =1 are the noise matrices defined in (1) and for t ∈ [ T ] , M t := A ∗ X t B ∗ .

Following Lemma C.2 we define the unobserved matrix C := ( X ⊤ X ) -1 X ⊤ M ∈ R mq × np . To analyze the influence of noise in the estimation process, we define D := ( X ⊤ X ) -1 X ⊤ E ∈ R mq × np . Finally, we derive from the observations ̂ C := ( X ⊤ X ) -1 X ⊤ Y ∈ R mq × np . This leads to ̂ C = C + D .

We assume that for all ( i, j, k ) ∈ [ n ] × [ p ] × [ q ] , the row sums ∑ m k =1 [ ̂ C ] ( i,j ) ( k,l ) are nonzero. It ensures that the plug-in estimator for A ∗ is well-defined. Notably, this assumption holds almost surely when the noise matrices ( E t ) T t =1 are drawn from a continuous distribution, as is the case in this study.

## 3.1 Definition of the estimators

Leveraging the results from Corollary 2.4, we define the plug-in estimators ˆ B ∈ R q × p of B ∗ , defined for all ( j, l ) ∈ D := [ p ] × [ q ] and the plug-in estimator ˜ A ∈ R n × m of A ∗ , defined for all ( i, k ) ∈ [ n ] × [ m ] , as follows:

<!-- formula-not-decoded -->

̸

where [ ˜ B ] ( i ) lj := 1 n -1 n ∑ r =1 r = i m ∑ s =1 [ ̂ C ] ( r,j ) ( s,l ) . In the definition of [ ˜ A ] ik , we ensure that the terms in the numerator do not appear in the denominator to preserve statistical independence of both terms. It is important to note that the entries of ˜ A are defined as the ratio of random variables. While the behavior of such ratios has been studied in the literature, particularly in Gaussian cases [17, 34], the results obtained through this approach would require heavy assumptions and remain challenging to interpret in our context.

When the Gaussian variables in the ratio are centered, the distribution of the ratio is known as the Cauchy distribution [25]. However, for non-centered Gaussian variables, the probability density function of the ratio takes a significantly more complex form [22], making the analysis cumbersome. Under certain conditions, it is possible to approximate the ratio with a normal distribution [15], but this requires additional assumptions on the model and would still yield results that are difficult to decipher. Consequently, we opt for a different approach that avoids these complications while retaining interpretability. Specifically, we observe that the plug-in estimator ˜ A does not fully exploit

a key property of the model: the entries of the matrix A ∗ are constrained to lie between 0 and 1 . This additional structure could be leveraged to improve the estimator's performance. Hence we define the estimator ˆ A ∈ R n × m of A ∗ , defined for all ( i, k ) ∈ [ n ] × [ m ] , as:

<!-- formula-not-decoded -->

## 3.2 Theoretical analysis with known variance

We present the matrix normal case with known fixed variance under the ORT assumption. The matrix normal distribution generalizes the multivariate normal distribution to matrix-valued random variables [20, 1, 35]. The sparse case is presented in the section 3.3.

Lemma 3.1. Under the assumption that X ⊤ X is full rank, the matrix ̂ C := ( X ⊤ X ) -1 X ⊤ Y ∈ R mq × np satisfies ̂ C ∼ MN mq × np ( C , ( X ⊤ X ) -1 , σ 2 I np ) .

Following the vast literature on linear regression and Gaussian sequence models [44, 19, 37, 12], we make the ORT assumption to capture a better understanding of the phenomenon at play. This assumption serves primarily to facilitate the theoretical analysis. Notably, the numerical experiments in Section 4 are conducted without relying on the ORT condition. We discuss relaxation of the ORT assumption in Appendix A.

Assumption 3.2 (ORT assumption) . We assume that the design matrix X satisfies X ⊤ X = T · I mq .

Under Assumption 3.2, Lemma 3.1 ensures that the entries of ̂ C are independent and normally distributed. The following theorem establishes non-asymptotic upper bounds on the convergence rates of ˆ B under various norms.

Theorem 3.3. Under Assumption 3.2, the estimator ˆ B introduced in (3) satisfies the following nonasymptotic bounds for any ϵ &gt; 0 :

<!-- formula-not-decoded -->

Here, ψ p,q := √ p + √ q 2 , ∥ · ∥ + is the elementwise maximum norm, ∥ · ∥ op is the operator norm, and ∥ · ∥ F is the Frobenius norm.

We observe that the convergence rate of the estimator ˆ B exhibits the anticipated dependence on the sample size T , improving as the number of observations increases. Notably, our analysis reveals a "blessing of dimensionality" effect, wherein the convergence rate accelerates as the row dimension n of the observed target matrices ( Y t ) T t =1 grows. Conversely, the convergence rate is negatively affected by the size pq of B ∗ . Furthermore, the variance parameter σ also exerts a detrimental influence on the convergence rate, as intuitively expected.

The following Lemma provides a probabilistic control over the event where the plug-in estimator ˜ A , defined in (3), coincides with its modified version ˆ A defined in (4). We assume that the entries of A ∗ are strictly bounded from below by 0 and from above by 1 . We also assume that the sum of the entries of B ∗ is positive.

Assumption 3.4. We assume that for all ( i, k ) ∈ [ n ] × [ m ] we have 0 &lt; [ A ∗ ] ik &lt; 1 . In addition we assume that β ∗ &gt; 0 where β ∗ := 1 pq p ∑ j =1 q ∑ l =1 [ B ∗ ] lj .

In model (1), A ∗ has non-negative entries with rows summing to one, so its entries lie in [0 , 1] . Assumption 3.4 strengthens this to entries in ]0 , 1[ . The sparse case is discussed in Appendix 3.3.

Lemma 3.5. Under Assumptions 3.2 and 3.4, the estimators ˜ A and ˆ A introduced in (3) and (4) satisfy the following property for any ( i, k ) ∈ [ n ] × [ m ] :

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Wefirst note that under Assumption 3.4, the quantities µ ik and ν ik are positive. We then observe that as the sample size T increases, the plug-in estimator ˜ A is more likely to coincide with its modified version ˆ A . The intuition behind this result is straightforward: as the sample size grows, [ ˜ A ] ik converges to [ A ∗ ] ik , which inherently lies between 0 and 1. Consequently, the modifications introduced in ˆ A become unnecessary as [ ˜ A ] ik naturally satisfies the constraints of the model. A similar phenomenon occurs as p , the number of columns of the response matrices, and q , the number of columns of the predictor matrices, increase. Larger p and q effectively provide more information about the structure of the model, leading to improved accuracy of the plug-in estimator ˜ A and reducing the need for corrections by ˆ A .

The following theorem provides a finite-sample analysis of the performance of ˆ A .

Theorem 3.6. Under Assumptions 3.2 and 3.4, the estimator ˆ A introduced in (4) satisfies for any ϵ &gt; 0 :

<!-- formula-not-decoded -->

where ν := min i,k ν ik , µ := min i,k µ ik with µ ik , ν ik and ˜ σ defined in Lemma 3.5.

We observe that the finite-sample performance of the estimator ˆ A improves with the sample size T , reflecting the benefit of more observations. Additionally, the convergence rate is positively influenced by increases in the column dimension p of the observed target matrices ( Y t ) T t =1 and the column dimension q of the predictor matrices ( X t ) T t =1 . This reflects a "blessing of dimensionality" effect, as additional columns provide richer information for the estimation process. Notably, this behavior contrasts with that of ˆ B , as detailed in Theorem 3.3, where increases in p and q have a detrimental impact. Moreover, the magnitude of β ∗ plays a crucial role in determining the convergence rate, with larger values of | β ∗ | leading to faster convergence. Conversely, the performance of ˆ A deteriorates as the size nm of A ∗ increases and the variance parameter σ negatively affects the convergence rate. Higher noise levels degrade the accuracy. We note that the degradation of both rates with increasing m breaks the symmetry between A ∗ and B ∗ . This is because of the assumption we impose on A ∗ .

## 3.3 Sparse Case with Known Variance

In this part, we extend our theoretical analysis to incorporate sparsity assumptions on the parameters A ∗ and B ∗ . By leveraging the sparse structure, we aim to develop estimation strategies tailored for high-dimensional settings, where many entries of the parameters are expected to be zero. This sparse framework addresses practical scenarios where dimensionality reduction is critical.

We propose the hard-thresholding estimator [19, 37] ˆ B S , defined as follows for all ( l, j ) ∈ [ q ] × [ p ] :

<!-- formula-not-decoded -->

where [ ˆ B ] lj are the entries of the initial estimator ˆ B defined in (3), and τ &gt; 0 is a user-defined threshold. The hard-thresholding operation enforces sparsity by setting small entries of ˆ B to zero, aligning the estimator with the sparse structure of B ∗ .

The following theorem establishes a non-asymptotic upper bound on the convergence rate of ˆ B S for the Frobenius norm under sparsity assumptions on the parameter B ∗ .

Theorem 3.7. Under Assumption 3.2, for any δ ∈ (0 , 1) , the estimator ˆ B S introduced in (5) , with the threshold τ := σ √ 2 m nT ( √ log(2 pq ) + √ log ( 2 δ ) ) , enjoys the following non-asymptotic properties on the same event holding with probability at least 1 -δ :

1. If ∥ B ∗ ∥ 0 denotes the number of nonzero coefficients in B ∗ , then

<!-- formula-not-decoded -->

2. If the entries of B ∗ satisfy min l ∈ [ q ] ,j ∈ [ p ] | [ B ∗ ] lj | &gt; 3 τ, then the support of ˆ B S perfectly matches that of B ∗ , namely supp ( ˆ B S ) = supp ( B ∗ ) , where supp ( · ) denotes the set of indices corresponding to the nonzero entries of a matrix.

Theorem 3.7 highlights the performance of the sparse estimator ˆ B S under sparsity assumptions on the true parameter B ∗ . The convergence rate of ˆ B S improves as the sparsity of B ∗ increases, demonstrating the benefits of leveraging sparse structures. The threshold τ exhibits favorable scaling with the sample size T and row dimension n , both of which contribute to reducing τ , enhancing the estimator's performance. Conversely, τ increases with the dimensions p and q , reflecting the greater challenge of estimation in higher-dimensional settings. Moreover the threshold τ scales with σ , capturing the adverse impact of higher noise levels on the estimation accuracy. This emphasizes the observations from Theorem 3.3. Finally the condition on the minimum magnitude of the entries of B ∗ ensures that the threshold τ enables exact recovery of the support of B ∗ , with high probability.

ˆ S ]

We now propose the hard-thresholding estimator A , defined as follows for all ( i, k ) ∈ [ n ] × [ m where ˆ γ ik := 1 pq p ∑ q ∑ [ ̂ C ] ( i,j ) ( k,l ) :

<!-- formula-not-decoded -->

The following theorem establishes a non-asymptotic upper bound on the convergence rate of ˆ A S for the Frobenius norm under sparsity assumptions on the parameter A ∗ .

Theorem 3.8. Under Assumption 3.2, for any δ ∈ (0 , 1) , the estimator ˆ A S defined in (6) , with the threshold τ := σ √ 2 Tpq ( √ log(2 nm ) + √ log ( 2 δ ) ) , satisfies the following non-asymptotic properties on an event holding with probability at least 1 -2 δ :

1. If ∥ A ∗ ∥ 0 denotes the number of nonzero coefficients in A ∗ , then

<!-- formula-not-decoded -->

where β ∗ := 1 pq ∑ p j =1 ∑ q l =1 [ B ∗ ] lj . and t δ satisfies σ √ 2 m t δ √ npqTπ exp ( -Tpqt 2 δ 2 mσ 2 ) + σ √ 2 t δ √ pqTπ exp ( -Tpqt 2 δ 2 σ 2 ) = δ.

2. If the entries of A ∗ satisfy min i ∈ [ n ] ,k ∈ [ m ] | [ A ∗ ] ik | &gt; 3 τ and if 3 β ∗ &gt; 1 , then the support of ˆ A S perfectly matches that of A ∗ , namely supp ( ˆ A S ) = supp ( A ∗ ) .

Remark 3.9 . The parameter t δ , which determines the concentration properties of the estimator, decreases as T , p , and q increase. Consequently, the estimator ˆ A S benefits from improved performance as the sample size T and latent dimensions p and q grow.

Theorem 3.8 characterizes the non-asymptotic properties of the sparse estimator ˆ A S under sparsity assumptions on the true parameter matrix A ∗ . The first result provides a Frobenius norm error bound that scales with the sparsity level ∥ A ∗ ∥ 0 . This bound is inversely related to the squared magnitude of β ∗ (the average entry in B ∗ ), indicating that stronger signals in the underlying parameter matrix B ∗ lead to improved estimation accuracy. This is similar to the phenomenon described in Theorem 3.6. The error bound also depends on both the threshold parameter τ and the concentration parameter t δ , which capture the impact of noise and sample size on the estimation performance. The threshold τ exhibits several important dependencies. It decreases with the sample size T , reflecting improved estimation with more observations. It similarly decreases with dimensions p and q , showcasing a beneficial effect of higher dimensionality. Conversely it scales with the noise level σ , capturing the detrimental impact of increased noise levels. Finally it grows logarithmically with the matrix dimensions n and m , indicating a mild sensitivity to the size of A ∗ . The second result establishes conditions for perfect support recovery of A ∗ . Specifically, when the minimum magnitude of the

<!-- formula-not-decoded -->

nonzero entries in A ∗ exceeds three times the threshold τ , and the average effect β ∗ is sufficiently large ( 3 β ∗ &gt; 1 ), the sparse estimator exactly recovers the support of A ∗ .

As noted in Remark 3.9, the concentration parameter t δ decreases with larger values of T , p , and q . This property, combined with the similar behavior of τ , demonstrates that the sparse estimator benefits from both increased sample size and higher latent dimensions, a particularly favorable characteristic for high-dimensional settings.

## 4 Numerical Simulations

## 4.1 Synthetic data

Now we evaluate the performance of the proposed estimators through numerical simulations. Simulation Setup : The simulations involve the generation of matrices A ∗ , B ∗ , X t , E t and Y t . By default, the parameters are set as n = 15 , m = 13 , p = 14 , q = 12 , T = 2000 , and σ = 1 . These default parameter values are adjusted to analyze the effects of n , m , p , q , T , and σ on the performance of the proposed estimators. A ∗ is a n × m matrix with random entries sampled from a uniform distribution over [0 , 1) and rows then normalized to sum to 1. B ∗ is a q × p matrix with random entries sampled from a uniform distribution over [0 , 1) . ( X t ) is a sequence of T matrices of size m × q with random entries sampled from a uniform distribution over [0 , 1) . ( E t ) is a sequence of T noise matrices of size n × p , with entries drawn from a Gaussian distribution with mean 0 and standard deviation σ = 1 . ( Y t ) is a sequence of T observation matrices, where Y t = A ∗ X t B ∗ + E t .

Estimation and Evaluation : The estimators ˆ A and ˆ B are computed using (4) and (3) respectively. To evaluate their performances, we vary the parameters n , m , p , q , and T to observe their impact on the estimation accuracy. For each parameter setting, we compute the Frobenius norm of the errors ∥ ˆ A -A ∥ F and ∥ ˆ B -B ∥ F together with the Operator norm of the errors ∥ ˆ A -A ∥ op and ∥ ˆ B -B ∥ op . These errors are plotted as functions of the varying parameters in Figure 1.

Validation of Theoretical Properties : From the plots, we observe that as T increases, Figure 1e, the errors in ˆ A and ˆ B decrease, indicating improved estimation accuracy with more data. As p and q increase, Figures 1c and 1d, the errors in ˆ A decrease and the errors in ˆ B increase. As m increases, Figures 1b, both the errors in ˆ A and ˆ B increase. As n increases, Figure 1a, the errors in ˆ A increase and the errors in ˆ B decrease. These results confirm the theoretical properties of the estimators, detailed in Theorems 3.6 and 3.3. Additionally, we have performed numerical simulations to support the statement from Corollary 2.4. Appendix B.2 provides additional experiments.

Figure 1: Evolution (EV) of the Frobenius norm (resp. operator norm) of ˆ A -A ∗ (in blue, resp. in green) and of ˆ B -B ∗ (in orange, resp. in red) with respect to ( w.r.t. ) different parameters.

<!-- image -->

Weobserve that the empirical error rates align closely with the theoretical rates (derived under ORT). It suggests that deviations from orthogonality may lead only to a mild degradation in performance. Moreover, the degradation of both rates with increasing m is confirmed by the simulations (Figure 1b). Thus, this observed asymmetry is not an artifact of loose analysis, but a consequence of the model's structural assumptions.

## 4.2 Real-world data

We also evaluate our proposed methods on real-world data using the CIFAR-10 dataset. It contains 50,000 training and 10,000 test RGB images, each of size 32 × 32 × 3 . The pixel values are nor-

malized to [0 , 1] for computational consistency. Our goal is to simulate noisy image transformations and assess the effectiveness of the correction techniques.

Noisy Transformations : We simulate noisy image transformations via left and right matrix multiplications with A ∗ and B ∗ respectively. First, we define A = I 32 + ϵE 1 , where E 1 is a 32 × 32 matrix with i.i.d. entries from a standard Gaussian distribution, and A ∗ is obtained by normalizing each row of A to sum to 1. Similarly, matrix B ∗ is given by B ∗ = I 32 + ϵE 2 , where E 2 is a 32 × 32 matrix with i.i.d. standard Gaussian entries. The parameter ϵ controls the noise level in both transformations. For both training and test images, the noisy transformation is applied independently to each color channel. The transformation for a given channel c ∈ { 1 , 2 , 3 } (corresponding to red, green, and blue) is defined as X ( c ) noisy = ( A ∗ ) -1 · X ( c ) or · ( B ∗ ) -1 , where X ( c ) or represents the c th channel of the original image and X ( c ) noisy is the corresponding noisy version.

Correction Process : The correction process is learnt on the training set. From the noisy transformed images, we estimate the correction matrices ˆ A ( c ) train and ˆ B ( c ) train using (4) and (3), processing color channel independently. Once the correction matrices ˆ A ( c ) train and ˆ B ( c ) train are computed, they are applied to the noisy test images to reconstruct the corrected test images. For each channel in a test image, the corrected channel is computed as X ( c ) corr = ˆ A ( c ) train · X ( c ) noisy · ˆ B ( c ) train . Figure 2 shows an example from the test set, illustrating the original image, its noisy version for ϵ = 0 . 02 , and the corresponding corrected image respectively.

Figure 2: Original, noisy, and corrected versions of the 11 th image from the test set for ϵ = 0 . 02 .

<!-- image -->

Evaluation of Correction Quality : To evaluate the effectiveness of the correction process, we compute Do,n := 3 ∑ c =1 ∥ X ( c ) or -X ( c ) noisy ∥ 2 F , the Frobenius distance between the original image and its noisy version, and Do,c = 3 ∑ c =1 ∥ X ( c ) or -X ( c ) corr ∥ 2 F the Frobenius distance between the original image and its corrected version. We plot in Figure 3 Do,n and Do,c as functions of the noise factor ϵ averaged over the entire test set. Appendix B.3 presents additional plots showing how reconstruction accuracy varies with the effective signal-to-noise ratio (SNR) under both Frobenius and max norms.

Figure 3: Do,n (blue) and Do,c (orange), averaged on the test set, as functions of ϵ . Error bars indicate standard deviations.

<!-- image -->

## Conclusion

The results demonstrate that the proposed correction process effectively mitigates the impact of noise. The corrected images closely approximate the original images, as shown by both qualitative (Figure 2) and quantitative (Figure 3) metrics. This methodology generalizes well to real-world data, underscoring the applicability of our framework beyond synthetic simulations.

## Acknowledgments

This research was carried out while the author was affiliated with Cornell University, where the main theoretical development and experiments were conducted. The completion of the manuscript and submission process took place after joining Capital Fund Management (CFM). The author gratefully acknowledges the academic environment at Cornell for fostering this work and the support of CFM during the finalization of the paper.

## References

- [1] Shane Barratt. A matrix gaussian distribution. arXiv preprint , arXiv:1804.11010, 2018.
- [2] Nayel Bettache. Matrix-valued Time Series in High Dimension . PhD thesis, Institut Polytechnique de Paris, 2024.
- [3] Nayel Bettache and Cristina Butucea. Two-sided matrix regression. arXiv preprint arXiv:2303.04694 , 2023.
- [4] Esther Boyle, Luca Regis, and Petar Jevtic. Matrix variate regression as a tool for insurers. Variance , 17(1), 2024.
- [5] Haïm Brezis. Functional Analysis, Sobolev Spaces and Partial Differential Equations . Springer Science &amp; Business Media, 2011.
- [6] Florentina Bunea, Yiyuan She, and Marten H. Wegkamp. Optimal selection of reduced rank estimators of high-dimensional matrices. The Annals of Statistics , 39(2):1282-1309, 2011.
- [7] Elynn Y. Chen and Jianqing Fan. Statistical inference for high-dimensional matrix-variate factor models. Journal of the American Statistical Association , 118(542):1038-1055, 2023.
- [8] Elynn Y Chen, Xin Yun, Rong Chen, and Qiwei Yao. Modeling multivariate spatial-temporal data with latent low-dimensional dynamics. arXiv preprint arXiv:2002.01305 , 2020.
- [9] Rong Chen, Han Xiao, and Dan Yang. Autoregressive models for matrix-valued time series. Journal of Econometrics , 222(1):539-560, 2021.
- [10] Rong Chen, Dan Yang, and Cun-Hui Zhang. Factor models for high-dimensional tensor time series. Journal of the American Statistical Association , 117(537):94-116, 2022.
- [11] Yin-Jen Chen and Minh Tang. Regression for matrix-valued data via kronecker products factorization. arXiv preprint , arXiv:2404.19220, 2024.
- [12] Julien Chhor, Rajarshi Mukherjee, and Subhabrata Sen. Sparse signal detection in heteroscedastic gaussian sequence models: sharp minimax rates. Bernoulli , 30(3):2127-2153, 2024.
- [13] Kenneth R Davidson and Stanislaw J Szarek. Local operator theory, random matrices and banach spaces. In Handbook of the geometry of Banach spaces , volume 1, pages 317-366. Elsevier, 2001.
- [14] A. Philip Dawid. Some matrix-variate distribution theory: notational considerations and a bayesian application. Biometrika , 68(1):265-274, 1981.
- [15] Eloísa Díaz-Francés and Francisco J Rubio. On the existence of a normal approximation to the distribution of the ratio of two independent normal random variables. Statistical Papers , 54:309-323, 2013.
- [16] Shanshan Ding and R. Dennis Cook. Matrix variate regressions and envelope models. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 80(2):387-408, 2018.
- [17] Robert Charles Geary. The frequency distribution of the quotient of two normal variates. Journal of the Royal Statistical Society , 93(3):442-446, 1930.

- [18] Christophe Giraud. Low rank multivariate regression. Electronic Journal of Statistics , 5:775799, 2011.
- [19] Christophe Giraud. Introduction to High-Dimensional Statistics . Chapman and Hall/CRC, 2021.
- [20] Arjun K. Gupta and Daya K. Nagar. Matrix variate distributions . Chapman and Hall/CRC, 2018.
- [21] Harold V. Henderson and Shayle R. Searle. The vec-permutation matrix, the vec operator and kronecker products: A review. Linear and Multilinear Algebra , 9(4):271-288, 1981.
- [22] David V Hinkley. On the ratio of two correlated normal random variables. Biometrika , 56(3):635-639, 1969.
- [23] Peter D. Hoff. Multilinear tensor regression for longitudinal relational data. The Annals of Applied Statistics , 9(3):1169-1193, 2015.
- [24] Hangjin Jiang, Baining Shen, Yuzhou Li, and Zhaoxing Gao. Regularized estimation of highdimensional matrix-variate autoregressive models. arXiv preprint , arXiv:2410.11320, 2024.
- [25] Norman L Johnson, Samuel Kotz, and Narayanaswamy Balakrishnan. Continuous univariate distributions, volume 2 , volume 289. John wiley &amp; sons, 1995.
- [26] Iain M. Johnstone. Gaussian estimation: Sequence and wavelet models. Unpublished Manuscript, December 2011.
- [27] Łukasz Kidzi´ nski and Trevor Hastie. Modeling longitudinal data using matrix completion. Journal of Computational and Graphical Statistics , 33(2):551-566, 2024.
- [28] Olga Klopp. Rank penalized estimators for high-dimensional matrices. Electronic Journal of Statistics , 5(none):1161 - 1183, 2011.
- [29] Dehan Kong, Baiguo An, Jingwen Zhang, and Hongtu Zhu. L2rm: Low-rank linear regression models for high-dimensional matrix responses. Journal of the American Statistical Association , 2020.
- [30] Clifford Lam and Zetai Cen. Matrix-valued factor model with time-varying main effects. arXiv preprint arXiv:2406.00128 , 2024.
- [31] Béatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. Annals of Statistics , pages 1302-1338, 2000.
- [32] Bing Li, Min Kyung Kim, and Naomi Altman. On dimension folding of matrix-or array-valued statistical objects. The Annals of Statistics , 38(2):1094-1121, 2010.
- [33] Charles F. Van Loan. The ubiquitous kronecker product. Journal of Computational and Applied Mathematics , 123(1):85-100, 2000. Numerical Analysis 2000. Vol. III: Linear Algebra.
- [34] George Marsaglia. Ratios of normal variables and ratios of sums of uniform variables. Journal of the American Statistical Association , 60(309):193-204, 1965.
- [35] Arak Mathai, Serge Provost, and Hans Haubold. Chapter 4: The matrix-variate gaussian distribution. In Multivariate Statistical Analysis in the Real and Complex Domains , pages 217-288. Springer, 2022.
- [36] Madan Lal Mehta. Random matrices . Elsevier, 2004.
- [37] Philippe Rigollet and Jan-Christian Hütter. High-dimensional statistics. arXiv preprint , arXiv:2310.19244, 2023.
- [38] Angelika Rohde and Alexandre B. Tsybakov. Estimation of high-dimensional low-rank matrices. The Annals of Statistics , 39(2):887-930, 2011.
- [39] Mark Rudelson. Invertibility of random matrices: norm of the inverse. Annals of Mathematics , pages 575-600, 2008.

- [40] Mark Rudelson and Roman Vershynin. Invertibility of random matrices: unitary and orthogonal perturbations. Journal of the American Mathematical Society , 27(2):293-338, 2014.
- [41] S Yaser Samadi and Lynne Billard. On a matrix-valued autoregressive model. Journal of Time Series Analysis , 2024.
- [42] Michel Talagrand. Upper and lower bounds for stochastic processes , volume 60. Springer, 2014.
- [43] Ruey S. Tsay. Matrix-variate time series analysis: A brief review and some new developments. International Statistical Review , 92(2):246-262, 2024.
- [44] Alexandre B. Tsybakov. Nonparametric estimators. Introduction to Nonparametric Estimation , pages 1-76, 2009.
- [45] Stephen Tu. Upper and lower tails for gaussian maxima. https: //stephentu.github.io/blog/probability-theory/2017/10/16/ upper-and-lower-tails-gaussian-maxima.html , 2017. Accessed: 2024-12-11.
- [46] Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027 , 2010.
- [47] D. J. De Waal. Matrix-valued distributions. Encyclopedia of Statistical Sciences , 2004.
- [48] Cheng Yu, Dong Li, Xinyu Zhang, and Howell Tong. Two-way threshold matrix autoregression. arXiv preprint , arXiv:2407.10272, 2024.
- [49] Ruofan Yu, Rong Chen, Han Xiao, and Yuefeng Han. Dynamic matrix factor models for high dimensional time series. arXiv preprint , arXiv:2407.05624, 2024.
- [50] Xu Zhang, Catherine C Liu, Jianhua Guo, KC Yuen, and AH Welsh. Modeling and learning on high-dimensional matrix-variate sequences. Journal of the American Statistical Association , pages 1-16, 2024.

## A On the Validity and Relaxation of Assumptions

In the main text, we assume that the noise matrices ( E t ) T t =1 have independent entries drawn from a centered Gaussian distribution with constant variance σ 2 . This corresponds to a matrix normal distribution with isotropic row and column covariances, and facilitates clean non-asymptotic analysis. However, our results can be extended to more general settings under mild additional technical effort.

In some proofs, we extend the analysis to a more general setting in which each E t is drawn independently from a matrix normal distribution MN n × p (0 , Σ r , Σ c ) [14, 47, 20, 1, 35], where Σ r ∈ R n × n and Σ c ∈ R p × p capture row-wise and column-wise dependencies in the noise. Imposing the normalization condition Σ c ⊗ Σ r = σ 2 I np corresponds to the setting presented in the core of the paper. Under this condition, the general matrix normal setting is equivalent to the isotropic case presented in the core paper.

This formulation allows us to present certain proofs, such as Lemma 3.1, within a broader framework.

ORT assumption. Consider the setting where each E t follows a matrix normal distribution MN n × p (0 , Σ r , Σ c ) , where Σ c ⊗ Σ r = σ 2 I np but without the ORT assumption. In this relaxed setting, Lemma 3.1 remains valid and continues to characterize the distribution of the central quantity ̂ C , although the matrix D now exhibits an anisotropic row covariance structure.

Extending the proofs of Theorems 3.6 and 3.3 to this setting is conceptually straightforward, but technically more involved. One would need to rely on concentration bounds for random matrices with independent but non-isotropic columns. These are available, for instance, in Section 5.5 of [46]. In this case, collinearities in X ⊤ X would naturally appear in the resulting bounds, as illustrated in results such as Theorem 5.62 of the same reference

Homoskedasticity. Assuming Σ c ⊗ Σ r = σ 2 I np , or equivalently that ( E t ) T t =1 have independent entries drawn from a centered Gaussian distribution with constant variance σ 2 plays a role analogous to the homoskedasticity assumption in classical linear regression. It reflects a setting in which the signal is entirely explained by the model, and the residuals are unstructured. While restrictive, this assumption is standard in theoretical analysis and provides a tractable foundation for deriving error bounds.

Relaxing this assumption to allow for general row-wise dependence while keeping column independence, still under the ORT Assumption, would require concentration results for random matrices with independent but non-identically distributed rows (see Section 5.4 of [46]). In this case, the noise term in our analysis would be governed by the full tensor-product covariance Σ c ⊗ Σ r , which would explicitly appear in the resulting bounds (e.g., Theorem 5.44 in [46]).

Relaxing both assumptions simultaneously presents a more significant challenge. To the best of our knowledge, current probabilistic tools do not yet offer sharp and tractable results in this fully general setting. However, this remains a promising direction for future work, potentially requiring new matrix concentration inequalities tailored to specific structured settings.

In summary, although the core analysis assumes isotropic Gaussian noise for clarity and tractability, the main techniques extend to more general noise structures under appropriate assumptions and with access to suitable matrix concentration inequalities.

## B Additional Numerical Analyses

This appendix presents complementary numerical studies that assess the robustness of the proposed estimators and quantify the alignment between empirical convergence rates and the theoretical predictions.

## B.1 Robustness to the Distribution and Normalization of A ∗ and B ∗

To verify that the simulation results in Section 4 are not sensitive to the amplitude or normalization of the true parameters, we repeated all experiments with entries of both A ∗ and B ∗ independently

drawn from a Uniform[0 , c ) distribution with c ∈ { 1 , 2 , 3 , 5 } , followed by row-wise ℓ 1 normalization of A ∗ to ensure identifiability. Across all values of c , the empirical convergence rates and qualitative dependencies with respect to the parameters ( n, m, p, q, T ) remain unchanged. This confirms that the results reported in Figure 1 are not specific to the original choice c = 1 .

Theoretical analysis shows that the model is identifiable once one of the parameter matrices is normalized. If instead the normalization were imposed on B ∗ , the algebraic expressions of Proposition 2.2 and Corollary 2.4 would be modified by exchanging the roles of m (the number of columns of A ∗ ) and q (the number of rows of B ∗ ). Consequently, the dependencies on these dimensions in the non-asymptotic bounds of Theorems 3.3 and 3.6 would also be interchanged. This structural asymmetry stems from the identifiability constraint itself and highlights that several normalization choices are possible. The decision to normalize A ∗ is primarily motivated by interpretability-its nonnegative, row-stochastic structure aligns with the notion of activation or mixing weights-and by analytical tractability of the resulting expressions.

## B.2 Quantitative Comparison Between Empirical and Theoretical Rates

This section reports the quantitative comparison between the empirical convergence slopes of the estimators ( ˆ A, ˆ B ) and the theoretical predictions derived from the finite-sample analysis. The objective is to evaluate how the estimation error scales with each model dimension ( n, m, p, q ) and with the sample size T under three norms: Frobenius, operator, and maximum absolute.

Experimental setup. For each parameter d ∈ { n, m, p, q, T } , we generated independent datasets while keeping all other quantities fixed, recomputed ( ˆ A, ˆ B ) , and measured their reconstruction errors

<!-- formula-not-decoded -->

The dependence of log( Err □ ( d )) on d was then fitted with a linear model

<!-- formula-not-decoded -->

where the function f ( d ) corresponds to:

<!-- formula-not-decoded -->

This distinction reflects that, in the simulations, both axes were represented on logarithmic scale for T , while for ( n, m, p, q ) the x -axis was linear and only the y -axis (error) was plotted in log scale. The fitted slope s ( □ ) d quantifies the empirical rate of variation of the estimation error with respect to d . For the sample size T , the theoretical prediction is s ( □ ) T = -1 2 .

Empirical slopes. Table 1 summarizes the fitted slopes for all parameters and norms. Positive slopes indicate that the error increases with the corresponding dimension, while negative slopes indicate a decrease.

Table 1: Empirical slopes s ( □ ) d of log( Err ) vs. log( d ) for ˆ A and ˆ B under the three norms.

| Parameter   | Max       | ˆ A Frobenius   | Operator   | Max       | ˆ B Frobenius   | Operator   |
|-------------|-----------|-----------------|------------|-----------|-----------------|------------|
| n           | +0.003    | +0.011          | +0.008     | - 0 . 011 | - 0 . 011       | - 0 . 012  |
| m           | +0.007    | +0.015          | +0.011     | +0.018    | +0.016          | +0.016     |
| p           | - 0 . 012 | - 0 . 011       | - 0 . 011  | +0.002    | +0.011          | +0.008     |
| q           | - 0 . 005 | - 0 . 005       | - 0 . 005  | +0.008    | +0.017          | +0.013     |
| T           | - 0 . 532 | - 0 . 496       | - 0 . 474  | - 0 . 579 | - 0 . 526       | - 0 . 526  |

Findings. Several patterns emerge consistently across all norms:

- Sample-size dependence. For both ˆ A and ˆ B , the slopes with respect to T lie between -0 . 58 and -0 . 47 , matching the theoretical prediction s T = -1 2 . This confirms that estimation errors decay at the expected T -1 / 2 rate.

- Dimensional dependencies. For ˆ A , errors increase with m and decrease with ( p, q ) , whereas for ˆ B the opposite trend holds-errors increase with ( m,p,q ) but slightly decrease with n . This asymmetric pattern is consistent with the theoretical structure of the model, where the identifiability constraint on A ∗ leads to mirrored dependencies in ˆ A and ˆ B .
- Norm-specific behavior. The Frobenius norm shows the strongest sensitivity to dimensional changes, reflecting its dependence on all matrix entries; the operator norm shows a weaker dependence consistent with a ( √ p + √ q ) scaling; and the maximum norm lies between these two regimes.
- Goodness of fit. The linear relationships between log( Err ) and log( d ) exhibit high explanatory power for most dimensions, with R 2 values above 0.9 in the majority of cases, confirming the robustness of the observed trends.

Plots. Figure 4 illustrates the empirical error curves for all parameters. Each subplot reports the average reconstruction error (in logarithmic scale) for ˆ A and ˆ B under the three norms. The first four panels correspond to variations in the structural dimensions ( n, m, p, q ) , while the bottom panel shows the dependence on the sample size T . The trends confirm the fitted slopes in Table 1: errors decrease approximately linearly on the log scale as T grows, consistent with the T -1 / 2 rate, and vary smoothly with ( n, m, p, q ) according to the theoretical sign pattern. Notably, ˆ B exhibits larger sensitivity to q and smaller sensitivity to n , while ˆ A shows the opposite, reflecting the asymmetric normalization of A ∗ . Across all panels, the Frobenius norm produces the steepest gradients, the operator norm the weakest, and the maximum norm lies in between, reproducing the hierarchy predicted by the theoretical bounds.

Summary. The empirical analysis confirms that the proposed estimators obey the predicted nonasymptotic scaling laws. The T -1 / 2 convergence rate holds precisely, and the dependence on ( n, m, p, q ) follows the signs and magnitudes expected from the theoretical bounds. These quantitative findings validate that the theoretical error expressions accurately capture the dominant sources of variation in finite-sample performance.

<!-- image -->

Parameter T

Figure 4: Empirical reconstruction errors of ˆ A and ˆ B versus model dimensions and sample size T , displayed on a logarithmic scale. Each curve corresponds to a given norm (Frobenius, operator, or max).

## B.3 Reconstruction Error as a Function of Noise Level

To further evaluate robustness to noise, we examined the reconstruction quality of the bilinear estimator under controlled perturbations of A ∗ and B ∗ . For each noise level ϵ , we generated perturbed matrices

<!-- formula-not-decoded -->

where Z A , Z B are Gaussian random matrices with i.i.d. N (0 , 1) entries, followed by row-wise normalization of A ϵ to preserve identifiability. Each perturbed pair ( A ϵ , B ϵ ) was used to transform the training and test datasets, and the reconstruction was then estimated using the procedure described in Section 4.

Effective signal-to-noise ratio (SNR). Rather than plotting the reconstruction error directly against ϵ , we parameterize the x-axis in terms of an effective signal-to-noise ratio (SNR), defined as

<!-- formula-not-decoded -->

where X is the original image and X noisy its transformed version. This reparametrization provides a scale-invariant measure of perturbation strength and directly reflects the degradation of signal energy in Frobenius norm.

Results. Figure 5 reports the reconstruction error as a function of the effective SNR for the Frobenius distance, while Figure 6 shows the analogous result for the element-wise maximum norm. In both cases, the distance between the original and corrected images (orange) is consistently below that between the original and noisy images (blue), confirming that the estimator effectively compensates for multiplicative perturbations in A ∗ and B ∗ . The monotonic growth of both curves as SNR decreases quantifies the degradation rate, with the Frobenius error emphasizing global reconstruction quality and the max-norm capturing local, element-wise discrepancies.

Figure 5: Reconstruction error versus effective SNR (Frobenius norm). Lower distances indicate improved correction quality. The x-axis is expressed in dB following the definition SNR F = 20 log 10 ( ∥ X ∥ F / ∥ X -X noisy ∥ F ) .

<!-- image -->

Figure 6: Reconstruction error versus effective SNR (element-wise maximum norm). This complementary metric highlights the preservation of fine-scale details under increasing noise.

<!-- image -->

Overall, the analysis confirms that the proposed correction procedure maintains stable performance across a broad range of noise intensities and that the reconstructed images preserve both global and local structure in accordance with the theoretical robustness guarantees.

## C Proofs

## C.1 Proofs of Section 2

We first analyze a simplified case of (2) where the design matrices ( X t ) T t =1 are the elements of the canonical basis of R m × q . In this setup we are given T = mq observations ( X t , M t ) T t =1 where each predictor X t is a matrix with all entries set to zero except for a single entry equal to one. This setting is well studied in the vector case, typically referred to as a Sequence Model. Such models have been widely explored in the literature [44, 26]. By focusing on this simple scenario, we aim to uncover insights that can extend to more complex cases.

Formally, for ( k, l ) ∈ [ m ] × [ q ] , let M ( k,l ) corresponds to the basis element E ( k,l ) , where E ( k,l ) is the matrix with all entries equal to zero except for a one in the k -th row and l -th column. The model (2) can then be expressed as:

<!-- formula-not-decoded -->

which corresponds to (2) for t = k +( l -1) m . The matrix E ( k,l ) acts as a selector, isolating the effect of the k -th row of A ∗ and the l -th row of B ∗ .

Lemma C.1. In the model (7) , the parameters A ∗ and B ∗ can be explicitly recovered from the observed matrices M ( k,l ) as follows:

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Proof of Lemma C.1. From the model (7), we have

<!-- formula-not-decoded -->

where E ( k,l ) is the matrix with a single entry of 1 at the ( k, l ) -th position and zeros elsewhere. Expanding this relationship element-wise gives:

<!-- formula-not-decoded -->

Step 1: Recovery of [ B ∗ ] lj .

Summing over k ∈ [ m ] for fixed l, j, i , we observe that:

<!-- formula-not-decoded -->

Since the rows of A ∗ have unit L 1 -norm, it follows that:

<!-- formula-not-decoded -->

Step 2: Recovery of [ A ∗ ] ik .

̸

From the model equation, for each ( k, l ) , we isolate [ A ∗ ] ik by dividing [ M ( k,l ) ] ij by [ B ∗ ] lj , provided that [ B ∗ ] lj = 0 :

̸

<!-- formula-not-decoded -->

Lemma C.2. In the model (2) , where the design matrices X t form a generating family of R m × q , the unobserved matrix C ∈ R mq × np satisfies the following equality:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma C.2. We notice that M = XC by construction, which concludes the proof.

Proof of Proposition 2.2. From Lemma C.2, the unobserved matrix C can be expressed as:

<!-- formula-not-decoded -->

This means that C can be computed directly from the observations ( X t , M t ) t ∈ [ T ] , provided that X ⊤ X is invertible, which is ensured by the assumption that the design matrices ( X t ) T t =1 form a generating family of R m × q .

Step 1: Relating C to A ∗ and B ∗ .

By the definition of C , each entry [ C ] ( i,j ) ( k,l ) corresponds to:

<!-- formula-not-decoded -->

for all i ∈ [ n ] , k ∈ [ m ] , l ∈ [ q ] , and j ∈ [ p ] . This bilinear structure allows us to recover A ∗ and B ∗ separately by exploiting their roles in this product.

Step 2: Recovering B ∗ .

To isolate [ B ∗ ] lj , we sum [ C ] ( i,j ) ( k,l ) over all k ∈ [ m ] for a fixed l, j, i . Specifically:

<!-- formula-not-decoded -->

Since the rows of A ∗ satisfy the L 1 -norm constraint (i.e., ∑ m k =1 [ A ∗ ] ik = 1 ), the summation simplifies to:

<!-- formula-not-decoded -->

This establishes the second equation in the proposition:

<!-- formula-not-decoded -->

Step 3: Recovering A ∗ .

To isolate [ A ∗ ] ik , we use the bilinear relationship:

<!-- formula-not-decoded -->

For a fixed i, k, l, j , we can solve for [ A ∗ ] ik provided that [ B ∗ ] lj = 0 :

<!-- formula-not-decoded -->

This establishes the first equation in the proposition:

̸

<!-- formula-not-decoded -->

Proof of Corollary 2.4. The result on B ∗ follows immediately from Proposition 2.2. To prove the result on A ∗ , we need to prove that for any n ∈ N ∗ , if there exist ( α 1 , . . . , α n ) ∈ R n and ( β 1 , . . . , β n ) ∈ R n such that:

then it follows that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Specifically, since the entries of A ∗ can be expressed as different ratios, all being equal, applying this result to the equations satisfied by A ∗ in Proposition 2.2 completes the proof.

We prove this result by induction on n .

Initially at n = 1 the result is trivially true.

Assume the statement holds at step n , i.e., if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then:

We now prove that the statement holds at step n +1 .

Suppose:

<!-- formula-not-decoded -->

By the definition of γ n +1 and the assumption of step n being true, we can write:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally multiplying the numerator by β n +1 and dividing α n +1 by the same factor provides:

<!-- formula-not-decoded -->

Thus, the statement holds at step n +1 .

## C.2 Proofs of Section 3

Proof of Lemma 3.1. We first recall that, by construction, we have:

<!-- formula-not-decoded -->

This allows to write:

Hence we deduce:

Next we prove the following, which will conclude:

<!-- formula-not-decoded -->

First we recall that the noise matrices ( E t ) T t =1 are assumed to be independent and follow the same matrix normal distribution, namely MN n × p (0 n × p , Σ r , Σ c ) . Hence by definition of the matrix normal distribution:

<!-- formula-not-decoded -->

This leads to derive the distribution of the matrix E ∈ R T × np defined in ( ?? ) as follows:

<!-- formula-not-decoded -->

Finally, the matrix ( X ⊤ X ) -1 X ⊤ being of full rank mq and ( X ⊤ X ) -1 being symmetric, by definition of the matrix D ∈ R mq × np and by property of affine transformations of matrix gaussian distributions, we conclude that

<!-- formula-not-decoded -->

Proof of Theorem 3.3. We start by deriving the finite-sampled inequalities on ˆ B . From Lemma 3.1 and Assumption 3.2 we deduce that vec ( ̂ C ) ∼ N mqnp ( vec ( C ) , Σ c ⊗ Σ r T ) . As mentioned in the introduction Σ c ⊗ Σ r = σ 2 I np which ensures that all entries of vec ( ̂ C ) are independent and follow the same Gaussian distribution. Hence for all ( k, l, i, j ) ∈ [ m ] × [ q ] × [ n ] × [ p ] we have ̂ C ( k,l ) , ( i,j ) -C ( k,l ) , ( i,j ) i.i.d ∼ N ( 0 , σ 2 T ) . Using the results from Proposition 2.2 and (3) lead to, for all ( l, j ) ∈ [ q ] × [ p ] ,

<!-- formula-not-decoded -->

Mill's inequality, Theorem E.1, ensures that for all ( l, j ) ∈ [ q ] × [ p ] , for any ϵ &gt; 0 ,

<!-- formula-not-decoded -->

The first inequality is then deduced by using a union bound.

The second inequality is immediately derived from the concentration of extreme singular values of Gaussian matrices with independent entries, see Corollary 5.35 in [46].

For the third inequality, we first note that ∥ ˆ B -B ∗ ∥ 2 F = q ∑ l =1 p ∑ j =1 ( [ ˆ B ] lj -[ B ∗ ] lj ) 2 follows a mσ 2 nT ·

χ 2 ( pq ) distribution. Then, Lemma 1 from [31] ensures that a random variable Z following a χ 2 ( pq ) distribution satisfies, for any ϵ &gt; 0 :

<!-- formula-not-decoded -->

Hence we deduce that

<!-- formula-not-decoded -->

We notice that, for any ϵ &gt; 0 , the following inequality holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It leads to:

The stated result follows by a change of variable.

Proof of Lemma 3.5. Let us fix ( i, k ) ∈ [ n ] × [ m ] . Using the definitions of ˆ A and ˜ A from (4) and (3), respectively, we find that [ ˜ A ] ik = [ ˆ A ] ik if and only if [ ˜ A ] ik ∈ [0 , 1] . Hence [ ˜ A ] ik = [ ˆ A ] ik if and only if the event A holds, where:

<!-- formula-not-decoded -->

This event is realized if and only if the event A + or the event A -holds, where:

and

<!-- formula-not-decoded -->

Using Fréchet inequalities stated in Theorem E.2, we get:

<!-- formula-not-decoded -->

Under Assumption 3.2, Lemma 3.1 gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where:

and

Under the same assumptions, as detailed in the proof of Theorem 3.3:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

By construction, ˆ γ ( k,i ) and ˆ β i are independent. In addition, as γ ∗ ( k,i ) = [ A ∗ ] ik β ∗ , Assumption 3.4 ensures that their expected values satisfy:

<!-- formula-not-decoded -->

Using the symmetry of the Gaussian distribution, we deduce:

<!-- formula-not-decoded -->

The event A + holds if B 1 and B 2 hold simultaneously, where:

<!-- formula-not-decoded -->

Using Fréchet inequalities, stated in Theorem E.2:

<!-- formula-not-decoded -->

We now bound from above the probability of the events B 1 and B 2 separately. For B 1 , we note:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As we have γ ∗ ( k,i ) &gt; 0 , using (9) and Mill's inequality stated in Theorem E.1 provides:

<!-- formula-not-decoded -->

For B 2 we note that:

<!-- formula-not-decoded -->

The complementary event ¯ B 2 is:

<!-- formula-not-decoded -->

As we have β ∗ -γ ∗ ( k,i ) &gt; 0 , using independence of ˆ γ ( k,i ) and ˆ β i , results from (9), (10) and Mill's inequality ensures:

<!-- formula-not-decoded -->

Finally, using:

we deduce the result.

Proof of Theorem 3.6. Let us fix ( i, k ) ∈ [ n ] × [ m ] . We work on the event A := { [ ˜ A ] ik = [ ˆ A ] ik } . Using (3), we get:

<!-- formula-not-decoded -->

Proposition 2.2 guarantees:

<!-- formula-not-decoded -->

By definition of β ∗ and using the reverse triangle's inequality we get:

where:

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The complementary event ¯ B 1 is:

Under Assumption 3.2, (9) and (10) hold. Using (9) and Mill's inequality stated in Theorem E.1, we get for any ϵ &gt; 0 :

<!-- formula-not-decoded -->

Using (10) and Mill's inequality, we find for any ϵ &gt; 0 :

<!-- formula-not-decoded -->

where ˙ n := ( n -1) .

Finally, on the event A , we have ∣ ∣ ∣ [ ˜ A ] ik ∣ ∣ ∣ ≤ 1 . We then use the result from Lemma 3.5, Fréchet inequality stated in Theorem E.2, and we conclude with a union bound.

Proof of Theorem 3.7. For all ( l, j ) ∈ [ q ] × [ p ] , (8) ensures that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where:

From the results in Appendix D, we obtain that the event A holds with probability at least 1 -δ where A is the event

<!-- formula-not-decoded -->

We recall the definition of the threshold:

<!-- formula-not-decoded -->

On the event A , we observe the following:

- If | [ ˆ B ] lj | &gt; 2 τ , then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- If | [ ˆ B ] lj | ≤ 2 τ , then:

From these observations, we deduce:

<!-- formula-not-decoded -->

Rewriting this equality with indicator functions, we have:

This leads to the bound:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we conclude:

This proves the first point.

For the second point, on the event A and under the stated assumption, we observe the following:

̸

Hence | [ ˆ B S ] lj | = 0 .

̸

̸

̸

Hence | [ B ∗ ] lj | = 0 .

Proof of Theorem 3.8. For all ( i, k ) ∈ [ n ] × [ m ] , (9) ensures that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where:

and

Proposition 2.2 ensures that

From the results in Appendix D, we obtain that the event A holds with probability at least 1 -δ where A is the event

<!-- formula-not-decoded -->

We recall the definition of the threshold:

<!-- formula-not-decoded -->

On the event A , we observe the following:

- If | ˆ γ ik | &gt; 2 τ , then:

<!-- formula-not-decoded -->

Thus we have [ A ∗ ] ik = 0 and because [ A ∗ ] ik is bounded from above by 1 we also have β ∗ ≥ τ.

- If | ˆ γ ik | ≤ 2 τ , then:

<!-- formula-not-decoded -->

- If | [ ˆ B S ] lj | = 0 , then | [ ˆ B ] lj | &gt; 2 τ . This provides the bound:

<!-- formula-not-decoded -->

- If | [ B ∗ ] lj | = 0 , then | [ B ∗ ] lj | &gt; 3 τ . This provides the bound:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition of the Frobenius norm, we obtain:

<!-- formula-not-decoded -->

̸

From these observations, we deduce:

<!-- formula-not-decoded -->

Rewriting this with indicator functions and using the previously stated implications, we have:

Multiplying all terms by | β ∗ | leads to the bound:

Using the equality

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the triangle inequality leads to where

and

Moreover we notice that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equations (9) and (10), together with Mill's inequality, stated in Theorem E.1, provide for any t &gt; 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Finally, we get for any δ ∈ (0 , 1) with probability at least 1 -δ :

̸

✶ Dividing by | β ∗ | -1 and using that γ ∗ ik = [ A ∗ ] ik β ∗ ensures that with probability at least 1 -δ :

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

By definition of the Frobenius norm, we obtain with probability at least 1 -δ :

<!-- formula-not-decoded -->

Finally we get with probability at least 1 -δ :

<!-- formula-not-decoded -->

For the second point, on the event A and under the stated assumptions, we observe the following:

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

- If | [ ˆ A S ] ik | = 0 , then | ˆ γ ik | &gt; 2 τ . This provides the bound:

<!-- formula-not-decoded -->

Hence | γ ∗ lj | = 0 and thus [ A ∗ ] ik = 0 .

̸

## D Tails of the Maximum of Absolute Gaussian Variables

In this appendix, adapted from [45], we analyze the upper and lower tails of the random variable max 1 ≤ i ≤ n | X i | , where X 1 , . . . , X n are independent and identically distributed (i.i.d.) random variables following a standard normal distribution N (0 , 1) . Specifically, we aim to establish bounds for the upper and lower tails of this random variable with high probability, which is crucial for understanding the behavior of maxima in Gaussian settings.

We begin by stating the main result.

Theorem D.1. Fix δ ∈ (0 , 1) , and let X 1 , . . . , X n be i.i.d. N (0 , 1) . With probability at least 1 -δ ,

<!-- formula-not-decoded -->

The asymmetry in the tails arises from the fact that max 1 ≤ i ≤ n | X i | is bounded below by zero almost surely, it is not bounded above by any fixed constant. To establish the theorem, we separately analyze the upper and lower tails and combine these results with a union bound.

## D.1 Upper Tail

The upper tail is analyzed using concentration results for the suprema of Gaussian processes. The key tool is Talagrand's concentration inequality, Lemma 2.10.6 in [42], stated in a simpler version as follows:

Lemma D.2. Consider T ⊆ R n and let g ∼ N (0 , I ) . Then, for any u &gt; 0 ,

<!-- formula-not-decoded -->

where s := sup t ∈ T E [ ⟨ t, g ⟩ 2 ] 1 / 2 .

To apply this inequality, we consider T = { e 1 , . . . , e n , -e 1 , . . . , -e n } , where e i is the i -th canonical basis vector in R n . For g ∼ N (0 , I ) , this gives

<!-- formula-not-decoded -->

Applying Lemma D.2, we obtain, for any δ &gt; 0 :

<!-- formula-not-decoded -->

Thus, with probability at least 1 -δ :

<!-- formula-not-decoded -->

The next step is to bound E [max 1 ≤ i ≤ n | X i | ] from above.

- If | [ A ∗ ] ik | = 0 , then | [ A ∗ ] ik | &gt; 3 τ . Using the equality γ ∗ ik = β ∗ [ A ∗ ] ik and the assumption on the bound satisfied by β ∗ provide:

<!-- formula-not-decoded -->

## D.2 Expected Maximum of Gaussian Variables

We now bound E [max 1 ≤ i ≤ n | X i | ] .

Proposition D.3. Let Z 1 , . . . , Z n be n random variables (not necessarily independent) with marginal distribution N (0 , 1) . Then,

<!-- formula-not-decoded -->

Proof. Fix any λ &gt; 0 . Observe that

<!-- formula-not-decoded -->

Taking the logarithm of both sides,

<!-- formula-not-decoded -->

Applying Jensen's inequality,

<!-- formula-not-decoded -->

Dividing through by λ ,

Optimizing by setting λ = √ 2 log n , we conclude

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For X 1 , . . . , X n i.i.d. following a standard normal distribution N (0 , 1) , considering the 2 n variables ( X 1 , . . . , X n , -X 1 , . . . , -X n ) , we have:

<!-- formula-not-decoded -->

Substituting this into (11) completes the proof of the upper tail.

## D.3 Lower Bound

We now analyze the lower tail of max 1 ≤ i ≤ n | X i | . Fix a positive τ &gt; 0 . Then,

<!-- formula-not-decoded -->

Using the independence of X 1 , . . . , X n , we get:

<!-- formula-not-decoded -->

The Gauss error function, defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

allows then to write:

The inequality erf( x ) 2 ≤ 1 -e -4 x 2 /π , which holds for all x ≥ 0 , provides:

<!-- formula-not-decoded -->

Finally, the inequality 1 -x ≤ e -x , which holds for all x ∈ R ensures:

<!-- formula-not-decoded -->

To derive the lower bound, set

<!-- formula-not-decoded -->

and solve for τ . We find that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

## E Probability bounds and inequalities

## E.1 Mill's inequality

Theorem E.1 (Mill's Inequality) . Let X be a Gaussian random variable with mean µ and variance σ 2 . Then, for any t &gt; 0 , the following inequality holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By symmetry, we also have:

Proof. A proof of this theorem can be found in [37].

## E.2 Fréchet inequalities

Theorem E.2 (Fréchet inequalities) . Let A and B be two events. The probability of their intersection satisfies:

<!-- formula-not-decoded -->

The probability of their union satisfies:

<!-- formula-not-decoded -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction are carefully aligned with the paper's actual contributions, as detailed in Section 1.3. These claims are neither overstated nor misleading. They accurately describe the scope of the work, including the introduction of optimization-free estimators, non-asymptotic error bounds, and the applicability to matrix-valued regression problems. The theoretical and empirical results presented in the core sections fully support these contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of the work are discussed in Appendix A, where we explicitly address the simplifying assumptions made in the theoretical analysis, such as the use of isotropic Gaussian noise, and explain how the results could be extended under more general settings. We also acknowledge the technical challenges involved in such extensions and provide references to the appropriate concentration inequalities required for handling nonisotropic noise or more complex covariance structures.

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

Justification: Each theoretical result in the paper is stated with a complete and explicit set of assumptions. The main assumptions are clearly outlined, and each theorem or lemma is accompanied by a full proof provided in Appendix C. The proofs are rigorous, and where simplifications are made (e.g., assuming isotropic noise), we clarify this explicitly and discuss how the results can be extended under milder assumptions in Appendix A.

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

Justification: The paper provides a detailed description of the experimental setup, including the data generation process, model parameters, and evaluation metrics used in all simulations. Each figure in the results section is directly tied to a clearly defined experimental protocol. The code is included in the supplementary material and all information necessary to reproduce the results relevant to the main claims and conclusions is fully disclosed. The experiments serve to validate the theoretical findings, and no critical step is omitted in their description.

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

Justification: The full implementation is provided as a Jupyter notebook in the supplementary material, including all code and data needed to reproduce the main experimental results. The notebook contains clear documentation and step-by-step instructions, ensuring that the simulations and figures presented in the paper can be faithfully reproduced. This supports the transparency and reproducibility of the empirical claims made in the work. See https://github.com/nayelbettache/BMLR .

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

Justification: For the synthetic experiments, all parameters, such as sample size, matrix dimensions, noise levels, and distributional assumptions, are fully specified. The estimators

are closed-form and do not involve optimization or hyperparameter tuning, so no additional training procedure is required. For the real-data experiments on CIFAR-10, the paper uses the standard train/test split provided by the dataset's official loader, which is explicitly mentioned in the text. No additional tuning or fine-tuning is performed, and all relevant details are disclosed to ensure full understanding of the results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: For the simulated data, the paper evaluates performance across a wide range of parameter settings (e.g., sample size, matrix dimensions, noise levels), which helps mitigate the effects of randomness and ensures that the empirical results robustly reflect the theoretical predictions. This systematic variation serves as a form of sensitivity analysis. For the real-world CIFAR-10 experiments, the paper explicitly reports error bars to convey the variability of the estimators and support the statistical significance of the results.

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

Justification: The experiments consist primarily of synthetic simulations and closed-form estimators, which are computationally lightweight and do not require specialized hardware. No GPU is needed, and all experiments can be run on a standard laptop or CPU-based machine. While exact runtimes are not reported, the simplicity and efficiency of the estimators ensure that the results are reproducible without significant computational resources. Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research adheres fully to the NeurIPS Code of Ethics. The work is theoretical and empirical in nature, with experiments conducted on synthetic data and the publicly available CIFAR-10 dataset. No personal, sensitive, or proprietary data is used. All methods are described transparently, reproducibility is supported through open supplementary materials, and no foreseeable negative societal impacts or misuse of the research have been identified.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: No dedicated section is included in the paper to discuss societal impacts. However, the work is theoretical in nature and focuses on a methodological contribution to matrix-valued regression. As such, it does not directly target any high-risk application domains. While the methods may have positive downstream impact in areas such as spatiotemporal modeling or image analysis, no immediate or foreseeable negative societal impacts are associated with the research.

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

Justification: The paper does not involve the release of high-risk models or datasets. It focuses on theoretical analysis and simulation-based evaluation, along with experiments on the publicly available CIFAR-10 dataset. No pretrained models, sensitive data, or generative systems are used.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All external assets used in the paper are publicly available and properly credited. The CIFAR-10 dataset is cited appropriately and used in accordance with its terms of use. Any third-party libraries or tools used for simulations or experiments (e.g., NumPy, TensorFlow, scikit-learn) follow open-source licenses and are acknowledged either directly in the text or within the supplementary notebook. No proprietary or restricted-use resources are employed.

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

Answer: [Yes]

Justification: The paper introduces new simulation code and experiments, which are provided in a well-documented Jupyter notebook included in the supplementary material. The code includes clear instructions, parameter settings, and explanations necessary to reproduce the results. No new datasets or models involving human subjects or requiring consent are introduced. All assets are anonymized to comply with the double-blind review policy.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing or research with human subjects. All experiments are conducted on synthetic data and the publicly available CIFAR-10 dataset, which does not contain personally identifiable information or require participant interaction.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve human subjects or study participants. All experiments are conducted using synthetic data or the CIFAR-10 dataset, which is publicly available and does not involve any human interaction or identifiable information. Therefore, no IRB approval was required.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core methods, theoretical analysis, and experimental components of the paper do not involve the use of large language models (LLMs) in any important, original, or non-standard way. Any use of LLMs was limited to editing support and does not affect the scientific contributions of the work.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
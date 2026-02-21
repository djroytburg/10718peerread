## Understanding Generalization in Physics Informed Models through Affine Variety Dimensions

Takeshi Koshizuka 1 and Issei Sato 1

1 Department of Computer Science, The University of Tokyo 1 {koshizuka-takeshi938444, sato}@g.ecc.u-tokyo.ac.jp

## Abstract

Physics-informed machine learning is gaining significant traction for enhancing statistical performance and sample efficiency through the integration of physical knowledge. However, current theoretical analyses often presume complete prior knowledge in non-hybrid settings, overlooking the crucial integration of observational data, and are frequently limited to linear systems, unlike the prevalent nonlinear nature of many real-world applications. To address these limitations, we introduce a unified residual form that unifies collocation and variational methods, enabling the incorporation of incomplete and complex physical constraints in hybrid learning settings. Within this formulation, we establish that the generalization performance of physics-informed regression in such hybrid settings is governed by the dimension of the affine variety associated with the physical constraint, rather than by the number of parameters. This enables a unified analysis that is applicable to both linear and nonlinear equations. We also present a method to approximate this dimension and provide experimental validation of our theoretical findings.

## 1 Introduction

In recent years, physics-informed machine learning (PIML) has garnered significant attention [35, 25, 12, 19]. PIML represents a hybrid approach that integrates physical knowledge into machine learning models for tasks involving physical phenomena. These hybrid models can leverage inherent physical structures, such as differential equations [36], conservation laws [23], and symmetries [3], as inductive biases. This integration has the potential to enhance both sample efficiency and generalization capabilities. These models have been empirically applied to a wide range of phenomena, with successful applications observed in areas including thrombus material properties [41], fluid dynamics [8, 24], turbulence [40], and heat transfer problems [9].

Despite these empirical successes, the theoretical analysis of PIML remains underdeveloped, which potentially undermines the reliability of these methods. Notably, in practical scenarios, prior knowledge of the governing differential equations, particularly their source terms or boundary conditions, is often incomplete. Consequently, learning frequently involves a hybrid approach of fitting to actual observational data alongside incorporating physical constraints. However, much of the existing theoretical research focuses on settings where complete prior knowledge of the differential equations is assumed. Furthermore, the scope of existing analyses is often limited to linear differential equations or systems exhibiting strong regularity, creating a gap between theory and application.

To bridge this gap, we propose a versatile analytical framework for physics-informed regression using linear hypothesis classes over nonlinear features in hybrid settings. Our key idea is to formulate the differential equation constraints by introducing a Unified Residual Form. This form, defined on a finite set of trial functions and a measure, provides a practical approximation of physical constraints. This formulation unifies the collocation-based constraints, used in Physics-Informed Neural Networks

(PINNs) [36], and the standard unified residual form constraints, used in the variational and finite element methods. Within this framework, the learning weights of a physics-informed linear regressor are shown to be defined on an affine variety associated with the differential equation. Crucially, our analysis aims to elucidate the impact of incorporating physical prior knowledge on the generalization capacity of these models. We then establish that the generalization capacity of these models is determined by the dimension of this affine variety, rather than by the number of parameters. This novel perspective enables a unified analysis applicable to various equations, including nonlinear ones. To support our theoretical findings, we introduce a method for approximately calculating the dimension of the affine variety and provide extensive experimental validation. Our results illustrate that even in scenarios with a large number of parameters relative to the amount of data, the incorporation of physical structure reduces the intrinsic dimension of the hypothesis space, thereby mitigating overfitting and corroborating our theoretical claims.

## 2 Related Work

Since the seminal work by Raissi et al. [36] on PINNs, PIML has rapidly emerged as a significant field of study. This area has been comprehensively surveyed in the literature by [35, 25, 12, 19]. Leveraging the high function approximation capabilities of neural networks [20, 26, 14], these models have been employed as versatile surrogates for solving various equations. In contrast, linear models are also used because of their interpretability, consistency with classical numerical solvers [5, 18], and the close relationship between Partial Differential Equations (PDEs) and kernel methods [37, 11, 27, 13, 16]. Recently, methods that exploit underlying conservation laws [23, 21] and symmetries [3, 13], in addition to the equations themselves, have also been developed.

Recent studies have made advances in the theoretical understanding of PINNs. Shin [38] rigorously showed that the minimizer of the PINN loss converges to the strong solution as the data size approaches infinity for linear elliptic and parabolic PDEs under certain conditions. These findings were extended by Shin et al. [39] into a general framework applicable to broader linear problems, with the loss function formulated in both strong and variational forms. Mishra and Molinaro [30, 31] use the stability properties of the underlying PDEs to derive upper bounds on the generalization error of PINNs. Subsequent research has applied this analytical framework to various specific equations [6, 29]. However, studies explicitly addressing the impact of physical structure on generalization capabilities are still limited. Arnone et al. [5] proved that for second-order elliptic PDEs, the physicsinformed linear estimator using a finite element basis converges at a rate surpassing the Sobolev minimax rate. Doumèche et al. [15] quantified the generalization capacity of the physics-informed estimator for general linear PDEs using the concept of effective dimension [10], a well-known metric in the analysis of the kernel method. The effects of incorporating the structures of nonlinear complex equations, as well as conservation laws and symmetries, into models on generalization, have yet to be thoroughly analyzed.

## 3 Minimax risk Analysis

In this section, we explain how introducing physical structures can improve the generalization capacity of linear models. In Section 3.1, we outline the problem setup. In Section 3.2, we perform a minimax risk analysis, showing that the generalization capacity is mainly determined by the dimension of the affine variety. In Section 3.3, we show that our theory aligns with existing theories on linear operators. Notations are summarized in Appendix A.

## 3.1 Problem Setup

We consider the regression problem, which aims to learn the unknown function f ∗ : R m → R that satisfies the differential equation. We have a dataset consisting of n observations, denoted as { ( x i , y i ) } n i =1 , where x i ∈ Ω ∪ ∂ Ω represents the input within the domain Ω ⊆ R m or the boundary ∂ Ω and y i ∈ R represents the corresponding output. Observations are sampled independently from a probability distribution P on the domain Ω ∪ ∂ Ω × R . The relationship between the observations and the true function can be expressed as:

<!-- formula-not-decoded -->

where ε i represents normally distributed noise with mean zero and variance σ 2 . The target function f ∗ is the solution of the differential equation, i.e ., D [ f ∗ ] = 0 for a given operator D : L 2 (Ω) → L 2 (Ω) , where L 2 (Ω) denotes the space of square-integrable functions on a domain Ω ⊆ R m . For a more detailed background on the problem setting, please refer to Appendix C.

Unified Residual Form: To formulate prior knowledge of the governing differential equations, we first introduce a unified residual form, which captures physical constraints in an integrated or averaged sense. Such formulations naturally arise in variational and finite element methods, and are particularly well-suited to hybrid settings where only partial physical supervision is available. Formally, let T := { ( ψ k , µ k ) } K k =1 be a finite collection of trial functions and measure pairs, where each ψ k : R m → R is a smooth trial function and µ k : Σ → R is a measure on the σ -algebra Σ over the domain Ω . Then, the unified residual form of the knwon differential equation D is defined by

<!-- formula-not-decoded -->

We then impose the differential equation constraint through the unified residual form defined above. The resulting physics-informed regression problem reads:

<!-- formula-not-decoded -->

where λ n is a regularization parameter and ∥ · ∥ denotes the standard L 2 norm. This formulation relaxes the classical smoothness requirements while still leveraging physics-informed constraints via a unified measure-based approach: choosing Borel measures leads to an approximation of standard weak solutions, whereas choosing Dirac measures leads to an approximation of the strong-form residuals used in the PINN framework.

Physics-Informed Linear Regression (PILR) Setup: Let B = { ϕ j : R m → R } d j =1 be a fixed basis. Define the basis vector ϕ ( x ) = [ ϕ 1 ( x ) , ϕ 2 ( x ) , . . . , ϕ d ( x )] ⊤ ∈ R d , the design matrix Φ = [ ϕ ( x 1 ) , ϕ ( x 2 ) , . . . , ϕ ( x n )] ⊤ ∈ R n × d , and the target vector y = [ y 1 , . . . , y n ] ⊤ ∈ R n .

The physics-informed feasible set is

<!-- formula-not-decoded -->

The problem Eq. (1) reduces to the physics-informed linear regression given by

<!-- formula-not-decoded -->

where V R = V ( D , B , T ) ∩ B 2 ( R ) is the affine variety constrained by the ℓ 2 -ball B 2 ( R ) with radius R &gt; 0 and ∥ · ∥ 2 is the ℓ 2 -norm.

The set of coefficients V constitutes an affine variety as it represents the set of solutions to the K polynomial equations in the d variables with real coefficients. For example, when m = 1 and D [ f ] = f · d d x f , the affine variety V is defined by the solution set of the polynomial equations p k ( w ) = ∑ d j,j ′ =1 ⟨ ( d d x ϕ j ) ϕ j ′ , ψ k ⟩ µ k w j w j ′ = 0 for k = 1 , . . . , K . We perform minimax risk analysis based on the dimension d V of this affine variety because the affine variety V is crucial to determine the size of the intrinsic hypothesis space.

Minimax risk: The goal of our analysis is to obtain the upper bound of the minimax risk for PILR in Eq. (3), which is defined by

<!-- formula-not-decoded -->

Here, w ∗ ∈ V R represents the optimal weight vector. The corresponding optimal hypothesis f w ∗ = w ∗⊤ ϕ within our hypothesis space H = { w ⊤ ϕ : w ∈ V R } is defined as the best approximation of the true function f ∗ : f w ∗ = w ∗⊤ ϕ = arg min f w ∈H ∥ f w -f ∗ ∥ 2 .

We strongly recommend referring to the example in Section 5.1 to intuitively understand our problem setting.

## 3.2 Main Theorem

In this section, we present an upper bound on the minimax risk for PILR. The bound is interpretable and sufficiently sharp, revealing how physical constraints reduce hypothesis complexity and enhance generalization. We begin by stating the definition and assumptions underpinning our analysis.

Definition 3.1 ( ( β, d V ) -regular set) . An affine variety V ⊆ R d is called a ( β, d V ) -regular set if the following conditions hold: (1) For almost all affine subspaces L ⊆ R d of dimension d L satisfying d -d L ≤ d V , the intersection V ∩ L has at most β path-connected components. (2) For almost all affine subspaces L ⊆ R d of dimension d L with d -d L &gt; d V , the intersection V ∩ L is empty. See Appendix B.2 for illustrative explanations.

Assumption 3.2 (Boundedness of basis functions) . For the basis function ϕ = [ ϕ 1 , . . . , ϕ d ] ⊤ , where ϕ j ∈ B , assume that there exists a positive constant M such that ∥ ϕ ( x ) ∥ 2 ≤ M for all x ∈ Ω .

Assumption 3.3. Assume there exists a constant η &gt; 0 such that 1 √ n ∥ Φ w ∥ 2 ≥ √ η ∥ w ∥ 2 for all w ∈ B 2 (2 R ) .

Assumption 3.4 (Stability of estimator) . Assume there exists a constant Γ &gt; 1 such that ∥ ˆ w 1 -ˆ w 2 ∥ 2 ≤ (Γ -1) ∥ w ∗ 1 -w ∗ 2 ∥ 2 , for the estimators ˆ w 1 and ˆ w 2 of the optimal weights w ∗ 1 and w ∗ 2 , respectively.

Next, we present the upper bound on the minimax risk. The complete proof is provided in Appendix D.

Theorem 3.5 (Minimax Risk Bound) . Let V ( D , B , T ) be the ( β, d V ) -regular affine variety defined in Eq. (2) . Suppose Assumptions 3.2-3.4 hold. Then, there exists a positive constant C , independent of n , d V , d , and β , such that for any δ ∈ (0 , 1) , with probability at least 1 -δ , the minimax risk for PILR defined by Eq. (4) is bounded by

<!-- formula-not-decoded -->

Proof Sketch. The proof proceeds in two steps. In the first step, we upper bound the minimax risk by the supremum of a sub-Gaussian random process defined over the metric space ( V R , ∥ · ∥ 2 ) . The second step utilizes Dudley's integral theorem, which bounds the supremum of the process by an integral involving its covering number, specifically: ∫ ∞ 0 √ N ( V R , ε, ∥ · ∥ 2 ) d ε . To apply Dudley's theorem effectively, we employ Lemma B.2 to obtain an explicit upper bound for the covering number. Substituting this bound into Dudley's integral and performing the integration yields the desired high-probability minimax risk bound.

Theorem 3.5 demonstrates that the minimax risk is primarily governed by the intrinsic dimension d V of the affine variety V , rather than the ambient input dimension d , particularly when the topological complexity parameter β is small. For comparison, standard least-squares estimation over an ℓ 2 -ball B 2 ( R ) ⊂ R d yields a minimax risk rate of order O ( √ d/n ) , which is optimal for unconstrained linear regression in d -dimensional space. In contrast, our result shows that when d V ≪ d , incorporating physical structure into the hypothesis space through differential constraints significantly sharpens the risk rate, yielding improved generalization.

On the Role of β . The parameter β captures the topological complexity of the affine variety and appears as a regularity constant in the generalization bound. Its upper bound can be estimated via the Petrovskii-Oleinik-Milnor inequality [33, 32, 28], which provides a bound on the sum of Betti numbers of a semialgebraic set. Specifically, if the variety V ∩ B 2 ( R ) ⊂ R d is defined by polynomial constraints { p k ( w ) } K k =1 of maximal degree ρ , then it is ( ρ (2 ρ -1) d +1 , d V ) -regular. This implies that as the degree ρ of the defining polynomials increases, the variety can exhibit more intricate topological features, such as additional holes and disconnected components.

How d V and β Arise from the Covering Argument. The minimax risk is bounded via Dudley's entropy integral, which requires control over the covering number N ( V R , ε, ∥ · ∥ 2 ) . Following the geometric approach of Zhang and Kileel [42], the affine variety V ⊂ R d is sliced using a family of

linear subspaces { L s } s ∈ N , and each intersection V ∩ L s is covered by Euclidean balls of radius ε . The total covering is then given by

<!-- formula-not-decoded -->

In this construction, the intrinsic dimension d V controls the number of subspaces required to sufficiently cover V , while the parameter β , corresponding to the sum of Betti numbers, governs the covering number of each individual section V ∩ L s . Topologically, β can be interpreted as quantifying the number of topological features (e.g., holes) in V , and thus reflects the local geometric complexity encountered within each subspace. For reference, the standard covering number of the Euclidean ball satisfies N ( B 2 ( R ) , ε, ∥ · ∥ 2 ) ≤ (1 + 2 R/ε ) d , highlighting the advantage of replacing ambient-dimension dependence with complexity parameters intrinsic to the constraint set.

Key Insights. A central contribution of our analysis is its interpretability through the lens of intrinsic complexity measures. The dimension d V plays a role analogous to the VC dimension in classification [1] or the pseudo-dimension in regression [34], serving as a proxy for the effective capacity of the hypothesis space. This dimensional viewpoint clarifies how the incorporation of physical constraints-via differential equation structure-can substantially reduce hypothesis complexity, even in high-dimensional ambient spaces. While this may come at the cost of slightly looser constants compared to minimax-optimal bounds, the resulting rate is still sharp enough to meaningfully capture the generalization benefit of physics-informed inductive bias. Empirical evidence supporting this theoretical advantage is presented in Section 5, and an alternative analysis via Rademacher complexity is provided in Appendix F.

Effect of the Trial Function Set T . The set of trial functions T encodes the imposed physical constraints, typically derived from a governing differential operator D . The cardinality K = |T | quantifies the amount of physical knowledge embedded in the learning problem. Increasing the number of trial functions leads to a more restrictive constraint set, which geometrically corresponds to a lower-dimensional affine variety. Specifically, if T 1 ⊂ T 2 , then it follows that

<!-- formula-not-decoded -->

which highlights how adding more physical constraints systematically reduces hypothesis complexity and improves generalization behavior.

## 3.3 Analysis on Linear Operator

We discuss the special case where D is a linear operator. The second term in Eq. (5) vanishes because the Petrovskii-Oleinik-Milnor inequality indicates β = 1 . Thus, the minimax risk is O ( √ d V log( d V d ) /n ) . Furthermore, the affine variety V is the solution set of a homogeneous system of linear equations. That is, the affine variety can be written as V ( D , B , T ) = { w : Dw = 0 } using the matrix D ∈ R K × d defined by D k,j := ⟨ D [ ϕ j ] , ψ k ⟩ µ k . The affine variety is a linear subspace of dimension d V = dimker D . From the rank-nullity theorem, d V = d -rank D , indicating that the higher the rank of the matrix D , the better the minimax risk of regression.

We show that our theory is consistent with existing theories. The effect of incorporating physical structure, represented by linear differential equations, on generalization has been analyzed within the framework of kernel methods by Doumèche et al. [15, 16]. They argued that the physical structure smooths the kernel and reduces the effective dimension, leading to an improvement in the ℓ 2 predictive error. We first present the definition of the physics-informed (PI) kernel.

Definition 3.6 (PI kernel [15, 16]) . Given a basis B = { ϕ j } d j =1 , trial functions (with a single measure) T = { ( ψ k , µ ) } K k =1 , and a linear operator D , the PI kernel associated with the affine variety V ( D , B , T ) = { w ∈ R d : Dw = 0 } is defined as:

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Here, I is the identity matrix, and the matrix T is positive semi-definite. The parameters ξ, ν ≥ 0 control the balance between the L 2 -regularization and the constraints derived from the operator D .

Doumèche et al. [16] showed the effective dimension d eff ( ξ, ν ) of the PI kernel is evaluated above by a computable quantity as follows:

<!-- formula-not-decoded -->

where σ ( · ) denotes the spectrum (set of eigenvalues) of the given matrix, B ∈ R d × d is the Gram matrix of the basis function, i.e ., B j,j ′ = ⟨ ϕ j , ϕ j ′ ⟩ µ . Next, we provide an explicit upper bound on the effective dimension of the PI kernel defined using the affine variety:

Proposition 3.7. The effective dimension of the PI kernel associated with the affine variety V ( D , B , T ) = { w : Dw = 0 } with dimension d V is upper bounded by

<!-- formula-not-decoded -->

where { α j } d j = d V denote the positive eigenvalues of the matrix D ⊤ TD .

Proposition 3.7 indicates that as the dimension of the affine variety d V = d -rank D decreases, the upper bound on the effective dimension of the PI kernel decreases accordingly. Since the matrix D ⊤ TD is positive semi-definite, all eigenvalues satisfy α j ≥ 0 . The intrinsic dimension d V corresponds precisely to the number of zero eigenvalues ( α j = 0 ). The terms in the second sum ( j &gt; d V ) involve strictly positive eigenvalues α j &gt; 0 . Given that ν &gt; 0 , we have 1 1+ ξ + να j &lt; 1 1+ ξ . Thus, when the intrinsic dimension d V decreases, the number of terms in the first sum (with the larger value 1 / (1 + ξ ) ) decreases, while the number of terms in the second sum (with smaller values 1 / (1 + ξ + να j ) ) increases. This shift towards smaller-valued terms leads to an overall reduction in the complexity bound.

Consequently, our theoretical results align with the existing PI kernel theory [15, 16]. The PI kernel framework from the previous literature quantifies the complexity of the hypothesis space through the entire spectrum of the matrix D combined with the base kernel ⟨ ϕ ( x ) , ϕ ( y ) ⟩ 2 , restricting the analysis primarily to linear target operators D . In contrast, our approach allows for analysis of linear and non-linear operators by focusing solely on the intrinsic dimension d V (the count of zero eigenvalues), rather than analyzing the entire eigenvalue spectrum.

## 4 On the Dimension of an Affine Variety

In general, the dimension of the affine variety V = { w ∈ R d : p k ( w ) = 0 , ∀ k = 1 , . . . , K } defined by polynomials { p k } K k =1 has many equivalent definitions. In particular, the following statements are all equivalent.

Definition 4.1. The maximal length of the chains V 0 ⊂ V 1 ⊂ . . . ⊂ V d V of non-empty subvarieties of V .

Definition 4.2. The degree of the denominator of the Hilbert series of the affine variety V .

Definition 4.3. The maximal dimension of the tangent vector spaces at the non-singular points U ⊆ V ⊂ R d of the variety, i.e ., d V = max w ∈ U d -rank [ ∇ p 1 ( w ) · · · ∇ p K ( w )] ⊤ .

Although Definition 4.1 clearly indicates that the dimension represents the complexity of the set V , it is difficult to calculate the dimension according to this definition. Definition 4.2 shows that the dimension represents the algebraic complexity of the polynomial ring. Definition 4.3 characterizes the dimension based on the local structure of the affine variety, making it suitable for numerical calculation as discussed in Section 4.2. It generalizes the rank-nullity theorem d V = d -rank D in the linear case, as mentioned in Section 3.3. The details of the concepts associated with these definitions are given in Appendix B.

## 4.1 Lower Bound

We demonstrate that the dimension d V of the affine variety can be characterized by the linear part of the operator D .

Proposition 4.4. Suppose the operator D can be decomposed as D = L + F , where L is a nonzero linear differential operator and F is a nonlinear operator. Then, we have d V ( L ) ≤ d V ( D ) .

Combining the result of Proposition 4.4 with Theorem 3.5 suggests that the nonlinear part F of the operator increases the affine variety dimension, which has a negative effect on generalization. Furthermore, the dimension of the affine variety associated with the linear part L can be easily calculated by the rank of the matrix. Therefore, the lower bound of the dimension of the affine variety associated with the nonlinear operator D can be easily determined, allowing us to estimate the minimum required amount of data n .

## 4.2 Numerical Calculation Method

According to Definition 4.2, the dimension of an affine variety is typically obtained by calculating the degree of the denominator of the Hilbert series, by using Gröbner bases. However, the worst-case time complexity of Buchberger's algorithm [7], the standard method for computing Gröbner bases, is double exponential in the number of variables d . Therefore, on the basis of 4.3, we approximate d V by sampling w ∗ 1 , . . . , w ∗ N from the affine variety V with a suitable distribution and then computing max w ∗ ∈{ w ∗ 1 ,..., w ∗ N } d -rank ( ∇ ⊤ [ p 1 ( w ∗ ) , . . . , p K ( w ∗ )] ⊤ ) . When the operator D is nonlinear, we perform simulations with various boundary conditions and project the obtained solutions onto the basis B to sample w ∗ ∈ V . For linear operators, the dimension does not depend on the particular weight w , and the rank of the matrix D discussed in Section 3.3 precisely determines d V . Assuming the use of standard rank computation algorithms, the computational complexity of this numerical approach is O ( N · min( K,d ) Kd ) for the nonlinear case, and O (min( K,d ) Kd ) for the linear case. This complexity is practical and feasible for most scenarios considered in our setting.

## 5 Experiments

To evaluate the generalization performance of physics-informed linear regression (PILR) compared to ridge regression (RR) using basis functions B , we conducted experiments on representative differential equations. We varied the data size n and parameter count d , and report test MSE (mean ± standard deviation) across 10 random initial or boundary conditions. Experimental details are provided in Appendix G.

When the operator D is linear, PILR approximates the solution to Eq. (3) as [16]:

<!-- formula-not-decoded -->

where M depends on hyperparameters ξ and ν (see Eq. (6)); setting ν = 0 yields RR.

For nonlinear equations, we train models by minimizing a soft-constrained loss using the Adam optimizer. Hyperparameters ξ and ν are tuned via validation MSE.

## 5.1 Learning Strong Solutions

In this section, we investigate the strong solutions of the classical harmonic oscillator and the diffusion equation with periodic boundary conditions, by employing the Dirac measure, which corresponds to the collocation method used in PINNs. The solutions to these equations can be obtained analytically. Through these straightforward examples, we demonstrate both analytically and numerically that the generalization performance is determined by the dimension of the affine variety.

Harmonic Oscillator The initial value problem of a harmonic oscillator D [ y ] = 0 with a spring constant k s and mass m s in the domain Ω = [0 , T ] is given by:

<!-- formula-not-decoded -->

where y 0 and v 0 are the initial position and velocity, respectively. The solution to the initial value problem is analytically given by y ( t ) = y 0 cos( ωt ) + v 0 ω sin( ωt ) , ω = √ k s /m s . The settings for the basis and the trial functions with the measure ϕ j ∈ B , ( ψ k , µ k ) ∈ T of indices 1 ≤ j ≤ d t and 1 ≤ k ≤ K t are as follows:

<!-- formula-not-decoded -->

Figure 1: Experimental results for the strong solutions. (a, b) Test MSE (log scale) vs. number of parameters for the harmonic oscillator (a) and diffusion equation (b). The plots compare RR and PILR for three different data sizes n , showing the mean and standard deviation across 10 initializations. (c) Predictions of harmonic oscillator using a 33-parameter model trained on 20 samples: RR and PILR, with training data points indicated.

<!-- image -->

where ω j := jπ T is the j -th frequency and δ x k is the Dirac measure centered at the point x k ∈ Ω , which is uniformly sampled from data.

Then, the dimension of the affine variety is d V = 2 , representing the essential degrees of freedom of the solution. Figure 1a supports our theory experimentally. For RR, the generalization performance degrades as the number of parameters d = 2 d t +1 increases due to overfitting, as shown in Fig. 1c. In contrast, for PILR, the performance remains stable regardless of the number of parameters d owing to the lower dimension of the affine variety d V = 2 .

Diffusion Equation The initial value problem for the one-dimensional diffusion equation D [ u ] = 0 with diffusion coefficient c and periodic boundary conditions is given by:

<!-- formula-not-decoded -->

We define the basis functions ϕ ∈ B and the test functions with measures ( ψ, µ ) ∈ T as follows:

<!-- formula-not-decoded -->

where the frequency is ω j = jπ/ Ξ . The indices are in the ranges 0 ≤ j ≤ d x , 0 ≤ j ′ ≤ d t , 1 ≤ k ≤ K t , and 1 ≤ k ′ ≤ K x .

The analytical solution is expressed as a linear combination of the above basis functions. The number of bases is d = 2 d x d t +1 , while the dimension of an affine variety is given by d V = 2min( d x , d t )+1 . Figure 1b shows the results when we set α = 1 . 0 , j max = 1 , d t = 2 , and vary d x . The results indicate that the generalization performance of PILR does not deteriorate as d x increases, in contrast to RR.

## 5.2 Learning Weak Solutions

In this section, we investigate weak solutions for the harmonic oscillator and the diffusion equation, employing a variational framework with Borel measures. The governing equations and basis functions are identical to those in Section 5.1.

Harmonic Oscillator We define the trial functions ψ k for 1 ≤ k ≤ K t as:

<!-- formula-not-decoded -->

where ω k := kπ T is the frequency. The associated measure µ k is the Lebesgue measure on Ω = [0 , T ] .

The dimension of the affine variety remains d V = 2 , consistent with the strong solutions. Experimental results in Fig. 2a confirm this. The performance of PILR is stable and independent of the number of basis functions, unlike RR, which shows performance degradation as model complexity increases.

Figure 2: Experimental results for the weak solutions. (a, b) Test MSE (log scale) vs. number of parameters for the harmonic oscillator (a) and diffusion equation (b). The plots compare RR and PILR for three different data sizes n , showing the mean and standard deviation across 10 initializations.

<!-- image -->

Diffusion Equation The trial functions ψ k,k ′ combine a piecewise constant basis in time and a Fourier basis in space. For indices 1 ≤ k ≤ K t and 1 ≤ k ′ ≤ K x , they are:

<!-- formula-not-decoded -->

where 1 [ t k ,t k +1 ] ( t ) is the indicator function for the time interval, defined as

<!-- formula-not-decoded -->

and ω k ′ = k ′ π Ξ . The associated measure is the Lebesgue measure on [ -Ξ , Ξ] × [0 , T ] .

The dimension of the affine variety, d V = 2min( d x , d t ) + 1 , is identical to the strong solution case. The results in Fig. 2b show that PILR's generalization performance remains robust as the number of spatial basis functions d x increases, demonstrating its advantage over RR.

## 5.3 Learning Numerical Solutions

In this section, we learn approximate solutions using numerical methods that use finite difference for four equations. In this setting, we consider the affine variety of the difference equation D h and the base functions B h and the trial functions with the measure T h corresponding to the numerical method with step size h . We first validate our theory using linear and nonlinear Bernoulli equations discretized by the explicit Euler method.

Discrete Bernoulli Equation We discretize the Bernoulli equation on the interval Ω = [0 , T ] with uniform step size h :

<!-- formula-not-decoded -->

where y τ = y ( t τ ) . We consider two parameter regimes ( P, Q, ρ ) set to (1 . 0 , 0 . 0 , 0 . 0) for the linear case and to (1 . 0 , 0 . 5 , 2) for the non-linear case. The initial value y 0 is sampled from N (0 , 1) , and the reference solution is calculated explicitly by Euler. Further details on the choice of n t , basis/trial functions, measure ( ψ τ , µ τ ) , and implementation are given in Appendix G.2.

Discrete Diffusion Equation We discretize the one-dimensional diffusion equation over Ω = [ -Ξ , Ξ] × [0 , T ] with step sizes h = ( h x , h t ) and diffusion coefficient c ( u ) :

<!-- formula-not-decoded -->

where u τ j = u ( x j , t τ ) . Weconsider two cases: c ( u ) = 1 . 0 for the linear case and c ( u ) = 0 . 1 / (1+ u 2 ) for the nonlinear case. Periodic boundary conditions are imposed in x . More details on the grid, the basis / trial functions, and the numerical setup are given in Appendix G.2.

Tables 1 and 2 show that PILR achieves a higher performance than RR for large values of d . While the dimension d V is independent of the time discretization step size in the Euler method, it depends on the spatial discretization step size in the FDM. We include supplementary experiments in Appendix H, where we fix the ambient dimension d and vary the size of the trial-function set T .

Table 1: Experimental results for the discrete linear and nonlinear Bernoulli equations approximated by the explicit Euler method. The settings include various step sizes h . The number of parameters (basis) d , and the calculated dimension of the affine variety d V .

| Settings   | D h h       | Linear Bernoulli eq.      | Linear Bernoulli eq.               | Nonlinear Bernoulli eq.            | Nonlinear Bernoulli eq.            |
|------------|-------------|---------------------------|------------------------------------|------------------------------------|------------------------------------|
| Settings   | D h h       | 1 / 100                   | 1 / 200                            | 1 / 100                            | 1 / 200                            |
| Dimensions | d           | 100                       | 200 1                              | 100                                | 200 1                              |
| Test MSE   | d V RR PILR | 1 48 ± 0 . 012 ± 0 . 0025 | 0 . 63 ± 0 . 43 0 . 011 ± 0 . 0013 | 1 0 . 60 ± 0 . 41 . 013 ± 0 . 0024 | 0 . 72 ± 0 . 49 0 . 013 ± 0 . 0018 |

Table 2: Experimental results for the discrete linear and nonlinear diffusion equations approximated by the FDM. The settings include various step sizes h = ( h t , h x ) . The number of parameters (basis) d , and the calculated dimension of the affine variety d V .

| Settings   | D h          | Linear diffusion eq.    | Linear diffusion eq.    | Nonlinear diffusion eq.   | Nonlinear diffusion eq.   |
|------------|--------------|-------------------------|-------------------------|---------------------------|---------------------------|
| Settings   | ( h t ,h x ) | (1 / 400 , 2 / 10) 4010 | (1 / 400 , 2 / 20) 8020 | (1 / 200 , 2 / 10) 2010   | (1 / 200 , 2 / 20) 4020   |
| Dimensions | d d V        | 10 2 . 21 ± 0 . 56      | 20 2 . 14 ± 0 . 57      | 10 1 . 12 ± 0 . 40        | 20 1 . 11 ± 0 . 40        |
| Test MSE   | RR PILR      | 1 . 13 ± 0 . 30         | 0 . 79 ± 0 . 16         | 0 . 26 ± 0 . 11           | 0 . 22 ± 0 . 10           |

## 5.4 Impact of Basis Misspecification on Generalization

This section considers a practical scenario where the basis functions are misspecified, a situation that can occur during manual design or through random selection, as in an Extreme Learning Machine (ELM) [22]. Any such misspecification can degrade performance by increasing the approximation error . As detailed in Appendix C.4, the total error is composed of this approximation error and an estimation error. While our theory demonstrates that physical constraints can reduce the estimation error, the overall model performance is limited by the magnitude of the approximation error.

To demonstrate this effect, we conducted an experiment on the Harmonic Oscillator, intentionally omitting the known analytical frequency from the basis functions. Other experimental settings were identical to those in Section 5.1. With 10 data points, the performance was exceptionally poor. For a basis size of d = 17 , the test MSE was approximately 1 . 435 ± 0 . 646 , of which the approximation error constituted nearly the entire amount at 1 . 430 . Increasing the basis size to d = 33 had a negligible effect; the test MSE remained high at 1 . 434 as the approximation error was unchanged.

This result clearly shows the total error being dominated by the approximation error. It underscores a prerequisite for our theory: the improvement in generalization from physics-informed constraints is achieved only when the model possesses sufficient expressive capacity to represent the true solution.

## 6 Conclusion

This study introduces a framework for analyzing physics-informed models through the lens of affine varieties induced by the governing differential equations. We establish that generalization performance is governed by the dimension of this variety, rather than the number of model parameters, a finding that unifies existing theories for linear equations. We further provide a method for calculating this dimension and present experimental validation confirming that this intrinsic dimension effectively mitigates overfitting in highly parameterized settings. Although our analysis centers on linear regression models, the proposed geometric framework is broadly applicable to both linear and nonlinear differential equations, as our experiments demonstrate. This work offers a foundational, geometric interpretation of generalization that establishes a promising, though challenging, direction for future theory-guided model selection, such as the optimal choice of basis and trial functions. Future work includes the extension and validation of our framework for other architectures, such as NNand ELM. The framework can also be extended to differential equations with unknown parameters by analyzing an augmented parameter space.

## References

- [1] Yaser S Abu-Mostafa. The vapnik-chervonenkis dimension: Information versus complexity in learning. Neural Computation , 1(3):312-317, 1989.
- [2] Radoslaw Adamczak. A tail inequality for suprema of unbounded empirical processes with applications to markov chains. 2008.
- [3] Tara Akhound-Sadegh, Laurence Perreault-Levasseur, Johannes Brandstetter, Max Welling, and Siamak Ravanbakhsh. Lie point symmetry and physics-informed networks. Advances in Neural Information Processing Systems , 36, 2024.
- [4] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery &amp; data mining , pages 2623-2631, 2019.
- [5] Eleonora Arnone, Alois Kneip, Fabio Nobile, and Laura M Sangalli. Some first results on the consistency of spatial regression with partial differential equation regularization. Statistica Sinica , 32(1):209-238, 2022.
- [6] Genming Bai, Ujjwal Koley, Siddhartha Mishra, and Roberto Molinaro. Physics informed neural networks (pinns) for approximating nonlinear dispersive pdes. arXiv preprint arXiv:2104.05584 , 2021.
- [7] Bruno Buchberger. A theoretical basis for the reduction of polynomials to canonical forms. ACM SIGSAM Bulletin , 10(3):19-29, 1976.
- [8] Shengze Cai, Zhicheng Wang, Frederik Fuest, Young Jin Jeon, Callum Gray, and George Em Karniadakis. Flow over an espresso cup: inferring 3-d velocity and pressure fields from tomographic background oriented schlieren via physics-informed neural networks. Journal of Fluid Mechanics , 915:A102, 2021.
- [9] Shengze Cai, Zhicheng Wang, Sifan Wang, Paris Perdikaris, and George Em Karniadakis. Physics-informed neural networks for heat transfer problems. Journal of Heat Transfer , 143(6): 060801, 2021.
- [10] Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics , 7:331-368, 2007.
- [11] Yifan Chen, Bamdad Hosseini, Houman Owhadi, and Andrew M Stuart. Solving and learning nonlinear pdes with gaussian processes. Journal of Computational Physics , 447:110668, 2021.
- [12] Salvatore Cuomo, Vincenzo Schiano Di Cola, Fabio Giampaolo, Gianluigi Rozza, Maziar Raissi, and Francesco Piccialli. Scientific machine learning through physics-informed neural networks: Where we are and what's next. Journal of Scientific Computing , 92(3):88, 2022.
- [13] David Dalton, Dirk Husmeier, and Hao Gao. Physics and lie symmetry informed gaussian processes. In Forty-first International Conference on Machine Learning , 2024.
- [14] Tim De Ryck and Siddhartha Mishra. Error analysis for physics-informed neural networks (pinns) approximating kolmogorov pdes. Advances in Computational Mathematics , 48(6):79, 2022.
- [15] Nathan Doumèche, Francis Bach, Gérard Biau, and Claire Boyer. Physics-informed machine learning as a kernel method. In The Thirty Seventh Annual Conference on Learning Theory , pages 1399-1450. PMLR, 2024.
- [16] Nathan Doumèche, Francis Bach, Gérard Biau, and Claire Boyer. Physics-informed kernel learning. arXiv preprint arXiv:2409.13786 , 2024.
- [17] Richard M Dudley. The sizes of compact subsets of hilbert space and continuity of gaussian processes. Journal of Functional Analysis , 1(3):290-330, 1967.

- [18] Federico Ferraccioli, Laura M Sangalli, and Livio Finos. Some first inferential tools for spatial regression with differential regularization. Journal of Multivariate Analysis , 189:104866, 2022.
- [19] Zhongkai Hao, Songming Liu, Yichi Zhang, Chengyang Ying, Yao Feng, Hang Su, and Jun Zhu. Physics-informed machine learning: A survey on problems, methods and applications. arXiv preprint arXiv:2211.08064 , 2022.
- [20] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks , 2(5):359-366, 1989.
- [21] Zheyuan Hu, Ameya D Jagtap, George Em Karniadakis, and Kenji Kawaguchi. When do extended physics-informed neural networks (xpinns) improve generalization? SIAM Journal on Scientific Computing , 44(5):A3158-A3182, 2022.
- [22] Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew. Extreme learning machine: theory and applications. Neurocomputing , 70(1-3):489-501, 2006.
- [23] Ameya D Jagtap, Ehsan Kharazmi, and George Em Karniadakis. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems. Computer Methods in Applied Mechanics and Engineering , 365:113028, 2020.
- [24] Xiaowei Jin, Shengze Cai, Hui Li, and George Em Karniadakis. Nsfnets (navier-stokes flow nets): Physics-informed neural networks for the incompressible navier-stokes equations. Journal of Computational Physics , 426:109951, 2021.
- [25] George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics , 3(6):422-440, 2021.
- [26] Gitta Kutyniok, Philipp Petersen, Mones Raslan, and Reinhold Schneider. A theoretical analysis of deep neural networks and parametric pdes. Constructive Approximation , 55(1):73-125, 2022.
- [27] Da Long, Zheng Wang, Aditi Krishnapriyan, Robert Kirby, Shandian Zhe, and Michael Mahoney. Autoip: A united framework to integrate physics into gaussian processes. In International Conference on Machine Learning , pages 14210-14222. PMLR, 2022.
- [28] John Milnor. On the betti numbers of real varieties. Proceedings of the American Mathematical Society , 15(2):275-280, 1964.
- [29] Siddhartha Mishra and Roberto Molinaro. Physics informed neural networks for simulating radiative transfer. Journal of Quantitative Spectroscopy and Radiative Transfer , 270:107705, 2021.
- [30] Siddhartha Mishra and Roberto Molinaro. Estimates on the generalization error of physicsinformed neural networks for approximating a class of inverse problems for pdes. IMA Journal of Numerical Analysis , 42(2):981-1022, 2022.
- [31] Siddhartha Mishra and Roberto Molinaro. Estimates on the generalization error of physicsinformed neural networks for approximating pdes. IMA Journal of Numerical Analysis , 43(1): 1-43, 2023.
- [32] Olga Arsen'evna Oleinik. Estimates of the betti numbers of real algebraic hypersurfaces. Matematicheskii Sbornik , 70(3):635-640, 1951.
- [33] Ivan Georgievich Petrovskii and Olga Arsen'evna Oleinik. On the topology of real algebraic surfaces. Izvestiya Rossiiskoi Akademii Nauk. Seriya Matematicheskaya , 13(5):389-402, 1949.
- [34] David Pollard. Empirical processes: theory and applications. Ims, 1990.
- [35] Rahul Rai and Chandan K Sahu. Driven by data or derived through physics? a review of hybrid physics guided machine learning techniques with cyber-physical system (cps) focus. IEEe Access , 8:71050-71073, 2020.
- [36] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: Adeep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics , 378:686-707, 2019.

- [37] Robert Schaback and Holger Wendland. Kernel techniques: from machine learning to meshless methods. Acta numerica , 15:543-639, 2006.
- [38] Yeonjong Shin. On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type pdes. Communications in Computational Physics , 28(5):2042-2074, 2020.
- [39] Yeonjong Shin, Zhongqiang Zhang, and George Em Karniadakis. Error estimates of residual minimization using neural networks for linear pdes. Journal of Machine Learning for Modeling and Computing , 4(4), 2023.
- [40] Rui Wang, Karthik Kashinath, Mustafa Mustafa, Adrian Albert, and Rose Yu. Towards physicsinformed deep learning for turbulent flow prediction. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery &amp; data mining , pages 1457-1466, 2020.
- [41] Minglang Yin, Xiaoning Zheng, Jay D Humphrey, and George Em Karniadakis. Non-invasive inference of thrombus material properties with physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering , 375:113603, 2021.
- [42] Yifan Zhang and Joe Kileel. Covering number of real algebraic varieties and beyond: Improved bounds and applications. arXiv e-prints , pages arXiv-2311, 2023.

## A Notation

| Symbol                          | Description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| Data                            |                                                                           |
| f ∗                             | True function to be learned                                               |
| Ω ⊆ R m                         | Input domain                                                              |
| m                               | Input dimension                                                           |
| n                               | Number of observations                                                    |
| ( x i , y i )                   | i -th observation ( x i : input, y i : output)                            |
| y                               | Target vector [ y 1 , . . . , y n ] ⊤                                     |
| σ 2                             | Noise variance                                                            |
| ε i                             | Normally distributed noise following N (0 ,σ 2 )                          |
| Affine Variety and Variables    | Affine Variety and Variables                                              |
| D                               | Differential operator                                                     |
| B = { ϕ j } d j =1              | Basis functions                                                           |
| ϕ j                             | j-th basis function from B                                                |
| ϕ ( x ) ∈ R d                   | Basis vector at x                                                         |
| Φ ∈ R n × d                     | Design matrix (i-th row is ϕ ( x i ) ⊤ )                                  |
| T = { ( ψ k ,µ k ) } K k =1     | Finite collection of trial function and measure pairs.                    |
| d                               | Number of basis functions &#124;B&#124; (ambient dimension)               |
| K                               | Number of trial functions &#124;T &#124;                                  |
| V ( D , B , T )                 | Affine variety defined by D , B , T (set of weight vectors)               |
| d V                             | Dimension of the affine variety V                                         |
| w , ˆ w , w ∗                   | Weight vectors (learnable, estimated, optimal)                            |
| B 2 ( R )                       | ℓ 2 ball of radius R                                                      |
| V R                             | Affine variety constrained by the ℓ 2 -ball ( V ∩ B 2 ( R ) )             |
| λ n                             | L 2 regularization parameter                                              |
| ∥ · ∥ 2 , ∥ · ∥                 | Vector ℓ 2 norm, function L 2 norm w.r.t. Borel measure                   |
| ⟨· , ·⟩ 2 , ⟨· , ·⟩ µ           | Vector Euclidean inner product, function inner product w.r.t. measure µ   |
| Geometric / Complexity Measures | Geometric / Complexity Measures                                           |
| V ⊆ R d                         | General affine variety                                                    |
| ( β,d V ) -regular set          | Regularity condition                                                      |
| d V                             | Dimension of V                                                            |
| codim( L )                      | Codimension d - d L                                                       |
| β                               | Upper bound on connected components of intersections                      |
| N ( V, ε, ∥ · ∥ 2 )             | ε -covering number w.r.t. ℓ 2 norm ∥ · ∥ 2 .                              |
| Analysis Constants              | Analysis Constants                                                        |
| M                               | Upper bound constant for the basis functions, such that ∥ ϕ ( x ) ∥ 2 ≤ M |
| η                               | Upper bound constant for the lower eigenvalue of design matrix Φ .        |
| Γ                               | Stability constant of the estimator                                       |
| δ                               | Probability parameter                                                     |
| Linear Operators / PI Kernel    | Linear Operators / PI Kernel                                              |
| D ,D k,j                        | Constraint matrix when the operator D is linear; D k,j is its entry       |
| L , F                           | Linear, nonlinear parts of D                                              |
| d V ( L ) , d V ( D )           | Dimensions under L , D                                                    |
| κ M ( x, y )                    | PI kernel defined with regularization matrix M                            |
| ξ, ν                            | Hyperparameters of the PI kernel (controlling the balance)                |
| B , B j,j ′                     | Basis Gram matrix, its entry                                              |
| T , T k,k ′                     | Trial Gram matrix, its entry                                              |
| d eff ( ξ, ν )                  | Effective dimension of the PI kernel                                      |
| σ ( · ) , α,α j                 | Spectrum of a matrix and its entry (eigenvalues)                          |
| Dimension Calculation           | Dimension Calculation                                                     |
| p k                             | Defining polynomial                                                       |
| N                               | Number of samples ( w ∗ 1 , . . . , w ∗ N )                               |

Figure 3: Illustration of the construction of the ε -covering of the affine variety V ⊆ R 2 and the associated loss landscape. The black curve represents a ( β, d V ) regular affine variety with dimension d V = 1 . The color gradients depict the loss landscape L ( w ) := ∑ K k =1 ∥ p k ( w ) ∥ 2 2 of the equations defining V = { w : p k ( w ) = 0 , ∀ k = 1 , . . . , K } . The blue dotted line represents a ℓ 2 ball of radius R . The affine variety constrained with the ℓ 2 ball is covered by ε -balls centered at the intersections of V with four given subspaces { L s } 4 s =1 , shown as red points. The upper bound on the number of intersections of every subspace with the variety is β , while the actual maximum number is 5 formed by the subspace L ⋆ (the yellow dotted line). The loss landscape of the equations is zero on V and locally convex around the points in V .

<!-- image -->

## B Mathematical Background on Affine Varieties

In this section, we provide a formal definition of several concepts related to affine varieties and review the definition of the dimension of an affine variety, as briefly described in Section 4.

An affine variety is a fundamental concept in algebraic geometry. It is a subset of an affine space, defined as the solution set to a system of polynomial equations. Let K [ w ] denote the set of polynomials in the variables w = ( w 1 , . . . , w d ) ∈ K d over a field K (often R or C ). An affine variety V ( p 1 , . . . , p K ) ⊆ K d defined by the polynomials p 1 , . . . , p K ∈ K [ w ] is given by:

<!-- formula-not-decoded -->

The geometry of an affine variety is determined by the set of all polynomials that "vanish" on V , i.e., those that become zero for every point in V . This set is called the ideal of the affine variety, denoted I ( V ) , and is defined as follows:

<!-- formula-not-decoded -->

The generating polynomial set { p k } K k =1 of the affine variety V is a subset of the ideal I ( V ) .

The coordinate ring over V , denoted K [ V ] , is introduced to identify polynomials that yield the same values on the variety V . Specifically, K [ V ] is defined as the quotient of the polynomial ring K [ w ] by the ideal I ( V ) , i.e., K [ w ] /I ( V ) . In the coordinate ring K [ V ] = K [ w ] /I ( V ) , the difference between p and q vanishes on V , i.e., p ( w ) = q ( w ) for all w ∈ V , or equivalently p -q ∈ I ( V ) . Thus, p and q are considered the same element. From another viewpoint, the coordinate ring K [ V ] can be considered as a set of polynomials not included in the ideal I ( V ) .

Based on the above definitions, we review the definition of the dimension d V of the affine variety in Appendix B.1 and the regularity in Appendix B.2.

## B.1 Dimension of Affine Varieties

## B.1.1 Geometric View

Considering the affine variety V as an affine space, we can naturally define a subvariety as an "subset" of the variety that also satisfies polynomial equations. Let q 1 , . . . , q S be polynomials in a ring. Define ⟨ q 1 , . . . , q S ⟩ as the smallest ideal generated by q 1 , . . . , q S ; that is, ⟨ q 1 , . . . , q S ⟩ consists of all finite

sums of the form ∑ S i =1 r i q i where each r i is in the ring: ⟨ q 1 , . . . , q S ⟩ = { ∑ S i =1 r i q i } . A subvariety U of V is defined as the zero set of a subset ideal ⟨ q 1 , . . . , q S ⟩ ⊆ K [ w ] /I ( V ) given by:

<!-- formula-not-decoded -->

By using the concept of subvarieties, the dimension of an affine variety is defined as follows:

Definition 4.1. The maximal length of the chains V 0 ⊂ V 1 ⊂ . . . ⊂ V d V of non-empty subvarieties of V .

This definition intuitively represents the size of V by the maximal length of an increasing sequence of subspaces. If the generating polynomials { p k } K k =1 are all linear, the dimension of V is defined as the maximal length of an increasing sequence of linear subspaces within V , which corresponds to the dimension of V as a linear space.

When we focus on the local structure, the following equivalent definition is obtained:

Definition 4.3. The maximal dimension of the tangent vector spaces at the non-singular points U ⊆ V ⊂ R d of the variety, i.e ., d V = max w ∈ U d -rank [ ∇ p 1 ( w ) · · · ∇ p K ( w )] ⊤ .

From this definition, we can see that the dimension d V is a global quantity that summarizes the local linearized structure of the affine variety V at a point.

For example, let K = R and V ⊂ R 3 be the plane: V = { ( x, y, z ) : x + y -z = 0 } . A chain of subvarieties within V is V 0 ⊂ V 1 ⊂ V 2 , where V 0 = { (0 , 0 , 0) } (a point, 0 -dimensional), V 1 = { ( t, 0 , t ) : t ∈ R } (a line, 1 -dimensional), and V 2 = V itself (the plane, 2 -dimensional). The maximal length of the nested subvarieties is two, i.e ., d V = 2 , which means that a plane has two degrees of freedom.

## B.1.2 Algebraic View

The structure of an affine variety is determined by the ideal I ( V ) . Intuitively, the larger I ( V ) is, the more polynomial constraints there are, which means that V becomes smaller, and consequently, the coordinate ring K [ V ] also becomes smaller. From this perspective, it is natural to expect a deep connection between the dimension of the coordinate ring K [ V ] (and similarly the ideal I ( V ) ) and the dimension of the affine variety V .

To explore this connection, we first discuss the dimension of the coordinate ring K [ V ] using Krull dimension. The ideal p ⊂ R in a polynomial ring R is prime if ∀ a, b ∈ R , ab ∈ p ⇒ a ∈ p or b ∈ p . The definition of the dimension of the affine variety through the Krull dimension is shown below.

Definition B.1. The Krull dimension of the coordinate ring K [ V ] : The maximum length d of the chain of prime ideals p 0 ⊂ p 1 ⊂ · · · ⊂ p d in the coordinate ring K [ V ] .

This definition signifies that the dimension of an affine variety is characterized in the world of polynomial sets by the maximal length of an increasing chain of 'subsets' within the coordinate ring, corresponding to Definition 4.1 from a geometric perspective.

In contrast, the size of the coordinate ring K [ V ] can also be measured using Hilbert series. First, by homogenizing the defining equations by adding one variable γ ∈ K , we embed the affine variety V ⊂ K d into the projective variety P ⊂ K d +1 . The projective variety P ( h 1 , . . . , h K ) ⊂ K d +1 , defined by the homogeneous polynomials h 1 , . . . , h K ∈ K [( w , γ )] , is given by:

<!-- formula-not-decoded -->

The dimension of the variety is also increased by one, i.e., d P = d V + 1 . The coordinate ring K [ P ] = K [( w,γ )] /I ( P ) of the projective variety P can be decomposed into subgroups (called the graded coordinate ring) as follows:

<!-- formula-not-decoded -->

where S ρ is the set of homogeneous polynomials of degree ρ modulo the ideal I ( P ) . As a metric for the size of the coordinate ring K [ P ] , the Hilbert function H( ρ ) and Hilbert-Poincaré series HS( t ) are

defined as follows:

<!-- formula-not-decoded -->

where dim denotes the Krull dimension and ρ 1 , . . . , ρ K are the degrees of the homogeneous polynomials h 1 , . . . , h K .

The Hilbert function represents the dimension of a "subspace" of the decomposed coordinate ring, and the Hilbert series is the generating function of the sequence of the Hilbert function, which is also a rational function with a pole at t = 1 . These measures indicate the growth of the dimension of the homogeneous components of the algebra with respect to the degree. According to the dimension theorem, the Krull dimension of the projective variety P matches the order of the Hilbert series at the pole t = 1 , which is one of the most important results in commutative algebra.

Therefore, the dimension of the affine variety is defined using the Hilbert series, as follows:

Definition 4.2. The degree of the denominator of the Hilbert series of the affine variety V .

Given the Gröbner basis of the ideal I ( P ) , the Hilbert series can be easily computed, leading to an efficient estimation of the dimension of the affine variety d V .

## B.2 Regularity of Affine Varieties

We informally define the concept of a regular set for real affine varieties, which is used in Section 3.2 (for a formal definition, see Definition 2.1 in [42]).

A affine variety V ⊆ R d is a ( β, d V ) -regular set if:

1. For almost all affine planes L with codim( L ) ≤ d V in R d , V ∩ L has at most β pathconnected components.
2. For almost all affine planes L with codim( L ) &gt; d V in R d , V ∩ L is empty.

The notion codim represents the codimension . For an affine subspace L ⊆ R d , its codimension is defined by codim( L ) = d -d L . Simply put, codimension is how many dimensions you are 'missing' when comparing a smaller space inside a bigger space.

A regular set restricts the complexity of a variety V . Intuitively, the complexity of V can be measured by the number of connected components in its cross sections. For instance, a complex shape may have cross sections that split into multiple connected components. The larger the number of connected components β , the more complex the topology of V . Moreover, the dimension at which we slice the variety is also important. If the slice (affine plane) is large enough in dimension, i.e ., the codimension is small ( &lt; d V ), then any intersection of the slice with V is limited to at most β connected pieces. Otherwise, the slice typically does not intersect V at all. For example, consider the circle V = { ( x, y ) ∈ R 2 : x 2 + y 2 -1 = 0 } . A line ( codim( L ) = 1 ) intersects the circle in at most two points. For a single point ( codim( L ) = 2 ), almost all points do not lie in the circle; that is, intersections with higher codimension affine subspaces are almost empty. This implies that the circle is a (2 , 1) -regular set.

## B.3 Covering number of Affine Varieties

Lemma B.2 (Zhang and Kileel [42]) . Let V ⊂ R d be a ( β, d V ) -regular set in the ball B 2 ( R ) with the radius R . Then for all ε ∈ (0 , diam ( V )] ,

<!-- formula-not-decoded -->

This upper bound is obtained by slicing the affine variety V with subspaces { L s } s ∈ N within R d and covering V with balls centered at the intersections of L s and V , i.e ., V ⊂ ⋃ s ⋃ v ∈ V ∩ L s B 2 ( v ; ε ) .

The covering for the two-dimensional case is illustrated in Fig. 3. The first term, (2 Rd V d/ε ) d V , represents the number of subspaces L s needed to cover the entire space. It is mainly determined by the intrinsic dimension d V of the affine variety, although it is still influenced by the ambient

dimension d . The quantity β in the second term denotes the number of intersections between a single subspace L and the variety V , and represents the covering number of V ∩ L . Topologically, it corresponds to the Betti numbers of the affine variety, which informally represent the number of holes in V . The upper bound on the quantity β is given, for example, by the Petrovskii-Oleinik-Milnor inequality [33, 32, 28]. Specifically, an affine variety V ∩ B 2 ( R ) defined by polynomials { p k } k ∈ [ K ] of maximum degree ρ and the ℓ 2 -ball is ( ρ (2 ρ -1) d +1 , d V ) -regular. This intuitively suggests that as the maximum degree of polynomials increases, the topology of the affine variety becomes more complex.

## C Detailed Background on Problem Formulation

This appendix expands the formulation introduced in the main text, highlighting why a hybrid physics-data approach is required.

## C.1 Governing System

Let Ω ⊂ R d be a bounded domain with boundary ∂ Ω . For a differential operator D : L 2 (Ω) → L 2 (Ω) , the true state f ∗ : Ω → R satisfies the boundary-value problem

<!-- formula-not-decoded -->

where v and g are smooth but may be only partially observed or inferred indirectly.

## C.2 Available Information

In practice one seldom knows v and g exactly; instead one has:

- Noisy pointwise observations. A dataset { ( x i , y i ) } N u i =1 with y i = f ∗ ( x i ) + ε i , ε i ∼ N (0 , σ 2 ) , where x i ∈ Ω ∪ ∂ Ω .
- Weak-form physics information. Linear functionals l k ( u ) := ⟨ u, ψ k ⟩ µ k , k = 1 , . . . , N r , with trial functions ψ k ∈ C ∞ c (Ω) and measures µ k , together with the corresponding targets l k ( v ) = l k ( D [ f ∗ ]) .

## C.3 Hybrid Surrogate Model

A representative hybrid approach is the Physics-Informed Neural Network (PINN) [36]. Given a neural surrogate f w : Ω → R , its parameters w are obtained by minimising

<!-- formula-not-decoded -->

with hyper-parameter λ &gt; 0 balancing empirical fit and physical consistency.

Connection to standard collocation method In the standard collocation method, the unified residual forms reduce to pointwise strong-form residuals: specifically, for each k , we set

<!-- formula-not-decoded -->

where δ x k is the Dirac measure at collocation point x k . In this case, the linear functional l k becomes

<!-- formula-not-decoded -->

and thus the physics term penalizes the pointwise physics residuals:

<!-- formula-not-decoded -->

## Limiting regimes.

- Pure data fitting: λ = 0 reduces Eq. (18) to standard supervised learning on D u .
- Fully physics-informed: If { l k } is dense and N r →∞ , the residual term enforces Eq. (16) everywhere.
- Truly hybrid: Finite N r with incomplete { l k } -typical in engineering-captures partial physics, while the data term compensates for the missing information.

## C.4 Error decomposition and analysis

To directly measure how physics-based inductive bias improves the generalization ability of the machine learning models, we fix the physics information as a known immutable prior. Let H base denote the unrestricted hypothesis class (e.g. linear models or neural networks). We consider two ways of incorporating the physics prior into the base hypothesis class H base :

- Hard setting: Enforce the differential equation residual exactly by:

<!-- formula-not-decoded -->

- Soft setting: Allow a relaxed residual tolerance:

<!-- formula-not-decoded -->

For H ∈ {H hard , H soft ( ε ) } , let ˆ f ∈ H be the learned solution and f ∗ H := arg min f ∈H ∥ f ∗ -f ∥ the best attainable approximation. By the triangle inequality, we have:

<!-- formula-not-decoded -->

In this decomposition:

- The term ∥ f ∗ -f ∗ H ∥ represents the approximation error: the best achievable error within the hypothesis class H .
- The term ∥ f ∗ H -ˆ f ∥ represents the estimation error: the deviation due to finite data.

Our primary focus is to derive bounds on the estimation error ∥ f ∗ H -ˆ f ∥ , thereby quantifying how well the learned solution converges to the best physics-constrained approximation.

In particular, our analysis centers on the hard constraint setting, where H = H hard on the linear base hypothesis

<!-- formula-not-decoded -->

with B a chosen basis. Under the hard constraint, the admissible hypothesis class amounts to restricting the parameter vector w to lie on an affine variety induced by the PDE residuals. More explicitly,

<!-- formula-not-decoded -->

where V ( D , B , T ) denotes the set of coefficient vectors w satisfying the algebraic constraints generated by the operator D acting on the basis B and tested against the functionals T = { l k } N r k =1 .

## C.5 Extension: Incomplete Operators with Learnable Parameters

The above framework can be generalized to settings where the governing differential operator itself is only partially known and contains learnable parameters. Formally, suppose that instead of a fixed operator D : L 2 (Ω) → L 2 (Ω) , we consider a parametric operator

<!-- formula-not-decoded -->

where c denotes a vector of unknown coefficients to be simultaneously estimated from data. In this case, the admissible hypothesis space is naturally

<!-- formula-not-decoded -->

defined as an augmented constraint set over the joint variable ( f, c ) .

The error decomposition then applies in this extended space: for ( f ∗ , c ∗ ) denoting the best attainable pair in H aug, the learned solution ( ˆ f, ˆ c ) satisfies

<!-- formula-not-decoded -->

with both terms now understood relative to the augmented parameter space.

## Typical cases.

- Unknown diffusion coefficient. Consider the diffusion equation ∂ t u -c ∆ u = 0 , where the diffusion constant c &gt; 0 is unknown. Here c = ( c ) is a scalar parameter. Under the hard constraint, the admissible hypothesis class can be written as

<!-- formula-not-decoded -->

where V ( · ) denotes the algebraic variety of coefficient-parameter pairs ( w , c ) that satisfy the residual constraints induced by T = { l k } N r k =1 . Thus both approximation and estimation errors are quantified in this augmented parameter space.

- Unknown diffusion term (learned surrogate). In cases where the diffusion operator itself is not specified, one may introduce a surrogate v θ to represent its action. The PDE constraint becomes

<!-- formula-not-decoded -->

leading to the hypothesis class

<!-- formula-not-decoded -->

Here v θ serves as a learnable proxy for the unknown diffusion term. Our error decomposition applies verbatim in this augmented parameter space, with approximation error defined relative to the best attainable pair ( w ∗ , θ ∗ ) and estimation error measuring the deviation of the learned ( ˆ w , ˆ θ ) from this target.

In summary, by enlarging the hypothesis space to include both explicit unknown coefficients and implicit unknown operator surrogates, the proposed error decomposition continues to hold, thereby providing a principled means of quantifying generalization in operator-learning settings with incomplete physics.

## D Proof for Theorem 3.5

Theorem 3.5 (Minimax Risk Bound) . Let V ( D , B , T ) be the ( β, d V ) -regular affine variety defined in Eq. (2) . Suppose Assumptions 3.2-3.4 hold. Then, there exists a positive constant C , independent of n , d V , d , and β , such that for any δ ∈ (0 , 1) , with probability at least 1 -δ , the minimax risk for PILR defined by Eq. (4) is bounded by

<!-- formula-not-decoded -->

Proof. Step 1: We first upper bound the prediction error by a term that represents the supremum of a empirical process in the metric space of the affine variety. Using Lemma D.1, we get:

<!-- formula-not-decoded -->

We denote x w := ε ⊤ Φ ( w -ˆ w ) as the random process in the metric space ( V R , ∥ · ∥ 2 ) . Note that the estimator ˆ w is a random variable depending on the parameter w and the noise ε . Then, the minimax risk is bounded as follows.

<!-- formula-not-decoded -->

The first inequality holds by Assumption 3.3.

Step 2: Next, we calculate the supremum of the empirical process x w using the covering number. For all w 1 , w 2 ∈ V R , it is shown that the variable x w 1 -x w 2 has sub-Gaussian increments with respect to the metric ∥ · ∥ 2 :

<!-- formula-not-decoded -->

where e is the zero-mean Gaussian random variable with variance nσ 2 . The second inequality holds by the Cauchy-Schwarz inequality and the third holds by the triangle inequality and Assumption 3.2. The last inequality holds by Assumption 3.4.

From Eq. (23), the random process x w 1 -x w 2 has sub-Gaussian increments as follows.

<!-- formula-not-decoded -->

where Z is the standard Gaussian random variable and ∥ · ∥ ψ 2 is the sub-Gaussian norm. For the centered random process z w := x w -E [ x w ] , ∥ z w 1 -z w 2 ∥ ψ 2 ≲ ∥ x w 1 -x w 2 ∥ ψ 2 holds because ∥ x w 1 -x w 2 ∥ ψ 2 is sub-Gaussian.

Using Lemma D.2, we obtain the following bound with some constant C 0 :

<!-- formula-not-decoded -->

Next, using Dudley's integral tail bound, we have:

<!-- formula-not-decoded -->

By incorporating the non-centered process x w , we obtain:

<!-- formula-not-decoded -->

To bound E [ x w ] , we note that:

<!-- formula-not-decoded -->

Here, the third inequality follows from the Cauchy-Schwarz inequality, and the fourth inequality is derived from the fact that | ε ⊤ Φ j | 2 / ( σ ∥ Φ j ∥ 2 ) 2 follows a chi-squared distribution with 1 degrees of freedom and ˆ w ∈ V R .

By combining Eq. (24), Eq. (25), and Eq. (26), we obtain the following bound with some constant C :

<!-- formula-not-decoded -->

This completes the proof.

Lemma D.1. Let ˆ w be a minimizer of the following optimization problem:

<!-- formula-not-decoded -->

where V R = V ( D , B , T ) ∩ B 2 ( R ) is the affine variety constrained with the ℓ 2 -ball, y = Φ w ∗ + ε is the observed vector, Φ is the design matrix, w ∗ ∈ V R is the true parameter vector, and ε = [ ε 1 , . . . , ε n ] ⊤ is the noise vector with each ε i independently following a zero-mean Gaussian distribution. Then, under these conditions, we have:

<!-- formula-not-decoded -->

Proof. Since ˆ w is a minimizer of Eq. (27), we have:

<!-- formula-not-decoded -->

The left-hand side can be expanded as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Expanding the left-hand side, we get:

<!-- formula-not-decoded -->

Subtracting ∥ ε ∥ 2 2 from both sides, we obtain:

<!-- formula-not-decoded -->

Thus, we have:

This completes the proof.

Lemma D.2. Let z w be the zero-mean random process in the metric space ( V R , ∥ · ∥ 2 ) , which have the following sub-Gaussian increments. For all w 1 , w 2 ∈ V R ,

<!-- formula-not-decoded -->

where ∥ · ∥ ψ 2 is the sub-Gaussian norm, A is a positive constant. Then, the expectation of the supremum of the process can be bounded as follows.

<!-- formula-not-decoded -->

where C is positive constant.

Proof. Using Dudley's integral inequality [17] to the zero-mean random process:

<!-- formula-not-decoded -->

Since the set V R is ( β, d V ) regular set from Lemma 2.13 by Zhang and Kileel [42], Lemma B.2 shows the upper bound of the covering number for any ε ∈ (0 , 2 R ] as follows.

<!-- formula-not-decoded -->

We substitute the above inequality to Eq. (29):

<!-- formula-not-decoded -->

The integral in the first term can be calculated using substitution and integration by parts. Let

<!-- formula-not-decoded -->

We substitute χ := 2 Rd V d, u := log( χ/ε ) into the integral:

<!-- formula-not-decoded -->

To solve the above integral, we use the formula for integration by parts:

<!-- formula-not-decoded -->

The integral in the second term can be upper bounded as follows.

<!-- formula-not-decoded -->

We obtain the following bound with some constant C .

<!-- formula-not-decoded -->

## E Proof for Proposition 3.7 and Proposition 4.4

Proposition 3.7. The effective dimension of the PI kernel associated with the affine variety V ( D , B , T ) = { w : Dw = 0 } with dimension d V is upper bounded by

<!-- formula-not-decoded -->

where { α j } d j = d V denote the positive eigenvalues of the matrix D ⊤ TD .

Proof. From Theorem 4.2 in [15] and Equation 15 in [16], the effective dimension is bounded as follows:

<!-- formula-not-decoded -->

where M := ξ I + ν D ⊤ TD ∈ R d × d and B ∈ R d × d is the Gram matrix of the basis functions, i.e., B j,j ′ = ⟨ ϕ j , ϕ j ′ ⟩ µ for all ϕ j , ϕ j ′ ∈ B .

Since the matrix D ⊤ TD is positive semi-definite, the eigenvalues of the matrix M in ascending order σ j ( · ) are given by

<!-- formula-not-decoded -->

Therefore, the matrix M is positive definite, and the eigenvalues of M -1 are α -1 for all α ∈ σ ( M ) . Combining this with Eq. (30), we obtain the first inequality. The second inequality is obtained when ν = 0 .

Proposition 4.4. Suppose the operator D can be decomposed as D = L + F , where L is a nonzero linear differential operator and F is a nonlinear operator. Then, we have d V ( L ) ≤ d V ( D ) .

̸

Proof. The point w = 0 lies on V ( D ) , and if L = 0 , it is not singular. The Jacobian rank of polynomials p k ( w ) = ⟨ D [ w ⊤ ϕ ] , ψ k ⟩ µ k in w = 0 is equal to d -d V ( L ) . By Definition 4.3, we have d V ( L ) ≤ d V ( D ) .

## F Minimax Risk Analysis for Physics-Informed Models with General Architectures via Rademacher Complexity

In this section, we extend our analysis to general model architectures parameterized by polynomial functions of the weights. Our primary objective is to establish a minimax risk framework grounded in Rademacher complexity. This allows us to handle richer hypothesis spaces while incorporating structural constraints imposed by physical laws.

## F.1 Notation and Definitions

To set the stage, we introduce several fundamental notions that will be used throughout the complexity analysis. We begin with norms for vector- and matrix-valued objects, which help measure the size and regularity of functions and parameters. These norms provide the foundation for bounding Rademacher complexity.

Definition F.1 (Mixed Norm for Vector-Valued Functions) . Let f : X → R d out be a vector-valued function. Its ( ∞ , p ) -norm is defined by

<!-- formula-not-decoded -->

The above norm enables us to uniformly control the p -norm magnitude of the function outputs across the entire input domain.

Definition F.2 (Matrix p -Norm) . For a matrix W ∈ R d in × d out , we regard W as a vector vec( W ) ∈ R d in d out . Its norm is defined using the standard vector p -norm:

<!-- formula-not-decoded -->

This definition allows us to consistently measure parameter magnitudes, regardless of whether they appear as vectors or matrices.

Definition F.3 (Rademacher Complexity) . Let F be a function class and S = { x 1 , . . . , x n } an i.i.d. sample. The empirical Rademacher complexity is defined as

<!-- formula-not-decoded -->

where τ 1 , . . . , τ n are independent Rademacher variables taking values in {± 1 } . The Rademacher complexity is obtained by further taking the expectation of ̂ R S ( F ) over the random sample S .

Rademacher complexity serves as a central tool for quantifying the richness of hypothesis spaces and will be essential in deriving minimax risk bounds.

## F.2 Definition of the Hypothesis Space

We now define the hypothesis space of interest. To ensure well-posedness of our analysis, the class is required to satisfy boundedness and Lipschitz continuity conditions.

Definition F.4 (Lipschitz Polynomial Hypothesis Space) . Let H denote a hypothesis space consisting of functions f w : R m → R , parameterized by w ∈ R d , where each f w is polynomial in w .

The parameter domain is restricted to

<!-- formula-not-decoded -->

where the affine variety V ( D , T ) is given by

<!-- formula-not-decoded -->

Here, B 2 ( R ) := { w ∈ R d : ∥ w ∥ 2 ≤ R } denotes the Euclidean ball of radius R .

Furthermore, there exists a constant ℓ H &gt; 0 such that

<!-- formula-not-decoded -->

ensuring Lipschitz continuity of the parameter-to-function mapping.

The above construction provides a general framework. Next, we highlight an important special case relevant to physics-informed neural networks (PINNs).

## Special Case: Polynomial PINN

Definition F.5 (Polynomial PINN Hypothesis Space) . Let H L denote the hypothesis space represented by a fully-connected neural network of depth L with polynomial activation ϕ :

<!-- formula-not-decoded -->

with parameter vector w = vec( W 1 , . . . , W L ) ∈ R d . The parameter domain is restricted to w ∈ V R = V ( D , T ) ∩ B 2 ( R ) , where V ( D , T ) is the affine variety in equation 33.

To ensure the polynomial PINN setting remains mathematically well-posed, we introduce additional assumptions on boundedness and Lipschitz continuity.

Assumption F.6 (Uniformly Bounded Target and Hypothesis Class) . The true regression function f ∗ : R m → R is uniformly bounded as ∥ f ∗ ∥ ∞ ≤ F max . Moreover, every hypothesis f w ∈ H satisfies the same bound: ∥ f w ∥ ∞ ≤ F max .

Assumption F.7 (Lipschitz Continuity and Boundedness of Polynomial Activation) . The polynomial activation function ϕ is uniformly bounded, ∥ ϕ ∥ ∞ , 2 ≤ M ϕ for some constant M ϕ &gt; 0 . Moreover, ϕ is Lipschitz continuous with constant L ϕ , i.e., for any z 1 , z 2 ∈ R :

<!-- formula-not-decoded -->

Finally, we show that under the above assumptions, polynomial PINNs inherit a Lipschitz property at the function level.

Lemma F.8 (Lipschitz Property of Polynomial PINNs) . Suppose Assumption F.7 holds. For two polynomial PINNs f w , f w ′ ∈ H L with parameters w , w ′ ∈ V R , the mapping from parameters to functions is Lipschitz continuous, i.e.,

<!-- formula-not-decoded -->

where the Lipschitz constant ℓ H L depends on the network architecture as

<!-- formula-not-decoded -->

Consequently, the polynomial PINN hypothesis space H L satisfies the Lipschitz condition in Definition F .4, and therefore

<!-- formula-not-decoded -->

Proof. Let h ℓ and h ′ ℓ be the outputs of layer ℓ for parameters w and w ′ respectively. We can establish a recursive inequality:

<!-- formula-not-decoded -->

Solving this recurrence relation for ∥ h L -h ′ L ∥ ∞ = ∥ f w -f w ′ ∥ ∞ yields the constant ℓ L . The relationship between covering numbers follows directly.

## F.3 Generalization Bound

Based on the assumptions, we can now control the Rademacher complexity of the Lipschitz polynomial hypothesis class H through a Dudley integral bound.

LemmaF.9 (Dudley Integral Bound for Physics-Informed Models) . Let H be the hypothesis space defined in Definition F .4, where the underlying affine variety V is ( β, d V ) -regular. Then the Rademacher complexity of H is bounded as

<!-- formula-not-decoded -->

for some constant C &gt; 0 .

Proof. First, for any w 1 , w 2 ∈ V R and corresponding f w 1 , f w 2 ∈ H , Lipsitz continuity reads

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Dudley's integral bound and Lemma B.2 on ℓ p -covers,

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

Similar calculation in Lemma D.2 shows

<!-- formula-not-decoded -->

Combining these estimates yields the stated bound.

Lemma F.10 (Maximum of sub-exponential random variables) . Let X 1 , . . . , X n be independent, identically distributed sub-exponential random variables satisfying ∥ X i ∥ ψ 1 ≤ ν for every i ∈ { 1 , . . . , n } . Then there exists an absolute constant C &gt; 0 such that

<!-- formula-not-decoded -->

Proof. From the assumption ∥ X i ∥ ψ 1 ≤ ν , we have the standard sub-exponential tail bound:

<!-- formula-not-decoded -->

Moreover, for all t ≥ 0 , we can uniformly bound the maximum via a union bound:

<!-- formula-not-decoded -->

Now consider two cases based on the relation between n and the exponent.

Hence

Case 1: Suppose

<!-- formula-not-decoded -->

Then, by inequality equation 38, we have

<!-- formula-not-decoded -->

which already provides a stronger tail decay than we ultimately require.

Case 2: Suppose

<!-- formula-not-decoded -->

Then, the probability on the right-hand side of Equation (38) satisfies

<!-- formula-not-decoded -->

which is vacuously bounded above by 1. Thus, it does not affect the validity of our bound.

In either case, we obtain the uniform upper bound

<!-- formula-not-decoded -->

This tail bound implies that for some universal constant C &gt; 0 .

Lemma F.11. Let g w ( x, y ) := ( f w ( x ) -y ) 2 -E ( X,Y ) [ ( f w ( X ) -Y ) 2 ] for f w ∈ H . Under Assumption F.6 assume the noise ϵ = Y -f ∗ ( X ) is sub-Gaussian with proxy variance σ 2 . Then there exists a constant C &gt; 0 (independent of w ) such that

<!-- formula-not-decoded -->

Proof. By the triangle inequality for the sub-Gaussian norm,

<!-- formula-not-decoded -->

Using the identity ∥ Z 2 ∥ ψ 1 = ∥ Z ∥ 2 ψ 2 for any Z , we get

<!-- formula-not-decoded -->

By the triangle inequality in the ψ 1 -norm,

<!-- formula-not-decoded -->

Finally, to bound the second moment,

<!-- formula-not-decoded -->

for suitable constant C &gt; 0 . This completes the proof.

<!-- formula-not-decoded -->

Lemma F.12. Let ℓ ( u, y ) = ( u -y ) 2 . The Rademacher complexity of the composite class ℓ ◦ H = { ( f w ( x ) -y ) 2 : f w ∈ H} satisfies

<!-- formula-not-decoded -->

where F max is a uniform bound on | f w ( x ) | and σ 2 is the noise variance.

Proof. Define for each example ( x i , y i ) the function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since | f w ( x ) | ≤ F max for all x and | y i | ≤ F max + | ε i | , it follows that

<!-- formula-not-decoded -->

Applying the Rademacher contraction lemma to the empirical Rademacher complexity ̂ R S ,

<!-- formula-not-decoded -->

Since E ε [ | ε | ] ≤ σ √ 2 /π for Gaussian noise,

<!-- formula-not-decoded -->

For any u, v ∈ R ,

This completes the proof.

Finally, combining the Rademacher complexity bound with Adamczak's concentration inequality [2] yields the following generalization bound for general physics-informed architectures.

Theorem F.13 (Generalization Bound for Physics-Informed Models) . Let f w ∈ H be the Lipschitz polynomial hypothesis defined by Definition F.4, and assume that Assumption F.6 holds. Then there exist constants C 0 , C 1 , C 2 &gt; 0 such that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. Apply Adamczak's concentration inequality [2] to the supremum:

<!-- formula-not-decoded -->

where C is constant, V and ν are quantities defined by

<!-- formula-not-decoded -->

From Lemmas F.10 and F.11, we have

<!-- formula-not-decoded -->

where C 1 , C 2 are constants.

By substituting the above quantities into inequality Equation (41) and converting it into a highprobability bound, we obtain, with probability at least 1 -δ , the following result:

<!-- formula-not-decoded -->

We bound the expectation via symmetrization and Rademacher complexity:

<!-- formula-not-decoded -->

where τ i are Rademacher variables and R ( ℓ ◦ H ) denotes the Rademacher complexity of the squared loss class.

Combining Lemmas F.9 and F.12 with Equations (42) and (43), these bounds complete the proof.

Remark F.14 (Limitations of Theorem F.13) . While Theorem F.13 provides a theoretical generalization bound for polynomial PINNs with neural network surrogates, two caveats are worth emphasizing. First, the affine variety induced by the hard constraints is defined by a system of high-degree polynomial equations. The resulting algebraic structure is computationally intractable to characterize explicitly, which limits both theoretical validation and practical implementation. Second, recent findings suggest that the generalization behavior of over-parameterized models such as neural networks is not governed solely by the geometry of the hypothesis space, but is strongly affected by the implicit bias of the optimization algorithm. Hence, bounds of the form in Theorem F.13 may substantially deviate from the empirically observed performance.

## G Experimental Detail

## G.1 Experiments on Strong Solution

In the experiments in Section 5.1, strong solutions to the equations are obtained analytically. The analytical solution with added Gaussian noise was used as data, the variance of the Gaussian noise was set to 0 . 01 . The hyperparameters L 2 regularization weights and differential equation constraint weights ξ and ν were searched in the range [1 e9 , 1 e2] using the Optuna library [4]. The configuration with the smallest MSE on the validation data among 100 candidates was selected. All experiments were conducted on a MacBook Air equipped with an Apple M3 chip and 64 GB of unified memory. No external GPU or cluster computing resources were used.

Harmonic Oscillator: The initial value problem of a harmonic oscillator D [ y ] = 0 with spring constant k s and mass m s on the domain Ω = [0 , T ] is given by:

<!-- formula-not-decoded -->

We set the parameters m s = k s = 1 . 0 , T = 2 π . The initial position and velocity [ y 0 , v 0 ] ⊤ are generated from the normal distribution N ( 1 , I ) , where 1 is an all-ones vector and I is the identity matrix. The solution to the initial value problem is analytically given by:

<!-- formula-not-decoded -->

The settings for the basis functions and the trial functions with the measure ϕ j ∈ B , ( ψ k , µ k ) ∈ T are as follows:

<!-- formula-not-decoded -->

where d t ∈ { 2 , 4 , 8 , 16 } is the set of the number of basis functions, and x k ∈ Ω is uniformly sampled from data with K = 100 .

Diffusion Equation: The initial value problem for the one-dimensional diffusion equation D [ u ] = 0 with diffusion coefficient c and periodic boundary conditions is given by:

<!-- formula-not-decoded -->

We set the parameters c = 1 . 0 , Ξ = π, T = 2 π . The initial value u 0 is given by:

<!-- formula-not-decoded -->

where [ A j , B j ] ⊤ are generated from the normal distribution N ( 1 , I ) for all j = 0 , . . . , j max and j max is set to 1 . The solution to the initial value problem is analytically given by:

<!-- formula-not-decoded -->

The settings for the basis functions and the trial functions with the measure ϕ j ∈ B , ( ψ k , µ k ) ∈ T are as follows:

<!-- formula-not-decoded -->

where d t = 2 , d x ∈ { 10 , 15 , 20 , 25 } are the sets of the number of basis functions, and ( x k , t k ) ∈ Ω is uniformly sampled from data with K = 50 × 500 .

## G.2 Experiments on Numerical Solution

In the experiments in Section 5.3, we numerically simulate the Bernoulli equation using the explicit Euler method and the diffusion equation using the finite difference method (FDM). The data used are the numerical solutions with added Gaussian noise of variance 0.01. The method for hyperparameter search is the same as described in Appendix G.1. For the nonlinear equations, we use the Adam optimizer with a learning rate of 1 × 10 -2 , along with an exponential learning rate scheduler. The training is performed for a maximum of 2000 epochs, utilizing an early stopping technique.

Discrete Bernoulli Equation: The discrete Bernoulli equation D h [ y ] = 0 with the step size h on the domain Ω = [0 , T ] is given by:

<!-- formula-not-decoded -->

where y τ = y ( t τ ) and y τ +1 = y ( t τ + h ) are evaluations on the grid { t τ } n t τ =1 with n t = T h . We set the constant parameters ( P, Q, ρ ) to (1 . 0 , 0 . 0 , 0 . 0) for the linear case and to (1 . 0 , 0 . 5 , 2 . 0) for the non-linear case. We use varying n t ∈ { 100 , 200 } with T = 1 . 0 for both cases. The initial state y 0 is generated from the standard normal distribution N (0 , 1) for both cases. The ground-truth solution to the initial value problem is numerically solved by the explicit Euler method with step size h . The settings for the basis functions and the trial functions with measure ϕ τ ∈ B h , ( ψ τ , µ τ ) ∈ T h are as follows:

<!-- formula-not-decoded -->

where n t = T h is the same as the number of basis and trial functions, corresponding to the ground-truth solutions.

Discrete Diffusion Equation: The one-dimensional discrete diffusion equation D h [ u ] = 0 with the step size h = [ h t , h x ] ⊤ and the diffusion coefficient c ( u ) on the domain Ω = [ -Ξ , Ξ] × [0 , T ] is given by:

<!-- formula-not-decoded -->

where u τ j := u ( x j , t τ ) , u τ +1 j := u ( x j , t τ + h t ) , and u τ j ± 1 := u ( x j ± h x , t τ ) are evaluations on the n x × n t size grid { x j } n x j =1 ×{ t τ } n t τ =1 , where n x := 2Ξ h x and n t := T h t . The periodic boundary condition is adopted in the spatial domain, i.e ., u τ n x + j = u τ j for any j ∈ [ d ] . The diffusion coefficient c ( u ) = 1 . 0 is used for the linear case and c ( u ) = 0 . 1 / (1 + u 2 ) for the nonlinear case. We use varying ( n t , n x ) ∈ { (400 , 10) , (400 , 20) , (400 , 30) } with Ξ = 1 . 0 and T = 1 . 0 for both cases. The initial value is generated with the same setting as shown in Eq. (44). The ground-truth solution to the initial value problem is numerically solved by the FDM with step sizes h t for the time domain and h x for the spatial domain. The settings for the basis functions and the trial functions with measure ϕ j,τ ∈ B h , ( ψ j,τ , µ j,τ ) ∈ T h are as follows:

<!-- formula-not-decoded -->

where n x = 2Ξ h x and n t = T h t are the same as the number of basis and trial functions, corresponding to the ground-truth solutions.

## H Additional Experimental Results

For each benchmark (discrete linear/nonlinear Bernoulli and Heat equations), we fix the number of basis functions d and vary the size of the trial-function set T . Reducing |T | relaxes the algebraic constraints on the learned solution, which in turn increases the dimension of the associated affine variety d V . As reported in Tables Figs. 4 and 5, when |T | decreases (and thus d V increases), the Test MSE steadily increases. This consistent rising trend of Test MSE with larger d V demonstrates that models endowed with fewer trial functions (i.e. weaker constraints) generalize more poorly.

## (a) Linear Bernoulli eq.

| Settings        | h     | 1 /                | 100              |                 |
|-----------------|-------|--------------------|------------------|-----------------|
| Dimensions      | d d V | 10                 | 100 20           | 40              |
| Test MSE (PILR) |       | 0 . 012 ± 0 . 0023 | 0 . 13 ± 0 . 082 | 0 . 33 ± 0 . 22 |

## (b) Nonlinear Bernoulli eq.

Figure 4: Experimental results for PILR on the discrete Bernoulli equations.

| Settings        | h     | 1 /             | 100             |                 |
|-----------------|-------|-----------------|-----------------|-----------------|
| Dimensions      | d d V | 10              | 100 20          | 40              |
| Test MSE (PILR) |       | 0 . 17 ± 0 . 11 | 0 . 21 ± 0 . 14 | 0 . 33 ± 0 . 23 |

(a) Linear Heat eq.

| Settings        | ( h t ,h x )   | (1 / 400 , 2 /   | 10)            |                |
|-----------------|----------------|------------------|----------------|----------------|
| Dimensions      | d d V          | 110              | 4010 210       | 410            |
| Test MSE (PILR) |                | 1 . 6 ± 0 . 35   | 1 . 9 ± 0 . 44 | 2 . 0 ± 0 . 49 |

## (b) Nonlinear Heat eq.

Figure 5: Experimental results for PILR on the discrete Heat equations.

| Settings        | ( h t ,h x )   | (1 / 200 , 2 /   | 10)             |                 |
|-----------------|----------------|------------------|-----------------|-----------------|
| Dimensions      | d d V          | 110              | 2010 210        | 410             |
| Test MSE (PILR) |                | 0 . 37 ± 0 . 11  | 0 . 43 ± 0 . 14 | 0 . 56 ± 0 . 19 |

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a

proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the main contributions, including the unification of collocation and variational methods via a unified residual form, the establishment that generalization is determined by the affine variety dimension, and the method to approximate this dimension, all of which are addressed in the paper (Sections 3, 4, and 5).

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are discussed in Section 6 (Conclusion).

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

Justification: Assumptions for theoretical results like Theorem 3.5 and Proposition 3.7 are stated. Proofs are provided in the appendices (e.g., Appendix D for Theorem 3.5, Appendix E for Proposition 3.7), with proof sketches or main ideas often presented in the main text.

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

Justification: Section 5 describes the experimental setup, and Appendix G provides further details on the experimental setup, including the equations, domain, boundary conditions, and network architectures used, which should allow for reproduction of the main findings.

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

Justification: We plan to release the full codebase as part of the supplementary material. The repository will include scripts and instructions to reproduce all main experiments. Since all data used in the experiments is synthetically generated, the released code also includes utilities to generate this data, ensuring full reproducibility without reliance on external datasets.

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

Justification: Section 5 and particularly Appendix G describe experimental settings, including PDEs, network architectures, number of data points, and training points. Details like specific optimizer parameters (e.g., learning rate) are present in Appendix G, which aims to provide details for reproducibility.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Experimental results are reported with means and standard deviations computed over multiple runs (10 random seeds per setting), as described in Section 5 and detailed in Appendix G. Error bars shown in the plots represent standard deviations. The randomness arises from the sampling of initial/boundary conditions and optimization initialization, which are fixed across methods for fair comparison.

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

Justification: Details on computational resources are provided in Appendix G.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research presented is theoretical and methodological, focusing on mathematical understanding of PIML, and does not appear to raise concerns conflicting with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper focuses on foundational theoretical aspects of physics-informed machine learning and does not explicitly discuss potential positive or negative societal impacts of this specific theoretical advancement. A broader impacts statement is not included.

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

Justification: The work is theoretical, focusing on understanding generalization in PIML. It does not introduce new models or datasets that pose a high risk for misuse requiring specific safeguards.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper primarily builds upon established mathematical concepts (e.g., differential equations, affine varieties) and uses synthetically generated data for experiments (Appendix G), not relying on external datasets or codebases that would require explicit licensing details.

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

Justification: The codebase implementing our proposed method (PILR) will be released as part of the supplementary material. It includes configuration files, training scripts, and utility functions for solving representative PDEs. A README file documents how to run each experiment, and the code is released to encourage reuse and extension by the community.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The research does not involve crowdsourcing or experiments with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The research does not involve human subjects, so IRB approval is not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core methodology of this research focuses on physics-informed machine learning theory and does not involve the use of LLMs as an important, original, or nonstandard component.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
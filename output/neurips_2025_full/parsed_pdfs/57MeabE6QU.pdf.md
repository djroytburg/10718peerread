## Distributionally Robust Performative Optimization

Zhuangzhuang Jia 1 , Yijie Wang 2 , Roy Dong 1 , Grani A. Hanasusanto 1

1 Department of Industrial and Enterprise Systems Engineering University of Illinois Urbana-Champaign 2 School of Economics and Management, Tongji University

{zj12,roydong,gah}@illinois.edu, yijiewang@tongji.edu.cn

## Abstract

In performative stochastic optimization, decisions can influence the distribution of random parameters, rendering the data-generating process itself decision-dependent. In practice, decision-makers rarely have access to the true distribution map and must instead rely on imperfect surrogate models, which can lead to severely suboptimal solutions under misspecification. Data scarcity or costly collection further exacerbates these challenges in real-world settings. To address these challenges, we propose a distributionally robust framework for performative optimization that explicitly accounts for ambiguity in the decision-dependent distribution. Our framework introduces three modeling paradigms that capture a broad range of applications in machine learning and decision-making under uncertainty. This latter setting has not previously been explored in the performative optimization literature. To tackle the intractability of the resulting nonconvex objectives, we develop an iterative algorithm named repeated robust risk minimization, which alternates between solving a decision-independent distributionally robust optimization problem and updating the ambiguity set based on the previous decision. This decoupling ensures computational tractability at each iteration while enhancing robustness to model uncertainty. We provide reformulations compatible with off-the-shelf solvers and establish theoretical guarantees on convergence and suboptimality. Extensive numerical experiments in strategic classification, revenue management, and portfolio optimization demonstrate significant performance gains over state-of-the-art baselines, highlighting the practical value of our approach.

## 1 Introduction

Decision-makers' actions often have a ripple effect on the external environment, which can lead to changes in the distribution of uncertain parameters. For instance, in the realm of portfolio management, institutional investors' decisions can have a profound effect on stock prices. This is partly due to their substantial capital, which can propel stock prices to rise (or fall) when they are buying (or selling) [57], and partly because their actions shape market sentiment towards those particular stocks [2]. Similarly, in revenue management, airlines often make forecasts and design pricing strategies based on the historical patterns of passenger behavior. However, passengers are not static; they often modify their behaviors in reaction to the new pricing strategies of airlines, which in turn shifts the overall pattern of consumer behavior [11].

When the solution of a stochastic optimization problem affects the distribution of the uncertain parameters, we call such a problem performative [40, 15]. The primary goal of the decision maker within such a dynamic environment is to find a decision that minimizes the expected risk after the environment has reacted to its deployment. A natural way of introducing this decision-dependent uncertainty is to construct a distributional map from the set of decisions to the space of distributions.

However, the true underlying map is typically unknown in reality. Consequently, practitioners usually rely on a nominal or reference distribution map constructed from historical observations or domain expertise. While methods grounded in the reference distribution map might perform satisfactorily on the observed samples, they often fail to achieve acceptable performance in out-ofsample circumstances.

This paper aims to tackle this core deficiency by leveraging the ideas of Distributionally Robust Optimization . Unlike traditional approaches that assume a single distribution map, distributionally robust performative optimization (DRPO) adopts a more flexible strategy: it establishes an ambiguity set of plausible distributions centered at the reference distribution. Next, the objective of the decision maker is to derive an optimal decision that minimizes the worst-case expected risk, where the worst case is taken over all distribution maps from within this ambiguity set. By optimizing against this adversarial perspective within a neighborhood of the reference map, it mitigates the overfitting issue and improves the out-of-sample performance.

The difficulty in solving DRPO stems from the dependence of the distribution map on the decision, which prohibits the direct use of existing solution schemes from the robust and distributionally robust optimization literature. For instance, even under the simplest setting where the loss function is linear and the distribution map is linear in decisions, the resultant problem is a nonconvex bilinear program. We address this challenge by designing a repeated robust risk minimization algorithm. Specifically, the decision maker repeatedly obtains an optimal decision that minimizes DRPR risk using the reference distribution from the previous iteration. Finally, we conduct a theoretical analysis of the algorithm, providing convergence and sub-optimality guarantees. Our main contributions can be summarized as follows:

1. General DRPO Framework : We propose a distributionally robust framework based on Wasserstein distance for performative optimization. This approach enables safe decision-making when we lack full information about the underlying decision-dependent distributions and only have access to some reference distributions. It encompasses a wide class of loss functions and can be applied to numerous machine learning and decision-making problems. As a byproduct of our reformulations, we identify a relatively general setting where the distributionally robust model is equivalent to Tikhonov regularization.
2. Repeated Robust Risk Minimization Algorithm : Wedevelop a repeated robust risk minimization algorithm for the problem that effectively mitigates the intractability of decision-dependent uncertainty. Our approach decouples the decisions associated with the ambiguity set from the expected loss, optimizing the latter while fixing the former to the previous decision in each iteration. This transforms the challenging DRPO problem into a sequence of tractable conic programs, rendering the framework computationally feasible and amenable to solutions via off-the-shelf solvers.
3. Rigorous Theoretical Guarantees: We provide convergence results of the repeated robust risk minimization algorithm to the stable solutions for our proposed models. To our knowledge, the convergence of such an algorithm has not previously been established in the distributionally robust settings. Our results show that, in general, the distributionally robust models lead to faster convergence than non-robust schemes. We further prove that our stable solutions are near to the robust performatively optimal ones.

## 1.1 Related Work

Stochastic Optimization: Our study closely relates to a body of research delving into endogenous uncertainty in stochastic optimization. Early examples include production planning problems with decision-dependent production costs [24] and oil planning problem with decision-dependent information discovery [18]. Their solution entails constructing a scenario-tree-based stochastic programming model and implementing a decomposition algorithm for resolution. Building on this, a subsequent study [19] extends the methodology to the multi-stage setting, focusing on production scheduling that minimizes costs while fulfilling diverse product demands. More broadly, [54] explore general multistage stochastic optimization problems beset with endogenous uncertainty, devising a conservative solution framework by leveraging piecewise linear decision rule approximations.

Robust Optimization: In robust optimization, decision-dependent uncertainty is usually imposed directly on the uncertainty set. Specific applications include customized endogenous uncertainty

sets for software partitioning [50] and robust scheduling, where the uncertainty set is constructed as a decision-dependent combination of simpler sets [56]. Decision-dependent sets have also been designed for primitive uncertainties in control systems [63]. The complexity analysis of the robust selection problem with decision-dependent information discovery is studied by [33], where they present polynomial complexity results for two special cases. For general settings, [37] show the problem is NP-complete and demonstrate the benefit of using endogenous uncertainty sets via a shortest-path problem. Algorithmic approaches have been developed, including exact nested decomposition schemes for two-stage problems [38] and approximation methods based on decision rules for the multistage setting [62].

Distributionally Robust Optimization: Our research expands upon the conventional framework of distributionally robust optimization by incorporating the influence of decision-making on the probability distribution. However, due to the problem difficulty, decision-dependent ambiguity sets generally lead to intractable reformulations, which could be computationally intensive for large instances. [61] study a broad class of distributionally robust optimization problems with decisiondependent moment ambiguity sets and conduct stability analysis. For the two stage setting, [31] analyze a wide range of decision-dependent ambiguity sets and establish non-convex semi-infinite reformulations and [23] explore decision-dependent information discovery using the K-adaptability approximation scheme. In multistage settings, [59] focus on moment-based ambiguity sets and derive a mixed-integer semidefinite programming reformulation. This decision-dependent DRO framework has been applied to a wide range of operations management problems, including nurse staffing [47], facility location [5], and retrofitting planning [14]. In contrast to the existing approaches that require an explicit specification of the ambiguity set, our scheme relies only on reference distributions at finitely many decisions.

Performative Supervised Learning: Our study also belongs to an emerging class of research on performative supervised learning, where algorithmic predictions can actively mold the surrounding environment and alter the underlying distributions of uncertain parameters [15, 20, 34, 40, 32]. The origins of performative learning can be traced back to studies on supervised learning under distribution drifts [3, 4, 16]. The foundational framework of performative prediction was introduced by [40], who designed a retraining algorithm and analyze the convergence behaviors to stochastic performatively stable points. Unfortunately, the convergence results do not extend to the robust settings, as taking worst-case expectations introduces non-smoothness into the objective function. Gradient-based methods have been developed in [40, 32, 15] for non-robust formulations, but they are inapplicable in robust settings due to the intractability of computing gradients of worst-case expectations. [28] has extended the scope of performative prediction to decision-making via performative omniprediction. We refer readers who are interested in performative learning to a comprehensive review [21].

Finally, we would like to highlight the difference between our paper and two related works on distributionally robust performative prediction. [39] propose a distributionally robust performative model to promote machine learning fairness. However, their robust smoothness assumption on the objective and robust sensitivity assumption on the worst-case distributions are unrealistic. Additionally, unlike the Wasserstein ambiguity set adopted in our paper, their phi-divergence ambiguity set precludes any continuous distribution, so it cannot ensure true probability distribution coverage guarantee. [58] study distributionally robust performative prediction based on a KL-divergence ambiguity set. Similar to [39], such an ambiguity set may fail to provide sufficient protections against distributions whose scenarios do not coincide with those in the reference distribution. In the paper, the authors develop an alternating minimization (coordinate descent) algorithm to solve the distributionally robust model. Unfortunately, such an algorithm may fail to converge, and the paper also does not provide any convergence guarantee. The algorithm also assumes the existence of a solver that can solve performative risk-sensitive minimization problems; however, to our knowledge, there is no existing literature that studies such problems and provides convergent algorithms.

## 1.2 Notation and Terminology

We use R + and R ++ to denote the sets of nonnegative and strictly positive real numbers, respectively. The identity matrix is denoted by I . For any n ∈ N , we define [ n ] as the index set { 1 , . . . , n } . The Dirac measure concentrating unit mass at ξ ∈ Ξ is denoted by δ ξ . For any real-valued matrix A , its Schattenq norm is defined as ∥ A ∥ q = (tr( A ⊤ A ) q/ 2 ) q . Random variables are designated with tilde signs (e.g., ˜ z ), while their realizations are denoted by the same symbols without tildes (e.g., z ).

## 2 Preliminaries

In stochastic optimization, the goal of the decision maker is to find a decision θ ∈ Θ ⊆ R d that minimizes the risk

<!-- formula-not-decoded -->

where the vector ˜ z ∈ Z ⊆ R m comprises the random parameters, and P denotes the underlying distribution. We assume that the loss function is convex in θ for any fixed z ∈ Z . When the solution of a stochastic optimization problem affects the distribution of the uncertain parameters, we call such problem performative [40, 15]. A natural way of introducing this decision-dependent uncertainty is through a map P ( · ) from the set of decisions to the space of distributions. Hence, in performative stochastic optimization problems, the objective of the decision maker is to obtain a decision θ that minimizes the performative risk

<!-- formula-not-decoded -->

where P ( θ ) is the distribution of uncertain parameters ˜ z given the choice of the decision θ . Unfortunately, the true underlying distribution map P ( θ ) is unknown to the decision makers and usually approximated using the reference distribution ˆ P ( θ ) = ∑ s ∈ [ S ] ˆ p s ( θ ) δ ˆ z s ( θ ) , where { ˆ z s ( θ ) } s ∈ [ S ] represent plausible scenarios and { ˆ p s ( θ ) } s ∈ [ S ] are their respective probabilities. This reference distribution could be derived from observations or expert knowledge elicitation.

Although methods based on a reference distribution may perform well when the true distribution closely aligns with it, their performance often deteriorates when this assumption is violated. Distributionally robust optimization has proven effective in addressing such distributional ambiguity in non-performative stochastic optimization settings. Unlike traditional approaches that assume a single distribution map, distributionally robust optimization adopts a more flexible strategy: it establishes an ambiguity set B ( ˆ P ( θ )) of distributions that are close to the reference distribution. In this paper, we construct this ambiguity set based on the Wasserstein distance. A formal definition of the Wasserstein distance is as follows.

Definition 1 (Wasserstein metric) . For any r ≥ 1 , let M ( Z ) be the space of all probability distributions Q supported on Z satisfying E Q [ c (˜ z , z 0 ) r ] = ∫ Ξ c ( z , z 0 ) r Q ( d z ) &lt; ∞ , where z 0 ∈ Z is some reference point and c ( z , z 0 ) is a non-negative, continuous and thus lower semi-continuous [55] reference metric on Z . The typer (1 ≤ r ) Wasserstein distance between two distributions Q 1 and Q 2 is defined as

<!-- formula-not-decoded -->

where Π( Q 1 × Q 2 ) is the set of all joint probability distributions of random vectors z 1 and z 2 with marginals Q 1 and Q 2 , respectively.

The Wasserstein metric offers a natural way of comparing two distributions when one is derived from the other by small perturbations. The decision variable π can be interpreted as a transportation plan for moving a mass distribution denoted by Q 1 to another one denoted by Q 2 , where the transportation cost between two points z 1 and z 2 is given by c ( z 1 , z 2 ) . Therefore, the r -Wasserstein distance can be viewed as the r -th root of the minimum transportation cost between Q 1 and Q 2 . An important advantage of the Wasserstein distance is its ability to handle distributions with non-overlapping supports, i.e., even when Q 1 and Q 2 have different supports, the Wasserstein distance provides a finite, meaningful value. By contrast, measures like KL-divergence become undefined under such scenarios. We now consider the following Wasserstein ambiguity set

<!-- formula-not-decoded -->

which is a neighborhood around the reference distribution ˆ P ( θ ) . The Wasserstein ambiguity set contains all the distributions whose r -Wasserstein distance from ˆ P ( θ ) is less than or equal to ρ .

Equipped with the Wasserstein ambiguity set, the decision maker minimizes the distributionally robust performative risk

<!-- formula-not-decoded -->

where Q is a distribution from within the prescribed ambiguity set B ( ˆ P ( θ )) . In other words, the model optimizes the expected risk over the worst-case distribution map, thereby mitigating overfitting to the reference distribution and improving generalization performance to other plausible distributions. We now introduce the following concepts regarding the optimality and stability of the solutions.

Definition 2 (Robust performative optimality) . A decision θ RPO is robust performatively optimal if the following relationship holds:

<!-- formula-not-decoded -->

Definition 3 (Robust performative stability) . A decision θ RPS is robust performatively stable if the following relationship holds:

<!-- formula-not-decoded -->

While distinct from the robust performatively optimal solution, a robust performatively stable solution constitutes a fixed point of the problem and is optimal with respect to the worst-case expected loss over the ambiguity set it induces.

Definition 4 (Robust decoupled performative risk) . We define

<!-- formula-not-decoded -->

as the robust decoupled performative risk , separating the decision η associated with the ambiguity set and the decision θ associated with the risk; then, θ RPS ∈ arg min θ J θ RPS ( θ ) .

## 3 Repeated Robust Risk Minimization Algorithm

In this section, we introduce the repeated robust risk minimization algorithm for solving the distributionally robust performative risk minimization problem and investigate its fundamental properties. The algorithm starts with an initial solution θ 0 , and for every t ≥ 0 , the subsequent θ t +1 can be obtained by solving the following robust risk minimization problem:

<!-- formula-not-decoded -->

The algorithm addresses the computational challenges posed by decision-dependent distributions by constructing the ambiguity set using the reference distribution based on the optimal decision from the previous iteration. Thus, it effectively decouples the current decision from the ambiguity set, simplifying the optimization process.

## 3.1 Models and Their Tractable Reformulations

We present three models that cover a broad spectrum of problems in machine learning and decisionmaking under uncertainty. We first consider the case where the loss function is given by the composition of a Lipschitz continuous function and a quadratic function. This class of problems is particularly relevant in machine learning and robust statistics, as it captures many commonly used models such as linear regression [36], logistic regression [22], and certain types of support vector machines [48].

Model 1. Assume that the loss function is defined as

<!-- formula-not-decoded -->

where ˜ Z = ( ˜ Y , ˜ z , ˜ z 0 ) includes random variables ˜ Y ∈ S N , ˜ z ∈ R d and ˜ z 0 ∈ R . The function L ( · ) is assumed to be L -Lipschitz continuous.

We consider the 1-Wasserstein ball, where the ground cost function c is given by the Schatten-∞ norm. Under this setting, (RRMP) is equivalent to the Tikhonov regularized problem

<!-- formula-not-decoded -->

This result demonstrates that, under mild assumptions, (RRMP) can be reformulated as a regularized risk minimization problem. This substantially generalizes the findings of [30], who established an equivalence to Tikhonov regularization for strongly convex quadratic loss functions and martingaleconstrained Wasserstein ambiguity sets. Our result reveals that Tikhonov regularization can be obtained for a broader class of loss functions without requiring complicating martingale constraints, thereby sharpening the theoretical understanding of the connection between distributional robustness and regularization.

The following model provides a general formulation that is typically considered in the distributionally robust optimization literature [43, 44]. This model is pertinent to many applications in decisionmaking under uncertainty, including in inventory management [29] and and energy systems [27]. To the best of our knowledge, such a formulation has not previously been proposed in the performative optimization literature, which mainly focuses on prediction problems. Note that if the L -lipschitz continuous function L in (5) is piecewise linear, then this model constitutes a generalization.

Model 2. Assume that the loss function is defined as

<!-- formula-not-decoded -->

where ˜ Z = ( ˜ Y , ˜ z , ˜ z 0 ) includes random variables ˜ Y ∈ S N , ˜ z ∈ R d and ˜ z 0 ∈ R . Each component Q j ( Z , θ ) is a quadratic function of the form

<!-- formula-not-decoded -->

with parameter-dependent coefficients given by affine functions a j ( θ ) = a j + A j θ , b j ( θ ) = b j + B j θ , and c j ( θ ) = c j 0 + c ⊤ j θ for all j ∈ [ J ] , where a j , b j ∈ R N , c j 0 ∈ R , A j , B j ∈ R N × d , and c j ∈ R d .

Consider the 1-Wasserstein ball with the Schatten-∞ norm ground cost. Then, the optimal value of the following exponential conic program provides an arbitrarily tight conservative approximation for (RRMP) .

<!-- formula-not-decoded -->

where ˆ Y S +1 = ρ I , ˆ z S +1 = 0 , ˆ z 0 S +1 = ρ , and µ ∈ R + is the smoothing parameter.

The formulation (6) relies on the exponential smoothing techniques described in [7, Section 2.2]. The primary motivation for this approach lies in the need to handle the nonsmoothness incurred by the inner maximization over quadratic functions, which will hinder the convergence of our proposed algorithm. To address this challenge, we apply a log-sum-exp smoothing approximation with smoothing parameter µ &gt; 0 , which yields a smoothed robust decoupled performative risk, denoted as J µ θ t . And problem (6) is equivalent to inf θ ∈ Θ J µ θ t ( θ ) (see details in Appendix D).

As is standard in exponential smoothing, the smoothed objective serves as a uniform upper bound to the original nonsmooth function: J θ t ( θ ) ≤ J µ θ t ( θ ) ∀ θ ∈ Θ , which ensures that optimization over the smooth surrogate does not underestimate the original objective. Importantly, J µ θ t epi-converges to J θ t as the smoothing parameter µ ↓ 0 [46, Theorem 7.17], which ensures convergence of optimal solutions whenever Θ is compact [46, Theorem 7.33].

Finally, inspired by minimax [52] and adversarially robust optimization literature [49], we turn to the setting where the loss function is convex-concave. In this setting, we exploit the 2-Wasserstein ambiguity set to ensure the convergence of the repeated risk minimization algorithm. This model further allows one to impose additional structural support information that may reduce the overconservatism of the distributionally robust solutions. The following theorem provides a convex reformulation for the model, leveraging convex conjugate representations and support functions.

Model 3. Let Z ⊆ R m be a nonempty, convex and closed set, and consider the 2-Wasserstein ball, where the ground cost c is given by the Euclidean norm on R d . Suppose that for every θ , the function ℓ ( z , θ ) is proper, concave, and upper-semicontinuous in z . Then, the optimal value of the following finite convex program provides an arbitrarily tight conservative approximation for (RRMP) :

<!-- formula-not-decoded -->

where τ ∈ R ++ is a constant, [ -ℓ ] ∗ ( ξ , θ ) = sup z ∈ R m ξ ⊤ z -[ -ℓ ( z , θ )] denotes the conjugate of -ℓ with respect to z , and σ Z ( ξ ) = sup z ∈Z ξ ⊤ z is the support function of Z ∈ R m .

Note that in problem (7), the loss function ℓ ( z , θ ) and the support set Z enter the formulation through the convex conjugate of the negative loss [ -ℓ ] ∗ ( · , θ ) , and the support function σ Z ( · ) , respectively. Both transformations yield convex functions under the assumptions stated in Model 3. Furthermore, the term (1 / 4 λ ) ∥ r s ∥ 2 2 is jointly convex in ( λ, r s ) [9, section 3.2.6]. Consequently, all objective and constraint functions in problem (7) are convex, and the overall optimization problem is manifestly convex.

The reformulation in Model 3 introduces the dual variable λ , which appears alongside the decision variable θ . This motivates the use of an augmented vector θ = ( θ , λ ) ∈ Θ × R + = Θ . To ensure strong convexity in the joint variable θ which is important for the convergence guarantees of the R 3 M algorithm, we introduce a regularization term τλ 2 , where τ &gt; 0 . This leads to the regularized robust decoupled performative risk, denoted by J τ θ t ( θ ) . And problem (7) is equivalent to inf θ ∈ Θ J τ θ t ( θ ) . Notice that for any θ ∈ Θ , inf λ ≥ 0 J τ θ t ( θ , λ ) converges to J θ t ( θ ) as τ ↓ 0 . If ℓ ( z , θ ) is lower semicontinuous in θ , then so is J θ t . Hence, inf λ ≥ 0 J τ θ t ( · , λ ) , epi-converges to J θ t ( · ) [46, Theorem 7.17], and the minimizer ˆ θ of (7) converges to a minimizer of (RRMP) whenever Θ is compact [46, Theorem 7.33].

## 3.2 Convergence Analysis

Wenowestablish convergence guarantees for the RRRM algorithm when applied to the reformulations introduced in Models 1, 2, and 3. As noted earlier, we employ the smoothed objective J µ θ t ( θ ) for (RRMP) in Model 2 and the regularized objective J τ θ t ( θ ) in Model 3. Each model requires specific assumptions to ensure contraction of the risk map and hence convergence:

- ( A1 ) Model 1: The loss function satisfies the γ -strong convexity (B1) and β -smoothness (B2).
- ( A2 ) Model 2: For all j ∈ [ J ] , A ⊤ j A j ≻ 0 and let α = 2min j ∈ [ J ] λ min ( A ⊤ j A j ) . In addition, the feasible set Θ and the support of ˆ P ( θ t ) are bounded.
- ( A3 ) Model 3: The loss function satisfies the γ -strong convexity (B1) and β -jointly smoothness (B2). Additionally the support Z has a finite diameter D &lt; ∞ .

Under these respective conditions, the algorithm converges linearly to a stable point.

Theorem 1. Suppose the loss functions in Models 1, 2, and 3 satisfy Assumptions ( A1 ), ( A2 ), and ( A3 ) respectively, and that the distribution map ˆ P ( · ) satisfies the ϵ -sensitivity condition (B3) . Then:

- (a) ∥ θ t +1 -θ ′ t +1 ∥ 2 ≤ ϵκ ∥ θ t -θ ′ t ∥ 2 for all θ t , θ ′ t ∈ Θ .
- (b) if ϵκ &lt; 1 , the iterate θ t of (RRMP) converges linearly to a unique performatively stable point θ RPS :

<!-- formula-not-decoded -->

where ∆ &gt; 0 is a predefined tolerance level. The fixed point θ RPS depends on the model and is denoted θ µ RPS under Model 2, and θ τ RPS under Model 3.

Here, κ = β/ ( γ + 2 ρL ) for Model 1, κ = Jdk 3 ( k 1 k 2 µ +1 ) /ρα for Model 2, and κ = ( β + 4 D ) / min { 2 τ, γ } for Model 3, where k 1 , k 2 , k 3 &lt; ∞ are model-dependent constants.

This convergence result highlights several compelling advantages of our DRPO framework, particularly when contrasted with traditional approaches to performative optimization that necessitate strong convexity and smoothness for convergence. First, our DRPO framework establishes convergence guarantees for loss functions that are convex but not necessarily strongly convex. Additionally, our exponential smoothing tricks applied in Model 2 enable the RRRM algorithm to converge for non-smooth loss functions. Finally, our DRPO framework accelerates the convergence rate compared with its non-robust counterpart. These advancements significantly broadens the scope of problems amenable to performative optimization and enhance the computational efficiency in practical problems.

## 3.3 Suboptimality Guarantees

Our next result demonstrates that the robust performatively stable solution θ RPS is close to the robust performatively optimal solution θ RPO whenever the ϵ -sensitivity of the distribution map is small, the loss function has a large strong convexity parameter γ , or the distributionally robust model induces a regularization with large strong convexity parameter ρ .

Theorem 2. Suppose all conditions in Theorem 1 hold. Additionally, assume that the loss function is L z -Lipschitz in z for both models, and L θ -Lipschitz in θ for Model 3. Then, the suboptimality gap between the robust performatively stable solution and the optimal solution under the true distribution is bounded as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, µ ′ ∈ [0 , 1] is a constant that satisfies µ ≤ µ ′ ∥ θ µ RPS -θ RPO ∥ 2 , and λ is the upper bound on the optimal λ as given in Lemma F.1.

The suboptimality bound for Model 2 also highlights the advantages of our DRPO framework alongside the smoothing tricks. For non-smooth, convex but not strongly-convex functions, the suboptimality bound under the traditional performative prediction framework can be arbitrarily large (as ρ → 0 , µ → 0 ) as suggested by our result.

## 4 Experiments

In this section, we present numerical experiments to evaluate the performance of our proposed models across three applications: strategic classification, revenue management, and portfolio optimization. All experiments were conducted on a laptop equipped with a 6-core, 2.3 GHz Intel Core i7 CPU and 16 GB of RAM. The optimization problems were implemented in Python 3.11.

## 4.1 Strategic Classification

We consider a simulated strategic classification problem from [40] using a class-balanced subset of a Kaggle credit scoring dataset [25]. The dataset contains features ˜ x ∈ R P about borrowers, such as their ages and the number of open loans. The outcomes ˜ y ∈ {-1 , 1 } are equal to 1 if the individual defaulted on a loan and -1 otherwise. The institution's objective is to predict whether an individual will default on their debt.

Under the strategic classification setting, individuals respond to the institution's classifier by altering their features to increase their likelihood of receiving a favorable classification. The institution employs logistic regression for classification, with ˜ z = ˜ x ˜ y , and the loss function is given by log(1 + exp( -x ⊤ θ y )) . This setting aligns with Theorem 1, where L represents the logloss function with Lipschitz constant L = 1 , and the quadratic function θ ⊤ Y θ +2 z ⊤ θ + z 0 simplifies to the affine function z ⊤ θ . See Appendix G for additional details.

We compare the performance of our robust models (with type-1 and type-2 Wasserstein ambiguity sets) against the alternating minimization algorithm under KL divergence ambiguity set (AMKL)

from [58]. Additionally, we include a non-robust model as a baseline for comparison. The training set is fixed at 200 samples, while approximately 3,600 data points are used for out-of-sample testing. This setup reflects realistic scenarios where data collection is costly or limited. In credit scoring, for example, obtaining labeled data often requires expensive evaluations, expert assessments, or lengthy observation periods. All models are trained for 40 iterations, with the robust parameter set to 0.1 for all robust variants.

Figure 1: Out of sample performance of different approaches

<!-- image -->

Figure 1 displays the box plots of 50 independent trials. As shown, our robust models outperform the AMKL algorithm, which performs quite poorly. This may be due to its susceptibility to local optima, as well as the limitations of the KL divergence ambiguity set, which may not sufficiently guard against distributional shifts that are not well represented in the reference distribution. All robust models significantly outperform the non-robust baseline, highlighting the effectiveness of distributional robustness. Finally, we find that the robust model with a type-2 Wasserstein ambiguity set outperforms the one with a type-1 Wasserstein ambiguity set. This arises primarily due to the geometry of the ambiguity set. As discussed in [10], selecting the optimal radius ρ is challenging, and the 2-Wasserstein ball often provides better performance because it offers a wider range of radius values for which the robust solution outperforms its non-robust counterpart.

## 4.2 Revenue Management

In this experiment, we address the revenue management problem where the decision-maker determines the unit price θ ≥ 0 for a fixed quantity of perishable products q ∈ Z ++ , such as hotel rooms or airplane seats, under uncertain demand ˜ z ∼ P ( θ ) , with higher prices inducing lower demand [1, 13, 42].

Following [41], we model the pricedependent demands using an additive function ˜ z ( θ ) = -aθ + b + ˜ ϵ. Here,

Figure 2: Out-of-sample performance of pricing schemes

<!-- image -->

unknown parameters a &gt; 0 and b &gt; 0 capture a deterministic linear demand curve, and ˜ ϵ is a random variable with a bounded support that is characterized by an unknown density function. Instead of the true model, we only have access to a surrogate model -¯ aθ + ¯ b , which deviates from the true model as the parameters ¯ a and ¯ b do not accurately represent their counterparts. Under this additive demand setting, the associated loss function becomes a piecewise quadratic function in θ , allowing us to apply Model 2 to formulate a robust version of the revenue management problem, which can be efficiently solved within seconds using an off-the-shelf commercial solver as it is a convex problem.

Figure 2 compares the out-of-sample performance of our robust scheme (orange) with benchmarks. Due to the non-smoothness of the loss function, the AMKL approach is not applicable in this setting. As shown in [41], when the true distribution map is available, the optimal price (green) can be derived in closed form, achieving the highest expected profit for each fixed quantity q in the left subplot. Similarly, the non-robust price (blue) can be obtained by treating the surrogate model as the true model. The confidence region in the left subplot, representing the 10th-90th percentile range of 100 independent tests, and the smaller optimality gap relative to the mean profit induced by the optimal price in the right subplot show that our robust price consistently outperforms the benchmark.

## 4.3 Demand Response Portfolio Optimization

We evaluate our robust scheme in a power system application, focusing on demand response (DR) portfolio optimization [51, 60]. In electricity markets, consumers capable of lowering electricity consumption during certain periods are called DR resources. In this experiment, we consider a DR aggregator (decision-maker) managing n DR resources over a planning horizon of T periods. The goal is to maximize the expected profit by determining commitment level θ t ∈ R n + to meet a required deterministic demand reduction D t at each time t ∈ [ T ] .

The challenge lies in the uncertainty of DR resource's performance, where the scheduled commitment level θ t may significantly differ from the actual reduction level ˜ θ t due to random noise ˜ z t . This noise is decision-dependent, as larger commitments lead to higher variability. In this experiment, we model the actual reduction of each resource i as ˜ θ t,i = θ t,i ˜ z t,i for all i ∈ [ n ] . Here, the multiplicative noise follows a beta distribution whose parameters α and β depend linearly on the commitment level: ˜ z t,i ∼ 2 · Beta ( α = β = a i θ t,i + b i ) with a i &lt; 0 . This beta distribution has a support of [0 , 2] and a mean of 1, regardless of the value of a i θ t,i + b i . Therefore, the decision θ t,i only influences the distribution shape, with higher commitment levels leading to heavier tails, hence, higher variability of the actual reduction level ˜ θ t,i .

We consider three DR resources with distinct characteristics: Resource 1 has high revenue but large variability, Resource 2 has low revenue but high predictability, and Resource 3 offers a balanced trade-off between the two. We follow the experiment setup in [12], including the loss function and the values of unit revenue, over-commitment cost, and under-commitment penalty. As their loss function is piecewise-linear, the resulting optimization problem can be formulated using Model 2 and efficiently solved. For further details, we refer the reader to the original paper. We conduct two out-of-sample tests: low and high demand loads over the planning horizon, corresponding to small

Figure 3: Out-of-sample performance of DR scheduling

<!-- image -->

D t and large D t , respectively. Figure 3 compares the mean profits of our robust scheme (orange) and the non-robust scheme (blue), showing that the robust approach outperforms the non-robust one. Notably, under high demand, the non-robust scheme often performs significantly poorly, resulting in losses in some cases.

## 5 Concluding Remarks

We have presented the first Wasserstein distributionally robust optimization framework for performative optimization. In contrast to existing approaches, our framework accommodates a broader class of problems in decision-making under uncertainty, thereby extending the original scope of performative prediction. We proposed an efficient algorithm and established its convergence and suboptimality guarantees. To our knowledge, these theoretical results have not been previously established in the literature on robust performative prediction. Our experimental results demonstrate the superiority of our approach over existing methods on the standard strategic classification benchmark, as well as in two decision-making applications: revenue management and demand response portfolio optimization.

Notably, as the ambiguity set radius ρ approaches zero, the robust objective coincides with the non-robust counterpart, which more directly targets the performative risk. This observation suggests a possible direction: designing algorithms that gradually shrink the ambiguity set over time, potentially trading robustness for improved approximation of the true performative risk as more information becomes available. Another promising direction is contextual performative optimization, where incorporating side information could further improve decision quality by enabling more accurate modeling of uncertainty.

Broader impacts. Our framework extends the scope of performative prediction beyond its original focus, enabling its application to a wider range of decision-making problems. In high-stakes settings, adopting a distributionally robust optimization perspective allows our approach to prioritize safe and reliable deployment in the presence of uncertainty and potential adversarial conditions.

## Acknowledgments and Disclosure of Funding

Grani A. Hanasusanto is supported in part by the National Science Foundation (NSF) under Grants CCF-2343869 and ECCS-2404413. Roy Dong is supported in part by NSF under Grant CCF-2236484. Yijie Wang is supported in part by the Fundamental Research Funds for the Central Universities. We thank Hyuk Park for assistance with the numerical experiments and the anonymous reviewers for their constructive feedback that helped improve this work.

## References

- [1] J. Alstrup, S.-E. Andersson, S. Boas, O. B. Madsen, and R. V. V. Vidal. Booking control increases profit at scandinavian airlines. Interfaces , 19(4):10-19, 1989.
- [2] M. Baker and J. Wurgler. Investor sentiment and the cross-section of stock returns. The journal of Finance , 61(4):1645-1680, 2006.
- [3] P. L. Bartlett. Learning with a slowly changing distribution. In Proceedings of the fifth annual workshop on Computational learning theory , pages 243-252, 1992.
- [4] P. L. Bartlett, S. Ben-David, and S. R. Kulkarni. Learning changing concepts by exploiting the structure of change. In Proceedings of the ninth annual conference on Computational learning theory , pages 131-139, 1996.
- [5] B. Basciftci, S. Ahmed, and S. Shen. Distributionally robust facility location problem under decision-dependent stochastic demand. European Journal of Operational Research , 292(2):548561, 2021.
- [6] D. Bertsekas. Convex optimization theory , volume 1. Athena Scientific, 2009.
- [7] D. Bertsekas. Convex optimization algorithms . Athena Scientific, 2015.
- [8] J. Blanchet and K. Murthy. Quantifying distributional model risk via optimal transport. Mathematics of Operations Research , 44(2):565-600, 2019.
- [9] S. P. Boyd and L. Vandenberghe. Convex optimization . Cambridge university press, 2004.
- [10] G. Byeon. Comparative analysis of two-stage distributionally robust optimization over 1wasserstein and 2-wasserstein balls. arXiv preprint arXiv:2501.05619 , 2025.
- [11] A. P. Calmon, F. D. Ciocan, and G. Romero. Revenue management with repeated customer interactions. Management Science , 67(5):2944-2963, 2021.
- [12] H. Chen, X. A. Sun, and H. Yang. Robust optimization with continuous decision-dependent uncertainty with applications to demand response management. SIAM Journal on Optimization , 33(3):2406-2434, 2023.
- [13] G. D. DeYong. The price-setting newsvendor: review and extensions. International Journal of Production Research , 58(6):1776-1804, 2020.
- [14] X. V. Doan. Distributionally robust optimization under endogenous uncertainty with an application in retrofitting planning. European Journal of Operational Research , 300(1):73-84, 2022.
- [15] D. Drusvyatskiy and L. Xiao. Stochastic optimization with decision-dependent distributions. Mathematics of Operations Research , 48(2):954-998, 2023.
- [16] J. Gama, I. Žliobait˙ e, A. Bifet, M. Pechenizkiy, and A. Bouchachia. A survey on concept drift adaptation. ACM computing surveys (CSUR) , 46(4):1-37, 2014.
- [17] B. Gao and L. Pavel. On the properties of the softmax function with application in game theory and reinforcement learning. arXiv preprint arXiv:1704.00805 , 2017.

- [18] V. Goel and I. E. Grossmann. A stochastic programming approach to planning of offshore gas field developments under uncertainty in reserves. Computers &amp; chemical engineering , 28(8):1409-1429, 2004.
- [19] V. Goel and I. E. Grossmann. A class of stochastic programs with decision dependent uncertainty. Mathematical programming , 108(2):355-394, 2006.
- [20] M. Hardt, M. Jagadeesan, and C. Mendler-Dünner. Performative power. Advances in Neural Information Processing Systems , 35:22969-22981, 2022.
- [21] M. Hardt and C. Mendler-Dünner. Performative prediction: Past and future. arXiv preprint arXiv:2310.16608 , 2023.
- [22] D. W. Hosmer Jr, S. Lemeshow, and R. X. Sturdivant. Applied logistic regression . John Wiley &amp;Sons, 2013.
- [23] Q. Jin, A. Georghiou, P. Vayanos, and G. A. Hanasusanto. Distributionally robust optimization with decision-dependent information discovery. arXiv preprint arXiv:2404.05900 , 2024.
- [24] T. W. Jonsbråten, R. J. Wets, and D. L. Woodruff. A class of stochastic programs withdecision dependent random elements. Annals of Operations Research , 82(0):83-106, 1998.
- [25] Kaggle. Give me some credit. https://www.kaggle.com/c/GiveMeSomeCredit/data , 2012.
- [26] L. V. Kantorovich and S. Rubinshtein. On a space of totally additive functions. Vestnik of the St. Petersburg University: Mathematics , 13(7):52-59, 1958.
- [27] J. H. Kim and W. B. Powell. Optimal energy commitments with storage and intermittent supply. Operations research , 59(6):1347-1360, 2011.
- [28] M. P. Kim and J. C. Perdomo. Making decisions under outcome performativity. In 14th Innovations in Theoretical Computer Science Conference (ITCS 2023) . Schloss-DagstuhlLeibniz Zentrum für Informatik, 2023.
- [29] S. Lee, H. Kim, and I. Moon. A data-driven distributionally robust newsvendor model with a wasserstein ambiguity set. Journal of the Operational Research Society , 72(8):1879-1897, 2021.
- [30] J. Li, S. Lin, J. Blanchet, and V. A. Nguyen. Tikhonov regularization is optimal transport robust under martingale constraints. Advances in Neural Information Processing Systems , 35:17677-17689, 2022.
- [31] F. Luo and S. Mehrotra. Distributionally robust optimization with decision dependent ambiguity sets. Optimization Letters , 14(8):2565-2594, 2020.
- [32] C. Mendler-Dünner, J. Perdomo, T. Zrnic, and M. Hardt. Stochastic optimization for performative prediction. Advances in Neural Information Processing Systems , 33:4929-4939, 2020.
- [33] G. Michel, J. Omer, and M. Poss. Robust selection problem with decision-dependent information discovery under budgeted uncertainty. In 23ème congrès annuel de la Société Française de Recherche Opérationnelle et d'Aide à la Décision , 2022.
- [34] J. P. Miller, J. C. Perdomo, and T. Zrnic. Outside the echo chamber: Optimizing the performative risk. In International Conference on Machine Learning , pages 7710-7720. PMLR, 2021.
- [35] P. Mohajerin Esfahani and D. Kuhn. Data-driven distributionally robust optimization using the wasserstein metric: Performance guarantees and tractable reformulations. Mathematical Programming , 171(1):115-166, 2018.
- [36] D. C. Montgomery, E. A. Peck, and G. G. Vining. Introduction to linear regression analysis . John Wiley &amp; Sons, 2021.

- [37] O. Nohadani and K. Sharma. Optimization under decision-dependent uncertainty. SIAM Journal on Optimization , 28(2):1773-1795, 2018.
- [38] R. Paradiso, A. Georghiou, S. Dabia, and D. Tönissen. Exact and approximate schemes for robust optimization problems with decision dependent information discovery. arXiv preprint arXiv:2208.04115 , 2022.
- [39] L. Peet-Pare, N. Hegde, and A. Fyshe. Long term fairness for minority groups via performative distributionally robust optimization. arXiv preprint arXiv:2207.05777 , 2022.
- [40] J. Perdomo, T. Zrnic, C. Mendler-Dünner, and M. Hardt. Performative prediction. In International Conference on Machine Learning , pages 7599-7609. PMLR, 2020.
- [41] N. C. Petruzzi and M. Dada. Pricing and the newsvendor problem: A review with extensions. Operations research , 47(2):183-194, 1999.
- [42] A. Popescu, P. Keskinocak, E. Johnson, M. LaDue, and R. Kasilingam. Estimating air-cargo overbooking based on a discrete show-up-rate distribution. Interfaces , 36(3):248-258, 2006.
- [43] H. Rahimian and S. Mehrotra. Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659 , 2019.
- [44] H. Rahimian and S. Mehrotra. Frameworks and results in distributionally robust optimization. Open Journal of Mathematical Optimization , 3:1-85, 2022.
- [45] R. T. Rockafellar. Convex Analysis . Princeton University Press, 1970.
- [46] R. T. Rockafellar and R. J.-B. Wets. Variational analysis , volume 317. Springer Science &amp; Business Media, 2009.
- [47] M. Ryu and R. Jiang. Nurse staffing under absenteeism: A distributionally robust optimization approach. arXiv preprint arXiv:1909.09875 , 2019.
- [48] S. Shafieezadeh-Abadeh, D. Kuhn, and P. M. Esfahani. Regularization via mass transportation. Journal of Machine Learning Research , 20(103):1-68, 2019.
- [49] A. Sinha, H. Namkoong, R. Volpi, and J. Duchi. Certifying some distributional robustness with principled adversarial training. arXiv preprint arXiv:1710.10571 , 2017.
- [50] S. A. Spacey, W. Wiesemann, D. Kuhn, and W. Luk. Robust software partitioning with multiple instantiation. INFORMS Journal on Computing , 24(3):500-515, 2012.
- [51] J.-H. Teng and C.-H. Hsieh. Modeling and investigation of demand response uncertainty on reliability assessment. Energies , 14(4):1104, 2021.
- [52] K. K. Thekumparampil, P. Jain, P. Netrapalli, and S. Oh. Efficient algorithms for smooth minimax optimization. Advances in neural information processing systems , 32, 2019.
- [53] J. v. Neumann. Zur theorie der gesellschaftsspiele. Mathematische annalen , 100(1):295-320, 1928.
- [54] P. Vayanos, D. Kuhn, and B. Rustem. Decision rules for information discovery in multi-stage stochastic programming. In 2011 50th IEEE Conference on Decision and Control and European Control Conference , pages 7368-7373. IEEE, 2011.
- [55] C. Villani. Topics in optimal transportation , volume 58. American Mathematical Soc., 2021.
- [56] R. Vujanic, P. Goulart, and M. Morari. Robust optimization of schedules affected by uncertain events. Journal of Optimization Theory and Applications , 171:1033-1054, 2016.
- [57] R. Wermers. Mutual fund herding and the impact on stock prices. the Journal of Finance , 54(2):581-622, 1999.
- [58] S. Xue and Y. Sun. Distributionally robust performative prediction. arXiv preprint arXiv:2412.04346 , 2024.

- [59] X. Yu and S. Shen. Multistage distributionally robust mixed-integer programming with decisiondependent moment-based ambiguity sets. Mathematical Programming , 196(1):1025-1064, 2022.
- [60] J. Zhang and A. D. Domínguez-García. Evaluation of demand response resource aggregation system capacity under uncertainty. IEEE Transactions on Smart Grid , 9(5):4577-4586, 2017.
- [61] J. Zhang, H. Xu, and L. Zhang. Quantitative stability analysis for distributionally robust optimization with moment constraints. SIAM Journal on Optimization , 26(3):1855-1882, 2016.
- [62] Q. Zhang and W. Feng. A unified framework for adjustable robust optimization with endogenous uncertainty. AIChe journal , 66(12):e17047, 2020.
- [63] X. Zhang, M. Kamgarpour, A. Georghiou, P. Goulart, and J. Lygeros. Robust optimal control with adjustable uncertainty sets. Automatica , 75:249-259, 2017.
- [64] J. Zhen, D. Kuhn, and W. Wiesemann. Mathematical foundations of robust and distributionally robust optimization. arXiv preprint arXiv:2105.00760 , 2021.

## Technical Appendices and Supplementary Material

## A Preliminary Definitions

Definition 5 (Generalized strong convexity) . We say that a loss function ℓ ( z , θ ) is γ -strongly convex in θ if

<!-- formula-not-decoded -->

for all θ , θ ′ ∈ Θ and z ∈ Z . If γ = 0 , this condition reduces to the standard definition of convexity. We will also use the following equivalent definition of strong convexity. A loss function ℓ ( z , θ ) is γ -strongly convex in θ if the function

<!-- formula-not-decoded -->

is convex for all z ∈ Z .

Definition 6 (Smoothness) . We say that a loss function ℓ ( z , θ ) is β -smooth if the gradient ∇ θ ℓ ( z , θ ) is β -Lipschitz in z , that is

<!-- formula-not-decoded -->

for all z , z ′ ∈ Z .

Definition 7 ( ϵ -sensitivity) . We say that a distribution map P ( · ) is ϵ -sensitive if for all θ, θ ′ ∈ Θ

<!-- formula-not-decoded -->

where W 1 denotes the 1-Wasserstein metric.

## B Background on Exponential Smoothing

Let Z = { z 1 , . . . , z n } denote a finite support set. Given a loss function ℓ : Θ × Z → R and a distribution ˆ p ∈ ∆ n , where ∆ n denotes the probability simplex. We define the exponentially smoothed objective as:

<!-- formula-not-decoded -->

where µ &gt; 0 is a smoothing parameter.

This function, also known as the log-sum-exp function, is a widely used smooth approximation to the pointwise maximum function and has well-established properties in convex analysis [Bertsekas, 2015; Section 2.2]. This function provides a smooth approximation to max i ℓ ( θ, z i ) whenever ˆ p i &gt; 0 for all i .

Properties. The function f µ ( θ ) satisfies the following:

- Approximation Bounds: The approximation error can be precisely quantified:

<!-- formula-not-decoded -->

provided ˆ p i &gt; 0 for all i . Thus, as µ → 0 , the smoothed objective f µ ( θ ) approaches the exact maximum, with the error vanishing linearly in µ up to a logarithmic multiplicative factor.

- Convexity and Differentiability: If each function ℓ ( θ, z i ) is convex in θ , then f µ ( θ ) is also convex, as it is a composition of convex functions closed under nonnegative weighted log-sum-exp operations. Moreover, f µ ( θ ) is continuously differentiable for all µ &gt; 0 , with gradient:

<!-- formula-not-decoded -->

where the weight vector π µ ( · ; θ ) ∈ ∆ n defines a softmax distribution:

<!-- formula-not-decoded -->

This smoothing mechanism not only ensures differentiability but also facilitates efficient computation of gradients for robust optimization objectives involving maxima.

## C Auxiliary Lemmas

Lemma C.1 (First-order optimality condition; Section 4.2.3 in [9]) . Let f be a convex function and let Ω be a closed convex set on which f is differentiable, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma C.2 ([26]) . A distribution map P ( · ) is ϵ -sensitive if and only if for all θ , θ ′ ∈ Θ , we have

<!-- formula-not-decoded -->

if and only if where

<!-- formula-not-decoded -->

is the space of all L -Lipschitz continuous functions.

Lemma C.3. If ℓ ( z , θ ) is γ -strongly convex, then the worst case expectation

<!-- formula-not-decoded -->

is γ -strongly convex in θ .

Proof. By the equivalent definition of γ -strong convexity in (B1'), we have that ℓ ( z , θ ) -γ 2 ∥ θ ∥ 2 is convex in θ . Hence, the worst-case expectation

<!-- formula-not-decoded -->

is convex in θ since the expectation and the pointwise supremum operations preserve convexity. Thus, we have

<!-- formula-not-decoded -->

is γ -strongly convex in θ by the definition (B1').

Lemma C.4. Let x ∈ R J and define the smooth maximum function

<!-- formula-not-decoded -->

for any µ &gt; 0 . Then the following bounds hold:

<!-- formula-not-decoded -->

Proof. Let M := max j ∈ [ J ] x j . Then for each j ∈ [ J ] , we have x j ≤ M , and hence:

<!-- formula-not-decoded -->

Taking logarithms and multiplying by µ , we obtain the upper bound:

<!-- formula-not-decoded -->

For the lower bound, observe that and introducing the matrix variable

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since at least one term in the sum equals e M/µ . Thus:

<!-- formula-not-decoded -->

Combining both bounds in (8) and (9), we have the desired result:

<!-- formula-not-decoded -->

## D Deferred Proofs Related to Reformulations

## D.1 Proof of Reformulation for Model 1

Proof. We begin by rewriting the random parameters ˜ Z as

<!-- formula-not-decoded -->

Next, we introduce the matrix variable

<!-- formula-not-decoded -->

which allows us to rewrite the loss function as ℓ ( Z , θ ) = L ( ⟨ Γ , Z ⟩ ) . According to [8, Remark 1], the worst-case expected loss over a 1-Wasserstein ambiguity set B ( ˆ P ( θ t )) , with cost induced by the Schatten-∞ norm, is given by:

<!-- formula-not-decoded -->

By applying [48, Lemma 47], the inner maximization problem can be simplified as:

<!-- formula-not-decoded -->

Hence, the worst-case expectation reduces to

<!-- formula-not-decoded -->

To conclude, we observe that

<!-- formula-not-decoded -->

This completes the proof.

## D.2 Proof of Reformulation for Model 2

Proof. Rewriting the random parameters ˜ Z as

<!-- formula-not-decoded -->

allow us to rewrite Q j ( Z , θ ) as ⟨ Γ j , Z ⟩ . By [35, Remark 6.6], the robust decoupled performative risk can be expressed as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Applying the exponential smoothing techniques described in [7, Section 2.2], we obtain the following smooth approximation of J η ( θ ) ;

<!-- formula-not-decoded -->

where µ ∈ R ++ is smoothing parameter. Introducing epigraphical variables, we reformulate the objective function as the optimal value of the convex program

<!-- formula-not-decoded -->

The last constraint of (13) is equivalent to ∑ j ∈ [ J ] µe ζ s,j /µ -t s /µ ≤ µ and can be reformulated using the exponential cone:

<!-- formula-not-decoded -->

where the exponential cone K exp is defined as

<!-- formula-not-decoded -->

To complete the proof, we substitute the expressions for ∥ Γ j ∥ 1 ∀ j ∈ [ J ] with

<!-- formula-not-decoded -->

and for Q j ( ˆ Z s , θ ) ∀ s ∈ [ S ] j ∈ [ J ] with

<!-- formula-not-decoded -->

This completes the proof.

## D.3 Proof of Reformulation for Model 3

Proof. By using Definition 1, the robust decoupled performative risk J θ t ( θ ) can be rewritten as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M ( Z ) denotes the space of all probability distributions Q supported on Z satisfying E Q [ ∥ z ∥ 2 2 ] = ∫ Z ∥ z ∥ 2 2 Q ( d z ) &lt; ∞ . This reformulation follows from the law of total probability, where Q s represents the conditional distribution of ˜ z given that the scenario ˆ z s ( θ t ) is realized. Using the Lagrangian, we have

<!-- formula-not-decoded -->

By the minimax theorem [53], which is valid under the assumption that ℓ is upper-semicontinuous and concave in z , and the support set Z is convex, we can exchange the supremum and infimum to obtain:

<!-- formula-not-decoded -->

From the fact that the space M ( Z ) contains all the Dirac distributions supported on Z , we have

<!-- formula-not-decoded -->

Adding a regularized term τλ 2 where τ ∈ R ++ is a positive constant, we have

<!-- formula-not-decoded -->

Therefore, minimizing the right-hand side provides an upper bound on J θ t ( θ ) . Next, we introduce auxiliary variables t s ∀ s ∈ [ S ] , which yields the equivalent formulation for the right hand side of the above inequality

<!-- formula-not-decoded -->

By the definition of conjugate functions, we have

<!-- formula-not-decoded -->

where χ Z denotes the characteristic function of Z . Based on results from [35, Theorem 4.2], [46, Theorem 11.23], and [64, Lemma B.8], the conjugate functions of infimal convolutions and 2 -norm balls is given by

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Substituting this back into the formulation (14), we thus obtain that J θ t ( θ ) is upper bounded by the optimal value of the following convex program:

<!-- formula-not-decoded -->

Combining with outer minimization over θ ∈ Θ completes the proof.

## E Deferred Proofs Related to Convergence

Lemma E.1. Consider the loss function defined in Model 2 and assume Θ is bounded. We define the smoothed loss function

<!-- formula-not-decoded -->

where Z ∼ ˆ P ( θ t ) satisfies and ∥ Z ∥ 2 ≤ k 1 for some constant k 1 &lt; ∞ . Then the gradient ∇ θ ℓ µ ( Z , θ ) is β -Lipschitz in Z for β = Jdk 3 ( k 1 k 2 µ +1 ) for some constants k 2 , k 3 &lt; ∞ defined below.

Proof. Since the coefficients a j ( θ ) , b j ( θ ) , and c j ( θ ) are affine for all j ∈ [ J ] and Θ is bounded, there exist some constants k 2 , k 3 &lt; ∞ such that

- ∥ Γ j ∥ 2 ≤ k 2 for all j ∈ [ J ] ,
- ∥∇ θ i Γ j ∥ 2 ≤ k 3 for all i ∈ [ d ] , j ∈ [ J ] .

By definition, we have the gradient of the smoothed loss (15):

<!-- formula-not-decoded -->

where the softmax weights are

<!-- formula-not-decoded -->

and the gradient of Q j with respect to θ is given by

<!-- formula-not-decoded -->

To prove that ∇ θ ℓ µ is Lipschitz in Z , we examine ∥∇ θ ℓ µ ( Z 1 , θ ) - ∇ θ ℓ µ ( Z 2 , θ ) ∥ 2 . We first decompose the difference and apply the triangle inequality

<!-- formula-not-decoded -->

Next, we bound the terms involved:

- Weight difference:

<!-- formula-not-decoded -->

where (a) comes from the Lipschitz continuity of the softmax function [17], and (b) uses the Cauchy-Schwarz inequality.

- Gradient:

<!-- formula-not-decoded -->

where ∇ θ i Γ j is a matrix whose ( i 1 , i 2 ) -th matrix slice is the gradient of the ( i 1 , i 2 ) -th component of the matrix with respect to θ i .

- Gradient difference:

<!-- formula-not-decoded -->

Substituting the above bounds into (16), we obtain

<!-- formula-not-decoded -->

Thus, the claim follows.

Lemma E.2. Consider the loss function defined in Model 2, and assume that A ⊤ j A j ≻ 0 for all j ∈ [ J ] . Define the smoothed loss function

<!-- formula-not-decoded -->

where Γ j is defined in (11) . Then ℓ reg µ ( θ ) is ρα -strongly convex in θ where

<!-- formula-not-decoded -->

Proof. By definition, the gradient of the smoothed loss (17) is given by

<!-- formula-not-decoded -->

where the softmax weights are defined as

<!-- formula-not-decoded -->

and the gradient of Γ j with respect to θ is given by

<!-- formula-not-decoded -->

Using the product rule and softmax identity:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Corollary E.1. Consider the setting of Model 2, and define the following smoothed loss function

<!-- formula-not-decoded -->

where ℓ µ ( Z , θ ) is the smoothed loss function defined in (15) , and ℓ reg µ ( θ ) is defined in (17) . Suppose that A ⊤ j A j ≻ 0 for all j ∈ [ J ] . Then g ( Z , θ ) is ρα -strongly convex in θ with α = 2 min j ∈ [ J ] λ min ( A ⊤ j A j ) , and the gradient ∇ θ g ( Z , θ ) is β -Lipschitz in Z for β = Jdk 3 ( k 1 k 2 µ +1 ) for some constants k 1 , k 2 , k 3 &lt; ∞ .

Proof. By a standard result in convex analysis, the function ℓ µ ( Z , θ ) is convex in θ , as the logsum-exp operator preserves convexity when applied to a collection of convex functions. Furthermore, Lemma E.2 establishes that ℓ reg µ ( θ ) is ρα -strongly convex, where

<!-- formula-not-decoded -->

As the sum of a convex function and a strongly convex function is strongly convex, g ( Z , θ ) is ρα -strongly convex.

The dependence of g on Z comes only through ℓ µ ( Z , θ ) . Therefore, the Lipschitz continuity of the gradient ∇ θ g ( Z , θ ) with respect to Z follows from Lemma 2, which provides the bound on the Lipschitz constant β . This concludes the proof.

Lemma E.3. Assume that the loss function ℓ ( z , θ ) is concave in z , and that the set Z has a finite diameter D = sup z , z ′ ∈Z ∥ z -z ′ ∥ 2 &lt; ∞ . Define the function

<!-- formula-not-decoded -->

Then the function g is α -strongly convex in ( θ , λ ) where

<!-- formula-not-decoded -->

and the gradient ∇ ( θ ,λ ) g ( v , ( θ , λ )) is ( β +4 D ) -Lipschitz in v .

Proof. Define

Then we have

<!-- formula-not-decoded -->

Notice that the first term is a convex combination of positive definite matrices A ⊤ j A j , so it is positive definite. The second term is a Gram matrix, hence positive semidefinite. Therefore, the minimum eigenvalue of the Hessian is lower bounded by the minimum eigenvalue of the first term and we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ϕ (( θ , λ ) , v , · ) is 2 λ -strongly concave in v , the maximizer

<!-- formula-not-decoded -->

is unique. By Danskin's theorem [6, Section 6.11], the function g is differentiable, and its gradient with respect to ( θ , λ ) is

<!-- formula-not-decoded -->

where z = z (( θ , λ ) , v ) . The Hessian is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α = min( γ, 2 τ ) . This proves strong convexity in ( θ , λ ) .

Before we move to the Lipschitz smoothness, we first prove several inequalities. Fix z 1 , z 2 ∈ Z , we have

<!-- formula-not-decoded -->

where (a) uses the difference of squares, and (b) uses the triangular inequality and the boundness of Z . Next we prove z ∗ (( θ , λ ) , v ) is 1-Lipschitz continuous in v as follows.

<!-- formula-not-decoded -->

where (a) uses strong concavity of ϕ (( θ , λ ) , v , · ) , (b) and (c) come from the first order optimality conditions for z ∗ (( θ , λ ) , v 1 )) and z ∗ (( θ , λ ) , v 2 )) :

<!-- formula-not-decoded -->

and (d) uses Cauchy-Schwarz inequality. Hence,

<!-- formula-not-decoded -->

Finally we show ∇ ( θ ,λ ) g ( v , ( θ , λ )) is Lipschitz in v , consider the gradient difference

<!-- formula-not-decoded -->

where (a) comes from the triangular inequality, (b) uses the β -jointly smoothness of ℓ ( z , θ ) and 19, and (c) uses 20. Hence the gradient ∇ θ ,λ g ( v , ( θ , λ )) is ( β +4 D ) -Lipschitz in v .

## E.1 Proof of Theorem 1

Proof. Let G ( θ t ) denote an optimal solution of (RRMP) at iteration t , i.e.,

<!-- formula-not-decoded -->

where θ t is the current solution and θ t +1 ∈ Θ denotes the optimal solution for next iteration.

We first prove the convergence result for Model 1. Observe that

<!-- formula-not-decoded -->

where g ( Z , θ ) = ℓ ( Z , θ ) + ρL ∥ ( θ , 1) ∥ 2 2 .

Fix η , η ′ ∈ Θ . Since J η ( · ) is ( γ +2 ρL ) -strongly convex, where 2 ρL comes from the strong convexity of the regularization term ρL ∥ ( θ , 1) ∥ 2 2 . we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality follows from the fact that

<!-- formula-not-decoded -->

in view of the first-order optimality condition in Lemma C.1 since G ( η ) ∈ arg min θ ∈ Θ J η ( θ ) . Combining the two inequalities, we obtain

<!-- formula-not-decoded -->

where the second inequality follows from the fact that ( G ( η ) -G ( η ′ )) ⊤ ∇ J η ′ ( G ( η ′ )) ≥ 0 in view of the first-order optimality condition of G ( η ′ ) . Next, we will upper bound (21) using Cauchy-Schwarz inequality, as follows:

<!-- formula-not-decoded -->

Here, (a) follows from the representation of the loss function, while (b) uses the KantorovichRubinstein Lemma C.2 since the loss function is β -jointly smooth from Lemma E.3 and the map ˆ P ( θ ) is ϵ -sensitive. Combining this bound with (21), we get

<!-- formula-not-decoded -->

Our claim (a) is then established by simply performing the change of variables η ← θ and η ′ ← θ ′ .

To prove claim (b), we observe that θ t = G ( θ t -1 ) by the definition of (RRMP), and θ RPS = G ( θ RPS ) by the definition of stability. Applying the result of the claim (a) yields

<!-- formula-not-decoded -->

Setting the right-hand side expression to be at most δ and solving for t completes the proof for Model 1. For Model 2. Observe that

<!-- formula-not-decoded -->

where g ( Z , θ ) = ℓ µ ( Z , θ ) + ℓ reg µ ( θ ) . Here ℓ µ ( Z , θ ) is the smoothed loss function defined in (15), and ℓ reg µ ( θ ) is defined in (17). From Corollary E.1, we know that g ( Z , θ ) is ρα -strongly convex

in θ with α = 2min j ∈ [ J ] λ min ( A ⊤ j A j ) , and the gradient ∇ θ g ( Z , θ ) is β -Lipschitz in Z for β = Jdk 3 ( k 1 k 2 µ +1 ) for some constants k 1 , k 2 , k 3 &lt; ∞ . Using the same techniques as in the proof of Model 1, we can therefore establish the desired result for Model 2.

Finally, for Model 3, one can observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma E.3, we know that g ( Z , θ ) is α -strongly convex in θ with α = min( γ, 2 τ ) , and that its gradient ∇ θ g ( Z , θ ) is ( β +4 D ) -Lipschitz continuous in Z . Hence, by applying the same arguments as in the preceding analysis, we obtain the desired result.

## F Deferred Proofs Related to Sub-optimality Guarantees

## F.1 Proof of Sub-optimality Guarantee for Model 1

Theorem 3. Suppose the loss function in Model 1 is γ -strongly convex in θ (B1) and is β -smooth (B2) . Furthermore, assume that the loss function ℓ ( z , θ ) is L z -Lipschitz in z . Then, the following suboptimality bound holds:

<!-- formula-not-decoded -->

Proof. Since J θ RPS ( θ ) is ( γ +2 ρL ) -strongly convex in θ , we have

<!-- formula-not-decoded -->

since ( θ RPO -θ RPS ) ⊤ ∇ J θ RPS ( θ RPS ) ≥ 0 by the optimality of θ RPS . Using the fact that J θ RPS ( θ RPS ) ≥ J θ RPO ( θ RPO ) , we can further upper bound the right-hand side

<!-- formula-not-decoded -->

where the second inequality holds due the ϵ -sensitivity of the distribution map ˆ P ( · ) and the L z -Lipschitz continuity of the loss function in z . In summary, we obtain

<!-- formula-not-decoded -->

Next, we derive a bound on the suboptimality of the robust performatively stable solution θ RPS . We have

<!-- formula-not-decoded -->

where the first inequality follows from the suboptimality of θ RPO in J θ RPS ( θ ) , the second inequality is from (22), and the last inequality is from (23). This completes the proof.

## F.2 Proof of Sub-optimality Guarantee for Model 2

Theorem 4. Consider the loss function ℓ ( Z , θ ) from Model 2. Suppose that A ⊤ j A j ≻ 0 for all j ∈ [ J ] , and that Y s ⪰ 0 for all s ∈ [ S ] . Additionally, assume that ℓ ( Z , θ ) is L z -Lipschitz in Z .

where

Then, the suboptimality of the stable point θ µ RPS of the smoothed robust objective satisfies:

<!-- formula-not-decoded -->

where α = 2min j ∈ [ J ] λ min ( A ⊤ j A j ) , µ ′ ∈ [0 , 1] is a constant that satisfies µ ≤ µ ′ ∥ θ µ RPS -θ RPO ∥ 2 , J µ θ t ( θ ) is the smoothed robust objective defined in (12) , and θ RPO denotes the minimizer of the original robust objective J θ t ( θ ) .

Proof. By Lemma C.4, we have for any θ ,

<!-- formula-not-decoded -->

Additionally, from Corollary E.1, the function J µ θ t ( θ ) is ρα -strongly convex, with α = 2 min j ∈ [ J ] λ min ( A ⊤ j A j ) . By the definition of strong convexity, we have:

<!-- formula-not-decoded -->

where the inequality follows from the first-order optimality condition of θ µ RPS , i.e.,

<!-- formula-not-decoded -->

We next bound the right-hand side of (26). Using the fact that J θ RPO ( θ RPO ) ≤ J θ µ RPS ( θ µ RPS ) ≤ J µ θ µ RPS ( θ µ RPS ) , we have

<!-- formula-not-decoded -->

where the second inequality comes from (25), and the last inequality holds due to the Lipschitz continuity of ℓ ( Z , θ ) in Z . Substituting (27) into the strong convexity inequality (26) gives:

<!-- formula-not-decoded -->

Assuming µ ≤ µ ′ ∥ θ µ RPS -θ RPO ∥ 2 where µ ′ ∈ [0 , 1] , we can divide both sides by ∥ θ µ RPS -θ RPO ∥ 2 to obtain:

<!-- formula-not-decoded -->

Finally, we derive a bound on the suboptimality of the robust performatively stable solution θ µ RPS :

<!-- formula-not-decoded -->

where the first inequality uses suboptimality of θ RPO in J µ θ µ RPS ( θ ) , the second follows from (27), and the last inequality uses the bound in (29). This concludes the proof.

## F.3 Proof of Sub-optimality Guarantee for Model 3

Theorem 5. Suppose that Z has a finite diameter D = sup z , z ′ ∈Z ∥ z -z ′ ∥ 2 &lt; ∞ and Θ is bounded. Assume that the loss function ℓ ( z , θ ) in Theorem (3) is L θ Lipschitz continuous in θ and L z -Lipschitz in z . Let J τ θ t ( θ ) denote the objective defined in problem (7) , and let θ τ RPS denote a robust performative stable point under this objective. Then the following suboptimality bound holds:

<!-- formula-not-decoded -->

where α := min( γ, 2 τ ) is the strong convexity parameter, and λ is the upper bound on the optimal λ as given in Lemma F.1.

Proof. From Lemma E.3, the objective function J τ θ t ( θ ) is α -strongly convex in θ = ( θ , λ ) with α = min( γ, 2 τ ) . Applying the strong convexity inequality, we obtain:

<!-- formula-not-decoded -->

where the inequality follows from ( θ τ RPO -θ τ RPS ) ⊤ ∇ J τ θ τ RPS ( θ τ RPS ) ≥ 0 by the optimality of θ τ RPS . Using the fact that J τ θ τ RPO ( θ τ RPO ) ≤ J τ θ τ RPS ( θ τ RPS ) , we can further upper bound the right-hand side

<!-- formula-not-decoded -->

where the second inequality holds due the ϵ -sensitivity of the distribution map ˆ P ( · ) and the L z -Lipschitz continuity of the loss function in z . In summary, we obtain

<!-- formula-not-decoded -->

Now consider the suboptimality decomposition:

<!-- formula-not-decoded -->

We now bound each term:

1. First term . Before we start, we first provide a bound on J τ θ τ RPS ( θ τ RPS ) -J τ θ τ RPS ( θ τ RPO ) . From Lemma (F.2), the function f ( Z , θ ) is Lipschitz in θ . Hence:

<!-- formula-not-decoded -->

where λ is the upper bound on the optimal λ defined in (38). Now we provide a bound on the first term, using the decomposition:

<!-- formula-not-decoded -->

where the first inequality comes from (31) and (34), the last inequality holds because of (32).

2. Second term. Using the fact that J τ η ( θ ) augments J η ( θ ) by τλ 2 :

<!-- formula-not-decoded -->

Substituting these into (33) , we obtain

<!-- formula-not-decoded -->

This concludes the proof.

Lemma F.1. Suppose that Z has a finite diameter D = sup z , z ′ ∈Z ∥ z -z ′ ∥ 2 &lt; ∞ and Θ is bounded. Assume that the loss function ℓ ( z , θ ) is L θ Lipschitz continuous in θ and L z -Lipschitz in z . Then there exist ℓ, ¯ ℓ ∈ R such that for all ( z , θ ) ∈ Z × Θ ,

<!-- formula-not-decoded -->

Consider the following univariate minimization problem

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the problem (37) admits a minimizer λ ∗ ≤ λ , where

<!-- formula-not-decoded -->

Proof. First, we observe that for fixed ( θ , ˆ z s ) , the function ℓ c ( z , θ , λ ) is convex and lower semicontinuous in λ , as it is the supremum of functions affine in λ . Since lower semi-continuity is preserved under expectation over a discrete distribution ˆ P ( θ t ) , the objective function in (37) is convex and lower semi-continuous in λ .

Next, we establish a lower bound. Note that for all λ ≥ 0 ,

<!-- formula-not-decoded -->

since the supremum is attained at z = ˆ z s . Hence, we can bound the objective from below:

<!-- formula-not-decoded -->

Thus, the infimum of problem (37) is attained for some λ ∗ ∈ [0 , + ∞ ) . By optimality of λ ∗ , we then have:

<!-- formula-not-decoded -->

where the second inequality comes from evaluating the objective at λ = 0 . Rearranging the inequality, we obtain:

<!-- formula-not-decoded -->

Solving the quadratic inequality yields the upper bound

<!-- formula-not-decoded -->

where

This concludes the proof.

Lemma F.2. Suppose Z has a finite diameter D = sup z , z ′ ∈Z ∥ z -z ′ ∥ 2 &lt; ∞ and the loss function ℓ ( z , θ ) is L θ Lipschitz in θ . For ˆ z s ∈ Z , define the function

<!-- formula-not-decoded -->

where θ := ( θ , λ ) ∈ Θ × R + := Θ , and

<!-- formula-not-decoded -->

Then for any θ , θ ′ ∈ Θ , the function f satisfies the Lipschitz bound:

<!-- formula-not-decoded -->

Proof. We start with the absolute difference between the two evaluations of f :

<!-- formula-not-decoded -->

Next, we bound the first term; the second is symmetric. Let

<!-- formula-not-decoded -->

Using the inequality

<!-- formula-not-decoded -->

we obtain

<!-- formula-not-decoded -->

For the first term, since ℓ is L θ -Lipschitz in θ :

<!-- formula-not-decoded -->

For the second term, observe that ∥ z -ˆ z s ∥ 2 ≤ D implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By symmetry, the same bound holds for f (ˆ z s , θ ′ ) -f (ˆ z s , θ ) . This concludes the proof.

## F.4 Proof of Theorem 2

The proof of Theorem 2 follows directly from the results of Theorem 3, Theorem 4 and Theorem 5.

## G Experiment Details

## G.1 Strategic Classification

Following [40, 32], we assume that individuals have linear utilities u ( θ , ˜ x ) = -θ ⊤ ˜ x and quadratic costs c ( ˜ x ′ , ˜ x ) = -1 2 ϵ ∥ ˜ x ′ -˜ x ∥ 2 2 , where ϵ is a positive constant regulating the cost of altering features and thus the sensitivity of the distribution map. In other words, individuals aim to minimize their assigned probability of default but are unable to change their true outcome ˜ y . We select S ⊆ [ P -1] strategic features, such as the number of open credit lines. Each time an individual manipulates their strategic features as depicted in [40, Section 5], the best response for an individual results in the update

<!-- formula-not-decoded -->

Thus, where ˜ x ′ S , ˜ x S , θ S ∈ R | S | .

Robust type-1. Consider the 1-Wasserstein ball, it follows from Theorem (1), at each time t , we can solve the following Tikhonov regularization problem

<!-- formula-not-decoded -->

Robust type-2. Consider the 2-Wasserstein ball, it follows from Proposition (1), at each time t , we can solve the following problem

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we introduce auxiliary variables t s ∀ s ∈ [ S ] and combine with outer minimization over θ ∈ Θ , which yield the equivalent formulation for the right hand side of the above inequality

<!-- formula-not-decoded -->

To handle the last constraints, we discretize over some finite set S α , i.e.,

<!-- formula-not-decoded -->

For the experiment we choose S α = {-0 . 9 , -0 . 8 , . . . , -0 . 1 } .

Proposition 1. Consider the logistic loss

<!-- formula-not-decoded -->

with z = ( x , ˆ y ) and θ ∈ R d . Consider the 2-Wasserstein ball, where the ground cost c is given by the Euclidean norm on R d . Assume the support set X is convex and closed, then we have

<!-- formula-not-decoded -->

where h ( α ) = ( α +1)log(1 + α ) -α log( -α ) .

Proof. We follow the argument in the proof of Theorem (3). For fixed θ t , we have

<!-- formula-not-decoded -->

where ˆ z = ˆ x ˆ y s . Note that label ˆ y s is fixed, we can simplify the inner supremum

<!-- formula-not-decoded -->

Next, substitute the logistic loss

<!-- formula-not-decoded -->

Using the Fenchel conjugate dual formulation [45] of the logistic loss:

<!-- formula-not-decoded -->

we rewrite the inner supremum as:

<!-- formula-not-decoded -->

We now compute the inner supremum over x . For fixed α , ˆ y s , and θ , the expression

<!-- formula-not-decoded -->

is a concave quadratic in x . The optimum is achieved at:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, the robust objective becomes:

<!-- formula-not-decoded -->

Alternative minimization under KL divergence ambiguity set (AMKL) Consider the Kullback-Leibler (KL) divergence ambiguity set. According to [58, Proposition 3.1], the distributionally robust optimization problem under KL divergence can be reformulated as:

<!-- formula-not-decoded -->

where ρ &gt; 0 is the radius of the ambiguity set. To solve this optimization problem, [58] propose an alternating minimization approach, which we refer to as AMKL. The procedure alternates between the following two steps:

1. θ -step: Fix the robustness parameter µ , and minimize the objective with respect to θ .
2. µ -step: Fix the model parameter θ , and minimize the objective with respect to µ .

This iterative process is repeated until convergence. In the θ -step, we apply the repeated risk minimization algorithm, following the suggestion in [58] that this subproblem can be addressed using any suitable performative risk minimization algorithm.

## G.2 Impact of the Robust Parameter ρ

We investigate the impact of the robust parameter ρ on out-of-sample performance. Specifically, we consider the revenue management problem described in section (4.2) under different quantities of perishable products, q ∈ { 200 , 300 , 500 } .

Figure 4: Out-of-sample performance as a function of the robust parameter ρ and estimated on the basis of 100 simulations.

<!-- image -->

Figure 4 illustrates the optimal mean profit as a function of ρ , averaged over 100 independent simulation runs. We observe that the out-of-sample performance improves as ρ increases up to a critical Wasserstein radius ρ ∗ , beyond which it begins to deteriorate. Notably, the robust model outperforms the non-robust counterpart over a wide range of ρ values. This pattern was consistently observed across all simulation settings and provides an empirical justification for adopting a distributionally robust approach.

Substituting x ∗ back in yields:

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our claims are supported by both theoretical analysis and empirical results. We present experiments on the standard strategic classification benchmark, as well as two decision-making problems in revenue management and demand response portfolio optimization.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations are discussed in the conclusion section.

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

Justification: All theoretical results have been proven rigorously with assumptions clearly stated.

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

Justification: The main algorithm and the formulations have been clearly described. The details for experiments are included in the main text and appendix.

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

Justification: The data is from a public dataset [25] or generated synthetically (see details in Section 4). The code is available upon request.

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

Justification: Experimental details have been discussed in the Experiments section as well as in the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide box plots for the out-of-sample performances.

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

Justification: Computational resources are provided in the Experiments section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have read and acknowledged the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss societal impacts in the conclusion. Our framework extends the scope of performative prediction beyond its original focus, enabling its application to a wider range of decision-making problems. In high-stakes settings, adopting a distributionally robust optimization perspective allows our approach to prioritize safe and reliable deployment in the presence of uncertainty and potential adversarial conditions.

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

Justification: No data or models are released.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The creators of the existing dataset are properly credited in Section 4 and in the reference [25]. We have cited the benchmark algorithms in the Experiments section.

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

Justification: The paper does not introduce new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

Justification: The paper does not use LLMs for important, original, or non-standard component of the core methods in this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
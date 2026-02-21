## Maximizing the Value of Predictions in Control: Accuracy Is Not Enough

## Yiheng Lin

California Institute of Technology Pasadena, CA, USA yihengl@caltech.edu

## Zaiwei Chen

Purdue University West Lafayette, IN, USA chen5252@purdue.edu

## Christopher Yeh

California Institute of Technology Pasadena, CA, USA cyeh@caltech.edu

## Adam Wierman

California Institute of Technology Pasadena, CA, USA adamw@caltech.edu

## Abstract

We study the value of stochastic predictions in online optimal control with random disturbances. Prior work provides performance guarantees based on prediction error but ignores the stochastic dependence between predictions and disturbances. We introduce a general framework modeling their joint distribution and define 'prediction power' as the control cost improvement from the optimal use of predictions compared to ignoring the predictions. In the time-varying Linear Quadratic Regulator (LQR) setting, we derive a closed-form expression for prediction power and discuss its mismatch with prediction accuracy and connection with online policy optimization. To extend beyond LQR, we study general dynamics and costs. We establish a lower bound on prediction power under two sufficient conditions that generalize the properties of the LQR setting, characterizing the fundamental benefit of incorporating stochastic predictions. We apply this lower bound to nonquadratic costs and show that even weakly dependent predictions yield significant performance gains.

## 1 Introduction

Understanding the benefits of predictions in control has received significant attention recently [1, 2, 3, 4, 5]. In this work, we study a class of discrete-time online optimal control problems in general time-varying systems, where random disturbances W t affect state transitions. The agent leverages a prediction vector containing information about future disturbances to minimize the expected total cost over a finite horizon T . To study the impact of using predictions, a fundamental question is how to model disturbances and their relationship to predictions. Prior works adopt different modeling approaches [1, 5, 6], each with distinct strengths and limitations.

A common paradigm assumes perfect predictions over a finite horizon k , yielding an elegant characterization of 'prediction power' that improves with larger k . Under this model, predictions exactly reveal future disturbances W t , . . . , W t + k -1 . [1] shows how prediction power grows with k in the LQR setting, and subsequent work extends this result to time-varying systems [3]. As a result, the marginal benefit of one additional prediction decays exponentially with k , offering insight into how to select k . However, longer-horizon predictions are more costly and less accurate, and real-world predictions are rarely perfect [7], making this idealized setting challenging in practice.

A natural extension of accurate predictions is to consider bounded prediction errors, which better captures practical challenges. Specifically, prediction errors measure the distance between the predicted and actual disturbances, and the resulting cost bounds depend on these errors [2, 5, 4]. This extension recovers the perfect predictions setting when errors shrink to zero. However, it can be overly pessimistic because, for any predictor, the same performance bound must also apply to an adversary that generates the worst prediction sequence to penalize the predictive policy subject to the same error bound. It overlooks stochastic dependencies between predictions and disturbances that can be valuable for improving control costs.

In this work, we propose a general stochastic model that captures the distributional dependencies between predictions and disturbances, without restricting prediction targets, horizon length, or requiring strict error bounds. Compared with previous stochastic methods [6, 8], our approach further relaxes problem-specific assumptions and directly focuses on the incremental benefit of predictions. Such benefits can be subtle-often overlooked by classical metrics like regret or competitive ratio. To capture them, we define prediction power as the improvement in expected total cost when predictions are fully exploited, which builds on and generalizes the notion from [1]. Our framework thus characterizes when and why predictions significantly boost online control performance.

Contributions. We introduce a general stochastic model (Definition 2.1) that describes how disturbances relate to all candidate predictors. We then define prediction power (Definition 2.4) which quantifies the incremental control-cost improvement gained by fully leveraging these predictions. To illustrate this concept, we derive an exact expression for prediction power in the benchmark setting of time-varying linear quadratic regulator (LQR) control (Theorem 3.2). Using this closed-form formula, we provide examples (e.g., Example 3.3) that illustrate why analyzing prediction accuracy is insufficient-improving prediction accuracy may not always improve prediction power. Finally, we demonstrate the connection between prediction power and online policy optimization (Example 3.4), highlighting how practical algorithms can attain (a portion of) the maximum potential.

We extend our analysis of prediction power beyond the LQR setting. This generalization poses significant challenges due to the lack of closed-form expressions for the optimal policy. Building on insights from the LQR analysis, we identify two key structural conditions: a quadratic growth condition on the optimal Q-function (Condition 4.1) and a positive semi-definite covariance condition on the optimal policy's actions (Condition 4.2). These conditions are sufficient to derive a general lower bound on prediction power, formalized in Theorem 4.3. We apply this result to the setting of time-varying linear dynamics with non-quadratic cost functions. Under assumptions on costs and on the joint distribution of predictions and disturbances, we establish a lower bound on prediction power (Theorem 4.8), demonstrating that even weak predictions can yield strict performance gains.

Related Literature. Our work is closely related to the line of works on using predictions in online control. Our prediction power is inspired by [1], which defines the prediction power as the maximum control cost improvement enabled by k steps of accurate predictions in the time-invariant LQR setting. Compared with [1], we extend the notion of prediction power to allow general dependencies between predictions and disturbances, and we consider more general dynamics/costs (Section 4). Rather than focusing on the prediction power, many works study the power of a certain policy class such as MPC [2, 3, 5, 4], Averaging Fixed Horizon Control [6, 8], Receding Horizon Gradient Descent [9, 10], and others [11]. While one can say the power of (generalized) MPC equals the prediction power in the LQR setting [1] (Section 3), they are not the same in general (see Appendix C.2).

Our work is, in part, motivated by both empirical and theoretical findings in the decision-focused learning (DFL) (also referred to as 'predict-then-optimize') literature that prediction models with the same prediction accuracy may have very different control costs (see [12] for a recent survey). Research on DFL typically considers predictions given as point estimates of some uncertain input to decision-makers modeled as optimization problems, such as stochastic optimization [13], linear programs [14], or model predictive control [15], although more recent works have started exploring other forms of predictions such as prediction sets [16, 17]. In contrast, our work does not require any particular form of decision-maker; instead, our main result characterizes the benefit of optimally leveraging predictions, for whatever form an optimal controller may take. Whereas DFL aims to design procedures for training prediction models that reduce downstream control costs, our work studies a more fundamental question about how much performance gain is achievable with better predictions. Another major difference between our work and typical DFL literature is that our controller must decide control actions sequentially in a dynamical system, where the predictions are

revealed in an online process. The controller can also build its knowledge from past disturbances and predictions to infer future disturbances or the optimal control action. Thus, our online setting presents unique challenges compared with making a one-shot decision for a classic optimization problem.

## 2 Problem Setting

We consider a finite-horizon discrete-time optimal control problem with time-varying dynamics and cost functions, where state transitions are subject to random disturbances:

```
Control dynamics: X t +1 = f t ( X t , U t ; W t ) , 0 ≤ t < T, with the initial state X 0 = x 0 ; Stage cost: h t ( X t , U t ) , 0 ≤ t < T, and terminal cost: h T ( X T ) . (1)
```

At each time step t , we let X t denote the system state and U t denote the control action chosen by an agent. The function f t : R n × R m × R k → R n defines how the next state X t +1 depends on the current state X t , the control action U t , and the random disturbance W t . The agent incurs a stage cost h t ( X t , U t ) at each time step t &lt; T and a terminal cost h T ( X T ) at the final time step T . At each time step t , the controller observes the past disturbance W t -1 and a (possibly random) prediction vector V t ( θ ) ∈ R d before selecting a control action U t , where θ is a parameter of the predictor generating the prediction. We formally define the concept of predictions and the parameter θ in the following.

Definition 2.1 (Predictions) . At each time step t , the predictor with parameter θ ∈ Θ provides a prediction V t ( θ ) , where Θ denotes the set of all possible predictor parameters. The predictions { V 0: T -1 ( θ ) } θ ∈ Θ and the disturbances W 0: T -1 live in the same probability space.

We do not require the prediction V t ( θ ) to have any specific form as a function of θ . The parameter θ is primarily used for distinguishing different predictor candidates.

Compared with previous works [3, 18] that assume predictions targeting specific disturbances, Definition 2.1 focuses on the stochastic relationship between predictions and system uncertainties, yielding a unified framework for comparing different forms of prediction based on their effectiveness for control-even if their precise nature is unknown. Because predictions and disturbances share the same probability space, we can compare prediction sequences V 0: T -1 ( θ ) and V 0: T -1 ( θ ′ ) , generated by different predictors with parameters θ and θ ′ .

Observe that the disturbances W 0: T -1 and predictions in Definition 2.1 do not depend on the current state or past trajectory, reflecting their exogenous nature. For example, consider the problem of quadcopter control in windy conditions [19]. In this case, the wind disturbances are not influenced by the quadcopter's state or control inputs. Under this causal relationship, we define the problem instance as Ξ = ( W 0: T -1 , { V 0: T -1 ( θ ) } θ ∈ Θ ) , and make the following assumption.

Assumption 2.2. The problem instance Ξ is sampled from the distribution of problem instances before the control process starts, i.e., it will not be affected by the controller's states/actions.

Let ξ = ( w 0: T -1 , { v 0: T -1 ( θ ) } θ ∈ Θ ) denote a realization of the problem instance, including disturbances and all parameterized predictions. Under Assumption 2.2, Ξ is viewed as realized to ξ before control begins, although the agent observes each disturbance and prediction step by step. Similar assumptions about oblivious environments or predictions appear in online optimization [20, 21], ensuring that future disturbances or predictions will not be affected by past states or actions. Hence, for a fixed predictor parameter θ , we define a predictive policy as a mapping from the current state and past disturbances and predictions to a control action.

Definition 2.3 (Predictive policy) . Consider a fixed predictor parameter θ . For each time step t , let I t ( θ ) := ( W 0: t -1 , V 0: t ( θ )) denote the history of past disturbances and predictions, and let F t ( θ ) := σ ( I t ( θ )) 1 . A predictive policy that applies to the predictor with parameter θ is a sequence of functions π 0: T -1 , where π t maps a state/history pair to a control action.

Given a fixed predictive policy sequence π = π 0: T -1 for a predictor parameter θ , we evaluate its performance via the expected total cost over Ξ : J π ( θ ) := E [ ∑ T -1 t =0 h t ( X t , U t ) + h T ( X T )] , where X 0 = x 0 , X t +1 = f t ( X t , U t ; W t ) , U t = π t ( X t ; I t ( θ )) , for t = 0 , . . . , T -1 . The optimal cost under θ is defined as J ∗ ( θ ) = min π J π ( θ ) , where the minimum is over all predictive policies that use the predictor parameter θ .

1 For any random variable Y , we use σ ( Y ) to denote the σ -algebra it generates.

̸

Following [1], we define prediction power by comparing against a baseline that provides minimal information (e.g., no prediction). Without loss of generality, let 0 ∈ Θ be the baseline predictor parameter so that any θ = 0 provides at least as much information as 0 , i.e., F t ( θ ) ⊇ F t ( 0 ) . Based on this baseline, we define prediction power as the maximum possible cost improvement achieved by using predictions under θ relative to the baseline, formally stated in Definition 2.4.

Definition 2.4 (Prediction power) . For a predictor with parameter θ , its prediction power in the optimal control problem (1) is P ( θ ) := J ∗ ( 0 ) -J ∗ ( θ ) .

Our definition of prediction power is based on the optimal control policy under a given predictor parameter and, therefore, is independent of any specific policy class. Many previous works have considered prediction-enabled improvement within a specific policy class [9, 10, 6], where they focus on changes in J π ( θ ) rather than J ∗ ( θ ) . In other works, policies include parameters that can be tuned to perform optimally under a specific predictor; that is, min π ∈ a policy class J π ( θ ) . While these approaches are useful in specific application scenarios, our definition, based on the general optimal policy, is more universal because: (1) imposing policy class constraints may lead to performance loss, and (2) the extent of improvement can depend on policy design and parameterization, which shifts the focus away from valuing predictions themselves.

When the baseline predictor 0 is no prediction, the prediction power is guaranteed to be non-negative. This is because the optimal policy for the no-prediction case is also a predictive policy when predictions are available, i.e., the controller can simply ignore the predictions. But the prediction power could be zero, for example, when the predictions are independent of the disturbances. Further, the prediction power does not increase if one replaces the original prediction V t ( θ ) by any function of the history I t ( θ ) , because this additional step cannot increase the information available at time step t .

Throughout this paper, we use ¯ π = ¯ π 0: T -1 and π θ = π θ 0: T -1 to denote the optimal policy for the predictor with parameter 0 and θ , respectively. In other words, J ¯ π ( 0 ) = J ∗ ( 0 ) and J π θ ( θ ) = J ∗ ( θ ) . To compare the policies π θ and ¯ π , we introduce the instance-dependent Q function , inspired by the Q function in the study of Markov decision processes (MDPs). For a given state-action pair ( x, u ) and problem instance ξ , the instance-dependent Q function for a policy π evaluates the remaining cost incurred by taking action u from state x and then following policy π for all future time steps. Recall that the history I τ ( θ ) contains all past disturbances and predictions that are observed until a time step τ . Using ι τ ( θ ) to denote the realization of I τ ( θ ) , the instance-dependent Q function is defined as

<!-- formula-not-decoded -->

subject to the constraints that x τ +1 = f τ ( x τ , u τ ; w τ ) for t ≤ τ &lt; T and u τ = π θ τ ( x τ ; ι τ ( θ )) for t &lt; τ &lt; T . Recall that ξ is the realization of problem instance Ξ that contains all disturbances and predictions for all time steps. Thus, the disturbance w τ and the history ι τ ( θ ) in (2) are decided by the problem instance ξ , which is an input to Q π θ t . Similarly, we define Q ¯ π t ( x, u ; ξ ) by replacing θ with 0 and π θ with ¯ π in (2). Importantly, our instance-dependent Q function is different from the classical definition of the Q function for MDPs or reinforcement learning (RL), where it is the expectation of the cost to go. The instance-dependent Q function denotes the actual remaining cost, which is a σ (Ξ) -measurable random variable . The classic definition of the Q function can be recovered by taking the conditional expectation, i.e., E [ Q π θ t ( x, u ; Ξ) | I t ( θ ) = ι t ( θ ) ] . It is worth noting that our instance-dependent Q function is about the cost instead of the reward , so lower values are better.

With this definition of the instance-dependent Q function, the policies ¯ π and π θ can be expressed as recursively minimizing the corresponding expected Q functions conditioned on the available history. Starting with C π θ T ( x ; ξ ) = h T ( x ) , for time step t = T -1 , . . . , 0 , we have

<!-- formula-not-decoded -->

Similar recursive relationships also define the optimal policy ¯ π for the baseline predictions, and we only need to replace θ with 0 and π θ with ¯ π in the above equations. The recursive equations in (3) can be viewed as a generalization of the classical Bellman optimality equation for general MDPs.

<!-- formula-not-decoded -->

## 3 LTV Dynamics with Quadratic Costs

We first characterize the prediction power (Definition 2.4) in a linear time-varying (LTV) dynamical system with quadratic costs, where the dynamics and costs are given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Q 0: T -1 , R 0: T -1 , and P T are symmetric positive definite. The classic linear quadratic regulator (LQR) problem, along with its time-varying variant that we consider, has been used widely as a benchmark setting in the learning-for-control literature. It also serves as a good approximation of nonlinear systems near equilibrium points, making it amenable to standard analytical tools. We begin by defining key quantities that will be useful for stating the main results in this section. For t = T -1 , . . . , 0 , we define the matrices H t , P t , and K t recursively according to

Moreover, we define the transition matrix Φ t 2 ,t 1 as Φ t 2 ,t 1 = I if t 2 ≤ t 1 and

The matrix K t is the feedback gain matrix in the optimal policy, and P t is the matrix that defines the quadratic term in the optimal cost-to-go function. To simplify notation, we define the shorthands W θ τ | t := E [ W τ | I t ( θ )] and w θ τ | t := E [ W τ | I t ( θ ) = ι t ( θ )] .

<!-- formula-not-decoded -->

Proposition 3.1. In the case of LTV dynamics with quadratic costs, the conditional expectation of the optimal Q function E [ Q π θ t ( x, u ; Ξ) | I t ( θ ) = ι t ( θ ) ] can be expressed as

<!-- formula-not-decoded -->

where ψ π θ t ( x ; ι t ( θ )) is a function of the state x and the history ι t ( θ ) that does not depend on the control action u . Here, ¯ u θ t ( ι t ( θ )) := -( R t + B ⊤ t P t +1 B t ) -1 B ⊤ t ∑ T -1 τ = t Φ ⊤ τ +1 ,t +1 P τ +1 w θ τ | t . And the optimal policy can be expressed as π θ t ( x ; ι t ( θ )) = -K t x + ¯ u θ t ( ι t ( θ )) .

We derive the closed-form expressions in Proposition 3.1 by induction following the backward recursive equations in (3); the full proof is deferred to Appendix A.1. With these expressions, we obtain a closed-form expression of the prediction power. We defer its proof to Appendix A.2.

Theorem 3.2. In the case of LTV dynamics with quadratic costs, the prediction power of the predictor with parameter θ is P ( θ ) = ∑ T -1 t =0 Tr { ( R t + B ⊤ t P t +1 B t ) E [ Cov [ ¯ u θ t ( I t ( θ )) | F t ( 0 ) ]]} .

While the optimal policy in Proposition 3.1 is restricted to the LQR case, we can interpret the optimal policy as planning according to the conditional expectation following the idea of model predictive control (MPC) [1], which is easier to generalize. The agent needs to solve an optimization problem and re-plan at every time step. At time step t , the agent solves

<!-- formula-not-decoded -->

subject to the constraints that X τ +1 = f τ ( X τ , u τ ; W τ ) for τ ≥ t and X t = x . Then, the agent commits to the first entry u t | t of the optimal solution as π θ t ( x ; ι t ( θ )) . In the LQR setting, we can further simplify it to be planning according to w θ τ | t , i.e.,

<!-- formula-not-decoded -->

subject to the constraints that x τ +1 = f τ ( x τ , u τ ; w θ τ | t ) for τ ≥ t and x t = x . We defer a detailed discussion and proof to Appendix A.3.

The MPC forms of the optimal policy in (7) extends the result in [1], which shows that MPC is the optimal predictive policy under the accurate prediction model in time-invariant LQR. When

the predictions are inaccurate, and the system is time-varying, MPC is still optimal if we solve the predictive optimal control problem in expectation (7).

Evaluation. One can follow the expressions in Theorem 3.2 to evaluate the prediction power, but it requires taking the conditional covariance on the top of conditional expectations ( ¯ u θ t ( ι t ( θ )) in Proposition 3.1). To avoid this recursive structure, an alternative way is to first construct the surrogate optimal action , which is defined as

<!-- formula-not-decoded -->

We call ¯ u ∗ t (Ξ) the surrogate-optimal action, because it is the optimal action that an agent should take with the oracle knowledge of all future disturbances at time t . The prediction power in Theorem 3.2 can be expressed as E [ Cov [¯ u ∗ t (Ξ) | I t ( 0 )]] -E [ Cov [¯ u ∗ t (Ξ) | I t ( θ )]] . Following this decomposition, we propose an evaluation approach that constructs ¯ u ∗ t (Ξ) before estimating its conditional covariance with respect to I t ( θ ) and I t ( 0 ) separately. We defer the details to Appendix A.4.

## 3.1 Prediction Power = Accuracy

̸

As Proposition 3.1 suggests, one way to implement the optimal policy is to predict each of the future disturbances W t : T -1 and generate the estimations w θ ( t : T -1) | t (i.e., the conditional expectation of future disturbances given the history ι t ( θ ) at time step t ) in deciding the action at time step t . However, two controllers with the same estimation error (as measured by mean squared error (MSE)) can have very different control costs. Because of this reason, the control cost bounds depend on the estimation errors in previous works [5, 2, 4] must be loose, so one cannot rely on them to infer or compare the values of different predictors.

To illustrate this point, we provide an example where the prediction power can change significantly when the prediction accuracy does not change.

Example 3.3. Consider the time-invariant LQR setting, i.e. , assume A t = A,B t = B,Q t = Q,R t = R for all t and P T = P is the solution to the Discrete-time Riccati Equation (DARE) in (4) . Suppose the disturbance is sampled W t i.i.d. ∼ N (0 , I ) at every time step t , where we use the notation I to denote the identity matrix. Let ρ ∈ [0 , √ 2 2 ] be a fixed coefficient. We construct a class of predictors from the disturbances { W t } by applying the affine transformation V t ( θ ) := ρθW t + ϵ t ( ρ, θ ) for θ ∈ R 2 × 2 that satisfies θθ ⊤ ⪯ 1 2 I , where the random noise ϵ t ( ρ, θ ) is independently sampled from a Gaussian distribution N (0 , I -ρ 2 θθ ⊤ ) .

We can construct θ such that V t ( θ ) and V t ( I ) achieve the same mean-square error (MSE) when predicting each individual entry of W t , yet P ( θ ) &gt; P ( I ) . To construct θ , note that ( W t , V t ( θ )) satisfies E [ W t | V t ( θ )] = ρθ ⊤ V t and Cov [ W t | V t ( θ )] = I -ρ 2 θ ⊤ θ . Thus, we can change θ without affecting the MSE of predicting each individual entry as long as the diagonal entries of θ ⊤ θ remain the same. However, by Theorem 3.2, we know the prediction power is equal to ρ 2 T · Tr { θ ⊤ θPHP } , where H = B ( R + B ⊤ PB ) -1 B ⊤ . Thus, the off-diagonal entries of θ ⊤ θ can also affect the value of Tr { θ ⊤ θPHP } . We instantiate this example with a 2-D double-integrator dynamical system in Appendix A.5.1: the predictors with parameters I and θ share the same MSE but their prediction powers are significantly different. 2

Example 3.3 shows how prediction power can vary even when the accuracy of predicting each entry of the disturbance W t remains the same, where the construction leverages the covariance between the predictions for different entries of W t . While the construction in Example 3.3 requires n ≥ 2 , we also provide an example with n = 1 and multiple steps of predictions in Appendix A.5. From these examples, it is clear that one should not use the MSEs of predicting future disturbances to infer the prediction power. The intuition behind Example 3.3 does not require a very specific choice of the function V t ( θ ) : The underlying idea is that the MSEs of estimating each individual entry of the multi-dimensional disturbance W t are insufficient to decide the prediction power. The off-diagonal entries of the covariance matrix Cov [ W t | V t ( θ )] also matter, and their impact on the prediction power depends on the dynamics ( A,B,Q,R ) . Therefore, a general accuracy metric (like MSE) that is unaware of ( A,B,Q,R ) can be misaligned with the prediction power. The mismatch also relates to the findings in the decision-focused learning literature discussed in the related work section.

2 The simulation code for all examples (Examples 3.3, 3.4, and A.2) can be found at https://github.com/ yihenglin97/Prediction-Power .

Figure 1: Example 3.4: Prediction V t (1) is available. Candidate policy: u t = -Kx t +Υ t v t (1) .

<!-- image -->

<!-- image -->

Figure 2: Example 3.4: Prediction V t (2) is available. Candidate policy: u t = -Kx t +Υ t v t (2) .

## 3.2 Prediction Power and Online Policy Optimization

The closed-form expression of the prediction power, presented in Theorem 3.2, characterizes the maximum potential of using a given prediction sequence V 0: T -1 ( θ ) . Here, we draw a connection between prediction power and online policy optimization [22, 23], which aims to learn and adapt the optimal control policy within a certain policy class over time: the prediction power serves as an improvement upper bound of applying online policy optimization to predictive policies, although it is generally unattainable. In the following example, we demonstrate this bound using M-GAPS [24], a state-of-the-art online policy optimization algorithm.

Example 3.4. We construct two scenarios under the same setting as Example 3.3. First, when the prediction is V t (1) := ρW t + ϵ t ( ρ, I ) , we let M-GAPS adapt within the candidate policy class u t = -Kx t +Υ t v t (1) , where Υ t ∈ R 1 × 2 is the policy parameter. 3 Here, the optimal predictive policy π 1 is contained in the candidate policy class. We plot the average cost improvement of M-GAPS and π 1 compared against the optimal no-prediction policy ¯ π in Figure 1. From the initialization Υ 0 = 0 , M-GAPS tunes Υ t to improve the average cost over time, and the average cost improvement against ¯ π converges towards the averaged prediction power P (1) /T .

In the second scenario, we change the prediction to apply M-GAPS to V t (2) := V t +1 (1) (i.e., the same prediction as before is made available 1-step ahead). We let M-GAPS adapt within the same candidate policy class u t = -Kx t +Υ t v t (2) , where the policy parameter is still Υ t ∈ R 1 × 2 . Unlike the first scenario, the optimal predictive policy π 2 is not contained in the candidate policy class, because π 2 uses both v t (2) and v t -1 (2) to decide the action. As a result, M-GAPS cannot achieve an improvement that is close to the averaged prediction power P (2) /T , which is achievable by π 2 (see Figure 2).

The details of Example 3.4 are provided in Appendix A.6. It demonstrates how prediction power serves as an upper bound for the cost improvement achieved by online policy optimization. Conversely, online policy optimization offers practical tools to achieve (part of) the potential benefit of using predictions without requiring explicit knowledge or estimation of the joint distribution between predictions and true disturbances.

## 4 Characterizing Prediction Power: A General-Purpose Theorem

In this section, we provide a theorem to characterize the prediction power P ( θ ) within the general problem setting introduced in Section 2. Our result relies on two conditions about a growth property of the expected Q function under π θ and the covariance of the optimal policy's action when conditioned on the σ -algebra F t ( 0 ) of the baseline. We state these conditions and provide intuitive explanations.

3 Intuitively, M-GAPS works by taking the gradient of the cost function with respect to the policy parameters at every time step, and it takes gradient steps to update Υ t , allowing it to converge towards the optimal policy parameters. Our goal is to highlight the connection between prediction power and online policy optimization, so the specific online policy optimization algorithms and their proofs are not the primary focus here.

Condition 4.1. For a sequence of positive semi-definite matrices M 0: T -1 , the following inequality holds for all time steps 0 ≤ t &lt; T : For any x ∈ R n , u ∈ R m , and history ι t ( θ ) ,

<!-- formula-not-decoded -->

The LQR setting (Section 3) satisfies Condition 4.1 with M t = R t + B ⊤ t P t +1 B t . But Condition 4.1 is applicable beyond the LQR setting. For example, we show it holds under non-quadratic cost functions in Section 4.1.

Condition 4.1 states that conditioned on any history ι t ( θ ) , the expected Q function of policy π θ grows at least quadratically as the action u deviates from the optimal policy's action. Note that one can always pick M t to be the all-zeros matrix to make Condition 4.1 hold, but the choice of M t will affect the prediction power bound in Theorem 4.3. When M t ≻ 0 , deviating from the action of policy π θ causes a non-negligible loss. The loss is characterized by the difference between the resulting Q function value and the cost-to-go function value. When this condition does not hold with any non-zero matrix M t , one can construct an extreme case when Q π θ t is a constant by letting all cost functions h 0: T be constants; in this case, the prediction power must be zero because every policy achieves the same total cost no matter what predictions they use.

Condition 4.2. One of the following holds for the optimal policy π θ :

(a) For positive semi-definite matrices Σ 0: T -1 , the following holds for all time steps 0 ≤ t &lt; T :

<!-- formula-not-decoded -->

(b) For nonnegative scalars σ 0: T -1 , the following holds for all time steps 0 ≤ t &lt; T :

<!-- formula-not-decoded -->

Before discussing the details, we note that by setting σ t = Tr (Σ t ) , Condition 4.2 (a) implies (and is therefore stronger than) Condition 4.2 (b). Similar to Condition 4.1, one can always pick Σ t to be all-zeros matrix to satisfy Condition 4.2 (a), but it will affect the prediction power bound. The LQR setting (Section 3) satisfies Condition 4.2 (a) with Σ t = E [ Cov [ ¯ u θ t ( I t ( θ )) | F t ( 0 ) ]] .

Note it is possible that the optimal action at different states has a positive variance in different directions, but there is no non-trivial lower bound on the covariance matrix as required by Condition 4.2 (a). In this case, Condition 4.2 (b) provides a weaker alternative and would be useful when we can only establish a lower bound on the trace of the optimal action's covariance matrix (e.g., Section 4.1).

Condition 4.2 (a) states that conditioned on the history I t ( 0 ) from the baseline, the covariance matrix of policy π θ 's action from any F t ( 0 ) -measurable state is positive semi-definite in expectation. Recall that F t ( 0 ) = σ ( I t ( 0 )) . To understand this, suppose that the agent only has access to the baseline information. Then, the agent cannot predict the action that policy π θ would take. This should usually hold because the action π θ t ( X ; I t ( θ )) is not F t ( 0 ) -measurable, and the lower bound in (11) implies the mean-square prediction error cannot improve below a certain threshold. When this condition does not hold with non-zero matrix Σ t (or scalar σ t ), one can design a policy ¯ π ′ that always picks the same action as π θ but only requires access to the baseline information I t ( 0 ) , which implies P ( θ ) = 0 because J ∗ ( 0 ) ≤ J ¯ π ′ ( 0 ) = J ∗ ( θ ) . This can happen, for example, when W 0: T -1 are deterministic.

Theorem 4.3. If Conditions 4.1 and 4.2 (a) hold with matrices M 0: T -1 and Σ 0: T -1 , then P ( θ ) ≥ ∑ T -1 t =0 Tr { M t Σ t } . Alternatively, if Conditions 4.1 and 4.2 (b) hold with matrices M 0: T -1 and scalars σ 0: T -1 , then P ( θ ) ≥ ∑ T -1 t =0 µ min ( M t ) · σ t , where µ min ( · ) returns the smallest eigenvalue.

We defer the proof of Theorem 4.3 to Appendix B. As a remark, in the LQR setting, the first inequality in Theorem 4.3 holds with equality, and it recovers the same expression as Theorem 3.2 in Section 3. There are two main takeaways of Theorem 4.3. First, recall that one can always pick M t and Σ t to be the all-zeros matrices to satisfy Conditions 4.1 and 4.2. In this case, Theorem 4.3 states that P ( θ ) ≥ 0 , which means that having predictions, no matter how weak they are, does not hurt. Second, to characterize the improvement in having predictions, Conditions 4.1 and 4.2 can establish a lower bound for the prediction power that is strictly positive if Tr { M t Σ t } &gt; 0 or µ min ( M t ) σ t &gt; 0 . We provide an example to help illustrate how Conditions 4.1 and 4.2 (a) can work together to ensure that the predictions can lead to a strict improvement on the control cost (see Figure 3 for an illustration).

Figure 3: An illustration of why predictions are helpful, corresponding to Example 4.4. The expected Q functions with perfect predictions (green and orange lines) have lower minima than the expected Q function with uninformative predictions (blue line).

<!-- image -->

## Example 4.4. Consider the following optimal control problem

<!-- formula-not-decoded -->

where each disturbance W t is sampled independently according to P ( W t = -1) = P ( W t = 1) = 1 2 . Suppose that the predictor with parameter θ can predict W t exactly ( i.e. , V t ( θ ) = W t ), while the baseline predictor is uninformative ( e.g. , V t ( 0 ) = 0 ). The Q functions, cumulative cost, and optimal actions under each predictor are

<!-- formula-not-decoded -->

The Q function Q π θ t is strongly convex in u , with Condition 4.1 holding for any M t ∈ [0 , 1] . Furthermore, the optimal action has positive variance, with Condition 4.2 (a) holding for any Σ t ∈ [0 , 1] . Thus, by Theorem 4.3, the prediction power satisfies P ( θ ) ≥ T . Indeed, by comparing the cumulative cost functions, we see that the predictor with parameter θ incurs a lower cumulative cost by exactly T (as expected by Theorem 3.2).

Figure 3 illustrates the expected Q functions at time t = T -1 and x = 0 , which the policies π θ t ( x ; I t ( θ )) and ¯ π t ( x ; I t ( 0 )) seek to minimize. The expected Q functions with perfect predictions have lower minima than the expected Q function with uninformative predictions.

Theorem 4.3 provides a useful tool to characterize the prediction power by reducing the problem of comparing two policies π θ and ¯ π over the whole horizon to studying the properties of one policy π θ at each time step. Our proof of Theorem 4.3 follows the same intuition as the widely-used performance difference lemma in RL (see Lemma 6.1 in [25]), but we adopt novel methods to compare the per-step 'advantage' of π θ along the trajectory of ¯ π with the conditional covariance of policy π θ 's actions. When only the baseline information is available, the agent must pick a suboptimal action (11) and incur a loss (10) at each step, which accumulates to the total cost difference.

While Theorem 4.3 applies to the general dynamical system and cost functions in (1), the two conditions with their key coefficients M t and Σ t (or σ t ) still depend on the optimal Q function and the optimal policy that are implicitly defined through the recursive equations (3). To instantiate Theorem 4.3, we need to derive explicit expressions of M t and Σ t under more specific dynamics/costs.

## 4.1 LTV Dynamics with General Costs

In this section, we consider an online optimal control problem with linear time-varying dynamics and more general cost functions compared with the LQR setting in Section 3.

<!-- formula-not-decoded -->

The LTV system with quadratic cost functions studied in Section 3 is a special case of (13). The setting is challenging because the optimal Q function/policy π θ do not have closed-form expressions like Proposition 3.1. To tackle it, we follow the recursive equations (3) to establish Conditions 4.1 and 4.2 (b). We make the following assumptions about the cost functions and dynamical matrices:

Assumption 4.5. For every time step t , h x t is µ x -strongly convex and ℓ x -smooth, and h u t is µ u -strongly convex and ℓ u -smooth. The dynamical matrices satisfy µ A I ⪯ A ⊤ t A t ⪯ ℓ A I and µ B I ⪯ B ⊤ t B t ⪯ ℓ B I . Further, we assume ℓ A &lt; 1 and µ B &gt; 0 .

Under Assumption 4.5, we can verify Condition 4.1 and show that the expected cost-to-go functions are well-conditioned. We state this result in Lemma 4.6 and defer its proof to Appendix C.4.

Lemma 4.6. Under Assumption 4.5, Condition 4.1 holds with M t = µ u I . Further, conditional expectation E [ C π θ t ( x ; Ξ) | I t ( θ ) = ι t ( θ )] as a function of x is µ t -strongly convex and ℓ t -smooth for any history ι t ( θ ) , where µ t and ℓ t are defined as follows: Let µ T = µ x and ℓ T = ℓ x ,

<!-- formula-not-decoded -->

To show Lemma 4.6 and (14), we prove properties of infimal convolution (see (35) in Appendix C.1) to preserve strongly convexity, smoothness, and the covariance of the input functions or variables.

To establish the second condition about the covariance of the optimal policy's action, we make the following assumption about the joint distribution of the disturbances and the predictions:

̸

Assumption 4.7. The disturbances and predictions can be grouped as pairs { ( W t , V t ( θ )) } T -1 t =0 , where ( W t , V t ( θ )) is joint Gaussian and independent of ( W t ′ , V t ′ ( θ )) when t = t ′ . Further, assume that the baseline is no prediction, i.e., V t ( 0 ) = 0 . And for θ ∈ Θ , there exists λ t ( θ ) ∈ R ≥ 0 such that Cov [ W t ] -Cov [ W t | V t ( θ )] ⪰ λ t ( θ ) I, for any 0 ≤ t &lt; T.

With Assumption 4.7 and Lemma 4.6, we can verify that Condition 4.2 (b) holds with

<!-- formula-not-decoded -->

Since Conditions 4.1 and 4.2 (b) hold, we apply Theorem 4.3 to obtain the prediction power bound.

Theorem 4.8. In the case of LTV dynamics with well-conditioned costs, suppose Assumptions 4.5 and 4.7 hold. The prediction power of the predictor with parameter θ is lower bounded by P ( θ ) ≥ ∑ T -1 t =0 µ u σ t , where σ t is defined in (15) .

We provide a more detailed proof outline and the proofs in Appendix C.1. As a remark, the lower bound of the prediction power in Theorem 4.8 shows that even weak predictions (i.e., small but non-zero λ t ( θ ) in Assumption 4.7) can help improve the control cost compared with the no-prediction baseline. Although Assumption 4.7 limits V t ( θ ) to be only correlated with W t , we provide a roadmap towards more general dependencies on all future W t : T -1 in Appendix E.

## 5 Concluding Remarks

In this work, we propose the metric of prediction power and characterize it in the time-varying LQR setting (Theorem 3.2). We extend our analysis to provide a lower bound for the general setting (Theorem 4.3), which is helpful for establishing the incremental value of (weak) predictions beyond LQR (Theorem 4.8). We emphasize that our framework is very broad. For example, if we let the parameter θ represent the dataset that the predictor is trained on, then the prediction power P ( θ ) effectively quantifies the value of that particular dataset with respect to the optimal control problem.

We would like to highlight three directions of research inspired by our results. First, while our work establishes prediction power of a predictor with parameter θ relative to a strictly less-informative baseline, it does not immediately enable comparison between two arbitrary parameters θ and θ ′ when our general lower bounds in Section 4 are not tight. Second, while we discuss how to evaluate prediction power of a given parameter θ , our work does not specify what the optimal θ is. The problem of learning the parameter that maximizes P ( θ ) may be interesting future work. Third, a natural future direction is to extend the current analysis to systems with partial observability. This remains challenging, as one must distinguish the disturbances in the system dynamics from the noises in the observation model, and the optimal policy's functional form becomes difficult to characterize due to its dependence on the entire history of partial observations and predictions.

## Acknowledgments and Disclosure of Funding

This work was supported by NSF grants (CCF-2326609, CNS-2146814, CPS-2136197, CNS2106403, NGSDI-2105648) and the Caltech Resnick Sustainability Institute.

## References

- [1] Chenkai Yu, Guanya Shi, Soon-Jo Chung, Yisong Yue, and Adam Wierman. The power of predictions in online control. Advances in Neural Information Processing Systems , 33:19942004, 2020.
- [2] Chenkai Yu, Guanya Shi, Soon-Jo Chung, Yisong Yue, and Adam Wierman. Competitive control with delayed imperfect information. In 2022 American Control Conference (ACC) , pages 2604-2610. IEEE, 2022.
- [3] Yiheng Lin, Yang Hu, Guanya Shi, Haoyuan Sun, Guannan Qu, and Adam Wierman. Perturbation-based regret analysis of predictive control in linear time varying systems. Advances in Neural Information Processing Systems , 34:5174-5185, 2021.
- [4] Yiheng Lin, Yang Hu, Guannan Qu, Tongxin Li, and Adam Wierman. Bounded-regret mpc via perturbation analysis: Prediction error, constraints, and nonlinearity. Advances in Neural Information Processing Systems , 35:36174-36187, 2022.
- [5] Runyu Zhang, Yingying Li, and Na Li. On the regret analysis of online lqr control with predictions. In 2021 American Control Conference (ACC) , pages 697-703. IEEE, 2021.
- [6] Niangjun Chen, Anish Agarwal, Adam Wierman, Siddharth Barman, and Lachlan LH Andrew. Online convex optimization using predictions. In Proceedings of the 2015 ACM SIGMETRICS International Conference on Measurement and Modeling of Computer Systems , pages 191-204, 2015.
- [7] Tianyu Chen, Yiheng Lin, Nicolas Christianson, Zahaib Akhtar, Sharath Dharmaji, Mohammad Hajiesmaili, Adam Wierman, and Ramesh K Sitaraman. Soda: An adaptive bitrate controller for consistent high-quality video streaming. In Proceedings of the ACM SIGCOMM 2024 Conference , pages 613-644, 2024.
- [8] Niangjun Chen, Joshua Comden, Zhenhua Liu, Anshul Gandhi, and Adam Wierman. Using predictions in online optimization: Looking forward with an eye on the past. ACMSIGMETRICS Performance Evaluation Review , 44(1):193-206, 2016.
- [9] Yingying Li, Guannan Qu, and Na Li. Using predictions in online optimization with switching costs: A fast algorithm and a fundamental limit. In 2018 Annual American Control Conference (ACC) , pages 3008-3013. IEEE, 2018.
- [10] Yingying Li, Xin Chen, and Na Li. Online optimal control with linear dynamics and predictions: Algorithms and regret analysis. Advances in Neural Information Processing Systems , 32, 2019.
- [11] Yiheng Lin, Gautam Goel, and Adam Wierman. Online optimization with predictions and non-convex losses. Proceedings of the ACM on Measurement and Analysis of Computing Systems , 4(1):1-32, 2020.
- [12] Jayanta Mandi, James Kotary, Senne Berden, Maxime Mulamba, Victor Bucarey, Tias Guns, and Ferdinando Fioretto. Decision-focused learning: Foundations, state of the art, benchmark and future opportunities. Journal of Artificial Intelligence Research , 80:1623-1701, August 2024.
- [13] Priya L. Donti, Brandon Amos, and J. Zico Kolter. Task-based End-to-end Model Learning in Stochastic Optimization. In Advances in Neural Information Processing Systems , volume 30, Long Beach, CA, USA, December 2017. Curran Associates, Inc.
- [14] Adam N. Elmachtoub and Paul Grigas. Smart 'Predict, then Optimize'. Management Science , 68(1):9-26, January 2022.

- [15] Brandon Amos, Ivan Jimenez, Jacob Sacks, Byron Boots, and J. Zico Kolter. Differentiable MPC for End-to-end Planning and Control. In Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018.
- [16] Christopher Yeh, Nicolas Christianson, Alan Wu, Adam Wierman, and Yisong Yue. End-to-end conformal calibration for optimization under uncertainty. Preprint arXiv:2409.20534 , 2024.
- [17] Irina Wang, Cole Becker, Bart Van Parys, and Bartolomeo Stellato. Learning Decision-Focused Uncertainty Sets in Robust Optimization, July 2024.
- [18] Tongxin Li, Ruixiao Yang, Guannan Qu, Guanya Shi, Chenkai Yu, Adam Wierman, and Steven Low. Robustness and consistency in linear quadratic control with untrusted predictions. Proceedings of the ACM on Measurement and Analysis of Computing Systems , 6(1):1-35, 2022.
- [19] Michael O'Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, and Soon-Jo Chung. Neural-fly enables rapid learning for agile flight in strong winds. Science Robotics , 7(66):eabm6597, 2022.
- [20] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2(3-4):157-325, 2016.
- [21] Daan Rutten, Nicolas Christianson, Debankur Mukherjee, and Adam Wierman. Smoothed online optimization with unreliable predictions. Proceedings of the ACM on Measurement and Analysis of Computing Systems , 7(1):1-36, 2023.
- [22] Yiheng Lin, James A. Preiss, Emile Timothy Anand, Yingying Li, Yisong Yue, and Adam Wierman. Online adaptive policy selection in time-varying systems: No-regret via contractive perturbations. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- [23] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control with adversarial disturbances. In International Conference on Machine Learning , pages 111-119. PMLR, 2019.
- [24] Yiheng Lin, James A Preiss, Fengze Xie, Emile Anand, Soon-Jo Chung, Yisong Yue, and Adam Wierman. Online policy optimization in unknown nonlinear systems. In The Thirty Seventh Annual Conference on Learning Theory , pages 3475-3522. PMLR, 2024.
- [25] Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the Nineteenth International Conference on Machine Learning , pages 267274, 2002.
- [26] Amir Beck. First-Order Methods in Optimization . SIAM, 2017.
- [27] Jerrold E Marsden and Anthony Tromba. Vector Calculus . Macmillan, 2003.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our claims in the abstract are justified by the major contributions, which include pointers to specific theorems and sections in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We highlight some limitations in the last paragraph of Concluding Remarks.

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

Justification: All of our theoretical results are based on rigorous assumptions and proofs. Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We describe the full problem settings and algorithm patameters in all simulations.

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

Justification: We submit the simulation code in the supplementary material.

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

Justification: We specify all the details in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We discuss why we believe each experiment observation is significant.

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

Justification: We discuss about compute resources for each simulation result.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We check the NeurIPS Code of Ethics and believe our work conforms with every code.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work focuses on theoretical research of online control. We do not see any potential societal impact.

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

Justification: Our paper does not pose such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use any specific assets beyond standard open-source scientific Python packages such as numpy and matplotlib for running experiments.

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

Justification: Upon paper acceptance, we will release our code on GitHub with a permissive license. Our paper does not introduce any new data or models.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this work does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Proofs and Examples for LTV Dynamics with Quadratic Costs

## A.1 Proof of Proposition 3.1

Recall that we introduce the shorthand

<!-- formula-not-decoded -->

We show by induction that

<!-- formula-not-decoded -->

together with the expression of the optimal cost-to-go function

<!-- formula-not-decoded -->

where recall that for t 2 &gt; t 1 ,

<!-- formula-not-decoded -->

and Ψ t ( I t ( θ )) is a function of the history observations/predictions which does not depend on x . Note that (16) holds when t = T because C π θ T ( x ; Ξ) = x ⊤ P T x .

Suppose that (16) holds for t +1 . Then, we have

<!-- formula-not-decoded -->

To simplify the notation, let

<!-- formula-not-decoded -->

We see that the expected Q function is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where t ( ; t ( )) is given by

<!-- formula-not-decoded -->

Using the expected Q function, we know that the optimal policy will pick the action

<!-- formula-not-decoded -->

Therefore, we see the optimal cost-to-go function at time step t is given by

<!-- formula-not-decoded -->

Note that the term -2¯ u θ t ( I t ( θ )) ⊤ R t K t x and the term +2¯ u θ t ( I t ( θ )) ⊤ B ⊤ t P t +1 ( A t -B t K t ) x cancel out because R t K t = B ⊤ t P t +1 ( A t -B t K t ) . We also note that the matrix in the first quadratic term can be simplified to

<!-- formula-not-decoded -->

where the last equation follows by the definition of P t in (5).

Therefore, we obtain that

<!-- formula-not-decoded -->

where the residual term ¯ ψ t ( I t ( θ )) is given by

<!-- formula-not-decoded -->

Thus, we have shown the statement of Proposition 3.1 and 16 by induction.

## A.2 Proof of Theorem 3.2

Note that the cost-to-go function C π θ t ( x ; ξ ) can be expressed as Q π θ t ( x, π θ t ( x ; ι t ( θ ); ξ )) . Substituting the expression of π θ t ( x ; ι t ( θ )) into the expression of E [ Q π θ t ( x, u ; Ξ) | I t ( θ ) = ι t ( θ ) ] in Proposition 3.1 gives that

<!-- formula-not-decoded -->

Substituting u = ¯ π t ( x ; ι t ( 0 )) into the above equation gives that

<!-- formula-not-decoded -->

where we use the expression of optimal policies in Proposition 3.1 in (18a) and rearrange the terms in (18b). Note that by Proposition 3.1, we have

<!-- formula-not-decoded -->

Therefore, by the tower rule and the definition of conditional covariance, we obtain that

<!-- formula-not-decoded -->

Let { ( ¯ X t , ¯ U t ) } denote the (random) trajectory achieved ¯ π 0: T -1 under problem instance Ξ . Since ¯ X t is F t ( 0 ) -measurable, by (19), we obtain that

<!-- formula-not-decoded -->

where we use ¯ U t = ¯ π t ( ¯ X t ; I t ( 0 )) . Note that we have

<!-- formula-not-decoded -->

Substituting this into (20) and taking expectation give that

<!-- formula-not-decoded -->

Summing (21) over t = 0 , 1 , . . . , T -1 , we obtain that

<!-- formula-not-decoded -->

Note that the left-hand side equals P ( θ ) . Thus, we have finished the proof of Theorem 3.2.

## A.3 Proof of the MPC form

In this section, we show that the MPC policies defined in (7) and (8) are equivalent to the optimal policy in Proposition 3.1.

To simplify the notation, we define the large vectors

<!-- formula-not-decoded -->

Follow the approach of system level thesis, we know the constraints that

<!-- formula-not-decoded -->

can be expressed equivalently by the affine relationship

<!-- formula-not-decoded -->

Let ⃗ Q = Diag ( Q t , . . . , Q T -1 , P T ) and ⃗ R = Diag ( R t , . . . , R T -1 ) . We know the objective function (with equality constraints)

<!-- formula-not-decoded -->

can be written equivalently in the unconstrained form

<!-- formula-not-decoded -->

We introduce the notations

<!-- formula-not-decoded -->

The MPC policy in (7) can be expressed as

<!-- formula-not-decoded -->

Because the objective function can be reduced to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last term is independent with x and ⃗ u . Thus, the MPC policy in (7) is equivalent to

<!-- formula-not-decoded -->

which is the MPC policy in (8).

Now, we show that (8) is equivalent to the optimal policy in Proposition 3.1. For any sequence w t : T -1 , let MPC ( x, w t : T -1 ) denote the first entry of the solution to

<!-- formula-not-decoded -->

To show that (8) is equivalent to the optimal policy in Proposition 3.1, we only need to show that

<!-- formula-not-decoded -->

holds for any sequence w t : T -1 . To see this, we consider the case when w t : T -1 are deterministic disturbances on and after time step t , i.e., the agent knows w t : T -1 exactly at time step t . In this scenario, we know the optimal policy is to follow the planned trajectory according to MPC in (22). On the other hand, by Proposition 3.1, we know the optimal action to take at time t is -K t x -( R t + B ⊤ t P t +1 B t ) -1 B ⊤ t ∑ T -1 τ = t Φ ⊤ τ +1 ,t +1 P τ +1 w t . Therefore, the first step planned by MPC must be identical with -K t x -( R t + B ⊤ t P t +1 B t ) -1 B ⊤ t ∑ T -1 τ = t Φ ⊤ τ +1 ,t +1 P τ +1 w t . Thus, (25) holds. And replacing w t :( T -1) with w θ t :( T -1) | t finishes the proof.

## A.4 Prediction Power Evaluation

Based on our discussion in Section 3, we propose an algorithm (cf. Algorithm 1) to evaluate the prediction power efficiently given a set of historical problem instances { ξ n } N n =1 . Recall that we define the surrogate-optimal action as

<!-- formula-not-decoded -->

which is the optimal action that an agent should take with the oracle knowledge of all future disturbances at time t . In the prediction power given by Theorem 3.2, we can express ¯ u θ t ( I t ( θ )) as E [¯ u ∗ t (Ξ) | I t ( θ )] by Proposition 3.1, which is the expectation of ¯ u ∗ t (Ξ) condition on the history at time step t .

We design Algorithm 1 as follows. While iterating backward from time step T -1 to 0 , the algorithm first constructs a dataset of the surrogate optimal action ¯ u ∗ t (Ξ) as the fitting target. Then, the algorithm estimates the covariance of ¯ u ∗ t (Ξ) when conditioning on I t ( 0 ) and I t ( θ ) , respectively, using a subroutine (Algorithm 2). The last step of Algorithm 1 gives the prediction power because E [ Cov [ ¯ u θ t ( I t ( θ )) | F t ( 0 ) ]] can be decomposed as E [ Cov [¯ u ∗ t (Ξ) | I t ( 0 )]] -E [ Cov [¯ u ∗ t (Ξ) | I t ( θ )]] , and we prove this result in Lemma A.1. This decomposition is helpful because otherwise, we would need to evaluate the conditional expectation inside another conditional expectation. Specifically, ¯ u θ t ( I t ( θ )) needs to be approximated by a learned regressor (say, ϕ ) that takes I t ( θ ) as an input. Then, to evaluate E [ Cov [ ¯ u θ t ( I t ( θ )) | F t ( 0 ) ]] , we would need to train another regressor to predict the output of ϕ . Our decomposition avoids this hierarchical dependence.

Lemma A.1. For any random variable X and two σ -algebras F ⊆ F ′ , the following equation holds

<!-- formula-not-decoded -->

Proof of Lemma A.1. By the law of total covariance, we see that

<!-- formula-not-decoded -->

## Algorithm 1 Prediction Power Evaluation

Require: Dataset D of problem instances { ξ n } N n =1 . 1: for t = T -1 , T -2 , . . . , 0 do 2: Compute P t , H t , K t and { Φ t,t ′ } t ′ ≥ t according to (5) and (6). 3: Compute M t = R t + B ⊤ t P t +1 B t . 4: for n = 1 , 2 , . . . , N do 5: Compute ¯ u ∗ t ( ξ n ) according to (9) in problem instance ξ n . 6: end for 7: Call Algorithm 2 to estimate Σ 0 t := E [ Cov [¯ u ∗ t (Ξ) | I t (0)]] using { (¯ u ∗ t ( ξ n ) , ι n t (0)) } N n =1 . 8: Call Algorithm 2 to estimate Σ θ t := E [ Cov [¯ u ∗ t (Ξ) | I t ( θ )]] using { (¯ u ∗ t ( ξ n ) , ι n t ( θ )) } N n =1 . 9: end for 10: return P ( θ ) = ∑ T -1 t =0 Tr { Σ 0 t M t } -∑ T -1 t =0 Tr { Σ θ t M t }

Taking expectation on both sides gives that

<!-- formula-not-decoded -->

which is equivalent to the statement of Lemma A.1.

Evaluation of the Expected Conditional Covariance. For two general random variables X and Y , we follow a standard procedure to evaluate the expectation of their conditional covariance E [ Cov [ Y | X ]] using a dataset { ( x n , y n ) } that is independently sampled from the joint distribution of ( X,Y ) (Algorithm 2). The algorithm first trains a regressor ψ that approximates the conditional expectation E [ Y | X ] , where we use the definition:

<!-- formula-not-decoded -->

Then, ψ is used for evaluating the conditional covariance. During training, we split the dataset into train, validation, and test sets in order to prevent overfitting.

## Algorithm 2 Expected Conditional Covariance Estimator (ECCE)

Require: Dataset D that consists of input/output pairs ( x n , y n ) .

- 2: Initialize a regressor ψ with input x and target output y .
- 1: Split the dataset D to D train , D val , and D test .
- 3: Fit ψ to D train with MSE and use D val to prevent overfitting.
- 4: return Σ := 1 | D test | ∑ n ∈ D test ( y n -ψ ( x n ))( y n -ψ ( x n )) ⊤ .

## A.5 Details of Examples in Section 3.1

In this section, we present the specific instantiation of Example 3.3 in Section 3.1 and another example (Example A.2) for the mismatch between prediction power and prediction accuracy.

## A.5.1 Instantiation of Example 3.3

We instantiate Example 3.3 with the following parameters:

<!-- formula-not-decoded -->

Under different values of coefficient ρ , we train a linear regressor to predict each entry of W t from V t ( θ ) (or V t ( I ) ) over a train dataset with 64000 independent samples. We plot in the MSE ρ curve on a test dataset with 16000 independent samples in Figure 4. From the plot, we see that the predictors V t ( θ ) and V t ( I ) achieve the same MSE when predicting each entry of W t under each ρ ∈ { 0 , 0 . 1 , . . . , 0 . 7 } .

Then, we use the trained linear regressors as W θ t | t and W I t | t to implement the optimal policy in Proposition 3.1. We plot the averaged total cost over 16000 trajectories with horizon T = 100 in

<!-- image -->

Figure 4: Example 3.3: MSE ρ curve.

Figure 5: Example 3.3: Control cost ρ curve.

<!-- image -->

Figure 6: Example A.2: MSE - time curve.

<!-- image -->

Figure 5. From the plot, we see that the optimal policies under the predictors V t ( θ ) and V t ( I ) achieve significantly different control costs when ρ &gt; 0 . We also plot the theoretical expected control cost in Figure 5 to verify this cost difference. Running this experiment takes about 50 seconds on Apple Mac mini with Apple M1 CPU.

## A.5.2 An One-dimension Example

We also provide an example with n = 1 , where the prediction V t ( θ ) is correlated with two steps of future disturbances W t and W t +1 .

Example A.2. Suppose the disturbance at each time step can be decomposed as W t = ∑ 2 i =0 W ( i ) t , where the { W ( i ) t } 2 i =0 are independently sampled from three mean-zero distributions. We compare two predictors: V t (1) = ( W (1) t , W (0) t +1 ) and V t (2) = P ( W (0) t + W (1) t ) +( A ⊤ -A ⊤ PH ) PW (0) t +1 . They have the same prediction power when used in the control problem because

<!-- formula-not-decoded -->

However, we know that F t (1) is a strict super set of F t (2) , thus V t (1) can achieve a better MSE than V t (2) when predicting the disturbances. This is empirically verified in a 1D LQR problem with A = B = Q = R = (1) and W ( i ) t i.i.d. ∼ N (0 , 1) , as we plot in Figures 6. In the simulation, we train linear regressors to predict W t and W t +1 with the history I t (1) or I t (2) for each time step t &lt; T = 100 over a train dataset of size 160000 . Then, we plot the MSE - time curve on a test dataset of size 40000 . Running this experiment takes about 270 seconds on Apple Mac mini with Apple M1 CPU.

## A.6 Details of Example 3.4

We instantiate Example 3.4 with the same dynamics and costs as Example 3.3, i.e.,

<!-- formula-not-decoded -->

To build the predictors, we sample the true disturbance W t i.i.d. ∼ N (0 , I ) and fix the coefficient ρ = 0 . 5 . The online policy optimization starts with the initial policy parameter Υ 0 = 0 . When implementing M-GAPS in both scenarios, we use the decaying learning rate sequence η t = (1 + t/ 1000) -0 . 5 . The optimal predictive policy for using V 0: t -1 (1) or V 0: t -1 (2) are π 1 0: T -1 and π 2 0: T -1 , whose closed-form expressions are given by Proposition 3.1. Note that for the history ι t (1) , the optimal predictive policy π 1 t only depends on v t (1) because all other entries are independent with future disturbances W t : T -1 . Similarly, for the history ι t (2) , the optimal predictive policy π 2 t only depends on v t -1 (2) and v t (2) .

In Figures 1 and 2, we compute the average cost improvement of M-GAPS (or the optimal predictive policy) against the optimal no-prediction controller ¯ π t ( x ) = -Kx . That is, on each problem instance ξ , we plot

<!-- formula-not-decoded -->

for time t = 0 , 1 , . . . , T -1 . The prediction power (averaged over time) is given by P ( θ ) /T . We simulate 30 random trajectories with T = 80000 and plot the mean with the 25-th and 75th percentiles as shaded areas. From the plots, we see that M-GAPS' average cost improvement converges towards the prediction power over time in the first scenario but stays far away with the prediction power in the second scenario. This is as expected, because the optimal predictive policy π 2 t is not in the candidate policy set of M-GAPS in the second scenario. Simulating the first scenario takes about 200 seconds on Apple Mac mini with Apple M1 CPU. The second takes about 210 seconds on the same hardware.

## B Proof of Theorem 4.3

Since we assume x 0 is the initial state (deterministic) and π θ is the optimal policy under the predictor with parameter θ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let { ¯ X 0: T , ¯ U 0: T -1 } be the trajectory of the baseline controller ¯ π 0: T -1 under instance Ξ starting from ¯ X 0 = x 0 . First, we will prove by backwards induction that the difference in cumulative costs between the optimal controller π θ and ¯ π has the following decomposition:

<!-- formula-not-decoded -->

For the base case at time T -1 , we apply the definition of C ¯ π T -1 to get

<!-- formula-not-decoded -->

For the inductive step, suppose that

<!-- formula-not-decoded -->

Note that for any t &lt; T ,

<!-- formula-not-decoded -->

Similarly, we also have that

Therefore,

<!-- formula-not-decoded -->

This completes the induction.

Next, define U t := π θ t ( ¯ X t ; I t ( θ )) . Note that U t is F t ( θ ) -measurable, and ¯ U t is F t ( 0 ) -measurable and therefore also F t ( θ ) -measurable. Because we assume the matrices M 0: T -1 satisfy Condition 4.1,

<!-- formula-not-decoded -->

Let ˜ U t := E [ U t | I t ( 0 )] . We see that

<!-- formula-not-decoded -->

where we use ( ¯ U t -˜ U t ) is F t ( 0 ) -measurable in (29a); we use the definition of ˜ U t in (29b). Applying the towering rule in (27) and substituting in (28) gives that

<!-- formula-not-decoded -->

If the stronger Condition 4.2 (a) holds, by (29), since ¯ X t is F t ( 0 ) -measurable, we have

Then, we can apply (31) in (30) to obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Else, if the weaker Condition 4.2 (b) holds, by (29), since ¯ X t is F t ( 0 ) -measurable, we have

Note that for any positive semi-definite matrices A,B,C such that A ⪰ C ⪰ 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since M t ⪰ µ min ( M t ) I , we can apply (33) in (30) to obtain that

<!-- formula-not-decoded -->

## C Proofs for LTV Dynamics with General Costs

In this section, we first provide a proof outline of Theorem 4.8 (Appendix C.1). Then, we discuss an example where the MPC in (7) is suboptimal (Appendix C.2). Lastly, we provide the proofs for the key technical results required by the proof of Theorem 4.8.

## C.1 Proof Outline of Theorem 4.8

Assumption 4.5 makes two requirements about the well-conditioned cost functions, which are standard in the literature of online optimization and control [3, 4]. For the last requirement, we additionally require ℓ A &lt; 1 , which implies that the system is open-loop stable. Under Assumption 4.5, the expected cost-to-go function is a well-conditioned function, which is important for establishing Conditions 4.1 and 4.2 (b). We state this result formally in Lemma 4.6 in Section 4.1, which establishes uniform bounds for the strongly convexity/smoothness of the conditional expectation of cost-to-go functions: µ t is uniformly bounded below by µ x and ℓ t is uniformly bounded above by ℓ x 1 -ℓ A . We present a proof sketch of Lemma 4.6 and defer the formal proof to Appendix C.4.

Starting from time step T , we know the cost-to-go C π θ T ( x ; Ξ) equals to the terminal cost h x t ( x ) . It satisfies the strong convexity/smoothness directly by Assumption 4.5. We repeat the following induction iterations: Given E [ C π θ t +1 ( x ; Ξ) | I t +1 ( θ ) ] at time t +1 ,we define an auxiliary function that adds in the disturbance residual W t -W θ t | t and condition on the history at time t :

<!-- formula-not-decoded -->

It can be expressed as E [ E [ C π θ t +1 ( x + W t -W θ t | t ; Ξ) | I t +1 ( θ ) ]∣ ∣ ∣ I t ( θ ) = ι t ( θ ) ] by the tower rule. Thus, we know function ¯ C π θ t +1 is strongly convex and smooth in x because these properties are preserved after taking the expectation. Then, we can obtain the expected cost-to-go function E [ C π θ t ( x ; Ξ) | I t ( θ ) = ι t ( θ ) ] = h x t ( x ) + min u ( h u t ( u ) + ¯ C π θ t +1 ( A t x + B t u + w θ t | t ; ι t ( θ )) ) . We use an existing tool called infimal convolution to study the optimal value of the this optimization problem as a function of x . Specifically, define an operator □ B : 4

<!-- formula-not-decoded -->

One can show that if f and ω are well-conditioned functions, then ( f □ B ω ) is also well-conditioned (see Appendix C.6 for the formal statement and proof). We can use this result to show the expected cost-to-go function E [ C π θ t ( x ; Ξ) | I t ( θ ) = ι t ( θ ) ] = h x t ( x ) + ( h u t □ ( -B t ) ¯ C π θ t +1 )( A t x + w θ t | t ; ι t ( θ )) , is also well-conditioned in x at time step t , which completes the induction.

4 If ω takes an additional parameter w , we denote ( f □ B ω ) ( x ; w ) := min u ∈ R m { f ( u ) + ω ( x -Bu ; w ) }

For the second condition on the covariance of π θ t 's actions, we note that λ t ( θ ) in Assumption 4.7 should be positive as long as V t ( θ ) has some weak correlation with W t . Under Assumption 4.7, we can express the optimal policy as

<!-- formula-not-decoded -->

While the original definition of ¯ C π θ t +1 in (34) requires the history ι t ( θ ) as an input, it no longer depends on the history under Assumption 4.7. We defer the proof to Appendix C.5.

We can express π θ t ( x ; I t ( θ )) as the solution to ( h u t □ ( -B t ) ¯ C π θ t +1 )( A t x + W t | t ) . For some distributions including Gaussian, the covariance in the input of an infimal convolution will be passed through to its optimal solution. Specifically, let u ( f □ B ω ) ( x ) denote the solution to the optimization problem (35). When ω and f are well-conditioned, we can derive a lower bound on the trace of the covariance Tr { Cov [ u ( f □ B ω ) ( X ) ]} that depends on the covariance of X . Due to space limit, we defer the formal statement of this result and its proof to Lemma C.2 in Appendix C.6. Using this property and the observation that π θ t ( x ; I t ( θ )) can be expressed as u ( h u t □ -Bt ¯ C π θ t +1 ) ( A t x + W θ t | t ) , we can directly verify that Condition 4.2 (b) holds with

<!-- formula-not-decoded -->

Since Lemma 4.6 and (37) imply that Conditions 4.1 and 4.2 (b) hold with M t = µ t I and σ t respectively, we can apply Theorem 4.3 to obtain the prediction power lower bound in Theorem 4.8.

## C.2 Example: MPC can be suboptimal

We first highlight the challenge by showing that MPC can be suboptimal, i.e., only planning and optimizing based on the current information might be suboptimal when the cost functions are not quadratic.

Consider a 2-step optimal control problem (1-dimension):

<!-- formula-not-decoded -->

The cost functions are given by

<!-- formula-not-decoded -->

Suppose the system starts at x 0 = 0 . At time step 0 , MPC (7) solves the optimization

Suppose W 1 is a random variable that satisfies P ( W 1 = 1) = p and P ( W 1 = 0) = 1 -p , where 0 &lt; p &lt; 1 . At time 0 , we don't have any knowledge about W 1 (i.e., W 1 is independent with I 0 ( θ ) ). However, at time 1 , we can predict W 1 exactly, which means σ ( W 1 ) ⊆ F 1 ( θ ) .

<!-- formula-not-decoded -->

Since I 0 ( θ ) is independent with W 1 , the optimization problem can be expressed equivalently as

<!-- formula-not-decoded -->

The equation holds because the planned trajectory must avoid the huge cost at time step 2 . Solving this gives u 0 = -1 3 . Thus, implementing MPC incurs a total cost that is at least 2 u 2 0 = 2 9 . In contrast, if one just pick u 0 = 0 , the agent can pick u 1 based on the prediction revealed at time step 2 :

<!-- formula-not-decoded -->

In this case, the expected cost incurred is p . Thus, we can claim that MPC is not the optimal policy when p &lt; 2 9 . The underlying reason that MPC is suboptimal is because it does not consider what

information may be available when we make the decision in the future. In this specific example, since W 1 is revealed at time 1 , we don't need to verify about the small probability event that leads to a huge loss.

We dive deeper into the reason why MPC (7) is optimal in the LQR setting (Section 3). Note that the expected optimal cost-to-go function at time step 1 is

<!-- formula-not-decoded -->

Here, u 1 is F 1 ( θ ) -measurable. And the true optimal policy at time 0 is decided by solving

<!-- formula-not-decoded -->

In general, we cannot use

<!-- formula-not-decoded -->

to replace E [ C π θ 1 ( X 1 ; Ξ) ∣ ∣ ∣ I 0 ( θ ) ] like what MPC does in (38) because here u 1 is F 0 ( θ ) -measurable in (40). Recall that u 1 is F 1 ( θ ) -measurable in (39) and F 0 ( θ ) is a subset of F 1 ( θ ) . However, in the LQR setting, as the closed-form expression (16), the part of E [ C π θ 1 ( X 1 ; Ξ) | I 0 ( θ ) ] that depends on X 1 will not change even if F 1 ( θ ) changes. Thus, we can assume F 1 ( θ ) = F 0 ( θ ) without affecting the optimal action at time 0 . Therefore, MPC's replacement of E [ C π θ 1 ( X 1 ; Ξ) | I 0 ( θ ) ] with (40) is valid in the LQR setting.

## C.3 Infimal Convolution Properties

The first result states that the variant of infimal convolution preserves the strong convexity/smoothness of the input functions. The proof can be found in Appendix C.6.

Lemma C.1. Consider a variant of infimal convolution defined as

<!-- formula-not-decoded -->

where f : R m → R , ω : R n → R , and B ∈ R n × m is a matrix. Suppose that f is a µ f -strongly convex function, and ω is a µ ω -strongly convex and ℓ ω -smooth function. Then, f □ B ω is a ( µ ω µ f µ f + ∥ B ∥ 2 µ ω ) -strongly convex and ℓ ω -smooth function. We also have ∇ ( f □ B ω )( x ) = ∇ ω ( x -Bu ( x )) .

The second result is about the optimal solution of the variant of infimal convolution. It states that for some distributions, the covariance on the input will induce a variance on the optimal solution. We state it in Lemma C.2 and defer the proof to Appendix C.7.

Lemma C.2. Let u ( f □ B ω ) ( x ) denote the solution to the optimization problem (35) . Suppose function f is µ f -strongly convex. Function ω is µ ω -strongly convex and ℓ ω -smooth. Suppose X is a random vector with bounded mean and Cov [ X ] = Σ ⪰ σ 0 I . Further, there exists a constant C &gt; 0 such that for any positive integer N , X can be decomposed as X = ∑ N i =1 X i for i.i.d. random vectors X i that satisfies E [ ∥ X i ∥ 4 ] ≤ C · N -2 . Then,

<!-- formula-not-decoded -->

As a remark, an example of X that satisfies the assumptions is the normal distribution X ∼ N (0 , Σ) . We have X i ∼ N (0 , Σ /N ) , thus E [ ∥ X i ∥ 4 ] ≤ 3 Tr { Σ } N -2 .

The next result (Lemma C.3) considers the case when there is an additional input w to function ω in the infimal convolution. When this additional parameter causes a covariance on the gradient ∇ 1 ω ( x, W ) , the optimal solution of the infimal convolution will also have a nonzero variance.

Lemma C.3. Suppose that ω ( x, w ) satisfies that ω ( · , w ) is an ℓ ω -smooth convex function for all w . For a random variable W , suppose that the following inequality holds for arbitrary fixed vector x ∈ R n ,

<!-- formula-not-decoded -->

Suppose that f : R m → R is a µ f -strongly convex and ℓ f -smooth function ( m ≤ n ). Let B be a matrix in R n × m . Then, the optimal solution of the infimal convolution

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds for arbitrary fixed vector x , where σ min ( B ) denotes the minimum singular value of B .

Lemma C.3 is useful for showing Lemma C.2. We defer its proof to Appendix C.8.

## C.4 Proof of Lemma 4.6

We use induction to show that E [ C π θ t ( x ; Ξ) | I t ( θ ) = ι t ( θ ) ] is a µ t -strongly convex and ℓ t -smooth function for any ι t ( θ ) , where the coefficients µ t and ℓ t are defined recursively in (14). To simplify the notation, we will omit ' I t ( θ ) = ' in the conditional expectations throughout this proof when conditioning on a realization of the history ι t ( θ ) .

Note that the statement holds for t = T , because E [ C π θ T ( x ; Ξ) | ι T ( θ ) ] = h x T ( x ) and the terminal cost h x T is µ x -strongly convex and ℓ x -smooth.

Suppose the statement holds for t +1 . We see that

<!-- formula-not-decoded -->

By the induction assumption, we know that E [ C π θ t +1 ( · ; Ξ) | ι t +1 ( θ ) ] is a µ t +1 -strongly convex and ℓ t +1 -smooth function for any ι t +1 ( θ ) . Thus, E [ C π θ t +1 ( · + W t ; Ξ) | ι t ( θ ) ] is also a µ t +1 -strongly convex and ℓ t +1 -smooth function. Therefore,

<!-- formula-not-decoded -->

is a µ u µ t +1 µ u + b 2 µ t +1 -strongly convex and ℓ t +1 -smooth function of x by Lemma C.1. By changing the variable from x to A t x , we see that

<!-- formula-not-decoded -->

is a µ A · µ u µ t +1 µ u + b 2 µ t +1 -strongly convex and ℓ A · ℓ t +1 -smooth function by Assumption 4.5. Since h x t is a µ x -strongly convex and ℓ x -smooth function, we see that E [ C π θ t ( x ; Ξ) | ι t ( θ ) ] is also a µ t -strongly convex and ℓ t -smooth function because

<!-- formula-not-decoded -->

## C.5 Proof of Theorem 4.8

Note that the optimal action at time step t is determined by

<!-- formula-not-decoded -->

satisfies that

This can be further simplified to

<!-- formula-not-decoded -->

The additional input I t ( θ ) is not required for ¯ C π θ t +1 because the function ¯ C π θ t +1 ( x ; ι t ( θ )) does not change with the history ι t ( θ ) under Assumption 4.7. The reason is that W t -W θ t | t and all future predictions and disturbances W t +1: T -1 , V θ t +1: T -1 are independent with the history I t ( θ ) . By (36), we see that

<!-- formula-not-decoded -->

Under Assumption 4.7, we see that

<!-- formula-not-decoded -->

and W θ t | t is Gaussian. Therefore, we can apply Lemma C.2 to obtain that

<!-- formula-not-decoded -->

Thus, Condition 4.2 (b) holds with σ t .

On the other hand, Condition 4.1 holds with M t = µ t I by Lemma 4.6. Therefore, by Theorem 4.3, we obtain that P ( θ ) ≥ ∑ T -1 t =0 µ u σ t .

## C.6 Proof of Lemma C.1

By the definition of conjugate, we see that

<!-- formula-not-decoded -->

where we use the definition of f □ B ω in (43a); we change the order of taking the maximum and use ⟨ y, Bu ⟩ = ⟨ B ⊤ y, u ⟩ in (43b); we use the definition of ω ∗ in (43c); we use the definition of f ∗ in (43d).

Since f □ B ω is convex, by Theorem 4.8 in [26], we know that

<!-- formula-not-decoded -->

Since ω is a µ ω -strongly convex and ℓ ω -smooth function, we know ω ∗ is an 1 ℓ ω -strongly convex and 1 µ ω -smooth function by the conjugate correspondence theorem [26]. Similarly, we know that f ∗ is a 1 µ f -smooth convex function. Thus, we know that ω ∗ ( y ) + f ∗ ( B ⊤ y ) is an 1 ℓ ω -strongly convex and ( 1 µ ω + ∥ B ∥ 2 µ f ) -smooth function. Therefore, by the conjugate correspondence theorem, we know that f □ B ω is a ( µ ω µ f µ f + ∥ B ∥ 2 µ ω ) -strongly convex and ℓ ω -smooth function.

Now, we show that

<!-- formula-not-decoded -->

Following a similar approach with the proof of Theorem 5.30 in [26], we define z = ∇ ω ( x -Bu ( x )) . Define function ϕ ( ξ ) := ( f □ B ω )( x + ξ ) -( f □ B ω )( x ) -⟨ ξ, z ⟩ . We see that

<!-- formula-not-decoded -->

where in (46a), we use

<!-- formula-not-decoded -->

we use the convexity of ω in (46b); we use the Cauchy-Schwarz inequality in (46c); we use the assumption that ω is ℓ ω -smooth in (46d).

Since ( f □ B ω ) is a convex function, ϕ is also convex, thus we see that

<!-- formula-not-decoded -->

Combining this with (46), we conclude that lim ∥ ξ ∥→ 0 | ϕ ( ξ ) | / ∥ ξ ∥ = 0 . Thus, (45) holds.

## C.7 Proof of Lemma C.2

By Theorem D.1, we see that

Then, we apply Lemma C.3 with the second function input to the infimal convolution as ˜ ω ( x, w ) := ω ( x + w ) . In the context of Lemma C.3, we set W = X , so the assumption about the covariance of the gradient holds with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that for any fixed w , ˜ ω ( · , w ) is µ ω -strongly convex. Therefore, we obtain that

<!-- formula-not-decoded -->

## C.8 Proof of Lemma C.3

Because function c is ℓ c -smooth, we have

<!-- formula-not-decoded -->

Because function f is ℓ f -smooth, we have

<!-- formula-not-decoded -->

where we use the triangle inequality in (48a); we use the smoothness of f in (48b).

Note that by the first-order optimality condition, we have

<!-- formula-not-decoded -->

Therefore, for any w,w ′ , we have that

<!-- formula-not-decoded -->

By combining (49) with (47) and (48), we obtain that

<!-- formula-not-decoded -->

holds for arbitrary w and w ′ . Let W ′ be a random vector independent of W and have the same distribution. By replacing w/w ′ with W/W ′ respectively, we see

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

by the triangle inequality. Taking the square of both sides of (50) and applying the AM-GM inequality gives that

<!-- formula-not-decoded -->

Let Y := ∇ 1 f ( x -B · E W [ u ( x, W )] , W ) -∇ 1 f ( x -B · E W [ u ( x, W )] , W ′ ) . Note that the righthand side of (51) can be expressed as ∥ ∥ B ⊤ Y ∥ ∥ 2 = Tr { B ⊤ ( Y Y ⊤ ) B } . By taking the expectations of both sides, we obtain that

<!-- formula-not-decoded -->

In the last inequality, we use the property that the trace of a positive semi-definite matrix equals the sum of its eigenvalues. Thus, it is greater than or equal to n times the smallest eigenvalue σ 0 σ min ( B ) 2 . Rearranging the terms finishes the proof.

## D Useful Technical Results

In this section, we state a useful result about what functions can pass the covariance of its input to the output in Theorem D.1, which is used to show Lemma C.2. We defer the proof to Appendix D.1.

Theorem D.1. Suppose that a function g : R d → R d satisfies

<!-- formula-not-decoded -->

Additionally, there exists a positive constant ℓ such that

<!-- formula-not-decoded -->

Suppose X is a random vector that satisfies | E [ X ] | &lt; ∞ and Cov [ X ] = Σ ⪰ µI . Further, there exists a constant C &gt; 0 such that for any positive integer N , X can be decomposed as X = ∑ N i =1 X i for i.i.d. random vectors X i that satisfy E [ ∥ X i ∥ 4 ] ≤ C · N -2 . Then, we have

<!-- formula-not-decoded -->

As a remark, the gradient of a well-conditioned function satisfies the conditions in (52).

## D.1 Proof of Theorem D.1

Without any loss of generality, we assume E [ X ] = 0 because we can view g ( E [ X ] + · ) as the function and subtract the mean from the random variables. The assumptions about g and X in Theorem D.1 still hold.

For any i ∈ [ d ] and ϵ ∈ R d , we have the Taylor series expansion Lagrangian form (see Chapter 3.2 of [27])

<!-- formula-not-decoded -->

where ¯ x ( i ) is a point on the line segment between x and x + ϵ . For notational convenience, let

<!-- formula-not-decoded -->

With the above notation, Eq. (54) can be equivalently written as

<!-- formula-not-decoded -->

From Eq. (53), we know that | v ( x, ϵ ) i | ≤ ℓ ∥ ϵ ∥ 2 , which implies

In addition, by Eq. (52), we see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting Eq. (55) into the above equation and rearranging the terms, we obtain which is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that the term subtracted from the right-hand side satisfies ∣ ∣ ϵ ⊤ · v 1 ( x, ϵ ) ∣ ∣ ≤ ℓ √ d ∥ ϵ ∥ 3 , which follows from Cauchy-Schwarz inequality and Eq. (56). Therefore, since the previous inequality holds for any ϵ ∈ R d , taking ϵ → 0 gives that

Before we proceed, we first state and prove a lemma that can convert the summation in Eq. (57) into a product form.

Lemma D.2. Let M ∈ R d × d be a real-valued matrix satisfying M + M ⊤ ⪰ 2 γI . Then, for any positive definite matrix Σ ⪰ µI , we have M Σ M ⊤ ⪰ µγ 2 I .

Proof of Lemma D.2. Since M + M ⊤ ⪰ 2 γI , we have for any x ∈ R d that where the last inequality follows from Σ ⪰ µI ⇒ ∥ Σ -1 / 2 x ∥ = √ x ⊤ Σ -1 x ≤ µ -1 / 2 ∥ x ∥ . Rearranging terms, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Squaring both sides concludes the proof.

Next, we state and prove a lemma about the lower bound of the covariance induced by an additive random noise on the input that is useful when the noise is sufficiently small.

Lemma D.3. Let ε be a mean-zero random vector in R d that satisfies δI ⪯ Cov [ ε ] and E [ ∥ ε ∥ 4 ] ≤ γ . Let g be a function that satisfies (52) and (53) . Then, for arbitrary fixed real vector x ∈ R d , we have

<!-- formula-not-decoded -->

Proof of Lemma D.3. We first derive bounds on the i th moment of ∥ ε ∥ ( i = 1 , 2 , 3 ). By Jensen's inequality, we have

<!-- formula-not-decoded -->

Using Jensen's inequality again, we obtain that

<!-- formula-not-decoded -->

Lastly, by the Cauchy-Schwartz inequality, we see that

<!-- formula-not-decoded -->

Note that by (55), we have

<!-- formula-not-decoded -->

Since E [ ε ] = 0 , we can further decompose (61) as

<!-- formula-not-decoded -->

By Lemma D.2 and (57), we know the first term in (62) is lower bounded by

<!-- formula-not-decoded -->

Define the residual term as the sum of the last 3 terms in (62):

<!-- formula-not-decoded -->

To show Lemma D.3, we only need to show

To see this, note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use the definition of the induced matrix norm in (66a); we use (52) and the triangle inequality in (66b); we use the Jensen's inequality and the definition of the induced matrix norm in (66c) and (66d); we use (56) in (66e); we use the bounds on the moments of ∥ ε ∥ (58), (59), and (60) in (66f).

<!-- formula-not-decoded -->

On the other hand, we know that Cov [ v 1 ( x, ε )] is a positive semi-definite matrix that satisfies

<!-- formula-not-decoded -->

-· ⪯ Thus, its induced matrix norm can be upper bounded by

<!-- formula-not-decoded -->

Using the bound of ∥ v 1 ( x, ε ) ∥ in (56) and the 4 th moment bound of ∥ ε ∥ , we obtain that

<!-- formula-not-decoded -->

Note that the norm of R (Equation (65)) can be upper bounded by the sum of the norms of the 3 separate terms. Thus, by combining the (66) and (67), we see that (65) holds.

Lastly, we consider the case when the input of g can be expressed as the sum of a sequence of mutual independent random vectors.

Lemma D.4. Let { X i } 1 ≤ i ≤ N be a sequence of mean-zero random vectors in R d that are mutually independent and satisfies δI ⪯ Cov [ X i ] and E [ ∥ X i ∥ 4 ] ≤ γ . Let g be a function that satisfies (52) and (53) . Then, for any positive integer N , we have

<!-- formula-not-decoded -->

Proof of Lemma D.4. We use an induction on N to show that (68) holds.

When N = 1 , (68) holds by setting x = 0 and ε = X 1 in Lemma D.3.

Suppose (68) holds for N -1 . Then, for N , by the law of total variance, we see that

<!-- formula-not-decoded -->

For the first term in (69), we define a new function

<!-- formula-not-decoded -->

Since the random variables { X i } 1 ≤ i ≤ N are mutually independent, we observe that

<!-- formula-not-decoded -->

One can verify that if g satisfies the conditions in (52) and (53), then ¯ g also satisfies the same conditions as g because

<!-- formula-not-decoded -->

On the other hand, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the Hessian upper/lower bounds, because ∇ 2 ¯ g i ( x ) = ∇ 2 E [ g i ( x + X N )] = E [ ∇ 2 g i ( x + X N ) ] ℓI ¯ g i ( x ) ℓI.

Therefore, by the induction assumption, we see that

<!-- formula-not-decoded -->

For the second term in (69), we note that for any realization x of ∑ N -1 i =1 X i , we have

<!-- formula-not-decoded -->

where the conditioning can be removed in the second step because the random variables { X i } 1 ≤ i ≤ N are mutually independent, so g ( x + X N ) is independent with ∑ N -1 i =1 X i ; and we use Lemma D.3 in the last inequality. Therefore, we obtain that

<!-- formula-not-decoded -->

Substituting (70) and (71) into (69) shows that (68) still holds for N . Thus, we have proved Lemma D.4 by induction.

Now we come back to the proof of Theorem D.1. By the assumption, we know the distribution of X is identical to the distribution of ∑ N i =1 X i , where X i are i.i.d. random vectors that satisfy E [ ∥ X i ∥ 4 ] ≤ C · N -2 . Thus, we have

<!-- formula-not-decoded -->

Note that each X i satisfies Cov [ X i ] = 1 N Cov [ X ] ⪰ µ N I . Applying Lemma D.4 gives that

<!-- formula-not-decoded -->

By letting N tend to infinity in the above inequality, we finish the proof of Theorem D.1.

## E Roadmap to Multi-step Prediction under Well-Conditioned Costs

A limitation of Assumption 4.7 in Section 4.1 is that it only allows the prediction V t ( θ ) to depend on the disturbance W t at time step t . A natural question is whether we can relax the assumption by allowing V t ( θ ) to depend on all future disturbances W t :( T -1) . In this section, we present a roadmap towards this generalization and discuss about the potential challenges.

First, we show that the expected cost-to-go function E [ C π θ t ( x ; Ξ) | I t ( θ ) ] can be expressed as a function that only depends on the conditional expectations W θ τ | t for all τ ≥ t , i.e., there exists a function ˜ C π θ t that satisfies

<!-- formula-not-decoded -->

,

We show (72) by induction on t = T, T -1 , . . . , 0 . Note that the statement holds for T . Suppose it holds for t +1 , by (34), we have

<!-- formula-not-decoded -->

where we use the induction assumption in the last equation. Define the random variables ε θ t | t := W t -W θ t | t and ε θ τ | t := W θ τ | ( t +1) -W θ τ | t . Using the properties of joint Gaussian distribution, we know that ε θ t :( T -1) | t are independent with I t ( θ ) . Therefore,

<!-- formula-not-decoded -->

Thus, ¯ C π θ t +1 ( x ; I t ( θ )) can be expressed as a function of x and W θ ( t +1):( T -1) | t , and we denote it as

<!-- formula-not-decoded -->

Therefore, we obtain that

<!-- formula-not-decoded -->

Therefore, E [ C π θ t ( x ; Ξ) | I t ( θ ) ] can also be expressed in the form ˜ C π θ t ( x ; W θ t :( T -1) | t ) . Thus, we have shown (72) by induction, with (73) as an intermediate result.

Note that the optimal policy is given by

<!-- formula-not-decoded -->

Therefore, by Lemma C.3, we need to establish a covariance lower bound of the gradient

<!-- formula-not-decoded -->

in order to derive a lower bound for the trace of the covariance matrix of π θ t ( x ; I t ( θ )) . While this is relatively straightforward when we only have W θ t | t because it is added directly with x , it is much more challenging to also consider the covariance caused by W θ ( t +1):( T -1) | t . This is because they affect ˜ ¯ C π θ t +1 through multiple steps of infimal convolutions. Nevertheless, we feel the approach that we describe here is promising if we can derive more properties that are preserved through the infimal convolution operators. We leave this direction as future work.
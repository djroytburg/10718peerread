## FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning

Li Zhang 1 , Zhongxuan Han 1 , Xiaohua Feng 1 , Jiaming Zhang 1 , Yuyuan Li 2 , Chaochao Chen 1 ∗ 1 Zhejiang University, 2 Hangzhou Dianzi University zhanglizl80@gmail.com, {zxhan, fengxiaohua, 22321350}@zju.edu.cn

y2li@hdu.edu.cn, zjuccc@zju.edu.cn

## Abstract

With the emerging application of Federated Learning (FL) in decision-making scenarios, it is imperative to regulate model fairness to prevent disparities across sensitive groups (e.g., female, male). Current research predominantly focuses on two concepts of group fairness within FL: Global Fairness (overall model disparity across all clients) and Local Fairness (the disparity within each client). However, the non-decomposable, non-differentiable nature of fairness criteria poses two fundamental, unresolved challenges for fair FL: (i) Harmonizing global and local fairness, especially in multi-class setting ; (ii) Enabling a controllable, optimal accuracy-fairness trade-off . To tackle these challenges, we propose a novel controllable federated group-fairness calibration framework, named FedFACT. FedFACT identifies the Bayes-optimal classifiers under both global and local fairness constraints, yielding models with minimal performance decline while guaranteeing fairness. Building on the characterization of the optimal fair classifiers, we reformulate fair federated learning as a personalized cost-sensitive learning problem for in-processing and a bi-level optimization for post-processing. Theoretically, we provide convergence and generalization guarantees for FedFACT to approach the near-optimal accuracy under given fairness levels. Extensive experiments on multiple datasets across various data heterogeneity demonstrate that FedFACT consistently outperforms baselines in balancing accuracy and global-local fairness.

## 1 Introduction

Federated learning (FL) is a collaborative distributed machine learning paradigm that allows multiple clients to jointly train a shared model while preserving the privacy of their local data [55]. As FL is increasingly adopted in high-stakes domains-healthcare [80, 65, 62], finance [15, 11, 72], pattern recognition [52, 63, 94, 86, 19], and recommender systems [10, 82, 35, 73]-ensuring fairness is imperative to prevent discrimination against demographic groups based on sensitive attributes [32, 90, 88, 64], such as race, gender, age, etc. Although a rich literature addresses group fairness in centralized settings [2, 3, 41, 13], these methods depend on full access to the entire dataset and centralized processing, imposing excessive communication overhead and elevating privacy concerns when directly applied in the FL context.

To provide fairness guarantees for federated algorithms, recent works have concentrated on two group-fairness concepts in FL: Global Fairness and Local Fairness [33, 25, 95, 18, 31]. Global fairness aims to identify a model that provides similar treatment to protected groups across the entire data distribution. Local fairness concerns models that mitigate disparities and deliver unbiased predictions for sensitive groups within each client's local data. Previous work [33] theoretically demonstrated that, under statistical heterogeneity across clients, global and local fairness can differ,

∗ Corresponding author

and both entail an inherent trade-off with predictive accuracy. As global and client-level biases can induce heterogeneous treatment disparities among sensitive groups, concurrently mitigating global and local disparities is vital for achieving group fairness in FL. For example, in constructing a federated prediction model for clinical decision-making within a hospital network [48], achieving global fairness substantially enhances performance for disadvantaged subgroups, while fairness at each hospital also carries heightened significance due to local deployment and legal requirements [21].

However, existing methods face certain challenges in controlling group fairness within FL: (i) Harmonizing global and local fairness, especially in multi-class classification. Divergent sensitive-group distributions from client heterogeneity separate global and local fairness, thereby imposing an intrinsic trade-off [33, 25]. Most fair FL approaches focus exclusively on either global or local fairness [31, 18, 93, 23, 85], thereby inevitably sacrificing the other objective and impeding the realization of both fairness criteria. Moreover, this research predominantly addresses fairness in the binary case, despite the ubiquity of multiclass tasks in practical FL scenarios. (ii) Enabling a controllable, optimal accuracy-fairness trade-off with theoretical guarantee. The non-decomposable, nondifferentiable nature of group-fairness measures poses significant optimization challenges [56, 17]. Existing frameworks typically rely on surrogate fairness losses [85, 31, 93, 18, 54], yet the inevitable surrogate-fairness gap [83, 53] produces suboptimal performance and undermines convergence stability, thus hampering the controllability of the accuracy-fairness trade-off.

To address these challenges, we propose a novel Fed erated groupFA irness C alibra T ion framework, named FedFACT, comprising in-processing and post-processing approaches. Our framework is capable of ensuring controllable global and local fairness with minimal accuracy deterioration, underpinned by provable convergence and consistency guarantees. To harmonize global and local fairness, we seek to find the optimal classifier under both dual fairness constraints in the multiclass case. To this end, specific characterizations of the federated Bayes-optimal fair classifiers are established for both the in-processing and post-processing phases in FL. Building on the Bayesoptimal fair classifier's structure, we develop efficient, privacy-preserving federated optimization strategies that realize a controllable and theoretically optimal fairness-accuracy trade-off. In detail, FedFACT reduces the in-processing task into a series of personalized cost-sensitive classification problems, and reformulates post-processing as a bi-level optimization that leverages the explicit form of the federated Bayes-optimal fair classifiers. We further derive theoretical convergence and generalization guarantees, demonstrating that our methods achieve near-optimal model performance while enforcing tunable global and local fairness constraints.

Our extensive experiments on multiple real-world datasets verify the efficiency and effectiveness of FedFACT, highlighting that FedFACT delivers a superior, controllable accuracy-fairness balance while maintaining competitive classification performance compared to state-of-the-art methods.

Our main contributions are summarized as follows:

- To the best of our knowledge, we are the first to propose a multi-class federated groupfairness calibration framework that approaches the Bayes-optimal fair classifiers, explicitly tailored to achieve a provably optimal and controllable balance between global fairness, local fairness, and accuracy.
- We further develop efficient algorithms to derive optimal classifiers under global-local fairness constraints at in-processing and post-processing stages with provable convergence and consistency guarantees. The in-processing fair classification is reduced to a sequence of personalized cost-sensitive learning problem, while the post-processing is formulated as a bi-level optimization, using the closed-form representation of Bayes-optimal fair classifiers.
- We conduct extensive experiments on multiple datasets with various data heterogeneity. The experimental results demonstrate that FedFACT outperforms existing methods in achieving superior balances among global fairness, local fairness, and accuracy. Experiments also show that FedFACT enables the flexible adjustment of the accuracy-fairness trade-off in FL.

## 2 Related Work

Group Fairness in Machine Learning. As summarized in previous work [56], group fairness is broadly defined as the absence of prejudice or favoritism toward a sensitive group based on their inherent characteristics. Common strategies for realizing group fairness in machine learning can be

classified into three categories: pre-, in-, and post-processing methods. Pre-processing [47, 42, 43] approaches aim to modify training data to eradicate underlying bias before model training. Inprocessing [45, 50, 2, 81, 83, 87, 92, 84] methods are developed to achieve fairness requirements by intervening during the training process. Post-processing [13, 20, 91, 78, 79, 37, 46] methods adjust the prediction results generated by a given model to adapt to fairness constraints after the training stage. However, because these methods require access to the full dataset, they are confined to mitigating disparities only at the local level.

Group Fairness in Federated Learning. Current methods primarily utilize in-processing and post-processing strategies to address global or local fairness issues. Concerning local fairness, prior work [12] highlights potential detrimental effects of FL on the group fairness of each clients, while [18] and [85] employ unified and personalized multi-objective optimization algorithms, respectively, to navigate the trade-off between local fairness and accuracy. Concerning global fairness, two main approaches are adaptive reweighting techniques [58, 31, 93, 1] and optimizing relaxed fairness objectives within FL [23, 75, 26], generally replacing the fairness metrics with surrogate functions.

Furthermore, previous work [33] offered a theoretical study elucidating the divergence between local and global fairness in FL, while revealing the intrinsic trade-off between these fairness objectives and accuracy. [25] formulates local and global fairness optimization into a linear program for minimal fairness cost, but it does not realize the Bayes-optimal balance between accuracy and fairness. [95] derives the Bayes-optimal classifier and decomposes the overall problem into client-specific optimizations, yet their approach applies only to binary classification in post-processing. Persistent challenges include inadequate accuracy-fairness flexibility and limited convergence guarantee in mitigating disparities within multi-class classification across various stages of FL training.

## 3 Preliminary

Notation. Denote by 1 m the m -dimensional all-ones vector and by 1 m × m the m × m all-ones matrix. Write the probability simplex as ∆ m = { q ∈ [0 , 1] m : ∥ q ∥ 1 = 1 } , and let e i be the i -th standard basis vector. Throughout this paper, bold lowercase letters denote vectors and bold uppercase letters denote matrices. For two equally sized matrices A and B , their Frobenius inner product is ⟨ A , B ⟩ = ∑ i,j a ij b ij . For positive integer n , [ n ] = { 1 , . . . , n } .

Fairness in classification. Let ( X,A,Y ) be a random tuple, where X ∈ X for some feature space X ⊆ R d , labels Y ∈ Y = [ m ] , and the discrete sensitive attribute A ∈ A . Given the randomized classifiers h : X × A → ∆ m , the prediction ̂ Y is associated with the random outputs of h defined by P ( ̂ Y = y | x ) = h y ( x ) . In this work, we generally focus on three popular groupfairness criteria-Demographic Parity (DP) [27], Equalized Opportunity (EOP) [37] and Equalized Odds [37]-in multiclass classification tasks with multiple sensitive attributes, as defined in prior works [20, 78, 79].

Group fairness in FL. A federated system consists of numerous decentralized clients, so that we consider the population data distribution represented by a jointly random tuple ( X,A,Y,K ) with total N clients. The k -th client possesses a local data dataset D k , k ∈ [ N ] . Each sample in D k is assumed to be drawn from local distribution, represented as { ( x k,i , a k,i , y k,i ) } n k i =1 , where n k represents the number of samples for client k . The Bayes score function is commonly used to characterize the performance-optimal classifier under fairness constraints [91, 13, 78, 20], which possesses a natural extension in the federated setting: η y ( x, a, c ) := P ( Y = y | X = x, A = a, K = k ) . We are interested in both local and global fairness criteria in the FL context following [33, 25, 95]:

Definition 1. (Global Fairness) The disparity regarding sensitive groups aroused by the federated model in global data distribution P ( X,A,Y ) across all clients.

Definition 2. (Local Fairness) The disparity regarding sensitive groups aroused by the federated model when evaluated on each client's data distribution P ( X,A,Y | K ) .

Confusion matrices. Confusion matrices encapsulate the information required to evaluate diverse performance metrics and assess group fairness constraints in classification tasks [81, 60, 46]. The population confusion matrix is C ∈ [0 , 1] m × m , with elements defined for i, j ∈ [ m ] as C i,j = P ( Y = i, ̂ Y = j ) . To capture both local and global fairness across multiple data distributions within

Table 1: Example of Confusion-Matrix-Based Group Fairness Constraints in Centralized &amp; Federated Learning.

| Fairness Criterion              | Demographic parity (DP)                                                                        | Equal Opportunity (EOP)                                                                               |
|---------------------------------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Group Constraints (Centralized) | ∣ ∣ ∣ P ( ̂ Y = y &#124; A = a ′ ) - P ( ̂ Y = y ) ∣ ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y                   | ∣ ∣ ∣ P ( ̂ Y = y &#124; A = a ′ ,Y = y ) - P ( ̂ Y = y &#124; Y = y ) ∣ ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y      |
| Matrix Notations (Centralized)  | ∣ ∣ ∑ a ∑ i ( I [ a = a ′ ] - p a ) C a i,y ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y                            | ∣ ∣ ∣ ∑ a ( p a ′ p a ′ ,y I [ a = a ′ ] - 1 p y ) C a y,y ∣ ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y                  |
| Global Fairness (Federated)     | ∣ ∣ ∣ ∑ a ∑ k ∑ i ( p k &#124; a ′ I [ a = a ′ ] - p a,k ) C a,k i,y ∣ ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y | ∣ ∣ ∣ ∑ a ∑ k ( p a ′ ,k p a ′ ,y I [ a = a ′ ] - p a,k p y ) C a,k i,y ∣ ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y     |
| Local Fairness (Federated)      | ∣ ∣ ∣ ∑ a ∑ i ( I [ a = a ′ ] - p a &#124; k ) C a,k i,y ∣ ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y             | ∣ ∣ ∣ ∑ a ∑ k ( p a ′ ,k p a ′ ,y,k I [ a = a ′ ] - p a,k p y,k ) C a,k i,y ∣ ∣ ∣ ∀ a ′ ∈ A , ∀ y ∈ Y |

FL, we propose the decentralized group-specific confusion matrices C a,k , a ∈ A , k ∈ [ N ] , with elements defined for i, j ∈ [ m ] as C a,k i,j ( h ) := P ( Y = i, ̂ Y = j | A = a, K = k ) .

Fairness and performance metrics. As presented in Table 1 (EO criterion and notation explanations see Appendix A), the global fairness constraints typically can be expressed by | D g u g ( h ) | ≤ ξ g , u g ∈ U g , where D g u g ( h ) = ∑ a ∑ k ⟨ D a,k u g , C a,k ( h ) ⟩ represents the constraints required to achieve the global fairness criterion. Similarly, the local fairness constraints are | D k u k ( h ) | ≤ ξ k , u k ∈ U k , k ∈ [ N ] , with D k u k ( h ) = ∑ a ⟨ D a,k u k , C a,k ( h ) ⟩ . For performance metrics, we consider a risk metric expressed as a linear function of the population confusion matrix, i.e. R ( h ) = ⟨ R , C ( h ) ⟩ = ∑ a ∑ k p a,k ⟨ R , C a,k ( h ) ⟩ . This formulation has been explored in multi-label and fair classification contexts [76, 81], and encompasses a variety of performance metrics, such as average recall and precision. In this paper, we primarily focus on standard classification error to set R = 1 m × m -I .

## 4 Methodology

## 4.1 Federated Bayes-Optimal Fair Classifier

To investigate the optimal classifier with the group fairness guarantee within FL, we consider the situation that there is a unified fairness constraint at the global level, and each client has additional local fairness restrictions in response to personalized demands. Therefore, it is appropriate to consider a personalized federated model to minimize classification risk and ensure both local and global fairness. Denoting the set of classifiers as H = { h : X → ∆ m } , the federated Bayes-optimal fair classification problem can be formulated as

<!-- formula-not-decoded -->

where ξ k , ξ g are positive bounds, D g ( h ) := { D g u g ( h ) } u g ∈U g , D k ( h ) := { D k u k ( h ) } u k ∈U k , and the inequality applies element-wise. FL model h = ( h 1 , . . . , h N ) comprises N local classifiers.

Before delving into the optimal solution for Problem (1), we present a formal result on the structure of the federated Bayes-optimal fair classifier. Proposition 1 indicates that the Bayes-optimal classifier can be decomposed into local deterministic classifiers for all clients. This observation provides valuable insights for the subsequent algorithm design. The proof and the discussion on feasibility are given in Appendix B.1.

Proposition 1. If (1) is feasible for any positive ξ k and ξ g , the client-wise classifier h ∗ k in federated Bayes-optimal fair classifier h ∗ = ( h ∗ 1 , . . . , h ∗ N ) can be expressed as the linear combination of some deterministic classifiers { h ′ k,i } d k i =1 , i.e., h ∗ k ( x ) = ∑ d k i =1 α k,i h ′ k,i ( x ) , α k ∈ ∆ d k .

## 4.2 In-processing Fair Federated Training via Cost Sensitive Learning

In this section, we aim to seek for the optimal solution of h = ( h 1 , . . . , h N ) , where each local classifier is attribute-blind and parameterized by ϕ k . A direct approach to solving (1) is to formulate an equivalent convex-concave saddle point problem in terms of its Lagrangian L ( h , λ, µ ) :

<!-- formula-not-decoded -->

where λ = { λ (1) , λ (2) } and µ = { µ (1) k , µ (2) k } N k =1 are the dual parameters. Let Λ := { λ ∈ R 2 |U g | ≥ 0 : ∥ λ ∥ 1 ≤ B d } and M := { µ ∈ R 2 ∑ k |U k | ≥ 0 : ∥ µ ∥ 1 ≤ B d } . Since h is random classifier, by Sion's minimax theorem [66], the primal problem can be written as a saddle-point optimization

<!-- formula-not-decoded -->

The boundedness of optimal λ ∗ , µ ∗ will be shown later. To derive the representation of optimal saddle point, we initially focus on the inner minimization optimization task, namely min h ∈H L ( h , λ, µ ) .

Proposition 2. Given non-negative λ and µ , then an optimal solution h ∗ = ( h ∗ 1 , . . . , h ∗ N ) to the inner problem min h ∈H L ( h , λ, µ ) is realized by local deterministic classifiers h ∗ k ( x ) , k ∈ [ N ] satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Proposition 2 is given in Appendix B.2. Notice that the optimal solution in (3) remains computationally intractable, because the point-wise distributions P ( A = a | x, k ) and the Bayesoptimal classifier η are unknown. We reformulate the task of solving h ∗ within a cost-sensitive learning framework, by designing sample-wise calibrated training losses for each h ∗ k ( x ) that can yield an equivalent objective.

Proposition 3. Let the personalized cost-sensitive loss for client k be defined by

<!-- formula-not-decoded -->

where s : X → R m is the scoring function, and M λ,µ ( a, k ) = M λ,µ ( a, k ) + κ 1 m × m with κ chosen to ensure all matrix entries are strictly positive. Denoting the optimal scoring function to minimize ℓ k over the local data distribution as s ∗ k ( x ) , then the loss ℓ k is calibrated for the inner problem min h ∈H L ( h , λ, µ ) , i.e., h ∗ k ( x ) = e y , y ∈ arg max j ∈ [ m ] [ s ∗ k ( x )] j is equivalent to that in (3) .

In practice, s is parameterized by ϕ k for k ∈ [ N ] , formulating the personalized optimization objective L k ( f ϕ k ) = ∑ n k i =1 ℓ k ( y k,i , s ( x k,i ; ϕ k ) , a k,i ) in the FL setting. Appendix B.3 provides the proof of Proposition 3 and further presents that the loss ℓ k in (4) is also calibrated for the unified federated Bayes-optimal fair classifier. Inspired by this property, we propose an efficient in-processing algorithm for group-fair classification within FL, as detailed in Algorithm 1.

At each iteration t , the personalized classifier h t k is obtained by ensembling the unified model θ t with the local model ϕ t k . The ensemble weight w t k and its update rule balance the contributions of the unified and local models. The following theorem establishes the personalized regret bound w.r.t. the best model parameter, and further demonstrates that our algorithm achieves an ϵ -approximate stochastic saddle point.

Theorem 4. Under mild assumptions, there exist constants B k , B L , such that the following cumulative regret upper bound is guaranteed for the ensemble personalized models:

<!-- formula-not-decoded -->

Furthermore, suppose that personalized models achieve a ρ t -approximate optimal response at iteration t , namely ̂ L ( h t , λ t , µ t ) ≤ min h ̂ L ( h , λ t , µ t ) + ρ t , denoting ¯ ρ T = ∑ T t =1 ρ t /T , then the sequences of model and bounded dual parameters comprise an approximate mixed Nash equilibrium:

<!-- formula-not-decoded -->

The complete Theorem 4 with its proof is provided in the Appendix B.4. The regret bound yields a convergence rate of O (1 / √ T ) by appropriately choosing the learning rate, reflecting the stability of the proposed algorithm. Moreover, as ρ t decreases with t , the algorithm will gradually converge to the optimal equilibrium. For instance, if ρ t ∝ C/ √ t , ϵ will also exhibit an O (1 / √ T ) convergence rate. The generalization error between the optimal solutions of the empirical dual and primal problems under finite samples is given in Appendix B.5 .

## Algorithm 1: FedFACT (In-processing)

```
Input : Datasets { x k,i , y k,i , a k,i } N k i =1 from client k , k ∈ K ; Communication round T ; Local round R ; Initial parameters λ 0 , µ 0 , θ 0 , ϕ 0 k , w 0 k , k ∈ K ; Learning rate { η, η d , η w } T t =1 ; for t = 0 , 1 , . . . , T do Each client k ∈ K in parallel do: Ensemble unified and local model: h t k ( x ) = e y , y ∈ arg max j ∈ [ m ] [ f t k,ens ( x )] j , where f t k,ens ( x ) = w t k f θ t ( x ) + (1 -w t k ) f ϕ t k ( x ) and f ϕ ( x ) := softmax( s ( x ; ϕ )) ; Update calibration matrix M λ t ,µ t k = ̂ M λ t ,µ t k + κ 1 m × m ; Update the weight w t +1 k = 1 1+ W t k ( w t k ) , W t k ( w t k ) = 1 -w t k w t k exp( -η w [ L k ( f ϕ t k ) -L k ( f θ t )]) ; Calculate update of global dual parameter ∆ λ t +1 k , and update local dual parameter µ t +1 k , ∆ λ ( i ) ,t +1 k,u g = (3 -2 i ) ∑ a ∈A 〈 ̂ D a,k u g , ̂ C a,k ( h t k ) 〉 -ξ g , i ∈ [1 , 2] , u g ∈ U g , µ ( i ) ,t +1 k,u k = Π M [ (3 -2 i ) ∑ a ∈A 〈 ̂ D a,k u k , ̂ C a,k ( h t k ) 〉 -ξ k ] , i ∈ [1 , 2] , u k ∈ U k ; Perform R local-batch update of θ t and ϕ t k , guided by loss L k with M λ t ,µ t k , θ t,r +1 k = θ t,r k -η k ∇ L k ( θ t,r ; B t,r k ) , ϕ t,r +1 k = ϕ t,r k -η k ∇ L k ( ϕ t,r ; B t,r k ) , r = 0 , · · · , R -1 ; Send last update ∆ θ t +1 k = θ t,R k -θ t to the server; Server do: Server aggregates { ∆ θ t +1 k } : θ t +1 = θ t + ∑ N k =1 p k ∆ θ t +1 k ; Update global dual parameter: λ t +1 = Π Λ ( λ t + η d ∑ N k =1 ∆ λ t +1 k ) ; Send θ t +1 , λ t +1 to clients; end Return Personalized classifier h = ( ¯ h 1 , · · · , ¯ h N ) , where ¯ h k := ∑ T t =1 h t k /T ;
```

## 4.3 Label-Free Federated Post-Fairness Calibration based on Plug-In Approach

This section introduces a post-hoc fairness approach that calibrates the classification probabilities of a pre-trained federated model. We formulate an closed-form representation of the federated Bayes optimal fair classifier under standard assumptions, and then derive the primal problem into bi-level optimization through the plug-in approach. To begin, we introduce the following assumption.

Assumption 1. ( η -continuity). For each client k , denoting P X a,k := P ( X | A = a, K = k ) , let the put forward distribution τ a,k := η ♯ P X a,k , a ∈ A be absolutely continuous with respect to the Lebesgue measure restricted to ∆ m .

Assumption 1, which can be met by adding minor random noises to τ a,k , is commonly used in the literature on post-processing fairness [13, 81, 20, 16]. Next, we derive a more explicit characterization of federated Bayes-optimal fair classifier.

Theorem 5. Under Assumption 1, suppose that the primal problem (1) is feasible for any ξ g , ξ k &gt; 0 and all non-zero columns of each D a,k u are distinct, the attribute-aware personalized classifier { h ∗ k } k ∈ [ N ] is Bayes-optimal under local and global fairness constraints, if h ∗ k ( x, a ) = e y , y ∈ arg max j ∈ [ m ] ( [ M λ ∗ ,µ ∗ ( a, k ) ] ⊤ η ( x, a, k ) ) j , where the dual parameters are determined from ( λ ∗ , µ ∗ ) ∈ arg min λ,µ ≥ 0 H ( λ, µ ) ,

<!-- formula-not-decoded -->

The optimal dual parameters ( λ ∗ , µ ∗ ) are bounded, and the optimality of λ and µ respectively guarantee global fairness and local fairness.

Proof of Theorem 5 is given in Appendix B.6. Since the dual parameter µ only related to local fairness constraints, each clients can finish update of this parameter without global aggregation. Therefore,

the federated Bayes-optimal classification problem can be reformulated into a bi-level optimization:

<!-- formula-not-decoded -->

where ̂ H k ( λ, µ k ) := 1 n k ∑ n k i =1 max y ∈ [ m ] ([ M λ,µ ( a i , k ) ] ⊤ η ( x i , a i , k ) ) y + ξ g ∥ λ ∥ 1 + ξ k ˆ p k ∥ µ k ∥ 1 is the plug-in estimation of (7). Considering that the non-smoothness of the optimization objective may lead to convergence issues in federated optimization [89], we replace the maximum operation in ̂ H k ( λ, µ k ) with soft-max weight function σ β ( x ) = ∑ m i =1 exp( x i /β ) ∑ j m =1 exp( x j /β ) x i , which reduces to the hard-maximum if temperature β → 0 . Denoting the relaxed local objective as ̂ H ′ k ( λ, µ k ) , we propose Algorithm 2 to solve the federated Bayes-optimal fair classifier.

```
Algorithm 2: FedFACT (Post-processing) Input : Datasets D k = { x k,i , y k,i , a k,i } N k i =1 from client k ∈ [ N ] ; Communication round T ; Local round R ; Initial parameters λ 0 , µ 0 , k ∈ [ N ] ; Learning rate η d ; Pre-trained ̂ η for t = 0 , 1 , . . . , T do Each client k ∈ [ N ] in parallel do: Perform R local-batch update of µ t k , i = 1 , 2 with ̂ H ′ k ( λ, µ k ) : µ t,r +1 k = Π M k ( µ t,r k -η d ∇ µ ̂ H ′ k ( λ, µ k )) , r = 0 , · · · , R -1 . Set µ t +1 k = µ t,R k , and calculate update of λ t : ∆ λ t +1 k = ∇ λ ̂ H ′ k ( λ t , µ t +1 k ) . Server do: Server aggregates { ∆ λ t +1 k } : λ t +1 = Π Λ ( λ t -η d ∑ N k =1 ˆ p k ∆ λ t +1 k ) ; Send λ t +1 to clients; end Return Classifiers { h 1 , . . . , h N } , h k ( x, a ) := arg max j ∈ [ m ] ([ ̂ M λ ∗ ,µ ∗ ( a, k ) ] ⊤ ̂ η ( x, a, k ) ) j ;
```

Proposition 6. The bi-level objectives ̂ H ′ k ( λ, µ k ) , k ∈ [ N ] are convex and L-smooth.

Proof of Proposition 6 is given in Appendix B.7. Existing research in FL [74] shows that the L -smoothness of the local objective suffice for Algorithm 2 to achieve an O (1 / √ T ) convergence rate. Moreover, owing to the equivalence of nested and joint minimization under convexity [66, 36], the corresponding bi-level optimization can approach the optimal solution of the empirical primal problem. Consequently, the remaining error arises from the generalization risk induced by finite sampling, which is explored in Appendix B.8 .

## 4.4 Discussion

In- versus post-processing. In- and post-processing interventions play complementary roles: the former removes bias in representations during training (incurring higher computational cost), and the latter adjusts fairness on model outputs with low overhead, unable to debias learned representations. Both of them support adaptable fairness calibration in resource-limited, heterogeneous FL. Note that combining the in-processing and post-processing methods is theoretically unjustifiable, as the in-processing classifier is not designed to approximate the Bayes score function.

Efficiency &amp; Privacy. Each iteration of our in- and post-processing methods requires only a single client-server interaction and is supported by convergence guarantees that demonstrate our algorithms' efficiency. This will be empirically validated in our experiments; FedFACT is also privacy-friendly. In-processing requires sharing λ alongside the unified model θ , while post-processing involves sharing only λ . These exchanges conform to standard FL [55] and preserve data confidentiality. Furthermore, differential privacy [28, 30] or encryption schemes [7] can be applied to further reinforce privacy.

## 5 Experiments

To comprehensively assess the proposed FedFACT framework, we conduct extensive experiments on four publicly available real-world datasets to answer the following Research Questions (RQ): RQ1 :

Table 2: Overall Experimental Results.

|           | Dataset              | Compas   | Compas   | Compas   | Adult   | Adult    | Adult   | CelebA   | CelebA   | CelebA   | ENEM   | ENEM     | ENEM    |
|-----------|----------------------|----------|----------|----------|---------|----------|---------|----------|----------|----------|--------|----------|---------|
| Partition | Method               | Acc      | D global | D local  | Acc     | D global | D local | Acc      | D global | D local  | Acc    | D global | D local |
|           | FedAvg               | 69.73    | 0.2766   | 0.3590   | 84.52   | 0.1765   | 0.2310  | 89.14    | 0.1435   | 0.1308   | 67.56  | 0.2620   | 0.2462  |
|           | FairFed              | 59.39    | 0.1008   | 0.1022   | 80.73   | 0.0983   | 0.1434  | 81.85    | 0.0704   | 0.1058   | 60.99  | 0.1165   | 0.1733  |
|           | FedFB                | 58.09    | 0.0879   | 0.0983   | 81.85   | 0.0751   | 0.1165  | 85.32    | 0.1188   | 0.0949   | 64.35  | 0.0814   | 0.1326  |
|           | FCFL                 | 56.53    | 0.0646   | 0.0614   | 81.91   | 0.0845   | 0.1455  | 83.83    | 0.0704   | 0.1090   | 61.07  | 0.1260   | 0.1189  |
|           | praFFed              | 59.93    | 0.0824   | 0.0968   | 80.96   | 0.0591   | 0.0763  | 85.45    | 0.0731   | 0.0862   | 62.60  | 0.0736   | 0.0806  |
| γ = 0 . 5 | Cost                 | 64.51    | 0.0585   | 0.0941   | 81.09   | 0.0262   | 0.0590  | 85.60    | 0.0314   | 0.0577   | 65.79  | 0.0487   | 0.0674  |
|           | FedFACT g (In)       | 61.19    | 0.0344   | 0.0761   | 82.05   | 0.0015   | 0.0408  | 86.49    | 0.0205   | 0.0544   | 63.79  | 0.0493   | 0.0568  |
|           | FedFACT l (In)       | 61.81    | 0.0600   | 0.0636   | 82.44   | 0.0140   | 0.0508  | 85.90    | 0.0461   | 0.0312   | 63.93  | 0.0434   | 0.0487  |
|           | FedFACT g & l (In)   | 61.17    | 0.0407   | 0.0732   | 82.04   | 0.0014*  | 0.0401  | 86.15    | 0.0382   | 0.0473   | 62.51  | 0.0366   | 0.0413  |
|           | FedFACT g (Post)     | 67.27    | 0.0128*  | 0.0660   | 82.83*  | 0.0173   | 0.0276  | 87.25    | 0.0089*  | 0.0253   | 66.15  | 0.0175   | 0.0181* |
|           | FedFACT l (Post)     | 67.49*   | 0.0315   | 0.0552*  | 82.79   | 0.0154   | 0.0267* | 87.36*   | 0.0127   | 0.0163*  | 66.54* | 0.0197   | 0.0240  |
|           | FedFACT g & l (Post) | 67.33    | 0.0139   | 0.0641   | 82.74   | 0.0134   | 0.0274  | 87.06    | 0.0093   | 0.0172   | 66.52  | 0.0162*  | 0.0184  |
|           | FedAvg               | 69.12    | 0.2513   | 0.4044   | 85.34   | 0.1596   | 0.2358  | 89.85    | 0.1360   | 0.1742   | 67.45  | 0.2037   | 0.3068  |
|           | FairFed              | 61.87    | 0.1825   | 0.2448   | 81.85   | 0.1074   | 0.1283  | 82.50    | 0.0672   | 0.1415   | 59.81  | 0.0949   | 0.1624  |
|           | FedFB                | 60.16    | 0.1284   | 0.1332   | 81.14   | 0.0949   | 0.1005  | 86.00    | 0.1121   | 0.1284   | 63.68  | 0.0780   | 0.2039  |
|           | FCFL                 | 59.96    | 0.1498   | 0.1507   | 79.10   | 0.0528   | 0.0596  | 84.50    | 0.0657   | 0.1458   | 61.05  | 0.1134   | 0.1425  |
|           | praFFed              | 60.42    | 0.0902   | 0.1062   | 80.12   | 0.0523   | 0.0606  | 85.13    | 0.0592   | 0.1152   | 60.84  | 0.0932   | 0.1246  |
| Hetero    | Cost                 | 63.01    | 0.0773   | 0.1044   | 81.04   | 0.0286   | 0.0567  | 86.28    | 0.0495   | 0.0761   | 62.11  | 0.0315   | 0.0730  |
|           | FedFACT g (In)       | 60.33    | 0.0665   | 0.0841   | 82.09*  | 0.0122   | 0.0962  | 86.18    | 0.0188   | 0.0731   | 62.05  | 0.0380   | 0.0577  |
|           | FedFACT l (In)       | 60.22    | 0.0730   | 0.0753   | 81.19   | 0.0208   | 0.0250  | 86.58    | 0.0424   | 0.0426   | 61.96  | 0.0471   | 0.0473  |
|           | FedFACT g & l (In)   | 61.44    | 0.0676   | 0.0789   | 81.19   | 0.0055   | 0.0239* | 86.38    | 0.0355   | 0.0634   | 61.48  | 0.0322   | 0.0364  |
|           | FedFACT g (Post)     | 64.36    | 0.0398*  | 0.0699   | 81.09   | 0.0047*  | 0.0257  | 87.95    | 0.0094   | 0.0344   | 65.32  | 0.0199*  | 0.0343  |
|           | FedFACT l (Post)     | 64.38    | 0.0479   | 0.0740   | 81.27   | 0.0049   | 0.0306  | 88.05*   | 0.0112   | 0.0217*  | 65.68* | 0.0246   | 0.0120* |
|           | FedFACT g & l (Post) | 64.41*   | 0.0408   | 0.0680*  | 81.31   | 0.0053   | 0.0293  | 87.75    | 0.0088*  | 0.0247   | 65.19  | 0.0201   | 0.0131  |

Does FedFACT outperform the existing methods in effectively achieving a global-local accuracyfairness balance? RQ2 : Is FedFACT capable of adjusting the trade-off between accuracy and global-local fairness (sensitivity analysis)? RQ3 : How do important hyper-parameters influence the performance of FedFACT? RQ4 : How about the communication efficiency and scalability of FedFACT?

## 5.1 Datasets and Experimental Settings

Due to space limitations, the detailed information in this section is provided in Appendix C .

Datasets. Experiments are conducted on four real-world datasets: Compas [22], Adult [4], CelebA [96], and ENEM [40], which are well established for assessing fairness in FL [31, 12, 24, 95].

Baselines. For binary classification, experiments are conducted on all four datasets. We compare our method with traditional federated baselines FedAvg [55] and five state-of-the-art methods tailored for addressing global and local fairness within FL, namely FairFed [31], FedFB [93], FCFL [18], praFFed [85] and the method in [25], denoted as Cost in our experiments. The reason for we did not include the experiments with [95] is explained in Appendix C.2. For multi-group or multiclass classification, the experiments are implemented on CelebA and ENEM datasets due to label limitations.

Data distribution. To model the statistical heterogeneity in the FL context, we investigate two data partitioning strategies: (i) Dirichlet partition : we control the distribution of the sensitive attribute at each client using a Dirichlet distribution Dir ( γ ) as proposed by [31]. A smaller γ indicates greater heterogeneity across clients. (ii) Heterogeneous split : Inspired by [33], we propose a partitioning method that introduces heterogeneous correlations between the sensitive attribute A and label Y . The correlation between A and Y for each client is controlled by a parameter randomly sampled from [0 , 1] , as detailed in Appendix C.3.

Evaluation. (i) Firstly , we partition each dataset into a 60% training set and the remaining 40% for test set. (ii) Secondly , the number of clients is set to 2 in Compas, and 5 in other datasets to ensure sufficient samples for local fairness estimation. (iii) Thirdly , we evaluate the FL model with Accuracy (Acc), global fairness metric ( D global ), and maximal local fairness metric among clients ( D local ), with smaller values of fairness metrics indicating a fairer FL model.

## 5.2 Overall Comparison (RQ1)

We perform extensive experiments comparing FedFACT against existing fair FL baselines under varying statistical heterogeneity. We set ξ g = ξ k = 0 . 01 for FedFACT. The subscript g and l

Table 3: Accuracy-Fairness Balance (Sensitivity Analysis).

| Dataset       | Compas (In-)   | Compas (In-)   | Compas (In-)   | Adult (In-)   | Adult (In-)   | Adult (In-)   | Compas (Post-)   | Compas (Post-)   | Compas (Post-)   | Adult (Post-)   | Adult (Post-)   | Adult (Post-)   |
|---------------|----------------|----------------|----------------|---------------|---------------|---------------|------------------|------------------|------------------|-----------------|-----------------|-----------------|
| ( ξ g , ξ l ) | Acc            | D global       | D local        | Acc           | D global      | D local       | Acc              | D global         | D local          | Acc             | D global        | D local         |
| (0.00,0.00)   | 61.17          | 0.0407         | 0.0732         | 82.04         | 0.0014        | 0.0401        | 67.33            | 0.0139           | 0.0641           | 82.74           | 0.0134          | 0.0274          |
| (0.02,0.00)   | 61.39          | 0.0548         | 0.0848         | 82.37         | 0.0028        | 0.0458        | 67.49            | 0.0315           | 0.0552           | 82.75           | 0.0154          | 0.0255          |
| (0.04,0.00)   | 61.81          | 0.0600         | 0.0836         | 82.44         | 0.0140        | 0.0508        | 67.92            | 0.0557           | 0.0692           | 82.83           | 0.0173          | 0.0276          |
| (0.00,0.02)   | 61.23          | 0.0418         | 0.0732         | 82.04         | 0.0018        | 0.0409        | 67.40            | 0.0134           | 0.0658           | 82.76           | 0.0139          | 0.0278          |
| (0.02,0.02)   | 61.50          | 0.0569         | 0.0895         | 82.41         | 0.0056        | 0.0450        | 67.93            | 0.0558           | 0.0624           | 82.77           | 0.0166          | 0.0262          |
| (0.04,0.02)   | 61.63          | 0.0665         | 0.0933         | 82.52         | 0.0080        | 0.0479        | 67.95            | 0.0623           | 0.0598           | 82.81           | 0.0150          | 0.0256          |
| (0.00,0.02)   | 61.23          | 0.0418         | 0.0732         | 82.04         | 0.0014        | 0.0408        | 67.27            | 0.0128           | 0.0660           | 82.79           | 0.0154          | 0.0267          |
| (0.02,0.04)   | 61.66          | 0.0556         | 0.0919         | 82.57         | 0.0089        | 0.0442        | 67.95            | 0.0536           | 0.0644           | 82.81           | 0.0174          | 0.0283          |
| (0.04,0.04)   | 62.39          | 0.0720         | 0.1105         | 82.66         | 0.0223        | 0.0449        | 68.03            | 0.0645           | 0.0598           | 82.81           | 0.0185          | 0.0278          |

represent the presence of global and local fairness constraints, respectively. It is essential to note that there is an inherent trade-off between accuracy and global-local fairness.

Binary classification results. We compare FedFACT with benchmarks tailored to binary classification in terms of the binary DP and EOP criteria on the four datasets. The results of DP are presented in Table 2. We also report the Pareto frontier in Appendix D.1 to evaluate the ability of FedFACT to strike a accuracy-fairness balance, along with the Pareto results of binary EOP criterion. Overall, when compared to existing SOTA methods, FedFACT demonstrates superior performance in achieving a balanced trade-off between accuracy and fairness.

Multi-Class results. We illustrate how FedFACT performs on multi-class prediction using CelebA and ENEM with DP and EOP constraints. As there are no established methods addressing multi-class fairness in federated learning, we conduct comparisons only between FedAvg and our proposed inand post-processing approaches, as shown in Figure 1. More experimental details are in Appendix D.2.

Figure 1: Multi-Class Fair Classification Results. The top line depict global and local multiclass Demographic Parity (DP) results, while the bottom line show global and local multiclass Equal Opportunity (EOP) outcomes.

<!-- image -->

The outcomes of the multiclass experiments do not parallel those in the binary setting, where postprocessing methods vastly outperform alternatives; instead, performance is comparatively lower. This can be attributed to the paucity of local samples for fairness evaluation at individual clients, which causes post-processing under joint global and local fairness constraints to incur significant generalization error and thus fail to precisely enforce local fairness. Under these conditions, the in-training approach, leveraging globally aggregated data, offers superior fairness calibration, thereby underscoring the complementarity of the two methods we introduce.

## 5.3 Flexibility of Adjusting Accuracy-Fairness Trade-Off (RQ2)

To investigate the capability of FedFACT in adjusting accuracy-fairness trade-off, we examine the Acc, D global and D local under different fairness relaxation of ( ξ g , ξ l ) with γ = 0 . 5 on Adult and Compas in Table 3. Here we set the local fairness levels ξ k for each client to the same value, denoted as ξ l . More experimental results are presented in Appendix D.3.

Sensitivity Analysis. Table 3 shows that, for a fixed global constraint ξ g , reducing ξ l diminishes both accuracy and local fairness-implying that stricter local fairness comes at the cost of overall performance. Conversely, by keeping ξ l constant, one can modulate global fairness via adjustments to ξ g . Note that the difference between the constraints and the fairness metrics arises due to the unavoidable generalization error with finite samples. In general, these findings substantiate our claim that FedFACT enables flexible control over the accuracy-fairness trade-off in FL.

## 5.4 Hyper-parameter Experiments (RQ3)

There is no tunable hyper-parameter in our proposed method except for the number of deterministic classifiers utilized to construct the weight classifiers. We gradually raise the number of classifiers forming the weighted classifier, starting with the most recent one and extending to the previous 10 classifiers. The detailed experimental results are provided in Appendix D.4 .

## 5.5 Efficiency and Scalability Study (RQ4)

In Appendix D.5 , we undertake extensive experiments to empirically demonstrate the communication efficiency and scalability to client number of the proposed method FedFACT.

## 6 Conclusion

This paper introduces a novel controllable Fed erated groupFA irness C alibra T ion framework called FedFACT, to ensure both group and local fairness within FL. FedFACT is proposed to learn the federated Bayes-optimal fair classifier in both in- and post-processing stages, which achieves a theoretically minimal accuracy loss with both fairness constraints. We developed efficient algorithms-with convergence and consistency guarantees-that reduce fair classification to personalized cost-sensitive learning for in-pocessing and bi-level optimization for post-processing. Extensive experiments on four publicly available real-world datasets demonstrate that FedFACT outperforms SOTA methods, exhibiting a remarkable ability to harmonious balance between accuracy and global-local fairness.

## Reproducibility Statement

Details for the experimental setting are provided in the beginning of Section 5 and Appendix C, and the code can be found at https://github.com/liizhang/FedFACT.

## Acknowledgments

This work was supported in part by the Hangzhou Key Scientific Research Plan (No. 2024SZD1A28), and National Natural Science Foundation of China (No. 62402148).

## References

- [1] Annie Abay, Yi Zhou, Nathalie Baracaldo, Shashank Rajamoni, Ebube Chuba, and Heiko Ludwig. Mitigating bias in federated learning, 2020.
- [2] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. A reductions approach to fair classification. In International conference on machine learning , pages 60-69. PMLR, 2018.
- [3] Wael Alghamdi, Hsiang Hsu, Haewon Jeong, Hao Wang, Peter Michalak, Shahab Asoodeh, and Flavio Calmon. Beyond adult and compas: Fair multi-class prediction via information projection. Advances in Neural Information Processing Systems , 35:38747-38760, 2022.
- [4] Arthur Asuncion, David Newman, et al. Uci machine learning repository, 2007.
- [5] Richard Berk, Hoda Heidari, Shahin Jabbari, Michael Kearns, and Aaron Roth. Fairness in criminal justice risk assessments: The state of the art. Sociological Methods &amp; Research , 50(1):3-44, 2021.

- [6] Dimitri P. Bertsekas. Stochastic optimization problems with nondifferentiable cost functionals. Journal of Optimization Theory and Applications , 12:218-231, 1973.
- [7] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacy-preserving machine learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security , CCS '17, page 1175-1191, New York, NY, USA, 2017. Association for Computing Machinery.
- [8] Stéphane Boucheron, Olivier Bousquet, and Gábor Lugosi. Theory of classification: A survey of some recent advances. ESAIM: probability and statistics , 9:323-375, 2005.
- [9] Stephen P Boyd and Lieven Vandenberghe. Convex optimization . Cambridge university press, 2004.
- [10] Robin Burke. Multisided fairness for recommendation. arXiv preprint arXiv:1707.00093 , 2017.
- [11] David Byrd and Antigoni Polychroniadou. Differentially private secure multi-party computation for federated learning in financial applications. In Proceedings of the first ACM international conference on AI in finance , pages 1-9, 2020.
- [12] Hongyan Chang and Reza Shokri. Bias propagation in federated learning. arXiv preprint arXiv:2309.02160 , 2023.
- [13] Wenlong Chen, Yegor Klochkov, and Yang Liu. Post-hoc bias scoring is optimal for fair classification. In The Twelfth International Conference on Learning Representations , 2024.
- [14] Jaewoong Cho, Gyeongjo Hwang, and Changho Suh. A fair classifier using kernel density estimation. In Proceedings of the 34th International Conference on Neural Information Processing Systems , NIPS '20, Red Hook, NY, USA, 2020. Curran Associates Inc.
- [15] Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big data , 5(2):153-163, 2017.
- [16] Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, Luca Oneto, and Massimiliano Pontil. Leveraging labeled and unlabeled data for consistent fair binary classification. Advances in Neural Information Processing Systems , 32, 2019.
- [17] Andrew Cotter, Heinrich Jiang, Maya Gupta, Serena Wang, Taman Narayan, Seungil You, and Karthik Sridharan. Optimization with non-differentiable constraints with applications to fairness, recall, churn, and other goals. Journal of Machine Learning Research , 20(172):1-59, 2019.
- [18] Sen Cui, Weishen Pan, Jian Liang, Changshui Zhang, and Fei Wang. Addressing algorithmic disparity and performance inconsistency in federated learning. Advances in Neural Information Processing Systems , 34:26091-26102, 2021.
- [19] Ittai Dayan, Holger R Roth, Aoxiao Zhong, Ahmed Harouni, Amilcare Gentili, Anas Z Abidin, Andrew Liu, Anthony Beardsworth Costa, Bradford J Wood, Chien-Sung Tsai, et al. Federated learning for predicting clinical outcomes in patients with covid-19. Nature medicine , 27(10):1735-1743, 2021.
- [20] Christophe Denis, Romuald Elie, Mohamed Hebiri, and François Hu. Fairness guarantees in multi-class classification with demographic parity. Journal of Machine Learning Research , 25(130):1-46, 2024.
- [21] Department of Health and Human Services, Centers for Medicare &amp; Medicaid Services, Office of the Secretary. Nondiscrimination in health programs and activities. Federal Register, vol. 89, no. 87, pp. 37522-37703, May 6, 2024. Document No. 2024-08711.
- [22] Julia Dressel and Hany Farid. The accuracy, fairness, and limits of predicting recidivism. Science advances , 4(1):eaao5580, 2018.

- [23] Wei Du, Depeng Xu, Xintao Wu, and Hanghang Tong. Fairness-aware agnostic federated learning. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM) , pages 181-189. SIAM, 2021.
- [24] Yuying Duan, Yijun Tian, Nitesh Chawla, and Michael Lemmon. Post-fair federated learning: Achieving group and community fairness in federated learning via post-processing. arXiv preprint arXiv:2405.17782 , 2024.
- [25] Yuying Duan, Gelei Xu, Yiyu Shi, and Michael Lemmon. The cost of local and global fairness in federated learning. arXiv preprint arXiv:2503.22762 , 2025.
- [26] Gerry Windiarto Mohamad Dunda and Shenghui Song. Fairness-aware federated minimax optimization with convergence guarantee, 2024.
- [27] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference , pages 214-226, 2012.
- [28] Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science , 9(3-4):211-407, 2014.
- [29] Ahmad-Reza Ehyaei, Golnoosh Farnadi, and Samira Samadi. Wasserstein distributionally robust optimization through the lens of structural causal models and individual fairness. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems , volume 37, pages 42430-42467. Curran Associates, Inc., 2024.
- [30] Ahmed El Ouadrhiri and Ahmed Abdelhadi. Differential privacy for deep and federated learning: A survey. IEEE access , 10:22359-22380, 2022.
- [31] Yahya H Ezzeldin, Shen Yan, Chaoyang He, Emilio Ferrara, and A Salman Avestimehr. Fairfed: Enabling group fairness in federated learning. In Proceedings of the AAAI conference on artificial intelligence , volume 37, pages 7494-7502, 2023.
- [32] Laura Gustafson, Chloe Rolland, Nikhila Ravi, Quentin Duval, Aaron Adcock, Cheng-Yang Fu, Melissa Hall, and Candace Ross. Facet: Fairness in computer vision evaluation benchmark. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) , pages 20313-20325, 2023.
- [33] Faisal Hamman and Sanghamitra Dutta. Demystifying local &amp; global fairness trade-offs in federated learning using partial information decomposition. In The Twelfth International Conference on Learning Representations , 2024.
- [34] Xiaotian Han, Jianfeng Chi, Yu Chen, Qifan Wang, Han Zhao, Na Zou, and Xia Hu. FFB: A fair fairness benchmark for in-processing group fairness methods. In The Twelfth International Conference on Learning Representations , 2024.
- [35] Zhongxuan Han, Chaochao Chen, Xiaolin Zheng, Li Zhang, and Yuyuan Li. Hypergraph convolutional network for user-oriented fairness in recommender systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval , pages 903-913, 2024.
- [36] Filip Hanzely and Peter Richtárik. Federated learning of a mixture of global and local models. arXiv preprint arXiv:2002.05516 , 2020.
- [37] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems , 29, 2016.
- [38] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [39] Xiaolin Hu, Shaojie Li, and Yong Liu. Generalization bounds for federated learning: Fast rates, unparticipating clients and unbounded losses. In The Eleventh International Conference on Learning Representations , 2023.

- [40] COMITÊ DE ÉTICA INEP. Instituto nacional de estudos e pesquisas educacionais anísio teixeira. Boletim de Serviço Eletrônico em , 30:04, 2018.
- [41] Nikola Jovanovi´ c, Mislav Balunovic, Dimitar Iliev Dimitrov, and Martin Vechev. Fare: Provably fair representation learning with practical certificates. In International Conference on Machine Learning , pages 15401-15420. PMLR, 2023.
- [42] Nikola Jovanovi´ c, Mislav Balunovic, Dimitar Iliev Dimitrov, and Martin Vechev. FARE: Provably fair representation learning with practical certificates. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 15401-15420. PMLR, 23-29 Jul 2023.
- [43] Jian Kang, Jingrui He, Ross Maciejewski, and Hanghang Tong. Inform: Individual fairness on graph mining. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining , KDD '20, page 379-389, New York, NY, USA, 2020. Association for Computing Machinery.
- [44] Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Steven Wu. Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 2564-2572. PMLR, 10-15 Jul 2018.
- [45] Dongha Kim, Kunwoong Kim, Insung Kong, Ilsang Ohn, and Yongdai Kim. Learning fair representation with a parametric integral probability metric. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 11074-11101. PMLR, 17-23 Jul 2022.
- [46] Joon Sik Kim, Jiahao Chen, and Ameet Talwalkar. FACT: A diagnostic for group fairness trade-offs. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 5264-5274. PMLR, 13-18 Jul 2020.
- [47] Peizhao Li and Hongfu Liu. Achieving fairness at no utility cost via data reweighing with influence. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 12917-12930. PMLR, 17-23 Jul 2022.
- [48] Siqi Li, Qiming Wu, Xin Li, Di Miao, Chuan Hong, Wenjun Gu, Yuqing Shang, Yohei Okada, Michael Hao Chen, Mengying Yan, et al. Fairfml: Fair federated machine learning with a case study on reducing gender disparities in cardiac arrest outcome prediction. arXiv preprint arXiv:2410.17269 , 2024.
- [49] Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. Ditto: Fair and robust federated learning through personalization. In International conference on machine learning , pages 6357-6368. PMLR, 2021.
- [50] Tianlin Li, Qing Guo, Aishan Liu, Mengnan Du, Zhiming Li, and Yang Liu. FAIRER: Fairness as decision rationale alignment. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 19471-19489. PMLR, 23-29 Jul 2023.
- [51] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. arXiv preprint arXiv:1907.02189 , 2019.
- [52] Yuxi Liu, Guibo Luo, and Yuesheng Zhu. Fedfms: Exploring federated foundation models for medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention , pages 283-293. Springer, 2024.

- [53] Michael Lohaus, Michael Perrot, and Ulrike Von Luxburg. Too relaxed to be fair. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 6360-6369. PMLR, 13-18 Jul 2020.
- [54] Disha Makhija, Xing Han, Joydeep Ghosh, and Yejin Kim. Achieving fairness across local and global models in federated learning. arXiv preprint arXiv:2406.17102 , 2024.
- [55] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics , pages 1273-1282. PMLR, 2017.
- [56] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. ACM computing surveys (CSUR) , 54(6):1-35, 2021.
- [57] Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. Long-tail learning via logit adjustment. In International Conference on Learning Representations , 2021.
- [58] Mehryar Mohri, Gary Sivek, and Ananda Theertha Suresh. Agnostic federated learning. In International conference on machine learning , pages 4615-4625. PMLR, 2019.
- [59] Harikrishna Narasimhan and Aditya Krishna Menon. Training over-parameterized models with non-decomposable objectives. In Proceedings of the 35th International Conference on Neural Information Processing Systems , NIPS '21, Red Hook, NY, USA, 2021. Curran Associates Inc.
- [60] Harikrishna Narasimhan, Harish G. Ramaswamy, Shiv Kumar Tavker, Drona Khurana, Praneeth Netrapalli, and Shivani Agarwal. Consistent multiclass algorithms for complex metrics and constraints. Journal of Machine Learning Research , 25(367):1-81, 2024.
- [61] Lawrence Narici and Edward Beckenstein. Topological vector spaces . Chapman and Hall/CRC, 2010.
- [62] Dinh C Nguyen, Quoc-Viet Pham, Pubudu N Pathirana, Ming Ding, Aruna Seneviratne, Zihuai Lin, Octavia Dobre, and Won-Joo Hwang. Federated learning for smart healthcare: A survey. ACM Computing Surveys (Csur) , 55(3):1-37, 2022.
- [63] Sarthak Pati, Ujjwal Baid, Brandon Edwards, Micah Sheller, Shih-Han Wang, G Anthony Reina, Patrick Foley, Alexey Gruzdev, Deepthi Karkada, Christos Davatzikos, et al. Federated learning enables big data for rare cancer boundary detection. Nature communications , 13(1):7346, 2022.
- [64] Yangyang Qu, Michele Panariello, Massimiliano Todisco, and Nicholas Evans. Reference-free adversarial sex obfuscation in speech. arXiv preprint arXiv:2508.02295 , 2025.
- [65] Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletari, Holger R Roth, Shadi Albarqouni, Spyridon Bakas, Mathieu N Galtier, Bennett A Landman, Klaus Maier-Hein, et al. The future of digital health with federated learning. NPJ digital medicine , 3(1):1-7, 2020.
- [66] R. Tyrrell Rockafellar and Roger J.-B. Wets. Variational Analysis . Springer Verlag, Heidelberg, Berlin, New York, 2009.
- [67] Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh. Fairbatch: Batch selection for model fairness. In International Conference on Learning Representations , 2021.
- [68] Milad Sefidgaran, Romain Chor, Abdellatif Zaidi, and Yijun Wan. Lessons from generalization error analysis of federated learning: You may communicate less often! In Forty-first International Conference on Machine Learning , 2024.
- [69] Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms . Cambridge University Press, 2014.
- [70] Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and Trends® in Machine Learning , 4(2):107-194, 2012.

- [71] Jacob Steinhardt and Percy Liang. Adaptivity and optimism: An improved exponentiated gradient algorithm. In International conference on machine learning , pages 1593-1601. PMLR, 2014.
- [72] Qianyi Sun, Zheyong Qiu, Hong Ye, and Zhiyao Wan. Multinational corporation location plan under multiple factors. In Journal of Physics: Conference Series , volume 1168, page 032012. IOP Publishing, 2019.
- [73] Zehua Sun, Yonghui Xu, Yong Liu, Wei He, Lanju Kong, Fangzhao Wu, Yali Jiang, and Lizhen Cui. A survey on federated recommendation systems. IEEE Transactions on Neural Networks and Learning Systems , 36(1):6-20, 2024.
- [74] Davoud Ataee Tarzanagh, Mingchen Li, Christos Thrampoulidis, and Samet Oymak. FedNest: Federated bilevel, minimax, and compositional optimization. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 21146-21179. PMLR, 17-23 Jul 2022.
- [75] Ganghua Wang, Ali Payani, Myungjin Lee, and Ramana Kompella. Mitigating group bias in federated learning: Beyond local fairness. arXiv preprint arXiv:2305.09931 , 2023.
- [76] Xiaoyan Wang, Ran Li, Bowei Yan, and Oluwasanmi Koyejo. Consistent classification with generalized metrics, 2019.
- [77] Robert C. Williamson, Elodie Vernet, and Mark D. Reid. Composite multiclass losses. Journal of Machine Learning Research , 17(222):1-52, 2016.
- [78] Ruicheng Xian, Lang Yin, and Han Zhao. Fair and optimal classification via post-processing. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 37977-38012. PMLR, 23-29 Jul 2023.
- [79] Ruicheng Xian and Han Zhao. A unified post-processing framework for group fairness in classification, 2024.
- [80] Jie Xu, Benjamin S Glicksberg, Chang Su, Peter Walker, Jiang Bian, and Fei Wang. Federated learning for healthcare informatics. Journal of healthcare informatics research , 5:1-19, 2021.
- [81] Forest Yang, Mouhamadou Cisse, and Sanmi Koyejo. Fairness with overlapping groups; a probabilistic perspective. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 4067-4078. Curran Associates, Inc., 2020.
- [82] Liu Yang, Ben Tan, Vincent W Zheng, Kai Chen, and Qiang Yang. Federated recommendation systems. In Federated Learning: Privacy and Incentive , pages 225-239. Springer, 2020.
- [83] Wei Yao, Zhanke Zhou, Zhicong Li, Bo Han, and Yong Liu. Understanding fairness surrogate functions in algorithmic fairness. Transactions on Machine Learning Research , 2024.
- [84] Mehdi Yazdani-Jahromi, Ali Khodabandeh Yalabadi, AmirArsalan Rajabi, Aida Tayebi, Ivan Garibay, and Ozlem Garibay. Fair bilevel neural network (fairbinn): On balancing fairness and accuracy via stackelberg equilibrium. Advances in Neural Information Processing Systems , 37:105780-105818, 2024.
- [85] Rongguang Ye, Wei-Bin Kou, and Ming Tang. Praffl: A preference-aware scheme in fair federated learning. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 , KDD '25, page 1797-1808, New York, NY, USA, 2025. Association for Computing Machinery.
- [86] Zhenyu Yu and Chee Seng Chan. Yuan: Yielding unblemished aesthetics through a unified network for visual imperfections removal in generated images. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 9716-9724, 2025.

- [87] Zhenyu Yu, Mohd Yamani Idna Idris, and Pei Wang. Physics-constrained symbolic regression from imagery. In 2nd AI for Math Workshop@ ICML 2025 , 2025.
- [88] Zhenyu Yu, Mohd Yamani Idna Idris, Pei Wang, Yuelong Xia, and Yong Xiang. Forgetme: Benchmarking the selective forgetting capabilities of generative models. Engineering Applications of Artificial Intelligence , 161:112087, 2025.
- [89] Honglin Yuan, Manzil Zaheer, and Sashank Reddi. Federated composite optimization. In International Conference on Machine Learning , pages 12253-12266. PMLR, 2021.
- [90] Boya Zeng, Yida Yin, and Zhuang Liu. Understanding bias in large-scale visual datasets. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems , volume 37, pages 61839-61871. Curran Associates, Inc., 2024.
- [91] Xianli Zeng, Guang Cheng, and Edgar Dobriban. Bayes-optimal fair classification with linear disparity constraints via pre-, in-, and post-processing. arXiv preprint arXiv:2402.02817 , 2024.
- [92] Yi Zeng, Xuelin Yang, Li Chen, Cristian Ferrer, Ming Jin, Michael Jordan, and Ruoxi Jia. Fairness-aware meta-learning via nash bargaining. Advances in Neural Information Processing Systems , 37:83235-83267, 2024.
- [93] Yuchen Zeng, Hongxu Chen, and Kangwook Lee. Improving fairness via federated learning. arXiv preprint arXiv:2110.15545 , 2021.
- [94] J. Zhang, W. Zhang, C. Tan, X. Li, and Q. Sun. Yolo-ppa based efficient traffic sign detection for cruise control in autonomous driving. arXiv preprint arXiv:2409.03320 , 2024.
- [95] Li Zhang, Chaochao Chen, Zhongxuan Han, Qiyong Zhong, and Xiaolin Zheng. Logofair: Post-processing for local and global fairness in federated learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 22470-22478, 2025.
- [96] Yuanhan Zhang, ZhenFei Yin, Yidong Li, Guojun Yin, Junjie Yan, Jing Shao, and Ziwei Liu. Celeba-spoof: Large-scale face anti-spoofing dataset with rich annotations. In Computer VisionECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XII 16 , pages 70-85. Springer, 2020.
- [97] Zhaowei Zhu, Yuanshun Yao, Jiankai Sun, Hang Li, and Yang Liu. Weak proxies are sufficient and preferable for fairness with missing sensitive attributes. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 43258-43288. PMLR, 23-29 Jul 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In Section 1

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In Appendix E

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

Justification: The theorems and proofs presented in both the Section 4 and the Appendix B are comprehensive and complete.

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

Justification: Code is provided in the supplemental material.

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

## Answer: [Yes]

Justification: Code is provided in the supplemental material.

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

Justification: In Section 5 and Appendix C

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In Section 5 and Appendix C

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

Justification: In Appendix C.1

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: In Section E

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

Justification: This question is not applicable as the paper does not release any data or models with a high risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: In Section 5

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

Justification: Provided in the supplemental material.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This question is not applicable as the paper does not involve crowdsourcing experiments or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This question is not applicable as the paper does not release any data or models with a high risk of misuse.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.

## A Fairness Criteria in Centralized and Federated Learning Setting

In this section, we provide supplementary discussion of the fairness criteria and their corresponding confusion-matrix formulations under both centralized and federated learning settings. First, in addition to the demographic parity (DP) and equal opportunity (EOP) notions introduced above, we here present the definitions of equality of odds (EO) along with their confusion-matrix representations. Next, we clarify how these fairness notions are formalized within FL, specifying the distinct fairness metrics employed at both the global and the local levels. Note that this paper adopts a subgrouplike fairness metric [81, 14, 44] to reduce the number of constraints, while our confusion-matrix representation is also applicable to the group-wise definitions of these fairness metrics [78, 79].

## A.1 Group Fairness Criteria

Probabilistic notations. We elucidate some probability notations in the Preliminaries 3 and Table 1. Here, we use p δ to denote the probability of event δ occurring. For example, p a := P ( A = a ) , p y = P ( Y = y ) , p a,k := P ( A = a, K = k ) , p k | a := P ( K = k | A = a ) , p a | k := P ( A = a | K = k ) , p a,y := P ( A = a, Y = y ) , p y,k := P ( Y = y, K = k ) , and p a,y,k := P ( A = a, Y = y, K = k ) .

Confusion-matrix-based fairness notations. For random tuple ( X,Y,A ) , the prediction of the (attribute-aware) classifier is defined as ̂ Y = h ( X,A ) . One may simply choose ̂ Y = h ( X ) to consider the attribute-blind setting. To represent group fairness constraints, previous works [81, 60] introduce the group-specific confusion matrices C a , a ∈ A to characterize the fairness constraints, where C a i,j := P ( Y = i, ̂ Y = j | A = a ) .

Example 1. For DP criterion,

<!-- formula-not-decoded -->

where P ( Y = y | A = a ′ ) = ∑ i ∈ [ m ] P ( ̂ Y = y, Y = i | A = a ′ ) = ∑ i ∈ [ m ] C a ′ i,y and P ( ̂ Y = y ) = ∑ a ∈A P ( A = a ) ∑ i ∈ [ m ] P ( ̂ Y = y, Y = i | A = a ) = ∑ a ∈A ∑ i ∈ [ m ] P ( A = a ) C a i,y . Hence, we have

<!-- formula-not-decoded -->

where D a a ′ ,y ∈ R m × m , and the y -th column elements of D a a ′ ,y are I [ a = a ′ ] -P ( A = a ) with all other elements set to 0 .

Example 2. For EOP criterion,

<!-- formula-not-decoded -->

where P ( Y = y | A = a ′ , Y = y ) = p a ′ p a ′ ,y C a ′ y,y and P ( ̂ Y = y | Y = y ) = ∑ a ∈A p a p y C a y,y .

Hence, we have

<!-- formula-not-decoded -->

where D a a ′ ,y ∈ R m × m , and the entry in the y -th row and y -th column is p a ′ p a ′ ,y I [ a = a ′ ] -p a p y with all other elements set to 0 .

Example 3. For EOP criterion, we follow [3] to introduce the mean equalized odds (MEO) constraint, and consider its subgroup-like representation:

<!-- formula-not-decoded -->

̸

̸

where TPR y ( a ) = P ( ̂ Y = y | Y = y, A = a ) , TPR y = P ( ̂ Y = y | Y = y ) and FPR y ( a ) = P ( ̂ Y = y | Y = y, A = a ) , FPR y = P ( ̂ Y = y | Y = y ) .

It shows that

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

where the entry in the y -th row and y -th column is p a ′ p a ′ ,y I [ a = a ′ ] -p a p y with all other elements set to 0 for D a, 0 a ′ ,y ∈ R m × m , and the entry in the y -th column is p a ′ ∑ y j = y p a ′ ,y j I [ a = a ′ ] -p a ∑ y j = y p y j except for the y -th row with all other elements set to 0 for D a, 1 a ′ ,y ∈ R m × m .

## A.2 Group Fairness notations in FL

As noted in the main text, fairness at the level of each client's dataset ( local fairness ) differs from fairness across the aggregate dataset of all clients ( global fairness ). Local fairness is defined with respect to each client's individual data distribution P ( X,Y,A | K ) , whereas global fairness is defined over the overall (aggregate) distribution P ( X,Y,A ) . Motivated by approaches that employ group-specific confusion matrices for fairness [81, 60], we propose the decentralized group-specific confusion matrices C a,k , a ∈ A , k ∈ [ N ] to capture both global and local fairness across multiple data distributions within FL, with elements defined for i, j ∈ [ m ] as C a,k i,j ( h ) := P ( Y = i, ̂ Y = j | A = a, K = k ) .

Example 4. For DP criterion, the global DP fairness metric is defined as

<!-- formula-not-decoded -->

where P ( Y = y | A = a ′ ) = ∑ k ∈ [ N ] ∑ i ∈ [ m ] p k | a ′ C a ′ ,k i,y ( h k ) , and P ( ̂ Y = y ) = ∑ a ∈A ∑ k ∈ [ N ] ∑ i ∈ [ m ] p a,k C a,k i,y ( h k ) . Hence, we have

<!-- formula-not-decoded -->

where D a,k a ′ ,y ∈ R m × m , and the y -th column elements of D a,k a ′ ,y are P ( K = k | A = a ′ ) I [ a = a ′ ] -P ( A = a, K = k ) with all other elements set to 0 .

The local DP fairness metric for k -th client is defined as

<!-- formula-not-decoded -->

where P ( Y = y | A = a ′ , K = k ) = ∑ i ∈ [ m ] C a ′ ,k i,y , and P ( ̂ Y = y | K = k ) = ∑ a ∈A ∑ i ∈ [ m ] p a | k P ( ̂ Y = y, Y = i | A = a, K = k ) . Hence, we have

<!-- formula-not-decoded -->

where D a,k a ′ ,y ∈ R m × m , and the y -th column elements of D a,k a ′ ,y are I [ a = a ′ ] -P ( A = a | K = k ) with all other elements set to 0 .

Example 5. For EOP criterion, the global EOP fairness metric is defined as

<!-- formula-not-decoded -->

where P ( Y = y | Y = y, A = a ′ ) = ∑ k ∈ [ N ] p a ′ ,k p a ′ ,y C a ′ ,k i,y ( h k ) , and P ( ̂ Y = y | Y = y ) = ∑ a ∈A ∑ k ∈ [ N ] p a,k p y C a,k i,y ( h k ) . Hence, we have

<!-- formula-not-decoded -->

where D a,k a ′ ,y ∈ R m × m , and the entry in the y -th row and y -th column is p a ′ ,k p a ′ ,y I [ a = a ′ ] -p a,k p y with all other elements set to 0 .

The local EOP fairness metric for k -th client is defined as

<!-- formula-not-decoded -->

where P ( ̂ Y = y | A = a ′ , Y = y, K = k ) = p a ′ ,k p a ′ ,y,k C p a ′ ,k i,y , and P ( ̂ Y = y | Y = y, K = k ) = ∑ a ∈A p a,k p y,k P ( ̂ Y = y, Y = i | A = a, K = k ) . Hence, we have

<!-- formula-not-decoded -->

where D a,k a ′ ,y ∈ R m × m , and the entry in the y -th row and y -th column is p a ′ ,k p a ′ ,y,k I [ a = a ′ ] -p a,k p y,k with all other elements set to 0 .

Example 6. For EOP criterion, the global EOP fairness metric is defined as

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

where P ( Y = y | Y = y, A = a ′ ) = ∑ k ∈ [ N ] ∑ y i = y p a ′ ,k ∑ y j = y p a ′ ,y j C a ′ ,k y i ,y ( h k ) , and P ( ̂ Y = y | Y = y ) = ∑ a ∈A ∑ k ∈ [ N ] ∑ y i = y p a,k ∑ y j = y p y j C a,k y i ,y ( h k ) . Hence, we have

̸

̸

<!-- formula-not-decoded -->

̸

̸

where the entry in the y -th row and y -th column is a ′ ,k p a ′ ,y I [ a = a ′ ] -p a,k p y with all other elements set to 0 for D a,k, 0 a ′ ,y ∈ R m × m , and the entry in the y -th column is p a ′ ,k ∑ y j = y p a ′ ,y j I [ a = a ′ ] -p a,k ∑ y j = y p y j except for the y -th row with all other elements set to 0 for D a,k, 1 a ′ ,y ∈ R m × m .

The local EOP fairness metric for k -th client is defined as

̸

<!-- formula-not-decoded -->

̸

̸

̸

where P ( ̂ Y = y | A = a ′ , Y = y, K = k ) = ∑ y i = y p a ′ ,k ∑ y j = y p a ′ ,y j ,k C p a ′ ,k y i ,y , and P ( ̂ Y = y | Y = y, K = k ) = ∑ a ∈A p a,k ∑ y j = y p y j ,k P ( ̂ Y = y, Y = i | A = a, K = k ) . Hence, we have

<!-- formula-not-decoded -->

̸

̸

where the entry in the y -th row and y -th column is a ′ ,k p a ′ ,y,k I [ a = a ′ ] -p a,k p y,k with all other elements set to 0 for D a,k, 0 a ′ ,y ∈ R m × m , and the entry in the y -th column is p a ′ ,k ∑ y j = y p a ′ ,y j ,k I [ a = a ′ ] -p a,k ∑ y j = y p y j ,k except for the y -th row with all other elements set to 0 for D a,k, 1 a ′ ,y ∈ R m × m .

## A.3 Other Fairness Notations

Fairness notions formulated as ratios metrics can be converted into linear constraints under certain conditions, and our framework is well suited to enforce these fairness constraints in federated learning environments. Specifically, the ratios metric or constraints, which are formulated as ∣ ∣ ∣ ∑ a ⟨ D a , C a ( h ) ⟩ ∑ a ⟨ G a , C a ( h ) ⟩ ∣ ∣ ∣ ≤ ξ with constant metrix D a , G a and group-specific confusion matrix C a ( h ) depend on classifier h , can certainly be transformed into multiple linear constraints if the sign of ∑ a ⟨ G a , C a ( h ) ⟩ is unchanged for any h . When the denominator's sign is uncertain, the feasible domain of C a ( h ) is non-convex, precluding its expression via linear constraints. In fact, since each entry of C a ( h ) lies in [0,1], whenever the entries of G a are sign-consistent, the corresponding ratio constraint admits a linear-constraint representation. For example, the Calibration within Groups (CG) which was proposed in [5] and further explored in [46], is a fairness metric in binary classification and can be formulated as FN a FN a + TN a = v 0 ; TP a TP a + FP a = v 1 , where TP a , F P a , F N a , T N a are derived from binary group-specific confusion matrix C a and 0 ≤ v 0 &lt; v 1 ≤ 1 and have no implicit dependence on any entries of the fairness-confusion tensor. Because this fairness criterion appears as a ratio metric and every element of corresponding G a is non-negative, it admits a linear-constraint representation and can be realized in our proposed distributed framework. Moreover, the ratio metrics presented in [3] also can be formulated into multiple linear constraints based on the above analysis.

̸

̸

̸

̸

## B Proofs and Discussion in Section 4

## B.1 Proof of Proposition 1

This section provides the proof of Proposition 1. The proof is primarily inspired by the characterization of the Bayes-optimal fair classifier in the centralized fair machine learning literature (e.g. Theorem 3.1 of [81], Proposition 10 of [60]).

Proof. We begin by casting the primal problem (1) into an optimization problem defined on the Cartesian product of confusion matrices. Consider the the set of achievable confusion matrices:

<!-- formula-not-decoded -->

where C |A|× N be the product space of all confusion matrices C a,k corresponding to sensitive group a ∈ A and k ∈ [ N ] associated with a given instance h ∈ H N of the problem. It is clear that the performance metric R and fairness metrics D g , D k , k ∈ [ N ] are continuous and bounded to C |A|× N ( h ) := { C a,k ( h k ) } a ∈A ,k ∈ [ N ] .

Convexity of C |A|× N . Let ∀ C 1 , C 2 ∈ C |A|× N be realized by classifier tuples h 1 , h 2 . For any ω ∈ [0 , 1] , define the mixed classifier h ′ = ω h 1 + (1 -ω ) h 2 , . By linearity of performance and fairness metrics, its confusion matrix satisfies

<!-- formula-not-decoded -->

Thus every convex combination of C 1 and C 2 lies in C |A|× N , establishing convexity.

Deterministic classifiers. It can be seen that, for any linear objective ϕ L ( C |A|× N ( h )) = ∑ a ∈A ∑ k ∈ [ N ] ⟨ L a,k , C a,k ( h k ) ⟩ , there is a deterministic classifiers h ∗ = ( h ∗ 1 , · · · , h ∗ N ) that is optimal for ϕ L (see proof in B.2). By the supporting-hyperplane theorem [9] for compact convex sets, for each point C b = { C a,k b } a ∈A ,k ∈ [ N ] ∈ ∂ C |A|× N , there exists a nonzero collection of matrices L b = { L a,k b } a ∈A , k ∈ [ N ] constitutes a hyperplane, such that for every C = { C a,k } ∈ C |A|× N we have ∑ a ∈A ∑ N k =1 ⟨ L a,k b , C a,k b ⟩ ≤ ∑ a ∈A ∑ N k =1 ⟨ L a,k b , C a,k ⟩ which is precisely the desired supporting-hyperplane condition at C b . In other words, we arrive at the conclusion that each boundary point of C |A|× N can be achieved by deterministic classifiers h ′ = ( h ′ 1 , · · · , h ′ N ) .

Combination of deterministic classifiers. Since C |A|× N is compact and convex, we know that its extreme points fall in its boundary. By the Krein-Milman theorem [61], we have that C |A|× N is equal to the convex hull of its extreme points. We further have from Caratheodory's theorem [9] that any C ∈ C |A|× N can be expressed as a convex combination of d k = |A| Nm 2 points in the extreme point set, where each extreme point can be characterized by deterministic classifiers. Hence, we have proved that the optimal solution h can be represented by the convex combination of deterministic classifiers. □

Discussion on feasibility. The only condition for the above theorem to hold is that the feasible set is non-empty, which is clearly satisfied by the mentioned fairness constraints, DP, EOP, and EO. For these fairness criteria, the classifier that always predicts a single, fixed label y ′ trivially meets ξ g = 0 , ξ k = 0 , k ∈ [ N ] , and hence satisfies the fairness constraints.

The number of deterministic classifiers. As for the number of deterministic classifiers required, the parameter d k in the proof scales with the number of nonzero entries in the linear performance and fairness constraints [60]. Since each matrix D a,k in our fairness formulation is zero except for one column, we in fact need far fewer than |A| Nm 2 classifiers. Moreover, under the continuity assumption 1, this number can be reduced even further [81].

## B.2 Proof of Proposition 2

Proof. We denote p a := P ( A = a ) , p k := P ( K = k ) , p a,k := P ( A = a, K = k ) , and P X k := P ( X | K = k ) . Consider the form Lagrangian function of federated Bayes-optimal fair classification

<!-- formula-not-decoded -->

The inner problem of Lagrangian dual ask we to solve min h ∈H L ( h , λ, µ ) given element-wise non-negative dual parameter λ and µ , which can be formulated as

<!-- formula-not-decoded -->

The next step is to derive the optimal solution of max h ∈H N V ( h , λ, µ ) . For this purpose, we perform manipulations of H to reveal its clear relationship with the personalized classifier h = ( h 1 , . . . , h N ) . Denote the condition distribution of X given sensitive attribute A = a on client K = k as P X a,k , i.e., P X a,k := P ( X | A = a, K = k ) , we have

<!-- formula-not-decoded -->

To derive the optimal solution of the inner optimization problem, it suffices to perform a pointwise maximization of the above objective: for fixed x, k , the classifier h k ( x ) selects the label that maximizes the term inside the expectation, i.e.,

<!-- formula-not-decoded -->

Thus, we have finished the proof of Proposition 2.

□

## B.3 Proof of Proposition 3 and Further Exploration

In this section, we prove that the representation in (4) is calibrated for both unified and personalized inner optimization problem. We begin by presenting the following lemma.

Lemma B.1. For any categorical distribution characterized by p ∈ ∆ m , the minimizer of the expected risk

<!-- formula-not-decoded -->

over all q ∈ ∆ m is unique and achieved at p = q .

This lemma is commonly used in the design of multiclass loss functions [77, 57, 59].

## B.3.1 Proof of Proposition 3

Proof. We aim to prove that for any fixed x ∈ X , k ∈ [ N ] , the optimal personalized scoring function s ∗ k : X → R m that minimizes the expected loss ℓ k ( y, s ( x ) , a ) over the local data distribution P ( X,A,Y | K = k ) recovers the personalized federated Bayes-optimal classifier h ∗ k ( x ) in Proposition 2.

It is equivalent to show that, for any x :

<!-- formula-not-decoded -->

To this end, by leveraging the properties of conditional expectation, the cost-sensitive loss is reformulated as a function of the marginal distribution ( X,K ) :

<!-- formula-not-decoded -->

Denoting v i ( x, k ) := ∑ a ∈A P ( A = a | x, k ) ([ M µ,λ ( a, k ) ] ⊤ η ( x, a, k ) ) i ,we have

<!-- formula-not-decoded -->

where c x,k = ∑ j ∈ [ m ] v j ( x, k ) can be treated as a constant for fixed x, k . According to Lemma B.1, given fixed x, k , an optimal personalized classifier s ∗ k ( x ) minimizing the cost-sensitive loss point-wise satisfies

<!-- formula-not-decoded -->

It presents that, for all i ∈ [ m ] , since ∑ i ∈ [ m ] η i ( x, a, k ) = 1 ,

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

The personalized classifier h ∗ k ( x ) ∈ arg min y ∈ [ m ] [ s ∗ k ( x )] y recovers that in Proposition 2. We finish the proof. □

## B.3.2 Exploration of Calibrated Loss for Unified Bayes-Optimal Classifier

We start from the inner optimization objective V ( h , λ, µ ) ,

<!-- formula-not-decoded -->

To derive the optimal solution of the inner optimization problem, it suffices to perform a point-wise maximization of the above objective: for fixed x , the classifier h ( x ) selects the label that maximizes the term inside the expectation, i.e.,

<!-- formula-not-decoded -->

Consider the calibrated loss function in (4),

<!-- formula-not-decoded -->

By leveraging Lemma B.1, and employing an approach analogous to that used in the proof of Proposition 3, it is clear that we can obtain

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

The unified classifier h ∗ ( x ) ∈ arg min y ∈ [ m ] [ s ∗ ( x )] y recovers that in Proposition 2. We have shown that the loss ℓ k in (4) is also calibrated for the unified federated Bayes-optimal fair classifier. □

## B.4 The Complete Formulation of Theorem 4 with Its Proof.

In this subsection, we fully articulate Theorem 4 through Theorem 7 and Theorem 8, which together form an extended version of the result in Theorem 4. Before proceeding, we first clarify some notations and assumptions.

With a little abuse of notation, let f t k,ens ( x ) := f ( x ; ϕ t k ) + f ( x ; θ t ) , f ( x ; ϕ t k ) = softmax( s k ( x ; ϕ t k )) and f ( x ; θ t ) = softmax( s k ( x ; θ t )) . The local objective for

<!-- formula-not-decoded -->

which is similar to L t k ( f ( x ; ϕ t k )) and L t k ( f t k,ens ( x )) .

Assumption 2. The local loss function L t 1 , · · · , L t N are convex, β -smooth and bounded by B L to model parameters ϕ k and θ , t ∈ [ T ] .

Assumption 3. Let B t,r k be sampled from the k -th device's local data uniformly at random. The variance of stochastic gradients in each client is bounded:

<!-- formula-not-decoded -->

for k ∈ [ N ] , t ∈ [ T ] .

Assumption 2 and 3 are standard in the convergence analysis of federated model [51, 49, 74]. Now we present Theorem 7 and Theorem 8, which together constitute an extended form of Theorem 4.

Theorem 7. Under assumptions 2 and 3, for the ensemble personalized models, denoting θ ∗ := arg min θ ∑ T t =1 ∑ N k =1 p k L t k ( f ( x ; θ )) and B k = RβB L + 3 2 βB L + 3 4 σ 2 , the following cumulative global regret upper bound of all clients is guaranteed:

<!-- formula-not-decoded -->

while denoting ϕ ∗ k := arg min ϕ k ∑ T t =1 L t k ( f ( x ; ϕ k )) , the k -th client achieves the following personalized regret upper bound:

<!-- formula-not-decoded -->

Theorem 8. Suppose that personalized models achieve a ρ t -approximate optimal response at iteration t , namely ̂ L ( h t , λ t , µ t ) ≤ min h ̂ L ( h , λ t , µ t ) + ρ t , denoting ¯ ρ = ∑ T t =1 ρ t /T , then the sequences of model and bounded dual parameters comprise an approximate mixed Nash equilibrium:

<!-- formula-not-decoded -->

## B.4.1 Proof of Theorem 7

The proof of Theorem 7 comprises proofs of the global regret bound, and the local regret bound.

(1) Global regret upper bound . In Algorithm 1, the model parameter is updated for R iterations locally. Therefore, for any θ ∈ Θ ,

<!-- formula-not-decoded -->

Denoting g t,r k = ∇ L t k ( f ( x ; θ t,r k )) and G t,r k = ∇ L t k ( f ( x ; θ t,r k ); B t,r k ) , the local update can be written as

<!-- formula-not-decoded -->

Summarizing the inequality for r = 0 , · · · , R -1 , it shows that

<!-- formula-not-decoded -->

By convexity, we have

<!-- formula-not-decoded -->

By the β -smoothness, it indicates that ∥ g t,r k ∥ 2 ≤ 2 βB L , and then

<!-- formula-not-decoded -->

Summing up over r = 0 , · · · , r ′ , it presents that

<!-- formula-not-decoded -->

Hence, summing up over r ′ = 0 , · · · , R -1 again, we have

<!-- formula-not-decoded -->

Combining (11), (12) and (13), and let η ≤ 1 βR , we obtain

<!-- formula-not-decoded -->

From (10), we know that

<!-- formula-not-decoded -->

Summing over time and dividing both sides by 1 2 ηRT , we obtain

<!-- formula-not-decoded -->

Plugging in θ = θ ∗ and θ 0 = 0 and considering the fact that θ T +1 -θ ≥ 0 , the result turns to

<!-- formula-not-decoded -->

Consider the update rule of ensemble weight w t k in Algorithm 1,

<!-- formula-not-decoded -->

Here, the update can be viewed as exponentiated gradient descent on the normalized weight vector w t k = ( w t k, 1 , w t k, 2 ) ∈ ∆ 2 , and w t k,i ∝ exp( -η w z k t,i ) , i = 1 , 2 , where z k t, 1 = L t k ( f ( x ; θ t )) , z k t, 2 = L t k ( f ( x ; ϕ t k )) . A well-known regret bound in online learning [70, 71] shows that, for any u = ( u 1 , u 2 ) ∈ ∆ 2 ,

<!-- formula-not-decoded -->

By the convexity of L t k , we have ∑ T t =1 L t k ( f t k,ens ) ≤ ∑ T t =1 w t k L t k ( f ( x ; θ t ))+(1 -w t k ) L t k ( f ( x ; ϕ t k )) . Plugging in u 1 = 1 , u 2 = 0 , it presents that

<!-- formula-not-decoded -->

Weighted summing (17) over all clients and dividing both sides by T , we obtain

<!-- formula-not-decoded -->

Combining (18) and (15), and denoting B k = RβB L + 3 2 βB L + 3 4 σ 2 , we obtain

<!-- formula-not-decoded -->

Thus, we finish the proof of the global regret upper bound.

(2) Local regret upper bound. Plugging in u 1 = 0 , u 2 = 1 in (16), it presents that

<!-- formula-not-decoded -->

Following the proof technique of global regret upper bound, from (14), since ϕ t,R k = ϕ t +1 k , and making η ≤ 1 βR , we have for any ϕ k ,

<!-- formula-not-decoded -->

Combining (20) and (21), and plugging in ϕ k = ϕ ∗ k and ϕ 0 k = 0 denoting B k = RβB L + 3 2 βB L + 3 4 σ 2 , the result turns to

<!-- formula-not-decoded -->

Thus, we finish the proof of the local regret upper bound.

## B.4.2 Proof of Theorem 8

The proof of Theorem 8 relies on Lemma B.2.

Lemma B.2. [70] Let f 1 , f 2 , . . . : Λ → R be a sequence of convex functions that we wish to minimize on a compact convex set Λ . Define the bound of the convex set B d ≥ max λ ∈ Λ ∥ Λ ∥ 2 , and B G ≥ ∥∇ f t ( λ t ) ∥ 2 is a uniform upper bound on the norms of the subgradients. Suppose that we perform T iterations of the following update, starting from λ (1) = argmin λ ∈ Λ ∥ λ ∥ 1 :

<!-- formula-not-decoded -->

where ∇ f t ( λ t ) ∈ ∂f t ( λ t ) is a subgradient of f t at ( λ , and Π Λ projects its argument onto Λ w.r.t. the Euclidean norm. Then:

<!-- formula-not-decoded -->

where λ ∗ ∈ Λ is an arbitrary reference vector.

Proof of Theorem 8. Consider the empirical form of the Lagrangian function ̂ L ( h , λ, µ ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

□

where ̂ M λ,µ ( a, k ) := I -1 ˆ p a,k [ ∑ u g ∈U g ( λ (1) u g -λ (2) u g ) ̂ D a,k u g -∑ u k ∈U k ( µ (1) k,u k -µ (2) k,u k ) ̂ D a,k u k ] . It is clear that the inner problem is linear to classifiers in the empirical case.

From the definition in Section 4, we have ∥ λ ∥ 1 ≤ B d , ∥ µ ∥ 1 ≤ B d . Since the norm of fairness metrics is less than 2 , setting the step size η d = B d / √ T , by Lemma B.2,

<!-- formula-not-decoded -->

where λ ∗ , µ ∗ are the optimal dual parameters satisfying ∥ λ ∥ 1 ≤ B d , ∥ µ ∥ 1 ≤ B d .

On the other hand, according to the sub-optimal assumption on the classifier h , we have

<!-- formula-not-decoded -->

where ¯ ρ := ∑ T t =1 ρ t /T . Combining (23) and (24), the result shows that

<!-- formula-not-decoded -->

Let h := 1 T ∑ T t =1 h t with h k := 1 T ∑ T t =1 h t k , k ∈ [ N ] , and let ¯ λ := 1 T ∑ T t =1 λ t , ¯ µ := 1 T ∑ T t =1 µ t denote the point-wise average of dual parameters. Therefore, due to the linearity of the empirical Lagrange function to classifiers and dual parameters, (25) can be formulated as

<!-- formula-not-decoded -->

which presents the approximate mixed Nash equilibrium of the stochastic saddle-point problem. □

## B.5 Generalization Error For In-processing Algorithm

We begin by introducing some notations and simplifications, which are commonly employed in generalization analyses of FL [39, 68]. Without loss of generalization, let n = n 1 = · · · = n k present the sample number in local datasets. For any class H = { h : X → [ m ] } , denote H y = { I { h ( x ) = y } : h ∈ H} and the maximal Vapnik-Chervonenkis dimension [69], V C ( H ) := max y ∈ [ m ] V C ( H y ) .

Theorem 9. If classifiers h = ( ¯ h 1 , . . . , ¯ h k ) with dual parameters ( ¯ λ, ¯ µ ) form a ϵ -saddle point of empirical Lagrangian ̂ L ( h , λ, µ ) , and an optimal solution h ∗ ∈ H satisfies both global and local fairness constraints, denoting ν ( n, H , δ ) = 2 √ 2 V C ( H ) log( n +1) n + √ 2 log( m 2 N/δ )) n , B g = max a ∈A ,k ∈ [ N ] ∥ D a,k u g ∥ 1 , Ω g n = max a ∈A ,k ∈ [ N ] ∥ D a,k u g -̂ D a,k u g ∥ ∞ , Ω p n := ∑ N k =1 | p k -ˆ p k | , and B k = max a ∈A ∥ D a,k u k ∥ 1 , Ω k n = max a ∈A ∥ D a,k u k -̂ D a,k u k ∥ ∞ , k ∈ [ N ] , then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Theorem 9 relies on the following lemma.

Lemma B.3. Let H : X → [ m ] , D a distribution over X× ∆ m ,of which { x i , y i } n i =1 are i.i.d samples. Denoting H y = { I { h ( x ) = y } : h ∈ H} and V C ( H ) = max y ∈ [ m ] V C ( H y ) , then with probability at least 1 -δ , for ∀ i, j ∈ [ m ] ,

<!-- formula-not-decoded -->

where F H := { f ( x ) = ∑ N j =1 α j h j ( x ) : α ∈ ∆ N , h j ∈ H , j ∈ [ m ] } .

Proof of Lemma B.3. Let ℓ i,j ( x, y ; h ) = I ( y = i ∧ h ( x ) = j ) . Then we have C i,j ( h ) = E [ ℓ i,j ( x, y ; h )] and ̂ C i,j ( h ) = 1 n ∑ n i =1 ℓ i,j ( x i , y i ; h ) . Hence, according to the classical result with respect to cost sensitive binary classification [8], with probability at least 1 -δ ,

<!-- formula-not-decoded -->

By the definition of V C ( H ) , it achieves the generalization bound.

## B.5.1 Proof of Theorem 9

Let the optimal solution h ∗ minimize the risk R ( h ) subjected to global and local fairness constraints | D g ( h ∗ ) | ≤ ξ g , | D k,l ( h ∗ ) | ≤ ξ k,l . With the properties of the saddle point, it is clear that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Considering the global fairness constraints, we first explore its concentration, for any h ∈ H ,

<!-- formula-not-decoded -->

The last inequality is by the Holder's inequality. Let ℓ a ′ ,k i,j ( x, y, a ; h ) = I ( y = i ∧ h ( x ) = j ∧ a = a ′ ) . Then we have C a ′ ,k i,j ( h ) = E [ ℓ a ′ ,k i,j ( x, y, a ; h )] and ̂ C a ′ i,j ( h ) = 1 n ∑ n i =1 ℓ i,j ( x i , y i ; h ) . By taking a union bound in Lemma B.3, we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Since ∥ ̂ C a,k ( h k ) ∥ 1 = 1 , taking the union bound again, denoting ν ( n, H , δ ) = 2 √ 2 V C ( H ) log( n +1) n + √ 2 log( m 2 N/δ )) n , it turns out that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we consider the optimality. Denoting u ∗ g := arg max u g ∈U g | ̂ D g u g ( h ) | , then we have

<!-- formula-not-decoded -->

where e 1 u ∗ g defines as the basis vector with 1 at the position of λ 1 u ∗ g . Let h satisfy the fairness constraints. With (27), we obtain

<!-- formula-not-decoded -->

Combining (31) and (32), it shows that

<!-- formula-not-decoded -->

Therefore, the result shows that max u g ∈U g | ̂ D g u ∗ g ( h ) | -ξ g ≤ 1+2 ϵ B d .

Now we consider the generalization error for the empirically optimal classifier h , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking the union bound over u g ∈ U g , we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

For local fairness constraints, | D k | ≤ ξ k , following the similar proof procedures as local fairness constraints, we have that

<!-- formula-not-decoded -->

For risk metric R ( h ) , it presents that

<!-- formula-not-decoded -->

By (27) and (28),

<!-- formula-not-decoded -->

Since we have ̂ R ( h ) = 1 -∑ N k =1 ˆ p k ⟨ I , C k ( h k ) ⟩ , it presents that

<!-- formula-not-decoded -->

By taking a union bound in Lemma B.3, we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Hence, denoting Ω p n := ∑ N k =1 | p k -ˆ p k | , we arrive that, for any h ∈ F H ,

<!-- formula-not-decoded -->

Therefore, combining (38), (39) and (40), we obtain

<!-- formula-not-decoded -->

This completes the proof.

## B.6 Proof of Theorem 5

We begin by introducing some definitions and lemmas, which are useful in the proof of Theorem 5. Definition 3. Let V be a real vector space and let A,B ⊆ V . The sum of A and B is defined by

<!-- formula-not-decoded -->

□

Lemma B.4. [6] The subdifferential of the function F ( x ) = E { f ( x, ω ) } at a point x is given by ∂F ( x ) = E { ∂f ( x, ω ) }

where f ( · , ω ) is a real-value convex function and the set E { ∂f ( x, ω ) } is defined as

<!-- formula-not-decoded -->

Lemma B.5. [66] Let f 1 , . . . , f m : R n → ( -∞ , + ∞ ] be convex functions. Define f ( x ) = max { f 1 ( x ) , . . . , f m ( x ) } , ∀ x ∈ R n . For x 0 ∈ ⋂ m i =1 dom f i , define I ( x 0 ) = { i | f i ( x 0 ) = f ( x 0 ) } . Then ∂f ( x 0 ) = conv ⋃ i ∈ I ( x 0 ) ∂f i ( x 0 ) .

Lemma B.6. [66] Let f : R d → R be a convex continuous function. We consider the minimizer x ∗ of the function f over the set B . Then for x ∗ to be locally optimal it is necessary that

<!-- formula-not-decoded -->

where N B denotes the normal cone of set B . If B = R d + , let K := { k ∈ [ d ] , x ∗ k = 0 } . Then there exists a subgradient ξ ∈ ∂f ( x ∗ ) , such that for all k ∈ [ d ] we have ξ k ≥ 0 and ∀ k ∈ K , ξ k = 0 .

̸

Proof of Theorem 5. From the above analysis, it follows that the Lagrange function can be written as

<!-- formula-not-decoded -->

We first consider the inner optimization problem min h ∈H N L ( h , λ, µ ) , which is equivalent to optimize

<!-- formula-not-decoded -->

where M λ,µ ( a, k ) := I -1 p a,k [ ∑ u g ∈U g ( λ (1) u g -λ (2) u g ) D a,k u g -∑ u k ∈U k ( µ (1) k,u k -µ (2) k,u k ) D a,k u k ] . Considering the personalized attribute-aware classifier h k ( x, a ) , k ∈ [ N ] in post-processing, the inner function turns to

<!-- formula-not-decoded -->

An explicit optimal solution of personalized classifier is that

<!-- formula-not-decoded -->

If the maximum entry of the output vector occurs at multiple indices, one of them is randomly selected as the predicted class. Thus, the dual problem can be formulated as

<!-- formula-not-decoded -->

Before exploring the optimal solution of outer optimization, we first prove that the optimal dual parameter λ ∗ ∈ R 2 |U g | ≥ 0 , µ ∗ ∈ R 2 ∑ N k =1 |U k | ≥ 0 is bounded. Define the Hilbert space on F := { f : X → R m } with inner product ⟨ f, g ⟩ = ∫ X f ⊤ gd P ( x ) . Then the classifier space H : X → ∆ m is a convex subset of F . Therefore, we can also consider the topology structure on H or H |A| . Since we assume that ∀ ξ g , ξ k &gt; 0 , the feasible set of the primal problem is non-empty, it indicates that the feasible set of the primal problem has non-empty interior for any positive ξ g , ξ k . It is clear that for ∀ ξ g , ξ k &gt; 0 , the dual problem

<!-- formula-not-decoded -->

where h fair denotes a classifier that satisfies fairness constraints for given ξ g , ξ k &gt; 0 . Hence, we arrive at

<!-- formula-not-decoded -->

holds for all λ, µ ≥ 0 . Notice that given λ, µ ≥ 0 , this inequality holds for any ξ g , ξ k &gt; 0 . Let ξ g → 0 , ξ k → 0 , combining (42) and (43) gives that

<!-- formula-not-decoded -->

Therefore, the dual problem has a lower bound

<!-- formula-not-decoded -->

It presents that, as ∥ λ ∥ 1 → ∞ or ∥ µ ∥ 1 → ∞ , there must be H ( λ, µ ) → ∞ , which conflicts with the dual problem min λ,µ H ( λ, µ ) . Hence, the optimal λ ∗ , µ ∗ of dual problem min λ,µ H ( λ, µ ) must have bounded norms, denoting as ∥ λ ∥ 1 ≤ B d , ∥ µ ∥ 1 ≤ B d .

Now we consider the differential of H ( λ, µ ) . It is clear that { S y = { x ∈ X : h k ( x, a ) = y } , y ∈ [ m ] } constructs a partition of the feature space X . Hence, for dual parameter λ (1) u g , since the outer objective H is convex to λ and µ , by the additivity subgradients and Lemma B.4, the differential ∂ ∂λ (1) ug H ( λ, µ ) can be formulated as

<!-- formula-not-decoded -->

With a slight abuse of notation, let score function f ( x, a, k ) = [ M λ,µ ( a, k ) ] ⊤ η ( x, a, k ) , by Lemma B.5, we have

̸

<!-- formula-not-decoded -->

̸

where B t y := {∃ t = y, f t ( x, a, k ) ≥ f i ( x, a, k ) , ∀ i ∈ [ m ]; f t ( x, a, k ) = f y ( x, a, k ) } with b t ∈ [0 , 1] . Since the convex hull is a interval here, by Caratheodory's theorem, it can be characterized by

two point here (the initial point e y and another point e t in the convex hull). Without loss of generality, we assume the existence of one e t such that f t ( x, a, k ) = f y ( x, a, k ) here. We know that f t ( x, a, k ) -f y ( x, a, k ) = [ η ( x, a, k )] ⊤ M λ,µ ( a, k )( e t -e y ) . With Asumption 1, we obtain that the measure of B t y is 0 , unless the t -th and y -th column of M λ,µ ( a, k ) are equal. An effective simplification is to exclude all λ ′ , µ ′ that cause M λ ′ ,µ ′ ( a, k )( e t -e y ) = 0 . Since we suppose that the non-zero columns of each D a,k u are distinct, the dual parameter λ ′ , µ ′ ∈ S t,y , such that M λ ′ ,µ ′ ( a, k )( e t -e y ) = 0 , constructs the empty relative interior in the dual parameter space. By the convexity of the objective function, we have inf λ,µ/ ∈ S t,y H ( λ, µ ) = min λ,µ H ( λ, µ ) , due to the density of ( λ, µ ) / ∈ S t,y .

Overall, under the assumptions of the theorem, we have that B t y has a measure of zero. It follows that

<!-- formula-not-decoded -->

In a similar manner, we can derive

<!-- formula-not-decoded -->

Considering paired optimal dual parameter λ ( i ) ∗ u g , i = 1 , 2 , by Lemma B.6, if λ (1) ∗ u g , λ (2) ∗ u g &gt; 0 , we have

<!-- formula-not-decoded -->

which leads to a contradiction. If λ (1) ∗ u g = 0 , λ (2) ∗ u g = 0 , we have

<!-- formula-not-decoded -->

If λ (1) ∗ u g = 0 , λ (2) ∗ u g &gt; 0 , we have

<!-- formula-not-decoded -->

If λ (1) ∗ u g &gt; 0 , λ (2) ∗ u g = 0 , we have

<!-- formula-not-decoded -->

Overall, we have shown that for all u g ∈ U g , | D g u g ( h λ ∗ ,µ ) | ≤ ξ g .

The local fairness guarantee also can be derived from the optimality of µ ∗ . The proof techniques are extremely similar to our proof with respect to λ ∗ . Hence, we omit the proof of the local fairness guarantee here. The result turns out that | D k u k ( h λ,µ ∗ ) | ≤ ξ k , k ∈ [ N ] .

The next step is to prove that the classifier h λ ∗ ,µ ∗ is the optimal solution of the primal problem (1). From the proof above, we can obtain that, for ∀ u g ∈ U g ,

<!-- formula-not-decoded -->

which satisfies the optimality conditions for the dual solution of the constrained optimization problem. The same holds for the local fairness constraints D k ( h λ ∗ ,µ ∗ ) . Consequently, the Lagrangian function equals to risk function when plugging in optimal classifier, L ( h λ ∗ ,µ ∗ , λ ∗ , µ ∗ ) = R ( h λ ∗ ,µ ∗ ) . For any other classifiers h ′ that satisfies the global and local fairness constraints, denoting its corresponding dual parameter to maximize the outer problem as λ ′ , µ ′ , it can be deduced that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we arrive at

This completes the proof.

## B.7 Proof of Prosition 6

Note that λ ∈ R 2 |U g | ≥ 0 and µ k ∈ R 2 |U k | ≥ 0 , the operator ∥ · ∥ 1 is linear in dual parameters' domain. We can just write

<!-- formula-not-decoded -->

where ̂ M λ,µ ( a, k ) := I -1 ˆ p a,k [ ∑ u g ∈U g ( λ (1) u g -λ (2) u g ) ̂ D a,k u g -∑ u k ∈U k ( µ (1) k,u k -µ (2) k,u k ) ̂ D a,k u k ] and σ β ( x ) = ∑ m i =1 exp( x i /β ) ∑ j m =1 exp( x j /β ) x i .

Convexity. The ̂ M λ,µ ( a, k ) is linear to λ and µ k , and the soft-max operator is convex. Since the composition of an affine mapping and a convex function preserves convexity, ̂ H ′ k ( λ, µ k ) is convex to λ and µ k .

Smoothness. Consider the soft-max weighted sum σ β ( x ) := ∑ m j =1 exp( x j /β ) ∑ ℓ m =1 exp( x ℓ /β ) x j , and its Hessian matrix is given by H σ ( x ) := ∇ 2 σ β ( x ) , [ H σ ( x )] i,j = p i β [( 2+ x i -¯ x β ) I ( i = j ) -p j ( 2+ x i + x j -¯ x β )] . For ∀ i, j ∈ m , if ∥ x ∥ 1 ≤ R ,

<!-- formula-not-decoded -->

Hence, its spectral norm is bounded,

<!-- formula-not-decoded -->

Then, there exists a finite constant L σ := m 4 β +6 R β 2 , such that ∥∇ 2 σ β ( x ) ∥ 2 ≤ L σ .

For each sample i = 1 , . . . , n k , define the affine map

<!-- formula-not-decoded -->

Set f i ( λ ) := σ β ( z i ( λ ) ) . and let f i ( λ ) = σ β ( z i ( λ ) ) , σ β ( x ) = ∑ m j =1 e x j /β ∑ ℓ m =1 e x ℓ /β x j , By the chain rule and second-order derivatives, ∇ λ f i ( λ ) = A ⊤ i ∇ x σ β ( z i ( λ ) ) , ∇ 2 λ f i ( λ ) = A ⊤ i [ ∇ 2 σ β ( z i ( λ ) )] A i . Hence, due to the boundedness of ∥ λ ∥ 1 , the inside z i ( λ ) is bounded, setting the upper bound as R for simplification here, ∥ ∥ ∇ 2 λ f i ( λ ) ∥ ∥ 2 ≤ ∥ A i ∥ 2 2 sup x ∥ ∥ ∇ 2 σ β ( x ) ∥ ∥ 2 = ∥ A i ∥ 2 2 L σ , showing f i is ∥ A i ∥ 2 2 L σ -smooth. The linear term in λ has zero Hessian. Therefore, since the average of smooth functions is smooth with averaged constants, the function ̂ H ′ k ( λ, µ k ) is L -smooth in λ with L = 1 n k ∑ n k i =1 ∥ A i ∥ 2 2 L σ . Following the similar proof procedure, we can obtain the smoothness of ̂ H ′ k ( λ, µ k ) to µ k . □

## B.8 Generalization Error For Post-Processing Algorithm

We begin by introducing some notations and simplifications, same as the proof of Theorem 9. Without loss of generalization, let n = n 1 = · · · = n k present the sample number in local datasets. Denote p a | k := P ( A = a | K = k ) , p min := min a ∈A ,k ∈ [ N ] p a | k . Assume n min ≥ 1 denotes the sample size of the sensitive group with the fewest observations across all clients.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(1) Let 0 &lt; δ &lt; 1 , suppose that n &gt; 2 |A| N ̂ B k p min ξ g + 1 2 p 2 min log 1 δ , then with probability at least 1 -2 |A| δ ,

<!-- formula-not-decoded -->

(2) Let 0 &lt; δ &lt; 1 , suppose that n &gt; 2 |A| ̂ B g p min ξ k + 1 2 p 2 min log 1 δ , then with probability at least 1 -2 |A| δ ,

<!-- formula-not-decoded -->

The proof of Theorem 10 needs the following lemma.

Lemma B.7. Let X 1 , . . . , X n be independent Bernoulli( p ) random variables and define S n = ∑ n i =1 X i . Fix any M ∈ (0 , np ) and confidence level δ ∈ (0 , 1) . If the sample size satisfies n ≥ 2 M p + 1 2 p 2 log 1 δ , then we have P ( S n &gt; M ) ≥ 1 -δ.

Proof of Lemma B.7. By Hoeffding's inequality, for any t &gt; 0 , P ( S n -E [ S n ] ≤ -t ) ≤ exp ( -2 t 2 n ) . Since E [ S n ] = np , set t = np -M. Then

<!-- formula-not-decoded -->

To guarantee P ( S n ≤ M ) ≤ δ , it suffices that 2 ( np -M ) 2 n ≥ log 1 δ . Substitute n = 2 M p + 1 2 p 2 log 1 δ .

Then np -M = ( 2 M p + 1 2 p 2 log 1 δ ) p -M = M + 1 2 p log 1 δ , and one can check

<!-- formula-not-decoded -->

Hence P ( S n ≤ M ) ≤ δ , i.e. P ( S n &gt; M ) ≥ 1 -δ .

## B.8.1 Proof of Theorem 10

We first consider the generalization error of the fairness constraints. Without loss of generalization, here we only prove the generalization error for global fairness constraints and corresponding parameter λ . The proof technique for local fairness constraints and corresponding parameter µ is extremely similar to that for global fairness constraints.

We know that the personalized attribute-aware empirical classifier can be written as

<!-- formula-not-decoded -->

As h depends on the Bayes score function η , we can consider the input as ( η k,i := η ( x k,i , a k,i , k ) , a k,i , y k,i ) . Let ℓ a ′ ,k i,j ( η, a, y ; h ) = I ( y = i ∧ h ( η ) = j ∧ a = a ′ ) . Then we have C a ′ ,k i,j ( h ) = E [ ℓ a ′ ,k i,j ( η, y, a ; h )] and ̂ C a ′ i,j ( h ) = 1 n ∑ n z =1 ℓ a ′ .k i,j ( η k,z , a k,z , y k,z ) . Then we turn to consider the VC dimension of the function class H i,j,a ′ := { h : ( x, a, y ) → I ( y = i ∧ h ( η ) = j ∧ a = a ′ ) } . Thanks to the classifier's specific structural form (53), we can directly state an explicit upper bound on its VC dimension: for given class j ,

̸

<!-- formula-not-decoded -->

which can be regarded as the intersection of m -1 half-spaces given η k,i , a k,i . A single halfspace function class can be viewed as the class of linear classifiers, possessing a VC dimension of m . By the additive property of VC dimension, for function classes {G i } i m =1 , V C ( ∧ m i =1 G i ) ≤ ∑ m i =1 V C ( G i ) ,

□

the function class H i,j,a ′ has VC dimension at most O ( |A| m 2 ) . By taking a union bound in the Lemma B.3, we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Hence, for the global fairness constraints D g u g with the empirical optimal solution ̂ h ∗ , by the generalization bound in (29), we have that,

<!-- formula-not-decoded -->

Now we consider the bound on empirical ̂ D g u g ( ̂ h ∗ ) . The empirical optimal dual parameter ̂ λ ∗ and ̂ µ ∗ are obtained by the empirical dual function:

<!-- formula-not-decoded -->

This representation is fully consistent with that given in (8) restricting to groupa observations within the k -th client's data, where n a,k denotes the sample number of group a in client k . Considering the subgradient of the empirical dual function w.r.t. λ (1) u g , by the additivity of subgradient,

<!-- formula-not-decoded -->

Denoting empirical score function ̂ f ( x, a, k ) = [ ̂ M λ,µ ( a, k ) ] ⊤ ̂ η ( x, a, k ) , by Lemma B.5, we have

̸

<!-- formula-not-decoded -->

̸

where B t y := { x : ∃ t = y, ̂ f t ( x, a, k ) ≥ ̂ f i ( x, a, k ) , ∀ i ∈ [ m ]; ̂ f t ( x, a, k ) = ̂ f y ( x, a, k ) } and b t ∈ [0 , 1] . According to Carathéodory's theorem, the subgradient interval can still be represented by two points. According to our assumption, the plug-in estimator ̂ η still meet the continuity assumption and we exclude singular λ ′ , µ ′ . Therefore, we know that

̸

<!-- formula-not-decoded -->

Hence, the subgradient falls into an interval. Since [ ̂ η ( x i , a, k )] ⊤ ̂ D a,k u g e t ≤ ∥ [ ̂ η ( x i , a, k )] ⊤ ̂ D a,k u g ∥ 1 ≤ B g , denoting n min := min a ∈A ,k ∈ [ N ] n a,k and ̂ B g , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, we obtain that

<!-- formula-not-decoded -->

In a similar manner, we can derive the range of subgradient for λ (2) u g ,

<!-- formula-not-decoded -->

Since we assume that n &gt; 2 |A| N ̂ B k p min ξ g + 1 2 p 2 min log 1 δ , by Lemma B.7, we have that with probability at last 1 -|A| δ ,

<!-- formula-not-decoded -->

Consider the optimality of ̂ λ (1) u g , ̂ λ (2) u g , by Lemma B.6, if ̂ λ (1) u g &gt; 0 , ̂ λ (2) u g &gt; 0 , we have 0 ∈ ∂ ∂λ (1) ug ̂ H ( ̂ λ ∗ , µ ) , 0 ∈ ∂ ∂λ (2) ug ̂ H ( ̂ λ ∗ , µ ) . Thus,

<!-- formula-not-decoded -->

which leads to a contradiction. For other cases, such as ̂ λ (1) u g = ̂ λ (2) u g = 0; ̂ λ (1) u g &gt; 0 , ̂ λ (2) u g = 0 , and ̂ λ (1) u g = 0 , ̂ λ (2) u g &gt; 0 , as discussed in the proof of Theorem 5, it turns out that

<!-- formula-not-decoded -->

By taking a union bound, we obtain that with probability at least 1 -2 |A| δ ,

<!-- formula-not-decoded -->

For local fairness constraints D k ( ̂ h ∗ ) , following the same proof procedures, we arrive at that with probability at least 1 -2 |A| δ ,

<!-- formula-not-decoded -->

## C Additional Datasets and Experimental Setting

## C.1 Datasets and Experimental Details

## C.1.1 Datasets

- The Compas dataset [22] comprises 6,172 criminal defendants from Broward County, Florida, between 2013 and 2014, with the task of predicting whether a defendant will recidivate within two years of their initial risk assessment. We consider the race of each individual as the sensitive attribute and train a logistic classifier as our prediction model.
- The Adult dataset [4] comprises more than 45000 samples based on 1994 U.S. census data, where the task is to predict whether the annual income of an individual is above $50,000. We consider the gender of each individual as the sensitive attribute and train the logistic regression as the classification model.
- The ENEM dataset [40] contains about 1.4 million samples from Brazilian college entrance exam scores along with student demographic information. We follow [3] to quantized the exam score into 2 or 5 classes as label, and consider race as sensitive attribute. As [3] used a random subset of 50K samples, we instead sample 100K data points to construct our federated dataset. We train multilayer perceptron (MLP) as the classification model.
- The CelebA dataset [96] is a facial image dataset consists of about 200k instances with 40 binary attribute annotations. We identify the binary feature smile as target attributes which aims to predict whether the individuals in the images exhibit a smiling expression. The race of individuals is chosen as sensitive attribute. We train Resnet18 [38] on CelebA as the classification model.

The determination of sensitive attributes and labels on three datasets has been verified significant in previous research [3, 34].

## C.1.2 Baselines

We compare the performance of FedFACT with traditional FedAvg [55] and five SOTA methods tailored for calibrating global and local fairness in FL, namely FairFed [31], FedFB [93], FCFL [18], praFFL [85], and the method in [25], denoted as Cost in our experiments.

- FedAvg serves as a core Federated Learning model and provides the baseline for our experiments. It works by computing updates on each client's local dataset and subsequently aggregating these updates on a central server via averaging.
- FairFed introduces an approach to adaptively adjust the aggregation weights of different clients based on their local fairness metric to train federated model with global fairness guarantee.
- FedFB presents a FairBatch-based approach [67] to compute the coefficients of FairBatch parameters on the server. This method integrates global reweighting for each client into the FedAvg framework to fulfill fairness objectives.
- FCFL proposed a two-stage optimization to solve a multi-objective optimization with fairness constraints. The prediction loss at each local client is treated as an objective, and FCFL maximize the worst-performing client while considering fairness constraints by optimizing a surrogate maximum function involving all objectives.
- praFFL proposed a preference-aware federated learning scheme that integrates client-specific preference vectors into both the shared and personalized model components via a hypernetwork. It is theoretically proven to linearly converge to Pareto-optimal personalized models for each client's preference.
- [25] proposed a convex-programming-based post-processing framework that characterizes and enforces the minimum accuracy loss required to satisfy specified levels of both local and global fairness constraints in multi-class federated learning by approximating the region under the ROC hypersurface with a simplex and solving a linear program, denoted as Cost in our experiments.

Meanwhile, we adapt FedFACT to focus solely on global or local fairness in FL, denoted as FedFACT g and FedFACT l . FedFACT g &amp; l indicates the algorithm simultaneously achieving global and local fairness. The FedFACT (In) presents the in-processing method and FedFACT (Post) presents the post-processing method.

## C.1.3 Parameter Settings

We provide hyperparameter selection ranges for each model in Table 4. For all other hyperparameters, we follow the codes provided by authors and retain their default parameter settings.

Table 4: Hyperparameter Selection Ranges

| Model                                                              | Hyperparameter                                                                                                             | Ranges                                |
|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| Global round Local round Local batch size Hidden                   | {0.0001, 0.001, 0.003, 0.005, 0.01, 0.03 ,0.05} {20, 30, 50, 80} {10, 20, 30, 50} {128, 256, 512} {16, 32, 64} {Adam, SGD} | General Learning rate layer Optimizer |
| Step size ( α )                                                    | {0.005, 0.01, 0.05, 0.3}                                                                                                   | FedFB                                 |
| Fairness budget ( β Local debiasing ( α                            | ) {0.01, 0.05, 0.5, 1} ) {0.005, 0.01, 0.05}                                                                               | FairFed                               |
| Fairness constraint (                                              | ϵ ) {0.01, 0.03, 0.05, 0.07}                                                                                               | FCFL                                  |
| Diversity ( τ p )                                                  | {10, 15, 20}                                                                                                               | praFFL                                |
| Classifier number w t k learning rate ( η w ) Dual parameter bound | 1 {0.03, 0.3} 5                                                                                                            | FedFACT (In)                          |
| FedFACT (Post) Dual                                                | 0.1 bound 5                                                                                                                | Temperature β parameter               |

For the fairness-control parameters, e.g., the parameter λ in praFFL [85] and the global and local fairness constraints in Cost [25], we impose stringent fairness requirements on the model in our overall comparative experiments, and we adjust the parameters governing the fairness metrics in the Pareto-curve experiments.

## C.1.4 Experiments Compute Resources

We conducted our experiments on a GPU server equipped with 8 CPUs and two NVIDIA RTX 4090s (24G).

## C.2 Discussion about FedFACT and LoGoFair [95]

LogoFair [95] is designed for binary-classification in federated learning under both global and local fairness constraints, seeking the Bayes-optimal classifier. By deriving a closed-form solution for the fair Bayes classifier, LogoFair reformulates the post-processing fairness adjustment as a bilevel optimization problem jointly solved by the server and clients, which is an approach conceptually analogous to our post-processing framework. In binary classification, FedFact and LogoFair both target Bayes-optimal classifiers under constraints disparity metrics expressed in linear form. Theoretically, for an identical fairness metric, our Bayes-optimal fair classifier characterization covers that of LogoFair. Consequently, we refrain from performing a comparative evaluation of the two approaches.

Our method differs by defining the loss at the client level, thereby achieving lower estimation error than the local group-specific objective in [95]. Crucially, by formulating the post-processing model over the probabilistic simplex instead of restricting outputs to the unit interval [0 , 1] in the binary case, our framework achieves enhanced scalability and naturally adaptable to multi-group, multiclass settings.

Note that, whether for binary or multiclass settings, our implementation of FedFACT is based on calibrating confusion matrices over the multi-dimensional probabilistic simplex.

## C.3 Heterogeneous Split of Client Distribution

We propose a partitioning method that introduces heterogeneous correlations between the sensitive attribute A and label Y , thereby further elucidating the trade-off between global fairness and local fairness [33].

Heterogeneous Split. We assume a dataset D of n samples, each with a binary attribute A and a binary label Y . We denote by n ij = |{ x ℓ , a ℓ , y ℓ : ( a ℓ = i, y ℓ = j ) }| the number of samples in joint class ( i, j ) for i, j ∈ { 0 , 1 } . Our goal is to partition D into N disjoint subsets (one per client) such that in client k ∈ [ N ] , the correlation between A and Y is controlled by a target parameter γ k ∈ [ a, b ] ⊆ [0 , 1] . To achieve this, we first assign each client k a weight γ k

<!-- formula-not-decoded -->

Then for each joint class ( i, j ) we compute the total weight W ( i,j ) = ∑ n k =1 w ( i,j ) k and assign to client k a preliminary count c ( i,j ) k = ⌊ ( w ( i,j ) k /W ( i,j ) ) n ij ⌋ . Any remaining samples are distributed one by one to the clients with the largest fractional remainders, so that ∑ N k =1 c ( i,j ) k = n ij . Finally, for each class ( i, j ) we shuffle its n ij sample indices and slice them into blocks of size c ( i,j ) k . Client k then collects all its four blocks across ( i, j ) , yielding a partition that in expectation realizes the desired within-client correlation γ k between A and Y .

This approach can be regarded as a generalization of the synergy-level-based heterogeneous split in [33] to the multi-client setting, where the A -Y correlation for each client is governed by a parameter randomly drawn from [ a, b ] ⊆ [0 , 1] , thereby yielding a more pronounced balance between global fairness and local fairness. Throughout the experimental evaluation, we set γ k ∈ [0 . 2 , 0 . 8] to guarantee that every client has a sufficient number of sensitive group samples to assess local fairness.

## D Detailed Experiments Results

## D.1 Comparison Result and binary EOP criterion

Parato Curves of DP. We have already presented the numerical comparison between our proposed method and the baselines in the main text; here, we report the Pareto curves illustrating the trade-off between global fairness and accuracy. More precisely, we compare the trade-off between accuracy and the global fairness measure, as well as the trade-off between accuracy and the local fairness measure, as a function of the fairness constraint.

The Pareto curve for the global DP criterion is shown in Figure 2 where the horizontal axis denotes accuracy and the vertical axis represents the fairness metric. Consequently, models located closer to the upper-right corner exhibit superior accuracy-fairness trade-offs. As illustrated in Figure 2, our method outperforms all existing state-of-the-art approaches when comparing accuracy against either global fairness in isolation.

Figure 2: The Pareto frontier on Compas, Adult, CelebA and ENEM datasets. The curve closer to the upper right corner indicates a better trade-off between accuracy and fairness.

<!-- image -->

This result not only demonstrates that our model achieves a more favorable accuracy-fairness balance but also highlights its controllability: by tuning the fairness constraints, one can satisfy diverse fairness requirements.

Parato Curves of EOP. In Figure 3, we illustrate the Pareto curve for the Equalized Odds (EO) criterion-accuracy. Because EOP enforces tighter constraints than DP, precise adherence in a federated context requires large per-group sample counts at each client. Hence, we also compare the global EOP here. Our framework still exceeds all state-of-the-art baselines in trading off accuracy against fairness.

Figure 3: The Pareto frontier on Compas, Adult, CelebA and ENEM datasets. The curve closer to the upper right corner indicates a better trade-off between accuracy and fairness.

<!-- image -->

## D.2 Details for Multi-Class Classification

Multi-Class fair datasets. We illustrate how FedFACT performs on multi-class prediction using CelebA and ENEM. For CelebA, with 'Gender' still serving as the sensitive attribute, We employ the binary attributes 'Smile' and 'Big\_Nose' to construct a multiclass task by mapping their joint values { 0 , 1 } × { 0 , 1 } onto a four-class label set { 0 , 1 , 2 , 3 } , thereby formulating a multiclass classification problem on the CelebA dataset. These attributes are commonly used in centralized machine learning literature [13, 97] to construct fairness-aware classification tasks. For ENEM, we follow [3] to quantize the Humanities exam score to 5 classes. In order to guarantee adequate per-group sample sizes at each client in heterogeneous settings for fairness evaluation (or some clients only hold less than 10 samples for specific group under heterogeneous partitioning), we adopt the four race labels 'Branca,' 'Preta,' 'Parda,' and 'Amarela' from the Race attribute as the sensitive groups. These datasets are partitioned into five clients under a heterogeneous split with γ = 1 .

Evaluation. In terms of baselines, only the Cost [25] algorithm is theoretically applicable to fairness optimization in multiclass federated learning scenarios. However, their experiments and code are limited to binary classification, and have already been used as binary baselines for comparison with our method. Consequently, we focus exclusively on reporting FedFACT's performance along with FedAvg in multiclass fairness, establishing it as a pioneering approach in this setting.

## D.3 Additional Experiments for Adjusting Accuracy-Fairness Trade-Off

In Table 5, we present additional experiments on the Compas and Adult datasets under the heterogeneous split to illustrate the adjustment of the accuracy-fairness trade-off. Compared to the results in the main text, this partitioning yields a more pronounced trade-off between global and local fairness.

Table 5: Additional Accuracy-Fairness Balance.

| Dataset       | Compas (In-)   | Compas (In-)   | Compas (In-)   | Adult (In-)   | Adult (In-)   | Adult (In-)   | Compas (Post-)   | Compas (Post-)   | Compas (Post-)   | Adult (Post-)   | Adult (Post-)   | Adult (Post-)   |
|---------------|----------------|----------------|----------------|---------------|---------------|---------------|------------------|------------------|------------------|-----------------|-----------------|-----------------|
| ( ξ g , ξ l ) | Acc            | D global       | D local        | Acc           | D global      | D local       | Acc              | D global         | D local          | Acc             | D global        | D local         |
| (0.00,0.00)   | 60.22          | 0.0404         | 0.0745         | 80.99         | 0.0021        | 0.0407        | 64.56            | 0.0083           | 0.0075           | 81.25           | 0.0139          | 0.0275          |
| (0.02,0.00)   | 60.61          | 0.0436         | 0.0734         | 81.04         | 0.0021        | 0.0423        | 64.78            | 0.0091           | 0.0099           | 81.56           | 0.0146          | 0.0285          |
| (0.04,0.00)   | 60.90          | 0.0490         | 0.0737         | 81.09         | 0.0039        | 0.0446        | 65.04            | 0.0123           | 0.0099           | 81.62           | 0.0147          | 0.0285          |
| (0.00,0.02)   | 60.80          | 0.0499         | 0.0744         | 81.18         | 0.0046        | 0.0411        | 64.94            | 0.0146           | 0.0214           | 81.82           | 0.0238          | 0.0381          |
| (0.02,0.02)   | 61.03          | 0.0503         | 0.0726         | 81.64         | 0.0315        | 0.0463        | 65.12            | 0.0311           | 0.0306           | 82.04           | 0.0240          | 0.0381          |
| (0.04,0.02)   | 61.32          | 0.0555         | 0.0774         | 81.65         | 0.0318        | 0.0467        | 65.57            | 0.0378           | 0.0371           | 82.16           | 0.0257          | 0.0397          |
| (0.00,0.04)   | 61.18          | 0.0581         | 0.0804         | 81.31         | 0.0053        | 0.0444        | 65.16            | 0.0294           | 0.0517           | 82.46           | 0.0350          | 0.0492          |
| (0.02,0.04)   | 61.39          | 0.0644         | 0.0753         | 81.67         | 0.0177        | 0.0452        | 65.16            | 0.0412           | 0.0419           | 82.49           | 0.0346          | 0.0497          |
| (0.04,0.04)   | 62.39          | 0.0878         | 0.0966         | 82.14         | 0.0486        | 0.0497        | 65.82            | 0.0507           | 0.0574           | 82.63           | 0.0350          | 0.0518          |

Note that the gap between the imposed constraints and the observed fairness metrics stems from the inevitable generalization error incurred with finite local samples. Consequently, global fairness exhibits greater controllability than local fairness. In practice, FedFACT remains capable of

tuning the accuracy-fairness balance according to the specified fairness constraints, highlighting the controllability inherent in our approach.

## D.4 Hyper-Parameter Experiments

In this subsection, we examine the impact of the number of classifiers in the in-processing method. Specifically, we incrementally increase the size of the weighted ensemble-from using only the most recently trained classifier up to including the ten preceding classifiers. Let N h represent the number of classifiers comprising the weighted ensemble. As reported in Table 6, we observe that augmenting the ensemble with multiple classifiers yields negligible improvements and can even degrade performance when earlier classifiers have not been fully trained. Consequently, in light of these empirical findings, all in-processing experiments in this work utilize only the single most recently obtained classifier.

Table 6: Hyper-Parameter Experimental Results.

|     | Compas   | Compas   | Compas   | Adult   | Adult    | Adult   | CelebA   | CelebA   | CelebA   | ENEM   | ENEM     | ENEM    |
|-----|----------|----------|----------|---------|----------|---------|----------|----------|----------|--------|----------|---------|
| N h | Acc      | D global | D local  | Acc     | D global | D local | Acc      | D global | D local  | Acc    | D global | D local |
| 1   | 61.17    | 0.0407   | 0.0732   | 82.04   | 0.0014   | 0.0401  | 86.15    | 0.0382   | 0.0473   | 65.33  | 0.0293   | 0.0387  |
| 2   | 61.29    | 0.0408   | 0.0731   | 81.24   | 0.0015   | 0.0416  | 85.54    | 0.0382   | 0.0482   | 65.54  | 0.0285   | 0.0392  |
| 5   | 61.18    | 0.0410   | 0.0723   | 81.63   | 0.0032   | 0.0397  | 85.91    | 0.0377   | 0.0472   | 65.41  | 0.0307   | 0.0390  |
| 10  | 61.14    | 0.0404   | 0.0736   | 81.91   | 0.0048   | 0.0399  | 86.59    | 0.0384   | 0.0471   | 65.11  | 0.0398   | 0.0383  |

## D.5 Efficiency and Scalability Study

In this section, we conduct out experiments with DP criterion to examine the communication cost and scalability of FedFACT.

Efficiency. We evaluate the communication efficiency of FedFACT by monitoring its performance across varying numbers of communication rounds T . As illustrated in Figure 4, the post-processing method, built upon a fully trained pre-trained model, consistently achieves convergence in fewer than 10 communication rounds, underscoring its high efficiency. The in-processing method likewise converges in under 40 iterations; given that it requires training the federated model from scratch, this performance is comparable to the convergence speed of FedAvg, making it highly effective compared to existing federated learning algorithms.

/u1D49F

/u1D49F

Figure 4: Communication Effectiveness Analysis. The convergence rates of both the in-processing (top row) and post-processing (bottom row) methods with respect to communication rounds on Compas, Adult, CelebA, and ENEM datasets.

<!-- image -->

Overall, whether employing the in-processing or post-processing method, all three performance metrics rapidly converge to stable values across each of the four datasets, empirically confirming both the communication efficiency and the overall effectiveness of FedFACT.

Scalability. We evaluate FedFACT's performance as the number of clients varies from 2 to 50 on all four datasets, with heterogeneity parameter γ = 5 to ensure that each local client has adequate samples for assessing local fairness. The results, shown in Figure 5, indicate that on each dataset, there is an upward shift in the metric as the client count increases. Enforcing fairness constraints, especially via the in-processing method, sometimes necessitates a modest loss in accuracy, and the post-processing approach on the Compas dataset exhibits pronounced fairness fluctuations due to substantial generalization error when sample sizes are small. Aside from this, our method reliably bounds the model's fairness, underscoring its robustness to variations in client population.

## E Broader Impacts and Limitations

Broader Impacts. This paper addresses critical fairness issues in FL. By embedding fairness constraints at both the global and client levels, our framework delivers models that distribute accuracy more equitably, bolstering user confidence and mitigating bias amplification. The contributions of this research enhance user satisfaction and promote social equity. This fairness-aware approach extends readily to high-stakes classification tasks beyond FL: for instance, clinical decision support in hospital networks, vision-based detection systems, and financial fraud alerts. Integrating fairness into decentralized model training promotes privacy-preserving, equitable AI, helps satisfy emerging regulatory requirements, and encourages broader adoption of responsible machine learning across diverse application domains.

Limitations. The primary limitation of FedFACT is the fairness representation, which contains the linear disparities such as communly used DP, EOP and EOP criteria, but it excludes some nonlinear formulations of fairness, e.g. Predictive Parity [22] and individual fairness [29]. Moreover, based on our generalization-error analysis, although the proposed method enables a controllable accuracyfairness trade-off for a given fairness metric, it still requires a sufficiently large local sample size to accurately estimate local fairness (whereas global fairness demands only an adequate overall sample size). While our empirical results compare favorably against existing approaches, exploiting dataset characteristics to optimize fairness may reduce the sample complexity needed for local fairness optimization. Addressing these limitations remains an important avenue for future work.

Figure 5: Scalability Analysis. The behavior of both the in-processing (top row) and post-processing (bottom row) methods as the number of clients increases from 2 to 50 across Compas, Adult, CelebA, and ENEM datasets.

<!-- image -->
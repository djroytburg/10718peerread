## Error Feedback under ( L 0 , L 1 ) -Smoothness: Normalization and Momentum

## Sarit Khirirat

KAUST ∗

sarit.khirirat@kaust.edu.sa

## Artem Riabinin

KAUST

## Abdurakhmon Sadiev

KAUST

abdurakhmon.sadiev@kaust.edu.sa

## Eduard Gorbunov

MBZUAI †

artem.riabinin@kaust.edu.sa

eduard.gorbunov@mbzuai.ac.ae

## Peter Richtárik

KAUST

peter.richtarik@kaust.edu.sa

## Abstract

We provide the first proof of convergence for normalized error feedback algorithms across a wide range of machine learning problems. Despite their popularity and efficiency in training deep neural networks, traditional analyses of error feedback algorithms rely on the smoothness assumption that does not capture the properties of objective functions in these problems. Rather, these problems have recently been shown to satisfy generalized smoothness assumptions, and the theoretical understanding of error feedback algorithms under these assumptions remains largely unexplored. Moreover, to the best of our knowledge, all existing analyses under generalized smoothness either i) focus on single-node settings or ii) make unrealistically strong assumptions for distributed settings, such as requiring data heterogeneity, and almost surely bounded stochastic gradient noise variance. In this paper, we propose distributed error feedback algorithms that utilize normalization to achieve the O (1 / √ K ) convergence rate for nonconvex problems under generalized smoothness. Our analyses apply for distributed settings without data heterogeneity conditions, and enable stepsize tuning that is independent of problem parameters. Additionally, we provide strong convergence guarantees of normalized error feedback algorithms for stochastic settings. Finally, we show that due to their larger allowable stepsizes, our new normalized error feedback algorithms outperform their non-normalized counterparts on various tasks, including the minimization of polynomial functions, logistic regression, and ResNet-20 training.

## 1 Introduction

Machine learning models achieve impressive prediction and classification power by employing sophisticated architectures, comprising vast numbers of model parameters, and requiring training on massive datasets. Distributed training has emerged as an important approach, where multiple

∗ Sarit Khirirat, Abdurakhmon Sadiev, Artem Riabinin, and Peter Richtárik are with the Center of Excellence for Generative AI, King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia.

† Eduard Gorbunov is with the Department of Statistics and Data Science, Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Abu Dhabi, United Arab Emirates.

machines with their own local training data collaborate to train a model efficiently within a reasonable time. Many optimization algorithms can be easily adapted for distributed training frameworks. For example, stochastic gradient descent ( SGD ) can be modified into distributed stochastic gradient descent within a data parallelism framework, and into federated averaging algorithms [1] in a federated learning framework. However, the communication overhead of running these distributed algorithms poses a significant barrier to scaling up to large models. For example, training the VGG-16 model [2] using distributed stochastic gradient descent involves communicating 138 . 34 million parameters, thus consuming over 500 MBof storage and posing an unmanageable burden on the communication network between machines.

One approach to mitigate the communication burden is to apply compression. In this approach, the information, such as gradients or model parameters, is compressed using sparsifiers or quantizers to be transmitted with much lower communicated bits between machines. However, while this reduces communication overhead, too coarse compression often brings substantial challenges in maintaining high training performance due to information loss, and in extreme cases, it may potentially lead to divergence. Therefore, error feedback mechanisms have been developed to improve the convergence performance of compression algorithms, while ensuring high communication efficiency. Examples of error feedback mechanisms include EF14 [3, 4, 5, 6, 7], EF21 [8, 9], EF21-SGDM [10], EF21-P [11], and EControl [12]. Several studies developing error feedback algorithms often assume the smoothness of an objective function, i.e., its gradient is Lipschitz continuous.

However, many modern learning problems, such as distributionally robust optimization [13] and deep neural network training, are often non-smooth. For instance, the gradient of the loss computed for deep neural networks, such as LSTM [14], ResNet20 [14], and transformer models [15], is not Lipschitz continuous. These empirical findings highlight the need for a new smoothness assumption. One such assumption is ( L 0 , L 1 ) -smoothness, originally introduced by Zhang et al. [14], for twice differentiable functions, and later extended to differentiable functions by Chen et al. [16].

To solve generalized smooth problems, clipping and normalization have been widely utilized in first-order algorithms. Gradient descent with gradient clipping was initially shown by Zhang et al. [14] to achieve lower iteration complexity, i.e., fewer iterations needed to attain a target solution accuracy, than classical gradient descent. Subsequent works have further refined the convergence theory of clipped gradient descent [17], and improved its convergence performance by employing momentum updates [18], variance reduction techniques [19], and adaptive step sizes [20, 21, 22]. Similar convergence results have been obtained for gradient descent using normalization [23], and its momentum variants [24], including generalized SignSGD [15]. However, these first-order algorithms have mostly been explored in training on a single machine. To the best of our knowledge, distributed algorithms under generalized smoothness have been investigated in only a few works, e.g., by Crawshaw et al. [25], Liu et al. [26]. Nonetheless, these works rely on assumptions limiting families of optimization problems, including data heterogeneity, almost sure variance bounds, and symmetric noise distributions around the mean assumptions. Furthermore, these first-order algorithms under generalized smoothness do not incorporate compression techniques to improve communication efficiency. These aspects motivate us to develop distributed communication-efficient algorithms for solving nonconvex generalized smooth problems .

## 1.1 Contributions

In this paper, we develop distributed error feedback algorithms for communication-efficient optimization under nonconvex, generalized smooth regimes. Our contributions are summarized below.

- Importance of normalization. Just as gradient clipping is crucial for gradient descent, we empirically demonstrate that normalization stabilizes the convergence of error feedback algorithms for minimizing nonconvex generalized smooth functions. In this paper, we introduce a variant of EF21 , a widely used error feedback algorithm by Richtárik et al. [8], which incorporates normalization to guarantee convergence for nonconvex, generalized smooth problems. In a single-node setting, this new method, which we call ||EF21-GD|| , or more compactly as ||EF21|| , provides larger stepsize, and faster convergence rate than its non-normalized counterpart EF21 for minimizing simple nonconvex polynomial functions that satisfy generalized smoothness, as shown by Figure 1.

√

- Convergence of normalized error feedback algorithms. We establish an O (1 / K ) convergence rate in the gradient norm for ||EF21|| on nonconvex generalized smooth problems. ||EF21|| achieves

Figure 1: The minimization of polynomial functions using EF21 with γ = 1 L + L √ β θ , and ||EF21||

<!-- image -->

with γ = ˆ γ √ K +1 , ˆ γ = 1 (blue line) and γ = 1 2 c 1 (green line). Here, we ran both algorithms for (1) L 0 = 4 , L 1 = 1 , and K = 2 , 000 (left), (2) L 0 = 4 , L 1 = 4 , and K = 5 , 000 (middle), and (3) L 0 = 4 , L 1 = 8 , and K = 16 , 000 (right).

the same rate as EF21 under L -smoothness by [8]. Our results are derived under standard assumptions, i.e., generalized smoothness and the existence of lower bounds on the objective function, and are applicable in distributed settings regardless of any data heterogeneity degree, unlike the results by Crawshaw et al. [25], Liu et al. [26]. Additionally, our stepsize rules for ||EF21|| ensure convergence without requiring knowledge of the generalized smoothness constants L 0 or L 1 , in contrast to Richtárik et al. [8], where the stepsize depends on the smoothness constant L (which is often inaccessible).

- Extension to stochastic settings. Furthermore, we propose a variant of EF21-SGDM , an error feedback algorithm with momentum updates by Fatkhullin et al. [10], that employs normalization for solving nonconvex, stochastic optimization under generalized smoothness. Specifically, we prove that ||EF21-SGDM|| with suitable stepsize choices attains the same O (1 /K 1 / 4 ) convergence rate in the gradient norm as EF21-SGDM .
- Numerical evaluation. We implemented ||EF21|| using the stepsize rules derived from our theory, and compared its performance against EF21 . Both algorithms were evaluated on three learning tasks: minimizing nonconvex polynomial functions, solving logistic regression with a nonconvex regularizer, and training ResNet-20 on the CIFAR-10 dataset. Thanks to its larger stepsizes, ||EF21|| outperforms EF21 , in terms of both convergence speed and solution accuracy across these tasks.

Table 1: Comparisons of complexities and assumptions between known and our results for EF21 variants. The complexity is defined by the iteration count K required by the algorithms to attain min k =0 , 1 ,...,K E [∥ ∥ ∇ f ( x k ) ∥ ∥ ] ≤ ϵ . ( L 0 , L 1 ) -smoothness refers to generalized smoothness in Assumption 3. The variance bound in expectation is defined in Assumption 5.

| Methods                                        | Complexity   | Smoothness   | Variance bound   | Normalization   |
|------------------------------------------------|--------------|--------------|------------------|-----------------|
| EF21 Richtárik et al. [8]                      | O (1 /ϵ 2 )  | L            | No               | No              |
| EF21-SGDM Fatkhullin et al. [10]               | O (1 /ϵ 4 )  | L            | expectation      | No              |
| &#124;&#124;EF21&#124;&#124; NEW (Alg. 1)      | O (1 /ϵ 2 )  | ( L 0 ,L 1 ) | No               | Yes             |
| &#124;&#124;EF21-SGDM&#124;&#124; NEW (Alg. 2) | O (1 /ϵ 4 )  | ( L 0 ,L 1 ) | Expectation      | Yes             |

## 2 Related Works

Error feedback. Error feedback mechanisms have been utilized in various algorithms with communication compression, leading to significant improvements in solution accuracy, while reducing communication. As the first version of these mechanisms, EF14 was introduced by Seide et al. [3], and later analyzed for first-order algorithms in both single-node [4, 27] and distributed settings [5, 6, 28, 29, 7, 30, 31, 32]. Next, EF21 is another error feedback variant proposed by Richtárik et al. [8], which offers strong convergence guarantees for distributed gradient algorithms with any

contractive compressors, without requiring bounded gradient norm or bounded data heterogeneity assumptions. EF21 can also be adapted for stochastic optimization through sufficiently large minibatches [9] or momentum updates [10]. More recently, EControl was developed by Gao et al. [12] to guarantee provably superior complexity results for distributed stochastic optimization compared to prior error feedback mechanisms. To the best of our knowledge, these existing works on error feedback have focused solely on optimization under traditional L -smoothness. In this paper, we introduce a normalized variant of the EF21 methods [8] for solving nonconvex generalized smooth problems. In particular, we prove that ||EF21|| under generalized smoothness achieves the same O (1 / √ K ) rate as EF21 under traditional smoothness, and demonstrate in experiments that ||EF21|| permits larger step sizes, and thus attains faster convergence than EF21 .

Non-smoothness assumptions. Empirical findings suggest that the traditional smoothness used for analyzing optimization algorithms does not capture the properties of objective functions in many machine learning problems, especially deep neural network training problems. This motivates researchers to consider different assumptions to replace this traditional smoothness condition. First introduced by Zhang et al. [14], the ( L 0 , L 1 ) -smoothness condition on a twice differentiable function f ( x ) is defined by ∥ ∥ ∇ 2 f ( x ) ∥ ∥ ≤ L 0 + L 1 ∥∇ f ( x ) ∥ for x ∈ R d . This ( L 0 , L 1 ) -smoothness has been extended to differentiable functions without assuming the existence of the Hessian. For instance, the smoothness with a differentiable function ℓ ( x ) [33], and symmetric generalized smoothness [16] cover the ( L 0 , L 1 ) -smoothness when the Hessian exists, and includes many important machine learning problems, such as phase retrieval problems [16], and distributionally robust optimization [34]. Other classes of non-smoothness assumptions, which are not related to the generalized smoothness but capture other optimization problems, include Hölder's continuity of the gradient [35], the relative smoothness [36], and the polynomial growth of the gradient norm [37]. In this paper, we impose the generalized smoothness condition to establish the convergence of ||EF21|| for solving deterministic and stochastic optimization.

Gradient clipping and normalization. Clipping and normalization are commonly employed in gradient-based methods for solving generalized smooth problems. Clipped (stochastic) gradient descent has been studied for both nonconvex and convex problems under ( L 0 , L 1 ) -smoothness conditions by Zhang et al. [14], Koloskova et al. [17]. Extensions to clipped gradient algorithms have been proposed, including momentum updates [18], variance reduction methods [19], and adaptive step sizes [20, 21, 22, 38]. Comparable complexities have been achieved for normalized gradient descent [23], and its momentum-based variants [24], including SignSGD [15] and its variance-reduction variants [39]. Convergence properties of gradient-based algorithms have also been explored under more generalized forms of non-uniform smoothness, extending beyond the ( L 0 , L 1 ) -smoothness by Zhang et al. [14] to cover a wider range of optimization problems. For example, variants of (stochastic) gradient descent have been analyzed under α -symmetric generalized smoothness by Chen et al. [16], and under ℓ -smoothness involving certain differentiable functions ℓ ( · ) by Li et al. [33, 21]. However, the majority of these analyses focus on the single-node setting. To the best of our knowledge, only a limited number of works, such as those by Crawshaw et al. [25], Liu et al. [26], have examined federated averaging algorithms for nonconvex problems under generalized smoothness. These works, however, often rely on restrictive assumptions, including data heterogeneity, almost sure variance bounds, and symmetric noise distributions centered around their means. In this paper, we develop distributed error feedback algorithms, which eliminate the need for the restrictive assumptions mentioned above, and rely on standard assumptions on objective functions and compressors.

## 3 Preliminaries

Notations. We use [ n ] to denote the set { 1 , 2 , . . . , n } , and E[ u ] to represent the expectation of a random variable u . Additionally, ∥·∥ indicates the Euclidean norm for vectors or the spectral norm for matrices, and ∥·∥ 1 is the ℓ 1 -norm for vectors, while ⟨ x, y ⟩ denotes the inner product between x and y in R d . Lastly, for a square matrix A ∈ R d × d , λ min ( A ) refers to its minimum eigenvalue, and I ∈ R d × d is the identity matrix.

Problem Formulation. We focus on the following distributed optimization problem:

<!-- formula-not-decoded -->

where n refers to the number of clients, and f i ( x ) is the loss of a model parameterized by vector x ∈ R d over its local data D i owned by client i ∈ [ n ] .

Assumptions. To facilitate our convergence analysis, we make standard assumptions on objective functions and compression operators.

Assumption 1 (Lower Boundedness of f ) . The function f is bounded from below, i.e.,

<!-- formula-not-decoded -->

Assumption 2 (Lower Boundedness of f i ) . For each i ∈ [ n ] , the function f i is bounded from below, i.e.,

<!-- formula-not-decoded -->

Assumptions 1 and 2 are standard for analyzing optimization algorithms for unconstrained problems.

Assumption 3 (Generalized Smoothness of f i ) . A function f i ( x ) is symmetrically generalized smooth if there exists L 0 , L 1 &gt; 0 such that for u θ = θx +(1 -θ ) y , and for all x, y ∈ R d ,

<!-- formula-not-decoded -->

Assumption 3 refers to symmetric generalized smoothness by Chen et al. [16], which covers asymmetric generalized smoothness [17, 16], and the original ( L 0 , L 1 ) -smoothness by [14]. Moreover, Assumption 3 covers the functions with unbounded classical smoothness constant, e.g., exponential function. Additionally, Assumption 3 with L 1 = 0 reduces to the traditional L 0 -smoothness [40, 41], under which the convergence of optimization algorithms has been extensively studied.

Assumption 4 (Contractive Compressor) . An operator C k : R d → R d is an α -contractive compressor if there exists α ∈ (0 , 1] such that for k ≥ 0 and v ∈ R d ,

<!-- formula-not-decoded -->

Furthermore, compressors defined by Assumption 4 cover topk sparsifiers [5, 4], low-rank approximation [42, 43], and various other compressors described by Safaryan et al. [44], Beznosikov et al. [45], Demidovich et al. [46].

Assumption 5 (Bounded Variance) . A stochastic gradient ∇ f i ( x ; ξ i ) with its sample ξ i ∼ D i is an unbiased estimator of ∇ f i ( x ) with bounded variance, i.e., for all x ∈ R d ,

<!-- formula-not-decoded -->

Assumption 5 is standard for stochastic optimization [47, 48, 49] that is only imposed on each local stochastic gradient, and it does not imply data heterogeneity, i.e., the bounded difference between each component function f i ( x ) and the global function f ( x ) .

## 4 Normalized Error Feedback ( ||EF21|| )

For nonconvex deterministic optimization under generalized smoothness, we develop a distributed error feedback algorithm. One challenge is that the generalized smoothness parameter scales with the gradient norm ∥ ∥ ∇ f ( x k ) ∥ ∥ . To resolve this issue, we apply gradient normalization to the algorithms. In particular, we consider ||EF21|| , the normalized version of EF21 [8] that updates the next iterates x k +1 using the ||EF21|| update. The full description of ||EF21|| can be found in Algorithm 1.

Our new method ||EF21|| , just like EF21 [8] under traditional smoothness, enjoys the O (1 / √ K ) convergence in the gradient norm under generalized smoothness, as shown below.

## Algorithm 1 Normalized Error Feedback ( ||EF21|| )

- 1: Input: Stepsize γ k &gt; 0 for k = 0 , 1 , . . . ; starting points x 0 , g -1 i ∈ R d for i ∈ { 1 , 2 , . . . , n } ; and α -contractive compressors C k : R d → R d for k = 0 , 1 , . . . .
- 2: for each iteration k = 0 , 1 , . . . , K do
- 3: for each client i = 1 , 2 , . . . , n in parallel do
- 4: Compute local gradient ∇ f i ( x k )
- 5: Transmit ∆ k i = C k ( ∇ f i ( x k ) -g k -1 i )
- 7: end for
- 6: Update g k i = g k -1 i +∆ k i
- 8: Central server computes g k = 1 n ∑ n i =1 g k i via g k i = g k -1 i +∆ k i
- 9: Central server updates x k +1 = x k -γ k g k ∥ g k ∥
- 10: end for
- 11: Output: x K +1

Theorem 1 (Convergence of ||EF21|| ) . Consider Problem (1), where Assumption 1 (lower bound on f ), Assumption 2 (lower bound on f i ), Assumption 3 (generalized smoothness of f i ), and Assumption 4 (contractive compressor) hold. Then, the iterates { x k } generated by ||EF21|| (Algorithm 1) with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for K ≥ 0 and γ &gt; 0 satify

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

√

Theorem 1 establishes the O (1 / K ) convergence in the expectation of gradient norms for ||EF21|| on nonconvex deterministic problems under generalized smoothness. This rate is the same as Theorem 1 of Richtárik et al. [8] for EF21 under traditional smoothness, and does not depend on data heterogeneity conditions in contrast to Crawshaw et al. [25], Liu et al. [26]. Also, our stepsize depends on any positive constant γ 0 , and total iteration count K , without needing to know smoothness constants L 0 , L 1 in contrast to Richtárik et al. [8]. Additionally, if we choose γ 0 = 1 / (8 cL 1 ) , then our convergence bound from Theorem 1 becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Comparisons between ||EF21|| and EF21 under traditional smoothness. For nonconvex, traditional smooth problems, ||EF21|| from Theorem 1 with L 1 = 0 achieves the same O (1 / √ K ) rate in the expectation of gradient norms as EF21 analyzed by Richtárik et al. [8], but with a larger convergence factor of 2 √ 2 . We refer to the derivation and discussion in details in Appendix C.

In the following section, we demonstrate how to integrate normalization into EF21-SGDM [10], an error feedback algorithm that allows each node to compute its local stochastic gradient, for solving nonconvex stochastic problems.

## 5 Normalized Error Feedback with Stochastic Gradients &amp; Momentum ( ||EF21-SGDM|| )

Having established the convergence of ||EF21|| for deterministic optimization, we will next develop a distributed error feedback algorithm that incorporate stochastic gradients and normalization to accommodate generalized smoothness conditions. In particular, we focus on ||EF21-SGDM|| (Algorithm 2),

## Algorithm 2 Normalized Error Feedback with Stochastic Gradients &amp; Momentum ( ||EF21-SGDM|| )

- 1: Input: Stepsizes γ k &gt; 0 and η k ∈ [0 , 1] for k = 0 , 1 , . . . ; starting points x 0 , g -1 i ∈ R d for i ∈ { 1 , 2 , . . . , n } , and v -1 i = ∇ f i ( x 0 i ; ξ 0 i ) with independent random samples ξ i for i ∈ { 1 , 2 , . . . , n } ; α -contractive compressors C k : R d → R d for k = 0 , 1 , . . .
- 2: for each iteration k = 0 , 1 , . . . , K do
- 3: for each client i = 1 , 2 , . . . , n in parallel do
- 4: Compute a local stochastic gradient ∇ f i ( x k ; ξ k i )
- 6: Transmit ∆ k i = C k ( v k i -g k -1 i )
- 5: Update a momentum estimator v k i = (1 -η k ) v k -1 i + η k ∇ f i ( x k ; ξ k i )
- 7: Update g k i = g k -1 i +∆ k i
- 9: Central server computes g k = 1 n ∑ n i =1 g k i via g k i = g k -1 i +∆ k i
- 8: end for
- 10: Central server updates x k +1 = x k -γ k g k ∥ g k ∥
- 11: end for
- 12: Output: x K +1

the normalized version of EF21-SGDM due to Fatkhullin et al. [10]. We also note that ||EF21-SGDM|| recovers many optimization algorithms of interest in the special cases. For instance, it reduces to

- normalized version of EF21 [8], which we call ||EF21|| , when we let η k = 1 and ∇ f i ( x k ; ξ k i ) = ∇ f i ( x k ) ,
- normalized version of EF21-SGD [9], which we call ||EF21-SGD|| , when we let η k = 1 , and
- normalized version of SGDM [50], which we call ||SGDM|| 3 , when we let η k = 1 -β k and C k ( · ) is the identity compressor/mapping.

In the next theorem, we demonstrate that ||EF21-SGDM|| attains the same O (1 /K 1 / 4 ) convergence rate as both EF21-SGDM and ||SGDM|| .

Theorem 2 (Convergence of ||EF21-SGDM|| ) . Consider Problem (1), where Assumption 1 (lower bound on f ), Assumption 2 (lower bound on f i ), Assumption 3 (generalized smoothness of f i ), Assumption 4 (contractive compressor), and Assumption 5 (bounded variance) hold. If g -1 i = 0 for i ∈ { 1 , . . . , n } and

<!-- formula-not-decoded -->

where C α := 1 - √ 1 -α , then the iterates { x k } generated by ||EF21-SGDM|| (Algorithm 2) satisfy for K ≥ 0

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Theorem 2, ||EF21-SGDM|| under generalized smoothness achieves the O (1 /K 1 / 4 ) convergence rate in the expectation of gradient norms. This rate is the same as that of EF21-SGDM , previously analyzed under traditional smoothness by Fatkhullin et al. [10, Theorem 3]. The result holds regardless of the data heterogeneity degree and the mini-batch size. We also notice that the stepsize γ 0 for ||EF21-SGDM|| , unlike in the case of ||EF21|| , depends on the generalized smoothness constant L 1 , and the compression parameter α . However, the considered choice of stepsizes is agnostic to σ and L 0 .

3 This method is also known as NSGD-M .

Furthermore, Theorem 2 with α = 1 (i.e., C k is the identity compressor) implies the convergence bound of the distributed version of normalized SGD with momentum ( ||SGDM|| ) [50] using β = 1 -η :

<!-- formula-not-decoded -->

For the single-node SGDM , where n = 1 and δ inf = 0 , our convergence bound in (5) with γ = Θ( 1 / L 1 ) achieves the O ( L 1 ( f ( x 0 ) -f inf )+ σ + L 0 / L 1 ( K +1) 1 / 4 ) convergence, which matches the rate obtained by Hübler et al. [24, Corollary 3]. Unlike the earlier results for single-node SGDM , our result holds for the multi-node regime. The bound in (5) for multi-node SGDM includes the σ / √ n -term indicating a √ n -fold reduction in the influence of stochastic variance noise σ , and the γL 2 1 δ inf -term accounting for the effect of data heterogeneity.

Novel proof techniques for ||EF21|| and ||EF21-SGDM|| under generalized smoothness. Our analysis demonstrates that ||EF21|| achieves the convergence rate under generalized smoothness equivalent to EF21 under traditional smoothness. However, our proof techniques differ significantly from prior work. We employ different Lyapunov functions. For ||EF21|| , we use V k := f ( x k ) -f inf + A n ∑ n i =1 ∥ ∥ ∇ f i ( x k ) -g k i ∥ ∥ , in constrast to Richtárik et al. [8] that uses V k := f ( x k ) -f inf + B n ∑ n i =1 ∥ ∥ ∇ f i ( x k ) -g k i ∥ ∥ 2 . For ||EF21-SGDM|| , we use V k := f ( x k ) -f inf + C n ∑ n i =1 ∥ ∥ v k i -g k i ∥ ∥ + D n ∑ n i =1 ∥ ∥ v k i -∇ f i ( x k ) ∥ ∥ , unlike Fatkhullin et al. [10] that uses V k := f ( x k ) -f inf + E n ∑ n i =1 ∥ ∥ v k i -g k i ∥ ∥ 2 + F n ∑ n i =1 ∥ ∥ v k i -∇ f i ( x k ) ∥ ∥ 2 . These new Lyapunov functions necessitate the Lyapunov- based convergence analysis, distinct from standard techniques for error feedback methods. Our analysis leverages Lemma 2 to handle generalized smoothness. For ||EF21|| , we rely on Lemma 4. For ||EF21-SGDM|| , we derive a new upper-bound on E [∥ ∥ v k -∇ f ( x k ) ∥ ∥ ] , unlike Fatkhullin et al. [10] to show the √ n -speedup for the term proportional to σ , and utilize non-uniform weights to obtain convergence in the gradient norm.

## 6 Experiments

In this section, we evaluate the performance of ||EF21|| , and compare it against EF21 [8]. We test these algorithms for three nonconvex, generalized smooth problems: the problem of minimizing polynomial functions, the logistic regression problem with a nonconvex regularization term over synthetic and benchmark datasets from LIBSVM [51], and the training of the ResNet-20 [52] model over the CIFAR10 [53] dataset 4 . For all experiments, we use a topk sparsifier, which is a k d -contractive compressor.

## 6.1 Logistic Regression with a Nonconvex Regularizer

First, we consider a logistic regression problem with a nonconvex regularizer, i.e., Problem (1) with

<!-- formula-not-decoded -->

where a i ∈ R d is the i th feature vector of data matrix A ∈ R n × d with its class label b i ∈ {-1 , 1 } , and λ &gt; 0 is a regularization parameter. Here, f ( x ) is nonconvex, and L -smooth with L = ∥ A ∥ 2 / (4 n ) + 2 λ . Also, each f i ( x ) is ˆ L i -smooth with ˆ L i = ∥ a i ∥ 2 / 4 + 2 λ , and generalized smooth with L 0 = 2 λ + λ √ d max i ∥ a i ∥ and L 1 = max i ∥ a i ∥ . The derivations of smoothness parameters can be found in Appendix H.

In this experiment, we initialized x 0 ∈ R d , where each coordinate was drawn from a standard normal distribution N (0 , 1) , and set λ = 0 . 1 . Here, the condition λ &gt; λ min ( A ⊤ A ) / (2 n ) ensures that f ( x ) is nonconvex. We ran ||EF21|| and EF21 on the following datasets: (1) two from LIBSVM [51]: Breast Cancer ( n = 683 , d = 10 , and scaled to [ -1 , 1] ), and a1a ( n = 1605 , d = 123 ); and (2) a synthetically generated dataset ( n = 20 , d = 10 ), where the data matrix A ∈ R n × d had entries drawn from N (0 , 1) , and the class label b i was set to either -1 or 1 with equal probability. For

4 We implemented EF21 and ||EF21|| on training the ResNet-20 model by using PyTorch. Our source codes can be found in the link to error-feedback-generalized-smoothness-paper.

EF21 , we selected the stepsize γ k = 1 / ( L + ˜ L √ β/θ ) with ˜ L = √ ∑ n i =1 ˆ L 2 i /n , θ = 1 - √ 1 -α , and β = (1 -α ) / (1 - √ 1 -α ) , given by Richtárik et al. [8, Theorem 1]. For ||EF21|| , we chose γ k = γ/ √ K +1 with γ &gt; 0 from Theorem 1, by setting γ 0 = 1 , K = 100 for the generated data and Breast Cancer , and K = 400 for a1a . We choose γ 0 = 1 , because ||EF21|| with γ 0 ∈ [1 , 10] converges faster than that with small values of γ 0 (e.g. 0 . 1 ), when we run the algorithm on a single node ( n = 1 ) for minimizing polynomial function and solving logistic regression. We determine K as the smallest number of iterations required to achieve the desired accuracy by performing a grid search with a stepsize of 50 .

Figure 2 shows that ||EF21|| outperforms the traditional EF21 on all evaluated datasets, achieving faster convergence and higher solution accuracy. This improvement results from the fact that the theoretical stepsize for ||EF21|| , as derived in Theorem 1, is larger than the stepsize for EF21 outlined by Richtárik et al. [8, Theorem 1].

Figure 2: Logistic regression with a nonconvex regularizer using normalized ||EF21|| and EF21 . We reported ∥ ∥ ∇ f ( x k ) ∥ ∥ 2 with respect to iteration count k . We used the constant stepsize γ = 1 √ β

<!-- image -->

for EF21 , and γ = √ K +1 , ˆ γ = 1 for ||EF21|| . Here, K = 100 for our generated data (left), and Breast Cancer (middle), while K = 400 for a1a (right).

L +˜ L θ ˆ γ

## 6.2 ResNet20 Training Over CIFAR-10

Next, we trained the ResNet20 [52] model on the CIFAR-10 [53] dataset, which was demonstrated empirically by Zhang et al. [14] to satisfy the ( L 0 , L 1 ) -smoothness condition. In these experiments, we used a topk compressor over 50 , 000 training images, with evaluation on 10 , 000 test images. The dataset was evenly distributed among 5 clients, each using a mini-batch size of 128 . Both algorithms were run for 100 epochs with a constant stepsize γ = 5 . Here, one epoch refers to a full pass through the entire dataset processed by all clients.

From Figure 3, under the same constant stepsize and the topk sparsifier with k = 0 . 01 d , ||EF21|| outperforms EF21 , in terms of convergence speed (in gradient norms and losses) and accuracy, relative to the number of bits communicated from each client to the server. Specifically, ||EF21|| achieved accuracy gains of up to 10 %over EF21 .

Figure 3: ResNet20 training on CIFAR-10 by using EF21 and ||EF21|| under the same stepsize γ = 5 and k = 0 . 1 d for a topk sparsifier.

<!-- image -->

## 7 Conclusion and Future Works

In this paper, we have demonstrated that normalization can be effectively combined with EF21 to develop distributed error feedback algorithms for solving nonconvex optimization problems under generalized smoothness conditions. Specifically, ||EF21|| and ||EF21-SGDM|| achieve convergence

rates of O (1 /K 1 / 2 ) in deterministic settings and O (1 /K 1 / 4 ) in stochastic settings, respectively. These convergence rates match those of the vanilla EF21 and EF21-SGDM algorithms. Unlike previous works on distributed algorithms under generalized smoothness, our analysis does not assume data heterogeneity or impose smoothness-dependent restrictions on the stepsize (in the deterministic case). Finally, our experiments confirm that ||EF21|| exhibits stronger convergence performance compared to the original EF21 , due to its larger allowable stepsizes.

Our work implies many promising research directions. One interesting direction is to extend our convergence results for ||EF21|| and ||EF21-SGDM|| to accommodate decreasing or adaptive stepsize schedules, as the constant stepsizes required by our current analysis can become impractically small when the total number of iterations is large. Another important direction is the development of distributed and federated algorithms that leverage clipping or normalization for minimizing nonconvex generalized smooth functions.

## Acknowledgements

The research reported in this publication was supported by funding from King Abdullah University of Science and Technology (KAUST): i) KAUST Baseline Research Scheme, ii) CRG Grant ORFSCRG12-2024-6460, and iii) Center of Excellence for Generative AI, under award number 5940.

## References

- [1] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics , pages 1273-1282. PMLR, 2017.
- [2] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015.
- [3] Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs. In Fifteenth annual conference of the international speech communication association , 2014.
- [4] Sebastian U Stich, Jean-Baptiste Cordonnier, and Martin Jaggi. Sparsified SGD with memory. Advances in Neural Information Processing Systems , 31, 2018.
- [5] Dan Alistarh, Torsten Hoefler, Mikael Johansson, Nikola Konstantinov, Sarit Khirirat, and Cédric Renggli. The convergence of sparsified gradient methods. Advances in Neural Information Processing Systems , 31, 2018.
- [6] Jiaxiang Wu, Weidong Huang, Junzhou Huang, and Tong Zhang. Error compensated quantized SGD and its applications to large-scale distributed optimization. In International conference on machine learning , pages 5325-5333. PMLR, 2018.
- [7] Eduard Gorbunov, Dmitry Kovalev, Dmitry Makarenko, and Peter Richtárik. Linearly converging error compensated SGD. Advances in Neural Information Processing Systems , 33: 20889-20900, 2020.
- [8] Peter Richtárik, Igor Sokolov, and Ilyas Fatkhullin. EF21: A new, simpler, theoretically better, and practically faster error feedback. Advances in Neural Information Processing Systems , 34: 4384-4396, 2021.
- [9] Ilyas Fatkhullin, Igor Sokolov, Eduard Gorbunov, Zhize Li, and Peter Richtárik. EF21 with bells &amp; whistles: Practical algorithmic extensions of modern error feedback. arXiv preprint arXiv:2110.03294 , 2021.
- [10] Ilyas Fatkhullin, Alexander Tyurin, and Peter Richtárik. Momentum provably improves error feedback! Advances in Neural Information Processing Systems , 36, 2024.

- [11] Kaja Gruntkowska, Alexander Tyurin, and Peter Richtárik. EF21-P and friends: Improved theoretical communication complexity for distributed optimization with bidirectional compression. In International Conference on Machine Learning , pages 11761-11807. PMLR, 2023.
- [12] Yuan Gao, Rustem Islamov, and Sebastian U Stich. EControl: Fast distributed optimization with compression and error control. In The Twelfth International Conference on Learning Representations , 2024.
- [13] Jikai Jin, Bohang Zhang, Haiyang Wang, and Liwei Wang. Non-convex distributionally robust optimization: Non-asymptotic analysis. Advances in Neural Information Processing Systems , 34:2771-2782, 2021.
- [14] Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. In International Conference on Learning Representations , 2020.
- [15] Michael Crawshaw, Mingrui Liu, Francesco Orabona, Wei Zhang, and Zhenxun Zhuang. Robustness to unbounded smoothness of generalized signSGD. Advances in neural information processing systems , 35:9955-9968, 2022.
- [16] Ziyi Chen, Yi Zhou, Yingbin Liang, and Zhaosong Lu. Generalized-smooth nonconvex optimization is as efficient as smooth nonconvex optimization. In International Conference on Machine Learning , pages 5396-5427. PMLR, 2023.
- [17] Anastasia Koloskova, Hadrien Hendrikx, and Sebastian U Stich. Revisiting gradient clipping: Stochastic bias and tight convergence guarantees. In International Conference on Machine Learning , pages 17343-17363. PMLR, 2023.
- [18] Bohang Zhang, Jikai Jin, Cong Fang, and Liwei Wang. Improved analysis of clipping algorithms for non-convex optimization. Advances in Neural Information Processing Systems , 33:1551115521, 2020.
- [19] Amirhossein Reisizadeh, Haochuan Li, Subhro Das, and Ali Jadbabaie. Variance-reduced clipping for non-convex optimization. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1-5. IEEE, 2025.
- [20] Bohan Wang, Yushun Zhang, Huishuai Zhang, Qi Meng, Ruoyu Sun, Zhi-Ming Ma, Tie-Yan Liu, Zhi-Quan Luo, and Wei Chen. Provable adaptivity of adam under non-uniform smoothness. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pages 2960-2969, 2024.
- [21] Haochuan Li, Alexander Rakhlin, and Ali Jadbabaie. Convergence of adam under relaxed assumptions. Advances in Neural Information Processing Systems , 36, 2024.
- [22] Yuki Takezawa, Han Bao, Ryoma Sato, Kenta Niwa, and Makoto Yamada. Parameter-free clipped gradient descent meets polyak. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [23] Shen-Yi Zhao, Yin-Peng Xie, and Wu-Jun Li. On the convergence and improvement of stochastic normalized gradient descent. Science China Information Sciences , 64:1-13, 2021.
- [24] Florian Hübler, Junchi Yang, Xiang Li, and Niao He. Parameter-agnostic optimization under relaxed smoothness. In International Conference on Artificial Intelligence and Statistics , pages 4861-4869. PMLR, 2024.
- [25] Michael Crawshaw, Yajie Bao, and Mingrui Liu. Federated learning with client subsampling, data heterogeneity, and unbounded smoothness: A new algorithm and lower bounds. Advances in Neural Information Processing Systems , 36, 2024.
- [26] Mingrui Liu, Zhenxun Zhuang, Yunwen Lei, and Chunyang Liao. A communication-efficient distributed gradient clipping algorithm for training deep neural networks. Advances in Neural Information Processing Systems , 35:26204-26217, 2022.

- [27] Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi. Error feedback fixes SignSGD and other gradient compression schemes. In International Conference on Machine Learning , pages 3252-3261. PMLR, 2019.
- [28] Hanlin Tang, Chen Yu, Xiangru Lian, Tong Zhang, and Ji Liu. Doublesqueeze: Parallel stochastic gradient descent with double-pass error-compensated compression. In International Conference on Machine Learning , pages 6155-6165. PMLR, 2019.
- [29] Debraj Basu, Deepesh Data, Can Karakus, and Suhas Diggavi. Qsparse-local-SGD: Distributed SGD with quantization, sparsification and local computations. Advances in Neural Information Processing Systems , 32, 2019.
- [30] Zhize Li, Dmitry Kovalev, Xun Qian, and Peter Richtarik. Acceleration for compressed gradient descent in distributed and federated optimization. In International Conference on Machine Learning , pages 5895-5904. PMLR, 2020.
- [31] Xun Qian, Peter Richtárik, and Tong Zhang. Error compensated distributed SGD can be accelerated. Advances in Neural Information Processing Systems , 34:30401-30413, 2021.
- [32] Hanlin Tang, Yao Li, Ji Liu, and Ming Yan. Errorcompensatedx: error compensation for variance reduced algorithms. Advances in Neural Information Processing Systems , 34:18102-18113, 2021.
- [33] Haochuan Li, Jian Qian, Yi Tian, Alexander Rakhlin, and Ali Jadbabaie. Convex and nonconvex optimization under generalized smoothness. Advances in Neural Information Processing Systems , 36, 2024.
- [34] Daniel Levy, Yair Carmon, John C Duchi, and Aaron Sidford. Large-scale methods for distributionally robust optimization. Advances in Neural Information Processing Systems , 33: 8847-8860, 2020.
- [35] Olivier Devolder, François Glineur, and Yurii Nesterov. First-order methods of smooth convex optimization with inexact oracle. Mathematical Programming , 146:37-75, 2014.
- [36] Heinz H Bauschke, Jérôme Bolte, and Marc Teboulle. A descent lemma beyond lipschitz gradient continuity: first-order methods revisited and applications. Mathematics of Operations Research , 42(2):330-348, 2017.
- [37] Vien V Mai and Mikael Johansson. Stability and convergence of stochastic gradient clipping: Beyond lipschitz continuity and smoothness. In International Conference on Machine Learning , pages 7325-7335. PMLR, 2021.
- [38] Eduard Gorbunov, Nazarii Tupitsa, Sayantan Choudhury, Alen Aliev, Peter Richtárik, Samuel Horváth, and Martin Takáˇ c. Methods for convex ( L 0 , L 1 ) -smooth optimization: Clipping, acceleration, and adaptivity. In The Thirteenth International Conference on Learning Representations , 2025.
- [39] Wei Jiang, Sifan Yang, Wenhao Yang, and Lijun Zhang. Efficient sign-based optimization: Accelerating convergence via variance reduction. Advances in Neural Information Processing Systems , 37:33891-33932, 2024.
- [40] Yurii Nesterov et al. Lectures on convex optimization , volume 137. Springer, 2018.
- [41] Amir Beck. First-order methods in optimization . SIAM, 2017.
- [42] Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. PowerSGD: practical low-rank gradient compression for distributed optimization. Advances in Neural Information Processing Systems , 32, 2019.
- [43] Mher Safaryan, Rustem Islamov, Xun Qian, and Peter Richtarik. FedNL: Making newton-type methods applicable to federated learning. In International Conference on Machine Learning , pages 18959-19010. PMLR, 2022.

- [44] Mher Safaryan, Egor Shulgin, and Peter Richtárik. Uncertainty principle for communication compression in distributed and federated learning and the search for an optimal compressor. Information and Inference: A Journal of the IMA , 11(2):557-580, 2022.
- [45] Aleksandr Beznosikov, Samuel Horváth, Peter Richtárik, and Mher Safaryan. On biased compression for distributed learning. Journal of Machine Learning Research , 24(276):1-50, 2023.
- [46] Yury Demidovich, Grigory Malinovsky, Igor Sokolov, and Peter Richtárik. A guide through the zoo of biased SGD. Advances in Neural Information Processing Systems , 36:23158-23171, 2023.
- [47] Arkadii S Nemirovski, Anatoli B Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization , 19(4):1574-1609, 2009.
- [48] Saeed Ghadimi and Guanghui Lan. Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization i: A generic algorithmic framework. SIAM Journal on Optimization , 22(4):1469-1492, 2012.
- [49] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization , 23(4):2341-2368, 2013.
- [50] Ashok Cutkosky and Harsh Mehta. Momentum improves normalized SGD. In International conference on machine learning , pages 2260-2268. PMLR, 2020.
- [51] Chih-Chung Chang and Chih-Jen Lin. LIBSVM: a library for support vector machines. ACM transactions on intelligent systems and technology (TIST) , 2(3):1-27, 2011.
- [52] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 770-778, 2016. doi: 10.1109/CVPR.2016.90.
- [53] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

## Contents

| 1   | Introduction                                                                                      | Introduction                                                                                      |   1 |
|-----|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-----|
|     | 1.1                                                                                               | Contributions . . . . . . . . . . . . . . . . . . . . . .                                         |   2 |
| 2   | Related Works                                                                                     | Related Works                                                                                     |   3 |
| 3   | Preliminaries                                                                                     | Preliminaries                                                                                     |   4 |
| 4   | Normalized Error Feedback ( &#124;&#124;EF21&#124;&#124; )                                        | Normalized Error Feedback ( &#124;&#124;EF21&#124;&#124; )                                        |   5 |
| 5   | Normalized Error Feedback with Stochastic Gradients &Momentum ( &#124;&#124;EF21-SGDM&#124;&#124; | Normalized Error Feedback with Stochastic Gradients &Momentum ( &#124;&#124;EF21-SGDM&#124;&#124; |   6 |
| 6   | Experiments                                                                                       | Experiments                                                                                       |   8 |
|     | 6.1                                                                                               | Logistic Regression with a Nonconvex Regularizer . .                                              |   8 |
|     | 6.2                                                                                               | ResNet20 Training Over CIFAR-10 . . . . . . . . . .                                               |   9 |
| 7   | Conclusion and Future Works                                                                       | Conclusion and Future Works                                                                       |   9 |
| A   | Lemmas                                                                                            | Lemmas                                                                                            |  14 |

| B   | Convergence Proof for &#124;&#124;EF21&#124;&#124; (Theorem        | 1)                                                                 |   17 |
|-----|--------------------------------------------------------------------|--------------------------------------------------------------------|------|
|     | B.1                                                                | Proof of Theorem 1 . . . . . . . . . . . . . . . .                 |   18 |
| C   | Discussion on Theorem 1                                            | Discussion on Theorem 1                                            |   19 |
| D   | Convergence of &#124;&#124;EF21&#124;&#124; for a Single-node Case | Convergence of &#124;&#124;EF21&#124;&#124; for a Single-node Case |   20 |
| E   | Convergence of &#124;&#124;EF21-SGDM&#124;&#124; (Theorem 2)       | Convergence of &#124;&#124;EF21-SGDM&#124;&#124; (Theorem 2)       |   21 |
|     | E.1                                                                | Auxiliary Lemmas . . . . . . . . . . . . . . . .                   |   21 |
|     | E.2                                                                | Proof of Theorem 2 . . . . . . . . . . . . . . . .                 |   26 |
| F   | Extension to Strongly Convex and Convex Problems                   | Extension to Strongly Convex and Convex Problems                   |   30 |
| G   | Additional Experimental Results                                    | Additional Experimental Results                                    |   31 |
|     | G.1                                                                | Minimization of Nonconvex Polynomial Functions                     |   31 |
|     | G.2                                                                | ResNet20 Training over CIFAR-10 . . . . . . . .                    |   32 |
| H   | Omitted Proof for Smoothness Parameters of Logistic Regression     | Omitted Proof for Smoothness Parameters of Logistic Regression     |   34 |

## A Lemmas

In this section, we introduce useful lemmas for our analysis. Lemmas 1 and 2 introduce inequalities by generalized smoothness, while Lemmas 3 and 4 present the descent inequality and convergence rate, respectively, when the normalized gradient descent update is applied.

Lemma 1. Let each f i ( x ) be generalized smooth with parameters L 0 , L 1 &gt; 0 , and lower bounded by f inf i , and let f ( x ) = 1 n ∑ n i =1 f i ( x ) . Then, for any x, y ∈ R d

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The first and second statements are derived in Chen et al. [16, Proposition 3.2]. Next, the third inequality follows from [38, Lemma 2.2]. Finally, averaging (7) for i = 1 , . . . , n and taking into account that f ( x ) = 1 n ∑ n i =1 f i ( x ) , we get (9).

Lemma 2. Let f i ( x ) be generalized smooth with parameters L 0 , L 1 &gt; 0 , and lower bounded by f inf i , and let f ( x ) be lower bounded by f inf . Then, for any x ∈ R d

<!-- formula-not-decoded -->

Proof. By the ( L 0 , L 1 ) -smoothness of f i ( x ) ,

<!-- formula-not-decoded -->

This condition implies

<!-- formula-not-decoded -->

Finally, by the fact that f ( x ) = 1 n ∑ n i =1 f i ( x ) ,

<!-- formula-not-decoded -->

Lemma 3. Let f ( x ) = 1 n ∑ n i =1 f i ( x ) , where each f i ( x ) is generalized smooth with parameters L 0 , L 1 &gt; 0 . Let x k +1 = x k -γ k ∥ v k ∥ v k for γ k &gt; 0 . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let each f i ( x ) be generalized smooth with L 0 , L 1 &gt; 0 , and f ( x ) = 1 n ∑ n i =1 f i ( x ) . By (9) of Lemma 1, and by the fact that x k +1 = x k -γ k ∥ v k ∥ v k for γ k &gt; 0 ,

<!-- formula-not-decoded -->

where we reach the last inequality by Cauchy-Schwarz inequality. Next, since

<!-- formula-not-decoded -->

we get

<!-- formula-not-decoded -->

Lemma 4. Let { V k } k ≥ 0 , { W k } k ≥ 0 be non-negative sequences satisfying

<!-- formula-not-decoded -->

for γ, b 1 , b 2 , b 3 &gt; 0 . Then,

<!-- formula-not-decoded -->

Proof. Define β k = β k -1 1+ b 1 exp( L 1 γ ) γ 2 for k = 0 , 1 , . . . and β -1 = 1 . Then, we can show that β k = 1 (1+ b 1 exp( L 1 γ ) γ 2 ) k +1 for k = 0 , 1 , . . . , and that

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

By the fact that β -1 = 1 , β K &gt; 0 , and V k +1 ≥ 0 ,

<!-- formula-not-decoded -->

Next, since

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

## B Convergence Proof for ||EF21|| (Theorem 1)

In this section, we derive the convergence rate results of ||EF21|| . We start with the following lemma technical lemma.

Lemma 5. Let Assumptions 3 and 4 hold. Then, the iterates { x k } generated by ||EF21|| (Algorithm 1) satisfy

<!-- formula-not-decoded -->

Proof. From the definition of the Euclidean norm, and by taking the expectation conditioned on x k +1 , g k i , and by the update of g k i from Algorithm 1

<!-- formula-not-decoded -->

where we use the concavity of the square root function, and Jensen's inequality for the concave function, i.e., E[ f ( x )] ≤ f (E[ x ]) if f ( x ) is concave. By the α -contractive property of compressors in (3), by the fact that ∥ ∥ ∇ f i ( x k +1 ) -g k i ∥ ∥ is a constant conditioned on x k +1 , g k i , and then by the triangle inequality, we have

<!-- formula-not-decoded -->

By the generalized smoothness of f i ( x ) in (2), and by the fact that x k +1 = x k -γ k g k ∥ g k ∥ ,

<!-- formula-not-decoded -->

Let γ k &gt; 0 be constants conditioned on x k +1 , g k i . Then, by the tower property, i.e.,

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

This concludes the proof.

Next, we present the following descent lemma for ||EF21|| .

Lemma 6. Let Assumptions 1-4 hold. Then, the iterates { x k } generated by ||EF21|| (Algorithm 1) satisfy

<!-- formula-not-decoded -->

Proof. For brevity, let A k = 2 γ k 1 - √ 1 -α . Then, we have V k := f ( x k ) -f inf + A k 1 n ∑ n i =1 ∥ ∥ ∇ f i ( x k ) -v k i ∥ ∥ , and from Lemma 3, we derive

<!-- formula-not-decoded -->

Identities ∇ f ( x k ) = 1 n ∑ n i =1 ∇ f i ( x k ) and g k = 1 n ∑ n i =1 g k i and the triangle inequality imply E [ V k +1 ] ≤ E [ f ( x k ) -f inf ] -γ k E [∥ ∥ ∇ f ( x k ) ∥ ∥ ]

<!-- formula-not-decoded -->

Next, we apply (11):

<!-- formula-not-decoded -->

If A k = 2 γ k 1 - √ 1 -α , and γ k satisfies γ k +1 ≤ γ k , then

Therefore,

<!-- formula-not-decoded -->

where c i = L i 2 +2 1 -αL i 1 - √ 1 -α for i = 0 , 1 .

## B.1 Proof of Theorem 1

Now, we are ready to prove Theorem 1. From Lemma 6 and 2, and by the fact that c 1 L 0 /L 1 = c 0 , we have

<!-- formula-not-decoded -->

where B = 2 c 0 + 8 c 1 L 1 n ∑ n i =1 ( f inf -f inf i ) . Using the fact that f ( x k ) -f inf ≤ V k , we derive

<!-- formula-not-decoded -->

Applying Lemma 4 with V k = E [ V k ] , W k = E [∥ ∥ ∇ f ( x k ) ∥ ∥ ] , b 1 = 8 c 1 L 1 , b 2 = 1 , and b 3 = B , we get

<!-- formula-not-decoded -->

Finally, if γ = γ 0 √ K +1 with γ 0 &gt; 0 , then exp( L 1 γ k ) ≤ exp( L 1 γ 0 ) , and thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Discussion on Theorem 1

In this section, we compare the convergence bound between ||EF21|| and EF21 under traditional smoothness. For nonconvex, traditional smooth problems, ||EF21|| from Theorem 1 with L 1 = 0 achieves the same O (1 / √ K ) rate in the expectation of gradient norms as EF21 analyzed by Richtárik et al. [8], but with a larger convergence factor. We prove this by assuming ∇ f i ( x 0 ) = g 0 i for all i . That is, Theorem 1 with L 0 = L , L 1 = 0 , γ 0 = √ ( f ( x 0 ) -f inf ) / (2 b ) , and b = L 2 +2 √ 1 -αL 1 - √ 1 -α implies that ||EF21|| achieves

<!-- formula-not-decoded -->

On the other hand, EF21 attains from Theorem 1 of [8] with L i = ˜ L = L (i.e., f i ( x ) has the same smoothness constant as f ( x ) ), and ˆ x K being chosen from the iterates x 0 , x 1 , . . . , x K uniformly at random

<!-- formula-not-decoded -->

In conclusion, the convergence bound of ||EF21|| is slower by a factor of 2 √ 2 than the original EF21 for nonconvex, L -smooth problems.

## D Convergence of ||EF21|| for a Single-node Case

In this section, we provide the convergence of ||EF21|| for a single-node case. In particular, the algorithm enjoys the O (1 /K ) convergence up to the error of c 0 γ 1 -c 1 exp( L 1 γ ) γ . In contrast to Theorem 1 for multi-node ||EF21|| , the next result for single-node ||EF21|| applies for any γ k = γ ∈ (0 , 1 / ( βc 1 )) with β ≥ 2 , c 1 = L 1 2 +2 √ 1 -αL 1 1 - √ 1 -α , and α ∈ (0 , 1] .

Theorem 3. Let Assumptions 1-4 hold. Then, the iterates { x k } generated by ||EF21|| (Algorithm 1) with n = 1 , γ k = γ = 1 / ( βc 1 ) and β ≥ 2 satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. In the single-node case, Lemma 5 implies

<!-- formula-not-decoded -->

Next, for brevity, let A k = 2 γ k 1 - √ 1 -α . Then, we have V k := f ( x k ) -f inf + A k 1 n ∑ n i =1 ∥ ∥ ∇ f i ( x k ) -g k i ∥ ∥ , and from Lemma 3, we derive

<!-- formula-not-decoded -->

If A k = 2 γ k 1 - √ 1 -α and γ k satisfies γ k +1 ≤ γ k , then

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

where c i = L i 2 +2 √ 1 -αL i 1 - √ 1 -α for i = 0 , 1 .

Finally, taking γ k = γ = 1 / ( βc 1 ) for β ≥ 2 , we get c 1 exp( L 1 γ ) γ = exp( L 1 / ( βc 1 )) / β ≤ exp(2 /β ) /β ≤ 0 . 7 &lt; 1 , and

<!-- formula-not-decoded -->

Rearranging the terms, we derive

<!-- formula-not-decoded -->

Noticing that V k ≥ 0 , we complete the proof.

## E Convergence of ||EF21-SGDM|| (Theorem 2)

In this section, we derive the convergence rate results of ||EF21-SGDM|| . We first introduce auxiliary lemmas in Section E.1, and later prove the convergence theorem (Theorem 2) in Section E.2.

## E.1 Auxiliary Lemmas

Now, we provide useful lemmas for analyzing ||EF21-SGDM|| . First, Lemma 7 shows the descent inequality of the normalized gradient descent update under Assumption 3 (generalized smoothness of f i ). Second, Lemmas 8 and 9 provide the upper-bound of the Euclidean distance between v k i and g k i , and of the Euclidean distance between v k i and ∇ f i ( x k ) , respectively.

Lemma 7. Consider the iterates { x k } generated by Algorithm 2. If Assumption 3 holds, then for any γ k &gt; 0 , η k ∈ [0 , 1] ,

<!-- formula-not-decoded -->

Proof. Applying the triangle inequality in Lemma 3, i.e., ∥ ∥ ∇ f ( x k ) -g k ∥ ∥ ≤ ∥ ∥ ∇ f ( x k ) -v k ∥ ∥ + ∥ ∥ v k -g k ∥ ∥ , we get

<!-- formula-not-decoded -->

which concludes the proof.

Lemma 8. Consider the iterates { x k } generated by Algorithm 2. If Assumptions 3, 4, and 5 hold, then for γ k &gt; 0 , η k ∈ [0 , 1] , and k ≥ 0 ,

<!-- formula-not-decoded -->

Proof. Taking conditional expectation with fixed F k +1 = { v k +1 i , x k +1 , g k i } , using the concavity of the squared root of the function, and applying the definition of g k i in Algorithm 2, we have

<!-- formula-not-decoded -->

Next, let γ k = γ &gt; 0 , and η k = η ∈ [0 , 1] . By the fact that v k +1 i , g k i are constants being conditioned on F k +1 , and by the triangle inequality,

<!-- formula-not-decoded -->

Here, the equality comes from the definition of v k +1 i in Algorithm 2. Next, by the triangle inequality,

<!-- formula-not-decoded -->

Next, using x k +1 -x k = -γ k g k ∥ g k ∥ , and taking the expectation, we obtain

<!-- formula-not-decoded -->

Finally, since

<!-- formula-not-decoded -->

we derive

<!-- formula-not-decoded -->

This concludes the proof.

Lemma 9. Consider the iterates { x k } generated by Algorithm 2. If Assumptions 3, and 5 hold, then for any γ k ≡ γ &gt; 0 , η k ≡ η , and k ≥ 0 ,

<!-- formula-not-decoded -->

In addition, for any k ≥ 0 ,

<!-- formula-not-decoded -->

Proof. We prove the result using the arguments similar to those given in the proof of Theorem 1 from Cutkosky and Mehta [50]. From the definition of v k +1 i , we have the following recursion for any k ≥ 0 :

<!-- formula-not-decoded -->

Next, from the recursion of v k +1 i , we obtain the following recursion for k ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Unrolling the recursion for H k i , we derive

<!-- formula-not-decoded -->

Averaging the above inequality, we get

<!-- formula-not-decoded -->

Next, taking the Euclidean norm, using the triangle inequality, and then taking the expectation, we obtain

<!-- formula-not-decoded -->

To bound E [∥ ∥ H k +1 ∥ ∥ ] , we need to bound the expectation of the last two terms. First, we bound term A 1 . Using the fact that ∥ G t ∥ ≤ 1 n ∑ n i =1 ∥ G t i ∥ , and the definition of G t i , we obtain

<!-- formula-not-decoded -->

Next, we bound term A 2 . Jensen's inequality and the tower property of the conditional expectation imply

<!-- formula-not-decoded -->

Moreover, due to independence of { ξ t i } n i =1 , we have

<!-- formula-not-decoded -->

Therefore, plugging the derived upper-bounds for A 1 , and for A 2 into (16), we obtain

<!-- formula-not-decoded -->

which is equivalent to (13).

To derive (14), we make a step back to the recursion from (15), which implies

<!-- formula-not-decoded -->

Next, we derive the upper bounds for B 1 and B 2 . For B 1 , we have

<!-- formula-not-decoded -->

and for B 2 , we obtain

<!-- formula-not-decoded -->

Plugging the derived upper bounds for B 1 and B 2 into (17) and using 1 -η ≤ 1 , we get

<!-- formula-not-decoded -->

which is equivalent to (14).

## E.2 Proof of Theorem 2

Now, we are ready to prove Theorem 2. For convenience, we introduce new notation:

<!-- formula-not-decoded -->

Using the new notation and noticing that E [ ∥ v k -g k ∥ ] ≤ A k , we rewrite the results of Lemmas 7, 8, and 9 as

<!-- formula-not-decoded -->

.

Moreover, since γ = γ 0 ( K +1) 3 / 4 with γ 0 ≤ 1 2 L 1 , we have exp( L 1 γ ) ≤ exp( L 1 γ 0 ) ≤ 2 and the above inequalities can be further simplified as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we introduce the Lyapunov function V k defined for any k ≥ 0 as

√

<!-- formula-not-decoded -->

α

where

a

:=

1

- √

γ

-

1

α

and

c

:=

a

1

-

. Then, using (18), (19), (21), we get

<!-- formula-not-decoded -->

2

To proceed, we rearrange the terms:

<!-- formula-not-decoded -->

Since a = 2 γ 1 - √ 1 -α , we have 2 γ a + √ 1 -α = 1 and

<!-- formula-not-decoded -->

Next, we bound B k using (20) and δ k ≤ V k :

<!-- formula-not-decoded -->

Summing up the above inequality with weights β k := ( 1 + 64 L 2 1 γ 2 1 - √ 1 -α + 32 L 2 1 γ 2 η ) -( k +1) for k = 0 , . . . , K and denoting S K := ∑ K k =0 β k and β -1 := 1 , we get

<!-- formula-not-decoded -->

By definition of β k , we have β k ≤ β k -1 and, in particular, β k ≤ 1 for all k ≥ 0 . Using these inequalities, we continue the derivation as follows:

<!-- formula-not-decoded -->

Rearranging the terms and dividing both sides of the above inequality by γS K , we obtain

<!-- formula-not-decoded -->

where in the last inequality we use V K +1 ≥ 0 and β -1 = 1 . Next, we estimate S K :

<!-- formula-not-decoded -->

Since η = 1 ( K +1) 1 / 2 and γ = γ 0 ( K +1) 3 / 4 with γ 0 ≤ 1 16 L 1 min { ( K +1) 1 / 2 (1 - √ 1 -α ) , 1 } , we have 32 L 2 1 γ 2 ( K +1) η ≤ 1 4 and 64 L 2 1 γ 2 ( K +1) 1 - √ 1 -α ≤ 1 4 . Plugging these inequalities into (23), we get S K ≥ ( K +1) / exp( 1 / 2 ) ≥ ( K +1) / 2 . Using this lower bound for S K and η = 1 ( K +1) 1 / 2 , γ = γ 0 ( K +1) 3 / 4 in

(22) , we get

<!-- formula-not-decoded -->

For the convenience, we define C α := 1 - √ 1 -α . Then, by definition of V 0 , we have

<!-- formula-not-decoded -->

Moreover, since g -1 i = 0 and v -1 i = ∇ f i ( x 0 i ; ξ 0 i ) for all i = 1 , . . . , n with independent { ξ 0 i } n i =1 , we have v 0 i = ∇ f i ( x 0 i ; ξ 0 i ) and g 0 i = C 0 ( ∇ f i ( x 0 i ; ξ 0 i )) for all i = 1 , . . . , n and

<!-- formula-not-decoded -->

Using these inequalities, we get

<!-- formula-not-decoded -->

which concludes the proof since 1 -C α C α ≤ 2 √ 1 -α α and 1 C α ≤ 1 α

## F Extension to Strongly Convex and Convex Problems

Our current analysis for ||EF21|| and ||EF21-SGDM|| , which are initially developed for minimizing non-convex functions, can be extended to strongly convex and convex functions.

Strongly convex problems. We can extend the convergence for ||EF21|| and ||EF21-SGDM|| to minimize strongly convex functions. Applying the µ -strong convexity condition of the function f , i.e. ∥ ∥ ∇ f ( x k ) ∥ ∥ 2 ≥ 2 µ ( f ( x k ) -f ( x ⋆ )) , where x ⋆ = arg min x ∈ R d f ( x ) , into the convergence bounds in Theorems 1 and 2 yields the convergence results in min k =0 , 1 ,...,K E [ √ f ( x k ) -f ( x ⋆ ) ] . However, these results do not imply the standard exponential convergence typically expected in strongly convex problems. This theoretical gap suggests a need for new analytical techniques, which involves tighter Lyapunov functions or more refined descent inequalities tailored to strongly convex functions.

Convex problems. We can extend the convergence for minimizing convex functions. This can be achieved by assuming that there exists the iterates { x k } satisfying ∥ ∥ x k -x ⋆ ∥ ∥ ≤ R for some R &gt; 0 . Hence, the convexity of the function f implies that

<!-- formula-not-decoded -->

Applying the above inequality to Theorems 1 and 2 yields the convergence bounds in min k =0 , 1 ,...,K E [ f ( x k ) -f ( x ⋆ ) ] .

## G Additional Experimental Results

In this section, we provide additional results for minimizing nonconvex polynomial functions, and for training the ResNet-20 model over the CIFAR-10 dataset.

## G.1 Minimization of Nonconvex Polynomial Functions

We ran ||EF21|| and EF21 in a single-node setting ( n = 1 ) for solving the following problem:

<!-- formula-not-decoded -->

where a i &gt; 0 , i = 1 , . . . , d , λ &gt; 0 .

Let us show that f ( x ) is non-convex (for the specific choice of a i ) and ( L 0 , L 1 ) -smooth. First, we prove that f ( x ) is non-convex. Indeed,

<!-- formula-not-decoded -->

is not positive definite matrix if we choose a i = λ 24 , x i = ± 1 for i = 1 , . . . , d . Second, we find L 0 , L 1 &gt; 0 such that

<!-- formula-not-decoded -->

This condition is equivalent to Assumption 3 (generalized smoothness) with L 0 , L 1 [16, Theorem 1]. Let us fix some L 1 &gt; 0 and choose L 0 = 9 λd 2 2 L 2 1 +2 λ . Since ∇ 2 h ( x ) ≼ 2 λI ,

<!-- formula-not-decoded -->

Also, notice that

<!-- formula-not-decoded -->

where (*) results from the fact that ∥ x ∥ 1 ≤ √ d ∥ x ∥ for x ∈ R d . Our goal is to show that

<!-- formula-not-decoded -->

To show this, we consider two cases: if | x i | ≤ 3 √ d L 1 , and otherwise.

1. If | x i | ≤ 3 √ d L 1 for all i = 1 , . . . , d , then 12 a i x 2 i ≤ 108 a i d L 2 1 . Thus, 12 ( a 1 x 2 1 + . . . + a d x 2 d ) ≤ 108 λd 2 24 L 2 1 = ˜ L 0 .
2. If | x j | &gt; 3 √ d L 1 for some j = 1 , . . . , d , then 12 a j x 2 j &lt; 4 L 1 √ d a j | x j | 3 , and the sum of the remaining terms (such that | x i | ≤ 3 √ d L 1 ) in 12 ( a 1 x 2 1 + . . . + a d x 2 d ) can be upper bounded by ˜ L 0 .

In conclusion, f ( x ) is ( L 0 , L 1 ) -smooth, where L 1 is any positive constant and L 0 = 9 λd 2 2 L 2 1 +2 λ .

Additionally, we can show that under certain additional constraints, f ( x ) is L -smooth with L = λ √ dD 2 2 +2 λ . If | x i | ≤ D for all i = 1 , . . . , d , then

<!-- formula-not-decoded -->

In the experiments, we estimate D based on the initial point x 0 ∈ R d .

In the following experiments, we used a topk sparsifier with k = 1 and α = k/d , setting d = 4 , L 1 = { 1 , 4 , 8 } , and L 0 = 4 (adjusting λ to maintain a constant L 0 ). The initial values x 0 were drawn from a normal distribution, x 0 i ∼ N (20 , 1) for i = 1 , . . . , d , with D estimated as 20. For EF21, we set γ k = 1 L + L √ β θ , using θ = 1 - √ 1 -α and β = 1 -α 1 - √ 1 -α , according to Theorem 1 of [8]. For √

||EF21|| , we chose γ k = 1 2 c 1 with c 1 = L 1 2 + 2 1 -αL 1 1 - √ 1 -α from Theorem 3, and γ k = γ 0 √ K +1 with γ 0 &gt; 0 , as specified in Theorem 1 with n = 1 .

The impact of γ 0 and K on the convergence of ||EF21|| . First, we investigate the impact of γ 0 and K on the convergence of ||EF21|| . Weevaluated γ 0 from the set { 0 . 1 , 1 , 10 } , and plotted the histogram representing the number of iterations required to achieve the target accuracy of ∥∇ f ( x ) ∥ 2 &lt; ϵ with ϵ = 10 -4 , using the stepsize rule γ = γ 0 √ K +1 . For each γ 0 , we determined K as the minimum number of iterations required to achieve the desired accuracy, found through a grid search with step sizes of 500 for γ 0 = 1 , 10 and 5000 for γ 0 = 0 . 1 . From Figure 4, for small values of γ 0 , such as

Figure 4: Number of iterations required to achieve the desired accuracy, ∥∇ f ( x ) ∥ 2 &lt; ϵ , ϵ = 10 -4 , using ||EF21|| with γ = γ 0 √ K +1 for different values of L 0 and L 1 .

<!-- image -->

0 . 1 , significantly more iterations are required to reach convergence compared to γ 0 values of 1 and 10 , which show similar performance (with the exception of the L 0 = 4 , L 1 = 1 case, where γ 0 = 10 converges faster). Based on this observation, we use γ 0 = 1 in all subsequent experiments and adjust only K to achieve convergence, identifying the minimum number of iterations needed to reach the target accuracy through a grid search with a step size of 500 .

Comparisons between EF21 and ||EF21|| . Next, we evaluate the performance of EF21 and ||EF21|| for a fixed L 0 = 4 and varying L 1 values of {1, 4, 8}. From Figure 1, ||EF21|| , regardless of the chosen stepsize γ , achieves the desired accuracy ∥∇ f ( x ) ∥ 2 &lt; ϵ with ϵ = 10 -4 faster than EF21 . Initially, however, EF21 converges more quickly, likely because ||EF21|| employs normalized gradients, which can be slower at the start due to the large gradients when the initial point is far from the stationary point. Moreover, as L 1 increases, both methods show slower convergence.

## G.2 ResNet20 Training over CIFAR-10

We included additional experimental results from running EF21 and ||EF21|| for training the ResNet20 model over the CIFAR-10 dataset. The parameter details were set to be the same as those in Section 6.2, with the exception that we vary k = 0 . 01 d, 0 . 5 d for a topk sparsifier. From Figures 5 and 6, ||EF21|| attains a higher accuracy improvement than EF21 , across different sparsification levels k .

Figure 5: ResNet20 training on CIFAR-10 by using EF21 and ||EF21|| under the same stepsize γ = 5 and k = 0 . 01 d for a topk sparsifier.

<!-- image -->

Figure 6: ResNet20 training on CIFAR-10 by using EF21 and ||EF21|| under the same stepsize γ = 5 and k = 0 . 05 d for a topk sparsifier.

<!-- image -->

## H Omitted Proof for Smoothness Parameters of Logistic Regression

In this section, we prove the generalized smoothness parameters L 0 , L 1 for logistic regression problems with a nonconvex regularizer, which are the following problems

<!-- formula-not-decoded -->

where a i ∈ R d is the i th feature vector of matrix A with its class label b i ∈ {-1 , 1 } , λ &gt; 0 .

First, we can prove that f ( x ) is L -smooth with L = 1 4 n ∥ A ∥ 2 +2 λ , and that each f i ( x ) is ˆ L i -smooth with ˆ L i = 1 4 ∥ a i ∥ 2 +2 λ .

Next, we show that each f i ( x ) is generalized smooth with L 0 = 2 λ + λ √ d max i ∥ a i ∥ and L 1 = max i ∥ a i ∥ , when the Hessian exists. By the fact that

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

After adding the nonconvex regularizer h ( x ) , we can show the following inequalities:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

By combining inequalities (25), (26), and (27), we obtain

<!-- formula-not-decoded -->

In conclusion, ∥ ∥ ∇ 2 f i ( x ) ∥ ∥ ≤ L 0 + L 1 ∥∇ f i ( x ) ∥ with L 0 ≤ 2 λ + λ √ d ∥ a i ∥ , and L 1 ≤ ∥ a i ∥ . This condition is equivalent to Assumption 3 (generalized smoothness) with L 0 , L 1 [16, Theorem 1].

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We develop error feedback algorithms that attain the first convergence guarantees under generalized smoothness, suitable for deep neural networks. Unlike existing works, we do not assume unrealistically strong assumptions for distributed settings. These claims are stated explicitly in the abstract and the introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Our results are limited to constant step sizes. In the conclusion, we identify these as limitations and propose promising future research directions, including extending our algorithms to incorporate decreasing or adaptive stepsizes.

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

Justification: All assumptions used in this work are detailed in the Preliminaries section. Complete proofs for all theorems and corollaries are provided in the appendix.

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

Justification: Details of the experimental setups were provided, encompassing the neural network models employed, the datasets utilized, the partitioning of data into training and validation sets, the specific hyperparameters chosen, and the computational infrastructure used.

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

Answer: [No]

Justification: The data and model sources are open and cited. We provide necessary details for reproducibility.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so âC´ sNoâCt' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Details regarding training and testing procedures, including data splits and hyperparameter settings, are provided in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Error bars are not reported because it would be too computationally expensive to run all the experiments multiple times.

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

## Answer: [No]

Justification: Our experiments on (1) minimizing simple functions and logistic functions, and on (2) ResNet20 training can be run on a machine with a single GPU.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The contribution of this paper is to provide the first convergence guarantee of error feedback algorithms for problems under generalized smoothness for deep neural network training. We ensure full reproducibility and fair comparisons by providing comprehensive experimental details in the appendix.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This paper introduces distributed error feedback methods for training deep neural networks. These methods utilize normalization to stabilize convergence under generalized smoothness conditions, which effectively model the challenges of neural network training. Crucially, they maintain the convergence rates of their standard smoothness counterparts without requiring unrealistic assumptions, such as bounded data heterogeneity or smoothness-dependent stepsize restrictions (in the deterministic setting).

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes]

Justification: The data and models utilized in this work are publicly available, aligning with open-access principles.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We acknowledge the sources of our datasets, the ResNet models, and the ResNet training implementation by citing the creators' respective publications. Furthermore, any modifications made to the software for our specific research investigation were done in accordance with the software's license and terms of use.

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

Justification: This paper does not introduce entirely new datasets or models as assets. Instead, we provide a detailed description of our novel algorithms, along with the specific datasets and model architectures used (which are publicly available). Our comprehensive implementation details should be sufficient for others to reproduce and modify our algorithms for implementation and testing.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
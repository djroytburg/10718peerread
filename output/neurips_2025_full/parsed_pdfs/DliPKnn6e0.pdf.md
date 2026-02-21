## Schr¨ odinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres

Samuel Howard Department of Statistics University of Oxford

Peter Potaptchik Department of Statistics University of Oxford

## Abstract

Recent advances in flow-based generative modelling have provided scalable methods for computing the Schr¨ odinger Bridge (SB) between distributions, a dynamic form of entropy-regularised Optimal Transport (OT) for the quadratic cost. The successful Iterative Markovian Fitting (IMF) procedure solves the SB problem via sequential bridge-matching steps, presenting an elegant and practical approach with many favourable properties over the more traditional Iterative Proportional Fitting (IPF) procedure. Beyond the standard setting, optimal transport can be generalised to the multi-marginal case in which the objective is to minimise a cost defined over several marginal distributions. Of particular importance are costs defined over a tree structure, from which Wasserstein barycentres can be recovered as a special case. In this work, we extend the IMF procedure to solve for the tree-structured SB problem. Our resulting algorithm inherits the many advantages of IMF over IPF approaches in the tree-based setting. In the case of Wasserstein barycentres, our approach can be viewed as extending the widely used fixed-point approach to use flow-based entropic OT solvers, while requiring only simple bridge-matching steps at each iteration. Our code is available at https: //github.com/samuel-howard/Tree\_SB\_Matching\_Barycentres .

## 1 Introduction

Transporting mass between two distributions is a ubiquitous problem with numerous applications in machine learning and beyond. Optimal Transport (OT) (Santambrogio, 2015; Peyr´ e and Cuturi, 2019) provides a principled approach for such problems, by seeking to minimise the total cost of transportation according to a chosen cost function. Since the introduction of Sinkhorn's algorithm (Cuturi, 2013) and more recent neural approaches (Makkuva et al., 2020), computational OT has seen great success across many domains such as biology (Schiebinger et al., 2019; Bunne et al., 2023), and extensively in machine learning (Genevay et al., 2018; Cuturi et al., 2019; Corenflos et al., 2021). Recently, ideas from the powerful flow-based approaches that have revolutionised generative modelling (Song et al., 2021; Peluchetti, 2022; Lipman et al., 2023; Liu et al., 2023; Albergo and Vanden-Eijnden, 2023) have been leveraged to solve the entropy-regularised dynamic OT problem, known as the Schr¨ odinger Bridge (SB). Such approaches provide significant scalability advantages, enabling approximation of OT maps between high-dimensional continuous datasets such as image data. Early flow-based SB solvers were based on the classical Iterative Proportional Fitting (IPF) scheme (De Bortoli et al., 2021; Vargas et al., 2021; Chen et al., 2022), but such methods have since been superseded by those based on the Iterative Markovian Fitting (IMF) scheme (Shi et al., 2023; Peluchetti, 2023) due to its many superior properties.

Beyond the standard OT problem between two marginals, multi-marginal OT aims to find a joint coupling over multiple marginals while minimising the total cost. Tree-structured costs are often

Corresponding author: howard@stats.ox.ac.uk

George Deligiannidis Department of Statistics University of Oxford

<!-- image -->

(a) Reciprocal process Π : The Y S ( × ) are sampled from the current coupling Π S over S . Conditional on the Y S , points ( × ) at unknown marginals V\S are sampled as Y S c |S ∼ Q S c |S . Brownian bridges are drawn along the edges between the samples.

<!-- image -->

(b) Markovianised process P : Vector fields are trained by bridge-matching along each edge. Samples ( × ) from the next coupling Π S are obtained by simulating the resulting SDEs along the tree structure, started at one of the known marginals.

Figure 1: The two stages of TreeIMF. On this tree, the marginals at the leaf vertices S (blue) are fixed. The marginal at the vertex in V\S (red) is not fixed, and can change during the procedure.

considered (Haasler et al., 2021), as they frequently arise in applications while also allowing for improved scalability by leveraging the tree structure. Of particular significance are star-shaped trees, as these correspond to the prominent Wasserstein barycentre problem (Agueh and Carlier, 2011). The Wasserstein barycentre provides a natural notion of 'average' for probability distributions, and is widely studied due to its importance in applications, including in Bayesian learning (Srivastava et al., 2018), clustering (Ye et al., 2017), and representation learning (Singh et al., 2020) to name a few. Computing barycentres is notoriously challenging (Altschuler and Boix-Adser` a, 2022). Many successful approaches approximate the solution with a finite set of points, but these in-sample methods struggle to scale well as dimensionality increases. Alternative methods aim to provide continuous approximations (Li et al., 2020) using neural parameterisations, and often require multi-level optimisation procedures (Fan et al., 2021; Korotin et al., 2022). Of particular relevance to our work is the diffusion-based approach of Noble et al. (2023) which extends the IPF-based Diffusion Schr¨ odinger Bridge (DSB) method of De Bortoli et al. (2021) to the tree-based SB setting, and is (to our knowledge) currently the only neural ODE/SDE approach for barycentre computation.

One of the most successful and elegant approaches for Wasserstein-2 barycentre computation is the iterative fixed-point approach, which involves iteratively updating a candidate barycentre ν by solving for the OT map from ν to each marginal µ i and updating ν according to the induced coupling. This approach was popularised by the seminal work of ´ Alvarez-Esteban et al. (2016) and has formed the basis of many algorithmic developments, including in machine learning (Korotin et al., 2022). It has strong performance, and has been observed to converge quickly in only a few iterations (Lindheim, 2023). However, the procedure requires solving a complete OT problem to each marginal at each iteration, which is expensive as solving even a single OT problem can be challenging.

Contributions In this work, we extend the IMF procedure to the tree-based SB problem (Haasler et al., 2021). Our TreeDSBM algorithm provides an IMF counterpart to the TreeDSB method from Noble et al. (2023), closing a clear gap in the existing literature (see Table 1) and translating the many benefits of IMF over IPF to the tree setting. For the specific case of Wasserstein barycentre computation, our algorithm can be viewed as extending the commonly used fixed-point approaches to the case of flow-based entropic OT solvers. In particular, we show that the iterations of the IMF scheme and the fixed-point barycentre solvers can be elegantly combined into a single iterative procedure, yielding a fixed-point-style algorithm that requires only inexpensive bridge-matching at each iteration. We demonstrate significantly improved empirical performance over TreeDSB, and show that flow-based barycentre solvers can offer competitive performance against existing algorithms for continuous Wasserstein-2 barycentre computation.

Table 1: Positioning of our TreeDSBM algorithm in the literature.

|         | SB                                                                      | TreeSB (Haasler et al., 2021)                |
|---------|-------------------------------------------------------------------------|----------------------------------------------|
| IPF IMF | DSB (De Bortoli et al., 2021) DSBM (Shi et al., 2023; Peluchetti, 2023) | TreeDSB (Noble et al., 2023) TreeDSBM (Ours) |

## 2 Background

## 2.1 Schr¨ odinger bridges, optimal transport, and Wasserstein barycentres

We begin by reviewing the standard Schr¨ odinger Bridge (SB) problem between two distributions, and its relation to Optimal Transport (OT). For notation used in the paper, see Appendix A. Given an initial and final measure µ 0 and µ T of a population, the SB problem aims to identify the most likely intermediate dynamics of the population under the assumption that the movement is driven by a stochastic reference process Q . The resulting dynamic SB problem is defined as

<!-- formula-not-decoded -->

The static SB problem instead considers only the coupling over the endpoints,

<!-- formula-not-decoded -->

Under mild assumptions, the two problems are equivalent; the dynamic SB solution can be expressed as a mixture of bridges over the static SB coupling, P SB = Π SB 0 ,T Q ·| 0 ,T (L´ eonard, 2013).

Connection to quadratic OT The reference path measure Q is usually considered to be that of a Brownian motion ( σB t ) t ∈ [0 ,T ] . In this case, the static SB problem can be rewritten as

<!-- formula-not-decoded -->

which is the entropy-regularised OT problem for the quadratic ground-cost c ( x 0 , x T ) = 1 2 ∥ x 0 -x T ∥ 2 and ε = σ 2 T (see the Appendix for an overview of OT). In the sequel, we assume that the reference process is a Brownian motion ( σB t ) t ∈ [0 ,T ] .

Wasserstein barycentres A key motivation for the tree-structured setting that we will consider is the important Wasserstein barycentre problem (Agueh and Carlier, 2011), which provides a natural notion of 'average' for probability distributions. Given ℓ measures ( µ 1 , ..., µ ℓ ) and weights ( λ 1 , ..., λ ℓ ) summing to 1, the Wasserstein-2 barycentre is defined as

<!-- formula-not-decoded -->

where W 2 2 ( µ i , ν ) denotes the optimal transport cost between µ i and ν for the quadratic ground-cost.

The Wasserstein barycentre problem is notoriously difficult to solve, as it involves several OT subproblems for which one of the marginals can change in the optimisation. An elegant approach leverages the fixed-point property x = ∑ i λ i T i ( x ) which holds at the solution (under mild assumptions), where each T i is the OT map from ν to µ i (Agueh and Carlier, 2011; ´ Alvarez-Esteban et al., 2016). Intuitively, this states that 'each point in the support of the barycentre is the average of the corresponding points in the marginals'. This property has motivated an iterative fixed-point approach for barycentre computation, which involves iteratively updating a candidate barycentre ν by solving for the OT maps T i and constructing the next iterate as ¯ T # ν using the pushforward map ¯ T ( x ) = ∑ i λ i T i ( x ) . Such methods have been shown to be highly successful in practice ( ´ AlvarezEsteban et al., 2016; Korotin et al., 2022; Lindheim, 2023; Tanguy et al., 2024).

## 2.2 Iterative Markovian Fitting

We now outline recent flow-based generative modelling approaches for solving for the SB problem, which will form the basis for our approach. The SB solution P SB can be characterised as the unique path measure that is both Markov, and a mixture of bridges P SB = P SB 0 ,T Q ·| 0 ,T , that has correct marginals P SB 0 = µ 0 , P SB T = µ T (L´ eonard, 2013). This property motivates the Iterative Markovian Fitting (IMF) procedure (Shi et al., 2023; Peluchetti, 2023), which solves for the SB solution by alternately projecting between Markovian processes and processes with the correct bridges. We refer to Shi et al. (2023) for full details of the IMF procedure, but recall here the basic presentation. We recall the following definitions.

Definition 2.1 ( Reciprocal class, Reciprocal projection ) . A path measure Π ∈ P ( C ) is in the reciprocal class R ( Q ) of Q if it is a mixture of bridges of Q conditional on their values at the endpoints, Π = Π 0 ,T Q ·| 0 ,T . For a path measure P ∈ P ( C ) , the reciprocal projection is defined to be the mixture of bridges according to its induced coupling, proj R ( Q ) ( P ) = P 0 ,T Q ·| 0 ,T .

Definition 2.2 ( Markovian class, Markovian projection ) . Let M denote the set of Markovian path measures associated to a diffusion of the form d X t = v ( t, X t )d t + σ t d B t , with v, σ locally Lipschitz. For reference process ( σB t ) t ∈ [0 ,T ] , the Markovian projection of a measure P ∈ R ( Q ) is defined to be (when well-defined) the path measure associated to the SDE

<!-- formula-not-decoded -->

It can be shown that, under mild conditions, these definitions coincide with the following minimisation problems over path measures (Shi et al., 2023),

<!-- formula-not-decoded -->

The IMF iterations are defined below, and converge to a unique fixed point which is the SB solution.

<!-- formula-not-decoded -->

Training via bridge-matching The IMF procedure requires learning the vector field corresponding the Markovian projections proj M (Π) . This is done by training a neural network v θ with a bridge-matching loss objective (Peluchetti, 2022), for which the drift in (5) is the optimum,

<!-- formula-not-decoded -->

Comparison to Iterative Proportional Fitting Traditionally, the standard way to solve the SB problem is via the Iterative Proportional Fitting (IPF) procedure (Fortet, 1940; Kullback, 1968; R¨ uschendorf, 1995), which minimises the KL divergence between subsequent iterations while alternating the endpoint measure that is fixed.

<!-- formula-not-decoded -->

This method was implemented using diffusion model-based approaches in De Bortoli et al. (2021), giving the Diffusion Schr¨ odinger Bridge (DSB) method (see also Vargas et al. (2021), Chen et al. (2022)). While allowing for arbitrary reference measures Q , this approach suffers from several shortfalls, such as only preserving both marginals at convergence (in comparison to IMF, which preserves both marginals at each iteration), as well as expensive trajectory caching, and 'forgetting' of the original reference measure (Fernandes et al., 2022). As a result, IPF approaches have largely been superseded by IMF approaches which avoid these issues.

## 2.3 Tree-structured Schr¨ odinger bridge

In this work, we consider the tree-structured Schrodinger Bridge problem (Haasler et al., 2021). In particular, we extend the IMF procedure to the dynamic Tree SB setting of Noble et al. (2023).

Stochastic processes on the tree In order to explain the dynamic Tree SB problem, we first need to define stochastic processes on a tree. We will consider trees T = ( V , E , ℓ ) with vertex set V and edge set E , where ℓ : E → R &gt; 0 is an edge-length function. One can extend the tree to a uniquely arcwise-connected metric space in the natural way; the arc connecting the two endpoints of an edge e is identified with a line segment of corresponding length T e = ℓ ( e ) , and they are connected according to the graph structure. Via a slight abuse of notation we will also denote this metric space as T . As such, we can define the space C ( T , R d ) of continuous paths from T into R d , and we let P ( C T ) := P ( C ( T , R d )) denote the space of probability measures over such paths. We will present the methodology according to a directed tree T r rooted at a vertex r , for which we can choose an ordered edge set E r corresponding to a depth-first traversal (though we will often omit the dependence on r in the notation). While this presentation may appear somewhat complex, the construction is quite natural; for an illustration of processes on a tree-structure, see Figures 1 and 4.

We will consider running SDEs along each edge according to the ordering E r in the directed tree structure. We sample X r ∼ P r and then sequentially simulate SDEs along each edge as we traverse the directed tree. For each edge e = ( u, v ) ∈ E r , we run an SDE d X e t = v e ( t, X e t )d t + σ e t d B e t for time t ∈ [0 , T e ] initialised at X e 0 = X u , and after simulation we let X v = X e T e . Such stochastic processes induce a path measure P ∈ P ( C T ) over the whole tree. Note that when the tree is a branch with 2 vertices and 1 edge, this recovers the standard case described earlier.

Tree-structured Schr¨ odinger bridges Now that we have constructed stochastic processes on the tree, the tree-structured Schr¨ odinger Bridge problem is defined analogously to the standard case. However, now the marginals may be fixed only at a subset of vertices S ⊂ V (see Figure 1). For a reference measure Q ∈ P ( C T ) , the dynamic and static TreeSB problems are defined respectively as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As in the standard case, using the chain rule for KL divergence gives (under mild conditions) equivalence of the two problems; the dynamic SB solution can be expressed as a mixture of bridges over the static SB coupling, P SB = Π SB V Q ·|V . In this work, we aim to solve the TreeSB dyn problem.

Connection to multi-marginal optimal transport In the remainder of the work, we will consider the reference measure Q to be associated with running Brownian motions ( σB e t ) t ∈ [0 ,T e ] along each edge e . The induced reference coupling Q V over the vertices is characterised by having independent Gaussian edge increments Y u -Y v ∼ N (0 , σ 2 T ( u,v ) ) . By evaluating the KL term, the static TreeSB problem can therefore be rewritten as

<!-- formula-not-decoded -->

This is precisely an entropy-regularised multi-marginal OT problem with a tree-structured quadratic cost function c ( x V ) = ∑ ( u,v ) ∈E 1 T ( u,v ) ∥ x u -x v ∥ 2 2 and entropy-regularisation ε = 2 σ 2 (see Appendix A for an overview). Note that each weight is the reciprocal of the corresponding edge length.

Connection to Wasserstein barycentres There are many ways to define entropy-regularised Wasserstein barycentres (see Appendix A for an overview). The work of Chizat (2023) unifies several of these, combining ε -regularised OT costs with an entropic penalty term on the barycentre weighted by τ . Noble et al. (2023) show that the following doubly-regularised barycentre problem

<!-- formula-not-decoded -->

is recovered from the TreeSB setting by considering a star-shaped tree with fixed marginals on the leaves and cost function c ( x 0: ℓ ) = ∑ ℓ i =1 λ i ∥ x 0 -x i ∥ 2 2 .

## 3 Iterative Markovian Fitting for tree-structured costs

In this work, we generalise the IMF procedure to the tree-based setting. Recall that standard IMF proceeds by iteratively Markovianising a mixture of bridges, where each bridge is a stochastic process conditioned on endpoints x 0 and x 1 , and the mixing coupling Π 0 , 1 is over the endpoint values x 0 , x 1 . Similarly, we consider Markovianising mixtures of bridges, but instead the mixing coupling is a joint distribution Π S over values at the observed vertices S , and the bridging processes are conditioned on these values. In the following, we justify that the IMF procedure extends to this setting. We then explain how the bridging and Markovianisation can be performed in practice.

## 3.1 Markovian and reciprocal processes on the tree

Recall that the IMF procedure is motivated by the characterisation of the SB as the unique Markov measure that is a mixture of its bridges. Our first contribution is to extend this characterisation to the tree-based setting. We begin with the following definitions, akin to those in the standard case.

Definition 3.1 ( Markov class ) . Let M T denote the set of Markov path measures on the tree T , defined via a diffusion along each edge of the form d X e t = v e ( t, X t )d t + σ e t d B e t , with v e , σ e locally Lipschitz.

Definition 3.2 ( Reciprocal class ) . Π ∈ P ( C T ) is in the reciprocal class R S ( Q ) of Q for observed vertices S if Π = Π S Q ·|S . That is, Π is a mixture of bridges of Q conditioned on the values at S .

Observe that compared to the corresponding Definition 2.1 in standard IMF, the bridges are now conditioned on the values at vertices in S , rather than on only the endpoints of a single edge. We now state the following result characterising the TreeSB solution.

Theorem 3.1 ( TreeSB characterisation ) . Under mild assumptions (in the Brownian case, namely that ∫ ∥ x ∥ 2 d µ i ( x ) &lt; ∞ and H( µ i ) &lt; ∞ for each i ∈ S ), there exists a unique solution to the dynamic TreeSB problem ( TreeSB dyn ) . The solution is the unique process P that is both Markov and in R S ( Q ) with correct marginals P i = µ i for i ∈ S .

## 3.2 Iterative Markovian Fitting on the tree

Theorem 3.1 suggests that an analogous Iterative Markovian Fitting procedure could be used to solve the dynamic TreeSB problem; in this section, we show that this is indeed the case. The constructions and proofs proceed in much the same way as in Shi et al. (2023), but crucially rely on the following simple decomposition of the KL divergence according to the tree structure, which applies for measures in M T and R S ( Q ) .

Lemma 3.2 ( KLdecomposition along tree ) . Take path measures P , ˜ P that share a marginal at the root, P r = ˜ P r , and factorise along the tree. Under mild assumptions, we have the KL decomposition

<!-- formula-not-decoded -->

To extend the IMF procedure to the TreeSB problem, we require to extend the definitions and propositions from the standard case to the tree-based setting. The definition of the reciprocal class projection remains much the same as in the standard case, the difference being that now the bridging processes are conditioned on their values at vertices in S .

Definition 3.3 ( Reciprocal projection ) . For a process P ∈ P ( C T ) , the reciprocal projection is defined to be the induced mixture of bridges Π ∗ = proj R S ( Q ) ( P ) = P S Q ·|S .

Proposition 3.3. For P ∈ P ( C T ) , the reciprocal projection Π ∗ = proj R S ( Q ) ( P ) solves the minimisation problem Π ∗ = arg min Π ∈R S ( Q ) KL( P ∥ Π) .

The Markovian projection requires some alterations to account for the tree structure.

Definition 3.4 ( Markovian projection ) . For Π ∈ R S ( Q ) , the Markovian projection onto M , denoted M ∗ = proj M (Π) , is defined as

<!-- formula-not-decoded -->

Π e denotes the restriction of Π to edge e , proj M e (Π e ) denotes its Markovian projection (Definition 2.2), and ∏ T denotes the composition of the resulting measures according to the tree structure.

Proposition 3.4. Under mild assumptions, the Markovian projection M ∗ = proj M (Π) solves the minimisation problem M ∗ = arg min M ∈M T KL(Π ∥ M ) .

The TreeIMF procedure is then defined via the usual iterations, started from an initialisation P 0 ∈ R S ( Q ) such that P 0 i = µ i for i ∈ S :

<!-- formula-not-decoded -->

Note that this procedure recovers standard IMF when the tree has a bridge structure (2 vertices, 1 edge), just as TreeDSB recovers DSB in this setting.

We now state the following result, which shows convergence of the IMF iterates to the TreeSB solution. This resembles Theorem 8 in Shi et al. (2023), and the proof proceeds in largely the same way using the appropriate modifications and in particular leveraging the KL decomposition in Lemma 3.2. We defer the details to Appendix B.

Theorem 3.5 ( Convergence of TreeIMF ) . Under mild conditions, the TreeSB dyn solution P ∗ is the unique fixed point of the TreeIMF iterates P n , and we have lim n →∞ KL( P n ∥ P ) = 0 .

## 3.3 Implementation

We have established the convergence of the TreeIMF procedure to the TreeSB solution. We now explain how the procedure can be implemented in practice.

Constructing the bridge processes Recall that the reciprocal projection relies on constructing bridges of the reference process Y associated to Q , conditioned on the values at vertices in S . Due to the tree structure and the Brownian dynamics of Q , such bridges can be constructed by first sampling the values of Y S c |S ∼ Q S c |S at the unseen vertices in S c = V\S , and then sampling Brownian bridges along each edge ( u, v ) ∈ E between Y u and Y v .

Note in particular that the conditional coupling Q S c |S is tractable because the static coupling Q V is a multivariate Gaussian. Specifically, Q V is characterised via independent Gaussian edge increments Y u -Y v ∼ N (0 , σ 2 T e ) , so Q V ( Y ) ∝ exp( -∑ ( u,v ) ∈E 1 2 σ 2 T e ∥ Y u -Y v ∥ 2 ) ∝ exp( -1 2 Y ⊤ LY ) where L is the precision matrix given by

<!-- formula-not-decoded -->

Using the formula for conditional multivariate Gaussians in terms of the precision matrix, we have Y S c | Y S = y S ∼ N (˜ µ, ˜ Σ) , using the block matrix expressions ˜ µ = -( L S c S c ) -1 L S c S y S and ˜ Σ = ( L S c S c ) -1 . Based on the structure of the tree, we can therefore obtain a sample of the bridging process by: (1) calculating this conditional joint distribution Q S c |S over the unseen vertices, (2) sampling Y S c |S according this distribution, and then (3) drawing Brownian bridges between Y u and Y v along each edge in the tree.

Performing the Markovian projection The above construction allows us to construct samples from a reciprocal process Π . Recall from Definition 3.4 that the Markovian projection then proceeds by performing individual Markovian projections along each edge. We therefore maintain a neurallyparameterised drift function v θ e for each edge e = ( u, v ) , each of which is trained according to the bridge-matching loss

<!-- formula-not-decoded -->

Bidirectional training After training the vector fields, let M denote the path measure of the resulting Markov process on T . Constructing the next reciprocal process requires samples from M , which can be obtained by running the learned diffusions from the root node r ∈ S along the tree. Note that while in theory we have M i = µ i for each i ∈ S , errors will accumulate in practice. We therefore follow Shi et al. (2023) and learn both forward and backward diffusion processes along each edge, which give equivalent representations of the path measures via the corresponding timereversals (see Appendix C for details). This enables simulation along the tree from any vertex in S to obtain samples from the coupling, helping to mitigate errors accumulating in the marginals. Note also that all edges train independently, so can be learned in parallel for faster computation.

In keeping with previous naming conventions, we call the proposed algorithm Tree Diffusion Schr¨ odinger Bridge Matching (TreeDSBM). The method is summarised in Algorithm 1 with full implementation details in Appendix C, and we provide an illustration of the two-step procedure in Figure 1. The algorithm can be initialised at any coupling Π 0 S over S with correct marginals; a standard choice would be the independent coupling Π 0 S = ⊗ i ∈S µ i . We provide a simplified and more explicit version of the algorithm when used for barycentre computation in Appendix C.

## Algorithm 1: TreeDSBM

```
Input: Initial coupling Π 0 S (e.g. independent), number of iterations N . Let Π 0 = Π 0 S Q ·|S ; for n ∈ { 0 , . . . , N -1 } do Learn 2 |E| vector fields using (15) with Π = Π n , to obtain Markovian process M n +1 ; Simulate M n +1 from a chosen root r ∈ S to obtain samples from M n +1 S ; Let Π n +1 = M n +1 S Q ·|S using obtained samples from M n +1 S ; end
```

Table 2: Sinkhorn divergence to the 'ground-truth' barycentre, for barycentre samples generated from each leaf vertex k . The Sinkhorn divergence is computed with entropy regularisation 0.01, using 5000 generated samples and 1500 points approximating the groundtruth (mean ± std, over 5 runs).

<!-- image -->

Figure 2: Comparison of the learned barycentre for TreeDSBM (6 IMF iterations) against TreeDSB (50 IPF iterations) and WIN. TreeDSB and TreeDSBM samples are generated from each leaf vertex k , and for WIN we plot samples using the weighted-pushforward expression for the barycentre. Also displayed is a close approximation to the ground-truth, using the in-sample method of Cuturi and Doucet (2014).

| Method                            | k = 0            | k = 1            | k = 2            |
|-----------------------------------|------------------|------------------|------------------|
| TreeDSBM (6 IMF) TreeDSB (50 IPF) | 1.14 ± 0.07 2.35 | 1.05 ± 0.07 4.04 | 1.08 ± 0.11 2.35 |
| WIN ( ∑ i λ i T i )# ν            |                  | 1.17             |                  |

## 3.4 Connection to fixed-point Wasserstein barycentre algorithms

Recall that the unregularised Wasserstein-2 barycentre can be computed using the iterative fixedpoint method, where each iteration proceeds by pushing-forward the current iterate through the map ∑ i λ i T i . In our setting, the barycentre problem corresponds to a star-shaped tree with fixed marginals on the leaves. In this case, the bridges of Q are conditioned on each of the leaf vertices and the only unknown point is Y 0 at the centre vertex, so the conditional distribution Q S c |S simplifies to

<!-- formula-not-decoded -->

Our method can therefore be viewed as a natural counterpart of fixed-point methods for barycentre computation, adapted to the case of flow-based entropic OT solvers. While a naive approach might consider using IMF to solve each OT sub-problem in the fixed-point scheme, resulting in nested iterations, our approach shows that the IMF and fixed-point iterations can in fact be elegantly combined into a single iterative procedure, along with a well-understood theoretical grounding. In particular, the expensive OT map computations required for the fixed-point procedure can instead be switched out for inexpensive bridge-matching procedures. Each iteration is therefore cheap, and empirically we found the TreeDSBM algorithm to retain the fast convergence property of the fixed-point procedure in terms of the number of iterations required.

## 4 Experiments

Synthetic 2 d barycentre We first examine the performance of TreeDSBM in a low-dimensional synthetic example, using the experimental setup of Noble et al. (2023). We compute the ( 1 3 , 1 3 , 1 3 ) -barycentre of a moon, spiral, and circle dataset with ε = 0 . 1 (recall σ = √ ε 2 ). We compare TreeDSBM ran for 6 IMF iterations against TreeDSB ran for 50 IPF iterations (using checkpoints provided by Noble et al. (2023)). In Figure 2, we plot the obtained samples from both methods, and for comparison also display the barycentre obtained using the in-sample method from Cuturi and Doucet (2014) to give a close approximation to the ground-truth. To quantitatively assess performance, we report the Sinkhorn divergence (Genevay et al., 2018) relative to this 'ground-truth' in Table 2. The results show that TreeDSBM significantly improves over TreeDSB in this setting, approximating the barycentre to a high degree of accuracy and at a much lower computational cost (with good convergence after only a few IMF iterations). It is able to successfully capture the complex nature of the barycentre in this challenging example (previously suggested as a potential limitation of dynamic solvers in Noble et al. (2023)), and the barycentres generated from different vertices k exhibit improved consistency. For details of the experimental setup, see Appendix D.1.

We additionally report results for the iterative WIN method (Korotin et al., 2022). We see that TreeDSBM performs competitively with this strong baseline for continuous Wasserstein-2 solvers. The results reported for WIN are for the combined map ( ∑ i λ i T i )# ν , as in our experiments the barycentre generator ν = G # ρ was unable to fit the true barycentre accurately, nor were the maps T -1 i # µ i . Additionally, we applied the W2CB (Korotin et al., 2021) and NOTWB (Kolesov et al.,

2024a) algorithms to this example, but were unable find hyperparameters to make the algorithms to converge to the correct solution. We hypothesise that the neural maps may struggle to model the discontinuous transports well, and also that this challenging example may result in difficult loss landscapes. We found TreeDSBM to exhibit stable training despite this challenging problem setting, and to also be the fastest of the algorithms to converge to the solution. We provide a runtime analysis, along with further results and discussion, in Appendix D.1.

MNIST 2,4,6 barycentre We also compare performance of TreeDSBM with TreeDSB on a higher dimensional image dataset, computing the ( 1 3 , 1 3 , 1 3 ) -barycentre between MNIST digits 2, 4, and 6 (LeCun et al., 2010). In this setting, TreeDSB is reported in Noble et al. (2023) to exhibit training instability for low entropy regularisation ε , causing it to struggle to match the marginal measures at the vertices. In contrast, the IMF approach of TreeDSBM ensures matching of these marginals, thus allowing the use of much smaller regularisation values. In fact, using a large regularisation for TreeDSBM would limit sample quality, because of the noise added when sampling from Q S c |S in the reciprocal process construction. We there-

Figure 3: Samples from the 2,4,6 MNIST barycentre.

<!-- image -->

fore use ε = 0 . 02 for TreeDSBM with 4 IMF iterations, and display a visual comparison with results reported in Noble et al. (2023) in Section 4. While it is difficult to validate the accuracy of such solutions, the obtained samples from TreeDSBM appear to display a greater resemblance to state-of-the-art barycentre methods for similar problems (see e.g. Kolesov et al. (2024a)), while again requiring significantly less computational cost than TreeDSB. We remark that the TreeDSBM samples in Figure 3b contain a small amount of noise, which is to be expected as we are solving for an entropy-regularised problem. In Appendix D.2 we also display samples with the noise reduced or removed, along with full experimental details and additional results.

Subset posterior aggregation The previous experiments have shown TreeDSBM to improve over its IPF counterpart TreeDSB. We now provide a more detailed comparison with existing stronglyperforming methods for continuous Wasserstein-2 barycentre estimation, namely WIN (Korotin et al., 2022), W2CB (Korotin et al., 2021), and NOTWB (Kolesov et al., 2024a). As a real-world applications experiment, we compare the algorithms in the subset posterior aggregation setting, a standard experiment in the barycentre literature (Staib et al., 2017; Li et al., 2020; Fan et al., 2021; Korotin et al., 2021). The barycentre of subset posteriors is known to be close to the true posterior (Srivastava et al., 2018), and as such it can be efficient to compute posteriors on only subsets of datasets before aggregating them using a barycentre algorithm. We consider the experimental setup and dataset used in Korotin et al. (2021) (the same dataset was also used previously in Li et al. (2020) and Fan et al. (2021)), which uses Poisson and negative-binomial regressions on a bikerental dataset (Fanaee-T, 2013), and report results for the BW 2 2 -UVP metric. We ran TreeDSBM for 4 IMF iterations with ε = 0 . 001 ; for full experimental details, see Appendix D.3.

From Table 3, we see that all methods perform strongly. Given that we do not have perfect access to the ground truth barycentre, it is difficult to conclusively say which performs best, but we see that TreeDSBM certainly performs competitively with these state-of-the-art approaches. Moreover, we found TreeDSBM to display fast training-both TreeDSBM and NOTWB obtained good results after only around 3 minutes of training, while W2CB took approximately 10 minutes and WIN took around 45 minutes. For more details regarding runtimes, see Appendix D.3.

Higher-dimensional Gaussian experiments We also report results for computing the barycentre of Gaussian distributions, in increasingly high dimensions. This is a standard experiment in the literature, because in this setting the ground-truth barycentre is also Gaussian and the parameters can be calculated accurately using a fixed point method ( ´ Alvarez-Esteban et al., 2016).

We follow the experimental setup previously used in (Korotin et al., 2021, 2022; Kolesov et al., 2024a), in which 3 Gaussian distributions and its ground-truth ( 1 3 , 1 3 , 1 3 )-barycentre are randomly generated, for each dimension in { 64 , 96 , 128 } . Wereport results for the BW 2 2 -UVP and L 2 -UVP metrics. We see that all the methods again perform well in this setting. For the BW 2 2 -UVP metric, W2CB and NOTWB appear to have a slight edge, but TreeDSBM is only slightly higher and is comparable with results for WIN. For the L 2 -UVP metric, results for TreeDSBM are higher than for W2CB and NOTWB, though the results are still low and are again comparable with WIN.

Table 3: BW 2 2 -UVP , % for different algorithms on the subset posterior aggregation experiment in Korotin et al. (2021), evaluated using 100,000 samples.

|                     |   WIN |   W2CB |   NOTWB |   TreeDSBM (Ours) |
|---------------------|-------|--------|---------|-------------------|
| ↓ Poisson           | 0.014 |  0.026 |   0.023 |             0.008 |
| ↓ Negative Binomial | 0.009 |  0.024 |   0.018 |             0.012 |

Table 4: L 2 -UVP , % and BW 2 2 -UVP , % for different algorithms on the high-dimensional Gaussian experiment, evaluated using 100,000 samples.

|                    |         |   WIN |   W2CB |   NOTWB |   TreeDSBM (Ours) |
|--------------------|---------|-------|--------|---------|-------------------|
| ↓ BW 2 2 - UVP , % | d = 64  |  0.2  |   0.04 |    0.08 |              0.14 |
| ↓ BW 2 2 - UVP , % | d = 96  |  0.3  |   0.07 |    0.1  |              0.15 |
| ↓ BW 2 2 - UVP , % | d = 128 |  0.38 |   0.12 |    0.14 |              0.27 |
|                    | d = 64  |  0.96 |   0.17 |    0.1  |              1.18 |
|                    | d = 96  |  1.2  |   0.2  |    0.1  |              1.13 |
|                    | d = 128 |  1.46 |   0.25 |    0.13 |              1.23 |

## 5 Discussion

Related work Beyond the IPF and IMF approaches, other approaches for the SB problem include adversarial solvers (Kim et al., 2024; Gushchin et al., 2023), parameterisation via potentials for Gaussian mixtures (Korotin et al., 2024; Gushchin et al., 2024), and variational approaches (Deng et al., 2024). For Wasserstein barycentre computation, standard approaches ensure tractability by representing the solution with a finite set of points (either updating only weightings (Benamou et al., 2015; Solomon et al., 2015; Cuturi and Peyr´ e, 2016; Staib et al., 2017; Dvurechenskii et al., 2018), or also the positions of the points in the support (Rabin et al., 2012; Cuturi and Doucet, 2014; Claici et al., 2018; Luise et al., 2019)). Such approaches are effective in lower dimensions but scale poorly, cannot be used to generate new samples, and do not capture the true continuous nature of the barycentre. To address these limitations, recent work has focused on learning continuous approximations. Li et al. (2020) optimise for the regularised dual potentials for general costs, while others parameterise Wasserstein-2 potentials with convex neural architectures using adversarial losses (Fan et al., 2021) or additional cycle-consistency regularisation (Korotin et al., 2021). Korotin et al. (2022) leverage the fixed-point property in ´ Alvarez-Esteban et al. (2016). We also highlight a recent line of works of Kolesov et al. (2024b,a) which consider bi-level adversarial approaches for general cost functions. Our approach utilises non-adversarial bridge-matching loss objectives, which provide stable training but requires multiple sequential IMF iterations. Finally, we emphasise that our approach tackles the tree-structured SB problem (Haasler et al., 2021; Noble et al., 2023), and thus is applicable beyond only barycentre problems (we consider a toy example in Appendix E.3).

Limitations Our method shares the same limitations as other flow-based SB solvers. It is restricted to quadratic cost functions for OT, and introduces an entropic bias. Inference is expensive in comparison to methods that use a single function evaluation, and our method is not sampling-free, requiring simulations of the current learned processes at each iteration. Future advances in the flow and SB literatures can aid in addressing these limitations. We also provide an additional experiment in Appendix E that highlights a possible limitation of our method in scenarios where there is a simple shared structure between the known marginals and the barycentre.

Conclusion and future work We have extended the IMF procedure to the tree-structure SB problem, providing a scalable flow-based approach that in particular can be used for entropic Wasserstein barycentre computation. Our TreeDSBM algorithm displays improved performance over its IPF counterpart TreeDSB, and demonstrates that flow-based approaches for barycentre estimation can offer a compelling alternative to established continuous Wasserstein-2 barycentre algorithms. Future directions can investigate improved architectures and implementation techniques inspired by progress in the flow-matching literature, as well as extensions to other data modalities.

## Acknowledgements

SH is supported by the EPSRC CDT in Modern Statistics and Statistical Machine Learning [grant number EP/S023151/1]. PP is supported by the EPSRC CDT in Modern Statistics and Statistical Machine Learning [EP/S023151/1], a Google PhD Fellowship, and an NSERC Postgraduate Scholarship (PGS D). GD was supported by the Engineering and Physical Sciences Research Council [grant number EP/Y018273/1]. The authors would like to thank James Thornton for helpful discussions.

## References

- Martial Agueh and Guillaume Carlier (2011). 'Barycenters in the Wasserstein Space'. In: SIAM Journal on Mathematical Analysis 43.2, pp. 904-924.
- Michael Samuel Albergo and Eric Vanden-Eijnden (2023). 'Building Normalizing Flows with Stochastic Interpolants'. In: International Conference on Learning Representations .
- Jason M. Altschuler and Enric Boix-Adser` a (2022). 'Wasserstein Barycenters Are NP-Hard to Compute'. In: SIAM Journal on Mathematics of Data Science 4.1, pp. 179-203.
- Pedro C. ´ Alvarez-Esteban, E. del Barrio, J.A. Cuesta-Albertos, and C. Matr´ an (2016). 'A fixedpoint approach to barycenters in Wasserstein space'. In: Journal of Mathematical Analysis and Applications 441.2, pp. 744-762.
- Brandon Amos, Lei Xu, and J. Zico Kolter (2017). 'Input Convex Neural Networks'. In: International Conference on Machine Learning .
- Jean-David Benamou, Guillaume Carlier, Marco Cuturi, Luca Nenna, and Gabriel Peyr´ e (2015). 'Iterative Bregman Projections for Regularized Transportation Problems'. In: SIAM Journal on Scientific Computing 37.2, A1111-A1138.
- V. I. Bogachev, T. I. Krasovitskii, and S. V. Shaposhnikov (2021). 'On uniqueness of probability solutions of the Fokker-Planck-Kolmogorov equation'. In: Sbornik: Mathematics 212.6, p. 745.
- David Bolin, Alexandre B. Simas, and Jonas Wallin (2024). Markov properties of Gaussian random fields on compact metric graphs . arXiv: 2304.03190 [math.PR] .
- James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang (2018). JAX: composable transformations of Python+NumPy programs . Version 0.3.13.
- Charlotte Bunne, Stefan G. Stark, Gabriele Gut, Jacobo Sarabia del Castillo, Mitch Levesque, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar R¨ atsch (2023). 'Learning Single-Cell Perturbation Responses using Neural Optimal Transport'. In: vol. 20. Nature Methods, pp. 1759-1768.
- Tianrong Chen, Guan-Horng Liu, Molei Tao, and Evangelos Theodorou (2023). 'Deep Momentum Multi-Marginal Schr¨ odinger Bridge'. In: Advances in Neural Information Processing Systems .
- Tianrong Chen, Guan-Horng Liu, and Evangelos Theodorou (2022). 'Likelihood Training of Schr¨ odinger Bridge using Forward-Backward SDEs Theory'. In: International Conference on Learning Representations .
- Yongxin Chen, Giovanni Conforti, Tryphon T. Georgiou, and Luigia Ripani (2019). 'Multi-marginal Schr¨ odinger Bridges'. In: Geometric Science of Information . Springer International Publishing, pp. 725-732.
- L´ ena¨ ıc Chizat (2023). Doubly Regularized Entropic Wasserstein Barycenters . arXiv: 2303.11844 [math.OC] .
- Sebastian Claici, Edward Chien, and Justin Solomon (2018). 'Stochastic Wasserstein Barycenters'. In: International Conference on Machine Learning .
- Adrien Corenflos, James Thornton, George Deligiannidis, and Arnaud Doucet (2021). 'Differentiable particle filtering via entropy-regularized optimal transport'. In: International Conference on Machine Learning .

- Marco Cuturi (2013). 'Sinkhorn Distances: Lightspeed Computation of Optimal Transport'. In: Advances in Neural Information Processing Systems .
- Marco Cuturi and Arnaud Doucet (2014). 'Fast Computation of Wasserstein Barycenters'. In: International Conference on Machine Learning .
- Marco Cuturi, Laetitia Meng-Papaxanthos, Yingtao Tian, Charlotte Bunne, Geoff Davis, and Olivier Teboul (2022). 'Optimal Transport Tools (OTT): A JAX Toolbox for all things Wasserstein'. In: arXiv preprint arXiv:2201.12324 .
- Marco Cuturi and Gabriel Peyr´ e (2016). 'A Smoothed Dual Approach for Variational Wasserstein Problems'. In: SIAM Journal on Imaging Sciences 9.1, pp. 320-343.
- Marco Cuturi, Olivier Teboul, and Jean-Philippe Vert (2019). 'Differentiable ranking and sorting using optimal transport'. In: Advances in Neural Information Processing Systems .
- Paolo Dai Pra (1991). 'A stochastic control approach to reciprocal diffusion processes'. In: Applied Mathematics and Optimization 23.1, pp. 313-329.
- Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, and Arnaud Doucet (2024). 'Schrodinger Bridge Flow for Unpaired Data Translation'. In: Conference on Neural Information Processing Systems .
- Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet (2021). 'Diffusion Schr¨ odinger Bridge with Applications to Score-Based Generative Modeling'. In: Advances in Neural Information Processing Systems .
- Wei Deng, Weijian Luo, Yixin Tan, Marin Bilos, Yu Chen, Yuriy Nevmyvaka, and Ricky T. Q. Chen (2024). 'Variational Schr¨ odinger Diffusion Models'. In: International Conference on Machine Learning .
- Prafulla Dhariwal and Alexander Nichol (2021). 'Diffusion Models Beat GANs on Image Synthesis'. In: Advances in Neural Information Processing Systems .
- Pavel Dvurechenskii, Darina Dvinskikh, Alexander Gasnikov, Cesar Uribe, and Angelia Nedich (2018). 'Decentralize and Randomize: Faster Algorithm for Wasserstein Barycenters'. In: Advances in Neural Information Processing Systems .
- Jiaojiao Fan, Amirhossein Taghvaei, and Yongxin Chen (2021). 'Scalable Computations of Wasserstein Barycenter via Input Convex Neural Networks'. In: International Conference on Machine Learning .
- Hadi Fanaee-T (2013). Bike Sharing . UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5W894.
- Kilian Fatras, Younes Zine, Szymon Majewski, R´ emi Flamary, R´ emi Gribonval, and Nicolas Courty (2021). Minibatch optimal transport distances; analysis and applications . arXiv: 2101.01792 [stat.ML] .
- David Lopes Fernandes, Francisco Vargas, Carl Henrik Ek, and Neill D. F. Campbell (2022). 'Shooting Schr¨ odinger's Cat'. In: Fourth Symposium on Advances in Approximate Bayesian Inference .
- R´ emi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aur ˜ A©lie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, L ˜ A©oGautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, and Titouan Vayer (2021). 'POT: Python Optimal Transport'. In: Journal of Machine Learning Research 22.78, pp. 1-8.
- Robert Fortet (1940). 'R´ esolution d'un syst` eme d'´ equations de M. Schr¨ odinger'. In: Journal de Math´ ematiques Pures et Appliqu´ ees 9e s´ erie, 19.1-4, pp. 83-105.
- Aude Genevay, Marco Cuturi, Gabriel Peyr´ e, and Francis Bach (2016). 'Stochastic Optimization for Large-scale Optimal Transport'. In: Advances in Neural Information Processing Systems .
- Aude Genevay, Gabriel Peyr´ e, and Marco Cuturi (2018). 'Learning generative models with Sinkhorn divergences'. In: International Conference on Artificial Intelligence and Statistics .

- Nikita Gushchin, Sergei Kholkin, Evgeny Burnaev, and Alexander Korotin (2024). 'Light and Optimal Schr¨ odinger Bridge Matching'. In: International Conference on Machine Learning .
- Nikita Gushchin, Alexander Kolesov, Alexander Korotin, Dmitry P Vetrov, and Evgeny Burnaev (2023). 'Entropic Neural Optimal Transport via Diffusion Processes'. In: Advances in Neural Information Processing Systems .
- Isabel Haasler, Axel Ringh, Yongxin Chen, and Johan Karlsson (2021). 'Multimarginal Optimal Transport with a Tree-Structured Cost and the Schr¨ odinger Bridge Problem'. In: SIAM Journal on Control and Optimization 59.4, pp. 2428-2453.
- Ben Hambly and Terry Lyons (2008). Some notes on trees and paths . arXiv: 0809 . 1365 [math.CA] .
- L.V. Kantorovich (1942). On translation of mass . Proceedings of the USSR Academy of Sciences, 37.
- Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine (2022). 'Elucidating the Design Space of Diffusion-Based Generative Models'. In: Advances in Neural Information Processing Systems .
- Beomsu Kim, Yu-Guan Hsieh, Michal Klein, marco cuturi, Jong Chul Ye, Bahjat Kawar, and James Thornton (2025). 'Simple ReFlow: Improved Techniques for Fast Flow Models'. In: International Conference on Learning Representations .
- Beomsu Kim, Gihyun Kwon, Kwanyoung Kim, and Jong Chul Ye (2024). 'Unpaired Image-toImage Translation via Neural Schr¨ odinger Bridge'. In: International Conference on Learning Representations .
- Diederick P Kingma and Jimmy Ba (2015). 'Adam: A method for stochastic optimization'. In: International Conference on Learning Representations .
- Alexander Kolesov, Petr Mokrov, Igor Udovichenko, Milena Gazdieva, Gudmund Pammer, Evgeny Burnaev, and Alexander Korotin (2024a). 'Estimating Barycenters of Distributions with Neural Optimal Transport'. In: International Conference on Machine Learning .
- Alexander Kolesov, Petr Mokrov, Igor Udovichenko, Milena Gazdieva, Gudmund Pammer, Anastasis Kratsios, Evgeny Burnaev, and Alexander Korotin (2024b). 'Energy-Guided Continuous Entropic Barycenter Estimation for General Costs'. In: Advances in Neural Information Processing Systems .
- Alexander Korotin, Vage Egiazarian, Lingxiao Li, and Evgeny Burnaev (2022). 'Wasserstein Iterative Networks for Barycenter Estimation'. In: Conference on Neural Information Processing Systems .
- Alexander Korotin, Nikita Gushchin, and Evgeny Burnaev (2024). 'Light Schr¨ odinger Bridge'. In: International Conference on Learning Representations .
- Alexander Korotin, Lingxiao Li, Justin Solomon, and Evgeny Burnaev (2021). 'Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization'. In: International Conference on Learning Representations .
- Alexander Korotin, Daniil Selikhanovych, and Evgeny Burnaev (2023). 'Neural Optimal Transport'. In: International Conference on Learning Representations .
- S. Kullback (1968). 'Probability Densities with Given Marginals'. In: The Annals of Mathematical Statistics 39.4, pp. 1236-1243.
- Yann LeCun, Corinna Cortes, and CJ Burges (2010). 'MNIST handwritten digit database'. In: ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist 2.
- Christian L´ eonard (2012). 'Girsanov Theory Under a Finite Entropy Condition'. In: S´ eminaire de Probabilit´ es XLIV . Berlin, Heidelberg: Springer Berlin Heidelberg, pp. 429-465.
- Lingxiao Li, Aude Genevay, Mikhail Yurochkin, and Justin M Solomon (2020). 'Continuous Regularized Wasserstein Barycenters'. In: Advances in Neural Information Processing Systems .
- Johannes von Lindheim (2023). 'Simple approximative algorithms for free-support Wasserstein barycenters'. In: Computational Optimization and Applications 85.1, pp. 213-246.

- Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le (2023). 'Flow Matching for Generative Modeling'. In: International Conference on Learning Representations .
- Xingchao Liu, Chengyue Gong, and Qiang Liu (2023). 'Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow'. In: International Conference on Learning Representations .
- Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang (2015). 'Deep Learning Face Attributes in the Wild.' In: ICCV . IEEE Computer Society, pp. 3730-3738.
- Giulia Luise, Saverio Salzo, Massimiliano Pontil, and Carlo Ciliberto (2019). 'Sinkhorn Barycenters with Free Support via Frank-Wolfe Algorithm'. In: Advances in Neural Information Processing Systems .
- Christian L´ eonard (2013). A survey of the Schr¨ odinger problem and some of its connections with optimal transport . arXiv: 1308.0215 [math.PR] .
- Christian L´ eonard, Sylvie Roelly, and Jean Zambrini (Jan. 2014). 'Reciprocal processes. A measuretheoretical point of view'. In: Probability Surveys 11.
- Ashok Makkuva, Amirhossein Taghvaei, Sewoong Oh, and Jason Lee (2020). 'Optimal transport mapping via input convex neural networks'. In: International Conference on Machine Learning .
- Gaspard Monge (1781). 'M´ emoire sur la th´ eorie des d´ eblais et des remblais'. In: Histoire de l'Acad´ emie Royale des Sciences , pp. 666-704.
- Maxence Noble, Valentin De Bortoli, Arnaud Doucet, and Alain Durmus (2023). 'Tree-Based Diffusion Schr¨ odinger Bridge with Applications to Wasserstein Barycenters'. In: Conference on Neural Information Processing Systems .
- Marcel Nutz (2021). Introduction to Entropic Optimal Transport . Lecture Notes.
- F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay (2011). 'Scikit-learn: Machine Learning in Python'. In: Journal of Machine Learning Research 12, pp. 2825-2830.
- Stefano Peluchetti (2022). Non-Denoising Forward-Time Diffusions .
- Stefano Peluchetti (2023). 'Diffusion Bridge Mixture Transports, Schr¨ odinger Bridge Problems and Generative Modeling'. In: Journal of Machine Learning Research 24.374, pp. 1-51.
- Gabriel Peyr´ e and Marco Cuturi (2019). 'Computational Optimal Transport'. In: Foundations and Trends in Machine Learning 11 (5-6), pp. 355-602.
- Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q. Chen (2023). 'Multisample Flow Matching: Straightening Flows with Minibatch Couplings'. In: International Conference on Machine Learning .
- Julien Rabin, Gabriel Peyr´ e, Julie Delon, and Marc Bernot (2012). 'Wasserstein Barycenter and Its Application to Texture Mixing'. In: Scale Space and Variational Methods in Computer Vision . Berlin, Heidelberg: Springer Berlin Heidelberg, pp. 435-446.
- Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015). 'U-Net: Convolutional Networks for Biomedical Image Segmentation'. In: Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015 . Cham: Springer International Publishing, pp. 234-241.
- Ludger R¨ uschendorf (1995). 'Convergence of the Iterative Proportional Fitting Procedure'. In: The Annals of Statistics 23.4, pp. 1160-1174.
- Filippo Santambrogio (2015). Optimal transport for applied mathematicians . Birkhauser.
- Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, Lia Lee, Jenny Chen, Justin Brumbaugh, Philippe Rigollet, Konrad Hochedlinger, Rudolf Jaenisch, Aviv Regev, and Eric S. Lander (2019). 'Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming'. In: Cell 176.4.

- Yunyi Shen, Renato Berlinghieri, and Tamara Broderick (2025). 'Multi-marginal Schr¨ odinger Bridges with Iterative Reference Refinement'. In: International Conference on Artificial Intelligence and Statistics .
- Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet (2023). 'Diffusion Schr¨ odinger Bridge Matching'. In: Conference on Neural Information Processing Systems .
- Sidak Pal Singh, Andreas Hug, Aymeric Dieuleveut, and Martin Jaggi (2020). 'Context Mover's Distance &amp; Barycenters: Optimal Transport of Contexts for Building Representations'. In: International Conference on Artificial Intelligence and Statistics .
- Justin Solomon, Fernando de Goes, Gabriel Peyr´ e, Marco Cuturi, Adrian Butscher, Andy Nguyen, Tao Du, and Leonidas Guibas (July 2015). 'Convolutional wasserstein distances: efficient optimal transportation on geometric domains'. In: ACM Trans. Graph. 34.4.
- Justin Solomon, Raif Rustamov, Leonidas Guibas, and Adrian Butscher (2014). 'Wasserstein Propagation for Semi-Supervised Learning'. In: International Conference on Machine Learning .
- Max Sommerfeld, J¨ orn Schrieber, Yoav Zemel, and Axel Munk (2019). 'Optimal Transport: Fast Probabilistic Approximation with Exact Solvers'. In: Journal of Machine Learning Research 20.105, pp. 1-23.
- Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever (2023). 'Consistency Models'. In: International Conference on Machine Learning .
- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole (2021). 'Score-Based Generative Modeling through Stochastic Differential Equations'. In: International Conference on Learning Representations .
- Sanvesh Srivastava, Cheng Li, and David B. Dunson (2018). 'Scalable Bayes via Barycenter in Wasserstein Space'. In: Journal of Machine Learning Research 19.8, pp. 1-35.
- Matthew Staib, Sebastian Claici, Justin M Solomon, and Stefanie Jegelka (2017). In: Advances in Neural Information Processing Systems .
- Eloi Tanguy, Julie Delon, and Natha¨ el Gozlan (2024). Computing Barycentres of Measures for Generic Transport Costs . arXiv: 2501.04016 [math.NA] .
- Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid RectorBrooks, Guy Wolf, and Yoshua Bengio (2024a). 'Improving and generalizing flow-based generative models with minibatch optimal transport'. In: Transactions on Machine Learning Research .
- Alexander Y. Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, and Yoshua Bengio (2024b). 'Simulation-Free Schr¨ odinger Bridges via Score and Flow Matching'. In: International Conference on Artificial Intelligence and Statistics .
- Francisco Vargas, Pierre Thodoroff, Austen Lamacraft, and Neil Lawrence (2021). 'Solving Schr¨ odinger Bridges via Maximum Likelihood'. In: Entropy 23.9.
- Jianbo Ye, Panruo Wu, James Z. Wang, and Jia Li (2017). 'Fast Discrete Distribution Clustering Using Wasserstein Barycenter With Sparse Support'. In: IEEE Transactions on Signal Processing 65.9, pp. 2317-2332.

## Appendix

The Appendix is structured in the following way. In Appendix A, we define the notation used and provide further background information regarding the problems discussed in the paper. In Appendix B, we provide the proofs of the results stated in the main body. In Appendix C, we describe in more detail how to implement the TreeDSBM algorithm, and also discuss extensions of the methodology for improving convergence speed. In Appendix D, we give a full overview of the experimental setups in the main body, and include further discussions of the findings. In Appendix E we provide additional experiments that could not be included in the main body due to space constraints. Finally, Appendix F includes the licenses for assets used in this work.

## A Background

## A.1 Notation

We denote by T a tree with an edge length function, T = ( V , E , ℓ ) , or its associated metric space via associating edges e with the line segment of length T e = ℓ ( e ) (see Section 2.3). We let P ( C ) = P ( C ([0 , T ] , R d )) and P ( C T ) := P ( C ( T , R d )) denote the space of path measures on the interval [0 , T ] and T respectively. For the interval [0 , T ] , M denotes the set of Markov measures, and R ( Q ) denotes the reciprocal class for a reference measure Q ∈ M (see Definitions 2.1 and 2.2). For a path measure Q ∈ P ( C ) , Q t denotes the marginal distribution at time t , and Q ·| 0 ,T denotes the diffusion bridge conditional on values at 0 , T . We use Π = Π 0 ,T Q ·| 0 ,T ∈ P ( C ) to denote mixtures of bridges Π( · ) = ∫ R d × R d Q ·| 0 ,T ( ·| x 0 , x T )Π 0 ,T (d x 0 , d x T ) . Similarly on the tree T , M T denotes the set of Markov measures, and R S ( Q ) denotes the reciprocal class for a reference measure Q ∈ M T and observed vertices S ⊂ V (see Definitions 3.1 and 3.2). For a path measure Q ∈ P ( C T ) , Q v denotes the marginal at vertex v ∈ V , Q V denotes the induced joint distribution over a set of vertices V , and Q ·|S denotes the bridge process conditional on values at vertices S . Wedenote mixtures of bridges as Π = Π S Q ·|S ∈ P ( C T ) , defined similarly to above. Finally, KL( π ∥ ˜ π ) denotes the Kullback-Leibler divergence between measures π, ˜ π , and H( π ) denotes the differential entropy.

## A.2 Optimal transport

We here provide a brief overview of optimal transport; for more details we refer to Santambrogio (2015), Peyr´ e and Cuturi (2019).

Monge For two measures µ 0 and µ 1 , the original OT formulation given by Monge (1781) aims to find a map T ∗ : R d → R d pushing µ 0 onto µ 1 , that is µ 1 = T ∗ # µ 0 , while minimising the total cost of transportation according to a given cost function c : R d × R d → R ,

<!-- formula-not-decoded -->

Kantorovich While intuitive, the Monge formulation of OT is restrictive in that it does not permit splitting of mass in the transportation. Instead, it is common to consider the more general Kantorovich formulation (Kantorovich, 1942), which searches over joint distributions π ∈ P ( R d × R d ) with the correct marginals,

<!-- formula-not-decoded -->

Entropic regularisation The entropy-regularised OT problem adds an entropic penalty term to the Kantorovich objective, which smooths the resulting transport plan.

<!-- formula-not-decoded -->

The resulting problem has many desirable properties; it enables differentiability with respect to the inputs, relaxes constraints on the corresponding potentials (Genevay et al., 2016), and for discrete measures enables efficient computation using Sinkhorn's algorithm (Cuturi, 2013).

## A.3 Multi-marginal optimal transport

One can generalise the above optimal transport problem to the case of multiple marginal distributions. For a cost function c : ( R d ) ℓ +1 → R and an 'observed' set S ⊂ { 0 , ..., ℓ } , multi-marginal optimal transport (mmOT) searches over joint distributions π ∈ P (( R d ) ℓ +1 ) matching the prescribed marginals on S to minimise the objective,

<!-- formula-not-decoded -->

As in the standard case, one can define an entropy-regularised version of the mmOT problem,

<!-- formula-not-decoded -->

Multi-marginal Schr¨ odinger bridges We emphasise here the distinction between the multimarginal OT and corresponding SB problem that we consider in this work, compared to the multimarginal SB problem that has been considered recently in works such as Chen et al. (2019), Chen et al. (2023), and Shen et al. (2025) to name a few. These lines of works aim to find processes that evolve in time while fitting to known marginals at several different timepoints, and are motivated by inferring population dynamics from snapshot data. We rather consider costs defined according to a tree structure, in which the marginals at some of the vertices may not be known (as considered in Haasler et al. (2021) and Noble et al. (2023)), with a particular focus on applications to Wasserstein barycentre problems.

## A.4 Wasserstein barycentres

Given ℓ measures ( µ 1 , ..., µ ℓ ) and weights ( λ 1 , ..., λ ℓ ) summing to 1, the Wasserstein-2 barycentre (Agueh and Carlier, 2011) is defined as

<!-- formula-not-decoded -->

where W 2 2 ( µ i , ν ) denotes the minimum attained by the OT solution in (17) for quadratic cost function c ( x, y ) = 1 2 ∥ x -y ∥ 2 2 . Observe that this can be cast as an mmOT problem by using a star-shaped cost function c ( x 0: ℓ ) = ∑ ℓ i =1 λ i ∥ x 0 -x i ∥ 2 2 , where the marginals are prescribed on the leaf nodes S = { 1 , ..., ℓ } and the centre vertex 0 is an unobserved measure (which at the solution is the barycentre). The Wasserstein barycentre is widely studied due to its importance in applications, including in Bayesian learning (Srivastava et al., 2018), clustering (Ye et al., 2017), and representation learning (Singh et al., 2020) to name a few.

Types of regularised barycentre There are many different ways to define an entropy-regularised formulation of the Wasserstein barycentre problem. Some formulations add inner regularisation, which replaces the Wasserstein distances with an entropy-regularised version, while others also consider outer regularisation in which an entropic penalty on the barycentre is added. The ( ε, τ ) -doubly-regularised barycentre of Chizat (2023) unifies many of these problems, and aims to minimise the objective

<!-- formula-not-decoded -->

where W 2 2 ,ε ( µ i , ν ) = min π : π 0 = µ i ,π 1 = ν {∫∫ 1 2 ∥ x -y ∥ 2 d π ( x, y ) + ε KL( π ∥ µ ⊗ ν ) } . We refer to Chizat (2023) for more details regarding the different types of entropy-regularised Wasserstein barycentres.

Tree-structured costs The case of tree-structured costs for multi-marginal optimal transport is often specifically studied in the literature (Haasler et al., 2021; Noble et al., 2023), as it recovers Wasserstein barycentres as a special case as described above, and also arises in the Wasserstein propagation problem (Solomon et al., 2014, 2015). In the discrete setting, Haasler et al. (2021) show that the tree structure can be leveraged to design an efficient Sinkhorn-based algorithm.

The metric space T Recall from Section 2.3 that we identify a tree T (with vertices, edge set, and length function ( V , E , ℓ ) ) with a metric space, by associating each edge e with the interval [0 , ℓ ( e )] , which are connected according to the tree structure. The same construction is used in Noble et al. (2023) for defining the dynamic TreeSB problem that we study in this work. For a rigorous description of such constructions, see for example Hambly and Lyons (2008), Bolin et al. (2024).

## B Proofs

In this section, we give proofs of the results stated in the main paper. We first provide the proof of the TreeSB characterisation in Theorem 3.1. We will follow the presentation and proof techniques of L´ eonard (2013), making the appropriate changes to extend to the tree-structured case. We then prove the convergence of the TreeIMF procedure stated in Theorem 3.1, following the presentation and proof techniques of Shi et al. (2023).

## B.1 Existence and uniqueness of TreeSB solution

We first provide results pertaining to the existence and uniqueness of the TreeSB solution. Note that while we follow the presentation of L´ eonard (2013), one could instead rely on the existence results in Noble et al. (2023).

Let us define another static minimisation problem only over observed vertices in S as

<!-- formula-not-decoded -->

We note here that the reference measures that we consider in the various SB problems are unbounded, as we consider the reference Brownian motions to be in stationarity. We will make use of properties of the KL divergence defined relative to these measures; see the Appendix of L´ eonard (2013) for a justification of why these properties still hold when the reference measures are unbounded.

We first provide a result detailing the relationships between the dynamic and static tree-structured SB problems.

Proposition B.1 (Compare to Proposition 2.3 in L´ eonard (2013)) . The tree-structured Schr¨ odinger Bridge problems ( TreeSB dyn ) , ( TreeSB stat ) , and ( TreeSB S stat ) admit at most one solution P SB ∈ P ( C T ) , Π SB V ∈ P (( R d ) |V| ) , and Π SB S ∈ P (( R d ) |S| ) respectively.

If P SB solves ( TreeSB dyn ) , then P SB V solves ( TreeSB stat ) . Conversely, if Π SB V solves ( TreeSB stat ) , then ( TreeSB dyn ) is solved by mixing Brownian bridges along each edge as P SB = Π SB V Q ·|V .

Moreover, if P SB solves ( TreeSB dyn ) , then P SB S solves ( TreeSB S stat ) . Conversely, if Π SB S solves ( TreeSB S stat ) , then ( TreeSB dyn ) is solved by the corresponding mixture of Q -bridges conditioned on values at S , P SB = Π SB S Q ·|S .

Proof. The first statement follows from the strict convexity of the ( TreeSB stat ), ( TreeSB dyn ), and ( TreeSB S stat ) problems.

The second follows by using the chain rule for KL divergence, conditioning on the values at the vertices V . We obtain

<!-- formula-not-decoded -->

We see that KL( P ∥ Q ) ≥ KL( P V ∥ Q V ) , with equality if and only if P ( ·| X V ) = Q ( ·| X V ) for P V -a.e. X V (assuming KL ( P ∥ Q ) &lt; ∞ ). Therefore, P solves the dynamic problem if and only if it decomposes as a mixture over bridges Q ·|V according to the coupling P V solving the static problem, i.e. P SB = Π SB V Q ·|V (if this were not true, then P SB V Q ·|V would be a valid solution with lower KL divergence relative to Q , contradicting optimality of P SB ). Note that such bridges just consist of Brownian bridges along the individual edges.

The third part follows similarly to the second. Consider the KL decomposition but instead conditioning only on the values on the observed vertices S ,

<!-- formula-not-decoded -->

Now we have KL( P ∥ Q ) ≥ KL( P S ∥ Q S ) , with equality if and only if P ( ·| X S ) = Q ( ·| X S ) for P S -a.e. X S (assuming KL( P ∥ Q ) &lt; ∞ ). So in particular, we have that P SB must be a mixture of Q -bridges according to its own coupling over S , i.e. P SB = P SB S Q ·|S .

In the following, we will utilise this equivalence between the ( TreeSB dyn ) and ( TreeSB S stat ) problems. We now present an auxiliary result giving a criterion for the tree-structured SB problems to have a solution, in the vein of Lemma 2.4 in L´ eonard (2013). This is a technical result required to deal with the fact that we are considering an unbounded reference measure Q .

Lemma B.2 (Compare to Lemma 2.4 in L´ eonard (2013)) . Let B : R d → [0 , ∞ ) be a measurable function such that

<!-- formula-not-decoded -->

and for each i ∈ S take a µ i ∈ P ( R d ) such that ∫ B d µ i &lt; ∞ .

Note that inf(TreeSB S stat ) = inf(TreeSB dyn ) ∈ ( -∞ , ∞ ] (from the previous result). The static and dynamic tree-structured SB problems ( TreeSB S stat ) and ( TreeSB dyn ) for the µ i have a (unique) solution if and only if inf(TreeSB S stat ) = inf(TreeSB dyn ) &lt; ∞ (that is, if and only if the marginals µ i are such that there exists some Π 0 ∈ P (( R d ) |S| ) satisfying Π 0 i = µ i for each i ∈ S , and KL(Π 0 ∥ Q S ) &lt; ∞ ) .

Proof. In light of the equivalence in Proposition B.1, we can consider just the static problem ( TreeSB S stat ). Since the marginals µ i are tight on R d , it easily follows that the closed constraint set Γ( { µ i } i ∈S ) := { Π ∈ P (( R d ) |S| : Π i = µ i ∀ i ∈ S} is uniformly tight and thus compact in P (( R d ) |S| . From the characterisation of KL divergence with respect to the unbounded measure Q S (see L´ eonard (2013), Appendix A), we have KL(Π ∥ Q S ) = KL(Π ∥ Q B S ) -∫ ( R d ) |S| ∑ i ∈S B ( x i )dΠ( x S ) -z B , where Q B S is the normalised measure Q B S = 1 z B exp( -⊕ S B ) Q S . For Π ∈ Γ( { µ i } i ∈S ) , we have

<!-- formula-not-decoded -->

The lower-semicontinuity of Π ↦→ KL(Π ∥ Q B S ) , and the assumption ∫ B d µ i &lt; ∞ for each i ∈ S , together imply that KL(Π ∥ Q S ) is lower bounded and lower semi-continuous on the compact set Γ( { µ i } i ∈S ) . Thus the static problem ( TreeSB S stat ) admits a solution if and only if inf ( TreeSB S stat ) &lt; ∞ .

In light of Lemma B.2, we can state the following result which provides conditions for the existence of the ( TreeSB dyn ) solution. Recall we will consider an unnormalised Brownian reference measure, which satisfies Q i = Leb for each i ∈ S , so the following result applies with m = Leb .

Proposition B.3 (Compare to Proposition 2.5 in L´ eonard (2013)) . Suppose that Q i = m for each i ∈ S , for a positive measure m . We have the following results.

- (a) For ( TreeSB S stat ) and ( TreeSB dyn ) to have a solution, it is necessary to have KL( µ i ∥ m ) &lt; ∞ for each i ∈ S .
- (b) For sufficient conditions: Suppose there exists measurable functions A,B : R d → R ≥ 0 satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (iii) ∫ R d ( A + B )d µ i &lt; ∞ for each i ∈ S .
- (iv) KL( µ i ∥ m ) &lt; ∞ for each i ∈ S .

Then there exists a unique solution to the SB problems ( TreeSB S stat ) and ( TreeSB dyn ) .

Proof. The first statement (a) follows by applying the result in Lemma B.2 and using the fact that KL( µ i ∥ Leb) ≤ KL(Π 0 ∥ Q S ) &lt; ∞ .

For the second, we take the independent coupling Π 0 S = ⊗ i ∈S µ i , and observe that (under the above assumptions) it satisfies the conditions in Lemma B.2. Clearly, it has the correct marginals Π 0 i = µ i by construction, so it remains only to check that KL(Π 0 S ∥ Q S ) &lt; ∞ . By expanding the KL divergence and using the inequalities in (i) and (iii), one sees that the inequalities in (iv) are sufficient to ensure KL(Π 0 S ∥ Q S ) &lt; ∞ .

## B.2 Characterisation of TreeSB solution

We now show the characterisation of the solution to Equation ( TreeSB dyn ) stated in Theorem 3.1, upon which the IMF procedure depends. We first recall that the solution P SB is a mixture of bridges of Q , conditional on the values at vertices in S .

Proposition B.4 ( Reciprocal process ) . The solution P SB to ( TreeSB dyn ) (if it exists) is in the reciprocal class R S ( Q ) , i.e. P SB = P SB S Q ·|S .

Proof.

This is a consequence of the third part of Proposition B.1.

We now show that for a Markov reference process Q , the ( TreeSB dyn ) solution is also Markov. Following L´ eonard (2013), to present the results we will use the following characterisation of Markovianity for a path measure P on the tree structure. Consider an edge e ∈ E and a time t e ∈ [0 , T e ] . Consider the time t e as splitting the continuous tree into two distinct sections (note that such a splitting may not be unique, as the chosen point may correspond to a vertex; in such cases consider any such split into two distinct parts). Denote the restrictions of a process X to these two parts as X ≤ and X ≥ . Then we say P is Markov to mean that P ( X ≤ ∈ · , X ≥ ∈ ··| X e t ) = P ( X ≤ ∈ ·| X e t ) P ( X ≥ ∈ · · | X e t ) for any such split (together with the technical assumption that some time-marginal of P is σ -finite; see discussion in L´ eonard (2013)). Note that the Brownian reference process considered in the main paper is Markov.

Proposition B.5 ( Markov process ) . For Markov reference measure Q , the solution P SB to ( TreeSB dyn ) (if it exists) is Markov.

Proof. The proof follows that of Proposition 2.10 in L´ eonard (2013). We outline the following changes to the notation for our setting, then the argument follows the same way.

Consider an edge e ∈ E and a time t e ∈ [0 , T e ] , along with a corresponding split of the tree at time t e as described above. We define the following notation: Let C ≤ T = { ω ≤ : ω ∈ C T } and C ≥ T = { ω ≥ : ω ∈ C T } be the spaces of continuous paths on the two sections of the tree respectively. For a path measure P ∈ P ( C T ) , let P t e ,z = P ( ·| X e t e = z ) ∈ P ( C T ) be the measure conditioned on the process X taking value x at the time t e , and moreover define its restrictions to the two sections as P t e ,z ≤ t e and P t e ,z ≤ t e respectively.

Similarly to L´ eonard (2013) we now make the following claim, from which the result follows.

Claim B.6. Fix a time on the tree t e as described above. Fix a z ∈ R d , a measure µ ∈ P ( R d ) , and path measures on the 'before' and 'after' sections ˜ P t e ,z ≤ ∈ P ( C ≤ T ∩ { X t e = z } ) and ˜ P t e ,z ≥ ∈ P ( C ≥ T ∩ { X t e = z } ) . Consider minimising KL( ·∥ Q ) over path measures P ∈ P ( C T ) constrained to satisfy P t e = µ , P t e ,z ≤ = ˜ P t e ,z ≤ , and P t e ,z ≥ = ˜ P t e ,z ≥ . Then the objective KL( ·∥ Q ) attains its unique minimum at P ∗ ( · ) = ∫ R d ˜ P t e ,z ≤ ⊗ ˜ P t e ,z ≥ µ (d z ) .

̸

Given the claim, the result follows according to the following argument: Suppose for a contradiction that the SB solution, here denoted ˜ P , was not Markov. Then, there exists some time t e and a correspond split of the tree such that ˜ P ( ·| X e t e ) = ˜ P ≤ ( ·| X e t e ) ⊗ ˜ P ≥ ( ·| X e t e ) . Applying the above claim with µ = ˜ P t e , we see that P ∗ and ˜ P have the same marginals at all time-points on the tree, but P ∗ attains a strictly lower KL divergence KL( P ∗ ∥ Q ) &lt; KL( ˜ P ∥ Q ) , contradicting the optimality of ˜ P .

The proof of the claim uses Jensen's inequality and proceeds exactly as the proof of Claim 2.11 in L´ eonard (2013), with the appropriate notation changes.

We now provide the characterisation of the TreeSB solution in Theorem 3.1. While the previous results have been for a general reference measure Q , we present the following results for Q associated to running Brownian motions ( σB e t ) t ∈ [0 ,T e ] along each edge, as considered in the main paper.

Theorem 3.1 ( TreeSB characterisation ) . Under mild assumptions (in the Brownian case, namely that ∫ ∥ x ∥ 2 d µ i ( x ) &lt; ∞ and H( µ i ) &lt; ∞ for each i ∈ S ), there exists a unique solution to the dynamic TreeSB problem ( TreeSB dyn ) . The solution is the unique process P that is both Markov and in R S ( Q ) with correct marginals P i = µ i for i ∈ S .

Proof. As we are considering a Brownian reference process, the assumptions in question are that ∫ ∥ x ∥ 2 d µ i ( x ) &lt; ∞ , and H( µ i ) &lt; ∞ for each i ∈ S . One can then verify (as in De Bortoli et al. (2024), Lemma D.2) that the criteria in Proposition B.3 hold by taking functions A and B to be quadratic, from which uniqueness and existence of the solution follow. From Proposition B.4 we have that the solution is in R S ( Q ) , and from Proposition B.5 we have that it is Markov.

We now need to show that if a measure P 0 is Markov and in R S ( Q ) , and has the correct marginals P 0 i = µ i for i ∈ S , then it is the TreeSB solution. Note first that as P 0 is Markov and reciprocal, its restriction to each edge e = ( u, v ) is also Markov and reciprocal along that edge. Thus, by Theorem 2.14 in L´ eonard et al. (2014) (noting that the required criterion holds for the Brownian reference measure) we have that d P 0 e d Q e = f u ( X u ) f v ( X v ) , Q e -a.e. for some non-negative measurable functions f u , f v . Recall that the path measures are a composition of the path measures along each edge according to the tree structure, so this means that d P 0 d Q = ∏ i ∈V f i ( X i ) , Q -a.e. for some non-negative measurable functions f i (via relabelling of the functions). Note too that P 0 is in R S ( Q ) , so we can also express the Radon-Nikodym derivative as d P 0 d Q = h ( { X i } i ∈S ) for some non-negative measurable function h . Equating the two expressions, we see that we in fact must have a decomposition only over vertices in S , d P 0 d Q = dΠ 0 d Q V = ∏ i ∈S f i ( X i ) , Q -a.e. for some non-negative measurable functions f i (where Π 0 denotes the static coupling of P 0 over the vertices V ).

The remainder follows the standard argument characterising the SB solution using the decomposition according to potentials (see e.g. Nutz (2021)). Consider static couplings in the constraint set Π ∈ Γ( { µ i } i ∈S ) := { Π ∈ P (( R d ) |V| : Π i = µ i ∀ i ∈ S} such that KL(Π | Q V ) &lt; ∞ . By the above, we have that E Π [log( dΠ 0 d Q V )] = ∑ i ∈S ∫ log f i d µ i , which is in particular independent of the choice of Π (for a precise statement taking care regarding the integrability of the potentials, follow the argument of Proposition 2.17 in Nutz (2021)). Therefore, for any such Π we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we thus see that Π 0 is the minimiser of the ( TreeSB stat ) problem.

## B.3 Properties of the tree-structured projections

We now move on to proving properties of the TreeIMF procedure. We follow the presentation of Shi et al. (2023). We begin by proving the properties of the reciprocal and Markovian projections defined in Definitions 3.3 and 3.4. For a full set of required assumptions, see the assumptions A.1, A.2 and A.3 in Shi et al. (2023) Appendix C.2, which are standard in the literature and we assume to hold along each edge.

The following result follows similarly to the standard IMF case.

Proposition 3.3. For P ∈ P ( C T ) , the reciprocal projection Π ∗ = proj R S ( Q ) ( P ) solves the minimisation problem Π ∗ = arg min Π ∈R S ( Q ) KL( P ∥ Π) .

Proof. This follows from the KL decomposition used in Proposition B.1, conditioning on the values on the observed vertices S :

<!-- formula-not-decoded -->

Given we are optimising Π over the reciprocal class R S ( Q ) , we have that the bridges Π( ·| X S ) = Q ( ·| X S ) are fixed. Thus, the minimiser is achieved by taking Π S = P S , that is Π = proj R S ( Q ) ( P ) .

To prove subsequent results, we require the following decomposition of the KL divergence according to the tree structure. Note that we consider Markov and reciprocal processes on the tree, both of which factorise according to the tree structure so the following decomposition can be applied.

Lemma 3.2 ( KLdecomposition along tree ) . Take path measures P , ˜ P that share a marginal at the root, P r = ˜ P r , and factorise along the tree. Under mild assumptions, we have the KL decomposition

<!-- formula-not-decoded -->

Proof. This is a consequence of the iterative application of the chain rule for KL divergence applied according to the tree structure, and the conditional independence caused by the tree structure. By first applying the chain rule for the KL divergence conditional on the value at the root vertex r , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that the edge set E is depth-wise ordered. We can again apply the KL chain rule to the term inside the expectation, now conditioned on the process along the first edge e 1 .

<!-- formula-not-decoded -->

We can iteratively apply similar decompositions as we traverse the edges according to the ordered edge set.

<!-- formula-not-decoded -->

where in the second line we use the factorisation property along the tree structure (here, s ( e ) denotes the starting vertex of an edge e ). Applying such decompositions for each edge in the ordered edge set E , one obtains

<!-- formula-not-decoded -->

as required.

We now consider the Markov projection defined in Definition 3.4. Note that the restriction of the TreeSB solution to each edge is itself an SB (because it is Markov and reciprocal), and thus can be associated with an SDE (Dai Pra, 1991). In the definition of the Markov class in Definition 3.1 we define the Markov class M T via considering SDEs with locally Lipschitz drifts. The restriction to locally Lipschitz drifts is a technical requirement for applying the entropic version of Girsanov's theorem; this is standard in the literature and does not affect our methodology. Following Shi et al. (2023), we now provide a result showing that the Markov projection also solves a minimisation problem.

Proposition 3.4. Under mild assumptions, the Markovian projection M ∗ = proj M (Π) solves the minimisation problem M ∗ = arg min M ∈M T KL(Π ∥ M ) .

Proof. Applying the KL decomposition in Lemma 3.2, we have

<!-- formula-not-decoded -->

We now analyse the individual KL expressions KL(Π e ( ·| X u ) ∥ M e ( ·| X u )) along each edge e = ( u, v ) , using the proof techniques of Proposition 2 in Shi et al. (2023).

In particular, applying the argument in the proof of Proposition 2 in Shi et al. (2023), one sees that each conditional process Π e ·| 0 is Markov and can be associated with ( X e t ) t ∈ [0 ,T e ] given by

<!-- formula-not-decoded -->

Therefore, letting the restriction of the Markov process to edge e (denoted above as M e ) be associated with a process d Y e t = v e ( t, Y e t )d t + σ d B e t such that KL(Π e ( ·| X u ) ∥ M e ( ·| X u )) &lt; ∞ , with v e locally Lipschitz, then (using e.g. L´ eonard (2012), Theorem 2.3) one obtains

<!-- formula-not-decoded -->

Thus, substituting back into (37) we have

<!-- formula-not-decoded -->

This expression is minimised by taking v ∗ e ( t, x ) = σ 2 E Π e T | t [ ∇ log Q e T | t ( X e T | X e t ) | X e t = x ] along each edge e ∈ E . This corresponds to performing a Markovian projection along each edge e according to the coupling Π e , which is exactly the definition of the tree-based Markovian projection in Definition 3.4.

We also note that, as an instance of bridge matching, along each edge the process Π e t and its corresponding Markovian projection M e, ∗ t satisfy the same Fokker-Planck equation (Peluchetti (2022), Theorem 2). Thus by the uniqueness of the solutions of the Fokker-Planck equations under A.1 and A.3 in Shi et al. (2023) (see e.g. Bogachev et al. (2021)), they share the same marginals M e, ∗ t = Π e t .

## B.4 TreeIMF convergence

Wefollow the presentation of convergence in Shi et al. (2023), but with the appropriate modifications to the proofs for the tree-based setting.

Lemma B.7 ( Pythagorean property , compare to Shi et al. (2023), Lemma 6) . Take a Markovian process M ∈ M T and a reciprocal process Π ∈ R S ( Q ) . Under mild assumptions, if KL(Π ∥ M ) &lt; ∞ then we have

<!-- formula-not-decoded -->

If KL( M ∥ Π) &lt; ∞ then we have

<!-- formula-not-decoded -->

Proof. Proof of (43) : The first identity follows from algebraic manipulations of expressions for the relevant KL divergences. From the proof of Proposition 3.4, we have

<!-- formula-not-decoded -->

(where we suppress the superscript e on the X t for notational convenience). Likewise, it can be shown that

<!-- formula-not-decoded -->

From another application of the expression in Proposition 3.4, we have

<!-- formula-not-decoded -->

where to obtain the second line, we have expanded out the square and taken expectations over X 0 in the cross-term.

Using these expressions, by applying the same algebraic manipulations as the proof of Lemma 6 in Shi et al. (2023) to each term in the summations, one obtains

<!-- formula-not-decoded -->

as required.

Proof of (44) : The second part also follows similarly to Shi et al. (2023), but instead conditioning on the values at S . Let Π ∗ = proj R S ( Q ) ( P ) = P S Q ·|S . Using the change of measure formula for KL divergence and the fact that Π and Π ∗ have the same bridges conditional on S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as required. Note that in the second line we have used the fact that Π and Π ∗ have the same bridges conditional on S , and in the third line we have used that P S = Π ∗ S by construction.

Algorithm 2: TreeDSBM for Wasserstein barycentre computation

Input: Initial coupling Π 0 S over measures µ i (e.g. independent),

Number of IMF iterations N ,

Entropic regularisation parameter σ .

Construct initial reciprocal process as Π 0 = Π 0 S Q ·|S . That is,

- Sample Y S from the initial coupling Π 0 S over the marginals µ i ;
- Sample from the unknown central marginal as Y 0 ∼ N ( ∑ λ i Y i , σ 2 Id ) ;
- Training samples are obtained from Brownian bridges along each edge i between Y and Y

for n ∈ { 0 , . . . , N -1 }

- i 0 . do

Learn 2 |E| vector fields using bridge-matching loss (15) along each edge, using samples from current reciprocal process Π n , to obtain Markovian process M n +1 ;

Construct next reciprocal process Π n +1 = M n +1 S Q ·|S using samples from M n +1 S . That is,

- Simulate M n +1 starting from a chosen root r ∈ S (one of the known measures µ i ) to obtain samples Y S from M n +1 S ;
- Sample from the unknown central marginal as Y 0 ∼ N ( ∑ λ i Y i , σ 2 Id ) ;
- Training samples are obtained from Brownian bridges along each edge i between Y i and Y 0 .

end

We can finally state the result regarding convergence of the TreeIMF iterates to the TreeSB solution.

Theorem 3.5 ( Convergence of TreeIMF ) . Under mild conditions, the TreeSB dyn solution P ∗ is the unique fixed point of the TreeIMF iterates P n , and we have lim n →∞ KL( P n ∥ P ) = 0 .

Proof. In light of the Pythagorean property in Lemma B.7, this follows from the same compactness argument as in Proposition 7 and Theorem 8 in Shi et al. (2023).

## C Implementation details and extensions

Algorithm for barycentre setting In Algorithm 2, we provide a simplified and more explicit version of Algorithm 1 for the case of computing Wasserstein barycentres.

Illustration of TreeDSBM The TreeDSBM procedure for Wasserstein barycentre computation (that is, for a star-shaped tree) is illustrated in Figure 1. We also provide a diagram for a more general tree structure in Figure 4.

## C.1 Implementation details and design choices

We implement the TreeDSBM procedure using the JAX framework (Bradbury et al., 2018). Below, we outline some of the design considerations for implementing the method.

Vector field parameterisation In our implementation, we use separate neural networks to parameterise the vector fields for each direction along each edge, totalling 2 |E| networks in total. Note that one could alternatively use a shared network along each edge with an additional binary input indicating the direction; this parameterisation was used in De Bortoli et al. (2024).

Loss function We incorporate both the forwards and backwards losses in Equation (15) into a single loss function, and thus optimise the forward and backwards directions simultaneously. One could incorporate all edges into a single loss function and train simultaneously, or alternatively could parallelise the edge optimisations across devices because the edges are optimised independently. This is a strength of our approach and can lead to large speed-ups in training, as training time does not need to increase in proportion to the number of edges.

Simulation As we train both directions along each edge, simulation of the SDEs along the tree structure can be initialised from any of the observed nodes in S . Following Shi et al. (2023), one can rotate the starting vertex between IMF iterations. This helps to mitigate any drift that accumulates in the marginals, as the coupling samples will only use true samples from the current starting marginal; for an analysis of this, see De Bortoli et al. (2024). In our experiments, we often did

(a) Reciprocal process Π : The Y S ( × ) are sampled from the current coupling Π S over S . Conditional on the Y S , points ( × ) at marginals V\S are sampled as Y S c |S ∼ Q S c |S . Brownian bridges are drawn along the edges between the samples.

<!-- image -->

(b) Markovianised process P : Vector fields are trained by bridge-matching along each edge. Samples ( × ) from the next coupling Π S are obtained by simulating the resulting SDEs along the tree structure, started at one of the known marginals.

<!-- image -->

Figure 4: The two stages of the TreeIMF procedure, for a non-star-shaped tree structure. On this tree, the marginals at the leaf vertices S = { 1 , 3 , 5 , 6 , 7 } (blue) are fixed. The marginals at vertices V\S (red) are not fixed, and change during the procedure.

not observe noticeable drift in the marginals, and instead generated samples in the coupling by initialising equally across the observed vertices in S . In our sampling, we used an Euler-Maruyama discretisation scheme with 50 uniformly-spaced steps.

Initial coupling Unless otherwise stated, we used the independent coupling Π 0 S = ⊗ i ∈S µ i as the initial coupling. We note that any coupling over V with correct marginals on S could be used, and we discuss some possible alternatives in subsequent sections.

Architectures Other than image experiments, we use a basic MLP-based vector field model. It consists of an MLP spatial embedding with hidden layers [128, 256] to embed into dimension 32, a time embedding consisting of a sine positional-encoding and an MLP with hidden layers [128, 256] also embedding into dimension 32, before concatenating the embeddings and passing through an MLP with hidden layers [512, 256, 128]. This is the same architecture used in De Bortoli et al. (2021) and Noble et al. (2023). In image experiments, we use the UNet architecture (Ronneberger et al., 2015) with the improvements from Dhariwal and Nichol (2021), using the JAX implementation from Song et al. (2023). For all experiments, we use the Adam optimiser (Kingma and Ba, 2015) with default parameters 0.9 and 0.999.

We note that for pointcloud experiments, capacity of the neural networks is not a limiting factor-any sufficiently-expressive network will work similarly well and fairly small MLP networks suffice, so using several networks for the different edges does not pose issues. One could also use a single network across the edges and additionally condition on the edge. This makes particular sense for problems with shared structure between edges, such as those for image data (and indeed these are settings for which the networks would be larger, and maintaining multiple networks could become a computational bottleneck).

Memory requirements The parallel nature of the TreeDSBM algorithm during training provides practitioners with a trade-off between memory consumption and wall-clock time. Namely, one can train the edges simultaneously if compute allows (either on a single GPU if enough memory, or parallelised across GPUs). If this cannot be done, one can train sequentially instead (in which case memory requirements for training would be comparable with standard bridge-matching). We report GPU consumption for different experiments in Table 5 (for sequential and joint training), for the hyperparameters used in the paper. These can of course be changed significantly by changing hyperparameters such as batch size. We note that in some of these experiments (e.g. the Gaussian experiments), peak GPU usage is due to the simulation and storage of the training samples for subsequent IMF steps, rather than during the network training. This can be reduced significantly by simulating in smaller batches, or by updating the cache during training rather than simulating all beforehand.

Table 5: Comparing memory usage (in MB) for sequential and joint training, for the hyperparameters used.

|                            |   Sequential Training (MB) |   Joint Training (MB) |
|----------------------------|----------------------------|-----------------------|
| 2d                         |                        447 |                   687 |
| Data Aggregation (Poisson) |                        705 |                  1219 |
| Gaussian ( d = 64 )        |                       1239 |                  1239 |
| MNIST 2,4,6                |                       4683 |                  6407 |

We remark that TreeDSBM has improved memory requirements compared to TreeDSB. In TreeDSB, training a time-reversal along an edge requires saving entire trajectories simulated along the reverse direction, whereas TreeDSBM only requires storing endpoints (this is one of the most significant benefits of IMF over IPF).

Alternative reference measures We have presented the methodology according to using Brownian reference measures along each edge. However, the methodology extends to other reference process, as long as the respective bridging processes conditioned on the values at S are tractable (for example, this is true for Ornstein-Uhlenbeck processes). The connection to quadratic-cost optimal transport is however less simple beyond the Brownian case. See Shi et al. (2023) for a more detailed treatment of this general case.

## C.2 Methodology extensions for improving convergence speed

We now discuss possible extensions of the TreeDSBM algorithm to improve convergence speed, in the vein of existing extensions of Schr¨ odinger bridge methodology for the standard two-marginal setting.

Warmstarting with minibatch mmOT couplings The TreeIMF procedure can be initialised with any coupling Π 0 S over S with correct marginals. For the standard SB problem, Tong et al. (2024b) note that the SB solution is a mixture of bridges mixed by a static ε -OT solution, and thus propose a single iteration of bridge matching on samples generated by a static ε -OT solver applied on minibatches (see also Pooladian et al. (2023), Tong et al. (2024a), and Fatras et al. (2021)).

Similarly, one can initialise TreeIMF using samples obtained from static mmOT solvers applied to minibatches. Such a procedure can speed up convergence to the TreeSB solution by initialising closer to the true solution. We remark, however, that such minibatching approaches can incur large errors (particularly in higher dimensions or for small minibatches; for example the Wasserstein-1 error grows as O ( B -1 / (2 d ) ) (Sommerfeld et al., 2019)), and so the advantages of such methods are lessened as dimensionality increases.

Flow-based IMF on the tree Iterative Markovian Fitting presents a mathematically elegant approach for solving the SB problem, with significant practical improvements over the IPF procedure. However, the iterative nature of the algorithm remains a downside - each iteration of the two-step procedure involves first simulating the current Markovian process, and then retraining a neural network with the bridge-matching loss. In practice, one might wonder if it is possible to perform the simulations and bridge-matching procedure simultaneously to avoid the expensive iterative nature of the algorithm. The recent work of De Bortoli et al. (2024) answered this in the affirmative and propose the α -IMF procedure, which instead corresponds to a discretisation of a continuous flow of processes that converge to the SB. We anticipate one could design an analogous methodology for the tree-based setting that we consider; we leave such extensions for future work.

## D Experimental details

Here, we provide experimental details and additional results and discussion regarding the experiments included in the main body.

## D.1 Synthetic 2 d barycentre

Datasets We follow the experimental setup of Noble et al. (2023). The marginals consist of moon, circle, and spiral datasets from scikit-learn (Pedregosa et al., 2011), centred and scaled by a factor of 7.0. We aim to learn the ( 1 3 , 1 3 , 1 3 )-barycentre of the dataset. This is a challenging problem-the

Figure 5: TreeDSBM samples in the 2 d experiment, comparing different regularisation values ε .

<!-- image -->

10

0

-10

10

0

10

10

0

-10

10

0

10

10

10

Figure 6: Progression of TreeDSBM ( ε = 0 . 1 ) barycentre approximations through the IMF iterations.

Table 6: Progression of the Sinkhorn divergence to the ground truth, and the average Sinkhorn divergence to the marginals, during the IMF iterations (mean ± std, over 5 runs).

|                  | IMF 1       | IMF 2       | IMF 3       | IMF 4       | IMF 5       | IMF 6       |
|------------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Fit to solution  | 17.7 ± 0.2  | 1.28 ± 0.04 | 1.21 ± 0.07 | 1.13 ± 0.05 | 1.12 ± 0.05 | 1.09 ± 0.06 |
| Fit to marginals | 0.16 ± 0.03 | 0.17 ± 0.03 | 0.18 ± 0.04 | 0.19 ± 0.03 | 0.19 ± 0.03 | 0.20 ± 0.02 |

lower-dimensional and discontinuous structures in the marginals mean that the barycentre has a complex and fragmented support, and the transport maps to the barycentre are highly discontinuous.

Hyperparameters For TreeDSBM, we use ε = 0 . 1 and run for 6 IMF iterations. For training the vector fields, we use 10,000 training steps and a batch size of 4096. We use the Adam optimiser (with default parameters 0.9, 0.999) with learning rate 1e-3 and exponential moving average parameter of 0.99. At inference we use the Euler-Maruyama scheme with 50 steps. We generate a batch of 10,000 training couplings for subsequent TreeIMF iterations (a third simulated from each marginal). For the other algorithms, we use their default parameters provided in their respective codebases.

Comparison with alternative algorithms For the ground-truth, we use the in-sample free-support barycentre algorithm of Cuturi and Doucet (2014) implemented in Python Optimal Transport (Flamary et al., 2021), with 1500 datapoints. Note that the aim of this experiment is not to outperform in-sample methods; it is known that such approaches perform well in low dimensions, but do not scale well as dimension increases. The in-sample method is used here to provide a close approximation to the ground-truth, allowing us to judge the success of the continuous Wasserstein-2 barycentre approaches that we compare.

For TreeDSB, we use the checkpoints provided by Noble et al. (2023) which were trained for 50 IPF iterations.

Wealso report results for the WIN algorithm from Korotin et al. (2022). This is an iterative algorithm inspired by ´ Alvarez-Esteban et al. (2016); it pushes forward a source latent distribution ρ through a function G to give a generative model ν = G # ρ for the barycentre, and also learns maps T i and T -1 i transporting from the generated barycentre to and from the marginals respectively. In our experiments, the barycentre generator ν = G # ρ was unable to fit the true barycentre accurately, nor were the maps T -1 i # µ i . We hypothesise that the neural maps struggle to model the discontinuous transports well. However, the combined map ( ∑ i λ i T i )# ν was able to give a good approximation of the true barycentre, which are the results we report in Figure 2 and Table 2.

Additionally, we applied the W2CB (Korotin et al., 2021) and NOTWB (Kolesov et al., 2024a) algorithms to this example, but were unable find hyperparameters to make the algorithms to converge to the correct solution. Again, this may be due to the difficulty in modelling the discontinuous transports with neural networks. We also anticipate that the loss landscapes caused by this complex

IMF 0

IMF 1

IMF 2

-10

-10

0

0

<!-- image -->

i

Figure 7: (a) The true TreeSB solution will have noise present due to entropy-regularisation. If one wishes to reduce this, one can instead (b) construct samples as ∑ i λ i Y i , or (c) use a smaller ε value.

example may have caused these methods to get stuck in local minima. It appears that the iterative schemes of TreeDSBM, TreeDSB, and WIN aid in overcoming such issues.

Convergence speed In Figure 6, we demonstrate the progression of the TreeDSBM barycentre approximation as we run the IMF iterations. We run with ε = 0 . 1 , and plot samples generated from leaf 0 . In Table 6, we also show how the Sinkhorn divergence to the ground truth evolves as the IMF iterations progress. After only two IMF iterations, TreeDSBM already gives a good approximation to the solution. Such behaviour reflects similar results reported in Lindheim (2023), which observes that iterative fixed-point approaches ( ´ Alvarez-Esteban et al., 2016) exhibit very fast convergence to the solution. TreeDSBM performs only a single bridge-matching iteration along each edge before updating the barycentre, rather than computing full OT maps (which would be expensive). As such, it strikes a good balance between the efficiency of iterative fixed-point-based approaches, without requiring full OT map computations before updating the barycentre approximation.

Runtime analysis We report approximate runtimes for the three methods that converged. All experiments were ran on a single Nvidia GeForce RTX 2080Ti GPU.

Our TreeDSBM implementation took approximately 1 minute for each IMF iteration when training the edges jointly, and took around 7 minutes to run the 6 IMF iterations. Note that one could also obtain good results using fewer IMF iterations or fewer training steps.

In contrast, the alternative methods were significantly slower. WIN required around 8000 training steps to obtain a good barycentre approximation, which took approximately 1 hour 20 minutes. The provided checkpoint for TreeDSB is for 50 IPF iterations, each of which would require 6 timereversal training procedures, and would thus take significantly longer to train than TreeDSBM.

Additional results In Figure 5, we plot TreeDSBM samples obtained for different values of entropy-regularisation ε . As expected, increasing ε leads to a slight blurring bias in the solution. In Table 6 we also report the quality of fit to the marginals, calculated by simulating from the 'moon' marginal to the centre and then out to the other leaf nodes, and averaging the resulting Sinkhorn divergences to ground-truth samples from these marginals. We see that in this experiment there is negligible drift accumulation in the marginals.

Weoverall found TreeDSBM to perform strongest in this experiment. Its bridge-matching objectives and iterative nature provide fast and stable training in what is a complex and challenging problem setting, and its dynamic-transport approach means that it is able to model the discontinuous transport maps accurately.

## D.2 MNIST 2,4,6 barycentre

We also follow the experimental setup of Noble et al. (2023) and compute the ( 1 3 , 1 3 , 1 3 )-barycentre of the 2,4 and 6 digits in the MNIST dataset (LeCun et al., 2010).

Hyperparameters We use the UNet architecture of Song et al. (2023), with 64 channels, channel multiples of (1,2,2), attention at layers (16,8), and 2 residual blocks at each layer. We train each bridge-matching procedure with 10,000 steps with batch size 64, at a learning rate of 1e-4 and with

exponential moving average weight of 0.999. For subsequent IMF iterations, we simulate 8,192 coupling samples. We run for 4 IMF iterations, beyond which we did not see much change between iterations. The TreeDSB samples plotted in Section 4 are those displayed in Noble et al. (2023), as we did not have the computational resources to run TreeDSB to convergence in this setting.

Role of ε In Section 4, the TreeDSBM barycentre has some noise present in the samples. We remark here that this noise should be present in an accurately computed solution, as we are solving for the ε -TreeSB solution which adds entropic regularisation. If one wishes for less noise in the samples, one can run with a smaller entropy-regularisation value ε . Alternatively, one could generate samples by sampling Y i according to the learned coupling, and weighting them as ∑ i λ i Y i (though note that would not be clear what kind of barycentre such samples would be from). We plot results for a smaller ε = 0 . 001 , and for the weighted coupling samples ∑ i λ i Y i for ε = 0 . 02 in Figure 7, to demonstrate that TreeDSBM can generate samples with minimal noise if so desired.

Convergence speed We provide an analysis of the convergence speed of TreeDSBM in terms of the number of IMF iterations required. Unfortunately there is no ground-truth barycentre to compare to in this example; this makes quantitative evaluation of the obtained barycentre difficult. We therefore instead assess the fit to the marginals (from which we have true samples) as a proxy for the success of the algorithm, along with the transport cost - the transport cost provides an indication of the optimality of the maps, while if the marginals are not fitted accurately, then the resulting barycentre will be unreliable. While not an ideal measure of the 'quality' of the barycentre itself, this does provide a quantitative and, importantly, tractable proxy for the 'success' of the algorithm.

To this end, we report in Table 7 the transport cost and FID values for samples from the marginals, for the 4 IMF iterations (note that we train a classifier and use the obtained features for the FID calculation, so these values should not be compared with those in other works). We initialise the sampling from 1000 unseen test samples of the digit 6, and report the FID values of the obtained 2s and 4s (averaged). We see that, as expected, the transport cost decreases as the IMF iterations proceed, indicating that the barycentre approximation is improving. We also observe that the FID scores increase slightly (though there is little visible difference) - this is a consequence of the drift that can accrue in the marginals, and is consistent with the expected behaviour of DSBM from which this effect is inherited (this can be reduced by training for longer or using the techniques discussed in Appendix C.1).

Table 7: Progression of the total transport cost and fit to the marginals (as measured by FID), during the IMF iterations. Note that the FID values are obtained using a trained classifier, so should not be compared to values in other works.

|                      |   IMF |   IMF 2 |   IMF 3 |   IMF 4 |
|----------------------|-------|---------|---------|---------|
| Transport cost       |   431 |     402 |     389 |     378 |
| Ave. FID (2s and 4s) |    61 |      82 |      89 |      93 |

## D.3 Subset posterior aggregation

The previous experiments have shown TreeDSBM to improve over its IPF counterpart TreeDSB. In the following experiments, we provide a more detailed comparison with current strongly-performing methods for continuous Wasserstein-2 barycentre estimation, by reporting results for standard experiments in the literature. We include comparisons against the methods WIN (Korotin et al., 2022), W2CB (Korotin et al., 2021), and the recent method NOTWB (Kolesov et al., 2024a). These methods have demonstrated strong empirical performance in their respective works, and are chosen here to be representative of different approaches in the literature-WIN is an iterative method inspired by ´ Alvarez-Esteban et al. (2016), W2CB is an Input Convex Neural Network-based approach (Amos et al., 2017; Makkuva et al., 2020), and NOTWB is based on recent Neural OT methodology (Korotin et al., 2023). We use the implementations in their publicly available code, to which we provide the links in Section F. We note that, as ever in barycentre studies, it is somewhat challenging to assess the performance of solvers due to the lack of the ground-truth solution (other than in certain specific examples). Here, we report results on standard experiments used in the literature.

Experimental setup Weworkwith the experimental setup and dataset used in Korotin et al. (2021) (the same dataset was also used previously in Li et al. (2020) and Fan et al. (2021)), which uses Poisson and negative-binomial regressions on a bike-rental dataset (Fanaee-T, 2013). The aim is to predict the hourly number of bike rentals using features including day of the week, weather conditions, and more. The dataset is 8-dimensional and is split into 5 distinct subsets each of size 100,000. The 'ground-truth' barycentre consists of 100,000 samples from the full dataset posterior.

Following the literature, we report the BW 2 2 -UVP metric between the 'ground-truth' and the obtained samples in Table 3. For methods that generate from each marginal, we report the average over generations from each marginal, and for WIN we report results for the barycentre generator. The BW 2 2 -UVP metric is defined as

<!-- formula-not-decoded -->

where the Bures-Wasserstein metric is defined as BW 2 2 ( ν, ˜ ν ) = W 2 2 ( N ( m ν , Σ ν ) , N ( m ˜ ν , Σ ˜ ν )) for the respective means and covariances of the distributions.

Hyperparameters For TreeDSBM, we use ε = 0 . 001 and run for 4 IMF iterations. For training the vector fields, we use 2000 training steps and a batch size of 4096. We use the Adam optimiser (with default parameters 0.9, 0.999) with learning rate 1e-3 and exponential moving average parameter of 0.99. At inference we use the Euler-Maruyama scheme with 50 steps. We generate a batch of 50,000 training couplings for subsequent TreeIMF iterations (10,000 simulated from each marginal).

For W2CB, we use a learning rate of 1e-4 in the negative binomial setting. For WIN, in the negative binomial case we rescale the source z -sampler by a factor of 10.0 to match the scale of the data better; without this, it did not appear to converge. We run W2CB and WIN for 10000 training iterations, and NOTWB for 2500 iterations. Other than those mentioned, we use the default parameters provided in the respective codebases.

Runtime analysis: We report approximate runtimes for the different approaches; all experiments were ran on a single Nvidia GeForce RTX 2080Ti GPU. To compare approximate time taken, we report time taken to for the methods to converge close to their final output - chosen by monitoring the BW 2 2 -UVP metric and choosing the time beyond which it no longer decreases significantly (note these are not the amount time used in Table 3, which we trained using the hyperparameters described above). This is somewhat subjective, but is a fairer comparison than just reporting times for running with default parameters. We remark that it may be possible to improve these runtimes with further hyperparameter tuning and by optimising the algorithm implementations, but investigating such optimisations is beyond the scope of this work.

The W2CB algorithm appeared to give good results after approximately 1000 training steps in both cases, which took around 10 minutes in our experiments. WIN converged after around 2500 iterations, which took around 45 minutes. NOTWB converged quickly after only around 200 iterations, which took around 2 minutes.

In both experiments, each TreeIMF iteration for our TreeDSBM implementation took approximately 20 seconds when training the edges jointly. TreeDSBM converged well using 4 IMF iterations, and training took around 2 minutes 30 seconds (including time for simulating training samples for the next iteration). This is comparable with NOTWB, the fastest of the alternative methods.

Convergence speed We provide the values of the BW 2 2 -UVP metric as the IMF iterations progress in Table 8, and again we observe very fast convergence.

Table 8: Progression of the BW 2 2 -UVP metric during the IMF iterations, for the subset posterior aggregation experiment (mean ± std, over 5 runs).

|                     | IMF 1       | IMF 2           | IMF 3           | IMF 4           |
|---------------------|-------------|-----------------|-----------------|-----------------|
| ↓ Poisson           | 31.1 ± 0.03 | 0.0085 ± 0.0003 | 0.0075 ± 0.0006 | 0.0076 ± 0.0005 |
| ↓ Negative Binomial | 31.0 ± 0.01 | 0.0123 ± 0.0003 | 0.0118 ± 0.0007 | 0.0121 ± 0.0004 |

## D.4 Higher-dimensional Gaussian experiments

Experimental setup We follow the experimental setup previously used in (Korotin et al., 2021, 2022; Kolesov et al., 2024a), in which 3 Gaussian distributions and its ground-truth ( 1 3 , 1 3 , 1 3 )-barycentre are randomly generated, for each dimension in { 64 , 96 , 128 } . We report results for BW 2 2 -UVP (which measures the quality of the overall generated barycentre) as described above, and additionally report the L 2 -UVP metric, which measures the quality of the individual maps to the barycentre and is defined for each marginal as

<!-- formula-not-decoded -->

where ˆ T denotes the learned map from the marginal to the barycentre, and T ∗ is the known ground truth mapping. Again, we provide the averages over the marginals.

Hyperparameters For TreeDSBM, we use ε = 1e-4 and run for 4 IMF iterations. For training the vector fields, we use 10,000 training steps and a batch size of 4096. We use learning rate 1e-3 and exponential moving average parameter of 0.99. We generate a batch of 50,000 training couplings for subsequent TreeIMF iterations (simulated equally from each marginal). Results reported for TreeDSBM are the average over 5 runs.

For the alternative methods, we run W2CB with learning rate 1e-4, and otherwise use the default parameters provided in their respective codebases.

Convergence speed We provide the values of the BW 2 2 -UVP metric as the IMF iterations progress in Table 8, and again we observe very fast convergence.

Table 9: Progression of the BW 2 2 -UVP and L 2 -UVP metrics during the IMF iterations, for the Gaussian d = 64 experiment (mean ± std, over 5 runs).

|                | IMF 1       | IMF 2       | IMF 3       | IMF 4       |
|----------------|-------------|-------------|-------------|-------------|
| ↓ BW 2 2 - UVP | 16.0 ± 0.01 | 0.12 ± 0.01 | 0.13 ± 0.02 | 0.14 ± 0.03 |
| ↓ L 2 - UVP    | 16.7 ± 0.01 | 1.19 ± 0.02 | 1.18 ± 0.02 | 1.18 ± 0.03 |

## Discussion of continuous Wasserstein-2 barycentre solver comparisons

Our experiments show that our TreeDSBM algorithm exhibits strong performance, and is competitive against state-of-the-art methods for continuous Wasserstein-2 barycentre estimation in a range of settings. In particular, TreeDSBM offers fast and stable training-even in complex settings-due to its bridge-matching loss objectives, and comes with a well-understood theoretical analysis.

The best choice of barycentre algorithm may depend on the specific problem setting. For example, if the transport maps exhibit complex behaviour (possibly due to lower-dimensional, manifold-like structures in the datasets) then the flow-based approach of TreeDSBM will likely perform strongly (such as in the 2 d barycentre experiment in Section 4). Also, when fast training is required then our experiments suggest TreeDSBM is a strong option. On the other hand, if fast inference is important then a one-step-generation solver such as NOTWB might be preferable. Note that one could incorporate distillation techniques from the flow-matching literature for improving the speed of TreeDSBM inference after training.

Overall, TreeDSBM offers a compelling new addition to the taxonomy of continuous Wasserstein-2 barycentre solvers, with distinctly different characteristics to alternative approaches due to its flowbased nature.

## E Additional experiments

## E.1 Further comments regarding computational considerations

Choice of entropy regularisation Choosing the entropy regularisation parameter ε is a perennial question in entropic OT. Standard methods to choose this value in commonly used OT libraries (for example, choosing in proportion to the costs) provide good guidance for choosing suitable

Table 10: Effect of entropy-regularisation parameter ε in the 2 d and data aggregation experiments.

| 2 d , Sinkhorn-divergence                | ε =1 . 0 1.24   | ε =0 . 3 0.99   | ε =0 . 1 1.02   |
|------------------------------------------|-----------------|-----------------|-----------------|
|                                          | ε = 1e-3        | ε = 3e-4        | ε = 1e-4        |
| Data Aggregation (Poisson), BW 2 2 - UVP | 0.012           | 0.008           | 0.008           |

Table 11: Effect of batch size in the 2 d and data aggregation experiments.

| Batch size                               |    64 |   246 |   1024 |   4096 |
|------------------------------------------|-------|-------|--------|--------|
| 2 d , Sinkhorn-divergence                | 1.57  | 1.24  |  1.04  |  1.04  |
| Data Aggregation (Poisson), BW 2 2 - UVP | 0.032 | 0.017 |  0.013 |  0.012 |

values. Typically, one may want to choose ε as small as possible the reduce the entropic bias. One advantage of TreeDSBM over TreeDSB is that it allows for much smaller epsilon (TreeDSB does not converge for too-small ε , as simulated trajectories struggle reach the other marginals). We provide a visualisation of the role of ε in the 2 d example in Figure 5, and also for two values of ε for the MNIST experiment in Figure 7. In Table 10, we also add some further quantitative results for different ε values, in the 2 d and subset posterior aggregation settings.

Fitting to the marginals One of the limitations of our approach is that errors can accumulate in the marginals as IMF iterations proceed; this is a limitation inherited from standard IMF and similar reflow methods. Standard techniques from the literature can be used to mitigate this (such as rotating the starting marginal as in Shi et al. (2023), or using the projection methods in Kim et al. (2025)). It is therefore important that the bridge-matching steps fit the marginals accurately, and hyperparameters should be chosen accordingly. To provide an indication of how the learned bridgematching quality affects the overall solution, we provide results for varying batch size on the 2 d and data aggregation experiments in Table 11, for the hyperparameters used in the paper. We have also provided results assessing how the fit to the marginals changes as the IMF iterations progress in the 2 d and MNIST experiments in Tables 6 and 7 respectively.

## E.2 Ave! Celeba benchmark

In this section, we provide an example that illustrates a potential limitation of our approach. As previously discussed, it is difficult to evaluate performance of barycentre algorithms in high dimensions due to the lack of a ground-truth. To combat this, Korotin et al. (2022) proposed the Ave, celeba! barycenter benchmark, which consists of 3 distributions of transformed CelebA faces (Liu et al., 2015), for which the ( 1 4 , 1 2 , 1 4 ) -Wasserstein-2 barycentre recovers the true CelebA dataset. The resulting dataset consists of around 67k samples in each marginal, and each image is shape 64 × 64 × 3 .

We consider applying the TreeDSBM in this example. It is known that performing bridge-matching between complex datasets such as images can be challenging, so for the first step we instead pretrain models using single bridge-matching iteration from a standard Gaussian to each marginal. To obtain the next coupling for training, we run the process from one of of the marginals to the latent representation in the Gaussian, and then out to the other marginals. This aids in learning, as there is often good structure preserved between the obtained samples from each marginal. For subsequent iterations we also warmstart the parameters from these pretrained models. The experiments were conducted on Nvidia A100 GPUs on Google Colaboratory.

Hyperparameters Weuse the UNet architecture of Song et al. (2023), with 128 channels, channel multiples of (1,2,2,2), attention at layers (32,16,8), and 4 residual blocks at each layer.

We use σ = 0 . 01 and train each bridge-matching procedure with batch size 32, at a learning rate of 1e-4 and with exponential moving average weight of 0.999. For pretraining, we run for 20,000 training steps (which takes approximately 5 hours), and for subsequent IMF iterations we run for 10,000 steps (which each take around 2.5 hours).

Figure 8: Samples from TreeDSBM applied to the Ave! Celeba benchmark. Many samples are transported well, but some pick up unwanted artifacts. We discuss the findings from this experiment below.

<!-- image -->

For training subsequent IMF iterations along each edge, we generate samples from the coupling from each datapoint in the corresponding marginal. This mitigates drift in the sample quality at the marginals, as we always use true datapoints from the marginal during training. We run for 2 IMF iterations, and did not see much change for subsequent IMF iterations beyond this.

Discussion of results TreeDSBM is able to scale to the high-dimensional setting, but the obtained samples do not match the visual quality of state-of-the-art results such as those reported in Kolesov et al. (2024a). Observe that some of the generated samples are good, but some contain additional artifacts that should not be present. We anticipate that this is due to the initial pretraining coupling. When generating samples Y i from the pretraining coupling, we simulate from one marginal to the Gaussian latent, and then out to the other marginals. This often results in strong structural similarities between the obtained Y i which yields good barycentre samples for training the next IMF iteration. However, sometimes the coupling samples Y i do not resemble each other, and the resulting barycentre sample consists of separate overlaid images. This appears to be difficult for the algorithm to recover from, resulting in the artifacts visible in some of the generated images.

Overall this suggests a limitation of TreeDSBM for this particular benchmark-the true transport maps are in fact very simple in this specific example (primarily just colour changes), but it is difficult for TreeDSBM to learn this because communication between the edges is infrequent and only occurs after each IMF iteration. In contrast, state-of-the-art methods for this benchmark optimise all the maps together and with much more interaction between them, which we anticipate is a better inductive bias for the shared structure present in this benchmark. Note that the fact that TreeDSBM optimises the edges separately is in fact a strength of the approach in many settings; it results in stable training without needing adversarial objectives, and allows for speed-ups by training the edges simultaneously. However, this experiment suggests that this may be a limitation of our method in scenarios where the true maps exhibit a lot of shared structure (as in the case in this example), as communication between edges occurs too infrequently to recognise this shared structure.

Weremark that we observed improved performance in this benchmark by using a shared architecture over the edges (conditioning on the edge and the direction), compared to using a different network along each edge. Such an architecture makes sense in this example, as there are shared features along the edges that the network can learn and this reduces the computational and memory cost. When using a shared architecture, the network also appeared to create more consistent samples from the initial coupling. However, there are still unwanted artifacts present in many of the generated samples, and so alternative methods such as Kolesov et al. (2024a) are likely more suitable for settings such as these, as discussed above.

Weanticipate that the performance of TreeDSBM in this setting could be improved through architectural changes and other implementation tricks from the flow-matching literature (for example, using preconditioned flow parameterisations (Karras et al., 2022), and techniques to mitigate marginal drift in reflow methods (Kim et al., 2025)). Such investigations offer promising directions for future work.

## E.3 Beyond star-shaped trees

So far, we have demonstrated the empirical performance of TreeDSBM only on star-shaped trees, as we have focused on computing Wasserstein barycentres. Finally, we demonstrate that our TreeDSBM also works for non-star shaped trees, and thus has potential applications beyond only barycentre computation. We consider a simple 2-dimensional example with the same tree structure

Figure 9: TreeDSBM applied to a non-star-shaped tree.

<!-- image -->

as shown in the TreeIMF diagram in Figure 4, with standard scikit-learn distributions on the observed leaves and each edge having length 1. As in the previous 2-dimensional barycentre problem, this is a challenging task due to the discontinuous transport maps and fragmented supports at the solution. We run TreeDSBM for 4 IMF iterations with ε = 0 . 1 , and train each edge for 10,000 iterations with learning rate 1e-3 and exponential moving average parameter of 0.99. We plot the obtained measures in Figure 9, and see that TreeDSBM is again able to learn the complex mappings required for this setting. While direct applications of non-star-shaped trees are less clear than in the barycentre case, examples have been studied in Haasler et al. (2021) and Solomon et al. (2015), and they could have potential applications for modelling temporal behaviour of population dynamics, for example if populations were known to split according to a known structure. We leave investigating possible applications of general tree-structured costs for future work.

## F Licenses

The following assets were used in this work.

- TreeDSB (Noble et al., 2023), MIT License https://github.com/maxencenoble/tree-diffusion-schrodinger-bridge
- WIN, Ave! Celeba dataset (Korotin et al., 2022), MIT License https://github.com/iamalexkorotin/WassersteinIterativeNetworks
- W2CB (Korotin et al., 2021), MIT License https://github.com/iamalexkorotin/Wasserstein2Barycenters
- NOTWB (Kolesov et al., 2024a), MIT License https://github.com/justkolesov/NOTBarycenters
- JAX Consistency Models (Song et al., 2023), Apache-2.0 License https://github.com/openai/consistency\_models\_cifar10
- Bike Sharing, UCI Machine Learning Repository (Fanaee-T, 2013), CC BY 4.0 License https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- MNIST digits classification dataset (LeCun et al., 2010), CC BY-SA 3.0 License
- OTT-JAX (Cuturi et al., 2022), Apache-2.0 License
- Python Optimal Transport (Flamary et al., 2021), MIT License

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In the abstract and introduction, we state that we extend the IMF procedure to the tree-based SB setting. In Section 3 we demonstrate the theoretical soundness of our proposed approach, and we explain how to implement it in Section 3.3. In Section 4 we demonstrate the empirical performance of our approach, showing it inherits the benefits on IMF over IPF in the tree-based setting.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of our approach in Section 5. We also provide an experiment illustrating a potential limitation of our approach in Appendix E.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We present simple statements of the theoretical results in the main text, with full assumptions and proofs included in Appendix B.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We report full implementation details of our reported experiments in Appendix C for reproducibility, including neural architecture and hyperparameter choices and data generation procedures.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide code to run our experiments in the supplementary material. For alternative algorithms, we use the open-source code provided by authors to which we provide links in Appendix F.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We include these details in Appendix D.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: It is not computationally feasible to include error bars for all results, due to long training times for some of the algorithms. For the TreeDSBM algorithm, we report results for mean ± std over 5 runs in Appendix D.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: These details are included in Appendix D.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed the Code of Ethics and confirm that our work conforms to these guidelines. We see no potential harmful consequences of our work, and include extensive reproducibility details in Appendix D.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is mostly theory and methodology focused for the tree-structured Sch¨ odinger Bridge problem. We do not see any immediate societal impacts, though certain applications may share similar societal consequences as in flow-based generative models upon which our approach is based.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification:

Guidelines: Our work poses no such risks.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite the works that we use, and include the licenses of code and datasets in Appendix F.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.
## Differentiable Cyclic Causal Discovery Under Unmeasured Confounders

## Muralikrishnna G. Sethuraman

School of Electrical &amp; Computer Engineering Georgia Institute of Technology muralikgs@gatech.edu

## Faramarz Fekri

School of Electrical &amp; Computer Engineering Georgia Institute of Technology faramarz.fekri@ece.gatech.edu

## Abstract

Understanding causal relationships between variables is fundamental across scientific disciplines. Most causal discovery algorithms rely on two key assumptions: (i) all variables are observed, and (ii) the underlying causal graph is acyclic. While these assumptions simplify theoretical analysis, they are often violated in real-world systems, such as biological networks. Existing methods that account for confounders either assume linearity or struggle with scalability. To address these limitations, we propose DCCD-CONF, a novel framework for differentiable learning of nonlinear cyclic causal graphs in the presence of unmeasured confounders using interventional data. Our approach alternates between optimizing the graph structure and estimating the confounder distribution by maximizing the loglikelihood of the data. Through experiments on synthetic data and real-world gene perturbation datasets, we show that DCCD-CONF outperforms state-of-the-art methods in both causal graph recovery and confounder identification. Additionally, we provide consistency guarantees for our framework, reinforcing its theoretical soundness.

## 1 Introduction

Modeling cause-effect relationships between variables is a fundamental problem in science [1, 2, 3], as it enables the prediction of a system's behavior under previously unseen perturbations. These relationships are typically represented using directed graphs (DGs), where nodes correspond to variables, and directed edges capture causal dependencies. Consequently, causal discovery reduces to learning the structure of these graphs.

Existing causal discovery algorithms can be broadly classified into three categories: (i) constraintbased methods, (ii) score-based methods, and (iii) hybrid methods. Constraint-based methods, such as the PC algorithm [4, 5, 6], search for causal graphs that best satisfy the independence constraints observed in the data. However, since the number of conditional independence tests grows exponentially with the number of nodes, these methods often struggle with scalability. Score-based methods, such as the GES algorithm [7, 8], learn graph structures by maximizing a penalized score function, such as the Bayesian Information Criterion (BIC), over the space of graphs. Given the vast search space, these methods often employ greedy strategies to reduce computational complexity. A significant breakthrough came with Zheng et al. [9], who introduced a continuous constraint formulation to restrict the search space to acyclic graphs, inspiring several extensions [10, 11, 12, 13, 14, 15] that frame causal discovery as a continuous optimization problem under various model assumptions. Hybrid methods [16, 17, 18] integrate aspects of both constraint-based and score-based approaches, leveraging independence constraints while optimizing a score function.

Most causal structure learning methods assume (i) a directed acyclic graph (DAG) with no directed cycles and (ii) complete observability, meaning no unmeasured confounders.

While these assumptions simplify the search space, they are often unrealistic, as real-world systems-especially in biology-frequently exhibit feedback loops and hidden confounders [19]. Enforcing these constraints can also increase computational complexity, particularly in ensuring acyclicity, which often requires solving challenging combinatorial or constrained optimization problems. These limitations hinder the practical applicability of existing methods in settings where such violations are unavoidable.

Several approaches have been developed to address the challenge of feedback loops within causal graphs. Early work by Richardson [20] extended constraint-based approaches for Directed Acyclic Graphs (DAGs) to accommodate directed cycles. Another key contribution came from Lacerda et al. [21], who generalized Independent Component Analysis (ICA)-based causal discovery to handle linear non-Gaussian cyclic graphs. More recently, a

Figure 1: (a) Example of a directed mixed graph G , where the bidirectional edges represent hidden confounders, with σ ij indicating their corresponding strengths; (b) Mutilated graph, do( I k )( G ) , resulting from the interventional experiment I k = { X 3 } , where all incoming edges (including bidirectional edges) to X 3 are removed.

<!-- image -->

growing body of research has focused on score-based methods for learning cyclic causal graphs [22, 23, 24, 25]. Additionally, some approaches leverage interventional data to improve structure recovery in cyclic systems. For instance, Hyttinen et al. [26] and Huetter and Rigollet [22] introduced frameworks that explicitly incorporate interventions to refine cyclic graph estimation. Sethuraman et al. [27] further advanced this line of research by introducing a differentiable framework for learning nonlinear cyclic graphs. Unlike differentiable DAG learners that enforce acyclicity through augmented Lagrangian-based solvers, their approach sidesteps these constraints by directly modeling the data likelihood, enabling more efficient and flexible learning of cyclic causal structures. However, their method assumes the absence of unmeasured confounders, which limits its applicability in real-world settings where hidden confounders are often present.

Causal discovery in the presence of latent confounders has seen limited development, with most existing approaches grounded in constraint-based methodologies. Extensions of the PC algorithm, such as the Fast Causal Inference (FCI) algorithm [28], construct a Partial Ancestral Graph (PAG) to represent the equivalence class of DAGs in the presence of unmeasured confounders, and can accommodate nonlinear dependencies depending on the chosen conditional independence tests. However, standard FCI does not incorporate interventional data, prompting extensions such as JCI-FCI [29] and related approaches [30] that combine observational and interventional settings. Jaber et al. [31] further advanced this line of work by allowing for unknown interventional targets. Additionally, Forré and Mooij [32] introduced σ -separation, a generalization of d -separation, enabling constraint-based causal discovery in the presence of both cycles and latent confounders. Suzuki and Yang [33] introduce LiNGAM-MMI, a generalization of the ICA-based LiNGAM [34] that quantifies and mitigates confounding via a KL divergence minimization. Integer-programming-based formulations have also been proposed for causal discovery under latent confounding, such as [35, 36]. A few recent approaches, such as Bhattacharya et al. [37], have explored continuous optimization frameworks using differentiable constraints, though these methods are currently limited to linear settings. In parallel, several works exist on causal inference under latent confounding-most notably Abadie et al. [38], Chernozhukov et al. [39]-propose doubly robust estimators that integrate outcome modeling, weighting, and cross-fitting for reliable effect estimation. Overall, a unified framework capable of handling nonlinearity, cycles, latent confounders, and interventions remains largely absent.

Contributions. In this work, we tackle three key challenges in causal discovery: directed cycles , nonlinearity , and unmeasured confounders . Our main contributions are:

- We introduce DCCD-CONF, a novel differentiable causal discovery framework for learning nonlinear cyclic relationships under Gaussian exogenous noise, with confounders modeled as correlations in the noise term.
- We show that exact maximization of the proposed score function results in identification of the interventional equivalence class of the ground truth graph.
- We conduct extensive evaluations, comparing DCCD-CONF with state-of-the-art causal discovery methods on both synthetic and real-world datasets.

Organization. The paper is structured as follows: Section 2 introduces the problem setup. In Section 3, we present DCCD-CONF, our differentiable framework for nonlinear cyclic causal discovery with unmeasured confounders. We then evaluate its effectiveness on synthetic and realworld datasets in Section 4. Finally, Section 5 concludes the paper.

## 2 Problem Setup

## 2.1 Structural Equations for Cyclic Causal Graphs

Let G = ( V , E , B ) represent a possibly cyclic directed mixed graph (DMG) that encodes the causal dependencies between the variables in the vertex set V = [ d ] , where [ d ] = { 1 , . . . , d } . E denotes the set of directional edges of the form i → j in G , and B denotes the set of bidirectional edges of the form i ↔ j in G . Each node i is associated with a random variable X i with the directed edge i → j ∈ E representing a causal relation between X i and X j , and the bidirectional edge i ↔ j ∈ B indicates the presence of a hidden confounder between X i and X j . Following the framework proposed by Bollen [40] and Pearl [41], we use structural equations model (SEM) to algebraically describe the system:

<!-- formula-not-decoded -->

where pa G ( i ) := { j ∈ [ d ] : j → i ∈ E} represents the parent set of X i in G , and X pa G ( i ) denotes the components of X = ( X 1 , . . . , X d ) indexed by the parent set pa G ( i ) . We exclude self-loops (edges of the form X i → X i ) from G , as their presence can lead to identifiability challenges [42]. The function F i , referred to as causal mechanism , encodes the functional relationship between X i and its parents X pa G ( i ) , and the exogenous noise variable Z i .

̸

The collection of exogenous noise variables Z = ( Z 1 , . . . , Z d ) account for the stochastic nature as well as the confounding observed in the system. We make the assumption that the exogenous noise vector follows a Gaussian distributions: Z ∼ N ( 0 , Σ Z ) . Notably, if ( Σ Z ) ij = 0 , then variables X i and X j are confounded, i.e., i ↔ j ∈ B . In other words, confounding is modeled through correlations in the exogenous noise variables. Intuitively, if X i and X j share a hidden cause, their unexplained variation (the part not accounted for by their observed parents) will tend to move together. By allowing the noise terms Z i and Z j to be correlated, this shared influence can be effectively captured. This formulation generalizes prior work by allowing cycles, extending both nonlinear cyclic models that assume independent noise terms [27], and acyclic models without confounders [43].

By collecting all the causal mechanisms into the joint function F = ( F 1 , . . . , F d ) , we can then combine (1) over i = 1 , . . . , d to obtain the equation

<!-- formula-not-decoded -->

We will use (2) to represent the causal system due to its simplicity for subsequent discussion. The observed data represents a snapshot of a dynamical process where the recursive equations in (2) define the system's state at its equilibrium. Thus, in our experiments we assume that the system has reached the equilibrium state. For a given random draw of Z , the value of X is defined as the solution to (2). To that end, we assume that (2) admits a unique fixed point for any given Z . We refer to the map f x : X ↦→ Z as the forward map, and f z : Z ↦→ X as the reverse map. In Section 3.1, we show that the chosen parametric family of functions indeed guarantees the existence of a unique fixed point. Under these restrictions, the probability density of X is well defined and is given by

<!-- formula-not-decoded -->

where J f x ( X ) denotes the Jacobian matrix of the function f x at X .

## 2.2 Interventions

In our work, we consider surgical interventions [41], also known as hard interventions , where all the incoming edges to the intervened nodes are removed from G . Given a set of intervened upon nodes (also known as interventional targets ), denoted as I ⊆ V , the structural equations in (1) are modified as follows

<!-- formula-not-decoded -->

where C i is a random variable sampled from a known distribution, i.e., C i ∼ p I ( C i ) . We denote do( I )( G ) to be the mutilated graph under the intervention I (see Figure 1). Note that X i is no longer confounded if it is intervened on.

We consider a family of K interventional experiments I = { I k } k ∈ [ K ] , where I k represents the interventional targets for the k -th experiment. Let U k ∈ { 0 , 1 } d × d denote a diagonal matrix with ( U k ) ii = 1 if i / ∈ I k , and ( U k ) ii = 0 if i ∈ I k . Similar to the observational setting, (4) can be vectorized to obtain the following form

<!-- formula-not-decoded -->

where C = ( C 1 , . . . , C d ) is a vector with C i ∼ p I ( C i ) if i ∈ I k , and C i = 0 otherwise. For the interventional targets I k ∈ I , let f ( I k ) x denote the forward map. Similar to the observational setting, we make the following assumption on the set of interventions.

Assumption 1 (Interventional stability) . Let I = { I k } k ∈ [ K ] be a family of interventional targets. For each I k ∈ I , the structural equations in (5) admits a unique fixed point given the exogenous noise vector Z .

Thus, the probability distribution of X for the interventional targets I k is given by

<!-- formula-not-decoded -->

where U k = { i : i ∈ V \ I } denotes the index of purely observed nodes, and p Z ( [ f ( I k ) x ( X ) ] U k ) is the marginal distribution of the combined vector Z , restricted to the components indexed by U k .

Given a family of interventions I , our goal is to learn the structure of the DMG by maximizing the log-likelihood of the data, in addition to identifying the variables that are being confounded by the unmeasured confounders Z . The next section presents our approach to addressing this problem.

## 3 DCCD-CONF: Differentiable Cyclic Causal Discovery with Confounders

In this section, we present our framework for differentiable learning of cyclic causal structures in the presence of unmeasured confounders. We start by modeling the causal mechanisms, then define the score function used for learning, followed by a theorem that validates its correctness. Finally, we outline the algorithm for estimating the model parameters.

## 3.1 Modeling Causal Mechanism

We model the structural equations in (2) using implicit flows [44], which define an invertible mapping between x and z by solving the root of a function G ( x , z ) = 0 , where G : R 2 d → R d . Specifically, we take G ( x , z ) = x -F ( x , z ) . General implicit mappings, however, do not guarantee invertibility or permit efficient computation of the log-determinant required for evaluating (6). To balance expressiveness with tractability, we adopt the structured form proposed by Lu et al. [44] for the causal mechanism:

<!-- formula-not-decoded -->

where g x and g z are restricted to be contractive functions. A function g : R d → R d is contractive if there exists a constant L &lt; 1 such that ∥ g ( x ) -g ( y ) ∥ ≤ L ∥ x -y ∥ for all x , y ∈ R d . This contractiveness ensures that the associated implicit map is uniquely solvable and invertible (see Theorem 1 in [44]). In other words, contractivity ensures that the process defined by the SEM converges to an equilibrium state.

Under this formulation, the forward map takes the form f x ( x ) = ( id + g z ) -1 ◦ ( id + g x )( x ) , where id denotes the identity map. Given x (or z ), the corresponding value of z (or x ) can be computed via a root-finding procedure, i.e., z = RootFind ( x -F ( x , · )) , specifically, we employ a quasi-Newton method (i.e., Broyden's method [45]) to find the root. To capture more complex nonlinear interactions between the observed variables X and latent confounders Z , multiple such implicit blocks can be stacked. This is true since f x is highly nonlinear and by suitably parameterizing g x and g z any nonlinear interaction between x and z can be modeled. For simplicity, we focus on a single implicit flow block for subsequent discussion.

We parameterize the functions g x and g z using neural networks. The adjacency matrix of the causal graph G is encoded as a binary matrix M G ∈ { 0 , 1 } d × d , representing the presence of directed edges and serving as a mask on the inputs to g x . The diagonal entries of M G are explicitly enforced to be zero to prevent self-loops. Similarly, the identity matrix is used to mask the inputs to g z . Consequently, the causal mechanism is defined as:

<!-- formula-not-decoded -->

where NN ( · | θ ) denotes a fully connected neural network parameterized by θ , ⊙ denotes the Hadamard product, and M G ∗ ,i is the i -th column of M G . The contractivity of g x and g z can be enforced by rescaling their weights using spectral normalization [46]. Moreover, the contractive nature of the causal mechanism facilitates efficient computation of the score function used for learning causal graphs, as discussed in Section 3.2.

While the contractivity assumption may seem restrictive, it ensures stability and well-posedness in the presence of directed cycles. If the causal graph is known to be acyclic, this assumption can be relaxed (see Appendix C.1).

## 3.2 Score function

Given a family of interventions I = { I k } k ∈ [ K ] , we would like to learn the parameters of the structural equation model, i.e., causal graph structure, causal mechanism, and confounder distribution. To that end, similar to prior work [27, 47, 15] in this domain we employ regularized log-likelihood of the observed nodes as the score function to be maximized. That is,

<!-- formula-not-decoded -->

where p ( k ) is the data generating distribution for the k -th interventional experiment I k , Σ Z is the parameter (covariance matrix) governing the confounder distribution p Z , θ = ( θ x , θ z ) is the combined causal mechanism parameters, and |G| denotes a sparsity enforcing regularizer on the edges of G , and p do( I k )( G ) ( X ) is given by (6).

We now present the main theoretical result of this paper. The following theorem establishes that, under appropriate assumptions, the graph ˆ G estimated by maximizing (9) belongs to the same general directed Markov equivalence class (introduced by [42]) as the ground truth graph G ∗ for each interventional setting I k ∈ I , denoted as ˆ G ≡ I G ∗ , see Appendix A.1. Due to space constraints we provide the proof sketch below, see Appendix A.3 for complete proof of Theorem 2.

Theorem 2. Let I = { I k } K k =1 be a family of interventional targets, let G ∗ denote the ground truth directed mixed graph, let p ( k ) denote the data generating distribution for I k , and ˆ G := arg max G S ( G ) . Then, under the Assumptions 1, A.13, A.14, and A.15, and for a suitably chosen λ &gt; 0 , we have that ˆ G ≡ I G ∗ . That is, ˆ G is I -Markov equivalent to G ∗ .

Theorem 2 rests on three key assumptions. Assumption A.13 ensures that the data-generating distribution lies within the model class, while Assumption A.14 guarantees that every statistical independence in the data corresponds to a σ -separation in the ground-truth graph. Finally, Assumption 1 prevents the score function from diverging to infinity.

Proof (Sketch). Building on the characterization of general directed Markov equivalence class by Bongers et al. [42], extended to the interventional setting, we show that any graph outside this equivalence class has a strictly lower score than the ground truth graph G ∗ . This follows from the fact that certain independencies present in the data are not captured by graphs outside the equivalence class. Combined with the expressiveness of the model class, this prevents such graphs from fitting the data properly.

If the intervention set consists of all single-node interventions, I = { I k } d k =1 with I k = { k } , Hyttinen et al. [26] showed that the ground truth DMG can be uniquely recovered in the linear setting. Moreover, in the absence of cycles and confounders, this result extends to the nonlinear case, as demonstrated by Brouillard et al. [15]. However, determining the necessary conditions on

interventional targets for perfect recovery in general DMGs with cycles and confounders remains an open problem. Nonetheless, in practice, we find that observational distribution in combination with single-node interventions across all nodes lead to perfect recovery of the ground truth, even in the nonlinear case, as shown in Section 4.

## 3.3 Updating model parameters

In practice, we use gradient based stochastic optimization to maximize (9). For this purpose, following Sethuraman et al. [27] and Brouillard et al. [15], the entries of adjacency matrix M ij are modeled as Bernoulli random variable with parameters b ij , grouped into the matrix σ ( B ) . We denote M ∼ σ ( B ) to indicate that M ij ∼ Bern ( b ij ) for all i, j ∈ [ d ] . In this formulation, the sparsity regularizer is ∥ M ∥ 0 , which is computationally intractable and thus we use the ℓ 1 -norm, ∥ M ∥ 1 as a proxy. Consequently, the score function in (9) is replaced by the following relaxation:

<!-- formula-not-decoded -->

where we replace the expectation with respect to data distribution in (9) with sum over the finite samples, x ( i,k ) represents the i -th data sample in the k -the interventional setting. We note that, since p Z = N ( 0 , Σ Z ) , the covariance of the exogenous confounder vector, Σ Z , is implicitly embedded within p do( I k )( G ) ( x ( i,k ) ) in the score function.

The optimization of the score function is carried in two steps. First, we optimize ˆ S ( B ) with respect to the neural network parameters θ and the graph structure parameters B . Next, we optimize ˆ S ( B ) with respect to the parameters of the exogenous noise distribution, Σ Z . However, maximizing ˆ S I ( B ) presents two main challenges: (i) computing log p X ( X ) is computationally expensive due to the presence of | det( J f ( I k ) x ( X )) | , which requires O ( d 2 ) gradient calls, and (ii) updating Σ Z via stochastic gradients could lead to stability issues as Σ Z may loose its positive definiteness.

We now describe how these challenges are addressed, along with the specific procedures for updating the individual model parameters.

## 3.3.1 Computing log determinant of the Jacobian

As discussed earlier, computing log | J f ( I k ) x ( X ) | is a significant challenge in maximizing the score function ˆ S ( B ) . To address this, we utilize the unbiased estimator of the log-determinant of the Jacobian introduced by Behrmann et al. [46], which is based on the power series expansion of log(1 + x ) . Since f ( I k ) x ( x ) = ( id + U k g z ) -1 ◦ ( id + U k g x )( x )

<!-- formula-not-decoded -->

where I ∈ R d × d denotes the identity matrix, J m U k g x represents the Jacobian matrix raised to the m -th power, and Tr denotes the trace of matrix. The series in (11) is guaranteed to converge if the causal functions g x and g z are contractive [48].

In practice, the power series is truncated to a finite number of terms, which may introduce bias into the estimator. To mitigate this issue, we follow the stochastic approach of Chen et al. [49]. Specifically, we sample a random cut-off point n ∼ p N ( n ) for truncating the power series and weight the i -term in the finite series by the inverse probability of the series not ending at i . This yields the following unbiased estimator

<!-- formula-not-decoded -->

The gradient calls can be reduced even further using the Hutchinson trace estimator [50], see Appendix B for more details.

## 3.3.2 Updating neural network and graph parameters.

In the first step of the parameter update, keeping Σ Z fixed, the parameters of the neural network θ and the graph structure B are updated using the backpropagation algorithm with stochastic gradients. The gradient of the score function ˆ S I ( B ) with respect to B is computed using the Straight-Through Gumbel estimator. This involves using Bernoulli samples in the forward pass while computing score, and using samples from Gumbel-Softmax distribution in the backward pass to compute the gradient, which can be differentiated using the reparameterization trick [51].

## 3.3.3 Updating the confounder-noise distribution parameters

In second parameter update step, we fix the value of θ and B and focus on the confounder-noise distribution parameter Σ Z . First, consider the case where no interventions are applied, i.e, I k = ∅ . Note that the dependence of ˆ S ( B ) on Σ Z arises solely from p Z , which is embedded within p do( I k )( G ) ( X ) . Therefore, we can thus ignore the remaining terms in ˆ S ( B ) and focus exclusively on p Z . Let { x ( i ) } N i =1 denote the observational data. From the forward map, we have z ( i ) = f x ( x ( i ) ) . Given that p Z = N ( 0 , Σ Z ) , the relevant parts of ˆ S ( B ) with respect to Σ Z , denoted as ˜ L ( I k ) , are expressed as:

<!-- formula-not-decoded -->

Simplifying (13) yields a more convenient form:

<!-- formula-not-decoded -->

where S = 1 n ∑ N i =1 z ( i ) ( z ( i ) ) ⊤ is the sample covariance of Z .

Maximizing (14) directly using backpropagation and stochastic gradients results in stability issues as Σ Z may lose its positive definiteness. However, Friedman et al. [52] demonstrated that the sparsity-regularized version of (14) is a concave optimization problem in Σ -1 Z that can be efficiently solved by optimizing the columns of Σ Z individually. This is achieved by formulating the column recovery as a lasso regression problem. We adopt this strategy while updating the Σ Z during the maximization of ˆ S ( B ) .

Let W = Σ Z be the estimate of the covariance matrix. We reorder W such that the column and row being updated can be placed at the end, resulting in the following partition

<!-- formula-not-decoded -->

Then, as shown by Friedman et al. [52], w 12 = W 11 β , where β is the solution to the following lasso regression problem, denoted as lasso ( W 11 , s 12 , ρ ) :

<!-- formula-not-decoded -->

where y = W -1 / 2 11 s 12 , and ρ is the regularization constant that promotes sparsity in Σ -1 Z .

In an interventional setting I k , the dependence of ˆ S ( B ) on Σ Z arrises from the marginal distribution of Z restricted to components indexed by U k , i.e., purely observed nodes. Since Z follows a Gaussian distribution, Z U k also follows a Gaussian distribution with Z U k ∼ N ( 0 , ˜ Σ I k ) . From the properties of Gaussian distribution [53], we have ˜ Σ I k = ( Σ Z ) U k , U k . Consequently, for the interventional setting I k , (14) becomes

<!-- formula-not-decoded -->

where S I k = 1 n ∑ N i =1 z ( i ) U k ( z ( i ) U k ) ⊤ is the sample covariance of Z corresponding to the purely observed nodes. In this case, we set W = ( Σ Z ) U k , U k and the rest of the update procedure remains the same. The overall parameter update procedure is summarized in Algorithm 1 in Appendix B.

Figure 2: Performance of causal graph and confounder recovery under varying problem dimensions. In all the cases the number of observed variables is fixed at d = 10 . (Top row, left column) number of latent confounders ranges from 2 to 8, (top row, right column) number of cycles ranges from 0 to 8. (Bottom row, left column) the degree of nonlinearity β is varied between 0 and 1, (bottom row, right column) the number of training interventions is varied between 0 and 10.

<!-- image -->

## 4 Experiments

The code for DCCD-CONF is available at the repository: https://github.com/muralikgs/ dccd\_conf .

We evaluated DCCD-CONF on both synthetic and real-world datasets, comparing its performance against several state-of-the-art baselines: NODAGS-Flow [27], LLC [26], DAGMA [54], and the linear ADMG recovery method proposed by Bhattacharya et al. [37] (which we refer to as ADMG). NODAGS-Flow learns nonlinear cyclic causal graphs but does not model unmeasured confounders. LLC accounts for confounders but is limited to linear cyclic SEMs. DAGMA handles nonlinearity under causal sufficiency while being limited to acyclic graphs. ADMG handles confounding but is limited to acyclic graphs and linear SEMs. Note that both DAGMA and ADMG do not natively support interventional data and hence we use these models in combination with the Joint Causal Inference (JCI) framework [29] and treat interventions as multiple contexts. We also include a comparison between DCCD-CONF and two constraint based models LiNGAM-MMI [33] and JCI-FCI [29] in the Appendix (see Appendix C).

## 4.1 Synthetic data

In all synthetic experiments, the cyclic graphs were generated using Erd˝ os-Rényi (ER) random graph model with the outgoing edge density set to 2. We evaluated DCCD-CONF and the baselines on both linear as well as nonlinear SEMs described in Section 2. Our training data set consists of observational data and single-node interventional over all the nodes in the graph, i.e, I = { ∅ , { 1 } , . . . , { d } } (unless stated otherwise), with N k = 500 samples per intervention. Furthermore, in all the experiments presented here, the SEM was constrained to be contractive. However, we also compare the performance of DCCD-CONF to the baselines on non-contractive SEMs in the appendix. For causal graph recovery (directed edges), we use the normalized structural Hamming distance (SHD) as the error metric. SHD counts the number of operations (addition, deletion, and reversal) needed to match the estimated causal graph to the ground truth, and normalization is done with respect to the number of nodes in the graph (lower the better). For confounder identification (bidirectional edges), we compare the non-diagonal entries of the estimated confounder-noise covariance matrix to those of the ground truth. We use F1 score as the error metric (higher the better). More details regarding the experimental setup is provided in Appendix B.

Impact of confounder count. We evaluate the performance of DCCD-CONF and the baselines using the previously defined error metrics, varying the confounder ratio (number of confounders divided by the number of nodes) from 0.2 to 0.8. In this case, the number of nodes in the graph is set to d = 10 . The results, summarized in Figure 2a, show that DCCD-CONF consistently achieves lower SHD across all confounder ratios in both linear and nonlinear SEMs. Notably, in nonlinear SEMs, DCCD-CONF outperforms all baselines in causal graph recovery. Additionally, it demonstrates competitive results in confounder identification, highlighting its robustness in both tasks.

Impact of number of cycles. With d = 10 nodes and a confounder ratio of 0.3, we vary the number of cycles in the graph from 0 to 8. Figure 2b compares the performance of DCCD-CONF with the baselines under this setting. As shown, increasing the number of cycles does not lead to any noticeable degradation in performance for either directed or bidirected edge recovery.

Impact of degree of nonlinearity. In this experiment, we vary the degree of nonlinearity in the SEM by adjusting β between 0 and 1, where

<!-- formula-not-decoded -->

The SEM is fully linear when β = 0 and fully nonlinear when β = 1 . Figure 2c summarizes the results. As shown, DCCD-CONF attains the highest performance as β approaches one, for both directed and bidirected edge recovery. When β is small (i.e., the system is more linear), LLC slightly outperforms DCCD-CONF, with both models performing comparably around β = 0 . 25 .

Impact of number of interventions. In this section, we evaluate graph recovery performance as the number of training interventions K varies from 0 to d , with d = 10 fixed. The case K = 0 corresponds to the observational dataset. Results for the nonlinear SEM setting are presented in Figure 2d. As illustrated, with fewer interventions all DCCD-CONF and the baselines tend to exhibit similar performance (less then 3 interventions). As the numbre of interventions increase, the performance gap widens with DCCD-CONF dominating all of the baselines. It is also worth noting that LLC cannot operate in the purely observational setting ( K = 0 ).

Scaling with nodes. We compare the performance of DCCD-CONF and the baselines as the number of nodes ( d ) varies from 10 to 80, with results summarized in Figure 3. The number of confounders is set to 0 . 3 d . As the number of nodes increases, SHD rises across all methods, reflecting the increased difficulty of causal graph recovery in larger graphs. However, DCCD-CONF consistently outperforms the baselines in many cases, achieving lower SHD and higher F1 score, suggesting superior scalability with increasing graph size.

## 4.2 Real World data

We evaluate DCCD-CONF on learning the causal graph structure of a gene regulatory network from real-world gene expression data with genetic interventions. Specifically, we use the PerturbCITE-seq dataset [55], which contains gene expression data from 218,331 melanoma cells across three conditions: (i)

Figure 3: Performance comparison between DCCD-CONF and the baselines and d is varied between 10 and 80.

<!-- image -->

Table 1: Results on Perturb-CITE-seq [55] gene perturbation dataset. The table presents the average Negative Log-Likelihood (NLL) on the test set, averaged over multiple trials (standard deviation is reported within parentheses).

| Method                    | Control                                                 | Co-Culture                                              | IFN - γ                                                 |
|---------------------------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| DCCD-CONF NODAGS LLC DCDI | 1.375 (0.103) 1.465 (0.015) 1.385 (0.039) 1.523 (0.036) | 1.245 (0.039) 1.406 (0.012) 1.325 (0.029) 1.367 (0.018) | 1.235 (0.338) 1.504 (0.009) 1.430 (0.048) 1.517 (0.041) |

control, (ii) co-culture, and (iii) IFNγ . Due to computational constraints, we restrict our analysis to a subset of 61 genes from the 20,000 genes in the genome, following the experimental setup

of Sethuraman et al. [27] (see Appendix B for details). Each cell condition is treated as a separate dataset consisting of single-node interventions on the selected 61 genes.

Since the dataset does not provide a ground truth causal graph, SHD cannot be used for direct performance comparison. Instead, we assess DCCD-CONF and the baselines based on predictive performance over unseen interventions. To evaluate performance, we split each dataset 90-10, using the smaller portion as the test set, and measure performance using negative log-likelihood (NLL) on the test data after model training (lower the better). The results are presented in Table 1. From Table 1, we can see that DCCD-CONF outperforms all the baselines across all the three cell conditions, showcasing the efficacy of the model and prevalence of confounders in real-world systems. Additionally, we also report the performance of DCCD-CONF and the baselines with respect to MAE on the test data error metric in Table 3 in Appendix C with two additional baselines: DCDFG [47] and Bicycle [56].

Additional experiments. Additionally, we also provide results in Appendix C for the following settings: (i) performance comparison on non-contractive SEMs when the underlying graph is restricted to DAGs, (ii) performance comparison as a function of training data size, (iii) performance comparison as a function of noise variance, (iv) performance comparison as a function outgoing edge density, and (v) performance comparison between DCCD-CONF and additional baselines: JCI-FCI and LiNGAM-MMI.

## 5 Discussion

In this work, we introduced DCCD-CONF, a novel differentiable causal discovery framework that handles directed cycles and unmeasured confounders, assuming Gaussian exogenous noise. It models causal mechanisms via neural networks and learns the causal graph structure by maximizing penalized data likelihood. We provide consistency guarantees in the large-sample regime and demonstrate, through extensive synthetic and real-world experiments, that DCCD-CONF outperforms state-of-the-art methods, maintaining robustness with increasing confounders and graph size. On the Perturb-CITE-seq dataset, our model achieves superior predictive accuracy.

While the focus of this work is limited to Gaussian exogenous noise, we plan to investigate other noise distributions for future research. Other future directions include supporting missing data, and relaxing interventional assumptions by incorporating soft interventions and unknown interventional targets.

## Acknowledgment

This material is based upon work supported by the National Science Foundation under Grant No. CCF-2007807 and 2502298.

## References

- [1] Karen Sachs, Omar Perez, Dana Pe'er, Douglas A. Lauffenburger, and Garry P. Nolan. Causal protein-signaling networks derived from multiparameter single-cell data. Science , 308(5721): 523-529, 2005.
- [2] Eran Segal, Dana Pe'er, Aviv Regev, Daphne Koller, Nir Friedman, and Tommi Jaakkola. Learning module networks. Journal of Machine Learning Research , 6(4), 2005.
- [3] Bin Zhang, Chris Gaiteri, Liviu-Gabriel Bodea, Zhi Wang, Joshua McElwee, Alexei A. Podtelezhnikov, Chunsheng Zhang, Tao Xie, Linh Tran, and Radu Dobrin. Integrated systems approach identifies genetic nodes and networks in late-onset Alzheimer's disease. Cell , 153(3):707-720, 2013.
- [4] Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. Causation, prediction, and search . MIT press, 2000.
- [5] Sofia Triantafillou and Ioannis Tsamardinos. Constraint-based causal discovery from multiple interventions over overlapping variable sets. The Journal of Machine Learning Research , 16(1): 2147-2205, 2015.

- [6] Christina Heinze-Deml, Jonas Peters, and Nicolai Meinshausen. Invariant causal prediction for nonlinear models. Journal of Causal Inference , 6(2), 2018.
- [7] Christopher Meek. Graphical Models: Selecting causal and statistical models . PhD thesis, Carnegie Mellon University, 1997.
- [8] Alain Hauser and Peter Bühlmann. Characterization and greedy learning of interventional markov equivalence classes of directed acyclic graphs. The Journal of Machine Learning Research , 13(1):2409-2464, 2012.
- [9] Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. DAGs with NO TEARS: Continuous optimization for structure learning. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31, 2018. URL https://proceedings.neurips.cc/paper/2018/ file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf .
- [10] Yue Yu, Jie Chen, Tian Gao, and Mo Yu. DAG-GNN: DAG structure learning with graph neural networks. In International Conference on Machine Learning , pages 7154-7163. PMLR, 2019.
- [11] Ignavier Ng, AmirEmad Ghassami, and Kun Zhang. On the role of sparsity and DAG constraints for learning linear dags. Advances in Neural Information Processing Systems , 33:17943-17954, 2020.
- [12] Ignavier Ng, Shengyu Zhu, Zhuangyan Fang, Haoyang Li, Zhitang Chen, and Jun Wang. Masked gradient-based causal structure learning. In Proceedings of the 2022 SIAM International Conference on Data Mining (SDM) , pages 424-432. SIAM, 2022.
- [13] Xun Zheng, Chen Dan, Bryon Aragam, Pradeep Ravikumar, and Eric Xing. Learning sparse nonparametric DAGs. In Silvia Chiappa and Roberto Calandra, editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics , volume 108, pages 3414-3425, 26-28 Aug 2020.
- [14] Hao-Chih Lee, Matteo Danieletto, Riccardo Miotto, Sarah T Cherng, and Joel T Dudley. Scaling structural learning with NO-BEARS to infer causal transcriptome networks. In Pacific Symposium on Biocomputing 2020 , pages 391-402. World Scientific, 2019.
- [15] Philippe Brouillard, Sébastien Lachapelle, Alexandre Lacoste, Simon Lacoste-Julien, and Alexandre Drouin. Differentiable causal discovery from interventional data. Advances in Neural Information Processing Systems , 33:21865-21877, 2020.
- [16] Ioannis Tsamardinos, Laura E Brown, and Constantin F Aliferis. The max-min hill-climbing bayesian network structure learning algorithm. Machine learning , 65(1):31-78, 2006.
- [17] Liam Solus, Yuhao Wang, Lenka Matejovicova, and Caroline Uhler. Consistency guarantees for permutation-based causal inference algorithms. arXiv preprint arXiv:1702.03530 , 2017.
- [18] Yuhao Wang, Liam Solus, Karren Yang, and Caroline Uhler. Permutation-based causal inference algorithms with interventions. Advances in Neural Information Processing Systems , 30, 2017.
- [19] Jacob W. Freimer, Oren Shaked, Sahin Naqvi, Nasa Sinnott-Armstrong, Arwa Kathiria, Christian M. Garrido, Amy F. Chen, Jessica T. Cortez, William J. Greenleaf, Jonathan K. Pritchard, and Alexander Marson. Systematic discovery and perturbation of regulatory genes in human T cells reveals the architecture of immune networks. Nature Genetics , pages 1-12, July 2022. ISSN 1546-1718. doi: 10.1038/s41588-022-01106-y. URL https://www.nature.com/articles/s41588-022-01106-y .
- [20] Thomas Richardson. A discovery algorithm for directed cyclic graphs. In Proceedings of the Twelfth international conference on Uncertainty in artificial intelligence , pages 454-461, 1996.
- [21] Gustavo Lacerda, Peter Spirtes, Joseph Ramsey, and Patrik O. Hoyer. Discovering cyclic causal models by independent components analysis. In Proceedings of the Twenty-Fourth Conference on Uncertainty in Artificial Intelligence , UAI'08, page 366-374, Arlington, Virginia, USA, 2008. AUAI Press. ISBN 0974903949.
- [22] Jan-Christian Huetter and Philippe Rigollet. Estimation rates for sparse linear cyclic causal models. In Jonas Peters and David Sontag, editors, Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence (UAI) , volume 124 of Proceedings of Machine Learning Research , pages 1169-1178. PMLR, 03-06 Aug 2020. URL https://proceedings.mlr. press/v124/huetter20a.html .

- [23] Carlos Améndola, Philipp Dettling, Mathias Drton, Federica Onori, and Jun Wu. Structure learning for cyclic linear causal models. In Conference on Uncertainty in Artificial Intelligence , pages 999-1008. PMLR, 2020.
- [24] Joris M Mooij and Tom Heskes. Cyclic causal discovery from continuous equilibrium data. In Uncertainty in Artificial Intelligence , 2013.
- [25] Mathias Drton, Christopher Fox, and Y. Samuel Wang. Computation of maximum likelihood estimates in cyclic structural equation models. The Annals of Statistics , 47(2):663 - 690, 2019. doi: 10.1214/17-AOS1602. URL https://doi.org/10.1214/17-AOS1602 .
- [26] Antti Hyttinen, Frederick Eberhardt, and Patrik O Hoyer. Learning linear cyclic causal models with latent variables. The Journal of Machine Learning Research , 13(1):3387-3439, 2012.
- [27] Muralikrishnna G Sethuraman, Romain Lopez, Rahul Mohan, Faramarz Fekri, Tommaso Biancalani, and Jan-Christian Hütter. Nodags-flow: Nonlinear cyclic causal structure learning. In International Conference on Artificial Intelligence and Statistics , pages 6371-6387. PMLR, 2023.
- [28] Peter Spirtes. An anytime algorithm for causal inference. In Thomas S. Richardson and Tommi S. Jaakkola, editors, Proceedings of the Eighth International Workshop on Artificial Intelligence and Statistics , volume R3 of Proceedings of Machine Learning Research , pages 278-285. PMLR, 04-07 Jan 2001. URL https://proceedings.mlr.press/r3/spirtes01a.html . Reissued by PMLR on 31 March 2021.
- [29] Joris M Mooij, Sara Magliacane, and Tom Claassen. Joint causal inference from multiple contexts. Journal of machine learning research , 21(99):1-108, 2020.
- [30] Murat Kocaoglu, Amin Jaber, Karthikeyan Shanmugam, and Elias Bareinboim. Characterization and learning of causal graphs with latent variables from soft interventions. Advances in Neural Information Processing Systems , 32, 2019.
- [31] Amin Jaber, Murat Kocaoglu, Karthikeyan Shanmugam, and Elias Bareinboim. Causal discovery from soft interventions with unknown targets: Characterization and learning. Advances in neural information processing systems , 33:9551-9561, 2020.
- [32] Patrick Forré and Joris M Mooij. Constraint-based causal discovery for non-linear structural causal models with cycles and latent confounders. arXiv preprint arXiv:1807.03024 , 2018.
- [33] Joe Suzuki and Tian-Le Yang. Generalization of lingam that allows confounding. In 2024 IEEE International Symposium on Information Theory (ISIT) , pages 3540-3545, 2024. doi: 10.1109/ISIT57864.2024.10619691.
- [34] Shohei Shimizu, Patrik O Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan. A linear non-gaussian acyclic model for causal discovery. Journal of Machine Learning Research , 7(10), 2006.
- [35] Petr Ryšav` y, Pavel Rytíˇ r, Xiaoyu He, Georgios Korpas, and Jakub Mareˇ cek. Exmag: Learning of maximally ancestral graphs. arXiv preprint arXiv:2503.08245 , 2025.
- [36] Rui Chen, Sanjeeb Dash, and Tian Gao. Integer programming for causal structure learning in the presence of latent variables. In International Conference on Machine Learning , pages 1550-1560. PMLR, 2021.
- [37] Rohit Bhattacharya, Tushar Nagarajan, Daniel Malinsky, and Ilya Shpitser. Differentiable causal discovery under unmeasured confounding. In International Conference on Artificial Intelligence and Statistics , pages 2314-2322. PMLR, 2021.
- [38] Alberto Abadie, Anish Agarwal, Raaz Dwivedi, and Abhin Shah. Doubly robust inference in causal latent factor models. arXiv preprint arXiv:2402.11652 , 2024.
- [39] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and causal parameters. Technical report, 2024.
- [40] Kenneth A Bollen. Structural equations with latent variables , volume 210. John Wiley &amp; Sons, 1989.
- [41] Judea Pearl. Causality . Cambridge University Press, 2 edition, 2009. doi: 10.1017/ CBO9780511803161.

- [42] Stephan Bongers, Patrick Forré, Jonas Peters, and Joris M Mooij. Foundations of structural causal models with cycles and latent variables. The Annals of Statistics , 49(5):2885-2915, 2021.
- [43] Jonas Peters and Peter Bühlmann. Identifiability of gaussian structural equation models with equal error variances. Biometrika , 101(1):219-228, 2014.
- [44] Cheng Lu, Jianfei Chen, Chongxuan Li, Qiuhao Wang, and Jun Zhu. Implicit normalizing flows. In International Conference on Learning Representations , 2021. URL https://openreview. net/forum?id=8PS8m9oYtNy .
- [45] Charles G Broyden. A class of methods for solving nonlinear simultaneous equations. Mathematics of computation , 19(92):577-593, 1965.
- [46] Jens Behrmann, Will Grathwohl, Ricky TQ Chen, David Duvenaud, and Jörn-Henrik Jacobsen. Invertible residual networks. In International Conference on Machine Learning , pages 573-582. PMLR, 2019.
- [47] Romain Lopez, Jan-Christian Hütter, Jonathan K Pritchard, and Aviv Regev. Large-scale differentiable causal discovery of factor graphs. In Advances in Neural Information Processing Systems , 2022.
- [48] Brian C. Hall. Lie Groups, Lie Algebras, and Representations , pages 333-366. Springer New York, New York, NY, 2013. ISBN 978-1-4614-7116-5. doi: 10.1007/978-1-4614-7116-5\_16. URL https://doi.org/10.1007/978-1-4614-7116-5\_16 .
- [49] Ricky TQ Chen, Jens Behrmann, David K Duvenaud, and Jörn-Henrik Jacobsen. Residual flows for invertible generative modeling. Advances in Neural Information Processing Systems , 32, 2019.
- [50] Michael F Hutchinson. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics-Simulation and Computation , 18(3):10591076, 1989.
- [51] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations , 2017. URL https://openreview. net/forum?id=rkE3y85ee .
- [52] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Sparse inverse covariance estimation with the graphical lasso. Biostatistics , 9(3):432-441, 2008.
- [53] Michael I Jordan. An introduction to probabilistic graphical models, 2003.
- [54] Kevin Bello, Bryon Aragam, and Pradeep Ravikumar. Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization. Advances in Neural Information Processing Systems , 35:8226-8239, 2022.
- [55] Chris J Frangieh, Johannes C Melms, Pratiksha I Thakore, Kathryn R Geiger-Schuller, Patricia Ho, Adrienne M Luoma, Brian Cleary, Livnat Jerby-Arnon, Shruti Malu, Michael S Cuoco, et al. Multimodal pooled Perturb-CITE-seq screens in patient models define mechanisms of cancer immune evasion. Nature genetics , 53(3):332-341, 2021.
- [56] Martin Rohbeck, Brian Clarke, Katharina Mikulik, Alexandra Pettet, Oliver Stegle, and Kai Ueltzhöffer. Bicycle: Intervention-based causal discovery with cycles. In Causal Learning and Reasoning , pages 209-242. PMLR, 2024.
- [57] Patrick Forré and Joris M Mooij. Markov properties for graphical models with cycles and latent variables. arXiv preprint arXiv:1710.08775 , 2017.
- [58] Peter L Spirtes. Directed cyclic graphical representations of feedback models. arXiv preprint arXiv:1302.4982 , 2013.
- [59] Thomas S. Richardson. A factorization criterion for acyclic directed mixed graphs. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence , UAI '09, page 462-470, Arlington, Virginia, USA, 2009. AUAI Press. ISBN 9780974903958.
- [60] Steffen L Lauritzen and Finn Verner Jensen. Local computation with valuations from a commutative semigroup. Annals of Mathematics and Artificial Intelligence , 21:51-69, 1997.
- [61] Peter Spirtes and Thomas S. Richardson. A polynomial time algorithm for determining dag equivalence in the presence of latent variables and selection bias. In David Madigan and Padhraic Smyth, editors, Proceedings of the Sixth International Workshop on Artificial Intelligence and

Statistics , volume R1 of Proceedings of Machine Learning Research , pages 489-500. PMLR, 0407 Jan 1997. URL https://proceedings.mlr.press/r1/spirtes97b.html . Reissued by PMLR on 30 March 2021.

- [62] R. Ayesha Ali, Thomas S. Richardson, and Peter Spirtes. Markov equivalence for ancestral graphs. The Annals of Statistics , 37(5B), October 2009. ISSN 0090-5364. doi: 10.1214/ 08-aos626. URL http://dx.doi.org/10.1214/08-AOS626 .
- [63] Jiji Zhang. Causal reasoning with ancestral graphs. Journal of Machine Learning Research , 9 (7), 2008.
- [64] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015. URL http://arxiv.org/abs/1412.6980 .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claim and contribution of the paper is clearly stated in the abstract and at the end of introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations and potential directions for addressing them in the discussions section.

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

Justification: The full set of assumptions are provided in Appendix A and assumption 1 is stated in Section 2. All the proofs are provided in Appendix A.

## Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: All the details regarding the experimental setup and the model configurations are provided in Appendix B.

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

Justification: The code is provided in the supplementary materials and will be made public upon publication of the paper.

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

Justification: All the details regarding the experimental setup and model configurations are provided in Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The standard deviation over repeated trials are reported in each plot and table.

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

Justification: All the compute resource for the synthetic and real-world data experiments are included in Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This paper conforms with every aspect of NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: We do not expect the paper to have any negative social impact.

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

Justification: This work poses no such risk.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: See Appendix B for details on the baselines.

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

Justification: No new assets are released in this work.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Human subjects were not involved in this work.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Human subjects were not involved in this work.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are not a part of this work.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendices

The appendices are structured as follows: Appendix A presents the theoretical foundations of differentiable cyclic causal discovery in the presence of unmeasured confounders, including the proof of Theorem 2, and a characterization of the equivalence class of DMGs that maximize the score function. Appendix B provides implementation details of DCCD-CONF and the baselines. Finally, Appendix C provides additional experimental results comparing DCCD-CONF with the baselines.

## A Theory

In this section, we establish the theoretical foundations of differentiable cyclic causal discovery in the presence of unmeasured confounders. We begin by reviewing key definitions and results from prior work that are essential for proving Theorem 2, starting with fundamental graph terminology.

## A.1 Preliminaries

Consider a directed mixed graph G = ( V , E , B ) . A path π between nodes i and j is a sequence ( i 0 , ε 1 , i 1 , . . . , ε n , i n ) , where { i 0 , . . . , i n } ⊆ V and { ε 1 , . . . , ε n } ⊆ E ∪ B , with i 0 = i and i n = j . A path is directed if each edge ε k follows the form i k -1 → i k for all k ∈ [ n ] . A cycle through node i consists of a directed path from i to some node j and an additional edge j → i . For any node i ∈ V , the ancestor set is defined as an G ( i ) := { j ∈ V | a directed path from j to i exists in G} , while the descendant set is given by de G ( i ) := { j ∈ V | a directed path from i to j exists in G} . The spouse set of a node i is defined as sp G ( i ) := { j ∈ V | j ↔ i ∈ B} . If i is both a spouse and an ancestor of j , this creates a almost directed cycle . A mixed graph is called ancestral if it contains neither a directed or an almost directed cycle. The strongly connected component of i, denoted sc G ( i ) , is the intersection of its ancestors and descendants: sc G ( i ) = an G ( i ) ∩ de G ( i ) . The district of a node i ∈ V is defined as dis G ( i ) = { j | j ↔··· ↔ i ∈ G or i = j } . We can apply these definitions to subsets U ⊆ V by taking union of over the items of the subset, for instance, an G ( U ) = ∪ i ∈U an G ( i ) . Avertex set A ⊆ V is said to be barren if i ∈ A has no descendants in G that are in A , however, i may have descendants in G not in A , that is, barren G ( A ) = { i | i ∈ A ; de G ( i ) ∩ A = { i }} . A subset A ⊆ V is ancestrally closed if A contains all of its ancestors. We define A ( G ) := {A | an G ( A ) = A} as the set of ancestrally closed sets in G .

̸

Definition A.1 (Collider) . For a directed mixed graph G = ( V , E , B ) , a node i k ∈ V in a path π = ( i 0 , ε 1 , i 1 , ε 2 , . . . , i n -1 , ε n , i n ) is called a collider if k = 0 , n (non-endpoint) and the two edges ε k , ε k +1 have their heads pointed at i , i.e., the subpath ( i k -1 , ε k , i k , ε k +1 , i k +1 ) is of the form i k -1 → i k ← i k +1 , i k -1 ↔ i k ← i k +1 , i k -1 → i k ↔ i k +1 , i k -1 ↔ i k ↔ i k +1 . The node i k is called a non-collider if i k is not a collider.

Note that the end points of a walk are always non-colliders. We now define the notion of d -separation extended to DMGs.

Definition A.2 ( d -separation) . Let G = ( V , E , B ) be a directed mixed graph and let C ⊆ V be a subset of nodes. A path π = ( i 0 , ε 1 , i 1 , ε 2 , . . . , i n -1 , ε n , i n ) is said to be d -blocked given C if

1. π contains a collider i k / ∈ an G ( C )
2. π contains a non-collider i k ∈ C .

The path π is said to be d -open given C if it is not d -blocked. Two subsets of nodes A,B ⊆ V is said to be d -separated given C if all paths between a and b , where a ∈ A and b ∈ B , is d -blocked given C , and is denoted by

<!-- formula-not-decoded -->

If the underlying graph is acyclic, d -separation implies conditional independence. That is, for subsets of nodes A,B,C ⊆ V ,

<!-- formula-not-decoded -->

where ⊥ p G denotes conditional independence, and p G denotes the observational distribution. This is known as the directed global Markov property of G [57]. However, in general, cyclic graphs do

Figure 4: (Left) Illustration of a directed mixed graph that disobeys directed global Markov property. (Right) The graph on the right represents the graph G after the acyclification process.

<!-- image -->

not obey the directed global Markov property as shown by the counterexample below taken from [42, 58].

Example A.3. Consider the SEM given by:

<!-- formula-not-decoded -->

and p Z is the standard normal distribution. One can check that X 1 is not independent of X 2 given { X 3 , X 4 } . However, the X 1 and X 2 are d -separated given { X 3 , X 4 } in the graph corresponding to the SEM (see Figure 4).

Forré and Mooij [57] introduced σ -separation as a generalization of d-separation to extend the directed global Markov property to cyclic graphs. This concept was motivated by applying d -separation to the acyclified version of the DMG. Before delving into σ -separation, we first define the acyclification procedure of a directed mixed graph, following [42].

Definition A.4 (Acyclification of a directed mixed graph) . Let G = ( V , E , B ) denote a directed mixed graph, the acyclification of G maps G to the acyclified graph acy( G ) = ( V , ˆ E , ˆ B ) , where j → i ∈ ˆ E if and only if j ∈ pa G (sc G ( i )) \ sc G ( i ) , and i ↔ j ∈ ˆ B if and only if there exists i ′ ∈ sc G ( i ) and j ′ ∈ sc G ( j ) such that i ′ = j ′ or i ′ ↔ j ′ ∈ B .

It is important to note that the existence of a acylified graph for an SEM relies on the solvability of the SEM over all the strongly connected components of the DMG corresponding to the SEM. This is to say that we have a solution for X sc G ( i ) given X pa(sc G ( i )) \ sc G ( i ) and Z sc G ( i ) . This is indeed the case as we assume that the forward map f x is invertible for the all the SEMs under consideration, see Bongers et al. [42] for more details. Figure 4 illustrates the acyclification process for the graph corresponding to Example A.3.

Definition A.5 ( σ -separation) . Let G = ( V , E , B ) be a directed mixed graph and let C ⊆ V be a subset of nodes. A path π = ( i 0 , ε 1 , i 1 , ε 2 , . . . , i n -1 , ε n , i n ) is said to be σ -blocked given C if

1. the first node of π , i 0 ∈ C or its last node i n ∈ C , or
2. π contains a collider i k / ∈ an G ( C )
3. π contains a non-collider i k ∈ C that points towards a neighbor that is not in the same strongly connected component as i k in G , i.e, such that i k -1 ← i k in π and i k -1 / ∈ sc G ( i k ) , or i k → i k +1 in π and i k +1 / ∈ sc G ( i k ) .

The path π is said to be σ -open given C if it is not σ -blocked. Two subsets of nodes A,B ⊆ V is said to be σ -separated given C if all paths between a and b , where a ∈ A and b ∈ B , is σ -blocked given C , and is denoted by

<!-- formula-not-decoded -->

Note that σ -separation reduces to d -separation for acyclic graphs, that is, when sc G ( i ) = { i } for all i ∈ V . The following result in [57] relates σ -separation and d -separation.

Proposition A.6 ([57]) . Let G = ( V , E , B ) be a directed mixed graph, then for A,B,C ⊆ V ,

<!-- formula-not-decoded -->

Figure 5: Illustration of the augmented graph G I corresponding to the set of interventional targets I = {∅ , { X 3 } , { X 4 }} . do( { X 3 } ) and do( { X 3 } ) corresponds to the graph obtained after hard interventions on X 3 and X 4 respectively. The augmented graph here is the union of the graphs G , do( { X 3 } ) , do( { X 4 } ) along with the context variables.

<!-- image -->

Using σ -separation we can now define the general directed global Markov property.

Definition A.7 (General directed global Markov property [57]) . Let G = ( V , E , B ) be a directed mixed graph and p G denote the probability density of the observations X . The probability density p G satisfies the general directed global Markov property if for A,B,C ⊆ V

<!-- formula-not-decoded -->

that is, X A and X B are conditionally independent given X C .

## A.2 Joint Causal Modelling and Markov properties

In order to incorporate multiple interventional settings into a single causal modeling framework, we follow the joint causal model introduced by [29], where we augment the system with a set of context variables C I = ( C 1 , . . . , C K ) each corresponding to a non-empty interventional setting. In this case, C k = ∅ for all k = 1 , . . . , K corresponds to the observational setting. We construct an augmented graph, denoted by G I consisting of both the system variables X and the context variables C I , such that the ch G ( C k ) = I k , and no context variable has any parent or a spouse. Figure 5 illustrates the augmented graph for the graph from Example A.3 and the intervention sets I = {∅ , { X 3 } , { X 4 }} . The new system containing both the observed variables and the context variables is called the meta system . Finally, given a family of interventional targets I = { I k } K k =1 and the corresponding context variable C k , the structural equations of the meta systems governing the observations X and C I has the following form:

̸

<!-- formula-not-decoded -->

We call the distribution over the context variables p ( C I ) the context distributions and as noted by Mooij et al. [29], the behavior of the system is usually invariant to the context distribution. We assume access to the context distribution as the interventional settings are known apriori. Note that, the observational distribution corresponds to p G I ( X | C 1 = · · · = C K = ∅ ) . Similarly, the interventional distribution for the interventional setting I k corresponds to p G I ( X | C k = ξ I k , C -k = ∅ ) , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore,

Recall that for the interventional setting I k , the probability density function governing the observations X is given by (6), which we repeat here for convenience

<!-- formula-not-decoded -->

̸

Definition A.8. Let G = ( V , E , B ) be a directed mixed graph, and I = { I k } K k =0 with I 0 = ∅ be a family of interventional targets. Let M I ( G ) denote the set of positive densities p G I : R 2 d → R such that p G I is given by (18) for all F : R 2 d → R d , with F i ( X , Z ) = F i ( X pa G ( i ) , Z i ) , such that the resulting forward map f x is unique and invertible, and Σ Z ≻ 0 and ( Σ Z ) ij = 0 if and only if i ↔ j ∈ B .

Proposition A.9. For a directed mixed graph G = ( V , E , B ) and a family of interventional targets I = { I k } K k =0 such that I 0 = ∅ , let p ∈ M I ( G ) , then p satisfies the general directed global Markov property relative to G I .

Proof. For an DMG G and a choice of F : R 2 d → R d such that f x is unique and invertible and Σ ≻ 0 , the structural equations are uniquely solvable with respect to each strongly connected component of G . Morover, the addition of context variables in the augmented graph does not introduce any new cycles. Therefore the meta system forms a simple SCM. Thus, from Theorem A.21 in [42], the distribution p G I is unique and it satisfies the general directed global Markov property.

We now define the notion of interventional Markov equivalence class for DMGs based on the set of distribution induced by them.

Definition A.10 ( I -Markov Equivalence Class) . Two directed mixed graphs G 1 and G 2 are I -Markov equivalent if and only if M I ( G 1 ) = M I ( G 2 ) , denoted as G 1 ≡ I G 2 . The set of all directed mixed graphs that are I -Markov equivalent to G 1 is the I -Markov equivalence class of G 1 , denoted as I -MEC ( G 1 ) .

From Proposition A.6, for a DMG G and a family of interventional targets I = { I k } K k =0 , any σ -separation statement in G I translates to a d -separation statement in the acyclified graph acy( G I ) . Consequently, the acyclified graph acy( G I ) is equivalent to the augmented graph G I . Furthermore, by the results of [59], p G I admits a factorization, as formalized in the theorem below.

Theorem A.11 ([59]) . A probability distribution p obeys the directed Markov property for an acyclic directed mixed graph G if and only if for every A ∈ A ( G ) ,

<!-- formula-not-decoded -->

where [ A ] G denotes a partition of A into sets { H 1 , . . . , H k } .

Each term in the factorization above is of the form p ( X H | X T ) , H,T ⊆ V , and H ∩ T = ∅ . Following Richardson [59], Lauritzen and Jensen [60] we refer to H as the head of the term p ( X H | X T ) , and T as the tail . An ordered pair of sets ( H,T ) form the head and tail of the factor associated with G if and only if all of the following conditions hold:

1. H = barren G ( an G ( H )) ,
2. If every nodes h ∈ H is connected via a path in the graph obtained by removing all the directed edges in the graph G when restricted to the nodes an G ( H ) , and
3. T = ( disan G ( H ) \ H ) ∪ pa G ( disan G ( H ) ) .

Proposition A.12. Let G = ( V , E , B ) be a directed mixed graph and I = { I k } K k =0 be a family of interventional targets. The set of interventional distributions p G I ∈ M I ( G ) if and only if p G I admits a factorization of the form given by (19) .

Proof. Since any p G I ∈ M I ( G ) is also Markov to acy( G I ) , the proposition above is a direct implication of applying of Theorem 19 on acy( G I ) .

## A.3 Proof of Theorem 2

We now present the main result of this paper. Recall the score function introduced in Section 3.2,

<!-- formula-not-decoded -->

where p ( k ) is the data-generating distribution for I k ∈ I , and ϕ = { θ , Σ Z } represents the set of all model parameters. In the context of the meta system, since we assume access to the context distribution, the score function above is equivalent to the following score:

<!-- formula-not-decoded -->

where p G I ( X , C | ϕ ) is given by (18) for a specific choice of ϕ , and p ∗ I denotes the joint groundtruth distribution for the observed and the context variables. We define P I ( G ) as the set of all distributions p G I ( X , C | ϕ ) that can be expressed by the model specified by equations (5) and (8). That is,

<!-- formula-not-decoded -->

From the above definition it is clear that P I ( G ) ⊆ M I ( G ) . Theorem 2 relies on the following set of assumptions. The first one ensures that the model is capable of representing the ground truth distribution.

Assumption A.13 (Sufficient Capacity) . The joint ground truth distribution p ∗ I is such that p ∗ I ∈ P I ( G ∗ ) , where G ∗ is the ground truth graph.

In other words, there exists a ϕ such that p ∗ I = p G I ( · | ϕ ) . The second assumption generalizes the notion of faithfulness assumption to the interventional setting.

Assumption A.14 ( I -σ -faithfulness) . Let V = ( X , C I ) , for any subset of nodes A,B,C ⊆ V∪ C I , and I k ∈ I

<!-- formula-not-decoded -->

The above assumption implies that any conditional independency observed in the data must imply a σ -separation in the corresponding interventional ground truth graph.

Assumption A.15 (Finite differential entropy) . For I = { I k } K k =0 ,

<!-- formula-not-decoded -->

The above assumption ensures that the hypothetical scenario where S ( G ∗ ) and S ( G ) are both infinity is avoided. This is formalized in the lemma below taken from [15].

Lemma A.16 (Finiteness of the score function [15]) . Under assumptions A.13 and A.15, |S I ( G ) | &lt; ∞ .

From the results of [15], we can now express the difference in score function between G ∗ and G as the minimization of KL diverengence plus the difference in the regularization terms.

Lemma A.17 (Rewritting the score function [15]) . Under assumptions A.13 and A.15, we have

<!-- formula-not-decoded -->

We will now prove the following technical lemma (adapted from [15]) which we will be used in proving Theorem 2.

Lemma A.18. Let G = ( V , E , B ) be a directed mixed graph, for a set of interventional targets I = { I k } K k =0 , and p ∗ / ∈ M I ( G )) , then

<!-- formula-not-decoded -->

Proof. Let V = ( X , C I ) , from theorem A.11, any p ∈ M I ( G )) admits a factorization of the form

<!-- formula-not-decoded -->

Let us define a new distribution ˆ p as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

This proves the lemma.

We are now ready to prove Theorem 2. Recall,

Theorem 2. Let I = { I k } K k =0 be a family of interventional targets, let G ∗ denote the ground truth directed mixed graph, p ( k ) denote the data generating distribution for I k , and ˆ G := arg max G S ( G ) . Then, under the Assumptions 1, A.13, A.14, and A.15, and for a suitably chosen λ &gt; 0 , we have that ˆ G ≡ I G ∗ . That is, ˆ G is I -Markov equivalent to G ∗ .

Proof. It is sufficient to show that for G / ∈ I -MEC ( G ∗ ) , the score function of ˆ G is strictly lower than the score function of G ∗ , i.e., S ( G ∗ ) &gt; S ( G ) . Since G / ∈ I -MEC ( G ∗ ) and p ∗ I ∈ M I ( G ∗ ) (by Assumption A.13), there must exist subsets of nodes A,B,C ⊆ V ∪ C I such that either:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or

If no such subsets exist, then G and G ∗ impose the same σ -separation constraints and thus induce the same set of distributions. This would imply that G ∈ I -MEC ( G ∗ ) , contradicting our assumption.

̸

̸

From proposition A.12, we see that ˆ p ∈ M I ( G )) and hence p = ˆ p . We will show that

<!-- formula-not-decoded -->

For an arbitrary p ∈ M I ( G )) , consider the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the equation above, we leverage the linearity of expectation, which holds under Assumption A.15, ensuring that we don't sum infinities of opposite signs. We now show that each term in the right hand side of the above equality is an expectation of KL divergence which is always in [0 , ∞ ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, E p ∗ log ˆ p ( V ) p ( V ) ∈ [0 , ∞ ) .

We now show that ˆ p = arg min p ∈M (do( I k )( G )) D KL ( p ∗ ∥ p ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the expectations in (26) are both in [0 , ∞ ) , splitting the expectation is valid. The very last inequality holds since p ∗ = ˆ p . Thus,

<!-- formula-not-decoded -->

Since p ∗ I ∈ M I ( G ∗ )) , it must be true that V A ̸⊥ p ( k ) V B | V C (Assumption A.14). Therefore p I ∗ doesn't satisfy the general directed Markov property with respect to G I and hence p ∗ I / ∈ M I ( G ) .

For convenience, let

Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use Lemma A.18 for the final inequality. Thus, from Lemma A.17

<!-- formula-not-decoded -->

Following [15], we now show that by choosing λ sufficiently small, the above equation is stictly positive. Note that if |G| ≥ |G ∗ | then S ( G ∗ ) -S ( G ) &gt; 0 . Let G + := {G | |G| &lt; |G ∗ |} . Choosing λ such that 0 &lt; λ &lt; min G∈ G + η ( G ) |G ∗ |-|G| we see that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, every graph outside of the general directed Markov equivalence class of ( G ∗ ) I has a strictly lower score.

## A.4 Characterization of Equivalence Class

Let G = ( V , E , B ) be a directed mixed graph, and consider a family of interventional targets I = I k K k =0 with I 0 = ∅ . From Proposition A.6, for any graph G 1 ∈ I -MEC ( G ) , the corresponding augmented graph G I 1 is equivalent to the acyclification acy( G I ) of G I . Several prior works have studied the characterization of equivalence classes of acyclic directed mixed graphs (ADMGs), including [61, 30, 62]. We now provide a graphical notion of the I -Markov equivalence class of a DMG G . A graph G is said to be maximal if there exists no inducing path (relative to the empty set) between any two non-adjacent nodes. An inducing path relative to a subset L is a path on which every non-endpoint node i / ∈ L is a collider on the path and every collider is an ancestor of an endpoint of the path. A Maximal Ancestral Graph (MAG) is one that is both ancestral and maximal. Given an ADMG acy( G I ) , it is possible to construct a MAG over the variable set V = ( X , C I ) that preserves both the independence structure and ancestral relationships encoded in acy( G I ) ; see [63] for details. We denote MAG (acy( G I )) to mean MAG that is constructed from acy( G I ) . Therefore all the independencies encoded in acy( G I ) is also present in MAG (acy( G I )) . Before present the condition for MAG equivalence, we introduce the notion of discriminating path. A path π = ( i 0 , ε 1 , . . . , i n -1 , ε n , i n ) in acy( G I ) is called a discriminating path for i n -1 if (1) π includes at lest three edges; (2) i n -1 is a non-endpoint node on π , and is adjacent to i n on π ; and (3) i 0 and i n are not adjacent, and every node in between i 0 and i n -1 is a collider on π and is a parent of i n . The following theorem from Spirtes and Richardson [61] characterizes the equivalence of MAGs.

Theorem A.19 (Spirtes and Richardson [61]) . Two MAGs G 1 and G 2 are Markov equivalent if and only if:

1. G 1 and G 2 have the same skeleton;
2. G 1 and G 2 have the same unshielded colliders; and
3. if π forms a discriminating path for i in G 1 and G 2 , then i is a collider on π if and only it is a collider on π in G 2 .

Therefore, from Proposition A.6 and Theorem A.19, two DMGs G 1 and G 2 are equivalent if and only if MAG (acy( G I 1 )) and MAG (acy( G I 2 )) satisfying the conditions of Theorem 2, i.e., MAG (acy( G I 1 )) and MAG (acy( G I 2 )) : (i) have the same skeleton, (ii) same unshielded colliders, and (iii) same discriminating paths with consistent colliders.

## B Implementation Details

In this section we provide the implementation details of DCCD-CONF and the baseline models along with the details of the experimental setup.

## B.1 Hutchinson trace estimator for computing log determinant of the Jacobian

Computing the log determinant of the Jacobian matrix present in (6) poses a significant challenge. However, following Behrmann et al. [46], in Section 3.3.1 showed that the log-determinant of the Jacobian can be estimated using the following estimator

<!-- formula-not-decoded -->

The estimator above still has a major drawback: computing the Tr( J U k g ) still requires O ( d 2 ) gradient calls to compute exactly. Fortunately, Hutchinson trace estimator [50] can be used to stochastically approximate the trace of the Jacobian matrix. This then results in the following estimator that can be computed efficiently via reverse-mode automatic differentiation

<!-- formula-not-decoded -->

## B.2 Parameter update via score maximization

As described in Section 3, the model parameters are updated in two stages. In the first stage, the parameters of the neural networks and the Gumbel-Softmax distribution, used to sample adjacency matrices, are updated via backpropagation using stochastic gradient descent. Since (5) forms an implicit block of an implicit normalizing flow, following [44], we directly estimate the gradients of ˆ S ( B ) with respect to x and ϕ = ( θ , B ) . The gradient computation involves two terms: ∂ ∂ ( · ) log det( I + J g x ( x , ϕ ))

and ∂ ˆ S ∂ z ∂ z ∂ ( · ) , where ( · ) is a placeholder for x and ϕ . From [44, 46], we use the following unbiased estimators for the gradients:

<!-- formula-not-decoded -->

On the other hand, ∂ ˆ S ( B ) ∂ z ∂ z ∂ ( · ) can be computed according to the implicit function theorem as follows:

<!-- formula-not-decoded -->

where G z ( z ) = g z ( z , ϕ ) + z , and recall that G ( x , z , ϕ ) = g x ( x , ϕ ) + x + g z ( z , ϕ ) + z . See [44] for more details. The procedure SGUPDATE shown in Algorithm 1 performs the gradient computation in (35) and (36).

In the second stage, the entries of the covariance matrix of the endogenous noise distribution are updated column-wise by solving a sequence of Lasso optimization problems. The complete parameter update procedure is summarized in Algorithm 1.

## B.3 DCCD-CONF and the baselines code details

DCCD-CONF. We implemented our framework using the libraries Pytorch and Scikit-learn in Python and the code used in running the experiments can be found in the following Github repository: https://github.com/muralikgs/dccd\_conf .

## Algorithm 1 PARAMETER UPDATE

```
Require: Family of interventional targets I = { I k } K k =1 , interventional dataset { x ( i,k ) } N k ,K i =1 ,k =1 , regularization coefficients λ and ρ . Ensure: Learned neural network parameters ˆ θ , graph structure parameters ˆ B , confounder-noise distribution parameters ˆ Σ Z . 1: Initialize the parameters: θ (0) ∼ p θ ( θ ) , B (0) ∼ p B ( B ) , and Σ Z = I 2: Iteration counter: t = 0 3: while NOT CONVERGED do 4: for k = 1 to K do 5: t ← t +1 6: W ← ( Σ ( t ) Z ) U k , U k 7: Compute score function ˜ L ( B ( t ) , θ ( t ) , W , I k ) 8: B ( t +1) , θ ( t +1) ← SGUPDATE ( ˜ L , B ( t ) , θ ( t ) ) 9: for j = 1 to d do 10: Push j -th row and column in W to the end 11: β ← lasso ( W 11 , s 12 , ρ ) 12: w 12 ← W 11 β 13: end for 14: ( Σ ( t +1) Z ) U k , U k ← W 15: end for 16: end while 17: ˆ θ , ˆ B , ˆ Σ Z ← θ ( t ) , B ( t ) , Σ ( t ) Z return ˆ θ , ˆ B , ˆ Σ Z
```

Starting with an initialization of the model parameters ( θ (0) , B (0) , Σ (0) Z ) , we iteratively alternate between maximizing the score function with respect to ( θ ( t ) , B ( t ) ) and Σ ( t ) Z , as described in Algorithm 1. Standard stochastic gradient updates are used for ( θ ( t ) , B ( t ) ) , while coordinate gradient descent, implemented via the Scikit-learn library, is applied to Σ ( t ) Z . For modeling the causal function g x , we follow the setup of Sethuraman et al. [27], employing neural networks (NNs) with dependency masks parameterized by a Gumbel-softmax distribution. The log-determinant of the Jacobian is computed using a power series expansion combined with the Hutchinson trace estimator. To mitigate bias from truncating the power series expansion, the number of terms is sampled from a Poisson distribution, as detailed in Section 3.3 and Appendix B.1. The final objective is optimized using the Adam optimizer [64].

The learning rate in all our experiments was set to 10 -2 . The neural network models used in our experiments contained one multi-layer perceptron layer. No nonlinearities were added to the neural networks for the linear SEM experiments. We used tanh activation for the nonlinear SEM experiments and for the experiments on the perturb-CITE-seq data set. The graph sparsity regularization constant λ was set to 10 -2 for all the experiments. The sparsity inducing regularization constant for the inverse covariance matrix of the confounder distribution, ρ , was set to 10 -1 in all the experiments. The models were trained and evaluated on NVIDIA RTX6000 GPUs.

Baselines. For NODAGS-Flow, we used the code provided by authors [27] available at https: //github.com/Genentech/nodags-flows . The default values were set for the hyperparameters. We implemented the LLC algorithm based on the details provided in [26]. The implementation can be found within the codes/baselines folder in the supplementary materials. For FCI, we used the implementation that is available in the causallearn python library ( https://github. com/py-why/causal-learn ). For DCDI, we used the codebase provided by the authors [15], available at https://github.com/slachapelle/dcdi . The default hyperparameters were used while training and evaluating the model. For DAGMA and ADMG, we used the codebase provided by the authors [54] and [37], available at https://github.com/kevinsbello/dagma and https: //gitlab.com/rbhatta8/dcd respectively. We implemented LiNGAM-MMI based on the details provided by the authors [33].

## B.4 Experimental setup

In this section, we describe how the data sets were generated for the various experiments conducted.

## B.4.1 Synthetic Experiments

We begin by sampling a directed graph using the Erd˝ os-Rényi (ER) random graph model with an edge density of 2 unless specified otherwise, which determines the directed edges in the DMG G . Next, we generate a random matrix and project it onto the space of positive definite matrices to obtain the confounder covariance matrix Σ Z , setting the maximum exogenous noise standard deviation to 0.5 unless specified otherwise. The nonzero off-diagonal entries of Σ Z correspond to the bidirectional edges in G . For the linear SEM, edge weights are sampled uniformly from Unif (( -0 . 9 , -0 . 2) ∪ (0 . 2 , 0 . 9)) . In all experiments except those on non-contractive SEMs, the edge weight matrix is rescaled to ensure a Lipschitz constant of less than one. For nonlinear SEMs, we apply a tanh nonlinearity to the linear system defined by the edge weights, i.e.,

<!-- formula-not-decoded -->

where W is the weighted adjacency matrix. In all the experiments, the training data consisted of 500 samples per interventional setting (unless specified otherwise).

Impact of Confounder Count In this experiment, the number of observed nodes in the graph is fixed at d = 10 . Training data consists of combination of observational data and single-node interventions over all nodes, i.e., I = ∅ ∪ {{ i } | i ∈ [ d ] } . The confounder ratio (number of confounders divided by the number of nodes) is varied from 0.2 to 0.8.

Impact of Cycles We fix the number of nodes in the graph d = 10 , The number of cycles in the graph is varied between 0 and 8, with the confounder ratio set to 0.3. The training data consists of observational data as well as single node interventions over all the nodes in the graph.

Impact of Nonlinearity In this setting, the degree of nonlinearity (controlled by β ) is varied between 0 and 1. That is, the data is generated from the following SEM:

<!-- formula-not-decoded -->

Here, β = 0 implies the SEM is purely linear and β = 1 . 0 implies the data is purely nonlinear. The confounder ratio is set to 0.3. The number of cycles is randomly set.

Scaling with Number of Nodes We fix the confounder ratio at 0.4. The total number of nodes in the graph is varied from 10 to 80. As in the previous setup, training data consists of combination of observational data and single-node interventions across all nodes, with 500 samples per interventional setting.

Scaling with Interventions We fix d = 10 and set the confounder ratio to 0.4. The number of interventions during training varies from 0 to d . Zero interventions corresponds to observational data. When fewer than d interventions are provided, the intervened nodes are selected arbitrarily. Each interventional setting consists of 500 samples.

Non-Contractive SEM In this case, we explicitly enforce a non-contractive causal mechanism F by rescaling edge weights to ensure that the Lipschitz constant of the edge weight matrix exceeds one. We set d = 10 and provide observational data and single-node interventions across all nodes, with 500 samples per intervention. The confounder ratio varies between 0.2 and 0.8.

Scaling with Training Samples To examine the sample requirements of DCCD-CONF, we set the confounder ratio to 0.4. Training data consists of observational data and single-node interventions over all nodes, while the number of samples per intervention is varied from 500 to 2500.

Scaling with outgoing edge density In this case, the outgoing edge density of the ER random graphs is varied from 1 to 4. The confounder ratio is set to 0.4 and the number of nodes d = 10 . The training data consists of observational data and single node experiments over all the nodes in the graph.

Scaling with noise standard deviation In this setting, we vary the maximum noise standard deviation between 0.2 and 0.8. The confounder ratio is set to 0.4 and the number of nodes d = 10 . The training data consists of observational data and single node experiments over all the nodes in the graph.

Evaluation Metrics Across all experiments, we use Structural Hamming Distance (SHD) to evaluate the accuracy of the estimated directed edges relative to the ground truth. SHD measures the number of modifications (edge additions, reversals, and deletions) required to match the estimated graph to the ground truth. For DCCD-CONF and NODAGS-Flow we fix a threshold value of 0.8 for the estimated adjacency matrix. The recovery of bidirectional edges is assessed using the F1 score, which is defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and TP, FP, FN denote true positives, false positives, and false negatives, respectively. We use a threshold of 0 . 01 for the estimated covariance matrix to identify the bidirectional edges.

Additionally, we also measure the performance of DCCD-CONF and the baselines using Area Under Precision-Recall Curve (AUPRC) as the error metric. AUPRC computes the area under the precision-recall curve evaluated at various threshold values (the higher the better).

## B.4.2 Gene Perturbation Data set

The dataset was obtained from the Single Cell Portal of the Broad Institute (accession code SCP1064). Following the experimental setup of Sethuraman et al. [27], we filtered out cells with fewer than 500 expressed genes and removed genes expressed in fewer than 500 cells. Due to computational constraints, we selected a subset of 61 perturbed genes (Table 2) from the full genome. The three experimental conditions-co-culture, IFNγ , and control-were partitioned into separate datasets, and models were trained and evaluated on each condition independently.

Table 2: The list of chosen genes from Perturb-CITE-seq dataset [55].

| ACSL3   | ACTA2   | B2M     | CCND1   | CD274   | CD58     | CD59   | CDK4   | CDK6   |
|---------|---------|---------|---------|---------|----------|--------|--------|--------|
| CDKN1A  | CKS1B   | CST3    | CTPS1   | DNMT1   | EIF3K    | EVA1A  | FKBP4  | FOS    |
| GSEC    | GSN     | HASPIN  | HLA-A   | HLA-B   | HLA-C    | HLA-E  | IFNGR1 | IFNGR2 |
| ILF2    | IRF3    | JAK1    | JAK2    | LAMP2   | LGALS3   | MRPL47 | MYC    | P2RX4  |
| PABPC1  | PAICS   | PET100  | PTMA    | PUF60   | RNASEH2A | RRS1   | SAT1   | SEC11C |
| SINHCAF | SMAD4   | SOX4    | SP100   | SSR2    | STAT1    | STOM   | TGFB1  | TIMP2  |
| TM4SF1  | TMED10  | TMEM173 | TOP1MT  | TPRKB   | TXNDC17  | VDAC2  |        |        |

## C Additional Results

## C.1 Additional synthetic experiments

Experiments on Non-contractive DAGs. We evaluated the performance of DCCD-CONF and baseline methods on non-contractive SEMs, where the ground truth DMG is acyclic. While DCCDCONF assumes a contractive causal mechanism, we adapt it for non-contractive settings using the preconditioning trick proposed by Sethuraman et al. [27]. This approach introduces a learnable diagonal preconditioning matrix Λ , transforming the causal mechanism as follows:

<!-- formula-not-decoded -->

where g x , as defined in (8), remains contractive (see Sethuraman et al. [27] for details). We vary the confounder ratio, and the results are summarized in Figure 6a. As shown in Figure, DCCD-CONF effectively learns the ADMG even in non-contractive SEM settings, demonstrating competitive performance against the baselines.

where

Figure 6: Performance comparison between DCCD-CONF and baseline methods on causal graph recovery and confounder identification, evaluated across varying model parameters. Each subplot details a specific experimental setting.

<!-- image -->

Performance comparison vs. sample size We also assess the sample requirements of DCCDCONF. Figure 6b summarizes the results obtained by varying the number of samples per intervention. As shown in the figure, when the confounder ratio is 0.4, DCCD-CONF achieves low SHD even with 500 samples per intervention and attains near-perfect accuracy from 2000 samples onward. However, performance declines slightly as the confounder ratio increases.

Performance comparison vs. max endogenous noise st. deviation In this setting, we compare DCCD-CONF with the baseline by varying the maximum standard deviation of the endogenous noise terms between 0.2 and 0.8, the results are summarized in Figure 6c. DCCD-CONF outperforms the baselines for all noise standard deviations. However, the performance of the models does deteriorate slightly as the noise standard deviation increases.

Performance comparison vs. outgoing edge density In this case, the expected number of outgoing edges from each node is varied between 1 and 4. This affects the sparsity of the resulting graph. The results are summarized in Figure 6d. As seen from Figure 6d, DCCD-CONF still outperforms the baselines, even though the performance of all the models worsens as the edge density increases.

## C.2 Additional performance metrics

In addition to SHD for directed edge recovery and F1 score for bi-directional edge recovery. We also AUPRC to compare DCCD-CONF and the baseline for all of the experimental settings stated in Appendix B.4.1. The results are summarized in Figure 7. Overall, DCCD-CONF performs better than LLC on nonlinear SEMs across all the settings, while achieving perfect AUPRC scores in several cases.

## C.3 Comparison with FCI-JCI

Here, we compare the performance of DCCD-CONF with FCI-JCI [29], which is an extension of FCI algorithm that is capable of handling multiple contexts (in this case interventional settings). FCI-JCI outputs a Partial Ancestral Graph (PAG), which is a graph structure that represents the equivalence class of MAGs. We define a modified SHD score in order to check if the DMG estimated by DCCD-CONF belongs to the same equivalence class of the ground truth DMG. To that end, we convert the ground-truth DMG and the estimated DMG to their augmented DMGs and then construct the MAG of the acyclified version of the augmented DMGs. The modified SHD score then computes the discrepancies in the conditions of Theorem A.19, i.e., we count: (i) the

Figure 7: Comparison of DCCD-CONF and baseline methods on causal graph recovery and confounder identification, measured using AUPRC across varying model parameters.

<!-- image -->

Figure 8: Performance comparison between DCCD-CONF and FCI-JCI with respect to modified SHD as the error metric. The number of observed nodes was set to d = 5 . For the left plot, all single node interventions along with the observational data were provided as training data. For the right plot, observational data and interventions over two of the nodes (randomly chosen) were used as training data.

<!-- image -->

number of extra edges ( N 1 ) using the skeletons of the estimated MAG and ground truth MAG, (2) number of mismatched unshielded colliders ( N 2 ), and (3) discrepancies in the discriminating paths ( N 3 ). Similarly, for FCI-JCI, we count the disagreements between the ground-truth MAG and the estimated PAG, i.e., mismatch in skeleton ( N 1 ) , mismatch in unshielded colliders ( N 2 ), invariant edge orientation discrepancies ( N 3 ). Finally, the modified SHD = N 1 + N 2 + N 3 . We compare DCCD-CONF and JCI-FCI over two different settings: (i) the training data consists of observational data and single node interventions over all the nodes in the graph, and (ii) the training data consists of observational data and interventions over 2 nodes (randomly chosen) in the graph. Due to the

complexity of computing the modified SHD (as it involves iterating over the discriminating paths and inducing paths) we fix the number of observed nodes to be d = 5 . In the both the cases, the confounder ratio is varied between 0.2 and 0.8, and nonlinear SEM is used to generate the data. The results are summarized in Figure 8.

As seen from Figure 8, DCCD-CONF outperforms FCI-JCI in the both the settings. However, the performance does decrease as the number of training interventions reduces. We attribute this to the increase sample requirements as the number of training interventions goes down.

## C.4 Comparison with LiNGAM-MMI

We compare the performance of DCCD-CONF with LiNGAMMMI [33], which extends the ICA-based LiNGAM [34] to handle hidden confounding. Since LiNGAM-MMI requires iterating over all possible node permutations, its computational cost grows rapidly with the number of nodes; hence, we restrict our comparison to d = 5 . The training data include both observational samples and

Figure 9: Performance comparison between DCCDCONF and LiNGAM-MMI on d = 5 node graphs.

<!-- image -->

single-node interventions for all nodes, with 500 samples per interventional setting. We vary the confounder ratio (i.e., the ratio of confounders to observed nodes) between 0.2 and 0.8. As shown in Figure 9, DCCD-CONF consistently outperforms LiNGAM-MMI across all tested confounder ratios.

## C.5 Hyperparameter Sensitivity

We evaluate the sensitivity of DCCD-CONF to its hyperparameters: (i) the directed edge sparsity regularization coefficient λ c , and (ii) the bidirected edge sparsity regularization coefficient ρ . Figure 10 summarizes the results for λ c , ρ ∈ 0 . 1 , 0 . 01 , 0 . 001 on graphs with d = 10 nodes and a confounder ratio of 0.3. The model was trained using both observational data and all single-node interventions. As shown in the figure, DCCD-CONF remains fairly robust to hyperparameter variations, achieving normalized SHD values below 0.3 and F1 scores above 0.65 for most combinations of λ c and ρ .

Figure 10: Illustration of DCCD-CONF performance for various choices of hyperparameters λ c (directed edge sparsity regularization coeff.) and ρ (bidirected edge sparsity regularization coeff.).

<!-- image -->

## C.6 Training Time Comparison

Figure 11 compares the training times of DCCD-CONF and baseline methods. Unlike the other algorithms, LLC does not require stochastic gradient-based training, as it relies solely on solving a series of linear regressions. Consequently, LLC is considerably faster, as shown in Figure 11. Excluding LLC, NODAGS-Flow is the most efficient in terms of runtime; however, it cannot account for confounders within its framework. DCCD-CONF achieves training times comparable to ADMG and DCDI while simultaneously handling confounders, cycles, and nonlinear dependencies. The training times are computed on d = 10 node graphs with training dataset consisting of observational and all single-node interventions. DCCD-CONF was trained for 200 epochs. All models were run on RTX6000 GPUs.

## C.7 Additional Results on Perturb-CITE-seq Dataset

In addition to test-set NLL, we evaluate the performance of DCCD-CONF and the baselines on the Perturb-CITE-seq dataset [55] using Interventional Mean Absolute Error (I-MAE) as the evaluation

Figure 11: Training time comparison between DCCD-CONF and the baselines

<!-- image -->

metric. I-MAE is computed as the mean of ∥ x + g x ( x ) ∥ 1 /d over all observations x in the held-out test set. Beyond the baselines discussed in Section 4, we also include Bicycle [56] and DCDFG [47] for comparison. Bicycle supports nonlinear and cyclic structures, whereas DCDFG restricts the search space to acyclic graphs. The results, summarized in Table 3, show that DCCD-CONF remains competitive with state-of-the-art methods under the I-MAE metric.

Table 3: Results on Perturb-CITE-seq [55] gene perturbation dataset. The table presents the average Mean Absolute Error (MAE) on the test set, averaged over multiple trials (standard deviation is reported within paranthesis).

| Method    | Control       | Co-Culture    | IFN - γ       |
|-----------|---------------|---------------|---------------|
| DCCD-CONF | 0.781 (0.037) | 0.765 (0.046) | 0.843 (0.035) |
| NODAGS    | 0.847 (0.018) | 0.762 (0.018) | 0.861 (0.023) |
| Bicycle   | 0.782 (0.042) | 0.735 (0.036) | 0.883 (0.028) |
| DCDFG     | 0.845 (0.066) | 0.774 (0.038) | 0.891 (0.041) |

Additionally, we report the adjacency matrix of the recovered causal graph for the cell condition 'Co-Culture' in Figure 12. DCCD-CONF identified 38 feedback cycles. This number validates prior work showing that gene regulatory networks are rich in feedback loops [55].

Figure 12: Adjacency matrix of learnt by DCCD-CONF for 'Co-Culture' cell condition of PerturbCITE-seq dataset.

<!-- image -->

## C.8 Protein Signaling Dataset

We further evaluate DCCD-CONF on a biological dataset for protein signaling network discovery [1], which is widely used as a benchmark for causal discovery algorithms.

The dataset contains continuous measurements of multiple phosphorylated proteins and phospholipid components in human immune system cells, with the corresponding network capturing the ordering of interactions among pathway components. Based on n = 7466 samples across m = 11 cell types, Sachs et al. [1] identified 20 edges in the underlying graph. Using the consensus network from [1] as ground truth, we evaluate performance using the Structural Hamming Distance (SHD) as the error metric. The results, summarized in Table 4, show that DCCD-CONF performs comparably to the baselines. The recovered directed graph is visualized in Figure 13.

Table 4: Performance comparison on Sachs et al. [1] protein signaling dataset.

| Method       |   SHD |
|--------------|-------|
| DCCD-CONF    |    18 |
| DAG-GNN [10] |    19 |
| DAGMA        |    21 |
| NOTEARS      |    22 |

Figure 13: (Left) Consensus graph from Sachs et al. [1], (right) graph learned by DCCD-CONF.

<!-- image -->
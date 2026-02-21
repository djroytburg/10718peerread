## Diffusion Network Inference for Cross-layer Cascades

## Siyu Huang ˚

Department of Statistics The Pennsylvania State University sph6006@psu.edu

Yubai Yuan ˚:

Department of Statistics The Pennsylvania State University yvy5509@psu.edu

## Abdul Basit Adeel

Department of Sociology and Criminology The Pennsylvania State University axa6372@psu.edu

## Abstract

A cascade over a network refers to the diffusion process where behavior changes occurring in one part of an interconnected population lead to a series of sequential changes throughout the entire population. In recent years, there has been a surge in interest and efforts to understand and model cascade mechanisms since they motivate many significant research topics across different disciplines. The propagation structure of cascades is governed by underlying diffusion networks that are often hidden. Inferring diffusion networks thus enables interventions in cascading process to maximize information propagation and provides insights into the Granger causality of interaction mechanisms among individuals. In this project, we propose a novel double network mixture model for inferring latent diffusion network in presence of strong cascade heterogeneity. The new model represents cascade pathways as a distributional mixture over diffusion networks that capture different cascading patterns at the population level. We develop a data-driven optimization method to infer diffusion networks using only visible temporal cascade records, avoiding the need to model complex and heterogeneous individual states. Both statistical and computational guarantees are established for the proposed method. We apply the proposed model to analyze research topic cascades in social sciences across U.S. universities and uncover the latent research topic diffusion network among top U.S. social science programs.

## 1 Introduction

Cascades over network refer to the diffusion processes where behavior changes in a part of an interconnected population lead to a series of sequential changes throughout the entire population. In recent years, there are surging interests and efforts to understand and model the cascade mechanism since it motivates many significant research topics in different areas, including social influence [12, 13, 20], information propagation via social media [32, 1], diffusion of policy and social norms [4, 33], viral marketing [25, 9], and contagion of infectious diseases [21, 35].

One fundamental problem is to understand the diffusion networks that govern cascade propagation patterns. However, diffusion networks are often hidden and need to be inferred from observed cascading behaviors. For example, in the case of infectious diseases, we can observe when an individual is infected but need to impute the missing information on who infects this individual.

˚ Co-first authors.

: Corresponding author.

More importantly, real-world cascading behaviors often exhibit strong heterogeneity and are jointly governed by different diffusion patterns. For example, in information cascade and epidemiology, cascades can diffuse among population via different transmission channels building on various social relations, and thus lead to heterogeneous propagation speeds and scales [24, 30]. Furthermore, cascading heterogeneity originates from the variability of individuals' statuses in engaging the cascade [39, 11]. In social media like X and Instagram, the spreading pattern and the speed of messages heavily depend on users' activity status. An active user will respond more instantly to interesting messages and accelerate the information cascade compared to inactive users [10]. As for volatility cascades in financial markets, structure heterogeneity in volatility diffusion depends on different time horizons of the agents in the market [41]. In these scenarios, individual statuses determine the transmission channels engaging in the cascades, and changes of individual statuses can also change the downstream cascade diffusion patterns. To conclude from these examples, diffusion patterns can exhibit combinatorial complexity as population grows. We present an example of cascade diffusion when transmission channels of individuals vary in Figure (1).

Figure 1: A cascade diffuses through nodes t A,B,C,D,E,F u on a two-layer network. Yellow nodes: activated nodes; gray nodes: inactivated nodes. t B,D u are activated via the observed network and t E,C,F u are activated via the latent network.

<!-- image -->

In this paper, we propose a novel double network mixture model to infer multiple diffusion networks simultaneously from heterogeneous cascade data. The proposed model introduces a distributional mixture of diffusion networks to capture the heterogeneous cascading patterns, where diffusion networks provide complementary connection information. The main advantage of the distributional mixture is to avoid modeling the complex individual status changes. Specifically, the proposed model can describe the diffusion process over multi-layer networks where cascades propagate across different layers alternatively. Compared to existing methods, the proposed method can uncover latent diffusion networks even when the number of diffusion patterns is exponential to the number of nodes. Furthermore, the parameter estimation in our model can be solved by a sequence of convex optimization problems, which leads to both statistical and computational guarantees for our diffusion network estimation.

## 2 Related works

Various directed probabilistic graphical models have been developed to infer diffusion networks from observed cascade samples [17, 19, 31, 8, 22, 18]. Generally, these models treat infection time as a continuous random variable and construct the likelihood of cascade samples based on the target diffusion network under local Markov assumption. To capture heterogeneous diffusion patterns, several multi-pattern cascade models have been proposed [37, 40], where cascade samples are adaptively clustered into groups and each group corresponds to a distinct diffusion network. Among these methods, ConNIe [29], NetRate [31], and MMRate [37] are popular representatives. Specifically, ConNIe employs a maximum likelihood formulation via convex programming, incorporating an l 1 -type penalty to promote sparsity in the inferred network. Building upon ConNIe, NetRate explicitly represents diffusion as a continuous-time probabilistic process, characterized by edge-specific transmission rates governing edge-wise diffusion probabilities. MMRate further extends this framework by accommodating multiple distinct diffusion patterns, assuming multiple heterogeneous latent networks, with each cascade diffusing via one network according to a certain probability.

## 3 Methodology

## 3.1 Continuous-time cascade on single network

Consider a network with N nodes where each node has two infection conditions in a single cascade: infected (activated) and uninfected (inactivated), and an infected node will always remain infected. A cascade is a N -dimensional temporal record t ' p t 1 , t 2 , ¨ ¨ ¨ , t N q , where t i is the infection time of the i -th node. Instead of infinite time horizon, we observe a cascade within a finite time window of length T , i.e., @ i, t i P r t 0 , t 0 ` T s , where t 0 : ' min 1 ď i ď N t t i u is the infection time of source node. We denote the infection time of the nodes not infected in the observation window as t 0 ` T . Without loss of generality, we assume t 0 ' 0 in this paper.

A cascade diffuses nodewisely over edges of a diffusion network. The continuous-time model formulates the cascade transmission from an infected node j to another infected node i with survival analysis models. Specifically, given a node j being infected at time t j , let f p t i | t j , λ ji q denote the likelihood of node i being infected by node j at time t i , where t i ě t j . The transmission rate λ ji represents how fast cascades diffuse from node j to node i . Accordingly, the cumulative probability function is F p t i | t j , λ ji q ' ş t i t j f p s | t j , λ ji q ds . We consider the hazard rate function defined as

<!-- formula-not-decoded -->

where S p t i | t j , λ ji q ' 1 ´ F p t i | t j , λ ji q is the survival function that indicates the probability of node i not being infected by node j until time t i . Typical parametric forms of hazard rate functions are Exp model: H p t i | t j , λ ji q ' λ ji ; Pow model: H p t i | t j , λ ji q ' λ ji 1 t i ´ t j ; Ray model: H p t i | t j , λ ji q ' λ ji p t i ´ t j q . Given that λ ji controls the likelihood and speed of transmission between node j to node i , the global cascading pattern over network can be captured by matrix Λ ' p λ ij q P R N ˆ N ` , where rows represent sender nodes and columns represent receiver nodes. Note that Λ can be asymmetric, i.e., λ ij ‰ λ ji , when node i and j are different in the capacity of launching transmission.

The diffusion process is typically modeled by independent cascade models [22], where any infected nodes can infect a node independently and a node stays infected once another node infects it. Therefore, one can formulate the likelihood of node i being infected by potential parent nodes t j : t j ă t i u at time t i as

<!-- formula-not-decoded -->

On the other hand, if node i is not activated by any parent nodes in the observation window, then the corresponding likelihood for t i : t i ' T u is

<!-- formula-not-decoded -->

Since infections of each node are conditionally independent given the corresponding parent nodes, we can decompose the likelihood of a cascade t as the multiplication of a series of conditional probabilities:

<!-- formula-not-decoded -->

Therefore, the full likelihood of independent cascade samples t t p c q u C c ' 1 is ś C c ' 1 P p t p c q ; Θ q . The advantage of the above model is that each cascade sample induces a directed acyclic graph, whose local Markov property allows the decomposition of likelihood and thus a reduction in computational complexity of inferring the diffusion network Θ .

## 3.2 Double network mixture model

In many applications such as social media, there exist multiple networks among an interconnected population that reflect different types of relations. Cascade diffusion pattern can change over different networks, and cascade can proceed on different networks alternatively and simultaneously due to the inter-layer interactions. The diffusion process on multi-layer networks can greatly increase the degree of freedom in possible diffusion pathways, which leads to strong heterogeneity in cascade observations. To model the multi-network cascade behavior, we propose the double network mixture model . Consider two diffusion networks Θ and Ψ among the same population with N units, where Θ ij and Ψ ij are the transmission rate from node i to node j corresponding to two relations. We introduce double diffusion indicators Z c i P t 0 , 1 u such that

<!-- formula-not-decoded -->

For cascade c , node i is activated via diffusion pathway on network Θ if Z c i ' 1 , and Z c i ' 0 if via network Ψ . Denote π ' t π i u N i ' 1 P r 0 , 1 s N where π i is the probability of node i engaging cascade via network Θ . Then the diffusion pathway of the cascade c can be represented as

<!-- formula-not-decoded -->

Therefore, Θ c is column-wise mixture of Θ and Ψ , which can vary for different cascades. Different to conventional mixture model, the proposed method allows Z c i to vary across different cascade samples and nodes. Therefore, the proposed model (3) can generate up to 2 N different types of diffusion patterns, which grows exponentially as network size increases. Therefore, the proposed method has the principled heterogeneity modeling for the diffusion patterns. The likelihood of the proposed double mixture model can be also explicitly formulated. Following P I and P U in (1), we have the probability for node i being infected in the c -th cascade given specific network as:

<!-- formula-not-decoded -->

and following (2) the probability for node i not being infected is

<!-- formula-not-decoded -->

Denote Z ' t Z c i u N,C i ' 1 ,c ' 1 , the joint distribution of diffusion indicators is

<!-- formula-not-decoded -->

Then we have the joint distribution of cascade samples and diffusion indicators Z as

<!-- formula-not-decoded -->

where Ω ' p π , Θ , Ψ q denote model parameters. And the marginal distribution of cascade samples is

<!-- formula-not-decoded -->

Another advantage of our model is that the posterior distribution of diffusion indicator t Z c i u can be calculated in an explicit form

<!-- formula-not-decoded -->

## 3.3 Model identification

In this subsection, we establish the identifiability of the double network mixture model. Denote R Θ i : ' t j P t 1 , ¨ ¨ ¨ , N u | Θ ji ą 0 u and R Ψ i : ' t j P t 1 , ¨ ¨ ¨ , N u | Ψ ji ą 0 u as the sets of nodes that can directly reach node i on Θ and Ψ , and t R i P r 0 , T s | R Θ i Y R Ψ i | : ' t t j | j P R Θ i Y R Ψ i u as the infectious times of these nodes. We have the following identifiability result for model in (9).

Proposition 1. For each node i, i ' 1 , ¨ ¨ ¨ , N , assume that 1) } Θ ¨ i } 1 `} Ψ ¨ i } 1 ą 0 and there exists j ‰ i such that Θ ji ‰ Ψ ji , and 2) survival function satisfies S p t i | t j , λ ji q ' exp t λ ji h p t i ´ t j qu for some differentiable function h p¨q . Then, the parameters p π i , Θ ¨ i , Ψ ¨ i q associated with node i in (9) are identifiable, i.e., if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption 2) can be satisfied by popular risk models including Exp model, Pow model, Ray model, and other additive risk models of information propagation, such as kernel hazard functions [10] and feature-enhanced hazard functions [36]. Proposition 1 shows that the network preferences t π i u N i ' 1 are identifiable and diffusion networks Θ and Ψ are column-wise identifiable. However, similar to the labeling non-identifiability issue in finite mixture model [23], Θ and Ψ may still not be globally identifiable without structural constraints due to column permutation. Specifically, the data distribution does not change if we swap Θ ¨ i and Ψ ¨ i in Θ and Ψ with other columns fixed.

Layer-specific network structure constraint Motivated by real-world applications, one can interpret Θ as the diffusion pathways over an observed social network A P t 0 , 1 u N ˆ N . Therefore, we can impose support constraints on Θ as

<!-- formula-not-decoded -->

Due to the sparse nature of real-world social networks, the support constraint also implicitly imposes sparsity constraint on Θ . On the other hand, one can interpret Ψ as the diffusion pathways via latent social relations of individuals that are not captured by the social network A . The magnitude of Ψ ij reflects the social distance between individual i and j in terms of their latent factors. It has been found that social distance typically has or can be approximated by low-rank structure [34, 26, 27], since high-dimensional social factors can always be embeded into a low dimensional latent space that preserves social distances [34]. Therefore, we impose low-rank structure on Ψ as

<!-- formula-not-decoded -->

Imposing the above support constraint and low-rank structure allows Θ and Ψ to capture complementary diffusion patterns driven by different types of relations. In addition, the structure constraints solve the above column-wise permutation issue [6]. Specifically, we introduce the matrix subspace Λ 1 p Θ q ' ␣ N P R N ˆ N | support p N q Ď A ( . We also perform SVD on Ψ ' U Σ V J where U , V P R N ˆ r and r is the rank of Ψ . Then, we define another matrix subspace as Λ 2 p Ψ q ' ␣ U X J ` Y V J | X,Y P R N ˆ k ( . We have the following result:

Proposition 2. Given that the assumptions in Proposition 1 hold, then Θ and Ψ are identifiable if:

<!-- formula-not-decoded -->

where } ¨ } 2 and } ¨ } 8 denote matrix operation norm and largest element in magnitude.

The first term in (23) controls the rank of Θ given a fixed sparsity level, where a larger value indicates a lower rank. The second term controls the sparsity level of Ψ given a fixed rank, where a larger value indicates a lower sparsity level. Intuitively, networks Θ and Ψ can be globally identified given that they are well-separated in terms of either rank or sparsity.

## 3.4 Model estimation

Combining the distribution of cascade samples in Section 3.2 and the network structure constraints in Section 3.3, we estimate the model parameters Ω via constrained likelihood maximization as

<!-- formula-not-decoded -->

where I is a N -byN matrix with all elements being 1 . However, both the likelihood function and the rank regularization are difficult to directly optimize. Therefore, we maximize the evidence lower bound of the log-likelihood function and replace the low-rank penalty with its convex relaxation as nuclear norm } ¨ } ‹ . The optimization problem can thus be reformulated as follow

<!-- formula-not-decoded -->

where E q p Z q is the expectation of Z over distribution q p Z q . The above objective function can be optimized via EM algorithm by iteratively updating q p Z q and Ω . Specifically, with the estimated parameters Ω p s q from the s -th step:

<!-- formula-not-decoded -->

In E-step, the posterior distribution of network indicators ˆ π c i ' P p Z c i | t p c q ; Ω p s q q can be explicitly updated via ( 10 ). In M-step, the objective function Q p Ω | Ω p s q q : ' 1 C ř C c ' 1 E q p Z ; Ω p s q q ' log P p t p c q , Z ; Ω q ‰ can be decomposed as Q p Ω | Ω p s q q ' Q 1 p Θ | Ω p s q q ` Q 2 p Ψ | Ω p s q q` Q 3 p π | Ω p s q q . The arguments Θ , Ψ , and π can thus be updated parallelly in M-step as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When the structural network A is not observed, one can impose l 1 -norm penalty to pursue sparsity structure in Θ . Accordingly, the constraint in optimization (12) is replaced by } Θ } 1 ď s 1 , where s 1 ą 0 is the sparsity tuning parameter. The main advantage of the proposed relaxation is that the M-step becomes a series of convex optimization problems. Specifically, denote parameter spaces Ξ Θ p s q : ' t Θ P r 0 , β 1 s N ˆ N | Θ dp I ´ A q ' 0 u with s ' } A } 0 , Ξ Ψ p ρ q : ' t Ψ P r 0 , β 2 s N ˆ N | } Ψ } ‹ ď ρ u , and Ξ π : ' t π P r ϵ, 1 ´ ϵ s N u , where β 1 , β 2 are nonnegative constants and 0 ă ϵ ă 0 . 5 . We have the following result:

Theorem 3.1. The parameter spaces Ξ Θ p s q , Ξ Ψ p ρ q , and Ξ π are convex sets for any s ą 0 , ρ ą 0 , and Q 3 p π | Ω p s q q is concave on π . If the hazard function H p t | t 1 , λ q satisfies B H 2 p t | t 1 ,λ q B λ 2 ' 0 for t ě t 1 , then Q 1 p Θ | Ω p s q q and Q 2 p Ψ | Ω p s q q are concave on Θ and Ψ , respectively. Furthermore, if for any node i , the probabilities of being source node P p v q ą 0 for v P R where R denotes the set of nodes from which i is reachable via a directed path, then E t r Q 1 p Θ | Ω p s q qs and E t r Q 2 p Ψ | Ω p s q qs are also strictly concave in terms of Θ and Ψ , respectively.

Theorem 3.1 guarantees that the optimization in M-step has a unique and optimal solution. Combining the theorem with the convergence guarantee of EM algorithm for convex ancillary function Q p Ω | Ω p s q q [2, 38], the above likelihood maximizer ˆ Ω is guaranteed to converge to the true Ω . In addition, the convexity assumption on hazard function can be satisfied by popular risk models including Exp model, Pow model, Ray model, and other additive risk models. We summarize and provide details for the above optimization process in the Appendix.

## 4 Numerical experiments on synthetic cascading data

We investigate the performance of the proposed method in recovering the diffusion networks based on synthetic cascading data. The performance of global transmission rates estimation on a network Θ is measured by normalized mean absolute error (MAE) as MAE p Θ q ' ř i,j | ˆ Θ ij ´ Θ ij |{ Θ ij . The

performance of network probability π estimation is measured as MAE p π q ' ř i | ˆ π i ´ π i |{ π i . In addition, we investigate the performance of recovering the structure of diffusion network. Specifically, given a diffusion network Θ we consider three topology estimation metrics accuracy, precision, and recall as Acc p Θ q ' ř i,j | I p ˆ Θ ij q ´ I p Θ ij q|{p ř i,j I p ˆ Θ ij q ` ř i,j I p Θ ij qq , Pre p Θ q ' ř i,j p I p ˆ Θ ij q ¨ I p Θ ij qq{ ř i,j I p ˆ Θ ij q , and Recall p Θ q ' ř i,j p I p ˆ Θ ij q ¨ I p Θ ij qq{ ř i,j I p Θ ij q , where I p α q ' 1 if α ą 0 and I p α q ' 0 otherwise. In the following numerical experiments, we fix the size of diffusion networks at N ' 200 .

## 4.1 Benchmark comparison under different network topologies

We compare the proposed method with baseline methods including NetRate [31], MMRate [37], and ConNIe [29] on diffusion network recovery. We generate diffusion networks Θ and Ψ to mimic different structure of real-world networks. Specifically, we consider the diffusion network Θ and its support A as random network, network with community structure, and scale-free network. On the other hand, we fix the latent diffusion network to be both low-rank (rank 5) and sparse (edge density 0.05). Given Θ and Ψ , we generate C ' 2 , 000 independent cascade samples based on double mixture model with Exp transmission model and observation window length being T ' 10 .

Table 1 shows the proposed method outperforms both NetRate and MMRate by achieving lower MAE of transmission rate estimation on diffusion network Θ under three network topologies and lower MAE of corresponding latent diffusion network Ψ . MAEcomparison does not include ConNIe since it only estimates network topology. In addition, we compare the proposed method with baseline methods in terms of network topology recovery on both Ψ and joint network ˆ Θ Y ˆ Ψ for a fair comparison since benchmark methods NetRate and ConNIe do not distinguish the different diffusion networks by design. Table 1 shows the proposed method achieves higher accuracy on topology recovery via both latent network ˆ Ψ and joint network ˆ Θ Y ˆ Ψ than NetRate, MMRate, and ConNIe under different underlying structures in Θ . It also shows that under different settings, our method significantly outperforms all other methods in precision while remains higher recall than MMRate and ConNIe, and achieves similar recall to NetRate. Since only MMRate differentiates diffusion networks and estimates probabilities of cascading diffusing over a specific network, we compare MAE p π q from the proposed method with that from MMRate. Table 1 shows that our method also outperforms MMRate in estimation of network selection probability under different network topologies.

Table 1: Diffusion network estimations from different methods under three Θ topologies (standard deviation in parenthesis, best performance highlighted in blue).

<!-- image -->

|         |                | MAE Θ                                  | MAE Ψ                                  | MAE π                                  | Acc Ψ                                  | Pre Ψ                                  | Rec Ψ                                  | Acc Θ Y Ψ                              | Pre Θ Y Ψ                              | Rec Θ Y Ψ                              |
|---------|----------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| OurAlg  | Rand Com Scale | 0.315(0.011) 0.264(0.005) 0.312(0.006) | 0.544(0.007) 0.507(0.004) 0.545(0.006) | 0.047(0.002) 0.049(0.002) 0.050(0.002) | 0.842(0.006) 0.846(0.005) 0.833(0.005) | 0.757(0.008) 0.767(0.007) 0.750(0.007) | 0.948(0.006) 0.942(0.004) 0.937(0.005) | 0.889(0.006) 0.888(0.004) 0.883(0.004) | 0.829(0.006) 0.832(0.006) 0.826(0.007) | 0.958(0.004) 0.954(0.003) 0.949(0.003) |
| NetRate | Rand Com Scale | 0.865(0.001) 0.851(0.000) 0.855(0.000) | 2.998(0.026) 3.103(0.014) 2.897(0.018) | - - -                                  | 0.172(0.002) 0.180(0.002) 0.179(0.002) | 0.095(0.001) 0.099(0.001) 0.099(0.001) | 0.964(0.005) 0.975(0.004) 0.929(0.003) | 0.207(0.003) 0.216(0.002) 0.220(0.002) | 0.116(0.002) 0.122(0.001) 0.125(0.001) | 0.942(0.004) 0.955(0.003) 0.922(0.003) |
| MMRate  | Rand Com Scale | 0.637(0.007) 0.630(0.008) 0.668(0.006) | 1.490(0.030) 1.665(0.036) 1.477(0.027) | 0.888(0.133) 0.881(0.137) 0.928(0.028) | 0.201(0.023) 0.219(0.017) 0.211(0.019) | 0.120(0.015) 0.130(0.012) 0.126(0.013) | 0.632(0.023) 0.705(0.008) 0.650(0.009) | 0.265(0.028) 0.281(0.020) 0.278(0.023) | 0.164(0.019) 0.173(0.015) 0.173(0.018) | 0.695(0.019) 0.755(0.006) 0.710(0.007) |
| ConNIe  | Rand Com       | - -                                    | - -                                    | - -                                    | 0.519(0.006) 0.519(0.005)              | 0.397(0.005) 0.398(0.005)              | 0.752(0.010) 0.748(0.007)              | 0.638(0.006) 0.638(0.005)              | 0.530(0.006) 0.532(0.007)              | 0.800(0.008) 0.796(0.006)              |
| ConNIe  | Scale          | -                                      | -                                      | -                                      | 0.523(0.006)                           | 0.402(0.005)                           | 0.748(0.009)                           | 0.646(0.006)                           | 0.543(0.006)                           | 0.797(0.008)                           |

## 4.2 Network recovery under different transmission models

In this subsection, we investigate the performance of our diffusion network estimation when cascade samples are generated from popular transmission models including Exp, Pow, and Ray models, respectively. Table 2 illustrates the transmission rates recovery and network topology recovery on Ψ under different transmission models and cascade sample sizes C . As the sample size C increases, both the parameter estimations (MAE) and topology recovery metrics (Acc, Pre, Rec) improve under different transmission models. In addition, the degree of improvement decreases as more cascade samples become available. This pattern indicates the consistency of the proposed diffusion network estimators, and the convergence of the proposed EM-type optimization. Notice that the diffusion network recovery based on cascade samples generated from Pow transmission model is better than

Exp and Ray model. This is because the Pow model introduces an additional parameter as time lag lower bound, which lowers the variation of activate time and overestimation of transmission rates.

Table 2: Diffusion network estimations from the proposed method under different cascade models and sample sizes.

|     |                                    | MAE Θ                                               | MAE Ψ                                               | MAE π                                               | Acc Ψ                                               | Pre Ψ                                               | Rec Ψ                                               |
|-----|------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| Exp | C = 500 C = 1000 C = 1500 C = 2000 | 0.320(0.019) 0.235(0.013) 0.216(0.012) 0.187(0.011) | 0.871(0.019) 0.667(0.007) 0.601(0.008) 0.562(0.007) | 0.095(0.006) 0.091(0.005) 0.085(0.004) 0.082(0.004) | 0.564(0.010) 0.743(0.009) 0.817(0.006) 0.857(0.004) | 0.431(0.011) 0.653(0.014) 0.746(0.008) 0.794(0.006) | 0.817(0.010) 0.862(0.007) 0.903(0.006) 0.930(0.005) |
| Ray | C = 500 C = 1000 C = 1500 C = 2000 | 0.287(0.018) 0.211(0.015) 0.179(0.011) 0.188(0.012) | 1.053(0.056) 0.990(0.100) 0.629(0.091) 0.605(0.081) | 0.152(0.008) 0.130(0.008) 0.134(0.004) 0.137(0.004) | 0.605(0.018) 0.712(0.019) 0.795(0.017) 0.807(0.011) | 0.525(0.024) 0.605(0.026) 0.702(0.026) 0.709(0.019) | 0.714(0.013) 0.866(0.012) 0.916(0.007) 0.937(0.011) |
| Pow | C = 500 C = 1000 C = 1500 C = 2000 | 0.171(0.008) 0.129(0.006) 0.112(0.005) 0.106(0.006) | 0.433(0.012) 0.328(0.007) 0.290(0.004) 0.271(0.004) | 0.144(0.010) 0.131(0.007) 0.125(0.008) 0.129(0.011) | 0.882(0.015) 0.956(0.004) 0.969(0.002) 0.973(0.003) | 0.832(0.024) 0.932(0.007) 0.946(0.004) 0.950(0.005) | 0.939(0.009) 0.982(0.004) 0.993(0.002) 0.997(0.001) |

## 5 Real cascading data analysis

In this section, we study the cascading patterns of research topics in sociology by discovering the diffusion networks among US universities. Geographic proximity is known to facilitate idea exchanges through collaborations and citations among colleagues within the same institution or nearby locations [3], which can be due to dependence on shared research resources and the need for coordination [28]. It is also known that research diffusion can happen via the latent network of scholars, which is also sometimes called the "invisible colleges" [7] to highlight the role of informal networks and latent ties. Our goal is to infer the latent research topic diffusion network and compare its difference with the geographic network in terms of topic cascading.

Data preparation and preprocess We select universities in both the list of 1965 ASA Guide to Graduate Departments of Sociology and the list of 2022 US News Best Sociology Programs in America. In addition, we exclude the universities whose sociology programs were established after 1965. Based on the above conditions, we finalize N ' 104 universities, which are considered as the target population of our study. We then create a geographical network A P t 0 , 1 u N ˆ N among the selected universities, where A ij ' 1 if university i and j are located in the same state.

Figure 2: Cascade transmission rates over geographical network Θ and latent diffusion network Ψ .

<!-- image -->

To construct research topics, we use Elsevier's Scopus API to compile a dataset of 29,725 unique articles from 23 top generalist sociology journals, including information on authors, their affiliations, article titles, and abstracts. Multiple keywords are extracted from the titles and abstracts of articles. After these preprocesses, we obtain 3,033 unique research topics that cover major research fields in sociology. For each research topic c , we construct the corresponding cascade samples as p t c 1 , t c 2 , ¨ ¨ ¨ , t c N q where t c i : ' ˜ t c i ´ ˜ t c 0 , with ˜ t c 0 being the publication date of the first article that involves topic c , and ˜ t c i being the publication date of the first article that involves topic c and is published by any scholar affiliated with university i . If university i never publishes any article that involves topic c , we set t c i to be year 2022.

Diffusion networks inference We simultaneously estimate the geographic diffusion network Θ and the latent diffusion network Ψ based on the proposed model. Figure (2) illustrates the inferred diffu-

sion networks ˆ Θ and ˆ Ψ , respectively. The size of nodes represents the out-degree of corresponding universities on networks. We see that the geographic diffusion network has a strong local community structure, especially among universities in California, the Great Lakes, and the northeast coast.

Figure (2) also shows that the latent diffusion network demonstrates a decentralized structure and contains many connections between the east and west coasts. This captures the national collaboration and academic mobility. We also compare estimated pairwise transmission rates on diffusion networks ˆ Θ and ˆ Ψ in Figure (3). In summary, the latent diffusion network is denser than the geographic network, and the magnitude of transmission rates on the geographic network is larger and has more variation compared to the latent diffusion network.

To provide an interpretation of the inferred geographic diffusion network Θ and latent diffusion network Ψ , we connect universities' positions on the network with universities' U.S. News Univer-

Figure 3: Comparison of distributions of pairwise transmission rates from geographic network Θ ij and latent network Ψ ij .

<!-- image -->

sity rankings on the sociology program. We choose these rankings since they are a systematic and popular summary of academic factors. Figure (4a) illustrates the association between program ranking and the node-wise betweenness centrality on latent diffusion network. The Betweenness centrality of a node i is defined as ř j ‰ i ‰ k σ jk p i q{ σ jk , where σ jk is the total number of shortest paths from node j to k , and σ jk p i q is the number of those paths that pass through i . Based on the figure, the universities'

Figure 4: Node-wise centrality vs Ranking on latent network Ψ .

<!-- image -->

betweenness centralities are relatively uniform and do not significantly decrease as ranking increases. This suggests that there does not exist any strong community or centralized topology in the latent network. When universities are grouped according to their ranking (top 20, middle 60, low 20 etc.), each group has different high-betweenness universities, serving as the bridges in idea diffusion. We also investigate the relation between university rankings and Pagerank centrality. Pagerank centrality measures the influence of a node on a network based on how many influential nodes it connects to. Figure (4b) shows that the universities' Pagerank centralities decrease as their rankings increase, which suggests that the idea exchanges among high-ranked universities are more frequent and faster than those of other universities. This pattern appears because high-ranked universities have a higher level of research activities. University prestige is also considered as an important proxy for the quality of ideas, which increases the likelihood of research ideas from higher-ranked universities to diffuse.

## 6 Scalability analysis

The implicit convex nature of our objective function allows the optimization process to be as efficient as gradient descent, even though the optimization of our method is based on EM-type update. The E-step in our algorithm has an explicit form as in (10). Furthermore, we have proved in Theorem

3.1 that the objective function Q p Ω q in the M-step is strictly concave and in each EM-iteration, we only perform gradient ascent once instead of maximizing Q p Ω q until convergence. Under these conditions, established proofs in [2] show that the above EM-update is equivalent to gradient decent, and our optimization thus enjoys gradient descent's geometric convergence rate.

Computational complexity of the proposed method is K ¨ O p N 2 ¨ C q ` K ¨ O p N 3 q , where N is network size, C is sample size, and K is the number of iterations executed. In practice, we require C " N for reasonable estimation. The first term thus dominates and the computational complexity is approximately the same as that of gradient descent. We can replace the standard SVD operation in the M -step by randomized SVD or Lanczos algorithm. Then, we can further reduce the computational complexity of the second term from O p N 3 q to O p rN 2 q or O pp r ` l q N 2 q where r ! N is the rank of network Ψ , N is the size of network, and l is a constant usually between 10 and 20.

We also numerically investigate both time and estimation performance of the proposed method in recovering diffusion networks based on synthetic cascading data in large network settings. We adopt the same criteria used in Section 4 to evaluate estimation performance and use average time per iteration to evaluate time performance. We generate diffusion networks Θ and Ψ of different sizes in a similar way to Section 4.2, fixing Ψ to be both low-rank (rank 5) and sparse (edge density 0.01). Given Θ and Ψ , we generate C ' 50 , 000 independent cascade samples based on double mixture model with Exp transmission model and observation window length being T ' 10 .

Table 3: Execution time of different methods under different network sizes (unit: second per iteration).

|         | N ' 500      | N ' 1000     | N ' 2000      | N ' 4000       |
|---------|--------------|--------------|---------------|----------------|
| OurAlg  | 2.395(0.001) | 8.201(0.019) | 33.442(0.031) | 132.955(0.046) |
| NetRate | 1.390(0.001) | 4.929(0.011) | 20.039(0.030) | 79.992(0.040)  |

Table 3 shows that the computation time of the proposed method per iteration is proportional to the squared network size N 2 . This aligns with the earlier theoretical analysis of computational complexity. In addition, for all network sizes N ' 500 , 1000 , 2000 , 4000 , the ratio between computational time per iteration of the proposed method and NetRate [31]is approximately a constant 1.65. Since NetRate [31] is well-known for its scalability, this result demonstrates the computational efficiency and scalability of the proposed method.

Table 4 illustrates the estimation performance of the proposed method for large networks. For networks of all sizes N ' 500 , 1000 , 2000 , the proposed method outperforms NetRate [31] on both transmission rate estimation and network topology recovery. For networks of sizes N ' 500 , 1000 , the proposed method achieves estimation performances comparable to those of smaller networks in Section 4. Additionally, when sample size is fixed, the performance of the proposed method decreases, indicating the need for more samples on larger networks to achieve accurate estimations.

Table 4: Diffusion network estimations from the proposed method under different network sizes.

|         |          | MAE Θ        | MAE Ψ        | Acc Ψ        | Pre Ψ        | Rec Ψ        |
|---------|----------|--------------|--------------|--------------|--------------|--------------|
|         | N = 500  | 0.398(0.001) | 0.711(0.019) | 0.894(0.003) | 0.820(0.005) | 0.983(0.001) |
| OurAlg  | N = 1000 | 0.411(0.005) | 0.797(0.007) | 0.779(0.001) | 0.666(0.001) | 0.939(0.003) |
|         | N = 2000 | 0.437(0.003) | 0.814(0.008) | 0.571(0.002) | 0.523(0.002) | 0.630(0.002) |
| NetRate | N = 500  | 0.743(0.014) | 1.053(0.038) | 0.078(0.008) | 0.041(0.004) | 0.818(0.007) |
|         | N = 1000 | 0.755(0.010) | 1.094(0.019) | 0.086(0.001) | 0.045(0.001) | 0.914(0.007) |
|         | N = 2000 | 0.759(0.003) | 1.096(0.002) | 0.083(0.000) | 0.044(0.000) | 0.913(0.000) |

## 7 Conclusion

In this paper, we propose a novel double network mixture model for heterogeneous cascading process on multi-layer networks. Our method can identify the latent diffusion network complementary to the observed network. Due to its convex formulation, our method has both statistical and computational guarantee in terms of estimating diffusion networks. A major future work is to generalize the mixture graph model to a system with more than two-layer networks and derive the model identification conditions. Extending the proposed method to inference on time-varying networks with timedependent parameters t Θ p t q , Ψ p t q , π p t qu or nonparametric transmission models are also interesting directions for future works.

## References

- [1] Eytan Adar and Lada A Adamic. Tracking information epidemics in blogspace. In The 2005 IEEE/WIC/ACM International Conference on Web Intelligence (WI'05) , pages 207-214. IEEE, 2005.
- [2] Sivaraman Balakrishnan, Martin J. Wainwright, and Bin Yu. Statistical guarantees for the EM algorithm: From population to sample-based analysis. The Annals of Statistics , 45(1):77 - 120, 2017.
- [3] Pierre-Alexandre Balland and David Rigby. The geography of complex knowledge. Economic geography , 93(1):1-23, 2017.
- [4] Sushil Bikhchandani, David Hirshleifer, and Ivo Welch. Learning from the behavior of others: Conformity, fads, and informational cascades. Journal of economic perspectives , 12(3):151-170, 1998.
- [5] Jian-Feng Cai, Emmanuel J Candès, and Zuowei Shen. A singular value thresholding algorithm for matrix completion. SIAM Journal on optimization , 20(4):1956-1982, 2010.
- [6] Venkat Chandrasekaran, Sujay Sanghavi, Pablo A Parrilo, and Alan S Willsky. Rank-sparsity incoherence for matrix decomposition. SIAM Journal on Optimization , 21(2):572-596, 2011.
- [7] Diana Crane. Invisible Colleges; Diffusion of Knowledge in Scientific Communities . University of Chicago Press, Chicago' 1972.
- [8] Hadi Daneshmand, Manuel Gomez-Rodriguez, Le Song, and Bernhard Schoelkopf. Estimating diffusion network structures: Recovery conditions, sample complexity &amp; soft-thresholding algorithm. In International conference on machine learning , pages 793-801. PMLR, 2014.
- [9] Nan Du, Le Song, Manuel Gomez Rodriguez, and Hongyuan Zha. Scalable influence estimation in continuous-time diffusion networks. Advances in neural information processing systems , 26, 2013.
- [10] Nan Du, Le Song, Ming Yuan, and Alex Smola. Learning networks of heterogeneous influence. Advances in neural information processing systems , 25, 2012.
- [11] Wenjing Duan, Bin Gu, and Andrew B Whinston. Informational cascades and software adoption on the internet: an empirical investigation. MIS quarterly , pages 23-48, 2009.
- [12] John RP French. The bases of social power. Studies in social power/University of Michigan Press , 1959.
- [13] Noah E Friedkin and Eugene C Johnsen. Social influence network theory: A sociological examination of small group dynamics , volume 33. Cambridge University Press, 2011.
- [14] Jerome Friedman, Trevor Hastie, Holger Höfling, and Robert Tibshirani. Pathwise coordinate optimization. The annals of applied statistics , 1(2):302-332, 2007.
- [15] P. E. Gill, W. Murray, and M. A. Saunders. Snopt: An sqp algorithm for large-scale constrained optimization. SIAM Review , 47:99-131, 2005.
- [16] Philip E. Gill, Walter Murray, Michael A. Saunders, and Elizabeth Wong. User's Guide for SNOPT Version 7.7: Software for Large-Scale Nonlinear Programming . Department of Mathematics, University of California, San Diego, La Jolla, CA 92093-0112, USA, March 2018. CCoM Technical Report 18-1.
- [17] Manuel Gomez-Rodriguez, Jure Leskovec, and Andreas Krause. Inferring networks of diffusion and influence. ACM Transactions on Knowledge Discovery from Data (TKDD) , 5(4):1-37, 2012.
- [18] Manuel Gomez-Rodriguez, Jure Leskovec, and Bernhard Schölkopf. Modeling information propagation with survival theory. In International conference on machine learning , pages 666-674. PMLR, 2013.

- [19] Manuel Gomez Rodriguez, Jure Leskovec, and Bernhard Schölkopf. Structure and dynamics of information pathways in online media. In Proceedings of the sixth ACM international conference on Web search and data mining , pages 23-32, 2013.
- [20] Márton Karsai, Gerardo Iñiguez, Riivo Kikas, Kimmo Kaski, and János Kertész. Local cascades induced global contagion: How heterogeneous thresholds, exogenous effects, and unconcerned behaviour govern online adoption spreading. Scientific reports , 6(1):27178, 2016.
- [21] Matt J Keeling and Ken TD Eames. Networks and epidemic models. Journal of the royal society interface , 2(4):295-307, 2005.
- [22] David Kempe, Jon Kleinberg, and Éva Tardos. Maximizing the spread of influence through a social network. In Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining , pages 137-146, 2003.
- [23] Daeyoung Kim and Bruce G Lindsay. Empirical identifiability in finite mixture models. Annals of the Institute of Statistical Mathematics , 67:745-772, 2015.
- [24] Alden S Klovdahl, John J Potterat, Donald E Woodhouse, John B Muth, Stephen Q Muth, and William W Darrow. Social networks and infectious disease: The colorado springs study. Social science &amp; medicine , 38(1):79-88, 1994.
- [25] Jure Leskovec, Lada A Adamic, and Bernardo A Huberman. The dynamics of viral marketing. ACM Transactions on the Web (TWEB) , 1(1):5-es, 2007.
- [26] David Liben-Nowell and Jon Kleinberg. The link-prediction problem for social networks. journal of the association for information science and technology (2007). Google Scholar Google Scholar Digital Library Digital Library , 2007.
- [27] Aditya Krishna Menon and Charles Elkan. Link prediction via matrix factorization. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2011, Athens, Greece, September 5-9, 2011, Proceedings, Part II 22 , pages 437-452. Springer, 2011.
- [28] Sarah Morrison-Smith and Jaime Ruiz. Challenges and barriers in virtual teams: a literature review. SN Applied Sciences , 2(6):1096, 2020.
- [29] Seth A Myers and Jure Leskovec. On the convexity of latent social network inference. Advances in Neural Information Processing Systems , 2, 2010.
- [30] Jeongha Oh, Anjana Susarla, and Yong Tan. Examining the diffusion of user-generated content in online social networks. Available at SSRN 1182631 , 2008.
- [31] Manuel Gomez Rodriguez, Jure Leskovec, David Balduzzi, and Bernhard Schölkopf. Uncovering the structure and temporal dynamics of information propagation. Network Science , 2(1):26-65, 2014.
- [32] Daniel M Romero, Brendan Meeder, and Jon Kleinberg. Differences in the mechanics of information diffusion across topics: idioms, political hashtags, and complex contagion on twitter. In Proceedings of the 20th international conference on World wide web , pages 695-704, 2011.
- [33] Charles R Shipan and Craig Volden. The mechanisms of policy diffusion. American journal of political science , 52(4):840-857, 2008.
- [34] Madeleine Udell and Alex Townsend. Why are big data matrices approximately low rank? SIAM Journal on Mathematics of Data Science , 1(1):144-160, 2019.
- [35] Jacco Wallinga and Peter Teunis. Different epidemic curves for severe acute respiratory syndrome reveal similar impacts of control measures. American Journal of epidemiology , 160(6):509-516, 2004.
- [36] Liaoruo Wang, Stefano Ermon, and John E Hopcroft. Feature-enhanced probabilistic models for diffusion network inference. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2012, Bristol, UK, September 24-28, 2012. Proceedings, Part II 23 , pages 499-514. Springer, 2012.

- [37] Senzhang Wang, Xia Hu, Philip S Yu, and Zhoujun Li. Mmrate: Inferring multi-aspect diffusion networks with multi-pattern cascades. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining , pages 1246-1255, 2014.
- [38] Zhaoran Wang, Quanquan Gu, Yang Ning, and Han Liu. High dimensional expectationmaximization algorithm: Statistical optimization and asymptotic normality. arXiv preprint arXiv:1412.8729 , 2014.
- [39] Duncan J Watts. A simple model of global cascades on random networks. Proceedings of the National Academy of Sciences , 99(9):5766-5771, 2002.
- [40] Ming Yu, Varun Gupta, and Mladen Kolar. Estimation of a low-rank topic-based model for information cascades. Journal of Machine Learning Research , 21(71):1-47, 2020.
- [41] Gilles Zumbach and Paul Lynch. Heterogeneous volatility cascade in financial markets. Physica A: Statistical Mechanics and its Applications , 298(3-4):521-529, 2001.

## A Additional details on methodology

## A.1 Detailed expression of objectives in M-step

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where P I , P U follow the definitions in Section 3.1.

## A.2 Optimization algorithm and discussions

We summarize the optimization of the reformulated problem in Section 3.4 via EM algorithm as Algorithm 1.

## Algorithm 1 First-order projected EM algorithm

Require: initialization Ω p 0 q ' t Θ p 0 q , Ψ p 0 q , π p 0 q u , observed network A , low-rank penalty µ , learning rate λ , and stopping criterion ϵ .

<!-- formula-not-decoded -->

E-step : update ˆ π c i ' P p Z i | t c ; Ω p s q q via its posterior distribution based on Ω p s q for i ' 1 , ¨ ¨ ¨ , N, c ' 1 , ¨ ¨ ¨ , C .

M-step : decompose Q p Ω | Ω p s q q ' Q 1 p Θ | Ω p s q q ` Q 2 p Ψ | Ω p s q q ` Q 3 p π | Ω p s q q

M.1: Update

Θ

via

Q

1

p

Θ

|

Ω

p

s

q

M.2: Update Ψ via Q 2 p Ψ | Ω p s q q :

ˇ

<!-- formula-not-decoded -->

ˇ

<!-- formula-not-decoded -->

M.2.2: perform SVD decomposition on Ψ ' U Λ V J

<!-- formula-not-decoded -->

M.3: Update π via Q 3 p π | Ω p s q q :

<!-- formula-not-decoded -->

## end while

In Algorithm 1, we utilize projected gradient ascent and proximal gradient ascent to update diffusion networks Θ and Ψ in M.1 and M.2 of M-step, where the latter can be implemented via singular value soft-thresholding operation [14, 5]. For computational efficiency and stability, we utilize the first-order EM algorithm [2] such that the ELBO is increased via one-step gradient ascend instead of maximized in M-step. In addition, both the gradients B Q 1 Θ and B Q 2 Ψ have closed forms and can be efficiently calculated. The closed form gradients are provided in A.3.

Likelihood-based parameter tuning The low-rank penalty µ in the above Algorithm 1 can be selected in a data-adapted manner. Specifically, we can first randomly separate the total cascade samples into a training subset C train and a validation subset C val , and estimate model parameters ˆ Ω µ on C train given a specific µ . Then we calculate the log-likelihood of validation samples as 1 | C val | ř c P C val log P p t p c q , ˆ Ω µ q , and select µ such that

<!-- formula-not-decoded -->

q

:

where G is a grid of candidates for penalty values.

Sparsity pursuit on Θ When the observed network A is not sparse enough or the sample size of cascade is relatively small, one can further add l 1 regularizer on Θ in the objective function Q 1 p Θ | Ω p s q q . Accordingly, the matrix Θ is updated via an additional soft-thresholding operator in the M.1 step of Algorithm 1. Previous results shows that adopting the l 1 regularization can improve the performance and efficiency to recover the diffusion network structure [8, 31].

## A.3 Closed form gradients in Algorithm 1

Denote p S Θ c q ji ' S p t c i | t c j , Θ ji q , p S Ψ c q ji ' S p t c i | t c j , Ψ ji q , p H Θ c q ji ' H p t c i | t c j , Θ ji q , and p H Ψ c q ji ' H p t c i | t c j , Ψ ji q , where S p¨ | ¨ , ¨q and H p¨ | ¨ , ¨q are survival and hazard functions, respectively.

In addition, we introduce two cascade sample indicator variables I p 1 q p c q ji and I p 2 q p c q ji for each node pair p j, i q such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, B Q 1 Θ and B Q 2 Ψ over each element in Θ and Ψ , respectively, can be explicitly formulated as:

<!-- formula-not-decoded -->

The above gradients can be calculated by matrix operations after translating two indicator functions into matrices.

## B Additional details on experimental setup

## B.1 Benchmark comparison under different network topologies

In this experiment, we generate diffusion networks with different topologies. Specifically, we consider the diffusion network Θ and its support A as random network, network with community structure, and scale-free network. For random network, we generate A via Erdos-Renyi model with edge generation probability being 0 . 01 . For the community structure, we generate A with a stochastic block model that contains four equal-sized communities. The generation probability of within-community edge is 0 . 05 and that of between-community edge is 0 . 01 . For scale-free network, we generate A via Barabási-Albert model where we set the number of edges to attach from a new node to existing nodes to be 1. After generating support A , we set its diagonal to be 0 and assign to each non-zero edge a weight that follows Unif p 1 , 5 q to construct Θ .

Wegenerate the latent diffusion network as Ψ ' Ψ 1 Ψ J 2 where Ψ 1 , Ψ 2 P R N ˆ 5 ` . Wefix the proportion of non-zero edges in Ψ 1 , Ψ 2 at 0 . 1 and sample weights from Unif p 1 , 2 q for these non-zero edges. The generated Ψ has an edge density of 0.05 and transmission rates between 1 and 8 on non-zero edges.

In addition, we impute edges on A to ensure that each row and column of Θ Y Ψ has at least one non-zero element, i.e., the combined network is connected. Given Θ and Ψ , we generate C ' 2 , 000 independent cascade samples based on the double mixture model with Exp transmission model and observation window length of T ' 10 .

and

## B.2 Network recovery under different transmission models

In this experiment, we generate the latent diffusion network Ψ with low-rank structure using the same procedure as in B.1.

Given Ψ and its support B , we then generate Θ 's support A by forcing one overlap with B per column, i.e. tp i, j q | A ij B ij ' 1 u , and one non-overlap with B per column, i.e. tp i, j q | A ij ' 1 , B ij ' 0 u . For Exp and Pow transmission models, we create Θ by sampling transmission rates from Unif p 2 , 5 q for non-zero edges of A . We slightly modify the generation process for Ray model by decreasing the transmission rates on non-zero edges of Θ and Ψ . We instead sample weights from Unif p 0 . 1 , 0 . 8 q for non-zero edges in Ψ 1 , Ψ 2 and weights from Unif p 0 . 02 , 2 q for those in Θ . This modification aims to force significantly large differences in infection time that contribute to large differences in probability of infection among potential parents.

We also add edges in Θ to avoid all-zero row or column in Θ Y Ψ to ensure diffusion network connectivity. The resulting Θ and Ψ have edge density at 0.01 and 0.05, respectively. Based on Θ and Ψ , we generate different numbers of cascade samples at C ' 500 , 1000 , 1500 , 2000 with a fix observation window length at T ' 10 . In the Pow transmission model, we select delay parameter δ ' 1 .

## B.3 Network recovery under large network settings

In this experiment, for each network size N ' 500 , 1000 , 2000 , 4000 , we generate the latent diffusion network Ψ following the same procedure as in B.2 but fixing its rank at 5 and edge density at 0.01. Given Ψ , we also follow the same procedure in B.2 to generate Θ . The resulting Θ has an edge density of 0.001. Based on Θ and Ψ , we generate C ' 50000 independent cascade samples with a fixed observation window length of T ' 10 .

To cope with memory limits when the network size N and the sample size C are large, in each outer iteration, we stream the data in batches of size B : we process one batch at a time to compute the contributions (likelihood terms, gradients) required by both the E-step and the M-step, accumulate these quantities across all r M { B s batches, and then carry out the parameter update using the aggregated totals-thus matching the effect of a full-batch EM update while keeping memory usage bounded. When more memory is available, one can increase the batch size B accordingly for better parallelization on GPU and thus shorter execution time per iteration.

## C Additional numerical results

## C.1 Network recovery under different support overlap

In this subsection, we investigate the performance of our proposed method under different levels of overlap between Θ and Ψ , defined as overlap p Θ , Ψ q ' ř i,j I p A ij ˆ B ij q{ ř i,j I p A ij ` B ij q , where A , B are the support of Θ , Ψ , respectively.

We generate Ψ = Ψ 1 Ψ T 2 following similar procedure in B.1 except that we sample weights of nonzero edges in Ψ 1 and Ψ 2 from Unif p 0 . 2 , 1 . 5 q and increase its rank to 30 to better control its overlap degree with Θ . The edge density of the generated Ψ is 0 . 025 and the transmission rates on its non-zero edges range from 0.05 to 2.

Given Ψ and its support B , we generate different supports A following the same procedure in B.2 while changing their overlaps and non-overlaps per column with B so that overlap p Θ , Ψ q takes three levels at about 0 . 1 , 0 . 3 , 0 . 5 . Then we create Θ by sampling transmission rates from Unif p 0 . 1 , 0 . 2 q Y p 1 . 9 , 2 . 0 q for the overlap tp i, j q | A ij B ij ' 1 u , and sampling from Unif p 1 , 2 q for non-overlap support tp i, j q | A ij ' 1 , B ij ' 0 u .

We also add edges in Θ with transmission rates from Unif p 1 , 2 q to avoid all-zero row or column in Θ Y Ψ to ensure diffusion network connectivity. Based on Θ and Ψ , we generate different numbers of cascade samples from Exp transmission model, and fix the observation window length at T ' 10 .

Table 5 shows that the diffusion network recovery performance of the proposed method improves as sample sizes increases. Additionally, as overlap level between Θ and Ψ increases, the transmission rates estimations MAE on both Θ , Ψ and π decreases while estimation of Ψ 's support improves.

This pattern uncovers a trade-off between network differentiation and topology recovery in network mixture model. As the overlap level increases, the joint diffusion complexity on Θ and Ψ decreases due to the shared diffusion pathways and support information. On the other hand, it becomes more difficult and requires more samples to differentiate Θ and Ψ on their overlap component.

Table 5: Diffusion network estimations from the proposed method under different degrees of overlapping between diffusion networks.

|   overlap p Θ , Ψ q |                                    | MAE Θ                                               | MAE Ψ                                               | MAE π                                               | Acc Ψ                                               | Pre Ψ                                               | Rec Ψ                                               |
|---------------------|------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
|                 0.1 | C = 500 C = 1000 C = 1500 C = 2000 | 0.325(0.014) 0.309(0.015) 0.307(0.016) 0.277(0.012) | 0.510(0.018) 0.414(0.016) 0.391(0.018) 0.360(0.012) | 0.314(0.013) 0.300(0.016) 0.297(0.016) 0.288(0.015) | 0.690(0.007) 0.793(0.008) 0.827(0.013) 0.853(0.004) | 0.553(0.009) 0.669(0.010) 0.711(0.018) 0.748(0.007) | 0.919(0.005) 0.974(0.003) 0.989(0.002) 0.994(0.002) |
|                 0.3 | C = 500 C = 1000 C = 1500 C = 2000 | 0.393(0.012) 0.388(0.018) 0.387(0.013) 0.366(0.017) | 0.577(0.017) 0.519(0.016) 0.507(0.018) 0.489(0.022) | 0.435(0.016) 0.454(0.015) 0.456(0.012) 0.452(0.018) | 0.764(0.008) 0.852(0.006) 0.887(0.008) 0.908(0.003) | 0.647(0.010) 0.756(0.010) 0.804(0.013) 0.835(0.004) | 0.933(0.008) 0.977(0.003) 0.989(0.002) 0.994(0.002) |
|                 0.5 | C = 500 C = 1000 C = 1500 C = 2000 | 0.458(0.014) 0.438(0.018) 0.442(0.018) 0.421(0.019) | 0.618(0.014) 0.555(0.013) 0.556(0.026) 0.526(0.019) | 0.533(0.017) 0.555(0.011) 0.560(0.011) 0.557(0.011) | 0.810(0.017) 0.884(0.034) 0.919(0.027) 0.935(0.030) | 0.712(0.023) 0.809(0.053) 0.863(0.046) 0.886(0.052) | 0.940(0.006) 0.975(0.006) 0.986(0.004) 0.992(0.004) |

## C.2 Performance under different time window length

In this subsection, we investigate the network recovery performance under different observation window lengths, which ranges in T P t 1 , 2 , 3 , 5 , 10 u .

We generate Θ and Ψ following the same procedure in C.1 and fix their overlap level at about 0 . 30 . We generate C ' 1500 cascade samples from transmission models Exp, Pow, Ray.

We illustrate the performance in Figure 5 and 6. In general, both transmission rate estimations and network topology recovery improve as the observation window becomes T longer since effective diffusion information increases within each cascade sample. Specifically, by Figure 5, the proposed method achieves better MAE of Θ and Ψ under Ray (green line with shaded one standard deviation error range) and Pow (yellow line) cascade model. MAE criterion under Exp model is not strictly monotone, but its overall trend matches that of Pow and Ray. Figure 6 shows that accuracy and precision of Ψ also increases as T increases and then remains stable. The results in this subsection suggest to set a large T in practice, which is also consistent with our theoretical analysis.

Figure 5: Parameter estimation under different time window lengths.

<!-- image -->

## C.3 Network recovery without support information

In this subsection, we investigate the performance of the proposed method when support A of diffusion network Θ is unobserved.

We consider diffusion networks Θ and Ψ of size N ' 100 , generated by similar scheme as in C.1. Since the size of network is reduced, rank of Ψ 1 , Ψ 2 , where Ψ ' Ψ 1 Ψ T 2 , is accordingly reduced to 15. The generated Θ and Ψ have edge densities of 0.02 and 0.05, receptively. Non-zero edges of Θ and Ψ have ranges 0.1 to 2 and 0.05 to 2.2, respectively. The level of overlap between the two networks is

Figure 6: Topology recovery of latent network estimation under different time window lengths.

<!-- image -->

fixed at 0.15. We generate cascade samples of different sizes C ' 2000 , 3000 , 4000 , 5000 and from all three different transmission models Exp, Pow, and Ray.

Table 6 shows that when support of Θ is not observed, performance of the proposed method on transmission rate estimation still increases as number of samples increases for all three cascade generation models. Benchmark values in Table 6 are obtained from the proposed method with information on the support of Θ . Figure 7 and 8 show the relationship between precision and recall of the estimated Θ and Ψ when their entries below some varying thresholds are truncated to 0. As sample size increases, the performance of the proposed method on network topology recovery increases (shown by the larger area under P-R curves). Although the proposed method without support information of Θ cannot achieve as good performance as when the information is available, its performance on both transmission rate estimation and network topology recovery still exhibits converging pattern to the benchmark performance as sample size increases.

Table 6: Parameter estimation from the proposed method when support of Θ is unknown.

|     |                                     | MAE Θ                                               | Acc Θ                                                            | MAE Ψ                                                            | Acc Ψ                                                            |
|-----|-------------------------------------|-----------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|
| Exp | C = 2000 C = 3000 C = 4000 C = 5000 | 0.607(0.028) 0.533(0.038) 0.447(0.107) 0.327(0.113) | 0.628(0.013) 0.702(0.016) 0.742(0.053) 0.787(0.047)              | 0.672(0.027) 0.604(0.030) 0.520(0.089) 0.414(0.095)              | 0.803(0.015) 0.883(0.007) 0.902(0.009) 0.904(0.024)              |
|     | Benchmark                           | 0.088(0.005)                                        | 0.976(0.003)                                                     | 0.210(0.008)                                                     | 0.926(0.006)                                                     |
| Ray | C = 2000 C = 3000 C = 4000 C = 5000 | 0.665(0.044) 0.553(0.112) 0.292(0.054) 0.232(0.022) | 0.573(0.021) 0.680(0.053) 0.824(0.027) 0.858(0.012)              | 0.699(0.039) 0.592(0.093) 0.376(0.042) 0.326(0.017)              | 0.601(0.020) 0.699(0.040) 0.811(0.016) 0.837(0.010)              |
|     | Benchmark C = 2000 C = 3000         | 0.067(0.002) 0.604(0.064) 0.447(0.051) 0.331(0.062) | 1.000(0.000) 0.732(0.026) 0.786(0.043) 0.822(0.025) 0.791(0.022) | 0.177(0.002) 0.537(0.051) 0.423(0.037) 0.344(0.048) 0.341(0.034) | 0.941(0.023) 0.855(0.010) 0.871(0.008) 0.879(0.010) 0.869(0.010) |
| Pow | C = 4000 C = 5000                   | 0.346(0.047)                                        |                                                                  |                                                                  |                                                                  |
|     |                                     | 0.061(0.002)                                        |                                                                  | 0.175(0.008)                                                     |                                                                  |
|     |                                     |                                                     | 0.976(0.003)                                                     |                                                                  |                                                                  |
|     | Benchmark                           |                                                     |                                                                  |                                                                  | 0.916(0.002)                                                     |

Figure 7: Precision-recall plot of Θ

<!-- image -->

Figure 8: Precision-recall plot of Ψ

<!-- image -->

## D Additional details on method implementation

To give proper initializations to our proposed method, we use estimation result Λ from NetRate [31]. For experiments in Section 4.1, 4.2, C.1, and C.2, we initialize Θ 0 ' Λ d A and Ψ 0 ' Λ d p 1 ´ A q , where A is the support of Θ .

We employ a new initialization method in C.3 since information on A , the support of Θ , is missing. We initialize Ψ 0 as the truncated SVD of Λ at rank r . We then initialize Θ 0 as the top γ % entries of the residual Λ ´ Ψ 0 . Both r and γ are tunable hyperparameters, where 1 ď r ď N and 0 ă γ ă 100 . In the optimization process of C.3, instead of forcing gradient updates of Θ taking non-zero values only on A , we introduce an additional l 1 regularizer on Θ in the loss function to promote its sparsity and update Θ via an additional l 1 soft-thresholding operator.

We also provide a short discussion in this section on the general strategy to choose the best hyperparameters r λ Θ , λ Ψ , µ, ρ Θ , ρ Ψ s , where λ denotes learning rate, µ denotes low-rank penalty, ρ denotes l 1 penalty, and the subscript denotes the network associated with.

In general, as network size, sample size, or edge density increases, smaller λ Θ , λ Ψ are required for stable and good performance of the algorithm. In terms of different cascade transmission models, the optimal learning rates for Pow model have the largest magnitude, while those for Ray model have the smallest. Level of overlap, length of observation window, or the rank of Ψ does not significantly affect learning rate choices.

For fixed t Θ , Ψ , π u , there exist best threshold λ Ψ ¨ µ for singular value soft-thresholding operation and best thresholds λ Θ ¨ ρ Θ , λ Ψ ¨ ρ Ψ for l 1 soft-thresholding operation. Thus, for fixed t Θ , Ψ , π u , we need to adjust penalties inversely proportional to learning rates for the best performance of the proposed algorithm. l 1 penalties should be increased as transmission rates on Θ , Ψ increase, while µ should be increased as rank of Ψ increases.

## E Proofs for theoretical results

## Proposition 1

Proof. To prove the column-wise identification between Θ ¨ i and Ψ ¨ i , we only need to consider node i 's direct parents in two networks as R Θ i : ' t j P t 1 , ¨ ¨ ¨ , N u | Θ ji ą 0 u and R Ψ i : ' t j P t 1 , ¨ ¨ ¨ , N u | Ψ ji ą 0 u as the sets of nodes that can directly reach node i on Θ and Ψ . Denote the activation times of the direct parents of node i as t R i P r 0 , T s | R Θ i Y R Ψ i | : ' t t j | j P R Θ i Y R Ψ i u .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the summation and multiplication are over j P R Θ i Y R Ψ i with p : ' | R Θ i Y R Ψ i | . Given that survival function satisfies S p t j ; λ ji q ' exp t λ ji h p t i ´ t j qu for some non-negative function h p¨q , then

the infection likelihood and hazard function are

<!-- formula-not-decoded -->

Denote w 1 p t j q ' π i h 1 p t i ´ t j q ś k S p t k ; Θ ki q , w 2 p t j q ' p 1 ´ π i q h 1 p t i ´ t j q ś k S p t k ; Ψ ki q , ˜ w 1 p t j q ' ˜ π i h 1 p t i ´ t j q ś k S p t k ; ˜ Θ ki q , and ˜ w 2 p t j q ' p 1 ´ ˜ π i q h 1 p t i ´ t j q ś k S p t k ; ˜ Ψ ki q . Then (15) can be re-written as

<!-- formula-not-decoded -->

which should hold for any t R i . Consider a 4 p ˆ 4 p matrix Γ with each row being r w 1 p t 1 q , ¨ ¨ ¨ , w 1 p t p q , w 2 p t 1 q , ¨ ¨ ¨ , w 2 p t p q , ˜ w 1 p t 1 q , ¨ ¨ ¨ , ˜ w 1 p t p q , ˜ w 2 p t 1 q , ¨ ¨ ¨ , ˜ w 2 p t p qs where each row takes different values of t R i . Based on assumption that there exist j ˚ ‰ i such that Θ j ˚ i ‰ Ψ j ˚ i , then up to permutation between ˜ Θ ¨ i and ˜ Ψ ¨ i , the columns corresponding to different t j will be linear independent. Without loss of generality, we only need to investigate the 4 p ˆ 4 submatrix Γ j with each row being r w 1 p t j q , w 2 p t j q , ˜ w 1 p t j q , ˜ w 2 p t j qs . We prove that w 1 p t j q ' ˜ w 1 p t j q , w 2 p t j q ' ˜ w 2 p t j q and Θ ji ' ˜ Θ ji , Ψ ji ' ˜ Ψ ji by contradiction argument. For two different values t j and t 1 j , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we can choose t j ‰ t 1 j for those j ˚ such that Θ j ˚ i ‰ Ψ j ˚ i . Then the above equation implies Θ j ˚ i ' Ψ j ˚ i , which cause contradiction. Therefore, the rank of Γ j is larger than 1.

If there exists a, b such that a ˆ w 1 p t j q ` b ˆ w 2 p t j q ' ˜ w 1 p t j q hold for any t R i . Then we let t j ' t i for all j P R Θ i Y R Ψ i except j ˚ . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a 1 ' a π i ˜ π i , b 1 ' b 1 ´ π i ˜ π i . Notice that at least one of Θ j ˚ i ´ ˜ Θ j ˚ i and Ψ j ˚ i ´ ˜ Θ j ˚ i is not zero and the monotonicity of exp p λt q in terms of t , the above exponential function equation cannot hold as long as h p t i ´ t q is not constant over t . Therefore, the rank of Γ j can not be 2 . Using the same argument above, we can show the rank of Γ j can not be 3 as well.

If the rank of Γ j is 4, i.e., Γ j is full rank, then Γ is also full rank by applying above argument on each j P R Θ i Y R Ψ i except j ˚ . Based on (17), we have Γ r Θ J ¨ i , Ψ J ¨ i , ˜ Θ J ¨ i , ˜ Ψ J ¨ i s J ' 0 , which leads to Θ J ¨ i ' Ψ J ¨ i ' ˜ Θ J ¨ i ' ˜ Ψ J ¨ i ' 0 and contradicts to assumption } Θ ¨ i } 1 `} Ψ ¨ i } 1 ą 0 . Therefore, w 1 p t j q ' ˜ w 1 p t j q , w 2 p t j q ' ˜ w 2 p t j q and Θ ji ' ˜ Θ ji , Ψ ji ' ˜ Ψ ji for all j . Finally, by the definition of w 1 p t j q and tildew 1 p t j q , we can derive π i ' ˜ π i . The statement based on P U can be similarly derived.

## Proposition 2

Proof. Given Proposition 1, we have the column-wise identification of Θ ¨ i and Ψ ¨ i , and only the identification of Θ and Ψ up to the column exchange between Θ and Ψ . Notice that Proposition 1 guarantee the identification of C ' Θ ` Ψ . Then if

<!-- formula-not-decoded -->

we have Λ 1 p Θ q X Λ 2 p Ψ q ' H based on Proposition 1 in [6]. Notice that Θ P Λ 1 p Θ q and Ψ P Λ 2 p Ψ q , then Θ and Ψ are identifiable conditioning on C is identifiable.

## Theorem 3.1

Proof. We first show Ξ Θ p s q , Ξ Ψ p ρ q , and Ξ π are convex sets for any s ą 0 , ρ ą 0 by checking that the convex combination of any two elements in Ξ Θ p s q , Ξ Ψ p ρ q , or Ξ π still lies in the set.

For any Θ 1 , Θ 2 P Ξ Θ p s q and 0 ď λ ď 1 ,

<!-- formula-not-decoded -->

Ξ Θ p s q is thus a convex set. Note that s ą 0 avoids the trivial case where A is a zero matrix.

It is easy to argue that Ξ π is a convex set using similar argument to (24).

For any Ψ 1 , Ψ 2 P Ξ Ψ p ρ q and 0 ď λ ď 1 ,

<!-- formula-not-decoded -->

By (25) and similar argument to (24), Ξ Ψ p ρ q is a convex set.

Next, we show Q 3 p π | Ω p s q q is concave over π . Recall the formulation of Q 3 p π | Ω p s q q in A.1

<!-- formula-not-decoded -->

When π P Ξ π , both log p π q and log p 1 ´ π q are concave over π . From the above formulation, Q 3 p π | Ω p s q q is concave over π by linearity of concavity.

Without loss of generality, we show Q 1 p Θ | Ω p s q q is concave on Θ under the assumption that hazard function H p t | t 1 , λ q satisfies B H 2 p t | t 1 , λ q{B λ 2 ' 0 for t ě t 1 .

Note that this assumption immediately implies the concavity of H p t | t 1 , λ q and the log concavity of S p t | t 1 , λ q over λ .

Recall the definition of P I and P U in Section 3.1 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rewrite (26) and (27) in the following form,

<!-- formula-not-decoded -->

Then by concavity and monotonicity of log function, composition rule for concavity, and linearity of concavity, both log P I p t i ; Θ ¨ i q and log P U p T ; Θ ¨ i q are concave over Θ ¨ i .

Concavity of Q 1 p Θ | Ω p s q q over Θ immediately follows from composition rule for concavity and the proof for Q 3 p π | Ω p s q q . Concavity of Q 2 p Θ | Ψ p s q q directly follows this proof by changing Θ to Ψ .

By the above proof, the linear combination ř j : t c j ą T ˆ π c j log P U p T ; Θ ¨ j q where ˆ π c j depends on Ω p s q instead of Θ is concave over Θ . Therefore, E t p ř j : t c j ą T ˆ π c j log P U p T ; Θ ¨ j qq is concave in terms of Θ . In addition, notice that B 2 E t r Q 1 p Θ | Ω p s q qs B 2 Θ is a p N 2 ´ N q ˆ p N 2 ´ N q diagonal block matrix as B 2 E t r Q 1 p Θ | Ω p s q qs B Θ ij B Θ kl ' 0 if j ‰ l , which leads to N blocks ´ B 2 E t r Q 1 p Θ | Ω p s q qs B Θ ji B Θ ki ¯ p N ´ 1 qˆp N ´ 1 q , i ' 1 , ¨ ¨ ¨ , N . And the i th block corresponds to the second derivative of

<!-- formula-not-decoded -->

Based on the above arguments, we only need to show that the strictly concavity for each block in B 2 E t r Q 1 p Θ | Ω p s q qs B 2 Θ , i.e., E t ` 1 C ř C c ' 1 1 t t c i ď T u ˆ π c i log P I p t i ; Θ ¨ i q ˘ . We then set C to be larger enough such that C 0 : ' ř C c ' 1 1 t t c i ď T u ą N ´ 1 and set the index of cascade where i is infected as 1 , ¨ ¨ ¨ , C 0 , and

<!-- formula-not-decoded -->

Then we denote Q ' 1 C ř C 0 c ' 1 ˆ π c i log P I p t i ; Θ ¨ i q and follow the argument of Lemma 10 in [8] to prove Q is positive definite. Following their notations and α : ' Θ ¨ i , we can write

<!-- formula-not-decoded -->

where D p α q ' 1 C ř C 0 c ' 1 D p t c ; α q and D p t c ; α q is a diagonal matrix with r D p t c ; α qs jj ' ´ ˆ π c i S 2 p t c i | t c j ; α k q ´ ˆ π c i h ´ 1 p t c , α q H 2 p t c i | t c j ; α k q , and h p t c , α q ' ř j : t c j ă t c i H p t c i | t c j ; α k q . In addition, X p α q is a p N ´ 1 q -byC 0 matrix as

<!-- formula-not-decoded -->

where each column X p t c ; α q ' a ˆ π c i h ´ 1 p t c , α q ∇ α h p t c , α q . Given the assumption that the probabilities of being source node P p v q ą 0 for v P R where R denotes the set of nodes from which i is reachable via a directed path, we can follow the same argument in Lemma 10 of [8] to show Q is strictly concave in terms of Θ . Then E t p Q q is also strictly concave. The strong concavity of E t r Q 2 p Ψ | Ω p s q qs in terms of Ψ can be similarly proved.

## F Information on computer resources

Numerical experiments in this paper were carried out in Google Colab on a single Nvidia A100 GPU with 80GB of memory available. Regardless of failed experiments, the entire numerical experiment section took approximately 50 hours of computation time.

In addition, to evaluate the benchmark algorithm ConNIe, we used SNOPT (version 7.7) as the underlying constrained optimization solver [15, 16].

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses the limitations of this work in the Conclusion section.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All assumptions are stated and all proofs are provided in either Methodology section or Appendix.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper fully discloses all the information needed to reproduce the main experimental results. Code for experiments are provided as supplementary material.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The paper provides open access to the data and code with sufficient instructions to faithfully reproduce the main experimental results.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies all the training and test details necessary to understand the results.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper reports standard deviations where appropriate with suitable explanations.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper provides information on computer resources in the Appendix.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: In Real cascading data analysis section, we apply our method to analyze the research topic cascades in social science across U.S. universities and uncover the latent research topic diffusion networks among the top U.S. social science programs.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: This paper uses open-domain data and code, properly crediting the license and creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: New code accompanying this paper is well documented and submitted as supplementary material.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]
## A Cramér-von Mises Approach to Incentivizing Truthful Data Sharing

## Alex Clinton

University of Wisconsin-Madison aclinton@wisc.edu

## Thomas Zeng

University of Wisconsin-Madison tpzeng@wisc.edu

## Xiaojin Zhu

University of Wisconsin-Madison jerryzhu@cs.wisc.edu

Yiding Chen Cornell University yc2773@cornell.edu

## Kirthevasan Kandasamy

University of Wisconsin-Madison kandasamy@cs.wisc.edu

## Abstract

Modern data marketplaces and data sharing consortia increasingly rely on incentive mechanisms to encourage agents to contribute data. However, schemes that reward agents based on the quantity of submitted data are vulnerable to manipulation, as agents may submit fabricated or low-quality data to inflate their rewards. Prior work has proposed comparing each agent's data against others' to promote honesty: when others contribute genuine data, the best way to minimize discrepancy is to do the same. Yet prior implementations of this idea rely on very strong assumptions about the data distribution (e.g. Gaussian), limiting their applicability. In this work, we develop reward mechanisms based on a novel two-sample test statistic inspired by the Cramér-von Mises statistic. Our methods strictly incentivize agents to submit more genuine data, while disincentivizing data fabrication and other types of untruthful reporting. We establish that truthful reporting constitutes a (possibly approximate) Nash equilibrium in both Bayesian and prior-agnostic settings. We theoretically instantiate our method in three canonical data sharing problems and show that it relaxes key assumptions made by prior work. Empirically, we demonstrate that our mechanism incentivizes truthful data sharing via simulations and on real-world language and image data.

## 1 Introduction

Data is invaluable for machine learning (ML). Yet many organizations and individuals lack the capability to collect sufficient data on their own. This has driven the emergence of data marketplaces [1-3]-where consumers purchase data from contributors with money-and consortia [4-6] for data sharing and federated learning-where agents share their own data in return for access to others' data. As such platforms depend critically on data from contributing agents, they incentivize these agents to contribute more data via commensurate rewards: consortia typically grant agents greater access to the pooled data [7, 8], while marketplaces provide correspondingly larger payments [9, 10].

However, most existing work implicitly assume that contributors will report data truthfully. In reality, strategic contributors may untruthfully report data to exploit the incentive scheme. As one such example, they may fabricate data -either through naïve random generation or sophisticated ML-based synthesis -to artificially inflate their submissions and maximize their own rewards. In naive incentive schemes, where rewards scale with the quantity of data, such behavior can flood the system with poor quality data which undermines trust in the platform.

The central challenge in preventing such strategic misreporting, including fabrication, is that consortia and marketplace operators typically lack ground-truth knowledge about the underlying data

distribution-if the ground truth were known, the very need for learning and data sharing would be obviated. To address this, prior work has proposed a simple and intuitive idea: compare each agent's data submission against the pooled submissions of other agents. In these mechanisms, when all agents' data come from the same distribution, truthful reporting constitutes a Nash equilibrium. Intuitively, when others contribute genuine data, minimizing the discrepancy between one's own submission and the aggregate submission of others also requires submitting genuine data.

Despite this promising intuition, prior work has succeeded only under strong assumptions about data distributions [7, 11] and/or narrow models of untruthful behavior [12-14]. Realizing this idea to general data distributions and arbitrary types of strategic misreporting has remained challenging.

Our contributions. This gap motivates the central premise of our work. We develop a mechanism where agents are rewarded based on a novel loss function that is inspired by two-sample testing. Our loss function, resembling the Cramér-von Mises (CvM) two-sample test statistic [15, 16], is computationally inexpensive, and applies to many different data types, including complex data modalities such as text and images. We design (approximate) Nash equilibria in which agents are incentivized to truthfully report data, without relying on restrictive assumptions about the underlying distribution or strategic behaviors. We theoretically demonstrate the application of our mechanism in three data sharing problems involving purchasing data, and data sharing without money. We empirically demonstrate its usefulness via experiments on synthetic and real world datasets.

## 1.1 Overview of Contributions

̸

Model. There are m agents. Each agent i possesses a dataset X i drawn from an unknown distribution P , and submits Y i , not necessarily truthfully (i.e. Y i = X i ). In data-sharing consortia or marketplaces, the goal is to design losses (negative rewards) L = { L i } i ∈ [ m ] , where agent i is rewarded according to -L i ( { Y j } j ∈ [ m ] ) , so as to incentivize truthful reporting. A natural and widely adopted approach [7, 11, 10], which we also follow, is to design L i as a function of the form L i ( Y i , Y -i ) , where Y -i = ( Y j ) j = i is the pooled submission of all agents except i . A high value of L i ( Y i , Y -i ) suggests that agent i 's data deviates from the rest, which may indicate untruthful behavior when other agents report truthfully (i.e. Y j = X j for all j = i ).

̸

Comparing an agent's submission to the pooled data from others can be naturally viewed as computing a two-sample test statistic -or simply, a two-sample test -between Y i and Y -i [15, 17]. This perspective motivates the design of our loss function.

Key technical challenges. There are two primary challenges in designing a loss. First, we should ensure that the loss L is truthful : specifically, when Y -i is drawn i.i.d. from P (i.e. all other agents report truthfully), the optimal strategy for agent i to minimize L i ( Y i , Y -i ) should be to also submit truthfully, i.e. Y i = X i . Without this property, agents may have an incentive to manipulate their submissions to reduce L i ( Y i , Y -i ) . However, many standard two-sample tests-such as Kolmogorov-Smirnov [17, 18], t -test [19], Mann-Whitney [20], and MMD [21]-are not provably truthful. The second challenge is to reward agents for higher quality submissions, i.e. L i should decrease as the quantity of the submitted (truthful) data increases.

While each challenge is easy to address in isolation, satisfying both simultaneously is far more difficult. For example, a mechanism that rewards agents equally is trivially truthful but offers no incentive to collect more data. Conversely, if losses are tied solely to the quantity of submitted data, the mechanism becomes vulnerable to data fabrication, leaving honest agents worse off.

A third, less central challenge is ensuring that we have a handle on the distribution of L i to enable its application in data sharing use cases. For instance, penalizing large values of L i requires understanding what constitutes 'large' under truthful reporting. Prior work addresses these three challenges only under strong assumptions on P (e.g. Gaussian [7, 11], Bernoulli [22], restricted class of exponential families [10]), or narrow models of untruthful reporting [12, 13, 22].

Our method and results. In §2, we consider a Bayesian setting in which each agent's data is drawn from an unknown distribution P , itself sampled from a known prior Π . We introduce our loss L which is inspired by the Cramér-von Mises (CvM) test. Leveraging this statistic along with user-specified data featurizations, we design a loss in which truthful reporting forms an exact Nash equilibrium (NE). Moreover, we show that L incentivizes the submission of larger datasets-an agent is strictly better off by submitting more truthful data. Our loss is also bounded, and decreases gracefully with the amount of data submitted, making it useful for data sharing applications as we will see in §4.

̸

However, this approach has two practical limitations. First, specifying a meaningful prior can be difficult, particularly for complex data modalities such as text or images. Second, even with a prior, computing L may be intractable when it requires expensive Bayesian posterior computations. In §3, we address these issues by replacing the above Bayesian version of our loss with a prior-agnostic version that is simpler to compute. We show that this leads to a truthful ε -approximate NE in both Bayesian and frequentist settings where ε approaches zero as the amount of data submitted increases. We also show that agents benefit from submitting more data, and that our new loss is also bounded and decreases gracefully with the amount of data submitted.

Applications. In §4, we theoretically demonstrate how our Bayesian method can be applied to solve three different data sharing problems, some of which have been studied in prior work, while relaxing their technical conditions. The first problem is incentivizing truthful data submissions via payments assuming agents already possess data [10]. The second is the design of a data marketplace where a buyer is willing to pay strategic agents to collect data on her behalf [23]. The third is a federated learning setting where agents wish to share data for ML tasks without the use of money [8].

Empirical evaluation. In §5, we empirically evaluate our methods on simulations, and real world image and language experiments. To simulate untruthful behavior, we consider agents who augment their datasets by fabricating samples using simple fitted models, or generative models such as diffusion models and LLMs [24-26]. Our results demonstrate that such untruthful submissions lead to larger losses compared to truthful reporting. This corroborates theoretical results for both methods and demonstrates that the prior-agnostic version is practically useful for real world data sharing.

## 1.2 Related Work

There has been growing interest in the incentive structures underlying data sharing, federated learning, and data marketplaces. A central goal in these settings is to incentivize data contributions. However, most prior work do not consider untruthful reporting. When they do, they either impose restrictive distributional assumptions, or limit how contributors may misreport.

Incentivizing data sharing without truthfulness requirements. A line of work addresses incentivizing data collection in federated learning [27, 8, 28-31, 9]. Other studies focus on incentivizing the sharing of private data [32] or truthful reporting of private data collection costs [14]. All of these works assume agents report data truthfully, and do not encounter the challenges we address here.

Restricted distributional assumptions. Cai et al. [9] study a principal-agent model where a principal selects measurement locations and compensates agents who exert costly effort to reduce observation noise. Their optimal contract relies on a known effort-to-data-quality function, which may be unknown or nonexistent in practice. Ghosh et al. [22] design a mechanism to purchase binary data under differential privacy, compensating agents for privacy loss. Chen et al. [10] drop the privacy constraint to handle non-binary data, proposing a fixed-budget mechanism that ensures truthful reporting, but requiring the data distribution to have finite support or belong to an exponential family. Other work focuses on incentivizing truthful reporting in Gaussian mean estimation for data sharing [7, 11] and data marketplaces [23]; however, as our experiments show, their approach-based on comparing means of the reported data-does not generalize beyond Gaussian data.

Restricted untruthful reporting. Falconer et al. [13] propose monetary incentives for data sharing, assuming agents can only fabricate data by duplicating existing entries. Dorner et al. [12] study mean estimation where agents may misreport only by adding a scalar to their true values.

Peer prediction. The peer prediction literature addresses a challenge similar to ours: eliciting truthful reports without access to ground truth. Prior work [33-36] uses reported signals to cross-validate agents' submissions, showing that truthful reporting forms an (approximate) Nash equilibrium. Techniques from [37, 38] have been applied to design payment-based mechanisms for data sharing [10], but these rely on strong assumptions about the data distribution (e.g., exponential families or finite support). It is not clear if these methods generally work when agents may change the number of signals (data points) they have, which is a critical consideration in data sharing use cases where fabrication is possible. More precisely, the mechanism designer does not know how many data points an agent holds, yet must still incentivize truthful reporting.

Practical applicability. The vast majority of the above works focus on theoretical development, but lack empirical evaluation, with their practicality unclear due to expensive Bayesian computations. In contrast, our prior-agnostic method is simple and performs well on real data.

Review of the Cramér-von Mises test. We briefly review the Cramér-von Mises (CvM) test [15]. Let X = { X 1 , . . . , X n } i.i.d. ∼ F 1 and Y = { Y 1 , . . . , Y m } i.i.d. ∼ F 2 be samples from R -valued distributions F 1 and F 2 , respectively. Let F X ( t ) = 1 | X | ∑ x ∈ X 1 { x ≤ t } and F Y ( t ) = 1 | Y | ∑ y ∈ Y 1 { y ≤ t } be the empirical CDFs (ECDFs) of X and Y . Set Z = ( X 1 , . . . , X n , Y 1 , . . . , Y m ) . The two-sample CvM test statistic is then defined below in (1). We have illustrated the CvM test in Fig. 1a.

<!-- formula-not-decoded -->

## 2 A Truthful Mechanism in a Bayesian Setting

In this section, we design a mechanism to reward agents based on the quality of their submitted data. We begin by specifying our model. To build intuition, we present a simplified single-variable version of our loss (mechanism) in §2.1. We then present the general version of our mechanism in §2.2.

Setting. There are m&gt; 2 agents, where each agent i ∈ [ m ] has a dataset X i = { X i, 1 , . . . , X i,n i } ⊂ { X i,j } ∞ j =1 of n i ∈ N points. Here { X i,j } ∞ j =1 are drawn i.i.d. from an unknown distribution P over X and X i ∈ X n i . We refer to X as the dataspace; examples include the space of images, text, or simply R d . In this section, we consider a Bayesian setting where P is drawn from a publicly known prior Π . A mechanism designer wishes to incentivize the agents to report their datasets truthfully by designing losses (negative rewards).

Let D = ⊔ ∞ ℓ =0 X ℓ be the collection of finite subsets of X , which forms the space of datasets an agent could possess. A mechanism for this problem is a normal form game which maps the agents' dataset submissions to a vector of losses, i.e. L ∈ { L ′ : D m → R m } . Once the mechanism L is published, each agent will submit a dataset Y i (not necessarily equal to X i ). An agent's strategy can be viewed as a function f i ∈ F = { f : D → D s.t. f is measurable } which maps their original dataset X i to Y i = f i ( X i ) . This allows for strategic data manipulations which may depend on the agent's own dataset. Let I be the identity (truthful) strategy which maps a dataset to itself, i.e. I ( X i ) = X i .

Agent i 's loss L i is the i 'th ouput of the mechanism L , and is a function of the strategies f = { f i } i ∈ [ m ] adopted by other agents and the initial datasets X = { X 1 , . . . , X m } , and can be written as L i = L i ( { f i } i ∈ [ m ] ) = L i ( { f i ( X i ) } i ∈ [ m ] ) to highlight or suppress these dependencies.

Requirements. The mechanism designer wishes to design L to satisfy two key properties:

1. Truthfulness: All agents submitting truthfully ( f = I ), is a Nash equilibrium, that is,

̸

- i ∀ i ∈ [ m ] , ∀ f i ∈ F , E [ L i ( { I } m j =1 ) ] ≤ E [ L i ( f i , { I } j = i )] .
2. More (data) is (strictly) better (MIB): Let X , X ′ be two datasets such that | X ′ | &gt; | X | . Then,

̸

- i i i i .

<!-- formula-not-decoded -->

Above, the expectation is with respect to the prior P ∼ Π , the data X i , X ′ i ∼ P for all i , and any randomness in the agent strategies f i and mechanism L . As discussed in §1.1 under 'Key technical challenges', while satisfying either of these requirement is easy, designing a mechanism which satisfies both simultaneously is significantly more difficult.

## 2.1 Warm-up when X = R

Algorithm 1 description. To build intuition, we first study the simple one-dimensional case X = R . The mechanism works by aggregating all of the submissions { Y i } m i =1 and for each agent i ∈ [ m ] , computing a (randomized) loss L i . To compute L i , an evaluation point T i is first randomly sampled from the data submitted by the other agents Y -i . The remaining data Z i is used to define the empirical CDF F Z i . The loss L i is then defined as the squared difference between this ECDF evaluated at T i , i.e. F Z i ( T i ) , and its conditional expectation given ( X i, 1 , . . . , X i, | Y i | , T i ) evaluated at ( Y i, 1 , . . . , Y i, | Y i | , T i ) . Finally, the mechanism outputs L i ∈ [0 , 1] as agent i 's loss.

Design intuition: The conditional expectation E [ F Z i ( T i ) | X i, 1 , . . . , X i, | Y i | , T i ] can be thought of as the best guess for F Z i ( T i ) having seen ( X i, 1 , . . . , X i, | Y i | ) . Thus, E [ F Z i ( T i ) | X i, 1 = Y i, 1 , . . . , X i, | Y i | = Y i, | Y i | , T i ] can be thought of as the best guess for F Z i ( T i )

<!-- image -->

Figure 1: Subfigure (a) shows the empirical CDFs (ECDF) for two datasets X = { X 1 , . . . , X n } , Y = { Y 1 , . . . , Y m } . The gray lines are the differences between the two curves at each point in ( X 1 , . . . , X n , Y 1 , . . . , Y m ) , and are used to calculate the two-sample CvM test in (1). Subfigure (b) replaces F Y ( t ) with E [ F Y ( t ) | X ] which can be thought of as the best approximation to F Y ( t ) based on having seen X .

## Algorithm 1 A single variable Cramér-von Mises style statistic

- 1: Input parameters : A prior Π over the set of R -valued distributions.
- 2: for each agent i ∈ [ m ] :

̸

- 3: Y -i ← ( Y j,ℓ ) j = i,ℓ ∈| Y j | .
- 4: Sample j ∼ Unif (1 , . . . , | Y -i | ) and set T i ← Y -i,j , Z i ← ( Y -i,ℓ ) ℓ = j .

̸

- 5: Return L i ← ( E [ F Z i ( T i ) | X i, 1 = Y i, 1 , . . . , X i, | Y i | = Y i, | Y i | , T i ] -F Z i ( T i ) ) 2 .

assuming that ( Y i, 1 , . . . , Y i, | Y i | ) is the agent's true data. A visual comparison of F Z i ( T i ) to E [ F Z i ( T i ) | X i, 1 , . . . , X i, | Y i | , T i ] can be seen in Fig. 1b.

The loss L i defined above is well-posed and computable. As demonstrated in our experiments (with derivations in Appendix E), closed-form expressions for L i can be derived in simple conjugate settings such as Gaussian-Gaussian and Bernoulli-Beta, enabling efficient implementations. For more complex prior distributions, numerical approximations using methods such as MCMC [39] or variational inference [40] can be employed.

Theoretical results. We now present the theoretical properties of Algorithm 1. To satisfy the MIB condition, we require that the prior Π meet a non-degeneracy condition, formalized in Definition 1. Intuitively, this condition ensures that the posterior changes upon observing an additional data point. Examples of degenerate priors include those that select a fixed distribution P with probability 1, or choose P to be a degenerate distribution δ x , x ∈ X with probability 1. In such cases, data sharing is meaningless, as the distribution is either fully known or revealed by a single sample. Thus, it is natural to assume Π is non-degenerate, so that additional data remains informative.

Definition 1. (Degenerate priors): Let P ∼ Π and { X i } ∞ i =1 , T, Z i.i.d. ∼ P . We say that Π is degenerate if for some n ∈ N , P ( Z ≤ T | T, X 1 , . . . , X n ) a.s. = P ( Z ≤ T | T, X 1 , . . . , X n +1 ) .

Theorem 1 shows that Algorithm 1 satisfies truthfulness for all priors Π , and MIB when Π is not degenerate. The key idea for truthfulness is that by computing the aforementioned conditional expectation, the mechanism performs, on behalf of agent i , the best possible guess for F Z i ( T i ) just using Y i . Thus, it is in agent i 's best interest if Y i = X i .

Theorem 1. The mechanism in Algorithm 1 satisfies truthfulness. Moreover, when Π is not degenerate, then Algorithm 1 also satisfies MIB.

While the previous theorem indicates that submitting more data is beneficial for the agent, it does not quantify how an agent's loss decreases as they contribute more data. The following proposition quantifies this by offering bounds on how an agent's expected loss decreases with the amount of data they submit, assuming all agents are truthful. This handle on E [ L i ] , along with the property that L i ∈ [0 , 1] , is useful for applying our mechanism to data sharing applications as we will see in §4.

## Algorithm 2 A feature-based Cramér-von Mises style statistic

̸

- 1: Input parameters : A prior Π over the set of X -valued distributions, feature maps { φ k } K k =1 . 2: for each agent i ∈ [ m ] : 3: Y -i ← ( Y j,ℓ ) j = i,ℓ ∈| Y j | . 4: Sample j ∼ Unif (1 , . . . , | Y -i | ) and set T i ← Y -i,j , Z i ← ( Y -i,ℓ ) ℓ = j . 5: for each feature k ∈ [ K ] : 6: Z k i ← ( φ k ( Z i,j ) ) | Z i | j =1 , T k i ← φ k ( T i ) . 7: L k i ← ( E [ F Z k i ( T k i ) ∣ ∣ X i, 1 = Y i, 1 , . . . , X i, | Y i | = Y i, | Y i | , T k i ] -F Z k i ( T k i ) ) 2 . 8: Return L i ← 1 K ∑ K k =1 L k i .

̸

Proposition 1. Let L i ( { I } m i =1 ) denote the value of L i when agents are truthful in Algorithm 1. Then, 0 ≤ E [ L i ( { I } m i =1 )] ≤ 1 4 ( 1 | X i | + 1 | Z i | ) . Moreover, when Π is a prior over the set of continuous R -valued distributions, 1 6 | Z i | ≤ E [ L i ( { I } m i =1 )] ≤ 1 6 ( 1 | X i | + 1 | Z i | ) .

## 2.2 A General Mechanism with Feature Maps

We now extend our mechanism and to handle data from arbitrary distributions. The key modification is the introduction of feature maps: functions chosen by the mechanism designer that transform general data distributions into R -valued distributions to apply our mechanism to.

Feature maps. We define a feature map to be any measurable function φ : X → R which maps the data to a single variable distribution. We will see that any collection of feature maps { φ k : X → R } K k =1 which map the data to a collection of single variable distributions supports a truthful mechanism. However, some feature maps perform better than others depending on the use case, so we allow the mechanism designer flexibility to select maps. For Euclidean data, coordinate projections may suffice, while for complex data like text or images, embeddings from deep learning models are more appropriate (as used in our experiments in §5).

Algorithm 2 description. The mechanism designer first specifies a collection of feature maps, { φ k } K k =1 based on the publicly known prior Π . After this, Algorithm 2 can be viewed as applying Algorithm 1 for each feature k ∈ [ K ] , making use of φ k to map general data in X to R .

The following theorem shows that Algorithm 2 is truthful, which is a result of the same arguments made in Theorem 1, now repeated for each feature map. For MIB, we require an analogous condition to the one given in Theorem 1, stating that more data leads to a more informative posterior distribution for at least one of the K features. To state this formally, we first extend Definition 1.

Definition 2. Let P ∼ Π and { X i } ∞ i =1 , T, Z i.i.d. ∼ P . We say that Π is degenerate for feature k ∈ [ K ] if for some n ∈ N ,

<!-- formula-not-decoded -->

Theorem 2. The mechanism in Algorithm 2 satisfies truthfulness. Moreover, if there is a feature k ∈ [ K ] , for which Π is not degenerate, then Algorithm 2 also satisfies MIB.

Proposition 9 (Appendix C.2), analogous to Proposition 1, quantifies how L i decreases with data size, which will be useful when using this loss in data sharing applications. Additionally, Proposition 8 (Appendix C.2) gives an explicit relationship for how the expected loss changes when an agent submits an additional data point, depending on the prior and feature maps. This exactly quantifies how much lower an agent's loss is when submitting more data.

## 3 A Prior Agnostic Mechanism

While our mechanism in §2 applies broadly in Bayesian settings, it has two practical limitations. First, specifying a meaningful prior can be difficult, especially for complex data like text or images. Second, even with a suitable prior, computing the conditional expectation in line 7 may be intractable due to

## Algorithm 3 A prior free Cramér-von Mises style statistic

- 1: Input parameters : Feature maps { φ k } K k =1 and an augment split map ψ : N → N : ψ ( n ) &lt; n -1 .
- 2: for each agent i ∈ [ m ] :

̸

- 3: Y -i ← ( Y j,ℓ ) j = i,ℓ ∈| Y j | .
- 4: Split Y -i into Y -i = ( { T i } , W i , Z i ) s.t. | W i | = ψ ( | Y -i | ) .
- 5: for each feature k ∈ [ K ] :
- 6: T k i ← φ k ( T i ) , W k i ← ( φ k ( W i,ℓ ) ) | W i | ℓ =1 , Z k i ← ( φ k ( Z i,ℓ ) ) | Z i | ℓ =1
- 7: L k i ← ( F ( Y k i ,W k i ) ( T k i ) -F Z k i ( T k i ) ) 2 .
- 8: Return L i ← 1 K ∑ K k =1 L k i .

the cost of Bayesian posterior inference. To address this, we introduce a prior-agnostic variant that is significantly easier to compute. The trade-off is that truthful reporting becomes an ε -approximate NE, where ε vanishes as the amount of submitted data grows.

Changes to Algorithm 2. Thus far, we have only focused on the Bayesian setting, assuming that agents wish to minimize their expected loss E P∼ Π [ E { X i } i m =1 ∼P [ L i ] ] . However, this modification also supports a frequentist view where agents wish to minimize their worst case expected loss over a class C possible distributions, i.e. sup P∈C E { X i } i m =1 ∼P [ L i ] . In the frequentist setting, the class C is the analog of the prior Π . As such, our prior agnostic mechanism does not have a prior Π as input.

Algorithm 3 computes each agent's loss as follows: first partition Y -i into three parts, (1) an evaluation point T i , (2) data to augment agent i 's submission with W i , and (3) data to compare agent i 's submission against Z i . The mechanism designer is free to choose how much data to allocate to W i as given by the map ψ . For each feature k ∈ [ K ] , we then obtain T k i , W k i , and Z k i by applying φ k . The main modification of the prior-agnostic mechanism is that the conditional expectation in line 7 of Algorithm 2, E [ F Z k i ( T k i ) ∣ ∣ X i = Y i , T k i ] , is replaced with F ( Y k i ,W k i ) ( T k i ) which serves as an easy to compute estimate for F Z k i ( T k i ) . Here F ( Y k i ,W k i ) denotes the ECDF from the combined data of Y k i and W k i . The reason we allow the mechanism designer the flexibility to supplement Y k i with W k i is that doing so allows them to decrease the ε parameter corresponding to truthfulness being an ε -approximate Nash in the following theorem. A reasonable choice for the size of W i is to set it so that | W i | + | Y i | = | Z i | .

Before stating the theorem, we define ε -approximate truthfulness for a mechanism in both the Bayesian and frequentist paradigms.

ε -Approximate Truthfulness: All agents submitting truthfully ( f i = I ), is an ε -approximate Nash equilibrium. In the Bayesian setting this means ∀ i ∈ [ m ] , ∀ f i ∈ F

̸

<!-- formula-not-decoded -->

In the frequentist setting this means ∀ i ∈ [ m ] , ∀ f i ∈ F

<!-- formula-not-decoded -->

̸

Algorithm 3 requires a similar non-degeneracy condition for MIB. In the Bayesian setting, the same condition given in Theorem 2 suffices. In the frequentist setting, we require that the class of distributions C is not solely comprised of distributions for which all of the feature map induced distributions are degenerate. The following theorem summarizes the main properties of Algorithm 3. We see that as the total amount of data increases, the approimate truthfulness parameter vanishes provided that the datasets ( X i , W i ) and Z i are balanced.

Theorem 3. The mechanism in Algorithm 3 is 1 4 ( 1 | X i | + | W i | + 1 | Z i | ) -approximately truthful in both the Bayesian and frequentist settings. Moreover, if there is a feature k ∈ [ K ] , for which Π is not degenerate, then Algorithm 3 satisfies MIB in the Bayesian setting. If it is not the case that C ⊆ { P ∈ M 1 ( X ) : ∀ k ∈ [ K ] , P ◦ ( φ k ) -1 ∈ δ x , x ∈ R } then Algorithm 3 satisfies MIB in the frequentist setting.

Proposition 10 (Appendix D) gives, for both the Bayesian and frequentist settings, bounds on how an agent's expected loss decreases with the amount of data they submit, assuming all agents are truthful. Moreover, when the pushforward P k = P ◦ ( φ k ) -1 is a.s. continuous ∀ k ∈ [ K ] , this proposition provides an exact expression for the expected loss in both the Bayesian and frequentist settings.

## 4 Applications to Data Sharing Problems

1. A data marketplace for purchasing existing data. Our first problem, studied by Chen et al. [10] is incentivizing agents to truthfully submit data using payments from a fixed budget B in a Bayesian setting. Their mechanism requires the data distribution to have finite support or belong to the exponential family to ensure budget feasibility (payments do not exceed B ) and individual rationality (agents receive non-negative payments). Our method removes these distributional assumptions.

In this setting, m agents each posses a dataset X i = { X i, 1 , . . . , X i,n i } with points drawn i.i.d. from an unknown distribution P in a Bayesian model. A data analyst with budget B wishes to purchase this data. Agents submit datasets { Y i } m i =1 in return for payments { π i ( { Y i } i ) } i m =1 . Chen et al. [10], building on Kong and Schoenebeck [38], design a truthful mechanism based on log pairwise mutual information, but their payments can be unbounded, violating budget feasibility and individual rationality. We address this using Algorithm 2 to construct bounded payments satisfying truthfulness, individual rationality, and budget feasibility without distributional assumptions. Algorithm 4 (see Appendix A.1) implements this, and Proposition 2 guarantees these properties.

2. A data marketplace to incentivize data collection at a cost. The second problem, studied by Chen et al. [23], involves designing a data marketplace in which a buyer wishes to pay agents to collect data on her behalf at a cost . They study a Gaussian mean estimation problem in a frequentist setting. We study a simplified Bayesian version without assuming Gaussianity.

In a data marketplace mechanism, the interaction between the buyer and agents takes place as follows. First, each agent chooses how much data to collect, n i ∈ N , paying a known per-sample cost c , and obtains the dataset X i = { X i, 1 , . . . , X i,n i } with data drawn i.i.d. from an unknown P ∼ Π . They submit Y i = f i ( X i ) to the mechanism, and in return, receive a payment π i ( { Y i } m i =1 ) charged to the buyer. The buyer derives value v : Z ≥ 0 → R ≥ 0 from the total amount of truthful data received. An agent's utility is their expected payment minus collection cost u a i = E [ π i ( { Y i } m i =1 )] -cn i , and the buyer's utility, when agents are truthful, is the valuation of the data received minus the expected sum of payments, u b = v ( ∑ m i =1 | Y i | ) -E [ ∑ m i =1 π i ( { Y i } m i =1 )] .

The goal of a data market mechanism is to incentivize agents to collect and truthfully report data. If not carefully designed, the mechanism may incentivize agents to fabricate data to earn payments without incurring collection costs, undermining market integrity and deterring buyers. To address this, we propose Algorithm 5 (see Appendix A.2), using Algorithm 2, which-unlike Chen et al. [23]-does not assume Gaussianity. Proposition 3 shows that, under a market feasibility condition, the mechanism is incentive compatible for agents and individually rational for buyers.

3. Federated learning. The third problem is a simple federated learning setting, similar to Karimireddy et al. [8], where agents share data to improve personalized models. Unlike their work, which assumes agents truthfully report collected data, we allow strategic misreporting.

Each of m agents, possess a dataset X i = { X i, 1 , . . . , X i,n i } of points drawn i.i.d. in a Bayesian model, and have a valuation function v i : N → R (increasing), quantifying the value of using a given amount of data for their machine learning task. Acting alone, an agent's utility is simply v i ( | X i | ) . When participating, the federated learning mechanism delopys a subset of the others' data submitted, Z i , for agent i 's task based on the quality of their submission f i ( X i ) . This result in a valuation of v i ( | Z i | ) when the others are truthful. Thus, an agent's utility when participating is defined as u i = E [ v i ( | Z i | )] . We propose Algorithm 6 (see Appendix A.3), based on Algorithm 2, which does not assume truthful reporting. Proposition 4 shows it is truthful and individually rational.

## 5 Experiments

Synthetic experiments. We consider two Bayesian models with conjugate priors (beta-Bernoulli and normal-normal) where the calculation of the conditional expectation in line 7 of Algorithm 1 is analytically tractable. In both setups, X = R and we will use the method in Algorithm 1.

Figure 2: (a): Losses when submitting truthfully, adding Bern(1 / 2) samples, and adding Bern(˜ p ) samples in the beta-Bernoulli experiment. (b): Losses when submitting truthfully and adding fabricated data between adjacent pairs of true data points in the normal-normal experiment. In (b), the CvM bar for fabrication behavior extends to ≈ 1 . 6 . Losses for truthful submission in each method and subfigure are normalized to 1 (gray lines); values &lt; 1 indicate fabrication improves performance, &gt; 1 means it worsens. A truthful mechanism should yield losses above 1 for all fabrication behavior.

<!-- image -->

Baselines: We compare our mechanism to three standard two-sample tests, used here as losses: (1) the KS-test KS ( Y i , Y -i ) = sup t ∈ R | F Y i ( t ) -F Y -i ( t ) | . (2) The CvM test (the direct version, not our adaptation): CvM ( Y i , Y -i ) (see (1)). (3) The mean difference (similar to the t -test): Mean-diff ( Y i , Y -i ) = ∣ ∣ 1 | Y i | ∑ y ∈ Y i y -1 | Y -i | ∑ y ∈ Y -i y ∣ ∣ , which has been used to incentivize truthful reporting for normal mean estimation in a frequentist settings [7, 23].

1) Beta-Bernoulli. Our first model is a beta-Bernoulli Bayesian model with p ∼ Beta (2 , 2) and then X i,j | p ∼ Bern( p ) i.i.d. We evaluate whether an agent can reduce their loss (increase rewards) by adding fabricated data to their submission. We consider two types of fabrication: (1) adding Bern(1 / 2) samples and (2) estimating p via ˜ p = 1 | X i,j | ∑ x ∈ X i,j x then adding Bern(˜ p ) samples. We compare this to an agent's loss when submitting truthfully, assuming in both cases that other agents are truthful. Fig. 2a shows average losses under Algorithm 1 and the three two-sample tests under truthful and non-truthful reporting. Under Algorithm 1, fabricated data always leads to higher loss, while the baselines yields lower loss under at least one fabrication strategy. Thus, the two-sample tests are susceptible to data fabrication whereas Algorithm 1 is not. Notably, Mean-diff, which is used in [7, 11], fails, showing their methods do not work beyond normal mean estimation settings. 2) Normal-normal. Our second experiment is a normal-normal Bayesian model, where µ ∼ N (0 , 1) and then X i,j | µ ∼ N ( µ, 1) i.i.d. Here, we fabricate data by inserting fake points in between real observations. Fig. 2b presents the results. Truthful reporting yields lower loss under Algorithm 1, CvM, and Mean-diff, while KS gives lower loss for fabrication, revealing its susceptibility.

Language data. Next, we evaluate our method and the above baselines on language data. For this, we use data from the SQuAD dataset [41], where each data point is a question about an article. We model the environment with m = 20 and m = 100 agents, where all agents have 2500 and 500 original data points respectively. We fabricate data by prompting Llama 3.2-1B-Instruct [26] to generate fake sentences based on the legitimate sentences that agent 1 has. We fabricate the same number of sentences in the original dataset. Agent 1 then submits the combined dataset, both true and fabricated, to the mechanism. We instantiate Algorithm 3 with feature maps obtained from the feature layer of the DistilBERT [42] encoder model, which corresponds to 768 features. We apply the baselines to the same set of features and take the average. We have provided additional details on the experimental set up and some true and fabricated sentences generated in Appendix B.1.

The results are presented in Table 1, showing that all methods perform well, obtaining a smaller loss for truthful submission when compared to fabricating. It is worth emphasizing that only our method is provably approximately truthful, and other methods may be susceptible to more sophisticated types of fabrication.

Image data. We perform a similar experiment on image data using the Oxford Flowers-102 dataset [43] dataset. where each data point is an image of a flower. We model the enviornment with m = 5 and m = 47 , where all agents have roughly 1000 and 100 original data points respectively.

Table 1: An agent's average loss ( ± the standard error) when reporting sentences truthfully/untruthfully, assuming the others are reporting truthfully. The experiments were run once assuming all agents had 500 sentences, then again assuming all agents had 2500 sentences. In each row the smaller loss is bolded.

|   Sentences | Method                                  | Avg. truthful loss                                                                               | Avg. untruthful loss                                                                            |
|-------------|-----------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
|         500 | Algorithm 3 KS-test CvM-test Mean diff. | 0.0003 ± 1 . 8 · 10 - 5 0.0379 ± 7 . 6 · 10 - 4 0.1547 ± 9 . 2 · 10 - 3 0.0043 ± 2 . 5 · 10 - 4  | 0.0011 ± 5 . 8 · 10 - 5 0.0524 ± 9 . 7 · 10 - 4 0.8598 ± 4 . 8 · 10 - 2 0.0095 ± 3 . 4 · 10 - 4 |
|        2500 | Algorithm 3 KS-test CvM-test Mean diff. | 0.00003 ± 3 . 3 · 10 - 6 0.0127 ± 2 . 4 · 10 - 4 0.1609 ± 7 . 1 · 10 - 3 0.0015 ± 8 . 4 · 10 - 5 | 0.0005 ± 7 . 1 · 10 - 6 0.0309 ± 1 . 2 · 10 - 4 3.2760 ± 3 . 4 · 10 - 2 0.0069 ± 5 . 9 · 10 - 5 |

We fabricate data by using Segmind Stable Diffusion-1B [25], a lightweight diffusion model, to generate fake images of flowers based on the legitimate pictures. We fabricate the same number of images that an agent possesses. Algorithm 3 is instantiated with 384 feature maps corresponding to the 384 nodes in the embedding layer of DeIT-small-distilled [44], a small vision transformer. As above, we apply the baselines to the same set of features and take the average. Additional details on the experimental set up can be found in Appendix B.2.

Table 2 shows that, similar to text, all methods perform well, truthful submission leads to a lower loss compared to the fabrication procedure detailed above.

Table 2: An agent's average loss ( ± the standard error) when reporting images truthfully/untruthfully, assuming the others are reporting truthfully. The experiments were run once assuming agent 1 had 100 images, then again assuming agent 1 had 1000 images. The 4,612 images in the test set of [43] were used to represent the data submitted by other agents. In each row the smaller loss is bolded.

|   Images | Method                                  | Avg. truthful loss                                                                              | Avg. untruthful loss                                                                            |
|----------|-----------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
|      100 | Algorithm 3 KS-test CvM-test Mean diff. | 0.0015 ± 3 . 2 · 10 - 5 0.0833 ± 4 . 2 · 10 - 4 0.1491 ± 2 . 6 · 10 - 3 0.0462 ± 1 . 0 · 10 - 3 | 0.0040 ± 1 . 2 · 10 - 4 0.0993 ± 1 . 3 · 10 - 3 0.7730 ± 2 . 0 · 10 - 2 0.0953 ± 1 . 1 · 10 - 3 |
|     1000 | Algorithm 3 KS-test CvM-test Mean diff. | 0.0002 ± 3 . 7 · 10 - 6 0.0290 ± 2 . 1 · 10 - 4 0.1458 ± 3 . 5 · 10 - 3 0.0157 ± 5 . 2 · 10 - 4 | 0.0032 ± 2 . 9 · 10 - 5 0.0738 ± 2 . 7 · 10 - 4 4.5478 ± 3 . 0 · 10 - 2 0.0896 ± 3 . 2 · 10 - 4 |

## 6 Conclusion

We study designing mechanisms that incentivize truthful data submission while rewarding agents for contributing more data. In the Bayesian setting, we propose a mechanism that satisfies these goals under a mild non-degeneracy condition on the prior. We additionally develop a prior-agnostic variant that applies in both Bayesian and frequentist settings. We illustrate the practical utility of our mechanisms by revisiting data sharing problems studied in prior work, relaxing their technical assumptions, and validating our approach through experiments on synthetic and real-world datasets.

Limitations. The mechanisms in §2 rely on Bayesian posterior computations, which may be computationally expensive for complex priors. We also require specifying feature maps that effectively represent the data. While this offers flexibility for the mechanism designer to select applicationspecific features, there is no universally optimal way to choose them.

## Acknowledgments and Disclosure of Funding

This work was partially supported by NSF grant IIS-2441796.

## References

- [1] Ads Data Hub. https://developers.google.com/ads-data-hub/marketers . Accessed: 2025-04-24.
- [2] Delta Sharing. https://docs.databricks.com/en/data-sharing/index.html . Accessed: 2025-04-24.
- [3] AWS Data Transfer Hub. https://aws.amazon.com/solutions/implementations/ data-transfer-hub/ . Accessed: 2025-04-24.
- [4] IBM Data Fabric. https://www.ibm.com/data-fabric, 2024.
- [5] DAT Freight and Analytics. URL: www.dat.com/sales-inquiry/freight-market-intelligenceconsortium, 2024. Accessed: July 9, 2024.
- [6] Snowflake Data Marketplace. https://www.snowflake.com/en/product/features/marketplace/.
- [7] Yiding Chen, Jerry Zhu, and Kirthevasan Kandasamy. Mechanism design for collaborative normal mean estimation. Advances in Neural Information Processing Systems , 36:49365-49402, 2023.
- [8] Sai Praneeth Karimireddy, Wenshuo Guo, and Michael I Jordan. Mechanisms that incentivize data sharing in federated learning. arXiv preprint arXiv:2207.04557 , 2022.
- [9] Yang Cai, Constantinos Daskalakis, and Christos Papadimitriou. Optimum statistical estimation with strategic data sources. In Conference on Learning Theory , pages 280-296. PMLR, 2015.
- [10] Yiling Chen, Yiheng Shen, and Shuran Zheng. Truthful data acquisition via peer prediction. Advances in Neural Information Processing Systems , 33:18194-18204, 2020.
- [11] Alex Clinton, Yiding Chen, Jerry Zhu, and Kirthevasan Kandasamy. Collaborative mean estimation among heterogeneous strategic agents: Individual rationality, fairness, and truthful contribution. In Forty-second International Conference on Machine Learning , 2025.
- [12] Florian E Dorner, Nikola Konstantinov, Georgi Pashaliev, and Martin Vechev. Incentivizing honesty among competitors in collaborative learning and optimization. Advances in Neural Information Processing Systems , 36:7659-7696, 2023.
- [13] Thomas Falconer, Jalal Kazempour, and Pierre Pinson. Towards replication-robust data markets. arXiv preprint arXiv:2310.06000 , 2023.
- [14] Rachel Cummings, Katrina Ligett, Aaron Roth, Zhiwei Steven Wu, and Juba Ziani. Accuracy for sale: Aggregating data with a variance constraint. In Proceedings of the 2015 conference on innovations in theoretical computer science , pages 317-324, 2015.
- [15] Harald Cramér. On the composition of elementary errors. Scandinavian Actuarial Journal , 1: 141-80, 1928.
- [16] Richard von Mises. Probability Statistics and Truth , volume 7. Springer-Verlag, 1939.
- [17] A. N. Kolmogorov. Sulla Determinazione Empirica di una Legge di Distribuzione. Giornale dell'Istituto Italiano degli Attuari , 4:83-91, 1933.
- [18] N. V. Smirnov. On the Estimation of the Discrepancy Between Empirical Curves of Distribution for Two Independent Samples. Bulletin of Moscow University , 2(2):3-16, 1939.
- [19] Student. The probable error of a mean. Biometrika , 6(1):1-25, 1908. doi: 10.1093/biomet/6.1.1.

- [20] H. B. Mann and D. R. Whitney. On a test of whether one of two random variables is stochastically larger than the other. The Annals of Mathematical Statistics , 18(1):50-60, 1947. doi: 10.1214/ aoms.1177730491.
- [21] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. In Journal of Machine Learning Research , volume 13, pages 723-773, 2012.
- [22] Arpita Ghosh, Katrina Ligett, Aaron Roth, and Grant Schoenebeck. Buying private data without verification. In Proceedings of the fifteenth ACM conference on Economics and computation , pages 931-948, 2014.
- [23] Keran Chen, Alex Clinton, and Kirthevasan Kandasamy. Incentivizing truthful data contributions in a marketplace for mean estimation. arXiv preprint arXiv:2502.16052 , 2025.
- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 10684-10695, 2022.
- [25] Yatharth Gupta, Vishnu V Jaddipal, Harish Prabhala, Sayak Paul, and Patrick Von Platen. Progressive knowledge distillation of stable diffusion xl using layer level loss. arXiv preprint arXiv:2401.02677 , 2024.
- [26] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
- [27] Avrim Blum, Nika Haghtalab, Richard Lanas Phillips, and Han Shao. One for one, or all for all: Equilibria and optimality of collaboration in federated learning. In International Conference on Machine Learning , pages 1005-1014. PMLR, 2021.
- [28] Yann Fraboni, Richard Vidal, and Marco Lorenzi. Free-rider attacks on model aggregation in federated learning. In International Conference on Artificial Intelligence and Statistics , pages 1846-1854. PMLR, 2021.
- [29] Jierui Lin, Min Du, and Jian Liu. Free-riders in federated learning: Attacks and defenses. arXiv preprint arXiv:1911.12560 , 2019.
- [30] Baihe Huang, Sai Praneeth Karimireddy, and Michael I Jordan. Evaluating and incentivizing diverse data contributions in collaborative learning. arXiv preprint arXiv:2306.05592 , 2023.
- [31] Yiling Chen, Nicole Immorlica, Brendan Lucier, Vasilis Syrgkanis, and Juba Ziani. Optimal data acquisition for statistical estimation. In Proceedings of the 2018 ACM Conference on Economics and Computation , pages 27-44, 2018.
- [32] Alireza Fallah, Ali Makhdoumi, Azarakhsh Malekian, and Asuman Ozdaglar. Optimal and differentially private data acquisition: Central and local mechanisms. Operations Research , 72 (3):1105-1123, 2024.
- [33] Nolan Miller, Paul Resnick, and Richard Zeckhauser. Eliciting informative feedback: The peer-prediction method. Management Science , 51(9):1359-1373, 2005.
- [34] Drazen Prelec. A bayesian truth serum for subjective data. science , 306(5695):462-466, 2004.
- [35] Anirban Dasgupta and Arpita Ghosh. Crowdsourced judgement elicitation with endogenous proficiency. In Proceedings of the 22nd international conference on World Wide Web , pages 319-330, 2013.
- [36] Yiling Chen, Shi Feng, and Fang-Yi Yu. Carrot and stick: Eliciting comparison data and beyond. arXiv preprint arXiv:2410.23243 , 2024.
- [37] Yuqing Kong and Grant Schoenebeck. Water from two rocks: Maximizing the mutual information. In Proceedings of the 2018 ACM Conference on Economics and Computation , pages 177-194, 2018.

- [38] Yuqing Kong and Grant Schoenebeck. An information theoretic framework for designing information elicitation mechanisms that reward truth-telling. ACM Transactions on Economics and Computation (TEAC) , 7(1):1-33, 2019.
- [39] Walter R Gilks, Sylvia Richardson, and David Spiegelhalter. Markov chain Monte Carlo in practice . CRC press, 1995.
- [40] Martin J Wainwright, Michael I Jordan, et al. Graphical models, exponential families, and variational inference. Foundations and Trends® in Machine Learning , 1(1-2):1-305, 2008.
- [41] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing , pages 2383-2392, 2016.
- [42] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. ArXiv , abs/1910.01108, 2019.
- [43] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics &amp; image processing , pages 722-729. IEEE, 2008.
- [44] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers &amp; distillation through attention, 2021.
- [45] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) , pages 4171-4186, 2019.
- [46] Rick Durrett. Probability: theory and examples , volume 49. Cambridge university press, 2019.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: A discussion of limitations can be found in the conclusion.

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

Justification: Yes, the paper provides complete proofs of all the results provided in the appendix.

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

Justification: Experiment details can be found under the experiments section and in the appendix.

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

Justification: Sufficient code to replicate the experiments is provided.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Hyperparameter choices are explained in the experiments section and in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Appropriate information about the statistical significance of experiments is provided in the appendix.

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

Justification: The appendix provides such sufficient information.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Yes, the research conducted conforms with the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our paper is a theory paper and does not have direct societal impact.

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

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Assets are properly cited.

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

Justification: No new assets are released.

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

Justification: The core contributions of this paper do not use LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Omitted application algorithms

## A.1 A data marketplace for purchasing existing data

Recall the problem setup from §4. Below we provide a short algorithm that incentivizes agents to truthfully report their data, { X i } m i =1 , using payments. The idea is to use Algorithm 2 to quantify the quality of an agent's submission and, based on it, determine what fraction of the budget to pay them.

Definition 3. We say an algorithm is budget feasible if the sum of the payments never exceeds the budget ( ∑ m i =1 π i ≤ B ) , and individually rational (for participants) if the payments are always nonnegative ( ∀ i ∈ [ m ] , π i ≥ 0) .

Algorithm 4 A data marketplace for purchasing existing data

- 1: Input parameters : A prior Π over the set of X -valued distributions, feature maps { φ k } K k =1 .
- 3: Execute Algorithm 2 with { Y i } m i =1 , Π , { φ k } K k =1 , to obtain the loss L i ∈ [0 , 1] for agent i .
- 2: Receive datasets Y 1 , . . . , Y m from the agents.
- 4: Pay agent i : π i ( { Y i } m i =1 ) = B m (1 -L i ( { Y i } m i =1 )) .

Proposition 2. Algorithm 4 is truthful, individually rational, and budget feasibility.

Proof. Since L i ∈ [0 , 1] , we have 0 ≤ π i ≤ B m , so it immediately follows that Algorithm 4 is both individually rational for the agents and budget feasible. For truthfulness, notice that for any f i ∈ F , we can appeal to Theorem 2 to get

̸

<!-- formula-not-decoded -->

̸

Therefore, Algorithm 4 is also truthful, as agents maximize their expected payment when submitting truthfully.

## A.2 A data marketplace to incentivize data collection at a cost

Recall the problem setup from §4, which is a simplified version of the problem studied by [23]. Our setting does not subsume [23], as they allow for agents to have varying collection costs, study a frequentist setting (whereas we consider a Bayesian setting), and derive payments that are easy to compute. We now motivate a solution to our simplified setting.

To facilitate data sharing between a buyer and agents, a mechanism must first determine how much data agents should be asked to collect based on the cost of data collection c , and the buyer's valuation function v . To do this, suppose that the buyer could collect data himself. In this case, he would choose to collect n OPT := argmax n ∈ N ( v ( n ) -cn ) points to maximize his utility. However, as he cannot, when there are m agents, the mechanism will ask each of them to collect n OPT m points on his behalf in exchange for payments.

An important detail is that for the marketplace to be feasible, an agent's expected payment must outweigh the cost of data collection. This requirement is reflected in the first technical condition in Proposition 3, which at a high level says that the change in an agents expected payment with respect to n i , when collecting n OPT m points, is at least c . This can be thought of requiring that the derivative with repect to n i , of the expected payment at n OPT m , be at least c . We also assume that Π is not degnerate for all the features and there are deminishing returns for collecting and submitting more data under Algorithm 2.

When these condition holds, Proposition 3 shows that it is individually rational for a buyer to participate in the marketplace, and in agents' best interest to collect n OPT m points and submit them truthfully.

The idea of Algorithm 5 is to determine what fraction of v ( n OPT ) m to pay agent i based on the quality of her submission, as measured by L i .

## Algorithm 5 A data marketplace to incentivize data collection at a cost

- 1: Input parameters : A prior Π over the set of X -valued distributions, feature maps { φ k } K k =1 .
- 3: Execute Algorithm 2 with { Y i } m i =1 , Π , { φ k } K k =1 , to obtain the loss L i ∈ [0 , 1] for agent i .
- 2: Receive datasets Y 1 , . . . , Y m from the agents.
- 4: Pay agent i : π i ( { Y i } m i =1 ) = v ( n OPT ) m (1 -αL i ) where α is given in Definition 4.
- 5: Charge the buyer: p ( { Y i } m i =1 ) = ∑ m i =1 π i ( { Y i } m i =1 ) .

Definition 4. For Algorithm 5 we introduce notation for the change in an agent's expected payment when collecting and submitting one more data point truthfully, assuming others are truthful:

<!-- formula-not-decoded -->

When Π is not degenerate ∀ k ∈ [ K ] , ∂ ∂n i E [ L i ({ n OPT m } m i =1 )] &lt; 0 (by Theorem 2) and we define

<!-- formula-not-decoded -->

Proposition 3. Suppose that the following technical conditions are satisfied in Algorithm 5:

<!-- formula-not-decoded -->

Π is not degenerate ∀ k ∈ [ K ] , and -∂ ∂n i E [ L i ( n i , n -i )] is decreasing in n i .

Then, the strategy profile {( n OPT m , I )} m i =1 is individually rational for the buyer, i.e.

<!-- formula-not-decoded -->

and incentive compatible for the agents, i.e. for any n i ∈ N , f i ∈ F ,

̸

<!-- formula-not-decoded -->

Proof. We start with individual rationality for the buyer. Notice that if the inequality holds then we have

<!-- formula-not-decoded -->

so α ∈ (0 , 1] . Since L i ∈ [0 , 1] , this implies that

<!-- formula-not-decoded -->

so summing over the payments to all agents we find

<!-- formula-not-decoded -->

Therefore, the strategy profile {( n OPT m , I )} m i =1 is individually rational for the buyer since

<!-- formula-not-decoded -->

̸

We now prove incentive compatibility for the agents in two parts. First we show that regardless of how much data an agent has collected, it is best for her to submit it truthfully when others follow the recommended strategy profile {( n OPT m , I )} j = i . Second, we show that v ( n OPT ) m is the optimal amount of data to collect based on our choice of α .

Fix n i . Unpacking the definition of an agent's utility and applying Theorem 2 we have

̸

<!-- formula-not-decoded -->

̸

This means that regardless of how much data agent i collects, it is best for them to submit it truthfully. For the second part we now assume { f i } m i =1 = { I } m i =1 and n -i = { n OPT m } m i =1 so for convenience we omit writing the dependence on these parts of the strategy profile for random variables.

Notice that since u a i ( n i ) is concave, the optimal amount of data for agent i to collect and submit is the smallest n i ∈ N such that

<!-- formula-not-decoded -->

i.e. the point at which the marginal increase in payment no longer offsets the collection cost of an additional point. By the definition of agent utilities and our choice of α we see

<!-- formula-not-decoded -->

This implies that n OPT m is the optimal amount of data to collect since -∂ ∂n i E [ L i ( n i , n -i )] is decreasing in n i . Putting both parts togeter we find that for any n i ∈ N , f i ∈ F ,

̸

<!-- formula-not-decoded -->

so we have incentive compatibility for the agents.

̸

̸

̸

̸

̸

## A.3 Federated learning

Recall the problem setup from §4. For convenvience we assume that ∀ i ∈ [ m ] , | X i | &lt; ∑ j = i | X j | .

̸

The idea of Algorithm 6 is to determine how much of the others' data agent i should receive for her task based on the quality of her submission, as measured by L i .

## Algorithm 6 Federated learning

- 1: Input parameters : A prior Π over the set of X -valued distributions, feature maps { φ k } K k =1 .
- 3: Execute Algorithm 2 with { Y i } m i =1 Π , { φ k } K k =1 , to obtain the loss L i ∈ [0 , 1] for agent i .
- 2: Receive datasets Y 1 , . . . , Y m from the agents.
- 4: for each agent i ∈ [ m ] :

<!-- formula-not-decoded -->

- 6: z i ← v -1 i ((1 -αL i ) T i )
- 7: Deploy Z i , a random subset of Y -i of size z i for agent i 's machine learning task.

In Algorithm 6 and Proposition 4 we assume that E [ L i ( { I } m i =1 )] &gt; 0 which ensures α is well defined by rulling out trivial data sharing problems.

̸

Proposition 4. Suppose that ∀ i ∈ [ m ] , | X i | &lt; ∑ j = i | X j | . Then Algorithm 6 is truthful and individually rational.

Proof. Fix f i ∈ F . Unpacking the definition of an agent's utility and applying Theorem 2, we have

̸

̸

<!-- formula-not-decoded -->

̸

Therefore, Algorithm 6 is truthful. For individual rationality, notice that by the definition of α and the assumption that | X i | &lt; | X -i | ( and thus v i ( X i ) &lt; v i ( X -i ) ) , we have

<!-- formula-not-decoded -->

Therefore, agent i is better off participating in Algorithm 6 than working alone so individual rationality is satisfied.

## B Extended experimental results and details

## B.1 Text based experiments

Our first real world experiment supposes that agents possess and wish to share text data drawn from a common distribution. To simulate this text distribution, we use data from the SQuAD 1 dataset [41] which contains 100,000 questions generated by providing crowdworkers with snippets from Wikipedia articles and asking them to formulate questions based on the snippet's content. We simulate

1 This work uses the Stanford Question Answering Dataset (SQuAD), which is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.

̸

data sharing when m = 20 and m = 100 , where agents have 2,500 and 500 original data points respectively.

When agents are truthful, they simply submit their sentences to the mechanism (Algorithm 3). However, an untruthful agent can fabricate fake sentences to augment their dataset with in hopes of achieving a lower loss. We consider when agents attempt to do this using an LLM (Llama 3.2-1BInstruct [26]) by prompting it to produce authentic looking sentences based on legitimate sentences Fig. 3 shows an example of the prompting and Table 4 shows examples of the LLM-generated sentences. For consistency, we filter out duplicates and any outputs not ending in a question mark.

| Prompt                                                                                                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Generate five new questions that follow the same style as the examples below.                                                                                                           |
| Each question should be separated by a newline.                                                                                                                                         |
| According to Southern Living, what are the three best restaurants in Richmond? When did the Arab oil producers lift the embargo? Complexity classes are generally classified into what? |
| About how many acres is Pippy Park?                                                                                                                                                     |
| Which BYU station offers content in both Spanish and Portuguese?                                                                                                                        |

Figure 3: Pictured above is an example prompt fed into Llama 3.2-1B-Instruct as part of an untruthful agent's submission function to generate fabricated text data. The agent uses their five questions drawn from the SQuAD to fabricate similar five additonal questions.

Table 3: Comparison of SQuAD questions versus LLM-generated fabrications.

| SQuAD questions (Real)                                                                                                                                                                                                                                                                                                                  | LLM-generated questions (Fabricated)                                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Which tribe did Temüjin move in with at nine years of age? What is the most widely known fictional work from the Islamic world? New Delhi played host to what major athletic competition in 2010? Why did the FCC reject systems such as MUSE? Along with the philosophies of music and art, what field of philosophy studies emotions? | What percentage of the population of France lived in urban areas as of 2019? The term solar eclipse refers to what phe- nomenon? Is it true that the first computer bug was an actual insect? How many Earth years is Neptune's south pole exposed to the Sun? Military spending based on conventional threats has been dismissed as what? |

To incentivize truthful submission, we instantiate Algorithm 3 with 768 feature maps corresponding to the 768 nodes in the embedding layer of DistilBERT [42], a lightweight encoder model distilled from the encoder transformer model Bert [45]. For simplicity, we chose the split map ψ ( n ) = 0 . As a point of comparison, we also apply the KS, CvM, and Mean diff. tests (described in §5), now to the 768 node feature space.

Our results comparing the average loss agent i receives when submitting truthfully/untruthfully, under the four methods, over five runs, are given in Table 1. We see that under all of the methods truthful submission results in a lower average loss than untruthful submission.

## B.2 Image based experiments

Our second experiment supposes that agents wish to share image data from a common distribution. To simulate this image distribution, we use data from the Oxford Flowers-102 dataset [43], which contains 6,149 images across 102 flower categories. We simulate data sharing when an agent 1 has 100 and 1,000 images as data points. We use the test dataset of [43], which consists of 4,612 images, to represent authentic data submitted by the other agents. In the two scenarios, this roughly corresponds to m = 47 agents each with 100 images and m = 5 agents each with 1000 images.

When agent are untruthful, they may fabricate images using a diffusion model to augment their dataset. We consider when agents use Segmind Stable Diffusion-1B [25], a lightweight diffusion

model, to do this. More specifically, for each sampled image, we use it in conjunction with the prompts and parameters in Table 4 to generate an additional fabricated image.

Table 4: Parameters and prompts used for Segmind Stable Diffusion-1B to generate the fabricated images. Here cls\_name is replaced with the type of flower being generated.

| Parameter            | Value                                                                                                                                                                                                                               |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Text Prompt          | Photorealistic photograph of a single {cls_name}, realistic colors, natural lighting, high detail, sharp focus on petals. Another unique photo of the same flower species.                                                          |
| Negative Prompt      | oversaturated, highly saturated, neon colors, garish colors, vibrant colors, illustration, painting, drawing, sketch, cartoon, anime, unrealistic, blurry, low quality, text, watermark, signature, border, frame, multiple flowers |
| Strength             | 0.7                                                                                                                                                                                                                                 |
| Guidance Scale       | 6                                                                                                                                                                                                                                   |
| Num. Inference Steps | 50                                                                                                                                                                                                                                  |

To discourage fabrication, Algorithm 3 is now instantiated with 384 feature maps corresponding to the 384 nodes in the embedding layer of DeIT-small-distilled [44], a small vision transformer. For simplicity, we chose the split map ψ ( n ) = 0 . As a point of comparison, we again apply the KS, CvM, and Mean diff. tests (described in §5), now to the 384 node feature space. Our results comparing the average loss agent i receives when submitting truthfully/untruthfully, under the four methods, over five runs, can be found in Table 2.

We find that truthful reporting outperforms untruthful reporting for all methods, demonstrating they are not susceptible to diffusion based fabrication.

## C Results and proofs omitted from Section 2

## C.1 Results and proofs omitted from Subsection 2.1

Theorem 1. The mechanism in Algorithm 1 satisfies truthfulness. Moreover, when Π is not degenerate, then Algorithm 1 also satisfies MIB.

Proof. For truthfulness we refer to Proposition 5.

For n = ( n 1 , . . . , n m ) ∈ N m , let L i ( n, { I } m i =1 ) denote the value of L i in Algorithm 1 when agent j ∈ [ m ] has n j data points and agents use { I } m i =1 ∈ F m . Proposition 6 tells us that

<!-- formula-not-decoded -->

where G j = σ ( X i, 1 , . . . X i,j , T i ) . Also notice that

<!-- formula-not-decoded -->

By definition Π being non-degenerate means that the conditional probabilities are not almost surely equal. This implies that

<!-- formula-not-decoded -->

so the MIB property is satisfied.

Proposition 5. Let L i ( { f i } m i =1 ) denote the value of L i in Algorithm 1 when all agents use { f i } m i =1 ∈ F m . Then, for any f i ∈ F , E [ L i ( { I } m i =1 )] ≤ E [ L i ( f i , { I } j = i )] .

̸

Proof. By definition E [ F Z i ( T i ) ∣ ∣ X i, 1 , . . . , X i, | Y i | , T i ] is ( X i, 1 , . . . , X i, | Y i | , T i ) -measurable, so there exists a measurable function g : R | Y i | +1 → R such that

<!-- formula-not-decoded -->

The conditional expectation E [ F Z i ( T i ) ∣ ∣ X i, 1 = Y i, 1 , . . . , X i, | Y i | = Y i, | Y i | , T i ] is shorthand for g ( Y i, 1 , . . . , Y i, | Y i | , T i ) . Since we assume f i is measurable, ( Y i, 1 , . . . , Y i, | Y i | ) = f i ( X i, 1 , . . . , X i,n i ) is ( X i, 1 , . . . , X i,n i ) -measurable. Therefore, we know that

̸

g ( Y i, 1 , . . . , Y i, | Y i | , T i ) = E [ F Z i ( T i ) ∣ ∣ X i, 1 = Y i, 1 , . . . , X i, | Y i | = Y i, | Y i | , T i ] is ( X i, 1 , . . . , X i,n i , T i ) -measurable. This lets us apply Lemma 5 to get E [ L i ( f i , { I } j = i )] = E [ ( E [ F Z i ( T i ) ∣ ∣ X i, 1 = Y i, 1 , . . . , X i, | Y i | = Y i, | Y i | , T i ] -F Z i ( T i ) ) 2 ] ≥ E [ ( E [ F Z i ( T i ) ∣ ∣ X i, 1 , . . . , X i,n i , T i ] -F Z i ( T i ) ) 2 ] = E [ L i ( { I } m )] .

<!-- formula-not-decoded -->

Proposition 6. For n = ( n 1 , . . . , n m ) ∈ N m let L i ( n, { I } m i =1 ) denote the value of L i in Algorithm 1 when agent j ∈ [ m ] has n j data points and agents use { I } m i =1 ∈ F m . Then

<!-- formula-not-decoded -->

Proof. For convenience define U = F Z i ( T i ) and V = E [ U |G n i ] . By the definition of L i in Algorithm 1 and conditional variance we have

<!-- formula-not-decoded -->

Similarly we have

<!-- formula-not-decoded -->

Let Y = E [ U |G n i +1 ] . We can now appeal to Lemma 4 to get

<!-- formula-not-decoded -->

Using the tower property gives

<!-- formula-not-decoded -->

Proposition 1. Let L i ( { I } m i =1 ) denote the value of L i when agents are truthful in Algorithm 1. Then, 0 ≤ E [ L i ( { I } m i =1 )] ≤ 1 4 ( 1 | X i | + 1 | Z i | ) . Moreover, when Π is a prior over the set of continuous R -valued distributions, 1 6 | Z i | ≤ E [ L i ( { I } m i =1 )] ≤ 1 6 ( 1 | X i | + 1 | Z i | ) .

Proof. Since F X i ( T i ) is σ ( X i, 1 , . . . , X i,n i , T i ) -measurable, Lemma 5 tells us that

<!-- formula-not-decoded -->

Now we can condition on P apply the first part of Lemma 2 to the inner expectation to get

<!-- formula-not-decoded -->

Recognizing that L i is non-negative, we conclude

<!-- formula-not-decoded -->

For the second part we assume that Π ∈ M 1 ( M c 1 ( R )) , i.e. Π is a distribution over the set of continuous R -valued probability distributions. Again conditioning on P , we can now apply the second part of Lemma 2 to the inner expectation to get

<!-- formula-not-decoded -->

so the upper bound improves to

<!-- formula-not-decoded -->

For the lower bound notice that σ ( X i, 1 , . . . , X i,n i , T i ) ⊆ σ ( X i, 1 , . . . , X i,n i , T i , P ) . Therefore Lemma 5 tells us that

<!-- formula-not-decoded -->

But appealing to Lemmas 3 then 1 (using that P ∈ M c 1 ( R ) ) gives

<!-- formula-not-decoded -->

which concludes the proof of the lower bound.

## C.2 Proofs omitted from Subsection 2.2

Theorem 2. The mechanism in Algorithm 2 satisfies truthfulness. Moreover, if there is a feature k ∈ [ K ] , for which Π is not degenerate, then Algorithm 2 also satisfies MIB.

Proof. For truthfulness we refer to Proposition 7.

For n = ( n 1 , . . . , n m ) ∈ N m , let L i ( n, { I } m i =1 ) denote the value of L i in Algorithm 2 when agent j ∈ [ m ] has n j data points and agents use { I } m i =1 ∈ F m . Proposition 8 tells us that

<!-- formula-not-decoded -->

where G k j = σ ( X i, 1 , . . . X i,j , T k i ) . Also observe that

<!-- formula-not-decoded -->

Since we assume there is a feaure k ∈ [ K ] for which Π is non-degenerate, the conditional probabilities are not almost surely equal for at least one feature. Therefore,

<!-- formula-not-decoded -->

so the MIB property is satisfied.

Proposition 7. Let L i ( { f i } m i =1 ) denote the value of L i in Algorithm 2 when all agents use { f i } m i =1 ∈ F m . Then, for any f i ∈ F , E [ L i ( { I } m i =1 )] ≤ E [ L i ( f i , { I } j = i )] .

Proof. By the definition of Algorithm 2 we have

̸

<!-- formula-not-decoded -->

̸

By definition E [ F Z k i ( T k i ) ∣ ∣ X i, 1 , . . . , X i, | Y i | , T k i ] is ( X i, 1 , . . . , X i, | Y i | , T k i ) -measurable, so there exists a measurable function g : X | Y i | × R → R such that

<!-- formula-not-decoded -->

The conditional expectation

<!-- formula-not-decoded -->

is shorthand for g ( Y i, 1 , . . . , Y i, | Y i | , T k i ) . Since we assume f i is measurable,

<!-- formula-not-decoded -->

is ( X i, 1 , . . . , X i,n i ) -measurable. Therefore, we know that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Repeatedly applying this argument for each feature gives us

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Proposition 8. For n = ( n 1 , . . . , n m ) ∈ N m let L i ( n, { I } m i =1 ) denote the value of L i in Algorithm 2 when agent j ∈ [ m ] has n j data points and agents use { I } m i =1 ∈ F m . Then

<!-- formula-not-decoded -->

where G k j = σ ( X i, 1 , . . . X i,j , T k i ) .

Proof. By the definition of Algorithm 2

<!-- formula-not-decoded -->

Let U k = F Z k i ( T k i ) . From the equation above, the tower property and definition of conditional variance tell us that

<!-- formula-not-decoded -->

An analogous argument gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the tower property now lets us conclude that

<!-- formula-not-decoded -->

Proposition 9. Let L i ( { I } m i =1 ) denote the value of L i when all agents are truthful in Algorithm 2. Then, 0 ≤ E [ L i ( { I } m i =1 )] ≤ 1 4 ( 1 | X i | + 1 | Z i | ) . Moreover, if ∀ k ∈ [ K ] , P k = P ◦ ( φ k ) -1 is a.s. continuous, then 1 6 | Z i | ≤ E [ L i ( { I } m i =1 )] ≤ 1 6 ( 1 | X i | + 1 | Z i | ) .

Proof. By definition

<!-- formula-not-decoded -->

Define X k i = ( φ k ( X i,j ) ) n i j =1 . We have F X k i ( T k i ) is ( X i, 1 , . . . , X i,n i , T k i ) -measurable. Therefore, Lemma 5 tells us that

<!-- formula-not-decoded -->

Applying the first part of Lemma 2 to the inner expectation gives

<!-- formula-not-decoded -->

so we conclude

<!-- formula-not-decoded -->

For the second part, when we assume P k is a.s. continuous, we can apply the second part of Lemma 2 to get

<!-- formula-not-decoded -->

so the upper bound improves to

<!-- formula-not-decoded -->

For the lower bound, note that σ ( X i, 1 , . . . , X i,n i , T k i ) ⊆ σ ( X i, 1 , . . . , X i,n i , T k i , P k ) so Lemma 5 gives us that

<!-- formula-not-decoded -->

Now appealing to Lemmas 3 then 1 gives

<!-- formula-not-decoded -->

Therefore, we conclude that 1 6 | Z i | ≤ E [ L i ( { I } m i =1 )] .

## D Proofs omitted from Section 3

Theorem 3. The mechanism in Algorithm 3 is 1 4 ( 1 | X i | + | W i | + 1 | Z i | ) -approximately truthful in both the Bayesian and frequentist settings. Moreover, if there is a feature k ∈ [ K ] , for which Π is not degenerate, then Algorithm 3 satisfies MIB in the Bayesian setting. If it is not the case that C ⊆ { P ∈ M 1 ( X ) : ∀ k ∈ [ K ] , P ◦ ( φ k ) -1 ∈ δ x , x ∈ R } then Algorithm 3 satisfies MIB in the frequentist setting.

Proof. For 1 4 ( 1 | X i | + | W i | + 1 | Z i | ) -approximate truthfulness we refer to Proposition 11.

Let P k = P ◦ ( φ k ) -1 . For MIB we first look at the Bayesian setting and then the frequentist setting.

For the Bayesian setting, from the assumption about Π we know that ∃ k ∈ [ K ] where ∀ n i ∈ N it is not the case that

<!-- formula-not-decoded -->

Nownotice that this implies that for at least one of the k ∈ [ K ] features, P ( P k ∈ { δ x : x ∈ X} ) &lt; 1 , or else the conditional probabilities above would automatically be equal for each k ∈ [ K ] .

We know from Proposition 12 that

<!-- formula-not-decoded -->

But notice that

<!-- formula-not-decoded -->

implies 1 K ∑ K k =1 E [ F P k ( T k i ) ( 1 -F P k ( T k i ))] &gt; 0 . Therefore

<!-- formula-not-decoded -->

which proves MIB for the Bayesian setting.

For the frequentist setting, we have from Proposition 13 that

<!-- formula-not-decoded -->

If it is not the case that

<!-- formula-not-decoded -->

then so we find

<!-- formula-not-decoded -->

which proves MIB for the frequentist setting.

<!-- formula-not-decoded -->

Proposition 10. Let L i ( { I } m i =1 ) denote the value of L i when all agents follow { I } m i =1 ∈ F m in Algorithm 3. Then,

<!-- formula-not-decoded -->

Moreover, if ∀ k ∈ [ K ] , P k = P ◦ ( φ k ) -1 is a.s. continuous, then

<!-- formula-not-decoded -->

Proof. By the definition of Algorithm 3 we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The first part of Lemma 2 tells us that in both the frequentist and Bayesian setting,

<!-- formula-not-decoded -->

so we find

<!-- formula-not-decoded -->

When ∀ k ∈ [ K ] , P k is a.s. continuous, we apply the second part of Lemma 2 to get

<!-- formula-not-decoded -->

in both the frequentist and Bayesian setting. Under this additional hypothesis, we get

<!-- formula-not-decoded -->

Proposition 11. Let L i ( { f i } m i =1 ) denote the value of L i in Algorithm 3 when agents use { f i } m i =1 ∈ F m . Let Π be a Bayesian prior, and C ⊆ M 1 ( X ) a class of X -valued distributions. Then, for any f i ∈ F

̸

<!-- formula-not-decoded -->

̸

where ε = 1 4 ( 1 | X i | + | W i | + 1 | Z i | ) . Moreover, if ∀ k ∈ [ K ] , P k = P ◦ ( φ k ) -1 is a.s. continuous in the Bayesian setting and ∀P ∈ C in the frequentist setting, then the above inequalities hold with ε = 1 6( | X i | + | W i | ) .

Proof. The first part of the claim, when ε = 1 4 ( 1 | X i | + | W i | + 1 | Z i | ) , follows immediately from Proposition 10 and recognizing that

̸

̸

<!-- formula-not-decoded -->

Now consider when ∀ k ∈ [ K ] , P k is a.s. continuous, where P has either been fixed in the frequentist setting or drawn in the Bayesian setting. By the defintion of Algorithm 3 we have

̸

<!-- formula-not-decoded -->

Thus in the Bayesian setting we have

<!-- formula-not-decoded -->

̸

To get a lower bound we apply Lemma 5 followed Lemmas 3 then 2 which give

<!-- formula-not-decoded -->

In the frequentist setting, independence and Lemma 1 give us

<!-- formula-not-decoded -->

Therefore,

̸

̸

<!-- formula-not-decoded -->

Together we have the following lower bound in both the frequentist and Bayesian setting

̸

̸

<!-- formula-not-decoded -->

From part two of Lemma 2 we have that when agents submit truthfully

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Combining this with the lower bounds, we conclude

<!-- formula-not-decoded -->

which completes the proof.

Proposition 12. For n = ( n 1 , . . . , n m ) ∈ N m let L i ( n, { f i } m i =1 ) denote the value of L i in Algorithm 3 when agent j ∈ [ m ] has n j data points and agents use { f i } m i =1 ∈ F m . Then where P k =

<!-- formula-not-decoded -->

Proof. By the definition of Algorithm 3 we have

<!-- formula-not-decoded -->

Let F P k be the CDF for P k . We start by rewriting each term in the sum above as

<!-- formula-not-decoded -->

Following the same steps in Lemma 2 up to equation (5) gives us

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

̸

̸

The same argument gives an analogous result for E [ L i ( n + e i , { I } m i =1 )] . Taking the difference we find

<!-- formula-not-decoded -->

Proposition 13. For n = ( n 1 , . . . , n m ) ∈ N m let L i ( n, { f i } m i =1 ) denote the value of L i in Algorithm 3 when agent j ∈ [ m ] has n j data points and agents use { f i } m i =1 ∈ F m . Then

<!-- formula-not-decoded -->

Proof. By the definition of Algorithm 3 we have

<!-- formula-not-decoded -->

Let F P k be the CDF for P k . Following the same steps in Lemma 2 up to equation (5) gives us

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

The same argument gives an analogous result for E [ L i ( n + e i , { I } m i =1 )] . Applying this to each feature, we find

<!-- formula-not-decoded -->

## E Examples of the conditional expectation in Algorithm 1

## E.1 The normal-normal model

<!-- formula-not-decoded -->

̸

̸

where

<!-- formula-not-decoded -->

Proof. Start by noticing that the conditional expectation can be rewritten as

<!-- formula-not-decoded -->

where the last line follows from the tower property. By the definition of our model we know that

<!-- formula-not-decoded -->

Recall from standard normal-normal conjugacy arguments that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we can write (2) as

<!-- formula-not-decoded -->

where ϕ ˜ µ, ˜ σ 2 is the PDF of a normal distribution with mean ˜ µ and variance ˜ σ 2 . Recall the following Gaussian integral formula

<!-- formula-not-decoded -->

By the change of variables x = µ -˜ µ ˜ σ we get Φ ( T -µ σ ) = Φ ( T -˜ µ σ -˜ σ σ x ) , so applying the formula gives us

Therefore,

<!-- formula-not-decoded -->

## E.2 The beta-bernoulli model

Proposition 15. Suppose that { f i } j = i = { I } j = i in Algorithm 1. Let p ∼ Beta ( α, β ) and X i = { X i, 1 , . . . , X i,n i } , Z i = { Z i, 1 , . . . , Z i, | Z i | } , where X i,j , T, Z i,j | p i.i.d. ∼ Bern( p ) , then

̸

̸

<!-- formula-not-decoded -->

Proof. Start by noticing that the conditional expectation can be rewritten as

<!-- formula-not-decoded -->

The law of total probability tells us that

<!-- formula-not-decoded -->

We now consider two cases based on whether T is 0 or 1. When T = 1 , (3) becomes

<!-- formula-not-decoded -->

When T = 0 , recall from standard Beta-Bernoulli conjugacy arguments that

<!-- formula-not-decoded -->

Also observe that when T = 0 , P ( Z i, 1 ≤ T | p, X i, 1 , . . . , X i,n i , T ) = 1 -p . Therefore, (3) becomes

<!-- formula-not-decoded -->

Recall that if Z ∼ Beta ( α 0 , β 0 ) then

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

so we conclude that

<!-- formula-not-decoded -->

Putting both cases together gives us

<!-- formula-not-decoded -->

## F Proofs of Technical results

In this section we derive a series of technical results which aid in the main proofs.

Lemma 1. Let P ∈ M c 1 ( R ) be a continuous probability distribution over R , and X = { X 1 , . . . , X n } , where X i , T i.i.d. ∼ P . Then

<!-- formula-not-decoded -->

Proof. Notice that for a fixed t ∈ R ,

<!-- formula-not-decoded -->

Using this observation and noticing that 1 { X i ≤ T } | T ∼ Bern ( F P ( T )) gives

<!-- formula-not-decoded -->

Since P is continuous, the probability integral transform (Lemma 6) tells us that if we set U := F P ( T ) then U ∼ Unif (0 , 1) . The above equation can now be written as

<!-- formula-not-decoded -->

which concludes the proof.

Lemma 2. Let P ∈ M 1 ( R ) be a probability distribution over R , and X = { X 1 , . . . , X n } , Y = { Y 1 , . . . , Y m } where X i , Y i , T i.i.d. ∼ P . Then

<!-- formula-not-decoded -->

Moreover, when P ∈ M c 1 ( R )

<!-- formula-not-decoded -->

Proof. We start with proving the inequality. Let F P ( t ) be the CDF of P . Notice that for a fixed t ∈ R , E [ F X ( t )] = 1 n ∑ n i =1 E [ 1 { X i ≤ t } ] = F P ( t ) . Together with independence we have

<!-- formula-not-decoded -->

Given T , F X ( T ) and F Y ( T ) are sums of i.i.d. bernoulli random variables, thus

<!-- formula-not-decoded -->

since F P ( T ) ∈ [0 , 1] .

For the equality, we rewrite (4) and apply Lemma 1 twice to get

<!-- formula-not-decoded -->

Lemma 3. Let Π ∈ M 1 ( M 1 ( R )) be a distribution over the collection of R -valued distributions. Suppose that P ∼ Π and then X = { X 1 , . . . , X n } , Y = { Y 1 , . . . , Y m } where X i , Y i , T i.i.d. ∼ P . Let F P ( t ) be the CDF of P . Then,

<!-- formula-not-decoded -->

Proof. Using conditional independence we have

<!-- formula-not-decoded -->

Lemma 4. Let F ⊆ G , suppose X ∈ L 2 , and define Y = E [ X |G ] then

<!-- formula-not-decoded -->

Proof. Applying the law of total variation with respect to F and G gives us

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Subtracting (6) from (7) gives

<!-- formula-not-decoded -->

Now notice that by the tower property we have

<!-- formula-not-decoded -->

Combining this with another application of the law of total variation yields

<!-- formula-not-decoded -->

Plugging this into the right hand side of (8) gives us

<!-- formula-not-decoded -->

## G Known results

In this section we present two well known results and give proofs of them for completeness.

Lemma 5 (Durrett [46] Theorem 4.1.15) . Let X be a random variable such that E [ X 2 ] &lt; ∞ and F be a σ -algebra on the underlying probability space. Then E [ X |F ] is the F -measurable random variable Y which minimizes E [ ( X -Y ) 2 ] .

Proof. Notice that if Z is F -measurable and E [ Z 2 ] &lt; ∞ then Z · E [ X |F ] = E [ Z · X |F ] which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now suppose that Y is F -measurable and E [ Y 2 ] &lt; ∞ , and define Z = E [ X |F ] -Y . Then,

<!-- formula-not-decoded -->

which implies that the mean squared error is minimized when Y = E [ X |F ] .

Lemma6 (Probability integral transform) . Suppose that X is a continuous R -valued random variable. Let U = F X ( X ) , i.e. the CDF of X evaluated at X . Then U ∼ Unif (0 , 1) .

Proof. As F X ( t ) may not be strictly increasing, define the generalized inverse CDF ˜ F -1 ( u ) = inf { t ∈ R : F X ( t ) ≥ u } . Now notice that we can write the CDF of U as

<!-- formula-not-decoded -->

from which we conclude that U ∼ Unif (0 , 1) .

Rearranging we find
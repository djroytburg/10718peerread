## The Computational Advantage of Depth in Learning High-Dimensional Hierarchical Targets

Yatin Dandi 1,2 , Luca Pesce 1 , Lenka Zdeborová 2 , and Florent Krzakala 1

1 Information, Learning and Physics Laboratory. Ecole Polytechnique Fédérale de Lausanne (EPFL), CH-1015 Lausanne, Switzerland.

2 Statistical Physics of Computation Laboratory. Ecole Polytechnique Fédérale de Lausanne (EPFL), CH-1015 Lausanne, Switzerland.

## Abstract

Understanding the advantages of deep neural networks trained by gradient descent (GD) compared to shallow models remains an open theoretical challenge. In this paper, we introduce a class of target functions (single and multi-index Gaussian hierarchical targets) that incorporate a hierarchy of latent subspace dimensionalities. This framework enables us to analytically study the learning dynamics and generalization performance of deep networks compared to shallow ones in the high-dimensional limit. Specifically, our main theorem shows that feature learning with GD successively reduces the effective dimensionality, transforming a high-dimensional problem into a sequence of lower-dimensional ones. This enables learning the target function with drastically less samples than with shallow networks. While the results are proven in a controlled training setting, we also discuss more common training procedures and argue that they learn through the same mechanisms.

Understanding the computational benefits of deep neural networks over their shallow counterparts is a central question in modern machine learning theory [76, 87]. While shallow models can approximate any complex functions [27], deep networks almost universally exhibit remarkable advantages in practice [49, 4]. There has been much progress in approximation theory on the advantage of depth (see e.g. [62, 79, 61, 68] and reference therein), however, the dynamics of learning with gradient descent is a more complex question. A fundamental open problem is thus:

Can one quantify the computational advantage of deep models trained with gradient-based methods with respect to shallow models in some analyzable setting?

One line of work on GD-based methods in deep networks leading to interesting results is in the setting of deep linear network -see e.g. [73, 46, 12, 51, 40]. While deep linear networks offer valuable insights into nonlinear learning dynamics, their simplicity renders them insufficient to capture the complexity of hierarchical feature learning.

Another popular line of research is to study the dynamics of gradient-based methods learning multiindex functions with shallow models [19, 17, 39, 22, 1, 81]. Multi-index functions provide a rich class of targets, but their efficient learnability by shallow two-layer networks [11, 52] undermines their utility as benchmarks for understanding the computational advantages of depth. This motivates the following consideration:

What is the natural model of targets to consider for understanding the emergent computational advantage of depth when training with gradient-based methods?

The present paper addresses both these questions. To answer the latter, we introduce a class of target functions designed to probe the hierarchical structure and computational potential of deep networks. These Multi-Index Gaussian-Hierachical Target (MIGHT) functions encapsulate a hierarchy of latent subspaces with varying dimensionalities. We then proceed to answer the former interrogative

by analyzing the learning dynamics of multi-layer neural networks on such targets, providing a characterization of the computational advantages afforded by depth. We show how depth enables a hierarchical decomposition of tasks, reducing the effective dimensionality at each layer, and leading to a quantifiable improvement in sample complexity over shallow models.

## 1 Hierarchical Targets and Main Results

## 1.1 Single-Index Gaussian Hierarchical Targets

Our simpler setting, where the task -using Gaussian i.i.d. data { x µ } n µ =1 ∈ R n × d - to learn the following Single-Index Gaussian Hierarchical Target (SIGHT) function class that we write in three equivalent forms as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here P k is a fixed polynomial applied component-wise, and d ε 1 denotes the dimensionality of the second-layer features (non-linear features) in the intermediate layer, which we choose to be ε 1 ∈ (0 , 1) . The first-layer features (linear features) are z ⋆ = W ⋆ x , where W ⋆ ∈ R d ε 1 × d has orthonormal unit vectors as rows, and a ∗ ∈ R d ε 1 is choosen randomly from a fixed distribution. We refer to the variable h ∗ as the index in the name of the class. This construction, a generalization of the hidden manifold model [43], is motivated by the compositional structure present in real-world functions and by the analysis carried over by [84, 66]. The strictly decreasing dimensionality of the features across depth allows us to avoid the pitfall of the original hidden manifold model [43] that turns out to be equivalent to a Gaussian linear target [41, 45, 64].

<!-- formula-not-decoded -->

## 1.2 Multi-Index Gaussian Hierarchical Targets

A simple generalization of the above construction is to include many non-linear features, leading to Multi-Index Gaussian Hierarchical Targets (MIGHT) defined as:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with now r directions, each with their own layer weights ( a m and W ⋆ m ), and polynomials ( P k,m ).

## 1.3 Deep Multi-Index Hierarchical Targets

Finally, we define the deep version of MIGHTs as

<!-- formula-not-decoded -->

with Gaussian data { x µ } n µ =1 ∈ R d , and where each features h ⋆ ℓ ( x ) ∈ R d ε ℓ are recursively defined as:

<!-- formula-not-decoded -->

with ℓ = 1 · · · L , m = 1 · · · d ε ℓ and where a ⋆ ℓ,m ∈ R d ε ℓ -1 -ε ℓ acts on the m th block of the previous layer feature h ⋆ ℓ -1 ( x ) (each of them being of size d ε ℓ -1 -ε ℓ ). Again P k,m,ℓ are fixed polynomials for ℓ =1 , · · · , L -1 ; d ε ℓ denotes the dimensionality of the features at layer ℓ , which we choose to be strictly decreasing across depth, i.e., 1 &gt; ε 1 &gt;ε 2 &gt; · · · &gt;ε L -1 &gt; 0 , with h ⋆ L ∈ R r being finite-dimensional. This "tree-like" construction ensures that for any layer index ℓ ∈ 1 , · · · , L , the hidden features h ⋆ ℓ,m ( x ) remain independent for different index m ∈ 1 , · · · , d ε ℓ . (Appendix D.1)

Finally, the 1 st -layer features are defined as

<!-- formula-not-decoded -->

where W ⋆ ∈ R d ε 1 × d has orthonormal unit vectors as rows. By explicitly incorporating multiple levels of non-linear feature transformations, each associated with a progressively reduced latent dimensionality, it models the deep hierarchical structure is a feature of complex real-world tasks, see e.g. [55, 49, 65, 24, 74]. We exemplify SIGHT (1) and MIGHT (4) functions in Fig. 3, and their deep version (6), where the tree structure of the deep version of these targets is apparent, in Fig. 4.

## 1.4 Learning Model

We now consider learning SIGHT and MIGHT functions f ⋆ ( x ) through an L -layer neural network, that is a standard multi-layer perceptron:

<!-- formula-not-decoded -->

where θ denotes the ensemble of trainable parameters { b ℓ , W ℓ , ℓ = 1 · · · L } . The hidden layer weights have dimension W ℓ ∈ R p ℓ × p ℓ -1 for ℓ ∈ { 2 , · · · , L -1 } with readout layer W L ∈ R p L and first layer W 1 ∈ R p 1 × d , and the biases b ℓ are in R p ℓ . We shall consider Empirical Risk Minimization (ERM) of the square loss ˆ R ( { x µ } ) = ∑ n µ =1 ( f ∗ ( x µ ) -ˆ f θ ( x µ ) ) 2 with gradient descent.

## 1.5 Main Results in a Nutshell

Figure 1: An illustration of the phase transitions in learning SIGHT according to the main Theorem 1 denoting the computational advantage of depth for two different target model: (a) generic shallow SIGHT function (eq. (1)) and (b) the example in eq. (11).

<!-- image -->

trained with Gradient Descent on Gaussian data, as both n (the number of data) and d (the dimension of the data) grow to infinity. We unveil a series of sharp thresholds in the sample complexity ratio κ = log n log d where neural networks learn the target with increasing accuracy. To summarize:

The backbone of our results is the analysis of the asymptotic performance of learning SIGHT and MIGHT functions using multi-layer networks

- Our targets offer a solvable playground to unveil the computational advantage of deep networks over shallow ones. The learning mechanism can be viewed as the reduction of the 'effective dimension" in which networks trained on f ⋆ ( x ) successively reduces the dimensionality of the search space: d ε 1 → d ε 2 → d ε 3 , · · · , → r. Depth acts as a progressive filter that distills data into lower-dimensional representations (a coarse-graining mechanism akin to renormalization in physics), enabling the learning of subsequent layers.
- We further explore the problem through numerical simulations using more realistic training procedures than those covered by the theorems. Our results suggest that the dimensionality reduction mechanism remains broadly applicable. Notably, we illustrate such a phenomenon using 3-layer networks training all the layers jointly with standard backpropagation. We provide the code of our simulations at https://github.com/IdePHICS/ComputationalDepth.
- We focus the rigorous analysis in the paper on the case of shallow SIGHT functions (eq. (1)) learned by 3 -layer networks, where each layer is trained sequentially and independently. We prove that a three-layer network trained in a layer-wise fashion can learn a SIGHT function f ⋆ ( x ) efficiently. Specifically, the network first recovers W ⋆ using ˜ O ( d ε 1 +1 ) samples, then reconstructs h ∗ with ˜ O ( d kε 1 ) samples (with k denoting the degree of P k , in case kε 1 &lt; 1 + ε 1 both happen at 1 + ε 1 ), and finally fits f ⋆ as a function of h ⋆ using only ˜ O (1) samples. This sample complexity aligns with predictions from the dimension-reduction/coarse-graining perspective, where earlier layers successively reduce the effective dimensionality of the learning problem. We also present additional results for deeper targets and networks.

## 1.6 Related Works

Random Feature Models A key attribute enabling the effectiveness of neural networks is their ability to adjust to low-dimensional features present in the training data. However, interestingly, much of the current theoretical understanding of neural networks comes from studying their lazy regime, where features are not learned during training. One of the most pre-eminent examples of such 'fixed-features' regimes are Random Feature (RF) models, initially introduced as a computationally efficient approximation to kernel methods by [69], they have gained attention as models of two-layer neural networks in the lazy regime. One of the main motivations is their sharp generalization guarantees in the high-dimensional limit [38, 42, 58, 59, 86, 34]. As mentioned, however, the performance of such methods, and of any kernel method in general, is limited. A fundamental theorem in [58] states that only a polynomial approximation up to degree κ RF of any target f ∗ , with κ RF = min( κ 1 , κ 2 ) when learning with n = d κ 1 data and p = d κ 2 features.

While even shallow networks can surpass these limitations [40, 17, 31], this relation for κ RF plays a fundamental role in our analysis.

Multi-index Models Despite the theoretical successes in describing fixed feature methods, the holy grail of machine learning theory remains a rigorous description of network adaptation to low-dimensional features. A popular model to study such low-dimensional structure in the learning performance is the multi-index model . For this class of target (denoted as f ⋆ MI ), the input datum x is projected on a r -dimensional subspace W ⋆ = { w ⋆ j , j ∈ 1 · · · r } and the input-output relation depend solely on a non-linear map g ⋆ of these r (linear) features :

<!-- formula-not-decoded -->

While the information theoretical performance is well understood [18, 14], there has been intense theoretical scrutiny to characterize the sample complexity needed to learn multi-index models with shallow models. On the one hand, kernel methods can only learn a polynomial approximation [58]; on the other hand, the situation in neural networks appears more complicated at first as the hardness of a given f ⋆ MI has been characterized by the 'information' and 'leap' exponents [19, 2, 30, 28, 32, 11, 52, 21, 77, 13]. It was shown, however, that simple modification of vanilla Stochastic Gradient Descent (SGD), such as Extra-Gradient methods or Sharpness Aware Minimizers, are able to attain sample complexity corresponding to Statistical Query (SQ) lower bound [11, 52], and are essentially optimal up to polylog factors in the dimension [28, 81]. A motivation of the present work is to go beyond such limitations and analyze hierarchical feature learning.

3-Layers Networks Substantial effort has been devoted to investigating the approximation advantages conferred by deeper neural network architectures [79, 36, 70]. However, it remains unclear how these approximation gaps translate into sample complexity ones for neural networks when trained through gradient descent. An important step towards the role of depth in neural networks has been carried over by [84, 66], who proved separation results between the test performance of 2 &amp; 3 layer networks. More precisely, [84] proved that 3-layer architectures with a fixed first layer can learn a target function of the form g ⋆ ( x ⊤ A x ) in n = ˜ O ( d 4 ) samples through a single-gradient step on the second layer, where x ∈ R d and A ∈ R d × d . In contrast, 2-layer networks require a super-polynomial number of samples in terms of the degree of g ⋆ . [66] subsequently improved the sample complexity to ˜ O ( d 2 ) and generalized the result to functions of p th -order polynomials. [37] further extended these results to learning multiple-nonlinear features. We go beyond these results to prove stronger separation results by analyzing fully trained networks without a fixed first layer.

Coarse-graining The dimensionality reduction we describe is closely related to the concept of learning features across different scales. This idea has been explored in the context of machine learning through connections with the renormalization group [85] in physics, where each scale corresponds to a distinct set of features. Such techniques have inspired studies of deep neural networks [57, 53, 56]. Here, we present a concrete example of such a coarse-graining mechanism, illustrating how hierarchical structures can be analyzed explicitly.

Hierarchical data models A key insight in explaining the superiority of deep over shallow networks is that depth enables neural networks to progressively reduce the effective dimensionality of the learned data representation [80, 5, 3, 72, 8, 35, 23]. This aligns with the latent hierarchical structure observed in real-world data, which deep models exploit through layer-wise composition. Leveraging these observations, the construction of hierarchical data models has been central in theoretical analysis [65, 6, 2, 74, 75, 24, 23]. Tree-like structures analogous to our SIGHT (1) and MIGHT (4) are considered in [68], leading to provable approximation benefits across depth. More generally, [67] linked compositional sparsity to efficient computability. Since learnability subsumes computability, such computational sparsity is expected to be necessary for efficiently learnable functions, further supporting our construction. However, since efficiently learnable functions form a strict subset of computable functions, functions learnable by gradient descent must possess additional structure on top of being compositions of local/sparse functions. Our class of targets shows that one such additional structure is obtained by insuring sufficient regularity/stability w.r.t intermediate features at each step. In our setting, such regularity is ensured by the presence of low-degree dependence on lower level features. This mirrors the dependence structure in real data, where for instance, the target labels for images/language datapoints have direct correlations with low-level features such as edges or bi-gram, trigram counts. Results supporting the benefits of deepth for tree-like hierarchical models have been provided by [65] and [23] who consider tree-structured inputs, in contrast to our focus on structured targets (but non structured input).

Universality A crucial role in our analysis is played by the asymptotic Gaussianity of h ⋆ ℓ ( x ) which leads to a simplified description of how dependencies on h ⋆ ℓ ( x ) propagate to lower-level features. Such a property is a crucial component of the analysis in [66, 84]. Specifically, [66, 84] showed that the projection of g ⋆ ( ⟨ He k ( x ) , A ⟩ ) on degreek Hermite polynomials lies along the non-linear feature ⟨ He k ( x ) , A ⟩ while g ⋆ has vanishing projections on lower degree terms. We generalize these results to describe the projections on all degree components.

## 2 Heuristic argument underlying the main results

Before presenting the main technical results, we describe here a heuristic argument describing the narrative behind the results. For concreteness, we focus here on learning a shallow SIGHT function (1) as a first step toward a broader understanding. For concreteness, we will discuss the following example (later used in Fig. 2):

<!-- formula-not-decoded -->

with a polynomial P 3 ( x ) = He 2 ( x )+He 3 ( x ) (the second and third Hermite polynomials), ε 1 = 1 / 2 , and discuss the performance of different learning architectures, highlighting the dimensionality reduction due to feature learning. The learning dynamics for general SIGHT (eq. (1)) and the particular example above (eq. (11)) are illustrated in Fig. 1 respectively in the top and bottom panel.

a) Kernel methods, or random feature models , can only learn a polynomial approximation of degree κ in the Hermite basis of f ⋆ if n = O ( d κ ) [58]. This is a strong limitation that leads to poor performance as the learning method is not sensitive to the presence of relevant low-dimensional structure, but rather only to the degree of the target. In the example (11), the lowest (Hermite) polynomial order is quadratic in x (as can be seen by expanding the tanh ): learning it thus requires n = O ( d 2 ) samples of data for a kernel method to beat random performance. Learning the cubic approximation would requires n = O ( d 3 ) samples, etc. The corresponding thresholds are sketched in orange in Fig. 1.

b) We now turn to two layer net of the form (we do not write explicitly the additional biases for clarity) with a number of neurons p at least of order Θ( d kε 1 + δ )

<!-- formula-not-decoded -->

Thanks to feature learning, such architecture should perform better: Indeed, for W 1 to learn the d × d ε 1 first-layer feature matrix W ⋆ , we need at least n = O ( d × d ε 1 ) data. If n ≫ d 1+ ε 1 , we thus expect that W 1 correlates with W ⋆ . Intuitively, W 1 is then close to a noisy random rotation of W ⋆ and behaves roughly as W 1 ≈ Z 1 W ⋆ + Z 2 (with Z 1 and Z 2 are essentially random matrices). The two-layer neural net thus now behaves as:

<!-- formula-not-decoded -->

Fitting now the outer weights w 2 leads, once again, to a random feature model, but now applied to the target eq. (2) seen as a function of z instead of eq (1) seen as a function of x . This leads to an effective Random Feature model with respect to the lower dimensional vector { z ⋆ ∈ R d ε 1 } . Thanks to this dimensional reduction from dimension d to the effective one d ε 1 , we just need n = ( d ε 1 ) κ samples of data to now fit a κ -th degree polynomial approximation of f ⋆ . This is a drastic improvement. Coming back to the example: with n = O ( d 1+ ε 1 =1 . 5 ) , κ = 1 . 5 , data samples a two-layer net learns the first layer representation W ⋆ , leading to a dimensionality reduction from d to √ d . From n = O ( d 3 ε 1 =1 . 5 ) , κ = 1 . 5 , we are also able to fit a (Hermite) polynomial approximation of degree 3 of the target viewed as a function of z . The next order in the expansion of (11) is power 6 in z , and thus will be fitted at κ = 3 . We discuss the extension of the above arguments to two-layer networks trained with a general gradient-based algorithm in App. B . c) We now finally consider a three-layer neural networks , with width p 2 = p 1 = Θ( d kε 1 + δ ) :

<!-- formula-not-decoded -->

We still expect that W 1 learns the first-layer features W ⋆ when n ≫ d 1+ ε 1 , at which point:

<!-- formula-not-decoded -->

However, contrary to the previously depicted shallow case, three-layer networks can further approximate h ⋆ by updating the second layer. With each power of d ε 1 we expect to be fit an

additional power approximation of h ⋆ and, in particular, with n = O ( d kε 1 ) , we expect the second layer preactivation h 2 ( x ) = W 2 σ ( W 1 x ) to correlate completely with the ( k -polynomial) features h ⋆ . Therefore, denoting again Z 4 , Z 5 as random matrices, a 3-layer network now acts as:

<!-- formula-not-decoded -->

Fitting now w 3 leads to a random feature model on the scalar h , which can be fitted perfectly with any growing number of samples n . In other words, through successive coarse-graining from d eff = d → √ d → 1 , we have reduced the dimension from a diverging one ( d ) to a finite one.

Note that generalization error as plotted in Fig. 1 can jump for two reasons as n increases: either because of a reduction of the dimension d eff , or because of an increase of polynomial fitting power within this dimension. The phenomenology is a bit simpler in the particular example (11), where the advantage of a three-layer net is considerable: for n = O ( d 1 . 5 ) , the network learns to represent the non-linear features h ⋆ directly, and thus can learn the entire function.

While such parameter counting sounds reasonable, this heuristic may fail for general data distributions, as high-degree polynomials may localize on low-dimensional structures and develop heavy tails. However, for Gaussian and spherical measures, isotropy and hypercontractivity ensures that such polynomials remain delocalized and well-concentrated [Lemma 5]. Our analysis relies on proving that such a property holds under feature learning, and even for deep non-linear hidden features.

This scenario, illustrated in Fig. 1, extends mutatis mutandis to generic deep multi-layer MIGHT functions, where a sequence of transitions emerges progressively across the layers. Consider for instance the following hierarchical target function from eq. (7) (see also Fig. 4):

<!-- formula-not-decoded -->

In this case we expect a reduction from d → d ε 1 → d ε 2 → 1 . The first one arises at n = O ( d ε 1 +1 ) when learning W ⋆ , then at n = O ( d kε 1 + ε 2 ) (to learn all the d ε 2 polynomials, each of them requiring d kε 1 data) and finally at n = O ( d k ′ ε 2 ) to learn the activation in the tanh (a single k ′ polynomial in dimension d ε 2 ). Note that while these must proceed in this order, some of these jumps can happen at the same value of κ . For instance, if k ′ ε 2 &lt;kε 1 + ε 2 , then the last two jumps arise simultaneously.

## 3 Main Theoretical Results

We now turn to the main part of our results that describe learning of the SIGHT and MIGHT function classes with deep neural networks trained by gradient descent. We present a rigorous analysis of gradient-based Empirical Risk Minimization (ERM). Since a complete rigorous analysis of gradient descent in deep networks is extremely challenging - and hitherto elusive- we first present a rigorous description for the SIGHT target of eq. (1) under a specific deep-learning schedule. This approach enables us to provide precise theorems that capture the hierarchical learning process. We analyze the following training procedure:

- Initilization: The parameters of the model ˆ f θ ( x ) = w ⊤ 3 σ ( W 2 σ ( W 1 x )) are initialized as W 1 ,i ∼ U ( S d -1 (1)) for i ∈ [ p 1 ] , W 2 = I p 1 , and w 3 ,i = 1 for i ∈ [ p ] , where U ( S d -1 (1)) denotes the uniform distribution on the unit sphere in R d .
- Neuron-wise spherical projections : While updating the first layer parameters W 1 , we utilize spherical-gradient and project each neuron onto the unit sphere. Such spherical projections are commonly utilized in the literature on two-layer networks [19, 1].
- Layer-wise training : (i) We first perform a pre-determined number T 1 of gradient updates on the first layer W 1 on independent batches of data for each step [30]. (ii) Subsequently, we re-initialize the second layer W 2 do a single large gradient step. (iii) Finally, we update w 3 through ridge regression. Layer-wise training procedures are a common simplifying assumption in the analysis of two-layer networks [29, 1, 30]. A complete analysis of the joint training remains open even for two-layer networks except for training of layers at differing time scales [20, 22]. An interesting direction for future work is to rigorously show separation results between deep and shallow networks by constructing targets where joint training of the layers provably surpasses layer-wise training. A first attempt in this direction was considered in [7], who illustrated the advantage of joint-training through the mechanism of 'backward feature correction".

- Pre-conditioning of gradient for the second layer : We use a pre-conditioning of the gradient step - broadly used in various optimization schedules (e.g. Adam [47])- using the sample-covariance of the features as preconditioning matrix, i.e.,

<!-- formula-not-decoded -->

Through the feature map x → σ ( W 1 x ) , the updates of W 2 in parameter space translate to updates to h 2 ( x ) in function space. Without such pre-conditioning, online SGD leads to a worse sample complexity of Θ( d 2 kε 1 ) as in the single-step analysis of [66], as we explain further in Appendix C.16. Although pre-conditioning plays an important role in the proof scheme, we argue that the core of the results hold in more realistic routine in Sec. 5.

- Uniform weighting : We set { a ⋆ i = 1 for all i ∈ 1 · · · d ε 1 } : This operation ensures isotropic dependence along all components, simplifying the analysis. While a ⋆ i = 1 is a particular choice of target weights, the training algorithm of the model is agnostic to this choice and we, therefore, obtain sample-complexity expected for a general non-linear feature of the form a ⋆ ⊤ P k ( W ⋆ x ) / √ d ε 1 .

With this algorithm, we can now study gradient descent and demonstrate the learning of a class of SIGHT function. The theorem will assume the following conditions:

- : We shall indeed require that the information exponent [19, 2, 1] of

̸

The condition on g ⋆ ( · ) is necessary, as gradient descent (without repetition) has a drastic worst complexity for exponents larger than 2 . We expect however, that the condition on P k ( · ) can be relaxed to information-exponent ≥ 2 instead of being exactly 2 . The information exponent of P k ( · ) being 1 results in linear components that do not require recovery of the full subspace spanned by W ⋆ . Thus setting the information exponent of P k ( · ) to 2 simplifies our analysis by avoiding the need for a separate treatment of such linear 'spikes".

Assumption 1. Let z ∼ N (0 , 1) denote a standard normal variable. We assume that E [ g ⋆ ( z ) z ] = 0 , E [ P k ( z ) He 2 ( z )] = 0 .

- Information exponent g ⋆ ( · ) is 1 and that of P k ( · ) is 2 :

̸

We further require the activations of the neural net σ ( · ) to be sufficiently expressive and to satisfy certain alignment conditions:

̸

Assumption 2. σ : R → E is analytic, non-polynomial with σ ′ (0) = 0 and there exist constants L 1 , L 2 ∈ R + , m ∈ N such that | σ ( x ) | ≤ L 1 + L 2 | x | m . Furthermore, σ ( · ) satisfies: (a) E [ σ ( z ) He j ( z )] = 0 for all 1 &lt; j ≤ k , (ii) E [ σ ( σ ( z )) He 2 ( z )] E [ P k ( z ) He 2 ( z )] &gt; 0 and iii) E [ σ ( σ ( z )) z ] = 0

̸

The last two conditions ensure that all neurons in W 1 recover spherical projections of W ⋆ . In the absence of the above conditions, we still expect recovery of W ⋆ but with anisotropy across neurons. Such an anisotropy is expected to complicate the subsequent analysis. We show in Appendix C.7 the existence of a σ ( · ) satisfying the above set of conditions.

Assumption 3. E z ∼ N (0 , 1) [ g ⋆ ( z ) He j ( z )] = 0 for 1 &lt; j ≤ k

The next assumption, however, is only a technical one that arises only because we used { a ⋆ i = 1 } . It could be relaxed by taking Gaussian values, or by performing more gradient steps on W 2 , but this would complicate the proof. We discuss this in detail in App. C.25:

Under the above assumptions, our main result now establishes hierarchical learning for the target of the form (1) by a three-layer network f ⋆ ( x ) by first recovering W ⋆ through the first-layer W 1 , next recovering h ⋆ ( x ) through the second layer pre-activations h 2 ( x ) = W 2 σ ( W 1 x ) and finally fitting f ⋆ ( x ) upon training the last layer w 3 . The full formal statement of the result is provided in Appendix C.1.

Theorem 1 (Informal) . Let f ⋆ ( x ) be as in Eq. (1) with ε 1 ∈ (0 , 1) and consider a three-layer model:

<!-- formula-not-decoded -->

with W 1 ∈ R p 1 × d , W 2 ∈ R p 2 × p 1 , w 3 ∈ R p 3 .

Let L c ( θ ) denote the correlation loss defined as L cl ( θ ) := -ˆ f θ ( x ) f ⋆ ( x ) . Under Ass. 1-3, for any 0 &lt; δ &lt; δ ′ &lt; 1 , there exist time-steps T 1 = O (polylog d ) such that with batch-size n 1 = Θ( d ε 1 +1+ δ ) , n 2 = Θ( d kε 1 + δ ) and p 2 = p 1 = Θ( d kε 1 + δ ′ ) , the following holds with high probability as d →∞ :

(i) T 1 steps of neuron-wise spherical SGD on correlation-loss L c ( θ ) applied to W 1 with step-size η = ˜ η √ p 2 √ d ε 1 on independent batches of size n 1 results in W 1 learning random projections along

W ⋆ upto error o d (1) . Concretely, there exists a sequence of random matrices Z ∈ R p 1 × d ε 1 with independent rows sampled uniformly on the unit sphere i.e z i ∼ U ( S (1)) :

<!-- formula-not-decoded -->

as d →∞ , ˜ η → 0 .

(ii) Subsequently, upon reinitializing W 2 = 0 d × d and w 3 with entries N (0 , 1) , a single pre-conditioned gradient step on correlation loss L c ( θ ) with step size η 2 = Θ( √ p 2 ) and using an independent size n 2 results in learning h ⋆ upto error o d (1) with the preactivation h 2 ( x ) = W 2 σ ( W 1 x ) ∈ R p 2 :

<!-- formula-not-decoded -->

where c = 0 denotes a constant and the o d (1) error is w.r.t the metric induced by L 2 ( N ( 0 , I d )) . (iii) Upon training W 1 , W 2 as above, updating w 3 with ridge-regression on Θ( d δ ) samples results in approximating f ⋆ ( x ) upto error o d (1) with the 3-layer predictor w ⊤ 3 σ ( W 2 σ ( W 1 x )) .

̸

The details of the initialization projections and preconditioning steps are provided in App. C.5. The condition p 2 = p 1 is again solely to simplify the analysis and we expect the results to hold for p 2 = Θ( d δ ) , p 1 = Θ( d kε 1 + δ ) .

Since each row of W ⋆ j contains d parameters, the complexity n 1 ≈ Θ( d ε 1 +1 ) matches the total number of parameters in W ⋆ 1 , · · · , W ⋆ r , and is therefore expected to be the information-theoretic scaling of the sample-complexity required for the (strong) recovery of W ⋆ 1 , · · · , W ⋆ r . Similarly, the complexity n 2 = Θ( d kε 1 ) is the expected minimum samplecomplexity required for the strong recovery of a degreek functions on a d ε 1 -dim. space.

Proof sketch We provide the full proof of the above result in App. C, and highlight the most important steps below:

- (ii) Low-dimensional dynamics for W 1 : Using the compositional Hermite-decomposition above, following [19, 10, 1], we show that the evolution of W 1 during the training of the first layer can be described through an effective dynamics on the overlaps W 1 ( W ⋆ ) ⊤ . Unlike the single/multi-index analysis of [19, 10, 1], the diverging dimensionality of W ⋆ , W 1 that appear in our approach, as well as the later use of the updated weights W 2 , requires a careful control over the error terms. Concretely, we show that the components of W 1 along W ⋆ , as well as the error terms, maintain isotropy and hypercontractivity through the dynamics. Moreover, such divergent dimensionality d ϵ of W ⋆ leads to 'strong recovery' of W ⋆ by W 1 . We refer to Appendix C.8 for details. (iii) Function-space decomposition of the 2 nd -

<!-- image -->

Figure 2: Numerical simulation: Generalization error versus κ = log n / log d for f ⋆ ( x ) = tanh(3 a ⋆ · P 3 ( W ⋆ x )) / √ d ε 1 =1 / 2 ) with different training protocols: (Top) kernel ridge regression (orange points) only beats the random performance (purple solid line) starting from n = d + ( d -1) d/ 2 , and is limited to quadratic approximation (orange line). 2 -layer net (green points), instead, starts to learn at κ = 1 . 5 (black dashed line) and can beat the quadratic limit (asymptotics is given by the green line). 3-layer net trained with layerwise training (blue markers) not only learn at κ = 1 . 5 (vertical line). but also surpasses the best possible 2-layer net error, illustrating the advantage of depth; (Bottom) comparison of layerwise training (blue) with joint training (red) of all the layers of a 3layer net with standard backpropagation.

- (i) Composition of Hermite decompositions : Building upon [84], we use the asymptotic Gaussianity of h ⋆ ( x ) to relate the Hermite decomposition of f ⋆ ( x ) to the one of h ⋆ ( x ) .

layer pre-activations : Gradient steps on W 2 extract statistics in features-space σ ( W 1 x ) . Similar to [84, 66, 37], we show that these statistics appear in the updates for the pre-activations h 2 ( x ) as projections of a perturbed version of f ⋆ on the conjugate Kernel defined by the first-layer:

<!-- formula-not-decoded -->

where X ∈ R n × d denotes the batch of data utilized in a gradient step and c &gt; 0 denotes a constant. (iv) Concentration of the sample-covariance matrix : In light of ( iii ) , the recovery of features

in h 2 ( x ) depends on the feature matrix σ ( W 1 X ) being able to approximate and span the relevant functional subspace, which requires both sufficiently many samples and sufficiently many neurons. Building on the matrix-concentration analysis of [58], we show that the projections onto the σ ( W 1 X ) up to degreek functions can be well approximated as long as n, p 1 = Θ( d kε 1 + δ ) . Lowdegree eigenfunctions concentrate faster since they span lower-dimensional subspaces.

From SIGHT to MIGHT While we expect similar results to hold in generality, the theorem is only fully proven for the class of target in eq. (1). While a complete proof for MIGHT is a difficult task, we discuss additional ( r &gt; 1 and ℓ &gt; 1 ) results in this and subsequent paragraphs.

MIGHT functions are interesting in illustrating the role of the information exponent in Ass. 1. It is easy to design counterexamples, for instance, the parity problem with y = sign( h ⋆ 1 h ⋆ 2 h ⋆ 3 ) violates Ass. 1. We illustrate some of these numerically in App. 5 (See Fig. 10). We believe, however, that with reusing batches, the information exponent could be replaced with the much permissive generative one [32, 52, 11]. SIGHT and MIGHT functions are indeed generalizations of the multiindex functions, and the properties of the latter such as information [19] and generative exponents [28], and the notion of trivial, easy and hard directions [81]) should translate to the former.

We first remark that part ( i ) of Theorem 1 ( weak-recovery of W ⋆ ), under suitable symmetry assumptions on g ⋆ ( · ) , holds for arbitrary r , and thus for MIGHT functions f ⋆ (and not only SIGHT ones) (see App. E). Establishing rigorously part ( ii ) for r &gt; 1 involves technical hurdles relating to the control in the Gaussian approximation of h ⋆ . We describe them in App. E.

From MIGHT to Deeper MIGHT Depth introduces more difficulties for rigorous studies, but our mathematical analysis can be extended for more general constructions. By the tree-like hierarchical construction of features (Eq. (7)) for general depth, the components h ⋆ ℓ ( x ) remain independent and asymptotically Gaussian. Generalizing Thm. 1 for L ≥ 3 in its full-generality requires however not only an extension of part ( ii ) of Thm. 1 to r &gt; 1 , but also a careful control over the non-asymptotic rates for the tails of h ⋆ ℓ ( x ) and the associated kernels.

Theorem 2. For L ∈ N , let f ⋆ ( x ) denote a target as in Eq. (6) with r = 1 , and let δ ′ , δ be arbitrary reals satisfying 0 &lt; δ &lt; δ ′ &lt; 1 . Consider a model of the form ˆ f θ ( x ) = w ⊤ L σ ( W L -1 σ ( Wh ⋆ L -1 ( x ))) with W ∈ R p L -2 × d ε L -2 having p L -2 = Θ( d kε ℓ -2 + δ ′ ) rows independently sampled as w i ∼ U ( S d ε L -2 (1)) . Under Ass. 1-3, after a single step of pre-conditioned SGD on W L -1 with batchsize Θ( d kε ℓ -2 + δ ) , step-size Θ( √ p L -1 ) , the pre-activations h L -1 ( x ) := W L -1 σ ( Wh ⋆ L -1 ( x )) satisfy, for a constant c &gt; 0 :

We instead prove a weaker, but useful, result corresponding to the hierarchical weak recovery of a single non-linear feature at a general level of depth L ∈ N , under an idealized scenario of perfect spherical recovery of hidden features at level L -1 . We refer to App. D for the full formal statement and its proof, which exploits the independence of components of h ⋆ L -1 ( x ) and the hyper-contractivity of the Gaussian measure:

<!-- formula-not-decoded -->

## 4 General Conjecture for Efficient Hierarchical Learning

Building on the above results, we now propose a general structure for hierarchical learning and conjecture its relevance in broader settings, as we briefly alluded to in Section 1.6 under 'Hierarchical data models". As highlighted in Assumption 1, our analysis requires the target nonlinearities g ⋆ ( · ) and P k ( · ) to have low-degree components (in our case Information Exponents 1 , 2 respectively).

More generally, for any such compositional target to be learnable through gradient descent, we conjecture that for every depth level ℓ = 1 , · · · , L , the intermediate representation h ⋆ ℓ ( x ) retains low-degree correlations with the target y = f ⋆ ( x ) or transformations of y . Concretely, defining the following require that at every layer ℓ : E [ f ⋆ ( x )( h ⋆ ℓ ( x )) ⊗ k ] = Θ(1) for some small k ∈ N . More formally, one may introduce the Compositional Information Exponent :

Definition 1 (Compositional Information Exponent) . Given a SIGHT or a MIGHT function f ⋆ ( x ) , we define the compositional information exponent at level ℓ as:

<!-- formula-not-decoded -->

Therefore we conjecture that compositional target learnable through gradient descent have, at every layer ℓ , low CIE( ℓ ) . Intuitively, even when the mapping from ( h ⋆ ℓ ( x )) to the final label y = f ⋆ ( x ) is highly non-linear, the low-degree correlations with the intermediate representations h ⋆ ℓ ( x ) ensure that it can be recovered through gradient descent. This aligns with the general wisdom

on the structure of real data. The non-trivial correlation of image labels with low-degree features: edges, shapes, and colors, or in the case of text, the correlation between labels and certain wordfrequencies or bi-gram counts (see for instance the discussion in [24]). Equivalently, such low-degree dependence can be interpreted as robustness of y with respect to perturbations in ( h ⋆ ℓ ( x )) , in which case the condition becomes to mantain a robust compositionality. This hypothesis generalises the classical notion of information exponent-originally formulated for the input layer-to all intermediate representations of a hierarchical model. Consequently, within the SIGHT and MIGHT classes, functions that are efficiently learnable by gradient descent are characterised by the presence of such low-degree alignments at each level. When this condition fails, for example, in parity functions whose first non-vanishing Hermite coefficient lies at large degrees, gradient descent fails to efficiently recover any intermediate features.

Beyond the information exponent, we believe this layer-wise information exponent condition can be replaced by a more generic one, at the price of a more complex analysis. For a start, with repeated passes and data reuse, it is natural to expect that the information exponent can be replaced by the generative exponent, as discussed for two-layer networks in [28, 32]. In such settings, the condition can be generalized to:

<!-- formula-not-decoded -->

for some transformation T : R → R and small k ∈ N .

Moreover, it may be possible to replace it by the more permissive 'staircase' picture once one goes beyond layer-wise training analyses. Making this precise is an exciting, but challenging, direction for future work.

## 5 Numerical Illustrations

While our theorems provide a rigorous control of learning with a particular, well-conditioned, training procedure, we numerically test the validity of our theory towards describing realistic training routines with mini-batch updates, finite (and rather low) dimensional examples, using multi-pass (instead of a single one) for the second layer, etc. For concreteness, we consider f ⋆ ( x ) =

tanh ( 3 a ⋆ ⊤ P 3 ( W ⋆ x ) √ d ε 1 =1 / 2 ) , a similar example as discussed in Section 3 and with, again, a polynomial

(i) First we compare the performance of kernel methods with those of a two-layer network. On the one hand, the former method should be able to fit the quadratic part of the target function as soon as n = O ( d 2 ) [60]. This is well observed, with a double descent peak when the number of data hits the number of features in a quadratic kernel, i.e. n peak = d ( d -1) / 2 + d +1 . On the other hand, two-layer networks are capable of recovering W ⋆ when n = O ( d 1 . 5 ) , therefore improving the test performance to quadratic and cubic fit when κ ≥ 1 . 5 .

P k =3 with second and third Hermite polynomials. We show simulations in Fig. 2 and discuss here the most salient observations:

(ii) We then train a three-layer network, with a layerwise approach resembling the procedure in Thm 1, where we train every layer in order, (first W 1 , then W 2 , etc.). We do not, however, follow the restrictions of the theorem and just perform a standard gradient descent (no reinitializing, no projection, using minibatch, etc.). Not only does the method starts to learn when n &gt; d 1 . 5 but it outperforms the 2-layer baseline in agreement with Thm. 1.

We refer to App. F for details on the numerical implementations, along with the analysis of the quality of the learned representations as a function of the sample complexity (see Fig. 9).

(iii) Lastly, we consider the standard training procedure -refered to as joint training - with backpropagation through the network with mini-batch gradient descent. The routine performs similarly to the layerwise approach, illustrating the generality of the dimensionality reduction beyond the assumptions of Thm. 1.

ConclusionWe introduced a theoretical framework for understanding the computational advantages of deep neural networks over shallow models when learning high-dimensional hierarchical functions, where depth facilitates a progressive reduction of effective dimensionality. We hope our paper will spark interest in these directions.

AcknowledgementWe thank Alex Damian, Jason Lee, Bruno Loureiro, Yue M. Lu, Theodor Misiakiewicz, Eshaan Nichani, Tomaso Poggio, Zhichao Wang, Denny Wu, and Mathieu Wyart for insightful discussions. We acknowledge funding from the Swiss National Science Foundation grants SNSF SMArtNet (grant number 212049), OperaGOST (grant number 200021 200390) and DSGIANGO (grant number 225837).

## References

- [1] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics. In The Thirty Sixth Annual Conference on Learning Theory , pages 2552-2623. PMLR, 2023.
- [2] Emmanuel Abbe, Enric Boix-Adsera, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks. In Conference on Learning Theory , pages 4782-4887. PMLR, 2022.
- [3] Alessandro Achille and Stefano Soatto. Emergence of invariance and disentanglement in deep representations. Journal of Machine Learning Research , 19(50):1-34, 2018.
- [4] Ben Adlam, Jaehoon Lee, Shreyas Padhy, Zachary Nado, and Jasper Snoek. Kernel regression with infinite-width neural networks on millions of examples. arXiv preprint arXiv:2303.05420 , 2023.
- [5] Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. Deep variational information bottleneck. In Proceedings of the International Conference on Learning Representations (ICLR) , 2017.
- [6] Zeyuan Allen-Zhu and Yuanzhi Li. What can resnet learn efficiently, going beyond kernels? Advances in Neural Information Processing Systems , 32, 2019.
- [7] Zeyuan Allen-Zhu and Yuanzhi Li. Backward feature correction: How deep learning performs deep (hierarchical) learning. In The Thirty Sixth Annual Conference on Learning Theory , pages 4598-4598. PMLR, 2023.
- [8] Alessio Ansuini, Alessandro Laio, Jakob H Macke, and Davide Zoccolan. Intrinsic dimension of data representations in deep neural networks. Advances in Neural Information Processing Systems , 32, 2019.
- [9] George B Arfken, Hans J Weber, and Frank E Harris. Mathematical methods for physicists: a comprehensive guide . Academic press, 2011.
- [10] Luca Arnaboldi, Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. Online learning and information exponents: The importance of batch size &amp; time/complexity tradeoffs. In International Conference on Machine Learning , pages 17301762. PMLR, 2024.
- [11] Luca Arnaboldi, Yatin Dandi, Florent Krzakala, Luca Pesce, and Ludovic Stephan. Repetita iuvant: Data repetition allows sgd to learn high-dimensional multi-index functions. arXiv preprint arXiv:2405.15459 , 2024.
- [12] Sanjeev Arora, Nadav Cohen, Noah Golowich, and Wei Hu. A convergence analysis of gradient descent for deep linear neural networks. In 7th International Conference on Learning Representations (ICLR) , 2019.
- [13] Gérard Ben Arous, Cédric Gerbelot, and Vanessa Piccolo. Stochastic gradient descent in high dimensions for multi-spiked tensor pca. arXiv preprint arXiv:2410.18162 , 2024.
- [14] Benjamin Aubin, Antoine Maillard, Florent Krzakala, Nicolas Macris, Lenka Zdeborová, et al. The committee machine: Computational to statistical gaps in learning a two-layers neural network. Advances in Neural Information Processing Systems , 31, 2018.
- [15] Guillaume Aubrun and Stanisław J Szarek. Alice and Bob meet Banach , volume 223. American Mathematical Soc., 2017.
- [16] Sheldon Axler. Measure, integration &amp; real analysis . Springer Nature, 2020.
- [17] Jimmy Ba, Murat Erdogdu, Taiji Suzuki, Denny Wu, and Tianzong Zhang. Generalization of two-layer neural networks: An asymptotic viewpoint. In International conference on learning representations , 2020.

- [18] Jean Barbier, Florent Krzakala, Nicolas Macris, Léo Miolane, and Lenka Zdeborová. Optimal errors and phase transitions in high-dimensional generalized linear models. Proceedings of the National Academy of Sciences , 116(12):5451-5460, 2019.
- [19] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. Journal of Machine Learning Research , 22(106):1-51, 2021.
- [20] Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. Foundations of Computational Mathematics , pages 1-84, 2024.
- [21] Alberto Bietti, Joan Bruna, and Loucas Pillaud-Vivien. On learning gaussian multi-index models with gradient flow. arXiv preprint arXiv:2310.19793 , 2023.
- [22] Alberto Bietti, Joan Bruna, Clayton Sanford, and Min Jae Song. Learning single-index models with shallow neural networks. Advances in Neural Information Processing Systems , 35:97689783, 2022.
- [23] Francesco Cagnetta, Leonardo Petrini, Umberto M. Tomasini, Alessandro Favero, and Matthieu Wyart. How deep neural networks learn compositional data: The random hierarchy model. Phys. Rev. X , 14:031001, Jul 2024.
- [24] Francesco Cagnetta and Matthieu Wyart. Towards a theory of how the structure of language is acquired by deep neural networks. In Advances in Neural Information Processing Systems , 2024.
- [25] Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics , 7:331-368, 2007.
- [26] Sourav Chatterjee. A generalization of the lindeberg principle. The Annals of Probability , 34(6), November 2006.
- [27] George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems , 2(4):303-314, 1989.
- [28] Alex Damian, Loucas Pillaud-Vivien, Jason D. Lee, and Joan Bruna. Computational-statistical gaps in gaussian single-index models. In Proceedings of the 37th Annual Conference on Learning Theory (COLT) , 2024.
- [29] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory , pages 5413-5452. PMLR, 2022.
- [30] Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. How twolayer neural networks learn, one (giant) step at a time. Journal of Machine Learning Research , 25(349):1-65, 2024.
- [31] Yatin Dandi, Luca Pesce, Hugo Cui, Florent Krzakala, Yue M Lu, and Bruno Loureiro. A random matrix theory perspective on the spectrum of learned features and asymptotic generalization capabilities. arXiv preprint arXiv:2410.18938 , 2024.
- [32] Yatin Dandi, Emanuele Troiani, Luca Arnaboldi, Luca Pesce, Lenka Zdeborová, and Florent Krzakala. The benefits of reusing batches for gradient descent in two-layer networks: Breaking the curse of information and leap exponents. arXiv preprint arXiv:2402.03220 , 2024.
- [33] Amit Daniely. Depth separation for neural networks. In Conference on Learning Theory , pages 690-696. PMLR, 2017.
- [34] Leonardo Defilippis, Bruno Loureiro, and Theodor Misiakiewicz. Dimension-free deterministic equivalents and scaling laws for random feature regression. In Advances in Neural Information Processing Systems , 2024.
- [35] Diego Doimo, Aldo Glielmo, Alessio Ansuini, and Alessandro Laio. Hierarchical nucleation in deep neural networks. Advances in Neural Information Processing Systems , 33:7526-7536, 2020.

- [36] Ronen Eldan and Ohad Shamir. The power of depth for feedforward neural networks. In Conference on learning theory , pages 907-940. PMLR, 2016.
- [37] Hengyu Fu, Zihao Wang, Eshaan Nichani, and Jason D. Lee. Learning hierarchical polynomials of multiple nonlinear features with three-layer networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2025.
- [38] Federica Gerace, Bruno Loureiro, Florent Krzakala, Marc Mézard, and Lenka Zdeborová. Generalisation error in learning with random features and the hidden manifold model. In International Conference on Machine Learning , pages 3452-3462. PMLR, 2020.
- [39] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. When do neural networks outperform kernel methods? Advances in Neural Information Processing Systems , 33:14820-14830, 2020.
- [40] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. The Annals of Statistics , 49(2), 2021.
- [41] Sebastian Goldt, Bruno Loureiro, Galen Reeves, Florent Krzakala, Marc Mézard, and Lenka Zdeborová. The gaussian equivalence of generative models for learning with shallow neural networks. In Mathematical and Scientific Machine Learning , pages 426-471. PMLR, 2022.
- [42] Sebastian Goldt, Bruno Loureiro, Galen Reeves, Florent Krzakala, Marc Mezard, and Lenka Zdeborova. The gaussian equivalence of generative models for learning with shallow neural networks. In Proceedings of the 2nd Mathematical and Scientific Machine Learning Conference , pages 426-471, 2022.
- [43] Sebastian Goldt, Marc Mézard, Florent Krzakala, and Lenka Zdeborová. Modeling the influence of data structure on learning in neural networks: The hidden manifold model. Physical Review X , 10(4):041044, 2020.
- [44] Harold Grad. Note on N-dimensional hermite polynomials. Communications on Pure and Applied Mathematics , 2(4):325-330, 1949.
- [45] Hong Hu and Yue M Lu. Universality laws for high-dimensional learning with random features. IEEE Transactions on Information Theory , 69(3):1932-1964, 2022.
- [46] Ziwei Ji and Matus Telgarsky. Gradient descent aligns the layers of deep linear networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2019.
- [47] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [48] Achim Klenke. Probability theory: a comprehensive course . Springer Science &amp; Business Media, 2013.
- [49] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature , 521(7553):436-444, 2015.
- [50] Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: Isoperimetry and Processes . Springer-Verlag, 1991. Google-Books-ID: juC1QgAACAAJ.
- [51] Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha SohlDickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems , 32, 2019.
- [52] Jason D. Lee, Kazusato Oko, Taiji Suzuki, and Denny Wu. Neural network learns lowdimensional polynomials with sgd near the information-theoretic limit. In Advances in Neural Information Processing Systems , 2024.
- [53] Shuo-Hui Li and Lei Wang. Neural network renormalization group. Physical review letters , 121(26):260601, 2018.
- [54] Yue M Lu and Horng-Tzer Yau. An equivalence principle for the spectrum of random innerproduct kernel matrices with polynomial scalings. arXiv preprint arXiv:2205.06308 , 2022.

- [55] Stephane Mallat. A wavelet tour of signal processing, 1999.
- [56] Tanguy Marchand, Misaki Ozawa, Giulio Biroli, and Stéphane Mallat. Multiscale data-driven energy estimation and generation. Physical Review X , 13(4), 2023.
- [57] Pankaj Mehta and David J Schwab. An exact mapping between the variational renormalization group and deep learning. arXiv preprint arXiv:1410.3831 , 2014.
- [58] Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration. Applied and Computational Harmonic Analysis , 59:3-84, 2022.
- [59] Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random feature and kernel methods: Hypercontractivity and kernel matrix concentration. Applied and Computational Harmonic Analysis , 59:3-84, 2022. Special Issue on Harmonic Analysis and Machine Learning.
- [60] Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics , 75(4):667-766, 2022.
- [61] Hrushikesh Mhaskar, Qianli Liao, and Tomaso Poggio. When and why are deep networks better than shallow ones? In Proceedings of the AAAI conference on artificial intelligence , volume 31, 2017.
- [62] Hrushikesh N Mhaskar and Tomaso Poggio. Deep vs. shallow networks: An approximation theory perspective. Analysis and Applications , 14(06):829-848, 2016.
- [63] Theodor Misiakiewicz. Spectrum of inner-product kernel matrices in the polynomial regime and multiple descent phenomenon in kernel ridge regression. arXiv preprint arXiv:2204.10425 , 2022.
- [64] Andrea Montanari and Basil N Saeed. Universality of empirical risk minimization. In Conference on Learning Theory , pages 4310-4312. PMLR, 2022.
- [65] Elchanan Mossel. Deep learning and hierarchal generative models. arXiv preprint arXiv:1612.09057 , 2016.
- [66] Eshaan Nichani, Alex Damian, and Jason D Lee. Provable guarantees for nonlinear feature learning in three-layer neural networks. Advances in Neural Information Processing Systems , 36, 2024.
- [67] Tomaso Poggio. How deep sparse networks avoid the curse of dimensionality: Efficiently computable functions are compositionally sparse. Technical Report CBMM Memo No. 138, Center for Brains, Minds and Machines (CBMM), 2023.
- [68] Tomaso Poggio, Hrushikesh Mhaskar, Lorenzo Rosasco, Brando Miranda, and Qianli Liao. Whyand when can deep-but not shallow-networks avoid the curse of dimensionality: a review. International Journal of Automation and Computing , 14(5):503-519, 2017.
- [69] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. Advances in neural information processing systems , 20, 2007.
- [70] Itay Safran and Jason Lee. Optimization-based separations for neural networks. In Conference on Learning Theory , pages 3-64. PMLR, 2022.
- [71] Itay Safran and Ohad Shamir. Depth-width tradeoffs in approximating natural functions with neural networks. In International conference on machine learning , pages 2979-2987. PMLR, 2017.
- [72] Andrew M. Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan D. Tracey, and David D. Cox. On the information bottleneck theory of deep learning. Journal of Statistical Mechanics: Theory and Experiment , 2019(12):124020, 2019.

- [73] Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In Proceedings of the International Conference on Learning Representations (ICLR) , 2014.
- [74] Antonio Sclocchi, Alessandro Favero, Noam Itzhak Levi, and Matthieu Wyart. Probing the latent hierarchical structure of data via diffusion models. In The Thirteenth International Conference on Learning Representations , 2025.
- [75] Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. Proceedings of the National Academy of Sciences , 122(1):e2408799121, 2025.
- [76] Terrence J Sejnowski. The unreasonable effectiveness of deep learning in artificial intelligence. Proceedings of the National Academy of Sciences , 117(48), 2020.
- [77] Berfin Simsek, Amire Bendjeddou, and Daniel Hsu. Learning gaussian multi-index models with gradient flow: Time complexity and directional convergence. arXiv preprint arXiv:2411.08798 , 2024.
- [78] Yitong Sun, Anna Gilbert, and Ambuj Tewari. On the approximation properties of random relu features. arXiv preprint arXiv:1810.04374 , 2018.
- [79] Matus Telgarsky. benefits of depth in neural networks. In Vitaly Feldman, Alexander Rakhlin, and Ohad Shamir, editors, 29th Annual Conference on Learning Theory , volume 49 of Proceedings of Machine Learning Research , pages 1517-1539, Columbia University, New York, New York, USA, 23-26 Jun 2016. PMLR.
- [80] Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw) . Ieee, 2015.
- [81] Emanuele Troiani, Yatin Dandi, Leonardo Defilippis, Lenka Zdeborová, Bruno Loureiro, and Florent Krzakala. Fundamental limits of weak learnability in high-dimensional multi-index models. arXiv preprint arXiv:2405.15480 , 2024.
- [82] Ramon Van Handel. Probability in high dimension. Lecture Notes (Princeton University) , 2(3):2-3, 2014.
- [83] Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027 , 2010.
- [84] Zihao Wang, Eshaan Nichani, and Jason D Lee. Learning hierarchical polynomials with three-layer neural networks. arXiv preprint arXiv:2311.13774 , 2023.
- [85] Kenneth G Wilson. Renormalization group and critical phenomena. ii. phase-space cell analysis of critical behavior. Physical Review B , 4(9):3184, 1971.
- [86] Lechao Xiao, Hong Hu, Theodor Misiakiewicz, Yue Lu, and Jeffrey Pennington. Precise learning curves and higher-order scalings for dot-product kernel regression. Advances in Neural Information Processing Systems , 35:4558-4570, 2022.
- [87] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM , 64(3):107-115, 2021.

## NeurIPS Paper Checklist

## (i) Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims in the abstract and introduction are supported by mathematical proofs or numerical results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## (ii) Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of the setting are pointed out in Section 3.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency

play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## (iii) Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The statement and the assumptions are stated in the main paper, while the proofs are presented in detail in Appendices C and D.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## (iv) Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: All the algorithms used for the simulation are described in the paper (and relevant cited literature). All plots are accompanied with description and parameters used.

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

## (v) Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code is provided in the GitHub repository associated with the manuscript. Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## (vi) Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Justification: We detail the choice of the parameters in Main theorems, caption of Figures and Section 5. We also provide Appendix F with further details regarding the numerics.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## (vii) Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experiments are carried over by considering multiple random seeds and they include error bars as described in Appendix F.4.

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

## (viii) Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide the relevant informations regarding the compute resources in Appendix F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## (ix) Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/ EthicsGuidelines ?

Answer: [Yes]

Justification: The manuscript is conform to the Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## (x) Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: Due to the theoretical nature of this work, we believe that discussion of positive and negative societal impacts is not required.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## (xi) Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

## Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## (xii) Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We work with synthetic data and we are the creator of the code and the model.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/ datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## (xiii) New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## (xiv) Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## (xv) Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## (xvi) Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/ 2025/LLM ) for what should or should not be described.

## A Illustration of SIGHT and MIGHT targets

We exemplify visually SIGHT (1) and MIGHT (4) functions in Fig. 3, and their deep version (6) in Fig. 4. These illustrations clarify how hierarchical compositions operate across layers and how depth progressively compresses the input through structured non-linear transformations. Specifically, they highlight the architectural transition from shallow models to deeper ones, where each layer reduces the effective dimensionality via localized polynomial projections. The tree-like structure of the deep targets, emphasizes the compositional nature of the learning task and motivates the layer-wise training regime analyzed in the main results.

Figure 3: SIGHT and MIGHT targets: Illustration of Single and Multi Index Gaussian Hierarchical Targets, i.e., SIGHT in eq. (2) and MIGHT in eq. (4). Left: A SIGHT function. Here we first go from x ∈ R d to z ∈ R d ε . After applying the polynomial transformation pointwise (not shown), this is projected to create a scalar h ⋆ ∈ R . One can then output the label y = g ⋆ ( h ⋆ ) . Right: A MIGHT function. Again, we go from x ∈ R d to z ∈ R d ε . After applying the polynomial transformation pointwise, we finally projecte on two values h ⋆ 4 , 1 and h ⋆ 4 , 2 , from which we create y as a two-index function y = g ⋆ ( h ⋆ 4 , 1 , h ⋆ 4 , 2 ) .

<!-- image -->

## B Depth Separation

Here, we further discuss the separation in sample complexity for deep versus shallow models complementing the exposition in the main text. Different works in approximation theory have established clear depth-separation results in expressive power. For example, [79] constructed a family of highly oscillatory functions (essentially obtained by iterative compositions of ReLU units) which a network of depth L and constant width can express in contrast to two-layer networks. Similarly, [71] demonstrated a depth separation using simple geometric indicator functions that can be efficiently realized by a network with an extra hidden layer, but cannot be approximated to high accuracy by any two-layer network of polynomial size. It is important to note that 'learning' in the context of these results refers to the ability of the architecture to approximate a fixed target function under a given input distribution. In contrast, the present work focuses on representation learning via gradient descent to recover hierarchical feature compositions in Gaussian data. The depth separation we highlight is not purely about static approximation power, but about data-efficient learning through hierarchical dimension reduction. A deep network can progressively extract and refine features across multiple layers, effectively performing stage-wise dimensionality reduction, such that each layer learns a meaningful intermediate representation of the data. Along these lines, [23] addressed under different data models similar questions and found that deep networks trained with gradient descent learn hierarchical features and progressively reduce dimensionality across layers. As highlighted in the main, the advances closest to our framework are due to [84, 66, 37], who established depth separation results between 2-layer and 3-layer networks under simpler

<!-- image -->

x

Figure 4: DeepSIGHTandMIGHT: Illustration of deep target functions. Left: A SIGHT function with depth L = 3 . Here we first go from x ∈ R d to h 1 ∈ R d ε 1 . After applying the polynomial transformation pointwise (not shown), we now divide h 1 into d ε 2 blocks of sizes d ε 1 -ε 2 . Each of these blocks is projected to create one of the components of h 1 ∈ R d ε 2 . After another polynomial transformation (not shown) we finally project to a single value h ⋆ 3 . We can then output the label y = g ⋆ ( h ⋆ 3 ) . Right: A MIGHT function with depth L = 4 . Again, we go from x ∈ R d to h 1 ∈ R d ε 1 . After applyging the polynomial transformation pointwise (not shown), we now divide h 1 into d ε 2 blocks of sizes d ε 1 -ε 2 . Each of these blocks is projected to create one of the components of h 2 ∈ R d ε 2 . We repeat this operation: we further divide h 2 into d ε 3 blocks of sizes d ε 2 -ε 3 and each of these blocs is projected to create one of the components of h 3 ∈ R d ε 3 . After another polynomial transformation (not shown) we finally project on two values h ⋆ 4 , 1 and h ⋆ 4 , 2 and create y as a two-index function y = g ⋆ ( h ⋆ 4 , 1 , h ⋆ 4 , 2 ) .

training setups, where the first layer remains fixed. In contrast, our analysis strengthens these separations by considering fully-trained architectures without fixed layers.

## B.1 Towards General Two-Layer Networks Lower Bound

In Section 2, we argued why a two-layer network, upon recovering W ⋆ (up to noisy random rotations), is insufficient for learning SIGHT targets (see eq. (13)). Ideally, one would hope to show that such barrier introduced holds for two-layer networks trained through a general gradient-based algorithm. While obtaining unconditional lower-bounds on two-layer networks trained under gradient descent remains a challenging open problem, we briefly comment on why we conjecture our class of SIGHT targets to be hard to learn with polynomial sample complexity. Amongst lowerbounds closest to our class of targets, [33] established that targets of the form g ⋆ ( x ⊤ A x ) , with

<!-- formula-not-decoded -->

A = U 0 I I 0 U ⊤ for some orthogonal matrix U ∈ R d × d cannot be learned under polynomial time and sample-complexity by a two-layer network.

We next discuss how such targets fall under the setup of MIGHT (4). Consider the setting of MIGHT with r = 2 , k = 2 , P k ( z ) = z 2 -1 and:

<!-- formula-not-decoded -->

Since h ⋆ j = 1 √ d ∑ d ϵ 1 i =1 ( ⟨ w ⋆ j,i , x ⟩ ) 2 -1 , we obtain that:

<!-- formula-not-decoded -->

where U ∈ R d × d is an orthogonal matrix whose columns are suitable orthonormal combinations of the vectors { w ⋆ 1 ,i } i ∪ { w ⋆ 2 ,i } i . We provide below a short derivation for the mapping.

Let m = d ε 1 and define the block-rotation R := 1 √ 2 ( I I I -I ) , which satisfies R ⊤ ( I 0 0 -I ) R = ( 0 I I 0 ) . Let U 0 ∈ R d × 2 m collect the 2 m orthonormal vectors { w ⋆ 1 ,i } ∪ { w ⋆ 2 ,i } , and complete it with an orthogonal complement U ⊥ to form ˜ U = [ U 0 | U ⊥ ] ∈ R d × d . Let U = ˜ U diag( R,I d -2 m ) , which is orthogonal. Then:

<!-- formula-not-decoded -->

This is exactly of the form g ⋆ ( x ⊤ A x ) with A = ˜ U ( 0 I I 0 ) ˜ U ⊤ .

## C Proofs of the main Results

## C.1 Full Statement of Theorem 1

Theorem 3. Let f ⋆ ( x ) be as in Eq. (1) with ε 1 ∈ (0 , 1) and consider a three-layer model:

<!-- formula-not-decoded -->

with W 1 ∈ R p 1 × d , W 2 ∈ R p 2 × p 1 , w 3 ∈ R p 3 .

Let L c ( θ ) denote the correlation loss defined as L cl ( θ ) := -ˆ f θ ( x ) f ⋆ ( x ) . Under Ass. 1-3, for any 0 &lt; δ &lt; δ ′ &lt; 1 , with batch-size n 1 = Θ( d ε 1 +1+ δ ) , n 2 = Θ( d kε 1 + δ ) and p 2 = p 1 = Θ( d kε 1 + δ ′ ) , the following holds with high probability as d →∞ :

Recovery by layer 1 : For each i ∈ [ p 1 ] , let w 1 i denote the i th neuron of W 1 . Suppose that w 1 i is updated as in Algorithm C.5 through spherical SGD on correlation loss L c ( θ ) , using step size η = ˜ η √ d ε 1 p 2 . Let the value at the t th iterate be denoted by w 1 ,t , i.e.:

<!-- formula-not-decoded -->

Let P W ⋆ ∈ R d × d denote the projection operator onto the subspace spanned by W ⋆ and define:

<!-- formula-not-decoded -->

Let τ i denote the stopping times defined by:

<!-- formula-not-decoded -->

then, for any κ &lt; 1 , ∃ a constant C κ &gt; 0 and ˜ η &gt; 0 such that w.h.p as d →∞ :

•

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recovery by layer 2 : Suppose that W 2 is re-initialized to W 2 = O d × d while w 3 is re-initialized with entries drawn from N (0 , 1) . Let Z = σ ( X ( W 1 ) ⊤ ) ∈ R n 2 × p 2 and consider a single pre-conditioned update of the form:

<!-- formula-not-decoded -->

There exists λ 2 ∈ R + with λ 2 = Θ( N n ) and step size η 2 = Θ( √ p 2 ) such that the pre-activation h 2 ( x ) = W 2 σ ( W 1 x ) ∈ R p 2 satisfies:

<!-- formula-not-decoded -->

Remark : Here, we introduced the regularization parameter λ 2 &gt; 0 since we assume that p 1 ≫ n 2 (overparameterized setting) and thus Z ⊤ Z is singular. Alternatively, one could consider the underparameterized setting p 1 ≪ n 2 without the need for an additional regularization.

Recovery by layer-3 : Let X ∈ R n 3 × d denote a matrix with rows containing n 3 independent samples from the input distribution N (0 , I d ) . Let H ∈ R n 3 × p 3 denote the corresponding pre-activation matrix with rows { W 2 σ ( W 1 x )) -f ⋆ ( x i ) , for i ∈ [ n 3 ] }. For n 3 = Θ( d δ ) , ∃ λ &gt; 0 such that the ridgeregression predictor ˆ w λ given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

satisfies:

## C.2 Proof Sketch

We prove each of the three parts of Theorem 1 in succession. We outline the proof for each of these parts below:

Part ( i ) :

- (i) The asymptotic composition of Hermite polynomials allows us to decompose the Hermite decomposition of f ⋆ ( x ) along Hermite polynomials applied to W ⋆ x .
- (ii) The leading order term in the Hermite-decomposition f ⋆ ( x ) lies along He 2 ( W ⋆ x ) , which contributes a linear drift to the dynamics of each neuron in W 1 , with the direction of the drift for neuron i given by u ⋆ i = W ⋆ ( W ⋆ ) ⊤ w 1 , 0 i , i.e, the initial direction of w 1 i projection onto W ⋆ .
- (iii) We show that the neuron w 1 i remains approximately isotropic w.r.t the rows of W ⋆ .
- (iv) Under the above isotropy and due to d ε 1 ≫ 1 , we show that the above linear term dominates throughput the weak-recovery and subsequent states of the dynamics.
- (v) As a consequence, each neuron in W 1 evolves primarily along u ⋆ i , with the noise controlled through the choice of batch-size. A stopping-time based analsyis then yields w ( t ) i → u ⋆ i .
- (vi) For subsquent use in part ( ii ) however, we require finer control over the distribution of w 1 i and its residual terms.
- (vii) Inductively, we show that the distribution of w 1 i conditioned on a suitable stopping-time is approximately uniform on the unit sphere along W ⋆ and maintains hypercontractivity.

Part ( ii ) :

- (i) Through results established in Part ( i ) , we show that the distribution of the updated weights W 1 approximately maintains hypercontractivity for the eigenfunctions of the randomfeatures Kernel associated to the features σ ( XW ⊤ 1 ) . This ensures the concentration of the associated sample covariances.

- (ii) Upon establishing concentration and spherical approximation along the subspace corresponding to W ⋆ , through an analysis similar to [58], we show that the feature matrix Z = σ ( XW ⊤ ) contains Θ( d k ) spikes with diverging eigenvalues and an isotropic bulk with eigenvalues O (1) .
- (iii) Under n, p 2 ≫ Θ( d kε 1 ) , we show that these spikes suffice for the pre-conditioned update

<!-- formula-not-decoded -->

to approximate f ⋆ ( x ) upto degree k -components. As a result, we obtain the recovery of h ⋆ ( x ) through h 2 ( x )

Part ( iii ) : Finally, fitting the target f ⋆ ( x ) upon training w 3 follows through universality of the random features Kernel associated with σ ( · ) and perturbation of the Kernel regression operators.

## C.3 Preliminaries

## C.3.1 Stochastic Domination

Throughout the analysis, much of our probabilistic error bounds will take the following form, which are standard for functions of random variables with finite Orlicz-norm such as sub-Gaussian/subExponential random variables:

<!-- formula-not-decoded -->

for some constants m&gt; 1 , k &gt; 0 , c &gt; 0 . A slightly weaker form of the bound takes the form:

<!-- formula-not-decoded -->

for any δ &gt; 0 and k ∈ N . To concisely represent such bounds, we use the following notation: Definition 2. [Stochastic dominance [54]] We say that a sequence of real or complex random variables X d in a normed space is stochastically dominated by another sequence Y d in the same space if for all ε &gt; 0 and k , the following holds for large enough d :

<!-- formula-not-decoded -->

We denote the above relation through the following notation:

<!-- formula-not-decoded -->

Through a union bound, we obtain that O ≺ is closed under addition, multiplication, i.e X 1 = O ≺ ( Y 1 ) and X 2 = O ≺ ( Y 2 ) imply that:

and:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, due to the flexibility of setting an arbitrarily large k in Eq. (36) , we observe that stochastic dominance is closed under unions of polynomially many events in d .

We will often exploit this while taking unions over p = O ( d ) neurons and n = O ( d ) samples. Furthermore, ≺ absorbs polylogarithmic factors i.e:

<!-- formula-not-decoded -->

subsumes exponential tail bounds of the form:

<!-- formula-not-decoded -->

for some α &gt; 0 , as well as polynomial tails of arbitrarily large degree:

<!-- formula-not-decoded -->

for some sequence of constants C k dependent on k .

The above bounds directly translate to the following control over moments:

Proposition 1. Let X d , Y d denote two sequences of random variables with:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 2. The above proposition follows directly through the following decomposition:

<!-- formula-not-decoded -->

where 1 denotes the indicator function. Using the property E [ Z ] = ∫ ∞ s =0 Pr[ Z &gt; s ] ds , the second term is bounded by 1 d k for any k and large enough d .

Asymptotic notation: In light of the above proposition, throughout the subsequent sections, we use the notation ˜ O to denote deterministic asymptotic bounds upto factors d δ for arbitrarily small δ &gt; 0 i.e:

<!-- formula-not-decoded -->

if for any δ &gt; 0 , f ( d ) ≤ d δ g ( d ) for large enough d

Through a standard application of the Lindeberg exchange technique [26, 82], we further have the following useful estimate:

Lemma1 (Non-asymptotic CLT -bound) . Let X 1 , . . . , X n ∈ R be n i.i.d random variables satisfying X i = O ≺ (1) . Then, for any function q : R → R with q ∈ C 3 ( R ) , ∥ q ′′′ ∥ ∞ &lt; ∞ and any δ &gt; 0 :

<!-- formula-not-decoded -->

where c 1 , c 2 denote constants dependent only on q .

Through standard truncation arguments over the tail of 1 √ d ( ∑ d i =1 X i ) , the above bound extends to all polynomials R → R of finite degree.

## C.3.2 Orthogonal Polynomials and Spherical Harmonics

Hermite Polynomials : A key role in our analysis is played by the decomposition of square integrable function with respect to the Gaussian measure in terms of the Hermite polynomials [44]. Definition 3 (Hermite decomposition) . Let f : R m → R be a function that is square integrable w.r.t the Gaussian measure. There exists a family of tensors ( C j ( f )) j ∈ N such that C j ( f ) is of order j and for all x ∈ R m ,

<!-- formula-not-decoded -->

where H j ( x ) is the j -th order Hermite tensor [44].

Gegenbauer and Associated Laguerre polynomials Let w ∼ U ( S d -1 ( √ d ) denote a random variable distributed uniformly on the sphere in R d of radiues √ d . Let µ d denote the associated push-forward measure of the projection √ d ⟨ w , e 1 ⟩ . The Gegenbauer polynomials Q d ℓ ( · ) [39] for ℓ ∈ N form an orthonormal basis w.r.t L 2 ( µ d ) with Q d ℓ ( · ) being a polynomial of degree ℓ . Therefore, for any f ∈ L 2 ( µ d ) and v ∈ R d with ∥ v ∥ = 1 , the following decomposition exists:

<!-- formula-not-decoded -->

Next, suppose that x ∼ N ( 0 , I d ) . Let τ d denote the associated pushforward measure of ∥ x ∥ 2 . Then, the associated Laguerre polynomials l d k ( · ) form an orthonormal basis w.r.t τ d [9].

Spherical Harmonics Recall that any inner-product Kernel can be diagonalized w.r.t L 2 ( U ( S d -1 ( √ d ))) along the basis of spherical Harmonics { Y ℓ,k } ℓ ∈ [ B ( d,k )] ,k ∈ N } , where B ( d, k ) denotes the number of spherical harmonics of degree k , satisfying B ( d, k ) = Θ( d k ) :

<!-- formula-not-decoded -->

then for any q ∈ N and δ &gt; 0 :

where λ k denotes the eigenvalue of K w.r.t the k -degree spherical harmonics Y l,k ( x ) . [40].

The Spherical Harmonics are related to the Gegenbauer polynomials through the following identity: Proposition 3. For any w 1 , w 2 ∼ U ( S d -1 ( √ d )) :

<!-- formula-not-decoded -->

We next recall that the Gaussian measure N ( 0 , I d ) admits the following tensor product decomposition:

<!-- formula-not-decoded -->

where U ( S d -1 ( √ d )) denotes the uniform measure on sphere of radius √ d

The above tensor product decomposition naturally relates the Hermite orthonormal basis w.r.t the Gaussian measure against the product of radial functions and Gegenbauer polynomials. In particular, we have the following relation:

Proposition 4. For any k ∈ N , the k th -degree Hermite polynomial lies in the subspace spanned by functions of the form:

<!-- formula-not-decoded -->

with 0 &lt; j ≤ k .

Proof. Recall that Y ℓ,j ( √ d x / ∥ x ∥ ) are homogenous polynomials of degree j . Upon restriction to the sphere of radius ∥ x ∥ , He k ( x ) is a polynomial of degree at-most k . Therefore, by Fubini's theorem, we obtain:

<!-- formula-not-decoded -->

for j &gt; k .

Proposition 5. For any k &gt; 2 and polynomial q ( x ) :

<!-- formula-not-decoded -->

Proof. The above is a direct consequence of Lemma 1 applied to the random variables ( He 2 ( ⟨ w ⋆ i , x ⟩ ) , He k ( ⟨ w ⋆ i , x ⟩ )) ∈ R 2 , whose higher-moments are bounded by Gaussian hypercontractivity (Lemma 5).

We utilize Gegenbauer polynomials and spherical Harmonics primarily due to the absence of results on eigenvectors of inner-product Kernel matrices under polynomial scalings. This is also the primary bottleneck towards the extension of our theory to multiple layers. Essentially, our analysis relies on showing the concentration of the sample-covariance matrix to the population covariance matrix along the degreek components.

## C.3.3 Spectral Norm of a tensor

Definition 4. For a symmetric positive-definite tensor T ∈ R d ⊗ k of order k , we define the spectral norm of T as follows:

<!-- formula-not-decoded -->

## C.3.4 Hermite-tensors and Gaussian-inner Products

We denote by He k for k ∈ N the normalized Hermite-polynomials forming an orthonormal basis w.r.t L 2 ( γ ) . For any f ∈ L 2 ( γ ) , we have:

<!-- formula-not-decoded -->

The Hermite tensors result in the following generalization of the above decomposition: Proposition 6. Let γ m denote the m -dimensional Gaussian measure. For any f, g : R m → R ∈ ℓ 2 ( R m , γ m ) , let C k ( f ) denote the k th -order Hermite-tensor, defined as:

<!-- formula-not-decoded -->

where He k ( z ) denotes the k th -order Hermite tensor on R m . Then:

<!-- formula-not-decoded -->

## C.3.5 Compact Self-Adjoint Operators

Proposition 7. Let A : L 2 ( µ, Ω) → L 2 ( µ, Ω) denote a bounded-linear operator on a hilbert space L 2 ( µ ) . Then:

We collect here the following well-known properties of bounded linear operators on a Hilbert space L 2 ( µ ) [16]:

- (i) If A is compact, self-adjoint then A can be diagonalized along a countable-basis of eigenvectors.
- (ii) Suppose that µ is σ -finite, then any integral operator I ( x, y ) : ω × ω → R with ∥ I ( x, y ) ∥ L 2 ( µ ) × L 2 ( µ ) &lt; ∞ is compact
- (iii) For a symmetric integral operator ∥ I ( x, y ) ∥ L 2 ( µ ) × L 2 ( µ ) &lt; ∞ :

<!-- formula-not-decoded -->

where { λ k } denote the eigenvalues associated with I ( · , · ) .

## C.3.6 Concentration in Orlicz-spaces

Definition 5. For any α ∈ R , define ψ α ( x ) = e x α -1 . The Orlicz norm for a real random variable X ; ∥ X ∥ ψ α is defined as

<!-- formula-not-decoded -->

Random variables exhibiting suitable bounds on orlicz norms of finite-order exhibit the following concentration inequality:

Theorem 4 (Theorem 6.2.3 in [50]) . Let X 1 , . . . , X n be n independent random variables with zero mean and second moment E X 2 i = σ 2 i . Then,

<!-- formula-not-decoded -->

## C.4 Useful Preliminary Results

A central result underlying our analysis for part ( ii ) of Theorem 1, based on [60] is the following matrix-concentration bound for matrices with independent heavy-tailed rows: (Theorem 5.48 in [83]) R n × p R p

Lemma2 . Let A ∈ be a random matrix with independent rows a i ∈ with covariance E [ a i a ⊤ i ] = Σ a and E [ max i ≤ n ∥ a ∥ 2 i ] ≤ m . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 3 (Weyl's inequality) . For any A,B ∈ R m × n , for all i ∈ N with i ≤ min( m,n ) :

<!-- formula-not-decoded -->

Lemma 4 (Resolvent Identity) . Let, A,B ∈ R p × p be two invertible matrices, then:

<!-- formula-not-decoded -->

Ournext central tool that will be utilized frequently throughout our analysis is the hypercontractivity w.r.t the Gaussian measure:

Lemma 5 (Gaussian Hypercontractivity, Proposition 5.48. in [15]) . For any polynomial q : R d → R of degree k and any p ∈ N , p ≥ 2 :

<!-- formula-not-decoded -->

where γ d denotes the standard Gaussian measure on R d and ∥ q ( z ) ∥ p,γ denotes the p -norm:

<!-- formula-not-decoded -->

Proposition 8. Let z ∼ N (0 , I d ) denote a d -dimensional Gaussian vectors. Suppose that X 1 , · · · , X k denote i.i.d random variables obtained by applying a fixed polynomial of degree p ∈ N to distinct subsets of coordinates of z . Then:

<!-- formula-not-decoded -->

Proof. Since q ( z ) = 1 √ k ( ∑ k i =1 X i ) is a polynomial in z with finite degree p , Lemma 5 implies that its higher-order moments are bounded as ∥ q ( x ) ∥ p ≤ C p ∥ q ( x ) ∥ 2 . The result then follows by noting that:

<!-- formula-not-decoded -->

Lemma 6 (Discrete Gronwall) . Let a t , b t , c t 1 , c t 2 be non-negative sequences satisfying:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We analogously have the corresponding upper bound i.e

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then, for any t ∈ N :

implies that:

## C.5 Full Algorithm

We describe the full algorithmic routine used in Theorem 1 in Algorithm 1.

## C.6 Leveraging Asymptotic Gaussianity

A crucial property of the non-linear feature h ⋆ ( x ) that we leverage is its asymptotic Gaussianity, not only w.r.t their marginals but w.r.t propagation to the lower-level features. Specifically, building on [84], we show that the high Hermite-degree functions of h ⋆ m ( x ) do not propagate projections along low Hermite-degree functions of W ⋆ x . To show this, we provide an inductive proof inspired by the combinatorial approach developed in [84], wherein the (entropically) dominant contributions in ( h ⋆ m ( x )) k arise from terms having the lowest degrees in He j ( ⟨ w ⋆ , x ⟩ ) .

## Algorithm 1 Layer-Wise Training for a Three-Layer Network

Input: Training data D , mini-batch sizes n 1 , n 2 , learning rates η 1 , η 2 , ridge regularization λ , iteration steps T 1 .

## Initialize:

<!-- formula-not-decoded -->

Layer 1 updates (correlation loss with spherical projections):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Sample mini-batch X, y ⊂ D of size n 1

For each neuron j in layer 1:

<!-- formula-not-decoded -->

end for

Fix layer 1, update layer 2:

(2)

<!-- formula-not-decoded -->

Fix layers 1,2, solve for W (3) via ridge regression:

Sample mini-batch X, y ⊂ D of size n 3

Form design matrix H :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 9. For any ε 1 &gt; 0 , and k ∈ N , let h ⋆ ( x ) for x ∈ R d denote a non-linear feature of the form:

<!-- formula-not-decoded -->

where P k denote polynomials of degree k satisfying E z ∼ N (0 , 1) [P k ( z )] = 0 and E z ∼ N (0 , 1) [P k ( z ) z ] = 0

Denote by S the set of indices in [ d ε 1 ] and by Γ m ( S ) the set of all m -permutations in S consisting of distinct values.

Then, the following holds for any m ∈ N :

<!-- formula-not-decoded -->

where r m ( x ) satisfies:

(i)

<!-- formula-not-decoded -->

(ii) For any k ∈ N and v ∈ R d :

<!-- formula-not-decoded -->

for some ˜ δ &gt; 0 , where recall that ˜ O subsumes factors of the form d δ for arbitrarily small δ &gt; 0 .

The above set of properties characterize, in particular the projections onto Hermite-polynomials of x of non-linear functions applied to h ⋆ ( x )

̸

Proof. The proof proceeds by induction. Similar to [84], the central idea is to utilize the fact that the Hermite-degree is additive for products of terms dependent on orthogonal subspaces. The entropically-dominant terms in He p ( h ⋆ ( x )) arise from products of ⟨ w ⋆ i , x ⟩ , ⟨ w ⋆ j , x ⟩ for i = j contributing a leading Hermite-degree of dk .

We show inductively that Equation 75 holds for any m ∈ N .

The base case m = 1 holds trivially. Suppose that the statement holds for some m ∈ N . Recall that the (normalized) Hermite polynomials satisfy the following recursion:

<!-- formula-not-decoded -->

Applying the above relation with x = h ⋆ ( x ) yields:

<!-- formula-not-decoded -->

The induction hypothesis on He m ( h ⋆ ( x )) , then implies:

<!-- formula-not-decoded -->

The first term splits into two components depending on whether i ∈ s or i / ∈ s :

<!-- formula-not-decoded -->

where we used that ∑ d ε 1 i =1 1 √ d ε 1 P k ( ⟨ w ⋆ i , x ⟩ ) r m ( x ) = O ≺ ( 1 √ d ε 1 ) through the closure undermuliplication of O ≺ ( · ) and Lemma 8. The second term is exactly the desired expression for He m ( h ⋆ ( x )) in Equation 75.

Next, we rewrite the first term as:

<!-- formula-not-decoded -->

By the induction hypothesis, T 1 cancels with -m √ m -1 m +1 He m -1 ( h ⋆ ( x )) upto an error O ≺ ( 1 √ d ε ) . It remains to show that T 2 is stochastically dominated as O ≺ ( 1 √ d ε ) . To achieve this, we note by Gaussian hypercontractivity (Lemma 5), it suffices to bound the second-moment of T 2 . We have:

̸

<!-- formula-not-decoded -->

where in the last line we used the fact that the cross-terms vanish for terms with He k ( ⟨ w ⋆ s i , x ⟩ ) appearing once. The desired bound is obtained by noting that by the inductive hypothesis:

<!-- formula-not-decoded -->

Therefore the first term contributes d ε 1 terms of order O ≺ ( d ( m -1) ε 1 ) while the second term consists of d 2 ε 1 terms of order O ≺ ( d ( m ) ε 1 ) . Therefore, both the terms are entropically sub-dominant compared to the factor 1 d ( m +1) ε 1 , yielding:

<!-- formula-not-decoded -->

It remains to show statement ( ii ) (Equation 77). We first consider the residual term:

<!-- formula-not-decoded -->

Recall that for any v ∈ R d and any r ( x ) :

<!-- formula-not-decoded -->

Therefore, by induction and the closure of stochastic domination under multplication, the above term satisfies ( ii ) .

For the remaining term T 2 , ( ii ) holds by noting that by Proposition 6, for each i ∈ √ d ε 1 , ∥ E [ C k ( r m ( x )) He k -1 ( ⟨ v , x ⟩ ) ⟨ x , w ⋆ i ⟩ ] ∥ 2 is a polynomial in {⟨ w ⋆ i , v ⟩} i ∈ √ d ε 1 of degree at-least k -1 . Since {⟨ x , w ⋆ i ⟩} are orthonormal functions:

<!-- formula-not-decoded -->

## C.7 Existence of activation satisfying Assumptions 2, 3

Consider any σ : R → R and a constant c &gt; 0 . Observe that the activation ˜ σ : R → R defined as: ˜ σ ( x ) := σ ( x ) -cx, (87)

<!-- formula-not-decoded -->

̸

satisfies:

Set σ ( x ) as a bounded-analytic function with E [ σ ( z ) He k ( z )] = 0 , for instance σ ( z ) = tanh ( z + a ) -bz , for some a, b = 0 such that such that E [ σ ( z ) He k ( z )] = 0 for all k ∈ N . Furthermore, we may further set a, b ∈ R such that E z ∼ N (0 , 1) [ σ ( σ ( z ))] &lt; 0 , for instance by setting b ≈ 0 and a ≈ 0 , a &lt; 0 .

̸

̸

Then, by Equation 88, the condition E z ∼ N (0 , 1) [˜ σ ((˜ σ )( z )) z ] = 0 corresponds to the following equation on c :

<!-- formula-not-decoded -->

By the choice of σ , g (0) &lt; 0 while the boundedness of σ further implies that g ( c ) →∞ as c →∞ . Hence ∃ c ∈ R such that g ( c ) = 0 . On the other hand, note that ˜ σ

## C.8 Feature Learning by the First Layer

In this section, we analyze the dynamics of W 1 (part ( i ) of Theorem 1). In fact, for subsequent usage in the dynamics of W 2 , w 3 , we require a stronger characterization of ( i ) of Theorem 1. To state the precise result, we first set up the required notation. Let D t = { X t , y t } denote the batch of samples at time-step t for t ∈ N . Observe that under the correlation loss, and with W 2 = I , each neuron w i for i ∈ [ p ] evolves independently. In-fact, the dynamics is equivalent to that of a two-layer network with modified activation ˜ σ = σ ( σ ( · )))

Therefore, the gradient descent dynamics on W 1 defines a stochastic mapping:

<!-- formula-not-decoded -->

applied to a random variable w 0 ∼ U ( S d -1 (1)) .

Let { F t } t ∈ N denote the filtration generated by D 1 , D 2 , · · · . Let U ⋆ ∈ R d denote the subspace spanned by the teacher weights W ⋆ . Define u ⋆ := P U ⋆ w 0 ∥ P U ⋆ w 0 ∥ to be the unit-vector along P W ⋆ w 0 . Our analysis proceeds by establishing the following:

- (i) The dynamics of w ( t ) is dominated by drift along the initial direction u ⋆ .
- (ii) The overlap of w ( t ) along W ⋆ grows linearly upto reaching a threshold κ &gt; 0 and subsequently w ( t ) reaches overlap κ in a constant number of iterations.
- (iii) The distribution of w ( t ) maintains isotropy and regularity of tails.

To see intuitively why the dynamics of w ( t ) is dominated by the drift along u ⋆ , consider the following heuristic sketch:

<!-- formula-not-decoded -->

where we substituted the decomposition in Proposition 9.

Next, we note that the degree j contribution in E [ h ⋆ ( x ) σ ′ ( w ( t ) ) x ] is of the form:

<!-- formula-not-decoded -->

For j = 2 , the above term results in a drift along u ⋆ i , while for j &gt; 2 , the contributions are suppressed as long as ⟨ w ⋆ i , w ( t ) ⟩ = O ≺ ( 1 √ d ϵ 1 ) . Analogously, the contributions from higher-order terms are suppressed as long as w ( t ) doesn't align with individual directiosn w ⋆ i .

We now move on to the full proof. Let κ &gt; 0 be fixed We introduce the following hitting time:

<!-- formula-not-decoded -->

Let F ⋆ denote the product sigma-algebra w.r.t { F t } . Since τ κ is measurable w.r.t σ ( F ⋆ ∪ F ( w 0 )) , the random variable w τ κ then admits a regular conditional distribution w.r.t F ⋆ , µ κ ( | F ⋆ ) [48].

Suppose that each neuron for i ∈ [ p 1 ] in Algorithm 1 is stopped at τ κ as defined above.

Let e 1 , · · · e d -d ε 1 denote a fixed basis for the complement of W ⋆ . The main result of this section establishes points ( i ) -( iii ) described above and constitutes the formal statement for part ( i ) of Theorem 1:

Theorem 5. For any 0 &lt; κ &lt; 1 , let µ κ ( ·| X 1 , X 2 , · · · ) denote the regular conditional measure over w τ κ conditioned on the sequence of datasets X 1 , X 2 , · · · associated with the natural filtration F t . Then, for any k ∈ N , there exists a sequence of 'high-probability" events E ∈ ∪ t ≥ 1 { F t } such that:

<!-- formula-not-decoded -->

- (ii) For any X 1 , X 2 , · · · ∈ E , the random variable w ∼ µ η ( ·| X 1 , X 2 , · · · ) , satisfies the following with probability 1 -Ce -C log d 2 as d →∞ :

<!-- formula-not-decoded -->

κ where κ + ≥ κ and: (a) u ⊥ ∈ U ⋆ , v ∈ U ⋆ ⊥ . (b) ∥ u ⊥ ∥ = O ≺ ( 1 d δ ) . (c)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

- (d) For any (deterministic) w ⋆ ∈ U ⋆ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where w 0 ∼ U ( S d -1 (1)) denotes the initialization of the neuron.

Properties ( c ) stipulates that w τ κ remains approximately isotropic with well-behaved tails along a fixed basis of U ⋆ and its complement. This is important for ensuring concentration of well-behaved functions of w τ κ in part ( ii ) . Maintaining this property throughout the dynamics further leads to a control over the higher-order terms. √

Corollary 1. Let w τ η be as defined in Theorem 5. Then, ∃ δ &gt; 0 and choice of step-size η = ˜ η d ε 1 for some ˜ η &gt; 0 such that:

<!-- formula-not-decoded -->

## C.9 Form of the Update

Under the initialization W 2 = I p , b 1 , b 2 = 0 , w 3 = 1 , for any i ∈ [ p 1 ] , the update to any neuron w in W 1 can be expressed as:

<!-- formula-not-decoded -->

In what follows, we denote the gradient update as:

<!-- formula-not-decoded -->

and its corresponding spherical version as:

<!-- formula-not-decoded -->

(e)

where we recall that ∥ ∥ w ( t ) ∥ ∥ = 1 by the spherical constraint.

Applying the Hermite decomposition to f ⋆ ( x ) and utilizing the composition of Hermite-coefficients established in Proposition 9 results in the following expansion for the gradient:

Lemma 7. Let C ⋆ k for k ∈ N denote the k th -order Hermite tensor of f ⋆ ( z ) and let { c k } ∞ k =1 be the Hermite coefficients of ˜ σ ( · ) . Then:

<!-- formula-not-decoded -->

Proof. The above is a direct consequence of Stein's Lemma applied to E [ g t ] :

<!-- formula-not-decoded -->

The first term vanishes under the orthogonal projection ( I d -( w ( t ) )( w ( t ) ) ⊤ ) while the second term results in Equation 102.

Combining the above with the recursive Hermite decomposition of f ⋆ ( x ) through Proposition 9 yields the following form for the expected updates:

Proposition 10. Let { µ k } ∞ k =1 , { c k } ∞ k =1 , { c ⋆ k } ∞ k =1 denote the Hermite coefficients of g ⋆ , ˜ σ, P k ⋆ respectively. Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r m ( x ) denotes the remainder for the degreem Hermite term in Equation 75.

Proof. Proposition 9 applied to f ⋆ ( x ) yields:

<!-- formula-not-decoded -->

Next, by expanding P k ( ⟨ w ⋆ s i , x ⟩ ) = ∑ k j =1 c ⋆ k He j ( ⟨ w ⋆ s i , x ⟩ ) , the first term in the RHS can be further decomposed as:

<!-- formula-not-decoded -->

Equation 103 then follows by noting that any term of the form ∏ m i =1 c ⋆ j i He j 1 ( ⟨ w ⋆ s i , w ⟩ ) appears in C ⋆ ℓ with ℓ = ∑ m i =1 j i .

The magnitude of the gradient updates is bounded through the following Lemma:

Proposition 11. Let g t ⊥ ,i := σf ⋆ ( x ) σ ′ ( σ ′ ( ⟨ w , x ⟩ )) x ( I -1 ε 2 ww ⊤ ) denote the spherical gradient for neuron i at time-step t . Then g ⊥ satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(ii) For any v ∈ R d with ∥ v ∥ = 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We begin by writing:

<!-- formula-not-decoded -->

By the assumption on σ , the composed activation ˜ σ and its derivatives are polynomially bounded. Therefore, applying standard concentration results for for independent random variables with bounded orlicz norm (Theorem 4), we obtain:

<!-- formula-not-decoded -->

which yields Equation 106.

Applying the same result (Theorem 4) to projections of g t ⊥ along v and P U ⋆ then yields Equations 107 and 108.

## C.10 Initial Overlaps

Before proceeding with the analysis, we collect the following result on the concentration of the initial overlaps along W ⋆ for the first-layer neurons at initialization: Lemma 8. For w ∼ N (0 , 1 d I ) ,

<!-- formula-not-decoded -->

Proof. Since w 0 ∼ U ( S d -1 (1)) , the squared overlap norm d d ε 1 ∥ ∥ W ⋆ w 0 1 ,i ∥ ∥ 2 2 is an average over d ε 1 sub-exponential random variables. Therefore, a standard application of Bernstein's inequality [83] yields an error probability of 1 -ce -log d 2 . The proposition then follows through a union bound over the p 1 neurons.

## C.11 Difference Inequality

Let P ⋆ U ⋆ := ( W ⋆ ) ⊤ ( W ⋆ ) denote the projector onto the subspace spanned by W ⋆ . For i ∈ [ p 1 ] , define u ⋆ i to be the unit-vector along the projection of w 0 i along W ⋆ :

<!-- formula-not-decoded -->

Further define m t = ⟨ u ⋆ , w ( t ) 1 ,i ⟩ and m t ⊥ = ∥ ∥ ( P ⋆ U ⋆ -u ⋆ ( u ⋆ ) ⊤ ) w ( t ) ∥ ∥ , denoting the projections of w ( t ) along u ⋆ i and its complement in the span of W ⋆ and:

<!-- formula-not-decoded -->

Additionally, we track the residual component in w ( t ) lying in U ⋆ ⊥ but orthogonal to w 0 :

<!-- formula-not-decoded -->

Our analysis relies on showing that the dynamics of w ( t ) is dominated by a linear drift along u ⋆ . This requires a control over the following additional terms:

- (i) Residual linear drift along U ⋆ ⊥ : This term is controlled through a bound on m t ⊥ .
- (ii) Contributions from higher-order terms: These are controlled through a bound on m t × .
- (iii) Noise in the gradient updates: This is supressed through the choice of batch-size n = Θ( d 1+ ε 1 + δ )

Recall that n 1 = Θ( d kε 1 + δ ) . Let ˜ δ be any arbitrary value satisfying 0 &lt; ˜ δ &lt; δ For any η &gt; 0 , we define the following stopping times:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The stopping time τ + κ simply accounts for the overlap reaching the desired value κ . The stopping time τ -˜ δ ensures that for t ≤ τ -˜ δ , the three residual contributions in g t ⊥ listed above, namely the drift along U ⋆ ⊥ , higher-order terms and the gradient noise remain supressed.

Note that by definition, m 0 i, ⊥ = 0 while m 0 i = 1 √ d ε 1 by Lemma 8 . While both m 0 i, ⊥ , m 0 i grow exponentially, we will show that for any 0 &lt; ˜ δ &lt; δ , there exists a small enough ˜ η such that with step size η = ˜ ηd ε 1 , τ + κ &gt; τ ⊥ ˜ δ with high-probability. Concretely, under small enough step-size, both m 0 i , m 0 i, ⊥ grow under approximately identical linear dynamics, ensuring that the initial lead in the magnitude of m 0 i is maintained till weak-recovery over the contributions from the remaining directions. √

Proposition 12. Let c = µ 1 c ⋆ 2 ˜ c 2 and η = ˜ η d ε 1 p 2 Define τ ⋆ := τ + κ ∧ τ -˜ δ For any ˜ δ &lt; δ ⊥ &lt; δ, κ and k ∈ N , and any constants c + m , c + ⊥ , c + × , c -m , c -⊥ , c -× such that c -m &lt; c &lt; c + m , c -⊥ &lt; c &lt; c + ⊥ , c -× &lt; c &lt; c + × there exists constants C 1 , C 2 , C 3 , C 4 and ˜ η such that for large enough d :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Before establishing the above proposition, we first show how it implies Theorem 5

## C.12 Proof of Theorem 5

Suppose that t ≤ τ ⋆ . Applying a union-bound over time-steps to ( i ) in Proposition 12 then implies that with probability at-least t (1 -1 d k ) , for all t ≤ τ ⋆ , we have with high-probability:

<!-- formula-not-decoded -->

Since τ ⋆ ≤ τ + κ , we further have that m t ≤ κ for all t ≤ τ ⋆ and thus ˜ η 2 C 1 m 3 t &lt; ˜ η 2 C 1 κm 2 t . Equation 121 then implies:

<!-- formula-not-decoded -->

which inductively implies the following intermediate bound:

<!-- formula-not-decoded -->

Since m 2 t ≤ κm t , the above simplifies to:

<!-- formula-not-decoded -->

Therefore, ∀ t &lt; τ ⋆ , we have:

By Lemma 8, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since for small enough ˜ η , c + m (1 -κ ) -˜ η 2 C 1 κ 2 &gt; 0 , Equation 125 implies:

<!-- formula-not-decoded -->

for some constant c κ,ε &gt; 0 .

Next, consider the orthogonal component m t ⊥ = 0 . Note that m 0 ⊥ = 0 by definition. Part ( ii ) of Proposition 12 along with the discrete Gronwall inequality (Lemma 6) implies:

<!-- formula-not-decoded -->

where we used that m t × ≤ 1 d ˜ δ since t ≤ τ -˜ δ .

Our goal next is to compare the above bound against the lower-bound given by Equation 123. Since | log(1 + a ) -log(1 + b ) | ≤ | a -b | for a, b &gt; 0 , we have:

<!-- formula-not-decoded -->

Therefore, we obtain the corresponding bound:

<!-- formula-not-decoded -->

For any 0 &lt; δ ⊥ &lt; ˜ δ , we may set | c + m -c -⊥ | , | c -m -c + ⊥ | and ˜ η small enough so that for any t ≤ c κ,ε log d :

Implying:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly by setting | c + m -c -× | , | c -m -c + × | small enough enough we have by part (iii) in Proposition 12:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, while the dynamics of m t , m t ⊥ , m t × evolves at arbitrarily close rates. The initial advantage in m t over m t ⊥ , m t × through initalization ensures that the hitting time for m t arrives first, ensuring that:

By the definition of τ ⋆ , this establishes all claims in Theorem 1 apart from ( e ) . To obtain e , note that by the form of the updates, r t ⊥ is updated solely through the gradient noise g t ⊥ -E [ g t ⊥ ] and normalization. Therefore, Lemma 11 implies that:

<!-- formula-not-decoded -->

## C.12.1 The conditioning input set

To complete the proof of Theorem 5, it remains to specify the high-probability set E κ, ˜ δ . For any ˜ δ &gt; 0 and κ ∈ R , consider the event:

<!-- formula-not-decoded -->

Equations 107, 110 and a union bound imply that for any k ∈ N :

<!-- formula-not-decoded -->

where the probability is w.r.t the joint measure over w 0 , X 1 , · · · X n By the law of total expectation:

<!-- formula-not-decoded -->

implying, for any k &gt; ˜ k ∈ N :

<!-- formula-not-decoded -->

taking the interesection over k ∈ N the required conditioning set E in Theorem 5.

## C.13 Proof of Proposition 12

Consider the 'effective" activation:

<!-- formula-not-decoded -->

Let ˜ c 2 = E z ∼ N (0 , 1) [˜ σ ( z ) He 2 ( z )] . Define the constant c = µ 2 c ⋆ 1 ˜ c 2 . By assumption 3, c &gt; 0 .

part ( i ) : Applying Proposition 10, we obtain the following decomposition for the update g ⊥ t :

<!-- formula-not-decoded -->

Since m t = ⟨ w ( t ) , u ⋆ ⟩ , ∥ w t ∥ = 1 the first term simplifies to:

<!-- formula-not-decoded -->

Next, for ∆ 1 , we separately consider the component along w ⋆ i for each i ∈ √ d ε 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Each term of the form ⟨ w ⋆ s i , w ⟩ is uniformly bounded as m t × √ d ε 1 . Since | Γ( S, m ) , s m = i | = Θ( d ( m -1) ε 1 ) , we obtain:

<!-- formula-not-decoded -->

for some constants c j with sup j | c j | &lt; ∞ . Therefore a geometric-series bound (applicable since m t × &lt; 1 ) yields:

<!-- formula-not-decoded -->

for some contant C &gt; 0 . Summing the above bound over i ∈ √ d ε 1 , results in the bound:

<!-- formula-not-decoded -->

Next, for ∆ 2 , we first apply the Hermite decomposition of ˜ σ ′ to obtain:

<!-- formula-not-decoded -->

By Assumption 2, ˜ c 1 = 0 while ( ii ) in Proposition 9 implies that the terms corresponding to j &gt; 2 are bounded as 1 √ d jε 1 ( m t × ) j

We therefore obtain:

<!-- formula-not-decoded -->

for some constant ˜ C &gt; 0 . Lastly, Lemma 7 implies that:

<!-- formula-not-decoded -->

Since t &lt; τ -˜ δ , the above bounds on ∆ 1 , ∆ 2 , ∆ 3 can be absorbed within arbitrarily small constants compared to m :

<!-- formula-not-decoded -->

for arbitrarily small constant C &gt; 0 .

This results in the bound:

<!-- formula-not-decoded -->

where ˜ c ≤ c + ˜ ε for arbitrarily small ˜ ε . Next, we use the inequality √ 1 + t -1 ≥ (1 -t 2 ) for t ≥ 0 to obtain:

<!-- formula-not-decoded -->

where in the last line we applied the control over the squared gradient norm in Lemma 11 and c g &lt; c can again be set arbitrarily close to c . Combining with Equation 150 yields part ( i ) .

Next, for part ( ii ) , introduce the operator:

<!-- formula-not-decoded -->

corresponding to the projection onto the orthogonal complement of u ⋆ in U ⋆ . Let u t ⊥ := ( I -( u ⋆ )( u ⋆ ) ⊤ ) P U ⋆ w ( t ) .

<!-- formula-not-decoded -->

Since ∥ P U ⊥ w t ∥ = m t ⊥ and ∥ P U ⋆ w t ∥ = m t , the first term simplifies to:

<!-- formula-not-decoded -->

By Equation 145, we have:

<!-- formula-not-decoded -->

for some constant C &gt; 0 .

Similarly, by Equation 147, we obtain a bound:

<!-- formula-not-decoded -->

The above combine to result in the term ˜ ηC 2 ( m t × ) in Equation 126. Finally, by Equation 108 in Proposition 11, ∆ 3 P U ⊥ is bounded as:

<!-- formula-not-decoded -->

yielding the last term in Equation 126.

Analogously, part ( iii ) follows by considering the terms ⟨ w ⋆ i , ∆ j ⟩ , i ∈ √ d ε 1 for j = 1 , 2 , 3 for ( iii ) respectively, with the bound on ∥ g t ∥ 2 remaining the same.

## C.14 Feature Learning by the Second Layer

To motivate our setup for the training of W 2 , we start with a heurestic discussion of the dynamics of gradient updates in the absence of pre-conditioning and projections. Throughout the subsequent discussions, we denote n 2 and p 1 by n, p respectively. The presentation of formal results towards the proof of part ( ii ) of Theorem 1 starts from Section C.18.

## C.15 Updates in Feature Space and Projection Onto the Kernel

Under correlation loss L c = -f ⋆ ( x ) ˆ f ( x ) , the gradient update for a neuron w i, 2 , i ∈ [ p ] in the second layer has the following form:

<!-- formula-not-decoded -->

Under the approximation ˆ f ( x ) ≈ 0 , the updated pre-activation out of at a fixed input x are thus given by:

<!-- formula-not-decoded -->

Letting h t 2 ,i ( x ) := ⟨ w t i, 2 , σ ( W 1 ( x )) ⟩ , we obtain:

<!-- formula-not-decoded -->

with X ∈ R n 2 × d the data matrix and Z denotes the feature-mapping σ ( W 1 X ⊤ ) .

We see that in the limit n, p 1 →∞ , the above update results in a projection of f ⋆ on the following Kernel (integral operator):

<!-- formula-not-decoded -->

where µ 1 denotes the distribution of the rows of W 1 obtained upon feature learning in part ( i ) .

## C.16 The Role of Preconditioning

In light of Equation 159, we obtain a dynamics of the form:

<!-- formula-not-decoded -->

where K 1 ( x , x ′ ) f ⋆ ( x ) σ ′ ( h t 2 ,i ( x )) denotes the projection of f ⋆ ( x ) σ ′ ( h t 2 ,i ( x )) onto the Kernel K 1 . Through a central limit theorem-based heurestic, we expect the noise to be of order O ( 1 √ n + 1 √ p ) [66]. However, the decay in K 1 's spectrum, entails that the degreek components in K 1 f ⋆ ( x ) σ ′ ( h t 2 ,i ( x )) are of order O ( 1 d k ) . Comparing O ( 1 √ n + 1 √ p ) and O ( 1 d k ) , one expects a sample-complexity of d 2 k for recovering h ⋆ ( x ) through a single (non-preconditioned) gradient step. For quadratic features, this is precisely the sample-complexity obtained in [66].

A possible way to get around the additional sample complexity would be to re-use a single batch of size O ( d k ε ) for up to O ( d k ε ) steps, ensuring that the projection on the Kernel is well-approximated at each step while the number of steps are enough for the dynamics described by Equation 159 to approximate ridge-regression, which effectively has the same effect as pre-conditioning through the removal of the learned components. However, analyses of gradient descent with the re-use of batches for a large number of iterations is expected to be challenging due to the accumulation of additional correlations and memory terms [32].

Therefore, to allow a simplified 'online" analysis we opt to include additional pre-conditioning in the updates, which effectively removes the extra 1 d kε factor from Equation 159.

Remark : Under the additional assumption that E z ∼ N (0 , 1) [ σ ( z ) z ] = 0 , [66] improved the samplecomplexity for recovery of quadratic features from O ( d 4 ) to O ( d 2 ) . Such an assumption on σ ( · ) however, appears insufficient towards reducing the general O ( d 2 k ) sample complexity to O ( d k ) for general degree k components.

## C.17 Main Result for part (ii)

This section deals with the recovery of the non-linear features h ⋆ ( x ) .

Theorem 6. Let W 1 2 denote the updated layer 2 weights after a single pre-conditioned gradient step with batch-size n 2 , with initialization W 0 2 = 0 p 2 × p 1 as in Algorithm 1:

<!-- formula-not-decoded -->

where W 1 ∈ R p 1 × d denotes the updated weight matrix with independent rows obtained as per Theorem 5. The updated pre-activations h 1 2 ( x ) = W 1 2 σ ( W 1 x ) then satisfy:

<!-- formula-not-decoded -->

where w 3 is the readout scalar weight and the remainder r ( x ) satisfies:

<!-- formula-not-decoded -->

## C.18 Structure of the Pre-conditioned Update

Let Z denote the feature matrix Z = σ ( XW ⊤ 1 ) applied to an independent data-matrix X ∈ R n 2 × d using the updated weights W 1 obtained in part ( i ) . Throughout the section, we assume that the threshold parameter κ &gt; 0 in Theorem 5 is fixed to some dimension-independent value and occasionally consider the limit κ → 0 (but after d → 0 ). Denote by Z ( x ) the same mapping applied to a fixed point x ∈ R d .

The proposition below expresses a pre-conditioned gradient update on W 2 as a "Kernel-ridge regression like" update to h t 2 ( x ) .

Proposition 13. Suppose that W 2 is re-initialized to 0 . The updated pre-activations h t 2 ( x ) satisfy for i ∈ [ p 2 ] :

<!-- formula-not-decoded -->

## C.19 Decomposition into Radial and Spherical Kernels

Let l d k , Q d k for k ∈ N denote the associated Laguerre and Gegenbauer polynomials in dimension d . Recall that U ⋆ denotes the span of W ⋆ .

From Theorem 5, for any i ∈ [ p 1 ] , the updated neuron w 1 i can be decomposed as:

<!-- formula-not-decoded -->

where ∥ ∥ w 1 i ∥ ∥ = 1 , u i ∈ U ⋆ and v i ∈ U ⋆ ⊥ .

For any x ∈ R d , denote by x ⋆ , x ⊥ , its components along U ⋆ and U ⋆ ⊥ respectively.

Next, we decompose the inner-product ⟨ w i , x ⟩ as:

<!-- formula-not-decoded -->

By the Gaussianity of x , the random variables ∥ x ⋆ ∥ , ⟨ u , x ⋆ ∥ x ⋆ ∥ ⟩ , ∥ ∥ x ⊥ ∥ ∥ , ⟨ v , x ⊥ ∥ x ⊥ ∥ ⟩ are mutually independent. The variables ∥ x ⋆ ∥ 2 and ∥ x ⊥ ∥ 2 are distributed as χ 2 variables with d ϵ 1 and d -d ϵ 1 degrees of freedom respectively and hence admit the associated Laguerre polynomials as an orthonormal basis (Section C.3.2). Therefore, ⟨ w , x ⟩ admits an orthonormal basis given by the tensor product of associated Laguerre and Gegenbauer polynomials.

By expanding σ along this bases of associated Laguerre and Gegenbauer polynomials, the activation σ ( ⟨ w , x ⟩ + b ) can then be decomposed as:

σ

(

⟨

w

i

,

x

⟩

+

b

)

<!-- formula-not-decoded -->

where l d k , P d k denote the associated Laguerre and Gegenbauer polynomials in dimension d respectively. The above convergence holds in L 2 w.r.t x ∼ N (0 , I d ) . Proposition 14. For all k ∈ N :

<!-- formula-not-decoded -->

Proof. Let d ⊥ := d -d ε 1 , then:

<!-- formula-not-decoded -->

Let r ⋆ = ∥ x ∥ 2 ⋆ -1 √ d and r ⊥ = ∥ x ∥ 2 ⊥ -1 √ d . Subsequently, the result follows through the Taylor expansion √ 1 + z = 1 + z 2 + o ( z 2 ) w.r.t r ⋆ , while noting that √ d ε 1 ⟨ u, x ⋆ ∥ x ∥ ⋆ ⟩ → N (0 , ˜ κ ) and √ ( d ⊥ ) ⟨ v , x ⊥ ∥ x ⊥ ∥ ⟩ → N (0 , 1 -˜ κ ) by Theorem 5 for some ˜ κ &gt; κ .

The Taylor expansion implies that the coefficient of r ⋆ converges to µ k ( b ) 1 √ d ε 1 k . On the other hand, the coefficient must also equal ∑ j ≥ k a d j, 0 , 0 , 0 c jk , where c jk denotes the coefficient of z k in the k th associated Laguerre polynomials.

In light of the above decomposition, we introduce the following sequence of radial Kernels, with k 1 = k 2 = j 2 = 0 :

<!-- formula-not-decoded -->

Proposition 15. Under Assumption 1, h ⋆ 1 , 2 , K d 0 admits uniformly continuous eigenfunctions ϕ 1 ,d , ϕ 2 ,d with associated eigenvalues λ 1 = Θ(1) , λ 2 = Θ( 1 √ d ε 1 ) such that r ⋆ ( x ) = 1 √ d ε 1 ( ∥ x ⋆ ∥ 2 -1) satisfies:

<!-- formula-not-decoded -->

where P denotes projection in L 2 ( µ ( x )) .

Proof. By proposition 14, a d k converge a.s to deterministic limits a d k as d → ∞ . We obtain the following limiting expression for K k ( x 1 , x 2 ) :

<!-- formula-not-decoded -->

Note that:

<!-- formula-not-decoded -->

By assumption on σ , µ 2 0 ( b ) is analytic in b and hence non-zero almost sure w.r.t b ∼ N (0 , 1) . Therefore, by the variational characterization of eigenvalues for compact self-adjoint operators [16], we obtain:

implying:

Similarly, we obtain:

<!-- formula-not-decoded -->

implying:

<!-- formula-not-decoded -->

̸

Analogously, since σ is analytic, with probability 1 , a j, 0 = 0 , ∀ j ∈ N , we obtain that the j th eigenvalue for K 0 ( x , x ′ ) satisfies:

<!-- formula-not-decoded -->

Equation 172 then follows by noting that:

<!-- formula-not-decoded -->

The continuity of the eigenfunctions then follows since we have:

<!-- formula-not-decoded -->

Since K ( x , x ′ ) is uniformly continuous in x .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.20 Decomposition of the Feature Matrix

For j i , k i ∈ N , let θ d j 1 ,k 1 ,j 2 ,k 2 denote the eigenfunctions of the radial Kernel K d k for k ∈ N , defined as (Generalizing Equation 171):

<!-- formula-not-decoded -->

Analogously, let κ d j 1 ,k 1 ,j 2 ,k 2 denote the eigenfunctions of the associated companion Kernel defined on the weights:

<!-- formula-not-decoded -->

Define:

<!-- formula-not-decoded -->

And for the conjugate:

<!-- formula-not-decoded -->

With a slight abuse of notation, we denote θ d j, 0 , 0 , 0 and κ d j, 0 , 0 , 0 by θ d j and κ d j respectively. These correspond to eigenvalues for the zeroth-order radial Kernel along U ⋆ .

We partition the indices j 1 , j 2 , k 1 , k 2 into three disjoint sets:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above partitioning is motivated as follows:

- (i) S 1 corresponds the set of eigenfunctions whose projections can be approximated via Z with n 2 , p 2 = Θ( d kε 1 ) and are relevant towards learning f ⋆ ( x ) .
- (ii) S 2 corresponds to the set of eigenfunctions whose projections can be approximated by Z but do not contribute to the learning of f ⋆ ( x ) .
- (iii) S 3 corresponds to the high-degree set of eigenfunctions for which the number of samples, neurons n 2 , p 2 are insufficient towards being approximated through Z .

Let Θ d j , K d j denote matrices with rows θ d j 1 ,k 1 ,j 2 ,k 2 ( r ⋆ , r ⊥ ) and κ d j 1 ,k 1 ,j 2 ,k 2 ( b, ∥ u ∥ , ∥ v ∥ ) respectively. Similarly, let Ψ j 1 ,j 2 ,k 1 ,k 2 ( X ) , Φ j 1 ,j 2 ,k 1 ,k 2 ( W ) denote matrices with rows ψ j 1 ,j 2 ,k 1 ,k 2 ( x ) and ϕ j 1 ,j 2 ,k 1 ,k 2 ( x )

Expressing Equation 168 in matrix form and applying Proposition 3 to expand each term Q d k ( · ) , we obtain the following decomposition:

<!-- formula-not-decoded -->

where D r j , D s j denote diagonal matrices with entries ( b d j ) 2 , ( a d j ) 2 respectively. We denote the above three-components corresponding to S 1 , S 2 , S 3 as Z 1 , Z 2 , Z 3 respectively.

<!-- formula-not-decoded -->

## C.21 Approximation of Eigenfunctions

Let M = | S 1 | ∪ | S 2 | . Since B ( d, k ) = Θ( d k ) (section C.3.2), we obtain M = Θ( d kε ) . We next show that the above partitioning of eigenfunctions translates to a 'spike"+bulk structure for Z , with the spikes arising from components corresponding to S 1 , S 2 allowing the reconstruction of the corresponding eigenfunctions through the sample-covariance. The higher-degree components S 3 , on the other hand, coalesce into a bulk.

For each x ∈ R d , let ψ S 1 ∪ S 2 ( x ) ∈ R M denote the combined vector of components along eigenfunctions indexed by S 1 , S 2 , i.e:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These properties are summarized in the following proposition, which constitutes the central result of this section:

Proposition 16. There exists a sequence c d with c d = O (1) such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Before proceeding with the proof, we highlight the key-takeaways from the above result. Points ( i ) , ( ii ) imply that the matrices Z 1 , Z 2 contribute M spikes to Z with left, right singular vectors aligned with ψ S 1 ∪ S 2 ( x ) and ϕ S 1 ∪ S 2 ( w ) respectively. Points ( ii ) , ( iv ) imply that the high-degree components Z 3 contribute an approximately isotropic bulk, that doesn't interfere with the spikes along ϕ S 1 ∪ S 2 ( w ) . Note that ( iv ) is necessary since the large rank of Z 3 could cause the corresponding components to collectively interfere with the low-degree components.

The crucial consequence is that the spikes in Z 1 , Z 2 allow effective reconstruction of the components along S 1 ∪ S 2 . In contrast, the failure of Z 3 to estimate the covariance structure along S 3 prevents the recovery of such high-degree components.

Proof. Equation 189 is a direct consequence of Lemma 2 and the hyper-contractivity of the spherical measure. Equation 190 however, requires additional control over the error in w .

Westart with showing that the covariance is well-approximated in expectation Let v ∈ R n , ∥ v ∥ = 1 denote an arbitrary fixed unit vector. Then:

<!-- formula-not-decoded -->

since ψ 2 i are uniformly lipschitz on S d , applying a taylor expansion on w around u ⋆ yields:

<!-- formula-not-decoded -->

Analogously, define:

where we used that h v ( w ) = E [ ∑ s ∈ S 1 ∪ S 2 v 2 s ϕ 2 i ( w ) ] is an even polynomial in w . Therefore, E [ ∇ h v ( w )] = 0 while ∥ ∥ E [ ∇ 2 h v ( w ) ]∥ ∥ ≤ C for some constant C &gt; 0 . Corollary 1 then ensures that the second order-term is bounded as O ( 1 d δ ) . Taking supremum over v for ∥ v ∥ = 1 , we obtain:

<!-- formula-not-decoded -->

for some δ &gt; 0 . We move on to establishing the concentration of Φ S 1 ( w ) . By Equation 28 in [63], spherical harmonics Y m,k of degree k ∈ N admit a basis with the following representing along the cartersian coordinates:

<!-- formula-not-decoded -->

where α ∈ N d contains at-most ℓ -non-zero entries. Therefore, Y α ( w ) is a polynomial in at-most ℓ coordinates in w along with the ℓ projection norms r j = √ ∑ j i =1 w 2 j . Applying part ii, c of Theorem 5 then implies that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Subsequently:

and:

Taking a union bound over the Θ( d kε 1 ) values of α yields:

<!-- formula-not-decoded -->

For the radial components recall that ∥ u ∥ , ∥ v ∥ = O ≺ (1) while the radial eigenfunctions are continuous.

We conclude that:

<!-- formula-not-decoded -->

Setting ˜ δ &lt; δ, δ ′ and recalling that, n 2 = Θ( d kε 1 + δ ) , p 2 = Θ( d kε 1 + δ ′ ) while | S 1 ∪ S 2 | = Θ( d kε ) , we may apply Lemma 2 to obtain:

<!-- formula-not-decoded -->

where we absorbed the d ˜ δ factor into the 1 p factor in the bound in Lemma 2 (Equation 63).

The proof of ( iii ) similarly follows from Propositions 4, 8 in [60]. We outline the central steps. First, via the expansion of σ ( · ) given by Equation 168, for any x , Ψ S 3 ( x ) ⊤ Φ S 3 ( w ) can be expressed as σ ( ⟨ w , x ⟩ + b ) -Ψ S 1 ∪ S 2 ( x ) ⊤ Φ S 1 ∪ S 2 ( w ) . Through Equation 196, Ψ S 3 ( x ) ⊤ Φ S 3 ( w ) therefore depends on w only through a finite number of coordinates in U ⋆ . Analogous to Equation 202 above, applying Lemma 2 and using p &gt;&gt; n , we obtain that:

<!-- formula-not-decoded -->

where G 3 ( X,X ) denotes the gram-matrix associated to the Kernel:

<!-- formula-not-decoded -->

applied to the data-matrix X ∈ R n × d .

The gram-matrix G 3 ( X ⋆ , X ⋆ ) now corresponds exactly to the spherical distribution on U ⋆ , with decay identical to the case of spherical data in [58]. Therefore, proposition 8 in [58] applies, which entails that the off-diagonal contributions from G 3 ( X ⋆ , X ⋆ ) are neglible in operator norm. This results in the bound:

<!-- formula-not-decoded -->

for some constant c &gt; 0 , implying ( iii ) and ( iv ) .

## C.22 Properties of the Feature-covariance Matrix

Having established Proposition 16 and the concentration of the top eigenvectors, the setting of Z is now reduced to the spike + 'bulk" structure in the proof of Theorem 1 in [60] with Θ( d kε 1 ) spikes arising from the eigenfunctions S 1 , S 2 corresponding to near-identity sample-covariances and a remaining bulk with uniformly-bounded operator norm. A consequence of such a structure is that the top singular vectors of Z align closely with these 'spikes". This ensures that projections onto Z 'reproduce" functions in S ∪ S 2

Therefore, the proofs of Propositions 6 , 7 in [60], based on perturbation inequalities for singular values, singular vectors, result in the following estimates for Z :

Proposition 17. 1 √ p Z admits a singular value decomposition

<!-- formula-not-decoded -->

such that:

<!-- formula-not-decoded -->

- (ii) ∥ S 3 -c 3 I ∥ = o d,p (1) , for some constant c 3 &gt; 0 .

<!-- formula-not-decoded -->

The proof of the above Proposition follows directly through Proposition 6 in [60]. The above result exactly charaterizes the projections of functions onto pre-conditioned features:

Proposition 18. For any g : R d → R such that the projections onto radial components of degree &gt; 2 are o d (1) , for any λ 2 = Θ( p n ) :

<!-- formula-not-decoded -->

The proof of part ( ii ) of Theorem 1 is then completed by showing that under Assumption 1, the projection onto S 1 is exactly along h ⋆ ( x ) :

Proposition 19. Under Assumption 1:

<!-- formula-not-decoded -->

Proof. By Assumption 3 and the composition of Hermite decompositions (Lemma 9), the nonvanishing terms along the radial component ∥ x ∥ 2 -1 consists of total input-degree2 and 2 k while the remaining terms on the complement of h ⋆ ℓ have degree at least 3( k + 1) &gt; k . S 1 therefore consists exactly of the subspace with effective degree k .

## C.23 Proof of Proposition 18

Let ˆ g ( x ) = Z ( x ) ⊤ ( 1 n Z ⊤ Z + λ 2 I ) -1 Z ⊤ g ( X ) . Proposition 18 is equivalent to ∥ ˆ g ( x ) -g ( x ) ∥ 2 2 = o d (1) . Expanding, we obtain:

<!-- formula-not-decoded -->

It therefore suffices to show that:

<!-- formula-not-decoded -->

and:

<!-- formula-not-decoded -->

Let g S 1 denote the vector with components:

<!-- formula-not-decoded -->

Let Λ ≤ 2 ,k denote the diagonal matrix with the corresponding eigenvalues.

Then the above terms can be expressed as:

<!-- formula-not-decoded -->

and:

<!-- formula-not-decoded -->

where Σ denotes the feature covariance:

<!-- formula-not-decoded -->

To compute the above terms, we use Proposition 17 to estimate certain intermediate quantities similar to Proposition 7 in [60]:

Proposition 20. Under the setup of Theorem 1, with the decomposition of eigenfunctions specified by Equation 186 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under the above proposition, the terms given by Equations 213, 214 simplify as follows:

<!-- formula-not-decoded -->

By Equation 216, the first term converges to ∥ P S 1 g ( x ) ∥ 2 while the other two terms are bounded as O ≺ ( 1 d δ ) by Equations 217,218 respectively.

Similarly,

<!-- formula-not-decoded -->

By Equation 218, the second term is bounded as O ≺ (1) . This completes the proof of part ( ii ) of Theorem 5.

<!-- formula-not-decoded -->

## C.24 Proof of part (iii): Fitting the Target

Upon the completion of part ( ii ) , the second-layer pre-activations h 2 ( x ) = W 2 σ ( W 1 x ) are approximately equivalent to those of a random feature-mapping applied to the scalar input h ⋆ ( x ) , with the random weights of the feature mapping given by ˜ w = cw 3 , with c = ησ ′ (0) as in Proposition 13. Hence, we introduce the Kernel K ( · , · ) : R 2 → R :

<!-- formula-not-decoded -->

For Z ∈ R n , we further denote by K ( Z, Z ) the corresponding gram-matrix K ( Z, Z ) ∈ R n × n , with entries

<!-- formula-not-decoded -->

Let H K denote the RKHS corresponding to the Kernel K . Let H ⋆ ∈ R n × p 1 further denote the matrix with rows h ⋆ ( x µ ) .

Since the moments of h ⋆ ( x ) are uniformly bounded in d , we obtain: Proposition 21. [[25]] For any δ &gt; 0 , and large enough d , ∃ constants c, C such that with λ = Θ( √ n ) :

<!-- formula-not-decoded -->

where H ⋆ ∈ R N contains independent samples h ⋆ ( x ) , and N K ( λ ) denotes the 'effective-dimension":

<!-- formula-not-decoded -->

which admits the following trivial bound:

<!-- formula-not-decoded -->

We next translate the above bound into generalization error through a control of the approximation error term.

Note that the uniform bounds on the moments of h ⋆ ( x ) and Markov's inequality, for any ε &gt; 0 , ∃ R ε &gt; 0 such that for large enough d :

<!-- formula-not-decoded -->

Next, define the following class of functions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then:

Restricted to the compact set B R ε , universality of random feature Kernels with non-polynomial, polynomially-bounded activations [78] implies that for any ε &gt; 0 , ∃ f ε ∈ H K such that:

<!-- formula-not-decoded -->

Therefore, by setting λ small enough such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we otbain:

By Cauchy-Shwartz and the uniform bound on E [ g ⋆ ( h ⋆ ) 2 ] , the last term in the RHS is bounded by Cε for some cosntant C &gt; 0 .

Subsequently, we may set n in Proposition 21 large enough such that:

<!-- formula-not-decoded -->

Implying that for small enough λ ( ε ) and large enough n ( ε, δ ) , with probability 1 -δ :

<!-- formula-not-decoded -->

for some constant C &gt; 0 .

Now, returning to the true features h 2 ( x ) , it remains to combine the above estimate with the concentration of the gram-matrix to the associated Kernel. This is established similar to the proof of Proposition 16 through Lemma 2.

Note that the above argument does not yield the dependence of λ, n on ε . Such an explicit dependence requires finer control on the approximation, source terms. For such an analysis, we refer to the explicit rademacher complexity based bounds for ReLU activation in [29].

We remark that more quantitave estimates can be obtained through rademacher-complexity based analysis for specification activations such as Relu [29].

## C.25 Relaxing Assumption 3

In this section, we adress the requirement of Assumption 3 and steps towards relaxing it. Assumption 3 simplifies our analysis by ensuring that P S 1 f ⋆ ( x ) is exactly h ⋆ ( x ) arising from the first-order Hermite coefficient of g ⋆ ( x ) . In general, however, the degreek approximation of f ⋆ ( x ) may contain additional components involving higher-degree dependence on h ⋆ ( x ) . For instance, if g ⋆ ( x ) has a non-zero second-order Hermite coeffient, then Lemma 9 implies that He 2 ( h ⋆ ( x )) can be decomposed into components of Hermite degree 4 , · · · , 2 k . Therefore, if k ≥ 4 , gradient updates to W 2 result in h 2 ( x ) ≈ c ( h ⋆ + higher order components ) . While ideally one would hope that the learning of such additional components would only help towards fitting f ⋆ ( x ) by w 3 , this would require the second-layer pre-activations to disentangle h ⋆ and the remaining components i.e. to specialize across non-linear features. Analysis of such a specialization remains challenging due to the reasons described in Appendix 2. Therefore, relaxing Assumption 3 requires going beyond the single-spike ( r = 1 ) non-linear feature learning.

Additionally, as we saw through the decomposition of the activation into radial and spherical arguments (Equation 168), the radial components exhibit slower-decay w.r.t the degree. Therefore, d kε samples, neurons suffice towards learning degreek components on 1 √ d ε 1 ∥ x ⋆ ∥ 2 which correspond to degree 2 k components on x . We believe this to be an artifact of our choice a ⋆ = 1 , which leads to a special dependence along the radial component. Going beyond the isotropic a ⋆ = 1 setting is however, challenging due to our reliance on diagonalization of the associated Kernel along a fixed basis.

## D Deeper networks: Proof of Theorem 2

## D.1 Independence of features

The independence of h ⋆ ( x ) follows by noting that by induction, for all ℓ ∈ [ L ] , distinct components of h ⋆ ℓ ( x ) depend on projections of x along distinct subspaces.

Lemma 9 (Block-wise independence of hidden features) . For every layer index ℓ ∈ [ L ] , the random variables { h ⋆ ℓ,m ( x ) } d ε ℓ m =1 are mutually independent and each has zero mean and unit variance.

Proof. We prove this result by induction.

- Base case ( ℓ = 1) . The first-layer features are h ⋆ 1 ( x ) = W ⋆ x with orthonormal rows. Hence the components ⟨ w ⋆ 1 ,m , x ⟩ are i.i.d. N (0 , 1) and independent.
- Induction step. Assume the claim holds for layer ℓ -1 . Fix m ∈ [ d ε ℓ ] and recall

<!-- formula-not-decoded -->

̸

̸

where B m is the m -th disjoint block of indices of size d ε ℓ -1 -ε ℓ . By the induction hypothesis the entries of h ⋆ ℓ -1 , B m ( x ) are independent of those in any other block B m ′ ( m ′ = m ) . Because P k,m,ℓ and the inner product with a ⋆ ℓ,m are deterministic maps, h ⋆ ℓ,m ( x ) depends only on block B m and is independent of h ⋆ ℓ,m ′ ( x ) for m ′ = m . The variance-normalization follow from the definition of P k,m,ℓ and the scaling 1 / √ d ε ℓ -1 -ε ℓ .

This concludes the proof.

## D.2 Proof of Theorem 2

By the independence and asymptotic Gaussianity of the features h ⋆ ℓ ( x ) we expect the above result to extend to a general number of layers. However, proving such a result in its full-generality requires accounting for the non-asymptotic rates for the tails of h ⋆ ℓ ( x ) and the associated kernels.

Instead, we prove a weaker result corresponding to the hierarchical weak-recovery of a single non-linear feature at a general level of depth, given by Theorem 2.

The central tool underlying our proof is a propagation of hyper-contractivity through the layers: Proposition 22 (Propagation of Hyper-contractivity) . Let f : R → R be a polynomial of finitedegree k . Then, for any ℓ ∈ N :

- (i) h ⋆ ( x ) = O ≺ (1) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof proceeds by induction. For ℓ = 1 , the statements hold by Gaussianhypercontractivity (Lemma 5) since He k for distinct w ⋆ i are uncorrelated, zero-mean random variables and thus h ⋆ 2 ( x ) has all moments of bounded order.

Suppose the statements hold for some ℓ ∈ N . Applying Lemma 1, we obtain:

<!-- formula-not-decoded -->

for any δ &gt; 0 . Subsequently, applying 8 leads to the following propagation of tails:

<!-- formula-not-decoded -->

where we used the bound h ⋆ ℓ ( x ) = O ≺ (1) by the induction hypothesis. By Lemma 8 and Equation 233, for any δ &gt; 0 , we obtain that:

<!-- formula-not-decoded -->

The above proposition establishes that the hidden features h ⋆ ℓ ( x ) maintain errors in means O ≺ ( 1 √ d ε ℓ ) and preserve tails of the form O ≺ (1) . Theorem 2 then follows by noting that the above error bounds suffice for Proposition 16 to hold for the feature-matrix σ ( Wh ⋆ L -1 ( x )) . Concretely, h ⋆ ℓ ( x ) = O ≺ (1) ensures that Lemma 2 applies while the errors in means, covariances suffice for the expected covariance of spherical harmonics to converge to I .

Analogous to Section C.20, we introduce the following partitioning of the indices:

<!-- formula-not-decoded -->

Above, we only have two partitions as opposed to the three partitions in Section C.20 since the features h ⋆ L -1 ( x ) are no longer partitioned into disjoint spaces, unlike the partitioning of x into x ⋆ , x ⊥ in Section C.14.

We again write:

<!-- formula-not-decoded -->

Unlike Proposition 16 that involved approximations in W , the above decomposition involves approximating h ⋆ L -1 ( x ) through equivalent Gaussian-inputs x . The proof follows that of Proposition 16, with Proposition 22 implying that:

<!-- formula-not-decoded -->

For the corresponding non-linear features along W , since the rows of W are independently sampled along U ( S d (1) , we directly have:

<!-- formula-not-decoded -->

The remainder of the proof follows that of Propositions 17 and 18.

## E Extension to MIGHTs

While our analysis is restricted to r = 1 , we discuss here the primary challenges and directions towards the extension of our results to r &gt; 1 :

- (i) Spherical recovery : Under the assumption, a ⋆ i = 1 , and µ 1 ( g ⋆ ) = c (1 , 1 , · · · ) for some constant c , our analysis for the recovery of W ⋆ 1 , · · · , W ⋆ r by W 1 remains identical. Specifically, each neuron w 1 i recovers u ⋆ = P W⋆ 1 , ··· ,W ⋆ r w 1 i ∥ ∥ ∥ P W⋆ 1 , ··· ,W ⋆ r w 1 i ∥ ∥ ∥ .
- (ii) Specialization : Even under the above symmetric setup, a single pre-conditioned gradient step only leads to recovery by h 2 ( x ) of the symmetric direction 1 √ r ∑ r i =1 h ⋆ i ( x ) . Hence, extension of our analysis to r &gt; 1 requires specialization through multiple pre-conditioned gradient steps. One promising approach to achieve such specialization is through the use of the staircase mechanism [2, 1] in the target g ⋆ ( · ) .

## F Details on the Numerical Investigation

In this section, we provide additional insights into the numerical illustrations presented in the main text. We refer to https://github.com/IdePHICS/ComputationalDepth for the code.

## F.1 Shallow methods

We illustrate in Fig. 2 the performance of two shallow methods: kernels (orange) and two-layer networks (green). At stake with three-layer architectures (red and blue), shallow methods are not able to perform non-linear feature learning, hence resulting in suboptimal performance. Below, we provide additional clarifications on these methods.

Kernel methods We consider a quadratic kernel k ( x 1 , x 2 ) = ( x ⊤ 1 , x 2 ) 2 + ( x ⊤ 1 , x 2 ) + c = φ ⊤ quad ( x 1 ) φ quad ( x 2 ) that is an optimal choice among kernel mappings in the data regime explored ( n = o d ( d 2+ δ ) ), as follows by the asymptotics results in [60]. The feature map φ quad is not learned, therefore we refer to kernel methods as 'fixed feature' methods. The lack of feature learning, and therefore adaptation to the relevant low-dimensional subspaces present in the SIGHT target f ⋆ , results in a large error value achieved by the best possible kernel methods (signaled with an orange solid line in Fig. 2) that serve as a lower bound for the simulations (shown as orange points). This bound coincides with the best quadratic approximation of the target as shown by [60]. The figure shows also neatly the presence of the double descent peak when the number of data equals the dimension of the feature space, sometimes called the interpolation peak: n peak = d ( d -1) / 2+ d +1 ; this is illustrated by a vertical orange dashed line in the left section of Fig. 2.

Figure 5: Reinitialization of subsequent layers: The plots compare the generalization error achieved by two variants of the layerwise procedure in Theorem 1. The left panel illustrates a routine with reinitialization of the subsequent layers against a procedure where this assumption is relaxed in the right panel. There is no substantial difference between the two algorithms when looking at the generalization performance. The target is f ⋆ ( x ) = tanh ( a ⋆ ⊤ P 3 ( W ⋆ x ) / √ d ε 1 =1 / 2 ) and the hyperparameters are listed in Sec. F.4.

<!-- image -->

Figure 6: Reuse of the same data batch over layers: The plots compare the generalization error achieved by two variants of the layerwise procedure in Theorem 1. The left panel illustrates a routine without using the same batch of data for different layers of training, while on the right this assumption is relaxed by always holding constant the total number of samples seen for every layer. There is no substantial difference between the two algorithms when looking at the generalization performance. The target is f ⋆ ( x ) = tanh ( a ⋆ ⊤ P 3 ( W ⋆ x ) / √ d ε 1 =1 / 2 ) and the hyperparameters are listed in Sec. F.4.

<!-- image -->

Two-layer networks Two-layer networks are able, on the other hand, to capture linear features in the SIGHT target f ⋆ (denoted W ⋆ in eq. (1)). This is exemplified in Fig. 2 by the green points, with a net decrease in the test error with respect to kernel methods (orange ones). The generalization error shows a transition around the expected κ = 1 . 5 , where Theorem 1 predicts that the linear features W ⋆ are recovered (shown in the illustration by a vertical black line). However, we observe that two-layer networks in this setting cannot surpass the green solid line, corresponding to the best quartic approximation of the target. This is explained by the fact that, although partial dimensionality reduction has been achieved d → d ε 1 = √ d , two-layer networks are still performing random features in a √ d -dimensional space. Therefore, with n ≃ p = O ( d 2 ) = O ( √ d 4 ) samples and neurons, we can fit the best quartic approximation of the target [60].

## F.2 Three layer networks

The results portrayed in Fig. 2 show a stark contrast between two and three-layer networks, with the latter surpassing the best possible performance for a shallow network (green solid line) thanks to the presence of non-linear feature learning.

We consider two training routines: a) the layerwise procedure, resembling Theorem 1 and algorithmically described in Alg. 1; b) training using backpropagation and vanilla regularized gradient descent for all the layers jointly.

Figure 8: Training/Validation loss: The plots illustrate the behavior of the training and validation losses as a function of the iteration time. It shows respectively on the left the layerwise training procedure inspired by Theorem 1 (Alg. 1), while on the right standard joint training using backpropagation. The target is f ⋆ ( x ) = tanh ( 3 a ⋆ ⊤ P 3 ( W ⋆ x ) / √ d ε 1 =1 / 2 ) and the hyperparameters are listed in Sec. F.4

<!-- image -->

Remark on Algorithm 1 Throughout this section we will consider a slight generalization of the routine in Alg. 1: we will update the second layer weights reusing a single batch of size O ( d kε ) for up to O ( d kε ) steps instead of using a single gradient step with preconditioning. We refer to Sec. C.16 for discussion on the difficulties of analyzing rigorously such routine.

Moreover, we do not follow all the theoretical prescriptions needed to prove rigorously the results and included in Alg. 1. The goal of Figures 5 and 6 is to exemplify the capability of lifting some of the theoretically needed assumptions. Respectively, in Fig. 5 we analyze the presence of reinitialization of subsequent layers, and in Fig. 6 we consider the presence of shared batches across layers. In both cases, we do not observe a stark difference between the two settings. Finally, while the proof scheme is limited to targets with a ⋆ i = 1 (see eq. (1)), we consider in the numerical simulations a ⋆ drawn from a Rademacher distribution rather than being constant.

Weplot the training and validation loss curves that guided our analysis in Fig. 8.

## F.3 Visualizing Feature Learning

We now show that this enhanced generalization performance is due to feature learning. Indeed, the key result in Thm 1 refers to the ability of three-layer networks to perform hierarchically dimensionality reduction through feature learning. To probe the quality of the learned representations, we shall introduce the 'overlaps' (or order parameters).

Definition 6. The order parameters for 3 -layer networks are the matrices M W ∈ R p 1 × rd ε 1 and M h ∈ R p 2 × r (with z ∼ N (0 , I d ) )

<!-- formula-not-decoded -->

The behavior of these quantities as a function of the sample complexity κ is portrayed in Fig. 7. Since we do not

Figure 7: Visualizing Feature Learning: The Frobenius norm of the overlaps M h , M W (Def. 6), respectively on the top and bottom panel, as a function of the sample complexity κ = log n log d for three-layer networks trained with the protocol described in Theorem 1 (blue circles) and standard backpropagation (red squares). Following Theorem 1, the behavior sharply changes around κ = 1 . 5 (vertical dashed line) where feature learning in both layers arises (same setting as in Fig. 2).

<!-- image -->

follow the strong prescription of Thm. 1, and are working with a low dimensional example, we do not expect a sharp 0 / 1 transition as in the idealized scenario, but instead, the components along

Figure 9: Visualizing Feature Learning: The plot shows the evolution of the Frobenius norm of the overlaps (Definition 6) as a function of the training time t for two different values of κ = log n log d , respectively κ = 1 . 2 on the left and κ = 2 on the right. Different training methods are illustrated with different colors: in blue the layerwise training (Alg. 1), in red standard joint training using backpropagation. The target is f ⋆ ( x ) = tanh ( a ⋆ ⊤ P 3 ( W ⋆ x ) / √ d ε 1 =1 / 2 ) and the hyperparameters are listed in Sec. F.4

<!-- image -->

W ⋆ to occupy a Θ(1) fraction (but not full) of the norm of W 1 . This is well obeyed (Fig. 7) and the predicted crossover at κ = 1 . 5 is clearly observed in both layerwise and joint training.

We exemplify in Fig. 9 the 'dual' plot of Fig. 7 by showing the evolution in time of the sufficient statistics M W , M h for two different values of κ = log n log d . The plot shows that when κ &lt; 1 . 5 (the critical threshold) feature learning is impossible, as it is reflected by the overlaps attaining the random guess value. On the other hand for κ &gt; 1 . 5 the overlaps grow far from the random initialization performance.

Additionally, we illustrate the evolution in time of the overlaps under the learning of MIGHT functions (eq. (4)) in Fig. 10. The figure exemplifies the necessity of Assumption 1 that refers to the generalization of the information exponents [19, 28] of the multi-index target literature to the present hierarchical setting.

## F.4 Hyperparameters

In every figure showing sufficient statistics or generalization errors, we average over 20 different seeds and plot the median. The regularization strengths for the different layers are optimized with standard hyperparameter sweeping for every value of κ plotted, while the other hyperparameters are considered fixed. More precisely, we fix:

- (i) First hidden layer size: p 1 = int( n 1 -δ max ) , with n max the maximal n probed in the respective plot and δ = 0 . 1
- (ii) Second hidden layer size: p 2 = 600 .
- (iii) Hidden layer size for two-layer network: p = int( p 1 / 25)
- (iv) Learning rates: while the orders of magnitude for the different learning rates as a function of d are provided in Alg. 1 for layerwise training we use fixed prefactor lr 1 = 1 , lr 2 = 2 . Concerning joint training we use instead for all the three layers all the prefactors equal to 0 . 2 .
- (v) Minibatch size: n b = int( 7 n 10 ) , with n = d κ .
- (vi) Iteration time: we follow the prescriptions of Theorem 1 iterating for T 1 = O (polylog( d )) steps and T 2 = O ( d 1 . 5 ) steps. In the numerical implementation we consider for layerwise training T 1 = int(15 log d) , T 2 = int(5 d 1 . 5 ) . On the other hand, for standard training using backpropagation, we iterate jointly all the layers for T 2 steps.

Figure 10: Easy and Hard Features: The plot shows the evolution of the Frobenius norm of the overlaps (Definition 6) as a function of the training time t for two different values MIGHT functions f ⋆ ( x ) = g ⋆ ( { h ⋆ l } r l =1 ) (See eq. (4)), with the non-linear features built as in Fig. 7, i.e., h ⋆ ( x ) ∝ a ⋆ ⊤ P 3 ( W ⋆ x ) and P 3 = He 2 +He 3 . The hyperparameters are listed in Sec. F.4. Different training methods are illustrated with different colors: in blue the layerwise training (Alg. 1), in red standard backpropagation. The overlap component along different directions ( h ⋆ l , l = 1 · · · r ) are signaled with different markers.

<!-- image -->
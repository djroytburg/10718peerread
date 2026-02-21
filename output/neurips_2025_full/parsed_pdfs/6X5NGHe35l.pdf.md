## Efficiently Verifiable Proofs of Data Attribution

Ari Karchmer ∗

Martin Pawelczyk †

Seth Neel ‡

## Abstract

Data attribution methods aim to answer useful counterfactual questions like "what would a ML model's prediction be if it were trained on a different dataset?' However, estimation of data attribution models through techniques like empirical influence or 'datamodeling' remains very computationally expensive. This causes a critical trust issue: if only a few computationally rich parties can obtain data attributions, how can resource-constrained parties trust that the provided attributions are indeed 'good,' especially when they are used for important downstream applications (e.g., data pricing)? In this paper, we address this trust issue by proposing an interactive verification paradigm for data attribution. An untrusted and computationally powerful Prover learns data attributions, and then engages in an interactive proof with a resource-constrained Verifier . Our main result is a protocol that provides formal completeness , soundness , and efficiency guarantees in the sense of Probably-Approximately-Correct (PAC) verification (Goldwasser et al., 2021). Specifically, if both Prover and Verifier follow the protocol, the Verifier accepts data attributions that are ε -close to the optimal data attributions (in terms of the Mean Squared Error) with probability 1 -δ . Conversely, if the Prover arbitrarily deviates from the protocol, even with infinite compute, then this is detected (or it still yields data attributions to the Verifier) except with probability δ . Importantly, our protocol ensures the Verifier's workload, measured by the number of independent model retrainings it must perform, scales only as O (1 /ε 2 ) ; i.e., independently of the dataset size. At a technical level, our results apply to efficiently verifying any linear function over the boolean hypercube computed by the Prover, making them broadly applicable to various attribution tasks.

## 1 Introduction

The attempt to understand and explain the behavior of complex machine learning systems has given rise to the field of attribution . Broadly, attribution methods seek to trace model outputs or behaviors back to their origins. This takes several forms. One prominent form is training data attribution , which aims to quantify the influence of individual training examples on model predictions, facilitating interpretability, debugging, and data valuation (e.g., Koh and Liang, 2017; Ghorbani and Zou, 2019; Ilyas et al., 2022). Complementary to this, component attribution (Shah et al., 2024) focuses on decomposing a model's prediction in terms of its internal components, such as convolutional filters or attention heads, to understand how they combine to shape model behavior.

While many applications of attribution may be of interest to the model developers, others are inherently of interest to third parties who may not have access to the internal details of how the attributions were computed. For instance, consider a framework where entities that supply data for training are compensated in proportion to the degree their training data influences model outputs, an idea which

∗ Morgan Stanley Machine Learning Research, Harvard Business School, akarchmer0@gmail.com

† Harvard Business School, martin.pawelczyk.1@gmail.com

‡ Google Research, Harvard Business School, sethneel@google.com

has been extensively discussed (Ghorbani and Zou, 2019; Castro Fernandez, 2023; Jia et al., 2023; Choe et al., 2024). In this setting, a user i may not only be concerned that the attributions for the dataset as a whole are accurate-meaning they have low predictive error in absolute terms-but that if the ground truth attribution for user j is lower than for user i , the received attributions also satisfy this property. As another example, suppose a doctor is using a model to provide diagnoses, and wants to understand what training data the model used to reach its conclusions, in order to sanity check them.

In each of these settings, rather than simply trusting that attributions are computed correctly, third parties may want to verify that the attributions are 'right,' before making important downstream decisions. The challenge arises because of a computational disparity-third parties using these attributions typically have far, far fewer computational resources than the model developer that is providing them, whether they are an individual or even another small lab in academia or industry. Thus, given the large computational expense associated with estimating ground truth attributions, we have the following motivating question:

## Is it possible to design a protocol by which a computationally-limited third party can 'verify' the correctness of attributions?

Our Contribution, in a Nutshell. In this paper, we demonstrate an interactive two-message protocol where the third party's computational cost, which is measured by the number of full model retrainings it must conduct, is independent of the attribution set size N (e.g., in data attribution N is dataset size, while in component attribution it is the number of components). That is, this number of required retrainings does not grow as a function of N . Our protocol comes with strong guarantees of correctness for the third party, even if the model developer deviates from the agreed protocol arbitrarily, and with infinite compute. To make this precise, we must specify (i) the form of attributions we study, (ii) how we measure computational cost, (iii) what our accuracy metric is, and finally (iv) what guarantees of correctness we can ( and should ) provide for our protocol.

While we will formalize each of these in section 2, we discuss each of them here in order to clearly state our contributions. Additionally, while our work applies to data attribution methods in general, we will proceed with predictive data attribution in mind, for purposes of exposition.

Predictive Training Data Attribution. In predictive training data attribution (TDA), the goal is to answer counterfactual questions: 'What would have happened to a model's behavior if we had trained it on a different subset of the data?' To formalize this, we consider a model output function , f : {± 1 } N → R . The domain of f are subsets x of a fixed training dataset S of size N (where x ∈ {± 1 } N and x i = 1 indicates inclusion of the i th data point from S and x i = -1 indicates exclusion). The model output function f ( x ) measures the (expected) model behavior if it were trained on S , and is evaluated by retraining a model θ ∼ T ( x ) via a training algorithm T , and then computing some function of θ , often the prediction θ ( z ) on an input point z (Ilyas et al., 2022), or the test error of θ on some subgroup of the data (Jain et al., 2024).

The objective of TDA is then to construct a predictive model , often a linear function g ( x ) = ⟨ a , x ⟩ , where a are the attribution scores , in order to predict the counterfactual effect of training on a different subset of the data. The quality of this predictive model can be measured by its Mean Squared Error (MSE) with respect to the true model output function f and a binomial product distribution B p over the subsets:

<!-- formula-not-decoded -->

Achieving a low MSE indicates that the attribution scores a accurately predict how different subsets of data affect the model's behavior. 4

Remark 1. While we will focus on the well-studied area of TDA throughout, the recently proposed 'component attribution, ' (Shah et al., 2024) that aims to determine which internal components of the model are primarily responsible for specific predictions, also fits neatly into our framework. In this case, S defined above corresponds to the set of total model components, sampling x ∼ B p corresponds to sampling a subset of components, and the model output function f consists of computing a forward

4 Note, f is technically a randomized function, but we will assume a fixed random seed, and that f becomes deterministic once the random seed is fixed.

pass on a test input z where activations from components ̸∈ x are set to a constant. The methods in this paper can also be used to verify the accuracy of model component attributions.

Specific Methods for Training Data Attribution. There are two major classes of algorithms for TDA: those based on retraining, and gradient-based methods like influence functions or representer point methods that leverage approximations to the model training process or architecture. In retraining based methods, first samples x i ⊂ S are drawn (typically from a product distribution), and the output function f ( x i ) is computed, which involves training a model θ ( x i ) and then computing the relevant statistic from θ ( x i ) . Different retraining-based attribution methods generally correspond to different techniques for estimating a from samples { ( x i , f ( x i ) } i M =1 computed by retraining. In Datamodels (Ilyas et al., 2022), a is computed via a Lasso regression to encourage sparsity in a , where the model output function computes the difference in correct class logit for a specific test input z . In Empirical Influence (Feldman and Zhang, 2020), the j th coefficient a j is estimated as the average of the sampled outcomes where the j th datapoint was included: M -1 ∑ x i ∈ M j f ( x i ) , where M j = { x i : ( x i ) j = 1 } . The clever work of Saunshi et al. (2022) proves that these two methods are essentially theoretically equivalent, though performance can vary in practice, in part due to optimizations such as applying the LASSO algorithm.

Most gradient-based TDA methods are based on approximations to the continuous influence function (Law, 1986; Koh and Liang, 2017; Park et al., 2023; Grosse et al., 2023; Guu et al., 2023). While these methods are more computationally efficient than exact retraining, the assumptions underlying influence functions (e.g., convexity, training to convergence) often do not hold in deep learning, potentially undermining their suitability for data attribution (Bae et al., 2022). Yet another class of TDA methods aim to approximate the model with a simpler model class, where the attributions can be computed analytically (Park et al., 2023; Yeh et al., 2018). Prior work has found that retraining methods can be more accurate than those based on influence functions or model approximations, in settings where they are both tractable to compute (Ilyas et al., 2022), but so far these retraining based methods are unable to scale to large models (Grosse et al., 2023).

Computational Cost. While highly accurate retraining-based predictive data attribution can prove invaluable for tasks like dataset selection (Engstrom et al., 2024) and sensitivity analysis (Broderick et al., 2020; Pawelczyk et al., 2023), they come with a large computational cost. Computing a fresh sample f ( x i ) corresponds to retraining an entirely new model from scratch , and the number of samples required to obtain accurate Datamodels (attributions) is large: the work of Ilyas et al. (2022) trains three million ResNet-9 networks on random subsets of the CIFAR-10 dataset to construct their attribution. The recent work by Georgiev et al. (2024) uses Datamodels for the less stringent task of machine unlearning, and still requires training &gt; 10 , 000 models. Methods based on the influence function avoid retraining but still require computations like inverse-Hessian-vector products, which can be intractable for very large models. Moreover, even methods based on influence functions may require ensembling over multiple training runs to achieve reasonable accuracy (Park et al., 2023).

Trust Issues: Naive Verification Fails. The computational barriers described above put TDA, especially retraining-based methods like Datamodels or Empirical Influence, beyond the reach of individual practitioners or even moderately resourced academic labs. This effectively centralizes the task within a few industrial or institutional powerhouses.

Centralization causes trust issues. If only a few parties can effectively perform attribution, then, given a candidate attribution a computed by the computationally powerful group, how can a third party efficiently establish the accuracy of a ? In keeping with the rich body of work relating to interactive proofs, going forward we will refer to the computationally powerful party as the Prover , and the resource-constrained third party as the Verifier .

A naive idea that seemingly fits into our setting, is for the Verifier to try to check the MSE of any proposed attributions a .

- The Verifier randomly samples a collection of subsets T ⊂ {± 1 } N of size m , obtains f ( x ) ∀ x ∈ T , and uses that data to estimate mse ( p ) ( f, ⟨ a , ·⟩ ) ≈ 1 m ∑ x i ∈ T ( ⟨ a , x i ⟩ -f ( x i )) 2 .

When m is proportional to 1 /ε 2 , the estimate is within an additive ε error from the true MSE with high probability. Hence, the Verifier can check that the Prover has given him attributions which have

MSE at most α + ε , for some pre-determined α . Most importantly, the Verifier can do this using only O (1 /ε 2 ) retraining procedures on the model-which is independent of the data set size N !

However, this is not actually satisfactory. To see why, consider again the setting of data valuation. For concreteness, imagine each data point is contributed by a user, under the agreement they will be compensated in proportion to the attribution scores for the data they contribute. Now, suppose the Prover proposes candidate attributions a such that mse ( p ) ( f, ⟨ a , ·⟩ ) = 0 . 22 . Then, when the Verifier checks the MSE, they will obtain an estimate α ′ ∈ [0 . 22 -ε, 0 . 22 + ε ] with high probability. But, what if there exists a ′ such that mse ( p ) ( f, ⟨ a ′ , ·⟩ ) = 0 . 022 ?

Indeed, in order to reduce total costs, a malicious Prover might cheat as follows:

- Compute a by brute-force Empirical Influence. 5
- Use a ′ = a · 1 2 (scalar multiplication).

Recall that in our simple data valuation setting, the Prover will pay out data sellers proportional to their attribution. Thus, by dividing the true attributions by 2, the Prover will save 50% money (at the necessary cost of increase to MSE).

As in the naive protocol, only checking the MSE fails to shed light on whether such cheating has been carried out by the Prover (without loss of generality on the exact cheating strategy). To emphasize the point further, consider the case that the Prover wants to advantage a subset of data contributors A over all other contributors, so they receive more payment. In order to do so, they modify a by increasing coordinates where i ∈ A by some value β &gt; 0 to produce a new attribution vector a ′ . If β is reasonably small, the modified a ′ might also have low MSE, and so a Verifier that merely checks the MSE is below a certain threshold would accept these attributions.

It is not a priori clear whether or not the Verifier has any way of defending itself against this form of cheating Prover, without having to compute the attributions itself. That is, without having to do the work of the Prover.

A Better Notion of Verifiability. As we have hopefully demonstrated, we need a more meaningful notion of verifiability, that goes beyond simply checking the MSE of the proposed attribution. To this end, we suggest estimating sub-optimality . Put simply, the Verifier should ensure the attribution is ' ε -close' to optimal. This can be written as verifying the fact that

<!-- formula-not-decoded -->

Here, Φ( S ) denotes the optimal attribution vector for the set S , with respect to MSE. That is,

<!-- formula-not-decoded -->

And so, in other words, verifying sub-optimality means checking if the error gap (equation 1) between the proposed attributions, and the optimal attributions, is small.

All in all, checking low sub-optimality tells the Verifier that for this specific setting, the provided attributions are nearly optimal, and therefore the Prover has acted in good faith. Thus, we propose to aim for the following (still informal) guarantees.

Informal verification Guarantees. When designing a verification protocol we will obtain:

- Completeness. If both Prover and Verifier follow the protocol, the Verifier obtains attribution scores that are approximately optimal with respect to predictive MSE, with high probability.
- Soundness. If the Prover deviates from the protocol in any way, then, with high probability, the Verifier either outputs 'abort' or still obtains attribution scores that are approximately optimal with respect to predictive MSE.
- Efficiency. The Verifier's workload scales independently of the dataset size N .

Remark 2. In practice, even an honest, powerful Prover might only compute an estimate ˆ a ⋆ of Φ( S ) due to computational constraints (e.g., using an influence function). The soundness guarantee

5 It is strongest to consider a malicious Prover to be computationally unbounded -as is customary in the theory of interactive proofs Goldwasser et al. (2019, 2021) or statistical zero knowledge (Vadhan, 1999), and others.

still holds relative to the true Φ( S ) . However, the completeness guarantee might be affected if the honest Prover's estimate ˆ a ⋆ is itself far from Φ( S ) . If err(ˆ a ⋆ , Φ( S )) is inherently large due to the Prover's own estimation limitations, the Verifier might reject even an honest Prover if ε is set too small. Therefore, the choice of ε in practice should reflect not only the Verifier's desired precision but also potentially incorporate a tolerance for the best achievable estimation error by an efficient (though powerful) honest Prover.

## 1.1 Our Contributions

Conceptual contributions. Conceptually, this work introduces the idea, motivation, and goal of efficient interactive verification of data attribution.

As we will see in the next section, our formalization of this goal is done via direct connection to the interactive PAC-verification framework (Definition 1) of Goldwasser et al. (2021). As we have discussed, a key challenge in verifying a proposed attribution vector a ′ is comparing its predictive quality against the optimal linear predictor Φ( S ) without actually computing Φ( S ) . The PACVerification paradigm turns out to be well-equipped to handle this challenge.

Technical contributions. As for technical contributions, we demonstrate two efficient protocols for the task of efficiently verifying optimality of proposed attributions.

For the first protocol, it turns out that recent work by Saunshi et al. (2022) actually provides a 'residual estimation' algorithm for exactly this task. More specifically, their algorithm estimates the optimal residual (that is, mse ( p ) ( f, ⟨ Φ( S ) , ·⟩ ) ) using only O (1 /ε 3 ) samples of the function f (Theorem 2), without needing to compute Φ( S ) itself. This naturally suggests a potential noninteractive verification protocol where:

- The Prover would compute Empirical Influence attributions a ′ and send them to the Verifier.
- The Verifier could independently estimate the optimal mse ( p ) ( f, ⟨ Φ( S ) , ·⟩ ) (within an additive factor of ε ) using the algorithm of Saunshi et al. (2022), and also independently estimate mse ( p ) ( f, ⟨ a ′ , ·⟩ ) (within an additive factor of ε ) using O (1 /ε 2 ) samples. The Verifier would then accept if the two estimates were ε -close.

The complexity of this non-interactive approach would be dominated by the residual estimation step, resulting in a Verifier cost of O (1 /ε 3 ) (derived from the sample complexity guarantee of Saunshi et al. (2022)). We consider this non-interactive protocol in detail in section 3.

Main Technical Contribution. Our main technical contribution is to build out the above noninteractive protocol into an interactive protocol. Our interaction strategy serves to reduce the Verifier's overall complexity to O (1 /ε 2 ) , and is implemented by a 'spot-checking' mechanism. The function of the 'spot-checking' mechanism is to further move some of the Verifier's work required for residual estimation, onto the prover. This mechanism necessarily requires an interaction, though we only use 2 messages (first the Verifier speaks, and then the Prover responds). To prove the intended qualities of the spot-checking mechanism, we do a non-black box analysis of the proof of the residual estimation algorithm of Saunshi et al. (2022), in order to demonstrate a degree of adversarial robustness in the procedure.

Weoutline our protocol in Section 4. The interactive protocol satisfies the completeness and soundness guarantees that we have discussed. With respect to efficiency, we improve upon the non-interactive protocol by obtaining a Verifier workload that scales as O (1 /ε 2 ) .

## 2 Formal PAC-verification Framework for Attribution and Main Theorem

To this point, we have not formalized what we mean by a Prover-Verifier protocol for data attribution. Let us begin with formalizing the interaction model.

Communication. As part of the interaction, we assume an asynchronous, reliable channel between the parties. An interaction consists of a finite sequence of messages w 1 , w 2 , . . . , w t ∈ { 0 , 1 } ∗ , sent alternately between the Verifier V and the Prover P . Messages are unrestricted bit-strings and may encode, for example, hashes, sketches, model weights, or random seeds.

Shared resources. Both the Prover and the Verifier will agree on certain information regarding the objective of the protocol. For instance, the protocol is executed with respect to a fixed training set S (with | S | = N ), a fixed model architecture, and a fixed objective function for model training which defines the model output function f : {± 1 } N → R , as introduced earlier. 6

Termination. After the last message V halts and outputs a single value ˆ a ∈ { abort }∪ R N , interpreted respectively as rejection or an accepted vector of attribution scores. No further interaction occurs once abort or ˆ a is produced.

## 2.1 The Protocol Guarantees

We are now ready to formalize our desired protocol guarantees. Namely, Completeness, Soundness, and Efficiency. To this end, we adapt a previous formalization of a similar setting called ProbablyApproximately-Correct Verification (PAC-Verification) (Goldwasser et al., 2021). After introducing the formalism, we will also comment on why it is necessary to use this framework, instead of other formalisms from Cryptography, such as Delegation of Computation protocols (see e.g., Goldwasser et al. (2015)).

Let X be the space of training examples. Recall the model output function f : {± 1 } N → R , which maps a representation of a training subset to a model behavior, and our goal to find an attribution vector a such that the linear predictor ⟨ a , ·⟩ has low mse ( p ) ( f, ⟨ a , ·⟩ ) .

Let Φ( S ) ∈ R N denote the optimal attribution vector with respect to MSE.

Our error function err( a , Φ( S )) measures the sub-optimality of a candidate attribution vector a in terms of its predictive MSE performance compared to Φ( S ) :

<!-- formula-not-decoded -->

Note that err( a , Φ( S )) ≥ 0 . The Verifier's goal is to accept a if this error gap is less than or equal to an accuracy threshold ε .

Finally, fix a Verifier cost function κ : (0 , 1) 2 → N .

Definition 1 (PAC-verification for data attribution, adapted from Goldwasser et al. (2021)) . Fix accuracy ε ∈ (0 , 1) and confidence δ ∈ (0 , 1) . An interactive proof system ( V , P ) is an ( ε, δ ) -PAC verifier for Φ if, for every dataset S , the following hold after V and P exchange messages, and then V outputs a value ˆ a ∈ { abort } ∪ R N :

Completeness. If P abides by the protocol, then

̸

<!-- formula-not-decoded -->

Soundness. For every (possibly computationally unbounded) dishonest prover P ′ ,

̸

<!-- formula-not-decoded -->

Efficiency. There exists a constant k &gt; 0 , such that κ ( ε, δ ) &lt; (1 /ε · log(1 /δ )) k .

The Verifier's cost is required to be independent of N = | S | . For our choices of Φ and err (as defined above), the honest Prover's cost can be shown to necessarily grow with N . Hence, our protocols allow the Verifier to expend arbitrarily less cost than the Prover.

Why PAC-Verification and not Delegation of Computation? One might initially consider employing general-purpose cryptographic protocols for delegation of computation or verifiable computation to ensure the prover P performed the expensive attribution computation correctly. However, such approaches are insufficient for the verification goal central to our work and discussed by Goldwasser et al. (2021). First of all, computational overhead for the Prover can be significant in existing cryptographic solutions; this already renders the approach very impractical in our setting, since the honest Prover may already be pushing the limits of their computational power. Second, Cryptographic

6 Later in the paper, we will assume that f : {± 1 } N → [ -b, b ] has a bounded range. This assumption is supported empirically by typical model output functions considered in the literature. For instance: change in correct-class margin at a specific test point (Ilyas et al., 2022).

delegation can typically only ensure that P executed a specific computation as promised , given some inputs. It cannot, in general, provide guarantees about the statistical quality of the output, particularly whether the resulting attribution scores a ′ are indeed ε -close to the optimal scores Φ( S ) according to our error metric err( · , · ) . We refer to section F for a continued discussion.

## 2.2 Main Theorem

The main result of this paper is an ( ε, δ ) -PAC verifier for data attribution, where correctness is measured by the MSE difference defined in err( · , · ) . At this point, we only need to define the Verifier cost function in order to state the theorem.

We will take the Verifier's cost to be the expected number of training runs they conduct (over randomness of the protocol execution). 7 We are now ready to state the theorem.

Theorem 1 (Main Protocol Theorem) . We assume that f : {± 1 } N → [ -b, b ] for some constant b . For any ε, δ ∈ (0 , 1) , Algorithm 1 is a ( ε, δ ) -PAC verifier for Φ (as defined above, with err measuring the MSE gap). The Verifier's cost function satisfies κ ∈ O (log(1 /δ ) /ε 2 ) . Furthermore, the interactive protocol requires only two messages.

## Scalability to Multiple Attribution Tasks.

While our core protocol (Algorithm 1) is presented for verifying attributions with respect to a single model output function f , a natural question arises regarding its applicability when attributions are needed for multiple scenarios simultaneously-for instance, across Z different test points or for Z distinct output metrics. Our framework extends efficiently to such cases. The key observation is that, to maintain an overall ( ε, δ ) -PAC verification guarantee across all Z attribution tasks, a union bound can be applied to the confidence parameters. This means that the number of challenge subsets requested from the Prover for residual estimation, and the number of local samples used by the Verifier for its final MSE checks, would increase logarithmically in Z . We refer to section E for details.

## 3 A 'Simple' Non-Interactive Verification Protocol

This section will use standard notation from Boolean Harmonic Analysis. We refer to section A.1 for the necessary definitions (see instead O'Donnell (2014) for the comprehensive treatment).

Before presenting our main interactive protocol, we first outline a simpler, non-interactive approach to PAC-verifying empirical influence attribution Φ( S ) . This serves to establish a baseline Verifier complexity and motivate the introduction of interaction to achieve greater efficiency.

Recall the verification goal defined in Definition 1. The Verifier V receives a candidate attribution vector a ′ from the Prover P and needs to determine if it is ε -close to Φ( S ) . Using our chosen error metric, this means verifying if:

<!-- formula-not-decoded -->

A close relationship exists between mse ( p ) ( f, ⟨ Φ( S ) , ·⟩ ) and the Boolean Fourier coefficients of f (see section A.1 for the definition and background on Fourier coefficients). In fact, as demonstrated by Saunshi et al. (2022), the optimal linear predictor ⟨ Φ( S ) , ·⟩ satisfies the identity:

<!-- formula-not-decoded -->

Therefore, the verification condition is equivalent to checking if the Prover's solution a ′ satisfies:

<!-- formula-not-decoded -->

The 'residual estimation' algorithm developed by Saunshi et al. (2022) provides a useful tool. As stated in Theorem 2, their algorithm, denoted RESIDUALESTIMATION, allows estimating B ≥ 2 up

7 This is but one way to define cost. For now, we only mention that this notion of sample complexity in data attribution is a commonly used efficiency metric in the field (see e.g, Ilyas et al. (2022); Park et al. (2023)). We emphasize this metric as each retraining constitutes a significant computational effort, potentially scaling with the dataset size N itself. Hence, it captures the primary bottleneck for the Verifier.

to an additive error ε ′ with high probability, using only n = O (1 / ( ε ′ ) 3 · polylog(1 /δ )) evaluations (samples) of the function f . Importantly, this algorithm estimates the residual error without needing to compute the optimal linear predictor Φ( S ) itself.

Theorem 2 (Residual Estimation, restated (Saunshi et al., 2022, Theorem 3.2)) . Let f : {± 1 } N → R . Let ˆ B ≥ 2 be the output of the residual estimation algorithm RESIDUALESTIMATION using degree d = 2 polynomial fitting based on noise stability estimates at points [0 , ρ, 2 ρ ] (along with ρ = 1 ), obtained using a total budget of n calls to f . If n = O (1 /ε 3 ) and ρ = Θ( √ ε ) , then with high probability (e.g., 1 -δ for small constant δ ),

<!-- formula-not-decoded -->

where B ≥ 2 = ∑ S : | S |≥ 2 ˆ f 2 S is the true residual error under the B p distribution.

This immediately suggests a non-interactive PAC-verification protocol:

1. Prover P : Computes a ′ (e.g., its best estimate of Φ( S ) ) and sends it to V .
2. Verifier V (Local Computation):
3. (a) Estimates the MSE of the Prover's solution: Compute ̂ mse ≈ mse ( p ) ( f, ⟨ a ′ , ·⟩ ) by drawing M = O (1 /ε 2 · polylog(1 /δ )) samples x ∼ B p , locally training models to evaluate f ( x ) for each sample, and calculating the empirical average squared error ( f ( x ) -⟨ a ′ , x ⟩ ) 2 .
4. (b) Estimates the optimal residual error: Compute ˆ B ≥ 2 ≈ B ≥ 2 by running RESIDUALESTIMATION ( f, ρ, n ) locally, setting the budget n = O (1 /ε 3 · polylog(1 /δ )) and ρ = Θ( √ ε ) . This requires locally evaluating f on n (pairs of) input subsets.
3. Verifier V (Decision): If ̂ mse ≤ ˆ B ≥ 2 + ε ′′ , output a ′ (where ε ′′ accounts for the combined estimation errors from steps (a) and (b), we can assume ε ′′ ≈ ε ). Otherwise, output abort .

See section B in the appendix for an overview of how the residual estimation algorithm works.

Informal Analysis of the Non-Interactive Protocol. Completeness and Soundness of this protocol can be seen to follow from the accuracy guarantees of standard MSE estimation (via Hoeffding/Chernoff bounds) and the guarantee of the residual estimation algorithm (Theorem 2), using a union bound over the failure probabilities.

The Verifier's efficiency is measured by the number of calls to f (i.e., model trainings). Step (a) requires O (1 /ε 2 · polylog(1 /δ )) calls. Step (b), the residual estimation, requires O (1 /ε 3 · polylog(1 /δ )) calls. The overall Verifier cost κ ( ε, δ ) is therefore dominated by the residual estimation:

<!-- formula-not-decoded -->

While this cost is independent of the dataset size N , the cubic dependence on 1 /ε can still be substantial.

Motivation for Interaction. The O (1 /ε 3 ) complexity arises directly from the need to execute the residual estimation algorithm locally. This motivates exploring an interactive approach. If V could somehow leverage the computational power of P to perform the O (1 /ε 3 ) function evaluations required for residual estimation, while still retaining statistical guarantees against a potentially dishonest P , the Verifier's workload could potentially be reduced.

This is precisely the approach taken in the remainder of this paper. We develop a two-message interactive protocol where V challenges P to perform the function evaluations needed for residual estimation. To ensure P 's responses are trustworthy, V employs a spot-checking mechanism, verifying a small random subset of P 's computations locally. As we will show, this spot-checking strategy, combined with a proof that the residual estimation algorithm is robust to a limited number of undetected errors (Section D.1, Lemma 1), allows V to achieve PAC verification with a reduced cost of O (1 /ε 2 · polylog(1 /δ )) , matching the complexity of the simpler MSE estimation step.

## 4 An Improved Interactive Protocol

In this section, we describe our improved interactive protocol. The formal protocol and its analysis are given in Appendix section C. Our protocol uses as a subroutine the residual estimation algorithm

(Algorithm 1) of Saunshi et al. (2022). As part of the analysis of our improved protocol, we will need to analyze this residual estimation algorithm precisely, using the details of the algorithm's implementation. In Appendix section B, we give a detailed overview of the algorithm.

## · Round 1 ( V → P ): Challenge Setup.

- The Verifier V generates a set of computational 'challenges' for the Prover P . These challenges consist of | E | = O (log(1 /δ ) /ε 3 ) randomly selected subsets of the training data S ∼ B p , using specific random seeds provided by V . Here | E | is the number of samples that will be used for the residual estimation algorithm.
- V secretly designates a randomly chosen subset of k = O (log(1 /δ ) /ε 2 ) of these challenges for later 'spot-checking.'
- V sends the list of training subsets and their corresponding random seeds to P .

## · Round 2 ( P → V ): Honest Prover's Response.

- The Prover P computes the optimal data attribution scores for the full dataset S .
- P performs the training runs requested by V in Round 1, using the specified subsets and random seeds.
- P sends the resulting trained model weights for all challenges, along with the computed attribution scores a , back to V .

## · Round 3 ( V ): Verification.

- Spot-Checking: V repeats training runs for the k challenges designated for spot-checking in Round 1. V checks if the resulting models are equivalent to those sent by P (using an equivalence checking procedure). If any inequivalence occurs, V aborts.
- Consistency Check: If the spot-checks pass, V uses the model results provided by P (for all non-spot-checked challenges) to perform two estimations:
- (a) For O (log(1 /δ ) /ε 2 ) models, evaluate the model output function f , and use them to estimate α = mse ( p ) ( f, ⟨ a , ·⟩ ) .
- (b) Run the residual estimation algorithm (Theorem 2) to estimate β = min w mse ( p ) ( f, ⟨ w , ·⟩ ) .
- V compares α and β . If α and β deviate by more than ε , V aborts.
- Output: If all checks pass, V accepts and outputs the attribution scores provided by P . Otherwise, V outputs abort .

Now that we have a sense of the key steps involved in the improved protocol, we can sketch a proof of its guarantees.

## 4.1 Proof Sketch of Main Theorem

Due to space constraints we defer the full proof to Appendix section D, and simply sketch the main ideas behind each correctness guarantee for the protocol.

Completeness. To argue completeness, we start with the observation that if the Prover is honest, then the 'spot-checks' used in the protocol will pass. In this case, the Verifier is able to accurately estimate both the optimal linear prediction error using the residual estimation algorithm, and the actual error of the Prover's candidate attribution scores using local training runs.

Since the honest Prover submits the optimal scores a ⋆ = Φ( S ) (or ϵ -close to optimal), these two errors are close, causing the final consistency check to pass and the Verifier to accept the correct attribution.

Soundness. To prove soundness, we will consider different cases. First, where a dishonest Prover lies about 'many' of the challenges, and second, where the dishonest Prover lies about 'a few' of the challenges. In case one, we show that the Prover will be caught with high probability by the Verifier's random spot-checks; intuitively if the Prover lies about more than a certain fraction of the requested trainings, the Verifier will catch the Prover after conducting the appropriate amount of spot checks, with high probability. In this case, the Verifier outputs 'abort' with high probability. In case two, the Prover might lie about less than that fraction of the challenges, and we will show that the Verifier's spot checks may all pass, but that despite this, the residual estimation will still be good. At a technical level, in Lemma 1 we show that the residual estimation algorithm of Saunshi et al. (2022) is robust to O (1 /ϵ ) adversarial corruptions. As a result, in this case, even if the Prover has submitted incorrect

candidate attribution scores a ′ , then the Verifier's local estimate of the high error mse ( p ) ( f, ⟨ a ′ , ·⟩ ) will then significantly exceed the estimated optimal error, causing the final consistency check to fail and the Verifier to abort.

Efficiency. To show efficiency, we will directly estimate the expected number of model trainings done by the Verifier, which is straightforward: it is the sum of the number of model trainings in Round 3 , which consists of O (log(1 /δ ) /ϵ 2 ) spot checks, and O (log(1 /δ ) /ϵ 2 ) to estimate the MSE. Thus the overall cost is O (log(1 /δ ) /ϵ 2 ) .

## References

- Bae, J., Ng, N., Lo, A., Ghassemi, M., and Grosse, R. B. (2022). If influence functions are the answer, then what is the question? Advances in Neural Information Processing Systems , 35:17953-17967.
- Broderick, T., Giordano, R., and Meager, R. (2020). An automatic finite-sample robustness metric: when can dropping a little data make a big difference? arXiv preprint arXiv:2011.14999 .
- Castro Fernandez, R. (2023). Data-sharing markets: Model, protocol, and algorithms to incentivize the formation of data-sharing consortia. Proc. ACM Manag. Data , 1(2).
- Choe, S. K., Ahn, H., Bae, J., Zhao, K., Kang, M., Chung, Y., Pratapa, A., Neiswanger, W., Strubell, E., Mitamura, T., Schneider, J., Hovy, E., Grosse, R., and Xing, E. (2024). What is your data worth to gpt? llm-scale data valuation with influence functions.
- Engstrom, L., Feldmann, A., and Madry, A. (2024). Dsdm: Model-aware dataset selection with datamodels. arXiv preprint arXiv:2401.12926 .
- Feldman, V. and Zhang, C. (2020). What neural networks memorize and why: Discovering the long tail via influence estimation. Advances in Neural Information Processing Systems , 33:2881-2891.
- Georgiev, K., Rinberg, R., Park, S. M., Garg, S., Ilyas, A., Madry, A., and Neel, S. (2024). Attributeto-delete: Machine unlearning via datamodel matching.
- Ghorbani, A. and Zou, J. (2019). Data shapley: Equitable valuation of data for machine learning.
- Goldwasser, S., Kalai, Y. T., and Rothblum, G. N. (2015). Delegating computation: interactive proofs for muggles. Journal of the ACM (JACM) , 62(4):1-64.
- Goldwasser, S., Micali, S., and Rackoff, C. (2019). The knowledge complexity of interactive proofsystems. In Providing sound foundations for cryptography: On the work of shafi goldwasser and silvio micali , pages 203-225.
- Goldwasser, S., Rothblum, G. N., Shafer, J., and Yehudayoff, A. (2021). Interactive proofs for verifying machine learning. In 12th Innovations in Theoretical Computer Science Conference (ITCS 2021) , pages 41-1. Schloss Dagstuhl-Leibniz-Zentrum für Informatik.
- Grosse, R., Bae, J., Anil, C., Elhage, N., Tamkin, A., Tajdini, A., Steiner, B., Li, D., Durmus, E., Perez, E., et al. (2023). Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296 .
- Guu, K., Webson, A., Pavlick, E., Dixon, L., Tenney, I., and Bolukbasi, T. (2023). Simfluence: Modeling the influence of individual training examples by simulating training runs.
- Ilyas, A., Park, S. M., Engstrom, L., Leclerc, G., and Madry, A. (2022). Datamodels: Predicting predictions from training data. arXiv preprint arXiv:2202.00622 .
- Jain, S., Hamidieh, K., Georgiev, K., Ilyas, A., Ghassemi, M., and Madry, A. (2024). Data debiasing with datamodels (d3m): Improving subgroup robustness via data selection.
- Jia, R., Dao, D., Wang, B., Hubis, F. A., Hynes, N., Gurel, N. M., Li, B., Zhang, C., Song, D., and Spanos, C. (2023). Towards efficient data valuation based on the shapley value.
- Koh, P. W. and Liang, P. (2017). Understanding black-box predictions via influence functions. In International conference on machine learning , pages 1885-1894. PMLR.

Law, J. (1986). Robust statistics-the approach based on influence functions.

O'Donnell, R. (2014). Analysis of boolean functions . Cambridge University Press.

- Park, S. M., Georgiev, K., Ilyas, A., Leclerc, G., and Madry, A. (2023). Trak: Attributing model behavior at scale. arXiv preprint arXiv:2303.14186 .
- Pawelczyk, M., Leemann, T., Biega, A., and Kasneci, G. (2023). On the trade-off between actionable explanations and the right to be forgotten. In International Conference on Learning Representations (ICLR) .
- Saunshi, N., Gupta, A., Braverman, M., and Arora, S. (2022). Understanding influence functions and datamodels via harmonic analysis. arXiv preprint arXiv:2210.01072 .
- Shah, H., Ilyas, A., and Madry, A. (2024). Decomposing and editing predictions by modeling model computation. arXiv preprint arXiv:2404.11534 .
- Vadhan, S. P. (1999). A study of statistical zero-knowledge proofs . PhD thesis, Massachusetts Institute of Technology.
- Yeh, C.-K., Kim, J. S., Yen, I. E. H., and Ravikumar, P. (2018). Representer point selection for explaining deep neural networks.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction claim the development of an interactive PACverification protocol for data attribution that is efficient for the Verifier, and provides formal completeness and soundness guarantees. These claims are directly addressed by the main theoretical contributions presented in Theorem 1 and Algorithm 1, with proofs and detailed explanations provided in the subsequent sections and appendices. The scope, while focusing on linear functions over the boolean hypercube and its applicability to data/component attribution, is also consistently maintained, though it is a very broad scope in the context of data attribution as almost all methods fall in this category.

Guidelines: [NA]

<!-- image -->

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses limitations. For instance, the (valid) assumption of a bounded model output function f (Line 229 footnote, Line 277) is stated. The practicalities and assumptions of the equivalence checking procedure are also briefly mentioned. The computational burden on the (assumed powerful) Prover for the interactive steps is also an implicit consideration.

Guidelines: [NA]

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Complete proofs of all central claims are included in the appendix, in addition to all proof sketches in body.

Guidelines: [NA]

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: This is a theory paper. Provable security or verifiability inherently requires theory, given that we provide guarantees against all possible adversaries. Future work on implementation may involve heuristics based on this work.

Guidelines: [NA]

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: There is no data or code that goes with this paper.

Guidelines: [NA]

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: No models were trained or tested as part of this paper.

Guidelines: [NA]

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: No statistical experiments are conducted as this is a theory paper consisting of mathematical proofs.

Guidelines: [NA]

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: There are no experiments since it is primarily theory paper.

Guidelines: [NA]

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research presents a theoretical framework for verifying data attributions. It does not raise immediate concerns under the NeurIPS Code of Ethics. The aim of the work is to improve trust and transparency in ML systems, which aligns with ethical considerations.

Guidelines: [NA]

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses as motivation issues on trust and safety of ML systems.

Guidelines: [NA]

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper focuses on a verification protocol for data attributions, not on the release of new data or models that might have a risk of misuse.

Guidelines: [NA]

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Relevant work is appropriately cited throughout the paper.

Guidelines: [NA]

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: No assets.

Guidelines: [NA]

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No human subjects were used as part of this work.

Guidelines: [NA]

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: There are no study participants.

Guidelines: [NA]

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The paper does not use LLMs as part of core methodology.

Guidelines: [NA]

## A Preliminaries

## A.1 Harmonic Analysis on the p-biased Cube

Our work (and important prior work like Goldwasser et al. (2021) and Saunshi et al. (2022)) leverage tools that are grounded in the analysis of real-valued functions defined on the discrete hypercube, specifically under a non-uniform measure known as the p -biased distribution (see e.g., O'Donnell (2014) for a comprehensive overview of Boolean Fourier analysis). In this section, we will go over the necessary concepts for understanding the mathematics presented in this paper.

Let N = | S | be the size of the training dataset. Recall that we represent a subset of the training data by a vector x ∈ {± 1 } N , where x i = 1 indicates the i -th datapoint is included and x i = -1 indicates it is excluded. The model's behavior (e.g., loss or logit difference on a test point) after training on the subset represented by x is given by a function f : {± 1 } N → R .

We consider the p -biased distribution B p over {± 1 } N , where each coordinate x i is independently chosen to be 1 with probability p and -1 with probability 1 -p . Let µ = E [ x i ] = p -(1 -p ) = 2 p -1 and σ 2 = Var( x i ) = E [ x 2 i ] -( E [ x i ]) 2 = 1 -(2 p -1) 2 = 4 p (1 -p ) . The uniform distribution corresponds to p = 1 / 2 , where µ = 0 and σ 2 = 1 .

The space of functions f : {± 1 } N → R forms a vector space equipped with the inner product ⟨ f, g ⟩ B p = E x ∼B p [ f ( x ) g ( x )] . An orthonormal basis for this space is given by the characters { ϕ S } S ⊆ [ N ] , defined as:

<!-- formula-not-decoded -->

̸

These basis functions satisfy E [ ϕ S ( x )] = 0 for S = ∅ and ⟨ ϕ S , ϕ T ⟩ B p = δ S,T (Kronecker delta). Any function f can be uniquely expanded in this basis as:

<!-- formula-not-decoded -->

where ˆ f S = ⟨ f, ϕ S ⟩ B p = E x ∼B p [ f ( x ) ϕ S ( x )] are the Fourier coefficients of f . Parseval's identity states:

<!-- formula-not-decoded -->

The coefficients ˆ f S capture the contribution of interactions among the datapoints in S to the function f . Of particular importance are the degree-1 coefficients ˆ f { i } , which, up to scaling by 2 /σ , correspond to the average influence of datapoint i and the optimal coefficients for linear datamodels (Saunshi et al., 2022, Theorem 2.2). To denote the total Fourier weight at degree k , let

<!-- formula-not-decoded -->

To analyze the structure of f , we use the concept of noise stability.

Definition 2 ( ρ -correlated variables) . For x ∈ {± 1 } N drawn from B p , we say a random variable x ′ is ρ -correlated to x if it is sampled coordinate-wise independently as follows: For each i ∈ [ N ] ,

- If x i = 1 , then x ′ i = -1 with probability (1 -p )(1 -ρ ) , and x ′ i = 1 otherwise.
- If x i = -1 , then x ′ i = 1 with probability p (1 -ρ ) , and x ′ i = -1 otherwise.

Crucially, x ′ is also distributed according to B p .

The noise stability of f at noise rate ρ ∈ [0 , 1] is defined as h f ( ρ ) = E x,x ′ [ f ( x ) f ( x ′ )] , where x ∼ B p and x ′ is ρ -correlated to x . It admits a simple expression in terms of the Fourier weights:

<!-- formula-not-decoded -->

This shows that the noise stability is a polynomial in ρ , whose coefficients are precisely the total Fourier weights B k at each degree k .

## B Efficient Estimation of Linear Datamodel Residual

As established by Saunshi et al. (2022) (specifically their Theorem 2.2), the quality of the best linear datamodel approximation θ 0 + ∑ N i =1 θ i x i for a function f : {± 1 } N → R under the p -biased distribution B p is determined by its residual error:

<!-- formula-not-decoded -->

where θ ⋆ denotes the coefficients that minimize MSE of the datamodel θ 0 + ∑ N i =1 θ i x i with respect to the model output function f (Saunshi et al. (2022) use the notation R ( θ ⋆ ) , so we include that here for clarity, even though throughout the paper we refer to MSE explicitly).

Saunshi et al. (2022) cleverly leverage the connection between degreek weight of f and the noise stability of f , h f ( ρ ) . Recall from (4) that

<!-- formula-not-decoded -->

Separately, the total squared norm is B = ∥ f ∥ 2 B p = ∑ N k =0 B k = h f (1) . Therefore, the residual error can be expressed purely in terms of the degree-0 and degree-1 Fourier weights and the total norm:

<!-- formula-not-decoded -->

The algorithm from Saunshi et al. (2022) leverages this fact as follows:

1. Estimate Noise Stability at Key Points: Choose a small number k of distinct noise levels { ρ 1 , . . . , ρ k } (e.g., 0 , ρ, 2 ρ for some small ρ ) and additionally use ρ = 1 . For each chosen ρ j (and for ρ = 1 ), estimate the noise stability value h f ( ρ j ) by sampling multiple pairs ( x, x ′ ) of ρ j -correlated inputs drawn from B p and averaging the product f ( x ) f ( x ′ ) . Let these estimates be ˆ y j ≈ h f ( ρ j ) and ˆ B ≈ h f (1) .
2. Low-Degree Polynomial Fitting: Since h f ( ρ ) = B 0 + B 1 ρ + B 2 ρ 2 + . . . , use the estimated values ( ρ j , ˆ y j ) from the previous step to fit a low-degree polynomial (e.g., degree d = 2 ) P ( ρ ) = ∑ d i =0 ˆ B i ρ i . This is done via (non-negative) least squares, solving for the coefficients ˆ B 0 , ˆ B 1 , . . . , ˆ B d .
3. Calculate Residual Estimate: Combine the estimated total norm ˆ B ≈ h f (1) and the estimated low-degree coefficients ˆ B 0 , ˆ B 1 obtained from the polynomial fit to compute the final estimate of the residual error: ˆ B ≥ 2 = ˆ B -ˆ B 0 -ˆ B 1 .

This approach allows estimating the quality of the best linear fit using a number of samples that is proportional to 1 /ε 3 for a desired approximation error ε but is independent of the dimension N .

## C Our Full Protocol

We present our protocol for PAC-verification of empirical influence (Theorem 1), detailed in Algorithm 1.

## Algorithm 1 Interactive PAC-Verification Protocol for Empirical Influence ( Φ )

## Shared Information:

Model Architecture, Training set S ⊆ X ( | S | = N ), subsampling probability p ,

Training loss function L , Stopping criteria for training.

Model output function f : {± 1 } N → R derived from training.

Equivalence check procedure CHECKEQUIV ( θ 1 , θ 2 ) .

## Verifier ( V ) Input:

Approximation parameter ε ∈ (0 , 1) , Confidence parameter δ ∈ (0 , 1) .

## Round 1: Verifier ( V ) → Prover ( P )

<!-- formula-not-decoded -->

- 2: Choose ρ ← Θ( √ ε ) . Sample n = O (1 /ε 3 · log(1 /δ )) pairs { ( x ( e ) , x ′ ( e ) ) } n e =1 where x ( e ) ∼ B p and x ′ ( e ) is ρ -correlated to x ( e ) . Let E be the set of all unique subsets appearing in these n pairs (at most 2 n subsets).

⋆

- 3: Choose random seeds R ←{ r e } x e ∈ E for training models on subsets in E .
- 4: Set spot-check set size k ← O (1 /ε 2 · log(1 /δ )) . Sample spot-check set C ⊆ E of size k uniformly at random. ⋆ k will ensure high prob. detection
- 5: Send ( E,R ) to P .

## Round 2: Prover ( P ) → Verifier ( V )

- 6: Compute attribution a ⋆ ← Φ( S ) .

⋆ Seeds R are indexed corresponding to subsets in E

⋆ Computationally expensive

- 7: For each x e ∈ E do
- 8: Train model with seed r e on subset x e to get weights θ e .
- 9: End For
- 10: Let θ = { θ e } x e ∈ E .
- 11: Send ( θ , a ⋆ ) to V .

V

Round 3: Verifier (

- 12: Spot-check:
- 13: For each x c ∈ C do
- 14: Train model with seed r c on subset x c to get local weights θ c . ⋆ V performs limited training

⋆ Sends all trained models and the computed attribution

- ˆ
- 15: Let θ ′ c be the model weights received from a (possibly dishonest) Prover P ′ for subset x c .
- ˆ ′
- 16: If not CHECKEQUIV( θ c , θ c )) then
- 17: Output: abort and Terminate .
- 18: End If
- 19: End For

20:

⋆ Check if Prover's result matches Verifier's

⋆ All spot-checks passed

- 21: Use received models θ ′ to define the function values f ′ ( x e ) for x e ∈ E .
- 22: Compute residual estimate ˆ B ≥ 2 ← RESIDUALESTIMATION ( f ′ ) on E,ρ, budget | E | .

23:

- 24: For each x m ∈ M do
- 25: Train model with fresh random seed r ′ m on subset x m to get weights θ m .
- 26: Evaluate f ( x m ) using θ m . ⋆ V trains models for MSE estimate
- 27: End For
- 28: Let g ( x ) = ⟨ a ⋆ , x ⟩ . Estimate ̂ mse p ( f, g ) ← 1 | M | ∑ x m ∈ M ( f ( x m ) -g ( x m )) 2 .
- 29: If ̂ mse p ( f, g ) &gt; ˆ B ≥ 2 + ε/ 2 then
- 30: Output: abort and Terminate .
- 31: Else
- 32: Output: a ⋆
- 33: End If

⋆ Uses function values derived from P 's models (for x e / ∈ C )

Subsets for residual estimation

) Verification and Output

⋆ Accept the Prover's attribution

The Equivalence Checking Procedure. Note that we assume access to an equivalence-checking procedure between model training runs. The exact implementation of CHECKEQUIV depends on the training setup. Ideally, even with fully deterministic training (with a fixed architecture, data subset x c , and randomness seed r c ), this check could be a direct comparison of model weights ( θ c = θ ′ c ). However, practical ML training can exhibit non-determinism (e.g., due to GPU parallelism). In such cases, CHECKEQUIV might involve comparing model outputs on a held-out test set, comparing final loss values within a tolerance, or potentially using cryptographic commitments or hashes if sufficient determinism can be enforced.

## D Proof of Main Theorem

In this section, we will prove that the protocol outlined in Algorithm 1 works to witness Theorem 1. For now, let us start with an outline of our proof.

Completeness. To argue completeness, we start with the observation that if the Prover is honest, then the 'spot-checks' used in the protocol will pass. In this case, the Verifier is able to accurately estimate both the optimal linear prediction error ( B ≥ 2 ) using the residual estimation algorithm, and the actual error of the Prover's candidate attribution scores using local training runs.

Since the honest Prover submits the optimal, or near optimal, scores a ⋆ = Φ( S ) , these two errors are close, causing the final consistency check to pass and the Verifier to accept the correct attribution.

Soundness. To prove soundness, we will consider different cases. First, where a dishonest Prover lies about 'many' of the requested model trainings ( θ ′ ), and second, where the dishonest Prover lies on 'a few' of the requested model trainings.

In case one, we show that the Prover will be caught with high probability by the Verifier's random spot-checks. In this case, the Verifier outputs 'abort' with high probability.

In case two, we will show that the Verifier's spot checks may all pass, but that despite this, the residual estimation will still be good. That is, we show that it is robust to some adversarial corruptions. We prove a standalone lemma to accomplish this.

As a result, in this case, even if the Prover has submitted incorrect candidate attribution scores a ′ , then the Verifier's local estimate of the high error mse ( p ) ( f, ⟨ a ′ , ·⟩ ) will then significantly exceed the estimated optimal error ˆ B ≥ 2 , causing the final consistency check to fail and the Verifier to abort.

Efficiency. To show efficiency, we will directly estimate the expected number of model trainings done by the Verifier.

## D.1 Supporting Lemma: Robust Residual Estimation

Before proving the main theorem, we establish the robustness of the RESIDUALESTIMATION subroutine against a limited number of adversarial corruptions.

Lemma 1 (Robust Residual Estimation) . Let f : {± 1 } N → [ -b, b ] be bounded. Let ˆ B ≥ 2 be the output of RESIDUALESTIMATION ( f, ρ, n ) using degree d = 2 , points [0 , ρ, 2 ρ ] (and ρ = 1 ), and based on n evaluations of f run on the same sample points, but where an adversary corrupts up to m = O (1 /ε ) of the n function evaluations, by replacing f ( x ) with f ′ ( x ) ∈ [ -b, b ] . Let n = O (1 /ε 3 log(1 /δ )) , and ρ = Θ( √ ε ) .

Then, with probability at least 1 -δ/ 4 (over the internal randomness of the algorithm):

<!-- formula-not-decoded -->

where the constant in ˜ O ( · ) depends on δ but not on n, m, ε, N .

Proof of Lemma 1. Let us first recap the RESIDUALESTIMATION algorithm.

RESIDUALESTIMATION uses n total function evaluations, distributed among estimating h f (0) , h f ( ρ ) , h f (2 ρ ) and h f (1) . Let n 0 , n ρ , n 2 ρ be the number of pairs ( x, x ′ ) sampled to estimate h f (0) , h f ( ρ ) , h f (2 ρ ) respectively, and let N 1 be the number of samples x used to estimate h f (1) = E [ f ( x ) 2 ] . The total number of function evaluations is n = ( n 0 + n ρ + n 2 ρ ) × 2 + N 1 .

Let ˜ y 0 , ˜ y ρ , ˜ y 2 ρ be the empirical estimates of h f (0) , h f ( ρ ) , h f (2 ρ ) obtained using the original function f evaluations from the n samples. Let ˜ B be the estimate of h f (1) = E [ f ( x ) 2 ] . Let ˜ y = [˜ y 0 , ˜ y ρ , ˜ y 2 ρ ] T . The algorithm RESIDUALESTIMATION uses these estimates, which are not corrupted by any adversary, to compute the uncorrupted residual estimate ˜ B ≥ 2 . Specifically, a vector ˜ z = [ ˜ B 0 , ˜ B 1 , ˜ B 2 ] T is found by solving the constrained least squares problem

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then ˜ B ≥ 2 = ˜ B -˜ z 1 -˜ z 2 ( ˜ z 1 would correspond to B 0 , ˜ z 2 would correspond to B 1 ).

Now, we consider the setting where an adversary corrupts m = O (1 /ε ) of the n function evaluations. Let f ′ ( x ) denote the value used in the computation after potential corruption. Since f : {± 1 } N → [ -b, b ] , we also have f ′ ( x ) ∈ [ -b, b ] . Let ˆ y 0 , ˆ y ρ , ˆ y 2 ρ be the estimates using the potentially corrupted values f ′ , and let ˆ B be the estimate of E [ f ′ ( x ) 2 ] . Let ˆ y = [ˆ y 0 , ˆ y ρ , ˆ y 2 ρ ] T . Let:

<!-- formula-not-decoded -->

Here,

<!-- formula-not-decoded -->

Our goal is to bound | ˆ B ≥ 2 -B ≥ 2 | , where B ≥ 2 is the 'true value' (assume exact estimation). We use the triangle inequality:

<!-- formula-not-decoded -->

Here z 1 , z 2 are the 'true coefficients' of the noise stability polynomial (recall equation 4).

To do this, we will apply the triangle inequality again, and bound each of the terms on the RHS above, by | ˆ B -B | ≤ | ˆ B -˜ B | + | ˜ B -B | , and similarly, | ˆ z i -z i | ≤ | ˆ z i -˜ z i | + | ˜ z i -z i | .

First, bound the difference in the estimates ˆ y i -˜ y i and ˆ B -˜ B . Consider ˆ y ρ -˜ y ρ . This is the difference between the average of f ′ ( x ) f ′ ( x ′ ) and f ( x ) f ( x ′ ) over n ρ = Θ( n ) pairs. An adversary corrupts m function evaluations total. Each term f ( x ) f ( x ′ ) uses two evaluations. At most m terms in the average can be affected by corruption. Let S ρ ⊆ [ n ρ ] be the indices of the affected terms. | S ρ | ≤ m .

<!-- formula-not-decoded -->

Since | f ( x ) f ( x ′ ) | ≤ b 2 and | f ′ ( x ) f ′ ( x ′ ) | ≤ b 2 , the difference in each term is bounded by 2 b 2 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then the maximum element-wise error is bounded:

<!-- formula-not-decoded -->

Similarly, and

Also,

Since n = O (1 /ε 3 log(1 /δ )) and m = O (1 /ε ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, we can bound the difference between the uncorrupted estimates ˜ y 0 , ˜ y ρ , ˜ y 2 ρ and the 'true values' h f (0) , h f ( ρ ) , h f (2 ρ ) using standard concentration inequalities. For instance, see Lemma A.1 of Saunshi et al. (2022), which shows that the estimations are close enough to the true values with probability 1 -δ , i.e. ˜ y ρ = ˜ h ( ρ ) = h ( ρ ) + γ ρ where | γ ρ | ≤ γ where γ = O ( √ log(1 /δ ) /n ) .

At this point, we can combine the adversarial error bound and estimation error bounds for the noise sensitivity estimates, and complete the analysis as do Saunshi et al. (2022) in their analysis of RESIDUALESTIMATION.

From above, we have, with high probability,

<!-- formula-not-decoded -->

Then, applying the analysis from Saunshi et al. (2022) (Theorem 3.2), of how the perturbed estimates affect the constrained least squares solutions, we get

<!-- formula-not-decoded -->

Hence,

Finally, for the setting of ρ = ε , and observing that B ≥ 3 ≤ b 2 we obtain the desired expression.

<!-- formula-not-decoded -->

## D.2 Proof of Main Theorem

Proof of Theorem 1. We prove Theorem 1 by establishing Completeness, Soundness, and Efficiency as per Definition 1. Recall that we assume | f ( x ) | ≤ b = O (1) . We set internal protocol confidence parameters for subroutines to δ ′ = δ/ 4 .

## D.3 Completeness

For completeness we can assume both V and P follow the protocol.

1. Prover's Output: Honest P computes a ⋆ = Φ( S ) and correct models θ . We have err( a ⋆ , Φ( S )) = 0 ≤ ε . P sends ( θ , a ⋆ ) .
2. Verifier's Spot-Check: Passes with probability 1, assuming deterministic training given fixed random seeds.

## 3. Verifier's Consistency Check:

- (a) Residual Estimation: V uses correct θ for f values. By Theorem 2, using | E | = O (1 /ε 3 log(1 /δ ′ )) , | ˆ B ≥ 2 -B ≥ 2 | &lt; ε/ 4 with probability at least 1 -δ ′ .
- (b) MSE Estimation: V uses local trainings for M . With | M | = O (1 /ε 2 log(1 /δ ′ )) , by Hoeffding's inequality, | ̂ mse p ( f, ⟨ a ⋆ , ·⟩ ) -B ≥ 2 | &lt; ε/ 4 with probability at least 1 -δ ′ .
- (c) Final Check: Compare ̂ mse p ( f, ⟨ a ⋆ , ·⟩ ) with ˆ B ≥ 2 + ε/ 2 . With probability at least 1 -2 δ ′ , both estimates are within ε/ 4 of B ≥ 2 . Then ̂ mse p ≤ B ≥ 2 + ε/ 4 and ˆ B ≥ 2 + ε/ 2 ≥ ( B ≥ 2 -ε/ 4) + ε/ 2 = B ≥ 2 + ε/ 4 . So, ̂ mse p ≤ ˆ B ≥ 2 + ε/ 2 . Therefore the final check passes.

<!-- formula-not-decoded -->

4. Verifier's Output: V outputs a ⋆ with err( a ⋆ , Φ( S )) = 0 ≤ ε .

The overall success probability is at least 1 -2 δ ′ = 1 -δ/ 2 ≥ 1 -δ . Therefore completeness is proved.

## D.4 Soundness

To analyze soundness, we will proceed by case analysis. The first case deals with a scenario where the Prover is very dishonest, in the sense that many of the requested model trainings from the set E (see Round 1 of the protocol) are wrong.

The second case will be mutually exclusive, and deal with the scenario where the Prover is mildly dishonest, in the sense that not too many of the requested model trainings are wrong. In this second case, the residual estimation robustness lemma (lemma 1) will be useful.

To begin, recall that the dishonest Prover P ′ is sending ( θ ′ , a ′ ) . Let W be the set of corrupted challenges where θ ′ e is not equivalent to a correctly trained model, and let the number of corruptions be m = | W | . Let | E | be the total number of challenges sent to the Prover for residual estimation. We define the threshold for "many" corruptions by setting a critical point m ⋆ = c/ε for a sufficiently large constant c . The number of spot-checks is k = O (1 /ε 2 · log(1 /δ )) .

̸

Case 1: m&gt;m ⋆ . In this case, we show that a spot-check will fail with high probability. The Verifier samples a set C of k challenges to check. The Prover is caught if C ∩ W = ∅ . The probability that the Prover is not caught is the probability that all k checks land outside of the corrupted set W . This probability can be upper-bounded by sampling with replacement:

<!-- formula-not-decoded -->

Since m&gt;m ⋆ = c/ε , we have:

<!-- formula-not-decoded -->

Substituting k = C ′ /ε 2 · log(4 /δ ) and | E | = C E /ε 3 · log(1 /δ ) for constants C ′ , C E , we get the exponent:

<!-- formula-not-decoded -->

By choosing the constant c in the definition of m ⋆ to be large enough, this exponent can be made smaller than -ln(4 /δ ) , ensuring the failure probability is less than δ/ 4 . Thus, we conclude V aborts with probability at least 1 -δ/ 4 .

Case 2: m ≤ m ⋆ . In this second case, all spot-checks may pass with some probability. Let us (conservatively) assume that all checks indeed pass, and then consider how this affects the Verifier's output.

We analyze the protocol, step by step, in this case.

- (1) Robust Residual Estimation: V runs RESIDUALESTIMATION using the function values derived from the models θ ′ sent by the Prover. Since m ≤ m ⋆ = O (1 /ε ) , the number of corruptions is within the required bound for Lemma 1. Therefore, the lemma applies directly. We have | ˆ B ≥ 2 -B ≥ 2 | &lt; ε/ 4 with probability at least 1 -δ ′ .
- (2) MSE Estimation: V estimates ̂ mse p ( f, g ′ ) for g ′ = ⟨ a ′ , ·⟩ . This happens with local samples in the set M . Therefore, we can conclude that | ̂ mse p ( f, g ′ ) -mse ( p ) ( f, g ′ ) | &lt; ε/ 4 with probability at least 1 -δ ′ , by applying standard Chernoff bounds.
- (3) Final Check Condition: Assume err( a ′ , Φ( S )) &gt; ε , which means mse ( p ) ( f, g ′ ) &gt; B ≥ 2 + ε . with probability at least 1 -2 δ ′ , we have ̂ mse p ( f, g ′ ) &gt; mse ( p ) ( f, g ′ ) -ε/ 4 &gt; ( B ≥ 2 + ε ) -ε/ 4 = B ≥ 2 +3 ε/ 4 . Also, ˆ B ≥ 2 &lt; B ≥ 2 + ε/ 4 . Then ˆ B ≥ 2 + ε/ 2 &lt; ( B ≥ 2 + ε/ 4) + ε/ 2 = B ≥ 2 +3 ε/ 4 . Since ̂ mse p ( f, g ′ ) &gt; B ≥ 2 +3 ε/ 4 and B ≥ 2 +3 ε/ 4 &gt; ˆ B ≥ 2 + ε/ 2 , the condition ̂ mse p ( f, g ′ ) &gt; ˆ B ≥ 2 + ε/ 2 holds.

- (4) Verifier Output: If err( a ′ , Φ( S )) &gt; ε and m ≤ m ⋆ , V outputs abort with probability at least 1 -2 δ ′ .

Combining Cases: If err( a ′ , Φ( S )) &gt; ε , the probability V does not abort is at most Pr[ Case 1 fails ] + Pr[ Case 2 fails ] ≤ δ/ 4 + 2 δ ′ = δ/ 4 + δ/ 2 = 3 δ/ 4 ≤ δ . Therefore Soundness holds.

## D.5 Efficiency

Finally, we consider efficiency. The Verifier's cost κ ( ε, δ ) is the number of model trainings.

1. Spot-Checking: The number of spot-checks is deterministic.

<!-- formula-not-decoded -->

2. MSE Estimation: | M | = O (1 /ε 2 · log(1 /δ )) .

Thus, the total cost is O (1 /ε 2 · log(1 /δ )) . Treating δ as constant yields κ ( ε, O (1)) ∈ O (1 /ε 2 ) .

## E Verifying Attributions for Multiple Test Points or Outputs

Our primary protocol (Algorithm 1) and its analysis (Theorem 1) are presented for verifying attribution with respect to a single model output function f . However, a common and practical requirement is to obtain attributions for a model's behavior across multiple distinct scenarios, such as its predictions on Z different test points, or changes in Z different output logits. This gives rise to a set of Z model output functions, { f z } Z z =1 , where each f z : {± 1 } N → [ -b, b ] . For each f z , there is a corresponding empirical influence attribution Φ( S, z ) and an optimal linear prediction residual B ≥ 2 ,z .

The goal is to extend our verification framework such that the Verifier V can, with overall confidence 1 -δ , simultaneously accept Z attribution vectors { a ′ z } Z z =1 from the Prover P if and only if err( a ′ z , Φ( S, z )) ≤ ε for all z ∈ [ Z ] . This section details how our protocol can be adapted to achieve this, maintaining efficiency and the two-message structure.

The core strategy involves adjusting the internal confidence parameters of the statistical estimation subroutines using a union bound. If the overall desired failure probability for the entire set of Z verifications is δ , then the failure probability allocated to any critical estimation step concerning an individual function f z must be reduced to δ ′ ≈ δ/Z . This tightening of per-instance confidence directly impacts the sample complexities, but only logarithmically in Z .

We formalize this extension with the following theorem:

Theorem 3 (PAC-Verification for Multiple Output Functions) . Assume that for each z ∈ [ Z ] , the model output function f z : {± 1 } N → [ -b, b ] for some constant b . For any ε ∈ (0 , 1) , and Z ≥ 1 , there exists an ( ε, δ ) -PAC verifier for the set of Z empirical influence operators { Φ( S, z ) } Z z =1 . The Verifier's expected cost function κ Z (number of local model trainings) satisfies κ Z ∈ O ((1 /ε 2 ) · polylog( Z ) · polylog(1 /δ )) . The interactive protocol requires only two messages.

Proof Sketch for Theorem 3. The proof adapts the logic of Theorem 1 by incorporating the union bound across the Z output functions. Let δ 0 = δ/ (4 Z ) be the target failure probability for individual statistical estimation steps (like residual estimation or MSE estimation for a single f z ) and for the spot-checking mechanism's success in detecting a certain level of fraud for any f z . The result follows immediately from the logarithmic dependence of the sample complexities on 1 δ in each step.

## F Why PAC-Verification and not Delegation of Computation?

As first mentioned in section 2, one might initially consider employing general-purpose cryptographic protocols for delegation of computation or verifiable computation to ensure the Prover P performed the expensive attribution computation correctly. However, such approaches are insufficient for the

verification goal central to our work. First of all, computational overhead of the cryptographic machinery is typically massive, and in our setting, the Prover has already pushed the limits of their feasible computations. Second, cryptographic protocols typically do not give guarantees for the statistical quality of the output. In particular, whether the resulting attribution scores a ′ are indeed ε -close to the optimal scores Φ( S ) .

For instance, if the Prover uses skewed, unrepresentative, or otherwise low-quality subsets for its internal estimation process (whether maliciously or inadvertently), a general-purpose delegation protocol might still verify the computation's execution correctly based on those flawed inputs, offering no protection against a poor-quality final result in terms of predictive accuracy. Note, the Verifier cannot supply the inputs due to computational constraints. A practical example of when the use of flawed inputs might occur is if the Prover itself obtains trained models from a public ledger or otherwise untrusted source.

Thus, when the Prover's computational steps are correct , standard cryptographic verification procedures would not necessarily identify bad attributions. In other words, it does not inherently assess whether the output meets a statistical benchmark defined relative to the underlying data generating process or the optimal achievable performance. The PAC-verification framework of Goldwasser et al. (2021) is designed precisely to handle verification of approximate correctness of the outcome relative to a statistical ideal (here, minimizing MSE), rather than merely the integrity of the computational steps taken.
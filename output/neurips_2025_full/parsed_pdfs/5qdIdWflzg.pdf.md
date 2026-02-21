## Efficient PAC Learning for Realizable-Statistic Models via Convex Surrogates

## Shivani Agarwal

University of Pennsylvania ashivani@seas.upenn.edu

## Abstract

A central question in the theory of machine learning concerns the identification of classes of data distributions for which one can provide computationally efficient learning algorithms with provable statistical learning guarantees. Indeed, in the context of probably approximately correct (PAC) learning, there has been much interest in exploring intermediate PAC learning models that, unlike the realizable PAC learning setting, allow for some stochasticity in the labels, and unlike the fully agnostic PAC learning setting, also admit computationally efficient learning algorithms with finite sample complexity bounds. Some examples of such models include random classification noise (RCN), probabilistic concepts, Massart noise, and generalized linear models (GLMs); in general, most of this work has focused on binary classification problems. In this paper, we study what we call realizablestatistic models (RSMs), wherein we allow stochastic labels but assume that some vector-valued statistic of the conditional label distribution comes from some known function class. RSMs are a flexible class of models that interpolate between the realizable and fully agnostic settings, and that also recover several previously studied models as special cases. We show that for a broad range of RSM learning problems, where the statistic of interest can be accurately estimated via a convex 'strongly proper composite' surrogate loss, minimizing this convex surrogate loss yields a computationally efficient learning algorithm with finite sample complexity bounds. We then apply this result to show that various commonly used (and in some cases, not so commonly used) convex surrogate risk minimization algorithms yield computationally efficient learning algorithms with finite sample complexity bounds for a variety of RSM learning problems including binary classification, multiclass classification, multi-label prediction, and subset ranking. For the special case of binary classification with sigmoid-of-linear class probabilities (also a special case of GLMs), our results show that minimizing the standard binary logistic loss has a similar sample complexity as the GLM-tron algorithm of Kakade et al. (2011), but is computationally more efficient. In terms of the distribution over the domain/instance space, our results are all distribution-independent. To our knowledge, these are the first such results for PAC learning with stochastic labels for such a broad range of learning problems.

## 1 Introduction

The probably approximately correct (PAC) learning model is a cornerstone in the theory of machine learning. The two most widely studied settings, namely the realizable and fully agnostic settings, both represent somewhat extreme tradeoffs between computational efficiency and statistical modeling power: The realizable setting, as originally proposed by Valiant [38], often admits computationally efficient learning algorithms, but makes the restrictive statistical assumption that examples are labeled by a deterministic target function (from some known function class); the (fully) agnostic setting [23, 29] allows for fully general joint probability distributions on the labeled examples, but often fails to admit computationally efficient learning algorithms. Consequently, there has been much interest

in exploring intermediate PAC learning models that both allow for some stochasticity in the labels, and admit computationally efficient learning algorithms with finite sample complexity bounds. Some examples of such models include random classification noise (RCN) [4, 13, 10, 17, 27, 21, 22, 30, 20], probabilistic concepts [28], Massart noise [36, 37, 35, 31, 6, 7, 8, 40, 45, 19, 15, 14], and (univariate) generalized linear models (GLMs) and single index models (SIMs) [26, 25]. In general, most of this work has focused on binary classification problems.

In this paper, we study what we call realizable-statistic models (RSMs), wherein we allow stochastic labels but assume that some vector-valued statistic of the conditional label distribution comes from some known function class. RSMs are a flexible class of models that interpolate between the realizable and fully agnostic settings, and that also recover several previously studied models as special cases. We show that for a broad range of RSM learning problems, where the statistic of interest can be accurately estimated via a convex 'strongly proper composite' surrogate loss, minimizing this convex surrogate loss yields a computationally efficient learning algorithm with finite sample complexity bounds. We then apply this result to show that various commonly used (and in some cases, not so commonly used) convex surrogate risk minimization algorithms yield computationally efficient learning algorithms with finite sample complexity bounds for a variety of RSM learning problems including binary classification, multiclass classification, multi-label prediction, and subset ranking. In terms of the distribution over the domain/instance space, our results are all distribution-independent.

Technically, our work involves the following components. First, after defining RSMs, we define the notion of 'strongly proper composite' surrogate losses for estimating a desired statistic τ (generalizing previous definitions of strongly proper composite surrogate losses for binary and multiclass class probability estimation [3, 42]). 1 Second, we give a general surrogate regret transfer bound for any RSM learning problem for which the statistic of interest can be accurately estimated via a strongly proper composite surrogate loss; this allows us to upper bound the target loss based regret in terms of the surrogate regret. Third, we use uniform convergence techniques to upper bound the surrogate regret of an (approximate) surrogate risk minimization algorithm, thus also upper bounding the target loss based regret for such an algorithm. We give two such results: one using d 1 covering numbers, and the other using Rademacher complexities. For the result in terms of Rademacher complexities, we make use of a vector-contraction inequality due to [32] to upper bound the Rademacher complexities of the loss function class ψ F associated with a vector-valued function class F and a surrogate loss ψ (that acts on vector-valued predictions and is Lipschitz w.r.t. the Euclidean metric) in terms of the Rademacher complexities of the real-valued projection classes F j . For the result in terms of d 1 covering numbers, we give a (to our knowledge, new) technical lemma that upper bounds the d 1 covering numbers of the loss function class ψ F associated with a vector-valued function class F and a surrogate loss ψ (that acts on vector-valued predictions and is Lipschitz w.r.t. the L 1 metric) in terms of the d 1 covering numbers of the projection classes F j ; this lemma may also be of independent interest. Finally, we show how these results can be applied to a variety of RSM learning problems.

While our results are broadly applicable to many RSM formulations, for each of the applications we consider, we include specific instantiations to RSM learning problems with sigmoid/softmax-of(multi-)linear forms for the statistics of interest, which can also be viewed as (multivariate) GLMs (see Table 1 for a summary). For the applications to binary classification (with 0-1 loss), multi-label learning (with Hamming loss), and subset ranking (with discounted cummulative gain (DCG) based loss), the Rademacher complexity based result gives tighter sample complexity bounds than those based on d 1 covering numbers. For the application to multiclass classification (with 0-1 loss), the two results are complementary: for n classes and data dimension p , the d 1 covering number based result gives a dimension-dependent sample complexity bound of ˜ O ( np/ϵ 2 ) for achieving squared estimation error ≤ ϵ ; the Rademacher complexity based result gives a dimension-independent bound of ˜ O ( n 2 /ϵ 2 ) . For the special case of binary classification with sigmoid-of-linear class probabilities, our results show that minimizing the standard binary logistic loss has a similar sample complexity as the GLM-tron algorithm of Kakade et al. (2011), but is computationally more efficient. In particular, the sample complexity for achieving squared estimation error ≤ ϵ is ˜ O (1 /ϵ 2 ) for both algorithms; however, the computational complexity of GLM-tron is ˜ O ( p/ϵ 3 ) , whereas that of the logistic regression algorithm is ˜ O ( p/ϵ 5 / 2 ) .

1 We note that the usage of the term 'proper' here is related to that in 'proper scoring rules' in the probability forecasting literature (see for example [12, 33, 34, 39, 1, 2] and references therein), and is distinct from that in 'proper learner' as commonly used in the PAC learning literature.

Table 1: Summary of selected PAC learning results with stochastic labels (results selected for comparison with ours, which are shown in red). Note that in terms of the distribution on the domain X , the results shown here are all distribution-independent. Here LTF stands for 'linear threshold function'. See Appendix A for details of the assumptions associated with RCN, Massart noise, GLM, and SIM. See Section 2 for details of notation used in the last row. The computational complexities listed for RSMs all assume implementations using Nesterov's accelerated gradient descent (AGD).

| Assumption on conditional label distribution P ( Y &#124; X = x )                                                | Learning target                                                                                                  | Sample complexity (for squared estimation error ≤ ϵ )                                                            | Sample complexity (for target loss based regret ≤ ϵ )                                                            | Computational complexity ( m = sample complexity from column 3 or 4)                                             |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Binary classification with 0-1 loss [ X ⊆ R p , Y = ̂ Y = {± 1 } ]                                               | Binary classification with 0-1 loss [ X ⊆ R p , Y = ̂ Y = {± 1 } ]                                               | Binary classification with 0-1 loss [ X ⊆ R p , Y = ̂ Y = {± 1 } ]                                               | Binary classification with 0-1 loss [ X ⊆ R p , Y = ̂ Y = {± 1 } ]                                               | Binary classification with 0-1 loss [ X ⊆ R p , Y = ̂ Y = {± 1 } ]                                               |
| Noisy LTF: RCN [10, 17, 21]                                                                                      | Best LTF                                                                                                         |                                                                                                                  | poly( p, 1 /ϵ )                                                                                                  | poly( p, 1 /ϵ )                                                                                                  |
| Noisy LTF: Massart noise [15]                                                                                    | Upper bound η on Massart noise                                                                                   |                                                                                                                  | ˜ O (poly( p ) /ϵ 3 )                                                                                            | poly( p, 1 /ϵ )                                                                                                  |
| GLM [25]                                                                                                         | Best LTF                                                                                                         | ˜ O (1 /ϵ 2 )                                                                                                    |                                                                                                                  | ˜ O ( m 3 / 2 p )                                                                                                |
| SIM [25]                                                                                                         | Best LTF                                                                                                         | (i) ˜ O ( p/ϵ 3 ) (ii) ˜ O (1 /ϵ 4 )                                                                             |                                                                                                                  | (i) ˜ O ( m 4 / 3 p ) (ii) ˜ O ( m 5 / 4 p )                                                                     |
| Sigmoid-of-linear [as special case of RSMs]                                                                      | Best LTF                                                                                                         | ˜ O (1 /ϵ 2 )                                                                                                    | ˜ O (1 /ϵ 4 )                                                                                                    | ˜ O ( m 5 / 4 p )                                                                                                |
| Multiclass classification with 0-1 loss ( n classes) [ X ⊆ R p , Y = ̂ Y = [ n ] ]                               | Multiclass classification with 0-1 loss ( n classes) [ X ⊆ R p , Y = ̂ Y = [ n ] ]                               | Multiclass classification with 0-1 loss ( n classes) [ X ⊆ R p , Y = ̂ Y = [ n ] ]                               | Multiclass classification with 0-1 loss ( n classes) [ X ⊆ R p , Y = ̂ Y = [ n ] ]                               | Multiclass classification with 0-1 loss ( n classes) [ X ⊆ R p , Y = ̂ Y = [ n ] ]                               |
| Softmax-of-multilinear [as special case of RSMs]                                                                 | Best multilinear multiclass classifier                                                                           | (i) ˜ O ( np/ϵ 2 ) (ii) ˜ O ( n 2 /ϵ 2 )                                                                         | (i) ˜ O ( np/ϵ 4 ) (ii) ˜ O ( n 2 /ϵ 4 )                                                                         | (i) ˜ O ( m 5 / 4 np ) (ii) ˜ O ( m 5 / 4 np )                                                                   |
| Multi-label prediction with Hamming loss ( s tags) [ X ⊆ R p , Y = ̂ Y = { 0 , 1 } s ]                           | Multi-label prediction with Hamming loss ( s tags) [ X ⊆ R p , Y = ̂ Y = { 0 , 1 } s ]                           | Multi-label prediction with Hamming loss ( s tags) [ X ⊆ R p , Y = ̂ Y = { 0 , 1 } s ]                           | Multi-label prediction with Hamming loss ( s tags) [ X ⊆ R p , Y = ̂ Y = { 0 , 1 } s ]                           | Multi-label prediction with Hamming loss ( s tags) [ X ⊆ R p , Y = ̂ Y = { 0 , 1 } s ]                           |
| Sigmoid-of-linear marginals [as special case of RSMs]                                                            | Best multilinear multi- label prediction model                                                                   | ˜ O ( s 3 /ϵ 2 )                                                                                                 | ˜ O ( s 5 /ϵ 4 )                                                                                                 | ˜ O ( m 5 / 4 sp )                                                                                               |
| Subset ranking with DCG metric ( s items, r rating levels) [ X ⊆ R p , Y = { 0 , 1 , . . . , r } s , ̂ Y = Π s ] | Subset ranking with DCG metric ( s items, r rating levels) [ X ⊆ R p , Y = { 0 , 1 , . . . , r } s , ̂ Y = Π s ] | Subset ranking with DCG metric ( s items, r rating levels) [ X ⊆ R p , Y = { 0 , 1 , . . . , r } s , ̂ Y = Π s ] | Subset ranking with DCG metric ( s items, r rating levels) [ X ⊆ R p , Y = { 0 , 1 , . . . , r } s , ̂ Y = Π s ] | Subset ranking with DCG metric ( s items, r rating levels) [ X ⊆ R p , Y = { 0 , 1 , . . . , r } s , ̂ Y = Π s ] |
| Sigmoid-of-linear scaled marginal expectations [as special case of RSMs]                                         | Best multilinear subset ranking model                                                                            | ˜ O ( s 3 /ϵ 2 )                                                                                                 | ˜ O ( r 4 s 5 /ϵ 4 )                                                                                             | ˜ O ( m 5 / 4 sp )                                                                                               |
| General learning problem (general X , Y , ̂ Y ) with general loss matrix L ∈ R Y× ̂ Y +                          | General learning problem (general X , Y , ̂ Y ) with general loss matrix L ∈ R Y× ̂ Y +                          | General learning problem (general X , Y , ̂ Y ) with general loss matrix L ∈ R Y× ̂ Y +                          | General learning problem (general X , Y , ̂ Y ) with general loss matrix L ∈ R Y× ̂ Y +                          | General learning problem (general X , Y , ̂ Y ) with general loss matrix L ∈ R Y× ̂ Y +                          |
| RSM: τ ◦ p ∈ Q , where p : X→ ∆ Y with p y ( x ) = P ( Y = y &#124; X = x ) , τ : ∆ Y → R d and Q ⊆ ( R d ) X    | Best prediction model in H ⊆ ̂ Y X , where H = pred ◦Q for pred : R d → ̂ Y s.t. ( τ , pred ) is L -calibrated   | ˜ O ( ρ 2 2 d 2 + B 2 γ 2 ϵ 2 ) where R m ( F j ) ≤ C √ m                                                        | ˜ O ( κ 4 ( ρ 2 2 d 2 + B 2 ) γ 2 ϵ 4 ) where R m ( F j ) ≤ C √ m                                                | ˜ O ( m 5 / 4 t ) where t = number of parameters to be learned                                                   |

Organization of the paper. Section 2 sets up the learning problem, defines RSMs, and gives our main results. Sections 3-6 then apply our results to binary classification, multiclass classification, multi-label learning, and subset ranking, respectively. All proofs can be found in the Appendix.

Notation. We denote by Z + the positive integers, and denote R + = [0 , ∞ ) , R ++ = (0 , ∞ ) , R = [ -∞ , ∞ ] , R + = [0 , ∞ ] . For a positive integer n , [ n ] := { 1 , . . . , n } , and Π n = { π : [ n ] → [ n ] | π is a bijection } . For a matrix A , we denote by a j the j -th column vector of A . For a finite set Y , ∆ Y := { p ∈ R Y + | ∑ y ∈Y p y = 1 } ; for Y = [ n ] , abbreviate ∆ n := ∆ [ n ] . We denote by 1 ( · ) the indicator function. For a vector u ∈ R n , argsort ( u ) := { π ∈ Π n | u i &gt; u j = ⇒ π ( i ) &lt; π ( j ) } . For two vectors u 1 , u 2 ∈ R n , the d 1 distance between them is d 1 ( u 1 , u 2 ) := 1 n ∥ u 1 -u 2 ∥ 1 . We use N 1 to denote d 1 covering numbers. For a set X , a class of real-valued functions F ⊆ { f : X→ R } , an integer m ∈ Z + , and an underlying probability distribution µ on X , we denote the Rademacher complexity of F for sample size m as R m ( F ) := E ( X 1 ,...,X m ) ∼ µ m [ E ( ϵ 1 ,...,ϵ m ) [sup f ∈F 1 m ∑ m i =1 ϵ i f ( X i )]] , where ϵ i are i.i.d. Rademacher random variables (each taking values +1 or -1 with probability 1 2 each). For a set C , an objective function f : C→ R , and a positive real number α &gt; 0 , an α -approximate minimizer of f over C returns a solution ̂ c ∈ C satisfying f ( ̂ c ) ≤ inf c ∈C f ( c ) + α .

## 2 Realizable-Statistic Models (RSMs) and Main Results

Section 2.1 sets up the learning problem and formally defines RSMs. Section 2.2 starts by defining some useful tools and then gives our main results.

## 2.1 Realizable-Statistic Models (RSMs)

Problem setup. We will consider a fairly general supervised learning setup. Specifically, let X be an instance space, and Y , ̂ Y be finite label and prediction spaces, respectively. 2 Let ℓ : Y × ̂ Y→ R + be a target loss function, where for each y ∈ Y , ̂ y ∈ ̂ Y , the loss ℓ ( y, ̂ y ) is the cost of predicting ̂ y when the true label is y ; equivalently, we will represent the loss function via a loss matrix L ∈ R Y× ̂ Y + , with ( y, ̂ y ) -th element given by L y, ̂ y = ℓ ( y, ̂ y ) . Let D ∈ ∆ X×Y be a joint probability distribution over X ×Y , which we will often write as D = ( µ, p ) , where µ ∈ ∆ X is the marginal of D over X and p : X→ ∆ Y denotes the conditional distribution over Y given an instance in X . Given a training sample S = (( X 1 , Y 1 ) , . . . , ( X m , Y m )) containing labeled examples drawn i.i.d. from D , the goal is to learn a prediction model h : X→ ̂ Y with small expected loss on a new example drawn from D , which we will refer to as the L -error or L -risk of h : er L D [ h ] = E ( X,Y ) ∼ D [ L Y,h ( X ) ] . In particular, for a class of models H ⊆ { h : X→ ̂ Y} and a class of probability distributions D ⊆ ∆ X×Y , a learning algorithm A that maps training samples S ∈ ∪ ∞ m =1 ( X × Y ) m to prediction models ̂ h S ∈ H is a probably approximately correct (PAC) learning algorithm for the learning problem ( L , H , D ) with target loss sample complexity function m L A : R + × (0 , 1] → Z + if for every ϵ &gt; 0 , δ ∈ (0 , 1] , every probability distribution D ∈ D and every m ≥ m L A ( ϵ, δ ) , P S ∼ D m ( er L D [ ̂ h S ] -inf h ∈H er L D [ h ] &gt; ϵ ) &lt; δ, and moreover, for every ϵ, δ , m L A ( ϵ, δ ) is the smallest integer satisfying the above. We will sometimes denote by er L D [ H ] := inf h ∈H er L D [ h ] the best L -error for D within H .

Realizable-statistic models (RSMs). The essence of our realizable-statistic models (RSMs) is to allow the labels to be stochastic and assume that some (vector-valued) 'statistic' of the conditional label distribution p ( x ) = ( P ( Y = y | X = x ) ) y ∈Y (associated with the underlying data distribution D ) belongs to some class of (vector-valued) functions Q ; in other words, we will assume that a statistic τ of the conditional label distribution p ( x ) is ' Q -realizable'. Formally, for any C ⊆ R d and d -dimensional statistic τ : ∆ Y →C , and any class of functions Q ⊆ { q : X→C} , define the class of ( τ , Q ) -RSM distributions over X × Y as follows:

<!-- formula-not-decoded -->

We will be interested in solving learning problems of the form ( L , H , D ( τ , Q ) -RSM ) . We note that the realizable and (fully) agnostic PAC learning models can both be recovered as special cases of RSMs; all the previously studied intermediate PAC learning models listed in Table 1 can also be recovered as special cases of RSMs (see Appendix B). Our algorithms for solving (certain types of) RSM learning problems of the form ( L , H , D ( τ , Q ) -RSM ) will typically do the following: given a training sample S , they will first (sometimes implicitly) find an estimate ̂ q S : X→C for the true statistic function q ∗ ( x ) = τ ( p ( x )) , and then will return a prediction model ̂ h S : X→ ̂ Y effectively constructed from ̂ q S . Accordingly, for such an algorithm A , in addition to its target loss sample complexity m L A defined above, we will also be interested in its squared τ -estimation error sample complexity function m τ A : R + × (0 , 1] → Z + , where for every ϵ &gt; 0 , δ ∈ (0 , 1] , m τ A ( ϵ, δ ) is the smallest integer such that every probability distribution D ∈ D and every m ≥ m τ A ( ϵ, δ ) , P S ∼ D m ( E X ∼ µ [ ∥ ̂ q S ( X ) -q ∗ ( X ) ∥ 2 2 ] &gt; ϵ ) &lt; δ.

## 2.2 Main Results

We start by defining some tools that will be needed for our main results - specifically, the tools of L -calibrated statistics and strongly proper composite surrogate losses. Before doing so, we recall:

Definition 1 ( Bayes L -error and Bayes L -optimal model ) . Let L ∈ R Y× ̂ Y + be any loss matrix. The Bayes L -error for D , denoted er L , ∗ D , is the smallest L -error under D over all possible prediction models: er L , ∗ D = inf h : X→ ̂ Y er L D [ h ] . A Bayes L -optimal model for D , denoted h L , ∗ D : X→ ̂ Y , is any prediction model that achieves the Bayes L -error for D : er L D [ h L , ∗ D ] = er L , ∗ D .

2 Our model and results easily extend to more general Y , ̂ Y ; we take these to be finite for simplicity.

Definition 2 ( L -calibrated statistics [2] ) . Let L ∈ R Y× ̂ Y + be any loss matrix. Let d ∈ Z + and C ⊆ R d . A statistic τ : ∆ Y →C is L -calibrated if ∃ a mapping pred : C→ ̂ Y such that for all distributions D = ( µ, p ) ∈ ∆ X×Y , a Bayes L -optimal model for D can be obtained from τ ( p ( x )) as h L , ∗ D ( x ) = pred ( τ ( p ( x ))) . We will also say the statistic-mapping pair ( τ , pred ) is L -calibrated.

The convex surrogate risk minimization algorithms we will consider will minimize the empirical surrogate risk 1 m ∑ m i =1 ψ ( y i , f ( x i )) , for some suitably defined convex surrogate loss ψ : Y ×C ′ → R + that acts on vector predictions in some convex set C ′ ⊆ R d ′ (for a suitable integer d ′ ), over some class of vector-valued functions F ⊆ { f : X→C ′ } to learn a vector-valued function ̂ f S ∈ F , and then will return a prediction model ̂ h S : X→ ̂ Y of the form ̂ h S ( x ) = decode ( ̂ f S ( x )) for a suitable decoding function decode : C ′ → ̂ Y . We will be especially interested in surrogate losses whose minimization yields accurate estimates of a desired statistic τ : ∆ Y →C . To this end, we define below the notion of strongly proper (composite) surrogate losses ψ for a statistic τ , for which the expected surrogate loss E Y ∼ p [ ψ ( Y, u )] is 'strongly' minimized at (possibly an invertible transformation of) the correct statistic value τ ( p ) ; this generalizes the definition of strongly proper (composite) surrogate losses for binary and multiclass class probability estimation [3, 42] to estimation of general statistics: 3

Definition 3 ( Strongly proper composite surrogate losses for a statistic τ ) . Let d ∈ Z + and C ⊆ R d , and let τ : ∆ Y →C be any statistic of interest. Let d ′ ∈ Z + , and let C ′ ⊆ R d ′ be such that C is in one-to-one correspondence with a subset of C ′ . If C is in one-to-one correspondence with C ′ itself, then let λ : C→C ′ be an invertible mapping with inverse λ -1 : C ′ →C ; otherwise, let λ : C→C ′ be a one-to-one mapping and let S = {S q : q ∈ C} be a partition of C ′ such that λ ( q ) ∈ S q ∀ q ∈ C , and let λ -1 : C ′ →C denote an 'extended' inverse that assigns λ -1 ( u ) = q ∀ u ∈ S q . Let γ &gt; 0 . A surrogate loss ψ : Y × C ′ → R + acting on C ′ is γ -strongly proper composite for statistic τ with link function λ if E Y ∼ p [ ψ ( Y, u ) -ψ ( Y, λ ( τ ( p )))] ≥ γ 2 ∥ λ -1 ( u ) -τ ( p ) ∥ 2 2 ∀ p ∈ ∆ Y , u ∈ C ′ .

Weare now ready to state our main results. We start by giving a general surrogate regret transfer bound for RSM learning problems for which the statistic of interest admits a strongly proper composite surrogate loss; this allows us to upper bound the target loss based regret in terms of the surrogate regret. Specifically, the theorem below effectively shows that given a target loss L , an L -calibrated statistic-mapping pair ( τ , pred ) satisfying a certain condition (which allows the L -regret to be upperbounded by the squared τ -estimation error), a class of 'statistic' functions Q , and a strongly proper composite surrogate loss ψ for τ with link function λ , for any data distribution D ∈ D ( τ , Q ) -RSM, both the squared τ -estimation error of any q ∈ Q and the target L -regret (excess L -risk) of a model h = pred ◦ q in the class of models H = pred ◦ Q can be upper bounded in terms of the surrogate ψ -regret (excess ψ -risk) of the vector-valued function f = λ ◦ q in the class of vector-valued functions F = λ ◦ Q . The proof of this theorem is inspired by the proof of a surrogate regret transfer bound given in a different context (Bayes consistent multi-label learning with the F -measure) by [43].

Theorem 1 ( Surrogate regret transfer bound for RSMs that admit strongly proper composite surrogate losses ) . Let X be any instance space and Y , ̂ Y be any label and prediction spaces, respectively. Let L ∈ R Y× ̂ Y + be a loss matrix. Let d ∈ Z + and C ⊆ R d . Let τ : ∆ Y →C and pred : C→ ̂ Y be such that ( τ , pred ) is an L -calibrated statistic-mapping pair, and suppose ∃ κ &gt; 0 s.t.

<!-- formula-not-decoded -->

Let Q ⊆ { q : X→C} be a class of 'statistic' functions, and let ψ : Y × R d → R + be a γ -strongly proper composite surrogate loss for τ with link function λ : C→ R d . 4 Let H ⊆ { h : X→ ̂ Y} be defined as H := pred ◦ Q = { h : X→ ̂ Y | ∃ q ∈ Q s.t. h ( x ) = pred ( q ( x )) ∀ x ∈ X} , let F ⊆ { f : X→ R d } be defined as F := λ ◦ Q = { f : X→ R d | ∃ q ∈ Q s.t. f ( x ) = λ ( q ( x )) ∀ x ∈ X} , and

3 The reason for introducing a new space C ′ ⊆ R d ′ is that often it is easier to minimize a surrogate loss acting on a space C ′ different from C (in many of our examples, we will have C ⊊ R d , d ′ = d and C ′ = R d ).

4 As in Definition 3, if C is in one-to-one correspondence with R d itself, then we will assume that λ : C→ R d is an invertible mapping with inverse λ -1 : R d →C ; otherwise, we will assume that λ : C→ R d is a one-to-one mapping and S = {S q : q ∈ C} is a partition of R d such that λ ( q ) ∈ S q ∀ q ∈ C , and λ -1 : R d →C denotes an 'extended' inverse that assigns λ -1 ( u ) = q ∀ u ∈ S q . Note that in the notation of Definition 3, here we have set d ′ = d and C ′ = R d (this is both for simplicity and because this suffices for our examples); however, the theorem easily extends to any suitable d ′ and C ′ .

define decode : R d → ̂ Y as decode := pred ◦ λ -1 . Suppose that ψ ( y, f ( x )) ∈ [0 , B ] ∀ x ∈ X , y ∈ Y , f ∈ F for some B &gt; 0 . Then for any f ∈ F and any D ∈ D ( τ , Q ) -RSM ,

<!-- formula-not-decoded -->

In practice, when applying the above theorem, it will often be the case that the class of 'statistic' functions Q is of the form Q = σ ◦ F for some pre-specified class of vector-valued functions F (such as bounded multi-linear functions) and some 'transfer' function σ ; in such settings, it can be helpful to choose a strongly proper composite surrogate loss whose inverse link function λ -1 is matched to σ (we will see several examples of this in the next few sections).

The above result can be combined with any upper bound on the surrogate ψ -regret in F to yield upper bounds on both the squared τ -estimation error and the target L -regret in H , which in turn can then be converted to sample complexity bounds. The following two results make this concrete for standard unregularized surrogate risk minimization; the first result makes use of d 1 covering numbers, while the second makes use of Rademacher complexities. For the result in terms of d 1 covering numbers, we make use of standard uniform convergence techniques, together with a (to our knowledge, new) technical lemma (given in Appendix B) that upper bounds the d 1 covering numbers of the loss function class ψ F = { ψ f : X × Y→ R + | ∃ f ∈ F s.t. ψ f ( x, y ) = ψ ( y, f ( x )) } associated with a vector-valued function class F and a surrogate loss ψ (that acts on vector-valued predictions and is Lipschitz w.r.t. the L 1 metric) in terms of the d 1 covering numbers of the real-valued projection function classes {F j } j (defined below); this lemma may also be of independent interest. For the result in terms of Rademacher complexities, we make use of uniform convergence techniques, together with a vector-contraction inequality due to [32] that upper bounds the Rademacher complexities of the loss function class ψ F associated with a vector-valued function class F and a surrogate loss ψ (that acts on vector-valued predictions and is Lipschitz w.r.t. the Euclidean metric) in terms of the Rademacher complexities of the real-valued projection classes {F j } j .

Theorem 2 ( RSM learning bounds for surrogate risk minimizers via d 1 covering numbers ) . Under the conditions of Theorem 1, suppose the surrogate loss ψ is ρ 1 -Lipschitz in the second argument with respect to the L 1 metric, so that ψ ( y, u 1 ) -ψ ( y, u 2 ) ≤ ρ 1 ∥ u 1 -u 2 ∥ 1 ∀ y, u 1 , u 2 , and suppose that the function classes F j = { f j : X→ R | ∃ f ∈ F s.t. f j ( x ) = ( f ( x )) j ∀ x } , j ∈ [ d ] each have bounded d 1 covering numbers N 1 ( ϵ, F j , m ) (polynomial in m and 1 /ϵ ). Then a surrogate risk minimization algorithm A which, given a training sample S of size m , finds an (16 B/ √ m ) -approximate minimizer ̂ f S ∈ F of the empirical surrogate risk 1 m ∑ m i =1 ψ ( y i , f ( x i )) over F , and produces a τ -statistic estimate ̂ q S ( x ) = λ -1 ( ̂ f S ( x )) and a prediction model ̂ h S ∈ H given by ̂ h S ( x ) = decode ( ̂ f S ( x )) (or equivalently, ̂ h S ( x ) = pred ( ̂ q S ( x )) ), is a PAC learning algorithm for the RSM learning problem ( L , H , D ( τ , Q ) -RSM ) with squared τ -estimation error sample complexity m τ A ( ϵ, δ ) ≤ min { m 0 ∈ Z + : m ≥ m 0 = ⇒ m ≥ 1152 B 2 γ 2 ϵ 2 ( ∑ d j =1 ln ( N 1 ( γϵ 48 ρ 1 d , F j , 2 m )) + ln ( 4 δ ))} , and with target loss sample complexity m L A ( ϵ, δ ) ≤ min { m ∈ Z + : m ≥ m 0 = ⇒ m ≥ 1152 κ 4 B 2 γ 2 ϵ 4 ( ∑ d j =1 ln ( N 1 ( γϵ 2 48 κ 2 ρ 1 d , F j , 2 m )) +ln ( 4 δ ))} . In particular, if the d 1 covering numbers of the function classes F j have upper bounds of the form N 1 ( ϵ, F j , m ) ≤ ϕ ( ϵ, F j ) (i.e., bounds independent of sample size m ), then m τ A ( ϵ, δ ) ≤ 1152 B 2 γ 2 ϵ 2 ( ∑ d j =1 ln ( ϕ ( γϵ 48 ρ 1 d , F j )) +ln ( 4 δ )) , and m L A ( ϵ, δ ) ≤ 1152 κ 4 B 2 γ 2 ϵ 4 ( ∑ d j =1 ln ( ϕ ( γϵ 2 48 κ 2 ρ 1 d , F j )) +ln ( 4 δ )) .

Theorem 3 ( RSM learning bounds for surrogate risk minimizers via Rademacher complexities ) . Under the conditions of Theorem 1, suppose the surrogate loss ψ is ρ 2 -Lipschitz in the second argument with respect to the Euclidean metric, so that ψ ( y, u 1 ) -ψ ( y, u 2 ) ≤ ρ 2 ∥ u 1 -u 2 ∥ 2 ∀ y, u 1 , u 2 , and suppose that the function classes F j = { f j : X→ R | ∃ f ∈ F s.t. f j ( x ) = ( f ( x )) j ∀ x } , j ∈ [ d ] each have non-negative, decreasing Rademacher complexities R m ( F j ) (decreasing in m ). Then a surrogate risk minimization algorithm A which, given a training sample S of size m , finds an ( B/ (2 √ m )) -approximate minimizer ̂ f S ∈ F of the empirical surrogate risk 1 m ∑ m i =1 ψ ( y i , f ( x i )) over F , and produces a τ -statistic estimate ̂ q S ( x ) = λ -1 ( ̂ f S ( x )) and a prediction model ̂ h S ∈ H given by ̂ h S ( x ) = decode ( ̂ f S ( x )) (or equivalently, ̂ h S ( x ) = pred ( ̂ q S ( x )) ), is a PAC learning algorithm for the RSM learning problem ( L , H , D ( τ , Q ) -RSM ) with squared τ -estimation error sample complexity m τ A ( ϵ, δ ) ≤ min { m 0 ∈ Z + : m ≥

m 0 = ⇒ 3 ( 2 √ 2 ρ 2 · ∑ d j =1 R m ( F j ) + B √ ln(2 /δ ) m ) ≤ γϵ 2 } , and with target loss sample complexity m L A ( ϵ, δ ) ≤ min { m ∈ Z + : m ≥ m 0 = ⇒ 3 ( 2 √ 2 ρ 2 · ∑ d j =1 R m ( F j ) + B √ ln(2 /δ ) m ) ≤ γϵ 2 2 κ 2 } . In particular, if ∃ C &gt; 0 such that the Rademacher complexities of the function classes F j have upper bounds of the form R m ( F j ) ≤ C/ √ m ∀ j ∈ [ d ] , then m τ A ( ϵ, δ ) ≤ 36 γ 2 ϵ 2 ( 2 √ 2 ρ 2 Cd + B √ ln(2 /δ ) ) 2 , and m L A ( ϵ, δ ) ≤ 36 κ 4 γ 2 ϵ 4 ( 2 √ 2 ρ 2 Cd + B √ ln(2 /δ ) ) 2 .

In Sections 3-6 below, we apply the above results to a variety of RSM learning problems, including binary classification, multiclass classification, multi-label prediction, and subset ranking. While our results are broadly applicable to many RSM formulations, for each of the applications below, we will include specific instantiations to RSM learning problems with sigmoid/softmax-of-(multi-)linear forms for the statistics of interest. To this end, we will make use of the following upper bounds on the d 1 covering numbers and the Rademacher complexity of (bounded) linear functions:

Proposition 4. Let R,W &gt; 0 . Let X ⊆ { x ∈ R p | ∥ x ∥ 2 ≤ R } . Let F linear = { f : X→ R | ∃ w ∈ R p , ∥ w ∥ 2 ≤ W s.t. f ( x ) = w ⊤ x ∀ x } . Then for any m ∈ Z + and any ϵ &gt; 0 : (i) N 1 ( ϵ, F linear , m ) ≤ (1 /ϵ ) p ; (ii) N 1 ( ϵ, F linear , m ) ≤ (4 R 2 W 2 /ϵ 2 + 1) ⌈ 2 R 2 W 2 /ϵ 2 ⌉ ; and (iii) 0 ≤ R m ( F linear ) ≤ RW/ √ m .

## 3 Binary Classification

Consider a binary classification problem with instance space X , label and prediction spaces Y = ̂ Y = {± 1 } , and the standard 0-1 loss L 0-1 ∈ R {± 1 }×{± 1 } + with ℓ 0-1 ( y, ̂ y ) = 1 ( ̂ y = y ) . Let C = [0 , 1] , and define the 'projection-onto-( +1 )th-component' statistic τ +1 : ∆ {± 1 } → [0 , 1] and mapping pred 0-1 : [0 , 1] →{± 1 } as

̸

<!-- formula-not-decoded -->

Then ( τ +1 , pred 0-1 ) is an L 0-1 -calibrated pair. Moreover, as is well known (also see Appendix C),

<!-- formula-not-decoded -->

Therefore, for any class of 'statistic' functions Q ⊆ { q : X→ [0 , 1] } and corresponding hypothesis class H = pred 0-1 ◦ Q , Theorem 2 establishes that any convex surrogate risk minimization algorithm minimizing a strongly proper composite surrogate loss for τ +1 over a suitable class of functions F ⊆ { f : X→ R } yields an efficient PAC learning algorithm for the RSM learning problem ( L 0-1 , H , D ( τ +1 , Q ) -RSM ) . While this result can be applied to any class Q and suitable surrogate loss ψ , the following theorem makes this concrete for the class of sigmoid-of-linear models Q sigmoid-of-linear and the binary logistic loss ψ log (defined below).

Theorem 5 ( PAC learning algorithm for binary classification with sigmoid-of-linear class probabilities ) . Consider a binary classification problem, with X ⊆ { x ∈ R p | ∥ x ∥ 2 ≤ R } for some R &gt; 0 , Y = ̂ Y = {± 1 } , and with the standard 0-1 loss L 0-1 as above. Let τ +1 and pred 0-1 be as defined above. Let σ : R → [0 , 1] be the sigmoid function σ ( u ) = 1 / (1 + e -u ) , and let

<!-- formula-not-decoded -->

for some W &gt; 0 . Let H linear := pred 0-1 ◦ Q sigmoid-of-linear , i.e. H linear = { h : X→{± 1 } | ∃ w ∈ R p , ∥ w ∥ 2 ≤ W s.t. h ( x ) = sign( w ⊤ x ) ∀ x } . Let ψ log : {± 1 } × R → R + be the binary logistic loss:

<!-- formula-not-decoded -->

Let F linear = { f : X→ R | ∃ w ∈ R p , ∥ w ∥ 2 ≤ W s.t. f ( x ) = w ⊤ x ∀ x } . Then an algorithm A which, given a training sample S of size m , finds an (ln(1 + e RW ) / (2 √ m )) -approximate minimizer ̂ f S ∈ F linear of the empirical surrogate risk 1 m ∑ m i =1 ψ log ( y i , f ( x i )) over F linear , and produces a τ +1 -statistic estimate ̂ q S ( x ) = σ ( ̂ f S ( x )) and prediction model ̂ h S ∈ H linear given by ̂ h S = sign ◦ ̂ f S (equivalently, ̂ h S = pred 0-1 ◦ ̂ q S ), is a PAC learning algorithm for the RSM learning problem ( L 0-1 , H linear , D ( τ +1 , Q sigmoid-of-linear ) -RSM ) with squared τ +1 -estimation error sample complexity m τ +1 A ( ϵ, δ ) = O ( 1 ϵ 2 ln ( 1 δ )) , and with target loss sample complexity m L 0-1 A ( ϵ, δ ) = O ( 1 ϵ 4 ln ( 1 δ )) .

## 4 Multiclass Classification

Consider now a multiclass classification problem with instance space X , label and prediction spaces Y = ̂ Y = [ n ] for n &gt; 2 , and the multiclass 0-1 loss L 0-1 ( n ) ∈ R n × n + with ℓ 0-1 ( y, ̂ y ) = 1 ( ̂ y = y ) . Let C = ∆ n , and define the 'identity' statistic τ id : ∆ n → ∆ n and mapping pred 0-1 ( n ) : ∆ n → [ n ] as

<!-- formula-not-decoded -->

Then ( τ id , pred 0-1 ( n ) ) is an L 0-1 ( n ) -calibrated pair. Moreover, as shown in Appendix D,

<!-- formula-not-decoded -->

Therefore, for any class of 'statistic' functions Q ⊆ { q : X→ ∆ n } and corresponding hypothesis class H = pred 0-1 ( n ) ◦ Q , Theorem 2 establishes that any convex surrogate risk minimization algorithm minimizing a strongly proper composite surrogate loss for τ id over a suitable class of functions F ⊆ { f : X→ R n } yields an efficient PAC learning algorithm for the RSM learning problem ( L 0-1 ( n ) , H , D ( τ id , Q ) -RSM ) . While this result can be applied to any class Q and suitable surrogate loss ψ , the following theorem makes this concrete for the class of softmax-of-multilinear models Q softmax-of-multilinear and the multiclass logistic loss ψ mlog (defined below).

Theorem 6 ( PAC learning algorithm for multiclass classification with softmax-of-multilinear class probabilities ) . Consider a multiclass classification problem, with X ⊆ { x ∈ R p | ∥ x ∥ 2 ≤ R } for some R &gt; 0 , Y = ̂ Y = [ n ] , and with the multiclass 0-1 loss L 0-1 ( n ) as above. Let τ id and pred 0-1 ( n ) be as defined above. Let σ : R n → ∆ n be the softmax function ( σ ( u )) y = e u y / ( ∑ n y ′ =1 e u y ′ ) ∀ y ∈ [ n ] , and let

<!-- formula-not-decoded -->

for some W &gt; 0 . Let H multiclass-linear := pred 0-1 ( n ) ◦ Q softmax-of-mlinear , i.e. H multiclass-linear = { h : X→ [ n ] | ∃ W ∈ R p × n , ∥ w y ∥ 2 ≤ W ∀ y s.t. h ( x ) ∈ argmax y ∈ [ n ] ( w ⊤ y x ) ∀ x } . Let ψ mlog : [ n ] × R n → R + be the multiclass logistic loss

<!-- formula-not-decoded -->

Define decode 0-1 ( n ) : R n → [ n ] as decode 0-1 ( n ) ( u ) ∈ argmax ̂ y ∈ [ n ] u ̂ y , and let F multiclass-linear = { f : X→ R n | ∃ W ∈ R p × n , ∥ w y ∥ 2 ≤ W ∀ y s.t. f ( x ) = W ⊤ x ∀ x } . Then an algorithm A which, given a training sample S of size m , finds an ((ln( n ) + 2 RW ) / (2 √ m )) -approximate minimizer ̂ f S ∈ F multiclass-linear of the empirical surrogate risk 1 m ∑ m i =1 ψ mlog ( y i , f ( x i )) over F multiclass-linear , and produces a τ id -statistic estimate ̂ q S ( x ) = σ ( ̂ f S ( x )) and a prediction model ̂ h S ∈ H multiclass-linear given by ̂ h S ( x ) = decode 0-1 ( n ) ( ̂ f S ( x )) (or equivalently, ̂ h S ( x ) = pred 0-1 ( n ) ( ̂ q S ( x )) ), is a PAC learning algorithm for the RSM learning problem ( L 0-1 ( n ) , H multiclass-linear , D ( τ id , Q softmax-of-mlinear ) -RSM ) with squared τ id -estimation error sample complexity m τ id A ( ϵ, δ ) and target loss sample complexity m L 0-1 ( n ) A ( ϵ, δ ) upper bounded as follows:

(i) (Dimension-dependent)

<!-- formula-not-decoded -->

## 5 Multi-Label Learning

̸

Next, consider a multi-label prediction problem such as in image tagging, with s tags [ s ] = { 1 , . . . , s } , several of which can be active in an instance simultaneously, and the goal is to predict for a new instance which of the s tags are active. Specifically, let X be any instance space, with label and prediction spaces Y = ̂ Y = { 0 , 1 } s (labels are represented as vectors y ∈ { 0 , 1 } s , with y j = 1 indicating that the j -th tag is active), and consider the Hamming loss L Ham ∈ R { 0 , 1 } s ×{ 0 , 1 } s + with ℓ Ham ( y , ̂ y ) = ∑ s j =1 1 ( ̂ y j = y j ) . Let C = [0 , 1] s , and define the s -dimensional 'marginals' statistic

̸

<!-- formula-not-decoded -->

τ marginals : ∆ { 0 , 1 } s → [0 , 1] s and mapping pred Ham : [0 , 1] s →{ 0 , 1 } s

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then ( τ marginals , pred Ham ) is an L Ham -calibrated pair. Moreover, as shown in Appendix E,

<!-- formula-not-decoded -->

Therefore, for any class of 'statistic' functions Q ⊆ { q : X→ [0 , 1] s } and corresponding hypothesis class H = pred Ham ◦Q , Theorem 2 establishes that any convex surrogate risk minimization algorithm minimizing a strongly proper composite surrogate loss for τ marginals over a suitable class of functions F ⊆ { f : X→ R s } yields an efficient PAC learning algorithm for the RSM learning problem ( L Ham , H , D ( τ marginals , Q ) -RSM ) . While this result can be applied to any class Q and suitable surrogate loss ψ , below we make this concrete for the class of sigmoid-of-multilinear models Q sigmoid-of-multilinear and the 'binary relevance' logistic-based multi-label surrogate loss ψ BRlog (defined below). 5

Theorem 7 ( PAC learning algorithm for multi-label prediction with sigmoid-of-multilinear marginals ) . Consider a multi-label prediction problem, with X ⊆ { x ∈ R p | ∥ x ∥ 2 ≤ R } for some R &gt; 0 , Y = ̂ Y = { 0 , 1 } s , and with the Hamming loss L Ham as above. Let τ marginals and pred Ham be as defined above. Let σ : R → [0 , 1] be the sigmoid function σ ( u ) = 1 / (1 + e -u ) , and let

<!-- formula-not-decoded -->

for some W &gt; 0 . Let H sign multilinear := pred Ham ◦ Q sigmoid-of-multilinear , i.e. H sign multilinear = { h : X→{ 0 , 1 } s | ∃ W ∈ R p × s , ∥ w j ∥ 2 ≤ W ∀ j s.t. h j ( x ) = sign( w ⊤ j x ) ∀ x , j } . Let ψ BRlog : { 0 , 1 } s × R s → R + be the 'binary relevance' logistic-based multi-label surrogate loss defined by

<!-- formula-not-decoded -->

Define decode Ham : R s →{ 0 , 1 } s as ( decode Ham ( u )) j := sign( u j ) ∀ j ∈ [ s ] , and let F multilinear = { f : X→ R s | ∃ W ∈ R p × s , ∥ w j ∥ 2 ≤ W ∀ j s.t. f ( x ) = W ⊤ x ∀ x } . Then an algorithm A which, given a training sample S of size m , finds an ( s ln(1+ e RW ) / (2 √ m )) -approximate minimizer ̂ f S ∈ F multilinear of the empirical surrogate risk 1 m ∑ m i =1 ψ BRlog ( y i , f ( x i )) over F multilinear , and produces a τ marginals -statistic estimate ( ̂ q S ( x )) j = σ (( ̂ f S ( x )) j ) and a prediction model ̂ h S ∈ H sign multilinear given by ̂ h S ( x ) = decode Ham ( ̂ f S ( x )) (or equivalently, ̂ h S ( x ) = pred Ham ( ̂ q S ( x )) ), is a PAC learning algorithm for the RSM learning problem ( L Ham , H sign multilinear , D ( τ marginals , Q sigmoid-of-multilinear ) -RSM ) with squared τ marginals -estimation error sample complexity m τ marginals A ( ϵ, δ ) = O ( s 2 ϵ 2 ( s +ln ( 1 δ )) ) , and with target loss sample complexity m L Ham A ( ϵ, δ ) = O ( s 4 ϵ 4 ( s +ln ( 1 δ )) ) .

## 6 Subset Ranking

As a final example, consider a subset ranking problem such as those that arise in information retrieval, wherein each instance contains a query and a subset of s documents, together with some relevance judgments for each of the s documents as labels, and given a new instance containing a new query and a new subset of s documents, the goal is to find a good ranking of the s documents for that query. Specifically, let X be any instance space, and let the label space be Y = { 0 , 1 , . . . , r } s , where each document is graded on a scale of 0 to r ; the prediction space is ̂ Y = Π s . A widely used performance measure for such problems is the discounted cumulative gain (DCG); in loss form, one version of the DCG loss L DCG is given by ℓ DCG ( y , ̂ π ) = Z -∑ s j =1 y j · disc ( ̂ π ( j )) , where disc : [ s ] → [0 , 1] is a non-increasing 'discount' function that discounts documents placed lower in the ranking, often taken to be disc ( a ) = 1 / (log 2 ( a +1)) , and Z is a constant that ensures non-negativity of the loss [18, 24]. Let C = [0 , 1] s , and define the s -dimensional 'scaled marginal expectations' property

5 The 'binary relevance' approach effectively solves s binary problems, one for each tag [41, 11]. One could also apply Theorem 5 s times (drawing a fresh sample of size O ( s 2 ϵ 2 (ln( s δ ))) for each tag), yielding a sample complexity of O ( s 3 ϵ 2 (ln( s δ ))) . The result of Theorem 7 improves over this by removing a multiplicative ln( s ) factor. We also note that contrary to popular belief, Theorem 7 indicates that the binary relevance approach does not require the s tags to be conditionally independent given x in order to be an effective learning algorithm.

<!-- formula-not-decoded -->

Then ( τ sc-marg-exp , pred DCG ) is an L DCG -calibrated pair. Moreover, as shown in Appendix F,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where disc = ( disc (1) , . . . , disc ( s )) ⊤ ∈ [0 , 1] s . Therefore, for any class of 'statistic' functions Q ⊆ { q : X→ [0 , 1] s } and corresponding hypothesis class H = pred DCG ◦ Q , Theorem 2 establishes that any convex surrogate risk minimization algorithm minimizing a strongly proper composite surrogate loss for τ sc-marg-exp over a suitable class of functions F ⊆ { f : X→ R s } yields an efficient PAC learning algorithm for the RSM learning problem ( L DCG , H , D ( τ sc-marg-exp , Q ) -RSM ) . While this result can be applied to any class Q and suitable surrogate loss ψ , the following theorem makes this concrete for the class of sigmoid-of-multilinear models Q sigmoid-of-multilinear and a suitably weighted multivariate logistic-based surrogate loss ψ wlog that we introduce here (defined below).

̂

Theorem 8 ( PAC learning algorithm for subset ranking with sigmoid-of-multilinear scaled marginal expectations ) . Consider a subset ranking problem, with X ⊆ { x ∈ R p | ∥ x ∥ 2 ≤ R } for some R &gt; 0 , Y = { 0 , 1 , . . . , r } s and ̂ Y = Π s , and with the DCG loss L DCG as above. Let τ sc-marg-exp and pred DCG be as defined above. Let σ and Q sigmoid-of-multilinear be as defined in Theorem 7, and let H sort multilinear := pred DCG ◦ Q sigmoid-of-multilinear , i.e. H sort multilinear = { h : X→ Π s | ∃ W ∈ R p × s , ∥ w j ∥ 2 ≤ W ∀ j s.t. h ( x ) ∈ argsort ( W ⊤ x ) ∀ x } . Let ψ wlog : { 0 , 1 , . . . , r } s × R s → R + be a multivariate weighted logistic-based surrogate loss defined by

<!-- formula-not-decoded -->

Define decode DCG : R s → Π s as decode DCG ( u ) ∈ argsort ( u ) , and let F multilinear be as defined in Theorem 7. Then an algorithm A which, given a training sample S of size m , finds an ( s ln(1 + e RW ) / (2 √ m )) -approximate minimizer ̂ f S ∈ F multilinear of the empirical surrogate risk 1 m ∑ m i =1 ψ wlog ( y i , f ( x i )) over F multilinear , and produces a τ sc-marg-exp -statistic estimate ( ̂ q S ( x )) j = σ (( ̂ f S ( x )) j ) and a prediction model ̂ h S ∈ H sort multilinear given by ̂ h S ( x ) = decode DCG ( ̂ f S ( x )) (or equivalently, ̂ h S ( x ) = pred DCG ( ̂ q S ( x )) ), is a PAC learning algorithm for the RSM learning problem ( L DCG , H sort multilinear , D ( τ sc-marg-exp , Q sigmoid-of-multilinear ) -RSM ) with squared τ sc-marg-exp -estimation error sample complexity m τ sc-marg-exp A ( ϵ, δ ) = O ( s 2 ϵ 2 ( s +ln ( 1 δ )) ) , and with target loss sample complexity m L DCG A ( ϵ, δ ) = O ( r 4 s 2 ·∥ disc ∥ 4 2 ϵ 4 ( s +ln ( 1 δ )) ) = O ( r 4 s 4 ϵ 4 ( s +ln ( 1 δ )) ) .

## 7 Conclusion

We have studied a flexible class of intermediate PAC leaning models that we call realizable-statistic models (RSMs), wherein we allow labels to be stochastic but assume that some vector-valued statistic of the conditional label distribution comes from a known function class. RSMs interpolate between the realizable and fully agnostic settings, and also recover several previously studied intermediate PAC learning models as special cases. We have shown that for RSMs where the statistic of interest can be estimated via a convex 'strongly proper composite' surrogate loss, minimizing this convex surrogate loss yields a computationally efficient learning algorithm with finite sample complexity bounds, and have demonstrated applications of these results to a broad range of RSM learning problems including binary and multiclass classification, multi-label learning, and subset ranking.

RSMs are also connected to the structured prediction framework studied in [16], where the target loss function can be written as ℓ ( y, ̂ y ) = ϕ 1 ( y ) ⊤ A ϕ 2 ( ̂ y ) for some embedding functions ϕ 1 : Y → R k , ϕ 2 : ̂ Y → R k and matrix A ∈ R k × k . 6 In particular, [16] effectively considers the 'conditional mean embedding' statistic q ∗ ( x ) = E [ ϕ 1 ( Y ) | X = x ] , and assumes that this statistic belongs to some class of functions (such as multilinear functions or a vector-valued RKHS); this statistic is then estimated to produce ̂ q ( x ) . Thus this setting can also be viewed as a special case of our RSM framework (indeed, the quadratic surrogate used in [16] is also a strongly proper composite surrogate for the above statistic; the target loss based sample complexity bounds of [16] are of the form ˜ O ( β/ϵ 4 ) , where β captures problem-dependent parameters, and are therefore comparable to our bounds).

6 More generally, [16] allows embedding into a Hilbert space F .

## Acknowledgments and Disclosure of Funding

Warm thanks to Rob Schapire for valuable discussions and suggestions related to this work; to Peter Bartlett for valuable pointers; and to Nishant Agarwal and Ananya Mukherjee for help with typesetting parts of this paper. Thanks also to the anonymous reviewers of this work for helpful comments. This material is based upon work supported in part by the US National Science Foundation (NSF) under Grant No. 1934876. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.

## References

- [1] Jacob D. Abernethy and Rafael M. Frongillo. A characterization of scoring rules for linear properties. In Proceedings of the 25th Annual Conference on Learning Theory (COLT) , 2012.
- [2] Arpit Agarwal and Shivani Agarwal. On consistent surrogate risk minimization and property elicitation. In Proceedings of The 28th Conference on Learning Theory (COLT) , 2015.
- [3] Shivani Agarwal. Surrogate regret bounds for the area under the ROC curve via strongly proper losses. In Proceedings of the 26th Annual Conference on Learning Theory (COLT) , 2013.
- [4] Dana Angluin and Philip D. Laird. Learning from noisy examples. Mach. Learn. , 2(4):343-370, 1988.
- [5] Martin Anthony and Peter L. Bartlett. Neural Network Learning: Theoretical Foundations . Cambridge University Press, 1999.
- [6] Pranjal Awasthi, Maria-Florina Balcan, Nika Haghtalab, and Ruth Urner. Efficient learning of linear separators under bounded noise. In Proceedings of The 28th Conference on Learning Theory, COLT 2015 , volume 40 of JMLR Workshop and Conference Proceedings , pages 167190. JMLR.org, 2015.
- [7] Pranjal Awasthi, Maria-Florina Balcan, Nika Haghtalab, and Hongyang Zhang. Learning and 1-bit compressed sensing under asymmetric noise. In Proceedings of the 29th Conference on Learning Theory, COLT 2016 , volume 49 of JMLR Workshop and Conference Proceedings , pages 152-192. JMLR.org, 2016.
- [8] Pranjal Awasthi, Maria-Florina Balcan, and Philip M. Long. The power of localization for efficiently learning linear separators with noise. J. ACM , 63(6):50:1-50:27, 2017.
- [9] Peter L. Bartlett and Shahar Mendelson. Rademacher and Gaussian complexities: Risk bounds and structural results. J. Mach. Learn. Res. , 3:463-482, 2002.
- [10] Avrim Blum, Alan M. Frieze, Ravi Kannan, and Santosh S. Vempala. A polynomial-time algorithm for learning noisy linear threshold functions. Algorithmica , 22(1/2):35-52, 1998.
- [11] Matthew R. Boutell, Jiebo Luo, Xipeng Shen, and Christopher M. Brown. Learning multi-label scene classification. Pattern Recognit. , 37(9):1757-1771, 2004.
- [12] Andreas Buja, Werner Stuetzle, and Yi Shen. Loss functions for binary class probability estimation: Structure and applications. Technical report, University of Pennsylvania, November 2005.
- [13] Tom Bylander. Learning linear threshold functions in the presence of classification noise. In Proceedings of the Seventh Annual ACM Conference on Computational Learning Theory, COLT 1994 , pages 340-347. ACM, 1994.
- [14] Gautam Chandrasekaran, Vasilis Kontonis, Konstantinos Stavropoulos, and Kevin Tian. Learning noisy halfspaces with a margin: Massart is no harder than random. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024 , 2024.

- [15] Sitan Chen, Frederic Koehler, Ankur Moitra, and Morris Yau. Classification under misspecification: Halfspaces, generalized linear models, and evolvability. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020 , 2020.
- [16] Carlo Ciliberto, Lorenzo Rosasco, and Alessandro Rudi. A general framework for consistent structured prediction with implicit loss embeddings. Journal of Machine Learning Research , 21(98):1-67, 2020.
- [17] Edith Cohen. Learning noisy perceptrons by a perceptron in polynomial time. In 38th Annual Symposium on Foundations of Computer Science, FOCS '97, 1997 , pages 514-523. IEEE Computer Society, 1997.
- [18] David Cossock and Tong Zhang. Statistical analysis of Bayes optimal subset ranking. IEEE Transactions on Information Theory , 54(11):5140-5154, 2008.
- [19] Ilias Diakonikolas, Themis Gouleakis, and Christos Tzamos. Distribution-independent PAC learning of halfspaces with Massart noise. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019 , pages 47514762, 2019.
- [20] Ilias Diakonikolas, Mingchen Ma, Lisheng Ren, and Christos Tzamos. Statistical query hardness of multiclass linear classification with random classification noise. arXiv 2502.11413, 2025.
- [21] John Dunagan and Santosh S. Vempala. A simple polynomial-time rescaling algorithm for solving linear programs. In Proceedings of the 36th Annual ACM Symposium on Theory of Computing , pages 315-320. ACM, 2004.
- [22] Vitaly Feldman, Parikshit Gopalan, Subhash Khot, and Ashok Kumar Ponnuswami. New results for learning noisy parities and halfspaces. In 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS 2006) , pages 563-574. IEEE Computer Society, 2006.
- [23] David Haussler. Decision theoretic generalizations of the PAC model for neural net and other learning applications. Inf. Comput. , 100(1):78-150, 1992.
- [24] Kalervo Järvelin and Jaana Kekäläinen. IR evaluation methods for retrieving highly relevant documents. In International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) , 2000.
- [25] Sham M. Kakade, Adam Kalai, Varun Kanade, and Ohad Shamir. Efficient learning of generalized linear and single index models with isotonic regression. In Advances in Neural Information Processing Systems 24: 25th Annual Conference on Neural Information Processing Systems 2011 , pages 927-935, 2011.
- [26] Adam Tauman Kalai and Ravi Sastry. The isotron algorithm: High-dimensional isotonic regression. In COLT 2009 - The 22nd Conference on Learning Theory , 2009.
- [27] Michael J. Kearns. Efficient noise-tolerant learning from statistical queries. J. ACM , 45(6):9831006, 1998.
- [28] Michael J. Kearns and Robert E. Schapire. Efficient distribution-free learning of probabilistic concepts. J. Comput. Syst. Sci. , 48(3):464-497, 1994.
- [29] Michael J. Kearns, Robert E. Schapire, and Linda Sellie. Toward efficient agnostic learning. Mach. Learn. , 17(2-3):115-141, 1994.
- [30] Philip M. Long and Rocco A. Servedio. Random classification noise defeats all convex potential boosters. Mach. Learn. , 78(3):287-304, 2010.
- [31] Pascal Massart and Élodie Nédélec. Risk bounds for statistical learning. Ann. Statist. , 34(5):2326-2366, 2006.
- [32] Andreas Maurer. A vector-contraction inequality for rademacher complexities. arXiv 1605.00251, 2016.

- [33] Mark D. Reid and Robert C. Williamson. Surrogate regret bounds for proper losses. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML 2009 , 2009.
- [34] Mark D. Reid and Robert C. Williamson. Composite binary losses. Journal of Machine Learning Research , 11:2387-2422, 2010.
- [35] Ronald L. Rivest and Robert H. Sloan. A formal model of hierarchical concept learning. Inf. Comput. , 114(1):88-114, 1994.
- [36] Robert H. Sloan. Types of noise in data for concept learning. In Proceedings of the First Annual Workshop on Computational Learning Theory, COLT '88 , pages 91-96. ACM/MIT, 1988.
- [37] Robert H. Sloan. Corrigendum to types of noise in data for concept learning. In Proceedings of the Fifth Annual ACM Conference on Computational Learning Theory, COLT 1992 , page 450. ACM, 1992.
- [38] Leslie G. Valiant. A theory of the learnable. Commun. ACM , 27(11):1134-1142, 1984.
- [39] Elodie Vernet, Robert C. Williamson, and Mark D. Reid. Composite multiclass losses. In Neural Information Processing Systems , 2011.
- [40] Songbai Yan and Chicheng Zhang. Revisiting perceptron: Efficient and label-optimal learning of halfspaces. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017 , pages 1056-1066, 2017.
- [41] Min-Ling Zhang and Zhi-Hua Zhou. A review on multi-label learning algorithms. Knowledge and Data Engineering, IEEE Transactions on , 26:1819-1837, 08 2014.
- [42] Mingyuan Zhang, Jane H. Lee, and Shivani Agarwal. Learning from noisy labels with no change to the training process. In Proceedings of the 38th International Conference on Machine Learning (ICML) , 2021.
- [43] Mingyuan Zhang, Harish Guruprasad Ramaswamy, and Shivani Agarwal. Convex calibrated surrogates for the multi-label f-measure. In Proceedings of the 37th International Conference on Machine Learning (ICML) , 2020.
- [44] Tong Zhang. Some theoretical results concerning the convergence of compositions of regularized linear functions. In Advances in Neural Information Processing Systems 12 , pages 370-378, 1999.
- [45] Yuchen Zhang, Percy Liang, and Moses Charikar. A hitting time analysis of stochastic gradient langevin dynamics. In Proceedings of the 30th Conference on Learning Theory, COLT 2017 , volume 65 of Proceedings of Machine Learning Research , pages 1980-2022. PMLR, 2017.

## Appendix

Organization of the Appendix. Appendix A is a supplement to Section 1 (introduction). Appendix B is a supplement to Section 2 (RSMs and main results). Appendix C is a supplement to Section 3 (binary classification). Appendix D is a supplement to Section 4 (multiclass classification). Appendix E is a supplement to Section 5 (multi-label learning). Appendix F is a supplement to Section 6 (subset ranking). Proofs of all the theorems in the main paper can be found in the relevant sections of this Appendix.

## A Supplement to Section 1 (Introduction)

Here we give details of the assumptions associated with the previously studied intermediate PAC learning models listed in Table 1.

Noisy LTF with random classification noise (RCN): This model assumes that there is a weight vector w ∈ R p and a noise parameter η ∈ (0 , 1 / 2) such that for any instance x , a deterministic binary label is first generated according to the sign of w ⊤ x , and then with probability η the label is flipped to the opposite sign. Equivalently, the model can be viewed as assuming that the conditional label distribution is of the form P ( Y = 1 | X = x ) = (1 -η ) · 1 ( w ⊤ x ≥ 0) + η · 1 ( w ⊤ x &lt; 0) .

Noisy LTF with Massart noise: This model assumes that there is a weight vector w ∈ R p and a "noise upper bound" parameter η ∈ (0 , 1 / 2) such that for any instance x , a deterministic binary label is first generated according to the sign of w ⊤ x , and then with some (unknown) probability η ( x ) ≤ η the label is flipped to the opposite sign. Equivalently, the model can be viewed as assuming that the conditional label distribution satisfies P ( Y = 1 | X = x ) ≥ (1 -η ) if w ⊤ x ≥ 0 and P ( Y = 1 | X = x ) ≤ η if w ⊤ x &lt; 0 .

Generalized linear model (GLM): The (univariate) GLMs considered in [26, 25] are for real-valued regression problems with bounded label spaces Y = ̂ Y ⊆ [0 , 1] , and assume that there is a weight vector w ∈ R p such that E [ Y | X = x ] = θ ( w ⊤ x ) for some known transfer function θ : R → [0 , 1] (it is often assumed that θ satisfies some condition such as a Lipschitz property). These include as a special case binary classification by setting Y = { 0 , 1 } .

Single index model (SIM): The assumption here is of a similar form as that for GLMs above, namely that E [ Y | X = x ] = θ ( w ⊤ x ) for some weight vector w and transfer function θ : R → [0 , 1] ; however unlike GLMs, where θ is assumed to be known, in SIMs, both the weight vector w and the transfer function θ are unknown (it is often assumed that θ satisfies some condition such as a Lipschitz property).

## B Supplement to Section 2 (RSMs and Main Results)

## B.1 Realizable and Agnostic PAC Learning as Special Cases of RSMs

We note here that both realizable and (fully) agnostic PAC learning can be recovered as extreme cases of RSMs. In the following, for a finite set Y and y ∈ Y , e y ∈ { 0 , 1 } Y denotes the unit vector with y -th element equal to 1 and all other elements equal to 0 .

Example 1 ( Realizable PAC learning as RSM ) . Let ̂ Y = Y , and let H ⊂ { h : X→Y} be a hypothesis class/class of prediction models. Let C = ∆ Y , and consider the identity property τ id : ∆ Y → ∆ Y defined as τ id ( p ) = p . Also define the class of functions Q one-hotH as Q one-hotH = { q : X→ ∆ Y ∣ ∣ ∃ h ∈ H s.t. q ( x ) = e h ( x ) ∀ x ∈ X } . Then it can be seen that

<!-- formula-not-decoded -->

where D H -realizable denotes the class of probability distributions D = ( µ, p ) ∈ ∆ X×Y wherein the label Y is (with probability 1 ) given by a deterministic function of the instance X , with the function belonging to H . Therefore, realizable PAC learning w.r.t. H for any loss L ∈ R Y×Y + is equivalent to the RSM learning problem ( L , H , D ( τ id , Q one-hotH ) -RSM ) .

Example 2 ( Agnostic PAC learning as RSM ) . Let H ⊂ { h : X→ ̂ Y} be a class of prediction models. Let C = ∆ Y , and consider the identity property τ id : ∆ Y → ∆ Y defined as τ id ( p ) = p . Define D all = ∆ X×Y and Q all = { q : X→ ∆ Y } . Then it can be seen that

<!-- formula-not-decoded -->

and therefore (fully) agnostic PAC learning w.r.t. H and any loss L ∈ R Y× ̂ Y + is equivalent to the RSM learning problem ( L , H , D ( τ id , Q all ) -RSM ) .

## B.2 Previously Studied Intermediate PAC Learning Models as Special Cases of RSMs

The RSM framework also recovers as special cases all the previously studied intermediate PAC learning models listed in Table 1.

Noisy LTF with RCN: Consider the statistic τ +1 defined in Section 3 and the class of statistic functions Q RCN-linear := { q : X → [0 , 1] | ∃ w ∈ R p , η ∈ (0 , 1 / 2) s.t. q ( x ) = (1 -η ) · 1 ( w ⊤ x ≥ 0)+ η · 1 ( w ⊤ x &lt; 0) } . Then the RSM learning problem ( L 0 -1 , H linear , D ( τ +1 ,Q RCN-linear ) -RSM ) captures exactly the problem of learning linear threshold functions with RCN.

Noisy LTF with Massart noise: Consider again the statistic τ +1 defined in Section 3, and now the class of statistic functions Q Massart-linear := { q : X → [0 , 1] | ∃ w ∈ R p , η ∈ (0 , 1 / 2) s.t. q ( x ) ≥ (1 -η ) if w ⊤ x ≥ 0 and q ( x ) ≤ η if w ⊤ x &lt; 0 } . Then the RSM learning problem ( L 0 -1 , H linear , D ( τ +1 ,Q Massart-linear ) -RSM ) captures exactly the problem of learning linear threshold functions with Massart noise.

GLM: Let θ : R → [0 , 1] be a fixed (known) transfer function. Consider again the statistic τ +1 defined in Section 3, and now the class of statistic functions Q θ GLM := { q : X → [0 , 1] | ∃ w ∈ R p s.t. q ( x ) = θ ( w ⊤ x ) } . Then the RSM learning problem ( L 0 -1 , H linear , D ( τ +1 ,Q θ GLM ) -RSM ) captures exactly the problem of learning GLMs with transfer function θ .

SIM: Consider again the statistic τ +1 defined in Section 3, and now the class of statistic functions Q SIM := { q : X → [0 , 1] | ∃ w ∈ R p , θ : R → [0 , 1] s.t. q ( x ) = θ ( w ⊤ x ) } . Then the RSM learning problem ( L 0 -1 , H linear , D ( τ +1 ,Q SIM ) -RSM ) captures exactly the problem of learning SIMs.

## B.3 Proof of Theorem 1

Recall that we denote

<!-- formula-not-decoded -->

Proof. (of Theorem 1) Let f ∈ F and D = ( µ, p ) ∈ D ( τ , Q ) -RSM. We start by setting up some notation. Define q ∗ ∈ Q as q ∗ ( x ) = τ ( p ( x )) ; define f ∗ ∈ F as f ∗ ( x ) = λ ( q ∗ ( x )) = λ ( τ ( p ( x ))) ; and define q ∈ Q as q ( x ) = λ -1 ( f ( x )) .

Now, we have er L D [ H ] = er L , ∗ D - to see this, note that since ( τ , pred ) is an L -calibrated statisticmapping pair, the Bayes optimal classifier h L , ∗ D satisfies

<!-- formula-not-decoded -->

and so h L , ∗ D ∈ pred ◦ Q = H , which gives er L , ∗ D = er L D [ H ] .

Also note that

<!-- formula-not-decoded -->

## B.4 Proof of Theorem 2

Let us start by stating the following uniform convergence result, which relates the empirical and expected surrogate risks for a bounded surrogate loss ψ acting on vector-valued predictions, uniformly for all functions in a vector-valued function class F , in terms of the d 1 covering numbers of the loss class ψ F . The result follows from a straightforward generalization of standard uniform convergence results for real-valued function classes (such as given in Chapter 17 of [5]) to vector-valued function classes.

Theorem 9 ( Uniform convergence for bounded (surrogate) loss classes in terms of d 1 covering numbers ) . Let X be any instance space and Y be any label space. Let d ∈ Z + and let ψ : Y × R d → R + be a (surrogate) loss function. Let F ⊆ { f : X→ R d } and suppose ψ ( y, f ( x )) ∈ [0 , B ] ∀ x ∈ X , y ∈ Y , f ∈ F for some B &gt; 0 . Then for any m ∈ Z + , any ϵ &gt; 0 , and any D ∈ ∆ X×Y ,

<!-- formula-not-decoded -->

We will now prove the following technical lemma, which upper bounds the d 1 covering numbers of the surrogate loss class ψ F - for surrogate losses ψ that act on vector-valued predictions and that are Lipschitz with respect to the L 1 metric - in terms of the d 1 covering numbers of the real-valued 'projection' function classes F j . This lemma may also be of independent interest.

<!-- formula-not-decoded -->

Lemma 1 ( Bounding d 1 covering numbers of loss function classes ψ F for Lipschitz losses ψ acting on vector-valued predictions ) . Let X and Y be any sets. Let ψ : Y × R d → R + be any (surrogate) loss function that is ρ 1 -Lipschitz in the second argument with respect to the L 1 metric, and F ⊆ { f : X→ R d } be any class of vector-valued functions on X . Let

<!-- formula-not-decoded -->

For each j ∈ [ d ] , let F j = { f j : X→ R | ∃ f ∈ F s.t. f j ( x ) = ( f ( x )) j ∀ x } . Then for any ϵ &gt; 0 and m ∈ Z + ,

<!-- formula-not-decoded -->

Proof. (of Lemma 1) Let ϵ &gt; 0 and m ∈ Z + . Fix any z = ( z 1 , . . . , z m ) = (( x 1 , y 1 ) , . . . , ( x m , y m )) ∈ ( X × Y ) m , and denote x = ( x 1 , . . . , x m ) ∈ X m . For each j ∈ [ d ] , let C j ⊂ R m be an ( ϵ/ρ 1 ) -cover for ( F j ) | x with respect to the d 1 distance. We will construct an ϵ -cover C ⊂ R m for ( ψ F ) | z with respect to the d 1 distance of size | C | ≤ ∏ d j =1 | C j | .

Let f ∈ F , and denote ( ψ f ) | z = ( ψ f ( z 1 ) , . . . , ψ f ( z m )) ∈ R m ; moreover, for each j ∈ [ d ] , let f j : X→ R be defined as f j ( x ) = ( f ( x )) j , and denote ( f j ) | x = ( f j ( x 1 ) , . . . , f j ( x m )) . For each j ∈ [ d ] , let u j = ( u j 1 , . . . , u j m ) ∈ C j be such that d 1 (( f j ) | x , u j ) ≤ ϵ/ ( ρ 1 d ) . For each i ∈ [ m ] , define the d -dimensional vector u i = ( u 1 i , . . . , u d i ) ∈ R d . Now consider the m -dimensional point v := ψ | (( y i , u i )) i m =1 = ( ψ ( y 1 , u 1 ) , . . . , ψ ( y m , u m )) ∈ R m . Then we have

<!-- formula-not-decoded -->

Therefore the set

<!-- formula-not-decoded -->

is an ϵ -cover for ( ψ F ) | z with respect to the d 1 distance. Since | C | ≤ ∏ d j =1 | C j | , the claim follows.

Next, the following result shows that uniform convergence of surrogate risks also implies (surrogate) learning results for approximate empirical risk minimizers. The proof technique is standard (such as given in Chapter 19 of [5]); we include a self-contained proof here for completeness.

Theorem 10 ( Uniform convergence implies bounded (surrogate) regret of approximate (surrogate) risk minimizers ) . Let X be any instance space and Y be any label space. Let d ∈ Z + and let ψ : Y × R d → R + be a (surrogate) loss function. Let F ⊆ { f : X→ R d } . Let m uc : R + × (0 , 1] → Z + be such that for every ϵ &gt; 0 , every δ ∈ (0 , 1] , every m ≥ m uc ( ϵ, δ ) , and every D ∈ ∆ X×Y ,

<!-- formula-not-decoded -->

Let ( α m ) m ∈ Z + be a sequence of positive real numbers such that for every ϵ &gt; 0 , every δ ∈ (0 , 1] , and every m ≥ m uc ( ϵ/ 3 , δ ) , we have α m ≤ ϵ/ 3 . Let A be an approximate surrogate risk minimization

algorithm which, given a training sample S = (( x 1 , y 1 ) , . . . , ( x m , y m )) ∈ ( X × Y ) m of size m , returns an α m -approximate minimizer ̂ f S ∈ F of the empirical ψ -risk 1 m ∑ m i =1 ψ ( y i , f ( x i )) over F , so that 1 m ∑ m i =1 ψ ( y i , ̂ f S ( x i )) ≤ inf f ∈F 1 m ∑ m i =1 ψ ( y i , f ( x i )) + α m . Then for every ϵ &gt; 0 , every δ ∈ (0 , 1] , every m ≥ m uc ( ϵ/ 3 , δ ) , and every D ∈ ∆ X×Y ,

<!-- formula-not-decoded -->

Proof. (of Theorem 10) Let ϵ &gt; 0 , δ ∈ (0 , 1] , and D ∈ ∆ X×Y . Let β &gt; 0 , and let f ∗ ∈ F be such that

<!-- formula-not-decoded -->

Let m ≥ m uc ( ϵ/ 3 , δ ) . Then we have the following with probability at least 1 -δ over the draw of S ∼ D m :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the above holds for all β &gt; 0 , we have that with probability at least 1 -δ over S ∼ D m ,

<!-- formula-not-decoded -->

and therefore,

This proves the claim.

Next, we define the surrogate sample complexity below:

Definition 4 ( Surrogate sample complexity ) . Let C ′ ⊆ R d ′ . Let ψ : Y × C ′ → R + be any surrogate loss, F ⊆ { f : X→C ′ } be a class of surrogate prediction models, and D ⊆ ∆ X×Y be a class of probability distributions. We will say an algorithm A that given a training sample S ∈ ∪ ∞ m =1 ( X × Y ) m returns a surrogate prediction model ̂ f S ∈ F is a learning algorithm for the surrogate loss learning problem ( ψ, F , D ) with surrogate sample complexity function m ψ A : R + × (0 , 1] → Z + if for every ϵ &gt; 0 , δ ∈ (0 , 1] , every distribution D ∈ D , and every m ≥ m ψ A ( ϵ, δ ) ,

<!-- formula-not-decoded -->

and moreover, for every ϵ, δ , m ψ A ( ϵ, δ ) is the smallest integer satisfying the above.

Bringing all the above together, under the conditions of Theorem 2, the following result upper bounds the surrogate sample complexity of an approximate surrogate risk minimization algorithm in terms of the d 1 covering numbers of the real-valued projection classes F j .

Theorem 11 ( Upper bounding surrogate sample complexity of an approximate surrogate risk minimizer via d 1 covering numbers ) . Under the conditions of Theorem 2, the (16 B/ √ m ) -approximate surrogate risk minimization algorithm A is a learning algorithm for the surrogate

learning problem ( ψ, F , ∆ X×Y ) with surrogate sample complexity upper bounded as

<!-- formula-not-decoded -->

In particular, if N 1 ( ϵ, F j , m ) ≤ ϕ ( ϵ, F j ) ∀ j ∈ [ d ] , then we have

<!-- formula-not-decoded -->

Proof. (of Theorem 11) Define m uc : R + × (0 , 1] → Z + as m uc ( ϵ, δ ) := min { m 0 ∈ Z + : m ≥ m 0 = ⇒

<!-- formula-not-decoded -->

Then by Theorem 9 and Lemma 1, we have that for every ϵ &gt; 0 , δ ∈ (0 , 1] , m ≥ m uc ( ϵ, δ ) , and D ∈ ∆ X×Y ,

<!-- formula-not-decoded -->

Next, define a sequence of positive real numbers ( α m ) m ∈ Z + as

<!-- formula-not-decoded -->

Then it can be verified that for every ϵ &gt; 0 , δ ∈ (0 , 1] , and m ≥ m uc ( ϵ/ 3 , δ ) , we have α m ≤ ϵ/ 3 . Therefore, by Theorem 10, an α m -approximate surrogate risk minimization algorithm as described satisfies for every ϵ &gt; 0 , δ ∈ (0 , 1] , m ≥ m uc ( ϵ/ 3 , δ ) , and D ∈ ∆ X×Y ,

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, if N 1 ( ϵ, F j , m ) ≤ ϕ ( ϵ, F j ) ∀ j ∈ [ d ] , this yields the stated bound.

Finally, we will also make use of the following proposition, whose proof follows directly from Theorem 1.

Proposition 12 ( Upper bounding squared τ -estimation error sample complexity and target loss sample complexity in terms of surrogate sample complexity ) . Under the conditions of Theorem 1, any learning algorithm A which given a training sample S , finds a surrogate prediction model ̂ f S ∈ F and produces a τ -statistic estimate ̂ q S ( x ) = λ -1 ( ̂ f S ( x )) and a prediction model ̂ h S ( x ) = decode ( ̂ f S ( x )) , satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. (of Proposition 12) Follows directly from Theorem 1.

The proof of Theorem 2 is now immediate:

Proof. (of Theorem 2) Follows directly from Theorem 11 and Proposition 12.

## B.5 Proof of Theorem 3

Let us start by stating the following uniform convergence result, which relates the empirical and expected surrogate risks for a bounded surrogate loss ψ acting on vector-valued predictions, uniformly for all functions in a vector-valued function class F , in terms of the Rademacher complexity of the loss class ψ F . The proof is standard (via an application of McDiarmid's inequality; see e.g., [9]).

Theorem 13 ( Uniform convergence for bounded (surrogate) loss classes in terms of Rademacher complexity ) . Let X be any instance space and Y be any label space. Let d ∈ Z + and let ψ : Y × R d → R + be a (surrogate) loss function. Let F ⊆ { f : X→ R d } and suppose ψ ( y, f ( x )) ∈ [0 , B ] ∀ x ∈ X , y ∈ Y , f ∈ F for some B &gt; 0 . Then for any m ∈ Z + , any δ ∈ (0 , 1] , and any D ∈ ∆ X×Y , we have with probability at least 1 -δ over the draw of S ∼ D m :

<!-- formula-not-decoded -->

We will make use of the vector-contraction inequality for Rademacher complexities, due to Maurer [32], which upper bounds the Rademacher complexity of the surrogate loss class ψ F - for surrogate losses ψ that act on vector-valued predictions and that are Lipschitz with respect to the Euclidean metric - in terms of the Rademacher complexities of the real-valued 'projection' function classes F j .

Lemma 2 ( Bounding Rademacher complexities of loss function classes ψ F for Lipschitz losses ψ acting on vector-valued predictions [32] ) . Let X and Y be any sets. Let ψ : Y × R d → R + be any (surrogate) loss function that is ρ 2 -Lipschitz in the second argument with respect to the Euclidean metric, and F ⊆ { f : X→ R d } be any class of vector-valued functions on X . Let

<!-- formula-not-decoded -->

For each j ∈ [ d ] , let F j = { f j : X→ R | ∃ f ∈ F s.t. f j ( x ) = ( f ( x )) j ∀ x } . Then for any m ∈ Z + ,

<!-- formula-not-decoded -->

Bringing the above together, under the conditions of Theorem 3, the following result upper bounds the surrogate sample complexity of an approximate surrogate risk minimization algorithm in terms of the Rademacher complexities of the real-valued projection classes F j .

Theorem 14 ( Upper bounding surrogate sample complexity of an approximate surrogate risk minimizer via Rademacher complexities ) . Under the conditions of Theorem 3, the ( B/ (2 √ m )) -approximate surrogate risk minimization algorithm A is a learning algorithm for the surrogate learning problem ( ψ, F , ∆ X×Y ) with surrogate sample complexity upper bounded as

<!-- formula-not-decoded -->

In particular, if ∃ C &gt; 0 such that the Rademacher complexities of the function classes F j have upper bounds of the form R m ( F j ) ≤ C/ √ m ∀ j ∈ [ d ] , then we have

<!-- formula-not-decoded -->

Proof. (of Theorem 14) Define m : R × (0 , 1] → Z as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then by Theorem 13 and Lemma 2, we have that for every ϵ &gt; 0 , δ ∈ (0 , 1] , m ≥ m uc ( ϵ, δ ) , and D ∈ ∆ X×Y ,

<!-- formula-not-decoded -->

Next, define a sequence of positive real numbers ( α m ) m ∈ Z + as

<!-- formula-not-decoded -->

Then it can be verified that for every ϵ &gt; 0 , δ ∈ (0 , 1] , and m ≥ m uc ( ϵ/ 3 , δ ) , we have α m ≤ ϵ/ 3 . Therefore, by Theorem 10, an α m -approximate surrogate risk minimization algorithm as described satisfies for every ϵ &gt; 0 , δ ∈ (0 , 1] , m ≥ m uc ( ϵ/ 3 , δ ) , and D ∈ ∆ X×Y ,

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, if R ( F ) ≤ C/ m ∀ j ∈ [ d ]

m j √ , this yields the stated bound.

The proof of Theorem 3 is now immediate:

Proof. (of Theorem 3) Follows directly from Theorem 14 and Proposition ?? .

## B.6 Proof of Proposition 4

## Proof. (of Proposition 4)

- (i) This is a well-known result (e.g., see [5]).
- (ii) This is a well-known result (e.g., see [44]).
- (iii) This is also a well-known result; we provide a self-contained proof here for completeness. The fact that R m ( F linear ) ≥ 0 follows directly from the fact F linear is closed under negation. For the upper

bound, we have

<!-- formula-not-decoded -->

## C Supplement to Section 3 (Binary Classification)

<!-- formula-not-decoded -->

Proof. (of Lemma 3) Calibration of ( τ +1 , pred 0-1 ) for L 0-1 is immediate, since the Bayes optimal classifier for L 0-1 is given by h L 0-1 , ∗ D ( x ) = sign( p +1 ( x ) -1 2 ) = pred 0-1 ( τ +1 ( x )) . Moreover, for any

p ∈ ∆ {± 1 } , q ∈ [0 , 1] , we have

̸

<!-- formula-not-decoded -->

̸

Proof. (of Theorem 5) Consider the (invertible) logit link function λ : [0 , 1] → R and its inverse λ -1 : R → [0 , 1] given by 7

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that λ -1 here is equivalent to the sigmoid function σ (defined in the theorem statement). We will observe/prove the following:

- (1) H linear = pred 0-1 ◦ Q sigmoid-of-linear ;
- (2) ψ log is a 4 -strongly proper composite surrogate loss for τ +1 with link function λ ;
- (3) sign = pred 0-1 ◦ λ -1 ;
- (4) F linear = λ ◦ Q sigmoid-of-linear ;
- (5) ψ log ( y, f ( x )) ≤ ln(1 + e RW ) ∀ ( x , y ) ∈ X × Y , f ∈ F linear ;
- (6) ψ log is 1 -Lipschitz with respect to the L 1 metric (equivalently Euclidean metric) on R ;
- (7) N 1 ( ϵ, F linear , m ) ≤ ( 1 ϵ ) p ;
- (8) 0 ≤ R m ( F linear ) ≤ RW/ √ m .

The result will then follow from Lemma 3 and Theorem 3.

Parts (1), (3), (4), (5) are immediate from the definitions.

Part (7) is a well-known result (e.g. see [5]).

Part (8) is a well-known result (see Proposition 4).

Part (2): ψ log is known to be a 4 -strongly proper composite loss for the property τ +1 (i.e., for binary class probability estimation) with link function λ as above [3].

Part (6): It is well known (and easy to verify) that the binary logistic loss ψ log is 1 -Lipschitz with respect to the L 1 (equivalently Euclidean) metric on R (to verify this, note that the absolute value of the derivative of ψ log with respect to the second argument is upper bounded by 1 ).

7 Note that in the notation of Definition 3, we use C ′ = R here. Technically, we would also need to extend the definitions of the surrogate loss ψ log (and the mapping decode = sign ) to act on R instead of R : we ignore this issue here for simplicity.

̸

̸

Combining all the above together with Lemma 3 and applying Theorem 3 (with κ = 2 , γ = 4 , ρ 2 = 1 , d = 1 , 0 ≤ R m ( F linear ) ≤ RW/ √ m , and B ≤ ln(1 + e RW ) ) gives the desired result with squared τ +1 estimation error sample complexity

<!-- formula-not-decoded -->

and with target loss sample complexity

<!-- formula-not-decoded -->

## D Supplement to Section 4 (Multiclass Classification)

Lemma 4. ( τ id , pred 0-1 ( n ) ) is an L 0-1 ( n ) -calibrated statistic-mapping pair, with

<!-- formula-not-decoded -->

Proof. (of Lemma 4) Calibration of ( τ id , pred 0-1 ( n ) ) for L 0-1 ( n ) is immediate, since the Bayes optimal classifier for L 0-1 ( n ) is given by h L 0-1 ( n ) , ∗ D ( x ) = pred 0-1 ( n ) ( p ( x )) = pred 0-1 ( n ) ( τ id ( p ( x ))) . Moreover, for any p , q ∈ ∆ n , we have

<!-- formula-not-decoded -->

(since the difference between any two columns of L 0-1 ( n ) has at most two non-zero entries, each with magnitude at most 1 )

Proof. (of Theorem 6) Consider the link function λ : ∆ n → R n with extended inverse λ -1 : R n → ∆ n given by 8

<!-- formula-not-decoded -->

Note that λ -1 here is equivalent to the softmax function σ (defined in the theorem statement). We will observe/prove the following:

- (1) H multiclass-linear = pred 0-1 ( n ) ◦ Q softmax-of-mlinear ;
- (2) ψ mlog is a 1 -strongly proper composite surrogate loss for τ id with link function λ ;
- (3) decode 0-1 ( n ) = pred 0-1 ( n ) ◦ λ -1 ;
- (4) F multiclass-linear = λ ◦ Q softmax-of-mlinear ;
- (5) ψ mlog ( y, f ( x )) ≤ ln( n ) + 2 RW ∀ ( x , y ) ∈ X × Y , f ∈ F multiclass-linear ;
- (6) ψ mlog is 1 -Lipschitz with respect to the L 1 metric and 2 -Lipschitz with respect to the Euclidean metric on R n ;
- (7) N 1 ( ϵ, F y multiclass-linear , m ) ≤ ( 1 ϵ ) p ∀ y ∈ [ n ] ;
- (8) 0 ≤ R m ( F y multiclass-linear ) ≤ RW/ √ m ∀ y ∈ [ n ] .

The result will then follow from Lemma 4, Theorem 2, and Theorem 3.

Parts (1), (3), (4), (5) are immediate from the definitions.

Part (7) is a well-known result (e.g. see [5]).

Part (8) is a well-known result (see Proposition 4).

Part (2): ψ mlog has been shown to be a 1 -strongly proper composite loss for the property τ id (i.e., for multiclass class probability estimation) with link function λ as above [42]. 9

Part (6): To see that ψ mlog is 1 -Lipschitz with respect to the L 1 metric, note that

̸

<!-- formula-not-decoded -->

Thus we have,

<!-- formula-not-decoded -->

8 Note that in the notation of Definition 3, we use C ′ = R n here. Technically, we would also need to extend the definitions of the surrogate loss ψ mlog and the mapping decode 0-1 ( n ) to act on R n instead of R n : we ignore this issue here for simplicity. Also note that here, C = ∆ n is in one-to-one correspondence with only a strict subset of C ′ = R n , and so we use an extended inverse; in particular, we use the partition S = {S p : p ∈ ∆ n } of C ′ = R n given by S p = { u ∈ R n | ∃ c ∈ R s.t. u y = ln( p y ) + c ∀ y } .

9 Note that [42] show this result for a slight variant of ψ mlog that acts on R n -1 rather than R n ; however, essentially the same proof works for the variant we use here as well.

Next, to see that ψ mlog is 2 -Lipschitz with respect to the Euclidean metric, note that

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

(i) Combining all the above together with Lemma 4 and applying Theorem 2 (with κ = √ 2 , γ = 1 , ρ 1 = 1 , d = n , N 1 ( ϵ, F y multiclass-linear , m ) ≤ ( 1 ϵ ) p ∀ y ∈ [ n ] , and B ≤ ln( n ) + 2 RW ) gives the desired result with squared τ id estimation error sample complexity

<!-- formula-not-decoded -->

and with target loss sample complexity

<!-- formula-not-decoded -->

(ii) Next, combining all the above together with Lemma 4 and applying Theorem 3 (with κ = √ 2 , γ = 1 , ρ 2 = 2 , d = n , 0 ≤ R m ( F y multiclass-linear ) ≤ RW/ √ m ∀ y ∈ [ n ] , and B ≤ ln( n ) + 2 RW ) gives the desired result with squared τ id estimation error sample complexity

<!-- formula-not-decoded -->

and with target loss sample complexity

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining the above bounds yields the desired results.

## E Supplement to Section 5 (Multi-Label Learning)

Lemma 5. ( τ marginals , pred Ham ) is an L Ham -calibrated statistic-mapping pair, with

<!-- formula-not-decoded -->

Proof. (of Lemma 5) Calibration of ( τ marginals , pred Ham ) for L Ham is immediate, since the Bayes optimal classifier for L Ham is given by h L Ham , ∗ D ( x ) = pred Ham ( τ marginals ( p ( x ))) . Moreover, for any

<!-- formula-not-decoded -->

̸

p ∈ ∆ { 0 , 1 } s , q ∈ [0 , 1] , we have E Y ∼ p [ L Ham Y , pred Ham ( q ) ] -min ̂ y ∈{ 0 , 1 } s E Y ∼ p [ L Ham Y , ̂ y ] = s ∑ j =1 E Y j [ L 0-1 Y j , ( pred Ham ( q )) j ] -min ̂ y ∈{ 0 , 1 } s s ∑ j =1 E Y j [ L 0-1 Y j , ̂ y j ] ( by linearity of expectation; here L 0-1 ∈ R { 0 , 1 }×{ 0 , 1 } + denotes the binary loss L 0-1 y, ̂ y = 1 ( ̂ y = y )) = s ∑ j =1 E Y j [ L 0-1 Y j , ( pred Ham ( q )) j ] -s ∑ j =1 min ̂ y j ∈{ 0 , 1 } E Y j [ L 0-1 Y j , ̂ y j ] = s ∑ j =1 ( E Y j [ L 0-1 Y j , ( pred Ham ( q )) j ] -min ̂ y j ∈{ 0 , 1 } E Y j [ L 0-1 Y j , ̂ y j ] ) ≤ s ∑ j =1 2 | q j -( τ marginals ( p )) j | ( by well-known result for binary 0-1 loss, as also shown in the proof of Theorem 5 ) = 2 ∥ q -τ marginals ( p ) ∥ 1 ≤ 2 √ s ∥ q -τ marginals ( p ) ∥ 2 .

Proof. (of Theorem 6) Consider the (invertible) link function λ : [0 , 1] s → R s and its inverse λ -1 : R s → [0 , 1] s given by 10

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that each component of λ -1 here is equivalent to the sigmoid function σ (defined in the theorem statement). We will observe/prove the following:

- (1) H sign multilinear = pred Ham ◦ Q sigmoid-of-multilinear ;
- (2) ψ BRlog is a 4 -strongly proper composite surrogate loss for τ marginals with link function λ ;
- (3) decode Ham = pred Ham ◦ λ -1 ;
- (4) F multilinear = λ ◦ Q sigmoid-of-multilinear ;
- (5) ψ BRlog ( y, f ( x )) ≤ s ln(1 + e RW ) ∀ ( x , y ) ∈ X × Y , f ∈ F multilinear ;
- (6) ψ BRlog is 1 -Lipschitz with respect to the L 1 metric and √ s -Lipschitz with respect to the Euclidean metric on R s ;
- (7) N 1 ( ϵ, F j multilinear , m ) ≤ ( 1 ϵ ) p ∀ j ∈ [ s ] ;
- (8) 0 ≤ R m ( F j multilinear ) ≤ RW/ √ m ∀ j ∈ [ s ] .

The result will then follow from Lemma 5, and Theorem 3.

Parts (1), (3), (4), (5) are immediate from the definitions.

10 Note that in the notation of Definition 3, we use C ′ = R s here. Technically, we would also need to extend the definitions of the surrogate loss ψ and the mapping decode to act on R s instead of R s : we ignore this issue here for simplicity.

## Part (7) is a well-known result (e.g. see [5]).

## Part (8) is a well-known result (see Proposition 4).

Part (2): The fact that ψ BRlog is a 4 -strongly proper composite loss for the property τ marginals with link function λ as above follows from 4 -strong proper compositeness of the binary logistic loss ( ψ log as defined in Theorem 5) for binary class probability estimation, applied separately to each component of the loss [3].

Part (6): The fact that ψ BRlog is 1 -Lipschitz with respect to the L 1 metric follows directly from the fact that the binary logistic loss ( ψ log as defined in Theorem 5) is 1 -Lipschitz with respect to the L 1 metric, applied separately to each component of the loss. This also implies it is √ s -Lipschitz with respect to the Euclidean metric.

Combining all the above together with Lemma 5 and applying Theorem 3 (with κ = 2 √ s , γ = 4 , ρ 2 = √ s , d = s , 0 ≤ R m ( F j ) ≤ RW/ √ m ∀ j , and B ≤ s ln(1 + e RW ) ) gives the desired result with squared τ marginals estimation error sample complexity

<!-- formula-not-decoded -->

and with target loss sample complexity

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F Supplement to Section 6 (Subset Ranking)

Lemma 6. ( τ sc-marg-exp , pred DCG ) is an L DCG -calibrated statistic-mapping pair, with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where disc = ( disc (1) , . . . , disc ( s )) ⊤ ∈ [0 , 1] s .

Proof. (of Lemma 6) Calibration of ( τ sc-marg-exp , pred DCG ) for L DCG is immediate, since the Bayes optimal classifier for L DCG is given by h L DCG , ∗ D ( x ) ∈ argsort ( τ sc-marg-exp ( p ( x ))) = pred DCG ( τ sc-marg-exp ( p ( x ))) . In the following, for any q ∈ [0 , 1] s , we will denote

<!-- formula-not-decoded -->

and for any ̂ π ∈ Π s , we will denote

<!-- formula-not-decoded -->

## [CONTINUED ON NEXT PAGE]

Then for any p ∈ ∆ { 0 , 1 ,...,r } s , q ∈ [0 , 1] s , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. (of Theorem 8) Consider the (invertible) link function λ : [0 , 1] s → R s and its inverse λ -1 : R s → [0 , 1] s given by 11

<!-- formula-not-decoded -->

Note that each component of λ -1 here is equivalent to the sigmoid function σ (defined in the theorem statement). We will observe/prove the following:

- (1) H sort multilinear = pred DCG ◦ Q sigmoid-of-multilinear ;
- (2) ψ wlog is a 4 -strongly proper composite surrogate loss for τ sc-marg-exp with link function λ ;
- (3) decode DCG = pred DCG ◦ λ -1 ;
- (4) F multilinear = λ ◦ Q sigmoid-of-multilinear ;

11 Note that in the notation of Definition 3, we use C ′ = R s here. Technically, we would also need to extend the definitions of the surrogate loss ψ and the mapping decode to act on R s instead of R s : we ignore this issue here for simplicity.

- (5) ψ wlog ( y, f ( x )) ≤ s ln(1 + e RW ) ∀ ( x , y ) ∈ X × Y , f ∈ F multilinear ;
- (6) ψ wlog is 1 -Lipschitz with respect to the L 1 metric and √ s -Lipschitz with respect to the Euclidean metric on R s ;

<!-- formula-not-decoded -->

- (8) 0 ≤ R m ( F j multilinear ) ≤ RW/ √ m ∀ j ∈ [ s ] .

The result will then follow from Lemma 6 and Theorem 3.

Parts (1), (3), (4), (5) are immediate from the definitions.

Part (7) is a well-known result (e.g. see [5]).

## Part (8) is a well-known result (see Proposition 4).

Part (2): We show here that ψ wlog is a 4 -strongly proper composite loss for the property τ sc-marg-exp with link function λ as above. In particular, we have:

<!-- formula-not-decoded -->

Y ∼ p [ ψ wlog ( Y , u ) -ψ wlog ( Y , λ ( τ sc-marg-exp ( p )))] = s ∑ j =1 (( E [ Y j ] r ) · ( ln(1 + e -u j ) -ln(1 + e -( λ ( τ sc-marg-exp ( p ))) j ) ) + ( 1 -E [ Y j ] r ) · ( ln(1 + e u j ) -ln(1 + e ( λ ( τ sc-marg-exp ( p ))) j ) ) ) = s ∑ j =1 ( ( τ sc-marg-exp ( p )) j · ( -ln(( λ -1 ( u )) j ) + ln(( τ sc-marg-exp ( p )) j ) ) + ( 1 -( τ sc-marg-exp ( p )) j ) · ( -ln(1 -( λ -1 ( u )) j ) + ln(1 -( τ sc-marg-exp ( p )) j ) ) ) ( by definition of λ and λ -1 ) = s ∑ j =1 ( ( τ sc-marg-exp ( p )) j · ln ( ( τ sc-marg-exp ( p )) j ( λ -1 ( u )) j ) + ( 1 -( τ sc-marg-exp ( p )) j ) · ln ( 1 -( τ sc-marg-exp ( p )) j 1 -( λ -1 ( u )) j )) = s ∑ j =1 D KL ( ( τ sc-marg-exp ( p )) j ∣ ∣ ∣ ∣ ∣ ∣ ( λ -1 ( u )) j ) ( by definition of Kullback-Leibler divergence for binary-valued random variables ) ≥ 1 2 s ∑ j =1 (∣ ∣ ∣ ( λ -1 ( u )) j -( τ sc-marg-exp ( p )) j ∣ ∣ ∣ + ∣ ∣ ∣ (1 -( λ -1 ( u )) j ) -(1 -( τ sc-marg-exp ( p )) j ) ∣ ∣ ∣ ) 2 ( by Pinsker's inequality and properties of the total variation distance ) = 1 2 s ∑ j =1 ( 2 ∣ ∣ ∣ ( λ -1 ( u )) j -( τ sc-marg-exp ( p )) j ∣ ∣ ∣ ) 2 = 2 ∥ λ -1 ( u ) -τ sc-marg-exp ( p ) ∥ 2 1 ≥ 2 ∥ λ -1 ( u ) -τ sc-marg-exp ( p ) ∥ 2 2 .

Thus ψ wlog is a 4 -strongly proper composite loss for the property τ sc-marg-exp with link function λ .

Part (6): It is easy to see that the weighted binary logistic loss ψ wlog , bin : [0 , 1] × R → R + defined as

<!-- formula-not-decoded -->

is 1 -Lipschitz (in particular, the absolute value of the derivative with respect to u is upper bounded by 1 ). The fact that the surrogate loss ψ wlog is 1 -Lipschitz with respect to the L 1 metric then

follows directly from this observation, applied separately to each component of the loss (with weight α = y j /r for component j ). This also implies it is √ s -Lipschitz with respect to the Euclidean metric.

Combining all the above together with Lemma 5 and applying Theorem 3 (with κ = 2 r · ∥ disc ∥ 2 , γ = 4 , ρ 2 = √ s , d = s , 0 ≤ R m ( F j ) ≤ RW/ √ m ∀ j , and B ≤ s ln(1 + e RW ) ) gives the desired result with squared τ sc-marg-exp estimation error sample complexity

<!-- formula-not-decoded -->

and with target loss sample complexity

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All the results are clearly stated and explained in the paper, and proofs are provided in the Appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [NA]

Justification: The paper studies learning problems that in many ways are more general than those studied in previous work. Both sample and computational complexity bounds are provided for the algorithms discussed. We do not foresee any significant limitations.

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

Justification: For all results in the paper, all assumptions are clearly stated and complete proofs are provided in the Appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper does not include experiments.

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

Answer: [NA]

Justification: The paper does not include experiments requiring code.

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

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

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

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have made every effort to conform with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: While this work can lead to a better understanding of various types of learning problems and their possible solutions, the work is largely theoretical/foundational in nature and does not have immediate societal impacts.

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

Justification: The paper poses no such risks (there are no data or models to be released).

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets of the form described.

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

Justification: The paper does not release new assets.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components (nor does the writing of the paper use LLMs in any form).

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
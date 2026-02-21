## On Agnostic PAC Learning in the Small Error Regime

Julian Asilis

USC

asilis@usc.edu

Mikael Møller Høgsgaard

Aarhus University hogsgaard@cs.au.dk

## Abstract

Binary classification in the classic PAC model exhibits a curious phenomenon: Empirical Risk Minimization (ERM) learners are suboptimal in the realizable case yet optimal in the agnostic case. Roughly speaking, this owes itself to the fact that non-realizable distributions D are more difficult to learn than realizable distributions - even when one discounts a learner's error by err( h ∗ D ) , i.e., the error of the best hypothesis in H . Thus, optimal agnostic learners are permitted to incur excess error on (easier-to-learn) distributions D for which τ = err( h ∗ D ) is small. Recent work of Hanneke, Larsen, and Zhivotovskiy (FOCS '24) addresses this shortcoming by including τ itself as a parameter in the agnostic error term. In this more fine-grained model, they demonstrate tightness of the error lower bound τ +Ω( √ τ ( d +log(1 /δ )) / m + d +log(1 /δ ) / m ) in a regime where τ &gt; d/m , and leave open the question of whether there may be a higher lower bound when τ ≈ d/m , with d denoting VC( H ) . In this work, we resolve this question by exhibiting a learner which achieves error c · τ + O ( √ τ ( d +log(1 /δ )) / m + d +log(1 /δ ) / m ) for a constant c ≤ 2 . 1 , matching the lower bound and demonstrating optimality when τ = O ( d/m ) . Further, our learner is computationally efficient and is based upon careful aggregations of ERM classifiers, making progress on two other questions of Hanneke, Larsen, and Zhivotovskiy (FOCS '24). We leave open the interesting question of whether our approach can be refined to lower the constant from 2.1 to 1, which would completely settle the complexity of agnostic learning.

## 1 Introduction

Binary classification stands as perhaps the most fundamental setting in supervised learning, and one of its best-understood. In Valiant's celebrated Probably Approximately Correct (PAC) framework, learning is formalized using a domain X in which unlabeled data points reside, the label space Y = {± 1 } , and a probability distribution D over X × Y . The purpose of a learner A is to receive a training set S = ( x i , y i ) i m =1 of points drawn i.i.d. from D and then emit a hypothesis A ( S ) : X → Y which is unlikely to mispredict the label of a new data point ( x, y ) drawn from D .

̸

At the center of a learning problem is a hypothesis class of functions H ⊆ Y X , with which the learner A must compete. More precisely, A is tasked with emitting a hypothesis f whose error , denoted L D ( f ) := P ( x,y ) ∼D ( f ( x ) = y ) , is nearly as small as that of the best hypothesis in H . We assume for simplicity such a best hypothesis exists and denote it as h ∗ D := arg min h ∈H L D ( h ) or when D is clear from context with h ∗ . Of course, the distribution D is unknown to A , and A is judged on its performance across a broad family of allowable distributions D . In realizable learning, D = {D : inf h ∈H L D ( h ) = 0 } , meaning A is promised that there always exists a ground truth hypothesis h ∗ ∈ H attaining zero error. In this case, for A to compete with the performance of h ∗ D simply requires that it attains error ≤ ϵ for all realizable distributions D (with high probability over the training set S ∼ D m , and using the smallest amount of data m possible). In agnostic learning, D is defined as the set of all probability distributions over X × Y , meaning any data-generating process

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Grigoris Velegkas ∗

Google Research gvelegkas@google.com

is allowed. Thus, A is not equipped with any information concerning D ∈ D at the outset, but in turn it is only required to emit a hypothesis f with L D ( f ) ≤ L D ( h ∗ D ) + ϵ .

Perhaps the most fundamental questions in PAC learning ask: What is the minimum number of samples required to achieve error at most ϵ ? And which learners attain these rates? For realizable binary classification, the optimal sample complexity of learning remained a foremost open question for decades, and was finally resolved by breakthrough work of Hanneke (2016) which built upon that of Simon (2015). Notably, optimal binary classification requires the use of learners which emit hypotheses f outside of the underlying hypothesis class H . Such learners are referred to as being improper . This has the effect of excluding Empirical Risk Minimization (ERM) from contention as an optimal learner for the realizable case. Recall that ERM learners proceed by selecting a hypothesis h S ∈ H incurring the fewest errors on the training set S . (For realizable learning, note that h S will then incur zero error on S .) ERM, then, is an example of a proper learner, which always emits a hypothesis in the underlying class H . Agnostic learning, however, exhibits no such properness barrier for optimal learning. In fact, ERM algorithms are themselves optimal agnostic learners, despite their shortcomings on realizable distributions. This somewhat counter-intuitive behavior owes itself to the fact that agnostic learners are judged on a worst-case basis across all possible distributions D . Simply put, non-realizable distributions are more difficult to learn than realizable distributions even when one discounts the error term by L D ( h ∗ D ) -and thus ERM learners are permitted to incur some unnecessary error on realizable distributions, so long as they remain within the optimal rates induced by the more difficult (non-realizable) distributions.

In light of this behavior, it is natural to ask for a more refined perspective on agnostic learning, in which the error incurred beyond τ := L D ( h ∗ D ) is itself studied as a function of τ . Note that this formalism addresses the previously-described issue by demanding that learners attain superior performance on distributions with smaller values of τ (such as realizable distributions, for which τ = 0 ). Precisely this perspective was recently studied by Hanneke et al. (2024), who established that all proper learners, including ERM, are sub-optimal for τ ∈ [Ω(ln 10 ( m/d )( d +ln(1 /δ )) /m ) , o (1)] , by establishing a new lower bound and an optimal learner in this regime. While settling the sample complexity for a wide range of τ , Hanneke et al. (2024) leave open several interesting questions.

Problem 1.1. Let d = VC( H ) and m = | S | . What is the optimal sample complexity of learning in the regime where τ ≈ d/m ?

Problem 1.2. Are learners based upon majority voting, such as bagging, optimal in the τ -based agnostic learning framework?

Problem 1.3. Can one design a computationally efficient learner which is optimal in the τ -based agnostic learning framework?

The primary focus of our work is to resolve Open Problem 1.1, thereby extending the understanding of optimal error rates across a broader range of τ . As part of our efforts, we also make progress on Open Problems 1.2 and 1.3, as we now describe.

## 1.1 Overview of Main Results

We now present our primary result and a brief overview of our approach.

Theorem 1.4. For any domain X , hypothesis class H of VC dimension d, number of samples m, parameter δ ∈ (0 , 1) , there is an algorithm such that for any distribution D over X × {1 , 1 } it returns a classifier h S : X → {1 , 1 } that, with probability at least 1 -δ , has error bounded by

<!-- formula-not-decoded -->

where τ is the error of the best hypothesis in H .

Notably, Theorem 1.4 settles the sample complexity of agnostic learning in the regime τ = O ( d/m ) , by exhibiting an optimal learner which attains existing lower bounds. Furthermore, it improves upon the learner of Hanneke et al. (2024) when τ = o (ln 5 ( m/d ) · d/m ) . In light of Theorem 1.4, only the polylog range τ ∈ [ ω (1) , o (ln 10 ( m/d ))] · d +ln(1 /δ ) m remains to have its optimal error rates characterized.

Let us briefly describe our approach. Hanneke et al. (2024), building upon Devroye et al. (1996), states that any learner, upon receiving m i.i.d. samples from D , must produce with probability at least

δ a hypothesis incurring error τ +Ω( √ τ ( d +ln(1 /δ )) /m +( d +ln(1 /δ )) /m ) for worst-case D . Our first elementary observation is that, due to this lower bound, by designing a learner whose error is bounded by c · τ + O ( √ τ ( d +ln(1 /δ )) /m +( d +ln(1 /δ )) /m ) for some numerical constant c ≥ 1 , we can get a tight bound in the regime τ = O ( d / m ) , thus resolving Open Problem 1.1. Furthermore, to make progress across the full regime τ ∈ [0 , 1 / 2 ] , it is crucial to obtain a constant c whose value is close to 1. Motivated by this observation, our primary result gives an algorithm that proceeds by taking majority votes over ERMs trained on carefully crafted subsamples of the training set S , and which achieves error 2 . 1 · τ + O ( √ τ ( d +ln(1 /δ )) /m +( d +ln(1 /δ )) /m ) . Notably, this brings us within striking distance of the constant c = 1 -weleave open the intriguing question of whether our technique can be refined to lower c to 1, which would completely settle the complexity of agnostic learning.

Our approach is inspired by Hanneke (2016), but introduces several new algorithmic components and ideas in the analysis. More concretely, we first modify Hanneke's sample splitting scheme, and then randomly select a small fraction of the resulting subsamples on which to run ERM , rather than running ERM on all such subsamples. This improves the ERM -oracle efficiency of the algorithm. In order to decrease the value of the constant multiplying τ , our main insight is to run two independent copies of the above classifier, as well as one ERM that is trained on elements coming from a certain 'region of disagreement' of the previous two classifiers. For any test point x, if both of the voting classifiers agree on a label y and have 'confidence' in their vote, we output y ; otherwise we output the prediction of the ERM . We hope that this idea of breaking ties between voting classifiers using ERM s that are trained on their region of disagreement can find further applications. Our resulting algorithm employs a voting scheme at its center, making progress on Open Problem 1.2, and is computationally efficient with respect to ERM oracle calls, making considerable progress on Open Problem 1.3. Finally, we combine this algorithm with the learner of Hanneke et al. (2024) to obtain a 'best-of-both-worlds' result. An overview of all the steps is presented in the start of Section 3.2.

## 1.2 Related Work

The PAC learning framework for statistical learning theory dates to the seminal work of Valiant (1984), with roots in prior work of Vapnik and Chervonenkis (Vapnik and Chervonenkis, 1964, 1974). In binary classification, finiteness of the VC dimension was first shown to characterize learnability by Blumer et al. (1989). Tight lower bounds on the sample complexity of learning VC classes in the realizable case were established by Ehrenfeucht et al. (1989) and matched by upper bounds of Hanneke (2016), building upon work of Simon (2015). Subsequent works have established different optimal PAC learners for the realizable setting (Aden-Ali et al., 2023, 2024; Høgsgaard, 2025; Larsen, 2023). For agnostic learning in the standard PAC framework, ERM is known to achieve sample complexity matching existing lower bounds (Anthony and Bartlett, 2009; Boucheron et al., 2005; Haussler, 1992). As described, we direct our attention to a more fine-grained view of agnostic learning, in which the error incurred by a learner above the best-in-class hypothesis h ∗ D is itself studied as a function of τ = L D ( h ∗ D ) . Bounds employing τ in the error term are sometimes referred to as first-order bounds and have been previously analyzed in fields such as online learning (Maurer and Pontil, 2009; Wagenmaker et al., 2022). Hanneke et al. (2024) appear to be the first to consider τ -optimal-dependence for upper bounds in PAC learning; we adopt their perspective in this work.

## 2 Preliminaries

Notation For a natural number m ∈ N , [ m ] denotes the set { 1 , . . . , m } . Random variables are written in bold face (e.g., x ) and their realizations in non-bold type face (e.g., x ). For a set Z , Z ∗ denotes the set of all finite sequences in Z , i.e., Z ∗ = ⋃ i ∈ N Z i . For a sequence S of length m and indices i ≤ j ∈ [ m ] , S [ i : j ] denotes the smallest contiguous subsequence of S which includes both its i th and j th entries. Furthermore, we employ 1-indexing for sequences. For S = ( a, b, c ) , for instance, S [1 : 2] = ( a, b ) . The symbol ⊔ is used to denote concatenation of sequences, as in ( a, b ) ⊔ ( c, d ) = ( a, b, c, d ) . When S, S ′ ∈ Z ∗ are sequences in Z and each element s ∈ S appears in S ′ no less frequently than in S , we write S ⊑ S ′ . For a sequence S and a set A we denote by S ⊓ A the longest subsequence of S that consists solely of elements of A . If S is a finite set, then E x ∼ S [ f ( x )] denotes the expected value of f over a uniformly random draw of x ∈ S .

Learning Theory Let us briefly recall the standard language of supervised learning. Unlabeled data points are drawn from a domain X , which we permit to be arbitrary throughout the paper. We

̸

̸

̸

study binary classification, in which data points are labeled by one of two labels in the label set Y = {± 1 } . A function h : X → Y is referred to as a hypothesis or classifier , and a collection of such functions H ⊆ Y X is a hypothesis class . Throughout the paper, we employ the 0-1 loss function ℓ 0 -1 : Y×Y → R ≥ 0 defined by ℓ 0 -1 ( y, y ′ ) = ✶ [ y = y ′ ] . A training set is a sequence of labeled data points S = ( ( x 1 , y 1 ) , . . . , ( x m , y m ) ) ∈ ( X × Y ) ∗ . A learner is a function which receives training sets and emits hypotheses, e.g., A : ( X ×Y ) ∗ →Y X . The purpose of a learner is to emit a hypothesis h which attains low error , or true error , with respect to an unknown probability distribution D over X × Y . That is, L D ( h ) = E ( x,y ) ∼D [ ✶ [ h ( x ) = y ]] . A natural proxy for the true error of h is its empirical error on a training set S = ( x i , y i ) i ∈ [ m ] , denoted L S ( h ) = E ( x,y ) ∼ S [ ✶ [ h ( x ) = y ]] . If A is a learner for H with the property that A ( S ) ∈ arg min h ∈H L S ( h ) for all training sets S , then A is said to be an empirical risk minimization (ERM) learner for H . Throughout the paper we will use A to denote an arbitrary ERM learner.

## 3 Proof Sketch

We now provide a detailed explanation of our approach and a comprehensive sketch of the proof. We divide our discussion into two parts. In the first, we present a simple approach that achieves an error bound of 15 τ + O ( √ τ ( d +ln(1 /δ )) / m + ( d +ln(1 /δ )) / m ) . Recall that this resolves the optimal sample complexity for the regime τ ≈ d / m . In the second part, we describe several modifications to the algorithm and new ideas in its analysis which drive the error down to 2 . 1 τ + O ( √ τ ( d +ln(1 /δ )) / m + ( d +ln(1 /δ )) / m ) .

## 3.1 First Approach: Multiplicative Constant 15

A crucial component of our algorithm is a scheme S ′ for recursively splitting the input training sequence S into subsequences, which adapts Hanneke's recursive splitting algorithm (Hanneke, 2016) and is formalized in Algorithm 1 and depicted in Figure 1. The algorithm takes two training sequences as input: S , the active set, and T , the union of elements chosen in previous recursive calls.

## Algorithm 1: Splitting algorithm S ′

```
Input: Training sequences S, T ∈ ( X × Y ) ∗ , where | S | = 3 k for k ∈ N . Output: Family of training sequences. if k ≥ 6 then Partition S into S 1 , S 2 , S 3 , with S i being the ( i -1) | S | / 3 + 1 to the i | S | / 3 training examples of S . Set for each i halloooooooooooooo S i, ⊔ = S i [1 : 3 k -4 ] , S i, ⊓ = S i [3 k -4 +1 : 3 k -1 ] , return [ S ′ ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) , S ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ) , S ′ ( S 3 , ⊔ ; S 3 , ⊓ ⊔ T )] else return S ⊔ T
```

Figure 1: The splitting process of algorithm S ′ . The active set S is split into three disjoint sets S 1 , S 2 , S 3 . Each of these is then split into a new active set (green) and a set of previously recursed-on samples (grey), which are passed down to subsequent recursive calls.

<!-- image -->

In comparison to Hanneke's scheme, we have made two modifications which allow us to control the constant multiplying τ , as we will explain shortly. First, we create disjoint splits of the data, in contrast to the overlapping subsets employed by Hanneke. Second, we include more elements in the sets S i, ⊓ , which are included in all subsequent training subsequences, than in S i, ⊔ , the set on which we make a recursive call.

̸

̸

Let us introduce additional notation. We denote s ′ ⊓ := | S | / | S 1 , ⊓ | , for S 1 , ⊓ as defined in Algorithm 1. For A an ERM learner and S ′ ( S ; T ) the splitting scheme of Algorithm 1, we set A ′ ( S ; T ) := ( A ( S ′ )) S ′ ∈S ′ ( S ; T ) , thought of as a multiset. For an example ( x, y ) , we define avg = ( A ′ ( S ; T ))( x, y ) = ∑ h ∈A ′ ( S ; T ) ✶ { h ( x ) = y } / |A ′ ( S ; T ) | , i.e., the fraction of incorrect hypotheses in A ′ ( S ; T ) for ( x, y ) . For a natural number t ∈ N , we let ˆ A ′ t ( S ; T ) , be the random multiset of t hypotheses drawn independently and uniformly at random from A ′ ( S ; T ) . In a slight overload of notation, we use t (i.e., t in bold face) to denote the randomness used to draw the t hypotheses from A ′ ( S ; T ) . Intuitively, one can think of ˆ A ′ t as a bagging algorithm where the subsampled training sequences are restricted to subsets of S ′ ( S ; T ) . Similarly, we define avg = ( ˆ A ′ t ( S ; T ))( x, y ) = ∑ h ∈ ˆ A ′ t ( S ; T ) ✶ { h ( x ) = y } / | ˆ A ′ t ( S ; T ) | . For a distribution D over X × {1 , 1 } , training sequences S, T ∈ ( X × {1 , 1 } ) ∗ , and α ∈ [0 , 1] , we let L α D ( A ′ ( S ; T )) = P ( x , y ) ∼D [avg = ( A ′ ( S ; T ))( x , y ) ≥ α ] , i.e., the probability that at least an α -fraction of the hypotheses in A ′ ( S ; T ) err on a new example drawn from D . We will overload the notation L D when considering majorities and write L D ( A ′ ( S ; T )) = L 0 . 5 D ( A ′ ( S ; T )) -the probability of an equal-weighted majority vote fails. Similarly, we define L α D ( ˆ A ′ t ( S ; T )) = P ( x , y ) ∼D [avg = ( ˆ A ′ t ( S ; T ))( x , y ) ≥ α ] and L D ( ˆ A ′ t ( S ; T )) = L 0 . 5 D ( ˆ A ′ t ( S ; T )) . Finally, we let ˆ A ′ t ( S ) = ˆ A ′ t ( S ; ∅ ) , and ˆ A ′ t ( S )( x ) = sign( ∑ h ∈ ˆ A ′ t ( S ; T ) h ( x )) .

̸

We now describe our first approach, which we break into three steps. The first step relates the error of L D ( ˆ A ′ t ( S )) to L 0 . 49 D ( A ′ ( S , ∅ )) , while the second and third steps bound the error of L 0 . 49 D ( A ′ ( S , ∅ )) . The first step borrows ideas from Larsen (2023) and the last step from Hanneke (2016), but there are several technical bottlenecks in the analysis that do not appear in these works, as they consider the realizable setting, for which τ = 0 .

Relating the error of ˆ A ′ t ( S ) to A ′ ( S , ∅ ) : We will demonstrate how to bound the error of the random classifier ˆ A ′ t ( S ) using the error of A ′ ( S , ∅ ) . First, let S ∼ D m and assume we have shown

<!-- formula-not-decoded -->

̸

with probability at least 1 -δ/ 2 . Consider the event E = { ( x, y ) : avg = ( A ′ ( S ; ∅ ))( x , y ) ≥ 49 / 100 } . By the law of total expectation, we bound the error of ˆ A ′ t ( S ) as L D ( ˆ A ′ t ( S )) ≤ P ( x , y ) ∼D ( E ) + E ( x , y ) ∼D [ ✶ { ˆ A ′ t ( S )( x ) = y } | ¯ E ] . We see that the first term on the right-hand side is L 0 . 49 D ( A ( S ; ∅ )) , which we have assumed for the moment can be bounded by Equation (1). We now argue that the second term can be bounded by O (( d + ln (1 /δ )) /m ) . Note that under the event ¯ E = { ( x, y ) : avg = ( A ( S ; ∅ ))( x, y ) &lt; 49 / 100 } , strictly more than half of the hypotheses in A ( S ; ∅ ) -from which the hypotheses of ˆ A ′ t ( S ) are drawn - are correct. Using this, combined with Hoeffding's inequality and switching the order of expectation (as t and ( x , y ) are independent), we get that E t [ E ( x , y ) ∼D [ ✶ { ˆ A ′ t ( S )( x ) = y } | ¯ E ]] ≤ exp( -Θ( t )) . Setting t = Θ(ln( m/ ( δ (ln (1 /δ ) + d )))) then gives E t [ E ( x , y ) ∼D [ ✶ { ˆ A ′ t ( S )( x ) = y } | ¯ E ]] = O (( δ (ln (1 /δ ) + d )) /m ) . By an application of Markov's inequality, this implies with probability at least 1 -δ/ 2 over the draws of hypotheses in ˆ A ′ t ( S ) that E ( x , y ) ∼D [ ✶ { ˆ A ′ t ( S )( x ) = y } | ¯ E ] = O (( d +ln(1 /δ )) /m ) . This bounds the second term in the error decomposition of L D ( ˆ A ′ t ( S ; ∅ )) and gives the claimed bound on ˆ A ′ t ( S ) .

̸

̸

Bounding the error of L 0 . 49 D ( A ′ ( S , ∅ )) : We now give the proof sketch of Equation (1). To this end, for a training sequence S ′ and hypothesis h we define ∑ = ( h, S ′ ) = ∑ ( x,y ) ∈ S ′ ✶ { h ( x ) = y } . Assume for the moment that for h ⋆ ∈ arg min L D ( h ) we have demonstrated that with probability at least 1 -δ/ 4 over S , and for some numerical constants c b , c c ,

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

In order to exploit Equation (2), we first observe that for each i ∈ { 1 , 2 , 3 } , we have that

̸

<!-- formula-not-decoded -->

̸

̸

as S ′ ⊑ S i and | S i | s ′ ⊓ /m = | S i | / | S i, ⊓ | = 1 / (1 -1 27 ) . We briefly remark that the multiplication with | S i | / | S i, ⊓ | is a source of a multiplicative factor on τ . However, it can be made arbitrarily close to 1 by making the split between S i, ⊔ and S i, ⊓ more imbalanced in the direction of S 1 , ⊓ , at the cost of larger constants c b , c c . Now, by an application of Bernstein's inequality on the hypothesis h ⋆ (Lemma B.3) for each i ∈ { 1 , 2 , 3 } , we have that 15 L S i ( h ⋆ ) is at most 15( τ + O ( √ τ ln (1 /δ ) /m +ln(1 /δ ) /m )) , with probability at least 1 -δ/ 4 over S i . Invoking the union bound and using that max S ′ ∈S ′ ( S , ) = max i ∈{ 1 , 2 , 3 } max S ′ ∈S ′ ( S i, ⊔ , S i, ⊓ ⊔∅ ) , we then have that max S ′ ∈S ′ ( S , ∅ ) 14Σ = ( h ⋆ , S ′ ) / ( m/s ′ ⊓ ) = 15( τ + O ( √ τ ln (1 /δ ) /m +ln(1 /δ ) /m )) with probability at least 1 -3 δ/ 4 over S . Finally, union bounding with the event in Equation (2) yields that both events hold with probability at least 1 -δ over S , from which the error bound of Equation (1) follows by inserting the bound of max S ′ ∈S ′ ( S , ∅ ) 14Σ = ( h ⋆ , S ′ ) / ( m/s ′ ⊓ ) into Equation (2).

̸

Relating the error of L 0 . 49 D ( A ′ ( S , ∅ )) to the empirical error of h ⋆ : Here we draw inspiration from the seminal ideas of Hanneke (2016) by analyzing the learner's loss recursively. However, certain aspects of our analysis will diverge from Hanneke's approach, which is tailored to the realizable setting. As in Hanneke (2016), we have the splitting scheme S ′ ( · , · ) receive two arguments, the second of which can be thought of as the concatenation of all previous training sequences created by the recursive calls of Algorithm 1. As we are in the agnostic case, this second argument can be any training sequence T ∈ ( X × {1 , 1 } ) ∗ . We will first demonstrate that with probability at least 1 -δ over S ,

̸

<!-- formula-not-decoded -->

̸

Setting T = ∅ and rescaling δ then yields Equation (2).

̸

̸

The first step of our analysis is to relate the error of L 0 . 49 D ( A ′ ( S ; T )) to that of the previous calls A ′ ( S i, ⊔ ; S i, ⊓ ⊔ T ) . First note that for any ( x, y ) such that avg = ( A ′ ( S ; T ))( x, y ) ≥ 49 / 100 , at least one of the calls A ′ ( S i, ⊔ ; S i, ⊓ ⊔ T ) for i ∈ { 1 , 2 , 3 } must have avg = ( A ′ ( S i, ⊔ ; S i, ⊓ ⊔ T ))( x, y ) ≥ 49 / 100 . Further, there must be at least ( 49 100 -1 3 ) 3 2 = 47 200 of the hypotheses in ∪ j ∈{ 1 , 2 , 3 }\ i A ′ ( S j, ⊔ ; S j, ⊓ ⊔ T ) that also fail on ( x, y ) . Using these observations, we have that if we draw a random index I ∈ { 1 , 2 , 3 } and a random hypothesis ˆ h ∈ ⊔ j ∈{ 1 , 2 , 3 }\ I A ′ ( S j, ⊔ ; S j, ⊓ ⊔ T ) , then for ( x, y ) such that avg = ( A ′ ( S ; T ))( x, y ) ≥ 49 / 100 it holds that P I , ˆ h [avg = ( A ( S I , ⊔ ; S I , ⊓ ⊔ T ))( x, y ) ≥ 49 / 100 , ˆ h ( x ) = y ] ≥ 1 3 47 200 ≥ 1 13 . Hence, we have that 13 E I , ˆ h [ P ( x , y ) ∼D [avg = ( A ( S I , ⊔ ; S I , ⊓ ⊔ T ))( x , y ) ≥ 49 / 100 , ˆ h ( x ) = y ]] ≥ L 0 . 49 D ( A ′ ( S ; T )) . Furthermore, by the definition of I ∈ { 1 , 2 , 3 } and ˆ h being a random hypothesis from ⊔ j ∈{ 1 , 2 , 3 }\ I A ′ ( S j, ⊔ ; S j, ⊓ ⊔ T ) , we conclude that

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

We will demonstrate Equation (4) by induction on | S | = 3 k . By setting c b and c c sufficiently large, we can assume the claim holds for k &lt; 9 . By Equation (5) and a union bound, it suffices to bound each of the six terms for different combinations of i, j by Equation (4) with probability at least 1 -δ/ 6 . Using symmetry, we can consider the case j = 1 and i = 2 . Note that 13 max h ∈A ′ ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) P ( x , y ) ∼D [avg = ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( x , y ) ≥ 49 / 100 , h ( x ) = y ] equals

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

Supposing that L 0 . 49 D ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T )) is upper bound by

̸

<!-- formula-not-decoded -->

then Equation (6) is bounded as in Equation (4) and we are done. Using that | S 2 , ⊔ | = m/ 3 4 , the inductive hypothesis gives that with probability at least 1 -δ/ 16 over S 2 that L 0 . 49 D ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T )) is at most

̸

<!-- formula-not-decoded -->

̸

Where we have used that S ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ) ⊆ S ( S ; T ) . Thus, it suffices to consider the case in which L 0 . 49 D ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T )) lies between the expressions in Equation (7) and Equation (8).

̸

Let A = { ( x, y ) | avg = ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( x, y ) ≥ 49 / 100 } and N = | S 1 , ⊓ ⊓ A | . Using a Chernoff bound, one has that N ≥ 13 m/ (14 s ′ ⊓ ) L 0 . 49 D ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T )) with probability at least 1 -δ/ 16 . As S 1 , ⊓ ⊓ A ∼ D ( · | avg = ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( x , y ) ≥ 49 / 100 ) , uniform convergence over H (see Lemma B.6) yields that with probability 1 -δ/ 16 over S 1 , ⊓ ⊓ A , Equation (6) is bounded by

̸

<!-- formula-not-decoded -->

Where C &gt; 0 is a universal constant. We now bound each term in Equation (9), considered after multiplying by L 0 . 49 D . Notably, this component of the analysis diverges considerably from that of Hanneke (2016) for the realizable case, in which one is assured that L S 1 , ⊓ ⊓ A ( h ) = 0 . For the first, we use the definitions of N and L S 1 , ⊓ ⊓ A (and abbreviate h ∈ A ′ ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) as h ∈ A ′ ) to observe that

̸

Further, using that for any S ′ ∈ S ′ ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) we have S 1 , ⊓ ⊓ A ⊆ S ′ , and h = A ( S ′ ) is a minimizer of ∑ ( x,y ) ∈ S ′ ✶ { h ( x ) = y } , combined with that S ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) ⊑ S ( S ; T ) , one can invoke Equation (10) to conclude that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

We note that the above step introduces a factor of 14 13 on τ , which can be brought arbitrarily close to 1 by using a tighter Chernoff bound for the size of N , at the cost of larger constants c c and c b in Equation (4). For the second term √ C ( d +2ln(48 /δ )) / N L 0 . 49 D ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T )) of Equation (9), we again use that N ≥ (13 m ) / (14 s ′ ⊓ ) L 0 . 49 D ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T )) to bound it from above by √ (14 C ( d +2ln(48 /δ )) L 0 . 49 D ( A ′ ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))) / (13 m/s ′ ⊓ ) . As we are considering the case in which L 0 . 49 D is bounded by the expression of Equation (8), one can show that the following upper bound holds for sufficiently large c b and c c :

̸

<!-- formula-not-decoded -->

Combining these bounds on each term of Equation (9) yields an upper bound of the form

̸

<!-- formula-not-decoded -->

̸

completing the inductive step and establishing Equation (4), as desired.

̸

## 3.2 Improved Approach: Multiplicative Constant 2.1

Roughly speaking, we apply two modifications to the previous approach in order to achieve a considerably smaller constant factor on τ . First, we use a different splitting scheme (Algorithm 2) which recursively splits the dataset into 27 subsamples, rather than 3. Second, rather than taking a majority vote over ERM learners on a single instance of the splitting algorithm, we split the training set into three parts, run two independent instances of the voting classifier on the first two parts, and one instance of ERM on a subsample of the third part. The subsample we train the ERM on is carefully chosen and depends on a certain 'region of disagreement' of the two voting classifiers. The final prediction at a given datapoint is as follows: If both voting classifiers agree on the predicted label with a certain notion of 'margin,' then we output that label, otherwise we output the prediction of the tiebreaker ERM . Lastly, we combine our approach with that of Hanneke et al. (2024) to achieve a best-of-both-worlds bound. We summarize our approach in the following point form and Figure 2.

1. Split the training set S into three equally-sized parts S 1 , S 2 , S 3 .
2. Split S 1 into three equally-sized parts, S 1 , 1 , S 1 , 2 , S 1 , 3 .
3. 2.1 Run the splitting scheme of Algorithm 2 on ( S 1 , 1 , ∅ ) and ( S 1 , 2 , ∅ ) . Let S 1 , S 2 be the resulting sets of training subsequences.
4. 2.2 Sample t = O ( ln ( m/ ( δ ( d +ln(1 /δ )))) ) sequences from each of S 1 and S 2 uniformly at random. Let ˆ S 1 , ˆ S 2 be the resulting collections of sequences.
5. 2.3 Train ERM learners on each sequence appearing in ˆ S 1 , ˆ S 2 . Denote by ˆ A t 1 , ˆ A t 2 the resulting collections of classifiers produced by these ERM s, respectively.

̸

- 2.4 Define the set S = 3 as follows: for any ( x, y ) ∈ S 1 , 3 if at least an 11 / 243 -fraction of the classifiers in ˆ A t 1 do not predict the label y for x , or likewise for ˆ A t 2 , then ( x, y ) ∈ S = 3 . Train an ERM on S = 3 , producing the classifier h tie .

̸

- 2.5 Let ˜ A 1 be the classifier which acts as follows on any given x : if a 232 / 243 -fraction of the classifiers of both ˆ A t 1 and ˆ A t 2 agree on a label y for x , then ˜ A 1 predicts y. Otherwise, it predicts h tie ( x ) .
3. Use S 2 to train the algorithm of Hanneke et al. (2024). Let ˜ A 2 be the resulting classifier.
4. Output the classifier among ˜ A 1 and ˜ A 2 which attains superior performance on S 3 .

̸

Figure 2: A flowchart of the final algorithm. The initial sample S is split into three parts. S 1 is used to construct our tie-breaking classifier ˜ A 1 . S 2 is used to train the algorithm of Hanneke et al. (2024), yielding ˜ A 2 . Finally, S 3 is used as a hold-out set to select the better of the two classifiers.

<!-- image -->

̸

Remark 3.1. Intuitively, h tie can be seen as 'stabilizing' the predictions produced by the classifiers in the collections ˆ A t 1 and ˆ A t 2 when at least one collection is judged to have low confidence. However, strictly speaking, we cannot demonstrate that the learner ˜ A 1 described by Step (2.) is stable in the sense of Bousquet and Elisseeff (2002). In particular, the classifiers in each of ˆ A t 1 and ˆ A t 2 are trained on overlapping samples, meaning that many classifiers could change by altering even one example in S 1 , 1 or S 1 , 2 , thereby altering S = 3 as well.

We now derive the generalization error for the steps under points 2 . 1 -2 . 5 . The derivation is presented in a bottom-up fashion, starting by examining the failure modes of the approach in Section 3.1 (which leads to the bound with 15 τ ), and amending these to obtain the approach of 2 . 1 -2 . 5 bounded by 2 . 1 τ . To this end, let us recall the three sources leading to multiplicative factors on τ in Section 3.1. First is

̸

the balance between the splits of S i in Algorithm 1, i.e. | S i | / | S i, ⊓ | = 1 / (1 -1 / 27 ) in Equation (3). Recall that this constant can be driven down to 1 by considering even more imbalanced splits of the dataset, at the expense of larger constants multiplying the remaining terms of the bound. Second is the error arising from Equation (11), which controls the size of N via a Chernoff bound. This can similarly be driven arbitrarily close to 1. Hence, the primary multiplicative overhead results from the third source: the argument in Equation (5) relating the error of A ′ ( S , T ) to that of its recursive calls.

̸

The constant arises from relating L 0 . 49 D ( A ′ ( S ; T )) to one of its previous iterates erring on a 49 / 100 -fraction of its classifiers, and one hypothesis from the remaining iterates also erring. Recall that to get this bound we first observe that, with probability at least 1 / 3 over a randomly drawn index I ∼ { 1 , 2 , 3 } , avg = ( A ′ ( S I , ⊔ ; S I , ⊓ ⊓ T ))( x , y ) ≥ 0 . 49 and a random hypothesis ˆ h ∈ ⊔ j ∈{ 1 , 2 , 3 }\ I A ′ ( S j, ⊔ ; S j, ⊓ ⊔ T ) , would, with probability at least 3 / 2 ( 49 / 100 -1 / 3 ) , make an error. Thus, we get the inequality in Equation (5), which multiplies τ by ≈ 13 . To decrease this constant we first make the following observation: assume that we have two voting classifiers A ′ ( S 1 ; ∅ ) , A ′ ( S 2 ; ∅ ) and an ( x, y ) such that avg = ( A ′ ( S 1 ; ∅ ))( x, y ) ≥ 232 / 243 and avg = ( A ′ ( S 2 ; ∅ ))( x, y ) ≥ 232 / 243 . Using a similar analysis as in Equation (5), we have

̸

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

Following a similar approach to that used to obtain Equation (1), we can bound Equation (14) by 1 . 1 τ + O ( √ τ ( d +ln(1 /δ )) /m +( d +ln(1 /δ )) /m ) . Thus, if we could bound the error using the event in Equation (13), we could obtain a better bound for the overall error of our algorithm.

̸

To this end, consider a third independent training sequence S 3 and let S = 3 be the training examples ( x, y ) in S 3 such that avg = ( A ′ ( S 1 ; ∅ ))( x, y ) ≥ 11 / 243 or avg = ( A ′ ( S 2 ; ∅ ))( x, y ) ≥ 11 / 243 . Let h tie = A ( S = 3 ) , i.e. h tie is the output of ERM-learner trained on S = 3 . We now introduce our tie-breaking idea. For a point x we let Tie 11 / 243 ( A ′ ( S 1 ; ∅ ) , A ′ ( S 2 ; ∅ ); h tie ) ( x ) = y , where y ∈ {± 1 } is the unique number for which ∑ h ∈A ′ ( S 1 ; ∅ ) ✶ { h ( x ) = y } / |A ′ ( S 1 ; ∅ ) | ≥ 232 / 243 and ∑ h ∈A ′ ( S 2 ; ∅ ) ✶ { h ( x ) = y } / |A ′ ( S 2 ; ∅ ) | ≥ 232 / 243 . If such a number does not exist, we set y = h tie ( x ) . Thus, this classifier errs on ( x, y ) if 1) avg = ( A ′ ( S 1 ; ∅ ))( x, y ) ≥ 232 / 243 and avg = ( A ′ ( S 2 ; ∅ ))( x, y ) ≥ 232 / 243 , or 2) for all y ′ ∈ {± 1 } , ∑ h ∈A ′ ( S 1 ; ∅ ) ✶ { h ( x ) = y ′ } / |A ′ ( S 1 ; ∅ ) | &lt; 232 / 243 or ∑ h ∈A ′ ( S 2 ; ∅ ) ✶ { h ( x ) = y ′ } / |A ′ ( S 2 ; ∅ ) | &lt; 232 / 243 and h tie ( x ) = y. Using the definition of condition 2), we have that

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

As observed below Equation (13), the first term can be bounded by 1 . 1 τ + O ( √ τ ( d +ln(1 /δ )) /m + ( d + ln (1 /δ )) /m ) . Thus, if we could bound the second term by τ + O ( √ τ ( d +ln(1 /δ )) /m + ( d + ln (1 /δ )) /m ) , we would be done. Proceeding as in Equation (6), let D = be the conditional distribution given avg = ( A ′ ( S 1 ; ∅ ))( x , y ) ≥ 11 / 243 or avg = ( A ′ ( S 2 ; ∅ ))( x , y ) ≥ 11 / 243 . Then,

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

Wenotice that h tie is the output of an ERM learner trained on S = 3 ∼ D = , thus, by standard guarantees on ERM learners (Theorem C.2), it holds with high probability that

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality follows from that h ⋆ ∈ H . Thus, we have that 3

̸

̸

̸

̸

̸

̸

̸

̸

Using a similar argument as in Equation (9), a Chernoff bound yields that | S = 3 | = Θ( m P x , y ∼D [avg = ( A ′ ( S 1 ; ∅ ))( x , y ) ≥ 11 / 243 or avg = ( A ′ ( S 2 ; ∅ ))( x , y ) ≥ 11 / 243 ]) with high probability. Thus, if the probability term were at most c ( τ +( d +ln(1 /δ )) /m ) , the above argument would be complete. However, the analysis from Section 3.1 only bounds the probability term when it is 49 / 100 , instead of 11 / 243 , which is not sufficient. Thus, we will now remedy this using a different splitting algorithm that allows for showing the more fine-grained error is small.

̸

̸

An approach with more than 3 splits: We have seen that the above argument requires that P [avg = ( A ′ ( S ; ∅ ))( x , y ) ≥ 11 / 243 ] ≤ c ( τ + ( d + ln (1 /δ )) /m ) with high probability. However, attempting to prove this by induction, as in Section 3.1, breaks down in the step at which we derived Equation (5). Recall that this is where we relate the condition avg = ( A ′ ( S ; T ))( x, y ) ≥ 11 / 243 to a previous recursive call also erring on an 11 / 243 -fraction of its voters and one of the hypotheses in the remaining recursive calls erring on ( x, y ) . To see why this argument fails, now consider ( x, y ) such that avg = ( A ′ ( S ; T ))( x, y ) ≥ 11 / 243 . Picking an index I ∼ { 1 , 2 , 3 } with probability at least 1 / 3 still returns avg = ( A ′ ( S I , ⊔ ; T ⊔ S I , ⊓ ))( x, y ) ≥ 11 / 243 . However, when picking a random hypothesis ˆ h ∈ ⊔ j ∈{ 1 , 2 , 3 }\ I A ′ ( S 1 ,j, ⊔ ; S j, ⊓ ⊔ T ) , the probability of ˆ h erring can only be lower bounded by 3 / 2 ( 11 / 243 -1 / 3 ) , which is negative! (all the errors might be in the recursive call I ). Thus, we cannot guarantee a lower bound on this probability when making only 3 recursive calls in Algorithm 1. However, if we make more recursive calls, e.g., 27 (chosen large enough to make the argument work), we get that with probability at least 1 / 27 over I ∼ { 1 , . . . , 27 } , it holds that A ( S I , ⊔ ; S I , ⊓ ⊔ T ) ≥ 11 / 243 and with probability at least 27 / 26(11 / 243 -1 / 27) ≈ 0 . 009 over ˆ h ∈ ⊔ j ∈{ 1 ,..., 27 }\ I A ( S 1 ,j, ⊔ ; S j, ⊓ ⊔ T ) , we have that ˆ h ( x ) = y. By 27 · (1 / 0 . 009) ≤ 3160 , this gives

̸

̸

̸

<!-- formula-not-decoded -->

̸

This is precisely the scheme we propose in Algorithm 2, with A ( S ; T ) = {A ( S ′ ) } S ′ ∈S ( S ; T ) . Now, defining ̂ A t ( S ) as t voters drawn from A ( S ; ∅ ) with t = Θ(ln( m/ ( δ ( d +ln(1 /δ ))))) , and mimicking the analysis of Section 3.1 yields that L 11 / 243 D ( ̂ A t ) ≤ c ( τ + ( d + ln (1 /δ ))) /m . Finally, roughly following the above arguments for L D (Tie 11 / 243 ( A ′ t 1 ( S 1 ) , A ′ t 2 ( S 2 ); h tie )) , but now with L D (Tie 11 / 243 ( ̂ A t 1 ( S 1 ) , ̂ A t 2 ( S 2 ); h tie )) , we obtain, for the latter, the claimed generalization error of 2 . 1 τ + O ( √ τ ( d +ln(1 /δ )) /m +( d +ln(1 /δ )) /m ) .

```
Algorithm 2: Splitting algorithm S Input: Training sequences S, T ∈ ( X × Y ) ∗ , where | S | = 3 k for k ∈ N . Output: Family of training sequences. if k ≥ 6 then Partition S into S 1 , . . . , S 27 , with S i being the ( i -1) | S | / 27 + 1 to the i | S | / 27 training examples of S . Set for each i halloooooooooooooo S i, ⊔ = S i [1 : 3 k -6 ] , S i, ⊓ = S i [3 k -6 +1 : 3 k -3 ] , return [ S ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) , . . . , S ( S 27 , ⊔ ; S 27 , ⊓ ⊔ T )] else return S ⊔ T
```

With the above classifier in hand, we can now, on two new independent training sequences, obtain the classifier of Hanneke et al. (2024) using the first sequences, and then on the second sequences choose the best of our classifier and the classifier of Hanneke et al. (2024) as the final classifier.

## 4 Conclusion

Westudy the fundamental problem of agnostic PAC learning and provide improved sample complexity bounds parametrized by τ , the error of the best-in-class hypothesis. Our results resolve the question of Hanneke et al. (2024) asking for optimal error rates in the regime τ ≈ d / m , and make progress on their questions regarding optimal learners for the full range of τ and efficient learners based upon majority votes of ERM s. The most interesting future direction is whether an improved analysis of our voting scheme or a modification of it can lead to optimal algorithms for the full range of τ.

̸

̸

̸

̸

## Acknowledgments and Disclosure of Funding

While this work was carried out, Mikael Møller Høgsgaard was supported by DFF Grant No. 906400068B, and the European Union (ERC,TUCLA, 101125203). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. Julian Asilis was supported by the NSF CAREER Award CCF-223926 and the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-1842487. Part of the work was done while Grigoris Velegkas was a PhD student at Yale University supported in part by the AI Institute for Learning-Enabled Optimization at Scale (TILOS).

## References

- Ishaq Aden-Ali, Yeshwanth Cherapanamjeri, Abhishek Shetty, and Nikita Zhivotovskiy. Optimal pac bounds without uniform convergence. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS) , pages 1203-1223. IEEE, 2023. 3
- Ishaq Aden-Ali, Mikael Møller Høandgsgaard, Kasper Green Larsen, and Nikita Zhivotovskiy. Majority-of-three: The simplest optimal learner? In The Thirty Seventh Annual Conference on Learning Theory , pages 22-45. PMLR, 2024. 3
- Martin Anthony and Peter L Bartlett. Neural network learning: Theoretical foundations . cambridge university press, 2009. 3
- Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K Warmuth. Learnability and the vapnik-chervonenkis dimension. Journal of the ACM (JACM) , 36(4):929-965, 1989. 3
- Stéphane Boucheron, Olivier Bousquet, and Gábor Lugosi. Theory of classification: A survey of some recent advances. ESAIM: probability and statistics , 9:323-375, 2005. 3
- Olivier Bousquet and André Elisseeff. Stability and generalization. Journal of machine learning research , 2(Mar):499-526, 2002. 8
- Olivier Bousquet, Steve Hanneke, Shay Moran, and Nikita Zhivotovskiy. Proper learning, helly number, and an optimal svm bound. In Conference on Learning Theory , pages 582-609. PMLR, 2020. 13, 15
- Luc Devroye, László Györfi, and Gábor Lugosi. A Probabilistic Theory of Pattern Recognition . Springer, 1996. 2
- Andrzej Ehrenfeucht, David Haussler, Michael Kearns, and Leslie Valiant. A general lower bound on the number of examples needed for learning. Information and Computation , 82(3):247-261, 1989. 3
- Steve Hanneke. The optimal sample complexity of pac learning. Journal of Machine Learning Research , 17(38):1-15, 2016. 2, 3, 4, 5, 6, 7
- Steve Hanneke, Kasper Green Larsen, and Nikita Zhivotovskiy. Revisiting agnostic pac learning. In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS) , pages 1968-1982. IEEE, 2024. 2, 3, 8, 10, 34
- David Haussler. Decision theoretic generalizations of the pac model for neural net and other learning applications. Information and computation , 100(1):78-150, 1992. 3
- Mikael Møller Høgsgaard. Efficient optimal pac learning. arXiv preprint arXiv:2502.03620 , 2025. 3
- Kasper Green Larsen. Bagging is an optimal PAC learner. In Gergely Neu and Lorenzo Rosasco, editors, The Thirty Sixth Annual Conference on Learning Theory, COLT 2023, 12-15 July 2023, Bangalore, India , volume 195 of Proceedings of Machine Learning Research , pages 450-468. PMLR, 2023. URL https://proceedings.mlr.press/v195/larsen23a.html . 3, 5
- Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample-variance penalization. In Annual Conference Computational Learning Theory , 2009. URL https: //api.semanticscholar.org/CorpusID:17090214 . 3

Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms . Cambridge University Press, 2014. 17, 20, 26

Hans U Simon. An almost optimal pac algorithm. In Conference on Learning Theory , pages 1552-1563. PMLR, 2015. 2, 3

Leslie G Valiant. A theory of the learnable. Communications of the ACM , 27(11):1134-1142, 1984. 3

Vladimir Vapnik and A Ya Chervonenkis. A class of algorithms for pattern recognition learning. Avtomat. i Telemekh , 25(6):937-945, 1964. 3

Vladimir Vapnik and Alexey Chervonenkis. Theory of pattern recognition, 1974. 3

Andrew J Wagenmaker, Yifang Chen, Max Simchowitz, Simon Du, and Kevin Jamieson. First-order regret in reinforcement learning with linear function approximation: A robust estimation approach. In International Conference on Machine Learning , pages 22384-22429. PMLR, 2022. 3

## A Preliminaries for Proof

In this section we give the preliminaries for the proof. For the reader's convenience, we restate Algorithm 2.

```
Algorithm 2: Splitting algorithm S Input: Training sequences S, T ∈ ( X × Y ) ∗ , where | S | = 3 k for k ∈ N . Output: Family of training sequences. if k ≥ 6 then Partition S into S 1 , . . . , S 27 , with S i being the ( i -1) | S | / 27 + 1 to the i | S | / 27 training examples of S . Set for each i halloooooooooooooo S i, ⊔ = S i [1 : 3 k -6 ] , S i, ⊓ = S i [3 k -6 +1 : 3 k -3 ] , return [ S ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) , . . . , S ( S 27 , ⊔ ; S 27 , ⊓ ⊔ T )] else return S ⊔ T
```

We first observe that for an input training sequence m = | S | = 3 k , the above algorithm makes l recursive calls where l ∈ N satisfies k -6( l -1) ≥ 6 and k -6 l &lt; 6 , that is, l is the largest number such that k/ 6 ≥ l . As l is a natural number, we get that l = ⌊ k/ 6 ⌋ . Furthermore, since k = log 3 ( m ) we get that l = ⌊ log 3 ( m ) / 6 ⌋ . For each of the l recursive calls 27 recursions are made. Thus, the total number of training sequences created in S is 27 l ≤ 3 3 log 3 ( m ) / (2 · 6) = m 1 / (4 ln(3)) ≥ m 0 . 22 . In what follows, we will use the quantity s ⊓ , which we define as | S | | S 1 , ⊓ | when running S ( S ; T ) with S ; T ∈ ( X × Y ) ∗ such that | S | = 3 k and k ≥ 6 . We notice that

<!-- formula-not-decoded -->

This ratio s ⊓ will later in the proof show up as a constant, | S i | / ( | S | s ⊓ ) = ( | S i, ⊔ | + | S i, ⊓ | ) / | S i, ⊓ | = 1 / (1 -1 / 27) , multiplied onto τ. Thus, from this relation we see that if the split of S i is imbalanced so that | S i, ⊓ | is larger than | S i, ⊔ | the constant multiplied on to τ become smaller.

̸

̸

Furthermore, in what follows, for the set of training sequences generated by S ( S ; T ) and a ERM -algorithm A , we write A ( S ; T ) for the set of classifiers the ERM -algorithm outputs when run on the training sequences in S ( S ; T ) , i.e., A ( S ; T ) = {A ( S ′ ) } S ′ ∈S ( S ; T ) , where this is understood as a multiset if the output of the ERM -algorithm is the same for different training sequences in S ( S ; T ) . Furthermore for an example ( x, y ) , we define avg = ( A ( S ; T ))( x, y ) = ∑ h ∈A ( S ; T ) ✶ { h ( x ) = y } / |A ( S ; T ) | , i.e., the average number of incorrect hypotheses in A ( S ; T ) . We notice that by the above comment about S ( S ; T ) having size at least m 0 . 22 , we have that A ( S ; T ) contains just as many hypothesis, each of which is the output of an ERM run on a training sequence of S ( S ; T ) . Thus as

allotted to earlier our algorithm do not run on all the sub training sequences created by S ( S ; T ) , as it calls the A algorithm O (ln ( m/ ( δ ( d +ln(1 /δ ))))) -times. Which leads us considering the following classifier.

̸

̸

Now, for a distribution D over X × {1 , 1 } , training sequences S ; T ∈ ( X × {1 , 1 } ) ∗ , and α ∈ [0 , 1] we will use L α D ( A ( S ; T )) = P ( x , y ) ∼D [avg = ( A ( S ; T ))( x , y ) ≥ α ] , i.e., the probability of at least a α -fraction of the hypotheses in A ( S ; T ) erroring on a new example drawn according to D . As above we also define L α D ( ̂ A t ( S ; T )) = P ( x , y ) ∼D [ avg = ( ̂ A t ( S ; T ))( x , y ) ≥ α ] , for ̂ A t ( S ; T ) . In the following we will for the case where T is the empty training sequence ∅ us ̂ A t ( S ) = ̂ A t ( S ; ∅ ) .

## A.1 Difficulties of the Proof

Let us take the opportunity to briefly describe the challenges associated with settling the sample complexity of agnostic learning in the regime τ = O ( d/m ) . In particular, we will demonstrate that a straightforward multiplicative Chernoff argument can not yield a result as general as Theorem 1.4.

To begin, notice that the true errors of N hypotheses can indeed be confidently estimated to within error τ using O (log( N/δ ) /τ ) samples, due to multiplicative Chernoff. However, if τ = O (log( N/δ ) /m ) , then the above yields a vacuous bound. An additive Chernoff bound would also in this case imply an additive O (log( N/δ ) /m ) error term. In the paper, we consider classes of finite VC dimension d but arbitrary, possibly infinite cardinality N , thereby preventing us from employing this bound. Even if one were to bound N as O (( m/d ) d ) using Sauer-Shelah, this would imply that multiplicative Chernoff is unhelpful when τ = O (( d log( m/d ) + log(1 /δ )) /m ) . This is a wide regime containing, for instance, τ = O ( d/m ) , a setting in which we demonstrate that our algorithm is optimal. Furthermore, an additive Chernoff bound would incur a suboptimal additive O ( d log( m/d ) /m ) term.

However, let us assume that a Chernoff bound argument for infinite hypothesis classes could yield Theorem B.2 for all τ (e.g., using an ϵ -covering argument). In this case, one would have demonstrated that an empirical risk minimizer h ′ has error 2 . 1 τ + O ( √ τ ( d +ln(1 /δ )) m ) + d +ln(1 /δ ) m . Then, taking τ = 0 (i.e, considering the realizable case), would demonstrate that ERM results in an error of O ( d +ln(1 /δ ) m ) . However, this is not generally true, owing to the worst-case lower bound on ERM learners of O ( d ln (1 /ε )+ln (1 /δ ) m ) , due to (Bousquet et al., 2020, Theorem 11). This strongly suggests that a Chernoff bound argument using a union bound over each function in the hypothesis class (or a discretization of it) cannot yield a result as general as Theorem 1.1, which holds across all τ .

Finally, we remark that our learner for Theorem B.2 is agnostic to τ (i.e., is oblivious to the value of τ ), whereas the previous Chernoff procedure requires knowledge of τ . This is a notable distinction, as precise knowledge of τ will typically be unavailable to the learner. In particular, ERM-based estimates of τ will be uninformative in the regime τ = o ( d/m ) .

## A Preliminaries for Proof

In this section we give the preliminaries for the proof. For the reader's convenience, we restate Algorithm 2.

For a natural number t, we let ̂ A t ( S ; T ) , be the random collection of t hypotheses drawn uniformly with replacement from A ( S ; T ) , with the draws being independent, where we see ̂ A t ( S ; T ) as a multiset so allowing for repetitions. We remark here that we will overload notation and use t (so t in bold font) to denote the randomness used to draw the t hypotheses from A ( S ; T ) in the following analysis of ̂ A t ( S ; T ) . Intuitively one can think of ̂ A t as a bagging algorithm where the subsampled training sequences are restricted to subsets of S ( S ; T ) rather than sampling with replacement from the training examples of S and T. In what follows we will consider this algorithm parametrized by t = O (ln ( m/ ( δ ( d +ln(1 /δ ))))) leading to a classifier with the same order of call to the ERM as stated in. Similarly to A ( S ; T ) we also define avg = ( ̂ A t ( S ; T ))( x, y ) = ∑ h ∈ ̂ A t ( S ; T ) ✶ { h ( x ) = y } / | ̂ A t ( S ; T ) |

̸

̸

## Algorithm 2: Splitting algorithm S

```
Input: Training sequences S, T ∈ ( X × Y ) ∗ , where | S | = 3 k for k ∈ N . Output: Family of training sequences. if k ≥ 6 then Partition S into S 1 , . . . , S 27 , with S i being the ( i -1) | S | / 27 + 1 to the i | S | / 27 training examples of S . Set for each i halloooooooooooooo S i, ⊔ = S i [1 : 3 k -6 ] , S i, ⊓ = S i [3 k -6 +1 : 3 k -3 ] , return [ S ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) , . . . , S ( S 27 , ⊔ ; S 27 , ⊓ ⊔ T )] else return S ⊔ T
```

We first observe that for an input training sequence m = | S | = 3 k , the above algorithm makes l recursive calls where l ∈ N satisfies k -6( l -1) ≥ 6 and k -6 l &lt; 6 , that is, l is the largest number such that k/ 6 ≥ l . As l is a natural number, we get that l = ⌊ k/ 6 ⌋ . Furthermore, since k = log 3 ( m ) we get that l = ⌊ log 3 ( m ) / 6 ⌋ . For each of the l recursive calls 27 recursions are made. Thus, the total number of training sequences created in S is 27 l ≤ 3 3 log 3 ( m ) / (2 · 6) = m 1 / (4 ln(3)) ≥ m 0 . 22 . In what follows, we will use the quantity s ⊓ , which we define as | S | | S 1 , ⊓ | when running S ( S ; T ) with S ; T ∈ ( X × Y ) ∗ such that | S | = 3 k and k ≥ 6 . We notice that

<!-- formula-not-decoded -->

This ratio s ⊓ will later in the proof show up as a constant, | S i | / ( | S | s ⊓ ) = ( | S i, ⊔ | + | S i, ⊓ | ) / | S i, ⊓ | = 1 / (1 -1 / 27) , multiplied onto τ. Thus, from this relation we see that if the split of S i is imbalanced so that | S i, ⊓ | is larger than | S i, ⊔ | the constant multiplied on to τ become smaller.

̸

Furthermore, in what follows, for the set of training sequences generated by S ( S ; T ) and a ERM -algorithm A , we write A ( S ; T ) for the set of classifiers the ERM -algorithm outputs when run on the training sequences in S ( S ; T ) , i.e., A ( S ; T ) = {A ( S ′ ) } S ′ ∈S ( S ; T ) , where this is understood as a multiset if the output of the ERM -algorithm is the same for different training sequences in S ( S ; T ) . Furthermore for an example ( x, y ) , we define avg = ( A ( S ; T ))( x, y ) = ∑ h ∈A ( S ; T ) ✶ { h ( x ) = y } / |A ( S ; T ) | , i.e., the average number of incorrect hypotheses in A ( S ; T ) . We notice that by the above comment about S ( S ; T ) having size at least m 0 . 22 , we have that A ( S ; T ) contains just as many hypothesis, each of which is the output of an ERM run on a training sequence of S ( S ; T ) . Thus as allotted to earlier our algorithm do not run on all the sub training sequences created by S ( S ; T ) , as it calls the A algorithm O (ln ( m/ ( δ ( d +ln(1 /δ ))))) -times. Which leads us considering the following classifier.

̸

̸

Now, for a distribution D over X × {1 , 1 } , training sequences S ; T ∈ ( X × {1 , 1 } ) ∗ , and α ∈ [0 , 1] we will use L α D ( A ( S ; T )) = P ( x , y ) ∼D [avg = ( A ( S ; T ))( x , y ) ≥ α ] , i.e., the probability of at least a α -fraction of the hypotheses in A ( S ; T ) erroring on a new example drawn according to D . As above we also define L α D ( ̂ A t ( S ; T )) = P ( x , y ) ∼D [ avg = ( ̂ A t ( S ; T ))( x , y ) ≥ α ] , for ̂ A t ( S ; T ) . In the following we will for the case where T is the empty training sequence ∅ us ̂ A t ( S ) = ̂ A t ( S ; ∅ ) .

̸

For a natural number t, we let ̂ A t ( S ; T ) , be the random collection of t hypotheses drawn uniformly with replacement from A ( S ; T ) , with the draws being independent, where we see ̂ A t ( S ; T ) as a multiset so allowing for repetitions. We remark here that we will overload notation and use t (so t in bold font) to denote the randomness used to draw the t hypotheses from A ( S ; T ) in the following analysis of ̂ A t ( S ; T ) . Intuitively one can think of ̂ A t as a bagging algorithm where the subsampled training sequences are restricted to subsets of S ( S ; T ) rather than sampling with replacement from the training examples of S and T. In what follows we will consider this algorithm parametrized by t = O (ln ( m/ ( δ ( d +ln(1 /δ ))))) leading to a classifier with the same order of call to the ERM as stated in. Similarly to A ( S ; T ) we also define avg = ( ̂ A t ( S ; T ))( x, y ) = ∑ h ∈ ̂ A t ( S ; T ) ✶ { h ( x ) = y } / | ̂ A t ( S ; T ) |

̸

̸

## A.1 Difficulties of the Proof

Let us take the opportunity to briefly describe the challenges associated with settling the sample complexity of agnostic learning in the regime τ = O ( d/m ) . In particular, we will demonstrate that a straightforward multiplicative Chernoff argument can not yield a result as general as Theorem 1.4.

To begin, notice that the true errors of N hypotheses can indeed be confidently estimated to within error τ using O (log( N/δ ) /τ ) samples, due to multiplicative Chernoff. However, if τ = O (log( N/δ ) /m ) , then the above yields a vacuous bound. An additive Chernoff bound would also in this case imply an additive O (log( N/δ ) /m ) error term. In the paper, we consider classes of finite VC dimension d but arbitrary, possibly infinite cardinality N , thereby preventing us from employing this bound. Even if one were to bound N as O (( m/d ) d ) using Sauer-Shelah, this would imply that multiplicative Chernoff is unhelpful when τ = O (( d log( m/d ) + log(1 /δ )) /m ) . This is a wide regime containing, for instance, τ = O ( d/m ) , a setting in which we demonstrate that our algorithm is optimal. Furthermore, an additive Chernoff bound would incur a suboptimal additive O ( d log( m/d ) /m ) term.

However, let us assume that a Chernoff bound argument for infinite hypothesis classes could yield Theorem B.2 for all τ (e.g., using an ϵ -covering argument). In this case, one would have demonstrated that an empirical risk minimizer h ′ has error 2 . 1 τ + O ( √ τ ( d +ln(1 /δ )) m ) + d +ln(1 /δ ) m . Then, taking τ = 0 (i.e, considering the realizable case), would demonstrate that ERM results in an error of O ( d +ln(1 /δ ) m ) . However, this is not generally true, owing to the worst-case lower bound on ERM learners of O ( d ln (1 /ε )+ln (1 /δ ) m ) , due to (Bousquet et al., 2020, Theorem 11). This strongly suggests that a Chernoff bound argument using a union bound over each function in the hypothesis class (or a discretization of it) cannot yield a result as general as Theorem 1.1, which holds across all τ .

Finally, we remark that our learner for Theorem B.2 is agnostic to τ (i.e., is oblivious to the value of τ ), whereas the previous Chernoff procedure requires knowledge of τ . This is a notable distinction, as precise knowledge of τ will typically be unavailable to the learner. In particular, ERM-based estimates of τ will be uninformative in the regime τ = o ( d/m ) .

## B Analysis of ̂ A t

As described in the proof sketch, we require a bound on L 10 / 243 D A ( S ; ∅ ) in order to upper bound L 11 / 243 D ( ̂ A t ) . Thus, we now present our error bound for Algorithm 2 when running A on each dataset generated on S ( S , ∅ ) . (We assume that | S | = 3 k for k ∈ N , at the cost of discarding a constant fraction of training points.)

Lemma B.1. There exists a universal constant c ≥ 1 such that: For any hypothesis class H of VC dimension d , distribution D over X × Y , failure parameter 0 &lt; δ &lt; 1 , training sequence size m = 3 k for k ≥ 6 , and training sequence S ∼ D m , with probability at least 1 -δ over S one has that

<!-- formula-not-decoded -->

Let us defer the proof of Lemma B.1 for the moment, and proceed with presenting the main theorem of this section, assuming the claim of Lemma B.1.

Theorem B.2. There exists a universal constant c ≥ 1 such that: For any hypothesis class H of VC dimension d , distribution D over X×Y , failure parameter 0 &lt; δ &lt; 1 , training sequence size m = 3 k for k ≥ 6 , training sequence S ∼ D m , and sampling size t ≥ 4 · 243 2 ln (2 m/ ( δ ( d +ln(1 /δ )))) , we have with probability at least 1 -δ over S and the randomness t used to draw ̂ A t ( S ) that:

<!-- formula-not-decoded -->

Proof. Let ̂ A t ( S ; ∅ ) = { ˆ h 1 , . . . , ˆ h t } , considered as a multiset, and recall that the ˆ h i are drawn uniformly at random from A ( S ; ∅ ) = {A ( S ′ ) } S ′ ∈S ( S ; ∅ ) , which is likewise treated as a multiset. Let

E S denote the event and

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

its complement. Now fix a realization S of S . Using the fact that P [ A ] = P [ A ∩ B ] + P [ A ∩ B ] , we have that

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

In pursuit of bounding the second term, consider a labeled example ( x, y ) ∈ ¯ E S . We may assume that ¯ E S is non-empty, as otherwise the term is simply 0 . Now, for any such labeled example ( x, y ) ∈ ¯ E S we have that ∑ t i =1 { ˆ h i ( x ) = y } has expectation

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where the final inequality follows from the fact that ( x, y ) ∈ ¯ E S . (Recall that we use the boldface symbol t to denote the randomness underlying the random variables ˆ h 1 , . . . , ˆ h t .) Now, since ✶ { ˆ h i ( x ) = y } is a collection of i.i.d. { 0 , 1 }-random variables, we have by the Chernoff inequality that where the final inequality follows from the fact that µ ( x,y ) ≥ t / 2 and

<!-- formula-not-decoded -->

The above implies that with probability at least 1 -δ ( d +ln(1 /δ )) 2 m , we have

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

This further implies that ∑ t i =1 ✶ { ˆ h i ( x ) = y } /t &lt; 11 / 243 with probability at most δ ( d +ln(1 /δ )) 2 m . As we demonstrated this fact for any pair ( x, y ) ∈ ¯ E S , an application of Markov's inequality yields that

̸

̸

̸

Note that the first inequality follows from an application of Markov's inequality and the observation that ¯ E S depends only upon ( x , y ) and t , which are independent from one another, meaning we can swap the order of expectation. The final inequality follows from the bound on the probability of ∑ t i =1 { ˆ h t ( x ) = y } /t ≥ 11 243 happening over t , for x, y ∈ ¯ E S .

✶ Thus, we conclude that with probability at least 1 -δ/ 2 over t , the random draw of the hypothesis in ̂ A t ( S ; ∅ ) is such that

̸

Furthermore, as we showed this for any realization S of S (and t and S are independent), we conclude that, with probability at least 1 -δ/ 2 over both t and S ,

<!-- formula-not-decoded -->

̸

Furthermore, by Lemma B.1, we have that with probability at least 1 -δ/ 2 over S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, as this event does not depend on the randomness t employed in drawing hypotheses for ̂ A t ( S ) , we conclude that the above also holds with probability at least 1 -δ/ 2 over both S and t . Now, applying a union bound over the event in Equation (18) and Equation (19), combined with the bound on L D ( ̂ A t ( S )) in Equation (17), we get that, with probability at least 1 -δ over S and t , it holds that

<!-- formula-not-decoded -->

As c is permitted by any absolute constant, this concludes the proof.

We now proceed to give the proof of Lemma B.1. Doing so will require two additional results, the first of which relates the empirical error of a hypothesis h to its true error.

Lemma B.3 (Shalev-Shwartz and Ben-David (2014) Lemma B.10) . Let D be a distribution over X × {1 , 1 } , h ∈ {-1 , 1 } X be a hypothesis, δ ∈ (0 , 1) a failure parameter, and m ∈ N a natural number. Then,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The second result we require is one which bounds the error of A ( S ; T ) for arbitrary training sets T (i.e., not merely T = ∅ , as we have previously considered).

Theorem B.4. There exists, universal constant c ≥ 1 such that: For any a hypothesis class H of VC dimension d , any distribution D over X × {1 , 1 } , any failure parameter δ ∈ (0 , 1) , any training sequence size m = 3 k , any training sequence T ∈ ( X ×{1 , 1 } ) ⋆ , and a random training sequence S ∼ D m , it holds with probability at least 1 -δ over S that:

̸

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

̸

̸

Let us first give the proof of Lemma B.1 assuming Theorem B.4, and subsequently offer the proof of Theorem B.4.

Proof of Lemma B.1. First note that for each i ∈ { 1 , . . . , 27 } ,

̸

<!-- formula-not-decoded -->

̸

The inequality follows from the fact that any S ′ ∈ S ( S i, ⊔ ; S i, ⊓ ) satisfies S ′ ⊏ S i, ⊓ ⊔ S i, ⊔ and the final equality uses the facts that | S i, ⊓ ⊔ S i, ⊔ | = | S i | = m/ 3 3 and s ⊓ = 3 3 / (1 -1 / 27) . Thus, invoking Lemma B.3 over S i for i ∈ { 1 , . . . , 27 } with failure parameter δ/ 28 , we have by a union bound that with probability at least 1 -27 δ/ 28 over S , each i ∈ { 1 , . . . , 27 } satisfies

̸

<!-- formula-not-decoded -->

where we have used that √ ab ≤ a + b. Furthermore, by Theorem B.4 we have that with probability at least 1 -δ/ 28 over S ,

̸

<!-- formula-not-decoded -->

Again invoking a union bound, we have with probability at least 1 -δ over S that

̸

<!-- formula-not-decoded -->

where the first inequality follows from Equation (20). This concludes the proof.

We now direct our attention to proving Theorem B.4. We will make use of another set of two lemmas, the first of which permits us to make a recursive argument over A -calls based on sub-training sequences created in Algorithm 2.

Lemma B.5. Let S, T ∈ ( X ×Y ) ∗ with | S | = 3 k for k ≥ 6 , and let D be a distribution over X ×Y . Then,

̸

̸

<!-- formula-not-decoded -->

̸

Proof. Let ( x, y ) be an example such that

<!-- formula-not-decoded -->

̸

As k ≥ 6 , Algorithm 2 calls itself when called with ( S ; T ) . Furthermore, as each of the 27 calls produce an equal number of subtraining sequences, it must be the case that

̸

<!-- formula-not-decoded -->

̸

̸

This in turn implies that there exists an ˆ i ∈ [27] satisfying the above inequality, i.e., such that

̸

<!-- formula-not-decoded -->

̸

̸

We further observe that for any i ∈ [27] ,

<!-- formula-not-decoded -->

̸

This implies, again for any arbitrary choice of i ∈ [27] , that

̸

<!-- formula-not-decoded -->

̸

Using the above, we can conclude that when ( x, y ) is such that avg = ( A ( S ; T ))( x, y ) ≥ 10 243 , then there exists an i ∈ [27] with avg = ( A ( S i, ⊔ ; S i, ⊓ ⊔ T ))( x, y ) ≥ 10 243 . Then by Equation (21), at least a 1 / 234 -fraction of hypotheses in ⊔ j ∈{ 1 ,..., 27 }\ i A ( S j, ⊔ ; S j, ⊓ ⊔ T ) err on ( x, y ) . Thus, if we let I be drawn uniformly at random from { 1 , . . . , 27 } and ˆ h be drawn uniformly at random from ⊔ j ∈{ 1 ,..., 27 }\ I A ( S j, ⊔ ; S j, ⊓ ⊔ T ) , then by the law of total probability we have that

̸

̸

̸

̸

<!-- formula-not-decoded -->

This implies in turn that

̸

<!-- formula-not-decoded -->

̸

Taking expectations with respect to ( x , y ) ∼ D , we have

̸

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

And as I ∈ { 1 , . . . , 27 } , then clearly

<!-- formula-not-decoded -->

Simply multiplying both sides by 27 / 26 , we have that

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

By combining Equations (22) to (24), we conclude that

̸

<!-- formula-not-decoded -->

which completes the proof.

The second lemma employed in the proof of Theorem B.4 is the standard uniform convergence property for VC classes.

Lemma B.6 (Shalev-Shwartz and Ben-David (2014), Theorem 6.8) . There exists a universal constant C &gt; 1 such that for any distribution D over X × {1 , 1 } and any hypothesis class H ⊆ {1 , 1 } X with finite VC-dimension d , it holds with probability at least 1 -δ over S ∼ D m that for all h ∈ H :

<!-- formula-not-decoded -->

We now present the proof of Theorem B.4, which concludes the section.

Proof of Theorem B.4. We induct on k ≥ 1 . In particular, we will demonstrate that for each k ≥ 1 and S ∼ D m with m = 3 k , and for any δ ∈ (0 , 1) , T ∈ ( X × Y ) ∗ , one has with probability at least 1 -δ over S that

̸

<!-- formula-not-decoded -->

̸

where s ⊓ = | S | | S i, ⊓ | = 27 1 -1 / 27 is the previously defined constant, C ≥ 1 is the constant from Lemma B.6, and c b and c c are the following constants:

<!-- formula-not-decoded -->

Note that applying √ ab ≤ a + b to Equation (25) would in fact suffice to complete the proof of Theorem B.4.

Thus it remains only to justify Equation (25). For any choice of δ ∈ (0 , 1) and T ∈ ( X × Y ) ∗ , first observe observe that if k ≤ 12 , the claim follows immediately from the fact that the right hand side of Equation (25) is at least 1. (Owing to the fact that c c ≥ 3 12 .)

We now proceed to the inductive step. For the sake of brevity, we will often suppress the distribution from which random variables are drawn when writing expectations and probabilities, e.g., P S rather than P S ∼D m . Now fix a choice of T ∈ ( X × Y ) ∗ , δ ∈ (0 , 1) , and k &gt; 12 . Let a S equal the right-hand side of Equation (25), i.e.,

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

Then invoking Lemma B.5 and a union bound, we have that

̸

<!-- formula-not-decoded -->

̸

̸

Thus it suffices to show that for i = j ∈ [27] ,

̸

<!-- formula-not-decoded -->

̸

as one can immediately apply this inequality with Equation (27). Then it remains to establish Equation (29). As the pairs ( S 1 , ⊔ , S 1 , ⊓ ) , . . . , ( S 27 , ⊔ , S 27 , ⊓ ) are all i.i.d., it suffices to demonstrate the inequality for, say, j = 1 and i = 2 . To this end, fix arbitrary realizations ( S k ) 3 ≤ k ≤ 27 of the random variables ( S k ) 3 ≤ k ≤ 27 ; we will demonstrate the claim for any such realization.

First note that if we happen to have realizations S 2 , ⊔ , S 2 , ⊓ of S 2 , ⊔ , S 2 , ⊓ such that

̸

<!-- formula-not-decoded -->

then we are done by monotonicity of measures, as

̸

<!-- formula-not-decoded -->

̸

Furthermore, consider any realization S 2 , ⊓ of S 2 , ⊓ . We note that by m = | S | = 3 k for k &gt; 12 , and by Algorithm 2, it holds that | S 2 , ⊔ | = 3 k -6 = m/ 3 6 . Thus, we may invoke the inductive hypothesis with A ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ) and failure parameter δ/ 24 in order to conclude that with probability at least 1 -δ/ 24 over S 2 , ⊔ ,

̸

<!-- formula-not-decoded -->

Furthermore, for any a, b, c, d &gt; 0 , we have that

<!-- formula-not-decoded -->

where the inequality follows from the fact that √ abc ≤ max( √ ba 2 , √ bc 2 ) ≤ a √ b + c √ b . Now, combining Equation (31) and Equation (32) (with b = c b , d = c c ), we obtain

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

Note that the second inequality makes use of the fact that c b ≥ 1 and c b ≤ c c . We thus conclude that for any realization S 2 , ⊓ of S 2 , ⊓ , the above inequality holds with probability at least 1 -δ/ 24 over S 2 , ⊔ . Further, as S 2 , ⊓ and S 2 , ⊔ are independent, the inequality also holds with probability at least 1 -δ/ 24 over S 2 , ⊓ , S 2 , ⊔ .

We now let

<!-- formula-not-decoded -->

̸

and consider the following three events over S 2 = ( S 2 , ⊓ , S 2 , ⊔ ) :

<!-- formula-not-decoded -->

By Equation (30), we have that for S 2 , ⊓ , S 2 , ⊔ ∈ E 2 , the bound in Equation (25) holds. Furthermore, from the comment below Equation (33), we have that S 2 , ⊓ , S 2 , ⊔ ∈ E 3 happens with probability at most δ/ 24 over S 2 , ⊓ , S 2 , ⊔ . For brevity, let a S denote the right-hand side of Equation (25). Then, using the law of total probability along with independence of S 1 and S 2 , we can conclude that

̸

̸

<!-- formula-not-decoded -->

̸

̸

Note that the second inequality follows from Equation (30), Equation (33) and P [ E 1 ] ≤ 1 . Thus, if we can bound the first term of the final line by 2 δ/ 24 , it will follow that Equation (29) holds with probability at least 1 -δ/ (26 · 27) , as claimed.

To this end, consider a realization S 2 of S 2 ∈ E 1 . For such an S 2 , we have that

̸

<!-- formula-not-decoded -->

Then, again invoking the law of total probability, we have that

̸

<!-- formula-not-decoded -->

̸

̸

̸

Now let A = { ( x, y ) ∈ ( X × Y ) | avg = ( A ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( x, y ) ≥ 10 243 } and N 1 = | S 1 , ⊓ ⊓ A | . As S 2 ∈ E 1 , we have that

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

Then, owing to the fact that S 1 , ⊓ ∼ D ( m/s ⊓ ) -note that m/s ⊓ = | S | / ( | S | / | S ⊓ | ) = | S ⊓ | -this implies that

<!-- formula-not-decoded -->

Thus, by a multiplicative Chernoff bound, we have

<!-- formula-not-decoded -->

Note that the second inequality uses the fact that c c = 3 12 ln (24 e ) 2 √ c b s ⊓ . Now let N 1 be any realization of N 1 such that

<!-- formula-not-decoded -->

̸

Notice that S 1 , ⊓ ⊓ A ∼ D N 1 ( · | avg = ( A ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( x , y ) ≥ 10 243 ) .

Now, combining Equation (35) and Lemma B.6, we have that with probability at least 1 -δ/ 24 over S 1 , ⊓ ⊓ A ,

̸

<!-- formula-not-decoded -->

̸

We now bound each of the two terms on the right hand side of Equation (37), considered after multiplying out the term associated with L 10 / 243 D . Beginning with the first term, and recalling that N 1 = | S 1 , ⊓ ⊓ A | , we have

̸

<!-- formula-not-decoded -->

̸

̸

Note that the first inequality uses the fact that N 1 ≥ L 10 / 243 D ( A ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( m/s ⊓ ) / 2 and the second inequality uses that h ∈ A ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) , meaning there exists an S ′ ∈ S ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) such that h = A ( S ′ ) . The third inequality follows from the fact that S 1 , ⊓ ⊓ A ⊏ ˜ S for any ˜ S ∈ S ( S 1 , ⊔ ; S 1 , ⊓ ⊔ T ) (and especially for S ′ ) and the final inequality from both the ERM -property of A on S ′ and the definition of Σ = ( h ⋆ , S ′ ) .

̸

̸

̸

We now bound the second term of Equation (37). In what follows, let β = C ( d +ln(24 e/δ )) . We will in the first inequality use that N 1 ≥ 1 2 · L 10 / 243 D ( A ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( m/s ⊓ ) :

̸

<!-- formula-not-decoded -->

̸

where the last inequality follows from β = C ( d + ln (24 e/δ )) ≤ ln (24 e ) C ( d + ln ( e/δ )) and rearrangement. We now bound each of the constant terms under the square roots. Beginning with the first term, we have

<!-- formula-not-decoded -->

where the inequality follows from c b = ( 5687 2 · 4 · 3 6 ln (24 e ) s ⊓ ) 2 . For the second term, we have that

<!-- formula-not-decoded -->

where the inequality follows by c c = 3 12 ln (24 e ) 2 √ c b s ⊓ and c b = ( 5687 2 · 4 · 3 6 ln (24 e ) s ⊓ ) 2 . Then we conclude from Equation (39) that

̸

<!-- formula-not-decoded -->

Thus, by applying Equation (38) and Equation (42) to Equation (37), we obtain that for any realization N 1 of N 1 ≥ 1 2 · L 10 / 243 D ( A ( S 2 , ⊔ ; S 2 , ⊓ ⊔ T ))( m/s ⊓ ) , it holds with probability at least 1 -δ/ 24 over

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

<!-- formula-not-decoded -->

Note that the second inequality follows from the fact that S ′ ∈ S ( S i, ⊔ ; S i, ⊓ ⊔ T ) for i = 1 , 2 , meaning S ′ ∈ S ( S ; T ) . The equality follows simply from the definition of a S in Equation (26).

Now, combining the above observations, we can conclude that for any realization S 2 ∈ E 1 of S 2 ,

̸

̸

<!-- formula-not-decoded -->

̸

̸

(Note that the first term on the right side of the first inequality is distributed across two lines of text.) In particular, the first inequality follows from the law of total expectation, and the second from Equation (36) (see the comment below the equations) and Equation (43). Thus, the above implies that the term in Equation (34) which conditions upon S 2 ∈ E 1 is bounded by δ/ 2 11 . Altogether, we can conclude that

̸

<!-- formula-not-decoded -->

̸

As δ / 2 10 ≤ δ / 26 · 27 , we arrive at Equation (29), which as previously argued concludes the proof due to the fact that ( S 1 , ⊔ , S 1 , ⊓ ) , . . . , ( S 27 , ⊔ , S 27 , ⊓ ) are i.i.d.

## C Augmentation of ̂ A t Through Tie-breaking

Let us assume we are given a training sequence S of size m = 3 k for k ≥ 1 . We then take S and split it into three disjoint, equal-sized training sequences S 1 , S 2 , and S 3 . We denote the sizes of S 1 , S 2 and S 3 as m ′ = m/ 3 . On S 1 and S 2 , we train ̂ A t 1 ( S 1 ) and ̂ A t 2 ( S 2 ) , where we recall that t 1 and t 2 denote the randomness used to draw the hypothesis in ̂ A t 1 ( S 1 ) and ̂ A t 2 ( S 2 ) from, respectively, A ( S 1 ; ∅ ) and A ( S 2 ; ∅ ) .

̸

We now evaluate ̂ A t 1 ( S 1 ) and ̂ A t 2 ( S 2 ) on S 3 and consider all the examples ( x, y ) ∈ S 3 , where avg = ( ̂ A t 1 ( S 1 ))( x, y ) ≥ 11 / 243 or avg = ( ̂ A t 2 ( S 2 ))( x, y ) ≥ 11 / 243 . Denote the set of all such examples as S = 3 . We now run the ERM -algorithm A on S = 3 to obtain h tie = A ( S = 3 ) .

̸

̸

̸

2 I.e., the restriction of D to those ( x, y ) pairs satisfying the given condition.

̸

̸

̸

̸

For a point x , let Tie 11 / 243 ( ̂ A t 1 ( S 1 ) , ̂ A t 2 ( S 2 ); h tie ) ( x ) be equal to the label y if both and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Otherwise, we set it to h tie ( x ) . In other words, if both ̂ A t 1 ( S 1 ) and ̂ A t 2 ( S 2 ) have at least 232 / 243 of their hypotheses agreeing on the same label y , we output that label; otherwise, we output the label of h tie ( x ) .

Notice that if there were a true label y and point x , such that we ended up outputting the answer of h tie ( x ) , then at least one of ̂ A t 1 ( S 1 ) and ̂ A t 2 ( S 2 ) has more than 11 / 243 incorrect answers, not equal to y on x , which we know by Lemma B.1 is unlikely. Furthermore, in the former case and the tie erring on ( x, y ) , then both ̂ A t 1 ( S 1 ) and ̂ A t 2 ( S 2 ) err with at least a 232 / 243 fraction of their hypotheses, which again is unlikely by Lemma B.1. Thus, both cases of possible error are unlikely, which we exploit in order to demonstrate the following theorem.

Theorem C.1. There exists a universal constant c ≥ 1 such that for any hypothesis class H of VC dimension d , distribution D over X × Y , failure parameter 0 &lt; δ &lt; 1 , training sequence size m = 3 k for k ≥ 5 , training sequence S ∼ D m , and sampling size t 1 , t 2 ≥ 4 · 243 2 ln ( 2 m/ ( δ ( d +ln(86 /δ ))) ) , we have, with probability at least 1 -δ over S 1 , S 2 , S 3 ∼ D m/ 3 and the randomness t 1 , t 2 used to draw ̂ A t 1 ( S 1 ; ∅ ) and ̂ A t 2 ( S 2 ; ∅ ) , that:

<!-- formula-not-decoded -->

In the proof of Theorem C.1 we will need the following ERM -theorem. Recall that we take A to be an ERM -algorithm, meaning A is proper (i.e., it always emits hypotheses in H ), and for any training sequence S that L S ( A ( S )) = min h ∈H L S ( h ) .

Theorem C.2. [Shalev-Shwartz and Ben-David (2014) Theorem 6.8] There exists a universal constant C ′ &gt; 1 such that for any distribution D over X ×{1 , 1 } , any hypothesis class H ⊆ {1 , 1 } X with VC dimension d , and any ERM -algorithm A , it holds with probability at least 1 -δ over S ∼ D m that for all h ∈ H :

<!-- formula-not-decoded -->

̸

Proof of Theorem C.1. First note that by the definition of Tie 11 / 243 , for Tie 11 / 243 ( ̂ A t 1 ( S 1 ) , ̂ A t 2 ( S 2 ) , h tie ) to err on a fixed example ( x, y ) , it must be the case that either there exists y ′ = y ∈ {-1 , 1 } such that

̸

<!-- formula-not-decoded -->

or the case that for all y ′ ∈ {-1 , 1 } ,

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

Thus, we have that

<!-- formula-not-decoded -->

̸

̸

̸

We now bound each of these terms separately. The first term we will soon bound by

̸

<!-- formula-not-decoded -->

̸

with probability at least 1 -82 δ/c u , over S 1 , S 2 , t 1 and t 2 . Likewise, the second term we will soon bound by

̸

<!-- formula-not-decoded -->

̸

with probability 1 -4 δ/c u at least over S 1 , S 2 , S 3 , t 1 and t 2 , where c, C, C ′ ≥ 1 are universal constants and c u = 86 . Applying a union bound over the above two events establishes the claim of the theorem.

̸

Let us begin by pursuing Equation (46). To this end, consider any realizations of S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 . We note that for an example ( x, y ) with avg = ( ̂ A t 1 ( S 1 ))( x, y ) ≥ 232 / 243 and avg = ( ̂ A t 2 ( S 2 ))( x, y ) ≥ 232 / 243 , it must be the case that both ̂ A t 1 ( S 1 ) and ̂ A t 2 ( S 2 ) have at least a 232 / 243 -fraction of hypotheses which err at ( x, y ) . Now let ˆ h be a random hypothesis drawn from ̂ A t 1 ( S 1 ) . Then with probability at least 232 / 243 , ˆ h ( x ) = y. Thus, for any such example ( x, y ) , we conclude that

̸

̸

̸

̸

Multiplying both sides of the above equation by 243 / 232 and taking expectation with respect to ( x , y ) ∼ D on both sides, we obtain

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

̸

̸

Now by ˆ h being drawn from ̂ A t 1 ( S 1 ) , which in turn is drawn from A ( S 1 ; ∅ ) , we conclude that ˆ h is contained in A ( S 1 ; ∅ ) . Thus,

̸

<!-- formula-not-decoded -->

̸

̸

can be upper bounded by max h ∈A ( S 1 ; ∅ ) P ( x , y ) ∼D [ h ( x ) = y, avg = ( ̂ A t 2 ( S 2 ))( x , y ) ≥ 232 / 243 ] . Using this observation and substituting it into Equation (48), we obtain that

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

As we demonstrated the above inequality for any realizations of S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 , the inequality also holds for the random variables. We now demonstrate that the right-hand side of the above expression can be bounded by

̸

<!-- formula-not-decoded -->

̸

with probability at least 1 -82 δ/c u over S 1 , S 2 , and t 2 . We denote the above event over S 1 , S 2 , and t 2 as E G . In pursuit of Equation (50), we now consider the following events over S 2 and t 2 :

<!-- formula-not-decoded -->

where c is at least the constant of Theorem B.2 and also greater than c ≥ 2 · 10 6 s ⊓ . We first notice that if S 2 and t 2 are realizations of S 2 and t 2 in E 1 , then by monotonicity of measures, we have that

̸

<!-- formula-not-decoded -->

̸

which would imply the event E G in Equation (50).

We now consider realizations S 2 and t 2 of S 2 and t 2 in E 2 . Let S 1 ,i, ⊓ denote ( S 1 ) i, ⊓ and S 1 ,i, ⊔ = ( S 1 ) i, ⊔ for i ∈ { 1 , . . . , 27 } . Using this notation, we have that

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

We now bound the max term associated to each i ∈ [27] . To this end, fix such an i and let A = { ( x, y ) | avg = ( ̂ A t 2 ( S 2 ))( x, y ) ≥ 11 / 243 } . Also, let N i denote the number of examples in S 1 ,i, ⊓ landing in A , i.e., N i = | S 1 ,i, ⊓ ⊓ A | . Now by S 2 and t 2 being realizations of S 2 and t 2 in E 2 , we have that P [ A ] = P ( x , y ) ∼D [ A ] = L D ( ̂ A t 2 ( S 2 )) ≥ c ln ( c u e/δ ) m ′ . Thus, as S 1 ,i, ⊓ ∼ D m ′ /s ⊓ , we have that E [ N i ] = P [ A ] m ′ /s ⊓ . Furthermore, this implies by Chernoff that with probability at least 1 -δ/c u over S 1 ,i, ⊓ ,

<!-- formula-not-decoded -->

where the last inequality follows by P [ A ] ≥ c ln ( c u e/δ ) m ′ and c ≥ 2 · 10 6 s ⊓ . Let now D ( · | A ) be the conditional distribution of A , i.e., for an event E over X × Y , we have that D ( E | A ) = P ( x , y ) ∼D [( x , y ) ∈ E ∩ A ] / P ( x , y ) ∼D [( x , y ) ∈ A ] . Since S 1 ,i, ⊓ ∼ D , it follows that S 1 ⊓ A ∼ D ( · | A ) N i . Consider now a realization N i of N i with

<!-- formula-not-decoded -->

Then by the law of total probability and definition of D ( · | A ) , we have that

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

where the last equality follows by the definition of D ( ·| A ) . Furthermore, by Lemma B.6, we have with probability at least 1 -δ/c u over S 1 ,i, ⊓ ⊓ A ∼ D ( · | A ) N i that

̸

<!-- formula-not-decoded -->

̸

where C ≥ 1 is the universal constant of Lemma B.6. We now bound each term, starting with the first. Now, max h ∈A ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) L S 1 ,i, ⊓ ⊓ A ( h ) is equal to max h ∈A ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) Σ = ( h, S 1 ,i, ⊓ ⊓ A ) /N i . And as any h ∈ A ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) is equal to h = A ( S ′ ) for some S ′ ∈ S ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) , we get that max h ∈A ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) Σ = ( h, S 1 ,i, ⊓ ⊓ A ) is equal to max S ′ ∈S ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) Σ = ( A ( S ′ ) , S 1 ,i, ⊓ ⊓ A ) . Furthermore, since any S ′ ∈ S ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) contains the training sequence S 1 ,i, ⊓ ⊓ A , we get that max S ′ ∈S ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) Σ = ( A ( S ′ ) , S 1 ,i, ⊓ ⊓ A ) ≤ max S ′ ∈S ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) Σ = ( A ( S ′ ) , S ′ ) .

̸

̸

̸

̸

Now we have that A ( S ′ ) is a ERM -algorithm run on S ′ , thus A ( S ′ ) ∈ H and any other hypothesis in H has a larger empirical error on S ′ than A ( S ′ ) , including h ⋆ . We thus have that max S ′ ∈S ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) Σ = ( A ( S ′ ) , S ′ ) ≤ max S ′ ∈S ( S 1 ,i, ⊔ ; S 1 ,i, ⊓ ) Σ = ( h ⋆ , S ′ ) , which, since S ′ is contained in S 1 ,i = ( S 1 ) i , can further be upper bounded by Σ = ( h ⋆ , S 1 ,i ) . We can thus conclude that

̸

̸

<!-- formula-not-decoded -->

Note we have used that | S 1 ,i | = m ′ / 3 3 . Now, using the fact that we have crystallized an event N i with N i ≥ (1 -1 / 1000) L D ( ̂ A t 2 ( S 2 )) m ′ /s ⊓ , we have that the first term of Equation (52) (after factoring out the multiplication) can be bounded as follows:

<!-- formula-not-decoded -->

where we, in the last inequality, have used that s ⊓ = 3 3 1 -1 27 . This concludes the bound on the first term in Equation (52).

Now, using that N i ≥ (1 -1 / 1000) L D ( ̂ A t 2 ( S 2 )) m ′ /s ⊓ , we get that the second term in Equation (52) can be bound as follows:

<!-- formula-not-decoded -->

In the second inequality we have used that s ⊓ = 3 3 1 -1 27 . We now use the assumption that we had realizations S 2 and t 2 of S 2 and t 2 in E 2 , i.e., such that L D ( ̂ A t 2 ( S 2 )) ≤ cτ + c ( d +ln( c u e/δ )) m ′ . This allows us to conclude that

<!-- formula-not-decoded -->

̸

̸

where we have used the inequality √ a + b ≤ √ a + √ b for a, b ≥ 0 in the last step.

Then, to summarize, we have seen that the event N i ≥ (1 -1 / 1000) P [ A ] m ′ /s ⊓ = (1 -1 / 1000) L D ( ̂ A t 2 ( S 2 )) m ′ /s ⊓ occurs with probability at least 1 -δ/c u , and that conditioned on this event, Equation (52) holds with probability at least 1 -δ/c u . Consequently, we have that with probability at least 1 -2 δ/c u over S 1 ,i each of Equation (52), Equation (53) and Equation (54) hold. Then, with probability at least 1 -2 δ/c u over S 1 ,i we have that

̸

<!-- formula-not-decoded -->

̸

Now, invoking a union bound over i ∈ { 1 , . . . , 27 } , we have that with probability at least 1 -54 δ/c u over S 1 , it holds that

̸

<!-- formula-not-decoded -->

̸

Furthermore, by Lemma B.3 and another union bound over S 1 , 1 , . . . , S 1 , 27 , we have that with probability at least 1 -27 δ/c u over S 1 ∼ D m ′ it holds that

<!-- formula-not-decoded -->

Thus, by applying the union bound over the events in Equation (56) and Equation (57), we get that with probability at least 1 -81 δ/c u over S 1 , it holds that

̸

<!-- formula-not-decoded -->

̸

Note that this suffices to give the event E G in Equation (50). We remark that we demonstrated the above for any realizations S 2 and t 2 of S 2 and t 2 in E 2 .

We now notice that by Theorem B.2, the fact that t 1 ≥ 4 · 243 2 ln (2 m/ ( δ ( d +ln(86 /δ )))) , and the choice of c u = 86 ), we have that P S 2 , t 2 [ E 3 ] ≤ δ/c u . Combining this with the conclusion below Equation (51) and Equation (58), and with the fact that S 1 , S 2 t 2 are independent, we have that

<!-- formula-not-decoded -->

Note that first equality follows from E 1 , E 2 , E 3 partitioning the outcomes of S 2 and t 2 and the first inequality follows from the conclusions below Equation (51) and Equation (58), which state that E G holds with probability 1 on E 1 and with probability at least 1 -81 δ/c u on E 2 . The second

inequality again follows from E 1 , E 2 , E 3 partitioning the outcomes of S 2 and t 2 and the bound P S 2 , t 2 [ E 3 ] ≤ δ/c u , which shows Equation (46).

We now proceed to show Equation (47), i.e., that with probability at least 1 -4 δ/c u over S 1 , S 2 , S 3 , t 1 , t 2 , it holds that

̸

<!-- formula-not-decoded -->

̸

̸

We denote this event E F . Towards proving the claim, we consider the following event over S 1 , S 2 , t 1 and t 2 ,

̸

<!-- formula-not-decoded -->

̸

As previously mentioned, we take c to be at least the constant of Theorem B.2 and to also satisfy c ≥ 2 · 10 6 s ⊓ .

Now, if we have realizations S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 in E 4 , then by monotonicity of measures, we obtain that

̸

<!-- formula-not-decoded -->

̸

̸

which would imply the event E F in Equation (59). Now consider a realization of S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 on the complement ¯ E 4 of E 4 . We then have that

̸

<!-- formula-not-decoded -->

̸

̸

We recall that S = 3 , are the examples in ( x, y ) ∈ S 3 for which

̸

<!-- formula-not-decoded -->

̸

̸

̸

In the following, let D ( · | =) be the conditional distribution of D given that avg = ( ̂ A t 1 ( S 1 ))( x, y ) ≥ 11 243 or avg = ( ̂ A t 2 ( S 2 ))( x, y ) ≥ 11 243 . That is, for an event B over ( X × Y ) , we have

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

Thus we have that S = 3 ∼ D ( B | =) . We now notice by S 3 ∼ D that

̸

̸

<!-- formula-not-decoded -->

̸

Using a Chernoff bound, this implies

̸

<!-- formula-not-decoded -->

̸

Note that we are also using the facts that

̸

<!-- formula-not-decoded -->

̸

̸

and that c ≥ 2 · 1000 2 s ⊓ . Now consider an outcome of N = = | S = 3 | where

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

Now, by Theorem C.2 we have that since h tie = A ( S = 3 ) and S = 3 ∼ D ( ·| =) , then with probability at least 1 -δ/c u over S = 3 ,

̸

̸

<!-- formula-not-decoded -->

̸

Note that the first inequality uses Theorem C.2 (and C ′ &gt; 1 is the universal constant of Theorem C.2), and the second inequality uses that h ⋆ ∈ H so it has error greater than the infimum. Now, using the law of total expectation, we have that

̸

̸

̸

<!-- formula-not-decoded -->

̸

We now bound each term in the above. First,

̸

<!-- formula-not-decoded -->

̸

̸

which is less than τ. Furthermore, for the second term, we have by

̸

̸

̸

<!-- formula-not-decoded -->

that

<!-- formula-not-decoded -->

̸

where the first inequality follows from plugging in that

̸

<!-- formula-not-decoded -->

̸

̸

in the denominator, and the second inequality follows from a union bound over the event

̸

<!-- formula-not-decoded -->

̸

Thus, we have shown that with probability at least 1 -2 δ/c u over S 3 , it holds that

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

which, in particular, also implies that with probability at least 1 -2 δ/c u over S 3

̸

<!-- formula-not-decoded -->

̸

Thus, since the above also upper bounds

̸

<!-- formula-not-decoded -->

̸

̸

for any realization S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 , and t 2 , on E 4 , (with probability 1 ), we conclude that for any realization S 1 , S 2 , t 1 , and t 2 of S 1 , S 2 , t 1 , and t 2 , it holds with probability at least 1 -2 δ/c u over S 3 that

̸

<!-- formula-not-decoded -->

̸

Let this event be denoted E 5 . Now, let E 6 be the event that

<!-- formula-not-decoded -->

which, by Theorem B.2, the fact that t 1 , t 2 ≥ 4 · 243 2 ln (2 m/ ( δ ( d +ln(86 /δ )))) , and a union bound, holds with probability at least 1 -2 δ/c u . (We let c &gt; 1 be a universal constant at least as large as the universal constant of Theorem B.2, and with c ≥ 2 · 10 6 s ⊓ .) Now, notice that for realizations S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 , on E 6 , it holds with probability at least 1 -2 δ/c u over S 3 that

̸

<!-- formula-not-decoded -->

̸

̸

where the first inequality follows by E 5 , the second inequality by E 6 , and the third inequality by √ a + b ≤ √ a + √ b. We notice that the above event is E F in Equation (59). Thus, by the above holding with probability at least 1 -2 δ/c u for any outcome of S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 on E 6 , and by S 1 , S 2 and S 3 being independent we conclude that

<!-- formula-not-decoded -->

Note that the equality follows from our previous reasoning, i.e., that for outcomes of S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 on E 6 , and outcomes S 3 of S 3 on E 5 , E F holds. The second inequality follows from the fact that E 5 holds with probability at least 1 -2 δ/c u over S 3 for any realizations S 1 , S 2 , t 1 and t 2 of S 1 , S 2 , t 1 and t 2 . The third inequality follows from the fact that E 6 holds with probability at least 1 -2 δ/c u for S 1 , S 2 , t 1 and t 2 , which concludes the proof of Equation (47), as desired.

̸

̸

## D Best-of-both-worlds Learner

In this section we demonstrate that splitting a training sample S ∼ D m into { S i ∼ D m/ 3 } i ∈ [3] followed by running the algorithm of Hanneke et al. (2024) on S 1 , running the algorithm of Theorem B.2 on S 2 , and selecting the one h min with smallest empirical error on S 3 gives the following error bound:

<!-- formula-not-decoded -->

In pursuit of the above, recall the error bound of (Hanneke et al., 2024, Theorem 3), which establishes the existence of a learner ˜ A which with probability at least 1 -δ over S ∼ D m incurs error at most

<!-- formula-not-decoded -->

for a universal constant c ′ ≥ 1 . We will in the following use ̂ A t to denote the algorithm of Theorem C.1. In what follows let m ′ = m/ 3 .

By invoking the previous bounds on ̂ A t and ˜ A with δ = δ/ 4 , and further employing a union bound, we have that with probability at least 1 -δ/ 2 over S 1 and S 2 , both ˜ A and ̂ A t will emit hypotheses satisfying:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c is the universal constant of Theorem C.1. Let G denote the event that S satisfies the previous condition, which, as we have noted, has probability at least 1 -δ/ 2 .

Now consider realizations S 1 and S 2 of S 1 and S 2 with ( S 1 , S 2 ) ∈ G . Further, let h min = arg min h ′ ∈{ ˜ A ( S 1 ) , ̂ A ( S 2 ; ∅ ) } ( L S 3 ( h ′ )) . We now invoke Lemma B.3 on S 3 with the classifiers ˜ A ( S 1 ) and ̂ A t ( S 2 ; ∅ ) (and failure probability δ/ 8 ) along with a union bound to see that with probability at least 1 -δ/ 2 over S 3 , we have that both choices of h ∈ { ˜ A ( S 1 ) , ̂ A ( S 2 ; ∅ ) } satisfy

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Note that the final inequality follows from the fact that √ ab ≤ a + b for a, b &gt; 0 . Then, using the previous inequalities along with the definition of h min , we have that:

<!-- formula-not-decoded -->

In particular, the first inequality follows from Equation (62) applied to h min , the second from the definition of h min , the third from Equation (63), and the fourth from the fact that √ a + b ≤ √ a + √ b for a, b &gt; 0 . Now using the equations in Equation (60) we have that

<!-- formula-not-decoded -->

since √ ab ≤ a + b for a, b &gt; 0 . (We are further assuming that m = Ω( d ) , a standard condition.) Then, using Equation (68) and the fact that √ a + b ≤ √ a + √ b for a, b &gt; 0 , we have

<!-- formula-not-decoded -->

Using the previous inequality along with Equation (60), we have that

<!-- formula-not-decoded -->

Similarly, using Equation (61) and √ ab ≤ a + b we have that

<!-- formula-not-decoded -->

Again using the fact that √ a + b ≤ √ a + √ b for a, b &gt; 0 , we have that

<!-- formula-not-decoded -->

Applying Equation (61) then yields

<!-- formula-not-decoded -->

Now plugging in the above expressions into Equation (64) which is exactly the minimum of ˜ A ( S 1 ) and ̂ A t ( S 2 ; ∅ ) we arrive at

<!-- formula-not-decoded -->

with probability at least 1 -δ/ 2 over S 3 for any realizations S 1 and S 2 of S 1 and S 2 in G , now let T denote the event of Equation (69), then we have by independence of S 1 , S 2 , S 3 that

<!-- formula-not-decoded -->

where the first inequality follows by independence of S 1 , S 2 , S 3 and G only depending upon S 1 , S 2 and the second inequality by Equation (69) holding (the event of T ) with probability at least 1 -δ/ 2 over S 3 for any S 1 , S 2 in G, . Lastly, note that

<!-- formula-not-decoded -->

due to G having probability at least 1 -δ/ 2 over S 1 , S 2 by Equation (60), which concludes the proof.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The error bound of theorem 1.4, presented in the introduction, is the main contribution of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss how there is still a small gap in the known lower bounds and upper bounds after our new result.

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

Justification: Each statement in the paper is proven.

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

Justification: No experiments are included in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- 4.1 If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- 4.2 If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- 4.3 If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- 4.4 We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: No experiments are included in the paper.

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

Justification: No experiments are included in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: No experiments are included in the paper.

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

Justification: No experiments are included in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have no concerns about our work violating the NeurIPS code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The work is theoretical and the results hold under a mathematical model.

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

Justification: We do not release any model or datasets.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not have any code or dataset.

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

Justification: We do not release any code or dataset.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We do not have any experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We have not experiments, so it does not apply.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not employ LLMs for any important or original components of the paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
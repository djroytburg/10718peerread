## Are Greedy Task Orderings Better Than Random in Continual Linear Regression?

Matan Tsipory ∗ †

Ran Levinstein ∗ †

Itay Evron ∗ †

Mark Kong ∗ ‡

Deanna Needell ‡

Daniel Soudry †

## Abstract

We analyze task orderings in continual learning for linear regression, assuming joint realizability of training data. We focus on orderings that greedily maximize dissimilarity between consecutive tasks, a concept briefly explored in prior work but still surrounded by open questions. Using tools from the Kaczmarz method literature, we formalize such orderings and develop geometric and algebraic intuitions around them. Empirically, we demonstrate that greedy orderings converge faster than random ones in terms of the average loss across tasks, both for linear regression with random data and for linear probing on CIFAR-100 classification tasks. Analytically, in a high-rank regression setting, we prove a loss bound for greedy orderings analogous to that of random ones. However, under general rank, we establish a repetition-dependent separation. Specifically, while prior work showed that for random orderings, with or without replacement, the average loss after k iterations is bounded by O (1 / √ k ) -weprove that single-pass greedy orderings may fail catastrophically, whereas those allowing repetition converge at rate O (1 / 3 √ k ) . Overall, we reveal nuances within and between greedy and random orderings.

## 1 Introduction

Continual learning is a subfield of machine learning in which a learner is exposed to tasks or datasets sequentially. In such setups, only a single task is fully accessible at any given time, due to, for instance, computational limitations, data retention or privacy constraints, or the temporal nature of the environment. While much of the continual learning research focuses on mitigating forgetting or improving transfer, the role of task ordering is not yet fully understood.

Understanding how task order affects learning and what characterizes optimal orderings is important for both theoretical and practical reasons. Such understanding can illuminate failure modes, clarify the interplay between forgetting and transfer, and guide the design of continual environments and algorithms. Furthermore, it can inform active control over task sequences in settings that permit it ( e.g., robotic environments), situating the problem at the intersection of continual learning, multitask learning, curriculum learning, and active learning. This line of inquiry raises impactful computational and financial questions in the era of large language models and foundation models:

- Can task ordering by itself mitigate forgetting, even under vanilla continual training?
- What constitutes an 'optimal' task ordering?
- Is it better to learn when adjacent tasks are similar or dissimilar?
- Can greedy strategies systematically outperform random task orderings?

∗ Equal contribution.

† Technion, Haifa.

‡ University of California, Los Angeles.

One compelling direction in the continual learning literature is the design of task orderings informed by task similarity. This idea appears in several earlier works, with varying degrees of emphasis and differing motivations [e.g., 56, 79, 90, 89, 71, 44, 63, 72]. Most closely related to our work is Bell and Lawrence [14], who were among the first to explicitly and systematically examine such orderings in continual learning. They hypothesized that optimal performance would arise when adjacent tasks are similar . Surprisingly, they empirically found the opposite-orderings with dissimilar adjacent tasks led to better performance. More recently, Li and Hiratani [62] reached a similar conclusion and further proposed arranging tasks from the least to the most 'typical'. While these studies are thoughtprovoking, they are either empirical [71, 14, 72], based on restrictive data assumptions [63, 62], or focused solely on task-incremental settings [79], with some of their findings appearing inconclusive or contradictory. This underscores the need for a more rigorous theoretical understanding.

To this end, we formalize 'similarity-guided' orderings through greedy task selection, leveraging tools from related fields. Building on a projection-based perspective of continual learning [33, 34], we introduce two greedy orderings-Maximum Distance and Maximum Residual-commonly studied in the Kaczmarz [75, 73] and projection methods literatures [2, 41]. Using these orderings, we build a geometric intuition for greediness, as illustrated in Figure 1. We then develop both analytical and empirical insightse.g., by experimenting on random and CIFAR-100 tasks and proving optimality results for high-rank tasks.

Surprisingly, under general tasks, although without-replacement random orderings are known to converge [35, 10], we show that single-pass greedy orderings may fail catastrophically. We further prove that this drawback is resolved under greedy orderings with repetition, for which we establish a dimensionality-independent convergence bound. Finally, we propose a hybrid scheme combining greedy and random orderings, highlighting its empirical and analytical benefits.

We hope that the theoretical foundations-perspectives, tools, and findings-laid out in this paper will inspire future work on task orderings and their potential to mitigate catastrophic forgetting.

## Summary of our contributions.

1. We formalize similarity-guided orderings in continual linear regression via greedy strategies, drawing on tools and intuitions from the Kaczmarz and projection literature (Section 3).
2. We empirically demonstrate that greedy orderings converge faster than random orderings, both on synthetic regression tasks and on CIFAR-100 -based classification tasks (Section 4.1).
3. We prove optimality and convergence guarantees for high-rank tasks (Section 4.2).
4. For general-rank data, we design adversarial task collections in which single-pass greedy orderings provably induce catastrophic forgetting, i.e., yield an Ω(1) loss even as T →∞ (Section 5.1).
5. In contrast, we prove an O (1 / 3 √ k ) upper bound for greedy orderings with repetition (Section 5.2).
6. We combine greedy and random orderings into a hybrid strategy that performs well empirically and inherits the bounds of random orderings, avoiding greedy failure modes (Section 5.3).

(a) A greedy ordering with dissimilar adjacent tasks.

<!-- image -->

(b) A greedy ordering with similar adjacent tasks.

<!-- image -->

Figure 1: Intuition. Consider a collection of jointly-realizable linear regression tasks ( e.g., A,B,C,D) . Each task has an affine solution space ( e.g., where X A w = y A ), and w ⋆ is an 'offline' joint solution at the intersection of all tasks. Employing a projection perspective on learning in continual models [33, 34], we see that transitions between dissimilar tasks ( A → D → B → C ) intuitively lead to faster convergence toward the intersection compared to transitions between similar tasks ( A → B → C → D ).

## 2 Setting and Background: Continual linear regression

We focus on continual linear regression, common in theoretical continual learning [e.g., 29, 7, 33, 35, 63, 78, 38, 46]. This setting, though simple, already gives rise to key continual learning phenomena, such as complex interactions between forgetting, task similarity, and overparameterization [see 39].

Notation. We reserve bold symbols for matrices and vectors, e.g., X , w . We use ∥·∥ to denote the Euclidean norm of vectors and the spectral (L2) norm of matrices. X + denotes the Moore-Penrose pseudoinverse of a matrix. Finally, we denote [ n ] = 1 , . . . , n .

Formally, the learner is given access to a task collection of T linear regression tasks, i.e., ( X 1 , y 1 ) , . . . , ( X T , y T ) where X m ∈ R n m × d , y m ∈ R n m . We denote the data 'radius' by R ≜ max m ∈ [ T ] ∥ X m ∥ . For k iterations, the learner sequentially learns the tasks according to a task ordering τ : [ k ] → [ T ] , which-as this paper shows-can be crucial in continual learning.

Scheme 1 Continual linear regression (to convergence)

Initialize w 0 = 0 d

For each iteration t = 1 , . . . , k :

w t ← Start from w t -1 and minimize the current task's loss L τ ( t ) ( w ) ≜ ∥ ∥ X τ ( t ) w -y τ ( t ) ∥ ∥ 2 with (S)GD to convergence

Output w k

We assume throughout the paper that there exist offline joint solutions that perfectly solve all T tasks jointly . This assumption is common 4 in many theoretical continual learning papers and facilitates the analysis [e.g., 33, 34, 35, 54, 39, 51]. Moreover, it naturally holds in highly overparameterized models and is thus linked to the linear dynamics of deep networks in the neural tangent kernel (NTK) regime [see 49, 23].

̸

Assumption 2.1 (Joint Linear Realizability of Training Data) . Assume the intersection of all task solution subspaces is non-empty, i.e., W ⋆ ≜ ⋂ T m =1 W m ≜ ⋂ T m =1 { w ∈ R d ∣ ∣ ∣ X m w = y m } = ∅ .

We focus on the joint solution with the minimum norm, often linked to improved generalization.

Definition 2.2 (Minimum-norm joint solution) . Denote specifically w ⋆ ≜ argmin w ∈W ⋆ ∥ w ∥ .

We follow prominent theoretical work [e.g., 29, 33, 34, 39, 35] and study the model's ability to not 'forget' previously seen data, and accumulate expertise on the training data (of all tasks). This focus isolates continual dynamics from statistical generalization effects that also arise in non-continual, stationary settings.

Definition 2.3 (Average loss) . The (training) loss of an individual task m ∈ [ T ] is defined as L m ( w ) ≜ ∥ X m w -y m ∥ 2 . The loss we analyze is the average across all T tasks, which in our realizable setting takes the following form:

<!-- formula-not-decoded -->

where we also normalize by the generally unavoidable scaling factors ∥ w ⋆ ∥ and R ≜ max m ∈ [ T ] ∥ X m ∥ .

Remark 2.4 (Forgetting vs. loss) . An alternative quantity considered in continual learning [20, 33] is the forgetting , defined as the loss degradation at iteration k across previously seen tasks only , i.e., 1 k ∑ k t =1 ( L τ ( t ) ( w k ) -L τ ( t ) ( w t ) ) , or simply as 1 k ∑ k t =1 ∥ ∥ X τ ( t ) w k -y τ ( t ) ∥ ∥ 2 in our realizable setting. Since we mostly focus on single-pass orderings where each task is seen once, the forgetting ultimately coincides with the average loss. Thus, we ease presentation and study the average loss.

4 A different trend in continual learning theory is to assume an underlying linear model, like we do, but allow additive label noise [e.g., 38, 63, 109, 28, 60, 61]. However, this comes at the cost of strong assumptions on the features -e.g., commutable covariance matrices or i.i.d. features across tasks. To some extent, the analysis in Section 5.1 of Evron et al. [33] suggests that, under such assumptions, task ordering has limited impact. Thus, it may not be a suitable starting point for studying similarity-guided orderings, in contrast to our assumption.

Another insightful quantity is the distance to w ⋆ .

Definition 2.5 (Distance to the joint solution) . After k iterations, the (squared) distance is,

<!-- formula-not-decoded -->

This distance upper bounds the loss, as can be shown using simple norm inequalities.

Proposition 2.6 (Linking the quantities) . After k iterations of Scheme 1 on jointly realizable tasks, the loss is upper bounded by the distance to the joint solution.

<!-- formula-not-decoded -->

In some cases, the distance remains large while the loss vanishes, showing that converging to w ⋆ is not mandatory for continual learning [33]. Focusing on the loss paves the way to universal convergence, independent of the problem's complexity, e.g., its condition number [85].

Geometric interpretation to learning. In each iteration of Scheme 1, the learner minimizes the squared loss of the current task to convergence. 5 Each iterate w t of this scheme above is known [33] to implicitly follow the following closed-form update rule,

<!-- formula-not-decoded -->

Conveniently, in our realizable setting, this update rule admits an intuitive geometric interpretation.

Evron et al. [33] identified an orthogonal projection operator,

<!-- formula-not-decoded -->

which we use for analysis only (Scheme 1 never explicitly computes pseudoinverses or SVDs).

Under the realizability assumption, y τ ( t ) = X τ ( t ) w ⋆ . We plug it into Eq. (1) and obtain:

<!-- formula-not-decoded -->

Figure 2: Projection dynamics.

<!-- image -->

Geometrically, w t -1 is projected by an affine projection onto the solution space of task τ ( t ) . In our paper, we adopt this projection-based perspective-proven useful in theoretical work on continual learning [33, 34]-to build intuition about greedy orderings.

## 3 Greedy task orderings: A formal approach and intuition

As discussed in Section 1, the learning order plays a crucial role in the dynamics of many machine learning settings. This phenomenon has also been observed in continual learning, both analytically and empirically. Several works have proposed leveraging 'similarity-guided' task orderings-placing dissimilar tasks consecutively. However, the existing literature still lacks the rigor and analytical tools needed to fully understand such orderings. To address this gap, this section draws on connections between continual linear regression and other research areas to formalize greedy task orderings and develop the mathematical tools necessary to study them.

Geometric intuition. As illustrated in Figure 2, the projection perspective allows us to decompose ∥ w t -w ⋆ ∥ 2 using projection properties and the Pythagorean theorem as:

<!-- formula-not-decoded -->

Thus, to try and minimize ∥ w t -w ⋆ ∥ 2 , one could greedily maximize ∥ ∥ ( I -P τ ( t ) )( w t -1 -w ⋆ ) ∥ ∥ 2 .

5 This simplifies the analysis; other choices exist as well, e.g., a fixed number of steps per task [51, 59].

<!-- formula-not-decoded -->

This has inspired a myriad of studies on Kaczmarz 6 and projection methods [e.g., 2, 75, 16] which employed ordering schemes that greedily maximize ∥ w t -1 -w t ∥ 2 , in the following spirit.

Definition 3.1 (Maximum Distance Ordering) . Greedily maximize the distance between iterates, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Our earlier Figure 1a illustrates the MD ordering and how it leads to faster convergence to w ⋆ .

Distance and task similarity. The distance between iterates w τ ( t -1) and w τ ( t ) reflects some angle between the affine solution subspaces of their corresponding tasks, and more generally, relates to the principal angles between these subspaces [33]. These angles can be used to quantify task similarity, as illustrated in the setting of Section 4.2 and Figure 1.

An alternative greedy ordering found in the literature is the Maximum Residual ordering [e.g., 2, 40, 75, 106]. This rule is easier to compute in full, or to estimate using a small validation set.

Definition 3.2 (Maximum Residual Ordering) . Greedily select the task exhibiting the greatest error:

<!-- formula-not-decoded -->

Notice that the MD and MR orderings are related since X m = X m X + m X m = X m ( I -P m ) , and,

<!-- formula-not-decoded -->

Single-pass orderings. Our paper mostly focuses on 'single-pass' greedy orderings, where each task is encountered exactly once. Although disallowing repetitions departs slightly from the motivating literature on projection methods, it is the more common-and arguably more natural-setup in continual learning [see 14, 62]. Even in curriculum or multitask learning, limiting tasks to a single pass can reduce training costs. Nonetheless, in Section 5.2, we examine the effect of repetitions.

Computational tractability of greedy policies. As explained above, the benefits of greedy orderings are intuitive. The cost of computing the greedy rules in Definition 3.1 and Eq. (4), of course, introduces a tradeoff between convergence rate and overall computational cost. Before continuing our investigation of these orderings, we briefly address their computational feasibility.

- (i) Estimation: Greedy rules can often be estimated efficiently in practical scenarios. For example, the maximum residual rule (Definition 3.2) requires the current loss of each available task. This quantity can be approximated using a small subset of samples or via dimensionality reduction techniques [e.g., see 31]. In App. C, we show empirically that the maximum residual rule performs comparably to the maximum distance rule, and that its performance remains unharmed even when approximated using only 1% of the data. In deep networks, computing that rule requires only forward passes and may reduce the number of gradient steps-thereby lowering overall time and memory costs by limiting costly backward passes [47].
- (ii) Heuristics : Our greedy rules rely on residuals to quantify the similarity between the current task and the remaining ones. This approach is exemplified in Figure 1 and Eq. (5) of Section 4.2, and is related to principal angles between subspaces [see 33]. Alternatively, one could utilize heuristic notions of task similaritye.g., predefined [56] or computed metrics that use Hessians [14], zero-shot performance [62], or task embeddings [1, 71].
- (iii) Structured tasks : If each step updates relatively few residuals ( e.g., in a Kaczmarz setting with sparse columns and rows), only few residuals must be recomputed [75].

6 The Kaczmarz method [52, 32], further explained in App. A, iteratively solves a linear system of equations.

7 In practice, the MD rule is easy to compute for rank-1 tasks, since it reduces to 1 ∥ x m ∥ 2 ∥ ∥ x ⊤ m w t -1 -y m ∥ ∥ 2 . In higher ranks, this rule is harder to compute exactly-but can be estimated, e.g., with a subsample of the task.

## 4 Benefits of greedy orderings

A natural competitor to greedy strategies is the random strategy, uniformly sampling tasks (rows or blocks in the Kaczmarz context) from the task collection [ T ] . That is,

<!-- formula-not-decoded -->

As mentioned earlier, greedy strategies have a long-standing history in the Kaczmarz method [75, 76] and its block variants [73, 68, 108, 106, 102] employing either deterministic [75] or probabilistic [11, 12, 107, 95] selection rules. In this context, greedy orderings often achieve better provable bounds on the distance to w ⋆ (Definition 2.5), compared to random orderings. In contrast, our focus-and that of related continual learning literature [e.g., 33, 34, 54, 51]-centers on convergence of the loss (Definition 2.3). While the loss is upper bounded by the distance to w ⋆ (Proposition 2.6), the existing Kaczmarz rates fall short in two ways: (i) they rely on repeating rows/blocks and do not apply in the single-pass regime central to continual learning; and (ii) even when repetition is allowed, their rates depend on data eigenvalues, potentially making them much slower than our data-independent O (1 / 3 √ k ) guarantee of Theorem 5.3.

In this section, we examine how well the advantages of greedy over random orderings carry over to the loss in continual settings-first empirically, then analytically.

## 4.1 Motivating experiments: Greedy outperforms random ordering

Here, we test different task ordering strategies on synthetic regression tasks and a more complex classification setup-using linear probing in a domain-incremental CIFAR-100 setting. 8

Regression tasks: Random data. The feature matrices X 1 , . . . , X T are drawn from a Gaussian distribution. We compare the two 'dissimilarity-maximizing' greedy strategies (MD, MR) to the random ordering (Eq. (4)) and a complementary minimum distance strategy (defined as in Definition 3.1, replacing argmax with argmin ). Full details, including more combinations of task count T , dimension d , and rank r , as well as experiments on anisotropic data, are provided in App. B.

Classification tasks: CIFAR-100 . Werandomly partition classes into continual binary classification tasks, similarly to Li and Hiratani [62]. We train a linear probe on top of a ResNet-20 embedder, pretrained on the original CIFAR-100 multiclass task [45, 55]. We optimize the cross-entropy loss of each task while employing L2 regularization towards the previous parameters [65]. We compare the performance to a 'joint' baseline, trained on all tasks together (not continually). See App. C for full details. There, we conduct further experiments on continual CIFAR-100 tasks as before, but using embeddings pretrained on (i) CIFAR-10 and (ii) CIFAR-100 with only half of the samples from each class (in this case, continual learning is performed on tasks formed from the other half). We also examine more computationally efficient greedy orderings, determined from only a fraction of the data-down to 1% (5 samples per class in CIFAR-100 ).

<!-- image -->

- (a) Regression (random data): Average loss over T = 50 regression tasks of rank r = 10 in d = 100 dimensions, sampled from an isotropic Gaussian distribution. Details in App. B.
- (b) Classification ( CIFAR-100 ): Average test accuracy over T = 50 binary classification tasks, generated by randomly partitioning CIFAR-100 classes (a domain-incremental setting). Details in App. C.

<!-- image -->

Figure 3: Task ordering comparison. Transitioning between dissimilar tasks consistently outperforms random transitions, with Greedy MR and MD achieving comparable performance.

8 We provide a code snippet for the regression experiments in App. H. The code for the classification experiments is accessible at https://github.com/matants/greedy\_ordering .

## 4.2 Provable benefits for high-rank, 'nearly determined' tasks

To further motivate greedy orderings, we analyze a simple setup where each task's matrix is of nearly full rank, i.e., rank( X m ) = d -1 , ∀ m ∈ [ T ] . Even in such a setup, it has been shown that arbitrary orderings of T →∞ may lead to catastrophic forgetting and maximal losses [33].

In this setup, each projector can be expressed as a rank1 operator, i.e., P m = I -X + m X m = v m v ⊤ m for a unit vector v m ∈ R d that spans the solution space of task m . Then, we can rewrite the Maximum Distance (Definition 3.1) rule to explicitly maximize dissimilarity between consecutive tasks, i.e.,

<!-- formula-not-decoded -->

Optimality of greedy orderings in terms of distance to w ⋆ . Earlier in Eq. (3), we motivated the MD ordering as greedily maximizing the decrease in ∥ w t -w ⋆ ∥ . Does this guarantee a minimal distance ∥ w T -w ⋆ ∥ at the end of the sequence? Here, we prove that the MD ordering yields a square-root approximation of the optimal distance at the end of learning. 9

Lemma 4.1 (Optimality guarantee when r = d -1 ) . Let w τ MD T and w τ ⋆ T be the iterates after learning T jointly realizable tasks of rank d -1 under the Maximum Distance ordering τ MD and an optimal ordering τ ⋆ that leads to a minimal distance to the joint solution w ⋆ . Then, their distances hold,

<!-- formula-not-decoded -->

The full proofs for this section are given in App. D.

What about the loss? The optimality of the distance does not imply optimality of the average loss, as exemplified in Figure 7 in Section 6. Instead, we now derive an upper bound for the loss.

Lemma 4.2 (Loss bound when r = d -1 ) . Under the Maximum Distance greedy ordering over T jointly-realizable tasks of rank d -1 , the loss of Scheme 1 after T iterations is upper bounded as,

<!-- formula-not-decoded -->

This rate matches a recent O ( d -r T ) bound for random orderings without replacement [35], whereas the best known rate for such orderings with general-rank tasks is O ( 1 / √ T ) [10]. This raises the question: for general-rank tasks, can single-pass greedy orderings still compete with random ones, or even outperform them, in worst-case analysis? Next, we show that they cannot.

## 5 Failure modes and surprises in greedy orderings

Under random orderings, with or without replacement, Attia et al. [10] proved a universal, dimensionality-independent rate of E τ Unif L ( w τ Unif k ) ≤ 13 / √ k . Surprisingly, we prove a clear separation in our setup: while single-pass greedy orderings can fail catastrophically , i.e., not decrease with the number of iterations k = T , greedy orderings with repetition enjoy a bound of O ( 1 / 3 √ k ) .

## 5.1 Greedy orderings can fail where random ones do not

We now present cases where the single-pass greedy ordering forgets catastrophically , i.e., suffers an Ω(1) loss, even after fitting a collection of T →∞ tasks. Specifically, we present an example in d = 3 dimensions where the loss does not diminish, and a construction in d = T +1 dimensions that exploits dimensionality to yield maximal forgetting. Full details and proofs are given in App. E.

9 Our optimality result is related to the optimal Hamiltonian path in a predefined similarity graph, as studied in related work [14, 62]. Specifically, in this section's high-rank case, similarity can be statically defined as s m,m ′ = cos 2 ( θ m,m ′ ) . Then, the greedy MD ordering approximates the Hamiltonian path τ ⋆ that maximizes ∏ T t =2 s τ ⋆ ( t -1) ,τ ⋆ ( t ) [see 37, 70]. However, under general-rank tasks, our greedy rules (Def. 3.1 and 3.2) are computed online at each iteration and depend not only on the previous task τ ( t -1) but also on the previous iterate w t -1 . Consequently, these rules do not correspond to a Hamiltonian path on any predefined graph.

Example 5.1 (Adversarial 3d construction) . For all T ∈ { 4 · 10 i -1 | i = 1 , 2 , . . . , 7 } , there exists a task collection of jointly-realizable tasks in d = 3 , such that L ( w τ MD T ) , L ( w τ MR T ) &gt; 2 . 78 · 10 -5 .

Theorem 5.2 (Greedy lower bound) . For any d ≥ 30 , there exists an adversarial task collection with T = d -1 jointly-realizable tasks (of different ranks) such that both greedy orderings (MD, MR) forget catastrophically . That is, the loss at the end of the sequence is, L ( w τ MD T ) , L ( w τ MR T ) ≥ 1 8 -1 4 d .

We demonstrate the behavior of an adversarial task collection using T = 999 tasks in d = 1000 dimensions. Our constructed collection 'tricks' the greedy orderings: slowly increasing not only the loss on all tasks, but also the forgetting of previous tasks. The model is thus unable to accumulate knowledge and practically forgets everything it learns.

Figure 4: Learning an adversarial collection.

<!-- image -->

## 5.2 Single-pass vs. repetition in greedy orderings

So far, we have focused on single-pass greedy orderings, in which each task is learned exactly once. These are conceptually related to without-replacement sampling and (re)shuffling techniques in SGD and the Kaczmarz method, where repetition-free strategies often converge faster than withreplacement sampling, both empirically [17, 76, 96] and in theory [69, 42, 15, 50, 43; but see 84, 26]. We ask: Does the advantage of orderings without repetition extend to greedy orderings?

Now, we show that repetition in greedy orderings avoids the failure mode of single-pass ones.

Theorem 5.3 (Dimensionality-independent bound for greedy orderings with repetition) . Under a Maximum Distance greedy ordering with repetition ( τ MD -R ) over T jointly-realizable tasks, the loss of Scheme 1 after k ≥ 2 iterations is upper bounded as L ( w τ MD -R k ) = O ( 1 / 3 √ k ) .

App. F provides the proof, a comparison to prior rates (Table 1), and details for the experiment below.

We evaluate the effect of repetition across orderings under random data. As in prior work, random sampling without replacement outperforms with replacement. In contrast, repetition benefits greedy orderings, likely due to larger updates and faster convergence to w ⋆ . The slowdown in the single-pass case likely reflects the exhaustion of high dissimilarities.

Intuitively, repetition in random orderings exposes the learner to less data, while in greedy selection it allows considering all tasks at each step.

Figure 5: The effect of repetitions.

<!-- image -->

However, these findings do not always hold, as seen in our classification example (App. C.5).

## 5.3 Extension: Hybrid task orderings

To leverage both the fast empirical convergence of greedy orderings and the analytical convergence guarantees of random ones, we introduce a 'hybrid' strategy: begin with greedy selection and switch to random once the decrements ∥ w t -1 -w t ∥ 2 fall below a threshold. Analytically, using a suitable threshold, we prove in Lemma G.1 that any bound for without-replacement random orderings, e.g., O (1 / √ T ) [10], extends to our hybrid scheme, showing it avoids the failure mode of Section 5.1.

Empirically, the hybrid ordering performs better than random but worse than greedy. This matches our intuition from Eq. (3) and Figure 1a: greedy selection takes larger 'steps' (or projections), particularly early on, when most tasks are still available. Once these projections diminish, we switch to the random ordering, which-unlike the greedy approach-cannot be adversarially 'tricked' into failure.

Further details and experiments appear in App. G.

Figure 6: Hybrid ordering experiment.

<!-- image -->

## 6 Discussion and related work

So far, we have studied greedy task orderings, demonstrating empirical and analytical benefits of transitioning between dissimilar tasks. Here, we expand on connections and ideas not yet fully covered to better situate our work within the existing literature. In App. A, we discuss further links to Kaczmarz methods, curriculum and active learning, coordinate descent, and example selection in SGD.

Task orderings in continual learning theory. Continual learning theory often treats task orderings as arbitrary. However, several analytical works [e.g., 33, 34, 35, 54, 51, 19] show that certain orderings-typically cyclic or random-can mitigate forgetting (matching empirical findings [58]). While some works downplay ordering effects-arguing they are often minor-and defer their study to future work [92], others design continual learning algorithms specifically for evolving sequences with very similar adjacent tasks [6]. We follow a different line of work that studies how pairwise task similarities, or dissimilarities , influence continual schemes.

A particularly relevant work by Bell and Lawrence [14] advocates pairwise task dissimilarity as a guiding principle and was among the first to empirically investigate similarity-guided task orderings. Tasks are represented as vertices in a complete graph, where edge weights correspond to a predefined distance between tasks, and Hamiltonian paths represent full task orderings. They hypothesized that a minimum-weight path (favoring similar tasks in succession) would yield the best continual performance. Yet their experiments on simple neural networks indicated the opposite: maximumweight paths, placing dissimilar tasks adjacently, often led to improved performance. Still, these results were not always statistically significant (see their Figure 5)-motivating us to revisit similarityguided task orderings from a more analytical perspective.

Li and Hiratani [62] conduct a deeper investigation into similarity-guided task orderings, obtaining more statistically robust empirical results. They likewise find that adjacent tasks should be dissimilar , and further explore task 'typicality' (discussed below). While deriving results for a linear regression model to support their empirical observations (on neural networks), they rely on a restrictive analytical data model in which features are randomly drawn from a simplified distribution across tasks. In contrast, our analysis accommodates arbitrary feature matrices, allowing richer and more realistic forms of task similarity. Like Bell and Lawrence [14], their goal is to characterize optimal orderings in general, whereas we formalize and analyze greedy orderings specifically, both as a practical strategy and as a proxy for optimal ones.

Ruvolo and Eaton [88] propose an 'information maximization' approach to task ordering, using a diversity-based heuristic related to our maximum residual strategy (Definition 3.2). However, their complex model limits rigorous theoretical analysis of the kind we provide.

Lin et al. [63] examine the role of task similarity and reach conclusions broadly aligned with ours. While influential, their work differs from ours in several ways. First, their analysis relies on a restrictive i.i.d. feature assumption across tasks. They also assume a distinct teacher model per task, unlike our setting, where all tasks share a single overparameterized model-as is common in modern deep learning. Consequently, their notion of task similarity relies on teacher similarity, rather than more practical measures such as residuals (as in Definition 3.2) or feature similarity. Although they note ordering effects in their expressions and briefly support them with classification experiments, task ordering is not their primary focus. In contrast, we offer a comprehensive treatment of similarity-guided orderings specifically-providing formal definitions, geometric intuitions, greedy strategies, optimality results, empirical validation, failure modes, and repetition analysis.

Can similarity-guided task orderings alone mitigate forgetting? Methods such as replay, regularization, and parameter isolation are widely used to mitigate forgetting in continual learning [53, 59, 83, 21, 87, 27]. However, they depart somewhat from standard ('vanilla') deep learning practices that apply plain gradient methods to the (current) loss. Interestingly, both our work and prior studies show that even without such mechanisms, task ordering alone strongly affects forgetting. For instance, simply randomizing the task order-with or without replacement-is known to alleviate forgetting [33, 35, 10, 58]. In contrast, we show that single-pass greedy ordering can exacerbate forgetting (Section 5.1), while allowing task repetitions mitigates this effect (Section 5.2). Moreover, ordering strategies can be combined with other approaches; for example, in our classification experiments we also employ regularization (Section 4.1). This underscores the importance of studying task ordering as a simple, complementary way to mitigate forgetting, while potentially keeping continual learning closer to standard deep learning practice.

Task typicality at the end of learning. Li and Hiratani [62] suggest that tasks should be arranged A from least to most 'typical'. While we did not focus on this aspect of orderings, our geometric interpretation can illustrate it.

Our motivation was to minimize the distance ∥ w k -w ⋆ ∥ 2 , which upper bounds the loss 1 T ∑ T m =1 ∥ X m ( w k -w ⋆ ) ∥ 2 . However, this bound can be loose, and minimizing the distance does not guarantee the lowest loss. For example, in Figure 7, although ∥ w A -w ⋆ ∥ 2 = ∥ w C -w ⋆ ∥ 2 , the point w C is a better ending point than w A , inducing a lower loss (the arrows represent the residuals). This happens because task C is more typicali.e., more similar to other tasks-than task A .

Figure 7: Task typicality.

<!-- image -->

The empirical advantage of greedy ordering may stem from a tendency to postpone typical tasks, perhaps causing its benefits to emerge only in later stages of training (see Figure 3b).

Regret today or loss tomorrow? In Section 3, we motivated the use of greedy orderings to minimize the distance to the joint solution ∥ w k -w ⋆ ∥ 2 , which in turn upper-bounds the average loss over all tasks: 1 T ∑ T m =1 ∥ X m ( w k -w ⋆ ) ∥ 2 . This objective is related, but not identical, to the notion of regret , which quantifies the loss along the optimization path on consecutive tasks, i.e., 1 k ∑ k t =1 ∥ X τ ( t ) ( w t -1 -w ⋆ ) ∥ 2 . From this definition and Figure 1, we observe that regretthough also upper-bounded by the distances ∥ w t -1 -w ⋆ ∥ 2 -can often benefit from transitions between similar tasks rather than dissimilar ones. In other words, to make accurate predictions during learninge.g., in decision-making-transitioning between similar tasks may be preferable. Conversely, to minimize average loss across taskse.g., in curriculum or multitask learning-our findings suggest that transitioning between dissimilar tasks is preferable.

Other continual setups. The majority of studies support our conclusion that sequential task dissimilarity is beneficial [e.g., 88, 14, 71, 81, 33, 63, 66, 91]. Still, the specific continual setup can dramatically influence the behavior of task orderings. We consider a 'domain-incremental' setting: learning the same problem across different domains, i.e., P ( X ) changes but P ( Y | X ) is fixed [99, 57]. Alternatively, 'task-incremental' setups involve distinct tasks-possibly with different P ( Y | X ) -with task identity known at both train and test time. There, prior work [79, 72] trained a separate model per task and found that similarity-maximizing orderings prevail, seemingly contradicting our findings. However, in such scenarios, the focus shifts from forgetting to inter-task transfer , benefiting from similar consecutive tasks (see discussion on regret). Hence, their results complement ours.

Others have studied 'class-incremental' learning (CIL), where each task introduces new objects or classes, aiming for strong overall performance ( e.g., in split benchmarks [97]). While some CIL papers suggest that consecutive task similarity is preferable [44, 67], a closer look reveals that they modify the class composition within each task, inducing high intra-task heterogeneity [44, 8]. This likely leads to wider minima and stronger 'transferability' to other tasks, thus explaining their improved results. Such configurations resemble curriculum more than continual learning. 10 To our knowledge, only Yang and Li [104] report contradictory results, possibly due to their empirical design. 11 Finally, we note that the effects discussed here are related to the interleaving effect in educational psychology [77, 86].

Future work. One could extend our findings to other settings-such as class- and task-incremental, discussed earlier-and to more complex continual learning methods, such as replay and regularization. Moreover, our linear realizability assumption could be relaxed to accommodate label noise or even extend to nonlinear models, possibly borrowing tools from Kaczmarz literature [13, 106]. It would also be interesting to combine our approach with common wisdom in curriculum learningi.e., to design orderings that account for both task similarity and difficulty.

Finally, a promising direction for achieving tighter upper bounds in continual linear regression (see Table 1) lies in probabilistic selection rules, inspired by randomized greedy Kaczmarz methods [11, 12, 107, 95], which could combine the strengths of greedy orderings with the robustness of randomness, akin to our proposed hybrid scheme.

10 The learner controls the internal task composition to create 'easier' tasks, as in curriculum learning [101].

11 They construct the first task using a random half of the classes. This strong 'pretraining' leads to low initial loss, as the model already learns half the classes. This resembles the failure modes discussed in Section 5.1.

## Acknowledgments and Disclosure of Funding

We thank Joseph (Seffi) Naor (Technion) for fruitful discussions. We thank Timothée Lesort (Université de Montréal, MILA-Quebec AI Institute) for fruitful discussions and valuable feedback.

The research of DS was funded by the European Union (ERC, A-B-C-Deep, 101039436). Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency (ERCEA). Neither the European Union nor the granting authority can be held responsible for them. DS also acknowledges the support of the Schmidt Career Advancement Chair in AI.

DN was partially supported by NSF DMS 2408912.

## References

- [1] A. Achille, M. Lam, R. Tewari, A. Ravichandran, S. Maji, C. C. Fowlkes, S. Soatto, and P. Perona. Task2vec: Task embedding for meta-learning. In Proceedings of the IEEE/CVF international conference on computer vision , pages 6430-6439, 2019. (cited on p. 5)
- [2] S. Agmon. The relaxation method for linear inequalities. Canadian Journal of Mathematics , 6:382-392, 1954. (cited on p. 2, 5)
- [3] G. Alain and Y. Bengio. Understanding intermediate layers using linear classifier probes. In ICLR 2017 Workshop Track , 2016. (cited on p. 25)
- [4] G. Alain, A. Lamb, C. Sankar, A. Courville, and Y. Bengio. Variance reduction in sgd by distributed importance sampling. arXiv preprint arXiv:1511.06481 , 2015. (cited on p. 50)
- [5] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European Conference on Computer Vision (ECCV) , pages 139-154, 2018. (cited on p. 27)
- [6] V. Alvarez, S. Mazuelas, and J. A. Lozano. Supervised learning with evolving tasks and performance guarantees. Journal of Machine Learning Research , 26(17):1-59, 2025. (cited on p. 9)
- [7] H. Asanuma, S. Takagi, Y. Nagano, Y. Yoshida, Y. Igarashi, and M. Okada. Statistical mechanical analysis of catastrophic forgetting in continual learning with teacher and student networks. Journal of the Physical Society of Japan , 90(10):104001, Oct 2021. (cited on p. 3)
- [8] N. Ashtekar, J. Zhu, and V. G. Honavar. Class incremental learning from first principles: A review. Transactions on Machine Learning Research , 2025. (cited on p. 10)
- [9] K. Atkinson. An Introduction to Numerical Analysis, 2nd Ed . Wiley India Pvt. Limited, 2008. ISBN 9788126518500. URL https://books.google.com/books?id=lPV8Fv2XEosC . (cited on p. 94)
- [10] A. Attia, M. Schliserman, U. Sherman, and T. Koren. Fast last-iterate convergence of sgd in the smooth interpolation regime. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems , 2025. (cited on p. 2, 7, 8, 9, 40, 50)
- [11] Z.-Z. Bai and W.-T. Wu. On greedy randomized kaczmarz method for solving large sparse linear systems. SIAM Journal on Scientific Computing , 40(1):A592-A606, 2018. (cited on p. 6, 10)
- [12] Z.-Z. Bai and W.-T. Wu. On relaxed greedy randomized kaczmarz methods for solving large sparse linear systems. Applied Mathematics Letters , 83:21-26, 2018. (cited on p. 6, 10)
- [13] Z.-Z. Bai and W.-T. Wu. On greedy randomized augmented kaczmarz method for solving large sparse inconsistent linear systems. SIAM Journal on Scientific Computing , 43(6):A3892A3911, 2021. (cited on p. 10)
- [14] S. J. Bell and N. D. Lawrence. The effect of task ordering in continual learning. arXiv preprint arXiv:2205.13323 , 2022. (cited on p. 2, 5, 7, 9, 10, 19)

- [15] P. Beneventano. On the trajectories of sgd without replacement. arXiv preprint arXiv:2312.16143 , 2023. (cited on p. 8)
- [16] P. A. Borodin and E. Kopecká. Alternating projections, remotest projections, and greedy approximation. Journal of Approximation Theory , 260:105486, 2020. ISSN 0021-9045. (cited on p. 5)
- [17] L. Bottou. Curiously fast convergence of some stochastic gradient descent algorithms. In Proceedings of the symposium on learning and data science, Paris , volume 8, pages 2624-2633. Citeseer, 2009. (cited on p. 8)
- [18] W. Cai, Y. Zhang, and J. Zhou. Maximizing expected model change for active learning in regression. In 2013 IEEE 13th international conference on data mining , pages 51-60. IEEE, 2013. (cited on p. 19)
- [19] X. Cai and J. Diakonikolas. Last iterate convergence of incremental methods and applications in continual learning. In The Thirteenth International Conference on Learning Representations , 2025. (cited on p. 9)
- [20] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings of the European conference on computer vision (ECCV) , pages 532-547, 2018. (cited on p. 3)
- [21] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny. Efficient lifelong learning with a-GEM. In International Conference on Learning Representations , 2019. (cited on p. 9)
- [22] Y. Chen. Pytorch cifar models. https://github.com/chenyaofo/ pytorch-cifar-models , 2021. (cited on p. 25, 101)
- [23] L. Chizat, E. Oyallon, and F. Bach. On lazy training in differentiable programming. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019. (cited on p. 3)
- [24] R. Das, X. Chen, B. Ieong, P. Bansal, and sujay sanghavi. Understanding the training speedup from sampling with approximate losses. In Forty-first International Conference on Machine Learning , 2024. (cited on p. 19)
- [25] J. A. De Loera, J. Haddock, and D. Needell. A sampling kaczmarz-motzkin algorithm for linear feasibility. SIAM Journal on Scientific Computing , 39(5):S66-S87, 2017. (cited on p. 50)
- [26] C. M. De Sa. Random reshuffling is not always better. Advances in Neural Information Processing Systems , 33, 2020. (cited on p. 8)
- [27] M. Delange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2021. (cited on p. 9)
- [28] M. Ding, K. Ji, D. Wang, and J. Xu. Understanding forgetting in continual learning with linear regression. In Forty-first International Conference on Machine Learning , 2024. (cited on p. 3)
- [29] T. Doan, M. Abbana Bennani, B. Mazoure, G. Rabusseau, and P. Alquier. A theoretical analysis of catastrophic forgetting through the ntk overlap matrix. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics , pages 1072-1080, 2021. (cited on p. 3)
- [30] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In International conference on machine learning , pages 647-655. PMLR, 2014. (cited on p. 25)
- [31] Y. C. Eldar and D. Needell. Acceleration of randomized kaczmarz method via the johnsonlindenstrauss lemma. Numerical Algorithms , 58:163-177, 2011. (cited on p. 5)

- [32] T. Elfving. Block-iterative methods for consistent and inconsistent linear equations. Numerische Mathematik , 35(1):1-12, 1980. (cited on p. 5, 19)
- [33] I. Evron, E. Moroshko, R. Ward, N. Srebro, and D. Soudry. How catastrophic can catastrophic forgetting be in linear regression? In Conference on Learning Theory (COLT) , pages 40284079. PMLR, 2022. (cited on p. 2, 3, 4, 5, 6, 7, 9, 10, 19, 40, 49)
- [34] I. Evron, E. Moroshko, G. Buzaglo, M. Khriesh, B. Marjieh, N. Srebro, and D. Soudry. Continual learning in linear classification on separable data. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 9440-9484. PMLR, 23-29 Jul 2023. (cited on p. 2, 3, 4, 6, 9, 25, 27)
- [35] I. Evron, R. Levinstein, M. Schliserman, U. Sherman, T. Koren, D. Soudry, and N. Srebro. Better rates for random task orderings in continual linear models. arXiv preprint arXiv:2504.04579 , 2025. (cited on p. 2, 3, 7, 9, 19, 40, 50, 54)
- [36] H. Fang, G. Fang, T. Yu, and P. Li. Efficient greedy coordinate descent via variable partitioning. In C. de Campos and M. H. Maathuis, editors, Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence , volume 161 of Proceedings of Machine Learning Research , pages 547-557. PMLR, 27-30 Jul 2021. (cited on p. 19, 50)
- [37] M. L. Fisher, G. L. Nemhauser, and L. A. Wolsey. An analysis of approximations for finding a maximum weight hamiltonian circuit. Operations Research , 27(4):799-809, 1979. (cited on p. 7)
- [38] D. Goldfarb and P. Hand. Analysis of catastrophic forgetting for random orthogonal transformation tasks in the overparameterized regime. In International Conference on Artificial Intelligence and Statistics , pages 2975-2993. PMLR, 2023. (cited on p. 3)
- [39] D. Goldfarb, I. Evron, N. Weinberger, D. Soudry, and P. Hand. The joint effect of task similarity and overparameterization on catastrophic forgetting - an analytical model. In The Twelfth International Conference on Learning Representations , 2024. (cited on p. 3)
- [40] M. Griebel and P. Oswald. Greedy and randomized versions of the multiplicative schwarz method. Linear Algebra and its Applications , 437(7):1596-1610, 2012. (cited on p. 5, 50)
- [41] L. Gubin, B. T. Polyak, and E. Raik. The method of projections for finding the common point of convex sets. USSR Computational Mathematics and Mathematical Physics , 7(6):1-24, 1967. (cited on p. 2)
- [42] M. Gürbüzbalaban, A. Ozdaglar, and P. A. Parrilo. Why random reshuffling beats stochastic gradient descent. Mathematical Programming , 186(1):49-84, Mar 2021. ISSN 1436-4646. doi: 10.1007/s10107-019-01440-w. (cited on p. 8, 19)
- [43] D. Han and J. Xie. A simple linear convergence analysis of the reshuffling kaczmarz method. arXiv preprint arXiv:2410.01140 , 2024. (cited on p. 8)
- [44] C. He, R. Wang, and X. Chen. Rethinking class orders and transferability in class incremental learning. Pattern Recognition Letters , 161:67-73, 2022. ISSN 0167-8655. (cited on p. 2, 10)
- [45] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770778, 2016. (cited on p. 6, 25)
- [46] N. Hiratani. Disentangling and mitigating the impact of task similarity for continual learning. In The Thirty-Eighth Annual Conference on Neural Information Processing Systems , 2024. (cited on p. 3, 19)
- [47] E. Hoffer, B. Weinstein, I. Hubara, S. Gofman, and D. Soudry. Infer2train: leveraging inference for better training of deep networks. In NeurIPS 2018 Workshop on Systems for ML , page 40, 2018. (cited on p. 5)

- [48] F. Hucht. Solving a specific difference equation. MathOverflow, 2024. URL https:// mathoverflow.net/q/474430 . URL:https://mathoverflow.net/q/474430 (version: 2024-0704). (cited on p. 59, 82)
- [49] A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018. (cited on p. 3)
- [50] H. Jeong and D. Needell. Linear convergence of reshuffling kaczmarz methods with sparse constraints. SIAM Journal on Scientific Computing , 2025. to appear. (cited on p. 8)
- [51] H. Jung, H. Cho, and C. Yun. Convergence and implicit bias of gradient descent on continual linear classification. In The Thirteenth International Conference on Learning Representations , 2025. (cited on p. 3, 4, 6, 9)
- [52] S. Kaczmarz. Angenäherte auflösung von systemen linearer gleichungen. Bull. Int. Acad. Pol. Sic. Let., Cl. Sci. Math. Nat. , pages 355-357, 1937. (cited on p. 5, 19)
- [53] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences , 114(13):3521-3526, 2017. (cited on p. 9, 27)
- [54] M. Kong, W. Swartworth, H. Jeong, D. Needell, and R. Ward. Nearly optimal bounds for cyclic forgetting. In Thirty-Seventh Conference on Neural Information Processing Systems , 2023. (cited on p. 3, 6, 9)
- [55] A. Krizhevsky, G. Hinton, et al. Learning multiple layers of features from tiny images, 2009. (cited on p. 6, 25, 101)
- [56] A. Lad, R. Ghani, Y. Yang, and B. Kisiel. Toward optimal ordering of prediction tasks. In Proceedings of the 2009 SIAM International Conference on Data Mining , pages 884-893. SIAM, 2009. (cited on p. 2, 5)
- [57] T. Lesort, M. Caccia, and I. Rish. Understanding continual learning settings with data distribution drift analysis. arXiv preprint arXiv:2104.01678 , 2021. (cited on p. 10)
- [58] T. Lesort, O. Ostapenko, P. Rodríguez, D. Misra, M. R. Arefin, L. Charlin, and I. Rish. Challenging common assumptions about catastrophic forgetting and knowledge accumulation. In Conference on Lifelong Learning Agents , pages 43-65. PMLR, 2023. (cited on p. 9)
- [59] R. Levinstein, A. Attia, M. Schliserman, U. Sherman, T. Koren, D. Soudry, and I. Evron. Optimal rates in continual linear regression via increasing regularization. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems , 2025. (cited on p. 4, 9, 27)
- [60] H. Li, J. Wu, and V. Braverman. Fixed design analysis of regularization-based continual learning. In S. Chandar, R. Pascanu, H. Sedghi, and D. Precup, editors, Proceedings of The 2nd Conference on Lifelong Learning Agents , volume 232 of Proceedings of Machine Learning Research , pages 513-533. PMLR, 22-25 Aug 2023. (cited on p. 3)
- [61] H. Li, J. Wu, and V. Braverman. Memory-statistics tradeoff in continual learning with structural regularization. arXiv preprint arXiv:2504.04039 , 2025. (cited on p. 3)
- [62] Z. Li and N. Hiratani. Optimal task order for continual learning of multiple tasks. In Fortysecond International Conference on Machine Learning , 2025. (cited on p. 2, 5, 6, 7, 9, 10, 27)
- [63] S. Lin, P. Ju, Y . Liang, and N. Shroff. Theory on forgetting and generalization of continual learning. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 21078-21100. PMLR, 23-29 Jul 2023. (cited on p. 2, 3, 9, 10)

- [64] Y. Lu, S. Y. Meng, and C. De Sa. A general analysis of example-selection for stochastic gradient descent. In International Conference on Learning Representations (ICLR) , volume 10, 2022. (cited on p. 19)
- [65] E. S. Lubana, P. Trivedi, D. Koutra, and R. P. Dick. How do quadratic regularizers prevent catastrophic forgetting: The role of interpolation. In ICML Workshop on Theory and Foundations of Continual Learning , 2021. (cited on p. 6)
- [66] G. Mantione-Holmes, J. Leo, and J. Kalita. Utilizing priming to identify optimal class ordering to alleviate catastrophic forgetting. In 2023 IEEE 17th International Conference on Semantic Computing (ICSC) , pages 57-64. IEEE, 2023. (cited on p. 10)
- [67] M. Masana, B. Twardowski, and J. Van de Weijer. On class orderings for incremental learning. arXiv preprint arXiv:2007.02145 , 2020. (cited on p. 10)
- [68] C.-Q. Miao and W.-T. Wu. On greedy randomized average block kaczmarz method for solving large linear systems. Journal of Computational and Applied Mathematics , 413:114372, 2022. (cited on p. 6)
- [69] K. Mishchenko, A. Khaled, and P. Richtárik. Random reshuffling: Simple analysis with vast improvements. Advances in Neural Information Processing Systems , 33:17309-17320, 2020. (cited on p. 8, 19)
- [70] J. Monnot. Approximation algorithms for the maximum hamiltonian path problem with specified endpoint (s). European Journal of Operational Research , 161(3):721-735, 2005. (cited on p. 7)
- [71] C. V. Nguyen, A. Achille, M. Lam, T. Hassner, V. Mahadevan, and S. Soatto. Toward understanding catastrophic forgetting in continual learning. arXiv preprint arXiv:1908.01091 , 2019. (cited on p. 2, 5, 10)
- [72] T. Nguyen, C. N. Nguyen, Q. Pham, B. T. Nguyen, S. Ramasamy, X. Li, and C. V. Nguyen. Sequence transferability and task order selection in continual learning. arXiv preprint arXiv:2502.06544 , 2025. (cited on p. 2, 10)
- [73] Y.-Q. Niu and B. Zheng. A greedy block kaczmarz algorithm for solving large-scale linear systems. Applied Mathematics Letters , 104:106294, 2020. (cited on p. 2, 6)
- [74] J. Nutini, M. Schmidt, I. Laradji, M. Friedlander, and H. Koepke. Coordinate descent converges faster with the gauss-southwell rule than random selection. In International Conference on Machine Learning , pages 1632-1641. PMLR, 2015. (cited on p. 19)
- [75] J. Nutini, B. Sepehry, I. Laradji, M. Schmidt, H. Koepke, and A. Virani. Convergence rates for greedy kaczmarz algorithms, and faster randomized kaczmarz rules using the orthogonality graph. In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence , UAI'16, page 547-556, Arlington, Virginia, USA, 2016. AUAI Press. ISBN 9780996643115. (cited on p. 2, 5, 6, 50)
- [76] P. Oswald and W. Zhou. Convergence analysis for kaczmarz-type methods in a hilbert space framework. Linear Algebra and its Applications , 478:131-161, 2015. (cited on p. 6, 8)
- [77] S. C. Pan. The interleaving effect: mixing it up boosts learning. Scientific American , 313(2), 2015. (cited on p. 10)
- [78] L. Peng, P. Giampouras, and R. Vidal. The ideal continual learner: An agent that never forgets. In International Conference on Machine Learning , 2023. (cited on p. 3)
- [79] A. Pentina, V. Sharmanska, and C. H. Lampert. Curriculum learning of multiple tasks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 5492-5500, 2015. (cited on p. 2, 10, 19)
- [80] S. Rajput, K. Lee, and D. Papailiopoulos. Permutation-based SGD: Is random optimal? In International Conference on Learning Representations , 2022. (cited on p. 19)

- [81] V. V. Ramasesh, E. Dyer, and M. Raghu. Anatomy of catastrophic forgetting: Hidden representations and task semantics. In International Conference on Learning Representations , 2020. (cited on p. 10)
- [82] A. Ramdas. Rows vs columns for linear systems of equations-randomized kaczmarz or coordinate descent? arXiv preprint arXiv:1406.5295 , 2014. (cited on p. 19)
- [83] S. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert. icarl: Incremental classifier and representation learning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 5533-5542, Los Alamitos, CA, USA, jul 2017. IEEE Computer Society. doi: 10.1109/CVPR.2017.587. (cited on p. 9)
- [84] B. Recht and C. Ré. Beneath the valley of the noncommutative arithmetic-geometric mean inequality: conjectures, case-studies, and consequences. In Conference on Learning Theory (COLT) , 2012. (cited on p. 8)
- [85] S. Reich and R. Zalas. Polynomial estimates for the method of cyclic projections in hilbert spaces. Numerical Algorithms , pages 1-26, 2023. (cited on p. 4)
- [86] D. Rohrer, R. F. Dedrick, and S. Stershic. Interleaved practice improves mathematics learning. Journal of Educational Psychology , 107(3):900, 2015. (cited on p. 10)
- [87] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671 , 2016. (cited on p. 9)
- [88] P. Ruvolo and E. Eaton. Active task selection for lifelong machine learning. In Twenty-seventh AAAI conference on artificial intelligence , 2013. (cited on p. 9, 10)
- [89] H. Sajjad, N. Durrani, F. Dalvi, Y . Belinkov, and S. V ogel. Neural machine translation training in a multi-domain scenario. arXiv preprint arXiv:1708.08712 , 2017. (cited on p. 2)
- [90] N. Sarafianos, T. Giannakopoulos, C. Nikou, and I. A. Kakadiaris. Curriculum learning for multi-task classification of visual attributes. In Proceedings of the IEEE International Conference on Computer Vision Workshops , pages 2608-2615, 2017. (cited on p. 2)
- [91] C. Schouten. Investigating task order in online class-incremental learning. Master's thesis, Department of Mathematics and Computer Science, AutoML Group, Eindhoven University of Technology, Netherlands, 2024. (cited on p. 10)
- [92] H. Shan, Q. Li, and H. Sompolinsky. Order parameters and phase transitions of continual learning in deep neural networks. arXiv preprint arXiv:2407.10315 , 2024. (cited on p. 9)
- [93] A. Shrivastava, A. K. Gupta, and R. B. Girshick. Training region-based object detectors with online hard example mining. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 761-769, 2016. (cited on p. 19)
- [94] P. Soviany, R. T. Ionescu, P. Rota, and N. Sebe. Curriculum learning: A survey. International Journal of Computer Vision , pages 1-40, 2022. (cited on p. 19)
- [95] Y. Su, D. Han, Y. Zeng, and J. Xie. On the convergence analysis of the greedy randomized kaczmarz method. arXiv preprint arXiv:2307.01988 , 2023. (cited on p. 6, 10)
- [96] R.-Y. Sun. Optimization for deep learning: An overview. Journal of the Operations Research Society of China , 8(2):249-294, 2020. (cited on p. 8)
- [97] S. Swaroop, C. V. Nguyen, T. D. Bui, and R. E. Turner. Improving and understanding variational continual learning. arXiv preprint arXiv:1905.02099 , 2019. (cited on p. 10)
- [98] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2818-2826, 2016. (cited on p. 25)
- [99] G. M. Van de Ven, T. Tuytelaars, and A. S. Tolias. Three types of incremental learning. Nature Machine Intelligence , 4(12):1185-1197, 2022. (cited on p. 10)

- [100] J. T. Wang, T. Wu, D. Song, P. Mittal, and R. Jia. GREATS: Online selection of high-quality data for LLM training in every iteration. In The Thirty-Eighth Annual Conference on Neural Information Processing Systems , 2024. (cited on p. 19)
- [101] X. Wang, Y. Chen, and W. Zhu. A survey on curriculum learning. IEEE transactions on pattern analysis and machine intelligence , 44(9):4555-4576, 2021. (cited on p. 10, 19)
- [102] A.-Q. Xiao, J.-F. Yin, and N. Zheng. On fast greedy block kaczmarz methods for solving large consistent linear systems. Computational and Applied Mathematics , 42(3):119, 2023. (cited on p. 6)
- [103] Y. Xu and B. Mirzasoleiman. Ordering for non-replacement sgd. arXiv preprint arXiv:2306.15848 , 2023. (cited on p. 19)
- [104] Z. Yang and H. Li. Task ordering matters for incremental learning. In 2021 International Symposium on Networks, Computers and Communications (ISNCC) , pages 1-6, 2021. (cited on p. 10)
- [105] Y. Zeng, D. Han, Y. Su, and J. Xie. Fast stochastic dual coordinate descent algorithms for linearly constrained convex optimization. arXiv preprint arXiv:2307.16702 , 2023. (cited on p. 19)
- [106] J. Zhang, Y. Wang, and J. Zhao. On maximum residual nonlinear kaczmarz-type algorithms for large nonlinear systems of equations. Journal of Computational and Applied Mathematics , 425:115065, 2023. (cited on p. 5, 6, 10)
- [107] J.-J. Zhang. A new greedy kaczmarz algorithm for the solution of very large linear systems. Applied Mathematics Letters , 91:207-212, 2019. (cited on p. 6, 10)
- [108] Y. Zhang and H. Li. Greedy motzkin-kaczmarz methods for solving linear systems. Numerical Linear Algebra with Applications , 29(4):e2429, 2022. (cited on p. 6)
- [109] X. Zhao, H. Wang, W. Huang, and W. Lin. A statistical theory of regularization-based continual learning. In Forty-first International Conference on Machine Learning , 2024. (cited on p. 3)

## Appendix contents

| A Further related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19   | A Further related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19   |   A Further related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19 |
|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| B Appendix to Section 4.1: Regression experiments . . . . . . . .                                                     | . . . . . . . . . . . . . .                                                                                           |                                                                                                                    21 |
| B.1                                                                                                                   | Isotropic data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                |                                                                                                                    21 |
| B.2                                                                                                                   | Anisotropic data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                  |                                                                                                                    22 |
| B.3                                                                                                                   | A note on statistical significance. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                     |                                                                                                                    24 |
| C Appendix to Section 4.1: Classification experiments . . . . . . . . . . . .                                         | . . . . . . . .                                                                                                       |                                                                                                                    25 |
| C.1                                                                                                                   | Code. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .               |                                                                                                                    25 |
| C.2                                                                                                                   | Experiment details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                   |                                                                                                                    25 |
| C.3                                                                                                                   | Out-of-domain feature extractors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                        |                                                                                                                    26 |
| C.4                                                                                                                   | Regularization ablation study. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                      |                                                                                                                    27 |
| C.5                                                                                                                   | Allowing task repetition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                    |                                                                                                                    27 |
| C.6                                                                                                                   | Rule calculation with partial data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                      |                                                                                                                    28 |
| D Appendix to Section 4.2: Proofs for 'nearly determined' tasks . . .                                                 | . . . . . . . . .                                                                                                     |                                                                                                                    29 |
| D.1                                                                                                                   | Optimality guarantee when r = d - 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . .                           |                                                                                                                    29 |
| D.2                                                                                                                   | Loss bound when r = d - 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                         |                                                                                                                    30 |
| E Appendix to Section 5.1: Lower bound's 'adversarial' constructions . .                                              | . . . . . .                                                                                                           |                                                                                                                    31 |
| E.1                                                                                                                   | General dimension construction and proof (Theorem 5.2) . . . . . . . . . . . . . . .                                  |                                                                                                                    31 |
| E.2                                                                                                                   | Adversarial 3d construction (Example 5.1) . . . . . . . . . . . . . . . . . . . . . . . . .                           |                                                                                                                    36 |
| F Appendix to Section 5.2: Single-pass vs. repetition . . . . . . . .                                                 | . . . . . . . . . . . . .                                                                                             |                                                                                                                    40 |
| F.1                                                                                                                   | Appendix to the upper bound for greedy orderings with repetition (Theorem 5.3).                                       |                                                                                                                    40 |
| F.2                                                                                                                   | Regression experiments on single-pass vs. repetition . . . . . . . . . . . . . . . . . .                              |                                                                                                                    48 |
| G Appendix to Section 5.3: Hybrid task ordering. . . .                                                                | . . . . . . . . . . . . . . . . . . . .                                                                               |                                                                                                                    50 |
| G.1                                                                                                                   | Hybrid ordering scheme . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                      |                                                                                                                    50 |
| G.2                                                                                                                   | Hybrid ordering upper bound. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                        |                                                                                                                    51 |
| G.3                                                                                                                   | Hybrid ordering experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                       |                                                                                                                    54 |
| H Appendix to Section 4.1: Code snippet for regression experiments . .                                                | . . . . . . . .                                                                                                       |                                                                                                                    55 |
| I Lower bound technical appendix: Delta positivity proof                                                              | . . . . . . . . . . . . . . . . .                                                                                     |                                                                                                                    59 |
| I.1                                                                                                                   | Proof outline. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                |                                                                                                                    59 |
| I.2                                                                                                                   | Auxiliary: Algebraic inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                       |                                                                                                                    59 |
| I.3                                                                                                                   | Proof body . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                |                                                                                                                    61 |
| I.4                                                                                                                   | Conclusion. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                 |                                                                                                                    79 |
| J Lower bound technical appendix: Properties of the recursive construction .                                          | . . .                                                                                                                 |                                                                                                                    81 |
| J.1                                                                                                                   | Proof outline. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                |                                                                                                                    81 |
| J.2                                                                                                                   | Full proof. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .               |                                                                                                                    83 |
| NeurIPS Paper Checklist.                                                                                              | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                           |                                                                                                                    96 |

## A Further related work

Here, we elaborate on additional connections not fully addressed in Section 6.

Alternative viewpoint: The Kaczmarz method. Our continual linear regression scheme maps directly to the Kaczmarz method [52, 32], a classical iterative projection algorithm for solving linear systems of equations. In our context, the solved system is, Xw = y , where

<!-- formula-not-decoded -->

Evron et al. [33] pointed out that Kaczmarz methods iteratively solve the 'block' systems of the form X τ ( t ) w = y τ ( t ) using an update rule equivalent to our continual update in Eq. (1). As a result, the observations and results in our paper extend naturally to the greedy Kaczmarz method. However, whereas Kaczmarz studies typically analyze convergence in terms of the distance to the intersection w ⋆ , we focus on the loss , i.e., the residuals (Definitions 2.5 and 2.3, respectively). For example, in the r = d -1 case of Section 4.2, this distinction allowed proving an upper bound on the loss and an 'approximation' result on the optimal distance (Lemmas 4.2 and 4.1, respectively).

It is worth noting that, via its connection to the Kaczmarz method, our continual linear regression scheme is also related to coordinate descent methods [82]. While prior work in this area shows that greedy selection can outperform random sampling, these results often rely on strong convexity assumptions [74, 36], and typically apply to the Kaczmarz method through a primal-dual lens [33, 105]-again, yielding only convergence to the intersection point w ⋆ .

Curriculum learning. Broadly, curriculum learning enhances training by controlling the order in which data are presented, to accelerate convergence or improve accuracy. The prevailing view is that examples should be ordered from easy to hard by their 'difficulty' [101, 94]. In contrast, we study similarity -guided orderings, aligning with recent findings in continual learning [14, 46].

A key distinction between curriculum and continual learning lies in the unit of ordering: while curriculum learning typically orders individual samples or batches, we focus on orderings of entire tasks. Moreover, curriculum learning often takes a single gradient step per sample, whereas continual learning optimizes each task to a low loss before proceeding. Nonetheless, given the maturity of curriculum learning and the interdisciplinary nature of our work, some curriculum studies that operate at the task level are directly relevant [e.g., 79] and were discussed in Section 6.

Example selection in SGD. Evron et al. [35] show that learning an entire (continual) linear regression task in our Scheme 1 reduces to taking a single large gradient step on a modified objective. While they use this reduction to analyze random orderings via last-iterate SGD analysis, we leverage it here to draw connections between greedy task orderings and greedy example selection in SGD.

Most of the example selection literature considers multi-epoch settings, where each sample is seen multiple times. In such regimes, it is common to randomly shuffle the dataset once or at the start of each epoch [e.g., 69, 42], but this is not necessarily optimal [80]. For instance, Lu et al. [64] show that greedy permutations-computed at the beginning of each epoch-can yield faster convergence than random ones. However, their analysis requires (1) multiple epochs and (2) very small step sizes, making it inapplicable to our single-pass continual setting.

Das et al. [24] show that a selection rule akin to our maximum residual rule (Definition 3.2) accelerates early convergence but may underperform random orderings asymptotically-aligning with our findings on the hybrid approach in Section 5.3. They also analyze an approximate rule, supporting our observations on computational tractability in Section 3. Others select greedily by gradient magnitude instead of loss [103], or 'mine' examples at the mini-batch level-selecting 'hard' samples with a high loss or ones that lead to a significant decrease in loss [93, 100].

Active Learning. Active learning aims to reduce labeling cost by querying the most informative samples for labeling, typically from a large unlabeled pool. This setting resembles ours, where the learner may apply a greedy maximum distance rule (Definition 3.1) to select the task or sample expected to induce the greatest model update. For example, a related idea is explored in Cai et al. [18], who propose a greedy maximum distance variant for regression. Since labels are unknown

at selection time in their active learning setup, they approximate the expected model change using a bootstrap method. Empirically, this approach identifies informative examples and consistently improves generalization across datasets.

## B Appendix to Section 4.1: Regression experiments

All figures report averages over 10 runs. In each run, we randomly sample a task collection to evaluate the different ordering strategies. Shaded regions (see App. F.2 and G.3) indicate ± 1 standard error intervals, even when not visually discernible. In App. B.3 we further discuss the statistical significance of our experiments.

Computational resources. All regression experiments-including those not shown-were completed within 4 hours on a home PC equipped with an Intel i5-9400F CPU and 16GB of RAM.

## B.1 Isotropic data

Figures 8 and 9 extend the previous experiment on isotropic data (Figure 3a) to varying dimensions d , ranks r and task counts T . Results confirm consistent patterns: greedy (dissimilarity maximizing) methods outperform random, and MD is better than MR across all settings (sometimes only slightly).

Figure 8: Comparing orderings for varying dimensions d and ranks r of the data matrices, for isotropic data. T = 50 . We observe that, for such isotropic data, the random ordering performance is determined solely by the ratio r/d . In contrast, greedy orderings that prioritize dissimilarity benefit from a lower dimension when r/d is fixed (to see that, focus on single columns in the grid). We hypothesize that this is because an increased task 'density' in lower dimensions: when r/d is fixed, increasing d increases d -r , expanding the set of possible task projections (see Eq. (2)). As a result, a fixed number of tasks T covers this space more sparsely in higher dimensions. In lower dimensions, the same T tasks yield denser coverage, increasing the likelihood that greedy dissimilarity-based selection identifies tasks with large projections.

<!-- image -->

In all strategies, higher task rank consistently yields improved performance (focus on single rows). This is because the solution subspaces are of rank d -r , so increasing r (with fixed d ) lowers the subspace rank, increasing the distances between them and resulting in larger projections.

Figure 9: Comparing orderings for varying task count T , for isotropic data. d = 100 , r = 10 . Dissimilarity-based greedy strategies become more effective as the number of tasks increases. This is since in an isotropic setting, where task directions are sampled uniformly, increasing the number of tasks increases the coverage of the unit sphere. This results in a higher probability of encountering task pairs with large angular separation between their solution subspaces, which greedy ordering utilizes.

<!-- image -->

## B.2 Anisotropic data

The following experiments were were performed with anisotropic data, sampled from a Gaussian distribution with exponentially decaying eigenvalues, as detailed in Scheme 2, resulting in high task correlation. This arises because tasks tend to align with the dominant eigen-directions, leading to strong pairwise similarity.

## Scheme 2 Generating tasks with high correlation

Require: Input dimension d , task rank r , number of tasks T , edge eigenvalues λ 1 = 10 -3 , λ d = 10 3

- 2: Compute SVD: A sym = USU ⊤
- 1: Sample A ∼ N (0 , 1) d × d and symmetrize: A sym ← 1 2 ( A + A ⊤ )
- 3: Define diagonal spectrum: Λ ← diag ( λ 1 exp ( ln ( λ d /λ 1 ) i d -1 )) d -1 i =0
- 4: Construct covariance: Σ ← UΛU ⊤
- 5: for t = 1 to T do
- 6: Sample Z t ∼ N (0 , 1) r × d
- 7: Set X t ← Z t Σ 1 / 2
- 8: end for
- 9: Output: { X t } T t

=1

Figures 10 and 11 below reveal some interesting trends compared to the isotropic case.

Figure 10: Comparing orderings for varying dimensions d and ranks r of the data matrices, for anisotropic data. T = 50 . Compared to the isotropic case (Figure 8), we observe slower rates for all strategies. This is easily explained by all pairwise distances between task solution subspaces becoming smaller, due to the higher correlation in the anisotropic case. Interestingly, as rank increases (focusing on a single row in the grid), the Maximum Residual (MR; Definition 3.2) ordering underperforms and seemingly aligns with the random one. This may stem from the combination of high rank and strong intra -task correlation, which leads to ill-conditioned data matrices (for each task). In such a case, small perturbations, or steps, in the solution space may cause disproportionately large changes in residuals. As a result, MR is misled into selecting tasks with large residuals that advance the iterate only marginally toward the intersection ( w ⋆ ).

<!-- image -->

Figure 11: Comparing orderings for varying task count T , for anisotropic data. d = 100 , r = 10 . Unlike in the isotropic case (Figure 9), greedy orderings do not significantly benefit from increasing the number of tasks T . This is likely since, in the anisotropic case, a large number of tasks must be added to induce the substantial 'angles' that greedy orderings can exploit. Put differently, under our anisotropic distribution, the probability that any set of 50 tasks are mutually orthogonal-and thus beneficial to greedy orderings-is extremely small for any reasonable number of tasks T .

<!-- image -->

## B.3 A note on statistical significance

All appendix figures include confidence intervals of ± 1 standard error, although these are often too narrow to be visible. While different task collections introduce slight variations in outcomes, the overall trends are highly consistent. This is illustrated in the following figure, where we replicate the plot from Figure 3a, overlaying individual runs from all 10 repeated experiments. Despite some run-to-run variability, the standard error remains small, reinforcing the robustness of our qualitative conclusions.

Figure 12: Showing variations across different experiments. Same as Figure 3a, we have T = 50 , r = 10 , d = 100 , with random isotropic data. Shaded plots represent each individual experiment. While minor variations exist across experiments, the low standard error confirms the consistency of the results.

<!-- image -->

## C Appendix to Section 4.1: Classification experiments

## C.1 Code

The code for the classification experiments with CIFAR-100 is available at https://github.com/ matants/greedy\_ordering .

Computational resources. All classification experiments were completed within a month's work on 4 NVIDIA GeForce GTX 1080 Ti GPUs.

## C.2 Experiment details

## C.2.1 Model: Linear probing on pretrained ResNet-20

Our experiments employ a frozen pretrained ResNet-20 classifier [45], where the final classification head was removed and replaced with the binary classification head that we train [30, 3].

## C.2.2 Tasks and benchmarks

We employ three benchmarks of domain-incremental CIFAR-100 -based binary classification:

- (A) Using a model pretrained on CIFAR-100 [55], taken from Chen [22], which achieves 68.83% top-1 classification accuracy on CIFAR-100 multiclass classification according to Chen [22]. The continual learning tasks are composed of CIFAR-100 classes randomly split to 50 pairs of binary classification tasks, such that all classes from the same superclass share a label. This is the setting used in all experiments unless stated otherwise , including the results presented in Figure 3b.
- (B) We partition the 500 training samples of each CIFAR-100 class to two distinct groups of 250 samples, and use one of the groups to train the ResNet-20 embedder on the original CIFAR-100 multiclass task, using the same training recipe as Chen [22] and achieving 61.57% top-1 classification accuracy on the CIFAR-100 test set after 200 training epochs. The partitioning and training code is included in our provided repository. We then employ a linear probe on top of the resulting model (with the classification head removed), and construct the continual learning tasks using the 250 samples per class that weren't used for training the embedder. The classes that compose each task are the same as in the previous benchmarks.
- (C) Using a model pretrained on CIFAR-10 , taken from Chen [22], with the same CIFAR-100 -based tasks as (A).

All presented results were composed by experimenting with 25 randomly generated task sets.

## C.2.3 Training

Training was performed with cross-entropy loss on the softmax of the classifier's output with label smoothing of 0.05 [98], with additive L2 regularization towards the previous parameters controlled by the hyperparameter λ , which is common in continual learning and is necessary to facilitate the projections view, as shown in Evron et al. [34]. In Figure 3b, λ = 5 , and we present an ablation study for λ in App. C.4, to address how it affects the performance of different ordering rules.

For each task we used the SGD optimizer with a learning rate of lr = 0 . 01 and ReduceLROnPlateau on epoch losses, trained for 40 epochs with a batch size of 64. As a baseline, we jointly trained a classifier on all tasks together, without regularization.

## C.2.4 Evaluation

We evaluate the performance of each ordering by calculating the average test (generalization) loss of all tasks after each seen task. Results are presented with 95% confidence intervals, calculated over the different randomly generated task sets, and over the permutations as well for random ordering.

## C.2.5 Ordering computation

For the Random rule (Eq. (4)), we use random sampling without replacement from the task set, unless stated otherwise. When presenting results for it, we use 4 random task permutations per task set. The Greedy MR rule (Definition 3.2) requires calculating the loss (without regularization) of all tasks after each task training, and choosing the task with the maximal loss. In App. C.6 we show its performance doesn't degrade when the losses are evaluated on a fraction of the dataset, as small as 1% of the data-5 samples per class in CIFAR-100 . The Greedy MD rule (Definition 3.1) was calculated by performing full training on each task, as elaborated above (App. C.2.3)-choosing the task that resulted in model parameters farthest from the current model parameters in terms of Euclidean distance. While the MD rule may seem impractical-we show that, in fact, the much simpler Greedy MR rule, that requires a single forward pass, achieves identical performance.

## C.3 Out-of-domain feature extractors

To evaluate how our proposed method extends to more general transfer learning settings, we employ multiple benchmarks as elaborated in App. C.2.2.

- (A) Figure 13a is the same figure as Figure 3b, shown here for completeness.
- (B) As shown in Figure 13b, this transferlearning setting behaves similarly to the case where tasks are drawn from the training data (Figure 3b), though with slightly weaker performance for both the joint baseline and all orderings.
- (C) As shown in Figure 13c, dissimilarityguided orderings still outperform random orderings, though less prominently than in other experiments. We hypothesize that this stems from the model's weak joint interpolation ability (indicated by the dashed curve), which more strongly violates our joint-realizability assumption (Assumption 2.1).

<!-- image -->

(a) Pretraining: Full CIFAR-100 ; Continual learning: Full CIFAR-100 .

<!-- image -->

- (b) Pretraining: Partitioned CIFAR-100 ; Continual learning: Remaining CIFAR-100 .
- (c) Pretraining: CIFAR-10 ; Continual learning: CIFAR-100 .

<!-- image -->

Figure 13: Comparison of orderings under different pretraining setups.

## C.4 Regularization ablation study

Regularization toward previous model parameters is a standard method to mitigate catastrophic forgetting [53, 5, 59], and is crucial for a projections view to emerge in continual classification [34]. Because of its central role, we perform an ablation study on the effect of regularization strength for completeness. As shown in Figure 14, without regularization our continual learning scheme collapses across all orderings, and greedy methods consistently outperform random ordering across all strengths examined. Interestingly, the optimal performance for greedy orderings occurs at smaller regularization strengths ( λ ) than for random ordering. This makes sense: greedy methods deliberately select tasks that push parameters further from their current values, so the regularization term has a stronger influence on the loss, requiring smaller λ for optimal effect.

Figure 14: Regularization strength ablation study.

<!-- image -->

## C.5 Allowing task repetition

As discussed in Section 5.2, in continual linear regression, allowing repetitions helps greedy orderings avoid failure modes and guarantees provable convergence. In our classification experiments, however, repetitions do not improve performance: as shown in Figure 15, where each scheme could select from all tasks for 100 iterations (instead of 50), repetitions actually harm the performance of greedy orderings, effectively canceling their advantage over random orderings. We observe a similar phenomenon in linear regression with anisotropic low-rank data, as detailed in App. F.2.

In the classification case, which departs substantially from jointly realizable linear regression, multiple factors could underlie this behavior. It would be interesting to examine how this relates to the connection between greedy ordering and 'periphery-to-core' ordering [62], which may break down when repetitions are allowed, or to the distance-to-teacher perspective explored in App. F.2 (Figure 23). As this lies beyond the scope of our paper, we leave it for future work.

Figure 15: The effect of task repetition on CIFAR-100 continual classification.

<!-- image -->

## C.6 Rule calculation with partial data

To assess practicality, we evaluate the more efficient Greedy MR method-which requires only forward passes-using fractions of the data and compute. Even with just 1% of the data (5 samples per CIFAR-100 class, i.e., 10 per binary task) to compute each task's loss, Greedy MR maintains its performance and remains stronger than random ordering.

Figure 16: Greedy MR rule calculation using partial data.

<!-- image -->

## D Appendix to Section 4.2: Proofs for 'nearly determined' tasks

## D.1 Optimality guarantee when r = d -1

Recall Lemma 4.1. Let w τ MD T and w τ ⋆ T be the iterates after learning T jointly realizable tasks of rank d -1 under the Maximum Distance ordering τ MD and an optimal ordering τ ⋆ that leads to a minimal distance to the joint solution w ⋆ . Then, their distances hold,

<!-- formula-not-decoded -->

Proof. The distance at the end of an ordering τ is

<!-- formula-not-decoded -->

Let τ = τ , τ be the greedy MD ordering and an optimal ordering leading to the minimal distance

<!-- formula-not-decoded -->

MD ⋆ { 2 .

Then, we have,

<!-- formula-not-decoded -->

where we define the index set ⋆ ⋆ Employing greediness, we get

<!-- formula-not-decoded -->

Then, since τ -1 ( τ ⋆ ( · )) 'covers' [ T ] and c ( i, j ) ≤ 1 , iterating over the entire 1 , . . . , T -1 will simply add elements to the product and make it smaller. That is,

<!-- formula-not-decoded -->

## D.2 Loss bound when r = d -1

Recall Lemma 4.2. Under the Maximum Distance greedy ordering over T jointly-realizable tasks of rank d -1 , the loss of Scheme 1 after T iterations is upper bounded as,

<!-- formula-not-decoded -->

Proof. We aim to bound the average loss using projection matrices,

<!-- formula-not-decoded -->

Since each task matrix X i has rank d -1 , each projection P i is rank 1 and can be written as P i = v i v ⊤ i for a unit vector v i . Substituting this and v τ (0) = 1 ∥ w ⋆ ∥ ( w 0 -w ⋆ ) , the bound becomes:

<!-- formula-not-decoded -->

Then, we use algebraic and projection properties to rewrite the greedy ordering as:

<!-- formula-not-decoded -->

Then, employing greediness as reformulated above and inequality of arithmetic and geometric mean, we obtain:

<!-- formula-not-decoded -->

Substituting back into the forgetting, it is now bounded as,

<!-- formula-not-decoded -->

eT

## E Appendix to Section 5.1: Lower bound's 'adversarial' constructions

## E.1 General dimension construction and proof (Theorem 5.2)

Recall Theorem 5.2. For any d ≥ 30 , there exists an adversarial task collection with T = d -1 jointly-realizable tasks of different rank such that both greedy orderings (MD, MR) forget catastrophically . That is, the loss at the end of the sequence is, L ( w τ MD T ) , L ( w τ MR T ) ≥ 1 8 -1 4 d .

Proof outline. For a given dimension d , we construct a sequence of d iterates ( w t ) d t =1 , corresponding to T = d -1 tasks ( X t ) d t =2 of decreasing rank, which are jointly-realizable with w ⋆ = 0 ( i.e., ∀ t ∈ { 2 ...T } , y t = 0 ), and show that:

1. Bottom line. Given this specific choice of tasks and matching iterates, the loss (or forgetting) is catastrophic as mentioned in the theorem.
2. The chosen iterates are validi.e., they can be obtained from a specific selection rule given the constructed task collection.
3. The chosen ordering adheres to greedy selection rules, both MD and MR, under the chosen tasks. This part is quite lengthy.

In the construction, we start the iterates from t = 1 and tasks from t = 2 , contrary to other parts of the paper, for no particular reason other than ease of notation. For this same reason we chose w ⋆ = 0 , and the iterates starting with w 1 = e 1 . The same construction holds for a shifted frame of reference where all iterates (and w ⋆ ) are shifted by -e 1 .

## E.1.1 Construction details

We first construct the iterates as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We denote x t ≜ ( w t ) 1 , defined recursively by

<!-- formula-not-decoded -->

̸

We now construct the tasks:

<!-- formula-not-decoded -->

Then, it is easy to see that P t ≜ I d -X + t X t = I d -I t +1: d -u t u ⊤ t = I t ︸︷︷︸ rank t -u t u ⊤ t .

Since w t = w t -1 , we are free to define the unit vector

<!-- formula-not-decoded -->

## E.1.2 Lower bounding the loss

For each task X m , its individual loss at time t = d is given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So the average loss after all iterates, which coincides with the forgetting (see Remark 2.4) is:

<!-- formula-not-decoded -->

## E.1.3 Proving that the iterates and tasks exist

In Lemma J.1, we prove that for all d ≥ 30 , t ∈ { 2 , . . . , d } , we have x 2 t -1 -4 β t ≥ 0 , so the square root in the recursive definition of x t (Eq. (7)) exists.

## E.1.4 Proving that the iterates can be formed from projections of the given tasks

As a sanity check, we notice that P t is a real symmetric matrix, and assert its idempotence,

<!-- formula-not-decoded -->

First, we show that w ⊤ t ( w t -w t -1 ) = 0 , as expected from orthogonality in projections:

<!-- formula-not-decoded -->

and it is readily seen that our construction choice of ( w t ) 1 = ( w t -1 ) 1 + √ ( w t -1 ) 2 1 -4 β t 2 implies

<!-- formula-not-decoded -->

Finally, we show that the iterates are indeed a sequence the corresponding projections:

<!-- formula-not-decoded -->

## E.1.5 Proving that the iterates adhere to greedy ordering rules

Maximum Distance (MD). We wish to prove that the greedy MD rule agrees with the ordering we chose. That is,

<!-- formula-not-decoded -->

By induction on the validity of the greediness for τ 2 , . . . , τ t -1 , the step is (and the induction base for t = 2 is shown exactly the same):

<!-- formula-not-decoded -->

Maximum Residual (MR). We wish to prove that the greedy MR rule agrees with the ordering we chose. That is,

<!-- formula-not-decoded -->

By induction on the validity of the greediness for τ 2 , . . . , τ t -1 , the step is (and the induction base for t = 2 is shown exactly the same):

<!-- formula-not-decoded -->

We get that the MR and MD rules coincide in this case.

How we prove greediness holds: Delta positivity. We wish to show monotonous decrease (w.r.t. k ≥ t ) of ( ( ( w k -1 -w k ) ⊤ w t -1 ) 2 ∥ w k -1 -w k ∥ 2 ) k , which will prove that the iterates we defined are valid under the greedy MD and MR orderings ( i.e., adhere to the rules in Def. 3.1 and 3.2).

The difference between consecutive iterates is

<!-- formula-not-decoded -->

We notice that ∀ k ≥ t the term ( w k -1 -w k ) ⊤ w t -1 is positive since,

<!-- formula-not-decoded -->

This means that we can alternatively show monotonous decrease ∀ k ≥ t for

<!-- formula-not-decoded -->

To this end, we wish to show that the next quantity is positive ∀ t ∈ { 2 , . . . , d -1 } (we are reminded that the first step is at t = 2 due to our choice, and that at the last step there is only one choice), ∀ k ∈ { t, . . . , d -1 } :

<!-- formula-not-decoded -->

Next, we will show this holds numerically for low dimensions ( d &lt; 25 , 000 ), and prove it analytically ∀ d ≥ 25 , 000 .

Showing delta positivity numerically for low dimensions. We use the following facts to write code that verifies ∆ t,k &gt; 0 ∀ d &lt; 25 , 000 , ∀ t ∈ { 2 , . . . , d -1 } , ∀ k ∈ { t, . . . , d -1 } :

<!-- formula-not-decoded -->

For each value of dimension d , we calculated the sequence ( x ) k using its recursive definition, and calculated ∆( d ) ≜ min { t,k | t ∈{ 2 ,...,d -1 } , k ∈{ t,...,d -1 }} ∆ t,k using these formulas. As shown in Figure 17, we found ∆( d ) remains positive for all d ∈ { 30 ... 47 , 000 } (for completeness, any dimension above 25 , 000 is redundant here). In addition, as will be seen analytically (Eq. (13)), we have that ∆( d ) should correlate with d -5 2 , and for completeness we show this holds numerically for the lower dimensions as well, by showing ∆( d ) · d 5 2 is approximately constant.

Computational resources. This numerical validation took 4 days to run on a home PC with i5-9400F CPU and 16GB RAM.

Figure 17: Numerical positivity of ∆( d ) ≜ min { t,k | t ∈{ 2 ,...,d -1 } , k ∈{ t,...,d -1 }} ∆ t,k

<!-- image -->

Showing delta positivity analytically for high dimensions. Due to the length of this part, we defer it to App. I, where we prove that ∀ d ≥ 25 , 000 , ∀ t ∈ { 2 , . . . , d -1 } , ∀ k ∈ { t, . . . , d -1 } ,

<!-- formula-not-decoded -->

Conclusion. Together with the numerical verification, we have established that ∆ t,k &gt; 0 for all k ≥ t and all d ≥ 30 . This completes the proof of the iterates' adherence to the greedy ordering rules, and thereby concludes the overall proof of the adversarial construction that yields a lower bound on the loss under single-pass greedy orderings.

## E.2 Adversarial 3d construction (Example 5.1)

Recall Example 5.1. For all T ∈ { 4 · 10 i -1 | i = 1 , 2 , . . . , 7 } , there exists a task collection of jointly-realizable tasks in d = 3 , such that L ( w τ MD T ) , L ( w τ MR T ) &gt; 2 . 78 · 10 -5 .

## E.2.1 Construction details

For simplicity we employ the joint solution w ⋆ = 0 , which means that for all tasks m ∈ [ T ] , y m = 0 . For some K ∈ N + , we construct T = 4 K -1 tasks by defining their solution subspaces as follows:

- K -1 copies of Span (( 0 , sin ( 1 √ K ) , cos ( 1 √ K ) )) ,
- K copies of Span (( 0 , -sin ( 1 √ K ) , cos ( 1 √ K ) )) ,
- K copies of Span (( 0 , sin ( 1 2 √ K ) , cos ( 1 2 √ K ) ) , ( 1 √ 2 , 0 , 1 √ 2 )) ,
- K copies of Span (( 0 , -sin ( 1 2 √ K ) , cos ( 1 2 √ K ) ) , ( 1 √ 2 , 0 , 1 √ 2 )) .

For a given solution subspace of rank d -r , a task feature matrix X ∈ R r × d of rank r ( i.e., with linearly independent rows) is defined such that each row of X is orthogonal to the solution subspace.

<!-- formula-not-decoded -->

## E.2.2 Lower bound explanation

This is not a formal proof, but an explanation of why the construction works. We stick to using greedy MDfor the intuition.

While learning tasks consecutively using greedy MD ordering, we start by alternating between projecting onto the 1 -D subspaces, since each 2 -D subspace contains a line (either Span (( 0 , sin ( 1 2 √ K ) , cos ( 1 2 √ K ) )) or Span (( 0 , -sin ( 1 2 √ K ) , cos ( 1 2 √ K ) )) ) between them and hence cannot be the farthest away. Once those are used up, we're left with the 2 -D subspaces, which we alternate between.

Morally, the angle between the 1 -D subspaces is O ( 1 √ K ) , so as K grows large, these first 2 K -1 steps should bring us to about (0 , 0 , Θ(1)) . The 2 -D subspaces intersect in Span (( 1 √ 2 , 0 , 1 √ 2 )) , and the angle between them is also Θ ( 1 √ K ) , so they move us a constant fraction of the way toward the closest point on Span (( 1 √ 2 , 0 , 1 √ 2 )) , which is (Θ(1) , 0 , Θ(1)) . Since the first half of the tasks are projections onto 1 -D subspaces near (0 , 0 , 1) , the x -coordinate contributes Θ(1) loss due to these tasks.

Quantifying the asymptotic loss. In the first 2 K -1 steps we multiply ∥ w ∥ by cos(2 / √ K ) each time, giving

<!-- formula-not-decoded -->

The direction is ( 0 , -sin ( 1 / √ K ) , cos ( 1 / √ K ) ) , so we end at (0 , o (1) , e -4 + o (1)) .

Next we project onto a 2 -D subspace; this is o (1) movement, so w becomes

<!-- formula-not-decoded -->

Let v be the closest point on Span (( 1 √ 2 , 0 , 1 √ 2 )) to w = ( 0 , 0 , e -4 ) + o (1) . Then

<!-- formula-not-decoded -->

All remaining 2 -D subspaces contain the line L = Span (( 1 √ 2 , 0 , 1 √ 2 )) , so the projections are onto lines through v perpendicular to L .

The angle between these lines equals the angle between the planes. Let u be the closest point on Span(1 , 0 , 1) to ( 0 , sin ( 1 / √ K ) , cos ( 1 / √ K ) ) ; then u = ( 1 2 + o (1) ) (1 , 0 , 1) . Thus the plane angle is

<!-- formula-not-decoded -->

Each of the remaining 2 K -1 projections multiplies dist( w , v ) by cos ( (1 + o (1)) √ 2 / √ K ) , so overall it is scaled by e -2 + o (1) .

Originally, so afterwards the distance is e -6 √ 2 2 + o (1) .

<!-- formula-not-decoded -->

The direction from v is parallel to

<!-- formula-not-decoded -->

so the final position of w T is

<!-- formula-not-decoded -->

Hence, the x -coordinate contributes the following approximate loss due to the first half of the tasks:

<!-- formula-not-decoded -->

## E.2.3 Experimental results

Constructing the tasks as explained in App. E.2.1, using QR decomposition to acquire X for each solution subspace, we observe that the task orderings for greedy MD (Definition 3.1) and greedy MR (Definition 3.2) coincide, as explained in App. E.2.2, for all T ∈ { 4 · 10 i -1 | i = 1 , 2 , . . . , 7 } . As observed in Figure 18a, greedy ordering results in alternating between the first two task groups until these are depleted, then switching to the other two task groups. Loss is is diminishing in a linear rate during learning the first half of the tasks, then increases to the predicted value (Eq. (8)) during learning the second half.

<!-- image -->

- (a) Chosen task group under greedy ordering.
- (b) Average loss during continual learning under greedy ordering.

<!-- image -->

Figure 18: Example of continual learning the 3d adversarial construction with greedy ordering. Chosen tasks coincide for MD and MR greedy orderings. K = 1000 , T = 3999 .

In Figure 19, we show Example 5.1 holds, and the validity of the theoretical value (Eq. (8)).

Figure 19: Final average loss after greedy ordering for the 3d adversarial construction, for different task counts. T = 4 K +1 .

<!-- image -->

## E.2.4 Code for reproducibility

```
1 import numpy as np 2 def orth_complement_rows(W, *, rtol=1e-12): 3 """ Given W in R^{(d-r)×d} (full row rank), return X in R^{r×d} 4 whose rows form an orthonormal basis of the orthogonal complement 5 of the row-space of W. """ 6 W = np.asarray(W, dtype=float) 7 d = W.shape[1] 8 # Full QR of W.T → Q is d×d and orthogonal 9 # The first rank columns of Q span the rows of W; 10 # the remaining columns span their orthogonal complement. 11 Q, _ = np.linalg.qr(W.T, mode='complete') # Q in R^{d×d} 12 rank = np.linalg.matrix_rank(W, tol=rtol) 13 # Take the last r = d -rank columns of Q, transpose to get rows 14 X = Q[:, rank:].T # X in R^{r×d} 15 return X 16 17 ORDERING = 'MD' # 'MR' / 'MD' 18 K = 1000 19 w1 = np.array([[0, np.sin(1/np.sqrt(K)), np.cos(1/np.sqrt(K))]]) 20 w2 = np.array([[0, -np.sin(1/np.sqrt(K)), np.cos(1/np.sqrt(K))]]) 21 w3 = np.array([[0, np.sin(0.5/np.sqrt(K)), np.cos(0.5/np.sqrt(K))], [1/np.sqrt(2), 0, 1/np.sqrt(2)]]) ↪ → 22 w4 = np.array([[0, -np.sin(0.5/np.sqrt(K)), np.cos(0.5/np.sqrt(K))], [1/np.sqrt(2), 0, 1/np.sqrt(2)]]) ↪ → 23 ws = [w1, w2, w3, w4] 24 tasks = [orth_complement_rows(w) for w in ws] 25 projections = [np.eye(3) -np.linalg.pinv(X) @ X for X in tasks] 26 total_collection_count = [K-1, K, K ,K] 27 T = np.sum(total_collection_count) 28 collection_count = total_collection_count.copy() 29 w = np.array([0, np.sin(1/np.sqrt(K)), np.cos(1/np.sqrt(K))]) 30 residual_per_projection = [np.linalg.norm(X @ w)**2 for X in tasks] 31 total_loss = (1/T) * np.asarray(residual_per_projection) @ np.asarray(total_collection_count) ↪ → 32 losses = [total_loss] 33 for i in range(T): 34 chosen_task = None 35 distance = -np.inf 36 new_w, new_w_candidate = None, None 37 for task_index in range(4): 38 if collection_count[task_index] > 0: 39 if ORDERING == 'MD': 40 new_w_candidate = projections[task_index] @ w 41 new_distance = np.linalg.norm(new_w_candidate -w)**2 42 elif ORDERING == 'MR': 43 new_distance = np.linalg.norm(tasks[task_index] @ w)**2 44 if new_distance > distance: 45 chosen_task = task_index 46 distance = new_distance 47 new_w = new_w_candidate 48 if new_w is None: 49 new_w = projections[chosen_task] @ w 50 w = new_w 51 collection_count[chosen_task] -= 1 52 residual_per_projection = [np.linalg.norm(X @ w)**2 for X in tasks] 53 total_loss = (1/T) * np.asarray(residual_per_projection) @ np.asarray(total_collection_count) ↪ → 54 losses.append(total_loss)
```

Listing 1: Code for 3d adversarial construction.

## F Appendix to Section 5.2: Single-pass vs. repetition

## F.1 Appendix to the upper bound for greedy orderings with repetition (Theorem 5.3)

Recall Theorem 5.3. Under a Maximum Distance greedy ordering with repetition ( τ MD -R ) over T jointly-realizable tasks, the loss of Scheme 1 after k ≥ 2 iterations is upper bounded as,

<!-- formula-not-decoded -->

The main propositions leading to the proof of Theorem 5.3 are in App. F.1.2, with auxiliary claims in App. F.1.3. We use geometric analysis to derive an upper bound on how fast iterates can grow, and then bound the iterates relative to the original distance to the teacher w ⋆ . Finally, we upper-bound the loss by the greedy iterate size when repetitions are allowed.

## F.1.1 Comparison to convergence rates of other task orderings

Table 1: Loss bounds in continual linear regression over jointly realizable tasks (based on Table 1 of Evron et al. [35]). The presented bounds are 'worst case': upper bounds apply to any collection of T jointly realizable tasks, while lower bounds are achieved by specific constructions. Bounds for random orderings apply to the expected loss. We omit scaling terms ( ∥ w ⋆ ∥ 2 R 2 ) and constant multiplicative factors (which are mild). ≜

Notation: k = iterations; d = dimensions; ¯ r, r max = average/maximum data matrix ranks; a, b min( a, b ) .

| Bound   | Paper / Ordering      | Single-pass Greedy   | Greedy with Repetition   | Random w/o Replacement   | Random with Replacement           |
|---------|-----------------------|----------------------|--------------------------|--------------------------|-----------------------------------|
| Upper   | Evron et al. [33]     | -                    | -                        | -                        | d - ¯ r k                         |
|         | Evron et al. [35]     | -                    | -                        | 1 4 √ T , d - ¯ r T      | 1 4 √ k , √ d - ¯ r k , √ T ¯ r k |
|         | Attia et al. [10]     | -                    | -                        | 1 √ T                    | 1 √ k                             |
|         | Ours                  | None                 | 1 3 √ k                  | -                        | -                                 |
| Lower   | Evron et al. [33] (*) | 1 T                  | 1 k                      | 1 T                      | 1 k                               |
|         | Ours                  | Ω(1)                 | -                        | -                        | -                                 |

(*) Although Evron et al. [33] did not state these lower bounds explicitly, their proof of Theorem 10 provides a 2 -task construction which, when replicated ⌊ T/ 2 ⌋ times, produces a Θ(1 /k ) bound for both random and greedy orderings with any general T .

## F.1.2 Deriving the bound

<!-- formula-not-decoded -->

Proof. Let us denote

Using orthogonality:

<!-- formula-not-decoded -->

Then u , v ∈ ker( P ) and are orthogonal to Pw 0 , Pw 1 ∈ im( P ) , respectively. Now decompose:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Subtracting Eq. (10) from Eq. (9), we get

<!-- formula-not-decoded -->

So,

<!-- formula-not-decoded -->

Corollary F.2. Consider the case in Eq. (2) , i.e., ( w t ) k t =0 are iterates such that w 0 = 0 and ∀ t ∈ [ k ] , ( w t -w ⋆ ) = P τ ( t ) ( w t -1 -w ⋆ ) , then ∀ k ≥ 2 ,

<!-- formula-not-decoded -->

Proof. We repeatedly apply the above proposition:

<!-- formula-not-decoded -->

Corollary F.3. If the first step is MD greedy, i.e., ∀ m ∈ [ T ] , ∥ ( I -P m ) ( w 0 -w ⋆ ) ∥ ≤ ∥ w 0 -w 1 ∥ , then ∀ k ≥ 2 ,

<!-- formula-not-decoded -->

Proposition F.4. Under greedy MD ordering, either single-pass or with repetition, we have ∀ k ≥ 1 ,

<!-- formula-not-decoded -->

Proof. Notice that

<!-- formula-not-decoded -->

where we used the fact that the orthogonal projection of a given point on a subspace is the closest point in this subspace to the given point in the Euclidean-norm sense.

If ∥ w k +1 -w k ∥ &lt; ∥ w 0 -w 1 ∥ , the proposition follows immediately. Otherwise ∥ w k +1 -w k ∥ ≥ ∥ w 0 -w 1 ∥ , and thus

( ∥ w k +1 -w k ∥ - ∥ w 0 -w 1 ∥ ) 2 ≤ ∥ w 0 -w k ∥ 2 . Plugging this into Corollary F.3, we get ( ∥ w k +1 -w k ∥ - ∥ w 0 -w 1 ∥ ) 2 ≤ ∥ w 0 -w k ∥ 2 ≤ ∥ w 0 -w 1 ∥ 2 -k ∑ t =2 ∥ w t -w t -1 ∥ 2 +2 ∥ w 0 -w 1 ∥ k ∑ t =2 ∥ w t -w t -1 ∥ ∥ w k +1 -w k ∥ 2 + k ∑ t =2 ∥ w t -w t -1 ∥ 2 ≤ 2 ∥ w 0 -w 1 ∥ ( ∥ w k +1 -w k ∥ + k ∑ t =2 ∥ w t -w t -1 ∥ ) ∥ w 0 -w 1 ∥ ≥ ∑ k +1 t =2 ∥ w t -w t -1 ∥ 2 2 ∑ k +1 ∥ w t -w t -1 ∥ .

<!-- formula-not-decoded -->

Using the same derivation, when all steps are greedy, we see ∥ w 2 -w 1 ∥ ≥ ∑ k +1 t =3 ∥ w t -w t -1 ∥ 2 2 ∑ k +1 ∥ w t -w t -1 ∥

t =3 , . . . , ∥ w k -w k -1 ∥ ≥ ∑ k +1 t = k +1 ∥ w t -w t -1 ∥ 2 2 ∑ k +1 t = k +1 ∥ w t -w t -1 ∥ = ∥ w k +1 -w k ∥ 2 2 ∥ w k +1 -w k ∥ = 1 2 ∥ w k +1 -w k ∥ , acquiring a lower bound ∀ t ∈ [ k ] : ∥ w t -w t -1 ∥ ≥ ∑ k +1 j = t +1 ∥ w j -w j -1 ∥ 2 2 ∑ k +1 j = t +1 ∥ w j -w j -1 ∥ . Hence, we define the sequence ( C t ) k +1 t =1 the backward recurrence C k +1 = ∥ w k +1 -w k ∥ and ∀ t ∈ [ k ] , C t = ∑ k +1 j = t +1 C 2 j 2 ∑ k +1 j = t +1 C j . By applying

Claim F.7, using the sequences a i ≜ ∥ w k +1 -i -w k -i ∥ , b i ≜ C k +1 -i for all i ∈ { 0 , . . . , k } , we observe that ∥ w 0 -w 1 ∥ ≥ C 1 .

We investigate the sequence ( C t ) in Corollary F.10, where we prove that if C k +1 = ∥ w k +1 -w k ∥ 0 , this sequence maintains ∀ k ≥ 1 ,

<!-- formula-not-decoded -->

(and indeed if ∥ w k +1 -w k ∥ = 0 , the proposition follows immediately), and thus we get that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by &gt;

Lemma F.5. Under greedy MD ordering, either single pass or with repetition, we have ∀ k ≥ 1 ,

<!-- formula-not-decoded -->

Proof. Applying the above proposition for a starting index t -1 (instead of 0 ), we see that ∀ k &gt; t ,

<!-- formula-not-decoded -->

From the Pythagorean theorem we have

<!-- formula-not-decoded -->

Plugging in the above proposition, we get

<!-- formula-not-decoded -->

where we used i =1 i 1 x

## We are now ready to prove Theorem 5.3:

Proof of Theorem 5.3. Under greedy MD ordering with repetitions we have:

<!-- formula-not-decoded -->

## F.1.3 Auxiliary claims

Claim F.6. For s, r &gt; 0 and α ≥ 1 2 define

<!-- formula-not-decoded -->

- (a) For all s, r &gt; 0 , we have ∂f s,r ( α ) ∂α ≥ 0 on α ∈ [ 1 2 , ∞ ) ; hence f is non-decreasing in α .
- (c) For all α ∈ [ 1 2 , 1 ] and r &gt; 0 , we have ∂f s,r ( α ) ∂s ≥ 0 , hence f s,r ( α ) is non-decreasing in s on that α -range.
- (b) For all α ≥ 1 2 and s &gt; 0 , we have ∂f s,r ( α ) ∂r &gt; 0 ; hence f s,r ( α ) is strictly increasing in r .

Proof. Below, we prove each statement separately, in order.

- (a) Derivative with respect to α :

<!-- formula-not-decoded -->

Expanding the numerator:

<!-- formula-not-decoded -->

Thus

<!-- formula-not-decoded -->

- (b) Derivative with respect to r :

<!-- formula-not-decoded -->

Expand the numerator term-by-term:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence ∂f s,r /∂r &gt; 0 .

- (c) Derivative with respect to s :

<!-- formula-not-decoded -->

For α ∈ [ 1 2 , 1] the factor α (1 -α ) ≥ 0 , yielding a non-negative derivative.

Claim F.7. Let n ≥ 1 and let ( a i ) n i =0 be a sequence of strictly positive numbers such that

<!-- formula-not-decoded -->

Define a second sequence ( b i ) n i =0 recursively by

<!-- formula-not-decoded -->

Then a n ≥ b n .

Proof. For either sequence x ∈ { a, b } and each k ≥ 1 set

<!-- formula-not-decoded -->

For k ≥ 1 we can express the defining relations as

<!-- formula-not-decoded -->

and, using Q ( x ) k -1 = R ( x ) k -1 S ( x ) k -1 ,

<!-- formula-not-decoded -->

where f s,r ( α ) is defined as in Claim F.6.

We now prove by induction that ∀ k ∈ { 0 , . . . , n } ,

<!-- formula-not-decoded -->

Base case k = 0 . All equalities a 0 = b 0 , R ( a ) 0 = R ( b ) 0 = a 0 , S ( a ) 0 = S ( b ) 0 = a 0 hold. Induction step k → k +1 . Assume the inequalities hold for k .

- (i) Comparing a k +1 and b k +1 . By the recursion and the inductive hypothesis,

<!-- formula-not-decoded -->

- (ii) Comparing S ( a ) k +1 and S ( b ) k +1 . From the above and the inductive hypothesis,

<!-- formula-not-decoded -->

- .
- (iii) Comparing R ( a ) k +1 and R ( b ) k +1 . From ( ∗ ) we have

<!-- formula-not-decoded -->

where a k +1 R ( a ) k ≥ 1 2 . By the inductive hypothesis S ( a ) k ≥ S ( b ) k and R ( a ) k ≥ R ( b ) k , and by Claim F.6 the function f s,r ( α ) is (a) increasing in α , (b) increasing in r for all α ≥ 1 2 , and (c) increasing in s when α = 1 2 . Therefore

<!-- formula-not-decoded -->

Hence the inequalities hold for k + 1 , concluding the proof by induction. Taking k = n yields a n ≥ b n , proving the claim.

Claim F.8. ∀ x &gt; -1 , log (1 + x ) ≥ x 1+ x .

Proof. For x &gt; -1 , define f ( x ) = ln(1 + x ) -x 1+ x , then

<!-- formula-not-decoded -->

Since f ′ ( x ) &lt; 0 for x &lt; 0 and f ′ ( x ) &gt; 0 for x &gt; 0 , the point x = 0 , where f (0) = 0 , is the global minimum of f . Hence ln(1 + x ) ≥ x 1 + x for all x &gt; -1 .

Claim F.9. Let λ &gt; µ &gt; 0 , κ &gt; 0 . Define the sequences ( A n ) ∞ n =0 , ( B n ) ∞ n =0 , ( ˜ C n ) ∞ n =0 by A 0 &gt; 0 , B 0 &gt; 0 , ˜ C 0 = κ B 0 A 0 , and for n ≥ 1 ,

<!-- formula-not-decoded -->

then ∀ n ≥ 1 , ˜ C n ≥ ˜ C 0 · γn -λ -µ 2 λ -µ , where γ ≜ exp ( -2 λ ( λ -µ ) µ (2 λ -µ ) ) .

Proof. It is readily seen that ∀ n ≥ 0 , A n &gt; 0 and B n &gt; 0 , immediately by induction. Define the helper sequence ∀ n ≥ 0 , f n = A 2 n B n &gt; 0 , then A n = √ B n f n and ∀ n ≥ 1 ,

<!-- formula-not-decoded -->

then f n ≥ f n -1 +2 λ -µ ≥ f 0 + n (2 λ -µ ) = A 2 0 B 0 + n (2 λ -µ ) . Now, observe ∀ n ≥ 1 ,

<!-- formula-not-decoded -->

and note that ∀ i ≥ 0 , 0 &lt; λ -µ f i + λ &lt; 1 . Taking the log we get,

<!-- formula-not-decoded -->

When n = 1 , we have log ˜ C 1 ≥ log ˜ C 0 -λ -µ µ &gt; log ˜ C 0 -λ -µ µ -λ -µ 2 λ -µ , hence ˜ C 1 ≥ ˜ C 0 exp ( -2 λ ( λ -µ ) µ (2 λ -µ ) ) , concluding the proof for all n ≥ 1 .

Corollary F.10. For ( C t ) k +1 t =1 defined by the backwards recurrence C k +1 &gt; 0 and ∀ t ∈ [ k ] , C t = ∑ k +1 j = t +1 C 2 j 2 ∑ k +1 j = t +1 C j , we have that ∀ k ≥ 1 ,

<!-- formula-not-decoded -->

Proof. By defining ∀ n ∈ { 0 , . . . , k } : A n = k +1 ∑ j = k +1 -n C j , B n = k +1 ∑ j = k +1 -n C 2 j , we note that A 0 = C k +1 &gt; 0 , B 0 = C 2 &gt; 0 , and for all 1 ≤ n ≤ k ,

<!-- formula-not-decoded -->

Defining the sequence ˜ C n ≜ κ B n A n , with κ = 1 2 , we have from the above claim ∀ n ≥ 1 ,

<!-- formula-not-decoded -->

where λ = 1 2 , µ = 1 4 , γ = exp ( -2 λ ( λ -µ ) µ (2 λ -µ ) ) = e -4 / 3 , and ˜ C 0 = 1 2 B 0 A 0 = 1 2 C k +1 . Thus, we have,

<!-- formula-not-decoded -->

Finally, observe that ˜ C n ≜ 1 2 B n A n = ∑ k +1 j = k +1 -n C 2 j 2 ∑ k +1 j = k +1 -n C j = C k -n , for all 0 ≤ n ≤ k -1 , and plugging in n = k -1 ≥ 1 , when k ≥ 2 , we get

<!-- formula-not-decoded -->

Specifically, when k = 1 we get C 1 = C 2 2 = C k +1 · 1 2 ≥ C k +1 · e -4 / 3 2 k -1 / 3 , concluding the proof for all k ≥ 1 .

## F.2 Regression experiments on single-pass vs. repetition

Here, we extend the experiment on the effect of repetition (Figure 5) to additional data regimes. Figure 5 was produced using the same data as Figure 3a, i.e., d = 100 , r = 10 , T = 50 . Throughout this section, the Maximum Distance ordering (Definition 3.1) is denoted by 'Greedy' for brevity.

Isotropic data. We find that the conclusions of Section 5.2 extend to more regimes: repetitions are beneficial in greedy ordering while replacement harms random ordering.

Figure 20: The effect of repetitions for varying dimensions d and ranks r of the data matrices, for isotropic data. T = 50 . Random orderings without-replacement consistently outperform their with-replacement counterparts. In contrast, greedy orderings with repetition outperform the singlepass variant. As explained in Section 5.2, repetition in greedy orderings is beneficial because it enables larger steps (and converging faster to the joint solution w ⋆ ).

<!-- image -->

Figure 21: The effect of repetitions for varying task count T , for isotropic data. d = 100 , r = 10 . As task count increases, the differences between with and without repetition diminish. Notice, however, that in all subplots we only learn the first 50 tasks. It is readily observed in the left subplot that the effect of repetition becomes pronounced in the latter parts of the task sequences. As can be expected, repetition offers less benefit when many diverse, unexplored tasks remain.

<!-- image -->

Anisotropic data. Next, we observe that the effect of repetitions diminishes for correlated data.

Figure 22: The effect of repetitions for varying dimensions d and ranks r of the data matrices, for anisotropic data. T = 50 . Previously in App. B (Figure 10), we explained that the performance of all ordering strategies deteriorates when the pairwise distances between task solution subspaces are small. This effect is even more pronounced for low-rank tasks (left columns), where the complementary high-rank solution subspaces overlap substantially. We also observe that in those low-rank regimes, different orderings exhibit more similar performance, and consequently, repetitions become less impactful. While we cannot fully explain the small performance degradation observed when allowing repetitions in greedy ordering for low-dimension, low-rank settings (top-left subfigure), it may be related to the slower convergence to the joint solution w ⋆ , nullifying the effect of the loss upper bound induced by the distance to w ⋆ (Proposition 2.6; see also Figure 23 below).

<!-- image -->

Below, we observe another aspect related to the 'diminished' effect of repetitions in this setting.

Figure 23: The distance to w ⋆ is a loose upper bound on the loss for high similarity tasks. Greedy ordering with d = 60 , T = 50 . While repetitions do lead to a slightly faster decrease in the squared distance to w ⋆ , this decrease remains slow when tasks are highly similar (as in the low-rank setting on the left). Consequently, the upper bound of Proposition 2.6 becomes looser, as the loss itself decreases more rapidly. This discrepancy makes it difficult to draw firm conclusions about the convergence of the loss, including the exact impact of repetitions. A similar gap between the loss and the distance to the joint solution w ⋆ in highly similar tasks was also noted by Evron et al. [33, Section 5.1 therein].

<!-- image -->

Remark. We omit the figure for the corresponding experiment with varying number of tasks T , as it offers no additional insights beyond those shown in Figure 21.

## G Appendix to Section 5.3: Hybrid task ordering

## G.1 Hybrid ordering scheme

Motivated by the success of greedy Kaczmarz and importance sampling methods [75, 4], as well as recent convergence bounds for random orderings in continual learning [35, 10], we introduce a 'hybrid' strategy in Section 5.3. Hybrid schemes have also been explored in the contexts of Kaczmarz methods [75, 25], coordinate descent [36], and multiplicative Schwarz methods [40].

In this approach, tasks are selected greedily as long as the decrements ∥ w t -1 -w t ∥ 2 (see Eq. (3)) remain above a threshold; afterward, selection switches to random sampling. The proposed hybrid method can be used with either the greedy Maximum Distance rule (Definition 3.1), as in Scheme 3, or the greedy Maximum Residual rule (Definition 3.2) as in Scheme 4.

```
Scheme 3 MDhybrid ordering ( τ H -MD) Input: β MD ∈ [ 0 , ∥ w 0 -w ⋆ ∥ 2 ] For each iteration t = 1 , . . . , T : # Use greedy selection as long as the threshold is met m ′ ← argmax m ∈ [ T ] \ τ H -MD (1: t -1) ∥ ( I -P m )( w t -1 -w ⋆ ) ∥ 2 # Compute greedy selection If ∥ ( I -P m ′ )( w t -1 -w ⋆ ) ∥ 2 ≥ β MD Then τ H -MD ( t ) ← m ′ Else Break τ H -MD ( t : T ) ∼ Unif ([ T ] \ τ H -MD (1 : t -1)) # Choose remaining tasks randomly w/o replacement
```

```
Scheme 4 MRhybrid ordering ( τ H -MR) Input: β MR ∈ [ 0 , R 2 ∥ w 0 -w ⋆ ∥ 2 ] # Reminder: R ≜ max m ∈ [ T ] ∥ X m ∥ For each iteration t = 1 , . . . , T : # Use greedy selection as long as the threshold is met m ′ ← argmax m ∈ [ T ] \ τ H -MR (1: t -1) ∥ X m w t -1 -y m ∥ 2 # Compute greedy selection If ∥ X m ′ w t -1 -y m ′ ∥ 2 ≥ β MR Then τ H -MR ( t ) ← m ′ Else Break τ H -MR ( t : T ) ∼ Unif ([ T ] \ τ H -MR (1 : t -1)) # Choose remaining tasks randomly w/o replacement
```

While our analysis sets the threshold β using ∥ w 0 -w ⋆ ∥ and R , the hybrid methods remain useful, e.g., with a heuristic β .

Analytically, using a suitable threshold β , any upper bound for without-replacement random orderings, e.g., an O ( 1 / √ k ) bound [10], can extend to our hybrid schemes, showing again that they avoid the failure mode of Section 5.1, as shown in the following Lemma G.1. Moreover, we show that the upper bound that we derive for hybrid orderings continues to improve as long as the stopping criterion is not triggered. Put more simply, it is beneficial to follow the greedy ordering as long as the resulting iterates are 'large enough' (however, the actual stopping time will depend on the data).

## G.2 Hybrid ordering upper bound

Lemma G.1 (Hybrid ordering bound) . Consider any known upper bound for the expected normalized loss (Definition 2.3) in random ordering without replacement over T jointly-realizable tasks, of the form E τ Unif [ L ( w τ Unif T )] ≤ C T α with C &gt; 0 and 0 &lt; α ≤ 1 , such that C T α ≤ 1 2 -α . Then, defining ˜ β min ≜ T α -C (1 -α ) CT , the following holds:

When β MD ≥ ∥ w 0 -w ⋆ ∥ 2 ˜ β min (or β MR ≥ R 2 ∥ w 0 -w ⋆ ∥ 2 ˜ β min ), the loss under Scheme 3 (or Scheme 4) is upper bounded as E τ H [ L ( w τ H T )] ≤ C T α . Furthermore, choosing β MD = ∥ w 0 -w ⋆ ∥ 2 ˜ β min (or β MR = R 2 ∥ w 0 -w ⋆ ∥ 2 ˜ β min ), i.e., postponing the stopping time as much as our (data-dependent) condition allows, leads to the tightest upper bound (in our derivations).

Proof. For MD and MR hybrid orderings, we denote β MD = ˜ β ∥ w 0 -w ⋆ ∥ 2 and β MR = ˜ βR 2 ∥ w 0 -w ⋆ ∥ 2 , respectively. Note the following holds for all m ∈ [ T ] , w ∈ R d :

<!-- formula-not-decoded -->

So, when ∥ X m w t -1 -y m ∥ 2 ≥ ˜ βR 2 ∥ w 0 -w ⋆ ∥ 2 , immediately ∥ ( I -P m ) ( w t -1 -w ⋆ ) ∥ 2 ≥ ˜ β ∥ w 0 -w ⋆ ∥ 2 , i.e., if the condition for continuing with greedy MR steps in Scheme 4 holds, then ∥ ( I -P m ) ( w t -1 -w ⋆ ) ∥ 2 ≥ ˜ β ∥ w 0 -w ⋆ ∥ 2 (this holds by definition for Scheme 3).

The last step t for which max m ∈ [ T ] \ τ (1: t -1) ∥ ( I -P m ) ( w t -1 -w ⋆ ) ∥ 2 ≥ ˜ β ∥ w 0 -w ⋆ ∥ 2 consecutively holds is some t = s , where 0 ≤ s ≤ T . The following holds:

<!-- formula-not-decoded -->

We are reminded of the definition for the (normalized) loss for a solution vector w with a task collection T , starting from some starting point w 0 and having a minimum norm joint solution w ⋆ :

<!-- formula-not-decoded -->

Running the hybrid scheme on the task collection [ T ] yields the following expected loss:

<!-- formula-not-decoded -->

where ( 1 ) is since s ≤ T , and ( 2 ) is since w s is deterministic. This means we can plug in any upper bound for the expected normalized loss of the random ordering, for the collection of T -s tasks [ T ] \ τ (1 : s ) with the starting point w s , replacing dependence on T with T -s . If we have an upper bound for the expected normalized loss of random ordering of f ( T ) tasks, which is a positive and decreasing function of T , we obtain the following upper bound for hybrid ordering:

<!-- formula-not-decoded -->

As a sanity check, setting s = 0 removes the greedy iterates, and the bound reduces to that of random ordering.

For the rest of the proof, we work with bounds of the following form:

<!-- formula-not-decoded -->

such that f ( T ) ≤ 1 , ∀ C, α, T . This only means that we ignore the cases where the bound on random orderings is entirely vacuous, i.e., it is larger than 1 .

We want a condition on ˜ β for which continuing with greedy iterates as long as ∥ ( I -P m ) ( w t -1 -w ⋆ ) ∥ 2 ≥ ˜ β ∥ w 0 -w ⋆ ∥ 2 , necessarily improves the bound. This means we want the bound to decrease with s . Thus, we demand ∀ s ∈ [ T ] : d d s ( 1 -˜ βs T ( s +( T -s ) f ( T -s )) ) ≤ 0 :

<!-- formula-not-decoded -->

When demanding this expression to be ≤ 0 , we get:

<!-- formula-not-decoded -->

As a sanity check, note that since f ( T -s ) ≤ 1 and f ′ ( T -s ) ≤ 0 , the numerator is non-negative and the denominator is positive.

Continuing:

<!-- formula-not-decoded -->

We demand this holds ∀ s ∈ [ T ] . We assume, for now, that C ( T -s ) α ≤ 1 . We will later examine the other case. Thus, plugging in f ( T -s ) = C ( T -s ) α :

<!-- formula-not-decoded -->

We are reminded that we assumed C ( T -s ) α ≤ 1 , thus ( T -s ) α ≥ C &gt; C (1 -α ) , so the denominator here is positive. Denote ˜ β -1 ≤ g ( s ) ≜ s + C ( T -s ) ( T -s ) α -C (1 -α ) . In order to find an upper bound on ˜ β -1 which holds for all s , we look for the minimum of g ( s ) . Differentiating, we get:

<!-- formula-not-decoded -->

Hence g ′ ( s ) ≥ 0 for s small enough such that ( T -s ) α ≥ 2 C (1 -α ) , and g ′ ( s ) &lt; 0 for larger values, up to the maximum value under the current assumption of s = T -C 1 /α . Hence, the minimum of g ( s ) in [ 0 , T -C 1 /α ] will be one of the boundary points:

<!-- formula-not-decoded -->

This can only be negative when T &lt; ( C (2 -α )) 1 /α , i.e., f ( T ) = C T α &gt; 1 2 -α , which is a case not covered in this Lemma, since we assumed C T α ≤ 1 2 -α (and the bound is quite useless if it is larger than 1 2 anyway). Thus, it is guaranteed that the lowest upper bound for ˜ β -1 is for s = 0 , and we get:

<!-- formula-not-decoded -->

Under this choice of ˜ β , the upper bound of Eq. (11) is monotonically decreasing with s as long as C ( T -s ) α ≤ 1 .

We now address the case of s large enough such that C ( T -s ) α &gt; 1 . In this case, our upper bound from Eq. (11) becomes: E τ L ([ T ] , w 0 ) [ w T ] ≤ 1 -˜ βs T ( s +( T -s ) · 1) = 1 -˜ βs , which is also monotonically decreasing with s . From continuity of the bound at s = T -C 1 /α (it does not matter that this might not be an integer), we get that the bound is monotonically decreasing with s for all s &gt; 0 when ˜ β ≥ ˜ β min = T α -C (1 -α ) CT , and thus beats the bound for random ordering, achieved at s = 0 .

## G.3 Hybrid ordering experiments

Figure 6 was acquired using the same data as Figure 3a, and using the dimension and rank-dependent upper bound of 2 ( d -r ) /k from Evron et al. [35] to set β , since the universal bound of 14 /k 1 / 4 requires more than 50 iterations to be effective. The hybrid method results with intermediate performance between random and greedy. The figures demonstrate that the hybrid approach combines trends we have seen earlier (App. B) for random and greedy MD, in terms of the effect of dimension, rank, task count and task correlation on the performance.

Figure 24: Hybrid performance for varying dimensions d and ranks r of the data matrices, for isotropic data. T = 50 . In high-rank and/or low-dimensional settings, the rank-dependent upper bound employed by the hybrid strategy in this case is lower, prompting an earlier transition from the greedy to the random phase. Interestingly, the performance of the random phase within the hybrid method is slightly inferior to that of fully random ordering-possibly because the initial greedy steps deplete the set of 'extreme' tasks that would otherwise drive greater progress.

<!-- image -->

Figure 25: Hybrid performance for varying task count T , for isotropic data. d = 100 , r = 10 . We see similar trends. Note that the previously observed slight drop in performance of the random iterates following the greedy phase is less pronounced with higher task counts, possibly since more extreme tasks remain available for selection.

<!-- image -->

Anisotropic data. Similar trends were observed under anisotropic data, and we therefore omit the corresponding figures for brevity.

## H Appendix to Section 4.1: Code snippet for regression experiments

The regression experiments are intentionally simple, and for completeness and reproducibility we provide a short code snippet. Running it generates a basic linear regression experiment on isotropic data, comparing random and greedy orderings (Eq. (4), Def. 3.1 and 3.2).

```
1 # Minimal Block Kaczmarz experiment with a runnable demo + simple plot 2 import numpy as np 3 import matplotlib.pyplot as plt 4 from numpy.linalg import pinv 5 from typing import List, Tuple, Optional 6 7 # --------Core utilities --------8 9 def generate_data(r: int, d: int, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[tuple], List[np.ndarray], List[np.ndarray]]: ↪ → ↪ → 10 """ 11 Returns: 12 X: (T*r, d) matrix; b: (T*r,) labels; w_true: (d,) teacher. 13 blocks: list of (X_t, b_t) with X_t in R^{r×d}, b_t in R^{r}. 14 pinv_blocks: list of pseudoinverses X_t^+. 15 md_proj: list of X_t^+ X_t (used by MD-based selection). 16 """ 17 if seed is not None: 18 np.random.seed(seed) 19 20 X_blocks = [np.random.randn(r, d) for _ in range(T)] 21 max_rad = max(np.linalg.norm(B, 2) for B in X_blocks) 22 X_blocks = [B / max_rad for B in X_blocks] 23 24 w_true = np.random.randn(d) 25 w_true /= np.linalg.norm(w_true) 26 27 X = np.vstack(X_blocks) 28 b = X @ w_true 29 30 X_blocks = [X[i*r:(i+1)*r, :] for i in range(T)] 31 b_blocks = [b[i*r:(i+1)*r] for i in range(T)] 32 pinv_blocks = [pinv(Xt) for Xt in X_blocks] 33 md_proj = [pinv_blocks[t] @ X_blocks[t] for t in range(T)] 34 blocks = list(zip(X_blocks, b_blocks)) 35 return X, b, w_true, blocks, pinv_blocks, md_proj
```

Listing 2: Data generation.

```
1 def _pick_uniform(T: int, used: Optional[List[int]], allow_repetition: bool) -> Optional[int]: ↪ → 2 if allow_repetition: 3 return int(np.random.randint(0, T)) 4 pool = list(set(range(T)) -set(used or [])) 5 return int(np.random.choice(pool)) if pool else None 6 7 8 def _pick_greedy_mr(blocks: List[tuple], w: np.ndarray, used: Optional[List[int]], allow_repetition: bool) -> Optional[int]: ↪ → 9 best, best_val = None, -np.inf 10 for i, (Xt, bt) in enumerate(blocks): 11 if (not allow_repetition) and used and i in used: 12 continue 13 val = np.linalg.norm(Xt @ w -bt) ** 2 14 if val >= best_val: 15 best, best_val = i, val 16 return best 17 18 19 def _pick_greedy_md(md_proj: List[np.ndarray], w: np.ndarray, w_true: np.ndarray, used: Optional[List[int]], allow_repetition: bool) -> Optional[int]: ↪ → ↪ → 20 best, best_val = None, -np.inf 21 for i, P in enumerate(md_proj): 22 if (not allow_repetition) and used and i in used: 23 continue 24 val = np.linalg.norm(P @ (w -w_true)) ** 2 25 if val >= best_val: 26 best, best_val = i, val 27 return best
```

Listing 3: Ordering strategies.

```
1 def block_kaczmarz( 2 blocks: List[tuple], 3 pinv_blocks: List[np.ndarray], 4 md_proj: List[np.ndarray], 5 w_true: np.ndarray, 6 T: int, 7 d: int, 8 max_iters: int, 9 selection: str = "uniform", # one of {"uniform","greedy_mr","greedy_md"} 10 allow_repetition: bool = False, 11 ) -> Tuple[np.ndarray, List[float], List[float]]: 12 """Runs Block Kaczmarz and returns (w, out_losses, dist_to_teacher) per iteration.""" ↪ → 13 w = np.zeros(d) 14 out_losses: List[float] = [] 15 dist: List[float] = [] 16 used: List[int] = [] 17 18 for _ in range(max_iters): 19 if selection == "uniform": 20 t = _pick_uniform(T, used, allow_repetition) 21 elif selection == "greedy_mr": 22 t = _pick_greedy_mr(blocks, w, used, allow_repetition) 23 elif selection == "greedy_md": 24 t = _pick_greedy_md(md_proj, w, w_true, used, allow_repetition) 25 else: 26 raise ValueError("selection must be one of {'uniform','greedy_mr','greedy_md'}") ↪ → 27 28 if t is None: # no available block under no-replacement 29 out_losses.append(np.nan) 30 dist.append(np.linalg.norm(w -w_true) ** 2) 31 continue 32 33 Xt, bt = blocks[t] 34 w += pinv_blocks[t] @ (bt -Xt @ w) # single block least squares step 35 36 if not allow_repetition: 37 used.append(t) 38 39 # metrics 40 loss = 0.0 41 for Xs, bs in blocks: 42 r = Xs @ w -bs 43 loss += np.linalg.norm(r) ** 2 44 out_losses.append(loss / T) 45 dist.append(np.linalg.norm(w -w_true) ** 2) 46 47 return w, out_losses, dist
```

Listing 4: Experiment loop.

```
1 # --------One reproducible experiment + simple figure --------2 3 # Problem size / schedule 4 d = 100 5 r = max(1, int(0.1 * d)) # rows per block (r/d = 0.1) 6 T = 50 # number of blocks 7 max_iters = T # single pass without replacement 8 seed = 42 # reproducible 9 10 # Data and precomputations 11 _, _, w_true, blocks, pinv_blocks, md_proj = generate_data(r, d, T, seed=seed) 12 13 # Compare three orderings (no replacement for apples-to-apples single pass) 14 strategies = [ 15 ("Random", {"selection": "uniform", "allow_repetition": False}), 16 ("Greedy MR", {"selection": "greedy_mr", "allow_repetition": False}), 17 ("Greedy MD", {"selection": "greedy_md", "allow_repetition": False}), 18 ] 19 20 results = {} 21 for name, opts in strategies: 22 _, out_losses, dist = block_kaczmarz( 23 blocks=blocks, 24 pinv_blocks=pinv_blocks, 25 md_proj=md_proj, 26 w_true=w_true, 27 T=T, 28 d=d, 29 max_iters=max_iters, 30 **opts 31 ) 32 results[name] = (np.asarray(out_losses), np.asarray(dist)) 33 34 # Print final metrics 35 print("Final loss and distance-to-teacher:") 36 for name in strategies: 37 key = name[0] 38 L, D = results[key] 39 # pick the last non-nan value 40 last_idx = np.where(~np.isnan(L))[0][-1] 41 print(f" {key:10s} loss={L[last_idx]:.4e} dist={D[last_idx]:.4e}") 42 43 # Simple comparison figure 44 plt.figure(figsize=(6, 4)) 45 xs = np.arange(1, max_iters + 1) 46 for name, (L, _) in results.items(): 47 plt.plot(xs, L[:max_iters], label=name) 48 plt.xlabel("Iterations (seen tasks)") 49 plt.ylabel("Average loss") 50 plt.yscale("log") 51 plt.grid(True, alpha=0.3) 52 plt.legend() 53 plt.tight_layout() 54 plt.show()
```

Listing 5: Simple ordering comparison experiment.

## I Lower bound technical appendix: Delta positivity proof

This section supplements App. E.1, which we recommend reviewing in advance. Here, we prove that ∀ d ≥ 25 , 000 , ∀ t ∈ { 2 , . . . , d -1 } , ∀ k ∈ { t, . . . , d -1 } ,

<!-- formula-not-decoded -->

In some places in our proofs, we will need a closed-form approximation of the first coordinates x k ≜ ( w k ) 1 which we obtain recursively. Such an approximation was suggested in Hucht [48]:

<!-- formula-not-decoded -->

This will be formalized and proven in App. J. In addition this gives us a lower bound x k ≥ 0 . 45 , ∀ k ∈ [ d ] when d ≥ 25 , 000 (Corollary J.3).

## I.1 Proof outline

The proof is straightforward: we decompose ∆ t,k to smaller parts, and attempt to lower bound each of these parts. We then combine all of these lower bounds to achieve an overall lower bound on ∆ t,k and find a sufficient condition on d for which this lower bound is positive. This condition, revealed in Eq. (13), is already satisfied when d ≥ 25 , 000 , concluding the proof. We begin by bounding some intermediate quantities that appear later in the derivation, and starting in App. I.3.6 we decompose and lower bound ∆ t,k .

## I.2 Auxiliary: Algebraic inequalities

<!-- formula-not-decoded -->

Proof. To show the upper bound, we define α = n/d ∈ (0 , 1] and f ( α ) = 1 -2 -α -α ln (2) , and notice that f is decreasing in (0 , 1] since

<!-- formula-not-decoded -->

Then, this means f ( α ) = 1 -2 -n/d -n ln(2) d ≤ lim α → 0 + f ( α ) = 0 as required.

Conversely, we get the lower bound by showing that the function g ( α = n d ) = 1 -2 -n/d -( n ln(2) d -n 2 ln 2 (2) 2 d 2 ) is increasing in (0 , 1] ,

<!-- formula-not-decoded -->

Claim I.2. For ∀ d, n, m ∈ N and k ∈ [ d ] , we have c nk -m ≥ 2 -n .

<!-- formula-not-decoded -->

Claim I.3. ∀ k ∈ [1 , d ] it holds that 1 -(1 -c ) ( k -1) = (( k -1) c -( k -2)) ∈ [0 , 1] .

Proof. It is clear that (1 -c ) ( k -1) ≜ ( 1 -2 -1 /d ) ( k -1) ≥ 0 . Then, we can simply show that from Claim I.1:

<!-- formula-not-decoded -->

Claim I.4. ∀ k ∈ [1 , d ] it holds that kc -( k -1) &gt; 0 .

Proof. From Claim I.1 we have 1 -c ≤ ln 2 d ⇒ c ≥ 1 -ln 2 d . Plugging that in we get:

(

ln 2

d

kc

-

(

k

-

1)

≥

k

1

-

-

k

+1 = 1

Claim I.5. ∀ k ∈ [ d ] it holds that β k ∈ [ 0 . 3 c 2 k -5 d , c 2 k -5 d ] .

Proof.

<!-- formula-not-decoded -->

Claim I.6. x k is decreasing and ∀ k ∈ [ d ] , x k ≤ 1 .

Proof. Decreasing follows immediately from positivity of β k (see Claim I.5) and the construction, and since x 1 = 1 we get ∀ k ∈ [ d ] , x k ≤ 1 .

Claim I.7. ∀ k ∈ [2 , d ] it holds that β k ≤ 1 cd .

<!-- formula-not-decoded -->

Claim I.8. ∀ a &gt; 0 , b ∈ R \ { 0 } such that a + b ≥ 0 , it holds that √ a + b &lt; √ a + b 2 √ a .

<!-- formula-not-decoded -->

Claim I.9. ∀ d ≥ 1 : 2 1 /d ≥ 1 + ln 2 d .

<!-- formula-not-decoded -->

Claim I.10. If | x k -˜ x k | ≤ ϵ and x k ≥ 0 , ∣ ∣ x 2 k -˜ x 2 k ∣ ∣ ≤ 2 x k ϵ d + ϵ 2 d .

Proof. Defining r = ˜ x k -x k , we have,

<!-- formula-not-decoded -->

)

k

d

-

ln 2

≥

1

-

ln 2

&gt;

0

.

## I.3 Proof body

<!-- formula-not-decoded -->

Proposition I.11. For any k ≥ 2 , it holds that,

<!-- formula-not-decoded -->

Proof. By construction, we have

<!-- formula-not-decoded -->

Define f z ( x ) = 1 2 ( z -√ z 2 -4 x ) for z ∈ [0 . 45 , 1] (see Claim I.6, Corollary J.3) and z 2 ≫ x &gt; 0 . Expand with Taylor: f z (0) = f z (0)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and notice that generally ∀ z 2 ≫ x &gt; 0 we have f ( n ) z ( x ) &gt; 0 .

Then, by Lagrange's form of the remainder, the error of the quadratic approximation (around x = 0 ) is given by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

since x 0 ∈ [0 , x ] .

We get that

<!-- formula-not-decoded -->

Finally, since β k ≤ c 2 k -5 d and x k -1 ∈ [0 . 45 , 1] we have

<!-- formula-not-decoded -->

Proposition I.12. For any k ≥ 2 (and d ≥ 25 , 000 ), it holds that,

<!-- formula-not-decoded -->

Proof. We employ the bounds we found for x k -1 -x k :

<!-- formula-not-decoded -->

Notice that from the bounds on β k , x k -1 , we have:

<!-- formula-not-decoded -->

Since d ≥ 10 , 000 , we obtain 2 β 3 k x k -1 ( x 2 k -1 -4 β k ) 5 / 2 ≤ 10 d β 2 k x 4 k -1 . Overall, we get

<!-- formula-not-decoded -->

Proposition I.13. For k ≥ 2 (and d ≥ 25 , 000 ), it holds that,

<!-- formula-not-decoded -->

Proof. We exploit the Taylor expansion of the following function (for z 2 ≫ x &gt; 0 ),

<!-- formula-not-decoded -->

Then, by Lagrange's form of the remainder, the error of the quadratic approximation (around x = 0 ) is given by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

since x 0 ∈ [0 , x ] .

Then, setting z = x k -1 ∈ [0 . 45 , 1] , we can now conclude that,

<!-- formula-not-decoded -->

We simplify the lower bound as ( x k -1 -x k ) 2 ≥ β 2 k x 2 k -1 + 2 β 3 k x 4 k -1 ≥ β 2 k x 2 k -1 .

Finally, for the upper bound, since β k ≤ c 2 k -5 d and x k -1 ∈ [0 . 45 , 1] we have

<!-- formula-not-decoded -->

I.3.2 Expanding the inner product ( w k -1 -w k ) ⊤ w t -1

Proposition I.14. Let t ∈ [ d ] and t &lt; k ≤ d (and d ≥ 25 , 000 ). Then,

<!-- formula-not-decoded -->

Proof. We use the expanded form of the inner product, that is,

Since we already showed x k -1 -x k ∈ [ β k x k -1 + β 2 k x 3 k -1 , β k x k -1 + β 2 k x 3 k -1 + 113 c 6 k -15 d 3 ] , we now have,

<!-- formula-not-decoded -->

and,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition I.15. For any k ≥ 2 (when d ≥ 25 , 000 ), h ( k ) ∈ [ c k -3 √ d , c k -3 √ d + 5 . 42 d 3 / 2 ] .

Proof. The lower bound is easy to obtain:

<!-- formula-not-decoded -->

To get the upper bound, we employ the inequality (1 -c ) ≜ 1 -2 -1 /d ≤ ln(2) , and get,

<!-- formula-not-decoded -->

## I.3.4 Bounding h ( k +1) h ( k )

Proposition I.16. For any k ≥ 2 (when d ≥ 500 ),

<!-- formula-not-decoded -->

Proof. We start by expanding the expression in a way that will be useful for both the upper and the lower bounds,

<!-- formula-not-decoded -->

For the upper bound. We show that,

<!-- formula-not-decoded -->

where (1) is since β k +1 &lt; β k , c &lt; 1 ; (2) is since β k &lt; 1 cd , 1 -c ≤ ln 2 d ; (3) is since x k ≤ x k -1 ; and (4) is since x k -1 ≤ 1 . To upper bound x 2 k -1 -x 2 k we use the recursive formula of x k , showing

that

<!-- formula-not-decoded -->

Back to our expression,

<!-- formula-not-decoded -->

where in the last inequality we used the fact that ∀ z &gt; 0 , √ 1 + z ≤ 1 + z 2 (since ( 1 + z 2 ) 2 = 1 + z + z 2 4 ≥ 1 + z = ( √ 1 + z ) 2 ).

For the lower bound. We show that,

<!-- formula-not-decoded -->

where (1) is since x k ≤ x k -1 . Since ( β k ) k is positive and decreasing, β 2 k +1 -β 2 k &lt; 0 , and so we can simplify the expression using the fact that √ 1 -z ≥ 1 -z, ∀ z ∈ (0 , 1) :

<!-- formula-not-decoded -->

Focusing on

<!-- formula-not-decoded -->

Using the previously derived bounds of 1 -c ∈ [ ln(2) d -ln 2 (2) 2 d 2 , ln(2) d ] , we can get,

<!-- formula-not-decoded -->

Notice that we can use the previously derived bound of 1 -c n ≤ n ln(2) d , thus obtaining

<!-- formula-not-decoded -->

Finally, we get,

<!-- formula-not-decoded -->

## I.3.5 Expanding the norm

<!-- formula-not-decoded -->

Proof. By construction we have

<!-- formula-not-decoded -->

Before, we proved that ( x k -1 -x k ) 2 ∈ [ β 2 k x 2 k -1 , β 2 k x 2 k -1 + 113 c 6 k -15 d 3 ] . Now, we show the resulting bounds for ∥ w k -1 -w k ∥ which employ that bound.

The lower bound is immediate, since

<!-- formula-not-decoded -->

The upper bound requires an additional algebraic inequality of ∀ a, b &gt; 0 : √ a + b &lt; √ a + b 2 √ a and the inequality of h ( k ) ≥ 1 √ d c k -3 , i.e.,

<!-- formula-not-decoded -->

## I.3.6 Combining the expansions

Proposition I.18. When d ≥ 25 , 000 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Keeping in mind that we wish to bound

<!-- formula-not-decoded -->

we start lower bounding the right expression. Using the bounds for ( w k -1 -w k ) ⊤ w t -1 and ∥ w k -1 -w k ∥ we derived above, we get,

<!-- formula-not-decoded -->

The right function is easily bounded as,

<!-- formula-not-decoded -->

The left function is further decomposed as,

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Overall we got,

<!-- formula-not-decoded -->

where (1) is since d ≥ 10 , 000 . Focusing on the left terms, we get the overall expression, which we need to show is positive . We again use previously-derived inequalities, to show,

<!-- formula-not-decoded -->

which we will bound separately below.

## I.3.7 The second term, A 2 ( k ) , is insignificant O ( 1 d 7 / 2 )

Proposition I.19. When d ≥ 25 , 000 ,

<!-- formula-not-decoded -->

Proof. We start from,

<!-- formula-not-decoded -->

Dissecting the terms in a ( k ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We already showed that x k x k -1 ∈ ( 1 -β k x 2 k -1 -( 1 + 10 d ) β 2 k x 4 k -1 , 1 -β k x 2 k -1 ) , and we simplify it further:

<!-- formula-not-decoded -->

Now, using the algebraic inequality ∀ z ∈ (0 , 1) , (1 -z ) 3 = 1 -3 z +3 z 2 -z 3 &gt; 1 -3 z , we get,

<!-- formula-not-decoded -->

.

Moreover, recall that we already showed that h ( k +1) h ( k ) ≥ c -5 . 5 c 2 k -5 x 2 k -1 d 2 . Now, focusing on a ( k ) ,

<!-- formula-not-decoded -->

And finally,

<!-- formula-not-decoded -->

thus concluding this part.

## I.3.8 The third term, A 3 ( k ) , is insignificant O ( 1 d 7 / 2 )

Proposition I.20. When d ≥ 25 , 000 ,

<!-- formula-not-decoded -->

Proof. Notice that,

<!-- formula-not-decoded -->

where we used the facts that 1 -c ≤ ln 2 d and h ( k ) ≤ c k -3 √ d + 5 . 42 d 3 / 2 .

Using h ( k +1) h ( k ) ∈ [ c -5 . 5 c 2 k -5 x 2 k -1 d 2 , c + 2 . 44 x 4 k c 2 k -3 d 2 ] , we finally get,

<!-- formula-not-decoded -->

## I.3.9 Back to the first term, A 1 ( k )

Proposition I.21. When d ≥ 25 , 000 ,

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

We are going to use the previously-derived lower bounds of x k x k -1 ≥ 1 -β k x 2 k -1 -( 1 + 10 d ) β 2 k x 4 k -1 and h ( k +1) h ( k ) ≥ c -5 . 5 c 2 k -5 x 2 k -1 d 2 . To lower bound β k β k +1 = 1 c 2 + 1 -c (1 -k (1 -c )) c 2 , we need a slightly stronger bound than before. Specifically, notice that for any z ∈ (0 , 1) , z 1 -z ≥ z . Then, since 1 -c ∈ [ ln(2) d -ln 2 (2) 2 d 2 , ln(2) d ] = ⇒ k (1 -c ) ∈ [ k d ln (2) -k 2 d 2 ln 2 (2) , k d ln (2) ] ⊆ (0 , 1) , and

<!-- formula-not-decoded -->

We now get,

<!-- formula-not-decoded -->

We are now ready to lower bound a ( k ) as,

<!-- formula-not-decoded -->

Using Claim I.9: 2 -c c = 2 c -1 = 2 · 2 1 /d -1 ≥ 2 ( 1 + ln(2) d ) -1 = 1 + ln(4) d , we get:

<!-- formula-not-decoded -->

Lower bounding negligible positive terms by 0 , we get,

<!-- formula-not-decoded -->

We will now simplify the least significant terms above further. We start from an upper bound to the Θ ( d -2 ) term (since its sign is negative in the expression above),

<!-- formula-not-decoded -->

Similarly, for the Θ ( d -3 ) term, we again employ the upper bound β k ≤ c 2 k -5 d ≤ 1 cd , and obtain,

<!-- formula-not-decoded -->

And so, we get the following lower bound,

<!-- formula-not-decoded -->

Back to the overall term we are trying to lower bound,

<!-- formula-not-decoded -->

where (1) is since h ( k ) ∈ [ c k -3 √ d , c k -3 √ d + 5 . 42 d 3 / 2 ] , β k +1 ≤ 1 d ; (2) is since d ≥ 10 , 000 and x k ≥ 0 . 45 . Furthermore,

<!-- formula-not-decoded -->

where (3) is since d ≥ 10 , 000 . Overall, we get,

<!-- formula-not-decoded -->

It remains to get a lower bound for β k +1 ( ln(4) d -β k x 2 k ) . First, we show

<!-- formula-not-decoded -->

where we used an algebraic property that 4 -z ≥ 1 4 , ∀ z ∈ [0 , 1] . Continuing,

<!-- formula-not-decoded -->

Below, we are going to use the closed-form approximation of x k , for which we have established | x k -˜ x k | ≤ 170 . 4 d = ϵ (Lemma J.2), and also note that ∣ ∣ x 2 k -˜ x 2 k ∣ ∣ ≤ 2 x k ϵ + ϵ 2 (Claim I.10). Also, recall that, ˜ x k = √ 1 -1 ln 4 +4 -k d ( 1 ln 4 -k d ) = √ 1 -1 ln 4 + c 2 k ( 1 ln 4 -k d ) . We now use these

relations to further refine the lower bound on b ( k ) :

<!-- formula-not-decoded -->

Focusing on the left nominator,

<!-- formula-not-decoded -->

To upper bound g ( x ) = 8 x · 4 -x (inside x ∈ [0 , 1] ), we show that

<!-- formula-not-decoded -->

solved by x = 1 ln(4) , which falls inside x ∈ [0 , 1] , meaning it is a global optimum.

The second derivative is

<!-- formula-not-decoded -->

meaning that the x = 1 ln(4) is the global maximum. Also note: ( 4 1 ln 4 ) ln 4 = 4 ⇒ 4 1 ln 4 = e . So overall, we get,

<!-- formula-not-decoded -->

Finally,

<!-- formula-not-decoded -->

Going back to β k +1 ( ln(4) d -β k x 2 k ) = β k +1 b ( k ) , we have

<!-- formula-not-decoded -->

Since c = 2 -1 /d ≥ 0 . 9999 , ∀ d ≥ 10 , 000 , and plugging in ϵ = 170 . 4 d :

<!-- formula-not-decoded -->

where (1) is since d ≥ 10 , 000 . The inside of the parenthesis is positive ∀ d ≥ ⌈ 488 . 88 0 . 14325 ⌉ = 3413 , so we can bound the expression by lower bounding β k +1 x 2 k .

<!-- formula-not-decoded -->

And then,

<!-- formula-not-decoded -->

where (1) is since x k ≥ 0 . 45 , c k ≥ c d = 0 . 5 .

## I.4 Conclusion

<!-- formula-not-decoded -->

where (1) is since c &lt; 1 , k ≥ 2 ; (2) is since 2 1 / 10000 ≤ 1 . 00007 , c -k ≤ c -d = 2 ; (3) is since x k -1 ≥ 0 . 45 ; and (4) is since d ≥ 10 , 000 . Plugging in the results of Propositions I.19, I.20 and I.21, we derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (1) is since x t -1 &gt; x k ; (2) is since x t -1 ≤ 1 ; (3) is since d ≥ 10 , 000 ; (4) is since x k ≤ 1 ; (5) is since x k ≥ 0 . 45 ; (6) is since c &lt; 1 , k ≥ 2 ; (7) is since c -2 k = 4 k/d ≤ 4 ; and (8) is since d ≥ 10 , 000 ⇒ c ≥ 0 . 9999 .

Finally, we conclude that,

<!-- formula-not-decoded -->

Hence, a sufficient condition for ( w k -1 -w k ) ⊤ w t -1 ∥ w k -1 -w k ∥ -( w k -w k +1 ) ⊤ w t -1 ∥ w k -w k +1 ∥ to be positive and monotonicity to hold, is that d ≥ ⌈ 666 . 82 0 . 0429 ⌉ = 15 , 544 . Since this is smaller than 25 , 000 , this concludes our proof of positivity of ∆ t,k .

## J Lower bound technical appendix: Properties of the recursive construction

This section complements App. E.1 by confirming that the recursive construction introduced there is well-defined; readers are encouraged to review it first for context.

Specifically, we prove that the recurrence defining the sequence ( x k ) is well-posed, in the sense that the square root is always taken over a nonnegative quantity:

Lemma J.1 (Existence of the recursive sequence) . Given the sequence ( x ) k recursively defined by x 1 = 1 , x k = x k -1 + √ x 2 k -1 -4 β k 2 , ∀ k ∈ { 2 , . . . , d } where c ≜ 2 -1 /d and β k ≜ (( k -1) c -( k -2)) c 2 k -5 d , we have ∀ d ≥ 30 , ∀ k ∈ { 2 , . . . , d } that

<!-- formula-not-decoded -->

In addition, we prove the following lemma:

Lemma J.2 (Approximation by closed-form reference) . Given the sequence ( x ) k recursively defined by x 1 = 1 , x k = x k -1 + √ x 2 k -1 -4 β k 2 , ∀ k ∈ { 2 , . . . , d } where c ≜ 2 -1 /d and β k ≜ (( k -1) c -( k -2)) c 2 k -5 d , and the sequence ˜ x k = √ 1 -1 ln 4 +4 -k d ( 1 ln 4 -k d ) , we have ∀ d ≥ 30 , ∀ k ∈ [ d ] :

<!-- formula-not-decoded -->

Before proving this lemma, we note the following will immediately hold: Corollary J.3 (Lower bound on x k ) . ∀ d ≥ 25 , 000 , ∀ k ∈ [ d ] : x k ≥ 0 . 45 .

Proof. x k is decreasing (Claim I.5), so ∀ k ∈ [ d ] :

<!-- formula-not-decoded -->

This bound is extensively used in the proof of App. I.

## J.1 Proof outline

First, we show the above holds numerically for 30 ≤ d &lt; 100 , 000 , as can be seen in Figure 26. We then prove analytically for d ≥ 100 , 000 , by constructing an ODE for which the sequence ( x ) k serves as an Euler trajectory. We then bound the distance between the solution to this ODE and a known function ˜ x ( τ ) . Combining this bound with Euler's method global truncation error bound, we obtain a bound for the distance between ( x ) k and (˜ x ) k . We then use this bound to show the existence of the sequence ( x ) k for all k ∈ { 2 , . . . , d } .

Computational resources The numerical validation took 6 hours to run on a home PC with i5-9400F CPU and 16GB RAM.

<!-- image -->

(a) | x k -˜ x k | ≤ 170 . 4 d . This is a loose, analytically derived upper bound.

<!-- image -->

(b) Actual upper bound is &lt; 4 . 5 d .

Figure 26: Numerical proof of Lemma J.2 for d&lt;100,000. Using the recursive definition of x k , we calculated the sequence for each value of d , ∀ k ∈ [ d ] , and compared with ˜ x k .

## J.1.1 Euler's method construction and bottom line

Here, we leverage the global truncation error of Euler's method to establish the bound. The auxiliary propositions supporting this result are proved in the following section. Define

<!-- formula-not-decoded -->

Then, using step size of h = 1 d in Euler's method we have the iterates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and thus

<!-- formula-not-decoded -->

which are exactly the iterates we want to solve for.

These are the Euler's iterates for the differential equation

<!-- formula-not-decoded -->

While it's hard to find an exact solution to this equation, we proved that for d ≥ 100 , 000 :

<!-- formula-not-decoded -->

in Proposition J.19, where we define the function suggested in Hucht [48],

<!-- formula-not-decoded -->

such that ˜ x k = ˜ x ( k d ) . Next, we bound the iterates using the global truncation error of Euler's method, obtaining

<!-- formula-not-decoded -->

Combining this with the previous result, Proposition J.26 yields

<!-- formula-not-decoded -->

Finally, we use this bound to show the iterates exists ∀ k ∈ [ d ] .

## J.2 Full proof

## J.2.1 Auxiliary propositions

We begin with preliminary claims and move on to the propositions used in the previous section. Claim J.4. f ( d ) ≜ -d ( 1 -2 -1 /d ) is decreasing ∀ d ≥ 1 .

<!-- formula-not-decoded -->

Claim J.5. ∀ d ≥ 1 : d ( 2 1 /d -1 ) ≥ ln 2 .

Proof. Using Taylor's expansion:

<!-- formula-not-decoded -->

Claim J.6. -d ( 1 -2 -1 /d ) ≥ -ln 2 (alternatively: 2 -1 /d ≥ 1 -ln 2 d ).

Proof. From Claim J.4 we know that -d ( 1 -2 -1 /d ) is decreasing with d, so we have,

<!-- formula-not-decoded -->

We recognize this as the definition of the derivative of 2 -x for x = 0 + , so we have:

<!-- formula-not-decoded -->

Claim J.7. β ( τ ) &gt; 0 , decreasing and convex for τ &lt; 1 ln 2 .

Proof. Reminder that β ( τ ) = ( ( dτ -1)2 -1 /d -( dτ -2) ) 2 (5 -2 dτ ) /d d , and d ≥ 1 .

Denote β ( τ ) = 1 d f ( τ ) g ( τ ) , where f ( τ ) = ( dτ -1) 2 -1 /d -( dτ -2) and g ( τ ) = 2 5 -2 dτ d .

We have ∀ τ, g ( τ ) &gt; 0 .

Note that from Claim J.6 1 -ln 2 d ≤ 2 -1 /d ≤ 1 , so:

<!-- formula-not-decoded -->

so f ( τ ) &gt; 0 for τ &lt; 1 ln 2 . Thus β ( τ ) &gt; 0 for τ &lt; 1 ln 2 .

Now we note that f ′ ( τ ) = d ( 2 -1 /d -1 ) &lt; 0 , ∀ d ≥ 1 , ∀ τ and g ′ ( τ ) = -2 ln 2 · 2 5 -2 dτ d &lt; 0 , ∀ d, ∀ τ

So:

<!-- formula-not-decoded -->

as long as g ( τ ) &gt; 0 and f ( τ ) ≥ 0 - which we get for τ &lt; 1 ln 2 .

Now note f ′′ ( τ ) = 0 and g ′′ ( τ ) = 4 ln 2 2 · 2 5 -2 dτ d &gt; 0 , so:

<!-- formula-not-decoded -->

as long as f ( τ ) ≥ 0 - which we get for τ &lt; 1 ln 2 .

Claim J.8. x ( τ ) defined by the ODE in Eq. (14) satisfies the ODE x (0) = 1 , x ′ ( τ ) = d √ x 2 -g ( τ ) -x 2 with g ( τ ) = 4 β ( τ + 1 d ) . We also have:

<!-- formula-not-decoded -->

Proof. Substituting x ′ ( τ ) = d √ x 2 -4 β ( τ + 1 d ) -x 2 in x ′ ( τ ) = d √ x 2 -g ( τ ) -x 2 we get g ( τ ) = 4 β ( τ + 1 d ) .

For τ ∈ [0 , 1] and d ≥ 3 , τ + 1 d ≤ 1 ln 2 . We get from Claim J.7 that β is decreasing, so:

<!-- formula-not-decoded -->

Remark J.9 . The solution to the ODE x ( τ ) x ′ ( τ ) = -f ( x ) , x (0) = 1 is

<!-- formula-not-decoded -->

Claim J.10. ∀ 0 ≤ a ≤ µ ≤ 1 : 1 -1 - √ 1 -µ µ a ≤ √ 1 -a ≤ 1 -a 2 .

Proof. The right side inequality is trivial: ( 1 -a 2 ) 2 = 1 -a + a 2 4 ≥ 1 -a = ( √ 1 -a ) 2 .

For the left side: denote f ( a ) = √ 1 -a . f is concave: f ′ ( a ) = -1 2 √ 1 -a , f ′′ ( a ) = -2 2 √ 1 -a 4(1 -a ) ≤ 0 . So we have ∀ 0 ≤ a ≤ µ ≤ 1 :

<!-- formula-not-decoded -->

Remark J.11 . The solution to the ODE in Eq. (14) is only defined when the quantity under the square root remains nonnegative, i.e.,

<!-- formula-not-decoded -->

Given that x (0) = 1 and that 4 β ( 1 d ) = 4 2 3 /d d &lt; 1 for all d ≥ 6 , i.e., x 2 (0) &gt; 4 β ( 1 d ) strictly , from continuity we have that x ( τ ) ≥ √ 4 β ( τ + 1 d ) &gt; 0 for all τ ∈ [0 , ζ ] , for some ζ ∈ (0 , 1] . Going forward we focus on τ ∈ [0 , ζ ] when stating facts about x ( τ ) , and eventually show that ζ = 1 in Corollary J.23.

Proposition J.12. Assuming ∀ τ ∈ [0 , ζ ] : 0 ≤ g ( τ ) x 2 ≤ 1 and x &gt; 0 , the solution of x (0) = 1 , x ′ ( τ ) = d √ x 2 -g ( τ ) -x 2 holds x ( τ ) ∈   √ 1 -d 1 - √ 1 -µ µ ∫ τ 0 g ( s ) d s, √ 1 -d 2 ∫ τ 0 g ( s ) d s   , for µ ≜ max s ∈ [0 ,ζ ] g ( s ) x ( s ) 2 .

Proof. Let 0 ≤ µ ≤ 1 such that ∀ τ ∈ [0 , ζ ] : 0 ≤ g ( τ ) x 2 ≤ µ ≤ 1 . From Claim J.10 we have:

<!-- formula-not-decoded -->

where the last transition is valid since x &gt; 0 . From Remark J.9, we know that the solution to the ODE x ( τ ) x ′ ( τ ) = -f ( x ) , x (0) = 1 is x ( τ ) = √ 1 -2 ∫ τ 0 f ( s ) d s . Put differently this means x ( τ ) = √ 1 + 2 ∫ τ 0 x ( s ) x ′ ( s ) d s (when x (0) = 1 ). We aim to plug this into the inequalities, so we now achieve the required form:

<!-- formula-not-decoded -->

Denoting A ≜ d ∫ τ 0 g ( s ) d s ≥ 0 , note that for the LHS,

<!-- formula-not-decoded -->

Hence it is legal to take a square root:

<!-- formula-not-decoded -->

Finally, plugging in x ( τ ) = 1 + 2 ∫ τ 0 x ( s ) x ′ ( s ) d s , we get:

<!-- formula-not-decoded -->

Proposition J.13. For d ≥ 100 , 000 and τ ∈ [0 , 1] , 0 ≤ d ∫ τ 0 g ( s ) d s ≤ 1 . 5821 .

Proof. For τ ∈ [0 , 1] and d ≥ 3 , τ + 1 d &lt; 1 ln 2 . We get from Claim J.7 that β is positive and thus d ∫ τ 0 g ( s ) d s = 4 d ∫ τ 0 β ( s + 1 d ) d s ≥ 0 . For the right side inequality, we have:

<!-- formula-not-decoded -->

From Claim J.5 we know that d ( 2 1 /d -1 ) ≥ ln 2 , so:

<!-- formula-not-decoded -->

Claim J.14. 1 - √ 1 -x x -1 2 ≤ x 2 for x ∈ (0 , 1] .

Proof. Note that

<!-- formula-not-decoded -->

so we define a ( x ) ≜ 1 1+ √ 1 -x .

This function is monotonically increasing, continuous and convex for x ∈ [0 , 1] :

<!-- formula-not-decoded -->

and thus from convexity we have for x ∈ (0 , 1] :

<!-- formula-not-decoded -->

Proposition J.15. For d ≥ 100 , 000 , µ ≜ max s ∈ [0 ,ζ ] g ( s ) x ( s ) 2 ≤ 19 . 158 d .

Proof. From Remark J.11 we note that x ( τ ) is positive, and since β ( τ + 1 d ) &gt; 0 , we know x ( τ ) is decreasing in [0 , ζ ] (Claim J.7), and thus the minimum of x ( τ ) 2 is x ( ζ ) 2 , and x ( ζ ) 2 ≤ x (0) 2 = 1 . Applying the upper bound of g ( s ) from Claim J.8, we get

<!-- formula-not-decoded -->

From Proposition J.12 we know that √ 1 -d 1 - √ 1 -µ µ ∫ ζ 0 g ( s ) d s ≤ x ( ζ ) .

Squaring both (positive) sides, substituting and denoting A = d ∫ ζ 0 g ( s ) d s , we get

<!-- formula-not-decoded -->

Note that A ≥ 0 (from Proposition J.13), and that f ( µ ) ≜ 1 - √ 1 -µ µ is increasing, since using Claim J.10, we have f ′ ( µ ) = 2 -µ -2 √ 1 -µ 2 µ 2 √ 1 -µ ≥ 2 -µ -2(1 -µ / 2 ) 2 µ 2 √ 1 -µ = 0 . Combining these and µ ≤ 4 2 3 /d d x ( ζ ) 2 , we get that,

<!-- formula-not-decoded -->

For simplicity denote z = 1 x ( ζ ) 2 , r = 4 2 3 /d d . Recall that z ≥ 1 , and we are looking for an upper bound for it, so we can have a lower bound for x ( ζ ) 2 . We have:

<!-- formula-not-decoded -->

Finding the roots, z 1 , 2 = 2 r + A (2 -A ) ± √ (2 r + A (2 -A )) 2 -4 r (2 A + r ) 2 r .

Since we are looking for an upper bound, we care about the smaller root:

<!-- formula-not-decoded -->

For d ≥ 100 , 000 , 4 r (2 -A ) 2 = 4 · 4 2 3 /d d (2 -A ) 2 ≤ 4 · 4 · 2 3 / 100000 100000 (2 -1 . 5821) 2 ≤ 10 -3 &lt; 1 , (we used Proposition J.13), so we can apply Claim J.14:

<!-- formula-not-decoded -->

Then, for d ≥ 10 5 : 1 x ( ζ ) 2 ≤ 4 . 7894 = ⇒ µ ≤ 4 . 7894 · 4 2 3 /d d ≤ 19 . 1576 · 2 3 / 10 5 d ≤ 19 . 158 d .

Proposition J.16. For d ≥ 100 , 000 , we have ∀ τ ∈ [0 , ζ ] , ∣ ∣ ∣ x ( τ ) -√ 1 -d 2 ∫ τ 0 g ( s ) d s ∣ ∣ ∣ ≤ 33 . 1539 d .

Proof. Denote A = d ∫ τ 0 g ( s ) d s . We know that

<!-- formula-not-decoded -->

From Proposition J.13 we have for d ≥ 100 , 000 that A ≤ 1 . 5821 , then A √ 1 -1 2 A ≤ 1 . 5821 √ 1 -1 . 5821 2 ≤ 3 . 4611 .

We further know from Claim J.14 that 1 - √ 1 -µ µ -1 2 ≤ µ 2 for µ ∈ (0 , 1] .

Combining these and applying Proposition J.15, we get:

<!-- formula-not-decoded -->

Claim J.17. ∀ x ∈ [0 , 1] : 2 x ≤ 1 + x .

Proof. 2 x is convex, so we get in [0 , 1] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We are reminded of the following:

<!-- formula-not-decoded -->

Define A ( τ ) ≜ d 2 ∫ τ 0 g ( s ) d s , B ( τ ) = 1 ln 4 -4 -τ ( 1 ln 4 -τ ) . We have A (0) = B (0) = 0 , and A ( τ ) -B ( τ ) is non negative and increasing for τ ∈ [0 , 1] :

<!-- formula-not-decoded -->

If we assume ln 4 ≥ 2 · 2 3 /d d ( 1 -2 -1 /d ) , then the derivative is in fact positive and we are done. If we assume the opposite we have:

<!-- formula-not-decoded -->

This means that

<!-- formula-not-decoded -->

In Proposition J.13 we saw that:

<!-- formula-not-decoded -->

Subtracting B (1) we get:

<!-- formula-not-decoded -->

leading to

<!-- formula-not-decoded -->

Now note that,

<!-- formula-not-decoded -->

Since 0 ≤ A ( τ ) -B ( τ ) , we have ˜ x ( τ ) ≥ √ 1 -d 2 ∫ τ 0 g ( s ) d s . Lower bounding the denominator:

<!-- formula-not-decoded -->

and so,

<!-- formula-not-decoded -->

Proposition J.19. For d ≥ 100 , 000 , we have ∀ τ ∈ [0 , ζ ] , | ˜ x ( τ ) -x ( τ ) | ≤ 38 . 9822 d .

Proof. From Proposition J.18 and Proposition J.16:

<!-- formula-not-decoded -->

Corollary J.20. For d ≥ 100 , 000 , we have ∀ τ ∈ [0 , ζ ] , x ( τ ) ≥ 0 . 4567 .

<!-- formula-not-decoded -->

1] :

Note that it is lowest at τ = 1 . Combined with | x ( τ ) -˜ x ( τ ) | ≤ 38 . 9822 d , we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim J.21. For a ≥ b ≥ 0 , √ a -b ≥ √ a -√ b .

Proof.

<!-- formula-not-decoded -->

Proposition J.22 (Existence of the solution to the ODE) . For d ≥ 100 , 000 , we have x 2 ( τ ) ≥ 4 β ( τ + 1 d ) , ∀ τ ∈ [0 , 1] .

Proof. Note the following for τ = ζ :

<!-- formula-not-decoded -->

for d ≥ 100 , 000 , from Claim J.8 and Corollary J.20. From continuity and the strict inequality, there exists δ &gt; 0 such that ∀ τ ∈ [ ζ, ζ + δ ] , x 2 ( τ ) ≥ 4 β ( τ + 1 d ) . Observing the definition of the ODE Eq. (14), we have the following for all τ ∈ [ ζ, ζ + δ ] :

<!-- formula-not-decoded -->

We can show that δ ≥ 0 . 3 √ d , by showing the solution must exist at τ = ζ + 0 . 3 √ d :

<!-- formula-not-decoded -->

Hence, we have that x 2 ( τ ) ≥ 4 β ( τ + 1 d ) for τ ∈ [ 0 , ζ + 0 . 3 √ d ] , and thus we can apply all previous claims replacing ζ with ζ + 0 . 3 √ d , and specifically,

<!-- formula-not-decoded -->

Which allows repeating all the previous steps without alteration. After repeating ⌈ √ d 0 . 3 ⌉ times, we get that x 2 ( τ ) ≥ 4 β ( τ + 1 d ) for τ ∈ [0 , 1] , concluding the proof.

Corollary J.23. All previous propositions, and specifically Proposition J.19 and Corollary J.20, apply ∀ τ ∈ [0 , 1] (indicating ζ = 1 ).

Proposition J.24. For d ≥ 100 , 000 , L ≜ max x,τ ∈ [0 , 1] ∣ ∣ d d x f ( τ, x ) ∣ ∣ ≤ 4 . 7955 .

Proof. L , the Lipschitz constant of f , is given by

<!-- formula-not-decoded -->

We have:

<!-- formula-not-decoded -->

Assume that x ≥ x min , τ ∈ [0 , 1] . from d ≥ 3 , τ + 1 d ≤ 1 ln 2 . From Claim J.7, we get β ( τ + 1 d ) ≥ 0 .This means that d d x f ( τ, x ) ≥ 0 . So

<!-- formula-not-decoded -->

For any fixed x , the maximum β ( τ + 1 d ) will maximize L . From Claim J.7,we know that β is decreasing with τ , so to maximize L , τ = 0 . To maximize d 2 ( x √ x 2 -4 β ( 1 d ) -1 ) , note that this function is decreasing with respect to x :

<!-- formula-not-decoded -->

hence the optimal x is x min . We get that

<!-- formula-not-decoded -->

Now,

<!-- formula-not-decoded -->

and applying Corollary J.20 we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. M is defined as an upper bound on the second derivative (absolute value) of x ( τ ) in the relevant interval:

<!-- formula-not-decoded -->

We have:

<!-- formula-not-decoded -->

For the first term:

<!-- formula-not-decoded -->

From Claim J.7 we know β is positive and decreasing, so d √ x 2 -4 β ( τ + 1 d ) is maximized at τ = 0 ; In addition ∂ ∂τ β ( τ + 1 d ) ≤ 0 , and thus the entire expression is non negative. We know β is convex, so the absolute value of the negative ∂ ∂τ β ( τ + 1 d ) is also maximized at τ = 0 . All in all, the entire expression is maximized at τ = 0 , and is bounded by:

<!-- formula-not-decoded -->

Moving to the second term, we have from Proposition J.24,

<!-- formula-not-decoded -->

Now we need to bound f ( τ, x ) = d √ x 2 -4 β ( τ + 1 d ) -x 2 , which is always negative. From Claim J.7 we know β is positive and decreasing, so its maximum, which minimizes this and thus maximizes its absolute value, is received at τ = 0 . We also know that f (0 , x ) increases with x (see the beginning of the proof for Proposition J.24), so its absolute value decreases with x . Utilizing these facts we get:

<!-- formula-not-decoded -->

To summarize, we have -2 . 1901 ≤ f ( τ, x ) ≤ 0 , and thus

<!-- formula-not-decoded -->

and in absolute value,

<!-- formula-not-decoded -->

Proposition J.26. For d ≥ 100 , 000 , k ∈ { 2 , . . . , d } for which the sequence ( x ) k exists, i.e., x 2 j -1 -4 β ( j d ) ≥ 0 for all 2 ≤ j ≤ k ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. As noted in App. J.1.1, the sequence ( x ) k are Euler's iterates for the ODE in Eq. (14). Using the global truncation error of Euler's method [9, chapter 6.2] we get:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

For d ≥ 100 , 000 we have L ≤ 4 . 7955 from Proposition J.24, and Proposition J.25 gives M ≤ 10 . 5027 .

So in total

<!-- formula-not-decoded -->

Combining this with Proposition J.19, we get

<!-- formula-not-decoded -->

Corollary J.27. For d ≥ 100 , 000 , k ∈ { 2 , . . . , d } for which the sequence ( x ) k exists, i.e., x 2 j -1 -4 β ( j d ) ≥ 0 for all 2 ≤ j ≤ k : x k ≥ 0 . 45 .

Proof. x k is decreasing (Claim I.5), so ∀ k ∈ [ d ] :

<!-- formula-not-decoded -->

Proposition J.28 (Existence of the sequence) . For d ≥ 100 , 000 , the sequence ( x ) k exists for all k ∈ [ d ] , i.e., x 2 j -1 -4 β j ≥ 0 , for all 2 ≤ j ≤ d .

The proof is similar to that of Proposition J.22, but simpler. We first note that

<!-- formula-not-decoded -->

which means x 2 exists. Now assuming for some k ∈ { 2 , . . . , d } , x k exists, then from Corollary J.27, x k ≥ 0 . 45 , and we have

<!-- formula-not-decoded -->

which means that x k +1 exists, and by induction the proof is done.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the setting (detailed in Section 2), contributions, and assumptions made (namely, continual linear regression setting under a joint realizability assumption). All results presented in the paper are outlined in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The exact setting and the main assumption are mentioned in the introduction and fully discussed in Section 2. In Section 6, we discuss, and settle, other findings from the literature; and also extend future directions to address our work's limitations.

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

Justification: Assumption 2.1 is clearly stated in Section 2. All theorems mention their assumptions and refer to their corresponding proofs in the appendices.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Regression experiments, done with synthetic random data, can easily be regenerated using the full details given in the appendices corresponding to each experiment (App. B, F.2 and G.3), and their code provided in App. H. A link to the code for the classification experiments is provided in App. C. The algorithms are simple and always formally stated (Schemes 1, 3 and 4). We detail the construction of the adversarial task collections (Section 5.1) in App. E.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of

closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Regression experiments, done with synthetic random data, can easily be regenerated using the full details given in the appendices corresponding to each experiment (App. B, F.2 and G.3), and the code available in App. H. A link to the code for the classification experiments is provided in App. C.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All details of hyperparameters and experiments with varying hyperparameters are detailed under Figure 3a and in the relevant appendices (App. B, App. F.2, App. G.3). The data is synthetic and we measure only the training performance (as explained in the paper).

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All figures of classification experiments include 95% confidence intervals, see App. C for details. The figures of regression experiments shown in the main body of the paper aim to show qualitative trends on random data, rather than quantitative results on real-world benchmarks. Thus, we defer confidence intervals to the full experiments in the respective appendices, see App. B for details.

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

Justification: Experiments and numerical validations (for Section 5.1) required very little CPU resources that are detailed in the corresponding appendices (App. B, C, E, F.2 and G.3).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research involves theoretical research, synthetic data experiments, and CIFAR experiments. It fully conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper involves theoretical research of continual learning, and has no societal impact.

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

Justification: No models or data were released. The provided research code compares different task orderings in continual learning, and has no risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] .

Justification: All creators of datasets and models used were properly cited. Specifically, we use CIFAR-100 and cite Krizhevsky et al. [55] accordingly in Section 4.1, and we employ pretrained models from Chen [22], cited appropriately in App. C.

## Guidelines:

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

Justification: Released code is documented appropriately.

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

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs were used only for editing purposes and code corrections.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
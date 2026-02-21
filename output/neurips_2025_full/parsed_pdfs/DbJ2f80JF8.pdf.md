## Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data

## Hancheng Min ∗ Zhihui Zhu † René Vidal ‡

∗ Institute of Natural Sciences &amp; School of Mathematical Sciences, Shanghai Jiao Tong University † Computer Science and Engineering, Ohio State University

‡ Electrical and Systems Engineering &amp; Department of Radiology, University of Pennsylvania hanchmin@sjtu.edu.cn, zhu.3400@osu.edu, vidalr@upenn.edu

## Abstract

Among many mysteries behind the success of deep networks lies the exceptional discriminative power of their learned representations as manifested by the intriguing Neural Collapse (NC) phenomenon, where simple feature structures emerge at the last layer of a trained neural network. Prior works on the theoretical understandings of NC have focused on analyzing the optimization landscape of matrix-factorization-like problems by considering the last-layer features as unconstrained free optimization variables and showing that their global minima exhibit NC. In this paper, we show that gradient flow on a two-layer ReLU network for classifying orthogonally separable data provably exhibits NC, thereby advancing prior results in two ways: First, we relax the assumption of unconstrained features, showing the effect of data structure and nonlinear activations on NC characterizations. Second, we reveal the role of the implicit bias of the training dynamics in facilitating the emergence of NC.

## 1 Introduction

Among many mysteries behind the success of deep learning lies the exceptional discriminative power of neural networks as manifested by the intriguing Neural Collapse (NC) phenomenon [1], where simple feature structures emerge in the last layer of a trained network. The NC phenomenon is typically characterized by the following three properties (see the top plot in Figure 1):

1. Intra-class variability collapse : The last-layer feature vectors of the data from the same class collapse into a singleton;
2. Maximal separation of the class means : The class means , i.e., the mean feature vectors for each class, are maximally separated;
3. Self-duality : The classifier weights align with the class means.

Prior works on the theoretical understandings of NC have focused on analyzing the optimization landscape of matrix factorization-like problems by considering the last-layer features as unconstrained free optimization variables [2-9], showing their global minima exhibit NC. Extensions of this so-called Unconstrained Feature Model (UFM) include adding nonlinearity and additional hidden layers [7, 8, 10, 11], studying local convergence of gradient-based optimization algorithms around global

∗ Corrrespondance to Hancheng Min

Figure 1: Visualization of the NC phenomenon.

<!-- image -->

minima [2, 12, 13]. Until recently, Jacot et al. [14] show convergence towards NC in wide networks. Therefore, the question of how training a neural network leads to NC has remained underexplored.

At the center of this problem lies the fact that practical neural networks are typically overparameterized, i.e., their number of parameters is several orders of magnitude larger than the number of available training examples. As a consequence, there are infinitely many parameter choices that can perfectly fit the data, and most critically, they do not necessarily correspond to a trained network that exhibits NC. However, NC is observed even for networks trained without explicit regularization [3], such as weight decay (often associated with the emergence of NC). This suggests a close relationship between the NC phenomenon and the implicit bias [15] of training algorithms.

Paper contributions . In this paper, we investigate this relationship between NC and implicit bias by analyzing the dynamics of gradient flow (GF) on a two-layer ReLU network for classification problems, focusing on orthogonally separable data; i.e., any pair of input data with the same or different labels is positively or negatively correlated, respectively. We make the following contributions:

1. In Section 3 we present Theorem 1, which shows that GF with small initialization provably converges to solutions that exhibit NC and provides precise NC characterizations for the trained network, as illustrated in the bottom plot of Figure 1. Compared to prior works for the unconstrained feature model, our results highlight the role of input data and ReLU nonlinearity in determining the NC characteristics. The former causes intra-class directional collapse instead of collapse to a singleton, i.e., the feature vectors of the data from the same class collapse into a one-dimensional subspace. The latter leads to orthogonal class means , instead of maximally separated class means, and a projected self-duality where the classifier weights align with the projected class means (the projection is obtained by subtracting the global mean of all features).
2. In Section 4.1, through our proof of Theorem 1 for the case of binary classification, we explain how the implicit bias of GF facilitates the emergence of NC as training proceeds. Using results on the implicit bias of GF [16-21], we show that in the early phase of training the neurons' directional alignment [16-20] with the training data makes inter-class features mutually orthogonal. Then, during the late phase of training, such inter-class separation subsequently promotes intra-class directional collapse due to the asymptotic max-margin bias [21] of GF.
3. In Section 4.2, in our proof sketch of Theorem 1 for the case of multi-class classification, we extend the aforementioned results on implicit bias for binary problems to multi-class problems. We make technical contributions in addressing new challenges that arise in the dynamic analysis due to the multi-dimensional network output and the cross-entropy loss.

In summary, our work bridges the theoretical analysis of NC and implicit bias of GF by drawing explicit connections between the two. Moreover, we further advance the theoretical understandings for both by highlighting the role of input data and nonlinear activations in NC characterization and by addressing the challenges in analyzing the implicit bias of GF in multi-class problems.

Notations . We denote the Euclidean norm of a vector x by ∥ x ∥ and its i -th entry by [ x ] i , denote the inner product between vectors x and y by ⟨ x , y ⟩ = x ⊤ y and write x ≥ 0 or x &gt; 0 if all the entries of x are non-negative or positive, respectively. For an n × m matrix A , we let ∥ A ∥ F denote the Frobenius norm of A . For a scalar-valued or matrix-valued function of time, F ( t ) , we let d dt F ( t ) denote its time derivative. We define 1 to be the vector of all ones, whose dimension will be clear from the context. We let I n be the identity matrix of order n and sometimes drop the subscript if its order is clear from the context. We let [ N ] := { 1 , · · · , N } and let S D -1 be the unit-sphere in R D .

## 2 Preliminaries

Orthogonally separable data . We consider a classification problem on a dataset { x i , y i } n i =1 of size n , where each data point x i ∈ R D is associated with its label y i ∈ R d y , and the number of unique elements in { y i } n i =1 determines the number of classes K . Throughout this paper, we assume:

Assumption 1 (Orthogonal separability) . Any pair of data with the same (different) label(s) are positively (negatively) correlated, i.e., ∃ 0 &lt;µ s ≤ 1 and 0 &lt;µ d ≤ 1 √ K -1 2 such that ∀ 1 ≤ i, j ≤ n ,

̸

<!-- formula-not-decoded -->

2 No dataset can satisfy orthogonal separability with µ d &gt; 1 √ K -1 .

Two-layer ReLU network . We are interested in solving this classification problem by training a widthh two-layer ReLU network f ( · ; θ ) : R D → R d y , parametrized by θ := { W , V } ∈ R D × h × R d y × h with W = [ w 1 , · · · , w h ] and V = [ v 1 , · · · , v h ] , and defined as

<!-- formula-not-decoded -->

where σ ( · ) = max { · , 0 } is the element-wise ReLU activation function. We consider networks with width h ≥ K ; we call ( w j , v j ) the j -th neuron pair in the network, w j its input neuron weight and v j its output neuron weight . Moreover, we let ϕ θ ( x ) = [ σ ( ⟨ w 1 , x ⟩ ) , σ ( ⟨ w 2 , x ⟩ ) , · · · , σ ( ⟨ w h , x ⟩ )] ⊤ ∈ R h be the last-layer feature of x , and V = [ v 1 , v 2 , · · · , v h ] ∈ R d y × h denote the last-layer classifier . Note that we have considered bias-free ReLU networks; see the remark in the Appendix A for extending the results to networks with biases.

Gradient flow with small initialization . Given some ℓ : R d y × R d y → R ≥ 0 such that ℓ ( y i , ˆ y i ) (expressions shown in later sections) measures the discrepancy between the actual label y i and a predicted label ˆ y i , we let L ( θ ) = ∑ n i =1 ℓ ( y i , f ( x i ; θ )) be the loss function . We train the network via gradient flow (GF), which can be viewed as gradient descent with infinitesimal step size:

<!-- formula-not-decoded -->

where ∂ θ denotes the Clarke sub-differential [22] operator. We study solutions (or trajectories ) θ ( t ) , t ≥ 0 that satisfies (3) almost everywhere. We assume that the initialization θ (0) satisfies

̸

Assumption 2 ( ϵ -small and balanced initialization) . The initialization θ (0) = { w j (0) , v j (0) } h j =1 satisfies the following: there exists an initialization shape { w j 0 , v j 0 } h j =1 with w j 0 , v j 0 = 0 , ∀ j and an initialization scale ϵ &gt; 0 such that ∀ j , w j (0) = ϵ w j 0 , v j (0) = ϵ v j 0 , ∥ w j 0 ∥ = ∥ v j 0 ∥ .

Aside from the two assumptions, we will introduce additional assumptions in different training scenarios when their respective settings become clear. For now, we shall remark on these two.

Remark 1. While the data assumption is strong, there are two main reasons for considering it: First, we investigate NC in shallow networks, whose single hidden layer has limited expressive power of collapsing features, thus one shall study more structured data, as also noted by Hong and Ling [23]. Second, as it will become clearer in Section 4, the emergence of NC is closely related to the asymptotic convergence of the network weights, whose precise characterization is limited to cases with structurally simple data [18, 20, 24]. Nonetheless, as we show in Section 5, simple real data satisfies orthogonal separability approximately, leading to NC characters that match our theorem.

Remark 2. Under the assumption of balanced initialization, we have ∥ w j (0) ∥ = ∥ v j (0) ∥ , ∀ j , and this balance is maintained throughout the GF trajectories, i.e., ∥ w j ( t ) ∥ = ∥ v j ( t ) ∥ , ∀ t, ∀ j [25]. This assumption of balanced initialization has been common in prior works of this type [18, 20, 24] for the sake of tractable analysis. Our experiments in Section 5 do not require balanced initialization.

## 3 Main result: Neural Collapse under GF on Two-layer ReLU Networks

Our main result shows that under small initialization, with some additional assumptions on the initialization shape, GF provably converges to neural collapse solutions on orthogonally separable data. Our results are presented for both binary classification and multi-class classification problems:

- Case one: Binary classification: We consider binary ( K = 2) data with scalar ± 1 labels, i.e. d y = 1 and y i ∈ {-1 , +1 } , ∀ i . Accordingly, the two-layer ReLU network f ( x ; θ ) has a scalar output ˆ y . The loss function can be either the exponential loss ℓ ( y, ˆ y ) = exp( -y ˆ y ) or the logistic loss ℓ ( y, ˆ y ) = 2 log(1 + exp( -y ˆ y )) . For this case, we use plain font to suggest that label y and network output ˆ y = f ( x ; θ ) are scalars. Moreover, we define the index sets I + := { i : y i = +1 } and I -:= { i : y i = -1 } for ± 1 -class data respectively.
- Case two: Multi-class classification: We consider multi-class ( K &gt; 2) data with one-hot labels, i.e., d y = K , and y i ∈ { e 1 , · · · , e K } , where e k is the k -th column of the identity matrix I K . Accordingly, the two-layer ReLU network has its output ˆ y = f ( x ; θ ) ∈ R K . The loss function is the Cross-Entropy (CE) loss ℓ ( y , ˆ y ) = -∑ K k =1 [ y ] k log exp([ˆ y ] k )[ y ] k ∑ l K =1 exp([ˆ y ] l ) . Moreover, we define the index sets I k := { i : y i = e k } , ∀ k ∈ [ K ] for data from each class.

Main result . Our main theorem follows. Note that our theorem requires additional assumptions on the data and initialization shape that vary depending on the case , thus we feel it is better to introduce and explain them alongside technical discussions on the convergence analysis in later sections.

Theorem 1 (NC of GF on Two-layer ReLU Networks) . Given orthogonally separable data (Assumption 1), ϵ -small and balanced initialization (Assumption 2) for a sufficiently small ϵ , and some additional case-dependent assumptions on the data and initialization shape, the limit ¯ θ := lim t →∞ θ ( t ) ∥ θ ( t ) ∥ F exists for any solution θ ( t ) , t ≥ 0 to (3) . Moreover, for the limit ¯ θ = { ¯ W , ¯ V } , we have: ∃ ¯ ϕ k ∈ S h -1 , k ∈ K , where K := { + , -} ( case one ) or K := [ K ] ( case two ), such that

1. ( Intra-class directional collapse ) The last-layer features of the training data satisfy that

<!-- formula-not-decoded -->

where γ k = max u ∈ S D -1 min i ∈I k ⟨ x i , u ⟩ , k ∈ K is the maximum margin achieved exclusively for classk data and u k is the corresponding max-margin direction;

2. ( Orthogonal class means ) The class means directions ¯ ϕ k , k ∈ K satisfy that

̸

<!-- formula-not-decoded -->

3. ( Projected self-duality ) The last-layer classifier satisfies that

<!-- formula-not-decoded -->

Note that GF on positively homogeneous networks with classification losses drives the network weights to diverge to infinity [26, 27]. It suffices to study the properties of the asymptotic weight direction ¯ θ since we have f ( · ; θ ) = ∥ θ ∥ 2 F f ( · ; θ ∥ θ ∥ F ) due to the positive homogeneity of f w.r.t. θ . The following remarks compare the NC characterizations in Theorem 1 with those in prior works.

NC in two-layer ReLU . Theorem 1 shows the following NC characters at the late stage of training:

- ( Intra-class directional collapse ) Unlike previous works [2-8], which study unconstrained feature models and show that features collapse to class means with equal length, our work addresses a more realistic and challenging setting involving input data. In this setting, the limited expressiveness of two-layer ReLU networks may prevent exact collapse to a singleton. Nevertheless, the result in (4) shows a direction collapse in the sense that all data points in the k -th class I k have their last-layer features ϕ ¯ θ ( x i ) collapse into a one-dimensional subspace spanned by ¯ ϕ k , though the features may have varying lengths. Consequently, the intra-class variability at the last layer is determined by the variability of projections {⟨ s k u k , x i ⟩} i , a significant reduction compared to the variability of the original data { x i } i . Moreover, if the features are normalized to unit norm (e.g., by applying RMSnorm [28]), they collapse exactly to their corresponding class means, shedding light on the role of the normalization layer in the neural collapse phenomenon.
- ( Orthogonal class means ) The result (5) suggests that the class-mean features are orthogonal to each other, forming a non-negative orthogonal frame when normalized. The orthogonal structure, rather than a simplex Equiangular Tight Frame (simplex ETF) 3 , arises because the features are always non-negative due to ReLU-but any orthogonal frame can be transformed into a simplex ETF by removing its global mean. This finding aligns with results from the unconstrained features model using ReLU as the activation [7].
- ( Projected self-duality ) In the case of binary classification, the classifier ¯ V converges to s + ¯ ϕ + -s -¯ ϕ -, which yields a maximum margin, as we will show in Section 4.1. For the case of multiclass classification, (6) implies that ¯ V ¯ V ⊤ = K K -1 ( I -1 K 11 ⊤ ) ¯ Φ ¯ Φ ⊤ ( I -1 K 11 ⊤ ) , where ¯ Φ = [ s 1 ¯ ϕ 1 , · · · , s K ¯ ϕ K ] . Since ¯ Φ ¯ Φ ⊤ is a diagonal matrix with positive diagonals, ¯ V forms a (scaled) simplex ETF, thereby achieving maximum margin. In particular, when the diagonal scales s k , k ∈ K are all equal, ¯ V becomes an exact simplex ETF, and each classifier converges to the corresponding projected class mean (projection is obtained by subtracting the global mean), up to a scaling factor-achieving self-duality between features and classifiers weights[1].

3 A K -simplex ETF in R h is a collection of points specified by the columns of ˜ E = √ K K -1 P ( I -1 k 11 ⊤ ) , where P ∈ R h × K and P ⊤ P = I .

Convergence of GF/GD to NC . Prior works on the convergence of gradient-based methods towards NC consider the mean squared loss [2, 12-14]; Besides, additional conditions such as initialization close to a global optimum [13], weight decay regularization [2, 13, 14], or large width [14] are needed. Compared with these works, we study the convergence under the cross-entropy loss without explicit regularization or width-overparametrization, showing that NC happens under a broader class of problems. Moreover, our results highlight the role of implicit bias of the training algorithm, which we shall discuss next.

## 4 Detailed Discussions: Connecting Neural Collapse with Implicit Bias of GF

## 4.1 Proof of Neural Collapse in Binary Classification

In this section, we provide a proof of Theorem 1 for binary classification of orthogonally separable data. Recall that I + and I -denote the index sets for data with positive and negative labels, respectively. Let N + ( t ) := { j ∈ [ h ] : sign( v j ( t )) = +1 } and N -( t ) := { j ∈ [ h ] : sign( v j ( t )) = -1 } denote the index set of neurons whose last-layer weights v j ( t ) at time t has positive and negative signs, respectively. Under Assumption 2, we have that sign( v j ( t )) = sign( v j (0)) [18, 20], thus N + ( t ) ≡ N + (0) , N -( t ) ≡ N -(0) , ∀ t , and we conveniently let N + := N + (0) and N -:= N -(0) .

Alignment phase of the GF . During the early phase of training, often referred to as alignment phase , several works [16-18, 20, 24, 29, 11, 30] have shown that the norm of the neuron weights, which is initially of scale O ( ϵ ) , remains small (of scale O ( ϵ 1 / 2 ) ) for an extended period of time of length Θ(log 1 ϵ ) . As a result, one focuses on understanding the directional dynamics of the input neuron weights during this phase, which can be approximated as follows ∀ j ∈ [ h ] :

<!-- formula-not-decoded -->

where Π w ⊥ j = ( I -w j w ⊤ j / ∥ w j ∥ 2 ) defines the projection onto the subspace orthogonal to w j . If one neglects the O ( ϵ ) term, the dynamics d d t w j ∥ w j ∥ , j ∈ [ h ] are decoupled. The dynamic behavior of w j ∥ w j ∥ critically depends on the stationary points of (7), which we shall address next.

̸

The following discussions assume the O ( ϵ ) error term is zero (only for the sake of the explanation here, the error terms are appropriately handled in the analyses). First of all, the directions w j ∥ w j ∥ that render ξ ij = 0 , ∀ i ∈ [ n ] are trivial stationary points of (7), and they form the 'dead region' for the neuron as all the activations to the data are zero (as its name suggest, neuron weights within dead region have zero gradient thus receive no update along GF). Next, the stationary points with some ξ ij = 0 are often called extremal vectors [16, 17, 29], and the analyses in the work of Phuong and Lampert [17], Min et al. [20] have suggested that for binary orthogonally separable data the only extremal vectors are the class mean directions: ¯ x + and ¯ x -. More importantly, for neurons with j ∈ N + , ¯ x + is an attractor, and ¯ x -is a repeller 4 (the opposite for j ∈ N -). Therefore, by following (7), the neurons weights with j ∈ N + either fall into the dead region, or converge in direction to the average direction ¯ x + of the positive class; while those with j ∈ N -either fall into the dead region or converge to ¯ x -.

Transient analysis: inter-class separation via alignment dynamics of neurons . Based on the discussions above, the convergence analysis requires a non-degenerate initialization shape { w j 0 } h j =1 such that 1) no input neuron weight is initialized to align with the repeller for (7), since moving away from the repeller can take a long time that cannot be quantified; and 2) there must exist at least one neuron weight per class that is guaranteed to converge to the average direction of that class , avoiding the uninteresting case of all neuron weights entering the dead region.

̸

Assumption 3 (Non-degenerate initialization) . Let ¯ x + = x + ∥ x + ∥ and ¯ x -= x -∥ x -∥ , where x + := ∑ i ∈I + x i and x -:= ∑ i ∈I -x i . The initialization shape { w j 0 } h j =1 satisfies that N + , N -= ∅ and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

4 Roughly speaking, an attractor or a repeller is a stationary point that has the flow around its neighborhood pointing towards or against it, respectively.

Figure 2: Convergence and implicit bias of GF in two-layer ReLU networks: (a) An example of orthogonally separable data (Assumption 1), gray region indicates the 'dead region" for neuron weights: { w : ⟨ w , x i ⟩ ≤ 0 , ∀ i ∈ [ n ] } ; (b) Weight initialization following Assumption 2, input neuron weights have small norm and random directions; (c) During alignment phase, inter-class separation is achieved through directional alignment between input neuron weights and data points, as described in (10); (d) Asymptotically, neuron weights diverge to infinity while their directions align with the class-wise max-margin directions, as described in (12).

<!-- image -->

Given a non-degenerate initialization 5 , whenever the neuron weights converge to the vicinity of their respective attracting class averages, they stay close to the class averages for the rest of the training, leading to the following inter-class separation, due to the orthogonally separability of the data:

Claim (Inter-class separation via alignment, based the analyses from Phuong and Lampert [17], Min et al. [20]) . Given orthogonally separable data (Assumption 1), ϵ -small, balanced and non-degenerate initialization (Assumptions 2, 3) for a sufficiently small ϵ , for any solution θ ( t ) , t ≥ 0 to (3) , ∃ T ∗ and ∅ ̸ = ˜ N + ⊆ N + and ∅ ̸ = ˜ N -⊆ N -such that ∀ t ≥ T ∗ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a result, the inter-class separation 〈 ϕ θ ( t ) ( x i ) , ϕ θ ( t ) ( x i ′ ) 〉 =0 , ∀ i ∈I + , i ′ ∈I -holds ∀ t ≥ T ∗ .

We refer to Figure 2c for an illustration of the weight-data alignment that achieves inter-class separation. As shown in this claim, all neuron weights with index j outside ˜ N + ∪ ˜ N -will stay within the dead region after T ∗ and can be disregarded in the subsequent analysis. Therefore, without loss of generality, we assume until the end of Section 4.1 that ˜ N + ∪ ˜ N -= [ h ] and reorder the indices such that ˜ N + = [ | ˜ N + | ] and ˜ N -= [ h ] -[ | ˜ N + | ] . To see that (10) indeed suggests inter-class separation characterized by last-layer features being mutually orthogonal, notice that ∀ t ≥ T ∗ , we have

<!-- formula-not-decoded -->

Asymptotic analysis: intra-class directional collapse and self-duality via max-margin bias . Now with the inter-class separation described in (10), we are ready to study the asymptotic convergence of the weights through the lens of max-margin bias. In particular, notice that given inter-class separation (10), we rewrite the loss as (where we define V + := [ v j ( t )] j ∈N + and V -:= [ v j ( t )] j ∈N -):

<!-- formula-not-decoded -->

This suggests that after T ∗ , the GF on { W + , V + } is fully decoupled from that on { W -, V -} , allowing one to study them separately. Moreover, each of the flows corresponds to training a twolayer linear network on positively correlated data with the same label, whose asymptotic convergence of the weight directions has been characterized in the work of Phuong and Lampert [17], mainly based on the analysis from Ji and Telgarsky [21] on the max-margin bias of GF in linear networks.

To be precise, for the GF on losses of the form ∑ i ∈I + ℓ ( y i , V W ⊤ x i } , Ji and Telgarsky [21] show that as time t → ∞ , both V and W ⊤ diverge to infinity and their limiting directions exist. In

5 Min et al. [20] has shown that the non-degeneracy is satisfied with high probability when the input weight shapes are randomly initialized.

the case of (11), this means θ = { W + , V + , W -, V -} diverge to infinity, and lim t →∞ θ ( t ) ∥ θ ( t ) ∥ := ¯ θ = { ¯ W + , ¯ V + , ¯ W -, ¯ V -} . Moreover, they show that the limiting directions satisfies the following alignment condition ¯ V + ¯ W ⊤ + ∝ u + , the class-wise max-margin direction we have defined in Theorem 1 and balancedness condition ¯ V ⊤ + ¯ V + = ¯ W ⊤ + ¯ W + (similar for ¯ W -, ¯ V -, with u + replaced by u -). It was first found in the work of Phuong and Lampert [17] that the only time these two conditions are satisfied is when ¯ W ⊤ + has rank 1 and the top left and right singular vectors align with ¯ V + and u + , respectively. We show that this necessarily implies NC. The formal results are:

Claim (Directional collapse and self-duality via max-margin bias, based on the analyses from Phuong and Lampert [17], Ji and Telgarsky [21]) . Given Assumptions 1,2,&amp;3, the limit ¯ θ := lim t →∞ θ ( t ) ∥ θ ( t ) ∥ F exists for any solution θ ( t ) , t ≥ 0 to (3) . For the limiting direction ¯ θ = { ¯ W + , ¯ V + , ¯ W -, ¯ V -} , ∃ g + ∈ S |N + |-1 , g -∈ S |N -|-1 , such that

<!-- formula-not-decoded -->

where s + and s -are defined in Theorem 1. As a result, we have the intra-class directional collapse

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and the self-duality between the last-layer classifier weight and the last-layer feature

<!-- formula-not-decoded -->

Note that the scaling factors s + , s -are determined based on the results in the work of Lyu and Li [27] that the limiting ¯ θ must satisfy another max-margin problem defined on the entire dataset.

Connecting NC with implicit bias of GF . In summary, we bridge the theoretical analysis of NC and the implicit bias of GF closer by showing how the latter facilitates the emergence of NC along GF. Notably, the inter-class separation is achieved by the directional alignment of neuron weights thanks to the small initialization scale. This resonates with a large amount of work [31-41] that identifies the small initialization as the active learning or feature learning regime, allowing simple weight and hidden feature structures to arise during the early phase of GF, which otherwise cannot be achieved if initialized in the so-called lazy regime [42-44]. Then, we have shown how the asymptotic max-margin bias promotes the intra-class directional collapse and self-duality after inter-class separation, where the key observation is that the max-margin bias often makes the weights asymptotically converge in direction to low-rank matrices, leading to low-dimensional projections that significantly reduce the variability within the input data. We note that prior work [3] studies the max-margin bias in UFM, while ours considers such bias in ReLU networks.

## 4.2 Proof Sketch of Neural Collapse in Multi-class Classification

As shown in the last section, the proof of the NC characterization for binary classification in Theorem 1 follows from existing results on the implicit bias of GF [17, 20, 21]. However, to understand similar NC characterizations in the case of multi-class classification, one needs to extend the implicit bias analyses to multi-class problems, which prior work rarely does. In this section, we provide a proof sketch of Theorem 1 for multi-class classification, emphasizing the additional challenges it brings to the convergence analysis by considering a multi-dimensional network output and the cross-entropy loss, and discussing our contributions in addressing these challenges.

Weight alignment in multi-class problems . Recall that in binary problems, the directional dynamics are studied only for the input weights w j , j ∈ [ h ] , because the output weights v j are scalars whose sign (i.e. the 'direction' of the scalar) remains the same as its initialization. However, for multi-class problems, each v j becomes a K -dimensional vector, whose directional dynamics are non-trivial. Indeed, we show that (See Appendix C.2) during the early phase of GF, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Π w ⊥ j , Π ⊥ v j and ξ ij are defined similarly as in (7), and ˜ E = √ K K -1 ( I -1 K 11 ⊤ ) . We let ˜ e k be the k -th column of ˜ E and call it the pseudo-label of class k , as opposed to the one-hot label e k .

From (16)(17) (still, we exclude the O ( ϵ ) error terms for discussions), we see that although the dynamics { d d t w j ∥ w j ∥ , d d t v j ∥ v j ∥ } , j ∈ [ h ] are decoupled among neuron pairs, the directional dynamics of each neuron pair, now concerning both input and output weights, are described by a Riemannian flow on S D -1 × S K -1 that is highly nonlinear (due to the ξ ij ) and has non-trivial interactions between the input and output weights. This is the major challenge to the convergence analysis of GF under small initialization for multi-class problems. For our purpose, we discuss the alignment of the neuron weights through the lens of stationary points.

Aside from trivial stationary points that correspond to the dead regions for the input neuron weight, { ¯ x k , ˜ e k } , k ∈ [ K ] , where ¯ x k = ∑ i ∈I k x i / ∥ ∑ i ∈I k x i ∥ , are attractors of (16)(17). To give a rough explanation, assume v j ∥ v j ∥ = ˜ e k , i.e. perfect alignment between output weights and the pseudo-label always holds, then one can write (16) as

̸

<!-- formula-not-decoded -->

resulting on a flow that pushes w j ∥ w j ∥ towards (against) the directions of data points in the k -th class (other classes), and eventually towards ¯ x k when sufficient alignment with the k -th class has led to ξ ij = 1 i ∈I k . Similarly, by assuming w j ∥ w j ∥ = ¯ x k , we can write (17) as d d t v j ∥ v j ∥ =

√ K -1 K Π ⊥ v j ( ∑ i ∈I k ⟨ x i , ¯ x k ⟩ ˜ e k ) , pushing the output weight toward the pseudo-label ˜ e k .

Transient analysis: inter-class separation via alignment dynamics of neurons . Based on the discussion above, there is the region of attraction (ROA) for each attractor { ¯ x k , ˜ e k } such that all the neuron weights initialized within the ROA are guaranteed to converge to { ¯ x k , ˜ e k } via (16)(17). Moreover, the boundaries between two ROAs of different (class average)-(pseudo-label) pairs together form an invariant set that does not converge to any of the attractors { ¯ x k , ˜ e k } , where there exist saddle points of (16)(17). The exact generalization of Assumption 3 to the multi-class case should assume that no weight direction falls on the aforementioned boundaries and each ROA contains at least one neuron pair. However, finding an analytic expression for the boundaries is a challenging problem by itself and far beyond the scope of this paper. Instead, our analysis identifies an invariant subset for each ROA, thus by initializing within those invariant subsets, we guarantee the directional convergence of the neuron weights to { ¯ x k , ˜ e k } , k ∈ [ K ] , which implies inter-class separation.

Formally, we let I w k = { i ∈ I k : 〈 x i , w ∥ w ∥ 〉 &gt; 0 } denote the index set for classk data points that activates the input neuron weight w , let A w k = ∑ i ∈I w k 〈 x i , w ∥ w ∥ 〉 be the aggregate alignment with the k -th class of the input neuron weight w , and let B v k = 〈 ˜ e k , v ∥ v ∥ 〉 be the alignment with the k -th pesudo-label of the output neuron v . Then we define the following:

Assumption 4 (Semi-local initialization) . The initialization shape { w j 0 , v j 0 } h j =1 satisfies that ∃ a partition {N k , k ∈ [ K ] } of [ h ] , such that ∀ k ∈ [ K ] , we have

̸

<!-- formula-not-decoded -->

̸

Given some additional assumption on the orthogonal separability of the data (see below), the condition in (18) defines an invariant subset of the ROA of the attractor { ¯ x k , ˜ e k } : any neuron weights initialized to satisfy (18) remains to do so during the alignment phase of GF, while getting attracted by { ¯ x k , ˜ e k } , leading to the desired inter-class separation:

Proposition 1 (Inter-class separation in multi-class problems) . Let K &gt; 2 . Given orthogonally separable data (Assumption 1) with X 2 max X 2 min µ d µ 2 s &lt; 2 K -3 , where X max := max i ∈ [ n ] ∥ x i ∥ and X min := min i ∈ [ n ] ∥ x i ∥ , ϵ -small, balanced and semi-local initialization (Assumptions 2, 4) for a sufficiently small ϵ , for any solution θ ( t ) , t ≥ 0 to (3) , ∃ T ∗ such that ∀ t ≥ T ∗ , we have

̸

W k ( t ) ⊤ x i { &gt; 0 , ∀ i ∈ I k ≤ 0 , ∀ i / ∈ I k , ∀ i ∈ [ n ] , k ∈ [ K ] , where W k ( t ) := [ w j ( t )] j ∈N k . As a result, the inter-class separation 〈 ϕ θ ( t ) ( x i ) , ϕ θ ( t ) ( x i ′ ) 〉 =0 , ∀ i ∈ I k , i ′ ∈ I k ′ , k = k ′ holds ∀ t ≥ T ∗ .

Asymptotic analysis: intra-class directional collapse and projected self-duality via max-margin bias . Once the inter-class separation is achieved, we can again decompose the loss function as L ( θ ) = ∑ K k =1 ∑ i ∈I k ℓ CE ( y i , V k W ⊤ k x i ) , where V k = [ v j ] j ∈N k , thus it suffices to study the GF on { W k , V k } for training a two-layer linear network on exclusively on classk data for each k ∈ [ K ] with cross-entropy loss. Another major technical contribution of our analysis is to extend the maxmargin results in the work of Phuong and Lampert [17], Ji and Telgarsky [21] to the multi-class problems (albeit under a special case that all data have the same label and are positively correlated), leading to the following asymptotic characterization of GF:

Proposition 2 (Variability collapse and self-duality in multi-class problems) . Let K &gt; 2 . Given Assumptions 1,2,&amp;4, the limit ¯ θ := lim t →∞ θ ( t ) ∥ θ ( t ) ∥ F exists for any solution θ ( t ) , t ≥ 0 to (3) . For the limiting direction ¯ θ = { ¯ W k , ¯ V k } K k =1 , ∃ g k ∈ S |N k |-1 , such that

<!-- formula-not-decoded -->

where s k , k ∈ [ K ] are defined in Theorem 1. As a result, the intra-class directional collapse (4) and projected self-duality (6) hold.

Limitations of current analysis . Aside from the orthogonal separability of the data, for which we have made remarks in Section 2, the convergence analysis for multi-class problems requires a stricter separability condition ( X 2 max X 2 min µ d µ 2 s &lt; 2 K -3 ), as shown in Proposition 1. This assumption is required to show that the subsets of the parameter space defined in Assumption 4 are invariant under GF. We believe such an assumption is not needed in practice, but our limited understanding of the ROAs and their invariant subsets has led to this additional technical condition to ensure directional convergence. Future research on better characterizations of the ROAs and their invariant sets will naturally relax or even potentially remove this requirement. The additional assumption (Assumption 4) on the initialization shape for multi-class problems is another limitation of our analysis for the early phase of GF, requiring all neuron weights to have decent alignment with one of the (class-average)-(pseudolabel) pair. Relaxing such an assumption necessitates a careful in-depth analysis of the neuron weight alignment dynamics shown in (16)(17), for which we have discussed the underlying challenges, and we leave it as an important future research direction. Nonetheless, we would like point out that: First, our result on asymptotic convergence of the weights is applicable whenever one can show that inter-class separation happens at sometime during the GF, and our transient analysis simply provides one condition under which the separation is guaranteed to happen; Moreover, the semi-local initialization can be satisfied if, instead of random initialization, one initializes all the neuron pair shapes { w j 0 , v j 0 } h j =1 by drawing uniformly from the (data)-(pesudo-label) pairs { x i , ˜ Ey i } n i =1 , which is a practically possible initialization scheme.

## 5 Numerical Experiments

We conduct experiments primarily for the purpose of validating our theoretical results. We first train a two-layer ReLU network for classifying three MNIST [45] digits and visualize the neuron weights alignment at the end of the training, thereby showing the NC characterizations in Theorem 1. Next, based on our remarks on intra-class directional collapse, our Theorem 1 suggests that proper normalization layers such as RMSNorm [28] can potentially lead to a more significant level of NC, and we conduct some preliminary experiments with ResNet [46] to verify this conjecture.

Validating Theorem 1 in MNIST digits classification . We train a two-layer ReLU network to classify three MNIST digits { 0 , 1 , 2 } . The experimental details are in Appendix B.1. Figure 3 visualizes the training results: First, we show that the dataset of MNIST digits, centered by the mean digit of the entire dataset, approximately satisfies the orthogonally separable assumption. Then, we visualized the neuron pairs, showing their respective alignment with the data and the pseudo-labels. Moreover, we visualize the top 3 principal components of the last-layer feature of the digits, together with the classifiers, whose structure matches the NC characterizations in Theorem 1.

Experiments on the role of normalization layers on NC . Next, we train a modified ResNet18 (by replacing the final linear classifier by a two-layer ReLU classifier) on MNIST and CIFAR10 [47] datasets. In addition, we add a normalization layer (Identity/None, LayerNorm [48], or RMSNorm [28]) before the ReLU classifier and vary the methods for normalization. The experimental details are in Appendix B.2. Figure 4 reports (repeated for 5 runs; mean(line) and std(shade) are

Figure 3: Validating Theorem 1 in classifying MNIST digits { 0 , 1 , 2 } . (a) Normalized correlation matrix of subsampled 500 MNIST digits; (b) (For the trained network) visualization of output neuron weights (as crosses, gray dashed line represents ˜ e k directions for references), the average input neuron weights (as grayscale image, surrounded by colored box), and the average of the digits (for comparison, next to the neuron weights); (c) PCA of raw digits data X , keeping the top 3 principal components (1000 points visualized) results in a ∼ 61% relative approximation error for X ; (d) (For the trained network) PCA of last-layer feature ϕ θ ( X ) and classifiers (rows of V ), keeping the top 3 components (1000 points visualized) results in a ∼ 0 . 2% relative approximation error for ϕ θ ( X ) .

<!-- image -->

Figure 4: Measuring NC in trained modified ResNet18 on MNIST (top) and CIFAR10 (bottom)

<!-- image -->

reported) the evolutions over training 50 epochs of the metrics NC1, NC2 and NC3 that measure the NC characteristics in Theorem 1 (lower value implies more prominent NC; definitions in Appendix B.2) at the last-layer of the ReLU classifier. Notably, using the RMSNorm layer significantly improves the intra-class directional collapse, as we conjectured in the remark for Theorem 1, suggesting potential practical value in using RMSNorm layers for promoting NC.

## 6 Conclusion

In this paper, we investigated the connection between NC and the implicit bias of GF through a convergence analysis of GF on two-layer ReLU networks for orthogonally separable data and showed that the implicit bias of GF facilitates the emergence of NC along the GF trajectory. Future work includes relaxing the assumptions on the data and initialization, and extending the convergence analysis to understand the emergence of NC in deeper networks; For example, similar early weight directional alignment and asymptotic max-margin bias have been studied in prior works [30, 26, 27], following the same high-level proof and utilizing these existing results on alignment and max-margin for deep networks might extend the current work to a more practical setting.

## Acknowledgments and Disclosure of Funding

This work is primarily done when H. Min was a postdoc at the University of Pennsylvania. The authors acknowledge the support of the NSF under grants 2031985 and IIS-2312840, the Simons Foundation under grant 814201, the ONR MURI Program under grant 503405-78051, and the University of Pennsylvania Startup Funds.

## References

- [1] Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. PNAS , 117(40):24652-24663, 2020.

- [2] Dustin G Mixon, Hans Parshall, and Jianzong Pi. Neural collapse with unconstrained features. Sampling Theory, Signal Processing, and Data Analysis , 20(2):11, 2022.
- [3] Wenlong Ji, Yiping Lu, Yiliang Zhang, Zhun Deng, and Weijie J Su. An unconstrained layer-peeled perspective on neural collapse. In ICLR , 2022.
- [4] Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A geometric analysis of neural collapse with unconstrained features. In NeurIPS , 2021.
- [5] Cong Fang, Hangfeng He, Qi Long, and Weijie J Su. Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training. PNAS , 118(43):e2103091118, 2021.
- [6] Jinxin Zhou, Xiao Li, Tianyu Ding, Chong You, Qing Qu, and Zhihui Zhu. On the optimization landscape of neural collapse under mse loss: Global optimality with unconstrained features. In ICML , 2022.
- [7] Tom Tirer and Joan Bruna. Extended unconstrained features model for exploring deep neural collapse. In ICML , 2022.
- [8] Peter Súkeník, Marco Mondelli, and Christoph H Lampert. Deep neural collapse is provably optimal for the deep unconstrained features model. NeurIPS , 2023.
- [9] Jiachen Jiang, Jinxin Zhou, Peng Wang, Qing Qu, Dustin G Mixon, Chong You, and Zhihui Zhu. Generalized neural collapse for a large number of classes. In ICML , 2024.
- [10] Akshay Rangamani and Andrzej Banburski-Fahey. Neural collapse in deep homogeneous classifiers and the role of weight decay. In ICASSP , 2022.
- [11] Peng Wang, Xiao Li, Can Yaras, Zhihui Zhu, Laura Balzano, Wei Hu, and Qing Qu. Understanding deep representation learning via layerwise feature compression and discrimination. arXiv preprint arXiv:2311.02960 , 2023.
- [12] Mengjia Xu, Akshay Rangamani, Qianli Liao, Tomer Galanti, and Tomaso Poggio. Dynamics in deep classifiers trained with the square loss: Normalization, low rank, neural collapse, and generalization bounds. Research , 6:0024, 2023.
- [13] Peng Wang, Huikang Liu, Can Yaras, Laura Balzano, and Qing Qu. Linear convergence analysis of neural collapse with unconstrained features. In OPT 2022: Optimization for Machine Learning (NeurIPS 2022 Workshop) , 2022.
- [14] Arthur Jacot, Peter Súkeník, Zihan Wang, and Marco Mondelli. Wide neural networks trained with weight decay provably exhibit neural collapse. In ICLR , 2025.
- [15] Gal Vardi. On the implicit bias in deep-learning algorithms. Communications of the ACM , 66 (6):86-93, 2023.
- [16] Hartmut Maennel, Olivier Bousquet, and Sylvain Gelly. Gradient descent quantizes relu network features. arXiv preprint arXiv:1803.08367 , 2018.
- [17] Mary Phuong and Christoph H Lampert. The inductive bias of relu networks on orthogonally separable data. In ICLR , 2021.
- [18] Etienne Boursier, Loucas Pullaud-Vivien, and Nicolas Flammarion. Gradient flow dynamics of shallow relu networks for square loss and orthogonal inputs. In NeurIPS , 2022.
- [19] Nikita Tsoy and Nikola Konstantinov. Simplicity bias of two-layer networks beyond linearly separable data. In ICML , 2024.
- [20] Hancheng Min, Enrique Mallada, and René Vidal. Early neuron alignment in two-layer relu networks with small initialization. In ICLR , 2024.
- [21] Ziwei Ji and Matus Telgarsky. Gradient descent aligns the layers of deep linear networks. In ICLR , 2019.

- [22] Frank H Clarke. Optimization and nonsmooth analysis . SIAM, 1990.
- [23] Wanli Hong and Shuyang Ling. Beyond unconstrained features: Neural collapse for shallow neural networks with general data. arXiv preprint arXiv:2409.01832 , 2024.
- [24] Dmitry Chistikov, Matthias Englert, and Ranko Lazic. Learning a neuron by a shallow reLU network: Dynamics and implicit bias for correlated inputs. In NeurIPS , 2023.
- [25] Simon S Du, Wei Hu, and Jason D Lee. Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced. In NeurIPS , 2018.
- [26] Ziwei Ji and Matus Telgarsky. Directional convergence and alignment in deep learning. In NeurIPS , 2020.
- [27] Kaifeng Lyu and Jian Li. Gradient descent maximizes the margin of homogeneous neural networks. In ICLR , 2019.
- [28] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS , 2019.
- [29] Etienne Boursier and Nicolas Flammarion. Early alignment in two-layer networks training is a two-edged sword. JMLR , 2025.
- [30] Akshay Kumar and Jarvis Haupt. Directional convergence near small initializations and saddles in two-homogeneous neural networks. TMLR , 2024.
- [31] Andrew M Saxe, James L Mcclelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural network. In ICLR , 2014.
- [32] Suriya Gunasekar, Blake Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nathan Srebro. Implicit regularization in matrix factorization. In NeurIPS , 2017.
- [33] Gauthier Gidel, Francis Bach, and Simon Lacoste-Julien. Implicit regularization of discrete gradient dynamics in linear neural networks. In NeurIPS , 2019.
- [34] Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In COLT , 2020.
- [35] Tao Luo, Zhi-Qin John Xu, Zheng Ma, and Yaoyu Zhang. Phase diagram for two-layer relu neural networks at infinite-width limit. JMLR , 22(71):1-47, 2021.
- [36] Dominik Stöger and Mahdi Soltanolkotabi. Small random initialization is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction. In NeurIPS , 2021.
- [37] Arthur Jacot, François Ged, Berfin ¸ Sim¸ sek, Clément Hongler, and Franck Gabriel. Saddle-tosaddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity. arXiv preprint arXiv:2106.15933 , 2021.
- [38] Noam Razin, Asaf Maman, and Nadav Cohen. Implicit regularization in hierarchical tensor factorization and deep convolutional neural networks. In International Conference on Machine Learning , pages 18422-18462. PMLR, 2022.
- [39] Ziqing Xu, Hancheng Min, Lachlan Ewen MacDonald, Jinqi Luo, Salma Tarmoun, Enrique Mallada, and Rene Vidal. Understanding the learning dynamics of lora: A gradient flow perspective on low-rank adaptation in matrix factorization. In AISTATS , 2025.
- [40] Zhenyu Zhu, Fanghui Liu, and Volkan Cevher. How gradient descent balances features: A dynamical analysis for two-layer neural networks. In ICLR , 2025.
- [41] Daniel Kunin, Giovanni Luca Marchetti, Feng Chen, Dhruva Karkada, James B Simon, Michael R DeWeese, Surya Ganguli, and Nina Miolane. Alternating gradient flows: A theory of feature learning in two-layer neural networks. arXiv preprint arXiv:2506.06489 , 2025.

- [42] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In NeurIPS , 2018.
- [43] Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In NeurIPS , 2019.
- [44] Vignesh Kothapalli and Tom Tirer. Can kernel methods explain how the data affects neural collapse? TMLR , 2025.
- [45] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE , 86(11):2278-2324, 1998.
- [46] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE CVPR , 2016.
- [47] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. Technical Report.
- [48] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450 , 2016.
- [49] Samuel Burer and Renato D. C. Monteiro. Local minima and convergence in low-rank semidefinite programming. Math. Program. , 103(3):427-444, July 2005.
- [50] Sanjeev Arora, Nadav Cohen, Noah Golowich, and Wei Hu. A convergence analysis of gradient descent for deep linear neural networks. In ICLR , 2018.
- [51] Hancheng Min, Salma Tarmoun, René Vidal, and Enrique Mallada. On the explicit role of initialization on the convergence and implicit bias of overparametrized linear networks. In ICML , 2021.
- [52] Hancheng Min, René Vidal, and Enrique Mallada. On the convergence of gradient flow on multi-layer linear models. In ICML , 2023.
- [53] Ziqing Xu, Hancheng Min, Salma Tarmoun, Enrique Mallada, and Rene Vidal. Linear convergence of gradient descent for finite width over-parametrized linear networks with general initialization. In AISTATS , 2023.
- [54] Arthur Castello B de Oliveira, Milad Siami, and Eduardo D Sontag. Dynamics and perturbations of overparameterized linear neural networks. In IEEE CDC , 2023.
- [55] Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. NeurIPS , 2019.
- [56] Liwei Jiang, Yudong Chen, and Lijun Ding. Algorithmic regularization in model-free overparametrized asymmetric matrix factorization. SIAM Journal on Mathematics of Data Science , 5(3):723-744, 2023.
- [57] Jikai Jin, Zhiyuan Li, Kaifeng Lyu, Simon Shaolei Du, and Jason D. Lee. Understanding incremental learning of gradient descent: A fine-grained analysis of matrix sensing. In ICML , 2023.
- [58] Daniel Kunin, Allan Raventós, Clémentine Dominé, Feng Chen, David Klindt, Andrew Saxe, and Surya Ganguli. Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning. In NeurIPS , 2024.
- [59] Hancheng Min and René Vidal. Understanding incremental learning with closed-form solution to gradient flow on overparamerterized matrix factorization. In IEEE CDC , 2025.
- [60] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR , 2022.
- [61] Yuanhe Zhang, Fanghui Liu, and Yudong Chen. Lora-one: One-step full gradient could suffice for fine-tuning large language models, provably and efficiently. In ICML , 2025.

- [62] Uijeong Jang, Jason D Lee, and Ernest K Ryu. Lora training in the ntk regime has no spurious local minima. In ICML , 2024.
- [63] Junsu Kim, Jaeyeon Kim, and Ernest K. Ryu. LoRA training provably converges to a low-rank global minimum or it fails loudly (but it probably won't fail). In ICML , 2025.
- [64] Salma Tarmoun, Guilherme França, Benjamin D Haeffele, and René Vidal. Understanding the dynamics of gradient flow in overparameterized linear models. In ICML , 2021.
- [65] Simon S Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In ICLR , 2019.
- [66] Samet Oymak and Mahdi Soltanolkotabi. Toward moderate overparameterization: Global convergence guarantees for training shallow neural networks. IEEE Journal on Selected Areas in Information Theory , 1(1):84-105, 2020.
- [67] Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha SohlDickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In NeurIPS .
- [68] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Limitations of lazy training of two-layers neural network. NeurIPS , 2019.
- [69] Gilad Yehudai and Ohad Shamir. On the power and limitations of random features for understanding neural networks. NeurIPS , 2019.
- [70] Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit. In COLT , 2019.
- [71] Lenaic Chizat and Francis Bach. On the global convergence of gradient descent for overparameterized models using optimal transport. NeurIPS , 2018.
- [72] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In COLT , 2022.
- [73] Matus Telgarsky. Feature selection and low test error in shallow low-rotation relu networks. In ICLR , 2023.
- [74] Spencer Frei, Niladri S Chatterji, and Peter Bartlett. Benign overfitting without linearity: Neural network classifiers trained by gradient descent for noisy linear data. In COLT , 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our abstract and introduction clearly state our claims and underlying assumptions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in Remark 1 and 2 and paragraph Limitations of current analysis .

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

Justification: Our assumptions are stated in the main paper, and we provide proofs of theorems in our technical appendices

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

Justification: We provide experimental details in Appendices B.1 and B.2.

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

Justification: We attach the code in the supplemental material

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

Justification: We provide experimental details in the Appendices B.1 and B.2

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: See Figure 4

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

Answer: [No]

Justification: The experiments are for validating our theoretical findings and do not requires heavy computer resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: the research follows the Code of Ethics

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: In this theory paper, there is no potential societal consequence that we feel must be specifically highlighted.

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

Justification: the paper poses no such risks

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: original papers that produced the datasets are cited

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

Justification: the paper does not release new assets

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: No LLMs used for writing this paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Additional Remarks and Related Works

We make the following remark on ReLU networks with biases

Extension to ReLU nets with biases When considering two-layer ReLU networks with biases, since σ ( ⟨ w , x ⟩ + b ) = σ ( ⟨ [ w ; b ] , [ x ; 1] ⟩ ) , adding a bias term effectively adds one homogenous coordinate to the entire dataset. Therefore, our results still hold if the augmented dataset satisfies the orthogonal separability condition. Notably, homogenous coordinate increases data correlation: ⟨ [ x i ; 1] , [ x j ; 1] ⟩ = ⟨ x i , x j ⟩ +1 , thus it mainly affects the negative correlation between data points with different labels. In the case when min i ∥ x i ∥ ≫ 1 , the orthogonal separability (among augmented data { [ x i ; 1] } n i =1 ) still holds with the bias term.

## A.1 Additional related works

Matrix factorization in deep learning theory A major part of the research efforts in the theoretical understandings of deep learning is through tractable mathematical problems, where similar phenomena can manifest to those observed in deep learning practice. Among these problems, matrix factorization [49] is an important one, which has been studied for understanding the convergence rate of gradient descent on neural networks [50-54], the implicit bias of neural network training algorithms [32, 55, 21, 31, 33, 37, 36, 56-59], and, as we already mentioned in the introduction, the NC phenomena [2-9]. More recently, the prevalence of LoRA [60] in practice has motivated many works on its theoretical properties [39, 61-63] through analyzing matrix factorization problems. While matrix factorization problems offer many valuable insights into various aspects of deep learning, they generally neglect the role of training data in these problems. For example, the convergence analysis often assumes input data with isotropic covariance [50, 64], and as we have discussed, the analysis of NC often assumes the last-layer features as an unconstrained optimization variable [2-4]. Given that NC is characterizing the ability of neural networks to map input data to structured latent features, it is also important to consider the role of input data. Indeed, our work highlights how the input data structure induces a directional collapse rather than a singleton collapse.

Gradient descent on shallow neural networks Theoretical properties of the gradient descent algorithms on shallow networks have been studied in different learning regimes and from various perspectives. Earlier works [65, 66] concern the convergence of gradient descent in the so-called kernel regime, where under specific settings (large network width, random weight initialization with large variance, etc.) the linearization around network weight initialization holds valid throughout training [42, 67]. However, limitations of such 'lazy regime" training [43] are identified through some specific student-teacher learning settings [68, 69]. This motivates the study of convergence in small initialization settings, often called active or feature learning regime [34]. From a dynamics perspective, in the infinite width limit with proper weight initialization scaling, the weight evolution during training can be characterized by some mean-field dynamics [70, 71]; In the finite width with vanishing weight initialization, the early training phase can be characterized by the directional alignment between the weights and the input data [16, 29, 19]. From a generalization perspective, learning in the feature learning regime can enjoy many advantages, such as sample efficiency [72, 73] or benign overfitting [74]. Our work studies the convergence of ReLU networks in the feature learning regime and contributes to this line of work in the following regards: First, prior works primarily concern the convergence of the weights, while our result discusses its implications on the learned last-layer feature. Second, from a technical perspective, our result addresses several challenges emerging from considering a multi-class problem with the cross-entropy loss.

## B Experimental Details

## B.1 Experimental details on classifying MNIST digits

Preprocessing data . We first preprocess the training data, i.e. digits { 0 , 1 , 2 } by centering: x i ← x i -¯ x , where ¯ x = ∑ i ∈ [ n ] x i /n is the global mean image of the entire training data. Then we have plotted the normalized correlation matrix [〈 x i ∥ x i ∥ , x i ′ ∥ x i ′ ∥ 〉] i,i ′ ∈ [ n ] of the centered data, showing in Figure 3(a) that two data points of the same digit are likely to have a positive correlation and those different digits are likely to have negative correlations. This suggests that the orthogonality separability assumption in Assumption 1 is approximately satisfied.

Training . Given the centered data, for a two-layer ReLU network (2) of width50 , we initialize all entries of the network weights with i.i.d. Gaussians with variance 10 -6 . Then we run SGD of batch size 1000 with learning rate 0 . 1 for 50 epochs. For the trained network, we visualize the output neuron weights v j , j ∈ [ h ] and determine the N k by letting N k = { j ∈ [ h ] : k = arg max k ′ ⟨ ˜ e k ′ , v j ⟩} , then also visualize the average direction of the input neuron weights ∑ j ∈N k w j ∥ ∑ j ∈N k w j ∥ for each group N k , as shown in Figure 3(b).

## B.2 Experimental details on normalization layers

Modified ResNet . We take the ResNet18 and ResNet50 implementations (The first conv layer is modified to accommodate MNIST and CIFAR10 input sizes) in Pytorch and replace the final linear classifier with a two-layer ReLU network of width1000 , and also add a normalization layer (Identity/None, LayerNorm, or RMSNorm) between the classifier and the feature extractor. The initialization follows the Pytorch default.

Training . For each choice of (model: ResNet18, ResNet50)-(Dataset: MNIST, CIFAR10), we repeat 5 runs (with different random seeds) of SGD of batch size 128 and learning rate 0 . 1 (for ResNet18) and 0 . 02 (for ResNet50) with momentum 0 . 95 for 50 epochs; and for every 20 epochs, we reduce the learning rate to 0 . 1 of its current value. We plot the NC metrics and test accuracy against training epochs in Figure 4.

Figure 5: Measuring NC in trained modified ResNet50 on MNIST (top) and CIFAR10 (bottom)

<!-- image -->

NC metrics . The NC metrics follow those used in prior works except for the projected self-duality. Given the class means ϕ k = ∑ i = ∈I k ϕ θ ( x i ) |I k | and global mean ¯ ϕ = ∑ K k =1 ϕ k /K , NC1 is defined to be the ratio between intra-class variance and the inter-class variance ∑ K k =1 ∑ i ∈I k ∥ ϕ θ ( x i ) -ϕ k ∥ 2 / |I k | ∑ K k =1 ∥ ϕ k -¯ ϕ ∥ 2 . NC2 is defined to be the proximity of the gram matrix of the class mean directions to the identity matrix ∥ G ∥ G ∥ F -1 √ K I ∥ F , where G = [ ¯ ϕ 1 · · · ¯ ϕ K ] ⊤ [ ¯ ϕ 1 · · · ¯ ϕ K ] . NC3 is defined to be the proximity of V ¯ Φ † to an identity matrix ∥ V ¯ Φ † ∥ V ¯ Φ † ∥ F -1 √ K -1 ˜ E ∥ F .

In the main paper, we have only provided the plot for ResNet18. We show the plot for ResNet50 in Figure 5.

## C Neural Alignment under Multi-class Orthogonally Separable Data

## C.1 Basics on neuron dynamics under multi-class problems

The differential inclusion ˙ θ ∈ -∇ θ L ( θ ) gives rise to the following characterization of the time derivatives of neuron weights ∀ j ∈ [ h ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It will become clear soon that it is convenient to decompose the weight dynamics into those of the weight norm and of the weight direction, for which we use the balancedness that ∥ w j ∥ ≡ ∥ v j ∥ , ∀ j :

(weight norm dynamics)

<!-- formula-not-decoded -->

(input neuron angular dynamics)

<!-- formula-not-decoded -->

(output neuron angular dynamics)

<!-- formula-not-decoded -->

where Π ⊥ u := ( I -uu ⊤ ∥ u ∥ 2 ) denote the project matrix onto the orthogonal complement of u .

By inspecting (22)(23)(24), we note that the dynamic of each neuron pair ( w j , v j ) is almost decoupled from each other except for the interaction through ˆ y i , i ∈ [ n ] . Interestingly, at the early phase of the GF, (we will show that) the norm of the weights remains close to zero, resulting in ˆ y i ≃ 1 K 1 , ∀ i ∈ [ n ] thus fully decouples the neuron pair dynamics, the precise statement on such an approximation ˆ y i ≃ 1 K 1 is as follow:

<!-- formula-not-decoded -->

Proof. First of all, we have

<!-- formula-not-decoded -->

We always have 1 -exp( -| z | ) ≤ | z | . Moreover, whenever | z | ≤ 1 , we have exp( | z | ) -1 ≤ 2 | z | . Therefore, we conclude that

<!-- formula-not-decoded -->

With (25), whenever ∥ f ( x i ; θ ) ∥ ≤ 1 , we have

<!-- formula-not-decoded -->

Now we bound ∥ ˆ y i -1 K 1 ∥ using entrywise bound. Notice that

<!-- formula-not-decoded -->

Finally, we have ∥ ˆ y i -1 K 1 ∥ ≤ √ K max k ∣ ∣ [ ˆ y i ] k -1 K ∣ ∣ ≤ 8 √ K ∥ f ( x i ; θ ) ∥ .

## C.2 Analyzing neuron dynamics during alignment phase

In this section, we show the formal statements for the alignment dynamics we have introduced in (16)(17). During the early phase of the GF training, the norms of the weights remain small (Lemma 2), leading to an approximate alignment dynamics in Lemma 3, which will be crucial for subsequent analysis.

Lemma 2. Given some balanced, ϵ -small initialization θ (0) with ϵ ≤ √ K 16 X max √ h , any solution θ ( t ) to the GF dynamics (3) satisfies that ∀ t ≤ 1 4 nX max log 1 √ hϵ := T ,

<!-- formula-not-decoded -->

The alignment phase refers to the training phase until T = 1 4 nX max log 1 √ hϵ . With Lemma (2), we can approximate the angular dynamics d d t w j ∥ w j ∥ and d d t v j ∥ v j ∥ throughout the alignment phase as follow:

Lemma 3. Given some balanced, ϵ -small initialization θ (0) with ϵ ≤ 1 16 X max √ h , any solution θ ( t ) to the GF dynamics (3) satisfies that ∀ t ≤ 1 4 nX max log 1 √ hϵ := T ,

<!-- formula-not-decoded -->

Proof of Lemma 2. From Section C.1, we have

<!-- formula-not-decoded -->

Let T := inf { t : max i | f ( x i ; θ ( t )) | &gt; 2 ϵX max √ h } , then ∀ t ≤ T, j ∈ [ h ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let τ j := inf { t : ∥ w j ( t ) ∥ 2 &gt; ϵ √ h } , and let j ∗ := arg min j τ j . Then τ j ∗ = min j τ j ≤ T , which can be shown by contradiction:

Suppose τ j ∗ &gt; T , then at t = T &lt; τ j ∗ , by the definition of τ j ∗ , we have max j ∥ w j ∥ 2 ≤ ϵ √ h , and by the definition of T and the continuity of θ ( t ) w.r.t. t , ∃ i ∗ ∈ [ n ] such that | f ( x i ∗ ; θ ( T )) | = 2 ϵX max √ h , therefore,

<!-- formula-not-decoded -->

which suggests that max j ∈ [ h ] ∥ w j ∥ 2 ≥ 2 ϵ √ h , a contradiction.

Now for t ≤ τ j ∗ ≤ T , we have

<!-- formula-not-decoded -->

By Grönwall's inequality, we have ∀ t ≤ τ j ∗

<!-- formula-not-decoded -->

Suppose τ j ∗ &lt; 1 4 nX max log 1 √ hϵ , then by the continuity of ∥ w j ∗ ( t ) ∥ 2 , we have

<!-- formula-not-decoded -->

where the last inequality is due to ϵ ≤ √ K 16 X max √ h . This leads to a contradiction. Therefore, one must have T ≥ τ j ∗ ≥ 1 4 nX max log ( 1 √ hϵ ) . This finishes the proof.

Proof of Lemma 3. We have shown in Section C.1 that

<!-- formula-not-decoded -->

Therefore, ∀ ≤ T ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we have

<!-- formula-not-decoded -->

where we note that applying Lemma 1 requires ∥ f ( x i ; θ ) ∥ ≤ 1 4 , which is guaranteed by Lemma 2 and our choice ϵ ≤ 1 16 X max √ h . We have shown the approximation error bound for d d t w j ∥ w j ∥ . A similar bound can be derived for d d t v j ∥ v j ∥ .

## C.3 Neural alignment under multi-class orthogonally separable data

Sufficient statement for Proposition 1 . It is easy to check that the following proposition is sufficient for Proposition 1 to hold.

Proposition 3 (Sufficient statement for Proposition 1) . Let K&gt; 2 . Given orthogonally separable data (Assumption 1) with X 2 max X 2 min µ d µ 2 s &lt; 2 K -3 , where X max := max i ∈ [ n ] ∥ x i ∥ and X min := min i ∈ [ n ] ∥ x i ∥ , ϵ -small, balanced and semi-local initialization (Assumptions 2, 4) for a sufficiently small ϵ , for any solution θ ( t ) , t ≥ 0 to (3) and any j ∈ N k , k ∈ [ K ] , define

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Therefore, we can study the dynamic behavior of each neuron pair individually, for convenience, let j ∈ N k , and we drop the index j .

For a neuron pair ( w , v ) , we have defined the following:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

(alignment between input neuron and k -th class)

(alignment between output neuron and k -th class)

Overview of the proof of Proposition 1 . First, we utilize the alignment dynamics in Lemma 3, to show that (Recall that T = 1 4 nX max log 1 √ hϵ )

Lemma 4. Given a neuron ( w j , v j ) , j ∈ N k , during the alignment phase t ≤ min { T ∗ j,k , T } , the following holds:

we mostly use this notation for clarity

,

1. v j ∥ v j ∥ remains close to its target pseudo-label: B k ≥ 1 -1 2( K -1) ;

̸

2. w j ∥ w j ∥ remains close to its target class: A k -2 ∑ k ′ = k A k ′ ≥ A k (0) -2 ∑ k ′ = k A k ′ (0) ;

̸

̸

3. Neuron w j ∥ w j ∥ does not deactivate data from target class, nor activate data from non-target class: |I w k | ≥ |I w (0) k | and |I w k ′ | ≤ |I w (0) k ′ | , ∀ k ′ = k .

The characterizations in 4 suggest that the neuron weight directions { w j ∥ w j ∥ , v j ∥ v j ∥ } remains close to the attractor { ¯ x k , ˜ e k } , and as the weights move closer to the attractor, |I w j k | increases to |I k | , and |I w j k ′ | decreases to 0. When the initialization scale ϵ is sufficiently small so that T is large, this Lemma will show that T ∗ j,k is finite, and we will provide an upper bound.

Then the following lemma shows that the desired property for neuron ( w j , v j ) still holds after T ∗ j,k

Lemma 5. for any neuron ( v j , w j ) , j ∈ N k , we have ∀ t &gt; T j,k

1. v j ∥ v j ∥ remains close to its target pseudo-label: B v j k ≥ √ 2 2 ;
2. w j ∥ w j ∥ is exclusively activated by data from its target class: |I w j k | = N k , |I w j k ′ | = 0 , ∀ k ′ = k .

. ∗ :

̸

The remaining parts of this section are dedicated to proving these two Lemmas. The next section will formally prove Proposition 3, thereby proving Proposition 1.

## C.3.1 Proof of Lemma 4

Basic dynamics . The main proof concerns the time derivatives of the alignment to classes d d t A k , d d t B k . With Lemma 3, we have their approximations during the alignment phase:

<!-- formula-not-decoded -->

where (39) uses the simple fact that ∑ i ′ ∈I k ′ : α i ′ ≥ 0 ξ i ′ α i ′ = ∑ i ′ ∈I k ′ : α i ′ &gt; 0 α i ′ = A k ′ , (38) uses the fact that β i ′ = 〈 ˜ Ey i ′ , v ∥ v ∥ 〉 = 〈 ˜ e k ′ , v ∥ v ∥ 〉 if i ′ ∈ I k ′ , (37) uses the fact that ξ i ′ = 0 if α i ′ &lt; 0 , and (36) uses Lemma 3, with the O ( ϵ ) term being

<!-- formula-not-decoded -->

whose norm can be upper bounded as follows:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

̸

where multiple facts used to derive (39) are also used here, and in (40), the O ( ϵ ) has its norm upper bounded by 16 √ K ϵnX 2 max √ h .

Axuillary Lemmas . The following lemmas will be needed.

Lemma 6. Given B k , k = 1 , · · · , K defined for a single neuron pair ( w , v ) , we have

̸

<!-- formula-not-decoded -->

Proof. With the following basic derivation

<!-- formula-not-decoded -->

Lemma 7. Given a dataset that satisfies Assumption 1, then the following is true:

- ∀ k and some a i , ∀ i ∈ I k , we have

̸

<!-- formula-not-decoded -->

- ∀ k = k ′ and some a i , b i ′ ≥ 0 , ∀ i ∈ I k , i ′ ∈ I i ′ , we have

<!-- formula-not-decoded -->

Proof. For the first inequality,

<!-- formula-not-decoded -->

For the second inequality,

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

Lemma 8. Given { z i , i ∈ I} with ⟨ z i , z j ⟩ ≤ 0 , ∀ i, j ∈ I , i = j , then ∥ ∑ i ∈I z i ∥ ≤ √∑ i ∈I ∥ z i ∥ 2 .

̸

̸

<!-- formula-not-decoded -->

Lemma 9. Given a dataset that satisfies Assumption 1, ∃ 0 &lt; ζ &lt; 1 such that ∀ k ∈ [ K ] and ∀ w ∈ { z : 0 &lt; |I z k | &lt; |I k |} , we have ∥ ∥ ∑ i ∈I k : α i &gt; 0 x i ∥ ∥ 2 -A 2 k ≥ µ s X 2 min ζ .

Proof. Notice that

<!-- formula-not-decoded -->

However, the nonnegative quantity ( 1 -〈 w ∥ w ∥ , ∑ i ∈I k : α i &gt; 0 x i ∥ ∑ i ∈I k : α i &gt; 0 x i ∥ 〉 2 ) can not be zero: Suppose it is zero, then w ∝ ± ∑ i ∈I k : α i &gt; 0 x i , which corresponds to either |I z k | = 0 or |I z k | = |I k | , a contradiction. We let its lowest value be ζ &gt; 0 . This finishes the proof.

The proof . Now we are ready to prove Lemma 4.

Proof of Lemma 4. We define the following:

<!-- formula-not-decoded -->

̸

Then it suffices to show that min { τ 1 , τ 2 , τ 3 } ≥ min { T ∗ j,k , T } , for which we prove them by contra-( ∗ )

diction. Note: In the proof we will use " ≥ " to represent an inequality that holds when ϵ is sufficiently small.

Case 1: min { τ 1 , τ 2 , τ 3 } = τ 1 .

At τ 1 , by the continuity of B k , we must have B k ( τ 1 ) = 1 -1 2( K -1) . Suppose τ 1 ≤ min { T ∗ j,k , T } , then we have the following derivation

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

The definition of τ 1 suggests that B k must drop below 1 -1 2( K -1) right after t = τ 1 , which contradicts that d d t B k ∣ ∣ t = τ 1 ≥ 0 . Therefore min { τ 1 , τ 2 , τ 3 } &gt; min { T ∗ j,k , T } can not be true under the case when min { τ 1 , τ 2 , τ 3 } = τ 1 .

Case 2: min { τ 1 , τ 2 , τ 3 } = τ 2 .

Again, we derive a contradiction by supposing τ 2 ≤ min { T ∗ , T } . Since min { τ 1 , τ 2 , τ 3 } = τ 2 , at τ 2 we still have B k ≥ 1 -1 2( K -1) &gt; 0 , and by Lemma 6, we also have B k ′ ≤ 2(1 -B k ) -1 K -1 ≤ 0 .

Starting from (39) restricted to t = τ 2 , we have for the target class,

̸

<!-- formula-not-decoded -->

and for non-target classes, we have

̸

<!-- formula-not-decoded -->

̸

̸

The last step to get (47) is to upper bound the three terms in (46) separately, which we defer to the end of this proof. Combining (45)(47), and recalling the upper bound on the norm of the O ( ϵ ) terms, we have

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

̸

The definition of τ 2 suggests that A k -2 ∑ k ′ = k A k ′ must drop below A k (0) -2 ∑ k ′ = k A k ′ (0) right after t = τ 2 , which contradicts that d d t ( A k -2 ∑ k ′ = k A k ′ )∣ ∣ ∣ t = τ 2 ≥ 0 . Therefore min { τ 1 , τ 2 , τ 3 } &gt; min { T ∗ j,k , T } can not be true under the case when min { τ 1 , τ 2 , τ 3 } = τ 2 .

̸

Case 3: min { τ 1 , τ 2 , τ 3 } = τ 3 . Finally, it remains to exclude the case when min { τ 1 , τ 2 , τ 3 } = τ 3 and τ 3 ≤ min { T ∗ j,k , T } . At t = τ 3 , either of the following must happen:

1. ∃ i ∈ I k such that α i = 0 and d d t α i &lt; 0 ;

̸

2. ∃ i ∈ I k ′ for some k ′ = k such that α i = 0 and d d t α i &gt; 0 ;

However, at t = τ 3 , ∀ i ∈ I k , we have

<!-- formula-not-decoded -->

̸

̸

therefore it can not be that ∃ i ∈ I k such that α i = 0 and d d t α i &lt; 0 . Next, ∀ i ∈ I k ′ , k ′ = k , we have

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

therefore it can not be that ∃ i ∈ I k ′ , k ′ = k such that α i = 0 and d d t α i &gt; 0 . By excluding both scenarios, min { τ 1 , τ 2 , τ 3 } &gt; min { T ∗ j,k , T } can not be true under the case when min { τ 1 , τ 2 , τ 3 } = τ 3 . The proof is complete once we add the derivations for 47.

̸

Complete the proof . Lastly, it remains to prove (47), which comes from the following derivations: For the first term,

<!-- formula-not-decoded -->

For the second term,

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

where (51) uses that -2 K -1 ≤ B k ′ ≤ 0 , ∀ k ′ = k by Lemma 6. And for the last term,

̸

̸

<!-- formula-not-decoded -->

̸

## C.3.2 Proof of Lemma 5

We will use the following lemma:

Lemma 10. For any ¯ v ∈ S K -1 such that ⟨ ¯ v , ˜ e 1 ⟩ = β ∈ [0 , 1] , then ∀ p such that p ≥ 0 , [ p ] 1 = 0 , and ⟨ p , 1 ⟩ = 1 , we have

<!-- formula-not-decoded -->

Proof. First of all, since min k&gt; 1 [¯ v ] k ≤ ⟨ p , ¯ v ⟩ ≤ max k&gt; 1 [¯ v ] k , we know that

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

Now given that ⟨ ¯ v , ˜ e 1 ⟩ = β , we can write ¯ v = β ˜ e 1 + √ 1 -β 2 y ⊥ , where y ⊥ ∈ S K -1 and y ⊥ ˜ e 1 . Therefore,

<!-- formula-not-decoded -->

Proof of Lemma 5. Without loss of generality, we prove this lemma for k = 1 . We define

̸

<!-- formula-not-decoded -->

̸

We need to show that min { T 1 , T 2 } = ∞ . We derive a contradiction by assuming it is finite.

Case one: min { T 1 , T 2 } = T 1 is finite

Assuming min { T 1 , T 2 } = T 1 is finite, our primary focus is the angular dynamics of v j ∥ v j ∥ , ∀ j ∈ N 1 ,

<!-- formula-not-decoded -->

and in particular those of its alignment with pseudo-label ˜ e 1 ,

<!-- formula-not-decoded -->

We shall focus on the term 〈 ˜ e 1 , Π ⊥ v j ( e 1 -ˆ y i ) 〉 . For each i ∈ I 1 , we let z ik = [ V Wx i ] k = [ ∑ j ∈N 1 v j w ⊤ j x i ] k , then

<!-- formula-not-decoded -->

thus we have

<!-- formula-not-decoded -->

from which we see that at t = T 1 , we have

<!-- formula-not-decoded -->

contradicting the definition of T 1 .

Case two: min { T 1 , T 2 } = T 2 is finite

Assuming min { T 1 , T 2 } = T 2 is finite, we shall focus on the time interval [ T ∗ j,k , T 2 ] , when we have

<!-- formula-not-decoded -->

From (54), we have ∀ t ≤ T 2

<!-- formula-not-decoded -->

Therefore, by the Fundamental Theorem of Calculus, we have

<!-- formula-not-decoded -->

which ensures that |I w j ( T 2 ) k | = |I k | and |I w j ( T 2 ) k ′ | = 0 , contradicting to the definition of T 2 . Therefore, the proof is finished by the fact that min { T 1 , T 2 } cannot be finite.

## C.4 Proof of Proposition 1

As we have discussed in Appendix C.3, it suffices to prove Proposition 3.

Proof of Proposition 3. We have shown that before min { T ∗ j,k , T } , the properties of the weights in Lemma 4 hold. We consider a sufficiently small ϵ such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Then we show that min { T ∗ j,k , T } = T ∗ j,k by contradiction: Suppose that T ≤ T ∗ j,k , then during [0 , T ] , we have, from (45),

<!-- formula-not-decoded -->

Then by the Fundamental Theorem of Calculus, we have

<!-- formula-not-decoded -->

which is a contradiction, knowing that A k cannot exceed X max |I k | . Therefore, we must have min { T ∗ j,k , T } = T ∗ j,k and T ∗ j,k ≤ T = 1 4 nX max log 1 √ hϵ is finite. Then the rest of the Proposition 3 follows Lemma 5.

## D Asymptotic Convergence Analysis under Multi-class Orthogonally Separable Data

## D.1 Basic results upon inter-class separation

With the loss decomposition upon inter-class separation, for which we have shown to persist after T ∗ = max j,k T ∗ j,k ,

<!-- formula-not-decoded -->

It suffices to study the following GF on ∑ n k i =1 ℓ CE ( y k,i , V k W ⊤ k x k,i ) :

<!-- formula-not-decoded -->

The following basic results can be obtained from [21, 27]

<!-- formula-not-decoded -->

3. W k , V k is a KKT point of

<!-- formula-not-decoded -->

## D.2 Proof of Proposition 2

Our proof of Proposition 2 follows the same strategy as those in [17, 21], with the major difference being that we are handling cross-entropy loss, in which we provide an extension of Lemma 2.11 in [21], stated as Lemma 13. Lemma 13 is central to our proof.

Lemma 11. Let γ := min 1 ≤ k ≤ K γ k , where γ k := min i ∈I k ⟨ ¯ u ∞ ,k , x i ⟩ , then γ ≥ µ s X min .

Proof. For any 1 ≤ k ≤ K , ¯ u ∞ ,k = ∑ i ∈I k a i x i , for some a i ≥ 0 , then immediately we have, ∀ i ∈ I k

<!-- formula-not-decoded -->

Lemma 12. γ ⊥ := min k ∈ [ K ] min ∥ ξ ∥ =1 ,ξ ⊥ ¯ u k max i ∈I k ⟨ ξ, x i ⟩ &gt; 0

Proof. This result is from Lemma 2.10 in [21]. Note that the referenced Lemma requires an additional assumption that the support vectors of x i , i ∈ I k span the ambient space, but the authors of [21] have commented that this condition can be relaxed to the case that the span of support vectors is the span of x i , i ∈ I k , which is true here given the positive correlations between x i , i ∈ I k .

̸

Lemma 13. Given some Θ = [ θ 1 , · · · , θ K ] ∈ R D × K and some 1 ≤ k ≤ K . If it holds that ∃ k ′ = k , ( θ k -θ k ′ ) ⊤ ¯ u ∞ ,k &gt; 0 and ∥ Π ⊥ ¯ u ∞ ( θ k -θ k ′ ) ∥ sufficiently large, then tr ( ( e k 1 T n -ˆ Y ) ⊤ Θ ⊤ Π ⊥ ¯ u ∞ X ) ≤ 0 , where ˆ Y = SoftmaxCol( Θ ⊤ X ) .

Proof. It suffices to prove the case when k = 1 (We discuss the others at the end of the proof). We start by the following derivations:

<!-- formula-not-decoded -->

̸

̸

̸

̸

<!-- formula-not-decoded -->

For the k -th summand, let we have

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

One can upper bound these two terms separately as follows:

̸

<!-- formula-not-decoded -->

and for the second term,

<!-- formula-not-decoded -->

̸

Therefore, putting (68)(70)(71) together, we have

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

when ∥ Π ⊥ ¯ u ∞ ( θ 1 -θ k ′ ) ∥ is sufficiently large.

̸

If k = 1 , consider the permutation matrix P 1 ↔ k that swap the 1 -st and k -th rows/columns of a matrix, then

<!-- formula-not-decoded -->

Following the derivations for the case k = 1 gives the desired result.

Proof of Proposition 2. Without loss of generality, we prove the case of k = 1 . The existence { ¯ W 1 , ¯ V 1 } is by [27]. We first show that ¯ W 1 ∝ u 1 g ⊤ 1 , which is equivalent to the statement that ∥ Π ⊥ u 1 ¯ W 1 ∥ F ∥ ¯ W 1 ∥ F = 0 , and we prove by contradiction. Suppose ∥ Π ⊥ u 1 ¯ W 1 ∥ F ∥ ¯ W 1 ∥ F &gt; 0 , which necessarily implies that ∃ ρ &gt; 0 such that ∥ Π ⊥ u 1 ¯ W 1 ¯ W ⊤ 1 ∥ F ∥ ¯ W 1 ¯ W ⊤ 1 ∥ F = ρ , then for any ϵ &gt; 0 , M &gt; 0 exists T ϵ,M &gt; 0 such that ∀ t ≥ T ϵ,M , we have ∥ ¯ W 1 ¯ V ⊤ 1 ∥ ¯ W 1 ¯ V ⊤ 1 ∥ F -W 1 V ⊤ 1 ∥ W 1 V ⊤ 1 ∥ F ∥ ≤ ϵ and ∥ W 1 V ⊤ 1 ∥ F ≥ M . We will make clear the choice of ϵ and M later.

Consider the time derivative of ∥ Π ⊥ ¯ u ∞ W ∥ 2 F :

<!-- formula-not-decoded -->

We would like to use the result in Lemma 13 so we should examine:

<!-- formula-not-decoded -->

̸

Therefore, ∃ δ &gt; 0 , k = 1 such that ∥ Π ⊥ ¯ u ∞ ¯ W 1 ¯ V T 1 ( e 1 -e k ) ∥ 2 = δ , otherwise, ∥ Π ⊥ ¯ u ∞ ¯ W 1 ¯ V T 1 ( e 1 -e k ) ∥ = 0 , ∀ k = 1 , which can not happen. Since e 1 -e k , k = 1 spans a k -1 -dimensional subspace orthogonal to 1 √ K , the projection of Π ⊥ ¯ u ∞ ¯ W 1 ¯ V T 1 onto this subspace is zero suggests Π ⊥ ¯ u ∞ ¯ W 1 ¯ V T 1 is rank1 and all columns of V k are aligned with 1 √ K , which contradicts our alignment result in Lemma 4 (these columns must have at least √ 2 2 cosine alignment with ˜ e 1 ).

̸

̸

Then ∀ t ≥ T ϵ,M , and for the k such that ∥ Π ⊥ ¯ u ∞ ¯ W 1 ¯ V T 1 ( e 1 -e k ) ∥ 2 = δ , we have

<!-- formula-not-decoded -->

Choose sufficiently small ϵ and sufficiently large M , we ensure that ∥ Π ⊥ ¯ u ∞ W 1 V T 1 ( e 1 -e k ) ∥ is sufficiently large to apply Lemma 13 so that d d t ∥ Π ⊥ ¯ u ∞ W 1 ∥ 2 F = 2tr ( ( e 1 1 T n -ˆ Y ) ⊤ V 1 W ⊤ 1 Π ⊥ ¯ u ∞ X ) ≤ 0 .

On the other hand, ∥ W k ∥ 2 F →∞ , contradicting our assumption that ∥ Π ⊥ u 1 ¯ W 1 ∥ F ∥ ¯ W 1 ∥ F &gt; 0 . This proves that ¯ W 1 ∝ u 1 g ⊤ 1 .

By balancedness ¯ V ⊤ 1 ¯ V 1 -¯ W ⊤ 1 ¯ W 1 = 0 , we know that ¯ V 1 ∝ ¯ v 1 g ⊤ 1 for some ¯ v 1 ∈ S K -1 . It remains to show that ¯ v 1 = ˜ e 1 , which is proved by again contradiction.

̸

Suppose ¯ v 1 = ˜ e 1 , then ∃ k ∗ such that [¯ v 1 ] k ∗ ≥ [¯ v 1 ] k , ∀ k = k ∗ , k = 1 , and not all equalities can be obtained. As a results, consider [0 -exp(˜ z i 2 ) · · · -exp(˜ z iK )] / ∑ k&gt; 1 exp(˜ z ik ) that appeared in (55), it converges to e k ∗ , ∀ i ∈ [ n ] . Based on this, for any ϵ 1 , ϵ 2 , ∃ T ϵ 1 ,ϵ 2 such that ∀ t &gt; T ϵ 1 ,ϵ 2 , we have max j ∈N 1 ∥ v j ∥ v j ∥ -¯ v 1 ∥ ≤ ϵ 1 and max i ∈ [ n ] ∥ [0 -exp(˜ z i 2 ) · · · -exp(˜ z iK )] / ∑ k&gt; 1 exp(˜ z ik ) -e k ∗ ∥ ≤ ϵ 2 . Therefore, for some j ∈ [ h ] and t &gt; T ϵ 1 ,ϵ 2 , we have, from (55)

̸

̸

<!-- formula-not-decoded -->

The right-hand side of (74) is positive and Θ( 1 t ) , by the fact that weight ∥ W 1 ∥ , ∥ V 1 ∥ grow at a rate Θ(log( t )) . Therefore (74) suggests the divergence of B v j j , a contradiction.

Finally, we have shown ¯ W 1 ∝ u 1 g ⊤ 1 and ¯ V 1 ∝ ˜ e 1 g ⊤ 1 , and the same for other k . The choices of s k are determined by the fact that ¯ W k , ¯ V k must be a KKT point of (66).
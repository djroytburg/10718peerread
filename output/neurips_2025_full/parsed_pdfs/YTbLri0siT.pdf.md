## Spike-timing-dependent Hebbian learning as noisy gradient descent

## Niklas Dexheimer 1 ∗ Sascha Gaudlitz 2 ∗ Johannes Schmidt-Hieber 1

n.dexheimer, a.j.schmidt-hieber @utwente.nl sascha.gaudlitz@hu-berlin.de

1 University of Twente 2 Humboldt-Universität zu Berlin { }

## Abstract

Hebbian learning is a key principle underlying learning in biological neural networks. We relate a Hebbian spike-timing-dependent plasticity rule to noisy gradient descent with respect to a non-convex loss function on the probability simplex. Despite the constant injection of noise and the non-convexity of the underlying optimization problem, one can rigorously prove that the considered Hebbian learning dynamic identifies the presynaptic neuron with the highest activity and that the convergence is exponentially fast in the number of iterations. This is non-standard and surprising as typically noisy gradient descent with fixed noise level only converges to a stationary regime where the noise causes the dynamic to fluctuate around a minimiser.

## 1 Introduction

Hebbian learning is a fundamental concept in computational neuroscience, dating back to Hebb [16]. In this work, we provide a rigorous analysis of a Hebbian spike-timing-dependent plasticity (STDP) rule. Those are learning rules for the synaptic strength parameters that only depend on the spike times of the involved neurons. More precisely, we consider a neural network composed of d presynaptic/input neurons, which are connected to one postsynaptic/output neuron. The presynaptic neurons communicate with the postsynaptic neuron by sending spike sequences, the so-called spike-trains. Reweighted by synaptic strength parameters w 1 , . . . , w d ≥ 0 , they contribute to the postsynaptic membrane potential. Whenever the postsynaptic membrane potential exceeds a threshold, the postsynaptic neuron emits a spike, and the membrane potential is reset to zero. Experiments have shown the following stylised facts, which lie at the core of Hebbian learning based on spikes: (1) Locality: The change of the synaptic weight w i depends only on the spike-train of neuron i and the postsynaptic spike-train. (2) Spike-timing: The change of the synaptic weight w i depends on the relative timing of presynaptic spikes of neuron i and

Figure 1: Neural network with a single output neuron

<!-- image -->

of the postsynaptic neuron. More precisely, a pre-post spike sequence tends to increase w i , whereas a post-pre sequence tends to decrease w i . We refer to Morrison et al. [28] for a more comprehensive list of experimental results on STDP rules.

Hebbian learning rules are well-studied if instead of the precise timings of pre- and postsynaptic spikes, only the mean firing rates are taken into account. These rate-based models exhibit many desirable properties, including performing streaming PCA [23, 19] and receptive field development [13, Section 11.1.4]. Much less is known if the precise timing of pre- and postsynaptic spikes is considered, since the intrinsic randomness of the dynamics complicates the mathematical analysis.

∗ These authors contributed equally to this work.

Our main contribution lies in connecting STDP to noisy gradient descent and providing a rigorous convergence analysis of the noisy learning scheme. To this end, we introduce a learning rule for the weights w 1 , . . . , w d , which captures the locality and spike-time dependence of Hebbian STDP. We rewrite the learning rule as a noisy gradient descent scheme with respect to a suitable loss function. The connection to noisy gradient descent and stochastic approximation [24, 34] paves the way for applying mathematical tools from stochastic process theory to analyse the STDP rule. Our analysis of STDP is inspired by the work on noisy gradient descent for non-convex loss functions of Mertikopoulos et al. [27]. By refining their arguments and carefully tracking the error terms, we show an exponentially fast alignment of the output neuron with the input neuron of the highest mean firing rate on an event of high probability. The specialisation of the output neuron to the input neuron of the highest intensity is related to the winner-take-all mechanism in decision making [12, 49, 31, 25, 40]. The competitive nature of Hebbian STDP has been observed by [39, 38, 15] and the specialisation to few input neurons is important for receptive field development [9]. By connecting Hebbian STDP to noisy gradient descent, we are able to provide a mathematical analysis beyond ensemble averages and to quantify the speed of convergence.

Taking into account the intrinsic geometry of the probability simplex, we also relate our learning rule to noisy mirror descent, more precisely to noisy entropic gradient descent, which has been proposed for brain-like learning by Neumann et al. [30], Cornford et al. [10].

The key contributions are:

1. STDP as noisy gradient descent. We deduce a new framework, in which Hebbian STDP is interpreted as noisy gradient descent. This connection allows us to employ powerful tools from the theory of stochastic processes for analysing Hebbian STDP.
2. Linear convergence. We prove the alignment of the output neuron with the input neuron of highest intensity at exponential rate on an event of high probability.
3. Connection to noisy mirror descent. We relate our learning rule to noisy mirror descent, more specifically to entropic gradient descent. This connection facilitates the integration of techniques from both areas, potentially leading to future synergistic effects.

## Related literature

Common approaches to understanding STDP restrict to the mean behaviour after taking the ensemble average, e.g. [13, 22, 15], or compute the full distribution using the master equation of the Markov process [13, Section 11.2.4]. Unfortunately, the latter is only feasible in specific scenarios. In [21], the authors consider a general noisy spike-time dependent dynamic which is transformed into a deterministic ODE by imposing a slow learning rate and using the self-averaging effect of the system. A stability analysis reveals structure formation and output stabilisation. One major difference to our work is the influence of the noise. In [21], the variance of the weights grows linearly and a careful comparison of time scales is required. In our work, despite a constant injection of noise into the system, the dynamic for the spike-triggering probabilities converges to a deterministic limit. Secondly, the use of recent ideas from the analysis of noisy SGD allows us to track the influence of the realised noise in every step. A considerable number of previous works derived STDP rules based on the minimisation of a loss function, typically corresponding to the minimization/maximization of some notion of energy or information, see [8, 6, 7, 44, 42, 32, 43, 36]. While this approach is appealing, the mathematical analysis of these learning rules is challenging due to the modifications required to achieve biological plausibility. In contrast, we start with a biologically plausible learning rule and utilise the arising loss function to derive mathematical convergence guarantees of the learning rule. The importance of the choice of a suitable metric for the derivation of the learning rule is laid out in [41]. We refer to [1, 48, 37, 14, 45] and the references therein for further results on STDP. In [20], a related learning rule is mathematically analyzed by relating it to expert aggregation.

## 1.1 Notation

Linear algebra. For a positive integer d , we write [ d ] := { 1 , . . . , d } and 1 := (1 , . . . , 1) ⊤ ∈ R d . For i ∈ [ d ] we denote by e i the i th standard basis vector of R d . The Hadamard product between two vectors a , b ∈ R d is denoted by a glyph[circledot] b := ( a 1 b 1 , . . . , a d b d ) ⊤ ∈ R d . We write I ∈ R d × d for the identity matrix on R d and ‖ u ‖ 2 = ∑ d i =1 u 2 i for the squared Euclidean norm of a vector u ∈ R d .

Probability. M (1 , p ) denotes the multinomial distribution with one trial ( n = 1 ) and probability vector p = ( p 1 , . . . , p d ) ⊤ , that is ξ ∼ M (1 , p ) if only only if P ( ξ = i ) = p i for any i ∈ [ d ] . We denote by

<!-- formula-not-decoded -->

the probability simplex in R d . We denote by /x31 A the indicator function of a set A .

## 1.2 Hebbian inspired learning rule

Inspired by Hebbian learning, we consider an unsupervised learning dynamic with d input (or presynaptic) neurons and one output (or postsynaptic) neuron. The i th input neuron has a mean firing rate λ i &gt; 0 describing the expected number of spikes per time unit. The vector λ = ( λ 1 , . . . , λ d ) contains the d mean firing rates. The strength of the connection between the i th input neuron and the output neuron is modulated by the weight parameter w i ≥ 0 , and changes to encode the information of the input firing rates.

We introduce a Hebbian STDP rule in Subsection 2.3 and show that, under some assumptions on the spike-trains, it is equivalent to the following dynamics. If w (0) = ( w 1 (0) , . . . , w d (0)) ⊤ are the d weights at initialisation, the updating rule from w ( k ) to w ( k +1) is given by

<!-- formula-not-decoded -->

w ( k +1) = w ( k ) glyph[circledot] ( 1 + α ( B ( k ) + Z ( k )) ) , (1) where α &gt; 0 is the learning rate and k = 0 , 1 , . . . denotes the postsynaptic spike time. The d -dimensional vector B ( k ) is the standard basis vector pointing to the presynaptic neuron, which triggered the ( k +1) st postsynaptic spike. It is given as B ( k ) = ∑ d i =1 /x31 ζ k = i e i , the one-hot encoding of independent multinomial random variables ζ k ∼ M (1 , p ( k )) , with k -dependent probability vector

Since the probabilities p i ( k ) model the probability that the ( k +1) st postsynaptic spike is triggered by neuron i = 1 , . . . , d , we call them (postsynaptic) spike-triggering-probabilities . The i.i.d. d -dimensional vectors Z ( k ) , k = 0 , 1 , . . . model the contribution of presynaptic spikes, which did not trigger the ( k +1) st postsynaptic spike, to the weight change. They are modelled to have i.i.d. components Z 1 ( k ) , . . . , Z d ( k ) , which are supported in [ -( Q -1) , ( Q -1)] , for some Q &gt; 1 , and centred such that E [ Z ( k )] = 0 .

In the remainder of the paper, we analyse the long-run behaviour of p ( k ) as k → ∞ under the learning rule Eq. (3). We say that the output neuron aligns with the j th input neuron if p j ( k ) → 1 as k →∞ . Since the input intensities λ 1 , . . . , λ d &gt; 0 are fixed throughout the dynamic, this condition is equivalent to w j ( k ) / ∑ d i =1 w i ( k ) → 1 as k →∞ . Figure 1 visualises the learning rule Eq. (2).

## 2 Representation as noisy gradient descent

We continue by relating the learning rule Eq. (1) to noisy gradient descent. For notational simplicity, define Y ( k ) := B ( k ) + Z ( k ) for k = 0 , 1 , . . . . Combining the weight updates Eq. (1) with the formula for the probabilities p from Eq. (2), we find

<!-- formula-not-decoded -->

The normalisation in the denominator and the multiplicative nature of the update ensures that the dynamic of p ( k ) is restricted to the probability simplex. By a Taylor expansion around α = 0 , we find

Since

<!-- formula-not-decoded -->

the random vectors

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 2: Contour plot of the loss function L from Eq. (6) on the probability simplex P for d = 3 with different overlays. Left: Three sample trajectories of Eq. (3) with different initial configurations p (0) . Middle: Stream plot of the gradient field given by Eq. (7). Right: 100 sample trajectories of Eq. (3) with p (0) = (0 . 3 , 0 . 3 , 0 . 4) ⊤ . All trajectories are simulated with 2000 iteration steps, learning rate α = 0 . 01 and Z ( k ) ∼ Unif([ -1 , 1] d ) .

<!-- image -->

are centred. The distribution of ξ ( k ) depends on w ( k ) and p ( k ) . Up to O ( α 2 ) -terms, we can write the learning rule Eq. (3) as a noisy gradient descent scheme

<!-- formula-not-decoded -->

for the loss function

<!-- formula-not-decoded -->

with gradient

Dropping O ( α 2 ) terms is only done for illustrative purposes. Our main result (Theorem 2.2) applies to the original learning rule Eq. (3). The subsequent lemma summarises the key properties of the loss function L from Eq. (6). For d = 3 , Figure 2 visualises the loss function L and the learning dynamics Eq. (3).

<!-- formula-not-decoded -->

Lemma 2.1. All critical points of the loss function Eq. (6) can be written as p ∗ = 1 | S | ∑ j ∈ S e j for some S ⊆ [ d ] . Every critical point with | S | ≥ 2 is a saddle point. The local minima of the loss function L from Eq. (6) are the standard basis vectors { e 1 , . . . , e d } . Furthermore, every local minimum of L is also a global minimum.

## 2.1 Linear convergence of the learning rule

We state the convergence guarantee for the learning rule Eq. (3). Renaming the indices, we can assume that p 1 (0) is the largest initial probability. Provided that p 1 (0) is strictly larger than each other component of p (0) , the following theorem shows linear convergence of the first component to 1 on an event Θ in expectation. The probability of Θ can be chosen arbitrarily close to 1 by reducing the learning rate α .

Theorem 2.2. Given ε ∈ (0 , 1) , assume

Then there exists an event Θ with probability ≥ 1 -ε/ 2 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, given δ &gt; 0 ,

<!-- formula-not-decoded -->

Remark 2.3 . If all weights are equal at the starting point of the learning algorithm, the assumption p 1 (0) -max i =2 ,...,d p i (0) &gt; 0 , is equivalent to requiring λ 1 (0) -max i =2 ,...,d λ i (0) &gt; 0 . In this case, the convergence of p ( k ) to e 1 corresponds to the network performing a winner-take-all mechanism [12, 49, 31, 40, 25]. The competitive selection of input neurons in Hebbian STDP has been observed by [39, 38, 15], among others. Our results extend existing findings by going beyond ensemble averages and also provide a rate of convergence for the random dynamics.

The convergence on an event of high probability is in line with other recent results on noisy/stochastic gradient descent for non-convex loss functions, see e.g. [27, 46] or [11, Theorem 2.5]. Contrary to these results, we can choose a constant learning rate and obtain linear convergence. To illustrate the reason for this, we give a brief overview of the proof of Theorem 2.2. The full proof can be found in Section A.1.

1. We restrict the analysis to the event Θ on which

<!-- formula-not-decoded -->

holds for all iterates k . On this event, the derivative of the first component can be bounded from below by

<!-- formula-not-decoded -->

2. As described in Eq. (3), we apply a Taylor approximation to the original dynamics. We bound the error term for the i th component p i ( k +1) by the order α 2 p i ( k )(1 -p i ( k )) . The approximation error is dominated by the gradient update on Θ , if the learning rate is small enough (see Eq. (8).
3. Similarly as in Eq. (5), we restate the learning increments of the dynamics as the sum of the true gradient and a centred noise vector ξ ( k ) . By Eq. (8), this decomposition yields linear convergence of p 1 ( k ) → 1 on Θ .
4. To find a lower bound for the probability of the chosen event Θ , we employ a similar strategy as Mertikopoulos et al. [27]. Through the representation Eq. (5), we can show that Θ occurs, as soon as M ( k ) := α ∑ k i =1 ξ ( i ) is uniformly bounded by some sufficiently small constant. As ( M ( k )) k ∈ N is a martingale, the probability of the latter event can be controlled through Doob's submartingale inequality (see Eq. (27)).
5. To apply Doob's submartingale inequality, we bound the second moment of M ( k ) . Since the variance of the components of ξ ( k ) is also dominated by 1 -p 1 ( k ) , we achieve a bound of the order α 2 ∑ ∞ i =1 (1 -p 1 ( i )) /x31 Θ . This series is summable as we have linear convergence to 0 , which allows us to choose a constant learning rate α .

## 2.2 Associated gradient flow

In this subsection, we consider the associated gradient flow of probabilities p ( t ) as a vector-valued function, which solves the ODE

<!-- formula-not-decoded -->

and is initialized by the probability vector p (0) . By definition, d d t ∑ d i =1 p i ( t ) = 0 , such that ∑ d i =1 p i ( t ) = ∑ d i =1 p i (0) = 1 . Since the updating rule is multiplicative, the gradient flow produces for all t ≥ 0 a probability vector. The gradient flow Eq. (9) also occurs as a specific replicator equation in evolutionary game theory, see Hofbauer and Sigmund [17, Chapter 7]. Elementary properties and an explicit solution for the gradient flow with d = 2 are derived in Section 2.2. Although the loss function L ( p ) in Eq. (6) does not satisfy a global Polyak-Łojasiewicz condition, in particular it is not globally convex, we can deduce the following convergence for the ODE Eq. (9).

Theorem 2.4. Assume for some ∆ &gt; 0 . Then

<!-- formula-not-decoded -->

that is linear convergence of p ( t ) → e 1 as t →∞ .

<!-- formula-not-decoded -->

## 2.3 Biological plausibility of the proposed learning rule

We study a biological neural network consisting of d input (or presynaptic) neurons, which are connected to one output (or postsynaptic) neuron. For the subsequent argument, we assume that the spike times of the d input neurons are given by the corresponding jump times of d independent Poisson processes ( X (1) t ) t ≥ 0 , . . . , ( X ( d ) t ) t ≥ 0 with respective intensities λ 1 , . . . , λ d . All neurons are excitatory and each connection between an input neuron j ∈ [ d ] and the output neuron has a time varying and nonnegative synaptic strength parameter, which we denote by w j ( t ) ≥ 0 .

An idealized model is that a spike of the j th input neuron at time τ causes an exponentially decaying contribution to the postsynaptic membrane potential of the form t ↦→ w j ( τ ) Ce -c ( t -τ ) /x31 t ≥ τ . We set the parameters c, C &gt; 0 to one, as this can always be achieved by a time change t ↦→ tc and a change of units of the voltage in the membrane potential.

If T j denotes the spike times of neuron j ∈ [ d ] , the postsynaptic membrane potential ( Y t ) t ≥ 0 is given by Y 0 = 0

Figure 3: Considered biological neural network with spike trains and membrane potential Y t of the postsynaptic neuron.

<!-- image -->

and Y t = ∑ d j =1 ∑ τ ∈T j ,τ ≤ t w j ( τ ) e -( t -τ ) for all t ≥ 0 until Y t ≥ S , where S &gt; 0 is a given threshold value. Once the threshold S is surpassed, the postsynaptic neuron emits a spike and its membrane potential is reset to its rest value, which we assume to be 0 . Afterwards, the incoming spikes will contribute to rebuilding the postsynaptic membrane potential. If t 0 := 0 &lt; t 1 &lt; t 2 &lt; . . . denote the postsynaptic spike times, the membrane potential at arbitrary time is therefore given by

<!-- formula-not-decoded -->

We consider the following pair-based spike-timing-dependent plasticity (STDP) rule ([14, Section 19.2.2]): A spike of the j th presynaptic neuron at time τ causes the weight parameter function t ↦→ w j ( t ) to decrease at τ by αe -( τ -t -) , where t -is the last postsynaptic spike time before τ and to increase at any postsynaptic spike time t k by α ∑ τ ∈T j ∩ ( t k , t k +1 ] e -( τ -t k ) , with α &gt; 0 the learning rate. As common in the literature, spike times that occurred before t k only have a minor influence and are neglected in the updating of the weights after t k . The term ∑ τ ∈T j ∩ ( t k , t k +1 ] e -( τ -t k ) is then the trace ([14, Equation (19.12)]) of the j th presynaptic neuron at time t k +1 .

For mathematical convenience, we will assume that all weight-updates in ( t k , t k +1 ] are delayed to the postsynaptic spike times t k in the sense that the learning rule becomes

The postsynaptic spike times t k are the moments at which the postsynaptic membrane potential Y t reaches the threshold S . They depend on the presynaptic spike times, however, the exact dependence is hard to characterise in the assumed model. For mathematical tractability, we will instead work with an adjusted rule to select the postsynaptic spike times t 1 , t 2 , . . . Since Y t only increases at the presynaptic spike times, t k +1 has to happen at a presynaptic spike time. Denote by τ j 1 , τ j 2 , . . . the spike times of the j -th presynaptic neuron after the previous postsynaptic spike time t k in increasing order. The distribution of t k +1 | t k is completely determined by the probabilities

<!-- formula-not-decoded -->

P ( t k +1 = τ jℓ ) = P ( t k +1 = τ jℓ ∣ ∣ t k +1 ∈ ( τ jm ) m ≥ 1 ) P ( t k +1 ∈ ( τ jm ) m ≥ 1 ) , j = 1 , . . . , d, ℓ = 1 , . . . Based on probabilistic arguments related to the underlying Poisson processes that we outline in Section A.3, we replace the probabilities P ( t k +1 ∈ ( τ jm ) m ≥ 1 ) by the probabilities

<!-- formula-not-decoded -->

Working with Eq. (12) instead of P ( t k +1 ∈ ( τ jm ) m ≥ 1 ) results in an approximation of the distribution of t k +1 . Lemma A.7 describes a setting, where Eq. (12) is exact. If all weights are much larger than the threshold S , every presynaptic spike causes a postsynaptic spike. The proof of Lemma A.7 can be adapted to this case to show that the probability that the j th neuron emits the first spike is λ j / ∑ d ℓ =1 λ ℓ . Since Hebbian learning is intrinsically unstable, we argue that the proposed approximation describes the dynamic at the beginning of the learning process. This view is corroborated by experimental results, see point (vi) of Morrison et al. [28, Section 2.1].

Compared to the original definition of t k +1 , the proposed sampling scheme has the advantage that the presynaptic spike times, which were not selected as postsynaptic spike time, add centred noise to the updates. More precisely, one can show that by the construction of t k and the properties of the underlying Poisson processes, the conditional distribution τ |{ τ ∈ ( t k , t k +1 ) } is uniformly distributed on ( t k , t k +1 ) . By the symmetry relation e -( b -u ) -e -( u -a ) = -( e -( b -v ) -e -( v -a ) ) ∈ [ -1 , 1] , which holds for all real numbers a ≤ u ≤ b with v = b + a -u ∈ [ a, b ] , this implies that conditionally on τ ∈ ( t k , t k +1 ) , the random variable e -( t k +1 -τ ) -e -( τ -t k ) is centred and supported on [ -1 , 1] . The update rule Eq. (11) then becomes

<!-- formula-not-decoded -->

with centred random variables Z ( τ, j ) satisfying | Z ( τ, j ) | ≤ 1 . Assuming that the postsynaptic firing rate is slow compared to the learning dynamic, we discard the term e t k -t k +1 glyph[lessmuch] 1 . Since j ∗ ( k +1) follows a multinomial distribution with parameters λ j w j ( t k ) / ( ∑ d ℓ =1 λ ℓ w ℓ ( t k )) , the term /x31 { j = j ∗ ( k +1) } corresponds to the j th component of B ( k ) in Eq. (1). This motivates the learning rule Eq. (1). Additional details on the derivation are given in Subsection A.3 of the supplementary material.

## 3 A mirror descent perspective

In this section, we rewrite the gradient flow Eq. (9) as natural gradient descent on the probability simplex and relate the discrete-time learning rule Eq. (3) for the probabilities p to noisy mirror gradient descent.

Recall from Eq. (5) that the learning rule Eq. (3) can be interpreted as noisy gradient descent with respect to the loss function L from Eq. (6) in the Euclidean geometry. As we consider a flow on probability vectors, a different perspective is to use the natural geometry of the probability simplex. To this end, we consider the interior of the probability simplex M := int( P ) as a Riemannian manifold with tangent space T p M = { x ∈ R : 1 ⊤ x = 0 } for every p ∈ M . A natural metric on M is given by the Fisher information metric / Shahshahani metric [4, 17], which is induced by the metric tensor d p : T p M×T p M→ R , ( u , v ) ↦→ u ⊤ diag( p ) -1 v at p ∈ M . Here, diag( p ) ∈ R d × d is the diagonal matrix with diagonal entries given by p . We refer to Figure 1 of Mertikopoulos and Sandholm [26] for an illustration of unit balls in this metric. The (Riemannian) gradient of the loss function ˜ L ( p ) = -‖ p ‖ 2 / 2 with respect to d p is given by ∇ d p ˜ L ( p ) = diag( p ) ∇ ˜ L ( p ) ∈ T p M , where we denote by ∇ ˜ L the Euclidean gradient of ˜ L . The Riemannian gradient flow is called natural gradient flow in information geometry [3] and Shahshahani gradient flow in evolutionary game theory [18]. When transforming the Euclidean gradient flow for L to a Riemannian gradient flow on the probability simplex, the part + ‖ p ‖ 2 1 is orthogonal to T p M . Consequently, it does not contribute a direction on the probability simplex. Consequently, the Riemannian gradient flow of ˜ L and the Euclidean gradient flow of L coincide.

<!-- formula-not-decoded -->

The mirror descent algorithm [29] prescribes the discrete-time optimisation algorithm where f : M→ R is the function to be minimised and Φ: M×M→ R + is a suitable proximity function. Euclidean gradient descent is recovered by the choice Φ( p , p ( k )) = ‖ p -p ( k ) ‖ 2 . It

is well-known that the natural gradient flow is the continuous-time analogue of the exponentiated gradient descent or entropic mirror descent , where Φ( p , p ( k )) = KL( p ‖ p ( k )) is chosen as the Kullback-Leibler divergence between p and p ( k ) [2, 47, 33]. Consequently, the gradient flow Eq. (9) can also be viewed as continuous-time version of entropic mirror descent with respect to f = ˜ L . This connection transfers to the discrete-time and noisy updating rule Eq. (3). An alternative approach for connecting our proposed discrete-time learning rule Eq. (3) to entropic mirror descent is included in Subsection A.4 of the supplementary material.

## 4 Multiple weight vectors

The learning rule Eq. (1) aligns the output neuron with the input neuron of the highest intensity, but no information about the remaining input neurons is unveiled. As a proof-of-concept, we generalise the learning algorithm Eq. (1) to estimate the order of the intensities λ 1 , . . . , λ d . To this end, we consider d different output neurons, which are connected to the d input neurons via the weight vectors w 1 , . . . , w d ∈ R d . The weights at time k are combined into the matrix

<!-- formula-not-decoded -->

and the corresponding probabilities p 1 , . . . , p d are combined into the matrix P ( k ) = [ p 1 ( k ) · · · p d ( k )] ∈ R d . By reordering the neurons, we can achieve λ 1 ≤ λ 2 ≤ · · · ≤ λ d . If the intensities are strictly ordered, our goal is the alignment of the j th output neuron with the j th input neuron, which amounts to the convergence of P ( k ) to the identity matrix I ∈ R d × d as time

Figure 4: Neural network with d input/output neurons.

<!-- image -->

increases. If multiple intensities are equal, convergence is up to permutations within the group of equal intensities. We propose Algorithm 1, which constitutes an STDP rule as lines 3 - 4 can be implemented using the spike-trains and the learning rule Eq. (11).

## Algorithm 1: Aligning multiple output neurons

```
Input: K ∈ N : number of iterations, W (0) ∈ R d × d : weight initialisation, α 1 , . . . , α d : learning rates of the output neurons. 1 for k = 0 , 1 , . . . , K -1 do 2 for j = 1 , . . . , d do 3 Receive B j ( k ) ∼ M (1 , p j ( k )) with p j ( k ) ← λ glyph[circledot] w j ( k ) / λ ⊤ w j ( k ) and Z j ( k ) ∼ Unif([ -1 , 1] d ) from spike trains; 4 Compute the base change ∆ w j ( k ) ← α j w j ( k ) glyph[circledot] ( B j ( k ) + Z j ( k )) ; 5 Update w j ( k +1) ← ∆ w j ( k ) -j -1 ∑ i =1 (∆ w j ( k )) ⊤ w i ( k ) ‖ w i ( k ) ‖ 2 w i ( k ); 6 end 7 end Output: The weight evolution W ( k ) = [ w 1 ( k ) · · · w d ( k )] , k = 0 , . . . , K and probability evolution P ( k ) = [ p 1 ( k ) · · · p d ( k )] , k = 1 , . . . , K .
```

Algorithm 1 is inspired by Sanger's rule [35] for learning d principal components in streaming PCA. The first weight vector w 1 ( k ) aligns with e 1 by Theorem 2.2 since its dynamic equals the learning rule Eq. (1). By removing the components of the change ∆ w j ( k ) in the direction of w 1 ( k ) , . . . , w j -1 ( k ) in line 5 of Algorithm 1, the weight vector w j ( k ) is forced to converge to e j , similarly to the Gram-Schmidt algorithm.

Simulations of the corresponding probability matrix P ( k ) with varying learning rates and Z drawn i.i.d. from Unif([ -1 , 1] d ) are included in Figure 5. We choose different learning rates for the d vectors satisfying α 1 &gt; · · · &gt; α d &gt; 0 . This ordering ensures fast convergence of the lower order weight vectors to the correct standard basis vector and counteracts the impact of initial misalignments of the higher order weight vectors. The simulation of Algorithm 1 shown in Figure 5 displays a decrease

Figure 5: Probability matrix P ( k ) arising from the weight dynamic W ( k ) of Algorithm 1 for dimensions n = d = 3 . The weights are initialised equally, and the intensities are given by λ = (10 , 7 . 5 , 5) ⊤ . The resulting initial probabilities are p 1 (0) = p 2 (0) = p 3 (0) = (4 / 9 , 1 / 3 , 2 / 9) ⊤ . Left: A single trajectory with learning rates 10 -3 (1 , 0 . 75 , 0 . 5) ⊤ and 4 × 10 4 iterations. The markers × and · correspond to the probabilities at k = 4 × 10 3 and k = 10 4 . Middle &amp; right: The Frobenius error ‖ P ( k ) -I ‖ 2 / 2 of 100 trajectories with learning rates 10 -3 (1 , 0 . 75 , 0 . 5) ⊤ and 10 -4 (1 , 0 . 75 , 0 . 5) ⊤ , respectively.

<!-- image -->

of the Frobenius error ‖ P ( k ) -I ‖ 2 / 2 over the iteration index k , when averaged (blue line). Nevertheless, we observe that for a single trajectory, the error can plateau around 1 and 2. Given that the probability vectors tend to converge to standard basis vectors { e 1 , . . . , e d } , a non-vanishing error is due to an incorrect ordering or duplicates. Consequently, the error ‖ P ( k ) -I ‖ 2 / 2 corresponds to the number of output neurons aligning with the incorrect input neuron, and plateaus at 1, 2 and 3 can arise. Theorem 2.2 shows that this phenomenon can be mitigated by slower learning rates, which is corroborated by decreasing the base learning rate from 10 -3 to 10 -4 in the simulation. A rigorous mathematical analysis of the Algorithm 1 is challenging due to joint updates in all read-out neurons. In Section A.5 we show that Theorem 2.2 is applicable if the learning is split into disjoint learning periods, and we derive theoretical convergence guarantees.

## 5 Extensions, discussion and limitations

Time-inhomogeneous intensities. We considered input spike trains generated from Poisson point processes with fixed intensity. It is natural to extend this to time-inhomogeneous intensities. Here we assume that the intensities of the input neurons are constant on the interval ( k, k +1] and are stored in the vector λ ( k ) . The intensities λ ( k ) and weights w ( k ) determine the spike-triggering-probabilities p ( k ) = λ ( k ) glyph[circledot] w ( k ) / λ ( k ) ⊤ w ( k ) = E [ Y ( k )] and the update formula Eq. (2) becomes

<!-- formula-not-decoded -->

˜ with ˜ p ( k ) := λ ( k +1) glyph[circledot] w ( k ) / ( λ ( k +1) ⊤ w ( k )) . A first order Taylor expansion yields

Since E [ Y ( k )] = p ( k ) , this means that p ( k +1) = ˜ p ( k ) glyph[circledot] ( 1 + α ( p ( k ) -˜ p ( k ) ⊤ p ( k ) 1 ) ) + centered noise + O ( α 2 ) . (17) Extending the gradient flow derivation to the time-inhomogeneous case, one can identify the ODE

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the logarithm is taken componentwise, as corresponding deterministic dynamic in continuous time, see Section A.6 for a derivation. The ODE can be interpreted as a replicator equation with (time-varying) fitness d d t log( λ ( t )) + p ( t ) , see e.g. Chapter 7 of [18]. An interesting scenario which lies beyond our mathematical analysis amounts to considering time-dependent mean firing rates that are piecewise constant, corresponding to the successive exposition to different input patterns [9].

Correlated inputs. Correlated inputs facilitate simultaneous spiking of different input spikes. The probability that input i and j spike at the same time is denoted by Γ ij , and naturally Γ ii := 1 , for all i ∈ [ d ] . We introduce the d × d random symmetric matrix C ( k ) = ( C i,j ( k )) i,j ∈ [ d ] with independently sampled entries

<!-- formula-not-decoded -->

C ( k ) describes the simultaneous spiking of the different inputs at the k -th post-synaptic spike, i.e. if C i,j ( k ) is 1 then inputs i and j both spike at the k -th post-synaptic spike if either i or j caused the post-synaptic spike. Compared to the original model, the random vector Z ( k ) remains the same, but the random vector B ( k ) = ( B 1 ( k ) , . . . , B d ( k )) ⊤ that encapsulates which of the presynaptic neurons caused the postsynaptic spike is replaced by S ( k ) = C ( k ) B ( k ) . S ( k ) encodes which of the inputs spike at a post-synaptic spike, and in particular it holds where Γ i, · ( k ) denotes the i -th row of Γ ( k ) . Since the only change in the dynamic of p ( k ) is replacing B ( k ) by S ( k ) in the definition of Y ( k ) , Eq. (4) still holds true, and we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which induces the following gradient flow

<!-- formula-not-decoded -->

This is again a replicator equation with fitness Γp , compare Section 7 of [18]. The associated Shahshahani-loss is given by x ↦→ -1 2 x ⊤ Γx . Thus, in the correlated model, the probabilities p follow a flow restricted to the probability simplex aimed at maximising the quadratic form associated to the matrix Γ . This property is similar to principal component analysis (PCA), as the goal there is to recover the eigenvector corresponding to the largest eigenvalue of the underlying covariance matrix. Thus, the behaviour of the correlated model can be interpreted as a form of PCA restricted to the probability simplex. Theorem A.13 in the supplementary material generalizes Theorem 2.2 to weakly correlated input neurons and is accompanied by simulations in Figure 6.

Small biological neural network. In this paper, we mathematically analyse the convergence behaviour of a small biological neural network with one layer composed of excitatory presynaptic/input neurons and multiple postsynaptic/output neurons. It is natural to generalise the setting to account for inhibitory neurons and more than one layer.

Weight explosion. The learning rule Eq. (1) for the weights w ( k ) causes them to increase without bound as the iteration index k increases. When the weights exceed the spike threshold, the model becomes biologically implausible and the derivation of the probabilities Eq. (12) is no longer valid. This unstable nature is well-known to be intrinsic to Hebbian learning algorithms and is commonly countered by soft or hard bounds, or by including mean-reverting terms to the dynamic Gerstner et al. [14, pages 497-498]. We follow a different route, namely viewing Hebbian learning as a temporal phase of limited length, which is followed by a stabilising homeostatic learning phase. This view is corroborated by experimental results, compare Point (vi) in Morrison et al. [28, Section 2.1].

Beyond Pair-based STDP rules. While pair-based learning rules such as Eq. (1) only account for the relative timing of one pre- and one postsynaptic spike time, also the voltage at the location of the synapse should be taken into account [37]. A natural generalization of our framework would be to extend the results to the model proposed in [9].

## Acknowledgments and Disclosure of Funding

All authors acknowledge support from ERC grant A2B (grant agreement no. 101124751). S. G. has been partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) CRC/TRR 388 "Rough Analysis, Stochastic Dynamics and Related Fields Project ID 516748464, Project B07. Parts of the research were carried out while the authors visited the Simons Institute in Berkeley. We thank Xiangyuan Li for pointing out some typos.

## References

- [1] L. Abbott and S. Song. Temporally asymmetric Hebbian learning, spike liming and neural response variability. In M. Kearns, S. Solla, and D. Cohn, editors, Advances in Neural Information Processing Systems , volume 11. MIT Press, 1998.
- [2] F. Alvarez, J. Bolte, and O. Brahic. Hessian Riemannian gradient flows in convex programming. SIAM Journal on Control and Optimization , 43(2):477-501, 2004.
- [3] S.-i. Amari. Natural gradient works efficiently in learning. Neural Computation , 10(2):251276, 1998.
- [4] S.-i. Amari and H. Nagaoka. Methods of Information Geometry . Translations of Mathematical Monographs. American Mathematical Society, 2007.
- [5] A. Beck and M. Teboulle. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters , 31(3):167-175, 2003.
- [6] A. Bell and L. Parra. Maximising sensitivity in a spiking network. In L. Saul, Y. Weiss, and L. Bottou, editors, Advances in Neural Information Processing Systems , volume 17. MIT Press, 2004.
- [7] S. Bohte and M. C. Mozer. Reducing spike train variability: A computational theory of spiketiming dependent plasticity. In L. Saul, Y. Weiss, and L. Bottou, editors, Advances in Neural Information Processing Systems , volume 17. MIT Press, 2004.
- [8] G. Chechik. Spike-Timing-Dependent Plasticity and Relevant Mutual Information Maximization. Neural Computation , 15(7):1481-1510, 2003.
- [9] C. Clopath, L. Büsing, E. Vasilaki, and W. Gerstner. Connectivity reflects coding: A model of voltage-based STDP with homeostasis. Nature Neuroscience , 13(3):344352, 2010.
- [10] J. Cornford, R. Pogodin, A. Ghosh, K. Sheng, B. A. Bicknell, O. Codol, B. A. Clark, G. Lajoie, and B. A. Richards. Brain-like learning with exponentiated gradients, 2024. URL http: //biorxiv.org/lookup/doi/10.1101/2024.10.25.620272 .
- [11] S. Dereich and A. Jentzen. Convergence rates for the Adam optimizer, 2024. URL https: //arxiv.org/abs/2407.21078 .
- [12] J. Feldman and D. Ballard. Connectionist models and their properties. Cognitive Science , 6 (3):205-254, 1982.
- [13] W. Gerstner and W. M. Kistler. Spiking Neuron Models: Single Neurons, Populations, Plasticity . Cambridge University Press, 2002.
- [14] W. Gerstner, W. M. Kistler, R. Naud, and L. Paninski. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition . Cambridge University Press, 2014.
- [15] M. Gilson, A. N. Burkitt, D. B. Grayden, D. A. Thomas, and J. L. Van Hemmen. Representation of input structure in synaptic weights by spike-timing-dependent plasticity. Physical Review E , 82(2):021912, 2010.
- [16] D. O. Hebb. The Organization of Behavior . Wiley, 1949.
- [17] J. Hofbauer and K. Sigmund. Evolutionary Games and Population Dynamics . Cambridge University Press, 1998.
- [18] J. Hofbauer and K. Sigmund. Evolutionary game dynamics. Bulletin of the American Mathematical Society , 40(4):479-519, 2003.
- [19] D. Huang, J. Niles-Weed, and R. Ward. Streaming k-PCA: Efficient guarantees for Ojas algorithm, beyond rank-one updates. In M. Belkin and S. Kpotufe, editors, Proceedings of Thirty Fourth Conference on Learning Theory , volume 134 of Proceedings of Machine Learning Research , pages 2463-2498. PMLR, 2021.

- [20] S. Jaffard, S. Vaiter, A. Muzy, and P. Reynaud-Bouret. Provable local learning rule by expert aggregation for a Hawkes network. In S. Dasgupta, S. Mandt, and Y. Li, editors, Proceedings of The 27th International Conference on Artificial Intelligence and Statistics , volume 238 of Proceedings of Machine Learning Research , pages 1837-1845. PMLR, 02-04 May 2024. URL https://proceedings.mlr.press/v238/jaffard24a.html .
- [21] R. Kempter, W. Gerstner, and J. L. Van Hemmen. Hebbian learning and spiking neurons. Physical Review E , 59(4):4498-4514, 1999.
- [22] R. Kempter, W. Gerstner, and J. L. V. Hemmen. Intrinsic stabilization of output rates by spikebased hebbian learning. Neural Computation , 13(12):2709-2741, 2001.
- [23] S. Kumar and P. Sarkar. Oja's algorithm for streaming sparse PCA. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems , volume 37, pages 74528-74578. Curran Associates, Inc., 2024.
- [24] T. L. Lai. Stochastic approximation: invited paper. The Annals of Statistics , 31(2):391 - 406, 2003.
- [25] N. Lynch, C. Musco, and M. Parter. Winner-Take-All computation in spiking neural networks, 2019. URL https://arxiv.org/abs/2407.21078 .
- [26] P. Mertikopoulos and W. H. Sandholm. Riemannian game dynamics. Journal of Economic Theory , 177:315-364, 2018.
- [27] P. Mertikopoulos, N. Hallak, A. Kavis, and V. Cevher. On the almost sure convergence of stochastic gradient descent in non-convex problems. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33. Curran Associates, Inc., 2020.
- [28] A. Morrison, M. Diesmann, and W. Gerstner. Phenomenological models of synaptic plasticity based on spike timing. Biological Cybernetics , 98(6):459-478, 2008.
- [29] A. S. Nemirovsky and D. B. Yudin. Problem complexity and method efficiency in optimization . Wiley-Interscience series in discrete mathematics. Wiley, 1983.
- [30] K. Neumann, C. Strub, and J. Steil. Intrinsic plasticity via natural gradient descent with application to drift compensation. Neurocomputing , 112:26-33, 2013.
- [31] M. Oster and S.-C. Liu. Spiking inputs to a winner-take-all network. In Y. Weiss, B. Schölkopf, and J. Platt, editors, Advances in Neural Information Processing Systems , volume 18. MIT Press, 2005.
- [32] J.-P. Pfister, T. Toyoizumi, D. Barber, and W. Gerstner. Optimal Spike-Timing-Dependent Plasticity for Precise Action Potential Firing in Supervised Learning. Neural Computation , 18 (6):1318-1348, 2006.
- [33] G. Raskutti and S. Mukherjee. The information geometry of mirror descent. IEEE Transactions on Information Theory , 61(3):1451-1457, 2015.
- [34] H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics , 22(3):400-407, 1951.
- [35] T. D. Sanger. Optimal unsupervised learning in a single-layer linear feedforward neural network. Neural Networks , 2(6):459-473, 1989.
- [36] B. Scellier and Y. Bengio. Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. Frontiers in Computational Neuroscience , 11, 2017.
- [37] J. Sjöström and W. Gerstner. Spike-timing dependent plasticity. Scholarpedia , 5(2):1362, 2010.
- [38] S. Song and L. Abbott. Cortical development and remapping through spike timing-dependent plasticity. Neuron , 32(2):339-350, 2001.

- [39] S. Song, K. D. Miller, and L. F. Abbott. Competitive Hebbian learning through spike-timingdependent synaptic plasticity. Nature Neuroscience , 3(9):919-926, 2000.
- [40] L. Su, C.-J. Chang, and N. Lynch. Spike-based winner-take-all computation: Fundamental limits and order-optimal circuits. Neural Computation , 31(12):2523-2561, 2019.
- [41] S. C. Surace, J.-P. Pfister, W. Gerstner, and J. Brea. On the choice of metric in gradient-based theories of brain function. PLOS Computational Biology , 16(4):e1007640, 2020.
- [42] T. Toyoizumi, J.-P. Pfister, K. Aihara, and W. Gerstner. Generalized BienenstockCooperMunro rule for spiking neurons that maximizes information transmission. Proceedings of the National Academy of Sciences , 102(14):5239-5244.
- [43] T. Toyoizumi, J.-P. Pfister, K. Aihara, and W. Gerstner. Spike-timing dependent plasticity and mutual information maximization for a spiking neuron model. In L. Saul, Y. Weiss, and L. Bottou, editors, Advances in Neural Information Processing Systems , volume 17. MIT Press, 2004.
- [44] T. Toyoizumi, J.-P. Pfister, K. Aihara, and W. Gerstner. Optimality Model of Unsupervised Spike-Timing-Dependent Plasticity: Synaptic Memory and Weight Distribution. Neural Computation , 19(3):639-671, 2007.
- [45] A. Vigneron and J. Martinet. A critical survey of STDP in spiking neural networks for pattern recognition. In 2020 International Joint Conference on Neural Networks (IJCNN) , pages 1-9, 2020.
- [46] S. Weissmann, S. Klein, W. Azizian, and L. Döring. Almost sure convergence of stochastic gradient methods under gradient domination. Transactions on Machine Learning Research , 2025.
- [47] A. Wibisono, A. C. Wilson, and M. I. Jordan. A variational perspective on accelerated methods in optimization. Proceedings of the National Academy of Sciences , 113(47), 2016.
- [48] Yang Dan and Mu-ming Poo. Spike timing-dependent plasticity of neural circuits. Neuron , 44 (1):23-30, 2004.
- [49] A. L. Yuille and N. M. Grzywacz. A Winner-Take-All mechanism based on presynaptic inhibition feedback. Neural Computation , 1(3):334-347, 1989.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Claim 1 (connection between Hebbian STDP and noisy gradient descent) is justified in Section 2. Claim 2 (alignment of postsynaptic neuron with input neuron of highest intensity) is proven in Theorem 2.2. Claim 3 (connection to mirror descent) is addressed in Section 3.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations (analysing a small network and weight explosion) of our analysis are included in Section 5.

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

Justification: The proof of the main result (Theorem 2.2) is outlined in Subsection 2.1. Its formal proof is carried out in the technical appendix. The remaining results are proven in the technical appendix.

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

Justification: The results of the paper are of theoretical nature. Figures 2 and 3 serve only as illustrations. The code used to generate the illustrations in both Figures is included in the submission and will be made publicly available.

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

Justification: The results of the paper are of theoretical nature. Figures 2 and 3 serve only as illustrations. The code used to generate the illustrations in both Figures is included in the submission and will be made publicly available.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The results of the paper are of theoretical nature. Figures 2 and 3 serve only as illustrations. The code used to generate the illustrations in both Figures is included in the submission and will be made publicly available.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The results of the paper are of theoretical nature. Figures 2 and 3 serve only as illustrations. The code used to generate the illustrations in both Figures is included in the submission and will be made publicly available.

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

Justification: The results of the paper are of theoretical nature. Figures 2 and 3 serve only as illustrations. The code used to generate the illustrations in both Figures is included in the submission and will be made publicly available.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The results of the paper are of theoretical nature and conform, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: The results of the paper are of theoretical nature and we are not aware of potential societal impacts.

Justification: [NA]

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

Justification: The results of the paper are of theoretical nature and do not use data or models that have a high-risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The results of the paper are of theoretical nature and do not use assets.

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

Justification: The results of the paper are of theoretical nature and we introduce no new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The results of the paper are of theoretical nature and no crowdsourcing experiments or research with human subjects has been conducted.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [NA]

Justification: The results of the paper are of theoretical nature and no crowdsourcing experiments or research with human subjects has been conducted.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: Our results were obtained without the use of LLM.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Technical Appendix

We first study the loss landscape

<!-- formula-not-decoded -->

Lemma 2.1 identifies the stationary points if we view the landscape as a function on R d .

Proof of Lemma 2.1. The formula Eq. (7) for the gradient ∇ L ( p ) shows that the set of critical points is given by

<!-- formula-not-decoded -->

To identify local the extrema, we compute the Hessian matrix

<!-- formula-not-decoded -->

where diag( p ) ∈ R d × d is the diagonal matrix with diagonal entries given by p . Substituting a critical point p ∗ with n ∈ [ d ] non-zero entries yields (up to permutations of rows and columns)

<!-- formula-not-decoded -->

where 1 n × n is the n × n matrix consisting of ones. The corresponding eigenvalues are where E ⊥ n is the orthogonal complement of E n in span( e 1 , . . . , e n ) . Consequently, only those critical points p ∗ ∈ Crit are local minima, which have n = 1 , i.e. p ∗ ∈ { e 1 , . . . , e d } . Since all local minima attain the same loss -1 / 12 and L ( p ) → ∞ as ‖ p ‖ → ∞ , every local minimum is also a global minimum.

<!-- formula-not-decoded -->

Remark A.1 . The eigenvalues of the Hessian of the loss function computed in Eq. (20) also imply that if n ≥ 2 , then p ∗ ∈ Crit is a saddle point in R d . Interestingly, when restricting to directions within the probability simplex, the case n = d is not a saddle point, but a maximum, since the direction ∑ d i =1 e i is orthogonal to P .

## A.1 Proofs for Subsection 2.1

In the following we will always assume that

<!-- formula-not-decoded -->

for some ∆ ∈ (0 , 1) . This is a deterministic constraint. The randomness occurs because of the noise in the updates. We assume that all random variables are defined on a filtered probability space (Ω , F , P ) , and denote by F n , n = 0 , 1 , . . . , the natural filtration of ( B ( n ) , Z ( n )) n ∈ N . By a slight abuse of notation we also introduce F -1 = {∅ , Ω } . In particular it then holds that p n is F n -1 -measurable for n = 0 , 1 , . . . . The starting point for the proof of the linear convergence of STDP is given by the following Lemma, which explicitly bounds the error term in the Taylor approximation contained in Eq. (3). Recall that | Y i ( k ) | ≤ Q , for all i ∈ [ d ] , k = 0 , 1 , . . .

<!-- formula-not-decoded -->

Lemma A.2. For i ∈ [ d ] and k = 0 , 1 , . . . define

and assume α &lt; 1 /Q . Then for any i ∈ [ d ] and k = 0 , 1 , . . . , there exists a random variable θ i ( k ) , satisfying such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By definition

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, a Taylor expansion around x = 0 gives that there exists some γ ∈ (0 , x ) , such that

<!-- formula-not-decoded -->

Hence we obtain that for some γ ∈ (0 , α ) ,

<!-- formula-not-decoded -->

Using that | Y i ( k ) | ≤ Q almost surely for all i ∈ [ d ] , k = 0 , 1 , . . . , the absolute value of the error term can be bounded as follows

̸

<!-- formula-not-decoded -->

Since p i ( k ) is F k -1 -measurable for any i ∈ [ d ] and E [ Y i ( k ) | F k -1 ] = p i ( k ) , we also obtain

<!-- formula-not-decoded -->

which concludes the proof.

Now, for a, b ∈ [ -Q,Q ] the first two derivatives of the function are given by

For ∆ given in Eq. (21), we define a sequence of benign events and due to the assumption Eq. (21) we set Ω(0) = Ω . On the above events, the gradient is bounded away from zero and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proposition A.3. If

then, on the event Ω( k ) ,

<!-- formula-not-decoded -->

Proof. By definition, we have on the event Ω( k ) ,

<!-- formula-not-decoded -->

The constraint imposed on the learning rate implies that α &lt; 1 /Q and Lemma A.2 becomes applicable. Now, combining the previous inequality with the assumption on α and applying Lemma A.2 with i = 1 , as well as Eq. (23), gives, on the event Ω( k ) ,

This concludes the proof.

<!-- formula-not-decoded -->

Having understood the dynamics of p on the favourable event Ω( k ) , we aim for a lower bound for its probability. A key step is the following Lemma, which states that Ω( k ) is fulfilled as soon as

<!-- formula-not-decoded -->

with ξ j ( ℓ ) defined in Eq. (22), exhibit a uniform concentration behaviour.

Lemma A.4. Define the sets

Then if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the following set inclusion holds for any k = 0 , 1 , . . .

<!-- formula-not-decoded -->

Using these properties, we can prove a recursive upper bound for 1 -p 1 ( k ) .

<!-- formula-not-decoded -->

Proof. Let u ∈ { 2 , . . . , d } be arbitrary. It follows by Lemma A.2, that on Ω( k ) the bound holds. Consequently, on Ω( k ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We have p u ( k ) ≤ 1 -p 1 ( k ) and thus, on Ω( k ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

invoking the constraint on the learning rate in the last step. From the assumptions on α we deduce that on the event Ω( k ) ,

<!-- formula-not-decoded -->

where we applied Eq. (24) in the third to last inequality. Because of Ω( k ) ⊆ Ω( k -1) it then follows,

<!-- formula-not-decoded -->

This gives

<!-- formula-not-decoded -->

We want to prove by induction that E ( k ) ⊆ Ω( k +1) for all k = 0 , 1 , . . . . For k = 0 , this directly follows from Eq. (26), since Ω(0) = Ω due to assumption Eq. (21). Assume the assertion holds for some k = 0 , 1 , . . . . Hence, it holds E ( k +1) ⊆ E ( k ) ⊆ Ω( k +1) , such that for any u ∈ { 2 , . . . , d } it holds on E ( k +1) by Eq. (26)

<!-- formula-not-decoded -->

which proves the assertion.

Having assembled the previous results, we are able to prove the linear convergence of STDP stated in Theorem 2.2. As Proposition A.3 already suggests the desired behaviour of p on Ω( k ) , the main

part of the proof is to show that Ω( k ) is satisfied with large probability. For that we deploy Doob's submartingale inequality, which states that for a martingale ( X n ) n ∈ N , any p ≥ 1 , and any u &gt; 0 ,

<!-- formula-not-decoded -->

This will be applied to derive lower bounds for the event E ( k ) defined in Lemma A.4, which are also lower bounds for the probability of Ω( k +1) by the same Lemma. For the reader's convenience we restate Theorem 2.2 before giving its proof.

Theorem 2.2. Given ε ∈ (0 , 1) , assume

<!-- formula-not-decoded -->

Then there exists an event Θ with probability ≥ 1 -ε/ 2 such that

Consequently, given δ &gt; 0 , it holds

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem 2.2. The recursive definition ensures that p ( k ) is F k -1 -measurable. Thus, also Ω( k ) ∈ F k -1 for any k = 0 , 1 , . . . . One can check that ( M i ( k )) k =0 , 1 ,... , defined in Eq. (25), forms a martingale for each i ∈ [ d ] . This allows us to apply Doob's submartingale inequality. To apply it with p = 2 , we deduce the following bound on the second moment,

E [ M 1 ( k ) 2 ]

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

where we used that E [ ξ 1 ( k 2 ) | F k 1 ] = 0 , for any k 2 &gt; k 1 = 0 , 1 , . . . , together with | Y 1 ( k ) | ≤ Q , the inequality ( a + b ) 2 ≤ 2( a 2 + b 2 ) for a, b ∈ R and 1 -p 1 ℓ ≤ 1 . Arguing similarly we obtain for u ∈ { 2 , . . . , d } ,

<!-- formula-not-decoded -->

Hence, applying a union bound, Doob's submartingale inequality Eq. (27) with p = 2 gives for any k = 0 , 1 , . . .

<!-- formula-not-decoded -->

Proposition A.3 gives for any k = 0 , 1 , . . . the bound

E [(1 -p 1 ( k +1)) /x31 Ω( k +1) ] ≤ E [(1 -p 1 ( k +1)) /x31 Ω( k ) ] ≤ E [( 1 -α ∆ 4 d ( 1 + ∆ 2 ( d -1) )) ( 1 -p 1 ( k ) ) /x31 Ω( k ) + αξ 1 ( k ) /x31 Ω( k ) ] = ( 1 -α ∆ 4 d ( 1 + ∆ 2 ( d -1) )) E [ ( 1 -p 1 ( k ) ) /x31 Ω( k ) ] , which implies

We set

The continuity of probability measures and Lemma A.4 then imply

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used that we can assume d ≥ 2 without loss of generality. Additionally, Eq. (28) and the elementary inequality 1 -x ≤ exp( -x ) , which is valid for any real number x, give

When d = 1 , the right hand side of this inequality is 0 . For d ≥ 2 , we can also use the bound d -1 ≥ d/ 2 . Together with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

this concludes the proof of the first statement. For the proof of the second statement, we apply Markov's inequality to obtain

Hence, if then,

<!-- formula-not-decoded -->

Lemma A.5. Given an initialization of the weights w (0) = ( w 1 (0) , . . . , w d (0)) ⊤ consider two d -dimensional intensity vectors λ = ( λ 1 , . . . , λ d ) ⊤ , ˜ λ = ( ˜ λ 1 , . . . , ˜ λ d ) ⊤ with positive entries. Assume that λ 1 w 1 (0) &gt; max i =2 ,...,d λ i w i (0) and ˜ λ d w d (0) &gt; max i =1 ,...,d -1 ˜ λ i w i (0) . Assume we run the learning rule Eq. (3) with intensity λ until time K ∗ and then, for k &gt; K ∗ , change the intensity to ˜ λ and run the learning dynamic until time k →∞ . If K ∗ is small, in particular, if K ∗ = 0 , the above convergence result can be applied to show that the dynamic converges to the corner e d . However, for any ε ∈ (0 , 1) and all sufficiently large K ∗ (depending on ε ), the dynamics will converge to e 1 with probability 1 -2 ε.

This result shows that the dynamic can be primed at the beginning to end up in one regime. Despite the noise and the infinite amount of data, the dynamic is unable to escape this domain of attraction. From the proof, one can derive quantitative bounds for K ∗ .

Proof. Given ε ∈ (0 , 1) , choose δ ∈ (0 , 1) such that

̸

Let ∆ = p 1 (0) -max i =1 p i (0) . Given ε, δ, ∆ choose

<!-- formula-not-decoded -->

̸

By Theorem 2.2, this guarantees that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the event ‖ p ( K ∗ ) -e 1 ‖ 1 &lt; δ, we have

<!-- formula-not-decoded -->

which can be combined into

̸

<!-- formula-not-decoded -->

Using the inequality Eq. (29), we obtain

̸

This means that restarting the learning rule Eq. (3) at time K ∗ with intensities ˜ λ 1 , . . . , ˜ λ d and weights w 1 ( K ∗ ) , . . . , w d ( K ∗ ) , shows that p 1 ( K ∗ ) &gt; max i =2 p i ( K ∗ ) . Applying Theorem 2.2 again shows convergence to e 1 with probability 1 -2 ε.

## A.2 Proofs for Subsection 2.2

This section contains additional material on the gradient flow Eq. (9), as well as the proofs for Lemma A.6 and Theorem 2.4.

For d = 2 , the gradient flow admits an explicit solution. In this case, p ( t ) = ( p 1 ( t ) , p 2 ( t )) ⊤ . If p (0) = (1 / 2 , 1 / 2) ⊤ , then this is a stationary solution and p ( t ) = p (0) = (1 / 2 , 1 / 2) ⊤ for all t ≥ 0 . If p 1 (0) &gt; 1 / 2 , then,

<!-- formula-not-decoded -->

If p 1 (0) &lt; 1 / 2 , then p 2 ( t ) = 1 -p 1 ( t ) &gt; 1 / 2 follows the dynamic in Eq. (30). This formula immediately implies that p 1 ( t ) converges exponentially fast to 1 .

Proof of Formula Eq. (30). Throughout the proof we set p ( t ) := p 1 ( t ) and do not use the previous notation p 1 ( t ) , p 2 ( t ) for the first and second probability. For d = 2 , the gradient flow ODE Eq. (9) becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rewriting this in the variable u ( t ) = 1 -2 p ( t ) gives the dynamic,

<!-- formula-not-decoded -->

̸

̸

This is solved by u ( t ) = -1 / √ Ce -t +1 since

Thus, p ( t ) = 1 2 (1 -u ( t )) solves Eq. (31). Finally, C is determined by the initial condition p (0) = 1 2 (1 -1 / √ C +1) .

<!-- formula-not-decoded -->

The following lemma summarises different properties of the gradient flow Eq. (9). In its statement, differentiable on [0 , 1] means differentiable on (0 , 1) and continuous on [0 , 1] .

Lemma A.6. The gradient flow Eq. (9) exhibits the following properties.

- (a) If ϕ : [0 , 1] → R is a convex and differentiable function, then t ↦→ ∑ d i =1 ϕ ( p i ( t )) is monotone increasing.

glyph[negationslash]

- (b) Let i, j ∈ [ d ] with i = j . If p i (0) &gt; p j (0) , respectively p i (0) = p j (0) , then p i ( t ) &gt; p j ( t ) , respectively p i ( t ) = p j ( t ) , for all t ≥ 0 . Moreover, if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.6 implies that the q -norm t ↦→| p ( t ) | q is monotonically increasing whenever 1 ≤ q &lt; ∞ . The result also implies that if instead ϕ is concave and differentiable, then t ↦→ ∑ d i =1 ϕ ( p i ( t )) is monotonically decreasing.

## Proof of Lemma A.6.

- (a) Since ϕ is convex, ϕ ′ is monotonically increasing. Thus, for a probability vector q = ( q 1 , . . . , q d ) , we have

<!-- formula-not-decoded -->

(If ϕ is strictly convex, then strict equality holds if and only if q is one of the stationary points described above.) Using this and the gradient flow formula

<!-- formula-not-decoded -->

proving the result.

- (b) By definition it holds for i, j ∈ [ d ]

From this we deduce that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then

which concludes the proof of the first statement since the exponential function is strictly positive. To prove the second statement, we have

<!-- formula-not-decoded -->

and similarly, for any i ∈ { 2 , . . . , d } ,

<!-- formula-not-decoded -->

Hence, whenever t ∈ [0 , ∆ / 2] , we find

<!-- formula-not-decoded -->

which also implies

<!-- formula-not-decoded -->

for all t ∈ [0 , ∆ / 2] . Therefore for any t ∈ [0 , ∆ / 2] , i ∈ [ d ] it holds

<!-- formula-not-decoded -->

which implies for any i ∈ { 2 , . . . , d } and t ∈ [0 , ∆ / 2] ,

<!-- formula-not-decoded -->

Applying this argument iteratively concludes the proof.

For the reader's convenience we restate Theorem 2.4 before giving its proof.

Theorem 2.4. Assume for some ∆ &gt; 0 . Then

<!-- formula-not-decoded -->

that is linear convergence of p ( t ) → e 1 as t →∞ .

Proof of Theorem 2.4. Arguing as in Eq. (24) and Eq. (23), Lemma A.6 Eq. (b) implies

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Grönwall's inequality entails which gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.3 Proofs for Subsection 2.3

In this subsection, we heuristically derive the expression for the probabilities

<!-- formula-not-decoded -->

in Eq. (12). To this end, we assume that the weights are small compared to the threshold S , that the weights are only updated at the postsynaptic spike times, and that ∑ d ℓ =1 λ ℓ w ℓ ( t k ) glyph[greatermuch] S . For convenience, we write w ℓ for w ℓ ( t k ) and all ℓ ∈ [ d ] . The constraint ∑ d ℓ =1 λ ℓ w ℓ glyph[greatermuch] S guarantees that after the postsynaptic spike time t k , the membrane potential Y t will again reach S and thus emit another spike at time t k +1 .

Taking the expectation of the membrane potential Y t = ∑ d j =1 ∑ τ ∈T j ∩ ( t k ,t ] w j e -( t -τ ) with respect to all except the j th spike-train, gives

̸

<!-- formula-not-decoded -->

for all t k ≤ t &lt; t k +1 .

̸

Introduce t ∗ := inf { t ≥ t k : Z t ≥ S -w j } and write t + for the first time after t ∗ where

̸

reaches the threshold S . If there are sufficiently many neurons, the probability that the j th presynaptic neuron spikes at time t ∗ is small and will be neglected. We have Z t ∗ = S -w j such that V t + = w j . Approximating 1 -e -( t + -t ∗ ) ≈ t + -t ∗ gives

<!-- formula-not-decoded -->

̸

The j th presynaptic neuron causes the next postsynaptic spike if and only if it spikes in the interval ( t ∗ , t + ) . The spike times of the j th presynaptic neuron are generated from a Poisson process with intensity λ j . Thus, if U ∼ Poisson( λ j ( t + -t ∗ )) , the probability that the j th presynaptic neuron spikes in ( t ∗ , t + ) is given by glyph[negationslash]

We can moreover approximate the denominator on the right hand side by the full sum ∑ d ℓ =1 w ℓ λ ℓ . Since the probabilities add up to one, we must have e t ∗ -t k ≈ 1 . This shows that the probability of the j th presynaptic neuron triggering the first postsynaptic spike after t k is approximately given by Eq. (33).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

LemmaA.7. Consider the setting outlined in Subsection 2.3. If at some time point t &gt; 0 , all weights are the same, then the probability that the j th neuron triggers the next postsynaptic spike after t is given by

<!-- formula-not-decoded -->

Proof. Since all weights are the same, we can denote their value by w . The j th neuron causes a postsynaptic spike if and only if it is the first one to spike after the postsynaptic membrane potential Y t has reached a level ≥ S -w . As t ∗ = inf { t ≥ t : Y t ≥ S -w } is a jump time and a stopping

time, we can restart the process at t ∗ . As the increments of Poisson processes are independent, and the time between the jumps is exponentially distributed with parameters λ j , the probability that the j th neuron causes the next presynaptic spike is given by

<!-- formula-not-decoded -->

where ( X i ) i ∈ [ d ] are independent random variables satisfying X i ∼ Exp( λ i ) . If U ∼ Exp( λ ) and V ∼ Exp( λ ′ ) are independent, then, U ∧ V ∼ Exp( λ + λ ′ ) and P ( U ≤ V ) = λ/ ( λ + λ ′ ) . Thus, min i = j X i ∼ Exp( ∑ i = j λ i ) , and

̸

̸

<!-- formula-not-decoded -->

The above lemma and the previous discussion give a motivation for the form of the probabilities in settings, where the weights are small compared to the threshold S or equal. We now give another heuristic, motivating our modelling choice. Let Y be the postsynaptic membrane potential. Then, in expectation Y grows linearly with slope λ ⊤ w . Furthermore, input i causes a postsynaptic spike if, and only if, it spikes at a time at which Y ≥ S -w i , where S &gt; 0 is the threshold level. As Y 's growth is approximately linear, the amount of time in which Y ≥ S -w i , holds is approximately equal to w i / λ ⊤ w . Now the probability that input i jumps in an interval of length w i / λ ⊤ w is given by 1 -exp( -λ i w i / λ ⊤ w ) ≈ λ i w i / λ ⊤ w , which is exactly our modelling choice in the independent setting.

## A.4 On the connection to entropic mirror descent

An alternative approach to connecting our proposed learning rule Eq. (3) for the probabilities p and the entropic mirror descent in discrete-time is as follows. The entropic mirror descent step Eq. (14) with Kullback-Leibler divergence and potential f can be solved explicitly and yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

see Section 5 of Beck and Teboulle [5] for details. With f ( p ) = ˜ L ( p ) = ‖ p ‖ 2 / 2 and the first order approximation exp( x ) ≈ 1 + x for small x we deduce for any i = 1 , . . . , d and k = 0 , 1 , . . . . As our proposed learning rule Eq. (3) is a noisy version of the last line, it is naturally connected to noisy entropic gradient descent.

## A.5 Theoretical results for the alignment of multiple read-out neurons

For a weight vector w ∈ R d with nonnegative entries let i ∗ := argmax i =1 ,...,d w ⊤ e i = argmax i =1 ,...,d w i and define the cosine-projection

<!-- formula-not-decoded -->

Assume that λ 1 &gt; · · · &gt; λ d and consider first learning the first weight vector w 1 using the learning rule in Eq. (1) while the remaining weight vectors w 2 , . . . , w d are fixed. Theorem 2.2 implies that after K iterations, we have w 1 ≈ e 1 . By projecting onto w ∗ 1 := P w 1 ( K ) and setting w 2 (0) ← w 2 (0) -w ∗ 1 w 2 (0) ⊤ w ∗ 1 / ‖ w ∗ 1 ‖ , we ensure that [ p 2 (0)] 2 = max i =1 ,...,d [ p 2 (0)] i and Theorem 2.2 applies and yields p 2 ( K ) ≈ e 2 . Proceeding successively, we arrive at Algorithm 2.

̸

Theorem A.8. Consider Algorithm 2. Assume that λ 1 &gt; · · · &gt; λ d and the minimal gap ∆ = min i =1 ,...,d -1 ([ p i (0)] i -max j&gt;i [ p i (0)] j ) &gt; 0 is positive. Then we have

<!-- formula-not-decoded -->

glyph[negationslash]

as K →∞ . More precisely, let δ &lt; κ/ (1 + κ ) for κ = min i =1 ,...,d λ i / max i =1 ,...,d λ i and ε &gt; 0 . Then

<!-- formula-not-decoded -->

## Algorithm 2: Sequential alignment of multiple output neurons

```
Input: K ∈ N : number of iterations for each learning period, W (0) ∈ R d × d : weight initialisation. 1 for j = 1 , . . . , d do 2 if j ≥ 2 then 3 w j (0) ← w j (0) -∑ j -1 i =1 w j (0) ⊤ w ∗ i ∥ w ∗ i ∥ 2 w ∗ i ; 4 end 5 for k = 0 , 1 , . . . , d do 6 Receive B j ( k ) ∼ M (1 , p j ( k )) with p j ( k ) ← λ glyph[circledot] w j ( k ) / λ ⊤ w j ( k ) and Z j ( k ) ∼ Unif([ -1 , 1] d ) from spike trains; 7 Update w j ( k +1) ← α w j ( k ) glyph[circledot] ( B j ( k ) + Z j ( k )); Set w ∗ j := P w j ( K ) to obtain p ∗ j = λ ⊤ w ∗ j . 8 end 9 end Output: The weight evolution W ( k ) = [ w 1 ( k ) · · · w d ( k )] , k = 0 , . . . , K , probability evolution P ( k ) = [ p 1 ( k ) · · · p d ( k )] , k = 1 , . . . , K and projections P ∗ = [ p ∗ 1 · · · p ∗ d ] .
```

Before proving Theorem A.8, we start with an auxiliary result on the order of the weights when the probability vector is close to a standard unit vector.

Lemma A.9. Assume that 0 &lt; λ min = min i =1 ,...,d λ i ≤ λ max = max i =1 ,...,d λ i &lt; ∞ and let κ = λ min /λ max . Consider a weight vector w ∈ R d with corresponding probability vector p = w glyph[circledot] λ / w ⊤ λ and let 0 &lt; δ &lt; 1 . Then the condition 1 -p 1 &lt; δ implies that max i =2 ,...,d w i ≤ w 1 κ -1 δ/ (1 -δ ) .

Proof. Since 1 -p 1 ≤ δ we know that ∑ d i =2 λ i w i ≤ δ 1 -δ λ 1 w 1 . By bounding the λ i from above and below we find such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem A.8. By Theorem 2.2 we know that P ( ‖ p 1 ( K ) -e 1 ‖ 1 ≤ δ ) ≥ 1 -ε for our choice of K . Note that the projection picks out the direction argmax i =1 ,...,d w 1 ( K ) ⊤ e i = argmax i =1 ,...,d [ w 1 ( K )] i . Consequently, Lemma A.9 and δ &lt; κ/ (1 -κ ) imply that w 1 ( K ) projects in the direction of e 1 with probability at least 1 -ε . We find that P ( p ∗ 1 = e 1 ) ≤ 1 -ε . By the adjustment of the weight vector w 2 (0) we remove its first component, such that [ p 2 (0)] 1 = 0 and [ p 2 (0)] 2 = max j =1 ,...,d [ p 2 (0)] i . By the assumption on [ p 2 (0)] 2 -max j =3 ,...,d [ p 2 (0)] j ≥ ∆ &gt; 0 we find that P ( ‖ p 2 ( K ) -e 2 ‖ 1 ≤ δ ) ≥ 1 -ε . By iteration and independence of the training windows we conclude the proof.

glyph[negationslash]

## A.6 Additional material for the extension to time-inhomogeneous intensities

Derivation of Eq. (18): Eq. (17) yields the deterministic update scheme

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜ ˜ Let T &gt; 0 be the time horizon and assume that the time-inhomogeneous intensities λ α change on the same scale as the weights and are given by λ α ( t ) = λ ( tα ) , t ∈ [0 , T/α ] , where λ ∈ C 2 b ([0 , T ]) is a universal, twice differentiable function with bounded derivatives and λ ( t ) ≥ λ min &gt; 0 , t ∈ [0 , T ] , componentwise. For vectors w , λ , λ ′ that are of the same length, and ∆ := λ ′ -λ , we have

Applying this with λ ′ = λ α ( k +1) , λ = λ α ( k ) , w = w ( k ) , yields

<!-- formula-not-decoded -->

Our assumptions on λ α imply that ˜ p ( k ) -p ( k ) ≲ ‖ ∆ λ α ( k ) ‖ ≲ α for all k = 0 , 1 , . . . , glyph[floorleft] T/α glyph[floorright] . This gives ˜ p ( k ) glyph[circledot] p ( k ) -˜ p ( k ) ˜ p ( k ) ⊤ p ( k ) = p ( k ) glyph[circledot] p ( k ) -p ( k ) p ( k ) ⊤ p ( k ) + O ( α ) . In combination with Eq. (35) we find

<!-- formula-not-decoded -->

Sending α → 0 and using that ∆ λ α ( t/α ) /α → d d t λ ( t ) as α → 0 we recognize Eq. (36) as an Euler-type scheme for the ODE

<!-- formula-not-decoded -->

where the logarithm is taken componentwise.

## A.7 Analysis of the correlated model

glyph[negationslash]

In the following we always assume that the off-diagonal entries of Γ are strictly smaller than 1 , that is, Γ i,j &lt; 1 for all i = j = 1 , . . . , d. This assumption is reasonable since a perfect correlation between input i and input j corresponds to one single input neuron with Poisson process intensity λ i + λ j . Under this assumption it is easy to see that the quadratic form associated to Γ is maximised on the probability simplex by the basis vectors e 1 , . . . , e d . Thus, the natural question to investigate is the same as in the original model: To which basis vector does the model converge? In the following section we investigate this question and show results for the weakly dependent case.

Figure 6: Correlated inputs with

<!-- image -->

<!-- formula-not-decoded -->

Contour plot of the Shahshahani loss function L ( p ) = -1 2 p ⊤ Γp on the probability simplex P for d = 3 with different overlays. Left: Three sample trajectories of Eq. (3) with different initial configurations p (0) . Middle: Stream plot of the gradient field given by Eq. (7). Right: 100 sample trajectories of Eq. (3) with p (0) = (0 . 3 , 0 . 3 , 0 . 4) ⊤ . All trajectories are simulated with 2000 iteration steps, learning rate α = 0 . 01 and Z ( k ) ∼ Unif([ -1 , 1] d ) .

The analysis of the correlated version of the model follows the same steps as in the independent case. For this we assume that all random variables are defined on a filtered probability space (Ω , F , P ) and denote by F n , n = 1 , 2 , . . . the natural filtration of ( B ( n ) , Z ( n ) , C ( n )) n ∈ N , and set F -1 = {∅ , Ω } . The proof follows exactly the same steps as the proof of Theorem 2.2. We start with the following result, which bounds the error of the Taylor approximation of the random dynamics.

Lemma A.10. For i ∈ [ d ] and k = 0 , 1 , . . . define

<!-- formula-not-decoded -->

and assume α &lt; 1 /Q . Then for any i ∈ [ d ] and k = 0 , 1 , . . . , there exists a random variable θ i ( k ) , satisfying such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. As the dynamic of p still follows Eq. (4) as in the uncorrelated model, we can argue as in the proof of Lemma A.2 to obtain that for some γ ∈ (0 , 1)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, since | Y i ( k ) | ≤ Q we obtain the same bound on the error term as in the proof of Lemma A.2

Since the p i ( k ) are F k -1 -measurable and C ( k ) is independent of F k -1 , we finally obtain E [ Y i ( k ) |F k -1 ] = Γ i, · p ( k ) .

In the following we always assume for ∆ p , ∆ Γ &gt; 0

<!-- formula-not-decoded -->

These assumptions are for example fulfilled in the following case

<!-- formula-not-decoded -->

since we can choose ∆ p = 0 . 7 , ∆ Γ = 0 . 64 , ν = 0 . 1 , and hence c ⋆ = 8 ∗ 10 -4 . In the independent case, which corresponds to ν = 0 , the assumptions given in Eq. (38) reduce to the original assumption given in Eq. (21). For k = 1 , 2 , . . . , we define a sequence of benign events

<!-- formula-not-decoded -->

## Proposition A.11. If

then, on the event Ω( k ) ,

<!-- formula-not-decoded -->

Proof. By definition, we have on the event Ω( k ) ,

<!-- formula-not-decoded -->

Due to assumption Eq. (38), Ω(0) = Ω . Additionally, since the assumptions in Eq. (38) imply the gradient to be bounded away from 0 , we can prove the following recursive upper bound for 1 -p 1 ( k ) .

<!-- formula-not-decoded -->

̸

The assumption on α implies α &lt; 1 /Q and thus Lemma A.10 becomes applicable. Now applying Lemma A.10 with i = 1 , gives on the event Ω( k ) ,

<!-- formula-not-decoded -->

This concludes the proof.

<!-- formula-not-decoded -->

As in the uncorrelated setting our goal is now to derive a lower bound for the probability of the favourable event Ω( k ) . For this we again follow the same strategy and rely on uniform concentration inequalities for martingales. In order apply those, we require the following lemma, which states that Ω( k ) is fulfilled as soon as

<!-- formula-not-decoded -->

with ξ j ( ℓ ) defined in Eq. (22), exhibits a uniform concentration behaviour. In the following we denote by ‖ Γ ‖ ∞ the row-sum norm of Γ , i.e. ‖ Γ ‖ ∞ = max i ∈ [ d ] ∑ d j =1 Γ i,j . Lemma A.12. Define the sets

<!-- formula-not-decoded -->

Then if

<!-- formula-not-decoded -->

the following set inclusion holds for any k = 0 , 1 , . . .

<!-- formula-not-decoded -->

Proof. Let u ∈ { 2 , . . . , d } be arbitrary. By Lemma A.10, on the event Ω( k ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since p u ( k ) ≤ 1 -p 1 ( k ) we also obtain on Ω( k ) ,

The assumptions on α then imply that on the event Ω( k ) ,

<!-- formula-not-decoded -->

Because of Ω( k ) ⊆ Ω( k -1) it then follows,

<!-- formula-not-decoded -->

This gives

<!-- formula-not-decoded -->

Now again let u ∈ { 2 , . . . , d } be given. Then it holds, on Ω( k )

̸

<!-- formula-not-decoded -->

Hence, arguing as in the derivation of Eq. (41) gives

With the above results we can now begin proving that E ( k ) ⊆ Ω( k +1) for all k = 0 , 1 , . . . . We do this by induction. For k = 0 , this directly follows from Eq. (41) and Eq. (42), since Ω(0) = Ω

<!-- formula-not-decoded -->

by assumption. Now, assume the assertion holds for some k = 0 , 1 , . . . . This implies E ( k +1) ⊆ E ( k ) ⊆ Ω( k +1) , such that for any u ∈ { 2 , . . . , d } it holds on E ( k +1) by Eq. (41)

<!-- formula-not-decoded -->

and additionally by Eq. (42) it holds on E ( k +1)

<!-- formula-not-decoded -->

which proves the assertion.

With the above results we are now able to prove and state the main theorem for the correlated case.

Theorem A.13. Given ε ∈ (0 , 1) , assume and

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then there exists an event Θ with probability ≥ 1 -ε/ 2 such that

Consequently, given δ &gt; 0 , it holds

<!-- formula-not-decoded -->

Before giving the proof of the above theorem, we want to remark that Theorem A.13 exactly recovers the result of Theorem 2.2 in the independent case. Indeed, in the independent case Γ is equal to the identity matrix and thus ∆ p = ∆ Γ , ν = 0 and c ⋆ = ∆ 2 p / 4 hold true, which gives the result of Theorem 2.2.

Proof of Theorem A.13. As in the uncorrelated case, the recursive definition ensures that p ( k ) is F k -1 -measurable. Thus, also Ω( k ) ∈ F k -1 for any k = 0 , 1 , . . . . Then ( M i ( k )) k =0 , 1 ,... , defined in Eq. (40), is a martingale for each i ∈ [ d ] . This allows us to apply Doob's submartingale inequality. For this, we deduce the following bound on the second moment,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we argued similarly as for the independent case. Furthermore, we obtain for u ∈ { 2 , . . . , d } ,

<!-- formula-not-decoded -->

Hence, applying a union bound, Doob's submartingale inequality Eq. (27) with p = 2 gives for any k = 0 , 1 , . . .

<!-- formula-not-decoded -->

Proposition A.11 gives for any k = 0 , 1 , . . . the bound

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We set

The continuity of probability measures and Lemma A.12 then imply

<!-- formula-not-decoded -->

Hence, if then,

<!-- formula-not-decoded -->

where we used that we can assume d ≥ 2 without loss of generality. Additionally, Eq. (43) and the elementary inequality 1 -x ≤ exp( -x ) , which is valid for any real number x, give

When d = 1 , the right hand side of this inequality is 0 . For d ≥ 2 , we can also use the bound d -1 ≥ d/ 2 . Together with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

this concludes the proof of the first statement. For the proof of the second statement, we apply Markov's inequality to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->